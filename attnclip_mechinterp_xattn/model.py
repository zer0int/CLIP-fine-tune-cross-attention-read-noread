"""OpenAI-style CLIP with explicit attention tensors and PIECES/x-attention modes.

The vision and text transformers use separate ``q_proj``, ``k_proj`` and
``v_proj`` modules.  Vision helpers expose attention patterns, Q/K/V tensors,
per-head weighted-value outputs ``z``, residual streams, activation patching,
QK/OV patching, head scaling, source-column suppression, and gradient-retained
``z`` nodes.  Full PIECES checkpoints additionally expose ``forward_modes`` and
the trained read/content/router details.

Trace usage::

    trace = model.encode_image(
        image_tensor,
        return_trace=True,
        register_threshold=70.0,
        max_registers=4,
        min_registers=1,
        cls_mask_includes_self=True,
        return_tokens=True,
    )

``register_threshold`` is applied to the L2 norm of each spatial token's final
visual residual (after the last block and before ``ln_post``).  An *implicit
register* is a spatial patch token selected by that high-norm criterion; unlike
``READ_NULL``, it is not a dedicated learned token.  ``register_mask`` has shape
``[batch, spatial_patches]`` and is ``True`` at selected patch positions.  It
excludes CLS and an explicit ``READ_NULL`` token.  Above-threshold positions are
limited to the largest ``max_registers`` norms; if fewer than
``min_registers`` remain, the largest norms are selected as a fallback.  Pass
``max_registers=None`` to disable the upper cap or ``min_registers=0`` to
disable the fallback.

Trace returns:

* ``image_embedding_full``: unmasked visual-backbone CLS embedding.
* ``image_embedding_patch_only``: final-block CLS attention recomputed with
  implicit-register source columns excluded.
* ``image_embedding_reg_only``: final-block CLS attention recomputed with only
  implicit-register source columns retained (plus CLS when requested).
* ``register_mask`` and ``patch_token_norms``: spatial mask and the L2 norms
  from which it was constructed.
* ``last_attn_probs`` and ``last_attn_logits``: final-block post-softmax
  probabilities and pre-softmax logits, ``[B, H, T, S]``.
* ``last_q``, ``last_k``, ``last_v`` and ``last_z``: final-block head-split
  tensors; Q/K/V are ``[B, H, T_or_S, head_dim]`` and z is ``[B, H, T, head_dim]``.
* ``last_xin``: final-block normalized attention input, ``[B, T, width]``.
* ``cls_skip_norm`` and ``cls_attn_norm``: L2 magnitudes of the final-block CLS
  residual and attention branches before the MLP.
* ``tokens_pre_ln_post_full`` when ``return_tokens=True``: final visual tokens,
  ``[B, T, width]``.

The patch-only and register-only paths change only the final block's CLS query
row.  Other query rows and all earlier blocks are unchanged.  On PIECES models,
these trace embeddings describe the visual backbone; use ``forward_modes`` for
read/content/router scoring and its ``return_details=True`` dictionary.

Additional image interfaces:

* ``encode_image_cached(image)`` returns the embedding and, per block,
  ``resid_pre`` plus head-split ``z``.
* ``encode_image_patched(image, spec)`` injects cached ``z`` using
  ``{block: {"inject_z": tensor, "inject_heads": [heads]}}``.
* ``encode_image_atp(image)`` retains each block's differentiable z node at
  ``block.attn._last_z_grad_node``.
* ``encode_image_inject_capture(image, spec, blocks)`` combines z injection
  with selective downstream z capture.
* ``encode_image_cached_qkov(image, blocks, cache_device=None)`` returns
  ``probs`` and ``v`` per requested block; ``encode_image_qkov_patched`` accepts
  ``inject_probs``, ``inject_v`` and ``inject_qkov_heads`` under each block.
* ``encode_image_suppress_measure(image, spec, blocks)`` removes and
  renormalizes attention mass at ``cols`` for optional ``heads``/``rows``, then
  returns downstream z.
* ``encode_image_capture_residual(image, blocks)`` returns the final embedding,
  post-block residuals and token L2 norms. ``encode_image_project(image, spec)``
  removes orthonormal directions ``U`` at boolean ``mask`` positions.
* ``encode_image_harvest_mtc(image, blocks, norm_block=-1)`` returns ``xin``,
  ``probs`` and ``z`` by block, plus token norms at ``norm_block``.

Set ``block.attn.head_scale = {head: scale}`` for persistent head scaling.
Tensor token axes include CLS and, after its insertion block, explicit
``READ_NULL``. Full PIECES text controls are ``<any>``, ``<text>``, ``<notext>``
and ``<null>``; only checkpoints containing the trained full bridge accept
``forward_modes`` or ``forward_hard``.
"""

import os
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Dict, Any, Sequence, Mapping, Callable
import numpy as np
import math
import warnings
import torch
from torch import Tensor
from torch import nn

from functools import wraps

from .precision_policy import fp32_island, to_fp32_tree


def _first_floating_tensor(value):
    if isinstance(value, torch.Tensor):
        return value if value.is_floating_point() else None
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_floating_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_floating_tensor(item)
            if found is not None:
                return found
    return None


def _fp32_custom_forward(function):
    """Run small trainable implant math in FP32 outside global AMP autocast."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        reference = _first_floating_tensor((args, kwargs))
        with fp32_island(reference):
            return function(
                self,
                *to_fp32_tree(args),
                **to_fp32_tree(kwargs),
            )
    return wrapped



def _fp32_normalize(value: torch.Tensor, dim: int = -1, eps: float = 1.0e-12) -> torch.Tensor:
    with fp32_island(value):
        return F.normalize(value.float(), dim=dim, eps=eps)


def _fp32_scaled_matmul(scale: torch.Tensor, left: torch.Tensor, right_t: torch.Tensor) -> torch.Tensor:
    with fp32_island(left):
        return scale.float() * (left.float() @ right_t.float())


def _fp32_scaled_einsum(
    equation: str, scale: torch.Tensor, *operands: torch.Tensor
) -> torch.Tensor:
    reference = operands[0] if operands else scale
    with fp32_island(reference):
        return scale.float() * torch.einsum(
            equation, *(operand.float() for operand in operands)
        )

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class _LinearWithBias(torch.nn.Linear):
    bias: Tensor
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)


class MultiheadAttention(nn.Module):
    """
    Patched MultiheadAttention with explicit Q/K/V projections as separate nn.Linear modules.

      - Optional capture of Q/K/V and pre-softmax logits for interpretability experiments.
      - Optional per-batch mask applied ONLY to the CLS query row (tgt index 0) over src tokens,
        implemented as a pre-softmax additive mask (renormalizes).
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # EXPLICIT Q/K/V LINEARS
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

        # existing head ablation support
        self.head_mask = None

        # caches for tracing (populated only when capture=True)
        self.last_q: Optional[torch.Tensor] = None          # [B, H, T, D]
        self.last_k: Optional[torch.Tensor] = None          # [B, H, S, D]
        self.last_v: Optional[torch.Tensor] = None          # [B, H, S, D]
        self.last_logits: Optional[torch.Tensor] = None     # [B, H, T, S]
        self.last_probs: Optional[torch.Tensor] = None      # [B, H, T, S]
        self.last_xin: Optional[torch.Tensor] = None        # [B, T, E]
        # z-capture: per-head outputs BEFORE out_proj [B, H, T, D]
        self.last_z: Optional[torch.Tensor] = None
        # grad-retention node for attribution patching (AtP)
        self._last_z_grad_node: Optional[torch.Tensor] = None
        # persistent per-head output scale factors {head_idx: scale_float}
        # scale=0.0 = ablation, scale=1.0 = no-op, scale>1 = amplification
        self.head_scale: Optional[Dict[int, float]] = None

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query, key, value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        attention_probs_forward_hook=None,
        attention_probs_backwards_hook=None,
        # tracing + CLS-row masking
        capture: bool = False,
        cls_src_keep_mask: Optional[torch.Tensor] = None,  # [B, src_len] bool; only applied to tgt index 0
        cls_mask_includes_self: bool = True,               # if False, also disallow CLS->CLS when masking
        # activation / attribution patching
        inject_z: Optional[torch.Tensor] = None,           # [B, H, T, D] — replace z for inject_heads
        inject_heads: Optional[list] = None,               # head indices to inject; None = all H
        retain_z_grad: bool = False,                       # retain grad on z node for AtP backward
        inject_probs: Optional[torch.Tensor] = None,       # [B, H, T, S] donor attention probabilities
        inject_v: Optional[torch.Tensor] = None,           # [B, H, S, D] donor value vectors
        inject_qkov_heads: Optional[list] = None,          # heads to QK/OV-patch; None = all H
        suppress_src_cols: Optional[torch.Tensor] = None,  # [B, S] bool source columns to remove
        suppress_heads: Optional[list] = None,             # affected heads; None = all H
        suppress_query_rows: Optional[torch.Tensor] = None,# [T] bool affected rows; None = all
    ):
        # Shapes:
        # query: [L, N, E]
        # key:   [S, N, E]
        # value: [S, N, E]
        L, N, E = query.shape
        S = key.shape[0]

        # Compute Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        head_dim = self.head_dim
        num_heads = self.num_heads

        # scaling
        q = q * (head_dim ** -0.5)

        def reshape(x_):
            # x_: [seq, batch, embed_dim] -> [batch*heads, seq, head_dim]
            x_ = x_.permute(1, 0, 2)  # [batch, seq, embed_dim]
            x_ = x_.view(x_.shape[0], x_.shape[1], num_heads, head_dim)  # [B, seq, H, D]
            x_ = x_.permute(0, 2, 1, 3)  # [B, H, seq, D]
            return x_.reshape(-1, x_.shape[2], head_dim)  # [B*H, seq, D]

        q_r = reshape(q)  # [B*H, L, D]
        k_r = reshape(k)  # [B*H, S, D]
        v_r = reshape(v)  # [B*H, S, D]

        # Optional pre-projected K/V prefix used by register-cache experiments.
        prefix_k = getattr(self, "_prefix_k", None)
        if prefix_k is not None:
            prefix_v = self._prefix_v
            k_r = torch.cat([prefix_k.to(device=k_r.device, dtype=k_r.dtype), k_r], dim=1)
            v_r = torch.cat([prefix_v.to(device=v_r.device, dtype=v_r.dtype), v_r], dim=1)

        # per-head masking by zeroing V for heads with mask=0
        if self.head_mask is not None:
            mask_flat = self.head_mask.to(v_r.device).float().repeat(N).view(N * num_heads, 1, 1)
            v_r = v_r * mask_flat

        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                pass
            else:
                raise RuntimeError("attn_mask has unsupported dimension")

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(torch.bool)

        # bias_k, bias_v
        if self.bias_k is not None and self.bias_v is not None:
            k_r = torch.cat([k_r, self.bias_k.repeat(k_r.size(0) // num_heads, 1, 1)], dim=1)
            v_r = torch.cat([v_r, self.bias_v.repeat(v_r.size(0) // num_heads, 1, 1)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        if self.add_zero_attn:
            k_r = torch.cat([k_r, torch.zeros((k_r.size(0), 1, head_dim), dtype=k_r.dtype, device=k_r.device)], dim=1)
            v_r = torch.cat([v_r, torch.zeros((v_r.size(0), 1, head_dim), dtype=v_r.dtype, device=v_r.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        src_len = k_r.size(1)
        tgt_len = q_r.size(1)

        # attention logits: [B*H, T, S]
        attn_logits = torch.bmm(q_r, k_r.transpose(1, 2))
        attn_logits = attn_logits.view(N, num_heads, tgt_len, src_len)  # [B, H, T, S]

        # CLS-row src masking (pre-softmax, renormalizes)
        if cls_src_keep_mask is not None:
            if cls_src_keep_mask.dtype != torch.bool:
                cls_src_keep_mask = cls_src_keep_mask.to(torch.bool)
            if cls_src_keep_mask.shape != (N, src_len):
                raise ValueError(f"cls_src_keep_mask must be [B, src_len] == {(N, src_len)}, got {tuple(cls_src_keep_mask.shape)}")

            # Optionally disallow CLS->CLS (src index 0) when masking
            if not cls_mask_includes_self:
                cls_src_keep_mask = cls_src_keep_mask.clone()
                cls_src_keep_mask[:, 0] = False

            neg_large = torch.finfo(attn_logits.dtype).min
            # apply ONLY to tgt index 0 (CLS query)
            disallow = (~cls_src_keep_mask).view(N, 1, 1, src_len)  # [B,1,1,S]
            attn_logits[:, :, 0:1, :] = attn_logits[:, :, 0:1, :].masked_fill(disallow, neg_large)

        if attn_mask is not None:
            # attn_mask is additive in this implementation
            attn_logits = attn_logits + attn_mask.unsqueeze(1)  # [B,1,T,S]

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # softmax
        attn_probs = F.softmax(attn_logits.view(N * num_heads, tgt_len, src_len), dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        if attention_probs_forward_hook is not None:
            attention_probs_forward_hook(attn_probs)

        if attention_probs_backwards_hook is not None and attn_probs.requires_grad:
            attn_probs.register_hook(attention_probs_backwards_hook)

        # QK/OV patching changes routing probabilities and values independently.
        probs_effective = attn_probs
        v_effective = v_r
        if inject_probs is not None or inject_v is not None:
            probs_bhts = attn_probs.view(N, num_heads, tgt_len, src_len)
            values_bhsd = v_r.view(N, num_heads, src_len, head_dim)
            heads = (
                list(range(num_heads))
                if inject_qkov_heads is None
                else [int(head) for head in inject_qkov_heads]
            )
            if inject_probs is not None:
                donor_probs = inject_probs.to(
                    device=probs_bhts.device, dtype=probs_bhts.dtype
                )
                if donor_probs.shape != probs_bhts.shape:
                    raise ValueError(
                        f"inject_probs must have shape {tuple(probs_bhts.shape)}, "
                        f"got {tuple(donor_probs.shape)}"
                    )
                probs_bhts = probs_bhts.clone()
                probs_bhts[:, heads] = donor_probs[:, heads]
            if inject_v is not None:
                donor_v = inject_v.to(
                    device=values_bhsd.device, dtype=values_bhsd.dtype
                )
                if donor_v.shape != values_bhsd.shape:
                    raise ValueError(
                        f"inject_v must have shape {tuple(values_bhsd.shape)}, "
                        f"got {tuple(donor_v.shape)}"
                    )
                values_bhsd = values_bhsd.clone()
                values_bhsd[:, heads] = donor_v[:, heads]
            probs_effective = probs_bhts.reshape(N * num_heads, tgt_len, src_len)
            v_effective = values_bhsd.reshape(N * num_heads, src_len, head_dim)

        # Remove selected source columns and renormalize the affected rows.
        if suppress_src_cols is not None:
            columns = suppress_src_cols.to(device=attn_probs.device, dtype=torch.bool)
            if columns.shape != (N, src_len):
                raise ValueError(
                    f"suppress_src_cols must have shape {(N, src_len)}, "
                    f"got {tuple(columns.shape)}"
                )
            rows = (
                torch.ones(tgt_len, dtype=torch.bool, device=attn_probs.device)
                if suppress_query_rows is None
                else suppress_query_rows.to(device=attn_probs.device, dtype=torch.bool)
            )
            if rows.shape != (tgt_len,):
                raise ValueError(
                    f"suppress_query_rows must have shape {(tgt_len,)}, got {tuple(rows.shape)}"
                )
            heads = (
                list(range(num_heads))
                if suppress_heads is None
                else [int(head) for head in suppress_heads]
            )
            probs_bhts = probs_effective.view(N, num_heads, tgt_len, src_len).clone()
            keep = (~columns)[:, None, None, :].to(probs_bhts.dtype)
            for head in heads:
                masked = probs_bhts[:, head] * keep[:, 0]
                denominator = masked.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
                renormalized = masked / denominator
                selection = rows[None, :, None]
                probs_bhts[:, head] = torch.where(
                    selection, renormalized, probs_bhts[:, head]
                )
            probs_effective = probs_bhts.reshape(N * num_heads, tgt_len, src_len)

        # attention output: [B*H, T, D]
        attn_output = torch.bmm(probs_effective, v_effective)
        attn_output = attn_output.view(N, num_heads, tgt_len, head_dim)

        # ── z-capture and injection (activation / attribution patching) ──────────
        # attn_output is [B, H, T, D] — the per-head weighted-value outputs before out_proj
        z_per_head = attn_output
        if retain_z_grad and torch.is_grad_enabled():
            z_per_head = z_per_head.clone()
            z_per_head.retain_grad()
            self._last_z_grad_node = z_per_head
        if capture:
            self.last_z = z_per_head.detach().clone()
        if inject_z is not None:
            _heads = inject_heads if inject_heads is not None else list(range(num_heads))
            z_per_head = z_per_head.clone()
            for _h in _heads:
                z_per_head[:, _h] = inject_z[:, _h].to(dtype=z_per_head.dtype, device=z_per_head.device)
        attn_output = z_per_head
        # ─────────────────────────────────────────────────────────────────────────

        # ── persistent head scaling (amplify / attenuate / ablate) ──────────────
        if self.head_scale:
            attn_output = attn_output.clone()
            for _h, _s in self.head_scale.items():
                attn_output[:, _h] = attn_output[:, _h] * _s
        # ────────────────────────────────────────────────────────────────────────

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(N, tgt_len, E)  # [B,T,E]
        attn_output = attn_output.permute(1, 0, 2)  # [T,B,E]
        attn_output = self.out_proj(attn_output)

        # capture tensors for experiments
        if capture:
            # reshape q/k/v to [B,H,T/D,S,D]
            q_c = q_r.view(N, num_heads, tgt_len, head_dim)
            k_c = k_r.view(N, num_heads, src_len, head_dim)
            v_c = v_r.view(N, num_heads, src_len, head_dim)
            self.last_q = q_c
            self.last_k = k_c
            self.last_v = v_c
            self.last_logits = attn_logits  # [B,H,T,S]
            self.last_probs = attn_probs.view(N, num_heads, tgt_len, src_len)
            self.last_xin = query.detach().permute(1, 0, 2).contiguous()

        if need_weights:
            attn_probs_out = attn_probs.view(N, num_heads, tgt_len, src_len)
            return attn_output, attn_probs_out  # NO AVERAGING
        else:
            return attn_output, None


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.attn_probs = None
        self.attn_grad = None


    def set_attn_probs(self, attn_probs):
        # attn_probs: [N*H, T, S] or [B,H,T,S]
        if attn_probs.dim() == 3:
            n_heads = self.attn.num_heads
            N = attn_probs.shape[0] // n_heads
            attn_probs = attn_probs.view(N, n_heads, attn_probs.shape[1], attn_probs.shape[2])
        self.attn_probs = attn_probs.detach().cpu()


    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    # allow returning weights + capture + CLS-row masking
    def attention(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
        cls_src_keep_mask: Optional[torch.Tensor] = None,
        cls_mask_includes_self: bool = True,
        capture: bool = False,
        inject_z: Optional[torch.Tensor] = None,
        inject_heads: Optional[list] = None,
        retain_z_grad: bool = False,
        inject_probs: Optional[torch.Tensor] = None,
        inject_v: Optional[torch.Tensor] = None,
        inject_qkov_heads: Optional[list] = None,
        suppress_src_cols: Optional[torch.Tensor] = None,
        suppress_heads: Optional[list] = None,
        suppress_query_rows: Optional[torch.Tensor] = None,
    ):
        use_backward_hook = torch.is_grad_enabled()
        attn_mask = None

        if self.attn_mask is not None:
            n_ctx = x.shape[0]
            attn_mask = self.attn_mask[..., -n_ctx:, -n_ctx:].to(dtype=x.dtype, device=x.device)

        attention_probs_backwards_hook = self.set_attn_grad if use_backward_hook else None

        attn_out, attn_w = self.attn(
            x, x, x,
            need_weights=need_weights,
            attn_mask=attn_mask,
            attention_probs_forward_hook=self.set_attn_probs,
            attention_probs_backwards_hook=attention_probs_backwards_hook,
            capture=capture,
            cls_src_keep_mask=cls_src_keep_mask,
            cls_mask_includes_self=cls_mask_includes_self,
            inject_z=inject_z,
            inject_heads=inject_heads,
            retain_z_grad=retain_z_grad,
            inject_probs=inject_probs,
            inject_v=inject_v,
            inject_qkov_heads=inject_qkov_heads,
            suppress_src_cols=suppress_src_cols,
            suppress_heads=suppress_heads,
            suppress_query_rows=suppress_query_rows,
        )
        return attn_out, attn_w

    # forward optionally returns extra info (but remains backwards-compatible)
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        cls_src_keep_mask: Optional[torch.Tensor] = None,
        cls_mask_includes_self: bool = True,
        capture: bool = False,
        inject_z: Optional[torch.Tensor] = None,
        inject_heads: Optional[list] = None,
        retain_z_grad: bool = False,
        inject_probs: Optional[torch.Tensor] = None,
        inject_v: Optional[torch.Tensor] = None,
        inject_qkov_heads: Optional[list] = None,
        suppress_src_cols: Optional[torch.Tensor] = None,
        suppress_heads: Optional[list] = None,
        suppress_query_rows: Optional[torch.Tensor] = None,
    ):
        ln1 = self.ln_1(x)
        attn_out, attn_w = self.attention(
            ln1,
            need_weights=return_attn,
            cls_src_keep_mask=cls_src_keep_mask,
            cls_mask_includes_self=cls_mask_includes_self,
            capture=capture,
            inject_z=inject_z,
            inject_heads=inject_heads,
            retain_z_grad=retain_z_grad,
            inject_probs=inject_probs,
            inject_v=inject_v,
            inject_qkov_heads=inject_qkov_heads,
            suppress_src_cols=suppress_src_cols,
            suppress_heads=suppress_heads,
            suppress_query_rows=suppress_query_rows,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if return_attn:
            return x, attn_w
        return x


class Transformer(nn.Module):
    """
    Switch from nn.Sequential to nn.ModuleList so we can:
      - run all but last block normally
      - re-run last block with custom CLS-row attention masks
      - optionally capture last-block attention tensors
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(
        self,
        x: torch.Tensor,
        capture_layers: Optional[set] = None,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            cap = (capture_layers is not None) and (i in capture_layers)
            x = blk(x, capture=cap)  # <-- pass capture flag
        return x

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        capture_blocks: Sequence[int],
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        capture = set(int(i) for i in capture_blocks)
        states: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            x = blk(x)
            if i in capture:
                states[i] = x.permute(1, 0, 2)  # [B,T,C], retain graph
        return x, states

    def forward_until(
        self,
        x: torch.Tensor,
        layer_idx_exclusive: int,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        for i in range(layer_idx_exclusive):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            x = self.resblocks[i](x)
        return x

    @torch.no_grad()
    def forward_cached(
        self,
        x: torch.Tensor,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Run all blocks with z-capture enabled.
        Returns (x_final [T,B,C], cache dict):
          cache[block_idx] = {
            'z':         [B, H, T, D]  per-head z before out_proj (detached)
            'resid_pre': [T, B, C]     residual stream entering the block (detached)
          }
        """
        cache: Dict[int, Dict] = {}
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            blk.attn.last_z = None
            cache[i] = {'resid_pre': x.detach().clone()}
            x = blk(x, capture=True)
            cache[i]['z'] = blk.attn.last_z   # [B, H, T, D] or None
        return x, cache

    def forward_patched(
        self,
        x: torch.Tensor,
        patch_spec: Dict,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Run all blocks; at blocks in patch_spec inject pre-computed z values.
        patch_spec: {block_idx: {'inject_z': Tensor[B,H,T,D], 'inject_heads': list[int]}}
        No grad — use for full activation patching sweeps.
        """
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            spec = patch_spec.get(i, {})
            x = blk(
                x,
                inject_z=spec.get('inject_z', None),
                inject_heads=spec.get('inject_heads', None),
            )
        return x

    def forward_atp(
        self,
        x: torch.Tensor,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Run all blocks with retain_z_grad=True so AtP gradients are captured.
        Returns x_final; grad nodes accessible via blk.attn._last_z_grad_node.
        Must be called under torch.enable_grad().
        """
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            blk.attn._last_z_grad_node = None
            x = blk(x, retain_z_grad=True)
        return x

    @torch.no_grad()
    def forward_inject_capture(
        self,
        x: torch.Tensor,
        inject_spec: Dict,
        capture_blocks: set,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Run all blocks; inject z at blocks in inject_spec, capture z at blocks
        in capture_blocks.  Only the requested blocks are captured — O(1) extra
        memory vs the full-cache version.

        inject_spec:   {block_idx: {'inject_z': Tensor[B,H,T,D], 'inject_heads': list}}
        capture_blocks: set of block indices whose z to return

        Returns (x_final [T,B,C], cache dict):
          cache[block_idx] = {'z': Tensor[B,H,T,D]}   — only for blocks in capture_blocks
        """
        cache: Dict[int, Dict] = {}
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            spec    = inject_spec.get(i, {})
            do_cap  = i in capture_blocks
            if do_cap:
                blk.attn.last_z = None
            x = blk(
                x,
                capture=do_cap,
                inject_z=spec.get('inject_z', None),
                inject_heads=spec.get('inject_heads', None),
            )
            if do_cap:
                cache[i] = {'z': blk.attn.last_z}
        return x, cache

    @torch.no_grad()
    def forward_cached_qkov(
        self,
        x: torch.Tensor,
        capture_blocks: set,
        cache_device=None,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        """Capture post-softmax attention probabilities and head-split values."""
        cache: Dict[int, Dict[str, torch.Tensor]] = {}
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            capture = i in capture_blocks
            if capture:
                blk.attn.last_probs = None
                blk.attn.last_v = None
            x = blk(x, capture=capture)
            if capture:
                probabilities = blk.attn.last_probs.detach()
                values = blk.attn.last_v.detach()
                if cache_device is not None:
                    probabilities = probabilities.to(cache_device)
                    values = values.to(cache_device)
                else:
                    probabilities = probabilities.clone()
                    values = values.clone()
                cache[i] = {"probs": probabilities, "v": values}
                for name in (
                    "last_q", "last_k", "last_v", "last_logits", "last_probs",
                    "last_z", "last_xin",
                ):
                    setattr(blk.attn, name, None)
        return x, cache

    @torch.no_grad()
    def forward_qkov_patched(
        self,
        x: torch.Tensor,
        patch_spec: Dict,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        """Patch attention probabilities and/or values at selected heads."""
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            spec = patch_spec.get(i, {})
            x = blk(
                x,
                inject_probs=spec.get("inject_probs"),
                inject_v=spec.get("inject_v"),
                inject_qkov_heads=spec.get("inject_qkov_heads"),
            )
        return x

    @torch.no_grad()
    def forward_suppress_measure(
        self,
        x: torch.Tensor,
        suppress_spec: Dict,
        measure_blocks: set,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        """Suppress selected source columns and capture downstream per-head z."""
        zcache: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            spec = suppress_spec.get(i)
            capture = i in measure_blocks
            if capture:
                blk.attn.last_z = None
            x = blk(
                x,
                capture=capture,
                suppress_src_cols=spec.get("cols") if spec else None,
                suppress_heads=spec.get("heads") if spec else None,
                suppress_query_rows=spec.get("rows") if spec else None,
            )
            if capture:
                zcache[i] = blk.attn.last_z.detach()
                for name in (
                    "last_q", "last_k", "last_v", "last_logits", "last_probs",
                    "last_z", "last_xin",
                ):
                    setattr(blk.attn, name, None)
        return x, zcache

    @torch.no_grad()
    def forward_capture_residual(
        self,
        x: torch.Tensor,
        blocks,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        """Capture post-block residual streams and final requested token norms."""
        requested = set(int(block) for block in blocks)
        if not requested:
            raise ValueError("blocks must contain at least one block index")
        invalid = sorted(block for block in requested if block < 0 or block >= self.layers)
        if invalid:
            raise ValueError(f"Invalid visual block indices: {invalid}")
        last_requested = max(requested)
        residuals: Dict[int, torch.Tensor] = {}
        token_norm = None
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            x = blk(x)
            if i in requested:
                residuals[i] = x.detach().permute(1, 0, 2).contiguous()
                if i == last_requested:
                    token_norm = x.detach().float().norm(dim=-1).permute(1, 0).contiguous()
        return x, residuals, token_norm

    @torch.no_grad()
    def forward_project_residual(
        self,
        x: torch.Tensor,
        spec: Dict,
        before_block_hook: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ):
        """Remove components in orthonormal direction sets at masked positions."""
        for i, blk in enumerate(self.resblocks):
            if before_block_hook is not None:
                x = before_block_hook(i, x)
            x = blk(x)
            block_spec = spec.get(i)
            if block_spec is None:
                continue
            residual = x.permute(1, 0, 2)
            directions = block_spec["U"].to(
                device=residual.device, dtype=residual.dtype
            )
            mask = block_spec["mask"].to(
                device=residual.device, dtype=residual.dtype
            ).unsqueeze(-1)
            if directions.dim() == 2:
                projection = (residual @ directions.t()) @ directions
            elif directions.dim() == 3:
                projection = torch.bmm(
                    torch.bmm(residual, directions.transpose(1, 2)), directions
                )
            else:
                raise ValueError("U must have shape [rank, width] or [B, rank, width]")
            residual = residual - mask * projection
            x = residual.permute(1, 0, 2).contiguous()
        return x


def _safe_masked_softmax(logits: torch.Tensor, keep_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax over ``dim`` with an all-masked-row safe fallback to zeros."""
    keep_mask = keep_mask.to(device=logits.device, dtype=torch.bool)
    while keep_mask.ndim < logits.ndim:
        keep_mask = keep_mask.unsqueeze(1)
    logits_fp32 = logits.float()
    masked = logits_fp32.masked_fill(~keep_mask, torch.finfo(torch.float32).min)
    probs = masked.softmax(dim=dim)
    valid = keep_mask.any(dim=dim, keepdim=True)
    return torch.where(valid, probs, torch.zeros_like(probs))


def _normalize_sigmoid_read_direction(value: torch.Tensor) -> torch.Tensor:
    """Normalize sigmoid-reader directions in FP32, then restore storage dtype.

    ``F.normalize`` defaults to eps=1e-12, which underflows to zero in FP16.
    A deliberately abstaining sigmoid reader can therefore turn an exact-zero
    feature into 0/0 NaNs unless normalization is performed in FP32.
    """
    return F.normalize(value.float(), dim=-1, eps=1.0e-12)


class SoftmaxCrossOnlyReadBridge(nn.Module):
    """PreNorm text-query -> visual-key/value attention.

    Text states influence only the attention weights. The returned read feature
    is built exclusively from visual values. A learned patch-textness bias can
    steer attention toward glyph-bearing patches without injecting text values.
    """

    def __init__(
        self,
        text_width: int,
        vision_width: int,
        bridge_width: int,
        output_dim: int,
        heads: int = 4,
    ):
        super().__init__()
        if bridge_width % heads != 0:
            raise ValueError(f"bridge_width={bridge_width} must be divisible by heads={heads}")
        self.bridge_width = int(bridge_width)
        self.heads = int(heads)
        self.head_dim = self.bridge_width // self.heads

        self.text_ln = LayerNorm(text_width)
        self.vision_ln = LayerNorm(vision_width)
        self.q_proj = nn.Linear(text_width, bridge_width, bias=True)
        self.k_proj = nn.Linear(vision_width, bridge_width, bias=True)
        self.v_proj = nn.Linear(vision_width, bridge_width, bias=True)
        self.out_proj = nn.Linear(bridge_width, output_dim, bias=True)

        # Registers begin with exactly zero authority.
        self.register_gate = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.register_gate)

    @_fp32_custom_forward
    def forward(
        self,
        text_query: torch.Tensor,
        visual_tokens: torch.Tensor,
        register_mask: Optional[torch.Tensor] = None,
        patch_textness: Optional[torch.Tensor] = None,
        textness_beta: Optional[torch.Tensor] = None,
        read_null_index: Optional[int] = None,
        return_attention: bool = False,
    ):
        if text_query.ndim != 2 or visual_tokens.ndim != 3:
            raise ValueError(
                f"Expected text_query [N,C] and visual_tokens [B,T,C], got "
                f"{tuple(text_query.shape)} and {tuple(visual_tokens.shape)}"
            )

        B, T, _ = visual_tokens.shape
        N = text_query.shape[0]
        q = self.q_proj(self.text_ln(text_query))
        k = self.k_proj(self.vision_ln(visual_tokens))
        v = self.v_proj(self.vision_ln(visual_tokens))

        q = q.view(N, self.heads, self.head_dim)
        k = k.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum('nhd,bhtd->bnht', q, k) * (self.head_dim ** -0.5)

        if patch_textness is not None:
            if patch_textness.shape != (B, T - 1):
                raise ValueError(
                    f"patch_textness must be {(B, T - 1)}, got {tuple(patch_textness.shape)}"
                )
            beta = 0.0 if textness_beta is None else textness_beta
            beta = torch.as_tensor(beta, device=logits.device, dtype=logits.dtype)
            bias = torch.zeros((B, T), device=logits.device, dtype=logits.dtype)
            safe_textness = patch_textness.to(logits.dtype).clamp(min=1.0e-4, max=1.0)
            bias[:, 1:] = safe_textness.log()
            logits = logits + beta * bias[:, None, None, :]

        # CLS is not visual evidence. Patch and register banks are normalized
        # separately so implicit registers cannot engulf ordinary patch memory.
        patch_keep = torch.ones((B, T), dtype=torch.bool, device=visual_tokens.device)
        patch_keep[:, 0] = False
        reg_keep = torch.zeros_like(patch_keep)
        if register_mask is not None:
            if register_mask.shape != (B, T - 1):
                raise ValueError(
                    f"register_mask must be {(B, T - 1)}, got {tuple(register_mask.shape)}"
                )
            reg_keep[:, 1:] = register_mask.to(device=visual_tokens.device, dtype=torch.bool)
            patch_keep[:, 1:] &= ~reg_keep[:, 1:]

        patch_probs = _safe_masked_softmax(logits, patch_keep[:, None, None, :], dim=-1)
        patch_out = torch.einsum('bnht,bhtd->bnhd', patch_probs, v)

        read_null_attention = None
        if read_null_index is not None:
            null_index = int(read_null_index)
            if null_index < 0:
                null_index += T
            if null_index <= 0 or null_index >= T:
                raise ValueError(
                    f"read_null_index={read_null_index} resolves to {null_index}, "
                    f"outside visual evidence tokens T={T}"
                )
            read_null_attention = patch_probs[..., null_index]

        reg_probs = None
        if register_mask is not None:
            reg_probs = _safe_masked_softmax(logits, reg_keep[:, None, None, :], dim=-1)
            reg_out = torch.einsum('bnht,bhtd->bnhd', reg_probs, v)
            patch_out = patch_out + self.register_gate.to(patch_out.dtype) * reg_out

        read_feature = self.out_proj(patch_out.reshape(B, N, self.bridge_width))
        if not return_attention:
            return read_feature
        return read_feature, {
            'patch_attention': patch_probs,
            'register_attention': reg_probs,
            'register_gate': self.register_gate,
            'patch_textness': patch_textness,
            'textness_beta': textness_beta,
            'read_null_attention': read_null_attention,
            'read_null_index': read_null_index,
        }



class SigmoidMassCrossOnlyReadBridge(nn.Module):
    """Independent patch-evidence attention with explicit bounded read authority.

    Patch weights are sigmoid gates rather than a token-normalized softmax.
    Registers may contribute to the pooled semantic direction, but never to the
    patch evidence mass.  ``evidence`` is returned separately so downstream
    cosine scoring can retain near-zero authority when no readable patch exists.
    """

    NUMERICS_VERSION = 2

    def __init__(
        self,
        text_width: int,
        vision_width: int,
        bridge_width: int,
        output_dim: int,
        heads: int = 4,
        sigmoid_bias_init: float = -2.0,
    ):
        super().__init__()
        if bridge_width % heads != 0:
            raise ValueError(f"bridge_width={bridge_width} must be divisible by heads={heads}")
        self.bridge_width = int(bridge_width)
        self.heads = int(heads)
        self.head_dim = self.bridge_width // self.heads

        self.text_ln = LayerNorm(text_width)
        self.vision_ln = LayerNorm(vision_width)
        self.q_proj = nn.Linear(text_width, bridge_width, bias=True)
        self.k_proj = nn.Linear(vision_width, bridge_width, bias=True)
        self.v_proj = nn.Linear(vision_width, bridge_width, bias=True)
        self.out_proj = nn.Linear(bridge_width, output_dim, bias=True)

        self.register_gate = nn.Parameter(torch.zeros(()))
        self.sigmoid_head_bias = nn.Parameter(torch.full((self.heads,), float(sigmoid_bias_init)))
        self.reset_parameters(sigmoid_bias_init=float(sigmoid_bias_init))

    def reset_parameters(self, sigmoid_bias_init: float = -2.0) -> None:
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.register_gate)
        nn.init.constant_(self.sigmoid_head_bias, float(sigmoid_bias_init))

    @_fp32_custom_forward
    def forward(
        self,
        text_query: torch.Tensor,
        visual_tokens: torch.Tensor,
        register_mask: Optional[torch.Tensor] = None,
        patch_textness: Optional[torch.Tensor] = None,
        textness_beta: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        if text_query.ndim != 2 or visual_tokens.ndim != 3:
            raise ValueError(
                f"Expected text_query [N,C] and visual_tokens [B,T,C], got "
                f"{tuple(text_query.shape)} and {tuple(visual_tokens.shape)}"
            )

        B, T, _ = visual_tokens.shape
        N = text_query.shape[0]
        q = self.q_proj(self.text_ln(text_query))
        normalized_visual = self.vision_ln(visual_tokens)
        k = self.k_proj(normalized_visual)
        v = self.v_proj(normalized_visual)

        q = q.view(N, self.heads, self.head_dim)
        k = k.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum('nhd,bhtd->bnht', q, k) * (self.head_dim ** -0.5)

        patch_keep = torch.ones((B, T), dtype=torch.bool, device=visual_tokens.device)
        patch_keep[:, 0] = False
        reg_keep = torch.zeros_like(patch_keep)
        if register_mask is not None:
            if register_mask.shape != (B, T - 1):
                raise ValueError(
                    f"register_mask must be {(B, T - 1)}, got {tuple(register_mask.shape)}"
                )
            reg_keep[:, 1:] = register_mask.to(device=visual_tokens.device, dtype=torch.bool)
            patch_keep[:, 1:] &= ~reg_keep[:, 1:]

        if patch_textness is not None:
            if patch_textness.shape != (B, T - 1):
                raise ValueError(
                    f"patch_textness must be {(B, T - 1)}, got {tuple(patch_textness.shape)}"
                )
            beta = 0.0 if textness_beta is None else textness_beta
            beta = torch.as_tensor(beta, device=logits.device, dtype=logits.dtype)
            # Compute the prior in FP32. In FP16, 1 - 1e-4 rounds to exactly
            # 1.0, so a saturated glyph probability produces +inf and the
            # initial beta=0 evaluates as 0 * inf = NaN.
            textness_logit = torch.logit(
                patch_textness.float(), eps=1.0e-4
            )
            textness_prior = beta.float() * textness_logit
            logits = logits.clone()
            logits[:, :, :, 1:] = (
                logits[:, :, :, 1:]
                + textness_prior[:, None, None, :].to(logits.dtype)
            )

        valid_patch_count = patch_keep.sum(dim=-1).clamp(min=1).to(logits.dtype)
        logits = (
            logits
            + self.sigmoid_head_bias.to(logits.dtype)[None, None, :, None]
            - valid_patch_count.log()[:, None, None, None]
        )
        patch_weights = torch.sigmoid(logits) * patch_keep[:, None, None, :].to(logits.dtype)
        patch_mass = patch_weights.sum(dim=-1)
        patch_direction = torch.einsum('bnht,bhtd->bnhd', patch_weights, v)
        patch_direction = patch_direction / (patch_mass[..., None] + 1.0e-6)

        reg_probs = None
        if register_mask is not None:
            reg_probs = _safe_masked_softmax(logits, reg_keep[:, None, None, :], dim=-1)
            reg_direction = torch.einsum('bnht,bhtd->bnhd', reg_probs, v)
            patch_direction = (
                patch_direction
                + self.register_gate.to(patch_direction.dtype) * reg_direction
            )

        evidence = 1.0 - torch.exp(-patch_mass.mean(dim=-1))
        read_feature = self.out_proj(patch_direction.reshape(B, N, self.bridge_width))
        if not return_attention:
            return read_feature
        return read_feature, {
            'patch_attention': patch_weights,
            'register_attention': reg_probs,
            'register_gate': self.register_gate,
            'patch_textness': patch_textness,
            'textness_beta': textness_beta,
            'patch_mass': patch_mass,
            'evidence': evidence,
        }


class SigmoidAllCrossOnlyReadBridge(nn.Module):
    """Raw independent-sigmoid version of the PIECES late read attention.

    This is deliberately *not* sigmoid-mass attention: no attention weights are
    renormalized after sigmoid, the pooled value is not divided by total mass,
    and no evidence multiplier is introduced downstream.  The only architectural
    change relative to :class:`SoftmaxCrossOnlyReadBridge` is replacing each
    patch/register-bank softmax by independent sigmoid edges.

    A dynamic ``-log(valid_source_count)`` offset gives approximately unit total
    mass at zero QK logits, while a learnable per-head offset lets each head
    subsequently choose how much total evidence it wants to read.
    """

    NUMERICS_VERSION = 1
    ATTENTION_NORMALIZATION = "independent_sigmoid_raw"
    NORMALIZES_ATTENTION = False

    def __init__(
        self,
        text_width: int,
        vision_width: int,
        bridge_width: int,
        output_dim: int,
        heads: int = 4,
    ):
        super().__init__()
        if bridge_width % heads != 0:
            raise ValueError(f"bridge_width={bridge_width} must be divisible by heads={heads}")
        self.bridge_width = int(bridge_width)
        self.heads = int(heads)
        self.head_dim = self.bridge_width // self.heads

        self.text_ln = LayerNorm(text_width)
        self.vision_ln = LayerNorm(vision_width)
        self.q_proj = nn.Linear(text_width, bridge_width, bias=True)
        self.k_proj = nn.Linear(vision_width, bridge_width, bias=True)
        self.v_proj = nn.Linear(vision_width, bridge_width, bias=True)
        self.out_proj = nn.Linear(bridge_width, output_dim, bias=True)

        # Keep the existing register authority mechanism exactly as before.
        self.register_gate = nn.Parameter(torch.zeros(()))
        # Offsets relative to the dynamic -log(valid_count) initialization.
        self.sigmoid_patch_head_bias = nn.Parameter(torch.zeros(self.heads))
        self.sigmoid_register_head_bias = nn.Parameter(torch.zeros(self.heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.register_gate)
        nn.init.zeros_(self.sigmoid_patch_head_bias)
        nn.init.zeros_(self.sigmoid_register_head_bias)

    @_fp32_custom_forward
    def forward(
        self,
        text_query: torch.Tensor,
        visual_tokens: torch.Tensor,
        register_mask: Optional[torch.Tensor] = None,
        patch_textness: Optional[torch.Tensor] = None,
        textness_beta: Optional[torch.Tensor] = None,
        read_null_index: Optional[int] = None,
        return_attention: bool = False,
    ):
        if text_query.ndim != 2 or visual_tokens.ndim != 3:
            raise ValueError(
                f"Expected text_query [N,C] and visual_tokens [B,T,C], got "
                f"{tuple(text_query.shape)} and {tuple(visual_tokens.shape)}"
            )

        B, T, _ = visual_tokens.shape
        N = text_query.shape[0]
        q = self.q_proj(self.text_ln(text_query))
        normalized_visual = self.vision_ln(visual_tokens)
        k = self.k_proj(normalized_visual)
        v = self.v_proj(normalized_visual)

        q = q.view(N, self.heads, self.head_dim)
        k = k.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum('nhd,bhtd->bnht', q, k) * (self.head_dim ** -0.5)

        # Preserve the softmax reader's glyph prior exactly: log(probability),
        # not logit(probability).  This keeps the experiment one-variable.
        if patch_textness is not None:
            if patch_textness.shape != (B, T - 1):
                raise ValueError(
                    f"patch_textness must be {(B, T - 1)}, got {tuple(patch_textness.shape)}"
                )
            beta = 0.0 if textness_beta is None else textness_beta
            beta = torch.as_tensor(beta, device=logits.device, dtype=torch.float32)
            safe_textness = patch_textness.float().clamp(min=1.0e-4, max=1.0)
            bias = torch.zeros((B, T), device=logits.device, dtype=torch.float32)
            bias[:, 1:] = safe_textness.log()
            logits = logits.float() + beta * bias[:, None, None, :]
        else:
            logits = logits.float()

        # Same two evidence banks as the softmax reader. CLS is never evidence;
        # implicit register patches are removed from the ordinary patch bank.
        patch_keep = torch.ones((B, T), dtype=torch.bool, device=visual_tokens.device)
        patch_keep[:, 0] = False
        reg_keep = torch.zeros_like(patch_keep)
        if register_mask is not None:
            if register_mask.shape != (B, T - 1):
                raise ValueError(
                    f"register_mask must be {(B, T - 1)}, got {tuple(register_mask.shape)}"
                )
            reg_keep[:, 1:] = register_mask.to(device=visual_tokens.device, dtype=torch.bool)
            patch_keep[:, 1:] &= ~reg_keep[:, 1:]

        # Independent raw sigmoid edges. There is intentionally NO division by
        # patch_mass/register_mass and NO sum-to-one normalization afterward.
        patch_count = patch_keep.sum(dim=-1).clamp(min=1).float()
        patch_logits = (
            logits
            + self.sigmoid_patch_head_bias.float()[None, None, :, None]
            - patch_count.log()[:, None, None, None]
        )
        patch_weights = torch.sigmoid(patch_logits)
        patch_weights = patch_weights * patch_keep[:, None, None, :].float()
        patch_out = torch.einsum('bnht,bhtd->bnhd', patch_weights.to(v.dtype), v)
        patch_mass = patch_weights.sum(dim=-1)

        read_null_attention = None
        if read_null_index is not None:
            null_index = int(read_null_index)
            if null_index < 0:
                null_index += T
            if null_index <= 0 or null_index >= T:
                raise ValueError(
                    f"read_null_index={read_null_index} resolves to {null_index}, "
                    f"outside visual evidence tokens T={T}"
                )
            read_null_attention = patch_weights[..., null_index]

        reg_weights = None
        reg_mass = None
        if register_mask is not None:
            reg_count = reg_keep.sum(dim=-1).clamp(min=1).float()
            reg_logits = (
                logits
                + self.sigmoid_register_head_bias.float()[None, None, :, None]
                - reg_count.log()[:, None, None, None]
            )
            reg_weights = torch.sigmoid(reg_logits)
            reg_weights = reg_weights * reg_keep[:, None, None, :].float()
            reg_out = torch.einsum('bnht,bhtd->bnhd', reg_weights.to(v.dtype), v)
            reg_mass = reg_weights.sum(dim=-1)
            patch_out = patch_out + self.register_gate.to(patch_out.dtype) * reg_out

        read_feature = self.out_proj(patch_out.reshape(B, N, self.bridge_width))
        if not return_attention:
            return read_feature
        return read_feature, {
            'patch_attention': patch_weights,
            'register_attention': reg_weights,
            'register_gate': self.register_gate,
            'patch_textness': patch_textness,
            'textness_beta': textness_beta,
            'patch_mass': patch_mass,
            'register_mass': reg_mass,
            'sigmoid_patch_head_bias': self.sigmoid_patch_head_bias,
            'sigmoid_register_head_bias': self.sigmoid_register_head_bias,
            'read_null_attention': read_null_attention,
            'read_null_index': read_null_index,
            'attention_normalization': self.ATTENTION_NORMALIZATION,
        }


# Backward-compatible symbol for old full-module pickles and external imports.
CrossOnlyReadBridge = SoftmaxCrossOnlyReadBridge


class PreNormVisualQueryPool(nn.Module):
    """Candidate-independent learned-query pooling over late visual tokens."""

    def __init__(
        self,
        vision_width: int,
        pool_width: int,
        output_dim: int,
        heads: int = 4,
        zero_output: bool = True,
    ):
        super().__init__()
        if pool_width % heads != 0:
            raise ValueError(f"pool_width={pool_width} must be divisible by heads={heads}")
        self.pool_width = int(pool_width)
        self.heads = int(heads)
        self.head_dim = self.pool_width // self.heads

        self.vision_ln = LayerNorm(vision_width)
        self.query = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.k_proj = nn.Linear(vision_width, pool_width, bias=True)
        self.v_proj = nn.Linear(vision_width, pool_width, bias=True)
        self.out_proj = nn.Linear(pool_width, output_dim, bias=True)
        self.zero_output = bool(zero_output)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, std=self.head_dim ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        if self.zero_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    @_fp32_custom_forward
    def forward(
        self,
        visual_tokens: torch.Tensor,
        return_attention: bool = False,
        exclude_last_token: bool = False,
    ):
        B, T, _ = visual_tokens.shape
        x = self.vision_ln(visual_tokens)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum('hd,bhtd->bht', self.query.to(k.dtype), k) * (self.head_dim ** -0.5)
        keep = torch.ones((B, T), dtype=torch.bool, device=visual_tokens.device)
        keep[:, 0] = False
        if exclude_last_token:
            if T < 3:
                raise ValueError("Cannot exclude READ_NULL from a visual sequence with fewer than 3 tokens")
            keep[:, -1] = False
        probs = _safe_masked_softmax(logits, keep[:, None, :], dim=-1)
        pooled = torch.einsum('bht,bhtd->bhd', probs, v).reshape(B, self.pool_width)
        out = self.out_proj(pooled)
        if return_attention:
            return out, probs
        return out


class SigmoidAllVisualQueryPool(nn.Module):
    """Raw independent-sigmoid replacement for PIECES content pooling softmax."""

    ATTENTION_NORMALIZATION = "independent_sigmoid_raw"
    NORMALIZES_ATTENTION = False

    def __init__(
        self,
        vision_width: int,
        pool_width: int,
        output_dim: int,
        heads: int = 4,
        zero_output: bool = True,
    ):
        super().__init__()
        if pool_width % heads != 0:
            raise ValueError(f"pool_width={pool_width} must be divisible by heads={heads}")
        self.pool_width = int(pool_width)
        self.heads = int(heads)
        self.head_dim = self.pool_width // self.heads

        self.vision_ln = LayerNorm(vision_width)
        self.query = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.k_proj = nn.Linear(vision_width, pool_width, bias=True)
        self.v_proj = nn.Linear(vision_width, pool_width, bias=True)
        self.out_proj = nn.Linear(pool_width, output_dim, bias=True)
        self.sigmoid_head_bias = nn.Parameter(torch.zeros(self.heads))
        self.zero_output = bool(zero_output)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, std=self.head_dim ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.sigmoid_head_bias)
        if self.zero_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    @_fp32_custom_forward
    def forward(
        self,
        visual_tokens: torch.Tensor,
        return_attention: bool = False,
        exclude_last_token: bool = False,
    ):
        B, T, _ = visual_tokens.shape
        x = self.vision_ln(visual_tokens)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum('hd,bhtd->bht', self.query.to(k.dtype), k) * (self.head_dim ** -0.5)
        keep = torch.ones((B, T), dtype=torch.bool, device=visual_tokens.device)
        keep[:, 0] = False
        if exclude_last_token:
            if T < 3:
                raise ValueError("Cannot exclude READ_NULL from a visual sequence with fewer than 3 tokens")
            keep[:, -1] = False
        valid_count = keep.sum(dim=-1).clamp(min=1).float()
        adjusted = (
            logits.float()
            + self.sigmoid_head_bias.float()[None, :, None]
            - valid_count.log()[:, None, None]
        )
        weights = torch.sigmoid(adjusted) * keep[:, None, :].float()
        pooled = torch.einsum('bht,bhtd->bhd', weights.to(v.dtype), v).reshape(B, self.pool_width)
        out = self.out_proj(pooled)
        if return_attention:
            return out, weights
        return out



class ContentCorrectionImplant(nn.Module):
    """Late visual correction without lexical reader or routing components."""

    def __init__(
        self,
        vision_width: int,
        output_dim: int,
        tap_blocks: Sequence[int],
        pool_width: int = 256,
        heads: int = 4,
        attention_architecture: str = "softmax",
        read_null_enabled: bool = False,
    ):
        super().__init__()
        self._tap_blocks = tuple(int(block) for block in tap_blocks)
        if not self._tap_blocks:
            raise ValueError("tap_blocks must be non-empty")
        pool_type = (
            SigmoidAllVisualQueryPool
            if attention_architecture == "sigmoid_all"
            else PreNormVisualQueryPool
        )
        self.content_pool = pool_type(
            vision_width=vision_width,
            pool_width=pool_width,
            output_dim=output_dim,
            heads=heads,
            zero_output=True,
        )
        self.content_tap_logits = nn.Parameter(torch.zeros(len(self._tap_blocks)))
        self.read_null_enabled = bool(read_null_enabled)

    def tap_block_list(self) -> List[int]:
        return list(self._tap_blocks)

    def capture_block_list(self) -> List[int]:
        return self.tap_block_list()

    @staticmethod
    def _states(
        states: Dict[int, torch.Tensor], blocks: Sequence[int]
    ) -> List[torch.Tensor]:
        missing = [block for block in blocks if block not in states]
        if missing:
            raise KeyError(f"Missing requested visual tap states for blocks: {missing}")
        return [states[block] for block in blocks]

    @_fp32_custom_forward
    def content_correction(
        self, states: Dict[int, torch.Tensor], return_details: bool = False
    ):
        blocks = self.tap_block_list()
        ordered = self._states(states, blocks)
        weights = self.content_tap_logits.float().softmax(dim=0)
        outputs = []
        attention = {}
        for block_index, state in zip(blocks, ordered):
            if return_details:
                output, probabilities = self.content_pool(
                    state,
                    return_attention=True,
                    exclude_last_token=self.read_null_enabled,
                )
                attention[block_index] = probabilities
            else:
                output = self.content_pool(
                    state, exclude_last_token=self.read_null_enabled
                )
            outputs.append(output)
        mixed = torch.einsum("k,kbd->bd", weights, torch.stack(outputs, dim=0))
        if return_details:
            return mixed, {
                "tap_weights": weights,
                "per_block_attention": attention,
            }
        return mixed


class EarlyOrthographicBridge(nn.Module):
    """Candidate-conditioned orthographic match from early visual states.

    A wide tokenwise MLP gives a supervised head enough capacity to recover a
    sparse glyph/orthography subspace without modifying the frozen ViT.  Text
    states affect attention weights only; returned features are visual values.
    """

    def __init__(
        self,
        text_width: int,
        vision_width: int,
        expanded_width: int,
        bridge_width: int,
        output_dim: int,
        heads: int = 4,
    ):
        super().__init__()
        if bridge_width % heads != 0:
            raise ValueError(f"bridge_width={bridge_width} must be divisible by heads={heads}")
        self.bridge_width = int(bridge_width)
        self.heads = int(heads)
        self.head_dim = self.bridge_width // self.heads
        self.text_ln = LayerNorm(text_width)
        self.patch_ln = LayerNorm(vision_width)
        self.patch_expand = nn.Linear(vision_width, expanded_width)
        self.patch_contract = nn.Linear(expanded_width, bridge_width)
        self.q_proj = nn.Linear(text_width, bridge_width)
        self.k_proj = nn.Linear(bridge_width, bridge_width)
        self.v_proj = nn.Linear(bridge_width, bridge_width)
        self.out_proj = nn.Linear(bridge_width, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (
            self.patch_expand, self.patch_contract, self.q_proj,
            self.k_proj, self.v_proj, self.out_proj,
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    @_fp32_custom_forward
    def forward(
        self,
        text_query: torch.Tensor,
        visual_tokens: torch.Tensor,
        return_attention: bool = False,
    ):
        patches = visual_tokens[:, 1:, :]
        B, P, _ = patches.shape
        N = text_query.shape[0]
        patch_hidden = F.gelu(self.patch_expand(self.patch_ln(patches)))
        patch_hidden = F.gelu(self.patch_contract(patch_hidden))
        q = self.q_proj(self.text_ln(text_query)).view(N, self.heads, self.head_dim)
        k = self.k_proj(patch_hidden).view(B, P, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(patch_hidden).view(B, P, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum('nhd,bhpd->bnhp', q, k) * (self.head_dim ** -0.5)
        probs = logits.softmax(dim=-1)
        pooled = torch.einsum('bnhp,bhpd->bnhd', probs, v)
        feature = self.out_proj(pooled.reshape(B, N, self.bridge_width))
        if return_attention:
            return feature, probs
        return feature


class SigmoidAllEarlyOrthographicBridge(nn.Module):
    """Raw independent-sigmoid replacement for PIECES orthographic softmax."""

    ATTENTION_NORMALIZATION = "independent_sigmoid_raw"
    NORMALIZES_ATTENTION = False

    def __init__(
        self,
        text_width: int,
        vision_width: int,
        expanded_width: int,
        bridge_width: int,
        output_dim: int,
        heads: int = 4,
    ):
        super().__init__()
        if bridge_width % heads != 0:
            raise ValueError(f"bridge_width={bridge_width} must be divisible by heads={heads}")
        self.bridge_width = int(bridge_width)
        self.heads = int(heads)
        self.head_dim = self.bridge_width // self.heads
        self.text_ln = LayerNorm(text_width)
        self.patch_ln = LayerNorm(vision_width)
        self.patch_expand = nn.Linear(vision_width, expanded_width)
        self.patch_contract = nn.Linear(expanded_width, bridge_width)
        self.q_proj = nn.Linear(text_width, bridge_width)
        self.k_proj = nn.Linear(bridge_width, bridge_width)
        self.v_proj = nn.Linear(bridge_width, bridge_width)
        self.out_proj = nn.Linear(bridge_width, output_dim)
        self.sigmoid_head_bias = nn.Parameter(torch.zeros(self.heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (
            self.patch_expand, self.patch_contract, self.q_proj,
            self.k_proj, self.v_proj, self.out_proj,
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.sigmoid_head_bias)

    @_fp32_custom_forward
    def forward(
        self,
        text_query: torch.Tensor,
        visual_tokens: torch.Tensor,
        return_attention: bool = False,
    ):
        patches = visual_tokens[:, 1:, :]
        B, P, _ = patches.shape
        N = text_query.shape[0]
        patch_hidden = F.gelu(self.patch_expand(self.patch_ln(patches)))
        patch_hidden = F.gelu(self.patch_contract(patch_hidden))
        q = self.q_proj(self.text_ln(text_query)).view(N, self.heads, self.head_dim)
        k = self.k_proj(patch_hidden).view(B, P, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(patch_hidden).view(B, P, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum('nhd,bhpd->bnhp', q, k) * (self.head_dim ** -0.5)
        adjusted = (
            logits.float()
            + self.sigmoid_head_bias.float()[None, None, :, None]
            - math.log(max(1, P))
        )
        weights = torch.sigmoid(adjusted)
        pooled = torch.einsum('bnhp,bhpd->bnhd', weights.to(v.dtype), v)
        feature = self.out_proj(pooled.reshape(B, N, self.bridge_width))
        if return_attention:
            return feature, weights
        return feature


class EarlySourceGate(nn.Module):
    """Patch-local source detector and global text-source/readability head."""

    N_STATS = 8

    def __init__(self, vision_width: int, expanded_width: int, hidden_width: int = 128):
        super().__init__()
        self.patch_ln = LayerNorm(vision_width)
        self.patch_expand = nn.Linear(vision_width, expanded_width)
        self.patch_contract = nn.Linear(expanded_width, hidden_width)
        self.patch_out = nn.Linear(hidden_width, 1)
        self.stats_fc1 = nn.Linear(self.N_STATS, hidden_width)
        self.stats_fc2 = nn.Linear(hidden_width, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (
            self.patch_expand, self.patch_contract, self.patch_out,
            self.stats_fc1, self.stats_fc2,
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    @_fp32_custom_forward
    def patch_logits(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        patches = visual_tokens[:, 1:, :]
        x = F.gelu(self.patch_expand(self.patch_ln(patches)))
        x = F.gelu(self.patch_contract(x))
        return self.patch_out(x).squeeze(-1)

    @_fp32_custom_forward
    def aggregate(self, patch_logits: torch.Tensor):
        probs = patch_logits.sigmoid()
        B, P = probs.shape
        grid = int(round(P ** 0.5))
        topk = probs.topk(k=min(8, P), dim=-1).values.mean(dim=-1)
        soft_area = torch.sigmoid((probs - 0.50) * 12.0).mean(dim=-1)
        if grid * grid == P:
            spatial = probs.view(B, grid, grid)
            row_max_mean = spatial.amax(dim=2).mean(dim=1)
            col_max_mean = spatial.amax(dim=1).mean(dim=1)
            row_density_max = spatial.mean(dim=2).amax(dim=1)
        else:
            row_max_mean = probs.amax(dim=-1)
            col_max_mean = probs.amax(dim=-1)
            row_density_max = probs.mean(dim=-1)
        stats = torch.stack(
            [
                probs.mean(dim=-1),
                probs.amax(dim=-1),
                topk,
                soft_area,
                row_max_mean,
                col_max_mean,
                row_density_max,
                patch_logits.mean(dim=-1),
            ],
            dim=-1,
        )
        logits = self.stats_fc2(F.gelu(self.stats_fc1(stats)))
        return logits, probs, stats


class CandidateTrustRouter(nn.Module):
    """Candidate-conditioned trust, separate from source availability."""

    N_FEATURES = 16

    def __init__(self, hidden_width: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(self.N_FEATURES, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width // 2)
        self.fc3 = nn.Linear(hidden_width // 2, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, -4.0)

    @_fp32_custom_forward
    def forward(
        self,
        content_logits: torch.Tensor,
        read_logits: torch.Tensor,
        null_logits: torch.Tensor,
        early_logits: torch.Tensor,
        content_image: torch.Tensor,
        read_image: torch.Tensor,
        content_text: torch.Tensor,
        read_text: torch.Tensor,
        source_logits: torch.Tensor,
        source_stats: torch.Tensor,
        injection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N = content_logits.shape
        dtype = content_logits.dtype
        content_logits = content_logits.detach()
        read_logits = read_logits.detach().to(dtype)
        null_logits = null_logits.detach().to(dtype).reshape(B, 1)
        early_logits = early_logits.detach().to(dtype)
        content_image = content_image.detach().to(dtype)
        read_image = read_image.detach().to(dtype)
        content_text = content_text.detach().to(dtype)
        read_text = read_text.detach().to(dtype)
        source_logits = source_logits.detach()
        source_stats = source_stats.detach().to(dtype)

        relative = read_logits - null_logits
        positive = 0.5 * (relative + torch.sqrt(relative.square() + 1.0e-4))
        ci_ri = torch.einsum('bd,bnd->bn', content_image, read_image)
        ct_rt = torch.einsum('nd,nd->n', content_text, read_text)[None, :].expand(B, N)
        source_probs = source_logits.float().sigmoid().to(dtype)
        source = source_probs[:, 0:1].expand(B, N)
        ordered = source_probs[:, 1:2].expand(B, N)
        glyph_mean = source_stats[:, 0:1].expand(B, N)
        glyph_max = source_stats[:, 1:2].expand(B, N)
        glyph_topk = source_stats[:, 2:3].expand(B, N)
        glyph_area = source_stats[:, 3:4].expand(B, N)
        if injection is None:
            inject = torch.zeros((B, N), dtype=dtype, device=content_logits.device)
        else:
            inject = injection.detach().to(dtype).reshape(B, 1).expand(B, N)

        features = torch.stack(
            [
                torch.tanh(content_logits / 20.0),
                torch.tanh(read_logits / 20.0),
                torch.tanh(relative / 20.0),
                torch.tanh(positive / 20.0),
                torch.tanh(early_logits / 20.0),
                torch.tanh((early_logits - relative) / 20.0),
                ci_ri,
                ct_rt,
                source,
                ordered,
                glyph_mean,
                glyph_max,
                glyph_topk,
                glyph_area,
                inject,
                inject * torch.tanh(relative / 20.0),
            ],
            dim=-1,
        )
        hidden = F.gelu(self.fc1(features))
        hidden = F.gelu(self.fc2(hidden))
        return torch.sigmoid(self.fc3(hidden).squeeze(-1))


class HardTextReadImplant(nn.Module):
    """Early source/orthography branches plus late lexical readout and trust."""

    def __init__(
        self,
        text_width: int,
        vision_width: int,
        output_dim: int,
        tap_blocks: Sequence[int],
        early_blocks: Sequence[int],
        bridge_width: int = 256,
        heads: int = 4,
        early_expanded_width: int = 4096,
        read_attention_architecture: str = "softmax",
        read_null_enabled: bool = False,
    ):
        super().__init__()
        late = tuple(int(x) for x in tap_blocks)
        early = tuple(int(x) for x in early_blocks)
        if not late or not early:
            raise ValueError('late and early tap sets must be non-empty')
        self.register_buffer('tap_blocks', torch.tensor(late, dtype=torch.long), persistent=True)
        self.register_buffer('ortho_tap_blocks', torch.tensor(early, dtype=torch.long), persistent=True)
        self.register_buffer('source_tap_blocks', torch.tensor(early, dtype=torch.long), persistent=True)
        self.register_buffer('bridge_heads_config', torch.tensor(int(heads), dtype=torch.long), persistent=True)
        self.register_buffer('early_expanded_width_config', torch.tensor(int(early_expanded_width), dtype=torch.long), persistent=True)

        architecture = str(read_attention_architecture)
        bridge_types = {
            "softmax": SoftmaxCrossOnlyReadBridge,
            "sigmoid_mass": SigmoidMassCrossOnlyReadBridge,
            "sigmoid_all": SigmoidAllCrossOnlyReadBridge,
        }
        if architecture not in bridge_types:
            raise ValueError(
                f"Unknown read_attention_architecture={architecture!r}; "
                f"expected one of {tuple(bridge_types)}"
            )
        self.read_attention_architecture = architecture
        self.read_null_enabled = bool(read_null_enabled)
        if self.read_null_enabled and architecture not in {"softmax", "sigmoid_all"}:
            raise ValueError("READ_NULL is supported by softmax and raw sigmoid_all PIECES attention")
        self.read_bridge = bridge_types[architecture](
            text_width=text_width,
            vision_width=vision_width,
            bridge_width=bridge_width,
            output_dim=output_dim,
            heads=heads,
        )
        content_pool_type = SigmoidAllVisualQueryPool if architecture == "sigmoid_all" else PreNormVisualQueryPool
        ortho_bridge_type = SigmoidAllEarlyOrthographicBridge if architecture == "sigmoid_all" else EarlyOrthographicBridge
        self.content_pool = content_pool_type(
            vision_width=vision_width,
            pool_width=bridge_width,
            output_dim=output_dim,
            heads=heads,
            zero_output=True,
        )
        self.orthographic_bridge = ortho_bridge_type(
            text_width=text_width,
            vision_width=vision_width,
            expanded_width=early_expanded_width,
            bridge_width=bridge_width,
            output_dim=output_dim,
            heads=heads,
        )
        self.source_head = EarlySourceGate(
            vision_width=vision_width,
            expanded_width=early_expanded_width,
            hidden_width=max(128, bridge_width),
        )
        self.trust_router = CandidateTrustRouter(hidden_width=128)

        self.read_tap_logits = nn.Parameter(torch.zeros(len(late)))
        self.content_tap_logits = nn.Parameter(torch.zeros(len(late)))
        self.ortho_tap_logits = nn.Parameter(torch.zeros(len(early)))
        self.source_tap_logits = nn.Parameter(torch.zeros(len(early)))

        self.glyph_bias_beta = nn.Parameter(torch.zeros(()))
        self.read_calibration_scale = nn.Parameter(torch.ones(()))
        self.null_abstain_weight = nn.Parameter(torch.zeros(()))
        self.auto_read_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer('read_probe', torch.zeros(output_dim), persistent=True)

    def reset_trust_router(self, auto_read_scale: float = 0.0) -> None:
        self.trust_router.reset_parameters()
        with torch.no_grad():
            self.auto_read_scale.fill_(float(auto_read_scale))

    def reset_orthographic_branch(self) -> None:
        self.orthographic_bridge.reset_parameters()
        with torch.no_grad():
            self.ortho_tap_logits.zero_()

    def reset_source_gate(self) -> None:
        self.source_head.reset_parameters()
        with torch.no_grad():
            self.source_tap_logits.zero_()

    @staticmethod
    def _replace_logits(old: nn.Parameter, count: int, reset_uniform: bool) -> nn.Parameter:
        if old.numel() != count:
            return nn.Parameter(torch.zeros(count, device=old.device, dtype=old.dtype))
        if reset_uniform:
            with torch.no_grad():
                old.zero_()
        return old

    @torch.no_grad()
    def set_tap_blocks(self, tap_blocks: Sequence[int], reset_uniform: bool = False) -> None:
        taps = tuple(dict.fromkeys(int(x) for x in tap_blocks))
        if not taps:
            raise ValueError('late tap blocks must be non-empty')
        self.tap_blocks = torch.tensor(taps, dtype=torch.long, device=self.tap_blocks.device)
        self.read_tap_logits = self._replace_logits(self.read_tap_logits, len(taps), reset_uniform)
        self.content_tap_logits = self._replace_logits(self.content_tap_logits, len(taps), reset_uniform)

    @torch.no_grad()
    def set_ortho_tap_blocks(self, tap_blocks: Sequence[int], reset_uniform: bool = False) -> None:
        taps = tuple(dict.fromkeys(int(x) for x in tap_blocks))
        if not taps:
            raise ValueError('orthographic tap blocks must be non-empty')
        self.ortho_tap_blocks = torch.tensor(taps, dtype=torch.long, device=self.ortho_tap_blocks.device)
        self.ortho_tap_logits = self._replace_logits(self.ortho_tap_logits, len(taps), reset_uniform)

    @torch.no_grad()
    def set_source_tap_blocks(self, tap_blocks: Sequence[int], reset_uniform: bool = False) -> None:
        taps = tuple(dict.fromkeys(int(x) for x in tap_blocks))
        if not taps:
            raise ValueError('source tap blocks must be non-empty')
        self.source_tap_blocks = torch.tensor(taps, dtype=torch.long, device=self.source_tap_blocks.device)
        self.source_tap_logits = self._replace_logits(self.source_tap_logits, len(taps), reset_uniform)

    @torch.no_grad()
    def set_read_probe(self, vector: Optional[torch.Tensor]) -> None:
        if vector is None:
            self.read_probe.zero_()
            return
        value = torch.as_tensor(vector, dtype=self.read_probe.dtype).reshape(-1)
        if value.numel() != self.read_probe.numel():
            raise ValueError(
                f'read probe width {value.numel()} != embedding width {self.read_probe.numel()}'
            )
        norm = value.norm()
        self.read_probe.copy_(value / norm if float(norm) > 0.0 else value)

    @_fp32_custom_forward
    def injection_feature(self, content_image_norm: torch.Tensor) -> torch.Tensor:
        probe = self.read_probe.to(device=content_image_norm.device, dtype=content_image_norm.dtype)
        if not bool(probe.abs().sum() > 0):
            return torch.zeros(content_image_norm.shape[0], device=content_image_norm.device, dtype=content_image_norm.dtype)
        return content_image_norm @ probe

    def tap_block_list(self) -> List[int]:
        return [int(x) for x in self.tap_blocks.detach().cpu().tolist()]

    def ortho_block_list(self) -> List[int]:
        return [int(x) for x in self.ortho_tap_blocks.detach().cpu().tolist()]

    def source_block_list(self) -> List[int]:
        return [int(x) for x in self.source_tap_blocks.detach().cpu().tolist()]

    def capture_block_list(self) -> List[int]:
        return sorted(set(self.tap_block_list() + self.ortho_block_list() + self.source_block_list()))

    @staticmethod
    def _weights(logits: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return logits.to(device=ref.device, dtype=torch.float32).softmax(dim=0)

    @staticmethod
    def _states(states: Dict[int, torch.Tensor], blocks: Sequence[int]) -> List[torch.Tensor]:
        missing = [b for b in blocks if b not in states]
        if missing:
            raise KeyError(f'Missing requested visual tap states for blocks: {missing}')
        return [states[b] for b in blocks]

    def _spatial_tap_state(self, state: torch.Tensor, block_idx: int) -> torch.Tensor:
        """Return CLS+spatial patches for branches that must not consume READ_NULL.

        READ_NULL is appended as the final visual token immediately before
        ``read_null_insert_block`` and remains the final token thereafter.  The
        SOURCE glyph map and early orthographic branch are spatial-patch
        branches, so post-insertion taps must drop that final token before their
        normal ``[:, 1:, :]`` patch slicing.  Pre-insertion taps are unchanged.

        This is the mixed-tap analogue of the existing late content-pool
        ``exclude_last_token`` handling.
        """
        if not self.read_null_enabled:
            return state
        insert_block = int(getattr(self, 'read_null_insert_block', 20))
        if int(block_idx) < insert_block:
            return state
        if state.ndim != 3 or int(state.shape[1]) < 3:
            raise ValueError(
                f'Invalid post-READ_NULL tap state at block {block_idx}: {tuple(state.shape)}'
            )
        return state[:, :-1, :]

    @_fp32_custom_forward
    def source_outputs(self, states: Dict[int, torch.Tensor], return_details: bool = False):
        blocks = self.source_block_list()
        ordered = self._states(states, blocks)
        weights = self._weights(self.source_tap_logits, ordered[0])
        spatial_states = [
            self._spatial_tap_state(state, block)
            for block, state in zip(blocks, ordered)
        ]
        per_block = [self.source_head.patch_logits(state) for state in spatial_states]
        mixed_patch_logits = torch.einsum('k,kbp->bp', weights, torch.stack(per_block, dim=0))
        source_logits, glyph_probs, stats = self.source_head.aggregate(mixed_patch_logits)
        if return_details:
            return source_logits, mixed_patch_logits, stats, {
                'tap_weights': weights,
                'per_block_logits': {b: x for b, x in zip(blocks, per_block)},
                'glyph_probs': glyph_probs,
            }
        return source_logits, mixed_patch_logits, stats

    @_fp32_custom_forward
    def presence_logits(self, states: Dict[int, torch.Tensor], return_details: bool = False):
        if return_details:
            source, _glyph, _stats, details = self.source_outputs(states, return_details=True)
            return source, details
        return self.source_outputs(states, return_details=False)[0]

    @_fp32_custom_forward
    def glyph_logits(self, states: Dict[int, torch.Tensor], return_details: bool = False):
        if return_details:
            _source, glyph, _stats, details = self.source_outputs(states, return_details=True)
            return glyph, details
        return self.source_outputs(states, return_details=False)[1]

    @_fp32_custom_forward
    def orthographic_features(
        self,
        states: Dict[int, torch.Tensor],
        text_query: torch.Tensor,
        return_details: bool = False,
    ):
        blocks = self.ortho_block_list()
        ordered = self._states(states, blocks)
        weights = self._weights(self.ortho_tap_logits, ordered[0])
        outputs = []
        attention = {}
        for block, state in zip(blocks, ordered):
            spatial_state = self._spatial_tap_state(state, block)
            if return_details:
                out, probs = self.orthographic_bridge(
                    text_query, spatial_state, return_attention=True
                )
                attention[block] = probs
            else:
                out = self.orthographic_bridge(text_query, spatial_state)
            outputs.append(out)
        mixed = torch.einsum('k,kbnd->bnd', weights, torch.stack(outputs, dim=0))
        if return_details:
            return mixed, {'tap_weights': weights, 'per_block_attention': attention}
        return mixed

    @_fp32_custom_forward
    def read_features(
        self,
        states: Dict[int, torch.Tensor],
        text_query: torch.Tensor,
        register_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
        return_evidence: bool = False,
    ):
        blocks = self.tap_block_list()
        ordered = self._states(states, blocks)
        weights = self._weights(self.read_tap_logits, ordered[0])
        glyph_probs = self.glyph_logits(states).sigmoid().detach()
        bridge_glyph_probs = glyph_probs
        bridge_register_mask = register_mask
        read_null_index = None
        if self.read_null_enabled:
            expected_tokens = int(glyph_probs.shape[1]) + 2  # CLS + spatial patches + READ_NULL
            bad_shapes = [tuple(state.shape) for state in ordered if int(state.shape[1]) != expected_tokens]
            if bad_shapes:
                raise RuntimeError(
                    "READ_NULL is enabled but late visual states do not contain exactly one "
                    f"extra token: expected T={expected_tokens}, got {bad_shapes}"
                )
            # The reader treats READ_NULL as a visual-evidence candidate with no
            # glyph penalty, but never as an implicit register.
            bridge_glyph_probs = torch.cat(
                [glyph_probs, torch.ones_like(glyph_probs[:, :1])], dim=1
            )
            if register_mask is not None:
                bridge_register_mask = torch.cat(
                    [register_mask, torch.zeros_like(register_mask[:, :1])], dim=1
                )
            read_null_index = expected_tokens - 1

        outputs = []
        evidence_outputs = []
        null_attention_outputs = []
        details = {}
        sigmoid_mass = self.read_attention_architecture == "sigmoid_mass"
        for block_idx, state in zip(blocks, ordered):
            if return_details or return_evidence or sigmoid_mass:
                out, attn = self.read_bridge(
                    text_query, state, register_mask=bridge_register_mask,
                    patch_textness=bridge_glyph_probs, textness_beta=self.glyph_bias_beta,
                    read_null_index=read_null_index,
                    return_attention=True,
                )
                if return_details:
                    details[block_idx] = attn
                if self.read_null_enabled:
                    null_attn = attn.get("read_null_attention")
                    if null_attn is None:
                        raise RuntimeError("READ_NULL attention diagnostics missing from softmax bridge")
                    null_attention_outputs.append(null_attn.mean(dim=-1))
                if sigmoid_mass:
                    evidence_outputs.append(attn["evidence"])
            else:
                out = self.read_bridge(
                    text_query, state, register_mask=bridge_register_mask,
                    patch_textness=bridge_glyph_probs, textness_beta=self.glyph_bias_beta,
                    read_null_index=read_null_index,
                )
            outputs.append(out)
        mixed = torch.einsum('k,kbnd->bnd', weights, torch.stack(outputs, dim=0))
        if sigmoid_mass:
            evidence = torch.einsum(
                'k,kbn->bn', weights, torch.stack(evidence_outputs, dim=0)
            ).clamp(min=0.0, max=1.0)
        else:
            evidence = torch.ones(
                mixed.shape[:2], dtype=mixed.dtype, device=mixed.device
            )
        if return_details:
            mixed_read_null_attention = None
            if self.read_null_enabled:
                mixed_read_null_attention = torch.einsum(
                    'k,kbn->bn', weights, torch.stack(null_attention_outputs, dim=0)
                ).clamp(min=0.0, max=1.0)
            detail_result = {
                'tap_weights': weights,
                'per_block': details,
                'glyph_probs': glyph_probs,
                'evidence': evidence,
                'read_attention_architecture': self.read_attention_architecture,
                'read_null_attention': mixed_read_null_attention,
                'read_null_index': read_null_index,
            }
            if return_evidence:
                return mixed, evidence, detail_result
            return mixed, detail_result
        if return_evidence:
            return mixed, evidence
        return mixed

    @_fp32_custom_forward
    def content_correction(self, states: Dict[int, torch.Tensor], return_details: bool = False):
        blocks = self.tap_block_list()
        ordered = self._states(states, blocks)
        weights = self._weights(self.content_tap_logits, ordered[0])
        outputs = []
        attention = {}
        for block_idx, state in zip(blocks, ordered):
            if return_details:
                out, probs = self.content_pool(
                    state, return_attention=True, exclude_last_token=self.read_null_enabled
                )
                attention[block_idx] = probs
            else:
                out = self.content_pool(
                    state, exclude_last_token=self.read_null_enabled
                )
            outputs.append(out)
        mixed = torch.einsum('k,kbd->bd', weights, torch.stack(outputs, dim=0))
        if return_details:
            return mixed, {'tap_weights': weights, 'per_block_attention': attention}
        return mixed

    @_fp32_custom_forward
    def calibrate_read_logits(
        self,
        raw_read_logits: torch.Tensor,
        source_logits: torch.Tensor,
        null_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.read_calibration_scale.to(raw_read_logits.dtype) * raw_read_logits
        if null_mask is not None and bool(null_mask.any()):
            unavailable = (-F.logsigmoid(source_logits[:, 1].detach())).to(raw_read_logits.dtype)
            bonus = self.null_abstain_weight.to(raw_read_logits.dtype) * unavailable[:, None]
            output = output + bonus * null_mask.to(raw_read_logits.dtype)[None, :]
        return output

    @staticmethod
    def positive_relative_read(relative: torch.Tensor) -> torch.Tensor:
        return 0.5 * (relative + torch.sqrt(relative.square() + 1.0e-4))

    @_fp32_custom_forward
    def trust_gate(self, **kwargs) -> torch.Tensor:
        return self.trust_router(**kwargs)


class VisualTransformer(nn.Module):
    def __init__(
        self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
        output_dim: int, read_null_enabled: bool = False, read_null_insert_block: int = 20,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.read_null_enabled = bool(read_null_enabled)
        self.read_null_insert_block = int(read_null_insert_block)
        if self.read_null_enabled:
            if not (0 <= self.read_null_insert_block < int(layers)):
                raise ValueError(
                    f"read_null_insert_block={self.read_null_insert_block} outside ViT layers={layers}"
                )
            self.read_null_token = nn.Parameter(torch.zeros(width))
            # Architecture metadata: this must survive state_dict/safetensors
            # export so a B13 checkpoint cannot silently reload at BXX old default.
            self.register_buffer(
                "read_null_insert_block_config",
                torch.tensor(self.read_null_insert_block, dtype=torch.long),
                persistent=True,
            )
        else:
            self.register_parameter("read_null_token", None)

    def _maybe_insert_read_null(self, block_idx: int, x_tbc: torch.Tensor) -> torch.Tensor:
        """Append trained READ_NULL immediately before its configured ViT block."""
        if not self.read_null_enabled or int(block_idx) != self.read_null_insert_block:
            return x_tbc
        null = self.read_null_token.to(dtype=x_tbc.dtype, device=x_tbc.device)
        null = null.view(1, 1, -1).expand(1, x_tbc.shape[1], -1)
        return torch.cat([x_tbc, null], dim=0)

    @property
    def read_null_index(self) -> Optional[int]:
        # READ_NULL is appended after all spatial patch tokens and stays last.
        if not self.read_null_enabled:
            return None
        return int(self.positional_embedding.shape[0])

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify, add CLS, add positional embedding, ln_pre, permute to [T,B,C]."""
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        return x.permute(1, 0, 2)  # [T, B, C]

    def _finalize_cls(self, x_tbc: torch.Tensor) -> torch.Tensor:
        """ln_post + proj on CLS token from [T,B,C]."""
        cls = self.ln_post(x_tbc.permute(1, 0, 2)[:, 0, :])
        if self.proj is not None:
            cls = cls @ self.proj
        return cls

    @torch.no_grad()
    def encode_image_cached(self, x: torch.Tensor):
        """
        Full forward with per-block z-cache.
        Returns (embedding [B, D], cache dict per block).
        """
        tokens = self._prepare_tokens(x)
        x_out, cache = self.transformer.forward_cached(
            tokens, before_block_hook=self._maybe_insert_read_null
        )
        return self._finalize_cls(x_out), cache

    @torch.no_grad()
    def encode_image_patched(self, x: torch.Tensor, patch_spec: Dict):
        """
        Forward with z-injection from patch_spec.
        patch_spec: {block_idx: {'inject_z': Tensor[B,H,T,D], 'inject_heads': list[int]}}
        Returns embedding [B, D].
        """
        tokens = self._prepare_tokens(x)
        x_out = self.transformer.forward_patched(
            tokens, patch_spec, before_block_hook=self._maybe_insert_read_null
        )
        return self._finalize_cls(x_out)

    def encode_image_atp(self, x: torch.Tensor):
        """
        Forward with retain_z_grad=True on all blocks (for AtP).
        Must be called under torch.enable_grad().
        Returns embedding [B, D]; grad nodes at blk.attn._last_z_grad_node per block.
        """
        tokens = self._prepare_tokens(x)
        x_out = self.transformer.forward_atp(
            tokens, before_block_hook=self._maybe_insert_read_null
        )
        return self._finalize_cls(x_out)

    @torch.no_grad()
    def encode_image_inject_capture(
        self,
        x: torch.Tensor,
        inject_spec: Dict,
        capture_blocks: set,
    ):
        """
        Forward with simultaneous z-injection and selective z-capture.
        Used for path patching: inject at source head, capture at circuit heads.
        Returns (embedding [B, D], cache {block: {'z': [B,H,T,D]}}).
        """
        tokens = self._prepare_tokens(x)
        x_out, cache = self.transformer.forward_inject_capture(
            tokens, inject_spec, capture_blocks, before_block_hook=self._maybe_insert_read_null
        )
        return self._finalize_cls(x_out), cache

    @torch.no_grad()
    def encode_image_cached_qkov(
        self, x: torch.Tensor, capture_blocks: set, cache_device=None
    ):
        """Capture attention probabilities and head-split values by block."""
        tokens = self._prepare_tokens(x)
        x_out, cache = self.transformer.forward_cached_qkov(
            tokens,
            capture_blocks,
            cache_device=cache_device,
            before_block_hook=self._maybe_insert_read_null,
        )
        return self._finalize_cls(x_out), cache

    @torch.no_grad()
    def encode_image_qkov_patched(self, x: torch.Tensor, patch_spec: Dict):
        """Patch attention probabilities and/or values by block and head."""
        tokens = self._prepare_tokens(x)
        x_out = self.transformer.forward_qkov_patched(
            tokens, patch_spec, before_block_hook=self._maybe_insert_read_null
        )
        return self._finalize_cls(x_out)

    @torch.no_grad()
    def encode_image_suppress_measure(
        self, x: torch.Tensor, suppress_spec: Dict, measure_blocks: set
    ):
        """Suppress source columns and capture downstream per-head z tensors."""
        tokens = self._prepare_tokens(x)
        x_out, zcache = self.transformer.forward_suppress_measure(
            tokens,
            suppress_spec,
            measure_blocks,
            before_block_hook=self._maybe_insert_read_null,
        )
        return self._finalize_cls(x_out), zcache

    @torch.no_grad()
    def encode_image_capture_residual(self, x: torch.Tensor, blocks):
        """Capture post-block residuals and token norms."""
        tokens = self._prepare_tokens(x)
        x_out, residuals, token_norm = self.transformer.forward_capture_residual(
            tokens, blocks, before_block_hook=self._maybe_insert_read_null
        )
        return self._finalize_cls(x_out), residuals, token_norm

    @torch.no_grad()
    def encode_image_project(self, x: torch.Tensor, spec: Dict):
        """Project masked residuals away from supplied orthonormal directions."""
        tokens = self._prepare_tokens(x)
        x_out = self.transformer.forward_project_residual(
            tokens, spec, before_block_hook=self._maybe_insert_read_null
        )
        return self._finalize_cls(x_out)

    @torch.no_grad()
    def encode_image_harvest_mtc(
        self, x: torch.Tensor, target_blocks, norm_block: int = -1
    ):
        """Return attention inputs, probabilities and z for MTC-style probes."""
        block_count = len(self.transformer.resblocks)
        norm_index = block_count + norm_block if norm_block < 0 else norm_block
        targets = set(int(block) for block in target_blocks)
        invalid = sorted(
            block for block in targets | {norm_index}
            if block < 0 or block >= block_count
        )
        if invalid:
            raise ValueError(f"Invalid visual block indices: {invalid}")
        if not targets:
            raise ValueError("target_blocks must contain at least one block index")

        output: Dict[int, Dict[str, torch.Tensor]] = {block: {} for block in targets}
        token_norm = None
        tokens = self._prepare_tokens(x)
        current = tokens
        deepest = max(max(targets), norm_index)
        for i, block in enumerate(self.transformer.resblocks):
            current = self._maybe_insert_read_null(i, current)
            capture = i in targets
            if capture:
                block.attn.last_xin = None
                block.attn.last_probs = None
                block.attn.last_z = None
            current = block(current, capture=capture)
            if i == norm_index:
                token_norm = current.detach().float().norm(dim=-1).permute(1, 0).contiguous()
            if capture:
                output[i]["xin"] = block.attn.last_xin.detach()
                output[i]["probs"] = block.attn.last_probs.detach()
                output[i]["z"] = block.attn.last_z.detach()
                for name in (
                    "last_q", "last_k", "last_v", "last_logits", "last_probs",
                    "last_z", "last_xin",
                ):
                    setattr(block.attn, name, None)
            if i >= deepest:
                break
        return output, token_norm

    @torch.no_grad()
    def _make_implicit_register_mask(
        self,
        patch_token_norms: torch.Tensor,   # [B, n_patches]
        register_threshold: float = 70.0,
        max_registers: Optional[int] = 4,
        min_registers: int = 1
    ) -> torch.Tensor:
        """Return ``[B, P]`` high-L2 spatial-token mask; True means register."""
        B, P = patch_token_norms.shape
        min_registers = int(min_registers)
        if min_registers < 0 or min_registers > P:
            raise ValueError(f"min_registers must be in [0, {P}], got {min_registers}")
        if max_registers is not None:
            max_registers = int(max_registers)
            if max_registers < min_registers or max_registers > P:
                raise ValueError(
                    f"max_registers must be in [{min_registers}, {P}], got {max_registers}"
                )
        mask = patch_token_norms > register_threshold

        if max_registers is not None:
            # cap registers per sample to max_registers by keeping highest norms among those above threshold
            out = torch.zeros_like(mask)
            for b in range(B):
                idx = torch.nonzero(mask[b], as_tuple=False).flatten()
                if idx.numel() == 0:
                    # fallback: top-min_registers by norm
                    topk = torch.topk(patch_token_norms[b], k=min_registers, largest=True).indices
                    out[b, topk] = True
                else:
                    k = min(max_registers, idx.numel())
                    topk = idx[torch.topk(patch_token_norms[b, idx], k=k, largest=True).indices]
                    out[b, topk] = True
            return out

        # ensure at least min_registers
        out = mask.clone()
        for b in range(B):
            if out[b].sum().item() < min_registers:
                topk = torch.topk(patch_token_norms[b], k=min_registers, largest=True).indices
                out[b, topk] = True
        return out

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        capture_blocks: Sequence[int],
        return_final_tokens: bool = True,
        register_threshold: float = 70.0,
        max_registers: Optional[int] = 8,
        min_registers: int = 1,
    ) -> Dict[str, Any]:
        x_tbc = self._prepare_tokens(x)
        x_tbc, states = self.transformer.forward_with_intermediates(
            x_tbc, capture_blocks, before_block_hook=self._maybe_insert_read_null
        )
        x_nld = x_tbc.permute(1, 0, 2)
        spatial = x_nld[:, 1:-1, :] if self.read_null_enabled else x_nld[:, 1:, :]
        patch_norms = spatial.float().norm(dim=-1)
        register_mask = self._make_implicit_register_mask(
            patch_norms.detach(),
            register_threshold=register_threshold,
            max_registers=max_registers,
            min_registers=min_registers,
        ).to(device=x_nld.device)
        return {
            'image_embedding': self._finalize_cls(x_tbc),
            'states': states,
            'register_mask': register_mask,
            'patch_token_norms': patch_norms,
            'final_tokens': x_nld if return_final_tokens else None,
            'read_null_index': self.read_null_index,
            'read_null_token_state': x_nld[:, -1, :] if self.read_null_enabled else None,
            'read_null_token_norm': (
                x_nld[:, -1, :].float().norm(dim=-1) if self.read_null_enabled else None
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        # trace mode
        return_trace: bool = False,
        register_threshold: float = 70.0,
        max_registers: Optional[int] = 8,
        min_registers: int = 1,
        cls_mask_includes_self: bool = True,
        return_tokens: bool = False,
        capture_layers: Optional[set] = None,
    ):
        """
        Default (return_trace=False): identical behavior to original: returns projected CLS embedding.

        return_trace=True: returns dict with:
          - image_embedding_full: [B, D]
          - image_embedding_patch_only: [B, D]
          - image_embedding_reg_only: [B, D]
          - register_mask: [B, n_patches] bool (implicit regs)
          - patch_token_norms: [B, n_patches] (pre-ln_post, post-last-block)
          - last_attn_probs: [B, H, T, S] (post-softmax) from last block
          - last_v: [B, H, S, head_dim] (value vectors) from last block
          - cls_skip_norm / cls_attn_norm (pre-MLP residual split diagnostics, last block)
          - optionally tokens_pre_ln_post: [B, 1+n_patches, width] if return_tokens=True
        """
        # patchify + add CLS
        x = self.conv1(x)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, n_patches]
        x = x.permute(0, 2, 1)  # [B, n_patches, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )  # [B, 1+n_patches, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # to LND
        x = x.permute(1, 0, 2)  # [T,B,C]
        last_idx = self.transformer.layers - 1

        if not return_trace:
            x = self.transformer(
                x, capture_layers=capture_layers, before_block_hook=self._maybe_insert_read_null
            )  # [T,B,C]
            x_bt = x.permute(1, 0, 2)                               # [B,T,C]
            cls = self.ln_post(x_bt[:, 0, :])

            if self.proj is not None:
                cls = cls @ self.proj

            if return_tokens:
                return {
                    "image_embedding_full": cls,
                    "tokens_pre_ln_post_full": x_bt,  # [B,T,C]
                }
            return cls

        # TRACE PATH
        # run up to (but excluding) last block
        x_pre_last = self.transformer.forward_until(
            x, last_idx, before_block_hook=self._maybe_insert_read_null
        )  # [T,B,C]
        x_pre_last = self._maybe_insert_read_null(last_idx, x_pre_last)

        # full last block with capture
        last_block = self.transformer.resblocks[last_idx]

        # for skip-vs-attn norm diagnostics: need pre-attn residual input for CLS
        ln1_full = last_block.ln_1(x_pre_last)
        attn_out_full, _ = last_block.attn(
            ln1_full, ln1_full, ln1_full,
            need_weights=True,
            attn_mask=None,  # vision blocks have no causal mask
            capture=True,
            cls_src_keep_mask=None,
            cls_mask_includes_self=cls_mask_includes_self
        )
        x_full = x_pre_last + attn_out_full
        x_full = x_full + last_block.mlp(last_block.ln_2(x_full))  # [T,B,C]

        # token-space for norms/mask
        tokens_full = x_full.permute(1, 0, 2)  # [B,T,C]
        spatial_full = tokens_full[:, 1:-1, :] if self.read_null_enabled else tokens_full[:, 1:, :]
        patch_token_norms = spatial_full.float().norm(dim=-1)  # [B, n_spatial_patches]

        register_mask = self._make_implicit_register_mask(
            patch_token_norms=patch_token_norms.detach(),
            register_threshold=register_threshold,
            max_registers=max_registers,
            min_registers=min_registers
        ).to(device=tokens_full.device)

        # Build CLS-row keep masks. READ_NULL is a distinct special token, not
        # an implicit register; keep it in BOTH recomputations so this diagnostic
        # continues to isolate patch-vs-register contribution rather than silently
        # turning into a READ_NULL ablation.
        B, n_patches = register_mask.shape
        src_len = 1 + n_patches + (1 if self.read_null_enabled else 0)

        keep_patch_only = torch.ones((B, src_len), dtype=torch.bool, device=tokens_full.device)
        keep_patch_only[:, 1:1 + n_patches] = ~register_mask

        keep_reg_only = torch.zeros((B, src_len), dtype=torch.bool, device=tokens_full.device)
        keep_reg_only[:, 0] = True
        keep_reg_only[:, 1:1 + n_patches] = register_mask
        if self.read_null_enabled:
            keep_reg_only[:, -1] = True

        # recompute last block with CLS-row masked attention (patch-only)
        ln1 = last_block.ln_1(x_pre_last)
        attn_out_patch, _ = last_block.attn(
            ln1, ln1, ln1,
            need_weights=False,
            attn_mask=None,
            capture=False,
            cls_src_keep_mask=keep_patch_only,
            cls_mask_includes_self=cls_mask_includes_self
        )
        x_patch = x_pre_last + attn_out_patch
        x_patch = x_patch + last_block.mlp(last_block.ln_2(x_patch))
        tokens_patch = x_patch.permute(1, 0, 2)

        # recompute last block with CLS-row masked attention (reg-only)
        attn_out_reg, _ = last_block.attn(
            ln1, ln1, ln1,
            need_weights=False,
            attn_mask=None,
            capture=False,
            cls_src_keep_mask=keep_reg_only,
            cls_mask_includes_self=cls_mask_includes_self
        )
        x_reg = x_pre_last + attn_out_reg
        x_reg = x_reg + last_block.mlp(last_block.ln_2(x_reg))
        tokens_reg = x_reg.permute(1, 0, 2)

        # final embeddings
        cls_full = self.ln_post(tokens_full[:, 0, :])
        cls_patch = self.ln_post(tokens_patch[:, 0, :])
        cls_reg = self.ln_post(tokens_reg[:, 0, :])

        if self.proj is not None:
            cls_full = cls_full @ self.proj
            cls_patch = cls_patch @ self.proj
            cls_reg = cls_reg @ self.proj

        # skip-vs-attn norms (last block, CLS only, pre-MLP)
        cls_skip = x_pre_last[0]          # [B,C]  (CLS token before attn residual add)
        cls_attn = attn_out_full[0]       # [B,C]  (CLS attention output)
        cls_skip_norm = cls_skip.norm(dim=-1)
        cls_attn_norm = cls_attn.norm(dim=-1)

        # last block captures (from MultiheadAttention)
        # NOTE: last_block.attn is MultiheadAttention; capture=True populated last_q/k/v/logits/probs
        last_attn_probs = last_block.attn.last_probs  # [B,H,T,S]
        last_q = last_block.attn.last_q               # [B,H,T,D]
        last_k = last_block.attn.last_k               # [B,H,S,D]
        last_v = last_block.attn.last_v               # [B,H,S,D]
        last_z = last_block.attn.last_z               # [B,H,T,D]
        last_xin = last_block.attn.last_xin           # [B,T,C]
        last_logits = last_block.attn.last_logits     # [B,H,T,S]

        out: Dict[str, Any] = {
            "image_embedding_full": cls_full,
            "image_embedding_patch_only": cls_patch,
            "image_embedding_reg_only": cls_reg,
            "register_mask": register_mask,
            "patch_token_norms": patch_token_norms,
            "last_attn_probs": last_attn_probs,
            "last_attn_logits": last_logits,
            "last_q": last_q,
            "last_k": last_k,
            "last_v": last_v,
            "last_z": last_z,
            "last_xin": last_xin,
            "cls_skip_norm": cls_skip_norm,
            "cls_attn_norm": cls_attn_norm,
            "read_null_index": self.read_null_index,
            "read_null_token_state": tokens_full[:, -1, :] if self.read_null_enabled else None,
            "read_null_token_norm": (
                tokens_full[:, -1, :].float().norm(dim=-1) if self.read_null_enabled else None
            ),
            "last_cls_to_read_null_attention": (
                last_attn_probs[:, :, 0, -1] if self.read_null_enabled else None
            ),
        }
        if return_tokens:
            out["tokens_pre_ln_post_full"] = tokens_full  # [B,T,C]
        return out


class CLIP(nn.Module):
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "use_positional_embedding_res"):
            self.use_positional_embedding_res = False
        if not hasattr(self, "read_attention_architecture"):
            self.read_attention_architecture = "softmax"
        if not hasattr(self, "read_null_enabled"):
            self.read_null_enabled = False
        if not hasattr(self, "read_null_insert_block"):
            self.read_null_insert_block = 20
        if not hasattr(self, "implant_kind"):
            self.implant_kind = (
                "full" if getattr(self, "read_implant", None) is not None else "none"
            )
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 use_positional_embedding_res: bool = False,     # <-- LongCLIP: internal-only switch
                 longclip_keep_len: int = 20,                    # <-- LongCLIP: Long-CLIP convention
                 hard_text_token_id: int = 49408,
                 no_text_token_id: int = 49409,
                 any_text_token_id: int = 49410,
                 null_text_token_id: int = 49411,
                 eot_token_id: int = 49407,
                 read_tap_blocks: Optional[Sequence[int]] = None,
                 ortho_tap_blocks: Optional[Sequence[int]] = None,
                 source_tap_blocks: Optional[Sequence[int]] = None,
                 early_expanded_width: int = 4096,
                 read_bridge_width: int = 256,
                 read_bridge_heads: int = 4,
                 read_attention_architecture: str = "softmax",
                 read_null_enabled: bool = False,
                 read_null_insert_block: int = 20,
                 implant_kind: str = "full",
                 ):
        super().__init__()

        self.context_length = context_length
        self.sot_token_id = 49406
        self.hard_text_token_id = int(hard_text_token_id)
        self.no_text_token_id = int(no_text_token_id)
        self.any_text_token_id = int(any_text_token_id)
        self.null_text_token_id = int(null_text_token_id)
        self.eot_token_id = int(eot_token_id)
        self.read_attention_architecture = str(read_attention_architecture)
        self.read_null_enabled = bool(read_null_enabled)
        self.read_null_insert_block = int(read_null_insert_block)
        self.implant_kind = str(implant_kind)
        if self.implant_kind not in {"none", "correction", "full"}:
            raise ValueError(
                f"implant_kind must be 'none', 'correction', or 'full'; got {self.implant_kind!r}"
            )
        self.use_positional_embedding_res = bool(use_positional_embedding_res)  # <-- LongCLIP
        self.longclip_keep_len = int(longclip_keep_len)                         # <-- LongCLIP

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            read_null_enabled=self.read_null_enabled,
            read_null_insert_block=self.read_null_insert_block,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        if self.implant_kind == "full":
            self.hard_text_embedding = nn.Parameter(torch.zeros(transformer_width))
            self.null_text_embedding = nn.Parameter(torch.zeros(transformer_width))
        else:
            self.register_parameter("hard_text_embedding", None)
            self.register_parameter("null_text_embedding", None)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

        if self.use_positional_embedding_res:                           # <-- LongCLIP
            self.positional_embedding_res = nn.Parameter(torch.empty(self.context_length, transformer_width))

            # masks are deterministic and should not be trained; keep them off state_dict if possible
            mask1 = torch.zeros(self.context_length, 1, dtype=torch.float32)
            keep_len = min(self.longclip_keep_len, self.context_length)
            mask1[:keep_len, :] = 1.0
            mask2 = 1.0 - mask1

            try:
                self.register_buffer("mask1", mask1, persistent=False)  # <-- LongCLIP
                self.register_buffer("mask2", mask2, persistent=False)  # <-- LongCLIP
            except TypeError:
                # older torch without persistent=
                self.register_buffer("mask1", mask1)                    # <-- LongCLIP
                self.register_buffer("mask2", mask2)                    # <-- LongCLIP    

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if read_tap_blocks is None:
            read_tap_blocks = (max(0, int(vision_layers) - 4), max(0, int(vision_layers) - 3))
        read_tap_blocks = tuple(dict.fromkeys(int(x) for x in read_tap_blocks))
        bad = [x for x in read_tap_blocks if x < 0 or x >= int(vision_layers)]
        if bad:
            raise ValueError(f'Invalid read_tap_blocks {bad} for {vision_layers} visual blocks')
        if self.implant_kind == "full":
            self.read_implant = HardTextReadImplant(
                text_width=transformer_width,
                vision_width=vision_width,
                output_dim=embed_dim,
                tap_blocks=read_tap_blocks,
                early_blocks=tuple(ortho_tap_blocks) if ortho_tap_blocks is not None else (tuple(x for x in (8, 12, 13) if x < int(vision_layers)) or (max(0, int(vision_layers) - 1),)),
                bridge_width=read_bridge_width,
                heads=read_bridge_heads,
                early_expanded_width=early_expanded_width,
                read_attention_architecture=self.read_attention_architecture,
                read_null_enabled=self.read_null_enabled,
            )
            self.read_implant.read_null_insert_block = int(self.read_null_insert_block)
            if source_tap_blocks is not None:
                self.read_implant.set_source_tap_blocks(source_tap_blocks, reset_uniform=True)
        elif self.implant_kind == "correction":
            self.read_implant = ContentCorrectionImplant(
                vision_width=vision_width,
                output_dim=embed_dim,
                tap_blocks=read_tap_blocks,
                pool_width=read_bridge_width,
                heads=read_bridge_heads,
                attention_architecture=self.read_attention_architecture,
                read_null_enabled=self.read_null_enabled,
            )
        else:
            self.read_implant = None

        self._clip_apply_content_correction_by_default = self.implant_kind in {
            "correction", "full"
        }

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if self.use_positional_embedding_res:                           # <-- LongCLIP
            nn.init.normal_(self.positional_embedding_res, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.q_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.k_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.v_proj.weight, std=attn_std)
            if block.attn.q_proj.bias is not None:
                nn.init.zeros_(block.attn.q_proj.bias)
            if block.attn.k_proj.bias is not None:
                nn.init.zeros_(block.attn.k_proj.bias)
            if block.attn.v_proj.bias is not None:
                nn.init.zeros_(block.attn.v_proj.bias)

            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            if block.attn.out_proj.bias is not None:
                nn.init.zeros_(block.attn.out_proj.bias)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            if block.mlp.c_fc.bias is not None:
                nn.init.zeros_(block.mlp.c_fc.bias)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            if block.mlp.c_proj.bias is not None:
                nn.init.zeros_(block.mlp.c_proj.bias)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def _eot_indices(self, text: torch.Tensor) -> torch.Tensor:
        matches = text.eq(self.eot_token_id)
        idx = matches.to(torch.int64).argmax(dim=-1)
        missing = ~matches.any(dim=-1)
        if missing.any():
            idx = idx.clone()
            idx[missing] = text[missing].argmax(dim=-1)
        return idx

    def parse_text_modes(self, text: torch.Tensor) -> torch.Tensor:
        """Return 0=any, 1=text, 2=notext and reject conflicting controls."""
        has_text = text.eq(self.hard_text_token_id).any(dim=-1)
        has_notext = text.eq(self.no_text_token_id).any(dim=-1)
        has_any = text.eq(self.any_text_token_id).any(dim=-1)
        conflicts = has_text.to(torch.int8) + has_notext.to(torch.int8) + has_any.to(torch.int8)
        if (conflicts > 1).any():
            rows = (conflicts > 1).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f'Conflicting <text>/<notext>/<any> controls in rows {rows}')
        modes = torch.zeros(text.shape[0], dtype=torch.long, device=text.device)
        modes[has_text] = 1
        modes[has_notext] = 2
        return modes

    def is_text_mode(self, text: torch.Tensor) -> torch.Tensor:
        return self.parse_text_modes(text).eq(1)

    def is_any_mode(self, text: torch.Tensor) -> torch.Tensor:
        return self.parse_text_modes(text).eq(0)

    def is_no_text_mode(self, text: torch.Tensor) -> torch.Tensor:
        return self.parse_text_modes(text).eq(2)

    def _compact_control_tokens(
        self,
        text: torch.Tensor,
        remove_ids: Sequence[int],
    ) -> torch.Tensor:
        """Remove routing IDs and left-compact through EOT without treating ID 0 as padding."""
        output = torch.zeros_like(text)
        remove = {int(x) for x in remove_ids}
        eot_indices = self._eot_indices(text)
        for row in range(text.shape[0]):
            tokens = text[row, : int(eot_indices[row].item()) + 1]
            kept = [int(token) for token in tokens.tolist() if int(token) not in remove]
            if not kept or kept[-1] != self.eot_token_id:
                kept.append(self.eot_token_id)
            kept = kept[: text.shape[1]]
            if kept[-1] != self.eot_token_id:
                kept[-1] = self.eot_token_id
            output[row, : len(kept)] = torch.tensor(kept, device=text.device, dtype=text.dtype)
        return output

    def _insert_hard_text_control(self, text: torch.Tensor) -> torch.Tensor:
        """Insert one literal <text> immediately after SOT, preserving EOT."""
        clean = self._compact_control_tokens(
            text,
            (self.hard_text_token_id, self.no_text_token_id, self.any_text_token_id),
        )
        output = torch.zeros_like(clean)
        eot_indices = self._eot_indices(clean)
        for row in range(clean.shape[0]):
            tokens = clean[row, : int(eot_indices[row].item()) + 1].tolist()
            if not tokens:
                tokens = [self.hard_text_token_id, self.eot_token_id]
            elif len(tokens) == 1:
                tokens = [tokens[0], self.hard_text_token_id, self.eot_token_id]
            else:
                tokens = [tokens[0], self.hard_text_token_id, *tokens[1:]]
            tokens = tokens[: clean.shape[1]]
            if tokens[-1] != self.eot_token_id:
                tokens[-1] = self.eot_token_id
            output[row, : len(tokens)] = torch.tensor(tokens, device=text.device, dtype=text.dtype)
        return output

    def prepare_mode_tokens(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.implant_kind != "full":
            raise RuntimeError("Text routing modes require a trained full PIECES checkpoint")
        modes = self.parse_text_modes(text)
        content_tokens = self._compact_control_tokens(
            text,
            (self.hard_text_token_id, self.no_text_token_id, self.any_text_token_id),
        )
        read_tokens = self._insert_hard_text_control(content_tokens)
        return {'modes': modes, 'content_tokens': content_tokens, 'read_tokens': read_tokens}

    @torch.no_grad()
    def set_hard_text_token_embedding(self, vector: torch.Tensor) -> None:
        if self.hard_text_embedding is None:
            raise RuntimeError("This checkpoint has no trained hard-text control embedding")
        if not (0 <= self.hard_text_token_id < self.token_embedding.num_embeddings):
            raise IndexError(
                f'hard_text_token_id={self.hard_text_token_id} is outside token embedding size '
                f'{self.token_embedding.num_embeddings}'
            )
        vector = torch.as_tensor(vector).reshape(-1)
        if vector.numel() != self.token_embedding.embedding_dim:
            raise ValueError(
                f'Expected hard token width {self.token_embedding.embedding_dim}, got {vector.numel()}'
            )
        self.hard_text_embedding.copy_(
            vector.to(device=self.hard_text_embedding.device, dtype=self.hard_text_embedding.dtype)
        )
        self.token_embedding.weight[self.hard_text_token_id].copy_(
            vector.to(device=self.token_embedding.weight.device, dtype=self.token_embedding.weight.dtype)
        )

    @torch.no_grad()
    def set_null_text_token_embedding(self, vector: torch.Tensor) -> None:
        if self.null_text_embedding is None:
            raise RuntimeError("This checkpoint has no trained null-text control embedding")
        if not (0 <= self.null_text_token_id < self.token_embedding.num_embeddings):
            raise IndexError(
                f'null_text_token_id={self.null_text_token_id} is outside token embedding size '
                f'{self.token_embedding.num_embeddings}'
            )
        vector = torch.as_tensor(vector).reshape(-1)
        if vector.numel() != self.token_embedding.embedding_dim:
            raise ValueError(
                f'Expected null token width {self.token_embedding.embedding_dim}, got {vector.numel()}'
            )
        self.null_text_embedding.copy_(
            vector.to(device=self.null_text_embedding.device, dtype=self.null_text_embedding.dtype)
        )
        self.token_embedding.weight[self.null_text_token_id].copy_(
            vector.to(device=self.token_embedding.weight.device, dtype=self.token_embedding.weight.dtype)
        )

    def is_null_candidate(self, text: torch.Tensor) -> torch.Tensor:
        return text.eq(self.null_text_token_id).any(dim=-1)

    def _null_read_tokens(self, device: torch.device) -> torch.Tensor:
        tokens = torch.zeros((1, self.context_length), dtype=torch.long, device=device)
        tokens[0, 0] = self.sot_token_id
        tokens[0, 1] = self.hard_text_token_id
        tokens[0, 2] = self.null_text_token_id
        tokens[0, 3] = self.eot_token_id
        return tokens

    def encode_image_base(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.type(self.dtype))

    def set_read_null_enabled(self, enabled: bool = True) -> None:
        """Synchronize READ_NULL state across CLIP, ViT, and the late implant.

        ``visual.read_null_enabled`` controls physical token insertion.  The
        top-level CLIP object and late implant also need the same state so that
        downstream pooling knows whether the final token is READ_NULL or a real
        spatial patch.  Keeping this operation in one method avoids structurally
        invalid RN-off/correction-on ablations.
        """
        enabled = bool(enabled)
        if enabled and getattr(self.visual, "read_null_token", None) is None:
            raise RuntimeError("This checkpoint has no trained READ_NULL token")
        self.read_null_enabled = enabled
        self.visual.read_null_enabled = enabled
        if self.read_implant is not None and hasattr(self.read_implant, "read_null_enabled"):
            self.read_implant.read_null_enabled = enabled

    def set_content_correction_enabled(self, enabled: bool = True) -> None:
        """Control whether ordinary ``encode_image`` applies trained correction."""
        if enabled and self.implant_kind == "none":
            raise RuntimeError("This checkpoint has no trained content correction")
        self._clip_apply_content_correction_by_default = bool(enabled)

    @_fp32_custom_forward
    def _content_image_from_info(
        self,
        image_info: Dict[str, Any],
        apply_content_correction: bool = True,
        return_details: bool = False,
    ):
        base_image = image_info['image_embedding']
        correction_details = None
        if apply_content_correction and self.read_implant is not None:
            if return_details:
                correction, correction_details = self.read_implant.content_correction(
                    image_info['states'], return_details=True
                )
            else:
                correction = self.read_implant.content_correction(image_info['states'])
        else:
            correction = torch.zeros_like(base_image)
        content_image = base_image + correction
        if return_details:
            return content_image, correction, correction_details
        return content_image

    def encode_image(
        self,
        image: torch.Tensor,
        return_trace: bool = False,
        register_threshold: float = 70.0,
        max_registers: Optional[int] = 8,
        min_registers: int = 1,
        cls_mask_includes_self: bool = True,
        return_tokens: bool = False,
        capture_layers: Optional[set] = None,
    ):
        if return_trace or return_tokens or capture_layers is not None:
            return self.visual(
                image.type(self.dtype),
                return_trace=return_trace,
                register_threshold=register_threshold,
                max_registers=max_registers,
                min_registers=min_registers,
                cls_mask_includes_self=cls_mask_includes_self,
                return_tokens=return_tokens,
                capture_layers=capture_layers,
            )
        apply_correction = bool(
            getattr(self, "_clip_apply_content_correction_by_default", True)
        )
        if (
            not apply_correction
            or self.read_implant is None
            or not hasattr(self.visual, 'forward_with_intermediates')
        ):
            return self.encode_image_base(image)
        image_info = self.encode_image_states(image, return_final_tokens=False)
        return self._content_image_from_info(image_info, apply_content_correction=True)

    def encode_image_cached(self, image: torch.Tensor):
        return self.visual.encode_image_cached(image.type(self.dtype))

    def encode_image_patched(self, image: torch.Tensor, patch_spec: Dict):
        return self.visual.encode_image_patched(image.type(self.dtype), patch_spec)

    def encode_image_atp(self, image: torch.Tensor):
        return self.visual.encode_image_atp(image.type(self.dtype))

    def encode_image_inject_capture(
        self, image: torch.Tensor, inject_spec: Dict, capture_blocks: set
    ):
        return self.visual.encode_image_inject_capture(
            image.type(self.dtype), inject_spec, capture_blocks
        )

    def encode_image_cached_qkov(
        self, image: torch.Tensor, capture_blocks: set, cache_device=None
    ):
        return self.visual.encode_image_cached_qkov(
            image.type(self.dtype), capture_blocks, cache_device=cache_device
        )

    def encode_image_qkov_patched(self, image: torch.Tensor, patch_spec: Dict):
        return self.visual.encode_image_qkov_patched(image.type(self.dtype), patch_spec)

    def encode_image_suppress_measure(
        self, image: torch.Tensor, suppress_spec: Dict, measure_blocks: set
    ):
        return self.visual.encode_image_suppress_measure(
            image.type(self.dtype), suppress_spec, measure_blocks
        )

    def encode_image_capture_residual(self, image: torch.Tensor, blocks):
        return self.visual.encode_image_capture_residual(image.type(self.dtype), blocks)

    def encode_image_project(self, image: torch.Tensor, spec: Dict):
        return self.visual.encode_image_project(image.type(self.dtype), spec)

    def encode_image_harvest_mtc(
        self, image: torch.Tensor, target_blocks, norm_block: int = -1
    ):
        return self.visual.encode_image_harvest_mtc(
            image.type(self.dtype), target_blocks, norm_block
        )

    def encode_image_states(
        self,
        image: torch.Tensor,
        tap_blocks: Optional[Sequence[int]] = None,
        return_final_tokens: bool = True,
    ) -> Dict[str, Any]:
        if self.read_implant is None or not hasattr(self.visual, 'forward_with_intermediates'):
            raise RuntimeError('Hard text implant requires a VisionTransformer backbone')
        taps = self.read_implant.capture_block_list() if tap_blocks is None else [int(x) for x in tap_blocks]
        return self.visual.forward_with_intermediates(
            image.type(self.dtype), taps, return_final_tokens=return_final_tokens
        )

    def _encode_text_hidden(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        if text.numel() and int(text.max()) >= self.token_embedding.num_embeddings:
            raise ValueError(
                "Token IDs exceed this checkpoint's vocabulary. PIECES control tokens "
                "require a trained full x-attention checkpoint."
            )
        x = self.token_embedding(text).type(self.dtype)
        hard_mask = text.eq(self.hard_text_token_id).unsqueeze(-1)
        if hard_mask.any():
            hard = self.hard_text_embedding.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
            x = torch.where(hard_mask, hard, x)
        null_mask = text.eq(self.null_text_token_id).unsqueeze(-1)
        if null_mask.any():
            null = self.null_text_embedding.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
            x = torch.where(null_mask, null, x)
        if self.use_positional_embedding_res:
            pos = self.positional_embedding.to(device=x.device, dtype=x.dtype)
            posr = self.positional_embedding_res.to(device=x.device, dtype=x.dtype)
            m1 = self.mask1.to(device=x.device, dtype=x.dtype)
            m2 = self.mask2.to(device=x.device, dtype=x.dtype)
            x = x + pos * m1 + posr * m2
        else:
            x = x + self.positional_embedding.to(device=x.device, dtype=x.dtype)

        hidden_pre_ln = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        hidden_post_ln = self.ln_final(hidden_pre_ln).type(self.dtype)
        eot_indices = self._eot_indices(text)
        batch = torch.arange(hidden_post_ln.shape[0], device=hidden_post_ln.device)
        feature = hidden_post_ln[batch, eot_indices] @ self.text_projection
        return {
            'text_embedding': feature,
            'hidden_pre_ln': hidden_pre_ln,
            'hidden_post_ln': hidden_post_ln,
            'eot_indices': eot_indices,
            'eot_hidden_pre_ln': hidden_pre_ln[batch, eot_indices],
            'read_mode_mask': text.eq(self.hard_text_token_id).any(dim=-1),
        }

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        # <any>/<notext> are routing controls, never lexical concepts.
        clean = self._compact_control_tokens(text, (self.no_text_token_id, self.any_text_token_id))
        return self._encode_text_hidden(clean)['text_embedding']

    def encode_text_states(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        clean = self._compact_control_tokens(text, (self.no_text_token_id, self.any_text_token_id))
        return self._encode_text_hidden(clean)

    def forward_modes(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        apply_content_correction: bool = True,
        return_details: bool = False,
    ):
        """Score mixed candidates with early source, early orthography, and late trust."""
        if self.implant_kind != "full" or self.read_implant is None:
            raise RuntimeError('Source-conditioned modes require a trained full PIECES checkpoint')

        prepared = self.prepare_mode_tokens(text)
        modes = prepared['modes']
        content_tokens = prepared['content_tokens']
        read_tokens = prepared['read_tokens']

        image_info = self.encode_image_states(image, return_final_tokens=return_details)
        content_image, correction, correction_details = self._content_image_from_info(
            image_info,
            apply_content_correction=apply_content_correction,
            return_details=True,
        )
        content_image_norm = _fp32_normalize(content_image)
        content_text_info = self._encode_text_hidden(content_tokens)
        content_text_norm = _fp32_normalize(content_text_info['text_embedding'])
        scale = self.logit_scale.float().exp()
        content_logits = _fp32_scaled_matmul(scale, content_image_norm, content_text_norm.t())
        logits = content_logits.clone()

        source_details = None
        if return_details:
            source_logits, glyph_logits, source_stats, source_details = self.read_implant.source_outputs(
                image_info['states'], return_details=True
            )
        else:
            source_logits, glyph_logits, source_stats = self.read_implant.source_outputs(
                image_info['states'], return_details=False
            )
        glyph_probs = glyph_logits.sigmoid()
        source_probs = source_logits.sigmoid()
        source_gate = source_probs[:, 0] * source_probs[:, 1]

        read_candidate_mask = ~modes.eq(2)
        read_candidate_indices = torch.nonzero(read_candidate_mask, as_tuple=False).flatten()
        full_raw_read_logits = torch.zeros_like(content_logits)
        full_read_logits = torch.zeros_like(content_logits)
        full_early_logits = torch.zeros_like(content_logits)
        full_trust_gate = torch.zeros_like(content_logits)
        full_route_gate = torch.zeros_like(content_logits)
        full_relative_read = torch.zeros_like(content_logits)
        full_auto_read_contribution = torch.zeros_like(content_logits)
        full_read_null_attention = torch.zeros_like(content_logits)
        null_query_read_null_attention = torch.zeros(
            image.shape[0], device=image.device, dtype=content_logits.dtype
        )
        null_read_logits = torch.zeros(image.shape[0], device=image.device, dtype=content_logits.dtype)
        read_details = None
        ortho_details = None
        read_feature_full = None
        read_text_embedding_full = None

        if read_candidate_mask.any():
            read_text_info = self._encode_text_hidden(read_tokens[read_candidate_mask])
            read_text_norm = _fp32_normalize(read_text_info['text_embedding'])
            query = read_text_info['eot_hidden_pre_ln']

            sigmoid_mass = self.read_attention_architecture == "sigmoid_mass"
            if return_details:
                if sigmoid_mass:
                    read_feature, read_evidence, read_details = self.read_implant.read_features(
                        image_info['states'], query,
                        register_mask=image_info['register_mask'],
                        return_details=True,
                        return_evidence=True,
                    )
                else:
                    read_feature, read_details = self.read_implant.read_features(
                        image_info['states'], query,
                        register_mask=image_info['register_mask'], return_details=True,
                    )
                    read_evidence = None
                ortho_feature, ortho_details = self.read_implant.orthographic_features(
                    image_info['states'], query, return_details=True
                )
                if self.read_null_enabled:
                    mixed_null = read_details.get('read_null_attention')
                    if mixed_null is None:
                        raise RuntimeError("READ_NULL enabled but reader returned no null attention")
                    full_read_null_attention[:, read_candidate_mask] = mixed_null.to(
                        full_read_null_attention.dtype
                    )
            else:
                if sigmoid_mass:
                    read_feature, read_evidence = self.read_implant.read_features(
                        image_info['states'], query,
                        register_mask=image_info['register_mask'],
                        return_evidence=True,
                    )
                else:
                    read_feature = self.read_implant.read_features(
                        image_info['states'], query, register_mask=image_info['register_mask']
                    )
                    read_evidence = None
                ortho_feature = self.read_implant.orthographic_features(
                    image_info['states'], query
                )

            read_feature_norm = (
                _normalize_sigmoid_read_direction(read_feature)
                if sigmoid_mass else _fp32_normalize(read_feature)
            )
            ortho_feature_norm = _fp32_normalize(ortho_feature)
            raw_read_logits = _fp32_scaled_einsum('bnd,nd->bn', scale, read_feature_norm, read_text_norm)
            if read_evidence is not None:
                raw_read_logits = raw_read_logits * read_evidence.to(raw_read_logits.dtype)
            early_logits = _fp32_scaled_einsum('bnd,nd->bn', scale, ortho_feature_norm, read_text_norm)
            candidate_null_mask = self.is_null_candidate(read_tokens[read_candidate_mask])
            read_logits = self.read_implant.calibrate_read_logits(
                raw_read_logits, source_logits, null_mask=candidate_null_mask
            )

            # Internal null baseline is always available, so <any> scores are
            # pairwise functions of image/candidate rather than candidate-bank composition.
            null_tokens = self._null_read_tokens(image.device)
            null_text_info = self._encode_text_hidden(null_tokens)
            null_text_norm = _fp32_normalize(null_text_info['text_embedding'])
            null_query = null_text_info['eot_hidden_pre_ln']
            if sigmoid_mass:
                null_feature, null_evidence = self.read_implant.read_features(
                    image_info['states'], null_query,
                    register_mask=image_info['register_mask'],
                    return_evidence=True,
                )
            else:
                if return_details and self.read_null_enabled:
                    null_feature, null_reader_details = self.read_implant.read_features(
                        image_info['states'], null_query,
                        register_mask=image_info['register_mask'], return_details=True
                    )
                    null_query_read_null_attention = null_reader_details[
                        'read_null_attention'
                    ][:, 0].to(content_logits.dtype)
                else:
                    null_feature = self.read_implant.read_features(
                        image_info['states'], null_query, register_mask=image_info['register_mask']
                    )
                null_evidence = None
            null_feature_norm = (
                _normalize_sigmoid_read_direction(null_feature[:, 0, :])
                if sigmoid_mass else _fp32_normalize(null_feature[:, 0, :])
            )
            raw_null = _fp32_scaled_einsum('bd,d->b', scale, null_feature_norm, null_text_norm[0])
            if null_evidence is not None:
                raw_null = raw_null * null_evidence[:, 0].to(raw_null.dtype)
            null_read_logits = self.read_implant.calibrate_read_logits(
                raw_null[:, None], source_logits,
                null_mask=torch.ones(1, dtype=torch.bool, device=image.device),
            )[:, 0]

            selected_content_logits = content_logits[:, read_candidate_mask]
            selected_content_text = content_text_norm[read_candidate_mask]
            trust_gate = self.read_implant.trust_gate(
                content_logits=selected_content_logits,
                read_logits=read_logits,
                null_logits=null_read_logits,
                early_logits=early_logits,
                content_image=content_image_norm,
                read_image=read_feature_norm,
                content_text=selected_content_text,
                read_text=read_text_norm,
                source_logits=source_logits,
                source_stats=source_stats,
                injection=self.read_implant.injection_feature(content_image_norm),
            )
            effective_gate = trust_gate * source_gate[:, None].detach()
            relative_read = read_logits - null_read_logits[:, None]

            selected_modes = modes[read_candidate_mask]
            forced_read = selected_modes.eq(1)
            auto = selected_modes.eq(0)
            selected_logits = selected_content_logits.clone()
            score_dtype = selected_logits.dtype
            selected_logits[:, forced_read] = read_logits[:, forced_read].to(score_dtype)
            if auto.any():
                positive_read = self.read_implant.positive_relative_read(
                    relative_read[:, auto]
                ).detach()
                # The early source head is intentionally evaluated in FP32 under
                # autocast, while CLIP similarities may be BF16/FP16.  Keep the
                # gate math stable, then cast the final additive score to the
                # destination similarity dtype before indexed assignment.
                auto_read_contribution = (
                    effective_gate[:, auto].to(positive_read.dtype)
                    * self.read_implant.auto_read_scale.to(positive_read.dtype)
                    * positive_read
                ).to(score_dtype)
                selected_logits[:, auto] = (
                    selected_content_logits[:, auto].to(score_dtype)
                    + auto_read_contribution
                )
                full_auto_read_contribution[:, read_candidate_indices[auto]] = (
                    auto_read_contribution.to(full_auto_read_contribution.dtype)
                )

            logits[:, read_candidate_mask] = selected_logits.to(logits.dtype)
            full_raw_read_logits[:, read_candidate_mask] = raw_read_logits.to(
                full_raw_read_logits.dtype
            )
            full_read_logits[:, read_candidate_mask] = read_logits.to(
                full_read_logits.dtype
            )
            full_early_logits[:, read_candidate_mask] = early_logits.to(
                full_early_logits.dtype
            )
            full_trust_gate[:, read_candidate_mask] = trust_gate.to(
                full_trust_gate.dtype
            )
            full_route_gate[:, read_candidate_mask] = effective_gate.to(
                full_route_gate.dtype
            )
            full_relative_read[:, read_candidate_mask] = relative_read.to(
                full_relative_read.dtype
            )
            if return_details:
                read_feature_full = read_feature
                read_text_embedding_full = read_text_info['text_embedding']

        if return_details:
            return {
                'logits_per_image': logits,
                'logits_per_text': logits.t(),
                'mode_ids': modes,
                'any_mode_mask': modes.eq(0),
                'read_mode_mask': modes.eq(1),
                'no_text_mode_mask': modes.eq(2),
                'content_tokens': content_tokens,
                'read_tokens': read_tokens,
                'base_image_embedding': image_info['image_embedding'],
                'content_image_embedding': content_image,
                'content_correction': correction,
                'content_text_embedding': content_text_info['text_embedding'],
                'raw_read_logits': full_raw_read_logits,
                'read_logits': full_read_logits,
                'null_read_logits': null_read_logits,
                'relative_read_logits': full_relative_read,
                'early_orthographic_logits': full_early_logits,
                'trust_gate': full_trust_gate,
                'source_gate': source_gate,
                'route_gate': full_route_gate,
                'auto_read_contribution': full_auto_read_contribution,
                'read_null_attention': full_read_null_attention,
                'null_query_read_null_attention': null_query_read_null_attention,
                'read_null_index': image_info.get('read_null_index'),
                'read_null_token_state': image_info.get('read_null_token_state'),
                'read_null_token_norm': image_info.get('read_null_token_norm'),
                'auto_read_scale': self.read_implant.auto_read_scale,
                'source_logits': source_logits,
                'source_stats': source_stats,
                # Compatibility aliases: channel 0=source present, channel 1=ordered/readable.
                'presence_logits': source_logits,
                'glyph_logits': glyph_logits,
                'glyph_probs': glyph_probs,
                'read_feature': read_feature_full,
                'read_text_embedding': read_text_embedding_full,
                'visual_states': image_info['states'],
                'visual_final_tokens': image_info.get('final_tokens'),
                'register_mask': image_info['register_mask'],
                'patch_token_norms': image_info['patch_token_norms'],
                'read_details': read_details,
                'orthographic_details': ortho_details,
                'content_details': correction_details,
                'source_details': source_details,
                'presence_details': source_details,
                'glyph_details': source_details,
            }
        return logits, logits.t()

    def forward_hard(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        apply_content_correction: bool = True,
        return_details: bool = False,
    ):
        """Backward-compatible legacy path: untagged=content, literal <text>=read."""
        if self.implant_kind != "full" or self.read_implant is None:
            raise RuntimeError('Hard text mode requires a trained full PIECES checkpoint')

        image_info = self.encode_image_states(image, return_final_tokens=return_details)
        text_info = self.encode_text_states(text)
        content_image, correction, correction_details = self._content_image_from_info(
            image_info,
            apply_content_correction=apply_content_correction,
            return_details=True,
        )
        image_norm = _fp32_normalize(content_image)
        text_norm = _fp32_normalize(text_info['text_embedding'])
        scale = self.logit_scale.float().exp()
        logits = _fp32_scaled_matmul(scale, image_norm, text_norm.t())

        read_mask = text_info['read_mode_mask']
        read_details = None
        if read_mask.any():
            query = text_info['eot_hidden_pre_ln'][read_mask]
            sigmoid_mass = self.read_attention_architecture == "sigmoid_mass"
            if return_details:
                if sigmoid_mass:
                    read_feature, read_evidence, read_details = self.read_implant.read_features(
                        image_info['states'], query,
                        register_mask=image_info['register_mask'],
                        return_details=True,
                        return_evidence=True,
                    )
                else:
                    read_feature, read_details = self.read_implant.read_features(
                        image_info['states'], query,
                        register_mask=image_info['register_mask'], return_details=True
                    )
                    read_evidence = None
            else:
                if sigmoid_mass:
                    read_feature, read_evidence = self.read_implant.read_features(
                        image_info['states'], query,
                        register_mask=image_info['register_mask'],
                        return_evidence=True,
                    )
                else:
                    read_feature = self.read_implant.read_features(
                        image_info['states'], query, register_mask=image_info['register_mask']
                    )
                    read_evidence = None
            read_feature = (
                _normalize_sigmoid_read_direction(read_feature)
                if sigmoid_mass else _fp32_normalize(read_feature)
            )
            read_text = text_norm[read_mask]
            raw_read_logits = _fp32_scaled_einsum('bnd,nd->bn', scale, read_feature, read_text)
            if read_evidence is not None:
                raw_read_logits = raw_read_logits * read_evidence.to(raw_read_logits.dtype)
            presence_logits_for_read = self.read_implant.presence_logits(image_info['states'])
            null_mask = self.is_null_candidate(text[read_mask])
            read_logits = self.read_implant.calibrate_read_logits(
                raw_read_logits, presence_logits_for_read, null_mask=null_mask
            )
            logits = logits.clone()
            logits[:, read_mask] = read_logits

        if return_details:
            presence_logits, presence_details = self.read_implant.presence_logits(
                image_info['states'], return_details=True
            )
            glyph_logits, glyph_details = self.read_implant.glyph_logits(
                image_info['states'], return_details=True
            )
            return {
                'logits_per_image': logits,
                'logits_per_text': logits.t(),
                'read_mode_mask': read_mask,
                'base_image_embedding': image_info['image_embedding'],
                'content_image_embedding': content_image,
                'content_correction': correction,
                'text_embedding': text_info['text_embedding'],
                'text_hidden_pre_ln': text_info['hidden_pre_ln'],
                'eot_indices': text_info['eot_indices'],
                'visual_states': image_info['states'],
                'visual_final_tokens': image_info.get('final_tokens'),
                'register_mask': image_info['register_mask'],
                'patch_token_norms': image_info['patch_token_norms'],
                'presence_logits': presence_logits,
                'glyph_logits': glyph_logits,
                'glyph_probs': glyph_logits.sigmoid(),
                'read_details': read_details,
                'content_details': correction_details,
                'presence_details': presence_details,
                'glyph_details': glyph_details,
            }
        return logits, logits.t()

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        if self.implant_kind != "full" or self.read_implant is None:
            image_features = _fp32_normalize(self.encode_image(image))
            text_features = _fp32_normalize(self.encode_text(text))
            logits = _fp32_scaled_matmul(
                self.logit_scale.float().exp(), image_features, text_features.t()
            )
            return logits, logits.t()
        return self.forward_modes(image, text, apply_content_correction=True, return_details=False)


def convert_weights(model: nn.Module):
    """Convert the native CLIP backbone to FP16 while keeping PIECES islands FP32.

    This intentionally mirrors OpenAI CLIP's mixed-precision policy for the
    backbone.  In particular, Conv/Linear layers must be converted together;
    converting only attention Q/K/V leaves ``visual.conv1`` in FP32 while the
    transformer projections are FP16.  Since ``CLIP.dtype`` is defined from
    ``visual.conv1.weight.dtype``, that mixed state feeds FP32 activations into
    FP16 attention and fails with ``mat1 and mat2 must have the same dtype``.

    Small trainable PIECES bridge/router/correction modules are restored to
    FP32 after the backbone conversion because their forwards explicitly run in
    FP32 islands.
    """
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, MultiheadAttention):
            for attr in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                module = getattr(l, attr, None)
                if module is not None and hasattr(module, "weight"):
                    module.weight.data = module.weight.data.half()
                    if module.bias is not None:
                        module.bias.data = module.bias.data.half()
            for attr in ["bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

    # The custom bridge/router/correction code uses explicit FP32 forward
    # islands. Keep its parameters in FP32 so those forwards remain dtype-safe
    # even when build_model()/clip.load() is used without the external loader.
    read_implant = getattr(model, "read_implant", None)
    if isinstance(read_implant, nn.Module):
        read_implant.float()

    for name in ("hard_text_embedding", "null_text_embedding"):
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            value.data = value.data.float()


def convert_state_dict_inproj_to_qkv(state_dict, prefix=''):
    """Convert in_proj_weight/in_proj_bias to q_proj/k_proj/v_proj in the given state_dict."""
    out = {}
    correction_only = _anytext_architecture_stage(state_dict) == "correction"
    correction_metadata = {
        "read_implant.tap_blocks",
        "read_implant.ortho_tap_blocks",
        "read_implant.source_tap_blocks",
        "read_implant.bridge_heads_config",
        "read_implant.early_expanded_width_config",
        "read_implant.read_probe",
    }
    for key, value in state_dict.items():
        if key in {
            "input_resolution",
            "context_length",
            "vocab_size",
        }:
            continue
        if correction_only and key in correction_metadata:
            continue
        if key.endswith('.attn.in_proj_weight'):
            D = value.shape[1]
            q = value[:D, :]
            k = value[D:2*D, :]
            v = value[2*D:, :]
            base = key[:-len('.in_proj_weight')]
            out[base + '.q_proj.weight'] = q
            out[base + '.k_proj.weight'] = k
            out[base + '.v_proj.weight'] = v
        elif key.endswith('.attn.in_proj_bias'):
            D = value.shape[0] // 3
            q = value[:D]
            k = value[D:2*D]
            v = value[2*D:]
            base = key[:-len('.in_proj_bias')]
            out[base + '.q_proj.bias'] = q
            out[base + '.k_proj.bias'] = k
            out[base + '.v_proj.bias'] = v
        else:
            out[key] = value
    return out



def _anytext_architecture_stage(state_dict: Mapping[str, torch.Tensor]) -> str:
    keys = tuple(map(str, state_dict.keys()))
    has_legacy = any(k.startswith("read_implant.presence_pool.") for k in keys)
    has_final = any(
        k.startswith((
            "read_implant.read_bridge.",
            "read_implant.source_head.",
            "read_implant.orthographic_bridge.",
            "read_implant.trust_router.",
        ))
        for k in keys
    )
    has_correction = any(
        k == "read_implant.content_tap_logits"
        or k.startswith("read_implant.content_pool.")
        for k in keys
    )
    if has_legacy and has_final:
        return "mixed-invalid"
    if has_legacy:
        return "legacy"
    if has_final:
        return "full"
    if has_correction:
        return "correction"
    return "none"

def build_model(
    state_dict: dict,
    hard_text_token_id: int = 49408,
    no_text_token_id: int = 49409,
    any_text_token_id: int = 49410,
    null_text_token_id: int = 49411,
    eot_token_id: int = 49407,
    read_attention_architecture: Optional[str] = None,
    read_null_enabled: Optional[bool] = None,
    read_null_insert_block: Optional[int] = None,
):
    state_dict = dict(state_dict)
    input_architecture_stage = _anytext_architecture_stage(state_dict)
    if input_architecture_stage == "mixed-invalid":
        raise RuntimeError(
            "Refusing a checkpoint containing both legacy and final PIECES parameters"
        )
    if input_architecture_stage == "legacy":
        raise RuntimeError(
            "Legacy PIECES checkpoints require their matching legacy model class. "
            "This module will not initialize an untrained final bridge around legacy weights."
        )
    implant_kind = input_architecture_stage

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    if vision_layers == 0:
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.transformer.resblocks") and k.endswith(".attn.q_proj.weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    checkpoint_vocab_size = state_dict["token_embedding.weight"].shape[0]
    vocab_size = int(checkpoint_vocab_size)
    if implant_kind == "full":
        required_vocab_size = max(
            int(hard_text_token_id), int(no_text_token_id),
            int(any_text_token_id), int(null_text_token_id),
        ) + 1
        if vocab_size < required_vocab_size:
            raise ValueError(
                f"Full PIECES checkpoint vocabulary has {vocab_size} rows but control "
                f"token IDs require at least {required_vocab_size}."
            )
        for key in ("hard_text_embedding", "null_text_embedding"):
            if key not in state_dict:
                raise KeyError(
                    f"Full PIECES checkpoint is missing trained parameter {key!r}; "
                    "refusing to initialize it."
                )
    elif any(key in state_dict for key in ("hard_text_embedding", "null_text_embedding")):
        raise RuntimeError(
            "Checkpoint contains control embeddings without a complete final PIECES bridge. "
            "Use its stage-specific model class."
        )

    checkpoint_bridge_keys = [
        key for key in state_dict if key.startswith("read_implant.read_bridge.")
    ]
    inferred_read_architecture = (
        "sigmoid_all"
        if (
            "read_implant.read_bridge.sigmoid_patch_head_bias" in state_dict
            or "read_implant.content_pool.sigmoid_head_bias" in state_dict
        )
        else (
            "sigmoid_mass"
            if "read_implant.read_bridge.sigmoid_head_bias" in state_dict
            else "softmax"
        )
    )
    if read_attention_architecture is None:
        selected_read_architecture = inferred_read_architecture
    else:
        selected_read_architecture = str(read_attention_architecture)
        if selected_read_architecture not in {"softmax", "sigmoid_mass", "sigmoid_all"}:
            raise ValueError(
                f"Unknown read_attention_architecture={selected_read_architecture!r}"
            )
        if (
            checkpoint_bridge_keys
            and selected_read_architecture != inferred_read_architecture
        ):
            raise ValueError(
                "Checkpoint read-attention architecture mismatch: "
                f"checkpoint={inferred_read_architecture!r}, "
                f"requested={selected_read_architecture!r}."
            )
        if (
            implant_kind == "correction"
            and inferred_read_architecture == "sigmoid_all"
            and selected_read_architecture != "sigmoid_all"
        ):
            raise ValueError(
                "Correction-pool architecture mismatch: checkpoint='sigmoid_all', "
                f"requested={selected_read_architecture!r}."
            )
        if (
            implant_kind == "correction"
            and selected_read_architecture == "sigmoid_all"
            and inferred_read_architecture != "sigmoid_all"
        ):
            raise ValueError(
                "Requested sigmoid_all correction but the trained sigmoid_head_bias is absent"
            )

    checkpoint_read_null_enabled = "visual.read_null_token" in state_dict
    checkpoint_read_null_insert = int(
        state_dict.get("visual.read_null_insert_block_config", torch.tensor(20)).item()
    )
    if read_null_enabled is None:
        selected_read_null_enabled = checkpoint_read_null_enabled
    else:
        selected_read_null_enabled = bool(read_null_enabled)
        if checkpoint_read_null_enabled and not selected_read_null_enabled:
            raise ValueError("Checkpoint contains READ_NULL but loader explicitly disabled it")
        if selected_read_null_enabled and not checkpoint_read_null_enabled:
            raise ValueError(
                "Loader requested READ_NULL but the checkpoint has no trained READ_NULL token"
            )
    selected_read_null_insert = (
        checkpoint_read_null_insert
        if read_null_insert_block is None
        else int(read_null_insert_block)
    )
    if checkpoint_read_null_enabled and selected_read_null_insert != checkpoint_read_null_insert:
        raise ValueError(
            "READ_NULL insertion block mismatch: "
            f"checkpoint={checkpoint_read_null_insert}, requested={selected_read_null_insert}"
        )
    if selected_read_null_enabled and selected_read_architecture not in {"softmax", "sigmoid_all"}:
        raise ValueError("READ_NULL is supported by softmax and raw sigmoid_all PIECES attention")

    if "read_implant.tap_blocks" in state_dict:
        read_tap_blocks = tuple(int(x) for x in state_dict["read_implant.tap_blocks"].tolist())
    else:
        read_tap_blocks = (max(0, int(vision_layers) - 4), max(0, int(vision_layers) - 3))
    if "read_implant.ortho_tap_blocks" in state_dict:
        ortho_tap_blocks = tuple(int(x) for x in state_dict["read_implant.ortho_tap_blocks"].tolist())
    else:
        ortho_tap_blocks = tuple(x for x in (8, 12, 13) if x < int(vision_layers))
    if "read_implant.source_tap_blocks" in state_dict:
        source_tap_blocks = tuple(int(x) for x in state_dict["read_implant.source_tap_blocks"].tolist())
    else:
        source_tap_blocks = ortho_tap_blocks
    early_expanded_width = int(
        state_dict.get("read_implant.early_expanded_width_config", torch.tensor(4096)).item()
    )
    if "read_implant.read_bridge.q_proj.weight" in state_dict:
        read_bridge_width = int(
            state_dict["read_implant.read_bridge.q_proj.weight"].shape[0]
        )
    elif "read_implant.content_pool.k_proj.weight" in state_dict:
        read_bridge_width = int(
            state_dict["read_implant.content_pool.k_proj.weight"].shape[0]
        )
    else:
        read_bridge_width = 256
    if "read_implant.bridge_heads_config" in state_dict:
        read_bridge_heads = int(state_dict["read_implant.bridge_heads_config"].item())
    elif "read_implant.content_pool.query" in state_dict:
        read_bridge_heads = int(state_dict["read_implant.content_pool.query"].shape[0])
    else:
        read_bridge_heads = 4
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    state_dict = convert_state_dict_inproj_to_qkv(state_dict)

    use_positional_embedding_res = ("positional_embedding_res" in state_dict)       # <--- Long-CLIP
    # -> sanity check when it IS present:
    if use_positional_embedding_res:
        pe = state_dict["positional_embedding"]
        per = state_dict["positional_embedding_res"]
        if pe.shape != per.shape:
            raise ValueError(
                f"positional_embedding_res shape mismatch: positional_embedding={tuple(pe.shape)} "
                f"vs positional_embedding_res={tuple(per.shape)}"
            )
    else:
        # If not using Long-CLIP, drop unexpected keys so strict loading works
        state_dict.pop("positional_embedding_res", None)

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        use_positional_embedding_res=use_positional_embedding_res,
        hard_text_token_id=hard_text_token_id,
        no_text_token_id=no_text_token_id,
        any_text_token_id=any_text_token_id,
        null_text_token_id=null_text_token_id,
        eot_token_id=eot_token_id,
        read_tap_blocks=read_tap_blocks,
        ortho_tap_blocks=ortho_tap_blocks,
        source_tap_blocks=source_tap_blocks,
        early_expanded_width=early_expanded_width,
        read_bridge_width=read_bridge_width,
        read_bridge_heads=read_bridge_heads,
        read_attention_architecture=selected_read_architecture,
        read_null_enabled=selected_read_null_enabled,
        read_null_insert_block=selected_read_null_insert,
        implant_kind=implant_kind,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)

    # Obsolete scalar aliases are not part of the final implementation.
    state_dict.pop("read_implant.readability_log_weight", None)
    state_dict.pop("read_implant.read_calibration_bias", None)

    if implant_kind == "correction":
        # Original full-module subsets may retain topology metadata that the
        # correction-only module keeps as ordinary Python attributes.
        for key in (
            "read_implant.tap_blocks",
            "read_implant.ortho_tap_blocks",
            "read_implant.source_tap_blocks",
            "read_implant.bridge_heads_config",
            "read_implant.early_expanded_width_config",
            "read_implant.read_probe",
        ):
            state_dict.pop(key, None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not exactly define the inferred architecture; refusing "
            f"untrained parameters. missing={missing}, unexpected={unexpected}"
        )
    built_architecture_stage = _anytext_architecture_stage(model.state_dict())
    if built_architecture_stage != implant_kind:
        raise RuntimeError(
            f"Builder produced architecture={built_architecture_stage!r}; "
            f"expected {implant_kind!r}."
        )
    model._anytext_build_info = {
        "architecture_stage": implant_kind,
        "input_architecture_stage": input_architecture_stage,
        "model_family": (
            "full_xattn" if implant_kind == "full"
            else "rn_correction" if implant_kind == "correction"
            else "rn_token" if selected_read_null_enabled
            else "vanilla"
        ),
        "read_attention_architecture": selected_read_architecture,
        "read_null_enabled": bool(selected_read_null_enabled),
        "read_null_insert_block": int(selected_read_null_insert),
    }
    return model.eval()
