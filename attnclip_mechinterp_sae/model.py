# model.py
# =============================================================================
# Add "trace" / "masked last-attn recompute" support for
# reproducing arXiv: 2505.05892v2 style experiments in CLIP:
#  - Return last-block attention weights (per-head), V/K/Q tensors, logits
#  - Compute patch-only / register-only CLS embeddings by masking *CLS row* in
#    the final self-attention (pre-softmax masking, renormalized)
#  - Provide token norms + derived implicit "register_mask" from high-norm patches
#  - Provide skip-vs-attn norm diagnostics (for the "skip dominance" analysis)
# =============================================================================
"""
Example: Call:

trace = model.encode_image(
    image_tensor,
    return_trace=True,
    register_threshold=70.0,
    max_registers=4,
    min_registers=1,
    cls_mask_includes_self=True,
    return_tokens=True,   # optional
)

Returns:
* `image_embedding_full`: standard CLIP image embedding
* `image_embedding_patch_only`: last-layer **CLS-row** attention masked to exclude implicit registers
* `image_embedding_reg_only`: last-layer **CLS-row** attention masked to include only implicit registers
* `register_mask`: your implicit register identification per image (high-norm patches, capped to 1–4 by default)
* `patch_token_norms`: norms used to define implicit registers
* `last_attn_probs`, `last_attn_logits`, `last_v`: what you need for attention-map faithfulness and “recompute” style probes
* `cls_skip_norm`, `cls_attn_norm`: last-block skip vs attention pathway magnitude
* optionally `tokens_pre_ln_post_full`: full token matrix pre-ln_post for additional diagnostics
"""

import os
from collections import OrderedDict
from typing import Tuple, Union, Optional, Dict, Any
import numpy as np
import warnings
import torch
from torch import Tensor
from torch import nn
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
        # attention input (== ln_1(resid)), head-agnostic source content [B, T, E]
        self.last_xin: Optional[torch.Tensor] = None
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
        # ── QK / OV split patching (Test A) ─────────────────────────────────────
        # Replace the post-softmax attention pattern (QK / "where") and/or the
        # value vectors (OV / "what") for selected heads, BEFORE the A@V product.
        # This is what z-injection (which replaces the *product*) cannot do: it
        # lets you donor-patch routing and content independently.
        inject_probs: Optional[torch.Tensor] = None,       # [B, H, T, S] donor attention pattern (post-softmax)
        inject_v: Optional[torch.Tensor] = None,           # [B, H, S, D] donor value vectors (head-split, pre-out_proj)
        inject_qkov_heads: Optional[list] = None,          # heads to QK/OV-patch; None = all H
        # ── scout-routing ablation (Test D) ─────────────────────────────────────
        # Zero the attention MASS that selected heads place on the given source
        # columns (e.g. text patches), then renormalize each affected query row so
        # the head still attends to a valid distribution over the REMAINING tokens.
        # This severs "this head reads from text patches" without injecting a donor
        # or corrupting unrelated routing. Applied to the post-softmax pattern.
        suppress_src_cols: Optional[torch.Tensor] = None,  # [B, S] bool: columns to zero (per image)
        suppress_heads: Optional[list] = None,             # heads to apply it to; None = all H
        suppress_query_rows: Optional[torch.Tensor] = None,# optional [T] bool: restrict to these tgt rows; None = all
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

        # ── RegCache: prepend clean pre-computed prefix K/V (post-projection, so the
        #    register never passes through the quantized activation path) ──────────
        _pk = getattr(self, "_prefix_k", None)
        if _pk is not None:
            _pv = self._prefix_v
            k_r = torch.cat([_pk.to(k_r.dtype).to(k_r.device), k_r], dim=1)
            v_r = torch.cat([_pv.to(v_r.dtype).to(v_r.device), v_r], dim=1)
        # ──────────────────────────────────────────────────────────────────────────

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

        # ── QK / OV split patching (Test A) ─────────────────────────────────────
        # Build the *effective* attention pattern / values used for the A@V product.
        # We keep the original attn_probs / v_r intact so the capture block below
        # still records the unpatched recipient tensors (capture + qkov-inject are
        # mutually exclusive in the intended workflow, but this keeps capture honest).
        _probs_eff = attn_probs                               # [B*H, T, S]
        _v_eff = v_r                                          # [B*H, S, D]
        if (inject_probs is not None) or (inject_v is not None):
            ap = attn_probs.view(N, num_heads, tgt_len, src_len)
            vv = v_r.view(N, num_heads, src_len, head_dim)
            if inject_probs is not None:
                ap = ap.clone()
                dp = inject_probs.to(dtype=ap.dtype, device=ap.device)
                if inject_qkov_heads is None:
                    ap[:] = dp
                else:
                    for _h in inject_qkov_heads:
                        ap[:, _h] = dp[:, _h]
            if inject_v is not None:
                vv = vv.clone()
                dv = inject_v.to(dtype=vv.dtype, device=vv.device)
                if inject_qkov_heads is None:
                    vv[:] = dv
                else:
                    for _h in inject_qkov_heads:
                        vv[:, _h] = dv[:, _h]
            _probs_eff = ap.reshape(N * num_heads, tgt_len, src_len)
            _v_eff = vv.reshape(N * num_heads, src_len, head_dim)
        # ─────────────────────────────────────────────────────────────────────────

        # ── scout-routing ablation: zero text columns, renormalize (Test D) ──────
        if suppress_src_cols is not None:
            ap2 = _probs_eff.view(N, num_heads, tgt_len, src_len).clone()
            col = suppress_src_cols.to(device=ap2.device).bool()        # [B, S]
            heads = suppress_heads if suppress_heads is not None else list(range(num_heads))
            keep = (~col).to(ap2.dtype)[:, None, None, :]               # [B,1,1,S]
            if suppress_query_rows is not None:
                qrow = suppress_query_rows.to(device=ap2.device).bool() # [T]
            else:
                qrow = torch.ones(tgt_len, dtype=torch.bool, device=ap2.device)
            for _h in heads:
                masked = ap2[:, _h] * keep[:, 0]                        # [B,T,S] zero suppressed cols
                denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                renorm = masked / denom                                 # renormalize over remaining
                # apply only to selected query rows; leave others intact
                sel = qrow[None, :, None].to(ap2.dtype)                 # [1,T,1]
                ap2[:, _h] = sel * renorm + (1.0 - sel) * ap2[:, _h]
            _probs_eff = ap2.reshape(N * num_heads, tgt_len, src_len)
        # ─────────────────────────────────────────────────────────────────────────

        # attention output: [B*H, T, D]
        attn_output = torch.bmm(_probs_eff, _v_eff)
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
            # query is [T,B,E] (== ln_1(resid), shared across heads) -> store [B,T,E]
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

    def forward(self, x: torch.Tensor, capture_layers: Optional[set] = None):
        for i, blk in enumerate(self.resblocks):
            cap = (capture_layers is not None) and (i in capture_layers)
            x = blk(x, capture=cap)  # <-- pass capture flag
        return x

    def forward_until(self, x: torch.Tensor, layer_idx_exclusive: int):
        for i in range(layer_idx_exclusive):
            x = self.resblocks[i](x)
        return x

    @torch.no_grad()
    def forward_cached(self, x: torch.Tensor):
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
            blk.attn.last_z = None
            cache[i] = {'resid_pre': x.detach().clone()}
            x = blk(x, capture=True)
            cache[i]['z'] = blk.attn.last_z   # [B, H, T, D] or None
        return x, cache

    def forward_patched(self, x: torch.Tensor, patch_spec: Dict):
        """
        Run all blocks; at blocks in patch_spec inject pre-computed z values.
        patch_spec: {block_idx: {'inject_z': Tensor[B,H,T,D], 'inject_heads': list[int]}}
        No grad — use for full activation patching sweeps.
        """
        for i, blk in enumerate(self.resblocks):
            spec = patch_spec.get(i, {})
            x = blk(
                x,
                inject_z=spec.get('inject_z', None),
                inject_heads=spec.get('inject_heads', None),
            )
        return x

    def forward_atp(self, x: torch.Tensor):
        """
        Run all blocks with retain_z_grad=True so AtP gradients are captured.
        Returns x_final; grad nodes accessible via blk.attn._last_z_grad_node.
        Must be called under torch.enable_grad().
        """
        for blk in self.resblocks:
            blk.attn._last_z_grad_node = None
            x = blk(x, retain_z_grad=True)
        return x

    @torch.no_grad()
    def forward_inject_capture(
        self,
        x: torch.Tensor,
        inject_spec: Dict,
        capture_blocks: set,
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

    # ── QK / OV split patching (Test A) ─────────────────────────────────────────
    @torch.no_grad()
    def forward_cached_qkov(self, x: torch.Tensor, capture_blocks: set, cache_device=None):
        """
        Run all blocks, capturing the post-softmax attention pattern (QK / 'where')
        and the head-split value vectors (OV / 'what') at the requested blocks.

        cache_device: if given (e.g. 'cpu'), the captured probs/v are moved there
        so the GPU never holds the full multi-block cache at once. They are moved
        back to the compute device automatically by the injection path. With 128 GB
        system RAM this is essentially free and removes the main Test-A memory cost.

        Returns (x_final [T,B,C], cache) where
          cache[block_idx] = {
            'probs': [B, H, T, S]   post-softmax attention (detached)
            'v':     [B, H, S, D]   value vectors, head-split, pre out_proj (detached)
          }
        These are the donor tensors fed back via forward_qkov_patched.
        """
        cache: Dict[int, Dict] = {}
        for i, blk in enumerate(self.resblocks):
            do_cap = i in capture_blocks
            if do_cap:
                blk.attn.last_probs = None
                blk.attn.last_v = None
            x = blk(x, capture=do_cap)
            if do_cap:
                # .clone() (or .to(cpu)) so we don't pin the big [B*H,T,S] base via a view,
                # then null all capture refs so the GPU storage is freed immediately.
                p = blk.attn.last_probs.detach()
                v = blk.attn.last_v.detach()
                if cache_device is not None:
                    p = p.to(cache_device); v = v.to(cache_device)
                else:
                    p = p.clone(); v = v.clone()
                cache[i] = {'probs': p, 'v': v}
                blk.attn.last_q = None
                blk.attn.last_k = None
                blk.attn.last_v = None
                blk.attn.last_logits = None
                blk.attn.last_probs = None
                blk.attn.last_z = None
        return x, cache

    @torch.no_grad()
    def forward_qkov_patched(self, x: torch.Tensor, patch_spec: Dict):
        """
        Run all blocks; at blocks in patch_spec, replace the attention pattern
        (QK) and/or the values (OV) for the listed heads with donor tensors.

        patch_spec: {block_idx: {
            'inject_probs':      Tensor[B,H,T,S] or None,   # donor 'where'
            'inject_v':          Tensor[B,H,S,D] or None,   # donor 'what'
            'inject_qkov_heads': list[int] or None,         # None = all heads
        }}

        Semantics (standard activation patching): the donor pattern/values are
        FROZEN (captured from the donor's clean forward). When patching multiple
        blocks at once, the value vectors that are *not* donor-patched are still
        recomputed live from the evolving (patched) residual, so cumulative
        QK patching is exact rather than an approximation.
        """
        for i, blk in enumerate(self.resblocks):
            spec = patch_spec.get(i, {})
            x = blk(
                x,
                inject_probs=spec.get('inject_probs', None),
                inject_v=spec.get('inject_v', None),
                inject_qkov_heads=spec.get('inject_qkov_heads', None),
            )
        return x
    # ────────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward_suppress_measure(self, x: torch.Tensor, suppress_spec: Dict,
                                 measure_blocks: set):
        """
        Scout-routing ablation + downstream measurement in one pass.

        suppress_spec: {block_idx: {
            'cols':  Tensor[B,S] bool   columns (e.g. text patches) to zero,
            'heads': list[int] or None  heads to apply it to,
            'rows':  Tensor[T] bool or None  restrict to these query rows,
        }}
        measure_blocks: blocks at which to capture the per-head pre-out_proj z
                        (to quantify the effect on the downstream text write).

        Returns (x_final [T,B,C], zcache {block: z[B,H,T,D] detached}).
        """
        zcache: Dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.resblocks):
            sp = suppress_spec.get(i, None)
            do_meas = i in measure_blocks
            if do_meas:
                blk.attn.last_z = None
            x = blk(
                x,
                capture=do_meas,
                suppress_src_cols=(sp['cols'] if sp else None),
                suppress_heads=(sp.get('heads') if sp else None),
                suppress_query_rows=(sp.get('rows') if sp else None),
            )
            if do_meas:
                zcache[i] = blk.attn.last_z.detach()
                blk.attn.last_z = None
                blk.attn.last_q = None; blk.attn.last_k = None; blk.attn.last_v = None
                blk.attn.last_logits = None; blk.attn.last_probs = None; blk.attn.last_xin = None
        return x, zcache

    @torch.no_grad()
    def forward_capture_residual(self, x: torch.Tensor, blocks):
        """Capture the RAW residual stream (between-block) at the given block boundaries,
        plus per-token L2 norm at the last requested block (for register-position masking).
        Returns (x_final [T,B,C], {block: resid [B,T,C] detached}, token_norm [B,T]).
        Raw residual is what forward_project_residual edits, so directions built from it
        live in the same space the intervention operates on."""
        want = set(int(b) for b in blocks)
        maxb = max(want) if want else -1
        out: Dict[int, torch.Tensor] = {}
        token_norm = None
        for i, blk in enumerate(self.resblocks):
            x = blk(x)
            if i in want:
                out[i] = x.detach().permute(1, 0, 2).contiguous()   # [B,T,C]
                if i == maxb:
                    token_norm = x.detach().norm(dim=-1).permute(1, 0).contiguous()  # [B,T]
            if i >= maxb and token_norm is not None:
                break
        return x, out, token_norm

    @torch.no_grad()
    def forward_project_residual(self, x: torch.Tensor, spec: Dict):
        """At each block in spec, project the residual at masked token positions onto the
        complement of an orthonormal direction set U (remove the U-component).
          spec[block] = {'mask': [B,T] bool, 'U': [r,D] (global) or [B,r,D] (per-image)}
        x' = x - (x @ U^T) @ U  applied only at mask positions. U MUST be orthonormal."""
        for i, blk in enumerate(self.resblocks):
            x = blk(x)
            s = spec.get(i, None)
            if s is None:
                continue
            U = s['U']; mask = s['mask']
            xb = x.permute(1, 0, 2)                          # [B,T,C]
            Uf = U.to(xb.dtype).to(xb.device)
            if Uf.dim() == 2:                                # global [r,D]
                coef = xb @ Uf.t()                           # [B,T,r]
                proj = coef @ Uf                             # [B,T,C]
            else:                                            # per-image [B,r,D]
                coef = torch.bmm(xb, Uf.transpose(1, 2))     # [B,T,r]
                proj = torch.bmm(coef, Uf)                   # [B,T,C]
            m = mask.to(xb.device).to(xb.dtype).unsqueeze(-1)  # [B,T,1]
            xb = xb - m * proj
            x = xb.permute(1, 0, 2).contiguous()
        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
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
        x_out, cache = self.transformer.forward_cached(tokens)
        return self._finalize_cls(x_out), cache

    @torch.no_grad()
    def encode_image_patched(self, x: torch.Tensor, patch_spec: Dict):
        """
        Forward with z-injection from patch_spec.
        patch_spec: {block_idx: {'inject_z': Tensor[B,H,T,D], 'inject_heads': list[int]}}
        Returns embedding [B, D].
        """
        tokens = self._prepare_tokens(x)
        x_out = self.transformer.forward_patched(tokens, patch_spec)
        return self._finalize_cls(x_out)

    def encode_image_atp(self, x: torch.Tensor):
        """
        Forward with retain_z_grad=True on all blocks (for AtP).
        Must be called under torch.enable_grad().
        Returns embedding [B, D]; grad nodes at blk.attn._last_z_grad_node per block.
        """
        tokens = self._prepare_tokens(x)
        x_out = self.transformer.forward_atp(tokens)
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
        x_out, cache = self.transformer.forward_inject_capture(tokens, inject_spec, capture_blocks)
        return self._finalize_cls(x_out), cache

    # ── QK / OV split patching (Test A) ─────────────────────────────────────────
    @torch.no_grad()
    def encode_image_cached_qkov(self, x: torch.Tensor, capture_blocks: set, cache_device=None):
        """Forward capturing per-block attention pattern (QK) and values (OV).
        Returns (embedding [B, D], cache {block: {'probs':[B,H,T,S], 'v':[B,H,S,D]}}).
        cache_device='cpu' offloads the cache to keep GPU residency minimal."""
        tokens = self._prepare_tokens(x)
        x_out, cache = self.transformer.forward_cached_qkov(tokens, capture_blocks, cache_device=cache_device)
        return self._finalize_cls(x_out), cache

    @torch.no_grad()
    def encode_image_qkov_patched(self, x: torch.Tensor, patch_spec: Dict):
        """Forward replacing attention pattern (QK) and/or values (OV) per head
        at the blocks in patch_spec. Returns embedding [B, D]."""
        tokens = self._prepare_tokens(x)
        x_out = self.transformer.forward_qkov_patched(tokens, patch_spec)
        return self._finalize_cls(x_out)

    @torch.no_grad()
    def encode_image_suppress_measure(self, x: torch.Tensor, suppress_spec: Dict, measure_blocks: set):
        """Scout-routing ablation + downstream z capture. Returns (embedding [B,D], zcache)."""
        tokens = self._prepare_tokens(x)
        x_out, zcache = self.transformer.forward_suppress_measure(tokens, suppress_spec, measure_blocks)
        return self._finalize_cls(x_out), zcache

    @torch.no_grad()
    def encode_image_capture_residual(self, x: torch.Tensor, blocks):
        """Capture raw residual at given blocks + token norm. Returns (emb [B,D], resid dict, token_norm)."""
        tokens = self._prepare_tokens(x)
        x_out, resid, tnorm = self.transformer.forward_capture_residual(tokens, blocks)
        return self._finalize_cls(x_out), resid, tnorm

    @torch.no_grad()
    def encode_image_project(self, x: torch.Tensor, spec: Dict):
        """Project register-position residuals onto complement of U at given blocks. Returns emb [B,D]."""
        tokens = self._prepare_tokens(x)
        x_out = self.transformer.forward_project_residual(tokens, spec)
        return self._finalize_cls(x_out)
    # ────────────────────────────────────────────────────────────────────────────

    # ── MTC head-locality harvest (Test C) ──────────────────────────────────────
    @torch.no_grad()
    def encode_image_harvest_mtc(self, x: torch.Tensor, target_blocks, norm_block: int = -1):
        """
        One forward; harvest the ingredients an attention-transcoder (MTC) needs to
        ask whether the text->CLS write is head-local or superposed across heads.

        For each block in target_blocks, returns:
          'xin'   : [B, T, E]      head-agnostic attention input (ln_1 of residual)
          'probs' : [B, H, T, S]   real post-softmax attention pattern (the 'where', given/frozen)
          'z'     : [B, H, T, D]   per-head pre-out_proj outputs (the 'what' actually written)
        Plus, once:
          'token_norm' : [B, T]    L2 residual norm at norm_block (for register-sink masking;
                                    cutoff ~60 isolates global sinks vs 30-50 for normal patches)

        Everything detached. The caller reduces to its target/source tokens and offloads.
        """
        n_blocks = len(self.transformer.resblocks)
        norm_idx = (n_blocks + norm_block) if norm_block < 0 else norm_block
        tgt = set(int(b) for b in target_blocks)
        max_needed = max(max(tgt), norm_idx)

        out: Dict[int, Dict[str, torch.Tensor]] = {b: {} for b in tgt}
        token_norm = None

        tokens = self._prepare_tokens(x)   # [T, B, C]
        xcur = tokens
        for i, blk in enumerate(self.transformer.resblocks):
            do_cap = i in tgt
            if do_cap:
                blk.attn.last_xin = None
                blk.attn.last_probs = None
                blk.attn.last_z = None
            xcur = blk(xcur, capture=do_cap)   # [T, B, C]
            if i == norm_idx:
                # residual norm per token at this block boundary: [T,B,C] -> [B,T]
                token_norm = xcur.norm(dim=-1).permute(1, 0).contiguous().detach()
            if do_cap:
                out[i]['xin'] = blk.attn.last_xin.detach()       # [B,T,E]
                out[i]['probs'] = blk.attn.last_probs.detach()   # [B,H,T,S]
                out[i]['z'] = blk.attn.last_z.detach()           # [B,H,T,D]
                blk.attn.last_xin = None
                blk.attn.last_probs = None
                blk.attn.last_z = None
                blk.attn.last_q = None
                blk.attn.last_k = None
                blk.attn.last_v = None
                blk.attn.last_logits = None
            if i >= max_needed:
                break   # early-exit: nothing past the deepest needed block matters
        return out, token_norm
    # ────────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _make_implicit_register_mask(
        self,
        patch_token_norms: torch.Tensor,   # [B, n_patches]
        register_threshold: float = 70.0,  # This is *only* the case for ViT-L!
        max_registers: Optional[int] = 4,
        min_registers: int = 1
    ) -> torch.Tensor:
        """
        Define implicit 'registers' as high-norm patch tokens.
        Returns bool mask [B, n_patches] where True = register.
        """
        B, P = patch_token_norms.shape
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
            x = self.transformer(x, capture_layers=capture_layers)  # [T,B,C]
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
        x_pre_last = self.transformer.forward_until(x, last_idx)  # [T,B,C]

        # full last block with capture
        last_block = self.transformer.resblocks[last_idx]

        # for skip-vs-attn norm diagnostics: need pre-attn residual input for CLS
        ln1_full = last_block.ln_1(x_pre_last)
        attn_out_full, attn_w_full = last_block.attn(
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
        patch_token_norms = tokens_full[:, 1:, :].norm(dim=-1)  # [B, n_patches]

        register_mask = self._make_implicit_register_mask(
            patch_token_norms=patch_token_norms.detach(),
            register_threshold=register_threshold,
            max_registers=max_registers,
            min_registers=min_registers
        ).to(device=tokens_full.device)

        # build CLS-row keep masks over src_len = 1 + n_patches
        B, n_patches = register_mask.shape
        src_len = 1 + n_patches

        # keep patches only: keep CLS (index 0) + non-register patches
        keep_patch_only = torch.ones((B, src_len), dtype=torch.bool, device=tokens_full.device)
        keep_patch_only[:, 1:] = ~register_mask

        # keep regs only: keep CLS + register patches
        keep_reg_only = torch.ones((B, src_len), dtype=torch.bool, device=tokens_full.device)
        keep_reg_only[:, 1:] = register_mask

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
        last_v = last_block.attn.last_v               # [B,H,S,D]
        last_logits = last_block.attn.last_logits     # [B,H,T,S]

        out: Dict[str, Any] = {
            "image_embedding_full": cls_full,
            "image_embedding_patch_only": cls_patch,
            "image_embedding_reg_only": cls_reg,
            "register_mask": register_mask,
            "patch_token_norms": patch_token_norms,
            "last_attn_probs": last_attn_probs,
            "last_attn_logits": last_logits,
            "last_v": last_v,
            "cls_skip_norm": cls_skip_norm,
            "cls_attn_norm": cls_attn_norm,
        }
        if return_tokens:
            out["tokens_pre_ln_post_full"] = tokens_full  # [B,T,C]
        return out


class CLIP(nn.Module):
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "use_positional_embedding_res"):
            self.use_positional_embedding_res = False
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
                 longclip_keep_len: int = 20                     # <-- LongCLIP: Long-CLIP convention
                 ):
        super().__init__()

        self.context_length = context_length
        self.use_positional_embedding_res = bool(use_positional_embedding_res)  # <-- LongCLIP
        self.longclip_keep_len = int(longclip_keep_len)                         # <-- LongCLIP

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
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

    # optional trace return for image encoder
    def encode_image(
        self,
        image,
        return_trace: bool = False,
        register_threshold: float = 70.0,
        max_registers: Optional[int] = 4,
        min_registers: int = 1,
        cls_mask_includes_self: bool = True,
        return_tokens: bool = False
    ):
        return self.visual(
            image.type(self.dtype),
            return_trace=return_trace,
            register_threshold=register_threshold,
            max_registers=max_registers,
            min_registers=min_registers,
            cls_mask_includes_self=cls_mask_includes_self,
            return_tokens=return_tokens
        )

    def encode_image_cached(self, image):
        """Cache all block z outputs. Returns (embedding, cache)."""
        return self.visual.encode_image_cached(image.type(self.dtype))

    def encode_image_patched(self, image, patch_spec: Dict):
        """Inject z from patch_spec during forward. Returns embedding."""
        return self.visual.encode_image_patched(image.type(self.dtype), patch_spec)

    def encode_image_atp(self, image):
        """Forward with grad-retained z nodes for AtP. Returns embedding."""
        return self.visual.encode_image_atp(image.type(self.dtype))

    def encode_image_inject_capture(self, image, inject_spec: Dict, capture_blocks: set):
        """Inject z at source heads, capture z at circuit heads. For path patching."""
        return self.visual.encode_image_inject_capture(
            image.type(self.dtype), inject_spec, capture_blocks
        )

    def encode_image_cached_qkov(self, image, capture_blocks: set, cache_device=None):
        """Capture per-block attention pattern (QK) and values (OV). For Test A donor pass."""
        return self.visual.encode_image_cached_qkov(image.type(self.dtype), capture_blocks, cache_device=cache_device)

    def encode_image_qkov_patched(self, image, patch_spec: Dict):
        """Replace attention pattern (QK) and/or values (OV) per head at given blocks. For Test A."""
        return self.visual.encode_image_qkov_patched(image.type(self.dtype), patch_spec)

    def encode_image_suppress_measure(self, image, suppress_spec: Dict, measure_blocks: set):
        """Scout-routing ablation (zero+renorm text columns for selected early heads) + downstream
        z capture at measure_blocks. Returns (embedding [B,D], zcache). For Test D."""
        return self.visual.encode_image_suppress_measure(image.type(self.dtype), suppress_spec, measure_blocks)

    def encode_image_capture_residual(self, image, blocks):
        """Capture raw residual + token norm at given blocks. Returns (emb, resid dict, token_norm). For Test G."""
        return self.visual.encode_image_capture_residual(image.type(self.dtype), blocks)

    def encode_image_project(self, image, spec: Dict):
        """Project register-position residuals onto complement of U at given blocks. For Test G."""
        return self.visual.encode_image_project(image.type(self.dtype), spec)

    def encode_image_harvest_mtc(self, image, target_blocks, norm_block: int = -1):
        """Harvest xin/probs/z at target_blocks + residual norm at norm_block. For Test C MTC."""
        return self.visual.encode_image_harvest_mtc(image.type(self.dtype), target_blocks, norm_block)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        if self.use_positional_embedding_res:               # <-- LongCLIP: Long-CLIP add (pos * mask1 + pos_res * mask2)
            pos = self.positional_embedding.to(device=x.device, dtype=x.dtype)
            posr = self.positional_embedding_res.to(device=x.device, dtype=x.dtype)
            m1 = self.mask1.to(device=x.device, dtype=x.dtype)
            m2 = self.mask2.to(device=x.device, dtype=x.dtype)
            x = x + pos * m1 + posr * m2
        else:
            x = x + self.positional_embedding.to(device=x.device, dtype=x.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image, return_trace=False)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
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


def convert_state_dict_inproj_to_qkv(state_dict, prefix=''):
    """Convert in_proj_weight/in_proj_bias to q_proj/k_proj/v_proj in the given state_dict."""
    out = {}
    for key, value in state_dict.items():
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


def build_model(state_dict: dict):
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
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
        use_positional_embedding_res=use_positional_embedding_res
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()