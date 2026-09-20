from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence, Mapping

import numpy as np
import math
import torch
import torch.nn.functional as F
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



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        # Optional diagnostics used only on sparse logging batches for READ_NULL.
        self.capture_read_null_attention = False
        self._read_null_attention_stats = None

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if self.capture_read_null_attention:
            out, weights = self.attn(
                x, x, x, need_weights=True, average_attn_weights=False, attn_mask=self.attn_mask
            )
            with torch.no_grad():
                w = weights.detach().float()
                if w.shape[-1] >= 3:
                    self._read_null_attention_stats = {
                        "cls_to_null": w[:, :, 0, -1].mean(),
                        "patch_to_null": w[:, :, 1:-1, -1].mean(),
                        "null_to_patches": w[:, :, -1, 1:-1].sum(dim=-1).mean(),
                        "null_to_cls": w[:, :, -1, 0].mean(),
                        "null_to_self": w[:, :, -1, -1].mean(),
                    }
                else:
                    self._read_null_attention_stats = None
            return out
        self._read_null_attention_stats = None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        capture_blocks: Sequence[int],
    ):
        capture = set(int(i) for i in capture_blocks)
        states: Dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.resblocks):
            x = block(x)
            if i in capture:
                states[i] = x.permute(1, 0, 2)  # [B,T,C], retain graph
        return x, states


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
        source_hidden_width: Optional[int] = None,
        trust_hidden_width: int = 128,
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
        selected_source_hidden_width = (
            max(128, bridge_width)
            if source_hidden_width is None
            else int(source_hidden_width)
        )
        self.source_head = EarlySourceGate(
            vision_width=vision_width,
            expanded_width=early_expanded_width,
            hidden_width=selected_source_hidden_width,
        )
        self.trust_router = CandidateTrustRouter(hidden_width=int(trust_hidden_width))

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


class VisionTransformer(nn.Module):
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
            # Zero is deliberately minimally invasive. Pre-LN attention can still
            # contextualize it immediately, while the parameter learns during 1B.5.
            self.read_null_token = nn.Parameter(torch.zeros(width))
            self.register_buffer(
                "read_null_insert_block_config",
                torch.tensor(self.read_null_insert_block, dtype=torch.long),
                persistent=True,
            )
        else:
            self.register_parameter("read_null_token", None)

        self.register_norm_threshold: float = 60.0
        self.register_min: int = 1
        self.register_max: int = 0
        self._last_register_mask: Optional[torch.Tensor] = None
        self._last_patch_l2_norms: Optional[torch.Tensor] = None

    def set_read_null_attention_capture(self, enabled: bool) -> None:
        """Capture sparse self-attention diagnostics on the late READ_NULL blocks."""
        enabled = bool(enabled) and self.read_null_enabled
        capture_blocks = {self.read_null_insert_block, self.read_null_insert_block + 1}
        for i, block in enumerate(self.transformer.resblocks):
            active = enabled and i in capture_blocks
            block.capture_read_null_attention = active
            if not active:
                block._read_null_attention_stats = None

    def _read_null_attention_diagnostics(self) -> Dict[int, Dict[str, torch.Tensor]]:
        result: Dict[int, Dict[str, torch.Tensor]] = {}
        if not self.read_null_enabled:
            return result
        for i, block in enumerate(self.transformer.resblocks):
            stats = getattr(block, "_read_null_attention_stats", None)
            if isinstance(stats, dict):
                result[int(i)] = stats
        return result

    def set_register_mask_config(self, threshold: float = 60.0, min_registers: int = 1, max_registers: int = 0) -> None:
        self.register_norm_threshold = float(threshold)
        self.register_min = int(min_registers)
        self.register_max = int(max_registers)

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x,
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        return x.permute(1, 0, 2)  # [T,B,C]

    def _finalize_cls(self, x_nld: torch.Tensor) -> torch.Tensor:
        cls = self.ln_post(x_nld[:, 0, :])
        if self.proj is not None:
            cls = cls @ self.proj
        return cls

    def _compute_register_mask_from_final_tokens(self, x_nld: torch.Tensor) -> torch.Tensor:
        spatial = x_nld[:, 1:-1, :] if self.read_null_enabled else x_nld[:, 1:, :]
        patch_norms = spatial.detach().float().norm(dim=-1)
        B, P = patch_norms.shape
        mask = torch.zeros((B, P), dtype=torch.bool, device=x_nld.device)
        min_regs = max(0, int(self.register_min))
        max_regs = int(self.register_max)
        for b in range(B):
            idx = torch.nonzero(
                patch_norms[b] > float(self.register_norm_threshold), as_tuple=False
            ).flatten()
            if idx.numel() < min_regs:
                k = min(max(1, min_regs), P)
                idx = torch.topk(patch_norms[b], k=k, largest=True).indices
            elif max_regs > 0 and idx.numel() > max_regs:
                idx = idx[torch.topk(patch_norms[b, idx], k=max_regs, largest=True).indices]
            mask[b, idx] = True
        self._last_patch_l2_norms = patch_norms
        self._last_register_mask = mask
        return mask

    def get_last_register_mask(self) -> Optional[torch.Tensor]:
        return self._last_register_mask

    def get_last_patch_l2_norms(self) -> Optional[torch.Tensor]:
        return self._last_patch_l2_norms

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        capture_blocks: Sequence[int],
        return_final_tokens: bool = True,
    ):
        x_tbc = self._prepare_tokens(x)
        if self.read_null_enabled:
            capture = set(int(i) for i in capture_blocks)
            states: Dict[int, torch.Tensor] = {}
            inserted = False
            for i, block in enumerate(self.transformer.resblocks):
                if i == self.read_null_insert_block:
                    null = self.read_null_token.to(dtype=x_tbc.dtype, device=x_tbc.device)
                    null = null.view(1, 1, -1).expand(1, x_tbc.shape[1], -1)
                    x_tbc = torch.cat([x_tbc, null], dim=0)
                    inserted = True
                x_tbc = block(x_tbc)
                if i in capture:
                    states[i] = x_tbc.permute(1, 0, 2)
            if not inserted:
                raise RuntimeError("READ_NULL token was never inserted into the ViT")
        else:
            x_tbc, states = self.transformer.forward_with_intermediates(x_tbc, capture_blocks)
        x_nld = x_tbc.permute(1, 0, 2)
        register_mask = self._compute_register_mask_from_final_tokens(x_nld)
        embedding = self._finalize_cls(x_nld)
        return {
            'image_embedding': embedding,
            'states': states,
            'register_mask': register_mask,
            'patch_token_norms': self._last_patch_l2_norms,
            'final_tokens': x_nld if return_final_tokens else None,
            'read_null_index': (x_nld.shape[1] - 1) if self.read_null_enabled else None,
            'read_null_vit_attention': self._read_null_attention_diagnostics(),
        }

    def forward(self, x: torch.Tensor):
        x_tbc = self._prepare_tokens(x)
        if self.read_null_enabled:
            inserted = False
            for i, block in enumerate(self.transformer.resblocks):
                if i == self.read_null_insert_block:
                    null = self.read_null_token.to(dtype=x_tbc.dtype, device=x_tbc.device)
                    null = null.view(1, 1, -1).expand(1, x_tbc.shape[1], -1)
                    x_tbc = torch.cat([x_tbc, null], dim=0)
                    inserted = True
                x_tbc = block(x_tbc)
            if not inserted:
                raise RuntimeError("READ_NULL token was never inserted into the ViT")
        else:
            x_tbc = self.transformer(x_tbc)
        x_nld = x_tbc.permute(1, 0, 2)
        self._compute_register_mask_from_final_tokens(x_nld)
        return self._finalize_cls(x_nld)


class CLIP(nn.Module):
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "use_positional_embedding_res"):
            self.use_positional_embedding_res = False
        if not hasattr(self, "read_attention_architecture"):
            self.read_attention_architecture = "softmax"
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
                 use_positional_embedding_res: bool = False,                    # <-- LongCLIP: internal-only switch
                 longclip_keep_len: int = 20,                                   # <-- LongCLIP: Long-CLIP convention
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
                 source_hidden_width: Optional[int] = None,
                 trust_hidden_width: int = 128,
                 read_attention_architecture: str = "softmax",
                 read_null_enabled: bool = False,
                 read_null_insert_block: int = 20,
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
        self.use_positional_embedding_res = bool(use_positional_embedding_res)  # <-- LongCLIP
        self.longclip_keep_len = int(longclip_keep_len)                         # <-- LongCLIP

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
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
        # Dedicated trainable embedding for the literal hard <text> ID.
        # It replaces the lookup result only at that ID, so the frozen base
        # token_embedding matrix never needs gradient masking.
        self.hard_text_embedding = nn.Parameter(torch.zeros(transformer_width))
        self.null_text_embedding = nn.Parameter(torch.zeros(transformer_width))

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

        if isinstance(self.visual, VisionTransformer):
            if read_tap_blocks is None:
                # ViT-L/14: outputs of B20 and B21 = inputs to B21/B22.
                read_tap_blocks = (max(0, int(vision_layers) - 4), max(0, int(vision_layers) - 3))
            read_tap_blocks = tuple(dict.fromkeys(int(x) for x in read_tap_blocks))
            bad = [x for x in read_tap_blocks if x < 0 or x >= int(vision_layers)]
            if bad:
                raise ValueError(f'Invalid read_tap_blocks {bad} for {vision_layers} visual blocks')
            self.read_implant = HardTextReadImplant(
                text_width=transformer_width,
                vision_width=vision_width,
                output_dim=embed_dim,
                tap_blocks=read_tap_blocks,
                early_blocks=tuple(ortho_tap_blocks) if ortho_tap_blocks is not None else (tuple(x for x in (8, 12, 13) if x < int(vision_layers)) or (max(0, int(vision_layers) - 1),)),
                bridge_width=read_bridge_width,
                heads=read_bridge_heads,
                early_expanded_width=early_expanded_width,
                source_hidden_width=source_hidden_width,
                trust_hidden_width=trust_hidden_width,
                read_attention_architecture=self.read_attention_architecture,
                read_null_enabled=self.read_null_enabled,
            )
            self.read_implant.read_null_insert_block = int(self.read_null_insert_block)
            if source_tap_blocks is not None:
                self.read_implant.set_source_tap_blocks(source_tap_blocks, reset_uniform=True)
        else:
            self.read_implant = None

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if self.use_positional_embedding_res:                           # <-- LongCLIP
            nn.init.normal_(self.positional_embedding_res, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

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
        modes = self.parse_text_modes(text)
        content_tokens = self._compact_control_tokens(
            text,
            (self.hard_text_token_id, self.no_text_token_id, self.any_text_token_id),
        )
        read_tokens = self._insert_hard_text_control(content_tokens)
        return {'modes': modes, 'content_tokens': content_tokens, 'read_tokens': read_tokens}

    @torch.no_grad()
    def set_hard_text_token_embedding(self, vector: torch.Tensor) -> None:
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

    def set_content_correction_enabled(self, enabled: bool = True) -> None:
        """Choose whether ordinary encode_image() applies the PIECES correction.

        Vanilla and RN-token-only checkpoints loaded into this x-attention-aware
        class default to ``False`` because their newly initialized bridge has not
        been trained. Bridge training can opt in explicitly; full and correction
        checkpoints default to ``True``.
        """
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

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        apply_correction = bool(
            getattr(self, '_clip_apply_content_correction_by_default', True)
        )
        if (
            not apply_correction
            or self.read_implant is None
            or not hasattr(self.visual, 'forward_with_intermediates')
        ):
            return self.encode_image_base(image)
        image_info = self.encode_image_states(image, return_final_tokens=False)
        return self._content_image_from_info(image_info, apply_content_correction=True)

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
        if self.read_implant is None:
            raise RuntimeError('Source-conditioned modes are available only for ViT backbones')

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
                'read_null_vit_attention': image_info.get('read_null_vit_attention', {}),
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
        if self.read_implant is None:
            raise RuntimeError('Hard text mode is available only for ViT backbones')

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
        # New public default for ViTs: untagged text is latent <any>; explicit
        # controls can be mixed column-wise. ResNet checkpoints retain the
        # ordinary CLIP path because the late-state implant is ViT-specific.
        if self.read_implant is None:
            image_features = F.normalize(self.encode_image_base(image), dim=-1)
            text_features = F.normalize(self.encode_text(text), dim=-1)
            logits = self.logit_scale.float().exp() * image_features @ text_features.t()
            return logits, logits.t()
        return self.forward_modes(image, text, apply_content_correction=True, return_details=False)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)



def _anytext_architecture_stage(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Classify checkpoint layout without consuming RNG or mutating state."""
    keys = tuple(map(str, state_dict.keys()))
    has_legacy = any(k.startswith("read_implant.presence_pool.") for k in keys)
    has_final = any(
        k.startswith((
            "read_implant.source_head.",
            "read_implant.orthographic_bridge.",
            "read_implant.trust_router.",
        ))
        for k in keys
    )
    if has_legacy and has_final:
        return "mixed-invalid"
    if has_legacy:
        return "legacy"
    if has_final:
        return "final"
    return "base-or-unknown"

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
    input_architecture_stage = _anytext_architecture_stage(state_dict)
    checkpoint_has_content_correction = (
        "read_implant.content_tap_logits" in state_dict
        or any(key.startswith("read_implant.content_pool.") for key in state_dict)
    )
    if input_architecture_stage == "mixed-invalid":
        raise RuntimeError(
            "Refusing mixed legacy/final AnyText checkpoint. A model may contain either "
            "legacy presence_pool or final source/orthography/trust modules, never both."
        )
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    checkpoint_vocab_size = state_dict["token_embedding.weight"].shape[0]
    vocab_size = max(
        int(checkpoint_vocab_size),
        int(hard_text_token_id) + 1,
        int(no_text_token_id) + 1,
        int(any_text_token_id) + 1,
        int(null_text_token_id) + 1,
    )

    checkpoint_bridge_keys = [
        key for key in state_dict if key.startswith("read_implant.read_bridge.")
    ]
    inferred_read_architecture = (
        "sigmoid_all"
        if "read_implant.read_bridge.sigmoid_patch_head_bias" in state_dict
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
        fresh_sigmoid_all_from_legacy = (
            input_architecture_stage == "legacy" and selected_read_architecture == "sigmoid_all"
        )
        if (
            checkpoint_bridge_keys
            and selected_read_architecture != inferred_read_architecture
            and not fresh_sigmoid_all_from_legacy
        ):
            raise ValueError(
                "Checkpoint read-attention architecture mismatch: "
                f"checkpoint={inferred_read_architecture!r}, "
                f"requested={selected_read_architecture!r}. "
                "Only the explicit legacy -> sigmoid_all fresh-PIECES migration is allowed; "
                "learned final PIECES weights are never silently converted."
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
        read_tap_blocks = (max(0, int(vision_layers) - 4), max(0, int(vision_layers) - 3)) if vit else None
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
    read_bridge_width = int(
        state_dict.get("read_implant.read_bridge.q_proj.weight", torch.empty(256, 1)).shape[0]
    )
    read_bridge_heads = int(
        state_dict.get("read_implant.bridge_heads_config", torch.tensor(4)).item()
    )
    source_hidden_width = int(
        state_dict.get(
            "read_implant.source_head.patch_contract.weight",
            torch.empty(max(128, read_bridge_width), early_expanded_width),
        ).shape[0]
    )
    trust_hidden_width = int(
        state_dict.get(
            "read_implant.trust_router.fc1.weight",
            torch.empty(128, CandidateTrustRouter.N_FEATURES),
        ).shape[0]
    )
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

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
        source_hidden_width=source_hidden_width,
        trust_hidden_width=trust_hidden_width,
        read_attention_architecture=selected_read_architecture,
        read_null_enabled=selected_read_null_enabled,
        read_null_insert_block=selected_read_null_insert,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)

    # OpenAI CLIP stores the vanilla backbone in FP16.  PIECES is a newly
    # trained bridge/router stack, however, and must remain FP32 so loading an
    # HF safetensors backbone is immediately safe for bridge fine-tuning.
    if model.read_implant is not None:
        model.read_implant.float()
    model.hard_text_embedding.data = model.hard_text_embedding.data.float()
    model.null_text_embedding.data = model.null_text_embedding.data.float()

    tok = state_dict["token_embedding.weight"]
    if tok.shape[0] < vocab_size:
        pad = torch.zeros(
            vocab_size - tok.shape[0], tok.shape[1], dtype=tok.dtype, device=tok.device
        )
        state_dict["token_embedding.weight"] = torch.cat([tok, pad], dim=0)
    # This branch intentionally replaces the old late presence/glyph/router stack.
    state_dict.pop("read_implant.readability_log_weight", None)
    state_dict.pop("read_implant.read_calibration_bias", None)
    fresh_pieces_init = bool(
        input_architecture_stage == "legacy"
        and selected_read_architecture == "sigmoid_all"
        and inferred_read_architecture != "sigmoid_all"
    )
    if fresh_pieces_init:
        # Training PIECES from scratch means exactly that: retain the established
        # base/hard-text-token scaffold, but do not migrate any learned legacy
        # read/content/presence PIECES parameters into the new attention topology.
        for stale_key in [k for k in state_dict if k.startswith("read_implant.")]:
            state_dict.pop(stale_key, None)

    legacy_prefixes = (
        "read_implant.presence_pool.",
        "read_implant.glyph_head.",
        "read_implant.auto_router.",
    )
    legacy_exact = {
        "read_implant.presence_tap_logits",
        "read_implant.glyph_tap_logits",
    }
    for stale_key in list(state_dict):
        if stale_key in legacy_exact or any(stale_key.startswith(p) for p in legacy_prefixes):
            state_dict.pop(stale_key, None)

    defaults = model.state_dict()
    for key in ("hard_text_embedding", "null_text_embedding"):
        if key not in state_dict:
            state_dict[key] = defaults[key]
    if selected_read_null_enabled:
        for key in ("visual.read_null_token", "visual.read_null_insert_block_config"):
            if key not in state_dict:
                state_dict[key] = defaults[key]

    for key, value in defaults.items():
        if key.startswith("read_implant.") and key not in state_dict:
            state_dict[key] = value

    model.load_state_dict(state_dict, strict=True)
    built_architecture_stage = _anytext_architecture_stage(model.state_dict())
    if built_architecture_stage != "final":
        raise RuntimeError(
            f"Final AnyText builder produced architecture={built_architecture_stage!r}; "
            "expected source_head + orthographic_bridge + trust_router and no presence_pool."
        )
    model._anytext_build_info = {
        "architecture_stage": "final",
        "input_architecture_stage": input_architecture_stage,
        "migrated_from_legacy": bool(input_architecture_stage == "legacy"),
        "fresh_pieces_init": bool(fresh_pieces_init),
        "read_attention_architecture": selected_read_architecture,
        "read_null_enabled": bool(selected_read_null_enabled),
        "read_null_insert_block": int(selected_read_null_insert),
    }
    # A vanilla or RN-token-only checkpoint gets a fresh bridge scaffold so it
    # can be fine-tuned immediately, but ordinary encode_image() must remain a
    # faithful backbone/RN call until learned correction weights are loaded.
    model._clip_apply_content_correction_by_default = bool(
        checkpoint_has_content_correction
    )
    return model.eval()
