from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from functools import wraps

from precision_policy import fp32_island, to_fp32_tree


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

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
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


class CrossOnlyReadBridge(nn.Module):
    """PreNorm text-query -> visual-key/value attention.

    Text states influence only the attention weights.  The returned read feature
    is built exclusively from visual values, so a candidate word cannot inject
    itself directly into the image representation.
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

        # Registers are available as a separately normalized memory bank, but
        # begin with exactly zero authority.  Training must earn their use.
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
        text_query: torch.Tensor,       # [N, C_text], normally pre-ln_final EOT state
        visual_tokens: torch.Tensor,    # [B, T, C_vision], pre-ln_post residual stream
        register_mask: Optional[torch.Tensor] = None,  # [B, T-1] over patch tokens
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

        q = q.view(N, self.heads, self.head_dim)                         # [N,H,D]
        k = k.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3) # [B,H,T,D]
        v = v.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3) # [B,H,T,D]
        logits = torch.einsum('nhd,bhtd->bnht', q, k) * (self.head_dim ** -0.5)

        # CLS is not visual evidence for the query.  Patch and register banks
        # are attended separately so high-norm implicit registers cannot engulf
        # the ordinary patch memory even before PreNorm.
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
        }


class SigmoidMassCrossOnlyReadBridge(nn.Module):
    """Independent patch-evidence attention for the isolated legacy curriculum.

    This class shares every projection with the legacy softmax reader but uses
    independent sigmoid patch weights and returns a bounded evidence term.  It
    exists only for explicit sigmoid experiments; the default softmax class and
    its state_dict remain exactly the working legacy architecture.
    """

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
        self.sigmoid_head_bias = nn.Parameter(
            torch.full((self.heads,), float(sigmoid_bias_init))
        )
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
            'patch_mass': patch_mass,
            'evidence': evidence,
        }


class SigmoidAllCrossOnlyReadBridge(nn.Module):
    """Raw independent-sigmoid replacement for legacy PIECES read attention.

    No post-sigmoid normalization, no division by attention mass, and no
    evidence multiplier.  Patch and register banks remain separate exactly as
    in the legacy softmax reader; only their token-attention normalization is
    replaced by independent sigmoid edges.
    """

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
        self.register_gate = nn.Parameter(torch.zeros(()))
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

    def forward(
        self,
        text_query: torch.Tensor,
        visual_tokens: torch.Tensor,
        register_mask: Optional[torch.Tensor] = None,
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

        # Dynamic -log(valid_count) starts a zero-QK head at ~unit total mass.
        # The learned offset then lets each head choose its own total read mass.
        # There is intentionally NO normalization after sigmoid.
        logits_fp32 = logits.float()
        patch_count = patch_keep.sum(dim=-1).clamp(min=1).float()
        patch_logits = (
            logits_fp32
            + self.sigmoid_patch_head_bias.float()[None, None, :, None]
            - patch_count.log()[:, None, None, None]
        )
        patch_weights = torch.sigmoid(patch_logits)
        patch_weights = patch_weights * patch_keep[:, None, None, :].float()
        patch_out = torch.einsum('bnht,bhtd->bnhd', patch_weights.to(v.dtype), v)
        patch_mass = patch_weights.sum(dim=-1)

        reg_weights = None
        reg_mass = None
        if register_mask is not None:
            reg_count = reg_keep.sum(dim=-1).clamp(min=1).float()
            reg_logits = (
                logits_fp32
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
            'patch_mass': patch_mass,
            'register_mass': reg_mass,
            'sigmoid_patch_head_bias': self.sigmoid_patch_head_bias,
            'sigmoid_register_head_bias': self.sigmoid_register_head_bias,
            'attention_normalization': self.ATTENTION_NORMALIZATION,
        }

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
    def forward(self, visual_tokens: torch.Tensor, return_attention: bool = False):
        B, T, _ = visual_tokens.shape
        x = self.vision_ln(visual_tokens)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum('hd,bhtd->bht', self.query.to(k.dtype), k) * (self.head_dim ** -0.5)
        keep = torch.ones((B, T), dtype=torch.bool, device=visual_tokens.device)
        keep[:, 0] = False
        probs = _safe_masked_softmax(logits, keep[:, None, :], dim=-1)
        pooled = torch.einsum('bht,bhtd->bhd', probs, v).reshape(B, self.pool_width)
        out = self.out_proj(pooled)
        if return_attention:
            return out, probs
        return out



class SigmoidAllVisualQueryPool(nn.Module):
    """Raw independent-sigmoid replacement for legacy PIECES visual pooling."""

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

    def forward(self, visual_tokens: torch.Tensor, return_attention: bool = False):
        B, T, _ = visual_tokens.shape
        x = self.vision_ln(visual_tokens)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.einsum('hd,bhtd->bht', self.query.to(k.dtype), k) * (self.head_dim ** -0.5)
        keep = torch.ones((B, T), dtype=torch.bool, device=visual_tokens.device)
        keep[:, 0] = False
        valid_count = keep.sum(dim=-1).clamp(min=1).float()
        adjusted = (
            logits.float()
            + self.sigmoid_head_bias.float()[None, :, None]
            - valid_count.log()[:, None, None]
        )
        weights = torch.sigmoid(adjusted) * keep[:, None, :].float()
        # Intentionally raw weighted sum: no division by weights.sum().
        pooled = torch.einsum('bht,bhtd->bhd', weights.to(v.dtype), v).reshape(B, self.pool_width)
        out = self.out_proj(pooled)
        if return_attention:
            return out, weights
        return out

class HardTextReadImplant(nn.Module):
    """Late hidden-state implant for explicit ``<text>`` mode.

    The ordinary CLIP path remains untouched.  The implant exposes:
      * a cross-only query-conditioned read feature,
      * a zero-initialized candidate-independent content correction,
      * zero-initialized ``text_present`` / ``text_readable`` logits.
    """

    def __init__(
        self,
        text_width: int,
        vision_width: int,
        output_dim: int,
        tap_blocks: Sequence[int],
        bridge_width: int = 256,
        heads: int = 4,
        read_attention_architecture: str = "softmax",
    ):
        super().__init__()
        taps = tuple(int(x) for x in tap_blocks)
        if not taps:
            raise ValueError('tap_blocks must contain at least one visual block index')
        self.register_buffer('tap_blocks', torch.tensor(taps, dtype=torch.long), persistent=True)
        self.register_buffer('bridge_heads_config', torch.tensor(int(heads), dtype=torch.long), persistent=True)

        architecture = str(read_attention_architecture)
        bridge_types = {
            "softmax": CrossOnlyReadBridge,
            "sigmoid_mass": SigmoidMassCrossOnlyReadBridge,
            "sigmoid_all": SigmoidAllCrossOnlyReadBridge,
        }
        if architecture not in bridge_types:
            raise ValueError(
                f"Unknown read_attention_architecture={architecture!r}; "
                f"expected one of {tuple(bridge_types)}"
            )
        self.read_attention_architecture = architecture
        self.read_bridge = bridge_types[architecture](
            text_width=text_width,
            vision_width=vision_width,
            bridge_width=bridge_width,
            output_dim=output_dim,
            heads=heads,
        )
        pool_type = SigmoidAllVisualQueryPool if architecture == "sigmoid_all" else PreNormVisualQueryPool
        self.content_pool = pool_type(
            vision_width=vision_width,
            pool_width=bridge_width,
            output_dim=output_dim,
            heads=heads,
            zero_output=True,
        )
        self.presence_pool = pool_type(
            vision_width=vision_width,
            pool_width=bridge_width,
            output_dim=2,
            heads=heads,
            zero_output=True,
        )

        n = len(taps)
        init_logits = torch.full((n,), -8.0)
        init_logits[0] = 0.0
        self.read_tap_logits = nn.Parameter(init_logits.clone())
        self.content_tap_logits = nn.Parameter(init_logits.clone())
        self.presence_tap_logits = nn.Parameter(init_logits.clone())

    def tap_block_list(self) -> List[int]:
        return [int(x) for x in self.tap_blocks.detach().cpu().tolist()]

    def _states_in_order(self, states: Dict[int, torch.Tensor]) -> List[torch.Tensor]:
        missing = [b for b in self.tap_block_list() if b not in states]
        if missing:
            raise KeyError(f'Missing requested visual tap states for blocks: {missing}')
        return [states[b] for b in self.tap_block_list()]

    @staticmethod
    def _weights(logits: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return logits.to(device=ref.device, dtype=torch.float32).softmax(dim=0)

    @_fp32_custom_forward
    def read_features(
        self,
        states: Dict[int, torch.Tensor],
        text_query: torch.Tensor,
        register_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
        return_evidence: bool = False,
    ):
        ordered = self._states_in_order(states)
        weights = self._weights(self.read_tap_logits, ordered[0])
        outputs = []
        evidence_outputs = []
        details = {}
        sigmoid_mass = self.read_attention_architecture == "sigmoid_mass"
        for block_idx, state in zip(self.tap_block_list(), ordered):
            if return_details or return_evidence or sigmoid_mass:
                out, attn = self.read_bridge(
                    text_query, state, register_mask=register_mask, return_attention=True
                )
                if return_details:
                    details[block_idx] = attn
                if sigmoid_mass:
                    evidence_outputs.append(attn["evidence"])
            else:
                out = self.read_bridge(text_query, state, register_mask=register_mask)
            outputs.append(out)
        stacked = torch.stack(outputs, dim=0)  # [K,B,N,D]
        mixed = torch.einsum('k,kbnd->bnd', weights, stacked)
        if sigmoid_mass:
            evidence = torch.einsum(
                'k,kbn->bn', weights, torch.stack(evidence_outputs, dim=0)
            ).clamp(min=0.0, max=1.0)
        else:
            evidence = torch.ones(mixed.shape[:2], dtype=mixed.dtype, device=mixed.device)
        if return_details:
            result = {
                'tap_weights': weights,
                'per_block': details,
                'evidence': evidence,
                'read_attention_architecture': self.read_attention_architecture,
            }
            if return_evidence:
                return mixed, evidence, result
            return mixed, result
        if return_evidence:
            return mixed, evidence
        return mixed

    @_fp32_custom_forward
    def content_correction(self, states: Dict[int, torch.Tensor], return_details: bool = False):
        ordered = self._states_in_order(states)
        weights = self._weights(self.content_tap_logits, ordered[0])
        outputs = []
        attn = {}
        for block_idx, state in zip(self.tap_block_list(), ordered):
            if return_details:
                out, probs = self.content_pool(state, return_attention=True)
                attn[block_idx] = probs
            else:
                out = self.content_pool(state)
            outputs.append(out)
        mixed = torch.einsum('k,kbd->bd', weights, torch.stack(outputs, dim=0))
        if return_details:
            return mixed, {'tap_weights': weights, 'per_block': attn}
        return mixed

    @_fp32_custom_forward
    def presence_logits(self, states: Dict[int, torch.Tensor], return_details: bool = False):
        ordered = self._states_in_order(states)
        weights = self._weights(self.presence_tap_logits, ordered[0])
        outputs = []
        attn = {}
        for block_idx, state in zip(self.tap_block_list(), ordered):
            if return_details:
                out, probs = self.presence_pool(state, return_attention=True)
                attn[block_idx] = probs
            else:
                out = self.presence_pool(state)
            outputs.append(out)
        mixed = torch.einsum('k,bkd->bd', weights, torch.stack(outputs, dim=1))
        if return_details:
            return mixed, {'tap_weights': weights, 'per_block': attn}
        return mixed


class VisionTransformer(nn.Module):
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

        self.register_norm_threshold: float = 60.0
        self.register_min: int = 1
        self.register_max: int = 0
        self._last_register_mask: Optional[torch.Tensor] = None
        self._last_patch_l2_norms: Optional[torch.Tensor] = None

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
        patch_norms = x_nld[:, 1:, :].detach().float().norm(dim=-1)
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
        }

    def forward(self, x: torch.Tensor):
        x_tbc = self._prepare_tokens(x)
        x_tbc = self.transformer(x_tbc)
        x_nld = x_tbc.permute(1, 0, 2)
        self._compute_register_mask_from_final_tokens(x_nld)
        return self._finalize_cls(x_nld)


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
                 use_positional_embedding_res: bool = False,                    # <-- LongCLIP: internal-only switch
                 longclip_keep_len: int = 20,                                   # <-- LongCLIP: Long-CLIP convention
                 hard_text_token_id: int = 49408,
                 eot_token_id: int = 49407,
                 read_tap_blocks: Optional[Sequence[int]] = None,
                 read_bridge_width: int = 256,
                 read_bridge_heads: int = 4,
                 read_attention_architecture: str = "softmax",
                 ):
        super().__init__()

        self.context_length = context_length
        self.hard_text_token_id = int(hard_text_token_id)
        self.eot_token_id = int(eot_token_id)
        self.read_attention_architecture = str(read_attention_architecture)
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
        # Dedicated trainable embedding for the literal hard <text> ID.
        # It replaces the lookup result only at that ID, so the frozen base
        # token_embedding matrix never needs gradient masking.
        self.hard_text_embedding = nn.Parameter(torch.zeros(transformer_width))

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
                bridge_width=read_bridge_width,
                heads=read_bridge_heads,
                read_attention_architecture=self.read_attention_architecture,
            )
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
            # Backward-compatible fallback for unusual externally produced token tensors.
            idx = idx.clone()
            idx[missing] = text[missing].argmax(dim=-1)
        return idx

    def is_text_mode(self, text: torch.Tensor) -> torch.Tensor:
        return text.eq(self.hard_text_token_id).any(dim=-1)

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
        # Keep the reserved row synchronized for transparent state inspection.
        self.token_embedding.weight[self.hard_text_token_id].copy_(
            vector.to(device=self.token_embedding.weight.device, dtype=self.token_embedding.weight.dtype)
        )

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_image_states(
        self,
        image: torch.Tensor,
        tap_blocks: Optional[Sequence[int]] = None,
        return_final_tokens: bool = True,
    ) -> Dict[str, Any]:
        if self.read_implant is None or not hasattr(self.visual, 'forward_with_intermediates'):
            raise RuntimeError('Hard text implant requires a VisionTransformer backbone')
        taps = self.read_implant.tap_block_list() if tap_blocks is None else [int(x) for x in tap_blocks]
        return self.visual.forward_with_intermediates(
            image.type(self.dtype), taps, return_final_tokens=return_final_tokens
        )

    def _encode_text_hidden(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.token_embedding(text).type(self.dtype)
        hard_mask = text.eq(self.hard_text_token_id).unsqueeze(-1)
        if hard_mask.any():
            hard = self.hard_text_embedding.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
            x = torch.where(hard_mask, hard, x)
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
            'read_mode_mask': self.is_text_mode(text),
        }

    def encode_text(self, text):
        return self._encode_text_hidden(text)['text_embedding']

    def encode_text_states(self, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._encode_text_hidden(text)

    def forward_hard(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        apply_content_correction: bool = True,
        return_details: bool = False,
    ):
        """Score ordinary and explicit ``<text>`` queries in one matrix.

        Ordinary columns use the normal CLIP cosine path (plus a zero-init
        candidate-independent correction when enabled).  Columns containing the
        literal hard token use cross-only text-query -> visual-value attention.
        """
        if self.read_implant is None:
            raise RuntimeError('Hard text mode is available only for ViT backbones')

        image_info = self.encode_image_states(image, return_final_tokens=return_details)
        text_info = self.encode_text_states(text)
        base_image = image_info['image_embedding']

        correction_details = None
        if apply_content_correction:
            if return_details:
                correction, correction_details = self.read_implant.content_correction(
                    image_info['states'], return_details=True
                )
            else:
                correction = self.read_implant.content_correction(image_info['states'])
            content_image = base_image + correction
        else:
            correction = torch.zeros_like(base_image)
            content_image = base_image

        image_norm = _fp32_normalize(content_image)
        text_norm = _fp32_normalize(text_info['text_embedding'])
        scale = self.logit_scale.float().exp()
        logits = _fp32_scaled_matmul(scale, image_norm, text_norm.t())

        read_mask = text_info['read_mode_mask']
        read_details = None
        if read_mask.any():
            query = text_info['eot_hidden_pre_ln'][read_mask]
            if return_details:
                read_feature, read_details = self.read_implant.read_features(
                    image_info['states'],
                    query,
                    register_mask=image_info['register_mask'],
                    return_details=True,
                )
            else:
                read_feature = self.read_implant.read_features(
                    image_info['states'], query, register_mask=image_info['register_mask']
                )
            read_feature = _fp32_normalize(read_feature)
            read_text = text_norm[read_mask]
            read_logits = _fp32_scaled_einsum('bnd,nd->bn', scale, read_feature, read_text)
            logits = logits.clone()
            logits[:, read_mask] = read_logits

        presence_details = None
        if return_details:
            presence_logits, presence_details = self.read_implant.presence_logits(
                image_info['states'], return_details=True
            )
            return {
                'logits_per_image': logits,
                'logits_per_text': logits.t(),
                'read_mode_mask': read_mask,
                'base_image_embedding': base_image,
                'content_image_embedding': content_image,
                'content_correction': correction,
                'text_embedding': text_info['text_embedding'],
                'text_hidden_pre_ln': text_info['hidden_pre_ln'],
                'eot_indices': text_info['eot_indices'],
                'visual_states': image_info['states'],
                'visual_final_tokens': image_info['final_tokens'],
                'register_mask': image_info['register_mask'],
                'patch_token_norms': image_info['patch_token_norms'],
                'presence_logits': presence_logits,
                'read_details': read_details,
                'content_details': correction_details,
                'presence_details': presence_details,
            }
        return logits, logits.t()

    def forward(self, image, text):
        # Untampered OpenAI/GmP path.  Explicit hard routing is opt-in through
        # forward_hard(), so old evaluation code remains bit-for-bit equivalent.
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.float().exp()
        logits_per_image = _fp32_scaled_matmul(logit_scale, image_features, text_features.t())
        return logits_per_image, logits_per_image.t()


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
    eot_token_id: int = 49407,
    read_attention_architecture: Optional[str] = None,
):
    input_stage = _anytext_architecture_stage(state_dict)
    if input_stage in {"final", "mixed-invalid"}:
        raise RuntimeError(
            "legacy_hard_text_pre.oaiclip refuses final/mixed AnyText state: "
            f"architecture={input_stage}. This package may only build base or legacy models."
        )
    inferred_read_architecture = (
        "sigmoid_all"
        if "read_implant.read_bridge.sigmoid_patch_head_bias" in state_dict
        else ("sigmoid_mass" if "read_implant.read_bridge.sigmoid_head_bias" in state_dict else "softmax")
    )
    selected_read_architecture = (
        inferred_read_architecture
        if read_attention_architecture is None
        else str(read_attention_architecture)
    )
    if selected_read_architecture not in {"softmax", "sigmoid_mass", "sigmoid_all"}:
        raise ValueError(f"Unknown read_attention_architecture={selected_read_architecture!r}")
    if any(k.startswith("read_implant.read_bridge.") for k in state_dict) and selected_read_architecture != inferred_read_architecture:
        raise ValueError(
            "Requested legacy reader architecture disagrees with checkpoint parameters: "
            f"requested={selected_read_architecture}, inferred={inferred_read_architecture}"
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
    vocab_size = max(int(checkpoint_vocab_size), int(hard_text_token_id) + 1)

    if "read_implant.tap_blocks" in state_dict:
        read_tap_blocks = tuple(int(x) for x in state_dict["read_implant.tap_blocks"].tolist())
    else:
        read_tap_blocks = (max(0, int(vision_layers) - 4), max(0, int(vision_layers) - 3)) if vit else None
    read_bridge_width = int(
        state_dict.get("read_implant.read_bridge.q_proj.weight", torch.empty(256, 1)).shape[0]
    )
    read_bridge_heads = int(
        state_dict.get("read_implant.bridge_heads_config", torch.tensor(4)).item()
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
        eot_token_id=eot_token_id,
        read_tap_blocks=read_tap_blocks,
        read_bridge_width=read_bridge_width,
        read_bridge_heads=read_bridge_heads,
        read_attention_architecture=selected_read_architecture,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)

    tok = state_dict["token_embedding.weight"]
    if tok.shape[0] < vocab_size:
        pad = torch.zeros(
            vocab_size - tok.shape[0], tok.shape[1], dtype=tok.dtype, device=tok.device
        )
        state_dict["token_embedding.weight"] = torch.cat([tok, pad], dim=0)
    defaults = model.state_dict()
    if "hard_text_embedding" not in state_dict:
        state_dict["hard_text_embedding"] = defaults["hard_text_embedding"]
    for key, value in defaults.items():
        if key.startswith("read_implant.") and key not in state_dict:
            state_dict[key] = value

    model.load_state_dict(state_dict, strict=True)
    built_stage = _anytext_architecture_stage(model.state_dict())
    if built_stage != "legacy":
        raise RuntimeError(
            f"legacy oaiclip build produced unexpected architecture={built_stage}; "
            "presence_pool must be retained and final modules must be absent."
        )
    model._anytext_build_info = {
        "architecture_stage": "legacy",
        "migrated_from_legacy": False,
        "read_attention_architecture": selected_read_architecture,
    }
    return model.eval()
