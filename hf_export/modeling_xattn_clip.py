from __future__ import annotations

import math
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import ModelOutput

try:  # local package import and Hugging Face custom-code import
    from .configuration_xattn_clip import XAttnCLIPConfig
except ImportError:  # pragma: no cover - remote custom-code import
    from configuration_xattn_clip import XAttnCLIPConfig


MODE_IDS = {"any": 0, "read": 1, "text": 1, "notext": 2}
_PIECES_FORCE_FP32 = ContextVar("xattn_pieces_force_fp32", default=True)


def _autocast_off(reference: torch.Tensor):
    if _PIECES_FORCE_FP32.get():
        return torch.autocast(device_type=reference.device.type, enabled=False)
    return nullcontext()


def _linear(module: nn.Linear, value: torch.Tensor) -> torch.Tensor:
    bias = None if module.bias is None else module.bias.float()
    return F.linear(value.float(), module.weight.float(), bias)


def _layer_norm(module: nn.LayerNorm, value: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(
        value.float(),
        module.normalized_shape,
        module.weight.float(),
        module.bias.float(),
        module.eps,
    )


def _normalize(value: torch.Tensor) -> torch.Tensor:
    with _autocast_off(value):
        return F.normalize(value.float(), dim=-1)


def _fp32_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    with _autocast_off(operands[0]):
        return torch.einsum(equation, *(operand.float() for operand in operands))


def _scaled_matmul(
    scale: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    with _autocast_off(left):
        return scale.float() * torch.matmul(left.float(), right.float())


def _scaled_einsum(
    equation: str, scale: torch.Tensor, *operands: torch.Tensor
) -> torch.Tensor:
    return scale.float() * _fp32_einsum(equation, *operands)


def _safe_masked_softmax(
    logits: torch.Tensor, keep: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    keep = keep.to(device=logits.device, dtype=torch.bool)
    masked = logits.float().masked_fill(~keep, torch.finfo(torch.float32).min)
    probabilities = masked.softmax(dim=dim)
    has_value = keep.any(dim=dim, keepdim=True)
    return torch.where(has_value, probabilities, torch.zeros_like(probabilities))


class CrossOnlyReadBridge(nn.Module):
    """Candidate-conditioned late visual readout used by PIECES."""

    def __init__(self, config: XAttnCLIPConfig):
        super().__init__()
        text_width = int(config.text_config.hidden_size)
        vision_width = int(config.vision_config.hidden_size)
        bridge_width = int(config.read_bridge_width)
        self.bridge_width = bridge_width
        self.heads = int(config.read_bridge_heads)
        self.head_dim = bridge_width // self.heads
        self.architecture = str(config.read_attention_architecture)
        self.text_ln = nn.LayerNorm(text_width, eps=config.text_config.layer_norm_eps)
        self.vision_ln = nn.LayerNorm(
            vision_width, eps=config.vision_config.layer_norm_eps
        )
        self.q_proj = nn.Linear(text_width, bridge_width)
        self.k_proj = nn.Linear(vision_width, bridge_width)
        self.v_proj = nn.Linear(vision_width, bridge_width)
        self.out_proj = nn.Linear(bridge_width, config.projection_dim)
        self.register_gate = nn.Parameter(torch.zeros(()))
        if self.architecture == "sigmoid_all":
            self.sigmoid_patch_head_bias = nn.Parameter(torch.zeros(self.heads))
            self.sigmoid_register_head_bias = nn.Parameter(torch.zeros(self.heads))
        else:
            self.register_parameter("sigmoid_patch_head_bias", None)
            self.register_parameter("sigmoid_register_head_bias", None)

    def forward(
        self,
        text_query: torch.Tensor,
        visual_tokens: torch.Tensor,
        *,
        register_mask: torch.Tensor | None = None,
        patch_textness: torch.Tensor | None = None,
        textness_beta: torch.Tensor | None = None,
        read_null_index: int | None = None,
        return_attention: bool = False,
    ):
        if text_query.ndim != 2 or visual_tokens.ndim != 3:
            raise ValueError(
                "text_query must be [N,C] and visual_tokens must be [B,T,C]"
            )
        with _autocast_off(visual_tokens):
            batch, token_count, _ = visual_tokens.shape
            candidates = text_query.shape[0]
            query = _linear(self.q_proj, _layer_norm(self.text_ln, text_query))
            vision = _layer_norm(self.vision_ln, visual_tokens)
            key = _linear(self.k_proj, vision)
            value = _linear(self.v_proj, vision)
            query = query.view(candidates, self.heads, self.head_dim)
            key = key.view(batch, token_count, self.heads, self.head_dim).permute(
                0, 2, 1, 3
            )
            value = value.view(batch, token_count, self.heads, self.head_dim).permute(
                0, 2, 1, 3
            )
            logits = torch.einsum("nhd,bhtd->bnht", query, key) / math.sqrt(
                self.head_dim
            )

            if patch_textness is not None:
                if patch_textness.shape != (batch, token_count - 1):
                    raise ValueError(
                        f"patch_textness must be {(batch, token_count - 1)}, "
                        f"got {tuple(patch_textness.shape)}"
                    )
                beta = torch.as_tensor(
                    0.0 if textness_beta is None else textness_beta,
                    device=logits.device,
                    dtype=torch.float32,
                )
                bias = torch.zeros((batch, token_count), device=logits.device)
                bias[:, 1:] = patch_textness.float().clamp(1.0e-4, 1.0).log()
                logits = logits.float() + beta * bias[:, None, None, :]

            patch_keep = torch.ones(
                (batch, token_count), dtype=torch.bool, device=visual_tokens.device
            )
            patch_keep[:, 0] = False
            register_keep = torch.zeros_like(patch_keep)
            if register_mask is not None:
                if register_mask.shape != (batch, token_count - 1):
                    raise ValueError(
                        f"register_mask must be {(batch, token_count - 1)}, "
                        f"got {tuple(register_mask.shape)}"
                    )
                register_keep[:, 1:] = register_mask.bool()
                patch_keep[:, 1:] &= ~register_keep[:, 1:]

            if self.architecture == "sigmoid_all":
                patch_count = patch_keep.sum(dim=-1).clamp(min=1).float()
                patch_logits = (
                    logits
                    + self.sigmoid_patch_head_bias.float()[None, None, :, None]
                    - patch_count.log()[:, None, None, None]
                )
                patch_attention = torch.sigmoid(patch_logits)
                patch_attention = patch_attention * patch_keep[:, None, None, :].float()
                pooled = torch.einsum("bnht,bhtd->bnhd", patch_attention, value)
                register_attention = None
                register_mass = None
                if register_mask is not None:
                    register_count = register_keep.sum(dim=-1).clamp(min=1).float()
                    register_logits = (
                        logits
                        + self.sigmoid_register_head_bias.float()[None, None, :, None]
                        - register_count.log()[:, None, None, None]
                    )
                    register_attention = torch.sigmoid(register_logits)
                    register_attention = register_attention * register_keep[:, None, None, :].float()
                    register_out = torch.einsum(
                        "bnht,bhtd->bnhd", register_attention, value
                    )
                    pooled = pooled + self.register_gate.float() * register_out
                    register_mass = register_attention.sum(dim=-1)
                patch_mass = patch_attention.sum(dim=-1)
            else:
                patch_attention = _safe_masked_softmax(
                    logits, patch_keep[:, None, None, :]
                )
                pooled = torch.einsum("bnht,bhtd->bnhd", patch_attention, value)
                register_attention = None
                register_mass = None
                if register_mask is not None:
                    register_attention = _safe_masked_softmax(
                        logits, register_keep[:, None, None, :]
                    )
                    register_out = torch.einsum(
                        "bnht,bhtd->bnhd", register_attention, value
                    )
                    pooled = pooled + self.register_gate.float() * register_out
                patch_mass = patch_attention.sum(dim=-1)

            read_null_attention = None
            if read_null_index is not None:
                null_index = int(read_null_index)
                if null_index < 0:
                    null_index += token_count
                if not 0 < null_index < token_count:
                    raise ValueError(
                        "read_null_index is outside visual evidence tokens"
                    )
                read_null_attention = patch_attention[..., null_index]
            result = _linear(
                self.out_proj, pooled.reshape(batch, candidates, self.bridge_width)
            )
            if not return_attention:
                return result
            return result, {
                "patch_attention": patch_attention,
                "register_attention": register_attention,
                "register_gate": self.register_gate,
                "patch_textness": patch_textness,
                "textness_beta": textness_beta,
                "patch_mass": patch_mass,
                "register_mass": register_mass,
                "read_null_attention": read_null_attention,
                "read_null_index": read_null_index,
                "attention_normalization": (
                    "independent_sigmoid_raw"
                    if self.architecture == "sigmoid_all"
                    else "softmax"
                ),
            }


class VisualQueryPool(nn.Module):
    """Candidate-independent late content correction; CLS and RN are excluded."""

    def __init__(self, config: XAttnCLIPConfig):
        super().__init__()
        vision_width = int(config.vision_config.hidden_size)
        self.pool_width = int(config.read_bridge_width)
        self.heads = int(config.read_bridge_heads)
        self.head_dim = self.pool_width // self.heads
        self.architecture = str(config.read_attention_architecture)
        self.vision_ln = nn.LayerNorm(
            vision_width, eps=config.vision_config.layer_norm_eps
        )
        self.query = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.k_proj = nn.Linear(vision_width, self.pool_width)
        self.v_proj = nn.Linear(vision_width, self.pool_width)
        self.out_proj = nn.Linear(self.pool_width, config.projection_dim)
        if self.architecture == "sigmoid_all":
            self.sigmoid_head_bias = nn.Parameter(torch.zeros(self.heads))
        else:
            self.register_parameter("sigmoid_head_bias", None)

    def forward(self, visual_tokens: torch.Tensor, *, return_attention: bool = False):
        with _autocast_off(visual_tokens):
            batch, token_count, _ = visual_tokens.shape
            vision = _layer_norm(self.vision_ln, visual_tokens)
            key = (
                _linear(self.k_proj, vision)
                .view(batch, token_count, self.heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            value = (
                _linear(self.v_proj, vision)
                .view(batch, token_count, self.heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            logits = torch.einsum("hd,bhtd->bht", self.query.float(), key)
            logits /= math.sqrt(self.head_dim)
            keep = torch.ones(
                (batch, token_count), dtype=torch.bool, device=visual_tokens.device
            )
            keep[:, 0] = False
            if token_count < 3:
                raise ValueError("Visual state has no room for both CLS and RN")
            keep[:, -1] = False
            if self.architecture == "sigmoid_all":
                count = keep.sum(dim=-1).clamp(min=1).float()
                adjusted = (
                    logits
                    + self.sigmoid_head_bias.float()[None, :, None]
                    - count.log()[:, None, None]
                )
                attention = torch.sigmoid(adjusted) * keep[:, None, :].float()
            else:
                attention = _safe_masked_softmax(logits, keep[:, None, :])
            pooled = torch.einsum("bht,bhtd->bhd", attention, value)
            result = _linear(self.out_proj, pooled.reshape(batch, self.pool_width))
            return (result, attention) if return_attention else result


class EarlyOrthographicBridge(nn.Module):
    """Candidate-conditioned orthographic match over early spatial states."""

    def __init__(self, config: XAttnCLIPConfig):
        super().__init__()
        text_width = int(config.text_config.hidden_size)
        vision_width = int(config.vision_config.hidden_size)
        self.bridge_width = int(config.read_bridge_width)
        self.heads = int(config.read_bridge_heads)
        self.head_dim = self.bridge_width // self.heads
        self.architecture = str(config.read_attention_architecture)
        self.text_ln = nn.LayerNorm(text_width, eps=config.text_config.layer_norm_eps)
        self.patch_ln = nn.LayerNorm(
            vision_width, eps=config.vision_config.layer_norm_eps
        )
        self.patch_expand = nn.Linear(vision_width, int(config.early_expanded_width))
        self.patch_contract = nn.Linear(
            int(config.early_expanded_width), self.bridge_width
        )
        self.q_proj = nn.Linear(text_width, self.bridge_width)
        self.k_proj = nn.Linear(self.bridge_width, self.bridge_width)
        self.v_proj = nn.Linear(self.bridge_width, self.bridge_width)
        self.out_proj = nn.Linear(self.bridge_width, config.projection_dim)
        if self.architecture == "sigmoid_all":
            self.sigmoid_head_bias = nn.Parameter(torch.zeros(self.heads))
        else:
            self.register_parameter("sigmoid_head_bias", None)

    def forward(
        self,
        text_query: torch.Tensor,
        visual_tokens: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        with _autocast_off(visual_tokens):
            patches = visual_tokens[:, 1:, :]
            batch, patch_count, _ = patches.shape
            candidates = text_query.shape[0]
            patch_hidden = F.gelu(
                _linear(self.patch_expand, _layer_norm(self.patch_ln, patches))
            )
            patch_hidden = F.gelu(_linear(self.patch_contract, patch_hidden))
            query = _linear(self.q_proj, _layer_norm(self.text_ln, text_query)).view(
                candidates, self.heads, self.head_dim
            )
            key = (
                _linear(self.k_proj, patch_hidden)
                .view(batch, patch_count, self.heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            value = (
                _linear(self.v_proj, patch_hidden)
                .view(batch, patch_count, self.heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            logits = torch.einsum("nhd,bhpd->bnhp", query, key)
            logits /= math.sqrt(self.head_dim)
            if self.architecture == "sigmoid_all":
                adjusted = (
                    logits.float()
                    + self.sigmoid_head_bias.float()[None, None, :, None]
                    - math.log(max(1, patch_count))
                )
                attention = torch.sigmoid(adjusted)
            else:
                attention = logits.float().softmax(dim=-1)
            pooled = torch.einsum("bnhp,bhpd->bnhd", attention, value)
            result = _linear(
                self.out_proj, pooled.reshape(batch, candidates, self.bridge_width)
            )
            return (result, attention) if return_attention else result


class EarlySourceGate(nn.Module):
    """Patch-local text-source detector and global readability head."""

    def __init__(self, config: XAttnCLIPConfig):
        super().__init__()
        vision_width = int(config.vision_config.hidden_size)
        hidden_width = int(config.source_hidden_width)
        self.patch_ln = nn.LayerNorm(
            vision_width, eps=config.vision_config.layer_norm_eps
        )
        self.patch_expand = nn.Linear(vision_width, int(config.early_expanded_width))
        self.patch_contract = nn.Linear(int(config.early_expanded_width), hidden_width)
        self.patch_out = nn.Linear(hidden_width, 1)
        self.stats_fc1 = nn.Linear(8, hidden_width)
        self.stats_fc2 = nn.Linear(hidden_width, 2)

    def patch_logits(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        with _autocast_off(visual_tokens):
            patches = visual_tokens[:, 1:, :]
            hidden = F.gelu(
                _linear(self.patch_expand, _layer_norm(self.patch_ln, patches))
            )
            hidden = F.gelu(_linear(self.patch_contract, hidden))
            return _linear(self.patch_out, hidden).squeeze(-1)

    def aggregate(self, patch_logits: torch.Tensor):
        with _autocast_off(patch_logits):
            patch_logits = patch_logits.float()
            probabilities = patch_logits.sigmoid()
            batch, patches = probabilities.shape
            grid = int(round(patches**0.5))
            topk = probabilities.topk(k=min(8, patches), dim=-1).values.mean(dim=-1)
            soft_area = torch.sigmoid((probabilities - 0.50) * 12.0).mean(dim=-1)
            if grid * grid == patches:
                spatial = probabilities.view(batch, grid, grid)
                row_max_mean = spatial.amax(dim=2).mean(dim=1)
                col_max_mean = spatial.amax(dim=1).mean(dim=1)
                row_density_max = spatial.mean(dim=2).amax(dim=1)
            else:
                row_max_mean = probabilities.amax(dim=-1)
                col_max_mean = probabilities.amax(dim=-1)
                row_density_max = probabilities.mean(dim=-1)
            statistics = torch.stack(
                (
                    probabilities.mean(dim=-1),
                    probabilities.amax(dim=-1),
                    topk,
                    soft_area,
                    row_max_mean,
                    col_max_mean,
                    row_density_max,
                    patch_logits.mean(dim=-1),
                ),
                dim=-1,
            )
            logits = _linear(
                self.stats_fc2, F.gelu(_linear(self.stats_fc1, statistics))
            )
            return logits, probabilities, statistics


class CandidateTrustRouter(nn.Module):
    """Candidate-conditioned trust, kept separate from source availability."""

    def __init__(self, config: XAttnCLIPConfig):
        super().__init__()
        hidden = int(config.trust_hidden_width)
        self.fc1 = nn.Linear(16, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, 1)

    def forward(
        self,
        *,
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
        injection: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with _autocast_off(content_logits):
            batch, candidates = content_logits.shape
            content_logits = content_logits.detach().float()
            read_logits = read_logits.detach().float()
            null_logits = null_logits.detach().float().reshape(batch, 1)
            early_logits = early_logits.detach().float()
            content_image = content_image.detach().float()
            read_image = read_image.detach().float()
            content_text = content_text.detach().float()
            read_text = read_text.detach().float()
            source_logits = source_logits.detach().float()
            source_stats = source_stats.detach().float()
            relative = read_logits - null_logits
            positive = 0.5 * (relative + torch.sqrt(relative.square() + 1.0e-4))
            image_cosine = torch.einsum("bd,bnd->bn", content_image, read_image)
            text_cosine = torch.einsum("nd,nd->n", content_text, read_text)[None, :]
            text_cosine = text_cosine.expand(batch, candidates)
            source_probabilities = source_logits.sigmoid()
            source = source_probabilities[:, 0:1].expand(batch, candidates)
            ordered = source_probabilities[:, 1:2].expand(batch, candidates)
            glyph_mean = source_stats[:, 0:1].expand(batch, candidates)
            glyph_max = source_stats[:, 1:2].expand(batch, candidates)
            glyph_topk = source_stats[:, 2:3].expand(batch, candidates)
            glyph_area = source_stats[:, 3:4].expand(batch, candidates)
            if injection is None:
                injected = torch.zeros_like(content_logits)
            else:
                injected = (
                    injection.detach()
                    .float()
                    .reshape(batch, 1)
                    .expand(batch, candidates)
                )
            features = torch.stack(
                (
                    torch.tanh(content_logits / 20.0),
                    torch.tanh(read_logits / 20.0),
                    torch.tanh(relative / 20.0),
                    torch.tanh(positive / 20.0),
                    torch.tanh(early_logits / 20.0),
                    torch.tanh((early_logits - relative) / 20.0),
                    image_cosine,
                    text_cosine,
                    source,
                    ordered,
                    glyph_mean,
                    glyph_max,
                    glyph_topk,
                    glyph_area,
                    injected,
                    injected * torch.tanh(relative / 20.0),
                ),
                dim=-1,
            )
            hidden = F.gelu(_linear(self.fc1, features))
            hidden = F.gelu(_linear(self.fc2, hidden))
            return torch.sigmoid(_linear(self.fc3, hidden).squeeze(-1))


class HardTextReadImplant(nn.Module):
    """Early source/orthography branches plus late lexical readout and trust."""

    def __init__(self, config: XAttnCLIPConfig):
        super().__init__()
        self.read_attention_architecture = str(config.read_attention_architecture)
        self.read_null_enabled = True
        self.read_null_insert_block = int(config.read_null_insert_block)
        self.register_buffer(
            "tap_blocks", torch.tensor(config.read_tap_blocks), persistent=True
        )
        self.register_buffer(
            "ortho_tap_blocks", torch.tensor(config.ortho_tap_blocks), persistent=True
        )
        self.register_buffer(
            "source_tap_blocks", torch.tensor(config.source_tap_blocks), persistent=True
        )
        self.register_buffer(
            "bridge_heads_config",
            torch.tensor(config.read_bridge_heads),
            persistent=True,
        )
        self.register_buffer(
            "early_expanded_width_config",
            torch.tensor(config.early_expanded_width),
            persistent=True,
        )
        self.read_bridge = CrossOnlyReadBridge(config)
        self.content_pool = VisualQueryPool(config)
        self.orthographic_bridge = EarlyOrthographicBridge(config)
        self.source_head = EarlySourceGate(config)
        self.trust_router = CandidateTrustRouter(config)
        self.read_tap_logits = nn.Parameter(torch.zeros(len(config.read_tap_blocks)))
        self.content_tap_logits = nn.Parameter(torch.zeros(len(config.read_tap_blocks)))
        self.ortho_tap_logits = nn.Parameter(torch.zeros(len(config.ortho_tap_blocks)))
        self.source_tap_logits = nn.Parameter(
            torch.zeros(len(config.source_tap_blocks))
        )
        self.glyph_bias_beta = nn.Parameter(torch.zeros(()))
        self.read_calibration_scale = nn.Parameter(torch.ones(()))
        self.null_abstain_weight = nn.Parameter(torch.zeros(()))
        self.auto_read_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "read_probe", torch.zeros(config.projection_dim), persistent=True
        )

    @staticmethod
    def _block_list(buffer: torch.Tensor) -> list[int]:
        return [int(value) for value in buffer.detach().cpu().tolist()]

    def capture_block_list(self) -> list[int]:
        return sorted(
            set(
                self._block_list(self.tap_blocks)
                + self._block_list(self.ortho_tap_blocks)
                + self._block_list(self.source_tap_blocks)
            )
        )

    def _spatial_state(self, state: torch.Tensor, block: int) -> torch.Tensor:
        return state[:, :-1, :] if block >= self.read_null_insert_block else state

    @staticmethod
    def _mix_weights(logits: torch.Tensor) -> torch.Tensor:
        return logits.float().softmax(dim=0)

    @staticmethod
    def _states(
        states: Mapping[int, torch.Tensor], blocks: Sequence[int]
    ) -> list[torch.Tensor]:
        missing = [block for block in blocks if block not in states]
        if missing:
            raise KeyError(f"Missing requested visual tap states: {missing}")
        return [states[block] for block in blocks]

    def source_outputs(
        self, states: Mapping[int, torch.Tensor], return_details: bool = False
    ):
        blocks = self._block_list(self.source_tap_blocks)
        ordered = self._states(states, blocks)
        weights = self._mix_weights(self.source_tap_logits)
        per_block = [
            self.source_head.patch_logits(self._spatial_state(state, block))
            for block, state in zip(blocks, ordered)
        ]
        mixed = _fp32_einsum("k,kbp->bp", weights, torch.stack(per_block))
        source_logits, glyph_probabilities, statistics = self.source_head.aggregate(
            mixed
        )
        if return_details:
            return (
                source_logits,
                mixed,
                statistics,
                {
                    "tap_weights": weights,
                    "per_block_logits": dict(zip(blocks, per_block)),
                    "glyph_probs": glyph_probabilities,
                },
            )
        return source_logits, mixed, statistics

    def glyph_logits(self, states: Mapping[int, torch.Tensor]) -> torch.Tensor:
        return self.source_outputs(states)[1]

    def orthographic_features(
        self,
        states: Mapping[int, torch.Tensor],
        text_query: torch.Tensor,
        return_details: bool = False,
    ):
        blocks = self._block_list(self.ortho_tap_blocks)
        ordered = self._states(states, blocks)
        weights = self._mix_weights(self.ortho_tap_logits)
        outputs = []
        attention: dict[int, torch.Tensor] = {}
        for block, state in zip(blocks, ordered):
            spatial = self._spatial_state(state, block)
            if return_details:
                output, probabilities = self.orthographic_bridge(
                    text_query, spatial, return_attention=True
                )
                attention[block] = probabilities
            else:
                output = self.orthographic_bridge(text_query, spatial)
            outputs.append(output)
        mixed = _fp32_einsum("k,kbnd->bnd", weights, torch.stack(outputs))
        if return_details:
            return mixed, {"tap_weights": weights, "per_block_attention": attention}
        return mixed

    def read_features(
        self,
        states: Mapping[int, torch.Tensor],
        text_query: torch.Tensor,
        register_mask: torch.Tensor,
        return_details: bool = False,
    ):
        blocks = self._block_list(self.tap_blocks)
        ordered = self._states(states, blocks)
        weights = self._mix_weights(self.read_tap_logits)
        glyph_probabilities = self.glyph_logits(states).sigmoid().detach()
        expected_tokens = glyph_probabilities.shape[1] + 2
        bad = [
            tuple(state.shape) for state in ordered if state.shape[1] != expected_tokens
        ]
        if bad:
            raise RuntimeError(
                f"Late states must contain CLS + patches + RN ({expected_tokens} tokens): {bad}"
            )
        bridge_glyph = torch.cat(
            (glyph_probabilities, torch.ones_like(glyph_probabilities[:, :1])), dim=1
        )
        bridge_register = torch.cat(
            (register_mask, torch.zeros_like(register_mask[:, :1])), dim=1
        )
        outputs = []
        attention: dict[int, dict[str, Any]] = {}
        null_attention = []
        for block, state in zip(blocks, ordered):
            if return_details:
                output, details = self.read_bridge(
                    text_query,
                    state,
                    register_mask=bridge_register,
                    patch_textness=bridge_glyph,
                    textness_beta=self.glyph_bias_beta,
                    read_null_index=expected_tokens - 1,
                    return_attention=True,
                )
                attention[block] = details
                null_attention.append(details["read_null_attention"].mean(dim=-1))
            else:
                output = self.read_bridge(
                    text_query,
                    state,
                    register_mask=bridge_register,
                    patch_textness=bridge_glyph,
                    textness_beta=self.glyph_bias_beta,
                    read_null_index=expected_tokens - 1,
                )
            outputs.append(output)
        mixed = _fp32_einsum("k,kbnd->bnd", weights, torch.stack(outputs))
        if not return_details:
            return mixed
        mixed_null = _fp32_einsum(
            "k,kbn->bn", weights, torch.stack(null_attention)
        ).clamp(0, 1)
        return mixed, {
            "tap_weights": weights,
            "per_block": attention,
            "glyph_probs": glyph_probabilities,
            "read_attention_architecture": self.read_attention_architecture,
            "read_null_attention": mixed_null,
            "read_null_index": expected_tokens - 1,
        }

    def content_correction(
        self, states: Mapping[int, torch.Tensor], return_details: bool = False
    ):
        blocks = self._block_list(self.tap_blocks)
        ordered = self._states(states, blocks)
        weights = self._mix_weights(self.content_tap_logits)
        outputs = []
        attention: dict[int, torch.Tensor] = {}
        for block, state in zip(blocks, ordered):
            if return_details:
                output, probabilities = self.content_pool(state, return_attention=True)
                attention[block] = probabilities
            else:
                output = self.content_pool(state)
            outputs.append(output)
        mixed = _fp32_einsum("k,kbd->bd", weights, torch.stack(outputs))
        if return_details:
            return mixed, {"tap_weights": weights, "per_block_attention": attention}
        return mixed

    def calibrate_read_logits(
        self,
        raw_read_logits: torch.Tensor,
        source_logits: torch.Tensor,
        null_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.read_calibration_scale.float() * raw_read_logits.float()
        if null_mask is not None and bool(null_mask.any()):
            unavailable = -F.logsigmoid(source_logits[:, 1].detach().float())
            bonus = self.null_abstain_weight.float() * unavailable[:, None]
            output = output + bonus * null_mask.float()[None, :]
        return output

    @staticmethod
    def positive_relative_read(relative: torch.Tensor) -> torch.Tensor:
        return 0.5 * (relative + torch.sqrt(relative.square() + 1.0e-4))

    def injection_feature(self, content_image: torch.Tensor) -> torch.Tensor:
        probe = self.read_probe.float().to(content_image.device)
        if not bool(probe.abs().sum() > 0):
            return torch.zeros(content_image.shape[0], device=content_image.device)
        return content_image.float() @ probe


@dataclass
class XAttnCLIPOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits_per_image: torch.Tensor | None = None
    logits_per_text: torch.Tensor | None = None
    image_embeds: torch.Tensor | None = None
    text_embeds: torch.Tensor | None = None
    mode_ids: torch.Tensor | None = None
    null_candidate_index: int | None = None
    details: dict[str, Any] | None = None


class XAttnCLIPModel(CLIPModel):
    """Hugging Face implementation of RN + full PIECES x-attention CLIP.

    ``any`` is the robust default. ``read`` returns one additional internal null
    candidate, ``text`` forces literal reading without exposing that candidate,
    ``notext`` disables the reader, ``classic`` uses ordinary RN-backbone cosine,
    and ``none`` respects per-candidate control tokens already in ``input_ids``.
    """

    config_class = XAttnCLIPConfig
    # Match oaiclip storage: RN + PIECES parameters remain FP32; RN is cast only at insertion.
    # Transformers 5.16.1 strict applies for both fp16 and bf16 loads.
    _keep_in_fp32_modules = [
        "read_null_token",
        "read_implant",
        "hard_text_embedding",
        "null_text_embedding",
    ]
    _keep_in_fp32_modules_strict = [
        "read_null_token",
        "read_implant",
        "hard_text_embedding",
        "null_text_embedding",
    ]

    def __init__(self, config: XAttnCLIPConfig, debug: bool = False):
        super().__init__(config)
        self.read_null_token = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size)
        )
        self.hard_text_embedding = nn.Parameter(
            torch.zeros(config.text_config.hidden_size)
        )
        self.null_text_embedding = nn.Parameter(
            torch.zeros(config.text_config.hidden_size)
        )
        self.read_implant = HardTextReadImplant(config)
        self.post_init()
        if debug:
            self.debug_summary()

    def debug_summary(self) -> None:
        print(
            "[full x-attn] "
            f"image={self.config.vision_config.image_size}; "
            f"RN before B{self.config.read_null_insert_block}; "
            f"late={self.config.read_tap_blocks}; ortho={self.config.ortho_tap_blocks}; "
            f"source={self.config.source_tap_blocks}; "
            f"attention={self.config.read_attention_architecture}; "
            f"default_mode={self.config.default_mode}; correction={self.config.correction}"
        )

    @staticmethod
    def describe_modes() -> dict[str, str]:
        return {
            "any": "PIECES robust fusion (default; untagged candidate semantics).",
            "read": "Forced literal reading plus an exposed internal NO_TEXT_DETECTED candidate.",
            "text": "Forced literal reading without an exposed null candidate.",
            "notext": "Content semantics and optional correction; reader disabled.",
            "classic": "Ordinary cosine from the RN-modified backbone; no PIECES correction.",
            "none": "Respect explicit <any>/<text>/<notext> controls per candidate.",
        }

    def _eot_indices(self, input_ids: torch.Tensor) -> torch.Tensor:
        matches = input_ids.eq(self.config.eot_token_id)
        indices = matches.long().argmax(dim=-1)
        missing = ~matches.any(dim=-1)
        if missing.any():
            indices = indices.clone()
            indices[missing] = input_ids[missing].argmax(dim=-1)
        return indices

    def _pad_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        context = int(self.config.text_config.max_position_embeddings)
        if input_ids.shape[1] > context:
            raise ValueError(
                f"Token sequence length {input_ids.shape[1]} exceeds context length {context}"
            )
        if input_ids.shape[1] == context:
            return input_ids
        return F.pad(input_ids, (0, context - input_ids.shape[1]), value=0)

    def parse_text_modes(self, input_ids: torch.Tensor) -> torch.Tensor:
        has_text = input_ids.eq(self.config.hard_text_token_id).any(dim=-1)
        has_notext = input_ids.eq(self.config.no_text_token_id).any(dim=-1)
        has_any = input_ids.eq(self.config.any_text_token_id).any(dim=-1)
        conflicts = (
            has_text.to(torch.int8) + has_notext.to(torch.int8) + has_any.to(torch.int8)
        )
        if (conflicts > 1).any():
            rows = (conflicts > 1).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Conflicting <text>/<notext>/<any> controls in rows {rows}"
            )
        modes = torch.zeros(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )
        modes[has_text] = 1
        modes[has_notext] = 2
        return modes

    def _compact_control_tokens(
        self, input_ids: torch.Tensor, remove_ids: Sequence[int]
    ) -> torch.Tensor:
        output = torch.zeros_like(input_ids)
        remove = {int(value) for value in remove_ids}
        eot_indices = self._eot_indices(input_ids)
        for row in range(input_ids.shape[0]):
            values = input_ids[row, : int(eot_indices[row].item()) + 1].tolist()
            kept = [int(value) for value in values if int(value) not in remove]
            if not kept or kept[-1] != self.config.eot_token_id:
                kept.append(self.config.eot_token_id)
            kept = kept[: input_ids.shape[1]]
            if kept[-1] != self.config.eot_token_id:
                kept[-1] = self.config.eot_token_id
            output[row, : len(kept)] = torch.tensor(
                kept, dtype=input_ids.dtype, device=input_ids.device
            )
        return output

    def _insert_control(self, input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
        clean = self._compact_control_tokens(
            input_ids,
            (
                self.config.hard_text_token_id,
                self.config.no_text_token_id,
                self.config.any_text_token_id,
            ),
        )
        output = torch.zeros_like(clean)
        eot_indices = self._eot_indices(clean)
        for row in range(clean.shape[0]):
            values = clean[row, : int(eot_indices[row].item()) + 1].tolist()
            if len(values) <= 1:
                values = [self.config.sot_token_id, token_id, self.config.eot_token_id]
            else:
                values = [values[0], token_id, *values[1:]]
            values = values[: clean.shape[1]]
            if values[-1] != self.config.eot_token_id:
                values[-1] = self.config.eot_token_id
            output[row, : len(values)] = torch.tensor(
                values, dtype=input_ids.dtype, device=input_ids.device
            )
        return output

    def _null_read_tokens(self, device: torch.device) -> torch.Tensor:
        tokens = torch.zeros(
            (1, self.config.text_config.max_position_embeddings),
            dtype=torch.long,
            device=device,
        )
        tokens[0, :4] = torch.tensor(
            (
                self.config.sot_token_id,
                self.config.hard_text_token_id,
                self.config.null_text_token_id,
                self.config.eot_token_id,
            ),
            device=device,
        )
        return tokens

    def _prepare_tokens(self, input_ids: torch.Tensor, mode: str):
        if mode not in {"any", "read", "text", "notext", "none"}:
            raise ValueError(f"Unsupported routed mode {mode!r}")
        routed = input_ids
        if mode in {"read", "text"}:
            routed = self._insert_control(input_ids, self.config.hard_text_token_id)
        elif mode == "notext":
            routed = self._insert_control(input_ids, self.config.no_text_token_id)
        elif mode == "any":
            routed = self._compact_control_tokens(
                input_ids,
                (
                    self.config.hard_text_token_id,
                    self.config.no_text_token_id,
                    self.config.any_text_token_id,
                ),
            )
        if mode == "read":
            routed = torch.cat((routed, self._null_read_tokens(routed.device)), dim=0)
        modes = self.parse_text_modes(routed)
        content = self._compact_control_tokens(
            routed,
            (
                self.config.hard_text_token_id,
                self.config.no_text_token_id,
                self.config.any_text_token_id,
            ),
        )
        read = self._insert_control(content, self.config.hard_text_token_id)
        return modes, content, read

    def _encode_text_hidden(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.text_model.embeddings.token_embedding(input_ids)
        hard = self.hard_text_embedding.to(embeddings.dtype).view(1, 1, -1)
        null = self.null_text_embedding.to(embeddings.dtype).view(1, 1, -1)
        embeddings = torch.where(
            input_ids.eq(self.config.hard_text_token_id).unsqueeze(-1), hard, embeddings
        )
        embeddings = torch.where(
            input_ids.eq(self.config.null_text_token_id).unsqueeze(-1), null, embeddings
        )
        hidden = self.text_model.embeddings(inputs_embeds=embeddings)
        causal_mask = create_causal_mask(
            config=self.text_model.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
        )
        encoded = self.text_model.encoder(
            inputs_embeds=hidden, attention_mask=causal_mask, is_causal=True
        ).last_hidden_state
        post_layer_norm = self.text_model.final_layer_norm(encoded)
        eot_indices = self._eot_indices(input_ids)
        batch = torch.arange(encoded.shape[0], device=encoded.device)
        projected = self.text_projection(post_layer_norm[batch, eot_indices])
        return {
            "text_embedding": projected,
            "hidden_pre_ln": encoded,
            "hidden_post_ln": post_layer_norm,
            "eot_indices": eot_indices,
            "eot_hidden_pre_ln": encoded[batch, eot_indices],
        }

    def _vision_with_intermediates(
        self,
        pixel_values: torch.Tensor,
        *,
        interpolate_pos_encoding: bool = False,
        return_final_tokens: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        vision = self.vision_model
        hidden = vision.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )
        hidden = vision.pre_layrnorm(hidden)
        capture = set(self.read_implant.capture_block_list())
        states: dict[int, torch.Tensor] = {}
        inserted = False
        for block, layer in enumerate(vision.encoder.layers):
            if block == self.config.read_null_insert_block:
                rn = self.read_null_token.to(device=hidden.device, dtype=hidden.dtype)
                hidden = torch.cat(
                    (hidden, rn.view(1, 1, -1).expand(hidden.shape[0], 1, -1)), dim=1
                )
                inserted = True
            hidden = layer(hidden, None, **kwargs)
            if block in capture:
                states[block] = hidden
        if not inserted:
            raise RuntimeError("RN token was never inserted")
        spatial = hidden[:, 1:-1, :]
        patch_norms = spatial.detach().float().norm(dim=-1)
        register_mask = torch.zeros_like(patch_norms, dtype=torch.bool)
        minimum = max(0, int(self.config.register_min))
        maximum = int(self.config.register_max)
        for row in range(patch_norms.shape[0]):
            indices = torch.nonzero(
                patch_norms[row] > float(self.config.register_norm_threshold),
                as_tuple=False,
            ).flatten()
            if indices.numel() < minimum:
                count = min(max(1, minimum), patch_norms.shape[1])
                indices = torch.topk(patch_norms[row], k=count).indices
            elif maximum > 0 and indices.numel() > maximum:
                chosen = torch.topk(patch_norms[row, indices], k=maximum).indices
                indices = indices[chosen]
            register_mask[row, indices] = True
        base = self.visual_projection(vision.post_layernorm(hidden[:, 0, :]))
        return {
            "image_embedding": base,
            "states": states,
            "register_mask": register_mask,
            "patch_token_norms": patch_norms,
            "final_tokens": hidden if return_final_tokens else None,
            "read_null_index": hidden.shape[1] - 1,
        }

    def _content_image(
        self, image_info: Mapping[str, Any], correction: bool, return_details: bool
    ):
        if correction:
            if return_details:
                delta, details = self.read_implant.content_correction(
                    image_info["states"], return_details=True
                )
            else:
                delta = self.read_implant.content_correction(image_info["states"])
                details = None
        else:
            delta = torch.zeros_like(image_info["image_embedding"])
            details = None
        return (
            image_info["image_embedding"]
            + delta.to(image_info["image_embedding"].dtype),
            delta,
            details,
        )

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
        correction: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        enabled = self.config.correction if correction is None else bool(correction)
        image_info = self._vision_with_intermediates(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_final_tokens=True,
            **kwargs,
        )
        feature, _, _ = self._content_image(image_info, enabled, False)
        return BaseModelOutputWithPooling(
            last_hidden_state=image_info["final_tokens"], pooler_output=feature
        )

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        del attention_mask, position_ids, kwargs
        clean = self._compact_control_tokens(
            self._pad_context(input_ids),
            (self.config.no_text_token_id, self.config.any_text_token_id),
        )
        info = self._encode_text_hidden(clean)
        return BaseModelOutputWithPooling(
            last_hidden_state=info["hidden_post_ln"],
            pooler_output=info["text_embedding"],
        )

    def _classic_forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool,
        return_loss: bool,
        return_details: bool,
        **kwargs,
    ) -> XAttnCLIPOutput:
        image_info = self._vision_with_intermediates(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_final_tokens=return_details,
            **kwargs,
        )
        clean = self._compact_control_tokens(
            input_ids,
            (
                self.config.hard_text_token_id,
                self.config.no_text_token_id,
                self.config.any_text_token_id,
            ),
        )
        text_info = self._encode_text_hidden(clean)
        image = _normalize(image_info["image_embedding"])
        text = _normalize(text_info["text_embedding"])
        logits = _scaled_matmul(self.logit_scale.float().exp(), image, text.t())
        loss = self._contrastive_loss(logits) if return_loss else None
        return XAttnCLIPOutput(
            loss=loss,
            logits_per_image=logits,
            logits_per_text=logits.t(),
            image_embeds=image,
            text_embeds=text,
            mode_ids=torch.full((input_ids.shape[0],), -1, device=input_ids.device),
            details={"classic_is_rn_backbone": True, "image_info": image_info}
            if return_details
            else None,
        )

    @staticmethod
    def _contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[0] != logits.shape[1]:
            raise ValueError(
                "return_loss=True requires equal image and candidate counts"
            )
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        ) / 2

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_loss: bool | None = None,
        interpolate_pos_encoding: bool = False,
        correction: bool | None = None,
        mode: str | None = None,
        return_details: bool = False,
        pieces_fp32: bool = True,
        **kwargs,
    ) -> XAttnCLIPOutput:
        del attention_mask, position_ids
        # Per-call precision policy: ordinary model/training use keeps PIECES in
        # FP32; the benchmark can opt into ambient CUDA FP16 autocast without
        # mutating any stored parameter dtype.
        _PIECES_FORCE_FP32.set(bool(pieces_fp32))
        if input_ids is None or pixel_values is None:
            raise ValueError("Both input_ids and pixel_values are required")
        input_ids = self._pad_context(input_ids)
        selected_mode = self.config.default_mode if mode is None else str(mode).lower()
        if selected_mode == "classic":
            return self._classic_forward(
                input_ids,
                pixel_values,
                interpolate_pos_encoding,
                bool(return_loss),
                return_details,
                **kwargs,
            )
        if selected_mode not in {"any", "read", "text", "notext", "none"}:
            raise ValueError(
                f"Unknown mode {selected_mode!r}; choices={tuple(self.describe_modes())}"
            )
        enabled = self.config.correction if correction is None else bool(correction)
        modes, content_tokens, read_tokens = self._prepare_tokens(
            input_ids, selected_mode
        )
        null_candidate_index = (
            content_tokens.shape[0] - 1 if selected_mode == "read" else None
        )

        image_info = self._vision_with_intermediates(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_final_tokens=return_details,
            **kwargs,
        )
        content_image, correction_vector, content_details = self._content_image(
            image_info, enabled, return_details
        )
        content_image_norm = _normalize(content_image)
        content_text_info = self._encode_text_hidden(content_tokens)
        content_text_norm = _normalize(content_text_info["text_embedding"])
        scale = self.logit_scale.float().exp()
        content_logits = _scaled_matmul(
            scale, content_image_norm, content_text_norm.t()
        )
        logits = content_logits.clone()

        if return_details:
            source_logits, glyph_logits, source_stats, source_details = (
                self.read_implant.source_outputs(
                    image_info["states"], return_details=True
                )
            )
        else:
            source_logits, glyph_logits, source_stats = (
                self.read_implant.source_outputs(image_info["states"])
            )
            source_details = None
        glyph_probabilities = glyph_logits.sigmoid()
        source_probabilities = source_logits.sigmoid()
        source_gate = source_probabilities[:, 0] * source_probabilities[:, 1]

        read_candidate_mask = ~modes.eq(2)
        full_raw_read = torch.zeros_like(content_logits)
        full_read = torch.zeros_like(content_logits)
        full_early = torch.zeros_like(content_logits)
        full_trust = torch.zeros_like(content_logits)
        full_route = torch.zeros_like(content_logits)
        full_relative = torch.zeros_like(content_logits)
        full_contribution = torch.zeros_like(content_logits)
        full_null_attention = torch.zeros_like(content_logits)
        null_read_logits = torch.zeros(
            pixel_values.shape[0], device=pixel_values.device
        )
        read_details = None
        ortho_details = None
        read_feature = None
        read_text_embedding = None

        if bool(read_candidate_mask.any()):
            selected_read_tokens = read_tokens[read_candidate_mask]
            read_text_info = self._encode_text_hidden(selected_read_tokens)
            read_text_norm = _normalize(read_text_info["text_embedding"])
            query = read_text_info["eot_hidden_pre_ln"]
            if return_details:
                read_feature, read_details = self.read_implant.read_features(
                    image_info["states"],
                    query,
                    image_info["register_mask"],
                    return_details=True,
                )
                ortho_feature, ortho_details = self.read_implant.orthographic_features(
                    image_info["states"], query, return_details=True
                )
                full_null_attention[:, read_candidate_mask] = read_details[
                    "read_null_attention"
                ]
            else:
                read_feature = self.read_implant.read_features(
                    image_info["states"], query, image_info["register_mask"]
                )
                ortho_feature = self.read_implant.orthographic_features(
                    image_info["states"], query
                )
            read_feature_norm = _normalize(read_feature)
            ortho_feature_norm = _normalize(ortho_feature)
            raw_read = _scaled_einsum(
                "bnd,nd->bn", scale, read_feature_norm, read_text_norm
            )
            early_logits = _scaled_einsum(
                "bnd,nd->bn", scale, ortho_feature_norm, read_text_norm
            )
            null_mask = selected_read_tokens.eq(self.config.null_text_token_id).any(
                dim=-1
            )
            read_logits = self.read_implant.calibrate_read_logits(
                raw_read, source_logits, null_mask
            )

            null_text_info = self._encode_text_hidden(
                self._null_read_tokens(pixel_values.device)
            )
            null_text_norm = _normalize(null_text_info["text_embedding"])
            null_feature = self.read_implant.read_features(
                image_info["states"],
                null_text_info["eot_hidden_pre_ln"],
                image_info["register_mask"],
            )
            null_feature_norm = _normalize(null_feature[:, 0, :])
            raw_null = _scaled_einsum(
                "bd,d->b", scale, null_feature_norm, null_text_norm[0]
            )
            null_read_logits = self.read_implant.calibrate_read_logits(
                raw_null[:, None],
                source_logits,
                torch.ones(1, dtype=torch.bool, device=pixel_values.device),
            )[:, 0]

            selected_content_logits = content_logits[:, read_candidate_mask]
            selected_content_text = content_text_norm[read_candidate_mask]
            trust_gate = self.read_implant.trust_router(
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
            route_gate = trust_gate * source_gate[:, None].detach()
            relative_read = read_logits - null_read_logits[:, None]
            selected_modes = modes[read_candidate_mask]
            forced = selected_modes.eq(1)
            automatic = selected_modes.eq(0)
            selected_logits = selected_content_logits.clone()
            score_dtype = selected_logits.dtype
            selected_logits[:, forced] = read_logits[:, forced].to(score_dtype)
            if bool(automatic.any()):
                positive = self.read_implant.positive_relative_read(
                    relative_read[:, automatic]
                ).detach()
                contribution = (
                    route_gate[:, automatic]
                    * self.read_implant.auto_read_scale.float()
                    * positive
                )
                contribution = contribution.to(score_dtype)
                selected_logits[:, automatic] = (
                    selected_content_logits[:, automatic].to(score_dtype) + contribution
                )
                auto_indices = torch.nonzero(
                    read_candidate_mask, as_tuple=False
                ).flatten()[automatic]
                full_contribution[:, auto_indices] = contribution.to(
                    full_contribution.dtype
                )
            logits[:, read_candidate_mask] = selected_logits.to(logits.dtype)
            full_raw_read[:, read_candidate_mask] = raw_read.to(full_raw_read.dtype)
            full_read[:, read_candidate_mask] = read_logits.to(full_read.dtype)
            full_early[:, read_candidate_mask] = early_logits.to(full_early.dtype)
            full_trust[:, read_candidate_mask] = trust_gate.to(full_trust.dtype)
            full_route[:, read_candidate_mask] = route_gate.to(full_route.dtype)
            full_relative[:, read_candidate_mask] = relative_read.to(
                full_relative.dtype
            )
            read_text_embedding = read_text_info["text_embedding"]

        loss = self._contrastive_loss(logits) if return_loss else None
        details = None
        if return_details:
            details = {
                "content_tokens": content_tokens,
                "read_tokens": read_tokens,
                "base_image_embedding": image_info["image_embedding"],
                "content_image_embedding": content_image,
                "content_correction": correction_vector,
                "content_text_embedding": content_text_info["text_embedding"],
                "raw_read_logits": full_raw_read,
                "read_logits": full_read,
                "null_read_logits": null_read_logits,
                "relative_read_logits": full_relative,
                "early_orthographic_logits": full_early,
                "trust_gate": full_trust,
                "source_gate": source_gate,
                "route_gate": full_route,
                "auto_read_contribution": full_contribution,
                "read_null_attention": full_null_attention,
                "auto_read_scale": self.read_implant.auto_read_scale,
                "source_logits": source_logits,
                "source_stats": source_stats,
                "presence_logits": source_logits,
                "glyph_logits": glyph_logits,
                "glyph_probs": glyph_probabilities,
                "read_feature": read_feature,
                "read_text_embedding": read_text_embedding,
                "visual_states": image_info["states"],
                "visual_final_tokens": image_info["final_tokens"],
                "register_mask": image_info["register_mask"],
                "patch_token_norms": image_info["patch_token_norms"],
                "read_details": read_details,
                "orthographic_details": ortho_details,
                "content_details": content_details,
                "source_details": source_details,
            }
        return XAttnCLIPOutput(
            loss=loss,
            logits_per_image=logits,
            logits_per_text=logits.t(),
            image_embeds=content_image_norm,
            text_embeds=content_text_norm,
            mode_ids=modes,
            null_candidate_index=null_candidate_index,
            details=details,
        )
