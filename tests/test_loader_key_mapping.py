from __future__ import annotations

from pathlib import Path

import torch

from utils_clip_loader.clip_anything_to_openai import (
    _convert_hf_clip_state_dict_to_openai,
    _detect_model_family_from_state,
    _is_standalone_rn_state,
    _restore_fp32_custom_islands,
)
from benchmark_utils.models import (
    DEFAULT_MODEL_ALIAS,
    DEFAULT_MODEL_PATH,
    ModelSpec,
    model_display_name,
    model_file_token,
)


def _affine(state, prefix: str, out_features: int, in_features: int) -> None:
    state[prefix + ".weight"] = torch.randn(out_features, in_features)
    state[prefix + ".bias"] = torch.randn(out_features)


def _tiny_hf_clip_state() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "text_model.embeddings.token_embedding.weight": torch.randn(10, 4),
        "text_model.embeddings.position_embedding.weight": torch.randn(3, 4),
        "vision_model.embeddings.patch_embedding.weight": torch.randn(4, 3, 2, 2),
        "vision_model.embeddings.class_embedding": torch.randn(4),
        "vision_model.embeddings.position_embedding.weight": torch.randn(5, 4),
        "text_projection.weight": torch.randn(2, 4),
        "visual_projection.weight": torch.randn(2, 4),
        "logit_scale": torch.tensor(2.5),
    }
    for prefix in (
        "text_model.final_layer_norm",
        "vision_model.pre_layrnorm",
        "vision_model.post_layernorm",
    ):
        _affine(state, prefix, 4, 4)
    for tower in ("text_model", "vision_model"):
        layer = f"{tower}.encoder.layers.0"
        _affine(state, layer + ".layer_norm1", 4, 4)
        _affine(state, layer + ".layer_norm2", 4, 4)
        _affine(state, layer + ".mlp.fc1", 8, 4)
        _affine(state, layer + ".mlp.fc2", 4, 8)
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            _affine(state, layer + f".self_attn.{projection}", 4, 4)
    return state


def test_full_xattn_custom_tensors_survive_reverse_conversion() -> None:
    state = _tiny_hf_clip_state()
    state["read_null_token"] = torch.randn(4)
    state["hard_text_embedding"] = torch.randn(4)
    state["null_text_embedding"] = torch.randn(4)
    state["read_implant.source_head.patch_out.weight"] = torch.randn(1, 4)
    converted = _convert_hf_clip_state_dict_to_openai(state)
    assert torch.equal(converted["visual.read_null_token"], state["read_null_token"])
    assert torch.equal(converted["hard_text_embedding"], state["hard_text_embedding"])
    assert "read_implant.source_head.patch_out.weight" in converted
    assert converted["transformer.resblocks.0.attn.in_proj_weight"].shape == (12, 4)


def test_correction_only_prefixes_are_reconstructed() -> None:
    state = _tiny_hf_clip_state()
    state["read_null_token"] = torch.randn(4)
    state["content_tap_logits"] = torch.randn(2)
    state["content_pool.query"] = torch.randn(2, 4)
    converted = _convert_hf_clip_state_dict_to_openai(state)
    assert "read_implant.content_tap_logits" in converted
    assert "read_implant.content_pool.query" in converted


def test_model_family_detection() -> None:
    tensor = torch.zeros(1)
    assert _detect_model_family_from_state({"read_implant.trust_router.fc1.weight": tensor}) == "full_xattn"
    assert _detect_model_family_from_state({"read_implant.read_bridge.q_proj.weight": tensor}) == "full_xattn"
    assert _detect_model_family_from_state({"content_tap_logits": tensor}) == "rn_correction"
    assert _detect_model_family_from_state({"read_implant.content_tap_logits": tensor}) == "rn_correction"
    assert _detect_model_family_from_state({"visual.read_null_token": tensor}) == "rn_token"
    assert _detect_model_family_from_state({"visual.conv1.weight": tensor}) == "vanilla"
    assert _detect_model_family_from_state(
        {"visual.read_null_token": tensor}, {"model_type": "xattn_clip"}
    ) == "rn_token"
    assert _is_standalone_rn_state({"read_null_token": tensor})
    assert not _is_standalone_rn_state(
        {"read_null_token": tensor, "visual.conv1.weight": tensor}
    )


def test_pickle_custom_island_is_fp32_without_widening_backbone() -> None:
    class _PickledModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.Linear(2, 2).half()
            self.read_implant = torch.nn.Linear(2, 2).half()
            self.hard_text_embedding = torch.nn.Parameter(torch.zeros(2).half())
            self.null_text_embedding = torch.nn.Parameter(torch.zeros(2).half())

    model = _PickledModel()
    _restore_fp32_custom_islands(model)

    assert model.visual.weight.dtype == torch.float16
    assert model.read_implant.weight.dtype == torch.float32
    assert model.hard_text_embedding.dtype == torch.float32
    assert model.null_text_embedding.dtype == torch.float32


def test_public_benchmark_default_model_helpers() -> None:
    spec = ModelSpec()
    assert spec.alias == DEFAULT_MODEL_ALIAS == "CLIP-xAttn-ModeMUX"
    assert spec.path == DEFAULT_MODEL_PATH == "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
    assert spec.base_model_or_path is None
    assert model_display_name(spec) == DEFAULT_MODEL_PATH
    assert model_file_token(spec) == "zer0int_CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"

    local = ModelSpec("local", r"models\full_xattn_model")
    assert model_display_name(local) == "full_xattn_model"
