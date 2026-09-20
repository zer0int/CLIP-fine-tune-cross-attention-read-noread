from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import (
    CLIPConfig,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizerFast,
    CLIPVisionConfig,
)

try:
    from .checkpoint_spec import CheckpointSpec
    from .configuration_rn_clip import RNCLIPConfig
    from .configuration_xattn_clip import XAttnCLIPConfig
    from .modeling_rn_clip import RNCLIPModel
    from .modeling_xattn_clip import XAttnCLIPModel
except ImportError:  # script execution from the package directory
    from checkpoint_spec import CheckpointSpec
    from configuration_rn_clip import RNCLIPConfig
    from configuration_xattn_clip import XAttnCLIPConfig
    from modeling_rn_clip import RNCLIPModel
    from modeling_xattn_clip import XAttnCLIPModel


VANILLA_CLIP_VOCAB_SIZE = 49_408
TOKENIZER_ASSET_NAMES = (
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def materialize_gmp_weights(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float]]]:
    """Reconstruct ordinary weights from any checkpoint ``theta``/``r`` pairs."""
    output = dict(state)
    report: dict[str, dict[str, float]] = {}
    for theta_key in sorted(key for key in state if key.endswith(".theta")):
        base = theta_key[: -len("theta")]
        radius_key = base + "r"
        weight_key = base + "weight"
        if radius_key not in state:
            raise KeyError(f"Found {theta_key} without matching {radius_key}")
        if weight_key in state:
            raise KeyError(
                f"Checkpoint contains both {weight_key} and ({theta_key}, {radius_key}); "
                "cannot determine a single authoritative tensor"
            )
        theta = state[theta_key]
        radius = state[radius_key]
        if theta.ndim != 2 or radius.numel() != theta.shape[0]:
            raise ValueError(
                f"Invalid GmP pair: {theta_key}={tuple(theta.shape)}, "
                f"{radius_key}={tuple(radius.shape)}"
            )
        weight = radius.reshape(-1, 1).to(theta.dtype) * F.normalize(theta, p=2, dim=1)
        weight = weight.contiguous()
        output[weight_key] = weight
        del output[theta_key]
        del output[radius_key]
        norms = weight.float().norm(dim=1)
        report[weight_key] = {
            "weight_norm_mean": float(norms.mean()),
            "weight_norm_min": float(norms.min()),
            "weight_norm_max": float(norms.max()),
        }
    leftovers = [key for key in output if key.endswith((".theta", ".r"))]
    if leftovers:
        raise RuntimeError(f"Unconverted GmP tensors remain: {leftovers[:20]}")
    return output, report


def make_text_config(spec: CheckpointSpec) -> CLIPTextConfig:
    if spec.source_vocab_size < VANILLA_CLIP_VOCAB_SIZE:
        raise ValueError(
            f"Checkpoint vocabulary {spec.source_vocab_size} is smaller than vanilla CLIP"
        )
    return CLIPTextConfig(
        vocab_size=VANILLA_CLIP_VOCAB_SIZE,
        hidden_size=spec.text_width,
        intermediate_size=spec.text_width * 4,
        projection_dim=spec.projection_dim,
        num_hidden_layers=spec.text_layers,
        num_attention_heads=spec.text_heads,
        max_position_embeddings=spec.context_length,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=spec.eot_token_id,
    )


def make_xattn_text_config(spec: CheckpointSpec) -> CLIPTextConfig:
    config = make_text_config(spec)
    config.vocab_size = spec.source_vocab_size
    return config


def make_vision_config(spec: CheckpointSpec) -> CLIPVisionConfig:
    return CLIPVisionConfig(
        hidden_size=spec.vision_width,
        intermediate_size=spec.vision_width * 4,
        projection_dim=spec.projection_dim,
        num_hidden_layers=spec.vision_layers,
        num_attention_heads=spec.vision_heads,
        num_channels=3,
        image_size=spec.image_size,
        patch_size=spec.patch_size,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
    )


def make_clip_config(spec: CheckpointSpec, *, custom_correction: bool) -> CLIPConfig:
    text = make_text_config(spec)
    vision = make_vision_config(spec)
    common: dict[str, Any] = dict(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        projection_dim=spec.projection_dim,
        logit_scale_init_value=2.6592,
    )
    if custom_correction:
        return RNCLIPConfig(
            **common,
            read_null_insert_block=spec.read_null_insert_block,
            read_tap_blocks=spec.read_tap_blocks,
            read_attention_architecture=spec.read_attention_architecture,
            read_bridge_width=spec.read_bridge_width,
            read_bridge_heads=spec.read_bridge_heads,
            correction=True,
            register_norm_threshold=spec.register_norm_threshold,
            register_min=spec.register_min,
            register_max=spec.register_max,
            source_checkpoint_fingerprint=spec.architecture_fingerprint,
        )
    config = CLIPConfig(**common)
    # Stock CLIP ignores these fields; the explicit adapter and downstream tools use them.
    config.read_null_insert_block = spec.read_null_insert_block
    config.read_null_enabled = True
    config.register_norm_threshold = spec.register_norm_threshold
    config.register_min = spec.register_min
    config.register_max = spec.register_max
    config.source_checkpoint_fingerprint = spec.architecture_fingerprint
    return config


def make_xattn_config(spec: CheckpointSpec) -> XAttnCLIPConfig:
    """Build the full config exclusively from checkpoint-defined architecture."""
    return XAttnCLIPConfig(
        text_config=make_xattn_text_config(spec).to_dict(),
        vision_config=make_vision_config(spec).to_dict(),
        projection_dim=spec.projection_dim,
        logit_scale_init_value=2.6592,
        read_null_insert_block=spec.read_null_insert_block,
        read_tap_blocks=spec.read_tap_blocks,
        ortho_tap_blocks=spec.ortho_tap_blocks,
        source_tap_blocks=spec.source_tap_blocks,
        read_attention_architecture=spec.read_attention_architecture,
        read_bridge_width=spec.read_bridge_width,
        read_bridge_heads=spec.read_bridge_heads,
        early_expanded_width=spec.early_expanded_width,
        source_hidden_width=spec.source_hidden_width,
        trust_hidden_width=spec.trust_hidden_width,
        hard_text_token_id=spec.hard_text_token_id,
        no_text_token_id=spec.no_text_token_id,
        any_text_token_id=spec.any_text_token_id,
        null_text_token_id=spec.null_text_token_id,
        eot_token_id=spec.eot_token_id,
        correction=True,
        default_mode="any",
        register_norm_threshold=spec.register_norm_threshold,
        register_min=spec.register_min,
        register_max=spec.register_max,
        source_checkpoint_fingerprint=spec.architecture_fingerprint,
    )


def _required(state: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        return state[key]
    except KeyError as error:
        raise KeyError(
            f"Checkpoint is missing required conversion tensor {key!r}"
        ) from error


def _set_parameter(
    parameter: torch.nn.Parameter, tensor: torch.Tensor, label: str
) -> None:
    """Copy a checkpoint tensor into an HF parameter using FP32 storage."""
    if tuple(parameter.shape) != tuple(tensor.shape):
        raise ValueError(
            f"Shape mismatch for {label}: HF={tuple(parameter.shape)}, "
            f"checkpoint={tuple(tensor.shape)}"
        )
    source = tensor.detach().contiguous().to(parameter.device)
    if source.is_floating_point():
        source = source.float()
        if parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    with torch.no_grad():
        parameter.copy_(source.to(dtype=parameter.dtype))


def _copy_affine(
    module: torch.nn.Module, state: Mapping[str, torch.Tensor], prefix: str
) -> None:
    _set_parameter(
        module.weight, _required(state, prefix + ".weight"), prefix + ".weight"
    )
    if getattr(module, "bias", None) is not None:
        _set_parameter(
            module.bias, _required(state, prefix + ".bias"), prefix + ".bias"
        )


def _copy_attention(target, state: Mapping[str, torch.Tensor], prefix: str) -> None:
    weight = _required(state, prefix + ".in_proj_weight")
    bias = _required(state, prefix + ".in_proj_bias")
    if weight.shape[0] % 3 or bias.shape[0] % 3:
        raise ValueError(f"Invalid packed QKV tensors under {prefix}")
    for name, source_weight, source_bias in zip(
        ("q_proj", "k_proj", "v_proj"), weight.chunk(3), bias.chunk(3)
    ):
        _set_parameter(
            getattr(target, name).weight, source_weight, f"{prefix}.{name}.weight"
        )
        _set_parameter(getattr(target, name).bias, source_bias, f"{prefix}.{name}.bias")
    _copy_affine(target.out_proj, state, prefix + ".out_proj")


def _copy_layers(target_layers, state: Mapping[str, torch.Tensor], prefix: str) -> None:
    for index, layer in enumerate(target_layers):
        source = f"{prefix}{index}"
        _copy_affine(layer.layer_norm1, state, source + ".ln_1")
        _copy_affine(layer.layer_norm2, state, source + ".ln_2")
        _copy_attention(layer.self_attn, state, source + ".attn")
        _copy_affine(layer.mlp.fc1, state, source + ".mlp.c_fc")
        _copy_affine(layer.mlp.fc2, state, source + ".mlp.c_proj")


def copy_text_backbone(target, state: Mapping[str, torch.Tensor]) -> None:
    row_count = int(target.embeddings.token_embedding.weight.shape[0])
    token_weight = _required(state, "token_embedding.weight")[:row_count]
    if token_weight.shape[0] != row_count:
        raise ValueError(
            f"Checkpoint has {token_weight.shape[0]} token rows, target requires {row_count}"
        )
    _set_parameter(
        target.embeddings.token_embedding.weight,
        token_weight,
        f"token_embedding.weight[:{row_count}]",
    )
    _set_parameter(
        target.embeddings.position_embedding.weight,
        _required(state, "positional_embedding"),
        "positional_embedding",
    )
    _copy_affine(target.final_layer_norm, state, "ln_final")
    _copy_layers(target.encoder.layers, state, "transformer.resblocks.")


def copy_clip_backbone(target: CLIPModel, state: Mapping[str, torch.Tensor]) -> None:
    copy_text_backbone(target.text_model, state)
    vision = target.vision_model
    _set_parameter(
        vision.embeddings.patch_embedding.weight,
        _required(state, "visual.conv1.weight"),
        "visual.conv1.weight",
    )
    _set_parameter(
        vision.embeddings.class_embedding,
        _required(state, "visual.class_embedding"),
        "visual.class_embedding",
    )
    _set_parameter(
        vision.embeddings.position_embedding.weight,
        _required(state, "visual.positional_embedding"),
        "visual.positional_embedding",
    )
    _copy_affine(vision.pre_layrnorm, state, "visual.ln_pre")
    _copy_affine(vision.post_layernorm, state, "visual.ln_post")
    _copy_layers(vision.encoder.layers, state, "visual.transformer.resblocks.")
    _set_parameter(
        target.text_projection.weight,
        _required(state, "text_projection").T.contiguous(),
        "text_projection.T",
    )
    _set_parameter(
        target.visual_projection.weight,
        _required(state, "visual.proj").T.contiguous(),
        "visual.proj.T",
    )
    _set_parameter(target.logit_scale, _required(state, "logit_scale"), "logit_scale")


def copy_correction(
    target: RNCLIPModel, state: Mapping[str, torch.Tensor]
) -> None:
    _set_parameter(
        target.read_null_token,
        _required(state, "visual.read_null_token").reshape(-1),
        "visual.read_null_token",
    )
    _set_parameter(
        target.content_tap_logits,
        _required(state, "read_implant.content_tap_logits"),
        "read_implant.content_tap_logits",
    )
    source_prefix = "read_implant.content_pool"
    _set_parameter(
        target.content_pool.query,
        _required(state, source_prefix + ".query"),
        source_prefix + ".query",
    )
    for name in ("vision_ln", "k_proj", "v_proj", "out_proj"):
        _copy_affine(
            getattr(target.content_pool, name), state, source_prefix + "." + name
        )
    if target.content_pool.sigmoid_head_bias is not None:
        _set_parameter(
            target.content_pool.sigmoid_head_bias,
            _required(state, source_prefix + ".sigmoid_head_bias"),
            source_prefix + ".sigmoid_head_bias",
        )

def _set_buffer(
    module: torch.nn.Module, name: str, tensor: torch.Tensor, label: str
) -> None:
    current = getattr(module, name)
    if tuple(current.shape) != tuple(tensor.shape):
        raise ValueError(
            f"Shape mismatch for {label}: HF={tuple(current.shape)}, "
            f"checkpoint={tuple(tensor.shape)}"
        )
    value = tensor.detach().clone().contiguous().to(current.device)
    if value.is_floating_point():
        value = value.float()
    setattr(module, name, value)


def copy_xattn(
    target: XAttnCLIPModel, state: Mapping[str, torch.Tensor]
) -> None:
    """Copy the full PIECES state into FP32 HF parameters and buffers."""
    for target_name, source_name in (
        ("read_null_token", "visual.read_null_token"),
        ("hard_text_embedding", "hard_text_embedding"),
        ("null_text_embedding", "null_text_embedding"),
    ):
        _set_parameter(
            getattr(target, target_name),
            _required(state, source_name).reshape(getattr(target, target_name).shape),
            source_name,
        )

    source_prefix = "read_implant."
    target_parameters = dict(target.read_implant.named_parameters())
    target_buffers = dict(target.read_implant.named_buffers())
    expected = {source_prefix + name for name in (*target_parameters, *target_buffers)}
    available = {key for key in state if key.startswith(source_prefix)}
    missing = sorted(expected - available)
    unexpected = sorted(available - expected)
    if missing or unexpected:
        raise ValueError(
            "Checkpoint x-attention state differs from the inferred HF architecture: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    for name, parameter in target_parameters.items():
        _set_parameter(parameter, state[source_prefix + name], source_prefix + name)
    for name in target_buffers:
        _set_buffer(
            target.read_implant,
            name,
            state[source_prefix + name],
            source_prefix + name,
        )


def save_processor(
    output_dir: Path,
    spec: CheckpointSpec,
    asset_dir: Path,
    *,
    xattn_special_tokens: bool = False,
) -> None:
    tokenizer = CLIPTokenizerFast.from_pretrained(str(asset_dir), local_files_only=True)
    if xattn_special_tokens:
        controls = ["<text>", "<notext>", "<any>", "<null>"]
        tokenizer.add_special_tokens({"additional_special_tokens": controls})
        actual = [tokenizer.convert_tokens_to_ids(token) for token in controls]
        expected = [
            spec.hard_text_token_id,
            spec.no_text_token_id,
            spec.any_text_token_id,
            spec.null_text_token_id,
        ]
        if actual != expected or len(tokenizer) != spec.source_vocab_size:
            raise ValueError(
                "Tokenizer control-token layout conflicts with checkpoint: "
                f"actual_ids={actual}, expected_ids={expected}, "
                f"tokenizer_size={len(tokenizer)}, checkpoint_size={spec.source_vocab_size}"
            )
    image_processor = CLIPImageProcessor(
        do_resize=True,
        size={"shortest_edge": spec.image_size},
        resample=3,
        do_center_crop=True,
        crop_size={"height": spec.image_size, "width": spec.image_size},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer).save_pretrained(
        output_dir
    )


def save_rn_token(
    output_dir: Path, state: Mapping[str, torch.Tensor], spec: CheckpointSpec
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "read_null_token.safetensors"
    token = (
        _required(state, "visual.read_null_token")
        .reshape(-1)
        .detach()
        .float()
        .cpu()
        .contiguous()
    )
    save_file(
        {"read_null_token": token},
        str(path),
        metadata={
            "format": "pt",
            "architecture": "ViT-L/14",
            "image_size": str(spec.image_size),
            "vision_width": str(spec.vision_width),
            "read_null_insert_block": str(spec.read_null_insert_block),
            "source_checkpoint_fingerprint": spec.architecture_fingerprint,
        },
    )
    return path


def save_vanilla_text_encoders(
    output_dir: Path,
    state: Mapping[str, torch.Tensor],
    spec: CheckpointSpec,
    asset_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_text_config(spec)
    plain = CLIPTextModel(config).eval()
    projected = CLIPTextModelWithProjection(config).eval()
    copy_text_backbone(plain, state)
    copy_text_backbone(projected.text_model, state)
    _set_parameter(
        projected.text_projection.weight,
        _required(state, "text_projection").T.contiguous(),
        "text_projection.T",
    )

    plain_path = (
        output_dir / f"{spec.checkpoint_name}_CLIPTextModel_for_t2i_genAI.safetensors"
    )
    projected_path = (
        output_dir / f"{spec.checkpoint_name}_CLIPTextModel_with_projection.safetensors"
    )
    save_file(
        {
            key: (value.float() if value.is_floating_point() else value)
            .detach()
            .cpu()
            .contiguous()
            for key, value in plain.state_dict().items()
        },
        str(plain_path),
        metadata={"format": "pt", "class": "CLIPTextModel"},
    )
    save_file(
        {
            key: (value.float() if value.is_floating_point() else value)
            .detach()
            .cpu()
            .contiguous()
            for key, value in projected.state_dict().items()
        },
        str(projected_path),
        metadata={"format": "pt", "class": "CLIPTextModelWithProjection"},
    )
    (output_dir / f"{plain_path.stem}_config.json").write_text(
        config.to_json_string(), encoding="utf-8"
    )
    (output_dir / f"{projected_path.stem}_config.json").write_text(
        config.to_json_string(), encoding="utf-8"
    )
    for name in TOKENIZER_ASSET_NAMES:
        shutil.copy2(asset_dir / name, output_dir / name)
    return [plain_path, projected_path]


def _save_model_fp32(model: torch.nn.Module, output_dir: Path) -> None:
    """Save every floating model tensor in FP32, independent of source checkpoint dtype."""
    model.float()
    bad = [
        f"{name}:{tensor.dtype}"
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    ]
    if bad:
        raise RuntimeError(
            "Internal conversion error: non-FP32 floating tensors remain before save: "
            + ", ".join(bad[:20])
        )
    model.save_pretrained(output_dir, safe_serialization=True)


def export_stock_rn_model(
    output_dir: Path,
    state: Mapping[str, torch.Tensor],
    spec: CheckpointSpec,
    asset_dir: Path,
) -> CLIPModel:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = CLIPModel(make_clip_config(spec, custom_correction=False)).eval()
    copy_clip_backbone(model, state)
    _save_model_fp32(model, output_dir)
    save_processor(output_dir, spec, asset_dir)
    save_rn_token(output_dir, state, spec)
    return model


def export_correction_model(
    output_dir: Path,
    state: Mapping[str, torch.Tensor],
    spec: CheckpointSpec,
    asset_dir: Path,
) -> RNCLIPModel:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = RNCLIPModel(make_clip_config(spec, custom_correction=True)).eval()
    copy_clip_backbone(model, state)
    copy_correction(model, state)
    _save_model_fp32(model, output_dir)
    save_processor(output_dir, spec, asset_dir)
    return model


def export_full_xattn_model(
    output_dir: Path,
    state: Mapping[str, torch.Tensor],
    spec: CheckpointSpec,
    asset_dir: Path,
) -> XAttnCLIPModel:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = XAttnCLIPModel(make_xattn_config(spec)).eval()
    copy_clip_backbone(model, state)
    copy_xattn(model, state)
    _save_model_fp32(model, output_dir)
    save_processor(output_dir, spec, asset_dir, xattn_special_tokens=True)
    return model


def write_conversion_manifest(
    output_dir: Path,
    spec: CheckpointSpec,
    json_mismatches: list[str],
    gmp_report: Mapping[str, Any],
) -> None:
    payload = {
        "authority": "pickle checkpoint tensors and serialized model attributes",
        "json_policy": "warn on mismatch; checkpoint wins",
        "checkpoint_spec": spec.to_dict(),
        "json_mismatches": json_mismatches,
        "gmp_reconstruction": gmp_report,
        "dtype_policy": (
            "all exported floating tensors are saved as FP32; at HF load time the "
            "CLIP backbone follows the requested dtype while custom RN/correction/PIECES "
            "parameters are kept in FP32"
        ),
        "transformers_target": "==5.16.1",
    }
    (output_dir / "checkpoint_authority.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
