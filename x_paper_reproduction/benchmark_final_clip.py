"""Strict loader and inference-precision helpers for final AnyText CLIP benchmarks.

This module intentionally accepts only complete ordinary ``oaiclip`` checkpoints.
Compact implant checkpoints are not full final models once the backbone has been
trained in the all-weights stage, and live GmP checkpoints are not benchmark-safe.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch


FINAL_REQUIRED_EXACT = {
    "hard_text_embedding",
    "null_text_embedding",
    "read_implant.read_tap_logits",
    "read_implant.content_tap_logits",
    "read_implant.ortho_tap_logits",
    "read_implant.source_tap_logits",
    "read_implant.auto_read_scale",
    "read_implant.null_abstain_weight",
    "read_implant.read_calibration_scale",
    "read_implant.glyph_bias_beta",
}
FINAL_REQUIRED_PREFIXES = (
    "read_implant.read_bridge.",
    "read_implant.content_pool.",
    "read_implant.orthographic_bridge.",
    "read_implant.source_head.",
    "read_implant.trust_router.",
)
LEGACY_ONLY_PREFIXES = (
    "read_implant.presence_pool.",
    "read_implant.glyph_head.",
    "read_implant.auto_router.",
)
COMPACT_FORMATS = {
    "gmp_anytext_stage1_v1",
    "gmp_anytext_stage1_v2_null_controls",
    "gmp_anytext_early_branch_v3",
}

# Final PIECES read-attention architectures supported by current oaiclip.
READ_ATTENTION_ARCHITECTURES = {"softmax", "sigmoid_mass", "sigmoid_all"}

# Architecture-identifying state keys. `sigmoid_all` must be checked first:
# its content/ortho pools also contain `sigmoid_head_bias`, while the unique
# late-reader patch/register biases distinguish it from `sigmoid_mass`.
SIGMOID_ALL_SIGNATURE = {
    "read_implant.read_bridge.sigmoid_patch_head_bias",
    "read_implant.read_bridge.sigmoid_register_head_bias",
    "read_implant.content_pool.sigmoid_head_bias",
    "read_implant.orthographic_bridge.sigmoid_head_bias",
}
SIGMOID_MASS_SIGNATURE = "read_implant.read_bridge.sigmoid_head_bias"


@dataclass(frozen=True)
class FinalCheckpointInfo:
    path: str
    checkpoint_container: str
    checkpoint_format: str
    read_attention_architecture: str
    parameter_count: int
    state_tensor_count: int
    inference_autocast: str


def torch_load_trusted(path: Union[str, Path]) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def _extract_state_dict(obj: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(obj, torch.nn.Module):
        return dict(obj.state_dict()), "full_module_pickle"
    if isinstance(obj, Mapping):
        fmt = str(obj.get("format", ""))
        if fmt in COMPACT_FORMATS or (
            isinstance(obj.get("implant_state_dict"), Mapping)
            and not isinstance(obj.get("state_dict"), Mapping)
            and not isinstance(obj.get("model_state_dict"), Mapping)
        ):
            raise RuntimeError(
                "Compact implant checkpoint rejected. It is not a complete final model "
                "after all-weights training because it omits the trained backbone. Use "
                "the exported '*__ungmp_oaiclip_fullmodel.pt' checkpoint instead."
            )
        for key in ("state_dict", "model_state_dict"):
            value = obj.get(key)
            if isinstance(value, Mapping):
                return {
                    str(name): tensor
                    for name, tensor in value.items()
                    if torch.is_tensor(tensor)
                }, f"wrapped_{key}"
        if any(torch.is_tensor(value) for value in obj.values()):
            return {
                str(name): tensor
                for name, tensor in obj.items()
                if torch.is_tensor(tensor)
            }, "raw_state_dict"
    raise RuntimeError(
        f"Unsupported checkpoint container {type(obj).__name__}; expected a complete "
        "oaiclip module or full state_dict wrapper."
    )


def _explicit_architecture(obj: Any) -> Optional[str]:
    value: Any = None
    if isinstance(obj, Mapping):
        value = obj.get("read_attention_architecture")
        metadata = obj.get("metadata")
        if value is None and isinstance(metadata, Mapping):
            value = metadata.get("read_attention_architecture")
    elif isinstance(obj, torch.nn.Module):
        value = getattr(obj, "read_attention_architecture", None)
        if value is None:
            export_info = getattr(obj, "_ungmp_export_info", None)
            if isinstance(export_info, Mapping):
                value = export_info.get("read_attention_architecture")
    if value is None:
        return None
    value = str(value)
    if value not in READ_ATTENTION_ARCHITECTURES:
        raise ValueError(
            f"Unknown checkpoint read-attention architecture: {value!r}; "
            f"supported={sorted(READ_ATTENTION_ARCHITECTURES)}"
        )
    return value


def _checkpoint_format(obj: Any) -> str:
    if isinstance(obj, Mapping):
        metadata = obj.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("format"):
            return str(metadata["format"])
        if obj.get("format"):
            return str(obj["format"])
    if isinstance(obj, torch.nn.Module):
        export_info = getattr(obj, "_ungmp_export_info", None)
        if isinstance(export_info, Mapping):
            return "ungmp_oaiclip_full_model"
    return "unknown_full_checkpoint"


def _infer_read_attention_architecture(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Infer the final PIECES attention architecture from state parameters."""
    keys = set(state_dict)

    has_sigmoid_all_patch = "read_implant.read_bridge.sigmoid_patch_head_bias" in keys
    has_sigmoid_all_register = "read_implant.read_bridge.sigmoid_register_head_bias" in keys
    has_sigmoid_mass = SIGMOID_MASS_SIGNATURE in keys

    if has_sigmoid_all_patch or has_sigmoid_all_register:
        missing = sorted(key for key in SIGMOID_ALL_SIGNATURE if key not in keys)
        if missing:
            raise RuntimeError(
                "Checkpoint has a partial sigmoid_all PIECES signature. "
                f"Missing architecture parameters: {missing}"
            )
        if has_sigmoid_mass:
            raise RuntimeError(
                "Checkpoint contains both sigmoid_all and sigmoid_mass late-reader "
                "parameters; refusing ambiguous final checkpoint."
            )
        return "sigmoid_all"

    if has_sigmoid_mass:
        return "sigmoid_mass"

    return "softmax"


def validate_final_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    explicit_architecture: Optional[str] = None,
) -> str:
    keys = set(state_dict)
    gmp_keys = sorted(
        key for key in keys if key.endswith(".theta") or key.endswith(".r")
    )
    if gmp_keys:
        raise RuntimeError(
            "Live GmP checkpoint rejected. Export an ordinary oaiclip full model first. "
            f"Example GmP keys: {gmp_keys[:8]}"
        )

    legacy_keys = sorted(
        key for key in keys if any(key.startswith(prefix) for prefix in LEGACY_ONLY_PREFIXES)
    )
    if legacy_keys:
        raise RuntimeError(
            "Legacy hard-text implant rejected; these benchmarks require the final "
            "source/orthography/trust architecture. "
            f"Example legacy keys: {legacy_keys[:8]}"
        )

    missing_exact = sorted(FINAL_REQUIRED_EXACT - keys)
    missing_prefixes = [
        prefix for prefix in FINAL_REQUIRED_PREFIXES
        if not any(key.startswith(prefix) for key in keys)
    ]
    if missing_exact or missing_prefixes:
        raise RuntimeError(
            "Checkpoint is not a complete final AnyText architecture. "
            f"Missing exact keys={missing_exact}; missing components={missing_prefixes}"
        )

    inferred = _infer_read_attention_architecture(state_dict)
    if explicit_architecture is not None and explicit_architecture != inferred:
        raise RuntimeError(
            "Checkpoint architecture metadata disagrees with its parameters: "
            f"metadata={explicit_architecture!r}, inferred={inferred!r}"
        )
    return inferred


def native_bf16_supported(device: Union[str, torch.device]) -> bool:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return bool(torch.cuda.is_bf16_supported())


def resolve_inference_autocast(
    device: Union[str, torch.device], requested: str = "auto"
) -> Optional[torch.dtype]:
    requested = str(requested).lower()
    resolved = torch.device(device)
    if requested in {"none", "fp32"}:
        return None
    if resolved.type != "cuda":
        if requested in {"bf16", "fp16"}:
            raise RuntimeError(f"{requested} CUDA autocast requested on {resolved.type}")
        return None
    if requested == "auto":
        return torch.bfloat16 if native_bf16_supported(resolved) else torch.float16
    if requested == "bf16":
        if not native_bf16_supported(resolved):
            raise RuntimeError(
                f"Native BF16 is unavailable on {torch.cuda.get_device_name(resolved)}"
            )
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    raise ValueError(f"Unknown inference autocast policy: {requested!r}")


def inference_autocast(
    device: Union[str, torch.device], requested: str = "auto"
):
    resolved = torch.device(device)
    dtype = resolve_inference_autocast(resolved, requested)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def autocast_name(device: Union[str, torch.device], requested: str = "auto") -> str:
    dtype = resolve_inference_autocast(device, requested)
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    return "fp32"


def assert_finite_forward_details(
    details: Mapping[str, Any], *, context: str
) -> None:
    critical = (
        "logits_per_image",
        "raw_read_logits",
        "read_logits",
        "null_read_logits",
        "relative_read_logits",
        "trust_gate",
        "source_logits",
        "source_gate",
        "route_gate",
        "glyph_probs",
    )
    failures = []
    for key in critical:
        value = details.get(key)
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            nonfinite = int((~torch.isfinite(value)).sum().item())
            failures.append(f"{key}:{nonfinite}/{value.numel()}")
    if failures:
        raise FloatingPointError(
            f"Non-finite final-model output during {context}: " + ", ".join(failures)
        )


def _build_model_from_loaded(
    clip_module: Any,
    loaded: Any,
    state_dict: Mapping[str, torch.Tensor],
    architecture: str,
    device: Union[str, torch.device],
):
    model_module = importlib.import_module(f"{clip_module.__name__}.model")
    clip_submodule = importlib.import_module(f"{clip_module.__name__}.clip")
    build_model = getattr(model_module, "build_model")
    model = build_model(
        dict(state_dict),
        hard_text_token_id=int(clip_submodule.TEXT_TOKEN_ID),
        no_text_token_id=int(clip_submodule.NO_TEXT_TOKEN_ID),
        any_text_token_id=int(clip_submodule.ANY_TOKEN_ID),
        null_text_token_id=int(clip_submodule.NULL_TOKEN_ID),
        eot_token_id=int(clip_submodule.EOT_TOKEN_ID),
        read_attention_architecture=architecture,
    ).to(device=device, dtype=torch.float32).eval()
    setter = getattr(clip_submodule, "_set_default_context_length_from_model", None)
    if callable(setter):
        setter(model)
    preprocess = getattr(clip_submodule, "_transform")(int(model.visual.input_resolution))
    return model, preprocess


def load_final_clip_checkpoint(
    clip_module: Any,
    model_path: Union[str, Path],
    *,
    device: Union[str, torch.device],
    inference_dtype: str = "auto",
):
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Final checkpoint not found: {path}")

    loaded = torch_load_trusted(path)
    state_dict, container = _extract_state_dict(loaded)
    explicit = _explicit_architecture(loaded)
    architecture = validate_final_state_dict(state_dict, explicit)
    model, preprocess = _build_model_from_loaded(
        clip_module, loaded, state_dict, architecture, device
    )

    # Model-level checks catch class/API drift even if state keys looked plausible.
    if not hasattr(model, "forward_modes"):
        raise AttributeError("Final oaiclip model does not expose forward_modes().")
    if model.read_implant is None:
        raise AttributeError("Final oaiclip model has no read_implant.")
    if str(model.read_attention_architecture) != architecture:
        raise RuntimeError(
            "Loaded model architecture changed during construction: "
            f"checkpoint={architecture}, model={model.read_attention_architecture}"
        )
    for attr in (
        "hard_text_token_id",
        "no_text_token_id",
        "any_text_token_id",
        "null_text_token_id",
    ):
        if not hasattr(model, attr):
            raise AttributeError(f"Final model lacks required control ID {attr}.")

    info = FinalCheckpointInfo(
        path=str(path.resolve()),
        checkpoint_container=container,
        checkpoint_format=_checkpoint_format(loaded),
        read_attention_architecture=architecture,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        state_tensor_count=len(state_dict),
        inference_autocast=autocast_name(device, inference_dtype),
    )
    return model, preprocess, info


def print_final_checkpoint_info(info: FinalCheckpointInfo) -> None:
    print(
        "[model] final checkpoint: "
        f"container={info.checkpoint_container} format={info.checkpoint_format} "
        f"architecture={info.read_attention_architecture} "
        f"parameters={info.parameter_count:,} autocast={info.inference_autocast}"
    )
