from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


DEFAULT_REGISTER_THRESHOLD = 60.0
DEFAULT_REGISTER_MIN = 1
DEFAULT_REGISTER_MAX = 0
SUPPORTED_IMAGE_SIZES = (224, 336)


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_name: str
    architecture_fingerprint: str
    image_size: int
    patch_size: int
    projection_dim: int
    vision_width: int
    vision_layers: int
    vision_heads: int
    text_width: int
    text_layers: int
    text_heads: int
    context_length: int
    source_vocab_size: int
    eot_token_id: int
    hard_text_token_id: int
    no_text_token_id: int
    any_text_token_id: int
    null_text_token_id: int
    read_null_enabled: bool
    read_null_insert_block: int
    read_attention_architecture: str
    read_tap_blocks: tuple[int, ...]
    ortho_tap_blocks: tuple[int, ...]
    source_tap_blocks: tuple[int, ...]
    read_bridge_width: int
    read_bridge_heads: int
    early_expanded_width: int
    source_hidden_width: int
    trust_hidden_width: int
    register_norm_threshold: float
    register_min: int
    register_max: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("read_tap_blocks", "ortho_tap_blocks", "source_tap_blocks"):
            result[key] = list(result[key])
        return result


def load_trusted_checkpoint(path: str | Path) -> Any:
    """Load a local checkpoint that the caller explicitly trusts."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        return torch.jit.load(str(checkpoint_path), map_location="cpu").eval()
    except Exception:
        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location="cpu")


def _strip_common_prefixes(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        output[key] = value.detach().cpu()
    return output


def extract_state_dict(loaded: Any) -> dict[str, torch.Tensor]:
    """Extract the authoritative tensor state from common checkpoint containers."""
    if isinstance(loaded, nn.Module):
        return _strip_common_prefixes(loaded.state_dict())
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Unsupported checkpoint object: {type(loaded)!r}")

    for key in ("state_dict", "model_state_dict"):
        value = loaded.get(key)
        if isinstance(value, Mapping):
            return _strip_common_prefixes(value)

    model = loaded.get("model")
    if isinstance(model, nn.Module):
        return _strip_common_prefixes(model.state_dict())
    if isinstance(model, Mapping):
        return _strip_common_prefixes(model)

    if "token_embedding.weight" in loaded or "visual.conv1.weight" in loaded:
        return _strip_common_prefixes(loaded)
    raise ValueError(
        f"No full model state_dict found; top-level keys={list(loaded)[:20]}"
    )


def _model_object(loaded: Any) -> nn.Module | None:
    if isinstance(loaded, nn.Module):
        return loaded
    if isinstance(loaded, Mapping) and isinstance(loaded.get("model"), nn.Module):
        return loaded["model"]
    return None


def _explicit_read_architecture(loaded: Any) -> str | None:
    value: Any = None
    if isinstance(loaded, nn.Module):
        value = getattr(loaded, "read_attention_architecture", None)
        info = getattr(loaded, "_ungmp_export_info", None)
        if value is None and isinstance(info, Mapping):
            value = info.get("read_attention_architecture")
    elif isinstance(loaded, Mapping):
        value = loaded.get("read_attention_architecture")
        for container_name in ("metadata", "args"):
            container = loaded.get(container_name)
            if value is None and isinstance(container, Mapping):
                value = container.get("read_attention_architecture")
    return str(value) if value is not None else None


def infer_read_architecture(loaded: Any, state: Mapping[str, torch.Tensor]) -> str:
    reader_keys = [key for key in state if key.startswith("read_implant.read_bridge.")]
    if not reader_keys:
        raise KeyError("Checkpoint has no read_implant.read_bridge tensors")
    inferred = (
        "sigmoid_all"
        if "read_implant.read_bridge.sigmoid_patch_head_bias" in state
        else (
            "sigmoid_mass"
            if "read_implant.read_bridge.sigmoid_head_bias" in state
            else "softmax"
        )
    )
    explicit = _explicit_read_architecture(loaded)
    if explicit is not None and explicit != inferred:
        raise ValueError(
            "Checkpoint metadata conflicts with checkpoint tensors: "
            f"metadata={explicit!r}, tensors={inferred!r}"
        )
    return inferred


def _layer_count(state: Mapping[str, torch.Tensor], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    indices = {
        int(match.group(1))
        for key in state
        if (match := pattern.match(key)) is not None
    }
    if not indices:
        raise KeyError(f"No transformer layers under {prefix!r}")
    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise ValueError(
            f"Non-contiguous transformer layers under {prefix}: {sorted(indices)}"
        )
    return len(indices)


def _tensor_int(state: Mapping[str, torch.Tensor], key: str, default: int) -> int:
    value = state.get(key)
    return int(value.item()) if torch.is_tensor(value) else int(default)


def _tensor_int_tuple(
    state: Mapping[str, torch.Tensor], key: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    value = state.get(key)
    if value is None:
        return default
    return tuple(int(item) for item in value.reshape(-1).tolist())


def _architecture_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
    return digest.hexdigest()


def infer_checkpoint_spec(
    checkpoint_path: str | Path,
    loaded: Any,
    state: Mapping[str, torch.Tensor],
) -> CheckpointSpec:
    """Infer every inference-relevant dimension from checkpoint contents."""
    required = (
        "visual.conv1.weight",
        "visual.class_embedding",
        "visual.positional_embedding",
        "visual.proj",
        "token_embedding.weight",
        "positional_embedding",
        "ln_final.weight",
        "text_projection",
        "visual.read_null_token",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"Checkpoint is missing required tensors: {missing}")

    legacy_prefixes = (
        "read_implant.presence_pool.",
        "read_implant.glyph_head.",
        "read_implant.auto_router.",
    )
    legacy = [key for key in state if key.startswith(legacy_prefixes)]
    final_prefixes = (
        "read_implant.source_head.",
        "read_implant.orthographic_bridge.",
        "read_implant.trust_router.",
    )
    missing_branches = [
        prefix
        for prefix in final_prefixes
        if not any(key.startswith(prefix) for key in state)
    ]
    if legacy or missing_branches:
        raise ValueError(
            "Full x-attention export requires the final PIECES architecture: "
            f"legacy_keys={legacy[:8]}, missing_branches={missing_branches}"
        )
    longclip_keys = [
        key
        for key in state
        if key == "positional_embedding_res" or key.startswith(("mask1", "mask2"))
    ]
    model = _model_object(loaded)
    if longclip_keys or bool(getattr(model, "use_positional_embedding_res", False)):
        raise ValueError(
            "This converter implements the vanilla CLIP text-position path only; "
            f"found LongCLIP position state/attributes: {longclip_keys[:8]}"
        )

    patch_weight = state["visual.conv1.weight"]
    if patch_weight.ndim != 4:
        raise ValueError(
            f"Expected ViT patch convolution, got {tuple(patch_weight.shape)}"
        )
    vision_width = int(patch_weight.shape[0])
    patch_size = int(patch_weight.shape[-1])
    position_count = int(state["visual.positional_embedding"].shape[0])
    grid_size = int(round(math.sqrt(position_count - 1)))
    if grid_size * grid_size + 1 != position_count:
        raise ValueError(
            f"Visual position count {position_count} is not a square patch grid plus CLS"
        )

    vision_layers = _layer_count(state, "visual.transformer.resblocks.")
    text_layers = _layer_count(state, "transformer.resblocks.")
    text_width = int(state["ln_final.weight"].numel())
    visual = getattr(model, "visual", None) if model is not None else None

    if visual is not None:
        serialized_resolution = getattr(visual, "input_resolution", None)
        if (
            serialized_resolution is not None
            and int(serialized_resolution) != grid_size * patch_size
        ):
            raise ValueError(
                "Checkpoint model attributes conflict with tensors: "
                f"visual.input_resolution={serialized_resolution}, "
                f"position_grid_resolution={grid_size * patch_size}"
            )
        serialized_insert = getattr(visual, "read_null_insert_block", None)
        insert_buffer = state.get("visual.read_null_insert_block_config")
        if (
            serialized_insert is not None
            and insert_buffer is not None
            and int(serialized_insert) != int(insert_buffer.item())
        ):
            raise ValueError(
                "Checkpoint model attributes conflict with tensors: "
                f"visual.read_null_insert_block={serialized_insert}, "
                f"buffer={int(insert_buffer.item())}"
            )

    read_architecture = infer_read_architecture(loaded, state)
    read_taps = _tensor_int_tuple(
        state,
        "read_implant.tap_blocks",
        (max(0, vision_layers - 4), max(0, vision_layers - 3)),
    )
    ortho_taps = _tensor_int_tuple(
        state,
        "read_implant.ortho_tap_blocks",
        tuple(x for x in (8, 12, 13) if x < vision_layers),
    )
    source_taps = _tensor_int_tuple(state, "read_implant.source_tap_blocks", ortho_taps)

    spec = CheckpointSpec(
        checkpoint_name=Path(checkpoint_path).stem,
        architecture_fingerprint=_architecture_fingerprint(state),
        image_size=grid_size * patch_size,
        patch_size=patch_size,
        projection_dim=int(state["text_projection"].shape[1]),
        vision_width=vision_width,
        vision_layers=vision_layers,
        vision_heads=vision_width // 64,
        text_width=text_width,
        text_layers=text_layers,
        text_heads=text_width // 64,
        context_length=int(state["positional_embedding"].shape[0]),
        source_vocab_size=int(state["token_embedding.weight"].shape[0]),
        eot_token_id=int(getattr(model, "eot_token_id", 49407)),
        hard_text_token_id=int(getattr(model, "hard_text_token_id", 49408)),
        no_text_token_id=int(getattr(model, "no_text_token_id", 49409)),
        any_text_token_id=int(getattr(model, "any_text_token_id", 49410)),
        null_text_token_id=int(getattr(model, "null_text_token_id", 49411)),
        read_null_enabled=True,
        read_null_insert_block=_tensor_int(
            state,
            "visual.read_null_insert_block_config",
            int(getattr(model, "read_null_insert_block", 20)),
        ),
        read_attention_architecture=read_architecture,
        read_tap_blocks=read_taps,
        ortho_tap_blocks=ortho_taps,
        source_tap_blocks=source_taps,
        read_bridge_width=int(state["read_implant.read_bridge.q_proj.weight"].shape[0]),
        read_bridge_heads=_tensor_int(state, "read_implant.bridge_heads_config", 4),
        early_expanded_width=_tensor_int(
            state, "read_implant.early_expanded_width_config", 4096
        ),
        source_hidden_width=int(
            state["read_implant.source_head.patch_contract.weight"].shape[0]
        ),
        trust_hidden_width=int(state["read_implant.trust_router.fc1.weight"].shape[0]),
        register_norm_threshold=float(
            getattr(visual, "register_norm_threshold", DEFAULT_REGISTER_THRESHOLD)
        ),
        register_min=int(getattr(visual, "register_min", DEFAULT_REGISTER_MIN)),
        register_max=int(getattr(visual, "register_max", DEFAULT_REGISTER_MAX)),
    )
    _validate_internal_shapes(spec, state)
    validate_supported_checkpoint(spec)
    return spec


def _validate_internal_shapes(
    spec: CheckpointSpec, state: Mapping[str, torch.Tensor]
) -> None:
    """Throw when serialized metadata/buffers conflict with tensor-defined dimensions."""
    expected_shapes = {
        "visual.class_embedding": (spec.vision_width,),
        "visual.positional_embedding": (
            (spec.image_size // spec.patch_size) ** 2 + 1,
            spec.vision_width,
        ),
        "visual.proj": (spec.vision_width, spec.projection_dim),
        "visual.read_null_token": (spec.vision_width,),
        "token_embedding.weight": (spec.source_vocab_size, spec.text_width),
        "positional_embedding": (spec.context_length, spec.text_width),
        "text_projection": (spec.text_width, spec.projection_dim),
        "read_implant.read_bridge.q_proj.weight": (
            spec.read_bridge_width,
            spec.text_width,
        ),
        "read_implant.read_bridge.k_proj.weight": (
            spec.read_bridge_width,
            spec.vision_width,
        ),
        "read_implant.read_bridge.v_proj.weight": (
            spec.read_bridge_width,
            spec.vision_width,
        ),
        "read_implant.content_pool.out_proj.weight": (
            spec.projection_dim,
            spec.read_bridge_width,
        ),
        "read_implant.content_tap_logits": (len(spec.read_tap_blocks),),
        "read_implant.read_tap_logits": (len(spec.read_tap_blocks),),
        "read_implant.ortho_tap_logits": (len(spec.ortho_tap_blocks),),
        "read_implant.source_tap_logits": (len(spec.source_tap_blocks),),
        "read_implant.read_probe": (spec.projection_dim,),
        "read_implant.orthographic_bridge.patch_expand.weight": (
            spec.early_expanded_width,
            spec.vision_width,
        ),
        "read_implant.orthographic_bridge.patch_contract.weight": (
            spec.read_bridge_width,
            spec.early_expanded_width,
        ),
        "read_implant.source_head.patch_expand.weight": (
            spec.early_expanded_width,
            spec.vision_width,
        ),
        "read_implant.source_head.patch_contract.weight": (
            spec.source_hidden_width,
            spec.early_expanded_width,
        ),
        "read_implant.source_head.stats_fc1.weight": (spec.source_hidden_width, 8),
        "read_implant.source_head.stats_fc2.weight": (2, spec.source_hidden_width),
        "read_implant.trust_router.fc1.weight": (spec.trust_hidden_width, 16),
        "read_implant.trust_router.fc2.weight": (
            spec.trust_hidden_width // 2,
            spec.trust_hidden_width,
        ),
        "read_implant.trust_router.fc3.weight": (
            1,
            spec.trust_hidden_width // 2,
        ),
    }
    mismatches = []
    for key, expected in expected_shapes.items():
        value = state.get(key)
        if value is None:
            mismatches.append(f"missing {key}")
        elif tuple(value.shape) != expected:
            mismatches.append(f"{key}={tuple(value.shape)} expected={expected}")
    if spec.read_bridge_width % spec.read_bridge_heads:
        mismatches.append(
            f"read_bridge_width={spec.read_bridge_width} is not divisible by "
            f"read_bridge_heads={spec.read_bridge_heads}"
        )
    if spec.trust_hidden_width < 2 or spec.trust_hidden_width % 2:
        mismatches.append(
            f"trust_hidden_width={spec.trust_hidden_width} must be an even value >= 2"
        )
    if spec.source_vocab_size <= max(
        spec.hard_text_token_id,
        spec.no_text_token_id,
        spec.any_text_token_id,
        spec.null_text_token_id,
    ):
        mismatches.append(
            "source vocabulary does not contain all serialized control-token IDs"
        )
    for label, taps in (
        ("read_tap_blocks", spec.read_tap_blocks),
        ("ortho_tap_blocks", spec.ortho_tap_blocks),
        ("source_tap_blocks", spec.source_tap_blocks),
    ):
        bad = [item for item in taps if not 0 <= item < spec.vision_layers]
        if bad:
            mismatches.append(f"{label} contains invalid blocks {bad}")
    if any(item < spec.read_null_insert_block for item in spec.read_tap_blocks):
        mismatches.append("read_tap_blocks precede RN insertion")
    if mismatches:
        raise ValueError(
            "Checkpoint tensors are internally inconsistent: " + "; ".join(mismatches)
        )


def validate_supported_checkpoint(spec: CheckpointSpec) -> None:
    expected = {
        "patch_size": 14,
        "projection_dim": 768,
        "vision_width": 1024,
        "vision_layers": 24,
        "vision_heads": 16,
        "text_width": 768,
        "text_layers": 12,
        "text_heads": 12,
        "context_length": 77,
    }
    failures = [
        f"{key}={getattr(spec, key)} expected={value}"
        for key, value in expected.items()
        if getattr(spec, key) != value
    ]
    if spec.image_size not in SUPPORTED_IMAGE_SIZES:
        failures.append(
            f"image_size={spec.image_size} expected one of {SUPPORTED_IMAGE_SIZES}"
        )
    if not (0 <= spec.read_null_insert_block < spec.vision_layers):
        failures.append(
            f"read_null_insert_block={spec.read_null_insert_block} outside visual blocks"
        )
    if failures:
        raise ValueError("Unsupported checkpoint architecture: " + "; ".join(failures))


def read_training_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def _training_json_value(training_config: Mapping[str, Any], key: str) -> Any:
    """Read either the old flat export JSON or the current single-config layout."""
    if key in training_config:
        return training_config[key]

    global_cfg = training_config.get("global")
    if isinstance(global_cfg, Mapping):
        if key in {
            "read_attention_architecture",
            "read_null_enabled",
            "read_null_insert_block",
            "image_size",
        } and key in global_cfg:
            return global_cfg[key]
        taps = global_cfg.get("tap_blocks")
        if isinstance(taps, Mapping):
            tap_key = {
                "read_tap_blocks": "late",
                "ortho_tap_blocks": "orthographic",
                "source_tap_blocks": "source",
            }.get(key)
            if tap_key is not None and tap_key in taps:
                return taps[tap_key]

    final_cfg = training_config.get("shared_args")
    if isinstance(final_cfg, Mapping):
        final_cfg = final_cfg.get("final_anytext")
        if isinstance(final_cfg, Mapping) and key in final_cfg:
            return final_cfg[key]

    return None


def warn_on_json_mismatches(
    spec: CheckpointSpec, training_config: Mapping[str, Any]
) -> list[str]:
    """Warn about JSON drift while retaining checkpoint-derived values.

    ``training_config.local.json`` uses nested ``global``/``tap_blocks`` sections,
    while older export metadata was flat.  Accept both so the optional config audit
    actually checks the current release config instead of silently skipping it.
    """
    comparisons: dict[str, Any] = {
        "read_attention_architecture": spec.read_attention_architecture,
        "read_null_enabled": spec.read_null_enabled,
        "read_null_insert_block": spec.read_null_insert_block,
        "read_tap_blocks": list(spec.read_tap_blocks),
        "ortho_tap_blocks": list(spec.ortho_tap_blocks),
        "source_tap_blocks": list(spec.source_tap_blocks),
        "image_size": spec.image_size,
        "register_norm_threshold": spec.register_norm_threshold,
        "register_min": spec.register_min,
        "register_max": spec.register_max,
    }
    mismatches: list[str] = []
    for key, checkpoint_value in comparisons.items():
        json_value = _training_json_value(training_config, key)
        if json_value is None:
            continue
        if isinstance(checkpoint_value, list):
            json_value = [int(item) for item in json_value]
        if json_value != checkpoint_value:
            message = (
                f"{key}: checkpoint={checkpoint_value!r}, json={json_value!r}; "
                "using checkpoint"
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            mismatches.append(message)
    return mismatches
