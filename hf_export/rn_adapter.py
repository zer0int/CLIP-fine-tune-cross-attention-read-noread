from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file


RN_FILENAME = "read_null_token.safetensors"
SUPPORTED_IMAGE_SIZES = (224, 336)


def _vision_model(model: Any):
    candidate = getattr(model, "vision_model", model)
    if hasattr(candidate, "vision_model"):
        candidate = candidate.vision_model
    if not all(hasattr(candidate, name) for name in ("embeddings", "encoder")):
        raise TypeError(
            "Expected CLIPModel, CLIPVisionModel, or CLIPVisionModelWithProjection"
        )
    return candidate


def _architecture(vision_model: Any) -> dict[str, int]:
    config = vision_model.config
    image_size = config.image_size
    patch_size = config.patch_size
    if isinstance(image_size, (list, tuple)):
        if len(set(image_size)) != 1:
            raise ValueError(f"Only square CLIP inputs are supported, got {image_size}")
        image_size = image_size[0]
    if isinstance(patch_size, (list, tuple)):
        if len(set(patch_size)) != 1:
            raise ValueError(f"Only square patches are supported, got {patch_size}")
        patch_size = patch_size[0]
    return {
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "vision_width": int(config.hidden_size),
        "vision_layers": int(config.num_hidden_layers),
        "vision_heads": int(config.num_attention_heads),
    }


def validate_vit_l_14(vision_model: Any) -> dict[str, int]:
    """Require the OpenAI-style ViT-L/14 tensor architecture at 224 or 336 px."""
    actual = _architecture(vision_model)
    expected = {
        "patch_size": 14,
        "vision_width": 1024,
        "vision_layers": 24,
        "vision_heads": 16,
    }
    failures = [
        f"{key}={actual[key]} expected={value}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if actual["image_size"] not in SUPPORTED_IMAGE_SIZES:
        failures.append(
            f"image_size={actual['image_size']} expected one of {SUPPORTED_IMAGE_SIZES}"
        )
    if failures:
        raise ValueError("RN token requires ViT-L/14: " + "; ".join(failures))
    return actual


def _resolve_rn_file(path_or_repo_id: str | Path, revision: str | None) -> Path:
    path = Path(path_or_repo_id).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        candidate = path / RN_FILENAME
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate.resolve()
    return Path(
        hf_hub_download(
            repo_id=str(path_or_repo_id), filename=RN_FILENAME, revision=revision
        )
    )


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _metadata_int(metadata: dict[str, str], key: str, fallback: int) -> int:
    value = metadata.get(key)
    return int(value) if value is not None else int(fallback)


def apply_read_null_token(
    model: Any,
    token_path_or_repo_id: str | Path,
    *,
    revision: str | None = None,
    debug: bool = False,
):
    """Append a learned RN token immediately before its checkpoint-defined ViT block.

    The helper is explicit because a stock Transformers CLIP class cannot grow a new
    sequence token merely by loading an extra safetensors key. It does not require
    ``trust_remote_code``.
    """
    vision = _vision_model(model)
    actual = validate_vit_l_14(vision)
    if getattr(vision, "_rn_adapter_handle", None) is not None:
        raise RuntimeError("An RN token is already applied to this vision model")

    token_path = _resolve_rn_file(token_path_or_repo_id, revision)
    tensors = load_file(str(token_path), device="cpu")
    if "read_null_token" not in tensors:
        raise KeyError(f"{token_path} has no 'read_null_token' tensor")
    token = tensors["read_null_token"]
    if tuple(token.shape) not in ((actual["vision_width"],), (1, actual["vision_width"])):
        raise ValueError(
            f"RN tensor shape {tuple(token.shape)} does not match vision width "
            f"{actual['vision_width']}"
        )
    token = token.reshape(actual["vision_width"])
    metadata = _metadata(token_path)
    insert_block = _metadata_int(metadata, "read_null_insert_block", 13)
    metadata_width = _metadata_int(metadata, "vision_width", actual["vision_width"])
    metadata_size = _metadata_int(metadata, "image_size", actual["image_size"])
    if metadata_width != actual["vision_width"]:
        raise ValueError(
            f"RN metadata vision_width={metadata_width} conflicts with model "
            f"vision_width={actual['vision_width']}"
        )
    if metadata_size not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(
            f"RN metadata image_size={metadata_size} is unsupported; expected one of "
            f"{SUPPORTED_IMAGE_SIZES}"
        )
    if not 0 <= insert_block < actual["vision_layers"]:
        raise ValueError(f"RN insertion block {insert_block} is outside the vision stack")

    device = vision.embeddings.class_embedding.device
    dtype = vision.embeddings.class_embedding.dtype
    vision.register_parameter(
        "read_null_token", torch.nn.Parameter(token.to(device=device, dtype=dtype))
    )
    vision.read_null_insert_block = insert_block

    def append_rn(_module, args):
        if not args:
            raise RuntimeError("CLIP encoder layer received no hidden states")
        hidden_states = args[0]
        rn = vision.read_null_token.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        ).view(1, 1, -1)
        rn = rn.expand(hidden_states.shape[0], 1, -1)
        return (torch.cat((hidden_states, rn), dim=1), *args[1:])

    layer = vision.encoder.layers[insert_block]
    vision._rn_adapter_handle = layer.register_forward_pre_hook(append_rn)
    vision._rn_adapter_source = str(token_path)

    if debug:
        print("[RN adapter] model architecture:", json.dumps(actual, sort_keys=True))
        print(f"[RN adapter] insertion: before zero-based block {insert_block}")
        print(
            f"[RN adapter] resolution: token source={metadata_size}, "
            f"target model={actual['image_size']} (224/336 cross-use is supported)"
        )
        print(
            "[RN adapter] token:",
            str(token_path),
            f"shape={tuple(token.shape)}",
            f"fp32_l2={token.float().norm().item():.8f}",
        )
        print(
            "[RN adapter] compatibility warning: the RN token supports OpenAI CLIP "
            "ViT-L/14 lineage fine-tunes only; lineage cannot be proven from dimensions"
        )
    return model


def remove_read_null_token(model: Any):
    """Remove an RN hook and/or an attached RN parameter from a stock HF model."""
    vision = _vision_model(model)
    handle = getattr(vision, "_rn_adapter_handle", None)
    if handle is not None:
        handle.remove()
    vision._rn_adapter_handle = None
    vision._rn_adapter_source = None
    vision.read_null_insert_block = None
    if hasattr(vision, "read_null_token"):
        delattr(vision, "read_null_token")
    return model


def load_clip_with_rn(
    base_model_id_or_path: str | Path,
    token_path_or_repo_id: str | Path,
    *,
    revision: str | None = None,
    debug: bool = False,
    **from_pretrained_kwargs,
):
    """Load stock ``CLIPModel`` weights and apply the separate RN token adapter."""
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(
        str(base_model_id_or_path), revision=revision, **from_pretrained_kwargs
    )
    return apply_read_null_token(
        model, token_path_or_repo_id, revision=revision, debug=debug
    )
