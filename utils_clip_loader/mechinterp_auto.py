from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from .clip_anything_to_openai import ClipLoadInfo


@dataclass(frozen=True)
class AutoMechinterpMeta:
    clip_module: str
    model_family: str
    canonical_state_sha256: str
    resolved_revision: Optional[str]


def _state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Stable hash over canonical OpenAI-style state tensors."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(int(x)) for x in tensor.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _revision_from_cache_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parts = Path(value).parts
    try:
        idx = parts.index("snapshots")
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    candidate = parts[idx + 1]
    if len(candidate) >= 7:
        return candidate
    return None


def _normalize_vanilla_sae_precision(model: torch.nn.Module, device: str) -> None:
    """Match OpenAI CLIP compute dtypes for the bundled split-QKV SAE runtime."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        model.float()
        return

    def _convert(module: torch.nn.Module) -> None:
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            if getattr(module, "weight", None) is not None:
                module.weight.data = module.weight.data.half()
            if getattr(module, "bias", None) is not None:
                module.bias.data = module.bias.data.half()

    model.apply(_convert)
    for module in model.modules():
        for name in ("text_projection", "proj"):
            value = getattr(module, name, None)
            if isinstance(value, torch.nn.Parameter):
                value.data = value.data.half()


def load_mechinterp_clip_anything(
    model_or_path: str,
    *,
    device: str = "cpu",
    jit: bool = False,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    strict: bool = True,
    allow_unsafe_hf_pickle: bool = False,
) -> tuple[torch.nn.Module, Any, "ClipLoadInfo", AutoMechinterpMeta]:
    """Load a CLIP checkpoint through the appropriate bundled mechinterp runtime.

    Vanilla CLIP checkpoints are instantiated through ``attnclip_mechinterp_sae``.
    Checkpoints with RN/correction/x-attention parameters are instantiated through
    ``attnclip_mechinterp_xattn``. Detection is state-dict based via the existing
    ``resolve_to_openai_state_dict`` loader, not inferred from a model repo name.

    This helper currently targets state-dict/HF style checkpoints. It deliberately
    does not execute remote code.
    """
    # Keep CLI/parser imports lightweight; the existing loader brings optional
    # runtime presentation dependencies (e.g. colorama) only when a model is
    # actually being resolved.
    from .clip_anything_to_openai import (
        _instantiate_and_load_openai_clip_from_state_dict,
        resolve_to_openai_state_dict,
    )

    canonical_state, detected = resolve_to_openai_state_dict(
        model_or_path,
        cache_dir=cache_dir,
        revision=revision,
        allow_unsafe_hf_pickle=allow_unsafe_hf_pickle,
    )
    family = str(detected.model_family)
    module_name = (
        "attnclip_mechinterp_sae" if family == "vanilla" else "attnclip_mechinterp_xattn"
    )
    clip_module = importlib.import_module(module_name)
    state_hash = _state_sha256(canonical_state)
    resolved_revision = _revision_from_cache_path(detected.resolved_path)

    model, preprocess, loaded = _instantiate_and_load_openai_clip_from_state_dict(
        clip_module,
        state_dict=canonical_state,
        device=device,
        jit=jit,
        strict=strict,
        resolved_path=detected.resolved_path,
        hf_config=detected.config,
        base_model_or_path=detected.base_model_or_path,
    )
    if module_name == "attnclip_mechinterp_sae":
        _normalize_vanilla_sae_precision(model, device)
    del canonical_state

    meta = AutoMechinterpMeta(
        clip_module=module_name,
        model_family=str(loaded.model_family),
        canonical_state_sha256=state_hash,
        resolved_revision=resolved_revision,
    )
    return model, preprocess, loaded, meta
