"""Single-model benchmark loading utilities.

The old public benchmark package kept a global model registry because it was used
for large local comparison sweeps.  Public reproduction now has one configured
model at a time; the default is the released ModeMUX checkpoint and every script
accepts an explicit override.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from utils_clip_loader.clip_anything_to_openai import ClipLoadInfo, load_openai_clip_anything

from .constants import DEFAULT_MODEL_ALIAS, DEFAULT_MODEL_PATH


@dataclass(frozen=True)
class ModelSpec:
    alias: str = DEFAULT_MODEL_ALIAS
    path: str = DEFAULT_MODEL_PATH
    base_model_or_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_alias(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("._") or "model"


def model_display_name(spec: ModelSpec) -> str:
    """Return an HF repo id, or only the final component of a local path."""
    value = str(spec.path).strip().rstrip("/\\")
    looks_local = (
        (len(value) >= 2 and value[1] == ":")
        or "\\" in value
        or value.startswith(("./", "../", "~/"))
        or value.count("/") != 1
        or Path(value).suffix.lower() in {".pt", ".pth", ".safetensors", ".bin"}
    )
    if looks_local:
        return value.replace("\\", "/").rsplit("/", 1)[-1]
    return value


def model_file_token(spec: ModelSpec) -> str:
    return safe_alias(model_display_name(spec).replace("/", "_"))


def load_model_spec(
    clip_module,
    spec: ModelSpec,
    *,
    device: str,
    strict: bool = True,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
):
    model, preprocess, info = load_openai_clip_anything(
        clip_module,
        spec.path,
        device=device,
        jit=False,
        cache_dir=cache_dir,
        revision=revision,
        strict=strict,
        base_model_or_path=spec.base_model_or_path,
        allow_unsafe_hf_pickle=False,
    )
    model.eval()
    return model, preprocess, info


def model_family(model, info: Optional[ClipLoadInfo] = None) -> str:
    if info is not None:
        return info.model_family
    return str(getattr(model, "_clip_model_family", "vanilla"))
