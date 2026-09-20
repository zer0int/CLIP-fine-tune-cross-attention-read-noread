"""Load CLIP checkpoints into the bundled OpenAI-style runtimes.

The public symbols are resolved lazily so importing a lightweight helper such as
``mechinterp_auto`` does not eagerly import optional presentation/runtime
packages used only by the full checkpoint loader.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "AutoMechinterpMeta",
    "ClipAnythingLoadError",
    "ClipLoadInfo",
    "load_mechinterp_clip_anything",
    "load_openai_clip_anything",
]


def __getattr__(name: str) -> Any:
    if name in {"AutoMechinterpMeta", "load_mechinterp_clip_anything"}:
        from .mechinterp_auto import AutoMechinterpMeta, load_mechinterp_clip_anything
        return {
            "AutoMechinterpMeta": AutoMechinterpMeta,
            "load_mechinterp_clip_anything": load_mechinterp_clip_anything,
        }[name]
    if name in {"ClipAnythingLoadError", "ClipLoadInfo", "load_openai_clip_anything"}:
        from .clip_anything_to_openai import (
            ClipAnythingLoadError,
            ClipLoadInfo,
            load_openai_clip_anything,
        )
        return {
            "ClipAnythingLoadError": ClipAnythingLoadError,
            "ClipLoadInfo": ClipLoadInfo,
            "load_openai_clip_anything": load_openai_clip_anything,
        }[name]
    raise AttributeError(name)
