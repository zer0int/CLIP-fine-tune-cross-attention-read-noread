"""Capability-aware inference helpers shared by the GitHub benchmarks."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .clip_anything_to_openai import ClipLoadInfo


def model_family(model, info: Optional[ClipLoadInfo] = None) -> str:
    """Infer the capability family without a global model registry."""
    if info is not None:
        return info.model_family
    return str(getattr(model, "_clip_model_family", "vanilla"))


@dataclass(frozen=True)
class SeparableMode:
    key: str
    label: str
    image_path: str
    use_read_null: Optional[bool] = None


@dataclass(frozen=True)
class FullXAttnCorrectionVariant:
    key: str
    label: str
    apply_content_correction: bool


def full_xattn_correction_variants(
    include_corr_off: bool = False,
) -> tuple[FullXAttnCorrectionVariant, ...]:
    """Return full-xattention CONTENT-correction inference variants.

    The correction-on variant keeps the historical benchmark keys/labels.
    ``include_corr_off=True`` adds a second pass through the same trained full
    model with only the separately parameterized CONTENT correction disabled;
    RN and the SOURCE/ORTHO/READ/router machinery remain unchanged.
    """
    variants = [
        FullXAttnCorrectionVariant(
            "corr_on",
            "CONTENT correction on",
            True,
        )
    ]
    if include_corr_off:
        variants.append(
            FullXAttnCorrectionVariant(
                "corr_off",
                "CONTENT correction off",
                False,
            )
        )
    return tuple(variants)


def full_xattn_variant_mode_key(
    base_key: str,
    variant: FullXAttnCorrectionVariant,
) -> str:
    """Preserve historical mode keys for correction-on; suffix only ablations."""
    return base_key if variant.apply_content_correction else f"{base_key}_corr_off"


def full_xattn_variant_mode_label(
    base_label: str,
    variant: FullXAttnCorrectionVariant,
) -> str:
    """Human-readable mode label for a full-model correction ablation."""
    return base_label if variant.apply_content_correction else f"{base_label} [corr off]"


@contextmanager
def temporary_content_correction(model, enabled: bool):
    """Temporarily toggle ordinary ``encode_image`` CONTENT correction.

    ``forward_modes`` accepts the correction choice explicitly, but image-only
    benchmarks such as the ImageNet linear probe use ``encode_image``.  This
    helper makes that ablation explicit and restores the model's prior default.
    """
    setter = getattr(model, "set_content_correction_enabled", None)
    if not callable(setter):
        raise RuntimeError(
            "CONTENT-correction toggle requested, but the model does not expose "
            "set_content_correction_enabled()"
        )
    attr = "_clip_apply_content_correction_by_default"
    previous = bool(getattr(model, attr, True))
    setter(bool(enabled))
    try:
        yield
    finally:
        setter(previous)


def inference_autocast(device: str, enabled: bool = True):
    if device.startswith("cuda") and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def separable_modes(
    model,
    info: Optional[ClipLoadInfo] = None,
) -> tuple[SeparableMode, ...]:
    family = model_family(model, info)
    if family == "full_xattn":
        return (
            SeparableMode("classic", "classic (RN backbone)", "base"),
            SeparableMode("notext", "<notext> / correction", "corrected"),
        )
    if family == "rn_correction":
        return (
            SeparableMode("correction_off", "RN, correction off", "base"),
            SeparableMode("correction_on", "RN, correction on", "corrected"),
        )
    if family == "rn_token":
        return (
            SeparableMode("vanilla", "vanilla (RN removed)", "base", False),
            SeparableMode("rn", "RN", "base", True),
        )
    return (SeparableMode("vanilla", "vanilla", "base"),)


def encode_image_mode(model, images: torch.Tensor, mode: SeparableMode) -> torch.Tensor:
    visual = getattr(model, "visual", None)
    previous_read_null = getattr(visual, "read_null_enabled", None)
    if mode.use_read_null is not None:
        if mode.use_read_null and getattr(visual, "read_null_token", None) is None:
            raise RuntimeError("RN mode requested, but the loaded vision tower has no RN token")
        visual.read_null_enabled = bool(mode.use_read_null)
    try:
        if mode.image_path == "base" and hasattr(model, "encode_image_base"):
            return model.encode_image_base(images)
        return model.encode_image(images)
    finally:
        if mode.use_read_null is not None:
            visual.read_null_enabled = previous_read_null


def normalized_text_features(model, tokens: torch.Tensor) -> torch.Tensor:
    return F.normalize(model.encode_text(tokens).float(), dim=-1)


def normalized_image_features(
    model,
    images: torch.Tensor,
    mode: SeparableMode,
) -> torch.Tensor:
    # Forward the selected inference mode so RN transplant models can switch
    # cleanly between their vanilla and RN-enabled image paths.
    return F.normalize(encode_image_mode(model, images, mode).float(), dim=-1)


def is_full_xattn(model, info: Optional[ClipLoadInfo] = None) -> bool:
    return model_family(model, info) == "full_xattn"
