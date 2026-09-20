from __future__ import annotations

import torch
from torch import nn

from utils_clip_loader.benchmark_runtime import (
    SeparableMode,
    full_xattn_correction_variants,
    full_xattn_variant_mode_key,
    full_xattn_variant_mode_label,
    normalized_image_features,
    temporary_content_correction,
)


class _RecordingVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.read_null_token = nn.Parameter(torch.zeros(2))
        self.read_null_enabled = True


class _RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _RecordingVisual()
        self.observed_read_null: list[bool] = []
        self._clip_apply_content_correction_by_default = True

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self.observed_read_null.append(self.visual.read_null_enabled)
        return torch.tensor([[3.0, 4.0]], device=images.device)

    def set_content_correction_enabled(self, enabled: bool = True) -> None:
        self._clip_apply_content_correction_by_default = bool(enabled)


def test_normalized_image_features_forwards_mode_and_restores_rn_state() -> None:
    model = _RecordingModel()
    images = torch.zeros(1, 3, 2, 2)
    vanilla_mode = SeparableMode(
        "vanilla",
        "vanilla (RN removed)",
        "base",
        False,
    )

    features = normalized_image_features(model, images, vanilla_mode)

    assert model.observed_read_null == [False]
    assert model.visual.read_null_enabled is True
    assert torch.allclose(features, torch.tensor([[0.6, 0.8]]))


def test_full_xattn_correction_variants_preserve_default_keys() -> None:
    default = full_xattn_correction_variants(False)
    assert len(default) == 1
    assert default[0].apply_content_correction is True
    assert full_xattn_variant_mode_key("any", default[0]) == "any"
    assert full_xattn_variant_mode_label("<any>", default[0]) == "<any>"

    both = full_xattn_correction_variants(True)
    assert [variant.key for variant in both] == ["corr_on", "corr_off"]
    assert full_xattn_variant_mode_key("any", both[1]) == "any_corr_off"
    assert full_xattn_variant_mode_label("<any>", both[1]) == "<any> [corr off]"


def test_temporary_content_correction_restores_previous_state() -> None:
    model = _RecordingModel()
    assert model._clip_apply_content_correction_by_default is True
    with temporary_content_correction(model, False):
        assert model._clip_apply_content_correction_by_default is False
    assert model._clip_apply_content_correction_by_default is True
