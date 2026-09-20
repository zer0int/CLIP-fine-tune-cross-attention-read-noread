"""Utilities for compositing manifest-indexed handwriting onto ordinary images.

The supported manifest is the concrete ``HandwritingOverlays/manifest.jsonl``
schema bundled with this training suite.  Rows are indexed by ``prompt_text``;
``png_path`` identifies the RGBA glyph source and ``bbox`` identifies the drawn
region.  Only exact normalized word matches are returned.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def normalize_handwriting_word(value: str) -> str:
    return " ".join(str(value).casefold().split())


@dataclass(frozen=True)
class HandwritingSample:
    sample_id: str
    word: str
    phase: str
    png_path: Path
    bbox: Optional[Tuple[int, int, int, int]]
    canonical: bool


class HandwritingOverlayIndex:
    """Exact-word index over the inspected handwriting manifest schema."""

    def __init__(self, root: Optional[Path]):
        self.root = Path(root) if root is not None else None
        self.by_word: Dict[str, List[HandwritingSample]] = {}
        self.manifest_path: Optional[Path] = None
        if self.root is not None:
            self._load()

    @property
    def available(self) -> bool:
        return bool(self.by_word)

    def words(self) -> Sequence[str]:
        return tuple(sorted(self.by_word))

    def samples(self, word: str) -> Sequence[HandwritingSample]:
        return tuple(self.by_word.get(normalize_handwriting_word(word), ()))

    def partitioned_samples(
        self,
        word: str,
        *,
        holdout_fraction: float,
        seed: int,
    ) -> Tuple[Sequence[HandwritingSample], Sequence[HandwritingSample]]:
        samples = list(self.samples(word))
        canonical = [sample for sample in samples if sample.canonical]
        samples = canonical or samples
        if len(samples) < 2 or holdout_fraction <= 0.0:
            return tuple(samples), tuple()
        ordered = sorted(
            samples,
            key=lambda sample: __import__("hashlib").sha1(
                f"{seed}|{sample.sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        holdout_n = max(1, min(len(ordered) - 1, int(round(len(ordered) * holdout_fraction))))
        return tuple(ordered[holdout_n:]), tuple(ordered[:holdout_n])

    def choose(
        self,
        word: str,
        rng: random.Random,
        *,
        sample_split: str = "any",
        holdout_fraction: float = 0.0,
        split_seed: int = 0,
    ) -> Optional[HandwritingSample]:
        train_samples, holdout_samples = self.partitioned_samples(
            word, holdout_fraction=holdout_fraction, seed=split_seed
        )
        if sample_split == "train":
            samples = train_samples
        elif sample_split == "heldout":
            # Never leak a training handwriting sample into held-out style
            # evaluation.  Callers fall back to digital rendering when a word
            # has no independently held-out handwriting asset.
            samples = holdout_samples
        elif sample_split == "any":
            samples = (*train_samples, *holdout_samples)
        else:
            raise ValueError(f"Unknown handwriting sample split: {sample_split}")
        return rng.choice(list(samples)) if samples else None

    def _load(self) -> None:
        assert self.root is not None
        candidates = (
            self.root / "manifest.jsonl",
            self.root / "manifests" / "manifest.jsonl",
        )
        manifest = next((path for path in candidates if path.is_file()), None)
        if manifest is None:
            return
        self.manifest_path = manifest
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid handwriting JSONL at {manifest}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {manifest}:{line_number}")
                sample = self._sample_from_row(row)
                if sample is not None:
                    self.by_word.setdefault(normalize_handwriting_word(sample.word), []).append(sample)

    def _sample_from_row(self, row: Mapping[str, Any]) -> Optional[HandwritingSample]:
        assert self.root is not None
        word = str(row.get("prompt_text") or "").strip()
        recorded = str(row.get("png_path") or "").strip()
        if not word or not recorded:
            return None
        basename = PureWindowsPath(recorded).name
        path_candidates = [
            self.root / "images" / basename,
            self.root / basename,
            Path(recorded),
        ]
        png_path = next((path for path in path_candidates if path.is_file()), None)
        if png_path is None:
            return None

        bbox_value = row.get("bbox")
        bbox: Optional[Tuple[int, int, int, int]] = None
        if isinstance(bbox_value, Mapping):
            try:
                bbox = (
                    int(round(float(bbox_value["x_min"]))),
                    int(round(float(bbox_value["y_min"]))),
                    int(round(float(bbox_value["x_max"]))),
                    int(round(float(bbox_value["y_max"]))),
                )
            except (KeyError, TypeError, ValueError):
                bbox = None

        return HandwritingSample(
            sample_id=str(row.get("sample_id") or basename),
            word=word,
            phase=str(row.get("phase") or "unknown"),
            png_path=png_path,
            bbox=bbox,
            canonical=bool(row.get("canonical_for_prompt_phase", True)),
        )


def _derive_alpha(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    lo, hi = alpha.getextrema()
    if hi > lo or lo < 255:
        return alpha

    gray = np.asarray(rgba.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return Image.new("L", rgba.size, 0)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    distance = np.abs(gray - background)
    scale = max(1.0, float(np.percentile(distance, 99.5)))
    derived = np.clip(distance * (255.0 / scale), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(derived, mode="L")


def load_handwriting_glyph(sample: HandwritingSample) -> Image.Image:
    """Load and tightly crop the handwriting glyph as RGBA, preserving alpha."""
    with Image.open(sample.png_path) as source:
        rgba = source.convert("RGBA")
    alpha = _derive_alpha(rgba)

    if sample.bbox is not None:
        x0, y0, x1, y1 = sample.bbox
        x0 = max(0, min(rgba.width, x0))
        y0 = max(0, min(rgba.height, y0))
        x1 = max(x0 + 1, min(rgba.width, x1))
        y1 = max(y0 + 1, min(rgba.height, y1))
        rgba = rgba.crop((x0, y0, x1, y1))
        alpha = alpha.crop((x0, y0, x1, y1))

    tight = alpha.getbbox()
    if tight is not None:
        rgba = rgba.crop(tight)
        alpha = alpha.crop(tight)
    rgba.putalpha(alpha)
    return rgba


def render_handwriting_crop(
    sample: HandwritingSample,
    *,
    fill: Tuple[int, int, int, int],
    stroke_fill: Tuple[int, int, int, int],
    stroke_width: int,
    plate: bool,
    plate_fill: Tuple[int, int, int, int],
    max_width: int,
    max_height: int,
    angle: float,
    scale: float = 1.0,
    width_scale: float = 1.0,
) -> Image.Image:
    """Recolor and fit a handwriting glyph into the ordinary renderer geometry."""
    glyph = load_handwriting_glyph(sample)
    alpha = glyph.getchannel("A")
    width = max(1, glyph.width)
    height = max(1, glyph.height)
    fit = min(max_width / width, max_height / height) * max(0.05, float(scale))
    new_width = max(1, int(round(width * fit * max(0.20, float(width_scale)))))
    new_height = max(1, int(round(height * fit)))
    alpha = alpha.resize((new_width, new_height), BICUBIC)

    pad = max(4, int(stroke_width) + 3)
    canvas = Image.new("RGBA", (new_width + 2 * pad, new_height + 2 * pad), (0, 0, 0, 0))
    if plate:
        plate_layer = Image.new("RGBA", canvas.size, plate_fill)
        canvas.alpha_composite(plate_layer)

    if stroke_width > 0:
        kernel = max(3, 2 * int(stroke_width) + 1)
        if kernel % 2 == 0:
            kernel += 1
        stroke_alpha = alpha.filter(ImageFilter.MaxFilter(kernel))
        stroke_layer = Image.new("RGBA", canvas.size, stroke_fill)
        placed_stroke = Image.new("L", canvas.size, 0)
        placed_stroke.paste(stroke_alpha, (pad, pad))
        stroke_layer.putalpha(placed_stroke.point(lambda value: value * stroke_fill[3] // 255))
        canvas.alpha_composite(stroke_layer)

    ink_layer = Image.new("RGBA", canvas.size, fill)
    placed_alpha = Image.new("L", canvas.size, 0)
    placed_alpha.paste(alpha, (pad, pad))
    ink_layer.putalpha(placed_alpha.point(lambda value: value * fill[3] // 255))
    canvas.alpha_composite(ink_layer)

    if abs(float(angle)) > 1.0e-3:
        canvas = canvas.rotate(float(angle), resample=BICUBIC, expand=True)
    return canvas


def _mask_orientation_degrees(mask: Image.Image) -> float:
    array = np.asarray(mask.convert("L"), dtype=np.float32)
    ys, xs = np.nonzero(array > 32.0)
    if xs.size < 8:
        return 0.0
    coords = np.stack((xs - xs.mean(), ys - ys.mean()), axis=1)
    covariance = coords.T @ coords / max(1, coords.shape[0] - 1)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    while angle > 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    return angle


def composite_handwriting_on_base(
    base: Image.Image,
    digital_variant: Image.Image,
    digital_mask: Optional[Image.Image],
    word: str,
    handwriting_index: Optional[HandwritingOverlayIndex],
    rng: random.Random,
    probability: float,
) -> Tuple[Image.Image, Optional[Image.Image], str, str]:
    """Substitute exact-word handwriting while retaining the ordinary base image.

    Placement, approximate support, orientation, and visible non-glyph plate
    pixels are inferred from the existing digital variant and its mask.  If any
    required evidence is unavailable, the digital renderer is returned intact.
    """
    if (
        handwriting_index is None
        or not handwriting_index.available
        or rng.random() >= max(0.0, min(1.0, float(probability)))
    ):
        return digital_variant, digital_mask, "digital", ""
    sample = handwriting_index.choose(word, rng)
    if sample is None or digital_mask is None:
        return digital_variant, digital_mask, "digital", ""

    base_rgb = base.convert("RGB")
    digital_rgb = digital_variant.convert("RGB")
    mask = digital_mask.convert("L")
    if mask.size != base_rgb.size:
        mask = mask.resize(base_rgb.size, Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST)
    bbox = mask.getbbox()
    if bbox is None:
        return digital_variant, digital_mask, "digital", ""
    x0, y0, x1, y1 = bbox
    support_w = max(1, x1 - x0)
    support_h = max(1, y1 - y0)

    mask_array = np.asarray(mask, dtype=np.uint8)
    digital_array = np.asarray(digital_rgb, dtype=np.uint8)
    base_array = np.asarray(base_rgb, dtype=np.uint8)
    glyph_pixels = mask_array > 32
    if glyph_pixels.any():
        median_rgb = tuple(int(value) for value in np.median(digital_array[glyph_pixels], axis=0))
    else:
        median_rgb = (0, 0, 0)
    stroke_rgb = (255, 255, 255) if sum(median_rgb) < 384 else (0, 0, 0)

    # Preserve visible plate/background changes outside the digital glyph mask.
    diff = np.abs(digital_array.astype(np.int16) - base_array.astype(np.int16)).max(axis=2)
    plate_mask = (diff > 12) & ~glyph_pixels
    out_array = base_array.copy()
    out_array[plate_mask] = digital_array[plate_mask]
    out = Image.fromarray(out_array, mode="RGB").convert("RGBA")

    crop = render_handwriting_crop(
        sample,
        fill=(*median_rgb, 255),
        stroke_fill=(*stroke_rgb, 255),
        stroke_width=1,
        plate=False,
        plate_fill=(0, 0, 0, 0),
        max_width=support_w,
        max_height=support_h,
        angle=_mask_orientation_degrees(mask),
    )
    if crop.width > support_w or crop.height > support_h:
        factor = min(support_w / crop.width, support_h / crop.height)
        crop = crop.resize(
            (max(1, int(round(crop.width * factor))), max(1, int(round(crop.height * factor)))),
            BICUBIC,
        )

    x = int(round((x0 + x1 - crop.width) / 2.0))
    y = int(round((y0 + y1 - crop.height) / 2.0))
    x = max(0, min(base_rgb.width - crop.width, x))
    y = max(0, min(base_rgb.height - crop.height, y))
    out.alpha_composite(crop, (x, y))

    glyph_mask = Image.new("L", base_rgb.size, 0)
    glyph_mask.paste(crop.getchannel("A"), (x, y))
    return (
        out.convert("RGB"),
        glyph_mask,
        f"handwriting_{sample.phase}",
        sample.sample_id,
    )
