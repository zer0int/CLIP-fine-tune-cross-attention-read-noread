#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage-1 source learning for the hard-text GmP CLIP continuation.

One invocation runs three explicit phases and writes a checkpoint after each:

  1A: patch glyph/textness + present/readable bootstrap
  1B: forced-read source purification
  1C: latent <any> router warm-up

The original visual and text towers remain frozen. The detachable content correction
joins only Phase 1C unless --router_only_1c is used. Handwriting is sampled as a
renderer substitution for an ImageNet supportive or adversarial digital slot, never
as an independently weighted semantic source.

This clean null-controls revision adds a dedicated forced-read abstention candidate,
matched pseudoword/mirrored-word controls, detached glyph priors, and an explicit
<any> adversarial hard negative while keeping all mixed-mode scoring in
model.forward_modes().
"""

from __future__ import annotations

import argparse
import hashlib
import csv
import importlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps

from training_support.precision_policy import (
    AMP_CHOICES,
    assert_trainable_parameters_fp32,
    fp32_function,
    fp32_island,
    autocast_context as precision_autocast_context,
    make_grad_scaler as precision_make_grad_scaler,
    precision_summary,
    resolve_precision,
)


from training_support.diagnostics.training_precision_diagnostics import (
    PrecisionDiagnostics,
    PrecisionDiagnosticsConfig,
)

from coco_trusted_reading_manifest import (
    labels_from_value,
    load_coco_reading_manifest,
    mask_from_targets,
    normalize_image_key,
    target_texts,
)

from training_support.data.continuous_text_math_imagenet_training import ContinuousTextMathImageNetBuilder
from training_support.data.clevr_property_binding_training import ClevrPropertyBindingBuilder
from training_support.diagnostics.pieces_sigmoid_attention_diagnostics import collect_pieces_sigmoid_attention_metrics
from training_support.font_discovery import find_font_files
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything
import router_auxiliary



CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
PHASES = ("1a", "1b", "1b5", "1c")


@dataclass
class Packet:
    images: List[torch.Tensor]
    patch_masks: List[torch.Tensor]
    mask_weights: List[float]
    present_targets: List[float]
    readable_targets: List[float]
    captions: List[str]
    positive: torch.Tensor
    source_triplets: List[Tuple[int, int, List[int]]] = field(default_factory=list)
    auto_triplets: List[Tuple[int, int, int]] = field(default_factory=list)
    invariance_pairs: List[Tuple[int, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PackedBatch:
    images: torch.Tensor
    patch_masks: torch.Tensor
    mask_weights: torch.Tensor
    present_targets: torch.Tensor
    readable_targets: torch.Tensor
    captions: List[str]
    positive: torch.Tensor
    source_triplets: List[Tuple[int, int, List[int]]]
    auto_triplets: List[Tuple[int, int, int]]
    invariance_pairs: List[Tuple[int, int]]
    packet_metadata: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def normalize_words(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
    return [token for token in cleaned.split() if token]


def choose_caption(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def amp_context(device: torch.device, amp_dtype: str):
    return precision_autocast_context(device, amp_dtype)


def image_and_mask_transform(
    image: Image.Image,
    mask: Optional[Image.Image],
    size: int,
    flip: bool = False,
) -> Tuple[torch.Tensor, Image.Image]:
    image = image.convert("RGB")
    if mask is None:
        mask = Image.new("L", image.size, 0)
    else:
        mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, NEAREST)

    width, height = image.size
    scale = size / min(width, height)
    resized = (max(size, round(width * scale)), max(size, round(height * scale)))
    image = image.resize(resized, BICUBIC)
    mask = mask.resize(resized, NEAREST)
    left = max(0, (resized[0] - size) // 2)
    top = max(0, (resized[1] - size) // 2)
    box = (left, top, left + size, top + size)
    image = image.crop(box)
    mask = mask.crop(box)
    if flip:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - CLIP_MEAN) / CLIP_STD
    return tensor, mask


def mask_to_patch_targets(mask: Image.Image, patch_count: int) -> torch.Tensor:
    grid = round(math.sqrt(patch_count))
    if grid * grid != patch_count:
        raise ValueError(f"Patch count {patch_count} is not square")
    array = np.asarray(mask, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)[None, None]
    pooled = F.adaptive_max_pool2d(tensor, (grid, grid))[0, 0]
    return pooled.flatten().clamp(0, 1)




CONTROL_STYLES = (
    {"fill": (255, 255, 255), "stroke": (0, 0, 0), "stroke_width": 3, "box": None},
    {"fill": (0, 0, 0), "stroke": (255, 255, 255), "stroke_width": 3, "box": None},
    {"fill": (255, 255, 255), "stroke": (0, 0, 0), "stroke_width": 2, "box": (0, 0, 0, 150)},
    {"fill": (0, 0, 0), "stroke": (255, 255, 255), "stroke_width": 2, "box": (255, 255, 255, 165)},
)
# Known-good historical unreadable control.  The readable crop may be digital
# or handwritten; only its horizontal mirror is labelled present-but-unreadable.
CONTROL_VARIANTS = ("mirrored",)


@dataclass(frozen=True)
class ControlRenderPlan:
    font_path: str
    font_size: int
    fill: Tuple[int, int, int]
    stroke: Tuple[int, int, int]
    stroke_width: int
    box: Optional[Tuple[int, int, int, int]]
    x: int
    y: int
    angle: float


def resolve_control_fonts(explicit: Sequence[str]) -> List[str]:
    return find_font_files(explicit)


def load_control_font(path: str, size: int):
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def matched_pseudoword(word: str, rng: random.Random) -> str:
    """Length/punctuation-matched nonword; guaranteed to differ when possible."""
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    chars: List[str] = []
    changed = False
    for ch in str(word):
        low = ch.lower()
        if low.isalpha():
            pool = vowels if low in vowels else consonants
            choices = [x for x in pool if x != low]
            repl = rng.choice(choices or list(pool))
            if ch.isupper():
                repl = repl.upper()
            chars.append(repl)
            changed = changed or repl != ch
        elif low.isdigit():
            choices = [x for x in "0123456789" if x != low]
            repl = rng.choice(choices)
            chars.append(repl)
            changed = True
        else:
            chars.append(ch)
    value = "".join(chars)
    if not changed and value:
        value = value + "x"
    return value



def make_control_render_plan(
    size: Tuple[int, int],
    texts: Sequence[str],
    rng: random.Random,
    font_paths: Sequence[str],
) -> ControlRenderPlan:
    width, height = size
    style = dict(rng.choice(CONTROL_STYLES))
    font_path = rng.choice(list(font_paths)) if font_paths else ""
    base_size = rng.randint(18, 42)
    if max((len(x) for x in texts), default=0) >= 14:
        base_size = rng.randint(15, 28)
    probe = ImageDraw.Draw(Image.new("RGB", size))
    chosen_size = 12
    max_w = max_h = 1
    for font_size in range(base_size, 11, -1):
        font = load_control_font(font_path, font_size)
        boxes = [
            probe.textbbox((0, 0), text, font=font, stroke_width=style["stroke_width"])
            for text in texts
        ]
        max_w = max(max(1, box[2] - box[0]) for box in boxes)
        max_h = max(max(1, box[3] - box[1]) for box in boxes)
        chosen_size = font_size
        if max_w <= int(0.78 * width) and max_h <= int(0.22 * height):
            break
    pad = 8
    max_x = max(pad, width - max_w - 2 * pad)
    max_y = max(pad, height - max_h - 2 * pad)
    x_options = [pad, max_x, max(pad, (width - max_w) // 2)]
    y_options = [pad, max_y, max(pad, (height - max_h) // 2), max(pad, int(0.2 * height))]
    return ControlRenderPlan(
        font_path=font_path,
        font_size=chosen_size,
        fill=tuple(style["fill"]),
        stroke=tuple(style["stroke"]),
        stroke_width=int(style["stroke_width"]),
        box=style["box"],
        x=int(rng.choice(x_options)),
        y=int(rng.choice(y_options)),
        angle=float(rng.uniform(-10.0, 10.0)),
    )



def render_control_word(
    base: Image.Image,
    text: str,
    plan: ControlRenderPlan,
    mirror_word: bool,
) -> Tuple[Image.Image, Image.Image]:
    base = base.convert("RGB")
    font = load_control_font(plan.font_path, plan.font_size)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    box = probe.textbbox((0, 0), text, font=font, stroke_width=plan.stroke_width)
    tw = max(1, box[2] - box[0])
    th = max(1, box[3] - box[1])
    pad = 8
    tile = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    mask = Image.new("L", tile.size, 0)
    draw = ImageDraw.Draw(tile)
    draw_mask = ImageDraw.Draw(mask)
    if plan.box is not None:
        draw.rounded_rectangle((2, 2, tile.width - 3, tile.height - 3), radius=5, fill=plan.box)
    draw.text(
        (pad - box[0], pad - box[1]),
        text,
        font=font,
        fill=plan.fill,
        stroke_width=plan.stroke_width,
        stroke_fill=plan.stroke,
    )
    draw_mask.text(
        (pad - box[0], pad - box[1]),
        text,
        font=font,
        fill=255,
        stroke_width=plan.stroke_width,
    )
    if mirror_word:
        tile = ImageOps.mirror(tile)
        mask = ImageOps.mirror(mask)
    if abs(plan.angle) > 1.0e-6:
        tile = tile.rotate(plan.angle, resample=BICUBIC, expand=True)
        mask = mask.rotate(plan.angle, resample=BICUBIC, expand=True)
    x = max(0, min(plan.x, max(0, base.width - tile.width)))
    y = max(0, min(plan.y, max(0, base.height - tile.height)))
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    full_mask = Image.new("L", base.size, 0)
    layer.alpha_composite(tile, (x, y))
    full_mask.paste(mask, (x, y), mask)
    rendered = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    return rendered, full_mask



def _matched_overlay_layer(
    base: Image.Image,
    readable: Image.Image,
    glyph_mask: Optional[Image.Image],
) -> Tuple[Image.Image, Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """Recover the actual rendered overlay, including handwriting and plates."""
    base_rgb = base.convert("RGB")
    readable_rgb = readable.convert("RGB")
    if readable_rgb.size != base_rgb.size:
        readable_rgb = readable_rgb.resize(base_rgb.size, BICUBIC)
    base_array = np.asarray(base_rgb, dtype=np.uint8)
    readable_array = np.asarray(readable_rgb, dtype=np.uint8)
    difference = np.abs(readable_array.astype(np.int16) - base_array.astype(np.int16)).max(axis=2)
    support = difference > 10
    if glyph_mask is None:
        glyph = np.zeros(support.shape, dtype=np.uint8)
    else:
        resized = glyph_mask.convert("L")
        if resized.size != base_rgb.size:
            resized = resized.resize(base_rgb.size, NEAREST)
        glyph = np.asarray(resized, dtype=np.uint8)
        support |= glyph > 8
    support_image = Image.fromarray((support.astype(np.uint8) * 255), mode="L")
    bbox = support_image.getbbox()
    if bbox is None:
        bbox = (0, 0, base_rgb.width, base_rgb.height)
    x0, y0, x1, y1 = bbox
    rgba = readable_rgb.crop(bbox).convert("RGBA")
    rgba.putalpha(support_image.crop(bbox))
    glyph_crop = Image.fromarray(glyph, mode="L").crop(bbox)
    return rgba, glyph_crop, support_image, bbox


def _crop_visual_distance(
    reference: Image.Image,
    candidate: Image.Image,
    support: Tuple[int, int, int, int],
) -> float:
    ref = np.asarray(reference.convert("RGB").crop(support), dtype=np.float32) / 255.0
    cand = np.asarray(candidate.convert("RGB").crop(support), dtype=np.float32) / 255.0
    if ref.size == 0:
        return 0.0
    return float(np.mean(np.abs(ref - cand)))


def make_matched_unreadable_control(
    base: Image.Image,
    readable: Image.Image,
    glyph_mask: Optional[Image.Image],
    variants: Sequence[str],
    rng: random.Random,
    min_visual_distance: float,
) -> Tuple[Image.Image, Image.Image, float, float, str, float, str]:
    """Restore the historical mirrored-only null control.

    The mirror is applied to the *actual* readable overlay recovered from the
    matched image pair, so handwriting remains matched in style and placement.
    Blur, pixelation, redaction, empty plates, torn glyphs and shuffled fragments
    are deliberately not assigned null/readability targets.
    """
    del rng
    requested = tuple(str(value) for value in variants)
    if requested != CONTROL_VARIANTS:
        raise ValueError(
            "Unreadable controls are intentionally fixed to mirrored-only; "
            f"received {requested!r}."
        )
    base_rgb = base.convert("RGB")
    readable_rgb = readable.convert("RGB")
    tile, glyph_crop, _full_support, bbox = _matched_overlay_layer(
        base_rgb, readable_rgb, glyph_mask
    )
    x0, y0, _x1, _y1 = bbox
    mirrored = ImageOps.mirror(tile)
    mirrored_mask = ImageOps.mirror(glyph_crop)

    def compose(candidate: Image.Image, candidate_mask: Image.Image) -> Tuple[Image.Image, Image.Image, float]:
        layer = Image.new("RGBA", base_rgb.size, (0, 0, 0, 0))
        layer.alpha_composite(candidate, (x0, y0))
        candidate_out = Image.alpha_composite(base_rgb.convert("RGBA"), layer).convert("RGB")
        candidate_full_mask = Image.new("L", base_rgb.size, 0)
        candidate_full_mask.paste(candidate_mask, (x0, y0), candidate_mask)
        candidate_distance = _crop_visual_distance(readable_rgb, candidate_out, bbox)
        return candidate_out, candidate_full_mask, candidate_distance

    out, full_mask, distance = compose(mirrored, mirrored_mask)
    fallback = "none"
    if distance < float(min_visual_distance):
        # Exact/symmetric text can be invariant to a plain horizontal mirror.
        # Preserve the mirrored-only control family, but give the mirrored crop a
        # tiny deterministic orientation offset, matching the variation already
        # present in the historical independently rendered mirror controls.
        for angle in (3.0, -3.0, 6.0, -6.0):
            candidate = mirrored.rotate(angle, resample=BICUBIC, expand=False)
            candidate_mask = mirrored_mask.rotate(angle, resample=BICUBIC, expand=False)
            candidate_out, candidate_full_mask, candidate_distance = compose(candidate, candidate_mask)
            if candidate_distance >= float(min_visual_distance):
                out, full_mask, distance = candidate_out, candidate_full_mask, candidate_distance
                fallback = f"mirror_rotation_{angle:+.1f}"
                break
    if distance < float(min_visual_distance):
        raise RuntimeError(
            "Could not construct a distinct mirrored control after deterministic "
            f"fallbacks: distance={distance:.6f} < {float(min_visual_distance):.6f}."
        )
    return out, full_mask, 1.0, 0.0, "mirrored", distance, fallback

def make_ocr_mask(row: Mapping[str, Any]) -> Image.Image:
    width = int(row.get("width") or 1)
    height = int(row.get("height") or 1)
    mask = Image.new("L", (width, height), 0)
    array = np.zeros((height, width), dtype=np.uint8)
    for item in row.get("ocr") or []:
        bbox = item.get("bbox_xywh_norm")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        x0 = max(0, min(width, round(x * width)))
        y0 = max(0, min(height, round(y * height)))
        x1 = max(x0, min(width, round((x + w) * width)))
        y1 = max(y0, min(height, round((y + h) * height)))
        array[y0:y1, x0:x1] = 255
    return Image.fromarray(array, mode="L")


def pack_packets(packets: Sequence[Packet]) -> PackedBatch:
    """Pack packets and merge exact duplicate captions into shared positives."""
    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    mask_weights: List[float] = []
    present: List[float] = []
    readable: List[float] = []
    captions: List[str] = []
    caption_to_index: Dict[str, int] = {}
    positive_pairs: List[Tuple[int, int]] = []
    source_triplets: List[Tuple[int, int, List[int]]] = []
    auto_triplets: List[Tuple[int, int, int]] = []
    invariance_pairs: List[Tuple[int, int]] = []
    metadata: List[Dict[str, Any]] = []
    image_offset = 0

    for packet in packets:
        ni = len(packet.images)
        images.extend(packet.images)
        masks.extend(packet.patch_masks)
        mask_weights.extend(packet.mask_weights)
        present.extend(packet.present_targets)
        readable.extend(packet.readable_targets)
        metadata.append(packet.metadata)

        local_to_global: Dict[int, int] = {}
        for local_index, caption in enumerate(packet.captions):
            if caption not in caption_to_index:
                caption_to_index[caption] = len(captions)
                captions.append(caption)
            local_to_global[local_index] = caption_to_index[caption]

        positive_indices = packet.positive.nonzero(as_tuple=False)
        for local_image, local_caption in positive_indices.tolist():
            positive_pairs.append((
                image_offset + int(local_image),
                local_to_global[int(local_caption)],
            ))

        for candidate_index, positive_image, negative_images in packet.source_triplets:
            source_triplets.append((
                local_to_global[candidate_index],
                image_offset + positive_image,
                [image_offset + idx for idx in negative_images],
            ))
        for image_index, positive_caption, negative_caption in packet.auto_triplets:
            auto_triplets.append((
                image_offset + image_index,
                local_to_global[positive_caption],
                local_to_global[negative_caption],
            ))
        invariance_pairs.extend(
            (image_offset + clean_idx, image_offset + variant_idx)
            for clean_idx, variant_idx in packet.invariance_pairs
        )
        image_offset += ni

    positive = torch.zeros((len(images), len(captions)), dtype=torch.bool)
    for image_index, caption_index in positive_pairs:
        positive[image_index, caption_index] = True

    return PackedBatch(
        images=torch.stack(images),
        patch_masks=torch.stack(masks),
        mask_weights=torch.tensor(mask_weights, dtype=torch.float32),
        present_targets=torch.tensor(present, dtype=torch.float32),
        readable_targets=torch.tensor(readable, dtype=torch.float32),
        captions=captions,
        positive=positive,
        source_triplets=source_triplets,
        auto_triplets=auto_triplets,
        invariance_pairs=invariance_pairs,
        packet_metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Dataset sources
# -----------------------------------------------------------------------------


class ImageNetPacketSource:
    def __init__(
        self,
        digital_root: Path,
        handwriting_root: Path,
        split: str,
        image_size: int,
        patch_count: int,
        handwriting_probability: float,
        augment_flip_probability: float,
        control_font_paths: Sequence[str] = (),
    ):
        self.digital_root = digital_root
        self.handwriting_root = handwriting_root
        self.split = split
        self.image_size = image_size
        self.patch_count = patch_count
        self.handwriting_probability = handwriting_probability
        self.augment_flip_probability = augment_flip_probability
        self.control_font_paths = resolve_control_fonts(control_font_paths)

        digital_rows = read_jsonl(digital_root / "manifests" / f"{split}.jsonl")
        self.digital_by_group: Dict[str, List[Dict[str, Any]]] = {}
        for row in digital_rows:
            self.digital_by_group.setdefault(str(row["group_id"]), []).append(row)

        self.handwriting_by_group: Dict[str, List[Dict[str, Any]]] = {}
        handwriting_manifest = handwriting_root / "manifests" / f"{split}.jsonl"
        if handwriting_manifest.is_file():
            for row in read_jsonl(handwriting_manifest):
                self.handwriting_by_group.setdefault(str(row["group_id"]), []).append(row)

        self.group_ids = sorted(self.digital_by_group)
        if not self.group_ids:
            raise RuntimeError(f"No ImageNet counterfactual groups found in {digital_root}")

    @staticmethod
    def _relation(rows: Sequence[Dict[str, Any]], relation: str) -> List[Dict[str, Any]]:
        return [row for row in rows if str(row.get("relation")) == relation]

    def _choose_readable(
        self,
        group_id: str,
        relation: str,
        rng: random.Random,
    ) -> Dict[str, Any]:
        digital = self._relation(self.digital_by_group[group_id], relation)
        handwriting = self._relation(self.handwriting_by_group.get(group_id, []), relation)
        if handwriting and rng.random() < self.handwriting_probability:
            return rng.choice(handwriting)
        if digital:
            return rng.choice(digital)
        if handwriting:
            return rng.choice(handwriting)
        raise RuntimeError(f"Group {group_id} has no {relation} readable variant")

    def _resolve(self, row: Mapping[str, Any]) -> Tuple[Path, Optional[Path], str]:
        source = str(row.get("source") or "")
        if source == "imagenet_clip_text_handwriting" or "handwritten_text" in row:
            root = self.handwriting_root
            image_rel = str(row["image"])
            mask_rel = str(row["mask"])
            text = str(row.get("handwritten_text") or "")
        else:
            root = self.digital_root
            image_rel = str(row["image_relpath"])
            mask_rel = row.get("mask_relpath")
            mask_rel = str(mask_rel) if mask_rel else ""
            text = str(row.get("overlay_text") or "")
        return root / image_rel, (root / mask_rel if mask_rel else None), text

    def sample(self, rng: random.Random) -> Packet:
        group_id = rng.choice(self.group_ids)
        rows = self.digital_by_group[group_id]
        clean_rows = [row for row in rows if str(row.get("relation")) == "none"]
        if not clean_rows:
            raise RuntimeError(f"Incomplete digital group {group_id}: no clean image")
        clean = rng.choice(clean_rows)
        support = self._choose_readable(group_id, "supportive", rng)
        adversarial = self._choose_readable(group_id, "adversarial", rng)

        clean_path, _, _ = self._resolve(clean)
        support_path, support_mask_path, support_text = self._resolve(support)
        adversarial_path, adversarial_mask_path, adversarial_text = self._resolve(adversarial)
        clean_pil = Image.open(clean_path).convert("RGB")

        pseudo_text = matched_pseudoword(adversarial_text, rng)
        render_plan = make_control_render_plan(
            clean_pil.size, (adversarial_text, pseudo_text), rng, self.control_font_paths
        )
        pseudo_pil, pseudo_mask = render_control_word(
            clean_pil, pseudo_text, render_plan, mirror_word=False
        )
        mirror_pil, mirror_mask = render_control_word(
            clean_pil, adversarial_text, render_plan, mirror_word=True
        )

        source_items: List[Tuple[Image.Image, Optional[Image.Image]]] = [
            (clean_pil, None),
            (Image.open(support_path), Image.open(support_mask_path) if support_mask_path and support_mask_path.is_file() else None),
            (Image.open(adversarial_path), Image.open(adversarial_mask_path) if adversarial_mask_path and adversarial_mask_path.is_file() else None),
            (pseudo_pil, pseudo_mask),
            (mirror_pil, mirror_mask),
        ]
        images: List[torch.Tensor] = []
        patch_masks: List[torch.Tensor] = []
        for image, mask in source_items:
            tensor, transformed_mask = image_and_mask_transform(
                image, mask, self.image_size, flip=False
            )
            images.append(tensor)
            patch_masks.append(mask_to_patch_targets(transformed_mask, self.patch_count))

        primary_label = str(clean.get("primary_label") or support.get("primary_label") or "object")
        captions = [
            f"a photo of a {primary_label}",
            f"<notext> a photo of a {primary_label}",
            f"<text> {support_text}",
            f"<text> {adversarial_text}",
            f"<text> {pseudo_text}",
            f'a photo of a {primary_label} with the text "{support_text}"',
            f'a photo of a {primary_label} with the text "{adversarial_text}"',
            f"a photo of a {adversarial_text}",
            "<text> <null>",
        ]

        positive = torch.zeros((5, len(captions)), dtype=torch.bool)
        positive[:, 0] = True
        positive[:, 1] = True
        positive[1, 2] = True
        positive[2, 3] = True
        # Single-variable ablation: readable pseudowords now supervise their
        # literal forced-read candidate instead of the historical <text> <null>.
        positive[3, 4] = True
        positive[1, 5] = True
        positive[2, 6] = True
        # Abstention remains positive only for clean and mirrored-unreadable rows.
        positive[0, 8] = True
        positive[4, 8] = True

        source_triplets = [
            (2, 1, [0]),
            (3, 2, [0, 3, 4]),
            (4, 3, [0, 1, 2, 4]),
            (8, 0, [1, 2, 3]),
            (8, 4, [1, 2, 3]),
        ]
        auto_triplets = [
            # On the adversarial overlay, ordinary object semantics must outrank
            # the visibly written class.  The same word remains positive in <text>.
            (2, 0, 7),
        ]

        return Packet(
            images=images,
            patch_masks=patch_masks,
            mask_weights=[1.0] * 5,
            present_targets=[0.0, 1.0, 1.0, 1.0, 1.0],
            readable_targets=[0.0, 1.0, 1.0, 1.0, 0.0],
            captions=captions,
            positive=positive,
            source_triplets=source_triplets,
            auto_triplets=auto_triplets,
            invariance_pairs=[(0, 1), (0, 2), (0, 3), (0, 4)],
            metadata={
                "source": "imagenet",
                "group_id": group_id,
                "support_renderer": str(support.get("renderer")),
                "adversarial_renderer": str(adversarial.get("renderer")),
                "support_text": support_text,
                "adversarial_text": adversarial_text,
                "pseudo_text": pseudo_text,
                "control_word": adversarial_text,
                "adversarial_any_negative": f"a photo of a {adversarial_text}",
            },
        )


class ContinuousMathImageNetPacketSource:
    """Continuous typography + procedural math hard-negative curriculum over ImageNet clean rows."""

    def __init__(
        self,
        digital_root: Path,
        wnid_json: Path,
        split: str,
        image_size: int,
        patch_count: int,
        control_font_paths: Sequence[str],
        words: str,
        math_families: str,
        crop_probability: float,
    ):
        grid = int(round(math.sqrt(patch_count)))
        if grid * grid != patch_count:
            raise ValueError(f"Patch count {patch_count} is not square")
        patch_size = image_size // grid
        fonts = resolve_control_fonts(control_font_paths)
        families = [x.strip() for x in str(math_families).split(",") if x.strip()]
        self.builder = ContinuousTextMathImageNetBuilder(
            digital_root=digital_root,
            wnid_json=wnid_json,
            split=split,
            image_size=image_size,
            patch_size=patch_size,
            font_paths=fonts,
            words=words,
            math_families=families,
            crop_probability=crop_probability,
        )
        self.image_size = int(image_size)
        self.patch_count = int(patch_count)
        self.group_ids = sorted(self.builder.group_meta)

    def sample(self, rng: random.Random) -> Packet:
        data = self.builder.sample(rng)
        images: List[torch.Tensor] = []
        patch_masks: List[torch.Tensor] = []
        for image, mask in zip(data.images, data.masks):
            tensor, transformed_mask = image_and_mask_transform(
                image, mask, self.image_size, flip=False
            )
            images.append(tensor)
            patch_masks.append(mask_to_patch_targets(transformed_mask, self.patch_count))
        positive = torch.zeros((len(images), len(data.captions)), dtype=torch.bool)
        for image_index, caption_index in data.positive_pairs:
            positive[int(image_index), int(caption_index)] = True
        return Packet(
            images=images,
            patch_masks=patch_masks,
            mask_weights=list(data.mask_weights),
            present_targets=list(data.present_targets),
            readable_targets=list(data.readable_targets),
            captions=list(data.captions),
            positive=positive,
            source_triplets=list(data.source_triplets),
            auto_triplets=list(data.auto_triplets),
            invariance_pairs=list(data.invariance_pairs),
            metadata=dict(data.metadata),
        )

    def coverage(self) -> Dict[str, Any]:
        return self.builder.coverage()

    def save_preview_grids(self, out_dir: Path, count: int, seed: int) -> List[Path]:
        return self.builder.save_preview_grids(out_dir, count=count, seed=seed)


class ClevrPropertyPacketSource:
    """Additive CLEVR color/count/shape literal-binding curriculum.

    Placement is computed from Canny edges on the already-square local image.
    Original CLEVR geometry is intentionally never consumed.
    """

    def __init__(
        self,
        image_root: Path,
        metadata_jsonl: Path,
        split: str,
        image_size: int,
        patch_count: int,
        control_font_paths: Sequence[str],
        colored_text_probability: float,
        ood_color_probability: float,
        standalone_count_probability: float,
        canny_low: int,
        canny_high: int,
        canny_dilate_px: int,
        placement_margin_px: int,
        max_obstacle_fraction: float,
        placement_stride_px: int,
        fill_contours: bool,
        min_contour_area_fraction: float,
        min_font_size: int,
        max_font_size: int,
    ):
        self.image_size = int(image_size)
        self.patch_count = int(patch_count)
        fonts = resolve_control_fonts(control_font_paths)
        self.builder = ClevrPropertyBindingBuilder(
            image_root=Path(image_root),
            metadata_jsonl=Path(metadata_jsonl),
            split=split,
            font_paths=fonts,
            colored_text_probability=colored_text_probability,
            ood_color_probability=ood_color_probability,
            standalone_count_probability=standalone_count_probability,
            canny_low=canny_low,
            canny_high=canny_high,
            canny_dilate_px=canny_dilate_px,
            placement_margin_px=placement_margin_px,
            max_obstacle_fraction=max_obstacle_fraction,
            placement_stride_px=placement_stride_px,
            fill_contours=fill_contours,
            min_contour_area_fraction=min_contour_area_fraction,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
        )

    def sample(self, rng: random.Random) -> Packet:
        data = self.builder.sample(rng)
        images: List[torch.Tensor] = []
        patch_masks: List[torch.Tensor] = []
        for image, mask in zip(data.images, data.masks):
            tensor, transformed_mask = image_and_mask_transform(
                image, mask, self.image_size, flip=False
            )
            images.append(tensor)
            patch_masks.append(mask_to_patch_targets(transformed_mask, self.patch_count))
        positive = torch.zeros((len(images), len(data.captions)), dtype=torch.bool)
        for image_index, caption_index in data.positive_pairs:
            positive[int(image_index), int(caption_index)] = True
        return Packet(
            images=images,
            patch_masks=patch_masks,
            mask_weights=list(data.mask_weights),
            present_targets=list(data.present_targets),
            readable_targets=list(data.readable_targets),
            captions=list(data.captions),
            positive=positive,
            source_triplets=list(data.source_triplets),
            auto_triplets=list(data.auto_triplets),
            invariance_pairs=list(data.invariance_pairs),
            metadata=dict(data.metadata),
        )

    def coverage(self) -> Dict[str, Any]:
        return self.builder.coverage()

    def save_previews(self, out_dir: Path, count: int, seed: int) -> List[Path]:
        return self.builder.save_previews(out_dir, count=count, seed=seed)


class TextCapsPacketSource:
    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int,
        patch_count: int,
        weak_mask_weight: float,
        augment_flip_probability: float,
    ):
        self.root = root
        self.image_size = image_size
        self.patch_count = patch_count
        self.weak_mask_weight = weak_mask_weight
        self.augment_flip_probability = augment_flip_probability
        self.rows = read_jsonl(root / "manifests" / f"{split}.jsonl")
        if not self.rows:
            raise RuntimeError(f"No TextCaps rows found in {root}")

    @staticmethod
    def _overlap_word(caption: str, ocr_tokens: Sequence[str]) -> Optional[str]:
        caption_words = set(normalize_words(caption))
        candidates: List[str] = []
        for token in ocr_tokens:
            normalized = normalize_words(str(token))
            if len(normalized) == 1 and len(normalized[0]) >= 3 and normalized[0] in caption_words:
                candidates.append(str(token).strip())
        return max(candidates, key=len) if candidates else None

    def sample(self, rng: random.Random) -> Packet:
        row = rng.choice(self.rows)
        captions = [str(c) for c in row.get("captions") or [] if str(c).strip()]
        if not captions:
            raise RuntimeError(f"TextCaps row {row.get('image_id')} has no human caption")
        caption = rng.choice(captions)
        ocr_tokens = [str(x) for x in row.get("ocr_tokens") or []]
        overlap = self._overlap_word(caption, ocr_tokens)
        caption_words = set(normalize_words(caption))
        ocr_words = {word for token in ocr_tokens for word in normalize_words(token)}
        caption_uses_ocr = bool(caption_words & ocr_words)

        image = Image.open(self.root / str(row["image"]))
        mask = make_ocr_mask(row)
        tensor, transformed_mask = image_and_mask_transform(
            image,
            mask,
            self.image_size,
            # OCR boxes and readable labels are orientation-specific.
            flip=False,
        )
        candidate_texts = [caption]
        if overlap is not None:
            candidate_texts.append(f"<text> {overlap}")
        if not caption_uses_ocr:
            candidate_texts.append(f"<notext> {caption}")
        null_index = len(candidate_texts)
        candidate_texts.append("<text> <null>")
        has_ocr = bool(row.get("ocr"))
        positive = torch.ones((1, len(candidate_texts)), dtype=torch.bool)
        # <null> means literal-read abstention, not generic uncertainty.
        positive[0, null_index] = not has_ocr
        return Packet(
            images=[tensor],
            patch_masks=[mask_to_patch_targets(transformed_mask, self.patch_count)],
            mask_weights=[self.weak_mask_weight if has_ocr else 0.0],
            present_targets=[1.0 if has_ocr else 0.0],
            readable_targets=[1.0 if has_ocr else 0.0],
            captions=candidate_texts,
            positive=positive,
            metadata={"source": "textcaps", "image_id": row.get("image_id")},
        )



class CocoSprightPacketSource:
    """COCO-SPRIGHT content labels plus optional high-trust literal readings."""

    def __init__(
        self,
        root: Path,
        normal_json: Path,
        reworded_json: Path,
        image_size: int,
        patch_count: int,
        changed_probability: float,
        augment_flip_probability: float,
        trusted_reading_json: Optional[Path] = None,
        trusted_reading_probability: float = 0.25,
        trusted_mask_weight: float = 0.50,
    ):
        self.root = root
        self.image_size = image_size
        self.patch_count = patch_count
        self.changed_probability = changed_probability
        self.augment_flip_probability = augment_flip_probability
        self.trusted_reading_probability = float(np.clip(trusted_reading_probability, 0.0, 1.0))
        self.trusted_mask_weight = float(max(0.0, trusted_mask_weight))
        normal = read_json(normal_json)
        reworded = read_json(reworded_json)
        self.reading_by_key = load_coco_reading_manifest(trusted_reading_json)
        normal_keys = {normalize_image_key(key) for key in normal}
        unmatched = sorted(set(self.reading_by_key) - normal_keys)
        if unmatched:
            raise ValueError(
                f"Trusted COCO-SPRIGHT manifest has {len(unmatched)} image key(s) absent "
                f"from the original label JSON; examples={unmatched[:5]}"
            )

        self.rows: List[Tuple[str, List[str], List[str], bool]] = []
        self.changed_rows: List[Tuple[str, List[str], List[str], bool]] = []
        for key, value in normal.items():
            normalized_key = normalize_image_key(key)
            if normalized_key in self.reading_by_key:
                continue
            originals = labels_from_value(value)
            rewrites = labels_from_value(reworded.get(key, value))
            if not originals:
                continue
            if len(rewrites) < len(originals):
                rewrites = [*rewrites, *originals[len(rewrites):]]
            changed = any(
                originals[index].strip() != rewrites[index].strip()
                for index in range(min(len(originals), len(rewrites)))
            )
            row = (str(key), originals, rewrites, changed)
            self.rows.append(row)
            if changed:
                self.changed_rows.append(row)
        self.reading_rows = list(self.reading_by_key.values())
        if not self.rows and not self.reading_rows:
            raise RuntimeError(f"No COCO-SPRIGHT labels found in {normal_json}")
        print(
            f"[coco] content_rows={len(self.rows)} trusted_reading={len(self.reading_rows)} "
            f"trusted_probability={self.trusted_reading_probability:.3f}"
        )

    @staticmethod
    def _caption_pair(
        originals: Sequence[str], rewrites: Sequence[str], rng: random.Random
    ) -> Tuple[str, str, bool, int]:
        index = rng.randrange(len(originals))
        original = str(originals[index])
        rewritten = str(rewrites[index] if index < len(rewrites) else original)
        return original, rewritten, original.strip() != rewritten.strip(), index

    def _sample_trusted(self, rng: random.Random) -> Packet:
        row = rng.choice(self.reading_rows)
        key = str(row["image_key"])
        image_path = self.root / key
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        targets = [dict(target) for target in row.get("reading_targets") or []]
        words = target_texts(targets)
        if not words:
            raise RuntimeError(f"Trusted COCO row has no reading targets: {key}")
        mask = mask_from_targets(image.size, targets)
        tensor, transformed_mask = image_and_mask_transform(
            image, mask, self.image_size, flip=False
        )
        originals = [str(value) for value in row.get("original_labels") or [] if str(value).strip()]
        rewrites = [str(value) for value in row.get("rewritten_labels") or [] if str(value).strip()]
        if not originals:
            raise RuntimeError(f"Trusted COCO row has no original labels: {key}")
        original, rewritten, changed, caption_index = self._caption_pair(originals, rewrites, rng)
        captions = [original]
        if changed:
            captions.extend([rewritten, f"<notext> {rewritten}"])
        captions.extend(f"<text> {word}" for word in words)
        null_index = len(captions)
        captions.append("<text> <null>")
        positive = torch.ones((1, len(captions)), dtype=torch.bool)
        positive[0, null_index] = False
        return Packet(
            images=[tensor],
            patch_masks=[mask_to_patch_targets(transformed_mask, self.patch_count)],
            mask_weights=[self.trusted_mask_weight],
            present_targets=[1.0],
            readable_targets=[1.0],
            captions=captions,
            positive=positive,
            metadata={
                "source": "coco_spright",
                "key": key,
                "changed": changed,
                "caption_index": caption_index,
                "trusted_reading": True,
                "reading_targets": words,
            },
        )

    def _sample_content(self, rng: random.Random) -> Packet:
        if self.changed_rows and rng.random() < self.changed_probability:
            key, originals, rewrites, _ = rng.choice(self.changed_rows)
        else:
            key, originals, rewrites, _ = rng.choice(self.rows)
        original, rewritten, changed, caption_index = self._caption_pair(originals, rewrites, rng)
        with Image.open(self.root / key) as source:
            image = source.convert("RGB")
        tensor, transformed_mask = image_and_mask_transform(
            image,
            None,
            self.image_size,
            flip=rng.random() < self.augment_flip_probability,
        )
        captions = [original]
        if changed:
            captions.extend([rewritten, f"<notext> {rewritten}"])
        return Packet(
            images=[tensor],
            patch_masks=[mask_to_patch_targets(transformed_mask, self.patch_count)],
            mask_weights=[0.0],
            present_targets=[-1.0],
            readable_targets=[-1.0],
            captions=captions,
            positive=torch.ones((1, len(captions)), dtype=torch.bool),
            metadata={
                "source": "coco_spright",
                "key": key,
                "changed": changed,
                "caption_index": caption_index,
                "trusted_reading": False,
            },
        )

    def sample(self, rng: random.Random) -> Packet:
        if self.reading_rows and (
            not self.rows or rng.random() < self.trusted_reading_probability
        ):
            return self._sample_trusted(rng)
        return self._sample_content(rng)





class SourceMixer:
    def __init__(self, sources: Dict[str, Any], weights: Dict[str, float], seed: int):
        self.sources = sources
        self.weights = weights
        self.rng = random.Random(seed)

    def set_weights(self, weights: Dict[str, float]) -> None:
        self.weights = weights

    def sample_packet(self) -> Packet:
        names = [name for name, weight in self.weights.items() if weight > 0 and name in self.sources]
        weights = [self.weights[name] for name in names]
        if not names:
            raise RuntimeError("No enabled dataset source")
        name = self.rng.choices(names, weights=weights, k=1)[0]
        return self.sources[name].sample(self.rng)

    def sample_packets(self, target_images: int) -> List[Packet]:
        packets: List[Packet] = []
        count = 0
        while count < target_images:
            packet = self.sample_packet()
            packets.append(packet)
            count += len(packet.images)
        return packets

    def sample_batch(self, target_images: int) -> PackedBatch:
        return pack_packets(self.sample_packets(target_images))


def sample_training_batch_with_additive_clevr(
    mixer: SourceMixer,
    target_images: int,
    clevr_source: Optional[ClevrPropertyPacketSource],
    enabled: bool,
    probability: float,
    packets_per_batch: int,
) -> PackedBatch:
    """Preserve the existing source mix, then append CLEVR packets on top.

    CLEVR does not receive a SourceMixer probability and therefore never steals
    probability mass from ImageNet / ImageNet-math / TextCaps / COCO.
    """
    packets = mixer.sample_packets(target_images)
    if enabled and clevr_source is not None and packets_per_batch > 0:
        if mixer.rng.random() < float(probability):
            for _ in range(int(packets_per_batch)):
                packets.append(clevr_source.sample(mixer.rng))
    return pack_packets(packets)


# -----------------------------------------------------------------------------
# Losses and phase configuration
# -----------------------------------------------------------------------------


@fp32_function
def multi_positive_clip_loss(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    positive = positive.to(device=logits.device, dtype=torch.bool)
    neg_inf = torch.finfo(logits.dtype).min
    row_valid = positive.any(dim=1)
    col_valid = positive.any(dim=0)
    losses: List[torch.Tensor] = []
    if row_valid.any():
        numerator = torch.logsumexp(logits.masked_fill(~positive, neg_inf), dim=1)
        denominator = torch.logsumexp(logits, dim=1)
        losses.append((denominator[row_valid] - numerator[row_valid]).mean())
    if col_valid.any():
        numerator = torch.logsumexp(logits.masked_fill(~positive, neg_inf), dim=0)
        denominator = torch.logsumexp(logits, dim=0)
        losses.append((denominator[col_valid] - numerator[col_valid]).mean())
    if not losses:
        return logits.sum() * 0.0
    return sum(losses) / len(losses)


@fp32_function
def weighted_mask_loss(
    glyph_logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    valid = weights > 0
    if not valid.any():
        return glyph_logits.sum() * 0.0
    logits = glyph_logits[valid]
    targets = targets[valid].to(logits.dtype)
    sample_weights = weights[valid].to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
    probs = logits.sigmoid()
    intersection = (probs * targets).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + 1.0) / (probs.sum(dim=1) + targets.sum(dim=1) + 1.0)
    return ((bce + dice) * sample_weights).sum() / sample_weights.sum().clamp_min(1.0e-6)


@fp32_function
def known_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    valid = targets >= 0
    if not valid.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid], targets[valid].to(logits.dtype))


@fp32_function
def source_ranking_loss(
    read_logits: torch.Tensor,
    triplets: Sequence[Tuple[int, int, List[int]]],
    margin: float,
) -> torch.Tensor:
    terms: List[torch.Tensor] = []
    for candidate, positive_image, negatives in triplets:
        positive_score = read_logits[positive_image, candidate]
        for negative_image in negatives:
            negative_score = read_logits[negative_image, candidate]
            terms.append(F.softplus(margin + negative_score - positive_score))
    if not terms:
        return read_logits.sum() * 0.0
    return torch.stack(terms).mean()


@fp32_function
def auto_hard_negative_loss(
    logits: torch.Tensor,
    triplets: Sequence[Tuple[int, int, int]],
    margin: float,
) -> torch.Tensor:
    """Direct <any> ranking: intended caption must beat a hard negative."""
    if not triplets:
        return logits.sum() * 0.0
    terms = [
        F.softplus(
            margin
            + logits[image_index, negative_caption]
            - logits[image_index, positive_caption]
        )
        for image_index, positive_caption, negative_caption in triplets
    ]
    return torch.stack(terms).mean()


@fp32_function
def invariance_loss(
    content_embeddings: torch.Tensor,
    pairs: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    if not pairs:
        return content_embeddings.sum() * 0.0
    normalized = F.normalize(content_embeddings, dim=-1)
    terms = [
        1.0 - (normalized[variant_idx] * normalized[clean_idx].detach()).sum()
        for clean_idx, variant_idx in pairs
    ]
    return torch.stack(terms).mean()


@fp32_function
def route_rent(route_gate: torch.Tensor, mode_ids: torch.Tensor) -> torch.Tensor:
    any_mask = mode_ids.eq(0)
    if not any_mask.any():
        return route_gate.sum() * 0.0
    return route_gate[:, any_mask].mean()


@fp32_function
def read_null_attention_objective(
    details: Mapping[str, Any],
    positive: torch.Tensor,
    null_text_token_id: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Candidate-specific supervision for the late ViT READ_NULL dustbin.

    Only explicit, non-null ``<text>`` candidates are supervised. A positive
    image/candidate pair should avoid READ_NULL; every contrastive negative for
    that literal candidate should prefer READ_NULL. Positive and negative terms
    are averaged separately so the large cross-batch negative bank cannot swamp
    the sparse literal positives.
    """
    attention = details.get("read_null_attention")
    if not isinstance(attention, torch.Tensor):
        zero = details["logits_per_image"].sum() * 0.0
        return zero, {
            "read_null_attention_supported": 0.0,
            "read_null_attention_unsupported": 0.0,
            "read_null_attention_gap": 0.0,
            "read_null_accuracy": 0.0,
            "read_null_supported_count": 0.0,
            "read_null_unsupported_count": 0.0,
        }

    mode_ids = details["mode_ids"]
    read_tokens = details["read_tokens"]
    explicit_literal = mode_ids.eq(1) & ~read_tokens.eq(int(null_text_token_id)).any(dim=-1)
    valid = explicit_literal[None, :].expand_as(attention)
    truth = positive.to(device=attention.device, dtype=torch.bool)
    supported = valid & truth
    unsupported = valid & ~truth
    p = attention.float().clamp(min=1.0e-6, max=1.0 - 1.0e-6)

    terms: List[torch.Tensor] = []
    if bool(supported.any()):
        terms.append(-torch.log1p(-p[supported]).mean())
    if bool(unsupported.any()):
        terms.append(-torch.log(p[unsupported]).mean())
    loss = torch.stack(terms).mean() if terms else p.sum() * 0.0

    supported_mean = float(p[supported].detach().mean()) if bool(supported.any()) else 0.0
    unsupported_mean = float(p[unsupported].detach().mean()) if bool(unsupported.any()) else 0.0
    acc_terms: List[torch.Tensor] = []
    if bool(supported.any()):
        acc_terms.append((p[supported] < 0.5).float().mean())
    if bool(unsupported.any()):
        acc_terms.append((p[unsupported] >= 0.5).float().mean())
    balanced_acc = float(torch.stack(acc_terms).mean().detach()) if acc_terms else 0.0

    metrics: Dict[str, float] = {
        "read_null_attention_supported": supported_mean,
        "read_null_attention_unsupported": unsupported_mean,
        "read_null_attention_gap": unsupported_mean - supported_mean,
        "read_null_accuracy": balanced_acc,
        "read_null_supported_count": float(supported.sum().item()),
        "read_null_unsupported_count": float(unsupported.sum().item()),
    }

    # Per late tap diagnostics (head-mean), useful for seeing whether B20/B21
    # independently learn the dustbin or one tap carries the whole behavior.
    read_details = details.get("read_details")
    if isinstance(read_details, Mapping):
        read_candidate_mask = ~mode_ids.eq(2)
        local_literal = explicit_literal[read_candidate_mask]
        local_truth = truth[:, read_candidate_mask]
        local_valid = local_literal[None, :].expand_as(local_truth)
        local_supported = local_valid & local_truth
        local_unsupported = local_valid & ~local_truth
        per_block = read_details.get("per_block")
        if isinstance(per_block, Mapping):
            for block, block_details in per_block.items():
                if not isinstance(block_details, Mapping):
                    continue
                block_attn = block_details.get("read_null_attention")
                if not isinstance(block_attn, torch.Tensor):
                    continue
                block_p = block_attn.float().mean(dim=-1).clamp(0.0, 1.0)
                prefix = f"read_null_b{int(block)}"
                metrics[f"{prefix}_supported"] = (
                    float(block_p[local_supported].detach().mean())
                    if bool(local_supported.any()) else 0.0
                )
                metrics[f"{prefix}_unsupported"] = (
                    float(block_p[local_unsupported].detach().mean())
                    if bool(local_unsupported.any()) else 0.0
                )
    return loss, metrics


def _mean_offdiag_cosine(vectors: torch.Tensor) -> float:
    vectors = F.normalize(vectors.detach().float(), dim=-1, eps=1.0e-12)
    n = int(vectors.shape[0])
    if n <= 1:
        return 1.0
    summed = vectors.sum(dim=0)
    value = (summed.square().sum() - float(n)) / float(n * (n - 1))
    return float(value.clamp(min=-1.0, max=1.0))


def read_null_token_metrics(model: torch.nn.Module, details: Mapping[str, Any]) -> Dict[str, float]:
    if not bool(getattr(model, "read_null_enabled", False)):
        return {}
    token = getattr(model.visual, "read_null_token", None)
    if not isinstance(token, torch.Tensor):
        raise RuntimeError("READ_NULL enabled but visual.read_null_token is missing")
    metrics: Dict[str, float] = {
        "read_null_token_param_norm": float(token.detach().float().norm()),
    }
    states = details.get("visual_states")
    if isinstance(states, Mapping):
        for block in model.read_implant.tap_block_list():
            state = states.get(int(block))
            if not isinstance(state, torch.Tensor):
                continue
            value = state[:, -1, :]
            metrics[f"read_null_token_norm_b{int(block)}"] = float(
                value.detach().float().norm(dim=-1).mean()
            )
            metrics[f"read_null_token_cos_b{int(block)}"] = _mean_offdiag_cosine(value)
    final_tokens = details.get("visual_final_tokens")
    if isinstance(final_tokens, torch.Tensor):
        value = final_tokens[:, -1, :]
        metrics["read_null_token_norm_final"] = float(
            value.detach().float().norm(dim=-1).mean()
        )
        metrics["read_null_token_cos_final"] = _mean_offdiag_cosine(value)
    null_query_attn = details.get("null_query_read_null_attention")
    if isinstance(null_query_attn, torch.Tensor):
        metrics["read_null_attention_null_query"] = float(
            null_query_attn.detach().float().mean()
        )
    vit_attention = details.get("read_null_vit_attention")
    if isinstance(vit_attention, Mapping):
        for block, block_stats in vit_attention.items():
            if not isinstance(block_stats, Mapping):
                continue
            for key in ("cls_to_null", "patch_to_null", "null_to_patches", "null_to_cls", "null_to_self"):
                value = block_stats.get(key)
                if isinstance(value, torch.Tensor):
                    metrics[f"read_null_vit_b{int(block)}_{key}"] = float(
                        value.detach().float().mean()
                    )
    return metrics


def _read_csv_rows_read_null(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_read_null_diagnostic_plots(out_dir: Path) -> None:
    rows = _read_csv_rows_read_null(out_dir / "training_log.csv")
    if not rows or "read_null_attention_supported" not in rows[0]:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def series(key: str):
        result = []
        for row in rows:
            raw = row.get(key, "")
            if raw in (None, ""):
                continue
            try:
                result.append((int(float(row["step"])), float(raw)))
            except (KeyError, ValueError):
                continue
        return result

    supported = series("read_null_attention_supported")
    unsupported = series("read_null_attention_unsupported")
    if supported or unsupported:
        fig, ax = plt.subplots(figsize=(12, 6))
        if supported:
            ax.plot([x for x, _ in supported], [y for _, y in supported], label="supported literal")
        if unsupported:
            ax.plot([x for x, _ in unsupported], [y for _, y in unsupported], label="unsupported literal")
        ax.axhline(0.5, linestyle="--", linewidth=1.0, label="decision midpoint")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("training step")
        ax.set_ylabel("READ_NULL attention mass")
        ax.set_title("Candidate-conditioned READ_NULL attention")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "read_null_attention_training.png", dpi=180)
        plt.close(fig)

    norm_keys = [key for key in rows[0] if key.startswith("read_null_token_norm_")]
    if norm_keys:
        fig, ax = plt.subplots(figsize=(12, 6))
        for key in norm_keys:
            values = series(key)
            if values:
                ax.plot([x for x, _ in values], [y for _, y in values], label=key.replace("read_null_token_norm_", ""))
        ax.set_xlabel("training step")
        ax.set_ylabel("L2 norm")
        ax.set_title("READ_NULL residual norm by location")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "read_null_token_norms_training.png", dpi=180)
        plt.close(fig)

    cos_keys = [key for key in rows[0] if key.startswith("read_null_token_cos_")]
    if cos_keys:
        fig, ax = plt.subplots(figsize=(12, 6))
        for key in cos_keys:
            values = series(key)
            if values:
                ax.plot([x for x, _ in values], [y for _, y in values], label=key.replace("read_null_token_cos_", ""))
        ax.set_ylim(-1.02, 1.02)
        ax.set_xlabel("training step")
        ax.set_ylabel("mean off-diagonal cosine across images")
        ax.set_title("READ_NULL image dependence")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "read_null_token_cosine_training.png", dpi=180)
        plt.close(fig)

    vit_keys = [key for key in rows[0] if key.startswith("read_null_vit_b")]
    if vit_keys:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for key in vit_keys:
            values = series(key)
            if values:
                ax.plot([x for x, _ in values], [y for _, y in values], label=key.replace("read_null_vit_", ""))
        ax.set_xlabel("training step")
        ax.set_ylabel("mean self-attention mass")
        ax.set_ylim(bottom=0.0)
        ax.set_title("ViT interaction with READ_NULL (logging batches)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / "read_null_vit_self_attention.png", dpi=180)
        plt.close(fig)

    loss_values = series("loss_read_null")
    acc_values = series("read_null_accuracy")
    if loss_values or acc_values:
        fig, ax = plt.subplots(figsize=(12, 6))
        if loss_values:
            ax.plot([x for x, _ in loss_values], [y for _, y in loss_values], label="dustbin loss")
        if acc_values:
            ax.plot([x for x, _ in acc_values], [y for _, y in acc_values], label="balanced attention accuracy")
        ax.set_xlabel("training step")
        ax.set_title("READ_NULL warmup diagnostics")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "read_null_loss_and_accuracy.png", dpi=180)
        plt.close(fig)


def phase_loss_weights(args: argparse.Namespace, phase: str) -> Dict[str, float]:
    if phase == "1a":
        return {
            "mp": 0.0,
            "source": 0.0,
            "ortho": 0.0,
            "mask": args.mask_weight,
            "presence": args.presence_weight,
            "invariance": 0.0,
            "auto_negative": 0.0,
            "rent": 0.0,
            "read_null": 0.0,
        }
    if phase == "1b":
        return {
            "mp": args.mp_weight_1b,
            "source": args.source_weight,
            "ortho": args.ortho_weight,
            "mask": args.mask_weight,
            "presence": args.presence_weight,
            "invariance": 0.0,
            "auto_negative": 0.0,
            "rent": 0.0,
            "read_null": 0.0,
        }
    if phase == "1b5":
        return {
            "mp": args.mp_weight_1b5,
            "source": args.source_weight_1b5,
            "ortho": args.ortho_weight_1b5,
            "mask": args.mask_weight_1b5,
            "presence": args.presence_weight_1b5,
            "invariance": 0.0,
            "auto_negative": 0.0,
            "rent": 0.0,
            "read_null": args.read_null_weight_1b5,
        }
    return {
        "mp": args.mp_weight_1c,
        "source": args.source_weight_1c,
        "ortho": args.ortho_weight_1c,
        "mask": args.mask_weight_1c,
        "presence": args.presence_weight_1c,
        "invariance": args.invariance_weight,
        "auto_negative": args.auto_negative_weight,
        "rent": args.rent_weight,
        "read_null": args.read_null_weight_1c,
    }

def set_trainable_phase(
    model: torch.nn.Module,
    phase: str,
    router_only_1c: bool = False,
) -> List[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if phase == "1a":
        trainable_prefixes = (
            "read_implant.source_head.",
            "read_implant.source_tap_logits",
        )
    elif phase == "1b":
        trainable_prefixes = (
            "hard_text_embedding",
            "null_text_embedding",
            "read_implant.read_bridge.",
            "read_implant.read_tap_logits",
            "read_implant.orthographic_bridge.",
            "read_implant.ortho_tap_logits",
            "read_implant.source_head.",
            "read_implant.source_tap_logits",
            "read_implant.glyph_bias_beta",
            "read_implant.read_calibration_scale",
            "read_implant.null_abstain_weight",
        )
    elif phase == "1b5":
        # Continuous text/math warmup: adapt the complete literal-evidence stack while
        # keeping semantic authority/router and the CLIP towers frozen.
        trainable_prefixes = (
            "visual.read_null_token",
            "read_implant.read_bridge.",
            "read_implant.read_tap_logits",
            "read_implant.orthographic_bridge.",
            "read_implant.ortho_tap_logits",
            "read_implant.source_head.",
            "read_implant.source_tap_logits",
            "read_implant.glyph_bias_beta",
        )
    else:
        if router_only_1c:
            trainable_prefixes = (
                "read_implant.trust_router.",
                "read_implant.auto_read_scale",
            )
        else:
            trainable_prefixes = (
                "hard_text_embedding",
                "null_text_embedding",
                "visual.read_null_token",
                "read_implant.read_bridge.",
                "read_implant.read_tap_logits",
                "read_implant.orthographic_bridge.",
                "read_implant.ortho_tap_logits",
                "read_implant.source_head.",
                "read_implant.source_tap_logits",
                "read_implant.glyph_bias_beta",
                "read_implant.read_calibration_scale",
                "read_implant.null_abstain_weight",
                "read_implant.trust_router.",
                "read_implant.auto_read_scale",
                "read_implant.content_pool.",
                "read_implant.content_tap_logits",
            )

    names: List[str] = []
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(prefix) for prefix in trainable_prefixes):
            parameter.requires_grad_(True)
            names.append(name)
    return names

def optimizer_for_phase(model: torch.nn.Module, args: argparse.Namespace, phase: str):
    groups: Dict[str, List[torch.nn.Parameter]] = {
        "new": [], "read": [], "token": [], "content": [], "read_null": [], "router": []
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "visual.read_null_token":
            groups["read_null"].append(parameter)
        elif name.startswith("read_implant.trust_router."):
            groups["router"].append(parameter)
        elif name in {"hard_text_embedding", "null_text_embedding"}:
            groups["token"].append(parameter)
        elif (
            name.startswith("read_implant.read_bridge.")
            or name == "read_implant.read_tap_logits"
            or name.startswith("read_implant.orthographic_bridge.")
            or name == "read_implant.ortho_tap_logits"
        ):
            groups["read"].append(parameter)
        elif name.startswith("read_implant.content_pool.") or name == "read_implant.content_tap_logits":
            groups["content"].append(parameter)
        else:
            groups["new"].append(parameter)
    parameter_groups = []
    if groups["new"]:
        new_factor = args.new_lr_factor_1b5 if phase == "1b5" else 1.0
        parameter_groups.append({"params": groups["new"], "lr": args.lr_new * new_factor})
    if groups["read"]:
        if phase == "1c":
            factor = args.read_lr_factor_1c
        elif phase == "1b5":
            factor = args.read_lr_factor_1b5
        else:
            factor = 1.0
        parameter_groups.append({"params": groups["read"], "lr": args.lr_read * factor})
    if groups["token"]:
        factor = args.read_lr_factor_1c if phase == "1c" else 1.0
        parameter_groups.append({"params": groups["token"], "lr": args.lr_hard_token * factor})
    if groups["content"]:
        parameter_groups.append({"params": groups["content"], "lr": args.lr_content})
    if groups["read_null"]:
        parameter_groups.append({"params": groups["read_null"], "lr": args.lr_read_null_token})
    if groups["router"]:
        router_lr = args.router_aux_router_lr if (phase == "1c" and args.router_aux_enabled) else args.lr_new
        parameter_groups.append({"params": groups["router"], "lr": router_lr, "group_name": "router"})
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

def cosine_lr(optimizer, step: int, total_steps: int, warmup_fraction: float) -> None:
    warmup = max(1, round(total_steps * warmup_fraction))
    if step < warmup:
        multiplier = (step + 1) / warmup
    else:
        progress = (step - warmup) / max(1, total_steps - warmup)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        base_lr = group.setdefault("initial_lr", group["lr"])
        group["lr"] = base_lr * multiplier


# -----------------------------------------------------------------------------
# Checkpoint and model loading
# -----------------------------------------------------------------------------


def load_clip_package(args: argparse.Namespace):
    if args.clip_package_root is not None:
        sys.path.insert(0, str(args.clip_package_root.resolve()))
    clip_module = importlib.import_module(args.clip_package)
    return clip_module


def torch_load_trusted(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def migrate_implant_state(model: torch.nn.Module, state: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Load only tensors that belong to the split source/orthography branch."""
    out = {str(key): value for key, value in state.items() if torch.is_tensor(value)}
    stale_prefixes = ("presence_pool.", "glyph_head.", "auto_router.")
    stale_exact = {
        "readability_log_weight", "read_calibration_bias",
        "presence_tap_logits", "glyph_tap_logits",
    }
    for key in list(out):
        if key in stale_exact or any(key.startswith(prefix) for prefix in stale_prefixes):
            out.pop(key, None)
    defaults = model.read_implant.state_dict()
    for key, value in defaults.items():
        out.setdefault(key, value)
    return out


def _checkpoint_state_and_architecture(loaded: Any) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
    explicit: Optional[str] = None
    state: Optional[Mapping[str, Any]] = None
    if isinstance(loaded, dict):
        explicit_value = loaded.get("read_attention_architecture")
        metadata = loaded.get("metadata")
        if explicit_value is None and isinstance(metadata, Mapping):
            explicit_value = metadata.get("read_attention_architecture")
        args_metadata = loaded.get("args")
        if explicit_value is None and isinstance(args_metadata, Mapping):
            explicit_value = args_metadata.get("read_attention_architecture")
        if explicit_value is not None:
            explicit = str(explicit_value)
        for key in ("implant_state_dict", "state_dict", "model_state_dict"):
            candidate = loaded.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
        if state is None and any(torch.is_tensor(value) for value in loaded.values()):
            state = loaded
    elif isinstance(loaded, torch.nn.Module):
        state = loaded.state_dict()
        explicit_value = getattr(loaded, "read_attention_architecture", None)
        explicit = str(explicit_value) if explicit_value is not None else None

    if explicit is not None and explicit not in {"softmax", "sigmoid_mass", "sigmoid_all"}:
        raise ValueError(f"Unknown checkpoint read-attention architecture: {explicit!r}")
    if state is None:
        return None, explicit
    reader_keys = [str(key) for key in state if "read_bridge." in str(key)]
    inferred = None
    if reader_keys:
        inferred = (
            "sigmoid_all"
            if any(str(key).endswith("sigmoid_patch_head_bias") for key in reader_keys)
            else (
                "sigmoid_mass"
                if any(str(key).endswith("sigmoid_head_bias") for key in reader_keys)
                else "softmax"
            )
        )
    if explicit is not None and inferred is not None and explicit != inferred:
        raise ValueError(
            "Checkpoint read-attention metadata disagrees with its parameters: "
            f"metadata={explicit!r}, inferred={inferred!r}"
        )
    return state, explicit or inferred


def load_optional_implant(model: torch.nn.Module, path: Optional[Path]) -> None:
    if path is None:
        return
    loaded = torch_load_trusted(path)
    state, checkpoint_architecture = _checkpoint_state_and_architecture(loaded)
    model_architecture = str(getattr(model, "read_attention_architecture", "softmax"))
    if checkpoint_architecture is not None and checkpoint_architecture != model_architecture:
        raise ValueError(
            "Cannot attach a checkpoint built for a different read-attention architecture: "
            f"model={model_architecture!r}, checkpoint={checkpoint_architecture!r}, path={path}"
        )
    if isinstance(loaded, dict) and "implant_state_dict" in loaded:
        implant = migrate_implant_state(model, loaded["implant_state_dict"])
        missing, unexpected = model.read_implant.load_state_dict(implant, strict=False)
        if "hard_text_embedding" in loaded:
            model.set_hard_text_token_embedding(loaded["hard_text_embedding"])
        if "null_text_embedding" in loaded:
            model.set_null_text_token_embedding(loaded["null_text_embedding"])
        saved_read_null = loaded.get("read_null_token")
        if (
            isinstance(saved_read_null, torch.Tensor)
            and bool(getattr(model, "read_null_enabled", False))
            and isinstance(getattr(model.visual, "read_null_token", None), torch.Tensor)
        ):
            model.visual.read_null_token.data.copy_(
                saved_read_null.to(
                    device=model.visual.read_null_token.device,
                    dtype=model.visual.read_null_token.dtype,
                )
            )
        print(
            f"[load implant] architecture={model_architecture} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        return
    if state is None:
        raise RuntimeError(f"Unsupported implant checkpoint: {path}")
    state = dict(state)
    state.pop("read_implant.readability_log_weight", None)
    state.pop("read_implant.read_calibration_bias", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"[load implant] architecture={model_architecture} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def reset_router_components(model: torch.nn.Module) -> None:
    """Reset source-separated <any> trust while preserving forced experts."""
    model.read_implant.reset_trust_router(auto_read_scale=0.0)



def activate_read_null_token(model: torch.nn.Module, insert_block: int) -> None:
    """Enable the canonical zero-init READ_NULL token at a phase boundary.

    This exists so a single fresh-PIECES run can reproduce the historical
    schedule exactly: phases 1A/1B have no extra visual token, then READ_NULL is
    introduced immediately before 1B.5.  It must be called before constructing
    that phase's optimizer.
    """
    insert_block = int(insert_block)
    if bool(getattr(model, "read_null_enabled", False)):
        if int(getattr(model, "read_null_insert_block", insert_block)) != insert_block:
            raise RuntimeError("READ_NULL is already enabled at a different insertion block")
        return
    visual = model.visual
    if not hasattr(visual, "transformer"):
        raise RuntimeError("READ_NULL dynamic activation requires a ViT visual tower")
    layers = len(visual.transformer.resblocks)
    if not (0 <= insert_block < layers):
        raise ValueError(f"read_null_insert_block={insert_block} outside ViT layers={layers}")
    current = getattr(visual, "read_null_token", None)
    if current is not None:
        raise RuntimeError("visual.read_null_token exists while read_null_enabled is false")
    width = int(visual.class_embedding.numel())
    token = torch.nn.Parameter(torch.zeros(
        width, device=visual.class_embedding.device, dtype=visual.class_embedding.dtype
    ))
    visual.read_null_token = token
    visual.read_null_enabled = True
    visual.read_null_insert_block = insert_block
    if "read_null_insert_block_config" in getattr(visual, "_buffers", {}):
        visual.read_null_insert_block_config.fill_(insert_block)
    else:
        visual.register_buffer(
            "read_null_insert_block_config",
            torch.tensor(insert_block, dtype=torch.long, device=visual.class_embedding.device),
            persistent=True,
        )
    model.read_null_enabled = True
    model.read_null_insert_block = insert_block
    model.read_implant.read_null_enabled = True
    print(
        f"[READ_NULL] activated at phase boundary: insert_before_B{insert_block} "
        f"initial_param_norm={float(token.detach().float().norm()):.6f}"
    )

def compact_checkpoint(
    model: torch.nn.Module,
    phase: str,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "format": "gmp_anytext_early_branch_v3",
        "auto_fusion_semantics": "content_plus_source_x_trust_x_positive_read_minus_null",
        "phase": phase,
        "epoch": epoch,
        "global_step": global_step,
        "base_model_path": str(args.model_path),
        "read_attention_architecture": str(model.read_attention_architecture),
        "hard_text_embedding": model.hard_text_embedding.detach().cpu(),
        "null_text_embedding": model.null_text_embedding.detach().cpu(),
        "read_null_token": (
            model.visual.read_null_token.detach().cpu()
            if bool(getattr(model, "read_null_enabled", False))
            and isinstance(getattr(model.visual, "read_null_token", None), torch.Tensor)
            else None
        ),
        "read_null_insert_block": int(getattr(model, "read_null_insert_block", 20)),
        "implant_state_dict": {
            key: value.detach().cpu() for key, value in model.read_implant.state_dict().items()
        },
        "metrics": dict(metrics),
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }


def export_merged(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "metadata": {
                "read_attention_architecture": str(model.read_attention_architecture),
                "read_null_enabled": bool(getattr(model, "read_null_enabled", False)),
                "read_null_insert_block": int(getattr(model, "read_null_insert_block", 20)),
                "format": "gmp_anytext_merged_v5_read_null",
            },
        },
        path,
    )


# -----------------------------------------------------------------------------
# Training and validation
# -----------------------------------------------------------------------------


def batch_to_device(batch: PackedBatch, device: torch.device) -> PackedBatch:
    return PackedBatch(
        images=batch.images.to(device, non_blocking=True),
        patch_masks=batch.patch_masks.to(device, non_blocking=True),
        mask_weights=batch.mask_weights.to(device, non_blocking=True),
        present_targets=batch.present_targets.to(device, non_blocking=True),
        readable_targets=batch.readable_targets.to(device, non_blocking=True),
        captions=batch.captions,
        positive=batch.positive.to(device, non_blocking=True),
        source_triplets=batch.source_triplets,
        auto_triplets=batch.auto_triplets,
        invariance_pairs=batch.invariance_pairs,
        packet_metadata=batch.packet_metadata,
    )


def _collect_final_precision_tensors(
    model: torch.nn.Module,
    batch: PackedBatch,
    details: Mapping[str, Any],
    destination: MutableMapping[str, torch.Tensor],
) -> None:
    """Collect critical final-model values without changing the training graph."""
    destination["final.batch.images"] = batch.images
    for key in (
        "logits_per_image", "raw_read_logits", "read_logits", "null_read_logits",
        "relative_read_logits", "early_orthographic_logits", "trust_gate",
        "source_gate", "route_gate", "auto_read_contribution", "source_logits",
        "source_stats", "glyph_logits", "glyph_probs", "base_image_embedding",
        "content_image_embedding", "content_correction", "read_feature",
        "read_text_embedding", "patch_token_norms",
    ):
        value = details.get(key)
        if isinstance(value, torch.Tensor):
            destination[f"final.{key}"] = value
    for detail_name in ("read_details", "orthographic_details", "content_details", "source_details"):
        detail = details.get(detail_name)
        if isinstance(detail, Mapping):
            for key, value in detail.items():
                if isinstance(value, torch.Tensor):
                    destination[f"final.{detail_name}.{key}"] = value
    states = details.get("visual_states")
    if isinstance(states, Mapping):
        for block in model.read_implant.capture_block_list():
            state = states.get(int(block))
            if isinstance(state, torch.Tensor):
                destination[f"final.visual_B{int(block):02d}.cls"] = state[:, 0]
                spatial = state[:, 1:-1] if bool(getattr(model, "read_null_enabled", False)) and int(block) >= int(getattr(model, "read_null_insert_block", 20)) else state[:, 1:]
                destination[f"final.visual_B{int(block):02d}.patch_norms"] = (
                    spatial.float().norm(dim=-1)
                )


def compute_batch_loss(
    model: torch.nn.Module,
    clip_module: Any,
    batch: PackedBatch,
    device: torch.device,
    args: argparse.Namespace,
    phase: str,
    diagnostic_tensors: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weights = phase_loss_weights(args, phase)

    if phase == "1a":
        image_info = model.encode_image_states(batch.images, return_final_tokens=False)
        source_logits, glyph_logits, source_stats = model.read_implant.source_outputs(
            image_info["states"]
        )
        loss_mask = weighted_mask_loss(glyph_logits, batch.patch_masks, batch.mask_weights)
        loss_presence = (
            known_bce(source_logits[:, 0], batch.present_targets)
            + known_bce(source_logits[:, 1], batch.readable_targets)
        )
        total = weights["mask"] * loss_mask + weights["presence"] * loss_presence
        if diagnostic_tensors is not None:
            diagnostic_tensors.update({
                "final.batch.images": batch.images,
                "final.phase1a.source_logits": source_logits,
                "final.phase1a.glyph_logits": glyph_logits,
                "final.phase1a.source_stats": source_stats,
                "final.loss.total": total.reshape(1),
                "final.loss.mask": loss_mask.reshape(1),
                "final.loss.presence": loss_presence.reshape(1),
            })
            for block in model.read_implant.source_block_list():
                state = image_info["states"][int(block)]
                diagnostic_tensors[f"final.visual_B{int(block):02d}.cls"] = state[:, 0]
                diagnostic_tensors[f"final.visual_B{int(block):02d}.patch_norms"] = state[:, 1:].float().norm(dim=-1)

        metrics = {
            "loss_total": float(total.detach()),
            "loss_mp": 0.0,
            "loss_source": 0.0,
            "loss_ortho": 0.0,
            "loss_mask": float(loss_mask.detach()),
            "loss_presence": float(loss_presence.detach()),
            "loss_invariance": 0.0,
            "loss_auto_negative": 0.0,
            "loss_rent": 0.0,
            "gate_mean": 0.0,
            "trust_mean": 0.0,
            "source_gate_mean": float((source_logits.sigmoid().prod(dim=-1)).detach().mean()),
            "glyph_mean": float(glyph_logits.detach().sigmoid().mean()),
            "glyph_topk": float(source_stats[:, 2].detach().mean()),
            "present_prob": float(source_logits[:, 0].detach().sigmoid().mean()),
            "readable_prob": float(source_logits[:, 1].detach().sigmoid().mean()),
            "auto_read_scale": float(model.read_implant.auto_read_scale.detach()),
            "glyph_bias_beta": float(model.read_implant.glyph_bias_beta.detach()),
            "null_abstain_weight": float(model.read_implant.null_abstain_weight.detach()),
        }
        return total, metrics

    tokens = clip_module.tokenize(batch.captions, truncate=True).to(device)
    details = model.forward_modes(batch.images, tokens, return_details=True)
    loss_mp = multi_positive_clip_loss(details["logits_per_image"], batch.positive)
    loss_source = source_ranking_loss(
        details["read_logits"], batch.source_triplets, args.source_margin
    )
    loss_ortho = source_ranking_loss(
        details["early_orthographic_logits"], batch.source_triplets, args.ortho_margin
    )
    loss_mask = weighted_mask_loss(
        details["glyph_logits"], batch.patch_masks, batch.mask_weights
    )
    loss_presence = (
        known_bce(details["source_logits"][:, 0], batch.present_targets)
        + known_bce(details["source_logits"][:, 1], batch.readable_targets)
    )
    loss_invariance = invariance_loss(
        details["content_image_embedding"], batch.invariance_pairs
    )
    loss_auto_negative = auto_hard_negative_loss(
        details["logits_per_image"], batch.auto_triplets, args.auto_negative_margin
    )
    loss_rent = route_rent(details["route_gate"], details["mode_ids"])
    read_null_active = bool(getattr(model, "read_null_enabled", False))
    if read_null_active:
        loss_read_null, read_null_metrics = read_null_attention_objective(
            details, batch.positive, model.null_text_token_id
        )
        read_null_metrics.update(read_null_token_metrics(model, details))
    else:
        loss_read_null = details["logits_per_image"].sum() * 0.0
        read_null_metrics = {}

    total = (
        weights["mp"] * loss_mp
        + weights["source"] * loss_source
        + weights["ortho"] * loss_ortho
        + weights["mask"] * loss_mask
        + weights["presence"] * loss_presence
        + weights["invariance"] * loss_invariance
        + weights["auto_negative"] * loss_auto_negative
        + weights["rent"] * loss_rent
        + weights["read_null"] * loss_read_null
    )
    if diagnostic_tensors is not None:
        _collect_final_precision_tensors(model, batch, details, diagnostic_tensors)
        diagnostic_tensors.update({
            "final.loss.total": total.reshape(1),
            "final.loss.multi_positive": loss_mp.reshape(1),
            "final.loss.source": loss_source.reshape(1),
            "final.loss.orthographic": loss_ortho.reshape(1),
            "final.loss.mask": loss_mask.reshape(1),
            "final.loss.presence": loss_presence.reshape(1),
            "final.loss.invariance": loss_invariance.reshape(1),
            "final.loss.auto_negative": loss_auto_negative.reshape(1),
            "final.loss.route_rent": loss_rent.reshape(1),
            "final.loss.read_null": loss_read_null.reshape(1),
        })

    metrics = {
        "loss_total": float(total.detach()),
        "loss_mp": float(loss_mp.detach()),
        "loss_source": float(loss_source.detach()),
        "loss_ortho": float(loss_ortho.detach()),
        "loss_mask": float(loss_mask.detach()),
        "loss_presence": float(loss_presence.detach()),
        "loss_invariance": float(loss_invariance.detach()),
        "loss_auto_negative": float(loss_auto_negative.detach()),
        "loss_rent": float(loss_rent.detach()),
        "loss_read_null": float(loss_read_null.detach()),
        "gate_mean": float(details["route_gate"].detach().mean()),
        "trust_mean": float(details["trust_gate"].detach().mean()),
        "source_gate_mean": float(details["source_gate"].detach().mean()),
        "glyph_mean": float(details["glyph_probs"].detach().mean()),
        "glyph_topk": float(details["source_stats"][:, 2].detach().mean()),
        "present_prob": float(details["source_logits"][:, 0].detach().sigmoid().mean()),
        "readable_prob": float(details["source_logits"][:, 1].detach().sigmoid().mean()),
        "null_read_mean": float(details["null_read_logits"].detach().mean()),
        "relative_read_mean": float(details["relative_read_logits"].detach().mean()),
        "early_ortho_mean": float(details["early_orthographic_logits"].detach().mean()),
        "auto_read_scale": float(model.read_implant.auto_read_scale.detach()),
        "glyph_bias_beta": float(model.read_implant.glyph_bias_beta.detach()),
        "null_abstain_weight": float(model.read_implant.null_abstain_weight.detach()),
        **read_null_metrics,
    }
    metrics.update(collect_pieces_sigmoid_attention_metrics(
        model, details, positive=batch.positive,
        present_targets=batch.present_targets, readable_targets=batch.readable_targets,
    ))
    return total, metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    clip_module: Any,
    mixer: SourceMixer,
    device: torch.device,
    args: argparse.Namespace,
    phase: str,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for _ in range(args.val_batches):
        batch = batch_to_device(mixer.sample_batch(args.logical_batch_images), device)
        with amp_context(device, args.amp_dtype):
            _, metrics = compute_batch_loss(model, clip_module, batch, device, args, phase)
        count += 1
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def preflight_mode_invariance(
    model: torch.nn.Module,
    clip_module: Any,
    batch: PackedBatch,
    device: torch.device,
    args: argparse.Namespace,
    enforce_zero_impact: bool = True,
) -> Dict[str, float]:
    model.eval()
    image = batch.images[: min(2, len(batch.images))].to(device)
    plain = clip_module.tokenize(["a photo of a cat", "dog"], truncate=True).to(device)
    notext = clip_module.tokenize(["<notext> a photo of a cat", "<notext> dog"], truncate=True).to(device)
    any_tokens = clip_module.tokenize(["<any> a photo of a cat", "dog"], truncate=True).to(device)
    read = clip_module.tokenize(["<text> cat", "<text> dog"], truncate=True).to(device)
    with amp_context(device, args.amp_dtype):
        legacy_plain = model.forward_hard(image, plain)[0]
        mode_notext = model.forward_modes(image, notext)[0]
        mode_any = model.forward_modes(image, any_tokens)[0]
        legacy_read = model.forward_hard(image, read)[0]
        mode_read = model.forward_modes(image, read)[0]
    result = {
        "notext_vs_legacy_max_abs": float((mode_notext - legacy_plain).abs().max()),
        "any_vs_legacy_max_abs": float((mode_any - legacy_plain).abs().max()),
        "text_vs_legacy_max_abs": float((mode_read - legacy_read).abs().max()),
    }
    nonfinite = {
        key: value for key, value in result.items()
        if not math.isfinite(float(value))
    }
    if nonfinite:
        raise FloatingPointError(
            "Mode migration preflight produced non-finite outputs before training: "
            f"{nonfinite}. This usually means a stale sigmoid FP16 implementation or "
            "a non-finite checkpoint."
        )
    tolerance = args.preflight_tolerance
    checked = (
        result.values()
        if enforce_zero_impact
        else [result["notext_vs_legacy_max_abs"]]
    )
    if any(value > tolerance for value in checked):
        raise RuntimeError(f"Mode migration preflight failed: {result}, tolerance={tolerance}")
    return result


def phase_mix(args: argparse.Namespace, phase: str) -> Dict[str, float]:
    if phase == "1a":
        return {
            "coco": args.mix_1a_coco,
            "textcaps": args.mix_1a_textcaps,
            "imagenet": args.mix_1a_imagenet,
        }
    if phase == "1b":
        return {
            "coco": args.mix_1b_coco,
            "textcaps": args.mix_1b_textcaps,
            "imagenet": args.mix_1b_imagenet,
        }
    if phase == "1b5":
        return {
            "coco": args.mix_1b5_coco,
            "textcaps": args.mix_1b5_textcaps,
            "imagenet": args.mix_1b5_imagenet,
            "imagenet_math": args.mix_1b5_imagenet_math if args.math_curriculum_enabled else 0.0,
        }
    return {
        "coco": args.mix_1c_coco,
        "textcaps": args.mix_1c_textcaps,
        "imagenet": args.mix_1c_imagenet,
        "imagenet_math": args.mix_1c_imagenet_math if args.math_curriculum_enabled else 0.0,
    }


def benchmark_quality(metrics: Mapping[str, float]) -> float:
    attacked = [
        float(value)
        for key, value in metrics.items()
        if (key.startswith("scam/") or key.startswith("rta/"))
        and key.endswith("/any_acc")
        and not any(clean in key for clean in ("NoSCAM", "NoRTA"))
    ]
    terms = attacked[:]
    if "mvt/any_acc" in metrics:
        terms.append(float(metrics["mvt/any_acc"]))
    return sum(terms) / len(terms) if terms else float("nan")



def _set_trainable_prefixes(model: torch.nn.Module, prefixes: Sequence[str]) -> List[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names: List[str] = []
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(True)
            names.append(name)
    return names


def _mini_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Mini benchmark selected no trainable parameters")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _mini_backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> None:
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], grad_clip
    )
    scaler.step(optimizer)
    scaler.update()


def _ranking_margin(
    logits: torch.Tensor,
    triplets: Sequence[Tuple[int, int, List[int]]],
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for candidate, positive_image, negatives in triplets:
        for negative_image in negatives:
            values.append(logits[positive_image, candidate] - logits[negative_image, candidate])
    return torch.stack(values).mean() if values else logits.sum() * 0.0


def _ranking_pair_accuracy(
    logits: torch.Tensor,
    triplets: Sequence[Tuple[int, int, List[int]]],
) -> torch.Tensor:
    """Scale-free positive-vs-negative ordering accuracy for source triplets."""
    values: List[torch.Tensor] = []
    for candidate, positive_image, negatives in triplets:
        positive_score = logits[positive_image, candidate]
        for negative_image in negatives:
            delta = positive_score - logits[negative_image, candidate]
            # Half credit for exact ties keeps this well-defined at initialization.
            values.append((delta > 0).to(logits.dtype) + 0.5 * (delta == 0).to(logits.dtype))
    return torch.stack(values).mean() if values else logits.sum() * 0.0


def _balanced_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    valid = targets >= 0
    if not bool(valid.any()):
        return float("nan")
    pred = logits[valid] >= 0
    truth = targets[valid] >= 0.5
    positives = truth
    negatives = ~truth
    tpr = (pred[positives] == truth[positives]).float().mean() if bool(positives.any()) else torch.tensor(1.0)
    tnr = (pred[negatives] == truth[negatives]).float().mean() if bool(negatives.any()) else torch.tensor(1.0)
    return float(0.5 * (tpr + tnr))


def _copy_module_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _restore_module_state(module: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    module.load_state_dict(dict(state), strict=True)


def _all_nonempty_combinations(values: Sequence[int], max_size: int = 0) -> List[Tuple[int, ...]]:
    import itertools
    unique = tuple(dict.fromkeys(int(x) for x in values))
    limit = len(unique) if max_size <= 0 else min(len(unique), int(max_size))
    return [combo for size in range(1, limit + 1) for combo in itertools.combinations(unique, size)]


def _parse_block_combo_specs(specs: Sequence[str]) -> List[Tuple[int, ...]]:
    """Parse CLI forms such as 12, 8+12, or 8,12,13 into unique tuples."""
    output: List[Tuple[int, ...]] = []
    for raw in specs:
        parts = [part for part in re.split(r"[+,]", str(raw).strip()) if part]
        if not parts:
            continue
        combo = tuple(dict.fromkeys(int(part) for part in parts))
        if combo and combo not in output:
            output.append(combo)
    if not output:
        raise ValueError("At least one mini source topology is required")
    return output


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=0))


def _aggregate_seed_rows(seed_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["stage"]), str(row["blocks"])), []).append(row)

    aggregates: List[Dict[str, Any]] = []
    for (stage, blocks), group in grouped.items():
        aggregate: Dict[str, Any] = {
            "stage": stage,
            "blocks": blocks,
            "num_seeds": len(group),
            "seeds": ",".join(str(int(row["seed"])) for row in group),
        }
        numeric_keys: List[str] = []
        for row in group:
            for key, value in row.items():
                if key in {"stage", "blocks", "seed"}:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) and key not in numeric_keys:
                    numeric_keys.append(key)
        for key in numeric_keys:
            values = [float(row[key]) for row in group if key in row and math.isfinite(float(row[key]))]
            mean, std = _mean_std(values)
            aggregate[key] = mean
            aggregate[f"{key}_std"] = std
        aggregates.append(aggregate)
    return aggregates


def _select_aggregate_winner(rows: Sequence[Mapping[str, Any]], stage: str) -> Dict[str, Any]:
    candidates = [dict(row) for row in rows if row.get("stage") == stage]
    if not candidates:
        raise RuntimeError(f"Mini benchmark produced no aggregate rows for stage {stage}")
    # Mean score wins; lower run-to-run variance breaks an exact tie.
    return max(candidates, key=lambda row: (float(row["score"]), -float(row.get("score_std", 0.0))))


def _representative_state(
    records: Sequence[Mapping[str, Any]],
    target_score: float,
) -> Mapping[str, Any]:
    if not records:
        raise RuntimeError("No mini-benchmark states were recorded")
    return min(
        records,
        key=lambda record: (abs(float(record["score"]) - float(target_score)), int(record["seed"])),
    )["state"]


def _tap_weight_metrics(prefix: str, blocks: Sequence[int], logits: torch.Tensor) -> Dict[str, float]:
    weights = logits.detach().float().softmax(dim=0).cpu().tolist()
    return {f"{prefix}_weight_b{int(block)}": float(weight) for block, weight in zip(blocks, weights)}


def _mini_eval_ortho(
    model: torch.nn.Module,
    clip_module: Any,
    mixer: SourceMixer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    margins: List[float] = []
    pair_accs: List[float] = []
    with torch.no_grad():
        for _ in range(args.mini_val_batches):
            batch = batch_to_device(mixer.sample_batch(args.mini_batch_images), device)
            tokens = clip_module.tokenize(batch.captions, truncate=True).to(device)
            with amp_context(device, args.amp_dtype):
                details = model.forward_modes(batch.images, tokens, return_details=True)
                logits = details["early_orthographic_logits"]
                loss = source_ranking_loss(logits, batch.source_triplets, args.ortho_margin)
                margin = _ranking_margin(logits, batch.source_triplets)
                pair_acc = _ranking_pair_accuracy(logits, batch.source_triplets)
            losses.append(float(loss))
            margins.append(float(margin))
            pair_accs.append(float(pair_acc))
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "margin": sum(margins) / max(1, len(margins)),
        "pair_acc": sum(pair_accs) / max(1, len(pair_accs)),
    }


def _mini_eval_source(
    model: torch.nn.Module,
    mixer: SourceMixer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {
        "loss": 0.0,
        "source_bal_acc": 0.0,
        "ordered_bal_acc": 0.0,
        "glyph_mean": 0.0,
    }
    with torch.no_grad():
        for _ in range(args.mini_val_batches):
            batch = batch_to_device(mixer.sample_batch(args.mini_batch_images), device)
            with amp_context(device, args.amp_dtype):
                info = model.encode_image_states(batch.images, return_final_tokens=False)
                source_logits, glyph_logits, _stats = model.read_implant.source_outputs(info["states"])
                mask = weighted_mask_loss(glyph_logits, batch.patch_masks, batch.mask_weights)
                bce = known_bce(source_logits[:, 0], batch.present_targets) + known_bce(
                    source_logits[:, 1], batch.readable_targets
                )
                loss = args.mask_weight * mask + args.presence_weight * bce
            totals["loss"] += float(loss)
            totals["source_bal_acc"] += _balanced_accuracy(source_logits[:, 0], batch.present_targets)
            totals["ordered_bal_acc"] += _balanced_accuracy(source_logits[:, 1], batch.readable_targets)
            totals["glyph_mean"] += float(glyph_logits.sigmoid().mean())
    return {key: value / max(1, args.mini_val_batches) for key, value in totals.items()}


def _mini_eval_late(
    model: torch.nn.Module,
    clip_module: Any,
    mixer: SourceMixer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "source_loss": 0.0,
        "source_margin": 0.0,
        "source_pair_acc": 0.0,
        "clean_null_acc": 0.0,
        "readable_reject_acc": 0.0,
    }
    with torch.no_grad():
        for _ in range(args.mini_val_batches):
            batch = batch_to_device(mixer.sample_batch(args.mini_batch_images), device)
            tokens = clip_module.tokenize(batch.captions, truncate=True).to(device)
            with amp_context(device, args.amp_dtype):
                details = model.forward_modes(batch.images, tokens, return_details=True)
                read_logits = details["read_logits"]
                loss = source_ranking_loss(read_logits, batch.source_triplets, args.source_margin)
                margin = _ranking_margin(read_logits, batch.source_triplets)
                pair_acc = _ranking_pair_accuracy(read_logits, batch.source_triplets)
            null = details["null_read_logits"]
            nonnull = details["read_mode_mask"] | details["any_mode_mask"]
            nonnull = nonnull & ~model.is_null_candidate(details["read_tokens"])
            if bool(nonnull.any()):
                best_word = read_logits[:, nonnull].amax(dim=1)
            else:
                best_word = torch.full_like(null, -1.0e9)
            clean = batch.present_targets <= 0.0
            readable = batch.readable_targets >= 1.0
            clean_acc = float((null[clean] > best_word[clean]).float().mean()) if bool(clean.any()) else 1.0
            read_acc = float((best_word[readable] > null[readable]).float().mean()) if bool(readable.any()) else 1.0
            totals["source_loss"] += float(loss)
            totals["source_margin"] += float(margin)
            totals["source_pair_acc"] += float(pair_acc)
            totals["clean_null_acc"] += clean_acc
            totals["readable_reject_acc"] += read_acc
    return {key: value / max(1, args.mini_val_batches) for key, value in totals.items()}


def _late_selection_score(metrics: Mapping[str, float]) -> float:
    """Goal-aligned, largely scale-free selector for the late lexical branch."""
    return (
        0.5 * float(metrics["source_pair_acc"])
        + 1.0 * float(metrics["clean_null_acc"])
        + 0.5 * float(metrics["readable_reject_acc"])
        - 0.1 * float(metrics["source_loss"])
    )


def _clamp_mini_late_scalars(model: torch.nn.Module, args: argparse.Namespace) -> None:
    with torch.no_grad():
        model.read_implant.glyph_bias_beta.clamp_(0.0, args.glyph_beta_max)
        model.read_implant.null_abstain_weight.clamp_(0.0, args.null_abstain_max)


def _write_mini_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_mini_rows(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[mini] matplotlib unavailable; skipping plots: {exc}")
        return
    for stage in ("ortho", "source", "late"):
        selected = [row for row in rows if row.get("stage") == stage]
        if not selected:
            continue
        labels = [str(row["blocks"]) for row in selected]
        scores = [float(row["score"]) for row in selected]
        errors = [float(row.get("score_std", 0.0)) for row in selected]
        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(labels)), 5))
        ax.bar(range(len(labels)), scores, yerr=errors, capsize=4)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_ylabel("mean selection score")
        ax.set_title(f"AnyText multi-seed mini benchmark — {stage}")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"mini_{stage}_scores.png", dpi=180)
        plt.close(fig)


def _write_mini_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> None:
    """Write a compact human-readable companion to the CSV and plots."""
    lines = [
        "# AnyText multi-seed mini topology benchmark",
        "",
        "## Selected topology",
        "",
        f"- Orthographic branch: `{'+'.join(map(str, selected['ortho_blocks']))}` (fixed)",
        f"- Source gate: `{'+'.join(map(str, selected['source_blocks']))}`",
        f"- Late readout: `{'+'.join(map(str, selected['late_blocks']))}`",
        f"- Steps per configuration/seed: `{int(selected['mini_steps_per_config'])}`",
        f"- Seeds: `{', '.join(map(str, selected['mini_seeds']))}`",
        "- Late selector: `0.5*pair_acc + 1.0*clean_null + 0.5*readable_reject - 0.1*source_loss`",
        "- `read_calibration_scale` was frozen during late topology comparison.",
        "",
    ]
    stage_columns = {
        "ortho": (
            "score", "score_std", "loss", "loss_std", "margin", "margin_std",
            "pair_acc", "pair_acc_std",
        ),
        "source": (
            "score", "score_std", "loss", "loss_std", "source_bal_acc",
            "source_bal_acc_std", "ordered_bal_acc", "ordered_bal_acc_std",
        ),
        "late": (
            "score", "score_std", "source_pair_acc", "source_pair_acc_std",
            "clean_null_acc", "clean_null_acc_std", "readable_reject_acc",
            "readable_reject_acc_std", "source_loss", "source_loss_std",
            "source_margin", "source_margin_std", "read_calibration_scale",
            "null_abstain_weight", "glyph_bias_beta",
        ),
    }
    for stage in ("ortho", "source", "late"):
        stage_rows = sorted(
            (dict(row) for row in rows if row.get("stage") == stage),
            key=lambda row: float(row.get("score", -float("inf"))),
            reverse=True,
        )
        columns = stage_columns[stage]
        lines.extend([
            f"## {stage.capitalize()} candidates",
            "",
            "| blocks | " + " | ".join(columns) + " |",
            "|---|" + "|".join("---:" for _ in columns) + "|",
        ])
        for row in stage_rows:
            values = []
            for column in columns:
                value = row.get(column, float("nan"))
                values.append(f"{float(value):.6f}")
            lines.append(f"| `{row['blocks']}` | " + " | ".join(values) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_mini_benchmark(
    model: torch.nn.Module,
    clip_module: Any,
    sources_train: Mapping[str, Any],
    sources_val: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Multi-seed topology search: fixed orthography -> source gate -> late read taps."""
    out_dir = args.out_dir / "mini_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = args.mini_steps_per_config
    if steps <= 0:
        steps = max(16, round(args.steps_per_epoch * args.mini_fraction))
    if args.mini_num_seeds <= 0:
        raise ValueError("--mini_num_seeds must be positive")
    mini_seeds = [int(args.seed + index * args.mini_seed_stride) for index in range(args.mini_num_seeds)]
    fixed_ortho = tuple(dict.fromkeys(int(x) for x in args.mini_fixed_ortho_blocks))
    if not fixed_ortho:
        raise ValueError("--mini_fixed_ortho_blocks must be non-empty")
    source_combos = _parse_block_combo_specs(args.mini_source_combos)
    late_combos = [
        (int(args.mini_late_anchor_block), int(second))
        for second in args.mini_late_second_blocks
        if int(second) != int(args.mini_late_anchor_block)
    ]
    late_combos = list(dict.fromkeys(late_combos))
    if not late_combos:
        raise ValueError("Late mini benchmark requires at least one second tap")

    print(
        f"[mini] steps/config/seed={steps} val_batches={args.mini_val_batches} "
        f"seeds={mini_seeds}"
    )
    print(f"[mini] fixed ortho={fixed_ortho}")
    print(f"[mini] source candidates={source_combos}")
    print(f"[mini] late candidates={late_combos}")

    seed_rows: List[Dict[str, Any]] = []
    state_records: Dict[Tuple[str, Tuple[int, ...]], List[Dict[str, Any]]] = {}

    # 1) Train the already-selected B8+B12+B13 orthographic topology once per seed.
    for run_seed in mini_seeds:
        set_seed(run_seed + 1000)
        train_mixer = SourceMixer({"imagenet": sources_train["imagenet"]}, {"imagenet": 1.0}, run_seed + 301)
        val_mixer = SourceMixer({"imagenet": sources_val["imagenet"]}, {"imagenet": 1.0}, run_seed + 401)
        model.read_implant.set_ortho_tap_blocks(fixed_ortho, reset_uniform=True)
        model.read_implant.reset_orthographic_branch()
        _set_trainable_prefixes(model, ("read_implant.orthographic_bridge.", "read_implant.ortho_tap_logits"))
        optimizer = _mini_optimizer(model, args.lr_read, args.weight_decay)
        scaler = precision_make_grad_scaler(device, args.amp_dtype)
        model.train()
        for _step in range(steps):
            batch = batch_to_device(train_mixer.sample_batch(args.mini_batch_images), device)
            tokens = clip_module.tokenize(batch.captions, truncate=True).to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp_dtype):
                details = model.forward_modes(batch.images, tokens, return_details=True)
                loss = source_ranking_loss(
                    details["early_orthographic_logits"], batch.source_triplets, args.ortho_margin
                )
            _mini_backward_step(loss, optimizer, model, scaler, args.grad_clip)
        metrics = _mini_eval_ortho(model, clip_module, val_mixer, device, args)
        score = -metrics["loss"] + 0.05 * metrics["margin"]
        row = {
            "stage": "ortho", "blocks": "+".join(map(str, fixed_ortho)),
            "seed": run_seed, "score": score, **metrics,
            **_tap_weight_metrics("ortho", fixed_ortho, model.read_implant.ortho_tap_logits),
        }
        seed_rows.append(row)
        state_records.setdefault(("ortho", fixed_ortho), []).append({
            "seed": run_seed,
            "score": score,
            "state": {
                "module": _copy_module_state(model.read_implant.orthographic_bridge),
                "tap": model.read_implant.ortho_tap_logits.detach().cpu().clone(),
            },
        })
        print(
            f"[mini ortho seed={run_seed}] {fixed_ortho} score={score:.5f} "
            f"loss={metrics['loss']:.5f} margin={metrics['margin']:.4f} pair={metrics['pair_acc']:.3f}"
        )

    aggregate_rows = _aggregate_seed_rows(seed_rows)
    ortho_winner = _select_aggregate_winner(aggregate_rows, "ortho")
    best_ortho_score = float(ortho_winner["score"])
    best_ortho_state = _representative_state(
        state_records[("ortho", fixed_ortho)], best_ortho_score
    )
    model.read_implant.set_ortho_tap_blocks(fixed_ortho, reset_uniform=True)
    _restore_module_state(model.read_implant.orthographic_bridge, best_ortho_state["module"])
    model.read_implant.ortho_tap_logits.data.copy_(best_ortho_state["tap"].to(device))
    print(
        f"[mini ortho mean] {fixed_ortho} score={best_ortho_score:.5f}"
        f"±{float(ortho_winner.get('score_std', 0.0)):.5f}"
    )

    # 2) Source gate: preserve the original early candidates and add B18/B19 daredevils.
    source_seed_rows: List[Dict[str, Any]] = []
    for combo in source_combos:
        for run_seed in mini_seeds:
            set_seed(run_seed + 2000)
            train_mixer = SourceMixer({"imagenet": sources_train["imagenet"]}, {"imagenet": 1.0}, run_seed + 302)
            val_mixer = SourceMixer({"imagenet": sources_val["imagenet"]}, {"imagenet": 1.0}, run_seed + 402)
            model.read_implant.set_source_tap_blocks(combo, reset_uniform=True)
            model.read_implant.reset_source_gate()
            _set_trainable_prefixes(model, ("read_implant.source_head.", "read_implant.source_tap_logits"))
            optimizer = _mini_optimizer(model, args.lr_new, args.weight_decay)
            scaler = precision_make_grad_scaler(device, args.amp_dtype)
            model.train()
            for _step in range(steps):
                batch = batch_to_device(train_mixer.sample_batch(args.mini_batch_images), device)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, args.amp_dtype):
                    info = model.encode_image_states(batch.images, return_final_tokens=False)
                    source_logits, glyph_logits, _stats = model.read_implant.source_outputs(info["states"])
                    mask = weighted_mask_loss(glyph_logits, batch.patch_masks, batch.mask_weights)
                    bce = known_bce(source_logits[:, 0], batch.present_targets) + known_bce(
                        source_logits[:, 1], batch.readable_targets
                    )
                    loss = args.mask_weight * mask + args.presence_weight * bce
                _mini_backward_step(loss, optimizer, model, scaler, args.grad_clip)
            metrics = _mini_eval_source(model, val_mixer, device, args)
            score = 0.5 * (metrics["source_bal_acc"] + metrics["ordered_bal_acc"]) - 0.05 * metrics["loss"]
            row = {
                "stage": "source", "blocks": "+".join(map(str, combo)),
                "seed": run_seed, "score": score, **metrics,
                **_tap_weight_metrics("source", combo, model.read_implant.source_tap_logits),
            }
            seed_rows.append(row)
            source_seed_rows.append(row)
            state_records.setdefault(("source", combo), []).append({
                "seed": run_seed,
                "score": score,
                "state": {
                    "module": _copy_module_state(model.read_implant.source_head),
                    "tap": model.read_implant.source_tap_logits.detach().cpu().clone(),
                },
            })
            print(
                f"[mini source seed={run_seed}] {combo} score={score:.5f} "
                f"source={metrics['source_bal_acc']:.3f} ordered={metrics['ordered_bal_acc']:.3f}"
            )

    aggregate_rows = _aggregate_seed_rows(seed_rows)
    source_winner = _select_aggregate_winner(aggregate_rows, "source")
    best_source = tuple(int(x) for x in str(source_winner["blocks"]).split("+"))
    best_source_score = float(source_winner["score"])
    best_source_state = _representative_state(
        state_records[("source", best_source)], best_source_score
    )
    model.read_implant.set_source_tap_blocks(best_source, reset_uniform=True)
    _restore_module_state(model.read_implant.source_head, best_source_state["module"])
    model.read_implant.source_tap_logits.data.copy_(best_source_state["tap"].to(device))
    for row in sorted(
        (row for row in aggregate_rows if row.get("stage") == "source"),
        key=lambda item: float(item["score"]), reverse=True,
    ):
        print(
            f"[mini source mean] ({row['blocks']}) score={float(row['score']):.5f}"
            f"±{float(row.get('score_std', 0.0)):.5f} "
            f"source={float(row['source_bal_acc']):.3f} ordered={float(row['ordered_bal_acc']):.3f}"
        )

    # 3) Late readout: B20 is mandatory; compare B18/B19/B21/B22 as second taps.
    baseline_read = _copy_module_state(model.read_implant.read_bridge)
    baseline_hard = model.hard_text_embedding.detach().cpu().clone()
    baseline_null = model.null_text_embedding.detach().cpu().clone()
    baseline_scalars = {
        "read_calibration_scale": model.read_implant.read_calibration_scale.detach().cpu().clone(),
        "null_abstain_weight": model.read_implant.null_abstain_weight.detach().cpu().clone(),
        "glyph_bias_beta": model.read_implant.glyph_bias_beta.detach().cpu().clone(),
    }

    for combo in late_combos:
        for run_seed in mini_seeds:
            set_seed(run_seed + 3000)
            train_mixer = SourceMixer({"imagenet": sources_train["imagenet"]}, {"imagenet": 1.0}, run_seed + 303)
            val_mixer = SourceMixer({"imagenet": sources_val["imagenet"]}, {"imagenet": 1.0}, run_seed + 403)
            model.read_implant.set_tap_blocks(combo, reset_uniform=True)
            if not args.mini_late_allow_content_mixture:
                # The proven content correction is already ~B20-only. Keep it fixed at
                # the anchor so B18/B19 cannot win or lose by perturbing the content lane.
                with torch.no_grad():
                    model.read_implant.content_tap_logits.fill_(-20.0)
                    model.read_implant.content_tap_logits[0] = 20.0
            _restore_module_state(model.read_implant.read_bridge, baseline_read)
            model.set_hard_text_token_embedding(baseline_hard)
            model.set_null_text_token_embedding(baseline_null)
            for name, value in baseline_scalars.items():
                getattr(model.read_implant, name).data.copy_(value.to(device))
            # Deliberately freeze read_calibration_scale: topology must win by ordering
            # and abstention, not by making all read logits louder.
            _set_trainable_prefixes(model, (
                "hard_text_embedding", "null_text_embedding",
                "read_implant.read_bridge.", "read_implant.read_tap_logits",
                "read_implant.null_abstain_weight", "read_implant.glyph_bias_beta",
            ))
            optimizer = _mini_optimizer(model, args.lr_read, args.weight_decay)
            scaler = precision_make_grad_scaler(device, args.amp_dtype)
            model.train()
            for _step in range(steps):
                batch = batch_to_device(train_mixer.sample_batch(args.mini_batch_images), device)
                tokens = clip_module.tokenize(batch.captions, truncate=True).to(device)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, args.amp_dtype):
                    details = model.forward_modes(batch.images, tokens, return_details=True)
                    source_loss = source_ranking_loss(
                        details["read_logits"], batch.source_triplets, args.source_margin
                    )
                    mp = multi_positive_clip_loss(details["logits_per_image"], batch.positive)
                    loss = source_loss + 0.25 * mp
                _mini_backward_step(loss, optimizer, model, scaler, args.grad_clip)
                _clamp_mini_late_scalars(model, args)
            metrics = _mini_eval_late(model, clip_module, val_mixer, device, args)
            score = _late_selection_score(metrics)
            scalar_metrics = {
                "read_calibration_scale": float(model.read_implant.read_calibration_scale.detach()),
                "null_abstain_weight": float(model.read_implant.null_abstain_weight.detach()),
                "glyph_bias_beta": float(model.read_implant.glyph_bias_beta.detach()),
            }
            row = {
                "stage": "late", "blocks": "+".join(map(str, combo)),
                "seed": run_seed, "score": score, **metrics, **scalar_metrics,
                **_tap_weight_metrics("read", combo, model.read_implant.read_tap_logits),
                **_tap_weight_metrics("content", combo, model.read_implant.content_tap_logits),
            }
            seed_rows.append(row)
            state_records.setdefault(("late", combo), []).append({
                "seed": run_seed,
                "score": score,
                "state": {
                    "read_bridge": _copy_module_state(model.read_implant.read_bridge),
                    "tap": model.read_implant.read_tap_logits.detach().cpu().clone(),
                    "content_tap": model.read_implant.content_tap_logits.detach().cpu().clone(),
                    "hard_text_embedding": model.hard_text_embedding.detach().cpu().clone(),
                    "null_text_embedding": model.null_text_embedding.detach().cpu().clone(),
                    "read_calibration_scale": model.read_implant.read_calibration_scale.detach().cpu().clone(),
                    "null_abstain_weight": model.read_implant.null_abstain_weight.detach().cpu().clone(),
                    "glyph_bias_beta": model.read_implant.glyph_bias_beta.detach().cpu().clone(),
                },
            })
            print(
                f"[mini late seed={run_seed}] {combo} score={score:.5f} "
                f"pair={metrics['source_pair_acc']:.3f} null={metrics['clean_null_acc']:.3f} "
                f"readable={metrics['readable_reject_acc']:.3f} "
                f"scale={scalar_metrics['read_calibration_scale']:.4f} "
                f"null_w={scalar_metrics['null_abstain_weight']:.4f} glyph_b={scalar_metrics['glyph_bias_beta']:.4f}"
            )

    aggregate_rows = _aggregate_seed_rows(seed_rows)
    late_winner = _select_aggregate_winner(aggregate_rows, "late")
    best_late = tuple(int(x) for x in str(late_winner["blocks"]).split("+"))
    best_late_score = float(late_winner["score"])
    best_late_state = _representative_state(
        state_records[("late", best_late)], best_late_score
    )
    for row in sorted(
        (row for row in aggregate_rows if row.get("stage") == "late"),
        key=lambda item: float(item["score"]), reverse=True,
    ):
        print(
            f"[mini late mean] ({row['blocks']}) score={float(row['score']):.5f}"
            f"±{float(row.get('score_std', 0.0)):.5f} "
            f"pair={float(row['source_pair_acc']):.3f} null={float(row['clean_null_acc']):.3f} "
            f"readable={float(row['readable_reject_acc']):.3f}"
        )

    selected = {
        "ortho_blocks": list(fixed_ortho),
        "source_blocks": list(best_source),
        "late_blocks": list(best_late),
        "scores": {
            "ortho_mean": best_ortho_score,
            "ortho_std": float(ortho_winner.get("score_std", 0.0)),
            "source_mean": best_source_score,
            "source_std": float(source_winner.get("score_std", 0.0)),
            "late_mean": best_late_score,
            "late_std": float(late_winner.get("score_std", 0.0)),
        },
        "mini_steps_per_config": steps,
        "mini_seeds": mini_seeds,
        "fixed_ortho_blocks": list(fixed_ortho),
        "source_candidates": [list(combo) for combo in source_combos],
        "late_candidates": [list(combo) for combo in late_combos],
        "late_selection_formula": (
            "0.5*source_pair_acc + 1.0*clean_null_acc + "
            "0.5*readable_reject_acc - 0.1*source_loss"
        ),
        "late_read_calibration_scale_frozen": True,
        "late_content_anchor_only": not args.mini_late_allow_content_mixture,
    }
    _write_mini_csv(out_dir / "mini_benchmark_results.csv", aggregate_rows)
    _write_mini_csv(out_dir / "mini_benchmark_per_seed.csv", seed_rows)
    _plot_mini_rows(out_dir, aggregate_rows)
    write_json(
        out_dir / "mini_benchmark_selected.json",
        {"selected": selected, "aggregate_rows": aggregate_rows, "seed_rows": seed_rows},
    )
    _write_mini_report(out_dir / "mini_benchmark_report.md", aggregate_rows, selected)
    torch.save(
        {
            "selected": selected,
            "orthographic_state": best_ortho_state,
            "source_state": best_source_state,
            "late_state": best_late_state,
        },
        out_dir / "mini_benchmark_selected_components.pt",
    )
    print("=" * 78)
    print(
        f"[mini selected] ortho={fixed_ortho} source={best_source} late={best_late} "
        f"over {len(mini_seeds)} seeds"
    )
    print(f"[mini outputs] {out_dir}")
    print("=" * 78)
    return selected

def resolve_authoritative_branch_config(args: argparse.Namespace) -> Dict[str, Any]:
    path = args.branch_config
    if path is None:
        return {}
    config = read_json(path)
    required = ("ortho_blocks", "source_blocks", "late_blocks", "early_expanded_width")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing keys in authoritative branch config {path}: {missing}")
    mapping = {
        "ortho_tap_blocks": [int(value) for value in config["ortho_blocks"]],
        "source_tap_blocks": [int(value) for value in config["source_blocks"]],
        "read_tap_blocks": [int(value) for value in config["late_blocks"]],
    }
    for attribute, expected in mapping.items():
        current = getattr(args, attribute, None)
        if current is not None and [int(value) for value in current] != expected:
            raise ValueError(
                f"{attribute}={list(current)} conflicts with authoritative {path}: {expected}"
            )
        setattr(args, attribute, expected)
    return config


def _rng_sha256() -> str:
    state = torch.get_rng_state().detach().cpu().contiguous()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def _migration_init_fingerprint(model: torch.nn.Module, seed: int, rng_before: str, rng_after: str) -> Dict[str, Any]:
    prefixes = (
        "read_implant.source_head.",
        "read_implant.orthographic_bridge.",
        "read_implant.trust_router.",
    )
    selected = []
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        value = tensor.detach().cpu().float().contiguous()
        raw = value.numpy().tobytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(raw)
        selected.append({
            "name": name,
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "mean": float(value.mean()) if value.numel() else 0.0,
            "std": float(value.std(unbiased=False)) if value.numel() else 0.0,
            "rms": float(value.square().mean().sqrt()) if value.numel() else 0.0,
            "abs_max": float(value.abs().max()) if value.numel() else 0.0,
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    build_info = dict(getattr(model, "_anytext_build_info", {}) or {})
    return {
        "seed": int(seed),
        "rng_state_sha256_before_model_load": rng_before,
        "rng_state_sha256_after_model_load": rng_after,
        "combined_final_branch_sha256": digest.hexdigest(),
        "tensor_count": len(selected),
        "build_info": build_info,
        "tensors": selected,
    }


def train(args: argparse.Namespace) -> None:
    if (
        args.start_phase != "1a"
        and args.implant_checkpoint is not None
        and args.reset_tap_logits_uniform
    ):
        raise ValueError(
            "Unsafe continuation: --reset_tap_logits_uniform would erase tap weights "
            "loaded from --implant_checkpoint. Resume phases 1b/1c with tap reset disabled, "
            "or restart at phase 1a without an implant checkpoint."
        )
    if args.reset_router_components and args.implant_checkpoint is None:
        raise ValueError(
            "--reset_router_components requires --implant_checkpoint so the "
            "forced experts can be restored before arbitration is reset."
        )
    if args.router_only_1c and args.start_phase != "1c":
        raise ValueError("--router_only_1c requires --start_phase 1c")
    if args.router_aux_enabled:
        if args.start_phase != "1c":
            raise ValueError("--router_aux_enabled continuation package requires --start_phase 1c")
        if args.run_benchmark_validation and args.select_by_benchmark_score:
            raise ValueError(
                "Evaluation-only benchmark firewall: SCAM/RTA monitoring may run during "
                "training, but benchmark metrics must not select checkpoints. "
                "Set --select_by_benchmark_score off."
            )
        if args.reset_router_components:
            raise ValueError("--router_aux_enabled is incompatible with --reset_router_components")
        if args.router_aux_bce_weight <= 0.0 or args.router_aux_rank_weight <= 0.0:
            raise ValueError("router auxiliary BCE/rank weights must both be > 0")
        if args.router_aux_rank_margin < 0.0:
            raise ValueError("router auxiliary rank margin must be >= 0")
        if args.router_aux_targets_per_class <= 0:
            raise ValueError("router_aux_targets_per_class must be positive")
    if args.run_benchmark_validation and args.select_by_benchmark_score:
        raise ValueError(
            "SCAM/RTA are evaluation-only: benchmark monitoring cannot be used for "
            "checkpoint selection. Disable --select_by_benchmark_score."
        )
    if args.read_null_enabled:
        if args.read_attention_architecture not in {"softmax", "sigmoid_all"}:
            raise ValueError("READ_NULL requires softmax or raw sigmoid_all PIECES attention")
        if args.read_null_start_phase not in PHASES:
            raise ValueError(f"Unknown --read_null_start_phase={args.read_null_start_phase!r}")
        if PHASES.index(args.read_null_start_phase) < PHASES.index("1b5"):
            raise ValueError("This branch intentionally introduces READ_NULL no earlier than phase 1b5")
        if int(args.read_null_insert_block) > min(int(x) for x in args.read_tap_blocks):
            raise ValueError(
                "READ_NULL must be inserted no later than the first late-reader tap; "
                f"insert={args.read_null_insert_block}, taps={args.read_tap_blocks}"
            )

    branch_config = resolve_authoritative_branch_config(args)
    started = time.time()
    set_seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    # Historical ownership of the architecture boundary: set_seed() above, then
    # construct/load the final model immediately.  Any missing final source/ortho/trust
    # modules are therefore initialized by this process under args.seed, exactly as
    # in the working scattercode.
    write_json(out_dir / "config.json", vars(args))
    clip_module = load_clip_package(args)
    rng_before_model_load = _rng_sha256()
    print(f"[model] loading {args.model_path}")
    initial_read_null_enabled = bool(
        args.read_null_enabled
        and PHASES.index(args.start_phase) >= PHASES.index(args.read_null_start_phase)
    )
    model, _, load_info = load_openai_clip_anything(
        clip_module,
        str(args.model_path),
        device=str(device),
        read_attention_architecture=args.read_attention_architecture,
        read_null_enabled=initial_read_null_enabled,
        read_null_insert_block=args.read_null_insert_block,
        reuse_full_model_pickle=False,
    )
    print(f"[model] source={load_info.source_kind} format={load_info.detected_format}")
    rng_after_model_load = _rng_sha256()
    # Keep trainable parameters and optimizer state in fp32; CUDA autocast handles
    # the heavy matrix multiplications exactly as in the prior three-stage trainer.
    model.float()
    if initial_read_null_enabled:
        if not bool(getattr(model, "read_null_enabled", False)):
            raise RuntimeError("Loader did not enable READ_NULL on the model")
        if not isinstance(getattr(model.visual, "read_null_token", None), torch.Tensor):
            raise RuntimeError("READ_NULL enabled but visual token parameter is missing")
        print(
            f"[READ_NULL] enabled insert_before_B{model.read_null_insert_block} "
            f"initial_param_norm={float(model.visual.read_null_token.detach().float().norm()):.6f}"
        )
    migration_fingerprint = _migration_init_fingerprint(
        model, args.seed, rng_before_model_load, rng_after_model_load
    )
    build_info = migration_fingerprint.get("build_info", {})
    if args.require_legacy_migration_input and not bool(build_info.get("migrated_from_legacy", False)):
        raise RuntimeError(
            "final_base requires a LEGACY joint input whose migration happens after set_seed(), "
            f"but loader build_info={build_info!r}. Refusing to train a pre-migrated/final checkpoint."
        )
    write_json(out_dir / "migration_init_fingerprint.json", migration_fingerprint)
    print(
        "[migration] "
        f"input={build_info.get('input_architecture_stage', 'unknown')} -> "
        f"final migrated={build_info.get('migrated_from_legacy', False)} "
        f"seed={args.seed} branch_sha256={migration_fingerprint['combined_final_branch_sha256']}"
    )

    requested_amp_dtype = str(args.amp_dtype)
    precision = resolve_precision(requested_amp_dtype, device)
    args.amp_dtype_requested = requested_amp_dtype
    args.amp_dtype = precision.resolved
    print(f"[precision] {precision_summary(precision)}")
    precision_diag = PrecisionDiagnostics(
        out_dir,
        "final_all_weights" if getattr(args, "all_weights_grad", False) else "final_base",
        PrecisionDiagnosticsConfig(
            enabled=args.precision_diagnostics,
            log_every_optimizer_steps=args.precision_log_every,
            example_values_per_tensor=args.precision_example_values,
            selected_parameter_limit=args.precision_parameter_limit,
            max_values_for_statistics=args.precision_max_values,
            save_plots=args.precision_save_plots,
        ),
        requested_precision=requested_amp_dtype,
        resolved_precision=precision.resolved,
        device=device,
    )
    write_json(out_dir / "config.json", vars(args))
    load_optional_implant(model, args.implant_checkpoint)
    router_aux_init_info = router_auxiliary.load_router_init_checkpoint(
        model, args.router_aux_init_checkpoint if args.router_aux_enabled else None
    )
    if router_aux_init_info.get("loaded"):
        print(
            "[router aux init] "
            f"{router_aux_init_info['path']} fc3_bias={router_aux_init_info['fc3_bias']:.4f} "
            f"fc3_weight_norm={router_aux_init_info['fc3_weight_norm']:.4f}"
        )
        write_json(out_dir / "router_aux_init.json", router_aux_init_info)
    if args.read_attention_architecture == "sigmoid_mass":
        numerics_version = getattr(
            model.read_implant.read_bridge, "NUMERICS_VERSION", None
        )
        if numerics_version != 2:
            raise RuntimeError(
                "Stale sigmoid implementation loaded: expected NUMERICS_VERSION=2 "
                f"but found {numerics_version!r}. Replace the active gmpclipattnamp/model.py "
                "before training."
            )
        print("[sigmoid numerics] version=2 fp32_textness_logit=enabled")
    elif args.read_attention_architecture == "sigmoid_all":
        components = {
            "read": model.read_implant.read_bridge,
            "ortho": model.read_implant.orthographic_bridge,
            "content": model.read_implant.content_pool,
        }
        stale = {
            name: (getattr(module, "ATTENTION_NORMALIZATION", None), getattr(module, "NORMALIZES_ATTENTION", None))
            for name, module in components.items()
            if getattr(module, "ATTENTION_NORMALIZATION", None) != "independent_sigmoid_raw"
            or bool(getattr(module, "NORMALIZES_ATTENTION", True))
        }
        if stale:
            raise RuntimeError(f"Stale/normalized sigmoid_all PIECES implementation loaded: {stale}")
        print("[PIECES sigmoid] raw independent sigmoid verified: read+ortho+content, no renormalization")

    if args.mini_config_json is not None:
        selected = read_json(args.mini_config_json)
        selected = selected.get("selected", selected) if isinstance(selected, dict) else selected
        args.ortho_tap_blocks = [int(x) for x in selected["ortho_blocks"]]
        args.source_tap_blocks = [int(x) for x in selected["source_blocks"]]
        args.read_tap_blocks = [int(x) for x in selected["late_blocks"]]
        print(f"[mini config] loaded {args.mini_config_json}")
        # branch_config.json remains authoritative even when a mini-search result
        # is supplied; a stale selection must fail rather than silently override it.
        branch_config = resolve_authoritative_branch_config(args)
        write_json(out_dir / "config.json", vars(args))

    late_blocks = args.read_tap_blocks or model.read_implant.tap_block_list()
    ortho_blocks = args.ortho_tap_blocks or model.read_implant.ortho_block_list()
    source_blocks = args.source_tap_blocks or model.read_implant.source_block_list()
    vision_layer_count = len(model.visual.transformer.resblocks)
    bad_blocks = sorted({
        int(block)
        for block in list(late_blocks) + list(ortho_blocks) + list(source_blocks)
        if int(block) < 0 or int(block) >= vision_layer_count
    })
    if bad_blocks:
        raise ValueError(
            f"Invalid branch tap blocks {bad_blocks} for {vision_layer_count} visual blocks"
        )
    model.read_implant.set_tap_blocks(
        late_blocks, reset_uniform=args.reset_tap_logits_uniform
    )
    model.read_implant.set_ortho_tap_blocks(
        ortho_blocks, reset_uniform=args.reset_tap_logits_uniform
    )
    model.read_implant.set_source_tap_blocks(
        source_blocks, reset_uniform=args.reset_tap_logits_uniform
    )
    if branch_config:
        actual_width = int(model.read_implant.early_expanded_width_config)
        expected_width = int(branch_config["early_expanded_width"])
        if actual_width != expected_width:
            raise ValueError(
                "Model early expanded width conflicts with authoritative branch config: "
                f"model={actual_width}, config={expected_width}"
            )

    if float(model.null_text_embedding.detach().abs().sum()) == 0.0:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 49411)
        value = torch.randn(
            model.null_text_embedding.numel(), generator=generator
        ) * 0.02
        model.set_null_text_token_embedding(value)

    if args.rank1_probe_npz is not None:
        data = np.load(args.rank1_probe_npz)
        if args.rank1_probe_key not in data:
            raise KeyError(
                f"Probe key {args.rank1_probe_key!r} not found in {args.rank1_probe_npz}; "
                f"available={list(data.files)}"
            )
        model.read_implant.set_read_probe(torch.from_numpy(np.asarray(data[args.rank1_probe_key])))
        print(f"[probe] enabled {args.rank1_probe_npz}:{args.rank1_probe_key}")
    else:
        model.read_implant.set_read_probe(None)
        print("[probe] disabled (zero rank-1 features)")

    if args.reset_router_components:
        reset_router_components(model)
        print("[reset] trust_router + auto_read_scale restored to conservative initialization")
    print(
        f"[model] ortho={model.read_implant.ortho_block_list()} "
        f"source={model.read_implant.source_block_list()} "
        f"late={model.read_implant.tap_block_list()} "
        f"weights={{'ortho': {model.read_implant.ortho_tap_logits.softmax(0).tolist()}, "
        f"'source': {model.read_implant.source_tap_logits.softmax(0).tolist()}, "
        f"'read': {model.read_implant.read_tap_logits.softmax(0).tolist()}, "
        f"'content': {model.read_implant.content_tap_logits.softmax(0).tolist()}}}"
    )
    write_json(
        out_dir / "resolved_branch_config.json",
        {
            "ortho_blocks": model.read_implant.ortho_block_list(),
            "source_blocks": model.read_implant.source_block_list(),
            "late_blocks": model.read_implant.tap_block_list(),
            "early_expanded_width": int(model.read_implant.early_expanded_width_config),
        },
    )
    model.train()

    patch_count = int((model.visual.positional_embedding.shape[0] - 1))
    sources_train = {
        "imagenet": ImageNetPacketSource(
            args.imagenet_text_root,
            args.imagenet_handwriting_root,
            "train",
            args.image_size,
            patch_count,
            args.handwriting_probability,
            args.flip_probability,
            args.control_font_paths,
        ),
        "imagenet_math": ContinuousMathImageNetPacketSource(
            args.imagenet_text_root,
            args.imagenet_wnid_json,
            "train",
            args.image_size,
            patch_count,
            args.control_font_paths,
            args.math_curriculum_words,
            args.math_curriculum_families,
            args.math_curriculum_crop_probability,
        ) if args.math_curriculum_enabled else None,
        "clevr": ClevrPropertyPacketSource(
            args.clevr_train_root,
            args.clevr_metadata_jsonl,
            "train",
            args.image_size,
            patch_count,
            args.control_font_paths,
            args.clevr_colored_text_probability,
            args.clevr_ood_color_probability,
            args.clevr_standalone_count_probability,
            args.clevr_canny_low,
            args.clevr_canny_high,
            args.clevr_canny_dilate_px,
            args.clevr_placement_margin_px,
            args.clevr_max_obstacle_fraction,
            args.clevr_placement_stride_px,
            args.clevr_fill_contours,
            args.clevr_min_contour_area_fraction,
            args.clevr_min_font_size,
            args.clevr_max_font_size,
        ) if args.clevr_enabled else None,
        "textcaps": TextCapsPacketSource(
            args.textcaps_root,
            "train",
            args.image_size,
            patch_count,
            args.textcaps_mask_weight,
            args.flip_probability,
        ),
        "coco": CocoSprightPacketSource(
            args.coco_root,
            args.coco_train_json,
            args.coco_train_gpt_json,
            args.image_size,
            patch_count,
            args.coco_changed_probability,
            args.flip_probability,
            args.coco_train_reading_json,
            args.coco_reading_probability,
            args.coco_reading_mask_weight,
        ),
    }
    sources_val = {
        "imagenet": ImageNetPacketSource(
            args.imagenet_text_root,
            args.imagenet_handwriting_root,
            "val",
            args.image_size,
            patch_count,
            args.handwriting_probability,
            0.0,
            args.control_font_paths,
        ),
        "imagenet_math": ContinuousMathImageNetPacketSource(
            args.imagenet_text_root,
            args.imagenet_wnid_json,
            "val",
            args.image_size,
            patch_count,
            args.control_font_paths,
            args.math_curriculum_words,
            args.math_curriculum_families,
            args.math_curriculum_crop_probability,
        ) if args.math_curriculum_enabled else None,
        "clevr": ClevrPropertyPacketSource(
            args.clevr_val_root,
            args.clevr_metadata_jsonl,
            "val",
            args.image_size,
            patch_count,
            args.control_font_paths,
            args.clevr_colored_text_probability,
            args.clevr_ood_color_probability,
            args.clevr_standalone_count_probability,
            args.clevr_canny_low,
            args.clevr_canny_high,
            args.clevr_canny_dilate_px,
            args.clevr_placement_margin_px,
            args.clevr_max_obstacle_fraction,
            args.clevr_placement_stride_px,
            args.clevr_fill_contours,
            args.clevr_min_contour_area_fraction,
            args.clevr_min_font_size,
            args.clevr_max_font_size,
        ) if args.clevr_enabled else None,
        "textcaps": TextCapsPacketSource(
            args.textcaps_root,
            "validation",
            args.image_size,
            patch_count,
            args.textcaps_mask_weight,
            0.0,
        ),
        "coco": CocoSprightPacketSource(
            args.coco_root,
            args.coco_val_json,
            args.coco_val_gpt_json,
            args.image_size,
            patch_count,
            args.coco_changed_probability,
            0.0,
            args.coco_val_reading_json,
            args.coco_reading_probability,
            args.coco_reading_mask_weight,
        ),
    }
    router_aux_sources = router_auxiliary.make_sources(
        core=sys.modules[__name__], model=model, args=args
    ) if args.router_aux_enabled else None
    if router_aux_sources is not None:
        write_json(out_dir / "router_aux_dataset_coverage.json", router_aux_sources["coverage"])
        print("[router aux firewall] auxiliary sources = ImageNet + salt_n_pepper only; benchmarks excluded")
        print(f"[router aux coverage] {router_aux_sources['coverage']}")

    train_mixer = SourceMixer(sources_train, phase_mix(args, "1a"), args.seed + 11)
    val_mixer = SourceMixer(sources_val, phase_mix(args, "1a"), args.seed + 29)

    math_coverage: Dict[str, Any] = {}
    if args.math_curriculum_enabled:
        tiny_source = sources_train.get("imagenet_math")
        if not isinstance(tiny_source, ContinuousMathImageNetPacketSource):
            raise RuntimeError("math_curriculum_enabled but imagenet_math source was not constructed")
        math_coverage = tiny_source.coverage()
        write_json(out_dir / "math_curriculum_coverage.json", math_coverage)
        preview_paths = tiny_source.save_preview_grids(
            out_dir / "math_curriculum_examples_grid",
            count=args.math_curriculum_preview_count,
            seed=args.seed + 731,
        )
        print(
            f"[math-curriculum] concepts={math_coverage.get('concepts')} "
            f"groups={math_coverage.get('groups')} previews={len(preview_paths)}"
        )

    clevr_coverage: Dict[str, Any] = {}
    if args.clevr_enabled:
        clevr_source = sources_train.get("clevr")
        if not isinstance(clevr_source, ClevrPropertyPacketSource):
            raise RuntimeError("clevr_enabled but CLEVR source was not constructed")
        clevr_coverage = clevr_source.coverage()
        write_json(out_dir / "clevr_property_coverage.json", clevr_coverage)
        clevr_preview_paths = clevr_source.save_previews(
            out_dir / "clevr_property_examples_canny",
            count=args.clevr_preview_count,
            seed=args.seed + 883,
        )
        print(
            f"[clevr-property] rows={clevr_coverage.get('rows')} "
            f"previews={len(clevr_preview_paths)} additive=true"
        )

    if args.mini_benchmark:
        run_mini_benchmark(model, clip_module, sources_train, sources_val, device, args)
        return

    first_batch = train_mixer.sample_batch(max(4, min(args.logical_batch_images, 8)))

    # Deterministically audit the new same-image hard negative before training.
    imagenet_audit = sources_train["imagenet"].sample(random.Random(args.seed + 101))
    adversarial_text = str(imagenet_audit.metadata["adversarial_text"])
    adversarial_any_negative = str(
        imagenet_audit.metadata["adversarial_any_negative"]
    )
    adversarial_any_index = imagenet_audit.captions.index(
        adversarial_any_negative
    )
    adversarial_read_index = imagenet_audit.captions.index(
        f"<text> {adversarial_text}"
    )
    if imagenet_audit.positive[:, adversarial_any_index].any():
        raise RuntimeError(
            "ImageNet adversarial <any> candidate unexpectedly has a positive."
        )
    if not bool(imagenet_audit.positive[2, adversarial_read_index]):
        raise RuntimeError(
            "ImageNet adversarial forced-read candidate lost its positive target."
        )

    preflight = preflight_mode_invariance(
        model, clip_module, first_batch, device, args,
        enforce_zero_impact=(
            args.start_phase == "1a"
            and args.implant_checkpoint is None
            and not bool(build_info.get("fresh_pieces_init", False))
        ),
    )
    dataset_summary = {
        "imagenet_train_groups": len(sources_train["imagenet"].group_ids),
        "imagenet_val_groups": len(sources_val["imagenet"].group_ids),
        "imagenet_handwriting_train_groups": len(sources_train["imagenet"].handwriting_by_group),
        "imagenet_math_enabled": bool(args.math_curriculum_enabled),
        "imagenet_math_groups": int(math_coverage.get("groups", 0)),
        "imagenet_math_concepts": int(math_coverage.get("concepts", 0)),
        "imagenet_math_words": list(math_coverage.get("words", [])),
        "clevr_enabled": bool(args.clevr_enabled),
        "clevr_additive": True if args.clevr_enabled else False,
        "clevr_rows": int(clevr_coverage.get("rows", 0)),
        "clevr_uses_metadata_geometry": bool(clevr_coverage.get("uses_metadata_geometry", False)),
        "textcaps_train_images": len(sources_train["textcaps"].rows),
        "textcaps_val_images": len(sources_val["textcaps"].rows),
        "coco_train_images": len(sources_train["coco"].rows),
        "coco_train_changed": len(sources_train["coco"].changed_rows),
        "coco_train_trusted_reading": len(sources_train["coco"].reading_rows),
        "coco_val_images": len(sources_val["coco"].rows),
        "coco_val_trusted_reading": len(sources_val["coco"].reading_rows),
        "sample_batch_images": int(first_batch.images.shape[0]),
        "sample_batch_captions": len(first_batch.captions),
        "sample_batch_sources": [meta.get("source") for meta in first_batch.packet_metadata],
        "sample_batch_negative_only_captions": int(
            (~first_batch.positive.any(dim=0)).sum().item()
        ),
        "imagenet_audit_adversarial_any_negative": adversarial_any_negative,
        "imagenet_audit_any_positive_count": int(
            imagenet_audit.positive[:, adversarial_any_index].sum().item()
        ),
        "imagenet_audit_text_positive": bool(
            imagenet_audit.positive[2, adversarial_read_index]
        ),
    }
    write_json(out_dir / "preflight.json", {"modes": preflight, "datasets": dataset_summary})
    print(f"[preflight] {preflight}")
    print(f"[datasets] {dataset_summary}")
    if args.preflight_only:
        print(f"[done] preflight only -> {out_dir}")
        return

    if args.router_aux_enabled:
        if router_aux_sources is None:
            raise RuntimeError("router_aux_enabled but auxiliary validation sources are missing")
        router_aux_baseline = router_auxiliary.evaluate_auxiliary(
            core=sys.modules[__name__], model=model, clip_module=clip_module,
            imagenet_source=router_aux_sources["imagenet_val"],
            salt_source=router_aux_sources["salt_val"],
            device=device, args=args, seed=int(args.router_aux_validation_seed),
            batches=int(args.router_aux_val_batches),
        )
        append_csv(
            out_dir / "router_aux_validation_log.csv",
            {"phase": "pre_1c", "epoch": 0, "step": 0, **router_aux_baseline},
        )
        print(
            "[router aux baseline] "
            f"auc={router_aux_baseline.get('router_aux_auc', float('nan')):.3f} "
            f"imagenet_auc={router_aux_baseline.get('router_aux_imagenet_auc', float('nan')):.3f} "
            f"trust(open/close)={router_aux_baseline.get('router_aux_trust_open', float('nan')):.3f}/"
            f"{router_aux_baseline.get('router_aux_trust_close', float('nan')):.3f} "
            f"pair_gap={router_aux_baseline.get('router_aux_pair_gap_mean', float('nan')):.3f}"
        )

    phase_epochs = {
        "1a": args.phase_1a_epochs,
        "1b": args.phase_1b_epochs,
        "1b5": args.phase_1b5_epochs,
        "1c": args.phase_1c_epochs,
    }
    start_index = PHASES.index(args.start_phase)
    global_step = 0

    for phase in PHASES[start_index:]:
        epochs = phase_epochs[phase]
        if epochs <= 0:
            continue
        print("=" * 90)
        print(f"[phase {phase}] epochs={epochs}")
        print("=" * 90)
        if (
            args.read_null_enabled
            and not bool(getattr(model, "read_null_enabled", False))
            and PHASES.index(phase) >= PHASES.index(args.read_null_start_phase)
        ):
            activate_read_null_token(model, args.read_null_insert_block)
        if phase == "1c" and abs(float(model.read_implant.auto_read_scale.detach())) < 1.0e-12:
            model.read_implant.auto_read_scale.data.fill_(args.auto_read_scale_start)

        trainable_names = set_trainable_phase(
            model, phase, router_only_1c=args.router_only_1c
        )
        assert_trainable_parameters_fp32(model, label=f"final phase {phase}")
        print(f"[phase {phase}] trainable tensors={len(trainable_names)}")
        write_json(out_dir / f"phase_{phase}_trainable.json", trainable_names)
        optimizer = optimizer_for_phase(model, args, phase)
        scaler = precision_make_grad_scaler(device, args.amp_dtype)
        total_steps = epochs * args.steps_per_epoch
        local_step = 0
        phase_optimizer_step = 0
        best_score = float("inf")

        train_mixer.set_weights(phase_mix(args, phase))
        val_mixer.set_weights(phase_mix(args, phase))
        write_json(out_dir / f"phase_{phase}_mix.json", phase_mix(args, phase))

        for epoch in range(1, epochs + 1):
            model.train()
            running: Dict[str, float] = {}
            optimizer.zero_grad(set_to_none=True)
            for step_in_epoch in range(1, args.steps_per_epoch + 1):
                clevr_probability = (
                    args.clevr_additive_probability_1b5
                    if phase == "1b5"
                    else args.clevr_additive_probability_1c
                )
                batch = batch_to_device(
                    sample_training_batch_with_additive_clevr(
                        train_mixer,
                        args.logical_batch_images,
                        sources_train.get("clevr"),
                        bool(args.clevr_enabled and phase in {"1b5", "1c"}),
                        clevr_probability,
                        args.clevr_packets_per_batch,
                    ),
                    device,
                )
                cosine_lr(optimizer, local_step, total_steps, args.warmup_fraction)
                do_step = (
                    step_in_epoch % args.grad_accum_steps == 0
                    or step_in_epoch == args.steps_per_epoch
                )
                next_optimizer_step = phase_optimizer_step + 1
                do_precision_log = do_step and precision_diag.should_log(
                    next_optimizer_step,
                    force=(epoch == epochs and step_in_epoch == args.steps_per_epoch),
                )
                diagnostic_tensors: Dict[str, torch.Tensor] = {}
                do_read_null_vit_log = bool(getattr(model, "read_null_enabled", False)) and ((global_step + 1) % args.log_every == 0)
                if hasattr(model.visual, "set_read_null_attention_capture"):
                    model.visual.set_read_null_attention_capture(do_read_null_vit_log)
                # Capture precision hooks on the *ordinary task forward only*.
                # The auxiliary performs a second no-grad model forward; keeping it
                # outside this context prevents aux activations from overwriting the
                # task tensors used by the precision diagnostics.
                with precision_diag.capture_model_hooks(model, do_precision_log) as hook_tensors:
                    with amp_context(device, args.amp_dtype):
                        task_loss, metrics = compute_batch_loss(
                            model, clip_module, batch, device, args, phase,
                            diagnostic_tensors=diagnostic_tensors if do_precision_log else None,
                        )

                with amp_context(device, args.amp_dtype):
                    loss = task_loss
                    aux_bce_weighted = task_loss * 0.0
                    aux_rank_weighted = task_loss * 0.0
                    if args.router_aux_enabled and phase == "1c":
                        if router_aux_sources is None:
                            raise RuntimeError("router_aux_enabled but auxiliary sources were not constructed")
                        aux_bce, aux_rank, aux_metrics = router_auxiliary.compute_auxiliary_loss(
                            core=sys.modules[__name__],
                            model=model, clip_module=clip_module,
                            imagenet_source=router_aux_sources["imagenet_train"],
                            salt_source=router_aux_sources["salt_train"],
                            rng=router_aux_sources["train_rng"],
                            device=device, args=args,
                        )
                        aux_bce_weighted = float(args.router_aux_bce_weight) * aux_bce
                        aux_rank_weighted = float(args.router_aux_rank_weight) * aux_rank
                        loss = task_loss + aux_bce_weighted + aux_rank_weighted
                        metrics = dict(metrics)
                        metrics["loss_task_main"] = float(task_loss.detach())
                        metrics["loss_router_aux_bce_weighted"] = float(aux_bce_weighted.detach())
                        metrics["loss_router_aux_rank_weighted"] = float(aux_rank_weighted.detach())
                        metrics.update(aux_metrics)
                        metrics["loss_total"] = float(loss.detach())
                        diagnostic_tensors["final.loss.router_aux_bce_weighted"] = aux_bce_weighted.reshape(1)
                        diagnostic_tensors["final.loss.router_aux_rank_weighted"] = aux_rank_weighted.reshape(1)
                        diagnostic_tensors["final.loss.total_with_router_aux"] = loss.reshape(1)
                    if not bool(torch.isfinite(loss).all()):
                        nonfinite_metrics = {
                            key: value for key, value in metrics.items()
                            if not math.isfinite(float(value))
                        }
                        diagnostic_path = out_dir / (
                            f"nonfinite_{phase}_epoch_{epoch:03d}_step_{step_in_epoch:05d}.json"
                        )
                        write_json(
                            diagnostic_path,
                            {
                                "phase": phase,
                                "epoch": epoch,
                                "step_in_epoch": step_in_epoch,
                                "global_step": global_step + 1,
                                "read_attention_architecture": str(
                                    model.read_attention_architecture
                                ),
                                "nonfinite_metrics": nonfinite_metrics,
                                "glyph_bias_beta": float(
                                    model.read_implant.glyph_bias_beta.detach().float()
                                ),
                                "sigmoid_head_bias": (
                                    model.read_implant.read_bridge.sigmoid_head_bias
                                    .detach().float().cpu().tolist()
                                    if hasattr(
                                        model.read_implant.read_bridge,
                                        "sigmoid_head_bias",
                                    )
                                    else None
                                ),
                            },
                        )
                        raise FloatingPointError(
                            "Non-finite loss detected before backward: "
                            f"phase={phase} epoch={epoch} step={step_in_epoch}; "
                            f"metrics={nonfinite_metrics}. Diagnostic: {diagnostic_path}"
                        )
                    if (
                        args.router_aux_enabled and phase == "1c" and do_step
                        and args.router_aux_grad_log_every > 0
                        and next_optimizer_step % args.router_aux_grad_log_every == 0
                    ):
                        metrics.update(router_auxiliary.gradient_contribution_metrics(
                            model=model, task_loss=task_loss,
                            weighted_bce_loss=aux_bce_weighted,
                            weighted_rank_loss=aux_rank_weighted,
                        ))
                    scaled_loss = loss / args.grad_accum_steps
                if hasattr(model.visual, "set_read_null_attention_capture"):
                    model.visual.set_read_null_attention_capture(False)
                scaler.scale(scaled_loss).backward()
                if do_step:
                    scaler.unscale_(optimizer)
                    if args.router_aux_enabled and phase == "1c":
                        metrics["router_grad_preclip_norm"] = router_auxiliary.router_grad_norm(model)
                    precision_snapshot = {}
                    if do_precision_log:
                        diagnostic_tensors["final.loss.scaled_for_accumulation"] = scaled_loss.reshape(1)
                        precision_snapshot = precision_diag.capture_before_update(
                            model=model, optimizer=optimizer, scaler=scaler,
                            optimizer_step=next_optimizer_step, micro_step=global_step + 1,
                            epoch=epoch, phase=phase, tensors=diagnostic_tensors,
                            hook_tensors=hook_tensors,
                            extra={
                                "autocast_requested": requested_amp_dtype,
                                "autocast_resolved": args.amp_dtype,
                                "gradient_state": "unscaled_preclip",
                                "grad_accum_steps": args.grad_accum_steps,
                            },
                        )
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        args.grad_clip,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    phase_optimizer_step += 1
                    if do_precision_log:
                        precision_diag.capture_after_update(precision_snapshot, model)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        model.token_embedding.weight[model.hard_text_token_id].copy_(
                            model.hard_text_embedding.to(model.token_embedding.weight.dtype)
                        )
                        model.token_embedding.weight[model.null_text_token_id].copy_(
                            model.null_text_embedding.to(model.token_embedding.weight.dtype)
                        )
                        model.read_implant.glyph_bias_beta.clamp_(0.0, args.glyph_beta_max)
                        model.read_implant.auto_read_scale.clamp_(0.0, args.auto_read_scale_max)
                        model.read_implant.null_abstain_weight.clamp_(0.0, args.null_abstain_max)
                        model.read_implant.read_calibration_scale.clamp_(0.1, 10.0)

                local_step += 1
                global_step += 1
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value
                if global_step % args.log_every == 0:
                    row = {
                        "phase": phase,
                        "epoch": epoch,
                        "step": global_step,
                        **metrics,
                    }
                    append_csv(out_dir / "training_log.csv", row)
                    if args.router_aux_enabled and phase == "1c":
                        append_csv(
                            out_dir / "router_aux_training_log.csv",
                            {"phase": phase, "epoch": epoch, "step": global_step, **{
                                k: v for k, v in metrics.items()
                                if k.startswith("router_aux_") or k.startswith("router_grad_")
                                or k.startswith("loss_router_aux_") or k == "loss_task_main"
                            }},
                        )
                    if args.read_attention_architecture == "sigmoid_all":
                        append_csv(
                            out_dir / "pieces_sigmoid_attention_log.csv",
                            {
                                "phase": phase, "epoch": epoch, "step": global_step,
                                **{k: v for k, v in metrics.items() if k.startswith("pieces_sig_")},
                            },
                        )
                    print(
                        f"[{phase} e{epoch} s{step_in_epoch}] "
                        f"loss={metrics['loss_total']:.4f} mp={metrics['loss_mp']:.4f} "
                        f"src={metrics['loss_source']:.4f} mask={metrics['loss_mask']:.4f} "
                        f"auto_neg={metrics['loss_auto_negative']:.4f} "
                        f"gate={metrics['gate_mean']:.4f} scale={metrics['auto_read_scale']:.4f} "
                        f"null={metrics['null_abstain_weight']:.4f} "
                        + (
                            f"rn={metrics.get('loss_read_null', 0.0):.4f} "
                            f"rn_sup={metrics.get('read_null_attention_supported', 0.0):.3f} "
                            f"rn_unsup={metrics.get('read_null_attention_unsupported', 0.0):.3f}"
                            if bool(getattr(model, "read_null_enabled", False)) else ""
                        )
                    )
                    if args.router_aux_enabled and phase == "1c":
                        print(
                            "    [router aux] "
                            f"auc={metrics.get('router_aux_auc', float('nan')):.3f} "
                            f"imagenet_auc={metrics.get('router_aux_imagenet_auc', float('nan')):.3f} "
                            f"z(open/close)={metrics.get('router_aux_z_open', float('nan')):.3f}/"
                            f"{metrics.get('router_aux_z_close', float('nan')):.3f} "
                            f"trust(open/close)={metrics.get('router_aux_trust_open', float('nan')):.3f}/"
                            f"{metrics.get('router_aux_trust_close', float('nan')):.3f} "
                            f"pair_gap={metrics.get('router_aux_pair_gap_mean', float('nan')):.3f}"
                        )
                    if args.read_attention_architecture == "sigmoid_all":
                        print(
                            "    [PIECES sigmoid] "
                            f"mass read={metrics.get('pieces_sig_read_mass_rollup', float('nan')):.3f} "
                            f"ortho={metrics.get('pieces_sig_ortho_mass_rollup', float('nan')):.3f} "
                            f"content={metrics.get('pieces_sig_content_mass_rollup', float('nan')):.3f}"
                        )
                    if bool(getattr(model, "read_null_enabled", False)):
                        append_csv(
                            out_dir / "read_null_training_log.csv",
                            {"phase": phase, "epoch": epoch, "step": global_step, **{
                                key: value for key, value in metrics.items()
                                if key.startswith("read_null_") or key == "loss_read_null"
                            }},
                        )

            epoch_train = {
                key: value / max(1, args.steps_per_epoch) for key, value in running.items()
            }
            val_metrics = evaluate(model, clip_module, val_mixer, device, args, phase)
            router_aux_val_metrics: Dict[str, float] = {}
            if args.router_aux_enabled and phase == "1c":
                if router_aux_sources is None:
                    raise RuntimeError("router_aux_enabled but auxiliary validation sources are missing")
                router_aux_val_metrics = router_auxiliary.evaluate_auxiliary(
                    core=sys.modules[__name__], model=model, clip_module=clip_module,
                    imagenet_source=router_aux_sources["imagenet_val"],
                    salt_source=router_aux_sources["salt_val"],
                    device=device, args=args, seed=int(args.router_aux_validation_seed),
                    batches=int(args.router_aux_val_batches),
                )
                append_csv(
                    out_dir / "router_aux_validation_log.csv",
                    {"phase": phase, "epoch": epoch, "step": global_step, **router_aux_val_metrics},
                )
                print(
                    "[router aux val] "
                    f"auc={router_aux_val_metrics.get('router_aux_auc', float('nan')):.3f} "
                    f"imagenet_auc={router_aux_val_metrics.get('router_aux_imagenet_auc', float('nan')):.3f} "
                    f"trust(open/close)={router_aux_val_metrics.get('router_aux_trust_open', float('nan')):.3f}/"
                    f"{router_aux_val_metrics.get('router_aux_trust_close', float('nan')):.3f} "
                    f"pair_gap={router_aux_val_metrics.get('router_aux_pair_gap_mean', float('nan')):.3f}"
                )
            math_val_metrics: Dict[str, float] = {}
            if args.math_curriculum_enabled and args.math_curriculum_val_batches > 0:
                tiny_eval_mixer = SourceMixer(
                    {"imagenet_math": sources_val["imagenet_math"]},
                    {"imagenet_math": 1.0},
                    args.seed + 1701,
                )
                original_val_batches = args.val_batches
                try:
                    args.val_batches = args.math_curriculum_val_batches
                    math_val_metrics = evaluate(
                        model, clip_module, tiny_eval_mixer, device, args, phase
                    )
                finally:
                    args.val_batches = original_val_batches
            clevr_val_metrics: Dict[str, float] = {}
            if args.clevr_enabled and args.clevr_val_batches > 0:
                clevr_eval_mixer = SourceMixer(
                    {"clevr": sources_val["clevr"]},
                    {"clevr": 1.0},
                    args.seed + 1889,
                )
                original_val_batches = args.val_batches
                try:
                    args.val_batches = args.clevr_val_batches
                    clevr_val_metrics = evaluate(
                        model, clip_module, clevr_eval_mixer, device, args, phase
                    )
                finally:
                    args.val_batches = original_val_batches

            benchmark_metrics: Dict[str, float] = {}
            if (
                args.run_benchmark_validation
                and (epoch % args.benchmark_every_epochs == 0 or epoch == epochs)
            ):
                from training_support.validation.benchmark_anytext_validation import run_validation_benchmarks
                benchmark_metrics = run_validation_benchmarks(
                    model, clip_module, device, args, image_and_mask_transform
                )
                write_json(
                    out_dir / f"benchmarks_{phase}_epoch_{epoch:03d}.json",
                    benchmark_metrics,
                )
            score = val_metrics["loss_total"]
            if not math.isfinite(float(score)):
                diagnostic_path = out_dir / f"nonfinite_validation_{phase}_epoch_{epoch:03d}.json"
                write_json(
                    diagnostic_path,
                    {
                        "phase": phase,
                        "epoch": epoch,
                        "global_step": global_step,
                        "read_attention_architecture": str(
                            model.read_attention_architecture
                        ),
                        "validation_metrics": val_metrics,
                    },
                )
                raise FloatingPointError(
                    "Non-finite validation loss; refusing to continue without a "
                    f"best checkpoint. Diagnostic: {diagnostic_path}"
                )
            quality = benchmark_quality(benchmark_metrics)
            if args.select_by_benchmark_score and math.isfinite(quality):
                # Existing checkpoint selection minimizes selection_score.
                score = -quality
            summary = {
                "phase": phase,
                "epoch": epoch,
                "global_step": global_step,
                "selection_score": score,
                "benchmark_quality": quality,
                **{f"train_{k}": v for k, v in epoch_train.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"math_val_{k}": v for k, v in math_val_metrics.items()},
                **{f"clevr_val_{k}": v for k, v in clevr_val_metrics.items()},
                **benchmark_metrics,
            }
            append_csv(out_dir / "epoch_summary.csv", summary)
            checkpoint = compact_checkpoint(
                model, phase, epoch, global_step, val_metrics, args
            )
            torch.save(checkpoint, out_dir / f"phase_{phase}_last.pt")
            if args.save_every_epoch:
                torch.save(
                    checkpoint,
                    out_dir / f"phase_{phase}_epoch_{epoch:03d}.pt",
                )
            if score < best_score:
                best_score = score
                torch.save(checkpoint, out_dir / f"phase_{phase}_best.pt")
            if bool(getattr(model, "read_null_enabled", False)) and args.read_null_save_plots:
                save_read_null_diagnostic_plots(out_dir)
            print(
                f"[phase {phase} val] loss={val_metrics['loss_total']:.4f} "
                f"late_source={val_metrics['loss_source']:.4f} "
                f"ortho={val_metrics['loss_ortho']:.4f} mask={val_metrics['loss_mask']:.4f} "
                f"source_gate={val_metrics['source_gate_mean']:.4f} "
                f"trust={val_metrics['trust_mean']:.4f} effective={val_metrics['gate_mean']:.4f}"
            )
            if math_val_metrics:
                print(
                    f"[phase {phase} math-val] loss={math_val_metrics.get('loss_total', float('nan')):.4f} "
                    f"src={math_val_metrics.get('loss_source', float('nan')):.4f} "
                    f"ortho={math_val_metrics.get('loss_ortho', float('nan')):.4f} "
                    f"present={math_val_metrics.get('present_prob', float('nan')):.3f} "
                    f"rn_sup={math_val_metrics.get('read_null_attention_supported', float('nan')):.3f} "
                    f"rn_unsup={math_val_metrics.get('read_null_attention_unsupported', float('nan')):.3f}"
                )
            if clevr_val_metrics:
                print(
                    f"[phase {phase} clevr-val] loss={clevr_val_metrics.get('loss_total', float('nan')):.4f} "
                    f"src={clevr_val_metrics.get('loss_source', float('nan')):.4f} "
                    f"ortho={clevr_val_metrics.get('loss_ortho', float('nan')):.4f} "
                    f"present={clevr_val_metrics.get('present_prob', float('nan')):.3f} "
                    f"rn_sup={clevr_val_metrics.get('read_null_attention_supported', float('nan')):.3f} "
                    f"rn_unsup={clevr_val_metrics.get('read_null_attention_unsupported', float('nan')):.3f}"
                )
            if benchmark_metrics:
                bench_text = (
                    f"[phase {phase} bench] quality={quality:.4f} "
                    f"SCAM_any={benchmark_metrics.get('scam/SCAM/any_acc', float('nan')):.4f} "
                    f"RTA_any={benchmark_metrics.get('rta/RTA/any_acc', float('nan')):.4f}"
                )
                if "mvt/any_acc" in benchmark_metrics:
                    bench_text += f" MVT_any={benchmark_metrics['mvt/any_acc']:.4f}"
                print(bench_text)

        # Continue from the best checkpoint of this phase.
        best = torch_load_trusted(out_dir / f"phase_{phase}_best.pt")
        model.read_implant.load_state_dict(best["implant_state_dict"], strict=True)
        model.set_hard_text_token_embedding(best["hard_text_embedding"])
        model.set_null_text_token_embedding(best["null_text_embedding"])
        if bool(getattr(model, "read_null_enabled", False)):
            saved_read_null = best.get("read_null_token")
            if not isinstance(saved_read_null, torch.Tensor):
                raise RuntimeError(f"Best phase {phase} checkpoint is missing READ_NULL token")
            model.visual.read_null_token.data.copy_(
                saved_read_null.to(
                    device=model.visual.read_null_token.device,
                    dtype=model.visual.read_null_token.dtype,
                )
            )
        if args.export_merged_each_phase:
            export_merged(
                model,
                out_dir / f"phase_{phase}_best_merged_state_dict.pt",
            )

    if args.save_stage_complete_checkpoint:
        torch.save(
            compact_checkpoint(model, "complete", 0, global_step, {}, args),
            out_dir / "stage1_complete.pt",
        )
    if args.export_merged_final:
        export_merged(model, out_dir / "stage1_complete_merged_state_dict.pt")
    elapsed = time.time() - started
    write_json(
        out_dir / "run_metadata.json",
        {
            "elapsed_seconds": elapsed,
            "global_step": global_step,
            "device": str(device),
            "patch_count": patch_count,
        },
    )
    precision_diag.finalize()
    print(f"[done] {out_dir} ({elapsed / 60.0:.1f} min)")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--clip_package", default="gmpclipattnamp")
    parser.add_argument("--clip_package_root", type=Path, default=None)
    parser.add_argument(
        "--read_attention_architecture",
        choices=("softmax", "sigmoid_mass", "sigmoid_all"),
        default="softmax",
        help="Explicit reader architecture; softmax preserves legacy behavior.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/clip_xattn_training/main/joint/best_merged_state_dict__ungmp_oaiclip_fullmodel.pt",
        help=(
            "Base/full CLIP model reference. May be a local checkpoint path or a Hugging Face repo id; "
            "kept as a string so repo ids retain forward slashes on Windows."
        ),
    )
    parser.add_argument("--implant_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--require_legacy_migration_input",
        action="store_true",
        help=(
            "Require --model_path to still contain the legacy presence_pool layout. "
            "The final loader must perform migration in this seeded process."
        ),
    )
    parser.add_argument("--start_phase", choices=PHASES, default="1a")
    parser.add_argument("--read_null_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--read_null_insert_block", type=int, default=13)
    parser.add_argument(
        "--read_null_start_phase", choices=PHASES, default="1b5",
        help="Phase boundary at which READ_NULL is introduced; allows fresh PIECES 1A/1B without the token.",
    )
    parser.add_argument("--ortho_weight_1b5", type=float, default=0.50)
    parser.add_argument("--mask_weight_1b5", type=float, default=0.20)
    parser.add_argument("--presence_weight_1b5", type=float, default=0.10)
    parser.add_argument("--new_lr_factor_1b5", type=float, default=0.25)
    parser.add_argument("--math_curriculum_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--math_curriculum_words", default="all")
    parser.add_argument(
        "--math_curriculum_families",
        default="sine,moire,checker,voronoi,fbm,noise,lines",
        help="Comma-separated pure mathematical non-text hard-negative families.",
    )
    parser.add_argument(
        "--math_curriculum_crop_probability",
        type=float,
        default=0.20,
        help="Probability that natural ImageNet text is huge and recoverably cropped (70-95%% ink visible).",
    )
    parser.add_argument("--math_curriculum_preview_count", type=int, default=8)
    parser.add_argument("--math_curriculum_val_batches", type=int, default=16)
    parser.add_argument("--clevr_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clevr_additive_probability_1b5", type=float, default=0.50)
    parser.add_argument("--clevr_additive_probability_1c", type=float, default=0.50)
    parser.add_argument("--clevr_packets_per_batch", type=int, default=1)
    parser.add_argument("--clevr_val_batches", type=int, default=16)
    parser.add_argument("--clevr_preview_count", type=int, default=8)
    parser.add_argument("--clevr_colored_text_probability", type=float, default=0.15)
    parser.add_argument("--clevr_ood_color_probability", type=float, default=0.08)
    parser.add_argument("--clevr_standalone_count_probability", type=float, default=0.15)
    parser.add_argument("--clevr_canny_low", type=int, default=45)
    parser.add_argument("--clevr_canny_high", type=int, default=120)
    parser.add_argument("--clevr_canny_dilate_px", type=int, default=3)
    parser.add_argument("--clevr_placement_margin_px", type=int, default=4)
    parser.add_argument("--clevr_max_obstacle_fraction", type=float, default=0.025)
    parser.add_argument("--clevr_placement_stride_px", type=int, default=3)
    parser.add_argument("--clevr_fill_contours", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clevr_min_contour_area_fraction", type=float, default=0.001)
    parser.add_argument("--clevr_min_font_size", type=int, default=8)
    parser.add_argument("--clevr_max_font_size", type=int, default=42)
    parser.add_argument("--read_null_weight_1b5", type=float, default=1.0)
    parser.add_argument("--read_null_weight_1c", type=float, default=0.25)
    parser.add_argument("--lr_read_null_token", type=float, default=1.0e-3)
    parser.add_argument("--read_lr_factor_1b5", type=float, default=0.25)
    parser.add_argument("--mp_weight_1b5", type=float, default=0.10)
    parser.add_argument("--source_weight_1b5", type=float, default=0.50)
    parser.add_argument("--mix_1b5_coco", type=float, default=0.15)
    parser.add_argument("--mix_1b5_textcaps", type=float, default=0.20)
    parser.add_argument("--mix_1b5_imagenet", type=float, default=0.65)
    parser.add_argument("--read_null_save_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reset_router_components",
        action="store_true",
        help=(
            "After loading --implant_checkpoint, reset only trust_router and "
            "auto_read_scale; preserve forced content/read/glyph components."
        ),
    )
    parser.add_argument(
        "--router_only_1c",
        action="store_true",
        help="During phase 1C, train only trust_router and auto_read_scale.",
    )
    parser.add_argument("--router_aux_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--router_aux_init_checkpoint", type=Path, default=None)
    parser.add_argument("--router_aux_salt_n_pepper_root", type=Path, default=Path("image_sets/salt_n_pepper"))
    parser.add_argument("--router_aux_seed", type=int, default=20260812)
    parser.add_argument("--router_aux_bce_weight", type=float, default=0.25)
    parser.add_argument("--router_aux_rank_weight", type=float, default=0.125)
    parser.add_argument("--router_aux_rank_margin", type=float, default=0.25)
    parser.add_argument("--router_aux_router_lr", type=float, default=5.0e-5)
    parser.add_argument("--router_aux_router_grad_clip", type=float, default=1.0)
    parser.add_argument("--router_aux_imagenet_packets_per_batch", type=int, default=4)
    parser.add_argument("--router_aux_salt_images_per_batch", type=int, default=3)
    parser.add_argument("--router_aux_targets_per_class", type=int, default=4)
    parser.add_argument("--router_aux_min_imagenet_open_fraction", type=float, default=0.5)
    parser.add_argument("--router_aux_expected_salt_n_pepper_images", type=int, default=186)
    parser.add_argument("--router_aux_salt_val_fraction", type=float, default=0.2)
    parser.add_argument("--router_aux_imagenet_handwriting_probability", type=float, default=0.0)
    parser.add_argument("--router_aux_val_batches", type=int, default=8)
    parser.add_argument("--router_aux_validation_seed", type=int, default=20270812)
    parser.add_argument("--router_aux_grad_log_every", type=int, default=100)
    parser.add_argument("--router_aux_prompt_templates", nargs="+", default=list(router_auxiliary.DEFAULT_PROMPT_TEMPLATES))
    parser.add_argument("--router_aux_description_templates", nargs="+", default=list(router_auxiliary.DEFAULT_DESCRIPTION_TEMPLATES))
    parser.add_argument("--router_aux_salt_label_variants", nargs="+", default=list(router_auxiliary.DEFAULT_SALT_LABEL_VARIANTS))
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        help="Save phase_<phase>_epoch_NNN.pt after every epoch.",
    )
    parser.add_argument(
        "--branch_config",
        type=Path,
        default=Path("branch_config.json"),
        help="Authoritative orthographic/source/late topology and early width.",
    )
    parser.add_argument(
        "--read_tap_blocks",
        type=int,
        nargs="+",
        default=None,
        help="Override visual tap blocks after checkpoint loading, e.g. 20 22.",
    )
    parser.add_argument(
        "--ortho_tap_blocks", type=int, nargs="+", default=None,
        help="Early candidate-conditioned orthographic blocks, e.g. 8 12.",
    )
    parser.add_argument(
        "--source_tap_blocks", type=int, nargs="+", default=None,
        help="Early candidate-independent source-gate blocks, e.g. 8 12 13.",
    )
    parser.add_argument(
        "--mini_config_json", type=Path, default=None,
        help="Load ortho/source/late block choices from mini_benchmark_selected.json.",
    )
    parser.add_argument(
        "--reset_tap_logits_uniform",
        action="store_true",
        help="Reset orthographic, source, late-read, and content tap mixtures uniformly.",
    )
    parser.add_argument(
        "--rank1_probe_npz",
        type=Path,
        default=None,
        help="Optional NPZ containing a frozen rank-1 text-injection direction; disabled when omitted.",
    )
    parser.add_argument("--rank1_probe_key", default="mean_dir")
    parser.add_argument("--mini_benchmark", action="store_true")
    parser.add_argument(
        "--mini_fixed_ortho_blocks", type=int, nargs="+", default=(8, 12, 13),
        help="Exact orthographic topology to retain and retrain per mini seed.",
    )
    parser.add_argument(
        "--mini_source_combos", nargs="+",
        default=("8", "12", "13", "8+12", "8+13", "12+13", "8+12+13", "18", "19"),
        help="Explicit source-gate topologies, written as e.g. 12 8+12 18.",
    )
    parser.add_argument("--mini_late_anchor_block", type=int, default=20)
    parser.add_argument(
        "--mini_late_second_blocks", type=int, nargs="+", default=(18, 19, 21, 22),
        help="Second late-read taps compared against the mandatory anchor block.",
    )
    parser.add_argument(
        "--mini_late_allow_content_mixture", action="store_true",
        help=(
            "Let the content-correction lane mix the candidate late taps during mini search. "
            "By default it is fixed to the B20 anchor so the comparison isolates reading."
        ),
    )
    parser.add_argument("--mini_num_seeds", type=int, default=3)
    parser.add_argument("--mini_seed_stride", type=int, default=1)
    parser.add_argument("--mini_fraction", type=float, default=0.10)
    parser.add_argument("--mini_steps_per_config", type=int, default=200)
    parser.add_argument("--mini_val_batches", type=int, default=8)
    parser.add_argument("--mini_batch_images", type=int, default=16)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/clip_xattn_training/final/sigmoid_all/base"))

    parser.add_argument("--imagenet_text_root", type=Path, required=True)
    parser.add_argument("--imagenet_wnid_json", type=Path, required=True)
    parser.add_argument("--imagenet_handwriting_root", type=Path, required=True)
    parser.add_argument("--textcaps_root", type=Path, required=True)
    parser.add_argument("--coco_root", type=Path, required=True)
    parser.add_argument("--coco_train_json", type=Path, required=True)
    parser.add_argument("--coco_train_gpt_json", type=Path, required=True)
    parser.add_argument("--coco_train_reading_json", type=Path, required=True)
    parser.add_argument("--coco_val_json", type=Path, required=True)
    parser.add_argument("--coco_val_gpt_json", type=Path, required=True)
    parser.add_argument(
        "--coco_val_reading_json",
        type=Path,
        default=None,
    )
    parser.add_argument("--clevr_train_root", type=Path, default=None)
    parser.add_argument("--clevr_val_root", type=Path, default=None)
    parser.add_argument("--clevr_metadata_jsonl", type=Path, default=None)


    parser.add_argument("--run_benchmark_validation", action="store_true")
    parser.add_argument("--benchmark_every_epochs", type=int, default=1)
    parser.add_argument("--benchmark_max_items", type=int, default=0)
    parser.add_argument("--benchmark_batch_size", type=int, default=32)
    parser.add_argument("--mvt_benchmark_batch_size", type=int, default=8)
    parser.add_argument("--select_by_benchmark_score", action="store_true")
    parser.add_argument("--benchmark_scam_repo", default="BLISS-e-V/SCAM")
    parser.add_argument("--benchmark_scam_revision", default=None)
    parser.add_argument("--benchmark_rta_repo", default="zer0int/RTA-100-Triplet")
    parser.add_argument("--benchmark_rta_revision", default=None)
    parser.add_argument("--benchmark_include_mvt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mvt_csv", type=Path, default=None)
    parser.add_argument("--mvt_image_root", type=Path, default=None)

    parser.add_argument("--phase_1a_epochs", type=int, default=2)
    parser.add_argument("--phase_1b_epochs", type=int, default=3)
    parser.add_argument("--phase_1b5_epochs", type=int, default=1)
    parser.add_argument("--phase_1c_epochs", type=int, default=3)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--logical_batch_images", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--val_batches", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--handwriting_probability", type=float, default=0.40)
    parser.add_argument("--textcaps_mask_weight", type=float, default=0.25)
    parser.add_argument("--coco_changed_probability", type=float, default=0.60)
    parser.add_argument("--coco_reading_probability", type=float, default=0.25)
    parser.add_argument("--coco_reading_mask_weight", type=float, default=0.50)
    parser.add_argument(
        "--flip_probability", type=float, default=0.0,
        help="Used only for non-text COCO-SPRIGHT images; readable overlays are never mirrored.",
    )
    parser.add_argument(
        "--control_font_paths",
        nargs="*",
        default=(),
        help="Optional fonts for matched pseudoword and mirrored-word controls.",
    )

    # Phase source mix. Handwriting stays inside the ImageNet share.
    parser.add_argument("--mix_1a_coco", type=float, default=0.00)
    parser.add_argument("--mix_1a_textcaps", type=float, default=0.35)
    parser.add_argument("--mix_1a_imagenet", type=float, default=0.65)
    parser.add_argument("--mix_1b_coco", type=float, default=0.15)
    parser.add_argument("--mix_1b_textcaps", type=float, default=0.20)
    parser.add_argument("--mix_1b_imagenet", type=float, default=0.65)
    parser.add_argument("--mix_1c_coco", type=float, default=0.30)
    parser.add_argument("--mix_1c_textcaps", type=float, default=0.25)
    parser.add_argument("--mix_1c_imagenet", type=float, default=0.45)
    parser.add_argument("--mix_1b5_imagenet_math", type=float, default=0.0)
    parser.add_argument("--mix_1c_imagenet_math", type=float, default=0.0)

    parser.add_argument("--lr_new", type=float, default=1.0e-3)
    parser.add_argument("--lr_read", type=float, default=2.0e-4)
    parser.add_argument("--lr_hard_token", type=float, default=5.0e-5)
    parser.add_argument("--lr_content", type=float, default=2.0e-5)
    parser.add_argument("--read_lr_factor_1c", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--mp_weight_1b", type=float, default=0.50)
    parser.add_argument("--mp_weight_1c", type=float, default=1.00)
    parser.add_argument("--source_weight", type=float, default=0.50)
    parser.add_argument("--source_weight_1c", type=float, default=0.25)
    parser.add_argument("--source_margin", type=float, default=1.0)
    parser.add_argument("--ortho_weight", type=float, default=0.50)
    parser.add_argument("--ortho_weight_1c", type=float, default=0.25)
    parser.add_argument("--ortho_margin", type=float, default=1.0)
    parser.add_argument("--mask_weight", type=float, default=0.20)
    parser.add_argument("--mask_weight_1c", type=float, default=0.10)
    parser.add_argument("--presence_weight", type=float, default=0.10)
    parser.add_argument("--presence_weight_1c", type=float, default=0.05)
    parser.add_argument("--invariance_weight", type=float, default=0.50)
    parser.add_argument("--auto_negative_weight", type=float, default=0.50)
    parser.add_argument("--auto_negative_margin", type=float, default=1.0)
    parser.add_argument("--rent_weight", type=float, default=0.01)

    parser.add_argument("--auto_read_scale_start", type=float, default=0.05)
    parser.add_argument("--null_abstain_max", type=float, default=8.0)
    parser.add_argument("--auto_read_scale_max", type=float, default=2.0)
    parser.add_argument("--glyph_beta_max", type=float, default=4.0)
    parser.add_argument("--preflight_tolerance", type=float, default=5.0e-3)
    parser.add_argument("--amp_dtype", choices=AMP_CHOICES, default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--precision_diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision_log_every", type=int, default=100)
    parser.add_argument("--precision_example_values", type=int, default=12)
    parser.add_argument("--precision_parameter_limit", type=int, default=48)
    parser.add_argument("--precision_max_values", type=int, default=262144)
    parser.add_argument("--precision_save_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--export_merged_final", action="store_true")
    parser.add_argument(
        "--save_stage_complete_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the redundant stage1_complete compact checkpoint. Disable when phase best/last PIECES checkpoints are sufficient.",
    )
    parser.add_argument(
        "--export_merged_each_phase",
        action="store_true",
        help=(
            "After restoring each phase's best compact checkpoint, export a complete "
            "loadable model as phase_<phase>_best_merged_state_dict.pt."
        ),
    )
    parser.add_argument("--preflight_only", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
