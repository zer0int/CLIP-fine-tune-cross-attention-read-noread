#!/usr/bin/env python3
"""Materialize the two ImageNet-derived CLIP Cross-Attn MUX corpora.

Prerequisites
-------------
Run prepare_training_data.py first. It installs portable recipe manifests under:

  <data-root>/imagenet_clip_text/manifests/{train,val}.jsonl
  <data-root>/imagenet_clip_text_handwriting/manifests/{train,val}.jsonl
  <data-root>/handwriting_overlays/{manifest.jsonl,images/...}

ImageNet itself is never downloaded by this project. Pass --imagenet-root to a
user-provided ImageNet/ILSVRC2012 tree.

Reproducibility boundary
------------------------
The distributed manifests are authoritative for source-image selection, labels,
relations, overlay text, and recorded geometry. Only the digital rows consumed by
the final trainer are materialized: clean, supportive, and adversarial. Historical
pseudo/mirror rows remain in the manifests for provenance; the final trainer
generates those controls dynamically. Digital readable rows retain bbox, angle,
font size, mirror and box-style metadata and are replayed from those values.
Handwriting rows created in the final expansion retain full render geometry; older
rows marked render_metadata.reused_existing retain the selected handwriting sample
but not their original render geometry. Those rows therefore use a stable,
deterministic geometry fallback derived from uid. This is surfaced in BUILD_INFO,
not hidden.

For maintainers who still have the historical rendered datasets, pass
--reference-digital-root / --reference-handwriting-root to report pixel MAE on a
sample while calibrating the clean-room renderer. Reference roots are read-only and
are never required for ordinary users.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    BICUBIC = Image.Resampling.BICUBIC
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    BICUBIC = Image.BICUBIC
    LANCZOS = Image.LANCZOS

IMAGE_SIZE = 224
JPEG_QUALITY = 95
EXPECTED_DIGITAL = {"train": 60000, "val": 6000}
EXPECTED_HANDWRITING = {"train": 13757, "val": 1307}
MATERIALIZED_DIGITAL_RELATIONS = {"none", "supportive", "adversarial"}
INK_PALETTE = ((0, 0, 0), (18, 24, 38), (20, 42, 70), (35, 24, 22))
COLOR_PALETTE = ((220, 30, 30), (20, 190, 230), (245, 210, 30), (40, 220, 80))


def log(message: str) -> None:
    print(f"[imagenet-derivatives] {message}", flush=True)


def stable_int(text: str, bits: int = 64) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[: bits // 8], "big", signed=False)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def normalize_relpath(value: str) -> str:
    text = str(value).replace("\\", "/")
    # Portable recipes should already be relative. Also tolerate the historical
    # absolute rows when a maintainer points this at an old manifest.
    match = re.search(r"(?:^|/)(train|val)/(.+)$", text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1).lower()}/{match.group(2)}"
    if re.match(r"^[A-Za-z]:/", text) or text.startswith("/"):
        raise ValueError(f"Cannot map absolute ImageNet source path to train/val: {value!r}")
    return text.lstrip("/")


def resolve_imagenet_source(imagenet_root: Path, row: Mapping[str, Any]) -> Path:
    rel = normalize_relpath(str(row.get("source_image_path") or ""))
    if not rel:
        raise ValueError(f"Recipe row {row.get('uid')} has no source_image_path")
    p = PurePosixPath(rel)
    candidates = [imagenet_root / p]
    parts = p.parts
    if parts and parts[0] == "train":
        tail = Path(*parts[1:])
        candidates.extend([
            imagenet_root / "ILSVRC2012_img_train" / tail,
            imagenet_root / tail,
        ])
    elif parts and parts[0] == "val":
        tail = Path(*parts[1:])
        basename = Path(*parts).name
        candidates.extend([
            imagenet_root / "ILSVRC2012_img_val" / tail,
            imagenet_root / "val" / basename,
            imagenet_root / "ILSVRC2012_img_val" / basename,
            imagenet_root / tail,
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve ImageNet source for {row.get('uid')}: recipe={rel!r}; "
        f"tried={[str(x) for x in candidates]}"
    )


def resize_short_side_center_crop(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size {image.size}")
    scale = float(size) / float(min(w, h))
    nw = max(size, int(round(w * scale)))
    nh = max(size, int(round(h * scale)))
    image = image.resize((nw, nh), BICUBIC)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return image.crop((left, top, left + size, top + size))


def atomic_save_jpeg(image: Image.Image, path: Path, quality: int = JPEG_QUALITY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    image.convert("RGB").save(tmp, format="JPEG", quality=int(quality), subsampling=0, optimize=False)
    os.replace(tmp, path)


def atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    image.save(tmp, format="PNG", optimize=False)
    os.replace(tmp, path)


def load_font(font_path: Optional[Path], size: int) -> ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(str(font_path), size=max(8, int(size)))
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=max(8, int(size)))
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=max(8, int(size)))
    except TypeError:
        return ImageFont.load_default()


def local_luminance(image: Image.Image, bbox: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    crop = image.convert("L").crop((max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)))
    arr = np.asarray(crop, dtype=np.float32)
    return float(arr.mean()) if arr.size else 127.5


def choose_digital_style(base: Image.Image, bbox: Tuple[int, int, int, int], uid: str) -> Dict[str, Any]:
    rng = random.Random(stable_int(f"digital-style|{uid}"))
    lum = local_luminance(base, bbox)
    if rng.random() < 0.18:
        fill = rng.choice(COLOR_PALETTE)
        stroke = (0, 0, 0) if sum(fill) > 360 else (255, 255, 255)
    elif lum > 128:
        fill, stroke = (5, 5, 5), (255, 255, 255)
    else:
        fill, stroke = (250, 250, 250), (0, 0, 0)
    return {"fill": fill, "stroke": stroke, "stroke_width": rng.choice((0, 1, 1, 2))}


def _fit_text_font(text: str, requested: int, font_path: Optional[Path], max_w: int, max_h: int, stroke: int) -> ImageFont.ImageFont:
    size = max(8, int(requested))
    probe = ImageDraw.Draw(Image.new("L", (max(8, max_w * 2), max(8, max_h * 2)), 0))
    while size >= 8:
        font = load_font(font_path, size)
        box = probe.textbbox((0, 0), text or " ", font=font, stroke_width=stroke)
        if max(1, box[2] - box[0]) <= max_w and max(1, box[3] - box[1]) <= max_h:
            return font
        size -= 1
    return load_font(font_path, 8)


def render_digital_row(
    base: Image.Image,
    row: Mapping[str, Any],
    font_path: Optional[Path],
) -> Tuple[Image.Image, Optional[Image.Image]]:
    relation = str(row.get("relation") or "")
    if relation == "none" or str(row.get("variant")) == "clean":
        return base.copy(), None

    text = str(row.get("overlay_text") or "")
    meta = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    text_meta = meta.get("text") if isinstance(meta, Mapping) and isinstance(meta.get("text"), Mapping) else {}
    bbox_value = text_meta.get("bbox_xyxy")
    if not text or not isinstance(bbox_value, Sequence) or len(bbox_value) != 4:
        raise ValueError(f"Digital recipe row lacks text/bbox: {row.get('uid')}")
    x0, y0, x1, y1 = (int(round(float(x))) for x in bbox_value)
    x0 = max(0, min(base.width - 1, x0)); y0 = max(0, min(base.height - 1, y0))
    x1 = max(x0 + 1, min(base.width, x1)); y1 = max(y0 + 1, min(base.height, y1))
    target_w, target_h = x1 - x0, y1 - y0
    uid = str(row.get("uid") or "")
    style = choose_digital_style(base, (x0, y0, x1, y1), uid)
    requested = int(text_meta.get("font_size") or max(12, target_h))
    angle = float(text_meta.get("angle_deg") or 0.0)
    mirrored = bool(text_meta.get("mirrored", False))
    style_box = bool(text_meta.get("style_box", False))

    font = _fit_text_font(text, requested, font_path, max(8, int(0.95 * IMAGE_SIZE)), max(8, int(0.30 * IMAGE_SIZE)), int(style["stroke_width"]))
    probe = ImageDraw.Draw(Image.new("L", (8, 8), 0))
    tb = probe.textbbox((0, 0), text, font=font, stroke_width=int(style["stroke_width"]))
    tw, th = max(1, tb[2] - tb[0]), max(1, tb[3] - tb[1])
    pad = max(5, int(style["stroke_width"]) + 4)
    tile = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    glyph = Image.new("L", tile.size, 0)
    draw = ImageDraw.Draw(tile); draw_mask = ImageDraw.Draw(glyph)
    if style_box:
        # This plate is generated content; glyph supervision remains glyph-only.
        lum = local_luminance(base, (x0, y0, x1, y1))
        plate = (255, 255, 255, 205) if lum < 128 else (0, 0, 0, 185)
        draw.rounded_rectangle((0, 0, tile.width - 1, tile.height - 1), radius=max(3, pad), fill=plate)
    draw.text(
        (pad - tb[0], pad - tb[1]), text, font=font,
        fill=tuple(style["fill"]) + (255,), stroke_width=int(style["stroke_width"]),
        stroke_fill=tuple(style["stroke"]) + (255,),
    )
    draw_mask.text(
        (pad - tb[0], pad - tb[1]), text, font=font, fill=255,
        stroke_width=int(style["stroke_width"]),
    )
    if mirrored:
        tile = ImageOps.mirror(tile); glyph = ImageOps.mirror(glyph)
    if abs(angle) > 1e-6:
        tile = tile.rotate(angle, resample=BICUBIC, expand=True)
        glyph = glyph.rotate(angle, resample=BICUBIC, expand=True)
    # The historical manifest stores the realized support bbox. Force the replay
    # into that exact support so downstream patch-mask geometry is preserved.
    tile = tile.resize((target_w, target_h), LANCZOS)
    glyph = glyph.resize((target_w, target_h), LANCZOS)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0)); layer.alpha_composite(tile, (x0, y0))
    rendered = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    mask = Image.new("L", base.size, 0); mask.paste(glyph, (x0, y0))
    return rendered, mask


@dataclass(frozen=True)
class HandwritingSample:
    sample_id: str
    path: Path
    bbox: Optional[Tuple[int, int, int, int]]


def parse_manifest_bbox(value: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(value, Mapping):
        return None
    try:
        return (
            int(round(float(value["x_min"]))), int(round(float(value["y_min"]))),
            int(round(float(value["x_max"]))), int(round(float(value["y_max"]))),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_handwriting_samples(root: Path) -> Dict[str, HandwritingSample]:
    manifest = root / "manifest.jsonl"
    rows = read_jsonl(manifest)
    samples: Dict[str, HandwritingSample] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        recorded = str(row.get("png_path") or row.get("image") or "").strip()
        basename = PureWindowsPath(recorded).name if recorded else f"{sample_id}.png"
        candidates = [root / "images" / basename, root / basename]
        path = next((p for p in candidates if p.is_file()), None)
        if sample_id and path is not None:
            samples[sample_id] = HandwritingSample(sample_id, path, parse_manifest_bbox(row.get("bbox")))
    if not samples:
        raise RuntimeError(f"No handwriting samples resolved from {manifest}")
    return samples


def derive_alpha(image: Image.Image) -> Image.Image:
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


def load_handwriting_alpha(sample: HandwritingSample) -> Image.Image:
    with Image.open(sample.path) as source:
        rgba = source.convert("RGBA")
    alpha = derive_alpha(rgba)
    if sample.bbox is not None:
        x0, y0, x1, y1 = sample.bbox
        x0 = max(0, min(rgba.width, x0)); y0 = max(0, min(rgba.height, y0))
        x1 = max(x0 + 1, min(rgba.width, x1)); y1 = max(y0 + 1, min(rgba.height, y1))
        alpha = alpha.crop((x0, y0, x1, y1))
    tight = alpha.getbbox()
    if tight is not None:
        alpha = alpha.crop(tight)
    return alpha


def fallback_handwriting_geometry(uid: str, alpha: Image.Image, size: int = IMAGE_SIZE) -> Dict[str, Any]:
    rng = random.Random(stable_int(f"handwriting-fallback|{uid}"))
    angle = rng.uniform(-16.0, 16.0)
    opacity = rng.uniform(0.82, 1.0)
    ink = rng.choice(INK_PALETTE)
    # Match the historical support scale: broad enough to be readable but not a
    # full-canvas watermark. Preserve glyph aspect ratio before rotation.
    max_w = int(round(size * rng.uniform(0.30, 0.66)))
    max_h = int(round(size * rng.uniform(0.12, 0.30)))
    ratio = alpha.width / max(1.0, float(alpha.height))
    w = min(max_w, max(12, int(round(max_h * ratio))))
    h = min(max_h, max(10, int(round(w / max(ratio, 1e-6)))))
    centers = ((0.50, 0.17), (0.50, 0.50), (0.50, 0.83), (0.25, 0.50), (0.75, 0.50))
    cx, cy = rng.choice(centers)
    cx = float(np.clip(cx + rng.uniform(-0.08, 0.08), 0.13, 0.87))
    cy = float(np.clip(cy + rng.uniform(-0.06, 0.06), 0.10, 0.90))
    x0 = int(round(cx * size - w / 2)); y0 = int(round(cy * size - h / 2))
    x0 = max(0, min(size - w, x0)); y0 = max(0, min(size - h, y0))
    return {"bbox_xyxy": [x0, y0, x0 + w, y0 + h], "angle_deg": angle, "opacity": opacity, "ink_rgb": list(ink)}


def render_handwriting_row(
    clean: Image.Image,
    row: Mapping[str, Any],
    samples: Mapping[str, HandwritingSample],
) -> Tuple[Image.Image, Image.Image, bool]:
    sample_id = str(row.get("handwriting_sample_id") or "")
    if sample_id not in samples:
        raise FileNotFoundError(f"Handwriting sample {sample_id!r} required by {row.get('uid')} is unavailable")
    alpha = load_handwriting_alpha(samples[sample_id])
    render_meta = dict(row.get("render_metadata") or {}) if isinstance(row.get("render_metadata"), Mapping) else {}
    reused = bool(render_meta.get("reused_existing", False)) or "bbox_xyxy" not in render_meta
    if reused:
        geometry = fallback_handwriting_geometry(str(row.get("uid") or sample_id), alpha)
    else:
        geometry = render_meta
    bbox = geometry.get("bbox_xyxy")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        raise ValueError(f"Handwriting geometry has no bbox for {row.get('uid')}")
    x0, y0, x1, y1 = (int(round(float(x))) for x in bbox)
    x0 = max(0, min(clean.width - 1, x0)); y0 = max(0, min(clean.height - 1, y0))
    x1 = max(x0 + 1, min(clean.width, x1)); y1 = max(y0 + 1, min(clean.height, y1))
    target_w, target_h = x1 - x0, y1 - y0
    angle = float(geometry.get("angle_deg") or 0.0)
    opacity = float(np.clip(float(geometry.get("opacity", 1.0)), 0.0, 1.0))
    ink_value = geometry.get("ink_rgb") or (0, 0, 0)
    ink = tuple(int(np.clip(int(v), 0, 255)) for v in ink_value[:3])

    # Rotate native glyph alpha first, then fit it into the realized support bbox.
    if abs(angle) > 1e-6:
        alpha = alpha.rotate(angle, resample=BICUBIC, expand=True)
    alpha = alpha.resize((target_w, target_h), LANCZOS)
    if opacity < 1.0:
        arr = np.asarray(alpha, dtype=np.float32) * opacity
        alpha = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    tile = Image.new("RGBA", (target_w, target_h), ink + (255,))
    tile.putalpha(alpha)
    layer = Image.new("RGBA", clean.size, (0, 0, 0, 0)); layer.alpha_composite(tile, (x0, y0))
    rendered = Image.alpha_composite(clean.convert("RGBA"), layer).convert("RGB")
    mask = Image.new("L", clean.size, 0); mask.paste(alpha, (x0, y0))
    return rendered, mask, reused


class ReferenceComparator:
    def __init__(self, digital_root: Optional[Path], handwriting_root: Optional[Path], limit: int):
        self.roots = {"digital": digital_root, "handwriting": handwriting_root}
        self.limit = max(0, int(limit))
        self.stats: Dict[str, List[float]] = defaultdict(list)
        self.exact: Counter[str] = Counter()

    def compare(self, kind: str, rel: str, generated_path: Path) -> None:
        root = self.roots.get(kind)
        if root is None or self.limit <= 0 or len(self.stats[kind]) >= self.limit:
            return
        ref = root / PurePosixPath(rel)
        if not ref.is_file():
            return
        with Image.open(generated_path) as a, Image.open(ref) as b:
            aa = np.asarray(a.convert("RGB"), dtype=np.float32)
            bb = np.asarray(b.convert("RGB").resize(a.size, BICUBIC), dtype=np.float32)
        mae = float(np.mean(np.abs(aa - bb)))
        self.stats[kind].append(mae)
        if mae == 0.0:
            self.exact[kind] += 1

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for kind, values in self.stats.items():
            if values:
                out[kind] = {
                    "samples": len(values),
                    "pixel_mae_mean_0_255": float(statistics.mean(values)),
                    "pixel_mae_median_0_255": float(statistics.median(values)),
                    "pixel_exact_samples": int(self.exact[kind]),
                }
        return out


def select_groups(rows: Sequence[Mapping[str, Any]], max_groups: int) -> List[Mapping[str, Any]]:
    if max_groups <= 0:
        return list(rows)
    keep: List[str] = []
    seen: set[str] = set()
    for row in rows:
        gid = str(row.get("group_id"))
        if gid not in seen:
            seen.add(gid); keep.append(gid)
            if len(keep) >= max_groups:
                break
    allowed = set(keep)
    return [row for row in rows if str(row.get("group_id")) in allowed]


def build_digital_split(
    *, rows: Sequence[Mapping[str, Any]], split: str, root: Path, imagenet_root: Path,
    font_path: Optional[Path], overwrite: bool, comparator: ReferenceComparator,
) -> Dict[str, int]:
    by_group: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    counts = Counter()
    for group_index, (gid, grows) in enumerate(by_group.items(), start=1):
        clean_rows = [r for r in grows if str(r.get("relation")) == "none"]
        if len(clean_rows) != 1:
            raise RuntimeError(f"Digital group {gid} expected exactly one clean row, got {len(clean_rows)}")
        clean_row = clean_rows[0]
        source_path = resolve_imagenet_source(imagenet_root, clean_row)
        with Image.open(source_path) as source:
            base = resize_short_side_center_crop(source, IMAGE_SIZE)
        for row in grows:
            relation = str(row.get("relation") or "")
            if relation not in MATERIALIZED_DIGITAL_RELATIONS:
                counts[f"skipped_{row.get('variant')}"] += 1
                continue
            rel = str(row["image_relpath"])
            out = root / PurePosixPath(rel)
            mask_rel = row.get("mask_relpath")
            mask_out = root / PurePosixPath(str(mask_rel)) if mask_rel else None
            if overwrite or not out.is_file() or (mask_out is not None and not mask_out.is_file()):
                rendered, mask = render_digital_row(base, row, font_path)
                atomic_save_jpeg(rendered, out, JPEG_QUALITY)
                if mask_out is not None:
                    if mask is None:
                        raise RuntimeError(f"Recipe requires mask but renderer returned none: {row.get('uid')}")
                    atomic_save_png(mask, mask_out)
            comparator.compare("digital", rel, out)
            counts[str(row.get("variant"))] += 1
        if group_index % 500 == 0:
            log(f"digital {split}: {group_index}/{len(by_group)} groups")
    return dict(counts)


def build_handwriting_split(
    *, rows: Sequence[Mapping[str, Any]], split: str, root: Path, digital_root: Path,
    samples: Mapping[str, HandwritingSample], overwrite: bool, comparator: ReferenceComparator,
) -> Dict[str, int]:
    counts = Counter(); reused_count = 0
    for index, row in enumerate(rows, start=1):
        rel = str(row["image"]); mask_rel = str(row["mask"])
        out = root / PurePosixPath(rel); mask_out = root / PurePosixPath(mask_rel)
        clean_rel = str(row.get("source_clean_image") or "").replace("\\", "/")
        # Sanitized recipe rows store source_clean_image relative to the digital root.
        marker = "/images/"
        if marker in clean_rel:
            clean_rel = "images/" + clean_rel.split(marker, 1)[1]
        elif clean_rel.startswith("images/"):
            pass
        else:
            clean_rel = f"images/{split}/{row['group_id']}__clean__s0.jpg"
        clean_path = digital_root / PurePosixPath(clean_rel)
        if not clean_path.is_file():
            raise FileNotFoundError(f"Digital clean image required for handwriting row is missing: {clean_path}")
        if overwrite or not out.is_file() or not mask_out.is_file():
            with Image.open(clean_path) as clean_source:
                clean = clean_source.convert("RGB")
            rendered, mask, reused = render_handwriting_row(clean, row, samples)
            atomic_save_jpeg(rendered, out, JPEG_QUALITY)
            atomic_save_png(mask, mask_out)
        else:
            reused = bool((row.get("render_metadata") or {}).get("reused_existing", False))
        reused_count += int(reused)
        comparator.compare("handwriting", rel, out)
        counts[str(row.get("variant"))] += 1
        if index % 1000 == 0:
            log(f"handwriting {split}: {index}/{len(rows)} rows")
    counts["fallback_geometry"] = reused_count
    return dict(counts)


def verify_outputs(root: Path, rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]], handwriting: bool) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for split, rows in rows_by_split.items():
        missing: List[str] = []
        for row in rows:
            if not handwriting and str(row.get("relation") or "") not in MATERIALIZED_DIGITAL_RELATIONS:
                continue
            image_rel = str(row["image"] if handwriting else row["image_relpath"])
            mask_rel = row.get("mask") if handwriting else row.get("mask_relpath")
            if not (root / PurePosixPath(image_rel)).is_file():
                missing.append(image_rel)
            if mask_rel and not (root / PurePosixPath(str(mask_rel))).is_file():
                missing.append(str(mask_rel))
            if len(missing) >= 10:
                break
        if missing:
            raise RuntimeError(f"{root.name}/{split} missing outputs; examples={missing}")
        result[split] = len(rows)
    return result


def patch_ready_state(data_root: Path, config_path: Optional[Path], full_build: bool, build_info: Mapping[str, Any]) -> None:
    if config_path is not None and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config, MutableMapping) and isinstance(config.get("paths"), MutableMapping):
            config["paths"]["imagenet_text_root"] = str((data_root / "imagenet_clip_text").resolve())
            config["paths"]["imagenet_handwriting_root"] = str((data_root / "imagenet_clip_text_handwriting").resolve())
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    state = data_root / "PREPARE_STATE.json"
    if state.is_file():
        payload = json.loads(state.read_text(encoding="utf-8"))
        if isinstance(payload, MutableMapping):
            payload["imagenet_derivatives_ready"] = bool(full_build)
            payload["imagenet_derivative_build"] = dict(build_info)
            write_json(state, payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/clip_cross_attn_mux"))
    p.add_argument("--imagenet-root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--config", type=Path, default=Path("training_config.local.json"))
    p.add_argument("--font-path", type=Path, default=None, help="Optional TTF/OTF used for digital overlays")
    p.add_argument("--overwrite", action="store_true", help="Regenerate outputs even when files already exist")
    p.add_argument("--max-groups-per-split", type=int, default=0, help="Calibration/smoke mode; 0 builds all groups")
    p.add_argument("--reference-digital-root", type=Path, default=None, help="Optional historical derivative root for pixel comparison")
    p.add_argument("--reference-handwriting-root", type=Path, default=None, help="Optional historical handwriting derivative root for pixel comparison")
    p.add_argument("--verify-reference-samples", type=int, default=32)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser()
    data_root = (project_root / data_root).resolve() if not data_root.is_absolute() else data_root.resolve()
    imagenet_root = args.imagenet_root.expanduser().resolve()
    config_path = args.config.expanduser()
    config_path = (project_root / config_path).resolve() if not config_path.is_absolute() else config_path.resolve()
    font_path = args.font_path.expanduser().resolve() if args.font_path else None
    if font_path is not None and not font_path.is_file():
        raise FileNotFoundError(f"--font-path does not exist: {font_path}")
    if not imagenet_root.is_dir():
        raise FileNotFoundError(f"ImageNet root does not exist: {imagenet_root}")

    digital_root = data_root / "imagenet_clip_text"
    handwriting_root = data_root / "imagenet_clip_text_handwriting"
    overlay_root = data_root / "handwriting_overlays"
    digital_rows: Dict[str, List[Dict[str, Any]]] = {}
    handwriting_rows: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val"):
        dp = digital_root / "manifests" / f"{split}.jsonl"
        hp = handwriting_root / "manifests" / f"{split}.jsonl"
        if not dp.is_file() or not hp.is_file():
            raise FileNotFoundError(
                f"ImageNet recipes are missing under {data_root}. Run prepare_training_data.py first."
            )
        digital_rows[split] = read_jsonl(dp)
        handwriting_rows[split] = read_jsonl(hp)
        if len(digital_rows[split]) != EXPECTED_DIGITAL[split]:
            raise RuntimeError(f"Digital {split}: expected {EXPECTED_DIGITAL[split]} recipe rows, found {len(digital_rows[split])}")
        if len(handwriting_rows[split]) != EXPECTED_HANDWRITING[split]:
            raise RuntimeError(f"Handwriting {split}: expected {EXPECTED_HANDWRITING[split]} recipe rows, found {len(handwriting_rows[split])}")

    max_groups = max(0, int(args.max_groups_per_split))
    selected_digital = {s: select_groups(rows, max_groups) for s, rows in digital_rows.items()}
    selected_handwriting: Dict[str, List[Dict[str, Any]]] = {}
    if max_groups > 0:
        for split, rows in handwriting_rows.items():
            allowed = {str(r["group_id"]) for r in selected_digital[split]}
            selected_handwriting[split] = [r for r in rows if str(r["group_id"]) in allowed]
    else:
        selected_handwriting = handwriting_rows

    ref_d = args.reference_digital_root.expanduser().resolve() if args.reference_digital_root else None
    ref_h = args.reference_handwriting_root.expanduser().resolve() if args.reference_handwriting_root else None
    comparator = ReferenceComparator(ref_d, ref_h, int(args.verify_reference_samples))
    samples = load_handwriting_samples(overlay_root)
    log(f"resolved {len(samples)} handwriting overlay samples")
    log(f"digital recipe rows: train={len(selected_digital['train'])}, val={len(selected_digital['val'])}")
    log(f"handwriting recipe rows: train={len(selected_handwriting['train'])}, val={len(selected_handwriting['val'])}")

    digital_counts: Dict[str, Any] = {}
    handwriting_counts: Dict[str, Any] = {}
    for split in ("train", "val"):
        log(f"building digital {split}...")
        digital_counts[split] = build_digital_split(
            rows=selected_digital[split], split=split, root=digital_root, imagenet_root=imagenet_root,
            font_path=font_path, overwrite=bool(args.overwrite), comparator=comparator,
        )
    for split in ("train", "val"):
        log(f"building handwriting {split}...")
        handwriting_counts[split] = build_handwriting_split(
            rows=selected_handwriting[split], split=split, root=handwriting_root,
            digital_root=digital_root, samples=samples, overwrite=bool(args.overwrite), comparator=comparator,
        )

    verify_outputs(digital_root, {s: selected_digital[s] for s in ("train", "val")}, handwriting=False)
    verify_outputs(handwriting_root, {s: selected_handwriting[s] for s in ("train", "val")}, handwriting=True)
    full_build = max_groups == 0
    comparison = comparator.summary()
    build_info = {
        "schema_version": 1,
        "renderer": "clip-cross-attn-mux-recipe-replay-v1",
        "image_size": IMAGE_SIZE,
        "jpeg_quality": JPEG_QUALITY,
        "full_build": full_build,
        "digital_recipe_rows": {s: len(selected_digital[s]) for s in ("train", "val")},
        "handwriting_recipe_rows": {s: len(selected_handwriting[s]) for s in ("train", "val")},
        "digital_variant_counts": digital_counts,
        "handwriting_variant_counts": handwriting_counts,
        "handwriting_overlay_samples": len(samples),
        "historical_replay_limitations": {
            "reused_handwriting_geometry_missing": True,
            "explanation": (
                "Rows whose render_metadata contains reused_existing retained the selected handwriting sample "
                "but not the original bbox/angle/opacity/ink parameters; those rows use deterministic fallback geometry."
            ),
        },
        "reference_comparison": comparison,
    }
    write_json(digital_root / "BUILD_INFO.json", build_info)
    write_json(handwriting_root / "BUILD_INFO.json", build_info)
    patch_ready_state(data_root, config_path if config_path.is_file() else None, full_build, build_info)

    if comparison:
        for kind, stats in comparison.items():
            log(
                f"reference {kind}: n={stats['samples']} median pixel MAE={stats['pixel_mae_median_0_255']:.4f}/255 "
                f"exact={stats['pixel_exact_samples']}"
            )
    if full_build:
        log("ImageNet derivatives complete and verified; PREPARE_STATE.json marked ready")
        log(f"training config: {config_path}")
    else:
        log("calibration/smoke build complete; PREPARE_STATE.json remains not-ready because only a subset was materialized")
    log("complete; this command will now exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
