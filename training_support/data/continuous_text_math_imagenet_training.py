from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .tiny_patch_imagenet_training import (
    BICUBIC,
    build_short_concept_taxonomy,
    crop_like_training,
    draw_patch_grid,
    occupied_patch_coords,
    parse_group_wnid,
    read_jsonl,
    stable_seed,
)


@dataclass
class RenderedText:
    image: Image.Image
    mask: Image.Image
    word: str
    font_name: str
    font_size: int
    stroke_width: int
    rotation_deg: float
    opacity: float
    crop_visible_fraction: float
    occupied_patches: List[Tuple[int, int]]
    ink_bbox: Tuple[int, int, int, int]
    regime: str


@dataclass
class ContinuousMathPacketData:
    images: List[Image.Image]
    masks: List[Image.Image]
    mask_weights: List[float]
    present_targets: List[float]
    readable_targets: List[float]
    captions: List[str]
    positive_pairs: List[Tuple[int, int]]
    source_triplets: List[Tuple[int, int, List[int]]]
    auto_triplets: List[Tuple[int, int, int]]
    invariance_pairs: List[Tuple[int, int]]
    metadata: Dict[str, Any]


def _taxonomy_membership(wnid: str, taxonomy: Mapping[str, Any]) -> List[str]:
    return sorted(word for word, spec in taxonomy.items() if wnid in spec.wnids)


def _local_luminance(image: Image.Image, box: Tuple[int, int, int, int]) -> float:
    arr = np.asarray(image.crop(box).convert("RGB"), dtype=np.float32) / 255.0
    if arr.size == 0:
        return 0.5
    return float((0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).mean())


def _font_mask(word: str, font_path: str, font_size: int, stroke_width: int) -> Image.Image:
    font = ImageFont.truetype(font_path, int(font_size))
    draw = ImageDraw.Draw(Image.new("L", (8, 8), 0))
    bbox = draw.textbbox((0, 0), word, font=font, stroke_width=int(stroke_width))
    w = max(1, int(bbox[2] - bbox[0]))
    h = max(1, int(bbox[3] - bbox[1]))
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.text(
        (-bbox[0], -bbox[1]),
        word,
        font=font,
        fill=255,
        stroke_width=int(stroke_width),
        stroke_fill=255,
    )
    tight = mask.getbbox()
    if tight is None:
        raise RuntimeError(f"Font rendered empty text for {word!r}")
    return mask.crop(tight)


def _choose_text_regime(rng: random.Random, crop_probability: float) -> str:
    if rng.random() < float(crop_probability):
        return "huge_cropped"
    # Deliberately dense coverage of both the previously broken microscopic
    # regime and the opposite giant-but-fitting extreme, while remaining
    # continuous within each interval.
    return rng.choices(
        ["micro", "small", "medium", "large", "huge_fit"],
        weights=[0.20, 0.20, 0.25, 0.20, 0.15],
        k=1,
    )[0]


def _sample_font_size(regime: str, rng: random.Random) -> int:
    ranges = {
        "micro": (5, 10),
        "small": (10, 24),
        "medium": (24, 64),
        "large": (64, 120),
        "huge_fit": (120, 230),
        "huge_cropped": (120, 300),
    }
    lo, hi = ranges[regime]
    # Log-uniform avoids massively overrepresenting the upper edge in pixel area.
    value = math.exp(rng.uniform(math.log(float(lo)), math.log(float(hi))))
    return max(lo, min(hi, int(round(value))))


def _fit_mask_to_box(
    word: str,
    font_path: str,
    desired_size: int,
    stroke_width: int,
    max_w: int,
    max_h: int,
) -> Tuple[Image.Image, int]:
    size = max(4, int(desired_size))
    for _ in range(20):
        mask = _font_mask(word, font_path, size, stroke_width)
        if mask.width <= max_w and mask.height <= max_h:
            return mask, size
        scale = min(max_w / max(1, mask.width), max_h / max(1, mask.height))
        new_size = max(4, int(math.floor(size * max(0.25, scale) * 0.98)))
        if new_size >= size:
            new_size = size - 1
        size = max(4, new_size)
    mask = _font_mask(word, font_path, size, stroke_width)
    if mask.width > max_w or mask.height > max_h:
        raise RuntimeError(f"Could not fit {word!r} into {max_w}x{max_h}")
    return mask, size


def _paste_mask_with_contrast(
    base: Image.Image,
    mask: Image.Image,
    x: int,
    y: int,
    rng: random.Random,
    opacity: float,
) -> Image.Image:
    result = base.convert("RGB").copy()
    visible_box = (
        max(0, x), max(0, y), min(result.width, x + mask.width), min(result.height, y + mask.height)
    )
    lum = _local_luminance(result, visible_box) if visible_box[2] > visible_box[0] and visible_box[3] > visible_box[1] else 0.5
    # Opposite-polarity text, but with continuously varying contrast/opacity.
    if lum >= 0.5:
        fg = tuple(rng.randint(0, 55) for _ in range(3))
    else:
        fg = tuple(rng.randint(200, 255) for _ in range(3))
    alpha = np.asarray(mask, dtype=np.float32) * float(opacity)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L")
    patch = Image.new("RGB", mask.size, fg)
    # PIL paste handles negative offsets and clips to the canvas.
    result.paste(patch, (int(x), int(y)), alpha_img)
    return result


def render_continuous_word(
    base: Image.Image,
    word: str,
    patch_size: int,
    font_paths: Sequence[str],
    rng: random.Random,
    crop_probability: float,
    preferred_region: Optional[Tuple[int, int, int, int]] = None,
    force_regime: Optional[str] = None,
) -> RenderedText:
    if not font_paths:
        raise RuntimeError("Continuous text curriculum requires at least one TrueType font")
    word_render = str(word).upper()
    regime = force_regime or _choose_text_regime(rng, crop_probability)
    font_path = rng.choice(list(font_paths))
    desired_size = _sample_font_size(regime, rng)
    stroke_width = rng.choices([0, 1, 2], weights=[0.55, 0.35, 0.10], k=1)[0]
    if desired_size <= 10:
        stroke_width = min(stroke_width, 1)
    rotation = rng.uniform(-18.0, 18.0) if rng.random() < 0.25 else 0.0
    opacity = rng.uniform(0.45, 1.0)

    W, H = base.size
    if regime != "huge_cropped":
        if preferred_region is None:
            rx, ry, rw, rh = 0, 0, W, H
        else:
            rx, ry, rw, rh = preferred_region
        safety = 1 if desired_size < 16 else 3
        max_w = max(4, rw - 2 * safety)
        max_h = max(4, rh - 2 * safety)
        mask, font_size = _fit_mask_to_box(
            word_render, font_path, desired_size, stroke_width, max_w, max_h
        )
        if abs(rotation) > 1.0e-6:
            mask = mask.rotate(rotation, resample=BICUBIC, expand=True)
            tight = mask.getbbox()
            if tight is not None:
                mask = mask.crop(tight)
            if mask.width > max_w or mask.height > max_h:
                scale = min(max_w / max(1, mask.width), max_h / max(1, mask.height))
                new_w = max(1, int(mask.width * scale))
                new_h = max(1, int(mask.height * scale))
                mask = mask.resize((new_w, new_h), BICUBIC)
        x = rng.randint(rx + safety, max(rx + safety, rx + rw - mask.width - safety))
        y = rng.randint(ry + safety, max(ry + safety, ry + rh - mask.height - safety))
        full_mask = Image.new("L", base.size, 0)
        full_mask.paste(mask, (x, y), mask)
        visible_fraction = 1.0
    else:
        # Recoverable giant crop: retain 70-95% of ink. This is deliberately
        # trained as literal-positive because CLIP genuinely completes cropped
        # words in the wild. Extreme fragments are not generated here.
        mask, font_size = _fit_mask_to_box(
            word_render,
            font_path,
            desired_size,
            stroke_width,
            max(W, int(round(W / 0.78))),
            max(H, int(round(H / 0.78))),
        )
        if abs(rotation) > 1.0e-6:
            mask = mask.rotate(rotation, resample=BICUBIC, expand=True)
            tight = mask.getbbox()
            if tight is not None:
                mask = mask.crop(tight)
        total_ink = float(np.asarray(mask, dtype=np.float32).sum()) + 1.0e-6
        accepted = None
        for _ in range(64):
            side = rng.choice(["left", "right", "top", "bottom"])
            crop_frac = rng.uniform(0.05, 0.30)
            if side == "left":
                x = -int(round(mask.width * crop_frac))
                y = rng.randint(-max(0, mask.height // 10), max(0, H - int(mask.height * 0.75)))
            elif side == "right":
                x = W - mask.width + int(round(mask.width * crop_frac))
                y = rng.randint(-max(0, mask.height // 10), max(0, H - int(mask.height * 0.75)))
            elif side == "top":
                y = -int(round(mask.height * crop_frac))
                x = rng.randint(-max(0, mask.width // 10), max(0, W - int(mask.width * 0.75)))
            else:
                y = H - mask.height + int(round(mask.height * crop_frac))
                x = rng.randint(-max(0, mask.width // 10), max(0, W - int(mask.width * 0.75)))
            full_mask = Image.new("L", base.size, 0)
            full_mask.paste(mask, (int(x), int(y)), mask)
            visible = float(np.asarray(full_mask, dtype=np.float32).sum()) / total_ink
            if 0.70 <= visible <= 0.95:
                accepted = (int(x), int(y), full_mask, visible)
                break
        if accepted is None:
            # Deterministic mild edge crop fallback. Keep as much as possible
            # in-frame so this branch remains recoverable literal supervision.
            if mask.width <= W:
                x = -max(1, int(mask.width * 0.05))
            else:
                x = -max(0, (mask.width - W) // 2)
            if mask.height <= H:
                y = max(0, (H - mask.height) // 2)
            else:
                y = -max(0, (mask.height - H) // 2)
            full_mask = Image.new("L", base.size, 0)
            full_mask.paste(mask, (x, y), mask)
            visible_fraction = float(np.asarray(full_mask, dtype=np.float32).sum()) / total_ink
        else:
            x, y, full_mask, visible_fraction = accepted

    result = _paste_mask_with_contrast(base, mask, int(x), int(y), rng, opacity)
    bbox = full_mask.getbbox()
    if bbox is None:
        raise RuntimeError("Rendered word has no visible ink")
    coords = occupied_patch_coords(full_mask, patch_size)
    return RenderedText(
        image=result,
        mask=full_mask,
        word=word,
        font_name=Path(font_path).name,
        font_size=int(font_size),
        stroke_width=int(stroke_width),
        rotation_deg=float(rotation),
        opacity=float(opacity),
        crop_visible_fraction=float(visible_fraction),
        occupied_patches=coords,
        ink_bbox=tuple(int(v) for v in bbox),
        regime=regime,
    )


# -----------------------------------------------------------------------------
# Pure mathematical hard-negative generators. None uses letter/symbol templates.
# -----------------------------------------------------------------------------


def _mesh(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx, yy


def _norm01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    return np.zeros_like(arr) if hi - lo < 1.0e-8 else (arr - lo) / (hi - lo)


def _palette(rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([rng.uniform(0.05, 0.95) for _ in range(3)], dtype=np.float32)
    b = np.array([rng.uniform(0.05, 0.95) for _ in range(3)], dtype=np.float32)
    if float(np.abs(a - b).mean()) < 0.18:
        b = np.clip(1.0 - a + rng.uniform(-0.08, 0.08), 0.0, 1.0)
    return a, b


def _colorize(field: np.ndarray, rng: random.Random) -> np.ndarray:
    f = _norm01(field)[..., None]
    a, b = _palette(rng)
    return np.clip(a[None, None, :] * (1.0 - f) + b[None, None, :] * f, 0.0, 1.0)


def math_sine(w: int, h: int, rng: random.Random) -> np.ndarray:
    x, y = _mesh(w, h)
    field = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.randint(1, 5)):
        theta = rng.uniform(0.0, 2 * math.pi)
        # cycles/pixel, deliberately spanning low through near-Nyquist structure.
        f = math.exp(rng.uniform(math.log(0.015), math.log(0.42)))
        phase = rng.uniform(0.0, 2 * math.pi)
        u = math.cos(theta) * x + math.sin(theta) * y
        field += rng.uniform(0.3, 1.0) * np.sin(2 * math.pi * f * u + phase)
    return _colorize(field, rng)


def math_moire(w: int, h: int, rng: random.Random) -> np.ndarray:
    x, y = _mesh(w, h)
    theta = rng.uniform(0.0, math.pi)
    theta2 = theta + rng.uniform(-0.25, 0.25)
    f1 = math.exp(rng.uniform(math.log(0.04), math.log(0.40)))
    f2 = max(0.01, min(0.48, f1 + rng.uniform(-0.035, 0.035)))
    u1 = math.cos(theta) * x + math.sin(theta) * y
    u2 = math.cos(theta2) * x + math.sin(theta2) * y
    field = np.sin(2 * math.pi * f1 * u1 + rng.uniform(0, 2*math.pi)) + np.sin(2 * math.pi * f2 * u2 + rng.uniform(0, 2*math.pi))
    return _colorize(field, rng)


def math_checker(w: int, h: int, rng: random.Random) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    sx = rng.randint(1, max(1, min(18, w // 2)))
    sy = rng.randint(1, max(1, min(18, h // 2)))
    phase_x, phase_y = rng.randint(0, sx), rng.randint(0, sy)
    field = (((xx + phase_x) // sx + (yy + phase_y) // sy) % 2).astype(np.float32)
    return _colorize(field, rng)


def math_voronoi(w: int, h: int, rng: random.Random) -> np.ndarray:
    # Chunk-friendly Voronoi over small training regions.
    n = max(3, min(80, int((w * h) / max(20.0, rng.uniform(35.0, 350.0)))))
    pts = np.array([[rng.uniform(0, w-1), rng.uniform(0, h-1)] for _ in range(n)], dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = (xx[..., None] - pts[:, 0])**2 + (yy[..., None] - pts[:, 1])**2
    nearest = np.argmin(d2, axis=-1)
    colors = np.array([[rng.random() for _ in range(3)] for _ in range(n)], dtype=np.float32)
    img = colors[nearest]
    # Add cell-boundary contrast without drawing symbol templates.
    sorted2 = np.partition(d2, 1, axis=-1)
    boundary = _norm01(np.sqrt(np.maximum(sorted2[..., 1], 0)) - np.sqrt(np.maximum(sorted2[..., 0], 0)))
    img = np.clip(0.8 * img + 0.2 * boundary[..., None], 0.0, 1.0)
    return img


def math_fbm(w: int, h: int, rng: random.Random) -> np.ndarray:
    field = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for octave in range(rng.randint(3, 6)):
        cell = max(2, int(round(2 ** (octave + 1))))
        gw = max(2, int(math.ceil(w / cell)) + 1)
        gh = max(2, int(math.ceil(h / cell)) + 1)
        coarse = np.array([[rng.random() for _ in range(gw)] for _ in range(gh)], dtype=np.float32)
        up = Image.fromarray((coarse * 255).astype(np.uint8), mode="L").resize((w, h), BICUBIC)
        field += amp * (np.asarray(up, dtype=np.float32) / 255.0)
        total += amp
        amp *= rng.uniform(0.42, 0.68)
    return _colorize(field / max(total, 1e-6), rng)


def math_noise(w: int, h: int, rng: random.Random) -> np.ndarray:
    seed = stable_seed(rng.random(), w, h)
    nrng = np.random.default_rng(seed)
    white = nrng.standard_normal((h, w)).astype(np.float32)
    radius = rng.uniform(0.4, 3.0)
    smooth_img = Image.fromarray((_norm01(white) * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
    smooth = np.asarray(smooth_img, dtype=np.float32) / 255.0
    if rng.random() < 0.5:
        field = white - (smooth - 0.5) * rng.uniform(1.0, 3.0)
    else:
        field = smooth + rng.uniform(0.05, 0.4) * white
    return _colorize(field, rng)


def math_lines(w: int, h: int, rng: random.Random) -> np.ndarray:
    # Random graph geometry only: no E/H/X/letter/symbol templates.
    bg = tuple(int(255 * rng.uniform(0.72, 0.98)) for _ in range(3))
    canvas = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(canvas)
    n = rng.randint(6, max(8, min(50, (w*h)//120)))
    for _ in range(n):
        color = tuple(int(255 * rng.uniform(0.03, 0.95)) for _ in range(3))
        width = rng.randint(1, max(1, min(w, h)//10))
        kind = rng.choice(["segment", "polyline", "arc", "box"])
        if kind == "segment":
            draw.line((rng.randrange(w), rng.randrange(h), rng.randrange(w), rng.randrange(h)), fill=color, width=width)
        elif kind == "polyline":
            pts = [(rng.randrange(w), rng.randrange(h)) for _ in range(rng.randint(3, 6))]
            draw.line(pts, fill=color, width=width, joint="curve")
        elif kind == "arc":
            x0, x1 = sorted((rng.randrange(w), rng.randrange(w)))
            y0, y1 = sorted((rng.randrange(h), rng.randrange(h)))
            if x1 > x0 and y1 > y0:
                start = rng.randrange(360)
                draw.arc((x0, y0, x1, y1), start=start, end=start+rng.randint(30, 300), fill=color, width=width)
        else:
            x0, x1 = sorted((rng.randrange(w), rng.randrange(w)))
            y0, y1 = sorted((rng.randrange(h), rng.randrange(h)))
            if x1 > x0 and y1 > y0:
                draw.rectangle((x0, y0, x1, y1), outline=color, width=max(1, width//2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


MATH_GENERATORS = {
    "sine": math_sine,
    "moire": math_moire,
    "checker": math_checker,
    "voronoi": math_voronoi,
    "fbm": math_fbm,
    "noise": math_noise,
    "lines": math_lines,
}

DEFAULT_FAMILY_WEIGHTS = {
    "lines": 0.22,
    "voronoi": 0.16,
    "moire": 0.14,
    "sine": 0.14,
    "fbm": 0.14,
    "checker": 0.10,
    "noise": 0.10,
}


def overlay_math_region(
    base: Image.Image,
    patch_size: int,
    families: Sequence[str],
    rng: random.Random,
) -> Tuple[Image.Image, str, Tuple[int, int, int, int], Dict[str, Any]]:
    grid_x = base.width // patch_size
    grid_y = base.height // patch_size
    family_weights = [DEFAULT_FAMILY_WEIGHTS.get(f, 1.0) for f in families]
    family = rng.choices(list(families), weights=family_weights, k=1)[0]

    # Mostly local patch groups, with occasional larger sign-like panels.
    span_choices = [1, 2, 3, 4, 6, 8]
    span_weights = [0.12, 0.18, 0.20, 0.22, 0.18, 0.10]
    sw = min(grid_x, rng.choices(span_choices, weights=span_weights, k=1)[0])
    sh = min(grid_y, rng.choices(span_choices, weights=span_weights, k=1)[0])
    # Slight preference for sign-like rectangular regions.
    if rng.random() < 0.40:
        sh = min(sh, max(1, sw // rng.choice([2, 3])))
    col = rng.randint(0, max(0, grid_x - sw))
    row = rng.randint(0, max(0, grid_y - sh))
    x, y, w, h = col * patch_size, row * patch_size, sw * patch_size, sh * patch_size
    arr = MATH_GENERATORS[family](w, h, rng)
    tex = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    alpha = rng.uniform(0.70, 1.0)
    if alpha < 0.999:
        underlying = base.crop((x, y, x+w, y+h)).convert("RGB")
        tex = Image.blend(underlying, tex, alpha=alpha)
    out = base.copy().convert("RGB")
    out.paste(tex, (x, y))
    meta = {
        "family": family,
        "patch_rect": [row, col, sh, sw],
        "pixel_rect": [x, y, w, h],
        "alpha": float(alpha),
    }
    return out, family, (x, y, w, h), meta


class ContinuousTextMathImageNetBuilder:
    """ImageNet semantic crossing + continuous typography + procedural hard negatives.

    Packet images:
      0 supportive clean
      1 supportive + literal WORD
      2 adversarial clean
      3 adversarial + same literal WORD
      4 adversarial + procedural math, NO TEXT
      5 same math twin + literal WORD
      6 same math twin + different real literal WRONG_WORD

    The math/no-text image is a true negative because every procedural family is
    generated without character templates. The two text-bearing math twins teach
    candidate specificity on an otherwise identical high-frequency background.
    """

    def __init__(
        self,
        digital_root: Path,
        wnid_json: Path,
        split: str,
        image_size: int,
        patch_size: int,
        font_paths: Sequence[str],
        words: str = "all",
        math_families: Sequence[str] = tuple(MATH_GENERATORS),
        crop_probability: float = 0.20,
    ):
        self.digital_root = Path(digital_root)
        self.wnid_json = Path(wnid_json)
        self.split = str(split)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid = self.image_size // self.patch_size
        self.font_paths = [str(x) for x in font_paths if Path(str(x)).is_file()]
        if not self.font_paths:
            raise RuntimeError("ContinuousTextMathImageNetBuilder found no usable TrueType fonts")
        self.crop_probability = float(crop_probability)
        if not (0.0 <= self.crop_probability <= 0.8):
            raise ValueError("crop_probability must be in [0, 0.8]")
        self.math_families = [str(x) for x in math_families if str(x) in MATH_GENERATORS]
        if not self.math_families:
            raise ValueError(f"No valid math families selected; available={sorted(MATH_GENERATORS)}")

        lookup_obj = json.loads(self.wnid_json.read_text(encoding="utf-8"))
        self.lookup: Dict[str, str] = {str(k): str(v) for k, v in lookup_obj.items()}
        taxonomy = build_short_concept_taxonomy(self.lookup)
        requested = [x.strip().lower() for x in str(words).split(",") if x.strip()]
        if requested == ["all"] or not requested:
            requested = sorted(taxonomy)
        unknown = sorted(set(requested) - set(taxonomy))
        if unknown:
            raise ValueError(f"Unknown curriculum words {unknown}; available={sorted(taxonomy)}")
        self.taxonomy = {w: taxonomy[w] for w in requested if taxonomy[w].wnids}

        manifest = self.digital_root / "manifests" / f"{self.split}.jsonl"
        rows = read_jsonl(manifest)
        self.by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            self.by_group[str(row.get("group_id"))].append(row)
        self.group_meta: Dict[str, Dict[str, Any]] = {}
        for gid, grows in self.by_group.items():
            wnid = parse_group_wnid(gid, grows)
            clean_rows = [r for r in grows if str(r.get("relation")) == "none"]
            if not wnid or wnid not in self.lookup or not clean_rows:
                continue
            self.group_meta[gid] = {
                "wnid": wnid,
                "class_label": self.lookup[wnid],
                "domain_words": _taxonomy_membership(wnid, self.taxonomy),
                "clean_rows": clean_rows,
            }

        all_gids = sorted(self.group_meta)
        self.support_pools: Dict[str, List[str]] = {}
        self.adversarial_pools: Dict[str, List[str]] = {}
        for word, spec in self.taxonomy.items():
            support = [g for g in all_gids if self.group_meta[g]["wnid"] in spec.wnids]
            cross: List[str] = []
            fallback: List[str] = []
            for g in all_gids:
                meta = self.group_meta[g]
                if meta["wnid"] in spec.wnids:
                    continue
                toks = set(re.findall(r"[a-z0-9]+", meta["class_label"].lower()))
                if word in toks:
                    continue
                fallback.append(g)
                other = meta["domain_words"]
                if any(self.taxonomy[ow].domain != spec.domain for ow in other):
                    cross.append(g)
            self.support_pools[word] = support
            self.adversarial_pools[word] = cross or fallback
        self.words = [w for w in sorted(self.taxonomy) if self.support_pools[w] and self.adversarial_pools[w]]
        if len(self.words) < 2:
            raise RuntimeError("Need at least two usable ImageNet concepts for candidate-specific math twins")

    @property
    def group_count(self) -> int:
        return len(self.group_meta)

    @property
    def concept_count(self) -> int:
        return len(self.words)

    def coverage(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "groups": self.group_count,
            "concepts": self.concept_count,
            "words": self.words,
            "math_families": self.math_families,
            "crop_probability": self.crop_probability,
            "support_groups": {w: len(self.support_pools[w]) for w in self.words},
            "adversarial_groups": {w: len(self.adversarial_pools[w]) for w in self.words},
        }

    def _clean(self, gid: str, rng: random.Random) -> Image.Image:
        row = rng.choice(self.group_meta[gid]["clean_rows"])
        rel = str(row["image_relpath"])
        return crop_like_training(Image.open(self.digital_root / rel).convert("RGB"), self.image_size)

    def sample(self, rng: random.Random) -> ContinuousMathPacketData:
        for _attempt in range(48):
            word = rng.choice(self.words)
            wrong_choices = [w for w in self.words if w != word]
            wrong_word = rng.choice(wrong_choices)
            support_gid = rng.choice(self.support_pools[word])
            adv_candidates = [g for g in self.adversarial_pools[word] if g != support_gid]
            if not adv_candidates:
                continue
            adv_gid = rng.choice(adv_candidates)
            support = self._clean(support_gid, rng)
            adversarial = self._clean(adv_gid, rng)

            try:
                support_word = render_continuous_word(
                    support, word, self.patch_size, self.font_paths, rng, self.crop_probability
                )
                # Independently sample scale/style on adversarial ImageNet: this
                # prevents a word from acquiring a single canonical raster.
                adv_word = render_continuous_word(
                    adversarial, word, self.patch_size, self.font_paths, rng, self.crop_probability
                )
                math_no_text, math_family, math_region, math_meta = overlay_math_region(
                    adversarial, self.patch_size, self.math_families, rng
                )
                # Matched counterfactual twins: both real words are rendered into
                # the exact same math background. Cap them to the math region so
                # the literal evidence must compete with the structured texture.
                math_word = render_continuous_word(
                    math_no_text,
                    word,
                    self.patch_size,
                    self.font_paths,
                    rng,
                    crop_probability=0.0,
                    preferred_region=math_region,
                )
                math_wrong = render_continuous_word(
                    math_no_text,
                    wrong_word,
                    self.patch_size,
                    self.font_paths,
                    rng,
                    crop_probability=0.0,
                    preferred_region=math_region,
                )
            except Exception:
                continue

            support_label = self.group_meta[support_gid]["class_label"]
            adv_label = self.group_meta[adv_gid]["class_label"]
            captions = [
                f"a photo of a {support_label}",                              # 0
                f"<notext> a photo of a {support_label}",                     # 1
                f"a photo of a {adv_label}",                                  # 2
                f"<notext> a photo of a {adv_label}",                         # 3
                f"<text> {word}",                                             # 4
                f"<text> {wrong_word}",                                       # 5
                f'a photo of a {support_label} with the text "{word}"',       # 6
                f'a photo of a {adv_label} with the text "{word}"',           # 7
                f'a photo of a {adv_label} with the text "{wrong_word}"',     # 8
                "<text> <null>",                                              # 9
                f"a photo of a {word}",                                       # 10 semantic attack candidate
                f"a photo of a {wrong_word}",                                 # 11 semantic wrong-word candidate
            ]

            positive_pairs: List[Tuple[int, int]] = []
            for img in (0, 1):
                positive_pairs.extend(((img, 0), (img, 1)))
            for img in (2, 3, 4, 5, 6):
                positive_pairs.extend(((img, 2), (img, 3)))
            # Same literal word in supportive/adversarial/math contexts.
            positive_pairs.extend(((1, 4), (3, 4), (5, 4)))
            # Same math background but a different real word: this is text and
            # is positive only for its own candidate, never treated as non-text.
            positive_pairs.append((6, 5))
            positive_pairs.extend(((1, 6), (3, 7), (5, 7), (6, 8)))
            # Explicit null positives only on genuinely text-free images.
            positive_pairs.extend(((0, 9), (2, 9), (4, 9)))

            # Candidate-specific ranking. In particular, image 5 vs 6 forces the
            # reader/orthographic branch to distinguish WORD from WRONG_WORD on
            # the same procedural background rather than merely detecting text.
            source_triplets = [
                (4, 1, [0, 2, 4, 6]),
                (4, 3, [0, 2, 4, 6]),
                (4, 5, [0, 2, 4, 6]),
                (5, 6, [0, 1, 2, 3, 4, 5]),
                (9, 0, [1, 3, 5, 6]),
                (9, 2, [1, 3, 5, 6]),
                (9, 4, [1, 3, 5, 6]),
            ]
            auto_triplets = [
                (3, 2, 10),
                (5, 2, 10),
                (6, 2, 11),
            ]

            blank_support = Image.new("L", support.size, 0)
            blank_adv = Image.new("L", adversarial.size, 0)
            return ContinuousMathPacketData(
                images=[
                    support,
                    support_word.image,
                    adversarial,
                    adv_word.image,
                    math_no_text,
                    math_word.image,
                    math_wrong.image,
                ],
                masks=[
                    blank_support,
                    support_word.mask,
                    blank_adv,
                    adv_word.mask,
                    Image.new("L", adversarial.size, 0),
                    math_word.mask,
                    math_wrong.mask,
                ],
                mask_weights=[1.0] * 7,
                present_targets=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                readable_targets=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                captions=captions,
                positive_pairs=positive_pairs,
                source_triplets=source_triplets,
                auto_triplets=auto_triplets,
                # Keep ordinary object semantics stable across overlays. The
                # math region is patch-local (<=8x8) so this is not a full-image
                # occlusion objective.
                invariance_pairs=[(0, 1), (2, 3), (2, 4), (2, 5), (2, 6)],
                metadata={
                    "source": "imagenet_math",
                    "concept_word": word,
                    "wrong_word": wrong_word,
                    "support_group_id": support_gid,
                    "support_wnid": self.group_meta[support_gid]["wnid"],
                    "support_label": support_label,
                    "adversarial_group_id": adv_gid,
                    "adversarial_wnid": self.group_meta[adv_gid]["wnid"],
                    "adversarial_label": adv_label,
                    "math": math_meta,
                    "support_text": self._render_meta(support_word),
                    "adversarial_text": self._render_meta(adv_word),
                    "math_text": self._render_meta(math_word),
                    "math_wrong_text": self._render_meta(math_wrong),
                    "adversarial_any_negative": f"a photo of a {word}",
                    "math_family": math_family,
                },
            )
        raise RuntimeError("Continuous math curriculum generator failed after 48 retries")

    @staticmethod
    def _render_meta(rendered: RenderedText) -> Dict[str, Any]:
        return {
            "word": rendered.word,
            "regime": rendered.regime,
            "font_name": rendered.font_name,
            "font_size": rendered.font_size,
            "stroke_width": rendered.stroke_width,
            "rotation_deg": rendered.rotation_deg,
            "opacity": rendered.opacity,
            "crop_visible_fraction": rendered.crop_visible_fraction,
            "occupied_patches": rendered.occupied_patches,
            "ink_bbox": rendered.ink_bbox,
        }

    def save_preview_grids(self, out_dir: Path, count: int, seed: int) -> List[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(int(seed))
        saved: List[Path] = []
        preview_indices = [1, 3, 4, 5, 6]
        labels = ["support_text", "adv_text", "math_no_text", "math_text", "math_wrong"]
        for i in range(max(0, int(count))):
            packet = self.sample(rng)
            idx = preview_indices[i % len(preview_indices)]
            mask = packet.masks[idx]
            preview = draw_patch_grid(packet.images[idx], mask, self.patch_size)
            name = (
                f"math_curriculum_{i:02d}_{packet.metadata['concept_word']}_"
                f"{packet.metadata['math_family']}_{labels[i % len(labels)]}.png"
            )
            path = out_dir / name
            preview.save(path)
            saved.append(path)
        return saved
