from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from training_support.font_discovery import find_font_files


DEFAULT_WORDS = (
    "reading", "signal", "vector", "tensor", "theorem", "archive", "silver",
    "meadow", "orbit", "lantern", "cobalt", "syntax", "kernel", "matrix",
    "latent", "bridge", "scalar", "diagram", "packet", "channel", "model",
    "attention", "vision", "pattern", "garden", "window", "memory", "paper",
)

DEFAULT_VARIANTS = ("text", "phase", "shred", "sine", "checker", "blank")
DEFAULT_POSITIONS = ("top", "center", "bottom", "left", "right")


@dataclass(frozen=True)
class SceneSpec:
    scene_id: int
    seed: int
    resolution: int
    patch_size: int
    grid_size: int
    position: str
    phrase: str
    bbox_px: tuple[int, int, int, int]
    bbox_patch: tuple[int, int, int, int]
    ink_fraction: float
    background_kind: str
    sine_theta: float
    sine_cycles_per_patch: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox_px"] = list(self.bbox_px)
        d["bbox_patch"] = list(self.bbox_patch)
        return d


SYNTHETIC_FONT_NAMES = (
    "arial.ttf",
    "calibri.ttf",
    "segoeui.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "FreeSans.ttf",
    "Arial.ttf",
    "Helvetica.ttc",
)


def _candidate_fonts(extra: Optional[Iterable[str]] = None) -> list[str]:
    explicit = tuple(str(value) for value in extra) if extra else ()
    return find_font_files(explicit, preferred_names=SYNTHETIC_FONT_NAMES)


def _load_font(max_size: int, text: str, box_wh: tuple[int, int], paths: list[str]) -> ImageFont.FreeTypeFont:
    w, h = box_wh
    if not paths:
        return ImageFont.load_default()
    path = paths[0]
    lo, hi = 6, max(6, int(max_size))
    best = ImageFont.truetype(path, size=lo)
    dummy = Image.new("L", (max(1, w), max(1, h)), 0)
    draw = ImageDraw.Draw(dummy)
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(path, size=mid)
        b = draw.textbbox((0, 0), text, font=font, stroke_width=0)
        tw, th = b[2] - b[0], b[3] - b[1]
        if tw <= int(w * 0.92) and th <= int(h * 0.78):
            best = font
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _smooth_background(rng: np.random.Generator, resolution: int, kind: str) -> np.ndarray:
    y, x = np.mgrid[0:resolution, 0:resolution].astype(np.float32)
    xn = x / max(1, resolution - 1)
    yn = y / max(1, resolution - 1)

    base = rng.uniform(180, 238, size=3).astype(np.float32)
    amp1 = rng.uniform(-28, 28, size=3).astype(np.float32)
    amp2 = rng.uniform(-22, 22, size=3).astype(np.float32)

    if kind == "solid":
        field1 = np.zeros_like(xn)
        field2 = np.zeros_like(xn)
    elif kind == "linear":
        theta = rng.uniform(0, 2 * np.pi)
        field1 = np.cos(theta) * (xn - 0.5) + np.sin(theta) * (yn - 0.5)
        field2 = 0.25 * ((xn - 0.5) ** 2 - (yn - 0.5) ** 2)
    elif kind == "radial":
        cx, cy = rng.uniform(0.25, 0.75, size=2)
        rr = np.sqrt((xn - cx) ** 2 + (yn - cy) ** 2)
        field1 = rr - rr.mean()
        field2 = (xn - 0.5) * (yn - 0.5)
    else:
        # Low-frequency analytic field, intentionally incapable of spelling anything.
        a = rng.uniform(0.7, 1.5)
        b = rng.uniform(0.7, 1.5)
        ph = rng.uniform(0, 2 * np.pi)
        field1 = np.sin(2 * np.pi * (a * xn + b * yn) + ph) * 0.20
        field2 = np.cos(2 * np.pi * (0.5 * b * xn - 0.5 * a * yn) - ph) * 0.15

    img = base[None, None, :] + field1[..., None] * amp1[None, None, :] + field2[..., None] * amp2[None, None, :]
    return np.clip(img, 0, 255).astype(np.uint8)


def _patch_bbox(grid: int, patch: int, position: str, width_frac: float, height_frac: float) -> tuple[tuple[int,int,int,int], tuple[int,int,int,int]]:
    wp = max(3, min(grid, int(round(grid * width_frac))))
    hp = max(2, min(grid, int(round(grid * height_frac))))

    if position == "top":
        x0p = (grid - wp) // 2
        y0p = max(0, grid // 8 - hp // 2)
    elif position == "bottom":
        x0p = (grid - wp) // 2
        y0p = min(grid - hp, (7 * grid) // 8 - hp // 2)
    elif position == "left":
        x0p = max(0, grid // 8 - wp // 2)
        y0p = (grid - hp) // 2
    elif position == "right":
        x0p = min(grid - wp, (7 * grid) // 8 - wp // 2)
        y0p = (grid - hp) // 2
    else:
        x0p = (grid - wp) // 2
        y0p = (grid - hp) // 2

    x0p = int(max(0, min(grid - wp, x0p)))
    y0p = int(max(0, min(grid - hp, y0p)))
    x1p, y1p = x0p + wp, y0p + hp
    return (x0p * patch, y0p * patch, x1p * patch, y1p * patch), (x0p, y0p, x1p, y1p)


def _render_text_mask(text: str, w: int, h: int, font_paths: list[str]) -> np.ndarray:
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max_size=max(8, int(h * 0.75)), text=text, box_wh=(w, h), paths=font_paths)
    b = draw.textbbox((0, 0), text, font=font)
    tw, th = b[2] - b[0], b[3] - b[1]
    x = (w - tw) // 2 - b[0]
    y = (h - th) // 2 - b[1]
    draw.text((x, y), text, fill=255, font=font)
    return np.asarray(canvas, dtype=np.uint8)


def _binary_from_score(score: np.ndarray, ink_fraction: float) -> np.ndarray:
    flat = score.reshape(-1)
    k = int(round(float(ink_fraction) * flat.size))
    k = max(1, min(flat.size - 1, k))
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros(flat.shape[0], dtype=np.uint8)
    out[idx] = 255
    return out.reshape(score.shape)


def _phase_scramble(mask: np.ndarray, rng: np.random.Generator, ink_fraction: float) -> np.ndarray:
    x = mask.astype(np.float32) / 255.0
    x = x - x.mean()
    spec = np.fft.rfft2(x)
    mag = np.abs(spec)
    phase = rng.uniform(-np.pi, np.pi, size=spec.shape)
    phase[0, 0] = 0.0
    scrambled = np.fft.irfft2(mag * np.exp(1j * phase), s=x.shape).real
    return _binary_from_score(scrambled, ink_fraction)


def _block_shred(mask: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    h, w = mask.shape
    block = max(1, int(block))
    ph = int(math.ceil(h / block) * block)
    pw = int(math.ceil(w / block) * block)
    padded = np.zeros((ph, pw), dtype=np.uint8)
    padded[:h, :w] = mask
    tiles = []
    for y in range(0, ph, block):
        for x in range(0, pw, block):
            tiles.append(padded[y:y+block, x:x+block].copy())
    rng.shuffle(tiles)
    out = np.zeros_like(padded)
    ncol = pw // block
    for i, tile in enumerate(tiles):
        y = (i // ncol) * block
        x = (i % ncol) * block
        out[y:y+block, x:x+block] = tile
    return out[:h, :w]


def _sine_mask(w: int, h: int, rng: np.random.Generator, patch_size: int, theta: float, cycles_per_patch: float, ink_fraction: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = np.cos(theta) * xx + np.sin(theta) * yy
    cycles_per_px = cycles_per_patch / float(patch_size)
    score = np.sin(2 * np.pi * cycles_per_px * u + rng.uniform(0, 2*np.pi))
    # deterministic tiny jitter breaks ties while preserving the frequency field.
    score = score + rng.normal(0.0, 1e-5, size=score.shape)
    return _binary_from_score(score, ink_fraction)


def _checker_mask(w: int, h: int, rng: np.random.Generator, patch_size: int, ink_fraction: float) -> np.ndarray:
    # Continuous 2-D checker score so quantile thresholding can match arbitrary
    # ink fractions without degenerating into random tie-breaking noise.
    cell = max(2, patch_size // 3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    score = np.cos(np.pi * xx / cell) * np.cos(np.pi * yy / cell)
    score += rng.normal(0.0, 1e-6, size=score.shape)
    return _binary_from_score(score, ink_fraction)


class PatchAlignedSyntheticBank:
    """Deterministic paired synthetic images for text-vs-high-frequency controls.

    Every variant of a scene shares the exact same background and patch-aligned
    bounding box.  The controls are matched to the readable-text mask's foreground
    fraction inside that box.  This deliberately avoids LAION-style unknown text.
    """

    def __init__(
        self,
        *,
        resolution: int,
        patch_size: int = 14,
        seed: int = 20260901,
        width_frac: float = 0.62,
        height_frac: float = 0.25,
        positions: Iterable[str] = DEFAULT_POSITIONS,
        words: Iterable[str] = DEFAULT_WORDS,
        font_paths: Optional[Iterable[str]] = None,
    ) -> None:
        if resolution % patch_size:
            raise ValueError(f"resolution={resolution} must be divisible by patch_size={patch_size}")
        self.resolution = int(resolution)
        self.patch_size = int(patch_size)
        self.grid_size = self.resolution // self.patch_size
        self.seed = int(seed)
        self.width_frac = float(width_frac)
        self.height_frac = float(height_frac)
        self.positions = tuple(positions)
        self.words = tuple(words)
        self.font_paths = _candidate_fonts(font_paths)
        if not self.positions:
            raise ValueError("positions cannot be empty")
        if len(self.words) < 2:
            raise ValueError("words needs at least two entries")

    def _rng(self, scene_id: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(scene_id) * 10007)

    def scene(self, scene_id: int) -> tuple[SceneSpec, dict[str, Image.Image]]:
        rng = self._rng(scene_id)
        py_rng = random.Random(self.seed + int(scene_id) * 7919)
        position = self.positions[scene_id % len(self.positions)]
        bbox_px, bbox_patch = _patch_bbox(
            self.grid_size, self.patch_size, position,
            self.width_frac, self.height_frac,
        )
        x0, y0, x1, y1 = bbox_px
        w, h = x1 - x0, y1 - y0

        n_words = 2 if py_rng.random() < 0.7 else 3
        chosen = [self.words[py_rng.randrange(len(self.words))] for _ in range(n_words)]
        phrase = " ".join(chosen)

        bg_kind = ("solid", "linear", "radial", "smooth")[scene_id % 4]
        bg = _smooth_background(rng, self.resolution, bg_kind)
        text_mask = _render_text_mask(phrase, w, h, self.font_paths)
        ink_fraction = float((text_mask > 0).mean())
        ink_fraction = min(0.45, max(0.02, ink_fraction))

        theta = float(rng.uniform(0, np.pi))
        cpp = float(rng.uniform(1.25, 3.5))
        masks = {
            "text": text_mask,
            "phase": _phase_scramble(text_mask, rng, ink_fraction),
            "shred": _block_shred(text_mask, rng, block=max(2, self.patch_size // 2)),
            "sine": _sine_mask(w, h, rng, self.patch_size, theta, cpp, ink_fraction),
            "checker": _checker_mask(w, h, rng, self.patch_size, ink_fraction),
            "blank": np.zeros((h, w), dtype=np.uint8),
        }

        images: dict[str, Image.Image] = {}
        for name, mask in masks.items():
            arr = bg.copy()
            if name != "blank":
                region = arr[y0:y1, x0:x1]
                m = (mask.astype(np.float32) / 255.0)[..., None]
                # Dark ink; background remains otherwise untouched.
                dark = np.full_like(region, 18, dtype=np.uint8)
                region[:] = np.clip(region.astype(np.float32) * (1.0 - m) + dark.astype(np.float32) * m, 0, 255).astype(np.uint8)
            images[name] = Image.fromarray(arr, mode="RGB")

        spec = SceneSpec(
            scene_id=int(scene_id), seed=self.seed, resolution=self.resolution,
            patch_size=self.patch_size, grid_size=self.grid_size,
            position=position, phrase=phrase, bbox_px=bbox_px, bbox_patch=bbox_patch,
            ink_fraction=ink_fraction, background_kind=bg_kind,
            sine_theta=theta, sine_cycles_per_patch=cpp,
        )
        return spec, images

    def preview(self, path: str | Path, n_scenes: int = 4, variants: Iterable[str] = DEFAULT_VARIANTS) -> Path:
        variants = tuple(variants)
        thumb = self.resolution
        canvas = Image.new("RGB", (thumb * len(variants), thumb * n_scenes), "white")
        for row in range(n_scenes):
            _, imgs = self.scene(row)
            for col, variant in enumerate(variants):
                canvas.paste(imgs[variant], (col * thumb, row * thumb))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        return path
