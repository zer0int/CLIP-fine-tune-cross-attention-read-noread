from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9
    BICUBIC = Image.BICUBIC


CLEVR_COLORS: Tuple[str, ...] = (
    "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"
)
OOD_COLORS: Tuple[str, ...] = ("pink", "orange")
CLEVR_SHAPES: Tuple[str, ...] = ("cube", "sphere", "cylinder")
NUMBER_WORDS: Dict[int, str] = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}
COLOR_RGB: Dict[str, Tuple[int, int, int]] = {
    "gray": (110, 110, 110),
    "red": (205, 45, 45),
    "blue": (45, 85, 205),
    "green": (45, 155, 60),
    "brown": (130, 85, 45),
    "purple": (135, 60, 175),
    "cyan": (45, 180, 190),
    "yellow": (220, 190, 45),
    "pink": (225, 95, 155),
    "orange": (230, 125, 35),
}


def _plural(shape: str, count: int) -> str:
    return shape if int(count) == 1 else shape + "s"


def _sentence_count_shape(count: int, shape: str) -> str:
    word = NUMBER_WORDS[int(count)]
    if int(count) == 1:
        return f"there is one {shape}"
    return f"there are {word} {_plural(shape, count)}"


def _sentence_count_color_shape(count: int, color: str, shape: str) -> str:
    word = NUMBER_WORDS[int(count)]
    if int(count) == 1:
        return f"there is one {color} {shape}"
    return f"there are {word} {color} {_plural(shape, count)}"


def _sentence_count_color(count: int, color: str) -> str:
    word = NUMBER_WORDS[int(count)]
    if int(count) == 1:
        return f"there is one {color} object"
    return f"there are {word} {color} objects"


@dataclass(frozen=True)
class ClevrFact:
    visible_text: str
    semantic_text: str
    kind: str
    count: Optional[int] = None
    color: Optional[str] = None
    shape: Optional[str] = None
    basis_kind: Optional[str] = None


@dataclass
class ClevrPacketData:
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


@dataclass
class CannyPlacement:
    x: int
    y: int
    box_width: int
    box_height: int
    edge_fraction: float
    obstacle_fraction: float
    canny_low: int
    canny_high: int


@dataclass
class RenderPair:
    true_image: Image.Image
    true_mask: Image.Image
    false_image: Image.Image
    false_mask: Image.Image
    placement: CannyPlacement
    font_name: str
    font_size: int
    stroke_width: int
    true_color: Tuple[int, int, int]
    false_color: Tuple[int, int, int]
    colored_style: bool
    congruent_color: Optional[bool]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(obj)
    return rows


def _normalize_rows(rows: Sequence[Mapping[str, Any]], split: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        row_split = str(row.get("split", row.get("_local_split", ""))).lower()
        if row_split not in {split.lower(), "validation" if split == "val" else split.lower()}:
            continue
        filename = str(row.get("image_filename", ""))
        objects = row.get("objects")
        if not filename or not isinstance(objects, list):
            continue
        clean_objects: List[Dict[str, str]] = []
        for obj in objects:
            if not isinstance(obj, Mapping):
                continue
            # IMPORTANT: only semantic object attributes are consumed here.
            # The square-image preprocessing changed geometry, so original
            # CLEVR coordinates are deliberately never read or used.
            color = str(obj.get("color", "")).lower()
            shape = str(obj.get("shape", "")).lower()
            if color not in CLEVR_COLORS or shape not in CLEVR_SHAPES:
                continue
            clean_objects.append({"color": color, "shape": shape})
        if not clean_objects:
            continue
        out.append({
            "image_filename": filename,
            "image_index": row.get("image_index"),
            "objects": clean_objects,
        })
    return out


def _inventory(objects: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    color_counts = Counter(str(obj["color"]) for obj in objects)
    shape_counts = Counter(str(obj["shape"]) for obj in objects)
    color_shape_counts = Counter((str(obj["color"]), str(obj["shape"])) for obj in objects)
    return {
        "n_objects": len(objects),
        "color_counts": color_counts,
        "shape_counts": shape_counts,
        "color_shape_counts": color_shape_counts,
    }


def _base_true_facts(inv: Mapping[str, Any]) -> List[ClevrFact]:
    color_counts: Counter = inv["color_counts"]
    shape_counts: Counter = inv["shape_counts"]
    cs_counts: Counter = inv["color_shape_counts"]
    facts: List[ClevrFact] = []

    for color, count in sorted(color_counts.items()):
        if count > 0:
            facts.append(ClevrFact(
                visible_text=color,
                semantic_text=f"a {color} object is visible",
                kind="color",
                color=color,
            ))
            if int(count) in NUMBER_WORDS:
                facts.append(ClevrFact(
                    visible_text=f"{NUMBER_WORDS[int(count)]} {color} {'object' if int(count) == 1 else 'objects'}",
                    semantic_text=_sentence_count_color(int(count), color),
                    kind="count_color",
                    count=int(count),
                    color=color,
                ))

    for shape, count in sorted(shape_counts.items()):
        if count > 0:
            facts.append(ClevrFact(
                visible_text=shape,
                semantic_text=f"a {shape} is visible",
                kind="shape",
                shape=shape,
            ))
            if int(count) in NUMBER_WORDS:
                facts.append(ClevrFact(
                    visible_text=f"{NUMBER_WORDS[int(count)]} {_plural(shape, int(count))}",
                    semantic_text=_sentence_count_shape(int(count), shape),
                    kind="count_shape",
                    count=int(count),
                    shape=shape,
                ))
                # Standalone count is a small augmentation. Its semantic basis is
                # the exact shape count, so the number word is never fabricated.
                facts.append(ClevrFact(
                    visible_text=NUMBER_WORDS[int(count)],
                    semantic_text=_sentence_count_shape(int(count), shape),
                    kind="count_word",
                    count=int(count),
                    shape=shape,
                    basis_kind="shape",
                ))

    for (color, shape), count in sorted(cs_counts.items()):
        if count <= 0:
            continue
        facts.append(ClevrFact(
            visible_text=f"{color} {shape}",
            semantic_text=f"a {color} {shape} is visible",
            kind="color_shape",
            color=color,
            shape=shape,
        ))
        # Indefinite article is existential, not an exact count assertion.
        facts.append(ClevrFact(
            visible_text=f"a {color} {shape}",
            semantic_text=f"a {color} {shape} is visible",
            kind="article_color_shape",
            color=color,
            shape=shape,
        ))
        if int(count) in NUMBER_WORDS:
            facts.append(ClevrFact(
                visible_text=f"{NUMBER_WORDS[int(count)]} {color} {_plural(shape, int(count))}",
                semantic_text=_sentence_count_color_shape(int(count), color, shape),
                kind="count_color_shape",
                count=int(count),
                color=color,
                shape=shape,
            ))
    return facts


def _false_color_candidates(inv: Mapping[str, Any], rng: random.Random, ood_probability: float) -> List[str]:
    present = set(inv["color_counts"].keys())
    in_domain = [color for color in CLEVR_COLORS if color not in present]
    ood = list(OOD_COLORS)
    rng.shuffle(in_domain)
    rng.shuffle(ood)
    if ood and rng.random() < float(ood_probability):
        return ood + in_domain
    return in_domain + ood


def _false_fact_for(true: ClevrFact, inv: Mapping[str, Any], rng: random.Random, ood_probability: float) -> Optional[ClevrFact]:
    shape_counts: Counter = inv["shape_counts"]
    color_counts: Counter = inv["color_counts"]
    cs_counts: Counter = inv["color_shape_counts"]

    candidates: List[ClevrFact] = []

    if true.kind == "color":
        for color in _false_color_candidates(inv, rng, ood_probability):
            if color_counts.get(color, 0) == 0:
                candidates.append(ClevrFact(
                    visible_text=color,
                    semantic_text=f"a {color} object is visible",
                    kind=true.kind,
                    color=color,
                ))

    elif true.kind == "shape":
        for shape in CLEVR_SHAPES:
            if shape != true.shape and shape_counts.get(shape, 0) == 0:
                candidates.append(ClevrFact(
                    visible_text=shape,
                    semantic_text=f"a {shape} is visible",
                    kind=true.kind,
                    shape=shape,
                ))

    elif true.kind in {"color_shape", "article_color_shape"}:
        # One-component near misses first.
        for color in list(CLEVR_COLORS) + list(OOD_COLORS):
            if color == true.color:
                continue
            if cs_counts.get((color, true.shape), 0) == 0:
                visible = f"{color} {true.shape}"
                if true.kind == "article_color_shape":
                    visible = "a " + visible
                candidates.append(ClevrFact(
                    visible_text=visible,
                    semantic_text=f"a {color} {true.shape} is visible",
                    kind=true.kind,
                    color=color,
                    shape=true.shape,
                ))
        for shape in CLEVR_SHAPES:
            if shape == true.shape:
                continue
            if cs_counts.get((true.color, shape), 0) == 0:
                visible = f"{true.color} {shape}"
                if true.kind == "article_color_shape":
                    visible = "a " + visible
                candidates.append(ClevrFact(
                    visible_text=visible,
                    semantic_text=f"a {true.color} {shape} is visible",
                    kind=true.kind,
                    color=true.color,
                    shape=shape,
                ))

    elif true.kind == "count_shape":
        actual = int(shape_counts.get(true.shape, 0))
        for count, word in NUMBER_WORDS.items():
            if count != actual:
                candidates.append(ClevrFact(
                    visible_text=f"{word} {_plural(true.shape, count)}",
                    semantic_text=_sentence_count_shape(count, str(true.shape)),
                    kind=true.kind,
                    count=count,
                    shape=true.shape,
                ))
        # Same count, wrong shape, only if the assertion is actually false.
        for shape in CLEVR_SHAPES:
            if shape != true.shape and int(shape_counts.get(shape, 0)) != int(true.count):
                candidates.append(ClevrFact(
                    visible_text=f"{NUMBER_WORDS[int(true.count)]} {_plural(shape, int(true.count))}",
                    semantic_text=_sentence_count_shape(int(true.count), shape),
                    kind=true.kind,
                    count=int(true.count),
                    shape=shape,
                ))

    elif true.kind == "count_word":
        if true.basis_kind == "shape" and true.shape:
            actual = int(shape_counts.get(true.shape, 0))
            for count, word in NUMBER_WORDS.items():
                if count != actual:
                    candidates.append(ClevrFact(
                        visible_text=word,
                        semantic_text=_sentence_count_shape(count, true.shape),
                        kind=true.kind,
                        count=count,
                        shape=true.shape,
                        basis_kind="shape",
                    ))

    elif true.kind == "count_color":
        actual = int(color_counts.get(true.color, 0))
        for count, word in NUMBER_WORDS.items():
            if count != actual:
                candidates.append(ClevrFact(
                    visible_text=f"{word} {true.color} {'object' if count == 1 else 'objects'}",
                    semantic_text=_sentence_count_color(count, str(true.color)),
                    kind=true.kind,
                    count=count,
                    color=true.color,
                ))
        for color in list(CLEVR_COLORS) + list(OOD_COLORS):
            if color == true.color:
                continue
            if int(color_counts.get(color, 0)) != int(true.count):
                candidates.append(ClevrFact(
                    visible_text=f"{NUMBER_WORDS[int(true.count)]} {color} {'object' if int(true.count) == 1 else 'objects'}",
                    semantic_text=_sentence_count_color(int(true.count), color),
                    kind=true.kind,
                    count=int(true.count),
                    color=color,
                ))

    elif true.kind == "count_color_shape":
        actual = int(cs_counts.get((true.color, true.shape), 0))
        for count, word in NUMBER_WORDS.items():
            if count != actual:
                candidates.append(ClevrFact(
                    visible_text=f"{word} {true.color} {_plural(true.shape, count)}",
                    semantic_text=_sentence_count_color_shape(count, str(true.color), str(true.shape)),
                    kind=true.kind,
                    count=count,
                    color=true.color,
                    shape=true.shape,
                ))
        for color in list(CLEVR_COLORS) + list(OOD_COLORS):
            if color == true.color:
                continue
            if int(cs_counts.get((color, true.shape), 0)) != int(true.count):
                candidates.append(ClevrFact(
                    visible_text=f"{NUMBER_WORDS[int(true.count)]} {color} {_plural(true.shape, int(true.count))}",
                    semantic_text=_sentence_count_color_shape(int(true.count), color, str(true.shape)),
                    kind=true.kind,
                    count=int(true.count),
                    color=color,
                    shape=true.shape,
                ))
        for shape in CLEVR_SHAPES:
            if shape == true.shape:
                continue
            if int(cs_counts.get((true.color, shape), 0)) != int(true.count):
                candidates.append(ClevrFact(
                    visible_text=f"{NUMBER_WORDS[int(true.count)]} {true.color} {_plural(shape, int(true.count))}",
                    semantic_text=_sentence_count_color_shape(int(true.count), str(true.color), shape),
                    kind=true.kind,
                    count=int(true.count),
                    color=true.color,
                    shape=shape,
                ))

    if not candidates:
        return None

    # Prefer same-domain near misses. OOD pink/orange remain a small augmentation,
    # except when no in-domain false candidate exists.
    in_domain = [c for c in candidates if c.color not in OOD_COLORS]
    ood = [c for c in candidates if c.color in OOD_COLORS]
    pool = candidates
    if in_domain and (not ood or rng.random() >= float(ood_probability)):
        pool = in_domain
    elif ood:
        pool = ood
    return rng.choice(pool)


def _font_bbox(text: str, font: ImageFont.FreeTypeFont, stroke_width: int) -> Tuple[int, int, int, int]:
    probe = Image.new("L", (16, 16), 0)
    draw = ImageDraw.Draw(probe)
    return draw.textbbox((0, 0), text, font=font, stroke_width=int(stroke_width))


def _render_text_mask(text: str, font_path: str, font_size: int, stroke_width: int) -> Image.Image:
    font = ImageFont.truetype(font_path, int(font_size))
    bbox = _font_bbox(text, font, stroke_width)
    width = max(1, int(bbox[2] - bbox[0]))
    height = max(1, int(bbox[3] - bbox[1]))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.text(
        (-bbox[0], -bbox[1]),
        text,
        font=font,
        fill=255,
        stroke_width=int(stroke_width),
        stroke_fill=255,
    )
    tight = mask.getbbox()
    if tight is None:
        raise RuntimeError(f"Empty text mask for {text!r}")
    return mask.crop(tight)


def _canny_obstacle_map(
    image: Image.Image,
    low: int,
    high: int,
    dilate_px: int,
    fill_contours: bool,
    min_contour_area_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, int(low), int(high), L2gradient=True)

    obstacle = edges.copy()
    if fill_contours:
        contours, _hier = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(image.width * image.height) * float(min_contour_area_fraction)
        fill = np.zeros_like(edges)
        for contour in contours:
            if abs(float(cv2.contourArea(contour))) >= min_area:
                cv2.drawContours(fill, [contour], -1, 255, thickness=cv2.FILLED)
        obstacle = cv2.bitwise_or(obstacle, fill)

    if int(dilate_px) > 0:
        k = 2 * int(dilate_px) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        obstacle = cv2.dilate(obstacle, kernel, iterations=1)
    return edges, obstacle


def _integral_fraction(integral: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    x0, y0 = int(x), int(y)
    x1, y1 = int(x + w), int(y + h)
    value = (
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )
    return float(value) / float(max(1, w * h))


def find_canny_placement(
    image: Image.Image,
    box_width: int,
    box_height: int,
    rng: random.Random,
    canny_low: int,
    canny_high: int,
    dilate_px: int,
    margin_px: int,
    max_obstacle_fraction: float,
    stride_px: int,
    fill_contours: bool,
    min_contour_area_fraction: float,
) -> Optional[CannyPlacement]:
    W, H = image.size
    bw = int(box_width) + 2 * int(margin_px)
    bh = int(box_height) + 2 * int(margin_px)
    if bw > W or bh > H:
        return None

    edges, obstacle = _canny_obstacle_map(
        image,
        canny_low,
        canny_high,
        dilate_px,
        fill_contours,
        min_contour_area_fraction,
    )
    edge_binary = (edges > 0).astype(np.float32)
    obstacle_binary = (obstacle > 0).astype(np.float32)
    edge_integral = cv2.integral(edge_binary)
    obstacle_integral = cv2.integral(obstacle_binary)

    stride = max(1, int(stride_px))
    xs = list(range(0, max(1, W - bw + 1), stride))
    ys = list(range(0, max(1, H - bh + 1), stride))
    if xs[-1] != W - bw:
        xs.append(W - bw)
    if ys[-1] != H - bh:
        ys.append(H - bh)

    candidates: List[Tuple[float, float, float, int, int]] = []
    cx, cy = W / 2.0, H / 2.0
    norm = max(1.0, math.hypot(cx, cy))
    for y in ys:
        for x in xs:
            obs = _integral_fraction(obstacle_integral, x, y, bw, bh)
            if obs > float(max_obstacle_fraction):
                continue
            edge = _integral_fraction(edge_integral, x, y, bw, bh)
            # Mild border preference: if equally clean, place text away from the
            # central object cluster without hardcoding a specific edge.
            rcx, rcy = x + bw / 2.0, y + bh / 2.0
            center_distance = math.hypot(rcx - cx, rcy - cy) / norm
            score = obs * 10.0 + edge * 2.0 - 0.08 * center_distance
            candidates.append((score, obs, edge, x, y))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    top = candidates[: min(12, len(candidates))]
    score, obs, edge, x, y = rng.choice(top)
    del score
    return CannyPlacement(
        x=int(x + margin_px),
        y=int(y + margin_px),
        box_width=int(box_width),
        box_height=int(box_height),
        edge_fraction=float(edge),
        obstacle_fraction=float(obs),
        canny_low=int(canny_low),
        canny_high=int(canny_high),
    )


def _local_luminance(image: Image.Image, x: int, y: int, w: int, h: int) -> float:
    crop = np.asarray(image.crop((x, y, x + w, y + h)).convert("RGB"), dtype=np.float32) / 255.0
    if crop.size == 0:
        return 0.5
    return float((0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]).mean())


def _choose_text_color(
    image: Image.Image,
    placement: CannyPlacement,
    fact: ClevrFact,
    rng: random.Random,
    colored_probability: float,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool, Optional[bool]]:
    if rng.random() < float(colored_probability):
        congruent: Optional[bool] = None
        if fact.color in COLOR_RGB and rng.random() < 0.5:
            chosen = str(fact.color)
            congruent = True
        else:
            choices = [c for c in CLEVR_COLORS if c != fact.color]
            chosen = rng.choice(choices)
            congruent = False if fact.color else None
        fg = COLOR_RGB[chosen]
        # Strong opposite-polarity outline. This includes the requested
        # colored-text-with-white-border family without making it universal.
        lum = 0.2126 * fg[0] + 0.7152 * fg[1] + 0.0722 * fg[2]
        stroke = (245, 245, 245) if lum < 145 else (20, 20, 20)
        return fg, stroke, True, congruent

    lum = _local_luminance(
        image,
        placement.x,
        placement.y,
        placement.box_width,
        placement.box_height,
    )
    if lum >= 0.5:
        return (15, 15, 15), (245, 245, 245), False, None
    return (245, 245, 245), (15, 15, 15), False, None


def _paste_text(
    base: Image.Image,
    text: str,
    font_path: str,
    font_size: int,
    stroke_width: int,
    placement: CannyPlacement,
    fg: Tuple[int, int, int],
    stroke: Tuple[int, int, int],
) -> Tuple[Image.Image, Image.Image]:
    font = ImageFont.truetype(font_path, int(font_size))
    bbox = _font_bbox(text, font, stroke_width)
    tw = int(bbox[2] - bbox[0])
    th = int(bbox[3] - bbox[1])
    x = int(placement.x + max(0, (placement.box_width - tw) // 2) - bbox[0])
    y = int(placement.y + max(0, (placement.box_height - th) // 2) - bbox[1])
    result = base.convert("RGB").copy()
    mask = Image.new("L", base.size, 0)
    d = ImageDraw.Draw(result)
    dm = ImageDraw.Draw(mask)
    d.text(
        (x, y), text, font=font, fill=fg,
        stroke_width=int(stroke_width), stroke_fill=stroke,
    )
    dm.text(
        (x, y), text, font=font, fill=255,
        stroke_width=int(stroke_width), stroke_fill=255,
    )
    return result, mask


def render_fact_pair_canny(
    base: Image.Image,
    true_fact: ClevrFact,
    false_fact: ClevrFact,
    font_paths: Sequence[str],
    rng: random.Random,
    colored_probability: float,
    canny_low: int,
    canny_high: int,
    canny_dilate_px: int,
    placement_margin_px: int,
    max_obstacle_fraction: float,
    stride_px: int,
    fill_contours: bool,
    min_contour_area_fraction: float,
    min_font_size: int,
    max_font_size: int,
) -> Optional[RenderPair]:
    if not font_paths:
        raise RuntimeError("CLEVR property curriculum requires at least one TrueType font")
    font_path = rng.choice(list(font_paths))
    stroke_width = rng.choices([1, 2], weights=[0.8, 0.2], k=1)[0]

    # Log-uniformly sample scale, then shrink only if the Canny-free placement
    # cannot fit. This avoids using any stale metadata geometry whatsoever.
    sampled = int(round(math.exp(rng.uniform(math.log(max(4, min_font_size)), math.log(max(min_font_size, max_font_size))))))
    sizes = []
    size = max(min_font_size, min(max_font_size, sampled))
    while size >= int(min_font_size):
        sizes.append(size)
        next_size = max(int(min_font_size) - 1, int(math.floor(size * 0.86)))
        if next_size >= size:
            next_size = size - 1
        size = next_size

    for font_size in sizes:
        try:
            true_mask = _render_text_mask(true_fact.visible_text.upper(), font_path, font_size, stroke_width)
            false_mask = _render_text_mask(false_fact.visible_text.upper(), font_path, font_size, stroke_width)
        except Exception:
            continue
        box_w = max(true_mask.width, false_mask.width)
        box_h = max(true_mask.height, false_mask.height)
        placement = find_canny_placement(
            base,
            box_w,
            box_h,
            rng,
            canny_low,
            canny_high,
            canny_dilate_px,
            placement_margin_px,
            max_obstacle_fraction,
            stride_px,
            fill_contours,
            min_contour_area_fraction,
        )
        if placement is None:
            continue

        true_fg, true_stroke, colored, congruent = _choose_text_color(
            base, placement, true_fact, rng, colored_probability
        )
        # Keep the counterfactual style matched. If colored mode was chosen, use
        # the same raster color rather than leaking semantic truth via style.
        false_fg, false_stroke = true_fg, true_stroke
        true_img, true_full_mask = _paste_text(
            base,
            true_fact.visible_text.upper(),
            font_path,
            font_size,
            stroke_width,
            placement,
            true_fg,
            true_stroke,
        )
        false_img, false_full_mask = _paste_text(
            base,
            false_fact.visible_text.upper(),
            font_path,
            font_size,
            stroke_width,
            placement,
            false_fg,
            false_stroke,
        )
        return RenderPair(
            true_image=true_img,
            true_mask=true_full_mask,
            false_image=false_img,
            false_mask=false_full_mask,
            placement=placement,
            font_name=Path(font_path).name,
            font_size=int(font_size),
            stroke_width=int(stroke_width),
            true_color=true_fg,
            false_color=false_fg,
            colored_style=colored,
            congruent_color=congruent,
        )
    return None


class ClevrPropertyBindingBuilder:
    def __init__(
        self,
        image_root: Path,
        metadata_jsonl: Path,
        split: str,
        font_paths: Sequence[str],
        colored_text_probability: float = 0.15,
        ood_color_probability: float = 0.08,
        standalone_count_probability: float = 0.15,
        canny_low: int = 45,
        canny_high: int = 120,
        canny_dilate_px: int = 3,
        placement_margin_px: int = 4,
        max_obstacle_fraction: float = 0.025,
        placement_stride_px: int = 3,
        fill_contours: bool = True,
        min_contour_area_fraction: float = 0.001,
        min_font_size: int = 8,
        max_font_size: int = 42,
    ):
        if cv2 is None:
            raise RuntimeError(
                "CLEVR property curriculum requires OpenCV (cv2) for Canny placement. "
                "Install opencv-python or set datasets.clevr_property_binding.include=false."
            )
        self.image_root = Path(image_root)
        self.metadata_jsonl = Path(metadata_jsonl)
        self.split = str(split).lower()
        if self.split == "validation":
            self.split = "val"
        self.font_paths = [str(path) for path in font_paths]
        self.colored_text_probability = float(colored_text_probability)
        self.ood_color_probability = float(ood_color_probability)
        self.standalone_count_probability = float(standalone_count_probability)
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.canny_dilate_px = int(canny_dilate_px)
        self.placement_margin_px = int(placement_margin_px)
        self.max_obstacle_fraction = float(max_obstacle_fraction)
        self.placement_stride_px = int(placement_stride_px)
        self.fill_contours = bool(fill_contours)
        self.min_contour_area_fraction = float(min_contour_area_fraction)
        self.min_font_size = int(min_font_size)
        self.max_font_size = int(max_font_size)

        if not self.image_root.is_dir():
            raise FileNotFoundError(f"CLEVR square image root missing: {self.image_root}")
        if not self.metadata_jsonl.is_file():
            raise FileNotFoundError(f"CLEVR metadata JSONL missing: {self.metadata_jsonl}")
        rows = _normalize_rows(read_jsonl(self.metadata_jsonl), self.split)
        self.rows: List[Dict[str, Any]] = [
            row for row in rows if (self.image_root / row["image_filename"]).is_file()
        ]
        if not self.rows:
            raise RuntimeError(
                f"No CLEVR {self.split} rows with local images under {self.image_root}"
            )
        self._coverage_cache = self._build_coverage()

    def _build_coverage(self) -> Dict[str, Any]:
        fact_kind_counts: Counter = Counter()
        image_object_counts: Counter = Counter()
        colors: Counter = Counter()
        shapes: Counter = Counter()
        for row in self.rows:
            inv = _inventory(row["objects"])
            image_object_counts[int(inv["n_objects"])] += 1
            colors.update(inv["color_counts"])
            shapes.update(inv["shape_counts"])
            for fact in _base_true_facts(inv):
                if fact.kind == "count_word" and self.standalone_count_probability <= 0:
                    continue
                fact_kind_counts[fact.kind] += 1
        return {
            "split": self.split,
            "rows": len(self.rows),
            "image_root": str(self.image_root),
            "metadata_jsonl": str(self.metadata_jsonl),
            "fact_kind_counts": dict(sorted(fact_kind_counts.items())),
            "scene_object_count_histogram": {str(k): v for k, v in sorted(image_object_counts.items())},
            "object_color_counts": dict(sorted(colors.items())),
            "object_shape_counts": dict(sorted(shapes.items())),
            "uses_metadata_geometry": False,
            "placement": "Canny edges + filled/dilated contours on the already-square local image",
        }

    def coverage(self) -> Dict[str, Any]:
        return dict(self._coverage_cache)

    def _load_image(self, row: Mapping[str, Any]) -> Image.Image:
        return Image.open(self.image_root / str(row["image_filename"])).convert("RGB")

    def _choose_fact_pair(self, inv: Mapping[str, Any], rng: random.Random) -> Tuple[ClevrFact, ClevrFact]:
        facts = _base_true_facts(inv)
        weighted: List[Tuple[ClevrFact, float]] = []
        kind_weights = {
            "color": 1.0,
            "shape": 0.8,
            "color_shape": 1.2,
            "article_color_shape": 0.8,
            "count_shape": 1.5,
            "count_color": 1.2,
            "count_color_shape": 1.8,
            "count_word": max(0.0, self.standalone_count_probability),
        }
        for fact in facts:
            false_fact = _false_fact_for(fact, inv, rng, self.ood_color_probability)
            if false_fact is not None:
                weighted.append((fact, kind_weights.get(fact.kind, 1.0)))
        if not weighted:
            raise RuntimeError("CLEVR row has no usable fact/counterfactual pair")
        true_fact = rng.choices(
            [item[0] for item in weighted],
            weights=[item[1] for item in weighted],
            k=1,
        )[0]
        false_fact = _false_fact_for(true_fact, inv, rng, self.ood_color_probability)
        if false_fact is None:
            raise RuntimeError("Failed to regenerate CLEVR counterfactual")
        return true_fact, false_fact

    def sample(self, rng: random.Random) -> ClevrPacketData:
        for _attempt in range(48):
            row = rng.choice(self.rows)
            inv = _inventory(row["objects"])
            try:
                true_fact, false_fact = self._choose_fact_pair(inv, rng)
            except Exception:
                continue
            base = self._load_image(row)
            render = render_fact_pair_canny(
                base=base,
                true_fact=true_fact,
                false_fact=false_fact,
                font_paths=self.font_paths,
                rng=rng,
                colored_probability=self.colored_text_probability,
                canny_low=self.canny_low,
                canny_high=self.canny_high,
                canny_dilate_px=self.canny_dilate_px,
                placement_margin_px=self.placement_margin_px,
                max_obstacle_fraction=self.max_obstacle_fraction,
                stride_px=self.placement_stride_px,
                fill_contours=self.fill_contours,
                min_contour_area_fraction=self.min_contour_area_fraction,
                min_font_size=self.min_font_size,
                max_font_size=self.max_font_size,
            )

            blank = Image.new("L", base.size, 0)
            if render is None:
                # Cluttered/no-safe-space case: keep the clean image as additive
                # ordinary content + explicit no-text evidence. Never obscure an
                # object merely to force a reading sample.
                captions = [
                    true_fact.semantic_text,
                    f"<notext> {true_fact.semantic_text}",
                    "<text> <null>",
                ]
                return ClevrPacketData(
                    images=[base],
                    masks=[blank],
                    mask_weights=[1.0],
                    present_targets=[0.0],
                    readable_targets=[0.0],
                    captions=captions,
                    positive_pairs=[(0, 0), (0, 1), (0, 2)],
                    source_triplets=[],
                    auto_triplets=[],
                    invariance_pairs=[],
                    metadata={
                        "source": "clevr_property",
                        "split": self.split,
                        "image_filename": row["image_filename"],
                        "placement_found": False,
                        "true_fact": true_fact.__dict__,
                        "false_fact": false_fact.__dict__,
                        "n_objects": int(inv["n_objects"]),
                    },
                )

            captions = [
                true_fact.semantic_text,                                           # 0 true image fact
                f"<notext> {true_fact.semantic_text}",                            # 1 content-only true fact
                f"<text> {true_fact.visible_text}",                              # 2 literal true overlay
                f"<text> {false_fact.visible_text}",                             # 3 literal false overlay (still real text)
                false_fact.semantic_text,                                          # 4 semantic counterfactual
                f'{true_fact.semantic_text} with the text "{true_fact.visible_text}"',   # 5
                f'{true_fact.semantic_text} with the text "{false_fact.visible_text}"',  # 6
                "<text> <null>",                                                  # 7
            ]
            positive_pairs = [
                (0, 0), (0, 1),
                (1, 0), (1, 1), (1, 2), (1, 5),
                (2, 0), (2, 1), (2, 3), (2, 6),
                (0, 7),
            ]
            source_triplets = [
                (2, 1, [0, 2]),
                (3, 2, [0, 1]),
                (7, 0, [1, 2]),
            ]
            auto_triplets = [
                # Correct scene fact must beat the semantically false content
                # implied by the visible counterfactual. The literal candidate
                # itself remains positive through caption 3.
                (2, 0, 4),
                (1, 0, 4),
            ]
            return ClevrPacketData(
                images=[base, render.true_image, render.false_image],
                masks=[blank, render.true_mask, render.false_mask],
                mask_weights=[1.0, 1.0, 1.0],
                present_targets=[0.0, 1.0, 1.0],
                readable_targets=[0.0, 1.0, 1.0],
                captions=captions,
                positive_pairs=positive_pairs,
                source_triplets=source_triplets,
                auto_triplets=auto_triplets,
                invariance_pairs=[(0, 1), (0, 2)],
                metadata={
                    "source": "clevr_property",
                    "split": self.split,
                    "image_filename": row["image_filename"],
                    "placement_found": True,
                    "true_fact": true_fact.__dict__,
                    "false_fact": false_fact.__dict__,
                    "n_objects": int(inv["n_objects"]),
                    "font_name": render.font_name,
                    "font_size": render.font_size,
                    "stroke_width": render.stroke_width,
                    "colored_style": render.colored_style,
                    "congruent_color": render.congruent_color,
                    "text_rgb": list(render.true_color),
                    "placement": render.placement.__dict__,
                },
            )
        raise RuntimeError("CLEVR property curriculum failed after 48 retries")

    def save_previews(self, out_dir: Path, count: int, seed: int) -> List[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(int(seed))
        saved: List[Path] = []
        for index in range(max(0, int(count))):
            packet = self.sample(rng)
            image_index = 1 if len(packet.images) >= 3 else 0
            image = packet.images[image_index].copy().convert("RGB")
            draw = ImageDraw.Draw(image)
            # Visualize Canny edges only in preview copies. Training images are
            # untouched. Cyan pixels = detected edge; yellow rectangle = chosen
            # free text box when placement succeeded.
            edges, _obstacle = _canny_obstacle_map(
                packet.images[0], self.canny_low, self.canny_high,
                self.canny_dilate_px, self.fill_contours,
                self.min_contour_area_fraction,
            )
            arr = np.asarray(image, dtype=np.uint8).copy()
            edge_mask = edges > 0
            arr[edge_mask] = np.array([0, 220, 255], dtype=np.uint8)
            image = Image.fromarray(arr, mode="RGB")
            draw = ImageDraw.Draw(image)
            placement = packet.metadata.get("placement")
            if isinstance(placement, Mapping):
                x = int(placement["x"])
                y = int(placement["y"])
                w = int(placement["box_width"])
                h = int(placement["box_height"])
                draw.rectangle((x, y, x + w, y + h), outline=(255, 230, 0), width=2)
            name = f"clevr_property_{index:02d}_{packet.metadata['image_filename']}"
            path = out_dir / name
            image.save(path)
            saved.append(path)
        return saved
