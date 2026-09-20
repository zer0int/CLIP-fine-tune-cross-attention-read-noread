#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Frozen-CLIP soft <text> prefix addressability test.

Purpose
-------
Test whether one learned soft prefix vector can tell CLIP's frozen text tower to
query the visual written-word mode already present in the frozen image tower.
The image examples are constructed without pixel occlusion:

    D_w = Conv1(render(word)) - Conv1(exact_empty_background)
    Conv1(image_with_word) = Conv1(real_image) + alpha * D_w

Only one text-side vector is trained. Every pretrained CLIP parameter remains
frozen, so ordinary CLIP prompts and embeddings are unchanged by construction.

The default word bank contains:
  * ImageNet class names used as deliberately incorrect inscriptions;
  * related/slang words such as feline and caturday;
  * non-object meme words such as lmaooo, hahaha, lol, mfw, and owo;
  * common object/animal words not exactly present in the canonical ImageNet
    label list.

A held-out word split tests whether the learned vector is a compositional mode
operator rather than a lookup table.

Example
-------
python train_clip_soft_text_token_imagenet.py ^
  --imagenet_root <path-to-ImageNet> ^
  --out_dir out_soft_text_token

Quick pipeline smoke test
-------------------------
python train_clip_soft_text_token_imagenet.py ^
  --imagenet_root <path-to-ImageNet> ^
  --out_dir out_soft_text_token_smoke ^
  --smoke

Dependencies
------------
torch, torchvision, pillow, numpy, OpenAI CLIP imported as ``clip``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    # Append rather than prepend: this stage must keep importing the vanilla
    # legacy CLIP packages located beside this script.
    sys.path.append(str(PROJECT_ROOT))

from training_support.diagnostics.training_precision_diagnostics import (
    PrecisionDiagnostics,
    PrecisionDiagnosticsConfig,
)
from training_support.font_discovery import find_font_files
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

from handwriting_overlay_utils import (
    HandwritingOverlayIndex,
    render_handwriting_crop,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


# -----------------------------------------------------------------------------
# Default word banks
# -----------------------------------------------------------------------------

# Explicitly requested and useful canonical ImageNet words are forced into the
# sampled ImageNet word set when present in the local label map.
DEFAULT_FORCE_IMAGENET_WORDS = [
    "school bus",
    "goldfinch",
    "tabby",
    "bee",
    "fire salamander",
    "sports car",
    "espresso",
    "volcano",
]

# Each tuple is (word, split, avoid-keywords).  Avoid-keywords prevent an
# accidental semantically matching base image in this deliberately mismatched
# first curriculum.
DEFAULT_RELATED_WORDS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("feline", "train", ("cat", "lynx", "tiger", "lion", "leopard", "jaguar", "cougar", "cheetah")),
    ("kitty", "train", ("cat", "lynx", "tiger", "lion", "leopard", "jaguar", "cougar", "cheetah")),
    ("canine", "train", ("dog", "wolf", "fox", "coyote", "dingo", "jackal")),
    ("doggo", "train", ("dog", "wolf", "fox", "coyote", "dingo", "jackal")),
    ("birb", "train", ("bird", "finch", "eagle", "owl", "duck", "hen", "cock", "parrot", "jay")),
    ("avian", "train", ("bird", "finch", "eagle", "owl", "duck", "hen", "cock", "parrot", "jay")),
    ("aquatic", "train", ("fish", "shark", "whale", "dolphin", "seal", "ray")),
    ("winged", "train", ("bird", "butterfly", "moth", "bee", "fly", "dragonfly")),
    ("striped", "train", ()),
    ("spotted", "train", ()),
    ("predator", "train", ()),
    ("house pet", "train", ("cat", "dog", "hamster", "rabbit", "parrot")),
    ("caturday", "heldout", ("cat", "lynx", "tiger", "lion", "leopard", "jaguar", "cougar", "cheetah")),
    ("kittenish", "heldout", ("cat", "lynx", "tiger", "lion", "leopard", "jaguar", "cougar", "cheetah")),
    ("sharky", "heldout", ("shark",)),
    ("insectoid", "heldout", ("insect", "bee", "wasp", "fly", "beetle", "moth", "butterfly")),
    ("reptilian", "heldout", ("snake", "lizard", "turtle", "crocodile", "alligator")),
    ("amphibian", "heldout", ("frog", "toad", "salamander", "newt")),
    ("four-legged", "heldout", ()),
    ("sea creature", "heldout", ("fish", "shark", "whale", "dolphin", "seal", "ray")),
]

DEFAULT_MEME_WORDS: List[Tuple[str, str]] = [
    ("hahaha", "train"),
    ("lol", "train"),
    ("uwu", "train"),
    ("bruh", "train"),
    ("yeet", "train"),
    ("sus", "train"),
    ("nope", "train"),
    ("yikes", "train"),
    ("omg", "train"),
    ("smh", "train"),
    ("rofl", "train"),
    ("oof", "train"),
    ("based", "train"),
    ("cringe", "train"),
    ("mood", "train"),
    ("lmaooo", "heldout"),
    ("mfw", "heldout"),
    ("owo", "heldout"),
    ("tfw", "heldout"),
    ("hehehe", "heldout"),
    ("xD", "heldout"),
    ("kthx", "heldout"),
    ("welp", "heldout"),
]

DEFAULT_OOD_COMMON_WORDS: List[Tuple[str, str]] = [
    ("capybara", "train"),
    ("axolotl", "train"),
    ("quokka", "train"),
    ("pangolin", "train"),
    ("manatee", "train"),
    ("wombat", "train"),
    ("meerkat", "train"),
    ("narwhal", "train"),
    ("ferret", "train"),
    ("hamster", "train"),
    ("alpaca", "train"),
    ("traffic cone", "train"),
    ("shopping cart", "train"),
    ("laptop", "train"),
    ("smartphone", "train"),
    ("headphones", "train"),
    ("scooter", "train"),
    ("coffee mug", "train"),
    ("water bottle", "train"),
    ("platypus", "heldout"),
    ("sloth", "heldout"),
    ("marmot", "heldout"),
    ("koala", "heldout"),
    ("red panda", "heldout"),
    ("toaster", "heldout"),
    ("microwave", "heldout"),
    ("desk lamp", "heldout"),
    ("backpack", "heldout"),
    ("doorbell", "heldout"),
    ("mailbox", "heldout"),
]


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class WordRecord:
    word_id: int
    word: str
    category: str
    split: str  # train or heldout
    source_wnid: str = ""
    avoid_keywords: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class RenderSpec:
    spec_id: int
    font_index: int
    font_scale: float
    x_frac: float
    y_frac: float
    rotation_deg: float
    text_rgb: Tuple[int, int, int]
    stroke_width: int
    stroke_rgb: Tuple[int, int, int]
    background_rgb: Tuple[int, int, int]
    seed: int


@dataclass
class ExampleRecord:
    example_id: int
    dataset_split: str  # train, seen_word_val, heldout_word_val
    word_id: int
    word: str
    category: str
    word_split: str
    target_wnid: str
    base_path: str
    base_wnid: str
    base_label: str
    render_spec_id: int
    alpha: float


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return out or "item"


def parse_semicolon_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(";") if x.strip()]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def torch_load_compat(path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def list_images_shallow(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# -----------------------------------------------------------------------------
# ImageNet class-label resolution
# -----------------------------------------------------------------------------


def parse_class_index_json(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    wnid_to_label: Dict[str, str] = {}
    label_to_wnid: Dict[str, str] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            if re.fullmatch(r"n\d{8}", str(key)):
                wnid = str(key)
                if isinstance(value, str):
                    label = value
                elif isinstance(value, (list, tuple)) and value:
                    label = str(value[-1])
                elif isinstance(value, dict):
                    label = str(value.get("label") or value.get("name") or wnid)
                else:
                    label = wnid
            elif isinstance(value, (list, tuple)) and len(value) >= 2 and re.fullmatch(r"n\d{8}", str(value[0])):
                wnid = str(value[0])
                label = str(value[1])
            elif isinstance(value, dict) and re.fullmatch(r"n\d{8}", str(value.get("wnid", ""))):
                wnid = str(value["wnid"])
                label = str(value.get("label") or value.get("name") or wnid)
            else:
                continue
            wnid_to_label[wnid] = label
            label_to_wnid.setdefault(normalize_phrase(label), wnid)

    if not wnid_to_label:
        raise ValueError(f"Could not parse any wnid/label pairs from {path}")
    return wnid_to_label, label_to_wnid


def resolve_imagenet_labels(train_root: Path, class_index_json: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, str]]:
    wnids = sorted(p.name for p in train_root.iterdir() if p.is_dir() and re.fullmatch(r"n\d{8}", p.name))
    if not wnids:
        raise RuntimeError(f"No ImageNet wnid class directories found under {train_root}")

    if class_index_json is not None:
        wnid_to_label, label_to_wnid = parse_class_index_json(class_index_json)
        missing = [w for w in wnids if w not in wnid_to_label]
        if missing:
            print(f"[labels] warning: mapping lacks {len(missing)} local wnids; they will be skipped")
        return wnid_to_label, label_to_wnid

    # torchvision ships the canonical 1000 category names locally.  The
    # ILSVRC2012 train wnid directories sort in the same canonical class order.
    try:
        from torchvision.models import ResNet50_Weights

        categories = list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
    except Exception as exc:
        raise RuntimeError(
            "Could not load torchvision's ImageNet category metadata. Supply "
            "--class_index_json pointing to imagenet_class_index.json."
        ) from exc

    if len(wnids) != 1000 or len(categories) != 1000:
        raise RuntimeError(
            f"Automatic torchvision mapping requires exactly 1000 wnid folders; found {len(wnids)}. "
            "Supply --class_index_json explicitly."
        )

    wnid_to_label = dict(zip(wnids, categories))
    sanity = {
        "n01440764": "tench",
        "n01531178": "goldfinch",
        "n02123045": "tabby",
    }
    for wnid, expected in sanity.items():
        actual = normalize_phrase(wnid_to_label.get(wnid, ""))
        if expected not in actual:
            raise RuntimeError(
                "torchvision category order does not match the local sorted wnid folders: "
                f"expected {wnid} -> {expected!r}, got {actual!r}. Supply --class_index_json."
            )
    label_to_wnid: Dict[str, str] = {}
    for wnid, label in wnid_to_label.items():
        label_to_wnid.setdefault(normalize_phrase(label), wnid)
    return wnid_to_label, label_to_wnid


def organized_imagenet_root(root: Path, known_wnids: Sequence[str]) -> bool:
    if not root.is_dir():
        return False
    probe = list(known_wnids)[:20]
    return any((root / wnid).is_dir() for wnid in probe)


# -----------------------------------------------------------------------------
# Word bank
# -----------------------------------------------------------------------------


def load_word_bank_json(path: Path) -> List[WordRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--word_bank_json must contain a JSON list")
    out: List[WordRecord] = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            item = {"word": item, "category": "custom", "split": "train"}
        if not isinstance(item, dict) or not item.get("word"):
            raise ValueError(f"Invalid word-bank entry at index {idx}: {item!r}")
        split = str(item.get("split", "train")).lower()
        if split not in {"train", "heldout"}:
            raise ValueError(f"Invalid split {split!r} for word {item['word']!r}")
        avoid = tuple(str(x).lower() for x in item.get("avoid_keywords", []))
        out.append(
            WordRecord(
                word_id=-1,
                word=str(item["word"]).strip(),
                category=str(item.get("category", "custom")),
                split=split,
                source_wnid=str(item.get("source_wnid", "")),
                avoid_keywords=avoid,
            )
        )
    return deduplicate_and_number_words(out)


def deduplicate_and_number_words(records: Sequence[WordRecord]) -> List[WordRecord]:
    out: List[WordRecord] = []
    seen = set()
    for rec in records:
        key = normalize_phrase(rec.word)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            WordRecord(
                word_id=len(out),
                word=rec.word,
                category=rec.category,
                split=rec.split,
                source_wnid=rec.source_wnid,
                avoid_keywords=tuple(rec.avoid_keywords),
            )
        )
    return out


def build_default_word_bank(
    wnid_to_label: Mapping[str, str],
    label_to_wnid: Mapping[str, str],
    num_imagenet_words: int,
    heldout_fraction: float,
    force_imagenet_words: Sequence[str],
    seed: int,
    smoke: bool,
) -> List[WordRecord]:
    rng = random.Random(seed + 1129)
    canonical = [(wnid, label) for wnid, label in wnid_to_label.items() if label.strip()]
    by_norm = {normalize_phrase(label): (wnid, label) for wnid, label in canonical}

    selected: List[Tuple[str, str]] = []
    selected_keys = set()
    for phrase in force_imagenet_words:
        norm_phrase = normalize_phrase(phrase)
        item = by_norm.get(norm_phrase)
        if item is None:
            matches = [(wnid, label) for wnid, label in canonical if norm_phrase in normalize_phrase(label)]
            if len(matches) == 1:
                item = matches[0]
        if item is not None and normalize_phrase(item[1]) not in selected_keys:
            selected.append(item)
            selected_keys.add(normalize_phrase(item[1]))

    remaining = [(w, l) for w, l in canonical if normalize_phrase(l) not in selected_keys]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, num_imagenet_words - len(selected))])
    selected = selected[:num_imagenet_words]

    rng.shuffle(selected)
    heldout_n = max(1, int(round(len(selected) * heldout_fraction))) if selected else 0
    heldout_set = {wnid for wnid, _ in selected[:heldout_n]}

    records: List[WordRecord] = []
    for wnid, label in selected:
        records.append(
            WordRecord(
                word_id=-1,
                word=label,
                category="imagenet_class",
                split="heldout" if wnid in heldout_set else "train",
                source_wnid=wnid,
                avoid_keywords=tuple(normalize_phrase(label).split()),
            )
        )

    related = DEFAULT_RELATED_WORDS
    memes = DEFAULT_MEME_WORDS
    ood = DEFAULT_OOD_COMMON_WORDS
    if smoke:
        related = related[:4] + [x for x in related if x[0] in {"caturday", "sharky"}]
        memes = [x for x in memes if x[0] in {"hahaha", "lol", "lmaooo", "mfw", "owo"}]
        ood = [x for x in ood if x[0] in {"capybara", "traffic cone", "platypus", "toaster"}]

    for word, split, avoid in related:
        records.append(WordRecord(-1, word, "related_word", split, "", tuple(avoid)))
    for word, split in memes:
        records.append(WordRecord(-1, word, "meme_nonobject", split, "", ()))

    canonical_norms = set(label_to_wnid.keys())
    for word, split in ood:
        # Preserve the requested common words, but label exact canonical matches
        # honestly instead of pretending they are out of distribution.
        category = "ood_common" if normalize_phrase(word) not in canonical_norms else "common_exact_imagenet_label"
        source = label_to_wnid.get(normalize_phrase(word), "")
        records.append(
            WordRecord(
                -1,
                word,
                category,
                split,
                source,
                tuple(normalize_phrase(word).split()),
            )
        )

    records = deduplicate_and_number_words(records)
    if not any(r.split == "train" for r in records) or not any(r.split == "heldout" for r in records):
        raise RuntimeError("Word bank must contain both train and heldout words")
    return records


def load_curated_overlay_words(
    overlay_dir: Path,
    *,
    heldout_fraction: float,
    seed: int,
) -> List[WordRecord]:
    """Load the curated read-stage bank while holding out whole ortho families."""
    csv_path = overlay_dir / "overlay_words_gmp_diverse.csv"
    txt_path = overlay_dir / "overlay_words_gmp_diverse.txt"
    if csv_path.is_file():
        rows = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                word = str(row.get("word") or "").strip()
                if word:
                    rows.append((word, str(row.get("overlay_stratum") or ""), index))
    elif txt_path.is_file():
        rows = [
            (line.strip(), "", index)
            for index, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines())
            if line.strip()
        ]
    else:
        raise FileNotFoundError(f"Missing curated overlay bank under {overlay_dir}")

    words = []
    seen = set()
    for word, stratum, index in rows:
        key = normalize_phrase(word)
        if key and key not in seen:
            seen.add(key)
            words.append((word, stratum, index))

    parent = {word: word for word, _, _ in words}

    def find(word: str) -> str:
        while parent[word] != word:
            parent[word] = parent[parent[word]]
            word = parent[word]
        return word

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_path = overlay_dir / "orthographic_pairs_selected.csv"
    if pair_path.is_file():
        valid = set(parent)
        with pair_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                a, b = str(row.get("word_a") or ""), str(row.get("word_b") or "")
                if a in valid and b in valid and a != b:
                    union(a, b)

    groups: Dict[str, List[str]] = {}
    for word, _, _ in words:
        groups.setdefault(find(word), []).append(word)
    units = list(groups.values())
    random.Random(seed + 8081).shuffle(units)
    target = int(round(len(words) * max(0.0, min(0.9, heldout_fraction))))
    heldout: set[str] = set()
    for unit in units:
        if len(heldout) >= target:
            break
        heldout.update(unit)
    if not heldout and words:
        heldout.update(units[0])
    if len(heldout) >= len(words) and units:
        heldout.difference_update(units[-1])

    return [
        WordRecord(
            word_id=-1,
            word=word,
            category=f"curated_overlay:{stratum or 'unknown'}",
            split="heldout" if word in heldout else "train",
            source_wnid="",
            avoid_keywords=tuple(normalize_phrase(word).split()),
        )
        for word, stratum, _ in words
    ]


def merge_word_banks(
    curated: Sequence[WordRecord],
    extras: Sequence[WordRecord],
) -> List[WordRecord]:
    """Curated split/family assignment wins; useful ImageNet metadata is merged."""
    extras_by_key = {normalize_phrase(record.word): record for record in extras}
    merged: List[WordRecord] = []
    used = set()
    for record in curated:
        key = normalize_phrase(record.word)
        extra = extras_by_key.get(key)
        merged.append(
            WordRecord(
                -1,
                record.word,
                record.category,
                record.split,
                extra.source_wnid if extra else record.source_wnid,
                extra.avoid_keywords if extra and extra.avoid_keywords else record.avoid_keywords,
            )
        )
        used.add(key)
    merged.extend(record for record in extras if normalize_phrase(record.word) not in used)
    return deduplicate_and_number_words(merged)


def augment_soft_word_bank_with_handwriting(
    records: Sequence[WordRecord],
    handwriting_index: Optional[HandwritingOverlayIndex],
    *,
    heldout_fraction: float,
    seed: int,
) -> List[WordRecord]:
    """Union exact handwriting prompt words into the soft-token curriculum."""
    base = list(records)
    if handwriting_index is None or not handwriting_index.available:
        return deduplicate_and_number_words(base)
    existing = {normalize_phrase(record.word) for record in base}
    additions: List[Tuple[str, str]] = []
    for normalized in handwriting_index.words():
        key = normalize_phrase(normalized)
        if not key or key in existing:
            continue
        samples = handwriting_index.samples(normalized)
        surface = samples[0].word.strip() if samples else str(normalized).strip()
        if surface:
            additions.append((key, surface))
            existing.add(key)
    additions.sort(key=lambda item: item[0])
    shuffled = list(additions)
    random.Random(int(seed) + 0x48414E44).shuffle(shuffled)
    heldout_n = int(round(len(shuffled) * max(0.0, min(0.9, float(heldout_fraction)))))
    if len(shuffled) > 1:
        heldout_n = max(1, min(len(shuffled) - 1, heldout_n))
    else:
        heldout_n = 0
    heldout = {key for key, _surface in shuffled[:heldout_n]}
    for key, surface in additions:
        base.append(
            WordRecord(
                -1,
                surface,
                "handwriting_manifest",
                "heldout" if key in heldout else "train",
                "",
                tuple(key.split()),
            )
        )
    return deduplicate_and_number_words(base)


# -----------------------------------------------------------------------------
# Rendered word controls
# -----------------------------------------------------------------------------


_XHEIGHT = "aceimnorsuvwxz"
_ASCENDERS = "bdfhklt"
_DESCENDERS = "gjpqy"
_NARROW = "il1"
_WIDE = "mw"
_DIGITS = "023456789"


def char_pool(ch: str) -> str:
    low = ch.lower()
    if low in _NARROW:
        return _NARROW
    if low in _WIDE:
        return _WIDE
    if low in _ASCENDERS:
        return _ASCENDERS
    if low in _DESCENDERS:
        return _DESCENDERS
    if low in _XHEIGHT:
        return _XHEIGHT
    if low.isdigit():
        return _DIGITS
    if low.isalpha():
        return _XHEIGHT + _ASCENDERS + _DESCENDERS
    return ch


def matched_pseudoword(phrase: str, seed: int) -> str:
    rng = random.Random(seed)
    out: List[str] = []
    for ch in phrase:
        if ch.isspace() or not (ch.isalpha() or ch.isdigit()):
            out.append(ch)
            continue
        pool = char_pool(ch)
        choices = [x for x in pool if x.lower() != ch.lower()]
        repl = rng.choice(choices or list(pool))
        out.append(repl.upper() if ch.isupper() else repl)
    candidate = "".join(out)
    return candidate[::-1] if candidate.lower() == phrase.lower() else candidate


def discover_fonts(extra: Sequence[str]) -> List[Path]:
    return [Path(value) for value in find_font_files(extra)]


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, stroke_width: int) -> Tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)


def fit_font(
    font_path: Optional[Path],
    text: str,
    desired_size: int,
    max_width: int,
    max_height: int,
    stroke_width: int,
) -> ImageFont.ImageFont:
    if font_path is None:
        return ImageFont.load_default()
    size = max(10, int(desired_size))
    probe = Image.new("L", (max_width + 64, max_height + 64), 0)
    draw = ImageDraw.Draw(probe)
    while size >= 10:
        font = ImageFont.truetype(str(font_path), size=size)
        bbox = text_bbox(draw, text, font, stroke_width)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            return font
        size -= 1
    return ImageFont.truetype(str(font_path), size=10)


def make_render_specs(count: int, seed: int, font_count: int) -> List[RenderSpec]:
    rng = random.Random(seed + 3301)
    safe_positions = [
        (0.32, 0.31),
        (0.50, 0.31),
        (0.32, 0.50),
        (0.50, 0.50),
        (0.32, 0.69),
        (0.50, 0.69),
    ]
    colors = [(0, 0, 0), (20, 20, 70), (70, 15, 15), (15, 55, 25)]
    out: List[RenderSpec] = []
    for idx in range(count):
        x, y = safe_positions[idx % len(safe_positions)]
        out.append(
            RenderSpec(
                spec_id=idx,
                font_index=idx % max(1, font_count),
                font_scale=rng.choice([0.16, 0.19, 0.22, 0.25]),
                x_frac=x,
                y_frac=y,
                rotation_deg=rng.choice([-5.0, -2.0, 0.0, 2.0, 5.0]),
                text_rgb=colors[idx % len(colors)],
                stroke_width=rng.choice([0, 0, 1]),
                stroke_rgb=(245, 245, 245),
                background_rgb=(245, 245, 245),
                seed=seed * 1000 + idx,
            )
        )
    return out


def fit_layer_to_bbox(layer: Image.Image, target_bbox: Tuple[int, int, int, int]) -> Image.Image:
    bbox = layer.getchannel("A").getbbox()
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    if bbox is None:
        return out
    crop = layer.crop(bbox)
    tw = max(1, target_bbox[2] - target_bbox[0])
    th = max(1, target_bbox[3] - target_bbox[1])
    crop = crop.resize((tw, th), resample=BICUBIC)
    out.alpha_composite(crop, dest=(target_bbox[0], target_bbox[1]))
    return out


def _layer_from_handwriting(
    sample: Any,
    spec: RenderSpec,
    image_size: int,
) -> Image.Image:
    crop = render_handwriting_crop(
        sample,
        fill=spec.text_rgb + (255,),
        stroke_fill=spec.stroke_rgb + (255,),
        stroke_width=spec.stroke_width,
        plate=False,
        plate_fill=(0, 0, 0, 0),
        max_width=max(1, int(image_size * 0.55)),
        max_height=max(1, int(image_size * 0.26)),
        angle=spec.rotation_deg,
    )
    layer = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
    x = int(round(spec.x_frac * image_size - crop.width / 2))
    y = int(round(spec.y_frac * image_size - crop.height / 2))
    x = max(0, min(image_size - crop.width, x))
    y = max(0, min(image_size - crop.height, y))
    layer.alpha_composite(crop, dest=(x, y))
    return layer


def normalized_visual_distance(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    bb = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    return float(np.mean(np.abs(aa - bb)))


def render_word_variants(
    word: str,
    spec: RenderSpec,
    image_size: int,
    fonts: Sequence[Path],
    *,
    dataset_split: str,
    handwriting_index: Optional[HandwritingOverlayIndex],
    handwriting_probability: float,
    handwriting_style_holdout_fraction: float,
    handwriting_style_seed: int,
    control_visual_min_distance: float,
) -> Tuple[Dict[str, Image.Image], Dict[str, Any]]:
    font_path = fonts[spec.font_index % len(fonts)] if fonts else None
    font = fit_font(
        font_path,
        word,
        desired_size=int(image_size * spec.font_scale),
        max_width=int(image_size * 0.55),
        max_height=int(image_size * 0.26),
        stroke_width=spec.stroke_width,
    )

    def render_layer(text: str) -> Image.Image:
        layer = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        bbox = text_bbox(draw, text, font, spec.stroke_width)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = int(round(spec.x_frac * image_size - tw / 2 - bbox[0]))
        y = int(round(spec.y_frac * image_size - th / 2 - bbox[1]))
        draw.text(
            (x, y),
            text,
            font=font,
            fill=spec.text_rgb + (255,),
            stroke_width=spec.stroke_width,
            stroke_fill=spec.stroke_rgb + (255,),
        )
        if abs(spec.rotation_deg) > 1e-8:
            layer = layer.rotate(spec.rotation_deg, resample=BICUBIC, center=(image_size / 2, image_size / 2))
        return layer

    style_rng = random.Random(spec.seed ^ int(hashlib.sha1(f"{dataset_split}|{word}".encode()).hexdigest()[:8], 16))
    handwriting_sample = None
    if (
        handwriting_index is not None
        and handwriting_index.available
        and style_rng.random() < max(0.0, min(1.0, handwriting_probability))
    ):
        handwriting_sample = handwriting_index.choose(
            word,
            style_rng,
            sample_split="train" if dataset_split == "train" else "heldout",
            holdout_fraction=handwriting_style_holdout_fraction,
            split_seed=handwriting_style_seed,
        )

    if handwriting_sample is None:
        real_layer = render_layer(word)
        renderer = "digital"
        handwriting_sample_id = ""
        handwriting_style_split = ""
    else:
        real_layer = _layer_from_handwriting(handwriting_sample, spec, image_size)
        renderer = f"handwriting_{handwriting_sample.phase}"
        handwriting_sample_id = handwriting_sample.sample_id
        handwriting_style_split = "train" if dataset_split == "train" else "heldout"

    bbox = real_layer.getchannel("A").getbbox() or (0, 0, image_size, image_size)
    real_crop = real_layer.crop(bbox)
    mirrored_layer = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
    mirrored_layer.alpha_composite(ImageOps.mirror(real_crop), dest=(bbox[0], bbox[1]))

    background = Image.new("RGB", (image_size, image_size), spec.background_rgb)

    def composite(layer: Image.Image) -> Image.Image:
        return Image.alpha_composite(background.convert("RGBA"), layer).convert("RGB")

    readable_image = composite(real_layer)
    mirrored_image = composite(mirrored_layer)
    mirror_distance = normalized_visual_distance(readable_image, mirrored_image)
    mirror_valid = mirror_distance >= control_visual_min_distance

    word_seed = int(hashlib.sha1(word.encode("utf-8")).hexdigest()[:8], 16)
    pseudo_text = ""
    pseudo_layer = None
    pseudo_distance = 0.0
    for attempt in range(24):
        candidate = matched_pseudoword(word, spec.seed ^ word_seed ^ (attempt * 0x9E3779B1))
        candidate_layer = fit_layer_to_bbox(render_layer(candidate), bbox)
        candidate_image = composite(candidate_layer)
        distance = normalized_visual_distance(readable_image, candidate_image)
        if normalize_phrase(candidate) != normalize_phrase(word) and distance >= control_visual_min_distance:
            pseudo_text = candidate
            pseudo_layer = candidate_layer
            pseudo_distance = distance
            break
    if pseudo_layer is None:
        pseudo_text = matched_pseudoword(word, spec.seed ^ word_seed)
        pseudo_layer = fit_layer_to_bbox(render_layer(pseudo_text), bbox)
        pseudo_distance = normalized_visual_distance(readable_image, composite(pseudo_layer))
    pseudo_valid = pseudo_distance >= control_visual_min_distance and normalize_phrase(pseudo_text) != normalize_phrase(word)

    variants = {
        "empty": background,
        "readable": readable_image,
        "mirrored": mirrored_image,
        "pseudo": composite(pseudo_layer),
    }
    meta = {
        "pseudo_text": pseudo_text,
        "pseudo_valid": bool(pseudo_valid),
        "mirror_valid": bool(mirror_valid),
        "pseudo_visual_distance": float(pseudo_distance),
        "mirror_visual_distance": float(mirror_distance),
        "renderer": renderer,
        "handwriting_sample_id": handwriting_sample_id,
        "handwriting_style_split": handwriting_style_split,
        "font_path": str(font_path) if font_path else "PIL_default",
        "bbox_x0": int(bbox[0]),
        "bbox_y0": int(bbox[1]),
        "bbox_x1": int(bbox[2]),
        "bbox_y1": int(bbox[3]),
    }
    return variants, meta


# -----------------------------------------------------------------------------
# Frozen visual tower: arbitrary Conv1 patch-grid forward
# -----------------------------------------------------------------------------


@torch.no_grad()
def visual_from_conv1_grid(model: nn.Module, patch_grid: torch.Tensor) -> torch.Tensor:
    visual = model.visual
    batch, width, gh, gw = patch_grid.shape
    x = patch_grid.reshape(batch, width, gh * gw).permute(0, 2, 1)
    cls = visual.class_embedding.to(device=x.device, dtype=x.dtype)[None, None, :].expand(batch, 1, -1)
    x = torch.cat([cls, x], dim=1)
    pos = visual.positional_embedding.to(device=x.device, dtype=x.dtype)
    if x.shape[1] != pos.shape[0]:
        raise RuntimeError(f"Token count {x.shape[1]} does not match positional embedding length {pos.shape[0]}")
    x = visual.ln_pre(x + pos)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x[:, 0, :])
    if visual.proj is not None:
        x = x.to(dtype=visual.proj.dtype) @ visual.proj
    return F.normalize(x.float(), dim=-1)


def threshold_conv1_delta(delta: torch.Tensor, relative_threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    norms = delta.float().pow(2).sum(dim=1).sqrt()
    maxima = norms.amax(dim=(1, 2), keepdim=True)
    mask = norms > maxima * float(relative_threshold)
    return delta * mask[:, None].to(delta.dtype), mask


def preprocess_stack(preprocess: Any, images: Sequence[Image.Image]) -> torch.Tensor:
    return torch.stack([preprocess(img.convert("RGB")) for img in images], dim=0)


# -----------------------------------------------------------------------------
# Soft-prefix text encoder
# -----------------------------------------------------------------------------


class SoftTextPrefix(nn.Module):
    """One learnable embedding inserted immediately after CLIP's SOT token."""

    def __init__(self, initial_vector: torch.Tensor):
        super().__init__()
        if initial_vector.ndim != 1:
            raise ValueError("initial_vector must be rank 1")
        self.embedding = nn.Parameter(initial_vector.detach().float().clone())

    def forward(self) -> torch.Tensor:
        return self.embedding


def encode_soft_prefix_text(
    model: nn.Module,
    clip_module: Any,
    words: Sequence[str],
    soft_vector: torch.Tensor,
    device: torch.device,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Encode [SOT, soft <text>, word tokens, EOT] and pool explicitly at EOT."""
    if not words:
        return torch.empty((0, model.text_projection.shape[1]), device=device, dtype=torch.float32)

    context_length = int(getattr(model, "context_length", model.positional_embedding.shape[0]))
    if context_length < 4:
        raise RuntimeError(f"Unexpected CLIP text context length: {context_length}")

    outputs: List[torch.Tensor] = []
    # Match OpenAI CLIP encode_text(): token embeddings are stored in fp32,
    # while convert_weights() makes the transformer/visual path fp16.  The
    # injected soft-prefix stream must therefore follow model.dtype, not the
    # embedding-table storage dtype.
    token_dtype = model.dtype
    for group in chunked(list(words), chunk_size):
        # Tokenize into one fewer slot, then insert the soft token after SOT.
        ids = clip_module.tokenize(list(group), context_length=context_length - 1, truncate=True).to(device)
        base_emb = model.token_embedding(ids).to(dtype=token_dtype)
        batch = base_emb.shape[0]
        seq = torch.zeros(
            (batch, context_length, base_emb.shape[-1]),
            device=device,
            dtype=token_dtype,
        )
        seq[:, 0] = base_emb[:, 0]
        seq[:, 1] = soft_vector.to(device=device, dtype=token_dtype)
        seq[:, 2:] = base_emb[:, 1:]

        # Original CLIP tokenization still has EOT as the maximum ID.  We use
        # that fact only on the unmodified sequence, then shift the position.
        eot_pos = ids.argmax(dim=-1) + 1
        if int(eot_pos.max().item()) >= context_length:
            raise RuntimeError("EOT position overflow after soft-token insertion")

        x = seq + model.positional_embedding[:context_length].to(device=device, dtype=token_dtype)
        x = x.permute(1, 0, 2)
        x = model.transformer(x)
        x = x.permute(1, 0, 2)
        x = model.ln_final(x)
        pooled = x[torch.arange(batch, device=device), eot_pos]
        pooled = pooled.to(dtype=model.text_projection.dtype) @ model.text_projection
        outputs.append(F.normalize(pooled.float(), dim=-1))
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_standard_text(
    model: nn.Module,
    clip_module: Any,
    prompts: Sequence[str],
    device: torch.device,
    chunk_size: int = 256,
) -> torch.Tensor:
    out: List[torch.Tensor] = []
    for group in chunked(list(prompts), chunk_size):
        ids = clip_module.tokenize(list(group), truncate=True).to(device)
        z = model.encode_text(ids)
        out.append(F.normalize(z.float(), dim=-1))
    return torch.cat(out, dim=0)


def initialize_soft_vector(
    model: nn.Module,
    clip_module: Any,
    init_words: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    ids = clip_module.tokenize(list(init_words), truncate=True).to(device)
    with torch.no_grad():
        emb = model.token_embedding(ids).float()
        # Existing EOT position is safely found by argmax on unmodified tokens.
        eot = ids.argmax(dim=-1)
        pieces: List[torch.Tensor] = []
        for row in range(ids.shape[0]):
            if int(eot[row].item()) > 1:
                pieces.append(emb[row, 1:int(eot[row].item())])
        if not pieces:
            raise RuntimeError("Could not obtain token pieces for soft-token initialization")
        vec = torch.cat(pieces, dim=0).mean(dim=0)
        median_norm = model.token_embedding.weight.float().norm(dim=1).median()
        vec = F.normalize(vec, dim=0) * median_norm
    return vec.detach().cpu()


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------


def discover_val_ground_truth(imagenet_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Validation ground-truth file does not exist: {explicit}")
        return explicit
    candidates = [
        imagenet_root / "ILSVRC2012_validation_ground_truth.txt",
        imagenet_root / "ILSVRC2012_devkit_t12" / "data" / "ILSVRC2012_validation_ground_truth.txt",
        imagenet_root / "devkit" / "data" / "ILSVRC2012_validation_ground_truth.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def build_flat_val_mapping(
    val_root: Path,
    ground_truth: Path,
    sorted_wnids: Sequence[str],
) -> Dict[str, List[Path]]:
    images = sorted(p for p in val_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    labels = [int(x.strip()) for x in ground_truth.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(images) != len(labels):
        raise RuntimeError(
            f"Flat val image/ground-truth count mismatch: {len(images)} images vs {len(labels)} labels"
        )
    if not labels or min(labels) < 1 or max(labels) > len(sorted_wnids):
        raise RuntimeError("Validation ground-truth class indices are outside the local wnid range")
    mapping: Dict[str, List[Path]] = {}
    for path, class_index in zip(images, labels):
        wnid = sorted_wnids[class_index - 1]
        mapping.setdefault(wnid, []).append(path)
    return mapping


class ImagePool:
    def __init__(
        self,
        root: Path,
        wnid_to_label: Mapping[str, str],
        rng: random.Random,
        class_to_images: Optional[Mapping[str, Sequence[Path]]] = None,
        source_name: Optional[str] = None,
    ):
        self.root = root
        self.wnid_to_label = dict(wnid_to_label)
        self.rng = rng
        self._cache: Dict[str, List[Path]] = {}
        self._provided: Optional[Dict[str, List[Path]]] = None
        if class_to_images is None:
            self.wnids = [w for w in sorted(self.wnid_to_label) if (root / w).is_dir()]
            self.source_key = f"folders:{root.resolve()}"
        else:
            self._provided = {
                str(wnid): [Path(p) for p in paths]
                for wnid, paths in class_to_images.items()
                if paths and str(wnid) in self.wnid_to_label
            }
            self.wnids = sorted(self._provided)
            self.source_key = source_name or f"mapped:{root.resolve()}"
        if not self.wnids:
            raise RuntimeError(f"No usable ImageNet classes found for pool {root}")

    def images_for(self, wnid: str) -> List[Path]:
        if wnid not in self._cache:
            if self._provided is None:
                images = list_images_shallow(self.root / wnid)
            else:
                images = list(self._provided.get(wnid, []))
            if not images:
                raise RuntimeError(f"No images found for {wnid} in pool {self.source_key}")
            self._cache[wnid] = images
        return self._cache[wnid]

    def sample(
        self,
        forbidden_wnid: str,
        avoid_keywords: Sequence[str],
        used_paths: set[str],
    ) -> Tuple[Path, str, str]:
        avoid = tuple(normalize_phrase(x) for x in avoid_keywords if x)
        for _ in range(500):
            wnid = self.rng.choice(self.wnids)
            if forbidden_wnid and wnid == forbidden_wnid:
                continue
            label = self.wnid_to_label[wnid]
            norm_label = normalize_phrase(label)
            if avoid and any(term and term in norm_label for term in avoid):
                continue
            path = self.rng.choice(self.images_for(wnid))
            key = str(path.resolve())
            if key in used_paths:
                continue
            used_paths.add(key)
            return path, wnid, label
        raise RuntimeError(
            f"Could not sample an unrelated image after 500 attempts; forbidden={forbidden_wnid}, avoid={avoid}"
        )


def build_example_manifest(
    words: Sequence[WordRecord],
    train_pool: ImagePool,
    val_pool: ImagePool,
    render_specs: Sequence[RenderSpec],
    train_examples_per_word: int,
    seen_val_examples_per_word: int,
    heldout_examples_per_word: int,
    alpha_min: float,
    alpha_max: float,
    seed: int,
) -> List[ExampleRecord]:
    rng = random.Random(seed + 7727)
    used_train: set[str] = set()
    used_val: set[str] = used_train if train_pool.source_key == val_pool.source_key else set()
    out: List[ExampleRecord] = []

    def add_examples(word: WordRecord, split: str, count: int, pool: ImagePool, used: set[str]) -> None:
        for _ in range(count):
            path, base_wnid, base_label = pool.sample(word.source_wnid, word.avoid_keywords, used)
            spec = rng.choice(list(render_specs))
            alpha = rng.uniform(alpha_min, alpha_max)
            out.append(
                ExampleRecord(
                    example_id=len(out),
                    dataset_split=split,
                    word_id=word.word_id,
                    word=word.word,
                    category=word.category,
                    word_split=word.split,
                    target_wnid=word.source_wnid,
                    base_path=str(path),
                    base_wnid=base_wnid,
                    base_label=base_label,
                    render_spec_id=spec.spec_id,
                    alpha=float(alpha),
                )
            )

    for word in words:
        if word.split == "train":
            add_examples(word, "train", train_examples_per_word, train_pool, used_train)
            add_examples(word, "seen_word_val", seen_val_examples_per_word, val_pool, used_val)
        else:
            add_examples(word, "heldout_word_val", heldout_examples_per_word, val_pool, used_val)
    return out


# -----------------------------------------------------------------------------
# Frozen image-embedding cache
# -----------------------------------------------------------------------------


@torch.no_grad()
def build_embedding_cache(
    model: nn.Module,
    preprocess: Any,
    examples: Sequence[ExampleRecord],
    words_by_id: Mapping[int, WordRecord],
    render_specs_by_id: Mapping[int, RenderSpec],
    fonts: Sequence[Path],
    image_size: int,
    batch_size: int,
    device: torch.device,
    delta_patch_threshold: float,
    sample_dir: Path,
    handwriting_index: Optional[HandwritingOverlayIndex],
    handwriting_probability: float,
    handwriting_style_holdout_fraction: float,
    handwriting_style_seed: int,
    control_visual_min_distance: float,
    control_conv1_min_distance: float,
) -> Dict[str, Any]:
    dtype = model.visual.conv1.weight.dtype
    clean_parts: List[torch.Tensor] = []
    readable_parts: List[torch.Tensor] = []
    mirrored_parts: List[torch.Tensor] = []
    pseudo_parts: List[torch.Tensor] = []
    active_counts: List[int] = []
    touches_45: List[int] = []
    touches_221: List[int] = []
    render_meta_rows: List[Dict[str, Any]] = []
    pseudo_texts: List[str] = []
    pseudo_valid: List[bool] = []
    mirror_valid: List[bool] = []

    ensure_dir(sample_dir)
    saved_samples = 0
    total = len(examples)

    for start in range(0, total, batch_size):
        batch = list(examples[start:start + batch_size])
        base_images: List[Image.Image] = []
        empty_images: List[Image.Image] = []
        readable_images: List[Image.Image] = []
        mirrored_images: List[Image.Image] = []
        pseudo_images: List[Image.Image] = []

        for ex in batch:
            try:
                with Image.open(ex.base_path) as im:
                    base_images.append(im.convert("RGB"))
            except Exception as exc:
                raise RuntimeError(f"Failed to load {ex.base_path}") from exc
            variants, meta = render_word_variants(
                ex.word,
                render_specs_by_id[ex.render_spec_id],
                image_size,
                fonts,
                dataset_split=ex.dataset_split,
                handwriting_index=handwriting_index,
                handwriting_probability=handwriting_probability,
                handwriting_style_holdout_fraction=handwriting_style_holdout_fraction,
                handwriting_style_seed=handwriting_style_seed,
                control_visual_min_distance=control_visual_min_distance,
            )
            empty_images.append(variants["empty"])
            readable_images.append(variants["readable"])
            mirrored_images.append(variants["mirrored"])
            pseudo_images.append(variants["pseudo"])
            render_meta_rows.append({"example_id": ex.example_id, **meta})
            pseudo_texts.append(str(meta["pseudo_text"]))

            if saved_samples < 24:
                variants["readable"].save(sample_dir / f"{ex.example_id:05d}__{safe_name(ex.word)}__readable.png")
                variants["mirrored"].save(sample_dir / f"{ex.example_id:05d}__{safe_name(ex.word)}__mirrored.png")
                variants["pseudo"].save(sample_dir / f"{ex.example_id:05d}__{safe_name(ex.word)}__pseudo.png")
                saved_samples += 1

        base_tensor = preprocess_stack(preprocess, base_images).to(device)
        empty_tensor = preprocess_stack(preprocess, empty_images).to(device)
        readable_tensor = preprocess_stack(preprocess, readable_images).to(device)
        mirrored_tensor = preprocess_stack(preprocess, mirrored_images).to(device)
        pseudo_tensor = preprocess_stack(preprocess, pseudo_images).to(device)

        base_grid = model.visual.conv1(base_tensor.to(dtype=dtype))
        empty_grid = model.visual.conv1(empty_tensor.to(dtype=dtype))
        real_delta, real_mask = threshold_conv1_delta(
            model.visual.conv1(readable_tensor.to(dtype=dtype)) - empty_grid,
            delta_patch_threshold,
        )
        mirror_delta, _ = threshold_conv1_delta(
            model.visual.conv1(mirrored_tensor.to(dtype=dtype)) - empty_grid,
            delta_patch_threshold,
        )
        pseudo_delta, _ = threshold_conv1_delta(
            model.visual.conv1(pseudo_tensor.to(dtype=dtype)) - empty_grid,
            delta_patch_threshold,
        )

        def relative_conv1_distance(candidate: torch.Tensor) -> torch.Tensor:
            real_flat = real_delta.float().flatten(1)
            candidate_flat = candidate.float().flatten(1)
            denominator = torch.maximum(
                real_flat.norm(dim=1), candidate_flat.norm(dim=1)
            ).clamp_min(1.0e-8)
            return (real_flat - candidate_flat).norm(dim=1) / denominator

        mirror_conv1_distance = relative_conv1_distance(mirror_delta).cpu()
        pseudo_conv1_distance = relative_conv1_distance(pseudo_delta).cpu()
        batch_meta = render_meta_rows[-len(batch):]
        for index, meta in enumerate(batch_meta):
            mirror_ok = bool(meta["mirror_valid"]) and float(mirror_conv1_distance[index]) >= control_conv1_min_distance
            pseudo_ok = bool(meta["pseudo_valid"]) and float(pseudo_conv1_distance[index]) >= control_conv1_min_distance
            mirror_valid.append(mirror_ok)
            pseudo_valid.append(pseudo_ok)
            meta["mirror_conv1_relative_l2"] = float(mirror_conv1_distance[index])
            meta["pseudo_conv1_relative_l2"] = float(pseudo_conv1_distance[index])
            meta["mirror_valid"] = mirror_ok
            meta["pseudo_valid"] = pseudo_ok

        alpha = torch.tensor([x.alpha for x in batch], device=device, dtype=base_grid.dtype)[:, None, None, None]
        all_grids = torch.cat(
            [
                base_grid,
                base_grid + alpha * real_delta,
                base_grid + alpha * mirror_delta,
                base_grid + alpha * pseudo_delta,
            ],
            dim=0,
        )
        all_emb = visual_from_conv1_grid(model, all_grids).cpu().to(torch.float16)
        b = len(batch)
        clean_parts.append(all_emb[0:b])
        readable_parts.append(all_emb[b:2 * b])
        mirrored_parts.append(all_emb[2 * b:3 * b])
        pseudo_parts.append(all_emb[3 * b:4 * b])

        flat_mask = real_mask.reshape(b, -1)
        active_counts.extend(flat_mask.sum(dim=1).cpu().tolist())
        touches_45.extend(flat_mask[:, 45].to(torch.int64).cpu().tolist() if flat_mask.shape[1] > 45 else [0] * b)
        touches_221.extend(flat_mask[:, 221].to(torch.int64).cpu().tolist() if flat_mask.shape[1] > 221 else [0] * b)

        print(f"[cache] {min(start + b, total)}/{total}")

    return {
        "clean": torch.cat(clean_parts, dim=0),
        "readable": torch.cat(readable_parts, dim=0),
        "mirrored": torch.cat(mirrored_parts, dim=0),
        "pseudo": torch.cat(pseudo_parts, dim=0),
        "word_id": torch.tensor([x.word_id for x in examples], dtype=torch.long),
        "split": [x.dataset_split for x in examples],
        "active_patches": torch.tensor(active_counts, dtype=torch.int16),
        "touches_patch45": torch.tensor(touches_45, dtype=torch.int8),
        "touches_patch221": torch.tensor(touches_221, dtype=torch.int8),
        "render_meta": render_meta_rows,
        "pseudo_text": pseudo_texts,
        "pseudo_valid": torch.tensor(pseudo_valid, dtype=torch.bool),
        "mirror_valid": torch.tensor(mirror_valid, dtype=torch.bool),
    }


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def build_word_example_index(examples: Sequence[ExampleRecord], split: str) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for idx, ex in enumerate(examples):
        if ex.dataset_split == split:
            out.setdefault(ex.word_id, []).append(idx)
    return out


def sample_balanced_batch(
    word_to_examples: Mapping[int, Sequence[int]],
    batch_words: int,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    word_ids = list(word_to_examples.keys())
    if not word_ids:
        raise RuntimeError("No training word examples")
    chosen = rng.sample(word_ids, min(batch_words, len(word_ids)))
    example_indices = [rng.choice(list(word_to_examples[w])) for w in chosen]
    return chosen, example_indices


def reciprocal_ranks(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_scores = logits.gather(1, targets[:, None])
    ranks = 1 + (logits > target_scores).sum(dim=1)
    return 1.0 / ranks.float()


def evaluate_mode(
    mode_name: str,
    text_embeddings: torch.Tensor,
    cache: Mapping[str, Any],
    examples: Sequence[ExampleRecord],
    words: Sequence[WordRecord],
    device: torch.device,
    batch_size: int = 1024,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    text_embeddings = text_embeddings.to(device)
    split_names = ["train", "seen_word_val", "heldout_word_val"]
    summary_rows: List[Dict[str, Any]] = []
    per_word_rows: List[Dict[str, Any]] = []

    word_by_id = {w.word_id: w for w in words}
    all_word_ids = [w.word_id for w in words]
    id_to_col = {wid: col for col, wid in enumerate(all_word_ids)}

    def eval_indices(indices: List[int]) -> Dict[str, np.ndarray]:
        all_top1: List[np.ndarray] = []
        all_top5: List[np.ndarray] = []
        all_rr: List[np.ndarray] = []
        all_pos: List[np.ndarray] = []
        all_clean: List[np.ndarray] = []
        all_mirror: List[np.ndarray] = []
        all_pseudo: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []
        for group in chunked(indices, batch_size):
            idx = torch.tensor(group, dtype=torch.long)
            pos = cache["readable"][idx].float().to(device)
            clean = cache["clean"][idx].float().to(device)
            mirror = cache["mirrored"][idx].float().to(device)
            pseudo = cache["pseudo"][idx].float().to(device)
            targets = torch.tensor([id_to_col[examples[i].word_id] for i in group], device=device)
            logits = pos @ text_embeddings.T
            top = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            target_text = text_embeddings[targets]
            all_top1.append((top[:, 0] == targets).float().cpu().numpy())
            all_top5.append((top == targets[:, None]).any(dim=1).float().cpu().numpy())
            all_rr.append(reciprocal_ranks(logits, targets).cpu().numpy())
            all_pos.append((pos * target_text).sum(dim=1).cpu().numpy())
            all_clean.append((clean * target_text).sum(dim=1).cpu().numpy())
            all_mirror.append((mirror * target_text).sum(dim=1).cpu().numpy())
            all_pseudo.append((pseudo * target_text).sum(dim=1).cpu().numpy())
            all_pred.append(top[:, 0].cpu().numpy())
        return {
            "top1": np.concatenate(all_top1),
            "top5": np.concatenate(all_top5),
            "rr": np.concatenate(all_rr),
            "pos": np.concatenate(all_pos),
            "clean": np.concatenate(all_clean),
            "mirror": np.concatenate(all_mirror),
            "pseudo": np.concatenate(all_pseudo),
            "pred": np.concatenate(all_pred),
        }

    for split in split_names:
        indices = [i for i, ex in enumerate(examples) if ex.dataset_split == split]
        if not indices:
            continue
        result = eval_indices(indices)
        categories = sorted({examples[i].category for i in indices})
        for category in ["ALL", *categories]:
            local = [j for j, i in enumerate(indices) if category == "ALL" or examples[i].category == category]
            if not local:
                continue
            arr = np.asarray(local, dtype=np.int64)
            summary_rows.append(
                {
                    "mode": mode_name,
                    "dataset_split": split,
                    "category": category,
                    "n": len(local),
                    "top1": float(result["top1"][arr].mean()),
                    "top5": float(result["top5"][arr].mean()),
                    "mrr": float(result["rr"][arr].mean()),
                    "target_sim_readable": float(result["pos"][arr].mean()),
                    "target_sim_clean": float(result["clean"][arr].mean()),
                    "target_sim_mirrored": float(result["mirror"][arr].mean()),
                    "target_sim_pseudo": float(result["pseudo"][arr].mean()),
                    "readable_minus_clean": float((result["pos"][arr] - result["clean"][arr]).mean()),
                    "readable_minus_mirrored": float((result["pos"][arr] - result["mirror"][arr]).mean()),
                    "readable_minus_pseudo": float((result["pos"][arr] - result["pseudo"][arr]).mean()),
                }
            )

        for wid in sorted({examples[i].word_id for i in indices}):
            local = [j for j, i in enumerate(indices) if examples[i].word_id == wid]
            arr = np.asarray(local, dtype=np.int64)
            rec = word_by_id[wid]
            pred_cols = result["pred"][arr]
            pred_words = [words[int(col)].word for col in pred_cols]
            common_pred = max(set(pred_words), key=pred_words.count) if pred_words else ""
            per_word_rows.append(
                {
                    "mode": mode_name,
                    "dataset_split": split,
                    "word_id": wid,
                    "word": rec.word,
                    "word_split": rec.split,
                    "category": rec.category,
                    "n": len(local),
                    "top1": float(result["top1"][arr].mean()),
                    "top5": float(result["top5"][arr].mean()),
                    "mrr": float(result["rr"][arr].mean()),
                    "target_sim_readable": float(result["pos"][arr].mean()),
                    "target_sim_clean": float(result["clean"][arr].mean()),
                    "readable_minus_clean": float((result["pos"][arr] - result["clean"][arr]).mean()),
                    "readable_minus_mirrored": float((result["pos"][arr] - result["mirror"][arr]).mean()),
                    "readable_minus_pseudo": float((result["pos"][arr] - result["pseudo"][arr]).mean()),
                    "most_common_wrong_or_right_prediction": common_pred,
                }
            )
    return summary_rows, per_word_rows


@torch.no_grad()
def content_attack_metrics(
    model: nn.Module,
    clip_module: Any,
    cache: Mapping[str, Any],
    examples: Sequence[ExampleRecord],
    device: torch.device,
) -> List[Dict[str, Any]]:
    labels = sorted({ex.base_label for ex in examples})
    text = encode_standard_text(model, clip_module, [f"a photo of a {x}" for x in labels], device)
    label_to_idx = {x: i for i, x in enumerate(labels)}
    rows: List[Dict[str, Any]] = []
    for split in ["train", "seen_word_val", "heldout_word_val"]:
        indices = [i for i, ex in enumerate(examples) if ex.dataset_split == split]
        if not indices:
            continue
        clean_sims: List[float] = []
        readable_sims: List[float] = []
        mirror_sims: List[float] = []
        pseudo_sims: List[float] = []
        for group in chunked(indices, 1024):
            idx = torch.tensor(group, dtype=torch.long)
            target = torch.tensor([label_to_idx[examples[i].base_label] for i in group], device=device)
            t = text[target]
            clean_sims.extend((cache["clean"][idx].float().to(device) * t).sum(dim=1).cpu().tolist())
            readable_sims.extend((cache["readable"][idx].float().to(device) * t).sum(dim=1).cpu().tolist())
            mirror_sims.extend((cache["mirrored"][idx].float().to(device) * t).sum(dim=1).cpu().tolist())
            pseudo_sims.extend((cache["pseudo"][idx].float().to(device) * t).sum(dim=1).cpu().tolist())
        rows.append(
            {
                "dataset_split": split,
                "n": len(indices),
                "base_class_sim_clean": float(np.mean(clean_sims)),
                "base_class_sim_readable": float(np.mean(readable_sims)),
                "base_class_sim_mirrored": float(np.mean(mirror_sims)),
                "base_class_sim_pseudo": float(np.mean(pseudo_sims)),
                "readable_minus_clean": float(np.mean(np.asarray(readable_sims) - np.asarray(clean_sims))),
            }
        )
    return rows


def train_soft_token(
    model: nn.Module,
    clip_module: Any,
    module: SoftTextPrefix,
    cache: Mapping[str, Any],
    examples: Sequence[ExampleRecord],
    words: Sequence[WordRecord],
    device: torch.device,
    steps: int,
    batch_words: int,
    lr: float,
    warmup_steps: int,
    logit_scale: float,
    margin: float,
    control_weight: float,
    clean_weight: float,
    norm_weight: float,
    eval_every: int,
    out_dir: Path,
    seed: int,
    precision_diag: PrecisionDiagnostics,
) -> List[Dict[str, Any]]:
    """Train the soft <text> operator with the original phase-zero semantics.

    Historical contract:
      * readable rendering of w is the only contrastive positive for <text> w;
      * clean, mirrored, and stroke-matched pseudoglyph variants are controls;
      * pseudowords are NEVER trained as literal <text> positives in this phase.

    The newer pseudoword-as-own-positive curriculum is intentionally quarantined
    under _old_stuff because it changes what the operator is optimized to mean.
    """
    module.to(device)
    module.train()
    optimizer = torch.optim.AdamW([module.embedding], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    initial = module.embedding.detach().clone()
    initial_norm = float(initial.norm().item())
    train_index = build_word_example_index(examples, "train")
    word_by_id = {w.word_id: w for w in words}
    rng = random.Random(seed + 991)
    trajectory: List[Dict[str, Any]] = []

    for step in range(steps):
        current_lr = cosine_lr(step, steps, warmup_steps, lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        chosen_word_ids, example_indices = sample_balanced_batch(train_index, batch_words, rng)
        idx = torch.tensor(example_indices, dtype=torch.long)
        pos = cache["readable"][idx].float().to(device)
        clean = cache["clean"][idx].float().to(device)
        mirror = cache["mirrored"][idx].float().to(device)
        pseudo = cache["pseudo"][idx].float().to(device)
        batch_phrases = [word_by_id[wid].word for wid in chosen_word_ids]

        # Historical phase-zero objective: one text query per intended readable word.
        text = encode_soft_prefix_text(model, clip_module, batch_phrases, module(), device)
        labels = torch.arange(len(chosen_word_ids), device=device)
        logits = float(logit_scale) * (pos @ text.T)
        loss_contrastive = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )

        target_sim = (pos * text).sum(dim=1)
        clean_sim = (clean * text).sum(dim=1)
        mirror_sim = (mirror * text).sum(dim=1)
        pseudo_sim = (pseudo * text).sum(dim=1)
        loss_clean = F.relu(float(margin) - target_sim + clean_sim).mean()
        loss_control = 0.5 * (
            F.relu(float(margin) - target_sim + mirror_sim).mean()
            + F.relu(float(margin) - target_sim + pseudo_sim).mean()
        )
        loss_norm = ((module.embedding.norm() / initial_norm) - 1.0).pow(2)
        loss = (
            loss_contrastive
            + clean_weight * loss_clean
            + control_weight * loss_control
            + norm_weight * loss_norm
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        diagnostic_step = step + 1
        do_precision_log = precision_diag.should_log(
            diagnostic_step, force=(diagnostic_step == steps)
        )
        precision_snapshot = {}
        if do_precision_log:
            precision_snapshot = precision_diag.capture_before_update(
                model=module,
                optimizer=optimizer,
                scaler=None,
                optimizer_step=diagnostic_step,
                micro_step=diagnostic_step,
                epoch=0,
                phase="soft_token",
                tensors={
                    "batch.readable_embedding": pos,
                    "batch.clean_embedding": clean,
                    "batch.mirrored_embedding": mirror,
                    "batch.pseudoword_embedding": pseudo,
                    "soft_text_embedding": text,
                    "contrastive_logits": logits,
                    "target_similarity": target_sim,
                    "clean_similarity": clean_sim,
                    "mirrored_similarity": mirror_sim,
                    "pseudoword_similarity": pseudo_sim,
                    "loss.total": loss.reshape(1),
                    "loss.contrastive": loss_contrastive.reshape(1),
                    "loss.clean": loss_clean.reshape(1),
                    "loss.control": loss_control.reshape(1),
                    "loss.norm": loss_norm.reshape(1),
                },
                extra={
                    "compute": "full_fp32_no_autocast",
                    "gradient_state": "unscaled_preclip",
                    "clip_max_norm": 5.0,
                    "objective_semantics": "historical_phase0_readable_positive_controls_negative",
                },
            )
        torch.nn.utils.clip_grad_norm_([module.embedding], max_norm=5.0)
        optimizer.step()
        if do_precision_log:
            precision_diag.capture_after_update(precision_snapshot, module)

        # Historical emergency norm guard, not continual projection.
        with torch.no_grad():
            norm = module.embedding.norm()
            max_norm = initial_norm * 2.0
            if float(norm.item()) > max_norm:
                module.embedding.mul_(max_norm / norm)

        if step == 0 or (step + 1) % max(1, eval_every) == 0 or step + 1 == steps:
            with torch.no_grad():
                train_top1 = float((logits.argmax(dim=1) == labels).float().mean().item())
                row = {
                    "step": step + 1,
                    "lr": current_lr,
                    "loss": float(loss.item()),
                    "loss_contrastive": float(loss_contrastive.item()),
                    "loss_clean": float(loss_clean.item()),
                    "loss_control": float(loss_control.item()),
                    "loss_norm": float(loss_norm.item()),
                    "batch_top1": train_top1,
                    "soft_norm": float(module.embedding.norm().item()),
                    "cosine_to_initial": float(F.cosine_similarity(module.embedding[None], initial[None]).item()),
                    "target_sim": float(target_sim.mean().item()),
                    "clean_sim": float(clean_sim.mean().item()),
                    "mirror_sim": float(mirror_sim.mean().item()),
                    "pseudo_sim": float(pseudo_sim.mean().item()),
                }
                trajectory.append(row)
                print(
                    f"[train] {step + 1:5d}/{steps} loss={row['loss']:.4f} "
                    f"top1={row['batch_top1']:.3f} pos-clean={row['target_sim'] - row['clean_sim']:+.4f} "
                    f"norm={row['soft_norm']:.3f}"
                )
                torch.save(
                    {
                        "soft_token": module.embedding.detach().cpu(),
                        "initial_soft_token": initial.detach().cpu(),
                        "step": step + 1,
                    },
                    out_dir / "soft_text_token_latest.pt",
                )
                write_csv(out_dir / "training_trajectory.csv", trajectory)

    return trajectory


@torch.no_grad()
def nearest_vocab_tokens(
    model: nn.Module,
    clip_module: Any,
    vector: torch.Tensor,
    topk: int = 50,
) -> List[Dict[str, Any]]:
    weight = F.normalize(model.token_embedding.weight.float(), dim=1)
    score = weight @ F.normalize(vector.float().to(weight.device), dim=0)
    values, indices = score.topk(min(topk, score.numel()))
    decoder = getattr(getattr(clip_module, "_tokenizer", None), "decoder", {})
    rows: List[Dict[str, Any]] = []
    for rank, (idx, val) in enumerate(zip(indices.cpu().tolist(), values.cpu().tolist()), start=1):
        token = decoder.get(idx, f"<token_{idx}>") if isinstance(decoder, dict) else f"<token_{idx}>"
        rows.append({"rank": rank, "token_id": idx, "token": token, "cosine": float(val)})
    return rows


def make_report(
    out_dir: Path,
    words: Sequence[WordRecord],
    examples: Sequence[ExampleRecord],
    summary_rows: Sequence[Mapping[str, Any]],
    content_rows: Sequence[Mapping[str, Any]],
) -> None:
    def find(mode: str, split: str, category: str = "ALL") -> Optional[Mapping[str, Any]]:
        for row in summary_rows:
            if row["mode"] == mode and row["dataset_split"] == split and row["category"] == category:
                return row
        return None

    lines = [
        "# Frozen CLIP soft `<text>` prefix experiment",
        "",
        f"Words: {len(words)} ({sum(w.split == 'train' for w in words)} train, {sum(w.split == 'heldout' for w in words)} held out)",
        f"Cached examples: {len(examples)}",
        "",
        "## Exact-word retrieval among the complete word bank",
        "",
        "| Mode | Split | Top-1 | Top-5 | MRR | Readable-clean | Readable-mirrored | Readable-pseudo |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ["plain_word", "text_that_says", "zero_prefix", "soft_before", "soft_after"]:
        for split in ["train", "seen_word_val", "heldout_word_val"]:
            row = find(mode, split)
            if row is None:
                continue
            lines.append(
                f"| {mode} | {split} | {row['top1']:.4f} | {row['top5']:.4f} | {row['mrr']:.4f} | "
                f"{row['readable_minus_clean']:+.4f} | {row['readable_minus_mirrored']:+.4f} | "
                f"{row['readable_minus_pseudo']:+.4f} |"
            )
    lines.extend(["", "## Base-image content effect", ""])
    for row in content_rows:
        lines.append(
            f"- `{row['dataset_split']}`: base-class cosine clean={row['base_class_sim_clean']:.4f}, "
            f"readable={row['base_class_sim_readable']:.4f}, delta={row['readable_minus_clean']:+.4f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Success on held-out words is the key addressability result; train-word accuracy alone can be memorization.",
            "- Ordinary CLIP text encoding is unchanged because no pretrained parameter was updated.",
            "- This test proves query-mode addressability, not yet confinement of visual text away from the ordinary image embedding.",
            "- Mirrored and pseudoword controls are independently forwarded and remain controls only; pseudowords are not trained as literal <text> positives in this phase.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--imagenet_root", type=Path, required=True)
    p.add_argument("--train_root", type=Path, default=None, help="Defaults to <imagenet_root>/train")
    p.add_argument("--val_root", type=Path, default=None, help="Defaults to <imagenet_root>/val")
    p.add_argument("--class_index_json", type=Path, default=None)
    p.add_argument("--val_ground_truth", type=Path, default=None, help="Official flat-val ground-truth txt; auto-detected when possible")
    p.add_argument("--word_bank_json", type=Path, default=None, help="Optional complete custom word-bank JSON")
    p.add_argument("--overlay_dir", type=Path, default=Path("overlay_selection"))
    p.add_argument(
        "--include_curated_overlay_words",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Merge the curated read-stage word bank into the soft-token curriculum (disabled for the historical narrow soft-token phase).",
    )
    p.add_argument("--curated_heldout_fraction", type=float, default=0.20)
    p.add_argument(
        "--handwriting_overlay_root",
        type=Path,
        default=None,
    )
    p.add_argument("--handwriting_probability", type=float, default=0.0)
    p.add_argument("--handwriting_style_holdout_fraction", type=float, default=0.20)
    p.add_argument("--handwriting_style_seed", type=int, default=20260726)
    p.add_argument("--control_visual_min_distance", type=float, default=0.035)
    p.add_argument("--control_conv1_min_distance", type=float, default=0.02)
    p.add_argument("--model", default="zer0int/CLIP-GmP-ViT-L-14")
    p.add_argument("--device", default="cuda")
    p.add_argument("--download_root", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=Path("outputs/soft_text_token"))
    p.add_argument("--seed", type=int, default=20260726)

    p.add_argument("--num_imagenet_words", type=int, default=64)
    p.add_argument("--heldout_fraction", type=float, default=0.25)
    p.add_argument("--force_imagenet_words", default=";".join(DEFAULT_FORCE_IMAGENET_WORDS))
    p.add_argument("--train_examples_per_word", type=int, default=8)
    p.add_argument("--seen_val_examples_per_word", type=int, default=3)
    p.add_argument("--heldout_examples_per_word", type=int, default=6)

    p.add_argument("--render_specs", type=int, default=6)
    p.add_argument("--font_paths", default="", help="Semicolon-separated extra TrueType/OpenType font paths")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--alpha_min", type=float, default=0.25)
    p.add_argument("--alpha_max", type=float, default=1.25)
    p.add_argument("--delta_patch_threshold", type=float, default=0.01)
    p.add_argument("--cache_batch_size", type=int, default=4, help="Base images per cache batch; four variants are forwarded")
    p.add_argument("--rebuild_cache", action="store_true")

    p.add_argument("--init_words", default="text;word;written;says;letters")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch_words", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--train_logit_scale", type=float, default=30.0)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--control_weight", type=float, default=1.0)
    p.add_argument("--clean_weight", type=float, default=1.0)
    p.add_argument("--norm_weight", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--text_chunk_size", type=int, default=256)
    p.add_argument("--precision_diagnostics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--precision_log_every", type=int, default=100)
    p.add_argument("--precision_example_values", type=int, default=12)
    p.add_argument("--precision_parameter_limit", type=int, default=48)
    p.add_argument("--precision_max_values", type=int, default=262144)
    p.add_argument("--precision_save_plots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true", help="Tiny end-to-end run for checking the pipeline")
    return p


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    args.num_imagenet_words = min(args.num_imagenet_words, 8)
    args.train_examples_per_word = 2
    args.seen_val_examples_per_word = 1
    args.heldout_examples_per_word = 2
    args.render_specs = min(args.render_specs, 2)
    args.steps = min(args.steps, 40)
    args.batch_words = min(args.batch_words, 16)
    args.eval_every = min(args.eval_every, 10)
    args.cache_batch_size = min(args.cache_batch_size, 2)


def main() -> int:
    args = build_argparser().parse_args()
    if args.smoke:
        apply_smoke_overrides(args)
    seed_everything(args.seed)

    train_root = args.train_root or (args.imagenet_root / "train")
    requested_val_root = args.val_root or (args.imagenet_root / "val")
    out_dir = ensure_dir(args.out_dir)
    tables_dir = ensure_dir(out_dir / "tables")
    render_samples_dir = ensure_dir(out_dir / "render_samples")
    precision_diag = PrecisionDiagnostics(
        out_dir,
        "soft_token",
        PrecisionDiagnosticsConfig(
            enabled=args.precision_diagnostics,
            log_every_optimizer_steps=args.precision_log_every,
            example_values_per_tensor=args.precision_example_values,
            selected_parameter_limit=args.precision_parameter_limit,
            max_values_for_statistics=args.precision_max_values,
            save_plots=args.precision_save_plots,
        ),
        requested_precision="fp32",
        resolved_precision="fp32",
        device=args.device,
    )

    if not train_root.is_dir():
        raise FileNotFoundError(f"ImageNet train root does not exist: {train_root}")

    wnid_to_label, label_to_wnid = resolve_imagenet_labels(train_root, args.class_index_json)
    known_wnids = sorted(wnid_to_label)
    flat_val_mapping: Optional[Dict[str, List[Path]]] = None
    val_source_name: str
    if organized_imagenet_root(requested_val_root, known_wnids):
        val_root = requested_val_root
        val_source_name = f"organized:{val_root.resolve()}"
        print(f"[data] using organized ImageNet val folders: {val_root}")
    elif requested_val_root.is_dir():
        ground_truth = discover_val_ground_truth(args.imagenet_root, args.val_ground_truth)
        if ground_truth is not None:
            flat_val_mapping = build_flat_val_mapping(requested_val_root, ground_truth, known_wnids)
            val_root = requested_val_root
            val_source_name = f"flat:{val_root.resolve()}:{ground_truth.resolve()}"
            print(f"[data] using flat ImageNet val with ground truth: {ground_truth}")
        else:
            val_root = train_root
            val_source_name = f"folders:{train_root.resolve()}"
            print(
                f"[data] warning: {requested_val_root} is flat and no validation ground-truth txt was found; "
                "using disjoint sampled train images for validation"
            )
    else:
        val_root = train_root
        val_source_name = f"folders:{train_root.resolve()}"
        print(
            f"[data] warning: validation root does not exist: {requested_val_root}; "
            "using disjoint sampled train images for validation"
        )

    handwriting_index = HandwritingOverlayIndex(args.handwriting_overlay_root)
    if args.word_bank_json is not None:
        words = load_word_bank_json(args.word_bank_json)
    else:
        default_words = build_default_word_bank(
            wnid_to_label=wnid_to_label,
            label_to_wnid=label_to_wnid,
            num_imagenet_words=args.num_imagenet_words,
            heldout_fraction=args.heldout_fraction,
            force_imagenet_words=parse_semicolon_list(args.force_imagenet_words),
            seed=args.seed,
            smoke=args.smoke,
        )
        if args.include_curated_overlay_words:
            curated_words = load_curated_overlay_words(
                args.overlay_dir,
                heldout_fraction=args.curated_heldout_fraction,
                seed=args.seed,
            )
            words = merge_word_banks(curated_words, default_words)
        else:
            words = default_words
        words = augment_soft_word_bank_with_handwriting(
            words,
            handwriting_index,
            heldout_fraction=args.curated_heldout_fraction,
            seed=args.seed,
        )
    word_by_id = {w.word_id: w for w in words}
    write_csv(tables_dir / "word_bank.csv", [asdict(w) for w in words])
    json_dump(out_dir / "word_bank.json", [asdict(w) for w in words])
    print(
        f"[words] {len(words)} total: {sum(w.split == 'train' for w in words)} train, "
        f"{sum(w.split == 'heldout' for w in words)} heldout"
    )

    fonts = discover_fonts(parse_semicolon_list(args.font_paths))
    print(f"[fonts] {len(fonts)} discovered" + (f": {fonts[0]}" if fonts else " (PIL default only)"))
    handwriting_words = set(handwriting_index.words())
    matched_train_words = sum(normalize_phrase(word.word) in handwriting_words for word in words if word.split == "train")
    matched_heldout_words = sum(normalize_phrase(word.word) in handwriting_words for word in words if word.split == "heldout")
    print(
        f"[handwriting] available={handwriting_index.available} manifest_words={len(handwriting_words)} "
        f"bank_matches={matched_train_words} train/{matched_heldout_words} heldout "
        f"probability={args.handwriting_probability:.3f} style_holdout={args.handwriting_style_holdout_fraction:.3f}"
    )
    render_specs = make_render_specs(args.render_specs, args.seed, len(fonts))
    render_by_id = {x.spec_id: x for x in render_specs}
    write_csv(tables_dir / "render_specs.csv", [asdict(x) for x in render_specs])

    train_pool = ImagePool(train_root, wnid_to_label, random.Random(args.seed + 1))
    val_pool = ImagePool(
        val_root,
        wnid_to_label,
        random.Random(args.seed + 2),
        class_to_images=flat_val_mapping,
        source_name=val_source_name,
    )
    examples = build_example_manifest(
        words=words,
        train_pool=train_pool,
        val_pool=val_pool,
        render_specs=render_specs,
        train_examples_per_word=args.train_examples_per_word,
        seen_val_examples_per_word=args.seen_val_examples_per_word,
        heldout_examples_per_word=args.heldout_examples_per_word,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        seed=args.seed,
    )
    write_csv(tables_dir / "example_manifest.csv", [asdict(x) for x in examples])
    print(f"[examples] {len(examples)}")

    print(f"[model] loading CLIP {args.model} through oaicliporg on {args.device}")
    import oaicliporg as clip

    device = torch.device(args.device)
    load_kwargs: Dict[str, Any] = {
        "device": device,
        "jit": False,
        # Rebuild through this stage's vanilla oaicliporg implementation.  A
        # trusted full-model pickle must not smuggle a later bridge class into
        # the hard-token-only phase.
        "reuse_full_model_pickle": False,
    }
    if args.download_root is not None:
        load_kwargs["cache_dir"] = str(args.download_root)
    model, preprocess, load_info = load_openai_clip_anything(
        clip, str(args.model), **load_kwargs
    )
    print(
        f"[model] source={load_info.source_kind} "
        f"format={load_info.detected_format}"
    )
    # Preserve OpenAI-CLIP native model dtype (normally fp16 on CUDA).
    # Only the soft token itself is optimized; do not silently promote the frozen backbone.
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[precision] soft_token_parameter=fp32 frozen_backbone_native={model.dtype} no_autocast")

    cache_signature_payload = {
        "model": args.model,
        "train_root": str(train_root.resolve()),
        "val_root": str(val_root.resolve()),
        "val_source": val_source_name,
        "examples": [asdict(x) for x in examples],
        "words": [asdict(x) for x in words],
        "render_specs": [asdict(x) for x in render_specs],
        "image_size": args.image_size,
        "delta_patch_threshold": args.delta_patch_threshold,
        "handwriting_manifest": str(handwriting_index.manifest_path) if handwriting_index.manifest_path else None,
        "handwriting_probability": args.handwriting_probability,
        "handwriting_style_holdout_fraction": args.handwriting_style_holdout_fraction,
        "handwriting_style_seed": args.handwriting_style_seed,
        "control_visual_min_distance": args.control_visual_min_distance,
        "control_conv1_min_distance": args.control_conv1_min_distance,
    }
    cache_signature = stable_hash(cache_signature_payload)
    cache_path = out_dir / "frozen_image_embedding_cache.pt"

    cache: Dict[str, Any]
    if cache_path.is_file() and not args.rebuild_cache:
        loaded = torch_load_compat(cache_path)
        if loaded.get("signature") == cache_signature:
            cache = loaded
            print(f"[cache] loaded {cache_path}")
        else:
            print("[cache] signature mismatch; rebuilding")
            cache = {}
    else:
        cache = {}

    if not cache:
        cache = build_embedding_cache(
            model=model,
            preprocess=preprocess,
            examples=examples,
            words_by_id=word_by_id,
            render_specs_by_id=render_by_id,
            fonts=fonts,
            image_size=args.image_size,
            batch_size=args.cache_batch_size,
            device=device,
            delta_patch_threshold=args.delta_patch_threshold,
            sample_dir=render_samples_dir,
            handwriting_index=handwriting_index,
            handwriting_probability=args.handwriting_probability,
            handwriting_style_holdout_fraction=args.handwriting_style_holdout_fraction,
            handwriting_style_seed=args.handwriting_style_seed,
            control_visual_min_distance=args.control_visual_min_distance,
            control_conv1_min_distance=args.control_conv1_min_distance,
        )
        cache["signature"] = cache_signature
        cache["model"] = args.model
        torch.save(cache, cache_path)
        print(f"[cache] saved {cache_path}")

    # Add cache diagnostics to a user-readable manifest.
    diag_rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        diag_rows.append(
            {
                **asdict(ex),
                "active_conv1_patches": int(cache["active_patches"][i].item()),
                "touches_patch45": int(cache["touches_patch45"][i].item()),
                "touches_patch221": int(cache["touches_patch221"][i].item()),
                **cache["render_meta"][i],
            }
        )
    write_csv(tables_dir / "example_manifest_with_render_diagnostics.csv", diag_rows)

    init_words = parse_semicolon_list(args.init_words)
    initial_vector = initialize_soft_vector(model, clip, init_words, device)
    soft = SoftTextPrefix(initial_vector)
    if soft.embedding.dtype != torch.float32:
        raise RuntimeError(f"Soft-token parameter must be FP32, found {soft.embedding.dtype}")

    all_phrases = [w.word for w in words]
    with torch.no_grad():
        plain_text = encode_standard_text(model, clip, all_phrases, device, args.text_chunk_size)
        text_that_says = encode_standard_text(
            model,
            clip,
            [f'text that says "{w}"' for w in all_phrases],
            device,
            args.text_chunk_size,
        )
        zero_prefix = encode_soft_prefix_text(
            model,
            clip,
            all_phrases,
            torch.zeros_like(initial_vector),
            device,
            args.text_chunk_size,
        )
        soft_before = encode_soft_prefix_text(
            model,
            clip,
            all_phrases,
            initial_vector.to(device),
            device,
            args.text_chunk_size,
        )

    summary_rows: List[Dict[str, Any]] = []
    per_word_rows: List[Dict[str, Any]] = []
    for mode_name, text_emb in [
        ("plain_word", plain_text),
        ("text_that_says", text_that_says),
        ("zero_prefix", zero_prefix),
        ("soft_before", soft_before),
    ]:
        summary, per_word = evaluate_mode(mode_name, text_emb, cache, examples, words, device)
        summary_rows.extend(summary)
        per_word_rows.extend(per_word)

    content_rows = content_attack_metrics(model, clip, cache, examples, device)
    write_csv(tables_dir / "content_attack_metrics.csv", content_rows)
    write_csv(tables_dir / "evaluation_summary_before.csv", summary_rows)
    write_csv(tables_dir / "evaluation_per_word_before.csv", per_word_rows)
    write_csv(tables_dir / "nearest_vocab_initial.csv", nearest_vocab_tokens(model, clip, initial_vector.to(device)))

    started = time.time()
    trajectory = train_soft_token(
        model=model,
        clip_module=clip,
        module=soft,
        cache=cache,
        examples=examples,
        words=words,
        device=device,
        steps=args.steps,
        batch_words=args.batch_words,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        logit_scale=args.train_logit_scale,
        margin=args.margin,
        control_weight=args.control_weight,
        clean_weight=args.clean_weight,
        norm_weight=args.norm_weight,
        eval_every=args.eval_every,
        out_dir=out_dir,
        seed=args.seed,
        precision_diag=precision_diag,
    )
    elapsed = time.time() - started

    soft.eval()
    with torch.no_grad():
        soft_after = encode_soft_prefix_text(
            model,
            clip,
            all_phrases,
            soft().detach(),
            device,
            args.text_chunk_size,
        )
    after_summary, after_per_word = evaluate_mode("soft_after", soft_after, cache, examples, words, device)
    summary_rows.extend(after_summary)
    per_word_rows.extend(after_per_word)

    write_csv(tables_dir / "evaluation_summary_all_modes.csv", summary_rows)
    write_csv(tables_dir / "evaluation_per_word_all_modes.csv", per_word_rows)
    write_csv(tables_dir / "nearest_vocab_final.csv", nearest_vocab_tokens(model, clip, soft().detach()))
    write_csv(out_dir / "training_trajectory.csv", trajectory)

    checkpoint = {
        "soft_token": soft().detach().cpu(),
        "initial_soft_token": initial_vector.cpu(),
        "model": args.model,
        "placement": "immediately_after_sot",
        "pooling": "explicit_eot_position",
        "context_length": int(model.context_length),
        "init_words": init_words,
        "word_bank": [asdict(w) for w in words],
        "args": vars(args),
        "cache_signature": cache_signature,
    }
    torch.save(checkpoint, out_dir / "soft_text_token_final.pt")

    run_meta = {
        "elapsed_training_seconds": elapsed,
        "device": str(device),
        "model_dtype": str(model.dtype),
        "soft_token_parameter_count": int(soft.embedding.numel()),
        "ordinary_clip_parameters_updated": 0,
        "cache_signature": cache_signature,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    json_dump(out_dir / "run_metadata.json", run_meta)
    make_report(out_dir, words, examples, summary_rows, content_rows)
    precision_diag.finalize()

    print(f"[done] outputs -> {out_dir}")
    print(f"[done] training time: {elapsed / 60.0:.2f} min")
    print(f"[done] soft token -> {out_dir / 'soft_text_token_final.pt'}")
    print(f"[done] report -> {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
