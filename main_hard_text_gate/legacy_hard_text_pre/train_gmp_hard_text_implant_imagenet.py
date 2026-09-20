#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train the hard ``<text>`` implant attached to the modified GmP CLIP model.

The default first-stage run keeps the entire GmP image/text backbone frozen and
trains only:

  * the dedicated hard ``<text>`` embedding,
  * the late PreNorm text-query -> visual-key/value read bridge,
  * the read tap mixture and register gate,
  * the text-present / text-readable head.

The content-correction head remains exactly zero for stage ``read``.  The same
script also supports later ``content`` and ``joint`` stages, so the data and
checkpoint formats do not need to change.

The training examples are ordinary ImageNet images with on-the-fly pixel
renderings from ``overlay_selection/overlay_words_gmp_diverse.txt``.  Readable,
mirrored, and stroke-class-matched pseudoglyph variants share the same location,
font, size, and color.  No Conv1 transplant is used here.

Typical first run
-----------------
python train_gmp_hard_text_implant_imagenet.py ^
  --stage read ^
  --model_path zer0int/CLIP-GmP-ViT-L-14 ^
  --imagenet_root <path-to-ImageNet> ^
  --wnid_json <path-to-imagenet_wnid_to_class.json> ^
  --overlay_dir overlay_selection ^
  --out_dir train_clip_hard_text_stage1_read

The compact ``best.pt`` and ``last.pt`` checkpoints contain only implant state,
the hard-token vector, optimizer/scaler state, manifests, and configuration.
Use ``--implant_checkpoint`` to attach such a saved piece of CLIP to a base
``--model_path``.  Leave it empty when ``--model_path`` already contains the
merged implant.  Use ``--export_merged_best`` to additionally write one directly
loadable full state_dict checkpoint after training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from torch.utils.data import BatchSampler, DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    # Preserve this directory at the front so its historical gmpclipattnamp
    # package remains authoritative for the legacy READ/CONTENT/JOINT stages.
    sys.path.append(str(PROJECT_ROOT))

from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything
from training_support.font_discovery import find_font_files
from training_support.diagnostics.training_precision_diagnostics import (
    PrecisionDiagnostics,
    PrecisionDiagnosticsConfig,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # plots are useful but not required for training
    plt = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# The canonical ImageNet domestic-dog block.  Wolves, foxes, hyenas, etc. are
# outside this range and remain available as ordinary non-dog classes.
DOG_WNID_MIN = 2085620
DOG_WNID_MAX = 2116738

CONTENT_TEMPLATES: Tuple[str, ...] = (
    "a photo of a {}",
    "a photograph of a {}",
    "an image of a {}",
    "in the image, there is a {}",
    "this image contains a {}",
    "a picture showing a {}",
    "a close-up photo of a {}",
    "a cropped photo of a {}",
    "a clear photo of a {}",
    "a natural image of a {}",
    "a detailed photograph of a {}",
    "a scene containing a {}",
    "the main subject is a {}",
    "there is a {} in the scene",
    "a photo featuring a {}",
    "a view of a {}",
    "a visual depiction of a {}",
    "an ordinary photo of a {}",
    "a real-world image of a {}",
    "one can see a {} in the image",
)

# Canonical form is deliberately overrepresented.  The rest prevents the EOT
# state from becoming an accidental one-template lookup program.
READ_TEMPLATES: Tuple[str, ...] = (
    "<text> {}",
    "<text> {}",
    "<text> {}",
    "<text> the word {}",
    "<text> text saying {}",
    "<text> written {}",
    "<text> the visible word {}",
    "<text> letters spelling {}",
    "<text> read {}",
    "<text> an inscription saying {}",
    "<text> the image says {}",
    "<text> typography: {}",
)

PSEUDO_CLASSES: Dict[str, str] = {
    "xheight": "aceimnorsuvwxz",
    "ascender": "bdfhklt",
    "descender": "gjpqy",
    "narrow": "il1",
    "wide": "mw",
    "digit": "023456789",
    "upper": "ABCDEFGHJKLMNPQRSTUVWXYZ",
    "punct": "-_=+~*#@!?",
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageRecord:
    path: str
    wnid: str
    label: str
    class_index: int
    split: str


@dataclass(frozen=True)
class WordRecord:
    global_index: int
    word: str
    split: str
    stratum: str = ""
    is_english: bool = False
    is_mature: bool = False
    selection_order: int = -1


@dataclass(frozen=True)
class RenderPlan:
    font_path: str
    font_size: int
    center_x: float
    center_y: float
    angle: float
    fill: Tuple[int, int, int, int]
    stroke_fill: Tuple[int, int, int, int]
    stroke_width: int
    plate: bool
    plate_fill: Tuple[int, int, int, int]


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


def json_dump(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_int(text: str, bits: int = 64) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=max(1, bits // 8)).digest()
    return int.from_bytes(digest, "little", signed=False)


def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "t"}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    """Append one row without ever producing a ragged CSV.

    If a later row introduces new fields, rewrite the existing file under the
    union header and leave blanks in older rows.  The old implementation used
    the current row's keys while appending beneath the original header, which
    created the malformed stage-1 validation log.
    """
    ensure_dir(path.parent)
    row_dict = {str(k): v for k, v in row.items()}
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row_dict)
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        old_rows = list(reader)

    # A None key means a legacy row had more values than the header.  Do not
    # silently perpetuate it; archive the file and start a clean log.
    malformed = any(None in r for r in old_rows)
    if malformed:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archive = path.with_name(f"{path.stem}_legacy_malformed_{stamp}{path.suffix}")
        path.replace(archive)
        print(f"[log] archived malformed legacy CSV -> {archive}")
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row_dict)
        return

    fields = old_fields + [k for k in row_dict if k not in old_fields]
    if fields != old_fields:
        old_rows.append(row_dict)
        write_csv(path, old_rows)
        return

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields, extrasaction="ignore")
        writer.writerow(row_dict)


def reset_logs_for_fresh_run(out_dir: Path) -> None:
    for name in ("training_log.csv", "validation_log.csv", "epoch_training_summary.csv", "pieces_sigmoid_attention_log.csv"):
        path = out_dir / name
        if path.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            archive = path.with_name(f"{path.stem}_previous_{stamp}{path.suffix}")
            path.replace(archive)
            print(f"[log] archived previous log -> {archive}")


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTS)


def is_dog_wnid(wnid: str) -> bool:
    try:
        value = int(str(wnid).lstrip("n"))
    except ValueError:
        return False
    return DOG_WNID_MIN <= value <= DOG_WNID_MAX


def semantic_collision(word: str, label: str) -> bool:
    w = re.sub(r"[^a-z0-9]+", "", normalize_phrase(word))
    l = re.sub(r"[^a-z0-9]+", "", normalize_phrase(label))
    if not w or not l:
        return False
    return w == l or (len(l) >= 4 and l in w) or (len(w) >= 4 and w in l)


def tensor_to_float(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu().item())


def amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


# -----------------------------------------------------------------------------
# ImageNet manifests and class balancing
# -----------------------------------------------------------------------------


def build_or_load_imagenet_manifests(args: argparse.Namespace, manifest_dir: Path) -> Tuple[List[ImageRecord], List[ImageRecord], List[Dict[str, Any]]]:
    train_csv = manifest_dir / "imagenet_train_manifest.csv"
    val_csv = manifest_dir / "imagenet_val_manifest.csv"
    classes_csv = manifest_dir / "imagenet_selected_classes.csv"

    if train_csv.exists() and val_csv.exists() and classes_csv.exists() and not args.rebuild_manifests:
        def load_records(path: Path) -> List[ImageRecord]:
            with path.open("r", encoding="utf-8", newline="") as f:
                return [
                    ImageRecord(
                        path=row["path"],
                        wnid=row["wnid"],
                        label=row["label"],
                        class_index=int(row["class_index"]),
                        split=row["split"],
                    )
                    for row in csv.DictReader(f)
                ]

        with classes_csv.open("r", encoding="utf-8", newline="") as f:
            classes = [dict(row) for row in csv.DictReader(f)]
            for row in classes:
                row["class_index"] = int(row["class_index"])
                row["is_dog"] = parse_bool(row["is_dog"])
        return load_records(train_csv), load_records(val_csv), classes

    wnid_to_label = read_json(args.wnid_json)
    if not isinstance(wnid_to_label, dict):
        raise TypeError(f"Expected a JSON object at {args.wnid_json}")

    train_root = args.train_root or (args.imagenet_root / "train")
    val_root = args.val_root or (args.imagenet_root / "val")
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(f"ImageNet train/val roots not found: {train_root}, {val_root}")

    available: List[Tuple[str, str]] = []
    for wnid, label in wnid_to_label.items():
        if (train_root / wnid).is_dir() and (val_root / wnid).is_dir():
            available.append((str(wnid), str(label)))
    if not available:
        raise RuntimeError("No WNID subfolders matched the supplied JSON")

    rng = random.Random(args.seed)
    dogs = [(w, l) for w, l in available if is_dog_wnid(w)]
    others = [(w, l) for w, l in available if not is_dog_wnid(w)]
    rng.shuffle(dogs)
    rng.shuffle(others)
    if args.max_dog_classes >= 0:
        dogs = dogs[: args.max_dog_classes]
    selected = others + dogs
    rng.shuffle(selected)
    if args.max_classes > 0:
        selected = selected[: args.max_classes]
    selected = sorted(selected, key=lambda x: x[0])

    train_records: List[ImageRecord] = []
    val_records: List[ImageRecord] = []
    class_rows: List[Dict[str, Any]] = []

    for class_index, (wnid, label) in enumerate(selected):
        tr = list_images(train_root / wnid)
        va = list_images(val_root / wnid)
        if not tr or not va:
            continue
        local_rng = random.Random(args.seed ^ stable_int(wnid))
        local_rng.shuffle(tr)
        local_rng.shuffle(va)
        tr = tr[: min(len(tr), args.train_images_per_class)]
        va = va[: min(len(va), args.val_images_per_class)]
        if not tr or not va:
            continue
        actual_index = len(class_rows)
        class_rows.append({
            "class_index": actual_index,
            "wnid": wnid,
            "label": label,
            "is_dog": is_dog_wnid(wnid),
            "n_train": len(tr),
            "n_val": len(va),
        })
        train_records.extend(ImageRecord(str(p), wnid, label, actual_index, "train") for p in tr)
        val_records.extend(ImageRecord(str(p), wnid, label, actual_index, "val") for p in va)

    if not class_rows:
        raise RuntimeError("Class selection produced no usable classes")

    write_csv(train_csv, [asdict(r) for r in train_records])
    write_csv(val_csv, [asdict(r) for r in val_records])
    write_csv(classes_csv, class_rows)
    return train_records, val_records, class_rows


# -----------------------------------------------------------------------------
# Overlay vocabulary and held-out word split
# -----------------------------------------------------------------------------


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        p = self.parent.setdefault(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_overlay_rows(overlay_dir: Path) -> List[Dict[str, Any]]:
    csv_path = overlay_dir / "overlay_words_gmp_diverse.csv"
    txt_path = overlay_dir / "overlay_words_gmp_diverse.txt"
    rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                word = str(row.get("word", ""))
                if not word:
                    continue
                rows.append({
                    "word": word,
                    "stratum": str(row.get("overlay_stratum", "")),
                    "is_english": parse_bool(row.get("is_english", False)),
                    "is_mature": parse_bool(row.get("is_mature", False)),
                    "selection_order": int(float(row.get("selection_order", i))),
                })
    elif txt_path.exists():
        rows = [
            {"word": line.rstrip("\r\n"), "stratum": "", "is_english": False, "is_mature": False, "selection_order": i}
            for i, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines())
            if line.rstrip("\r\n")
        ]
    else:
        raise FileNotFoundError(f"Missing {csv_path} and {txt_path}")

    # Preserve exact first occurrence; surface strings are programs here.
    seen = set()
    deduped = []
    for row in sorted(rows, key=lambda r: int(r["selection_order"])):
        word = row["word"]
        if word not in seen:
            seen.add(word)
            deduped.append(row)
    if len(deduped) < 8:
        raise RuntimeError(f"Only {len(deduped)} overlay words found")
    return deduped


def load_orthographic_pairs(overlay_dir: Path, valid_words: Sequence[str]) -> List[Tuple[str, str]]:
    path = overlay_dir / "orthographic_pairs_selected.csv"
    if not path.exists():
        return []
    valid = set(valid_words)
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            a, b = str(row.get("word_a", "")), str(row.get("word_b", ""))
            if a and b and a != b and a in valid and b in valid:
                pairs.append((a, b))
    return pairs


def build_or_load_word_split(args: argparse.Namespace, manifest_dir: Path) -> Tuple[List[WordRecord], Dict[str, str]]:
    split_csv = manifest_dir / "overlay_word_split.csv"
    pair_map_json = manifest_dir / "orthographic_pair_map.json"
    if split_csv.exists() and pair_map_json.exists() and not args.rebuild_manifests:
        records: List[WordRecord] = []
        with split_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                records.append(WordRecord(
                    global_index=int(row["global_index"]),
                    word=row["word"],
                    split=row["split"],
                    stratum=row.get("stratum", ""),
                    is_english=parse_bool(row.get("is_english", False)),
                    is_mature=parse_bool(row.get("is_mature", False)),
                    selection_order=int(row.get("selection_order", -1)),
                ))
        return records, {str(k): str(v) for k, v in read_json(pair_map_json).items()}

    raw = load_overlay_rows(args.overlay_dir)
    words = [r["word"] for r in raw]
    pairs = load_orthographic_pairs(args.overlay_dir, words)
    uf = UnionFind(words)
    for a, b in pairs:
        uf.union(a, b)
    groups: Dict[str, List[str]] = defaultdict(list)
    for w in words:
        groups[uf.find(w)].append(w)

    row_by_word = {r["word"]: r for r in raw}
    grouped_by_stratum: Dict[str, List[List[str]]] = defaultdict(list)
    for members in groups.values():
        strata = [row_by_word[w].get("stratum", "") for w in members]
        stratum = max(set(strata), key=strata.count) if strata else ""
        grouped_by_stratum[stratum].append(members)

    rng = random.Random(args.seed + 173)
    val_words: set[str] = set()
    for stratum, units in grouped_by_stratum.items():
        rng.shuffle(units)
        target = int(round(sum(len(u) for u in units) * args.heldout_word_fraction))
        count = 0
        for unit in units:
            if count >= target:
                break
            val_words.update(unit)
            count += len(unit)

    # Guarantee nonempty train and validation sets.
    if not val_words:
        val_words.add(words[-1])
    if len(val_words) >= len(words):
        val_words.remove(words[0])

    records: List[WordRecord] = []
    for global_index, row in enumerate(raw):
        records.append(WordRecord(
            global_index=global_index,
            word=row["word"],
            split="val" if row["word"] in val_words else "train",
            stratum=str(row.get("stratum", "")),
            is_english=bool(row.get("is_english", False)),
            is_mature=bool(row.get("is_mature", False)),
            selection_order=int(row.get("selection_order", global_index)),
        ))

    pair_map: Dict[str, str] = {}
    for a, b in pairs:
        pair_map.setdefault(a, b)
        pair_map.setdefault(b, a)
    write_csv(split_csv, [asdict(r) for r in records])
    json_dump(pair_map_json, pair_map)
    return records, pair_map


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def find_font_candidates(extra: Sequence[str] = ()) -> List[str]:
    out = find_font_files(extra)
    if not out:
        raise FileNotFoundError("No usable TrueType/OpenType fonts found; pass --font_paths")
    return out


def classify_char(ch: str) -> str:
    if ch.isupper():
        return "upper"
    if ch.isdigit():
        return "digit"
    if ch in PSEUDO_CLASSES["narrow"]:
        return "narrow"
    if ch in PSEUDO_CLASSES["wide"]:
        return "wide"
    if ch in PSEUDO_CLASSES["ascender"]:
        return "ascender"
    if ch in PSEUDO_CLASSES["descender"]:
        return "descender"
    if ch.isalpha():
        return "xheight"
    if ch.isspace():
        return "space"
    return "punct"


def matched_pseudoword(word: str) -> str:
    rng = random.Random(stable_int("pseudo::" + word))
    out: List[str] = []
    for ch in word:
        cls = classify_char(ch)
        if cls == "space":
            out.append(" ")
            continue
        pool = PSEUDO_CLASSES.get(cls, PSEUDO_CLASSES["punct"])
        candidates = [c for c in pool if c.casefold() != ch.casefold()]
        out.append(rng.choice(candidates or list(pool)))
    return "".join(out)


@lru_cache(maxsize=8192)
def choose_font_for_word(word: str, preferred: str, all_fonts_key: Tuple[str, ...]) -> str:
    candidates = [preferred] + [p for p in all_fonts_key if p != preferred]
    replacement = "�" * max(1, len(word.replace(" ", "")))
    best, best_score = preferred, -1e30
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size=36)
            mask = font.getmask(word or " ")
            bbox = mask.getbbox()
            area = 0.0 if bbox is None else float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            rep = font.getmask(replacement)
            same_missing = mask.size == rep.size and bytes(mask) == bytes(rep)
            score = area - (1e12 if same_missing else 0.0)
            if score > best_score:
                best_score = score
                best = path
        except Exception:
            continue
    return best


def random_resized_crop(image: Image.Image, size: int, rng: random.Random, train: bool) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if not train:
        scale = size / min(w, h)
        nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
        image = image.resize((nw, nh), BICUBIC)
        left = (nw - size) // 2
        top = (nh - size) // 2
        return image.crop((left, top, left + size, top + size))

    area = w * h
    for _ in range(10):
        target = area * rng.uniform(0.72, 1.0)
        aspect = math.exp(rng.uniform(math.log(0.82), math.log(1.22)))
        cw = int(round(math.sqrt(target * aspect)))
        ch = int(round(math.sqrt(target / aspect)))
        if 0 < cw <= w and 0 < ch <= h:
            left = rng.randint(0, w - cw)
            top = rng.randint(0, h - ch)
            image = image.crop((left, top, left + cw, top + ch)).resize((size, size), BICUBIC)
            break
    else:
        image = ImageOps.fit(image, (size, size), method=BICUBIC)
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    if rng.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.12))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.85, 1.15))
    return image


def image_to_clip_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CLIP_MEAN, dtype=tensor.dtype)[:, None, None]
    std = torch.tensor(CLIP_STD, dtype=tensor.dtype)[:, None, None]
    return (tensor - mean) / std


def local_luminance(image: Image.Image, cx: float, cy: float, frac: float = 0.30) -> float:
    w, h = image.size
    half_w = max(2, int(w * frac * 0.5))
    half_h = max(2, int(h * frac * 0.5))
    x = int(cx * w)
    y = int(cy * h)
    crop = image.crop((max(0, x - half_w), max(0, y - half_h), min(w, x + half_w), min(h, y + half_h)))
    arr = np.asarray(crop.convert("L"), dtype=np.float32)
    return float(arr.mean()) if arr.size else 127.5


def fit_font(word: str, path: str, requested: int, max_width: int, max_height: int, stroke: int) -> ImageFont.FreeTypeFont:
    size = int(requested)
    while size >= 9:
        font = ImageFont.truetype(path, size=size)
        box = font.getbbox(word or " ", stroke_width=stroke)
        if max(1, box[2] - box[0]) <= max_width and max(1, box[3] - box[1]) <= max_height:
            return font
        size -= 2
    return ImageFont.truetype(path, size=9)


def make_render_plan(base: Image.Image, word: str, fonts: Sequence[str], rng: random.Random, image_size: int) -> RenderPlan:
    centers = [(0.50, 0.17), (0.50, 0.50), (0.50, 0.83), (0.25, 0.50), (0.75, 0.50)]
    cx, cy = rng.choice(centers)
    cx = float(np.clip(cx + rng.uniform(-0.08, 0.08), 0.13, 0.87))
    cy = float(np.clip(cy + rng.uniform(-0.06, 0.06), 0.10, 0.90))
    preferred = rng.choice(list(fonts))
    font_path = choose_font_for_word(word, preferred, tuple(fonts))
    requested = rng.randint(max(20, int(image_size * 0.10)), max(28, int(image_size * 0.25)))
    lum = local_luminance(base, cx, cy)
    if rng.random() < 0.18:
        palette = [(220, 30, 30), (20, 190, 230), (245, 210, 30), (40, 220, 80)]
        rgb = rng.choice(palette)
        stroke_rgb = (0, 0, 0) if sum(rgb) > 360 else (255, 255, 255)
    elif lum > 128:
        rgb, stroke_rgb = (5, 5, 5), (255, 255, 255)
    else:
        rgb, stroke_rgb = (250, 250, 250), (0, 0, 0)
    stroke_width = rng.choice([0, 1, 1, 2])
    plate = rng.random() < 0.22
    plate_fill = (255, 255, 255, rng.randint(165, 225)) if lum < 128 else (0, 0, 0, rng.randint(145, 210))
    return RenderPlan(
        font_path=font_path,
        font_size=requested,
        center_x=cx,
        center_y=cy,
        angle=rng.uniform(-9.0, 9.0),
        fill=(*rgb, 255),
        stroke_fill=(*stroke_rgb, 255),
        stroke_width=stroke_width,
        plate=plate,
        plate_fill=plate_fill,
    )


def render_local_crop(word: str, plan: RenderPlan, image_size: int) -> Image.Image:
    font = fit_font(word, plan.font_path, plan.font_size, int(image_size * 0.90), int(image_size * 0.30), plan.stroke_width)
    dummy = Image.new("RGBA", (image_size * 2, image_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    box = draw.textbbox((0, 0), word or " ", font=font, stroke_width=plan.stroke_width)
    tw = max(1, box[2] - box[0])
    th = max(1, box[3] - box[1])
    pad = max(4, plan.stroke_width + 3)
    crop = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    cd = ImageDraw.Draw(crop)
    if plan.plate:
        cd.rounded_rectangle((0, 0, crop.width - 1, crop.height - 1), radius=max(2, pad), fill=plan.plate_fill)
    cd.text(
        (pad - box[0], pad - box[1]),
        word,
        font=font,
        fill=plan.fill,
        stroke_width=plan.stroke_width,
        stroke_fill=plan.stroke_fill if plan.stroke_width else None,
    )
    if abs(plan.angle) > 1e-3:
        crop = crop.rotate(plan.angle, resample=BICUBIC, expand=True)
    return crop


def paste_center(base: Image.Image, crop: Image.Image, cx: float, cy: float) -> Image.Image:
    out = base.convert("RGBA")
    x = int(round(cx * out.width - crop.width / 2))
    y = int(round(cy * out.height - crop.height / 2))
    x = int(np.clip(x, 0, max(0, out.width - crop.width)))
    y = int(np.clip(y, 0, max(0, out.height - crop.height)))
    out.alpha_composite(crop, dest=(x, y))
    return out.convert("RGB")


def make_variants(base: Image.Image, word: str, fonts: Sequence[str], rng: random.Random, image_size: int) -> Dict[str, Image.Image]:
    plan = make_render_plan(base, word, fonts, rng, image_size)
    readable_crop = render_local_crop(word, plan, image_size)
    mirror_crop = ImageOps.mirror(readable_crop)
    pseudo_crop = render_local_crop(matched_pseudoword(word), plan, image_size)
    if pseudo_crop.size != readable_crop.size:
        pseudo_crop = pseudo_crop.resize(readable_crop.size, BICUBIC)
    return {
        "clean": base,
        "readable": paste_center(base, readable_crop, plan.center_x, plan.center_y),
        "mirrored": paste_center(base, mirror_crop, plan.center_x, plan.center_y),
        "pseudo": paste_center(base, pseudo_crop, plan.center_x, plan.center_y),
    }


# -----------------------------------------------------------------------------
# Dataset and samplers
# -----------------------------------------------------------------------------


class OverlayImageDataset(Dataset):
    def __init__(
        self,
        images: Sequence[ImageRecord],
        words: Sequence[WordRecord],
        fonts: Sequence[str],
        image_size: int,
        seed: int,
        train: bool,
    ):
        self.images = list(images)
        self.words = list(words)
        self.fonts = tuple(fonts)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.train = bool(train)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, key: Any) -> Dict[str, Any]:
        if isinstance(key, (tuple, list)):
            image_idx, word_idx = int(key[0]), int(key[1])
        else:
            image_idx = int(key)
            word_idx = image_idx % len(self.words)
        image_rec = self.images[image_idx]
        word_rec = self.words[word_idx]
        local_seed = self.seed ^ stable_int(f"{self.epoch}|{image_idx}|{word_idx}|{image_rec.path}|{word_rec.word}")
        rng = random.Random(local_seed)
        with Image.open(image_rec.path) as im:
            base = random_resized_crop(im, self.image_size, rng, self.train)
        variants = make_variants(base, word_rec.word, self.fonts, rng, self.image_size)
        return {
            "clean": image_to_clip_tensor(variants["clean"]),
            "readable": image_to_clip_tensor(variants["readable"]),
            "pseudo": image_to_clip_tensor(variants["pseudo"]),
            "mirrored": image_to_clip_tensor(variants["mirrored"]),
            "word": word_rec.word,
            "word_global_index": word_rec.global_index,
            "word_local_index": word_idx,
            "word_split": word_rec.split,
            "class_index": image_rec.class_index,
            "label": image_rec.label,
            "wnid": image_rec.wnid,
            "path": image_rec.path,
            "semantic_collision": semantic_collision(word_rec.word, image_rec.label),
        }


class UniqueWordBatchSampler(BatchSampler):
    def __init__(self, n_images: int, n_words: int, batch_size: int, seed: int, drop_last: bool = True):
        if batch_size > n_words:
            raise ValueError(f"batch_size={batch_size} exceeds n_words={n_words}; unique-word batches impossible")
        self.n_images = int(n_images)
        self.n_words = int(n_words)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        rng = random.Random(self.seed + self.epoch * 1009)
        images = list(range(self.n_images))
        rng.shuffle(images)
        words = list(range(self.n_words))
        rng.shuffle(words)
        wpos = 0
        batch: List[Tuple[int, int]] = []
        for image_idx in images:
            if wpos + self.batch_size > len(words) and not batch:
                rng.shuffle(words)
                wpos = 0
            word_idx = words[wpos]
            wpos += 1
            batch.append((image_idx, word_idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                if wpos + self.batch_size > len(words):
                    rng.shuffle(words)
                    wpos = 0
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_images // self.batch_size
        return math.ceil(self.n_images / self.batch_size)


class FixedPairBatchSampler(BatchSampler):
    def __init__(self, n_images: int, n_words: int, batch_size: int, max_examples: int = 0):
        self.n = min(n_images, max_examples) if max_examples > 0 else n_images
        self.n_words = n_words
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        batch: List[Tuple[int, int]] = []
        for i in range(self.n):
            batch.append((i, i % self.n_words))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return math.ceil(self.n / self.batch_size)


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------


def extract_soft_token(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        tensor = obj
    elif isinstance(obj, dict):
        tensor = None
        for key in ("soft_token", "embedding", "token", "hard_text_embedding"):
            value = obj.get(key)
            if torch.is_tensor(value):
                tensor = value
                break
        if tensor is None:
            raise KeyError(f"No soft-token tensor found in {path}; keys={list(obj)[:20]}")
    else:
        raise TypeError(f"Unsupported soft-token checkpoint type: {type(obj)}")
    return tensor.detach().float().reshape(-1)


def load_compact_implant_checkpoint(model: nn.Module, path: Path, strict: bool = True) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if "implant_state_dict" not in ckpt:
        raise KeyError(f"Not a compact hard-implant checkpoint: {path}")
    model.read_implant.load_state_dict(ckpt["implant_state_dict"], strict=strict)
    model.set_hard_text_token_embedding(ckpt["hard_text_embedding"])
    return ckpt


def _strip_common_state_prefixes(state: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
        out[name] = value
    return out


def load_implant_attachment(model: nn.Module, path: Path, strict: bool = True) -> Dict[str, Any]:
    """Attach a saved hard-token implant to an already loaded base model.

    Accepted inputs:
      * compact ``best.pt`` / ``last.pt`` / ``best_inference.pt`` files;
      * a raw or wrapped full state_dict containing ``read_implant.*`` and
        ``hard_text_embedding``;
      * a serialized nn.Module with those attributes.

    Only the hard token and implant are applied.  Backbone weights in an
    attachment file are intentionally ignored.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, nn.Module):
        state = obj.state_dict()
        meta: Dict[str, Any] = {"format": "serialized_module"}
    elif isinstance(obj, Mapping):
        if "implant_state_dict" in obj:
            if "hard_text_embedding" not in obj:
                raise KeyError(f"Implant checkpoint lacks hard_text_embedding: {path}")
            model.read_implant.load_state_dict(obj["implant_state_dict"], strict=strict)
            model.set_hard_text_token_embedding(obj["hard_text_embedding"])
            return dict(obj)
        state_obj = obj.get("state_dict", obj.get("model_state_dict", obj))
        if not isinstance(state_obj, Mapping):
            raise TypeError(f"Unsupported wrapped state_dict in {path}: {type(state_obj)}")
        state = state_obj
        meta = dict(obj)
    else:
        raise TypeError(f"Unsupported implant attachment type in {path}: {type(obj)}")

    state = _strip_common_state_prefixes(state)
    implant_state = {
        key[len("read_implant."):]: value
        for key, value in state.items()
        if key.startswith("read_implant.")
    }
    token = state.get("hard_text_embedding")
    if not implant_state or not torch.is_tensor(token):
        raise KeyError(
            f"Could not find read_implant.* plus hard_text_embedding in attachment {path}"
        )
    model.read_implant.load_state_dict(implant_state, strict=strict)
    model.set_hard_text_token_embedding(token)
    return meta


def compact_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Any,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    args: argparse.Namespace,
    class_rows: Sequence[Mapping[str, Any]],
    words: Sequence[WordRecord],
) -> Dict[str, Any]:
    return {
        "format": "gmp_hard_text_implant_v1",
        "base_model_path": str(args.model_path),
        "implant_checkpoint": str(args.implant_checkpoint) if args.implant_checkpoint else "",
        "stage": args.stage,
        "read_attention_architecture": str(args.read_attention_architecture),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics),
        "hard_text_embedding": model.hard_text_embedding.detach().cpu(),
        "implant_state_dict": {k: v.detach().cpu() for k, v in model.read_implant.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "classes": list(class_rows),
        "words": [asdict(w) for w in words],
    }


def export_merged_state_dict(model: nn.Module, path: Path) -> None:
    # Synchronize the reserved inspection row with the actual dedicated vector.
    with torch.no_grad():
        model.token_embedding.weight[model.hard_text_token_id].copy_(
            model.hard_text_embedding.to(model.token_embedding.weight.dtype)
        )
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)


def set_trainable_stage(model: nn.Module, stage: str) -> Dict[str, List[nn.Parameter]]:
    for p in model.parameters():
        p.requires_grad_(False)
    groups: Dict[str, List[nn.Parameter]] = defaultdict(list)

    if stage in {"read", "joint"}:
        model.hard_text_embedding.requires_grad_(True)
        groups["hard_token"].append(model.hard_text_embedding)
        for p in model.read_implant.read_bridge.parameters():
            p.requires_grad_(True)
            groups["read"].append(p)
        model.read_implant.read_tap_logits.requires_grad_(True)
        groups["gates"].append(model.read_implant.read_tap_logits)
        for p in model.read_implant.presence_pool.parameters():
            p.requires_grad_(True)
            groups["presence"].append(p)
        model.read_implant.presence_tap_logits.requires_grad_(True)
        groups["gates"].append(model.read_implant.presence_tap_logits)

    if stage in {"content", "joint"}:
        for p in model.read_implant.content_pool.parameters():
            p.requires_grad_(True)
            groups["content"].append(p)
        model.read_implant.content_tap_logits.requires_grad_(True)
        groups["gates"].append(model.read_implant.content_tap_logits)

    model.eval()
    model.read_implant.train()
    return groups


def make_optimizer(model: nn.Module, groups: Dict[str, List[nn.Parameter]], args: argparse.Namespace) -> torch.optim.Optimizer:
    specs = []
    lr_map = {
        "hard_token": args.lr_hard_token,
        "read": args.lr_read,
        "presence": args.lr_presence,
        "content": args.lr_content,
        "gates": args.lr_gates,
    }
    for name, params in groups.items():
        params = [p for p in params if p.requires_grad]
        if not params:
            continue
        specs.append({"params": params, "lr": lr_map[name], "weight_decay": args.weight_decay, "name": name})
    if not specs:
        raise RuntimeError(f"No trainable parameters for stage={args.stage}")
    return torch.optim.AdamW(specs, betas=(0.9, 0.98), eps=1e-8)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def optimizer_lr(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    return {str(group.get("name", i)): float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}


def set_cosine_lr(optimizer: torch.optim.Optimizer, step: int, total_steps: int, warmup_steps: int) -> None:
    if total_steps <= 0:
        return
    if step < warmup_steps:
        factor = max(1e-8, (step + 1) / max(1, warmup_steps))
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    for group in optimizer.param_groups:
        base = group.setdefault("initial_lr", group["lr"])
        group["lr"] = base * factor


@torch.no_grad()
def encode_prompt_ensemble(
    model: nn.Module,
    clip_module: Any,
    labels: Sequence[str],
    templates: Sequence[str],
    device: torch.device,
    amp_dtype: str,
    chunk_size: int = 128,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for label in labels:
        prompts = [t.format(label) for t in templates]
        feats: List[torch.Tensor] = []
        for start in range(0, len(prompts), chunk_size):
            tokens = clip_module.tokenize(prompts[start:start + chunk_size], truncate=True).to(device)
            with amp_context(device, amp_dtype):
                f = model.encode_text(tokens)
            feats.append(F.normalize(f.float(), dim=-1))
        mean = torch.cat(feats, dim=0).mean(dim=0)
        rows.append(F.normalize(mean, dim=0))
    return torch.stack(rows, dim=0).to(device)


@torch.no_grad()
def encode_ordinary_word_bank(
    model: nn.Module,
    clip_module: Any,
    words: Sequence[str],
    device: torch.device,
    amp_dtype: str,
    chunk_size: int = 128,
) -> torch.Tensor:
    prompts = ["a photo of a {}".format(w) for w in words]
    out: List[torch.Tensor] = []
    for start in range(0, len(prompts), chunk_size):
        tokens = clip_module.tokenize(prompts[start:start + chunk_size], truncate=True).to(device)
        with amp_context(device, amp_dtype):
            out.append(F.normalize(model.encode_text(tokens).float(), dim=-1))
    return torch.cat(out, dim=0).to(device)


def read_cosines_from_states(
    model: nn.Module,
    image_info: Mapping[str, Any],
    text_info: Mapping[str, torch.Tensor],
    return_details: bool = False,
):
    if return_details:
        features, details = model.read_implant.read_features(
            image_info["states"],
            text_info["eot_hidden_pre_ln"],
            register_mask=image_info["register_mask"],
            return_details=True,
        )
    else:
        features = model.read_implant.read_features(
            image_info["states"],
            text_info["eot_hidden_pre_ln"],
            register_mask=image_info["register_mask"],
            return_details=False,
        )
        details = None
    # This feature L2 normalization is the pre-existing cosine-scoring path.
    # It is intentionally NOT attention normalization and remains unchanged.
    features = F.normalize(features.float(), dim=-1)
    text = F.normalize(text_info["text_embedding"].float(), dim=-1)
    cos = torch.einsum("bnd,nd->bn", features, text)
    if return_details:
        return cos, details
    return cos


def _diag_normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Diagnostic entropy only; never fed back into the model."""
    w = weights.float().clamp_min(0.0)
    mass = w.sum(dim=-1, keepdim=True)
    p = torch.where(mass > 0, w / mass.clamp_min(1.0e-12), torch.zeros_like(w))
    return -(p * p.clamp_min(1.0e-12).log()).sum(dim=-1)


def _mean_or_nan(x: torch.Tensor) -> float:
    return float(x.float().mean().item()) if x.numel() else float("nan")


def sigmoid_all_read_presence_metrics(
    model: nn.Module,
    read_details: Optional[Mapping[str, Any]],
    presence_details: Optional[Mapping[str, Any]],
    base_batch: int,
) -> Dict[str, float]:
    """Mass/entropy diagnostics for legacy READ/JOINT sigmoid attention."""
    if getattr(model, "read_attention_architecture", "softmax") != "sigmoid_all":
        return {}
    out: Dict[str, float] = {}
    B = int(base_batch)
    if read_details is not None:
        for block_idx, detail in read_details.get("per_block", {}).items():
            w = detail.get("patch_attention")
            if w is None:
                continue
            # [3B,N,H,T] -> mass/normalized entropy [3B,N,H].
            mass = w.float().sum(dim=-1)
            ent = _diag_normalized_entropy(w)
            H = int(mass.shape[-1])
            N = int(mass.shape[1])
            supported = torch.zeros((mass.shape[0], N), dtype=torch.bool, device=mass.device)
            diag_n = min(B, N, mass.shape[0])
            if diag_n:
                ii = torch.arange(diag_n, device=mass.device)
                supported[ii, ii] = True
            unsupported = ~supported
            for h in range(H):
                out[f"pieces_read_b{int(block_idx)}_h{h}_mass_supported"] = _mean_or_nan(mass[..., h][supported])
                out[f"pieces_read_b{int(block_idx)}_h{h}_mass_unsupported"] = _mean_or_nan(mass[..., h][unsupported])
                out[f"pieces_read_b{int(block_idx)}_h{h}_entropy_supported"] = _mean_or_nan(ent[..., h][supported])
                out[f"pieces_read_b{int(block_idx)}_h{h}_entropy_unsupported"] = _mean_or_nan(ent[..., h][unsupported])
            rw = detail.get("register_attention")
            if rw is not None:
                rm = rw.float().sum(dim=-1)
                for h in range(int(rm.shape[-1])):
                    out[f"pieces_read_b{int(block_idx)}_h{h}_register_mass"] = _mean_or_nan(rm[..., h])
        bridge = model.read_implant.read_bridge
        for h, v in enumerate(bridge.sigmoid_patch_head_bias.detach().float().cpu().tolist()):
            out[f"pieces_read_h{h}_patch_bias"] = float(v)
        for h, v in enumerate(bridge.sigmoid_register_head_bias.detach().float().cpu().tolist()):
            out[f"pieces_read_h{h}_register_bias"] = float(v)

    if presence_details is not None:
        for block_idx, w in presence_details.get("per_block", {}).items():
            if w is None:
                continue
            mass = w.float().sum(dim=-1)  # [3B,H]
            ent = _diag_normalized_entropy(w)
            groups = {"readable": slice(0, B), "clean": slice(B, 2*B), "control": slice(2*B, 3*B)}
            for name, sl in groups.items():
                for h in range(int(mass.shape[-1])):
                    out[f"pieces_presence_b{int(block_idx)}_h{h}_mass_{name}"] = _mean_or_nan(mass[sl, h])
                    out[f"pieces_presence_b{int(block_idx)}_h{h}_entropy_{name}"] = _mean_or_nan(ent[sl, h])
        pool = model.read_implant.presence_pool
        for h, v in enumerate(pool.sigmoid_head_bias.detach().float().cpu().tolist()):
            out[f"pieces_presence_h{h}_bias"] = float(v)
    return out


def sigmoid_all_content_metrics(
    model: nn.Module, details: Optional[Mapping[str, Any]], base_batch: int
) -> Dict[str, float]:
    """Mass/entropy diagnostics for legacy CONTENT/JOINT sigmoid pooling."""
    if getattr(model, "read_attention_architecture", "softmax") != "sigmoid_all" or details is None:
        return {}
    out: Dict[str, float] = {}
    B = int(base_batch)
    for block_idx, w in details.get("per_block", {}).items():
        if w is None:
            continue
        mass = w.float().sum(dim=-1)
        ent = _diag_normalized_entropy(w)
        groups = {"clean": slice(0, B), "readable": slice(B, 2*B), "pseudo": slice(2*B, 3*B)}
        for name, sl in groups.items():
            for h in range(int(mass.shape[-1])):
                out[f"pieces_content_b{int(block_idx)}_h{h}_mass_{name}"] = _mean_or_nan(mass[sl, h])
                out[f"pieces_content_b{int(block_idx)}_h{h}_entropy_{name}"] = _mean_or_nan(ent[sl, h])
    pool = model.read_implant.content_pool
    for h, v in enumerate(pool.sigmoid_head_bias.detach().float().cpu().tolist()):
        out[f"pieces_content_h{h}_bias"] = float(v)
    return out


def corrected_image_embeddings(model: nn.Module, image_info: Mapping[str, Any], apply: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    base = image_info["image_embedding"].float()
    if apply:
        correction = model.read_implant.content_correction(image_info["states"]).float()
    else:
        correction = torch.zeros_like(base)
    return F.normalize(base + correction, dim=-1), correction


# -----------------------------------------------------------------------------
# Losses and batch construction
# -----------------------------------------------------------------------------


def random_read_prompt(word: str, rng: random.Random) -> str:
    return rng.choice(READ_TEMPLATES).format(word)


def build_query_bank(
    target_words: Sequence[str],
    train_words: Sequence[str],
    pair_map: Mapping[str, str],
    rng: random.Random,
    extra_random_negatives: int,
) -> List[str]:
    bank = list(target_words)
    used = set(bank)
    for word in target_words:
        mate = pair_map.get(word)
        if mate and mate in train_words and mate not in used:
            bank.append(mate)
            used.add(mate)
    pool = [w for w in train_words if w not in used]
    rng.shuffle(pool)
    bank.extend(pool[: max(0, extra_random_negatives)])
    return bank


def bce_logits_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = logits.sigmoid()
    pred = probs >= 0.5
    true = targets >= 0.5
    out: Dict[str, float] = {}
    for j, name in enumerate(("present", "readable")):
        p, t = pred[:, j], true[:, j]
        tp = (p & t).sum().item()
        fp = (p & ~t).sum().item()
        fn = (~p & t).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        out[f"{name}_precision"] = precision
        out[f"{name}_recall"] = recall
        out[f"{name}_f1"] = f1
        out[f"{name}_acc"] = float((p == t).float().mean().item())
    return out


def compute_read_and_presence_loss(
    model: nn.Module,
    clip_module: Any,
    batch: Mapping[str, Any],
    train_word_strings: Sequence[str],
    pair_map: Mapping[str, str],
    device: torch.device,
    amp_dtype: str,
    args: argparse.Namespace,
    step_seed: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    words = list(batch["word"])
    B = len(words)
    rng = random.Random(step_seed)
    bank_words = build_query_bank(words, train_word_strings, pair_map, rng, args.extra_random_negatives)
    query_prompts = [random_read_prompt(w, rng) for w in bank_words]
    query_tokens = clip_module.tokenize(query_prompts, truncate=True).to(device, non_blocking=True)

    control_name = "pseudo" if rng.random() < 0.5 else "mirrored"
    images = torch.cat([
        batch["readable"],
        batch["clean"],
        batch[control_name],
    ], dim=0).to(device, non_blocking=True)

    # Vision is fully frozen.  Explicit no_grad avoids retaining a useless graph
    # while the text tower still propagates to the dedicated hard-token vector.
    with torch.no_grad(), amp_context(device, amp_dtype):
        image_info = model.encode_image_states(images, return_final_tokens=False)
    sigmoid_diag = getattr(model, "read_attention_architecture", "softmax") == "sigmoid_all"
    with amp_context(device, amp_dtype):
        text_info = model.encode_text_states(query_tokens)
        if sigmoid_diag:
            cos, read_details = read_cosines_from_states(model, image_info, text_info, return_details=True)
            presence_logits, presence_details = model.read_implant.presence_logits(
                image_info["states"], return_details=True
            )
            presence_logits = presence_logits.float()
        else:
            cos = read_cosines_from_states(model, image_info, text_info)
            presence_logits = model.read_implant.presence_logits(image_info["states"]).float()
            read_details = None
            presence_details = None

    read = cos[:B]
    clean = cos[B:2 * B]
    control = cos[2 * B:3 * B]
    labels = torch.arange(B, device=device)
    scaled = read * args.read_logit_scale
    loss_image = F.cross_entropy(scaled, labels)
    loss_text = F.cross_entropy(scaled[:, :B].t(), labels)

    diag_read = read[labels, labels]
    diag_clean = clean[labels, labels]
    diag_control = control[labels, labels]
    loss_clean_margin = F.relu(args.read_control_margin - (diag_read - diag_clean)).mean()
    loss_control_margin = F.relu(args.read_control_margin - (diag_read - diag_control)).mean()

    present_targets = torch.cat([
        torch.ones(B, 2, device=device),
        torch.zeros(B, 2, device=device),
        torch.tensor([[1.0, 0.0]], device=device).expand(B, 2),
    ], dim=0)
    loss_presence = F.binary_cross_entropy_with_logits(presence_logits, present_targets)

    initial_norm = float(args.initial_hard_token_norm)
    token_norm = model.hard_text_embedding.float().norm()
    loss_token_norm = ((token_norm / max(1e-8, initial_norm)) - 1.0).pow(2)
    reg_gate = model.read_implant.read_bridge.register_gate.float()
    loss_reg_gate = reg_gate.abs()

    total = (
        args.read_ce_weight * loss_image
        + args.read_symmetric_weight * loss_text
        + args.read_control_weight * (loss_clean_margin + loss_control_margin)
        + args.presence_weight * loss_presence
        + args.hard_token_norm_weight * loss_token_norm
        + args.register_gate_weight * loss_reg_gate
    )

    with torch.no_grad():
        batch_top1 = float((scaled.argmax(dim=1) == labels).float().mean().item())
        pair_acc = float(((diag_read > diag_clean) & (diag_read > diag_control)).float().mean().item())
    metrics = {
        "loss_read_image_ce": tensor_to_float(loss_image),
        "loss_read_text_ce": tensor_to_float(loss_text),
        "loss_clean_margin": tensor_to_float(loss_clean_margin),
        "loss_control_margin": tensor_to_float(loss_control_margin),
        "loss_presence": tensor_to_float(loss_presence),
        "loss_token_norm": tensor_to_float(loss_token_norm),
        "loss_register_gate": tensor_to_float(loss_reg_gate),
        "batch_read_top1": batch_top1,
        "batch_paired_control_acc": pair_acc,
        "diag_read_cos": tensor_to_float(diag_read.mean()),
        "diag_clean_cos": tensor_to_float(diag_clean.mean()),
        "diag_control_cos": tensor_to_float(diag_control.mean()),
        "hard_token_norm": tensor_to_float(token_norm),
        "register_gate": tensor_to_float(reg_gate),
        "control_is_pseudo": 1.0 if control_name == "pseudo" else 0.0,
    }
    metrics.update(sigmoid_all_read_presence_metrics(
        model, read_details, presence_details, B
    ))
    return total, metrics


def compute_content_loss(
    model: nn.Module,
    clip_module: Any,
    batch: Mapping[str, Any],
    class_bank: torch.Tensor,
    ordinary_word_bank: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B = len(batch["word"])
    images = torch.cat([batch["clean"], batch["readable"], batch["pseudo"]], dim=0).to(device, non_blocking=True)
    with torch.no_grad(), amp_context(device, amp_dtype):
        image_info = model.encode_image_states(images, return_final_tokens=False)
    sigmoid_diag = getattr(model, "read_attention_architecture", "softmax") == "sigmoid_all"
    with amp_context(device, amp_dtype):
        if sigmoid_diag:
            base = image_info["image_embedding"].float()
            correction, content_details = model.read_implant.content_correction(
                image_info["states"], return_details=True
            )
            correction = correction.float()
            corrected = F.normalize(base + correction, dim=-1)
        else:
            corrected, correction = corrected_image_embeddings(model, image_info, apply=True)
            content_details = None

    clean = corrected[:B]
    readable = corrected[B:2 * B]
    pseudo = corrected[2 * B:3 * B]
    teacher = F.normalize(image_info["image_embedding"][:B].float(), dim=-1).detach()

    loss_clean_embed = (1.0 - (clean * teacher).sum(dim=-1)).mean()
    loss_overlay_embed = (1.0 - (readable * teacher).sum(dim=-1)).mean()
    loss_pseudo_embed = (1.0 - (pseudo * teacher).sum(dim=-1)).mean()

    labels = batch["class_index"].to(device, non_blocking=True).long()
    scale = args.content_logit_scale
    teacher_logits = scale * teacher @ class_bank.t()
    clean_logits = scale * clean @ class_bank.t()
    readable_logits = scale * readable @ class_bank.t()
    loss_class = 0.5 * (F.cross_entropy(clean_logits, labels) + F.cross_entropy(readable_logits, labels))

    temp = args.distill_temperature
    teacher_prob = F.softmax(teacher_logits / temp, dim=-1)
    loss_distill = 0.5 * (
        F.kl_div(F.log_softmax(clean_logits / temp, dim=-1), teacher_prob, reduction="batchmean")
        + F.kl_div(F.log_softmax(readable_logits / temp, dim=-1), teacher_prob, reduction="batchmean")
    ) * (temp * temp)

    word_indices = batch["word_global_index"].to(device, non_blocking=True).long()
    correct_text = class_bank[labels]
    overlay_text = ordinary_word_bank[word_indices]
    correct_score = (readable * correct_text).sum(dim=-1)
    overlay_score = (readable * overlay_text).sum(dim=-1)
    noncollision = (~batch["semantic_collision"].to(device, non_blocking=True).bool()).float()
    attack_losses = F.relu(args.content_attack_margin - (correct_score - overlay_score)) * noncollision
    loss_attack = attack_losses.sum() / noncollision.sum().clamp_min(1.0)

    correction_norm = correction.float().norm(dim=-1)
    loss_correction_norm = correction_norm.pow(2).mean()
    total = (
        args.content_embedding_weight * (loss_clean_embed + loss_overlay_embed + 0.5 * loss_pseudo_embed)
        + args.content_class_weight * loss_class
        + args.content_distill_weight * loss_distill
        + args.content_attack_weight * loss_attack
        + args.content_correction_norm_weight * loss_correction_norm
    )
    metrics = {
        "loss_clean_embed": tensor_to_float(loss_clean_embed),
        "loss_overlay_embed": tensor_to_float(loss_overlay_embed),
        "loss_pseudo_embed": tensor_to_float(loss_pseudo_embed),
        "loss_content_class": tensor_to_float(loss_class),
        "loss_content_distill": tensor_to_float(loss_distill),
        "loss_content_attack": tensor_to_float(loss_attack),
        "loss_correction_norm": tensor_to_float(loss_correction_norm),
        "batch_content_clean_top1": float((clean_logits.argmax(dim=1) == labels).float().mean().item()),
        "batch_content_overlay_top1": float((readable_logits.argmax(dim=1) == labels).float().mean().item()),
        "batch_binary_attack_acc": float((correct_score > overlay_score).float().mean().item()),
        "content_correction_norm": tensor_to_float(correction_norm.mean()),
    }
    metrics.update(sigmoid_all_content_metrics(model, content_details, B))
    return total, metrics


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def encode_read_query_bank(
    model: nn.Module,
    clip_module: Any,
    words: Sequence[str],
    device: torch.device,
    amp_dtype: str,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden: List[torch.Tensor] = []
    text: List[torch.Tensor] = []
    prompts = ["<text> {}".format(w) for w in words]
    for start in range(0, len(prompts), chunk_size):
        tokens = clip_module.tokenize(prompts[start:start + chunk_size], truncate=True).to(device)
        with amp_context(device, amp_dtype):
            info = model.encode_text_states(tokens)
        hidden.append(info["eot_hidden_pre_ln"].detach())
        text.append(F.normalize(info["text_embedding"].float(), dim=-1).detach())
    return torch.cat(hidden, dim=0), torch.cat(text, dim=0)


def binary_f1_from_lists(pred: Sequence[bool], true: Sequence[bool]) -> Tuple[float, float, float, float]:
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(true, dtype=bool)
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    acc = float(np.mean(p == t)) if len(p) else 0.0
    return precision, recall, f1, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    clip_module: Any,
    loader: DataLoader,
    all_words: Sequence[WordRecord],
    class_bank: torch.Tensor,
    ordinary_word_bank: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    model.read_implant.eval()
    word_strings = [w.word for w in all_words]
    query_hidden, query_text = encode_read_query_bank(
        model, clip_module, word_strings, device, amp_dtype, args.eval_query_chunk
    )
    n_words = len(word_strings)

    n = 0
    top1 = top5 = 0
    reciprocal_rank = 0.0
    paired_clean = paired_pseudo = paired_mirror = 0
    target_read_sum = target_clean_sum = target_pseudo_sum = target_mirror_sum = 0.0
    content_clean_correct = content_overlay_correct = binary_correct = both_correct = 0
    correction_norm_sum = 0.0
    presence_pred: List[List[bool]] = [[], []]
    presence_true: List[List[bool]] = [[], []]

    for batch in loader:
        B = len(batch["word"])
        images = torch.cat([
            batch["readable"], batch["clean"], batch["pseudo"], batch["mirrored"]
        ], dim=0).to(device, non_blocking=True)
        with amp_context(device, amp_dtype):
            image_info = model.encode_image_states(images, return_final_tokens=False)
            presence = model.read_implant.presence_logits(image_info["states"]).float()
            corrected, correction = corrected_image_embeddings(
                model, image_info, apply=args.stage in {"content", "joint"}
            )

        readable_scores = torch.empty(B, n_words, dtype=torch.float32, device="cpu")
        target_idx = batch["word_global_index"].long()
        target_read = torch.empty(B, dtype=torch.float32)
        target_clean = torch.empty(B, dtype=torch.float32)
        target_pseudo = torch.empty(B, dtype=torch.float32)
        target_mirror = torch.empty(B, dtype=torch.float32)

        for start in range(0, n_words, args.eval_query_chunk):
            end = min(n_words, start + args.eval_query_chunk)
            with amp_context(device, amp_dtype):
                feature = model.read_implant.read_features(
                    image_info["states"],
                    query_hidden[start:end],
                    register_mask=image_info["register_mask"],
                    return_details=False,
                )
            feature = F.normalize(feature.float(), dim=-1)
            cos = torch.einsum("bnd,nd->bn", feature, query_text[start:end])
            readable_scores[:, start:end] = cos[:B].cpu()
            mask = (target_idx >= start) & (target_idx < end)
            if mask.any():
                rows = torch.nonzero(mask, as_tuple=False).flatten()
                cols = (target_idx[rows] - start).to(device)
                target_read[rows] = cos[rows.to(device), cols].cpu()
                target_clean[rows] = cos[(rows + B).to(device), cols].cpu()
                target_pseudo[rows] = cos[(rows + 2 * B).to(device), cols].cpu()
                target_mirror[rows] = cos[(rows + 3 * B).to(device), cols].cpu()

        order = readable_scores.argsort(dim=1, descending=True)
        target_cpu = target_idx.cpu()
        top1_pred = order[:, 0]
        top5_pred = order[:, : min(5, n_words)]
        read_correct = top1_pred.eq(target_cpu)
        top1 += int(read_correct.sum().item())
        top5 += int((top5_pred == target_cpu[:, None]).any(dim=1).sum().item())
        ranks = (order == target_cpu[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        reciprocal_rank += float((1.0 / ranks.float()).sum().item())
        paired_clean += int((target_read > target_clean).sum().item())
        paired_pseudo += int((target_read > target_pseudo).sum().item())
        paired_mirror += int((target_read > target_mirror).sum().item())
        target_read_sum += float(target_read.sum().item())
        target_clean_sum += float(target_clean.sum().item())
        target_pseudo_sum += float(target_pseudo.sum().item())
        target_mirror_sum += float(target_mirror.sum().item())

        labels = batch["class_index"].to(device).long()
        clean_img = corrected[B:2 * B]
        overlay_img = corrected[:B]
        clean_logits = args.content_logit_scale * clean_img @ class_bank.t()
        overlay_logits = args.content_logit_scale * overlay_img @ class_bank.t()
        clean_ok = clean_logits.argmax(dim=1).eq(labels)
        overlay_ok = overlay_logits.argmax(dim=1).eq(labels)
        content_clean_correct += int(clean_ok.sum().item())
        content_overlay_correct += int(overlay_ok.sum().item())

        overlay_text = ordinary_word_bank[target_idx.to(device)]
        correct_text = class_bank[labels]
        binary_ok = (overlay_img * correct_text).sum(dim=-1) > (overlay_img * overlay_text).sum(dim=-1)
        binary_correct += int(binary_ok.sum().item())
        both_correct += int((read_correct.to(device) & overlay_ok).sum().item())
        correction_norm_sum += float(correction[:B].float().norm(dim=-1).sum().item())

        pred = presence.sigmoid() >= 0.5
        true = torch.cat([
            torch.ones(B, 2, dtype=torch.bool, device=device),
            torch.zeros(B, 2, dtype=torch.bool, device=device),
            torch.tensor([[True, False]], device=device).expand(B, 2),
            torch.tensor([[True, False]], device=device).expand(B, 2),
        ], dim=0)
        for j in range(2):
            presence_pred[j].extend(pred[:, j].cpu().tolist())
            presence_true[j].extend(true[:, j].cpu().tolist())
        n += B

    metrics: Dict[str, float] = {
        "n_eval": float(n),
        "read_top1": top1 / max(1, n),
        "read_top5": top5 / max(1, n),
        "read_mrr": reciprocal_rank / max(1, n),
        "readable_gt_clean": paired_clean / max(1, n),
        "readable_gt_pseudo": paired_pseudo / max(1, n),
        "readable_gt_mirror": paired_mirror / max(1, n),
        "target_read_cos": target_read_sum / max(1, n),
        "target_clean_cos": target_clean_sum / max(1, n),
        "target_pseudo_cos": target_pseudo_sum / max(1, n),
        "target_mirror_cos": target_mirror_sum / max(1, n),
        "content_clean_top1": content_clean_correct / max(1, n),
        "content_overlay_top1": content_overlay_correct / max(1, n),
        "content_binary_attack_acc": binary_correct / max(1, n),
        "both_content_and_read_top1": both_correct / max(1, n),
        "content_correction_norm": correction_norm_sum / max(1, n),
        "hard_token_norm": tensor_to_float(model.hard_text_embedding.float().norm()),
        "register_gate": tensor_to_float(model.read_implant.read_bridge.register_gate.float()),
    }
    for j, name in enumerate(("present", "readable")):
        precision, recall, f1, acc = binary_f1_from_lists(presence_pred[j], presence_true[j])
        metrics[f"presence_{name}_precision"] = precision
        metrics[f"presence_{name}_recall"] = recall
        metrics[f"presence_{name}_f1"] = f1
        metrics[f"presence_{name}_acc"] = acc
    for prefix, logits in (
        ("read_tap", model.read_implant.read_tap_logits),
        ("content_tap", model.read_implant.content_tap_logits),
        ("presence_tap", model.read_implant.presence_tap_logits),
    ):
        weights = logits.detach().float().softmax(dim=0).cpu().tolist()
        for i, value in enumerate(weights):
            metrics[f"{prefix}_weight_{i}"] = float(value)
    model.read_implant.train()
    return metrics


def selection_score(stage: str, metrics: Mapping[str, float], baseline: Mapping[str, float]) -> float:
    read_score = (
        metrics.get("read_top1", 0.0)
        + 0.35 * metrics.get("read_mrr", 0.0)
        + 0.15 * metrics.get("readable_gt_pseudo", 0.0)
        + 0.10 * metrics.get("presence_readable_f1", 0.0)
    )
    clean_drop = max(0.0, baseline.get("content_clean_top1", 0.0) - metrics.get("content_clean_top1", 0.0))
    content_score = (
        metrics.get("content_overlay_top1", 0.0)
        + 0.35 * metrics.get("content_binary_attack_acc", 0.0)
        - 2.0 * clean_drop
    )
    if stage == "read":
        return read_score
    if stage == "content":
        return content_score
    return read_score + content_score + 0.25 * metrics.get("both_content_and_read_top1", 0.0)


# -----------------------------------------------------------------------------
# Plotting and reports
# -----------------------------------------------------------------------------


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def plot_logs(out_dir: Path) -> None:
    if plt is None:
        return
    train_rows = load_csv_rows(out_dir / "training_log.csv")
    val_rows = load_csv_rows(out_dir / "validation_log.csv")
    epoch_rows = load_csv_rows(out_dir / "epoch_training_summary.csv")
    if train_rows:
        steps = [int(float(r["global_step"])) for r in train_rows]
        available = set().union(*(r.keys() for r in train_rows))
        keys = [k for k in ("loss_total", "loss_read", "loss_content", "loss_presence") if k in available]
        fig, ax = plt.subplots(figsize=(10, 6))
        for key in keys:
            vals = [float(r.get(key, "nan")) for r in train_rows]
            ax.plot(steps, vals, label=key)
        ax.set_xlabel("global step")
        ax.set_ylabel("loss")
        ax.set_title("Hard <text> implant training losses")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "training_losses.png", dpi=160)
        plt.close(fig)
    if val_rows:
        epochs = [int(float(r["epoch"])) for r in val_rows]
        available = set().union(*(r.keys() for r in val_rows))
        keys = [k for k in (
            "read_top1", "read_mrr", "content_clean_top1", "content_overlay_top1",
            "content_binary_attack_acc", "both_content_and_read_top1",
        ) if k in available]
        fig, ax = plt.subplots(figsize=(10, 6))
        for key in keys:
            vals = [float(r.get(key, "nan")) for r in val_rows]
            ax.plot(epochs, vals, marker="o", label=key)
        ax.set_xlabel("epoch")
        ax.set_ylabel("metric")
        ax.set_ylim(0.0, 1.02)
        ax.set_title("Validation metrics")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "validation_metrics.png", dpi=160)
        plt.close(fig)
    if epoch_rows:
        epochs = [int(float(r["epoch"])) for r in epoch_rows]
        available = set().union(*(r.keys() for r in epoch_rows))
        keys = [k for k in (
            "train_mean_loss_total", "train_mean_loss_read",
            "train_mean_loss_content", "train_mean_loss_presence",
        ) if k in available]
        if keys:
            fig, ax = plt.subplots(figsize=(10, 6))
            for key in keys:
                vals = [float(r.get(key, "nan")) for r in epoch_rows]
                ax.plot(epochs, vals, marker="o", label=key)
            ax.set_xlabel("epoch")
            ax.set_ylabel("mean training loss")
            ax.set_title("Epoch training summaries")
            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "epoch_training_summary.png", dpi=160)
            plt.close(fig)


def write_report(out_dir: Path, args: argparse.Namespace, baseline: Mapping[str, float], best: Mapping[str, float], total_params: int, trainable_params: int) -> None:
    lines = [
        "# GmP hard `<text>` implant training",
        "",
        f"- Stage: `{args.stage}`",
        f"- Base model: `{args.model_path}`",
        f"- Implant attachment: `{args.implant_checkpoint or '(already inside model_path / none)'}`",
        f"- Total parameters: {total_params:,}",
        f"- Trainable parameters: {trainable_params:,}",
        f"- Trainable fraction: {100.0 * trainable_params / max(1, total_params):.6f}%",
        "",
        "## Baseline validation",
        "",
    ]
    for key in sorted(baseline):
        lines.append(f"- `{key}`: {baseline[key]:.6f}")
    lines.extend(["", "## Best validation", ""])
    for key in sorted(best):
        lines.append(f"- `{key}`: {best[key]:.6f}")
    lines.extend([
        "",
        "## Guardrails",
        "",
        "- The ordinary GmP backbone is frozen.",
        "- Stage `read` leaves the content-correction output exactly zero.",
        "- Text queries only control attention weights; read values come from visual tokens.",
        "- The compact checkpoint must be applied to the same modified GmP model implementation.",
        "- Validation metrics and epoch-level training summaries are logged in separate CSV files.",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.out_dir is None:
        args.out_dir = Path(f"outputs/legacy_hard_text_{args.stage}")
    if args.smoke:
        args.max_classes = min(args.max_classes if args.max_classes > 0 else 8, 8)
        args.max_dog_classes = min(args.max_dog_classes, 2)
        args.train_images_per_class = 1
        args.val_images_per_class = 1
        args.epochs = 1
        args.batch_size = min(args.batch_size, 2)
        args.max_train_batches = 2
        args.val_read_examples = min(args.val_read_examples, 16)
        if args.out_dir == Path(f"outputs/legacy_hard_text_{args.stage}"):
            args.out_dir = Path(f"outputs/legacy_hard_text_{args.stage}_smoke")

    out_dir = ensure_dir(args.out_dir)
    if not args.resume:
        reset_logs_for_fresh_run(out_dir)
    manifest_dir = ensure_dir(out_dir / "manifests")
    json_dump(out_dir / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})

    if args.clip_package_root:
        sys.path.insert(0, str(args.clip_package_root.resolve()))
    clip_module = importlib.import_module(args.clip_package)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    print(f"[model] loading {args.model_path} through {args.clip_package}")
    model, _, load_info = load_openai_clip_anything(
        clip_module,
        str(args.model_path),
        device=str(device),
        jit=False,
        read_attention_architecture=args.read_attention_architecture,
        reuse_full_model_pickle=False,
    )
    print(f"[model] source={load_info.source_kind} format={load_info.detected_format}")
    # Keep fp32 master weights; autocast performs the large frozen matmuls in
    # fp16/bf16 while AdamW updates the tiny implant in fp32.
    model.float()
    if model.read_implant is None:
        raise RuntimeError("Loaded model has no hard-text implant; install the modified model.py first")

    legacy_attachment = args.init_implant_checkpoint
    if args.implant_checkpoint and legacy_attachment:
        raise ValueError("Use only one of --implant_checkpoint or deprecated --init_implant_checkpoint")
    attachment = args.implant_checkpoint or legacy_attachment
    if attachment:
        args.implant_checkpoint = attachment
    if legacy_attachment:
        print("[warn] --init_implant_checkpoint is deprecated; use --implant_checkpoint")
    if attachment:
        if not attachment.exists():
            raise FileNotFoundError(f"Implant attachment not found: {attachment}")
        print(f"[model] attaching learned hard-token pieces from {attachment}")
        load_implant_attachment(model, attachment, strict=True)
    elif args.soft_token_path:
        if not args.soft_token_path.exists():
            raise FileNotFoundError(f"Soft-token initializer not found: {args.soft_token_path}")
        vector = extract_soft_token(args.soft_token_path)
        model.set_hard_text_token_embedding(vector)
        print(f"[model] hard token initialized from {args.soft_token_path}; norm={vector.norm().item():.6f}")
    else:
        # Deliberately preserve exactly whatever --model_path supplied.  This is
        # the full-checkpoint path for an already merged piece of CLIP.
        norm = model.hard_text_embedding.detach().float().norm().item()
        print(f"[model] no implant attachment requested; using implant already in --model_path (token norm={norm:.6f})")

    start_epoch = 0
    global_step = 0
    resume_ckpt: Optional[Dict[str, Any]] = None
    if args.resume:
        print(f"[resume] {args.resume}")
        resume_ckpt = load_compact_implant_checkpoint(model, args.resume, strict=True)
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))

    args.initial_hard_token_norm = float(model.hard_text_embedding.detach().float().norm().item())
    trainable_groups = set_trainable_stage(model, args.stage)
    optimizer = make_optimizer(model, trainable_groups, args)
    scaler = make_grad_scaler(device, args.amp_dtype)
    if resume_ckpt is not None:
        if resume_ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

    total_params, trainable_params = count_parameters(model)
    print(f"[model] trainable {trainable_params:,} / {total_params:,} params")
    print(f"[model] taps={model.read_implant.tap_block_list()} stage={args.stage} amp={args.amp_dtype}")

    precision_diag = PrecisionDiagnostics(
        out_dir=out_dir,
        stage=args.stage,
        config=PrecisionDiagnosticsConfig(
            enabled=bool(args.precision_diagnostics),
            log_every_optimizer_steps=int(args.precision_log_every),
            example_values_per_tensor=int(args.precision_example_values),
            selected_parameter_limit=int(args.precision_parameter_limit),
            max_values_for_statistics=int(args.precision_max_values),
            save_plots=bool(args.precision_save_plots),
        ),
        requested_precision=args.amp_dtype,
        resolved_precision=args.amp_dtype,
        device=device,
    )

    train_images, val_images, class_rows = build_or_load_imagenet_manifests(args, manifest_dir)
    all_words, pair_map = build_or_load_word_split(args, manifest_dir)
    train_words = [w for w in all_words if w.split == "train"]
    val_words = [w for w in all_words if w.split == "val"]
    if not train_words or not val_words:
        raise RuntimeError(f"Bad word split: train={len(train_words)}, val={len(val_words)}")
    print(
        f"[data] classes={len(class_rows)} train_images={len(train_images)} val_images={len(val_images)} "
        f"words={len(all_words)} ({len(train_words)} train / {len(val_words)} held out)"
    )

    extra_fonts = [x.strip() for x in args.font_paths.split(";") if x.strip()]
    fonts = find_font_candidates(extra_fonts)
    json_dump(manifest_dir / "fonts.json", fonts)
    train_dataset = OverlayImageDataset(train_images, train_words, fonts, args.image_size, args.seed, train=True)
    val_dataset = OverlayImageDataset(val_images, val_words, fonts, args.image_size, args.seed + 999, train=False)
    train_sampler = UniqueWordBatchSampler(len(train_images), len(train_words), args.batch_size, args.seed, drop_last=True)
    val_sampler = FixedPairBatchSampler(len(val_images), len(val_words), args.eval_batch_size, args.val_read_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=max(0, min(args.num_workers, 2)),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    class_labels = [str(r["label"]) for r in class_rows]
    all_word_strings = [w.word for w in all_words]
    print("[text] encoding diverse ImageNet prompt ensemble")
    class_bank = encode_prompt_ensemble(
        model, clip_module, class_labels, CONTENT_TEMPLATES[: args.num_content_templates],
        device, args.amp_dtype, args.text_chunk_size,
    )
    ordinary_word_bank = encode_ordinary_word_bank(
        model, clip_module, all_word_strings, device, args.amp_dtype, args.text_chunk_size
    )

    # Baseline is always measured with exactly the same held-out words/images.
    print("[eval] baseline")
    baseline = evaluate(model, clip_module, val_loader, all_words, class_bank, ordinary_word_bank, device, args.amp_dtype, args)
    baseline_row = {"epoch": -1, "global_step": global_step, "selection_score": selection_score(args.stage, baseline, baseline), **baseline}
    append_csv(out_dir / "validation_log.csv", baseline_row)
    print(
        f"[baseline] read_top1={baseline['read_top1']:.4f} mrr={baseline['read_mrr']:.4f} "
        f"content clean/overlay={baseline['content_clean_top1']:.4f}/{baseline['content_overlay_top1']:.4f}"
    )

    total_batches = len(train_loader)
    if args.max_train_batches > 0:
        total_batches = min(total_batches, args.max_train_batches)
    optimizer_steps_per_epoch = math.ceil(total_batches / max(1, args.grad_accum_steps))
    total_steps = max(1, args.epochs * optimizer_steps_per_epoch)
    warmup_steps = int(round(args.warmup_fraction * total_steps))
    best_score = -float("inf")
    best_metrics: Dict[str, float] = dict(baseline)
    best_epoch = -1
    train_word_strings = [w.word for w in train_words]

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        model.eval()
        model.read_implant.train()
        running: Dict[str, float] = defaultdict(float)
        n_running = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            set_cosine_lr(optimizer, global_step, total_steps, warmup_steps)
            loss_total = torch.zeros((), device=device)
            metrics: Dict[str, float] = {}

            if args.stage in {"read", "joint"}:
                loss_read, read_metrics = compute_read_and_presence_loss(
                    model, clip_module, batch, train_word_strings, pair_map,
                    device, args.amp_dtype, args, args.seed + global_step * 7919,
                )
                loss_total = loss_total + loss_read
                metrics.update(read_metrics)
                metrics["loss_read"] = tensor_to_float(loss_read)
            else:
                metrics["loss_read"] = 0.0

            if args.stage in {"content", "joint"}:
                loss_content, content_metrics = compute_content_loss(
                    model, clip_module, batch, class_bank, ordinary_word_bank,
                    device, args.amp_dtype, args,
                )
                loss_total = loss_total + loss_content
                metrics.update(content_metrics)
                metrics["loss_content"] = tensor_to_float(loss_content)
            else:
                metrics["loss_content"] = 0.0

            scaler.scale(loss_total / args.grad_accum_steps).backward()
            do_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or (batch_idx + 1 == total_batches)
            if do_step:
                scaler.unscale_(optimizer)
                next_optimizer_step = global_step + 1
                do_precision_log = precision_diag.should_log(
                    next_optimizer_step,
                    force=(next_optimizer_step == total_steps),
                )
                precision_snapshot = {}
                if do_precision_log:
                    precision_snapshot = precision_diag.capture_before_update(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        optimizer_step=next_optimizer_step,
                        micro_step=batch_idx + 1,
                        epoch=epoch,
                        phase=args.stage,
                        tensors={
                            "legacy.loss.total": loss_total.reshape(1),
                        },
                        extra={
                            "autocast_requested": args.amp_dtype,
                            "autocast_resolved": args.amp_dtype,
                            "gradient_state": "unscaled_preclip",
                            "grad_accum_steps": args.grad_accum_steps,
                        },
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                if do_precision_log:
                    precision_diag.capture_after_update(precision_snapshot, model)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    model.read_implant.read_bridge.register_gate.clamp_(-args.register_gate_clamp, args.register_gate_clamp)
                    # Synchronize the reserved row for transparent inspection.
                    model.token_embedding.weight[model.hard_text_token_id].copy_(
                        model.hard_text_embedding.to(model.token_embedding.weight.dtype)
                    )
                global_step += 1
                metrics["grad_norm"] = tensor_to_float(grad_norm)

            metrics["loss_total"] = tensor_to_float(loss_total)
            metrics["epoch"] = float(epoch)
            metrics["batch"] = float(batch_idx)
            metrics["global_step"] = float(global_step)
            metrics.update({f"lr_{k}": v for k, v in optimizer_lr(optimizer).items()})
            append_csv(out_dir / "training_log.csv", metrics)
            pieces_metrics = {
                key: value for key, value in metrics.items()
                if key.startswith("pieces_")
            }
            if pieces_metrics:
                append_csv(
                    out_dir / "pieces_sigmoid_attention_log.csv",
                    {
                        "stage": args.stage,
                        "epoch": float(epoch),
                        "batch": float(batch_idx),
                        "global_step": float(global_step),
                        **pieces_metrics,
                    },
                )
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    running[key] += float(value)
            n_running += 1

            if batch_idx == 0 or (batch_idx + 1) % args.log_every == 0:
                pieces_mass = [
                    float(v) for k, v in metrics.items()
                    if k.startswith("pieces_") and "_mass_" in k and isinstance(v, (int, float)) and math.isfinite(float(v))
                ]
                pieces_msg = (f" pieces_mass_mean={sum(pieces_mass)/len(pieces_mass):.3f}" if pieces_mass else "")
                print(
                    f"[train] e{epoch:02d} b{batch_idx + 1:04d}/{total_batches:04d} "
                    f"loss={metrics['loss_total']:.4f} read={metrics.get('loss_read', 0.0):.4f} "
                    f"content={metrics.get('loss_content', 0.0):.4f} "
                    f"top1={metrics.get('batch_read_top1', float('nan')):.3f} "
                    f"reg={tensor_to_float(model.read_implant.read_bridge.register_gate):+.4f}"
                    f"{pieces_msg}"
                )

        epoch_train = {f"train_mean_{k}": v / max(1, n_running) for k, v in running.items()}
        print(f"[eval] epoch {epoch}")
        val_metrics = evaluate(
            model, clip_module, val_loader, all_words, class_bank, ordinary_word_bank,
            device, args.amp_dtype, args,
        )
        score = selection_score(args.stage, val_metrics, baseline)
        val_row = {"epoch": epoch, "global_step": global_step, "selection_score": score, **val_metrics}
        epoch_train_row = {"epoch": epoch, "global_step": global_step, **epoch_train}
        append_csv(out_dir / "validation_log.csv", val_row)
        append_csv(out_dir / "epoch_training_summary.csv", epoch_train_row)
        print(
            f"[val] score={score:.5f} read top1/mrr={val_metrics['read_top1']:.4f}/{val_metrics['read_mrr']:.4f} "
            f"controls={val_metrics['readable_gt_pseudo']:.4f}/{val_metrics['readable_gt_mirror']:.4f} "
            f"content clean/overlay={val_metrics['content_clean_top1']:.4f}/{val_metrics['content_overlay_top1']:.4f} "
            f"both={val_metrics['both_content_and_read_top1']:.4f}"
        )

        last_ckpt = compact_checkpoint(
            model, optimizer, scaler, epoch, global_step, val_metrics, args, class_rows, all_words
        )
        torch.save(last_ckpt, out_dir / "last.pt")
        if score > best_score:
            best_score = score
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            torch.save(last_ckpt, out_dir / "best.pt")
            torch.save(
                {
                    "format": "gmp_hard_text_implant_inference_v1",
                    "base_model_path": str(args.model_path),
                    "implant_checkpoint": str(args.implant_checkpoint) if args.implant_checkpoint else "",
                    "hard_text_embedding": model.hard_text_embedding.detach().cpu(),
                    "implant_state_dict": {k: v.detach().cpu() for k, v in model.read_implant.state_dict().items()},
                    "metrics": dict(val_metrics),
                    "stage": args.stage,
                    "read_attention_architecture": str(args.read_attention_architecture),
                    "epoch": epoch,
                },
                out_dir / "best_inference.pt",
            )
            print(f"[best] epoch={epoch} score={best_score:.6f}")
        plot_logs(out_dir)

    elapsed = time.time() - started
    if (out_dir / "best.pt").exists():
        load_compact_implant_checkpoint(model, out_dir / "best.pt", strict=True)
    if args.export_merged_best:
        print("[export] writing merged best state_dict (large file)")
        export_merged_state_dict(model, out_dir / "best_merged_state_dict.pt")

    run_meta = {
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "class_count": len(class_rows),
        "train_image_count": len(train_images),
        "val_image_count": len(val_images),
        "train_word_count": len(train_words),
        "val_word_count": len(val_words),
        "font_count": len(fonts),
    }
    json_dump(out_dir / "run_metadata.json", run_meta)
    write_report(out_dir, args, baseline, best_metrics, total_params, trainable_params)
    plot_logs(out_dir)
    precision_diag.finalize()
    print(f"[done] outputs -> {out_dir}")
    print(f"[done] elapsed -> {elapsed / 60.0:.2f} min")
    print(f"[done] best epoch/score -> {best_epoch}/{best_score:.6f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=("read", "content", "joint"), default="read")
    p.add_argument("--clip_package", default="gmpclipattnamp")
    p.add_argument("--clip_package_root", type=Path, default=None)
    p.add_argument(
        "--read_attention_architecture",
        choices=("softmax", "sigmoid_mass", "sigmoid_all"),
        default="softmax",
        help="PIECES token-attention architecture. sigmoid_all uses raw independent sigmoid edges for read/content/presence.",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="zer0int/CLIP-GmP-ViT-L-14",
        help=(
            "Base CLIP model reference: OpenAI model name, Hugging Face repo id "
            "such as 'zer0int/CLIP-GmP-ViT-L-14', or a local checkpoint path. "
            "Keep this as a string: pathlib.Path would rewrite HF repo '/' to '\\' on Windows."
        ),
    )
    p.add_argument(
        "--implant_checkpoint", type=Path, default=None,
        help=("Optional compact/full pickle containing learned hard_text_embedding and read_implant weights. "
              "When empty, keep exactly the implant already present in --model_path."),
    )
    p.add_argument(
        "--soft_token_path", type=Path, default=None,
        help="Optional soft-token initializer for a fresh stage-1 base model; ignored when --implant_checkpoint is set",
    )
    p.add_argument("--init_implant_checkpoint", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--resume", type=Path, default=None, help="Resume implant, optimizer, scaler, epoch, and global step")

    p.add_argument("--imagenet_root", type=Path, required=True)
    p.add_argument("--train_root", type=Path, default=None)
    p.add_argument("--val_root", type=Path, default=None)
    p.add_argument("--wnid_json", type=Path, required=True)
    p.add_argument("--overlay_dir", type=Path, default=Path("overlay_selection"))
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--rebuild_manifests", action="store_true")

    p.add_argument("--max_classes", type=int, default=0, help="0 uses every available non-dog class plus the dog cap")
    p.add_argument("--max_dog_classes", type=int, default=24)
    p.add_argument("--train_images_per_class", type=int, default=6)
    p.add_argument("--val_images_per_class", type=int, default=2)
    p.add_argument("--heldout_word_fraction", type=float, default=0.20)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--font_paths", default="", help="Semicolon-separated extra font paths")

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4, help="Base images; vision sees 3x this many variants in stage read")
    p.add_argument("--eval_batch_size", type=int, default=3, help="Base images; validation forwards 4x variants")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_train_batches", type=int, default=0)
    p.add_argument("--val_read_examples", type=int, default=512)
    p.add_argument("--eval_query_chunk", type=int, default=64)
    p.add_argument("--text_chunk_size", type=int, default=128)
    p.add_argument("--num_content_templates", type=int, default=20)

    p.add_argument("--lr_read", type=float, default=1.0e-3)
    p.add_argument("--lr_hard_token", type=float, default=2.0e-4)
    p.add_argument("--lr_presence", type=float, default=1.0e-3)
    p.add_argument("--lr_content", type=float, default=8.0e-4)
    p.add_argument("--lr_gates", type=float, default=3.0e-4)
    p.add_argument("--weight_decay", type=float, default=1.0e-2)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp_dtype", choices=("fp16", "bf16", "none"), default="fp16")
    p.add_argument("--precision_diagnostics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--precision_log_every", type=int, default=100)
    p.add_argument("--precision_example_values", type=int, default=12)
    p.add_argument("--precision_parameter_limit", type=int, default=48)
    p.add_argument("--precision_max_values", type=int, default=262144)
    p.add_argument("--precision_save_plots", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--read_logit_scale", type=float, default=30.0)
    p.add_argument("--read_ce_weight", type=float, default=1.0)
    p.add_argument("--read_symmetric_weight", type=float, default=0.5)
    p.add_argument("--read_control_weight", type=float, default=1.0)
    p.add_argument("--read_control_margin", type=float, default=0.06, help="Cosine margin readable over clean/control")
    p.add_argument("--presence_weight", type=float, default=0.5)
    p.add_argument("--hard_token_norm_weight", type=float, default=0.02)
    p.add_argument("--register_gate_weight", type=float, default=0.01)
    p.add_argument("--register_gate_clamp", type=float, default=1.0)
    p.add_argument("--extra_random_negatives", type=int, default=8)

    p.add_argument("--content_logit_scale", type=float, default=30.0)
    p.add_argument("--content_embedding_weight", type=float, default=2.0)
    p.add_argument("--content_class_weight", type=float, default=0.5)
    p.add_argument("--content_distill_weight", type=float, default=0.5)
    p.add_argument("--content_attack_weight", type=float, default=1.0)
    p.add_argument("--content_correction_norm_weight", type=float, default=1.0e-3)
    p.add_argument("--content_attack_margin", type=float, default=0.05)
    p.add_argument("--distill_temperature", type=float, default=2.0)

    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260726)
    p.add_argument("--device", default="cuda")
    p.add_argument("--export_merged_best", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
