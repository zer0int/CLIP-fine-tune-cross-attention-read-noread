#!/usr/bin/env python3
from __future__ import annotations

"""
ImageNet-1k frozen-feature linear probe: full TRAIN -> held-out VAL
=================================================================

Purpose
-------
Fix the protocol bug in the older benchmark that randomly split ILSVRC2012 VAL
80/20.  This script instead uses the official extracted ImageNet folders:

    train: ILSVRC2012/train   (~1,281,167 images)
    val:   ILSVRC2012/val     (50,000 images)

The CLIP image encoder is frozen.  Deterministic model-specific CLIP eval
preprocessing is used for both train and val.  L2-normalized image embeddings
are extracted ONCE, then a single linear classifier is trained on the complete
train split and evaluated exactly once on the held-out val split.

Model
-----
One configured model is evaluated per run. The default is the released ModeMUX
checkpoint. For a full x-attention model, CONTENT correction is explicitly ON;
no text candidate / READ bridge is involved in this image-only benchmark.

Scientific guardrails
---------------------
* ImageNet VAL is never used to train, tune, early-stop, or select an epoch.
* The probe recipe is fixed in advance and shared by all three models.
* The backbone is frozen and image embeddings are L2-normalized.
* Feature extraction and probe fitting are separate.  Cached features allow a
  crashed run to resume without re-encoding a million images.
* No plots; final output is a compact ASCII table + CSV/JSON.

This is a proper ImageNet TRAIN->VAL linear-separability comparison.  It is NOT
claimed to reproduce any one paper's exact optimizer/hyperparameter protocol;
it intentionally keeps the earlier benchmark's AdamW/LR/weight-decay recipe
while correcting the data split and removing validation leakage.

Run from the repository root containing:
    oaiclip/
    utils_clip_loader/

Typical:
    python benchmark.py run imagenet_linear_probe

Useful direct-script overrides:
    --model zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX
    --image-batch 400
    --probe-batch 8192
    --workers 4
    --epochs 10
    --cache-dtype float32     # default; float16 halves disk use

Caches are intentionally persistent because train embeddings are expensive.
Delete <output>/feature_cache/ when you deliberately want fresh extraction, or
pass --no-reuse-cache.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Allow both `python -m benchmarks.imagenet_linear_probe` and direct script execution.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import oaiclip as clip
from benchmark_utils.models import DEFAULT_MODEL_ALIAS, DEFAULT_MODEL_PATH
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything
from utils_clip_loader.benchmark_runtime import (
    inference_autocast,
    is_full_xattn,
    temporary_content_correction,
)


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_TRAIN = r"path/to/ILSVRC2012/train"
DEFAULT_VAL = r"path/to/ILSVRC2012/val"
DEFAULT_DEVKIT = Path("utils_datasets/imagenet/ILSVRC2012_devkit_t12")
DEFAULT_OUT = Path("out_bench_results/imagenet_linear_probe")

EXPECTED_TRAIN = 1_281_167
EXPECTED_VAL = 50_000
EXPECTED_CLASSES = 1000

DEFAULT_IMAGE_BATCH = 400
DEFAULT_PROBE_BATCH = 8192
DEFAULT_WORKERS = 4
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_SEED = 20260829

IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    ref: str
    base_model_or_path: str | None = None


# =============================================================================
# Reproducibility / utilities
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # torch.initial_seed() is already derived from the DataLoader generator.
    s = int(torch.initial_seed() % (2**32))
    random.seed(s)
    np.random.seed(s)


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"


def safe_token(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def path_fingerprint(ref: str) -> dict[str, Any]:
    """Cheap cache invalidation for local files/directories."""
    p = Path(ref)
    if not p.exists():
        return {"ref": ref, "kind": "nonlocal"}
    if p.is_file():
        st = p.stat()
        return {
            "ref": str(p.resolve()),
            "kind": "file",
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }

    # For HF-style directories, fingerprint the files that actually define the
    # model/config without recursively hashing gigabytes.
    names = (
        "model.safetensors",
        "pytorch_model.bin",
        "config.json",
        "configuration_xattn_clip.py",
        "modeling_xattn_clip.py",
    )
    items = []
    for name in names:
        q = p / name
        if q.is_file():
            st = q.stat()
            items.append((name, int(st.st_size), int(st.st_mtime_ns)))
    return {"ref": str(p.resolve()), "kind": "directory", "items": items}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# ImageNet manifest
# =============================================================================

def class_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.startswith("n") and p.name[1:].isdigit()
    )


def scan_split(root: Path, wnids: Sequence[str]) -> tuple[list[str], torch.Tensor, list[int]]:
    paths: list[str] = []
    labels: list[int] = []
    per_class: list[int] = []

    for class_index, wnid in enumerate(tqdm(wnids, desc=f"scan {root.name}", unit="class")):
        d = root / wnid
        if not d.is_dir():
            raise FileNotFoundError(f"Missing class directory: {d}")
        files = sorted(
            str(p)
            for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        per_class.append(len(files))
        paths.extend(files)
        labels.extend([class_index] * len(files))

    return paths, torch.tensor(labels, dtype=torch.long), per_class


def load_devkit_class_mapping(devkit_root: Path) -> dict[int, str]:
    meta_path = devkit_root / "data" / "meta.mat"
    if not meta_path.is_file():
        raise FileNotFoundError(f"ImageNet devkit meta.mat not found: {meta_path}")
    try:
        import scipy.io as sio
    except Exception as exc:
        raise RuntimeError("scipy is required to read the bundled ImageNet devkit meta.mat") from exc

    meta = sio.loadmat(str(meta_path), squeeze_me=True)["synsets"]
    nums_children = list(zip(*meta))[4]
    leaves = [meta[idx] for idx, num_children in enumerate(nums_children) if int(num_children) == 0]
    idcs, wnids = list(zip(*leaves))[:2]
    mapping = {int(idx): str(wnid) for idx, wnid in zip(idcs, wnids)}
    if len(mapping) != EXPECTED_CLASSES or set(mapping) != set(range(1, EXPECTED_CLASSES + 1)):
        raise RuntimeError(f"Unexpected ImageNet devkit leaf mapping: {len(mapping)} entries")
    return mapping


def scan_flat_val(
    val_root: Path,
    train_wnids: Sequence[str],
    devkit_root: Path,
) -> tuple[list[str], torch.Tensor, list[int]]:
    files = sorted(
        p for p in val_root.iterdir()
        if p.is_file() and p.name.startswith("ILSVRC2012_val_") and p.suffix.lower() in IMAGE_EXTS
    )
    gt_path = devkit_root / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"ImageNet validation ground truth not found: {gt_path}")
    ids = [int(x.strip()) for x in gt_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(files) != len(ids):
        raise RuntimeError(f"Flat ImageNet val mismatch: {len(files):,} images vs {len(ids):,} labels")

    id_to_wnid = load_devkit_class_mapping(devkit_root)
    wnid_to_index = {wnid: i for i, wnid in enumerate(train_wnids)}
    labels: list[int] = []
    counts = [0] * len(train_wnids)
    for ilsvrc_id in ids:
        wnid = id_to_wnid.get(int(ilsvrc_id))
        if wnid not in wnid_to_index:
            raise RuntimeError(f"Devkit WNID {wnid!r} for class ID {ilsvrc_id} is absent from train/")
        label = wnid_to_index[wnid]
        labels.append(label)
        counts[label] += 1
    return [str(p) for p in files], torch.tensor(labels, dtype=torch.long), counts


def build_imagenet_manifest(
    train_root: Path,
    val_root: Path,
    devkit_root: Path,
    *,
    strict_counts: bool,
) -> dict[str, Any]:
    train_wnids = class_dirs(train_root)
    if len(train_wnids) != EXPECTED_CLASSES:
        raise RuntimeError(f"Expected 1000 train class folders, found {len(train_wnids)}")

    train_paths, train_labels, train_counts = scan_split(train_root, train_wnids)
    val_wnids = class_dirs(val_root)
    if val_wnids:
        if train_wnids != val_wnids:
            train_only = sorted(set(train_wnids) - set(val_wnids))[:10]
            val_only = sorted(set(val_wnids) - set(train_wnids))[:10]
            raise RuntimeError(
                "Train/val class folders differ. "
                f"train_only={train_only}, val_only={val_only}"
            )
        val_paths, val_labels, val_counts = scan_split(val_root, train_wnids)
        val_layout = "organized_wnid_folders"
    else:
        val_paths, val_labels, val_counts = scan_flat_val(val_root, train_wnids, devkit_root)
        val_layout = "official_flat_with_bundled_devkit"

    if strict_counts:
        if len(train_paths) != EXPECTED_TRAIN:
            raise RuntimeError(
                f"Expected {EXPECTED_TRAIN:,} train images, found {len(train_paths):,}. "
                "Use --no-strict-counts only if this is intentional."
            )
        if len(val_paths) != EXPECTED_VAL:
            raise RuntimeError(
                f"Expected {EXPECTED_VAL:,} val images, found {len(val_paths):,}. "
                "Use --no-strict-counts only if this is intentional."
            )
        if any(x != 50 for x in val_counts):
            bad = [(train_wnids[i], n) for i, n in enumerate(val_counts) if n != 50][:10]
            raise RuntimeError(f"Unexpected val class counts; examples: {bad}")

    digest = hashlib.sha256()
    digest.update(str(train_root.resolve()).encode())
    digest.update(str(val_root.resolve()).encode())
    digest.update(str(devkit_root.resolve()).encode())
    digest.update(val_layout.encode())
    digest.update("\n".join(train_wnids).encode())
    digest.update(str(len(train_paths)).encode())
    digest.update(str(len(val_paths)).encode())

    return {
        "wnids": train_wnids,
        "train_paths": train_paths,
        "train_labels": train_labels,
        "train_per_class": train_counts,
        "val_paths": val_paths,
        "val_labels": val_labels,
        "val_per_class": val_counts,
        "val_layout": val_layout,
        "devkit_root": str(devkit_root.resolve()),
        "fingerprint": digest.hexdigest(),
    }


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[str], labels: torch.Tensor, preprocess: Any):
        self.paths = paths
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
            x = self.preprocess(image)
        return x, self.labels[index]


def make_image_loader(
    paths: Sequence[str],
    labels: torch.Tensor,
    preprocess: Any,
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    ds = ImagePathDataset(paths, labels, preprocess)
    g = torch.Generator()
    g.manual_seed(seed)
    kwargs: dict[str, Any] = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=g,
    )
    if workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
            worker_init_fn=seed_worker,
        )
    return DataLoader(**kwargs)


# =============================================================================
# Model loading / image feature extraction
# =============================================================================

def load_model(spec: ModelSpec, device: str):
    print("\n" + "=" * 100)
    print(f"[model] {spec.key}: {spec.label}")
    print(f"[ref]   {spec.ref}")
    print("=" * 100)

    model, preprocess, info = load_openai_clip_anything(
        clip,
        spec.ref,
        device=device,
        jit=False,
        strict=True,
        base_model_or_path=spec.base_model_or_path,
        allow_unsafe_hf_pickle=False,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    full = bool(is_full_xattn(model, info))
    print(f"[loader] full_xattn={full}")
    return model, preprocess, info, full


def correction_context(model: torch.nn.Module, enabled: bool, is_full: bool):
    if not is_full:
        return nullcontext()
    return temporary_content_correction(model, enabled)


def cache_manifest_expected(
    *,
    spec: ModelSpec,
    split: str,
    root: Path,
    n: int,
    dataset_fp: str,
    cache_dtype: str,
    content_correction: bool | None,
) -> dict[str, Any]:
    return {
        "schema": 2,
        "model_key": spec.key,
        "model_label": spec.label,
        "model_ref": spec.ref,
        "model_fingerprint": path_fingerprint(spec.ref),
        "split": split,
        "split_root": str(root.resolve()),
        "n": int(n),
        "dataset_fingerprint": dataset_fp,
        "l2_normalized": True,
        "preprocess": "model-specific deterministic CLIP eval preprocess",
        "content_correction": content_correction,
        "dtype": cache_dtype,
    }


def manifest_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not bool(got.get("complete", False)):
        return False
    for key, value in expected.items():
        if got.get(key) != value:
            return False
    feature_path = Path(got.get("feature_path", ""))
    return feature_path.is_file()


@torch.inference_mode()
def extract_or_reuse_features(
    *,
    spec: ModelSpec,
    model: torch.nn.Module,
    preprocess: Any,
    info: Any,
    is_full: bool,
    split: str,
    split_root: Path,
    paths: Sequence[str],
    labels: torch.Tensor,
    dataset_fp: str,
    cache_dir: Path,
    cache_dtype: str,
    image_batch: int,
    workers: int,
    device: str,
    seed: int,
    reuse_cache: bool,
) -> Path:
    model_cache = cache_dir / spec.key
    model_cache.mkdir(parents=True, exist_ok=True)
    feature_path = model_cache / f"{split}_features_{cache_dtype}.npy"
    manifest_path = model_cache / f"{split}_features_{cache_dtype}.json"

    content_correction: bool | None = True if is_full else None
    expected = cache_manifest_expected(
        spec=spec,
        split=split,
        root=split_root,
        n=len(paths),
        dataset_fp=dataset_fp,
        cache_dtype=cache_dtype,
        content_correction=content_correction,
    )

    if reuse_cache and manifest_matches(manifest_path, expected):
        got = json.loads(manifest_path.read_text(encoding="utf-8"))
        q = Path(got["feature_path"])
        arr = np.load(q, mmap_mode="r")
        if arr.shape[0] == len(paths):
            print(f"[cache] reuse {spec.key}/{split}: {q} shape={arr.shape} dtype={arr.dtype}")
            return q
        print("[cache] manifest matched but shape did not; rebuilding")

    # Incomplete or stale cache is never trusted.
    if feature_path.exists():
        feature_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    loader = make_image_loader(
        paths,
        labels,
        preprocess,
        batch_size=image_batch,
        workers=workers,
        seed=seed + (0 if split == "train" else 1),
    )

    np_dtype = np.float32 if cache_dtype == "float32" else np.float16
    mmap = None
    feature_dim = None
    cursor = 0
    t0 = time.time()

    print(
        f"[extract] {spec.key}/{split}: n={len(paths):,}, batch={image_batch}, "
        f"workers={workers}, cache_dtype={cache_dtype}"
    )

    ctx = correction_context(model, True, is_full)
    with ctx:
        for images, batch_labels in tqdm(loader, desc=f"{spec.key}/{split}", unit="batch"):
            images = images.to(device, non_blocking=True)
            with inference_autocast(device):
                z = model.encode_image(images)
            if isinstance(z, (tuple, list)):
                z = z[0]
            if not torch.is_tensor(z) or z.ndim != 2:
                raise RuntimeError(f"Unexpected encode_image output: {type(z)} shape={getattr(z, 'shape', None)}")

            z = F.normalize(z.float(), dim=-1, eps=1e-12)
            z_np = z.cpu().numpy().astype(np_dtype, copy=False)

            if mmap is None:
                feature_dim = int(z_np.shape[1])
                mmap = np.lib.format.open_memmap(
                    feature_path,
                    mode="w+",
                    dtype=np_dtype,
                    shape=(len(paths), feature_dim),
                )

            end = cursor + len(z_np)
            mmap[cursor:end] = z_np

            # Cheap ordering sanity check against the manifest labels.
            expected_labels = labels[cursor:end]
            if not torch.equal(batch_labels.cpu().long(), expected_labels):
                raise RuntimeError(f"DataLoader label/order mismatch at rows {cursor}:{end}")
            cursor = end

            del images, z, z_np, batch_labels

    if mmap is None or feature_dim is None:
        raise RuntimeError(f"No features extracted for {spec.key}/{split}")
    if cursor != len(paths):
        raise RuntimeError(f"Extraction ended at {cursor}, expected {len(paths)}")

    mmap.flush()
    del mmap

    elapsed = time.time() - t0
    manifest = dict(expected)
    manifest.update(
        {
            "complete": True,
            "feature_path": str(feature_path.resolve()),
            "shape": [len(paths), feature_dim],
            "feature_dim": feature_dim,
            "elapsed_seconds": elapsed,
        }
    )
    write_json(manifest_path, manifest)
    print(f"[cache] wrote {feature_path} ({human_bytes(feature_path.stat().st_size)})")
    print(f"[extract] elapsed {elapsed / 60.0:.1f} min")
    return feature_path


# =============================================================================
# Linear probe
# =============================================================================

class LinearProbe(nn.Module):
    def __init__(self, dim: int, classes: int = EXPECTED_CLASSES):
        super().__init__()
        self.fc = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def load_features_to_ram(path: Path) -> torch.Tensor:
    arr = np.load(path, mmap_mode=None)
    # np.load owns writable memory; torch shares it without another copy.
    x = torch.from_numpy(arr)
    print(f"[RAM] {path.name}: shape={tuple(x.shape)} dtype={x.dtype} bytes={human_bytes(x.numel() * x.element_size())}")
    return x


@torch.inference_mode()
def evaluate_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str,
    batch_size: int,
) -> tuple[float, float, float]:
    probe.eval()
    n = len(features)
    top1 = 0
    top5 = 0
    loss_sum = 0.0

    for st in tqdm(range(0, n, batch_size), desc="VAL once", unit="batch"):
        en = min(n, st + batch_size)
        x = features[st:en].to(device=device, dtype=torch.float32)
        y = labels[st:en].to(device=device)
        logits = probe(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        pred1 = logits.argmax(dim=1)
        pred5 = logits.topk(5, dim=1).indices
        top1 += int(pred1.eq(y).sum().item())
        top5 += int(pred5.eq(y[:, None]).any(dim=1).sum().item())

    return 100.0 * top1 / n, 100.0 * top5 / n, loss_sum / n


def train_probe_fixed_recipe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    device: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[LinearProbe, dict[str, Any]]:
    if train_features.ndim != 2 or val_features.ndim != 2:
        raise ValueError("Expected [N,D] features")
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError("Train/val embedding dims differ")

    dim = int(train_features.shape[1])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    probe = LinearProbe(dim).to(device=device, dtype=torch.float32)
    optimizer = AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(train_features)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history: list[dict[str, Any]] = []
    t0 = time.time()

    print(
        f"[probe] fixed recipe: AdamW lr={lr:g} wd={weight_decay:g} "
        f"epochs={epochs} batch={batch_size}; NO validation peeking"
    )

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        correct = 0
        seen = 0
        e0 = time.time()

        progress = tqdm(range(0, n, batch_size), desc=f"probe epoch {epoch+1}/{epochs}", unit="batch")
        for st in progress:
            idx = perm[st : min(n, st + batch_size)]
            # Gather on CPU, then transfer.  Cached features may be FP16 or FP32;
            # classifier arithmetic stays FP32 for a clean, stable probe.
            x = train_features[idx].to(device=device, dtype=torch.float32)
            y = train_labels[idx].to(device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = probe(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            b = len(idx)
            loss_sum += float(loss.item()) * b
            correct += int(logits.argmax(dim=1).eq(y).sum().item())
            seen += b

        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / seen,
            "train_top1": 100.0 * correct / seen,
            "elapsed_seconds": time.time() - e0,
        }
        history.append(row)
        print(
            f"[probe] epoch {epoch+1:02d}: loss={row['train_loss']:.6f} "
            f"top1={row['train_top1']:.3f}% time={row['elapsed_seconds']/60:.2f} min"
        )

    # Crucial: official val is touched exactly once, after the fixed number of epochs.
    val_top1, val_top5, val_loss = evaluate_probe(
        probe,
        val_features,
        val_labels,
        device=device,
        batch_size=batch_size,
    )
    stats = {
        "history": history,
        "val_top1": val_top1,
        "val_top5": val_top5,
        "val_loss": val_loss,
        "probe_elapsed_seconds": time.time() - t0,
    }
    return probe, stats


# =============================================================================
# Output
# =============================================================================

def print_ascii_results(rows: Sequence[dict[str, Any]]) -> None:
    headers = ("model", "train_n", "val_n", "dim", "Top-1", "Top-5", "val CE")
    body = []
    for r in rows:
        body.append(
            (
                str(r["model"]),
                f"{int(r['train_n']):,}",
                f"{int(r['val_n']):,}",
                str(int(r["feature_dim"])),
                f"{float(r['top1']):.3f}%",
                f"{float(r['top5']):.3f}%",
                f"{float(r['val_loss']):.5f}",
            )
        )
    widths = [len(h) for h in headers]
    for row in body:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]

    def line(vals: Sequence[str]) -> str:
        return " | ".join(v.ljust(w) for v, w in zip(vals, widths))

    print("\n" + "=" * (sum(widths) + 3 * (len(widths) - 1)))
    print(line(headers))
    print("-+-".join("-" * w for w in widths))
    for row in body:
        print(line(row))
    print("=" * (sum(widths) + 3 * (len(widths) - 1)))


def save_results(out: Path, rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "imagenet_trainval_linear_probe.csv"
    fields = [
        "model",
        "label",
        "ref",
        "train_n",
        "val_n",
        "feature_dim",
        "top1",
        "top5",
        "val_loss",
        "cache_dtype",
        "content_correction",
        "probe_elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    write_json(out / "run_config.json", config)
    write_json(out / "results.json", list(rows))

    lines = [
        "IMAGENET-1K FROZEN-FEATURE LINEAR PROBE: OFFICIAL TRAIN -> VAL",
        "=" * 78,
        "",
        f"train_n={config['train_n']:,}",
        f"val_n={config['val_n']:,}",
        "val policy: evaluated exactly once after the fixed training recipe",
        "features: model-specific CLIP eval preprocess; L2-normalized",
        "",
    ]
    for r in rows:
        lines.append(
            f"{r['model']:<12} Top-1={r['top1']:.3f}%  "
            f"Top-5={r['top5']:.3f}%  CE={r['val_loss']:.5f}"
        )
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ImageNet-1k frozen-feature linear probe: full official train -> held-out val"
    )
    p.add_argument("--train", type=Path, default=Path(DEFAULT_TRAIN))
    p.add_argument("--val", type=Path, default=Path(DEFAULT_VAL))
    p.add_argument("--devkit", type=Path, default=DEFAULT_DEVKIT)
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    p.add_argument("--base-model-or-path", default=None)
    p.add_argument("--output", "--output-dir", dest="output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--image-batch", type=int, default=DEFAULT_IMAGE_BATCH)
    p.add_argument("--probe-batch", type=int, default=DEFAULT_PROBE_BATCH)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dtype", choices=("float32", "float16"), default="float32")
    p.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-counts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "feature_cache"
    probe_dir = args.output / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    print("\n[protocol] official ImageNet TRAIN -> held-out VAL")
    print("[protocol] VAL is NOT used for early stopping / model selection")
    print(f"[train] {args.train}")
    print(f"[val]   {args.val}")
    print(f"[devkit] {args.devkit}")
    print(f"[out]   {args.output}")
    print(f"[device] {args.device}; tf32={args.tf32}")

    manifest = build_imagenet_manifest(
        args.train,
        args.val,
        args.devkit,
        strict_counts=args.strict_counts,
    )
    train_paths: list[str] = manifest["train_paths"]
    train_labels: torch.Tensor = manifest["train_labels"]
    val_paths: list[str] = manifest["val_paths"]
    val_labels: torch.Tensor = manifest["val_labels"]

    print(
        f"[ImageNet] classes={len(manifest['wnids'])} "
        f"train={len(train_paths):,} val={len(val_paths):,} layout={manifest['val_layout']}"
    )

    spec = ModelSpec(
        key=safe_token(args.model_alias),
        label=args.model_alias,
        ref=args.model,
        base_model_or_path=args.base_model_or_path,
    )

    rows: list[dict[str, Any]] = []

    for model_index, spec in enumerate((spec,)):
        model, preprocess, info, full = load_model(spec, args.device)

        train_cache = extract_or_reuse_features(
            spec=spec,
            model=model,
            preprocess=preprocess,
            info=info,
            is_full=full,
            split="train",
            split_root=args.train,
            paths=train_paths,
            labels=train_labels,
            dataset_fp=manifest["fingerprint"],
            cache_dir=cache_dir,
            cache_dtype=args.cache_dtype,
            image_batch=args.image_batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed + 100 * model_index,
            reuse_cache=args.reuse_cache,
        )
        val_cache = extract_or_reuse_features(
            spec=spec,
            model=model,
            preprocess=preprocess,
            info=info,
            is_full=full,
            split="val",
            split_root=args.val,
            paths=val_paths,
            labels=val_labels,
            dataset_fp=manifest["fingerprint"],
            cache_dir=cache_dir,
            cache_dtype=args.cache_dtype,
            image_batch=args.image_batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed + 100 * model_index,
            reuse_cache=args.reuse_cache,
        )

        # Free the expensive backbone before loading ~4 GiB of train features.
        del model, preprocess, info
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        train_x = load_features_to_ram(train_cache)
        val_x = load_features_to_ram(val_cache)
        feature_dim = int(train_x.shape[1])

        probe, stats = train_probe_fixed_recipe(
            train_x,
            train_labels,
            val_x,
            val_labels,
            device=args.device,
            batch_size=args.probe_batch,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,  # same initialization/training seed for every model
        )

        probe_path = probe_dir / f"{safe_token(spec.key)}_linear_probe.pt"
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in probe.state_dict().items()},
                "feature_dim": feature_dim,
                "classes": EXPECTED_CLASSES,
                "model_key": spec.key,
                "model_ref": spec.ref,
                "recipe": {
                    "optimizer": "AdamW",
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                    "batch_size": args.probe_batch,
                    "seed": args.seed,
                    "val_peeking": False,
                },
            },
            probe_path,
        )

        history_path = probe_dir / f"{safe_token(spec.key)}_train_history.json"
        write_json(history_path, stats["history"])

        row = {
            "model": spec.key,
            "label": spec.label,
            "ref": spec.ref,
            "train_n": len(train_x),
            "val_n": len(val_x),
            "feature_dim": feature_dim,
            "top1": float(stats["val_top1"]),
            "top5": float(stats["val_top5"]),
            "val_loss": float(stats["val_loss"]),
            "cache_dtype": args.cache_dtype,
            "content_correction": True if full else None,
            "probe_elapsed_seconds": float(stats["probe_elapsed_seconds"]),
        }
        rows.append(row)

        print(
            f"\n[result] {spec.key}: Top-1={row['top1']:.3f}% "
            f"Top-5={row['top5']:.3f}% CE={row['val_loss']:.5f}"
        )
        print_ascii_results(rows)

        # Save after every model so a multi-hour run is crash-resilient.
        config = {
            "protocol": "official ILSVRC2012 train -> held-out val frozen-feature linear probe",
            "train_root": str(args.train),
            "val_root": str(args.val),
            "train_n": len(train_paths),
            "val_n": len(val_paths),
            "classes": len(manifest["wnids"]),
            "dataset_fingerprint": manifest["fingerprint"],
            "model": {"alias": args.model_alias, "path": args.model, "base_model_or_path": args.base_model_or_path},
            "models_completed": [r["model"] for r in rows],
            "image_batch": args.image_batch,
            "probe_batch": args.probe_batch,
            "workers": args.workers,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "cache_dtype": args.cache_dtype,
            "reuse_cache": args.reuse_cache,
            "tf32": args.tf32,
            "feature_preprocessing": "model-specific deterministic CLIP eval preprocess; L2 normalized embeddings",
            "val_policy": "evaluated once after fixed epochs; no early stopping or hyperparameter selection on val",
            "full_xattn_image_lane": "RN backbone + separate CONTENT correction ON; no candidate-conditioned READ bridge",
            "val_layout": manifest["val_layout"],
            "devkit_root": manifest["devkit_root"],
        }
        save_results(args.output, rows, config)

        del probe, train_x, val_x
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print_ascii_results(rows)
    print(f"\n[done] {args.output / 'imagenet_trainval_linear_probe.csv'}")
    print(f"[done] {args.output / 'SUMMARY.txt'}")


if __name__ == "__main__":
    freeze_support()
    main()
