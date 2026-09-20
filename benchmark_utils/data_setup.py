"""Interactive benchmark-data setup and validation helpers."""
from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .config import optional_path

COCO_VAL2014_URL = "https://images.cocodataset.org/zips/val2014.zip"
COCO_VAL2017_URL = "https://images.cocodataset.org/zips/val2017.zip"
COCO_EXPECTED = {"val2014": 40504, "val2017": 5000}
SCAM_REPO = "BLISS-e-V/SCAM"
RTA_REPO = "zer0int/RTA-100-Triplet"


def yn(prompt: str, *, default: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes:
        print(f"{prompt} [auto: yes]")
        return True
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        value = input(prompt + suffix).strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please enter y or n.")


def ask_path(prompt: str, default: Path | None = None) -> Path | None:
    default_text = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{default_text}: ").strip().strip('"')
    if not value:
        return default
    return Path(value).expanduser()


def _safe_extract(zf: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for info in zf.infolist():
        if info.is_dir():
            continue
        rel = PurePosixPath(info.filename.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Unsafe ZIP member: {info.filename}")
        target = (destination / Path(*rel.parts)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Unsafe ZIP member: {info.filename}") from exc
    zf.extractall(destination)


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "CLIP-ModeMUX benchmark setup/1.0"})
    print(f"[download] {url}")
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
        total_text = response.headers.get("Content-Length")
        total = int(total_text) if total_text and total_text.isdigit() else None
        copied = 0
        next_report = 64 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            if copied >= next_report:
                if total:
                    print(f"  {copied / 2**20:.1f}/{total / 2**20:.1f} MiB")
                else:
                    print(f"  {copied / 2**20:.1f} MiB")
                next_report += 64 * 1024 * 1024
    os.replace(part, destination)
    print(f"[download] saved {destination} ({destination.stat().st_size / 2**20:.1f} MiB)")
    return destination


def image_count(root: Path) -> int:
    if not root.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def install_coco_split(split: str, data_root: Path) -> Path:
    if split not in COCO_EXPECTED:
        raise ValueError(split)
    coco_root = data_root / "COCO"
    image_root = coco_root / split
    expected = COCO_EXPECTED[split]
    if image_count(image_root) == expected:
        print(f"[COCO] {split}: already ready at {image_root} ({expected:,} images)")
        return image_root

    url = COCO_VAL2014_URL if split == "val2014" else COCO_VAL2017_URL
    downloads = coco_root / "_downloads"
    archive = downloads / f"{split}.zip"
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        download_file(url, archive)
    print(f"[COCO] extracting {archive} -> {coco_root}")
    coco_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        _safe_extract(zf, coco_root)
    count = image_count(image_root)
    if count != expected:
        raise RuntimeError(f"COCO {split}: expected {expected:,} images, found {count:,} in {image_root}")
    print(f"[COCO] {split}: READY ({count:,} images)")
    return image_root


def install_objectnet_mvt(project_root: Path, data_root: Path) -> Path:
    # Reuse the release helper that already handles the official ZIP-inside-ZIP
    # layout and validates all 4,771 filenames + image decoding.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from prepare_objectnet_mvt import STIMULI_URL, download_once, extract_expected, load_expected

    csv_path = project_root / "utils_datasets" / "mvt" / "human_responses_dedup.csv"
    expected = load_expected(csv_path)
    root = data_root / "ObjectNet-MVT"
    images = root / "all"
    existing = {p.name for p in images.iterdir()} if images.is_dir() else set()
    if set(expected).issubset(existing):
        print(f"[ObjectNet-MVT] already ready at {images} ({len(expected):,} images)")
        return images
    archive = root / "_downloads" / "flash_data_release_2023.zip"
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        download_once(STIMULI_URL, archive, timeout=120.0)
    report = extract_expected(archive, expected, images, reset=False)
    print(f"[ObjectNet-MVT] READY: {report['image_count']:,} images at {images}")
    return images


def warm_typo_cache(cache_dir: Path | None) -> None:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("`datasets` is required; install requirements.txt first") from exc
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[typo] caching {SCAM_REPO}")
    load_dataset(SCAM_REPO, split="train", **kwargs)
    print(f"[typo] caching {RTA_REPO}")
    load_dataset(RTA_REPO, split="train", **kwargs)
    print("[typo] READY")


def validate_imagenet_root(root: Path | None, project_root: Path) -> tuple[bool, str]:
    if root is None:
        return False, "not configured"
    train = root / "train"
    val = root / "val"
    if not train.is_dir() or not val.is_dir():
        return False, f"expected train/ and val/ under {root}"
    train_classes = sum(1 for p in train.iterdir() if p.is_dir() and p.name.startswith("n"))
    if train_classes != 1000:
        return False, f"train/ has {train_classes} WNID folders; expected 1000"
    # Validation can be organized in class folders OR be the official flat archive.
    val_classes = sum(1 for p in val.iterdir() if p.is_dir() and p.name.startswith("n"))
    flat_count = sum(1 for p in val.iterdir() if p.is_file() and p.name.startswith("ILSVRC2012_val_"))
    if val_classes == 1000:
        return True, "organized train/ + organized val/"
    if flat_count == 50000:
        devkit = project_root / "utils_datasets" / "imagenet" / "ILSVRC2012_devkit_t12"
        if (devkit / "data" / "meta.mat").is_file() and (devkit / "data" / "ILSVRC2012_validation_ground_truth.txt").is_file():
            return True, "organized train/ + official flat val/ (bundled devkit labels)"
        return False, "flat val found, but bundled devkit metadata is missing"
    return False, f"val/ has neither 1000 WNID folders nor 50,000 official flat validation images"


def status_rows(cfg: Mapping[str, Any], project_root: Path) -> list[tuple[str, str, str]]:
    data = cfg["datasets"]
    rows: list[tuple[str, str, str]] = []

    hf_cache = optional_path(cfg.get("hf_cache_dir"), project_root)
    rows.append(("SCAM + RTA", "READY/ON-DEMAND", str(hf_cache) if hf_cache else "Hugging Face default cache"))

    mvt = optional_path(data.get("objectnet_mvt_root"), project_root)
    if mvt and mvt.is_dir():
        count = image_count(mvt)
        rows.append(("ObjectNet-MVT", "READY" if count >= 4771 else "INCOMPLETE", f"{mvt} ({count:,} files)"))
    else:
        rows.append(("ObjectNet-MVT", "MISSING", str(mvt) if mvt else "not configured"))

    for key, split in (("coco_val2014_root", "COCO val2014"), ("coco_val2017_root", "COCO val2017")):
        root = optional_path(data.get(key), project_root)
        count = image_count(root) if root else 0
        expected = COCO_EXPECTED[split.split()[-1]]
        rows.append((split, "READY" if count == expected else "MISSING", f"{root or 'not configured'}" + (f" ({count:,}/{expected:,})" if root else "")))

    imagenet = optional_path(data.get("imagenet_root"), project_root)
    ok, note = validate_imagenet_root(imagenet, project_root)
    rows.append(("ImageNet-1k", "READY" if ok else "OPTIONAL/MISSING", f"{imagenet or 'not configured'} — {note}"))
    return rows
