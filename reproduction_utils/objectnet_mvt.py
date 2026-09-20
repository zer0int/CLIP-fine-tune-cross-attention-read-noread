"""Canonical ObjectNet-MVT population support for paper-reproduction probes.

The workspace/sink probes only need a diverse natural-image population.  Public
reproduction therefore reuses the repository's benchmark ObjectNet-MVT install
and the bundled deduplicated 4,771-image index, then derives a deterministic
480-image label/domain-balanced manifest for the expensive mechanistic probes.

Dataset acquisition itself stays in ``benchmark_utils.data_setup`` /
``prepare_objectnet_mvt.py`` so reproduction has no second downloader or dataset
interpretation to drift out of sync with the benchmark path.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

EXPECTED_MVT_IMAGES = 4771
DEFAULT_SAMPLE_SIZE = 480
DEFAULT_SAMPLE_SEED = 20260915
DEDUP_CSV_REL = Path("utils_datasets/mvt/human_responses_dedup.csv")


def dedup_csv_path(project_root: Path) -> Path:
    return project_root / DEDUP_CSV_REL


def load_dedup_rows(project_root: Path) -> list[dict[str, str]]:
    """Load and validate the canonical one-row-per-image MVT index."""
    path = dedup_csv_path(project_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"ObjectNet-MVT deduplicated index is missing: {path}. "
            "The repository should include utils_datasets/mvt/human_responses_dedup.csv."
        )

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "label", "objectnet"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected MVT CSV schema in {path}: {reader.fieldnames}")
        for line_no, raw in enumerate(reader, start=2):
            name = Path(str(raw.get("image") or "")).name
            label = str(raw.get("label") or "").strip()
            objectnet_raw = str(raw.get("objectnet") or "").strip().lower()
            if not name or not label:
                raise ValueError(f"Missing image/label at {path}:{line_no}")
            if name in seen:
                raise ValueError(f"Deduplicated MVT index repeats image {name!r} at {path}:{line_no}")
            seen.add(name)
            if objectnet_raw in {"true", "1", "yes"}:
                domain = "objectnet"
            elif objectnet_raw in {"false", "0", "no"}:
                domain = "imagenet"
            else:
                raise ValueError(f"Invalid objectnet flag {raw.get('objectnet')!r} at {path}:{line_no}")
            rows.append(
                {
                    "image": name,
                    "label": label,
                    "domain": domain,
                    "image_duration": str(raw.get("image_duration") or ""),
                }
            )

    if len(rows) != EXPECTED_MVT_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_MVT_IMAGES:,} unique MVT images in {path}, found {len(rows):,}"
        )
    return rows


def expected_names(project_root: Path) -> set[str]:
    return {row["image"] for row in load_dedup_rows(project_root)}


def root_inventory(root: Path | None) -> set[str]:
    if root is None or not root.is_dir():
        return set()
    return {p.name for p in root.iterdir() if p.is_file()}


def is_complete_root(root: Path | None, project_root: Path) -> bool:
    """True only when every filename from the canonical dedup index is present."""
    if root is None or not root.is_dir():
        return False
    return expected_names(project_root).issubset(root_inventory(root))


def missing_names(root: Path | None, project_root: Path) -> list[str]:
    return sorted(expected_names(project_root) - root_inventory(root))


def _selection_hash(seed: int, text: str) -> str:
    return hashlib.sha256(f"{int(seed)}\0{text}".encode("utf-8")).hexdigest()


def _balanced_sample(rows: Iterable[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    """Round-robin over label×domain strata, hash-ordering within each stratum.

    Labels are traversed in deterministic hash order and each label emits one item
    from every available domain before moving on.  For the canonical 480-image
    sample (50 labels × 2 domains), this gives exactly 240 ObjectNet and 240
    ImageNet stimuli, with 4–5 images from every label×domain stratum.
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["label"], row["domain"])].append(row)
    if not groups:
        return []

    for key, items in groups.items():
        items.sort(
            key=lambda row: (
                _selection_hash(seed, f"{key[0]}\0{key[1]}\0{row['image']}"),
                row["image"].lower(),
            )
        )

    labels = sorted({label for label, _domain in groups}, key=lambda label: _selection_hash(seed, f"label\0{label}"))
    domains = sorted({domain for _label, domain in groups})

    selected: list[dict[str, str]] = []
    depth = 0
    while len(selected) < sample_size:
        added = False
        for label in labels:
            for domain in domains:
                items = groups.get((label, domain), [])
                if depth < len(items):
                    selected.append(items[depth])
                    added = True
                    if len(selected) == sample_size:
                        break
            if len(selected) == sample_size:
                break
        if not added:
            break
        depth += 1
    return selected


def build_workspace_manifest(
    *,
    project_root: Path,
    image_root: Path,
    output_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> Path:
    """Create the deterministic MVT manifest consumed by workspace probes."""
    rows = load_dedup_rows(project_root)
    available = root_inventory(image_root)
    missing = [row["image"] for row in rows if row["image"] not in available]
    if missing:
        raise RuntimeError(
            f"ObjectNet-MVT root is missing {len(missing):,}/{len(rows):,} canonical images: {image_root}. "
            f"Examples: {missing[:8]}"
        )
    if sample_size <= 0 or sample_size > len(rows):
        raise ValueError(f"objectnet_mvt_sample_size must be in [1, {len(rows)}], got {sample_size}")

    selected = _balanced_sample(rows, sample_size, seed)
    if len(selected) != sample_size:
        raise RuntimeError(f"Could select only {len(selected)} MVT images; requested {sample_size}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stim_id",
        "source",
        "path",
        "image_name",
        "label",
        "mvt_domain",
        "image_duration",
        "selection_rank",
        "selection_hash",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(selected):
            image_path = image_root / row["image"]
            writer.writerow(
                {
                    "stim_id": f"mvt:{row['domain']}:{row['image']}",
                    "source": f"mvt_{row['domain']}",
                    "path": str(image_path.resolve()),
                    "image_name": row["image"],
                    "label": row["label"],
                    "mvt_domain": row["domain"],
                    "image_duration": row["image_duration"],
                    "selection_rank": rank,
                    "selection_hash": _selection_hash(seed, row["image"]),
                }
            )

    by_domain: dict[str, int] = defaultdict(int)
    by_label: dict[str, int] = defaultdict(int)
    by_stratum: dict[str, int] = defaultdict(int)
    for row in selected:
        by_domain[row["domain"]] += 1
        by_label[row["label"]] += 1
        by_stratum[f"{row['label']}::{row['domain']}"] += 1

    metadata = {
        "dataset": "ObjectNet-MVT",
        "deduplicated_index": str(dedup_csv_path(project_root).resolve()),
        "canonical_unique_images": len(rows),
        "image_root": str(image_root.resolve()),
        "sample_size": sample_size,
        "seed": seed,
        "strategy": "label_domain_round_robin_sha256",
        "domain_counts": dict(sorted(by_domain.items())),
        "label_counts": dict(sorted(by_label.items())),
        "stratum_count_min": min(by_stratum.values()),
        "stratum_count_max": max(by_stratum.values()),
        "manifest": str(output_path.resolve()),
        "selection_sha256": hashlib.sha256(
            "\n".join(row["image"] for row in selected).encode("utf-8")
        ).hexdigest(),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path
