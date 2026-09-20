#!/usr/bin/env python3
"""Prepare the external training data required by CLIP Cross-Attn MUX.

This utility reconstructs the *runtime filesystem* expected by the training code
from four sources:

1. zer0int/CLIP-Cross-Attn-MUX-Training-Data-Pack
   - CLEVR derivative (Parquet -> PNG files)
   - handwriting overlays (Parquet -> PNG files)
   - salt-and-pepper images (Parquet -> PNG files)
   - project-authored runtime labels/manifests/word banks
   - portable ImageNet derivative recipes (manifests only)
2. shwetkm/TextCaps-Caption-Summary
   - upstream image bytes only; the project's exact TextCaps manifests come from
     the training-data pack
3. SPRIGHT-T2I/spright_coco
   - upstream WebDataset image bytes only; project labels come from the pack
4. Evaluation-only typo benchmarks
   - BLISS-e-V/SCAM
   - zer0int/RTA-100-Triplet
   Their exact revisions are prefetched/pinned for epoch-end monitoring only.

ImageNet itself is never downloaded or redistributed. Pass --imagenet-root to an
existing user-provided ImageNet installation. The current version of this script
installs the *recipes* for the two ImageNet-derived corpora and writes their
final target roots into training_config.local.json, but does not yet render the
ImageNet text/handwriting derivatives. That construction step is intentionally
kept separate until the exact renderer has been clean-room verified.

The script is restartable: existing correct outputs are reused, and --reset
removes only the prepared data root (never the user's ImageNet tree or HF cache).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

PACK_REPO_ID = "zer0int/CLIP-Cross-Attn-MUX-Training-Data-Pack"
TEXTCAPS_REPO_ID = "shwetkm/TextCaps-Caption-Summary"
SPRIGHT_REPO_ID = "SPRIGHT-T2I/spright_coco"
SCAM_REPO_ID = "BLISS-e-V/SCAM"
RTA_REPO_ID = "zer0int/RTA-100-Triplet"

EXPECTED = {
    "clevr_train": 3824,
    "clevr_validation": 394,
    "handwriting": 206,
    "salt_n_pepper": 186,
    "textcaps_train": 21953,
    "textcaps_validation": 3166,
    "spright_label_images": 40504,
}

PACK_RAW_FILES = {
    "runtime_metadata/coco_spright/short-coco-spright-train-0_9.json": "coco_spright/labels/short-coco-spright-train-0_9.json",
    "runtime_metadata/coco_spright/short-coco-spright-train-0_9_gpt_oss.json": "coco_spright/labels/short-coco-spright-train-0_9_gpt_oss.json",
    "runtime_metadata/coco_spright/short-coco-spright-val-10_11.json": "coco_spright/labels/short-coco-spright-val-10_11.json",
    "runtime_metadata/coco_spright/short-coco-spright-val-10_11_gpt_oss.json": "coco_spright/labels/short-coco-spright-val-10_11_gpt_oss.json",
    "runtime_metadata/coco_spright/coco_spright_reading_supervised_conf999.json": "coco_spright/labels/coco_spright_reading_supervised_conf999.json",
    "runtime_metadata/imagenet/imagenet_wnid_to_class.json": "imagenet_support/imagenet_wnid_to_class.json",
    "runtime_metadata/overlay_selection/overlay_words_gmp_diverse.csv": "overlay_selection/overlay_words_gmp_diverse.csv",
    "runtime_metadata/overlay_selection/overlay_words_gmp_diverse.txt": "overlay_selection/overlay_words_gmp_diverse.txt",
    "runtime_metadata/overlay_selection/orthographic_pairs_selected.csv": "overlay_selection/orthographic_pairs_selected.csv",
    "runtime_metadata/textcaps/manifests/train.jsonl": "textcaps/manifests/train.jsonl",
    "runtime_metadata/textcaps/manifests/validation.jsonl": "textcaps/manifests/validation.jsonl",
    "shareable_assets/clevr/clevr_subset_objects_compact.jsonl": "clevr/clevr_subset_objects_compact.jsonl",
    "shareable_assets/handwriting_overlays/manifest.jsonl": "handwriting_overlays/manifest.jsonl",
    "restricted_recipes/imagenet_clip_text/manifests/train.jsonl": "imagenet_clip_text/manifests/train.jsonl",
    "restricted_recipes/imagenet_clip_text/manifests/val.jsonl": "imagenet_clip_text/manifests/val.jsonl",
    "restricted_recipes/imagenet_handwriting/manifests/train.jsonl": "imagenet_clip_text_handwriting/manifests/train.jsonl",
    "restricted_recipes/imagenet_handwriting/manifests/val.jsonl": "imagenet_clip_text_handwriting/manifests/val.jsonl",
}

CONFIG_PATHS = {
    "imagenet_wnid_json": "imagenet_support/imagenet_wnid_to_class.json",
    "overlay_selection": "overlay_selection",
    "handwriting_overlays": "handwriting_overlays",
    "imagenet_text_root": "imagenet_clip_text",
    "textcaps_root": "textcaps",
    "coco_root": "coco_spright",
    "coco_train_json": "coco_spright/labels/short-coco-spright-train-0_9.json",
    "coco_train_gpt_json": "coco_spright/labels/short-coco-spright-train-0_9_gpt_oss.json",
    "coco_val_json": "coco_spright/labels/short-coco-spright-val-10_11.json",
    "coco_val_gpt_json": "coco_spright/labels/short-coco-spright-val-10_11_gpt_oss.json",
    "coco_train_reading_json": "coco_spright/labels/coco_spright_reading_supervised_conf999.json",
    "imagenet_handwriting_root": "imagenet_clip_text_handwriting",
    "clevr_train_root": "clevr/clevr_images_train_square",
    "clevr_val_root": "clevr/clevr_images_val_square",
    "clevr_metadata_jsonl": "clevr/clevr_subset_objects_compact.jsonl",
    "salt_n_pepper_root": "salt_n_pepper",
}


@dataclass
class PrepareSummary:
    schema_version: int
    pack_repo_id: str
    pack_revision: str
    textcaps_repo_id: str
    textcaps_revision: Optional[str]
    spright_repo_id: str
    spright_revision: Optional[str]
    scam_repo_id: str
    scam_revision: Optional[str]
    rta_repo_id: str
    rta_revision: Optional[str]
    counts: Dict[str, int]
    imagenet_root: Optional[str]
    imagenet_derivatives_ready: bool
    generated_config: Optional[str]


def log(message: str) -> None:
    print(f"[xattn-data] {message}", flush=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def ensure_repo_file(snapshot_root: Path, rel: str) -> Path:
    path = snapshot_root / PurePosixPath(rel)
    if not path.is_file():
        raise FileNotFoundError(f"Training-data pack is missing required file: {rel}")
    return path


def copy_runtime_files(snapshot_root: Path, data_root: Path) -> int:
    copied = 0
    for source_rel, target_rel in PACK_RAW_FILES.items():
        source = ensure_repo_file(snapshot_root, source_rel)
        target = data_root / PurePosixPath(target_rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied


def resolve_pack_revision(repo_id: str, requested_revision: Optional[str], cache_dir: Optional[Path]) -> Tuple[str, Path]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required. Install it with: pip install -U huggingface_hub") from exc

    api = HfApi()
    info = api.dataset_info(repo_id=repo_id, revision=requested_revision)
    sha = str(info.sha)
    log(f"pack revision: {sha}")
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=sha,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return sha, Path(snapshot)


def verify_pack_parquet_hashes(snapshot_root: Path) -> Dict[str, Dict[str, int]]:
    manifest_path = ensure_repo_file(snapshot_root, "HF_PACK_MANIFEST.json")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("HF_PACK_MANIFEST.json must be an object")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("HF_PACK_MANIFEST.json contains no shard records")
    configs: Dict[str, Dict[str, int]] = {}
    for item in shards:
        if not isinstance(item, Mapping):
            raise ValueError("Invalid shard record in HF_PACK_MANIFEST.json")
        rel = str(item["path"])
        expected_hash = str(item["sha256"]).casefold()
        expected_rows = int(item["rows"])
        path = ensure_repo_file(snapshot_root, rel)
        actual_hash = sha256_file(path)
        if actual_hash.casefold() != expected_hash:
            raise RuntimeError(f"Pack shard checksum mismatch: {rel}")
        config = str(item["config"])
        split = str(item["split"])
        configs.setdefault(config, {})[split] = configs.setdefault(config, {}).get(split, 0) + expected_rows
    return configs


def parquet_image_bytes(image_value: Any, *, shard: Path, row_index: int) -> Tuple[bytes, Optional[str]]:
    if not isinstance(image_value, Mapping):
        raise RuntimeError(f"Expected image struct in {shard} row {row_index}, got {type(image_value).__name__}")
    data = image_value.get("bytes")
    path = image_value.get("path")
    if data is None:
        raise RuntimeError(f"Embedded image bytes are missing in {shard} row {row_index}; path={path!r}")
    return bytes(data), None if path is None else str(path)


def extract_pack_parquet(snapshot_root: Path, data_root: Path) -> Dict[str, int]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to unpack the HF training pack. Install it with: pip install -U pyarrow") from exc

    jobs = [
        (
            "clevr_train",
            "shareable_assets/clevr/data/train-*.parquet",
            data_root / "clevr/clevr_images_train_square",
            EXPECTED["clevr_train"],
        ),
        (
            "clevr_validation",
            "shareable_assets/clevr/data/validation-*.parquet",
            data_root / "clevr/clevr_images_val_square",
            EXPECTED["clevr_validation"],
        ),
        (
            "handwriting",
            "shareable_assets/handwriting_overlays/data/train-*.parquet",
            data_root / "handwriting_overlays/images",
            EXPECTED["handwriting"],
        ),
        (
            "salt_n_pepper",
            "shareable_assets/salt_n_pepper/data/train-*.parquet",
            data_root / "salt_n_pepper",
            EXPECTED["salt_n_pepper"],
        ),
    ]

    counts: Dict[str, int] = {}
    for name, pattern, target_dir, expected in jobs:
        target_dir.mkdir(parents=True, exist_ok=True)
        shards = sorted(snapshot_root.glob(pattern))
        if not shards:
            raise FileNotFoundError(f"No Parquet shards match {pattern}")
        seen_names: Set[str] = set()
        count = 0
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=128, columns=["image", "file_name"]):
                image_col = batch.column(batch.schema.get_field_index("image"))
                name_col = batch.column(batch.schema.get_field_index("file_name"))
                for local_index in range(batch.num_rows):
                    file_name = str(name_col[local_index].as_py())
                    basename = PureWindowsPath(file_name).name
                    if not basename or basename != PurePosixPath(basename).name:
                        raise RuntimeError(f"Unsafe/nonportable Parquet file_name: {file_name!r}")
                    if basename in seen_names:
                        raise RuntimeError(f"Duplicate image basename while unpacking {name}: {basename}")
                    seen_names.add(basename)
                    image_value = image_col[local_index].as_py()
                    data, embedded_path = parquet_image_bytes(image_value, shard=shard, row_index=count)
                    if embedded_path is not None and PureWindowsPath(embedded_path).name != basename:
                        raise RuntimeError(
                            f"Parquet image path/file_name mismatch in {shard}: {embedded_path!r} vs {basename!r}"
                        )
                    target = target_dir / basename
                    if target.is_file() and target.read_bytes() == data:
                        pass
                    else:
                        target.write_bytes(data)
                    count += 1
        if count != expected:
            raise RuntimeError(f"{name}: expected {expected} rows, unpacked {count}")
        loose_count = sum(1 for p in target_dir.iterdir() if p.is_file())
        if loose_count != expected:
            raise RuntimeError(f"{name}: expected {expected} files after unpack, found {loose_count}")
        counts[name] = count
        log(f"unpacked {name}: {count} images")
    return counts


def expected_textcaps_rows(manifest_path: Path, split: str) -> Dict[str, str]:
    rows = read_jsonl(manifest_path)
    expected_count = EXPECTED[f"textcaps_{split}"]
    if len(rows) != expected_count:
        raise RuntimeError(f"TextCaps {split}: expected {expected_count} manifest rows, found {len(rows)}")
    result: Dict[str, str] = {}
    for row in rows:
        image_id = str(row.get("image_id") or "").strip()
        rel = str(row.get("image") or "").replace("\\", "/")
        if not image_id or not rel:
            raise RuntimeError(f"TextCaps {split} manifest row missing image_id/image")
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts:
            raise RuntimeError(f"Unsafe TextCaps image path in manifest: {rel!r}")
        if len(pure.parts) < 3 or pure.parts[0] != "images" or pure.parts[1] != split:
            raise RuntimeError(f"Unexpected TextCaps {split} image path: {rel!r}")
        if image_id in result:
            raise RuntimeError(f"Duplicate TextCaps image_id in manifest: {image_id}")
        result[image_id] = rel
    return result


def write_hf_image_value(image_value: Any, target: Path) -> None:
    if not isinstance(image_value, Mapping):
        raise RuntimeError(f"Expected Hugging Face image struct, got {type(image_value).__name__}")
    data = image_value.get("bytes")
    cached_path = image_value.get("path")
    target.parent.mkdir(parents=True, exist_ok=True)
    if data is not None:
        payload = bytes(data)
        if target.is_file() and target.read_bytes() == payload:
            return
        target.write_bytes(payload)
        return
    if cached_path:
        source = Path(str(cached_path))
        if not source.is_file():
            raise FileNotFoundError(f"HF image cache path does not exist: {source}")
        if target.is_file() and target.stat().st_size == source.stat().st_size and sha256_file(target) == sha256_file(source):
            return
        shutil.copy2(source, target)
        return
    raise RuntimeError("HF image record contains neither bytes nor a cache path")


def materialize_textcaps(data_root: Path, cache_dir: Optional[Path], requested_revision: Optional[str]) -> Tuple[str, Dict[str, int]]:
    try:
        from datasets import Image, load_dataset
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError("datasets and huggingface_hub are required for TextCaps. Install with: pip install -U datasets huggingface_hub") from exc

    info = HfApi().dataset_info(TEXTCAPS_REPO_ID, revision=requested_revision)
    revision = str(info.sha)
    log(f"TextCaps revision: {revision}")
    counts: Dict[str, int] = {}
    textcaps_root = data_root / "textcaps"

    for split in ("train", "validation"):
        expected = expected_textcaps_rows(textcaps_root / "manifests" / f"{split}.jsonl", split)
        dataset = load_dataset(
            TEXTCAPS_REPO_ID,
            split=split,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        dataset = dataset.cast_column("image", Image(decode=False))
        found: Set[str] = set()
        for row in dataset:
            image_id = str(row.get("image_id") or "").strip()
            rel = expected.get(image_id)
            if rel is None:
                continue
            if image_id in found:
                raise RuntimeError(f"Duplicate TextCaps upstream image_id: {image_id}")
            write_hf_image_value(row["image"], textcaps_root / PurePosixPath(rel))
            found.add(image_id)
        missing = sorted(set(expected) - found)
        if missing:
            raise RuntimeError(f"TextCaps {split}: {len(missing)} required image(s) absent upstream; examples={missing[:10]}")
        counts[f"textcaps_{split}"] = len(found)
        log(f"materialized TextCaps {split}: {len(found)} images")
    return revision, counts


def spright_expected_keys(data_root: Path) -> Set[str]:
    labels_root = data_root / "coco_spright/labels"
    expected: Set[str] = set()
    for name in ("short-coco-spright-train-0_9.json", "short-coco-spright-val-10_11.json"):
        payload = load_json(labels_root / name)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected object label JSON: {labels_root / name}")
        for raw in payload.keys():
            key = str(raw).replace("\\", "/")
            pure = PurePosixPath(key)
            if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 3 or pure.parts[0] != "data":
                raise RuntimeError(f"Unexpected SPRIGHT image key: {key!r}")
            expected.add(pure.as_posix())
    if len(expected) != EXPECTED["spright_label_images"]:
        raise RuntimeError(
            f"Expected {EXPECTED['spright_label_images']} unique active SPRIGHT image keys, found {len(expected)}"
        )
    return expected


def materialize_spright(data_root: Path, cache_dir: Optional[Path], requested_revision: Optional[str]) -> Tuple[str, int]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for SPRIGHT. Install with: pip install -U huggingface_hub") from exc

    info = HfApi().dataset_info(SPRIGHT_REPO_ID, revision=requested_revision)
    revision = str(info.sha)
    log(f"SPRIGHT revision: {revision}")
    snapshot = Path(
        snapshot_download(
            repo_id=SPRIGHT_REPO_ID,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["data/*.tar"],
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    tar_paths = sorted((snapshot / "data").glob("*.tar"), key=lambda p: (len(p.stem), p.stem))
    if not tar_paths:
        raise FileNotFoundError("SPRIGHT snapshot contains no data/*.tar shards")

    expected = spright_expected_keys(data_root)
    found: Set[str] = set()
    coco_root = data_root / "coco_spright"
    for tar_path in tar_paths:
        shard = tar_path.stem
        with tarfile.open(tar_path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                member_name = PurePosixPath(member.name).name
                if not member_name.casefold().endswith((".jpg", ".jpeg")):
                    continue
                # The project's labels key these records as data/<tar-stem>/<member-basename>.
                rel = PurePosixPath("data") / shard / member_name
                key = rel.as_posix()
                if key not in expected:
                    continue
                if key in found:
                    raise RuntimeError(f"Duplicate SPRIGHT image key across TARs: {key}")
                fileobj = archive.extractfile(member)
                if fileobj is None:
                    raise RuntimeError(f"Could not read TAR member: {tar_path}:{member.name}")
                target = coco_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = fileobj.read()
                if target.is_file() and target.read_bytes() == payload:
                    pass
                else:
                    target.write_bytes(payload)
                found.add(key)
        log(f"SPRIGHT shard {shard}: matched {sum(1 for k in found if k.startswith(f'data/{shard}/'))} active images")

    missing = sorted(expected - found)
    if missing:
        raise RuntimeError(f"SPRIGHT: {len(missing)} active image(s) missing from upstream TARs; examples={missing[:20]}")
    log(f"materialized SPRIGHT: {len(found)} active images")
    return revision, len(found)


def prefetch_hf_dataset_repo(
    repo_id: str,
    cache_dir: Optional[Path],
    requested_revision: Optional[str],
    *,
    label: str,
) -> str:
    try:
        from datasets import Image as HFImage, load_dataset
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(
            "datasets and huggingface_hub are required for benchmark prefetching. "
            "Install them with: pip install -U datasets huggingface_hub"
        ) from exc

    info = HfApi().dataset_info(repo_id, revision=requested_revision)
    revision = str(info.sha)
    dataset = load_dataset(
        repo_id,
        split="train",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    required = {"image", "object_label", "attack_word", "id"}
    missing = sorted(required.difference(dataset.column_names))
    if missing:
        raise RuntimeError(f"{label} benchmark is missing required columns: {missing}")
    dataset = dataset.cast_column("image", HFImage(decode=False))
    variants: Set[str] = set()
    for row in dataset:
        variant = str(row.get("type") or "").strip()
        raw_id = str(row.get("id") or "").strip()
        if not variant:
            if label == "SCAM":
                variant = next((x for x in ("NoSCAM", "SynthSCAM", "SCAM") if raw_id.startswith(x)), "")
            else:
                variant = next((x for x in ("NoRTA", "SynthRTA", "RTA") if raw_id.startswith(x)), "")
        if variant:
            variants.add(variant)
    expected = {"NoSCAM", "SCAM", "SynthSCAM"} if label == "SCAM" else {"NoRTA", "RTA", "SynthRTA"}
    if not expected.issubset(variants):
        raise RuntimeError(
            f"{label} benchmark variants changed: expected {sorted(expected)}, found {sorted(variants)}"
        )
    log(f"{label} benchmark cached: {repo_id}@{revision} ({len(dataset)} rows)")
    return revision


def verify_imagenet_root(root: Optional[Path], wnid_json: Path) -> Dict[str, Any]:
    if root is None:
        return {"provided": False}
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet root does not exist: {root}")
    wnids = load_json(wnid_json)
    if not isinstance(wnids, Mapping) or len(wnids) < 1000:
        raise RuntimeError(f"ImageNet WNID mapping looks incomplete: {wnid_json}")
    # Accept the trainer's supported layouts; only do a lightweight prerequisite check here.
    train_candidates = [root / "train", root / "ILSVRC2012_img_train"]
    train_root = next((p for p in train_candidates if p.is_dir()), None)
    if train_root is None:
        # Some users may pass the organized train directory itself.
        present_wnids = sum(1 for wnid in list(wnids.keys())[:100] if (root / str(wnid)).is_dir())
        if present_wnids < 50:
            raise RuntimeError(
                f"Could not recognize an ImageNet train layout under {root}. Expected root/train/<wnid>/... "
                "or an organized <wnid>/... directory."
            )
        train_root = root
    return {"provided": True, "root": str(root), "train_root_detected": str(train_root)}


def resolve_config_template(project_root: Path, config_template: Path) -> Path:
    path = config_template
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training config template does not exist: {path}")
    return path


def generate_local_config(
    project_root: Path,
    template_path: Path,
    output_path: Path,
    data_root: Path,
    imagenet_root: Optional[Path],
    scam_revision: Optional[str],
    rta_revision: Optional[str],
) -> Path:
    config = load_json(template_path)
    if not isinstance(config, MutableMapping) or not isinstance(config.get("paths"), MutableMapping):
        raise ValueError(f"Expected training config with object 'paths': {template_path}")
    paths: MutableMapping[str, Any] = config["paths"]
    # Never carry maintainer-local prerequisite/evaluation paths into a generated
    # user config. ImageNet is explicit; typo validation is repository-backed.
    paths["imagenet_root"] = (
        str(imagenet_root.expanduser().resolve()) if imagenet_root is not None else None
    )
    for key in ("mvt_csv", "mvt_image_root"):
        if key in paths:
            paths[key] = None
    for key, relative in CONFIG_PATHS.items():
        paths[key] = str((data_root / PurePosixPath(relative)).resolve())
    # Preserve null: the current final training config has no validation trusted-reading manifest.
    paths["coco_val_reading_json"] = None

    # Router artifacts are pipeline-internal outputs, never user dataset paths.  Do not
    # carry an old run root into a generated local config: the launcher/router derive
    # these from global.train_root and the configured stage output_subdir values.
    paths["router_utility_a4_checkpoint"] = None
    router_utility = config.get("router_utility")
    if not isinstance(router_utility, MutableMapping):
        raise ValueError(f"Expected training config with object 'router_utility': {template_path}")
    router_utility["start_checkpoint"] = None
    router_utility["output_dir"] = None
    router_utility["base_model_override"] = None

    shared_args = config.get("shared_args")
    if not isinstance(shared_args, MutableMapping) or not isinstance(shared_args.get("final_anytext"), MutableMapping):
        raise ValueError(f"Expected training config with shared_args.final_anytext: {template_path}")
    final_anytext: MutableMapping[str, Any] = shared_args["final_anytext"]
    final_anytext["benchmark_scam_repo"] = SCAM_REPO_ID
    final_anytext["benchmark_scam_revision"] = scam_revision
    final_anytext["benchmark_rta_repo"] = RTA_REPO_ID
    final_anytext["benchmark_rta_revision"] = rta_revision
    final_anytext["benchmark_include_mvt"] = False
    final_anytext["select_by_benchmark_score"] = False

    output = output_path
    if not output.is_absolute():
        output = project_root / output
    output = output.resolve()
    write_json(output, config)
    return output


def verify_prepared_tree(data_root: Path, *, require_upstream: bool) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    image_dirs = {
        "clevr_train": data_root / "clevr/clevr_images_train_square",
        "clevr_validation": data_root / "clevr/clevr_images_val_square",
        "handwriting": data_root / "handwriting_overlays/images",
        "salt_n_pepper": data_root / "salt_n_pepper",
    }
    for key, directory in image_dirs.items():
        if not directory.is_dir():
            raise RuntimeError(f"Prepared directory missing: {directory}")
        count = sum(1 for p in directory.iterdir() if p.is_file())
        if count != EXPECTED[key]:
            raise RuntimeError(f"{key}: expected {EXPECTED[key]} files, found {count}")
        counts[key] = count

    for _, relative in PACK_RAW_FILES.items():
        path = data_root / PurePosixPath(relative)
        if not path.is_file():
            raise RuntimeError(f"Prepared runtime file missing: {path}")

    if require_upstream:
        for split in ("train", "validation"):
            expected = expected_textcaps_rows(data_root / f"textcaps/manifests/{split}.jsonl", split)
            missing = [rel for rel in expected.values() if not (data_root / "textcaps" / PurePosixPath(rel)).is_file()]
            if missing:
                raise RuntimeError(f"TextCaps {split}: {len(missing)} images missing after preparation")
            counts[f"textcaps_{split}"] = len(expected)
        expected_spright = spright_expected_keys(data_root)
        missing = [key for key in expected_spright if not (data_root / "coco_spright" / PurePosixPath(key)).is_file()]
        if missing:
            raise RuntimeError(f"SPRIGHT: {len(missing)} images missing after preparation")
        counts["spright_label_images"] = len(expected_spright)
    return counts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=PACK_REPO_ID, help="HF training-data-pack repo id")
    parser.add_argument("--repo-revision", default=None, help="Optional pack revision; main is resolved to a commit SHA")
    parser.add_argument("--textcaps-revision", default=None, help="Optional TextCaps revision; main is resolved to a commit SHA")
    parser.add_argument("--spright-revision", default=None, help="Optional SPRIGHT revision; main is resolved to a commit SHA")
    parser.add_argument("--scam-revision", default=None, help="Optional SCAM revision; main is resolved to a commit SHA")
    parser.add_argument("--rta-revision", default=None, help="Optional RTA-100-Triplet revision; main is resolved to a commit SHA")
    parser.add_argument("--data-root", type=Path, default=Path("data/clip_cross_attn_mux"), help="Prepared runtime data root")
    parser.add_argument("--imagenet-root", type=Path, default=None, help="User-supplied ImageNet root (never modified)")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="CLIP Cross-Attn MUX repository root")
    parser.add_argument("--config-template", type=Path, default=Path("training_config.json"), help="Training config to copy/patch")
    parser.add_argument("--config-output", type=Path, default=Path("training_config.local.json"), help="Generated local config")
    parser.add_argument("--hf-cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--pack-only", action="store_true", help="Install only the project's HF pack; skip upstream TextCaps/SPRIGHT and benchmark prefetch")
    parser.add_argument("--no-config", action="store_true", help="Do not generate training_config.local.json")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate only --data-root before preparing")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    cache_dir = args.hf_cache_dir.expanduser().resolve() if args.hf_cache_dir is not None else None
    imagenet_root = args.imagenet_root.expanduser().resolve() if args.imagenet_root is not None else None

    log(f"project: {project_root}")
    log(f"data   : {data_root}")
    if args.reset and data_root.exists():
        log(f"resetting prepared data root: {data_root}")
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    pack_revision, snapshot_root = resolve_pack_revision(args.repo_id, args.repo_revision, cache_dir)
    pack_configs = verify_pack_parquet_hashes(snapshot_root)
    log(f"pack shard checksums: OK ({sum(sum(v.values()) for v in pack_configs.values())} rows)")

    raw_count = copy_runtime_files(snapshot_root, data_root)
    log(f"installed {raw_count} runtime/recipe metadata files")
    counts = extract_pack_parquet(snapshot_root, data_root)

    imagenet_status = verify_imagenet_root(imagenet_root, data_root / "imagenet_support/imagenet_wnid_to_class.json")
    if imagenet_status.get("provided"):
        log(f"ImageNet prerequisite: OK ({imagenet_status['root']})")
    else:
        log("ImageNet prerequisite: NOT PROVIDED (downloadable assets can still be prepared)")

    textcaps_revision: Optional[str] = None
    spright_revision: Optional[str] = None
    scam_revision: Optional[str] = None
    rta_revision: Optional[str] = None
    if not args.pack_only:
        textcaps_revision, textcaps_counts = materialize_textcaps(data_root, cache_dir, args.textcaps_revision)
        counts.update(textcaps_counts)
        spright_revision, spright_count = materialize_spright(data_root, cache_dir, args.spright_revision)
        counts["spright_label_images"] = spright_count
        scam_revision = prefetch_hf_dataset_repo(
            SCAM_REPO_ID, cache_dir, args.scam_revision, label="SCAM"
        )
        rta_revision = prefetch_hf_dataset_repo(
            RTA_REPO_ID, cache_dir, args.rta_revision, label="RTA"
        )

    generated_config: Optional[Path] = None
    if not args.no_config:
        template = resolve_config_template(project_root, args.config_template)
        generated_config = generate_local_config(
            project_root=project_root,
            template_path=template,
            output_path=args.config_output,
            data_root=data_root,
            imagenet_root=imagenet_root,
            scam_revision=scam_revision,
            rta_revision=rta_revision,
        )
        log(f"generated config: {generated_config}")

    counts.update(verify_prepared_tree(data_root, require_upstream=not args.pack_only))

    # The derivative roots contain exact portable recipes now, but not rendered ImageNet pixels yet.
    derivatives_ready = False
    summary = PrepareSummary(
        schema_version=2,
        pack_repo_id=args.repo_id,
        pack_revision=pack_revision,
        textcaps_repo_id=TEXTCAPS_REPO_ID,
        textcaps_revision=textcaps_revision,
        spright_repo_id=SPRIGHT_REPO_ID,
        spright_revision=spright_revision,
        scam_repo_id=SCAM_REPO_ID,
        scam_revision=scam_revision,
        rta_repo_id=RTA_REPO_ID,
        rta_revision=rta_revision,
        counts=dict(sorted(counts.items())),
        imagenet_root=str(imagenet_root) if imagenet_root is not None else None,
        imagenet_derivatives_ready=derivatives_ready,
        generated_config=str(generated_config) if generated_config is not None else None,
    )
    write_json(data_root / "PREPARE_STATE.json", asdict(summary))

    log("prepare complete; this command will now exit")
    if imagenet_root is not None:
        log("remaining required step: materialize the ImageNet-derived digital-text and handwriting datasets")
        log(
            "next command: python build_imagenet_derivatives.py "
            f"--project-root {project_root} --data-root {data_root} --imagenet-root {imagenet_root}"
        )
    else:
        log("remaining required step: supply ImageNet, then run build_imagenet_derivatives.py")
    log("optional evaluation setup: python prepare_objectnet_mvt.py --project-root <repo> --data-root <prepared-data-root>")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[xattn-data] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[xattn-data] ERROR: {exc}", file=sys.stderr)
        raise
