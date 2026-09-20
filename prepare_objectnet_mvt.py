#!/usr/bin/env python3
"""Optionally install the official ObjectNet-MVT evaluation images and enable MVT monitoring.

This helper is deliberately non-fatal by default. It makes one attempt to fetch the
official MVT *stimuli* release, validates that the archive contains every image named
by utils_datasets/mvt/human_responses_dedup.csv, extracts only those images, and then
patches training_config.local.json to enable evaluation-only MVT monitoring.

ObjectNet/MVT is never a training source. If the official host rejects automated
access, the config is left unchanged and the script prints the official manual URL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

DOWNLOAD_PAGE = "https://objectnet.dev/mvt/download.html"
STIMULI_URL = "https://objectnet.dev/mvt/data_release/flash_data_release_2023.zip"
RESULTS_URL = "https://objectnet.dev/mvt/data_release/reproduce_results_2023.zip"
DEFAULT_CSV_REL = Path("utils_datasets/mvt/human_responses_dedup.csv")
DEFAULT_DATA_REL = Path("objectnet_mvt")
EXPECTED_UNIQUE_IMAGES = 4771


def log(message: str) -> None:
    print(f"[objectnet-mvt] {message}", flush=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_expected(csv_path: Path) -> Dict[str, str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"MVT label CSV not found: {csv_path}")
    expected: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"Unexpected MVT CSV schema in {csv_path}: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            name = Path(str(row.get("image") or "")).name
            label = str(row.get("label") or "").strip()
            if not name or not label:
                raise ValueError(f"Missing image/label at {csv_path}:{row_number}")
            previous = expected.get(name)
            if previous is not None and previous != label:
                raise ValueError(f"MVT image {name!r} has inconsistent labels: {previous!r} vs {label!r}")
            expected[name] = label
    if len(expected) != EXPECTED_UNIQUE_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_UNIQUE_IMAGES} unique MVT images in {csv_path}, found {len(expected)}"
        )
    return expected


def download_once(url: str, destination: Path, timeout: float) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; CLIP-Cross-Attn-MUX data preparer; +https://objectnet.dev/mvt/download.html)",
            "Accept": "application/zip,application/octet-stream;q=0.9,*/*;q=0.1",
        },
    )
    log(f"trying official stimuli archive once: {url}")
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response, part.open("wb") as out:
            status = getattr(response, "status", None)
            content_type = str(response.headers.get("Content-Type") or "")
            total_text = response.headers.get("Content-Length")
            total = int(total_text) if total_text and total_text.isdigit() else None
            if status not in (None, 200):
                raise RuntimeError(f"HTTP status {status}")
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
                        log(f"downloaded {copied / 2**20:.1f}/{total / 2**20:.1f} MiB")
                    else:
                        log(f"downloaded {copied / 2**20:.1f} MiB")
                    next_report += 64 * 1024 * 1024
            log(f"download response: status={status or 200}, content-type={content_type or 'unknown'}, bytes={copied}")
    except Exception:
        part.unlink(missing_ok=True)
        raise
    if not zipfile.is_zipfile(part):
        # Keep no HTML/challenge/error body around in the prepared data tree.
        part.unlink(missing_ok=True)
        raise RuntimeError("official response was not a valid ZIP archive")
    os.replace(part, destination)
    return destination


def archive_member_map(zf: zipfile.ZipFile, expected: Set[str]) -> Dict[str, zipfile.ZipInfo]:
    found: Dict[str, zipfile.ZipInfo] = {}
    duplicates: Dict[str, List[str]] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        base = PurePosixPath(info.filename.replace("\\", "/")).name
        if base not in expected:
            continue
        if base in found:
            duplicates.setdefault(base, [found[base].filename]).append(info.filename)
        else:
            found[base] = info
    if duplicates:
        examples = list(duplicates.items())[:5]
        raise RuntimeError(f"MVT ZIP has duplicate expected basenames; examples={examples}")
    return found


def nested_zip_candidates(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    """Return nested ZIP members, prioritizing the official cropped_images.zip payload."""
    candidates = [
        info for info in zf.infolist()
        if not info.is_dir() and PurePosixPath(info.filename.replace("\\", "/")).suffix.lower() == ".zip"
    ]

    def key(info: zipfile.ZipInfo) -> Tuple[int, int, str]:
        normalized = info.filename.replace("\\", "/")
        base = PurePosixPath(normalized).name.lower()
        return (0 if base == "cropped_images.zip" else 1, normalized.count("/"), normalized.lower())

    return sorted(candidates, key=key)


def materialize_nested_zip(
    outer: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    cache_dir: Path,
) -> Path:
    """Stream one nested ZIP member to disk without loading the ~200 MiB payload into RAM."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = PurePosixPath(member.filename.replace("\\", "/")).name
    target = cache_dir / base
    if target.is_file() and target.stat().st_size == int(member.file_size) and zipfile.is_zipfile(target):
        return target

    part = target.with_suffix(target.suffix + ".part")
    part.unlink(missing_ok=True)
    with outer.open(member, "r") as source, part.open("wb") as out:
        shutil.copyfileobj(source, out, length=1024 * 1024)
    actual_size = part.stat().st_size
    if actual_size != int(member.file_size):
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Nested ZIP size mismatch for {member.filename}: expected {member.file_size}, got {actual_size}"
        )
    if not zipfile.is_zipfile(part):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Nested archive member is not a valid ZIP: {member.filename}")
    os.replace(part, target)
    return target


def resolve_image_payload_archive(
    outer_zip_path: Path,
    expected_names: Set[str],
    cache_dir: Path,
) -> Tuple[Path, Optional[str]]:
    """Find the ZIP that actually contains the MVT images.

    The 2023 ObjectNet release is ZIP-inside-ZIP:
      flash_data_release_2023.zip / data_release_2023 / cropped_images.zip / ...images...

    Older or manually repacked copies may expose the images directly, so keep that
    path working too.
    """
    with zipfile.ZipFile(outer_zip_path, "r") as outer:
        direct = archive_member_map(outer, expected_names)
        if len(direct) == len(expected_names):
            return outer_zip_path, None

        nested = nested_zip_candidates(outer)
        if not nested:
            missing = sorted(expected_names - set(direct))
            raise RuntimeError(
                f"Official MVT release has no nested ZIP payload and is missing {len(missing)} expected image names; "
                f"examples={missing[:10]}"
            )

        scanned: List[str] = []
        best_count = len(direct)
        best_missing = sorted(expected_names - set(direct))
        for member in nested:
            nested_path = materialize_nested_zip(outer, member, cache_dir)
            scanned.append(member.filename)
            with zipfile.ZipFile(nested_path, "r") as inner:
                found = archive_member_map(inner, expected_names)
            if len(found) > best_count:
                best_count = len(found)
                best_missing = sorted(expected_names - set(found))
            if len(found) == len(expected_names):
                log(f"found MVT image payload in nested archive: {member.filename}")
                return nested_path, member.filename

    raise RuntimeError(
        f"Official MVT release did not contain all {len(expected_names)} expected images. "
        f"Scanned nested ZIPs={scanned}; best match={best_count}/{len(expected_names)}; "
        f"missing examples={best_missing[:10]}"
    )


def extract_expected(zip_path: Path, expected: Mapping[str, str], image_root: Path, reset: bool) -> Dict[str, Any]:
    if reset and image_root.exists():
        shutil.rmtree(image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    expected_names = set(expected)
    payload_zip, payload_member = resolve_image_payload_archive(
        zip_path, expected_names, image_root.parent / "_downloads"
    )
    with zipfile.ZipFile(payload_zip, "r") as zf:
        members = archive_member_map(zf, expected_names)
        missing = sorted(expected_names - set(members))
        if missing:
            raise RuntimeError(
                f"Resolved MVT image payload is missing {len(missing)} image names expected by the bundled CSV; "
                f"examples={missing[:10]}"
            )
        for index, name in enumerate(sorted(expected_names), start=1):
            target = image_root / name
            if target.is_file() and target.stat().st_size > 0:
                continue
            info = members[name]
            with zf.open(info, "r") as source, target.open("wb") as out:
                shutil.copyfileobj(source, out, length=1024 * 1024)
            if index % 1000 == 0:
                log(f"extracted {index}/{len(expected_names)} MVT images")
    present = {p.name for p in image_root.iterdir() if p.is_file()}
    missing_after = sorted(expected_names - present)
    if missing_after:
        raise RuntimeError(f"MVT extraction incomplete: {len(missing_after)} expected images absent")

    # Image-decode verification is intentionally dependency-light: Pillow is already
    # required by the training code, but keep import local so failure is explicit.
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to verify MVT images") from exc
    for name in sorted(expected_names):
        path = image_root / name
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise RuntimeError(f"Invalid/corrupt MVT image after extraction: {path}") from exc
    return {
        "image_count": len(expected_names),
        "zip_sha256": sha256_file(zip_path),
        "payload_zip_sha256": sha256_file(payload_zip),
        "payload_member": payload_member,
        "image_root": str(image_root.resolve()),
    }


def patch_config(config_path: Path, csv_path: Path, image_root: Path) -> None:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Local training config not found: {config_path}. Run prepare_training_data.py first."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, MutableMapping):
        raise ValueError(f"Expected JSON object in {config_path}")
    paths = config.get("paths")
    shared = config.get("shared_args")
    if not isinstance(paths, MutableMapping) or not isinstance(shared, MutableMapping):
        raise ValueError(f"Unexpected training config schema: {config_path}")
    final = shared.get("final_anytext")
    if not isinstance(final, MutableMapping):
        raise ValueError(f"Missing shared_args.final_anytext in {config_path}")
    paths["mvt_csv"] = str(csv_path.resolve())
    paths["mvt_image_root"] = str(image_root.resolve())
    final["benchmark_include_mvt"] = True
    # Preserve the evaluation firewall: ObjectNet must never select checkpoints.
    final["select_by_benchmark_score"] = False
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_source_record(root: Path, payload: Mapping[str, Any]) -> None:
    path = root / "SOURCE.json"
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def manual_message() -> None:
    log("automatic ObjectNet-MVT installation was not completed; training_config.local.json was left unchanged")
    log(f"manual download page: {DOWNLOAD_PAGE}")
    log("download the 'Human classification judgments ... Cropped stimuli images' release (flash_data_release_2023.zip), then rerun with --zip-path <file>")
    log(f"note: {RESULTS_URL} is the model-results/reproduction bundle, not the cropped MVT stimulus images")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("data/clip_cross_attn_mux"))
    parser.add_argument("--config", type=Path, default=Path("training_config.local.json"))
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_REL)
    parser.add_argument("--zip-path", type=Path, default=None, help="Use an already-downloaded official flash_data_release_2023.zip")
    parser.add_argument("--url", default=STIMULI_URL, help="Official stimuli ZIP URL")
    parser.add_argument("--timeout", type=float, default=90.0, help="Single HTTP request timeout in seconds")
    parser.add_argument("--reset", action="store_true", help="Re-extract MVT image payload; never deletes other prepared data")
    parser.add_argument("--strict", action="store_true", help="Return nonzero if automatic download/extraction fails")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser()
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    else:
        data_root = data_root.resolve()
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    csv_path = args.csv.expanduser()
    if not csv_path.is_absolute():
        csv_path = (project_root / csv_path).resolve()

    expected = load_expected(csv_path)
    log(f"bundled labels: {len(expected)} unique MVT images")
    root = data_root / DEFAULT_DATA_REL
    image_root = root / "images"
    archive = root / "_downloads" / "flash_data_release_2023.zip"

    try:
        if args.zip_path is not None:
            supplied = args.zip_path.expanduser().resolve()
            if not supplied.is_file() or not zipfile.is_zipfile(supplied):
                raise RuntimeError(f"--zip-path is not a valid ZIP: {supplied}")
            archive = supplied
            log(f"using supplied official archive: {archive}")
        elif not archive.is_file():
            download_once(str(args.url), archive, float(args.timeout))
        elif not zipfile.is_zipfile(archive):
            archive.unlink(missing_ok=True)
            download_once(str(args.url), archive, float(args.timeout))
        else:
            log(f"reusing cached official archive: {archive}")

        result = extract_expected(archive, expected, image_root, reset=bool(args.reset))
        local_csv = root / "human_responses_dedup.csv"
        local_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, local_csv)
        patch_config(config_path, local_csv, image_root)
        write_source_record(
            root,
            {
                "source_page": DOWNLOAD_PAGE,
                "stimuli_url": STIMULI_URL,
                "results_url_not_used": RESULTS_URL,
                "archive_sha256": result["zip_sha256"],
                "payload_archive_member": result.get("payload_member"),
                "payload_archive_sha256": result.get("payload_zip_sha256"),
                "image_count": result["image_count"],
                "label_csv": "human_responses_dedup.csv",
                "image_root": "images",
                "purpose": "evaluation-only MVT monitor; never a training source",
            },
        )
        log(f"validated {result['image_count']} official MVT stimulus images")
        log(f"enabled MVT in: {config_path}")
        log("ObjectNet-MVT remains evaluation-only; benchmark checkpoint selection is disabled")
        log("complete; this command will now exit")
        return 0
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        log(f"automatic attempt failed: {type(exc).__name__}: {exc}")
        manual_message()
        return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
