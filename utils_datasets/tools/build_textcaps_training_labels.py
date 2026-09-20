#!/usr/bin/env python3
"""Build high-trust TextCaps reading labels from a PaddleOCR index.

Accepted input formats
----------------------
1. The completed JSON produced by ``build_textcaps_paddleocr_index.py``.
2. Its resumable JSONL progress sidecar, including a partial scan.

The default policy is intentionally conservative for CLIP reading supervision:
- upright OCR index with confidence >= 0.999;
- at most three accepted OCR detections in the whole image;
- at least two alphabetic characters per retained target;
- reject pure numbers, isolated characters, URLs and email-like strings;
- reject words too small to remain meaningful after CLIP resizing;
- deduplicate repeated normalized words while preserving all of their boxes;
- every retained word is a positive target for the image;
- ``<text><null>`` is a negative for every retained image.

No OCR-negative/null labels are created. Images without trusted OCR are simply
absent from the supervised manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

SCRIPT_VERSION = "2026-08-05-v1"
URL_RE = re.compile(r"(?:https?://|www\.|\b[a-z0-9.-]+\.(?:com|org|net|io|co|gov|edu)\b)", re.I)
EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", re.I)
WHITESPACE_RE = re.compile(r"\s+")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    """Unicode-aware normalization while preserving meaningful word spacing."""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    characters: List[str] = []
    for character in value:
        if character.isalnum():
            characters.append(character)
        elif character in {"'", "’", "-"}:
            characters.append("'")
        else:
            characters.append(" ")
    return WHITESPACE_RE.sub(" ", "".join(characters)).strip(" '-")


def alphabetic_count(value: str) -> int:
    return sum(character.isalpha() for character in value)


def alphanumeric_count(value: str) -> int:
    return sum(character.isalnum() for character in value)


def load_rows(path: Path) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Return unique rows, source format, and source metadata.

    Progress JSONL is treated like the OCR scanner: the last row for an image
    wins, so an interrupted/resumed file cannot duplicate training labels.
    """
    if path.suffix.casefold() == ".jsonl" or ".jsonl" in path.name.casefold():
        by_image: Dict[str, Dict[str, Any]] = {}
        malformed = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, Mapping) or not row.get("image"):
                    malformed += 1
                    continue
                by_image[str(row["image"])] = dict(row)
        return list(by_image.values()), "progress_jsonl", {"malformed_rows": malformed}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("entries"), list):
        rows = [dict(row) for row in payload["entries"] if isinstance(row, Mapping)]
        metadata = {
            "source_schema_version": payload.get("schema_version"),
            "source_generator": payload.get("generator"),
            "source_settings": payload.get("settings"),
            "source_summary": payload.get("summary"),
        }
        return rows, "completed_json", metadata
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)], "json_list", {}
    raise ValueError(
        f"Unsupported input schema in {path}. Expected completed OCR JSON with 'entries' or progress JSONL."
    )


def bbox_metrics(detection: Mapping[str, Any]) -> Tuple[float, float]:
    box = detection.get("bbox_xywh_norm")
    if not isinstance(box, Sequence) or len(box) != 4:
        return 0.0, 0.0
    _, _, width, height = [max(0.0, float(value)) for value in box]
    return height, width * height


def detection_rejection_reason(
    detection: Mapping[str, Any],
    *,
    min_confidence: float,
    min_alpha_chars: int,
    max_text_chars: int,
    min_box_height: float,
    min_box_area: float,
) -> str | None:
    text = str(detection.get("text", "")).strip()
    normalized = normalize_text(detection.get("norm_text") or text)
    confidence = float(detection.get("best_confidence", detection.get("confidence", 0.0)))
    if confidence < min_confidence:
        return "below_confidence"
    if not normalized:
        return "empty_after_normalization"
    if URL_RE.search(text) or EMAIL_RE.search(text):
        return "url_or_email"
    if alphabetic_count(normalized) < min_alpha_chars:
        if alphanumeric_count(normalized) > 0 and alphabetic_count(normalized) == 0:
            return "numeric_or_nonlexical"
        return "too_few_letters"
    if alphanumeric_count(normalized) > max_text_chars:
        return "too_long"
    height, area = bbox_metrics(detection)
    if height < min_box_height:
        return "box_too_short"
    if area < min_box_area:
        return "box_too_small"
    return None


def compact_box(detection: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "text": str(detection.get("text", "")),
        "confidence": float(detection.get("best_confidence", detection.get("confidence", 0.0))),
        "polygon_original": detection.get("polygon_original"),
        "bbox_xyxy_original": detection.get("bbox_xyxy_original"),
        "bbox_xywh_norm": detection.get("bbox_xywh_norm"),
        "rotations_observed": sorted(
            {
                int(observation.get("rotation_applied_degrees", 0))
                for observation in detection.get("observations", [])
                if isinstance(observation, Mapping)
            }
        ),
    }


def group_targets(detections: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate text identity while retaining every spatial occurrence."""
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for detection in detections:
        normalized = normalize_text(detection.get("norm_text") or detection.get("text", ""))
        grouped[normalized].append(detection)

    targets: List[Dict[str, Any]] = []
    for normalized, occurrences in grouped.items():
        ordered = sorted(
            occurrences,
            key=lambda item: float(item.get("best_confidence", item.get("confidence", 0.0))),
            reverse=True,
        )
        best = ordered[0]
        targets.append(
            {
                "text": str(best.get("text", normalized)),
                "norm_text": normalized,
                "best_confidence": float(best.get("best_confidence", best.get("confidence", 0.0))),
                "occurrence_count": len(ordered),
                "occurrences": [compact_box(item) for item in ordered],
            }
        )
    targets.sort(key=lambda item: (-item["best_confidence"], item["norm_text"]))
    return targets


def make_training_entry(row: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    category = "ultra_clean_single" if len(targets) == 1 else "clean_sparse_multi"
    target_texts = [str(target["text"]) for target in targets]
    return {
        "image": str(row["image"]),
        "absolute_path": row.get("absolute_path"),
        "split": row.get("split"),
        "width": row.get("width"),
        "height": row.get("height"),
        "category": category,
        "raw_accepted_detection_count": int(row.get("accepted_detection_count", len(row.get("detections", [])))),
        "unique_target_count": len(targets),
        "targets": list(targets),
        "target_texts": target_texts,
        "positive_text_queries": [f"<text> {text}" for text in target_texts],
        "source_targets": {
            "present": 1.0,
            "readable": 1.0,
            "text_null": 0.0,
        },
        "multi_positive": len(targets) > 1,
    }


def manifest_payload(
    *,
    source: Path,
    source_format: str,
    source_metadata: Mapping[str, Any],
    settings: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    category: str,
) -> Dict[str, Any]:
    by_split = Counter(str(entry.get("split", "unknown")) for entry in entries)
    return {
        "schema_version": 1,
        "generator": Path(__file__).name,
        "generator_version": SCRIPT_VERSION,
        "source_ocr_path": source.resolve().as_posix(),
        "source_ocr_sha256": sha256_file(source),
        "source_format": source_format,
        "source_metadata": dict(source_metadata),
        "settings": dict(settings),
        "category": category,
        "semantics": {
            "all_targets_are_positive_for_the_same_image": True,
            "present": 1.0,
            "readable": 1.0,
            "text_null": 0.0,
            "ocr_absence_creates_null_target": False,
        },
        "summary": {
            "images": len(entries),
            "targets": sum(int(entry["unique_target_count"]) for entry in entries),
            "by_split": dict(sorted(by_split.items())),
        },
        "entries": list(entries),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed OCR JSON or progress JSONL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("utils_datasets/textcaps/derived_reading_conf999"),
    )
    parser.add_argument("--min-confidence", type=float, default=0.999)
    parser.add_argument("--max-detections", type=int, default=3, help="Reject dense images above this raw accepted count.")
    parser.add_argument("--min-alpha-chars", type=int, default=2)
    parser.add_argument("--max-text-chars", type=int, default=40)
    parser.add_argument(
        "--min-box-height",
        type=float,
        default=0.025,
        help="Minimum normalized box height; 0.025 is about 5.6 pixels at CLIP 224px.",
    )
    parser.add_argument("--min-box-area", type=float, default=0.0005)
    parser.add_argument(
        "--write-rejected",
        action="store_true",
        help="Write an audit JSON containing rejected image paths and reasons.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.max_detections < 1:
        raise ValueError("--max-detections must be >= 1")

    rows, source_format, source_metadata = load_rows(source)
    settings = {
        "min_confidence": float(args.min_confidence),
        "max_raw_accepted_detections": int(args.max_detections),
        "min_alpha_chars": int(args.min_alpha_chars),
        "max_text_chars": int(args.max_text_chars),
        "min_box_height_norm": float(args.min_box_height),
        "min_box_area_norm": float(args.min_box_area),
        "rotations_expected": [0],
    }

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    seen_images: set[str] = set()

    for row in sorted(rows, key=lambda item: str(item.get("image", ""))):
        image = str(row.get("image", ""))
        if not image or image in seen_images:
            continue
        seen_images.add(image)
        if row.get("status") == "error":
            reasons = ["ocr_error"]
        else:
            detections = [item for item in row.get("detections", []) if isinstance(item, Mapping)]
            raw_count = int(row.get("accepted_detection_count", len(detections)))
            reasons = []
            if raw_count == 0 or not detections:
                reasons.append("no_accepted_detections")
            if raw_count > args.max_detections:
                reasons.append("dense_too_many_detections")

        if reasons:
            reason_counts.update(reasons)
            rejected.append({"image": image, "split": row.get("split"), "reasons": reasons})
            continue

        valid_detections: List[Mapping[str, Any]] = []
        detection_reasons: Counter[str] = Counter()
        for detection in detections:
            reason = detection_rejection_reason(
                detection,
                min_confidence=args.min_confidence,
                min_alpha_chars=args.min_alpha_chars,
                max_text_chars=args.max_text_chars,
                min_box_height=args.min_box_height,
                min_box_area=args.min_box_area,
            )
            if reason is None:
                valid_detections.append(detection)
            else:
                detection_reasons[reason] += 1

        targets = group_targets(valid_detections)
        if not targets:
            reasons = ["no_valid_lexical_targets"]
            if detection_reasons:
                reasons.extend(f"all:{name}" for name in sorted(detection_reasons))
            reason_counts.update(reasons)
            rejected.append({"image": image, "split": row.get("split"), "reasons": reasons})
            continue
        if len(targets) > args.max_detections:
            reasons = ["too_many_unique_targets"]
            reason_counts.update(reasons)
            rejected.append({"image": image, "split": row.get("split"), "reasons": reasons})
            continue

        accepted.append(make_training_entry(row, targets))

    single = [entry for entry in accepted if entry["category"] == "ultra_clean_single"]
    multi = [entry for entry in accepted if entry["category"] == "clean_sparse_multi"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "all": output_dir / "textcaps_reading_supervised_conf999.json",
        "single": output_dir / "textcaps_reading_single_conf999.json",
        "multi": output_dir / "textcaps_reading_sparse_multi_conf999.json",
        "summary": output_dir / "textcaps_reading_filter_summary.json",
    }
    atomic_write_json(
        outputs["all"],
        manifest_payload(
            source=source,
            source_format=source_format,
            source_metadata=source_metadata,
            settings=settings,
            entries=accepted,
            category="all_high_trust_reading",
        ),
    )
    atomic_write_json(
        outputs["single"],
        manifest_payload(
            source=source,
            source_format=source_format,
            source_metadata=source_metadata,
            settings=settings,
            entries=single,
            category="ultra_clean_single",
        ),
    )
    atomic_write_json(
        outputs["multi"],
        manifest_payload(
            source=source,
            source_format=source_format,
            source_metadata=source_metadata,
            settings=settings,
            entries=multi,
            category="clean_sparse_multi",
        ),
    )

    summary = {
        "schema_version": 1,
        "generator": Path(__file__).name,
        "generator_version": SCRIPT_VERSION,
        "source_ocr_path": source.as_posix(),
        "source_ocr_sha256": sha256_file(source),
        "source_format": source_format,
        "settings": settings,
        "counts": {
            "source_rows": len(rows),
            "accepted_images": len(accepted),
            "single_images": len(single),
            "sparse_multi_images": len(multi),
            "accepted_targets": sum(entry["unique_target_count"] for entry in accepted),
            "rejected_images": len(rejected),
        },
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "outputs": {name: path.as_posix() for name, path in outputs.items() if name != "summary"},
        "source_is_partial": source_format == "progress_jsonl",
    }
    if args.write_rejected:
        rejected_path = output_dir / "textcaps_reading_rejected_audit.json"
        atomic_write_json(
            rejected_path,
            {
                "schema_version": 1,
                "source_ocr_path": source.as_posix(),
                "summary": {"images": len(rejected), "reasons": dict(sorted(reason_counts.items()))},
                "entries": rejected,
            },
        )
        summary["outputs"]["rejected_audit"] = rejected_path.as_posix()
    atomic_write_json(outputs["summary"], summary)

    print(f"[script] build_textcaps_training_labels {SCRIPT_VERSION}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
