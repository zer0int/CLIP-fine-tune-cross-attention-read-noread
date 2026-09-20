#!/usr/bin/env python3
"""Split COCO-SPRIGHT into high-confidence reading supervision and a remainder.

Inputs:
  * the original COCO-SPRIGHT label JSON;
  * the GPT-OSS rewritten label JSON;
  * the OCR/word-match dump used to request those rewrites.

Any OCR detection with confidence >= ``--min-confidence`` and an allowed
rotation becomes a readable literal-text target.  Caption links are retained
separately, with ``exact`` and ``compound_exact`` treated as strict links by
default.  The script writes one rich reading manifest and two loader-compatible
JSON dictionaries for the remaining unsupervised labels.
"""
from __future__ import annotations

SCRIPT_VERSION = "2026-08-05-prefix-aware-v2"

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from PIL import Image, ImageOps

DEFAULT_ALLOWED_ROTATIONS = (0, 90, 270)
DEFAULT_STRICT_MATCH_TYPES = ("exact", "compound_exact")


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def normalize_whitespace(value: str) -> str:
    return " ".join(str(value).split()).strip()


def classify_label_alignment(
    dump_labels: Sequence[str],
    current_labels: Sequence[str],
) -> str:
    """Classify whether the dump captions and current short captions are aligned.

    The OCR dump was generated from the full COCO-SPRIGHT captions, while the
    training files use their shortened counterparts.  A current caption is
    therefore considered aligned when its normalized token sequence is exactly
    equal to, or an exact prefix of, the corresponding dump caption.
    """
    if len(dump_labels) != len(current_labels):
        return "structural_mismatch"
    if list(dump_labels) == list(current_labels):
        return "exact"

    whitespace_equal = True
    saw_short_prefix = False
    for dump_caption, current_caption in zip(dump_labels, current_labels):
        if normalize_whitespace(dump_caption) != normalize_whitespace(current_caption):
            whitespace_equal = False
        dump_tokens = normalize_text(dump_caption).split()
        current_tokens = normalize_text(current_caption).split()
        if dump_tokens == current_tokens:
            continue
        if current_tokens and dump_tokens[: len(current_tokens)] == current_tokens:
            saw_short_prefix = True
            continue
        return "structural_mismatch"

    if whitespace_equal:
        return "whitespace_equivalent"
    if saw_short_prefix:
        return "short_caption_prefix_of_dump"
    return "token_equivalent"


def locate_span_in_current_caption(
    caption: str,
    span_text: str,
    span_norm: str,
) -> Optional[Tuple[int, int, str]]:
    """Locate a dump span in the current shortened caption.

    Raw case-insensitive matching preserves useful offsets.  Token-normalized
    matching is the fallback for harmless punctuation/whitespace differences.
    """
    if not caption or not span_norm:
        return None
    raw_start = caption.casefold().find(span_text.casefold()) if span_text else -1
    if raw_start >= 0:
        return raw_start, raw_start + len(span_text), "raw_casefold"

    caption_tokens = normalize_text(caption).split()
    span_tokens = normalize_text(span_norm).split()
    if not span_tokens or len(span_tokens) > len(caption_tokens):
        return None
    for index in range(len(caption_tokens) - len(span_tokens) + 1):
        if caption_tokens[index : index + len(span_tokens)] == span_tokens:
            return -1, -1, "normalized_tokens"
    return None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def labels_from_value(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, Mapping):
        for key in ("labels", "captions", "caption"):
            if key in value:
                return labels_from_value(value[key])
    raise TypeError(f"Unsupported label value: {type(value).__name__}")


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in a]
    bx0, by0, bx1, by1 = [float(value) for value in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def bbox_from_polygon(points: Sequence[Sequence[float]]) -> List[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_bbox(bbox: Sequence[float], width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return [
        x0 / width,
        y0 / height,
        max(0.0, x1 - x0) / width,
        max(0.0, y1 - y0) / height,
    ]


def resolve_image_path(coco_root: Path, image_key: str, recorded_path: str) -> Path:
    primary = coco_root / Path(str(image_key).replace("\\", "/"))
    if primary.is_file():
        return primary
    recorded = Path(recorded_path)
    if recorded.is_file():
        return recorded
    return primary


def image_size(path: Path, allow_missing: bool) -> Tuple[Optional[int], Optional[int]]:
    if not path.is_file():
        if allow_missing:
            return None, None
        raise FileNotFoundError(f"COCO image does not exist: {path}")
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        return int(image.width), int(image.height)


def dedupe_ocr(
    detections: Sequence[Mapping[str, Any]],
    min_confidence: float,
    allowed_rotations: set[int],
    min_alnum_chars: int,
    dedupe_iou: float,
) -> List[Dict[str, Any]]:
    accepted: List[Dict[str, Any]] = []
    for item in sorted(detections, key=lambda row: float(row.get("conf", 0.0)), reverse=True):
        confidence = float(item.get("conf", 0.0))
        rotation = int(item.get("rotation", 0))
        text = str(item.get("text") or "").strip()
        norm_text = str(item.get("norm_text") or normalize_text(text))
        if confidence < min_confidence or rotation not in allowed_rotations:
            continue
        if sum(character.isalnum() for character in norm_text) < min_alnum_chars:
            continue
        polygon = [[float(x), float(y)] for x, y in item.get("box") or []]
        bbox = [float(value) for value in item.get("bbox") or bbox_from_polygon(polygon)]
        observation = {
            "text": text,
            "norm_text": norm_text,
            "confidence": confidence,
            "rotation": rotation,
            "polygon": polygon,
            "bbox_xyxy": bbox,
        }
        target: Optional[Dict[str, Any]] = None
        for group in accepted:
            if group["norm_text"] != norm_text:
                continue
            if bbox_iou(group["bbox_xyxy"], bbox) >= dedupe_iou:
                target = group
                break
        if target is None:
            target = {
                **observation,
                "observations": [],
                "caption_links": [],
                "strict_caption_links": [],
            }
            accepted.append(target)
        target["observations"].append(observation)
        if confidence > float(target["confidence"]):
            target.update(observation)
    return accepted


def link_caption_spans(
    targets: List[Dict[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    current_labels: Sequence[str],
    min_confidence: float,
    allowed_rotations: set[int],
    strict_types: set[str],
    dedupe_iou: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_links: List[Dict[str, Any]] = []
    strict_links: List[Dict[str, Any]] = []
    discarded_links: List[Dict[str, Any]] = []
    for span in spans:
        confidence = float(span.get("ocr_conf", 0.0))
        rotation = int(span.get("ocr_rotation", 0))
        if confidence < min_confidence or rotation not in allowed_rotations:
            continue
        ocr_text = str(span.get("ocr_text") or "").strip()
        ocr_norm = str(span.get("ocr_norm") or normalize_text(ocr_text))
        polygon = [[float(x), float(y)] for x, y in span.get("ocr_box") or []]
        bbox = bbox_from_polygon(polygon) if polygon else None
        caption_index = int(span["caption_index"])
        span_text = str(span.get("span_text") or "")
        span_norm = str(span.get("span_norm") or normalize_text(span_text))
        link = {
            "caption_index": caption_index,
            "span_text": span_text,
            "span_norm": span_norm,
            "dump_start": int(span.get("start", -1)),
            "dump_end": int(span.get("end", -1)),
            "match_type": str(span.get("match_type") or ""),
            "ocr_text": ocr_text,
            "ocr_norm": ocr_norm,
            "ocr_confidence": confidence,
            "ocr_rotation": rotation,
            "ocr_polygon": polygon,
        }

        if caption_index < 0 or caption_index >= len(current_labels):
            link["discard_reason"] = "caption_index_out_of_range"
            discarded_links.append(link)
            continue
        location = locate_span_in_current_caption(
            current_labels[caption_index], span_text, span_norm
        )
        if location is None:
            link["discard_reason"] = "span_absent_from_current_short_caption"
            discarded_links.append(link)
            continue
        current_start, current_end, current_match_method = location
        link["current_start"] = current_start
        link["current_end"] = current_end
        link["current_match_method"] = current_match_method

        all_links.append(link)
        is_strict = link["match_type"] in strict_types
        if is_strict:
            strict_links.append(link)
        best_target: Optional[Dict[str, Any]] = None
        best_iou = -1.0
        for target in targets:
            if target["norm_text"] != ocr_norm:
                continue
            overlap = bbox_iou(target["bbox_xyxy"], bbox) if bbox is not None else 0.0
            if overlap > best_iou:
                best_target, best_iou = target, overlap
        if best_target is not None and (bbox is None or best_iou >= dedupe_iou):
            best_target["caption_links"].append(link)
            if is_strict:
                best_target["strict_caption_links"].append(link)
    return all_links, strict_links, discarded_links


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ocr-dump",
        type=Path,
        default=Path("utils_datasets/coco_spright/coco-sprite-train-0_9_dump_gpt_oss.json"),
    )
    parser.add_argument(
        "--original-labels",
        type=Path,
        default=Path("utils_datasets/coco_spright/short-coco-spright-train-0_9.json"),
    )
    parser.add_argument(
        "--rewritten-labels",
        type=Path,
        default=Path("utils_datasets/coco_spright/short-coco-spright-train-0_9_gpt_oss.json"),
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("SPRIGHT/COCO/data-square"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("utils_datasets/coco_spright/derived_reading_conf999"),
    )
    parser.add_argument("--min-confidence", type=float, default=0.999)
    parser.add_argument("--allowed-rotations", nargs="+", type=int, default=list(DEFAULT_ALLOWED_ROTATIONS))
    parser.add_argument("--strict-match-types", nargs="+", default=list(DEFAULT_STRICT_MATCH_TYPES))
    parser.add_argument("--min-alnum-chars", type=int, default=3)
    parser.add_argument("--dedupe-iou", type=float, default=0.50)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument(
        "--allow-label-mismatches",
        action="store_true",
        help="Write outputs even if dump/original/rewrite caption alignment differs.",
    )
    return parser.parse_args()


def main() -> int:
    print(f"[script] build_coco_spright_reading_manifests {SCRIPT_VERSION}")
    args = parse_args()
    original_raw = read_json(args.original_labels)
    rewritten_raw = read_json(args.rewritten_labels)
    dump = read_json(args.ocr_dump)
    if not isinstance(original_raw, Mapping) or not isinstance(rewritten_raw, Mapping):
        raise TypeError("Original and rewritten COCO-SPRIGHT files must be JSON objects keyed by image path")
    entries = dump.get("entries") if isinstance(dump, Mapping) else None
    if not isinstance(entries, list):
        raise TypeError("OCR dump must contain a top-level 'entries' list")

    original: Dict[str, List[str]] = {str(key): labels_from_value(value) for key, value in original_raw.items()}
    rewritten: Dict[str, List[str]] = {str(key): labels_from_value(value) for key, value in rewritten_raw.items()}
    dump_by_key: Dict[str, Mapping[str, Any]] = {str(entry["image_key"]): entry for entry in entries}

    allowed_rotations = {int(value) for value in args.allowed_rotations}
    if 180 in allowed_rotations:
        raise ValueError("Rotation 180 is intentionally excluded from readable supervision")
    strict_types = {str(value) for value in args.strict_match_types}

    reading_rows: List[Dict[str, Any]] = []
    unsupervised_original: Dict[str, List[str]] = {}
    unsupervised_rewritten: Dict[str, List[str]] = {}
    fatal_mismatch_counts: Counter[str] = Counter()
    alignment_counts: Counter[str] = Counter()
    discarded_link_counts: Counter[str] = Counter()
    match_type_counts: Counter[str] = Counter()
    selected_text_counts: Counter[str] = Counter()

    for image_key in sorted(original):
        original_labels = original[image_key]
        rewritten_labels = rewritten.get(image_key, original_labels)
        if len(rewritten_labels) != len(original_labels):
            fatal_mismatch_counts["rewritten_label_count_mismatch"] += 1
        entry = dump_by_key.get(image_key)
        if entry is None:
            unsupervised_original[image_key] = original_labels
            unsupervised_rewritten[image_key] = rewritten_labels
            continue

        dump_labels = labels_from_value(entry.get("labels") or [])
        label_alignment = classify_label_alignment(dump_labels, original_labels)
        alignment_counts[label_alignment] += 1
        if label_alignment == "structural_mismatch":
            fatal_mismatch_counts["dump_original_structural_mismatch"] += 1

        targets = dedupe_ocr(
            entry.get("ocr") or [],
            args.min_confidence,
            allowed_rotations,
            args.min_alnum_chars,
            args.dedupe_iou,
        )
        if not targets:
            unsupervised_original[image_key] = original_labels
            unsupervised_rewritten[image_key] = rewritten_labels
            continue

        image_path = resolve_image_path(args.coco_root, image_key, str(entry.get("image_path") or ""))
        width, height = image_size(image_path, args.allow_missing_images)
        all_links, strict_links, discarded_links = link_caption_spans(
            targets,
            entry.get("flagged_spans") or [],
            original_labels,
            args.min_confidence,
            allowed_rotations,
            strict_types,
            args.dedupe_iou,
        )
        for link in discarded_links:
            discarded_link_counts[str(link["discard_reason"])] += 1
        for link in all_links:
            match_type_counts[link["match_type"]] += 1
        for target in targets:
            selected_text_counts[target["norm_text"]] += 1
            target["present_target"] = 1.0
            target["readable_target"] = 1.0
            target["text_null_positive"] = False
            if width is not None and height is not None:
                target["bbox_xywh_norm"] = normalize_bbox(target["bbox_xyxy"], width, height)
            else:
                target["bbox_xywh_norm"] = None

        label_pairs = []
        max_labels = max(len(original_labels), len(rewritten_labels))
        for index in range(max_labels):
            old = original_labels[index] if index < len(original_labels) else None
            new = rewritten_labels[index] if index < len(rewritten_labels) else None
            label_pairs.append(
                {
                    "caption_index": index,
                    "original": old,
                    "rewritten": new,
                    "changed": old != new,
                    "strictly_ocr_linked": any(link["caption_index"] == index for link in strict_links),
                    "ocr_linked": any(link["caption_index"] == index for link in all_links),
                }
            )

        reading_rows.append(
            {
                "image_key": image_key,
                "image_path": image_path.as_posix(),
                "width": width,
                "height": height,
                "original_labels": original_labels,
                "rewritten_labels": rewritten_labels,
                "dump_labels": dump_labels,
                "label_alignment": label_alignment,
                "label_pairs": label_pairs,
                "reading_targets": targets,
                "caption_links": all_links,
                "strict_caption_links": strict_links,
                "discarded_dump_caption_links": discarded_links,
                "flagged_caption_indices": [int(value) for value in entry.get("flagged_caption_indices") or []],
                "has_strict_caption_link": bool(strict_links),
                "supervision": {
                    "present": 1.0,
                    "readable": 1.0,
                    "text_null_positive": False,
                },
            }
        )

    extra_dump_keys = sorted(set(dump_by_key) - set(original))
    extra_rewritten_keys = sorted(set(rewritten) - set(original))
    summary = {
        "original_images": len(original),
        "dump_entries": len(entries),
        "reading_supervised_images": len(reading_rows),
        "reading_targets": sum(len(row["reading_targets"]) for row in reading_rows),
        "images_with_any_caption_link": sum(bool(row["caption_links"]) for row in reading_rows),
        "images_with_strict_caption_link": sum(row["has_strict_caption_link"] for row in reading_rows),
        "unsupervised_remainder_images": len(unsupervised_original),
        "extra_dump_keys": len(extra_dump_keys),
        "extra_rewritten_keys": len(extra_rewritten_keys),
        "label_alignment": dict(sorted(alignment_counts.items())),
        "fatal_mismatches": dict(fatal_mismatch_counts),
        "discarded_caption_links": dict(sorted(discarded_link_counts.items())),
        "caption_match_types": dict(sorted(match_type_counts.items())),
        "unique_reading_texts": len(selected_text_counts),
    }

    if fatal_mismatch_counts and not args.allow_label_mismatches:
        raise RuntimeError(
            "Structural caption alignment mismatch detected; refusing to merge. "
            f"Counts={dict(fatal_mismatch_counts)}. Re-run with "
            "--allow-label-mismatches only after inspection."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reading_payload = {
        "schema_version": 2,
        "generator": "build_coco_spright_reading_manifests.py",
        "sources": {
            "ocr_dump": args.ocr_dump.resolve().as_posix(),
            "original_labels": args.original_labels.resolve().as_posix(),
            "rewritten_labels": args.rewritten_labels.resolve().as_posix(),
            "coco_root": args.coco_root.resolve().as_posix(),
        },
        "selection": {
            "min_confidence": float(args.min_confidence),
            "allowed_rotations": sorted(allowed_rotations),
            "strict_match_types": sorted(strict_types),
            "min_alnum_chars": int(args.min_alnum_chars),
            "dedupe_iou": float(args.dedupe_iou),
        },
        "summary": summary,
        "rows": reading_rows,
    }
    atomic_write_json(output_dir / "coco_spright_reading_supervised_conf999.json", reading_payload)
    atomic_write_json(output_dir / "coco_spright_unsupervised_original_conf999.json", unsupervised_original)
    atomic_write_json(output_dir / "coco_spright_unsupervised_rewritten_conf999.json", unsupervised_rewritten)
    atomic_write_json(
        output_dir / "coco_spright_reading_summary.json",
        {
            **summary,
            "settings": reading_payload["selection"],
            "extra_dump_keys_sample": extra_dump_keys[:20],
            "extra_rewritten_keys_sample": extra_rewritten_keys[:20],
            "top_reading_texts": selected_text_counts.most_common(100),
        },
    )
    print(json.dumps(summary, indent=2))
    print(f"[saved] {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
