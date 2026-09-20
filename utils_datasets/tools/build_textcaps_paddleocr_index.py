#!/usr/bin/env python3
"""Build a high-confidence PaddleOCR index for TextCaps images.

The script records only OCR detections whose recognition confidence is at least
``--min-confidence``.  By default it scans upright images only.  Optional
explicit -90/+90 degree image rotations can be enabled with ``--rotations``;
180 degrees is intentionally unsupported.

The final JSON contains only images with at least one accepted detection.  A
JSONL progress sidecar is written after every image so interrupted scans can be
resumed safely.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_ROTATIONS = {0, -90, 90}


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def bbox_from_polygon(points: Sequence[Sequence[float]]) -> List[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def polygon_from_box(box: Sequence[float]) -> List[List[float]]:
    x0, y0, x1, y1 = [float(value) for value in box]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def clip_polygon(points: Sequence[Sequence[float]], width: int, height: int) -> List[List[float]]:
    return [
        [
            max(0.0, min(float(width), float(point[0]))),
            max(0.0, min(float(height), float(point[1]))),
        ]
        for point in points
    ]


def map_polygon_to_original(
    points: Sequence[Sequence[float]],
    rotation: int,
    original_width: int,
    original_height: int,
) -> List[List[float]]:
    """Map coordinates from an explicitly rotated image back to the original."""
    mapped: List[List[float]] = []
    for point in points:
        x_rot, y_rot = float(point[0]), float(point[1])
        if rotation == 0:
            x, y = x_rot, y_rot
        elif rotation == 90:  # input was rotated 90 degrees counter-clockwise
            x, y = float(original_width) - y_rot, x_rot
        elif rotation == -90:  # input was rotated 90 degrees clockwise
            x, y = y_rot, float(original_height) - x_rot
        else:
            raise ValueError(f"Unsupported rotation {rotation}; valid={sorted(SUPPORTED_ROTATIONS)}")
        mapped.append([x, y])
    return clip_polygon(mapped, original_width, original_height)


def rotate_image(image: Image.Image, rotation: int) -> Image.Image:
    if rotation == 0:
        return image
    if rotation == 90:
        return image.transpose(Image.Transpose.ROTATE_90)
    if rotation == -90:
        return image.transpose(Image.Transpose.ROTATE_270)
    raise ValueError(f"Unsupported rotation {rotation}")


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


def _as_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _as_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_python(item) for item in value]
    return value


def _unwrap_v3_result(result: Any) -> Mapping[str, Any]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    payload = _as_python(payload)
    if not isinstance(payload, Mapping):
        return {}
    nested = payload.get("res")
    return nested if isinstance(nested, Mapping) else payload


def parse_v3_results(results: Iterable[Any]) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for result in results:
        payload = _unwrap_v3_result(result)
        texts = list(payload.get("rec_texts") or [])
        scores = list(payload.get("rec_scores") or [])
        polygons = list(payload.get("rec_polys") or [])
        boxes = list(payload.get("rec_boxes") or [])
        for index, text in enumerate(texts):
            if index >= len(scores):
                continue
            if index < len(polygons):
                polygon = _as_python(polygons[index])
            elif index < len(boxes):
                polygon = polygon_from_box(_as_python(boxes[index]))
            else:
                continue
            detections.append(
                {
                    "text": str(text),
                    "confidence": float(scores[index]),
                    "polygon": [[float(x), float(y)] for x, y in polygon],
                }
            )
    return detections


def _looks_like_v2_line(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (list, tuple))
        and isinstance(value[1], (list, tuple))
        and len(value[1]) >= 2
        and isinstance(value[1][0], str)
    )


def _iter_v2_lines(value: Any) -> Iterator[Sequence[Any]]:
    if _looks_like_v2_line(value):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_v2_lines(item)


def parse_v2_results(results: Any) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    for line in _iter_v2_lines(results):
        polygon, recognition = line
        text, confidence = recognition[0], recognition[1]
        detections.append(
            {
                "text": str(text),
                "confidence": float(confidence),
                "polygon": [[float(x), float(y)] for x, y in polygon],
            }
        )
    return detections


class PaddleOCRAdapter:
    def __init__(self, lang: str, device: str):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("PaddleOCR is not installed in this Python environment") from exc

        self.backend = "v3_predict"
        try:
            self.ocr = PaddleOCR(
                lang=lang,
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_rec_score_thresh=0.0,
            )
        except (TypeError, ValueError):
            self.backend = "legacy_ocr"
            self.ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=False,
                use_gpu=str(device).casefold().startswith("gpu"),
                show_log=False,
            )

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        array = np.asarray(image.convert("RGB"))
        if self.backend == "v3_predict":
            try:
                return parse_v3_results(self.ocr.predict(array))
            except (AttributeError, TypeError):
                self.backend = "legacy_ocr"
        try:
            results = self.ocr.ocr(array, cls=False)
        except TypeError:
            results = self.ocr.ocr(array)
        return parse_v2_results(results)


def merge_observations(observations: Sequence[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for observation in sorted(observations, key=lambda row: float(row["confidence"]), reverse=True):
        target: Optional[Dict[str, Any]] = None
        for group in groups:
            if group["norm_text"] != observation["norm_text"]:
                continue
            if bbox_iou(group["bbox_xyxy_original"], observation["bbox_xyxy_original"]) >= iou_threshold:
                target = group
                break
        if target is None:
            target = {
                "text": observation["text"],
                "norm_text": observation["norm_text"],
                "best_confidence": float(observation["confidence"]),
                "polygon_original": observation["polygon_original"],
                "bbox_xyxy_original": observation["bbox_xyxy_original"],
                "bbox_xywh_norm": observation["bbox_xywh_norm"],
                "observations": [],
            }
            groups.append(target)
        target["observations"].append(observation)
        if float(observation["confidence"]) > float(target["best_confidence"]):
            target.update(
                {
                    "text": observation["text"],
                    "best_confidence": float(observation["confidence"]),
                    "polygon_original": observation["polygon_original"],
                    "bbox_xyxy_original": observation["bbox_xyxy_original"],
                    "bbox_xywh_norm": observation["bbox_xywh_norm"],
                }
            )
    return groups


def scan_image(
    adapter: PaddleOCRAdapter,
    image_path: Path,
    image_root: Path,
    rotations: Sequence[int],
    min_confidence: float,
    min_alnum_chars: int,
    dedupe_iou: float,
) -> Dict[str, Any]:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    observations: List[Dict[str, Any]] = []
    for rotation in rotations:
        rotated = rotate_image(image, rotation)
        for detection in adapter.predict(rotated):
            confidence = float(detection["confidence"])
            text = str(detection["text"]).strip()
            norm_text = normalize_text(text)
            if confidence < min_confidence:
                continue
            if sum(character.isalnum() for character in norm_text) < min_alnum_chars:
                continue
            polygon_rotated = detection["polygon"]
            polygon_original = map_polygon_to_original(polygon_rotated, rotation, width, height)
            x0, y0, x1, y1 = bbox_from_polygon(polygon_original)
            observations.append(
                {
                    "text": text,
                    "norm_text": norm_text,
                    "confidence": confidence,
                    "rotation_applied_degrees": int(rotation),
                    "polygon_rotated": polygon_rotated,
                    "polygon_original": polygon_original,
                    "bbox_xyxy_original": [x0, y0, x1, y1],
                    "bbox_xywh_norm": [
                        x0 / width,
                        y0 / height,
                        max(0.0, x1 - x0) / width,
                        max(0.0, y1 - y0) / height,
                    ],
                }
            )
    detections = merge_observations(observations, dedupe_iou)
    return {
        "image": image_path.relative_to(image_root).as_posix(),
        "absolute_path": image_path.resolve().as_posix(),
        "split": image_path.relative_to(image_root).parts[0],
        "width": width,
        "height": height,
        "accepted_detection_count": len(detections),
        "detections": detections,
    }


def enumerate_images(image_root: Path, splits: Sequence[str]) -> List[Path]:
    images: List[Path] = []
    for split in splits:
        split_root = image_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"TextCaps split directory does not exist: {split_root}")
        images.extend(
            path
            for path in split_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
    return sorted(images, key=lambda path: path.as_posix().casefold())


def read_progress(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid progress JSONL at {path}:{line_number}: {exc}") from exc
            rows[str(row["image"])] = row
    return rows


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("TextCaps/images"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--output", type=Path, default=Path("textcaps_paddleocr_conf999.json"))
    parser.add_argument(
        "--rotations",
        nargs="+",
        type=int,
        default=[0],
        help="Explicit image rotations to scan. Valid: 0 -90 90. Default: upright only.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.999)
    parser.add_argument("--min-alnum-chars", type=int, default=1)
    parser.add_argument("--dedupe-iou", type=float, default=0.50)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 scans all images.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true", help="Delete existing progress and restart.")
    parser.add_argument("--keep-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rotations = list(dict.fromkeys(int(value) for value in args.rotations))
    invalid = sorted(set(rotations) - SUPPORTED_ROTATIONS)
    if invalid:
        raise ValueError(f"Unsupported rotations {invalid}; valid={sorted(SUPPORTED_ROTATIONS)}")
    if 180 in rotations or -180 in rotations:
        raise ValueError("180-degree OCR is intentionally unsupported")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be in [0, 1]")

    image_root = args.image_root.resolve()
    output = args.output.resolve()
    progress = output.with_suffix(output.suffix + ".progress.jsonl")
    if args.overwrite:
        output.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)

    images = enumerate_images(image_root, args.splits)
    if args.limit > 0:
        images = images[: args.limit]
    completed = read_progress(progress)
    adapter = PaddleOCRAdapter(args.lang, args.device)

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with progress.open("a", encoding="utf-8") as progress_handle:
        for index, image_path in enumerate(images, start=1):
            relative = image_path.relative_to(image_root).as_posix()
            if relative in completed:
                continue
            try:
                row = scan_image(
                    adapter,
                    image_path,
                    image_root,
                    rotations,
                    args.min_confidence,
                    args.min_alnum_chars,
                    args.dedupe_iou,
                )
                row["status"] = "ok"
            except Exception as exc:  # preserve the path and continue the dataset scan
                row = {
                    "image": relative,
                    "absolute_path": image_path.resolve().as_posix(),
                    "split": relative.split("/", 1)[0],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "accepted_detection_count": 0,
                    "detections": [],
                }
            progress_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress_handle.flush()
            completed[relative] = row
            if args.log_every > 0 and index % args.log_every == 0:
                accepted = sum(row.get("accepted_detection_count", 0) > 0 for row in completed.values())
                elapsed = time.time() - started
                print(f"[scan] {index}/{len(images)} accepted_images={accepted} elapsed={elapsed:.1f}s", flush=True)

    ordered_rows = [completed[path.relative_to(image_root).as_posix()] for path in images]
    errors = [row for row in ordered_rows if row.get("status") == "error"]
    accepted_rows = [
        {key: value for key, value in row.items() if key != "status"}
        for row in ordered_rows
        if row.get("status") == "ok" and int(row.get("accepted_detection_count", 0)) > 0
    ]
    split_counts: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        split_rows = [row for row in ordered_rows if row.get("split") == split]
        split_counts[split] = {
            "processed_images": len(split_rows),
            "accepted_images": sum(int(row.get("accepted_detection_count", 0)) > 0 for row in split_rows),
            "accepted_detections": sum(int(row.get("accepted_detection_count", 0)) for row in split_rows),
            "errors": sum(row.get("status") == "error" for row in split_rows),
        }

    payload = {
        "schema_version": 1,
        "generator": "build_textcaps_paddleocr_index.py",
        "image_root": image_root.as_posix(),
        "paddleocr_version": package_version("paddleocr"),
        "paddlepaddle_version": package_version("paddlepaddle") or package_version("paddlepaddle-gpu"),
        "settings": {
            "splits": list(args.splits),
            "rotations": rotations,
            "min_confidence": float(args.min_confidence),
            "min_alnum_chars": int(args.min_alnum_chars),
            "dedupe_iou": float(args.dedupe_iou),
            "lang": args.lang,
            "device": args.device,
            "backend": adapter.backend,
        },
        "summary": {
            "processed_images": len(ordered_rows),
            "accepted_images": len(accepted_rows),
            "accepted_detections": sum(row["accepted_detection_count"] for row in accepted_rows),
            "errors": len(errors),
            "by_split": split_counts,
        },
        "entries": accepted_rows,
        "errors": [
            {
                "image": row["image"],
                "error_type": row.get("error_type"),
                "error": row.get("error"),
            }
            for row in errors
        ],
    }
    atomic_write_json(output, payload)
    if not args.keep_progress:
        progress.unlink(missing_ok=True)
    print(json.dumps(payload["summary"], indent=2))
    print(f"[saved] {output}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
