"""COCO-SPRIGHT high-trust reading manifest helpers.

This branch intentionally restores ONLY the derived COCO-SPRIGHT supervision.
Historical TextCaps continues to read TextCaps/manifests/*.jsonl and
there is deliberately no TextCaps derived-reading loader in this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

def normalize_image_key(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    text = text.lstrip("/")
    if text.casefold().startswith("images/"):
        text = text[7:]
    return text.casefold()

def _read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _as_entries(payload: Any, field: str) -> List[Dict[str, Any]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get(field), list):
        raise ValueError(f"Expected a JSON object with list field {field!r}")
    return [dict(row) for row in payload[field] if isinstance(row, Mapping)]

def _require_high_trust_coco(payload: Mapping[str, Any], path: Path) -> None:
    selection = payload.get("selection") or {}
    confidence = float(selection.get("min_confidence", 0.0))
    rotations = {int(value) % 360 for value in selection.get("allowed_rotations") or []}
    if confidence < 0.999:
        raise ValueError(f"COCO-SPRIGHT reading manifest is not high-trust: min_confidence={confidence} in {path}")
    if 180 in rotations or not rotations.issubset({0, 90, 270}):
        raise ValueError(f"COCO-SPRIGHT manifest contains unsupported rotations={sorted(rotations)} in {path}")

def _valid_normalized_box(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return False
    try:
        x, y, width, height = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return (
        all(np.isfinite(item) for item in (x, y, width, height))
        and width > 0.0
        and height > 0.0
        and x < 1.0
        and y < 1.0
        and x + width > 0.0
        and y + height > 0.0
    )

def _target_has_box(target: Mapping[str, Any]) -> bool:
    if _valid_normalized_box(target.get("bbox_xywh_norm")):
        return True
    return any(
        isinstance(occurrence, Mapping) and _valid_normalized_box(occurrence.get("bbox_xywh_norm"))
        for occurrence in target.get("occurrences") or []
    )

def _validate_reading_targets(targets: Sequence[Mapping[str, Any]], *, source: str) -> None:
    if not targets:
        raise ValueError(f"{source} has no readable targets")
    for index, target in enumerate(targets):
        text = str(target.get("text") or target.get("norm_text") or "").strip()
        if not text:
            raise ValueError(f"{source} target {index} has no text")
        if not _target_has_box(target):
            raise ValueError(f"{source} target {text!r} has no valid normalized OCR box")

def load_coco_reading_manifest(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trusted COCO-SPRIGHT reading manifest does not exist: {path}")
    payload = _read_json(path)
    _require_high_trust_coco(payload, path)
    rows = _as_entries(payload, "rows")
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = normalize_image_key(row.get("image_key"))
        targets = [dict(target) for target in row.get("reading_targets") or [] if isinstance(target, Mapping)]
        if not key or not targets:
            continue
        _validate_reading_targets(targets, source=f"COCO-SPRIGHT manifest row {key}")
        row["reading_targets"] = targets
        if key in out:
            raise ValueError(f"Duplicate COCO reading-manifest key after normalization: {key}")
        out[key] = row
    return out

def target_texts(targets: Sequence[Mapping[str, Any]]) -> List[str]:
    seen = set()
    values: List[str] = []
    for target in targets:
        text = str(target.get("text") or target.get("norm_text") or "").strip()
        norm = " ".join(text.casefold().split())
        if text and norm not in seen:
            seen.add(norm)
            values.append(text)
    return values

def _iter_normalized_boxes(target: Mapping[str, Any]) -> Iterable[Sequence[float]]:
    direct = target.get("bbox_xywh_norm")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)) and len(direct) == 4:
        yield direct
    for occurrence in target.get("occurrences") or []:
        if not isinstance(occurrence, Mapping):
            continue
        box = occurrence.get("bbox_xywh_norm")
        if isinstance(box, Sequence) and not isinstance(box, (str, bytes)) and len(box) == 4:
            yield box

def mask_from_targets(image_size: Tuple[int, int], targets: Sequence[Mapping[str, Any]]) -> Image.Image:
    width, height = [int(value) for value in image_size]
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for target in targets:
        for box in _iter_normalized_boxes(target):
            x, y, w, h = [float(value) for value in box]
            x0 = max(0, min(width, int(round(x * width))))
            y0 = max(0, min(height, int(round(y * height))))
            x1 = max(x0 + 1, min(width, int(round((x + max(0.0, w)) * width))))
            y1 = max(y0 + 1, min(height, int(round((y + max(0.0, h)) * height))))
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
    return mask

def labels_from_value(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []
