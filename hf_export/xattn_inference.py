from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


NO_TEXT_DETECTED = "NO_TEXT_DETECTED"


def resolve_xattn_model_reference(
    model: str | Path | None = None,
    *,
    script_dir: Path | None = None,
    working_dir: Path | None = None,
) -> str | Path:
    """Resolve and validate a local full x-attention export or preserve an HF ID.

    A generated benchmark normally lives inside ``full_xattn_model`` and can
    load ``.`` directly.  The conversion-package copy instead lives beside an
    output directory, so the default also searches one level down for a unique
    ``full_xattn_model/config.json``.  Explicit Hugging Face repository IDs are
    returned unchanged because they cannot be validated without downloading.
    """

    script_root = (script_dir or Path(__file__).resolve().parent).resolve()
    working_root = (working_dir or Path.cwd()).resolve()
    raw_model = None if model is None else str(model).strip()

    if raw_model not in {None, "", "."}:
        requested = Path(raw_model).expanduser()
        if not requested.exists():
            return raw_model
        search_roots = [requested.resolve()]
    else:
        search_roots = [script_root, working_root]

    unique_roots: list[Path] = []
    for root in search_roots:
        if root not in unique_roots:
            unique_roots.append(root)

    attempted: dict[Path, str] = {}

    def inspect(candidate: Path) -> bool:
        candidate = candidate.resolve()
        config_path = candidate / "config.json"
        if not config_path.is_file():
            attempted[candidate] = "config.json is missing"
            return False
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            attempted[candidate] = f"config.json cannot be read: {error}"
            return False
        model_type = config.get("model_type")
        if model_type != "xattn_clip":
            attempted[candidate] = (
                f"config.json model_type={model_type!r}, expected 'xattn_clip'"
            )
            return False
        return True

    # Prefer a model directory containing the running script or current shell.
    direct_matches = [root for root in unique_roots if inspect(root)]
    if direct_matches:
        return direct_matches[0]

    nested_candidates: list[Path] = []
    for root in unique_roots:
        if not root.is_dir():
            continue
        candidates = [root / "full_xattn_model"]
        candidates.extend(sorted(root.glob("*/full_xattn_model")))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in nested_candidates:
                nested_candidates.append(resolved)

    nested_matches = [
        candidate for candidate in nested_candidates if inspect(candidate)
    ]
    if len(nested_matches) == 1:
        return nested_matches[0]
    if len(nested_matches) > 1:
        choices = "\n".join(f"  - {candidate}" for candidate in nested_matches)
        raise ValueError(
            "Multiple full x-attention exports were found. Select one explicitly "
            f"with --model:\n{choices}"
        )

    details = "\n".join(
        f"  - {candidate}: {reason}" for candidate, reason in attempted.items()
    )
    raise FileNotFoundError(
        "Could not locate a full x-attention Hugging Face export. Point --model "
        "at the directory containing config.json and model.safetensors.\n"
        f"Checked:\n{details}"
    )


DEFAULT_PROMPT = "a photo of a {label}"


@dataclass(frozen=True)
class XAttnPrediction:
    label: str
    score: float
    candidate_index: int


def labels_with_null(
    labels: Sequence[str], null_candidate_index: int | None
) -> list[str]:
    """Map the model's exposed read-mode null column to a stable public label."""
    output = list(labels)
    if null_candidate_index is not None:
        if null_candidate_index != len(output):
            raise ValueError(
                "The model returned a null column at an unexpected candidate index: "
                f"index={null_candidate_index}, label_count={len(output)}"
            )
        output.append(NO_TEXT_DETECTED)
    return output


def predict_candidates(
    model,
    processor,
    images: Any | Sequence[Any],
    labels: Sequence[str],
    *,
    mode: str = "any",
    correction: bool | None = None,
    prompt: str = DEFAULT_PROMPT,
    device: str | torch.device | None = None,
) -> list[XAttnPrediction]:
    """Rank a shared candidate bank for one image or an image batch.

    In ``read`` mode the model appends its internal ``<text><null>`` candidate;
    a winning null column is returned as ``NO_TEXT_DETECTED``.
    """
    if not labels:
        raise ValueError("At least one candidate label is required")
    model_device = next(model.parameters()).device
    target_device = torch.device(device) if device is not None else model_device
    prompts = [prompt.format(label=label) for label in labels]
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(target_device)
    pixel_values = inputs["pixel_values"].to(target_device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mode=mode,
            correction=correction,
        )
    output_labels = labels_with_null(labels, output.null_candidate_index)
    scores = output.logits_per_image.float()
    indices = scores.argmax(dim=-1)
    return [
        XAttnPrediction(
            label=output_labels[int(index)],
            score=float(scores[row, index].cpu()),
            candidate_index=int(index),
        )
        for row, index in enumerate(indices)
    ]
