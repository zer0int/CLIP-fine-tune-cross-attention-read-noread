from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoProcessor

from checkpoint_spec import load_trusted_checkpoint
from xattn_inference import resolve_xattn_model_reference


DETAIL_KEYS = (
    "logits_per_image",
    "base_image_embedding",
    "content_image_embedding",
    "content_correction",
    "content_text_embedding",
    "raw_read_logits",
    "read_logits",
    "null_read_logits",
    "relative_read_logits",
    "early_orthographic_logits",
    "trust_gate",
    "source_gate",
    "route_gate",
    "auto_read_contribution",
    "source_logits",
    "source_stats",
    "glyph_logits",
    "register_mask",
    "patch_token_norms",
)


def _source_model(loaded: Any):
    if isinstance(loaded, torch.nn.Module):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("model"), torch.nn.Module):
        return loaded["model"]
    raise TypeError(
        "The checkpoint contains only tensors, so original-forward parity cannot be run. "
        "Use the full trusted pickle/module checkpoint."
    )


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_is_bool = left.dtype == torch.bool
    right_is_bool = right.dtype == torch.bool
    left = left.detach().cpu()
    right = right.detach().cpu()
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "source_shape": list(left.shape),
            "hf_shape": list(right.shape),
        }
    if left_is_bool or right_is_bool:
        mismatches = int((left.bool() != right.bool()).sum())
        return {
            "shape_match": True,
            "mismatch_count": mismatches,
            "exact": mismatches == 0,
        }
    left = left.float()
    right = right.float()
    difference = (left - right).abs()
    return {
        "shape_match": True,
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
        "allclose_atol_2e-4_rtol_2e-4": bool(
            torch.allclose(left, right, atol=2.0e-4, rtol=2.0e-4)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Localize original-pickle versus converted full x-attention differences"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="full x-attention export directory; default: auto-discover locally",
    )
    parser.add_argument("--output", type=Path, default=Path("xattn_parity_report.json"))
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_reference = resolve_xattn_model_reference(args.model)
    original = _source_model(load_trusted_checkpoint(args.checkpoint)).eval().to(device)
    converted = (
        AutoModel.from_pretrained(model_reference, trust_remote_code=True)
        .eval()
        .to(device)
    )
    processor = AutoProcessor.from_pretrained(model_reference, trust_remote_code=True)
    prompts = ["a photo of a dog", "a photo of the word banana", "a photo of a sign"]
    input_ids = processor(text=prompts, padding="max_length", return_tensors="pt")[
        "input_ids"
    ].to(device)
    image_size = int(converted.config.vision_config.image_size)
    pixel_values = torch.randn(
        args.batch_size, 3, image_size, image_size, device=device
    )

    with torch.inference_mode():
        source_output = original.forward_modes(
            pixel_values, input_ids, apply_content_correction=True, return_details=True
        )
        hf_output = converted(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mode="none",
            correction=True,
            return_details=True,
        )
    hf_details = dict(hf_output.details or {})
    hf_details["logits_per_image"] = hf_output.logits_per_image
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model": str(args.model.resolve()),
        "seed": args.seed,
        "components": {},
        "visual_states": {},
    }
    for key in DETAIL_KEYS:
        source_value = source_output.get(key)
        hf_value = hf_details.get(key)
        if torch.is_tensor(source_value) and torch.is_tensor(hf_value):
            report["components"][key] = _comparison(source_value, hf_value)
        else:
            report["components"][key] = {
                "comparable": False,
                "source_type": type(source_value).__name__,
                "hf_type": type(hf_value).__name__,
            }
    for block, source_state in source_output["visual_states"].items():
        hf_state = hf_details["visual_states"].get(int(block))
        report["visual_states"][str(block)] = _comparison(source_state, hf_state)
    numeric = [
        (name, values["max_abs"])
        for name, values in report["components"].items()
        if "max_abs" in values
    ]
    numeric += [
        (f"visual_state_B{name}", values["max_abs"])
        for name, values in report["visual_states"].items()
        if "max_abs" in values
    ]
    numeric.sort(key=lambda item: item[1], reverse=True)
    report["largest_differences"] = [
        {"component": name, "max_abs": value} for name, value in numeric[:12]
    ]
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output.resolve()}")
    for item in report["largest_differences"]:
        print(f"{item['component']:36s} max_abs={item['max_abs']:.8g}")


if __name__ == "__main__":
    main()
