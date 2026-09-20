#!/usr/bin/env python3
r"""READ/null diagnostics on local controls.
Commands: hallucinations (content/read analysis), tap_transplants (frozen late-tap interventions), diagnostic (raw/calibrated logits and RN attention).
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_analysis import (_normalize_rows)

# HALLUCINATIONS
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor

from training_support.font_discovery import find_font_files


# -------------------------------------------------------------------------------------------------
#  Experiment definition
# -------------------------------------------------------------------------------------------------

PROMPT_WORDS = (
    "banana",
    "bananas",
    "apple",
    "apples",
    "grape",
    "grapes",
    "sticker",
    "bird",
    "cat",
    "dog",
    "word",
    "text",
)

PROMPT_TEMPLATES = (
    ("photo", "a photo of a {prompt_word}"),
    ("plain", "{prompt_word}"),
    ("article", "a {prompt_word}"),
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_CORRECTION_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_FULL_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_IMAGE_DIR = Path("image_sets/misc")
DEFAULT_OUTPUT_DIR = Path("OUT_BRIDGE_INSPECTION/no_text_hallucination_probe_v2")

# Previously identified correction/content-pool typography-selective head.
DEFAULT_CORRECTION_TEXT_HEAD = 3


@dataclass(frozen=True)
class hallucinations_Sample:
    condition: str
    image_index: int
    image_name: str
    source_path: Path
    image: Image.Image
    overlay_word: str | None


# -------------------------------------------------------------------------------------------------
#  Small utilities
# -------------------------------------------------------------------------------------------------


def _as_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _feature_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    for name in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Cannot locate feature tensor in {type(output)!r}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "condition",
        "image",
        "image_index",
        "overlay_word",
        "template",
        "word",
        "model",
        "mode",
    ]
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hallucinations__image_paths(image_dir: Path) -> list[Path]:
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found in {image_dir.resolve()}")
    return paths


def _open_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


CONTROL_FONT_NAMES = (
    "arial.ttf",
    "arialbd.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
    "Arial.ttf",
    "Helvetica.ttc",
)


def _find_control_font(explicit: str | None, size: int) -> ImageFont.ImageFont:
    explicit_paths = (explicit,) if explicit else ()
    for candidate in find_font_files(
        explicit_paths,
        preferred_names=CONTROL_FONT_NAMES,
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def hallucinations__find_font(explicit: str | None, size: int) -> ImageFont.ImageFont:
    return _find_control_font(explicit, size)


def hallucinations__add_text_control(
    image: Image.Image,
    word: str,
    image_index: int,
    font_path: str | None,
) -> Image.Image:
    """Create a deterministic positive text control without touching the source file."""
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    width, height = output.size
    font_size = max(18, int(round(min(width, height) * 0.13)))
    font = hallucinations__find_font(font_path, font_size)
    stroke = max(2, int(round(font_size * 0.055)))
    bbox = draw.textbbox((0, 0), word, font=font, stroke_width=stroke)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(4, (width - text_w) // 2)
    position = image_index % 3
    if position == 0:
        y = max(4, int(height * 0.08))
    elif position == 1:
        y = max(4, (height - text_h) // 2)
    else:
        y = max(4, int(height * 0.86) - text_h)
    draw.text(
        (x, y),
        word,
        font=font,
        fill=(255, 255, 255),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0),
    )
    return output


def _build_samples(
    image_dir: Path,
    output_dir: Path,
    add_text_control: bool,
    font_path: str | None,
) -> dict[str, list[hallucinations_Sample]]:
    paths = hallucinations__image_paths(image_dir)
    clean: list[hallucinations_Sample] = []
    controls: list[hallucinations_Sample] = []
    control_dir = output_dir / "text_controls"
    if add_text_control:
        control_dir.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(paths):
        image = _open_rgb(path)
        clean.append(
            hallucinations_Sample(
                condition="clean",
                image_index=index,
                image_name=path.name,
                source_path=path,
                image=image,
                overlay_word=None,
            )
        )
        if add_text_control:
            word = PROMPT_WORDS[index % len(PROMPT_WORDS)]
            controlled = hallucinations__add_text_control(image, word, index, font_path)
            save_path = control_dir / f"{path.stem}__text_{word}.png"
            controlled.save(save_path)
            controls.append(
                hallucinations_Sample(
                    condition="text_control",
                    image_index=index,
                    image_name=path.name,
                    source_path=path,
                    image=controlled,
                    overlay_word=word,
                )
            )

    result = {"clean": clean}
    if controls:
        result["text_control"] = controls
    return result


def _batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _text_ids(processor: Any, template: str, device: torch.device) -> torch.Tensor:
    prompts = [template.format(prompt_word=word) for word in PROMPT_WORDS]
    encoded = processor(
        text=prompts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device)


def _pixel_values(processor: Any, samples: list[hallucinations_Sample], device: torch.device) -> torch.Tensor:
    encoded = processor(images=[sample.image for sample in samples], return_tensors="pt")
    return encoded["pixel_values"].to(device)


def _pixel_values_to_rgb(pixel_values: torch.Tensor, processor: Any) -> list[np.ndarray]:
    image_processor = getattr(processor, "image_processor", processor)
    mean = torch.tensor(
        getattr(image_processor, "image_mean", (0.48145466, 0.4578275, 0.40821073)),
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        getattr(image_processor, "image_std", (0.26862954, 0.26130258, 0.27577711)),
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    rgb = pixel_values.detach().float().cpu() * std + mean
    rgb = rgb.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return [array for array in rgb]


# -------------------------------------------------------------------------------------------------
#  Correction-pool attention extraction (works for both correction and full HF models)
# -------------------------------------------------------------------------------------------------


def _content_pool_attention(pool: torch.nn.Module, visual_tokens: torch.Tensor) -> torch.Tensor:
    """Reproduce the correction pool's attention weights exactly, returning [B,H,T]."""
    tokens = visual_tokens.float()
    batch, token_count, _ = tokens.shape
    x = F.layer_norm(
        tokens,
        pool.vision_ln.normalized_shape,
        pool.vision_ln.weight.float(),
        pool.vision_ln.bias.float(),
        pool.vision_ln.eps,
    )
    keys = F.linear(x, pool.k_proj.weight.float(), pool.k_proj.bias.float())
    keys = keys.view(batch, token_count, pool.heads, pool.head_dim).permute(0, 2, 1, 3)
    logits = torch.einsum("hd,bhtd->bht", pool.query.float(), keys)
    logits = logits / math.sqrt(pool.head_dim)

    keep = torch.ones((batch, token_count), dtype=torch.bool, device=tokens.device)
    keep[:, 0] = False  # CLS
    if token_count < 3:
        raise ValueError("Visual sequence is too short to exclude CLS and RN")
    keep[:, -1] = False  # RN

    if str(pool.architecture) == "sigmoid_all":
        valid_count = keep.sum(dim=-1).clamp(min=1).float()
        adjusted = logits + pool.sigmoid_head_bias.float()[None, :, None]
        adjusted = adjusted - valid_count.log()[:, None, None]
        attention = torch.sigmoid(adjusted) * keep[:, None, :].float()
    else:
        masked = logits.masked_fill(~keep[:, None, :], torch.finfo(logits.dtype).min)
        attention = masked.softmax(dim=-1)
        attention = attention * keep[:, None, :].float()
    return attention


def _mix_content_attention(
    pool: torch.nn.Module,
    states: dict[int, torch.Tensor],
    tap_logits: torch.Tensor,
    blocks: list[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor]:
    tap_weights = tap_logits.float().softmax(dim=0)
    per_block = {
        block: _content_pool_attention(pool, states[block]) for block in blocks
    }
    stacked = torch.stack([per_block[block] for block in blocks], dim=0)
    mixed = torch.einsum("k,kbht->bht", tap_weights, stacked)
    return mixed, per_block, tap_weights


def _spatial_map(attention: torch.Tensor, head_index: int) -> torch.Tensor:
    """Extract one head's spatial patch attention as [B,G,G], dropping CLS/RN."""
    if head_index < 0 or head_index >= attention.shape[1]:
        raise ValueError(
            f"Requested correction head H{head_index}, but attention has {attention.shape[1]} heads"
        )
    patches = attention[:, head_index, 1:-1]
    patch_count = patches.shape[-1]
    grid = int(round(math.sqrt(patch_count)))
    if grid * grid != patch_count:
        raise ValueError(f"Cannot reshape {patch_count} spatial patches to a square grid")
    return patches.reshape(patches.shape[0], grid, grid)


def _register_mask_from_norms(
    patch_norms: torch.Tensor,
    threshold: float,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    """Mirror the HF model's final-token register selection exactly."""
    register_mask = torch.zeros_like(patch_norms, dtype=torch.bool)
    minimum = max(0, int(minimum))
    maximum = int(maximum)
    for row in range(patch_norms.shape[0]):
        indices = torch.nonzero(
            patch_norms[row] > float(threshold), as_tuple=False
        ).flatten()
        if indices.numel() < minimum:
            count = min(max(1, minimum), patch_norms.shape[1])
            indices = torch.topk(patch_norms[row], k=count).indices
        elif maximum > 0 and indices.numel() > maximum:
            chosen = torch.topk(patch_norms[row, indices], k=maximum).indices
            indices = indices[chosen]
        register_mask[row, indices] = True
    return register_mask


def _grid_map(values: torch.Tensor) -> torch.Tensor:
    """Reshape [B,P] spatial patch values into [B,G,G]."""
    patch_count = values.shape[-1]
    grid = int(round(math.sqrt(patch_count)))
    if grid * grid != patch_count:
        raise ValueError(f"Cannot reshape {patch_count} patches to a square grid")
    return values.reshape(values.shape[0], grid, grid)


def _attention_metrics(
    mixed_attention: torch.Tensor,
    per_block: dict[int, torch.Tensor],
    head_index: int,
    register_mask: torch.Tensor | None = None,
    threshold_mask: torch.Tensor | None = None,
    patch_norms: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    patches = mixed_attention[:, :, 1:-1].float()
    head_mass = patches.sum(dim=-1)
    head_mean = patches.mean(dim=-1)
    head_max = patches.amax(dim=-1)
    topk = patches.topk(k=min(8, patches.shape[-1]), dim=-1).values.mean(dim=-1)
    other_indices = [index for index in range(patches.shape[1]) if index != head_index]
    other_mass = head_mass[:, other_indices].mean(dim=-1)
    h3_mass = head_mass[:, head_index]

    if register_mask is not None:
        if register_mask.shape != (patches.shape[0], patches.shape[-1]):
            raise ValueError(
                f"register_mask shape {tuple(register_mask.shape)} does not match "
                f"patch attention {(patches.shape[0], patches.shape[-1])}"
            )
        register_mask = register_mask.bool()
    if threshold_mask is not None:
        if threshold_mask.shape != (patches.shape[0], patches.shape[-1]):
            raise ValueError(
                f"threshold_mask shape {tuple(threshold_mask.shape)} does not match "
                f"patch attention {(patches.shape[0], patches.shape[-1])}"
            )
        threshold_mask = threshold_mask.bool()
    if patch_norms is not None:
        if patch_norms.shape != (patches.shape[0], patches.shape[-1]):
            raise ValueError(
                f"patch_norms shape {tuple(patch_norms.shape)} does not match "
                f"patch attention {(patches.shape[0], patches.shape[-1])}"
            )
        patch_norms = patch_norms.float()

    rows: list[dict[str, float]] = []
    for row in range(patches.shape[0]):
        metrics: dict[str, float] = {
            f"h{head_index}_mass": _as_float(h3_mass[row]),
            f"h{head_index}_mean": _as_float(head_mean[row, head_index]),
            f"h{head_index}_max": _as_float(head_max[row, head_index]),
            f"h{head_index}_top8": _as_float(topk[row, head_index]),
            f"h{head_index}_minus_other_mass": _as_float(h3_mass[row] - other_mass[row]),
            f"h{head_index}_over_other_mass": _as_float(
                h3_mass[row] / (other_mass[row] + 1.0e-12)
            ),
        }
        for head in range(patches.shape[1]):
            metrics[f"h{head}_mass"] = _as_float(head_mass[row, head])

        h = patches[row, head_index]
        if register_mask is not None:
            reg = register_mask[row]
            nonreg = ~reg
            reg_mass = h[reg].sum() if bool(reg.any()) else h.new_zeros(())
            nonreg_mass = h[nonreg].sum() if bool(nonreg.any()) else h.new_zeros(())
            total = reg_mass + nonreg_mass
            argmax = int(h.argmax().item())
            metrics.update(
                {
                    "register_count": int(reg.sum().item()),
                    f"h{head_index}_register_mass": _as_float(reg_mass),
                    f"h{head_index}_nonregister_mass": _as_float(nonreg_mass),
                    f"h{head_index}_register_fraction": _as_float(
                        reg_mass / (total + 1.0e-12)
                    ),
                    f"h{head_index}_argmax_is_register": int(bool(reg[argmax].item())),
                    f"h{head_index}_max_register": (
                        _as_float(h[reg].max()) if bool(reg.any()) else 0.0
                    ),
                    f"h{head_index}_max_nonregister": (
                        _as_float(h[nonreg].max()) if bool(nonreg.any()) else 0.0
                    ),
                }
            )
            if patch_norms is not None:
                norm = patch_norms[row]
                metrics.update(
                    {
                        "patch_norm_max": _as_float(norm.max()),
                        "patch_norm_mean": _as_float(norm.mean()),
                        f"h{head_index}_argmax_patch_norm": _as_float(norm[argmax]),
                        f"h{head_index}_attention_weighted_patch_norm": _as_float(
                            (h * norm).sum() / (h.sum() + 1.0e-12)
                        ),
                    }
                )
        if threshold_mask is not None:
            high = threshold_mask[row]
            high_mass = h[high].sum() if bool(high.any()) else h.new_zeros(())
            total = h.sum()
            argmax = int(h.argmax().item())
            metrics.update(
                {
                    "norm60_count": int(high.sum().item()),
                    f"h{head_index}_norm60_mass": _as_float(high_mass),
                    f"h{head_index}_norm60_fraction": _as_float(
                        high_mass / (total + 1.0e-12)
                    ),
                    f"h{head_index}_argmax_is_norm60": int(bool(high[argmax].item())),
                }
            )

        for block, attention in per_block.items():
            block_patches = attention[row, head_index, 1:-1].float()
            metrics[f"b{block}_h{head_index}_mass"] = _as_float(block_patches.sum())
            metrics[f"b{block}_h{head_index}_max"] = _as_float(block_patches.max())
            if register_mask is not None:
                reg = register_mask[row]
                block_total = block_patches.sum()
                block_reg = (
                    block_patches[reg].sum() if bool(reg.any()) else block_patches.new_zeros(())
                )
                argmax = int(block_patches.argmax().item())
                metrics[f"b{block}_h{head_index}_register_mass"] = _as_float(block_reg)
                metrics[f"b{block}_h{head_index}_register_fraction"] = _as_float(
                    block_reg / (block_total + 1.0e-12)
                )
                metrics[f"b{block}_h{head_index}_argmax_is_register"] = int(
                    bool(reg[argmax].item())
                )
            if threshold_mask is not None:
                high = threshold_mask[row]
                block_high = (
                    block_patches[high].sum()
                    if bool(high.any())
                    else block_patches.new_zeros(())
                )
                metrics[f"b{block}_h{head_index}_norm60_fraction"] = _as_float(
                    block_high / (block_patches.sum() + 1.0e-12)
                )
                metrics[f"b{block}_h{head_index}_argmax_is_norm60"] = int(
                    bool(high[int(block_patches.argmax().item())].item())
                )
        rows.append(metrics)
    return rows


# -------------------------------------------------------------------------------------------------
#  Correction-only model
# -------------------------------------------------------------------------------------------------


def _correction_backend_modules(model: Any) -> tuple[str, torch.nn.Module, torch.Tensor]:
    """Resolve the candidate-independent CONTENT correction across HF export generations."""
    if hasattr(model, "_vision_with_rn") and hasattr(model, "content_pool"):
        return "rn_correction_export", model.content_pool, model.content_tap_logits
    if hasattr(model, "_vision_with_intermediates") and hasattr(model, "read_implant"):
        return (
            "xattn_content_branch",
            model.read_implant.content_pool,
            model.read_implant.content_tap_logits,
        )
    raise TypeError(
        "Correction control requires either an RN correction HF model "
        "(_vision_with_rn + content_pool) or the final x-attn HF model "
        "(_vision_with_intermediates + read_implant.content_pool). "
        f"Loaded {type(model).__name__}."
    )


def _correction_visual_batch(
    model: Any,
    pixel_values: torch.Tensor,
    backend: str,
    blocks: list[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor]:
    """Return base embedding, CONTENT tap states, and final visual tokens."""
    if backend == "rn_correction_export":
        vision_outputs, tapped = model._vision_with_rn(pixel_values)
        base_image = model.visual_projection(vision_outputs.pooler_output).float()
        states = dict(zip(blocks, tapped))
        final_tokens = vision_outputs.last_hidden_state
    elif backend == "xattn_content_branch":
        image_info = model._vision_with_intermediates(
            pixel_values, return_final_tokens=True
        )
        base_image = image_info["image_embedding"].float()
        missing = [block for block in blocks if block not in image_info["states"]]
        if missing:
            raise RuntimeError(
                f"Final x-attn model did not capture CONTENT tap blocks {missing}"
            )
        states = {block: image_info["states"][block] for block in blocks}
        final_tokens = image_info["final_tokens"]
        if final_tokens is None:
            raise RuntimeError("Final x-attn model did not return final visual tokens")
    else:
        raise ValueError(f"Unknown correction backend {backend!r}")
    return base_image, states, final_tokens


def _run_correction_model(
    model_path: str | Path,
    samples_by_condition: dict[str, list[hallucinations_Sample]],
    batch_size: int,
    device: torch.device,
    head_index: int,
):
    print(f"[correction] Loading {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    static_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    maps: dict[str, dict[str, np.ndarray]] = {}
    register_masks: dict[str, dict[str, np.ndarray]] = {}
    model_register_masks: dict[str, dict[str, np.ndarray]] = {}
    patch_norm_maps: dict[str, dict[str, np.ndarray]] = {}
    rgb_views: dict[str, dict[str, np.ndarray]] = {}
    model_meta: dict[str, Any] = {
        "path": str(model_path),
        "class": type(model).__name__,
        "dtype": str(next(model.parameters()).dtype),
        "correction_head": head_index,
        "register_norm_threshold": float(model.config.register_norm_threshold),
        "register_min": int(model.config.register_min),
        "register_max": int(model.config.register_max),
    }

    # The historical correction-only export and the final x-attn export store
    # the same CONTENT correction under different object layouts.
    correction_backend, content_pool, content_tap_logits = _correction_backend_modules(model)
    model_meta["correction_backend"] = correction_backend

    for condition, samples in samples_by_condition.items():
        maps[condition] = {}
        register_masks[condition] = {}
        model_register_masks[condition] = {}
        patch_norm_maps[condition] = {}
        rgb_views[condition] = {}
        for _, batch_samples in _batches(samples, batch_size):
            pixel_values = _pixel_values(processor, batch_samples, device)
            for sample, rgb in zip(batch_samples, _pixel_values_to_rgb(pixel_values, processor)):
                rgb_views[condition][sample.image_name] = rgb

            with torch.inference_mode():
                blocks = [int(value) for value in model.config.read_tap_blocks]
                base_image, states, final_tokens = _correction_visual_batch(
                    model, pixel_values, correction_backend, blocks
                )

                mixed_attention, per_block, tap_weights = _mix_content_attention(
                    content_pool,
                    states,
                    content_tap_logits,
                    blocks,
                )
                final_spatial = final_tokens[:, 1:-1, :].detach().float()
                patch_norms = final_spatial.norm(dim=-1)
                threshold_mask = patch_norms > float(model.config.register_norm_threshold)
                register_mask = _register_mask_from_norms(
                    patch_norms,
                    float(model.config.register_norm_threshold),
                    int(model.config.register_min),
                    int(model.config.register_max),
                )
                attention_stats = _attention_metrics(
                    mixed_attention,
                    per_block,
                    head_index,
                    register_mask=register_mask,
                    threshold_mask=threshold_mask,
                    patch_norms=patch_norms,
                )
                head_maps = _spatial_map(mixed_attention, head_index)
                register_grids = _grid_map(threshold_mask.float())
                model_register_grids = _grid_map(register_mask.float())
                patch_norm_grids = _grid_map(patch_norms)

                weights = content_tap_logits.float().softmax(dim=0)
                corrections = torch.stack([content_pool(states[block]) for block in blocks])
                correction_vector = torch.einsum("k,kbd->bd", weights, corrections).float()
                corrected_image = base_image + correction_vector

                base_norm = _normalize_rows(base_image)
                corrected_norm = _normalize_rows(corrected_image)
                correction_norm = correction_vector.norm(dim=-1)
                correction_relative = correction_norm / (base_image.norm(dim=-1) + 1.0e-12)
                base_corrected_cos = (base_norm * corrected_norm).sum(dim=-1)

            for local_index, sample in enumerate(batch_samples):
                row: dict[str, Any] = {
                    "model": "correction",
                    "condition": condition,
                    "image": sample.image_name,
                    "image_index": sample.image_index,
                    "overlay_word": sample.overlay_word or "",
                    "correction_vector_norm": _as_float(correction_norm[local_index]),
                    "correction_relative_norm": _as_float(correction_relative[local_index]),
                    "base_corrected_cos": _as_float(base_corrected_cos[local_index]),
                }
                for block, weight in zip(blocks, tap_weights):
                    row[f"tap_weight_b{block}"] = _as_float(weight)
                row.update(attention_stats[local_index])
                static_rows.append(row)
                maps[condition][sample.image_name] = (
                    head_maps[local_index].detach().float().cpu().numpy()
                )
                register_masks[condition][sample.image_name] = (
                    register_grids[local_index].detach().float().cpu().numpy()
                )
                model_register_masks[condition][sample.image_name] = (
                    model_register_grids[local_index].detach().float().cpu().numpy()
                )
                patch_norm_maps[condition][sample.image_name] = (
                    patch_norm_grids[local_index].detach().float().cpu().numpy()
                )

            # Text embeddings are template-specific but image embeddings are not.
            for template_name, template in PROMPT_TEMPLATES:
                input_ids = _text_ids(processor, template, device)
                with torch.inference_mode():
                    text_features = _feature_tensor(model.get_text_features(input_ids=input_ids))
                    text_norm = _normalize_rows(text_features)
                    scale = model.logit_scale.detach().float().exp()
                    logits_off = scale * (base_norm @ text_norm.t())
                    logits_on = scale * (corrected_norm @ text_norm.t())
                for local_index, sample in enumerate(batch_samples):
                    for word_index, word in enumerate(PROMPT_WORDS):
                        candidate_rows.append(
                            {
                                "model": "correction",
                                "condition": condition,
                                "image": sample.image_name,
                                "image_index": sample.image_index,
                                "overlay_word": sample.overlay_word or "",
                                "template": template_name,
                                "word": word,
                                "score_correction_off": _as_float(logits_off[local_index, word_index]),
                                "score_correction_on": _as_float(logits_on[local_index, word_index]),
                                "score_delta": _as_float(
                                    logits_on[local_index, word_index]
                                    - logits_off[local_index, word_index]
                                ),
                            }
                        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        static_rows,
        candidate_rows,
        maps,
        register_masks,
        model_register_masks,
        patch_norm_maps,
        rgb_views,
        model_meta,
    )


# -------------------------------------------------------------------------------------------------
#  Full x-attention model
# -------------------------------------------------------------------------------------------------


def _content_logits_from_details(model: Any, details: dict[str, Any]) -> torch.Tensor:
    image = _normalize_rows(details["content_image_embedding"])
    text = _normalize_rows(details["content_text_embedding"])
    scale = model.logit_scale.detach().float().exp()
    return scale * (image @ text.t())


def _source_static_metrics(details: dict[str, Any], row: int) -> dict[str, float]:
    source_logits = details["source_logits"].detach().float()
    source_probs = source_logits.sigmoid()
    stats = details["source_stats"].detach().float()
    values = {
        "source_present_logit": _as_float(source_logits[row, 0]),
        "source_readable_logit": _as_float(source_logits[row, 1]),
        "source_present_prob": _as_float(source_probs[row, 0]),
        "source_readable_prob": _as_float(source_probs[row, 1]),
        "source_gate": _as_float(details["source_gate"][row]),
        "glyph_mean": _as_float(stats[row, 0]),
        "glyph_max": _as_float(stats[row, 1]),
        "glyph_top8": _as_float(stats[row, 2]),
        "glyph_soft_area": _as_float(stats[row, 3]),
        "glyph_row_max_mean": _as_float(stats[row, 4]),
        "glyph_col_max_mean": _as_float(stats[row, 5]),
        "glyph_row_density_max": _as_float(stats[row, 6]),
        "glyph_logit_mean": _as_float(stats[row, 7]),
    }
    return values


def _mixed_read_bridge_spatial_attention(
    read_component_details: dict[str, Any],
    bank: str = "patch_attention",
) -> torch.Tensor:
    """Mix one late read-bridge attention bank over learned taps and mean heads."""
    tap_weights = read_component_details["tap_weights"].detach().float()
    per_block = read_component_details["per_block"]
    blocks = list(per_block)
    first = per_block[blocks[0]][bank]
    if first is None:
        raise RuntimeError(f"Read bridge did not return {bank}")
    stacked = torch.stack(
        [per_block[block][bank].detach().float() for block in blocks],
        dim=0,
    )  # [K,B,N,H,T]
    mixed = torch.einsum("k,kbnht->bnht", tap_weights, stacked)
    spatial = mixed[:, :, :, 1:-1].mean(dim=2)  # drop CLS/RN, mean heads -> [B,N,P]
    patch_count = spatial.shape[-1]
    grid = int(round(math.sqrt(patch_count)))
    if grid * grid != patch_count:
        raise ValueError(
            f"Cannot reshape {patch_count} read-bridge spatial patches to a square grid"
        )
    return spatial.reshape(spatial.shape[0], spatial.shape[1], grid, grid)


def _run_full_model(
    model_path: str | Path,
    samples_by_condition: dict[str, list[hallucinations_Sample]],
    batch_size: int,
    device: torch.device,
    head_index: int,
):
    print(f"[full] Loading {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    static_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    h3_maps: dict[str, dict[str, np.ndarray]] = {}
    glyph_maps: dict[str, dict[str, np.ndarray]] = {}
    source_tap_maps: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    register_masks: dict[str, dict[str, np.ndarray]] = {}
    model_register_masks: dict[str, dict[str, np.ndarray]] = {}
    patch_norm_maps: dict[str, dict[str, np.ndarray]] = {}
    read_bridge_best: dict[str, dict[str, dict[str, Any]]] = {}
    rgb_views: dict[str, dict[str, np.ndarray]] = {}
    model_meta: dict[str, Any] = {
        "path": str(model_path),
        "class": type(model).__name__,
        "dtype": str(next(model.parameters()).dtype),
        "correction_head": head_index,
        "default_mode": str(getattr(model.config, "default_mode", "")),
        "correction_default": bool(getattr(model.config, "correction", True)),
        "register_norm_threshold": float(model.config.register_norm_threshold),
        "register_min": int(model.config.register_min),
        "register_max": int(model.config.register_max),
        "read_calibration_scale": _as_float(model.read_implant.read_calibration_scale),
        "null_abstain_weight": _as_float(model.read_implant.null_abstain_weight),
        "auto_read_scale": _as_float(model.read_implant.auto_read_scale),
        "glyph_bias_beta": _as_float(model.read_implant.glyph_bias_beta),
        "register_gate": _as_float(model.read_implant.read_bridge.register_gate),
        "source_tap_blocks": [
            int(value)
            for value in model.read_implant.source_tap_blocks.detach().cpu().tolist()
        ],
        "source_tap_weights": [
            _as_float(value)
            for value in model.read_implant.source_tap_logits.float().softmax(dim=0)
        ],
    }
    for block in model_meta["source_tap_blocks"]:
        source_tap_maps[int(block)] = {}

    for condition, samples in samples_by_condition.items():
        h3_maps[condition] = {}
        glyph_maps[condition] = {}
        register_masks[condition] = {}
        model_register_masks[condition] = {}
        patch_norm_maps[condition] = {}
        for block in source_tap_maps:
            source_tap_maps[block][condition] = {}
        read_bridge_best[condition] = {}
        rgb_views[condition] = {}

        for _, batch_samples in _batches(samples, batch_size):
            pixel_values = _pixel_values(processor, batch_samples, device)
            for sample, rgb in zip(batch_samples, _pixel_values_to_rgb(pixel_values, processor)):
                rgb_views[condition][sample.image_name] = rgb

            static_captured = False

            for template_name, template in PROMPT_TEMPLATES:
                input_ids = _text_ids(processor, template, device)
                with torch.inference_mode():
                    # mode='read' is the public <text> + exposed internal <null> condition.
                    read_output = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        mode="read",
                        correction=True,
                        return_details=True,
                    )
                    # mode='any' is the automatic robust mode (<any> semantics).
                    any_output = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        mode="any",
                        correction=True,
                        return_details=True,
                    )

                read_details = dict(read_output.details or {})
                any_details = dict(any_output.details or {})
                content_logits = _content_logits_from_details(model, any_details)
                null_index = read_output.null_candidate_index
                if null_index is None:
                    raise RuntimeError("mode='read' did not expose a null candidate")
                if int(null_index) != len(PROMPT_WORDS):
                    raise RuntimeError(
                        f"Expected null candidate at {len(PROMPT_WORDS)}, got {null_index}"
                    )
                read_component_details = read_details.get("read_details")
                if not read_component_details:
                    raise RuntimeError("Full read mode did not return read_details")
                read_bridge_spatial = _mixed_read_bridge_spatial_attention(
                    read_component_details, "patch_attention"
                )
                read_bridge_register = _mixed_read_bridge_spatial_attention(
                    read_component_details, "register_attention"
                )

                if not static_captured:
                    # Candidate-independent diagnostics: correction-pool H3 and early source/glyph head.
                    content_details = any_details.get("content_details")
                    if not content_details:
                        raise RuntimeError(
                            "Full model did not return content_details; correction=True is required"
                        )
                    blocks = [int(value) for value in model.read_implant.tap_blocks.detach().cpu().tolist()]
                    mixed_attention = torch.einsum(
                        "k,kbht->bht",
                        content_details["tap_weights"].float(),
                        torch.stack(
                            [
                                content_details["per_block_attention"][block].float()
                                for block in blocks
                            ],
                            dim=0,
                        ),
                    )
                    per_block = {
                        block: content_details["per_block_attention"][block].float()
                        for block in blocks
                    }
                    register_mask = any_details["register_mask"].detach().bool()
                    patch_norms = any_details["patch_token_norms"].detach().float()
                    threshold_mask = patch_norms > float(model.config.register_norm_threshold)
                    attention_stats = _attention_metrics(
                        mixed_attention,
                        per_block,
                        head_index,
                        register_mask=register_mask,
                        threshold_mask=threshold_mask,
                        patch_norms=patch_norms,
                    )
                    head_maps = _spatial_map(mixed_attention, head_index)
                    register_grids = _grid_map(threshold_mask.float())
                    model_register_grids = _grid_map(register_mask.float())
                    patch_norm_grids = _grid_map(patch_norms)

                    glyph = any_details["glyph_probs"].detach().float()
                    patch_count = glyph.shape[-1]
                    grid = int(round(math.sqrt(patch_count)))
                    if grid * grid != patch_count:
                        raise ValueError(
                            f"Cannot reshape {patch_count} source glyph patches to a square grid"
                        )
                    glyph_grid = glyph.reshape(glyph.shape[0], grid, grid)
                    source_details = any_details.get("source_details") or {}
                    source_tap_weights = source_details.get("tap_weights")
                    source_per_block = source_details.get("per_block_logits", {})
                    per_source_maps: dict[int, torch.Tensor] = {}
                    for source_block, source_logits_by_patch in source_per_block.items():
                        per_source_maps[int(source_block)] = _grid_map(
                            source_logits_by_patch.detach().float().sigmoid()
                        )
                    correction_norm = any_details["content_correction"].detach().float().norm(dim=-1)

                    for local_index, sample in enumerate(batch_samples):
                        row: dict[str, Any] = {
                            "model": "full",
                            "condition": condition,
                            "image": sample.image_name,
                            "image_index": sample.image_index,
                            "overlay_word": sample.overlay_word or "",
                            "content_correction_norm": _as_float(correction_norm[local_index]),
                        }
                        row.update(_source_static_metrics(any_details, local_index))
                        row.update(attention_stats[local_index])
                        for block, weight in zip(blocks, content_details["tap_weights"]):
                            row[f"tap_weight_b{block}"] = _as_float(weight)
                        if source_tap_weights is not None:
                            for source_block, weight in zip(
                                source_per_block.keys(), source_tap_weights
                            ):
                                row[f"source_tap_weight_b{int(source_block)}"] = _as_float(weight)
                        for source_block, source_logits_by_patch in source_per_block.items():
                            probs = source_logits_by_patch[local_index].detach().float().sigmoid()
                            row[f"source_b{int(source_block)}_glyph_mean"] = _as_float(probs.mean())
                            row[f"source_b{int(source_block)}_glyph_max"] = _as_float(probs.max())
                            row[f"source_b{int(source_block)}_glyph_top8"] = _as_float(
                                probs.topk(k=min(8, probs.numel())).values.mean()
                            )
                            row[f"source_b{int(source_block)}_glyph_soft_area"] = _as_float(
                                torch.sigmoid((probs - 0.50) * 12.0).mean()
                            )
                        static_rows.append(row)
                        h3_maps[condition][sample.image_name] = (
                            head_maps[local_index].detach().cpu().numpy()
                        )
                        glyph_maps[condition][sample.image_name] = (
                            glyph_grid[local_index].detach().cpu().numpy()
                        )
                        register_masks[condition][sample.image_name] = (
                            register_grids[local_index].detach().cpu().numpy()
                        )
                        model_register_masks[condition][sample.image_name] = (
                            model_register_grids[local_index].detach().cpu().numpy()
                        )
                        patch_norm_maps[condition][sample.image_name] = (
                            patch_norm_grids[local_index].detach().cpu().numpy()
                        )
                        for source_block, source_grid in per_source_maps.items():
                            source_tap_maps[source_block][condition][sample.image_name] = (
                                source_grid[local_index].detach().cpu().numpy()
                            )
                    static_captured = True

                read_word_scores = read_output.logits_per_image[:, : len(PROMPT_WORDS)].detach().float()
                null_scores = read_output.logits_per_image[:, int(null_index)].detach().float()
                any_scores = any_output.logits_per_image.detach().float()

                for local_index, sample in enumerate(batch_samples):
                    max_read_score, max_read_index = read_word_scores[local_index].max(dim=-1)
                    max_read_index_i = int(max_read_index)
                    null_score = null_scores[local_index]
                    read_margin = max_read_score - null_score
                    null_rank = 1 + int((read_word_scores[local_index] > null_score).sum().item())

                    previous = read_bridge_best[condition].get(sample.image_name)
                    margin_value = _as_float(read_margin)
                    if previous is None or margin_value > float(previous["read_minus_null"]):
                        read_bridge_best[condition][sample.image_name] = {
                            "map": read_bridge_spatial[
                                local_index, max_read_index_i
                            ].detach().cpu().numpy(),
                            "register_map": read_bridge_register[
                                local_index, max_read_index_i
                            ].detach().cpu().numpy(),
                            "read_minus_null": margin_value,
                            "word": PROMPT_WORDS[max_read_index_i],
                            "template": template_name,
                        }

                    max_any_score, max_any_index = any_scores[local_index].max(dim=-1)
                    max_any_index_i = int(max_any_index)
                    max_content_score, max_content_index = content_logits[local_index].max(dim=-1)
                    max_content_index_i = int(max_content_index)

                    contribution = any_details["auto_read_contribution"][local_index, : len(PROMPT_WORDS)].float()
                    relative = any_details["relative_read_logits"][local_index, : len(PROMPT_WORDS)].float()
                    route = any_details["route_gate"][local_index, : len(PROMPT_WORDS)].float()
                    trust = any_details["trust_gate"][local_index, : len(PROMPT_WORDS)].float()
                    rn_attention = any_details["read_null_attention"][local_index, : len(PROMPT_WORDS)].float()

                    max_contribution, max_contribution_index = contribution.max(dim=-1)
                    max_relative, max_relative_index = relative.max(dim=-1)
                    max_route, max_route_index = route.max(dim=-1)
                    max_trust, max_trust_index = trust.max(dim=-1)

                    internal_null = read_details["null_read_logits"][local_index].float()
                    overlay_index = (
                        PROMPT_WORDS.index(sample.overlay_word)
                        if sample.overlay_word in PROMPT_WORDS
                        else None
                    )

                    prompt_row: dict[str, Any] = {
                        "condition": condition,
                        "image": sample.image_name,
                        "image_index": sample.image_index,
                        "overlay_word": sample.overlay_word or "",
                        "template": template_name,
                        "read_top_word": PROMPT_WORDS[max_read_index_i],
                        "read_top_score": _as_float(max_read_score),
                        "null_score": _as_float(null_score),
                        "read_minus_null": _as_float(read_margin),
                        "null_rank": null_rank,
                        "null_wins": int(read_margin <= 0),
                        "exposed_null_internal_abs": _as_float((null_score - internal_null).abs()),
                        "read_top_rn_attention": _as_float(
                            read_details["read_null_attention"][
                                local_index, max_read_index_i
                            ]
                        ),
                        "read_top_relative": _as_float(
                            read_details["relative_read_logits"][
                                local_index, max_read_index_i
                            ]
                        ),
                        "read_top_trust": _as_float(
                            read_details["trust_gate"][local_index, max_read_index_i]
                        ),
                        "read_top_route": _as_float(
                            read_details["route_gate"][local_index, max_read_index_i]
                        ),
                        "any_top_word": PROMPT_WORDS[max_any_index_i],
                        "any_top_score": _as_float(max_any_score),
                        "content_top_word": PROMPT_WORDS[max_content_index_i],
                        "content_top_score": _as_float(max_content_score),
                        "any_top_changed_from_content": int(max_any_index_i != max_content_index_i),
                        "max_auto_contribution": _as_float(max_contribution),
                        "max_auto_contribution_word": PROMPT_WORDS[int(max_contribution_index)],
                        "mean_auto_contribution": _as_float(contribution.mean()),
                        "max_relative_read": _as_float(max_relative),
                        "max_relative_read_word": PROMPT_WORDS[int(max_relative_index)],
                        "max_route_gate": _as_float(max_route),
                        "max_route_gate_word": PROMPT_WORDS[int(max_route_index)],
                        "max_trust_gate": _as_float(max_trust),
                        "max_trust_gate_word": PROMPT_WORDS[int(max_trust_index)],
                        "mean_rn_attention": _as_float(rn_attention.mean()),
                    }
                    if overlay_index is not None:
                        prompt_row.update(
                            {
                                "overlay_read_score": _as_float(
                                    read_word_scores[local_index, overlay_index]
                                ),
                                "overlay_read_minus_null": _as_float(
                                    read_word_scores[local_index, overlay_index] - null_score
                                ),
                                "overlay_read_top1": int(max_read_index_i == overlay_index),
                                "overlay_any_score": _as_float(
                                    any_scores[local_index, overlay_index]
                                ),
                                "overlay_any_top1": int(max_any_index_i == overlay_index),
                            }
                        )
                    prompt_rows.append(prompt_row)

                    for word_index, word in enumerate(PROMPT_WORDS):
                        candidate_rows.append(
                            {
                                "model": "full",
                                "condition": condition,
                                "image": sample.image_name,
                                "image_index": sample.image_index,
                                "overlay_word": sample.overlay_word or "",
                                "template": template_name,
                                "word": word,
                                "read_score": _as_float(read_word_scores[local_index, word_index]),
                                "read_minus_null": _as_float(
                                    read_word_scores[local_index, word_index] - null_score
                                ),
                                "any_score": _as_float(any_scores[local_index, word_index]),
                                "content_score": _as_float(content_logits[local_index, word_index]),
                                "auto_read_contribution": _as_float(
                                    any_details["auto_read_contribution"][local_index, word_index]
                                ),
                                "raw_read_logit": _as_float(
                                    any_details["raw_read_logits"][local_index, word_index]
                                ),
                                "read_logit": _as_float(
                                    any_details["read_logits"][local_index, word_index]
                                ),
                                "relative_read_logit": _as_float(
                                    any_details["relative_read_logits"][local_index, word_index]
                                ),
                                "early_orthographic_logit": _as_float(
                                    any_details["early_orthographic_logits"][local_index, word_index]
                                ),
                                "trust_gate": _as_float(
                                    any_details["trust_gate"][local_index, word_index]
                                ),
                                "route_gate": _as_float(
                                    any_details["route_gate"][local_index, word_index]
                                ),
                                "rn_attention": _as_float(
                                    any_details["read_null_attention"][local_index, word_index]
                                ),
                            }
                        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        static_rows,
        prompt_rows,
        candidate_rows,
        h3_maps,
        glyph_maps,
        source_tap_maps,
        register_masks,
        model_register_masks,
        patch_norm_maps,
        read_bridge_best,
        rgb_views,
        model_meta,
    )


# -------------------------------------------------------------------------------------------------
#  Aggregation / correlations
# -------------------------------------------------------------------------------------------------


def _index_rows(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def _worst_clean_summary(
    full_prompt_rows: list[dict[str, Any]],
    full_static_rows: list[dict[str, Any]],
    correction_static_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    full_static = _index_rows(full_static_rows, "condition", "image")
    corr_static = _index_rows(correction_static_rows, "condition", "image")
    clean = [row for row in full_prompt_rows if row["condition"] == "clean"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in clean:
        grouped.setdefault(str(row["image"]), []).append(row)

    output: list[dict[str, Any]] = []
    for image, rows in sorted(grouped.items()):
        worst = max(rows, key=lambda row: float(row["read_minus_null"]))
        any_worst = max(rows, key=lambda row: float(row["max_auto_contribution"]))
        combined = dict(worst)
        combined["worst_any_template"] = any_worst["template"]
        combined["worst_any_max_auto_contribution"] = any_worst[
            "max_auto_contribution"
        ]
        combined["worst_any_max_auto_word"] = any_worst[
            "max_auto_contribution_word"
        ]
        combined["worst_any_max_relative_read"] = max(
            float(row["max_relative_read"]) for row in rows
        )
        combined["worst_any_max_route_gate"] = max(
            float(row["max_route_gate"]) for row in rows
        )
        combined["worst_any_max_trust_gate"] = max(
            float(row["max_trust_gate"]) for row in rows
        )
        fs = full_static[("clean", image)]
        cs = corr_static[("clean", image)]
        for key, value in fs.items():
            if key not in {"model", "condition", "image", "image_index", "overlay_word"}:
                combined[f"full_{key}"] = value
        for key, value in cs.items():
            if key not in {"model", "condition", "image", "image_index", "overlay_word"}:
                combined[f"corr_{key}"] = value
        output.append(combined)
    return output


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    target = np.asarray([float(row["read_minus_null"]) for row in rows], dtype=np.float64)
    predictors = [
        "full_source_present_prob",
        "full_source_readable_prob",
        "full_source_gate",
        "full_glyph_mean",
        "full_glyph_max",
        "full_glyph_top8",
        "full_glyph_soft_area",
        "full_h3_mass",
        "full_h3_max",
        "full_h3_top8",
        "full_h3_minus_other_mass",
        "full_h3_over_other_mass",
        "corr_h3_mass",
        "corr_h3_max",
        "corr_h3_top8",
        "corr_h3_minus_other_mass",
        "corr_h3_over_other_mass",
        "worst_any_max_auto_contribution",
        "worst_any_max_relative_read",
        "worst_any_max_route_gate",
        "worst_any_max_trust_gate",
        "read_top_rn_attention",
        "null_score",
    ]
    output: list[dict[str, Any]] = []
    target_rank = _rankdata(target)
    for key in predictors:
        if not all(key in row for row in rows):
            continue
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        output.append(
            {
                "target": "worst_clean_read_minus_null",
                "predictor": key,
                "n": len(values),
                "pearson_r": _corr(values, target),
                "spearman_rho": _corr(_rankdata(values), target_rank),
            }
        )
    output.sort(
        key=lambda row: -abs(float(row["spearman_rho"]))
        if math.isfinite(float(row["spearman_rho"]))
        else float("inf")
    )
    return output


def _source_bias_sweep(
    full_static_rows: list[dict[str, Any]],
    start: float = 0.0,
    stop: float = -8.0,
    step: float = -0.5,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Offline common bias sweep on both source logits; no model forward pass required."""
    if step == 0:
        raise ValueError("source bias sweep step cannot be zero")
    clean = [row for row in full_static_rows if row["condition"] == "clean"]
    text = [row for row in full_static_rows if row["condition"] == "text_control"]
    if not clean:
        return []

    def gate(row: dict[str, Any], bias: float) -> float:
        p = 1.0 / (1.0 + math.exp(-(float(row["source_present_logit"]) + bias)))
        r = 1.0 / (1.0 + math.exp(-(float(row["source_readable_logit"]) + bias)))
        return p * r

    values: list[float] = []
    current = float(start)
    if step < 0:
        while current >= stop - 1.0e-9:
            values.append(current)
            current += step
    else:
        while current <= stop + 1.0e-9:
            values.append(current)
            current += step

    rows: list[dict[str, Any]] = []
    for bias in values:
        clean_gates = np.asarray([gate(row, bias) for row in clean], dtype=np.float64)
        text_gates = np.asarray([gate(row, bias) for row in text], dtype=np.float64)
        row: dict[str, Any] = {
            "source_logit_bias": bias,
            "gate_threshold": threshold,
            "clean_n": len(clean),
            "clean_gate_mean": float(clean_gates.mean()),
            "clean_gate_max": float(clean_gates.max()),
            "clean_gate_p95": float(np.quantile(clean_gates, 0.95)),
            "clean_above_threshold": int((clean_gates >= threshold).sum()),
            "clean_above_threshold_fraction": float((clean_gates >= threshold).mean()),
        }
        if len(text_gates):
            row.update(
                {
                    "text_control_n": len(text),
                    "text_control_gate_mean": float(text_gates.mean()),
                    "text_control_gate_min": float(text_gates.min()),
                    "text_control_gate_p05": float(np.quantile(text_gates, 0.05)),
                    "text_control_above_threshold": int((text_gates >= threshold).sum()),
                    "text_control_above_threshold_fraction": float(
                        (text_gates >= threshold).mean()
                    ),
                    "separation_min_text_minus_max_clean": float(
                        text_gates.min() - clean_gates.max()
                    ),
                }
            )
        rows.append(row)
    return rows


# -------------------------------------------------------------------------------------------------
#  Plotting
# -------------------------------------------------------------------------------------------------


def _import_plotting():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import TwoSlopeNorm

    return plt, cm, TwoSlopeNorm


def _resize_map(values: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(values.astype(np.float32), mode="F")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def _rgba_overlay(
    rgb: np.ndarray,
    attention: np.ndarray,
    global_max: float,
    low_alpha: float,
    high_alpha: float,
    alpha_cutoff: float,
) -> np.ndarray:
    _, cm, _ = _import_plotting()
    height, width = rgb.shape[:2]
    resized = _resize_map(attention, width, height)
    normalized = np.clip(resized / max(global_max, 1.0e-12), 0.0, 1.0)
    rgba = cm.get_cmap("turbo")(normalized)
    # Preserve the previous "dark toe is transparent" convention: exact zero vanishes,
    # the <= cutoff toe ramps up to low_alpha, and salient values use high_alpha.
    toe = np.clip(normalized / max(alpha_cutoff, 1.0e-12), 0.0, 1.0)
    alpha = np.where(
        normalized <= alpha_cutoff,
        low_alpha * toe,
        high_alpha,
    )
    heat_rgb = rgba[..., :3]
    composite = rgb * (1.0 - alpha[..., None]) + heat_rgb * alpha[..., None]
    return np.clip(composite, 0.0, 1.0)


def _draw_patch_mask_boxes(
    composite: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Draw white/black patch boxes around flagged register positions."""
    if mask is None:
        return composite
    mask = np.asarray(mask) > 0.5
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D patch mask, got {mask.shape}")
    height, width = composite.shape[:2]
    rows, cols = mask.shape
    image = Image.fromarray((np.clip(composite, 0.0, 1.0) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for row, col in np.argwhere(mask):
        x0 = int(round(col * width / cols))
        x1 = int(round((col + 1) * width / cols)) - 1
        y0 = int(round(row * height / rows))
        y1 = int(round((row + 1) * height / rows)) - 1
        # Black outer line + white inner line stays visible over turbo and image content.
        draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0), width=3)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=(255, 255, 255), width=1)
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_attention_outputs(
    output_dir: Path,
    label: str,
    maps: dict[str, dict[str, np.ndarray]],
    rgb_views: dict[str, dict[str, np.ndarray]],
    static_rows: list[dict[str, Any]],
    metric_key: str,
    low_alpha: float,
    high_alpha: float,
    alpha_cutoff: float,
    extra_title_keys: tuple[str, ...] = (),
    patch_masks: dict[str, dict[str, np.ndarray]] | None = None,
) -> None:
    plt, _, _ = _import_plotting()
    all_values = [array for condition in maps.values() for array in condition.values()]
    if not all_values:
        return
    global_max = max(float(np.max(array)) for array in all_values)
    static_index = _index_rows(static_rows, "condition", "image")

    for condition, condition_maps in maps.items():
        names = sorted(condition_maps)
        individual_dir = output_dir / "overlays" / label / condition
        individual_dir.mkdir(parents=True, exist_ok=True)
        composites: list[tuple[str, np.ndarray, float, str]] = []
        for name in names:
            composite = _rgba_overlay(
                rgb_views[condition][name],
                condition_maps[name],
                global_max,
                low_alpha,
                high_alpha,
                alpha_cutoff,
            )
            if patch_masks is not None:
                composite = _draw_patch_mask_boxes(
                    composite,
                    patch_masks.get(condition, {}).get(name),
                )
            metric_row = static_index[(condition, name)]
            metric = float(metric_row.get(metric_key, float("nan")))
            extras = " | ".join(
                f"{key}={metric_row[key]}"
                for key in extra_title_keys
                if key in metric_row and str(metric_row[key])
            )
            Image.fromarray((composite * 255).astype(np.uint8)).save(
                individual_dir / f"{Path(name).stem}__{label}.png"
            )
            composites.append((name, composite, metric, extras))

        cols = 4
        rows = math.ceil(len(composites) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.2 * rows), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, (name, composite, metric, extras) in zip(axes.flat, composites):
            axis.imshow(composite)
            suffix = f"\n{extras}" if extras else ""
            axis.set_title(f"{name}\n{metric_key}={metric:.4g}{suffix}", fontsize=9)
            axis.axis("off")
        mask_note = " | white boxes = final norm-selected registers" if patch_masks is not None else ""
        fig.suptitle(
            f"{label} — {condition} | turbo, global max normalization={global_max:.4g}{mask_note}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output_dir / f"{label}__{condition}__gallery.png", dpi=170)
        plt.close(fig)


def _save_prompt_heatmap(
    output_dir: Path,
    prompt_rows: list[dict[str, Any]],
    condition: str,
    value_key: str,
    filename: str,
    title: str,
    center_zero: bool,
) -> None:
    plt, _, TwoSlopeNorm = _import_plotting()
    rows = [row for row in prompt_rows if row["condition"] == condition]
    if not rows:
        return
    images = sorted({str(row["image"]) for row in rows})
    template_names = [name for name, _ in PROMPT_TEMPLATES]
    lookup = {(str(row["image"]), str(row["template"])): float(row[value_key]) for row in rows}
    matrix = np.asarray(
        [[lookup[(image, template)] for template in template_names] for image in images],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(7.0, max(5.0, 0.42 * len(images) + 1.5)))
    kwargs: dict[str, Any] = {"aspect": "auto"}
    if center_zero and np.min(matrix) < 0 < np.max(matrix):
        kwargs["norm"] = TwoSlopeNorm(vcenter=0.0, vmin=float(np.min(matrix)), vmax=float(np.max(matrix)))
        kwargs["cmap"] = "coolwarm"
    image = ax.imshow(matrix, **kwargs)
    ax.set_xticks(np.arange(len(template_names)), template_names)
    ax.set_yticks(np.arange(len(images)), images)
    ax.set_title(title)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------
#  Console summary
# -------------------------------------------------------------------------------------------------


def _print_summary(
    prompt_rows: list[dict[str, Any]],
    worst_clean: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> None:
    clean = [row for row in prompt_rows if row["condition"] == "clean"]
    print()
    print("=" * 118)
    print("NO-TEXT HALLUCINATION SUMMARY")
    print("=" * 118)
    if clean:
        null_wins = sum(int(row["null_wins"]) for row in clean)
        print(
            f"<text>+<null>: null wins {null_wins}/{len(clean)} image-template cases "
            f"({null_wins / len(clean):.1%}); false-read cases={len(clean) - null_wins}."
        )
        for template_name, _ in PROMPT_TEMPLATES:
            subset = [row for row in clean if row["template"] == template_name]
            wins = sum(int(row["null_wins"]) for row in subset)
            print(
                f"  template={template_name:<7s} null wins {wins:2d}/{len(subset):2d} "
                f"({wins / len(subset):.1%}) | "
                f"mean max auto contribution={_safe_mean(float(row['max_auto_contribution']) for row in subset):+.3f}"
            )

    if worst_clean:
        print()
        print("Worst template per CLEAN image (positive read-null margin = hallucinated word beats <null>):")
        header = (
            f"{'image':28s} {'tmpl':7s} {'word':10s} {'read-null':>10s} {'null-rk':>7s} "
            f"{'auto+':>8s} {'srcP':>6s} {'srcR':>6s} {'H3full':>9s} {'H3corr':>9s} {'RNattn':>8s}"
        )
        print(header)
        print("-" * len(header))
        for row in sorted(worst_clean, key=lambda item: float(item["read_minus_null"]), reverse=True):
            print(
                f"{str(row['image'])[:28]:28s} "
                f"{str(row['template']):7s} "
                f"{str(row['read_top_word'])[:10]:10s} "
                f"{float(row['read_minus_null']):+10.3f} "
                f"{int(row['null_rank']):7d} "
                f"{float(row['worst_any_max_auto_contribution']):+8.3f} "
                f"{float(row['full_source_present_prob']):6.3f} "
                f"{float(row['full_source_readable_prob']):6.3f} "
                f"{float(row['full_h3_mass']):9.3f} "
                f"{float(row['corr_h3_mass']):9.3f} "
                f"{float(row['read_top_rn_attention']):8.4f}"
            )

    controls = [row for row in prompt_rows if row["condition"] == "text_control"]
    if controls:
        overlay_top1 = [row for row in controls if "overlay_read_top1" in row]
        print()
        if overlay_top1:
            read_acc = _safe_mean(float(row["overlay_read_top1"]) for row in overlay_top1)
            any_acc = _safe_mean(float(row["overlay_any_top1"]) for row in overlay_top1)
            print(
                f"Synthetic positive controls: overlay-word top-1 = {read_acc:.1%} in <text>+<null>, "
                f"{any_acc:.1%} in <any>."
            )

    if correlations:
        print()
        print("Exploratory correlations with worst CLEAN read-null margin (n is tiny; use as hints, not inference):")
        for row in correlations[:10]:
            print(
                f"  {row['predictor']:<38s} "
                f"Spearman={float(row['spearman_rho']):+7.3f}  "
                f"Pearson={float(row['pearson_r']):+7.3f}"
            )

    print()
    print("Register/H3 quick check (worst CLEAN rows):")
    for row in worst_clean:
        key = f"full_h3_norm60_fraction"
        arg_key = f"full_h3_argmax_is_norm60"
        if key in row:
            print(
                f"  {str(row['image']):<28s} "
                f"H3 norm>60-frac={float(row[key]):6.3f} "
                f"argmax>60={int(row.get(arg_key, 0))} "
                f">60={int(row.get('full_norm60_count', 0))} model-regs={int(row.get('full_register_count', 0))}"
            )


# -------------------------------------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------------------------------------


def hallucinations_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure false text-reading on hand-selected no-text images in the HF correction and "
            "full x-attention CLIP exports, with correction-pool H3 and source/glyph diagnostics."
        )
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--correction-model", type=str, default=DEFAULT_CORRECTION_MODEL,
                        help="HF repo id or local HF model directory")
    parser.add_argument("--full-model", type=str, default=DEFAULT_FULL_MODEL,
                        help="HF repo id or local HF model directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--correction-text-head", type=int, default=DEFAULT_CORRECTION_TEXT_HEAD)
    parser.add_argument(
        "--skip-text-control",
        action="store_true",
        help="Do not create paired positive controls with one prompt-pool word overlaid on each image.",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Optional TTF path for generated positive controls; defaults to Arial/DejaVu fallback.",
    )
    # Previous H3 overlay convention: turbo + dark toe kept mostly transparent.
    parser.add_argument("--low-alpha", type=float, default=0.20)
    parser.add_argument("--high-alpha", type=float, default=0.75)
    parser.add_argument("--alpha-cutoff", type=float, default=0.15)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 <= args.low_alpha <= 1.0:
        raise ValueError("--low-alpha must be in [0,1]")
    if not 0.0 <= args.high_alpha <= 1.0:
        raise ValueError("--high-alpha must be in [0,1]")
    if not 0.0 < args.alpha_cutoff <= 1.0:
        raise ValueError("--alpha-cutoff must be in (0,1]")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[probe] device={device}")
    print(f"[probe] images={args.images.resolve()}")
    print(f"[probe] output={output_dir}")
    print(f"[probe] prompt words={PROMPT_WORDS}")
    print(f"[probe] templates={[template for _, template in PROMPT_TEMPLATES]}")

    samples_by_condition = _build_samples(
        args.images,
        output_dir,
        add_text_control=not args.skip_text_control,
        font_path=args.font,
    )
    print(
        "[probe] conditions: "
        + ", ".join(f"{name}={len(samples)}" for name, samples in samples_by_condition.items())
    )

    (
        correction_static,
        correction_candidates,
        correction_h3_maps,
        correction_register_masks,
        correction_model_register_masks,
        correction_patch_norm_maps,
        correction_rgb,
        correction_meta,
    ) = _run_correction_model(
        args.correction_model,
        samples_by_condition,
        args.batch_size,
        device,
        args.correction_text_head,
    )

    (
        full_static,
        full_prompts,
        full_candidates,
        full_h3_maps,
        full_glyph_maps,
        full_source_tap_maps,
        full_register_masks,
        full_model_register_masks,
        full_patch_norm_maps,
        full_read_bridge_best,
        full_rgb,
        full_meta,
    ) = _run_full_model(
        args.full_model,
        samples_by_condition,
        args.batch_size,
        device,
        args.correction_text_head,
    )

    worst_clean = _worst_clean_summary(full_prompts, full_static, correction_static)
    correlations = _correlations(worst_clean)
    source_bias_sweep = _source_bias_sweep(full_static)

    _write_csv(output_dir / "correction_static_metrics.csv", correction_static)
    _write_csv(output_dir / "correction_candidate_scores.csv", correction_candidates)
    _write_csv(output_dir / "full_static_metrics.csv", full_static)
    _write_csv(output_dir / "full_prompt_summary.csv", full_prompts)
    _write_csv(output_dir / "full_candidate_scores.csv", full_candidates)
    _write_csv(output_dir / "clean_worst_case_summary.csv", worst_clean)
    _write_csv(output_dir / "clean_correlations.csv", correlations)
    _write_csv(output_dir / "source_logit_bias_sweep.csv", source_bias_sweep)
    (output_dir / "learned_scalars.json").write_text(
        json.dumps(
            {
                "read_calibration_scale": full_meta["read_calibration_scale"],
                "null_abstain_weight": full_meta["null_abstain_weight"],
                "auto_read_scale": full_meta["auto_read_scale"],
                "glyph_bias_beta": full_meta["glyph_bias_beta"],
                "register_gate": full_meta["register_gate"],
                "source_tap_blocks": full_meta["source_tap_blocks"],
                "source_tap_weights": full_meta["source_tap_weights"],
                "register_norm_threshold": full_meta["register_norm_threshold"],
                "register_min": full_meta["register_min"],
                "register_max": full_meta["register_max"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    full_read_bridge_maps: dict[str, dict[str, np.ndarray]] = {}
    full_read_bridge_register_maps: dict[str, dict[str, np.ndarray]] = {}
    full_read_bridge_rows: list[dict[str, Any]] = []
    for condition, entries in full_read_bridge_best.items():
        full_read_bridge_maps[condition] = {}
        full_read_bridge_register_maps[condition] = {}
        sample_lookup = {sample.image_name: sample for sample in samples_by_condition[condition]}
        for image_name, entry in entries.items():
            sample = sample_lookup[image_name]
            full_read_bridge_maps[condition][image_name] = entry["map"]
            full_read_bridge_register_maps[condition][image_name] = entry["register_map"]
            full_read_bridge_rows.append(
                {
                    "model": "full",
                    "condition": condition,
                    "image": image_name,
                    "image_index": sample.image_index,
                    "overlay_word": sample.overlay_word or "",
                    "read_minus_null": entry["read_minus_null"],
                    "word": entry["word"],
                    "template": entry["template"],
                }
            )
    _write_csv(output_dir / "full_read_bridge_worst_maps.csv", full_read_bridge_rows)

    metadata = {
        "images": str(args.images.resolve()),
        "image_count": len(samples_by_condition["clean"]),
        "conditions": list(samples_by_condition),
        "prompt_words": list(PROMPT_WORDS),
        "prompt_templates": {name: template for name, template in PROMPT_TEMPLATES},
        "correction_text_head": args.correction_text_head,
        "attention_overlay": {
            "cmap": "turbo",
            "normalization": "global max across clean + text-control maps for each probe",
            "low_alpha": args.low_alpha,
            "high_alpha": args.high_alpha,
            "alpha_cutoff": args.alpha_cutoff,
            "dark_toe": "alpha ramps 0 -> low_alpha through the cutoff",
        },
        "correction_model": correction_meta,
        "full_model": full_meta,
        "mode_mapping": {
            "<text>+<null>": "mode='read' (forced read candidates plus exposed internal null candidate)",
            "<any>": "mode='any' (automatic robust fusion)",
        },
        "notes": [
            "All models are loaded at their exported FP32 dtype; this probe does not use autocast.",
            "H3 means zero-based correction/content-pool head 3, mixed over the learned B20/B21 tap weights.",
            "H3/norm white boxes mark raw final-patch L2 > register_norm_threshold (60 by default); model register-bank selection is tracked separately.",
            "The read bridge patch bank explicitly excludes register-mask patches; register_attention is plotted separately.",
            "Per-source-tap glyph maps expose whether a clean false positive originates at a specific early source tap.",
            "source_logit_bias_sweep.csv is a post-hoc common bias sweep only; it does not modify or rerun the model.",
            "Correlation output is exploratory with only 16 clean images; no p-values are reported.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    _save_attention_outputs(
        output_dir,
        "correction_H3",
        correction_h3_maps,
        correction_rgb,
        correction_static,
        f"h{args.correction_text_head}_mass",
        args.low_alpha,
        args.high_alpha,
        args.alpha_cutoff,
        patch_masks=correction_register_masks,
    )
    _save_attention_outputs(
        output_dir,
        "correction_patch_norm",
        correction_patch_norm_maps,
        correction_rgb,
        correction_static,
        "patch_norm_max",
        args.low_alpha,
        args.high_alpha,
        args.alpha_cutoff,
        patch_masks=correction_register_masks,
    )
    _save_attention_outputs(
        output_dir,
        "full_H3",
        full_h3_maps,
        full_rgb,
        full_static,
        f"h{args.correction_text_head}_mass",
        args.low_alpha,
        args.high_alpha,
        args.alpha_cutoff,
        patch_masks=full_register_masks,
    )
    _save_attention_outputs(
        output_dir,
        "full_patch_norm",
        full_patch_norm_maps,
        full_rgb,
        full_static,
        "patch_norm_max",
        args.low_alpha,
        args.high_alpha,
        args.alpha_cutoff,
        patch_masks=full_register_masks,
    )
    _save_attention_outputs(
        output_dir,
        "full_source_glyph",
        full_glyph_maps,
        full_rgb,
        full_static,
        "glyph_top8",
        args.low_alpha,
        args.high_alpha,
        args.alpha_cutoff,
    )
    for source_block, source_maps in sorted(full_source_tap_maps.items()):
        _save_attention_outputs(
            output_dir,
            f"full_source_glyph_B{source_block}",
            source_maps,
            full_rgb,
            full_static,
            f"source_b{source_block}_glyph_top8",
            args.low_alpha,
            args.high_alpha,
            args.alpha_cutoff,
        )

    _save_attention_outputs(
        output_dir,
        "full_read_bridge_worst",
        full_read_bridge_maps,
        full_rgb,
        full_read_bridge_rows,
        "read_minus_null",
        args.low_alpha,
        args.high_alpha,
        args.alpha_cutoff,
        extra_title_keys=("word", "template"),
        patch_masks=full_model_register_masks,
    )
    _save_attention_outputs(
        output_dir,
        "full_read_bridge_register_worst",
        full_read_bridge_register_maps,
        full_rgb,
        full_read_bridge_rows,
        "read_minus_null",
        args.low_alpha,
        args.high_alpha,
        args.alpha_cutoff,
        extra_title_keys=("word", "template"),
        patch_masks=full_model_register_masks,
    )

    _save_prompt_heatmap(
        output_dir,
        full_prompts,
        "clean",
        "read_minus_null",
        "clean_read_minus_null_heatmap.png",
        "CLEAN: max word read logit - <null> logit (positive = false text read)",
        center_zero=True,
    )
    _save_prompt_heatmap(
        output_dir,
        full_prompts,
        "clean",
        "max_auto_contribution",
        "clean_any_auto_contribution_heatmap.png",
        "CLEAN <any>: maximum automatic read contribution across prompt pool",
        center_zero=False,
    )
    if "text_control" in samples_by_condition:
        _save_prompt_heatmap(
            output_dir,
            full_prompts,
            "text_control",
            "overlay_read_minus_null",
            "text_control_overlay_read_minus_null_heatmap.png",
            "TEXT CONTROL: overlaid word read logit - <null> logit",
            center_zero=True,
        )

    _print_summary(full_prompts, worst_clean, correlations)
    print()
    print(f"[probe] Wrote diagnostics to: {output_dir}")
    print("[probe] Primary files:")
    for name in (
        "clean_worst_case_summary.csv",
        "clean_correlations.csv",
        "source_logit_bias_sweep.csv",
        "learned_scalars.json",
        "full_prompt_summary.csv",
        "full_candidate_scores.csv",
        "correction_static_metrics.csv",
        "full_static_metrics.csv",
        "clean_read_minus_null_heatmap.png",
        "clean_any_auto_contribution_heatmap.png",
        "correction_H3__clean__gallery.png",
        "correction_patch_norm__clean__gallery.png",
        "full_H3__clean__gallery.png",
        "full_patch_norm__clean__gallery.png",
        "full_source_glyph__clean__gallery.png",
        "full_source_glyph_B*__clean__gallery.png",
        "full_read_bridge_worst__clean__gallery.png",
        "full_read_bridge_register_worst__clean__gallery.png",
    ):
        print(f"  - {name}")


# TAP TRANSPLANTS
import random
import types
from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm


DEFAULT_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
TAP_TRANSPLANTS_DEFAULT_OUT = Path("out_bench_testing/late_read_tap_transplants_misc_controls_BEFORE_wise_ft")
PROMPT = "a photo of a {}"


CONDITIONS: dict[str, dict[int, int]] = {
    "native_B20_B21": {20: 20},
}

EXTRA_CAPTURE_BLOCKS = (19, 22, 23)


@dataclass(frozen=True)
class tap_transplants_Sample:
    pair_id: str
    condition: str
    image_index: int
    image_name: str
    source_path: Path
    image: Image.Image
    candidate: str

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()

def late_tap_blocks(implant: Any) -> list[int]:
    """
    Return the trained late READ tap block indices across both model APIs.

    Current HF XAttnCLIPModel uses:
        implant._block_list(implant.tap_blocks)

    Older/OpenAI-style PIECES used:
        implant.tap_block_list()

    Prefer the current HF helper when present, then fall back to the public-ish
    buffer itself so this microscope does not depend on one naming generation.
    """
    if hasattr(implant, "_block_list"):
        return [int(x) for x in implant._block_list(implant.tap_blocks)]
    if hasattr(implant, "tap_block_list"):
        return [int(x) for x in implant.tap_block_list()]
    value = implant.tap_blocks.detach().cpu().reshape(-1).tolist()
    return [int(x) for x in value]

def late_tap_weights(implant: Any) -> torch.Tensor:
    """
    Match the model's learned late-tap mixing semantics when possible.
    Current HF uses _mix_weights(); older PIECES used softmax directly.
    """
    if hasattr(implant, "_mix_weights"):
        return implant._mix_weights(implant.read_tap_logits).detach().float().cpu()
    return implant.read_tap_logits.detach().float().softmax(dim=0).cpu()

def tap_transplants_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def _cpu_tree(value: Any):
    if torch.is_tensor(value):
        return value.detach().float().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(x) for x in value)
    return value

class LateReadSlotController:
    """
    Capture extra visual states, then substitute only the state seen by one
    trained READ slot. CONTENT/correction never sees the substitution.

    This targets the current HF HardTextReadImplant API exactly:
        read_features(states, text_query, register_mask, return_details=False)
    """

    def __init__(self, model: Any):
        self.model = model
        self.implant = model.read_implant
        self.swaps: dict[int, int] = {}
        self.capture_calls = False
        self.calls: list[dict[str, Any]] = []

        self._orig_capture = self.implant.capture_block_list
        self._orig_read = self.implant.read_features

    def install(self) -> None:
        original_capture = self._orig_capture
        extras = set(EXTRA_CAPTURE_BLOCKS)

        def capture_patched(implant_self):
            return sorted(set(original_capture()) | extras)

        self.implant.capture_block_list = types.MethodType(capture_patched, self.implant)

        controller = self
        original_read = self._orig_read

        def read_patched(
            implant_self,
            states,
            text_query,
            register_mask=None,
            return_details=False,
        ):
            proxy = dict(states)
            provenance = {}
            for slot, actual in controller.swaps.items():
                if actual not in states:
                    raise KeyError(
                        f"Late READ transplant needs captured B{actual}, "
                        f"available={sorted(states)}"
                    )
                proxy[int(slot)] = states[int(actual)]
                provenance[int(slot)] = int(actual)

            if not controller.capture_calls:
                return original_read(
                    proxy,
                    text_query,
                    register_mask=register_mask,
                    return_details=return_details,
                )

            # Force details for microscope capture while preserving caller semantics.
            mixed, details = original_read(
                proxy,
                text_query,
                register_mask=register_mask,
                return_details=True,
            )

            slots = late_tap_blocks(implant_self)
            state_snapshots = {
                slot: proxy[slot].detach().float().cpu()
                for slot in slots
            }
            controller.calls.append(
                {
                    "text_query": text_query.detach().float().cpu(),
                    "register_mask": (
                        None if register_mask is None else register_mask.detach().cpu()
                    ),
                    "states": state_snapshots,
                    "details": _cpu_tree(details),
                    "provenance": provenance,
                }
            )

            if return_details:
                return mixed, details
            return mixed

        self.implant.read_features = types.MethodType(read_patched, self.implant)

    def restore(self) -> None:
        self.implant.capture_block_list = self._orig_capture
        self.implant.read_features = self._orig_read

    def set_condition(self, swaps: Mapping[int, int]) -> None:
        self.swaps = {int(k): int(v) for k, v in swaps.items()}
        self.calls.clear()

    def begin_capture(self) -> None:
        self.calls.clear()
        self.capture_calls = True

    def end_capture(self) -> list[dict[str, Any]]:
        self.capture_calls = False
        return list(self.calls)

def _query_locations(calls: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Identify candidate and null query locations from the actual model calls.

    With one supplied candidate, current read mode normally evaluates a bank
    containing candidate + exposed null in one read_features call, then may
    evaluate the internal null baseline separately.  We avoid hard-coding that:
      candidate = query 0 of the first nonempty call;
      null      = last query of the first call if it has N>=2,
                  otherwise query 0 of the next call.
    """
    if not calls:
        raise RuntimeError("No read_features calls captured")
    first = next((i for i, c in enumerate(calls) if c["text_query"].shape[0] >= 1), None)
    if first is None:
        raise RuntimeError("Captured read_features calls contain no text queries")
    n = int(calls[first]["text_query"].shape[0])
    candidate = (first, 0)
    if n >= 2:
        null = (first, n - 1)
    else:
        second = next(
            (i for i in range(first + 1, len(calls)) if calls[i]["text_query"].shape[0] >= 1),
            None,
        )
        if second is None:
            raise RuntimeError(
                "Could not locate <null> query: only one one-query read_features call captured"
            )
        null = (second, 0)
    return candidate, null

def _block_patch_write_norm(
    model: Any,
    call: Mapping[str, Any],
    *,
    slot: int,
    query_index: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Exact ordinary patch-bank token contribution norm after read_bridge.out_proj,
    using the bridge's returned patch_attention.  Register-bank writes are not
    folded into this map; visual RN is returned separately.
    """
    details = call["details"]
    per_block = details["per_block"]
    if slot not in per_block:
        # JSON/cpu tree may preserve int keys, but tolerate str.
        block_details = per_block[str(slot)]
    else:
        block_details = per_block[slot]

    attn = block_details["patch_attention"].float()  # [B,N,H,T]
    if attn.ndim != 4:
        raise RuntimeError(f"Unexpected patch_attention shape {tuple(attn.shape)}")

    state = call["states"][slot].to(next(model.read_implant.read_bridge.parameters()).device)
    state = state.float()
    bridge = model.read_implant.read_bridge

    with torch.no_grad(), torch.autocast(device_type=state.device.type, enabled=False):
        normalized = bridge.vision_ln(state)
        v = bridge.v_proj(normalized)
        B, T, _ = v.shape
        H = int(attn.shape[2])
        if v.shape[-1] % H:
            raise RuntimeError(
                f"v width {v.shape[-1]} not divisible by attention heads {H}"
            )
        HD = v.shape[-1] // H
        v = v.view(B, T, H, HD).permute(0, 2, 1, 3)

        a = attn.to(v.device)[:, query_index:query_index + 1, :, :]
        token_head = a[..., None] * v[:, None, :, :, :]
        token_pre = token_head.permute(0, 1, 3, 2, 4).reshape(B, 1, T, H * HD)
        token_feature = F.linear(
            token_pre,
            bridge.out_proj.weight.float(),
            bias=None,
        )
        norms = token_feature.norm(dim=-1)[0, 0]  # [T]

    # Current late state: CLS + 256 spatial + RN.
    if norms.numel() < 3:
        raise RuntimeError("Late state is too short for CLS + spatial + RN")
    spatial = norms[1:-1].detach().cpu().numpy().astype(np.float32)
    rn_write = float(norms[-1].detach().cpu())

    rn_attn = block_details.get("read_null_attention")
    if rn_attn is None:
        per_head_rn = np.full((H,), np.nan, dtype=np.float32)
    else:
        # [B,N,H]
        per_head_rn = (
            rn_attn.float()[0, query_index].detach().cpu().numpy().astype(np.float32)
        )
    return spatial, rn_write, per_head_rn

def _mixed_rn_attention(call: Mapping[str, Any], query_index: int) -> float:
    value = call["details"].get("read_null_attention")
    if value is None:
        return float("nan")
    return float(value.float()[0, query_index])

def _grid(values: np.ndarray) -> np.ndarray:
    n = int(values.size)
    g = int(round(math.sqrt(n)))
    if g * g != n:
        raise RuntimeError(f"Patch count {n} is not square")
    return values.reshape(g, g)

def _overlay_map(ax, image: Image.Image, values: np.ndarray, title: str, vmax: float) -> None:
    arr = np.asarray(image.convert("RGB"))
    ax.imshow(arr)
    grid = _grid(values)
    h, w = arr.shape[:2]
    ph = h / grid.shape[0]
    pw = w / grid.shape[1]
    cmap = matplotlib.colormaps[plt.rcParams["image.cmap"]]
    denom = max(float(vmax), 1.0e-12)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            x = float(grid[r, c]) / denom
            if x <= 0:
                continue
            color = cmap(min(max(x, 0.0), 1.0))
            rect = Rectangle(
                (c * pw, r * ph),
                pw,
                ph,
                facecolor=color,
                edgecolor="none",
                alpha=min(0.80, 0.08 + 0.72 * x),
            )
            ax.add_patch(rect)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def _split_patch_overlay(
    ax,
    image: Image.Image,
    left_values: np.ndarray,
    right_values: np.ndarray,
    title: str,
    vmax: float,
) -> None:
    arr = np.asarray(image.convert("RGB"))
    ax.imshow(arr)
    left = _grid(left_values)
    right = _grid(right_values)
    if left.shape != right.shape:
        raise ValueError("split maps differ in shape")
    h, w = arr.shape[:2]
    ph = h / left.shape[0]
    pw = w / left.shape[1]
    cmap = matplotlib.colormaps[plt.rcParams["image.cmap"]]
    denom = max(float(vmax), 1.0e-12)

    for r in range(left.shape[0]):
        for c in range(left.shape[1]):
            for side, value in (("left", left[r, c]), ("right", right[r, c])):
                x = float(value) / denom
                if x <= 0:
                    continue
                color = cmap(min(max(x, 0.0), 1.0))
                x0 = c * pw + (0.0 if side == "left" else 0.5 * pw)
                rect = Rectangle(
                    (x0, r * ph),
                    0.5 * pw,
                    ph,
                    facecolor=color,
                    edgecolor="none",
                    alpha=min(0.82, 0.08 + 0.74 * x),
                )
                ax.add_patch(rect)

    ax.set_title(title, fontsize=9)
    ax.axis("off")


# =================================================================================================
# Local misc clean + deterministic added-text controls
# =================================================================================================

def tap_transplants__image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    return paths


def tap_transplants__find_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    return _find_control_font(font_path, size)


def tap_transplants__add_text_control(
    image: Image.Image,
    word: str,
    image_index: int,
    font_path: str | None,
) -> Image.Image:
    """
    Exact construction from probe_read_null_controls.py hallucinations:
    white text, black stroke, deterministic top/center/bottom location.
    """
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    width, height = output.size
    font_size = max(18, int(round(min(width, height) * 0.13)))
    font = tap_transplants__find_font(font_path, font_size)
    stroke = max(2, int(round(font_size * 0.055)))
    bbox = draw.textbbox((0, 0), word, font=font, stroke_width=stroke)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(4, (width - text_w) // 2)
    position = image_index % 3
    if position == 0:
        y = max(4, int(height * 0.08))
    elif position == 1:
        y = max(4, (height - text_h) // 2)
    else:
        y = max(4, int(height * 0.86) - text_h)
    draw.text(
        (x, y),
        word,
        font=font,
        fill=(255, 255, 255),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0),
    )
    return output


def build_samples(
    image_dir: Path,
    out_dir: Path,
    font_path: str | None,
) -> tuple[list[tap_transplants_Sample], list[tap_transplants_Sample]]:
    paths = tap_transplants__image_paths(image_dir)
    clean: list[tap_transplants_Sample] = []
    controls: list[tap_transplants_Sample] = []
    control_dir = out_dir / "text_controls"
    control_dir.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        word = PROMPT_WORDS[index % len(PROMPT_WORDS)]
        pair_id = f"{index:04d}_{path.stem}"

        clean.append(
            tap_transplants_Sample(
                pair_id=pair_id,
                condition="clean",
                image_index=index,
                image_name=path.name,
                source_path=path,
                image=image,
                candidate=word,
            )
        )

        controlled = tap_transplants__add_text_control(image, word, index, font_path)
        control_path = control_dir / f"{path.stem}__text_{word}.png"
        controlled.save(control_path)
        controls.append(
            tap_transplants_Sample(
                pair_id=pair_id,
                condition="text_control",
                image_index=index,
                image_name=path.name,
                source_path=path,
                image=controlled,
                candidate=word,
            )
        )

    print(f"[dataset] {len(clean)} clean + {len(controls)} text controls")
    return clean, controls


def preprocess_images(
    processor: Any,
    samples: Sequence[tap_transplants_Sample],
    device: torch.device,
) -> torch.Tensor:
    encoded = processor(
        images=[sample.image for sample in samples],
        return_tensors="pt",
    )
    return encoded["pixel_values"].to(device, non_blocking=device.type == "cuda")


def tap_transplants_encode_candidate(
    processor: Any,
    word: str,
    device: torch.device,
) -> torch.Tensor:
    encoded = processor(
        text=[PROMPT.format(word)],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device)


def batches(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


# =================================================================================================
# Forced READ/<null> scoring
# =================================================================================================

@torch.inference_mode()
def score_samples(
    model: Any,
    processor: Any,
    controller: LateReadSlotController,
    samples: Sequence[tap_transplants_Sample],
    *,
    swaps: Mapping[int, int],
    batch_size: int,
    device: torch.device,
    amp: bool,
    transplant_name: str,
) -> list[dict[str, Any]]:
    """
    Batch by candidate word so mode='read' returns exactly:
        column 0 = candidate
        column 1 = exposed <null>
    for every image in the batch.
    """
    controller.set_condition(swaps)
    rows: list[dict[str, Any]] = []

    by_word: dict[str, list[tap_transplants_Sample]] = {}
    for sample in samples:
        by_word.setdefault(sample.candidate, []).append(sample)

    for word in PROMPT_WORDS:
        group = by_word.get(word, [])
        if not group:
            continue
        ids = tap_transplants_encode_candidate(processor, word, device)

        for _, batch_samples in tqdm(
            list(batches(group, batch_size)),
            desc=f"{transplant_name} {batch_samples[0].condition if False else ''}{word}",
            leave=False,
        ):
            pixels = preprocess_images(processor, batch_samples, device)
            with amp_context(device, amp):
                output = model(
                    input_ids=ids,
                    pixel_values=pixels,
                    mode="read",
                    correction=True,
                    return_details=False,
                    pieces_fp32=True,
                )
            logits = output.logits_per_image.detach().float().cpu()
            if logits.shape != (len(batch_samples), 2):
                raise RuntimeError(
                    f"Expected candidate + <null> => {(len(batch_samples), 2)}, "
                    f"got {tuple(logits.shape)}"
                )

            for i, sample in enumerate(batch_samples):
                candidate_logit = float(logits[i, 0])
                null_logit = float(logits[i, 1])
                margin = candidate_logit - null_logit
                expected_read = sample.condition == "text_control"
                correct = margin > 0 if expected_read else margin < 0
                rows.append(
                    {
                        "transplant": transplant_name,
                        "pair_id": sample.pair_id,
                        "condition": sample.condition,
                        "image_index": sample.image_index,
                        "image": sample.image_name,
                        "source_path": str(sample.source_path),
                        "candidate": sample.candidate,
                        "candidate_logit": candidate_logit,
                        "null_logit": null_logit,
                        "candidate_minus_null": margin,
                        "predicted": "candidate" if margin > 0 else "<null>",
                        "expected": "candidate" if expected_read else "<null>",
                        "correct": bool(correct),
                    }
                )

            del pixels, output
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []

    transplants = list(CONDITIONS)
    for transplant in transplants:
        tr = [r for r in rows if r["transplant"] == transplant]

        for condition in ("clean", "text_control"):
            sub = [r for r in tr if r["condition"] == condition]
            margins = np.asarray([float(r["candidate_minus_null"]) for r in sub], dtype=np.float64)
            correct = np.asarray([bool(r["correct"]) for r in sub], dtype=bool)
            summary.append(
                {
                    "transplant": transplant,
                    "condition": condition,
                    "count": len(sub),
                    "accuracy": float(correct.mean()) if len(correct) else float("nan"),
                    "mean_candidate_minus_null": float(margins.mean()) if len(margins) else float("nan"),
                    "median_candidate_minus_null": float(np.median(margins)) if len(margins) else float("nan"),
                    "candidate_win_rate": float((margins > 0).mean()) if len(margins) else float("nan"),
                    "null_win_rate": float((margins < 0).mean()) if len(margins) else float("nan"),
                }
            )

        clean = {r["pair_id"]: r for r in tr if r["condition"] == "clean"}
        text = {r["pair_id"]: r for r in tr if r["condition"] == "text_control"}
        common = sorted(set(clean) & set(text))
        for pair_id in common:
            c = clean[pair_id]
            t = text[pair_id]
            paired.append(
                {
                    "transplant": transplant,
                    "pair_id": pair_id,
                    "image_index": c["image_index"],
                    "image": c["image"],
                    "candidate": c["candidate"],
                    "clean_margin": float(c["candidate_minus_null"]),
                    "text_margin": float(t["candidate_minus_null"]),
                    "text_minus_clean_margin": float(
                        t["candidate_minus_null"] - c["candidate_minus_null"]
                    ),
                    "clean_null_correct": bool(float(c["candidate_minus_null"]) < 0),
                    "text_read_correct": bool(float(t["candidate_minus_null"]) > 0),
                    "paired_success": bool(
                        float(c["candidate_minus_null"]) < 0
                        and float(t["candidate_minus_null"]) > 0
                    ),
                }
            )

        pp = [r for r in paired if r["transplant"] == transplant]
        if pp:
            summary.append(
                {
                    "transplant": transplant,
                    "condition": "PAIRED",
                    "count": len(pp),
                    "accuracy": float(np.mean([r["paired_success"] for r in pp])),
                    "mean_candidate_minus_null": float(np.mean([r["text_minus_clean_margin"] for r in pp])),
                    "median_candidate_minus_null": float(np.median([r["text_minus_clean_margin"] for r in pp])),
                    "candidate_win_rate": float(np.mean([r["text_read_correct"] for r in pp])),
                    "null_win_rate": float(np.mean([r["clean_null_correct"] for r in pp])),
                }
            )

    return summary, paired


def print_summary(summary: Sequence[Mapping[str, Any]]) -> None:
    print()
    print("=" * 112)
    print("LATE READ TAP TRANSPLANTS — LOCAL MISC CLEAN vs ADDED-TEXT CONTROLS")
    print("=" * 112)
    print(
        f"{'Transplant':<20} {'Condition':<14} {'N':>5} {'Correct':>9} "
        f"{'Mean cand-null':>15} {'Median':>12} {'Cand win':>10} {'Null win':>10}"
    )
    print("-" * 112)
    for r in summary:
        print(
            f"{r['transplant']:<20} {r['condition']:<14} {int(r['count']):>5d} "
            f"{r['accuracy']:>9.3%} "
            f"{r['mean_candidate_minus_null']:>+15.4f} "
            f"{r['median_candidate_minus_null']:>+12.4f} "
            f"{r['candidate_win_rate']:>10.3%} "
            f"{r['null_win_rate']:>10.3%}"
        )


def native_delta_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    native = {
        (r["pair_id"], r["condition"]): r
        for r in rows if r["transplant"] == "native_B20_B21"
    }
    out = []
    for r in rows:
        if r["transplant"] == "native_B20_B21":
            continue
        base = native[(r["pair_id"], r["condition"])]
        out.append(
            {
                **dict(r),
                "native_candidate_minus_null": float(base["candidate_minus_null"]),
                "delta_margin_vs_native": float(
                    r["candidate_minus_null"] - base["candidate_minus_null"]
                ),
                "native_correct": bool(base["correct"]),
                "correct_changed": bool(r["correct"]) != bool(base["correct"]),
            }
        )
    return out


# =================================================================================================
# Overlay capture: same exact READ bridge microscope, but on local clean/text pairs
# =================================================================================================

@torch.inference_mode()
def capture_one(
    model: Any,
    processor: Any,
    controller: LateReadSlotController,
    sample: tap_transplants_Sample,
    swaps: Mapping[int, int],
    *,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    ids = tap_transplants_encode_candidate(processor, sample.candidate, device)
    pixels = preprocess_images(processor, [sample], device)
    controller.set_condition(swaps)
    controller.begin_capture()
    try:
        with amp_context(device, amp):
            output = model(
                input_ids=ids,
                pixel_values=pixels,
                mode="read",
                correction=True,
                return_details=False,
                pieces_fp32=True,
            )
        logits = output.logits_per_image.detach().float().cpu()
    finally:
        calls = controller.end_capture()
    return logits, calls


def render_pair_overlay(
    *,
    model: Any,
    processor: Any,
    controller: LateReadSlotController,
    clean: tap_transplants_Sample,
    text_control: tap_transplants_Sample,
    transplant: str,
    swaps: Mapping[int, int],
    out_path: Path,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    if len(swaps) != 1:
        raise ValueError("Expected exactly one slot transplant")
    slot, actual = next(iter(swaps.items()))

    captures = {}
    for label, sample in (("clean", clean), ("text", text_control)):
        for which, use_swaps in (("native", {}), ("transplant", swaps)):
            logits, calls = capture_one(
                model, processor, controller, sample, use_swaps,
                device=device, amp=amp,
            )
            cand_loc, null_loc = _query_locations(calls)
            cand_call, cand_q = calls[cand_loc[0]], cand_loc[1]
            null_call, null_q = calls[null_loc[0]], null_loc[1]

            cand_map, cand_rn_write, _ = _block_patch_write_norm(
                model, cand_call, slot=slot, query_index=cand_q
            )
            null_map, null_rn_write, _ = _block_patch_write_norm(
                model, null_call, slot=slot, query_index=null_q
            )
            captures[(label, which)] = {
                "logits": logits,
                "candidate_map": cand_map,
                "null_map": null_map,
                "candidate_rn": _mixed_rn_attention(cand_call, cand_q),
                "null_rn": _mixed_rn_attention(null_call, null_q),
                "candidate_rn_write": cand_rn_write,
                "null_rn_write": null_rn_write,
            }

    fig, axes = plt.subplots(2, 5, figsize=(19, 8))
    for row_i, (label, sample) in enumerate((("clean", clean), ("text", text_control))):
        n = captures[(label, "native")]
        t = captures[(label, "transplant")]
        n_margin = float(n["logits"][0, 0] - n["logits"][0, 1])
        t_margin = float(t["logits"][0, 0] - t["logits"][0, 1])

        axes[row_i, 0].imshow(sample.image)
        axes[row_i, 0].axis("off")
        axes[row_i, 0].set_title(
            f"{label}: {sample.image_name}\n"
            f"candidate={sample.candidate!r}\n"
            f"cand-null {n_margin:+.3f} -> {t_margin:+.3f}",
            fontsize=9,
        )

        vmax_c = max(float(n["candidate_map"].max()), float(t["candidate_map"].max()), 1e-12)
        vmax_n = max(float(n["null_map"].max()), float(t["null_map"].max()), 1e-12)

        _overlay_map(
            axes[row_i, 1], sample.image, n["candidate_map"],
            f"native B{slot} candidate\nRN edge={n['candidate_rn']:.3f}",
            vmax_c,
        )
        _overlay_map(
            axes[row_i, 2], sample.image, t["candidate_map"],
            f"B{slot}<-B{actual} candidate\nRN edge={t['candidate_rn']:.3f}",
            vmax_c,
        )
        _split_patch_overlay(
            axes[row_i, 3], sample.image,
            n["candidate_map"], t["candidate_map"],
            "candidate split\nLEFT native | RIGHT transplant",
            vmax_c,
        )

        rel_native = n["candidate_map"] - n["null_map"]
        rel_trans = t["candidate_map"] - t["null_map"]
        delta = np.abs(rel_trans - rel_native)
        _overlay_map(
            axes[row_i, 4], sample.image, delta,
            "|Δ(candidate write - null write)|",
            max(float(delta.max()), 1e-12),
        )

    fig.suptitle(
        f"{transplant}: trained READ slot B{slot} fed B{actual}\n"
        f"same image pair: CLEAN versus deterministic ADDED TEXT",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    row = {
        "transplant": transplant,
        "pair_id": clean.pair_id,
        "image": clean.image_name,
        "candidate": clean.candidate,
        "slot_block": slot,
        "actual_block": actual,
    }
    for label in ("clean", "text"):
        for which in ("native", "transplant"):
            cap = captures[(label, which)]
            row[f"{label}_{which}_margin"] = float(cap["logits"][0, 0] - cap["logits"][0, 1])
            row[f"{label}_{which}_candidate_rn"] = cap["candidate_rn"]
            row[f"{label}_{which}_null_rn"] = cap["null_rn"]
            row[f"{label}_{which}_candidate_rn_write"] = cap["candidate_rn_write"]
            row[f"{label}_{which}_null_rn_write"] = cap["null_rn_write"]
    row["png"] = str(out_path)
    return row


def choose_overlay_pairs(
    paired_rows: Sequence[Mapping[str, Any]],
    transplant: str,
    count: int,
) -> list[str]:
    native = {
        r["pair_id"]: r
        for r in paired_rows
        if r["transplant"] == "native_B20_B21"
    }
    changed = []
    for r in paired_rows:
        if r["transplant"] != transplant:
            continue
        base = native[r["pair_id"]]
        # Rank by change in BOTH sides of the detector:
        # clean abstention margin + text positive margin.
        delta_clean = float(r["clean_margin"] - base["clean_margin"])
        delta_text = float(r["text_margin"] - base["text_margin"])
        score = abs(delta_clean) + abs(delta_text)
        changed.append((score, r["pair_id"]))
    changed.sort(reverse=True)
    return [pair_id for _, pair_id in changed[:count]]


# =================================================================================================
# Main
# =================================================================================================

def parity_check(
    model: Any,
    processor: Any,
    controller: LateReadSlotController,
    sample: tap_transplants_Sample,
    *,
    device: torch.device,
    amp: bool,
) -> None:
    ids = tap_transplants_encode_candidate(processor, sample.candidate, device)
    pixels = preprocess_images(processor, [sample], device)

    controller.set_condition({})
    with torch.inference_mode(), amp_context(device, amp):
        a = model(
            input_ids=ids,
            pixel_values=pixels,
            mode="read",
            correction=True,
            return_details=False,
            pieces_fp32=True,
        ).logits_per_image.detach().float()

    patched = controller.implant.read_features
    controller.implant.read_features = controller._orig_read
    try:
        with torch.inference_mode(), amp_context(device, amp):
            b = model(
                input_ids=ids,
                pixel_values=pixels,
                mode="read",
                correction=True,
                return_details=False,
                pieces_fp32=True,
            ).logits_per_image.detach().float()
    finally:
        controller.implant.read_features = patched

    delta = float((a - b).abs().max())
    print(f"[parity] native proxy vs unpatched READ max_abs={delta:.3e}")
    if delta > 5e-5:
        raise RuntimeError(f"Native proxy parity failed: {delta:.3e}")


def tap_transplants_main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF repo id or local HF model directory")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--out", type=Path, default=TAP_TRANSPLANTS_DEFAULT_OUT)
    p.add_argument("--font-path", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overlay-count", type=int, default=12, help="Per transplant condition.")
    p.add_argument("--no-overlays", action="store_true")
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[model] {args.model}")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    for param in model.parameters():
        param.requires_grad_(False)

    implant = model.read_implant
    native_taps = tuple(late_tap_blocks(implant))
    if native_taps != (20, 21):
        raise RuntimeError(f"Expected late READ taps (20,21), got {native_taps}")

    print(
        "[read taps] "
        + " ".join(
            f"B{b}={w:.6f}"
            for b, w in zip(native_taps, late_tap_weights(implant).tolist())
        )
    )
    print(f"[RN] insert before B{int(model.config.read_null_insert_block)}")
    print("[dataset policy] LOCAL image_sets/misc only; paired deterministic text controls")
    print("[metric] forced READ candidate vs <null>; <any> is intentionally not evaluated")

    clean, controls = build_samples(args.image_dir, args.out, args.font_path)
    all_samples = clean + controls
    sample_lookup = {(s.pair_id, s.condition): s for s in all_samples}

    controller = LateReadSlotController(model)
    controller.install()

    all_rows: list[dict[str, Any]] = []
    try:
        parity_check(
            model, processor, controller, clean[0],
            device=device, amp=args.amp,
        )

        for transplant, swaps in CONDITIONS.items():
            print()
            print("=" * 100)
            print(f"[condition] {transplant} swaps={swaps or 'native'}")
            print("=" * 100)

            all_rows.extend(
                score_samples(
                    model, processor, controller, clean,
                    swaps=swaps,
                    batch_size=args.batch_size,
                    device=device,
                    amp=args.amp,
                    transplant_name=transplant,
                )
            )
            all_rows.extend(
                score_samples(
                    model, processor, controller, controls,
                    swaps=swaps,
                    batch_size=args.batch_size,
                    device=device,
                    amp=args.amp,
                    transplant_name=transplant,
                )
            )

        summary, paired = summarize_rows(all_rows)
        deltas = native_delta_rows(all_rows)

        print_summary(summary)
        write_csv(args.out / "per_image_read_null.csv", all_rows)
        write_csv(args.out / "summary.csv", summary)
        write_csv(args.out / "paired_clean_text_controls.csv", paired)
        write_csv(args.out / "delta_vs_native.csv", deltas)

        overlay_rows = []
        if not args.no_overlays:
            for transplant, swaps in CONDITIONS.items():
                if not swaps:
                    continue
                chosen = choose_overlay_pairs(paired, transplant, args.overlay_count)
                print(f"[overlays] {transplant}: {len(chosen)} pairs")
                for rank, pair_id in enumerate(tqdm(chosen, desc=f"overlay {transplant}", leave=False)):
                    c = sample_lookup[(pair_id, "clean")]
                    t = sample_lookup[(pair_id, "text_control")]
                    path = args.out / "overlays" / transplant / f"{rank:02d}_{pair_id}.png"
                    overlay_rows.append(
                        render_pair_overlay(
                            model=model,
                            processor=processor,
                            controller=controller,
                            clean=c,
                            text_control=t,
                            transplant=transplant,
                            swaps=swaps,
                            out_path=path,
                            device=device,
                            amp=args.amp,
                        )
                    )
            write_csv(args.out / "overlay_summary.csv", overlay_rows)

        tap_transplants_write_json(
            args.out / "run_metadata.json",
            {
                "model": str(args.model),
                "image_dir": str(args.image_dir),
                "conditions": CONDITIONS,
                "native_read_taps": list(native_taps),
                "native_read_tap_weights": late_tap_weights(implant).tolist(),
                "prompt_words": list(PROMPT_WORDS),
                "prompt": PROMPT,
                "dataset": (
                    "image_sets/misc clean originals + deterministic added-text controls "
                    "from probe_read_null_controls.py hallucinations"
                ),
                "evaluation": "forced mode='read' candidate vs exposed <null> only",
                "note": (
                    "<any> intentionally omitted because semantic content recognition "
                    "is not a text-presence ground truth."
                ),
            },
        )

    finally:
        controller.restore()

    print()
    print(f"[done] {args.out}")


# DIAGNOSTIC
matplotlib.use("Agg")
from PIL import Image


DIAGNOSTIC_DEFAULT_OUT = Path("out_bench_testing/read_null_diagnostic_wise_ft_model")

DEFAULT_CASES = (
    "apple-blank.jpg::ipod::no_text",
    "apple-emptylabel-ipod.jpg::ipod::no_text",
    "apple-ipod.jpg::ipod::clean_text",
    "apple-dopi-mirror.jpg::ipod::mirror",
    "apple-emptylabel-doublemirror.jpg::ipod::mirror",
)


@dataclass(frozen=True)
class Case:
    image: Path
    candidate: str
    role: str


def diagnostic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def safe_slug(text: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return value.strip("._") or "item"


def parse_case(spec: str, image_dir: Path) -> Case:
    parts = spec.split("::")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(
            f"Case must be image::candidate[::role], got {spec!r}"
        )
    path = Path(parts[0])
    if not path.is_absolute():
        path = image_dir / path
    return Case(
        image=path,
        candidate=parts[1],
        role=parts[2] if len(parts) == 3 else "unspecified",
    )


def load_cases(args) -> list[Case]:
    cases: list[Case] = []

    if args.cases_json is not None:
        payload = json.loads(args.cases_json.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--cases-json must contain a JSON list")
        for row in payload:
            path = Path(str(row["image"]))
            if not path.is_absolute():
                path = args.image_dir / path
            cases.append(
                Case(
                    image=path,
                    candidate=str(row["candidate"]),
                    role=str(row.get("role", "unspecified")),
                )
            )

    for spec in args.case:
        cases.append(parse_case(spec, args.image_dir))

    # When no explicit cases are given, recover the historical local controls
    # without assuming they all live in exactly one image_sets subfolder.
    if not cases:
        print("[cases] no explicit cases supplied; resolving historical controls")
        search_dirs = [args.image_dir]
        special = Path("image_sets/special_natural")
        if special not in search_dirs:
            search_dirs.append(special)

        for spec in DEFAULT_CASES:
            name, candidate, role = spec.split("::")
            found = next((d / name for d in search_dirs if (d / name).is_file()), None)
            if found is not None:
                cases.append(Case(found, candidate, role))
            else:
                print(f"  [skip missing] {name}")

        # The famous orange-peel family has moved folders/names over time.
        # Auto-add any filename containing 'bananorange'; its diagnostic literal
        # candidate is intentionally 'orange'.  Do not guess candidates for other
        # arbitrary textures.
        if not args.no_auto_bananorange:
            seen_paths = {c.image.resolve() for c in cases if c.image.exists()}
            for directory in search_dirs:
                if not directory.is_dir():
                    continue
                for path in sorted(directory.iterdir()):
                    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                        continue
                    if "bananorange" not in path.stem.casefold():
                        continue
                    if path.resolve() in seen_paths:
                        continue
                    cases.append(Case(path, "orange", "orange_peel"))
                    seen_paths.add(path.resolve())

    if not cases:
        raise RuntimeError(
            "No diagnostic cases resolved. Supply e.g. "
            '--case "bananorange.jpg::orange::orange_peel"'
        )

    missing = [str(c.image) for c in cases if not c.image.is_file()]
    if missing:
        raise FileNotFoundError("Missing diagnostic images:\n  " + "\n  ".join(missing))

    # Exact de-dup by path/candidate/role, preserving order.
    deduped = []
    seen = set()
    for case in cases:
        key = (str(case.image.resolve()), case.candidate, case.role)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def preprocess_image(processor: Any, image: Image.Image, device: torch.device) -> torch.Tensor:
    encoded = processor(images=[image.convert("RGB")], return_tensors="pt")
    return encoded["pixel_values"].to(device)


def diagnostic_encode_candidate(processor: Any, candidate: str, device: torch.device) -> torch.Tensor:
    encoded = processor(
        text=[PROMPT.format(candidate)],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device)


class ReadSpy:
    """Capture native SOURCE and read_features details without changing outputs."""

    def __init__(self, model: Any):
        self.model = model
        self.implant = model.read_implant
        self._orig_read = self.implant.read_features
        self._orig_source = self.implant.source_outputs
        self.calls: list[dict[str, Any]] = []
        self.source_logits: torch.Tensor | None = None
        self.source_stats: torch.Tensor | None = None
        self.enabled = False

    def install(self) -> None:
        spy = self
        orig_read = self._orig_read
        orig_source = self._orig_source

        def source_patched(implant_self, states, return_details=False):
            result = orig_source(states, return_details=return_details)
            if spy.enabled:
                spy.source_logits = result[0].detach().float().cpu()
                spy.source_stats = result[2].detach().float().cpu()
            return result

        def read_patched(
            implant_self,
            states,
            text_query,
            register_mask=None,
            return_details=False,
        ):
            if not spy.enabled:
                return orig_read(
                    states,
                    text_query,
                    register_mask=register_mask,
                    return_details=return_details,
                )

            mixed, details = orig_read(
                states,
                text_query,
                register_mask=register_mask,
                return_details=True,
            )

            spy.calls.append(
                {
                    "text_query_shape": list(text_query.shape),
                    "details": _cpu_tree(details),
                }
            )

            if return_details:
                return mixed, details
            return mixed

        self.implant.source_outputs = types.MethodType(source_patched, self.implant)
        self.implant.read_features = types.MethodType(read_patched, self.implant)

    def restore(self) -> None:
        self.implant.source_outputs = self._orig_source
        self.implant.read_features = self._orig_read

    def begin(self) -> None:
        self.calls.clear()
        self.source_logits = None
        self.source_stats = None
        self.enabled = True

    def end(self) -> None:
        self.enabled = False


def query_locations(calls: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int], tuple[int, int]]:
    if not calls:
        raise RuntimeError("No read_features calls captured")
    first = next(
        (i for i, c in enumerate(calls) if int(c["text_query_shape"][0]) >= 1),
        None,
    )
    if first is None:
        raise RuntimeError("No nonempty read query call captured")
    n = int(calls[first]["text_query_shape"][0])
    candidate = (first, 0)
    if n >= 2:
        null = (first, n - 1)
    else:
        second = next(
            (
                i for i in range(first + 1, len(calls))
                if int(calls[i]["text_query_shape"][0]) >= 1
            ),
            None,
        )
        if second is None:
            raise RuntimeError("Could not locate a <null> read query call")
        null = (second, 0)
    return candidate, null


def mixed_rn_attention(call: Mapping[str, Any], q_index: int) -> float:
    value = call["details"].get("read_null_attention")
    if value is None:
        return float("nan")
    return float(value.float()[0, q_index])


def per_block_rn_attention(call: Mapping[str, Any], q_index: int) -> dict[int, float]:
    result = {}
    for block, details in call["details"]["per_block"].items():
        value = details.get("read_null_attention")
        if value is None:
            result[int(block)] = float("nan")
        else:
            # per block [B,N,H]; report mean head edge weight
            result[int(block)] = float(value.float()[0, q_index].mean())
    return result


class IdentityCalibration:
    def __init__(self, implant: Any):
        self.implant = implant
        self.original = implant.calibrate_read_logits

    def __enter__(self):
        def identity(implant_self, raw_read_logits, source_logits, null_mask=None):
            return raw_read_logits
        self.implant.calibrate_read_logits = types.MethodType(identity, self.implant)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.implant.calibrate_read_logits = self.original


@torch.inference_mode()
def run_forward(
    model: Any,
    ids: torch.Tensor,
    pixels: torch.Tensor,
    *,
    device: torch.device,
    amp: bool,
):
    with amp_context(device, amp):
        output = model(
            input_ids=ids,
            pixel_values=pixels,
            mode="read",
            correction=True,
            return_details=False,
            pieces_fp32=True,
        )
    logits = output.logits_per_image.detach().float().cpu()
    if logits.shape != (1, 2):
        raise RuntimeError(
            "Expected one candidate + exposed <null> => logits shape (1,2); "
            f"got {tuple(logits.shape)}"
        )
    return logits


def plot_case(row: Mapping[str, Any], image: Image.Image, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image.convert("RGB"))
    axes[0].axis("off")
    axes[0].set_title(
        f"{row['role']}\n{Path(row['image']).name}\ncandidate={row['candidate']!r}",
        fontsize=9,
    )

    labels = ["candidate", "<null>"]
    x = np.arange(2)
    width = 0.36
    axes[1].bar(
        x - width / 2,
        [row["raw_candidate"], row["raw_null"]],
        width,
        label="raw",
    )
    axes[1].bar(
        x + width / 2,
        [row["calibrated_candidate"], row["calibrated_null"]],
        width,
        label="calibrated",
    )
    axes[1].set_xticks(x, labels)
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].legend()
    axes[1].set_title(
        f"margin raw {row['raw_margin']:+.3f}\n"
        f"cal {row['calibrated_margin']:+.3f}",
        fontsize=9,
    )

    labels2 = ["RN cand", "RN null", "SOURCE readable", "SOURCE present", "SOURCE gate"]
    vals = [
        row["rn_attention_candidate"],
        row["rn_attention_null"],
        row["source_readable_probability"],
        row["source_present_probability"],
        row["source_gate"],
    ]
    axes[2].bar(np.arange(len(vals)), vals)
    axes[2].set_xticks(np.arange(len(vals)), labels2, rotation=30, ha="right")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title(
        f"null bonus={row['null_bonus']:+.3f}\n"
        f"scale={row['read_calibration_scale']:.3f}",
        fontsize=9,
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def diagnostic_main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF repo id or local HF model directory")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--out", type=Path, default=DIAGNOSTIC_DEFAULT_OUT)
    p.add_argument("--case", action="append", default=[],
                   help='Repeatable image::candidate[::role] spec.')
    p.add_argument("--cases-json", type=Path, default=None)
    p.add_argument("--no-auto-bananorange", action="store_true",
                   help="Disable automatic orange-peel case discovery when no explicit cases are supplied.")
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[model] {args.model}")
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    for param in model.parameters():
        param.requires_grad_(False)

    cases = load_cases(args)
    print("[cases]")
    for c in cases:
        print(f"  {c.role:14s} {c.image} :: {c.candidate}")

    implant = model.read_implant
    print(
        "[implant API] "
        f"_block_list={hasattr(implant, '_block_list')} "
        f"_mix_weights={hasattr(implant, '_mix_weights')} "
        f"capture_block_list={hasattr(implant, 'capture_block_list')} "
        f"tap_block_list={hasattr(implant, 'tap_block_list')} "
        "read_features_api=current_hf"
    )
    scale = float(implant.read_calibration_scale.detach().float().cpu())
    abstain = float(implant.null_abstain_weight.detach().float().cpu())
    tap_blocks = late_tap_blocks(implant)
    tap_weights = late_tap_weights(implant).tolist()

    print(
        "[late READ] "
        + " ".join(f"B{b}={w:.6f}" for b, w in zip(tap_blocks, tap_weights))
    )
    print(f"[calibration] scale={scale:.6f} null_abstain_weight={abstain:.6f}")

    spy = ReadSpy(model)
    spy.install()
    rows = []

    try:
        for case in cases:
            image = Image.open(case.image).convert("RGB")
            pixels = preprocess_image(processor, image, device)
            ids = diagnostic_encode_candidate(processor, case.candidate, device)

            # Native calibrated forward + exact RN/source capture.
            spy.begin()
            calibrated = run_forward(
                model, ids, pixels, device=device, amp=args.amp
            )
            spy.end()

            cand_loc, null_loc = query_locations(spy.calls)
            cand_call = spy.calls[cand_loc[0]]
            null_call = spy.calls[null_loc[0]]
            rn_cand = mixed_rn_attention(cand_call, cand_loc[1])
            rn_null = mixed_rn_attention(null_call, null_loc[1])
            rn_cand_blocks = per_block_rn_attention(cand_call, cand_loc[1])
            rn_null_blocks = per_block_rn_attention(null_call, null_loc[1])

            if spy.source_logits is None:
                raise RuntimeError("SOURCE logits were not captured")
            source_logits = spy.source_logits[0]
            source_present = float(torch.sigmoid(source_logits[0]))
            source_readable = float(torch.sigmoid(source_logits[1]))
            source_gate = source_present * source_readable

            # Same forward, but expose PRE-calibration raw reader logits.
            with IdentityCalibration(implant):
                raw = run_forward(
                    model, ids, pixels, device=device, amp=args.amp
                )

            raw_candidate = float(raw[0, 0])
            raw_null = float(raw[0, 1])
            cal_candidate = float(calibrated[0, 0])
            cal_null = float(calibrated[0, 1])

            scaled_raw_candidate = scale * raw_candidate
            scaled_raw_null = scale * raw_null
            null_bonus = cal_null - scaled_raw_null
            candidate_cal_parity = cal_candidate - scaled_raw_candidate

            row = {
                "role": case.role,
                "image": str(case.image),
                "candidate": case.candidate,
                "raw_candidate": raw_candidate,
                "raw_null": raw_null,
                "raw_margin": raw_candidate - raw_null,
                "calibrated_candidate": cal_candidate,
                "calibrated_null": cal_null,
                "calibrated_margin": cal_candidate - cal_null,
                "read_calibration_scale": scale,
                "null_abstain_weight": abstain,
                "scaled_raw_candidate": scaled_raw_candidate,
                "scaled_raw_null": scaled_raw_null,
                "null_bonus": null_bonus,
                "candidate_calibration_parity_error": candidate_cal_parity,
                "source_present_logit": float(source_logits[0]),
                "source_readable_logit": float(source_logits[1]),
                "source_present_probability": source_present,
                "source_readable_probability": source_readable,
                "source_gate": source_gate,
                "rn_attention_candidate": rn_cand,
                "rn_attention_null": rn_null,
            }
            for block in tap_blocks:
                row[f"B{block}_rn_attention_candidate"] = rn_cand_blocks.get(block, float("nan"))
                row[f"B{block}_rn_attention_null"] = rn_null_blocks.get(block, float("nan"))

            rows.append(row)

            print()
            print(f"[{case.role}] {case.image.name} :: {case.candidate!r}")
            print(
                f"  RAW        candidate={raw_candidate:+10.5f} "
                f"null={raw_null:+10.5f} margin={raw_candidate-raw_null:+10.5f}"
            )
            print(
                f"  CALIBRATED candidate={cal_candidate:+10.5f} "
                f"null={cal_null:+10.5f} margin={cal_candidate-cal_null:+10.5f}"
            )
            print(
                f"  RN edge    candidate={rn_cand:8.4f} null={rn_null:8.4f} | "
                + " ".join(
                    f"B{b} cand/null={rn_cand_blocks.get(b,float('nan')):.4f}/"
                    f"{rn_null_blocks.get(b,float('nan')):.4f}"
                    for b in tap_blocks
                )
            )
            print(
                f"  SOURCE     present={source_present:.4f} "
                f"readable={source_readable:.4f} gate={source_gate:.4f}"
            )
            print(
                f"  NULL CAL   scale*raw_null={scaled_raw_null:+.5f} "
                f"bonus={null_bonus:+.5f} | "
                f"candidate parity err={candidate_cal_parity:+.3e}"
            )

            plot_case(
                row,
                image,
                args.out / "plots" / f"{len(rows)-1:02d}_{safe_slug(case.image.stem)}_{safe_slug(case.candidate)}.png",
            )

    finally:
        spy.restore()

    write_csv(args.out / "read_null_diagnostic.csv", rows)
    diagnostic_write_json(
        args.out / "read_null_diagnostic.json",
        {
            "model": str(args.model),
            "prompt": PROMPT,
            "read_tap_blocks": tap_blocks,
            "read_tap_weights": tap_weights,
            "read_calibration_scale": scale,
            "null_abstain_weight": abstain,
            "rows": rows,
        },
    )

    print()
    print("=" * 123)
    print("READ / NULL DIAGNOSTIC SUMMARY")
    print("=" * 123)
    print(
        f"{'role':<15} {'image':<32} {'candidate':<14} "
        f"{'raw Δ':>9} {'cal Δ':>9} {'RN cand':>8} {'RN null':>8} "
        f"{'src read':>8} {'null bonus':>10}"
    )
    print("-" * 123)
    for r in rows:
        print(
            f"{r['role']:<15} {Path(r['image']).name:<32.32s} {r['candidate']:<14.14s} "
            f"{r['raw_margin']:>+9.3f} {r['calibrated_margin']:>+9.3f} "
            f"{r['rn_attention_candidate']:>8.3f} {r['rn_attention_null']:>8.3f} "
            f"{r['source_readable_probability']:>8.3f} {r['null_bonus']:>+10.3f}"
        )

    print()
    print(f"[done] {args.out}")
    print("[reminder] inference only; no model weights changed or saved.")


def main(argv=None):
    """Dispatch a workflow; each subcommand retains its original CLI options."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {'hallucinations': hallucinations_main, 'tap_transplants': tap_transplants_main, 'diagnostic': diagnostic_main}
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("\nCommands: " + ", ".join(commands))
        print("Use: python " + __file__ + " COMMAND --help")
        return
    command = argv.pop(0)
    if command not in commands:
        raise SystemExit("Unknown command: " + command)
    previous = sys.argv
    sys.argv = [previous[0] + " " + command, *argv]
    try:
        return commands[command]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()

