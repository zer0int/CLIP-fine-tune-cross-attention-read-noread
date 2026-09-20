#!/usr/bin/env python3
r"""Locate where source-PIECE text evidence moves after RN insertion.

Purpose
-------
For a chosen post-RN target block (default zero-based B13), compare the frozen source
PIECE spatial map with and without the RN token:

    shared trunk:  embeddings -> B0..B12

    NO_RN branch:  B13..B_target
    RN branch:     append learned RN before B13, then B13..B_target

The script is aimed at the concrete microscope question:
    "Where did the text go?"

So it produces:
1) per-image quantitative relocation metrics
2) per-image overlay panels
3) compact galleries for the most interesting examples
4) top-k patch coordinates and movement summaries

Default selection criterion:
    largest AP drop from NO_RN -> RN at the target block,
    separately for handwritten and digital subsets.

Notes
-----
- Blocks are zero-based.
- The source head owns CLS removal internally, so we preserve CLS when calling
  source_head.patch_logits(); for RN states we remove only the final RN token.
- Diff masks are aligned to the actual 224x224 ViT input seen by the model.
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_analysis import (PairSample, _autocast, _batches, _family, _modality, _pair_key, _safe_mean, _safe_median, configure_reproducibility, load_samples, preprocess_images)


import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


# ================================================================================================
# Configuration
# ================================================================================================

DEFAULT_FULL_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_OUTPUT_DIR = Path("out_bench_testing/rn_text_relocation_overlays")

SUBSET_ORDER = (
    "NoSCAM",
    "SCAM",
    "SynthSCAM",
    "NoRTA",
    "RTA",
    "SynthRTA",
)
ATTACK_SUBSETS = ("SCAM", "SynthSCAM", "RTA", "SynthRTA")
HANDWRITTEN_SUBSETS = ("SCAM", "RTA")
DIGITAL_SUBSETS = ("SynthSCAM", "SynthRTA")

DEFAULT_REFERENCE_BLOCK = 10
DEFAULT_TARGET_BLOCK = 13
EXPECTED_RN_INSERT_BLOCK = 13

DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 20260829
DEFAULT_TOPK = 8
DEFAULT_TOP_N_PER_MODALITY = 8

DEFAULT_DIFF_THRESHOLDS = (4.0, 8.0, 12.0, 16.0)
DEFAULT_PRIMARY_DIFF_THRESHOLD = 12.0
DEFAULT_PATCH_PIXEL_FRACTION = 0.01


# ================================================================================================
# Utilities
# ================================================================================================


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "subset",
        "modality",
        "family",
        "pair_key",
        "id",
        "correct_label",
        "distractor_label",
    ]
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ================================================================================================
# Dataset
# ================================================================================================


def paired_attacks(
    sample_sets: dict[str, list[PairSample]],
) -> list[tuple[PairSample, PairSample]]:
    indexed: dict[tuple[str, str, str], PairSample] = {}
    for subset in SUBSET_ORDER:
        for sample in sample_sets[subset]:
            indexed[(_family(subset), _pair_key(sample.sample_id), subset)] = sample

    pairs: list[tuple[PairSample, PairSample]] = []
    for family in ("SCAM", "RTA"):
        no_subset = f"No{family}"
        for attacked_subset in (family, f"Synth{family}"):
            keys = sorted(
                {
                    key
                    for fam, key, subset in indexed
                    if fam == family and subset == attacked_subset
                }
            )
            for key in keys:
                base = indexed.get((family, key, no_subset))
                attacked = indexed.get((family, key, attacked_subset))
                if base is not None and attacked is not None:
                    pairs.append((base, attacked))
    return pairs


# ================================================================================================
# Preprocessing / diff masks
# ================================================================================================


def _undo_clip_normalization(pixel_values: torch.Tensor, processor: Any) -> torch.Tensor:
    image_processor = getattr(processor, "image_processor", processor)

    mean = torch.tensor(
        getattr(image_processor, "image_mean", (0.48145466, 0.4578275, 0.40821073)),
        dtype=torch.float32,
        device=pixel_values.device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        getattr(image_processor, "image_std", (0.26862954, 0.26130258, 0.27577711)),
        dtype=torch.float32,
        device=pixel_values.device,
    ).view(1, 3, 1, 1)

    rgb01 = pixel_values.detach().float() * std + mean
    return (rgb01.clamp(0.0, 1.0) * 255.0).float()


def _patch_fraction_map(mask: np.ndarray, grid: int) -> np.ndarray:
    height, width = mask.shape
    if height % grid != 0 or width % grid != 0:
        raise ValueError(f"Processed image {width}x{height} not divisible by grid={grid}")

    patch_h = height // grid
    patch_w = width // grid
    return (
        mask.reshape(grid, patch_h, grid, patch_w)
        .transpose(0, 2, 1, 3)
        .mean(axis=(2, 3))
    )


@torch.inference_mode()
def collect_diff_masks(
    processor: Any,
    pairs: Sequence[tuple[PairSample, PairSample]],
    *,
    patch_count: int,
    thresholds: Sequence[float],
    primary_threshold: float,
    patch_pixel_fraction: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], list[np.ndarray]]:
    grid = int(round(math.sqrt(patch_count)))
    if grid * grid != patch_count:
        raise ValueError(f"{patch_count} patches are not a square grid")

    rows: list[dict[str, Any]] = []
    patch_masks: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    rgb_images: list[np.ndarray] = []

    print()
    print("[diff] Building paired attack-text masks ...")

    for base, attacked in tqdm(pairs, desc="paired diff", leave=False):
        pixel_values = preprocess_images(processor, (base, attacked), device)
        rgb255 = _undo_clip_normalization(pixel_values, processor).cpu().numpy()

        diff = np.max(np.abs(rgb255[1] - rgb255[0]), axis=0)
        attacked_rgb = np.transpose(rgb255[1], (1, 2, 0)).astype(np.uint8)

        primary_pixel_mask = diff > float(primary_threshold)
        primary_patch_fraction = _patch_fraction_map(primary_pixel_mask.astype(np.float32), grid)
        covered = primary_patch_fraction >= float(patch_pixel_fraction)

        row: dict[str, Any] = {
            "subset": attacked.subset,
            "modality": _modality(attacked.subset),
            "family": _family(attacked.subset),
            "pair_key": _pair_key(attacked.sample_id),
            "base_id": base.sample_id,
            "attacked_id": attacked.sample_id,
            "correct_label": attacked.correct_label,
            "distractor_label": attacked.distractor_label,
            "diff_mask_semantics": (
                "high_confidence_digital_text_mask"
                if attacked.subset in DIGITAL_SUBSETS
                else "handwritten_attack_change_proxy"
            ),
            "primary_diff_threshold": float(primary_threshold),
            "patch_pixel_fraction_threshold": float(patch_pixel_fraction),
            "primary_pixel_fraction": float(primary_pixel_mask.mean()),
            "primary_patch_count": int(covered.sum()),
            "primary_patch_fraction": float(covered.mean()),
        }

        for threshold in thresholds:
            mask = diff > float(threshold)
            patch_fraction = _patch_fraction_map(mask.astype(np.float32), grid)
            suffix = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
            row[f"t{suffix}_pixel_fraction"] = float(mask.mean())
            row[f"t{suffix}_patch_count_any"] = int((patch_fraction > 0.0).sum())
            row[f"t{suffix}_patch_count_covered"] = int((patch_fraction >= float(patch_pixel_fraction)).sum())

        rows.append(row)
        patch_masks.append(covered.astype(np.uint8))
        rgb_images.append(attacked_rgb)
        meta.append(
            {
                "mask_row_index": len(meta),
                "subset": attacked.subset,
                "modality": _modality(attacked.subset),
                "family": _family(attacked.subset),
                "pair_key": _pair_key(attacked.sample_id),
                "base_id": base.sample_id,
                "attacked_id": attacked.sample_id,
                "correct_label": attacked.correct_label,
                "distractor_label": attacked.distractor_label,
            }
        )

        del pixel_values

    return rows, np.stack(patch_masks, axis=0), meta, rgb_images


# ================================================================================================
# RN / no-RN branch through target block
# ================================================================================================

@torch.inference_mode()
def run_shared_trunk_rn_branch(
    model: Any,
    pixel_values: torch.Tensor,
    *,
    reference_block: int,
    target_block: int,
    rn_insert_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (reference_block < rn_insert_block <= target_block):
        raise ValueError("Assumes reference < RN insert <= target")

    vision = model.vision_model

    hidden = vision.embeddings(pixel_values, interpolate_pos_encoding=False)
    hidden = vision.pre_layrnorm(hidden)

    reference_state = None
    for block in range(rn_insert_block):
        hidden = vision.encoder.layers[block](hidden, None)
        if block == reference_block:
            reference_state = hidden

    if reference_state is None:
        raise RuntimeError(f"Failed to capture B{reference_block}")

    hidden_no_rn = hidden
    rn = model.read_null_token.to(device=hidden.device, dtype=hidden.dtype)
    hidden_rn = torch.cat(
        (hidden, rn.view(1, 1, -1).expand(hidden.shape[0], 1, -1)),
        dim=1,
    )

    for block in range(rn_insert_block, target_block + 1):
        layer = vision.encoder.layers[block]
        hidden_no_rn = layer(hidden_no_rn, None)
        hidden_rn = layer(hidden_rn, None)

    return reference_state, hidden_no_rn, hidden_rn


def source_patch_logits(model: Any, state: torch.Tensor, *, has_rn: bool) -> torch.Tensor:
    visual_tokens = state[:, :-1, :] if has_rn else state
    return model.read_implant.source_head.patch_logits(visual_tokens).detach().float()


@torch.inference_mode()
def collect_probe_logits(
    model: Any,
    processor: Any,
    sample_sets: dict[str, list[PairSample]],
    *,
    reference_block: int,
    target_block: int,
    rn_insert_block: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    print()
    print("[probe] Shared trunk -> exact RN/no-RN branch ...")

    for subset in ATTACK_SUBSETS:
        samples = sample_sets[subset]

        for start, batch_samples in tqdm(
            list(_batches(samples, batch_size)),
            desc=f"B{reference_block}/B{target_block} {subset}",
            leave=False,
        ):
            pixel_values = preprocess_images(processor, batch_samples, device)

            with _autocast(device, amp):
                state_ref, state_no_rn, state_rn = run_shared_trunk_rn_branch(
                    model,
                    pixel_values,
                    reference_block=reference_block,
                    target_block=target_block,
                    rn_insert_block=rn_insert_block,
                )

            logits_ref = source_patch_logits(model, state_ref, has_rn=False)
            logits_no = source_patch_logits(model, state_no_rn, has_rn=False)
            logits_rn = source_patch_logits(model, state_rn, has_rn=True)

            if not (logits_ref.shape == logits_no.shape == logits_rn.shape):
                raise RuntimeError(
                    f"Shape mismatch: ref={tuple(logits_ref.shape)}, noRN={tuple(logits_no.shape)}, RN={tuple(logits_rn.shape)}"
                )

            expected_patches = (
                int(model.config.vision_config.image_size)
                // int(model.config.vision_config.patch_size)
            ) ** 2
            if logits_ref.shape[-1] != expected_patches:
                raise RuntimeError(
                    f"Expected {expected_patches} patch logits, got {logits_ref.shape[-1]}"
                )

            stacked = torch.stack((logits_ref, logits_no, logits_rn), dim=1)
            arrays.append(stacked.cpu().numpy().astype(np.float32, copy=False))

            for local_index, sample in enumerate(batch_samples):
                meta.append(
                    {
                        "raw_row_index": len(meta),
                        "subset": subset,
                        "modality": _modality(subset),
                        "family": _family(subset),
                        "pair_key": _pair_key(sample.sample_id),
                        "id": sample.sample_id,
                        "image_index": start + local_index,
                        "correct_label": sample.correct_label,
                        "distractor_label": sample.distractor_label,
                    }
                )

            del pixel_values, state_ref, state_no_rn, state_rn, logits_ref, logits_no, logits_rn, stacked

    return np.concatenate(arrays, axis=0), meta


# ================================================================================================
# Metrics
# ================================================================================================

def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.bool_).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    ranked = y[order].astype(np.float64)
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, len(ranked) + 1, dtype=np.float64)
    return float((precision * ranked).sum() / positives)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.bool_).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)

    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end

    pos_rank_sum = float(ranks[y].sum())
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _topk_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    k = min(int(topk), scores.size)
    return np.argpartition(scores, scores.size - k)[-k:]


def _center_of_mass_positive(grid: np.ndarray) -> tuple[float, float]:
    positive = np.clip(grid.astype(np.float64), 0.0, None)
    total = float(positive.sum())
    if total <= 1.0e-12:
        return float("nan"), float("nan")
    yy, xx = np.indices(positive.shape, dtype=np.float64)
    return float((yy * positive).sum() / total), float((xx * positive).sum() / total)


def _grid_metrics(scores: np.ndarray, mask: np.ndarray, *, topk: int) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    background = ~mask
    text_count = int(mask.sum())

    positive = scores > 0.0
    positive_count = int(positive.sum())
    intersection = positive & mask
    intersection_count = int(intersection.sum())

    top_indices = _topk_indices(scores, topk)
    topk_text_fraction = float(mask[top_indices].mean())

    text_mean = float(scores[mask].mean()) if text_count else float("nan")
    background_mean = float(scores[background].mean()) if background.any() else float("nan")
    text_median = float(np.median(scores[mask])) if text_count else float("nan")
    background_median = float(np.median(scores[background])) if background.any() else float("nan")

    grid_side = int(round(math.sqrt(scores.size)))
    grid = scores.reshape(grid_side, grid_side)
    com_y, com_x = _center_of_mass_positive(grid)

    return {
        "patch_ap": _average_precision(mask, scores),
        "patch_roc_auc": _roc_auc(mask, scores),
        "topk_text_fraction": topk_text_fraction,
        "text_mean_logit": text_mean,
        "background_mean_logit": background_mean,
        "text_minus_background_mean": text_mean - background_mean,
        "text_median_logit": text_median,
        "background_median_logit": background_median,
        "text_minus_background_median": text_median - background_median,
        "positive_patch_count": positive_count,
        "positive_patch_fraction": float(positive.mean()),
        "positive_precision_for_text": intersection_count / positive_count if positive_count else float("nan"),
        "text_recall_by_positive_logits": intersection_count / text_count if text_count else float("nan"),
        "raw_mean": float(scores.mean()),
        "raw_median": float(np.median(scores)),
        "raw_min": float(scores.min()),
        "raw_max": float(scores.max()),
        "raw_topk_signed_mean": float(scores[top_indices].mean()),
        "topk_indices_json": json.dumps(sorted(int(i) for i in top_indices.tolist())),
        "center_of_mass_y": com_y,
        "center_of_mass_x": com_x,
    }


def build_relocation_rows(
    raw_logits: np.ndarray,
    raw_meta: list[dict[str, Any]],
    diff_rows: list[dict[str, Any]],
    diff_masks: np.ndarray,
    diff_meta: list[dict[str, Any]],
    *,
    reference_block: int,
    target_block: int,
    topk: int,
) -> list[dict[str, Any]]:
    raw_index = {item["id"]: int(item["raw_row_index"]) for item in raw_meta}
    raw_meta_by_id = {item["id"]: item for item in raw_meta}
    diff_index = {item["attacked_id"]: int(item["mask_row_index"]) for item in diff_meta}
    diff_by_id = {row["attacked_id"]: row for row in diff_rows}

    rows: list[dict[str, Any]] = []

    for attacked_id, mask_index in diff_index.items():
        raw_row_index = raw_index.get(attacked_id)
        if raw_row_index is None:
            continue

        meta = raw_meta_by_id[attacked_id]
        diff_row = diff_by_id[attacked_id]
        text_mask = diff_masks[mask_index].astype(bool).reshape(-1)

        scores_ref = raw_logits[raw_row_index, 0].astype(np.float64, copy=False)
        scores_no = raw_logits[raw_row_index, 1].astype(np.float64, copy=False)
        scores_rn = raw_logits[raw_row_index, 2].astype(np.float64, copy=False)

        ref_metrics = _grid_metrics(scores_ref, text_mask, topk=topk)
        no_metrics = _grid_metrics(scores_no, text_mask, topk=topk)
        rn_metrics = _grid_metrics(scores_rn, text_mask, topk=topk)

        top_no = set(json.loads(no_metrics["topk_indices_json"]))
        top_rn = set(json.loads(rn_metrics["topk_indices_json"]))
        top_common = len(top_no & top_rn)
        top_union = len(top_no | top_rn)

        grid_side = int(round(math.sqrt(scores_no.size)))
        no_grid = scores_no.reshape(grid_side, grid_side)
        rn_grid = scores_rn.reshape(grid_side, grid_side)
        delta_grid = rn_grid - no_grid
        gain_grid = np.clip(delta_grid, 0.0, None)
        loss_grid = np.clip(-delta_grid, 0.0, None)
        gain_top = _topk_indices(gain_grid.reshape(-1), topk)
        loss_top = _topk_indices(loss_grid.reshape(-1), topk)

        row = {
            "subset": meta["subset"],
            "modality": meta["modality"],
            "family": meta["family"],
            "pair_key": meta["pair_key"],
            "id": attacked_id,
            "correct_label": meta["correct_label"],
            "distractor_label": meta["distractor_label"],
            "mask_semantics": diff_row["diff_mask_semantics"],
            "reference_block": int(reference_block),
            "target_block": int(target_block),
            "text_patch_count": int(text_mask.sum()),
            "text_patch_fraction": float(text_mask.mean()),
            "ref_patch_ap": ref_metrics["patch_ap"],
            "no_rn_patch_ap": no_metrics["patch_ap"],
            "rn_patch_ap": rn_metrics["patch_ap"],
            "rn_minus_no_rn_patch_ap": rn_metrics["patch_ap"] - no_metrics["patch_ap"],
            "ref_topk_text_fraction": ref_metrics["topk_text_fraction"],
            "no_rn_topk_text_fraction": no_metrics["topk_text_fraction"],
            "rn_topk_text_fraction": rn_metrics["topk_text_fraction"],
            "rn_minus_no_rn_topk_text_fraction": rn_metrics["topk_text_fraction"] - no_metrics["topk_text_fraction"],
            "no_rn_text_minus_background_mean": no_metrics["text_minus_background_mean"],
            "rn_text_minus_background_mean": rn_metrics["text_minus_background_mean"],
            "rn_minus_no_rn_text_minus_background_mean": rn_metrics["text_minus_background_mean"] - no_metrics["text_minus_background_mean"],
            "no_rn_raw_topk_signed_mean": no_metrics["raw_topk_signed_mean"],
            "rn_raw_topk_signed_mean": rn_metrics["raw_topk_signed_mean"],
            "rn_minus_no_rn_raw_topk_signed_mean": rn_metrics["raw_topk_signed_mean"] - no_metrics["raw_topk_signed_mean"],
            "no_rn_positive_precision_for_text": no_metrics["positive_precision_for_text"],
            "rn_positive_precision_for_text": rn_metrics["positive_precision_for_text"],
            "no_rn_text_recall_by_positive_logits": no_metrics["text_recall_by_positive_logits"],
            "rn_text_recall_by_positive_logits": rn_metrics["text_recall_by_positive_logits"],
            "no_rn_center_of_mass_y": no_metrics["center_of_mass_y"],
            "no_rn_center_of_mass_x": no_metrics["center_of_mass_x"],
            "rn_center_of_mass_y": rn_metrics["center_of_mass_y"],
            "rn_center_of_mass_x": rn_metrics["center_of_mass_x"],
            "topk_intersection_count": top_common,
            "topk_union_count": top_union,
            "topk_jaccard": top_common / top_union if top_union else float("nan"),
            "gain_topk_text_fraction": float(text_mask[gain_top].mean()),
            "loss_topk_text_fraction": float(text_mask[loss_top].mean()),
            "gain_topk_indices_json": json.dumps(sorted(int(i) for i in gain_top.tolist())),
            "loss_topk_indices_json": json.dumps(sorted(int(i) for i in loss_top.tolist())),
            "no_rn_topk_indices_json": no_metrics["topk_indices_json"],
            "rn_topk_indices_json": rn_metrics["topk_indices_json"],
        }
        rows.append(row)

    return rows


def summarize_relocation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["modality"]), str(row["family"]))].append(row)
        grouped[(str(row["modality"]), "ALL")].append(row)

    output: list[dict[str, Any]] = []
    for modality in ("handwritten", "digital"):
        for family in ("SCAM", "RTA", "ALL"):
            group = grouped.get((modality, family), [])
            if not group:
                continue
            output.append(
                {
                    "modality": modality,
                    "family": family,
                    "count": len(group),
                    "mean_ref_patch_ap": _safe_mean(row["ref_patch_ap"] for row in group),
                    "mean_no_rn_patch_ap": _safe_mean(row["no_rn_patch_ap"] for row in group),
                    "mean_rn_patch_ap": _safe_mean(row["rn_patch_ap"] for row in group),
                    "mean_rn_minus_no_rn_patch_ap": _safe_mean(row["rn_minus_no_rn_patch_ap"] for row in group),
                    "median_rn_minus_no_rn_patch_ap": _safe_median(row["rn_minus_no_rn_patch_ap"] for row in group),
                    "mean_no_rn_topk_text_fraction": _safe_mean(row["no_rn_topk_text_fraction"] for row in group),
                    "mean_rn_topk_text_fraction": _safe_mean(row["rn_topk_text_fraction"] for row in group),
                    "mean_topk_jaccard": _safe_mean(row["topk_jaccard"] for row in group),
                    "mean_gain_topk_text_fraction": _safe_mean(row["gain_topk_text_fraction"] for row in group),
                    "mean_loss_topk_text_fraction": _safe_mean(row["loss_topk_text_fraction"] for row in group),
                }
            )
    return output


# ================================================================================================
# Overlay rendering
# ================================================================================================

def _upsample_grid(grid: np.ndarray, image_size: int) -> np.ndarray:
    patch_side = int(round(math.sqrt(grid.size)))
    grid_2d = grid.reshape(patch_side, patch_side)
    patch_size = image_size // patch_side
    return np.kron(grid_2d, np.ones((patch_size, patch_size), dtype=np.float32))


def _normalize_positive_map(grid: np.ndarray, vmax: float | None = None) -> tuple[np.ndarray, float]:
    positive = np.clip(grid.astype(np.float32), 0.0, None)
    if vmax is None:
        vmax = float(positive.max())
    if vmax <= 1.0e-12:
        return np.zeros_like(positive), vmax
    return np.clip(positive / vmax, 0.0, 1.0), vmax


def _normalize_abs_map(grid: np.ndarray, vmax: float | None = None) -> tuple[np.ndarray, float]:
    values = np.abs(grid.astype(np.float32))
    if vmax is None:
        vmax = float(values.max())
    if vmax <= 1.0e-12:
        return np.zeros_like(values), vmax
    return np.clip(values / vmax, 0.0, 1.0), vmax


def _patch_rectangles(indices: list[int], grid_side: int) -> list[tuple[int, int]]:
    return [(int(index) // grid_side, int(index) % grid_side) for index in indices]


def render_single_panel(
    *,
    row: dict[str, Any],
    attacked_rgb: np.ndarray,
    text_patch_mask: np.ndarray,
    no_grid: np.ndarray,
    rn_grid: np.ndarray,
    output_path: Path,
    topk: int,
    cmap_name: str = "turbo",
) -> None:
    image_size = attacked_rgb.shape[0]
    grid_side = text_patch_mask.shape[0]
    patch_size = image_size // grid_side

    delta_grid = rn_grid - no_grid
    gain_grid = np.clip(delta_grid, 0.0, None)
    loss_grid = np.clip(-delta_grid, 0.0, None)

    shared_abs_max = float(
        max(
            np.clip(no_grid, 0.0, None).max(),
            np.clip(rn_grid, 0.0, None).max(),
            1.0e-12,
        )
    )
    delta_abs_max = float(max(np.abs(delta_grid).max(), 1.0e-12))

    no_img = _upsample_grid(no_grid, image_size)
    rn_img = _upsample_grid(rn_grid, image_size)
    gain_img = _upsample_grid(gain_grid, image_size)
    loss_img = _upsample_grid(loss_grid, image_size)
    mask_img = _upsample_grid(text_patch_mask.astype(np.float32), image_size)

    no_norm, _ = _normalize_positive_map(no_img, shared_abs_max)
    rn_norm, _ = _normalize_positive_map(rn_img, shared_abs_max)
    gain_norm, _ = _normalize_positive_map(gain_img, delta_abs_max)
    loss_norm, _ = _normalize_positive_map(loss_img, delta_abs_max)

    cmap = plt.get_cmap(cmap_name)

    def overlay_rgba(norm_map: np.ndarray) -> np.ndarray:
        rgba = cmap(norm_map)
        rgba[..., 3] = norm_map  # black / zero becomes fully transparent
        return rgba

    no_rgba = overlay_rgba(no_norm)
    rn_rgba = overlay_rgba(rn_norm)
    gain_rgba = overlay_rgba(gain_norm)
    loss_rgba = overlay_rgba(loss_norm)

    no_top = json.loads(row["no_rn_topk_indices_json"])
    rn_top = json.loads(row["rn_topk_indices_json"])
    gain_top = json.loads(row["gain_topk_indices_json"])
    loss_top = json.loads(row["loss_topk_indices_json"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.reshape(2, 3)

    title = (
        f"{row['id']} | {row['subset']} | {row['correct_label']} vs {row['distractor_label']}\n"
        f"B{row['target_block']} AP noRN={row['no_rn_patch_ap']:.3f} "
        f"RN={row['rn_patch_ap']:.3f} Δ={row['rn_minus_no_rn_patch_ap']:+.3f} | "
        f"topk Jaccard={row['topk_jaccard']:.3f}"
    )
    fig.suptitle(title, fontsize=12)

    # 1) attacked image
    ax = axes[0, 0]
    ax.imshow(attacked_rgb)
    ax.set_title("Attacked image (ViT input)")
    ax.axis("off")

    # 2) text mask
    ax = axes[0, 1]
    ax.imshow(attacked_rgb)
    red = np.zeros((image_size, image_size, 4), dtype=np.float32)
    red[..., 0] = 1.0
    red[..., 3] = 0.45 * mask_img
    ax.imshow(red)
    ax.set_title(f"Paired text mask ({int(text_patch_mask.sum())} patches)")
    for py in range(grid_side + 1):
        ax.axhline(py * patch_size - 0.5, color="white", linewidth=0.25, alpha=0.25)
        ax.axvline(py * patch_size - 0.5, color="white", linewidth=0.25, alpha=0.25)
    ax.axis("off")

    # 3) no RN overlay
    ax = axes[0, 2]
    ax.imshow(attacked_rgb)
    ax.imshow(no_rgba)
    ax.set_title(f"B{row['target_block']} NO_RN source PIECE")
    for r, c in _patch_rectangles(no_top, grid_side):
        rect = plt.Rectangle(
            (c * patch_size, r * patch_size),
            patch_size,
            patch_size,
            fill=False,
            linewidth=1.4,
            edgecolor="white",
        )
        ax.add_patch(rect)
    ax.axis("off")

    # 4) RN overlay
    ax = axes[1, 0]
    ax.imshow(attacked_rgb)
    ax.imshow(rn_rgba)
    ax.set_title(f"B{row['target_block']} RN source PIECE")
    for r, c in _patch_rectangles(rn_top, grid_side):
        rect = plt.Rectangle(
            (c * patch_size, r * patch_size),
            patch_size,
            patch_size,
            fill=False,
            linewidth=1.4,
            edgecolor="white",
        )
        ax.add_patch(rect)
    ax.axis("off")

    # 5) RN gain overlay
    ax = axes[1, 1]
    ax.imshow(attacked_rgb)
    ax.imshow(gain_rgba)
    ax.set_title(
        f"RN gain (RN - noRN)+ | top{topk} on text={row['gain_topk_text_fraction']:.3f}"
    )
    for r, c in _patch_rectangles(gain_top, grid_side):
        rect = plt.Rectangle(
            (c * patch_size, r * patch_size),
            patch_size,
            patch_size,
            fill=False,
            linewidth=1.4,
            edgecolor="white",
        )
        ax.add_patch(rect)
    ax.axis("off")

    # 6) RN loss overlay
    ax = axes[1, 2]
    ax.imshow(attacked_rgb)
    ax.imshow(loss_rgba)
    ax.set_title(
        f"RN loss (noRN - RN)+ | top{topk} on text={row['loss_topk_text_fraction']:.3f}"
    )
    for r, c in _patch_rectangles(loss_top, grid_side):
        rect = plt.Rectangle(
            (c * patch_size, r * patch_size),
            patch_size,
            patch_size,
            fill=False,
            linewidth=1.4,
            edgecolor="white",
        )
        ax.add_patch(rect)
    ax.axis("off")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_gallery(
    image_paths: list[Path],
    output_path: Path,
    *,
    cols: int = 2,
    thumb_width: int = 1100,
    gutter: int = 20,
    bg: tuple[int, int, int] = (16, 16, 16),
) -> None:
    if not image_paths:
        return

    images = [Image.open(path).convert("RGB") for path in image_paths]
    images = [img.resize((thumb_width, int(img.height * thumb_width / img.width))) for img in images]

    rows = int(math.ceil(len(images) / cols))
    col_width = max(img.width for img in images)
    row_heights: list[int] = []
    for row_index in range(rows):
        row_images = images[row_index * cols : (row_index + 1) * cols]
        row_heights.append(max(img.height for img in row_images))

    width = cols * col_width + (cols + 1) * gutter
    height = sum(row_heights) + (rows + 1) * gutter
    canvas = Image.new("RGB", (width, height), bg)

    y = gutter
    for row_index in range(rows):
        x = gutter
        row_images = images[row_index * cols : (row_index + 1) * cols]
        for img in row_images:
            canvas.paste(img, (x, y))
            x += col_width + gutter
        y += row_heights[row_index] + gutter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


# ================================================================================================
# Main
# ================================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render RN/no-RN relocation overlays for the frozen source PIECE."
    )
    parser.add_argument("--full-model", type=str, default=DEFAULT_FULL_MODEL,
                        help="HF repo id or local HF model directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-block", type=int, default=DEFAULT_REFERENCE_BLOCK)
    parser.add_argument("--target-block", type=int, default=DEFAULT_TARGET_BLOCK)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA FP16 autocast for the stock ViT; source PIECE remains FP32.",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--top-n-per-modality", type=int, default=DEFAULT_TOP_N_PER_MODALITY)
    parser.add_argument(
        "--diff-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_DIFF_THRESHOLDS),
    )
    parser.add_argument("--primary-diff-threshold", type=float, default=DEFAULT_PRIMARY_DIFF_THRESHOLD)
    parser.add_argument("--patch-pixel-fraction", type=float, default=DEFAULT_PATCH_PIXEL_FRACTION)
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional exact attacked sample IDs to render in addition to automatic selection.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.top_n_per_modality <= 0:
        raise ValueError("--top-n-per-modality must be positive")
    if not (0.0 < args.patch_pixel_fraction <= 1.0):
        raise ValueError("--patch-pixel-fraction must be in (0,1]")

    configure_reproducibility(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[run] device={device}")
    print(f"[run] model={args.full_model}")
    print(f"[run] output={output_dir}")
    print(f"[run] reference B{args.reference_block}, target B{args.target_block}")

    sample_sets = load_samples()
    pairs = paired_attacks(sample_sets)

    print(f"[dataset] paired attacked images={len(pairs)}")

    print(f"[model] Loading {args.full_model}")
    model = AutoModel.from_pretrained(args.full_model, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.full_model, trust_remote_code=True)

    if not hasattr(model, "read_null_token") or not hasattr(model, "read_implant"):
        raise TypeError("Expected full x-attention model with read_null_token and read_implant")

    rn_insert_block = int(model.config.read_null_insert_block)
    if rn_insert_block != EXPECTED_RN_INSERT_BLOCK:
        print(
            f"[warn] Expected RN insertion before B{EXPECTED_RN_INSERT_BLOCK}, "
            f"model says B{rn_insert_block}. Proceeding with model value."
        )

    vision_layers = len(model.vision_model.encoder.layers)
    if not (0 <= args.reference_block < vision_layers):
        raise ValueError(f"reference block outside 0..{vision_layers-1}")
    if not (0 <= args.target_block < vision_layers):
        raise ValueError(f"target block outside 0..{vision_layers-1}")
    if not (args.reference_block < rn_insert_block <= args.target_block):
        raise ValueError(
            f"Require reference < RN insert <= target, got B{args.reference_block}, "
            f"RN before B{rn_insert_block}, target B{args.target_block}"
        )

    raw_logits, raw_meta = collect_probe_logits(
        model,
        processor,
        sample_sets,
        reference_block=args.reference_block,
        target_block=args.target_block,
        rn_insert_block=rn_insert_block,
        batch_size=args.batch_size,
        device=device,
        amp=args.amp,
    )

    patch_count = int(raw_logits.shape[-1])
    grid = int(round(math.sqrt(patch_count)))
    if grid * grid != patch_count:
        raise ValueError(f"Unexpected patch count {patch_count}")

    thresholds = sorted(
        {*(float(value) for value in args.diff_thresholds), float(args.primary_diff_threshold)}
    )

    diff_rows, diff_masks, diff_meta, attacked_rgbs = collect_diff_masks(
        processor,
        pairs,
        patch_count=patch_count,
        thresholds=thresholds,
        primary_threshold=float(args.primary_diff_threshold),
        patch_pixel_fraction=float(args.patch_pixel_fraction),
        device=device,
    )

    _write_csv(output_dir / "attack_text_diff_coverage.csv", diff_rows)
    _write_csv(output_dir / "attack_text_masks_meta.csv", diff_meta)
    np.save(output_dir / "attack_text_patch_masks.npy", diff_masks)
    np.save(output_dir / "Bref_Btarget_rn_relocation_raw_logits.npy", raw_logits)
    _write_csv(output_dir / "Bref_Btarget_rn_relocation_raw_logits_meta.csv", raw_meta)

    relocation_rows = build_relocation_rows(
        raw_logits,
        raw_meta,
        diff_rows,
        diff_masks,
        diff_meta,
        reference_block=args.reference_block,
        target_block=args.target_block,
        topk=args.topk,
    )
    relocation_summary = summarize_relocation(relocation_rows)

    _write_csv(output_dir / "rn_relocation_by_image.csv", relocation_rows)
    _write_csv(output_dir / "rn_relocation_summary.csv", relocation_summary)

    print()
    print("=" * 120)
    print("RN TEXT RELOCATION SUMMARY")
    print("=" * 120)
    print(
        f"{'modality':<12} {'family':<6} {'count':>6} "
        f"{'noRN AP':>10} {'RN AP':>10} {'RN-noRN':>10} "
        f"{'noRN topk@text':>15} {'RN topk@text':>15} {'topk Jaccard':>14}"
    )
    print("-" * 120)
    for row in relocation_summary:
        print(
            f"{row['modality']:<12} {row['family']:<6} {row['count']:>6d} "
            f"{row['mean_no_rn_patch_ap']:>10.4f} {row['mean_rn_patch_ap']:>10.4f} {row['mean_rn_minus_no_rn_patch_ap']:>+10.4f} "
            f"{row['mean_no_rn_topk_text_fraction']:>15.4f} {row['mean_rn_topk_text_fraction']:>15.4f} {row['mean_topk_jaccard']:>14.4f}"
        )

    # --------------------------------------------------------------------------------------------
    # Select examples and render overlays.
    # --------------------------------------------------------------------------------------------
    diff_meta_by_id = {item["attacked_id"]: item for item in diff_meta}
    diff_row_index_by_id = {item["attacked_id"]: int(item["mask_row_index"]) for item in diff_meta}
    attacked_rgb_by_id = {
        item["attacked_id"]: attacked_rgbs[int(item["mask_row_index"])]
        for item in diff_meta
    }
    raw_index_by_id = {item["id"]: int(item["raw_row_index"]) for item in raw_meta}

    selected_ids: list[str] = []
    selected_rows: list[dict[str, Any]] = []

    for modality in ("handwritten", "digital"):
        group = [row for row in relocation_rows if row["modality"] == modality]
        group.sort(
            key=lambda item: (
                float(item["rn_minus_no_rn_patch_ap"]),     # most negative = biggest drop
                float(item["rn_minus_no_rn_topk_text_fraction"]),
                float(item["topk_jaccard"]),
            )
        )
        selected_rows.extend(group[: args.top_n_per_modality])

    if args.sample_ids:
        lookup = {row["id"]: row for row in relocation_rows}
        for sample_id in args.sample_ids:
            if sample_id in lookup:
                selected_rows.append(lookup[sample_id])

    dedup = {}
    for row in selected_rows:
        dedup[row["id"]] = row
    selected_rows = list(dedup.values())
    selected_rows.sort(key=lambda item: (item["modality"], float(item["rn_minus_no_rn_patch_ap"])))

    per_image_dir = output_dir / "per_image_panels"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    rendered_paths_by_modality: dict[str, list[Path]] = {"handwritten": [], "digital": []}
    selection_table: list[dict[str, Any]] = []

    for row in tqdm(selected_rows, desc="render overlays", leave=False):
        sample_id = row["id"]
        raw_row_index = raw_index_by_id[sample_id]
        mask_row_index = diff_row_index_by_id[sample_id]

        no_grid = raw_logits[raw_row_index, 1].reshape(grid, grid)
        rn_grid = raw_logits[raw_row_index, 2].reshape(grid, grid)
        text_patch_mask = diff_masks[mask_row_index].reshape(grid, grid)
        attacked_rgb = attacked_rgb_by_id[sample_id]

        filename = (
            f"{row['modality']}__{row['subset']}__"
            f"{sample_id.replace('/', '_').replace(':', '_')}.png"
        )
        panel_path = per_image_dir / filename

        render_single_panel(
            row=row,
            attacked_rgb=attacked_rgb,
            text_patch_mask=text_patch_mask,
            no_grid=no_grid,
            rn_grid=rn_grid,
            output_path=panel_path,
            topk=args.topk,
        )

        rendered_paths_by_modality[row["modality"]].append(panel_path)
        selection_table.append(
            {
                "panel_path": str(panel_path),
                **row,
            }
        )

    _write_csv(output_dir / "selected_overlay_examples.csv", selection_table)

    handwritten_gallery = output_dir / "gallery_handwritten_biggest_RN_relocation.png"
    digital_gallery = output_dir / "gallery_digital_biggest_RN_relocation.png"
    combined_gallery = output_dir / "gallery_combined_biggest_RN_relocation.png"

    _make_gallery(rendered_paths_by_modality["handwritten"], handwritten_gallery, cols=2)
    _make_gallery(rendered_paths_by_modality["digital"], digital_gallery, cols=2)
    _make_gallery(
        rendered_paths_by_modality["handwritten"] + rendered_paths_by_modality["digital"],
        combined_gallery,
        cols=2,
    )

    metadata = {
        "full_model": str(args.full_model),
        "device": str(device),
        "stored_dtype": str(next(model.parameters()).dtype),
        "amp": bool(args.amp),
        "reference_block_zero_based": int(args.reference_block),
        "target_block_zero_based": int(args.target_block),
        "rn_insert_block_zero_based": int(rn_insert_block),
        "topk": int(args.topk),
        "top_n_per_modality": int(args.top_n_per_modality),
        "patch_count": int(patch_count),
        "patch_grid": [int(grid), int(grid)],
        "raw_logits_channels": [
            f"B{args.reference_block}_shared_pre_RN",
            f"B{args.target_block}_NO_RN",
            f"B{args.target_block}_RN",
        ],
        "diff_thresholds_8bit": thresholds,
        "primary_diff_threshold_8bit": float(args.primary_diff_threshold),
        "patch_pixel_fraction_threshold": float(args.patch_pixel_fraction),
        "selected_sample_ids": [row["id"] for row in selected_rows],
        "rendered_panel_dir": str(per_image_dir),
        "galleries": {
            "handwritten": str(handwritten_gallery),
            "digital": str(digital_gallery),
            "combined": str(combined_gallery),
        },
        "interpretation": {
            "selection": (
                "Automatic selection is by largest AP drop RN - noRN "
                "(most negative first), separately for handwritten and digital."
            ),
            "gain_map": (
                "RN gain overlay shows positive (RN - noRN) patch-logit changes."
            ),
            "loss_map": (
                "RN loss overlay shows positive (noRN - RN) patch-logit changes."
            ),
            "turbo_overlay": (
                "Positive source evidence is rendered with turbo and alpha tied to intensity, "
                "so zero/black is transparent."
            ),
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"[run] Wrote results to: {output_dir}")
    print("[run] Primary outputs:")
    for name in (
        "rn_relocation_by_image.csv",
        "rn_relocation_summary.csv",
        "selected_overlay_examples.csv",
        "gallery_handwritten_biggest_RN_relocation.png",
        "gallery_digital_biggest_RN_relocation.png",
        "gallery_combined_biggest_RN_relocation.png",
        "per_image_panels/",
        "run_metadata.json",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
