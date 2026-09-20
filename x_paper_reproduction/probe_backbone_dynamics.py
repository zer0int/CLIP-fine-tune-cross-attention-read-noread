#!/usr/bin/env python3
r"""GIPU / backbone-dynamics analysis for the CLIP x-attention paper.

This script complements probe_cross_attention_bridge.py.  The bridge
probe analyzes SOURCE / ORTHO / READ / correction.  This script analyzes the visual
transformer substrate itself and compares three computations:

    1. pretrained_gmp
       zer0int/CLIP-GmP-ViT-L-14 (explicit pre-xattn GmP comparison baseline;
       this is not used as a synonym for vanilla OpenAI CLIP)

    2. trained_vanilla
       the jointly-trained x-attention checkpoint rebuilt as a literal stock HF CLIPModel:
       no bridge, no correction, no READ_NULL token, no custom control-token rows.

    3. trained_rn_classic
       the same jointly-trained visual/text backbone with READ_NULL physically inserted,
       while the bridge is not used.  This is the visual computation underlying x-attn
       mode="classic".

Fixed dataset: image_sets/demoset and the paper PROMPT_WORDS only.

Main measurements
-----------------
A. Tracked-register lineage and score attribution, B0..B23
   * Register positions are inferred ONCE per image from the final spatial state using
       ||x_patch||_2 >= --register-threshold  (default 60)
     and those same spatial indices are tracked backward through every earlier block,
     even while their norms are ordinary / camouflaged.
   * Raw CLS->tracked-register attention fraction.
   * Raw patch-query incoming attention to tracked-register columns.
   * Positive gradient*attention score attribution fraction for annotated present prompts:

       sum_{p in tracked registers} mean_h ReLU(A_h * d cos / d A_h)[CLS,p]
       ----------------------------------------------------------------------
       sum_{p in all spatial patches} mean_h ReLU(A_h * d cos / d A_h)[CLS,p]

     The target is ordinary CLIP cosine similarity for each runtime variant.

B. Image-invariant register subspace, B0..B23
   * For each image and block, average the states at the tracked register positions.
   * Fit an UNCENTERED SVD across image-wise register means.
   * Report cross-image pairwise cosine before removal and after removing rank 1, 2, 3.
   * Save the first 3 directions and singular-energy statistics.
   * Compare PC1 and rank-2 subspace alignment between model variants.

C. mu-cache lineage
   * Learn mu1 from B23 high-norm register means.
   * Hold that B23 direction FIXED and measure cosine to every spatial patch at B0..B23.
   * Cache candidates are non-register-lineage patches, currently below the register norm
     threshold, with cos(patch, mu1_B23) >= --cache-cos-threshold.
   * Report candidate fraction and edge-ring enrichment over blocks.

D. Image-local scratchpad coherence
   * At each block, remove the BLOCK-LOCAL rank-2 invariant register subspace.
   * Exclude tracked-register and mu-cache candidates from scratchpad candidate selection.
   * For every remaining patch, find its maximum cosine to a spatially DISTANT patch in
     the SAME image and its maximum cosine to any eligible patch in OTHER images.
   * Save within-image distant cosine, cross-image nearest cosine, their gap, partner
     distance/index, and a conservative binary candidate mask.

E. Paper-oriented maps and temporal role strips
   * For selected blocks, save compact panels showing input / norm / tracked register
     lineage / fixed-mu1 cache map / scratchpad gap / semantic grad*attention.
   * For EVERY image, save B0..B23 temporal role strips with one patch per row and
     one transformer block per column.  Two orderings are emitted:
       - spatial_order: original row-major image-patch identity;
       - role_sorted: tracked-register lineage, mu-cache-bearing patches,
         scratchpad-bearing patches, then remaining patches.
     Each strip shows dynamic role class, residual norm, fixed-B23-mu1 cosine,
     scratchpad locality gap, and positive semantic grad*attention.

Output root defaults to:
    out_bridge_analysis_for_paper/gipu_backbone_dynamics

No bridge tap relocation/transplantation is performed.
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_analysis import (DemoSpec, ensure_dir)


import argparse
import csv
import gc
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoProcessor, CLIPConfig, CLIPModel


# =================================================================================================
# Fixed experiment definition
# =================================================================================================

PROMPT_WORDS = (
    "banana",
    "apple",
    "granny smith",
    "ipod",
    "iphone",
    "pineapple",
    "bird",
    "goldfinch",
    "bee",
    "bumblebee",
    "shampoo",
    "shower gel",
    "detergent",
    "cat",
    "dog",
    "raccoon",
    "badger",
    "word",
    "text",
    "typography",
)

DEFAULT_PROMPT_TEMPLATE = "a photo of a {prompt_word}"
DEFAULT_IMAGE_DIR = Path("image_sets/demoset")
DEFAULT_XATTN_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_PRETRAINED = "zer0int/CLIP-GmP-ViT-L-14"
DEFAULT_OUTPUT = Path("out_bridge_analysis_for_paper_GIPU_v2/gipu_backbone_dynamics")
VANILLA_CLIP_VOCAB_SIZE = 49_408


DEMO_SPECS = (
    DemoSpec("apple_ipod.png", ("granny smith", "apple"), ("ipod",), "granny smith/apple; rendered text: ipod"),
    DemoSpec("apple_none.png", ("granny smith", "apple"), (), "granny smith/apple; no attack text"),
    DemoSpec("bananorange_pineapple.png", ("banana",), ("pineapple",), "banana/orange composite; rendered text: pineapple"),
    DemoSpec("bananorange_none.png", ("banana",), (), "banana/orange composite; no attack text"),
    DemoSpec("bottle_shampoo.png", ("shampoo",), ("shampoo",), "shampoo bottle; rendered text agrees"),
    DemoSpec("bottle_shower.png", ("shower gel",), ("shower gel",), "shower-gel bottle; rendered text agrees"),
    DemoSpec("cat_cat.png", ("cat",), ("cat",), "cat; rendered text agrees"),
    DemoSpec("dog_cat.png", ("cat",), ("dog",), "cat; rendered text: dog"),
    DemoSpec("goldfinch_bumblebee.png", ("bird", "goldfinch"), ("bumblebee",), "goldfinch/bird; rendered text: bumblebee"),
    DemoSpec("goldfinch_none.png", ("bird", "goldfinch"), (), "goldfinch/bird; no attack text"),
    DemoSpec("thecat_raccoon.png", ("cat", "raccoon"), (), "cat with raccoon face in mirror"),
    DemoSpec("thecat_none.png", ("cat",), (), "cat with normal cat mirror reflection"),
)


@dataclass
class DemoSample:
    spec: DemoSpec
    path: Path
    image: Image.Image

    @property
    def stem(self) -> str:
        return Path(self.spec.filename).stem


@dataclass
class VariantResult:
    label: str
    invariant_rows: list[dict[str, Any]]
    raw_attention_rows: list[dict[str, Any]]
    attribution_rows: list[dict[str, Any]]
    cache_rows: list[dict[str, Any]]
    scratch_rows: list[dict[str, Any]]
    bases: dict[int, np.ndarray]
    selected_maps: dict[tuple[str, int], dict[str, np.ndarray]]
    register_masks: dict[str, np.ndarray]


# =================================================================================================
# Generic utilities
# =================================================================================================


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    preferred = ["variant", "image", "word", "block", "head"]
    union = set().union(*(row.keys() for row in rows))
    for key in preferred:
        if key in union:
            keys.append(key)
            seen.add(key)
    for key in sorted(union):
        if key not in seen:
            keys.append(key)
            seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def npz(path: Path, **arrays: Any) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def as_float(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().float().cpu())
    return float(x)


def batches(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def open_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return image.convert("RGB")


def load_samples(image_dir: Path) -> list[DemoSample]:
    samples: list[DemoSample] = []
    missing: list[str] = []
    allowed = set(PROMPT_WORDS)
    for spec in DEMO_SPECS:
        unknown = (set(spec.present_words) | set(spec.text_words)) - allowed
        if unknown:
            raise ValueError(f"{spec.filename} contains labels outside PROMPT_WORDS: {sorted(unknown)}")
        path = image_dir / spec.filename
        if not path.is_file():
            missing.append(str(path))
        else:
            samples.append(DemoSample(spec, path, open_rgb(path)))
    if missing:
        raise FileNotFoundError("Missing demoset files:\n  " + "\n  ".join(missing))
    return samples


def parse_block_list(value: str, n_blocks: int = 24) -> list[int]:
    value = value.strip().lower()
    if value == "all":
        return list(range(n_blocks))
    out: list[int] = []
    for part in value.split(","):
        if not part.strip():
            continue
        block = int(part)
        if block < 0:
            block += n_blocks
        if not 0 <= block < n_blocks:
            raise ValueError(f"Block {part} resolves outside 0..{n_blocks - 1}")
        if block not in out:
            out.append(block)
    return out


def grid_side(patch_count: int) -> int:
    side = int(round(math.sqrt(patch_count)))
    if side * side != patch_count:
        raise ValueError(f"Patch count {patch_count} is not square")
    return side


def to_grid(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(values):
        arr = values.detach().float().cpu().numpy()
    else:
        arr = np.asarray(values)
    side = grid_side(arr.shape[-1])
    return arr.reshape(*arr.shape[:-1], side, side)


def finite_range(arr: np.ndarray, signed: bool = False) -> tuple[float, float]:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return (-1, 1) if signed else (0, 1)
    if signed:
        vmax = max(float(np.percentile(np.abs(x), 99.0)), 1e-12)
        return -vmax, vmax
    return float(np.min(x)), max(float(np.percentile(x, 99.0)), float(np.min(x)) + 1e-12)


def save_map_overlay(
    rgb: np.ndarray,
    grid: np.ndarray,
    path: Path,
    *,
    title: str,
    cmap: str = "magma",
    signed: bool = False,
    alpha: float = 0.55,
) -> None:
    ensure_dir(path.parent)
    vmin, vmax = finite_range(grid, signed=signed)
    fig, ax = plt.subplots(figsize=(4.7, 4.7), dpi=150)
    ax.imshow(rgb)
    ax.imshow(
        grid,
        cmap=cmap,
        alpha=alpha,
        interpolation="nearest",
        extent=(0, rgb.shape[1], rgb.shape[0], 0),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    fig.tight_layout(pad=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_gipu_panel(
    rgb: np.ndarray,
    maps: Mapping[str, np.ndarray],
    path: Path,
    *,
    title: str,
) -> None:
    ensure_dir(path.parent)
    specs = [
        ("patch norm", maps["norm"], "viridis", False),
        ("tracked register lineage", maps["register_mask"], "Reds", False),
        ("fixed B23 μ1 cosine", maps["mu1_cos"], "coolwarm", True),
        ("scratchpad gap", maps["scratch_gap"], "coolwarm", True),
        (maps.get("semantic_title", "semantic grad×attn"), maps["semantic_gradattn"], "magma", False),
    ]
    fig, axes = plt.subplots(1, 1 + len(specs), figsize=(3.0 * (1 + len(specs)), 3.25), dpi=160)
    axes = np.atleast_1d(axes)
    axes[0].imshow(rgb)
    axes[0].set_title("input", fontsize=9)
    axes[0].set_axis_off()
    for ax, (name, grid, cmap, signed) in zip(axes[1:], specs):
        vmin, vmax = finite_range(grid, signed=signed)
        ax.imshow(rgb)
        ax.imshow(
            grid,
            cmap=cmap,
            alpha=0.56,
            interpolation="nearest",
            extent=(0, rgb.shape[1], rgb.shape[0], 0),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(name, fontsize=8)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _role_sorted_patch_order(
    register_mask: np.ndarray,
    cache_by_block: np.ndarray,
    scratch_by_block: np.ndarray,
    mu1_cos_by_block: np.ndarray,
    scratch_gap_by_block: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Return a stable patch order for the timing-diagram view.

    Group precedence is deliberately simple and fixed for the whole B0..B23 strip:
      1) eventual/tracked B23 register lineage;
      2) non-register patches that enter the mu-cache at least once;
      3) remaining patches that become scratchpad candidates at least once;
      4) everything else.

    Within the cache group, sort by maximum fixed-mu1 cosine over time.  Within the
    scratch group, sort by maximum finite scratchpad gap.  This is ONLY a display order;
    the raw arrays retain original spatial patch indexing.
    """
    reg = np.asarray(register_mask, dtype=bool).reshape(-1)
    cache = np.asarray(cache_by_block, dtype=bool)
    scratch = np.asarray(scratch_by_block, dtype=bool)
    mu = np.asarray(mu1_cos_by_block, dtype=np.float32)
    gap = np.asarray(scratch_gap_by_block, dtype=np.float32)
    patch_count = reg.size

    ever_cache = cache.any(axis=0) & ~reg
    ever_scratch = scratch.any(axis=0) & ~reg & ~ever_cache
    other = ~reg & ~ever_cache & ~ever_scratch

    def finite_max(a: np.ndarray, axis: int = 0) -> np.ndarray:
        finite = np.where(np.isfinite(a), a, -np.inf)
        out = finite.max(axis=axis)
        out[~np.isfinite(out)] = -np.inf
        return out

    mu_max = finite_max(mu)
    gap_max = finite_max(gap)

    reg_idx = np.flatnonzero(reg)
    cache_idx = np.flatnonzero(ever_cache)
    scratch_idx = np.flatnonzero(ever_scratch)
    other_idx = np.flatnonzero(other)

    if reg_idx.size:
        reg_idx = reg_idx[np.argsort(-mu_max[reg_idx], kind="stable")]
    if cache_idx.size:
        cache_idx = cache_idx[np.argsort(-mu_max[cache_idx], kind="stable")]
    if scratch_idx.size:
        scratch_idx = scratch_idx[np.argsort(-gap_max[scratch_idx], kind="stable")]
    # Keep ordinary patches in original spatial order; this preserves a weak spatial cue.

    groups = [
        ("register", reg_idx),
        ("mu-cache", cache_idx),
        ("scratchpad", scratch_idx),
        ("other", other_idx),
    ]
    order_parts = [idx for _, idx in groups if idx.size]
    order = np.concatenate(order_parts) if order_parts else np.arange(patch_count)

    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for name, idx in groups:
        if not idx.size:
            continue
        start = cursor
        cursor += int(idx.size)
        spans.append((name, start, cursor))
    return order.astype(np.int64), spans


def _semantic_log_display(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Log10 display transform for sparse positive grad*attention without per-block renorm."""
    x = np.asarray(values, dtype=np.float64)
    positive = x[np.isfinite(x) & (x > 0)]
    if not positive.size:
        return np.full_like(x, -12.0, dtype=np.float32), -12.0, -11.0
    # A floor several orders below the smallest robust positive value keeps exact zeros dark
    # while preserving across-block magnitude differences.
    ref = float(np.percentile(positive, 5.0))
    eps = max(ref * 1.0e-3, 1.0e-20)
    y = np.log10(np.clip(x, 0.0, None) + eps)
    finite = y[np.isfinite(y)]
    vmin = float(np.percentile(finite, 2.0))
    vmax = float(np.percentile(finite, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return y.astype(np.float32), vmin, vmax


def save_temporal_role_strip(
    *,
    variant: str,
    image_name: str,
    semantic_word: str,
    norm_by_block: np.ndarray,
    mu1_cos_by_block: np.ndarray,
    scratch_gap_by_block: np.ndarray,
    semantic_gradattn_by_block: np.ndarray,
    cache_by_block: np.ndarray,
    scratch_by_block: np.ndarray,
    register_mask: np.ndarray,
    order: np.ndarray,
    group_spans: Sequence[tuple[str, int, int]],
    order_name: str,
    path: Path,
) -> None:
    """Save one-image B0..B23 timing diagram: rows=patches, columns=blocks."""
    ensure_dir(path.parent)
    norm = np.asarray(norm_by_block, dtype=np.float32)[:, order].T
    mu = np.asarray(mu1_cos_by_block, dtype=np.float32)[:, order].T
    gap = np.asarray(scratch_gap_by_block, dtype=np.float32)[:, order].T
    sem = np.asarray(semantic_gradattn_by_block, dtype=np.float32)[:, order].T
    cache = np.asarray(cache_by_block, dtype=bool)[:, order].T
    scratch = np.asarray(scratch_by_block, dtype=bool)[:, order].T
    reg = np.asarray(register_mask, dtype=bool).reshape(-1)[order]

    # Dynamic operational role per patch/block.  Register lineage has precedence and is
    # intentionally constant over time; cache and scratchpad membership are block-local.
    role = np.zeros_like(cache, dtype=np.int8)
    role[scratch] = 1
    role[cache] = 2
    role[reg, :] = 3

    log_norm = np.log1p(np.clip(norm, 0.0, None))
    finite_norm = log_norm[np.isfinite(log_norm)]
    norm_vmin = float(np.percentile(finite_norm, 1.0)) if finite_norm.size else 0.0
    norm_vmax = float(np.percentile(finite_norm, 99.5)) if finite_norm.size else 1.0
    if norm_vmax <= norm_vmin:
        norm_vmax = norm_vmin + 1.0

    finite_gap = gap[np.isfinite(gap)]
    gap_vmax = float(np.percentile(np.abs(finite_gap), 99.0)) if finite_gap.size else 1.0
    gap_vmax = max(gap_vmax, 1.0e-6)

    sem_log, sem_vmin, sem_vmax = _semantic_log_display(sem)

    from matplotlib.colors import BoundaryNorm, ListedColormap
    role_cmap = ListedColormap(["#e8e8e8", "#c17c74", "#6c8ebf", "#3f2a78"])
    role_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], role_cmap.N)

    fig, axes = plt.subplots(5, 1, figsize=(10.6, 13.0), dpi=170, sharex=True)
    matrices = [
        (role, role_cmap, role_norm, None, "operational role"),
        (log_norm, "viridis", None, (norm_vmin, norm_vmax), "log(1 + residual L2 norm)"),
        (mu, "coolwarm", None, (-1.0, 1.0), "cosine to fixed B23 mu1"),
        (gap, "coolwarm", None, (-gap_vmax, gap_vmax), "scratchpad gap: within-distant - cross-image"),
        (sem_log, "magma", None, (sem_vmin, sem_vmax), f"log10 positive grad x attn: {semantic_word}"),
    ]

    ims = []
    for ax, (matrix, cmap, norm_obj, limits, ylabel) in zip(axes, matrices):
        kwargs: dict[str, Any] = {
            "aspect": "auto",
            "interpolation": "nearest",
            "origin": "upper",
            "cmap": cmap,
        }
        if norm_obj is not None:
            kwargs["norm"] = norm_obj
        elif limits is not None:
            kwargs["vmin"], kwargs["vmax"] = limits
        im = ax.imshow(matrix, **kwargs)
        ims.append(im)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        for _, _, stop in group_spans[:-1]:
            ax.axhline(stop - 0.5, color="black", linewidth=0.45, alpha=0.45)

    # Role legend as categorical colorbar.
    cbar0 = fig.colorbar(ims[0], ax=axes[0], fraction=0.018, pad=0.015, ticks=[0, 1, 2, 3])
    cbar0.ax.set_yticklabels(["other", "scratch", "cache", "register"])
    cbar0.ax.tick_params(labelsize=7)
    for ax, im in zip(axes[1:], ims[1:]):
        cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015)
        cb.ax.tick_params(labelsize=7)

    if group_spans:
        # Tiny register/cache groups can be only 1-4 rows tall, so putting categorical
        # tick labels at their row midpoints makes them collide.  Keep the separator lines
        # in the raster and summarize the row groups once above the first panel instead.
        group_summary = " | ".join(
            f"{name}={stop-start}" for name, start, stop in group_spans
        )
        axes[0].text(
            0.0, 1.018, f"row groups: {group_summary}",
            transform=axes[0].transAxes, ha="left", va="bottom", fontsize=7,
        )
        for ax in axes:
            ax.set_yticks([])
    else:
        patch_count = len(order)
        ticks = np.linspace(0, patch_count - 1, 5).round().astype(int)
        for ax in axes:
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(int(order[t])) for t in ticks], fontsize=7)

    n_blocks = norm_by_block.shape[0]
    axes[-1].set_xticks(np.arange(n_blocks))
    axes[-1].set_xticklabels([f"B{i}" for i in range(n_blocks)], rotation=0, fontsize=7)
    axes[-1].set_xlabel("transformer block (time)")
    fig.suptitle(
        f"{variant} | {Path(image_name).stem} | temporal token roles | {order_name}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_temporal_role_strips(
    *,
    variant: str,
    samples: Sequence[DemoSample],
    root: Path,
    register_masks: torch.Tensor,
    cache_maps: Mapping[int, Mapping[str, torch.Tensor]],
    scratch_maps: Mapping[int, Mapping[str, torch.Tensor]],
    semantic_temporal: Mapping[str, Mapping[str, Any]],
) -> None:
    """Build one timing diagram per image in spatial and role-sorted patch order."""
    blocks = sorted(set(cache_maps) & set(scratch_maps))
    if not blocks:
        return
    if blocks != list(range(max(blocks) + 1)):
        print(f"[warn] {variant}: temporal strip has non-contiguous blocks {blocks}")

    for image_index, sample in enumerate(samples):
        sem_entry = semantic_temporal.get(sample.spec.filename)
        if sem_entry is None:
            print(f"[warn] {variant} {sample.spec.filename}: missing semantic temporal attribution")
            continue

        norm = np.stack([cache_maps[b]["norm"][image_index].numpy() for b in blocks], axis=0)
        mu = np.stack([cache_maps[b]["mu1_cos"][image_index].numpy() for b in blocks], axis=0)
        cache = np.stack([cache_maps[b]["cache_mask"][image_index].numpy() for b in blocks], axis=0).astype(bool)
        gap = np.stack([scratch_maps[b]["gap"][image_index].numpy() for b in blocks], axis=0)
        scratch = np.stack([scratch_maps[b]["candidate_mask"][image_index].numpy() for b in blocks], axis=0).astype(bool)
        semantic = np.asarray(sem_entry["positive_gradattn_by_block"], dtype=np.float32)
        semantic_word = str(sem_entry["word"])
        reg = register_masks[image_index].numpy().astype(bool)

        if semantic.shape != norm.shape:
            raise RuntimeError(
                f"{variant} {sample.spec.filename}: temporal semantic shape {semantic.shape} "
                f"!= structural shape {norm.shape}"
            )

        spatial_order = np.arange(norm.shape[1], dtype=np.int64)
        role_order, role_spans = _role_sorted_patch_order(reg, cache, scratch, mu, gap)

        # For row-major spatial order, label by patch index rather than imposing role-group separators.
        npz(
            root / "temporal_role_strips" / "raw" / variant / f"{sample.stem}.npz",
            blocks=np.asarray(blocks, dtype=np.int16),
            patch_norm_by_block=norm,
            mu1_cos_by_block=mu,
            cache_mask_by_block=cache.astype(np.uint8),
            scratch_gap_by_block=gap,
            scratch_mask_by_block=scratch.astype(np.uint8),
            semantic_positive_gradattn_by_block=semantic,
            semantic_word=np.asarray(semantic_word),
            register_mask=reg.astype(np.uint8),
            spatial_order=spatial_order,
            role_sorted_order=role_order,
            role_group_names=np.asarray([name for name, _, _ in role_spans]),
            role_group_starts=np.asarray([start for _, start, _ in role_spans], dtype=np.int16),
            role_group_stops=np.asarray([stop for _, _, stop in role_spans], dtype=np.int16),
        )

        save_temporal_role_strip(
            variant=variant,
            image_name=sample.spec.filename,
            semantic_word=semantic_word,
            norm_by_block=norm,
            mu1_cos_by_block=mu,
            scratch_gap_by_block=gap,
            semantic_gradattn_by_block=semantic,
            cache_by_block=cache,
            scratch_by_block=scratch,
            register_mask=reg,
            order=spatial_order,
            group_spans=(),
            order_name="spatial patch order",
            path=root / "temporal_role_strips" / variant / "spatial_order" / f"{sample.stem}.png",
        )
        save_temporal_role_strip(
            variant=variant,
            image_name=sample.spec.filename,
            semantic_word=semantic_word,
            norm_by_block=norm,
            mu1_cos_by_block=mu,
            scratch_gap_by_block=gap,
            semantic_gradattn_by_block=semantic,
            cache_by_block=cache,
            scratch_by_block=scratch,
            register_mask=reg,
            order=role_order,
            group_spans=role_spans,
            order_name="role-sorted patch order",
            path=root / "temporal_role_strips" / variant / "role_sorted" / f"{sample.stem}.png",
        )


def processor_pixels(processor: Any, samples: Sequence[DemoSample], device: torch.device) -> torch.Tensor:
    encoded = processor(images=[sample.image for sample in samples], return_tensors="pt")
    return encoded["pixel_values"].to(device)


def pixels_to_rgb(pixel_values: torch.Tensor, processor: Any) -> list[np.ndarray]:
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
    return list(rgb.clamp(0, 1).permute(0, 2, 3, 1).numpy())


def tokenize_prompts(processor: Any, template: str, device: torch.device) -> torch.Tensor:
    texts = [template.format(prompt_word=word) for word in PROMPT_WORDS]
    encoded = processor(
        text=texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device)


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


# =================================================================================================
# Bare-vanilla reconstruction
# =================================================================================================

def build_bare_vanilla_clip(source_model: torch.nn.Module) -> CLIPModel:
    """Build literal stock HF CLIP from the jointly-trained x-attn checkpoint."""
    source_config = source_model.config
    text_config = dict(source_config.text_config.to_dict())
    source_vocab = int(text_config["vocab_size"])
    if source_vocab < VANILLA_CLIP_VOCAB_SIZE:
        raise RuntimeError(f"Source vocab {source_vocab} < vanilla CLIP vocab {VANILLA_CLIP_VOCAB_SIZE}")
    text_config["vocab_size"] = VANILLA_CLIP_VOCAB_SIZE
    vision_config = dict(source_config.vision_config.to_dict())
    config = CLIPConfig(
        text_config=text_config,
        vision_config=vision_config,
        projection_dim=int(source_config.projection_dim),
        logit_scale_init_value=float(source_config.logit_scale_init_value),
    )
    target = CLIPModel(config)
    src = source_model.state_dict()
    template = target.state_dict()
    copied: dict[str, torch.Tensor] = {}
    for key, tgt in template.items():
        if key not in src:
            raise KeyError(f"Stock CLIP tensor missing from x-attn checkpoint: {key}")
        value = src[key]
        if tuple(value.shape) == tuple(tgt.shape):
            copied[key] = value.detach().cpu().to(tgt.dtype)
        elif key == "text_model.embeddings.token_embedding.weight" and value.shape[0] >= tgt.shape[0]:
            copied[key] = value[:tgt.shape[0]].detach().cpu().to(tgt.dtype).contiguous()
        else:
            raise RuntimeError(
                f"Unexpected stock tensor shape mismatch {key}: source={tuple(value.shape)} target={tuple(tgt.shape)}"
            )
    target.load_state_dict(copied, strict=True)
    forbidden = ("read_implant", "read_null", "hard_text_embedding", "null_text_embedding", "content_pool")
    leaked = [k for k in target.state_dict() if any(x in k for x in forbidden)]
    if leaked:
        raise RuntimeError(f"Custom keys leaked into trained_vanilla: {leaked[:10]}")
    print(
        f"[vanilla] reconstructed stock CLIP: vocab {source_vocab}->{VANILLA_CLIP_VOCAB_SIZE}; "
        f"RN={hasattr(target, 'read_null_token')}; bridge={hasattr(target, 'read_implant')}"
    )
    return target


# =================================================================================================
# Exact differentiable HF CLIP visual layer
# =================================================================================================

def validate_clip_layer(layer: Any) -> None:
    for name in ("layer_norm1", "layer_norm2", "self_attn", "mlp"):
        if not hasattr(layer, name):
            raise TypeError(f"Expected CLIPEncoderLayer.{name}")
    for name in ("q_proj", "k_proj", "v_proj", "out_proj", "num_heads"):
        if not hasattr(layer.self_attn, name):
            raise TypeError(f"Expected CLIPAttention.{name}")


def manual_layer(layer: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard HF CLIPEncoderLayer forward, returning post-softmax attention [B,H,T,T]."""
    validate_clip_layer(layer)
    residual = hidden_states
    x = layer.layer_norm1(hidden_states)
    attn = layer.self_attn
    batch, tokens, _ = x.shape
    heads = int(attn.num_heads)
    q = F.linear(x, attn.q_proj.weight, attn.q_proj.bias)
    k = F.linear(x, attn.k_proj.weight, attn.k_proj.bias)
    v = F.linear(x, attn.v_proj.weight, attn.v_proj.bias)
    head_dim = q.shape[-1] // heads
    scale = float(getattr(attn, "scale", head_dim ** -0.5))
    q = (q * scale).view(batch, tokens, heads, head_dim).transpose(1, 2)
    k = k.view(batch, tokens, heads, head_dim).transpose(1, 2)
    v = v.view(batch, tokens, heads, head_dim).transpose(1, 2)
    probs = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
    dropout_p = float(getattr(attn, "dropout", 0.0))
    used = F.dropout(probs, p=dropout_p, training=False) if dropout_p > 0 else probs
    context = torch.matmul(used, v)
    context = context.transpose(1, 2).reshape(batch, tokens, heads * head_dim)
    attn_out = F.linear(context, attn.out_proj.weight, attn.out_proj.bias)
    hidden = residual + attn_out
    hidden = hidden + layer.mlp(layer.layer_norm2(hidden))
    return hidden, probs


def set_eager_attention(model: torch.nn.Module) -> None:
    for layer in model.vision_model.encoder.layers:
        attn = layer.self_attn
        if hasattr(attn, "config"):
            attn.config._attn_implementation = "eager"


def manual_visual_forward(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    insert_rn: bool,
    require_graph: bool,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Return projected image embedding, per-block attention, per-block post states."""
    vision = model.vision_model
    hidden = vision.embeddings(pixel_values, interpolate_pos_encoding=False)
    hidden = vision.pre_layrnorm(hidden)
    if require_graph and not hidden.requires_grad:
        hidden = hidden.detach().requires_grad_(True)

    rn_token = getattr(model, "read_null_token", None)
    insert_block = int(getattr(model.config, "read_null_insert_block", -1))
    if insert_rn and rn_token is None:
        raise RuntimeError("insert_rn=True but model has no read_null_token")

    attentions: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    inserted = False
    for block, layer in enumerate(vision.encoder.layers):
        if insert_rn and block == insert_block:
            rn = rn_token.to(device=hidden.device, dtype=hidden.dtype)
            hidden = torch.cat((hidden, rn.view(1, 1, -1).expand(hidden.shape[0], 1, -1)), dim=1)
            inserted = True
        hidden, probs = manual_layer(layer, hidden)
        attentions.append(probs)
        states.append(hidden)
    if insert_rn and not inserted:
        raise RuntimeError("READ_NULL was never inserted")
    pooled = vision.post_layernorm(hidden[:, 0, :])
    projected = model.visual_projection(pooled)
    return projected, attentions, states


@torch.no_grad()
def official_final_tokens(model: torch.nn.Module, pixel_values: torch.Tensor, *, insert_rn: bool) -> torch.Tensor:
    if insert_rn:
        if hasattr(model, "_vision_with_intermediates"):
            info = model._vision_with_intermediates(pixel_values, return_final_tokens=True)
            return info["final_tokens"]
        raise TypeError("RN variant lacks _vision_with_intermediates for validation")
    outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
    return outputs.last_hidden_state


def validate_manual_visual(model: torch.nn.Module, pixel_values: torch.Tensor, *, insert_rn: bool) -> dict[str, float]:
    with torch.no_grad():
        _, _, states = manual_visual_forward(model, pixel_values, insert_rn=insert_rn, require_graph=False)
        manual = states[-1].float()
        official = official_final_tokens(model, pixel_values, insert_rn=insert_rn).float()
        diff = (manual - official).abs()
        cos = F.cosine_similarity(manual.reshape(manual.shape[0], -1), official.reshape(official.shape[0], -1), dim=-1)
    result = {
        "max_abs": as_float(diff.max()),
        "mean_abs": as_float(diff.mean()),
        "cosine": as_float(cos.mean()),
    }
    if result["max_abs"] > 1e-3 or result["cosine"] < 0.999999:
        raise RuntimeError("Manual visual forward mismatch: " + json.dumps(result))
    return result


@torch.no_grad()
def encode_text_embeddings(model: torch.nn.Module, processor: Any, template: str, device: torch.device) -> torch.Tensor:
    input_ids = tokenize_prompts(processor, template, device)
    if hasattr(model, "_encode_text_hidden"):
        ids = model._pad_context(input_ids)
        ids = model._compact_control_tokens(
            ids,
            (
                model.config.hard_text_token_id,
                model.config.no_text_token_id,
                model.config.any_text_token_id,
            ),
        )
        embedding = model._encode_text_hidden(ids)["text_embedding"]
    else:
        text_out = model.text_model(input_ids=input_ids, return_dict=True)
        embedding = model.text_projection(text_out.pooler_output)
    return normalize_rows(embedding).detach()


# =================================================================================================
# Register masks, invariant subspace, cache, scratchpads
# =================================================================================================

def pairwise_cosine_summary(x: torch.Tensor) -> dict[str, float]:
    if x.shape[0] < 2:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    norms = x.norm(dim=-1)
    valid = norms > 1e-10
    x = x[valid]
    if x.shape[0] < 2:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    z = F.normalize(x, dim=-1)
    sim = z @ z.t()
    tri = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1)
    vals = sim[tri[0], tri[1]].detach().float().cpu().numpy()
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def orient_pc1(basis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if basis.shape[0] and torch.dot(basis[0], x.mean(dim=0)) < 0:
        basis = basis.clone()
        basis[0] = -basis[0]
    return basis


def fit_uncentered_basis(x: torch.Tensor, max_rank: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows x features -> orthonormal right singular directions [K,D], singular values."""
    if x.shape[0] < 1:
        raise ValueError("Cannot fit basis to empty matrix")
    _, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
    k = min(max_rank, vh.shape[0])
    basis = orient_pc1(vh[:k].contiguous(), x.float())
    return basis, s


def remove_subspace(x: torch.Tensor, basis: torch.Tensor, rank: int) -> torch.Tensor:
    rank = min(rank, basis.shape[0])
    if rank <= 0:
        return x
    b = basis[:rank]
    return x - (x @ b.t()) @ b


def spatial_distance_matrix(side: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij")
    coords = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1).float()
    return torch.cdist(coords, coords, p=2)


def edge_mask(side: int, ring: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij")
    mask = (yy < ring) | (xx < ring) | (yy >= side - ring) | (xx >= side - ring)
    return mask.reshape(-1)


def register_means_for_block(
    patch_states: torch.Tensor,
    register_masks: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    rows: list[torch.Tensor] = []
    indices: list[int] = []
    for i in range(patch_states.shape[0]):
        mask = register_masks[i]
        if bool(mask.any()):
            rows.append(patch_states[i, mask].mean(dim=0))
            indices.append(i)
    if not rows:
        return torch.empty((0, patch_states.shape[-1]), dtype=patch_states.dtype), indices
    return torch.stack(rows, dim=0), indices


def analyze_invariant_subspace(
    variant: str,
    states_by_block: Sequence[torch.Tensor],
    register_masks: torch.Tensor,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], dict[int, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    bases_np: dict[int, np.ndarray] = {}
    bases_t: dict[int, torch.Tensor] = {}
    for block, states in enumerate(states_by_block):
        means, valid_images = register_means_for_block(states.float(), register_masks)
        if means.shape[0] < 2:
            rows.append({"variant": variant, "block": block, "valid_images": means.shape[0]})
            continue
        basis, s = fit_uncentered_basis(means, max_rank=3)
        bases_t[block] = basis
        bases_np[block] = basis.cpu().numpy()
        energy = s.square()
        energy = energy / energy.sum().clamp_min(1e-20)
        row: dict[str, Any] = {
            "variant": variant,
            "block": block,
            "valid_images": int(means.shape[0]),
            "register_mean_norm_mean": as_float(means.norm(dim=-1).mean()),
            "pc1_energy": as_float(energy[0]) if energy.numel() > 0 else float("nan"),
            "pc12_energy": as_float(energy[:2].sum()) if energy.numel() >= 2 else float("nan"),
            "pc123_energy": as_float(energy[:3].sum()) if energy.numel() >= 3 else float("nan"),
        }
        for rank in range(4):
            residual = remove_subspace(means, basis, rank)
            summary = pairwise_cosine_summary(residual)
            row[f"pair_cos_rank{rank}_mean"] = summary["mean"]
            row[f"pair_cos_rank{rank}_median"] = summary["median"]
            row[f"pair_cos_rank{rank}_min"] = summary["min"]
            row[f"pair_cos_rank{rank}_max"] = summary["max"]
            row[f"residual_norm_rank{rank}_mean"] = as_float(residual.norm(dim=-1).mean())
        rows.append(row)
        npz(
            root / "register_invariance" / "basis" / variant / f"B{block:02d}.npz",
            basis=basis.cpu().numpy(),
            singular_values=s.cpu().numpy(),
            register_means=means.cpu().numpy(),
            valid_image_indices=np.asarray(valid_images, dtype=np.int64),
        )
    return rows, bases_np, bases_t


def compute_cache_maps(
    states_by_block: Sequence[torch.Tensor],
    register_masks: torch.Tensor,
    final_mu1: torch.Tensor,
    *,
    register_threshold: float,
    cache_cos_threshold: float,
    edge_ring: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, torch.Tensor]]]:
    patch_count = states_by_block[0].shape[1]
    side = grid_side(patch_count)
    edge = edge_mask(side, edge_ring, torch.device("cpu"))
    edge_area_fraction = as_float(edge.float().mean())
    rows: list[dict[str, Any]] = []
    maps: dict[int, dict[str, torch.Tensor]] = {}
    mu = F.normalize(final_mu1.float(), dim=0)
    for block, states in enumerate(states_by_block):
        x = states.float()
        norms = x.norm(dim=-1)
        cos = torch.einsum("bpd,d->bp", F.normalize(x, dim=-1), mu)
        candidate = (~register_masks) & (norms < register_threshold) & (cos >= cache_cos_threshold)
        maps[block] = {"norm": norms, "mu1_cos": cos, "cache_mask": candidate}
        for i in range(x.shape[0]):
            cand = candidate[i]
            count = int(cand.sum())
            edge_count = int((cand & edge).sum())
            edge_frac = edge_count / count if count else float("nan")
            rows.append(
                {
                    "block": block,
                    "image_index": i,
                    "cache_count": count,
                    "cache_fraction": count / patch_count,
                    "cache_cos_mean": as_float(cos[i, cand].mean()) if count else float("nan"),
                    "cache_cos_max": as_float(cos[i].max()),
                    "cache_edge_count": edge_count,
                    "cache_edge_fraction": edge_frac,
                    "edge_area_fraction": edge_area_fraction,
                    "cache_edge_enrichment": edge_frac / edge_area_fraction if count and edge_area_fraction > 0 else float("nan"),
                }
            )
    return rows, maps


def compute_scratchpad_maps(
    states_by_block: Sequence[torch.Tensor],
    register_masks: torch.Tensor,
    cache_maps: Mapping[int, Mapping[str, torch.Tensor]],
    local_bases: Mapping[int, torch.Tensor],
    *,
    min_distance: float,
    within_threshold: float,
    gap_threshold: float,
    topk: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, torch.Tensor]]]:
    patch_count = states_by_block[0].shape[1]
    side = grid_side(patch_count)
    distance = spatial_distance_matrix(side, torch.device("cpu"))
    distant = distance >= float(min_distance)
    rows: list[dict[str, Any]] = []
    maps: dict[int, dict[str, torch.Tensor]] = {}

    for block, states in enumerate(states_by_block):
        if block not in local_bases or local_bases[block].shape[0] < 2:
            continue
        basis = local_bases[block][:2].float()
        residual = remove_subspace(states.float(), basis, 2)
        z = F.normalize(residual, dim=-1)
        cache_mask = cache_maps[block]["cache_mask"]
        eligible = (~register_masks) & (~cache_mask)
        n_images = z.shape[0]

        within_all = torch.full((n_images, patch_count), float("nan"))
        cross_all = torch.full_like(within_all, float("nan"))
        gap_all = torch.full_like(within_all, float("nan"))
        partner_all = torch.full((n_images, patch_count), -1, dtype=torch.long)
        partner_dist_all = torch.full((n_images, patch_count), float("nan"))
        candidate_all = torch.zeros((n_images, patch_count), dtype=torch.bool)

        for i in range(n_images):
            zi = z[i]
            valid = eligible[i]
            sim = zi @ zi.t()
            allowed = distant.clone()
            allowed[:, ~valid] = False
            allowed[~valid, :] = False
            masked = sim.masked_fill(~allowed, -torch.inf)
            within, partner = masked.max(dim=-1)
            source_has_partner = torch.isfinite(within) & valid
            within = torch.where(source_has_partner, within, torch.full_like(within, float("nan")))
            partner_safe = partner.clamp_min(0)
            p_dist = distance[torch.arange(patch_count), partner_safe]
            p_dist = torch.where(source_has_partner, p_dist, torch.full_like(p_dist, float("nan")))

            other_vectors: list[torch.Tensor] = []
            for j in range(n_images):
                if j == i:
                    continue
                vj = eligible[j]
                if bool(vj.any()):
                    other_vectors.append(z[j, vj])
            if other_vectors:
                other = torch.cat(other_vectors, dim=0)
                cross = (zi @ other.t()).max(dim=-1).values
                cross = torch.where(valid, cross, torch.full_like(cross, float("nan")))
            else:
                cross = torch.full((patch_count,), float("nan"))

            gap = within - cross
            candidate = valid & torch.isfinite(gap) & (within >= within_threshold) & (gap >= gap_threshold)

            within_all[i] = within
            cross_all[i] = cross
            gap_all[i] = gap
            partner_all[i] = torch.where(source_has_partner, partner, torch.full_like(partner, -1))
            partner_dist_all[i] = p_dist
            candidate_all[i] = candidate

            valid_gap = gap[valid & torch.isfinite(gap)]
            k = min(topk, int(valid_gap.numel()))
            if k > 0:
                top_values = torch.topk(valid_gap, k=k).values
                top_gap_mean = as_float(top_values.mean())
                top_gap_min = as_float(top_values.min())
            else:
                top_gap_mean = float("nan")
                top_gap_min = float("nan")
            cand_count = int(candidate.sum())
            rows.append(
                {
                    "block": block,
                    "image_index": i,
                    "eligible_count": int(valid.sum()),
                    "scratch_candidate_count": cand_count,
                    "scratch_candidate_fraction": cand_count / max(int(valid.sum()), 1),
                    "within_distant_mean": as_float(within[valid & torch.isfinite(within)].mean())
                    if bool((valid & torch.isfinite(within)).any()) else float("nan"),
                    "cross_image_max_mean": as_float(cross[valid & torch.isfinite(cross)].mean())
                    if bool((valid & torch.isfinite(cross)).any()) else float("nan"),
                    "gap_mean": as_float(gap[valid & torch.isfinite(gap)].mean())
                    if bool((valid & torch.isfinite(gap)).any()) else float("nan"),
                    "topk_gap_mean": top_gap_mean,
                    "topk_gap_min": top_gap_min,
                    "candidate_partner_distance_mean": as_float(p_dist[candidate].mean()) if cand_count else float("nan"),
                }
            )

        maps[block] = {
            "within": within_all,
            "cross": cross_all,
            "gap": gap_all,
            "partner": partner_all,
            "partner_distance": partner_dist_all,
            "candidate_mask": candidate_all,
        }
    return rows, maps


# =================================================================================================
# Raw attention and gradient*attention attribution
# =================================================================================================

def raw_attention_metrics(
    variant: str,
    attentions: Sequence[torch.Tensor],
    register_masks: torch.Tensor,
    sample_names: Sequence[str],
    *,
    patch_count: int,
    insert_rn: bool,
    rn_insert_block: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block, attn in enumerate(attentions):
        a = attn.detach().float().cpu()
        for i, name in enumerate(sample_names):
            reg = register_masks[i]
            for head in range(a.shape[1]):
                cls_patch = a[i, head, 0, 1:1 + patch_count]
                cls_den = cls_patch.sum().clamp_min(1e-20)
                patch_to_patch = a[i, head, 1:1 + patch_count, 1:1 + patch_count]
                incoming_den = patch_to_patch.sum().clamp_min(1e-20)
                row = {
                    "variant": variant,
                    "image": name,
                    "block": block,
                    "head": head,
                    "tracked_register_count": int(reg.sum()),
                    "cls_patch_mass": as_float(cls_patch.sum()),
                    "cls_register_mass": as_float(cls_patch[reg].sum()) if bool(reg.any()) else 0.0,
                    "cls_register_fraction_of_patch_attention": as_float(cls_patch[reg].sum() / cls_den)
                    if bool(reg.any()) else 0.0,
                    "patch_queries_register_incoming_fraction": as_float(patch_to_patch[:, reg].sum() / incoming_den)
                    if bool(reg.any()) else 0.0,
                }
                if bool(reg.any()):
                    reg_queries = patch_to_patch[reg]
                    reg_den = reg_queries.sum().clamp_min(1e-20)
                    row["register_queries_to_register_fraction"] = as_float(reg_queries[:, reg].sum() / reg_den)
                else:
                    row["register_queries_to_register_fraction"] = float("nan")
                if insert_rn and block >= rn_insert_block:
                    row["cls_to_rn"] = as_float(a[i, head, 0, -1])
                    row["rn_to_cls"] = as_float(a[i, head, -1, 0])
                    row["rn_to_rn"] = as_float(a[i, head, -1, -1])
                rows.append(row)
    return rows


def attribution_for_variant(
    variant: str,
    model: torch.nn.Module,
    processor: Any,
    samples: Sequence[DemoSample],
    root: Path,
    register_masks: torch.Tensor,
    *,
    insert_rn: bool,
    prompt_template: str,
    device: torch.device,
    patch_count: int,
    map_blocks: set[int],
    selected_maps: dict[tuple[str, int], dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    text_norm = encode_text_embeddings(model, processor, prompt_template, device)
    word_index = {w: i for i, w in enumerate(PROMPT_WORDS)}
    rows: list[dict[str, Any]] = []
    semantic_temporal: dict[str, dict[str, Any]] = {}

    original_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        for image_index, sample in enumerate(samples):
            pixel = processor_pixels(processor, [sample], device)
            image_embedding, probs_list, _ = manual_visual_forward(
                model, pixel, insert_rn=insert_rn, require_graph=True
            )
            image_norm = normalize_rows(image_embedding)
            cosines = image_norm @ text_norm.t()
            target_words = list(dict.fromkeys((*sample.spec.present_words, *sample.spec.text_words)))
            if not target_words:
                target_words = [PROMPT_WORDS[int(cosines[0].argmax())]]

            for target_number, word in enumerate(target_words):
                score = cosines[0, word_index[word]]
                grads = torch.autograd.grad(
                    score,
                    probs_list,
                    retain_graph=target_number < len(target_words) - 1,
                    create_graph=False,
                    allow_unused=False,
                )
                reg = register_masks[image_index].to(device)
                gradattn_by_block: list[np.ndarray] = []
                for block, (probs, grad) in enumerate(zip(probs_list, grads)):
                    gx = (probs * grad).clamp(min=0).mean(dim=1)[0, 0, 1:1 + patch_count]
                    gradattn_by_block.append(gx.detach().float().cpu().numpy())
                    total = gx.sum()
                    reg_mass = gx[reg].sum() if bool(reg.any()) else torch.zeros((), device=device)
                    fraction = reg_mass / total.clamp_min(1e-20)
                    rows.append(
                        {
                            "variant": variant,
                            "image": sample.spec.filename,
                            "word": word,
                            "block": block,
                            "is_known_present": int(word in sample.spec.present_words),
                            "is_text_word": int(word in sample.spec.text_words),
                            "classic_cosine": as_float(score),
                            "positive_gradattn_spatial_mass": as_float(total),
                            "positive_gradattn_register_mass": as_float(reg_mass),
                            "register_attribution_fraction": as_float(fraction),
                            "tracked_register_count": int(reg.sum()),
                        }
                    )
                    if (
                        block in map_blocks
                        and word == sample.spec.present_words[0]
                    ):
                        entry = selected_maps.setdefault((sample.spec.filename, block), {})
                        entry["semantic_gradattn"] = to_grid(gx.detach().cpu())
                        entry["semantic_title"] = f"grad×attn: {word}"
                gradattn_array = np.stack(gradattn_by_block, axis=0)
                npz(
                    root / "raw" / variant / "semantic_gradattn" / sample.stem / f"{word.replace(' ', '_')}.npz",
                    positive_gradattn_by_block=gradattn_array,
                    register_mask=register_masks[image_index].numpy(),
                    classic_cosine=np.asarray(as_float(score), dtype=np.float32),
                )
                # The temporal role strip uses the first annotated-present semantic prompt
                # as its stable foreground readout. Other present/text prompts remain in raw NPZs.
                if sample.spec.present_words and word == sample.spec.present_words[0]:
                    semantic_temporal[sample.spec.filename] = {
                        "word": word,
                        "positive_gradattn_by_block": gradattn_array,
                        "classic_cosine": as_float(score),
                    }
            del image_embedding, probs_list, cosines, pixel
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for p, flag in zip(model.parameters(), original_requires_grad):
            p.requires_grad_(flag)
    return rows, semantic_temporal


# =================================================================================================
# Variant analysis
# =================================================================================================

def analyze_variant(
    *,
    label: str,
    model: torch.nn.Module,
    processor: Any,
    samples: Sequence[DemoSample],
    root: Path,
    device: torch.device,
    batch_size: int,
    insert_rn: bool,
    register_threshold: float,
    cache_cos_threshold: float,
    edge_ring: int,
    scratch_min_distance: float,
    scratch_within_threshold: float,
    scratch_gap_threshold: float,
    scratch_topk: int,
    prompt_template: str,
    map_blocks: set[int],
) -> VariantResult:
    print(f"\n{'=' * 90}\n[variant] {label} | insert_rn={insert_rn}\n{'=' * 90}")
    model = model.eval().to(device).float()
    set_eager_attention(model)
    n_blocks = len(model.vision_model.encoder.layers)
    if n_blocks != 24:
        raise RuntimeError(f"Expected ViT-L/14 with 24 blocks, got {n_blocks}")
    patch_count = int(model.config.vision_config.num_channels)  # overwritten below; just sanity placeholder

    # Determine actual patch count from config image/patch sizes.
    image_size = int(model.config.vision_config.image_size)
    patch_size = int(model.config.vision_config.patch_size)
    side = image_size // patch_size
    patch_count = side * side
    rn_insert_block = int(getattr(model.config, "read_null_insert_block", -1))

    first_pixel = processor_pixels(processor, [samples[0]], device)
    validation = validate_manual_visual(model, first_pixel, insert_rn=insert_rn)
    write_json(root / "_meta" / f"manual_validation_{label}.json", validation)
    print(f"[validate] {validation}")
    del first_pixel

    states_chunks: list[list[torch.Tensor]] = [[] for _ in range(n_blocks)]
    register_mask_chunks: list[torch.Tensor] = []
    raw_rows: list[dict[str, Any]] = []
    rgb_lookup: dict[str, np.ndarray] = {}

    # Pass 1: collect all spatial states on CPU and raw attention metrics.
    with torch.inference_mode():
        for start, batch_samples in batches(samples, batch_size):
            print(f"[{label}] structural images {start + 1}-{start + len(batch_samples)} / {len(samples)}")
            pixel = processor_pixels(processor, batch_samples, device)
            rgbs = pixels_to_rgb(pixel, processor)
            for sample, rgb in zip(batch_samples, rgbs):
                rgb_lookup[sample.spec.filename] = rgb
            _, attentions, states = manual_visual_forward(model, pixel, insert_rn=insert_rn, require_graph=False)
            final_patches = states[-1][:, 1:1 + patch_count, :].float()
            final_norms = final_patches.norm(dim=-1)
            masks = (final_norms >= register_threshold).cpu()
            for local, sample in enumerate(batch_samples):
                count = int(masks[local].sum())
                if count == 0:
                    print(f"[warn] {label} {sample.spec.filename}: no B23 patch >= {register_threshold:g}")
            register_mask_chunks.append(masks)
            names = [s.spec.filename for s in batch_samples]
            raw_rows.extend(
                raw_attention_metrics(
                    label,
                    attentions,
                    masks,
                    names,
                    patch_count=patch_count,
                    insert_rn=insert_rn,
                    rn_insert_block=rn_insert_block,
                )
            )
            for block in range(n_blocks):
                spatial = states[block][:, 1:1 + patch_count, :].detach().float().cpu()
                states_chunks[block].append(spatial)
            del attentions, states, pixel
            if device.type == "cuda":
                torch.cuda.empty_cache()

    states_by_block = [torch.cat(chunks, dim=0) for chunks in states_chunks]
    register_masks = torch.cat(register_mask_chunks, dim=0).bool()
    register_masks_dict = {
        sample.spec.filename: register_masks[i].numpy().astype(np.uint8)
        for i, sample in enumerate(samples)
    }
    reg_rows = []
    for i, sample in enumerate(samples):
        final_norms = states_by_block[-1][i].norm(dim=-1)
        mask = register_masks[i]
        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        reg_rows.append(
            {
                "variant": label,
                "image": sample.spec.filename,
                "register_threshold": register_threshold,
                "register_count": int(mask.sum()),
                "register_indices_json": json.dumps(indices),
                "final_register_norm_mean": as_float(final_norms[mask].mean()) if bool(mask.any()) else float("nan"),
                "final_patch_norm_max": as_float(final_norms.max()),
            }
        )
    write_csv(root / "register_lineage" / label / "register_positions.csv", reg_rows)

    invariant_rows, bases_np, bases_t = analyze_invariant_subspace(
        label, states_by_block, register_masks, root
    )
    if (n_blocks - 1) not in bases_t or bases_t[n_blocks - 1].shape[0] < 2:
        raise RuntimeError(f"{label}: cannot fit rank-2 B23 invariant register basis")
    final_mu1 = bases_t[n_blocks - 1][0]

    cache_rows, cache_maps = compute_cache_maps(
        states_by_block,
        register_masks,
        final_mu1,
        register_threshold=register_threshold,
        cache_cos_threshold=cache_cos_threshold,
        edge_ring=edge_ring,
    )
    for row in cache_rows:
        row["variant"] = label
        row["image"] = samples[int(row.pop("image_index"))].spec.filename

    scratch_rows, scratch_maps = compute_scratchpad_maps(
        states_by_block,
        register_masks,
        cache_maps,
        bases_t,
        min_distance=scratch_min_distance,
        within_threshold=scratch_within_threshold,
        gap_threshold=scratch_gap_threshold,
        topk=scratch_topk,
    )
    for row in scratch_rows:
        row["variant"] = label
        row["image"] = samples[int(row.pop("image_index"))].spec.filename

    selected_maps: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    # Raw per-patch diagnostics are cheap enough to keep for ALL 24 blocks.  Figure spam is
    # restricted to --map-blocks, but GitHub/mechinterp users can regenerate any map later.
    for block in range(n_blocks):
        if block not in scratch_maps:
            continue
        for i, sample in enumerate(samples):
            npz(
                root / "raw" / label / f"B{block:02d}" / f"{sample.stem}.npz",
                patch_norm=cache_maps[block]["norm"][i].numpy(),
                register_mask=register_masks[i].numpy(),
                mu1_cos=cache_maps[block]["mu1_cos"][i].numpy(),
                cache_mask=cache_maps[block]["cache_mask"][i].numpy(),
                scratch_within=scratch_maps[block]["within"][i].numpy(),
                scratch_cross=scratch_maps[block]["cross"][i].numpy(),
                scratch_gap=scratch_maps[block]["gap"][i].numpy(),
                scratch_partner=scratch_maps[block]["partner"][i].numpy(),
                scratch_partner_distance=scratch_maps[block]["partner_distance"][i].numpy(),
                scratch_candidate_mask=scratch_maps[block]["candidate_mask"][i].numpy(),
            )

    for block in map_blocks:
        if block not in scratch_maps:
            continue
        for i, sample in enumerate(samples):
            key = (sample.spec.filename, block)
            entry = selected_maps.setdefault(key, {})
            entry["rgb"] = rgb_lookup[sample.spec.filename]
            entry["norm"] = to_grid(cache_maps[block]["norm"][i])
            entry["register_mask"] = to_grid(register_masks[i].float())
            entry["mu1_cos"] = to_grid(cache_maps[block]["mu1_cos"][i])
            entry["cache_mask"] = to_grid(cache_maps[block]["cache_mask"][i].float())
            entry["scratch_within"] = to_grid(scratch_maps[block]["within"][i])
            entry["scratch_cross"] = to_grid(scratch_maps[block]["cross"][i])
            entry["scratch_gap"] = to_grid(scratch_maps[block]["gap"][i])
            entry["scratch_mask"] = to_grid(scratch_maps[block]["candidate_mask"][i].float())

            map_root = root / "maps" / label / f"B{block:02d}" / sample.stem
            save_map_overlay(entry["rgb"], entry["norm"], map_root / "patch_norm.png", title=f"{label} | {sample.stem} | B{block} norm", cmap="viridis")
            save_map_overlay(entry["rgb"], entry["register_mask"], map_root / "tracked_register_lineage.png", title=f"{label} | {sample.stem} | B{block} tracked B23 register positions", cmap="Reds")
            save_map_overlay(entry["rgb"], entry["mu1_cos"], map_root / "mu1_cos.png", title=f"{label} | {sample.stem} | B{block} cos to fixed B23 μ1", cmap="coolwarm", signed=True)
            save_map_overlay(entry["rgb"], entry["cache_mask"], map_root / "mu_cache_mask.png", title=f"{label} | {sample.stem} | B{block} μ-cache candidates", cmap="Reds")
            save_map_overlay(entry["rgb"], entry["scratch_gap"], map_root / "scratchpad_gap.png", title=f"{label} | {sample.stem} | B{block} within-distant minus cross-image cosine", cmap="coolwarm", signed=True)
            save_map_overlay(entry["rgb"], entry["scratch_mask"], map_root / "scratchpad_candidates.png", title=f"{label} | {sample.stem} | B{block} scratchpad candidates", cmap="Reds")

    # Pass 2: prompt-conditioned positive grad*attention through every block.
    attribution_rows, semantic_temporal = attribution_for_variant(
        label,
        model,
        processor,
        samples,
        root,
        register_masks,
        insert_rn=insert_rn,
        prompt_template=prompt_template,
        device=device,
        patch_count=patch_count,
        map_blocks=map_blocks,
        selected_maps=selected_maps,
    )

    # A one-image timing diagram: rows are patch identities, columns are B0..B23.
    # Build this while all structural arrays are still resident, then free the state cache.
    build_temporal_role_strips(
        variant=label,
        samples=samples,
        root=root,
        register_masks=register_masks,
        cache_maps=cache_maps,
        scratch_maps=scratch_maps,
        semantic_temporal=semantic_temporal,
    )

    # Add semantic maps and build compact panels.
    for (image_name, block), entry in selected_maps.items():
        if "semantic_gradattn" not in entry:
            continue
        sample = next(s for s in samples if s.spec.filename == image_name)
        map_root = root / "maps" / label / f"B{block:02d}" / sample.stem
        save_map_overlay(
            entry["rgb"],
            entry["semantic_gradattn"],
            map_root / "semantic_gradattn.png",
            title=f"{label} | {sample.stem} | B{block} {entry['semantic_title']}",
            cmap="magma",
        )
        save_gipu_panel(
            entry["rgb"],
            entry,
            root / "paper_panels" / label / f"B{block:02d}" / f"{sample.stem}.png",
            title=f"{label} | {sample.stem} | B{block}",
        )

    # Full numerical arrays are available in the selected-block NPZs; tables cover every block.
    write_csv(root / "metrics" / f"{label}_register_invariance.csv", invariant_rows)
    write_csv(root / "metrics" / f"{label}_raw_attention.csv", raw_rows)
    write_csv(root / "metrics" / f"{label}_register_attribution.csv", attribution_rows)
    write_csv(root / "metrics" / f"{label}_mu_cache.csv", cache_rows)
    write_csv(root / "metrics" / f"{label}_scratchpads.csv", scratch_rows)

    # Free the ~300 MB CPU state cache before loading the next model variant.
    del states_by_block, states_chunks, cache_maps, scratch_maps, bases_t
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return VariantResult(
        label=label,
        invariant_rows=invariant_rows,
        raw_attention_rows=raw_rows,
        attribution_rows=attribution_rows,
        cache_rows=cache_rows,
        scratch_rows=scratch_rows,
        bases=bases_np,
        selected_maps=selected_maps,
        register_masks=register_masks_dict,
    )


# =================================================================================================
# Aggregate plots / model comparisons
# =================================================================================================

def nanmean_group(rows: Sequence[Mapping[str, Any]], key: str, block: int, predicate=None) -> float:
    values: list[float] = []
    for row in rows:
        if int(row.get("block", -1)) != block:
            continue
        if predicate is not None and not predicate(row):
            continue
        try:
            value = float(row[key])
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def plot_variant_curves(
    results: Sequence[VariantResult],
    root: Path,
    *,
    n_blocks: int = 24,
) -> None:
    blocks = np.arange(n_blocks)

    # 1) Main paper measurement: positive semantic grad*attention fraction on tracked registers.
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    for result in results:
        ys = [
            nanmean_group(
                result.attribution_rows,
                "register_attribution_fraction",
                b,
                predicate=lambda r: int(r.get("is_known_present", 0)) == 1,
            )
            for b in blocks
        ]
        ax.plot(blocks, ys, marker="o", markersize=3, label=result.label)
    ax.set_xlabel("ViT block")
    ax.set_ylabel("Register fraction of positive grad×attention")
    ax.set_title("Semantic score attribution to tracked late-register positions")
    ax.set_xticks(blocks)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    ensure_dir(root / "plots")
    fig.savefig(root / "plots" / "register_attribution_fraction_present_by_block.png", bbox_inches="tight")
    plt.close(fig)

    # 2) Raw CLS attention fraction to the same tracked positions.
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    for result in results:
        ys = [nanmean_group(result.raw_attention_rows, "cls_register_fraction_of_patch_attention", b) for b in blocks]
        ax.plot(blocks, ys, marker="o", markersize=3, label=result.label)
    ax.set_xlabel("ViT block")
    ax.set_ylabel("CLS attention fraction to tracked registers")
    ax.set_title("Raw CLS attention to tracked late-register positions")
    ax.set_xticks(blocks)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "plots" / "raw_cls_register_attention_fraction_by_block.png", bbox_inches="tight")
    plt.close(fig)

    # 3) Rank-removal curves per model.
    for result in results:
        lookup = {int(r["block"]): r for r in result.invariant_rows if "pair_cos_rank0_mean" in r}
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
        for rank in range(4):
            ys = [float(lookup[b][f"pair_cos_rank{rank}_mean"]) if b in lookup else float("nan") for b in blocks]
            ax.plot(blocks, ys, marker="o", markersize=3, label=f"remove rank {rank}" if rank else "raw")
        ax.set_xlabel("ViT block")
        ax.set_ylabel("Mean pairwise cosine of image register-means")
        ax.set_title(f"Image-invariant register subspace | {result.label}")
        ax.set_xticks(blocks)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "plots" / f"register_invariance_rank_removal_{result.label}.png", bbox_inches="tight")
        plt.close(fig)

    # 4) rank-2 residual comparison across variants.
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    for result in results:
        lookup = {int(r["block"]): r for r in result.invariant_rows if "pair_cos_rank2_mean" in r}
        ys = [float(lookup[b]["pair_cos_rank2_mean"]) if b in lookup else float("nan") for b in blocks]
        ax.plot(blocks, ys, marker="o", markersize=3, label=result.label)
    ax.set_xlabel("ViT block")
    ax.set_ylabel("Pairwise cosine after removing invariant rank-2")
    ax.set_title("Image-dependent residual of tracked register means")
    ax.set_xticks(blocks)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "plots" / "register_rank2_residual_cosine_by_block.png", bbox_inches="tight")
    plt.close(fig)

    # 5) mu-cache occupancy and edge enrichment.
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    for result in results:
        ys = [nanmean_group(result.cache_rows, "cache_fraction", b) for b in blocks]
        ax.plot(blocks, ys, marker="o", markersize=3, label=result.label)
    ax.set_xlabel("ViT block")
    ax.set_ylabel("Fraction of non-register patches carrying fixed B23 μ1")
    ax.set_title("μ-cache occupancy across the visual transformer")
    ax.set_xticks(blocks)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "plots" / "mu_cache_fraction_by_block.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    for result in results:
        ys = [nanmean_group(result.cache_rows, "cache_edge_enrichment", b) for b in blocks]
        ax.plot(blocks, ys, marker="o", markersize=3, label=result.label)
    ax.axhline(1.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("ViT block")
    ax.set_ylabel("Cache edge enrichment (1 = area expectation)")
    ax.set_title("Spatial bias of μ-cache candidates")
    ax.set_xticks(blocks)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "plots" / "mu_cache_edge_enrichment_by_block.png", bbox_inches="tight")
    plt.close(fig)

    # 6) scratchpad signature: within-image distant similarity versus cross-image similarity and gap.
    for result in results:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
        within = [nanmean_group(result.scratch_rows, "within_distant_mean", b) for b in blocks]
        cross = [nanmean_group(result.scratch_rows, "cross_image_max_mean", b) for b in blocks]
        gap = [nanmean_group(result.scratch_rows, "topk_gap_mean", b) for b in blocks]
        ax.plot(blocks, within, marker="o", markersize=3, label="within-image distant max")
        ax.plot(blocks, cross, marker="o", markersize=3, label="cross-image nearest max")
        ax.plot(blocks, gap, marker="o", markersize=3, label="top-k locality gap")
        ax.set_xlabel("ViT block")
        ax.set_ylabel("Cosine / cosine gap")
        ax.set_title(f"Image-local scratchpad coherence | {result.label}")
        ax.set_xticks(blocks)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "plots" / f"scratchpad_coherence_{result.label}.png", bbox_inches="tight")
        plt.close(fig)


def compare_bases(results: Sequence[VariantResult], root: Path, n_blocks: int = 24) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            pair_name = f"{a.label}__vs__{b.label}"
            for block in range(n_blocks):
                if block not in a.bases or block not in b.bases:
                    continue
                ba = torch.from_numpy(a.bases[block]).float()
                bb = torch.from_numpy(b.bases[block]).float()
                pc1 = abs(as_float(torch.dot(F.normalize(ba[0], dim=0), F.normalize(bb[0], dim=0))))
                rank = min(2, ba.shape[0], bb.shape[0])
                principal = torch.linalg.svdvals(ba[:rank] @ bb[:rank].t()) if rank else torch.empty(0)
                rows.append(
                    {
                        "model_a": a.label,
                        "model_b": b.label,
                        "pair": pair_name,
                        "block": block,
                        "pc1_abs_cosine": pc1,
                        "rank2_principal_cosine_mean": as_float(principal.mean()) if principal.numel() else float("nan"),
                        "rank2_principal_cosine_min": as_float(principal.min()) if principal.numel() else float("nan"),
                    }
                )
    write_csv(root / "metrics" / "cross_model_invariant_basis_alignment.csv", rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    for pair in sorted(set(r["pair"] for r in rows)):
        subset = {int(r["block"]): r for r in rows if r["pair"] == pair}
        ys = [float(subset[b]["pc1_abs_cosine"]) if b in subset else float("nan") for b in range(n_blocks)]
        ax.plot(range(n_blocks), ys, marker="o", markersize=3, label=pair)
    ax.set_xlabel("ViT block")
    ax.set_ylabel("|cos(μ1_a, μ1_b)|")
    ax.set_title("Alignment of the invariant register direction between models")
    ax.set_xticks(range(n_blocks))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "plots" / "cross_model_mu1_alignment_by_block.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def write_summary(results: Sequence[VariantResult], root: Path) -> None:
    lines = ["GIPU backbone dynamics summary", "=" * 80, ""]
    for result in results:
        lines.append(result.label)
        lines.append("-" * len(result.label))
        inv = {int(r["block"]): r for r in result.invariant_rows if "pair_cos_rank0_mean" in r}
        for block in (12, 16, 20, 23):
            if block not in inv:
                continue
            r = inv[block]
            lines.append(
                f"B{block:02d} register-mean pair cos: raw={r['pair_cos_rank0_mean']:.4f} "
                f"-r1={r['pair_cos_rank1_mean']:.4f} -r2={r['pair_cos_rank2_mean']:.4f} "
                f"-r3={r['pair_cos_rank3_mean']:.4f}"
            )
        attr_b23 = nanmean_group(
            result.attribution_rows,
            "register_attribution_fraction",
            23,
            predicate=lambda r: int(r.get("is_known_present", 0)) == 1,
        )
        raw_b23 = nanmean_group(result.raw_attention_rows, "cls_register_fraction_of_patch_attention", 23)
        cache_b23 = nanmean_group(result.cache_rows, "cache_fraction", 23)
        edge_b23 = nanmean_group(result.cache_rows, "cache_edge_enrichment", 23)
        scratch_b23 = nanmean_group(result.scratch_rows, "topk_gap_mean", 23)
        lines.append(f"B23 present-prompt register grad×attn fraction: {attr_b23:.4f}")
        lines.append(f"B23 raw CLS register-attention fraction:       {raw_b23:.4f}")
        lines.append(f"B23 fixed-μ1 cache patch fraction:              {cache_b23:.4f}")
        lines.append(f"B23 cache edge enrichment:                      {edge_b23:.4f}")
        lines.append(f"B23 scratchpad top-k locality gap:              {scratch_b23:.4f}")
        lines.append("")
    (root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =================================================================================================
# CLI / main
# =================================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--xattn-model", type=str, default=DEFAULT_XATTN_MODEL,
                   help="Jointly-trained x-attn HF repo id or local HF model directory")
    p.add_argument("--pretrained", type=str, default=DEFAULT_PRETRAINED,
                   help="Explicit GmP comparison baseline path or HF repo id; this is the pre-xattn GmP backbone, not generic vanilla CLIP.")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--register-threshold", type=float, default=60.0)
    p.add_argument("--cache-cos-threshold", type=float, default=0.80)
    p.add_argument("--edge-ring", type=int, default=2)
    p.add_argument("--scratch-min-distance", type=float, default=8.0,
                   help="Minimum Euclidean patch-grid distance for a within-image partner.")
    p.add_argument("--scratch-within-threshold", type=float, default=0.85)
    p.add_argument("--scratch-gap-threshold", type=float, default=0.15)
    p.add_argument("--scratch-topk", type=int, default=8,
                   help="Threshold-free summary: average the top-k within-minus-cross gaps per image/block.")
    p.add_argument("--map-blocks", type=str, default="0,6,8,12,16,20,23",
                   help="Blocks for exhaustive image overlays/panels; metrics are always B0..B23. Use 'all' for all maps.")
    p.add_argument("--skip-pretrained", action="store_true")
    p.add_argument("--skip-trained-vanilla", action="store_true")
    p.add_argument("--skip-trained-rn", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if "{prompt_word}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain {prompt_word}")
    if args.edge_ring < 1:
        raise ValueError("--edge-ring must be >= 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    root = args.output_dir
    if root.exists():
        if args.overwrite:
            shutil.rmtree(root)
        elif any(root.iterdir()):
            raise FileExistsError(f"Output is non-empty: {root}\nUse --overwrite or a new --output-dir.")
    ensure_dir(root)
    samples = load_samples(args.image_dir)
    map_blocks = set(parse_block_list(args.map_blocks, 24))

    write_json(
        root / "_meta" / "run.json",
        {
            "argv": sys.argv,
            "xattn_model": str(args.xattn_model),
            "pretrained": args.pretrained,
            "image_dir": str(args.image_dir),
            "output_dir": str(args.output_dir),
            "prompt_words": list(PROMPT_WORDS),
            "prompt_template": args.prompt_template,
            "register_definition": f"B23 spatial residual L2 norm >= {args.register_threshold}",
            "register_positions_propagated_backward": True,
            "cache_mu_definition": "uncentered PC1 of B23 per-image tracked-register means; fixed across blocks",
            "scratchpad_definition": {
                "remove": "block-local invariant rank-2 register-mean subspace",
                "min_spatial_distance": args.scratch_min_distance,
                "within_cos_threshold": args.scratch_within_threshold,
                "within_minus_cross_gap_threshold": args.scratch_gap_threshold,
                "exclude_tracked_registers": True,
                "exclude_mu_cache_candidates": True,
            },
            "map_blocks": sorted(map_blocks),
            "temporal_role_strips": {
                "enabled": True,
                "rows": "patch identities",
                "columns": "B0..B23",
                "metrics": [
                    "operational role",
                    "log(1 + residual L2 norm)",
                    "cosine to fixed B23 mu1",
                    "scratchpad within-distant minus cross-image gap",
                    "log10 positive semantic grad*attention",
                ],
                "orders": ["spatial_order", "role_sorted"],
                "role_precedence": ["tracked register", "mu-cache", "scratchpad", "other"],
            },
            "no_bridge_relocation": True,
        },
    )
    write_csv(
        root / "_meta" / "dataset.csv",
        [
            {
                "image": s.spec.filename,
                "path": str(s.path),
                "present_words": json.dumps(s.spec.present_words),
                "text_words": json.dumps(s.spec.text_words),
                "note": s.spec.note,
            }
            for s in samples
        ],
    )

    results: list[VariantResult] = []

    # -------------------------------------------------------------------------------------------------
    # 1) Original pre-xattn GmP CLIP.
    # -------------------------------------------------------------------------------------------------
    if not args.skip_pretrained:
        print(f"[load] pretrained baseline: {args.pretrained}")
        pretrained = CLIPModel.from_pretrained(args.pretrained).eval().float()
        pretrained_processor = AutoProcessor.from_pretrained(args.pretrained)
        result = analyze_variant(
            label="pretrained_gmp",
            model=pretrained,
            processor=pretrained_processor,
            samples=samples,
            root=root,
            device=device,
            batch_size=args.batch_size,
            insert_rn=False,
            register_threshold=args.register_threshold,
            cache_cos_threshold=args.cache_cos_threshold,
            edge_ring=args.edge_ring,
            scratch_min_distance=args.scratch_min_distance,
            scratch_within_threshold=args.scratch_within_threshold,
            scratch_gap_threshold=args.scratch_gap_threshold,
            scratch_topk=args.scratch_topk,
            prompt_template=args.prompt_template,
            map_blocks=map_blocks,
        )
        results.append(result)
        del pretrained, pretrained_processor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Load the jointly-trained x-attention checkpoint once on CPU.  It supplies both remaining variants.
    if not (args.skip_trained_vanilla and args.skip_trained_rn):
        print(f"[load] jointly-trained x-attn source: {args.xattn_model}")
        source = AutoModel.from_pretrained(args.xattn_model, trust_remote_code=True).eval().cpu().float()
        source_processor = AutoProcessor.from_pretrained(args.xattn_model, trust_remote_code=True)
        for required in ("read_null_token", "read_implant", "_vision_with_intermediates"):
            if not hasattr(source, required):
                raise TypeError(f"Expected full x-attn model, missing {required}")

        # ---------------------------------------------------------------------------------------------
        # 2) Same learned weights, physically stripped to stock CLIP.
        # ---------------------------------------------------------------------------------------------
        if not args.skip_trained_vanilla:
            trained_vanilla = build_bare_vanilla_clip(source)
            result = analyze_variant(
                label="trained_vanilla",
                model=trained_vanilla,
                processor=source_processor,
                samples=samples,
                root=root,
                device=device,
                batch_size=args.batch_size,
                insert_rn=False,
                register_threshold=args.register_threshold,
                cache_cos_threshold=args.cache_cos_threshold,
                edge_ring=args.edge_ring,
                scratch_min_distance=args.scratch_min_distance,
                scratch_within_threshold=args.scratch_within_threshold,
                scratch_gap_threshold=args.scratch_gap_threshold,
                scratch_topk=args.scratch_topk,
                prompt_template=args.prompt_template,
                map_blocks=map_blocks,
            )
            results.append(result)
            del trained_vanilla
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            source = source.cpu()

        # ---------------------------------------------------------------------------------------------
        # 3) Same learned weights + READ_NULL, but no bridge score: x-attn classic visual computation.
        # ---------------------------------------------------------------------------------------------
        if not args.skip_trained_rn:
            result = analyze_variant(
                label="trained_rn_classic",
                model=source,
                processor=source_processor,
                samples=samples,
                root=root,
                device=device,
                batch_size=args.batch_size,
                insert_rn=True,
                register_threshold=args.register_threshold,
                cache_cos_threshold=args.cache_cos_threshold,
                edge_ring=args.edge_ring,
                scratch_min_distance=args.scratch_min_distance,
                scratch_within_threshold=args.scratch_within_threshold,
                scratch_gap_threshold=args.scratch_gap_threshold,
                scratch_topk=args.scratch_topk,
                prompt_template=args.prompt_template,
                map_blocks=map_blocks,
            )
            results.append(result)

        del source, source_processor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("All variants were skipped")

    # Combined tables.
    write_csv(root / "metrics" / "register_invariance_all.csv", [r for x in results for r in x.invariant_rows])
    write_csv(root / "metrics" / "raw_attention_all.csv", [r for x in results for r in x.raw_attention_rows])
    write_csv(root / "metrics" / "register_attribution_all.csv", [r for x in results for r in x.attribution_rows])
    write_csv(root / "metrics" / "mu_cache_all.csv", [r for x in results for r in x.cache_rows])
    write_csv(root / "metrics" / "scratchpads_all.csv", [r for x in results for r in x.scratch_rows])

    plot_variant_curves(results, root)
    compare_bases(results, root)
    write_summary(results, root)

    print("\n[done]")
    print(f"output: {root}")
    print(f"summary: {root / 'summary.txt'}")
    print(f"main register-attribution curve: {root / 'plots' / 'register_attribution_fraction_present_by_block.png'}")
    print(f"rank-removal curves: {root / 'plots'}")
    print(f"selected GIPU panels: {root / 'paper_panels'}")
    print(f"temporal role strips: {root / 'temporal_role_strips'}")


if __name__ == "__main__":
    main()
