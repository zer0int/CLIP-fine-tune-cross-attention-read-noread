#!/usr/bin/env python3
r"""REG-tail -> compute-cache transport versus RN hijack microscope.

This is a focused mechanistic probe for the B12 -> B13 register/cache transition in
the full HF x-attention CLIP model.

Core questions
--------------
1. What rank of the visible-register representation is genuinely image-invariant?
   Do NOT guess.  Measure up to rank N (default 16) across 100 paired images from
   every SCAM/RTA subset, and verify the candidate invariant directions at B12,
   B13, and B20, with and without RN.

2. After removing only the empirically invariant register subspace, where does the
   image-dependent B12 REG tail appear at B13?

3. Does adding RN steal B13 attention mass / OV write from the visible REG sources
   at those same compute-cache destinations?  Is the remembered ~0.2 REG retention
   real?  Which heads do it (especially H5/H11)?

4. Where are the normal-norm hidden-mu tokens?
   Detect them from the *rank-1 register carrier direction only*, with a measured
   high-cosine cluster threshold.  Do not force their count.

5. Produce REG-vs-RN overlays on exactly the same images, including a split-token
   visualization where the left half of every patch shows REG contribution and the
   right half shows RN contribution.

Definitions used here
---------------------
VISIBLE REG:
    spatial patch residual norm > register_norm_threshold (default 60).
    No cap.  No forced fallback.  Counts are logged exactly as observed.

INVARIANT REGISTER SUBSPACE:
    non-centered low-rank SVD/PCA of unit-normalized visible-register residuals.
    B12-all-subsets is the primary basis.  For each candidate component, require
    high absolute cosine with a component in every subset/state verification basis.
    Components are accepted contiguously from rank 1 and selection STOPS at the
    first failure.  Rank 1 is retained as the known-safe carrier even if a chosen
    strict threshold produces a warning.

HIDDEN MU:
    normal-norm patches with high signed cosine to the oriented B12 rank-1 register
    carrier.  The threshold is estimated from a two-cluster fit to all normal-norm
    patch cosines.  Expected counts (~2x visible REG) are reported, never imposed.

B12 REG DATA TAIL:
    B12 visible-register residual after projection onto the measured invariant
    register subspace has been removed.

COMPUTE-CACHE DESTINATIONS:
    among B13 normal patches (not visible REG, not hidden-mu), the top-K patches
    whose de-invarianted B13-noRN residual lies most strongly in the per-image
    subspace spanned by the de-invarianted B12 REG tails.

B13 REG/RN TRANSPORT:
    actual B13 self-attention decomposition at those same destinations:
      - attention mass to visible REG sources
      - attention mass to hidden-mu sources
      - attention mass to RN
      - source-group OV write norm
      - per-head source mass/write
      - RN K-logit advantage over the strongest visible-REG K

The REG-tail subspace localization is a residual-stream measurement.
The attention/OV decomposition is a separate measurement of the actual B13 write.
Keeping those distinct avoids pretending LayerNorm makes residual-tail subtraction
linearly equivalent to subtracting a V-space carrier.

Blocks are zero-based.
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_analysis import (PairSample, _autocast, _batches, _family, _modality, _pair_key, _safe_mean, _safe_median, configure_reproducibility, load_samples, preprocess_images)


import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


# ================================================================================================
# Defaults
# ================================================================================================

DEFAULT_FULL_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_OUTPUT_DIR = Path(
    "out_bench_testing/reg_tail_vs_rn_cache_hijack"
)

SUBSET_ORDER = (
    "NoSCAM",
    "SCAM",
    "SynthSCAM",
    "NoRTA",
    "RTA",
    "SynthRTA",
)

FAMILY_SUBSETS = {
    "SCAM": ("NoSCAM", "SCAM", "SynthSCAM"),
    "RTA": ("NoRTA", "RTA", "SynthRTA"),
}

ATTACK_SUBSETS = (
    "SCAM",
    "SynthSCAM",
    "RTA",
    "SynthRTA",
)
HANDWRITTEN_SUBSETS = ("SCAM", "RTA")
DIGITAL_SUBSETS = ("SynthSCAM", "SynthRTA")
NO_TEXT_SUBSETS = ("NoSCAM", "NoRTA")

B12 = 12
B13 = 13
B20 = 20

DEFAULT_CALIBRATION_PER_SUBSET = 100
DEFAULT_MAX_INVARIANT_RANK = 16
DEFAULT_INVARIANT_COS_THRESHOLD = 0.95
DEFAULT_REGISTER_NORM_THRESHOLD = 60.0
DEFAULT_CACHE_TOPK = 8
DEFAULT_TOPK_OVERLAY = 8
DEFAULT_OVERLAY_N_PER_MODALITY = 6
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 20260829
DEFAULT_HEADS_OF_INTEREST = (5, 11)


# ================================================================================================
# Data / utilities
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
        "state",
        "block",
        "head",
        "pair_key",
        "id",
        "correct_label",
        "distractor_label",
    ]
    keys = set().union(*(row.keys() for row in rows))
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(keys - set(fields)))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_min(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.min(vals)) if vals else float("nan")


def _safe_max(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.max(vals)) if vals else float("nan")


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a.float()
    b = b.float()
    return (a @ b) / (a.norm() * b.norm()).clamp_min(eps)


def _normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(eps)


def _remove_subspace(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove columns of orthonormal basis [D,K] from x [...,D]."""
    x = x.float()
    if basis.numel() == 0:
        return x
    basis = basis.to(device=x.device, dtype=torch.float32)
    return x - (x @ basis) @ basis.t()


def _orthonormal_span(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Return an orthonormal basis [D,R] for row vectors x [N,D]."""
    if x.numel() == 0 or x.shape[0] == 0:
        return x.new_zeros((x.shape[-1], 0), dtype=torch.float32)
    x = x.float()
    # Tiny N (usually ~1-10 visible registers), so exact SVD is cheap.
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    if s.numel() == 0:
        return x.new_zeros((x.shape[-1], 0), dtype=torch.float32)
    keep = s > max(float(s[0]) * eps, eps)
    if not keep.any():
        return x.new_zeros((x.shape[-1], 0), dtype=torch.float32)
    return vh[keep].t().contiguous()


def _subspace_projection_fraction(
    x: torch.Tensor,
    basis: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Fraction of vector norm lying in basis span; x [P,D], basis [D,R]."""
    x = x.float()
    if basis.numel() == 0 or basis.shape[1] == 0:
        return torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
    proj = (x @ basis) @ basis.t()
    return proj.norm(dim=-1) / x.norm(dim=-1).clamp_min(eps)


def _subspace_overlap(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, int]:
    """Mean/max principal-angle cosine between spans of row vectors a and b."""
    qa = _orthonormal_span(a)
    qb = _orthonormal_span(b)
    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return float("nan"), float("nan"), 0
    s = torch.linalg.svdvals(qa.t() @ qb).clamp(0.0, 1.0)
    return float(s.mean()), float(s.max()), int(s.numel())


# ================================================================================================
# Dataset loading / paired 100-per-subset calibration selection
# ================================================================================================


def select_paired_calibration(
    sample_sets: dict[str, list[PairSample]],
    *,
    per_subset: int,
    seed: int,
) -> dict[str, list[PairSample]]:
    """
    Select the same pair keys across No / handwritten / synthetic variants.

    This gives exactly N images per subset when possible and keeps text/no-text
    comparisons paired rather than independently sampled.
    """
    rng = random.Random(seed)
    selected: dict[str, list[PairSample]] = {name: [] for name in SUBSET_ORDER}

    for family, subsets in FAMILY_SUBSETS.items():
        maps = {
            subset: {
                _pair_key(sample.sample_id): sample
                for sample in sample_sets[subset]
            }
            for subset in subsets
        }
        common = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
        if len(common) < per_subset:
            raise RuntimeError(
                f"{family}: only {len(common)} fully paired keys, requested {per_subset}"
            )
        keys = sorted(rng.sample(common, per_subset))
        for subset in subsets:
            selected[subset] = [maps[subset][key] for key in keys]

    return selected


# ================================================================================================
# HF vision helpers
# ================================================================================================


def _spatial(state: torch.Tensor, *, has_rn: bool) -> torch.Tensor:
    return state[:, 1:-1, :] if has_rn else state[:, 1:, :]


def visible_register_mask(
    state: torch.Tensor,
    *,
    has_rn: bool,
    norm_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = _spatial(state, has_rn=has_rn).detach().float()
    norms = spatial.norm(dim=-1)
    # Intentionally: no max cap and no minimum fallback.
    mask = norms > float(norm_threshold)
    return mask, norms


@torch.inference_mode()
def run_to_b20_branches(
    model: Any,
    pixel_values: torch.Tensor,
    *,
    rn_insert_block: int,
) -> dict[str, torch.Tensor]:
    """
    Shared through post-B12, then no-RN / RN branches through B20.

    Returned states are post-block residual states.
    """
    vision = model.vision_model

    hidden = vision.embeddings(
        pixel_values,
        interpolate_pos_encoding=False,
    )
    hidden = vision.pre_layrnorm(hidden)

    b12 = None
    for block in range(rn_insert_block):
        hidden = vision.encoder.layers[block](hidden, None)
        if block == B12:
            b12 = hidden

    if b12 is None:
        raise RuntimeError("Failed to capture post-B12 state")

    hidden_no = b12
    rn = model.read_null_token.to(device=b12.device, dtype=b12.dtype)
    hidden_rn = torch.cat(
        (
            b12,
            rn.view(1, 1, -1).expand(b12.shape[0], 1, -1),
        ),
        dim=1,
    )

    b13_no = None
    b13_rn = None
    b20_no = None
    b20_rn = None

    for block in range(rn_insert_block, B20 + 1):
        layer = vision.encoder.layers[block]
        hidden_no = layer(hidden_no, None)
        hidden_rn = layer(hidden_rn, None)

        if block == B13:
            b13_no = hidden_no
            b13_rn = hidden_rn
        if block == B20:
            b20_no = hidden_no
            b20_rn = hidden_rn

    if any(x is None for x in (b13_no, b13_rn, b20_no, b20_rn)):
        raise RuntimeError("Failed to capture B13/B20 branch states")

    return {
        "B12_shared": b12,
        "B13_noRN": b13_no,
        "B13_RN": b13_rn,
        "B20_noRN": b20_no,
        "B20_RN": b20_rn,
    }


@torch.inference_mode()
def run_to_b13_branches(
    model: Any,
    pixel_values: torch.Tensor,
    *,
    rn_insert_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return B12 shared, post-B13 noRN, post-B13 RN."""
    vision = model.vision_model

    hidden = vision.embeddings(
        pixel_values,
        interpolate_pos_encoding=False,
    )
    hidden = vision.pre_layrnorm(hidden)

    for block in range(rn_insert_block):
        hidden = vision.encoder.layers[block](hidden, None)

    b12 = hidden

    hidden_no = b12
    rn = model.read_null_token.to(device=b12.device, dtype=b12.dtype)
    hidden_rn = torch.cat(
        (
            b12,
            rn.view(1, 1, -1).expand(b12.shape[0], 1, -1),
        ),
        dim=1,
    )

    layer = vision.encoder.layers[B13]
    b13_no = layer(hidden_no, None)
    b13_rn = layer(hidden_rn, None)

    return b12, b13_no, b13_rn


# ================================================================================================
# Stage 1: collect visible-register vectors at B12/B13/B20
# ================================================================================================

@torch.inference_mode()
def collect_register_calibration(
    model: Any,
    processor: Any,
    selected: dict[str, list[PairSample]],
    *,
    rn_insert_block: int,
    norm_threshold: float,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[
    dict[tuple[str, str], torch.Tensor],
    dict[tuple[str, str], torch.Tensor],
    dict[tuple[str, str], torch.Tensor],
    list[dict[str, Any]],
    dict[str, torch.Tensor],
    dict[tuple[str, str], torch.Tensor],
]:
    """
    Returns:
      register_vectors_by_state_subset[(state, subset)] -> [Nreg,D]
      register_vectors_by_state_all[(state,"ALL")]       -> [Nreg,D]
      register_image_index[(state, id)]                  -> [R,D]
      count rows
      B12 spatial states by id in CPU float16 (for hidden-mu threshold)
      B12 visible-reg masks by (subset,id) in CPU bool
    """
    per_state_subset_lists: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    per_image: dict[tuple[str, str], torch.Tensor] = {}
    count_rows: list[dict[str, Any]] = []
    b12_spatial_by_id: dict[str, torch.Tensor] = {}
    b12_regmask_by_key: dict[tuple[str, str], torch.Tensor] = {}

    state_has_rn = {
        "B12_shared": False,
        "B13_noRN": False,
        "B13_RN": True,
        "B20_noRN": False,
        "B20_RN": True,
    }

    print()
    print("[stage1] Collecting visible-register populations at B12/B13/B20 ...")

    for subset in SUBSET_ORDER:
        samples = selected[subset]

        for start, batch_samples in tqdm(
            list(_batches(samples, batch_size)),
            desc=f"registers {subset}",
            leave=False,
        ):
            pixel_values = preprocess_images(processor, batch_samples, device)

            with _autocast(device, amp):
                states = run_to_b20_branches(
                    model,
                    pixel_values,
                    rn_insert_block=rn_insert_block,
                )

            for state_name, state in states.items():
                has_rn = state_has_rn[state_name]
                mask, norms = visible_register_mask(
                    state,
                    has_rn=has_rn,
                    norm_threshold=norm_threshold,
                )
                spatial = _spatial(state, has_rn=has_rn).detach().float()

                for local_index, sample in enumerate(batch_samples):
                    image_id = sample.sample_id
                    reg_idx = torch.nonzero(mask[local_index], as_tuple=False).flatten()
                    reg_vectors = spatial[local_index, reg_idx].cpu()

                    if reg_vectors.numel():
                        per_state_subset_lists[(state_name, subset)].append(reg_vectors)

                    per_image[(state_name, image_id)] = reg_vectors

                    count_rows.append(
                        {
                            "subset": subset,
                            "modality": _modality(subset),
                            "family": _family(subset),
                            "state": state_name,
                            "id": image_id,
                            "pair_key": _pair_key(image_id),
                            "register_count": int(reg_idx.numel()),
                            "register_indices_json": json.dumps(
                                [int(v) for v in reg_idx.cpu().tolist()]
                            ),
                            "mean_patch_norm": float(norms[local_index].mean()),
                            "max_patch_norm": float(norms[local_index].max()),
                            "mean_register_norm": (
                                float(norms[local_index, reg_idx].mean())
                                if reg_idx.numel()
                                else float("nan")
                            ),
                        }
                    )

                if state_name == "B12_shared":
                    for local_index, sample in enumerate(batch_samples):
                        b12_spatial_by_id[sample.sample_id] = (
                            spatial[local_index].to(dtype=torch.float16).cpu()
                        )
                        b12_regmask_by_key[(subset, sample.sample_id)] = (
                            mask[local_index].cpu()
                        )

            del pixel_values, states

    by_state_subset: dict[tuple[str, str], torch.Tensor] = {}
    by_state_all: dict[tuple[str, str], torch.Tensor] = {}

    for state_name in state_has_rn:
        all_chunks: list[torch.Tensor] = []
        for subset in SUBSET_ORDER:
            chunks = per_state_subset_lists.get((state_name, subset), [])
            if chunks:
                matrix = torch.cat(chunks, dim=0)
            else:
                matrix = torch.empty((0, model.config.vision_config.hidden_size))
            by_state_subset[(state_name, subset)] = matrix
            if matrix.numel():
                all_chunks.append(matrix)

        by_state_all[(state_name, "ALL")] = (
            torch.cat(all_chunks, dim=0)
            if all_chunks
            else torch.empty((0, model.config.vision_config.hidden_size))
        )

    return (
        by_state_subset,
        by_state_all,
        per_image,
        count_rows,
        b12_spatial_by_id,
        b12_regmask_by_key,
    )


# ================================================================================================
# Invariant-rank measurement
# ================================================================================================

def lowrank_basis(
    x: torch.Tensor,
    rank: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Non-centered low-rank decomposition of UNIT-NORMALIZED register vectors.

    Returns:
      basis [D,K]
      singular-value-like scores [K]
    """
    if x.shape[0] < 2:
        raise RuntimeError(f"Need >=2 register vectors, got {x.shape[0]}")

    x = _normalize_rows(x.cpu())
    q = min(int(rank), int(x.shape[0]) - 1, int(x.shape[1]))
    if q < 1:
        raise RuntimeError("Low-rank basis q < 1")

    # pca_lowrank(center=False) is randomized but far cheaper than dozens of
    # full 1024-D SVDs.  Seed every call deterministically.
    torch.manual_seed(seed)
    _, s, v = torch.pca_lowrank(
        x,
        q=q,
        center=False,
        niter=4,
    )
    return v.float().contiguous(), s.float().contiguous()


def build_all_bases(
    by_state_subset: dict[tuple[str, str], torch.Tensor],
    by_state_all: dict[tuple[str, str], torch.Tensor],
    *,
    max_rank: int,
    seed: int,
) -> tuple[
    dict[tuple[str, str], torch.Tensor],
    dict[tuple[str, str], torch.Tensor],
]:
    bases: dict[tuple[str, str], torch.Tensor] = {}
    singulars: dict[tuple[str, str], torch.Tensor] = {}

    state_names = ("B12_shared", "B13_noRN", "B13_RN", "B20_noRN", "B20_RN")

    print()
    print("[SVD] Measuring rank-n register directions ...")

    serial = 0
    for state in state_names:
        for subset in ("ALL",) + SUBSET_ORDER:
            matrix = (
                by_state_all[(state, "ALL")]
                if subset == "ALL"
                else by_state_subset[(state, subset)]
            )
            if matrix.shape[0] < 2:
                continue
            basis, s = lowrank_basis(
                matrix,
                max_rank,
                seed=seed + serial,
            )
            serial += 1
            bases[(state, subset)] = basis
            singulars[(state, subset)] = s

    return bases, singulars


def orient_primary_basis(
    primary_basis: torch.Tensor,
    primary_vectors: torch.Tensor,
) -> torch.Tensor:
    basis = primary_basis.clone().float()
    # Orient each component so average visible-register projection is positive.
    mean_vec = _normalize_rows(primary_vectors).mean(dim=0)
    for k in range(basis.shape[1]):
        if float(mean_vec @ basis[:, k]) < 0.0:
            basis[:, k].mul_(-1.0)
    return basis


def select_invariant_rank(
    bases: dict[tuple[str, str], torch.Tensor],
    singulars: dict[tuple[str, str], torch.Tensor],
    primary_vectors: torch.Tensor,
    *,
    max_rank: int,
    cos_threshold: float,
) -> tuple[torch.Tensor, int, list[dict[str, Any]]]:
    """
    Primary basis = B12 / ALL.

    Candidate component k must match some top-N component in EVERY available:
      - subset basis at B12/B13/B20
      - RN/noRN state basis
    by abs cosine >= threshold.

    Accept contiguously from rank 1 and STOP at first failure.
    """
    primary_raw = bases[("B12_shared", "ALL")]
    primary = orient_primary_basis(primary_raw, primary_vectors)

    verifiers = sorted(
        key
        for key in bases
        if key != ("B12_shared", "ALL")
    )

    rows: list[dict[str, Any]] = []
    contiguous = 0
    stopped = False

    for k in range(min(max_rank, primary.shape[1])):
        u = primary[:, k]
        matches: dict[str, float] = {}

        for key in verifiers:
            candidate = bases[key]
            cosines = torch.abs(candidate.t() @ u)
            match = float(cosines.max()) if cosines.numel() else float("nan")
            matches[f"{key[0]}__{key[1]}"] = match

        finite = [v for v in matches.values() if math.isfinite(v)]
        min_match = min(finite) if finite else float("nan")
        passed = bool(
            math.isfinite(min_match)
            and min_match >= float(cos_threshold)
        )

        if not stopped and passed:
            contiguous += 1
        elif not stopped:
            stopped = True

        s = singulars[("B12_shared", "ALL")]
        row = {
            "component_1based": k + 1,
            "primary_singular_value": float(s[k]) if k < len(s) else float("nan"),
            "min_abs_cosine_across_verifiers": min_match,
            "passes_invariant_threshold": int(passed),
            "accepted_contiguously": int(k < contiguous),
        }
        row.update(matches)
        rows.append(row)

    if contiguous < 1:
        print(
            "[warn] Chosen invariant-cos threshold rejected even rank-1. "
            "Rank-1 is the known-safe image-invariant carrier, so retaining rank 1 "
            "and flagging the threshold mismatch."
        )
        contiguous = 1

    invariant_basis = primary[:, :contiguous].contiguous()
    return invariant_basis, contiguous, rows


# ================================================================================================
# Hidden-mu detection from rank-1 only
# ================================================================================================

def _two_means_1d(values: np.ndarray, iterations: int = 100) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise RuntimeError("Not enough values for two-cluster hidden-mu threshold")

    c0 = float(np.quantile(values, 0.50))
    c1 = float(np.quantile(values, 0.995))
    if c1 <= c0:
        c1 = float(values.max())

    for _ in range(iterations):
        d0 = np.abs(values - c0)
        d1 = np.abs(values - c1)
        high = d1 < d0
        if high.all() or (~high).all():
            break
        n0 = float(values[~high].mean())
        n1 = float(values[high].mean())
        if abs(n0 - c0) < 1e-10 and abs(n1 - c1) < 1e-10:
            c0, c1 = n0, n1
            break
        c0, c1 = n0, n1

    low_center, high_center = sorted((c0, c1))
    threshold = 0.5 * (low_center + high_center)
    return low_center, high_center, threshold


def measure_hidden_mu_threshold(
    b12_spatial_by_id: dict[str, torch.Tensor],
    b12_regmask_by_key: dict[tuple[str, str], torch.Tensor],
    selected: dict[str, list[PairSample]],
    carrier_u1: torch.Tensor,
    *,
    override_threshold: float | None,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    carrier = F.normalize(carrier_u1.float(), dim=0)

    cos_values: list[np.ndarray] = []

    for subset in SUBSET_ORDER:
        for sample in selected[subset]:
            x = b12_spatial_by_id[sample.sample_id].float()
            regmask = b12_regmask_by_key[(subset, sample.sample_id)].bool()
            cos = F.cosine_similarity(
                x,
                carrier.view(1, -1).expand_as(x),
                dim=-1,
            )
            cos_values.append(cos[~regmask].cpu().numpy())

    all_cos = np.concatenate(cos_values)
    low_center, high_center, measured_threshold = _two_means_1d(all_cos)
    threshold = (
        float(override_threshold)
        if override_threshold is not None
        else float(measured_threshold)
    )

    count_rows: list[dict[str, Any]] = []

    for subset in SUBSET_ORDER:
        for sample in selected[subset]:
            x = b12_spatial_by_id[sample.sample_id].float()
            regmask = b12_regmask_by_key[(subset, sample.sample_id)].bool()
            cos = F.cosine_similarity(
                x,
                carrier.view(1, -1).expand_as(x),
                dim=-1,
            )
            hidden = (~regmask) & (cos >= threshold)
            reg_count = int(regmask.sum())
            hidden_count = int(hidden.sum())

            count_rows.append(
                {
                    "subset": subset,
                    "modality": _modality(subset),
                    "family": _family(subset),
                    "id": sample.sample_id,
                    "pair_key": _pair_key(sample.sample_id),
                    "visible_register_count": reg_count,
                    "hidden_mu_count": hidden_count,
                    "hidden_mu_to_register_ratio": (
                        hidden_count / reg_count
                        if reg_count
                        else float("nan")
                    ),
                    "hidden_mu_indices_json": json.dumps(
                        [int(v) for v in torch.nonzero(hidden, as_tuple=False).flatten().tolist()]
                    ),
                    "hidden_mu_mean_carrier_cos": (
                        float(cos[hidden].mean())
                        if hidden.any()
                        else float("nan")
                    ),
                    "highest_nonreg_carrier_cos": float(cos[~regmask].max()),
                }
            )

    info = {
        "low_cluster_center": low_center,
        "high_cluster_center": high_center,
        "measured_threshold": measured_threshold,
        "used_threshold": threshold,
        "threshold_overridden": override_threshold is not None,
        "all_normal_patch_count": int(all_cos.size),
        "all_normal_patch_mean_cos": float(all_cos.mean()),
        "all_normal_patch_p95_cos": float(np.quantile(all_cos, 0.95)),
        "all_normal_patch_p99_cos": float(np.quantile(all_cos, 0.99)),
    }

    return threshold, info, count_rows


# ================================================================================================
# Cross-block register-tail sanity check
# ================================================================================================

def register_tail_crossblock_rows(
    per_image_registers: dict[tuple[str, str], torch.Tensor],
    selected: dict[str, list[PairSample]],
    invariant_basis: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    comparisons = (
        ("B12_shared", "B13_noRN", "B12_to_B13_noRN"),
        ("B12_shared", "B13_RN", "B12_to_B13_RN"),
        ("B13_noRN", "B20_noRN", "B13_to_B20_noRN"),
        ("B13_RN", "B20_RN", "B13_to_B20_RN"),
    )

    for subset in SUBSET_ORDER:
        for sample in selected[subset]:
            for a_name, b_name, label in comparisons:
                a = per_image_registers.get((a_name, sample.sample_id))
                b = per_image_registers.get((b_name, sample.sample_id))
                if a is None or b is None or a.shape[0] == 0 or b.shape[0] == 0:
                    mean_overlap = max_overlap = centroid_cos = float("nan")
                    principal_count = 0
                else:
                    a_tail = _remove_subspace(a, invariant_basis)
                    b_tail = _remove_subspace(b, invariant_basis)
                    mean_overlap, max_overlap, principal_count = _subspace_overlap(a_tail, b_tail)
                    a_cent = a_tail.mean(dim=0)
                    b_cent = b_tail.mean(dim=0)
                    centroid_cos = float(_cosine(a_cent, b_cent))

                rows.append(
                    {
                        "subset": subset,
                        "modality": _modality(subset),
                        "family": _family(subset),
                        "id": sample.sample_id,
                        "pair_key": _pair_key(sample.sample_id),
                        "comparison": label,
                        "tail_subspace_mean_principal_cos": mean_overlap,
                        "tail_subspace_max_principal_cos": max_overlap,
                        "tail_principal_count": principal_count,
                        "tail_centroid_cos": centroid_cos,
                    }
                )

    return rows


# ================================================================================================
# Exact B13 attention decomposition
# ================================================================================================

def validate_attention_architecture(layer: Any) -> None:
    if not hasattr(layer, "layer_norm1") or not hasattr(layer, "self_attn"):
        raise TypeError("Expected HF CLIPEncoderLayer with layer_norm1/self_attn")
    attn = layer.self_attn
    for name in ("q_proj", "k_proj", "v_proj", "out_proj", "num_heads"):
        if not hasattr(attn, name):
            raise TypeError(f"Expected CLIPAttention.{name}")


@torch.inference_mode()
def manual_attention(
    layer: Any,
    hidden_states: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Recompute B13 self-attention in FP32 for diagnostics.

    q is scaled exactly as standard HF CLIP attention:
        q_proj(LN(x)) * head_dim^-0.5
    """
    validate_attention_architecture(layer)
    attn = layer.self_attn

    x = layer.layer_norm1(hidden_states).float()
    B, T, D = x.shape
    H = int(attn.num_heads)
    Dh = D // H

    def lin(module: Any) -> torch.Tensor:
        bias = module.bias.float() if module.bias is not None else None
        return F.linear(x, module.weight.float(), bias)

    q = lin(attn.q_proj)
    k = lin(attn.k_proj)
    v = lin(attn.v_proj)

    scale = float(getattr(attn, "scale", Dh ** -0.5))
    q = q * scale

    q = q.view(B, T, H, Dh).transpose(1, 2).contiguous()
    k = k.view(B, T, H, Dh).transpose(1, 2).contiguous()
    v = v.view(B, T, H, Dh).transpose(1, 2).contiguous()

    logits = torch.matmul(q, k.transpose(-1, -2))
    probs = torch.softmax(logits, dim=-1)
    del logits

    return {
        "q": q,
        "k": k,
        "v": v,
        "probs": probs,
    }


def head_output_matrices(layer: Any) -> torch.Tensor:
    """
    [H,Dh,D] matrices mapping one head's z vector to residual write.
    """
    attn = layer.self_attn
    W = attn.out_proj.weight.detach().float()  # [Dout,Din]
    H = int(attn.num_heads)
    D = int(W.shape[1])
    Dh = D // H
    mats = []
    for h in range(H):
        sl = slice(h * Dh, (h + 1) * Dh)
        mats.append(W[:, sl].t().contiguous())
    return torch.stack(mats, dim=0)  # [H,Dh,D]


def group_transport(
    probs: torch.Tensor,
    v: torch.Tensor,
    out_mats: torch.Tensor,
    *,
    query_indices: torch.Tensor,
    source_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One image only.

    probs [H,T,T], v [H,T,Dh], out_mats [H,Dh,D]
    query_indices [Q], source_indices [S]

    Returns:
      mass       [H,Q]
      head_norm  [H,Q]
      total_norm [Q]
    """
    H = probs.shape[0]
    Q = int(query_indices.numel())

    if Q == 0:
        return (
            torch.zeros((H, 0), device=probs.device),
            torch.zeros((H, 0), device=probs.device),
            torch.zeros((0,), device=probs.device),
        )

    if source_indices.numel() == 0:
        return (
            torch.zeros((H, Q), device=probs.device),
            torch.zeros((H, Q), device=probs.device),
            torch.zeros((Q,), device=probs.device),
        )

    a = probs[:, query_indices][:, :, source_indices]   # [H,Q,S]
    vv = v[:, source_indices, :]                        # [H,S,Dh]
    z = torch.einsum("hqs,hsd->hqd", a, vv)            # [H,Q,Dh]
    contrib = torch.einsum("hqd,hde->hqe", z, out_mats)  # [H,Q,D]

    mass = a.sum(dim=-1)
    head_norm = contrib.norm(dim=-1)
    total_norm = contrib.sum(dim=0).norm(dim=-1)
    return mass, head_norm, total_norm


def source_key_logit_gap(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    query_indices: torch.Tensor,
    reg_source_indices: torch.Tensor,
    rn_source_index: int,
) -> torch.Tensor:
    """
    RN pre-softmax K-logit minus strongest visible-REG K-logit.

    q,k are one image [H,T,Dh].  Returns [H,Q].
    """
    if query_indices.numel() == 0:
        return torch.empty((q.shape[0], 0), device=q.device)
    if reg_source_indices.numel() == 0:
        return torch.full(
            (q.shape[0], query_indices.numel()),
            float("nan"),
            device=q.device,
        )

    qq = q[:, query_indices, :]                      # [H,Q,Dh]
    kk_reg = k[:, reg_source_indices, :]             # [H,R,Dh]
    kk_rn = k[:, rn_source_index, :]                 # [H,Dh]

    reg_logits = torch.einsum("hqd,hrd->hqr", qq, kk_reg)
    reg_max = reg_logits.max(dim=-1).values
    rn_logits = torch.einsum("hqd,hd->hq", qq, kk_rn)
    return rn_logits - reg_max


def rn_head_ov_rank1_fraction(layer: Any, rn_input_state: torch.Tensor) -> dict[str, Any]:
    """
    Constant RN source value projected through each output-head slice.

    Measure whether the 16 head-specific RN OV directions themselves occupy a
    nearly rank-1 residual subspace.
    """
    diag = manual_attention(layer, rn_input_state)
    v = diag["v"][0, :, -1, :]   # [H,Dh]
    mats = head_output_matrices(layer).to(v.device)
    ov = torch.einsum("hd,hde->he", v, mats)  # [H,D]
    s = torch.linalg.svdvals(ov.float())
    energy = s.square()
    fraction = float(energy[0] / energy.sum().clamp_min(1e-12))
    return {
        "rn_head_ov_rank1_energy_fraction": fraction,
        "rn_head_ov_singular_values": [float(x) for x in s.cpu().tolist()],
    }


# ================================================================================================
# Stage 2: cache destinations + REG/RN transport
# ================================================================================================

def hidden_mu_mask_for_state(
    b12_spatial: torch.Tensor,
    regmask: torch.Tensor,
    carrier_u1: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    carrier = F.normalize(
        carrier_u1.to(device=b12_spatial.device, dtype=torch.float32),
        dim=0,
    )
    x = b12_spatial.float()
    cos = F.cosine_similarity(
        x,
        carrier.view(1, -1).expand_as(x),
        dim=-1,
    )
    return (~regmask) & (cos >= float(threshold))


def select_cache_indices(
    payload_score_no_rn: torch.Tensor,
    regmask: torch.Tensor,
    hidden_mu_mask: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    candidate = (~regmask) & (~hidden_mu_mask)
    indices = torch.nonzero(candidate, as_tuple=False).flatten()
    if indices.numel() == 0:
        return indices
    k = min(int(topk), int(indices.numel()))
    local = torch.topk(
        payload_score_no_rn[indices],
        k=k,
        largest=True,
    ).indices
    return indices[local]


def _mean_or_nan(x: torch.Tensor) -> float:
    return float(x.mean()) if x.numel() else float("nan")


def _ratio_float(num: float, den: float, eps: float = 1e-12) -> float:
    if not math.isfinite(num) or not math.isfinite(den):
        return float("nan")
    return num / max(abs(den), eps)


@torch.inference_mode()
def collect_cache_transport(
    model: Any,
    processor: Any,
    selected: dict[str, list[PairSample]],
    *,
    invariant_basis: torch.Tensor,
    carrier_u1: torch.Tensor,
    hidden_mu_threshold: float,
    register_norm_threshold: float,
    cache_topk: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, np.ndarray]],
    list[dict[str, Any]],
]:
    """
    Per-image transport metrics + all-head rows + overlay maps.

    overlay_maps[id] contains:
      payload_no, payload_rn,
      reg_mass_no, reg_mass_rn, rn_mass,
      reg_ov_no, reg_ov_rn, rn_ov,
      regmask, hidden_mu_mask, cachemask
    """
    layer = model.vision_model.encoder.layers[B13]
    validate_attention_architecture(layer)
    out_mats = head_output_matrices(layer).to(device=device)

    image_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    overlay_maps: dict[str, dict[str, np.ndarray]] = {}
    overlay_meta: list[dict[str, Any]] = []

    inv = invariant_basis.to(device=device, dtype=torch.float32)
    carrier = carrier_u1.to(device=device, dtype=torch.float32)

    print()
    print("[stage2] Measuring B12 REG-tail cache placement and B13 REG/RN transport ...")

    for subset in SUBSET_ORDER:
        samples = selected[subset]

        for start, batch_samples in tqdm(
            list(_batches(samples, batch_size)),
            desc=f"transport {subset}",
            leave=False,
        ):
            pixel_values = preprocess_images(processor, batch_samples, device)

            with _autocast(device, amp):
                b12, b13_no, b13_rn = run_to_b13_branches(
                    model,
                    pixel_values,
                    rn_insert_block=int(model.config.read_null_insert_block),
                )

            # Attention diagnostics must be computed on the B13 INPUT state.
            rn = model.read_null_token.to(device=b12.device, dtype=b12.dtype)
            b13_input_no = b12
            b13_input_rn = torch.cat(
                (
                    b12,
                    rn.view(1, 1, -1).expand(b12.shape[0], 1, -1),
                ),
                dim=1,
            )

            attn_no = manual_attention(layer, b13_input_no)
            attn_rn = manual_attention(layer, b13_input_rn)

            b12_spatial = b12[:, 1:, :].float()
            b13_no_spatial = b13_no[:, 1:, :].float()
            b13_rn_spatial = b13_rn[:, 1:-1, :].float()

            regmask_batch, norms_batch = visible_register_mask(
                b12,
                has_rn=False,
                norm_threshold=register_norm_threshold,
            )

            for local_index, sample in enumerate(batch_samples):
                image_id = sample.sample_id
                regmask = regmask_batch[local_index]
                reg_idx = torch.nonzero(regmask, as_tuple=False).flatten()

                hidden_mu = hidden_mu_mask_for_state(
                    b12_spatial[local_index],
                    regmask,
                    carrier,
                    hidden_mu_threshold,
                )
                hidden_idx = torch.nonzero(hidden_mu, as_tuple=False).flatten()

                # ------------------------------------------
                # Original image-conditioned B12 REG tail.
                # ------------------------------------------
                b12_deinv = _remove_subspace(
                    b12_spatial[local_index],
                    inv,
                )
                reg_tail = b12_deinv[reg_idx]
                reg_tail_basis = _orthonormal_span(reg_tail)

                # Where that same per-image REG-tail subspace is readable after B13.
                b13_no_deinv = _remove_subspace(
                    b13_no_spatial[local_index],
                    inv,
                )
                b13_rn_deinv = _remove_subspace(
                    b13_rn_spatial[local_index],
                    inv,
                )
                payload_no = _subspace_projection_fraction(
                    b13_no_deinv,
                    reg_tail_basis,
                )
                payload_rn = _subspace_projection_fraction(
                    b13_rn_deinv,
                    reg_tail_basis,
                )

                cache_idx = select_cache_indices(
                    payload_no,
                    regmask,
                    hidden_mu,
                    topk=cache_topk,
                )
                cache_mask = torch.zeros_like(regmask)
                cache_mask[cache_idx] = True

                # Token indices: CLS=0, spatial patch p => token p+1.
                patch_query_tokens = torch.arange(
                    1,
                    1 + b12_spatial.shape[1],
                    device=device,
                    dtype=torch.long,
                )
                cache_query_tokens = cache_idx.long() + 1
                reg_source_tokens = reg_idx.long() + 1
                hidden_source_tokens = hidden_idx.long() + 1
                rn_source_token = int(b13_input_rn.shape[1] - 1)

                # ------------------------------------------
                # Full spatial maps of actual source-group transport.
                # ------------------------------------------
                no_probs = attn_no["probs"][local_index]
                no_v = attn_no["v"][local_index]
                rn_probs = attn_rn["probs"][local_index]
                rn_v = attn_rn["v"][local_index]

                reg_mass_no_map_h, reg_head_no_map, reg_ov_no_map = group_transport(
                    no_probs,
                    no_v,
                    out_mats,
                    query_indices=patch_query_tokens,
                    source_indices=reg_source_tokens,
                )
                reg_mass_rn_map_h, reg_head_rn_map, reg_ov_rn_map = group_transport(
                    rn_probs,
                    rn_v,
                    out_mats,
                    query_indices=patch_query_tokens,
                    source_indices=reg_source_tokens,
                )
                rn_mass_map_h, rn_head_map, rn_ov_map = group_transport(
                    rn_probs,
                    rn_v,
                    out_mats,
                    query_indices=patch_query_tokens,
                    source_indices=torch.tensor(
                        [rn_source_token],
                        device=device,
                        dtype=torch.long,
                    ),
                )

                hidden_mass_no_map_h, hidden_head_no_map, hidden_ov_no_map = group_transport(
                    no_probs,
                    no_v,
                    out_mats,
                    query_indices=patch_query_tokens,
                    source_indices=hidden_source_tokens,
                )
                hidden_mass_rn_map_h, hidden_head_rn_map, hidden_ov_rn_map = group_transport(
                    rn_probs,
                    rn_v,
                    out_mats,
                    query_indices=patch_query_tokens,
                    source_indices=hidden_source_tokens,
                )

                reg_mass_no_map = reg_mass_no_map_h.mean(dim=0)
                reg_mass_rn_map = reg_mass_rn_map_h.mean(dim=0)
                rn_mass_map = rn_mass_map_h.mean(dim=0)

                # Cache-local slices: patch map index == spatial patch index.
                c = cache_idx
                reg_mass_no_cache = _mean_or_nan(reg_mass_no_map[c])
                reg_mass_rn_cache = _mean_or_nan(reg_mass_rn_map[c])
                rn_mass_cache = _mean_or_nan(rn_mass_map[c])

                reg_ov_no_cache = _mean_or_nan(reg_ov_no_map[c])
                reg_ov_rn_cache = _mean_or_nan(reg_ov_rn_map[c])
                rn_ov_cache = _mean_or_nan(rn_ov_map[c])

                hidden_mass_no_cache = _mean_or_nan(hidden_mass_no_map_h.mean(dim=0)[c])
                hidden_mass_rn_cache = _mean_or_nan(hidden_mass_rn_map_h.mean(dim=0)[c])

                payload_no_cache = _mean_or_nan(payload_no[c])
                payload_rn_cache = _mean_or_nan(payload_rn[c])

                # RN K hijack gap at cache query positions.
                key_gap = source_key_logit_gap(
                    attn_rn["q"][local_index],
                    attn_rn["k"][local_index],
                    query_indices=cache_query_tokens,
                    reg_source_indices=reg_source_tokens,
                    rn_source_index=rn_source_token,
                )

                row = {
                    "subset": subset,
                    "modality": _modality(subset),
                    "family": _family(subset),
                    "id": image_id,
                    "pair_key": _pair_key(image_id),
                    "correct_label": sample.correct_label,
                    "distractor_label": sample.distractor_label,
                    "visible_register_count": int(reg_idx.numel()),
                    "hidden_mu_count": int(hidden_idx.numel()),
                    "hidden_mu_to_register_ratio": (
                        int(hidden_idx.numel()) / int(reg_idx.numel())
                        if reg_idx.numel()
                        else float("nan")
                    ),
                    "cache_patch_count": int(cache_idx.numel()),
                    "visible_register_indices_json": json.dumps(
                        [int(v) for v in reg_idx.cpu().tolist()]
                    ),
                    "hidden_mu_indices_json": json.dumps(
                        [int(v) for v in hidden_idx.cpu().tolist()]
                    ),
                    "cache_indices_json": json.dumps(
                        [int(v) for v in cache_idx.cpu().tolist()]
                    ),
                    "B12_mean_register_norm": (
                        float(norms_batch[local_index, reg_idx].mean())
                        if reg_idx.numel()
                        else float("nan")
                    ),
                    "B13_noRN_reg_payload_cache_mean": payload_no_cache,
                    "B13_RN_reg_payload_cache_mean": payload_rn_cache,
                    "reg_payload_retention_RN_over_noRN": _ratio_float(
                        payload_rn_cache,
                        payload_no_cache,
                    ),
                    "B13_noRN_REG_attention_mass_cache": reg_mass_no_cache,
                    "B13_RN_REG_attention_mass_cache": reg_mass_rn_cache,
                    "REG_attention_mass_retention_RN_over_noRN": _ratio_float(
                        reg_mass_rn_cache,
                        reg_mass_no_cache,
                    ),
                    "B13_RN_RN_attention_mass_cache": rn_mass_cache,
                    "RN_over_REG_attention_mass_cache": _ratio_float(
                        rn_mass_cache,
                        reg_mass_rn_cache,
                    ),
                    "B13_noRN_REG_OV_norm_cache": reg_ov_no_cache,
                    "B13_RN_REG_OV_norm_cache": reg_ov_rn_cache,
                    "REG_OV_retention_RN_over_noRN": _ratio_float(
                        reg_ov_rn_cache,
                        reg_ov_no_cache,
                    ),
                    "B13_RN_RN_OV_norm_cache": rn_ov_cache,
                    "RN_over_REG_OV_norm_cache": _ratio_float(
                        rn_ov_cache,
                        reg_ov_rn_cache,
                    ),
                    "B13_noRN_hidden_mu_attention_mass_cache": hidden_mass_no_cache,
                    "B13_RN_hidden_mu_attention_mass_cache": hidden_mass_rn_cache,
                    "hidden_mu_attention_mass_retention_RN_over_noRN": _ratio_float(
                        hidden_mass_rn_cache,
                        hidden_mass_no_cache,
                    ),
                    "mean_RN_K_logit_minus_best_REG_K_logit_cache_all_heads": (
                        float(torch.nanmean(key_gap))
                        if key_gap.numel()
                        else float("nan")
                    ),
                }
                image_rows.append(row)

                # ------------------------------------------
                # All-head cache diagnostics.
                # ------------------------------------------
                for h in range(int(layer.self_attn.num_heads)):
                    if c.numel():
                        reg_mass_no_h = float(reg_mass_no_map_h[h, c].mean())
                        reg_mass_rn_h = float(reg_mass_rn_map_h[h, c].mean())
                        rn_mass_h = float(rn_mass_map_h[h, c].mean())
                        reg_ov_no_h = float(reg_head_no_map[h, c].mean())
                        reg_ov_rn_h = float(reg_head_rn_map[h, c].mean())
                        rn_ov_h = float(rn_head_map[h, c].mean())
                        gap_h = float(torch.nanmean(key_gap[h])) if key_gap.shape[1] else float("nan")
                    else:
                        reg_mass_no_h = reg_mass_rn_h = rn_mass_h = float("nan")
                        reg_ov_no_h = reg_ov_rn_h = rn_ov_h = float("nan")
                        gap_h = float("nan")

                    head_rows.append(
                        {
                            "subset": subset,
                            "modality": _modality(subset),
                            "family": _family(subset),
                            "id": image_id,
                            "pair_key": _pair_key(image_id),
                            "head": h,
                            "REG_mass_noRN_cache": reg_mass_no_h,
                            "REG_mass_RN_cache": reg_mass_rn_h,
                            "REG_mass_retention": _ratio_float(
                                reg_mass_rn_h,
                                reg_mass_no_h,
                            ),
                            "RN_mass_cache": rn_mass_h,
                            "RN_over_REG_mass": _ratio_float(
                                rn_mass_h,
                                reg_mass_rn_h,
                            ),
                            "REG_OV_noRN_cache": reg_ov_no_h,
                            "REG_OV_RN_cache": reg_ov_rn_h,
                            "REG_OV_retention": _ratio_float(
                                reg_ov_rn_h,
                                reg_ov_no_h,
                            ),
                            "RN_OV_cache": rn_ov_h,
                            "RN_over_REG_OV": _ratio_float(
                                rn_ov_h,
                                reg_ov_rn_h,
                            ),
                            "RN_K_logit_minus_best_REG_K_logit": gap_h,
                        }
                    )

                # Save compact maps for later rendering.
                overlay_maps[image_id] = {
                    "payload_no": payload_no.detach().cpu().numpy().astype(np.float32),
                    "payload_rn": payload_rn.detach().cpu().numpy().astype(np.float32),
                    "reg_mass_no": reg_mass_no_map.detach().cpu().numpy().astype(np.float32),
                    "reg_mass_rn": reg_mass_rn_map.detach().cpu().numpy().astype(np.float32),
                    "rn_mass": rn_mass_map.detach().cpu().numpy().astype(np.float32),
                    "reg_ov_no": reg_ov_no_map.detach().cpu().numpy().astype(np.float32),
                    "reg_ov_rn": reg_ov_rn_map.detach().cpu().numpy().astype(np.float32),
                    "rn_ov": rn_ov_map.detach().cpu().numpy().astype(np.float32),
                    "regmask": regmask.detach().cpu().numpy().astype(np.uint8),
                    "hidden_mu_mask": hidden_mu.detach().cpu().numpy().astype(np.uint8),
                    "cachemask": cache_mask.detach().cpu().numpy().astype(np.uint8),
                }

                overlay_meta.append(
                    {
                        "map_row_index": len(overlay_meta),
                        "subset": subset,
                        "modality": _modality(subset),
                        "family": _family(subset),
                        "id": image_id,
                        "pair_key": _pair_key(image_id),
                        "correct_label": sample.correct_label,
                        "distractor_label": sample.distractor_label,
                    }
                )

            del (
                pixel_values,
                b12,
                b13_no,
                b13_rn,
                b13_input_no,
                b13_input_rn,
                attn_no,
                attn_rn,
            )

    return image_rows, head_rows, overlay_maps, overlay_meta


# ================================================================================================
# Summaries
# ================================================================================================

def summarize_register_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["state"]), str(row["subset"]))].append(row)
        grouped[(str(row["state"]), "ALL")].append(row)

    output: list[dict[str, Any]] = []
    for state in ("B12_shared", "B13_noRN", "B13_RN", "B20_noRN", "B20_RN"):
        for subset in ("ALL",) + SUBSET_ORDER:
            group = grouped.get((state, subset), [])
            if not group:
                continue
            counts = [int(r["register_count"]) for r in group]
            output.append(
                {
                    "state": state,
                    "subset": subset,
                    "count_images": len(group),
                    "register_count_mean": float(np.mean(counts)),
                    "register_count_median": float(np.median(counts)),
                    "register_count_min": int(np.min(counts)),
                    "register_count_max": int(np.max(counts)),
                    "zero_register_images": int(sum(c == 0 for c in counts)),
                }
            )
    return output


def summarize_hidden_mu(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subset"])].append(row)
        grouped["ALL"].append(row)

    output = []
    for subset in ("ALL",) + SUBSET_ORDER:
        group = grouped.get(subset, [])
        if not group:
            continue
        output.append(
            {
                "subset": subset,
                "count_images": len(group),
                "visible_register_count_mean": _safe_mean(
                    r["visible_register_count"] for r in group
                ),
                "hidden_mu_count_mean": _safe_mean(
                    r["hidden_mu_count"] for r in group
                ),
                "hidden_mu_count_median": _safe_median(
                    r["hidden_mu_count"] for r in group
                ),
                "hidden_mu_count_min": _safe_min(
                    r["hidden_mu_count"] for r in group
                ),
                "hidden_mu_count_max": _safe_max(
                    r["hidden_mu_count"] for r in group
                ),
                "hidden_mu_to_register_ratio_mean": _safe_mean(
                    r["hidden_mu_to_register_ratio"] for r in group
                ),
            }
        )
    return output


def summarize_transport(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["modality"]), str(row["family"]))].append(row)
        grouped[(str(row["modality"]), "ALL")].append(row)
        grouped[("ALL", "ALL")].append(row)

    output = []
    for modality, family in sorted(grouped):
        group = grouped[(modality, family)]
        output.append(
            {
                "modality": modality,
                "family": family,
                "count": len(group),
                "visible_register_count_mean": _safe_mean(
                    r["visible_register_count"] for r in group
                ),
                "hidden_mu_count_mean": _safe_mean(
                    r["hidden_mu_count"] for r in group
                ),
                "REG_attention_mass_retention_mean": _safe_mean(
                    r["REG_attention_mass_retention_RN_over_noRN"] for r in group
                ),
                "REG_attention_mass_retention_median": _safe_median(
                    r["REG_attention_mass_retention_RN_over_noRN"] for r in group
                ),
                "RN_over_REG_attention_mass_mean": _safe_mean(
                    r["RN_over_REG_attention_mass_cache"] for r in group
                ),
                "REG_OV_retention_mean": _safe_mean(
                    r["REG_OV_retention_RN_over_noRN"] for r in group
                ),
                "REG_OV_retention_median": _safe_median(
                    r["REG_OV_retention_RN_over_noRN"] for r in group
                ),
                "RN_over_REG_OV_mean": _safe_mean(
                    r["RN_over_REG_OV_norm_cache"] for r in group
                ),
                "REG_payload_retention_mean": _safe_mean(
                    r["reg_payload_retention_RN_over_noRN"] for r in group
                ),
                "hidden_mu_mass_retention_mean": _safe_mean(
                    r["hidden_mu_attention_mass_retention_RN_over_noRN"] for r in group
                ),
                "RN_K_minus_best_REG_K_mean": _safe_mean(
                    r["mean_RN_K_logit_minus_best_REG_K_logit_cache_all_heads"]
                    for r in group
                ),
            }
        )
    return output


def summarize_heads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["head"])].append(row)

    output = []
    for head in sorted(grouped):
        group = grouped[head]
        output.append(
            {
                "head": head,
                "count": len(group),
                "REG_mass_noRN_mean": _safe_mean(r["REG_mass_noRN_cache"] for r in group),
                "REG_mass_RN_mean": _safe_mean(r["REG_mass_RN_cache"] for r in group),
                "REG_mass_retention_mean": _safe_mean(r["REG_mass_retention"] for r in group),
                "RN_mass_mean": _safe_mean(r["RN_mass_cache"] for r in group),
                "RN_over_REG_mass_mean": _safe_mean(r["RN_over_REG_mass"] for r in group),
                "REG_OV_retention_mean": _safe_mean(r["REG_OV_retention"] for r in group),
                "RN_over_REG_OV_mean": _safe_mean(r["RN_over_REG_OV"] for r in group),
                "RN_K_logit_minus_best_REG_K_mean": _safe_mean(
                    r["RN_K_logit_minus_best_REG_K_logit"] for r in group
                ),
            }
        )
    return output


def summarize_tail_crossblock(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["comparison"])].append(row)

    output = []
    for comparison in sorted(grouped):
        group = grouped[comparison]
        output.append(
            {
                "comparison": comparison,
                "count": len(group),
                "mean_tail_subspace_mean_principal_cos": _safe_mean(
                    r["tail_subspace_mean_principal_cos"] for r in group
                ),
                "median_tail_subspace_mean_principal_cos": _safe_median(
                    r["tail_subspace_mean_principal_cos"] for r in group
                ),
                "mean_tail_subspace_max_principal_cos": _safe_mean(
                    r["tail_subspace_max_principal_cos"] for r in group
                ),
                "mean_tail_centroid_cos": _safe_mean(
                    r["tail_centroid_cos"] for r in group
                ),
            }
        )
    return output


# ================================================================================================
# Plot helpers
# ================================================================================================

def save_invariant_rank_plot(
    output_dir: Path,
    diagnostics: list[dict[str, Any]],
    *,
    threshold: float,
    selected_rank: int,
) -> str:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = [int(row["component_1based"]) for row in diagnostics]
    y = [float(row["min_abs_cosine_across_verifiers"]) for row in diagnostics]
    ax.plot(x, y, marker="o")
    ax.axhline(float(threshold), linewidth=1.0, alpha=0.5)
    ax.axvline(float(selected_rank) + 0.5, linewidth=1.0, alpha=0.35)
    ax.set_xlabel("B12 register SVD component (1-based)")
    ax.set_ylabel("Minimum abs cosine across subset/block verifiers")
    ax.set_title(f"Measured invariant register rank = {selected_rank}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    name = "invariant_rank_diagnostics.png"
    fig.savefig(output_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return name


def save_head_plot(
    output_dir: Path,
    head_summary: list[dict[str, Any]],
) -> str:
    heads = [int(r["head"]) for r in head_summary]
    retention = [float(r["REG_mass_retention_mean"]) for r in head_summary]
    rn_ratio = [float(r["RN_over_REG_mass_mean"]) for r in head_summary]

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.plot(heads, retention, marker="o", label="REG mass retention RN/noRN")
    ax.plot(heads, rn_ratio, marker="o", label="RN / remaining REG mass")
    ax.axhline(1.0, linewidth=1.0, alpha=0.4)
    ax.set_xlabel("B13 attention head")
    ax.set_ylabel("Ratio on noRN-defined cache destinations")
    ax.set_title("B13 REG transport retention and RN dominance by head")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    name = "B13_headwise_REG_vs_RN.png"
    fig.savefig(output_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return name


def _processed_rgb(
    processor: Any,
    sample: PairSample,
    device: torch.device,
) -> np.ndarray:
    pixel_values = preprocess_images(processor, (sample,), device)
    image_processor = getattr(processor, "image_processor", processor)

    mean = torch.tensor(
        getattr(
            image_processor,
            "image_mean",
            (0.48145466, 0.4578275, 0.40821073),
        ),
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        getattr(
            image_processor,
            "image_std",
            (0.26862954, 0.26130258, 0.27577711),
        ),
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)

    rgb = (pixel_values.float() * std + mean).clamp(0.0, 1.0)
    rgb = (rgb[0].permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    return rgb


def _upsample_patch_map(values: np.ndarray, image_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    grid = int(round(math.sqrt(values.size)))
    patch = image_size // grid
    return np.kron(values.reshape(grid, grid), np.ones((patch, patch), dtype=np.float32))


def _rgba_intensity(values: np.ndarray, shared_max: float | None = None) -> np.ndarray:
    """
    Use matplotlib's current default colormap; alpha is normalized intensity.
    No explicit color palette is imposed.
    """
    positive = np.clip(values.astype(np.float32), 0.0, None)
    vmax = float(positive.max()) if shared_max is None else float(shared_max)
    if vmax <= 1e-12:
        norm = np.zeros_like(positive)
    else:
        norm = np.clip(positive / vmax, 0.0, 1.0)
    cmap = plt.get_cmap()
    rgba = cmap(norm)
    rgba[..., 3] = norm
    return rgba


def _draw_patch_boxes(
    ax: Any,
    mask: np.ndarray,
    *,
    image_size: int,
    linestyle: str,
    linewidth: float,
) -> None:
    mask = np.asarray(mask).reshape(-1).astype(bool)
    grid = int(round(math.sqrt(mask.size)))
    patch = image_size // grid
    for index in np.flatnonzero(mask):
        r, c = divmod(int(index), grid)
        rect = plt.Rectangle(
            (c * patch, r * patch),
            patch,
            patch,
            fill=False,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        ax.add_patch(rect)


def _split_patch_overlay(
    left_values: np.ndarray,
    right_values: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """
    Same default colormap on both halves; left/right geometry is the label.
    Joint normalization preserves relative magnitude.
    """
    left = np.clip(np.asarray(left_values, dtype=np.float32), 0.0, None)
    right = np.clip(np.asarray(right_values, dtype=np.float32), 0.0, None)
    shared = max(float(left.max()), float(right.max()), 1e-12)

    grid = int(round(math.sqrt(left.size)))
    patch = image_size // grid
    half = patch // 2

    canvas = np.zeros((image_size, image_size, 4), dtype=np.float32)
    cmap = plt.get_cmap()

    for idx in range(left.size):
        r, c = divmod(idx, grid)
        y0 = r * patch
        x0 = c * patch

        lv = float(np.clip(left[idx] / shared, 0.0, 1.0))
        rv = float(np.clip(right[idx] / shared, 0.0, 1.0))

        lc = np.array(cmap(lv), dtype=np.float32)
        rc = np.array(cmap(rv), dtype=np.float32)
        lc[3] = lv
        rc[3] = rv

        canvas[y0 : y0 + patch, x0 : x0 + half] = lc
        canvas[y0 : y0 + patch, x0 + half : x0 + patch] = rc

    return canvas


def render_overlay_panel(
    *,
    sample: PairSample,
    image_row: dict[str, Any],
    maps: dict[str, np.ndarray],
    rgb: np.ndarray,
    output_path: Path,
) -> None:
    image_size = int(rgb.shape[0])
    P = int(maps["regmask"].size)
    grid = int(round(math.sqrt(P)))

    regmask = maps["regmask"].astype(bool)
    hidden = maps["hidden_mu_mask"].astype(bool)
    cache = maps["cachemask"].astype(bool)

    payload_shared = max(
        float(maps["payload_no"].max()),
        float(maps["payload_rn"].max()),
        1e-12,
    )
    ov_shared = max(
        float(maps["reg_ov_no"].max()),
        float(maps["reg_ov_rn"].max()),
        float(maps["rn_ov"].max()),
        1e-12,
    )

    payload_no_rgba = _rgba_intensity(
        _upsample_patch_map(maps["payload_no"], image_size),
        shared_max=payload_shared,
    )
    payload_rn_rgba = _rgba_intensity(
        _upsample_patch_map(maps["payload_rn"], image_size),
        shared_max=payload_shared,
    )
    reg_ov_no_rgba = _rgba_intensity(
        _upsample_patch_map(maps["reg_ov_no"], image_size),
        shared_max=ov_shared,
    )
    reg_ov_rn_rgba = _rgba_intensity(
        _upsample_patch_map(maps["reg_ov_rn"], image_size),
        shared_max=ov_shared,
    )
    rn_ov_rgba = _rgba_intensity(
        _upsample_patch_map(maps["rn_ov"], image_size),
        shared_max=ov_shared,
    )

    split_original_vs_rn = _split_patch_overlay(
        maps["reg_ov_no"],
        maps["rn_ov"],
        image_size,
    )
    split_remaining_vs_rn = _split_patch_overlay(
        maps["reg_ov_rn"],
        maps["rn_ov"],
        image_size,
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    fig.suptitle(
        f"{sample.sample_id} | {sample.subset} | "
        f"{sample.correct_label} vs {sample.distractor_label}\n"
        f"REG mass retention={image_row['REG_attention_mass_retention_RN_over_noRN']:.3f} | "
        f"REG OV retention={image_row['REG_OV_retention_RN_over_noRN']:.3f} | "
        f"RN/REG OV={image_row['RN_over_REG_OV_norm_cache']:.3f}",
        fontsize=12,
    )

    # 1
    ax = axes[0, 0]
    ax.imshow(rgb)
    _draw_patch_boxes(
        ax,
        regmask,
        image_size=image_size,
        linestyle="-",
        linewidth=1.6,
    )
    _draw_patch_boxes(
        ax,
        hidden,
        image_size=image_size,
        linestyle="--",
        linewidth=1.2,
    )
    _draw_patch_boxes(
        ax,
        cache,
        image_size=image_size,
        linestyle=":",
        linewidth=1.5,
    )
    ax.set_title(
        f"B12 groups | REG={int(regmask.sum())}, hiddenμ={int(hidden.sum())}, cache={int(cache.sum())}\n"
        "solid=REG, dashed=hiddenμ, dotted=cache"
    )
    ax.axis("off")

    # 2
    ax = axes[0, 1]
    ax.imshow(rgb)
    ax.imshow(payload_no_rgba)
    _draw_patch_boxes(ax, cache, image_size=image_size, linestyle=":", linewidth=1.3)
    ax.set_title("B13 noRN: de-carriered B12 REG-tail subspace")
    ax.axis("off")

    # 3
    ax = axes[0, 2]
    ax.imshow(rgb)
    ax.imshow(payload_rn_rgba)
    _draw_patch_boxes(ax, cache, image_size=image_size, linestyle=":", linewidth=1.3)
    ax.set_title("B13 RN: remaining original REG-tail subspace")
    ax.axis("off")

    # 4
    ax = axes[0, 3]
    ax.imshow(rgb)
    ax.imshow(rn_ov_rgba)
    _draw_patch_boxes(ax, cache, image_size=image_size, linestyle=":", linewidth=1.3)
    ax.set_title("B13 RN: actual RN-source OV write")
    ax.axis("off")

    # 5
    ax = axes[1, 0]
    ax.imshow(rgb)
    ax.imshow(reg_ov_no_rgba)
    _draw_patch_boxes(ax, cache, image_size=image_size, linestyle=":", linewidth=1.3)
    ax.set_title("B13 noRN: actual visible-REG OV write")
    ax.axis("off")

    # 6
    ax = axes[1, 1]
    ax.imshow(rgb)
    ax.imshow(reg_ov_rn_rgba)
    _draw_patch_boxes(ax, cache, image_size=image_size, linestyle=":", linewidth=1.3)
    ax.set_title("B13 RN: remaining visible-REG OV write")
    ax.axis("off")

    # 7
    ax = axes[1, 2]
    ax.imshow(rgb)
    ax.imshow(split_original_vs_rn)
    _draw_patch_boxes(ax, cache, image_size=image_size, linestyle=":", linewidth=1.3)
    ax.set_title("Split patches: LEFT=noRN REG write | RIGHT=RN write")
    ax.axis("off")

    # 8
    ax = axes[1, 3]
    ax.imshow(rgb)
    ax.imshow(split_remaining_vs_rn)
    _draw_patch_boxes(ax, cache, image_size=image_size, linestyle=":", linewidth=1.3)
    ax.set_title("Split patches: LEFT=remaining REG | RIGHT=RN")
    ax.axis("off")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_gallery(
    image_paths: list[Path],
    output_path: Path,
    *,
    cols: int = 2,
    thumb_width: int = 1100,
    gutter: int = 18,
) -> None:
    if not image_paths:
        return

    images = [Image.open(path).convert("RGB") for path in image_paths]
    resized = []
    for img in images:
        height = int(round(img.height * thumb_width / img.width))
        resized.append(img.resize((thumb_width, height)))

    rows = int(math.ceil(len(resized) / cols))
    col_width = thumb_width
    row_heights = []
    for r in range(rows):
        chunk = resized[r * cols : (r + 1) * cols]
        row_heights.append(max(img.height for img in chunk))

    width = cols * col_width + (cols + 1) * gutter
    height = sum(row_heights) + (rows + 1) * gutter
    # Do not choose an explicit plotting palette; PIL default black canvas is fine.
    canvas = Image.new("RGB", (width, height))

    y = gutter
    for r in range(rows):
        x = gutter
        chunk = resized[r * cols : (r + 1) * cols]
        for img in chunk:
            canvas.paste(img, (x, y))
            x += col_width + gutter
        y += row_heights[r] + gutter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


# ================================================================================================
# Main
# ================================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure B12 register invariant rank, hidden-mu population, and "
            "B13 visible-REG -> cache transport hijacking by RN."
        )
    )
    parser.add_argument("--full-model", type=str, default=DEFAULT_FULL_MODEL,
                        help="HF repo id or local HF model directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--calibration-per-subset",
        type=int,
        default=DEFAULT_CALIBRATION_PER_SUBSET,
    )
    parser.add_argument(
        "--max-invariant-rank",
        type=int,
        default=DEFAULT_MAX_INVARIANT_RANK,
    )
    parser.add_argument(
        "--invariant-cos-threshold",
        type=float,
        default=DEFAULT_INVARIANT_COS_THRESHOLD,
    )
    parser.add_argument(
        "--register-norm-threshold",
        type=float,
        default=DEFAULT_REGISTER_NORM_THRESHOLD,
    )
    parser.add_argument(
        "--hidden-mu-cos-threshold",
        type=float,
        default=None,
        help=(
            "Optional override. Default is measured by a two-cluster fit to "
            "normal-norm B12 patch cosine with the rank-1 register carrier."
        ),
    )
    parser.add_argument("--cache-topk", type=int, default=DEFAULT_CACHE_TOPK)
    parser.add_argument(
        "--overlay-n-per-modality",
        type=int,
        default=DEFAULT_OVERLAY_N_PER_MODALITY,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--heads-of-interest",
        type=int,
        nargs="+",
        default=list(DEFAULT_HEADS_OF_INTEREST),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA FP16 outer autocast for stock ViT forwards.",
    )
    parser.add_argument("--skip-overlays", action="store_true")
    args = parser.parse_args()

    if args.calibration_per_subset <= 0:
        raise ValueError("--calibration-per-subset must be positive")
    if args.max_invariant_rank <= 0:
        raise ValueError("--max-invariant-rank must be positive")
    if not (0.0 < args.invariant_cos_threshold <= 1.0):
        raise ValueError("--invariant-cos-threshold must be in (0,1]")
    if args.cache_topk <= 0:
        raise ValueError("--cache-topk must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    configure_reproducibility(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[run] device={device}")
    print(f"[run] model={args.full_model}")
    print(f"[run] output={output_dir}")
    print(
        f"[run] calibration={args.calibration_per_subset} paired images/subset "
        f"({args.calibration_per_subset * len(SUBSET_ORDER)} total)"
    )

    all_samples = load_samples()
    selected = select_paired_calibration(
        all_samples,
        per_subset=args.calibration_per_subset,
        seed=args.seed,
    )

    print("[dataset] selected paired calibration:")
    for subset in SUBSET_ORDER:
        print(f"  {subset:10s}: {len(selected[subset])}")

    print(f"[model] Loading {args.full_model}")
    model = AutoModel.from_pretrained(
        args.full_model,
        trust_remote_code=True,
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(
        args.full_model,
        trust_remote_code=True,
    )

    if not hasattr(model, "read_null_token"):
        raise TypeError("Expected full x-attention model with read_null_token")

    rn_insert_block = int(model.config.read_null_insert_block)
    if rn_insert_block != B13:
        raise ValueError(
            f"This microscope targets RN insertion before B13; model says B{rn_insert_block}"
        )

    vision_layers = len(model.vision_model.encoder.layers)
    if vision_layers <= B20:
        raise ValueError(f"Need at least B20, model has {vision_layers} layers")

    validate_attention_architecture(model.vision_model.encoder.layers[B13])

    # --------------------------------------------------------------------------------------------
    # Stage 1: visible registers and rank-n invariant subspace.
    # --------------------------------------------------------------------------------------------
    (
        by_state_subset,
        by_state_all,
        per_image_registers,
        register_count_rows,
        b12_spatial_by_id,
        b12_regmask_by_key,
    ) = collect_register_calibration(
        model,
        processor,
        selected,
        rn_insert_block=rn_insert_block,
        norm_threshold=args.register_norm_threshold,
        batch_size=args.batch_size,
        device=device,
        amp=args.amp,
    )

    register_count_summary = summarize_register_counts(register_count_rows)
    _write_csv(output_dir / "visible_register_counts_by_image.csv", register_count_rows)
    _write_csv(output_dir / "visible_register_counts_summary.csv", register_count_summary)

    print()
    print("[REG] visible high-norm population (NO CAP, NO FALLBACK):")
    for row in register_count_summary:
        if row["subset"] == "ALL":
            print(
                f"  {row['state']:10s}: mean={row['register_count_mean']:.2f} "
                f"median={row['register_count_median']:.1f} "
                f"min={row['register_count_min']} max={row['register_count_max']} "
                f"zero={row['zero_register_images']}"
            )

    bases, singulars = build_all_bases(
        by_state_subset,
        by_state_all,
        max_rank=args.max_invariant_rank,
        seed=args.seed,
    )

    invariant_basis, invariant_rank, invariant_diag = select_invariant_rank(
        bases,
        singulars,
        by_state_all[("B12_shared", "ALL")],
        max_rank=args.max_invariant_rank,
        cos_threshold=args.invariant_cos_threshold,
    )

    np.save(
        output_dir / "B12_measured_invariant_register_basis.npy",
        invariant_basis.cpu().numpy().astype(np.float32),
    )
    _write_csv(
        output_dir / "invariant_rank_diagnostics.csv",
        invariant_diag,
    )
    invariant_plot = save_invariant_rank_plot(
        output_dir,
        invariant_diag,
        threshold=args.invariant_cos_threshold,
        selected_rank=invariant_rank,
    )

    print()
    print(f"[SVD] selected invariant rank = {invariant_rank}")
    for row in invariant_diag[: min(8, len(invariant_diag))]:
        print(
            f"  PC{row['component_1based']:02d}: "
            f"min verifier |cos|={row['min_abs_cosine_across_verifiers']:.6f} "
            f"{'KEEP' if row['accepted_contiguously'] else 'stop/data'}"
        )

    # Rank-1 carrier direction specifically for hidden-mu finding.
    carrier_u1 = invariant_basis[:, 0].clone()

    hidden_mu_threshold, hidden_mu_info, hidden_mu_rows = measure_hidden_mu_threshold(
        b12_spatial_by_id,
        b12_regmask_by_key,
        selected,
        carrier_u1,
        override_threshold=args.hidden_mu_cos_threshold,
    )
    hidden_mu_summary = summarize_hidden_mu(hidden_mu_rows)

    _write_csv(output_dir / "hidden_mu_counts_by_image.csv", hidden_mu_rows)
    _write_csv(output_dir / "hidden_mu_counts_summary.csv", hidden_mu_summary)
    (output_dir / "hidden_mu_detection.json").write_text(
        json.dumps(hidden_mu_info, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "[hidden μ] rank-1 carrier cosine clusters: "
        f"low={hidden_mu_info['low_cluster_center']:.5f}, "
        f"high={hidden_mu_info['high_cluster_center']:.5f}, "
        f"threshold={hidden_mu_threshold:.5f}"
    )
    all_hidden = next(row for row in hidden_mu_summary if row["subset"] == "ALL")
    print(
        f"[hidden μ] mean visible REG={all_hidden['visible_register_count_mean']:.2f}, "
        f"mean hiddenμ={all_hidden['hidden_mu_count_mean']:.2f}, "
        f"mean hiddenμ/REG={all_hidden['hidden_mu_to_register_ratio_mean']:.3f}"
    )

    # B13/B20 tail data-change double-check.
    tail_rows = register_tail_crossblock_rows(
        per_image_registers,
        selected,
        invariant_basis,
    )
    tail_summary = summarize_tail_crossblock(tail_rows)
    _write_csv(output_dir / "register_tail_crossblock_by_image.csv", tail_rows)
    _write_csv(output_dir / "register_tail_crossblock_summary.csv", tail_summary)

    print()
    print("[REG tail] de-invarianted cross-block subspace similarity:")
    for row in tail_summary:
        print(
            f"  {row['comparison']:18s}: "
            f"mean principal cos={row['mean_tail_subspace_mean_principal_cos']:.4f}, "
            f"centroid cos={row['mean_tail_centroid_cos']:.4f}"
        )

    # --------------------------------------------------------------------------------------------
    # Constant RN head-OV rank check.
    # --------------------------------------------------------------------------------------------
    first_sample = selected[SUBSET_ORDER[0]][0]
    pv = preprocess_images(processor, (first_sample,), device)
    with _autocast(device, args.amp):
        # Obtain shared post-B12 state.
        hidden = model.vision_model.embeddings(pv, interpolate_pos_encoding=False)
        hidden = model.vision_model.pre_layrnorm(hidden)
        for block in range(rn_insert_block):
            hidden = model.vision_model.encoder.layers[block](hidden, None)

    rn = model.read_null_token.to(device=hidden.device, dtype=hidden.dtype)
    b13_input_rn_single = torch.cat(
        (hidden, rn.view(1, 1, -1)),
        dim=1,
    )
    rn_rank_info = rn_head_ov_rank1_fraction(
        model.vision_model.encoder.layers[B13],
        b13_input_rn_single,
    )
    del pv, hidden, b13_input_rn_single

    print(
        f"[RN V/OV] rank-1 energy across 16 head-specific RN OV directions: "
        f"{rn_rank_info['rn_head_ov_rank1_energy_fraction']:.6f}"
    )

    # --------------------------------------------------------------------------------------------
    # Stage 2: same cache addresses, compare REG vs RN actual B13 transport.
    # --------------------------------------------------------------------------------------------
    (
        transport_rows,
        head_rows,
        overlay_maps,
        overlay_meta,
    ) = collect_cache_transport(
        model,
        processor,
        selected,
        invariant_basis=invariant_basis,
        carrier_u1=carrier_u1,
        hidden_mu_threshold=hidden_mu_threshold,
        register_norm_threshold=args.register_norm_threshold,
        cache_topk=args.cache_topk,
        batch_size=args.batch_size,
        device=device,
        amp=args.amp,
    )

    transport_summary = summarize_transport(transport_rows)
    head_summary = summarize_heads(head_rows)

    _write_csv(output_dir / "cache_transport_by_image.csv", transport_rows)
    _write_csv(output_dir / "cache_transport_summary.csv", transport_summary)
    _write_csv(output_dir / "cache_transport_by_head_by_image.csv", head_rows)
    _write_csv(output_dir / "cache_transport_head_summary.csv", head_summary)

    # Save maps in one compressed archive aligned by explicit IDs.
    map_ids = [item["id"] for item in overlay_meta]
    map_names = (
        "payload_no",
        "payload_rn",
        "reg_mass_no",
        "reg_mass_rn",
        "rn_mass",
        "reg_ov_no",
        "reg_ov_rn",
        "rn_ov",
        "regmask",
        "hidden_mu_mask",
        "cachemask",
    )
    npz_payload = {"ids": np.array(map_ids, dtype=object)}
    for name in map_names:
        npz_payload[name] = np.stack([overlay_maps[i][name] for i in map_ids], axis=0)
    np.savez_compressed(output_dir / "cache_transport_maps.npz", **npz_payload)
    _write_csv(output_dir / "cache_transport_maps_meta.csv", overlay_meta)

    head_plot = save_head_plot(output_dir, head_summary)

    print()
    print("=" * 118)
    print("B13 REG -> CACHE RETENTION / RN HIJACK")
    print("=" * 118)
    print(
        f"{'modality':<12} {'family':<6} {'n':>5} "
        f"{'REG mass ret':>13} {'RN/REG mass':>12} "
        f"{'REG OV ret':>11} {'RN/REG OV':>10} "
        f"{'payload ret':>11} {'RN K gap':>10}"
    )
    print("-" * 118)
    for row in transport_summary:
        if row["family"] != "ALL":
            continue
        print(
            f"{row['modality']:<12} {row['family']:<6} {row['count']:>5d} "
            f"{row['REG_attention_mass_retention_mean']:>13.4f} "
            f"{row['RN_over_REG_attention_mass_mean']:>12.4f} "
            f"{row['REG_OV_retention_mean']:>11.4f} "
            f"{row['RN_over_REG_OV_mean']:>10.4f} "
            f"{row['REG_payload_retention_mean']:>11.4f} "
            f"{row['RN_K_minus_best_REG_K_mean']:>+10.4f}"
        )

    print()
    print("[heads] requested heads of interest:")
    head_lookup = {int(row["head"]): row for row in head_summary}
    for h in args.heads_of_interest:
        if h not in head_lookup:
            print(f"  H{h}: unavailable")
            continue
        row = head_lookup[h]
        print(
            f"  H{h}: REG mass retention={row['REG_mass_retention_mean']:.4f}, "
            f"RN/REG mass={row['RN_over_REG_mass_mean']:.4f}, "
            f"REG OV retention={row['REG_OV_retention_mean']:.4f}, "
            f"RN/REG OV={row['RN_over_REG_OV_mean']:.4f}, "
            f"RN K-bestREG K={row['RN_K_logit_minus_best_REG_K_mean']:+.4f}"
        )

    # --------------------------------------------------------------------------------------------
    # Overlays: select strongest RN-dominance cases from attacked handwriting/digital.
    # --------------------------------------------------------------------------------------------
    overlay_files: list[str] = []
    if not args.skip_overlays:
        row_by_id = {row["id"]: row for row in transport_rows}
        sample_by_id = {
            sample.sample_id: sample
            for subset in SUBSET_ORDER
            for sample in selected[subset]
        }

        selected_overlay_rows: list[dict[str, Any]] = []

        for modality in ("handwritten", "digital"):
            group = [
                row
                for row in transport_rows
                if row["modality"] == modality
                and row["subset"] in ATTACK_SUBSETS
                and math.isfinite(float(row["RN_over_REG_OV_norm_cache"]))
            ]
            group.sort(
                key=lambda row: (
                    -float(row["RN_over_REG_OV_norm_cache"]),
                    float(row["REG_OV_retention_RN_over_noRN"]),
                )
            )
            selected_overlay_rows.extend(group[: args.overlay_n_per_modality])

        # Also a few no-attack controls with strongest RN dominance.
        controls = [
            row
            for row in transport_rows
            if row["subset"] in NO_TEXT_SUBSETS
            and math.isfinite(float(row["RN_over_REG_OV_norm_cache"]))
        ]
        controls.sort(
            key=lambda row: -float(row["RN_over_REG_OV_norm_cache"])
        )
        selected_overlay_rows.extend(controls[: max(2, args.overlay_n_per_modality // 2)])

        panel_dir = output_dir / "per_image_REG_vs_RN_panels"
        panel_dir.mkdir(parents=True, exist_ok=True)

        gallery_groups: dict[str, list[Path]] = defaultdict(list)
        selection_rows: list[dict[str, Any]] = []

        for row in tqdm(selected_overlay_rows, desc="render REG/RN overlays", leave=False):
            image_id = row["id"]
            sample = sample_by_id[image_id]
            rgb = _processed_rgb(processor, sample, device)

            safe_id = image_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            panel_path = panel_dir / f"{row['modality']}__{row['subset']}__{safe_id}.png"

            render_overlay_panel(
                sample=sample,
                image_row=row,
                maps=overlay_maps[image_id],
                rgb=rgb,
                output_path=panel_path,
            )

            gallery_groups[row["modality"]].append(panel_path)
            selection_rows.append(
                {
                    "panel_path": str(panel_path),
                    **row,
                }
            )

        _write_csv(output_dir / "selected_overlay_examples.csv", selection_rows)

        for group_name, paths in gallery_groups.items():
            gallery_path = output_dir / f"gallery_REG_vs_RN_{group_name}.png"
            make_gallery(paths, gallery_path, cols=2)
            overlay_files.append(str(gallery_path))

        combined = [
            path
            for key in ("handwritten", "digital", "no_attack")
            for path in gallery_groups.get(key, [])
        ]
        combined_path = output_dir / "gallery_REG_vs_RN_combined.png"
        make_gallery(combined, combined_path, cols=2)
        overlay_files.append(str(combined_path))

    # --------------------------------------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------------------------------------
    metadata = {
        "full_model": str(args.full_model),
        "device": str(device),
        "amp": bool(args.amp),
        "calibration_per_subset": int(args.calibration_per_subset),
        "selected_total_images": int(
            sum(len(selected[s]) for s in SUBSET_ORDER)
        ),
        "paired_selection": True,
        "blocks_zero_based": {
            "mature_register_source": B12,
            "hijack_block": B13,
            "late_invariant_check": B20,
        },
        "rn_insert_block": rn_insert_block,
        "register_norm_threshold": float(args.register_norm_threshold),
        "register_cap": None,
        "register_minimum_fallback": None,
        "max_invariant_rank_tested": int(args.max_invariant_rank),
        "invariant_cos_threshold": float(args.invariant_cos_threshold),
        "measured_invariant_rank": int(invariant_rank),
        "invariant_rank_diagnostics_plot": invariant_plot,
        "hidden_mu": hidden_mu_info,
        "cache_topk": int(args.cache_topk),
        "cache_definition": (
            "Top normal-norm/non-hidden-mu B13-noRN patches by projection fraction "
            "onto the per-image de-invarianted B12 visible-register tail subspace."
        ),
        "heads_of_interest": [int(h) for h in args.heads_of_interest],
        "head_plot": head_plot,
        "rn_head_ov": rn_rank_info,
        "overlay_files": overlay_files,
        "important_distinctions": [
            (
                "REG-tail localization is measured in residual space after subtracting "
                "only the empirically invariant register basis."
            ),
            (
                "REG/RN transport uses actual B13 self-attention probabilities and OV "
                "source-group contributions.  Residual-tail subtraction is not treated "
                "as linearly equivalent to V-space subtraction through LayerNorm."
            ),
            (
                "Visible registers are raw norm>threshold only: no cap, no forced minimum."
            ),
            (
                "Hidden-mu count is measured from the rank-1 carrier cosine cluster and "
                "is never forced to the expected ~2x visible-register count."
            ),
        ],
    }

    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"[run] Wrote results to: {output_dir}")
    print("[run] Primary outputs:")
    for name in (
        "visible_register_counts_summary.csv",
        "invariant_rank_diagnostics.csv",
        "B12_measured_invariant_register_basis.npy",
        "hidden_mu_detection.json",
        "hidden_mu_counts_summary.csv",
        "register_tail_crossblock_summary.csv",
        "cache_transport_summary.csv",
        "cache_transport_head_summary.csv",
        "cache_transport_by_image.csv",
        "cache_transport_maps.npz",
        "B13_headwise_REG_vs_RN.png",
        "invariant_rank_diagnostics.png",
        "gallery_REG_vs_RN_combined.png",
        "per_image_REG_vs_RN_panels/",
        "run_metadata.json",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
