#!/usr/bin/env python3
'\nCLIP CRAZY-SPACE CARTOGRAPHY\nRN local control-manifold atlas on RTA-100 Multilingual\n======================================================\n\nThis script deliberately goes beyond the 1-D alpha sweeps.\n\nIt maps a local chart of the learned RN B13 control subspace and asks whether\nthe reachable downstream states form a privileged, structured low-dimensional\nobject rather than "any random nonlinear slice through a transformer".\n\nRequired beside this script:\n    probe_tools_rn_control.py\n\nDefault dataset:\n    zer0int/RTA-100-Multilingual\n\nDefault languages:\n    en,de,ar,zh,ru\n\nWhat it computes\n----------------\nA) DENSE 2-D CONTROL SURFACES\n   - PC1 x PC2 and PC1 x PC4\n   - 17 x 17 grid over [-2, 2]^2\n   - true RN basis\n   - matched random-orthogonal feature directions\n   - SynthRTA and paired NoRTA controls\n   - English and native-language READ queries\n   - full scalar dashboard + B21 patch-mean state + final embedding\n\nB) LOCAL SURFACE GEOMETRY\n   - B21-state tangent singular values across every grid cell\n   - tangent anisotropy / local area element\n   - scalar relative-READ gradient + Hessian eigenvalues\n   - PCA3 and optional Isomap3 coordinates\n   - ASCII PLY triangle meshes for "surface gazing"\n   - full high-D mean state arrays remain in NPZ\n\nC) CROSS-LANGUAGE "SHADOW" TESTS\n   - correlation of pairwise state-space distances\n   - local nearest-neighbor overlap\n   - tangent-plane principal-cosine alignment\n   - low-D orthogonal Procrustes residual\n   - scalar behavioral-surface correlation\n\nD) 4-D RIDGE / BASIN SEARCH\n   - Sobol search in PC1..PC4 under an L2 bound\n   - optional local refinement around max/min READ points\n   - top-k ridge / trough point clouds\n   - PCA of alpha-space ridge loci\n   - same search in matched random orthogonal directions\n\nE) 4-D LOCAL STATE JACOBIAN\n   - at alpha=0 and at discovered ridge maxima/minima\n   - B21 patch-mean Jacobian singular spectrum\n   - local effective rank / participation ratio\n\nF) "ASK CLIP THE OTHER WAY" / CLIP-ESE\n   - caches vocab_deduped.txt as <text> candidate Q + text embeddings\n   - forced-READ search over the whole vocabulary at selected manifold landmarks\n   - top-10 entries\n   - actual CLIP re-encoding of ordered 2-word phrases\n   - pair beam -> 3-word phrases\n   - both space-joined and concatenated CLIPese compounds\n   - deliberately only at representative landmarks, not every grid cell\n\nImportant matched-random control\n--------------------------------\nFor PC j, the real RN component is:\n    c_j(token) = <delta_token, u_j> u_j\n\nThe random control preserves the exact per-token coefficient <delta,u_j> but\nreplaces u_j with a random unit vector r_j orthogonal to the entire learned RN\nbasis:\n    c_j^rand(token) = <delta_token, u_j> r_j\n\nThus token locations and scalar amplitudes are matched; only the feature\ndirection is rotated out of the learned RN control subspace.\n\nRN-context intervention convention\n----------------------------------\nAs in the previous knob suite:\n    ordinary B13 state = RN-OFF state + controlled subspace perturbation\nwhile retaining the actual learned RN token state from the genuine B13 RN run\nfor downstream B20/B21 READ_NULL machinery.\n\nalpha=0 therefore means:\n    RN exists downstream, but its B13 write into ordinary tokens is removed.\n\nDefault scale\n-------------\nThe atlas defaults to 10 exact shared sample_keys x 5 languages.  That is already\na lot of geometry.  Increase --atlas-keys if Mr. Bigglesworth demands more.\n\nOutputs\n-------\n  rn_control_manifold_atlas/\n    data/\n      surface_raw.csv\n      surface_summary.csv\n      surface_local_geometry.csv\n      surface_state_means.npz\n      cross_language_geometry.csv\n      behavior_surface_similarity.csv\n      ridge_raw.csv\n      ridge_extrema.csv\n      ridge_locus_summary.csv\n      state_jacobian_raw.csv\n      state_jacobian_summary.csv\n      clipese_top10.csv\n      clipese_best_phrases.csv\n      ...\n    ply/\n      *.ply\n    plots/\n      selected surface / geometry / ridge figures\n    SUMMARY.txt\n    compact_summary_rn_control_manifold.zip\n\nThe compact summary ZIP excludes the giant raw CSVs and reusable vocab cache, but\nkeeps aggregate tables, PLY meshes, semantic landmark results and selected plots.\n'
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()


import argparse
import csv
import hashlib
import json
import math
import random
import sys
import zipfile
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import probe_tools_rn_control as base


DEFAULT_CHECKPOINT = base.DEFAULT_CHECKPOINT
DEFAULT_DATASET_REPO = "zer0int/RTA-100-Multilingual"
DEFAULT_LANGUAGES = "en,de,ar,zh,ru"
DEFAULT_QUERY_MODES = "english,native"
DEFAULT_PLANES = "1x2,1x4"
DEFAULT_GRID_POINTS = 17
DEFAULT_GRID_BOUND = 2.0
DEFAULT_ATLAS_KEYS = 10
DEFAULT_BASIS_PAIRS_PER_LANGUAGE = 6
DEFAULT_SURFACE_BATCH = 48
DEFAULT_RIDGE_POINTS = 768
DEFAULT_RIDGE_BOUND = 2.0
DEFAULT_RIDGE_RADIUS = 3.0
DEFAULT_RIDGE_BATCH = 64
DEFAULT_RIDGE_TOPK = 16
DEFAULT_RIDGE_REFINE_STEPS = 2
DEFAULT_RIDGE_REFINE_RANDOM = 32
DEFAULT_STATE_JAC_EPS = 0.20
DEFAULT_VOCAB = "vocab_deduped.txt"
DEFAULT_VOCAB_TOPK = 10
DEFAULT_VOCAB_BATCH = 1024
DEFAULT_VOCAB_SCORE_BATCH = 1024
DEFAULT_PAIR_BEAM = 12
DEFAULT_RANDOM_SEED = 20260902

STATE_BLOCK = 21


# =============================================================================
# Basic helpers
# =============================================================================

def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    base.save_rows(path, rows)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_json(path: Path, payload: Any) -> None:
    base.save_json(path, payload)


def parse_strs(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_planes(text: str) -> list[tuple[int, int]]:
    out = []
    for part in parse_strs(text):
        bits = part.lower().replace("pc", "").split("x")
        if len(bits) != 2:
            raise ValueError(f"Bad plane {part!r}; use e.g. 1x2,1x4")
        a, b = int(bits[0]), int(bits[1])
        if a == b or min(a, b) < 1:
            raise ValueError(part)
        out.append((a, b))
    return out


def safe_float(x: Any) -> float:
    return base.safe_float(x)


def stable_slug(text: str) -> str:
    return base.stable_slug(text)


def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return base.normalize(x, dim=dim, eps=eps)


def row_rankdata(x: np.ndarray) -> np.ndarray:
    """Simple average-rank-free rank transform; ties are rare for our distances."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok], b[ok]
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-12 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(row_rankdata(np.asarray(a).reshape(-1)),
                   row_rankdata(np.asarray(b).reshape(-1)))


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    xx = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(xx + xx.T - 2.0 * (x @ x.T), 0.0)
    return np.sqrt(d2)


def upper_triangle_values(d: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(d.shape[0], k=1)
    return d[i, j]


def effective_rank_from_singular_values(s: np.ndarray) -> float:
    s = np.asarray(s, np.float64)
    e = s * s
    den = float(np.sum(e * e))
    if den <= 1e-20:
        return 0.0
    return float((e.sum() ** 2) / den)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Learned + matched-random feature bases
# =============================================================================

def random_basis_orthogonal_to(
    learned_basis_kd: torch.Tensor,
    count: int,
    seed: int,
) -> torch.Tensor:
    """
    Generate count unit feature directions orthogonal to the full learned basis.
    """
    b = learned_basis_kd.float()
    qlearn = torch.linalg.qr(b.T, mode="reduced").Q  # [D,K]
    g = torch.Generator(device=b.device)
    g.manual_seed(seed)

    cols = []
    while len(cols) < count:
        v = torch.randn(b.shape[1], generator=g, device=b.device, dtype=torch.float32)
        v = v - qlearn @ (qlearn.T @ v)
        if cols:
            qprev = torch.stack(cols, dim=1)
            v = v - qprev @ (qprev.T @ v)
        n = v.norm()
        if float(n) > 1e-5:
            cols.append(v / n)
    return torch.stack(cols, dim=0)  # [count,D]


def per_pc_components(
    delta_btd: torch.Tensor,
    learned_basis_kd: torch.Tensor,
    random_basis_kd: torch.Tensor,
    pc_count: int = 4,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Return true and matched-random per-PC components [1,T,D].

    Random uses the SAME tokenwise coefficients from learned PC j.
    """
    d = delta_btd.float()
    rn = []
    rnd = []
    for j in range(pc_count):
        u = learned_basis_kd[j].float()
        r = random_basis_kd[j].float()
        coeff = torch.einsum("btd,d->bt", d, u)  # [1,T]
        rn.append(coeff[..., None] * u[None, None, :])
        rnd.append(coeff[..., None] * r[None, None, :])
    return rn, rnd


# =============================================================================
# Arbitrary alpha intervention batches
# =============================================================================

def make_alpha_post_batch(
    pair: base.B13Pair,
    components: Sequence[torch.Tensor],
    alphas_np: np.ndarray,
) -> torch.Tensor:
    """
    alphas_np [B,K].  components K x [1,T,D].
    Returns post-B13 [T+RN,B,D], retaining the genuine B13 RN token.
    """
    a = torch.as_tensor(alphas_np, dtype=torch.float32, device=pair.delta_ord_btd.device)
    comps = torch.cat([c.float() for c in components], dim=0)  # [K,T,D]
    delta = torch.einsum("bk,ktd->btd", a, comps)
    base_btd = pair.base_post_tbc.permute(1, 0, 2).float()
    ordinary = base_btd + delta
    ordinary_tbc = ordinary.to(pair.base_post_tbc.dtype).permute(1, 0, 2)
    rn = pair.full_post_tbc[-1:].expand(-1, ordinary.shape[0], -1)
    return torch.cat([ordinary_tbc, rn], dim=0)


def anchor_run(
    model: torch.nn.Module,
    pair: base.B13Pair,
    capture: set[int],
) -> dict[str, Any]:
    zeros = np.zeros((1, 1), dtype=np.float32)
    post = make_alpha_post_batch(pair, [torch.zeros_like(pair.delta_ord_btd)], zeros)
    return base.run_downstream(model, post, pair.early_states, capture, has_rn=True)


def state_vector_from_run(
    run: dict[str, Any],
    block: int = STATE_BLOCK,
) -> torch.Tensor:
    """
    B21 ordinary spatial patch mean [B,D].
    """
    s = base.aligned_ordinary_state(run["states"][block], True).float()
    return s[:, 1:, :].mean(dim=1)


# =============================================================================
# Surface evaluation
# =============================================================================

@dataclass
class SurfaceAccumulator:
    state_sum: np.ndarray
    embedding_sum: np.ndarray
    count: int


def grid_values(points: int, bound: float) -> np.ndarray:
    return np.linspace(-bound, bound, points, dtype=np.float32)


def grid_alpha_pairs(points: int, bound: float) -> tuple[np.ndarray, np.ndarray]:
    vals = grid_values(points, bound)
    rows = []
    for a in vals:
        for b in vals:
            rows.append((float(a), float(b)))
    return vals, np.asarray(rows, dtype=np.float32)


def evaluate_surface_visual(
    model: torch.nn.Module,
    pair: base.B13Pair,
    components4: Sequence[torch.Tensor],
    plane: tuple[int, int],
    grid_pairs: np.ndarray,
    query_specs: list[dict[str, Any]],
    capture: set[int],
    batch_size: int,
    anchor: dict[str, Any],
    common_visual_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """
    One visual surface. Query specs may contain many language/query pairs;
    visual forward is reused for all of them.

    Returns raw scalar rows, [N,Dstate] B21 state delta, [N,Demb] embedding delta.
    """
    i0, j0 = plane[0] - 1, plane[1] - 1
    anchor_state = state_vector_from_run(anchor)
    anchor_emb = anchor["embedding"].float()

    scalar_rows: list[dict[str, Any]] = []
    state_chunks = []
    emb_chunks = []

    for st in range(0, len(grid_pairs), batch_size):
        gp = grid_pairs[st:st + batch_size]
        alpha4 = np.zeros((len(gp), 4), dtype=np.float32)
        alpha4[:, i0] = gp[:, 0]
        alpha4[:, j0] = gp[:, 1]

        post = make_alpha_post_batch(pair, components4, alpha4)
        run = base.run_downstream(model, post, pair.early_states, capture, has_rn=True)
        sv = state_vector_from_run(run) - anchor_state
        ev = run["embedding"].float() - anchor_emb
        state_chunks.append(sv.cpu().numpy().astype(np.float32))
        emb_chunks.append(ev.cpu().numpy().astype(np.float32))

        for spec in query_specs:
            obs = base.score_run_batch(
                model,
                run,
                spec["queries"],
                spec["attack_sem_en"],
                spec["object_sem_en"],
            )
            for bi, o in enumerate(obs):
                row = dict(common_visual_meta)
                row.update({
                    "language": spec["language"],
                    "query_mode": spec["query_mode"],
                    "query_candidate": spec["query_candidate"],
                    "attack_word_en": spec["attack_word_en"],
                    "object_label_en": spec["object_label_en"],
                    "pc_a": plane[0],
                    "pc_b": plane[1],
                    "alpha_a": float(gp[bi, 0]),
                    "alpha_b": float(gp[bi, 1]),
                    "grid_index": st + bi,
                })
                row.update(o)
                scalar_rows.append(row)

        del run, post
        base.cleanup_cuda()

    return (
        scalar_rows,
        np.concatenate(state_chunks, axis=0),
        np.concatenate(emb_chunks, axis=0),
    )


# =============================================================================
# Local geometry of mean 2-D state surfaces
# =============================================================================

def finite_diff_first(arr: np.ndarray, h: float, axis: int) -> np.ndarray:
    return np.gradient(arr, h, axis=axis, edge_order=2)


def finite_diff_second(arr: np.ndarray, h: float, axis: int) -> np.ndarray:
    first = np.gradient(arr, h, axis=axis, edge_order=2)
    return np.gradient(first, h, axis=axis, edge_order=2)


def surface_local_geometry(
    state_grid: np.ndarray,   # [G,G,D]
    scalar_grid: np.ndarray,  # [G,G]
    h: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    da = finite_diff_first(state_grid, h, axis=0)
    db = finite_diff_first(state_grid, h, axis=1)
    d2aa = finite_diff_second(state_grid, h, axis=0)
    d2bb = finite_diff_second(state_grid, h, axis=1)
    d2ab = finite_diff_first(da, h, axis=1)

    f_a = finite_diff_first(scalar_grid, h, axis=0)
    f_b = finite_diff_first(scalar_grid, h, axis=1)
    f_aa = finite_diff_second(scalar_grid, h, axis=0)
    f_bb = finite_diff_second(scalar_grid, h, axis=1)
    f_ab = finite_diff_first(f_a, h, axis=1)

    G = state_grid.shape[0]
    rows = []
    tangent_q = np.zeros((G, G, state_grid.shape[-1], 2), dtype=np.float32)

    for i in range(G):
        for j in range(G):
            J = np.stack([da[i, j], db[i, j]], axis=0)  # [2,D]
            s = np.linalg.svd(J, compute_uv=False)
            q, _ = np.linalg.qr(J.T)
            tangent_q[i, j, :, :2] = q[:, :2].astype(np.float32)

            gram = J @ J.T
            area = math.sqrt(max(float(np.linalg.det(gram)), 0.0))
            hess = np.array([[f_aa[i, j], f_ab[i, j]],
                             [f_ab[i, j], f_bb[i, j]]], dtype=np.float64)
            he = np.linalg.eigvalsh(hess)

            # Extrinsic bending proxy: remove tangent component of second derivatives.
            Q = q[:, :2]
            curv_terms = []
            for vec in (d2aa[i, j], d2ab[i, j], d2bb[i, j]):
                normal = vec - Q @ (Q.T @ vec)
                curv_terms.append(float(np.linalg.norm(normal)))

            rows.append({
                "grid_i": i,
                "grid_j": j,
                "tangent_sigma1": float(s[0]) if len(s) > 0 else 0.0,
                "tangent_sigma2": float(s[1]) if len(s) > 1 else 0.0,
                "tangent_condition": float(s[0] / max(s[1], 1e-12)) if len(s) > 1 else float("inf"),
                "local_area_element": area,
                "scalar_grad_norm": float(math.sqrt(f_a[i, j] ** 2 + f_b[i, j] ** 2)),
                "scalar_hessian_eig_min": float(he[0]),
                "scalar_hessian_eig_max": float(he[-1]),
                "state_normal_curvature_proxy": float(np.mean(curv_terms)),
            })

    return rows, {
        "da": da,
        "db": db,
        "tangent_q": tangent_q,
    }


def pca3_surface(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xc = np.asarray(x, np.float64) - np.asarray(x, np.float64).mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    coords = u[:, :3] * s[:3]
    frac = (s[:8] ** 2) / max(float(np.sum(s ** 2)), 1e-12)
    return coords.astype(np.float32), frac.astype(np.float64)


def try_isomap3(x: np.ndarray, neighbors: int = 10) -> Optional[np.ndarray]:
    try:
        from sklearn.manifold import Isomap
        return Isomap(n_neighbors=neighbors, n_components=3).fit_transform(np.asarray(x, np.float64)).astype(np.float32)
    except Exception as exc:
        print(f"[isomap] skipped: {exc}")
        return None


def write_ply_surface(
    path: Path,
    xyz: np.ndarray,
    grid_points: int,
    alpha_pairs: np.ndarray,
    scalar: np.ndarray,
) -> None:
    """
    ASCII PLY triangle mesh with extra scalar/alpha properties + vertex colors.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, np.float64)
    scalar = np.asarray(scalar, np.float64)
    finite = scalar[np.isfinite(scalar)]
    if len(finite):
        lo, hi = np.quantile(finite, [0.02, 0.98])
        if hi <= lo:
            hi = lo + 1.0
        t = np.clip((scalar - lo) / (hi - lo), 0.0, 1.0)
    else:
        t = np.zeros(len(scalar))
    cmap = plt.get_cmap("viridis")
    rgb = (cmap(t)[:, :3] * 255).astype(np.uint8)

    faces = []
    G = grid_points
    for i in range(G - 1):
        for j in range(G - 1):
            a = i * G + j
            b = (i + 1) * G + j
            c = (i + 1) * G + (j + 1)
            d = i * G + (j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        for prop in ("x", "y", "z", "alpha_a", "alpha_b", "relative_read"):
            f.write(f"property float {prop}\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for i in range(len(xyz)):
            f.write(
                f"{xyz[i,0]:.8g} {xyz[i,1]:.8g} {xyz[i,2]:.8g} "
                f"{alpha_pairs[i,0]:.8g} {alpha_pairs[i,1]:.8g} "
                f"{scalar[i]:.8g} {int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n"
            )
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


# =============================================================================
# Cross-language geometry
# =============================================================================

def tangent_fields_from_surface(x_grid: np.ndarray, h: float) -> np.ndarray:
    da = finite_diff_first(x_grid, h, axis=0)
    db = finite_diff_first(x_grid, h, axis=1)
    G, _, D = x_grid.shape
    out = np.zeros((G, G, D, 2), dtype=np.float32)
    for i in range(G):
        for j in range(G):
            J = np.stack([da[i,j], db[i,j]], axis=1)  # [D,2]
            q, _ = np.linalg.qr(J)
            out[i,j,:,:2] = q[:,:2]
    return out


def mean_tangent_principal_cosines(q1: np.ndarray, q2: np.ndarray) -> tuple[float, float]:
    vals1, vals2 = [], []
    for i in range(q1.shape[0]):
        for j in range(q1.shape[1]):
            s = np.linalg.svd(q1[i,j].T @ q2[i,j], compute_uv=False)
            vals1.append(float(s[0]))
            vals2.append(float(s[1]) if len(s) > 1 else float("nan"))
    return float(np.nanmean(vals1)), float(np.nanmean(vals2))


def knn_overlap(x: np.ndarray, y: np.ndarray, k: int = 8) -> float:
    dx = pairwise_distances(x)
    dy = pairwise_distances(y)
    vals = []
    for i in range(len(x)):
        nx = set(np.argsort(dx[i])[1:k+1].tolist())
        ny = set(np.argsort(dy[i])[1:k+1].tolist())
        vals.append(len(nx & ny) / max(len(nx | ny), 1))
    return float(np.mean(vals))


def procrustes_residual_lowd(x: np.ndarray, y: np.ndarray, dim: int = 32) -> float:
    X = np.asarray(x, np.float64)
    Y = np.asarray(y, np.float64)
    joint = np.concatenate([X, Y], axis=0)
    jc = joint - joint.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(jc, full_matrices=False)
    q = vh[: min(dim, vh.shape[0])].T
    X = (X - X.mean(axis=0, keepdims=True)) @ q
    Y = (Y - Y.mean(axis=0, keepdims=True)) @ q
    X /= max(np.linalg.norm(X), 1e-12)
    Y /= max(np.linalg.norm(Y), 1e-12)
    u, _, vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    R = u @ vt
    return float(np.linalg.norm(X @ R - Y))


# =============================================================================
# 4-D ridge search
# =============================================================================

def sobol_points_4d(n: int, bound: float, radius: float, seed: int) -> np.ndarray:
    eng = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)
    out = []
    need = n
    while need > 0:
        draw = eng.draw(max(need * 2, 128)).cpu().numpy()
        x = (draw * 2.0 - 1.0) * bound
        keep = np.linalg.norm(x, axis=1) <= radius
        x = x[keep]
        if len(x):
            take = x[:need]
            out.append(take)
            need -= len(take)
    pts = np.concatenate(out, axis=0).astype(np.float32)
    # Guarantee anchor exists.
    pts[0] = 0.0
    return pts


def clip_alpha_points(x: np.ndarray, bound: float, radius: float) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float32), -bound, bound)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    scale = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return x * scale


def evaluate_alpha_cloud(
    model: torch.nn.Module,
    pair: base.B13Pair,
    components4: Sequence[torch.Tensor],
    alpha_points: np.ndarray,
    query_specs: list[dict[str, Any]],
    capture: set[int],
    batch_size: int,
    common_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = []
    states = []
    for st in range(0, len(alpha_points), batch_size):
        ap = alpha_points[st:st+batch_size]
        post = make_alpha_post_batch(pair, components4, ap)
        run = base.run_downstream(model, post, pair.early_states, capture, has_rn=True)
        states.append(state_vector_from_run(run).cpu().numpy().astype(np.float32))

        for spec in query_specs:
            obs = base.score_run_batch(
                model, run, spec["queries"], spec["attack_sem_en"], spec["object_sem_en"]
            )
            for bi, o in enumerate(obs):
                r = dict(common_meta)
                r.update({
                    "language": spec["language"],
                    "query_mode": spec["query_mode"],
                    "query_candidate": spec["query_candidate"],
                    "point_index": st + bi,
                    "alpha1": float(ap[bi,0]),
                    "alpha2": float(ap[bi,1]),
                    "alpha3": float(ap[bi,2]),
                    "alpha4": float(ap[bi,3]),
                    "alpha_norm": float(np.linalg.norm(ap[bi])),
                })
                r.update(o)
                rows.append(r)

        del run, post
        base.cleanup_cuda()
    return rows, np.concatenate(states, axis=0)


def refine_points_around_extrema(
    raw_rows: list[dict[str, Any]],
    query_modes: list[str],
    scale: float,
    random_count: int,
    bound: float,
    radius: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = []
    for qm in query_modes:
        q = [r for r in raw_rows if r.get("query_mode") == qm]
        if not q:
            continue
        centers.append(max(q, key=lambda r: safe_float(r.get("relative_calibrated"))))
        centers.append(min(q, key=lambda r: safe_float(r.get("relative_calibrated"))))

    pts = []
    eye = np.eye(4, dtype=np.float32)
    for c in centers:
        center = np.array([c[f"alpha{i}"] for i in range(1,5)], dtype=np.float32)
        pts.append(center)
        for j in range(4):
            pts.append(center + scale * eye[j])
            pts.append(center - scale * eye[j])
        for _ in range(random_count):
            d = rng.normal(size=4).astype(np.float32)
            d /= max(np.linalg.norm(d), 1e-12)
            pts.append(center + scale * d)
    if not pts:
        return np.empty((0,4), np.float32)
    pts = clip_alpha_points(np.stack(pts), bound, radius)
    # stable unique-ish rounding
    rounded = np.round(pts, 5)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return pts[np.sort(idx)]


def ridge_extrema_and_loci(
    rows: list[dict[str, Any]],
    topk: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        key = (
            r["sample_key"], r["condition"], r["language"],
            r["query_mode"], r["basis_kind"],
        )
        groups.setdefault(key, []).append(r)

    extrema = []
    locus_points: dict[tuple, list[np.ndarray]] = {}

    for key, rs in groups.items():
        vals = np.array([safe_float(r.get("relative_calibrated")) for r in rs])
        order = np.argsort(vals)
        for mode, sel in (
            ("min", order[: min(topk, len(order))]),
            ("max", order[-min(topk, len(order)):]),
        ):
            chosen = [rs[int(i)] for i in sel]
            best = min(chosen, key=lambda r: safe_float(r["relative_calibrated"])) if mode == "min" \
                else max(chosen, key=lambda r: safe_float(r["relative_calibrated"]))
            row = {
                "sample_key": key[0],
                "condition": key[1],
                "language": key[2],
                "query_mode": key[3],
                "basis_kind": key[4],
                "extremum": mode,
                "relative_calibrated": safe_float(best["relative_calibrated"]),
            }
            for i in range(1,5):
                row[f"alpha{i}"] = safe_float(best[f"alpha{i}"])
            row["alpha_norm"] = safe_float(best["alpha_norm"])
            extrema.append(row)

            lk = (key[1], key[2], key[3], key[4], mode)
            locus_points.setdefault(lk, []).extend([
                np.array([r[f"alpha{i}"] for i in range(1,5)], dtype=np.float64)
                for r in chosen
            ])

    locus = []
    for key, pts in sorted(locus_points.items(), key=lambda kv: str(kv[0])):
        X = np.stack(pts)
        Xc = X - X.mean(axis=0, keepdims=True)
        _, s, vh = np.linalg.svd(Xc, full_matrices=False)
        frac = (s*s) / max(float(np.sum(s*s)), 1e-12)
        row = {
            "condition": key[0],
            "language": key[1],
            "query_mode": key[2],
            "basis_kind": key[3],
            "extremum": key[4],
            "n_points": len(X),
            "mean_radius": float(np.linalg.norm(X, axis=1).mean()),
        }
        for i in range(4):
            row[f"center_alpha{i+1}"] = float(X.mean(axis=0)[i])
            row[f"pca_fraction_{i+1}"] = float(frac[i]) if i < len(frac) else 0.0
            for j in range(4):
                row[f"pc{i+1}_axis{j+1}"] = float(vh[i,j]) if i < vh.shape[0] else 0.0
        locus.append(row)

    return extrema, locus


# =============================================================================
# 4-D downstream state Jacobian
# =============================================================================

def state_jacobian_at(
    model: torch.nn.Module,
    pair: base.B13Pair,
    components4: Sequence[torch.Tensor],
    center: np.ndarray,
    eps: float,
    capture: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    for j in range(4):
        m = center.copy(); m[j] -= eps
        p = center.copy(); p[j] += eps
        pts.extend([m, p])
    pts = np.asarray(pts, np.float32)

    post = make_alpha_post_batch(pair, components4, pts)
    run = base.run_downstream(model, post, pair.early_states, capture, has_rn=True)
    state = state_vector_from_run(run).cpu().numpy()  # [8,D]
    J = []
    for j in range(4):
        J.append((state[2*j+1] - state[2*j]) / (2.0 * eps))
    J = np.stack(J, axis=0)  # [4,D]
    s = np.linalg.svd(J, compute_uv=False)
    return J, s


# =============================================================================
# CLIP-ESE forced <text> vocabulary search
# =============================================================================

@dataclass
class ForcedVocabCache:
    words: list[str]
    q: np.memmap
    text: np.memmap
    cache_dir: Path


def resolve_vocab_path(path: Path) -> Path:
    # The documented default lives beside this script.  reproduce.py launches
    # probes from the repository root, so a bare relative Path must not be
    # interpreted only relative to cwd.  Explicit absolute/custom paths still
    # win unchanged.
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_absolute():
        beside_script = Path(__file__).resolve().parent / path
        if beside_script.is_file():
            return beside_script
    raise FileNotFoundError(
        f"Missing {path}. Put vocab_deduped.txt beside the script or pass --vocab."
    )


def load_vocab(path: Path) -> list[str]:
    path = resolve_vocab_path(path)
    words, seen = [], set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        w = raw.strip()
        if w and w not in seen:
            seen.add(w); words.append(w)
    if not words:
        raise RuntimeError(f"Empty vocab: {path}")
    return words


def encode_forced_candidates(
    model: torch.nn.Module,
    clip_mod: Any,
    texts: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = [f"<text> {x}" for x in texts]
    try:
        toks = clip_mod.tokenize(payload, truncate=True).to(device)
    except TypeError:
        toks = clip_mod.tokenize(payload).to(device)
    with torch.no_grad(), base.model_autocast_context(model):
        prep = model.prepare_mode_tokens(toks)
        info = model._encode_text_hidden(prep["read_tokens"])
    return info["eot_hidden_pre_ln"].float(), info["text_embedding"].float()


def get_forced_vocab_cache(
    model: torch.nn.Module,
    clip_mod: Any,
    vocab_path: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    checkpoint: str,
    vocab_limit: int = 0,
) -> ForcedVocabCache:
    vocab_path = resolve_vocab_path(vocab_path)
    words = load_vocab(vocab_path)
    if vocab_limit > 0:
        words = words[:vocab_limit]
    cache_dir.mkdir(parents=True, exist_ok=True)
    q_path = cache_dir / "forced_text_q_f16.npy"
    t_path = cache_dir / "forced_text_embedding_f16.npy"
    words_path = cache_dir / "vocab_entries.txt"
    meta_path = cache_dir / "meta.json"

    wanted = {
        "vocab_sha256": sha256_file(vocab_path),
        "n_words": len(words),
        "checkpoint": str(checkpoint),
    }

    if q_path.exists() and t_path.exists() and words_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cached_words = words_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if all(meta.get(k) == v for k, v in wanted.items()) and cached_words == words:
            q = np.load(q_path, mmap_mode="r")
            t = np.load(t_path, mmap_mode="r")
            print(f"[clipese cache] reuse {len(words):,} candidates Q={q.shape} text={t.shape}")
            return ForcedVocabCache(words, q, t, cache_dir)

    print(f"[clipese cache] embedding {len(words):,} forced-<text> candidates")
    q0, t0 = encode_forced_candidates(model, clip_mod, words[:1], device)
    qmm = np.lib.format.open_memmap(q_path, mode="w+", dtype=np.float16,
                                   shape=(len(words), q0.shape[1]))
    tmm = np.lib.format.open_memmap(t_path, mode="w+", dtype=np.float16,
                                   shape=(len(words), t0.shape[1]))
    for st in range(0, len(words), batch_size):
        batch = words[st:st+batch_size]
        q, t = encode_forced_candidates(model, clip_mod, batch, device)
        qmm[st:st+len(batch)] = q.cpu().numpy().astype(np.float16)
        tmm[st:st+len(batch)] = t.cpu().numpy().astype(np.float16)
        qmm.flush(); tmm.flush()
        print(f"[clipese cache] {st+len(batch):,}/{len(words):,}")
    words_path.write_text("\n".join(words), encoding="utf-8")
    meta = {
        **wanted,
        "q_dim": int(q0.shape[1]),
        "text_dim": int(t0.shape[1]),
        "note": "pre-LN EOT hidden query + text embedding for '<text> VOCAB_ENTRY'",
    }
    save_json(meta_path, meta)
    return ForcedVocabCache(words, np.load(q_path, mmap_mode="r"),
                            np.load(t_path, mmap_mode="r"), cache_dir)


def forced_read_relative_scores(
    model: torch.nn.Module,
    run: dict[str, Any],
    q_np: np.ndarray,
    text_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Score arbitrary forced-<text> candidates on one visual state.
    Returns calibrated candidate - internal-null score for each candidate.
    """
    implant = model.read_implant
    states = run["states"]
    reg = run["register_mask"]

    # Null baseline once.
    with torch.no_grad(), base.model_autocast_context(model):
        ntok = model._null_read_tokens(device)
        ninfo = model._encode_text_hidden(ntok)
        nfeat = implant.read_features(states, ninfo["eot_hidden_pre_ln"], register_mask=reg)
        ntext = normalize(ninfo["text_embedding"])
        scale = model.logit_scale.float().exp()
        nraw = scale * torch.einsum("bnd,nd->bn", normalize(nfeat), ntext)
        source_logits = implant.presence_logits(states)
        ncal = implant.calibrate_read_logits(
            nraw, source_logits,
            null_mask=torch.ones(1, device=device, dtype=torch.bool),
        )[0,0].float()

        q = torch.from_numpy(np.asarray(q_np, np.float32)).to(device)
        t = torch.from_numpy(np.asarray(text_np, np.float32)).to(device)
        feat = implant.read_features(states, q, register_mask=reg)
        raw = scale * torch.einsum("bnd,nd->bn", normalize(feat), normalize(t))
        cal = implant.calibrate_read_logits(
            raw, source_logits,
            null_mask=torch.zeros(q.shape[0], device=device, dtype=torch.bool),
        )[0].float()
        rel = cal - ncal
    return rel.cpu().numpy().astype(np.float32)


def forced_vocab_topk(
    model: torch.nn.Module,
    run: dict[str, Any],
    cache: ForcedVocabCache,
    device: torch.device,
    score_batch: int,
    topk: int,
) -> list[tuple[str, float]]:
    best: list[tuple[float, int]] = []
    for st in range(0, len(cache.words), score_batch):
        en = min(st + score_batch, len(cache.words))
        scores = forced_read_relative_scores(
            model, run, cache.q[st:en], cache.text[st:en], device
        )
        k = min(topk, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        for i in idx:
            best.append((float(scores[i]), st + int(i)))
        best = sorted(best, reverse=True)[:topk]
    return [(cache.words[i], s) for s, i in best]


def vocab_cosine_topk(
    run: dict[str, Any],
    cache: ForcedVocabCache,
    device: torch.device,
    score_batch: int,
    topk: int,
) -> list[tuple[str, float]]:
    """
    Ordinary normalized image-embedding cosine against the SAME <text>-prepared
    text embeddings cached for the forced-reader search.
    """
    image = normalize(run["embedding"])[0].float().to(device)
    best: list[tuple[float, int]] = []
    for st in range(0, len(cache.words), score_batch):
        en = min(st + score_batch, len(cache.words))
        t = torch.from_numpy(np.asarray(cache.text[st:en], np.float32)).to(device)
        sims = (normalize(t) @ image).detach().cpu().numpy()
        k = min(topk, len(sims))
        idx = np.argpartition(sims, -k)[-k:]
        for i in idx:
            best.append((float(sims[i]), st + int(i)))
        best = sorted(best, reverse=True)[:topk]
    return [(cache.words[i], s) for s, i in best]


def score_phrase_cosines(
    model: torch.nn.Module,
    clip_mod: Any,
    run: dict[str, Any],
    phrases: list[str],
    device: torch.device,
    batch: int,
) -> np.ndarray:
    vals = []
    image = normalize(run["embedding"])[0].float().to(device)
    for st in range(0, len(phrases), batch):
        p = phrases[st:st+batch]
        _q, t = encode_forced_candidates(model, clip_mod, p, device)
        vals.append((normalize(t) @ image).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vals) if vals else np.empty(0, np.float32)


def clipese_phrase_beam_cosine(
    model: torch.nn.Module,
    clip_mod: Any,
    run: dict[str, Any],
    top_words: list[str],
    top_single_scores: list[float],
    device: torch.device,
    pair_beam: int,
    text_batch: int,
) -> dict[str, Any]:
    words = list(dict.fromkeys(top_words))[:10]
    best_phrase = words[0] if words else ""
    best_score = float(top_single_scores[0]) if top_single_scores else -float("inf")
    best_len = 1

    pair_phrases = []
    pair_meta = []
    for a, b in permutations(words, 2):
        pair_phrases.append(f"{a} {b}"); pair_meta.append((a,b,"space"))
        pair_phrases.append(f"{a}{b}"); pair_meta.append((a,b,"concat"))

    pair_scores = score_phrase_cosines(
        model, clip_mod, run, pair_phrases, device, text_batch
    ) if pair_phrases else np.empty(0, np.float32)
    if len(pair_scores):
        pi = int(np.argmax(pair_scores))
        if float(pair_scores[pi]) > best_score:
            best_score = float(pair_scores[pi]); best_phrase = pair_phrases[pi]; best_len = 2

    triple_phrases = []
    if len(pair_scores):
        order = np.argsort(-pair_scores)[: min(pair_beam, len(pair_scores))]
        for pi in order:
            a, b, _ = pair_meta[int(pi)]
            for c in words:
                if c in {a,b}:
                    continue
                triple_phrases.append(f"{a} {b} {c}")
                triple_phrases.append(f"{a}{b}{c}")
                triple_phrases.append(f"{a} {b}{c}")
    triple_phrases = list(dict.fromkeys(triple_phrases))
    triple_scores = score_phrase_cosines(
        model, clip_mod, run, triple_phrases, device, text_batch
    ) if triple_phrases else np.empty(0, np.float32)
    if len(triple_scores):
        ti = int(np.argmax(triple_scores))
        if float(triple_scores[ti]) > best_score:
            best_score = float(triple_scores[ti]); best_phrase = triple_phrases[ti]; best_len = 3

    return {
        "best_phrase": best_phrase,
        "best_score": best_score,
        "best_length": best_len,
        "gain_over_best_single": best_score - (float(top_single_scores[0]) if top_single_scores else float("nan")),
    }


def score_phrase_list(
    model: torch.nn.Module,
    clip_mod: Any,
    run: dict[str, Any],
    phrases: list[str],
    device: torch.device,
    batch: int,
) -> np.ndarray:
    vals = []
    for st in range(0, len(phrases), batch):
        p = phrases[st:st+batch]
        q, t = encode_forced_candidates(model, clip_mod, p, device)
        vals.append(forced_read_relative_scores(
            model, run, q.cpu().numpy(), t.cpu().numpy(), device
        ))
    return np.concatenate(vals) if vals else np.empty(0, np.float32)


def clipese_phrase_beam(
    model: torch.nn.Module,
    clip_mod: Any,
    run: dict[str, Any],
    top_words: list[str],
    top_single_scores: list[float],
    device: torch.device,
    pair_beam: int,
    text_batch: int,
) -> dict[str, Any]:
    words = list(dict.fromkeys(top_words))[:10]
    best_phrase = words[0] if words else ""
    best_score = float(top_single_scores[0]) if top_single_scores else -float("inf")
    best_len = 1

    pair_phrases = []
    pair_meta = []
    for a, b in permutations(words, 2):
        pair_phrases.append(f"{a} {b}"); pair_meta.append((a,b,"space"))
        pair_phrases.append(f"{a}{b}"); pair_meta.append((a,b,"concat"))

    pair_scores = score_phrase_list(
        model, clip_mod, run, pair_phrases, device, text_batch
    ) if pair_phrases else np.empty(0, np.float32)

    if len(pair_scores):
        pi = int(np.argmax(pair_scores))
        if float(pair_scores[pi]) > best_score:
            best_score = float(pair_scores[pi]); best_phrase = pair_phrases[pi]; best_len = 2

    triple_phrases = []
    if len(pair_scores):
        order = np.argsort(-pair_scores)[: min(pair_beam, len(pair_scores))]
        for pi in order:
            a, b, _ = pair_meta[int(pi)]
            for c in words:
                if c in {a,b}:
                    continue
                triple_phrases.append(f"{a} {b} {c}")
                triple_phrases.append(f"{a}{b}{c}")
                triple_phrases.append(f"{a} {b}{c}")
    triple_phrases = list(dict.fromkeys(triple_phrases))
    triple_scores = score_phrase_list(
        model, clip_mod, run, triple_phrases, device, text_batch
    ) if triple_phrases else np.empty(0, np.float32)

    if len(triple_scores):
        ti = int(np.argmax(triple_scores))
        if float(triple_scores[ti]) > best_score:
            best_score = float(triple_scores[ti]); best_phrase = triple_phrases[ti]; best_len = 3

    return {
        "best_phrase": best_phrase,
        "best_score": best_score,
        "best_length": best_len,
        "gain_over_best_single": best_score - (float(top_single_scores[0]) if top_single_scores else float("nan")),
    }


# =============================================================================
# Representative semantic landmarks
# =============================================================================

def choose_representative_keys(surface_rows: list[dict[str, Any]], languages: list[str]) -> dict[str, str]:
    """
    Median native-query anchor READ on RN PC1xPC2 SynthRTA.
    """
    out = {}
    for lang in languages:
        q = [
            r for r in surface_rows
            if r["condition"] == "synth"
            and r["language"] == lang
            and r["query_mode"] == "native"
            and r["basis_kind"] == "rn"
            and int(r["pc_a"]) == 1 and int(r["pc_b"]) == 2
            and abs(float(r["alpha_a"])) < 1e-7
            and abs(float(r["alpha_b"])) < 1e-7
        ]
        if not q:
            continue
        vals = np.asarray([safe_float(r["relative_calibrated"]) for r in q])
        med = float(np.nanmedian(vals))
        best = min(q, key=lambda r: abs(safe_float(r["relative_calibrated"]) - med))
        out[lang] = str(best["sample_key"])
    return out


# =============================================================================
# Plotting
# =============================================================================

def plot_scalar_surface(
    rows: list[dict[str, Any]],
    lang: str,
    query_mode: str,
    basis_kind: str,
    plane: tuple[int,int],
    grid_points: int,
    out: Path,
) -> None:
    q = [
        r for r in rows
        if r["condition"] == "synth"
        and r["language"] == lang
        and r["query_mode"] == query_mode
        and r["basis_kind"] == basis_kind
        and int(r["pc_a"]) == plane[0]
        and int(r["pc_b"]) == plane[1]
    ]
    if not q:
        return
    # rows are already aggregate summary here
    q = sorted(q, key=lambda r: (float(r["alpha_a"]), float(r["alpha_b"])))
    z = np.asarray([safe_float(r["relative_calibrated_mean"]) for r in q]).reshape(grid_points, grid_points)
    a = np.asarray(sorted({float(r["alpha_a"]) for r in q}))
    b = np.asarray(sorted({float(r["alpha_b"]) for r in q}))
    A, B = np.meshgrid(b, a)  # plotting x=PCb, y=PCa

    fig = plt.figure(figsize=(7.2, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(B, A, z, linewidth=0, antialiased=True)
    ax.set_xlabel(f"PC{plane[1]} α")
    ax.set_ylabel(f"PC{plane[0]} α")
    ax.set_zlabel("relative READ")
    ax.set_title(f"{lang} · {query_mode} · {basis_kind} · PC{plane[0]}×PC{plane[1]}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cross_language_matrix(rows: list[dict[str, Any]], measure: str, title: str, out: Path) -> None:
    if not rows:
        return
    langs = sorted({r["language_a"] for r in rows} | {r["language_b"] for r in rows})
    mat = np.eye(len(langs), dtype=float)
    for r in rows:
        i, j = langs.index(r["language_a"]), langs.index(r["language_b"])
        mat[i,j] = mat[j,i] = safe_float(r.get(measure))
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(mat, vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    ax.set_xticks(range(len(langs))); ax.set_xticklabels(langs)
    ax.set_yticks(range(len(langs))); ax.set_yticklabels(langs)
    ax.set_title(title)
    for i in range(len(langs)):
        for j in range(len(langs)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Compact ZIP
# =============================================================================

def build_handoff_zip(output_dir: Path, include: list[Path]) -> Path:
    zpath = output_dir / "compact_summary_rn_control_manifold.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in include:
            if p.exists() and p.is_file():
                z.write(p, arcname=p.relative_to(output_dir).as_posix())
    return zpath


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="RN local control-manifold atlas")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--module-root", default=".")
    ap.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    ap.add_argument("--dataset-root", default="")
    ap.add_argument("--languages", default=DEFAULT_LANGUAGES)
    ap.add_argument("--query-modes", default=DEFAULT_QUERY_MODES)
    ap.add_argument("--atlas-keys", type=int, default=DEFAULT_ATLAS_KEYS)
    ap.add_argument("--basis-pairs-per-language", type=int, default=DEFAULT_BASIS_PAIRS_PER_LANGUAGE)
    ap.add_argument("--planes", default=DEFAULT_PLANES)
    ap.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS)
    ap.add_argument("--grid-bound", type=float, default=DEFAULT_GRID_BOUND)
    ap.add_argument("--surface-batch", type=int, default=DEFAULT_SURFACE_BATCH)
    ap.add_argument("--include-norta-surfaces", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ridge-points", type=int, default=DEFAULT_RIDGE_POINTS)
    ap.add_argument("--ridge-bound", type=float, default=DEFAULT_RIDGE_BOUND)
    ap.add_argument("--ridge-radius", type=float, default=DEFAULT_RIDGE_RADIUS)
    ap.add_argument("--ridge-batch", type=int, default=DEFAULT_RIDGE_BATCH)
    ap.add_argument("--ridge-topk", type=int, default=DEFAULT_RIDGE_TOPK)
    ap.add_argument("--ridge-refine-steps", type=int, default=DEFAULT_RIDGE_REFINE_STEPS)
    ap.add_argument("--ridge-refine-random", type=int, default=DEFAULT_RIDGE_REFINE_RANDOM)
    ap.add_argument("--state-jac-eps", type=float, default=DEFAULT_STATE_JAC_EPS)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--skip-vocab", action="store_true")
    ap.add_argument("--resume-after-atlas", action="store_true",
                    help="reuse an already-computed atlas stage in --output-dir and continue from CLIP-ESE/postprocess; validates structural compatibility and skips the expensive atlas sweep")
    ap.add_argument("--vocab-limit", type=int, default=0,
                    help="debug/smoke helper: use only the first N deduplicated vocab entries; 0 = full vocabulary")
    ap.add_argument("--vocab-topk", type=int, default=DEFAULT_VOCAB_TOPK)
    ap.add_argument("--vocab-build-batch", type=int, default=DEFAULT_VOCAB_BATCH)
    ap.add_argument("--vocab-score-batch", type=int, default=DEFAULT_VOCAB_SCORE_BATCH)
    ap.add_argument("--pair-beam", type=int, default=DEFAULT_PAIR_BEAM)
    ap.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="rn_control_manifold_atlas_its_a_full_double_encoder_all_the_way")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    languages = parse_strs(args.languages)
    query_modes = parse_strs(args.query_modes)
    planes = parse_planes(args.planes)
    if max(max(p) for p in planes) > 4:
        raise ValueError("This atlas currently maps PC1..PC4")
    for q in query_modes:
        if q not in {"english", "native"}:
            raise ValueError(q)

    out = Path(args.output_dir)
    data_dir = out / "data"
    plot_dir = out / "plots"
    ply_dir = out / "ply"
    cache_dir = out / "cache"
    for d in (data_dir, plot_dir, ply_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Dataset + model
    # -------------------------------------------------------------------------
    norta, attacks, norta_idx, attack_idx, shared_keys = base.prepare_datasets(
        args.dataset_repo,
        args.dataset_root,
        languages,
        0,  # load full intersection; atlas sampling below
        args.seed,
    )
    if args.atlas_keys > 0 and len(shared_keys) > args.atlas_keys:
        rng = random.Random(args.seed)
        atlas_keys = sorted(rng.sample(shared_keys, args.atlas_keys))
    else:
        atlas_keys = list(shared_keys)

    loaded = base.load_model(args.checkpoint, package_root=args.module_root, device=args.device)
    model, preprocess, clip_mod, device = (
        loaded.model, loaded.preprocess, loaded.clip_module, loaded.device
    )
    model.eval()

    target_res = int(model.visual.input_resolution)
    patch = int(model.visual.conv1.kernel_size[0])
    insert_block = int(model.visual.read_null_insert_block)
    capture_list = [int(x) for x in model.read_implant.capture_block_list()]
    early_needed = {b for b in capture_list if b < insert_block}
    surface_capture = set(capture_list) | {STATE_BLOCK}

    if args.resume_after_atlas:
        print("[resume] reusing completed atlas stage; skipping expensive manifold sweep")
        required_stage = [
            data_dir / "atlas_basis.npz",
            data_dir / "sample_manifest.csv",
            data_dir / "surface_raw.csv",
            data_dir / "surface_summary.csv",
            data_dir / "ridge_extrema.csv",
            data_dir / "ridge_locus_summary.csv",
            data_dir / "cross_language_geometry.csv",
            data_dir / "state_jacobian_summary.csv",
        ]
        missing_stage = [str(path) for path in required_stage if not path.is_file()]
        if missing_stage:
            raise FileNotFoundError(
                "--resume-after-atlas requested, but the previous atlas stage is incomplete. Missing:\n  "
                + "\n  ".join(missing_stage)
            )

        basis_npz = np.load(data_dir / "atlas_basis.npz", allow_pickle=True)
        learned_np = np.asarray(basis_npz["learned_basis"], dtype=np.float32)
        random_np = np.asarray(basis_npz["random_basis"], dtype=np.float32)
        if learned_np.ndim != 2 or random_np.ndim != 2 or learned_np.shape[0] < 4 or random_np.shape[0] < 4:
            raise RuntimeError(
                f"Saved atlas basis is incompatible: learned={learned_np.shape}, random={random_np.shape}"
            )
        learned_basis = torch.from_numpy(learned_np[:4]).to(device)
        random_basis = torch.from_numpy(random_np[:4]).to(device)
        singular_np = np.asarray(basis_npz["singular_values"], dtype=np.float64)
        explained_np = np.asarray(basis_npz["explained_energy"], dtype=np.float64)
        basis_info = {
            "singular_values": torch.from_numpy(singular_np),
            "explained_energy": explained_np,
        }

        manifest_rows = load_rows(data_dir / "sample_manifest.csv")
        manifest_keys = {str(row["sample_key"]) for row in manifest_rows}
        manifest_langs = {str(row["language"]) for row in manifest_rows}
        if manifest_keys != {str(k) for k in atlas_keys}:
            raise RuntimeError(
                "Saved atlas sample keys do not match the current --atlas-keys/dataset selection; "
                "refusing to reuse stale manifold output."
            )
        if manifest_langs != set(languages):
            raise RuntimeError(
                f"Saved atlas languages {sorted(manifest_langs)} do not match current languages {sorted(languages)}"
            )

        surface_raw = load_rows(data_dir / "surface_raw.csv")
        surface_summary = load_rows(data_dir / "surface_summary.csv")
        ridge_extrema = load_rows(data_dir / "ridge_extrema.csv")
        ridge_locus = load_rows(data_dir / "ridge_locus_summary.csv")
        cross_rows = load_rows(data_dir / "cross_language_geometry.csv")

        expected_surface_rows = (
            len(atlas_keys) * len(languages) * 2 * len(planes) * len(query_modes)
            * (2 if args.include_norta_surfaces else 1) * (args.grid_points ** 2)
        )
        if len(surface_raw) != expected_surface_rows:
            raise RuntimeError(
                f"Saved surface_raw.csv has {len(surface_raw):,} rows, expected {expected_surface_rows:,} "
                "for the current atlas/language/plane/query/grid settings. Refusing stale resume."
            )

        stage_meta_path = data_dir / "atlas_stage_complete.json"
        if stage_meta_path.is_file():
            stage_meta = json.loads(stage_meta_path.read_text(encoding="utf-8"))
            expected_meta = {
                "languages": languages,
                "query_modes": query_modes,
                "atlas_keys": [str(k) for k in atlas_keys],
                "planes": [list(p) for p in planes],
                "grid_points": int(args.grid_points),
                "include_norta_surfaces": bool(args.include_norta_surfaces),
                "basis_pairs_per_language": int(args.basis_pairs_per_language),
            }
            for field, expected in expected_meta.items():
                if stage_meta.get(field) != expected:
                    raise RuntimeError(
                        f"Saved atlas stage metadata mismatch for {field}: "
                        f"saved={stage_meta.get(field)!r} current={expected!r}"
                    )
            print(f"[resume] verified atlas stage metadata: {stage_meta_path}")
        else:
            print(
                "[resume] WARNING: legacy partial run has no atlas_stage_complete.json; "
                "structural checks passed, but checkpoint provenance cannot be independently proven. "
                "Use this only for the interrupted run you intend to continue."
            )
    else:
        # -------------------------------------------------------------------------
        # Fit learned basis, then matched random orthogonal directions.
        # -------------------------------------------------------------------------
        print("[basis] fitting RN control basis")
        basis_info = base.fit_basis_from_dataset(
            model, preprocess, norta, attacks, norta_idx, attack_idx,
            shared_keys, languages, args.basis_pairs_per_language,
            early_needed, max_rank=4, device=device,
        )
        learned_basis = basis_info["basis_kd"][:4].to(device).float()
        random_basis = random_basis_orthogonal_to(learned_basis, 4, args.seed + 991)

        np.savez_compressed(
            data_dir / "atlas_basis.npz",
            learned_basis=learned_basis.cpu().numpy(),
            random_basis=random_basis.cpu().numpy(),
            singular_values=basis_info["singular_values"].cpu().numpy(),
            explained_energy=np.asarray(basis_info["explained_energy"]),
            fit_items=np.asarray(basis_info["fit_items"], dtype=object),
        )

        # Text cache.
        query_cache: dict[str, dict[str, Any]] = {}
        sem_cache: dict[str, torch.Tensor] = {}

        def get_query(label: str):
            if label not in query_cache:
                query_cache[label] = base.prepare_read_queries(model, clip_mod, label, device)
            return query_cache[label]

        def get_sem(label: str):
            if label not in sem_cache:
                sem_cache[label] = base.prompt_bank_embedding(
                    model, clip_mod, label,
                    ["{label}", "a photo of {label}", "the image depicts {label}",
                     "there is {label}", "a picture of {label}"],
                    device,
                )
            return sem_cache[label]

        # -------------------------------------------------------------------------
        # Dense surfaces + ridge raw data.
        # -------------------------------------------------------------------------
        grid_vals, grid_pairs = grid_alpha_pairs(args.grid_points, args.grid_bound)
        h = float(grid_vals[1] - grid_vals[0])
        surface_raw: list[dict[str, Any]] = []
        ridge_raw: list[dict[str, Any]] = []
        state_jac_raw: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []

        # Mean state accumulation keyed by condition/language/basis/plane.
        state_acc: dict[tuple, SurfaceAccumulator] = {}

        def add_state_acc(key: tuple, state: np.ndarray, emb: np.ndarray):
            if key not in state_acc:
                state_acc[key] = SurfaceAccumulator(
                    state_sum=np.zeros_like(state, dtype=np.float64),
                    embedding_sum=np.zeros_like(emb, dtype=np.float64),
                    count=0,
                )
            state_acc[key].state_sum += state
            state_acc[key].embedding_sum += emb
            state_acc[key].count += 1

        # We need ridge best points later for semantic landmarks.
        extrema_centers_by_case: dict[tuple, list[dict[str, Any]]] = {}

        progress = tqdm(atlas_keys, desc="atlas sample_key", unit="key")
        for key_index, key in enumerate(progress):
            # Gather language rows first.
            lang_rows = {lang: dict(attacks[lang][attack_idx[lang][key]]) for lang in languages}

            # ----------------------------- NoRTA pair -----------------------------
            nrow = dict(norta[norta_idx[key]])
            npil = base.ensure_pil(nrow["image"])
            nimg = base.preprocess_pil(preprocess, npil, device)
            npair = base.build_b13_pair(model, nimg, key, "norta", early_needed)
            n_rn_comp, n_rand_comp = per_pc_components(
                npair.delta_ord_btd, learned_basis, random_basis, 4
            )
            n_anchor = anchor_run(model, npair, surface_capture)

            # Query specs for ALL languages on the shared NoRTA visual run.
            n_query_specs = []
            for lang, srow in lang_rows.items():
                attack_en = str(srow["attack_word_en"])
                object_en = str(srow["object_label_en"])
                attack_native = str(srow["attack_word"])
                for qm in query_modes:
                    qlabel = attack_en if qm == "english" else attack_native
                    n_query_specs.append({
                        "language": lang,
                        "query_mode": qm,
                        "query_candidate": qlabel,
                        "queries": get_query(qlabel),
                        "attack_sem_en": get_sem(attack_en),
                        "object_sem_en": get_sem(object_en),
                        "attack_word_en": attack_en,
                        "object_label_en": object_en,
                    })

            if args.include_norta_surfaces:
                for basis_kind, comps in (("rn", n_rn_comp), ("random", n_rand_comp)):
                    for plane in planes:
                        rows, states, embs = evaluate_surface_visual(
                            model, npair, comps, plane, grid_pairs, n_query_specs,
                            surface_capture, args.surface_batch, n_anchor,
                            {
                                "sample_key": key,
                                "condition": "norta",
                                "basis_kind": basis_kind,
                            },
                        )
                        surface_raw.extend(rows)
                        # visual state itself is language independent here
                        add_state_acc(("norta", "shared", basis_kind, plane), states, embs)

            # ------------------------ per-language SynthRTA -----------------------
            for lang in languages:
                srow = lang_rows[lang]
                spil = base.ensure_pil(srow["image"])
                simg = base.preprocess_pil(preprocess, spil, device)
                spair = base.build_b13_pair(model, simg, key, "synth", early_needed)
                bbox = base.bbox_patch_mask(
                    srow["bbox"], spil.size, target_res, patch, device
                )
                srn, srand = per_pc_components(
                    spair.delta_ord_btd, learned_basis, random_basis, 4
                )
                s_anchor = anchor_run(model, spair, surface_capture)

                attack_en = str(srow["attack_word_en"])
                object_en = str(srow["object_label_en"])
                attack_native = str(srow["attack_word"])
                object_native = str(srow["object_label"])

                manifest_rows.append({
                    "sample_key": key,
                    "language": lang,
                    "attack_word_en": attack_en,
                    "object_label_en": object_en,
                    "attack_word_native": attack_native,
                    "object_label_native": object_native,
                    "bbox_patch_count": int(bbox.sum().item()),
                })

                specs = []
                for qm in query_modes:
                    qlabel = attack_en if qm == "english" else attack_native
                    specs.append({
                        "language": lang,
                        "query_mode": qm,
                        "query_candidate": qlabel,
                        "queries": get_query(qlabel),
                        "attack_sem_en": get_sem(attack_en),
                        "object_sem_en": get_sem(object_en),
                        "attack_word_en": attack_en,
                        "object_label_en": object_en,
                    })

                for basis_kind, comps in (("rn", srn), ("random", srand)):
                    # Dense 2-D charts.
                    for plane in planes:
                        rows, states, embs = evaluate_surface_visual(
                            model, spair, comps, plane, grid_pairs, specs,
                            surface_capture, args.surface_batch, s_anchor,
                            {
                                "sample_key": key,
                                "condition": "synth",
                                "basis_kind": basis_kind,
                            },
                        )
                        surface_raw.extend(rows)
                        add_state_acc(("synth", lang, basis_kind, plane), states, embs)

                    # 4-D Sobol ridge / basin search.
                    seed = args.seed + 10000 * key_index + 137 * (languages.index(lang)+1) + (0 if basis_kind=="rn" else 700000)
                    pts = sobol_points_4d(
                        args.ridge_points, args.ridge_bound, args.ridge_radius, seed
                    )
                    rr, _ = evaluate_alpha_cloud(
                        model, spair, comps, pts, specs, surface_capture,
                        args.ridge_batch,
                        {"sample_key": key, "condition": "synth", "basis_kind": basis_kind},
                    )
                    ridge_raw.extend(rr)

                    # Local refinement around current max/min for all query modes.
                    all_rr = list(rr)
                    for step in range(args.ridge_refine_steps):
                        scale = args.ridge_bound * (0.30 / (2 ** step))
                        refine = refine_points_around_extrema(
                            all_rr, query_modes, scale, args.ridge_refine_random,
                            args.ridge_bound, args.ridge_radius,
                            seed + 9000 + step,
                        )
                        if len(refine):
                            new_rr, _ = evaluate_alpha_cloud(
                                model, spair, comps, refine, specs, surface_capture,
                                args.ridge_batch,
                                {"sample_key": key, "condition": "synth", "basis_kind": basis_kind},
                            )
                            ridge_raw.extend(new_rr)
                            all_rr.extend(new_rr)

                    # State Jacobian at anchor + current max/min for each query.
                    centers = [("anchor", np.zeros(4, dtype=np.float32), "visual")]
                    for qm in query_modes:
                        qrows = [r for r in all_rr if r["query_mode"] == qm]
                        if qrows:
                            mx = max(qrows, key=lambda r: safe_float(r["relative_calibrated"]))
                            mn = min(qrows, key=lambda r: safe_float(r["relative_calibrated"]))
                            for name, r in (("max", mx), ("min", mn)):
                                c = np.array([r[f"alpha{i}"] for i in range(1,5)], dtype=np.float32)
                                centers.append((name, c, qm))

                    seen_centers = set()
                    for landmark, center, qm_label in centers:
                        ck = tuple(np.round(center, 5).tolist()) + (landmark, qm_label)
                        if ck in seen_centers:
                            continue
                        seen_centers.add(ck)
                        J, svals = state_jacobian_at(
                            model, spair, comps, center,
                            args.state_jac_eps, surface_capture,
                        )
                        row = {
                            "sample_key": key,
                            "condition": "synth",
                            "language": lang,
                            "basis_kind": basis_kind,
                            "landmark": landmark,
                            "query_mode": qm_label,
                            "effective_rank": effective_rank_from_singular_values(svals),
                        }
                        for i in range(4):
                            row[f"alpha{i+1}"] = float(center[i])
                            row[f"sigma{i+1}"] = float(svals[i]) if i < len(svals) else 0.0
                        row["sigma1_over_sigma2"] = float(svals[0] / max(svals[1],1e-12))
                        row["sigma2_over_sigma3"] = float(svals[1] / max(svals[2],1e-12))
                        row["sigma3_over_sigma4"] = float(svals[2] / max(svals[3],1e-12))
                        state_jac_raw.append(row)

                del spair, simg, spil, s_anchor
                base.cleanup_cuda()

            del npair, nimg, npil, n_anchor
            base.cleanup_cuda()

        save_rows(data_dir / "surface_raw.csv", surface_raw)
        save_rows(data_dir / "ridge_raw.csv", ridge_raw)
        save_rows(data_dir / "state_jacobian_raw.csv", state_jac_raw)
        save_rows(data_dir / "sample_manifest.csv", manifest_rows)

        # -------------------------------------------------------------------------
        # Aggregate scalar surfaces.
        # -------------------------------------------------------------------------
        surface_summary = base.aggregate_rows(
            surface_raw,
            ["condition","language","query_mode","basis_kind","pc_a","pc_b","alpha_a","alpha_b","grid_index"],
            [
                "relative_calibrated",
                "candidate_read_null_B20",
                "candidate_read_null_B21",
                "null_read_null_B21",
                "early_ortho",
                "image_attack_en_logit",
                "image_object_en_logit",
                "attack_minus_object_en",
            ],
        )
        save_rows(data_dir / "surface_summary.csv", surface_summary)

        # -------------------------------------------------------------------------
        # Save mean high-D surface states.
        # -------------------------------------------------------------------------
        state_npz = {}
        state_meta = []
        for key, acc in state_acc.items():
            mean_state = (acc.state_sum / acc.count).astype(np.float32)
            mean_emb = (acc.embedding_sum / acc.count).astype(np.float32)
            name = stable_slug(f"{key[0]}__{key[1]}__{key[2]}__pc{key[3][0]}x{key[3][1]}")
            state_npz[name + "__b21_patchmean"] = mean_state
            state_npz[name + "__final_embedding"] = mean_emb
            state_meta.append({
                "array_prefix": name,
                "condition": key[0],
                "language": key[1],
                "basis_kind": key[2],
                "pc_a": key[3][0],
                "pc_b": key[3][1],
                "count": acc.count,
            })
        np.savez_compressed(data_dir / "surface_state_means.npz", **state_npz)
        save_rows(data_dir / "surface_state_mean_manifest.csv", state_meta)

        # -------------------------------------------------------------------------
        # Local geometry + PLY + surface coordinate CSV.
        # -------------------------------------------------------------------------
        local_geom_rows = []
        surface_coord_rows = []
        pca_spectrum_rows = []
        tangent_cache: dict[tuple, np.ndarray] = {}
        mean_state_cache: dict[tuple, np.ndarray] = {}

        for key, acc in state_acc.items():
            condition, lang, basis_kind, plane = key
            mean_state = (acc.state_sum / acc.count).astype(np.float32)
            mean_state_cache[key] = mean_state
            G = args.grid_points
            state_grid = mean_state.reshape(G, G, -1)

            # Use native query scalar for synth; English for shared NoRTA PLY coloring.
            qm = "native" if condition == "synth" and "native" in query_modes else query_modes[0]
            sl = [
                r for r in surface_summary
                if r["condition"] == condition
                and r["language"] == (lang if condition=="synth" else languages[0])
                and r["query_mode"] == qm
                and r["basis_kind"] == basis_kind
                and int(r["pc_a"]) == plane[0]
                and int(r["pc_b"]) == plane[1]
            ]
            sl = sorted(sl, key=lambda r: int(r["grid_index"]))
            scalar = np.asarray([safe_float(r["relative_calibrated_mean"]) for r in sl], dtype=np.float64)
            if len(scalar) != G*G:
                # NoRTA state is shared but scalar rows are language-specific; first language is enough for coloring.
                print("[geometry] scalar mismatch, skipping", key, len(scalar))
                continue
            scalar_grid = scalar.reshape(G,G)

            geom, aux = surface_local_geometry(state_grid, scalar_grid, h)
            tangent_cache[key] = aux["tangent_q"]

            for r in geom:
                rr = {
                    "condition": condition, "language": lang, "basis_kind": basis_kind,
                    "pc_a": plane[0], "pc_b": plane[1],
                    "alpha_a": float(grid_vals[r["grid_i"]]),
                    "alpha_b": float(grid_vals[r["grid_j"]]),
                }
                rr.update(r)
                local_geom_rows.append(rr)

            coords3, frac = pca3_surface(mean_state)
            iso3 = try_isomap3(mean_state, neighbors=max(6, min(12, G-1)))

            for i in range(len(coords3)):
                surface_coord_rows.append({
                    "condition": condition, "language": lang, "basis_kind": basis_kind,
                    "pc_a": plane[0], "pc_b": plane[1],
                    "grid_index": i,
                    "alpha_a": float(grid_pairs[i,0]),
                    "alpha_b": float(grid_pairs[i,1]),
                    "pca_x": float(coords3[i,0]), "pca_y": float(coords3[i,1]), "pca_z": float(coords3[i,2]),
                    "relative_read": float(scalar[i]),
                    "isomap_x": float(iso3[i,0]) if iso3 is not None else float("nan"),
                    "isomap_y": float(iso3[i,1]) if iso3 is not None else float("nan"),
                    "isomap_z": float(iso3[i,2]) if iso3 is not None else float("nan"),
                })
            for i, f in enumerate(frac, 1):
                pca_spectrum_rows.append({
                    "condition": condition, "language": lang, "basis_kind": basis_kind,
                    "pc_a": plane[0], "pc_b": plane[1],
                    "component": i, "explained_fraction": float(f),
                })

            stem = stable_slug(f"{condition}__{lang}__{basis_kind}__PC{plane[0]}xPC{plane[1]}")
            write_ply_surface(
                ply_dir / f"{stem}__B21_PCA3.ply",
                coords3, G, grid_pairs, scalar,
            )
            if iso3 is not None:
                write_ply_surface(
                    ply_dir / f"{stem}__B21_Isomap3.ply",
                    iso3, G, grid_pairs, scalar,
                )

        save_rows(data_dir / "surface_local_geometry.csv", local_geom_rows)
        save_rows(data_dir / "surface_3d_coordinates.csv", surface_coord_rows)
        save_rows(data_dir / "surface_pca_spectrum.csv", pca_spectrum_rows)

        # -------------------------------------------------------------------------
        # Cross-language state geometry on SynthRTA.
        # -------------------------------------------------------------------------
        cross_rows = []
        behavior_rows = []

        for basis_kind in ("rn","random"):
            for plane in planes:
                for ia, la in enumerate(languages):
                    for lb in languages[ia+1:]:
                        ka = ("synth",la,basis_kind,plane)
                        kb = ("synth",lb,basis_kind,plane)
                        if ka not in mean_state_cache or kb not in mean_state_cache:
                            continue
                        A = mean_state_cache[ka]
                        B = mean_state_cache[kb]
                        dA, dB = pairwise_distances(A), pairwise_distances(B)
                        t1, t2 = mean_tangent_principal_cosines(
                            tangent_cache[ka], tangent_cache[kb]
                        )
                        cross_rows.append({
                            "basis_kind": basis_kind,
                            "pc_a": plane[0], "pc_b": plane[1],
                            "language_a": la, "language_b": lb,
                            "distance_pearson": pearson(upper_triangle_values(dA), upper_triangle_values(dB)),
                            "distance_spearman": spearman(upper_triangle_values(dA), upper_triangle_values(dB)),
                            "knn8_jaccard": knn_overlap(A,B,k=8),
                            "tangent_principal_cos1_mean": t1,
                            "tangent_principal_cos2_mean": t2,
                            "procrustes_residual_32d": procrustes_residual_lowd(A,B,dim=32),
                        })

                        for qm in query_modes:
                            qa = [
                                r for r in surface_summary
                                if r["condition"]=="synth" and r["language"]==la
                                and r["query_mode"]==qm and r["basis_kind"]==basis_kind
                                and int(r["pc_a"])==plane[0] and int(r["pc_b"])==plane[1]
                            ]
                            qb = [
                                r for r in surface_summary
                                if r["condition"]=="synth" and r["language"]==lb
                                and r["query_mode"]==qm and r["basis_kind"]==basis_kind
                                and int(r["pc_a"])==plane[0] and int(r["pc_b"])==plane[1]
                            ]
                            qa = sorted(qa,key=lambda r:int(r["grid_index"]))
                            qb = sorted(qb,key=lambda r:int(r["grid_index"]))
                            if len(qa)==len(qb)==args.grid_points**2:
                                va=np.asarray([safe_float(r["relative_calibrated_mean"]) for r in qa])
                                vb=np.asarray([safe_float(r["relative_calibrated_mean"]) for r in qb])
                                behavior_rows.append({
                                    "basis_kind": basis_kind,
                                    "pc_a":plane[0],"pc_b":plane[1],
                                    "query_mode":qm,
                                    "language_a":la,"language_b":lb,
                                    "relative_read_pearson":pearson(va,vb),
                                    "relative_read_spearman":spearman(va,vb),
                                })

        save_rows(data_dir / "cross_language_geometry.csv", cross_rows)
        save_rows(data_dir / "behavior_surface_similarity.csv", behavior_rows)

        # -------------------------------------------------------------------------
        # Ridge extrema + loci.
        # -------------------------------------------------------------------------
        ridge_extrema, ridge_locus = ridge_extrema_and_loci(ridge_raw, args.ridge_topk)
        save_rows(data_dir / "ridge_extrema.csv", ridge_extrema)
        save_rows(data_dir / "ridge_locus_summary.csv", ridge_locus)

        jac_summary = base.aggregate_rows(
            state_jac_raw,
            ["condition","language","basis_kind","landmark","query_mode"],
            ["effective_rank","sigma1","sigma2","sigma3","sigma4",
             "sigma1_over_sigma2","sigma2_over_sigma3","sigma3_over_sigma4"],
        )
        save_rows(data_dir / "state_jacobian_summary.csv", jac_summary)


        save_json(
            data_dir / "atlas_stage_complete.json",
            {
                "languages": languages,
                "query_modes": query_modes,
                "atlas_keys": [str(k) for k in atlas_keys],
                "planes": [list(p) for p in planes],
                "grid_points": int(args.grid_points),
                "include_norta_surfaces": bool(args.include_norta_surfaces),
                "basis_pairs_per_language": int(args.basis_pairs_per_language),
                "checkpoint": str(args.checkpoint),
            },
        )
        print(f"[checkpoint] atlas stage complete: {data_dir / 'atlas_stage_complete.json'}")

    # -------------------------------------------------------------------------
    # Forced <text> CLIP-ESE at representative landmarks.
    # -------------------------------------------------------------------------
    clipese_top_rows = []
    clipese_phrase_rows = []
    representative = choose_representative_keys(surface_raw, languages)
    save_json(data_dir / "clipese_representative_keys.json", representative)

    if not args.skip_vocab:
        vocab_cache = get_forced_vocab_cache(
            model, clip_mod, Path(args.vocab), cache_dir / "forced_vocab",
            device, args.vocab_build_batch, args.checkpoint, args.vocab_limit,
        )

        for lang in languages:
            if lang not in representative:
                continue
            key = representative[lang]
            srow = dict(attacks[lang][attack_idx[lang][key]])
            pil = base.ensure_pil(srow["image"])
            img = base.preprocess_pil(preprocess, pil, device)
            pair = base.build_b13_pair(model, img, key, "synth", early_needed)
            rncomp, randcomp = per_pc_components(
                pair.delta_ord_btd, learned_basis, random_basis, 4
            )

            landmarks: list[tuple[str,str,np.ndarray]] = [
                ("anchor","rn",np.zeros(4,np.float32)),
            ]
            # Native-query extrema from ridge search, RN and random.
            for bk in ("rn","random"):
                q = [
                    r for r in ridge_extrema
                    if r["sample_key"]==key and r["language"]==lang
                    and r["query_mode"]=="native" and r["basis_kind"]==bk
                ]
                for ext in ("max","min"):
                    hit = [r for r in q if r["extremum"]==ext]
                    if hit:
                        rr = hit[0]
                        alpha = np.array([rr[f"alpha{i}"] for i in range(1,5)],np.float32)
                        landmarks.append((f"ridge_{ext}_native",bk,alpha))

            # RN PC1xPC2 grid extrema, native query.
            qsurf = [
                r for r in surface_raw
                if r["sample_key"]==key and r["condition"]=="synth"
                and r["language"]==lang and r["query_mode"]=="native"
                and r["basis_kind"]=="rn"
                and int(r["pc_a"])==1 and int(r["pc_b"])==2
            ]
            if qsurf:
                for ext, rr in (
                    ("surface_max_native",max(qsurf,key=lambda r:safe_float(r["relative_calibrated"]))),
                    ("surface_min_native",min(qsurf,key=lambda r:safe_float(r["relative_calibrated"]))),
                ):
                    alpha=np.zeros(4,np.float32)
                    alpha[0]=float(rr["alpha_a"]); alpha[1]=float(rr["alpha_b"])
                    landmarks.append((ext,"rn",alpha))

            # Stable unique.
            seen=set()
            for landmark,bk,alpha in landmarks:
                lk=(landmark,bk,tuple(np.round(alpha,5)))
                if lk in seen: continue
                seen.add(lk)
                comps = rncomp if bk=="rn" else randcomp
                post=make_alpha_post_batch(pair,comps,alpha[None,:])
                run=base.run_downstream(model,post,pair.early_states,surface_capture,has_rn=True)

                top_read=forced_vocab_topk(
                    model,run,vocab_cache,device,args.vocab_score_batch,args.vocab_topk
                )
                top_cos=vocab_cosine_topk(
                    run,vocab_cache,device,args.vocab_score_batch,args.vocab_topk
                )
                for metric,top in (("forced_relative_read",top_read),("image_text_cosine",top_cos)):
                    for rank,(word,score) in enumerate(top,1):
                        clipese_top_rows.append({
                            "sample_key":key,"language":lang,"landmark":landmark,
                            "basis_kind":bk,"metric":metric,"rank":rank,"vocab_entry":word,
                            "score":score,
                            **{f"alpha{i+1}":float(alpha[i]) for i in range(4)},
                        })

                phrase_read=clipese_phrase_beam(
                    model,clip_mod,run,
                    [x[0] for x in top_read],[x[1] for x in top_read],
                    device,args.pair_beam,args.vocab_build_batch,
                )
                phrase_cos=clipese_phrase_beam_cosine(
                    model,clip_mod,run,
                    [x[0] for x in top_cos],[x[1] for x in top_cos],
                    device,args.pair_beam,args.vocab_build_batch,
                )
                for metric,phrase,top in (
                    ("forced_relative_read",phrase_read,top_read),
                    ("image_text_cosine",phrase_cos,top_cos),
                ):
                    clipese_phrase_rows.append({
                        "sample_key":key,"language":lang,"landmark":landmark,
                        "basis_kind":bk,"metric":metric,
                        **{f"alpha{i+1}":float(alpha[i]) for i in range(4)},
                        **phrase,
                        "top10": " | ".join(f"{w}:{s:.3f}" for w,s in top),
                    })
                del run,post
                base.cleanup_cuda()

            del pair,img,pil
            base.cleanup_cuda()

    save_rows(data_dir / "clipese_top10.csv", clipese_top_rows)
    save_rows(data_dir / "clipese_best_phrases.csv", clipese_phrase_rows)

    # -------------------------------------------------------------------------
    # Selected plots.
    # -------------------------------------------------------------------------
    for lang in languages:
        for plane in planes:
            for qm in query_modes:
                plot_scalar_surface(
                    surface_summary,lang,qm,"rn",plane,args.grid_points,
                    plot_dir / f"surface__{lang}__{qm}__rn__PC{plane[0]}xPC{plane[1]}.png"
                )

    rn_cross = [r for r in cross_rows if r["basis_kind"]=="rn" and int(r["pc_a"])==1 and int(r["pc_b"])==2]
    rnd_cross = [r for r in cross_rows if r["basis_kind"]=="random" and int(r["pc_a"])==1 and int(r["pc_b"])==2]
    plot_cross_language_matrix(
        rn_cross,"distance_spearman",
        "RN PC1×PC2: cross-language surface distance geometry",
        plot_dir/"cross_language_distance_geometry__rn_PC1xPC2.png"
    )
    plot_cross_language_matrix(
        rnd_cross,"distance_spearman",
        "Random matched PC1×PC2: cross-language distance geometry",
        plot_dir/"cross_language_distance_geometry__random_PC1xPC2.png"
    )
    plot_cross_language_matrix(
        rn_cross,"tangent_principal_cos2_mean",
        "RN PC1×PC2: weaker tangent principal cosine",
        plot_dir/"cross_language_tangent_alignment__rn_PC1xPC2.png"
    )

    # -------------------------------------------------------------------------
    # Config + summary.
    # -------------------------------------------------------------------------
    config = {
        "checkpoint":args.checkpoint,
        "dataset_repo":args.dataset_repo,
        "languages":languages,
        "query_modes":query_modes,
        "atlas_keys":atlas_keys,
        "planes":planes,
        "grid_points":args.grid_points,
        "grid_bound":args.grid_bound,
        "ridge_points":args.ridge_points,
        "ridge_bound":args.ridge_bound,
        "ridge_radius":args.ridge_radius,
        "ridge_refine_steps":args.ridge_refine_steps,
        "state_jac_eps":args.state_jac_eps,
        "vocab":args.vocab,
        "skip_vocab":args.skip_vocab,
        "vocab_limit":args.vocab_limit,
        "state_block":STATE_BLOCK,
        "random_control":"matched tokenwise learned-PC coefficients, random feature directions orthogonal to learned RN basis",
        "ply_note":"PLY is a PCA3 / Isomap3 view. Full undistorted high-D mean B21 surfaces are in surface_state_means.npz.",
    }
    save_json(data_dir/"config.json",config)

    summary_lines = [
        "RN LOCAL CONTROL-MANIFOLD ATLAS",
        "="*72,
        "",
        f"languages: {','.join(languages)}",
        f"atlas sample_keys: {len(atlas_keys)}",
        f"surface planes: {planes}, grid={args.grid_points}x{args.grid_points}, bound=±{args.grid_bound}",
        f"ridge Sobol points: {args.ridge_points} + {args.ridge_refine_steps} refinement steps",
        "",
        "Learned B13 RN basis energy:",
    ]
    for i,(sv,ef) in enumerate(zip(
        basis_info["singular_values"].cpu().numpy(),
        basis_info["explained_energy"],
    ),1):
        summary_lines.append(f"  PC{i}: singular={float(sv):.6g}, pooled-energy={float(ef):.6f}")

    summary_lines += ["", "Cross-language RN PC1xPC2 geometry:"]
    for r in rn_cross:
        summary_lines.append(
            f"  {r['language_a']}-{r['language_b']}: "
            f"distance Spearman={safe_float(r['distance_spearman']):.4f}, "
            f"kNN8={safe_float(r['knn8_jaccard']):.4f}, "
            f"tangent cos2={safe_float(r['tangent_principal_cos2_mean']):.4f}, "
            f"Procrustes={safe_float(r['procrustes_residual_32d']):.4f}"
        )

    summary_lines += ["", "Ridge locus PCA (RN basis, native query, maxima):"]
    for r in ridge_locus:
        if r["basis_kind"]=="rn" and r["query_mode"]=="native" and r["extremum"]=="max":
            summary_lines.append(
                f"  {r['language']}: n={r['n_points']} "
                f"alpha-PC1={safe_float(r['pca_fraction_1']):.4f}, "
                f"alpha-PC2={safe_float(r['pca_fraction_2']):.4f}, "
                f"mean radius={safe_float(r['mean_radius']):.3f}"
            )

    if clipese_phrase_rows:
        summary_lines += ["", "Forced-<text> CLIP-ese landmark phrases:"]
        for r in clipese_phrase_rows:
            if r["basis_kind"]=="rn":
                summary_lines.append(
                    f"  {r['language']} / {r['landmark']} / {r['metric']}: "
                    f"{r['best_phrase']} ({safe_float(r['best_score']):+.4f})"
                )

    summary_lines += [
        "",
        "Important:",
        "  - 'manifold' remains a hypothesis; this atlas explicitly compares against",
        "    matched random feature directions to test whether the learned RN chart is special.",
        "  - PLY is only a 3-D embedding. Inspect PCA spectrum before calling it 'undistorted'.",
        "  - Full high-dimensional mean B21 state surfaces are preserved in NPZ.",
        "",
    ]
    (out/"SUMMARY.txt").write_text("\n".join(summary_lines),encoding="utf-8")

    # -------------------------------------------------------------------------
    # Compact handoff: aggregate science + PLY + selected plots, not giant raws/cache.
    # -------------------------------------------------------------------------
    include = [
        data_dir/"config.json",
        data_dir/"atlas_basis.npz",
        data_dir/"surface_summary.csv",
        data_dir/"surface_local_geometry.csv",
        data_dir/"surface_state_means.npz",
        data_dir/"surface_state_mean_manifest.csv",
        data_dir/"surface_3d_coordinates.csv",
        data_dir/"surface_pca_spectrum.csv",
        data_dir/"cross_language_geometry.csv",
        data_dir/"behavior_surface_similarity.csv",
        data_dir/"ridge_extrema.csv",
        data_dir/"ridge_locus_summary.csv",
        data_dir/"state_jacobian_summary.csv",
        data_dir/"clipese_representative_keys.json",
        data_dir/"clipese_top10.csv",
        data_dir/"clipese_best_phrases.csv",
        out/"SUMMARY.txt",
    ]
    include += sorted(ply_dir.glob("*.ply"))
    include += sorted(plot_dir.glob("*.png"))

    zpath=build_handoff_zip(out,include)
    print("\n[done]")
    print("  full local data:",data_dir.resolve())
    print("  PLY surfaces:   ",ply_dir.resolve())
    print("  plots:          ",plot_dir.resolve())
    print("  compact summary:   ",zpath.resolve())




if __name__ == "__main__":
    main()
