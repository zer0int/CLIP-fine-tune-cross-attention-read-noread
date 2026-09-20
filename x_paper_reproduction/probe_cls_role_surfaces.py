#!/usr/bin/env python3
r"""CLS / register role-plane surface atlas — RTA triplets, rank-4 extension, Blender Col (v4 ID fix)
===================================================================================

This v3 keeps every mu1/mu2 assay from v2, but changes the stimulus bank to
paired RTA-100-Triplet examples and adds a strictly supplemental rank-4 view.
No bridge/router/correction is executed.

Fixed role basis
----------------
mu1 and mu2 are loaded EXACTLY from the previous intact/no-RN GIPU oracle:

    <oracle_root>/<model>/mu_basis.npz

They are not refit on RTA.  mu3 and mu4 extend the same old register-mean
population: remove span(mu1,mu2) from the saved register means, then take the
top two uncentered residual SVD directions.  Thus the original rank-2 story is
preserved and rank-3/rank-4 are genuinely additional axes rather than a new
basis with silently rotated mu1/mu2.

RTA pairing
-----------
The default bank is five literal paired triplets from:

    zer0int/RTA-100-Triplet

with row types NoRTA, SynthRTA and RTA.  IDs are paired after stripping each
row's own prefix, so e.g. NoRTA_img_0001, SynthRTA_img_0001 and RTA_img_0001
share sample_key=img_0001.  object_label and attack_word are validated across
all three members.  Selected images are cached under <out>/rta_images/.

Fresh current-image register masks
----------------------------------
The basis is fixed, but per-image addresses are recomputed on the current RTA
images using the same no-RN intact visual tower:

  * pre-B13 visible register mask: residual norm >= 70 by default;
  * B23 register lineage: residual norm >= 60, capped to 1--4 with top-norm
    fallback.

Masks are then frozen while a local response surface is sampled.

What is preserved from v2
-------------------------
1. Natural CLS/register trajectories in the mu1/mu2 role plane.
2. CLS transduction surfaces at --anchor_blocks.
3. +mu2 moved-center sequences for Blender interpolation.
4. B13 RN and CLS-copy H5 "vectorial pheromone" surfaces.
5. Local mu1/mu2 Jacobians and angle plots.

New rank-4 paired-image atlas
-----------------------------
At --rta_compare_blocks (default B12,B20,B21,B22), sample both:

    PC1xPC2 == mu1 x mu2   (the original role plane)
    PC3xPC4 == mu3 x mu4   (the residual rank-4 extension)

for every selected NoRTA/SynthRTA/RTA image separately.  Each perturbation
changes CLS only and executes exactly one block.  The script records all four
CLS-source directional writes plus patch->CLS attention and CLS->fixed-register
lineage attention.  Condition-mean surfaces get PNG + PLY; literal per-image
triplets get compact PLY-only exports so they can be interpolated directly in
Blender without producing thousands of redundant PNGs.

Blender vertex colors
---------------------
Every surface and every *_field.ply now writes vertex RGBA properties

    red green blue alpha

which Blender imports as the color attribute named ``Col``.

  * surface PLY Col = the scalar/contour color mapping;
  * field PLY Col   = the gradient-arrow magnitude color mapping.

The existing *_field.ply still contains alpha coordinates, scalar value,
vector components and vector_magnitude.  Geometry exactly overlaps its surface.

Important output layout
-----------------------
<out>/
    rta_manifest.csv
    rank4_basis_audit.csv
    natural_role_plane*.csv/png
    cls_surface_points.csv
    cls_move_surface_points.csv
    b13_probe_surface_points.csv
    rta_rank4_surface_points_per_image.csv
    rta_rank4_surface_points_condition_mean.csv

    <model>/rta_role_oracle.npz

    surfaces/<model>/...
        existing v2 mu1/mu2 surfaces and move sequences
        RTA_CONDITION_MEAN/<condition>/Bxx/PC1xPC2|PC3xPC4/...
        RTA_PAIRED/<sample_key>/<condition>/Bxx/PC1xPC2|PC3xPC4/*.ply

    compact_summary_workspace_cls_role_surfaces.zip

House rule: pandas columns are accessed with bracket syntax only.  Attention
heads have already caused enough suffering.
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
import probe_tools_backbone as _backbone_tools
from probe_tools_analysis import (cosine_rows, normalize_probs, projected_direction_per_head, savefig, select_register_mask)


import argparse
import csv
import gc
import importlib
import importlib.util
import json
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm.auto import tqdm


MODEL_ORDER = ("pretrained", "gmp", "finetune_stripped")

DEFAULT_MANIFEST = "nop_bc/fixed_sink_manifest.csv"  # v2 compatibility only; v3 defaults to HF RTA triplets
DEFAULT_RTA_REPO = "zer0int/RTA-100-Triplet"
RTA_CONDITIONS = ("NoRTA", "SynthRTA", "RTA")
DEFAULT_ORACLE_ROOT = r"cls_gipu_exchange_no_rn"
DEFAULT_OUT = r"cls_mu_role_plane_surfaces_RTA_rank4"
DEFAULT_XATTN_CHECKPOINT = "REPLACE_WITH_CHECKPOINT.pt"

EPS = 1e-12


# =============================================================================
# Generic helpers
# =============================================================================

def parse_ints(text: str) -> list[int]:
    out: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(token))
    return sorted(set(out))


def parse_planes4(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for token in str(text).split(","):
        token = token.strip().lower().replace("pc", "")
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"Plane must look like 1x2 or 3x4, got {token!r}")
        a, b = token.split("x", 1)
        pair = (int(a), int(b))
        if pair[0] not in (1,2,3,4) or pair[1] not in (1,2,3,4) or pair[0] == pair[1]:
            raise ValueError(f"Invalid rank-4 plane {pair}")
        if pair not in out:
            out.append(pair)
    return out


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_local(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    alt = Path(__file__).resolve().parent / path_text
    return alt if alt.exists() else path


def save_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def meanfinite(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def pearson_rows(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    af = a.float() - a.float().mean(dim=-1, keepdim=True)
    bf = b.float() - b.float().mean(dim=-1, keepdim=True)
    num = (af * bf).sum(dim=-1)
    den = af.square().sum(dim=-1).sqrt() * bf.square().sum(dim=-1).sqrt()
    out = num / den.clamp_min(eps)
    return torch.where(den > eps, out, torch.full_like(out, float("nan")))


def source_direction_write(
    probs_bhts: torch.Tensor,
    values_bhsd: torch.Tensor,
    out_proj_weight: torch.Tensor,
    direction_d: torch.Tensor,
    source_index: int,
) -> torch.Tensor:
    """
    Signed residual contribution along `direction_d` from one source token.
    Returns [B,T_query].
    """
    heads = probs_bhts.shape[1]
    w = projected_direction_per_head(out_proj_weight, direction_d, heads)
    vdir = torch.einsum("bhsd,hd->bhs", values_bhsd.float(), w)
    return (
        probs_bhts[:, :, :, source_index]
        * vdir[:, :, source_index][:, :, None]
    ).sum(dim=1)


def exact_direction_write(
    attention_output_tbd: torch.Tensor,
    direction_d: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("tbd,d->bt", attention_output_tbd.float(), direction_d.float())


def normalize_values(base, values0, batch: int, heads: int, tokens: int) -> torch.Tensor:
    return base.normalize_qkv_shape(values0, batch, heads, tokens).float()


def rgba_u8(colors: np.ndarray) -> np.ndarray:
    """Convert matplotlib-like RGBA floats [0,1] to uint8 RGBA."""
    arr = np.asarray(colors, np.float64)
    if arr.shape[-1] == 3:
        alpha = np.ones(arr.shape[:-1] + (1,), dtype=np.float64)
        arr = np.concatenate([arr, alpha], axis=-1)
    arr = np.clip(arr, 0.0, 1.0)
    return np.rint(arr * 255.0).astype(np.uint8)


def write_ascii_surface_ply(
    path: Path,
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    z: np.ndarray,
    z_scale: float = 1.0,
    rgba: Optional[np.ndarray] = None,
) -> None:
    """
    Regular grid surface. Coordinates are (a,b,z*z_scale).

    If ``rgba`` is supplied as [na,nb,4] floats in [0,1], the PLY writes the
    conventional vertex properties ``red green blue alpha``. Blender's PLY
    importer exposes these as the color attribute named ``Col``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    na, nb = z.shape
    if rgba is None:
        rgba8 = np.full((na, nb, 4), 255, dtype=np.uint8)
    else:
        rgba8 = rgba_u8(rgba)
        if rgba8.shape != (na, nb, 4):
            raise ValueError(f"surface RGBA shape {rgba8.shape} != {(na, nb, 4)}")

    verts = []
    cols = []
    for ia, a in enumerate(a_vals):
        for ib, b in enumerate(b_vals):
            verts.append((float(a), float(b), float(z[ia, ib] * z_scale)))
            cols.append(tuple(int(x) for x in rgba8[ia, ib]))

    faces = []
    def idx(i, j):
        return i * nb + j

    for ia in range(na - 1):
        for ib in range(nb - 1):
            v00 = idx(ia, ib)
            v10 = idx(ia + 1, ib)
            v01 = idx(ia, ib + 1)
            v11 = idx(ia + 1, ib + 1)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment Blender imports red/green/blue/alpha as vertex color attribute Col\n")
        handle.write(f"element vertex {len(verts)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for (x, y, zz), (r, g, b, a8) in zip(verts, cols):
            handle.write(f"{x:.8g} {y:.8g} {zz:.8g} {r:d} {g:d} {b:d} {a8:d}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def write_ascii_vector_field_ply(
    path: Path,
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    z: np.ndarray,
    grad_a: np.ndarray,
    grad_b: np.ndarray,
    scalar_name: str = "scalar",
    normalized: bool = True,
    constant_size_scale: float = 0.22,
    width_ratio: float = 0.40,
    rgba: Optional[np.ndarray] = None,
) -> None:
    """
    Export a triangle-glyph vector field for Blender with vertex attributes:
        alpha_a, alpha_b, <scalar_name>, vector_alpha_a, vector_alpha_b,
        vector_magnitude, and RGBA vertex colors.

    ``red green blue alpha`` are intentionally used because Blender imports
    them as the ``Col`` color attribute. Coordinates exactly overlap the
    corresponding surface PLY.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    U = np.asarray(grad_b, np.float64)  # horizontal component
    V = np.asarray(grad_a, np.float64)  # vertical component
    mag = np.sqrt(U * U + V * V)
    max_mag = max(float(np.nanmax(mag)), 1e-12)

    H = len(a_vals)
    W = len(b_vals)
    if rgba is None:
        rgba8 = np.full((H, W, 4), 255, dtype=np.uint8)
    else:
        rgba8 = rgba_u8(rgba)
        if rgba8.shape != (H, W, 4):
            raise ValueError(f"field RGBA shape {rgba8.shape} != {(H, W, 4)}")

    db = float(np.median(np.diff(b_vals))) if W > 1 else 1.0
    da = float(np.median(np.diff(a_vals))) if H > 1 else 1.0
    base = min(abs(da), abs(db))
    base_len = constant_size_scale * base

    verts = []
    faces = []
    scalar_values = []
    alpha_a = []
    alpha_b = []
    vector_alpha_a = []
    vector_alpha_b = []
    vector_magnitude = []
    vertex_colors = []
    face_index = 0

    for ia, a in enumerate(a_vals):
        for ib, b in enumerate(b_vals):
            m = float(mag[ia, ib])
            if m <= 1e-14:
                continue
            u = float(U[ia, ib]) / m
            v = float(V[ia, ib]) / m
            if normalized:
                scale = 1.0
            else:
                scale = 0.20 + 0.80 * (m / max_mag)
            length = base_len * scale
            width = width_ratio * length
            x = float(a)
            y = float(b)
            zz = float(z[ia, ib])
            tip = np.array([x + 0.55 * length * v, y + 0.55 * length * u, zz], dtype=np.float64)
            back = np.array([x - 0.25 * length * v, y - 0.25 * length * u, zz], dtype=np.float64)
            perp = np.array([-u, v, 0.0], dtype=np.float64)
            left = back + 0.5 * width * perp
            right = back - 0.5 * width * perp
            tri = np.stack([tip, left, right], axis=0)
            verts.append(tri)
            faces.append((face_index, face_index + 1, face_index + 2))
            col = tuple(int(x) for x in rgba8[ia, ib])
            for _ in range(3):
                scalar_values.append(float(z[ia, ib]))
                alpha_a.append(float(a))
                alpha_b.append(float(b))
                vector_alpha_a.append(float(grad_a[ia, ib]))
                vector_alpha_b.append(float(grad_b[ia, ib]))
                vector_magnitude.append(m)
                vertex_colors.append(col)
            face_index += 3

    if not verts:
        return

    vertices = np.concatenate(verts, axis=0)
    with path.open('w', encoding='ascii', newline='\n') as handle:
        handle.write('ply\nformat ascii 1.0\n')
        handle.write('comment Blender imports red/green/blue/alpha as vertex color attribute Col\n')
        handle.write(f'element vertex {len(vertices)}\n')
        handle.write('property float x\nproperty float y\nproperty float z\n')
        handle.write('property float alpha_a\nproperty float alpha_b\n')
        handle.write(f'property float {scalar_name}\n')
        handle.write('property float vector_alpha_a\nproperty float vector_alpha_b\nproperty float vector_magnitude\n')
        handle.write('property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n')
        handle.write(f'element face {len(faces)}\n')
        handle.write('property list uchar int vertex_indices\nend_header\n')
        for i, (x, y, zz) in enumerate(vertices):
            r, g, b, a8 = vertex_colors[i]
            handle.write(
                f'{x:.8g} {y:.8g} {zz:.8g} '
                f'{alpha_a[i]:.8g} {alpha_b[i]:.8g} {scalar_values[i]:.8g} '
                f'{vector_alpha_a[i]:.8g} {vector_alpha_b[i]:.8g} {vector_magnitude[i]:.8g} '
                f'{r:d} {g:d} {b:d} {a8:d}\n'
            )
        for face in faces:
            handle.write(f'3 {face[0]} {face[1]} {face[2]}\n')


def format_signed_factor(value: float) -> str:
    text = f"{value:+.3f}"
    text = text.replace('+', 'p').replace('-', 'm')
    return text


def summarize_grid(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(list(group_cols), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(group_cols, key)}
        row["n"] = len(group)
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").to_numpy(np.float64)
            vals = vals[np.isfinite(vals)]
            row[metric] = float(vals.mean()) if vals.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# RN + oracle
# =============================================================================

def load_trained_rn_token(base, args) -> tuple[torch.Tensor, dict[str, Any]]:
    checkpoint = Path(args.xattn_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    for name in (args.xattn_module, args.pickle_module, "oaiclip", "clip"):
        if not name:
            continue
        try:
            importlib.import_module(name)
        except Exception:
            pass

    obj = base.D.safe_torch_load(checkpoint)
    state = base.D.extract_state_dict(obj, str(checkpoint))
    del obj

    key = "visual.read_null_token"
    if key not in state:
        candidates = [k for k in state if k.endswith("visual.read_null_token")]
        if len(candidates) != 1:
            raise KeyError(f"RN token not found uniquely: {candidates[:10]}")
        key = candidates[0]

    token = state[key].detach().float().cpu().clone()
    audit = {
        "checkpoint": str(checkpoint),
        "state_key": key,
        "shape": list(token.shape),
        "norm": float(token.norm()),
    }
    return token, audit


@dataclass
class Oracle:
    mu1: np.ndarray
    mu2: np.ndarray
    mu3: np.ndarray
    mu4: np.ndarray
    b23_reg_mask: np.ndarray
    b13_reg_mask: np.ndarray
    stim_id: np.ndarray
    source_path: str

    @property
    def basis4(self) -> np.ndarray:
        return np.stack([self.mu1, self.mu2, self.mu3, self.mu4], axis=0)


def _normalize_rows_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def _extended_basis4_from_oracle_npz(data: Mapping[str, Any], width: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Preserve the previously fitted mu1/mu2 EXACTLY, then extend the same
    register-mean population with two orthogonal residual SVD directions.

    This is deliberately not a new RTA fit: rank-3/rank-4 are an extension of
    the existing role basis, so the old mu1/mu2 story is not silently replaced.
    """
    basis2 = np.asarray(data["mu_basis"], np.float32)[:2]
    if basis2.shape != (2, width):
        raise RuntimeError(f"oracle mu basis shape {basis2.shape} != (2,{width})")
    basis2 = _normalize_rows_np(basis2)

    if "register_means" not in data:
        raise KeyError(
            "Previous mu_basis.npz has no register_means; cannot extend to rank 4 "
            "without refitting. Re-run the no-RN CLS/GIPU oracle or use v2."
        )
    X = np.asarray(data["register_means"], np.float64)
    if X.ndim != 2 or X.shape[1] != width:
        raise RuntimeError(f"register_means shape {X.shape} incompatible with width={width}")

    # Remove the fixed rank-2 span, then take two residual uncentered directions.
    residual = X - (X @ basis2.T) @ basis2
    _u, s, vt = np.linalg.svd(residual, full_matrices=False)
    if vt.shape[0] < 2:
        raise RuntimeError("Not enough residual rank to define mu3/mu4")
    extra = vt[:2].astype(np.float64)

    # Numerically enforce orthogonality to the fixed first two axes and each other.
    rows = [basis2[0].astype(np.float64), basis2[1].astype(np.float64)]
    for v in extra:
        w = v.copy()
        for q in rows:
            w -= np.dot(w, q) * q
        n = np.linalg.norm(w)
        if n < 1e-8:
            raise RuntimeError("Residual rank extension collapsed during orthogonalization")
        rows.append(w / n)
    basis4 = np.stack(rows, axis=0).astype(np.float32)
    return basis4, s[:2].astype(np.float32)


def load_extended_basis4(oracle_root: Path, model_name: str, width: int) -> tuple[np.ndarray, str, np.ndarray]:
    path = oracle_root / model_name / "mu_basis.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing previous oracle: {path}\n"
            "Run the no-RN CLS/GIPU probe first or pass --oracle_root."
        )
    with np.load(path, allow_pickle=True) as data:
        basis4, residual_s = _extended_basis4_from_oracle_npz(data, width)
    return basis4, str(path), residual_s


def canonical_rta_pair_id(raw: Any, variant: str) -> str:
    """
    Canonicalize subset-prefixed RTA IDs without regex character-class games.

    Examples:
        RTA_img_0001      -> img_0001
        SynthRTA_img_0001 -> img_0001
        NoRTA_img_0001    -> img_0001

    Prefix matching is case-insensitive. Only the requested subset prefix is
    removed; the remainder is then stripped of ordinary separators.
    """
    value = str(raw).strip()
    prefix = str(variant).strip()
    if not value or not prefix:
        return value

    if value.lower().startswith(prefix.lower()):
        remainder = value[len(prefix):]
        remainder = remainder.lstrip(" \t\r\n_:-./")
        return remainder if remainder else value
    return value


def _natural_id_key(value: str) -> tuple:
    # Natural sort: img_2 before img_10.
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def _row_image(row: Mapping[str, Any]) -> Image.Image:
    im = row["image"]
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if isinstance(im, dict):
        if im.get("path"):
            return Image.open(im["path"]).convert("RGB")
        if im.get("bytes"):
            import io
            return Image.open(io.BytesIO(im["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported HF image object: {type(im)}")


def build_rta_manifest(args, out: Path) -> pd.DataFrame:
    """Load paired NoRTA/SynthRTA/RTA rows and cache the selected images locally."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "RTA v3 requires Hugging Face datasets (`pip install datasets`) on the run machine."
        ) from exc
    ds = load_dataset(args.rta_repo, split=args.rta_split)
    required = {"type", "image", "id"}
    missing = sorted(required - set(ds.column_names))
    if missing:
        raise KeyError(f"{args.rta_repo} missing required columns {missing}; got {ds.column_names}")

    maps: dict[str, dict[str, int]] = {c: {} for c in RTA_CONDITIONS}
    # Read only scalar Arrow columns here; avoid decoding thousands of images.
    type_col = ds["type"]
    id_col = ds["id"]
    for i, (typ, raw_id) in enumerate(zip(type_col, id_col)):
        cond = str(typ)
        if cond not in maps:
            continue
        key = canonical_rta_pair_id(raw_id, cond)
        if key in maps[cond]:
            raise RuntimeError(f"Duplicate canonical {cond} id: {key}")
        maps[cond][key] = i

    shared = set.intersection(*(set(maps[c]) for c in RTA_CONDITIONS))
    if not shared:
        raise RuntimeError("No shared NoRTA/SynthRTA/RTA canonical IDs found")

    if args.rta_pair_ids:
        requested = [x.strip() for x in str(args.rta_pair_ids).split(",") if x.strip()]
        keys = []
        for raw in requested:
            # Accept full subset-prefixed IDs or canonical img_XXXX names.
            key = raw
            for cond in RTA_CONDITIONS:
                key = canonical_rta_pair_id(key, cond)
            if key not in shared:
                raise KeyError(f"Requested RTA pair id {raw!r} -> {key!r} is not shared by all 3 variants")
            keys.append(key)
    else:
        keys = sorted(shared, key=_natural_id_key)[: int(args.rta_triplets)]

    cache = out / "rta_images"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in keys:
        # Validate the semantic pair before writing anything.
        objs, atks = [], []
        triplet_rows = {}
        for cond in RTA_CONDITIONS:
            row = ds[int(maps[cond][key])]
            triplet_rows[cond] = row
            objs.append(str(row.get("object_label", "")).strip())
            atks.append(str(row.get("attack_word", "")).strip())
        if len(set(objs)) != 1 or len(set(atks)) != 1:
            raise RuntimeError(f"Triplet {key}: label mismatch object={objs}, attack={atks}")

        for cond in RTA_CONDITIONS:
            row = triplet_rows[cond]
            # Keep the user's familiar literal IDs in filenames when possible.
            raw_id = str(row["id"])
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id)
            path = cache / f"{safe}.png"
            if not path.is_file():
                _row_image(row).save(path, format="PNG")
            rows.append({
                "path": str(path),
                "stim_id": raw_id,
                "sample_key": key,
                "condition": cond,
                "object_label": objs[0],
                "attack_word": atks[0],
            })

    manifest = pd.DataFrame(rows)
    # Group by triplet, with clean -> synthetic -> natural attack for easy inspection.
    cond_order = {c: i for i, c in enumerate(RTA_CONDITIONS)}
    manifest["_cond_order"] = manifest["condition"].map(cond_order)
    manifest = manifest.sort_values(["sample_key", "_cond_order"], key=lambda s: s.map(_natural_id_key) if s.name == "sample_key" else s).drop(columns="_cond_order").reset_index(drop=True)
    manifest.to_csv(out / "rta_manifest.csv", index=False)
    return manifest


@torch.no_grad()
def build_rta_oracle(bundle, manifest: pd.DataFrame, args, model_dir: Path) -> Oracle:
    """
    Keep the old mu1/mu2 basis (plus residual mu3/mu4 extension), but recompute
    B13-visible and B23-lineage register masks on the current RTA images.
    """
    visual = bundle.model.visual
    width = int(visual.positional_embedding.shape[1])
    patches = int(visual.positional_embedding.shape[0] - 1)
    basis4, source_path, residual_s = load_extended_basis4(Path(args.oracle_root), bundle.name, width)

    n = len(manifest)
    b13_all = np.zeros((n, patches), np.uint8)
    b23_all = np.zeros((n, patches), np.uint8)

    for start in tqdm(range(0, n, args.batch_size), desc=f"{bundle.name}/RTA masks", unit="batch"):
        chunk = manifest.iloc[start:start + args.batch_size]
        tensors = []
        for row in chunk.itertuples(index=False):
            with Image.open(str(row.path)) as image:
                tensors.append(bundle.preprocess(image.convert("RGB")))
        images = torch.stack(tensors).to(bundle.device, dtype=bundle.model.dtype)
        x = visual._prepare_tokens(images)
        B = images.shape[0]
        b13 = torch.zeros(B, patches, dtype=torch.bool, device=bundle.device)
        for block_index, block in enumerate(visual.transformer.resblocks):
            if block_index == 13:
                prenorm = x[1:1 + patches].float().norm(dim=-1).T
                b13 = prenorm >= float(args.b13_register_threshold)
            x = block(x)
        final_patch = x[1:1 + patches].permute(1, 0, 2).float()
        b23 = select_register_mask(
            final_patch.norm(dim=-1),
            float(args.final_register_threshold),
            int(args.final_register_min),
            int(args.final_register_max),
        )
        b13_all[start:start + B] = b13.cpu().numpy().astype(np.uint8)
        b23_all[start:start + B] = b23.cpu().numpy().astype(np.uint8)
        del images, x, final_patch, b13, b23
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()

    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "rta_role_oracle.npz",
        basis4=basis4,
        mu1=basis4[0], mu2=basis4[1], mu3=basis4[2], mu4=basis4[3],
        residual_extension_singular_values=residual_s,
        b13_reg_mask=b13_all,
        b23_reg_mask=b23_all,
        stim_id=manifest["stim_id"].astype(str).to_numpy(dtype=object),
        sample_key=manifest["sample_key"].astype(str).to_numpy(dtype=object),
        condition=manifest["condition"].astype(str).to_numpy(dtype=object),
        source_oracle=np.asarray(source_path, dtype=object),
    )
    return Oracle(
        mu1=basis4[0], mu2=basis4[1], mu3=basis4[2], mu4=basis4[3],
        b23_reg_mask=b23_all,
        b13_reg_mask=b13_all,
        stim_id=manifest["stim_id"].astype(str).to_numpy(dtype=object),
        source_path=source_path,
    )


def load_oracle(
    oracle_root: Path,
    model_name: str,
    manifest: pd.DataFrame,
    width: int,
    patches: int,
) -> Oracle:
    """v2 compatibility path for local-manifest post-hoc use."""
    path = oracle_root / model_name / "mu_basis.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        basis4, _ = _extended_basis4_from_oracle_npz(data, width)
        b23 = np.asarray(data["b23_reg_mask"], np.uint8)
        b13 = np.asarray(data["b13_reg_mask"], np.uint8)
        ids = np.asarray(data["stim_id"], dtype=object).astype(str)
    current = manifest["stim_id"].astype(str).to_numpy()
    if len(ids) < len(current) or not np.array_equal(ids[:len(current)], current):
        raise RuntimeError(f"Oracle stim_id order does not match current manifest for {model_name}")
    if b23.shape[1] != patches or b13.shape[1] != patches:
        raise RuntimeError("Oracle patch-count mismatch")
    return Oracle(
        mu1=basis4[0], mu2=basis4[1], mu3=basis4[2], mu4=basis4[3],
        b23_reg_mask=b23[:len(current)], b13_reg_mask=b13[:len(current)],
        stim_id=ids[:len(current)], source_path=str(path),
    )


# =============================================================================
# Baseline state capture
# =============================================================================

@torch.no_grad()
def collect_baseline_states(
    bundle,
    manifest: pd.DataFrame,
    oracle: Oracle,
    args,
) -> tuple[
    dict[int, list[torch.Tensor]],
    pd.DataFrame,
]:
    """
    Return:
      pre_states_by_block[block] = list of CPU [T,B,D] tensors, one per image batch
      natural trajectory dataframe (per block/model mean calculated later)
    """
    visual = bundle.model.visual
    capture_blocks = sorted(set(args.anchor_blocks) | set(args.rta_compare_blocks) | set(range(0, 14)))
    max_block = max(capture_blocks)

    states: dict[int, list[torch.Tensor]] = {b: [] for b in capture_blocks}
    rows: list[dict[str, Any]] = []

    mu1 = torch.from_numpy(oracle.mu1).to(bundle.device).float()
    mu2 = torch.from_numpy(oracle.mu2).to(bundle.device).float()
    mu3 = torch.from_numpy(oracle.mu3).to(bundle.device).float()
    mu4 = torch.from_numpy(oracle.mu4).to(bundle.device).float()

    for start in tqdm(
        range(0, len(manifest), args.batch_size),
        desc=f"{bundle.name}/baseline",
        unit="batch",
    ):
        chunk = manifest.iloc[start:start + args.batch_size]
        tensors, ids = [], []
        for row in chunk.itertuples(index=False):
            with Image.open(str(row.path)) as image:
                tensors.append(bundle.preprocess(image.convert("RGB")))
            ids.append(str(row.stim_id))
        conditions = chunk["condition"].astype(str).tolist() if "condition" in chunk.columns else ["all"] * len(chunk)
        sample_keys = chunk["sample_key"].astype(str).tolist() if "sample_key" in chunk.columns else ids
        images = torch.stack(tensors).to(bundle.device, dtype=bundle.model.dtype)
        batch = images.shape[0]

        fixed_reg = torch.from_numpy(
            oracle.b23_reg_mask[start:start + batch].astype(bool)
        ).to(bundle.device)

        x = visual._prepare_tokens(images)

        for block_index in range(max_block + 1):
            if block_index in capture_blocks:
                states[block_index].append(x.detach().float().cpu())

                patch = x[1:1 + fixed_reg.shape[1]].permute(1, 0, 2).float()
                reg_count = fixed_reg.sum(dim=-1).float().clamp_min(1.0)
                reg_mean = (
                    patch * fixed_reg[:, :, None].float()
                ).sum(dim=1) / reg_count[:, None]
                cls = x[0].float()

                cls_mu1_cos = cosine_rows(cls, mu1.view(1, -1).expand_as(cls))
                cls_mu2_cos = cosine_rows(cls, mu2.view(1, -1).expand_as(cls))
                reg_mu1_cos = cosine_rows(
                    reg_mean, mu1.view(1, -1).expand_as(reg_mean)
                )
                reg_mu2_cos = cosine_rows(
                    reg_mean, mu2.view(1, -1).expand_as(reg_mean)
                )
                cls_mu3_cos = cosine_rows(cls, mu3.view(1, -1).expand_as(cls))
                cls_mu4_cos = cosine_rows(cls, mu4.view(1, -1).expand_as(cls))
                reg_mu3_cos = cosine_rows(reg_mean, mu3.view(1, -1).expand_as(reg_mean))
                reg_mu4_cos = cosine_rows(reg_mean, mu4.view(1, -1).expand_as(reg_mean))

                cls_mu1_coef = cls @ mu1
                cls_mu2_coef = cls @ mu2
                cls_mu3_coef = cls @ mu3
                cls_mu4_coef = cls @ mu4
                reg_mu1_coef = reg_mean @ mu1
                reg_mu2_coef = reg_mean @ mu2
                reg_mu3_coef = reg_mean @ mu3
                reg_mu4_coef = reg_mean @ mu4

                for bi in range(batch):
                    rows.append({
                        "model_name": bundle.name,
                        "stim_id": ids[bi],
                        "sample_key": sample_keys[bi],
                        "condition": conditions[bi],
                        "block": int(block_index),
                        "cls_mu1_cos": float(cls_mu1_cos[bi]),
                        "cls_mu2_cos": float(cls_mu2_cos[bi]),
                        "cls_mu1_coef": float(cls_mu1_coef[bi]),
                        "cls_mu2_coef": float(cls_mu2_coef[bi]),
                        "cls_mu3_cos": float(cls_mu3_cos[bi]),
                        "cls_mu4_cos": float(cls_mu4_cos[bi]),
                        "cls_mu3_coef": float(cls_mu3_coef[bi]),
                        "cls_mu4_coef": float(cls_mu4_coef[bi]),
                        "cls_norm": float(cls[bi].norm()),
                        "reg_mu1_cos": float(reg_mu1_cos[bi]),
                        "reg_mu2_cos": float(reg_mu2_cos[bi]),
                        "reg_mu1_coef": float(reg_mu1_coef[bi]),
                        "reg_mu2_coef": float(reg_mu2_coef[bi]),
                        "reg_mu3_cos": float(reg_mu3_cos[bi]),
                        "reg_mu4_cos": float(reg_mu4_cos[bi]),
                        "reg_mu3_coef": float(reg_mu3_coef[bi]),
                        "reg_mu4_coef": float(reg_mu4_coef[bi]),
                        "reg_norm": float(reg_mean[bi].norm()),
                    })

            if block_index <= max_block:
                x = visual.transformer.resblocks[block_index](x)

        del images, x
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()

    return states, pd.DataFrame(rows)


# =============================================================================
# CLS transduction surfaces
# =============================================================================

@torch.no_grad()
def evaluate_cls_surface_batch(
    *,
    base,
    bundle,
    block_index: int,
    pre_tbd_cpu: torch.Tensor,
    fixed_b13_bp_cpu: torch.Tensor,
    mu1_cpu: torch.Tensor,
    mu2_cpu: torch.Tensor,
    grid_points: Sequence[tuple[float, float]],
    args,
    center_a_mu1: float = 0.0,
    center_b_mu2: float = 0.0,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    One image-batch at one block, many grid points.
    Expand grid points into batch dimension and execute exactly one block.
    """
    visual = bundle.model.visual
    block = visual.transformer.resblocks[block_index]

    pre = pre_tbd_cpu.to(bundle.device, dtype=bundle.model.dtype)
    batch0 = pre.shape[1]
    tokens = pre.shape[0]
    patch_count = tokens - 1
    heads = int(block.attn.num_heads)

    fixed_b13 = fixed_b13_bp_cpu.to(bundle.device).bool()
    mu1 = mu1_cpu.to(bundle.device).float()
    mu2 = mu2_cpu.to(bundle.device).float()

    rows = []

    for gp_start in range(0, len(grid_points), args.grid_chunk):
        points = grid_points[gp_start:gp_start + args.grid_chunk]
        g = len(points)

        # [T,B,G,D] -> [T,B*G,D], image-major within each grid point
        x = pre[:, :, None, :].expand(tokens, batch0, g, pre.shape[-1]).clone()

        a = torch.tensor([p[0] for p in points], device=bundle.device, dtype=x.dtype)
        b = torch.tensor([p[1] for p in points], device=bundle.device, dtype=x.dtype)
        a_total = a + float(center_a_mu1)
        b_total = b + float(center_b_mu2)
        delta = (
            a_total[:, None] * mu1.to(x.dtype)[None, :]
            + b_total[:, None] * mu2.to(x.dtype)[None, :]
        )  # [G,D]

        x[0] = x[0] + delta[None, :, :]
        x = x.permute(0, 2, 1, 3).reshape(tokens, g * batch0, -1)

        ln1 = block.ln_1(x)
        attention_output, probs0 = block.attention(
            ln1,
            need_weights=True,
            capture=True,
        )
        probs = normalize_probs(base, probs0, g * batch0, heads, tokens)
        values = normalize_values(
            base, block.attn.last_v, g * batch0, heads, tokens
        )

        cls_mu1 = source_direction_write(
            probs,
            values,
            block.attn.out_proj.weight,
            mu1,
            source_index=0,
        )
        cls_mu2 = source_direction_write(
            probs,
            values,
            block.attn.out_proj.weight,
            mu2,
            source_index=0,
        )
        exact_mu1 = exact_direction_write(attention_output, mu1)

        # Spatial queries only.
        cls_mu1_patch = cls_mu1[:, 1:1 + patch_count]
        cls_mu2_patch = cls_mu2[:, 1:1 + patch_count]
        exact_mu1_patch = exact_mu1[:, 1:1 + patch_count]

        cls_ratio = (
            cls_mu1_patch.abs().sum(dim=-1)
            / exact_mu1_patch.abs().sum(dim=-1).clamp_min(EPS)
        )
        corr = pearson_rows(exact_mu1_patch, cls_mu1_patch)

        patch_to_cls_attn = probs[:, :, 1:1 + patch_count, 0].mean(dim=(1, 2))

        # CLS -> fixed B13 register set, using the repeated per-image mask.
        mask = fixed_b13[None].expand(g, batch0, patch_count).reshape(
            g * batch0, patch_count
        )
        cls_to_b13 = (
            probs[:, :, 0, 1:1 + patch_count]
            * mask[:, None, :].float()
        ).sum(dim=-1).mean(dim=1)

        # Reshape [G*B] -> [G,B].
        metrics = {
            "cls_source_mu1_signed_mean": cls_mu1_patch.mean(dim=-1),
            "cls_source_mu2_signed_mean": cls_mu2_patch.mean(dim=-1),
            "cls_source_mu1_abs_mean": cls_mu1_patch.abs().mean(dim=-1),
            "cls_source_mu1_abs_l1_ratio": cls_ratio,
            "mu1_delta_corr_cls_source": corr,
            "patch_query_to_cls_attn_mean": patch_to_cls_attn,
            "cls_query_to_b13reg_attn": cls_to_b13,
        }

        for local_g, (aa, bb) in enumerate(points):
            sl = slice(local_g * batch0, (local_g + 1) * batch0)
            row = {
                "model_name": bundle.name,
                "surface_type": "cls_transduction",
                "block": int(block_index),
                "a_mu1": float(aa),
                "b_mu2": float(bb),
                "center_a_mu1": float(center_a_mu1),
                "center_b_mu2": float(center_b_mu2),
                "n_images": int(batch0),
            }
            if extra_fields:
                row.update(dict(extra_fields))
            for name, tensor in metrics.items():
                row[name] = float(tensor[sl].float().mean())
            rows.append(row)

        base.clear_attn_cache(block)
        del x, ln1, attention_output, probs, values
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()

    return rows


def aggregate_surface_batches(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    metrics = [
        "cls_source_mu1_signed_mean",
        "cls_source_mu2_signed_mean",
        "cls_source_mu1_abs_mean",
        "cls_source_mu1_abs_l1_ratio",
        "mu1_delta_corr_cls_source",
        "patch_query_to_cls_attn_mean",
        "cls_query_to_b13reg_attn",
    ]
    group_cols = ["model_name", "surface_type", "block", "a_mu1", "b_mu2"]
    for optional in (
        "move_family",
        "move_index",
        "move_factor",
        "center_a_mu1",
        "center_b_mu2",
    ):
        if optional in frame.columns:
            group_cols.append(optional)
    return summarize_grid(
        frame,
        group_cols=tuple(group_cols),
        metric_cols=metrics,
    )


# =============================================================================
# B13 probe "pheromone" surfaces
# =============================================================================

@torch.no_grad()
def evaluate_b13_probe_surface_batch(
    *,
    base,
    bundle,
    pre_b13_tbd_cpu: torch.Tensor,
    fixed_b13_bp_cpu: torch.Tensor,
    rn_token_cpu: torch.Tensor,
    mu1_cpu: torch.Tensor,
    mu2_cpu: torch.Tensor,
    grid_points: Sequence[tuple[float, float]],
    center: str,
    args,
) -> list[dict[str, Any]]:
    visual = bundle.model.visual
    block = visual.transformer.resblocks[13]

    pre = pre_b13_tbd_cpu.to(bundle.device, dtype=bundle.model.dtype)
    batch0 = pre.shape[1]
    patch_count = pre.shape[0] - 1
    heads = int(block.attn.num_heads)
    if args.probe_head >= heads:
        raise ValueError(f"probe_head H{args.probe_head} >= heads={heads}")

    fixed_b13 = fixed_b13_bp_cpu.to(bundle.device).bool()
    rn = rn_token_cpu.to(bundle.device).float()
    mu1 = mu1_cpu.to(bundle.device).float()
    mu2 = mu2_cpu.to(bundle.device).float()

    rows = []

    for gp_start in range(0, len(grid_points), args.grid_chunk):
        points = grid_points[gp_start:gp_start + args.grid_chunk]
        g = len(points)

        base_x = pre[:, :, None, :].expand(
            pre.shape[0], batch0, g, pre.shape[-1]
        ).clone()

        aa = torch.tensor([p[0] for p in points], device=bundle.device, dtype=pre.dtype)
        bb = torch.tensor([p[1] for p in points], device=bundle.device, dtype=pre.dtype)
        delta = (
            aa[:, None] * mu1.to(pre.dtype)[None, :]
            + bb[:, None] * mu2.to(pre.dtype)[None, :]
        )

        if center == "RN":
            probe = rn.to(pre.dtype)[None, None, :].expand(batch0, g, -1).clone()
        elif center == "CLS_COPY":
            probe = pre[0][:, None, :].expand(batch0, g, -1).clone()
        else:
            raise ValueError(center)

        probe = probe + delta[None, :, :]

        # [T,B,G,D] -> [T,B*G,D], append probe last.
        x = base_x.permute(0, 2, 1, 3).reshape(pre.shape[0], g * batch0, -1)
        probe_flat = probe.permute(1, 0, 2).reshape(g * batch0, -1)
        x = torch.cat([x, probe_flat[None]], dim=0)

        tokens = x.shape[0]
        probe_index = tokens - 1

        ln1 = block.ln_1(x)
        _out, probs0 = block.attention(
            ln1,
            need_weights=True,
            capture=True,
        )
        probs = normalize_probs(base, probs0, g * batch0, heads, tokens)
        h = args.probe_head
        p = probs[:, h]

        mask = fixed_b13[None].expand(g, batch0, patch_count).reshape(
            g * batch0, patch_count
        )

        cls_to_probe = p[:, 0, probe_index]
        probe_to_reg = (
            p[:, probe_index, 1:1 + patch_count] * mask.float()
        ).sum(dim=-1)

        # Mean fixed-register query -> probe.
        reg_query_to_probe_num = (
            p[:, 1:1 + patch_count, probe_index] * mask.float()
        ).sum(dim=-1)
        reg_query_to_probe_den = mask.sum(dim=-1).float().clamp_min(1.0)
        reg_query_to_probe = reg_query_to_probe_num / reg_query_to_probe_den

        cls_to_reg = (
            p[:, 0, 1:1 + patch_count] * mask.float()
        ).sum(dim=-1)
        probe_to_cls = p[:, probe_index, 0]

        metrics = {
            "h5_cls_to_probe_attn": cls_to_probe,
            "h5_probe_to_b13reg_attn": probe_to_reg,
            "h5_b13reg_query_to_probe_attn_mean": reg_query_to_probe,
            "h5_cls_to_b13reg_attn": cls_to_reg,
            "h5_probe_to_cls_attn": probe_to_cls,
        }

        for local_g, (a0, b0) in enumerate(points):
            sl = slice(local_g * batch0, (local_g + 1) * batch0)
            row = {
                "model_name": bundle.name,
                "surface_type": "b13_probe",
                "probe_center": center,
                "block": 13,
                "head": int(h),
                "a_mu1": float(a0),
                "b_mu2": float(b0),
                "n_images": int(batch0),
            }
            for name, tensor in metrics.items():
                row[name] = float(tensor[sl].float().mean())
            rows.append(row)

        base.clear_attn_cache(block)
        del x, ln1, probs
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()

    return rows


def aggregate_probe_batches(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    metrics = [
        "h5_cls_to_probe_attn",
        "h5_probe_to_b13reg_attn",
        "h5_b13reg_query_to_probe_attn_mean",
        "h5_cls_to_b13reg_attn",
        "h5_probe_to_cls_attn",
    ]
    return summarize_grid(
        frame,
        group_cols=(
            "model_name",
            "surface_type",
            "probe_center",
            "block",
            "head",
            "a_mu1",
            "b_mu2",
        ),
        metric_cols=metrics,
    )


# =============================================================================
# RTA paired per-image rank-4 plane surfaces
# =============================================================================

@torch.no_grad()
def evaluate_rta_cls_plane_batch_per_image(
    *,
    base,
    bundle,
    block_index: int,
    pre_tbd_cpu: torch.Tensor,
    fixed_b13_bp_cpu: torch.Tensor,
    fixed_b23_bp_cpu: torch.Tensor,
    basis4_cpu: torch.Tensor,
    plane: tuple[int, int],
    grid_points: Sequence[tuple[float, float]],
    metadata: Sequence[Mapping[str, Any]],
    args,
) -> list[dict[str, Any]]:
    """
    Same one-block CLS perturbation assay as the original mu1/mu2 surface,
    but returns one row PER IMAGE and supports either PC1xPC2 or PC3xPC4.

    The four output directions are always measured, so a PC3xPC4 perturbation
    can be inspected for cross-plane modulation of the old mu1/mu2 broadcast.
    """
    if len(metadata) != pre_tbd_cpu.shape[1]:
        raise ValueError("metadata length must match image batch")
    pa, pb = int(plane[0]), int(plane[1])
    if pa not in (1, 2, 3, 4) or pb not in (1, 2, 3, 4) or pa == pb:
        raise ValueError(f"invalid rank-4 plane {plane}")

    visual = bundle.model.visual
    block = visual.transformer.resblocks[block_index]
    pre = pre_tbd_cpu.to(bundle.device, dtype=bundle.model.dtype)
    batch0 = pre.shape[1]
    tokens = pre.shape[0]
    patch_count = tokens - 1
    heads = int(block.attn.num_heads)

    basis4 = basis4_cpu.to(bundle.device).float()
    axis_a = basis4[pa - 1]
    axis_b = basis4[pb - 1]
    fixed_b13 = fixed_b13_bp_cpu.to(bundle.device).bool()
    fixed_b23 = fixed_b23_bp_cpu.to(bundle.device).bool()

    rows: list[dict[str, Any]] = []
    for gp_start in range(0, len(grid_points), args.grid_chunk):
        points = grid_points[gp_start:gp_start + args.grid_chunk]
        g = len(points)

        x = pre[:, :, None, :].expand(tokens, batch0, g, pre.shape[-1]).clone()
        aa = torch.tensor([q[0] for q in points], device=bundle.device, dtype=x.dtype)
        bb = torch.tensor([q[1] for q in points], device=bundle.device, dtype=x.dtype)
        delta = aa[:, None] * axis_a.to(x.dtype)[None, :] + bb[:, None] * axis_b.to(x.dtype)[None, :]
        x[0] = x[0] + delta[None, :, :]
        x = x.permute(0, 2, 1, 3).reshape(tokens, g * batch0, -1)

        ln1 = block.ln_1(x)
        attention_output, probs0 = block.attention(ln1, need_weights=True, capture=True)
        probs = normalize_probs(base, probs0, g * batch0, heads, tokens)
        values = normalize_values(base, block.attn.last_v, g * batch0, heads, tokens)

        source_writes = []
        for d in range(4):
            w = source_direction_write(
                probs,
                values,
                block.attn.out_proj.weight,
                basis4[d],
                source_index=0,
            )[:, 1:1 + patch_count]
            source_writes.append(w)

        plane_write = torch.sqrt(
            source_writes[pa - 1].pow(2) + source_writes[pb - 1].pow(2)
        ).mean(dim=-1)
        patch_to_cls = probs[:, :, 1:1 + patch_count, 0].mean(dim=(1, 2))

        m13 = fixed_b13[None].expand(g, batch0, patch_count).reshape(g * batch0, patch_count)
        m23 = fixed_b23[None].expand(g, batch0, patch_count).reshape(g * batch0, patch_count)
        cls_patch_attn = probs[:, :, 0, 1:1 + patch_count]
        cls_to_b13 = (cls_patch_attn * m13[:, None, :].float()).sum(dim=-1).mean(dim=1)
        cls_to_b23 = (cls_patch_attn * m23[:, None, :].float()).sum(dim=-1).mean(dim=1)

        metrics: dict[str, torch.Tensor] = {
            "patch_query_to_cls_attn_mean": patch_to_cls,
            "cls_query_to_b13reg_attn": cls_to_b13,
            "cls_query_to_b23reg_attn": cls_to_b23,
            "cls_source_plane_abs_mean": plane_write,
        }
        for d, w in enumerate(source_writes, start=1):
            metrics[f"cls_source_mu{d}_signed_mean"] = w.mean(dim=-1)
            metrics[f"cls_source_mu{d}_abs_mean"] = w.abs().mean(dim=-1)

        for local_g, (a0, b0) in enumerate(points):
            for bi in range(batch0):
                flat = local_g * batch0 + bi
                meta = dict(metadata[bi])
                row = {
                    "model_name": bundle.name,
                    "surface_type": "rta_per_image_cls_plane",
                    "block": int(block_index),
                    "pc_a": pa,
                    "pc_b": pb,
                    "plane": f"PC{pa}xPC{pb}",
                    "a_coord": float(a0),
                    "b_coord": float(b0),
                    "stim_id": str(meta.get("stim_id", "")),
                    "sample_key": str(meta.get("sample_key", "")),
                    "condition": str(meta.get("condition", "")),
                    "object_label": str(meta.get("object_label", "")),
                    "attack_word": str(meta.get("attack_word", "")),
                }
                for name, tensor in metrics.items():
                    row[name] = float(tensor[flat].float())
                rows.append(row)

        base.clear_attn_cache(block)
        del x, ln1, attention_output, probs, values, source_writes
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()

    return rows


def aggregate_rta_plane_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    metric_cols = [
        c for c in frame.columns
        if c.startswith("cls_source_") or c in {
            "patch_query_to_cls_attn_mean",
            "cls_query_to_b13reg_attn",
            "cls_query_to_b23reg_attn",
        }
    ]
    group_cols = ["model_name", "condition", "block", "pc_a", "pc_b", "plane", "a_coord", "b_coord"]
    rows = []
    for key, q in frame.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row["n_images"] = int(q["sample_key"].nunique())
        for c in metric_cols:
            row[c] = float(q[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _rta_render_frame(group: pd.DataFrame) -> pd.DataFrame:
    return group.rename(columns={"a_coord": "a_mu1", "b_coord": "b_mu2"})


def render_scalar_surface_ply_only(
    frame: pd.DataFrame,
    metric: str,
    out_dir: Path,
    stem: str,
) -> None:
    """Fast Blender-only export for the many literal paired RTA surfaces."""
    a, b, z = grid_matrix(frame, metric)
    ga, gb = np.gradient(z, a, b, edge_order=2)
    mag = np.sqrt(ga ** 2 + gb ** 2)

    from matplotlib.colors import Normalize
    cmap = plt.get_cmap(matplotlib.rcParams.get("image.cmap", "viridis"))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        return
    if abs(zmax - zmin) < 1e-12:
        zmax = zmin + 1e-12
    mmin, mmax = float(np.nanmin(mag)), float(np.nanmax(mag))
    if abs(mmax - mmin) < 1e-12:
        mmax = mmin + 1e-12
    surface_rgba = cmap(Normalize(vmin=zmin, vmax=zmax)(z))
    field_rgba = cmap(Normalize(vmin=mmin, vmax=mmax)(mag))

    z_centered = z - float(np.nanmean(z))
    z_scale = 1.0 / max(float(np.nanstd(z)), 1e-6)
    z_surface = z_centered * z_scale
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ascii_surface_ply(
        out_dir / f"{stem}_surface.ply", a, b, z_centered,
        z_scale=z_scale, rgba=surface_rgba,
    )
    write_ascii_vector_field_ply(
        out_dir / f"{stem}_field.ply", a, b, z_surface, ga, gb,
        scalar_name=metric, normalized=True, rgba=field_rgba,
    )


def render_rta_rank4_surfaces(per_image: pd.DataFrame, aggregate: pd.DataFrame, root: Path) -> None:
    if per_image.empty:
        return
    metric_specs = (
        ("cls_source_mu1_abs_mean", "mu1_broadcast", "CLS-source |mu1| patch write"),
        ("cls_source_plane_abs_mean", "plane_write", "CLS-source write magnitude in perturbed plane"),
        ("patch_query_to_cls_attn_mean", "patch_to_cls", "mean patch-query attention to CLS"),
        ("cls_query_to_b23reg_attn", "cls_to_reglineage", "CLS attention to fixed B23 register lineage"),
    )

    # Condition means: full contour/surface PNG + colored Blender PLY.
    for (model, cond, block, pa, pb), q in aggregate.groupby(
        ["model_name", "condition", "block", "pc_a", "pc_b"], sort=True
    ):
        plane = f"PC{int(pa)}xPC{int(pb)}"
        out_dir = root / str(model) / "RTA_CONDITION_MEAN" / str(cond) / f"B{int(block):02d}" / plane
        qr = _rta_render_frame(q)
        for metric, stem, label in metric_specs:
            if metric not in qr.columns:
                continue
            render_scalar_surface(
                qr,
                metric=metric,
                out_dir=out_dir,
                stem=stem,
                title=f"{model} {cond} B{int(block)} {plane}: {label}",
                x_label=rf"CLS displacement along $\mu_{{{int(pa)}}}$",
                y_label=rf"CLS displacement along $\mu_{{{int(pb)}}}$",
            )

    # Literal same-image triplets: Blender PLY only by default. This avoids
    # generating thousands of near-duplicate PNGs while preserving all geometry
    # needed to interpolate NoRTA <-> SynthRTA <-> RTA in Blender.
    for (model, sample_key, cond, block, pa, pb), q in per_image.groupby(
        ["model_name", "sample_key", "condition", "block", "pc_a", "pc_b"], sort=True
    ):
        plane = f"PC{int(pa)}xPC{int(pb)}"
        out_dir = root / str(model) / "RTA_PAIRED" / str(sample_key) / str(cond) / f"B{int(block):02d}" / plane
        qr = _rta_render_frame(q)
        for metric, stem, _label in metric_specs:
            if metric in qr.columns:
                render_scalar_surface_ply_only(qr, metric, out_dir, stem)


def plot_rta_natural_rank4(natural: pd.DataFrame, path: Path) -> None:
    """Compact condition-wise trajectories in PC1xPC2 and PC3xPC4."""
    if natural.empty or "condition" not in natural.columns:
        return
    fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(5.2 * len(MODEL_ORDER), 9.0), squeeze=False)
    for col, model in enumerate(MODEL_ORDER):
        qmodel = natural[natural["model_name"] == model]
        if qmodel.empty:
            continue
        for row_idx, (pa, pb) in enumerate(((1, 2), (3, 4))):
            ax = axes[row_idx, col]
            for cond in RTA_CONDITIONS:
                q = qmodel[qmodel["condition"] == cond]
                if q.empty:
                    continue
                g = q.groupby("block", sort=True)[[f"cls_mu{pa}_coef", f"cls_mu{pb}_coef"]].mean()
                ax.plot(g[f"cls_mu{pa}_coef"], g[f"cls_mu{pb}_coef"], marker="o", label=cond)
                for b, rr in g.iterrows():
                    if int(b) in (0, 6, 9, 12, 13, 20, 21, 22):
                        ax.text(rr[f"cls_mu{pa}_coef"], rr[f"cls_mu{pb}_coef"], str(int(b)), fontsize=7)
            ax.axhline(0, linewidth=.6)
            ax.axvline(0, linewidth=.6)
            ax.set_xlabel(rf"CLS coefficient on $\mu_{{{pa}}}$")
            ax.set_ylabel(rf"CLS coefficient on $\mu_{{{pb}}}$")
            ax.set_title(f"{model}: natural CLS PC{pa}xPC{pb}")
            ax.grid(alpha=.18)
            if row_idx == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Paired RTA natural CLS trajectories in the fixed rank-4 register-role basis")
    fig.tight_layout(rect=[0, 0, 1, .96])
    savefig(fig, path)


# =============================================================================
# Jacobians / gradients / plots
# =============================================================================

def grid_matrix(
    frame: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_vals = np.sort(frame["a_mu1"].unique().astype(float))
    b_vals = np.sort(frame["b_mu2"].unique().astype(float))
    pivot = frame.pivot(index="a_mu1", columns="b_mu2", values=metric)
    pivot = pivot.reindex(index=a_vals, columns=b_vals)
    return a_vals, b_vals, pivot.to_numpy(np.float64)


def local_jacobian_rows(cls_surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, block), group in cls_surface.groupby(
        ["model_name", "block"], sort=True
    ):
        a, b, y1 = grid_matrix(group, "cls_source_mu1_signed_mean")
        _a, _b, y2 = grid_matrix(group, "cls_source_mu2_signed_mean")

        gy1_a, gy1_b = np.gradient(y1, a, b, edge_order=2)
        gy2_a, gy2_b = np.gradient(y2, a, b, edge_order=2)

        ia = int(np.argmin(np.abs(a)))
        ib = int(np.argmin(np.abs(b)))

        J = np.array([
            [gy1_a[ia, ib], gy1_b[ia, ib]],
            [gy2_a[ia, ib], gy2_b[ia, ib]],
        ], dtype=np.float64)

        u, s, vt = np.linalg.svd(J, full_matrices=False)
        fro = float(np.linalg.norm(J))
        off = float(np.sqrt(J[0, 1] ** 2 + J[1, 0] ** 2))
        diag = float(np.sqrt(J[0, 0] ** 2 + J[1, 1] ** 2))

        rows.append({
            "model_name": model_name,
            "block": int(block),
            "J_mu1out_mu1in": float(J[0, 0]),
            "J_mu1out_mu2in": float(J[0, 1]),
            "J_mu2out_mu1in": float(J[1, 0]),
            "J_mu2out_mu2in": float(J[1, 1]),
            "J_fro_norm": fro,
            "offdiag_norm": off,
            "diag_norm": diag,
            "offdiag_to_diag_ratio": off / max(diag, EPS),
            "singular_1": float(s[0]),
            "singular_2": float(s[1]),
            "rank1_energy_fraction": float(s[0] ** 2 / max(np.sum(s ** 2), EPS)),
            "input_singular_dir1_mu1": float(vt[0, 0]),
            "input_singular_dir1_mu2": float(vt[0, 1]),
            "output_singular_dir1_mu1": float(u[0, 0]),
            "output_singular_dir1_mu2": float(u[1, 0]),
        })
    return pd.DataFrame(rows)


def probe_gradient_rows(probe_surface: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, center), group in probe_surface.groupby(
        ["model_name", "probe_center"], sort=True
    ):
        a, b, z = grid_matrix(group, "h5_cls_to_probe_attn")
        ga, gb = np.gradient(z, a, b, edge_order=2)
        ia = int(np.argmin(np.abs(a)))
        ib = int(np.argmin(np.abs(b)))
        g = np.array([ga[ia, ib], gb[ia, ib]], np.float64)
        rows.append({
            "model_name": model_name,
            "probe_center": center,
            "block": 13,
            "head": int(group["head"].iloc[0]),
            "grad_cls_to_probe_wrt_mu1": float(g[0]),
            "grad_cls_to_probe_wrt_mu2": float(g[1]),
            "gradient_norm": float(np.linalg.norm(g)),
            "gradient_angle_deg_from_plus_mu1": float(
                np.degrees(np.arctan2(g[1], g[0]))
            ),
        })
    return pd.DataFrame(rows)


def plot_natural_role_plane(natural: pd.DataFrame, path: Path) -> None:
    summary = natural.groupby(
        ["model_name", "block"], sort=True
    )[
        ["cls_mu1_cos", "cls_mu2_cos", "reg_mu1_cos", "reg_mu2_cos"]
    ].mean().reset_index()

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(16, 5.3), sharex=True, sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        z = summary[summary["model_name"] == model_name].sort_values("block")

        ax.plot(
            z["cls_mu1_cos"], z["cls_mu2_cos"],
            marker="o", label="CLS trajectory",
        )
        ax.plot(
            z["reg_mu1_cos"], z["reg_mu2_cos"],
            marker="s", label="future-register lineage",
        )

        for row in z.itertuples(index=False):
            if int(row.block) in (0, 6, 9, 10, 12, 13):
                ax.annotate(
                    f"B{int(row.block)}",
                    (float(row.cls_mu1_cos), float(row.cls_mu2_cos)),
                    fontsize=8,
                )

        ax.axhline(0, linewidth=.8)
        ax.axvline(0, linewidth=.8)
        ax.set_title(model_name)
        ax.set_xlabel(r"$\cos(x,\mu_1)$")
        ax.grid(alpha=.2)

    axes[0].set_ylabel(r"$\cos(x,\mu_2)$")
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        "Natural role geometry in the fixed register rank-2 plane\n"
        "CLS begins near mu1-orthogonal / -mu2; tracked register lineage evolves toward +mu1"
    )
    fig.tight_layout(rect=[0, 0, 1, .92])
    savefig(fig, path)


def plot_natural_role_plane_angles(natural: pd.DataFrame, path_png: Path, path_csv: Path) -> None:
    summary = natural.groupby(["model_name", "block"], sort=True)[
        ["cls_mu1_coef", "cls_mu2_coef", "reg_mu1_coef", "reg_mu2_coef"]
    ].mean().reset_index()

    rows = []
    for row in summary.itertuples(index=False):
        cls_angle = float(np.degrees(np.arctan2(float(row.cls_mu2_coef), float(row.cls_mu1_coef))))
        reg_angle = float(np.degrees(np.arctan2(float(row.reg_mu2_coef), float(row.reg_mu1_coef))))
        rows.append({
            "model_name": row.model_name,
            "block": int(row.block),
            "cls_plane_angle_deg": cls_angle,
            "reg_plane_angle_deg": reg_angle,
        })
    angle_df = pd.DataFrame(rows)
    angle_df.to_csv(path_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.6), sharex=True)
    for model_name in MODEL_ORDER:
        z = angle_df[angle_df["model_name"] == model_name].sort_values("block")
        axes[0].plot(z["block"], z["cls_plane_angle_deg"], marker='o', label=model_name)
        axes[1].plot(z["block"], z["reg_plane_angle_deg"], marker='s', label=model_name)
    axes[0].set_title('CLS plane angle over depth')
    axes[1].set_title('Register-lineage plane angle over depth')
    axes[0].set_ylabel('angle (deg)')
    axes[1].set_ylabel('angle (deg)')
    axes[1].set_xlabel('ViT block')
    for ax in axes:
        ax.axhline(0, linewidth=.8)
        ax.axhline(90, linewidth=.5, alpha=.25)
        ax.axhline(-90, linewidth=.5, alpha=.25)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Natural trajectory angles inside the fixed role plane')
    fig.tight_layout(rect=[0, 0, 1, .95])
    savefig(fig, path_png)


def render_scalar_surface(
    frame: pd.DataFrame,
    metric: str,
    out_dir: Path,
    stem: str,
    title: str,
    x_label: str = r"CLS displacement along $\mu_1$",
    y_label: str = r"CLS displacement along $\mu_2$",
) -> None:
    a, b, z = grid_matrix(frame, metric)
    ga, gb = np.gradient(z, a, b, edge_order=2)

    # 2D contour + normalized gradient arrows.
    A, B = np.meshgrid(a, b, indexing="ij")
    mag = np.sqrt(ga ** 2 + gb ** 2)
    ua = ga / np.maximum(mag, EPS)
    ub = gb / np.maximum(mag, EPS)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    contour = ax.contourf(A, B, z, levels=24)
    step = max(1, len(a) // 7)
    quiver = ax.quiver(
        A[::step, ::step],
        B[::step, ::step],
        ua[::step, ::step],
        ub[::step, ::step],
        mag[::step, ::step],
        angles="xy",
        scale_units="xy",
        scale=1.8,
    )
    # Reuse the *actual matplotlib mappings* for Blender PLY colors.
    # Surface vertices get the contour-field scalar colors; field triangles get
    # the arrow/quiver magnitude colors.  PLY red/green/blue/alpha becomes Col.
    fig.canvas.draw()
    surface_rgba = contour.cmap(contour.norm(z))
    field_rgba = quiver.cmap(quiver.norm(mag))
    ax.scatter([0], [0], marker="x", s=70, label="native")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title + "\ncontours + normalized gradient direction")
    fig.colorbar(contour, ax=ax, label=metric)
    ax.legend()
    ax.grid(alpha=.15)
    savefig(fig, out_dir / f"{stem}_contour_field.png")

    # 3D response surface.
    fig = plt.figure(figsize=(8.5, 6.5))
    ax3 = fig.add_subplot(111, projection="3d")
    surf = ax3.plot_surface(A, B, z, linewidth=0, antialiased=True)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.set_zlabel(metric)
    ax3.set_title(title)
    fig.colorbar(surf, ax=ax3, shrink=.65, pad=.1)
    savefig(fig, out_dir / f"{stem}_surface.png")

    # CSV field + PLY.
    field_rows = []
    for ia, aa in enumerate(a):
        for ib, bb in enumerate(b):
            field_rows.append({
                "a_mu1": float(aa),
                "b_mu2": float(bb),
                "scalar": float(z[ia, ib]),
                "grad_mu1": float(ga[ia, ib]),
                "grad_mu2": float(gb[ia, ib]),
                "grad_norm": float(mag[ia, ib]),
            })
    save_rows(out_dir / f"{stem}_field.csv", field_rows)

    z_centered = z - float(np.nanmean(z))
    z_scale = 1.0 / max(float(np.nanstd(z)), 1e-6)
    z_surface = z_centered * z_scale
    write_ascii_surface_ply(
        out_dir / f"{stem}_surface.ply",
        a,
        b,
        z_centered,
        z_scale=z_scale,
        rgba=surface_rgba,
    )
    write_ascii_vector_field_ply(
        out_dir / f"{stem}_field.ply",
        a,
        b,
        z_surface,
        ga,
        gb,
        scalar_name=metric,
        normalized=True,
        rgba=field_rgba,
    )


def render_cls_surfaces(cls_surface: pd.DataFrame, root: Path) -> None:
    for (model_name, block), group in cls_surface.groupby(
        ["model_name", "block"], sort=True
    ):
        out_dir = root / model_name / f"B{int(block):02d}"

        render_scalar_surface(
            group,
            metric="cls_source_mu1_abs_mean",
            out_dir=out_dir,
            stem="cls_mu1_write",
            title=f"{model_name} B{int(block)}: CLS-source |mu1| patch write",
        )
        render_scalar_surface(
            group,
            metric="cls_source_mu1_abs_l1_ratio",
            out_dir=out_dir,
            stem="cls_mu1_share",
            title=f"{model_name} B{int(block)}: CLS share of exact patch |Delta mu1|",
        )
        render_scalar_surface(
            group,
            metric="patch_query_to_cls_attn_mean",
            out_dir=out_dir,
            stem="patch_to_cls_attention",
            title=f"{model_name} B{int(block)}: mean patch-query attention to CLS",
        )

        if int(block) >= 10:
            render_scalar_surface(
                group,
                metric="cls_query_to_b13reg_attn",
                out_dir=out_dir,
                stem="cls_to_future_registers",
                title=f"{model_name} B{int(block)}: CLS attention to fixed future B13 registers",
            )


def render_probe_surfaces(probe_surface: pd.DataFrame, root: Path) -> None:
    for (model_name, center), group in probe_surface.groupby(
        ["model_name", "probe_center"], sort=True
    ):
        out_dir = root / model_name / f"B13_{center}"
        render_scalar_surface(
            group,
            metric="h5_cls_to_probe_attn",
            out_dir=out_dir,
            stem="h5_cls_to_probe",
            title=f"{model_name} B13 H5: CLS -> {center} probe",
        )
        render_scalar_surface(
            group,
            metric="h5_probe_to_b13reg_attn",
            out_dir=out_dir,
            stem="h5_probe_to_registers",
            title=f"{model_name} B13 H5: {center} probe -> fixed registers",
        )


def render_cls_move_sequences(cls_move_surface: pd.DataFrame, root: Path) -> None:
    if cls_move_surface.empty:
        return
    metric_specs = [
        ("patch_query_to_cls_attn_mean", "patch_to_cls_attention", "mean patch-query attention to CLS"),
        ("cls_query_to_b13reg_attn", "cls_to_future_registers", "CLS attention to fixed future B13 registers"),
    ]
    for (model_name, block, move_index, move_factor), group in cls_move_surface.groupby(
        ["model_name", "block", "move_index", "move_factor"],
        sort=True,
    ):
        factor_text = format_signed_factor(float(move_factor))
        for metric, stem_base, title_tail in metric_specs:
            if metric == "cls_query_to_b13reg_attn" and int(block) < 10:
                continue
            out_dir = root / model_name / f"B{int(block):02d}" / f"CLS_MOVE_{stem_base.upper()}"
            stem = f"{stem_base}_{int(move_index):02d}_{factor_text}"
            render_scalar_surface(
                group,
                metric=metric,
                out_dir=out_dir,
                stem=stem,
                title=(
                    f"{model_name} B{int(block)} move {int(move_index):02d} ({float(move_factor):+.3f} mu2): "
                    f"{title_tail}"
                ),
            )


# =============================================================================
# Report
# =============================================================================

def write_summary(
    out: Path,
    natural: pd.DataFrame,
    jac: pd.DataFrame,
    probe_grad: pd.DataFrame,
    args,
) -> None:
    nat = natural.groupby(["model_name", "block"], sort=True)[
        ["cls_mu1_cos", "cls_mu2_cos", "reg_mu1_cos", "reg_mu2_cos"]
    ].mean().reset_index()

    lines = [
        "CLS / REGISTER RANK-2 ROLE-PLANE SURFACE ATLAS",
        "=" * 76,
        "",
        "Natural geometry:",
        "",
        "model                 block   CLS cos(mu1)  CLS cos(mu2)  angle(CLS,mu2)",
        "-" * 78,
    ]

    for model_name in args.models:
        for block in (0, 6, 9, 12, 13):
            row = nat[
                (nat["model_name"] == model_name)
                & (nat["block"] == block)
            ]
            if not len(row):
                continue
            r = row.iloc[0]
            c2 = float(np.clip(r["cls_mu2_cos"], -1, 1))
            angle = float(np.degrees(np.arccos(c2)))
            lines.append(
                f"{model_name:<21} B{block:<5d} "
                f"{float(r['cls_mu1_cos']):>12.4f} "
                f"{float(r['cls_mu2_cos']):>12.4f} "
                f"{angle:>16.2f} deg"
            )

    lines += [
        "",
        "Local CLS transduction Jacobian at native point:",
        "",
        "Rows = output signed patch write [mu1,mu2]; columns = CLS input displacement [mu1,mu2].",
        "",
        "model                 block    dmu1/dmu1    dmu1/dmu2    dmu2/dmu1    dmu2/dmu2   off/diag",
        "-" * 99,
    ]
    for row in jac.itertuples(index=False):
        lines.append(
            f"{row.model_name:<21} B{int(row.block):<5d} "
            f"{row.J_mu1out_mu1in:>12.5g} "
            f"{row.J_mu1out_mu2in:>12.5g} "
            f"{row.J_mu2out_mu1in:>12.5g} "
            f"{row.J_mu2out_mu2in:>12.5g} "
            f"{row.offdiag_to_diag_ratio:>10.4f}"
        )

    lines += [
        "",
        "B13 H5 vectorial-pheromone gradient at native probe center:",
        "",
        "model                 center       d(CLS->probe)/dmu1  d(CLS->probe)/dmu2  angle",
        "-" * 91,
    ]
    for row in probe_grad.itertuples(index=False):
        lines.append(
            f"{row.model_name:<21} {row.probe_center:<12} "
            f"{row.grad_cls_to_probe_wrt_mu1:>20.6g} "
            f"{row.grad_cls_to_probe_wrt_mu2:>20.6g} "
            f"{row.gradient_angle_deg_from_plus_mu1:>8.2f}"
        )

    lines += [
        "",
        "Interpretation guardrails:",
        "  * mu1/mu2 are the fixed uncentered register basis from the prior no-RN run.",
        "  * A strong d(mu1 output)/d(mu2 input) is evidence of cross-axis transduction,",
        "    not by itself proof that mu2's semantic meaning is 'CLS polarity'.",
        "  * CLS is perturbed while patches are held fixed, so these are local response",
        "    surfaces rather than natural jointly-evolved states.",
        "  * The B13 RN/CLS-copy probe surfaces test address attractiveness in this plane;",
        "    a flat RN surface means its Siren Call K largely lives outside this rank-2 span.",
        "",
    ]

    (out / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CLS/register role-plane surfaces on paired RTA triplets, preserving "
            "the old mu1/mu2 basis and adding residual rank-3/rank-4 axes."
        )
    )

    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="v2 compatibility only; v3 builds its manifest from RTA-100-Triplet")
    parser.add_argument("--oracle_root", default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))

    # RTA is the default stimulus bank for ALL v3 extraction.
    parser.add_argument("--rta_repo", default=DEFAULT_RTA_REPO)
    parser.add_argument("--rta_split", default="train")
    parser.add_argument("--rta_triplets", type=int, default=5)
    parser.add_argument(
        "--rta_pair_ids",
        default="",
        help=(
            "Optional comma-separated canonical or subset-prefixed IDs. "
            "Example: img_0001,img_0002 or RTA_img_0001,NoRTA_img_0002. "
            "If omitted, the first --rta_triplets shared IDs are used."
        ),
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grid", type=int, default=17)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--grid_chunk", type=int, default=8)
    parser.add_argument("--anchor_blocks", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23")
    parser.add_argument("--probe_head", type=int, default=5)
    parser.add_argument("--cls_move_blocks", default="10,12,13,16,18,19,20,21,22")
    parser.add_argument("--cls_move_steps", type=int, default=10)
    parser.add_argument("--cls_move_span", type=float, default=6.0)

    # Supplemental literal same-image triplet atlas.  PC1xPC2 is the old role
    # plane; PC3xPC4 is the new residual rank-4 extension.
    parser.add_argument("--rta_compare_blocks", default="5,6,7,8,9,10,11,12,13,16,18,20,21,22")
    parser.add_argument("--rta_compare_planes", default="1x2,3x4")
    parser.add_argument("--rta_compare_grid", type=int, default=17)
    parser.add_argument("--rta_compare_radius", type=float, default=6.0)

    # Fresh RTA-image masks; basis itself remains the previous fixed oracle.
    parser.add_argument("--final_register_threshold", type=float, default=60.0)
    parser.add_argument("--final_register_min", type=int, default=1)
    parser.add_argument("--final_register_max", type=int, default=4)
    parser.add_argument("--b13_register_threshold", type=float, default=60.0)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip_module", default="attnclip_mechinterp_sae")
    parser.add_argument("--model_spec", default="ViT-L/14")
    parser.add_argument(
        "--gmp_checkpoint",
        default=r"ViT-L-14-BEST-smooth-GmP-ft-pickle-OpenAI.pt",
    )
    parser.add_argument("--xattn_checkpoint", default=DEFAULT_XATTN_CHECKPOINT)
    parser.add_argument("--xattn_module", default="oaiclip")
    parser.add_argument("--pickle_module", default="clip")

    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--postprocess_only",
        action="store_true",
        help="Re-render existing CSV surfaces (including Col-colored PLY) without loading CLIP.",
    )

    args = parser.parse_args()
    args.models = tuple(token.strip() for token in args.models.split(",") if token.strip())
    args.anchor_blocks = parse_ints(args.anchor_blocks)
    args.cls_move_blocks = parse_ints(args.cls_move_blocks)
    args.rta_compare_blocks = parse_ints(args.rta_compare_blocks)
    try:
        args.rta_compare_planes = parse_planes4(args.rta_compare_planes)
    except ValueError as exc:
        parser.error(str(exc))

    if args.grid < 5 or args.grid % 2 == 0:
        parser.error("--grid must be odd and >=5 so the native (0,0) point exists")
    if args.rta_compare_grid < 5 or args.rta_compare_grid % 2 == 0:
        parser.error("--rta_compare_grid must be odd and >=5")
    if args.radius <= 0 or args.rta_compare_radius <= 0:
        parser.error("surface radii must be positive")
    if args.rta_triplets <= 0 and not args.rta_pair_ids:
        parser.error("--rta_triplets must be positive when --rta_pair_ids is empty")
    unknown = set(args.models) - set(MODEL_ORDER)
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")
    return args


def postprocess(out: Path, args) -> None:
    natural_path = out / "natural_role_plane.csv"
    cls_path = out / "cls_surface_points.csv"
    cls_move_path = out / "cls_move_surface_points.csv"
    probe_path = out / "b13_probe_surface_points.csv"
    rta_per_image_path = out / "rta_rank4_surface_points_per_image.csv"
    rta_mean_path = out / "rta_rank4_surface_points_condition_mean.csv"

    missing = [
        str(path)
        for path in (natural_path, cls_path, probe_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "--postprocess_only requires completed CSVs. Missing:\n  "
            + "\n  ".join(missing)
        )

    natural = pd.read_csv(natural_path)
    cls_surface = pd.read_csv(cls_path)
    probe_surface = pd.read_csv(probe_path)
    cls_move_surface = pd.read_csv(cls_move_path) if cls_move_path.is_file() else pd.DataFrame()
    rta_per_image = pd.read_csv(rta_per_image_path) if rta_per_image_path.is_file() else pd.DataFrame()
    rta_mean = pd.read_csv(rta_mean_path) if rta_mean_path.is_file() else pd.DataFrame()

    jac = local_jacobian_rows(cls_surface)
    probe_grad = probe_gradient_rows(probe_surface)
    jac.to_csv(out / "cls_local_jacobians.csv", index=False)
    probe_grad.to_csv(out / "b13_probe_local_gradients.csv", index=False)

    plot_natural_role_plane(natural, out / "natural_role_plane.png")
    plot_natural_role_plane_angles(
        natural,
        out / "natural_role_plane_angles.png",
        out / "natural_role_plane_angles.csv",
    )
    plot_rta_natural_rank4(natural, out / "natural_role_plane_rank4_by_rta_condition.png")

    render_cls_surfaces(cls_surface, out / "surfaces")
    if not cls_move_surface.empty:
        render_cls_move_sequences(cls_move_surface, out / "surfaces")
    render_probe_surfaces(probe_surface, out / "surfaces")
    if not rta_per_image.empty and not rta_mean.empty:
        render_rta_rank4_surfaces(rta_per_image, rta_mean, out / "surfaces")

    write_summary(out, natural, jac, probe_grad, args)

    # Append the v3-specific audit/interpretation to the human-readable summary.
    with (out / "SUMMARY.txt").open("a", encoding="utf-8") as handle:
        handle.write("\nRTA / rank-4 v3 additions:\n")
        handle.write("  * All extraction stimuli are paired NoRTA/SynthRTA/RTA images from RTA-100-Triplet.\n")
        handle.write("  * mu1/mu2 are preserved EXACTLY from the previous no-RN oracle.\n")
        handle.write("  * mu3/mu4 are residual uncentered SVD directions fit on the SAME previous register-mean population after removing span(mu1,mu2).\n")
        handle.write("  * B13/B23 masks are recomputed on the current RTA images, then held fixed over every local surface.\n")
        handle.write("  * Colored PLY uses red/green/blue/alpha properties; Blender imports these as the vertex color attribute Col.\n")
        handle.write("  * RTA_PAIRED directories contain literal same-image per-condition PLYs for Blender interpolation.\n")

    include = [
        out / "config.json",
        out / "rn_token_audit.json",
        out / "rta_manifest.csv",
        out / "rank4_basis_audit.csv",
        natural_path,
        out / "natural_role_plane.png",
        out / "natural_role_plane_angles.csv",
        out / "natural_role_plane_angles.png",
        out / "natural_role_plane_rank4_by_rta_condition.png",
        cls_path,
        cls_move_path,
        out / "cls_local_jacobians.csv",
        probe_path,
        out / "b13_probe_local_gradients.csv",
        rta_per_image_path,
        rta_mean_path,
        out / "SUMMARY.txt",
    ]
    include += sorted(out.glob("*/rta_role_oracle.npz"))
    include += sorted((out / "surfaces").rglob("*.png"))
    include += sorted((out / "surfaces").rglob("*.csv"))
    include += sorted((out / "surfaces").rglob("*.ply"))

    zpath = out / "compact_summary_workspace_cls_role_surfaces.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(
        zpath,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=7,
    ) as archive:
        for path in include:
            if path.is_file():
                archive.write(path, arcname=path.relative_to(out).as_posix())

    print("[postprocess] done")
    print("[compact summary]", zpath)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.postprocess_only:
        postprocess(out, args)
        return

    base = _backbone_tools
    manifest = build_rta_manifest(args, out)
    print("[RTA manifest]", len(manifest), "images /", manifest["sample_key"].nunique(), "paired triplets")
    print(manifest[["stim_id", "sample_key", "condition", "object_label", "attack_word"]].to_string(index=False))

    save_json(out / "config.json", vars(args))

    rn_token, rn_audit = load_trained_rn_token(base, args)
    save_json(out / "rn_token_audit.json", rn_audit)
    print(f"[RN] norm={rn_audit['norm']:.6f}")

    grid_axis = np.linspace(-args.radius, args.radius, args.grid, dtype=np.float32)
    grid_points = [(float(a), float(b)) for a in grid_axis for b in grid_axis]
    rta_axis = np.linspace(
        -args.rta_compare_radius,
        args.rta_compare_radius,
        args.rta_compare_grid,
        dtype=np.float32,
    )
    rta_grid_points = [(float(a), float(b)) for a in rta_axis for b in rta_axis]

    natural_frames = []
    cls_rows_all: list[dict[str, Any]] = []
    cls_move_rows_all: list[dict[str, Any]] = []
    probe_rows_all: list[dict[str, Any]] = []
    rta_rank4_rows_all: list[dict[str, Any]] = []
    basis_audit_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        print(f"\n================ {model_name} ================")
        model_dir = out / model_name
        model_dir.mkdir(exist_ok=True)

        bundle = base.load_bundle(model_name, args, model_dir / "load_audit")
        bundle.name = model_name
        visual = bundle.model.visual
        width = int(visual.positional_embedding.shape[1])

        if rn_token.numel() != width:
            raise RuntimeError(f"RN width {rn_token.numel()} != visual width {width}")

        oracle = build_rta_oracle(bundle, manifest, args, model_dir)
        print(f"[fixed basis source] {oracle.source_path}")
        G = oracle.basis4 @ oracle.basis4.T
        for i in range(4):
            basis_audit_rows.append({
                "model_name": model_name,
                "axis": f"mu{i+1}",
                "norm": float(np.linalg.norm(oracle.basis4[i])),
                "max_abs_cos_other_axis": float(np.max(np.abs(np.delete(G[i], i)))),
                "source_oracle": oracle.source_path,
            })

        states, natural = collect_baseline_states(bundle, manifest, oracle, args)
        natural_frames.append(natural)

        mu1_cpu = torch.from_numpy(oracle.mu1).float()
        mu2_cpu = torch.from_numpy(oracle.mu2).float()
        basis4_cpu = torch.from_numpy(oracle.basis4).float()

        # Existing mu1/mu2 surfaces, now evaluated on the paired RTA bank.
        for block_index in args.anchor_blocks:
            print(f"[surface] {model_name} B{block_index} CLS mu1/mu2 plane on RTA bank")
            block_batches = states[block_index]
            model_block_rows: list[dict[str, Any]] = []
            for batch_index, pre_cpu in enumerate(block_batches):
                start = batch_index * args.batch_size
                batch = pre_cpu.shape[1]
                fixed_b13_cpu = torch.from_numpy(oracle.b13_reg_mask[start:start + batch].astype(bool))
                model_block_rows.extend(
                    evaluate_cls_surface_batch(
                        base=base,
                        bundle=bundle,
                        block_index=block_index,
                        pre_tbd_cpu=pre_cpu,
                        fixed_b13_bp_cpu=fixed_b13_cpu,
                        mu1_cpu=mu1_cpu,
                        mu2_cpu=mu2_cpu,
                        grid_points=grid_points,
                        args=args,
                    )
                )
            aggregated = aggregate_surface_batches(model_block_rows)
            cls_rows_all.extend(aggregated.to_dict("records"))

        # Existing +mu2 moved-center animation surfaces, still mu1/mu2 and now RTA-based.
        if args.cls_move_steps > 0 and args.cls_move_blocks:
            move_factors = np.linspace(0.0, args.cls_move_span, args.cls_move_steps, dtype=np.float32)
            for block_index in args.cls_move_blocks:
                if block_index not in states:
                    continue
                print(f"[surface] {model_name} B{block_index} CLS +mu2 move sequence")
                block_batches = states[block_index]
                for move_index, move_factor in enumerate(move_factors):
                    model_move_rows: list[dict[str, Any]] = []
                    for batch_index, pre_cpu in enumerate(block_batches):
                        start = batch_index * args.batch_size
                        batch = pre_cpu.shape[1]
                        fixed_b13_cpu = torch.from_numpy(oracle.b13_reg_mask[start:start + batch].astype(bool))
                        model_move_rows.extend(
                            evaluate_cls_surface_batch(
                                base=base,
                                bundle=bundle,
                                block_index=block_index,
                                pre_tbd_cpu=pre_cpu,
                                fixed_b13_bp_cpu=fixed_b13_cpu,
                                mu1_cpu=mu1_cpu,
                                mu2_cpu=mu2_cpu,
                                grid_points=grid_points,
                                args=args,
                                center_a_mu1=0.0,
                                center_b_mu2=float(move_factor),
                                extra_fields={
                                    "surface_type": "cls_move",
                                    "move_family": "plus_mu2",
                                    "move_index": int(move_index),
                                    "move_factor": float(move_factor),
                                },
                            )
                        )
                    aggregated_move = aggregate_surface_batches(model_move_rows)
                    cls_move_rows_all.extend(aggregated_move.to_dict("records"))

        # Existing B13 RN / CLS-copy pheromone surfaces, now on paired RTA images.
        pre_b13_batches = states[13]
        for center in ("RN", "CLS_COPY"):
            print(f"[surface] {model_name} B13 H{args.probe_head} {center} probe")
            center_rows: list[dict[str, Any]] = []
            for batch_index, pre_cpu in enumerate(pre_b13_batches):
                start = batch_index * args.batch_size
                batch = pre_cpu.shape[1]
                fixed_b13_cpu = torch.from_numpy(oracle.b13_reg_mask[start:start + batch].astype(bool))
                center_rows.extend(
                    evaluate_b13_probe_surface_batch(
                        base=base,
                        bundle=bundle,
                        pre_b13_tbd_cpu=pre_cpu,
                        fixed_b13_bp_cpu=fixed_b13_cpu,
                        rn_token_cpu=rn_token,
                        mu1_cpu=mu1_cpu,
                        mu2_cpu=mu2_cpu,
                        grid_points=grid_points,
                        center=center,
                        args=args,
                    )
                )
            aggregated_probe = aggregate_probe_batches(center_rows)
            probe_rows_all.extend(aggregated_probe.to_dict("records"))

        # New literal paired-image rank-4 atlas. Same sample_key, three variants,
        # same fixed basis, same fixed per-image register lineage over each surface.
        for block_index in args.rta_compare_blocks:
            if block_index not in states:
                raise RuntimeError(f"RTA compare block B{block_index} was not captured")
            print(f"[RTA rank4] {model_name} B{block_index}")
            block_batches = states[block_index]
            for batch_index, pre_cpu in enumerate(block_batches):
                start = batch_index * args.batch_size
                batch = pre_cpu.shape[1]
                chunk = manifest.iloc[start:start + batch]
                metadata = chunk[[
                    "stim_id", "sample_key", "condition", "object_label", "attack_word"
                ]].to_dict("records")
                fixed_b13_cpu = torch.from_numpy(oracle.b13_reg_mask[start:start + batch].astype(bool))
                fixed_b23_cpu = torch.from_numpy(oracle.b23_reg_mask[start:start + batch].astype(bool))
                for plane in args.rta_compare_planes:
                    rta_rank4_rows_all.extend(
                        evaluate_rta_cls_plane_batch_per_image(
                            base=base,
                            bundle=bundle,
                            block_index=block_index,
                            pre_tbd_cpu=pre_cpu,
                            fixed_b13_bp_cpu=fixed_b13_cpu,
                            fixed_b23_bp_cpu=fixed_b23_cpu,
                            basis4_cpu=basis4_cpu,
                            plane=plane,
                            grid_points=rta_grid_points,
                            metadata=metadata,
                            args=args,
                        )
                    )

        del bundle, states, oracle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    natural_all = pd.concat(natural_frames, ignore_index=True)
    cls_surface_all = pd.DataFrame(cls_rows_all)
    probe_surface_all = pd.DataFrame(probe_rows_all)
    cls_move_surface_all = pd.DataFrame(cls_move_rows_all)
    rta_per_image_all = pd.DataFrame(rta_rank4_rows_all)
    rta_mean_all = aggregate_rta_plane_rows(rta_per_image_all)

    natural_all.to_csv(out / "natural_role_plane.csv", index=False)
    cls_surface_all.to_csv(out / "cls_surface_points.csv", index=False)
    cls_move_surface_all.to_csv(out / "cls_move_surface_points.csv", index=False)
    probe_surface_all.to_csv(out / "b13_probe_surface_points.csv", index=False)
    rta_per_image_all.to_csv(out / "rta_rank4_surface_points_per_image.csv", index=False)
    rta_mean_all.to_csv(out / "rta_rank4_surface_points_condition_mean.csv", index=False)
    pd.DataFrame(basis_audit_rows).to_csv(out / "rank4_basis_audit.csv", index=False)

    # Expensive extraction is safely on disk before the large Blender render pass.
    postprocess(out, args)


if __name__ == "__main__":
    main()
