#!/usr/bin/env python3
r"""RTA register geometry. Commands: secondary (paired register subspaces), grad_attention (late attribution), principal_angles (reuse saved register means).
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_analysis import (parse_strs, unit_rows)
from probe_tools_backbone import safe_torch_load, extract_state_dict

# SECONDARY
import argparse
import gc
import importlib
import json
import math
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from datasets import load_dataset
from tqdm.auto import tqdm


DEFAULT_GMP = r"GMP_CHECKPOINT.pt"
DEFAULT_XATTN = "REPLACE_WITH_CHECKPOINT.pt"
DEFAULT_GMP_MODULE = "attnclip_mechinterp_sae"
DEFAULT_DATASET = "zer0int/RTA-100-Triplet"
SECONDARY_DEFAULT_OUTPUT = r"out_rta_register_secondary_mode"
SECONDARY_DEFAULT_BLOCKS = "8-23"
DEFAULT_SEED = 20260908

MODEL_ORDER = ("gmp", "xattn_stripped")
CONDITION_ORDER = ("NoRTA", "RTA", "SynthRTA")
ATTACK_ORDER = ("RTA", "SynthRTA")


# =============================================================================
# Generic helpers
# =============================================================================

def secondary_parse_blocks(text: str) -> list[int]:
    out: list[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(tok))
    return sorted(set(out))


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > eps else float("nan")


def row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def mean_offdiag_cosine(x: np.ndarray) -> float:
    """
    O(ND), no NxN matrix needed.
    """
    y = row_normalize(x)
    n = len(y)
    if n < 2:
        return float("nan")
    s = y.sum(axis=0)
    return float((s @ s - n) / (n * (n - 1)))


def resultant_length(x: np.ndarray) -> float:
    y = row_normalize(x)
    if len(y) == 0:
        return float("nan")
    return float(np.linalg.norm(y.sum(axis=0)) / len(y))


def residual_remove(x: np.ndarray, basis_rd: np.ndarray, rank: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    u = np.asarray(basis_rd[:rank], np.float64)
    if rank <= 0:
        return x.copy()
    return x - (x @ u.T) @ u


def orient_basis(basis: np.ndarray, x: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis, np.float64).copy()
    mean = np.asarray(x, np.float64).mean(axis=0)
    for i in range(len(basis)):
        if float(basis[i] @ mean) < 0:
            basis[i] *= -1.0
    return basis


def secondary_fit_uncentered_basis(x: np.ndarray, rank: int = 3) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Exact top right-singular vectors via the smaller Gram matrix when possible.
    Returns [rank,D], singular values, total squared Frobenius energy.
    """
    x = np.asarray(x, np.float64)
    n, d = x.shape
    k = min(int(rank), n, d)
    if k < 1:
        raise ValueError(f"Cannot fit basis to shape {x.shape}")

    total = float(np.sum(x * x))
    if n <= d:
        gram = x @ x.T
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1][:k]
        evals = np.clip(evals[order], 0.0, None)
        left = evecs[:, order]
        sing = np.sqrt(evals)
        basis = []
        for j in range(k):
            if sing[j] <= 1e-12:
                v = np.zeros(d, np.float64)
            else:
                v = (left[:, j].T @ x) / sing[j]
            basis.append(v)
        basis = row_normalize(np.stack(basis, axis=0))
    else:
        gram = x.T @ x
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1][:k]
        evals = np.clip(evals[order], 0.0, None)
        sing = np.sqrt(evals)
        basis = row_normalize(evecs[:, order].T)

    basis = orient_basis(basis, x)
    return basis, sing, total


def bootstrap_mean_ci(
    vals: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    x = np.asarray(vals, np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"),) * 4
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    if len(x) == 1 or n_boot <= 0:
        return mean, sem, mean, mean
    rng = np.random.default_rng(seed)
    # Chunk bootstrap to avoid huge temporary allocations.
    boots = np.empty(n_boot, np.float64)
    chunk = 1000
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, len(x), size=(m, len(x)))
        boots[done:done+m] = x[idx].mean(axis=1)
        done += m
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return mean, sem, float(lo), float(hi)


def signflip_pvalue(vals: Sequence[float], *, n_perm: int, seed: int) -> float:
    x = np.asarray(vals, np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    obs = abs(float(x.mean()))
    if obs <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    ge = 1
    total = 1
    chunk = 2000
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, len(x)))
        means = (signs * x[None, :]).mean(axis=1)
        ge += int(np.sum(np.abs(means) >= obs))
        total += m
        done += m
    return float(ge / total)


def stable_fold_assignments(n: int, k: int, seed: int) -> np.ndarray:
    k = min(max(2, int(k)), n)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.empty(n, np.int64)
    for j, idx in enumerate(perm):
        folds[idx] = j % k
    return folds


# =============================================================================
# Model loading / bare visual transplant
# =============================================================================

def load_gmp_model(module_name: str, spec: str):
    mod = importlib.import_module(module_name)
    load_fn = getattr(mod, "load")
    try:
        model, preprocess = load_fn(spec, device="cpu", jit=False, read_null_enabled=False)
    except TypeError:
        model, preprocess = load_fn(spec, device="cpu", jit=False)
    return model.float().eval(), preprocess


def _resolve_qkv_target(
    key: str,
    src: Mapping[str, torch.Tensor],
) -> torch.Tensor | None:
    # target packed -> source split
    if key.endswith(".attn.in_proj_weight"):
        p = key[:-len("in_proj_weight")]
        qs = [p + f"{q}_proj.weight" for q in ("q", "k", "v")]
        if all(q in src for q in qs):
            return torch.cat([src[q] for q in qs], dim=0)
    if key.endswith(".attn.in_proj_bias"):
        p = key[:-len("in_proj_bias")]
        qs = [p + f"{q}_proj.bias" for q in ("q", "k", "v")]
        if all(q in src for q in qs):
            return torch.cat([src[q] for q in qs], dim=0)

    # target split -> source packed
    m = re.search(r"^(.*\.attn\.)([qkv])_proj\.(weight|bias)$", key)
    if m:
        p, which, kind = m.groups()
        packed_key = p + ("in_proj_weight" if kind == "weight" else "in_proj_bias")
        if packed_key in src:
            packed = src[packed_key]
            chunks = packed.chunk(3, dim=0)
            return chunks[{"q": 0, "k": 1, "v": 2}[which]]
    return None


def _source_visual_state(
    source: torch.nn.Module | Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a visual-tower state dict with ``visual.`` stripped when present.

    ``source`` may be an instantiated visual module or a full CLIP state dict.
    Register-geometry stripping deliberately supports the latter so an x-attn
    donor can be used strictly as a weight source without instantiating RN, READ,
    bridge, router, or any other custom runtime machinery.
    """
    if isinstance(source, torch.nn.Module):
        return {str(k): v for k, v in source.state_dict().items()}
    raw = {str(k): v for k, v in source.items() if torch.is_tensor(v)}
    visual = {k[len("visual."):]: v for k, v in raw.items() if k.startswith("visual.")}
    return visual if visual else raw


def load_xattn_weight_source(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    """Load and validate a full trained x-attn checkpoint as inert tensor data.

    No model class is instantiated here.  The source must contain explicit x-attn
    donor evidence; otherwise a vanilla checkpoint could silently masquerade as
    the trained visual source used by this comparison.
    """
    path = Path(checkpoint).expanduser()
    obj = safe_torch_load(path)
    state = extract_state_dict(obj, source=str(path))
    keys = set(state)
    required_markers = (
        "hard_text_embedding",
        "read_implant.read_bridge.q_proj.weight",
        "visual.read_null_token",
    )
    missing = [k for k in required_markers if k not in keys]
    if missing:
        raise RuntimeError(
            "Expected a full trained x-attn donor checkpoint, but required donor "
            f"markers are missing: {missing}. Source={path}"
        )
    return state


def transplant_visual_state(
    target_visual: torch.nn.Module,
    source_visual: torch.nn.Module | Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """
    Copy every ordinary target visual parameter from trained x-attn visual weights.
    The source may be a visual module or a full checkpoint state dict.  Custom RN
    and bridge keys are source-only and therefore can never enter the clean target
    architecture.  If the clean module exposes an inert read_null parameter, it is
    left untouched and manual_bare_visual_forward never uses it.
    """
    dst = target_visual.state_dict()
    src = _source_visual_state(source_visual)
    load: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatched: list[str] = []

    def custom_target_key(k: str) -> bool:
        kl = k.lower()
        return "read_null" in kl or "register_token" in kl

    for k, dv in dst.items():
        if custom_target_key(k):
            rows.append({"key": k, "status": "target_custom_inert_not_loaded"})
            load[k] = dv
            continue
        sv = src.get(k)
        source_kind = "exact"
        if sv is None:
            sv = _resolve_qkv_target(k, src)
            source_kind = "qkv_converted"
        if sv is None:
            missing.append(k)
            rows.append({"key": k, "status": "missing_source"})
            continue
        if tuple(sv.shape) != tuple(dv.shape):
            mismatched.append(k)
            rows.append({
                "key": k, "status": "shape_mismatch",
                "source_shape": list(sv.shape), "target_shape": list(dv.shape),
            })
            continue
        load[k] = sv.detach().cpu().to(dtype=dv.dtype)
        # Parameter change magnitude audit.
        a = dv.detach().cpu().float().reshape(-1)
        b = sv.detach().cpu().float().reshape(-1)
        denom = float(a.norm() * b.norm())
        rows.append({
            "key": k,
            "status": "loaded",
            "source_kind": source_kind,
            "mean_abs_delta": float((a - b).abs().mean()),
            "cosine_before_after": float((a @ b) / denom) if denom > 0 else float("nan"),
        })

    if missing or mismatched:
        raise RuntimeError(
            "Could not construct clean x-attn-trained visual tower. "
            f"missing={missing[:12]} mismatched={mismatched[:12]}"
        )
    target_visual.load_state_dict(load, strict=True)
    return rows


class BareVisual:
    """
    Manual vanilla CLIP ViT path.  It never calls any RN/bridge/custom forward.
    """
    def __init__(self, visual: torch.nn.Module):
        self.visual = visual

    @torch.no_grad()
    def capture_patch_states(
        self,
        images_bchw: torch.Tensor,
        blocks: Sequence[int],
        *,
        amp: bool = False,
    ) -> dict[int, torch.Tensor]:
        v = self.visual
        requested = set(map(int, blocks))
        max_block = max(requested)

        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp and images_bchw.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with ctx:
            x = v.conv1(images_bchw)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            cls = v.class_embedding.to(x.dtype)
            cls = cls + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls, x], dim=1)
            x = x + v.positional_embedding.to(x.dtype)
            x = v.ln_pre(x)
            x = x.permute(1, 0, 2)  # [T,B,D]

            out: dict[int, torch.Tensor] = {}
            for b, block in enumerate(v.transformer.resblocks):
                x = block(x)
                if b in requested:
                    # Spatial patches only, CPU float32.
                    out[b] = x[1:, :, :].permute(1, 0, 2).detach().float().cpu()
                if b >= max_block:
                    break
        return out


# =============================================================================
# Dataset / pairing
# =============================================================================

def load_rta_triplet_subsets(
    repo: str,
    split: str,
    subset_names: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    """
    zer0int/RTA-100-Triplet is a single HF dataset/config.  NoRTA/RTA/SynthRTA
    are values in the row-level `type` column, NOT HF configs.

    This intentionally mirrors the standard benchmark loader:
        load_dataset(repo, split="train")
        subset = row["type"]
    """
    ds = load_dataset(repo, split=split)
    required = {"type", "image", "id"}
    missing = sorted(required - set(ds.column_names))
    if missing:
        raise KeyError(
            f"{repo} split={split!r} is missing required column(s) {missing}; "
            f"columns={ds.column_names}"
        )

    wanted = set(map(str, subset_names))
    rows_by_subset: dict[str, list[int]] = {name: [] for name in subset_names}
    for i, typ in enumerate(ds["type"]):
        typ = str(typ)
        if typ in wanted:
            rows_by_subset[typ].append(i)

    empty = [name for name, idx in rows_by_subset.items() if not idx]
    if empty:
        seen = sorted(set(map(str, ds["type"])))
        raise RuntimeError(
            f"Missing requested row subset(s) {empty}; values observed in `type`={seen}"
        )

    # select() preserves the original row contents and is cheap because HF Dataset
    # keeps Arrow-backed indices rather than eagerly duplicating images.
    subsets = {
        name: ds.select(rows_by_subset[name])
        for name in subset_names
    }
    return "default", subsets


PAIR_KEY_CANDIDATES = (
    "id", "sample_id", "image_id", "key", "uid", "source_id",
    "original_index", "index", "idx", "filename", "file_name", "path",
)


def infer_pair_key(dsets: Mapping[str, Any], requested: str) -> str:
    if requested != "auto":
        for name, ds in dsets.items():
            if requested not in ds.column_names:
                raise KeyError(f"pair key {requested!r} missing from {name}: {ds.column_names}")
        return requested

    common = set.intersection(*(set(ds.column_names) for ds in dsets.values()))
    for key in PAIR_KEY_CANDIDATES:
        if key not in common:
            continue
        ok = True
        for ds in dsets.values():
            vals = [str(x) for x in ds[key]]
            if len(vals) != len(set(vals)):
                ok = False
                break
        if ok:
            return key
    return "__row_index__"


def infer_image_column(ds) -> str:
    for k in ("image", "img", "photo"):
        if k in ds.column_names:
            return k
    raise KeyError(f"No image column found; columns={ds.column_names}")


def row_image(row: Mapping[str, Any], image_col: str) -> Image.Image:
    im = row[image_col]
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if isinstance(im, dict):
        if im.get("path"):
            return Image.open(im["path"]).convert("RGB")
        if im.get("bytes"):
            import io
            return Image.open(io.BytesIO(im["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image object: {type(im)}")


def canonical_rta_pair_id(raw: Any, variant: str) -> str:
    """
    RTA-100-Triplet IDs are variant-prefixed (e.g. NoRTA_..., RTA_..., SynthRTA_...).
    Strip ONLY the known row variant prefix so the three members of a triplet
    share one canonical identifier.

    This mirrors the pairing policy already used in probe_rn_test_arena.py.
    """
    value = str(raw).strip()
    if not value:
        return ""
    pattern = rf"^{re.escape(str(variant))}[\s_:\-./]*"
    stripped = re.sub(pattern, "", value, count=1, flags=re.IGNORECASE)
    return stripped if stripped else value


def _validate_triplet_rows(
    dsets: Mapping[str, Any],
    index_map: Mapping[str, int],
    *,
    context: str,
) -> None:
    """
    Pairing sanity check independent of IDs: all three variants must describe
    the same object and attack word.
    """
    obj = {}
    atk = {}
    for cond in CONDITION_ORDER:
        row = dsets[cond][int(index_map[cond])]
        obj[cond] = str(row.get("object_label", "")).strip()
        atk[cond] = str(row.get("attack_word", "")).strip()

    if len(set(obj.values())) != 1:
        raise RuntimeError(f"Triplet {context}: object_label mismatch: {obj}")
    if len(set(atk.values())) != 1:
        raise RuntimeError(f"Triplet {context}: attack_word mismatch: {atk}")


def build_paired_rows(
    dsets: Mapping[str, Any],
    pair_key: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Safely pair NoRTA/RTA/SynthRTA.

    Preferred path for RTA-100-Triplet:
      canonicalized `id` after stripping each subset's own prefix.

    Fallback:
      if canonical-ID overlap is unexpectedly weak but all subsets have equal
      length, accept dataset order ONLY after validating object_label and
      attack_word for every triplet.  This reproduces the conservative fallback
      used by the existing RN arena scripts.
    """
    if pair_key == "__row_index__":
        lens = {k: len(v) for k, v in dsets.items()}
        if len(set(lens.values())) != 1:
            raise RuntimeError(f"Cannot pair by row index; unequal lengths: {lens}")
        rows = []
        n = next(iter(lens.values()))
        for i in range(n):
            idx = {cond: i for cond in CONDITION_ORDER}
            _validate_triplet_rows(dsets, idx, context=f"ordered_{i}")
            row = {"sample_key": f"ordered_{i:06d}", "pairing_method": "validated_order"}
            for cond in CONDITION_ORDER:
                row[f"{cond}_index"] = i
            rows.append(row)
    else:
        maps: dict[str, dict[str, int]] = {}
        duplicate_keys: dict[str, list[str]] = {}
        for cond, ds in dsets.items():
            m: dict[str, int] = {}
            dups: list[str] = []
            for i in range(len(ds)):
                raw = ds[i][pair_key]
                key = canonical_rta_pair_id(raw, cond) if pair_key == "id" else str(raw)
                if not key:
                    continue
                if key in m:
                    dups.append(key)
                else:
                    m[key] = i
            maps[cond] = m
            duplicate_keys[cond] = sorted(set(dups))

        if any(duplicate_keys.values()):
            raise RuntimeError(
                f"Duplicate canonical pairing keys encountered: "
                f"{ {k:v[:12] for k,v in duplicate_keys.items() if v} }"
            )

        shared = set.intersection(*(set(m.keys()) for m in maps.values()))
        expected = min(len(ds) for ds in dsets.values())

        if len(shared) >= max(1, int(0.95 * expected)):
            keys = sorted(shared)
            rows = []
            for key in keys:
                idx = {cond: int(maps[cond][key]) for cond in CONDITION_ORDER}
                _validate_triplet_rows(dsets, idx, context=f"id:{key}")
                row = {"sample_key": key, "pairing_method": "canonical_id"}
                for cond in CONDITION_ORDER:
                    row[f"{cond}_index"] = idx[cond]
                rows.append(row)
        else:
            # Conservative fallback already used in the user's RN arena code:
            # equal length + label-validated order.
            lens = {cond: len(ds) for cond, ds in dsets.items()}
            if len(set(lens.values())) != 1:
                raise RuntimeError(
                    "Could not safely pair RTA triplets: canonical-ID overlap "
                    f"{len(shared)}/{expected}, unequal subset lengths={lens}"
                )
            n = next(iter(lens.values()))
            rows = []
            for i in range(n):
                idx = {cond: i for cond in CONDITION_ORDER}
                _validate_triplet_rows(dsets, idx, context=f"ordered_{i}")
                row = {
                    "sample_key": f"ordered_{i:06d}",
                    "pairing_method": "validated_order_fallback",
                }
                for cond in CONDITION_ORDER:
                    row[f"{cond}_index"] = i
                rows.append(row)
            print(
                f"[pairing] canonical ID overlap weak ({len(shared)}/{expected}); "
                f"using fully label-validated dataset order ({n} triplets)"
            )

    if limit > 0 and len(rows) > limit:
        rng = random.Random(seed)
        rows = sorted(rng.sample(rows, limit), key=lambda r: str(r["sample_key"]))

    return rows


# =============================================================================
# Register extraction
# =============================================================================

def select_register_indices(
    final_norms_bp: torch.Tensor,
    *,
    threshold: float,
    min_registers: int,
    max_registers: int,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    arr = final_norms_bp.detach().cpu().numpy()
    for norms in arr:
        idx = np.where(norms >= threshold)[0]
        if len(idx) > max_registers:
            order = idx[np.argsort(norms[idx])[::-1]]
            idx = order[:max_registers]
        if len(idx) < min_registers:
            order = np.argsort(norms)[::-1]
            idx = order[:min_registers]
        out.append(np.sort(idx.astype(np.int64)))
    return out


def mask_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(map(int, a)), set(map(int, b))
    den = len(sa | sb)
    return float(len(sa & sb) / den) if den else 1.0


def means_at_indices(states: dict[int, torch.Tensor], indices: list[np.ndarray]) -> dict[int, np.ndarray]:
    out = {}
    for b, x_bpd in states.items():
        rows = []
        for i, idx in enumerate(indices):
            rows.append(x_bpd[i, torch.as_tensor(idx, dtype=torch.long), :].mean(dim=0).numpy())
        out[int(b)] = np.stack(rows, axis=0).astype(np.float32)
    return out


# =============================================================================
# Cross-fitting / statistics
# =============================================================================

def crossfit_secondary_mode(
    arrays: dict[tuple[str, str, int], np.ndarray],
    sample_keys: Sequence[str],
    blocks: Sequence[int],
    *,
    folds: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_sample = []
    fold_rows = []
    n = len(sample_keys)
    fold_id = stable_fold_assignments(n, folds, seed)

    for model in MODEL_ORDER:
        for cond in CONDITION_ORDER:
            for b in blocks:
                X = arrays[(model, cond, b)].astype(np.float64)
                for f in range(int(fold_id.max()) + 1):
                    test = np.where(fold_id == f)[0]
                    train = np.where(fold_id != f)[0]
                    basis, sing, total = secondary_fit_uncentered_basis(X[train], rank=3)

                    Xt = X[test]
                    r1 = residual_remove(Xt, basis, 1)
                    r2 = residual_remove(Xt, basis, 2)
                    r3 = residual_remove(Xt, basis, 3)

                    s1 = Xt @ basis[0]
                    s2 = Xt @ basis[1]
                    total_i = np.sum(Xt * Xt, axis=1)
                    r1_i = np.sum(r1 * r1, axis=1)
                    frac1 = (s1 * s1) / np.maximum(total_i, 1e-12)
                    frac2_post1 = (s2 * s2) / np.maximum(r1_i, 1e-12)

                    for local_j, idx in enumerate(test):
                        per_sample.append({
                            "sample_key": sample_keys[idx],
                            "sample_index": int(idx),
                            "fold": int(f),
                            "model": model,
                            "condition": cond,
                            "block": int(b),
                            "rank1_fraction_total": float(frac1[local_j]),
                            "pc2_fraction_post_pc1": float(frac2_post1[local_j]),
                            "residual1_norm": float(np.linalg.norm(r1[local_j])),
                            "residual2_norm": float(np.linalg.norm(r2[local_j])),
                        })

                    floor = -1.0 / max(len(test) - 1, 1) if len(test) > 1 else float("nan")
                    fold_rows.append({
                        "model": model, "condition": cond, "block": int(b), "fold": int(f),
                        "n_test": int(len(test)),
                        "rank1_removed_mean_pairwise_cosine": mean_offdiag_cosine(r1),
                        "rank2_removed_mean_pairwise_cosine": mean_offdiag_cosine(r2),
                        "rank3_removed_mean_pairwise_cosine": mean_offdiag_cosine(r3),
                        "rank1_removed_resultant_length": resultant_length(r1),
                        "rank2_removed_resultant_length": resultant_length(r2),
                        "finite_n_cosine_floor": floor,
                    })
    return per_sample, fold_rows


def paired_attack_crossfit(
    arrays: dict[tuple[str, str, int], np.ndarray],
    sample_keys: Sequence[str],
    blocks: Sequence[int],
    *,
    folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    n = len(sample_keys)
    fold_id = stable_fold_assignments(n, folds, seed)

    for model in MODEL_ORDER:
        for b in blocks:
            clean = arrays[(model, "NoRTA", b)].astype(np.float64)
            attacks = {a: arrays[(model, a, b)].astype(np.float64) for a in ATTACK_ORDER}
            for f in range(int(fold_id.max()) + 1):
                test = np.where(fold_id == f)[0]
                train = np.where(fold_id != f)[0]

                # Critical anti-leakage choice: fit only on training-fold NoRTA.
                basis, _sing, _total = secondary_fit_uncentered_basis(clean[train], rank=3)

                for attack_name, A in attacks.items():
                    N = clean[test]
                    At = A[test]
                    N1 = residual_remove(N, basis, 1)
                    A1 = residual_remove(At, basis, 1)
                    N2 = residual_remove(N, basis, 2)
                    A2 = residual_remove(At, basis, 2)

                    d = At - N
                    d1 = residual_remove(d, basis, 1)
                    d2_score = d @ basis[1]
                    dtotal = np.sum(d * d, axis=1)
                    d1norm2 = np.sum(d1 * d1, axis=1)

                    n_pc2 = N @ basis[1]
                    a_pc2 = At @ basis[1]

                    for j, idx in enumerate(test):
                        c1 = cosine(N1[j], A1[j])
                        c2 = cosine(N2[j], A2[j])
                        rows.append({
                            "sample_key": sample_keys[idx],
                            "sample_index": int(idx),
                            "fold": int(f),
                            "model": model,
                            "attack_condition": attack_name,
                            "block": int(b),
                            "paired_cos_after_rank1": c1,
                            "paired_cos_after_rank2": c2,
                            "delta_cos_remove_pc2": float(c2 - c1),
                            "attack_contrast_norm": float(math.sqrt(max(dtotal[j], 0.0))),
                            "attack_contrast_pc2_fraction_total": float(
                                d2_score[j] ** 2 / max(dtotal[j], 1e-12)
                            ),
                            "attack_contrast_pc2_fraction_post_pc1": float(
                                d2_score[j] ** 2 / max(d1norm2[j], 1e-12)
                            ),
                            "pc2_coefficient_norta": float(n_pc2[j]),
                            "pc2_coefficient_attack": float(a_pc2[j]),
                            "pc2_coefficient_delta": float(a_pc2[j] - n_pc2[j]),
                            "abs_pc2_coefficient_delta": float(abs(a_pc2[j] - n_pc2[j])),
                        })
    return rows


def summarize_per_sample(
    rows: list[dict[str, Any]],
    group_fields: Sequence[str],
    measures: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    out = []
    for key, g in df.groupby(list(group_fields), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {k: v for k, v in zip(group_fields, key)}
        row["n"] = int(len(g))
        for mi, m in enumerate(measures):
            vals = pd.to_numeric(g[m], errors="coerce").to_numpy(np.float64)
            mean, sem, lo, hi = bootstrap_mean_ci(
                vals, n_boot=n_boot, seed=seed + 1009 * mi + len(out)
            )
            row[m + "_mean"] = mean
            row[m + "_sem"] = sem
            row[m + "_ci95_low"] = lo
            row[m + "_ci95_high"] = hi
            finite = vals[np.isfinite(vals)]
            row[m + "_median"] = float(np.median(finite)) if len(finite) else float("nan")
        out.append(row)
    return out


def model_difference_rows(
    paired_rows: list[dict[str, Any]],
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(paired_rows)
    measures = (
        "delta_cos_remove_pc2",
        "attack_contrast_pc2_fraction_post_pc1",
        "attack_contrast_pc2_fraction_total",
        "abs_pc2_coefficient_delta",
    )
    out = []
    idx_fields = ["sample_key", "attack_condition", "block"]
    for m in measures:
        p = df.pivot_table(index=idx_fields, columns="model", values=m, aggfunc="first").reset_index()
        if not set(MODEL_ORDER).issubset(p.columns):
            continue
        p["diff"] = p["xattn_stripped"] - p["gmp"]
        for (attack, block), g in p.groupby(["attack_condition", "block"]):
            vals = g["diff"].to_numpy(np.float64)
            mean, sem, lo, hi = bootstrap_mean_ci(
                vals, n_boot=n_boot, seed=seed + hash((m, attack, int(block))) % 100000
            )
            out.append({
                "attack_condition": attack,
                "block": int(block),
                "measure": m,
                "difference_definition": "xattn_stripped_minus_gmp",
                "n": int(np.isfinite(vals).sum()),
                "mean_difference": mean,
                "sem": sem,
                "ci95_low": lo,
                "ci95_high": hi,
                "paired_signflip_p": signflip_pvalue(
                    vals,
                    n_perm=n_perm,
                    seed=seed + 17 + hash((m, attack, int(block))) % 100000,
                ),
            })
    return out


def secondary_model_difference_rows(
    crossfit_rows: list[dict[str, Any]],
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(crossfit_rows)
    m = "pc2_fraction_post_pc1"
    p = df.pivot_table(
        index=["sample_key", "condition", "block"],
        columns="model", values=m, aggfunc="first",
    ).reset_index()
    p["diff"] = p["xattn_stripped"] - p["gmp"]
    out = []
    for (cond, block), g in p.groupby(["condition", "block"]):
        vals = g["diff"].to_numpy(np.float64)
        mean, sem, lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed + int(block) * 37)
        out.append({
            "condition": cond,
            "block": int(block),
            "measure": m,
            "difference_definition": "xattn_stripped_minus_gmp",
            "n": int(np.isfinite(vals).sum()),
            "mean_difference": mean,
            "sem": sem,
            "ci95_low": lo,
            "ci95_high": hi,
            "paired_signflip_p": signflip_pvalue(
                vals, n_perm=n_perm, seed=seed + int(block) * 101
            ),
        })
    return out


# =============================================================================
# Descriptive full-sample basis diagnostics
# =============================================================================

def full_basis_diagnostics(
    arrays: dict[tuple[str, str, int], np.ndarray],
    blocks: Sequence[int],
) -> list[dict[str, Any]]:
    out = []
    basis_cache: dict[tuple[str, str, int], np.ndarray] = {}

    for model in MODEL_ORDER:
        for cond in CONDITION_ORDER:
            for b in blocks:
                X = arrays[(model, cond, b)].astype(np.float64)
                basis, sing, total = secondary_fit_uncentered_basis(X, rank=3)
                basis_cache[(model, cond, b)] = basis
                r1 = residual_remove(X, basis, 1)
                r2 = residual_remove(X, basis, 2)
                r3 = residual_remove(X, basis, 3)
                s2 = float(sing[1] ** 2) if len(sing) > 1 else float("nan")
                rem1 = max(total - float(sing[0] ** 2), 1e-12)
                floor = -1.0 / max(len(X) - 1, 1) if len(X) > 1 else float("nan")
                out.append({
                    "model": model, "condition": cond, "block": int(b), "n": int(len(X)),
                    "rank1_energy_fraction": float(sing[0] ** 2 / max(total, 1e-12)),
                    "rank2_component_energy_fraction_total": float(s2 / max(total, 1e-12)),
                    "pc2_share_of_post_pc1_energy": float(s2 / rem1),
                    "raw_mean_pairwise_cosine": mean_offdiag_cosine(X),
                    "rank1_removed_mean_pairwise_cosine": mean_offdiag_cosine(r1),
                    "rank2_removed_mean_pairwise_cosine": mean_offdiag_cosine(r2),
                    "rank3_removed_mean_pairwise_cosine": mean_offdiag_cosine(r3),
                    "rank1_removed_resultant_length": resultant_length(r1),
                    "rank2_removed_resultant_length": resultant_length(r2),
                    "finite_n_cosine_floor": floor,
                })

    # Direction persistence relative to B11, per model/condition.
    for row in out:
        model, cond, b = row["model"], row["condition"], row["block"]
        if (model, cond, 11) in basis_cache:
            row["pc2_abs_cos_to_B11"] = abs(
                cosine(basis_cache[(model, cond, b)][1], basis_cache[(model, cond, 11)][1])
            )
        else:
            row["pc2_abs_cos_to_B11"] = float("nan")

    # Cross-model PC1/PC2 alignment at each block/condition.
    lookup = {(r["model"], r["condition"], r["block"]): r for r in out}
    for cond in CONDITION_ORDER:
        for b in blocks:
            kg = ("gmp", cond, b)
            kx = ("xattn_stripped", cond, b)
            if kg not in basis_cache or kx not in basis_cache:
                continue
            c1 = abs(cosine(basis_cache[kg][0], basis_cache[kx][0]))
            c2 = abs(cosine(basis_cache[kg][1], basis_cache[kx][1]))
            lookup[kg]["cross_model_pc1_abs_cosine"] = c1
            lookup[kx]["cross_model_pc1_abs_cosine"] = c1
            lookup[kg]["cross_model_pc2_abs_cosine"] = c2
            lookup[kx]["cross_model_pc2_abs_cosine"] = c2
    return out


# =============================================================================
# Plotting
# =============================================================================

def _summary_lookup(summary: pd.DataFrame, **kwargs) -> pd.DataFrame:
    q = summary
    for k, v in kwargs.items():
        q = q[q[k] == v]
    return q.sort_values("block")


def plot_secondary_mode(summary_rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), dpi=180, sharey=True)
    for ax, cond in zip(axes, CONDITION_ORDER):
        for model in MODEL_ORDER:
            q = _summary_lookup(df, model=model, condition=cond)
            x = q["block"].to_numpy()
            y = q["pc2_fraction_post_pc1_mean"].to_numpy()
            lo = q["pc2_fraction_post_pc1_ci95_low"].to_numpy()
            hi = q["pc2_fraction_post_pc1_ci95_high"].to_numpy()
            ax.plot(x, y, marker="o", label=model)
            ax.fill_between(x, lo, hi, alpha=0.16)
        ax.set_title(cond)
        ax.set_xlabel("ViT block")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("cross-fitted PC2 fraction of post-PC1 register residual")
    axes[-1].legend()
    fig.suptitle("Secondary register mode after rank-1 carrier removal", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_attack_factorization(summary_rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5), dpi=180, sharex=True)
    for ri, attack in enumerate(ATTACK_ORDER):
        for model in MODEL_ORDER:
            q = _summary_lookup(df, model=model, attack_condition=attack)
            x = q["block"].to_numpy()
            for ci, field in enumerate((
                "delta_cos_remove_pc2",
                "attack_contrast_pc2_fraction_post_pc1",
            )):
                y = q[field + "_mean"].to_numpy()
                lo = q[field + "_ci95_low"].to_numpy()
                hi = q[field + "_ci95_high"].to_numpy()
                axes[ri, ci].plot(x, y, marker="o", label=model)
                axes[ri, ci].fill_between(x, lo, hi, alpha=0.16)

        axes[ri, 0].axhline(0.0, linewidth=0.8)
        axes[ri, 0].set_ylabel(f"{attack}\nmean")
        axes[ri, 0].set_title(
            r"$\Delta\cos$: remove PC2 after PC1"
            if ri == 0 else ""
        )
        axes[ri, 1].set_title(
            "attack-contrast PC2 fraction\nconditional on PC1 removed"
            if ri == 0 else ""
        )
        for ci in range(2):
            axes[ri, ci].grid(alpha=0.25)
            axes[ri, ci].set_xlabel("ViT block")
    axes[0, 1].legend()
    fig.suptitle("Paired RTA / SynthRTA factorization into the secondary register mode", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_rank_removal(diag_rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(diag_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), dpi=180, sharey=True)
    for ax, cond in zip(axes, CONDITION_ORDER):
        for model in MODEL_ORDER:
            q = _summary_lookup(df, model=model, condition=cond)
            ax.plot(
                q["block"], q["rank1_removed_mean_pairwise_cosine"],
                marker="o", label=f"{model}: remove rank1",
            )
            ax.plot(
                q["block"], q["rank2_removed_mean_pairwise_cosine"],
                marker="x", linestyle="--", label=f"{model}: remove rank2",
            )
        ax.set_title(cond)
        ax.set_xlabel("ViT block")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean pairwise cosine of register means")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Rank-stripped register residual coherence (descriptive full-sample basis)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_pc2_persistence(diag_rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(diag_rows)
    fig, ax = plt.subplots(figsize=(8.3, 5.2), dpi=180)
    for model in MODEL_ORDER:
        q = _summary_lookup(df, model=model, condition="NoRTA")
        ax.plot(q["block"], q["pc2_abs_cos_to_B11"], marker="o", label=model)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("ViT block")
    ax.set_ylabel(r"$|\cos(\mathrm{PC2}_B,\mathrm{PC2}_{B11})|$")
    ax.set_title("Persistence of the secondary NoRTA register direction")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def secondary_main() -> None:
    ap = argparse.ArgumentParser(
        description="Statistical RTA test of the bridge-trained secondary register/workspace mode."
    )
    ap.add_argument("--gmp", default=DEFAULT_GMP)
    ap.add_argument("--xattn", default=DEFAULT_XATTN)
    ap.add_argument("--gmp-module", default=DEFAULT_GMP_MODULE)
    ap.add_argument("--module-root", default=".", help="Repository root containing the CLIP model modules.")
    ap.add_argument("--dataset-repo", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--pair-key", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="0 = all paired rows")
    ap.add_argument("--blocks", default=SECONDARY_DEFAULT_BLOCKS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--register-threshold", type=float, default=60.0)
    ap.add_argument("--min-registers", type=int, default=1)
    ap.add_argument("--max-registers", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--output-dir", default=SECONDARY_DEFAULT_OUTPUT)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        try:
            torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
        except Exception:
            pass

    blocks = secondary_parse_blocks(args.blocks)
    if 23 not in blocks:
        blocks = sorted(set(blocks + [23]))  # needed to define register addresses
        print("[note] added B23 because final register addresses are defined there")

    out = Path(args.output_dir)
    data_dir = out / "data"
    plot_dir = out / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    resolved_config, dsets = load_rta_triplet_subsets(
        args.dataset_repo,
        args.split,
        CONDITION_ORDER,
    )
    resolved_configs = {cond: resolved_config for cond in CONDITION_ORDER}
    resolved_splits = {cond: args.split for cond in CONDITION_ORDER}
    for cond in CONDITION_ORDER:
        print(
            f"[dataset] {cond}: config={resolved_config} split={args.split} "
            f"type={cond} n={len(dsets[cond])}"
        )

    pair_key = infer_pair_key(dsets, args.pair_key)
    paired = build_paired_rows(dsets, pair_key, args.limit, args.seed)

    # Pairing audit uses canonicalized IDs, because the raw dataset IDs carry
    # the subset prefix (NoRTA/RTA/SynthRTA).
    pairing_methods: dict[str, int] = {}
    for r in paired:
        pm = str(r.get("pairing_method", "unknown"))
        pairing_methods[pm] = pairing_methods.get(pm, 0) + 1

    audit_payload: dict[str, Any] = {
        "pair_key": pair_key,
        "paired_count": len(paired),
        "pairing_methods": pairing_methods,
        "subset_row_counts": {cond: len(dsets[cond]) for cond in CONDITION_ORDER},
    }
    if pair_key == "id":
        canonical_sets = {
            cond: {
                canonical_rta_pair_id(raw, cond)
                for raw in dsets[cond]["id"]
                if canonical_rta_pair_id(raw, cond)
            }
            for cond in CONDITION_ORDER
        }
        shared_ids = set.intersection(*canonical_sets.values())
        union_ids = set.union(*canonical_sets.values())
        audit_payload.update({
            "raw_ids_are_variant_prefixed": True,
            "canonical_shared_id_count": len(shared_ids),
            "canonical_union_id_count": len(union_ids),
            "canonical_id_sets_identical": all(v == shared_ids for v in canonical_sets.values()),
            "canonical_id_examples": sorted(shared_ids)[:20],
        })
        print(
            f"[pairing audit] canonical shared ids={len(shared_ids)} "
            f"union ids={len(union_ids)} "
            f"identical_sets={all(v == shared_ids for v in canonical_sets.values())}; "
            f"methods={pairing_methods}"
        )
    save_json(data_dir / "dataset_pairing_audit.json", audit_payload)

    if len(paired) < max(args.folds * 2, 10):
        raise RuntimeError(f"Too few paired samples: n={len(paired)}")
    print(f"[pairing] key={pair_key}; paired n={len(paired)}")

    image_cols = {cond: infer_image_column(ds) for cond, ds in dsets.items()}
    save_csv(data_dir / "sample_manifest.csv", paired)

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------
    if args.module_root and args.module_root != ".":
        import sys
        sys.path.insert(0, str(Path(args.module_root).resolve()))

    print("[model] loading clean GmP baseline")
    gmp_model, preprocess = load_gmp_model(args.gmp_module, args.gmp)

    print("[model] loading second clean GmP shell for stripped x-attn visual weights")
    stripped_model, _pre2 = load_gmp_model(args.gmp_module, args.gmp)

    print("[model] loading x-attn checkpoint tensors on CPU as VISUAL WEIGHT SOURCE ONLY")
    xstate = load_xattn_weight_source(args.xattn)

    audit = transplant_visual_state(stripped_model.visual, xstate)
    for row in audit:
        row["source_checkpoint"] = args.xattn
        row["target_clean_shell"] = args.gmp
        row["manual_bare_visual_forward"] = True
        row["rn_inserted"] = False
        row["bridge_called"] = False
    save_csv(data_dir / "model_transplant_audit.csv", audit)

    # Explicit custom-state audit.
    source_custom = [
        k for k in xstate.keys()
        if (
            "read_implant" in k.lower()
            or "read_null" in k.lower()
            or "hard_text" in k.lower()
            or "null_text" in k.lower()
            or "correction" in k.lower()
            or "router" in k.lower()
        )
    ]
    save_json(data_dir / "stripping_audit.json", {
        "native_custom_key_count": len(source_custom),
        "native_custom_key_examples": source_custom[:100],
        "policy": (
            "All custom keys are ignored. Only ordinary visual weights accepted by "
            "the clean GmP visual shell are transplanted. Analysis uses a manual "
            "vanilla ViT forward and never inserts RN or calls bridge code."
        ),
    })

    del xstate
    gc.collect()

    device = torch.device(args.device)
    gmp_model = gmp_model.to(device).float().eval()
    stripped_model = stripped_model.to(device).float().eval()
    for m in (gmp_model, stripped_model):
        for p in m.parameters():
            p.requires_grad_(False)

    runners = {
        "gmp": BareVisual(gmp_model.visual),
        "xattn_stripped": BareVisual(stripped_model.visual),
    }

    # -------------------------------------------------------------------------
    # Extract register means.  Frozen NoRTA addresses per model/sample.
    # -------------------------------------------------------------------------
    n = len(paired)
    arrays: dict[tuple[str, str, int], np.ndarray] = {}
    mask_audit: list[dict[str, Any]] = []
    storage: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)

    for model_name in MODEL_ORDER:
        runner = runners[model_name]
        print(f"\n[extract] model={model_name}")
        for st in tqdm(range(0, n, args.batch_size), desc=model_name, unit="batch"):
            batch_meta = paired[st: st + args.batch_size]

            # NoRTA first: defines fixed register addresses.
            clean_pils = [
                row_image(dsets["NoRTA"][r["NoRTA_index"]], image_cols["NoRTA"])
                for r in batch_meta
            ]
            clean_img = torch.stack([preprocess(im) for im in clean_pils], dim=0).to(device)
            clean_states = runner.capture_patch_states(clean_img, blocks, amp=args.amp)
            clean_final = clean_states[23]
            clean_norms = clean_final.norm(dim=-1)
            frozen_idx = select_register_indices(
                clean_norms,
                threshold=args.register_threshold,
                min_registers=args.min_registers,
                max_registers=args.max_registers,
            )
            clean_means = means_at_indices(clean_states, frozen_idx)
            for b in blocks:
                storage[(model_name, "NoRTA", b)].append(clean_means[b])

            # Mask audit for clean.
            for j, r in enumerate(batch_meta):
                mask_audit.append({
                    "sample_key": r["sample_key"], "model": model_name,
                    "condition": "NoRTA", "register_source": "NoRTA",
                    "register_count": int(len(frozen_idx[j])),
                    "register_indices": "|".join(map(str, frozen_idx[j].tolist())),
                    "own_vs_frozen_jaccard": 1.0,
                    "final_register_norm_mean": float(clean_norms[j, frozen_idx[j]].mean()),
                    "final_patch_norm_median": float(clean_norms[j].median()),
                })

            # Paired attacks: use frozen NoRTA addresses; own attack mask is audit only.
            for cond in ATTACK_ORDER:
                pils = [
                    row_image(dsets[cond][r[f"{cond}_index"]], image_cols[cond])
                    for r in batch_meta
                ]
                img = torch.stack([preprocess(im) for im in pils], dim=0).to(device)
                states = runner.capture_patch_states(img, blocks, amp=args.amp)
                final = states[23]
                norms = final.norm(dim=-1)
                own_idx = select_register_indices(
                    norms,
                    threshold=args.register_threshold,
                    min_registers=args.min_registers,
                    max_registers=args.max_registers,
                )
                means = means_at_indices(states, frozen_idx)
                for b in blocks:
                    storage[(model_name, cond, b)].append(means[b])

                for j, r in enumerate(batch_meta):
                    mask_audit.append({
                        "sample_key": r["sample_key"], "model": model_name,
                        "condition": cond, "register_source": "frozen_NoRTA",
                        "register_count": int(len(frozen_idx[j])),
                        "register_indices": "|".join(map(str, frozen_idx[j].tolist())),
                        "own_attack_register_indices": "|".join(map(str, own_idx[j].tolist())),
                        "own_vs_frozen_jaccard": mask_jaccard(frozen_idx[j], own_idx[j]),
                        "final_register_norm_mean": float(norms[j, frozen_idx[j]].mean()),
                        "final_patch_norm_median": float(norms[j].median()),
                    })
                del img, states, final, norms
            del clean_img, clean_states, clean_final, clean_norms
            if device.type == "cuda" and ((st // args.batch_size + 1) % 8 == 0):
                torch.cuda.empty_cache()

    for key, chunks in storage.items():
        arrays[key] = np.concatenate(chunks, axis=0).astype(np.float32)
    save_csv(data_dir / "register_mask_audit.csv", mask_audit)

    # Local forensic archive (deliberately omitted from the compact summary ZIP).
    npz_payload = {}
    for (model, cond, b), arr in arrays.items():
        npz_payload[f"{model}__{cond}__B{b}"] = arr
    np.savez_compressed(data_dir / "register_means.npz", **npz_payload)

    # -------------------------------------------------------------------------
    # Primary cross-fitted analyses
    # -------------------------------------------------------------------------
    crossfit_rows, fold_rows = crossfit_secondary_mode(
        arrays,
        [r["sample_key"] for r in paired],
        blocks,
        folds=args.folds,
        seed=args.seed,
    )
    save_csv(data_dir / "crossfit_pc2_per_sample.csv", crossfit_rows)
    save_csv(data_dir / "crossfit_fold_coherence.csv", fold_rows)

    crossfit_summary = summarize_per_sample(
        crossfit_rows,
        ["model", "condition", "block"],
        ["rank1_fraction_total", "pc2_fraction_post_pc1"],
        n_boot=args.bootstrap,
        seed=args.seed + 100,
    )
    save_csv(data_dir / "crossfit_pc2_summary.csv", crossfit_summary)

    secondary_diff = secondary_model_difference_rows(
        crossfit_rows,
        n_boot=args.bootstrap,
        n_perm=args.permutations,
        seed=args.seed + 200,
    )
    save_csv(data_dir / "secondary_mode_model_difference.csv", secondary_diff)

    paired_rows = paired_attack_crossfit(
        arrays,
        [r["sample_key"] for r in paired],
        blocks,
        folds=args.folds,
        seed=args.seed + 333,
    )
    save_csv(data_dir / "paired_attack_per_sample.csv", paired_rows)

    paired_summary = summarize_per_sample(
        paired_rows,
        ["model", "attack_condition", "block"],
        [
            "paired_cos_after_rank1",
            "paired_cos_after_rank2",
            "delta_cos_remove_pc2",
            "attack_contrast_pc2_fraction_total",
            "attack_contrast_pc2_fraction_post_pc1",
            "abs_pc2_coefficient_delta",
        ],
        n_boot=args.bootstrap,
        seed=args.seed + 300,
    )
    save_csv(data_dir / "paired_attack_summary.csv", paired_summary)

    model_diff = model_difference_rows(
        paired_rows,
        n_boot=args.bootstrap,
        n_perm=args.permutations,
        seed=args.seed + 400,
    )
    save_csv(data_dir / "model_difference_summary.csv", model_diff)

    # -------------------------------------------------------------------------
    # Descriptive old-style rank-removal curves / direction persistence.
    # -------------------------------------------------------------------------
    diag_rows = full_basis_diagnostics(arrays, blocks)
    save_csv(data_dir / "basis_diagnostics.csv", diag_rows)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    plot_secondary_mode(
        crossfit_summary,
        plot_dir / "secondary_mode_crossfit.png",
    )
    plot_attack_factorization(
        paired_summary,
        plot_dir / "attack_factorization.png",
    )
    plot_rank_removal(
        diag_rows,
        plot_dir / "rank_removal_coherence.png",
    )
    plot_pc2_persistence(
        diag_rows,
        plot_dir / "pc2_persistence.png",
    )

    # -------------------------------------------------------------------------
    # Config / concise summary.
    # -------------------------------------------------------------------------
    config = {
        "gmp_checkpoint": args.gmp,
        "xattn_checkpoint": args.xattn,
        "gmp_module": args.gmp_module,
        "dataset_repo": args.dataset_repo,
        "resolved_configs": resolved_configs,
        "resolved_splits": resolved_splits,
        "pair_key": pair_key,
        "n_paired_samples": len(paired),
        "blocks": blocks,
        "register_threshold": args.register_threshold,
        "min_registers": args.min_registers,
        "max_registers": args.max_registers,
        "register_pair_policy": "NoRTA final B23 register addresses frozen across paired RTA/SynthRTA",
        "crossfit_folds": args.folds,
        "bootstrap_resamples": args.bootstrap,
        "paired_signflip_permutations": args.permutations,
        "amp": bool(args.amp),
        "tf32": bool(args.tf32),
        "xattn_stripping": {
            "bridge": "not instantiated in clean shell / never called",
            "rn": "never inserted by manual bare visual forward",
            "custom_source_keys": "ignored",
            "ordinary_visual_weights": "transplanted into clean GmP visual shell",
        },
        "primary_inference": (
            "cross-fitted per-sample PC2 dominance and paired attack factorization; "
            "full-sample rank-removal curves are descriptive"
        ),
    }
    save_json(data_dir / "config.json", config)

    # Pull a few paper-facing headline rows if blocks exist.
    cf = pd.DataFrame(crossfit_summary)
    ps = pd.DataFrame(paired_summary)
    md = pd.DataFrame(model_diff)
    lines = [
        "RTA REGISTER SECONDARY-MODE ANALYSIS",
        "=" * 76,
        "",
        "Models:",
        f"  GmP:             {args.gmp}",
        f"  x-attn stripped: {args.xattn}",
        "  stripped evaluation uses ordinary visual weights only; no RN insertion and no bridge calls.",
        "",
        f"Dataset: {args.dataset_repo}",
        f"Paired samples: {len(paired)}",
        f"Pair key: {pair_key}",
        f"Blocks: {blocks}",
        "",
        "Primary tests are cross-fitted.  Attack-pair bases are fitted on training-fold NoRTA only.",
        "",
    ]
    for b in (11, 12, 20, 23):
        if b not in blocks:
            continue
        lines.append(f"B{b}:")
        for cond in CONDITION_ORDER:
            bits = []
            for model in MODEL_ORDER:
                q = cf[
                    (cf["model"] == model)
                    & (cf["condition"] == cond)
                    & (cf["block"] == b)
                ]
                if len(q):
                    r = q.iloc[0]
                    bits.append(
                        f"{model} PC2/postPC1={r['pc2_fraction_post_pc1_mean']:.4f}"
                        f" [{r['pc2_fraction_post_pc1_ci95_low']:.4f},"
                        f"{r['pc2_fraction_post_pc1_ci95_high']:.4f}]"
                    )
            if bits:
                lines.append(f"  {cond}: " + " | ".join(bits))
        for attack in ATTACK_ORDER:
            bits = []
            for model in MODEL_ORDER:
                q = ps[
                    (ps["model"] == model)
                    & (ps["attack_condition"] == attack)
                    & (ps["block"] == b)
                ]
                if len(q):
                    r = q.iloc[0]
                    bits.append(
                        f"{model} dCosPC2={r['delta_cos_remove_pc2_mean']:+.4f}, "
                        f"contrastPC2/postPC1={r['attack_contrast_pc2_fraction_post_pc1_mean']:.4f}"
                    )
            if bits:
                lines.append(f"  {attack}: " + " | ".join(bits))
        lines.append("")

    lines += [
        "Interpretation guardrails:",
        "  * Cross-fitted PC2 metrics test a secondary low-rank register mode without evaluating each image",
        "    on a basis fit to itself.",
        "  * Paired RTA/SynthRTA metrics use frozen NoRTA register addresses and a basis fit on training-fold",
        "    NoRTA only; attack-dependent register relocation is audited separately.",
        "  * Geometric factorization does not by itself establish causal mediation of robustness.",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Compact handoff.
    handoff = [
        data_dir / "model_transplant_audit.csv",
        data_dir / "stripping_audit.json",
        data_dir / "dataset_pairing_audit.json",
        data_dir / "sample_manifest.csv",
        data_dir / "register_mask_audit.csv",
        data_dir / "basis_diagnostics.csv",
        data_dir / "crossfit_pc2_per_sample.csv",
        data_dir / "crossfit_pc2_summary.csv",
        data_dir / "crossfit_fold_coherence.csv",
        data_dir / "secondary_mode_model_difference.csv",
        data_dir / "paired_attack_per_sample.csv",
        data_dir / "paired_attack_summary.csv",
        data_dir / "model_difference_summary.csv",
        data_dir / "config.json",
        out / "SUMMARY.txt",
    ] + sorted(plot_dir.glob("*.png"))

    zpath = out / "compact_summary_workspace_register_geometry_secondary.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for p in handoff:
            if p.exists() and p.is_file():
                z.write(p, arcname=p.relative_to(out).as_posix())

    print("\n[done]")
    print("  output:", out.resolve())
    print("  summary:", (out / "SUMMARY.txt").resolve())
    print("  compact summary:", zpath.resolve())
    print("  local raw means:", (data_dir / "register_means.npz").resolve())

    del gmp_model, stripped_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# GRAD ATTENTION
import importlib.util
import sys
from typing import Any, Mapping, Sequence

matplotlib.use("Agg")
import torch.nn.functional as F


GRAD_ATTENTION_DEFAULT_OUTPUT = r"out_rta_late_gradattn_trajectory"
GRAD_ATTENTION_DEFAULT_BLOCKS = "18-23"
DEFAULT_CONDITIONS = "RTA,SynthRTA"


# =============================================================================
# Previous-analysis import
# =============================================================================


# =============================================================================
# Generic helpers
# =============================================================================

def grad_attention_parse_blocks(text: str) -> list[int]:
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


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > eps else float("nan")


def normalize_positive_map(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, np.float64)
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if s <= eps:
        return np.full_like(x, np.nan, dtype=np.float64)
    return x / s


def entropy_effective_count(p: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    p = np.asarray(p, np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return float("nan"), float("nan")
    h = float(-(p * np.log(np.maximum(p, eps))).sum())
    return h, float(math.exp(h))


def safe_diff_of_means(vals: np.ndarray) -> float:
    x = np.asarray(vals, np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


# =============================================================================
# Text-area parsing and mask construction
# =============================================================================

def parse_text_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """
    Accept:
      {"bbox":[x1,y1,x2,y2],"format":"xyxy", ...}
      JSON string with the same
      plain [x1,y1,x2,y2]
    """
    if value is None:
        return None

    obj = value
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
        except Exception:
            nums = re.findall(r"[-+]?\d*\.?\d+", s)
            if len(nums) >= 4:
                return tuple(map(float, nums[:4]))  # type: ignore
            return None

    if isinstance(obj, Mapping):
        bbox = obj.get("bbox")
        fmt = str(obj.get("format", "xyxy")).lower()
        if bbox is None:
            return None
        vals = list(map(float, bbox))
        if len(vals) != 4:
            return None
        if fmt == "xyxy":
            return tuple(vals)  # type: ignore
        if fmt in {"xywh", "coco"}:
            x, y, w, h = vals
            return (x, y, x + w, y + h)
        return tuple(vals)  # safest known benchmark format is xyxy

    if isinstance(obj, (list, tuple)) and len(obj) == 4:
        return tuple(map(float, obj))  # type: ignore

    return None


def bbox_mask_after_clip_resize_crop(
    bbox_xyxy: tuple[float, float, float, float],
    original_size_wh: tuple[int, int],
    out_size: int,
) -> torch.Tensor:
    """
    Approximate OpenAI CLIP eval transform:
      Resize(shorter side -> out_size), then CenterCrop(out_size).

    Used only as audit/fallback; pixel diff is the default.
    """
    w, h = original_size_wh
    x1, y1, x2, y2 = bbox_xyxy
    if min(w, h) <= 0:
        return torch.zeros(out_size, out_size, dtype=torch.bool)

    scale = float(out_size) / float(min(w, h))
    new_w = float(w) * scale
    new_h = float(h) * scale
    crop_x = max(0.0, (new_w - out_size) / 2.0)
    crop_y = max(0.0, (new_h - out_size) / 2.0)

    xx1 = x1 * scale - crop_x
    xx2 = x2 * scale - crop_x
    yy1 = y1 * scale - crop_y
    yy2 = y2 * scale - crop_y

    ix1 = max(0, min(out_size, int(math.floor(xx1))))
    ix2 = max(0, min(out_size, int(math.ceil(xx2))))
    iy1 = max(0, min(out_size, int(math.floor(yy1))))
    iy2 = max(0, min(out_size, int(math.ceil(yy2))))

    mask = torch.zeros(out_size, out_size, dtype=torch.bool)
    if ix2 > ix1 and iy2 > iy1:
        mask[iy1:iy2, ix1:ix2] = True
    return mask


def pixel_diff_mask(
    clean_chw: torch.Tensor,
    attack_chw: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """
    Both tensors are AFTER the same deterministic CLIP preprocessing and
    normalization, so geometry is exactly aligned with the ViT patches.
    """
    d = (attack_chw.float() - clean_chw.float()).abs().amax(dim=0)
    return d > float(threshold)


def pixel_to_patch_coverage(mask_hw: torch.Tensor, patch_size: int) -> np.ndarray:
    x = mask_hw.float()[None, None]
    cov = F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)
    return cov[0, 0].reshape(-1).cpu().numpy().astype(np.float32)


def build_text_patch_masks(
    analysis_mod,
    dsets: Mapping[str, Any],
    paired: list[dict[str, Any]],
    preprocess,
    attack_conditions: Sequence[str],
    *,
    diff_threshold: float,
    diff_min_pixels: int,
    diff_max_fraction: float,
    patch_size: int,
    input_size: int,
    mask_source: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    coverage = {
        cond: np.zeros((len(paired), (input_size // patch_size) ** 2), np.float32)
        for cond in attack_conditions
    }
    audit: list[dict[str, Any]] = []

    for i, pair in enumerate(tqdm(paired, desc="text masks", unit="triplet")):
        clean_row = dsets["NoRTA"][pair["NoRTA_index"]]
        clean_pil = analysis_mod.row_image(clean_row, "image")
        clean_t = preprocess(clean_pil).cpu()

        for cond in attack_conditions:
            attack_row = dsets[cond][pair[f"{cond}_index"]]
            attack_pil = analysis_mod.row_image(attack_row, "image")
            attack_t = preprocess(attack_pil).cpu()

            dm = pixel_diff_mask(clean_t, attack_t, threshold=diff_threshold)
            diff_pixels = int(dm.sum())
            diff_fraction = float(dm.float().mean())

            bbox = parse_text_bbox(attack_row.get("text_area"))
            bm = (
                bbox_mask_after_clip_resize_crop(bbox, attack_pil.size, input_size)
                if bbox is not None
                else torch.zeros(input_size, input_size, dtype=torch.bool)
            )

            diff_valid = (
                diff_pixels >= int(diff_min_pixels)
                and diff_fraction <= float(diff_max_fraction)
            )

            if mask_source == "diff":
                if diff_valid:
                    final_mask = dm
                    used = "diff"
                elif bbox is not None:
                    final_mask = bm
                    used = "bbox_fallback"
                else:
                    raise RuntimeError(
                        f"{pair['sample_key']} {cond}: invalid pixel diff "
                        f"(pixels={diff_pixels}, fraction={diff_fraction:.4g}) "
                        "and no text_area bbox available."
                    )
            elif mask_source == "bbox":
                if bbox is None:
                    raise RuntimeError(f"{pair['sample_key']} {cond}: no text_area bbox")
                final_mask = bm
                used = "bbox"
            elif mask_source == "union":
                final_mask = dm | bm if bbox is not None else dm
                used = "union"
            elif mask_source == "intersection":
                final_mask = dm & bm if bbox is not None else dm
                used = "intersection"
            else:
                raise ValueError(mask_source)

            coverage[cond][i] = pixel_to_patch_coverage(final_mask, patch_size)

            inter = float((dm & bm).sum()) if bbox is not None else float("nan")
            union = float((dm | bm).sum()) if bbox is not None else float("nan")
            audit.append({
                "sample_key": pair["sample_key"],
                "condition": cond,
                "mask_source_used": used,
                "pixel_diff_threshold": float(diff_threshold),
                "diff_pixels": diff_pixels,
                "diff_fraction": diff_fraction,
                "bbox_present": bbox is not None,
                "bbox_xyxy": "" if bbox is None else "|".join(f"{x:.3f}" for x in bbox),
                "bbox_diff_pixel_iou": inter / union if bbox is not None and union > 0 else float("nan"),
                "patches_with_any_text_pixel": int((coverage[cond][i] > 0).sum()),
                "mean_patch_text_coverage_on_touched": (
                    float(coverage[cond][i][coverage[cond][i] > 0].mean())
                    if np.any(coverage[cond][i] > 0)
                    else 0.0
                ),
            })

    return coverage, audit


# =============================================================================
# Exact manual OpenAI-style late attention with exposed probabilities
# =============================================================================

def manual_self_attention(
    attn: torch.nn.Module,
    x_tbd: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    OpenAI CLIP MultiheadAttention self-attention, written explicitly so the
    returned softmax probabilities are the actual tensor used by V aggregation.
    """
    if getattr(attn, "bias_k", None) is not None or getattr(attn, "bias_v", None) is not None:
        raise RuntimeError("bias_k/bias_v attention is unsupported in this audit")
    if bool(getattr(attn, "add_zero_attn", False)):
        raise RuntimeError("add_zero_attn=True is unsupported in this audit")
    if getattr(attn, "head_mask", None) is not None:
        raise RuntimeError("head_mask is unsupported in this exact-attention audit")
    if getattr(attn, "_prefix_k", None) is not None or getattr(attn, "_prefix_v", None) is not None:
        raise RuntimeError("prefix K/V state is unsupported in this exact-attention audit")

    T, B, D = x_tbd.shape
    H = int(attn.num_heads)
    if D % H:
        raise RuntimeError(f"embed dim {D} not divisible by heads {H}")
    hd = D // H

    # Support both attention representations used by the repository:
    #   1) stock torch/OpenAI MultiheadAttention: packed in_proj_weight/bias
    #   2) attnclip_mechinterp_sae: explicit q_proj/k_proj/v_proj modules
    # They are algebraically identical for the bare self-attention path analyzed
    # here.  Do not infer the layout from the model name; inspect the module.
    packed = getattr(attn, "in_proj_weight", None)
    q_proj = getattr(attn, "q_proj", None)
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)

    if packed is not None:
        qkv = F.linear(x_tbd, packed, getattr(attn, "in_proj_bias", None))
        q, k, v = qkv.chunk(3, dim=-1)
    elif q_proj is not None and k_proj is not None and v_proj is not None:
        q = q_proj(x_tbd)
        k = k_proj(x_tbd)
        v = v_proj(x_tbd)
    else:
        raise RuntimeError(
            "Unsupported visual attention projection layout: expected either "
            "packed in_proj_weight or explicit q_proj/k_proj/v_proj modules"
        )

    def split_heads(z: torch.Tensor) -> torch.Tensor:
        # T,B,D -> B,H,T,hd
        return z.permute(1, 0, 2).reshape(B, T, H, hd).permute(0, 2, 1, 3)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    logits = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
    if attn_mask is not None:
        m = attn_mask.to(device=logits.device, dtype=logits.dtype)
        if m.ndim == 2:
            logits = logits + m[None, None, :, :]
        elif m.ndim == 3:
            logits = logits + m
        else:
            raise RuntimeError(f"Unsupported attention mask shape {tuple(m.shape)}")

    probs = torch.softmax(logits, dim=-1)
    context = torch.matmul(probs, v)  # B,H,T,hd
    context = context.permute(0, 2, 1, 3).reshape(B, T, D).permute(1, 0, 2)
    out = F.linear(context, attn.out_proj.weight, attn.out_proj.bias)
    return out, probs


class BareLateGradAttn:
    def __init__(self, visual: torch.nn.Module, blocks: Sequence[int]):
        self.visual = visual
        self.blocks = list(map(int, blocks))
        self.block_set = set(self.blocks)
        self.first = min(self.blocks)
        self.last = max(self.blocks)

        if self.last >= len(self.visual.transformer.resblocks):
            raise ValueError(
                f"Requested B{self.last} but model has only "
                f"{len(self.visual.transformer.resblocks)} blocks"
            )

    def _stem(self, images_bchw: torch.Tensor) -> torch.Tensor:
        v = self.visual
        x = v.conv1(images_bchw)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(
            x.shape[0], 1, x.shape[-1],
            dtype=x.dtype, device=x.device,
        )
        x = torch.cat([cls, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        return x.permute(1, 0, 2)  # T,B,D

    def _finish(self, x_tbd: torch.Tensor) -> torch.Tensor:
        v = self.visual
        x = v.ln_post(x_tbd[0])
        if getattr(v, "proj", None) is not None:
            x = x @ v.proj
        return x

    @torch.no_grad()
    def embedding_native_bare(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """
        Reference *bare visual* forward using the module's ordinary residual
        blocks directly.  This deliberately bypasses model.encode_image(),
        because the clean GmP shell may still expose PIECES/content-correction
        machinery at the root model level even when RN is disabled.

        The path is exactly:
            stem -> ordinary resblocks B0..B23 -> ln_post -> visual proj
        with no RN insertion and no bridge/correction call.
        """
        x = self._stem(images_bchw)
        for block in self.visual.transformer.resblocks:
            x = block(x)
        return self._finish(x)

    @torch.no_grad()
    def embedding_manual_full(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """Same bare visual path, but with self-attention written explicitly."""
        x = self._stem(images_bchw)
        for block in self.visual.transformer.resblocks:
            y = block.ln_1(x)
            mask = getattr(block, "attn_mask", None)
            a, _p = manual_self_attention(block.attn, y, mask)
            x = x + a
            x = x + block.mlp(block.ln_2(x))
        return self._finish(x)

    def maps_and_score(
        self,
        images_bchw: torch.Tensor,
        target_text_bd: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          maps       [B, n_blocks, n_patches]
          scores     [B]
          final_norm [B, n_patches]
        """
        # Everything before the first requested block is pure forward state;
        # no gradient history is needed.
        with torch.no_grad():
            x = self._stem(images_bchw)
            for b in range(self.first):
                x = self.visual.transformer.resblocks[b](x)

        # Start the differentiable graph only here.
        x = x.detach().requires_grad_(True)

        probs_by_block: dict[int, torch.Tensor] = {}
        final_patch_norm = None

        for b in range(self.first, len(self.visual.transformer.resblocks)):
            block = self.visual.transformer.resblocks[b]
            y = block.ln_1(x)
            mask = getattr(block, "attn_mask", None)
            a, probs = manual_self_attention(block.attn, y, mask)
            x = x + a
            x = x + block.mlp(block.ln_2(x))

            if b in self.block_set:
                probs_by_block[b] = probs
            if b == 23:
                final_patch_norm = x[1:].permute(1, 0, 2).float().norm(dim=-1)
            if b >= self.last and self.last == len(self.visual.transformer.resblocks) - 1:
                # normal case B23; continue no further
                pass

        image_emb = F.normalize(self._finish(x).float(), dim=-1)
        text_emb = F.normalize(target_text_bd.float(), dim=-1)
        score = (image_emb * text_emb).sum(dim=-1)

        probs_list = [probs_by_block[b] for b in self.blocks]
        grads = torch.autograd.grad(
            score.sum(),
            probs_list,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        maps = []
        for b, p, g in zip(self.blocks, probs_list, grads):
            if p.ndim != 4:
                raise RuntimeError(f"B{b}: expected [B,H,T,T] probs, got {tuple(p.shape)}")
            # Positive local target attribution from CLS query to spatial patches.
            gx = (p * g).clamp(min=0).mean(dim=1)[:, 0, 1:]
            maps.append(gx.detach().float().cpu())

        if final_patch_norm is None:
            # If requested blocks did not include B23, final state still traversed all
            # blocks above, so this should not happen.
            final_patch_norm = x[1:].permute(1, 0, 2).float().norm(dim=-1)

        maps_bkp = torch.stack(maps, dim=1).numpy().astype(np.float32)
        scores_b = score.detach().cpu().numpy().astype(np.float32)
        norms_bp = final_patch_norm.detach().cpu().numpy().astype(np.float32)

        return maps_bkp, scores_b, norms_bp


# =============================================================================
# Text embedding bank and model construction
# =============================================================================

def build_text_bank(
    model: torch.nn.Module,
    clip_mod,
    labels: Sequence[str],
    prompt_template: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    model = model.to(device).eval()
    unique = sorted(set(map(str, labels)))
    for st in tqdm(range(0, len(unique), batch_size), desc="text embeddings", unit="batch"):
        batch = unique[st:st + batch_size]
        prompts = [prompt_template.format(label=x) for x in batch]
        try:
            toks = clip_mod.tokenize(prompts, truncate=True).to(device)
        except TypeError:
            toks = clip_mod.tokenize(prompts).to(device)
        with torch.no_grad():
            z = F.normalize(model.encode_text(toks).float(), dim=-1)
        for label, vec in zip(batch, z.detach().cpu().numpy()):
            out[label] = vec.astype(np.float32)
    return out


def select_register_indices_np(
    final_norms_np: np.ndarray,
    threshold: float,
    min_registers: int,
    max_registers: int,
) -> np.ndarray:
    n, p = final_norms_np.shape
    mask = np.zeros((n, p), dtype=bool)
    for i in range(n):
        norms = final_norms_np[i]
        idx = np.where(norms >= threshold)[0]
        if len(idx) > max_registers:
            idx = idx[np.argsort(norms[idx])[::-1][:max_registers]]
        if len(idx) < min_registers:
            idx = np.argsort(norms)[::-1][:min_registers]
        mask[i, idx] = True
    return mask


# =============================================================================
# Dataset/model forward
# =============================================================================

def process_model_maps(
    model_name: str,
    model: torch.nn.Module,
    preprocess,
    dsets: Mapping[str, Any],
    paired: list[dict[str, Any]],
    conditions: Sequence[str],
    labels: Sequence[str],
    text_bank: Mapping[str, np.ndarray],
    blocks: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    validation_cos_min: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """
    Process NoRTA plus attacked conditions.
    """
    visual = model.visual.to(device).float().eval()
    for p in visual.parameters():
        p.requires_grad_(False)

    runner = BareLateGradAttn(visual, blocks)
    n = len(paired)
    maps: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    clean_final_norms = None

    all_conditions = ["NoRTA"] + list(conditions)
    validated = False

    for cond in all_conditions:
        cond_maps = None
        cond_scores = np.zeros(n, np.float32)
        cond_norms = np.zeros((n, 256), np.float32)

        progress = tqdm(
            range(0, n, batch_size),
            desc=f"{model_name}/{cond}",
            unit="batch",
        )
        for st in progress:
            inds = list(range(st, min(n, st + batch_size)))
            pils = []
            tvecs = []
            for i in inds:
                pair = paired[i]
                row = dsets[cond][pair[f"{cond}_index"]]
                pils.append(analysis_mod.row_image(row, "image"))
                tvecs.append(text_bank[labels[i]])

            images = torch.stack([preprocess(im) for im in pils], dim=0).to(device)
            target = torch.from_numpy(np.stack(tvecs, axis=0)).to(device)

            if not validated:
                # Validate the explicit attention algebra against the SAME bare
                # vanilla visual computation that is analyzed below.  Do not call
                # model.encode_image(): in these oaiclip shells that root method can
                # still invoke content_correction/read_implant, which is both the
                # wrong scientific path for this stripped comparison and may reside
                # on CPU while only model.visual has been moved to CUDA.
                with torch.no_grad():
                    native_bare = F.normalize(
                        runner.embedding_native_bare(images).float(), dim=-1
                    )
                    manual = F.normalize(
                        runner.embedding_manual_full(images).float(), dim=-1
                    )
                    c = F.cosine_similarity(native_bare, manual, dim=-1)
                    min_cos = float(c.min().cpu())
                    max_abs = float((native_bare - manual).abs().max().cpu())
                print(
                    f"[bare visual validation] {model_name}: min cosine={min_cos:.9f} "
                    f"max_abs(normalized)={max_abs:.6g}"
                )
                if min_cos < validation_cos_min:
                    raise RuntimeError(
                        f"Explicit-attention bare visual forward failed validation for {model_name}: "
                        f"min cosine={min_cos}, threshold={validation_cos_min}"
                    )
                validated = True

            m, s, norms = runner.maps_and_score(images, target)
            if cond_maps is None:
                cond_maps = np.zeros(
                    (n, m.shape[1], m.shape[2]),
                    dtype=np.float32,
                )
                if m.shape[2] != 256:
                    raise RuntimeError(
                        f"Expected 256 patches for ViT-L/14@224, got {m.shape[2]}"
                    )

            cond_maps[inds] = m
            cond_scores[inds] = s
            cond_norms[inds] = norms

            del images, target, m, s, norms
            if device.type == "cuda":
                torch.cuda.empty_cache()

        assert cond_maps is not None
        maps[cond] = cond_maps
        scores[cond] = cond_scores
        if cond == "NoRTA":
            clean_final_norms = cond_norms.copy()

    if clean_final_norms is None:
        raise RuntimeError("NoRTA final norms were not collected")

    return maps, scores, clean_final_norms


# =============================================================================
# Metrics
# =============================================================================

def top_clean_support_mask(
    shared_clean_p: np.ndarray,
    text_binary: np.ndarray,
    support_fraction: float,
) -> np.ndarray:
    eligible = ~text_binary
    idx = np.where(eligible & np.isfinite(shared_clean_p))[0]
    mask = np.zeros_like(text_binary, dtype=bool)
    if len(idx) == 0:
        return mask
    k = max(1, int(math.ceil(float(support_fraction) * len(idx))))
    order = idx[np.argsort(shared_clean_p[idx])[::-1][:k]]
    mask[order] = True
    return mask


def compute_metrics(
    paired: list[dict[str, Any]],
    labels: Sequence[str],
    blocks: Sequence[int],
    conditions: Sequence[str],
    maps_by_model: Mapping[str, Mapping[str, np.ndarray]],
    scores_by_model: Mapping[str, Mapping[str, np.ndarray]],
    register_masks: Mapping[str, np.ndarray],
    text_coverage: Mapping[str, np.ndarray],
    *,
    text_patch_min_coverage: float,
    support_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_model_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    n = len(paired)
    for cond in conditions:
        cov_np = text_coverage[cond]
        for i in range(n):
            text_binary = cov_np[i] >= float(text_patch_min_coverage)

            # Shared paired-clean reference for fair cross-model visual recovery.
            shared_clean: dict[int, np.ndarray] = {}
            shared_support: dict[int, np.ndarray] = {}
            for bi, b in enumerate(blocks):
                gclean = normalize_positive_map(maps_by_model["gmp"]["NoRTA"][i, bi])
                xclean = normalize_positive_map(maps_by_model["xattn_stripped"]["NoRTA"][i, bi])
                if np.all(np.isfinite(gclean)) and np.all(np.isfinite(xclean)):
                    sh = 0.5 * (gclean + xclean)
                    sh = sh / max(float(sh.sum()), 1e-12)
                else:
                    sh = np.full(256, np.nan, np.float64)
                shared_clean[b] = sh
                shared_support[b] = top_clean_support_mask(
                    sh, text_binary, support_fraction
                )

            # Per-model attacked metrics.
            for model in MODEL_ORDER:
                regmask = register_masks[model][i]
                for bi, b in enumerate(blocks):
                    raw_attack = maps_by_model[model][cond][i, bi]
                    raw_clean = maps_by_model[model]["NoRTA"][i, bi]
                    p_attack = normalize_positive_map(raw_attack)
                    p_clean = normalize_positive_map(raw_clean)

                    zero_map = not np.all(np.isfinite(p_attack))
                    if zero_map:
                        text_mass_binary = float("nan")
                        text_mass_weighted = float("nan")
                        register_mass = float("nan")
                        clean_cos = float("nan")
                        clean_nontext_cos = float("nan")
                        shared_cos = float("nan")
                        shared_nontext_cos = float("nan")
                        support_mass = float("nan")
                        ent = float("nan")
                        eff = float("nan")
                    else:
                        text_mass_binary = float(p_attack[text_binary].sum())
                        text_mass_weighted = float(np.nansum(p_attack * cov_np[i]))
                        register_mass = float(p_attack[regmask].sum())

                        clean_cos = cosine_np(raw_attack, raw_clean)
                        keep = ~text_binary
                        clean_nontext_cos = cosine_np(raw_attack[keep], raw_clean[keep])

                        sh = shared_clean[b]
                        shared_cos = cosine_np(raw_attack, sh)
                        shared_nontext_cos = cosine_np(raw_attack[keep], sh[keep])

                        support = shared_support[b]
                        support_mass = float(p_attack[support].sum()) if support.any() else float("nan")
                        ent, eff = entropy_effective_count(p_attack)

                    per_model_rows.append({
                        "sample_key": paired[i]["sample_key"],
                        "sample_index": i,
                        "object_label": labels[i],
                        "condition": cond,
                        "model": model,
                        "block": int(b),
                        "target_score": float(scores_by_model[model][cond][i]),
                        "clean_target_score": float(scores_by_model[model]["NoRTA"][i]),
                        "zero_positive_map": bool(zero_map),
                        "text_patch_count": int(text_binary.sum()),
                        "text_mass_binary": text_mass_binary,
                        "text_mass_coverage_weighted": text_mass_weighted,
                        "register_mass": register_mass,
                        "attack_to_clean_cosine": clean_cos,
                        "attack_to_clean_nontext_cosine": clean_nontext_cos,
                        "attack_to_shared_clean_cosine": shared_cos,
                        "attack_to_shared_clean_nontext_cosine": shared_nontext_cos,
                        "shared_clean_support_mass": support_mass,
                        "map_entropy": ent,
                        "effective_patch_count": eff,
                    })

            # Cross-model attacked-map metrics.
            for bi, b in enumerate(blocks):
                g = maps_by_model["gmp"][cond][i, bi]
                x = maps_by_model["xattn_stripped"][cond][i, bi]
                keep = ~text_binary
                txt = text_binary

                gp = normalize_positive_map(g)
                xp = normalize_positive_map(x)
                if np.all(np.isfinite(gp)) and np.all(np.isfinite(xp)):
                    g_text = float(gp[txt].sum())
                    x_text = float(xp[txt].sum())
                    g_reg = float(gp[register_masks["gmp"][i]].sum())
                    x_reg = float(xp[register_masks["xattn_stripped"][i]].sum())
                else:
                    g_text = x_text = g_reg = x_reg = float("nan")

                cross_rows.append({
                    "sample_key": paired[i]["sample_key"],
                    "sample_index": i,
                    "object_label": labels[i],
                    "condition": cond,
                    "block": int(b),
                    "cross_model_map_cosine": cosine_np(g, x),
                    "cross_model_nontext_cosine": cosine_np(g[keep], x[keep]),
                    "cross_model_text_only_cosine": (
                        cosine_np(g[txt], x[txt]) if int(txt.sum()) >= 2 else float("nan")
                    ),
                    "abs_text_mass_difference": abs(x_text - g_text),
                    "xattn_minus_gmp_text_mass": x_text - g_text,
                    "abs_register_mass_difference": abs(x_reg - g_reg),
                    "xattn_minus_gmp_register_mass": x_reg - g_reg,
                })

    return per_model_rows, cross_rows


def summarize_rows(
    analysis_mod,
    rows: list[dict[str, Any]],
    group_fields: Sequence[str],
    metrics: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    out = []
    for gi, (key, g) in enumerate(df.groupby(list(group_fields), dropna=False)):
        if not isinstance(key, tuple):
            key = (key,)
        row = {k: v for k, v in zip(group_fields, key)}
        row["n"] = int(len(g))
        for mi, metric in enumerate(metrics):
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(np.float64)
            mean, sem, lo, hi = analysis_mod.bootstrap_mean_ci(
                vals,
                n_boot=n_boot,
                seed=seed + 1009 * mi + 7919 * gi,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sem"] = sem
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        out.append(row)
    return out


def trajectory_test_row(
    analysis_mod,
    name: str,
    condition: str,
    values: np.ndarray,
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
    expected_direction: str,
) -> dict[str, Any]:
    vals = np.asarray(values, np.float64)
    mean, sem, lo, hi = analysis_mod.bootstrap_mean_ci(
        vals, n_boot=n_boot, seed=seed
    )
    p = analysis_mod.signflip_pvalue(
        vals, n_perm=n_perm, seed=seed + 17
    )
    return {
        "condition": condition,
        "test": name,
        "expected_direction": expected_direction,
        "n": int(np.isfinite(vals).sum()),
        "mean": mean,
        "sem": sem,
        "ci95_low": lo,
        "ci95_high": hi,
        "paired_signflip_p": p,
    }


def build_trajectory_tests(
    analysis_mod,
    per_model_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
    conditions: Sequence[str],
    blocks: Sequence[int],
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> list[dict[str, Any]]:
    pdf = pd.DataFrame(per_model_rows)
    cdf = pd.DataFrame(cross_rows)
    out: list[dict[str, Any]] = []

    required = {19, 20, 21, 22, 23}
    if not required.issubset(set(blocks)):
        print("[trajectory] B19-B23 not all requested; skipping fixed B21/B23 tests")
        return out

    for ci, cond in enumerate(conditions):
        cq = cdf[cdf["condition"] == cond].pivot(
            index="sample_key",
            columns="block",
            values="cross_model_map_cosine",
        )
        vals = cq[21].to_numpy() - 0.5 * (cq[20].to_numpy() + cq[22].to_numpy())
        out.append(trajectory_test_row(
            analysis_mod,
            "B21_cross_model_cosine_vs_immediate_neighbors",
            cond,
            vals,
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed + 1000 * ci + 1,
            expected_direction="positive if B21 is a local convergence maximum",
        ))

        vals = cq[21].to_numpy() - np.mean(
            np.stack([cq[b].to_numpy() for b in (19, 20, 22, 23)], axis=0),
            axis=0,
        )
        out.append(trajectory_test_row(
            analysis_mod,
            "B21_cross_model_cosine_vs_B19_B20_B22_B23_mean",
            cond,
            vals,
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed + 1000 * ci + 2,
            expected_direction="positive if B21 is a broader convergence peak",
        ))

        # Non-text convergence control.
        nq = cdf[cdf["condition"] == cond].pivot(
            index="sample_key",
            columns="block",
            values="cross_model_nontext_cosine",
        )
        vals = nq[21].to_numpy() - 0.5 * (nq[20].to_numpy() + nq[22].to_numpy())
        out.append(trajectory_test_row(
            analysis_mod,
            "B21_nontext_cross_model_cosine_vs_immediate_neighbors",
            cond,
            vals,
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed + 1000 * ci + 3,
            expected_direction="tests whether convergence extends beyond text patches",
        ))

        # Helper for model difference-in-differences.
        def did(metric: str, b0: int = 21, b1: int = 23) -> np.ndarray:
            q = pdf[pdf["condition"] == cond]
            piv = q.pivot_table(
                index="sample_key",
                columns=["model", "block"],
                values=metric,
                aggfunc="first",
            )
            x = piv[("xattn_stripped", b1)] - piv[("xattn_stripped", b0)]
            g = piv[("gmp", b1)] - piv[("gmp", b0)]
            return (x - g).to_numpy(np.float64)

        out.append(trajectory_test_row(
            analysis_mod,
            "B21_to_B23_text_mass_change_DiD_xattn_minus_gmp",
            cond,
            did("text_mass_binary"),
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed + 1000 * ci + 4,
            expected_direction="negative if stripped x-attn disengages from text more strongly",
        ))

        out.append(trajectory_test_row(
            analysis_mod,
            "B21_to_B23_clean_reference_cosine_change_DiD_xattn_minus_gmp",
            cond,
            did("attack_to_clean_cosine"),
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed + 1000 * ci + 5,
            expected_direction="positive if stripped x-attn recovers toward its paired clean map more strongly",
        ))

        out.append(trajectory_test_row(
            analysis_mod,
            "B21_to_B23_shared_clean_cosine_change_DiD_xattn_minus_gmp",
            cond,
            did("attack_to_shared_clean_cosine"),
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed + 1000 * ci + 6,
            expected_direction="positive if stripped x-attn recovers toward the common clean visual reference",
        ))

        out.append(trajectory_test_row(
            analysis_mod,
            "B21_to_B23_shared_clean_support_mass_change_DiD_xattn_minus_gmp",
            cond,
            did("shared_clean_support_mass"),
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed + 1000 * ci + 7,
            expected_direction="positive if stripped x-attn returns more attribution to clean-supported visual patches",
        ))

        out.append(trajectory_test_row(
            analysis_mod,
            "B21_to_B23_register_mass_change_DiD_xattn_minus_gmp",
            cond,
            did("register_mass"),
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed + 1000 * ci + 8,
            expected_direction="descriptive register-routing control",
        ))

    return out


# =============================================================================
# Plotting
# =============================================================================

def _summary_filter(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    q = df
    for k, v in kwargs.items():
        q = q[q[k] == v]
    return q.sort_values("block")


def grad_attention_plot_cross_model(cross_summary: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(cross_summary)
    fig, axes = plt.subplots(1, len(df["condition"].unique()), figsize=(12.8, 4.6), dpi=180, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, cond in zip(axes, sorted(df["condition"].unique())):
        q = _summary_filter(df, condition=cond)
        x = q["block"].to_numpy()
        for metric, label in (
            ("cross_model_map_cosine", "full map"),
            ("cross_model_nontext_cosine", "non-text patches"),
        ):
            y = q[f"{metric}_mean"].to_numpy()
            lo = q[f"{metric}_ci95_low"].to_numpy()
            hi = q[f"{metric}_ci95_high"].to_numpy()
            ax.plot(x, y, marker="o", label=label)
            ax.fill_between(x, lo, hi, alpha=0.16)
        ax.axvline(21, linestyle="--", linewidth=1)
        ax.set_title(cond)
        ax.set_xlabel("ViT block")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("GmP vs stripped-xattn grad×attention cosine")
    axes[-1].legend()
    fig.suptitle("Cross-model convergence of target-specific late grad×attention", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_model_metric(
    metric_summary: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    out: Path,
) -> None:
    df = pd.DataFrame(metric_summary)
    conditions = sorted(df["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(12.8, 4.6), dpi=180, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, cond in zip(axes, conditions):
        for model in MODEL_ORDER:
            q = _summary_filter(df, condition=cond, model=model)
            x = q["block"].to_numpy()
            y = q[f"{metric}_mean"].to_numpy()
            lo = q[f"{metric}_ci95_low"].to_numpy()
            hi = q[f"{metric}_ci95_high"].to_numpy()
            ax.plot(x, y, marker="o", label=model)
            ax.fill_between(x, lo, hi, alpha=0.16)
        ax.axvline(21, linestyle="--", linewidth=1)
        ax.set_title(cond)
        ax.set_xlabel("ViT block")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend()
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_paper_summary(
    metric_summary: list[dict[str, Any]],
    cross_summary: list[dict[str, Any]],
    out: Path,
) -> None:
    mdf = pd.DataFrame(metric_summary)
    cdf = pd.DataFrame(cross_summary)
    conditions = sorted(mdf["condition"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0), dpi=190)

    # A: cross-model full cosine
    for cond in conditions:
        q = _summary_filter(cdf, condition=cond)
        axes[0, 0].plot(
            q["block"], q["cross_model_map_cosine_mean"],
            marker="o", label=cond,
        )
    axes[0, 0].set_title("Cross-model target-map similarity")
    axes[0, 0].set_ylabel("cosine")

    # B: text mass, use RTA solid conceptual comparison by model; both conditions as separate lines.
    for cond in conditions:
        for model in MODEL_ORDER:
            q = _summary_filter(mdf, condition=cond, model=model)
            axes[0, 1].plot(
                q["block"], q["text_mass_binary_mean"],
                marker="o", label=f"{cond} | {model}",
            )
    axes[0, 1].set_title("Attribution on attack-text patches")
    axes[0, 1].set_ylabel("fraction of positive grad×attention")

    # C: clean map recovery
    for cond in conditions:
        for model in MODEL_ORDER:
            q = _summary_filter(mdf, condition=cond, model=model)
            axes[1, 0].plot(
                q["block"], q["attack_to_clean_cosine_mean"],
                marker="o", label=f"{cond} | {model}",
            )
    axes[1, 0].set_title("Recovery toward paired NoRTA target map")
    axes[1, 0].set_ylabel("attack-to-clean cosine")

    # D: shared clean support mass
    for cond in conditions:
        for model in MODEL_ORDER:
            q = _summary_filter(mdf, condition=cond, model=model)
            axes[1, 1].plot(
                q["block"], q["shared_clean_support_mass_mean"],
                marker="o", label=f"{cond} | {model}",
            )
    axes[1, 1].set_title("Attribution on shared clean-support patches")
    axes[1, 1].set_ylabel("fraction of positive grad×attention")

    for ax in axes.flat:
        ax.axvline(21, linestyle="--", linewidth=1)
        ax.set_xlabel("ViT block")
        ax.grid(alpha=0.25)

    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=6.5)
    fig.suptitle("Late visual readout trajectory under typographic attack", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def grad_attention_main() -> None:
    global analysis_mod

    ap = argparse.ArgumentParser(
        description="Statistical B18-B23 grad×attention trajectory on RTA-100 triplets."
    )
    ap.add_argument("--gmp", default=DEFAULT_GMP)
    ap.add_argument("--xattn", default=DEFAULT_XATTN)
    ap.add_argument("--gmp-module", default=DEFAULT_GMP_MODULE)
    ap.add_argument("--module-root", default=".")
    ap.add_argument("--dataset-repo", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--blocks", default=GRAD_ATTENTION_DEFAULT_BLOCKS)
    ap.add_argument("--limit", type=int, default=0, help="0 = all paired triplets")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--text-batch-size", type=int, default=128)
    ap.add_argument("--prompt-template", default="a photo of a {label}")
    ap.add_argument("--mask-source", choices=("diff", "bbox", "union", "intersection"), default="diff")
    ap.add_argument("--pixel-diff-threshold", type=float, default=0.05)
    ap.add_argument("--diff-min-pixels", type=int, default=4)
    ap.add_argument("--diff-max-fraction", type=float, default=0.35)
    ap.add_argument("--text-patch-min-coverage", type=float, default=0.005)
    ap.add_argument("--clean-support-fraction", type=float, default=0.20)
    ap.add_argument("--register-threshold", type=float, default=60.0)
    ap.add_argument("--min-registers", type=int, default=1)
    ap.add_argument("--max-registers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--validation-cos-min", type=float, default=0.99999)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--output-dir", default=GRAD_ATTENTION_DEFAULT_OUTPUT)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        try:
            torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
        except Exception:
            pass

    device = torch.device(args.device)
    conditions = parse_strs(args.conditions)
    blocks = grad_attention_parse_blocks(args.blocks)
    if not blocks:
        raise ValueError("No blocks requested")
    if min(blocks) < 0 or max(blocks) > 23:
        raise ValueError(f"Expected ViT-L/14 blocks 0..23, got {blocks}")
    if 23 not in blocks:
        print("[note] B23 is not in requested map blocks; forward still reaches B23 for final score/norms.")

    out = Path(args.output_dir)
    data_dir = out / "data"
    plot_dir = out / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Dataset: use the already-fixed row-subset + canonical-ID pairing helpers.
    # -------------------------------------------------------------------------
    # Pairing helper in the prior audited script expects the full triplet bank,
    # so load all three row subsets even if --conditions later selects only one
    # attacked condition for this analysis.
    resolved_config, dsets = analysis_mod.load_rta_triplet_subsets(
        args.dataset_repo,
        args.split,
        ("NoRTA", "RTA", "SynthRTA"),
    )
    pair_key = analysis_mod.infer_pair_key(dsets, "auto")
    paired = analysis_mod.build_paired_rows(
        dsets, pair_key, args.limit, args.seed
    )
    if not paired:
        raise RuntimeError("No paired RTA triplets found")

    labels = [
        str(dsets["NoRTA"][p["NoRTA_index"]]["object_label"]).strip()
        for p in paired
    ]

    manifest_rows = []
    for i, p in enumerate(paired):
        row = {
            "sample_index": i,
            "sample_key": p["sample_key"],
            "pairing_method": p.get("pairing_method", ""),
            "object_label": labels[i],
        }
        for cond in ("NoRTA",) + tuple(conditions):
            rr = dsets[cond][p[f"{cond}_index"]]
            row[f"{cond}_id"] = str(rr.get("id", ""))
            row[f"{cond}_attack_word"] = str(rr.get("attack_word", ""))
        manifest_rows.append(row)
    save_csv(data_dir / "sample_manifest.csv", manifest_rows)

    # -------------------------------------------------------------------------
    # Load GmP; build shared text bank and derive preprocessing geometry.
    # -------------------------------------------------------------------------
    if args.module_root and args.module_root != ".":
        sys.path.insert(0, str(Path(args.module_root).resolve()))

    clip_mod = importlib.import_module(args.gmp_module)
    print("[model] loading GmP")
    gmp_model, preprocess = analysis_mod.load_gmp_model(args.gmp_module, args.gmp)

    print("[text] building shared GmP object-label embedding bank")
    text_bank = build_text_bank(
        gmp_model,
        clip_mod,
        labels,
        args.prompt_template,
        device,
        args.text_batch_size,
    )

    # Input/patch geometry from actual model.
    visual = gmp_model.visual
    patch_size = int(
        visual.conv1.kernel_size[0]
        if isinstance(visual.conv1.kernel_size, tuple)
        else visual.conv1.kernel_size
    )
    input_size = int(getattr(visual, "input_resolution", 224))
    if input_size % patch_size:
        raise RuntimeError(f"input_resolution={input_size}, patch_size={patch_size}")
    grid = input_size // patch_size
    if grid * grid != 256:
        raise RuntimeError(
            f"This script expects ViT-L/14@224 = 256 patches; got grid {grid}x{grid}"
        )

    # -------------------------------------------------------------------------
    # Exact text masks from paired pixel diff, bbox as audit/fallback.
    # -------------------------------------------------------------------------
    print("[mask] building attack-text patch coverage from paired NoRTA diff")
    text_coverage, mask_audit = build_text_patch_masks(
        analysis_mod,
        dsets,
        paired,
        preprocess,
        conditions,
        diff_threshold=args.pixel_diff_threshold,
        diff_min_pixels=args.diff_min_pixels,
        diff_max_fraction=args.diff_max_fraction,
        patch_size=patch_size,
        input_size=input_size,
        mask_source=args.mask_source,
    )
    save_csv(data_dir / "text_mask_audit.csv", mask_audit)
    np.savez_compressed(
        data_dir / "text_patch_coverage.npz",
        **{cond: arr for cond, arr in text_coverage.items()},
    )

    # -------------------------------------------------------------------------
    # GmP maps.
    # -------------------------------------------------------------------------
    print("[grad×attn] GmP")
    gmp_maps, gmp_scores, gmp_norms = process_model_maps(
        "gmp",
        gmp_model,
        preprocess,
        dsets,
        paired,
        conditions,
        labels,
        text_bank,
        blocks,
        batch_size=args.batch_size,
        device=device,
        validation_cos_min=args.validation_cos_min,
    )

    # Free GmP model before constructing the stripped trained backbone.
    gmp_model = gmp_model.to("cpu")
    del gmp_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Clean shell + trained visual weights only.
    # -------------------------------------------------------------------------
    print("[model] constructing stripped x-attn visual backbone in clean GmP shell")
    stripped_model, _ = analysis_mod.load_gmp_model(args.gmp_module, args.gmp)

    xstate = analysis_mod.load_xattn_weight_source(args.xattn)
    audit = analysis_mod.transplant_visual_state(
        stripped_model.visual,
        xstate,
    )
    for row in audit:
        row["source_checkpoint"] = args.xattn
        row["target_clean_shell"] = args.gmp
        row["manual_bare_visual_forward"] = True
        row["rn_inserted"] = False
        row["bridge_called"] = False
    save_csv(data_dir / "model_transplant_audit.csv", audit)

    del xstate
    gc.collect()

    print("[grad×attn] stripped x-attn visual")
    x_maps, x_scores, x_norms = process_model_maps(
        "xattn_stripped",
        stripped_model,
        preprocess,
        dsets,
        paired,
        conditions,
        labels,
        text_bank,
        blocks,
        batch_size=args.batch_size,
        device=device,
        validation_cos_min=args.validation_cos_min,
    )

    stripped_model = stripped_model.to("cpu")
    del stripped_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    maps_by_model = {
        "gmp": gmp_maps,
        "xattn_stripped": x_maps,
    }
    scores_by_model = {
        "gmp": gmp_scores,
        "xattn_stripped": x_scores,
    }

    # Fixed NoRTA register addresses, separately for each model.
    register_masks = {
        "gmp": select_register_indices_np(
            gmp_norms,
            args.register_threshold,
            args.min_registers,
            args.max_registers,
        ),
        "xattn_stripped": select_register_indices_np(
            x_norms,
            args.register_threshold,
            args.min_registers,
            args.max_registers,
        ),
    }

    # -------------------------------------------------------------------------
    # Metrics.
    # -------------------------------------------------------------------------
    print("[metrics] per-model + cross-model")
    per_model_rows, cross_rows = compute_metrics(
        paired,
        labels,
        blocks,
        conditions,
        maps_by_model,
        scores_by_model,
        register_masks,
        text_coverage,
        text_patch_min_coverage=args.text_patch_min_coverage,
        support_fraction=args.clean_support_fraction,
    )
    save_csv(data_dir / "per_sample_gradattn_metrics.csv", per_model_rows)
    save_csv(data_dir / "cross_model_gradattn_metrics.csv", cross_rows)

    metric_names = [
        "target_score",
        "text_mass_binary",
        "text_mass_coverage_weighted",
        "register_mass",
        "attack_to_clean_cosine",
        "attack_to_clean_nontext_cosine",
        "attack_to_shared_clean_cosine",
        "attack_to_shared_clean_nontext_cosine",
        "shared_clean_support_mass",
        "map_entropy",
        "effective_patch_count",
    ]
    metric_summary = summarize_rows(
        analysis_mod,
        per_model_rows,
        ["model", "condition", "block"],
        metric_names,
        n_boot=args.bootstrap,
        seed=args.seed + 100,
    )
    save_csv(data_dir / "metric_summary.csv", metric_summary)

    cross_metric_names = [
        "cross_model_map_cosine",
        "cross_model_nontext_cosine",
        "cross_model_text_only_cosine",
        "abs_text_mass_difference",
        "xattn_minus_gmp_text_mass",
        "abs_register_mass_difference",
        "xattn_minus_gmp_register_mass",
    ]
    cross_summary = summarize_rows(
        analysis_mod,
        cross_rows,
        ["condition", "block"],
        cross_metric_names,
        n_boot=args.bootstrap,
        seed=args.seed + 200,
    )
    save_csv(data_dir / "cross_model_summary.csv", cross_summary)

    trajectory = build_trajectory_tests(
        analysis_mod,
        per_model_rows,
        cross_rows,
        conditions,
        blocks,
        n_boot=args.bootstrap,
        n_perm=args.permutations,
        seed=args.seed + 300,
    )
    save_csv(data_dir / "trajectory_tests.csv", trajectory)

    # Raw forensic data, excluded from compact handoff.
    npz_payload = {}
    for model in MODEL_ORDER:
        for cond in ("NoRTA",) + tuple(conditions):
            npz_payload[f"{model}__{cond}__gradattn"] = maps_by_model[model][cond]
            npz_payload[f"{model}__{cond}__target_score"] = scores_by_model[model][cond]
        npz_payload[f"{model}__NoRTA__final_patch_norm"] = (
            gmp_norms if model == "gmp" else x_norms
        )
        npz_payload[f"{model}__NoRTA__register_mask"] = register_masks[model].astype(np.uint8)
    np.savez_compressed(data_dir / "gradattn_maps.npz", **npz_payload)

    # -------------------------------------------------------------------------
    # Plots.
    # -------------------------------------------------------------------------
    grad_attention_plot_cross_model(
        cross_summary,
        plot_dir / "01_cross_model_gradattn_cosine.png",
    )
    plot_model_metric(
        metric_summary,
        "text_mass_binary",
        "fraction of positive grad×attention",
        "Attribution on attack-text patches",
        plot_dir / "02_text_attribution_fraction.png",
    )
    plot_model_metric(
        metric_summary,
        "attack_to_clean_cosine",
        "attack-to-clean map cosine",
        "Recovery toward paired NoRTA target map",
        plot_dir / "03_clean_reference_recovery.png",
    )
    plot_model_metric(
        metric_summary,
        "shared_clean_support_mass",
        "fraction of positive grad×attention",
        "Attribution on shared clean-reference support",
        plot_dir / "04_shared_clean_support_mass.png",
    )
    plot_model_metric(
        metric_summary,
        "register_mass",
        "fraction of positive grad×attention",
        "Attribution on fixed NoRTA register addresses",
        plot_dir / "05_register_attribution_fraction.png",
    )
    plot_paper_summary(
        metric_summary,
        cross_summary,
        plot_dir / "06_paper_trajectory_summary.png",
    )

    # -------------------------------------------------------------------------
    # Config + summary.
    # -------------------------------------------------------------------------
    config = {
        "gmp_checkpoint": args.gmp,
        "xattn_checkpoint": args.xattn,
        "dataset_repo": args.dataset_repo,
        "dataset_config": resolved_config,
        "split": args.split,
        "n_triplets": len(paired),
        "conditions": conditions,
        "blocks": blocks,
        "prompt_template": args.prompt_template,
        "target_text_encoder": "shared GmP text encoder for both visual models",
        "gradattn_definition": "mean_heads(ReLU(A*d cosine(object)/dA))[CLS->patch]",
        "mask_source": args.mask_source,
        "pixel_diff_threshold_normalized_input": args.pixel_diff_threshold,
        "text_patch_min_coverage": args.text_patch_min_coverage,
        "clean_support_fraction": args.clean_support_fraction,
        "register_threshold": args.register_threshold,
        "register_pair_policy": "each model uses fixed NoRTA B23 register addresses",
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "tf32": bool(args.tf32),
        "manual_forward_validation": "explicit attention vs native bare visual blocks; root encode_image intentionally bypassed",
        "manual_forward_validation_cos_min": args.validation_cos_min,
        "scratchpads": "not recomputed; intentionally out of scope",
    }
    save_json(data_dir / "config.json", config)

    tdf = pd.DataFrame(trajectory)
    lines = [
        "RTA LATE TARGET-SPECIFIC GRADxATTENTION TRAJECTORY",
        "=" * 78,
        "",
        f"triplets: {len(paired)}",
        f"conditions: {conditions}",
        f"blocks: {blocks}",
        "",
        "Target:",
        f"  {args.prompt_template}",
        "  shared GmP text encoder for both visual models",
        "",
        "Text mask:",
        f"  source={args.mask_source}; pixel-diff threshold={args.pixel_diff_threshold}",
        "  diff is computed AFTER identical CLIP preprocessing, hence patch-aligned.",
        "",
        "Primary trajectory tests:",
    ]
    if len(tdf):
        for _, r in tdf.iterrows():
            lines.append(
                f"  {r['condition']} | {r['test']}: "
                f"{r['mean']:+.6f} "
                f"[{r['ci95_low']:+.6f},{r['ci95_high']:+.6f}] "
                f"p={r['paired_signflip_p']:.6g}"
            )
    else:
        lines.append("  (fixed B19-B23 tests skipped because required blocks were not all requested)")

    lines += [
        "",
        "Interpretation guardrails:",
        "  * text attribution is grounded by the paired changed-pixel mask;",
        "  * 'clean visual recovery' means recovery toward paired NoRTA target attribution,",
        "    not an externally annotated object segmentation;",
        "  * positive local grad×attention is block-specific causal attribution, not",
        "    a recursive attention rollout and not a complete causal decomposition;",
        "  * register mass uses fixed NoRTA register addresses; scratchpads are omitted.",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Compact summary archive: everything useful except raw maps / mask arrays.
    handoff = [
        data_dir / "sample_manifest.csv",
        data_dir / "text_mask_audit.csv",
        data_dir / "model_transplant_audit.csv",
        data_dir / "per_sample_gradattn_metrics.csv",
        data_dir / "cross_model_gradattn_metrics.csv",
        data_dir / "metric_summary.csv",
        data_dir / "cross_model_summary.csv",
        data_dir / "trajectory_tests.csv",
        data_dir / "config.json",
        out / "SUMMARY.txt",
    ] + sorted(plot_dir.glob("*.png"))

    zpath = out / "compact_summary_workspace_register_geometry_grad_attention.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for p in handoff:
            if p.exists():
                z.write(p, arcname=p.relative_to(out).as_posix())

    print("\n[done]")
    print("  output:", out.resolve())
    print("  compact summary:", zpath.resolve())
    print("  local raw maps:", (data_dir / "gradattn_maps.npz").resolve())


# PRINCIPAL ANGLES
from typing import Any

matplotlib.use("Agg")


DEFAULT_INPUT = r"out_rta_register_secondary_mode/data/register_means.npz"
PRINCIPAL_ANGLES_DEFAULT_OUTPUT = r"out_rta_register_secondary_mode/residual_subspace_b20_b22"
MODELS = ("gmp", "xattn_stripped")
CONDITIONS = ("NoRTA", "RTA", "SynthRTA")
BLOCKS = (20, 21, 22)


def principal_angles_fit_uncentered_basis(x: np.ndarray, rank: int = 4):
    """
    Exact top right-singular vectors via the smaller Gram matrix.
    Returns orthonormal basis [rank,D], singular values, total energy.
    """
    x = np.asarray(x, np.float64)
    n, d = x.shape
    k = min(rank, n, d)
    total = float(np.sum(x * x))

    if n <= d:
        gram = x @ x.T
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1][:k]
        evals = np.clip(evals[order], 0.0, None)
        left = evecs[:, order]
        s = np.sqrt(evals)
        basis = []
        for j in range(k):
            if s[j] <= 1e-12:
                basis.append(np.zeros(d, np.float64))
            else:
                basis.append((left[:, j].T @ x) / s[j])
        basis = unit_rows(np.stack(basis, axis=0))
    else:
        gram = x.T @ x
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1][:k]
        evals = np.clip(evals[order], 0.0, None)
        s = np.sqrt(evals)
        basis = unit_rows(evecs[:, order].T)

    # Deterministic signs for individual-vector reporting only.
    mean = x.mean(axis=0)
    for j in range(len(basis)):
        if float(basis[j] @ mean) < 0:
            basis[j] *= -1
    return basis, s, total


def abs_cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return abs(float(a @ b / den)) if den > 1e-12 else float("nan")


def principal_cosines(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A,B are row-orthonormal bases. QR defensively re-orthogonalizes.
    """
    qa = np.linalg.qr(np.asarray(A, np.float64).T, mode="reduced")[0]
    qb = np.linalg.qr(np.asarray(B, np.float64).T, mode="reduced")[0]
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def projection_fraction(v: np.ndarray, basis: np.ndarray) -> float:
    v = np.asarray(v, np.float64)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    B = unit_rows(np.asarray(basis, np.float64))
    return float(np.sum((B @ v) ** 2))


def principal_angles_plot_cross_model(rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), dpi=180, sharey=True)

    for ax, cond in zip(axes, CONDITIONS):
        q = df[df["condition"] == cond].sort_values("block")
        x = q["block"].to_numpy()

        ax.plot(x, q["pc2_abs_cosine"], marker="o", label="individual PC2")
        ax.plot(
            x, q["residual_rank2_principal_cosine_min"],
            marker="s", label="span(PC2,PC3): min principal cosine",
        )
        ax.plot(
            x, q["residual_rank3_principal_cosine_min"],
            marker="^", label="span(PC2,PC3,PC4): min principal cosine",
        )
        ax.plot(
            x, q["residual_rank3_principal_cosine_mean"],
            marker="x", linestyle="--",
            label="rank-3 residual: mean principal cosine",
        )

        ax.set_title(cond)
        ax.set_xlabel("ViT block")
        ax.set_xticks(BLOCKS)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("GmP vs stripped-trained alignment")
    axes[-1].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(
        "B20-B22 residual register-subspace alignment after removing PC1",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_temporal(rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), dpi=180, sharey=True)

    transitions = ("B20->B21", "B21->B22")
    for col, cond in enumerate(CONDITIONS):
        for row_i, trans in enumerate(transitions):
            ax = axes[row_i, col]
            q = df[
                (df["condition"] == cond)
                & (df["transition"] == trans)
            ]
            labels = []
            vals2 = []
            vals3 = []
            for model in MODELS:
                z = q[q["model"] == model]
                if len(z):
                    labels.append(model)
                    vals2.append(float(z.iloc[0]["residual_rank2_principal_cosine_min"]))
                    vals3.append(float(z.iloc[0]["residual_rank3_principal_cosine_min"]))
            xp = np.arange(len(labels))
            w = 0.34
            ax.bar(xp - w/2, vals2, width=w, label="rank-2 residual")
            ax.bar(xp + w/2, vals3, width=w, label="rank-3 residual")
            ax.set_xticks(xp)
            ax.set_xticklabels(labels, rotation=15)
            ax.set_ylim(0.0, 1.02)
            ax.set_title(f"{cond} | {trans}")
            ax.grid(axis="y", alpha=0.25)

    axes[0, 0].set_ylabel("minimum principal cosine")
    axes[1, 0].set_ylabel("minimum principal cosine")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Within-model temporal stability of post-PC1 residual subspaces", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def principal_angles_main() -> None:
    ap = argparse.ArgumentParser(
        description="B20-B22 principal-angle diagnostic for RTA register residual subspaces."
    )
    ap.add_argument("--register-means", default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", default=PRINCIPAL_ANGLES_DEFAULT_OUTPUT)
    args = ap.parse_args()

    src = Path(args.register_means)
    if not src.is_file():
        raise FileNotFoundError(
            f"{src} does not exist. Re-run probe_register_geometry.py secondary "
            "once (it saves data/register_means.npz locally), or pass --register-means."
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = np.load(src)
    arrays: dict[tuple[str, str, int], np.ndarray] = {}
    missing = []
    for model in MODELS:
        for cond in CONDITIONS:
            for b in BLOCKS:
                key = f"{model}__{cond}__B{b}"
                if key not in raw:
                    missing.append(key)
                else:
                    arrays[(model, cond, b)] = np.asarray(raw[key], np.float64)
    if missing:
        raise KeyError(f"Missing expected NPZ arrays: {missing}")

    bases: dict[tuple[str, str, int], np.ndarray] = {}
    singular: dict[tuple[str, str, int], np.ndarray] = {}
    spectrum_rows = []

    for model in MODELS:
        for cond in CONDITIONS:
            for b in BLOCKS:
                X = arrays[(model, cond, b)]
                basis, s, total = principal_angles_fit_uncentered_basis(X, rank=4)
                bases[(model, cond, b)] = basis
                singular[(model, cond, b)] = s

                residual_energy = max(total - float(s[0] ** 2), 1e-12)
                spectrum_rows.append({
                    "model": model,
                    "condition": cond,
                    "block": b,
                    "n": int(len(X)),
                    "sigma1": float(s[0]),
                    "sigma2": float(s[1]),
                    "sigma3": float(s[2]),
                    "sigma4": float(s[3]),
                    "sigma2_over_sigma3": float(s[1] / max(s[2], 1e-12)),
                    "sigma3_over_sigma4": float(s[2] / max(s[3], 1e-12)),
                    "pc2_share_post_pc1": float(s[1] ** 2 / residual_energy),
                    "pc3_share_post_pc1": float(s[2] ** 2 / residual_energy),
                    "pc4_share_post_pc1": float(s[3] ** 2 / residual_energy),
                })

    cross_rows = []
    for cond in CONDITIONS:
        for b in BLOCKS:
            A = bases[("gmp", cond, b)]
            B = bases[("xattn_stripped", cond, b)]

            pc2 = abs_cos(A[1], B[1])
            p2 = principal_cosines(A[1:3], B[1:3])
            p3 = principal_cosines(A[1:4], B[1:4])

            cross_rows.append({
                "condition": cond,
                "block": b,
                "pc2_abs_cosine": pc2,

                "residual_rank2_principal_cosine_1": float(p2[0]),
                "residual_rank2_principal_cosine_2": float(p2[1]),
                "residual_rank2_principal_cosine_mean": float(np.mean(p2)),
                "residual_rank2_principal_cosine_min": float(np.min(p2)),

                "residual_rank3_principal_cosine_1": float(p3[0]),
                "residual_rank3_principal_cosine_2": float(p3[1]),
                "residual_rank3_principal_cosine_3": float(p3[2]),
                "residual_rank3_principal_cosine_mean": float(np.mean(p3)),
                "residual_rank3_principal_cosine_min": float(np.min(p3)),

                # Is the "lost" PC2 still inside the other model's residual space?
                "gmp_pc2_in_xattn_rank2_fraction": projection_fraction(A[1], B[1:3]),
                "xattn_pc2_in_gmp_rank2_fraction": projection_fraction(B[1], A[1:3]),
                "gmp_pc2_in_xattn_rank3_fraction": projection_fraction(A[1], B[1:4]),
                "xattn_pc2_in_gmp_rank3_fraction": projection_fraction(B[1], A[1:4]),
            })

    temporal_rows = []
    for model in MODELS:
        for cond in CONDITIONS:
            for ba, bb in ((20, 21), (21, 22)):
                A = bases[(model, cond, ba)]
                B = bases[(model, cond, bb)]
                p2 = principal_cosines(A[1:3], B[1:3])
                p3 = principal_cosines(A[1:4], B[1:4])
                temporal_rows.append({
                    "model": model,
                    "condition": cond,
                    "block_a": ba,
                    "block_b": bb,
                    "transition": f"B{ba}->B{bb}",
                    "pc2_abs_cosine": abs_cos(A[1], B[1]),
                    "residual_rank2_principal_cosine_1": float(p2[0]),
                    "residual_rank2_principal_cosine_2": float(p2[1]),
                    "residual_rank2_principal_cosine_mean": float(np.mean(p2)),
                    "residual_rank2_principal_cosine_min": float(np.min(p2)),
                    "residual_rank3_principal_cosine_1": float(p3[0]),
                    "residual_rank3_principal_cosine_2": float(p3[1]),
                    "residual_rank3_principal_cosine_3": float(p3[2]),
                    "residual_rank3_principal_cosine_mean": float(np.mean(p3)),
                    "residual_rank3_principal_cosine_min": float(np.min(p3)),
                })

    save_csv(out / "residual_subspace_principal_angles.csv", cross_rows)
    save_csv(out / "residual_subspace_spectrum.csv", spectrum_rows)
    save_csv(out / "temporal_residual_subspace_alignment.csv", temporal_rows)

    npz_payload = {}
    for (model, cond, b), basis in bases.items():
        npz_payload[f"{model}__{cond}__B{b}__basis_rank4"] = basis.astype(np.float32)
        npz_payload[f"{model}__{cond}__B{b}__singular_values"] = singular[(model, cond, b)].astype(np.float32)
    np.savez_compressed(out / "residual_subspace_bases.npz", **npz_payload)

    principal_angles_plot_cross_model(
        cross_rows,
        out / "residual_subspace_principal_angles_b20_b22.png",
    )
    plot_temporal(
        temporal_rows,
        out / "residual_subspace_temporal_alignment_b20_b22.png",
    )

    cdf = pd.DataFrame(cross_rows)
    sdf = pd.DataFrame(spectrum_rows)
    lines = [
        "B20-B22 POST-PC1 RESIDUAL SUBSPACE PRINCIPAL-ANGLE DIAGNOSTIC",
        "=" * 78,
        "",
        f"input: {src}",
        "",
        "Definitions:",
        "  residual rank-2 = span(PC2, PC3)",
        "  residual rank-3 = span(PC2, PC3, PC4)",
        "",
    ]
    for cond in CONDITIONS:
        lines.append(cond)
        q = cdf[cdf["condition"] == cond].sort_values("block")
        for _, r in q.iterrows():
            lines.append(
                f"  B{int(r.block)}: PC2={r.pc2_abs_cosine:.6f} | "
                f"rank2 min={r.residual_rank2_principal_cosine_min:.6f} | "
                f"rank3 min={r.residual_rank3_principal_cosine_min:.6f} | "
                f"G-PC2 in X-r3={r.gmp_pc2_in_xattn_rank3_fraction:.6f} | "
                f"X-PC2 in G-r3={r.xattn_pc2_in_gmp_rank3_fraction:.6f}"
            )
        lines.append("")

    lines += [
        "Interpretation:",
        "  If B21 individual PC2 cosine drops while residual rank-2/rank-3 principal",
        "  cosines remain high and PC2 projects strongly into the opposite residual",
        "  subspace, the anomaly is predominantly basis rotation / near-degeneracy.",
        "  If the smaller principal cosines also fall, B21 contains a genuine",
        "  cross-model residual-subspace divergence.",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done]", out.resolve())
    print("  cross-model:", (out / "residual_subspace_principal_angles.csv").resolve())
    print("  spectrum:", (out / "residual_subspace_spectrum.csv").resolve())
    print("  temporal:", (out / "temporal_residual_subspace_alignment.csv").resolve())
    print("  figure:", (out / "residual_subspace_principal_angles_b20_b22.png").resolve())


from types import SimpleNamespace
analysis_mod = SimpleNamespace(bootstrap_mean_ci=bootstrap_mean_ci, build_paired_rows=build_paired_rows, infer_pair_key=infer_pair_key, load_gmp_model=load_gmp_model, load_rta_triplet_subsets=load_rta_triplet_subsets, load_xattn_weight_source=load_xattn_weight_source, row_image=row_image, signflip_pvalue=signflip_pvalue, transplant_visual_state=transplant_visual_state)


def main(argv=None):
    """Dispatch a workflow; each subcommand retains its original CLI options."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {'secondary': secondary_main, 'grad_attention': grad_attention_main, 'principal_angles': principal_angles_main}
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

