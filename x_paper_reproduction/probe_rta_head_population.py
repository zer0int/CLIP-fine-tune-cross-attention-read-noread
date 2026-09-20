#!/usr/bin/env python3
r"""RTA head-population RN factorial and exact OV response motifs.
Commands: analyze (collect or --reuse_raw), contact_sheets (render existing motif CSVs).
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_analysis import (RNPayload, clear_cuda, r2_score, safe_corr)

# ANALYZE
import argparse
import gc
import importlib
import io
import json
import math
import os
import random
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm.auto import tqdm

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

# Project CLIP modules are imported lazily.  They are the only project-specific
# imports used by this file; no helper .py script is imported or executed.
clip = None
orgclip = None


def ensure_clip_modules() -> None:
    global clip, orgclip
    if clip is None:
        import attnclip_mechinterp_xattn as _clip
        clip = _clip
    if orgclip is None:
        import attnclip_mechinterp_sae as _orgclip
        orgclip = _orgclip


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_DATASET = "zer0int/RTA-100-Triplet"
DEFAULT_SPLIT = "train"
CONDITION_ORDER = ("NoRTA", "RTA", "SynthRTA")
ATTACK_ORDER = ("RTA", "SynthRTA")
RN_MODE_ORDER = ("rn_off", "rn_on")
RN_INSERT_BLOCK = 13

DEFAULT_ORACLE_ROOT = r"cls_gipu_exchange_no_rn"
DEFAULT_BASELINE_DIR = r"rta100_mu2_head_population_grammar"
DEFAULT_OUT = r"rta100_mu2_head_population_grammar_with_RN"
DEFAULT_GMP_CHECKPOINT = "GMP_CHECKPOINT.pt"
DEFAULT_XATTN_CHECKPOINT = "REPLACE_WITH_CHECKPOINT.pt"
MODEL_ORDER = ("pretrained", "gmp", "finetune_stripped")

POPULATIONS = (
    "REG_FROZEN",
    "REG_OWN",
    "TEXT",
    "NONREG_NONTEXT",
    "MU1_CACHE",
    "SCRATCH_RESIDUAL",
    "ORDINARY",
)

EPS = 1e-12


# =============================================================================
# Generic helpers
# =============================================================================

@dataclass
class ModelBundle:
    model: torch.nn.Module
    preprocess: Any
    device: torch.device
    name: str
    source: str


def parse_floats(text: str) -> list[float]:
    return sorted(set(float(x.strip()) for x in str(text).split(",") if x.strip()))


def parse_blocks(text: str, n_blocks: int = 24) -> list[int]:
    value = str(text).strip().lower()
    if value in {"all", "*"}:
        return list(range(n_blocks))
    out: set[int] = set()
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.update(range(a, b + step, step))
        else:
            out.add(int(tok))
    bad = sorted(b for b in out if b < 0 or b >= n_blocks)
    if bad:
        raise ValueError(f"Invalid block(s) {bad}; valid range is 0..{n_blocks-1}")
    return sorted(out)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def savefig(fig, path: Path, dpi: int = 190) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def safe_div(a, b):
    return np.asarray(a, np.float64) / np.maximum(np.asarray(b, np.float64), EPS)


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, np.float64)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    idx = np.flatnonzero(finite)
    if len(idx) == 0:
        return q
    vals = p[idx]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    back = np.empty_like(adjusted)
    back[order] = adjusted
    q[idx] = back
    return q


def paired_effect_stats(delta: np.ndarray) -> dict[str, float]:
    d = np.asarray(delta, np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {
            "n": 0,
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "std_delta": np.nan,
            "cohen_dz": np.nan,
            "t_p": np.nan,
            "wilcoxon_p": np.nan,
        }
    mean = float(np.mean(d))
    median = float(np.median(d))
    std = float(np.std(d, ddof=1)) if n > 1 else 0.0
    dz = mean / std if std > EPS else (np.inf if abs(mean) > EPS else 0.0)
    if scipy_stats is not None and n > 1:
        try:
            t_p = float(scipy_stats.ttest_1samp(d, 0.0, nan_policy="omit").pvalue)
        except Exception:
            t_p = np.nan
        try:
            if np.all(np.abs(d) <= EPS):
                w_p = 1.0
            else:
                w_p = float(
                    scipy_stats.wilcoxon(
                        d,
                        zero_method="wilcox",
                        correction=False,
                        alternative="two-sided",
                    ).pvalue
                )
        except Exception:
            w_p = np.nan
    else:
        t_p = np.nan
        w_p = np.nan
    return {
        "n": int(n),
        "mean_delta": mean,
        "median_delta": median,
        "std_delta": std,
        "cohen_dz": float(dz),
        "t_p": t_p,
        "wilcoxon_p": w_p,
    }


def clear_attn_cache(block) -> None:
    for attr in (
        "last_logits", "last_probs", "last_v", "last_z",
        "last_q", "last_k", "last_xin",
    ):
        if hasattr(block.attn, attr):
            setattr(block.attn, attr, None)


def side_from_patch_count(patches: int) -> int:
    side = int(round(math.sqrt(int(patches))))
    if side * side != int(patches):
        raise ValueError(f"Patch count {patches} is not square.")
    return side


def select_register_mask(
    norms_p: torch.Tensor,
    threshold: float,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    p = int(norms_p.numel())
    out = torch.zeros(p, dtype=torch.bool, device=norms_p.device)
    idx = torch.nonzero(norms_p >= float(threshold), as_tuple=False).flatten()
    if maximum > 0 and idx.numel() > int(maximum):
        idx = idx[torch.topk(norms_p[idx], k=int(maximum)).indices]
    if idx.numel() < int(minimum):
        idx = torch.topk(norms_p, k=min(int(minimum), p)).indices
    out[idx] = True
    return out


def normalize_probs_shape(
    probs0: torch.Tensor,
    batch: int,
    heads: int,
    tokens: int,
) -> torch.Tensor:
    p = probs0
    if p.ndim == 3 and batch == 1 and p.shape[0] == heads:
        return p.unsqueeze(0)
    if p.ndim != 4:
        raise RuntimeError(f"Expected [B,H,T,T] attention probs, got {tuple(p.shape)}")
    if p.shape[0] == batch and p.shape[1] == heads and p.shape[2] == tokens:
        return p
    if p.shape[0] == heads and p.shape[1] == batch and p.shape[2] == tokens:
        return p.permute(1, 0, 2, 3)
    raise RuntimeError(
        f"Cannot normalize attention probs shape {tuple(p.shape)} "
        f"for B={batch}, H={heads}, T={tokens}"
    )


def normalize_qkv_shape(
    x0: torch.Tensor,
    batch: int,
    heads: int,
    tokens: int,
) -> torch.Tensor:
    if x0 is None:
        raise RuntimeError("Q/K/V cache missing; capture-enabled attention is required.")
    t = x0
    if t.ndim != 4:
        raise RuntimeError(f"Expected 4D Q/K/V cache, got {tuple(t.shape)}")
    if t.shape[0] == batch and t.shape[1] == heads and t.shape[2] == tokens:
        return t
    if t.shape[0] == heads and t.shape[1] == batch and t.shape[2] == tokens:
        return t.permute(1, 0, 2, 3)
    if t.shape[0] == tokens and t.shape[1] == batch and t.shape[2] == heads:
        return t.permute(1, 2, 0, 3)
    raise RuntimeError(
        f"Cannot normalize Q/K/V shape {tuple(t.shape)} "
        f"for B={batch}, H={heads}, T={tokens}"
    )


def safe_torch_load(path: Path) -> Any:
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(str(path), weights_only=False, **kwargs)
    except TypeError:
        return torch.load(str(path), **kwargs)


def strip_common_prefixes(state: Mapping[str, Any]) -> dict[str, Any]:
    out = {str(k): v for k, v in state.items()}
    for prefix in ("module.", "model.", "clip."):
        keys = list(out)
        if keys and sum(k.startswith(prefix) for k in keys) >= max(1, int(0.9 * len(keys))):
            out = {
                k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in out.items()
            }
    return out


def looks_like_clip_state(obj: Mapping[str, Any]) -> bool:
    keys = set(map(str, obj.keys()))
    return (
        "visual.conv1.weight" in keys
        and (
            "token_embedding.weight" in keys
            or any(k.startswith("transformer.resblocks.") for k in keys)
            or any(k.startswith("visual.transformer.resblocks.") for k in keys)
        )
    )


def extract_state_dict(obj: Any, source: str) -> dict[str, Any]:
    if isinstance(obj, torch.nn.Module):
        return strip_common_prefixes(obj.state_dict())
    if isinstance(obj, Mapping):
        candidate = strip_common_prefixes(obj)
        if looks_like_clip_state(candidate):
            return candidate
        for key in ("state_dict", "model_state_dict", "model", "clip", "module"):
            if key not in obj:
                continue
            value = obj[key]
            if isinstance(value, torch.nn.Module):
                return strip_common_prefixes(value.state_dict())
            if isinstance(value, Mapping):
                candidate = strip_common_prefixes(value)
                if looks_like_clip_state(candidate):
                    return candidate
    raise TypeError(f"Could not extract a CLIP state_dict from {source}: {type(obj)}")


def freeze_eval(model: torch.nn.Module) -> torch.nn.Module:
    model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def assert_explicit_qkv(model: torch.nn.Module, label: str) -> None:
    attn = model.visual.transformer.resblocks[0].attn
    missing = [name for name in ("q_proj", "k_proj", "v_proj", "out_proj") if not hasattr(attn, name)]
    if missing:
        raise RuntimeError(f"{label}: explicit-QKV capture model required; missing {missing}")


def _import_pickle_compat_modules(args) -> None:
    ensure_clip_modules()
    for name in (args.xattn_module, args.pickle_module, "oaiclip", "clip"):
        if not name:
            continue
        try:
            importlib.import_module(name)
        except Exception:
            pass


def load_rn_payload(args) -> RNPayload:
    _import_pickle_compat_modules(args)
    path = Path(args.xattn_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = safe_torch_load(path)
    state = extract_state_dict(obj, str(path))
    del obj
    key = "visual.read_null_token"
    if key not in state:
        candidates = [k for k in state if k.endswith("visual.read_null_token")]
        if len(candidates) != 1:
            raise KeyError(f"RN token not found uniquely in {path}: {candidates[:10]}")
        key = candidates[0]
    token = state[key].detach().float().cpu().reshape(-1).contiguous()
    cfg_key = "visual.read_null_insert_block_config"
    insert_block = int(state[cfg_key].item()) if cfg_key in state else int(args.rn_insert_block)
    if insert_block != int(args.rn_insert_block):
        raise RuntimeError(
            f"RN checkpoint says pre-B{insert_block}; requested pre-B{args.rn_insert_block}."
        )
    print(f"[RN] dim={token.numel()} norm={float(token.norm()):.6f} insert=pre-B{insert_block}")
    return RNPayload(token=token, insert_block=insert_block, checkpoint=str(path))


def _fresh_pretrained(args, device: torch.device):
    ensure_clip_modules()
    model, preprocess = orgclip.load(args.model_spec, device=device, jit=False)
    model = freeze_eval(model)
    assert_explicit_qkv(model, "pretrained")
    return model, preprocess


def _convert_visual_state_for_capture(source: dict[str, Any]) -> dict[str, Any]:
    ensure_clip_modules()
    try:
        model_mod = importlib.import_module("attnclip_mechinterp_sae.model")
        convert = getattr(model_mod, "convert_state_dict_inproj_to_qkv", None)
        if convert is not None:
            source = convert(dict(source))
    except Exception:
        # If the source already uses explicit q/k/v keys, no conversion is needed.
        pass
    return source


def _strict_visual_transplant(
    model: torch.nn.Module,
    state: Mapping[str, Any],
    audit_path: Path,
    *,
    exclude_rn: bool,
    label: str,
) -> None:
    source = {
        k: v for k, v in state.items()
        if k.startswith("visual.")
        and torch.is_tensor(v)
        and (not exclude_rn or "read_null" not in k.lower())
    }
    source = _convert_visual_state_for_capture(source)
    target = model.state_dict()
    target_visual_keys = [k for k in target if k.startswith("visual.")]
    load_state: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    mismatched: list[dict[str, Any]] = []
    for key in target_visual_keys:
        if key not in source:
            missing.append(key)
            continue
        src = source[key].detach().cpu()
        dst = target[key].detach().cpu()
        if tuple(src.shape) != tuple(dst.shape):
            mismatched.append({
                "key": key,
                "source_shape": list(src.shape),
                "target_shape": list(dst.shape),
            })
            continue
        load_state[key] = src.to(dtype=target[key].dtype)
    audit = {
        "label": label,
        "source_visual_keys": len(source),
        "target_visual_keys": len(target_visual_keys),
        "loaded_visual_keys": len(load_state),
        "missing_target_visual_keys": missing,
        "shape_mismatch": mismatched,
        "exclude_rn": bool(exclude_rn),
    }
    save_json(audit_path, audit)
    if missing or mismatched:
        raise RuntimeError(
            f"Strict visual transplant failed for {label}: missing={len(missing)}, "
            f"mismatched={len(mismatched)}; see {audit_path}"
        )
    merged = model.state_dict()
    merged.update(load_state)
    model.load_state_dict(merged, strict=True)


def load_bundle(model_name: str, args, audit_dir: Path) -> ModelBundle:
    device = torch.device(args.device)
    ensure_clip_modules()

    if model_name == "pretrained":
        model, preprocess = _fresh_pretrained(args, device)
        return ModelBundle(model, preprocess, device, model_name, f"{args.clip_module}:{args.model_spec}")

    if model_name == "gmp":
        checkpoint = Path(args.gmp_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"[model] loading GmP: {checkpoint}")
        try:
            model, preprocess = orgclip.load(str(checkpoint), device=device, jit=False)
            model = freeze_eval(model)
            assert_explicit_qkv(model, "gmp")
            return ModelBundle(model, preprocess, device, model_name, str(checkpoint))
        except Exception as direct_error:
            print(f"[model] orgclip.load(GmP) failed; attempting strict visual transplant: {direct_error}")
            model, preprocess = _fresh_pretrained(args, torch.device("cpu"))
            _import_pickle_compat_modules(args)
            obj = safe_torch_load(checkpoint)
            state = extract_state_dict(obj, str(checkpoint))
            del obj
            _strict_visual_transplant(
                model,
                state,
                audit_dir / "gmp_visual_transplant_audit.json",
                exclude_rn=True,
                label="gmp",
            )
            model = freeze_eval(model.to(device))
            assert_explicit_qkv(model, "gmp")
            return ModelBundle(model, preprocess, device, model_name, str(checkpoint))

    if model_name == "finetune_stripped":
        checkpoint = Path(args.xattn_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"[model] loading all-weights visual backbone from: {checkpoint}")
        model, preprocess = _fresh_pretrained(args, torch.device("cpu"))
        _import_pickle_compat_modules(args)
        obj = safe_torch_load(checkpoint)
        state = extract_state_dict(obj, str(checkpoint))
        del obj
        if any(k.endswith(".theta") or k.endswith(".r") for k in state):
            raise RuntimeError(
                "The all-weights checkpoint still contains GmP theta/r parameters; "
                "expected the materialized _ungmp_ checkpoint."
            )
        _strict_visual_transplant(
            model,
            state,
            audit_dir / "xattn_stripped_visual_transplant_audit.json",
            exclude_rn=True,
            label="finetune_stripped",
        )
        model = freeze_eval(model.to(device))
        assert_explicit_qkv(model, "finetune_stripped")
        return ModelBundle(model, preprocess, device, model_name, str(checkpoint))

    raise ValueError(f"Unknown model {model_name!r}")

# =============================================================================
# Dataset / pairing
# =============================================================================

def load_rta_triplet_subsets(
    repo: str,
    split: str,
    subset_names: Sequence[str],
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "This extraction path requires `datasets` (Hugging Face). "
            "Install it or use --reuse_raw with an existing extraction."
        ) from exc

    ds = load_dataset(repo, split=split)
    required = {"type", "image", "id"}
    missing = sorted(required - set(ds.column_names))
    if missing:
        raise KeyError(
            f"{repo} split={split!r} is missing required columns {missing}; "
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
            f"Missing requested row subset(s) {empty}; observed `type`={seen}"
        )

    return {
        name: ds.select(rows_by_subset[name])
        for name in subset_names
    }


def row_image(row: Mapping[str, Any], image_col: str = "image") -> Image.Image:
    im = row[image_col]
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if isinstance(im, dict):
        if im.get("path"):
            return Image.open(im["path"]).convert("RGB")
        if im.get("bytes"):
            return Image.open(io.BytesIO(im["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image object: {type(im)}")


def canonical_rta_pair_id(raw: Any, variant: str) -> str:
    value = str(raw).strip()
    if not value:
        return ""
    pattern = rf"^{re.escape(str(variant))}[\s_:\-./]*"
    stripped = re.sub(pattern, "", value, count=1, flags=re.IGNORECASE)
    return stripped if stripped else value


def validate_triplet_rows(
    dsets: Mapping[str, Any],
    index_map: Mapping[str, int],
    context: str,
) -> None:
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
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    maps: dict[str, dict[str, int]] = {}

    for cond, ds in dsets.items():
        m: dict[str, int] = {}
        for i in range(len(ds)):
            key = canonical_rta_pair_id(ds[i]["id"], cond)
            if not key:
                continue
            if key in m:
                raise RuntimeError(f"Duplicate canonical ID {key!r} in {cond}")
            m[key] = i
        maps[cond] = m

    shared = set.intersection(*(set(m.keys()) for m in maps.values()))
    expected = min(len(ds) for ds in dsets.values())

    rows: list[dict[str, Any]] = []
    if len(shared) >= max(1, int(0.95 * expected)):
        for key in sorted(shared):
            idx = {cond: int(maps[cond][key]) for cond in CONDITION_ORDER}
            validate_triplet_rows(dsets, idx, context=f"id:{key}")
            row = {"sample_key": key, "pairing_method": "canonical_id"}
            for cond in CONDITION_ORDER:
                row[f"{cond}_index"] = idx[cond]
            rows.append(row)
    else:
        lens = {cond: len(ds) for cond, ds in dsets.items()}
        if len(set(lens.values())) != 1:
            raise RuntimeError(
                "Could not safely pair RTA triplets: canonical overlap "
                f"{len(shared)}/{expected}; unequal lengths={lens}"
            )
        n = next(iter(lens.values()))
        for i in range(n):
            idx = {cond: i for cond in CONDITION_ORDER}
            validate_triplet_rows(dsets, idx, context=f"ordered:{i}")
            row = {
                "sample_key": f"ordered_{i:06d}",
                "pairing_method": "validated_order",
            }
            for cond in CONDITION_ORDER:
                row[f"{cond}_index"] = i
            rows.append(row)

    if limit > 0 and limit < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, k=limit)
        rows.sort(key=lambda r: str(r["sample_key"]))

    return rows


# =============================================================================
# Basis
# =============================================================================

def load_basis2(
    oracle_root: Path,
    model_name: str,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Load the stable uncentered-SVD register basis actually stored by the
    existing no-RN CLS/register oracle.

    Important: the current oracle is intentionally rank-2 (mu1, mu2).  Later
    residual-rank analyses that discussed weaker rank-3/4 directions used
    separate residualized SVD procedures and are not assumed to be serialized
    in this file.
    """
    path = oracle_root / model_name / "mu_basis.npz"
    if not path.is_file():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    if "mu_basis" in data:
        basis = np.asarray(data["mu_basis"], np.float64)
    elif "basis4" in data:
        basis = np.asarray(data["basis4"], np.float64)
    else:
        raise KeyError(f"No mu_basis/basis4 in {path}: {data.files}")

    if basis.ndim != 2 or basis.shape[0] < 2 or basis.shape[1] != width:
        raise RuntimeError(
            f"Need >=2 basis vectors of width {width}; found {basis.shape}"
        )

    mu1 = basis[0].copy()
    mu1 /= max(np.linalg.norm(mu1), 1e-12)

    mu2 = basis[1].copy()
    mu2 -= np.dot(mu2, mu1) * mu1
    mu2 /= max(np.linalg.norm(mu2), 1e-12)

    return (
        torch.from_numpy(mu1.astype(np.float32)),
        torch.from_numpy(mu2.astype(np.float32)),
        str(path),
    )


# =============================================================================
# Exact preprocessed TEXT mask
# =============================================================================

def patchify_score(score_hw: torch.Tensor, side: int) -> torch.Tensor:
    h, w = score_hw.shape
    if h % side != 0 or w % side != 0:
        raise ValueError(
            f"Preprocessed image {h}x{w} is not divisible by patch grid {side}."
        )
    ph = h // side
    pw = w // side
    return (
        score_hw.reshape(side, ph, side, pw)
        .permute(0, 2, 1, 3)
        .mean(dim=(2, 3))
        .reshape(-1)
    )


def dilate_mask(mask_p: np.ndarray, side: int, radius: int) -> np.ndarray:
    mask = np.asarray(mask_p, bool).reshape(side, side)
    if radius <= 0:
        return mask.reshape(-1)

    out = mask.copy()
    for _ in range(radius):
        src = out.copy()
        dst = src.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r0 = max(0, -dr)
                r1 = min(side, side - dr)
                c0 = max(0, -dc)
                c1 = min(side, side - dc)
                dst[
                    r0 + dr:r1 + dr,
                    c0 + dc:c1 + dc,
                ] |= src[r0:r1, c0:c1]
        out = dst
    return out.reshape(-1)


def text_mask_from_preprocessed_pair(
    clean_chw: torch.Tensor,
    attacked_chw: torch.Tensor,
    side: int,
    args,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    diff = (attacked_chw.float() - clean_chw.float()).abs().mean(dim=0)
    score = patchify_score(diff, side).cpu().numpy().astype(np.float64)

    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med))) * 1.4826
    mx = float(np.max(score))

    if mx <= float(args.text_diff_absolute_floor):
        raise RuntimeError(
            f"Paired images appear identical after preprocess: max diff={mx:.6g}"
        )

    threshold = max(
        med + float(args.text_diff_mad_mult) * mad,
        float(args.text_diff_relmax) * mx,
        float(args.text_diff_absolute_floor),
    )
    mask = score >= threshold

    min_patches = int(args.text_mask_min_patches)
    max_patches = int(args.text_mask_max_patches)

    if int(mask.sum()) < min_patches:
        top = np.argsort(score)[-min(min_patches, len(score)):]
        mask[:] = False
        mask[top] = True

    if max_patches > 0 and int(mask.sum()) > max_patches:
        top = np.argsort(score)[-max_patches:]
        mask[:] = False
        mask[top] = True

    if int(args.text_mask_dilate) > 0:
        mask = dilate_mask(mask, side, int(args.text_mask_dilate))

    audit = {
        "diff_max": mx,
        "diff_median": med,
        "diff_mad_sigma": mad,
        "threshold": float(threshold),
        "text_patch_count": int(mask.sum()),
        "text_patch_fraction": float(mask.mean()),
        "selected_score_mean": float(score[mask].mean()) if np.any(mask) else 0.0,
        "unselected_score_mean": float(score[~mask].mean()) if np.any(~mask) else 0.0,
    }
    return mask.astype(bool), score, audit


# =============================================================================
# Batched native capture: shared through B12, branch at pre-B13
# =============================================================================


def _run_one_visual_block(block, x: torch.Tensor) -> torch.Tensor:
    ln1 = block.ln_1(x)
    attn_out, _ = block.attention(ln1, need_weights=True, capture=True)
    x_attn = x + attn_out
    ln2 = block.ln_2(x_attn)
    x = x_attn + block.mlp.c_proj(block.mlp.gelu(block.mlp.c_fc(ln2)))
    clear_attn_cache(block)
    return x


@torch.inference_mode()
def capture_pre_states_dual_batch(
    bundle: ModelBundle,
    images_bchw: torch.Tensor,
    rn_payload: RNPayload,
    args,
) -> tuple[
    dict[str, dict[int, torch.Tensor]],
    dict[str, torch.Tensor],
    dict[str, np.ndarray],
    int,
    int,
]:
    """
    Compute RN-off and RN-on native trajectories without recomputing B0--B12.

    Returns
    -------
    pre_states[rn_mode][block] : float32 [T,B,D], GPU by default
    final_x[rn_mode]           : CPU float32 [T,B,D]
    own_reg_masks[rn_mode]     : bool [B,P]
    patch_count
    side

    For blocks < RN_INSERT_BLOCK, rn_on and rn_off reference the same exact
    pre-state tensor.  At pre-B13 the stream is branched and RN is appended as
    the final token in the rn_on branch.
    """
    visual = bundle.model.visual

    if bundle.device.type == "cuda" and not images_bchw.is_pinned():
        try:
            images_bchw = images_bchw.pin_memory()
        except RuntimeError:
            pass

    images = images_bchw.to(
        bundle.device,
        dtype=bundle.model.dtype,
        non_blocking=(bundle.device.type == "cuda"),
    )
    x_common = visual._prepare_tokens(images)

    patch_count = int(visual.positional_embedding.shape[0] - 1)
    side = side_from_patch_count(patch_count)
    blocks = visual.transformer.resblocks
    insert = int(rn_payload.insert_block)

    pre_states: dict[str, dict[int, torch.Tensor]] = {
        "rn_off": {},
        "rn_on": {},
    }

    def snapshot(x: torch.Tensor) -> torch.Tensor:
        if args.prestate_storage == "gpu":
            # The visual forward is functional here: subsequent blocks create
            # new tensors rather than mutating this pre-state in place.
            return x.detach()
        return x.detach().float().cpu()

    # Exact common prefix.  RN is inserted immediately before block `insert`,
    # so B0..B(insert-1) have identical pre-states and are evaluated once later.
    for block_index in range(insert):
        saved_pre = snapshot(x_common)
        pre_states["rn_off"][block_index] = saved_pre
        pre_states["rn_on"][block_index] = saved_pre
        x_common = _run_one_visual_block(blocks[block_index], x_common)

    x_off = x_common
    rn = rn_payload.token.to(bundle.device, dtype=x_common.dtype).reshape(1, 1, -1)
    rn = rn.expand(1, x_common.shape[1], -1)
    x_on = torch.cat([x_common, rn], dim=0)

    for block_index in range(insert, len(blocks)):
        pre_states["rn_off"][block_index] = snapshot(x_off)
        pre_states["rn_on"][block_index] = snapshot(x_on)
        x_off = _run_one_visual_block(blocks[block_index], x_off)
        x_on = _run_one_visual_block(blocks[block_index], x_on)

    final_x = {
        "rn_off": x_off.detach().float().cpu(),
        "rn_on": x_on.detach().float().cpu(),
    }

    own_reg_masks: dict[str, np.ndarray] = {}
    for rn_mode in RN_MODE_ORDER:
        final_patch = (
            final_x[rn_mode][1:1 + patch_count]
            .permute(1, 0, 2)
            .contiguous()
        )
        masks = []
        for i in range(final_patch.shape[0]):
            norms = final_patch[i].norm(dim=-1)
            reg = select_register_mask(
                norms,
                threshold=args.final_register_threshold,
                minimum=args.final_register_min,
                maximum=args.final_register_max,
            )
            masks.append(reg.cpu().numpy().astype(bool))
        own_reg_masks[rn_mode] = np.stack(masks, axis=0)

    return pre_states, final_x, own_reg_masks, patch_count, side


# =============================================================================
# Fast exact per-head mechanics
# =============================================================================

def block_value_metrics(
    block,
    mu1_d: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute per-head:
        G_h = W_Oblock^T W_Oblock   for exact ||OV(v)|| without materializing D
        m_h = W_Oblock^T mu1        for signed mu1 projection
    """
    wout = block.attn.out_proj.weight.detach().float().to(device)
    width = int(wout.shape[0])
    heads = int(block.attn.num_heads)
    dh = width // heads

    G = []
    M = []
    mu1 = mu1_d.float().to(device)
    for h in range(heads):
        wh = wout[:, h * dh:(h + 1) * dh]  # [D,dh]
        G.append(wh.T @ wh)
        M.append(wh.T @ mu1)
    return torch.stack(G, dim=0), torch.stack(M, dim=0)


def projected_value_norm_and_mu1(
    values_nhtd: torch.Tensor,
    G_hdd: torch.Tensor,
    M_hd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = values_nhtd.float()
    norm2 = torch.einsum("nhtd,hde,nhte->nht", v, G_hdd, v)
    norm = torch.sqrt(torch.clamp(norm2, min=0.0))
    mu1 = torch.einsum("nhtd,hd->nht", v, M_hd)
    return norm, mu1


def masked_sum_bhp(values_bhp: np.ndarray, mask_bp: np.ndarray) -> np.ndarray:
    return np.einsum(
        "bhp,bp->bh",
        np.asarray(values_bhp, np.float64),
        np.asarray(mask_bp, np.float64),
    )


@torch.inference_mode()
def evaluate_block_slider_primitives(
    bundle: ModelBundle,
    pre_tbd_cpu: torch.Tensor,
    block_index: int,
    factors: Sequence[float],
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    patch_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    One actual attention call handles B images x F slider states at one block.

    The expensive attention/QKV computation is independent of register-mask
    policy.  We therefore return patchwise slider primitives once and aggregate
    them against the RN-off and RN-on own-register masks separately.  This lets
    the identical pre-B13 state be evaluated only once.

    Returns
    -------
    slider : dict of [B,F,H,P] arrays
    native : alpha=0 dict of [B,H,P] arrays for population analysis
    """
    block = bundle.model.visual.transformer.resblocks[block_index]
    if pre_tbd_cpu.device == bundle.device and pre_tbd_cpu.dtype == bundle.model.dtype:
        pre = pre_tbd_cpu
    else:
        pre = pre_tbd_cpu.to(
            bundle.device,
            dtype=bundle.model.dtype,
            non_blocking=(bundle.device.type == "cuda" and pre_tbd_cpu.device.type == "cpu" and pre_tbd_cpu.is_pinned()),
        )
    T, B, D = pre.shape
    factors = [float(x) for x in factors]
    F = len(factors)

    x = (
        pre[:, :, None, :]
        .expand(T, B, F, D)
        .reshape(T, B * F, D)
        .clone()
    )
    fvec = torch.tensor(factors, device=bundle.device, dtype=x.dtype).repeat(B)
    x[0] = x[0] + fvec[:, None] * mu2.to(bundle.device, dtype=x.dtype)[None, :]

    ln1 = block.ln_1(x)
    _attn_out, probs0 = block.attention(ln1, need_weights=True, capture=True)

    heads = int(block.attn.num_heads)
    tokens = int(x.shape[0])
    N = B * F

    probs = normalize_probs_shape(probs0, N, heads, tokens).float()
    values = normalize_qkv_shape(block.attn.last_v, N, heads, tokens).float()
    q = normalize_qkv_shape(block.attn.last_q, N, heads, tokens).float()
    k = normalize_qkv_shape(block.attn.last_k, N, heads, tokens).float()

    G, M = block_value_metrics(block, mu1, bundle.device)
    pv_norm, pv_mu1 = projected_value_norm_and_mu1(values, G, M)

    ps = slice(1, 1 + patch_count)
    a_c2p = probs[:, :, ps, 0]
    a_p2c = probs[:, :, 0, ps]

    c2p_write = a_c2p * pv_norm[:, :, 0][:, :, None]
    p2c_write = a_p2c * pv_norm[:, :, ps]
    c2p_mu1 = a_c2p * pv_mu1[:, :, 0][:, :, None]
    p2c_mu1 = a_p2c * pv_mu1[:, :, ps]

    dh = int(q.shape[-1])
    qk_cls_patch = torch.einsum(
        "nhd,nhpd->nhp",
        q[:, :, 0, :],
        k[:, :, ps, :],
    ) / math.sqrt(dh)

    def BFHP(t: torch.Tensor) -> np.ndarray:
        return (
            t.detach()
            .float()
            .cpu()
            .numpy()
            .reshape(B, F, heads, patch_count)
        )

    slider = {
        "c2p_write": BFHP(c2p_write),
        "p2c_write": BFHP(p2c_write),
        "c2p_attn": BFHP(a_c2p),
        "p2c_attn": BFHP(a_p2c),
        "c2p_mu1": BFHP(c2p_mu1),
        "p2c_mu1": BFHP(p2c_mu1),
        "qk_cls_patch": BFHP(qk_cls_patch),
    }

    zero_candidates = [i for i, f in enumerate(factors) if abs(float(f)) <= 1e-12]
    if len(zero_candidates) != 1:
        raise RuntimeError("Need exactly one factor=0.")
    zi = zero_candidates[0]
    native = {key: value[:, zi] for key, value in slider.items()}

    clear_attn_cache(block)
    del x, ln1, probs, values, q, k, pv_norm, pv_mu1
    return slider, native


def aggregate_slider_curves(
    slider: Mapping[str, np.ndarray],
    factors: Sequence[float],
    reg_frozen_bp: np.ndarray,
    reg_own_bp: np.ndarray,
) -> list[dict[str, Any]]:
    """Cheap mask aggregation of already-computed [B,F,H,P] primitives."""
    factors = [float(x) for x in factors]
    records: list[dict[str, Any]] = []

    for fi, factor in enumerate(factors):
        c2p_write = slider["c2p_write"][:, fi]
        p2c_write = slider["p2c_write"][:, fi]
        c2p_attn = slider["c2p_attn"][:, fi]
        p2c_attn = slider["p2c_attn"][:, fi]
        c2p_mu1 = slider["c2p_mu1"][:, fi]

        records.append({
            "factor": factor,
            "frozen_C2R_write": masked_sum_bhp(c2p_write, reg_frozen_bp),
            "frozen_R2C_write": masked_sum_bhp(p2c_write, reg_frozen_bp),
            "frozen_C2R_attn": masked_sum_bhp(c2p_attn, reg_frozen_bp),
            "frozen_R2C_attn": masked_sum_bhp(p2c_attn, reg_frozen_bp),
            "own_C2R_write": masked_sum_bhp(c2p_write, reg_own_bp),
            "own_R2C_write": masked_sum_bhp(p2c_write, reg_own_bp),
            "own_C2R_attn": masked_sum_bhp(c2p_attn, reg_own_bp),
            "own_R2C_attn": masked_sum_bhp(p2c_attn, reg_own_bp),
            "CLS_source_mu1_patch_mean": np.mean(c2p_mu1, axis=-1),
        })
    return records


# =============================================================================
# Population masks
# =============================================================================

def build_population_masks(
    patch_bpd: torch.Tensor,
    reg_frozen_bp: np.ndarray,
    reg_own_bp: np.ndarray,
    text_bp: np.ndarray,
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    args,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """
    patch_bpd: CPU float32 [B,P,D]

    REG_FROZEN/REG_OWN are raw and may overlap TEXT.

    The operational partition below excludes union(REG_FROZEN, REG_OWN, TEXT):

        MU1_CACHE
            ordinary-looking patches strongly aligned with the invariant mu1
            carrier rail.

        SCRATCH_RESIDUAL
            remaining patches with unusually high normalized residual energy
            outside span(mu1,mu2).

        ORDINARY
            everything left.

    SCRATCH_RESIDUAL deliberately avoids pretending that the current
    mu_basis.npz stores stable mu3/mu4 directions.  It asks the weaker and
    cleaner question: "where is unusually much patch state living outside the
    stable rank-2 role/interface plane?"
    """
    x = patch_bpd.float()
    norms = x.norm(dim=-1).clamp_min(EPS)

    m1 = mu1.float()
    m2 = mu2.float()

    proj1 = x @ m1
    proj2 = x @ m2

    cos1 = proj1 / norms

    # Fraction of normalized patch-state energy lying OUTSIDE span(mu1,mu2).
    # Clamp for numerical safety because the basis is orthonormalized.
    rank2_energy2 = proj1.square() + proj2.square()
    residual2 = torch.clamp(norms.square() - rank2_energy2, min=0.0)
    residual_frac = torch.sqrt(residual2) / norms

    cos1_np = cos1.numpy()
    residual_np = residual_frac.numpy()

    reg_frozen = np.asarray(reg_frozen_bp, bool)
    reg_own = np.asarray(reg_own_bp, bool)
    text = np.asarray(text_bp, bool)

    reg_any = reg_frozen | reg_own
    excluded = reg_any | text

    mu_cache = (~excluded) & (
        cos1_np >= float(args.mu1_cache_cos_threshold)
    )

    scratch = np.zeros_like(mu_cache)
    for i in range(x.shape[0]):
        candidate = (~excluded[i]) & (~mu_cache[i])
        vals = residual_np[i][candidate]
        if len(vals) == 0:
            continue

        q = float(np.quantile(
            vals,
            float(args.scratch_residual_quantile),
        ))
        threshold = max(
            q,
            float(args.scratch_residual_min_fraction),
        )
        scratch[i] = candidate & (residual_np[i] >= threshold)

    nonreg_nontext = (~reg_any) & (~text)
    ordinary = nonreg_nontext & (~mu_cache) & (~scratch)

    masks = {
        "REG_FROZEN": reg_frozen,
        "REG_OWN": reg_own,
        "TEXT": text,
        "NONREG_NONTEXT": nonreg_nontext,
        "MU1_CACHE": mu_cache,
        "SCRATCH_RESIDUAL": scratch,
        "ORDINARY": ordinary,
    }

    audit = []
    for i in range(x.shape[0]):
        audit.append({
            "reg_frozen_count": int(reg_frozen[i].sum()),
            "reg_own_count": int(reg_own[i].sum()),
            "reg_text_overlap_frozen": int((reg_frozen[i] & text[i]).sum()),
            "reg_text_overlap_own": int((reg_own[i] & text[i]).sum()),
            "text_count": int(text[i].sum()),
            "mu1_cache_count": int(mu_cache[i].sum()),
            "scratch_residual_count": int(scratch[i].sum()),
            "ordinary_count": int(ordinary[i].sum()),
            "nonreg_nontext_count": int(nonreg_nontext[i].sum()),
            "mu1_cache_cos_mean": float(
                cos1_np[i][mu_cache[i]].mean()
            ) if np.any(mu_cache[i]) else np.nan,
            "scratch_residual_fraction_mean": float(
                residual_np[i][scratch[i]].mean()
            ) if np.any(scratch[i]) else np.nan,
        })

    return masks, audit


# =============================================================================
# Apply native head mechanics to token populations
# =============================================================================

def population_metrics_for_logical_batch(
    native: Mapping[str, np.ndarray],
    physical_indices: np.ndarray,
    masks: Mapping[str, np.ndarray],
    patch_count: int,
) -> dict[str, np.ndarray]:
    """
    native arrays: [Bphysical,H,P]
    physical_indices: [Blogical]
    masks: [Blogical,P]
    returns wide arrays [Blogical,H] keyed by "<POP>_<metric>"
    """
    idx = np.asarray(physical_indices, int)

    c2w = np.asarray(native["c2p_write"])[idx]
    p2w = np.asarray(native["p2c_write"])[idx]
    c2a = np.asarray(native["c2p_attn"])[idx]
    p2a = np.asarray(native["p2c_attn"])[idx]
    c2m = np.asarray(native["c2p_mu1"])[idx]
    p2m = np.asarray(native["p2c_mu1"])[idx]
    qk = np.asarray(native["qk_cls_patch"])[idx]

    all_c2w = np.sum(c2w, axis=-1)
    all_p2w = np.sum(p2w, axis=-1)
    all_c2a = np.sum(c2a, axis=-1)
    all_p2a = np.sum(p2a, axis=-1)

    out: dict[str, np.ndarray] = {}
    for pop, mask in masks.items():
        mask = np.asarray(mask, bool)
        count = np.sum(mask, axis=-1).astype(np.float64)
        area = count / float(patch_count)

        metrics = {
            "C2P_write": masked_sum_bhp(c2w, mask),
            "P2C_write": masked_sum_bhp(p2w, mask),
            "C2P_attn": masked_sum_bhp(c2a, mask),
            "P2C_attn": masked_sum_bhp(p2a, mask),
            "C2P_mu1_signed": masked_sum_bhp(c2m, mask),
            "P2C_mu1_signed": masked_sum_bhp(p2m, mask),
        }

        # Signed QK summary.  Some operational populations are legitimately
        # empty for some image/block pairs (especially MU1_CACHE /
        # SCRATCH_RESIDUAL).  Avoid np.nanmean(all-NaN), which is numerically
        # fine but emits a RuntimeWarning for every empty population.
        mask3 = mask[:, None, :]
        count_b1 = np.sum(mask, axis=-1, dtype=np.float64)[:, None]  # [B,1]

        qk_sum = np.sum(
            np.where(mask3, qk, 0.0),
            axis=-1,
            dtype=np.float64,
        )
        qk_mean = np.full_like(qk_sum, np.nan, dtype=np.float64)
        np.divide(
            qk_sum,
            count_b1,
            out=qk_mean,
            where=count_b1 > 0,
        )

        qk_abs = np.where(mask3, np.abs(qk), -np.inf)
        qk_max_abs = np.max(qk_abs, axis=-1)
        qk_max_abs[~np.isfinite(qk_max_abs)] = np.nan

        out[f"{pop}_count"] = np.repeat(count[:, None], c2w.shape[1], axis=1)
        out[f"{pop}_area_fraction"] = np.repeat(area[:, None], c2w.shape[1], axis=1)
        out[f"{pop}_QK_mean"] = qk_mean
        out[f"{pop}_QK_max_abs"] = qk_max_abs

        for name, arr in metrics.items():
            out[f"{pop}_{name}"] = arr

        out[f"{pop}_C2P_write_share"] = safe_div(
            metrics["C2P_write"], all_c2w
        )
        out[f"{pop}_P2C_write_share"] = safe_div(
            metrics["P2C_write"], all_p2w
        )
        out[f"{pop}_C2P_attn_share"] = safe_div(
            metrics["C2P_attn"], all_c2a
        )
        out[f"{pop}_P2C_attn_share"] = safe_div(
            metrics["P2C_attn"], all_p2a
        )

        area2 = np.maximum(area[:, None], 1.0 / float(patch_count))
        out[f"{pop}_C2P_write_enrichment"] = (
            out[f"{pop}_C2P_write_share"] / area2
        )
        out[f"{pop}_P2C_write_enrichment"] = (
            out[f"{pop}_P2C_write_share"] / area2
        )
        out[f"{pop}_C2P_attn_enrichment"] = (
            out[f"{pop}_C2P_attn_share"] / area2
        )
        out[f"{pop}_P2C_attn_enrichment"] = (
            out[f"{pop}_P2C_attn_share"] / area2
        )

    return out


# =============================================================================
# Curve fitting / motif descriptors
# =============================================================================


def fit_curve(
    factors: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    factors = np.asarray(factors, np.float64)
    y = np.asarray(y, np.float64)
    max_abs = max(float(np.max(np.abs(factors))), EPS)
    x = factors / max_abs
    c, s, b0 = np.polyfit(x, y, deg=2)
    pred = c * x * x + s * x + b0
    zero_idx = int(np.argmin(np.abs(factors)))
    return {
        "b0": float(b0),
        "slope": float(s),
        "curvature": float(c),
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "native": float(y[zero_idx]),
        "range": float(np.max(y) - np.min(y)),
        "peak": float(np.max(y)),
        "endpoint_mean": float(0.5 * (y[0] + y[-1])),
    }


def fit_curve_group(z: pd.DataFrame, policy: str) -> dict[str, float]:
    z = z.sort_values("factor_mu2")
    factors = z["factor_mu2"].to_numpy(np.float64)
    c2r = z[f"{policy}_C2R_write"].to_numpy(np.float64)
    r2c = z[f"{policy}_R2C_write"].to_numpy(np.float64)
    fc = fit_curve(factors, c2r)
    fr = fit_curve(factors, r2c)
    scale = max(float(np.max(c2r)), float(np.max(r2c)), EPS)
    out = {
        f"{policy}_response_scale": scale,
        f"{policy}_curve_corr": safe_corr(c2r / scale, r2c / scale),
    }
    for prefix, fit in (("C2R", fc), ("R2C", fr)):
        for key, value in fit.items():
            out[f"{policy}_{prefix}_{key}"] = float(value)
        out[f"{policy}_{prefix}_slope_rel"] = float(fit["slope"] / scale)
        out[f"{policy}_{prefix}_curvature_rel"] = float(fit["curvature"] / scale)
        out[f"{policy}_{prefix}_rmse_rel"] = float(fit["rmse"] / scale)
    return out


def fit_per_image_curves(raw_curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["sample_key", "rn_mode", "condition", "block", "head"]
    for keys, z in tqdm(
        raw_curves.groupby(group_cols, sort=True),
        desc="fit per-image head curves",
        unit="head",
    ):
        sample_key, rn_mode, condition, block, head = keys
        row = {
            "sample_key": sample_key,
            "rn_mode": rn_mode,
            "condition": condition,
            "block": int(block),
            "head": int(head),
        }
        row.update(fit_curve_group(z, "frozen"))
        row.update(fit_curve_group(z, "own"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def classify_mean_motif(row: Mapping[str, Any], args) -> tuple[str, str]:
    sC = float(row["frozen_C2R_slope_rel"])
    sR = float(row["frozen_R2C_slope_rel"])
    cC = float(row["frozen_C2R_curvature_rel"])
    cR = float(row["frozen_R2C_curvature_rel"])
    st = float(args.motif_slope_threshold)
    ct = float(args.motif_curvature_threshold)
    dom = float(args.motif_dominance_ratio)
    if (
        abs(sC) < args.motif_insensitive_threshold
        and abs(sR) < args.motif_insensitive_threshold
        and abs(cC) < args.motif_insensitive_threshold
        and abs(cR) < args.motif_insensitive_threshold
    ):
        return "insensitive", "weak response"
    if sC <= -st and sR >= st:
        return "push_pull_exchange", f"sC={sC:+.3f}, sR={sR:+.3f}"
    if sC >= st and sR <= -st:
        return "opposite_push_pull", f"sC={sC:+.3f}, sR={sR:+.3f}"
    if sC <= -st and sR <= -st:
        return "common_mode_attenuation", f"sC={sC:+.3f}, sR={sR:+.3f}"
    if sC >= st and sR >= st:
        return "common_mode_gain", f"sC={sC:+.3f}, sR={sR:+.3f}"
    if abs(sR) >= st and abs(sR) >= dom * max(abs(sC), EPS):
        return "read_gain", f"|sR| dominates: {sR:+.3f}"
    if abs(sC) >= st and abs(sC) >= dom * max(abs(sR), EPS):
        return "broadcast_gain", f"|sC| dominates: {sC:+.3f}"
    if cC >= ct and cC >= 0.85 * max(abs(sC), st):
        return "source_null_curvature", f"cC={cC:+.3f}"
    if cR <= -ct and abs(cR) >= 0.85 * max(abs(sR), st):
        return "read_optimum", f"cR={cR:+.3f}"
    return "mixed", f"sC={sC:+.3f}, sR={sR:+.3f}, cC={cC:+.3f}, cR={cR:+.3f}"


def fit_condition_mean_curves(
    raw_curves: pd.DataFrame,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_curves = (
        raw_curves
        .groupby(["rn_mode", "condition", "block", "head", "factor_mu2"], as_index=False)
        .mean(numeric_only=True)
    )
    rows = []
    for keys, z in mean_curves.groupby(
        ["rn_mode", "condition", "block", "head"], sort=True
    ):
        rn_mode, condition, block, head = keys
        row = {
            "rn_mode": rn_mode,
            "condition": condition,
            "block": int(block),
            "head": int(head),
        }
        row.update(fit_curve_group(z, "frozen"))
        row.update(fit_curve_group(z, "own"))
        group, reason = classify_mean_motif(row, args)
        row["primary_motif"] = group
        row["motif_reason"] = reason
        row["push_pull_sign"] = bool(
            row["frozen_C2R_slope_rel"] < 0
            and row["frozen_R2C_slope_rel"] > 0
        )
        rows.append(row)
    fits = pd.DataFrame(rows).sort_values(
        ["rn_mode", "condition", "block", "head"]
    ).reset_index(drop=True)
    return mean_curves, fits


def normalized_response_vector(
    mean_curves: pd.DataFrame,
    *,
    rn_mode: str,
    condition: str,
    block: int,
    head: int,
    common_scale: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    z = mean_curves[
        (mean_curves["rn_mode"] == rn_mode)
        & (mean_curves["condition"] == condition)
        & (mean_curves["block"] == int(block))
        & (mean_curves["head"] == int(head))
    ].sort_values("factor_mu2")
    if z.empty:
        raise RuntimeError(
            f"Missing mean curve: rn={rn_mode}, condition={condition}, B{block}, H{head}"
        )
    c = z["frozen_C2R_write"].to_numpy(np.float64)
    r = z["frozen_R2C_write"].to_numpy(np.float64)
    scale = common_scale
    if scale is None:
        scale = max(float(np.max(c)), float(np.max(r)), EPS)
    vec = np.concatenate([c / scale, r / scale])
    return vec, float(scale)


# =============================================================================
# Paired stats
# =============================================================================

MU2_STAT_METRICS = (
    "frozen_C2R_slope_rel",
    "frozen_R2C_slope_rel",
    "frozen_C2R_curvature_rel",
    "frozen_R2C_curvature_rel",
    "frozen_curve_corr",
    "frozen_response_scale",
    "own_C2R_slope_rel",
    "own_R2C_slope_rel",
)


def _attach_fdr(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["t_q"] = np.nan
    df["wilcoxon_q"] = np.nan
    for _keys, idx in df.groupby(list(group_cols)).groups.items():
        ii = np.asarray(list(idx), int)
        df.loc[ii, "t_q"] = bh_fdr(df.loc[ii, "t_p"].to_numpy())
        df.loc[ii, "wilcoxon_q"] = bh_fdr(df.loc[ii, "wilcoxon_p"].to_numpy())
    return df


def _vectorized_effect_rows(
    delta_long: pd.DataFrame,
    *,
    meta: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Exact paired statistics for every Bxx/Hxx cell in one vectorized SciPy call.

    delta_long must contain exactly one `delta` per sample_key/block/head.
    This avoids thousands of Python-level scipy.stats calls while preserving the
    same image-paired t-test, Wilcoxon, Cohen dz, median and mean definitions.
    """
    if delta_long.empty:
        return []

    pivot = delta_long.pivot(
        index="sample_key",
        columns=["block", "head"],
        values="delta",
    ).sort_index(axis=1)
    arr = pivot.to_numpy(np.float64)
    finite = np.isfinite(arr)
    n = finite.sum(axis=0).astype(int)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(arr, axis=0)
        median = np.nanmedian(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1)
    std = np.where(n > 1, std, 0.0)
    dz = np.zeros_like(mean)
    nz_std = std > EPS
    dz[nz_std] = mean[nz_std] / std[nz_std]
    dz[(~nz_std) & (np.abs(mean) > EPS)] = np.sign(mean[(~nz_std) & (np.abs(mean) > EPS)]) * np.inf

    t_p = np.full(arr.shape[1], np.nan, np.float64)
    w_p = np.full(arr.shape[1], np.nan, np.float64)

    if scipy_stats is not None:
        eligible_t = n > 1
        if np.any(eligible_t):
            try:
                tres = scipy_stats.ttest_1samp(
                    arr[:, eligible_t],
                    0.0,
                    axis=0,
                    nan_policy="omit",
                )
                t_p[eligible_t] = np.asarray(tres.pvalue, np.float64).reshape(-1)
            except Exception:
                pass

        all_zero = np.zeros(arr.shape[1], dtype=bool)
        for j in range(arr.shape[1]):
            vals = arr[finite[:, j], j]
            all_zero[j] = len(vals) > 0 and np.all(np.abs(vals) <= EPS)
        w_p[all_zero] = 1.0

        eligible_w = (n > 0) & (~all_zero)
        if np.any(eligible_w):
            try:
                # Modern SciPy supports axis + nan_policy here. This is much
                # faster than one Wilcoxon invocation per block/head.
                wres = scipy_stats.wilcoxon(
                    arr[:, eligible_w],
                    axis=0,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    nan_policy="omit",
                )
                w_p[eligible_w] = np.asarray(wres.pvalue, np.float64).reshape(-1)
            except TypeError:
                # Older SciPy: retain exact definitions with a small compatibility
                # loop only over block/head cells, not over the entire pipeline.
                cols = np.flatnonzero(eligible_w)
                for j in cols:
                    vals = arr[finite[:, j], j]
                    try:
                        w_p[j] = float(scipy_stats.wilcoxon(
                            vals,
                            zero_method="wilcox",
                            correction=False,
                            alternative="two-sided",
                        ).pvalue)
                    except Exception:
                        w_p[j] = np.nan
            except Exception:
                # If vectorized Wilcoxon fails for a pathological matrix, retry
                # cellwise rather than dropping the statistic.
                cols = np.flatnonzero(eligible_w)
                for j in cols:
                    vals = arr[finite[:, j], j]
                    try:
                        w_p[j] = float(scipy_stats.wilcoxon(
                            vals,
                            zero_method="wilcox",
                            correction=False,
                            alternative="two-sided",
                        ).pvalue)
                    except Exception:
                        w_p[j] = np.nan

    rows: list[dict[str, Any]] = []
    for j, col in enumerate(pivot.columns):
        block, head = col
        rows.append({
            **dict(meta),
            "block": int(block),
            "head": int(head),
            "n": int(n[j]),
            "mean_delta": float(mean[j]),
            "median_delta": float(median[j]),
            "std_delta": float(std[j]),
            "cohen_dz": float(dz[j]),
            "t_p": float(t_p[j]),
            "wilcoxon_p": float(w_p[j]),
        })
    return rows


def _paired_delta_long(
    clean: pd.DataFrame,
    attack: pd.DataFrame,
    metric: str,
    *,
    extra_key: Sequence[str] = (),
) -> pd.DataFrame:
    key = ["sample_key", *extra_key, "block", "head"]
    a = clean[key + [metric]].rename(columns={metric: "clean"})
    b = attack[key + [metric]].rename(columns={metric: "attack"})
    z = a.merge(b, on=key, how="inner", validate="one_to_one")
    z["delta"] = z["attack"].to_numpy(np.float64) - z["clean"].to_numpy(np.float64)
    return z[["sample_key", "block", "head", "delta"]]


def paired_significance_mu2(per_image_fits: pd.DataFrame) -> pd.DataFrame:
    """Attack-vs-NoRTA statistics separately for rn_off and rn_on."""
    rows: list[dict[str, Any]] = []
    for rn_mode in RN_MODE_ORDER:
        qmode = per_image_fits[per_image_fits["rn_mode"] == rn_mode]
        clean = qmode[qmode["condition"] == "NoRTA"]
        for attack in ATTACK_ORDER:
            atk = qmode[qmode["condition"] == attack]
            for metric in MU2_STAT_METRICS:
                delta = _paired_delta_long(clean, atk, metric)
                rows.extend(_vectorized_effect_rows(
                    delta,
                    meta={"rn_mode": rn_mode, "comparison": attack, "metric": metric},
                ))
    return _attach_fdr(pd.DataFrame(rows), ["rn_mode", "comparison", "metric"])


def paired_significance_mu2_rn_effect(per_image_fits: pd.DataFrame) -> pd.DataFrame:
    """
    RN causal difference-of-differences on fitted mu2 descriptors:

        [(attack - NoRTA)_rn_on] - [(attack - NoRTA)_rn_off].
    """
    rows: list[dict[str, Any]] = []
    key = ["sample_key", "block", "head"]
    for attack in ATTACK_ORDER:
        off_clean = per_image_fits[(per_image_fits["rn_mode"] == "rn_off") & (per_image_fits["condition"] == "NoRTA")]
        off_atk = per_image_fits[(per_image_fits["rn_mode"] == "rn_off") & (per_image_fits["condition"] == attack)]
        on_clean = per_image_fits[(per_image_fits["rn_mode"] == "rn_on") & (per_image_fits["condition"] == "NoRTA")]
        on_atk = per_image_fits[(per_image_fits["rn_mode"] == "rn_on") & (per_image_fits["condition"] == attack)]
        for metric in MU2_STAT_METRICS:
            a = _paired_delta_long(off_clean, off_atk, metric).rename(columns={"delta": "off_delta"})
            b = _paired_delta_long(on_clean, on_atk, metric).rename(columns={"delta": "on_delta"})
            z = a.merge(b, on=key, how="inner", validate="one_to_one")
            z["delta"] = z["on_delta"] - z["off_delta"]
            rows.extend(_vectorized_effect_rows(
                z[["sample_key", "block", "head", "delta"]],
                meta={"comparison": attack, "metric": metric},
            ))
    return _attach_fdr(pd.DataFrame(rows), ["comparison", "metric"])


def population_metric_columns(
    df: pd.DataFrame,
    *,
    extended: bool = False,
) -> list[str]:
    """
    Metrics receiving inferential tests.

    Raw native_population_traffic.csv always contains the full metric set.
    The default statistical scope is the paper-relevant causal traffic subset;
    --extended_stats restores the exhaustive old table without another GPU run.
    """
    if extended:
        pops = (
            "TEXT", "REG_FROZEN", "REG_OWN", "MU1_CACHE", "SCRATCH_RESIDUAL",
            "NONREG_NONTEXT", "ORDINARY",
        )
        suffixes = (
            "P2C_write_share", "P2C_attn_share", "C2P_write_share", "C2P_attn_share",
            "P2C_write_enrichment", "P2C_attn_enrichment",
            "C2P_write_enrichment", "C2P_attn_enrichment",
        )
    else:
        pops = ("TEXT", "REG_FROZEN", "REG_OWN", "MU1_CACHE", "SCRATCH_RESIDUAL")
        suffixes = (
            "P2C_write_share", "P2C_attn_share",
            "C2P_write_share", "C2P_attn_share",
        )
    return [f"{pop}_{suffix}" for pop in pops for suffix in suffixes if f"{pop}_{suffix}" in df.columns]


def paired_significance_population(traffic: pd.DataFrame, *, extended: bool = False) -> pd.DataFrame:
    """Attack-vs-NoRTA population traffic separately for RN off/on."""
    rows: list[dict[str, Any]] = []
    metrics = population_metric_columns(traffic, extended=extended)
    for rn_mode in RN_MODE_ORDER:
        qmode = traffic[traffic["rn_mode"] == rn_mode]
        for comparison in ATTACK_ORDER:
            zc = qmode[(qmode["comparison"] == comparison) & (qmode["condition"] == "NoRTA")]
            za = qmode[(qmode["comparison"] == comparison) & (qmode["condition"] == comparison)]
            for metric in metrics:
                delta = _paired_delta_long(zc, za, metric, extra_key=("comparison",))
                rows.extend(_vectorized_effect_rows(
                    delta,
                    meta={"rn_mode": rn_mode, "comparison": comparison, "metric": metric},
                ))
    return _attach_fdr(pd.DataFrame(rows), ["rn_mode", "comparison", "metric"])


def paired_significance_population_rn_effect(traffic: pd.DataFrame, *, extended: bool = False) -> pd.DataFrame:
    """Population-traffic RN difference-of-differences."""
    rows: list[dict[str, Any]] = []
    metrics = population_metric_columns(traffic, extended=extended)
    key = ["sample_key", "block", "head"]
    for comparison in ATTACK_ORDER:
        off_clean = traffic[(traffic["rn_mode"] == "rn_off") & (traffic["comparison"] == comparison) & (traffic["condition"] == "NoRTA")]
        off_atk = traffic[(traffic["rn_mode"] == "rn_off") & (traffic["comparison"] == comparison) & (traffic["condition"] == comparison)]
        on_clean = traffic[(traffic["rn_mode"] == "rn_on") & (traffic["comparison"] == comparison) & (traffic["condition"] == "NoRTA")]
        on_atk = traffic[(traffic["rn_mode"] == "rn_on") & (traffic["comparison"] == comparison) & (traffic["condition"] == comparison)]
        for metric in metrics:
            a = _paired_delta_long(off_clean, off_atk, metric, extra_key=("comparison",)).rename(columns={"delta": "off_delta"})
            b = _paired_delta_long(on_clean, on_atk, metric, extra_key=("comparison",)).rename(columns={"delta": "on_delta"})
            z = a.merge(b, on=key, how="inner", validate="one_to_one")
            z["delta"] = z["on_delta"] - z["off_delta"]
            rows.extend(_vectorized_effect_rows(
                z[["sample_key", "block", "head", "delta"]],
                meta={"comparison": comparison, "metric": metric},
            ))
    return _attach_fdr(pd.DataFrame(rows), ["comparison", "metric"])


# =============================================================================
# Motif transition + exceptions
# =============================================================================


def motif_transitions(
    condition_fits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    attack_transitions
        NoRTA -> attack separately within rn_off/rn_on.
    exceptions
        Heads not following the mature push-pull sign after B12.
    rn_transitions
        attack(no RN) -> attack(+RN), with clean RN controls retained elsewhere.
    """
    transition_rows = []
    exception_rows = []
    rn_rows = []

    for rn_mode in RN_MODE_ORDER:
        qmode = condition_fits[condition_fits["rn_mode"] == rn_mode]
        clean = qmode[qmode["condition"] == "NoRTA"]
        for attack in ATTACK_ORDER:
            atk = qmode[qmode["condition"] == attack]
            z = clean.merge(
                atk,
                on=["block", "head"],
                suffixes=("_clean", "_attack"),
                how="inner",
            )
            for row in z.itertuples(index=False):
                transition_rows.append({
                    "rn_mode": rn_mode,
                    "comparison": attack,
                    "block": int(row.block),
                    "head": int(row.head),
                    "clean_motif": row.primary_motif_clean,
                    "attack_motif": row.primary_motif_attack,
                    "motif_changed": bool(row.primary_motif_clean != row.primary_motif_attack),
                    "clean_push_pull_sign": bool(row.push_pull_sign_clean),
                    "attack_push_pull_sign": bool(row.push_pull_sign_attack),
                    "delta_s_C2R": float(row.frozen_C2R_slope_rel_attack - row.frozen_C2R_slope_rel_clean),
                    "delta_s_R2C": float(row.frozen_R2C_slope_rel_attack - row.frozen_R2C_slope_rel_clean),
                    "delta_c_C2R": float(row.frozen_C2R_curvature_rel_attack - row.frozen_C2R_curvature_rel_clean),
                    "delta_c_R2C": float(row.frozen_R2C_curvature_rel_attack - row.frozen_R2C_curvature_rel_clean),
                })

    for row in condition_fits.itertuples(index=False):
        if int(row.block) < 12:
            continue
        if not bool(row.push_pull_sign):
            exception_rows.append({
                "rn_mode": row.rn_mode,
                "condition": row.condition,
                "block": int(row.block),
                "head": int(row.head),
                "primary_motif": row.primary_motif,
                "s_C2R": float(row.frozen_C2R_slope_rel),
                "s_R2C": float(row.frozen_R2C_slope_rel),
                "c_C2R": float(row.frozen_C2R_curvature_rel),
                "c_R2C": float(row.frozen_R2C_curvature_rel),
                "reason": row.motif_reason,
            })

    for attack in ATTACK_ORDER:
        off = condition_fits[
            (condition_fits["rn_mode"] == "rn_off")
            & (condition_fits["condition"] == attack)
        ]
        on = condition_fits[
            (condition_fits["rn_mode"] == "rn_on")
            & (condition_fits["condition"] == attack)
        ]
        z = off.merge(on, on=["block", "head"], suffixes=("_off", "_on"), how="inner")
        for row in z.itertuples(index=False):
            rn_rows.append({
                "comparison": attack,
                "block": int(row.block),
                "head": int(row.head),
                "motif_off": row.primary_motif_off,
                "motif_on": row.primary_motif_on,
                "motif_changed": bool(row.primary_motif_off != row.primary_motif_on),
                "push_pull_off": bool(row.push_pull_sign_off),
                "push_pull_on": bool(row.push_pull_sign_on),
                "rn_delta_s_C2R": float(row.frozen_C2R_slope_rel_on - row.frozen_C2R_slope_rel_off),
                "rn_delta_s_R2C": float(row.frozen_R2C_slope_rel_on - row.frozen_R2C_slope_rel_off),
                "rn_delta_c_C2R": float(row.frozen_C2R_curvature_rel_on - row.frozen_C2R_curvature_rel_off),
                "rn_delta_c_R2C": float(row.frozen_R2C_curvature_rel_on - row.frozen_R2C_curvature_rel_off),
            })

    return (
        pd.DataFrame(transition_rows),
        pd.DataFrame(exception_rows),
        pd.DataFrame(rn_rows),
    )


# =============================================================================
# Plots
# =============================================================================


def _mean_curve_arrays(
    mean_curves: pd.DataFrame,
    rn_mode: str,
    condition: str,
    block: int,
    head: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = mean_curves[
        (mean_curves["rn_mode"] == rn_mode)
        & (mean_curves["condition"] == condition)
        & (mean_curves["block"] == int(block))
        & (mean_curves["head"] == int(head))
    ].sort_values("factor_mu2")
    if z.empty:
        raise RuntimeError(
            f"Missing mean curve rn={rn_mode}, condition={condition}, B{block}, H{head}"
        )
    return (
        z["factor_mu2"].to_numpy(np.float64),
        z["frozen_C2R_write"].to_numpy(np.float64),
        z["frozen_R2C_write"].to_numpy(np.float64),
    )


def _plot_curve_pair(
    ax,
    x: np.ndarray,
    c2r: np.ndarray,
    r2c: np.ndarray,
    *,
    scale: float,
    label: str,
    marker: str,
    linewidth: float = 1.35,
) -> None:
    line, = ax.plot(
        x,
        c2r / scale,
        marker=marker,
        linewidth=linewidth,
        label=f"{label} C→R",
    )
    ax.plot(
        x,
        r2c / scale,
        marker=marker,
        linewidth=linewidth,
        linestyle="--",
        color=line.get_color(),
        label=f"{label} R→C",
    )


def _sheet_for_states(
    mean_curves: pd.DataFrame,
    *,
    block: int,
    attack: str,
    states: Sequence[tuple[str, str, str, str]],
    out_path: Path,
    title: str,
) -> None:
    """
    states entries: (rn_mode, condition, display_label, marker)
    """
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 11.5))
    axes = axes.ravel()

    for head in range(16):
        ax = axes[head]
        series = []
        values = []
        for rn_mode, condition, display, marker in states:
            x, c, r = _mean_curve_arrays(
                mean_curves, rn_mode, condition, block, head
            )
            series.append((x, c, r, display, marker))
            values.extend([c, r])
        scale = max(max(float(np.max(v)), EPS) for v in values)
        for x, c, r, display, marker in series:
            _plot_curve_pair(
                ax, x, c, r,
                scale=scale,
                label=display,
                marker=marker,
            )
        ax.axvline(0.0, linewidth=.55, alpha=.35)
        ax.set_ylim(-.04, 1.06)
        ax.grid(alpha=.15)
        ax.set_title(f"B{int(block):02d} H{head:02d}", fontsize=9)
        ax.text(
            .02, .03, f"peak={scale:.3g}",
            transform=ax.transAxes, fontsize=6.5,
            ha="left", va="bottom",
        )
        if head == 0:
            ax.legend(fontsize=5.7, loc="upper right")

    fig.suptitle(title, fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, .95])
    savefig(fig, out_path)


def mean_curve_contact_sheets_no_rn(
    mean_curves: pd.DataFrame,
    out_root: Path,
    blocks: Sequence[int],
) -> list[Path]:
    """Compatibility view equivalent to the original no-RN experiment."""
    made = []
    for attack in ATTACK_ORDER:
        comp_dir = out_root / attack
        comp_dir.mkdir(parents=True, exist_ok=True)
        states = (
            ("rn_off", "NoRTA", "NoRTA", "o"),
            ("rn_off", attack, attack, "s"),
        )
        for block in blocks:
            path = comp_dir / f"B{int(block):02d}_{attack}_vs_NoRTA.png"
            _sheet_for_states(
                mean_curves,
                block=int(block),
                attack=attack,
                states=states,
                out_path=path,
                title=(
                    f"B{int(block):02d}: {attack} vs NoRTA — mean μ2 response (RN off)\n"
                    "solid=C→frozen-REG, dashed=frozen-REG→CLS; "
                    "each head normalized by one shared state/route maximum"
                ),
            )
            made.append(path)
    return made


def mean_curve_rn_triad_sheets(
    mean_curves: pd.DataFrame,
    out_root: Path,
    blocks: Sequence[int],
) -> list[Path]:
    """Primary appendix sheet: NoRTA vs attack vs attack+RN."""
    made = []
    for attack in ATTACK_ORDER:
        comp_dir = out_root / attack
        comp_dir.mkdir(parents=True, exist_ok=True)
        states = (
            ("rn_off", "NoRTA", "NoRTA", "o"),
            ("rn_off", attack, attack, "s"),
            ("rn_on", attack, f"{attack}+RN", "D"),
        )
        for block in blocks:
            path = comp_dir / f"B{int(block):02d}_NoRTA_vs_{attack}_vs_{attack}_RN.png"
            _sheet_for_states(
                mean_curves,
                block=int(block),
                attack=attack,
                states=states,
                out_path=path,
                title=(
                    f"B{int(block):02d}: NoRTA vs {attack} vs {attack}+RN — mean μ2 response\n"
                    "fixed register addresses are RN-off NoRTA B23 registers; "
                    "solid=C→R, dashed=R→CLS"
                ),
            )
            made.append(path)
    return made


def mean_curve_rn_factorial_sheets(
    mean_curves: pd.DataFrame,
    out_root: Path,
    blocks: Sequence[int],
) -> list[Path]:
    """Full clean/attack x RN audit, mainly for appendix/debugging."""
    made = []
    for attack in ATTACK_ORDER:
        comp_dir = out_root / attack
        comp_dir.mkdir(parents=True, exist_ok=True)
        states = (
            ("rn_off", "NoRTA", "NoRTA", "o"),
            ("rn_on", "NoRTA", "NoRTA+RN", "^"),
            ("rn_off", attack, attack, "s"),
            ("rn_on", attack, f"{attack}+RN", "D"),
        )
        for block in blocks:
            path = comp_dir / f"B{int(block):02d}_{attack}_RN_factorial.png"
            _sheet_for_states(
                mean_curves,
                block=int(block),
                attack=attack,
                states=states,
                out_path=path,
                title=(
                    f"B{int(block):02d}: {attack} × RN full μ2 factorial\n"
                    "fixed register addresses are RN-off NoRTA B23 registers"
                ),
            )
            made.append(path)
    return made


def changed_motif_contact_sheets(
    mean_curves: pd.DataFrame,
    rn_transitions: pd.DataFrame,
    out_root: Path,
) -> list[Path]:
    """
    Appendix contact sheets for heads whose *primary operational motif* changes
    when RN is inserted.

    One separate sheet is written for RTA and SynthRTA because the changed-head
    sets need not be nested.  Each panel shows the same three-state comparison
    used by the paper-facing B13 examples:

        NoRTA, attack, attack+RN

    solid  = CLS -> frozen-REG exact OV/write
    dashed = frozen-REG -> CLS exact OV/write

    The panel title records the categorical motif transition itself.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    summary_rows: list[dict[str, object]] = []

    for attack in ATTACK_ORDER:
        changed = rn_transitions[
            (rn_transitions["comparison"] == attack)
            & (rn_transitions["motif_changed"].astype(bool))
        ].copy()
        changed = changed.sort_values(["block", "head"]).reset_index(drop=True)

        for row in changed.itertuples(index=False):
            summary_rows.append({
                "comparison": attack,
                "block": int(row.block),
                "head": int(row.head),
                "motif_off": str(row.motif_off),
                "motif_on": str(row.motif_on),
            })

        if changed.empty:
            continue

        ncols = 3
        nrows = int(math.ceil(len(changed) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.0 * ncols, 4.1 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        states = (
            ("rn_off", "NoRTA", "NoRTA", "o"),
            ("rn_off", attack, attack, "s"),
            ("rn_on", attack, f"{attack}+RN", "D"),
        )

        for ax, row in zip(axes, changed.itertuples(index=False)):
            block = int(row.block)
            head = int(row.head)

            series = []
            values = []
            for rn_mode, condition, display, marker in states:
                x, c, r = _mean_curve_arrays(
                    mean_curves,
                    rn_mode,
                    condition,
                    block,
                    head,
                )
                series.append((x, c, r, display, marker))
                values.extend([c, r])

            scale = max(
                max(float(np.max(np.abs(v))), EPS)
                for v in values
            )

            for x, c, r, display, marker in series:
                _plot_curve_pair(
                    ax,
                    x,
                    c,
                    r,
                    scale=scale,
                    label=display,
                    marker=marker,
                    linewidth=1.25,
                )

            ax.axvline(0.0, linewidth=.55, alpha=.35)
            ax.set_ylim(-.04, 1.06)
            ax.grid(alpha=.15)
            ax.set_title(
                f"B{block:02d} H{head:02d} — "
                f"{str(row.motif_off).replace('_', ' ')} → "
                f"{str(row.motif_on).replace('_', ' ')}",
                fontsize=9,
            )
            ax.text(
                .02,
                .03,
                f"peak={scale:.3g}",
                transform=ax.transAxes,
                fontsize=6.5,
                ha="left",
                va="bottom",
            )
            ax.set_xlabel(r"CLS displacement $\alpha\mu_2$", fontsize=8)
            ax.set_ylabel("normalized exact OV/write", fontsize=8)

            # One legend per sheet is enough; placing it on panel 0 keeps this
            # consistent with the existing 16-head sheets.
            if ax is axes[0]:
                ax.legend(fontsize=6.0, loc="upper right")

        for ax in axes[len(changed):]:
            ax.set_axis_off()

        path = out_root / f"changed_primary_motif_heads_{attack}.png"
        fig.suptitle(
            f"Heads whose primary operational motif changes under RN — {attack}\n"
            "NoRTA vs attack vs attack+RN; "
            "solid=CLS→frozen-REG, dashed=frozen-REG→CLS",
            fontsize=13.5,
        )
        fig.tight_layout(rect=[0, 0, 1, .96])
        savefig(fig, path)
        made.append(path)

    summary_path = out_root / "changed_primary_motif_heads.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    made.append(summary_path)

    return made


def head_heatmap(
    stat_df: pd.DataFrame,
    comparison: str,
    metric: str,
    out_path: Path,
    title: str,
    *,
    rn_mode: Optional[str] = None,
    value_label: str = "mean paired delta",
) -> None:
    z = stat_df[
        (stat_df["comparison"] == comparison)
        & (stat_df["metric"] == metric)
    ]
    if rn_mode is not None and "rn_mode" in z.columns:
        z = z[z["rn_mode"] == rn_mode]
    if z.empty:
        return

    blocks = sorted(z["block"].unique())
    heads = sorted(z["head"].unique())
    mat = np.full((max(blocks) + 1, max(heads) + 1), np.nan)
    qmat = np.full_like(mat, np.nan)
    for row in z.itertuples(index=False):
        mat[int(row.block), int(row.head)] = float(row.mean_delta)
        q = row.wilcoxon_q
        if not np.isfinite(q):
            q = row.t_q
        qmat[int(row.block), int(row.head)] = float(q)

    vmax = max(float(np.nanmax(np.abs(mat))), 1e-8)
    fig, ax = plt.subplots(figsize=(10.3, 8.5))
    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    sig = np.argwhere(qmat < 0.05)
    if len(sig):
        ax.scatter(sig[:, 1], sig[:, 0], marker=".", s=9)
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"H{i}" for i in range(16)], rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"B{i:02d}" for i in range(mat.shape[0])])
    ax.set_xlabel("attention head")
    ax.set_ylabel("ViT block")
    ax.set_title(title + "\nblack dot: paired q<0.05")
    fig.colorbar(im, ax=ax, fraction=.035, pad=.025, label=value_label)
    fig.tight_layout()
    savefig(fig, out_path)


def make_key_heatmaps(
    mu2_stats: pd.DataFrame,
    pop_stats: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    made = []
    for rn_mode in RN_MODE_ORDER:
        mode_dir = out_dir / rn_mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for attack in ATTACK_ORDER:
            specs = [
                (mu2_stats, "frozen_C2R_slope_rel", f"{attack}_delta_s_C2R.png", f"{attack}: change in μ2 slope of CLS→frozen-REG write"),
                (mu2_stats, "frozen_R2C_slope_rel", f"{attack}_delta_s_R2C.png", f"{attack}: change in μ2 slope of frozen-REG→CLS write"),
                (pop_stats, "TEXT_P2C_write_share", f"{attack}_TEXT_to_CLS_write_share_delta.png", f"{attack}: TEXT→CLS exact OV-write share change"),
                (pop_stats, "TEXT_P2C_attn_share", f"{attack}_TEXT_to_CLS_attn_share_delta.png", f"{attack}: TEXT→CLS attention-share change"),
                (pop_stats, "TEXT_C2P_write_share", f"{attack}_CLS_to_TEXT_write_share_delta.png", f"{attack}: CLS→TEXT exact OV-write share change"),
                (pop_stats, "REG_FROZEN_P2C_write_share", f"{attack}_REG_FROZEN_to_CLS_write_share_delta.png", f"{attack}: frozen-REG→CLS write-share change"),
                (pop_stats, "MU1_CACHE_P2C_write_share", f"{attack}_MU1_CACHE_to_CLS_write_share_delta.png", f"{attack}: diffuse μ1-cache→CLS write-share change"),
                (pop_stats, "SCRATCH_RESIDUAL_P2C_write_share", f"{attack}_SCRATCH_RESIDUAL_to_CLS_write_share_delta.png", f"{attack}: operational scratchpad→CLS write-share change"),
            ]
            for source_df, metric, filename, title in specs:
                path = mode_dir / filename
                head_heatmap(
                    source_df,
                    attack,
                    metric,
                    path,
                    f"{title} [{rn_mode}]",
                    rn_mode=rn_mode,
                    value_label="attack − NoRTA",
                )
                if path.is_file():
                    made.append(path)
    return made


def make_rn_effect_heatmaps(
    mu2_rn_stats: pd.DataFrame,
    pop_rn_stats: pd.DataFrame,
    out_dir: Path,
    *,
    extended: bool = False,
) -> list[Path]:
    """Difference-of-differences heatmaps: negative = RN reduces attack effect."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for attack in ATTACK_ORDER:
        specs = [
            (mu2_rn_stats, "frozen_C2R_slope_rel", f"{attack}_RN_effect_mu2_s_C2R.png", f"{attack}: RN effect on μ2 CLS→REG slope"),
            (mu2_rn_stats, "frozen_R2C_slope_rel", f"{attack}_RN_effect_mu2_s_R2C.png", f"{attack}: RN effect on μ2 REG→CLS slope"),
            (pop_rn_stats, "TEXT_P2C_write_share", f"{attack}_RN_effect_TEXT_to_CLS_write_share.png", f"{attack}: RN effect on TEXT→CLS exact OV/write share"),
            (pop_rn_stats, "TEXT_P2C_attn_share", f"{attack}_RN_effect_TEXT_to_CLS_attn_share.png", f"{attack}: RN effect on TEXT→CLS attention share"),
            (pop_rn_stats, "REG_FROZEN_P2C_write_share", f"{attack}_RN_effect_REG_to_CLS_write_share.png", f"{attack}: RN effect on frozen-REG→CLS write share"),
        ]
        if extended:
            specs += [
                (pop_rn_stats, "MU1_CACHE_P2C_write_share", f"{attack}_RN_effect_MU1_CACHE_to_CLS_write_share.png", f"{attack}: RN effect on μ1-cache→CLS write share"),
                (pop_rn_stats, "SCRATCH_RESIDUAL_P2C_write_share", f"{attack}_RN_effect_SCRATCH_to_CLS_write_share.png", f"{attack}: RN effect on scratch-residual→CLS write share"),
            ]
        for source_df, metric, filename, title in specs:
            path = out_dir / filename
            head_heatmap(
                source_df,
                attack,
                metric,
                path,
                title,
                value_label="[(attack-clean) RN] − [(attack-clean) no RN]",
            )
            if path.is_file():
                made.append(path)
    return made


def text_mask_sheets(
    examples: list[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    per_page = 12
    for page_idx in range(int(math.ceil(len(examples) / per_page))):
        page = examples[page_idx * per_page:(page_idx + 1) * per_page]
        fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.5))
        axes = axes.ravel()
        for ax, ex in zip(axes, page):
            image = ex["image"]
            mask = np.asarray(ex["mask"], bool)
            side = int(ex["side"])
            h, w = image.shape[:2]
            ax.imshow(image)
            overlay = np.ma.array(
                mask.reshape(side, side).astype(float),
                mask=~mask.reshape(side, side),
            )
            ax.imshow(
                overlay,
                extent=(0, w, h, 0),
                interpolation="nearest",
                alpha=.55,
                cmap="Reds",
                vmin=0,
                vmax=1,
            )
            ax.set_title(
                f"{ex['attack']} {ex['sample_key']}\n{int(mask.sum())} text patches",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes[len(page):]:
            ax.axis("off")
        fig.suptitle(
            "TEXT masks from exact post-preprocess NoRTA↔attack pixel differences",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, .96])
        path = out_dir / f"text_mask_sheet{page_idx + 1:03d}.png"
        savefig(fig, path)
        made.append(path)
    return made


def b13_rn_head_candidates(
    mean_curves: pd.DataFrame,
    condition_fits: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank B13 heads within their RN-off/NoRTA motif by how strongly RN moves the
    attacked response surface.  Also report whether the move approaches clean.
    """
    baseline = condition_fits[
        (condition_fits["rn_mode"] == "rn_off")
        & (condition_fits["condition"] == "NoRTA")
        & (condition_fits["block"] == RN_INSERT_BLOCK)
    ].set_index("head")

    rows = []
    for head in range(16):
        if head not in baseline.index:
            continue
        family = str(baseline.loc[head, "primary_motif"])
        per_attack = []
        for attack in ATTACK_ORDER:
            series = []
            for rn_mode, condition in (
                ("rn_off", "NoRTA"),
                ("rn_off", attack),
                ("rn_on", attack),
                ("rn_on", "NoRTA"),
            ):
                x, c, r = _mean_curve_arrays(
                    mean_curves, rn_mode, condition, RN_INSERT_BLOCK, head
                )
                series.append((rn_mode, condition, c, r))
            common_scale = max(
                max(float(np.max(c)), float(np.max(r)), EPS)
                for _rm, _cond, c, r in series
            )
            vec = {
                (rm, cond): np.concatenate([c / common_scale, r / common_scale])
                for rm, cond, c, r in series
            }
            clean = vec[("rn_off", "NoRTA")]
            atk = vec[("rn_off", attack)]
            atk_rn = vec[("rn_on", attack)]
            clean_rn = vec[("rn_on", "NoRTA")]
            d_attack_clean = float(np.linalg.norm(atk - clean))
            d_rn_attack = float(np.linalg.norm(atk_rn - atk))
            d_rn_clean = float(np.linalg.norm(atk_rn - clean))
            d_clean_rn = float(np.linalg.norm(clean_rn - clean))
            d_attackrn_cleanrn = float(np.linalg.norm(atk_rn - clean_rn))
            did_vec = (atk_rn - clean_rn) - (atk - clean)
            d_rn_did = float(np.linalg.norm(did_vec))
            restoration = (
                (d_attack_clean - d_rn_clean) / d_attack_clean
                if d_attack_clean > 1e-6
                else float("nan")
            )
            per_attack.append((
                attack,
                d_attack_clean,
                d_rn_attack,
                d_rn_clean,
                d_clean_rn,
                d_attackrn_cleanrn,
                d_rn_did,
                restoration,
            ))

        row = {
            "block": RN_INSERT_BLOCK,
            "head": int(head),
            "baseline_motif": family,
            "baseline_s_C2R": float(baseline.loc[head, "frozen_C2R_slope_rel"]),
            "baseline_s_R2C": float(baseline.loc[head, "frozen_R2C_slope_rel"]),
        }
        for attack, d0, de, dr, dc, drr, ddid, rest in per_attack:
            row[f"{attack}_attack_clean_distance"] = d0
            row[f"{attack}_rn_effect_distance"] = de
            row[f"{attack}_attackRN_clean_distance"] = dr
            row[f"{attack}_cleanRN_clean_distance"] = dc
            row[f"{attack}_attackRN_cleanRN_distance"] = drr
            row[f"{attack}_rn_did_distance"] = ddid
            row[f"{attack}_restoration_fraction"] = rest
        row["mean_rn_effect_distance"] = float(np.mean([x[2] for x in per_attack]))
        row["mean_rn_did_distance"] = float(np.mean([x[6] for x in per_attack]))
        restoration_vals = np.asarray([x[7] for x in per_attack], np.float64)
        row["mean_restoration_fraction"] = (
            float(np.nanmean(restoration_vals))
            if np.any(np.isfinite(restoration_vals))
            else float("nan")
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Family-relative rank uses the clean-controlled RN difference-of-differences,
    # so a generic clean+RN shift cannot win the paper-head selection by itself.
    df["family_effect_rank"] = (
        df.groupby("baseline_motif")["mean_rn_did_distance"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return df.sort_values(
        ["baseline_motif", "family_effect_rank", "head"]
    ).reset_index(drop=True)


def choose_b13_paper_heads(candidates: pd.DataFrame) -> dict[str, int]:
    families = ("push_pull_exchange", "read_gain", "broadcast_gain")
    selected: dict[str, int] = {}
    for family in families:
        z = candidates[candidates["baseline_motif"] == family]
        if len(z):
            selected[family] = int(
                z.sort_values(
                    ["mean_rn_did_distance", "mean_restoration_fraction"],
                    ascending=[False, False],
                ).iloc[0]["head"]
            )
            continue

        # Conservative fallback based on baseline slopes if classifier thresholds
        # happen to place every head in adjacent categories.
        z = candidates.copy()
        if z.empty:
            continue
        if family == "push_pull_exchange":
            score = np.maximum(-z["baseline_s_C2R"], 0) * np.maximum(z["baseline_s_R2C"], 0)
        elif family == "read_gain":
            score = np.abs(z["baseline_s_R2C"]) / np.maximum(np.abs(z["baseline_s_C2R"]), EPS)
        else:
            score = np.abs(z["baseline_s_C2R"]) / np.maximum(np.abs(z["baseline_s_R2C"]), EPS)
        selected[family] = int(z.iloc[int(np.argmax(score.to_numpy()))]["head"])
    return selected


def _plot_b13_representatives(
    mean_curves: pd.DataFrame,
    selected: Mapping[str, int],
    attack: str,
    out_path: Path,
) -> None:
    families = [f for f in ("push_pull_exchange", "read_gain", "broadcast_gain") if f in selected]
    if not families:
        return
    fig, axes = plt.subplots(1, len(families), figsize=(4.35 * len(families), 3.8), squeeze=False)
    states = (
        ("rn_off", "NoRTA", "NoRTA", "o"),
        ("rn_off", attack, attack, "s"),
        ("rn_on", attack, f"{attack}+RN", "D"),
    )
    for ax, family in zip(axes.ravel(), families):
        head = selected[family]
        series = []
        values = []
        for rn_mode, condition, display, marker in states:
            x, c, r = _mean_curve_arrays(mean_curves, rn_mode, condition, RN_INSERT_BLOCK, head)
            series.append((x, c, r, display, marker))
            values.extend([c, r])
        scale = max(max(float(np.max(v)), EPS) for v in values)
        for x, c, r, display, marker in series:
            _plot_curve_pair(ax, x, c, r, scale=scale, label=display, marker=marker, linewidth=1.7)
        ax.axvline(0.0, linewidth=.7, alpha=.35)
        ax.set_ylim(-.04, 1.06)
        ax.grid(alpha=.15)
        label = family.replace("_", " ")
        ax.set_title(f"B13 H{head:02d} — {label}")
        ax.set_xlabel(r"CLS displacement $\alpha\mu_2$")
        ax.text(.02, .03, f"raw peak={scale:.3g}", transform=ax.transAxes, fontsize=7)
    axes[0, 0].set_ylabel("normalized exact OV/write response")
    axes[0, -1].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"B13 RN intervention under the fixed μ2 slider — {attack}\n"
        "solid=CLS→frozen-REG; dashed=frozen-REG→CLS",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, .91])
    savefig(fig, out_path, dpi=220)


def plot_b13_representatives_both(
    mean_curves: pd.DataFrame,
    selected: Mapping[str, int],
    out_path: Path,
) -> None:
    families = [f for f in ("push_pull_exchange", "read_gain", "broadcast_gain") if f in selected]
    if not families:
        return
    fig, axes = plt.subplots(2, len(families), figsize=(4.3 * len(families), 7.0), squeeze=False, sharex=True)
    for ri, attack in enumerate(ATTACK_ORDER):
        states = (
            ("rn_off", "NoRTA", "NoRTA", "o"),
            ("rn_off", attack, attack, "s"),
            ("rn_on", attack, f"{attack}+RN", "D"),
        )
        for ci, family in enumerate(families):
            ax = axes[ri, ci]
            head = selected[family]
            series = []
            values = []
            for rn_mode, condition, display, marker in states:
                x, c, r = _mean_curve_arrays(mean_curves, rn_mode, condition, RN_INSERT_BLOCK, head)
                series.append((x, c, r, display, marker))
                values.extend([c, r])
            scale = max(max(float(np.max(v)), EPS) for v in values)
            for x, c, r, display, marker in series:
                _plot_curve_pair(ax, x, c, r, scale=scale, label=display, marker=marker, linewidth=1.5)
            ax.axvline(0.0, linewidth=.6, alpha=.35)
            ax.set_ylim(-.04, 1.06)
            ax.grid(alpha=.15)
            if ri == 0:
                ax.set_title(f"H{head:02d} — {family.replace('_', ' ')}")
            if ci == 0:
                ax.set_ylabel(f"{attack}\nnormalized response")
            if ri == 1:
                ax.set_xlabel(r"CLS displacement $\alpha\mu_2$")
    axes[0, -1].legend(fontsize=6.5, loc="best")
    axes[1, -1].legend(fontsize=6.5, loc="best")
    fig.suptitle(
        "NoRTA vs attack vs attack+RN under the B13 μ2 slider\n"
        "solid=CLS→frozen-REG; dashed=frozen-REG→CLS",
        fontsize=13.5,
    )
    fig.tight_layout(rect=[0, 0, 1, .93])
    savefig(fig, out_path, dpi=220)


def make_b13_paper_outputs(
    mean_curves: pd.DataFrame,
    condition_fits: pd.DataFrame,
    out_dir: Path,
) -> tuple[list[Path], pd.DataFrame, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = b13_rn_head_candidates(mean_curves, condition_fits)
    candidates_path = out_dir / "b13_rn_head_candidates.csv"
    candidates.to_csv(candidates_path, index=False)
    selected = choose_b13_paper_heads(candidates)
    save_json(out_dir / "b13_rn_selected_heads.json", selected)

    made = [candidates_path, out_dir / "b13_rn_selected_heads.json"]
    for attack in ATTACK_ORDER:
        path = out_dir / f"b13_rn_mu2_representative_heads_{attack}.png"
        _plot_b13_representatives(mean_curves, selected, attack, path)
        if path.is_file():
            made.append(path)
    both = out_dir / "b13_rn_mu2_representative_heads_both.png"
    plot_b13_representatives_both(mean_curves, selected, both)
    if both.is_file():
        made.append(both)
    return made, candidates, selected


# =============================================================================
# Leaderboards / summary / legacy parity audit
# =============================================================================


def text_funnel_leaderboards(
    pop_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_metric = "TEXT_P2C_write_share"
    z = pop_stats[pop_stats["metric"] == key_metric].copy()
    if z.empty:
        return pd.DataFrame(), pd.DataFrame()
    z["abs_mean_delta"] = np.abs(z["mean_delta"])
    leaderboard = z.sort_values(
        ["rn_mode", "comparison", "abs_mean_delta"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    leaders = (
        z.sort_values(
            ["rn_mode", "comparison", "block", "mean_delta"],
            ascending=[True, True, True, False],
        )
        .groupby(["rn_mode", "comparison", "block"], as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
    return leaderboard, leaders


def compare_legacy_native_population(
    fresh_traffic: pd.DataFrame,
    legacy_path: Path,
) -> pd.DataFrame:
    """
    Compare freshly recomputed rn_off traffic against the old no-RN CSV.

    The old file is never used to fill missing fresh measurements.  This is
    strictly a parity audit to detect accidental definition drift after making
    the script standalone.
    """
    if not legacy_path.is_file():
        print(f"[legacy parity] not found; skipping: {legacy_path}")
        return pd.DataFrame()

    legacy = pd.read_csv(legacy_path)
    fresh = fresh_traffic[fresh_traffic["rn_mode"] == "rn_off"].copy()
    key = ["sample_key", "comparison", "condition", "block", "head"]
    missing_key = [c for c in key if c not in legacy.columns or c not in fresh.columns]
    if missing_key:
        print(f"[legacy parity] incompatible key columns, skipping: {missing_key}")
        return pd.DataFrame()

    numeric = sorted(
        c for c in fresh.columns
        if c in legacy.columns
        and c not in key
        and pd.api.types.is_numeric_dtype(fresh[c])
        and pd.api.types.is_numeric_dtype(legacy[c])
    )
    if not numeric:
        print("[legacy parity] no shared numeric columns; skipping")
        return pd.DataFrame()

    a = fresh[key + numeric]
    b = legacy[key + numeric]
    merged = a.merge(b, on=key, suffixes=("_fresh", "_legacy"), how="inner")
    rows = []
    for metric in numeric:
        x = pd.to_numeric(merged[f"{metric}_fresh"], errors="coerce").to_numpy(np.float64)
        y = pd.to_numeric(merged[f"{metric}_legacy"], errors="coerce").to_numpy(np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        d = x[valid] - y[valid]
        rows.append({
            "metric": metric,
            "matched_rows": int(valid.sum()),
            "mean_abs_diff": float(np.mean(np.abs(d))),
            "max_abs_diff": float(np.max(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d * d))),
            "pearson_r": safe_corr(x[valid], y[valid]),
        })
    out = pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False).reset_index(drop=True)
    print(
        f"[legacy parity] matched {len(merged)} rows; "
        f"shared numeric metrics={len(out)}"
    )
    return out


def write_summary(
    out: Path,
    args,
    condition_fits: pd.DataFrame,
    transitions: pd.DataFrame,
    rn_transitions: pd.DataFrame,
    exceptions: pd.DataFrame,
    leaders: pd.DataFrame,
    mu2_rn_stats: pd.DataFrame,
    pop_rn_stats: pd.DataFrame,
    reg_audit: pd.DataFrame,
    candidates: pd.DataFrame,
    selected_heads: Mapping[str, int],
    legacy_parity: pd.DataFrame,
) -> None:
    lines = [
        "RTA-100 MU2 + TOKEN-POPULATION GRAMMAR WITH RN",
        "=" * 88,
        "",
        f"model: {args.model}",
        f"dataset: {args.dataset} / {args.split}",
        f"slider factors: {args.mu2_factors}",
        f"RN: exact learned token from {args.xattn_checkpoint}",
        f"RN insertion: pre-B{args.rn_insert_block}",
        "primary frozen-register policy: RN-off NoRTA B23 addresses across ALL states",
        "",
        "Condition-mean push-pull sign counts by block:",
        "",
    ]

    for rn_mode in RN_MODE_ORDER:
        lines.append(f"[{rn_mode}]")
        lines.append("block   NoRTA   RTA   SynthRTA")
        lines.append("-" * 34)
        for block in sorted(condition_fits["block"].unique()):
            counts = {}
            for cond in CONDITION_ORDER:
                z = condition_fits[
                    (condition_fits["rn_mode"] == rn_mode)
                    & (condition_fits["condition"] == cond)
                    & (condition_fits["block"] == block)
                ]
                counts[cond] = int(np.sum(z["push_pull_sign"]))
            lines.append(
                f"B{int(block):02d}     {counts['NoRTA']:>2d}/16   "
                f"{counts['RTA']:>2d}/16   {counts['SynthRTA']:>2d}/16"
            )
        lines.append("")

    lines += [
        "B13 representative-head selection:",
        "",
    ]
    for family in ("push_pull_exchange", "read_gain", "broadcast_gain"):
        if family not in selected_heads:
            continue
        h = int(selected_heads[family])
        z = candidates[candidates["head"] == h]
        if len(z):
            r = z.iloc[0]
            lines.append(
                f"  {family:<22} H{h:02d}  "
                f"mean RN-DID displacement={float(r.mean_rn_did_distance):.4f}  "
                f"raw RN displacement={float(r.mean_rn_effect_distance):.4f}  "
                f"mean restoration={float(r.mean_restoration_fraction):+.3f}"
            )
        else:
            lines.append(f"  {family:<22} H{h:02d}")

    lines += [
        "",
        "Attack-induced motif transitions within each RN state:",
        "",
    ]
    if len(transitions):
        for rn_mode in RN_MODE_ORDER:
            for attack in ATTACK_ORDER:
                z = transitions[
                    (transitions["rn_mode"] == rn_mode)
                    & (transitions["comparison"] == attack)
                ]
                lines.append(
                    f"  {rn_mode:<6} {attack:<9}: "
                    f"{int(z['motif_changed'].sum())}/{len(z)} block-heads changed motif"
                )

    lines += [
        "",
        "RN-induced motif changes on attacked images:",
        "",
    ]
    if len(rn_transitions):
        for attack in ATTACK_ORDER:
            z = rn_transitions[rn_transitions["comparison"] == attack]
            lines.append(
                f"  {attack:<9}: {int(z['motif_changed'].sum())}/{len(z)} "
                "block-heads changed primary operational motif"
            )

    lines += [
        "",
        "Largest RN difference-of-differences in B13 mu2 slopes:",
        "  DID = [(attack-clean) RN] - [(attack-clean) no RN]",
        "",
    ]
    if len(mu2_rn_stats):
        zall = mu2_rn_stats[
            (mu2_rn_stats["block"] == RN_INSERT_BLOCK)
            & (mu2_rn_stats["metric"].isin(["frozen_C2R_slope_rel", "frozen_R2C_slope_rel"]))
        ].copy()
        zall["abs_delta"] = np.abs(zall["mean_delta"])
        for attack in ATTACK_ORDER:
            lines.append(f"  [{attack}]")
            for row in zall[zall["comparison"] == attack].sort_values("abs_delta", ascending=False).head(10).itertuples(index=False):
                q = row.wilcoxon_q if np.isfinite(row.wilcoxon_q) else row.t_q
                lines.append(
                    f"    B13 H{int(row.head):02d} {row.metric:<24} "
                    f"DID={float(row.mean_delta):+.4f} dz={float(row.cohen_dz):+.3f} q={float(q):.3g}"
                )

    lines += [
        "",
        "Largest positive TEXT->CLS write-share attack shifts:",
        "",
    ]
    if len(leaders):
        for rn_mode in RN_MODE_ORDER:
            for attack in ATTACK_ORDER:
                lines.append(f"  [{rn_mode} / {attack}]")
                z = leaders[
                    (leaders["rn_mode"] == rn_mode)
                    & (leaders["comparison"] == attack)
                ].sort_values("mean_delta", ascending=False).head(8)
                for row in z.itertuples(index=False):
                    q = row.wilcoxon_q if np.isfinite(row.wilcoxon_q) else row.t_q
                    lines.append(
                        f"    B{int(row.block):02d} H{int(row.head):02d} "
                        f"Δshare={float(row.mean_delta):+.4f} dz={float(row.cohen_dz):+.3f} q={float(q):.3g}"
                    )

    lines += [
        "",
        "Largest RN DID effects on TEXT->CLS write share:",
        "",
    ]
    if len(pop_rn_stats):
        z = pop_rn_stats[pop_rn_stats["metric"] == "TEXT_P2C_write_share"].copy()
        z["abs_delta"] = np.abs(z["mean_delta"])
        for attack in ATTACK_ORDER:
            lines.append(f"  [{attack}]")
            for row in z[z["comparison"] == attack].sort_values("abs_delta", ascending=False).head(10).itertuples(index=False):
                q = row.wilcoxon_q if np.isfinite(row.wilcoxon_q) else row.t_q
                lines.append(
                    f"    B{int(row.block):02d} H{int(row.head):02d} "
                    f"RN-DID={float(row.mean_delta):+.4f} dz={float(row.cohen_dz):+.3f} q={float(q):.3g}"
                )

    lines += [
        "",
        "Register relocation audit vs RN-off clean frozen addresses:",
        "",
    ]
    if len(reg_audit):
        for rn_mode in RN_MODE_ORDER:
            for cond in CONDITION_ORDER:
                z = reg_audit[
                    (reg_audit["rn_mode"] == rn_mode)
                    & (reg_audit["condition"] == cond)
                ]
                if len(z):
                    lines.append(
                        f"  {rn_mode:<6} {cond:<9}: mean Jaccard={float(z['frozen_own_jaccard'].mean()):.3f}"
                    )

    lines += [
        "",
        "Legacy native_population_traffic parity audit:",
        "",
    ]
    if legacy_parity.empty:
        lines.append("  not available / not comparable")
    else:
        worst = legacy_parity.sort_values("max_abs_diff", ascending=False).head(8)
        for row in worst.itertuples(index=False):
            lines.append(
                f"  {row.metric:<48} max|Δ|={float(row.max_abs_diff):.4g} "
                f"mean|Δ|={float(row.mean_abs_diff):.4g} r={float(row.pearson_r):.5f}"
            )

    lines += [
        "",
        "Interpretation guardrails:",
        "  * REG_FROZEN is RN-off NoRTA B23 and is fixed across condition and RN state.",
        "  * TEXT uses the same attack-derived spatial mask on paired clean/attacked images.",
        "  * RN-effect statistics use clean/attack difference-of-differences.",
        "  * SCRATCH_RESIDUAL remains an operational residual-workspace candidate.",
        "  * The old native_population_traffic.csv is parity-only, never a data fallback.",
        "",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main extraction loop
# =============================================================================


def _prepare_one_triplet(
    pair: Mapping[str, Any],
    dsets: Mapping[str, Any],
    preprocess,
    side: int,
    args,
) -> dict[str, Any]:
    pil_map: dict[str, Image.Image] = {}
    prep_map: dict[str, torch.Tensor] = {}
    meta_map: dict[str, dict[str, str]] = {}

    for cond in CONDITION_ORDER:
        row = dsets[cond][int(pair[f"{cond}_index"])]
        pil = row_image(row)
        prep = preprocess(pil)
        pil_map[cond] = pil
        prep_map[cond] = prep
        meta_map[cond] = {
            "object_label": str(row.get("object_label", "")),
            "attack_word": str(row.get("attack_word", "")),
            "row_id": str(row.get("id", "")),
        }

    masks: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    audits: dict[str, dict[str, float]] = {}
    for attack in ATTACK_ORDER:
        mask, score, audit = text_mask_from_preprocessed_pair(
            prep_map["NoRTA"],
            prep_map[attack],
            side,
            args,
        )
        masks[attack] = mask
        scores[attack] = score
        audits[attack] = audit

    return {
        "sample_key": str(pair["sample_key"]),
        "pil": pil_map,
        "prep": prep_map,
        "meta": meta_map,
        "text_masks": masks,
        "text_scores": scores,
        "text_audits": audits,
    }


def _prepare_batch_triplets(
    batch_pairs: Sequence[Mapping[str, Any]],
    dsets: Mapping[str, Any],
    preprocess,
    side: int,
    args,
) -> list[dict[str, Any]]:
    workers = max(0, int(args.preprocess_workers))
    if workers <= 1 or len(batch_pairs) <= 1:
        return [
            _prepare_one_triplet(pair, dsets, preprocess, side, args)
            for pair in batch_pairs
        ]

    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(batch_pairs))) as ex:
            return list(ex.map(
                lambda pair: _prepare_one_triplet(pair, dsets, preprocess, side, args),
                batch_pairs,
            ))
    except Exception as exc:
        # No scientific fallback or changed data path: simply redo the same pure
        # preprocessing serially if the host's dataset/PIL stack dislikes threads.
        print(f"[preprocess] threaded path failed ({exc}); retrying serially")
        return [
            _prepare_one_triplet(pair, dsets, preprocess, side, args)
            for pair in batch_pairs
        ]


def process_dataset(
    bundle: ModelBundle,
    dsets,
    pairs,
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    rn_payload: RNPayload,
    args,
):
    curve_rows: list[dict[str, Any]] = []
    pop_rows: list[dict[str, Any]] = []
    text_audit_rows: list[dict[str, Any]] = []
    reg_audit_rows: list[dict[str, Any]] = []
    text_examples: list[dict[str, Any]] = []

    batch_triplets = int(args.batch_triplets)
    patch_count_expected = int(bundle.model.visual.positional_embedding.shape[0] - 1)
    side_expected = side_from_patch_count(patch_count_expected)

    for start in tqdm(
        range(0, len(pairs), batch_triplets),
        desc="RTA-100 RN-factorial batches",
        unit="batch",
    ):
        batch_pairs = pairs[start:start + batch_triplets]
        n_trip = len(batch_pairs)

        prepared = _prepare_batch_triplets(
            batch_pairs,
            dsets,
            bundle.preprocess,
            side_expected,
            args,
        )

        physical_images: list[torch.Tensor] = []
        physical_meta: list[dict[str, Any]] = []
        text_masks_by_triplet: list[dict[str, np.ndarray]] = []

        # Physical image order: t0 NoRTA,RTA,Synth; t1 NoRTA,RTA,Synth; ...
        for local_t, item in enumerate(prepared):
            for cond in CONDITION_ORDER:
                physical_images.append(item["prep"][cond])
                m = item["meta"][cond]
                physical_meta.append({
                    "sample_key": item["sample_key"],
                    "condition": cond,
                    "triplet_local": local_t,
                    "object_label": m["object_label"],
                    "attack_word": m["attack_word"],
                    "row_id": m["row_id"],
                })

            text_masks_by_triplet.append(item["text_masks"])
            for attack in ATTACK_ORDER:
                text_audit_rows.append({
                    "sample_key": item["sample_key"],
                    "attack": attack,
                    **item["text_audits"][attack],
                })
                if len(text_examples) < int(args.text_mask_example_count):
                    text_examples.append({
                        "sample_key": item["sample_key"],
                        "attack": attack,
                        "image": np.asarray(item["pil"][attack].resize((224, 224))),
                        "mask": item["text_masks"][attack].copy(),
                        "side": side_expected,
                    })

        images_bchw = torch.stack(physical_images, dim=0).contiguous()

        (
            pre_states_by_mode,
            final_x_by_mode,
            own_reg_masks_by_mode,
            patch_count,
            side,
        ) = capture_pre_states_dual_batch(
            bundle,
            images_bchw,
            rn_payload,
            args,
        )
        if patch_count != patch_count_expected or side != side_expected:
            raise RuntimeError(
                f"Patch geometry changed unexpectedly: P={patch_count}, side={side}; "
                f"expected P={patch_count_expected}, side={side_expected}"
            )

        Bphys = len(physical_meta)

        # PRIMARY anchor: RN-off clean B23 register addresses, frozen across the
        # complete condition x RN factorial.
        frozen_reg_masks = np.zeros_like(own_reg_masks_by_mode["rn_off"])
        for t in range(n_trip):
            clean_idx = 3 * t
            frozen = own_reg_masks_by_mode["rn_off"][clean_idx]
            for j in range(3):
                frozen_reg_masks[clean_idx + j] = frozen

        for rn_mode in RN_MODE_ORDER:
            own_reg_masks = own_reg_masks_by_mode[rn_mode]
            for i, meta in enumerate(physical_meta):
                a = frozen_reg_masks[i]
                b = own_reg_masks[i]
                inter = int(np.sum(a & b))
                union = int(np.sum(a | b))
                reg_audit_rows.append({
                    "sample_key": meta["sample_key"],
                    "rn_mode": rn_mode,
                    "condition": meta["condition"],
                    "frozen_reg_count": int(a.sum()),
                    "own_reg_count": int(b.sum()),
                    "intersection": inter,
                    "union": union,
                    "frozen_own_jaccard": float(inter / max(union, 1)),
                })

        # Logical population views: each attack has paired clean-site and attack
        # rows.  The same text-site mask is used in both RN states.
        logical_meta: list[dict[str, Any]] = []
        logical_phys: list[int] = []
        logical_text: list[np.ndarray] = []
        for t, pair in enumerate(batch_pairs):
            clean_idx = 3 * t
            rta_idx = clean_idx + 1
            synth_idx = clean_idx + 2
            for attack, attack_idx in (("RTA", rta_idx), ("SynthRTA", synth_idx)):
                mask = text_masks_by_triplet[t][attack]
                logical_meta.append({
                    "sample_key": pair["sample_key"],
                    "comparison": attack,
                    "condition": "NoRTA",
                    "object_label": physical_meta[clean_idx]["object_label"],
                    "attack_word": physical_meta[clean_idx]["attack_word"],
                })
                logical_phys.append(clean_idx)
                logical_text.append(mask)
                logical_meta.append({
                    "sample_key": pair["sample_key"],
                    "comparison": attack,
                    "condition": attack,
                    "object_label": physical_meta[attack_idx]["object_label"],
                    "attack_word": physical_meta[attack_idx]["attack_word"],
                })
                logical_phys.append(attack_idx)
                logical_text.append(mask)

        logical_phys_np = np.asarray(logical_phys, int)
        logical_text_np = np.stack(logical_text, axis=0).astype(bool)

        n_blocks = len(bundle.model.visual.transformer.resblocks)
        for block_index in range(n_blocks):
            slider_by_mode: dict[str, Mapping[str, np.ndarray]] = {}
            native_by_mode: dict[str, Mapping[str, np.ndarray]] = {}

            # Before pre-B13 the state is exactly identical, so perform the
            # expensive B*F attention call once and reuse its patchwise result.
            if block_index < int(rn_payload.insert_block):
                slider, native = evaluate_block_slider_primitives(
                    bundle,
                    pre_states_by_mode["rn_off"][block_index],
                    block_index,
                    args.mu2_factors,
                    mu1,
                    mu2,
                    patch_count,
                )
                slider_by_mode["rn_off"] = slider
                slider_by_mode["rn_on"] = slider
                native_by_mode["rn_off"] = native
                native_by_mode["rn_on"] = native
            else:
                for rn_mode in RN_MODE_ORDER:
                    slider, native = evaluate_block_slider_primitives(
                        bundle,
                        pre_states_by_mode[rn_mode][block_index],
                        block_index,
                        args.mu2_factors,
                        mu1,
                        mu2,
                        patch_count,
                    )
                    slider_by_mode[rn_mode] = slider
                    native_by_mode[rn_mode] = native

            for rn_mode in RN_MODE_ORDER:
                curve_records = aggregate_slider_curves(
                    slider_by_mode[rn_mode],
                    args.mu2_factors,
                    frozen_reg_masks,
                    own_reg_masks_by_mode[rn_mode],
                )
                heads = curve_records[0]["frozen_C2R_write"].shape[1]
                for rec in curve_records:
                    factor = float(rec["factor"])
                    for i, meta in enumerate(physical_meta):
                        for h in range(heads):
                            curve_rows.append({
                                "sample_key": meta["sample_key"],
                                "rn_mode": rn_mode,
                                "condition": meta["condition"],
                                "object_label": meta["object_label"],
                                "attack_word": meta["attack_word"],
                                "block": int(block_index),
                                "head": int(h),
                                "factor_mu2": factor,
                                "frozen_C2R_write": float(rec["frozen_C2R_write"][i, h]),
                                "frozen_R2C_write": float(rec["frozen_R2C_write"][i, h]),
                                "frozen_C2R_attn": float(rec["frozen_C2R_attn"][i, h]),
                                "frozen_R2C_attn": float(rec["frozen_R2C_attn"][i, h]),
                                "own_C2R_write": float(rec["own_C2R_write"][i, h]),
                                "own_R2C_write": float(rec["own_R2C_write"][i, h]),
                                "own_C2R_attn": float(rec["own_C2R_attn"][i, h]),
                                "own_R2C_attn": float(rec["own_R2C_attn"][i, h]),
                                "CLS_source_mu1_patch_mean": float(rec["CLS_source_mu1_patch_mean"][i, h]),
                            })

                patch_bpd = (
                    pre_states_by_mode[rn_mode][block_index][1:1 + patch_count]
                    .permute(1, 0, 2)
                    .contiguous()
                    .detach()
                    .float()
                    .cpu()
                )
                patch_logical = patch_bpd[logical_phys_np]
                frozen_logical = frozen_reg_masks[logical_phys_np]
                own_logical = own_reg_masks_by_mode[rn_mode][logical_phys_np]

                masks, mask_audit = build_population_masks(
                    patch_logical,
                    frozen_logical,
                    own_logical,
                    logical_text_np,
                    mu1,
                    mu2,
                    args,
                )
                pop_metrics = population_metrics_for_logical_batch(
                    native_by_mode[rn_mode],
                    logical_phys_np,
                    masks,
                    patch_count,
                )
                H = next(iter(pop_metrics.values())).shape[1]
                for li, meta in enumerate(logical_meta):
                    audit_row = mask_audit[li]
                    for h in range(H):
                        row = {
                            "sample_key": meta["sample_key"],
                            "rn_mode": rn_mode,
                            "comparison": meta["comparison"],
                            "condition": meta["condition"],
                            "object_label": meta["object_label"],
                            "attack_word": meta["attack_word"],
                            "block": int(block_index),
                            "head": int(h),
                        }
                        for key, arr in pop_metrics.items():
                            row[key] = float(arr[li, h])
                        for key, value in audit_row.items():
                            row[f"mask_{key}"] = value
                        pop_rows.append(row)

            # Explicitly release large B*F*H*P arrays before next block.
            slider_by_mode.clear()
            native_by_mode.clear()

        del pre_states_by_mode, final_x_by_mode, own_reg_masks_by_mode, images_bchw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(pop_rows),
        pd.DataFrame(text_audit_rows),
        pd.DataFrame(reg_audit_rows),
        text_examples,
    )


# =============================================================================
# CLI
# =============================================================================


def analyze_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone RTA-100 all-head mu2 + native population grammar with "
            "a pre-B13 RN off/on factorial."
        )
    )

    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all paired triplets; positive = deterministic debug subset.",
    )

    parser.add_argument("--model", choices=MODEL_ORDER, default="finetune_stripped")
    parser.add_argument("--oracle_root", default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--baseline_dir", default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)

    parser.add_argument("--mu2_factors", default="-4,-2,0,2,4")
    parser.add_argument(
        "--sheet_blocks",
        default="13",
        help=(
            "Blocks for expensive 16-head contact sheets. Default=13 because the "
            "paper RN/mu2 close-loop is a B13 experiment. Use e.g. 13,20,22 or "
            "'all'. All blocks are always present in the CSV/statistical outputs."
        ),
    )
    parser.add_argument(
        "--include_no_rn_compat_sheets",
        action="store_true",
        help="Also render the old two-condition RN-off contact sheets for selected blocks.",
    )
    parser.add_argument(
        "--extended_stats",
        action="store_true",
        help=(
            "Run inferential tests on every legacy population enrichment/share metric. "
            "Default tests only the paper-relevant traffic set; raw CSVs remain exhaustive."
        ),
    )
    parser.add_argument(
        "--extended_plots",
        action="store_true",
        help=(
            "Render the exhaustive within-RN-state heatmap set and extra RN-effect "
            "population heatmaps. Default keeps the paper-oriented plot set compact."
        ),
    )
    parser.add_argument("--batch_triplets", type=int, default=4)
    parser.add_argument(
        "--preprocess_workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
        help=(
            "CPU thread count for image decode/preprocess. Set 0 or 1 for serial. "
            "The GPU forward remains single-device FP32."
        ),
    )
    parser.add_argument(
        "--prestate_storage",
        choices=("gpu", "cpu"),
        default="gpu",
        help=(
            "Keep block pre-states on GPU to avoid GPU->CPU->GPU round trips during "
            "the mu2 slider. For ViT-L/14 with the default batch this uses well under "
            "1 GB extra VRAM. Use cpu only if VRAM is unusually constrained."
        ),
    )

    # TEXT mask from exact post-preprocess pixel difference.
    parser.add_argument("--text_diff_mad_mult", type=float, default=6.0)
    parser.add_argument("--text_diff_relmax", type=float, default=0.12)
    parser.add_argument("--text_diff_absolute_floor", type=float, default=1e-5)
    parser.add_argument("--text_mask_min_patches", type=int, default=2)
    parser.add_argument("--text_mask_max_patches", type=int, default=64)
    parser.add_argument("--text_mask_dilate", type=int, default=0)
    parser.add_argument("--text_mask_example_count", type=int, default=36)

    # Population detectors.
    parser.add_argument("--mu1_cache_cos_threshold", type=float, default=0.50)
    parser.add_argument(
        "--scratch_residual_quantile",
        type=float,
        default=0.90,
        help=(
            "Among non-reg/non-text/non-mu1-cache patches, select the upper "
            "quantile of normalized residual energy outside span(mu1,mu2)."
        ),
    )
    parser.add_argument(
        "--scratch_residual_min_fraction",
        type=float,
        default=0.50,
        help=(
            "Absolute floor on normalized residual fraction outside span(mu1,mu2) "
            "for SCRATCH_RESIDUAL."
        ),
    )

    # Mean motif classifier.
    parser.add_argument("--motif_slope_threshold", type=float, default=0.10)
    parser.add_argument("--motif_curvature_threshold", type=float, default=0.12)
    parser.add_argument("--motif_insensitive_threshold", type=float, default=0.06)
    parser.add_argument("--motif_dominance_ratio", type=float, default=1.8)

    # Register detector. OpenAI CLIP-L empirically always has at least one
    # high-norm register in this regime, hence minimum=1 is deliberate here.
    parser.add_argument("--final_register_threshold", type=float, default=60.0)
    parser.add_argument("--final_register_min", type=int, default=1)
    parser.add_argument("--final_register_max", type=int, default=4)

    # Manual RN insertion; bridge-side modules are never executed.
    parser.add_argument("--rn_insert_block", type=int, default=RN_INSERT_BLOCK)

    # Direct model loaders.
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip_module", default="attnclip_mechinterp_sae")
    parser.add_argument("--xattn_module", default="attnclip_mechinterp_xattn")
    parser.add_argument("--pickle_module", default="oaiclip")
    parser.add_argument("--model_spec", default="ViT-L/14")
    parser.add_argument("--gmp_checkpoint", default=DEFAULT_GMP_CHECKPOINT)
    parser.add_argument("--xattn_checkpoint", default=DEFAULT_XATTN_CHECKPOINT)

    parser.add_argument("--seed", type=int, default=20260909)

    parser.add_argument(
        "--reuse_raw",
        action="store_true",
        help=(
            "If the new RN-factorial raw CSVs already exist in out_dir, skip GPU "
            "extraction and regenerate fits/stats/plots only. Old no-RN CSVs are "
            "never accepted as substitutes."
        ),
    )
    parser.add_argument(
        "--skip_legacy_population_compare",
        action="store_true",
        help=(
            "Do not compare freshly recomputed rn_off native_population_traffic.csv "
            "against baseline_dir/native_population_traffic.csv."
        ),
    )

    args = parser.parse_args()
    args.mu2_factors = parse_floats(args.mu2_factors)
    try:
        args.sheet_blocks = parse_blocks(args.sheet_blocks, n_blocks=24)
    except ValueError as exc:
        parser.error(str(exc))

    if len(args.mu2_factors) < 5:
        parser.error("--mu2_factors needs at least five values.")
    if 0.0 not in args.mu2_factors:
        parser.error("--mu2_factors must include 0.")
    if min(args.mu2_factors) >= 0 or max(args.mu2_factors) <= 0:
        parser.error("--mu2_factors must span negative and positive values.")
    if args.batch_triplets < 1:
        parser.error("--batch_triplets must be >= 1.")
    if args.preprocess_workers < 0:
        parser.error("--preprocess_workers must be >= 0.")
    if int(args.rn_insert_block) != RN_INSERT_BLOCK:
        parser.error(
            f"This paper experiment is fixed to pre-B{RN_INSERT_BLOCK}; "
            f"got --rn_insert_block={args.rn_insert_block}."
        )

    return args


# =============================================================================
# Synthetic regression
# =============================================================================

def synthetic_regression() -> None:
    # Fit signs.
    f = np.asarray([-4, -2, 0, 2, 4], np.float64)
    x = f / 4.0

    fc = fit_curve(f, 1.0 - .5*x)
    fr = fit_curve(f, .4 + .4*x)
    assert fc["slope"] < 0 and fr["slope"] > 0

    fu = fit_curve(f, .2 + .6*x*x)
    assert fu["curvature"] > 0

    fi = fit_curve(f, .8 - .5*x*x)
    assert fi["curvature"] < 0

    # FDR monotonicity sanity.
    q = bh_fdr([.001, .01, .2, .8])
    assert np.all(np.isfinite(q))
    assert q[0] <= q[1] <= q[2] <= q[3]


# =============================================================================
# Main
# =============================================================================


def _validate_factorial_raw(raw_curves: pd.DataFrame, traffic: pd.DataFrame) -> None:
    for name, df in (("raw_mu2_head_curves", raw_curves), ("native_population_traffic", traffic)):
        if "rn_mode" not in df.columns:
            raise RuntimeError(
                f"{name}.csv has no rn_mode column. This looks like an old no-RN run; "
                "--reuse_raw deliberately refuses to treat it as the new factorial."
            )
        modes = set(map(str, df["rn_mode"].dropna().unique()))
        missing = set(RN_MODE_ORDER) - modes
        if missing:
            raise RuntimeError(f"{name}.csv is missing RN states: {sorted(missing)}")


def analyze_main() -> None:
    args = analyze_parse_args()
    synthetic_regression()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", vars(args))

    script_snapshot = out / "probe_rta_head_population.py"
    try:
        if Path(__file__).resolve() != script_snapshot.resolve():
            shutil.copy2(Path(__file__), script_snapshot)
    except Exception as exc:
        print(f"[audit] could not snapshot script into output directory: {exc}")

    raw_curve_path = out / "raw_mu2_head_curves.csv"
    traffic_path = out / "native_population_traffic.csv"
    text_audit_path = out / "text_mask_audit.csv"
    reg_audit_path = out / "register_relocation_audit.csv"

    bundle: Optional[ModelBundle] = None
    text_examples: list[dict[str, Any]] = []

    if (
        args.reuse_raw
        and raw_curve_path.is_file()
        and traffic_path.is_file()
        and text_audit_path.is_file()
        and reg_audit_path.is_file()
    ):
        print("[reuse] loading NEW RN-factorial raw CSVs; no GPU extraction...")
        raw_curves = pd.read_csv(raw_curve_path)
        traffic = pd.read_csv(traffic_path)
        text_audit = pd.read_csv(text_audit_path)
        reg_audit = pd.read_csv(reg_audit_path)
        _validate_factorial_raw(raw_curves, traffic)
        basis_path = str(Path(args.oracle_root) / args.model / "mu_basis.npz")
        pair_count = int(raw_curves["sample_key"].nunique())
        rn_info = {
            "checkpoint": args.xattn_checkpoint,
            "insert_block": int(args.rn_insert_block),
            "reused_raw": True,
        }
    else:
        print("[dataset] loading RTA-100 triplets...")
        dsets = load_rta_triplet_subsets(
            args.dataset,
            args.split,
            CONDITION_ORDER,
        )
        pairs = build_paired_rows(
            dsets,
            limit=args.limit,
            seed=args.seed,
        )
        pair_count = len(pairs)
        print(f"[dataset] paired triplets: {pair_count}")

        print("[model] loading capture-enabled visual backbone directly...")
        bundle = load_bundle(
            args.model,
            args,
            out / args.model / "load_audit",
        )

        width = int(bundle.model.visual.positional_embedding.shape[1])
        mu1, mu2, basis_path = load_basis2(
            Path(args.oracle_root),
            args.model,
            width,
        )
        print(f"[mu basis] {basis_path}")

        rn_payload = load_rn_payload(args)
        if rn_payload.token.numel() != width:
            raise RuntimeError(
                f"RN width mismatch: token={rn_payload.token.numel()} visual_width={width}"
            )
        rn_info = {
            "checkpoint": rn_payload.checkpoint,
            "insert_block": int(rn_payload.insert_block),
            "token_dim": int(rn_payload.token.numel()),
            "token_norm": float(rn_payload.token.norm()),
            "reused_raw": False,
        }

        (
            raw_curves,
            traffic,
            text_audit,
            reg_audit,
            text_examples,
        ) = process_dataset(
            bundle,
            dsets,
            pairs,
            mu1,
            mu2,
            rn_payload,
            args,
        )

        _validate_factorial_raw(raw_curves, traffic)
        print("[save] fresh RN-factorial extraction tables...")
        raw_curves.to_csv(raw_curve_path, index=False)
        traffic.to_csv(traffic_path, index=False)
        text_audit.to_csv(text_audit_path, index=False)
        reg_audit.to_csv(reg_audit_path, index=False)

    # ------------------------------------------------------------------
    # Fits / motifs
    # ------------------------------------------------------------------
    print("[fit] per-image mu2 head curves...")
    per_image_fits = fit_per_image_curves(raw_curves)
    per_image_path = out / "per_image_mu2_head_fits.csv"
    per_image_fits.to_csv(per_image_path, index=False)

    print("[fit] condition-mean head response motifs...")
    mean_curves, condition_fits = fit_condition_mean_curves(raw_curves, args)
    mean_curves_path = out / "condition_mean_mu2_curves.csv"
    condition_fits_path = out / "condition_mean_mu2_head_fits.csv"
    mean_curves.to_csv(mean_curves_path, index=False)
    condition_fits.to_csv(condition_fits_path, index=False)

    transitions, exceptions, rn_transitions = motif_transitions(condition_fits)
    transitions_path = out / "condition_mean_motif_transitions.csv"
    exceptions_path = out / "push_pull_exception_heads.csv"
    rn_transitions_path = out / "condition_mean_rn_motif_transitions.csv"
    transitions.to_csv(transitions_path, index=False)
    exceptions.to_csv(exceptions_path, index=False)
    rn_transitions.to_csv(rn_transitions_path, index=False)

    # ------------------------------------------------------------------
    # Paired statistics: within-state and RN causal DID
    # ------------------------------------------------------------------
    print("[stats] attack-vs-clean mu2 response changes in each RN state...")
    mu2_stats = paired_significance_mu2(per_image_fits)
    mu2_stats_path = out / "paired_significance_mu2.csv"
    mu2_stats.to_csv(mu2_stats_path, index=False)

    print("[stats] RN difference-of-differences on mu2 response...")
    mu2_rn_stats = paired_significance_mu2_rn_effect(per_image_fits)
    mu2_rn_stats_path = out / "paired_significance_mu2_rn_effect.csv"
    mu2_rn_stats.to_csv(mu2_rn_stats_path, index=False)

    print("[stats] attack-vs-clean native population traffic in each RN state...")
    pop_stats = paired_significance_population(traffic, extended=args.extended_stats)
    pop_stats_path = out / "paired_significance_population.csv"
    pop_stats.to_csv(pop_stats_path, index=False)

    print("[stats] RN difference-of-differences on population traffic...")
    pop_rn_stats = paired_significance_population_rn_effect(traffic, extended=args.extended_stats)
    pop_rn_stats_path = out / "paired_significance_population_rn_effect.csv"
    pop_rn_stats.to_csv(pop_rn_stats_path, index=False)

    leaderboard, per_block_leaders = text_funnel_leaderboards(pop_stats)
    leaderboard_path = out / "text_funnel_leaderboard.csv"
    leaders_path = out / "per_block_text_funnel_leaders.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    per_block_leaders.to_csv(leaders_path, index=False)

    # ------------------------------------------------------------------
    # Old native_population_traffic.csv is parity-only, never a fallback.
    # ------------------------------------------------------------------
    legacy_parity = pd.DataFrame()
    legacy_parity_path = out / "legacy_native_population_traffic_parity.csv"
    if not args.skip_legacy_population_compare:
        legacy_source = Path(args.baseline_dir) / "native_population_traffic.csv"
        legacy_parity = compare_legacy_native_population(traffic, legacy_source)
        if len(legacy_parity):
            legacy_parity.to_csv(legacy_parity_path, index=False)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    if args.include_no_rn_compat_sheets:
        print(f"[plot] no-RN compatibility sheets for blocks={args.sheet_blocks}...")
        old_sheet_paths = mean_curve_contact_sheets_no_rn(
            mean_curves,
            out / "mu2_condition_sheets_no_rn",
            args.sheet_blocks,
        )
    else:
        old_sheet_paths = []

    print(f"[plot] NoRTA vs attack vs attack+RN 16-head sheets for blocks={args.sheet_blocks}...")
    triad_sheet_paths = mean_curve_rn_triad_sheets(
        mean_curves,
        out / "mu2_rn_triad_sheets",
        args.sheet_blocks,
    )

    print(f"[plot] full condition x RN factorial sheets for blocks={args.sheet_blocks}...")
    factorial_sheet_paths = mean_curve_rn_factorial_sheets(
        mean_curves,
        out / "mu2_rn_factorial_sheets",
        args.sheet_blocks,
    )

    if args.extended_plots:
        print("[plot] exhaustive attack-vs-clean head heatmaps for each RN state...")
        heatmap_paths = make_key_heatmaps(
            mu2_stats,
            pop_stats,
            out / "heatmaps",
        )
    else:
        heatmap_paths = []

    print("[plot] paper RN causal difference-of-differences heatmaps...")
    rn_heatmap_paths = make_rn_effect_heatmaps(
        mu2_rn_stats,
        pop_rn_stats,
        out / "heatmaps_rn_effect",
        extended=args.extended_plots,
    )

    print("[plot] B13 paper-head candidates and compact comparisons...")
    paper_paths, candidates, selected_heads = make_b13_paper_outputs(
        mean_curves,
        condition_fits,
        out / "paper",
    )

    print("[plot] appendix contact sheets for RN-induced primary-motif changes...")
    changed_motif_sheet_paths = changed_motif_contact_sheets(
        mean_curves,
        rn_transitions,
        out / "appendix_changed_motif_heads",
    )

    if text_examples:
        print("[plot] TEXT-mask audit sheets...")
        text_mask_paths = text_mask_sheets(text_examples, out / "text_masks")
    else:
        text_mask_paths = []

    # ------------------------------------------------------------------
    # Summary / audit
    # ------------------------------------------------------------------
    write_summary(
        out,
        args,
        condition_fits,
        transitions,
        rn_transitions,
        exceptions,
        per_block_leaders,
        mu2_rn_stats,
        pop_rn_stats,
        reg_audit,
        candidates,
        selected_heads,
        legacy_parity,
    )

    audit = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "paired_triplets": int(pair_count),
        "mu_basis": basis_path,
        "mu2_factors": [float(x) for x in args.mu2_factors],
        "rn": rn_info,
        "rn_modes": list(RN_MODE_ORDER),
        "primary_register_policy": (
            "RN-off NoRTA B23 register addresses frozen across condition and RN state"
        ),
        "secondary_register_policy": "condition- and RN-state-own B23 register addresses",
        "text_mask_policy": (
            "exact CLIP-preprocessed attack-vs-NoRTA patch pixel difference; "
            "same attack-derived spatial mask applied to clean and attacked views"
        ),
        "population_share_denominator": (
            "original spatial patch traffic only; appended RN token excluded"
        ),
        "efficiency": {
            "shared_native_prefix_through_block": int(args.rn_insert_block - 1),
            "rn_branch_at_pre_block": int(args.rn_insert_block),
            "slider_pre_rn_state_reused": True,
            "preprocess_workers": int(args.preprocess_workers),
            "prestate_storage": args.prestate_storage,
            "sheet_blocks": [int(x) for x in args.sheet_blocks],
            "extended_stats": bool(args.extended_stats),
            "extended_plots": bool(args.extended_plots),
            "pinned_input_if_cuda": True,
            "precision": "FP32",
        },
        "population_definitions": {
            "MU1_CACHE": f"non-reg/non-text cos(mu1)>={args.mu1_cache_cos_threshold}",
            "SCRATCH_RESIDUAL": (
                f"remaining residual-outside-rank2 energy >= max(q{args.scratch_residual_quantile}, "
                f"{args.scratch_residual_min_fraction})"
            ),
        },
        "legacy_population_parity_source": (
            None if args.skip_legacy_population_compare
            else str(Path(args.baseline_dir) / "native_population_traffic.csv")
        ),
        "legacy_population_parity_metrics": int(len(legacy_parity)),
        "selected_b13_paper_heads": {k: int(v) for k, v in selected_heads.items()},
        "scipy_available": bool(scipy_stats is not None),
        "raw_curve_rows": int(len(raw_curves)),
        "population_rows": int(len(traffic)),
        "no_rn_sheet_count": int(len(old_sheet_paths)),
        "triad_sheet_count": int(len(triad_sheet_paths)),
        "factorial_sheet_count": int(len(factorial_sheet_paths)),
        "heatmap_count": int(len(heatmap_paths)),
        "rn_effect_heatmap_count": int(len(rn_heatmap_paths)),
        "paper_output_count": int(len(paper_paths)),
        "changed_motif_sheet_output_count": int(len(changed_motif_sheet_paths)),
        "text_mask_sheet_count": int(len(text_mask_paths)),
    }
    audit_path = out / "audit.json"
    save_json(audit_path, audit)

    # ------------------------------------------------------------------
    # Zip everything useful.  Avoid recursive globbing so stale files from an
    # older run cannot silently enter the bundle.
    # ------------------------------------------------------------------
    include = [
        out / "config.json",
        audit_path,
        out / "SUMMARY.txt",
        script_snapshot,
        raw_curve_path,
        traffic_path,
        text_audit_path,
        reg_audit_path,
        per_image_path,
        mean_curves_path,
        condition_fits_path,
        transitions_path,
        rn_transitions_path,
        exceptions_path,
        mu2_stats_path,
        mu2_rn_stats_path,
        pop_stats_path,
        pop_rn_stats_path,
        leaderboard_path,
        leaders_path,
    ]
    if legacy_parity_path.is_file():
        include.append(legacy_parity_path)
    include += old_sheet_paths
    include += triad_sheet_paths
    include += factorial_sheet_paths
    include += heatmap_paths
    include += rn_heatmap_paths
    include += paper_paths
    include += changed_motif_sheet_paths
    include += text_mask_paths

    # De-duplicate paths while preserving order.
    seen = set()
    include_unique = []
    for q in include:
        q = Path(q)
        key = str(q.resolve()) if q.exists() else str(q)
        if key not in seen:
            seen.add(key)
            include_unique.append(q)

    zpath = out / "compact_summary_workspace_rta_head_population.zip"
    if zpath.exists():
        zpath.unlink()
    print("[zip] packaging result bundle...")
    with zipfile.ZipFile(
        zpath,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=7,
    ) as archive:
        for q in include_unique:
            if q.is_file():
                archive.write(q, arcname=q.relative_to(out).as_posix())

    print("[compact summary]", zpath)

    if bundle is not None:
        del bundle
    clear_cuda()


# CONTACT SHEETS
# -*- coding: utf-8 -*-


from matplotlib.lines import Line2D


MOTIF_DISPLAY = {
    "broadcast_gain": "broadcast gain",
    "common_mode_attenuation": "common-mode atten.",
    "insensitive": "insensitive",
    "mixed": "mixed",
    "push_pull_exchange": "push-pull exchange",
    "read_gain": "read gain",
    "read_optimum": "read optimum",
    "source_null_curvature": "source-null curvature",
}


def format_motif_name(name: str) -> str:
    return MOTIF_DISPLAY.get(str(name), str(name).replace("_", " "))


def load_required_csvs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    transitions_path = input_dir / "condition_mean_rn_motif_transitions.csv"
    curves_path = input_dir / "condition_mean_mu2_curves.csv"

    if not transitions_path.exists():
        raise FileNotFoundError(f"Missing required file: {transitions_path}")
    if not curves_path.exists():
        raise FileNotFoundError(f"Missing required file: {curves_path}")

    transitions_df = pd.read_csv(transitions_path)
    curves_df = pd.read_csv(curves_path)

    required_transition_cols = {
        "comparison",
        "block",
        "head",
        "motif_off",
        "motif_on",
        "motif_changed",
    }
    required_curve_cols = {
        "rn_mode",
        "condition",
        "block",
        "head",
        "factor_mu2",
        "frozen_C2R_write",
        "frozen_R2C_write",
    }

    missing_transition_cols = required_transition_cols.difference(transitions_df.columns)
    missing_curve_cols = required_curve_cols.difference(curves_df.columns)

    if missing_transition_cols:
        raise RuntimeError(
            "condition_mean_rn_motif_transitions.csv is missing columns: "
            f"{sorted(missing_transition_cols)}"
        )
    if missing_curve_cols:
        raise RuntimeError(
            "condition_mean_mu2_curves.csv is missing columns: "
            f"{sorted(missing_curve_cols)}"
        )

    return transitions_df, curves_df


def build_changed_summary(transitions_df: pd.DataFrame) -> pd.DataFrame:
    changed_df = transitions_df.loc[transitions_df["motif_changed"].astype(bool)].copy()

    rta_df = (
        changed_df.loc[changed_df["comparison"] == "RTA", ["block", "head", "motif_off", "motif_on"]]
        .rename(columns={"motif_off": "RTA_motif_off", "motif_on": "RTA_motif_on"})
        .copy()
    )
    rta_df["in_RTA"] = True

    synth_df = (
        changed_df.loc[changed_df["comparison"] == "SynthRTA", ["block", "head", "motif_off", "motif_on"]]
        .rename(columns={"motif_off": "SynthRTA_motif_off", "motif_on": "SynthRTA_motif_on"})
        .copy()
    )
    synth_df["in_SynthRTA"] = True

    summary_df = pd.merge(
        rta_df,
        synth_df,
        on=["block", "head"],
        how="outer",
    )

    summary_df["in_RTA"] = summary_df["in_RTA"].fillna(False)
    summary_df["in_SynthRTA"] = summary_df["in_SynthRTA"].fillna(False)

    def classify_row(row: pd.Series) -> str:
        if row["in_RTA"] and row["in_SynthRTA"]:
            return "shared"
        if row["in_RTA"]:
            return "RTA_only"
        if row["in_SynthRTA"]:
            return "SynthRTA_only"
        return "none"

    summary_df["membership"] = summary_df.apply(classify_row, axis=1)
    summary_df = summary_df.sort_values(["block", "head"]).reset_index(drop=True)
    return summary_df


def decide_sheet_mode(summary_df: pd.DataFrame) -> str:
    rta_set = set(
        map(tuple, summary_df.loc[summary_df["in_RTA"], ["block", "head"]].itertuples(index=False, name=None))
    )
    synth_set = set(
        map(tuple, summary_df.loc[summary_df["in_SynthRTA"], ["block", "head"]].itertuples(index=False, name=None))
    )

    if synth_set.issubset(rta_set) or rta_set.issubset(synth_set):
        return "nested"
    return "separate"


def select_changed_heads(
    transitions_df: pd.DataFrame,
    comparison: str,
) -> pd.DataFrame:
    selected_df = transitions_df.loc[
        (transitions_df["comparison"] == comparison)
        & (transitions_df["motif_changed"].astype(bool))
    ].copy()

    selected_df = selected_df.sort_values(["block", "head"]).reset_index(drop=True)
    return selected_df


def get_panel_curves(
    curves_df: pd.DataFrame,
    comparison: str,
    block: int,
    head: int,
) -> pd.DataFrame:
    panel_df = curves_df.loc[
        (curves_df["block"] == block)
        & (curves_df["head"] == head)
        & (
            (
                (curves_df["condition"] == "NoRTA")
                & (curves_df["rn_mode"] == "rn_off")
            )
            | (
                (curves_df["condition"] == comparison)
                & (curves_df["rn_mode"].isin(["rn_off", "rn_on"]))
            )
        )
    ].copy()

    panel_df = panel_df.sort_values(["condition", "rn_mode", "factor_mu2"]).reset_index(drop=True)
    return panel_df


def get_panel_peak(panel_df: pd.DataFrame) -> float:
    peak = panel_df[["frozen_C2R_write", "frozen_R2C_write"]].abs().to_numpy().max()
    if not math.isfinite(float(peak)) or float(peak) <= 0.0:
        return 1.0
    return float(peak)


def plot_single_head_panel(
    ax: plt.Axes,
    curves_df: pd.DataFrame,
    comparison: str,
    block: int,
    head: int,
    motif_off: str,
    motif_on: str,
) -> None:
    panel_df = get_panel_curves(curves_df, comparison, block, head)

    if panel_df.empty:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    raw_peak = get_panel_peak(panel_df)

    curve_specs = [
        ("NoRTA", "rn_off", "NoRTA", "tab:blue"),
        (comparison, "rn_off", comparison, "tab:orange"),
        (comparison, "rn_on", f"{comparison}+RN", "tab:green"),
    ]

    min_y = 0.0
    max_y = 1.0

    for condition, rn_mode, label, color in curve_specs:
        subset_df = panel_df.loc[
            (panel_df["condition"] == condition)
            & (panel_df["rn_mode"] == rn_mode)
        ].sort_values("factor_mu2")

        if subset_df.empty:
            continue

        x = subset_df["factor_mu2"].to_numpy()
        y_c2r = subset_df["frozen_C2R_write"].to_numpy() / raw_peak
        y_r2c = subset_df["frozen_R2C_write"].to_numpy() / raw_peak

        min_y = min(min_y, float(y_c2r.min()), float(y_r2c.min()))
        max_y = max(max_y, float(y_c2r.max()), float(y_r2c.max()))

        ax.plot(
            x,
            y_c2r,
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=4.5,
            color=color,
        )
        ax.plot(
            x,
            y_r2c,
            marker="o",
            linestyle="--",
            linewidth=1.8,
            markersize=4.5,
            color=color,
        )

    ax.axvline(0.0, color="tab:blue", linewidth=1.0, alpha=0.35)
    ax.grid(True, alpha=0.25)

    ax.set_title(
        f"B{block:02d} H{head:02d} — {format_motif_name(motif_off)} → {format_motif_name(motif_on)}",
        fontsize=10,
    )
    ax.text(
        0.02,
        0.04,
        f"raw peak={raw_peak:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
    )

    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(min(-0.05, min_y - 0.05), max(1.05, max_y + 0.05))
    ax.set_xlabel(r"CLS displacement $\alpha \mu_2$")
    ax.set_ylabel("normalized exact OV/write response")


def make_sheet_legend(comparison: str) -> list[Line2D]:
    return [
        Line2D([0], [0], color="tab:blue", marker="o", linestyle="-", label="NoRTA C→R"),
        Line2D([0], [0], color="tab:blue", marker="o", linestyle="--", label="NoRTA R→C"),
        Line2D([0], [0], color="tab:orange", marker="o", linestyle="-", label=f"{comparison} C→R"),
        Line2D([0], [0], color="tab:orange", marker="o", linestyle="--", label=f"{comparison} R→C"),
        Line2D([0], [0], color="tab:green", marker="o", linestyle="-", label=f"{comparison}+RN C→R"),
        Line2D([0], [0], color="tab:green", marker="o", linestyle="--", label=f"{comparison}+RN R→C"),
    ]


def save_changed_motif_sheet(
    curves_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    comparison: str,
    output_path: Path,
) -> None:
    n_panels = len(selected_df)
    if n_panels == 0:
        print(f"[skip] no changed-motif heads for {comparison}")
        return

    ncols = 3
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.5 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    axes_flat = list(axes.flat)

    for ax, row in zip(axes_flat, selected_df.itertuples(index=False)):
        plot_single_head_panel(
            ax=ax,
            curves_df=curves_df,
            comparison=comparison,
            block=int(row.block),
            head=int(row.head),
            motif_off=str(row.motif_off),
            motif_on=str(row.motif_on),
        )

    for ax in axes_flat[n_panels:]:
        ax.set_axis_off()

    legend_handles = make_sheet_legend(comparison)

    fig.suptitle(
        f"Heads whose primary motif changes under RN — {comparison}\n"
        r"solid = CLS$\rightarrow$frozen-REG; dashed = frozen-REG$\rightarrow$CLS",
        fontsize=18,
        y=0.995,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=11,
    )

    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.90])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    print(f"[saved] {output_path}")


def save_summary_files(
    summary_df: pd.DataFrame,
    sheet_mode: str,
    output_dir: Path,
) -> None:
    summary_csv_path = output_dir / "changed_motif_heads_summary.csv"
    summary_txt_path = output_dir / "changed_motif_heads_summary.txt"

    summary_df.to_csv(summary_csv_path, index=False)

    rta_count = int(summary_df["in_RTA"].sum())
    synth_count = int(summary_df["in_SynthRTA"].sum())
    shared_count = int((summary_df["membership"] == "shared").sum())
    rta_only_count = int((summary_df["membership"] == "RTA_only").sum())
    synth_only_count = int((summary_df["membership"] == "SynthRTA_only").sum())

    lines = []
    lines.append("Changed primary operational motif under RN\n")
    lines.append(f"sheet_mode: {sheet_mode}\n")
    lines.append(f"RTA changed heads: {rta_count}\n")
    lines.append(f"SynthRTA changed heads: {synth_count}\n")
    lines.append(f"shared heads: {shared_count}\n")
    lines.append(f"RTA-only heads: {rta_only_count}\n")
    lines.append(f"SynthRTA-only heads: {synth_only_count}\n")

    lines.append("\nShared heads:\n")
    shared_df = summary_df.loc[summary_df["membership"] == "shared"]
    if shared_df.empty:
        lines.append("  (none)\n")
    else:
        for row in shared_df.itertuples(index=False):
            lines.append(
                f"  B{int(row.block):02d} H{int(row.head):02d} | "
                f"RTA: {row.RTA_motif_off} -> {row.RTA_motif_on} | "
                f"SynthRTA: {row.SynthRTA_motif_off} -> {row.SynthRTA_motif_on}\n"
            )

    lines.append("\nRTA-only heads:\n")
    rta_only_df = summary_df.loc[summary_df["membership"] == "RTA_only"]
    if rta_only_df.empty:
        lines.append("  (none)\n")
    else:
        for row in rta_only_df.itertuples(index=False):
            lines.append(
                f"  B{int(row.block):02d} H{int(row.head):02d} | "
                f"{row.RTA_motif_off} -> {row.RTA_motif_on}\n"
            )

    lines.append("\nSynthRTA-only heads:\n")
    synth_only_df = summary_df.loc[summary_df["membership"] == "SynthRTA_only"]
    if synth_only_df.empty:
        lines.append("  (none)\n")
    else:
        for row in synth_only_df.itertuples(index=False):
            lines.append(
                f"  B{int(row.block):02d} H{int(row.head):02d} | "
                f"{row.SynthRTA_motif_off} -> {row.SynthRTA_motif_on}\n"
            )

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[saved] {summary_csv_path}")
    print(f"[saved] {summary_txt_path}")


def contact_sheets_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save contact sheets for heads with RN-induced motif changes."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing condition_mean_rn_motif_transitions.csv and condition_mean_mu2_curves.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save contact sheets and summaries. Defaults to input_dir / 'appendix_contact_sheets'",
    )
    return parser.parse_args()


def contact_sheets_main() -> None:
    args = contact_sheets_parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir / "appendix_contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)

    transitions_df, curves_df = load_required_csvs(input_dir)
    summary_df = build_changed_summary(transitions_df)
    sheet_mode = decide_sheet_mode(summary_df)

    print(f"[info] sheet_mode = {sheet_mode}")
    print(f"[info] RTA changed heads = {int(summary_df['in_RTA'].sum())}")
    print(f"[info] SynthRTA changed heads = {int(summary_df['in_SynthRTA'].sum())}")

    save_summary_files(summary_df, sheet_mode, output_dir)

    # NEW: current data are not nested, so separate sheets are the correct default.
    rta_changed_df = select_changed_heads(transitions_df, "RTA")
    synth_changed_df = select_changed_heads(transitions_df, "SynthRTA")

    save_changed_motif_sheet(
        curves_df=curves_df,
        selected_df=rta_changed_df,
        comparison="RTA",
        output_path=output_dir / "changed_motif_heads_RTA.png",
    )
    save_changed_motif_sheet(
        curves_df=curves_df,
        selected_df=synth_changed_df,
        comparison="SynthRTA",
        output_path=output_dir / "changed_motif_heads_SynthRTA.png",
    )

    print("[done]")


def main(argv=None):
    """Dispatch a workflow; each subcommand retains its original CLI options."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {'analyze': analyze_main, 'contact_sheets': contact_sheets_main}
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

