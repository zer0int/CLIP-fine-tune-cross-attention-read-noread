#!/usr/bin/env python3
r"""Single-image CLS/register mechanics. Commands: example (native and counterfactual atlas), motifs (all-head response catalogue).
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
import probe_tools_backbone as _backbone_tools
from probe_tools_analysis import (normalize_probs, r2_score, safe_corr)

# EXAMPLE
import argparse
import gc
import importlib
import importlib.util
import json
import math
import random
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm.auto import tqdm


MODEL_ORDER = ("pretrained", "gmp", "finetune_stripped")

DEFAULT_IMAGE = "image_sets/demoset/bottle_shower.png"
DEFAULT_ORACLE_ROOT = r"cls_gipu_exchange_no_rn"
EXAMPLE_DEFAULT_OUT = r"bottle_shower_cls_reg_atlas"
DEFAULT_XATTN_CHECKPOINT = "REPLACE_WITH_CHECKPOINT.pt"

EPS = 1e-12


# =============================================================================
# Generic helpers
# =============================================================================

def parse_ints(text: str) -> list[int]:
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


def example_parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def resolve_local(path_text: str) -> Path:
    p = Path(path_text)
    if p.exists():
        return p
    alt = Path(__file__).resolve().parent / path_text
    return alt if alt.exists() else p


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def example_savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def normalized_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, np.float64)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= EPS:
        return 1.0
    p = p / s
    nz = p[p > 0]
    h = -float(np.sum(nz * np.log(nz)))
    return h / max(math.log(len(p)), EPS)


def side_from_patch_count(patches: int) -> int:
    side = int(round(math.sqrt(patches)))
    if side * side != patches:
        raise ValueError(f"Patch count {patches} is not square.")
    return side


def register_indices(mask_p: np.ndarray) -> list[int]:
    return np.flatnonzero(np.asarray(mask_p, bool)).astype(int).tolist()


def select_register_mask(
    norms_p: torch.Tensor,
    threshold: float,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    p = norms_p.numel()
    out = torch.zeros(p, dtype=torch.bool, device=norms_p.device)
    idx = torch.nonzero(norms_p >= threshold, as_tuple=False).flatten()
    if maximum > 0 and idx.numel() > maximum:
        idx = idx[torch.topk(norms_p[idx], k=maximum).indices]
    if idx.numel() < minimum:
        idx = torch.topk(norms_p, k=min(minimum, p)).indices
    out[idx] = True
    return out


def normalize_qkv(base, x0, batch: int, heads: int, tokens: int) -> torch.Tensor:
    return base.normalize_qkv_shape(x0, batch, heads, tokens).float()


# =============================================================================
# Basis + optional RN
# =============================================================================

def load_mu_basis(oracle_root: Path, model_name: str, width: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    path = oracle_root / model_name / "mu_basis.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing mu basis: {path}\n"
            "Pass --oracle_root pointing at the completed no-RN CLS/GIPU oracle."
        )
    data = np.load(path, allow_pickle=True)
    if "mu_basis" in data:
        basis = np.asarray(data["mu_basis"], np.float32)
    elif "basis4" in data:
        basis = np.asarray(data["basis4"], np.float32)
    else:
        raise KeyError(f"No mu_basis/basis4 in {path}: {data.files}")
    if basis.ndim != 2 or basis.shape[0] < 2 or basis.shape[1] != width:
        raise RuntimeError(f"Unexpected basis shape {basis.shape}; width={width}")
    mu1 = torch.from_numpy(basis[0]).float()
    mu2 = torch.from_numpy(basis[1]).float()
    mu1 = mu1 / mu1.norm().clamp_min(EPS)
    mu2 = mu2 - torch.dot(mu2, mu1) * mu1
    mu2 = mu2 / mu2.norm().clamp_min(EPS)
    return mu1, mu2, str(path)


def load_trained_rn_token(base, args) -> torch.Tensor:
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
    return state[key].detach().float().cpu().clone()


# =============================================================================
# One-image capture
# =============================================================================

@dataclass
class BlockCapture:
    block: int
    pre_td: torch.Tensor       # CPU [T,D], pre-LN1 residual for this block
    probs_htt: torch.Tensor    # CPU [H,T,T]
    values_htd: torch.Tensor   # CPU [H,T,dh]
    q_htd: torch.Tensor        # CPU [H,T,dh]
    k_htd: torch.Tensor        # CPU [H,T,dh]


@dataclass
class NativeRun:
    captures: dict[int, BlockCapture]
    final_x_td: torch.Tensor
    patch_count: int
    side: int
    fixed_reg_mask: np.ndarray
    fixed_reg_indices: list[int]


@torch.no_grad()
def capture_native_run(base, bundle, image_tensor_chw: torch.Tensor, rn_token_cpu: Optional[torch.Tensor], args) -> NativeRun:
    visual = bundle.model.visual
    images = image_tensor_chw[None].to(bundle.device, dtype=bundle.model.dtype)
    x = visual._prepare_tokens(images)
    patch_count = int(visual.positional_embedding.shape[0] - 1)
    side = side_from_patch_count(patch_count)
    captures: dict[int, BlockCapture] = {}

    for block_index, block in enumerate(visual.transformer.resblocks):
        if args.rn and block_index == args.rn_insert_block:
            if rn_token_cpu is None:
                raise RuntimeError("--rn requested but RN token was not loaded.")
            rn = rn_token_cpu.to(bundle.device, dtype=x.dtype).reshape(1, 1, -1)
            rn = rn.expand(1, x.shape[1], -1)
            x = torch.cat([x, rn], dim=0)

        pre = x
        ln1 = block.ln_1(x)
        attn_out, probs0 = block.attention(ln1, need_weights=True, capture=True)
        heads = int(block.attn.num_heads)
        tokens = x.shape[0]
        probs = normalize_probs(base, probs0, 1, heads, tokens)[0]
        values = normalize_qkv(base, block.attn.last_v, 1, heads, tokens)[0]
        q = normalize_qkv(base, block.attn.last_q, 1, heads, tokens)[0]
        k = normalize_qkv(base, block.attn.last_k, 1, heads, tokens)[0]

        captures[block_index] = BlockCapture(
            block=block_index,
            pre_td=pre[:, 0].detach().float().cpu(),
            probs_htt=probs.detach().float().cpu(),
            values_htd=values.detach().float().cpu(),
            q_htd=q.detach().float().cpu(),
            k_htd=k.detach().float().cpu(),
        )

        x_attn = x + attn_out
        ln2 = block.ln_2(x_attn)
        x = x_attn + block.mlp.c_proj(block.mlp.gelu(block.mlp.c_fc(ln2)))

        for attr in ("last_logits", "last_probs", "last_v", "last_z", "last_q", "last_k", "last_xin"):
            if hasattr(block.attn, attr):
                setattr(block.attn, attr, None)

    final_x = x[:, 0].detach().float().cpu()
    final_patch = final_x[1:1 + patch_count]
    final_norm = final_patch.norm(dim=-1)
    reg = select_register_mask(
        final_norm,
        threshold=args.final_register_threshold,
        minimum=args.final_register_min,
        maximum=args.final_register_max,
    ).cpu().numpy().astype(bool)

    return NativeRun(
        captures=captures,
        final_x_td=final_x,
        patch_count=patch_count,
        side=side,
        fixed_reg_mask=reg,
        fixed_reg_indices=register_indices(reg),
    )


# =============================================================================
# Exact head mechanics
# =============================================================================

@dataclass
class HeadMaps:
    cls_query_patch_attn: np.ndarray       # [P]
    cls_source_patch_write_norm: np.ndarray
    cls_source_patch_mu1: np.ndarray
    cls_to_reg_write_norm: np.ndarray      # [P], NaN outside REG
    reg_to_cls_write_norm: np.ndarray      # [P], NaN outside REG
    qk_cls_to_reg_signed: np.ndarray       # [P], NaN outside REG

    cls_to_reg_total: float
    reg_to_cls_total: float
    cls_to_reg_attn: float
    reg_to_cls_attn: float
    exchange_balance: float
    qk_reg_max_abs: float
    qk_reg_signed_at_max: float


def per_head_projected_value_norm_and_mu1(
    values_htd: torch.Tensor,
    out_proj_weight_dd: torch.Tensor,
    mu1_d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        projected value vector norm [H,T]
        projected value signed mu1 coefficient [H,T]

    Each token/head value is mapped through that head's slice of W_O.
    """
    values = values_htd.float()
    wout = out_proj_weight_dd.float()
    mu1 = mu1_d.float()
    heads, tokens, dh = values.shape
    width = wout.shape[0]
    if width != heads * dh:
        raise RuntimeError(f"W_O width {width}, heads*dh={heads*dh}")

    norms = []
    mu1s = []
    for h in range(heads):
        wh = wout[:, h * dh:(h + 1) * dh]         # [D,dh]
        projected = values[h] @ wh.T              # [T,D]
        norms.append(projected.norm(dim=-1))
        mu1s.append(projected @ mu1)
    return torch.stack(norms, 0), torch.stack(mu1s, 0)


def build_head_maps(
    capture: BlockCapture,
    block,
    head: int,
    patch_count: int,
    reg_mask_p: np.ndarray,
    mu1_d: torch.Tensor,
) -> HeadMaps:
    probs = capture.probs_htt.float()
    values = capture.values_htd.float()
    q = capture.q_htd.float()
    k = capture.k_htd.float()
    heads = probs.shape[0]
    if not (0 <= head < heads):
        raise ValueError(head)

    pv_norm, pv_mu1 = per_head_projected_value_norm_and_mu1(
        values,
        block.attn.out_proj.weight.detach().cpu(),
        mu1_d.cpu(),
    )

    reg = torch.from_numpy(np.asarray(reg_mask_p, bool))
    spatial = slice(1, 1 + patch_count)

    # CLS query reading spatial patches.
    cls_q_patch = probs[head, 0, spatial]

    # CLS as a V source writing into each spatial query.
    a_from_cls = probs[head, spatial, 0]
    cls_source_norm = a_from_cls * pv_norm[head, 0]
    cls_source_mu1 = a_from_cls * pv_mu1[head, 0]

    # CLS -> fixed register queries.
    c2r = cls_source_norm.clone()
    c2r[~reg] = float("nan")

    # Fixed register V sources -> CLS query.
    r2c = probs[head, 0, spatial] * pv_norm[head, spatial]
    r2c = r2c.clone()
    r2c[~reg] = float("nan")

    c2r_total = float(torch.nansum(c2r))
    r2c_total = float(torch.nansum(r2c))
    c2r_attn = float(a_from_cls[reg].sum()) if bool(reg.any()) else 0.0
    r2c_attn = float(cls_q_patch[reg].sum()) if bool(reg.any()) else 0.0
    balance = (r2c_total - c2r_total) / max(r2c_total + c2r_total, EPS)

    # Raw scaled QK logits for CLS query -> register keys.
    dh = q.shape[-1]
    logits = (q[head, 0][None, :] * k[head, spatial]).sum(dim=-1) / math.sqrt(dh)
    qk = logits.clone()
    qk[~reg] = float("nan")
    if bool(reg.any()):
        vals = logits[reg]
        j = int(torch.argmax(vals.abs()))
        qk_abs = float(vals[j].abs())
        qk_signed = float(vals[j])
    else:
        qk_abs = 0.0
        qk_signed = 0.0

    return HeadMaps(
        cls_query_patch_attn=cls_q_patch.numpy(),
        cls_source_patch_write_norm=cls_source_norm.numpy(),
        cls_source_patch_mu1=cls_source_mu1.numpy(),
        cls_to_reg_write_norm=c2r.numpy(),
        reg_to_cls_write_norm=r2c.numpy(),
        qk_cls_to_reg_signed=qk.numpy(),
        cls_to_reg_total=c2r_total,
        reg_to_cls_total=r2c_total,
        cls_to_reg_attn=c2r_attn,
        reg_to_cls_attn=r2c_attn,
        exchange_balance=balance,
        qk_reg_max_abs=qk_abs,
        qk_reg_signed_at_max=qk_signed,
    )


def native_metrics_frame(
    run: NativeRun,
    bundle,
    mu1: torch.Tensor,
) -> tuple[pd.DataFrame, dict[tuple[int, int], HeadMaps]]:
    rows = []
    maps: dict[tuple[int, int], HeadMaps] = {}
    visual = bundle.model.visual

    for block_index, capture in run.captures.items():
        block = visual.transformer.resblocks[block_index]
        heads = capture.probs_htt.shape[0]
        for h in range(heads):
            hm = build_head_maps(
                capture,
                block,
                h,
                run.patch_count,
                run.fixed_reg_mask,
                mu1,
            )
            maps[(block_index, h)] = hm

            p = np.asarray(hm.cls_query_patch_attn, np.float64)
            patch_mass = max(float(p.sum()), EPS)
            p_norm = p / patch_mass

            # "Position scouting" should prefer heads that split the spatial
            # field into two broad, internally compact levels (row/column/half
            # scanners), rather than a tiny-mass numerical contrast or a
            # one-patch spike.  Sort into an exactly balanced lower/upper half:
            # a binary 50/50 scanner has large between-half separation and
            # nearly zero within-half variance.
            ordered = np.sort(p_norm)
            half = len(ordered) // 2
            low = ordered[:half]
            high = ordered[-half:]
            low_mean = float(low.mean())
            high_mean = float(high.mean())
            separation = max(high_mean - low_mean, 0.0)
            within = 0.5 * (float(low.std()) + float(high.std()))
            separation_rel = separation / max(float(p_norm.mean()), EPS)
            compactness = separation / max(separation + 2.0 * within, EPS)

            # sqrt(patch_mass) suppresses heads that look contrasty only after
            # conditioning on a vanishing amount of CLS->patch attention,
            # without requiring the scout to devote all CLS mass to patches.
            scout_score = math.sqrt(patch_mass) * separation_rel * compactness

            mu = np.asarray(hm.cls_source_patch_mu1, np.float64)
            puff_signed_mean = float(np.mean(mu))
            puff_abs_mean = float(np.mean(np.abs(mu)))
            puff_coherence = abs(puff_signed_mean) / max(puff_abs_mean, EPS)
            puff_score = abs(puff_signed_mean)

            bidir = math.sqrt(max(hm.cls_to_reg_total, 0.0) * max(hm.reg_to_cls_total, 0.0))
            exchange_sum = hm.cls_to_reg_total + hm.reg_to_cls_total

            rows.append({
                "block": block_index,
                "head": h,
                "scout_balanced_contrast": scout_score,
                "scout_patch_mass": patch_mass,
                "scout_half_separation_rel": separation_rel,
                "scout_half_compactness": compactness,
                "scout_entropy_deficit": 1.0 - normalized_entropy(p_norm),
                "puff_signed_mu1_mean": puff_signed_mean,
                "puff_abs_mu1_mean": puff_abs_mean,
                "puff_sign_coherence": puff_coherence,
                "puff_score": puff_score,
                "cls_to_reg_write_total": hm.cls_to_reg_total,
                "reg_to_cls_write_total": hm.reg_to_cls_total,
                "cls_to_reg_attn": hm.cls_to_reg_attn,
                "reg_to_cls_attn": hm.reg_to_cls_attn,
                "exchange_write_sum": exchange_sum,
                "exchange_write_geomean": bidir,
                "exchange_balance": hm.exchange_balance,
                "qk_reg_max_abs": hm.qk_reg_max_abs,
                "qk_reg_signed_at_max": hm.qk_reg_signed_at_max,
            })
    return pd.DataFrame(rows), maps


# =============================================================================
# Role-plane state
# =============================================================================

def role_plane_frame(run: NativeRun, mu1: torch.Tensor, mu2: torch.Tensor) -> pd.DataFrame:
    rows = []
    reg = torch.from_numpy(run.fixed_reg_mask)
    for block_index, capture in run.captures.items():
        pre = capture.pre_td.float()
        cls = pre[0]
        patches = pre[1:1 + run.patch_count]
        reg_mean = patches[reg].mean(dim=0)

        def proj(x):
            c1 = float(x @ mu1)
            c2 = float(x @ mu2)
            norm = float(x.norm())
            return c1, c2, c1 / max(norm, EPS), c2 / max(norm, EPS), math.degrees(math.atan2(c2, c1))

        c1, c2, cc1, cc2, ca = proj(cls)
        r1, r2, rc1, rc2, ra = proj(reg_mean)
        rows.append({
            "block": block_index,
            "cls_mu1_coef": c1,
            "cls_mu2_coef": c2,
            "cls_mu1_cos": cc1,
            "cls_mu2_cos": cc2,
            "cls_plane_angle_deg": ca,
            "reg_mu1_coef": r1,
            "reg_mu2_coef": r2,
            "reg_mu1_cos": rc1,
            "reg_mu2_cos": rc2,
            "reg_plane_angle_deg": ra,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Phase selection
# =============================================================================

# v3 changes: scouting capped to B0--B2; any panel that displays REG→CLS now also shows a separate CLS box.

PHASES = (
    ("scouting", (0, 2), "scout_balanced_contrast"),  # modified: keep scouting in the genuine early pre-puff regime
    ("mu1_puff", (6, 10), "puff_score"),
    ("reg_formation", (11, 13), "exchange_write_sum"),
    ("workspace", (14, 19), "exchange_write_geomean"),
    ("exchange", (20, 21), "exchange_write_geomean"),
    ("late_gate", (22, 22), "qk_reg_max_abs"),
)


def choose_phases(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, (lo, hi), score_col in PHASES:
        z = metrics[(metrics["block"] >= lo) & (metrics["block"] <= hi)].copy()
        if z.empty:
            continue
        z = z.sort_values(score_col, ascending=False)
        r = z.iloc[0]
        row = {
            "phase": phase,
            "range": f"B{lo}-B{hi}",
            "score_metric": score_col,
            "block": int(r["block"]),
            "head": int(r["head"]),
            "score": float(r[score_col]),
        }
        for col in (
            "cls_to_reg_write_total",
            "reg_to_cls_write_total",
            "exchange_balance",
            "puff_signed_mu1_mean",
            "qk_reg_signed_at_max",
        ):
            row[col] = float(r[col])
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Counterfactual block capture + mu2 slider
# =============================================================================

@torch.no_grad()
def evaluate_counterfactual_block(
    base,
    bundle,
    capture: BlockCapture,
    block_index: int,
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    factor_mu2: float,
    patch_count: int,
    reg_mask_p: np.ndarray,
) -> dict[int, HeadMaps]:
    block = bundle.model.visual.transformer.resblocks[block_index]
    x = capture.pre_td[:, None, :].to(bundle.device, dtype=bundle.model.dtype).clone()
    x[0, 0] = x[0, 0] + float(factor_mu2) * mu2.to(bundle.device, dtype=x.dtype)

    ln1 = block.ln_1(x)
    _out, probs0 = block.attention(ln1, need_weights=True, capture=True)
    heads = int(block.attn.num_heads)
    tokens = x.shape[0]
    probs = normalize_probs(base, probs0, 1, heads, tokens)[0]
    values = normalize_qkv(base, block.attn.last_v, 1, heads, tokens)[0]
    q = normalize_qkv(base, block.attn.last_q, 1, heads, tokens)[0]
    k = normalize_qkv(base, block.attn.last_k, 1, heads, tokens)[0]

    cf = BlockCapture(
        block=block_index,
        pre_td=x[:, 0].detach().float().cpu(),
        probs_htt=probs.detach().float().cpu(),
        values_htd=values.detach().float().cpu(),
        q_htd=q.detach().float().cpu(),
        k_htd=k.detach().float().cpu(),
    )

    out = {}
    for h in range(heads):
        out[h] = build_head_maps(
            cf,
            block,
            h,
            patch_count,
            reg_mask_p,
            mu1,
        )

    for attr in ("last_logits", "last_probs", "last_v", "last_z", "last_q", "last_k", "last_xin"):
        if hasattr(block.attn, attr):
            setattr(block.attn, attr, None)
    return out


def scan_mu2_transduction_heads(
    base,
    bundle,
    run: NativeRun,
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    block_index: int,
    eps: float,
) -> tuple[pd.DataFrame, list[int]]:
    capture = run.captures[block_index]
    minus = evaluate_counterfactual_block(
        base, bundle, capture, block_index, mu1, mu2, -eps,
        run.patch_count, run.fixed_reg_mask,
    )
    plus = evaluate_counterfactual_block(
        base, bundle, capture, block_index, mu1, mu2, +eps,
        run.patch_count, run.fixed_reg_mask,
    )

    rows = []
    for h in sorted(minus):
        m = float(np.mean(minus[h].cls_source_patch_mu1))
        p = float(np.mean(plus[h].cls_source_patch_mu1))
        deriv = (p - m) / (2.0 * eps)
        rows.append({
            "block": block_index,
            "head": h,
            "mu1_mean_at_minus_eps": m,
            "mu1_mean_at_plus_eps": p,
            "d_mu1write_d_cls_mu2": deriv,
            "abs_derivative": abs(deriv),
        })
    df = pd.DataFrame(rows).sort_values("abs_derivative", ascending=False).reset_index(drop=True)

    derivs = df["d_mu1write_d_cls_mu2"].to_numpy(np.float64)
    # Anchor the coherent group to the strongest head, rather than allowing
    # many tiny opposite-sign heads to decide the group sign by vote.
    dominant_sign = np.sign(derivs[0]) if len(derivs) else 1.0
    if dominant_sign == 0:
        dominant_sign = 1.0
    max_abs = max(float(np.max(np.abs(derivs))), EPS)
    coherent = df[
        (np.sign(df["d_mu1write_d_cls_mu2"]) == dominant_sign)
        & (df["abs_derivative"] >= 0.20 * max_abs)
    ]["head"].astype(int).tolist()
    if not coherent:
        coherent = [int(df.iloc[0]["head"])]
    return df, coherent[:4]


def aggregate_head_maps(head_maps: dict[int, HeadMaps], heads: Sequence[int]) -> HeadMaps:
    hs = [head_maps[int(h)] for h in heads]
    def nansum_maps(name):
        arr = np.stack([np.asarray(getattr(hm, name), np.float64) for hm in hs], axis=0)
        valid = np.any(np.isfinite(arr), axis=0)
        out = np.nansum(arr, axis=0)
        out[~valid] = np.nan
        return out

    cls_q = np.sum([hm.cls_query_patch_attn for hm in hs], axis=0)
    cls_src_norm = np.sum([hm.cls_source_patch_write_norm for hm in hs], axis=0)
    cls_mu1 = np.sum([hm.cls_source_patch_mu1 for hm in hs], axis=0)
    c2r = nansum_maps("cls_to_reg_write_norm")
    r2c = nansum_maps("reg_to_cls_write_norm")
    qk = nansum_maps("qk_cls_to_reg_signed")

    ctot = float(np.nansum(c2r))
    rtot = float(np.nansum(r2c))
    balance = (rtot - ctot) / max(rtot + ctot, EPS)

    return HeadMaps(
        cls_query_patch_attn=cls_q,
        cls_source_patch_write_norm=cls_src_norm,
        cls_source_patch_mu1=cls_mu1,
        cls_to_reg_write_norm=c2r,
        reg_to_cls_write_norm=r2c,
        qk_cls_to_reg_signed=qk,
        cls_to_reg_total=ctot,
        reg_to_cls_total=rtot,
        cls_to_reg_attn=float(sum(hm.cls_to_reg_attn for hm in hs)),
        reg_to_cls_attn=float(sum(hm.reg_to_cls_attn for hm in hs)),
        exchange_balance=balance,
        qk_reg_max_abs=float(np.nanmax(np.abs(qk))) if np.any(np.isfinite(qk)) else 0.0,
        qk_reg_signed_at_max=float(qk[np.nanargmax(np.abs(qk))]) if np.any(np.isfinite(qk)) else 0.0,
    )


# =============================================================================
# Plot helpers
# =============================================================================

def draw_reg_boxes(ax, reg_mask_p: np.ndarray, side: int, image_w: int, image_h: int) -> None:
    pw = image_w / side
    ph = image_h / side
    for idx in register_indices(reg_mask_p):
        row, col = divmod(idx, side)
        ax.add_patch(Rectangle(
            (col * pw, row * ph), pw, ph,
            fill=False, linewidth=1.4, edgecolor="black",
        ))


def plot_patch_map(
    ax,
    image_rgb: np.ndarray,
    values_p: np.ndarray,
    side: int,
    *,
    title: str,
    cmap: str,
    norm,
    reg_mask_p: Optional[np.ndarray] = None,
    alpha: float = 0.68,
    annotate_reg_values: bool = False,
    info: Optional[str] = None,
):
    h, w = image_rgb.shape[:2]
    ax.imshow(image_rgb)
    vals = np.asarray(values_p, np.float64).reshape(side, side)

    if reg_mask_p is not None:
        mask = np.asarray(reg_mask_p, bool).reshape(side, side)
        vals = np.ma.array(vals, mask=~mask | ~np.isfinite(vals))
    else:
        vals = np.ma.array(vals, mask=~np.isfinite(vals))

    im = ax.imshow(
        vals,
        extent=(0, w, h, 0),
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        alpha=alpha,
    )
    if reg_mask_p is not None:
        draw_reg_boxes(ax, reg_mask_p, side, w, h)

    if annotate_reg_values and reg_mask_p is not None:
        pw, ph = w / side, h / side
        flat = np.asarray(values_p, np.float64)
        for idx in register_indices(reg_mask_p):
            if not np.isfinite(flat[idx]):
                continue
            row, col = divmod(idx, side)
            ax.text(
                (col + .5) * pw, (row + .5) * ph,
                f"{flat[idx]:.3g}",
                ha="center", va="center",
                fontsize=6,
                bbox=dict(boxstyle="round,pad=.15", facecolor="white", alpha=.70, edgecolor="none"),
            )

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if info:
        ax.text(
            .02, .98, info,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=7.2,
            bbox=dict(boxstyle="round,pad=.25", facecolor="white", alpha=.82, edgecolor="none"),
        )
    return im


def add_cls_read_box(
    ax,
    value: float,
    *,
    cmap: str,
    norm,
    label: str = "REG→CLS",
    size: float = 0.070,
    y: float = -0.155,
) -> None:
    """
    Draw CLS as a separate, visibly larger square beneath a patch map.

    The square uses the SAME colormap/norm as the register edge-write map.
    `value` is the summed exact REG->CLS edge-write norm for the selected head.
    """
    color = plt.get_cmap(cmap)(norm(float(value)))
    x = 0.035
    rect = Rectangle(
        (x, y),
        size,
        size,
        transform=ax.transAxes,
        clip_on=False,
        facecolor=color,
        edgecolor="black",
        linewidth=1.25,
    )
    ax.add_patch(rect)
    ax.text(
        x + size / 2,
        y + size / 2,
        "CLS",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        clip_on=False,
    )
    ax.text(
        x + size + 0.018,
        y + size / 2,
        f"{label} = {float(value):.4g}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.0,
        clip_on=False,
    )


def global_normalization(
    maps: dict[tuple[int, int], HeadMaps],
    slider_aggregates: Optional[dict[float, HeadMaps]] = None,
) -> dict[str, float]:
    attention = []
    scout_conditional = []
    signed_mu1 = []
    write_norm = []
    write_totals = []
    qk = []

    for (block_index, _head), hm in maps.items():
        raw_attn = np.asarray(hm.cls_query_patch_attn, np.float64)
        attention.append(raw_attn)
        if int(block_index) <= 5:
            scout_conditional.append(
                raw_attn / max(float(np.nansum(raw_attn)), EPS)
            )
        signed_mu1.append(np.asarray(hm.cls_source_patch_mu1, np.float64))
        write_norm.append(np.asarray(hm.cls_to_reg_write_norm, np.float64))
        write_norm.append(np.asarray(hm.reg_to_cls_write_norm, np.float64))
        write_totals.extend([hm.cls_to_reg_total, hm.reg_to_cls_total])
        qk.append(np.asarray(hm.qk_cls_to_reg_signed, np.float64))

    if slider_aggregates:
        for hm in slider_aggregates.values():
            signed_mu1.append(np.asarray(hm.cls_source_patch_mu1, np.float64))
            write_norm.append(np.asarray(hm.cls_to_reg_write_norm, np.float64))
            write_norm.append(np.asarray(hm.reg_to_cls_write_norm, np.float64))

    def finite_max(arrs, abs_value=False, floor=1e-8):
        vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrs if np.any(np.isfinite(a))])
        if not len(vals):
            return floor
        if abs_value:
            vals = np.abs(vals)
        return max(float(np.max(vals)), floor)

    edge_vmax = finite_max(write_norm)
    total_vmax = max([float(v) for v in write_totals] + [0.0])
    return {
        "attention_vmax": finite_max(attention),
        "scout_conditional_vmax": finite_max(
            scout_conditional if scout_conditional else attention
        ),
        "mu1_abs_vmax": finite_max(signed_mu1, abs_value=True),
        # One shared scale for register-edge patches AND the separate CLS box.
        "write_norm_vmax": max(edge_vmax, total_vmax, 1e-8),
        "qk_abs_vmax": finite_max(qk, abs_value=True),
    }


# =============================================================================
# Figures
# =============================================================================

def plot_native_exchange_by_block(metrics: pd.DataFrame, out: Path) -> pd.DataFrame:
    agg = metrics.groupby("block", sort=True)[
        ["cls_to_reg_write_total", "reg_to_cls_write_total", "cls_to_reg_attn", "reg_to_cls_attn"]
    ].sum().reset_index()
    agg["exchange_balance"] = (
        (agg["reg_to_cls_write_total"] - agg["cls_to_reg_write_total"])
        / (agg["reg_to_cls_write_total"] + agg["cls_to_reg_write_total"]).clip(lower=EPS)
    )
    agg.to_csv(out / "native_exchange_by_block.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.7), sharex=True)
    axes[0].plot(agg["block"], agg["cls_to_reg_write_total"], marker="o", label="CLS → REG exact OV write")
    axes[0].plot(agg["block"], agg["reg_to_cls_write_total"], marker="o", label="REG → CLS exact OV write")
    axes[0].set_ylabel("summed edge-write norm")
    axes[0].set_title("Native bidirectional CLS/register traffic — all heads")
    axes[0].legend()
    axes[0].grid(alpha=.2)

    axes[1].plot(agg["block"], agg["exchange_balance"], marker="o")
    axes[1].axhline(0, linewidth=.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("(REG→CLS − CLS→REG) / total")
    axes[1].set_xlabel("ViT block")
    axes[1].set_title("Exchange push–pull balance (+ = register read into CLS)")
    axes[1].grid(alpha=.2)

    for ax in axes:
        for x in (6, 11, 14, 20, 22):
            ax.axvline(x, linewidth=.6, alpha=.22)

    fig.tight_layout()
    example_savefig(fig, out / "01_native_exchange_by_block.png")
    return agg


def plot_role_angles(role: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(role["block"], role["cls_plane_angle_deg"], marker="o", label="CLS plane angle")
    ax.plot(role["block"], role["reg_plane_angle_deg"], marker="s", label="fixed REG mean plane angle")
    ax.axhline(0, linewidth=.8)
    ax.axhline(90, linewidth=.5, alpha=.25)
    ax.axhline(-90, linewidth=.5, alpha=.25)
    for x in (6, 11, 14, 20, 22):
        ax.axvline(x, linewidth=.6, alpha=.22)
    ax.set_xlabel("ViT block")
    ax.set_ylabel("angle in span(mu1,mu2), degrees")
    ax.set_title("Single-image CLS and register-lineage role-plane angle")
    ax.legend()
    ax.grid(alpha=.2)
    example_savefig(fig, out / "02_single_image_role_plane_angles.png")


def plot_phase_atlas(
    phase_df: pd.DataFrame,
    maps: dict[tuple[int, int], HeadMaps],
    image_rgb: np.ndarray,
    run: NativeRun,
    norms: dict[str, float],
    out: Path,
) -> None:
    panels = out / "panels"
    panels.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 10.3))
    axes = axes.ravel()

    for ax, row in zip(axes, phase_df.itertuples(index=False)):
        hm = maps[(int(row.block), int(row.head))]
        phase = str(row.phase)
        info = (
            f"B{int(row.block)} H{int(row.head)}\n"
            f"C→R={hm.cls_to_reg_total:.4g}\n"
            f"R→C={hm.reg_to_cls_total:.4g}\n"
            f"balance={hm.exchange_balance:+.3f}"
        )

        if phase == "scouting":
            raw = np.asarray(hm.cls_query_patch_attn, np.float64)
            raw_mass = max(float(raw.sum()), EPS)
            values = raw / raw_mass
            cmap = "viridis"
            norm = Normalize(0, norms["scout_conditional_vmax"])
            title = (
                f"Scouting: conditional CLS query → patches\n"
                f"B{row.block} H{row.head}  (patch mass renorm.=1)"
            )
            regmask = None
            selected_metric = phase_df.loc[
                phase_df["phase"] == "scouting", "score"
            ].iloc[0]
            info = (
                f"B{int(row.block)} H{int(row.head)}\n"
                f"raw patch mass={raw_mass:.4g}\n"
                f"50/50 contrast={float(selected_metric):.3g}"
            )
        elif phase == "mu1_puff":
            values = hm.cls_source_patch_mu1
            cmap = "turbo"#"coolwarm"
            norm = TwoSlopeNorm(vmin=-norms["mu1_abs_vmax"], vcenter=0, vmax=norms["mu1_abs_vmax"])
            title = f"μ1 puff: CLS-source signed μ1 write\nB{row.block} H{row.head}"
            regmask = None
            info += f"\nmean μ1={np.mean(values):+.4g}"
        elif phase in ("reg_formation", "workspace", "exchange"):
            # Keep the spatial map semantically clean: every highlighted patch
            # is one exact CLS->REG edge.  The reverse REG->CLS aggregate is
            # rendered as a separate CLS token square underneath.
            values = np.asarray(hm.cls_to_reg_write_norm, np.float64)
            cmap = "viridis"
            norm = Normalize(0, norms["write_norm_vmax"])
            title = (
                f"{phase.replace('_',' ').title()}: CLS → REG exact OV write\n"
                f"B{row.block} H{row.head}"
            )
            regmask = run.fixed_reg_mask
            info = (
                f"B{int(row.block)} H{int(row.head)}\n"
                f"CLS→REG total={hm.cls_to_reg_total:.4g}\n"
                f"balance={hm.exchange_balance:+.3f}"
            )
        else:
            values = hm.qk_cls_to_reg_signed
            cmap = "turbo"#"coolwarm"
            norm = TwoSlopeNorm(vmin=-norms["qk_abs_vmax"], vcenter=0, vmax=norms["qk_abs_vmax"])
            title = f"Late QK gate: CLS query ↔ REG keys\nB{row.block} H{row.head}"
            regmask = run.fixed_reg_mask
            info += f"\nmax |QK|={hm.qk_reg_max_abs:.3g}"

        im = plot_patch_map(
            ax, image_rgb, values, run.side,
            title=title, cmap=cmap, norm=norm,
            reg_mask_p=regmask,
            annotate_reg_values=regmask is not None,
            info=info,
        )
        if phase in ("reg_formation", "workspace", "exchange"):
            add_cls_read_box(
                ax,
                hm.reg_to_cls_total,
                cmap=cmap,
                norm=norm,
            )

        # Save each selected panel separately.
        f2, a2 = plt.subplots(
            figsize=(4.8, 5.25)
            if phase in ("reg_formation", "workspace", "exchange")
            else (4.8, 4.8)
        )
        im2 = plot_patch_map(
            a2, image_rgb, values, run.side,
            title=title, cmap=cmap, norm=norm,
            reg_mask_p=regmask,
            annotate_reg_values=regmask is not None,
            info=info,
        )
        if phase in ("reg_formation", "workspace", "exchange"):
            add_cls_read_box(
                a2,
                hm.reg_to_cls_total,
                cmap=cmap,
                norm=norm,
            )
        f2.colorbar(im2, ax=a2, fraction=.046, pad=.04)
        example_savefig(f2, panels / f"phase_{phase}_B{int(row.block):02d}_H{int(row.head):02d}.png")

    for ax in axes[len(phase_df):]:
        ax.axis("off")

    fig.suptitle("Bottle/shower-gel single-image CLS/register computation atlas", fontsize=15)
    fig.tight_layout(rect=[0, .025, 1, .96], h_pad=3.2)
    example_savefig(fig, out / "03_phase_atlas.png")


def plot_mu2_head_scan(
    df: pd.DataFrame,
    chosen: Sequence[int],
    out: Path,
    filename_prefix: str = "",
) -> None:
    z = df.sort_values("head")
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    bars = ax.bar(z["head"], z["d_mu1write_d_cls_mu2"])
    chosen_set = set(int(h) for h in chosen)
    for patch, h in zip(bars, z["head"]):
        if int(h) in chosen_set:
            patch.set_hatch("//")
    ax.axhline(0, linewidth=.8)
    ax.set_xlabel("attention head")
    ax.set_ylabel("d mean(CLS-source μ1 write) / d(CLS·μ2)")
    ax.set_title("Local μ2 → μ1 transduction by head (hatched = selected coherent heads)")
    ax.grid(axis="y", alpha=.2)
    example_savefig(fig, out / f"{filename_prefix}04_mu2_transduction_heads.png")


def plot_mu2_slider(
    slider: dict[float, HeadMaps],
    selected_heads: Sequence[int],
    image_rgb: np.ndarray,
    run: NativeRun,
    norms: dict[str, float],
    block_index: int,
    out: Path,
    filename_prefix: str = "",
    write_norm_vmax_override: Optional[float] = None,
) -> pd.DataFrame:
    factors = sorted(slider)
    n = len(factors)
    write_norm_vmax = float(write_norm_vmax_override if write_norm_vmax_override is not None else norms["write_norm_vmax"])
    fig, axes = plt.subplots(3, n, figsize=(3.25 * n, 10.5), squeeze=False)  # modified: extra height for CLS value boxes

    rows = []
    for col, factor in enumerate(factors):
        hm = slider[factor]
        rows.append({
            "factor_mu2": factor,
            "block": block_index,
            "heads": "|".join(str(int(h)) for h in selected_heads),
            "cls_source_mu1_signed_mean": float(np.mean(hm.cls_source_patch_mu1)),
            "cls_source_mu1_abs_mean": float(np.mean(np.abs(hm.cls_source_patch_mu1))),
            "cls_to_reg_write_total": hm.cls_to_reg_total,
            "reg_to_cls_write_total": hm.reg_to_cls_total,
            "exchange_balance": hm.exchange_balance,
            "cls_to_reg_attn": hm.cls_to_reg_attn,
            "reg_to_cls_attn": hm.reg_to_cls_attn,
        })

        info = (
            f"ΔCLS={factor:+g} μ2\n"
            f"C→R={hm.cls_to_reg_total:.4g}\n"
            f"R→C={hm.reg_to_cls_total:.4g}\n"
            f"bal={hm.exchange_balance:+.3f}"
        )

        im0 = plot_patch_map(
            axes[0, col], image_rgb, hm.cls_source_patch_mu1, run.side,
            title=f"{factor:+g} μ2" if col else f"Signed CLS→patch μ1 write\n{factor:+g} μ2",
            cmap="coolwarm",
            norm=TwoSlopeNorm(
                vmin=-norms["mu1_abs_vmax"], vcenter=0, vmax=norms["mu1_abs_vmax"]
            ),
            info=info,
        )

        im1 = plot_patch_map(
            axes[1, col], image_rgb, hm.cls_to_reg_write_norm, run.side,
            title="CLS → REG" if col == 0 else "",
            cmap="viridis",
            norm=Normalize(0, write_norm_vmax),
            reg_mask_p=run.fixed_reg_mask,
            annotate_reg_values=True,
        )

        im2 = plot_patch_map(
            axes[2, col], image_rgb, hm.reg_to_cls_write_norm, run.side,
            title="REG → CLS" if col == 0 else "",
            cmap="viridis",
            norm=Normalize(0, write_norm_vmax),
            reg_mask_p=run.fixed_reg_mask,
            annotate_reg_values=True,
        )
        add_cls_read_box(  # modified: always show the REG→CLS total as a separate CLS token box
            axes[2, col],
            hm.reg_to_cls_total,
            cmap="viridis",
            norm=Normalize(0, write_norm_vmax),
        )

    fig.suptitle(
        f"B{block_index} μ2 slider — exact selected-head writes; heads={','.join(map(str, selected_heads))}\n"
        f"Shared signed μ1 scale and shared CLS↔REG write-norm scale (vmax={write_norm_vmax:.4g})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, .03, 1, .94], h_pad=3.2)  # modified: preserve space for clip_on=False CLS boxes
    example_savefig(fig, out / f"{filename_prefix}05_mu2_slider_exchange.png")

    slider_df = pd.DataFrame(rows)
    slider_df.to_csv(out / f"{filename_prefix}mu2_slider_metrics.csv", index=False)

    # Scalar summary.
    fig, axes = plt.subplots(2, 1, figsize=(8.7, 7.0), sharex=True)
    axes[0].plot(slider_df["factor_mu2"], slider_df["cls_to_reg_write_total"], marker="o", label="CLS → REG")
    axes[0].plot(slider_df["factor_mu2"], slider_df["reg_to_cls_write_total"], marker="o", label="REG → CLS")
    axes[0].set_ylabel("exact OV edge-write norm")
    axes[0].legend()
    axes[0].grid(alpha=.2)

    axes[1].plot(slider_df["factor_mu2"], slider_df["exchange_balance"], marker="o", label="exchange balance")
    ax2 = axes[1].twinx()
    ax2.plot(
        slider_df["factor_mu2"],
        slider_df["cls_source_mu1_signed_mean"],
        marker="s",
        label="mean CLS-source μ1 write",
    )
    axes[1].axhline(0, linewidth=.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("CLS displacement along μ2")
    axes[1].set_ylabel("exchange balance")
    ax2.set_ylabel("mean signed μ1 write")
    axes[1].grid(alpha=.2)

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.suptitle(f"B{block_index}: μ2 push–pull and μ1 transduction")
    fig.tight_layout(rect=[0, 0, 1, .96])
    example_savefig(fig, out / f"{filename_prefix}06_mu2_slider_scalar.png")
    return slider_df


def mu2_slider_write_scale(slider: dict[float, HeadMaps]) -> float:
    vals: list[float] = []
    for hm in slider.values():
        vals.extend([
            float(np.nanmax(np.abs(hm.cls_to_reg_write_norm))),
            float(np.nanmax(np.abs(hm.reg_to_cls_write_norm))),
            float(abs(hm.cls_to_reg_total)),
            float(abs(hm.reg_to_cls_total)),
        ])
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else 1e-8


def assign_mu2_slider_write_norms(
    slider_by_block: dict[int, dict[float, HeadMaps]]
) -> tuple[dict[int, float], pd.DataFrame, dict[str, Any]]:
    rows = []
    per_block_scale: dict[int, float] = {}
    for block_index, slider in sorted(slider_by_block.items()):
        scale = max(mu2_slider_write_scale(slider), 1e-8)
        per_block_scale[int(block_index)] = float(scale)
        rows.append({"block": int(block_index), "raw_block_write_scale": float(scale)})

    scales = np.array([row["raw_block_write_scale"] for row in rows], dtype=np.float64)
    positive = scales[scales > 0]
    if len(positive) == 0:
        global_scale = 1e-8
        mode = "global"
        ratio = 1.0
        block_to_scale = {int(b): global_scale for b in per_block_scale}
        block_to_group = {int(b): "global" for b in per_block_scale}
    else:
        ratio = float(np.max(positive) / max(np.min(positive), 1e-8))
        global_scale = float(np.max(positive))
        if ratio <= 30.0:
            mode = "global"
            block_to_scale = {int(b): global_scale for b in per_block_scale}
            block_to_group = {int(b): "global" for b in per_block_scale}
        else:
            # modified: when magnitudes differ by orders of magnitude, share scales only within nearby magnitude bins.
            exponents = {int(b): int(math.floor(math.log10(max(v, 1e-8)))) for b, v in per_block_scale.items()}
            grouped_blocks: dict[int, list[int]] = {}
            for b, exp in exponents.items():
                grouped_blocks.setdefault(exp, []).append(int(b))
            grouped_scales = {
                exp: max(per_block_scale[b] for b in blocks)
                for exp, blocks in grouped_blocks.items()
            }
            block_to_scale = {int(b): float(grouped_scales[exp]) for b, exp in exponents.items()}
            block_to_group = {int(b): f"log10_{exp:+d}" for b, exp in exponents.items()}
            mode = "magnitude_bins" if len(grouped_blocks) < len(per_block_scale) else "per_block_bins"

    detail_rows = []
    for row in rows:
        block_index = int(row["block"])
        detail_rows.append({
            **row,
            "assigned_group": block_to_group[block_index],
            "assigned_write_norm_vmax": float(block_to_scale[block_index]),
            "global_ratio_max_over_min": float(ratio),
            "normalization_mode": mode,
        })
    detail_df = pd.DataFrame(detail_rows).sort_values("block").reset_index(drop=True)
    meta = {
        "mode": mode,
        "global_ratio_max_over_min": float(ratio),
        "global_max_write_scale": float(global_scale if len(positive) else 1e-8),
        "block_to_group": {str(k): v for k, v in block_to_group.items()},
        "block_to_scale": {str(k): float(v) for k, v in block_to_scale.items()},
    }
    return block_to_scale, detail_df, meta


# =============================================================================
# NPZ export
# =============================================================================

def save_selected_maps(
    path: Path,
    phase_df: pd.DataFrame,
    maps: dict[tuple[int, int], HeadMaps],
    slider: dict[float, HeadMaps],
    reg_mask: np.ndarray,
) -> None:
    payload: dict[str, np.ndarray] = {
        "fixed_reg_mask": np.asarray(reg_mask, np.uint8),
    }
    for row in phase_df.itertuples(index=False):
        hm = maps[(int(row.block), int(row.head))]
        stem = f"phase_{row.phase}_B{int(row.block):02d}_H{int(row.head):02d}"
        payload[stem + "_cls_query_patch_attn"] = hm.cls_query_patch_attn
        payload[stem + "_cls_source_patch_mu1"] = hm.cls_source_patch_mu1
        payload[stem + "_cls_to_reg_write_norm"] = hm.cls_to_reg_write_norm
        payload[stem + "_reg_to_cls_write_norm"] = hm.reg_to_cls_write_norm
        payload[stem + "_qk_cls_to_reg_signed"] = hm.qk_cls_to_reg_signed

    for factor, hm in slider.items():
        f = str(factor).replace("-", "m").replace("+", "p").replace(".", "p")
        payload[f"mu2_{f}_cls_source_patch_mu1"] = hm.cls_source_patch_mu1
        payload[f"mu2_{f}_cls_to_reg_write_norm"] = hm.cls_to_reg_write_norm
        payload[f"mu2_{f}_reg_to_cls_write_norm"] = hm.reg_to_cls_write_norm

    np.savez_compressed(path, **payload)


# =============================================================================
# CLI / main
# =============================================================================

def example_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image computational CLS/register figure atlas."
    )

    parser.add_argument("--image_path", default=DEFAULT_IMAGE)
    parser.add_argument("--model", choices=MODEL_ORDER, default="finetune_stripped")
    parser.add_argument("--oracle_root", default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--out_dir", default=EXAMPLE_DEFAULT_OUT)

    parser.add_argument(
        "--rn",
        action="store_true",
        help="Append the learned READ_NULL token immediately before B13. "
             "No bridge/router/correction is executed.",
    )
    parser.add_argument("--rn_insert_block", type=int, default=13)

    parser.add_argument("--mu2_block", type=int, default=12)
    parser.add_argument("--mu2_fd_eps", type=float, default=0.5)
    parser.add_argument("--mu2_factors", default="-4,-2,0,2,4")

    parser.add_argument("--final_register_threshold", type=float, default=60.0)
    parser.add_argument("--final_register_min", type=int, default=1)
    parser.add_argument("--final_register_max", type=int, default=4)

    # Loader compatibility with the existing NOP/BROADCAST helper.
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

    args = parser.parse_args()
    args.mu2_factors = example_parse_floats(args.mu2_factors)
    if 0.0 not in args.mu2_factors:
        args.mu2_factors.append(0.0)
        args.mu2_factors = sorted(set(args.mu2_factors))
    if not (0 <= args.mu2_block <= 23):
        parser.error("--mu2_block must be 0..23")
    return args


def example_main() -> None:
    args = example_parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "panels").mkdir(exist_ok=True)
    save_json(out / "config.json", vars(args))

    image_path = resolve_local(args.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image_pil = Image.open(image_path).convert("RGB")
    image_rgb = np.asarray(image_pil)

    base = _backbone_tools
    bundle = base.load_bundle(args.model, args, out / args.model / "load_audit")
    bundle.name = args.model

    visual = bundle.model.visual
    width = int(visual.positional_embedding.shape[1])
    mu1, mu2, basis_path = load_mu_basis(
        Path(args.oracle_root), args.model, width
    )
    print(f"[mu basis] {basis_path}")

    image_tensor = bundle.preprocess(image_pil)
    rn_token = load_trained_rn_token(base, args) if args.rn else None

    print("[capture] native single-image forward...")
    run = capture_native_run(base, bundle, image_tensor, rn_token, args)
    print(
        f"[register lineage] indices={run.fixed_reg_indices}; "
        f"count={len(run.fixed_reg_indices)}"
    )

    metrics, maps = native_metrics_frame(run, bundle, mu1)
    metrics.to_csv(out / "native_per_block_head_metrics.csv", index=False)

    phase_df = choose_phases(metrics)
    phase_df.to_csv(out / "phase_selection.csv", index=False)
    print("[phases]")
    print(phase_df[["phase", "block", "head", "score_metric", "score"]].to_string(index=False))

    role = role_plane_frame(run, mu1, mu2)
    role.to_csv(out / "single_image_role_plane.csv", index=False)

    # Native depth summaries.
    plot_native_exchange_by_block(metrics, out)
    plot_role_angles(role, out)

    # Local mu2 -> mu1 head scan + slider for every block.
    print("[mu2 transduction] scanning all blocks for separate slider plots...")
    all_transduction: list[pd.DataFrame] = []
    all_slider_frames: list[pd.DataFrame] = []
    slider_by_block: dict[int, dict[float, HeadMaps]] = {}
    transduction_by_block: dict[int, pd.DataFrame] = {}
    selected_heads_by_block: dict[int, list[int]] = {}

    for block_index in tqdm(range(len(run.captures)), desc="mu2 all-block sweep", unit="block"):
        trans_df_b, selected_heads_b = scan_mu2_transduction_heads(
            base,
            bundle,
            run,
            mu1,
            mu2,
            block_index=block_index,
            eps=args.mu2_fd_eps,
        )
        trans_df_b["selected_coherent"] = trans_df_b["head"].isin(selected_heads_b)
        transduction_by_block[int(block_index)] = trans_df_b.copy()
        selected_heads_by_block[int(block_index)] = list(selected_heads_b)
        all_transduction.append(trans_df_b.assign(block=block_index))

        slider_b: dict[float, HeadMaps] = {}
        native_capture = run.captures[block_index]
        for factor in args.mu2_factors:
            per_head = evaluate_counterfactual_block(
                base,
                bundle,
                native_capture,
                block_index,
                mu1,
                mu2,
                factor,
                run.patch_count,
                run.fixed_reg_mask,
            )
            slider_b[float(factor)] = aggregate_head_maps(per_head, selected_heads_b)
        slider_by_block[int(block_index)] = slider_b

    block_write_norms, norm_detail_df, norm_meta = assign_mu2_slider_write_norms(slider_by_block)
    norm_detail_df.to_csv(out / "mu2_slider_write_norm_scheme.csv", index=False)
    save_json(out / "mu2_slider_write_norm_scheme.json", norm_meta)
    print(f"[mu2 normalization] mode={norm_meta['mode']}; max/min ratio={norm_meta['global_ratio_max_over_min']:.3g}")

    for block_index in tqdm(range(len(run.captures)), desc="mu2 plotting", unit="block"):
        trans_df_b = transduction_by_block[block_index]
        selected_heads_b = selected_heads_by_block[block_index]
        slider_b = slider_by_block[block_index]
        prefix = f"B{block_index:02d}_"

        plot_mu2_head_scan(trans_df_b, selected_heads_b, out, filename_prefix=prefix)

        block_norms = global_normalization(maps, slider_b)
        block_norms["write_norm_vmax"] = float(block_write_norms[block_index])
        slider_df_b = plot_mu2_slider(
            slider_b,
            selected_heads_b,
            image_rgb,
            run,
            block_norms,
            block_index,
            out,
            filename_prefix=prefix,
            write_norm_vmax_override=float(block_write_norms[block_index]),
        )
        slider_df_b["normalization_mode"] = norm_meta["mode"]
        slider_df_b["write_norm_vmax"] = float(block_write_norms[block_index])
        slider_df_b["assigned_group"] = norm_meta["block_to_group"][str(block_index)]
        all_slider_frames.append(slider_df_b)

    transduction_all_df = pd.concat(all_transduction, ignore_index=True)
    transduction_all_df.to_csv(out / "mu2_transduction_head_scan_all_blocks.csv", index=False)
    slider_all_df = pd.concat(all_slider_frames, ignore_index=True)
    slider_all_df.to_csv(out / "mu2_slider_metrics_all_blocks.csv", index=False)

    # Keep the original unprefixed outputs for the requested reference block as a convenience.
    transduction_df = transduction_by_block[int(args.mu2_block)].copy()
    selected_heads = selected_heads_by_block[int(args.mu2_block)]
    slider = slider_by_block[int(args.mu2_block)]
    transduction_df.to_csv(out / "mu2_transduction_head_scan.csv", index=False)
    slider_df = slider_all_df[slider_all_df["block"] == int(args.mu2_block)].copy()
    slider_df.to_csv(out / "mu2_slider_metrics.csv", index=False)
    print(f"[mu2 transduction] B{args.mu2_block} selected coherent heads = {selected_heads}")

    norms = global_normalization(maps, slider)
    save_json(out / "normalization.json", norms)

    plot_phase_atlas(
        phase_df, maps, image_rgb, run, norms, out
    )
    plot_mu2_head_scan(transduction_df, selected_heads, out)
    plot_mu2_slider(
        slider,
        selected_heads,
        image_rgb,
        run,
        norms,
        args.mu2_block,
        out,
    )

    save_selected_maps(
        out / "selected_patch_maps.npz",
        phase_df,
        maps,
        slider,
        run.fixed_reg_mask,
    )

    # Small audit.
    audit = {
        "model": args.model,
        "rn": bool(args.rn),
        "image_path": str(image_path),
        "mu_basis": basis_path,
        "patch_count": run.patch_count,
        "side": run.side,
        "fixed_register_indices": run.fixed_reg_indices,
        "selected_mu2_transduction_heads": selected_heads,
        "mu2_block": args.mu2_block,
        "mu2_factors": args.mu2_factors,
        "normalization": norms,
    }
    save_json(out / "audit.json", audit)

    include = [
        out / "config.json",
        out / "audit.json",
        out / "normalization.json",
        out / "native_per_block_head_metrics.csv",
        out / "native_exchange_by_block.csv",
        out / "phase_selection.csv",
        out / "single_image_role_plane.csv",
        out / "mu2_transduction_head_scan.csv",
        out / "mu2_slider_metrics.csv",
        out / "mu2_transduction_head_scan_all_blocks.csv",
        out / "mu2_slider_metrics_all_blocks.csv",
        out / "mu2_slider_write_norm_scheme.csv",
        out / "selected_patch_maps.npz",
    ]
    include += sorted(out.glob("*.png"))
    include += sorted((out / "panels").glob("*.png"))

    zpath = out / "compact_summary_workspace_single_image_example.zip"
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

    print("[compact summary]", zpath)

    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# MOTIFS
from typing import Any, Sequence

matplotlib.use("Agg")


MOTIFS_DEFAULT_OUT = r"bottle_shower_mu2_all_head_motifs_catalogue"

GROUP_ORDER = (
    "push_pull_exchange",
    "opposite_push_pull",
    "read_gain",
    "broadcast_gain",
    "source_null_curvature",
    "read_optimum",
    "insensitive",
    "mixed_odd",
)


# =============================================================================
# Generic helpers
# =============================================================================


def motifs_parse_floats(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    return sorted(set(vals))


def motifs_savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def monotonic_direction(y: np.ndarray, tol: float = 1e-12) -> int:
    d = np.diff(np.asarray(y, np.float64))
    if np.all(d >= -tol) and np.any(d > tol):
        return +1
    if np.all(d <= tol) and np.any(d < -tol):
        return -1
    if np.all(np.abs(d) <= tol):
        return 0
    return 2  # non-monotonic


# =============================================================================
# Five-point extraction
# =============================================================================

def collect_all_head_slider_curves(
    story,
    base,
    bundle,
    run,
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    factors: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for block_index in tqdm(
        range(len(run.captures)),
        desc="all-head mu2 sweep",
        unit="block",
    ):
        capture = run.captures[block_index]

        for factor in factors:
            per_head = story.evaluate_counterfactual_block(
                base,
                bundle,
                capture,
                block_index,
                mu1,
                mu2,
                float(factor),
                run.patch_count,
                run.fixed_reg_mask,
            )

            for head, hm in sorted(per_head.items()):
                rows.append({
                    "block": int(block_index),
                    "head": int(head),
                    "factor_mu2": float(factor),

                    "cls_to_reg_write_total": float(hm.cls_to_reg_total),
                    "reg_to_cls_write_total": float(hm.reg_to_cls_total),
                    "exchange_balance": float(hm.exchange_balance),

                    "cls_to_reg_attn": float(hm.cls_to_reg_attn),
                    "reg_to_cls_attn": float(hm.reg_to_cls_attn),

                    "cls_source_mu1_signed_mean": float(
                        np.mean(np.asarray(hm.cls_source_patch_mu1, np.float64))
                    ),
                    "cls_source_mu1_abs_mean": float(
                        np.mean(np.abs(np.asarray(hm.cls_source_patch_mu1, np.float64)))
                    ),

                    "qk_reg_max_abs": float(hm.qk_reg_max_abs),
                    "qk_reg_signed_at_max": float(hm.qk_reg_signed_at_max),
                })

    return pd.DataFrame(rows).sort_values(
        ["block", "head", "factor_mu2"]
    ).reset_index(drop=True)


# =============================================================================
# Curve fits
# =============================================================================

def fit_one_curve(
    factors: np.ndarray,
    y: np.ndarray,
    max_abs_factor: float,
) -> dict[str, float]:
    x = np.asarray(factors, np.float64) / max(max_abs_factor, EPS)
    y = np.asarray(y, np.float64)

    # polyfit returns c*x^2 + s*x + b0
    c, s, b0 = np.polyfit(x, y, deg=2)
    pred = c * x * x + s * x + b0

    zero_idx = int(np.argmin(np.abs(factors)))
    endpoint_mean = 0.5 * (float(y[0]) + float(y[-1]))
    native = float(y[zero_idx])

    return {
        "b0": float(b0),
        "slope": float(s),
        "curvature": float(c),
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "native": native,
        "endpoint_mean": endpoint_mean,
        "native_minus_endpoint_mean": native - endpoint_mean,
        "monotonic_code": int(monotonic_direction(y)),
        "range": float(np.max(y) - np.min(y)),
        "peak": float(np.max(y)),
        "minimum": float(np.min(y)),
    }


def fit_all_heads(
    curves: pd.DataFrame,
) -> pd.DataFrame:
    factors_all = np.sort(curves["factor_mu2"].unique().astype(float))
    max_abs_factor = float(np.max(np.abs(factors_all)))
    if len(factors_all) < 5:
        raise ValueError(
            "Need at least five slider factors for stable slope/curvature motif analysis."
        )
    if not np.any(np.isclose(factors_all, 0.0)):
        raise ValueError("Slider factors must include 0.")

    rows: list[dict[str, Any]] = []

    # Block-level activity scale is useful for distinguishing a flat but real
    # head from numerical dust.
    block_peak: dict[int, float] = {}
    for block, group in curves.groupby("block", sort=True):
        vals = np.r_[
            group["cls_to_reg_write_total"].to_numpy(np.float64),
            group["reg_to_cls_write_total"].to_numpy(np.float64),
        ]
        block_peak[int(block)] = max(float(np.max(vals)), EPS)

    for (block, head), group in curves.groupby(["block", "head"], sort=True):
        z = group.sort_values("factor_mu2")
        factors = z["factor_mu2"].to_numpy(np.float64)
        c2r = z["cls_to_reg_write_total"].to_numpy(np.float64)
        r2c = z["reg_to_cls_write_total"].to_numpy(np.float64)

        fc = fit_one_curve(factors, c2r, max_abs_factor)
        fr = fit_one_curve(factors, r2c, max_abs_factor)

        head_scale = max(
            float(np.max(c2r)),
            float(np.max(r2c)),
            EPS,
        )
        bscale = block_peak[int(block)]

        c2r_rel = c2r / head_scale
        r2c_rel = r2c / head_scale
        curve_corr = safe_corr(c2r_rel, r2c_rel)

        row: dict[str, Any] = {
            "block": int(block),
            "head": int(head),
            "head_response_scale": float(head_scale),
            "block_response_scale": float(bscale),
            "activity_fraction_of_block_peak": float(head_scale / bscale),
            "curve_corr_C2R_R2C": curve_corr,
        }

        for prefix, fit in (("C2R", fc), ("R2C", fr)):
            for key, value in fit.items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_slope_rel"] = float(fit["slope"] / head_scale)
            row[f"{prefix}_curvature_rel"] = float(fit["curvature"] / head_scale)
            row[f"{prefix}_range_rel"] = float(fit["range"] / head_scale)
            row[f"{prefix}_rmse_rel"] = float(fit["rmse"] / head_scale)

        # Endpoint / native values are useful for debug without reopening raw CSV.
        row.update({
            "C2R_at_min_factor": float(c2r[0]),
            "C2R_at_zero": float(c2r[int(np.argmin(np.abs(factors)))]),
            "C2R_at_max_factor": float(c2r[-1]),
            "R2C_at_min_factor": float(r2c[0]),
            "R2C_at_zero": float(r2c[int(np.argmin(np.abs(factors)))]),
            "R2C_at_max_factor": float(r2c[-1]),
        })

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["block", "head"]).reset_index(drop=True)


# =============================================================================
# Motif classification
# =============================================================================

def classify_motifs(fits: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for row in fits.itertuples(index=False):
        sC = float(row.C2R_slope_rel)
        sR = float(row.R2C_slope_rel)
        cC = float(row.C2R_curvature_rel)
        cR = float(row.R2C_curvature_rel)
        r2C = float(row.C2R_r2)
        r2R = float(row.R2C_r2)
        activity = float(row.activity_fraction_of_block_peak)
        corr = float(row.curve_corr_C2R_R2C) if np.isfinite(row.curve_corr_C2R_R2C) else float("nan")

        slope_thr = float(args.slope_threshold)
        curv_thr = float(args.curvature_threshold)
        dominance = float(args.dominance_ratio)

        active = activity >= float(args.activity_floor_fraction)
        weak_effect = (
            max(abs(sC), abs(sR), abs(cC), abs(cR))
            < float(args.insensitive_threshold)
        )
        poor_fit = min(r2C, r2R) < float(args.poor_fit_r2)

        # Independent memberships. These are intentionally not mutually exclusive.
        push_pull = active and (sC <= -slope_thr) and (sR >= slope_thr)
        opposite_push_pull = active and (sC >= slope_thr) and (sR <= -slope_thr)

        read_gain = (
            active
            and abs(sR) >= slope_thr
            and abs(sR) >= dominance * max(abs(sC), EPS)
        )
        broadcast_gain = (
            active
            and abs(sC) >= slope_thr
            and abs(sC) >= dominance * max(abs(sR), EPS)
        )

        source_null = (
            active
            and cC >= curv_thr
            and cC >= float(args.curvature_dominance) * max(abs(sC), slope_thr)
            and float(row.C2R_native_minus_endpoint_mean) < 0.0
        )
        read_optimum = (
            active
            and cR <= -curv_thr
            and abs(cR) >= float(args.curvature_dominance) * max(abs(sR), slope_thr)
            and float(row.R2C_native_minus_endpoint_mean) > 0.0
        )

        insensitive = (not active) or weak_effect

        membership = {
            "push_pull_exchange": bool(push_pull),
            "opposite_push_pull": bool(opposite_push_pull),
            "read_gain": bool(read_gain),
            "broadcast_gain": bool(broadcast_gain),
            "source_null_curvature": bool(source_null),
            "read_optimum": bool(read_optimum),
            "insensitive": bool(insensitive),
        }

        # Scores decide only the concise primary contact-sheet bucket.
        scores = {
            "push_pull_exchange": (
                min(-sC, sR) / slope_thr if push_pull else 0.0
            ),
            "opposite_push_pull": (
                min(sC, -sR) / slope_thr if opposite_push_pull else 0.0
            ),
            "read_gain": (
                abs(sR) / slope_thr if read_gain else 0.0
            ),
            "broadcast_gain": (
                abs(sC) / slope_thr if broadcast_gain else 0.0
            ),
            "source_null_curvature": (
                cC / curv_thr if source_null else 0.0
            ),
            "read_optimum": (
                abs(cR) / curv_thr if read_optimum else 0.0
            ),
            "insensitive": (1.0 if insensitive else 0.0),
        }

        # If the quadratic approximation is genuinely poor, preserve that as
        # "mixed_odd" even if a crude endpoint slope would otherwise match a motif.
        if poor_fit and not insensitive:
            primary = "mixed_odd"
            reason = (
                f"poor quadratic fit: R2_C2R={r2C:.3f}, R2_R2C={r2R:.3f}"
            )
        else:
            candidates = [
                (group, score)
                for group, score in scores.items()
                if score > 0
            ]
            if candidates:
                # Tie-break with GROUP_ORDER for deterministic sheets.
                candidates.sort(
                    key=lambda kv: (
                        -kv[1],
                        GROUP_ORDER.index(kv[0]),
                    )
                )
                primary = candidates[0][0]
                reason = (
                    f"highest motif score={candidates[0][1]:.3f}; "
                    f"sC={sC:+.3f}, sR={sR:+.3f}, "
                    f"cC={cC:+.3f}, cR={cR:+.3f}"
                )
            else:
                primary = "mixed_odd"
                reason = (
                    f"no clean thresholded motif; "
                    f"sC={sC:+.3f}, sR={sR:+.3f}, "
                    f"cC={cC:+.3f}, cR={cR:+.3f}"
                )

        labels = [g for g in GROUP_ORDER[:-1] if membership.get(g, False)]
        if not labels:
            labels = ["mixed_odd"]

        out = {
            "block": int(row.block),
            "head": int(row.head),
            "primary_group": primary,
            "secondary_memberships": "|".join(labels),
            "classification_reason": reason,

            "active": bool(active),
            "poor_quadratic_fit": bool(poor_fit),
            "weak_effect": bool(weak_effect),

            "push_pull_exchange": bool(push_pull),
            "opposite_push_pull": bool(opposite_push_pull),
            "read_gain": bool(read_gain),
            "broadcast_gain": bool(broadcast_gain),
            "source_null_curvature": bool(source_null),
            "read_optimum": bool(read_optimum),
            "insensitive": bool(insensitive),

            "score_push_pull_exchange": float(scores["push_pull_exchange"]),
            "score_opposite_push_pull": float(scores["opposite_push_pull"]),
            "score_read_gain": float(scores["read_gain"]),
            "score_broadcast_gain": float(scores["broadcast_gain"]),
            "score_source_null_curvature": float(scores["source_null_curvature"]),
            "score_read_optimum": float(scores["read_optimum"]),

            "s_C2R_rel": sC,
            "s_R2C_rel": sR,
            "c_C2R_rel": cC,
            "c_R2C_rel": cR,
            "curve_corr_C2R_R2C": corr,
            "C2R_r2": r2C,
            "R2C_r2": r2R,
            "activity_fraction_of_block_peak": activity,
            "head_response_scale": float(row.head_response_scale),
        }
        rows.append(out)

    memberships = pd.DataFrame(rows).sort_values(
        ["block", "head"]
    ).reset_index(drop=True)

    # Per-block summary explicitly answers "does every block contain one?"
    summary_rows = []
    for block, group in memberships.groupby("block", sort=True):
        r = {"block": int(block), "n_heads": int(len(group))}
        for motif in GROUP_ORDER:
            r[f"n_{motif}"] = int(np.sum(group["primary_group"] == motif))

        # Strongest clean push-pull score, regardless of primary assignment.
        if len(group):
            idx = int(group["score_push_pull_exchange"].to_numpy().argmax())
            best = group.iloc[idx]
            r["best_push_pull_head"] = int(best["head"])
            r["best_push_pull_score"] = float(best["score_push_pull_exchange"])

            finite_corr = group[np.isfinite(group["curve_corr_C2R_R2C"])]
            if len(finite_corr):
                best_corr = finite_corr.sort_values(
                    "curve_corr_C2R_R2C", ascending=True
                ).iloc[0]
                r["most_anticorrelated_head"] = int(best_corr["head"])
                r["most_anticorrelated_curve_corr"] = float(
                    best_corr["curve_corr_C2R_R2C"]
                )
            else:
                r["most_anticorrelated_head"] = -1
                r["most_anticorrelated_curve_corr"] = float("nan")
        summary_rows.append(r)

    block_summary = pd.DataFrame(summary_rows).sort_values("block")
    return memberships, block_summary


# =============================================================================
# Contact sheets
# =============================================================================

def get_head_curve(curves: pd.DataFrame, block: int, head: int) -> pd.DataFrame:
    return curves[
        (curves["block"] == int(block))
        & (curves["head"] == int(head))
    ].sort_values("factor_mu2")


def plot_contact_sheet_page(
    group_name: str,
    page_df: pd.DataFrame,
    curves: pd.DataFrame,
    out_path: Path,
    page_number: int,
    total_pages: int,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 9.2), squeeze=False)
    axes_flat = axes.ravel()

    for ax, row in zip(axes_flat, page_df.itertuples(index=False)):
        z = get_head_curve(curves, int(row.block), int(row.head))
        x = z["factor_mu2"].to_numpy(np.float64)
        c = z["cls_to_reg_write_total"].to_numpy(np.float64)
        r = z["reg_to_cls_write_total"].to_numpy(np.float64)
        scale = max(float(np.max(c)), float(np.max(r)), EPS)
        cn = c / scale
        rn = r / scale

        ax.plot(x, cn, marker="o", linewidth=1.4, label="C→R")
        ax.plot(x, rn, marker="s", linewidth=1.4, label="R→C")
        ax.axvline(0, linewidth=.6, alpha=.45)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=.18)
        ax.set_title(f"B{int(row.block):02d} H{int(row.head):02d}", fontsize=10)

        ax.text(
            .02, .03,
            (
                f"sC={float(row.s_C2R_rel):+.2f} sR={float(row.s_R2C_rel):+.2f}\n"
                f"cC={float(row.c_C2R_rel):+.2f} cR={float(row.c_R2C_rel):+.2f}\n"
                f"ρ={float(row.curve_corr_C2R_R2C):+.2f}  peak={scale:.3g}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            bbox=dict(
                boxstyle="round,pad=.18",
                facecolor="white",
                alpha=.72,
                edgecolor="none",
            ),
        )

        if ax is axes_flat[0]:
            ax.legend(fontsize=7, loc="upper right")

    for ax in axes_flat[len(page_df):]:
        ax.axis("off")

    fig.suptitle(
        f"{group_name} — normalized μ2 slider response "
        f"(sheet {page_number:03d}/{total_pages:03d})\n"
        "Each panel normalized by its own max(C→R,R→C); raw scale shown as `peak`.",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, .94])
    motifs_savefig(fig, out_path)


def make_contact_sheets(
    memberships: pd.DataFrame,
    curves: pd.DataFrame,
    out_root: Path,
) -> list[Path]:
    made: list[Path] = []

    for group_name in GROUP_ORDER:
        z = memberships[
            memberships["primary_group"] == group_name
        ].sort_values(["block", "head"]).reset_index(drop=True)

        if z.empty:
            continue

        group_dir = out_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        total_pages = int(math.ceil(len(z) / 12.0))

        for page_idx in range(total_pages):
            page = z.iloc[page_idx * 12:(page_idx + 1) * 12].copy()
            path = group_dir / f"sheet{page_idx + 1:03d}_{group_name}.png"
            plot_contact_sheet_page(
                group_name,
                page,
                curves,
                path,
                page_idx + 1,
                total_pages,
            )
            made.append(path)

    return made


# =============================================================================
# Odd-head debug
# =============================================================================

def plot_odd_head_debug(
    membership_row: pd.Series,
    curves: pd.DataFrame,
    fit_row: pd.Series,
    out_path: Path,
) -> None:
    block = int(membership_row["block"])
    head = int(membership_row["head"])
    z = get_head_curve(curves, block, head)

    x = z["factor_mu2"].to_numpy(np.float64)
    c = z["cls_to_reg_write_total"].to_numpy(np.float64)
    r = z["reg_to_cls_write_total"].to_numpy(np.float64)
    scale = max(float(np.max(c)), float(np.max(r)), EPS)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0))

    axes[0, 0].plot(x, c, marker="o", label="C→R exact OV write")
    axes[0, 0].plot(x, r, marker="s", label="R→C exact OV write")
    axes[0, 0].set_title("Raw write curves")
    axes[0, 0].set_ylabel("write norm")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=.2)

    axes[0, 1].plot(x, c / scale, marker="o", label="C→R")
    axes[0, 1].plot(x, r / scale, marker="s", label="R→C")
    axes[0, 1].set_ylim(-0.05, 1.08)
    axes[0, 1].set_title("Per-head normalized write curves")
    axes[0, 1].grid(alpha=.2)

    axes[1, 0].plot(
        x,
        z["cls_to_reg_attn"].to_numpy(np.float64),
        marker="o",
        label="C→R attention",
    )
    axes[1, 0].plot(
        x,
        z["reg_to_cls_attn"].to_numpy(np.float64),
        marker="s",
        label="R→C attention",
    )
    axes[1, 0].set_title("Routing component")
    axes[1, 0].set_xlabel("CLS displacement along μ2")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=.2)

    ax = axes[1, 1]
    ax.plot(
        x,
        z["cls_source_mu1_signed_mean"].to_numpy(np.float64),
        marker="o",
        label="mean signed CLS→patch μ1 write",
    )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        z["qk_reg_signed_at_max"].to_numpy(np.float64),
        marker="s",
        label="signed max CLS↔REG QK",
    )
    ax.axhline(0, linewidth=.7)
    ax2.axhline(0, linewidth=.7)
    ax.set_title("μ1 transduction + QK gate")
    ax.set_xlabel("CLS displacement along μ2")
    ax.set_ylabel("μ1 write")
    ax2.set_ylabel("QK")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=7, loc="best")
    ax.grid(alpha=.2)

    fig.suptitle(
        f"Odd/mixed response debug — B{block:02d} H{head:02d}\n"
        f"{membership_row['classification_reason']}\n"
        f"R² C→R={float(fit_row['C2R_r2']):.3f}; "
        f"R² R→C={float(fit_row['R2C_r2']):.3f}; "
        f"corr={float(fit_row['curve_corr_C2R_R2C']):+.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, .91])
    motifs_savefig(fig, out_path)


def make_odd_debug(
    memberships: pd.DataFrame,
    fits: pd.DataFrame,
    curves: pd.DataFrame,
    out_dir: Path,
    save_all: bool = False,
) -> tuple[pd.DataFrame, list[Path]]:
    paths = []
    debug_rows = []

    merged = memberships.merge(
        fits,
        on=["block", "head"],
        suffixes=("", "_fit"),
        how="left",
    )

    for _, row in merged.sort_values(["block", "head"]).iterrows():
        is_odd = (
            str(row["primary_group"]) == "mixed_odd"
            or bool(row["poor_quadratic_fit"])
        )
        if not is_odd and not save_all:
            continue

        path = out_dir / f"B{int(row['block']):02d}_H{int(row['head']):02d}_debug.png"
        fit_row = fits[
            (fits["block"] == int(row["block"]))
            & (fits["head"] == int(row["head"]))
        ].iloc[0]

        plot_odd_head_debug(row, curves, fit_row, path)
        paths.append(path)

        debug_rows.append({
            "block": int(row["block"]),
            "head": int(row["head"]),
            "primary_group": str(row["primary_group"]),
            "poor_quadratic_fit": bool(row["poor_quadratic_fit"]),
            "classification_reason": str(row["classification_reason"]),
            "debug_figure": path.name,
        })

    return pd.DataFrame(debug_rows), paths


# =============================================================================
# Text summary
# =============================================================================

def write_summary(
    out: Path,
    memberships: pd.DataFrame,
    block_summary: pd.DataFrame,
    args,
) -> None:
    lines = [
        "ALL-HEAD MU2 RESPONSE MOTIF ATLAS",
        "=" * 78,
        "",
        f"model: {args.model}",
        f"image: {args.image_path}",
        f"slider factors: {args.mu2_factors}",
        "",
        "Primary-group counts:",
        "",
    ]

    counts = (
        memberships["primary_group"]
        .value_counts()
        .reindex(GROUP_ORDER, fill_value=0)
    )
    for group, n in counts.items():
        lines.append(f"  {group:<24} {int(n):>4d}")

    lines += [
        "",
        "Per-block strongest anti-correlated / push-pull head:",
        "",
        "block   best push-pull       most anti-correlated",
        "-" * 62,
    ]
    for row in block_summary.itertuples(index=False):
        lines.append(
            f"B{int(row.block):02d}     "
            f"H{int(row.best_push_pull_head):02d} score={float(row.best_push_pull_score):6.3f}    "
            f"H{int(row.most_anticorrelated_head):02d} rho={float(row.most_anticorrelated_curve_corr):+7.3f}"
        )

    lines += [
        "",
        "Classification thresholds:",
        f"  slope_threshold={args.slope_threshold}",
        f"  curvature_threshold={args.curvature_threshold}",
        f"  insensitive_threshold={args.insensitive_threshold}",
        f"  dominance_ratio={args.dominance_ratio}",
        f"  curvature_dominance={args.curvature_dominance}",
        f"  activity_floor_fraction={args.activity_floor_fraction}",
        f"  poor_fit_r2={args.poor_fit_r2}",
        "",
        "Important: `primary_group` is a plotting bucket, not a claim that a head",
        "has one immutable semantic role. Boolean membership columns preserve",
        "hybrid motifs. `mixed_odd` and poor-fit heads have full debug figures.",
        "",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Synthetic classifier regression
# =============================================================================

def synthetic_fit_regression() -> None:
    factors = np.asarray([-4, -2, 0, 2, 4], np.float64)
    max_abs = 4.0
    x = factors / max_abs

    examples = {
        "push_pull": (1.0 - 0.5*x, 0.4 + 0.4*x),
        "opposite": (0.4 + 0.4*x, 1.0 - 0.5*x),
        "source_null": (0.2 + 0.6*x*x, np.full_like(x, 0.2)),
        "read_optimum": (np.full_like(x, 0.2), 0.8 - 0.5*x*x),
    }
    for name, (c, r) in examples.items():
        fc = fit_one_curve(factors, c, max_abs)
        fr = fit_one_curve(factors, r, max_abs)
        if name == "push_pull":
            assert fc["slope"] < 0 and fr["slope"] > 0
        elif name == "opposite":
            assert fc["slope"] > 0 and fr["slope"] < 0
        elif name == "source_null":
            assert fc["curvature"] > 0
        elif name == "read_optimum":
            assert fr["curvature"] < 0


# =============================================================================
# CLI / main
# =============================================================================

def motifs_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify all B00..B23/H00..H15 mu2 slider response motifs."
    )

    parser.add_argument("--image_path", default=DEFAULT_IMAGE)
    parser.add_argument("--model", choices=MODEL_ORDER, default="finetune_stripped")
    parser.add_argument("--oracle_root", default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--out_dir", default=MOTIFS_DEFAULT_OUT)

    parser.add_argument("--mu2_factors", default="-4,-2,0,2,4")

    # Operational motif thresholds are dimensionless after per-head scaling.
    parser.add_argument("--slope_threshold", type=float, default=0.10)
    parser.add_argument("--curvature_threshold", type=float, default=0.12)
    parser.add_argument("--insensitive_threshold", type=float, default=0.06)
    parser.add_argument("--dominance_ratio", type=float, default=1.8)
    parser.add_argument("--curvature_dominance", type=float, default=0.85)
    parser.add_argument("--activity_floor_fraction", type=float, default=0.005)
    parser.add_argument("--poor_fit_r2", type=float, default=0.85)

    parser.add_argument(
        "--save_all_head_debug",
        action="store_true",
        help="Save full debug plots for every head, not only odd/poor-fit heads.",
    )

    # Optional RN insertion, matching the story script.
    parser.add_argument("--rn", action="store_true")
    parser.add_argument("--rn_insert_block", type=int, default=13)

    parser.add_argument("--final_register_threshold", type=float, default=60.0)
    parser.add_argument("--final_register_min", type=int, default=1)
    parser.add_argument("--final_register_max", type=int, default=4)

    # Loader compatibility.
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

    args = parser.parse_args()
    args.mu2_factors = motifs_parse_floats(args.mu2_factors)

    if len(args.mu2_factors) < 5:
        parser.error("--mu2_factors must contain at least five values.")
    if 0.0 not in args.mu2_factors:
        parser.error("--mu2_factors must include 0.")
    if min(args.mu2_factors) >= 0 or max(args.mu2_factors) <= 0:
        parser.error("--mu2_factors must span negative and positive values.")

    return args


def motifs_main() -> None:
    args = motifs_parse_args()
    synthetic_fit_regression()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    contact_root = out / "contact_sheets"
    odd_root = out / "odd_heads"
    contact_root.mkdir(exist_ok=True)
    odd_root.mkdir(exist_ok=True)
    save_json(out / "config.json", vars(args))


    base = _backbone_tools

    image_path = resolve_local(args.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image_pil = Image.open(image_path).convert("RGB")

    bundle = base.load_bundle(
        args.model,
        args,
        out / args.model / "load_audit",
    )
    bundle.name = args.model

    visual = bundle.model.visual
    width = int(visual.positional_embedding.shape[1])
    mu1, mu2, basis_path = story.load_mu_basis(
        Path(args.oracle_root),
        args.model,
        width,
    )
    print(f"[mu basis] {basis_path}")

    image_tensor = bundle.preprocess(image_pil)
    rn_token = story.load_trained_rn_token(base, args) if args.rn else None

    print("[capture] native single-image forward...")
    run = story.capture_native_run(
        base,
        bundle,
        image_tensor,
        rn_token,
        args,
    )
    print(
        f"[register lineage] indices={run.fixed_reg_indices}; "
        f"count={len(run.fixed_reg_indices)}"
    )

    print("[slider] collecting all 24 x 16 head response curves...")
    curves = collect_all_head_slider_curves(
        story,
        base,
        bundle,
        run,
        mu1,
        mu2,
        args.mu2_factors,
    )
    curves.to_csv(out / "mu2_head_slider_curves.csv", index=False)

    print("[fit] quadratic response descriptors...")
    fits = fit_all_heads(curves)
    fits.to_csv(out / "mu2_head_motif_fits.csv", index=False)

    print("[classify] operational head motifs...")
    memberships, block_summary = classify_motifs(fits, args)
    memberships.to_csv(out / "mu2_head_motif_memberships.csv", index=False)
    block_summary.to_csv(out / "block_motif_summary.csv", index=False)

    print("[plot] contact sheets...")
    contact_paths = make_contact_sheets(
        memberships,
        curves,
        contact_root,
    )

    print("[debug] odd / poor-fit heads...")
    odd_debug_df, odd_paths = make_odd_debug(
        memberships,
        fits,
        curves,
        odd_root,
        save_all=args.save_all_head_debug,
    )
    odd_debug_df.to_csv(out / "odd_head_debug.csv", index=False)

    write_summary(out, memberships, block_summary, args)

    audit = {
        "model": args.model,
        "rn": bool(args.rn),
        "image_path": str(image_path),
        "mu_basis": basis_path,
        "fixed_register_indices": [
            int(x) for x in run.fixed_reg_indices
        ],
        "slider_factors": [float(x) for x in args.mu2_factors],
        "n_blocks": int(curves["block"].nunique()),
        "n_heads_per_block": int(curves.groupby("block")["head"].nunique().max()),
        "n_head_curves": int(len(fits)),
        "n_mixed_odd": int(np.sum(memberships["primary_group"] == "mixed_odd")),
        "n_poor_fit": int(np.sum(memberships["poor_quadratic_fit"])),
        "contact_sheet_count": int(len(contact_paths)),
        "odd_debug_figure_count": int(len(odd_paths)),
        "thresholds": {
            "slope_threshold": args.slope_threshold,
            "curvature_threshold": args.curvature_threshold,
            "insensitive_threshold": args.insensitive_threshold,
            "dominance_ratio": args.dominance_ratio,
            "curvature_dominance": args.curvature_dominance,
            "activity_floor_fraction": args.activity_floor_fraction,
            "poor_fit_r2": args.poor_fit_r2,
        },
    }
    save_json(out / "audit.json", audit)

    include = [
        out / "config.json",
        out / "audit.json",
        out / "SUMMARY.txt",
        out / "mu2_head_slider_curves.csv",
        out / "mu2_head_motif_fits.csv",
        out / "mu2_head_motif_memberships.csv",
        out / "block_motif_summary.csv",
        out / "odd_head_debug.csv",
    ]
    include += contact_paths
    include += odd_paths

    zpath = out / "compact_summary_workspace_single_image_motifs.zip"
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
                archive.write(
                    path,
                    arcname=path.relative_to(out).as_posix(),
                )

    print("[compact summary]", zpath)

    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


from types import SimpleNamespace
story = SimpleNamespace(evaluate_counterfactual_block=evaluate_counterfactual_block, capture_native_run=capture_native_run, load_mu_basis=load_mu_basis, load_trained_rn_token=load_trained_rn_token)


def main(argv=None):
    """Dispatch a workflow; each subcommand retains its original CLI options."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {'example': example_main, 'motifs': motifs_main}
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

