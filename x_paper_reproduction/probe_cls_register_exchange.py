#!/usr/bin/env python3
r"""CLS <-> GIPU exchange probe, NO RN token (v3 postprocess recursion fix)
========================================

Focused follow-up to:
    probe_broadcast_sinks.py

Questions
---------
1) B20 H14:
   Is the conserved CLS-preferring broadcast head actually a two-way exchange
   between CLS and the register/cache/workspace substrate?

2) B12 -> B13:
   In the ordinary (no-RN) ViT, H5/H11 and neighboring heads interrogate
   register addresses.  Is CLS one of the queries participating in that read,
   or is the operation predominantly patch/workspace -> register?

3) B6 -> B10:
   The GIPU analysis shows the early spread of the conserved mu-like state.
   Does CLS materially WRITE that direction into patches, or is the spread
   mostly patch/register/cache internal?

4) Register-pump ablation:
   When the B11/B12 high-norm register implementation is removed, does the
   CLS route become more important, consistent with a redundant/stabilizing
   workspace implementation?

No RN token is inserted anywhere in this script.

Models
------
Same three ordinary visual towers as the sink analysis:
  pretrained
  gmp
  finetune_stripped

The stripped x-attn visual uses ordinary ViT weights only.  The loader and
strict visual-transplant logic are imported from the supplied sink script, so
the model construction is identical to the previous NOP/BROADCAST run.

Operational GIPU roles
----------------------
A first intact pass fits a per-model invariant register basis:

  * B23 register lineage:
      final B23 patch norm >= --final_register_threshold, capped to 1--4 by
      default with top-norm fallback.

  * mu1/mu2:
      top two uncentered right-singular directions of the per-image B23
      register means.

In the second pass, every selected block partitions spatial tokens into:

  REG_LINEAGE
      the fixed B23 register addresses tracked backward;

  MU_CACHE
      non-register tokens with current norm below --cache_norm_max and signed
      cosine to fixed B23 mu1 >= --cache_cos_threshold;

  SCRATCH_PROXY
      remaining tokens with high within-image distant coherence after removing
      span(mu1,mu2).  This is intentionally a *proxy*, not the full earlier
      scratchpad detector: it does NOT subtract the cross-image maximum;

  PATCH_OTHER
      the remaining spatial tokens.

The original intact pre-B13 visible-register mask is also tracked separately as
B13_REG.  It is non-exclusive and is used specifically for the B12/B13
register-read question.

Directed measurements
----------------------
For all heads at selected blocks:

  CLS gather:
      Q=CLS -> source role
      attention mass
      projected |A*V| source fraction
      signed mu1/mu2 residual-stream contribution

  CLS broadcast:
      query role -> K/V=CLS
      attention to CLS
      projected |A*V| fraction supplied by CLS
      signed mu1/mu2 contribution supplied by CLS

  B13 register interrogation:
      each query role -> B13_REG keys/values

  Head write:
      each head's total attention update projected onto mu1/mu2.

Early mu-spread decomposition
-----------------------------
For every image/block, the exact attention-stage delta along mu1/mu2 is
decomposed into source contributions:

    CLS + REG_LINEAGE + MU_CACHE + SCRATCH_PROXY + PATCH_OTHER + out_proj bias.

This lets us ask directly whether the early patchwise mu1 write is explained by
the CLS source.  A numerical decomposition error is logged.

Outputs
-------
<out>/
  config.json
  <model>/mu_basis.npz
  <model>/mu_basis_audit.csv
  all_head_flow_summary.csv
  focus_head_per_image.csv
  cls_state_per_image.csv
  early_mu_source_decomposition_per_image.csv
  role_counts_per_image.csv
  focus_summary.csv
  early_mu_summary.csv
  plots/
    01_CLS_to_B13reg_focus_heads.png
    02_B20_H14_bidirectional_exchange.png
    03_early_mu1_source_fraction.png
    04_CLS_mu_state_by_block.png
    05_H14_GIPU_exchange_trajectory.png
    06_mu1_head_write_heatmap_*.png
    07_no_pump_H14_delta.png
  SUMMARY.txt
  compact_summary_workspace_cls_register_exchange.zip

The compact handoff omits no essential tables; there are no Blender assets. :)
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
import probe_tools_backbone as _backbone_tools
from probe_tools_analysis import (select_register_mask)


import argparse
import gc
import json
import math
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


DEFAULT_MANIFEST = "nop_bc/fixed_sink_manifest.csv"
DEFAULT_OUT = r"cls_gipu_exchange_no_rn"

MODEL_ORDER = ("pretrained", "gmp", "finetune_stripped")
MODE_INTACT = "intact"
MODE_NOPUMP = "no_pump"

DEFAULT_BLOCKS = "6,7,8,9,10,11,12,13,20,21,22"
DEFAULT_FOCUS_HEADS = "5,11,12,14,15"
DEFAULT_FOCUS_BLOCKS = "10,11,12,13,20,21,22"

ROLE_NAMES = ("REG_LINEAGE", "MU_CACHE", "SCRATCH_PROXY", "PATCH_OTHER")
ALL_SOURCE_NAMES = ("CLS",) + ROLE_NAMES


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


def qnan(x: Iterable[float], q: float = 0.5) -> float:
    a = np.asarray(list(x), np.float64)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if len(a) else float("nan")


def meanfinite(x: Iterable[float]) -> float:
    a = np.asarray(list(x), np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def pearson_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x,y [B,P] -> [B], NaN if one vector has no variance.
    """
    x = x.float()
    y = y.float()
    xm = x - x.mean(dim=-1, keepdim=True)
    ym = y - y.mean(dim=-1, keepdim=True)
    nx = torch.sqrt((xm * xm).sum(dim=-1))
    ny = torch.sqrt((ym * ym).sum(dim=-1))
    den = nx * ny
    r = (xm * ym).sum(dim=-1) / den.clamp_min(eps)
    return torch.where(den > eps, r, torch.full_like(r, float("nan")))


def fit_uncentered_basis(x_nd: np.ndarray, rank: int = 2) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_nd, np.float64)
    n, d = x.shape
    k = min(rank, n, d)
    if n <= d:
        gram = x @ x.T
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1][:k]
        evals = np.clip(evals[order], 0.0, None)
        s = np.sqrt(evals)
        left = evecs[:, order]
        vecs = []
        for j in range(k):
            if s[j] <= 1e-12:
                vecs.append(np.zeros(d, np.float64))
            else:
                vecs.append((left[:, j].T @ x) / s[j])
        basis = np.stack(vecs)
    else:
        _u, s, vt = np.linalg.svd(x, full_matrices=False)
        basis = vt[:k]

    basis /= np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)

    # Deterministic orientation.  mu1 points toward the global register mean.
    mean = x.mean(axis=0)
    if float(basis[0] @ mean) < 0:
        basis[0] *= -1
    if len(basis) > 1:
        # Orient mu2 toward the mean residual after mu1 removal.
        r = x - (x @ basis[0:1].T) @ basis[0:1]
        rmean = r.mean(axis=0)
        if float(basis[1] @ rmean) < 0:
            basis[1] *= -1
    return basis.astype(np.float32), s[:k].astype(np.float32)


def mask_to_pipe(mask_p: torch.Tensor) -> str:
    idx = torch.nonzero(mask_p, as_tuple=False).flatten().cpu().tolist()
    return "|".join(map(str, idx))


def projected_direction_per_head(
    Wout_dd: torch.Tensor,
    direction_d: torch.Tensor,
    heads: int,
) -> torch.Tensor:
    """
    For each head h, return w_h [dh] such that:
        (z_h @ W_h^T) dot direction == z_h dot w_h
    where W_h = Wout[:, head_slice].
    """
    D = Wout_dd.shape[0]
    dh = D // heads
    ws = []
    for h in range(heads):
        W = Wout_dd[:, h * dh:(h + 1) * dh]
        ws.append(W.T @ direction_d)
    return torch.stack(ws, dim=0)  # [H,dh]


def role_mean_bh(
    x_bht: torch.Tensor,
    mask_bp: torch.Tensor,
) -> torch.Tensor:
    """
    x [B,H,T], mask [B,P] over spatial token ids 1..P -> [B,H].
    Empty role -> NaN.
    """
    spatial = x_bht[:, :, 1:]
    m = mask_bp[:, None, :].float()
    den = m.sum(dim=-1)
    num = (spatial * m).sum(dim=-1)
    out = num / den.clamp_min(1.0)
    return torch.where(den > 0, out, torch.full_like(out, float("nan")))


def role_sum_bh(
    x_bht: torch.Tensor,
    mask_bp: torch.Tensor,
) -> torch.Tensor:
    spatial = x_bht[:, :, 1:]
    return (spatial * mask_bp[:, None, :].float()).sum(dim=-1)


def source_sum_cls_bh(
    x_bht: torch.Tensor,
    mask_bp: torch.Tensor,
) -> torch.Tensor:
    """
    x [B,H,T_source] for Q=CLS; sum selected patch sources.
    """
    return (x_bht[:, :, 1:] * mask_bp[:, None, :].float()).sum(dim=-1)


def make_far_mask(side: int, min_distance: float, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij",
    )
    xy = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1).float()
    dist = torch.cdist(xy, xy)
    return dist >= float(min_distance)


@dataclass
class ModelOracle:
    mu_basis: np.ndarray                 # [2,D]
    singular_values: np.ndarray          # [2]
    b23_reg_mask: np.ndarray             # [N,P], uint8
    b13_reg_mask: np.ndarray             # [N,P], uint8
    b23_reg_count: np.ndarray            # [N]
    b13_reg_count: np.ndarray            # [N]


# =============================================================================
# First pass: fit invariant mu basis + fixed lineages
# =============================================================================

@torch.no_grad()
def fit_model_oracle(base, bundle, manifest: pd.DataFrame, args, model_dir: Path) -> ModelOracle:
    v = bundle.model.visual
    nimg = len(manifest)

    # Infer geometry.
    first = bundle.preprocess(Image.open(str(manifest.iloc[0].path)).convert("RGB"))
    _ = first
    P = int(v.positional_embedding.shape[0] - 1)
    D = int(v.positional_embedding.shape[1])

    b23_mask_all = np.zeros((nimg, P), np.uint8)
    b13_mask_all = np.zeros((nimg, P), np.uint8)
    b23_count = np.zeros(nimg, np.int16)
    b13_count = np.zeros(nimg, np.int16)
    reg_means = np.zeros((nimg, D), np.float32)

    for st in tqdm(range(0, nimg, args.batch_size), desc="mu-basis pass", unit="batch"):
        q = manifest.iloc[st:st + args.batch_size]
        tensors = []
        for r in q.itertuples(index=False):
            with Image.open(str(r.path)) as im:
                tensors.append(bundle.preprocess(im.convert("RGB")))
        images = torch.stack(tensors).to(bundle.device, dtype=bundle.model.dtype)

        x = v._prepare_tokens(images)
        B = x.shape[1]

        local_b13 = torch.zeros(B, P, dtype=torch.bool, device=x.device)

        for b, blk in enumerate(v.transformer.resblocks):
            if b == 13:
                pren = x[1:].float().norm(dim=-1).T
                local_b13 = pren >= float(args.b13_register_threshold)

            # Intact ordinary forward, no RN.
            x = blk(x)

        final_patch = x[1:].permute(1, 0, 2).float()
        fnorm = final_patch.norm(dim=-1)
        local_b23 = select_register_mask(
            fnorm,
            args.final_register_threshold,
            args.final_register_min,
            args.final_register_max,
        )

        for j in range(B):
            gi = st + j
            b23_mask_all[gi] = local_b23[j].cpu().numpy().astype(np.uint8)
            b13_mask_all[gi] = local_b13[j].cpu().numpy().astype(np.uint8)
            b23_count[gi] = int(local_b23[j].sum())
            b13_count[gi] = int(local_b13[j].sum())
            reg_means[gi] = final_patch[j, local_b23[j]].mean(dim=0).cpu().numpy()

        del images, x, final_patch, fnorm
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()

    basis, s = fit_uncentered_basis(reg_means, rank=2)

    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "mu_basis.npz",
        mu_basis=basis,
        singular_values=s,
        b23_reg_mask=b23_mask_all,
        b13_reg_mask=b13_mask_all,
        b23_reg_count=b23_count,
        b13_reg_count=b13_count,
        register_means=reg_means,
        stim_id=manifest.stim_id.astype(str).to_numpy(dtype=object),
    )

    rows = []
    for i, r in enumerate(manifest.itertuples(index=False)):
        m23 = torch.from_numpy(b23_mask_all[i].astype(bool))
        m13 = torch.from_numpy(b13_mask_all[i].astype(bool))
        inter = int((m23 & m13).sum())
        union = int((m23 | m13).sum())
        rows.append({
            "stim_id": str(r.stim_id),
            "source": str(r.source),
            "b23_register_count": int(b23_count[i]),
            "b13_visible_register_count": int(b13_count[i]),
            "b23_b13_jaccard": inter / union if union else 1.0,
            "b23_register_indices": mask_to_pipe(m23),
            "b13_register_indices": mask_to_pipe(m13),
        })
    pd.DataFrame(rows).to_csv(model_dir / "mu_basis_audit.csv", index=False)

    print(
        f"[mu] {model_dir.name}: sigma1={float(s[0]):.4g} "
        f"sigma2={float(s[1]):.4g}; mean B23 regs={b23_count.mean():.3f}; "
        f"mean B13 visible regs={b13_count.mean():.3f}"
    )

    return ModelOracle(
        mu_basis=basis,
        singular_values=s,
        b23_reg_mask=b23_mask_all,
        b13_reg_mask=b13_mask_all,
        b23_reg_count=b23_count,
        b13_reg_count=b13_count,
    )


# =============================================================================
# Dynamic GIPU roles
# =============================================================================

def dynamic_roles(
    patch_bpd: torch.Tensor,
    fixed_reg_bp: torch.Tensor,
    mu1_d: torch.Tensor,
    mu2_d: torch.Tensor,
    far_pp: torch.Tensor,
    args,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Returns exclusive role masks + mu1 cosine + distant coherence.
    """
    patch = patch_bpd.float()
    norms = patch.norm(dim=-1)
    cos1 = F.cosine_similarity(
        patch,
        mu1_d.view(1, 1, -1),
        dim=-1,
        eps=1e-8,
    )

    reg = fixed_reg_bp.bool()
    cache = (
        (~reg)
        & (norms < float(args.cache_norm_max))
        & (cos1 >= float(args.cache_cos_threshold))
    )

    # Within-image distant-coherence proxy after removing fixed span(mu1,mu2).
    basis = torch.stack([mu1_d, mu2_d], dim=0)  # [2,D]
    coeff = torch.einsum("bpd,kd->bpk", patch, basis)
    resid = patch - torch.einsum("bpk,kd->bpd", coeff, basis)
    xn = F.normalize(resid, dim=-1, eps=1e-8)
    sim = torch.bmm(xn, xn.transpose(1, 2))
    sim = sim.masked_fill(~far_pp[None, :, :], float("-inf"))
    distant = sim.max(dim=-1).values
    distant = torch.where(
        torch.isfinite(distant),
        distant,
        torch.full_like(distant, float("nan")),
    )

    scratch = (
        (~reg)
        & (~cache)
        & torch.isfinite(distant)
        & (distant >= float(args.scratch_proxy_threshold))
    )
    other = (~reg) & (~cache) & (~scratch)

    roles = {
        "REG_LINEAGE": reg,
        "MU_CACHE": cache,
        "SCRATCH_PROXY": scratch,
        "PATCH_OTHER": other,
    }
    return roles, cos1, distant


# =============================================================================
# Aggregator
# =============================================================================

class MeanAccumulator:
    def __init__(self):
        self.sum: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.count: Dict[Tuple[Any, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add(self, key: Tuple[Any, ...], metric: str, values: np.ndarray | torch.Tensor | float):
        if isinstance(values, torch.Tensor):
            a = values.detach().float().cpu().numpy().reshape(-1)
        else:
            a = np.asarray(values, np.float64).reshape(-1)
        a = a[np.isfinite(a)]
        if not len(a):
            return
        self.sum[key][metric] += float(a.sum())
        self.count[key][metric] += int(len(a))

    def rows(self, key_names: Sequence[str]) -> list[dict[str, Any]]:
        out = []
        for key in sorted(self.sum.keys()):
            row = {k: v for k, v in zip(key_names, key)}
            for metric, s in self.sum[key].items():
                n = self.count[key][metric]
                row[metric + "_mean"] = s / n if n else float("nan")
                row[metric + "_n"] = n
            out.append(row)
        return out


# =============================================================================
# Focus metric extraction
# =============================================================================

def _focus_row(
    model: str,
    mode: str,
    stim_id: str,
    source: str,
    block: int,
    head: int,
) -> dict[str, Any]:
    return {
        "model_name": model,
        "mode": mode,
        "stim_id": stim_id,
        "source": source,
        "block": int(block),
        "head": int(head),
    }


def _put_role_metrics(
    row: dict[str, Any],
    prefix: str,
    role: str,
    attn_bh: torch.Tensor,
    av_bh: torch.Tensor,
    mu1_bh: torch.Tensor,
    mu2_bh: torch.Tensor,
    bi: int,
    h: int,
) -> None:
    tag = role.lower()
    row[f"{prefix}_{tag}_attn"] = float(attn_bh[bi, h])
    row[f"{prefix}_{tag}_avfrac"] = float(av_bh[bi, h])
    row[f"{prefix}_{tag}_mu1_write"] = float(mu1_bh[bi, h])
    row[f"{prefix}_{tag}_mu2_write"] = float(mu2_bh[bi, h])


# =============================================================================
# Second pass
# =============================================================================

@torch.no_grad()
def collect_mode(
    base,
    bundle,
    manifest: pd.DataFrame,
    oracle: ModelOracle,
    model_name: str,
    mode: str,
    args,
    all_acc: MeanAccumulator,
    focus_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    early_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
) -> None:
    v = bundle.model.visual
    blocks = set(args.blocks)
    focus_heads = set(args.focus_heads)
    focus_blocks = set(args.focus_blocks)
    nimg = len(manifest)

    H = int(v.transformer.resblocks[0].attn.num_heads)
    D = int(v.positional_embedding.shape[1])
    P = int(v.positional_embedding.shape[0] - 1)
    side = int(round(math.sqrt(P)))
    if side * side != P:
        raise RuntimeError(f"Expected square patch grid, P={P}")

    mu1 = torch.from_numpy(oracle.mu_basis[0]).to(bundle.device, dtype=torch.float32)
    mu2 = torch.from_numpy(oracle.mu_basis[1]).to(bundle.device, dtype=torch.float32)
    far = make_far_mask(side, args.scratch_min_distance, bundle.device)

    for st in tqdm(
        range(0, nimg, args.batch_size),
        desc=f"{model_name}/{mode}",
        unit="batch",
    ):
        q = manifest.iloc[st:st + args.batch_size]
        tensors, ids, sources = [], [], []
        for r in q.itertuples(index=False):
            with Image.open(str(r.path)) as im:
                tensors.append(bundle.preprocess(im.convert("RGB")))
            ids.append(str(r.stim_id))
            sources.append(str(r.source))
        images = torch.stack(tensors).to(bundle.device, dtype=bundle.model.dtype)
        B = images.shape[0]

        fixed_reg = torch.from_numpy(
            oracle.b23_reg_mask[st:st+B].astype(bool)
        ).to(bundle.device)
        fixed_b13 = torch.from_numpy(
            oracle.b13_reg_mask[st:st+B].astype(bool)
        ).to(bundle.device)

        x = v._prepare_tokens(images)

        for b, blk in enumerate(v.transformer.resblocks):
            pre = x
            ln1 = blk.ln_1(x)
            attn_out, probs0 = blk.attention(
                ln1,
                need_weights=True,
                capture=True,
            )
            probs = base.normalize_probs_shape(probs0, B, H, P + 1).float()
            vv = base.normalize_qkv_shape(blk.attn.last_v, B, H, P + 1).float()

            x_attn = x + attn_out
            ln2 = blk.ln_2(x_attn)
            fc = blk.mlp.c_fc(ln2)
            gelu = blk.mlp.gelu(fc)
            gelu = base.apply_pump_ablation(gelu, b, mode)
            x_post = x_attn + blk.mlp.c_proj(gelu)

            if b in blocks:
                patch = pre[1:].permute(1, 0, 2).float()
                roles, cos1, distant = dynamic_roles(
                    patch,
                    fixed_reg,
                    mu1,
                    mu2,
                    far,
                    args,
                )

                # Role audit.
                for bi in range(B):
                    rr = {
                        "model_name": model_name,
                        "mode": mode,
                        "stim_id": ids[bi],
                        "source": sources[bi],
                        "block": b,
                        "b13_reg_count_fixed_intact": int(fixed_b13[bi].sum()),
                    }
                    for role, mask in roles.items():
                        rr[f"{role.lower()}_count"] = int(mask[bi].sum())
                    vals_cache = cos1[bi, roles["MU_CACHE"][bi]]
                    vals_scr = distant[bi, roles["SCRATCH_PROXY"][bi]]
                    rr["mu_cache_mean_cos_mu1"] = (
                        float(vals_cache.mean()) if vals_cache.numel() else float("nan")
                    )
                    rr["scratch_proxy_mean_distant_coherence"] = (
                        float(vals_scr.mean()) if vals_scr.numel() else float("nan")
                    )
                    role_rows.append(rr)

                Wout = blk.attn.out_proj.weight.detach().float()
                vproj = torch.empty(B, H, P + 1, device=bundle.device)
                for h in range(H):
                    vproj[:, h] = base.projected_row_norms_batch(vv[:, h], Wout, h)

                wmu1 = projected_direction_per_head(Wout, mu1, H)  # [H,dh]
                wmu2 = projected_direction_per_head(Wout, mu2, H)
                vmu1 = torch.einsum("bhtd,hd->bht", vv, wmu1)
                vmu2 = torch.einsum("bhtd,hd->bht", vv, wmu2)

                # -----------------------------------------------------------------
                # Q=CLS gather from source roles.
                # -----------------------------------------------------------------
                cls_a = probs[:, :, 0, :]                    # [B,H,T]
                cls_av = cls_a * vproj
                cls_av_den = cls_av.sum(dim=-1).clamp_min(1e-12)
                cls_mu1_token = cls_a * vmu1
                cls_mu2_token = cls_a * vmu2

                gather: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

                # CLS self source.
                gather["CLS"] = (
                    cls_a[:, :, 0],
                    cls_av[:, :, 0] / cls_av_den,
                    cls_mu1_token[:, :, 0],
                    cls_mu2_token[:, :, 0],
                )

                for role, mask in roles.items():
                    gather[role] = (
                        source_sum_cls_bh(cls_a, mask),
                        source_sum_cls_bh(cls_av, mask) / cls_av_den,
                        source_sum_cls_bh(cls_mu1_token, mask),
                        source_sum_cls_bh(cls_mu2_token, mask),
                    )

                # Non-exclusive B13 register probe.
                gather_b13 = (
                    source_sum_cls_bh(cls_a, fixed_b13),
                    source_sum_cls_bh(cls_av, fixed_b13) / cls_av_den,
                    source_sum_cls_bh(cls_mu1_token, fixed_b13),
                    source_sum_cls_bh(cls_mu2_token, fixed_b13),
                )

                # -----------------------------------------------------------------
                # CLS source -> query roles: "broadcast" leg.
                # -----------------------------------------------------------------
                a_from_cls = probs[:, :, :, 0]              # [B,H,Tq]
                av_all = probs * vproj[:, :, None, :]
                av_den_q = av_all.sum(dim=-1).clamp_min(1e-12)
                av_from_cls = a_from_cls * vproj[:, :, 0][:, :, None]
                avfrac_from_cls = av_from_cls / av_den_q

                mu1_from_cls = a_from_cls * vmu1[:, :, 0][:, :, None]
                mu2_from_cls = a_from_cls * vmu2[:, :, 0][:, :, None]

                broadcast: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
                for role, mask in roles.items():
                    broadcast[role] = (
                        role_mean_bh(a_from_cls, mask),
                        role_mean_bh(avfrac_from_cls, mask),
                        role_mean_bh(mu1_from_cls, mask),
                        role_mean_bh(mu2_from_cls, mask),
                    )

                # -----------------------------------------------------------------
                # Query roles -> B13 register keys/values.
                # -----------------------------------------------------------------
                b13_mask_t = torch.cat(
                    [torch.zeros(B, 1, dtype=torch.bool, device=bundle.device), fixed_b13],
                    dim=1,
                )
                a_to_b13 = (
                    probs * b13_mask_t[:, None, None, :].float()
                ).sum(dim=-1)
                av_to_b13 = (
                    av_all * b13_mask_t[:, None, None, :].float()
                ).sum(dim=-1) / av_den_q

                # Signed mu contribution from B13-reg sources.
                mu1_tok = probs * vmu1[:, :, None, :]
                mu2_tok = probs * vmu2[:, :, None, :]
                mu1_to_b13 = (
                    mu1_tok * b13_mask_t[:, None, None, :].float()
                ).sum(dim=-1)
                mu2_to_b13 = (
                    mu2_tok * b13_mask_t[:, None, None, :].float()
                ).sum(dim=-1)

                interrogation: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
                # CLS query.
                interrogation["CLS"] = (
                    a_to_b13[:, :, 0],
                    av_to_b13[:, :, 0],
                    mu1_to_b13[:, :, 0],
                    mu2_to_b13[:, :, 0],
                )
                for role, mask in roles.items():
                    interrogation[role] = (
                        role_mean_bh(a_to_b13, mask),
                        role_mean_bh(av_to_b13, mask),
                        role_mean_bh(mu1_to_b13, mask),
                        role_mean_bh(mu2_to_b13, mask),
                    )

                # -----------------------------------------------------------------
                # Total head write to mu1/mu2 per query role.
                # -----------------------------------------------------------------
                z = torch.matmul(probs, vv)  # [B,H,T,dh]
                zmu1 = torch.einsum("bhtd,hd->bht", z, wmu1)
                zmu2 = torch.einsum("bhtd,hd->bht", z, wmu2)

                # Aggregate all heads, all images.
                for h in range(H):
                    key = (model_name, mode, b, h)
                    for role in ALL_SOURCE_NAMES:
                        ga, gav, gm1, gm2 = gather[role]
                        all_acc.add(key, f"cls_gather_{role.lower()}_attn", ga[:, h])
                        all_acc.add(key, f"cls_gather_{role.lower()}_avfrac", gav[:, h])
                        all_acc.add(key, f"cls_gather_{role.lower()}_mu1_write", gm1[:, h])
                        all_acc.add(key, f"cls_gather_{role.lower()}_mu2_write", gm2[:, h])

                    all_acc.add(key, "cls_gather_b13_reg_attn", gather_b13[0][:, h])
                    all_acc.add(key, "cls_gather_b13_reg_avfrac", gather_b13[1][:, h])
                    all_acc.add(key, "cls_gather_b13_reg_mu1_write", gather_b13[2][:, h])
                    all_acc.add(key, "cls_gather_b13_reg_mu2_write", gather_b13[3][:, h])

                    for role in ROLE_NAMES:
                        ba, bav, bm1, bm2 = broadcast[role]
                        all_acc.add(key, f"cls_broadcast_to_{role.lower()}_attn", ba[:, h])
                        all_acc.add(key, f"cls_broadcast_to_{role.lower()}_avfrac", bav[:, h])
                        all_acc.add(key, f"cls_broadcast_to_{role.lower()}_mu1_write", bm1[:, h])
                        all_acc.add(key, f"cls_broadcast_to_{role.lower()}_mu2_write", bm2[:, h])

                        ia, iav, im1, im2 = interrogation[role]
                        all_acc.add(key, f"{role.lower()}_query_to_b13reg_attn", ia[:, h])
                        all_acc.add(key, f"{role.lower()}_query_to_b13reg_avfrac", iav[:, h])
                        all_acc.add(key, f"{role.lower()}_query_to_b13reg_mu1_write", im1[:, h])
                        all_acc.add(key, f"{role.lower()}_query_to_b13reg_mu2_write", im2[:, h])

                        all_acc.add(key, f"{role.lower()}_head_mu1_write", role_mean_bh(zmu1, roles[role])[:, h])
                        all_acc.add(key, f"{role.lower()}_head_mu2_write", role_mean_bh(zmu2, roles[role])[:, h])

                    all_acc.add(key, "cls_query_to_b13reg_attn", interrogation["CLS"][0][:, h])
                    all_acc.add(key, "cls_query_to_b13reg_avfrac", interrogation["CLS"][1][:, h])
                    all_acc.add(key, "cls_head_mu1_write", zmu1[:, h, 0])
                    all_acc.add(key, "cls_head_mu2_write", zmu2[:, h, 0])

                # -----------------------------------------------------------------
                # Per-image focus heads.
                # -----------------------------------------------------------------
                if b in focus_blocks:
                    for h in sorted(focus_heads):
                        if h >= H:
                            continue
                        for bi in range(B):
                            row = _focus_row(
                                model_name, mode, ids[bi], sources[bi], b, h
                            )
                            # CLS gather.
                            for role in ALL_SOURCE_NAMES:
                                ga, gav, gm1, gm2 = gather[role]
                                _put_role_metrics(
                                    row, "cls_gather_from", role,
                                    ga, gav, gm1, gm2, bi, h,
                                )
                            row["cls_gather_from_b13_reg_attn"] = float(gather_b13[0][bi, h])
                            row["cls_gather_from_b13_reg_avfrac"] = float(gather_b13[1][bi, h])
                            row["cls_gather_from_b13_reg_mu1_write"] = float(gather_b13[2][bi, h])
                            row["cls_gather_from_b13_reg_mu2_write"] = float(gather_b13[3][bi, h])

                            # CLS broadcast to query roles.
                            for role in ROLE_NAMES:
                                ba, bav, bm1, bm2 = broadcast[role]
                                _put_role_metrics(
                                    row, "cls_broadcast_to", role,
                                    ba, bav, bm1, bm2, bi, h,
                                )

                            # B13-register interrogation by query class.
                            for role in ("CLS",) + ROLE_NAMES:
                                ia, iav, im1, im2 = interrogation[role]
                                _put_role_metrics(
                                    row, "b13reg_read_by", role,
                                    ia, iav, im1, im2, bi, h,
                                )

                            row["cls_total_head_mu1_write"] = float(zmu1[bi, h, 0])
                            row["cls_total_head_mu2_write"] = float(zmu2[bi, h, 0])
                            focus_rows.append(row)

                # -----------------------------------------------------------------
                # Exact all-head early mu source decomposition.
                # -----------------------------------------------------------------
                # vmu [B,H,T]; sum source contributions over heads for each query.
                def total_source_mu(vmu_bht: torch.Tensor, src_mask_bp: torch.Tensor) -> torch.Tensor:
                    src_t = torch.cat(
                        [torch.zeros(B, 1, dtype=torch.bool, device=bundle.device), src_mask_bp],
                        dim=1,
                    )
                    contrib = (
                        probs
                        * vmu_bht[:, :, None, :]
                        * src_t[:, None, None, :].float()
                    ).sum(dim=-1).sum(dim=1)  # [B,Tq]
                    return contrib

                cls_src_mu1 = (
                    probs[:, :, :, 0] * vmu1[:, :, 0][:, :, None]
                ).sum(dim=1)
                cls_src_mu2 = (
                    probs[:, :, :, 0] * vmu2[:, :, 0][:, :, None]
                ).sum(dim=1)

                src_mu1 = {"CLS": cls_src_mu1}
                src_mu2 = {"CLS": cls_src_mu2}
                for role in ROLE_NAMES:
                    src_mu1[role] = total_source_mu(vmu1, roles[role])
                    src_mu2[role] = total_source_mu(vmu2, roles[role])

                exact_delta_mu1 = torch.einsum(
                    "tbd,d->bt", attn_out.float(), mu1
                )
                exact_delta_mu2 = torch.einsum(
                    "tbd,d->bt", attn_out.float(), mu2
                )

                bias = blk.attn.out_proj.bias
                bias_mu1 = (
                    float(torch.dot(bias.float(), mu1)) if bias is not None else 0.0
                )
                bias_mu2 = (
                    float(torch.dot(bias.float(), mu2)) if bias is not None else 0.0
                )

                recon_mu1 = sum(src_mu1.values()) + bias_mu1
                recon_mu2 = sum(src_mu2.values()) + bias_mu2
                err1 = (recon_mu1 - exact_delta_mu1).abs().max(dim=-1).values
                err2 = (recon_mu2 - exact_delta_mu2).abs().max(dim=-1).values

                # Patch-only source contribution diagnostics.
                d1p = exact_delta_mu1[:, 1:]
                d2p = exact_delta_mu2[:, 1:]
                c1p = {k: v[:, 1:] for k, v in src_mu1.items()}
                c2p = {k: v[:, 1:] for k, v in src_mu2.items()}

                # State of CLS relative to fixed mu basis.
                pre_cls = pre[0].float()
                attn_cls = x_attn[0].float()
                post_cls = x_post[0].float()

                def state_proj(z_bd: torch.Tensor, direction: torch.Tensor):
                    coef = z_bd @ direction
                    cos = F.cosine_similarity(
                        z_bd,
                        direction.view(1, -1),
                        dim=-1,
                        eps=1e-8,
                    )
                    return coef, cos

                pre_m1, pre_c1 = state_proj(pre_cls, mu1)
                pre_m2, pre_c2 = state_proj(pre_cls, mu2)
                att_m1, att_c1 = state_proj(attn_cls, mu1)
                att_m2, att_c2 = state_proj(attn_cls, mu2)
                post_m1, post_c1 = state_proj(post_cls, mu1)
                post_m2, post_c2 = state_proj(post_cls, mu2)

                for bi in range(B):
                    sr = {
                        "model_name": model_name,
                        "mode": mode,
                        "stim_id": ids[bi],
                        "source": sources[bi],
                        "block": b,
                        "cls_pre_mu1_coef": float(pre_m1[bi]),
                        "cls_pre_mu1_cos": float(pre_c1[bi]),
                        "cls_pre_mu2_coef": float(pre_m2[bi]),
                        "cls_pre_mu2_cos": float(pre_c2[bi]),
                        "cls_postattn_mu1_coef": float(att_m1[bi]),
                        "cls_postattn_mu1_cos": float(att_c1[bi]),
                        "cls_postattn_mu2_coef": float(att_m2[bi]),
                        "cls_postattn_mu2_cos": float(att_c2[bi]),
                        "cls_postblock_mu1_coef": float(post_m1[bi]),
                        "cls_postblock_mu1_cos": float(post_c1[bi]),
                        "cls_postblock_mu2_coef": float(post_m2[bi]),
                        "cls_postblock_mu2_cos": float(post_c2[bi]),
                    }
                    state_rows.append(sr)

                    er = {
                        "model_name": model_name,
                        "mode": mode,
                        "stim_id": ids[bi],
                        "source": sources[bi],
                        "block": b,
                        "mu1_decomp_max_abs_error": float(err1[bi]),
                        "mu2_decomp_max_abs_error": float(err2[bi]),
                        "patch_delta_mu1_abs_mean": float(d1p[bi].abs().mean()),
                        "patch_delta_mu2_abs_mean": float(d2p[bi].abs().mean()),
                    }
                    den1 = float(d1p[bi].abs().sum().clamp_min(1e-12))
                    den2 = float(d2p[bi].abs().sum().clamp_min(1e-12))
                    for role in ALL_SOURCE_NAMES:
                        er[f"mu1_{role.lower()}_source_abs_l1_ratio"] = (
                            float(c1p[role][bi].abs().sum()) / den1
                        )
                        er[f"mu2_{role.lower()}_source_abs_l1_ratio"] = (
                            float(c2p[role][bi].abs().sum()) / den2
                        )
                        er[f"mu1_{role.lower()}_source_signed_mean"] = float(
                            c1p[role][bi].mean()
                        )
                        er[f"mu2_{role.lower()}_source_signed_mean"] = float(
                            c2p[role][bi].mean()
                        )
                        er[f"mu1_delta_corr_{role.lower()}_source"] = float(
                            pearson_torch(
                                d1p[bi:bi+1], c1p[role][bi:bi+1]
                            )[0]
                        )
                        er[f"mu2_delta_corr_{role.lower()}_source"] = float(
                            pearson_torch(
                                d2p[bi:bi+1], c2p[role][bi:bi+1]
                            )[0]
                        )

                    # Does CLS source preferentially write mu1 into current cache /
                    # scratch-proxy patches?
                    for role in ROLE_NAMES:
                        mask = roles[role][bi]
                        if int(mask.sum()):
                            er[f"mu1_cls_source_mean_on_{role.lower()}"] = float(
                                c1p["CLS"][bi, mask].mean()
                            )
                            er[f"mu1_delta_mean_on_{role.lower()}"] = float(
                                d1p[bi, mask].mean()
                            )
                        else:
                            er[f"mu1_cls_source_mean_on_{role.lower()}"] = float("nan")
                            er[f"mu1_delta_mean_on_{role.lower()}"] = float("nan")

                    early_rows.append(er)

            x = x_post
            base.clear_attn_cache(blk)
            del probs, vv, attn_out, ln1, x_attn, ln2, fc, gelu

        del images, x
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()


# =============================================================================
# Summaries / plots
# =============================================================================

def summarize_focus(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "mode", "block", "head"]
    num = [c for c in df.columns if c not in keys + ["stim_id", "source"]]
    rows = []
    for k, q in df.groupby(keys):
        r = dict(zip(keys, k))
        r["n_images"] = len(q)
        for c in num:
            vals = pd.to_numeric(q[c], errors="coerce")
            r[c + "_mean"] = float(vals.mean())
            r[c + "_median"] = qnan(vals)
        rows.append(r)
    return pd.DataFrame(rows)


def summarize_early(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "mode", "block"]
    num = [c for c in df.columns if c not in keys + ["stim_id", "source"]]
    rows = []
    for k, q in df.groupby(keys):
        r = dict(zip(keys, k))
        r["n_images"] = len(q)
        for c in num:
            vals = pd.to_numeric(q[c], errors="coerce")
            r[c + "_mean"] = float(vals.mean())
            r[c + "_median"] = qnan(vals)
        rows.append(r)
    return pd.DataFrame(rows)


def summarize_state(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_name", "mode", "block"]
    rows = []
    for k, q in df.groupby(keys):
        r = dict(zip(keys, k))
        r["n_images"] = len(q)
        for c in [x for x in df.columns if x.startswith("cls_")]:
            vals = pd.to_numeric(q[c], errors="coerce")
            r[c + "_mean"] = float(vals.mean())
            r[c + "_median"] = qnan(vals)
        rows.append(r)
    return pd.DataFrame(rows)


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_cls_to_b13reg(focus: pd.DataFrame, pdir: Path):
    q = focus[
        (focus["mode"] == MODE_INTACT)
        & (focus["head"].isin([5, 11, 12, 14, 15]))
        & (focus["block"].between(10, 13))
    ]
    if not len(q):
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        z = q[q["model_name"] == model]
        for h in [5, 11, 12, 14, 15]:
            y = z[z["head"] == h].sort_values("block")
            if len(y):
                ax.plot(
                    y["block"],
                    y.cls_gather_from_b13_reg_attn_mean,
                    marker="o",
                    label=f"H{h}",
                )
        ax.set_title(model)
        ax.set_xlabel("Block")
        ax.set_xticks([10, 11, 12, 13])
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Q=CLS attention mass to intact B13 register set")
    axes[-1].legend(ncol=2, fontsize=8)
    fig.suptitle("CLS participation in the B12/B13 register read")
    fig.tight_layout(rect=[0, 0, 1, .94])
    savefig(fig, pdir / "01_CLS_to_B13reg_focus_heads.png")


def plot_b20_h14(focus: pd.DataFrame, pdir: Path):
    q = focus[(focus["block"] == 20) & (focus["head"] == 14)]
    if not len(q):
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharey="row")
    for ci, model in enumerate(MODEL_ORDER):
        for ri, mode in enumerate([MODE_INTACT, MODE_NOPUMP]):
            z = q[(q["model_name"] == model) & (q["mode"] == mode)]
            if not len(z):
                continue
            r = z.iloc[0]
            roles = list(ROLE_NAMES)
            gather = [r.get(f"cls_gather_from_{x.lower()}_attn_mean", np.nan) for x in roles]
            broad = [r.get(f"cls_broadcast_to_{x.lower()}_attn_mean", np.nan) for x in roles]
            x = np.arange(len(roles))
            w = .36
            ax = axes[ri, ci]
            ax.bar(x - w/2, gather, width=w, label="CLS query -> role")
            ax.bar(x + w/2, broad, width=w, label="role query -> CLS")
            ax.set_xticks(x)
            ax.set_xticklabels(["REG", "CACHE", "SCRATCH*", "OTHER"], rotation=20)
            ax.set_title(f"{model} / {mode}")
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Mean attention mass")
    axes[1, 0].set_ylabel("Mean attention mass")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("B20 H14 bidirectional CLS <-> GIPU exchange (*scratch proxy)")
    fig.tight_layout(rect=[0, 0, 1, .95])
    savefig(fig, pdir / "02_B20_H14_bidirectional_exchange.png")


def plot_early_mu(early: pd.DataFrame, pdir: Path):
    q = early[(early["mode"] == MODE_INTACT) & (early["block"].between(6, 13))]
    if not len(q):
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        z = q[q["model_name"] == model].sort_values("block")
        for role, label in [
            ("cls", "CLS source"),
            ("reg_lineage", "register-lineage sources"),
            ("mu_cache", "mu-cache sources"),
            ("scratch_proxy", "scratch-proxy sources"),
        ]:
            col = f"mu1_{role}_source_abs_l1_ratio_mean"
            if col in z:
                ax.plot(z["block"], z[col], marker="o", label=label)
        ax.set_title(model)
        ax.set_xlabel("Block")
        ax.set_xticks(range(6, 14))
        ax.grid(alpha=.25)
    axes[0].set_ylabel(r"$\ell_1$ source contribution / exact patch $|\Delta\mu_1|$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Who writes the early mu1 attention update?")
    fig.tight_layout(rect=[0, 0, 1, .94])
    savefig(fig, pdir / "03_early_mu1_source_fraction.png")


def plot_cls_state(state: pd.DataFrame, pdir: Path):
    q = state[state["mode"] == MODE_INTACT]
    if not len(q):
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        z = q[q["model_name"] == model].sort_values("block")
        ax.plot(z["block"], z["cls_pre_mu1_cos_mean"], marker="o", label="CLS pre / mu1")
        ax.plot(z["block"], z["cls_postattn_mu1_cos_mean"], marker="o", label="CLS post-attn / mu1")
        ax.plot(z["block"], z["cls_pre_mu2_cos_mean"], marker="s", ls="--", label="CLS pre / mu2")
        ax.plot(z["block"], z["cls_postattn_mu2_cos_mean"], marker="s", ls="--", label="CLS post-attn / mu2")
        ax.axhline(0, color=".5", lw=.8)
        ax.set_title(model)
        ax.set_xlabel("Block")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Cosine to fixed B23 register basis direction")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Does CLS itself enter the invariant register/workspace subspace?")
    fig.tight_layout(rect=[0, 0, 1, .94])
    savefig(fig, pdir / "04_CLS_mu_state_by_block.png")


def plot_h14_trajectory(focus: pd.DataFrame, pdir: Path):
    q = focus[(focus["mode"] == MODE_INTACT) & (focus["head"] == 14)]
    if not len(q):
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        z = q[q["model_name"] == model].sort_values("block")
        ax.plot(
            z["block"], z["cls_gather_from_reg_lineage_attn_mean"],
            marker="o", label="CLS -> REG",
        )
        ax.plot(
            z["block"], z["cls_gather_from_mu_cache_attn_mean"],
            marker="o", label="CLS -> CACHE",
        )
        ax.plot(
            z["block"], z["cls_broadcast_to_reg_lineage_attn_mean"],
            marker="s", ls="--", label="REG -> CLS key",
        )
        ax.plot(
            z["block"], z["cls_broadcast_to_mu_cache_attn_mean"],
            marker="s", ls="--", label="CACHE -> CLS key",
        )
        ax.set_title(model)
        ax.set_xlabel("Block")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Mean attention")
    axes[-1].legend(fontsize=8)
    fig.suptitle("H14: CLS exchange with register lineage and mu-cache")
    fig.tight_layout(rect=[0, 0, 1, .94])
    savefig(fig, pdir / "05_H14_GIPU_exchange_trajectory.png")


def plot_mu1_head_heatmaps(allh: pd.DataFrame, pdir: Path):
    # Heatmap: mean total head mu1 write into ordinary patch queries.
    for model in MODEL_ORDER:
        for mode in [MODE_INTACT, MODE_NOPUMP]:
            q = allh[(allh["model_name"] == model) & (allh["mode"] == mode)]
            col = "patch_other_head_mu1_write_mean"
            if not len(q) or col not in q:
                continue
            piv = q.pivot(index="block", columns="head", values=col)
            piv = piv.reindex(index=sorted(q["block"].unique()), columns=range(16))
            arr = piv.to_numpy(float)
            vmax = max(1e-8, float(np.nanquantile(np.abs(arr), .98)))
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(
                arr,
                aspect="auto",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                origin="upper",
            )
            ax.set_xticks(range(16))
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels([f"B{x}" for x in piv.index])
            ax.set_xlabel("Head")
            ax.set_ylabel("Block")
            ax.set_title(f"{model} / {mode}: head write into mu1 on PATCH_OTHER queries")
            fig.colorbar(im, ax=ax, label="Mean signed residual contribution")
            savefig(fig, pdir / f"06_mu1_head_write_heatmap_{model}_{mode}.png")


def plot_no_pump_h14(focus: pd.DataFrame, pdir: Path):
    q = focus[(focus["block"] == 20) & (focus["head"] == 14)]
    if not len(q) or MODE_NOPUMP not in set(q["mode"]):
        return
    rows = []
    metrics = [
        ("cls_gather_from_b13_reg_attn_mean", "CLS -> B13 reg"),
        ("cls_gather_from_reg_lineage_attn_mean", "CLS -> B23-reg lineage"),
        ("cls_broadcast_to_reg_lineage_attn_mean", "REG query -> CLS"),
        ("cls_broadcast_to_mu_cache_attn_mean", "CACHE query -> CLS"),
    ]
    for model in MODEL_ORDER:
        qi = q[(q["model_name"] == model) & (q["mode"] == MODE_INTACT)]
        qn = q[(q["model_name"] == model) & (q["mode"] == MODE_NOPUMP)]
        if not len(qi) or not len(qn):
            continue
        for col, label in metrics:
            rows.append({
                "model": model,
                "metric": label,
                "delta": float(qn.iloc[0].get(col, np.nan) - qi.iloc[0].get(col, np.nan)),
            })
    d = pd.DataFrame(rows)
    if not len(d):
        return
    labels = [x[1] for x in metrics]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(labels))
    w = .25
    for mi, model in enumerate(MODEL_ORDER):
        z = d[d["model"] == model].set_index("metric").reindex(labels)
        ax.bar(x + (mi - 1) * w, z.delta, width=w, label=model)
    ax.axhline(0, color=".4", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("no-pump minus intact attention")
    ax.set_title("B20 H14: how pump ablation redistributes the CLS exchange")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    savefig(fig, pdir / "07_no_pump_H14_delta.png")


def build_report(
    out: Path,
    focus: pd.DataFrame,
    early: pd.DataFrame,
    state: pd.DataFrame,
    allh: pd.DataFrame,
    args,
):
    lines = [
        "# CLS <-> GIPU exchange, no RN",
        "",
        "This is a targeted follow-up to the NOP/BROADCAST sink analysis.",
        "No RN token is inserted in any condition.",
        "",
        "## Role definitions",
        "",
        "- `REG_LINEAGE`: intact B23 high-norm register addresses tracked backward.",
        "- `MU_CACHE`: non-register, low-norm tokens with signed cosine >= "
        f"{args.cache_cos_threshold:g} to fixed B23 mu1.",
        "- `SCRATCH_PROXY`: high within-image distant coherence after removing span(mu1,mu2); "
        "**not** the full cross-image-gap scratchpad detector.",
        "- `B13_REG`: intact pre-B13 visible-register addresses; non-exclusive focus probe.",
        "",
        "## Focus questions",
        "",
    ]

    def frow(model, mode, block, head):
        q = focus[
            (focus["model_name"] == model)
            & (focus["mode"] == mode)
            & (focus["block"] == block)
            & (focus["head"] == head)
        ]
        return q.iloc[0] if len(q) else None

    lines += [
        "### B12/B13: does CLS interrogate registers?",
        "",
        "| model | B12 H14 CLS→B13REG attn | B13 H5 CLS→B13REG attn | "
        "B13 H5 CLS→B13REG AV | B13 H11 CLS→B13REG attn |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        a = frow(model, MODE_INTACT, 12, 14)
        b = frow(model, MODE_INTACT, 13, 5)
        c = frow(model, MODE_INTACT, 13, 11)
        if a is None or b is None or c is None:
            continue
        lines.append(
            f"| {model} | {a.cls_gather_from_b13_reg_attn_mean:.3f} | "
            f"{b.cls_gather_from_b13_reg_attn_mean:.3f} | "
            f"{b.cls_gather_from_b13_reg_avfrac_mean:.3f} | "
            f"{c.cls_gather_from_b13_reg_attn_mean:.3f} |"
        )

    lines += ["", "### B20 H14: bidirectional exchange", ""]
    lines += [
        "| model/mode | CLS→REG attn | REG→CLS attn | CLS→CACHE attn | CACHE→CLS attn | "
        "CLS→B13REG attn |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        for mode in [MODE_INTACT, MODE_NOPUMP]:
            r = frow(model, mode, 20, 14)
            if r is None:
                continue
            lines.append(
                f"| {model}/{mode} | "
                f"{r.cls_gather_from_reg_lineage_attn_mean:.3f} | "
                f"{r.cls_broadcast_to_reg_lineage_attn_mean:.3f} | "
                f"{r.cls_gather_from_mu_cache_attn_mean:.3f} | "
                f"{r.cls_broadcast_to_mu_cache_attn_mean:.3f} | "
                f"{r.cls_gather_from_b13_reg_attn_mean:.3f} |"
            )

    lines += [
        "",
        "### Early mu1 spread",
        "",
        "The `mu1_*_source_abs_l1_ratio` columns are source-wise absolute contribution "
        "magnitudes divided by the exact patch attention-stage |Δmu1|.  They are not "
        "fractions constrained to sum to one because source contributions can cancel.",
        "",
        "| model | block | CLS source ratio | REG source ratio | CACHE source ratio | "
        "corr(Δmu1, CLS-source mu1) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        z = early[
            (early["model_name"] == model)
            & (early["mode"] == MODE_INTACT)
            & (early["block"].isin([6, 8, 10, 12, 13]))
        ].sort_values("block")
        for r in z.itertuples(index=False):
            lines.append(
                f"| {model} | {int(r.block)} | "
                f"{getattr(r, 'mu1_cls_source_abs_l1_ratio_mean', float('nan')):.3f} | "
                f"{getattr(r, 'mu1_reg_lineage_source_abs_l1_ratio_mean', float('nan')):.3f} | "
                f"{getattr(r, 'mu1_mu_cache_source_abs_l1_ratio_mean', float('nan')):.3f} | "
                f"{getattr(r, 'mu1_delta_corr_cls_source_mean', float('nan')):.3f} |"
            )

    maxerr1 = float(pd.to_numeric(
        pd.read_csv(out / "early_mu_source_decomposition_per_image.csv")["mu1_decomp_max_abs_error"],
        errors="coerce",
    ).max())
    maxerr2 = float(pd.to_numeric(
        pd.read_csv(out / "early_mu_source_decomposition_per_image.csv")["mu2_decomp_max_abs_error"],
        errors="coerce",
    ).max())
    lines += [
        "",
        "## Numerical audit",
        "",
        f"- max |mu1 source-decomposition error|: `{maxerr1:.6g}`",
        f"- max |mu2 source-decomposition error|: `{maxerr2:.6g}`",
        "",
        "## Interpretation guardrails",
        "",
        "- Hard-sink thresholds are not used for the directed CLS/GIPU flow metrics; all images contribute.",
        "- `|A*V|` uses projected V row norms and is a magnitude diagnostic, not a signed vector decomposition.",
        "- Signed mu1/mu2 source writes are an exact linear decomposition of the attention update "
        "apart from the separately included out-projection bias.",
        "- `SCRATCH_PROXY` is deliberately not called a true scratchpad: the earlier GIPU detector "
        "also subtracts cross-image coherence.",
        "- Any claim that CLS causes B20 broadcast still requires an edge/head intervention; this "
        "script establishes routing geometry and pump dependence.",
        "",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="No-RN CLS/GIPU directed-flow and mu-write probe."
    )

    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--models", default=",".join(MODEL_ORDER))
    ap.add_argument("--blocks", default=DEFAULT_BLOCKS)
    ap.add_argument("--focus_heads", default=DEFAULT_FOCUS_HEADS)
    ap.add_argument("--focus_blocks", default=DEFAULT_FOCUS_BLOCKS)
    ap.add_argument("--include_no_pump", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument(
        "--postprocess_only",
        action="store_true",
        help="Reuse completed CSV/NPZ outputs and only rebuild summaries, plots, report, and handoff ZIP.",
    )

    # GIPU role definitions.
    ap.add_argument("--final_register_threshold", type=float, default=60.0)
    ap.add_argument("--final_register_min", type=int, default=1)
    ap.add_argument("--final_register_max", type=int, default=4)
    ap.add_argument("--b13_register_threshold", type=float, default=70.0)
    ap.add_argument("--cache_norm_max", type=float, default=60.0)
    ap.add_argument("--cache_cos_threshold", type=float, default=0.80)
    ap.add_argument("--scratch_proxy_threshold", type=float, default=0.85)
    ap.add_argument("--scratch_min_distance", type=float, default=8.0)

    # Loader args mirrored from the supplied sink script.
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip_module", default="attnclip_mechinterp_sae")
    ap.add_argument("--model_spec", default="ViT-L/14")
    ap.add_argument(
        "--gmp_checkpoint",
        default=r"ViT-L-14-BEST-smooth-GmP-ft-pickle-OpenAI.pt",
    )
    ap.add_argument(
        "--xattn_checkpoint",
        default="REPLACE_WITH_CHECKPOINT.pt",
    )

    ap.add_argument("--xattn_module", default="oaiclip")
    ap.add_argument("--pickle_module", default="clip")
    ap.add_argument("--seed", type=int, default=20260908)

    args = ap.parse_args()
    args.models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    args.blocks = parse_ints(args.blocks)
    args.focus_heads = parse_ints(args.focus_heads)
    args.focus_blocks = parse_ints(args.focus_blocks)
    unknown = set(args.models) - set(MODEL_ORDER)
    if unknown:
        ap.error(f"unknown --models: {sorted(unknown)}")
    return args


def postprocess_existing_outputs(out: Path, args) -> None:
    """Rebuild summaries, plots, report, and handoff ZIP from completed extraction CSVs."""
    pdir = out / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    required = {
        "allh": out / "all_head_flow_summary.csv",
        "focus": out / "focus_head_per_image.csv",
        "state": out / "cls_state_per_image.csv",
        "early": out / "early_mu_source_decomposition_per_image.csv",
        "roles": out / "role_counts_per_image.csv",
    }
    missing = [str(p) for p in required.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "--postprocess_only requires the completed extraction CSVs. Missing:\n  "
            + "\n  ".join(missing)
        )

    print("[postprocess] loading completed extraction outputs...")
    allh = pd.read_csv(required["allh"])
    focus_df = pd.read_csv(required["focus"])
    state_df = pd.read_csv(required["state"])
    early_df = pd.read_csv(required["early"])
    _roles_df = pd.read_csv(required["roles"])  # existence/content audit; no recomputation needed

    print("[postprocess] rebuilding summaries...")
    focus_s = summarize_focus(focus_df)
    early_s = summarize_early(early_df)
    state_s = summarize_state(state_df)

    focus_s.to_csv(out / "focus_summary.csv", index=False)
    early_s.to_csv(out / "early_mu_summary.csv", index=False)
    state_s.to_csv(out / "cls_state_summary.csv", index=False)

    print("[postprocess] rebuilding plots...")
    plot_cls_to_b13reg(focus_s, pdir)
    plot_b20_h14(focus_s, pdir)
    plot_early_mu(early_s, pdir)
    plot_cls_state(state_s, pdir)
    plot_h14_trajectory(focus_s, pdir)
    plot_mu1_head_heatmaps(allh, pdir)
    plot_no_pump_h14(focus_s, pdir)

    print("[postprocess] rebuilding report...")
    build_report(out, focus_s, early_s, state_s, allh, args)

    handoff = [
        out / "config.json",
        out / "all_head_flow_summary.csv",
        out / "focus_head_per_image.csv",
        out / "focus_summary.csv",
        out / "cls_state_per_image.csv",
        out / "cls_state_summary.csv",
        out / "early_mu_source_decomposition_per_image.csv",
        out / "early_mu_summary.csv",
        out / "role_counts_per_image.csv",
        out / "SUMMARY.txt",
    ]
    for model in args.models:
        handoff += [
            out / model / "mu_basis.npz",
            out / model / "mu_basis_audit.csv",
        ]
    handoff += sorted(pdir.glob("*.png"))

    zpath = out / "compact_summary_workspace_cls_register_exchange.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(
        zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7
    ) as z:
        for p in handoff:
            if p.exists():
                z.write(p, arcname=p.relative_to(out).as_posix())

    print("[postprocess] done")
    print("[compact summary]", zpath)

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.postprocess_only:
        postprocess_existing_outputs(out, args)
        return

    base = _backbone_tools

    pdir = out / "plots"
    pdir.mkdir(exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        alt = Path(__file__).resolve().parent / args.manifest
        if alt.is_file():
            manifest_path = alt
    manifest = pd.read_csv(manifest_path)
    missing = [p for p in manifest.path.astype(str) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(manifest)} manifest images missing; first={missing[0]}"
        )

    all_acc = MeanAccumulator()
    focus_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    early_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        print(f"\n================ {model_name} ================")
        model_dir = out / model_name
        model_dir.mkdir(exist_ok=True)

        bundle = base.load_bundle(model_name, args, model_dir / "load_audit")

        # First intact pass establishes fixed lineages and mu basis.
        oracle = fit_model_oracle(base, bundle, manifest, args, model_dir)

        collect_mode(
            base, bundle, manifest, oracle, model_name, MODE_INTACT, args,
            all_acc, focus_rows, state_rows, early_rows, role_rows,
        )
        if args.include_no_pump:
            collect_mode(
                base, bundle, manifest, oracle, model_name, MODE_NOPUMP, args,
                all_acc, focus_rows, state_rows, early_rows, role_rows,
            )

        del bundle, oracle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    allh = pd.DataFrame(
        all_acc.rows(["model_name", "mode", "block", "head"])
    )
    focus_df = pd.DataFrame(focus_rows)
    state_df = pd.DataFrame(state_rows)
    early_df = pd.DataFrame(early_rows)
    roles_df = pd.DataFrame(role_rows)

    allh.to_csv(out / "all_head_flow_summary.csv", index=False)
    focus_df.to_csv(out / "focus_head_per_image.csv", index=False)
    state_df.to_csv(out / "cls_state_per_image.csv", index=False)
    early_df.to_csv(out / "early_mu_source_decomposition_per_image.csv", index=False)
    roles_df.to_csv(out / "role_counts_per_image.csv", index=False)

    focus_s = summarize_focus(focus_df)
    early_s = summarize_early(early_df)
    state_s = summarize_state(state_df)
    focus_s.to_csv(out / "focus_summary.csv", index=False)
    early_s.to_csv(out / "early_mu_summary.csv", index=False)
    state_s.to_csv(out / "cls_state_summary.csv", index=False)

    plot_cls_to_b13reg(focus_s, pdir)
    plot_b20_h14(focus_s, pdir)
    plot_early_mu(early_s, pdir)
    plot_cls_state(state_s, pdir)
    plot_h14_trajectory(focus_s, pdir)
    plot_mu1_head_heatmaps(allh, pdir)
    plot_no_pump_h14(focus_s, pdir)

    build_report(out, focus_s, early_s, state_s, allh, args)

    # Compact handoff.  Per-image focus/early/state tables are included because
    # they are the statistically useful data for follow-up; no giant raw tensor dump.
    handoff = [
        out / "config.json",
        out / "all_head_flow_summary.csv",
        out / "focus_head_per_image.csv",
        out / "focus_summary.csv",
        out / "cls_state_per_image.csv",
        out / "cls_state_summary.csv",
        out / "early_mu_source_decomposition_per_image.csv",
        out / "early_mu_summary.csv",
        out / "role_counts_per_image.csv",
        out / "SUMMARY.txt",
    ]
    for model in args.models:
        handoff += [
            out / model / "mu_basis.npz",
            out / model / "mu_basis_audit.csv",
        ]
    handoff += sorted(pdir.glob("*.png"))

    zpath = out / "compact_summary_workspace_cls_register_exchange.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(
        zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7
    ) as z:
        for p in handoff:
            if p.exists():
                z.write(p, arcname=p.relative_to(out).as_posix())

    print("\n[done]", out)
    print("[compact summary]", zpath)


if __name__ == "__main__":
    main()
