#!/usr/bin/env python3
r"""CHANNEL-TAGGED NOP/BROADCAST ATLAS FOR CLIP ViT-L/14
====================================================

Purpose
-------
Extend the existing `probe_clip_nop_broadcast_sinks_v2.py` dual-algorithm
measurement with residual-backbone interventions and explicit 565/650/123 tags.

The original NOP/BROADCAST definitions are preserved exactly:

    hard sink event: strongest incoming-attention target >= 0.50

    NOP:
        target V norm / other-token mean V norm < 0.20

    BROADCAST:
        target V norm ratio >= 0.20
        AND projected head-output stable rank <= 1.10

    otherwise: OTHER

Global residual interventions
-----------------------------
At resid_pre of EVERY visual transformer block, ALL tokens:

    normal
    zero565
    zero650
    zero565_650
    zero123       # counterfactual control

Each is crossed with:

    intact
    no_pump       # exact B11/B12 post-QuickGELU register-pump ablation

Default models:

    pretrained
    gmp
    finetune_stripped

Why "channel tagged"?
----------------------
For every block/head/image sink event we record, for channels 565, 650 and 123:

1. residual/LN state
   - CLS raw residual coordinate
   - sink-target raw residual coordinate
   - CLS ln_1 coordinate
   - sink-target ln_1 coordinate

2. exact Q/K routing contribution to the CLS->sink logit
   If q = Wq x + b and k = Wk y + b, for coordinate c:

       q_side_c = < x_cls[c] Wq[:,c], k_sink > / sqrt(dh)
       k_side_c = < q_cls, y_sink[c] Wk[:,c] > / sqrt(dh)
       cross_c  = < x_cls[c] Wq[:,c], y_sink[c] Wk[:,c] > / sqrt(dh)

   and the exact change from removing c on BOTH Q and K sides is:

       delta_both_zero_c = -q_side_c - k_side_c + cross_c

3. target V / cargo write
   - coordinate c of W_O^h V_sink
   - CLS-query realized edge write A[CLS,sink] * (W_O^h V_sink)[c]

This lets us ask whether a NOP/BROADCAST/PATCH/CLS event is 565-routing-heavy,
650-cargo-heavy, generic one-coordinate behavior (123), etc., without confusing
"channel" with "sink type".

Plots
-----
The script deliberately makes a lot of plots:

* algorithm phase trajectories: PATCH-NOP vs CLS-BROADCAST through B0..B23
* per-regime rates by block
* head x block heatmaps for CLS/PATCH NOP/BROADCAST
* NOP/BROADCAST stable-rank vs V-ratio clouds
* 565/650/123 Q/K/cargo violin plots by regime and sink type
* routing/cargo channel-tag heatmaps
* zero565 vs zero650 head-level quadrant plots
* conditional CLS->patch spatial maps for every requested condition/block
* scanner/orienteering score heatmaps (KL-from-uniform, row/column anisotropy)
* residual norm curves
* Plotly Sankey regime transitions (when plotly is installed; PNG when kaleido works)

No giant Q/K activation dumps are saved.

Default output
--------------
    clip_nop_broadcast_channel_tagged

Final handoff
-------------
    compact_summary_workspace_broadcast_channel_interventions.zip
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
# -*- coding: utf-8 -*-


import argparse
import gc
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# -----------------------------------------------------------------------------
# Load the user's existing dual-algorithm base.  If the script is not in the
# project root, a bundled reference/ copy is accepted.
# -----------------------------------------------------------------------------
import probe_tools_backbone as BASE


# =============================================================================
# Constants
# =============================================================================

DEFAULT_OUT = r"clip_nop_broadcast_channel_tagged_pre_gmp_special_delivery_v2"
DEFAULT_MODELS = (BASE.MODEL_PRE, BASE.MODEL_GMP)

REGISTER_PUMP_UNITS: Dict[int, Tuple[int, ...]] = {
    11: (9, 987, 1100, 1967, 2555, 3661, 3784),
    12: (42, 183, 983, 1571, 1816, 2687, 3002, 3008, 3868),
}

TAG_CHANNELS = (565, 650, 123)
CHANNEL_CONDITIONS: Dict[str, Tuple[int, ...]] = {
    "normal": tuple(),
    "zero565": (565,),
    "zero650": (650,),
    "zero565_650": (565, 650),
    "zero123": (123,),
}
PUMP_MODES = ("intact", "no_pump")

N_BLOCKS = 24
N_HEADS = 16
EPS = 1e-12
HANDOFF_ZIP = "compact_summary_workspace_broadcast_channel_interventions.zip"

# Controlled paired dataset used by default.
SPECIAL_DELIVERY_DIR = Path("image_sets/special_delivery")
SPECIAL_DELIVERY_FILES = (
    "apple.png",
    "apple_adv.png",
    "bananorange.jpg",
    "bananorange_adv.jpg",
    "bee.JPEG",
    "bee_adv.JPEG",
    "catdog.png",
    "catdog_adv.png",
    "checker_08freq.png",
    "checker_08freq_adv.png",
    "checker_16freq.png",
    "checker_16freq_adv.png",
    "checker_32freq.png",
    "checker_32freq_adv.png",
    "cup.png",
    "cup_adv.png",
    "finch1.png",
    "finch1_adv.png",
    "finch2.png",
    "finch2_adv.png",
    "fish.png",
    "fish_adv.png",
    "fractal_sine_03.png",
    "fractal_sine_03_adv.png",
    "fractal_sine_05.png",
    "fractal_sine_05_adv.png",
    "fractal_sine_08.png",
    "fractal_sine_08_adv.png",
    "gradient_03_grad_radial.png",
    "gradient_03_grad_radial_adv.png",
    "neuron1.png",
    "neuron1_adv.png",
    "neuron2.png",
    "neuron2_adv.png",
    "neuron3.png",
    "neuron3_adv.png",
    "neuron4.png",
    "neuron4_adv.png",
    "poodle.png",
    "poodle_adv.png",
    "shower.png",
    "shower_adv.png",
    "sine_04hz_000deg.png",
    "sine_04hz_000deg_adv.png",
    "sine_04hz_045deg.png",
    "sine_04hz_045deg_adv.png",
    "sine_04hz_090deg.png",
    "sine_04hz_090deg_adv.png",
    "stop.png",
    "stop_adv.png",
    "word.png",
    "word_adv.png",
)

# Plot palette is explicit because the user asked for color-coded populations.
COND_COLORS = {
    "normal": "black",
    "zero565": "tab:blue",
    "zero650": "tab:red",
    "zero565_650": "tab:purple",
    "zero123": "tab:gray",
}
PUMP_LINESTYLES = {"intact": "-", "no_pump": "--"}
REGIME_COLORS = {
    "NOP": "tab:orange",
    "BROADCAST": "tab:green",
    "OTHER": "tab:blue",
    "NO_SINK": "tab:gray",
}


# =============================================================================
# Utilities
# =============================================================================

def safe_name(x: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(x))


def mode_tag(pump_mode: str, channel_condition: str) -> str:
    return f"{pump_mode}__{channel_condition}"


def clamp_residual_channels(x: torch.Tensor, channels: Sequence[int]) -> torch.Tensor:
    if not channels:
        return x
    out = x.clone()
    idx = torch.as_tensor(list(map(int, channels)), device=out.device, dtype=torch.long)
    if int(idx.max()) >= int(out.shape[-1]):
        raise IndexError(f"Residual width {out.shape[-1]} too small for channels {list(channels)}")
    out.index_fill_(-1, idx, 0)
    return out


def apply_pump(gelu: torch.Tensor, block: int, pump_mode: str) -> torch.Tensor:
    if pump_mode != "no_pump" or block not in REGISTER_PUMP_UNITS:
        return gelu
    idx = torch.as_tensor(REGISTER_PUMP_UNITS[block], device=gelu.device, dtype=torch.long)
    out = gelu.clone()
    out.index_fill_(-1, idx, 0)
    return out


def patch_grid_shape(n_patches: int) -> Optional[Tuple[int, int]]:
    s = int(round(math.sqrt(n_patches)))
    return (s, s) if s * s == n_patches else None


def entropy_norm(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.asarray(p, np.float64)
    n = p.shape[axis]
    q = np.clip(p, 1e-30, None)
    h = -(q * np.log(q)).sum(axis=axis)
    return h / max(math.log(max(n, 2)), EPS)


def scanner_scores_from_mean_map(mean_map: np.ndarray) -> Dict[str, float]:
    """Positional/orienteering scores from a mean conditional CLS->patch map."""
    p = np.asarray(mean_map, np.float64).reshape(-1)
    p = p / max(p.sum(), EPS)
    u = np.full_like(p, 1.0 / len(p))
    kl = float(np.sum(p * np.log(np.clip(p / u, 1e-30, None))))
    ent = float(entropy_norm(p[None, :])[0])
    grid = patch_grid_shape(len(p))
    row_kl = col_kl = anis = np.nan
    if grid is not None:
        g = p.reshape(grid)
        r = g.sum(axis=1); c = g.sum(axis=0)
        ur = np.full_like(r, 1.0 / len(r)); uc = np.full_like(c, 1.0 / len(c))
        row_kl = float(np.sum(r * np.log(np.clip(r / ur, 1e-30, None))))
        col_kl = float(np.sum(c * np.log(np.clip(c / uc, 1e-30, None))))
        anis = float(abs(row_kl - col_kl))
    return {
        "scanner_kl_from_uniform": kl,
        "scanner_entropy_norm": ent,
        "scanner_row_kl": row_kl,
        "scanner_col_kl": col_kl,
        "scanner_row_col_anisotropy": anis,
    }


def qnan(s: pd.Series, q: float = .5) -> float:
    a = pd.to_numeric(s, errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if len(a) else np.nan


def route_tag_values(
    blk,
    pre: torch.Tensor,
    ln1: torch.Tensor,
    qq: torch.Tensor,
    kk: torch.Tensor,
    vv: torch.Tensor,
    probs: torch.Tensor,
    bi: int,
    h: int,
    si: int,
    dh: int,
) -> Dict[str, float]:
    """Exact per-channel routing/cargo tags for one CLS->sink edge."""
    out: Dict[str, float] = {}
    q_cls = qq[bi, h, 0].float()
    k_sink = kk[bi, h, si].float()
    v_sink = vv[bi, h, si].float()

    Wq = blk.attn.q_proj.weight.detach().float()
    Wk = blk.attn.k_proj.weight.detach().float()
    Wout = blk.attn.out_proj.weight.detach().float()
    sl = slice(h * dh, (h + 1) * dh)
    Wh = Wout[:, sl]
    sink_write = Wh @ v_sink
    sink_write_norm = float(sink_write.norm().clamp_min(EPS))
    cls_attn = float(probs[bi, h, 0, si])
    scale = math.sqrt(dh)

    route_abs = {}
    cargo_abs = {}

    for c in TAG_CHANNELS:
        xq = float(ln1[0, bi, c])
        xk = float(ln1[si, bi, c])
        dq = Wq[sl, c] * xq
        dk = Wk[sl, c] * xk
        qside = float(torch.dot(dq, k_sink) / scale)
        kside = float(torch.dot(q_cls, dk) / scale)
        cross = float(torch.dot(dq, dk) / scale)
        delta_both = -qside - kside + cross

        # The sink classifier itself is based on incoming attention averaged over
        # ALL queries, not only CLS.  Tag that algorithmic routing geometry too.
        xq_all = ln1[:, bi, c].float()                                  # [T]
        dq_all = xq_all[:, None] * Wq[sl, c].float()[None, :]          # [T,dh]
        qside_mean = float((dq_all * k_sink[None, :]).sum(-1).mean() / scale)
        kside_mean = float((qq[bi, h].float() * dk[None, :]).sum(-1).mean() / scale)
        cross_mean = float((dq_all * dk[None, :]).sum(-1).mean() / scale)
        delta_both_mean = -qside_mean - kside_mean + cross_mean

        cargo = float(sink_write[c])
        edge_cargo = cls_attn * cargo

        out[f"cls_raw_ch{c}"] = float(pre[0, bi, c])
        out[f"sink_raw_ch{c}"] = float(pre[si, bi, c])
        out[f"cls_ln1_ch{c}"] = xq
        out[f"sink_ln1_ch{c}"] = xk
        out[f"qside_logit_ch{c}"] = qside
        out[f"kside_logit_ch{c}"] = kside
        out[f"qk_cross_ch{c}"] = cross
        out[f"qk_bothzero_delta_ch{c}"] = delta_both
        out[f"meanq_qside_logit_ch{c}"] = qside_mean
        out[f"meanq_kside_logit_ch{c}"] = kside_mean
        out[f"meanq_qk_cross_ch{c}"] = cross_mean
        out[f"meanq_qk_bothzero_delta_ch{c}"] = delta_both_mean
        out[f"sink_vwrite_ch{c}"] = cargo
        out[f"sink_vwrite_absfrac_ch{c}"] = abs(cargo) / sink_write_norm
        out[f"cls_edge_write_ch{c}"] = edge_cargo

        route_abs[c] = abs(delta_both)
        cargo_abs[c] = abs(cargo)

    cls_route_abs = dict(route_abs)
    mean_route_abs = {c: abs(out[f"meanq_qk_bothzero_delta_ch{c}"]) for c in TAG_CHANNELS}
    out["cls_routing_dominant_channel"] = int(max(cls_route_abs, key=cls_route_abs.get))
    out["sink_routing_dominant_channel"] = int(max(mean_route_abs, key=mean_route_abs.get))
    # Backward-friendly alias: 'routing' without qualifier means sink-algorithm / mean-query.
    out["routing_dominant_channel"] = out["sink_routing_dominant_channel"]
    out["cargo_dominant_channel"] = int(max(cargo_abs, key=cargo_abs.get))
    out["cls_routing_565_vs_123"] = cls_route_abs[565] / max(cls_route_abs[123], EPS)
    out["cls_routing_650_vs_123"] = cls_route_abs[650] / max(cls_route_abs[123], EPS)
    out["routing_565_vs_123"] = mean_route_abs[565] / max(mean_route_abs[123], EPS)
    out["routing_650_vs_123"] = mean_route_abs[650] / max(mean_route_abs[123], EPS)
    out["cargo_565_vs_123"] = cargo_abs[565] / max(cargo_abs[123], EPS)
    out["cargo_650_vs_123"] = cargo_abs[650] / max(cargo_abs[123], EPS)
    return out


# =============================================================================
# Tagged extraction
# =============================================================================

@torch.no_grad()
def collect_batch_tagged(
    bundle,
    images: torch.Tensor,
    ids: Sequence[str],
    sources: Sequence[str],
    model_name: str,
    pump_mode: str,
    channel_condition: str,
    args,
    seed_oracle,
    reference_refs: Optional[Mapping[int, Mapping[str, torch.Tensor]]] = None,
):
    v = bundle.model.visual
    channels = CHANNEL_CONDITIONS[channel_condition]
    x = v._prepare_tokens(images.to(bundle.device, dtype=bundle.model.dtype))
    B = int(images.shape[0])
    H = int(v.transformer.resblocks[0].attn.num_heads)
    T = int(x.shape[0])
    P = T - 1
    dh = int(v.transformer.resblocks[0].attn.embed_dim // H)
    tag = mode_tag(pump_mode, channel_condition)

    b13_regs = [set(s) for s in seed_oracle.b13_regs]
    primary_reg = list(seed_oracle.b13_primary_reg)
    probe_patch = list(seed_oracle.probe_patch)
    intact_b20 = [set(s) for s in seed_oracle.b20_newnorm] if seed_oracle.b20_newnorm else [set() for _ in range(B)]
    local_b20 = [set() for _ in range(B)]

    head_rows: List[dict] = []
    layer_rows: List[dict] = []
    norm_rows: List[dict] = []
    probe_rows: List[dict] = []
    refs_out: Dict[int, Dict[str, torch.Tensor]] = {}

    spatial_attn_sum = np.zeros((N_BLOCKS, H, P), np.float64)
    spatial_av_sum = np.zeros((N_BLOCKS, H, P), np.float64)
    spatial_count = np.zeros((N_BLOCKS, H), np.int64)

    for b, blk in enumerate(v.transformer.resblocks):
        # Global residual intervention: every block sees the selected coordinate(s)
        # clamped before ln_1 and before the residual skip connection of this block.
        x = clamp_residual_channels(x, channels)
        pre = x
        ln1 = blk.ln_1(x)
        attn_out, probs0 = blk.attention(ln1, need_weights=True, capture=True)
        probs = BASE.normalize_probs_shape(probs0, B, H, T).float()
        vv = BASE.normalize_qkv_shape(blk.attn.last_v, B, H, T).float()
        kk = BASE.normalize_qkv_shape(blk.attn.last_k, B, H, T).float()
        qq = BASE.normalize_qkv_shape(blk.attn.last_q, B, H, T).float()
        Wout = blk.attn.out_proj.weight.detach().float()

        raw_norm = pre.float().norm(dim=-1).T
        ln_norm = ln1.float().norm(dim=-1).T
        hin = probs.mean(dim=2)  # incoming mass per key [B,H,T]
        strengths, sidx = hin.max(dim=-1)

        vnorm = vv.norm(dim=-1)
        vproj = torch.empty_like(vnorm)
        for h in range(H):
            vproj[:, h] = BASE.projected_row_norms_batch(vv[:, h], Wout, h)

        cls_a = probs[:, :, 0, :]
        cls_av_r = cls_a * vproj
        avr_den = cls_av_r.sum(dim=-1).clamp_min(1e-12)

        # Conditional CLS->patch map (user's scanner/orienteering view).
        pa = cls_a[:, :, 1:]
        pav = cls_av_r[:, :, 1:]
        pa_cond = pa / pa.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pav_cond = pav / pav.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        spatial_attn_sum[b] += pa_cond.sum(dim=0).cpu().numpy()
        spatial_av_sum[b] += pav_cond.sum(dim=0).cpu().numpy()
        spatial_count[b] += B

        # Head-mean transition target.
        pmean = probs.mean(dim=1)
        inflow = pmean.mean(dim=1)
        layer_strength, layer_idx = inflow.max(dim=-1)
        for bi in range(B):
            si = int(layer_idx[bi])
            typ = "CLS" if si == 0 else "PATCH"
            patch = si - 1
            layer_rows.append({
                "model_name": model_name,
                "mode": tag,
                "pump_mode": pump_mode,
                "channel_condition": channel_condition,
                "stim_id": ids[bi],
                "source": sources[bi],
                "block": b,
                "sink_idx": si,
                "sink_patch_idx": patch if si > 0 else -1,
                "sink_type": typ,
                "sink_strength": float(layer_strength[bi]),
                "sink_raw_norm": float(raw_norm[bi, si]),
                "sink_ln1_norm": float(ln_norm[bi, si]),
                "is_b13_register": int(si > 0 and patch in b13_regs[bi]),
            })

        # Fixed baseline probe address for residual/QKV drift.
        raw_probe = torch.stack([pre[p + 1, bi].float() for bi, p in enumerate(probe_patch)], dim=0)
        ln_probe = torch.stack([ln1[p + 1, bi].float() for bi, p in enumerate(probe_patch)], dim=0)
        q_probe = torch.stack([qq[bi, :, p + 1] for bi, p in enumerate(probe_patch)], dim=0)
        k_probe = torch.stack([kk[bi, :, p + 1] for bi, p in enumerate(probe_patch)], dim=0)
        v_probe = torch.stack([vv[bi, :, p + 1] for bi, p in enumerate(probe_patch)], dim=0)

        if reference_refs is None:
            refs_out[b] = {
                "raw": raw_probe.detach().cpu(),
                "ln": ln_probe.detach().cpu(),
                "q": q_probe.detach().cpu(),
                "k": k_probe.detach().cpu(),
                "v": v_probe.detach().cpu(),
            }
            raw_cos = torch.ones(B)
            ln_cos = torch.ones(B)
            q_cos = torch.ones(B, H)
            k_cos = torch.ones(B, H)
            v_cos = torch.ones(B, H)
        else:
            ref = reference_refs[b]
            raw_cos = F.cosine_similarity(raw_probe.cpu(), ref["raw"].float(), dim=-1, eps=1e-8)
            ln_cos = F.cosine_similarity(ln_probe.cpu(), ref["ln"].float(), dim=-1, eps=1e-8)
            q_cos = F.cosine_similarity(q_probe.cpu(), ref["q"].float(), dim=-1, eps=1e-8)
            k_cos = F.cosine_similarity(k_probe.cpu(), ref["k"].float(), dim=-1, eps=1e-8)
            v_cos = F.cosine_similarity(v_probe.cpu(), ref["v"].float(), dim=-1, eps=1e-8)

        for bi in range(B):
            pi = probe_patch[bi] + 1
            probe_rows.append({
                "model_name": model_name,
                "mode": tag,
                "pump_mode": pump_mode,
                "channel_condition": channel_condition,
                "stim_id": ids[bi],
                "source": sources[bi],
                "block": b,
                "probe_patch_idx": probe_patch[bi],
                "probe_is_visible_b13_reg": int(primary_reg[bi] >= 0),
                "probe_raw_norm": float(raw_norm[bi, pi]),
                "probe_ln1_norm": float(ln_norm[bi, pi]),
                "probe_raw_cos_to_baseline": float(raw_cos[bi]),
                "probe_ln_cos_to_baseline": float(ln_cos[bi]),
                "probe_headmean_inflow": float(hin[bi, :, pi].mean()),
                "probe_headmean_cls_attention": float(cls_a[bi, :, pi].mean()),
                **{f"probe_raw_ch{c}": float(pre[pi, bi, c]) for c in TAG_CHANNELS},
                **{f"probe_ln1_ch{c}": float(ln1[pi, bi, c]) for c in TAG_CHANNELS},
            })

        # Per-head events.
        for bi in range(B):
            regs = b13_regs[bi]
            for h in range(H):
                si = int(sidx[bi, h])
                st = float(strengths[bi, h])
                is_sink = bool(st >= args.sink_threshold)
                typ = "CLS" if si == 0 else "PATCH"
                patch = si - 1

                arow = cls_a[bi, h]
                avr = cls_av_r[bi, h]
                cls_prob = arow / arow.sum().clamp_min(EPS)
                cls_ent = float(-(cls_prob.clamp_min(1e-30) * cls_prob.clamp_min(1e-30).log()).sum() / math.log(T))
                patch_mass = float(arow[1:].sum())
                patch_cond = arow[1:] / arow[1:].sum().clamp_min(EPS)
                patch_ent = float(-(patch_cond.clamp_min(1e-30) * patch_cond.clamp_min(1e-30).log()).sum() / math.log(P))
                patch_to_cls = float(probs[bi, h, 1:, 0].mean())

                row = {
                    "model_name": model_name,
                    "mode": tag,
                    "pump_mode": pump_mode,
                    "channel_condition": channel_condition,
                    "stim_id": ids[bi],
                    "source": sources[bi],
                    "block": b,
                    "head": h,
                    "sink_idx": si,
                    "sink_patch_idx": patch if si > 0 else -1,
                    "sink_type": typ,
                    "sink_strength": st,
                    "is_sink_event": int(is_sink),
                    "sink_raw_norm": float(raw_norm[bi, si]),
                    "sink_raw_norm_ratio": float(raw_norm[bi, si] / raw_norm[bi].mean().clamp_min(EPS)),
                    "sink_ln1_norm": float(ln_norm[bi, si]),
                    "sink_ln1_norm_ratio": float(ln_norm[bi, si] / ln_norm[bi].mean().clamp_min(EPS)),
                    "is_b13_register": int(si > 0 and patch in regs),
                    "cls_attention_entropy_norm": cls_ent,
                    "cls_patch_conditional_entropy_norm": patch_ent,
                    "cls_attn_to_cls": float(arow[0]),
                    "cls_attn_to_patches": patch_mass,
                    "patch_to_cls_mean": patch_to_cls,
                    "cls_av_resid_to_cls_frac": float(avr[0] / avr_den[bi, h]),
                    "cls_av_resid_to_patches_frac": float(avr[1:].sum() / avr_den[bi, h]),
                    "probe_q_cos_to_baseline": float(q_cos[bi, h]),
                    "probe_k_cos_to_baseline": float(k_cos[bi, h]),
                    "probe_v_cos_to_baseline": float(v_cos[bi, h]),
                }

                if is_sink:
                    vh = vv[bi, h]
                    kh = kk[bi, h]
                    qh = qq[bi, h]
                    vn, kn, qn = vh.norm(dim=-1), kh.norm(dim=-1), qh.norm(dim=-1)
                    other = torch.ones(T, dtype=torch.bool, device=vh.device)
                    other[si] = False
                    value_ratio = float(vn[si] / vn[other].mean().clamp_min(EPS))
                    key_ratio = float(kn[si] / kn[other].mean().clamp_min(EPS))
                    query_ratio = float(qn[si] / qn[other].mean().clamp_min(EPS))
                    z = probs[bi, h] @ vh
                    sr_head = BASE.stable_rank(z)
                    sr_resid = BASE.projected_stable_rank(z, Wout, h)
                    vproj_one = BASE.projected_row_norms(vh, Wout, h)
                    value_ratio_resid = float(vproj_one[si] / vproj_one[other].mean().clamp_min(EPS))
                    zproj = BASE.projected_row_norms(z, Wout, h)
                    update_ratio = float(zproj.mean() / raw_norm[bi].mean().clamp_min(EPS))
                    regime = BASE.event_regime(value_ratio, sr_resid)
                    row.update({
                        "value_norm_ratio": value_ratio,
                        "value_norm_ratio_residual": value_ratio_resid,
                        "key_norm_ratio": key_ratio,
                        "query_norm_ratio": query_ratio,
                        "stable_rank_head": sr_head,
                        "stable_rank_residual": sr_resid,
                        "head_update_to_resid_ratio": update_ratio,
                        "regime": regime,
                    })
                else:
                    row.update({
                        "value_norm_ratio": np.nan,
                        "value_norm_ratio_residual": np.nan,
                        "key_norm_ratio": np.nan,
                        "query_norm_ratio": np.nan,
                        "stable_rank_head": np.nan,
                        "stable_rank_residual": np.nan,
                        "head_update_to_resid_ratio": np.nan,
                        "regime": "NO_SINK",
                    })

                row.update(route_tag_values(blk, pre, ln1, qq, kk, vv, probs, bi, h, si, dh))
                head_rows.append(row)

        # Standard block continuation.
        x_attn = x + attn_out
        ln2 = blk.ln_2(x_attn)
        fc = blk.mlp.c_fc(ln2)
        gelu = blk.mlp.gelu(fc)
        gelu = apply_pump(gelu, b, pump_mode)
        x = x_attn + blk.mlp.c_proj(gelu)

        if b == BASE.B20:
            postn = x[1:].float().norm(dim=-1).T
            local_b20 = []
            for bi in range(B):
                idx = set(torch.nonzero(postn[bi] >= args.b20_readout_threshold, as_tuple=False).flatten().cpu().tolist())
                local_b20.append(idx - b13_regs[bi])
            if pump_mode == "intact" and channel_condition == "normal":
                intact_b20 = [set(s) for s in local_b20]

        n = x[1:].float().norm(dim=-1).T
        for bi in range(B):
            norm_rows.append({
                "model_name": model_name,
                "mode": tag,
                "pump_mode": pump_mode,
                "channel_condition": channel_condition,
                "stim_id": ids[bi],
                "source": sources[bi],
                "block": b,
                "spatial_norm_max_post": float(n[bi].max()),
                "spatial_norm_p95_post": float(torch.quantile(n[bi], .95)),
                "spatial_norm_mean_post": float(n[bi].mean()),
                "spatial_norm_median_post": float(n[bi].median()),
                "count_gt20_post": int((n[bi] > 20).sum()),
                "count_gt40_post": int((n[bi] > 40).sum()),
                "count_gt70_post": int((n[bi] > 70).sum()),
                **{f"spatial_ch{c}_absmean_post": float(x[1:, :, c].float().abs().mean()) for c in TAG_CHANNELS},
                **{f"cls_ch{c}_post": float(x[0, bi, c]) for c in TAG_CHANNELS},
            })

        BASE.clear_attn_cache(blk)

    spatial = {
        "attn_sum": spatial_attn_sum,
        "av_sum": spatial_av_sum,
        "count": spatial_count,
    }
    return head_rows, layer_rows, norm_rows, probe_rows, refs_out, spatial


def build_special_delivery_manifest(image_dir: Path, out: Path) -> Path:
    """Build the exact 12-image clean/adv manifest locally; no synthetic stimuli."""
    rows = []
    for fn in SPECIAL_DELIVERY_FILES:
        p = image_dir / fn
        if not p.is_file():
            raise FileNotFoundError(f"Missing SPECIAL DELIVERY image: {p}")
        stem = Path(fn).stem
        is_adv = stem.endswith("_adv")
        pair = stem[:-4] if is_adv else stem
        rows.append({
            "stim_id": stem,
            "source": "adv" if is_adv else "clean",
            "pair": pair,
            "is_adv": int(is_adv),
            "path": str(p),
        })
    mpath = out / "special_delivery_manifest.csv"
    pd.DataFrame(rows).to_csv(mpath, index=False)
    return mpath


# =============================================================================
# Extraction orchestration
# =============================================================================

def append_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def save_spatial_npz(path: Path, sums: Mapping[str, np.ndarray]) -> None:
    count = sums["count"].astype(np.float64)
    denom = np.maximum(count[..., None], 1.0)
    np.savez_compressed(
        path,
        attn_mean=sums["attn_sum"] / denom,
        av_mean=sums["av_sum"] / denom,
        count=sums["count"],
    )


def run_extraction(args, out: Path, model_name: str) -> None:
    model_dir = out / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[model] loading {model_name}")
    bundle = BASE.load_bundle(model_name, args, model_dir / "load_audit")

    manifest = pd.read_csv(args.manifest)
    missing = [p for p in manifest.path.astype(str) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(manifest)} manifest images missing; first: {missing[0]}")

    paths = {
        "head": out / f"channel_tagged_head_events_{model_name}.csv",
        "layer": out / f"channel_tagged_layer_events_{model_name}.csv",
        "norm": out / f"channel_tagged_norm_rows_{model_name}.csv",
        "probe": out / f"channel_tagged_probe_rows_{model_name}.csv",
    }
    if args.restart:
        for p in paths.values():
            if p.exists():
                p.unlink()

    spatial_totals: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    all_tags = [mode_tag(pm, cc) for pm in PUMP_MODES for cc in CHANNEL_CONDITIONS]
    for tag in all_tags:
        spatial_totals[tag] = {"attn_sum": None, "av_sum": None, "count": None}

    for st in range(0, len(manifest), args.batch_size):
        q = manifest.iloc[st:st + args.batch_size]
        tensors, ids, sources = [], [], []
        for r in q.itertuples(index=False):
            with Image.open(r.path) as im:
                tensors.append(bundle.preprocess(im.convert("RGB")))
            ids.append(str(r.stim_id))
            sources.append(str(r.source))
        batch = torch.stack(tensors, dim=0)

        # Always identify B13 future-register addresses from the intact/no-channel pilot.
        seed = BASE.get_b13_oracle(bundle, batch, args)

        # Reference run FIRST; all other residual/QKV cosines compare to it.
        h, l, n, p, refs, spatial = collect_batch_tagged(
            bundle, batch, ids, sources, model_name,
            "intact", "normal", args, seed, reference_refs=None,
        )
        tag0 = mode_tag("intact", "normal")
        append_csv(paths["head"], h); append_csv(paths["layer"], l)
        append_csv(paths["norm"], n); append_csv(paths["probe"], p)
        for k in ("attn_sum", "av_sum", "count"):
            spatial_totals[tag0][k] = spatial[k].copy() if spatial_totals[tag0][k] is None else spatial_totals[tag0][k] + spatial[k]
        del h, l, n, p, spatial

        for pump_mode in PUMP_MODES:
            for cc in CHANNEL_CONDITIONS:
                if pump_mode == "intact" and cc == "normal":
                    continue
                tag = mode_tag(pump_mode, cc)
                h, l, n, p, _refs_unused, spatial = collect_batch_tagged(
                    bundle, batch, ids, sources, model_name,
                    pump_mode, cc, args, seed, reference_refs=refs,
                )
                append_csv(paths["head"], h); append_csv(paths["layer"], l)
                append_csv(paths["norm"], n); append_csv(paths["probe"], p)
                for k in ("attn_sum", "av_sum", "count"):
                    spatial_totals[tag][k] = spatial[k].copy() if spatial_totals[tag][k] is None else spatial_totals[tag][k] + spatial[k]
                del h, l, n, p, spatial, _refs_unused
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"[{model_name}] {min(st + len(q), len(manifest))}/{len(manifest)}")
        del refs, seed, batch, tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for tag, sums in spatial_totals.items():
        if sums["attn_sum"] is not None:
            save_spatial_npz(out / f"cls_query_spatial_{model_name}_{tag}.npz", sums)

    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Summaries
# =============================================================================

def head_summary(head: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_name", "pump_mode", "channel_condition", "mode", "block", "head"]
    metric_cols = [
        "sink_strength", "value_norm_ratio", "value_norm_ratio_residual",
        "stable_rank_residual", "key_norm_ratio", "query_norm_ratio",
        "cls_attention_entropy_norm", "cls_patch_conditional_entropy_norm",
        "cls_attn_to_cls", "cls_attn_to_patches", "patch_to_cls_mean",
        "cls_av_resid_to_cls_frac", "cls_av_resid_to_patches_frac",
        "routing_565_vs_123", "routing_650_vs_123", "cargo_565_vs_123", "cargo_650_vs_123",
    ]
    for c in TAG_CHANNELS:
        metric_cols += [
            f"qside_logit_ch{c}", f"kside_logit_ch{c}", f"qk_bothzero_delta_ch{c}",
            f"meanq_qside_logit_ch{c}", f"meanq_kside_logit_ch{c}", f"meanq_qk_bothzero_delta_ch{c}",
            f"sink_vwrite_ch{c}", f"sink_vwrite_absfrac_ch{c}", f"cls_edge_write_ch{c}",
        ]

    for key, q in head.groupby(keys, sort=False):
        rec = dict(zip(keys, key))
        rec["n_images"] = len(q)
        rec["sink_rate"] = float(q.is_sink_event.mean())
        for typ in ("CLS", "PATCH"):
            rec[f"{typ.lower()}_sink_rate"] = float(((q.is_sink_event == 1) & (q.sink_type == typ)).mean())
            rec[f"{typ.lower()}_nop_rate"] = float(((q.regime == "NOP") & (q.sink_type == typ)).mean())
            rec[f"{typ.lower()}_broadcast_rate"] = float(((q.regime == "BROADCAST") & (q.sink_type == typ)).mean())
            rec[f"{typ.lower()}_other_rate"] = float(((q.regime == "OTHER") & (q.sink_type == typ)).mean())
        rec["nop_rate"] = float((q.regime == "NOP").mean())
        rec["broadcast_rate"] = float((q.regime == "BROADCAST").mean())
        rec["other_rate"] = float((q.regime == "OTHER").mean())
        for c in metric_cols:
            if c in q:
                rec[c + "_median"] = qnan(q[c])
                rec[c + "_mean"] = float(pd.to_numeric(q[c], errors="coerce").mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def block_summary(hs: pd.DataFrame) -> pd.DataFrame:
    rate_cols = [c for c in hs.columns if c.endswith("_rate")]
    metric_cols = [c for c in hs.columns if c.endswith("_median") or c.endswith("_mean")]
    rows = []
    keys = ["model_name", "pump_mode", "channel_condition", "mode", "block"]
    for key, q in hs.groupby(keys, sort=False):
        rec = dict(zip(keys, key))
        for c in rate_cols:
            rec[c] = float(q[c].mean())
        for c in metric_cols:
            rec[c] = float(pd.to_numeric(q[c], errors="coerce").mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def scanner_summary_from_npz(out: Path, models: Sequence[str], hs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in models:
        modes = hs.loc[hs.model_name.eq(model), "mode"].drop_duplicates().tolist()
        for mode in modes:
            p = out / f"cls_query_spatial_{model}_{mode}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            a = z["attn_mean"]  # [B,H,P]
            for b in range(a.shape[0]):
                for h in range(a.shape[1]):
                    s = scanner_scores_from_mean_map(a[b, h])
                    pm, cc = mode.split("__", 1)
                    rows.append({
                        "model_name": model, "mode": mode,
                        "pump_mode": pm, "channel_condition": cc,
                        "block": b, "head": h, **s,
                    })
    return pd.DataFrame(rows)


def channel_regime_summary(head: pd.DataFrame) -> pd.DataFrame:
    q = head[head.is_sink_event.eq(1)].copy()
    rows = []
    keys = ["model_name", "pump_mode", "channel_condition", "block", "sink_type", "regime"]
    metrics = []
    for c in TAG_CHANNELS:
        metrics += [
            f"qside_logit_ch{c}", f"kside_logit_ch{c}", f"qk_bothzero_delta_ch{c}",
            f"meanq_qside_logit_ch{c}", f"meanq_kside_logit_ch{c}", f"meanq_qk_bothzero_delta_ch{c}",
            f"sink_vwrite_ch{c}", f"sink_vwrite_absfrac_ch{c}", f"cls_edge_write_ch{c}",
        ]
    for key, g in q.groupby(keys, sort=False):
        rec = dict(zip(keys, key)); rec["n"] = len(g)
        for m in metrics:
            rec[m + "_median"] = qnan(g[m])
            rec[m + "_absmean"] = float(np.nanmean(np.abs(pd.to_numeric(g[m], errors="coerce"))))
        rec["routing_dom565_frac"] = float((g.routing_dominant_channel == 565).mean())
        rec["routing_dom650_frac"] = float((g.routing_dominant_channel == 650).mean())
        rec["routing_dom123_frac"] = float((g.routing_dominant_channel == 123).mean())
        rec["cls_routing_dom565_frac"] = float((g.cls_routing_dominant_channel == 565).mean())
        rec["cls_routing_dom650_frac"] = float((g.cls_routing_dominant_channel == 650).mean())
        rec["cls_routing_dom123_frac"] = float((g.cls_routing_dominant_channel == 123).mean())
        rec["cargo_dom565_frac"] = float((g.cargo_dominant_channel == 565).mean())
        rec["cargo_dom650_frac"] = float((g.cargo_dominant_channel == 650).mean())
        rec["cargo_dom123_frac"] = float((g.cargo_dominant_channel == 123).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# Plot helpers
# =============================================================================

def savefig(fig, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_phase_trajectories(bs: pd.DataFrame, out: Path) -> None:
    pdir = out / "plots" / "phase_trajectories"; pdir.mkdir(parents=True, exist_ok=True)
    for model in bs.model_name.unique():
        for pump in PUMP_MODES:
            q = bs[(bs.model_name == model) & (bs.pump_mode == pump)]
            if q.empty: continue
            fig, ax = plt.subplots(figsize=(9, 8))
            for cc, g in q.groupby("channel_condition", sort=False):
                g = g.sort_values("block")
                ax.plot(
                    g.patch_nop_rate, g.cls_broadcast_rate,
                    marker="o", ms=4, lw=1.8,
                    color=COND_COLORS.get(cc), alpha=.82, label=cc,
                )
                for _, r in g.iterrows():
                    if int(r["block"]) in (0, 1, 2, 7, 9, 11, 12, 13, 16, 18, 20, 22, 23):
                        ax.annotate(f"B{int(r['block'])}", (r["patch_nop_rate"], r["cls_broadcast_rate"]), fontsize=6, alpha=.75)
            ax.set_xlabel("PATCH-NOP rate across heads/images")
            ax.set_ylabel("CLS-BROADCAST rate across heads/images")
            ax.set_title(f"{model} / {pump}: algorithm phase trajectory B0→B23")
            ax.grid(alpha=.2); ax.legend()
            savefig(fig, pdir / f"PHASE_TRAJECTORY__{safe_name(model)}__{pump}.png")


def plot_regime_rate_curves(bs: pd.DataFrame, out: Path) -> None:
    pdir = out / "plots" / "regime_rates"; pdir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "cls_nop_rate", "cls_broadcast_rate", "patch_nop_rate", "patch_broadcast_rate",
        "nop_rate", "broadcast_rate",
    ]
    for model in bs.model_name.unique():
        for pump in PUMP_MODES:
            q = bs[(bs.model_name == model) & (bs.pump_mode == pump)]
            if q.empty: continue
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(12, 5.8))
                for cc, g in q.groupby("channel_condition", sort=False):
                    g = g.sort_values("block")
                    ax.plot(g.block, g[metric], marker="o", ms=3, color=COND_COLORS.get(cc), alpha=.82, label=cc)
                ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(-.02, 1.02)
                ax.set_xlabel("Block"); ax.set_ylabel(metric)
                ax.set_title(f"{model} / {pump}: {metric}")
                ax.grid(alpha=.2); ax.legend(ncol=3)
                savefig(fig, pdir / f"{safe_name(model)}__{pump}__{metric}.png")


def plot_head_heatmaps(hs: pd.DataFrame, out: Path) -> None:
    pdir = out / "plots" / "head_heatmaps"; pdir.mkdir(parents=True, exist_ok=True)
    metrics = ["cls_nop_rate", "cls_broadcast_rate", "patch_nop_rate", "patch_broadcast_rate"]
    for model in hs.model_name.unique():
        for pump in hs.loc[hs.model_name.eq(model), "pump_mode"].drop_duplicates().tolist():
            for cc in hs.loc[(hs.model_name.eq(model)) & (hs.pump_mode.eq(pump)), "channel_condition"].drop_duplicates().tolist():
                q = hs[(hs.model_name == model) & (hs.pump_mode == pump) & (hs.channel_condition == cc)]
                if q.empty:
                    continue
                for metric in metrics:
                    piv = q.pivot(index="head", columns="block", values=metric).reindex(index=range(N_HEADS), columns=range(N_BLOCKS))
                    arr = piv.to_numpy(dtype=float)
                    if not np.isfinite(arr).any():
                        continue
                    fig, ax = plt.subplots(figsize=(14, 7))
                    im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1, cmap="viridis")
                    ax.set_xticks(range(N_BLOCKS)); ax.set_xticklabels(range(N_BLOCKS))
                    ax.set_yticks(range(N_HEADS)); ax.set_yticklabels([f"H{i}" for i in range(N_HEADS)])
                    ax.set_xlabel("Block"); ax.set_ylabel("Head")
                    ax.set_title(f"{model} / {pump} / {cc}: {metric}")
                    cb = fig.colorbar(im, ax=ax)
                    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                    savefig(fig, pdir / f"{safe_name(model)}__{pump}__{cc}__{metric}.png")


def plot_nop_broadcast_clouds(head: pd.DataFrame, out: Path) -> None:
    pdir = out / "plots" / "nop_broadcast_clouds"; pdir.mkdir(parents=True, exist_ok=True)
    q = head.copy()
    q["value_norm_ratio"] = pd.to_numeric(q["value_norm_ratio"], errors="coerce")
    q["stable_rank_residual"] = pd.to_numeric(q["stable_rank_residual"], errors="coerce")
    q = q[np.isfinite(q["value_norm_ratio"]) & np.isfinite(q["stable_rank_residual"])].copy()
    if q.empty:
        return
    for model in q.model_name.unique():
        for pump in q.loc[q.model_name.eq(model), "pump_mode"].drop_duplicates().tolist():
            for cc in q.loc[(q.model_name.eq(model)) & (q.pump_mode.eq(pump)), "channel_condition"].drop_duplicates().tolist():
                g = q[(q.model_name == model) & (q.pump_mode == pump) & (q.channel_condition == cc)]
                if g.empty:
                    continue
                fig, ax = plt.subplots(figsize=(9, 7))
                plotted = 0
                for regime in ("NOP", "BROADCAST", "OTHER", "NO_SINK"):
                    z = g[g.regime == regime]
                    if len(z):
                        ax.scatter(z.stable_rank_residual, z.value_norm_ratio, s=10, alpha=.25, label=regime, color=REGIME_COLORS.get(regime, '#888888'))
                        plotted += len(z)
                if plotted == 0:
                    plt.close(fig)
                    continue
                ax.axhline(BASE.PAPER_NOP_VALUE_RATIO, ls="--", lw=1, color='black', alpha=.7)
                ax.axvline(BASE.PAPER_RANK1_STABLE_RANK, ls="--", lw=1, color='black', alpha=.7)
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlabel("projected stable rank")
                ax.set_ylabel("target V norm ratio")
                ax.set_title(f"{model} / {pump} / {cc}: dual-algorithm sink cloud")
                ax.grid(alpha=.15); ax.legend()
                savefig(fig, pdir / f"{safe_name(model)}__{pump}__{cc}.png")


def _violin(ax, groups: List[np.ndarray], labels: List[str], title: str, ylabel: str) -> None:
    good = [(g, l) for g, l in zip(groups, labels) if len(g)]
    if not good:
        ax.set_visible(False); return
    vals, labs = zip(*good)
    parts = ax.violinplot(vals, showmedians=True, showextrema=False)
    ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.axhline(0, lw=.8); ax.grid(axis="y", alpha=.18)


def plot_channel_violins(head: pd.DataFrame, out: Path) -> None:
    pdir = out / "plots" / "channel_tag_violins"; pdir.mkdir(parents=True, exist_ok=True)
    # Use normal channel condition so the tags describe the intact coordinate system;
    # compare intact/no-pump separately.
    q = head[(head.channel_condition == "normal") & (head.is_sink_event == 1)].copy()
    for model in q.model_name.unique():
        for pump in PUMP_MODES:
            g = q[(q.model_name == model) & (q.pump_mode == pump)]
            if g.empty: continue
            for sink_type in ("CLS", "PATCH"):
                s = g[g.sink_type == sink_type]
                fig, axes = plt.subplots(2, 3, figsize=(17, 9))
                for j, metric_base in enumerate(("qside_logit_ch", "kside_logit_ch", "sink_vwrite_ch")):
                    groups=[]; labels=[]
                    for regime in ("NOP", "BROADCAST", "OTHER"):
                        for c in TAG_CHANNELS:
                            vals = pd.to_numeric(s[s.regime == regime][f"{metric_base}{c}"], errors="coerce").dropna().to_numpy(float)
                            if len(vals) > 4000:
                                vals = vals[::max(1, len(vals)//4000)]
                            groups.append(vals); labels.append(f"{regime}\nch{c}")
                    _violin(axes[0, j], groups, labels, f"{metric_base} by regime", metric_base)

                # Absolute both-zero routing effect is especially interpretable.
                groups=[]; labels=[]
                for regime in ("NOP", "BROADCAST", "OTHER"):
                    for c in TAG_CHANNELS:
                        vals = np.abs(pd.to_numeric(s[s.regime == regime][f"qk_bothzero_delta_ch{c}"], errors="coerce").dropna().to_numpy(float))
                        groups.append(vals); labels.append(f"{regime}\nch{c}")
                _violin(axes[1,0], groups, labels, "|exact Q+K remove effect|", "|Δ QK logit|")

                for ax, ratio_col, title in [
                    (axes[1,1], "routing_565_vs_123", "routing 565/control123"),
                    (axes[1,2], "cargo_650_vs_123", "cargo 650/control123"),
                ]:
                    groups=[]; labels=[]
                    for regime in ("NOP", "BROADCAST", "OTHER"):
                        vals = pd.to_numeric(s[s.regime == regime][ratio_col], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna().to_numpy(float)
                        vals = np.clip(vals, 0, np.quantile(vals, .99) if len(vals) else 1)
                        groups.append(vals); labels.append(regime)
                    _violin(ax, groups, labels, title, "ratio")

                fig.suptitle(f"{model} / {pump} / {sink_type} sinks: residual-channel tags", y=.995)
                savefig(fig, pdir / f"{safe_name(model)}__{pump}__{sink_type}.png")


def plot_channel_tag_heatmaps(crs: pd.DataFrame, out: Path) -> None:
    pdir = out / "plots" / "channel_tag_heatmaps"; pdir.mkdir(parents=True, exist_ok=True)
    for model in crs.model_name.unique():
        for pump in PUMP_MODES:
            q = crs[(crs.model_name == model) & (crs.pump_mode == pump) & (crs.channel_condition == "normal")]
            if q.empty: continue
            for sink_type in ("CLS", "PATCH"):
                s=q[q.sink_type==sink_type]
                for metric in ("routing_dom565_frac", "routing_dom650_frac", "cargo_dom565_frac", "cargo_dom650_frac"):
                    piv=s.pivot_table(index="regime", columns="block", values=metric, aggfunc="mean").reindex(index=["NOP","BROADCAST","OTHER"], columns=range(N_BLOCKS))
                    fig,ax=plt.subplots(figsize=(14,4))
                    im=ax.imshow(piv.to_numpy(float),aspect="auto",vmin=0,vmax=1,cmap="viridis")
                    ax.set_xticks(range(N_BLOCKS)); ax.set_xticklabels(range(N_BLOCKS)); ax.set_yticks(range(3)); ax.set_yticklabels(piv.index)
                    ax.set_xlabel("Block"); ax.set_title(f"{model}/{pump}/{sink_type}: {metric}")
                    fig.colorbar(im,ax=ax); savefig(fig,pdir/f"{safe_name(model)}__{pump}__{sink_type}__{metric}.png")


def plot_head_quadrant(hs: pd.DataFrame, out: Path) -> None:
    """User-requested style: zero650 PATCH-NOP effect vs zero565 CLS-BROADCAST effect."""
    pdir=out/"plots"/"head_quadrants";pdir.mkdir(parents=True,exist_ok=True)
    for model in hs.model_name.unique():
        for pump in PUMP_MODES:
            q=hs[(hs.model_name==model)&(hs.pump_mode==pump)]
            base=q[q.channel_condition=="normal"][["block","head","patch_nop_rate","cls_broadcast_rate"]].rename(columns={"patch_nop_rate":"base_patch_nop","cls_broadcast_rate":"base_cls_bc"})
            z650=q[q.channel_condition=="zero650"][["block","head","patch_nop_rate"]].rename(columns={"patch_nop_rate":"z650_patch_nop"})
            z565=q[q.channel_condition=="zero565"][["block","head","cls_broadcast_rate"]].rename(columns={"cls_broadcast_rate":"z565_cls_bc"})
            m=base.merge(z650,on=["block","head"]).merge(z565,on=["block","head"])
            if m.empty:continue
            m["dx"]=m.z650_patch_nop-m.base_patch_nop
            m["dy"]=m.z565_cls_bc-m.base_cls_bc
            fig,ax=plt.subplots(figsize=(9,8))
            sc=ax.scatter(m.dx,m.dy,c=m.block,cmap="viridis",s=35,alpha=.72)
            score=np.sqrt(m.dx*m.dx+m.dy*m.dy)
            for i in np.argsort(score.to_numpy())[-25:]:
                r=m.iloc[int(i)];ax.annotate(f"B{int(r['block'])}H{int(r['head'])}",(r["dx"],r["dy"]),fontsize=6)
            ax.axhline(0,lw=.8);ax.axvline(0,lw=.8);ax.grid(alpha=.18)
            ax.set_xlabel("Δ PATCH-NOP from zero650")
            ax.set_ylabel("Δ CLS-BROADCAST from zero565")
            ax.set_title(f"{model}/{pump}: which heads depend on which backbone rail?")
            fig.colorbar(sc,ax=ax,label="block")
            savefig(fig,pdir/f"HEAD_QUADRANT__{safe_name(model)}__{pump}.png")
            m.to_csv(pdir/f"HEAD_QUADRANT_DATA__{safe_name(model)}__{pump}.csv",index=False)


def plot_norm_curves(norm: pd.DataFrame, out: Path) -> None:
    pdir=out/"plots"/"norm_curves";pdir.mkdir(parents=True,exist_ok=True)
    agg=norm.groupby(["model_name","pump_mode","channel_condition","block"],as_index=False).agg(
        maxnorm=("spatial_norm_max_post","median"),p95=("spatial_norm_p95_post","median"),count70=("count_gt70_post","mean")
    )
    for model in agg.model_name.unique():
        for pump in PUMP_MODES:
            q=agg[(agg.model_name==model)&(agg.pump_mode==pump)]
            fig,axes=plt.subplots(1,2,figsize=(15,5.5))
            for cc,g in q.groupby("channel_condition",sort=False):
                g=g.sort_values("block");col=COND_COLORS.get(cc)
                axes[0].plot(g.block,g.maxnorm,marker="o",ms=3,color=col,label=f"{cc} max")
                axes[0].plot(g.block,g.p95,marker=".",ls="--",alpha=.6,color=col,label=f"{cc} p95")
                axes[1].plot(g.block,g.count70,marker="o",ms=3,color=col,label=cc)
            for ax in axes:ax.set_xticks(range(N_BLOCKS));ax.grid(alpha=.2);ax.set_xlabel("Block")
            axes[0].set_ylabel("spatial residual norm");axes[1].set_ylabel("mean count >70")
            axes[0].set_title("norm geyser");axes[1].set_title("massive-token count")
            axes[1].legend(ncol=2,fontsize=8)
            fig.suptitle(f"{model}/{pump}: channel dependence of register/readout norm structure")
            savefig(fig,pdir/f"NORM_CURVES__{safe_name(model)}__{pump}.png")


def plot_scanner_heatmaps(scanner: pd.DataFrame, out: Path) -> None:
    pdir=out/"plots"/"scanner_orienteering";pdir.mkdir(parents=True,exist_ok=True)
    metrics=("scanner_kl_from_uniform","scanner_row_kl","scanner_col_kl","scanner_row_col_anisotropy")
    for model in scanner.model_name.unique():
        for pump in scanner.loc[scanner.model_name.eq(model),"pump_mode"].drop_duplicates().tolist():
            for cc in scanner.loc[(scanner.model_name.eq(model)) & (scanner.pump_mode.eq(pump)), "channel_condition"].drop_duplicates().tolist():
                q=scanner[(scanner.model_name==model)&(scanner.pump_mode==pump)&(scanner.channel_condition==cc)]
                if q.empty:
                    continue
                for metric in metrics:
                    piv=q.pivot(index="head",columns="block",values=metric).reindex(index=range(N_HEADS),columns=range(N_BLOCKS))
                    arr = piv.to_numpy(dtype=float)
                    if not np.isfinite(arr).any():
                        continue
                    fig,ax=plt.subplots(figsize=(14,7));im=ax.imshow(arr,aspect="auto",cmap="magma")
                    ax.set_xticks(range(N_BLOCKS));ax.set_xticklabels(range(N_BLOCKS));ax.set_yticks(range(N_HEADS));ax.set_yticklabels([f"H{i}" for i in range(N_HEADS)])
                    ax.set_xlabel("Block");ax.set_ylabel("Head");ax.set_title(f"{model}/{pump}/{cc}: {metric}")
                    cb = fig.colorbar(im,ax=ax)
                    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                    savefig(fig,pdir/f"{safe_name(model)}__{pump}__{cc}__{metric}.png")


def plot_spatial_maps(out: Path, models: Sequence[str], hs: pd.DataFrame, plot_all: bool=True) -> None:
    if not plot_all:return
    pdir=out/"plots"/"cls_spatial_maps";pdir.mkdir(parents=True,exist_ok=True)
    for model in models:
        modes=hs.loc[hs.model_name.eq(model),"mode"].unique().tolist()
        for mode in modes:
            p=out/f"cls_query_spatial_{model}_{mode}.npz"
            if not p.exists():continue
            z=np.load(p);a=z["attn_mean"]
            P=a.shape[-1];grid=patch_grid_shape(P)
            if grid is None:continue
            for b in range(a.shape[0]):
                fig,axes=plt.subplots(4,4,figsize=(14,14))
                vmax=float(np.nanmax(a[b]))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                im = None
                for h,ax in enumerate(axes.flat):
                    im=ax.imshow(a[b,h].reshape(grid),vmin=0,vmax=vmax,cmap="viridis")
                    ax.set_title(f"H{h}");ax.set_xticks([]);ax.set_yticks([])
                fig.suptitle(f"{model} / {mode} / B{b}: mean conditional CLS→patch attention\npatch mass renormalized to 1 within each image/head")
                # Put the colorbar in a fixed outside slot and format ticks deterministically.
                fig.subplots_adjust(left=.05, right=.88, top=.92, bottom=.04, wspace=.18, hspace=.24)
                cax = fig.add_axes([0.90, 0.18, 0.022, 0.64])
                cb = fig.colorbar(im, cax=cax)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
                fig.savefig(pdir/f"CLS_SPATIAL__{safe_name(model)}__{safe_name(mode)}__B{b:02d}.png",dpi=150)
                plt.close(fig)


def plot_sankey(hs: pd.DataFrame, out: Path) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        print("[plotly] unavailable; skipping Sankey")
        return
    pdir=out/"plots"/"sankey";pdir.mkdir(parents=True,exist_ok=True)

    def dominant_regime(row) -> str:
        vals={"NOP":row.nop_rate,"BROADCAST":row.broadcast_rate,"OTHER":row.other_rate,"NO_SINK":1-row.sink_rate}
        return max(vals,key=vals.get)

    q=hs.copy();q["dominant_regime"]=q.apply(dominant_regime,axis=1)
    for model in q.model_name.unique():
        for pump in PUMP_MODES:
            base=q[(q.model_name==model)&(q.pump_mode==pump)&(q.channel_condition=="normal")][["block","head","dominant_regime"]].rename(columns={"dominant_regime":"src"})
            for cc in ("zero565","zero650","zero565_650","zero123"):
                dst=q[(q.model_name==model)&(q.pump_mode==pump)&(q.channel_condition==cc)][["block","head","dominant_regime"]].rename(columns={"dominant_regime":"dst"})
                m=base.merge(dst,on=["block","head"])
                if m.empty:continue
                trans=m.groupby(["src","dst"]).size().reset_index(name="n")
                left=[f"normal:{r}" for r in ("NOP","BROADCAST","OTHER","NO_SINK")]
                right=[f"{cc}:{r}" for r in ("NOP","BROADCAST","OTHER","NO_SINK")]
                labels=left+right;idx={x:i for i,x in enumerate(labels)}
                src=[];dstidx=[];val=[]
                for r in trans.itertuples(index=False):
                    src.append(idx[f"normal:{r.src}"]);dstidx.append(idx[f"{cc}:{r.dst}"]);val.append(int(r.n))
                fig=go.Figure(data=[go.Sankey(node=dict(label=labels,pad=15,thickness=18),link=dict(source=src,target=dstidx,value=val))])
                fig.update_layout(title_text=f"{model}/{pump}: dominant head-regime transitions normal → {cc}",font_size=11)
                html=pdir/f"SANKEY__{safe_name(model)}__{pump}__{cc}.html";fig.write_html(str(html),include_plotlyjs="cdn")
                try:
                    fig.write_image(str(pdir/f"SANKEY__{safe_name(model)}__{pump}__{cc}.png"),scale=2)
                except Exception as e:
                    print(f"[kaleido] PNG skipped: {e}")


# =============================================================================
# Report / bundle
# =============================================================================

def write_report(out: Path, hs: pd.DataFrame, bs: pd.DataFrame, scanner: pd.DataFrame, crs: pd.DataFrame) -> None:
    lines=[
        "# Channel-tagged NOP/BROADCAST atlas", "",
        "Existing paper-adapted dual-algorithm thresholds were not changed:",
        f"- hard sink threshold = {BASE.PAPER_HEAD_SINK_THRESHOLD:g}",
        f"- NOP V-ratio threshold = {BASE.PAPER_NOP_VALUE_RATIO:g}",
        f"- BROADCAST projected stable-rank threshold = {BASE.PAPER_RANK1_STABLE_RANK:g}", "",
        "## Conditions", "",
        "Channel conditions: normal, zero565, zero650, zero565_650, zero123.",
        "Each is crossed with intact and exact B11/B12 no-pump.", "",
        "## Biggest channel-sensitive algorithm shifts", "",
    ]
    base=bs[bs.channel_condition=="normal"][["model_name","pump_mode","block","patch_nop_rate","cls_broadcast_rate"]].rename(columns={"patch_nop_rate":"base_patch_nop","cls_broadcast_rate":"base_cls_bc"})
    for cc in ("zero565","zero650","zero565_650","zero123"):
        z=bs[bs.channel_condition==cc][["model_name","pump_mode","block","patch_nop_rate","cls_broadcast_rate"]].merge(base,on=["model_name","pump_mode","block"])
        z["d_patch_nop"]=z.patch_nop_rate-z.base_patch_nop;z["d_cls_bc"]=z.cls_broadcast_rate-z.base_cls_bc
        z["score"]=np.sqrt(z.d_patch_nop**2+z.d_cls_bc**2)
        lines += [f"### {cc}"]
        for r in z.nlargest(12,"score").itertuples(index=False):
            lines.append(f"- {r.model_name}/{r.pump_mode}/B{int(r.block)}: ΔPATCH-NOP={r.d_patch_nop:+.4f}, ΔCLS-BC={r.d_cls_bc:+.4f}")
        lines += [""]
    lines += [
        "## Scanner/orienteering", "",
        "`scanner_orienteering_scores.csv` quantifies KL-from-uniform and row/column anisotropy of the mean conditional CLS→patch map.", "",
        "## Channel tags", "",
        "For every sink event the CSV contains exact Q-side, K-side and both-side coordinate contributions for 565/650/123, plus target projected-V cargo coordinates.", "",
        "## First plots", "",
        "- `plots/phase_trajectories/`", "- `plots/head_quadrants/`", "- `plots/channel_tag_violins/`", "- `plots/scanner_orienteering/`", "- `plots/cls_spatial_maps/`", "- `plots/sankey/`", "",
        "## Main tables", "",
        "- `channel_tagged_head_summary.csv`", "- `channel_tagged_block_summary.csv`", "- `channel_regime_summary.csv`", "- `scanner_orienteering_scores.csv`", "- raw per-image head/norm/probe CSVs", "",
    ]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def build_zip(out: Path) -> Path:
    zpath=out/HANDOFF_ZIP
    with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in out.rglob("*"):
            if not p.is_file() or p==zpath:continue
            # Keep the handoff compact-ish: include all plots/tables but skip any accidental giant (>80MB) file.
            if p.stat().st_size > 80*1024*1024:continue
            z.write(p,p.relative_to(out).as_posix())
    return zpath


# =============================================================================
# Postprocess
# =============================================================================

def postprocess(args, out: Path) -> None:
    heads=[];layers=[];norms=[];probes=[]
    for model in args.models:
        hp=out/f"channel_tagged_head_events_{model}.csv"
        lp=out/f"channel_tagged_layer_events_{model}.csv"
        npth=out/f"channel_tagged_norm_rows_{model}.csv"
        pp=out/f"channel_tagged_probe_rows_{model}.csv"
        if hp.exists():heads.append(pd.read_csv(hp))
        if lp.exists():layers.append(pd.read_csv(lp))
        if npth.exists():norms.append(pd.read_csv(npth))
        if pp.exists():probes.append(pd.read_csv(pp))
    if not heads:raise RuntimeError("No head event CSVs found")
    H=pd.concat(heads,ignore_index=True);L=pd.concat(layers,ignore_index=True) if layers else pd.DataFrame();N=pd.concat(norms,ignore_index=True) if norms else pd.DataFrame();P=pd.concat(probes,ignore_index=True) if probes else pd.DataFrame()
    hs=head_summary(H);bs=block_summary(hs);crs=channel_regime_summary(H);scanner=scanner_summary_from_npz(out,args.models,hs)
    hs.to_csv(out/"channel_tagged_head_summary.csv",index=False)
    bs.to_csv(out/"channel_tagged_block_summary.csv",index=False)
    crs.to_csv(out/"channel_regime_summary.csv",index=False)
    scanner.to_csv(out/"scanner_orienteering_scores.csv",index=False)
    if len(P):P.to_csv(out/"channel_tagged_probe_rows_ALL.csv",index=False)

    plot_phase_trajectories(bs,out)
    plot_regime_rate_curves(bs,out)
    plot_head_heatmaps(hs,out)
    plot_nop_broadcast_clouds(H,out)
    plot_channel_violins(H,out)
    plot_channel_tag_heatmaps(crs,out)
    plot_head_quadrant(hs,out)
    if len(N):plot_norm_curves(N,out)
    if len(scanner):plot_scanner_heatmaps(scanner,out)
    plot_spatial_maps(out,args.models,hs,plot_all=args.plot_all_spatial)
    plot_sankey(hs,out)
    write_report(out,hs,bs,scanner,crs)
    z=build_zip(out);print(f"[compact summary] {z}")


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--out_dir",default=DEFAULT_OUT)
    ap.add_argument("--manifest",default="")
    ap.add_argument("--image_dir",default=str(SPECIAL_DELIVERY_DIR))
    ap.add_argument("--models",default=",".join(DEFAULT_MODELS))
    ap.add_argument("--batch_size",type=int,default=4)
    ap.add_argument("--sink_threshold",type=float,default=BASE.PAPER_HEAD_SINK_THRESHOLD)
    ap.add_argument("--massive_threshold",type=float,default=BASE.DEFAULT_MASSIVE_TOKEN_NORM)
    ap.add_argument("--b20_readout_threshold",type=float,default=30.0)
    ap.add_argument("--restart",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--postprocess_only",action="store_true")
    ap.add_argument("--plot_all_spatial",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--clip_module",default=BASE.S.DEFAULT_CLIP_MODULE)
    ap.add_argument("--model_spec",default=BASE.S.DEFAULT_MODEL_SPEC)
    ap.add_argument("--xattn_checkpoint",default=BASE.DEFAULT_XATTN_CHECKPOINT)
    ap.add_argument("--xattn_module",default="oaiclip")
    ap.add_argument("--pickle_module",default="clip")
    ap.add_argument("--gmp_checkpoint",default=BASE.DEFAULT_CHECKPOINTS[BASE.MODEL_GMP])
    ap.add_argument("--regression_checkpoint",default=BASE.DEFAULT_CHECKPOINTS[BASE.MODEL_REG])
    ap.add_argument("--brut_checkpoint",default=BASE.DEFAULT_CHECKPOINTS[BASE.MODEL_BRUT])
    ap.add_argument("--sae_hinge_checkpoint",default=BASE.DEFAULT_CHECKPOINTS[BASE.MODEL_SAE])
    args=ap.parse_args(argv)
    args.models=tuple(x.strip() for x in args.models.split(",") if x.strip())
    unknown=set(args.models)-set(BASE.MODEL_ORDER)
    if unknown:ap.error(f"unknown models: {sorted(unknown)}")
    return args


def main(argv=None):
    args=parse_args(argv)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);(out/"plots").mkdir(exist_ok=True)
    if not args.manifest:
        args.manifest = str(build_special_delivery_manifest(Path(args.image_dir), out))
    print(f"[dataset] controlled SPECIAL DELIVERY pairs: {args.manifest}")
    print(f"[models] pretrained + GmP only by default: {args.models}")
    (out/"config.json").write_text(json.dumps(vars(args),indent=2,default=str),encoding="utf-8")
    (out/"register_pump_units.json").write_text(json.dumps({str(k):list(v) for k,v in REGISTER_PUMP_UNITS.items()},indent=2),encoding="utf-8")
    if not args.postprocess_only:
        for model in args.models:
            run_extraction(args,out,model)
    postprocess(args,out)
    print(f"[done] {out}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
