from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_backbone import B20, BatchOracle, DEFAULT_CHECKPOINTS, DEFAULT_MASSIVE_TOKEN_NORM, DEFAULT_XATTN_CHECKPOINT, MODEL_BRUT, MODEL_FT, MODEL_GMP, MODEL_ORDER, MODEL_PRE, MODEL_REG, MODEL_SAE, MODE_NOPUMP, PAPER_HEAD_SINK_THRESHOLD, PAPER_NOP_VALUE_RATIO, PAPER_RANK1_STABLE_RANK, REGISTER_PUMP_UNITS, S, apply_pump_ablation, clear_attn_cache, event_regime, get_b13_oracle, load_bundle, normalize_probs_shape, normalize_qkv_shape, projected_row_norms, projected_row_norms_batch, projected_stable_rank, stable_rank

import argparse
import gc
import json
import math
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F


# =============================================================================
# Paper-adapted constants / our CLIP controls
# =============================================================================


MODE_INTACT = "intact"


# Complete functional B11/B12 pump list after the neuron-paranoia detour.

N_BLOCKS = 24
N_HEADS = 16

DEFAULT_OUT = r"clip_nop_broadcast_sinks_retake"


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def qnan(x: pd.Series, q: float = .5) -> float:
    a = pd.to_numeric(x, errors="coerce").to_numpy(np.float64)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if len(a) else float("nan")


# =============================================================================
# Model loading
# =============================================================================


# =============================================================================
# Oracle and extraction
# =============================================================================


def _sum_index_set(vec: torch.Tensor, patch_set: set) -> torch.Tensor:
    """vec [...,T], patch ids are 0-based spatial; return sum over token ids +1."""
    if not patch_set:
        return torch.zeros(vec.shape[:-1], device=vec.device, dtype=vec.dtype)
    idx = torch.as_tensor([p + 1 for p in sorted(patch_set)], device=vec.device, dtype=torch.long)
    return vec.index_select(-1, idx).sum(dim=-1)


def _source_label(si: int, regs: set, b20_intact: set, b20_local: set) -> str:
    if si == 0:
        return "CLS"
    p = si - 1
    if p in regs:
        return "B13_REG"
    if p in b20_local:
        return "B20_NEW_LOCAL"
    if p in b20_intact:
        return "B20_NEW_INTACT"
    return "PATCH_OTHER"


@torch.no_grad()
def collect_batch(bundle, images: torch.Tensor, ids: Sequence[str], sources: Sequence[str],
                  model_name: str, mode: str, args, seed_oracle: BatchOracle):
    """Full streaming pass with paper diagnostics + CLS-query + LN probes."""
    v = bundle.model.visual
    x = v._prepare_tokens(images.to(bundle.device, dtype=bundle.model.dtype))
    B = images.shape[0]
    H = int(v.transformer.resblocks[0].attn.num_heads)
    T = int(x.shape[0])
    P = T - 1
    dh = int(v.transformer.resblocks[0].attn.embed_dim // H)

    b13_regs = [set(s) for s in seed_oracle.b13_regs]
    primary_reg = list(seed_oracle.b13_primary_reg)
    primary_patch_sink = list(seed_oracle.b13_primary_patch_sink)
    probe_patch = list(seed_oracle.probe_patch)
    intact_b20 = [set(s) for s in seed_oracle.b20_newnorm] if seed_oracle.b20_newnorm else [set() for _ in range(B)]
    local_b20 = [set() for _ in range(B)]
    intact_refs = seed_oracle.intact_refs if mode == MODE_NOPUMP else {}
    refs_out: Dict[int, Dict[str, torch.Tensor]] = {}

    head_rows: List[dict] = []
    layer_rows: List[dict] = []
    norm_rows: List[dict] = []
    ln_token_rows: List[dict] = []
    ln_head_rows: List[dict] = []

    # Spatial sums are conditional over PATCH sources only, so the heatmaps show
    # WHERE CLS looks when it looks at patches, independently of total patch mass.
    spatial_attn_sum = np.zeros((N_BLOCKS, H, P), np.float64)
    spatial_av_sum = np.zeros((N_BLOCKS, H, P), np.float64)
    spatial_count = np.zeros((N_BLOCKS, H), np.int64)

    for b, blk in enumerate(v.transformer.resblocks):
        pre = x
        ln1 = blk.ln_1(x)
        attn_out, probs0 = blk.attention(ln1, need_weights=True, capture=True)
        probs = normalize_probs_shape(probs0, B, H, T).float()
        vv = normalize_qkv_shape(blk.attn.last_v, B, H, T).float()
        kk = normalize_qkv_shape(blk.attn.last_k, B, H, T).float()
        qq = normalize_qkv_shape(blk.attn.last_q, B, H, T).float()
        Wout = blk.attn.out_proj.weight.detach().float()

        raw_norm = pre.float().norm(dim=-1).T
        ln_norm = ln1.float().norm(dim=-1).T
        hin = probs.mean(dim=2)  # [B,H,key]
        strengths, sidx = hin.max(dim=-1)

        # Precompute projected V row norms for |A*V| in residual-stream units.
        vnorm = vv.norm(dim=-1)  # [B,H,T]
        vproj = torch.empty_like(vnorm)
        for h in range(H):
            vproj[:, h] = projected_row_norms_batch(vv[:, h], Wout, h)

        cls_a = probs[:, :, 0, :]  # Q=CLS
        cls_av_h = cls_a * vnorm
        cls_av_r = cls_a * vproj
        avh_den = cls_av_h.sum(dim=-1).clamp_min(1e-12)
        avr_den = cls_av_r.sum(dim=-1).clamp_min(1e-12)

        # Spatial CLS-query maps, conditioned on patch sources.
        pa = cls_a[:, :, 1:]
        pav = cls_av_r[:, :, 1:]
        pa_cond = pa / pa.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pav_cond = pav / pav.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        spatial_attn_sum[b] += pa_cond.sum(dim=0).cpu().numpy()
        spatial_av_sum[b] += pav_cond.sum(dim=0).cpu().numpy()
        spatial_count[b] += B

        # Paper-style head-averaged sink transition point.
        pmean = probs.mean(dim=1)
        inflow = pmean.mean(dim=1)
        layer_strength, layer_idx = inflow.max(dim=-1)
        for bi in range(B):
            si = int(layer_idx[bi])
            typ = "CLS" if si == 0 else "PATCH"
            patch = si - 1
            layer_rows.append({
                "model_name": model_name, "mode": mode, "stim_id": ids[bi], "source": sources[bi],
                "block": b, "sink_idx": si, "sink_patch_idx": patch if si > 0 else -1,
                "sink_type": typ, "sink_strength": float(layer_strength[bi]),
                "sink_raw_norm": float(raw_norm[bi, si]), "sink_ln1_norm": float(ln_norm[bi, si]),
                "is_b13_register": int(si > 0 and patch in b13_regs[bi]),
                "is_b20_newnorm_intact": int(si > 0 and patch in intact_b20[bi]),
                "is_b20_newnorm_local": int(si > 0 and patch in local_b20[bi]),
            })

        # Intact-vs-no-pump LN/QKV probe at the SAME B13 intact probe address.
        raw_probe = torch.stack([pre[p + 1, bi].float() for bi, p in enumerate(probe_patch)], dim=0)
        ln_probe = torch.stack([ln1[p + 1, bi].float() for bi, p in enumerate(probe_patch)], dim=0)
        q_probe = torch.stack([qq[bi, :, p + 1] for bi, p in enumerate(probe_patch)], dim=0)
        k_probe = torch.stack([kk[bi, :, p + 1] for bi, p in enumerate(probe_patch)], dim=0)
        v_probe = torch.stack([vv[bi, :, p + 1] for bi, p in enumerate(probe_patch)], dim=0)

        if mode == MODE_INTACT:
            refs_out[b] = {
                "raw": raw_probe.detach().cpu(), "ln": ln_probe.detach().cpu(),
                "q": q_probe.detach().cpu(), "k": k_probe.detach().cpu(), "v": v_probe.detach().cpu(),
            }
            raw_cos = torch.ones(B)
            ln_cos = torch.ones(B)
            q_cos = torch.ones(B, H)
            k_cos = torch.ones(B, H)
            v_cos = torch.ones(B, H)
        else:
            ref = intact_refs[b]
            raw_cos = F.cosine_similarity(raw_probe.cpu(), ref["raw"].float(), dim=-1, eps=1e-8)
            ln_cos = F.cosine_similarity(ln_probe.cpu(), ref["ln"].float(), dim=-1, eps=1e-8)
            q_cos = F.cosine_similarity(q_probe.cpu(), ref["q"].float(), dim=-1, eps=1e-8)
            k_cos = F.cosine_similarity(k_probe.cpu(), ref["k"].float(), dim=-1, eps=1e-8)
            v_cos = F.cosine_similarity(v_probe.cpu(), ref["v"].float(), dim=-1, eps=1e-8)

        for bi in range(B):
            pi = probe_patch[bi] + 1
            ln_token_rows.append({
                "model_name": model_name, "mode": mode, "stim_id": ids[bi], "source": sources[bi], "block": b,
                "probe_patch_idx": probe_patch[bi], "probe_is_visible_b13_reg": int(primary_reg[bi] >= 0),
                "probe_raw_norm": float(raw_norm[bi, pi]), "probe_ln1_norm": float(ln_norm[bi, pi]),
                "probe_raw_cos_to_intact": float(raw_cos[bi]), "probe_ln_cos_to_intact": float(ln_cos[bi]),
                "probe_headmean_inflow": float(hin[bi, :, pi].mean()),
                "probe_headmean_cls_attention": float(cls_a[bi, :, pi].mean()),
            })

        # Per-head sink event + CLS-query + LN/QKV diagnostics.
        for bi in range(B):
            regs = b13_regs[bi]
            b20i = intact_b20[bi]
            b20l = local_b20[bi]
            pi = probe_patch[bi] + 1
            for h in range(H):
                si = int(sidx[bi, h])
                st = float(strengths[bi, h])
                is_sink = st >= args.sink_threshold
                typ = "CLS" if si == 0 else "PATCH"
                patch = si - 1

                # Q=CLS source decomposition, both A and |A*V|.
                arow = cls_a[bi, h]
                avh = cls_av_h[bi, h]
                avr = cls_av_r[bi, h]
                reg_a = float(_sum_index_set(arow, regs))
                reg_avh = float(_sum_index_set(avh, regs) / avh_den[bi, h])
                reg_avr = float(_sum_index_set(avr, regs) / avr_den[bi, h])
                b20i_a = float(_sum_index_set(arow, b20i))
                b20l_a = float(_sum_index_set(arow, b20l))
                b20i_avr = float(_sum_index_set(avr, b20i) / avr_den[bi, h])
                b20l_avr = float(_sum_index_set(avr, b20l) / avr_den[bi, h])
                cls_avr_frac = float(avr[0] / avr_den[bi, h])
                patch_avr_frac = float(avr[1:].sum() / avr_den[bi, h])
                top_av_idx = int(torch.argmax(avr).item())
                top_a_idx = int(torch.argmax(arow).item())

                row = {
                    "model_name": model_name, "mode": mode, "stim_id": ids[bi], "source": sources[bi],
                    "block": b, "head": h, "sink_idx": si, "sink_patch_idx": patch if si > 0 else -1,
                    "sink_type": typ, "sink_strength": st, "is_sink_event": int(is_sink),
                    "sink_raw_norm": float(raw_norm[bi, si]),
                    "sink_raw_norm_ratio": float(raw_norm[bi, si] / raw_norm[bi].mean().clamp_min(1e-12)),
                    "sink_ln1_norm": float(ln_norm[bi, si]),
                    "sink_ln1_norm_ratio": float(ln_norm[bi, si] / ln_norm[bi].mean().clamp_min(1e-12)),
                    "is_b13_register": int(si > 0 and patch in regs),
                    "is_b20_newnorm_intact": int(si > 0 and patch in b20i),
                    "is_b20_newnorm_local": int(si > 0 and patch in b20l),
                    # Specialized Q=CLS diagnostics.
                    "cls_attn_to_cls": float(arow[0]),
                    "cls_attn_to_patches": float(arow[1:].sum()),
                    "cls_attn_to_b13_regs": reg_a,
                    "cls_attn_to_probe": float(arow[pi]),
                    "cls_attn_to_b20_intact": b20i_a,
                    "cls_attn_to_b20_local": b20l_a,
                    "cls_av_head_to_b13_regs_frac": reg_avh,
                    "cls_av_resid_to_cls_frac": cls_avr_frac,
                    "cls_av_resid_to_patches_frac": patch_avr_frac,
                    "cls_av_resid_to_b13_regs_frac": reg_avr,
                    "cls_av_resid_to_probe_frac": float(avr[pi] / avr_den[bi, h]),
                    "cls_av_resid_to_b20_intact_frac": b20i_avr,
                    "cls_av_resid_to_b20_local_frac": b20l_avr,
                    "cls_av_top_idx": top_av_idx,
                    "cls_av_top_type": _source_label(top_av_idx, regs, b20i, b20l),
                    "cls_attn_top_idx": top_a_idx,
                    "cls_attn_top_type": _source_label(top_a_idx, regs, b20i, b20l),
                }

                if is_sink:
                    vh = vv[bi, h]
                    kh = kk[bi, h]
                    qh = qq[bi, h]
                    vn, kn, qn = vh.norm(dim=-1), kh.norm(dim=-1), qh.norm(dim=-1)
                    other = torch.ones(T, dtype=torch.bool, device=vh.device); other[si] = False
                    value_ratio = float(vn[si] / vn[other].mean().clamp_min(1e-12))
                    key_ratio = float(kn[si] / kn[other].mean().clamp_min(1e-12))
                    query_ratio = float(qn[si] / qn[other].mean().clamp_min(1e-12))
                    z = probs[bi, h] @ vh
                    sr_head = stable_rank(z)
                    sr_resid = projected_stable_rank(z, Wout, h)
                    vproj_one = projected_row_norms(vh, Wout, h)
                    value_ratio_resid = float(vproj_one[si] / vproj_one[other].mean().clamp_min(1e-12))
                    zproj = projected_row_norms(z, Wout, h)
                    update_ratio = float(zproj.mean() / raw_norm[bi].mean().clamp_min(1e-12))
                    zn = F.normalize(z, dim=-1, eps=1e-8)
                    zm = F.normalize(z.mean(dim=0), dim=0, eps=1e-8)
                    shared_cos = float((zn @ zm).mean())
                    row.update({
                        "value_norm_ratio": value_ratio, "value_norm_ratio_residual": value_ratio_resid,
                        "key_norm_ratio": key_ratio, "query_norm_ratio": query_ratio,
                        "stable_rank_head": sr_head, "stable_rank_residual": sr_resid,
                        "head_update_to_resid_ratio": update_ratio, "update_shared_cosine": shared_cos,
                        "regime": event_regime(value_ratio, sr_resid),
                    })
                else:
                    row.update({
                        "value_norm_ratio": np.nan, "value_norm_ratio_residual": np.nan,
                        "key_norm_ratio": np.nan, "query_norm_ratio": np.nan,
                        "stable_rank_head": np.nan, "stable_rank_residual": np.nan,
                        "head_update_to_resid_ratio": np.nan, "update_shared_cosine": np.nan,
                        "regime": "NO_SINK",
                    })
                head_rows.append(row)

                # LN / QKV geometry for the fixed intact B13 probe address.
                otherp = torch.ones(T, dtype=torch.bool, device=vv.device); otherp[pi] = False
                patch_other = torch.ones(T, dtype=torch.bool, device=vv.device); patch_other[0] = False; patch_other[pi] = False
                kp = kk[bi, h, pi]; vp = vv[bi, h, pi]; qp = qq[bi, h, pi]
                kmean = kk[bi, h, otherp].mean(dim=0)
                kpatchmean = kk[bi, h, patch_other].mean(dim=0)
                scale = math.sqrt(dh)
                q_all = qq[bi, h]
                logit_probe = (q_all * kp).sum(dim=-1) / scale
                logit_other = (q_all * kmean).sum(dim=-1) / scale
                logit_patch = (q_all * kpatchmean).sum(dim=-1) / scale
                ln_head_rows.append({
                    "model_name": model_name, "mode": mode, "stim_id": ids[bi], "source": sources[bi],
                    "block": b, "head": h, "probe_patch_idx": probe_patch[bi],
                    "probe_is_visible_b13_reg": int(primary_reg[bi] >= 0),
                    "probe_key_norm_ratio": float(kp.norm() / kk[bi, h, otherp].norm(dim=-1).mean().clamp_min(1e-12)),
                    "probe_value_norm_ratio": float(vp.norm() / vv[bi, h, otherp].norm(dim=-1).mean().clamp_min(1e-12)),
                    "probe_query_norm_ratio": float(qp.norm() / qq[bi, h, otherp].norm(dim=-1).mean().clamp_min(1e-12)),
                    "probe_k_cos_to_intact": float(k_cos[bi, h]),
                    "probe_v_cos_to_intact": float(v_cos[bi, h]),
                    "probe_q_cos_to_intact": float(q_cos[bi, h]),
                    "probe_inflow_strength": float(hin[bi, h, pi]),
                    "probe_cls_attention": float(cls_a[bi, h, pi]),
                    "probe_cls_av_resid_frac": float(avr[pi] / avr_den[bi, h]),
                    "probe_mean_query_qk_advantage": float((logit_probe - logit_other).mean()),
                    "probe_cls_query_qk_advantage": float((logit_probe - logit_other)[0]),
                    "probe_mean_query_qk_advantage_vs_patches": float((logit_probe - logit_patch).mean()),
                    "probe_cls_query_qk_advantage_vs_patches": float((logit_probe - logit_patch)[0]),
                })

        x_attn = x + attn_out
        ln2 = blk.ln_2(x_attn)
        fc = blk.mlp.c_fc(ln2)
        gelu = blk.mlp.gelu(fc)
        gelu = apply_pump_ablation(gelu, b, mode)
        x = x_attn + blk.mlp.c_proj(gelu)

        # Local post-B20 new-norm population; intact pass becomes the shared intact oracle.
        if b == B20:
            postn = x[1:].float().norm(dim=-1).T
            local_b20 = []
            for bi in range(B):
                idx = set(torch.nonzero(postn[bi] >= args.b20_readout_threshold, as_tuple=False).flatten().cpu().tolist())
                local_b20.append(idx - b13_regs[bi])
            if mode == MODE_INTACT:
                intact_b20 = [set(s) for s in local_b20]

        n = x[1:].float().norm(dim=-1).T
        for bi in range(B):
            norm_rows.append({
                "model_name": model_name, "mode": mode, "stim_id": ids[bi], "source": sources[bi], "block": b,
                "spatial_norm_max_post": float(n[bi].max()),
                "spatial_norm_p95_post": float(torch.quantile(n[bi], .95)),
                "spatial_norm_mean_post": float(n[bi].mean()),
                "spatial_norm_median_post": float(n[bi].median()),
                "count_gt20_post": int((n[bi] > 20).sum()),
                "count_gt30_post": int((n[bi] > 30).sum()),
                "count_gt40_post": int((n[bi] > 40).sum()),
                "count_gt60_post": int((n[bi] > 60).sum()),
                "count_gt100_post": int((n[bi] > 100).sum()),
            })
        clear_attn_cache(blk)
        del probs, vv, kk, qq, vproj

    out_oracle = BatchOracle(
        b13_regs=b13_regs,
        b13_primary_reg=primary_reg,
        b13_primary_patch_sink=primary_patch_sink,
        probe_patch=probe_patch,
        b20_newnorm=[set(s) for s in intact_b20],
        intact_refs=refs_out if mode == MODE_INTACT else seed_oracle.intact_refs,
    )
    spatial = {
        "attn_sum": spatial_attn_sum,
        "av_sum": spatial_av_sum,
        "count": spatial_count,
    }
    return head_rows, layer_rows, norm_rows, ln_token_rows, ln_head_rows, out_oracle, spatial


def append_csv(path: Path, rows: List[dict]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _save_spatial_npz(path: Path, sums: dict) -> None:
    count = sums["count"].astype(np.float64)
    denom = np.maximum(count[..., None], 1.0)
    np.savez_compressed(path,
                        attn_mean=sums["attn_sum"] / denom,
                        av_mean=sums["av_sum"] / denom,
                        count=sums["count"])


def _extraction_paths(out: Path, model_name: str) -> Tuple[Dict[str, Path], Path, List[Path]]:
    paths = {
        "head": out / f"sink_head_events_{model_name}.csv",
        "layer": out / f"sink_layer_events_{model_name}.csv",
        "norm": out / f"sink_norm_rows_{model_name}.csv",
        "ln_token": out / f"ln_probe_token_rows_{model_name}.csv",
        "ln_head": out / f"ln_probe_head_rows_{model_name}.csv",
    }
    progress_path = out / f"progress_{model_name}.json"
    spatial = [out / f"cls_query_spatial_{model_name}_{MODE_INTACT}.npz"]
    return paths, progress_path, spatial


def _model_loader_signature(args, model_name: str) -> dict:
    """Minimal provenance needed before reusing a completed model extraction leg."""
    sig = {"model_name": str(model_name)}
    if model_name == MODEL_PRE:
        sig.update({"clip_module": str(args.clip_module), "model_spec": str(args.model_spec)})
    elif model_name == MODEL_GMP:
        sig.update({"gmp_checkpoint": str(args.gmp_checkpoint)})
    elif model_name == MODEL_FT:
        sig.update({"xattn_checkpoint": str(args.xattn_checkpoint)})
    elif model_name == MODEL_REG:
        sig.update({"regression_checkpoint": str(args.regression_checkpoint)})
    elif model_name == MODEL_BRUT:
        sig.update({"brut_checkpoint": str(args.brut_checkpoint)})
    elif model_name == MODEL_SAE:
        sig.update({"sae_hinge_checkpoint": str(args.sae_hinge_checkpoint)})
    return sig


def _model_extraction_complete(out: Path, model_name: str, n_rows: int, include_no_pump: bool, args) -> bool:
    paths, progress_path, spatial = _extraction_paths(out, model_name)
    if include_no_pump:
        spatial.append(out / f"cls_query_spatial_{model_name}_{MODE_NOPUMP}.npz")
    if not all(p.is_file() for p in [*paths.values(), *spatial, progress_path]):
        return False
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        return (
            int(progress.get("next_row", -1)) >= int(n_rows)
            and progress.get("loader_signature") == _model_loader_signature(args, model_name)
        )
    except Exception:
        return False


def run_extraction(args, out: Path, model_name: str) -> None:
    model_dir = out / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    missing = [p for p in manifest.path.astype(str) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(manifest)} manifest images missing; first: {missing[0]}")

    paths, progress_path, spatial_paths = _extraction_paths(out, model_name)
    if args.include_no_pump:
        spatial_paths.append(out / f"cls_query_spatial_{model_name}_{MODE_NOPUMP}.npz")

    complete = _model_extraction_complete(
        out, model_name, len(manifest), bool(args.include_no_pump), args
    )
    if complete and not args.restart:
        print(f"[{model_name}] cached extraction complete ({len(manifest)}/{len(manifest)}); reusing")
        return

    existing = [p for p in [*paths.values(), progress_path, *spatial_paths] if p.exists()]
    if existing:
        reason = "forced --restart" if args.restart else "incomplete prior model leg"
        print(f"[{model_name}] {reason}; restarting this model extraction from row 0")
        for path in existing:
            path.unlink()

    print(f"[model] loading {model_name}")
    bundle = load_bundle(model_name, args, model_dir / "load_audit")

    # Mid-model resume is intentionally not attempted: paired intact/no-pump
    # spatial accumulators and LN references are batch-coupled. A completed model
    # leg is reusable; an incomplete leg is safely restarted from row 0.
    spatial_totals = {
        MODE_INTACT: {"attn_sum": None, "av_sum": None, "count": None},
        MODE_NOPUMP: {"attn_sum": None, "av_sum": None, "count": None},
    }

    for st in range(0, len(manifest), args.batch_size):
        q = manifest.iloc[st:st + args.batch_size]
        tensors, ids, sources = [], [], []
        for r in q.itertuples(index=False):
            with Image.open(r.path) as im:
                tensors.append(bundle.preprocess(im.convert("RGB")))
            ids.append(str(r.stim_id)); sources.append(str(r.source))
        batch = torch.stack(tensors, dim=0)

        # Small intact pilot makes B13 future-register identities available from B0.
        seed = get_b13_oracle(bundle, batch, args)
        h, l, n, lt, lh, oracle, spatial = collect_batch(
            bundle, batch, ids, sources, model_name, MODE_INTACT, args, seed)
        append_csv(paths["head"], h); append_csv(paths["layer"], l); append_csv(paths["norm"], n)
        append_csv(paths["ln_token"], lt); append_csv(paths["ln_head"], lh)
        for k in ("attn_sum", "av_sum", "count"):
            if spatial_totals[MODE_INTACT][k] is None:
                spatial_totals[MODE_INTACT][k] = spatial[k].copy()
            else:
                spatial_totals[MODE_INTACT][k] += spatial[k]
        del h, l, n, lt, lh, spatial
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if args.include_no_pump:
            h, l, n, lt, lh, _, spatial = collect_batch(
                bundle, batch, ids, sources, model_name, MODE_NOPUMP, args, oracle)
            append_csv(paths["head"], h); append_csv(paths["layer"], l); append_csv(paths["norm"], n)
            append_csv(paths["ln_token"], lt); append_csv(paths["ln_head"], lh)
            for k in ("attn_sum", "av_sum", "count"):
                if spatial_totals[MODE_NOPUMP][k] is None:
                    spatial_totals[MODE_NOPUMP][k] = spatial[k].copy()
                else:
                    spatial_totals[MODE_NOPUMP][k] += spatial[k]
            del h, l, n, lt, lh, spatial

        progress_path.write_text(json.dumps({
            "next_row": min(st + len(q), len(manifest)),
            "loader_signature": _model_loader_signature(args, model_name),
        }, indent=2))
        if (st // args.batch_size) % max(1, args.log_every) == 0 or st + len(q) >= len(manifest):
            print(f"[{model_name}] {min(st+len(q),len(manifest))}/{len(manifest)}")
        del batch, tensors, oracle, seed
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    for mode, sums in spatial_totals.items():
        if sums["attn_sum"] is not None:
            _save_spatial_npz(out / f"cls_query_spatial_{model_name}_{mode}.npz", sums)

    del bundle
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# =============================================================================
# Summaries
# =============================================================================

def head_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_name", "mode", "block", "head"]
    median_cols = [
        "sink_strength", "value_norm_ratio", "value_norm_ratio_residual", "key_norm_ratio",
        "query_norm_ratio", "stable_rank_head", "stable_rank_residual",
        "head_update_to_resid_ratio", "update_shared_cosine", "sink_raw_norm_ratio", "sink_ln1_norm_ratio",
        "cls_attn_to_cls", "cls_attn_to_patches", "cls_attn_to_b13_regs", "cls_attn_to_probe",
        "cls_av_resid_to_cls_frac", "cls_av_resid_to_patches_frac", "cls_av_resid_to_b13_regs_frac",
        "cls_av_resid_to_probe_frac", "cls_av_resid_to_b20_intact_frac", "cls_av_resid_to_b20_local_frac",
    ]
    for k, q in df.groupby(keys, sort=True):
        e = q[q.is_sink_event.eq(1)]
        row = dict(zip(keys, k))
        row["n_images"] = len(q); row["n_sink"] = len(e)
        row["sink_frequency"] = len(e) / max(1, len(q))
        # Absolute rates across ALL images, not conditional on sink occurrence.
        for typ in ("CLS", "PATCH"):
            row[f"{typ.lower()}_argmax_rate_all"] = float(q.sink_type.eq(typ).mean())
            row[f"{typ.lower()}_hard_sink_rate_all"] = float((q.is_sink_event.eq(1) & q.sink_type.eq(typ)).mean())
        row["nop_rate_all"] = float(q.regime.eq("NOP").mean())
        row["broadcast_rate_all"] = float(q.regime.eq("BROADCAST").mean())
        row["other_sink_rate_all"] = float(q.regime.eq("OTHER").mean())
        row["cls_nop_rate_all"] = float((q.regime.eq("NOP") & q.sink_type.eq("CLS")).mean())
        row["patch_nop_rate_all"] = float((q.regime.eq("NOP") & q.sink_type.eq("PATCH")).mean())
        row["cls_broadcast_rate_all"] = float((q.regime.eq("BROADCAST") & q.sink_type.eq("CLS")).mean())
        row["patch_broadcast_rate_all"] = float((q.regime.eq("BROADCAST") & q.sink_type.eq("PATCH")).mean())
        row["cls_other_rate_all"] = float((q.regime.eq("OTHER") & q.sink_type.eq("CLS") & q.is_sink_event.eq(1)).mean())
        row["patch_other_rate_all"] = float((q.regime.eq("OTHER") & q.sink_type.eq("PATCH") & q.is_sink_event.eq(1)).mean())

        # CLS-query metrics use all images.
        for c in [c for c in median_cols if c.startswith("cls_")]:
            row[c + "_median_all"] = qnan(q[c])
            row[c + "_mean_all"] = float(pd.to_numeric(q[c], errors="coerce").mean())

        if len(e):
            for c in [c for c in median_cols if not c.startswith("cls_")]:
                row[c + "_median"] = qnan(e[c])
            row["cls_fraction_given_sink"] = float(e.sink_type.eq("CLS").mean())
            row["patch_fraction_given_sink"] = float(e.sink_type.eq("PATCH").mean())
            row["b13_reg_fraction_given_patch_sink"] = float(e.loc[e.sink_type.eq("PATCH"), "is_b13_register"].mean()) if e.sink_type.eq("PATCH").any() else 0.0
            row["b20_intact_fraction_given_patch_sink"] = float(e.loc[e.sink_type.eq("PATCH"), "is_b20_newnorm_intact"].mean()) if e.sink_type.eq("PATCH").any() else 0.0
            row["b20_local_fraction_given_patch_sink"] = float(e.loc[e.sink_type.eq("PATCH"), "is_b20_newnorm_local"].mean()) if e.sink_type.eq("PATCH").any() else 0.0
            rc = e.regime.value_counts(normalize=True)
            row["nop_fraction_given_sink"] = float(rc.get("NOP", 0))
            row["broadcast_fraction_given_sink"] = float(rc.get("BROADCAST", 0))
            row["other_fraction_given_sink"] = float(rc.get("OTHER", 0))
            row["majority_sink_type"] = "CLS" if row["cls_fraction_given_sink"] >= .5 else "PATCH"
            row["paper_regime"] = event_regime(row["value_norm_ratio_median"], row["stable_rank_residual_median"])
        else:
            for c in [c for c in median_cols if not c.startswith("cls_")]:
                row[c + "_median"] = np.nan
            for c in ["cls_fraction_given_sink", "patch_fraction_given_sink", "b13_reg_fraction_given_patch_sink",
                      "b20_intact_fraction_given_patch_sink", "b20_local_fraction_given_patch_sink",
                      "nop_fraction_given_sink", "broadcast_fraction_given_sink", "other_fraction_given_sink"]:
                row[c] = 0.0
            row["majority_sink_type"] = "NONE"; row["paper_regime"] = "NO_SINK"
        rows.append(row)
    return pd.DataFrame(rows)


def layer_summary(head_df: pd.DataFrame, layer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, mode, b), q in head_df.groupby(["model_name", "mode", "block"]):
        e = q[q.is_sink_event.eq(1)]
        lq = layer_df[(layer_df.model_name == m) & (layer_df["mode"] == mode) & (layer_df.block == b)]
        patch_e = e[e.sink_type.eq("PATCH")]
        d = {
            "model_name": m, "mode": mode, "block": b,
            "head_sink_event_rate": len(e) / max(1, len(q)),
            "head_cls_argmax_rate_all": float(q.sink_type.eq("CLS").mean()),
            "head_patch_argmax_rate_all": float(q.sink_type.eq("PATCH").mean()),
            "head_cls_sink_rate_all": float((q.is_sink_event.eq(1) & q.sink_type.eq("CLS")).mean()),
            "head_patch_sink_rate_all": float((q.is_sink_event.eq(1) & q.sink_type.eq("PATCH")).mean()),
            "head_nop_rate_all": float(q.regime.eq("NOP").mean()),
            "head_broadcast_rate_all": float(q.regime.eq("BROADCAST").mean()),
            "head_other_sink_rate_all": float(q.regime.eq("OTHER").mean()),
            "head_cls_nop_rate_all": float((q.regime.eq("NOP") & q.sink_type.eq("CLS")).mean()),
            "head_patch_nop_rate_all": float((q.regime.eq("NOP") & q.sink_type.eq("PATCH")).mean()),
            "head_cls_broadcast_rate_all": float((q.regime.eq("BROADCAST") & q.sink_type.eq("CLS")).mean()),
            "head_patch_broadcast_rate_all": float((q.regime.eq("BROADCAST") & q.sink_type.eq("PATCH")).mean()),
            "head_cls_other_rate_all": float((q.regime.eq("OTHER") & q.sink_type.eq("CLS") & q.is_sink_event.eq(1)).mean()),
            "head_patch_other_rate_all": float((q.regime.eq("OTHER") & q.sink_type.eq("PATCH") & q.is_sink_event.eq(1)).mean()),
            "head_sink_strength_median": qnan(e.sink_strength) if len(e) else np.nan,
            "head_nop_fraction_given_sink": float(e.regime.eq("NOP").mean()) if len(e) else 0,
            "head_broadcast_fraction_given_sink": float(e.regime.eq("BROADCAST").mean()) if len(e) else 0,
            "head_cls_fraction_given_sink": float(e.sink_type.eq("CLS").mean()) if len(e) else 0,
            "head_patch_fraction_given_sink": float(e.sink_type.eq("PATCH").mean()) if len(e) else 0,
            "head_b13_reg_fraction_given_patch_sink": float(patch_e.is_b13_register.mean()) if len(patch_e) else 0,
            "head_b20_intact_fraction_given_patch_sink": float(patch_e.is_b20_newnorm_intact.mean()) if len(patch_e) else 0,
            "head_b20_local_fraction_given_patch_sink": float(patch_e.is_b20_newnorm_local.mean()) if len(patch_e) else 0,
            "meanhead_sink_strength_median": qnan(lq.sink_strength) if len(lq) else np.nan,
            "meanhead_cls_fraction": float(lq.sink_type.eq("CLS").mean()) if len(lq) else 0,
            "meanhead_patch_fraction": float(lq.sink_type.eq("PATCH").mean()) if len(lq) else 0,
            "meanhead_b13_reg_fraction": float(lq.is_b13_register.mean()) if len(lq) else 0,
            "meanhead_b20_intact_fraction": float(lq.is_b20_newnorm_intact.mean()) if len(lq) else 0,
            "meanhead_b20_local_fraction": float(lq.is_b20_newnorm_local.mean()) if len(lq) else 0,
            "cls_attn_to_b13_regs_mean": float(q.cls_attn_to_b13_regs.mean()),
            "cls_av_to_b13_regs_mean": float(q.cls_av_resid_to_b13_regs_frac.mean()),
            "cls_attn_to_probe_mean": float(q.cls_attn_to_probe.mean()),
            "cls_av_to_probe_mean": float(q.cls_av_resid_to_probe_frac.mean()),
            "cls_av_to_cls_mean": float(q.cls_av_resid_to_cls_frac.mean()),
            "cls_av_to_patches_mean": float(q.cls_av_resid_to_patches_frac.mean()),
        }
        rows.append(d)
    return pd.DataFrame(rows)


def ln_summary(token_df: pd.DataFrame, head_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trows, hrows = [], []
    for k, q in token_df.groupby(["model_name", "mode", "block"], sort=True):
        d = dict(zip(["model_name", "mode", "block"], k))
        for c in ["probe_raw_norm", "probe_ln1_norm", "probe_raw_cos_to_intact", "probe_ln_cos_to_intact",
                  "probe_headmean_inflow", "probe_headmean_cls_attention"]:
            d[c + "_median"] = qnan(q[c])
            d[c + "_mean"] = float(q[c].mean())
        d["visible_reg_fraction"] = float(q.probe_is_visible_b13_reg.mean())
        trows.append(d)
    for k, q in head_df.groupby(["model_name", "mode", "block", "head"], sort=True):
        d = dict(zip(["model_name", "mode", "block", "head"], k))
        for c in ["probe_key_norm_ratio", "probe_value_norm_ratio", "probe_query_norm_ratio",
                  "probe_k_cos_to_intact", "probe_v_cos_to_intact", "probe_q_cos_to_intact",
                  "probe_inflow_strength", "probe_cls_attention", "probe_cls_av_resid_frac",
                  "probe_mean_query_qk_advantage", "probe_cls_query_qk_advantage",
                  "probe_mean_query_qk_advantage_vs_patches", "probe_cls_query_qk_advantage_vs_patches"]:
            d[c + "_median"] = qnan(q[c])
            d[c + "_mean"] = float(q[c].mean())
        hrows.append(d)
    return pd.DataFrame(trows), pd.DataFrame(hrows)


def norm_summary(norm_df: pd.DataFrame) -> pd.DataFrame:
    agg = []
    cols = ["spatial_norm_max_post", "spatial_norm_p95_post", "spatial_norm_mean_post", "spatial_norm_median_post",
            "count_gt20_post", "count_gt30_post", "count_gt40_post", "count_gt60_post", "count_gt100_post"]
    for k, q in norm_df.groupby(["model_name", "mode", "block"], sort=True):
        d = dict(zip(["model_name", "mode", "block"], k))
        for c in cols:
            d[c + "_median"] = qnan(q[c])
            d[c + "_mean"] = float(q[c].mean())
        agg.append(d)
    return pd.DataFrame(agg)


# =============================================================================
# Plotting
# =============================================================================

def _savefig(fig, path: Path, dpi: int = 180):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _nonwhite_diverging_cmap():
    """Cold→dark→warm diverging map with NO white/yellow midpoint.

    White-background scatter plots made viridis' late yellow points effectively
    disappear.  This fixed palette remains visible at every block value and also
    gives signed QK/cosine heatmaps a dark rather than white zero.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "clip_cold_dark_warm",
        ["#2348A5", "#2878B8", "#1F9E89", "#2A2633", "#8C2D04", "#CC3D08", "#E4571E"],
        N=256,
    )


def _dark_sequential_cmap():
    """Sequential map truncated before pale/yellow values."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "clip_dark_seq",
        ["#140A24", "#35105B", "#6F1D7A", "#A92E6F", "#D84A55", "#ED6A2C"],
        N=256,
    )


BLOCK_CMAP = _nonwhite_diverging_cmap()
SIGNED_CMAP = _nonwhite_diverging_cmap()
SEQUENTIAL_CMAP = "viridis"  # generic matrix/spatial heatmaps: bright-on-dark is desirable here
REGIME_COLORS = {"NOP": "#1874B4", "BROADCAST": "#E4571E", "OTHER": "#696969"}


def _heatmap(ax, mat, title, xlabel="Block", ylabel="Head", vmin=None, vmax=None, cmap=None, cbar_label=None):
    if cmap is None:
        cmap = SEQUENTIAL_CMAP
    im = ax.imshow(mat, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xticks(range(N_BLOCKS)); ax.set_yticks(range(N_HEADS))
    return im


def plot_transition_clouds(pdir: Path, model: str, mode: str, h: pd.DataFrame, l: pd.DataFrame, args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rng = np.random.default_rng(0)

    # All images, head-mean winner: the paper-style geyser/handoff plot.
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"CLS": "tab:blue", "PATCH": "tab:orange"}
    for typ in ("CLS", "PATCH"):
        q = l[l.sink_type.eq(typ)]
        jitter = rng.normal(0, .045, len(q))
        ax.scatter(q.block.to_numpy() + jitter, q.sink_strength, s=10, alpha=.15,
                   color=colors[typ], label=f"{typ} is strongest head-mean incoming target")
    ax.axhline(args.sink_threshold, ls="--", lw=1.2, color="0.35", label=f"hard-sink reference = {args.sink_threshold:g}")
    for x, txt in [(12, "B11/12\npump/commit"), (18, "B18\nlate readout"), (20, "B20\nreadout geyser"), (22, "B22\nterminal sink"), (23, "B23\nfinal")]:
        ax.axvline(x, lw=.6, color="0.82", zorder=0)
        ax.text(x, 1.005, txt, ha="center", va="bottom", fontsize=7, color="0.35")
    ax.set(xlabel="Block", ylabel="Max incoming attention mass (heads averaged first)",
           title=f"{model} / {mode}: CLS ↔ patch sink handoff — ALL images")
    ax.set_xticks(range(N_BLOCKS)); ax.set_xlim(-.5, 23.5); ax.set_ylim(0, 1.08)
    ax.legend(loc="upper left", fontsize=8)
    _savefig(fig, pdir / f"sink_transition_ALL_{model}_{mode}.png")

    # Per-head points only when they meet the hard sink cutoff.
    e = h[h.is_sink_event.eq(1)]
    fig, ax = plt.subplots(figsize=(14, 6))
    for typ in ("CLS", "PATCH"):
        q = e[e.sink_type.eq(typ)]
        if len(q) > 50000:
            q = q.sample(50000, random_state=0)
        jitter = rng.normal(0, .05, len(q))
        ax.scatter(q.block.to_numpy() + jitter, q.sink_strength, s=6, alpha=.08, color=colors[typ], label=typ)
    ax.axhline(args.sink_threshold, ls="--", lw=1.2, color="0.35")
    ax.set(xlabel="Block", ylabel="Per-head incoming attention mass",
           title=f"{model} / {mode}: CLS ↔ patch sink handoff — HARD SINK events only")
    ax.set_xticks(range(N_BLOCKS)); ax.set_xlim(-.5,23.5); ax.set_ylim(args.sink_threshold-.03,1.02)
    ax.legend(title="Strongest key type")
    _savefig(fig, pdir / f"sink_transition_HARD_{model}_{mode}.png")

    # Role-coded hard-patch events: old register vs B20 local/intact vs other.
    pe = e[e.sink_type.eq("PATCH")].copy()
    pe["role"] = "other patch"
    pe.loc[pe.is_b20_newnorm_intact.eq(1), "role"] = "B20 new-norm (intact address)"
    pe.loc[pe.is_b20_newnorm_local.eq(1), "role"] = "B20 new-norm (local mode)"
    pe.loc[pe.is_b13_register.eq(1), "role"] = "B13 visible-register address"
    role_colors = {
        "B13 visible-register address": "tab:red",
        "B20 new-norm (local mode)": "tab:orange",
        "B20 new-norm (intact address)": "tab:purple",
        "other patch": "0.55",
    }
    fig, ax = plt.subplots(figsize=(14, 6))
    for role in role_colors:
        q = pe[pe.role.eq(role)]
        if len(q) > 40000: q = q.sample(40000, random_state=1)
        jitter = rng.normal(0,.05,len(q))
        ax.scatter(q.block+jitter, q.sink_strength, s=7, alpha=.12, color=role_colors[role], label=role)
    ax.set(xlabel="Block", ylabel="Per-head incoming mass", title=f"{model} / {mode}: HARD patch sinks by token role")
    ax.set_xticks(range(N_BLOCKS)); ax.set_xlim(-.5,23.5); ax.set_ylim(args.sink_threshold-.03,1.02)
    ax.legend(fontsize=8)
    _savefig(fig, pdir / f"sink_transition_roles_HARD_{model}_{mode}.png")


def plot_cls_patch_pies(pdir: Path, model: str, mode: str, h: pd.DataFrame):
    import matplotlib.pyplot as plt
    # 24 small pies is intentionally redundant: it makes the topology flip visually unavoidable.
    for variant in ("ALL_ARGMAX", "HARD_ABSOLUTE"):
        fig, axes = plt.subplots(4, 6, figsize=(15, 10))
        for b, ax in enumerate(axes.flat):
            q = h[h.block.eq(b)]
            if variant == "ALL_ARGMAX":
                vals = [int(q.sink_type.eq("CLS").sum()), int(q.sink_type.eq("PATCH").sum())]
                labels = ["CLS", "Patch"]
                colors = ["tab:blue", "tab:orange"]
            else:
                vals = [int((q.is_sink_event.eq(1)&q.sink_type.eq("CLS")).sum()),
                        int((q.is_sink_event.eq(1)&q.sink_type.eq("PATCH")).sum()),
                        int(q.is_sink_event.eq(0).sum())]
                labels = ["CLS sink", "Patch sink", "No hard sink"]
                colors = ["tab:blue", "tab:orange", "0.85"]
            if sum(vals):
                ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                       wedgeprops={"linewidth": .4, "edgecolor": "white"})
            ax.set_title(f"B{b}\n" + (f"patch={vals[1]/max(1,sum(vals)):.2f}" if variant=="ALL_ARGMAX" else f"hard={(vals[0]+vals[1])/max(1,sum(vals)):.2f}"), fontsize=9)
        handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=9, label=l)
                   for l,c in zip(labels, colors)]
        fig.legend(handles=handles, loc="lower center", ncol=len(labels), frameon=False)
        fig.suptitle(f"{model} / {mode}: CLS vs patch topology — {variant}", fontsize=14)
        fig.subplots_adjust(bottom=.08, top=.93, wspace=.05, hspace=.28)
        fig.savefig(pdir / f"cls_patch_pies_{variant}_{model}_{mode}.png", dpi=180)
        plt.close(fig)


def plot_head_maps(pdir: Path, model: str, mode: str, hs: pd.DataFrame):
    import matplotlib.pyplot as plt
    maps = [
        ("sink_frequency", "hard sink frequency", "head_specificity_HARD"),
        ("cls_hard_sink_rate_all", "CLS hard-sink frequency", "head_specificity_CLS_HARD"),
        ("patch_hard_sink_rate_all", "patch hard-sink frequency", "head_specificity_PATCH_HARD"),
        ("cls_argmax_rate_all", "CLS is strongest key (no cutoff)", "head_specificity_CLS_ALL"),
        ("patch_argmax_rate_all", "patch is strongest key (no cutoff)", "head_specificity_PATCH_ALL"),
        ("cls_nop_rate_all", "CLS NOP event frequency", "head_specificity_CLS_NOP"),
        ("patch_nop_rate_all", "patch NOP event frequency", "head_specificity_PATCH_NOP"),
        ("cls_broadcast_rate_all", "CLS broadcast event frequency", "head_specificity_CLS_BROADCAST"),
        ("patch_broadcast_rate_all", "patch broadcast event frequency", "head_specificity_PATCH_BROADCAST"),
    ]
    for col, label, stem in maps:
        piv = hs.pivot(index="head", columns="block", values=col).reindex(index=range(N_HEADS), columns=range(N_BLOCKS))
        fig, ax = plt.subplots(figsize=(13, 6))
        im = _heatmap(ax, piv.to_numpy(), f"{model} / {mode}: {label}", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label=label)
        _savefig(fig, pdir / f"{stem}_{model}_{mode}.png")

    # User-requested heads-on-X / blocks-on-Y map for CLS broadcast sinks.
    q = hs.copy()
    fig, ax = plt.subplots(figsize=(10, 9))
    sizes = 30 + 1800 * q.cls_broadcast_rate_all.to_numpy()
    sc = ax.scatter(q["head"], q.block, s=sizes, c=q.cls_broadcast_rate_all, cmap=SEQUENTIAL_CMAP, vmin=0,
                    vmax=max(.05, float(q.cls_broadcast_rate_all.max())), alpha=.8, edgecolors="0.25", linewidths=.25)
    ax.set(xlabel="Head", ylabel="Block", title=f"{model} / {mode}: heads that use CLS as a BROADCAST sink\nmarker area = absolute event frequency")
    ax.set_xticks(range(N_HEADS)); ax.set_yticks(range(N_BLOCKS)); ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, label="P(CLS broadcast event) across images")
    _savefig(fig, pdir / f"cls_broadcast_head_bubble_{model}_{mode}.png")


def plot_cls_sink_atlas(pdir: Path, model: str, mode: str, hs: pd.DataFrame, args):
    """CLS-only sink atlas.

    Generic sink plots are intentionally a democracy across all possible sink keys.
    This family conditions on the strongest key actually being CLS AND crossing the
    hard-sink threshold, then exposes which heads use CLS as NOP/BROADCAST/OTHER.

    No new forward-pass data is required; all rates are in sink_head_summary.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    q = hs.copy().sort_values(["block", "head"]).reset_index(drop=True)
    if len(q) == 0:
        return

    # Backward-compatible with existing v2 summaries: derive OTHER if older CSVs
    # were written before cls_other_rate_all was added explicitly.
    if "cls_other_rate_all" not in q.columns:
        q["cls_other_rate_all"] = np.clip(
            pd.to_numeric(q["cls_hard_sink_rate_all"], errors="coerce").fillna(0)
            - pd.to_numeric(q["cls_nop_rate_all"], errors="coerce").fillna(0)
            - pd.to_numeric(q["cls_broadcast_rate_all"], errors="coerce").fillna(0),
            0, 1,
        )

    rate_cols = {"NOP": "cls_nop_rate_all", "BROADCAST": "cls_broadcast_rate_all", "OTHER": "cls_other_rate_all"}
    regime_colors = REGIME_COLORS
    no_color = "#ECECEC"

    def mats():
        out = {}
        for reg, col in rate_cols.items():
            out[reg] = q.pivot(index="block", columns="head", values=col).reindex(index=range(N_BLOCKS), columns=range(N_HEADS)).fillna(0).to_numpy(float)
        hard = q.pivot(index="block", columns="head", values="cls_hard_sink_rate_all").reindex(index=range(N_BLOCKS), columns=range(N_HEADS)).fillna(0).to_numpy(float)
        return out, hard

    rm, hard = mats()
    stack = np.stack([rm["NOP"], rm["BROADCAST"], rm["OTHER"]], axis=-1)
    names = np.array(["NOP", "BROADCAST", "OTHER"], dtype=object)
    dom_idx = stack.argmax(axis=-1)
    dominant = names[dom_idx]

    # 1) Tall categorical CLS-only atlas.  Cell text/ruler is P(CLS hard sink),
    # while color answers *what kind of CLS sink* that head uses.
    fig, ax = plt.subplots(figsize=(9.5, 15.5))
    for b in range(N_BLOCKS):
        for h in range(N_HEADS):
            r = float(hard[b, h])
            if r <= 0:
                face = no_color
            else:
                face = regime_colors[str(dominant[b, h])]
            ax.add_patch(Rectangle((h-.5, b-.5), 1, 1, facecolor=face, edgecolor="0.45", linewidth=.35))
            if r > 0:
                txt = "." if r < .005 else f"{100*r:.0f}%"
                ax.text(h, b-.06, txt, ha="center", va="center", fontsize=6.3,
                        color="white" if face != no_color else "0.2", fontweight="bold" if r >= .05 else "normal")
                # Tiny black ruler = absolute CLS-sink incidence, independent of algorithm.
                ax.add_patch(Rectangle((h-.46, b+.34), .86*min(1.0, r), .055, facecolor="0.08", edgecolor="none"))
    ax.set_xlim(-.5, N_HEADS-.5); ax.set_ylim(N_BLOCKS-.5, -.5)
    ax.set_xticks(range(N_HEADS)); ax.set_yticks(range(N_BLOCKS))
    ax.set_xlabel("Head"); ax.set_ylabel("Block")
    ax.set_title(f"{model} / {mode}: CLS-only hard-sink atlas\ncolor = dominant CLS-sink algorithm; printed % + black ruler = P(CLS is hard sink)")
    handles = [Patch(facecolor=regime_colors[k], label=f"CLS {k} dominant") for k in ("NOP","BROADCAST","OTHER")]
    handles.append(Patch(facecolor=no_color, edgecolor="0.45", label="CLS never a hard sink"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01,1), fontsize=8, frameon=True)
    _savefig(fig, pdir/f"CLS_sink_HEADxBLOCK_DOMINANT_{model}_{mode}.png", 210)

    # 2) Mixed mosaic: within each cell, width encodes the unconditional P(image)
    # of CLS-NOP / CLS-BROADCAST / CLS-OTHER.  Remaining width = CLS is not a hard sink.
    # Unlike conditional regime composition, a one-off event cannot fill the whole cell.
    fig, ax = plt.subplots(figsize=(9.5, 15.5))
    for b in range(N_BLOCKS):
        for h in range(N_HEADS):
            ax.add_patch(Rectangle((h-.5,b-.5),1,1,facecolor=no_color,edgecolor="0.45",linewidth=.35))
            x0 = h-.5
            for reg in ("NOP","BROADCAST","OTHER"):
                w = float(rm[reg][b,h])
                if w > 0:
                    ax.add_patch(Rectangle((x0,b-.5),w,1,facecolor=regime_colors[reg],edgecolor="none"))
                    x0 += w
            r=float(hard[b,h])
            if r>0:
                txt="." if r<.005 else f"{100*r:.0f}%"
                ax.text(h,b,txt,ha="center",va="center",fontsize=6.1,color="0.08",fontweight="bold" if r>=.05 else "normal")
    ax.set_xlim(-.5,N_HEADS-.5); ax.set_ylim(N_BLOCKS-.5,-.5)
    ax.set_xticks(range(N_HEADS)); ax.set_yticks(range(N_BLOCKS)); ax.set_xlabel("Head"); ax.set_ylabel("Block")
    ax.set_title(f"{model} / {mode}: CLS-only sink mosaic\ncell widths = absolute image frequency of CLS NOP / BROADCAST / OTHER; pale remainder = CLS not hard sink")
    handles=[Patch(facecolor=regime_colors[k],label=f"CLS {k}") for k in ("NOP","BROADCAST","OTHER")]
    handles.append(Patch(facecolor=no_color,edgecolor="0.45",label="CLS not hard sink"))
    ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(1.01,1),fontsize=8)
    _savefig(fig,pdir/f"CLS_sink_HEADxBLOCK_MIXED_MOSAIC_{model}_{mode}.png",220)

    # 3) Three directly comparable absolute-rate maps. Generic scalar heatmaps use
    # viridis again; this plot is intentionally boring and quantitatively direct.
    fig, axes = plt.subplots(1,3,figsize=(16,8),sharey=True)
    for ax, reg in zip(axes,("NOP","BROADCAST","OTHER")):
        im=ax.imshow(rm[reg],origin="upper",aspect="auto",vmin=0,vmax=max(.05,float(stack.max())),cmap="viridis")
        ax.set_title(f"CLS {reg}: P(event)"); ax.set_xlabel("Head"); ax.set_xticks(range(N_HEADS))
    axes[0].set_ylabel("Block"); axes[0].set_yticks(range(N_BLOCKS))
    fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02,pad=.02,label="Fraction of images")
    fig.suptitle(f"{model} / {mode}: CLS hard-sink algorithm rates by head/block")
    fig.savefig(pdir/f"CLS_sink_HEADxBLOCK_RATE_TRIPTYCH_{model}_{mode}.png",dpi=190,bbox_inches="tight")
    plt.close(fig)

    # 4) Number of heads whose CLS-sink behavior is meaningfully present. This is
    # the clean human-readable 0..16 view requested in discussion.
    for threshold, label in [(0.0,"ANY"),(.05,"GE5PCT")]:
        rows=[]
        for b in range(N_BLOCKS):
            counts={k:0 for k in ("NOP","BROADCAST","OTHER")}
            active=0
            for h in range(N_HEADS):
                r=float(hard[b,h])
                if r > threshold:
                    active += 1
                    counts[str(dominant[b,h])] += 1
            rows.append((b,active,counts))
        fig, ax=plt.subplots(figsize=(13,6))
        x=np.arange(N_BLOCKS); bottom=np.zeros(N_BLOCKS)
        for reg in ("NOP","BROADCAST","OTHER"):
            vals=np.array([r[2][reg] for r in rows],dtype=float)
            ax.bar(x,vals,bottom=bottom,color=regime_colors[reg],label=reg,width=.8)
            bottom += vals
        for b,total,_ in rows:
            if total:
                ax.text(b,total+.15,str(total),ha="center",va="bottom",fontsize=7,fontweight="bold")
        ax.set(xlabel="Block",ylabel="Number of heads (0..16)",title=f"{model} / {mode}: heads that use CLS as a hard sink — {label}")
        ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(0,16.9); ax.legend(title="Dominant CLS-sink algorithm",ncol=3)
        _savefig(fig,pdir/f"CLS_sink_HEAD_COUNTS_by_block_{label}_{model}_{mode}.png",190)

    # 5) Per-head trajectories: preserves outliers completely; no head democracy.
    fig, axes=plt.subplots(4,4,figsize=(15,12),sharex=True,sharey=True)
    for h,ax in enumerate(axes.flat):
        z=q[q["head"].eq(h)].sort_values("block")
        ax.plot(z.block,z.cls_nop_rate_all,color=regime_colors["NOP"],lw=1.25,label="NOP")
        ax.plot(z.block,z.cls_broadcast_rate_all,color=regime_colors["BROADCAST"],lw=1.25,label="BROADCAST")
        zz=z["cls_other_rate_all"] if "cls_other_rate_all" in z.columns else np.clip(z.cls_hard_sink_rate_all-z.cls_nop_rate_all-z.cls_broadcast_rate_all,0,1)
        ax.plot(z.block,zz,color=regime_colors["OTHER"],lw=1.1,label="OTHER")
        ax.set_title(f"H{h}",fontsize=9); ax.set_xticks([0,4,8,12,16,20,23]); ax.grid(alpha=.12,lw=.5)
    axes[0,0].set_ylim(0,1); fig.supxlabel("Block"); fig.supylabel("P(CLS hard-sink event)")
    handles=[plt.Line2D([0],[0],color=regime_colors[k],lw=2,label=k) for k in ("NOP","BROADCAST","OTHER")]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False)
    fig.suptitle(f"{model} / {mode}: CLS-sink algorithm trajectory of every head",fontsize=14)
    fig.subplots_adjust(bottom=.07,top=.93,hspace=.35)
    fig.savefig(pdir/f"CLS_sink_PER_HEAD_TRAJECTORIES_{model}_{mode}.png",dpi=190,bbox_inches="tight")
    plt.close(fig)

def plot_cls_broadcast_violin(pdir: Path, model: str, mode: str, h: pd.DataFrame):
    import matplotlib.pyplot as plt
    e = h[(h.is_sink_event.eq(1)) & h.sink_type.eq("CLS") & h.regime.eq("BROADCAST")]
    if not len(e):
        return
    data, pos, counts = [], [], []
    for head in range(N_HEADS):
        v = e.loc[e["head"].eq(head), "sink_strength"].to_numpy(np.float64)
        if len(v):
            data.append(v); pos.append(head); counts.append(len(v))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(12,6))
    vp = ax.violinplot(data, positions=pos, widths=.75, showmeans=False, showmedians=True, showextrema=False)
    for body in vp['bodies']:
        body.set_alpha(.35)
    for x, n in zip(pos, counts):
        ax.text(x, .505, str(n), rotation=90, ha='center', va='bottom', fontsize=6, color='0.35')
    ax.set(xlabel="Head", ylabel="CLS broadcast sink strength",
           title=f"{model} / {mode}: CLS BROADCAST sink strength by head\ncounts printed at cutoff line; block location is shown in the companion bubble map")
    ax.set_xticks(range(N_HEADS)); ax.axhline(PAPER_HEAD_SINK_THRESHOLD, color='0.5', ls='--', lw=.8)
    ax.set_ylim(.49,1.01)
    _savefig(fig,pdir/f"cls_broadcast_strength_violin_by_head_{model}_{mode}.png")

def plot_nop_broadcast_scatter(pdir: Path, model: str, mode: str, hs: pd.DataFrame, args):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    variants = [
        ("PERSISTENT", hs[(hs.sink_frequency >= args.persistent_head_frequency) & hs.value_norm_ratio_median.notna() & hs.stable_rank_residual_median.notna()]),
        ("ANY_HARD_SINK", hs[(hs.n_sink > 0) & hs.value_norm_ratio_median.notna() & hs.stable_rank_residual_median.notna()]),
    ]
    for name, q in variants:
        fig, ax = plt.subplots(figsize=(10, 7.5))
        if len(q):
            # Frequency controls marker size in the unrestricted plot; persistent remains easy to read.
            sizes = 35 + (220 if name == "ANY_HARD_SINK" else 80) * q.sink_frequency.to_numpy()
            sc = ax.scatter(q.stable_rank_residual_median, q.value_norm_ratio_median,
                            c=q.block, cmap=BLOCK_CMAP, s=sizes, marker="o", alpha=.82)
            patch = q.majority_sink_type.eq("PATCH")
            if patch.any():
                ax.scatter(q.loc[patch, "stable_rank_residual_median"], q.loc[patch, "value_norm_ratio_median"],
                           c=q.loc[patch, "block"], cmap=BLOCK_CMAP, s=sizes[patch.to_numpy()]+25,
                           marker="x", linewidths=1.6)
            fig.colorbar(sc, ax=ax, label="Block")
        ax.axhline(PAPER_NOP_VALUE_RATIO, ls="--", lw=1.4, color="crimson")
        ax.axvline(PAPER_RANK1_STABLE_RANK, ls="--", lw=1.4, color="dodgerblue")
        ax.set(xlabel="Stable rank of per-head residual update", ylabel="Sink value norm ratio",
               title=f"{model} / {mode}: NOP vs Broadcast — {name}")
        subtitle = (f"head/layer included iff hard-sink frequency ≥ {args.persistent_head_frequency:g}"
                    if name == "PERSISTENT" else "every head/layer with ≥1 hard-sink image; marker size ∝ sink frequency")
        ax.text(.01, .01, subtitle, transform=ax.transAxes, fontsize=8, va="bottom", color="0.35")
        ax.legend(handles=[
            Line2D([0],[0],marker='o',color='k',lw=0,label='CLS-majority'),
            Line2D([0],[0],marker='x',color='k',lw=0,label='Patch-majority'),
            Line2D([0],[0],ls='--',color='crimson',label='NOP value-ratio threshold'),
            Line2D([0],[0],ls='--',color='dodgerblue',label='broadcast stable-rank threshold'),
        ], loc="best", fontsize=8)
        _savefig(fig, pdir / f"nop_vs_broadcast_{name}_{model}_{mode}.png", 190)


def plot_nop_broadcast_mechanism_clear(pdir: Path, model: str, mode: str, h: pd.DataFrame, args):
    """Plot NOP vs BROADCAST in a deliberately human-readable form.

    The paper's two mechanisms differ in *different axes*:
      NOP       -> sink V norm is near zero
      BROADCAST -> V is nontrivial but residual update is approximately rank-1

    So this view uses fixed regime colors and block-level summaries rather than
    thousands of block-colored head points.
    """
    import matplotlib.pyplot as plt
    e = h[(h.is_sink_event.eq(1)) & h.regime.isin(["NOP", "BROADCAST", "OTHER"])].copy()
    if not len(e):
        return

    # 1) Three-panel mechanism summary by block.
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    denom = max(1, int(h.groupby(["block","head"]).size().groupby("block").sum().median()))
    # Use exact per-block denominator, not the median above; denominator variable only documents intent.
    for regime in ("NOP", "BROADCAST", "OTHER"):
        color = REGIME_COLORS[regime]
        rows = []
        for b in range(N_BLOCKS):
            hb = h[h.block.eq(b)]
            eb = e[(e.block.eq(b)) & e.regime.eq(regime)]
            rows.append((
                b,
                len(eb) / max(1, len(hb)),
                float(np.nanmedian(eb.value_norm_ratio)) if len(eb) else np.nan,
                float(np.nanmedian(eb.stable_rank_residual)) if len(eb) else np.nan,
            ))
        rr = pd.DataFrame(rows, columns=["block","rate","vnorm","srank"])
        axes[0].plot(rr.block, rr.rate, marker="o", lw=2, color=color, label=regime)
        axes[1].plot(rr.block, rr.vnorm, marker="o", lw=2, color=color, label=regime)
        axes[2].plot(rr.block, rr.srank, marker="o", lw=2, color=color, label=regime)

    axes[0].set_ylabel("Absolute event rate\n(all head×image obs.)")
    axes[0].set_title("How often each sink algorithm occurs")
    axes[1].axhline(PAPER_NOP_VALUE_RATIO, color="0.25", ls="--", lw=1.2, label=f"NOP V threshold = {PAPER_NOP_VALUE_RATIO:g}")
    axes[1].set_ylabel("Median sink V-norm ratio")
    axes[1].set_title("NOP signature: sink carries approximately zero V magnitude")
    axes[2].axhline(PAPER_RANK1_STABLE_RANK, color="0.25", ls="--", lw=1.2, label=f"rank-1 threshold = {PAPER_RANK1_STABLE_RANK:g}")
    axes[2].set_ylabel("Median residual-update stable rank")
    axes[2].set_title("BROADCAST signature: nonzero V, approximately rank-1 residual update")
    axes[2].set_xlabel("Block"); axes[2].set_xticks(range(N_BLOCKS))
    axes[0].legend(ncol=3); axes[1].legend(fontsize=8); axes[2].legend(fontsize=8)
    fig.suptitle(f"{model} / {mode}: NOP vs BROADCAST — mechanism overview", fontsize=15)
    _savefig(fig, pdir / f"nop_vs_broadcast_MECHANISM_OVERVIEW_{model}_{mode}.png", 190)

    # 2) One point per (block, regime): retains the paper's two diagnostic axes but
    # removes the head-level point explosion.  Fixed colors mean the mechanism is
    # readable immediately; text labels say which block generated each point.
    rows = []
    for (b, regime), q in e.groupby(["block", "regime"], sort=True):
        rows.append({
            "block": int(b), "regime": regime, "n": len(q),
            "vnorm": float(np.nanmedian(q.value_norm_ratio)),
            "srank": float(np.nanmedian(q.stable_rank_residual)),
            "rate": len(q) / max(1, len(h[h.block.eq(b)])),
        })
    rr = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 8))
    for regime in ("NOP", "BROADCAST", "OTHER"):
        q = rr[rr.regime.eq(regime)]
        if not len(q): continue
        sizes = 60 + 1600*q.rate.to_numpy(float)
        ax.scatter(q.srank, q.vnorm, s=sizes, color=REGIME_COLORS[regime], alpha=.78,
                   edgecolor="0.15", linewidth=.45, label=regime)
        for r in q.itertuples(index=False):
            if r.rate >= .002 or regime != "OTHER":
                ax.annotate(f"B{r.block}", (r.srank, r.vnorm), xytext=(3,3), textcoords="offset points", fontsize=6, color="0.2")
    ax.axhline(PAPER_NOP_VALUE_RATIO, ls="--", lw=1.3, color="0.25")
    ax.axvline(PAPER_RANK1_STABLE_RANK, ls="--", lw=1.3, color="0.25")
    ax.text(.01,.99,"NOP: below horizontal line",transform=ax.transAxes,va="top",ha="left",fontsize=9,color=REGIME_COLORS["NOP"])
    ax.text(.99,.99,"BROADCAST: left of vertical, above horizontal",transform=ax.transAxes,va="top",ha="right",fontsize=9,color=REGIME_COLORS["BROADCAST"])
    ax.set(xlabel="Median residual-update stable rank", ylabel="Median sink V-norm ratio",
           title=f"{model} / {mode}: NOP vs BROADCAST — one point per block/regime\nmarker area ∝ absolute event rate")
    ax.legend(title="Regime", fontsize=9)
    _savefig(fig, pdir / f"nop_vs_broadcast_BLOCK_REGIME_CENTROIDS_{model}_{mode}.png", 190)


def plot_massive_ln_scatter(pdir: Path, model: str, mode: str, h: pd.DataFrame, args):
    import matplotlib.pyplot as plt
    for hard in (False, True):
        q = h[h.is_sink_event.eq(1)] if hard else h
        if len(q) > 50000: q = q.sample(50000, random_state=0)
        name = "HARD" if hard else "ALL"
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sc = axes[0].scatter(q.sink_raw_norm_ratio, q.sink_strength, c=q.block, cmap=BLOCK_CMAP, s=8, alpha=.30)
        axes[0].axhline(args.sink_threshold, ls="--", color="0.35", lw=1)
        axes[0].set(xlabel="Raw residual norm ratio: strongest key / token mean", ylabel="Incoming sink strength",
                    title="Residual massiveness vs sinkness")
        axes[1].scatter(q.sink_ln1_norm_ratio, q.sink_strength, c=q.block, cmap=BLOCK_CMAP, s=8, alpha=.30)
        axes[1].axhline(args.sink_threshold, ls="--", color="0.35", lw=1)
        axes[1].set(xlabel="LN1 output norm ratio: strongest key / token mean", ylabel="Incoming sink strength",
                    title="LN1 norm vs sinkness")
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="Block")
        fig.suptitle(f"{model} / {mode}: massiveness, LN and sinkness — {name}")
        fig.savefig(pdir / f"massive_and_LN_vs_sink_{name}_{model}_{mode}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_role_overlap(pdir: Path, model: str, mode: str, h: pd.DataFrame):
    import matplotlib.pyplot as plt
    e = h[(h.is_sink_event.eq(1)) & h.sink_type.eq("PATCH")]
    rows = []
    for b, q in e.groupby("block"):
        rows.append({
            "block": int(b), "b13": float(q.is_b13_register.mean()),
            "b20_intact": float(q.is_b20_newnorm_intact.mean()),
            "b20_local": float(q.is_b20_newnorm_local.mean()), "n": len(q)
        })
    if not rows: return
    rr = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(rr.block, rr.b13, marker="o", label="patch sink lands on intact B13 register address")
    ax.plot(rr.block, rr.b20_local, marker="o", color="tab:orange", label="patch sink lands on LOCAL B20 new-norm address")
    ax.plot(rr.block, rr.b20_intact, marker="o", ls="--", label="patch sink lands on INTACT B20 new-norm address")
    ax.set(xlabel="Block", ylabel="Fraction of hard PATCH-sink events", title=f"{model} / {mode}: which promoted population receives sink attention?")
    ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(0,1); ax.legend(fontsize=8)
    _savefig(fig, pdir / f"sink_role_overlap_{model}_{mode}.png")


def plot_regime_rates(pdir: Path, model: str, ls: pd.DataFrame):
    import matplotlib.pyplot as plt
    modes = sorted(ls["mode"].unique())

    # PRESERVED: combined absolute fraction of ALL head/image observations.
    fig, ax = plt.subplots(figsize=(13, 6))
    for mode in modes:
        q = ls[ls["mode"].eq(mode)]
        ax.plot(q.block, q.head_nop_rate_all, marker="o", label=f"{mode}: NOP / all")
        ax.plot(q.block, q.head_broadcast_rate_all, marker="s", ls="--", label=f"{mode}: broadcast / all")
    ax.set(xlabel="Block", ylabel="Absolute fraction of ALL head×image observations",
           title=f"{model}: NOP / broadcast absolute event rates (FIXED denominator)")
    ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(0,1); ax.legend(ncol=2, fontsize=8)
    _savefig(fig, pdir / f"regime_rate_ABSOLUTE_by_block_{model}.png")

    # PRESERVED: old combined conditional view, honestly named.
    fig, ax = plt.subplots(figsize=(13, 6))
    for mode in modes:
        q = ls[ls["mode"].eq(mode)]
        ax.plot(q.block, q.head_nop_fraction_given_sink, marker="o", label=f"{mode}: NOP | hard sink")
        ax.plot(q.block, q.head_broadcast_fraction_given_sink, marker="s", ls="--", label=f"{mode}: broadcast | hard sink")
    ax.set(xlabel="Block", ylabel="Fraction conditional on hard-sink events",
           title=f"{model}: regime composition GIVEN a hard sink (old view, relabeled)")
    ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(0,1); ax.legend(ncol=2, fontsize=8)
    _savefig(fig, pdir / f"regime_composition_GIVEN_SINK_by_block_{model}.png")

    # NEW: split intact/no-pump so curves can never hide beneath one another.
    for mode in modes:
        q = ls[ls["mode"].eq(mode)].sort_values("block")

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(q.block, q.head_nop_fraction_given_sink, marker="o", lw=2,
                color=REGIME_COLORS["NOP"], label="NOP | hard sink")
        ax.plot(q.block, q.head_broadcast_fraction_given_sink, marker="s", lw=2,
                color=REGIME_COLORS["BROADCAST"], label="BROADCAST | hard sink")
        other = 1.0 - q.head_nop_fraction_given_sink.fillna(0) - q.head_broadcast_fraction_given_sink.fillna(0)
        # Other is meaningful only where hard sinks actually exist.
        other = other.where(q.head_sink_event_rate > 0, np.nan)
        ax.plot(q.block, other, marker="^", lw=1.5, color=REGIME_COLORS["OTHER"], label="OTHER | hard sink")
        ax.set(xlabel="Block", ylabel="Fraction conditional on hard-sink events",
               title=f"{model} / {mode}: NOP vs BROADCAST GIVEN a hard sink — separated view")
        ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(0,1); ax.legend(fontsize=9, ncol=3)
        _savefig(fig, pdir / f"regime_composition_GIVEN_SINK_by_block_{model}_{mode}.png")

        # NEW: absolute stacked bars.  This simultaneously shows algorithm and incidence,
        # preventing a block with one rare NOP sink from looking as important as a block
        # where half of all head×image observations are NOP sinks.
        nop = q.head_nop_rate_all.to_numpy(float)
        broad = q.head_broadcast_rate_all.to_numpy(float)
        oth = q.head_other_sink_rate_all.to_numpy(float)
        none = np.clip(1.0 - nop - broad - oth, 0.0, 1.0)
        x = q.block.to_numpy(int)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x, nop, color=REGIME_COLORS["NOP"], label="NOP hard-sink events")
        ax.bar(x, broad, bottom=nop, color=REGIME_COLORS["BROADCAST"], label="BROADCAST hard-sink events")
        ax.bar(x, oth, bottom=nop+broad, color=REGIME_COLORS["OTHER"], label="OTHER hard-sink events")
        ax.bar(x, none, bottom=nop+broad+oth, color="#D8D8D8", alpha=.45, label="No hard sink")
        ax.set(xlabel="Block", ylabel="Fraction of ALL head×image observations",
               title=f"{model} / {mode}: absolute sink-algorithm composition")
        ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(0,1); ax.legend(fontsize=8, ncol=4, loc="upper center")
        _savefig(fig, pdir / f"regime_composition_ABSOLUTE_STACKED_by_block_{model}_{mode}.png")


def _head_regime_arrays(hsm: pd.DataFrame):
    """Return dense [block, head] absolute rates for NOP/BROADCAST/OTHER/hard sink."""
    shp = (N_BLOCKS, N_HEADS)
    out = {k: np.zeros(shp, dtype=np.float64) for k in ("NOP", "BROADCAST", "OTHER", "SINK")}
    for r in hsm.itertuples(index=False):
        b, h = int(r.block), int(r.head)
        if not (0 <= b < N_BLOCKS and 0 <= h < N_HEADS):
            continue
        out["NOP"][b,h] = float(getattr(r, "nop_rate_all", 0.0) or 0.0)
        out["BROADCAST"][b,h] = float(getattr(r, "broadcast_rate_all", 0.0) or 0.0)
        out["OTHER"][b,h] = float(getattr(r, "other_sink_rate_all", 0.0) or 0.0)
        out["SINK"][b,h] = float(getattr(r, "sink_frequency", 0.0) or 0.0)
    return out


def _alpha_for_frequency(freq: float) -> float:
    """Keep rare cells visible without making a 0.2% event look as strong as a 90% event."""
    if not np.isfinite(freq) or freq <= 0:
        return 0.0
    return float(np.clip(0.28 + 0.72*np.sqrt(freq), 0.28, 1.0))


def plot_head_block_regime_atlas(pdir: Path, model: str, mode: str, hsm: pd.DataFrame, args):
    """Human-facing head×block atlas.

    This intentionally uses position for head identity and color/hatch for mechanism.
    It also emits an exact mixed-composition mosaic, a head-count summary, a 16-panel
    trajectory view, and a deliberately redundant 16-head-color/hatch view for visual
    archaeology.  User asked for EVERYTHING. :P
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.colors import to_rgb

    A = _head_regime_arrays(hsm)
    nop, bro, oth, sink = A["NOP"], A["BROADCAST"], A["OTHER"], A["SINK"]
    rates = np.stack([nop, bro, oth], axis=-1)
    reg_names = np.array(["NOP", "BROADCAST", "OTHER"], dtype=object)
    dominant_idx = np.argmax(rates, axis=-1)
    dominant = reg_names[dominant_idx]
    dominant = dominant.astype(object)
    dominant[sink <= 0] = "NONE"

    # ------------------------------------------------------------------
    # 1) Main phone-ish categorical matrix: heads X, blocks Y.
    # Fixed categorical color = mechanism; darkness + printed % = incidence.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 16.5))
    none_color = "#E1E1E1"
    for b in range(N_BLOCKS):
        for h in range(N_HEADS):
            f = sink[b,h]
            reg = dominant[b,h]
            if reg == "NONE":
                fc = none_color
                ax.add_patch(Rectangle((h-.5,b-.5),1,1,facecolor=fc,edgecolor="#222222",
                                       linewidth=.35,alpha=.72))
            else:
                fc = REGIME_COLORS[str(reg)]
                ax.add_patch(Rectangle((h-.5,b-.5),1,1,facecolor=fc,edgecolor="#222222",
                                       linewidth=.35,alpha=.94))
                # Small black frequency ruler along the bottom of the cell.  This keeps
                # categorical colors saturated (no more almost-white rare events) while
                # still showing absolute incidence without relying only on text.
                ax.add_patch(Rectangle((h-.46,b+.37),.92*min(1.0,f),.07,facecolor="#161616",
                                       edgecolor="none",alpha=.90))
            if f >= .01:
                txt_color = "white" if reg != "NONE" else "#111111"
                ax.text(h,b,f"{100*f:.0f}%",ha="center",va="center",fontsize=6.2,
                        color=txt_color,fontweight="bold" if f >= .20 else "normal")
            elif f > 0:
                ax.text(h,b,"·",ha="center",va="center",fontsize=8,color="white" if reg!="NONE" else "#111111")
    ax.set_xlim(-.5,N_HEADS-.5); ax.set_ylim(N_BLOCKS-.5,-.5)
    ax.set_xticks(range(N_HEADS)); ax.set_yticks(range(N_BLOCKS))
    ax.set_xlabel("Head"); ax.set_ylabel("Block")
    ax.set_title(f"{model} / {mode}: head × block sink-regime atlas\n"
                 "color = dominant algorithm; printed % + black ruler = hard-sink incidence", pad=14)
    ax.legend(handles=[
        Patch(facecolor=REGIME_COLORS["NOP"],label="NOP dominant"),
        Patch(facecolor=REGIME_COLORS["BROADCAST"],label="BROADCAST dominant"),
        Patch(facecolor=REGIME_COLORS["OTHER"],label="OTHER dominant"),
        Patch(facecolor=none_color,label="no hard-sink event"),
    ],loc="upper left",bbox_to_anchor=(1.005,1.0),ncol=1,fontsize=8,frameon=True)
    _savefig(fig,pdir/f"regime_HEADxBLOCK_DOMINANT_{model}_{mode}.png",200)

    # ------------------------------------------------------------------
    # 2) Exact mixed-composition mosaic.  Every cell is a mini stacked bar
    # conditional on that head actually having a hard sink.  Opacity + annotation
    # still encode the absolute event rate, so a rare mixed head cannot look huge.
    # NOP gets horizontal hatch, BROADCAST is solid, OTHER cross-hatched.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5,16.5))
    hatch = {"NOP":"---", "BROADCAST":"", "OTHER":"xx"}
    for b in range(N_BLOCKS):
        for h in range(N_HEADS):
            f = sink[b,h]
            if f <= 0:
                ax.add_patch(Rectangle((h-.5,b-.5),1,1,facecolor=none_color,edgecolor="#555",
                                       linewidth=.25,alpha=.55))
                continue
            vals = rates[b,h]
            den = vals.sum()
            cond = vals/den if den > 0 else np.zeros(3)
            y0 = b-.5
            for reg, frac in zip(reg_names,cond):
                if frac <= 0: continue
                ax.add_patch(Rectangle((h-.5,y0),1,float(frac),facecolor=REGIME_COLORS[str(reg)],
                                       edgecolor="#222",linewidth=.25,hatch=hatch[str(reg)],alpha=.94))
                y0 += float(frac)
            # same black ruler = absolute hard-sink incidence; cell partition itself is conditional.
            ax.add_patch(Rectangle((h-.46,b+.37),.92*min(1.0,f),.07,facecolor="#161616",edgecolor="none",alpha=.90))
            if f >= .01:
                ax.text(h,b,f"{100*f:.0f}%",ha="center",va="center",fontsize=5.8,
                        color="white",fontweight="bold" if f>=.2 else "normal")
    ax.set_xlim(-.5,N_HEADS-.5); ax.set_ylim(N_BLOCKS-.5,-.5)
    ax.set_xticks(range(N_HEADS)); ax.set_yticks(range(N_BLOCKS)); ax.set_xlabel("Head"); ax.set_ylabel("Block")
    ax.set_title(f"{model} / {mode}: exact per-head regime mixture\n"
                 "cell partition = P(regime | hard sink); printed % = P(hard sink)")
    ax.legend(handles=[
        Patch(facecolor=REGIME_COLORS["NOP"],hatch="---",label="NOP"),
        Patch(facecolor=REGIME_COLORS["BROADCAST"],label="BROADCAST"),
        Patch(facecolor=REGIME_COLORS["OTHER"],hatch="xx",label="OTHER"),
        Patch(facecolor=none_color,label="no hard sink"),
    ],loc="upper left",bbox_to_anchor=(1.005,1.0),ncol=1,fontsize=8)
    _savefig(fig,pdir/f"regime_HEADxBLOCK_MIXED_MOSAIC_{model}_{mode}.png",220)

    # ------------------------------------------------------------------
    # 3) Three directly comparable rate maps.  Same 0..1 scale, same dark cmap,
    # so NOP vs BROADCAST vs OTHER can be eyeballed without point-cloud archaeology.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1,3,figsize=(20,7),sharex=True,sharey=True)
    for ax, reg, mat in zip(axes,("NOP","BROADCAST","OTHER"),(nop,bro,oth)):
        im=ax.imshow(mat.T,origin="lower",aspect="auto",vmin=0,vmax=1,cmap=SEQUENTIAL_CMAP)
        ax.set_title(f"{reg}: P(event | image)"); ax.set_xlabel("Block"); ax.set_xticks(range(N_BLOCKS))
        ax.set_yticks(range(N_HEADS)); ax.set_ylabel("Head")
        fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    fig.suptitle(f"{model} / {mode}: per-head sink-algorithm event rates",fontsize=15)
    _savefig(fig,pdir/f"regime_HEADxBLOCK_RATE_TRIPTYCH_{model}_{mode}.png",190)

    # ------------------------------------------------------------------
    # 4) Count heads rather than head×image fractions.  Emit ANY and >=5% views.
    # A head is assigned to its most common sink algorithm if it clears threshold.
    # ------------------------------------------------------------------
    for label, minfreq in (("ANY",0.0),("GE5PCT",float(getattr(args,"head_regime_min_frequency",.05)))):
        counts={r:[] for r in ("NOP","BROADCAST","OTHER","NONE")}
        for b in range(N_BLOCKS):
            c={r:0 for r in counts}
            for h in range(N_HEADS):
                f=sink[b,h]
                active=(f>0) if minfreq<=0 else (f>=minfreq)
                if not active: c["NONE"]+=1
                else: c[str(reg_names[int(np.argmax(rates[b,h]))])]+=1
            for r in counts: counts[r].append(c[r])
        x=np.arange(N_BLOCKS)
        fig,ax=plt.subplots(figsize=(14,7))
        bottom=np.zeros(N_BLOCKS)
        for reg in ("NOP","BROADCAST","OTHER"):
            y=np.asarray(counts[reg],float)
            ax.bar(x,y,bottom=bottom,color=REGIME_COLORS[reg],label=reg)
            for bx,yy,bb in zip(x,y,bottom):
                if yy>=1: ax.text(bx,bb+yy/2,f"{int(yy)}",ha="center",va="center",fontsize=7,color="white",fontweight="bold")
            bottom+=y
        y=np.asarray(counts["NONE"],float)
        ax.bar(x,y,bottom=bottom,color=none_color,label=("no hard-sink event" if minfreq<=0 else f"hard-sink frequency < {minfreq:.0%}"))
        for bx,total in zip(x,bottom):
            if total>0: ax.text(bx,total+.15,f"{int(total)} active",ha="center",va="bottom",fontsize=6.5,color="#333")
        ax.set_ylim(0,17.7); ax.set_yticks(range(0,17,2)); ax.set_xticks(range(N_BLOCKS))
        ax.set(xlabel="Block",ylabel="Number of heads (out of 16)",
               title=f"{model} / {mode}: dominant sink algorithm by head — {label}")
        ax.legend(ncol=4,fontsize=8,loc="upper center")
        _savefig(fig,pdir/f"regime_HEAD_COUNTS_by_block_{label}_{model}_{mode}.png",190)

        # Active-only zoom: same counts, but omit the inactive gray mass and crop y to
        # the maximum number of active heads.  This is the literal "how many of 16 heads
        # are NOP/BROADCAST/OTHER here?" view.
        active=np.asarray(counts["NOP"],float)+np.asarray(counts["BROADCAST"],float)+np.asarray(counts["OTHER"],float)
        top=max(1.5,float(active.max())+1.2)
        fig,ax=plt.subplots(figsize=(14,7.5))
        bottom=np.zeros(N_BLOCKS)
        for reg in ("NOP","BROADCAST","OTHER"):
            y=np.asarray(counts[reg],float)
            ax.bar(x,y,bottom=bottom,color=REGIME_COLORS[reg],label=reg)
            for bx,yy,bb in zip(x,y,bottom):
                if yy>=1: ax.text(bx,bb+yy/2,f"{int(yy)}",ha="center",va="center",fontsize=7,color="white",fontweight="bold")
            bottom+=y
        for bx,total in zip(x,active):
            if total>0: ax.text(bx,total+.10,f"{int(total)}",ha="center",va="bottom",fontsize=7,color="#222",fontweight="bold")
        ax.set_ylim(0,top); ax.set_xticks(range(N_BLOCKS)); ax.set_xlabel("Block")
        ax.set_ylabel("Number of active heads (out of 16)")
        ax.set_title(f"{model} / {mode}: ACTIVE heads by dominant sink algorithm — {label} (zoom)")
        ax.legend(ncol=3,fontsize=8,loc="upper center")
        _savefig(fig,pdir/f"regime_HEAD_COUNTS_ACTIVE_ONLY_by_block_{label}_{model}_{mode}.png",195)

    # ------------------------------------------------------------------
    # 5) 16 small multiples: each head's absolute NOP/BROADCAST/OTHER rates over depth.
    # Common y scale preserves head-to-head comparison.
    # ------------------------------------------------------------------
    ymax=float(np.nanmax(rates)) if np.isfinite(rates).any() else 1.0
    ymax=min(1.0,max(.05,ymax*1.05))
    fig,axes=plt.subplots(4,4,figsize=(16,13),sharex=True,sharey=True)
    for head,ax in enumerate(axes.flat):
        ax.plot(range(N_BLOCKS),nop[:,head],color=REGIME_COLORS["NOP"],lw=1.4,label="NOP")
        ax.plot(range(N_BLOCKS),bro[:,head],color=REGIME_COLORS["BROADCAST"],lw=1.4,label="BROADCAST")
        ax.plot(range(N_BLOCKS),oth[:,head],color=REGIME_COLORS["OTHER"],lw=1.15,label="OTHER")
        ax.set_title(f"H{head}",fontsize=9); ax.set_ylim(0,ymax); ax.set_xlim(0,23)
        ax.set_xticks([0,4,8,12,16,20,23]); ax.grid(alpha=.16,lw=.5)
    axes[0,0].legend(fontsize=7,ncol=3,loc="upper left")
    fig.supxlabel("Block"); fig.supylabel("Absolute event rate across images")
    fig.suptitle(f"{model} / {mode}: per-head NOP / BROADCAST / OTHER trajectories",fontsize=15)
    _savefig(fig,pdir/f"regime_PER_HEAD_TRAJECTORIES_{model}_{mode}.png",190)

    # ------------------------------------------------------------------
    # 6) Deliberately redundant realization of the user's original 16-color idea.
    # Position still identifies the head, but each head also owns a stable color;
    # marker shape identifies dominant mechanism and marker area = event frequency.
    # This is allowed to be visually exuberant; the matrices above are the serious plots.
    # ------------------------------------------------------------------
    from colorsys import hsv_to_rgb
    head_colors=[hsv_to_rgb(i/N_HEADS,.72,.72) for i in range(N_HEADS)]
    markers={"NOP":"_","BROADCAST":"s","OTHER":"x"}
    fig,ax=plt.subplots(figsize=(10.5,16.5))
    for b in range(N_BLOCKS):
        for h in range(N_HEADS):
            f=sink[b,h]
            if f<=0: continue
            reg=str(dominant[b,h])
            size=35+520*f
            # '_' is literally a horizontal line = NOP; solid square = BROADCAST;
            # x = OTHER.  Mixed composition is visible in the exact mosaic companion.
            ax.scatter(h,b,s=size,c=[head_colors[h]],marker=markers[reg],linewidths=2 if reg!="BROADCAST" else .4,
                       edgecolors="#1a1a1a" if reg=="BROADCAST" else None,alpha=.9)
    ax.set_xlim(-.5,N_HEADS-.5); ax.set_ylim(N_BLOCKS-.5,-.5)
    ax.set_xticks(range(N_HEADS)); ax.set_yticks(range(N_BLOCKS)); ax.set_xlabel("Head (also uniquely colored)"); ax.set_ylabel("Block")
    ax.set_title(f"{model} / {mode}: 16-head color / regime glyph view\n"
                 "horizontal line = NOP, solid square = BROADCAST, × = OTHER; size = frequency")
    ax.grid(alpha=.15,lw=.5)
    _savefig(fig,pdir/f"regime_HEAD_ID_GLYPHS_{model}_{mode}.png",200)


def plot_absolute_regime_zoom(pdir: Path, model: str, ls: pd.DataFrame):
    """Readable companion to the preserved full 0..1 absolute composition plot."""
    import matplotlib.pyplot as plt
    for mode in sorted(ls["mode"].unique()):
        q=ls[ls["mode"].eq(mode)].sort_values("block")
        nop=q.head_nop_rate_all.to_numpy(float)
        bro=q.head_broadcast_rate_all.to_numpy(float)
        oth=q.head_other_sink_rate_all.to_numpy(float)
        total=nop+bro+oth
        x=q.block.to_numpy(int)
        maxv=float(np.nanmax(total)) if len(total) else 0.0
        top=min(1.0,max(.08,maxv*1.28+.018))
        fig,ax=plt.subplots(figsize=(14,7.5))
        ax.bar(x,nop,color=REGIME_COLORS["NOP"],label="NOP")
        ax.bar(x,bro,bottom=nop,color=REGIME_COLORS["BROADCAST"],label="BROADCAST")
        ax.bar(x,oth,bottom=nop+bro,color=REGIME_COLORS["OTHER"],label="OTHER")
        for i,b in enumerate(x):
            if total[i]>0:
                ax.text(b,total[i]+top*.012,f"{100*total[i]:.1f}%",ha="center",va="bottom",fontsize=7,color="#222",fontweight="bold")
            bottoms=(0,nop[i],nop[i]+bro[i]); vals=(nop[i],bro[i],oth[i])
            for reg,bb,v in zip(("NOP","BROADCAST","OTHER"),bottoms,vals):
                if v>=max(.008,top*.045):
                    ax.text(b,bb+v/2,f"{100*v:.1f}",ha="center",va="center",fontsize=6.2,color="white")
        ax.set_ylim(0,top); ax.set_xticks(range(N_BLOCKS)); ax.set_xlabel("Block")
        ax.set_ylabel("Fraction of ALL head×image observations")
        ax.set_title(f"{model} / {mode}: absolute sink-algorithm composition — ZOOM + annotations\n"
                     "numbers above bars = total hard-sink rate; segment labels are percentage points")
        ax.legend(ncol=3,loc="upper center",fontsize=8)
        _savefig(fig,pdir/f"regime_composition_ABSOLUTE_SINK_ONLY_ZOOM_ANNOTATED_{model}_{mode}.png",200)

def plot_cls_query(pdir: Path, model: str, mode: str, hs: pd.DataFrame, ls: pd.DataFrame):
    import matplotlib.pyplot as plt
    q = hs[hs["mode"].eq(mode)]
    metrics = [
        ("cls_attn_to_b13_regs_mean_all", "CLS→B13 register attention mass", "cls_to_B13reg_ATTN"),
        ("cls_av_resid_to_b13_regs_frac_mean_all", "CLS→B13 register |A·V| fraction", "cls_to_B13reg_AV"),
        ("cls_av_resid_to_probe_frac_mean_all", "CLS→intact B13 probe |A·V| fraction", "cls_to_probe_AV"),
        ("cls_av_resid_to_cls_frac_mean_all", "CLS→CLS |A·V| fraction", "cls_to_CLS_AV"),
    ]
    for col, label, stem in metrics:
        piv = q.pivot(index="head", columns="block", values=col).reindex(index=range(N_HEADS), columns=range(N_BLOCKS))
        fig, ax = plt.subplots(figsize=(13,6))
        im = _heatmap(ax, piv.to_numpy(), f"{model} / {mode}: {label}", vmin=0, vmax=max(.01, np.nanquantile(piv.to_numpy(), .995)))
        fig.colorbar(im, ax=ax, label=label)
        _savefig(fig, pdir / f"{stem}_head_heatmap_{model}_{mode}.png")

    # Block summary: separates actual Q=CLS behavior from generic incoming-sink topology.
    lq = ls[(ls.model_name.eq(model)) & (ls["mode"].eq(mode))]
    fig, ax = plt.subplots(figsize=(13,6))
    ax.plot(lq.block, lq.cls_av_to_cls_mean, marker="o", label="|A·V| from CLS source")
    ax.plot(lq.block, lq.cls_av_to_b13_regs_mean, marker="o", label="|A·V| from B13 visible-register set")
    ax.plot(lq.block, lq.cls_av_to_probe_mean, marker="o", label="|A·V| from intact B13 probe address")
    ax.plot(lq.block, lq.cls_av_to_patches_mean, marker="o", alpha=.6, label="|A·V| from ALL patch sources")
    ax.set(xlabel="Block", ylabel="Mean fraction of Q=CLS |A·V| source mass",
           title=f"{model} / {mode}: what CLS actually reads")
    ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(0,1); ax.legend(fontsize=8)
    _savefig(fig, pdir / f"CLS_query_AV_source_fractions_{model}_{mode}.png")


def plot_spatial_maps(pdir: Path, out: Path, model: str, mode: str, args):
    import matplotlib.pyplot as plt
    npz = out / f"cls_query_spatial_{model}_{mode}.npz"
    if not npz.is_file(): return
    d = np.load(npz)
    attn, av = d["attn_mean"], d["av_mean"]
    P = attn.shape[-1]
    side = int(round(math.sqrt(P)))
    if side * side != P:
        print(f"[plot] cannot make spatial maps: {P} patches is not square")
        return
    blocks = range(N_BLOCKS) if args.spatial_map_blocks.strip().lower() == "all" else [int(x) for x in args.spatial_map_blocks.split(",")]
    for b in blocks:
        for kind, arr, label in [("ATTN", attn, "Mean conditional CLS→patch attention"),
                                 ("AV", av, "Mean conditional CLS→patch |A·V| mass (after W_O norm)")]:
            maps = arr[b].reshape(N_HEADS, side, side)
            vmax = max(1e-8, float(np.quantile(maps, .995)))
            fig, axes = plt.subplots(4,4,figsize=(10,10))
            im = None
            for h, ax in enumerate(axes.flat):
                im = ax.imshow(maps[h], cmap=SEQUENTIAL_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_title(f"H{h}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{model} / {mode} / B{b}: {label}\npatch mass renormalized to 1 within each image/head", fontsize=12)
            if im is not None:
                fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.02, pad=.02)
            fig.savefig(pdir / f"CLS_spatial_{kind}_{model}_{mode}_B{b:02d}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)


def plot_ln_investigation(pdir: Path, model: str, token_s: pd.DataFrame, head_s: pd.DataFrame):
    import matplotlib.pyplot as plt
    if MODE_INTACT not in token_s["mode"].unique() or MODE_NOPUMP not in token_s["mode"].unique():
        return
    ti = token_s[token_s["mode"].eq(MODE_INTACT)].sort_values("block")
    tn = token_s[token_s["mode"].eq(MODE_NOPUMP)].sort_values("block")

    fig, axes = plt.subplots(3,1,figsize=(13,11),sharex=True)
    axes[0].plot(ti.block, ti.probe_raw_norm_median, marker="o", label="intact raw norm")
    axes[0].plot(tn.block, tn.probe_raw_norm_median, marker="o", label="no-pump raw norm")
    axes[0].set_ylabel("Probe residual norm"); axes[0].legend()
    axes[1].plot(ti.block, ti.probe_ln1_norm_median, marker="o", label="intact LN1 norm")
    axes[1].plot(tn.block, tn.probe_ln1_norm_median, marker="o", label="no-pump LN1 norm")
    axes[1].set_ylabel("Probe LN1 output norm"); axes[1].legend()
    axes[2].plot(ti.block, ti.probe_headmean_inflow_median, marker="o", label="intact incoming attention")
    axes[2].plot(tn.block, tn.probe_headmean_inflow_median, marker="o", label="no-pump incoming attention")
    axes[2].set_ylabel("Head-mean inflow to same probe"); axes[2].set_xlabel("Block"); axes[2].legend()
    axes[2].set_xticks(range(N_BLOCKS))
    fig.suptitle(f"{model}: same intact-B13 probe address — massiveness, LN and sink attraction")
    _savefig(fig, pdir / f"LN_probe_norm_and_inflow_{model}.png")

    # No-pump directional preservation relative to intact.
    hn = head_s[head_s["mode"].eq(MODE_NOPUMP)]
    block_h = hn.groupby("block")[["probe_k_cos_to_intact_median", "probe_v_cos_to_intact_median", "probe_q_cos_to_intact_median"]].median().reset_index()
    fig, ax = plt.subplots(figsize=(13,6))
    ax.plot(tn.block, tn.probe_raw_cos_to_intact_median, marker="o", label="raw residual cosine")
    ax.plot(tn.block, tn.probe_ln_cos_to_intact_median, marker="o", label="LN1 cosine")
    ax.plot(block_h.block, block_h.probe_k_cos_to_intact_median, marker="o", label="K cosine (median head)")
    ax.plot(block_h.block, block_h.probe_v_cos_to_intact_median, marker="o", label="V cosine (median head)")
    ax.plot(block_h.block, block_h.probe_q_cos_to_intact_median, marker="o", alpha=.7, label="Q cosine (median head)")
    ax.set(xlabel="Block", ylabel="Cosine(no-pump, intact) at SAME patch", title=f"{model}: what LayerNorm/QKV preserve after pump removal")
    ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(-.05,1.02); ax.legend(fontsize=8)
    _savefig(fig, pdir / f"LN_probe_direction_cosines_{model}.png")

    # QK advantage curves intact vs no-pump.
    fig, axes = plt.subplots(2,1,figsize=(13,9),sharex=True)
    for mode, ls in [(MODE_INTACT,"-"),(MODE_NOPUMP,"--")]:
        q = head_s[head_s["mode"].eq(mode)].groupby("block")[[
            "probe_mean_query_qk_advantage_median", "probe_cls_query_qk_advantage_median",
            "probe_mean_query_qk_advantage_vs_patches_median", "probe_cls_query_qk_advantage_vs_patches_median"
        ]].median().reset_index()
        axes[0].plot(q.block, q.probe_mean_query_qk_advantage_vs_patches_median, marker="o", ls=ls, label=f"{mode}: vs patches")
        axes[0].plot(q.block, q.probe_mean_query_qk_advantage_median, marker=".", ls=ls, alpha=.35, label=f"{mode}: vs all keys")
        axes[1].plot(q.block, q.probe_cls_query_qk_advantage_vs_patches_median, marker="o", ls=ls, label=f"{mode}: vs patches")
        axes[1].plot(q.block, q.probe_cls_query_qk_advantage_median, marker=".", ls=ls, alpha=.35, label=f"{mode}: vs all keys")
    axes[0].axhline(0,color="0.5",lw=.8); axes[1].axhline(0,color="0.5",lw=.8)
    axes[0].set_ylabel("Mean-query QK logit advantage")
    axes[1].set_ylabel("CLS-query QK logit advantage"); axes[1].set_xlabel("Block"); axes[1].set_xticks(range(N_BLOCKS))
    axes[0].legend(); axes[1].legend(); fig.suptitle(f"{model}: does pump removal change the normalized KEY advantage?")
    _savefig(fig, pdir / f"LN_probe_QK_advantage_{model}.png")

    # Per-head K cosine heatmap and QK advantage heatmaps.
    for metric, title, stem, vmin, vmax, cmap in [
        ("probe_k_cos_to_intact_median", "K direction cosine: no-pump vs intact", "LN_K_cosine", -1, 1, SIGNED_CMAP),
        ("probe_v_cos_to_intact_median", "V direction cosine: no-pump vs intact", "LN_V_cosine", -1, 1, SIGNED_CMAP),
    ]:
        piv = hn.pivot(index="head", columns="block", values=metric).reindex(index=range(N_HEADS), columns=range(N_BLOCKS))
        fig, ax = plt.subplots(figsize=(13,6)); im = _heatmap(ax,piv.to_numpy(),f"{model}: {title}",vmin=vmin,vmax=vmax,cmap=cmap)
        fig.colorbar(im,ax=ax,label="cosine"); _savefig(fig,pdir/f"{stem}_head_heatmap_{model}.png")

    for mode in (MODE_INTACT, MODE_NOPUMP):
        q = head_s[head_s["mode"].eq(mode)]
        piv = q.pivot(index="head", columns="block", values="probe_cls_query_qk_advantage_vs_patches_median").reindex(index=range(N_HEADS), columns=range(N_BLOCKS))
        lim = max(.1, float(np.nanquantile(np.abs(piv.to_numpy()), .98)))
        fig, ax = plt.subplots(figsize=(13,6)); im = _heatmap(ax,piv.to_numpy(),f"{model} / {mode}: CLS-query QK advantage to intact B13 probe vs mean PATCH key",vmin=-lim,vmax=lim,cmap=SIGNED_CMAP)
        fig.colorbar(im,ax=ax,label="logit advantage vs mean other key"); _savefig(fig,pdir/f"LN_CLS_QK_advantage_head_heatmap_{model}_{mode}.png")


def plot_norm_geyser(pdir: Path, model: str, mode: str, norm_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    q = norm_df[norm_df["mode"].eq(mode)]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(14,6))
    # Explicit colors/legend. Orange replaces the hard-to-see yellow-ish series.
    series = [
        ("spatial_norm_max_post", "max patch norm", "tab:blue", .16),
        ("spatial_norm_p95_post", "95th percentile patch norm", "tab:orange", .16),
        ("spatial_norm_median_post", "median patch norm", "0.35", .10),
    ]
    for col, label, color, alpha in series:
        jitter = rng.normal(0,.045,len(q))
        ax.scatter(q.block+jitter, q[col], s=9, alpha=alpha, color=color, label=label)
    ax.axvspan(10.6,12.4,color="0.92",zorder=0,label="B11/B12 pump window")
    ax.axvline(20,color="tab:red",lw=1,ls="--",label="B20 MLP creates second norm population")
    ax.set(xlabel="Block (post-MLP state)", ylabel="Spatial token residual norm", title=f"{model} / {mode}: norm-population GEYSER")
    ax.set_xticks(range(N_BLOCKS)); ax.set_xlim(-.5,23.5); ax.legend(fontsize=8, ncol=2)
    _savefig(fig,pdir/f"norm_geyser_{model}_{mode}.png")

    # Counts above thresholds make the B20 population size explicit.
    sm = norm_summary(q)
    fig, ax = plt.subplots(figsize=(13,6))
    for c,label in [("count_gt20_post_mean",">20"),("count_gt30_post_mean",">30"),("count_gt40_post_mean",">40"),("count_gt60_post_mean",">60"),("count_gt100_post_mean",">100")]:
        ax.plot(sm.block, sm[c], marker="o", label=label)
    ax.set(xlabel="Block", ylabel="Mean number of spatial tokens", title=f"{model} / {mode}: size of high-norm populations")
    ax.set_xticks(range(N_BLOCKS)); ax.legend(title="Residual norm cutoff")
    _savefig(fig,pdir/f"norm_population_counts_{model}_{mode}.png")


def plot_model(out: Path, model: str, h: pd.DataFrame, l: pd.DataFrame, n: pd.DataFrame,
               lt: pd.DataFrame, lh: pd.DataFrame, hs: pd.DataFrame, ls: pd.DataFrame,
               lts: pd.DataFrame, lhs: pd.DataFrame, args):
    pdir = out / "plots" / model
    pdir.mkdir(parents=True, exist_ok=True)
    modes = sorted(h["mode"].unique())
    for mode in modes:
        hm = h[h["mode"].eq(mode)]; lm = l[l["mode"].eq(mode)]; hsm = hs[hs["mode"].eq(mode)]
        plot_transition_clouds(pdir, model, mode, hm, lm, args)
        plot_cls_patch_pies(pdir, model, mode, hm)
        plot_head_maps(pdir, model, mode, hsm)
        plot_cls_sink_atlas(pdir, model, mode, hsm, args)
        plot_cls_broadcast_violin(pdir, model, mode, hm)
        plot_nop_broadcast_scatter(pdir, model, mode, hsm, args)
        plot_nop_broadcast_mechanism_clear(pdir, model, mode, hm, args)
        plot_head_block_regime_atlas(pdir, model, mode, hsm, args)
        plot_massive_ln_scatter(pdir, model, mode, hm, args)
        plot_role_overlap(pdir, model, mode, hm)
        plot_cls_query(pdir, model, mode, hs, ls)
        plot_spatial_maps(pdir, out, model, mode, args)
        plot_norm_geyser(pdir, model, mode, n)
    plot_regime_rates(pdir, model, ls)
    plot_absolute_regime_zoom(pdir, model, ls)
    plot_ln_investigation(pdir, model, lts, lhs)


def plot_cross_model(out: Path, L: pd.DataFrame, LT: pd.DataFrame, N: pd.DataFrame):
    import matplotlib.pyplot as plt
    pdir = out / "plots" / "COMPARE_MODELS"; pdir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("meanhead_patch_fraction", "Head-mean strongest target is PATCH", "meanhead_patch_fraction", (0,1)),
        ("head_patch_sink_rate_all", "Absolute hard PATCH-sink event rate", "patch_hard_sink_rate", (0,1)),
        ("head_cls_sink_rate_all", "Absolute hard CLS-sink event rate", "cls_hard_sink_rate", (0,1)),
        ("head_broadcast_rate_all", "Absolute BROADCAST event rate", "broadcast_rate", (0,1)),
        ("head_nop_rate_all", "Absolute NOP event rate", "nop_rate", (0,1)),
        ("cls_av_to_b13_regs_mean", "Q=CLS |A·V| fraction from B13 registers", "CLS_AV_B13reg", (0,1)),
        ("cls_av_to_probe_mean", "Q=CLS |A·V| fraction from B13 probe", "CLS_AV_probe", (0,1)),
    ]
    for mode in L["mode"].unique():
        for col, ylabel, stem, ylim in metrics:
            fig, ax = plt.subplots(figsize=(13,6))
            for model in MODEL_ORDER:
                q = L[(L.model_name.eq(model)) & L["mode"].eq(mode)]
                if len(q): ax.plot(q.block, q[col], marker="o", ms=3, label=model)
            ax.set(xlabel="Block", ylabel=ylabel, title=f"ALL MODELS / {mode}: {ylabel}")
            ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(*ylim); ax.legend(fontsize=8,ncol=2)
            _savefig(fig,pdir/f"COMPARE_{stem}_{mode}.png")

        # Norm p95 / count>30 comparison.
        fig, axes = plt.subplots(2,1,figsize=(13,10),sharex=True)
        for model in MODEL_ORDER:
            q = N[(N.model_name.eq(model)) & N["mode"].eq(mode)]
            if len(q):
                axes[0].plot(q.block,q.spatial_norm_p95_post_median,marker="o",ms=3,label=model)
                axes[1].plot(q.block,q.count_gt30_post_mean,marker="o",ms=3,label=model)
        axes[0].set_ylabel("Median image p95 patch norm"); axes[1].set_ylabel("Mean # patches norm>30")
        axes[1].set_xlabel("Block"); axes[1].set_xticks(range(N_BLOCKS)); axes[0].legend(fontsize=8,ncol=2)
        fig.suptitle(f"ALL MODELS / {mode}: norm-population geometry")
        _savefig(fig,pdir/f"COMPARE_norm_geometry_{mode}.png")

    # No-pump LN cosine comparison across models.
    q = LT[LT["mode"].eq(MODE_NOPUMP)]
    if len(q):
        fig, ax = plt.subplots(figsize=(13,6))
        for model in MODEL_ORDER:
            z = q[q.model_name.eq(model)]
            if len(z): ax.plot(z.block,z.probe_ln_cos_to_intact_median,marker="o",ms=3,label=model)
        ax.set(xlabel="Block",ylabel="LN1 cosine(no-pump,intact)",title="ALL MODELS: how much pump ablation rotates the normalized B13 probe")
        ax.set_xticks(range(N_BLOCKS)); ax.set_ylim(-.05,1.02); ax.legend(fontsize=8,ncol=2)
        _savefig(fig,pdir/"COMPARE_LN_probe_cosine_no_pump.png")


# =============================================================================
# Reporting / compact handoff bundle
# =============================================================================

def write_report(out: Path, H: pd.DataFrame, L: pd.DataFrame, LT: pd.DataFrame, LH: pd.DataFrame,
                 N: pd.DataFrame, args):
    lines = [
        "# CLIP sink / NOP / broadcast / CLS-query / LN diagnostics v2.3 CLS-plotfix", "",
        "Paper-adapted sink diagnostics plus CLIP-specific extensions.", "",
        f"- hard head sink: strongest incoming key mean attention >= **{args.sink_threshold:g}**",
        f"- NOP-like: sink V-norm ratio < **{PAPER_NOP_VALUE_RATIO}**",
        f"- BROADCAST-like: non-NOP plus residual-update stable rank <= **{PAPER_RANK1_STABLE_RANK}**",
        f"- persistent paper-style scatter: hard-sink frequency >= **{args.persistent_head_frequency:g}**",
        "- Q=CLS plots use both raw attention and |A·V| after each head's W_O slice.",
        "- `no_pump` zeros the complete B11/B12 post-QuickGELU pump lists.",
        "- LN probe tracks the SAME intact B13 address under intact/no-pump, including raw/LN/Q/K/V cosine and QK-logit advantage.", "",
    ]
    focus = [11,12,13,18,20,21,22,23]
    for model in [m for m in MODEL_ORDER if m in set(L.model_name)]:
        lines += [f"## {model}", ""]
        for mode in L[L.model_name.eq(model)]["mode"].unique():
            q = L[(L.model_name.eq(model)) & L["mode"].eq(mode) & L.block.isin(focus)]
            lines += [f"### {mode}", "",
                      "| B | hard sink/all | patch sink/all | CLS sink/all | NOP/all | broadcast/all | meanhead patch | CLS→reg AV |",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for r in q.itertuples(index=False):
                lines.append(f"| {r.block} | {r.head_sink_event_rate:.3f} | {r.head_patch_sink_rate_all:.3f} | {r.head_cls_sink_rate_all:.3f} | {r.head_nop_rate_all:.3f} | {r.head_broadcast_rate_all:.3f} | {r.meanhead_patch_fraction:.3f} | {r.cls_av_to_b13_regs_mean:.3f} |")
            lines.append("")
    lines += [
        "## Plot denominator warning", "",
        "`regime_rate_ABSOLUTE...` uses ALL head×image observations as denominator and is the preferred rate plot.",
        "`regime_composition_GIVEN_SINK...` preserves the old conditional view; v2.1 ALSO emits separate intact/no-pump versions so lines cannot hide under each other.", "",
        "## Guardrails", "",
        "- sink topology, residual massiveness, and algorithmic function are measured separately.",
        "- v2.3 adds dedicated CLS-only head×block atlases/counts/trajectories so rare CLS-sink heads are never averaged away; generic sequential heatmaps are restored to viridis.",
        "- B13 register labels are visible-register addresses from the intact model's pre-B13 residual norm threshold; if a model has no >threshold register, the LN probe falls back to its strongest B13 spatial sink and flags that fact.",
        "- B20 local new-norm labels are computed separately in intact and no-pump; B20 attention cannot target the post-B20 population until B21.",
        "- Q=CLS spatial maps condition on PATCH-source mass to reveal spatial motifs; scalar plots separately show how much total CLS read comes from patches versus CLS/registers.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_send_bundle(out: Path, args) -> Path:
    """Create the small ZIP the user can send back instead of hundreds of MB raw CSVs."""
    keep_names = [
        "REPORT.md", "config.json", "register_pump_units.json",
        "sink_head_summary_ALL.csv", "sink_layer_summary_ALL.csv",
        "ln_probe_token_summary_ALL.csv", "ln_probe_head_summary_ALL.csv",
        "norm_summary_ALL.csv", "focus_heads_B11_B12_B13_B18_B20_B21_B22_B23.csv",
        "cls_broadcast_heads_ALL.csv", "plot_data_by_block_ALL.csv",
    ]
    zpath = out / "compact_summary_workspace_broadcast_sinks.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for name in keep_names:
            p = out / name
            if p.is_file(): z.write(p, arcname=name)
        # Compact numeric CLS-query spatial means are much smaller than the 24x contact-sheet PNG forest.
        for p in out.glob("cls_query_spatial_*.npz"):
            z.write(p, arcname=p.name)
        # Model-load audits make a weird checkpoint failure diagnosable from the small handoff.
        for p in out.glob("*/load_audit/*"):
            if p.is_file() and p.suffix.lower() in {".json", ".csv"}:
                z.write(p, arcname=str(p.relative_to(out)))
        # Include only selected overview plots; full per-block CLS spatial maps stay outside.
        pbase = out / "plots"
        if pbase.is_dir():
            for p in pbase.rglob("*.png"):
                n = p.name
                if (n.startswith("sink_transition_ALL_") or n.startswith("norm_geyser_") or
                    n.startswith("CLS_query_AV_source_fractions_") or n.startswith("LN_probe_") or
                    n.startswith("cls_broadcast_head_bubble_") or n.startswith("cls_broadcast_strength_violin_") or
                    n.startswith("regime_rate_ABSOLUTE_") or n.startswith("regime_composition_GIVEN_SINK_by_block_") or
                    n.startswith("regime_composition_ABSOLUTE_STACKED_by_block_") or
                    n.startswith("nop_vs_broadcast_MECHANISM_OVERVIEW_") or
                    n.startswith("nop_vs_broadcast_BLOCK_REGIME_CENTROIDS_") or
                    n.startswith("regime_HEADxBLOCK_") or n.startswith("regime_HEAD_COUNTS_") or
                    n.startswith("regime_PER_HEAD_TRAJECTORIES_") or n.startswith("regime_HEAD_ID_GLYPHS_") or
                    n.startswith("regime_composition_ABSOLUTE_SINK_ONLY_ZOOM_ANNOTATED_") or
                    n.startswith("CLS_sink_") or
                    "COMPARE_" in n or n.startswith("nop_vs_broadcast_PERSISTENT_")):
                    z.write(p, arcname=str(p.relative_to(out)))
    return zpath


def postprocess(args, out: Path):
    hs_all, ls_all, lts_all, lhs_all, ns_all = [], [], [], [], []
    for model in args.models:
        hp = out / f"sink_head_events_{model}.csv"
        lp = out / f"sink_layer_events_{model}.csv"
        npth = out / f"sink_norm_rows_{model}.csv"
        ltp = out / f"ln_probe_token_rows_{model}.csv"
        lhp = out / f"ln_probe_head_rows_{model}.csv"
        if not all(p.is_file() for p in (hp,lp,npth,ltp,lhp)):
            print(f"[post] skip incomplete model {model}")
            continue
        print(f"[post] {model}")
        h = pd.read_csv(hp); l = pd.read_csv(lp); n = pd.read_csv(npth); lt = pd.read_csv(ltp); lh = pd.read_csv(lhp)
        hs = head_summary(h); ls = layer_summary(h,l); lts,lhs = ln_summary(lt,lh); ns = norm_summary(n)
        hs.to_csv(out / f"sink_head_summary_{model}.csv", index=False)
        ls.to_csv(out / f"sink_layer_summary_{model}.csv", index=False)
        lts.to_csv(out / f"ln_probe_token_summary_{model}.csv", index=False)
        lhs.to_csv(out / f"ln_probe_head_summary_{model}.csv", index=False)
        ns.to_csv(out / f"norm_summary_{model}.csv", index=False)
        plot_model(out,model,h,l,n,lt,lh,hs,ls,lts,lhs,args)
        hs_all.append(hs); ls_all.append(ls); lts_all.append(lts); lhs_all.append(lhs); ns_all.append(ns)
        del h,l,n,lt,lh
        gc.collect()

    if not hs_all:
        raise RuntimeError("No complete model outputs found for postprocessing")
    H = pd.concat(hs_all,ignore_index=True); L = pd.concat(ls_all,ignore_index=True)
    LT = pd.concat(lts_all,ignore_index=True); LH = pd.concat(lhs_all,ignore_index=True); N = pd.concat(ns_all,ignore_index=True)
    H.to_csv(out / "sink_head_summary_ALL.csv", index=False)
    L.to_csv(out / "sink_layer_summary_ALL.csv", index=False)
    LT.to_csv(out / "ln_probe_token_summary_ALL.csv", index=False)
    LH.to_csv(out / "ln_probe_head_summary_ALL.csv", index=False)
    N.to_csv(out / "norm_summary_ALL.csv", index=False)

    focus = H[H.block.isin([11,12,13,18,20,21,22,23])]
    focus.to_csv(out / "focus_heads_B11_B12_B13_B18_B20_B21_B22_B23.csv", index=False)
    H[H.cls_broadcast_rate_all.gt(0)].sort_values(["model_name","mode","block","cls_broadcast_rate_all"], ascending=[True,True,True,False]).to_csv(out / "cls_broadcast_heads_ALL.csv", index=False)
    # One tidy layer-level table is sufficient to recreate most overview plots.
    L.to_csv(out / "plot_data_by_block_ALL.csv", index=False)

    plot_cross_model(out,L,LT,N)
    write_report(out,H,L,LT,LH,N,args)
    z = build_send_bundle(out,args)
    print(f"[summary bundle] {z}")


# =============================================================================
# CLI / tests
# =============================================================================

def self_test():
    torch.manual_seed(0)
    x = torch.randn(257,64)
    assert 1 <= stable_rank(x) <= 64.001
    one = torch.randn(257,1) @ torch.randn(1,64)
    assert abs(stable_rank(one)-1) < 1e-3
    W = torch.randn(1024,1024)
    assert 1 <= projected_stable_rank(x,W,3) <= 64.001
    z = torch.ones(257,1) @ torch.randn(1,64)
    assert abs(projected_stable_rank(z,W,2)-1) < 1e-3
    vb = torch.randn(3,257,64)
    rb = projected_row_norms_batch(vb,W,4)
    r0 = projected_row_norms(vb[0],W,4)
    assert torch.allclose(rb[0],r0,atol=1e-4,rtol=1e-4)
    assert event_regime(.1,5)=="NOP"
    assert event_regime(.8,1.02)=="BROADCAST"
    assert event_regime(.8,2)=="OTHER"
    print("[self-test] PASS")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--postprocess_only", action="store_true")
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default=str(script_dir()/"nop_bc"/"fixed_sink_manifest.csv"))
    ap.add_argument("--models", default=",".join((MODEL_PRE, MODEL_GMP, MODEL_FT)))
    ap.add_argument("--include_no_pump", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--sink_threshold", type=float, default=PAPER_HEAD_SINK_THRESHOLD)
    ap.add_argument("--persistent_head_frequency", type=float, default=.5)
    ap.add_argument("--head_regime_min_frequency", type=float, default=.05, help="minimum per-head hard-sink frequency for the GE5PCT-style head-count plot")
    ap.add_argument("--massive_threshold", type=float, default=DEFAULT_MASSIVE_TOKEN_NORM)
    ap.add_argument("--b20_readout_threshold", type=float, default=30.0)
    ap.add_argument("--spatial_map_blocks", default="all", help="all or comma-separated block indices")
    ap.add_argument("--restart", action=argparse.BooleanOptionalAction, default=False, help="force rerunning even fully completed per-model extraction legs")
    ap.add_argument("--log_every", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip_module", default=S.DEFAULT_CLIP_MODULE)
    ap.add_argument("--model_spec", default=S.DEFAULT_MODEL_SPEC)
    ap.add_argument("--xattn_checkpoint", default=DEFAULT_XATTN_CHECKPOINT)
    ap.add_argument("--xattn_module", default="oaiclip")
    ap.add_argument("--pickle_module", default="clip", help="module to pre-import before torch.load of ordinary OpenAI CLIP pickles")
    ap.add_argument("--gmp_checkpoint", default=DEFAULT_CHECKPOINTS[MODEL_GMP])
    ap.add_argument("--regression_checkpoint", default=DEFAULT_CHECKPOINTS[MODEL_REG])
    ap.add_argument("--brut_checkpoint", default=DEFAULT_CHECKPOINTS[MODEL_BRUT])
    ap.add_argument("--sae_hinge_checkpoint", default=DEFAULT_CHECKPOINTS[MODEL_SAE])
    args = ap.parse_args()
    args.models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    unknown = set(args.models) - set(MODEL_ORDER)
    if unknown: ap.error(f"unknown --models entries: {sorted(unknown)}")
    return args


def run(args):
    if args.self_test:
        self_test(); return 0
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    out = Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); (out/"plots").mkdir(exist_ok=True)
    (out/"config.json").write_text(json.dumps(vars(args),indent=2,default=str),encoding="utf-8")
    (out/"register_pump_units.json").write_text(json.dumps({str(k):list(v) for k,v in REGISTER_PUMP_UNITS.items()},indent=2),encoding="utf-8")
    if not args.postprocess_only:
        for model in args.models:
            run_extraction(args,out,model)
    postprocess(args,out)
    print(f"[done] {out}")
    print(f"[compact summary] {out/'compact_summary_workspace_broadcast_sinks.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
