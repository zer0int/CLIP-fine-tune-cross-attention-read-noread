#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B20 push-pull mediation probe: selected hidden-neuron write -> residual coordinates
650 / 715 -> B21 multi-head routing -> B22 565Q x 650K gate -> final embedding.

This is a follow-up to probe_B20_SHARPENERS_FLATTENERS.py. It reads the saved
signed B20 subfamilies and intervenes on the exact per-token c_proj write of the
ranked push-pull group (default: top 16 of sharpH-up / flatH-down).

The key distinction is between:
  * necessity: remove only coordinate 650, 715, or both from the group's write;
  * sufficiency/add-back: ablate the group's ordinary-patch write, then restore
    only coordinate 650, only 715, or both;
  * localization: compare all-token group ablation against ordinary-patch-only
    group ablation;
  * reconstruction: ordinary ablation + exact full group write must reproduce
    baseline up to numerical precision.

Per intervention the probe records:
  * final normalized image embedding;
  * all B21 head top1 / entropy / margin on ordinary patch queries;
  * B22 CLS attention masses to frozen registers / ordinary patches / RN;
  * exact FP32 B22 coordinate QK terms for 650/565, including 565Q*650K;
  * the selected B20 group's exact per-token write energy and coordinate share.

No historical conclusion is hard-coded: 650 and 715 are only the tested residual
coordinates, and the push-pull neuron IDs are read from prior output files.
No pickle outputs are written.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from safetensors.torch import save_file as _save_st, load_file as _load_st
except Exception as e:
    raise RuntimeError("safetensors is required") from e

EPS = 1e-12
FORMAT_VERSION = 1
MEDIATION_COLUMNS = [
    "model",
    "metric",
    "addback",
    "baseline",
    "ordinary_ablate",
    "addback_value",
    "restoration_fraction",
]
DEFAULT_MODES = [
    "baseline",
    "group_ablate_all",
    "group_ablate_ordinary",
    "ordinary_ablate_plus_650",
    "ordinary_ablate_plus_715",
    "ordinary_ablate_plus_650_715",
    "ordinary_remove_650",
    "ordinary_remove_715",
    "ordinary_remove_650_715",
    "ordinary_reconstruct",
]


def amp_context(device: str, enabled: bool):
    if enabled and str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def save_st(tensors: Dict[str, torch.Tensor], filename: Path, metadata: Optional[Dict[str, Any]] = None):
    packed = {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}
    filename.parent.mkdir(parents=True, exist_ok=True)
    _save_st(packed, str(filename), metadata={str(k): str(v) for k, v in (metadata or {}).items()})


def parse_ints(s):
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_strs(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [Path.cwd(), here.parent, *here.parents]:
        if (p / "attnclip_mechinterp_sae").is_dir() and (p / "attnclip_mechinterp_xattn").is_dir() and (p / "utils_clip_loader").is_dir():
            return p
    raise FileNotFoundError("Could not locate repo root; pass --repo_root")


def _import_runtime(args):
    repo = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import attnclip_mechinterp_sae as clip_sae
    import attnclip_mechinterp_xattn as clip_x
    from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything, resolve_to_openai_state_dict
    return repo, clip_sae, clip_x, load_openai_clip_anything, resolve_to_openai_state_dict


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _resolve_xattn_state(clip_sae, resolve_fn, args):
    sd, info = resolve_fn(
        args.xattn_model,
        cache_dir=(args.hf_cache_dir or None),
        revision=(args.xattn_revision or None),
        allow_unsafe_hf_pickle=False,
    )
    conv = getattr(getattr(clip_sae, "model", None), "convert_state_dict_inproj_to_qkv", None)
    if callable(conv):
        sd = conv(sd)
    return sd, info


def _load_gmp(clip_sae, load_any, args, device):
    try:
        m, p, li = load_any(
            clip_sae, args.gmp_checkpoint, device=device, jit=False, strict=True,
            reuse_full_model_pickle=False,
        )
        return _freeze(m), p, {"source": args.gmp_checkpoint, "mode": "state_dict_rebuild", "loader": str(li)}
    except Exception as first_error:
        src, _pp, li = load_any(
            clip_sae, args.gmp_checkpoint, device="cpu", jit=False, strict=True,
            reuse_full_model_pickle=True,
        )
        m, p, _ = load_any(clip_sae, args.pretrained_model, device=device, jit=False, strict=True, allow_unsafe_hf_pickle=False)
        srcsd = src.state_dict()
        conv = getattr(getattr(clip_sae, "model", None), "convert_state_dict_inproj_to_qkv", None)
        if callable(conv):
            srcsd = conv(srcsd)
        tgt = m.state_dict()
        filt = {
            k: v.to(dtype=tgt[k].dtype)
            for k, v in srcsd.items()
            if k.startswith("visual.") and k in tgt and tuple(v.shape) == tuple(tgt[k].shape)
        }
        miss = sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
        if miss:
            raise RuntimeError(f"GmP fallback missing visual keys: {miss[:12]}") from first_error
        m.load_state_dict(filt, strict=False)
        del src
        gc.collect()
        return _freeze(m), p, {
            "source": args.gmp_checkpoint,
            "mode": "trusted_pickle_visual_transplant",
            "loader": str(li),
            "first_error": repr(first_error),
        }


def _load_bare_xattn(clip_sae, load_any, resolve_fn, args, device):
    m, p, _ = load_any(clip_sae, args.pretrained_model, device=device, jit=False, strict=True, allow_unsafe_hf_pickle=False)
    sd, info = _resolve_xattn_state(clip_sae, resolve_fn, args)
    tgt = m.state_dict()
    filt = {}
    ignored = []
    for k, v in sd.items():
        if (
            not k.startswith("visual.")
            or k in {"visual.read_null_token", "visual.read_null_insert_block_config"}
            or k not in tgt
            or tuple(v.shape) != tuple(tgt[k].shape)
        ):
            ignored.append(k)
            continue
        filt[k] = v.to(dtype=tgt[k].dtype)
    miss = sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
    if miss:
        raise RuntimeError(f"bare_xattn missing visual keys: {miss[:12]}")
    inc = m.load_state_dict(filt, strict=False)
    return _freeze(m), p, {
        "source": args.xattn_model,
        "mode": "bare_xattn",
        "loaded_visual_keys": len(filt),
        "ignored_key_count": len(ignored),
        "load_missing": list(inc.missing_keys),
        "load_unexpected": list(inc.unexpected_keys),
        "loader": str(info),
    }


@dataclass
class Variant:
    name: str
    model: Any
    preprocess: Any
    source_info: Dict[str, Any]
    is_full_xattn: bool = False

    @property
    def visual(self):
        return self.model.visual

    def maybe_insert_rn(self, block_idx: int, x: torch.Tensor) -> torch.Tensor:
        if self.is_full_xattn:
            return self.visual._maybe_insert_read_null(block_idx, x)
        return x


def load_variant(name, args) -> Variant:
    _repo, clip_sae, clip_x, load_any, resolve_fn = _import_runtime(args)
    if name == "pretrained":
        m, p, li = load_any(clip_sae, args.pretrained_model, device=args.device, jit=False, strict=True, allow_unsafe_hf_pickle=False)
        return Variant(name, _freeze(m), p, {"source": args.pretrained_model, "loader": str(li)}, False)
    if name == "gmp":
        m, p, i = _load_gmp(clip_sae, load_any, args, args.device)
        return Variant(name, m, p, i, False)
    if name == "bare_xattn":
        m, p, i = _load_bare_xattn(clip_sae, load_any, resolve_fn, args, args.device)
        return Variant(name, m, p, i, False)
    if name == "full_xattn":
        m, p, li = load_any(
            clip_x, args.xattn_model, device=args.device, jit=False,
            cache_dir=(args.hf_cache_dir or None), revision=(args.xattn_revision or None),
            strict=True, allow_unsafe_hf_pickle=False,
        )
        return Variant(name, _freeze(m), p, {"source": args.xattn_model, "mode": "full_xattn", "loader": str(li)}, True)
    raise ValueError(name)


def load_manifest(sharp_flat_root: Path, image_dir: str, max_images: int):
    p = sharp_flat_root / "image_manifest.csv"
    if p.exists():
        d = pd.read_csv(p)
        if max_images > 0:
            d = d.iloc[:max_images].copy()
        return d
    root = Path(image_dir)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    fs = sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in exts)
    if max_images > 0:
        fs = fs[:max_images]
    if not fs:
        raise FileNotFoundError(f"No images under {root} and no manifest at {p}")
    return pd.DataFrame({"stim_id": [x.stem for x in fs], "path": [str(x) for x in fs]})


def load_batch(preprocess, rows, device):
    from PIL import Image
    xs = []
    for p in rows.path:
        with Image.open(p) as im:
            xs.append(preprocess(im.convert("RGB")))
    return torch.stack(xs, 0).to(device, non_blocking=True)


def resolve_c_proj(blk):
    cp = getattr(blk.mlp, "c_proj", None)
    if cp is not None and hasattr(cp, "weight") and getattr(cp, "weight").ndim == 2:
        return cp
    weighted = [
        m for m in blk.mlp.modules()
        if m is not blk.mlp and hasattr(m, "weight") and isinstance(getattr(m, "weight"), torch.Tensor)
        and getattr(m, "weight").ndim == 2
    ]
    if len(weighted) < 2:
        raise RuntimeError("Could not identify MLP c_proj")
    return weighted[-1]


def frozen_register_mask(pre13_tbc, P, threshold, max_registers, min_registers):
    n = pre13_tbc[1:1 + P].float().norm(dim=-1).T
    mask = torch.zeros_like(n, dtype=torch.bool)
    for i in range(n.shape[0]):
        idx = torch.nonzero(n[i] >= threshold, as_tuple=False).flatten()
        if idx.numel() < min_registers:
            idx = torch.topk(n[i], k=min(min_registers, P)).indices
        if max_registers > 0 and idx.numel() > max_registers:
            vals = n[i, idx]
            idx = idx[torch.topk(vals, k=max_registers).indices]
        mask[i, idx] = True
    return mask, n


def load_group_spec(root: Path, model: str, group_name: str, top_k: int):
    p = root / model / "subfamily_indices.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing sharpen/flatten output: {p}")
    d = json.loads(p.read_text())
    groups = d.get("groups", {})
    if group_name not in groups:
        raise KeyError(f"{group_name!r} not present in {p}; available={sorted(groups)}")
    ids = list(map(int, groups[group_name]))
    if top_k > 0:
        ids = ids[:min(top_k, len(ids))]
    if not ids:
        raise RuntimeError(f"Selected group is empty: {model} / {group_name}")
    return ids, int(d["sharp_head"]), int(d["flat_head"]), d


def ordinary_token_mask(regmask: torch.Tensor, T: int, P: int, device) -> torch.Tensor:
    # [T,B,1], excludes CLS, frozen registers, and any RN token.
    B = regmask.shape[0]
    m = torch.zeros((T, B, 1), dtype=torch.bool, device=device)
    m[1:1 + P, :, 0] = (~regmask).T.to(device)
    return m


def group_write_from_hidden(cp, h: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    idx = torch.as_tensor(list(ids), dtype=torch.long, device=h.device)
    hs = h.index_select(-1, idx)
    W = cp.weight.index_select(1, idx).to(device=h.device, dtype=h.dtype)
    # c_proj convention is out = h @ W.T + bias.
    return F.linear(hs, W, bias=None)


def axis_projection(write: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    y = torch.zeros_like(write)
    for a in axes:
        y[..., int(a)] = write[..., int(a)]
    return y


def transform_cproj_output(output: torch.Tensor, h: torch.Tensor, cp, ids: Sequence[int], mode: str,
                           regmask: torch.Tensor, P: int, axis_a: int, axis_b: int):
    if mode == "baseline":
        return output, group_write_from_hidden(cp, h, ids)
    gw = group_write_from_hidden(cp, h, ids)
    if mode == "group_ablate_all":
        return output - gw, gw
    om = ordinary_token_mask(regmask, output.shape[0], P, output.device).to(dtype=output.dtype)
    pa = axis_projection(gw, [axis_a])
    pb = axis_projection(gw, [axis_b])
    pab = pa + pb
    if mode == "group_ablate_ordinary":
        delta = -gw
    elif mode == f"ordinary_ablate_plus_{axis_a}":
        delta = -gw + pa
    elif mode == f"ordinary_ablate_plus_{axis_b}":
        delta = -gw + pb
    elif mode == f"ordinary_ablate_plus_{axis_a}_{axis_b}":
        delta = -gw + pab
    elif mode == f"ordinary_remove_{axis_a}":
        delta = -pa
    elif mode == f"ordinary_remove_{axis_b}":
        delta = -pb
    elif mode == f"ordinary_remove_{axis_a}_{axis_b}":
        delta = -pab
    elif mode == "ordinary_reconstruct":
        # Intentionally execute the subtract+add arithmetic as a numerical sanity check.
        delta = -gw + gw
    else:
        raise ValueError(f"Unknown intervention mode: {mode}")
    return output + om * delta, gw


def b21_rows(probs: torch.Tensor, regmask: torch.Tensor, P: int, stim_ids: Sequence[str], model: str, mode: str):
    B, H, T, S = probs.shape
    p = probs.float()
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1) / math.log(max(S, 2))
    top = torch.topk(p, k=min(2, S), dim=-1).values
    top1 = top[..., 0]
    margin = top[..., 0] - (top[..., 1] if S > 1 else 0.0)
    rows = []
    for i, sid in enumerate(stim_ids):
        om = ~regmask[i]
        for h in range(H):
            vals = {
                "top1": top1[i, h, 1:1 + P][om],
                "entropy": ent[i, h, 1:1 + P][om],
                "margin": margin[i, h, 1:1 + P][om],
            }
            rows.append({
                "model": model, "intervention": mode, "stim_id": sid, "head": h,
                **{k: float(v.mean().cpu()) if v.numel() else float("nan") for k, v in vals.items()},
            })
    return rows


def b22_attention_rows(probs: torch.Tensor, regmask: torch.Tensor, P: int, stim_ids: Sequence[str], model: str, mode: str):
    B, H, T, S = probs.shape
    p = probs.float()
    q = p[:, :, 0, :]  # CLS query.
    ent = -(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(-1) / math.log(max(S, 2))
    rows = []
    for i, sid in enumerate(stim_ids):
        rm = regmask[i]
        om = ~rm
        for h in range(H):
            patch = q[i, h, 1:1 + P]
            rows.append({
                "model": model,
                "intervention": mode,
                "stim_id": sid,
                "head": h,
                "cls_top1": float(q[i, h].max().cpu()),
                "cls_entropy": float(ent[i, h].cpu()),
                "cls_self_mass": float(q[i, h, 0].cpu()),
                "cls_to_register_mass": float(patch[rm].sum().cpu()) if int(rm.sum()) else 0.0,
                "cls_to_ordinary_mass": float(patch[om].sum().cpu()) if int(om.sum()) else 0.0,
                "cls_to_extra_mass": float(q[i, h, 1 + P:].sum().cpu()) if S > P + 1 else 0.0,
            })
    return rows


def qk_pair_terms(blk, ln1_tbc: torch.Tensor, pair: Tuple[int, int], regmask: torch.Tensor,
                  P: int, block: int, stim_ids: Sequence[str], model: str, mode: str):
    a, b = pair
    attn = blk.attn
    Wq = attn.q_proj.weight.detach().float()
    Wk = attn.k_proj.weight.detach().float()
    H = attn.num_heads
    D = attn.head_dim
    scale = 1.0 / math.sqrt(D)
    x = ln1_tbc.permute(1, 0, 2).float()  # [B,T,E]
    bq = attn.q_proj.bias.detach().float() if attn.q_proj.bias is not None else None
    bk = attn.k_proj.bias.detach().float() if attn.k_proj.bias is not None else None
    qfull = F.linear(x[:, 0], Wq, bq).view(x.shape[0], H, D)
    kfull = F.linear(x[:, 1:1 + P], Wk, bk).view(x.shape[0], P, H, D).permute(0, 2, 1, 3)
    qcomp = {c: (x[:, 0, c, None] * Wq[:, c][None, :]).view(x.shape[0], H, D) for c in (a, b)}
    kcomp = {
        c: (x[:, 1:1 + P, c, None] * Wk[:, c][None, None, :]).view(x.shape[0], P, H, D).permute(0, 2, 1, 3)
        for c in (a, b)
    }
    rows = []
    for i, sid in enumerate(stim_ids):
        groups = {
            "register": regmask[i],
            "ordinary": ~regmask[i],
            "all_patches": torch.ones(P, dtype=torch.bool, device=regmask.device),
        }
        for gname, m in groups.items():
            if int(m.sum()) == 0:
                continue
            kf = kfull[i, :, m, :].mean(1)
            exact = (qfull[i] * kf).sum(-1) * scale
            terms = {}
            for qc in (a, b):
                for kc in (a, b):
                    kk = kcomp[kc][i, :, m, :].mean(1)
                    terms[(qc, kc)] = (qcomp[qc][i] * kk).sum(-1) * scale
            for h in range(H):
                rows.append({
                    "model": model,
                    "intervention": mode,
                    "stim_id": sid,
                    "block": block,
                    "head": h,
                    "source_group": gname,
                    "q_axis_a": a,
                    "q_axis_b": b,
                    "exact_full_qk_logit": float(exact[h].cpu()),
                    f"q{a}_k{a}": float(terms[(a, a)][h].cpu()),
                    f"q{a}_k{b}": float(terms[(a, b)][h].cpu()),
                    f"q{b}_k{a}": float(terms[(b, a)][h].cpu()),
                    f"q{b}_k{b}": float(terms[(b, b)][h].cpu()),
                    "cross_pair_sum": float((terms[(a, b)][h] + terms[(b, a)][h]).cpu()),
                })
    return rows


def group_write_stats(gw: torch.Tensor, regmask: torch.Tensor, P: int, stim_ids: Sequence[str],
                      model: str, axis_a: int, axis_b: int):
    # gw [T,B,E]; all statistics are descriptive only and do not affect selection.
    rows = []
    x = gw.float().permute(1, 0, 2)  # [B,T,E]
    for i, sid in enumerate(stim_ids):
        groups = {
            "ordinary": (x[i, 1:1 + P][~regmask[i]] if int((~regmask[i]).sum()) else None),
            "register": (x[i, 1:1 + P][regmask[i]] if int(regmask[i].sum()) else None),
            "cls": x[i, 0:1],
            "extra": x[i, 1 + P:] if x.shape[1] > P + 1 else None,
        }
        for gname, v in groups.items():
            if v is None or v.numel() == 0:
                continue
            energy = float(v.square().sum().cpu())
            pair_energy = float((v[..., axis_a].square().sum() + v[..., axis_b].square().sum()).cpu())
            rows.append({
                "model": model,
                "stim_id": sid,
                "token_group": gname,
                "n_tokens": int(v.shape[0]),
                "write_rms_all_dims": float(v.square().mean().sqrt().cpu()),
                "mean_token_l2": float(v.norm(dim=-1).mean().cpu()),
                f"axis_{axis_a}_mean": float(v[..., axis_a].mean().cpu()),
                f"axis_{axis_a}_mean_abs": float(v[..., axis_a].abs().mean().cpu()),
                f"axis_{axis_a}_rms": float(v[..., axis_a].square().mean().sqrt().cpu()),
                f"axis_{axis_b}_mean": float(v[..., axis_b].mean().cpu()),
                f"axis_{axis_b}_mean_abs": float(v[..., axis_b].abs().mean().cpu()),
                f"axis_{axis_b}_rms": float(v[..., axis_b].square().mean().sqrt().cpu()),
                "pair_energy_fraction": pair_energy / max(energy, EPS),
            })
    return rows


@torch.inference_mode()
def forward_intervention(variant: Variant, images: torch.Tensor, stim_ids: Sequence[str], args,
                         group_ids: Sequence[int], mode: str):
    v = variant.visual
    blocks = list(v.transformer.resblocks)
    images = images.to(dtype=v.conv1.weight.dtype)
    x = v._prepare_tokens(images)
    P = x.shape[0] - 1
    regmask = None
    p21 = None
    p22 = None
    qkrows = []
    write_rows = []

    for li, blk in enumerate(blocks):
        x = variant.maybe_insert_rn(li, x)
        if li == args.register_block:
            regmask, _ = frozen_register_mask(
                x, P, args.register_threshold, args.max_registers, args.min_registers
            )
        ln1 = blk.ln_1(x)
        if li == args.qk_block:
            if regmask is None:
                raise RuntimeError("register mask missing before qk block")
            qkrows.extend(qk_pair_terms(
                blk, ln1, (args.axis_a, args.q_axis), regmask, P, li, stim_ids,
                variant.name, mode,
            ))
        need = li in {args.block + 1, args.qk_block}
        with amp_context(args.device, args.amp):
            attn, probs = blk.attention(ln1, need_weights=need, capture=False)
        pa = x + attn
        hh = None
        box = {}
        if li == args.block:
            if regmask is None:
                raise RuntimeError("register mask missing before B20 intervention")
            cp = resolve_c_proj(blk)

            def hook(mod, inp, output):
                h = inp[0]
                new, gw = transform_cproj_output(
                    output, h, mod, group_ids, mode, regmask, P,
                    args.axis_a, args.axis_b,
                )
                box["gw"] = gw.detach()
                return new

            hh = cp.register_forward_hook(hook)
        try:
            with amp_context(args.device, args.amp):
                mlp = blk.mlp(blk.ln_2(pa))
        finally:
            if hh is not None:
                hh.remove()
        x = pa + mlp
        if li == args.block:
            if "gw" not in box:
                raise RuntimeError("B20 c_proj hook did not capture group write")
            if mode == "baseline":
                write_rows.extend(group_write_stats(
                    box["gw"], regmask, P, stim_ids, variant.name, args.axis_a, args.axis_b
                ))
        if li == args.block + 1:
            p21 = probs.detach()
        if li == args.qk_block:
            p22 = probs.detach()

    with amp_context(args.device, args.amp):
        emb = v._finalize_cls(x)
    emb = F.normalize(emb.float(), dim=-1)
    if p21 is None or p22 is None or regmask is None:
        raise RuntimeError("required B21/B22 captures missing")
    return (
        emb,
        b21_rows(p21, regmask, P, stim_ids, variant.name, mode),
        b22_attention_rows(p22, regmask, P, stim_ids, variant.name, mode),
        qkrows,
        write_rows,
    )


def intervention_signature(variant: Variant, manifest: pd.DataFrame, args, group_ids: Sequence[int], mode: str):
    return {
        "format_version": FORMAT_VERSION,
        "model": variant.name,
        "mode": mode,
        "source": str(variant.source_info.get("source", "")),
        "group_name": args.group_name,
        "group_top_k": args.group_top_k,
        "group_ids": list(map(int, group_ids)),
        "axis_a": args.axis_a,
        "axis_b": args.axis_b,
        "q_axis": args.q_axis,
        "block": args.block,
        "qk_block": args.qk_block,
        "register_block": args.register_block,
        "n_images": len(manifest),
        "stim_ids": manifest["stim_id"].astype(str).tolist(),
        "amp": bool(args.amp),
    }


def run_cached_intervention(variant: Variant, manifest: pd.DataFrame, args, model_out: Path,
                            group_ids: Sequence[int], mode: str):
    cdir = model_out / "intervention_cache" / mode
    cdir.mkdir(parents=True, exist_ok=True)
    sig = intervention_signature(variant, manifest, args, group_ids, mode)
    sp = cdir / "signature.json"
    ep = cdir / "embeddings.safetensors"
    p21p = cdir / "b21_head_metrics.csv.gz"
    p22p = cdir / "b22_attention_metrics.csv.gz"
    qkp = cdir / "b22_qk_terms.csv.gz"
    wrp = cdir / "group_write_stats.csv.gz"
    needed = [sp, ep, p21p, p22p, qkp]
    if all(p.exists() for p in needed) and json.loads(sp.read_text()) == sig:
        emb = _load_st(str(ep))["embeddings"].float()
        p21 = pd.read_csv(p21p)
        p22 = pd.read_csv(p22p)
        qk = pd.read_csv(qkp)
        wr = pd.read_csv(wrp) if wrp.exists() else pd.DataFrame()
        print(f"[{variant.name}] {mode}: resume")
        return emb, p21, p22, qk, wr

    embs = []
    p21rows, p22rows, qkrows, writerows = [], [], [], []
    for st in range(0, len(manifest), args.batch_size):
        rows = manifest.iloc[st:st + args.batch_size]
        ims = load_batch(variant.preprocess, rows, args.device)
        e, a, b, q, w = forward_intervention(
            variant, ims, rows["stim_id"].astype(str).tolist(), args, group_ids, mode
        )
        embs.append(e.cpu())
        p21rows.extend(a)
        p22rows.extend(b)
        qkrows.extend(q)
        writerows.extend(w)
        print(f"[{variant.name}] {mode}: {min(st + len(rows), len(manifest))}/{len(manifest)}")
        del ims, e
        gc.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    emb = torch.cat(embs, 0)
    p21 = pd.DataFrame(p21rows)
    p22 = pd.DataFrame(p22rows)
    qk = pd.DataFrame(qkrows)
    wr = pd.DataFrame(writerows)
    save_st({"embeddings": emb}, ep, {"model": variant.name, "intervention": mode})
    p21.to_csv(p21p, index=False, compression="gzip")
    p22.to_csv(p22p, index=False, compression="gzip")
    qk.to_csv(qkp, index=False, compression="gzip")
    if not wr.empty:
        wr.to_csv(wrp, index=False, compression="gzip")
    sp.write_text(json.dumps(sig, indent=2))
    return emb, p21, p22, qk, wr


def mean_metric(df: pd.DataFrame, head: int, col: str):
    g = df[df["head"].eq(int(head))]
    return float(g[col].mean()) if len(g) else float("nan")


def choose_b22_gate_head(qk_base: pd.DataFrame, args):
    col = f"q{args.q_axis}_k{args.axis_a}"
    g = qk_base[qk_base["source_group"].eq("register")]
    if g.empty:
        return -1
    z = g.groupby("head")[col].apply(lambda x: np.mean(np.abs(x.to_numpy(float))))
    return int(z.idxmax())


def summarize_model(model: str, mode_data: Dict[str, Tuple[torch.Tensor, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
                    sharp_head: int, flat_head: int, args, out: Path):
    base_emb, base21, base22, baseqk, _ = mode_data["baseline"]
    gate_head = choose_b22_gate_head(baseqk, args)
    qcol = f"q{args.q_axis}_k{args.axis_a}"
    rows = []
    for mode, (emb, p21, p22, qk, _wr) in mode_data.items():
        cosd = 1.0 - (base_emb * emb).sum(-1)
        qreg = qk[qk["source_group"].eq("register") & qk["head"].eq(gate_head)]
        rows.append({
            "model": model,
            "intervention": mode,
            "n_group_neurons": args.group_top_k,
            "sharp_head": sharp_head,
            "flat_head": flat_head,
            "b22_gate_head": gate_head,
            "mean_final_cosine_distance_from_baseline": float(cosd.mean()),
            "max_final_cosine_distance_from_baseline": float(cosd.max()),
            "sharp_top1": mean_metric(p21, sharp_head, "top1"),
            "sharp_entropy": mean_metric(p21, sharp_head, "entropy"),
            "sharp_margin": mean_metric(p21, sharp_head, "margin"),
            "flat_top1": mean_metric(p21, flat_head, "top1"),
            "flat_entropy": mean_metric(p21, flat_head, "entropy"),
            "flat_margin": mean_metric(p21, flat_head, "margin"),
            "b22_gate_cls_to_register_mass": mean_metric(p22, gate_head, "cls_to_register_mass") if gate_head >= 0 else float("nan"),
            "b22_gate_cls_to_ordinary_mass": mean_metric(p22, gate_head, "cls_to_ordinary_mass") if gate_head >= 0 else float("nan"),
            f"b22_gate_{qcol}_signed": float(qreg[qcol].mean()) if len(qreg) else float("nan"),
            f"b22_gate_{qcol}_abs": float(np.mean(np.abs(qreg[qcol].to_numpy(float)))) if len(qreg) else float("nan"),
            "b22_gate_exact_full_qk_logit": float(qreg["exact_full_qk_logit"].mean()) if len(qreg) else float("nan"),
        })
    summary = pd.DataFrame(rows)
    base = summary.set_index("intervention").loc["baseline"]
    for c in [
        "sharp_top1", "sharp_entropy", "sharp_margin",
        "flat_top1", "flat_entropy", "flat_margin",
        "b22_gate_cls_to_register_mass", "b22_gate_cls_to_ordinary_mass",
        f"b22_gate_{qcol}_signed", "b22_gate_exact_full_qk_logit",
    ]:
        summary[f"delta_{c}"] = summary[c] - float(base[c])
    summary.to_csv(out / "intervention_summary.csv", index=False)

    # Mediation: ordinary ablation is the reference loss-of-function state.
    med_rows = []
    sm = summary.set_index("intervention")
    ab_name = "group_ablate_ordinary"
    if ab_name in sm.index:
        targets = [
            "sharp_top1", "flat_top1", "sharp_entropy", "flat_entropy",
            "b22_gate_cls_to_register_mass", f"b22_gate_{qcol}_signed",
        ]
        addbacks = [
            f"ordinary_ablate_plus_{args.axis_a}",
            f"ordinary_ablate_plus_{args.axis_b}",
            f"ordinary_ablate_plus_{args.axis_a}_{args.axis_b}",
            "ordinary_reconstruct",
        ]
        for metric in targets:
            b = float(sm.loc["baseline", metric])
            a = float(sm.loc[ab_name, metric])
            den = b - a
            for m in addbacks:
                if m not in sm.index:
                    continue
                x = float(sm.loc[m, metric])
                med_rows.append({
                    "model": model,
                    "metric": metric,
                    "addback": m,
                    "baseline": b,
                    "ordinary_ablate": a,
                    "addback_value": x,
                    "restoration_fraction": (x - a) / den if abs(den) > EPS else float("nan"),
                })
        removes = [
            f"ordinary_remove_{args.axis_a}",
            f"ordinary_remove_{args.axis_b}",
            f"ordinary_remove_{args.axis_a}_{args.axis_b}",
        ]
        for metric in targets:
            b = float(sm.loc["baseline", metric])
            a = float(sm.loc[ab_name, metric])
            den = a - b
            for m in removes:
                if m not in sm.index:
                    continue
                x = float(sm.loc[m, metric])
                med_rows.append({
                    "model": model,
                    "metric": metric,
                    "addback": m,
                    "baseline": b,
                    "ordinary_ablate": a,
                    "addback_value": x,
                    "restoration_fraction": (x - b) / den if abs(den) > EPS else float("nan"),
                })
    mediation = pd.DataFrame(med_rows, columns=MEDIATION_COLUMNS)
    mediation.to_csv(out / "mediation_ratios.csv", index=False)

    # Full B21 / B22 aggregate profiles.
    p21_all = pd.concat([v[1] for v in mode_data.values()], ignore_index=True)
    p22_all = pd.concat([v[2] for v in mode_data.values()], ignore_index=True)
    qk_all = pd.concat([v[3] for v in mode_data.values()], ignore_index=True)
    p21_all.groupby(["model", "intervention", "head"], as_index=False)[["top1", "entropy", "margin"]].mean().to_csv(out / "b21_head_profile.csv", index=False)
    p22_all.groupby(["model", "intervention", "head"], as_index=False)[["cls_top1", "cls_entropy", "cls_self_mass", "cls_to_register_mass", "cls_to_ordinary_mass", "cls_to_extra_mass"]].mean().to_csv(out / "b22_attention_profile.csv", index=False)
    qk_all.groupby(["model", "intervention", "source_group", "head"], as_index=False).mean(numeric_only=True).to_csv(out / "b22_qk_profile.csv", index=False)

    return summary, mediation, gate_head


def plot_model(out: Path, summary: pd.DataFrame, mediation: pd.DataFrame, sharp: int, flat: int, gate_head: int, args):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    p = out / "plots"
    p.mkdir(exist_ok=True)
    order = [m for m in DEFAULT_MODES if m in set(summary["intervention"])]
    g = summary.set_index("intervention").loc[order]
    x = np.arange(len(g))

    fig, ax = plt.subplots(figsize=(13, 7))
    w = 0.38
    ax.bar(x - w / 2, g["delta_sharp_top1"], width=w, label=f"Δ B21 H{sharp} top1")
    ax.bar(x + w / 2, g["delta_flat_top1"], width=w, label=f"Δ B21 H{flat} top1")
    ax.axhline(0, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_ylabel("change from baseline")
    ax.set_title("B20 push-pull coordinate mediation of B21 routing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(p / "01_B21_MEDIATION.png", dpi=200)
    plt.close(fig)

    hp = pd.read_csv(out / "b21_head_profile.csv")
    keep = ["baseline", "group_ablate_ordinary", f"ordinary_ablate_plus_{args.axis_a}_{args.axis_b}"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for mode in keep:
        z = hp[hp["intervention"].eq(mode)].sort_values("head")
        if len(z):
            ax.plot(z["head"], z["top1"], marker="o", label=mode)
    ax.set_xlabel("B21 head")
    ax.set_ylabel("ordinary-query mean top1")
    ax.set_title("B21 multi-head operating regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(p / "02_B21_HEAD_PROFILE.png", dpi=200)
    plt.close(fig)

    qp = pd.read_csv(out / "b22_qk_profile.csv")
    qcol = f"q{args.q_axis}_k{args.axis_a}"
    fig, ax = plt.subplots(figsize=(11, 6))
    for mode in keep:
        z = qp[qp["intervention"].eq(mode) & qp["source_group"].eq("register")].sort_values("head")
        if len(z):
            ax.plot(z["head"], z[qcol], marker="o", label=mode)
    if gate_head >= 0:
        ax.axvline(gate_head, ls="--", alpha=.5)
    ax.set_xlabel("B22 head")
    ax.set_ylabel(qcol + " on register keys")
    ax.set_title("B22 coordinate gate after B20/B21 intervention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(p / "03_B22_Q565_K650_PROFILE.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, g["mean_final_cosine_distance_from_baseline"])
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_ylabel("final embedding cosine distance")
    ax.set_title("Downstream effect of coordinate mediation interventions")
    fig.tight_layout()
    fig.savefig(p / "04_FINAL_EMBEDDING_DISTANCE.png", dpi=200)
    plt.close(fig)

    if not mediation.empty:
        z = mediation[
            mediation["metric"].isin(["sharp_top1", "flat_top1"])
            & mediation["addback"].str.startswith("ordinary_ablate_plus")
        ].copy()
        if len(z):
            labels = sorted(z["addback"].unique())
            xx = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(9, 6))
            ww = .38
            for j, metric in enumerate(["sharp_top1", "flat_top1"]):
                gg = z[z["metric"].eq(metric)].set_index("addback").reindex(labels)
                ax.bar(xx + (j - .5) * ww, gg["restoration_fraction"], width=ww, label=metric)
            ax.axhline(1.0, lw=1, ls="--")
            ax.axhline(0.0, lw=1)
            ax.set_xticks(xx)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("fraction of ordinary-ablation effect restored")
            ax.set_title("Coordinate add-back mediation")
            ax.legend()
            fig.tight_layout()
            fig.savefig(p / "05_RESTORATION_FRACTIONS.png", dpi=200)
            plt.close(fig)


def write_report(out: Path, model: str, group_ids: Sequence[int], sharp: int, flat: int,
                 gate_head: int, summary: pd.DataFrame, mediation: pd.DataFrame, args):
    sm = summary.set_index("intervention")
    lines = [
        "B20 PUSHPULL MEDIATION: 650 / 715",
        "=" * 42,
        "",
        f"model: {model}",
        f"group: {args.group_name}",
        f"group top-k: {len(group_ids)}",
        f"neurons: {','.join(map(str, group_ids))}",
        f"B21 diagnostic heads: sharp=H{sharp}, flat=H{flat}",
        f"B22 strongest |q{args.q_axis}*k{args.axis_a}| register head: H{gate_head}",
        "",
    ]
    for mode in summary["intervention"]:
        r = sm.loc[mode]
        lines.append(
            f"{mode:34s} final_d={r['mean_final_cosine_distance_from_baseline']:.7f} "
            f"ΔH{sharp}={r['delta_sharp_top1']:+.6f} ΔH{flat}={r['delta_flat_top1']:+.6f} "
            f"ΔB22reg={r['delta_b22_gate_cls_to_register_mass']:+.6f}"
        )
    if not mediation.empty:
        lines += ["", "MEDIATION RATIOS", "-"]
        for _, r in mediation.iterrows():
            if r["metric"] in {"sharp_top1", "flat_top1", f"b22_gate_q{args.q_axis}_k{args.axis_a}_signed"}:
                lines.append(f"{r['metric']:38s} {r['addback']:34s} {r['restoration_fraction']:+.4f}")
    (out / "B20_PUSHPULL_MEDIATION_650_715.txt").write_text("\n".join(lines), encoding="utf-8")


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return df if len(df.columns) else None


def combine(models: Sequence[str], root: Path):
    summaries, meds, writes = [], [], []
    for m in models:
        p = root / m / "intervention_summary.csv"
        q = root / m / "mediation_ratios.csv"
        w = root / m / "group_write_stats.csv.gz"

        # intervention_summary is the required proof that this model completed.
        if not p.exists():
            raise FileNotFoundError(f"missing required intervention summary for {m}: {p}")
        summaries.append(pd.read_csv(p))

        # Smoke/baseline-only runs legitimately have no mediation rows.  Older
        # failed smoke runs may also have left a headerless one-byte CSV.
        med = _read_optional_csv(q)
        if med is not None and not med.empty:
            meds.append(med)
        write = _read_optional_csv(w)
        if write is not None and not write.empty:
            writes.append(write)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(root / "ALL_MODELS_MEDIATION_SUMMARY.csv", index=False)
    if meds:
        pd.concat(meds, ignore_index=True).to_csv(root / "ALL_MODELS_MEDIATION_RATIOS.csv", index=False)
    if writes:
        pd.concat(writes, ignore_index=True).to_csv(root / "ALL_MODELS_GROUP_WRITE_STATS.csv.gz", index=False, compression="gzip")


def self_test():
    torch.manual_seed(0)
    T, B, M, E = 7, 2, 8, 12
    h = torch.randn(T, B, M)
    cp = torch.nn.Linear(M, E, bias=True)
    out = cp(h)
    regmask = torch.tensor([[False, True, False, False, False, False], [False, False, True, False, False, False]])
    ids = [1, 3, 5]
    a, b = 6, 7
    base, gw = transform_cproj_output(out, h, cp, ids, "baseline", regmask, 6, a, b)
    assert torch.allclose(base, out)
    recon, _ = transform_cproj_output(out, h, cp, ids, "ordinary_reconstruct", regmask, 6, a, b)
    assert torch.allclose(recon, out, atol=1e-6, rtol=1e-6)
    ab, _ = transform_cproj_output(out, h, cp, ids, "group_ablate_ordinary", regmask, 6, a, b)
    add, _ = transform_cproj_output(out, h, cp, ids, f"ordinary_ablate_plus_{a}_{b}", regmask, 6, a, b)
    om = ordinary_token_mask(regmask, T, 6, out.device).expand_as(out)
    # Outside ordinary tokens every ordinary-scoped intervention is identity.
    assert torch.allclose(ab[~om], out[~om])
    # Pair add-back differs from ablation only at the requested coordinates.
    delta = (add - ab)[om].reshape(-1, E)
    other = [i for i in range(E) if i not in {a, b}]
    assert torch.allclose(delta[:, other], torch.zeros_like(delta[:, other]), atol=1e-6)
    # All-token ablation equals literal subtraction of selected-neuron write.
    allab, _ = transform_cproj_output(out, h, cp, ids, "group_ablate_all", regmask, 6, a, b)
    assert torch.allclose(allab, out - gw, atol=1e-6, rtol=1e-6)
    print("self-test passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", default="")
    p.add_argument("--sharp_flat_root", default="out_paper_reproduction/conv1/b20_sharpeners_flatteners")
    p.add_argument("--output_dir", default="out_paper_reproduction/conv1/b20_pushpull_650_715")
    p.add_argument("--image_dir", default="image_sets/special_natural")
    p.add_argument("--models", default="pretrained,gmp,bare_xattn,full_xattn")
    p.add_argument("--pretrained_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--gmp_checkpoint", default="zer0int/CLIP-GmP-ViT-L-14")
    p.add_argument("--xattn_model", default="zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX")
    p.add_argument("--xattn_revision", default="")
    p.add_argument("--hf_cache_dir", default="")
    p.add_argument("--group_name", default="pushpull_sharpH_up_flatH_down")
    p.add_argument("--group_top_k", type=int, default=16)
    p.add_argument("--axis_a", type=int, default=650)
    p.add_argument("--axis_b", type=int, default=715)
    p.add_argument("--q_axis", type=int, default=565, help="query coordinate paired with axis_a in B22 QK audit")
    p.add_argument("--block", type=int, default=20)
    p.add_argument("--qk_block", type=int, default=22)
    p.add_argument("--register_block", type=int, default=13)
    p.add_argument("--register_threshold", type=float, default=70.0)
    p.add_argument("--max_registers", type=int, default=4)
    p.add_argument("--min_registers", type=int, default=1)
    p.add_argument("--modes", default=",".join(DEFAULT_MODES))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return 0

    args.models = parse_strs(args.models)
    args.modes = parse_strs(args.modes)
    # Rewrite default mode names if custom axes are supplied.
    replacements = {
        "ordinary_ablate_plus_650": f"ordinary_ablate_plus_{args.axis_a}",
        "ordinary_ablate_plus_715": f"ordinary_ablate_plus_{args.axis_b}",
        "ordinary_ablate_plus_650_715": f"ordinary_ablate_plus_{args.axis_a}_{args.axis_b}",
        "ordinary_remove_650": f"ordinary_remove_{args.axis_a}",
        "ordinary_remove_715": f"ordinary_remove_{args.axis_b}",
        "ordinary_remove_650_715": f"ordinary_remove_{args.axis_a}_{args.axis_b}",
    }
    args.modes = [replacements.get(x, x) for x in args.modes]

    sfroot = Path(args.sharp_flat_root)
    outroot = Path(args.output_dir)
    outroot.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(sfroot, args.image_dir, args.max_images)
    manifest.to_csv(outroot / "image_manifest.csv", index=False)

    for name in args.models:
        print(f"\n=== MODEL {name} ===")
        group_ids, sharp, flat, prior = load_group_spec(sfroot, name, args.group_name, args.group_top_k)
        print(f"[{name}] neurons n={len(group_ids)}: {group_ids}")
        print(f"[{name}] diagnostic B21 heads H{sharp} / H{flat}")
        variant = load_variant(name, args)
        md = outroot / name
        md.mkdir(parents=True, exist_ok=True)
        mode_data = {}
        write_df = pd.DataFrame()
        for mode in args.modes:
            mode_data[mode] = run_cached_intervention(variant, manifest, args, md, group_ids, mode)
            if mode == "baseline" and not mode_data[mode][4].empty:
                write_df = mode_data[mode][4]
        if not write_df.empty:
            write_df.to_csv(md / "group_write_stats.csv.gz", index=False, compression="gzip")
        summary, mediation, gate_head = summarize_model(name, mode_data, sharp, flat, args, md)
        plot_model(md, summary, mediation, sharp, flat, gate_head, args)
        write_report(md, name, group_ids, sharp, flat, gate_head, summary, mediation, args)
        (md / "audit.json").write_text(json.dumps({
            "format_version": FORMAT_VERSION,
            "model": name,
            "source_info": variant.source_info,
            "sharp_flat_root": str(sfroot),
            "group_name": args.group_name,
            "group_top_k": len(group_ids),
            "group_ids": group_ids,
            "sharp_head": sharp,
            "flat_head": flat,
            "axis_a": args.axis_a,
            "axis_b": args.axis_b,
            "q_axis": args.q_axis,
            "qk_block": args.qk_block,
            "n_images": len(manifest),
            "modes": args.modes,
        }, indent=2))
        del variant
        gc.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    combine(args.models, outroot)
    print(f"\nDone -> {outroot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
