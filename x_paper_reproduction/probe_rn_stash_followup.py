#!/usr/bin/env python3
'\nRN stash-impersonation / B11->B12->B13 follow-up\n=================================================\n\nQuestions\n---------\n1. Does READ_NULL (RN), after B13 LN1, resemble the native B12 high-norm\n   stash/register source in K-space, V-space, query-addressability, or W_O V\n   payload space?\n2. Does B13 use RN as a substitute source for the native stash, especially when\n   the known B11/B12 register-pump MLP units are ablated?\n3. Where does the low-rank RN pulse go spatially after B13? Which patch queries\n   read RN, and where do the B13 low-rank directions remain visible downstream?\n4. Is this B13 stash/readout geometry conserved in vanilla OpenAI ViT-L/14 when\n   the exact trained RN token is transplanted into the same block?\n5. For B11/B12, can we quantitatively support the informal "plant -> blast ->\n   recover" account using register norm trajectories and direct contribution\n   of the known 4096-D post-QuickGELU pump neurons?\n\nThe script is intentionally low-memory: all scene statistics are streamed and\nonly small per-head/per-patch accumulators plus bounded B13 SVD rows are kept.\n'
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()


import argparse
import contextlib
import csv
import gc
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from x_paper_reproduction.rn_control_mechinterp.synthetic import PatchAlignedSyntheticBank
from x_paper_reproduction.rn_control_mechinterp.core import (
    VisualConditionRunner,
    import_attnclip,
    infer_model_autocast_dtype,
    model_autocast_context,
    preprocess_pil_batch,
)
from x_paper_reproduction.rn_control_mechinterp.analysis import randomized_svd_rows, principal_angles


DEFAULT_CHECKPOINT = "REPLACE_WITH_CHECKPOINT.pt"
REG_NEURONS_BLOCK_11 = [9, 987, 1100, 1967, 2555, 3661, 3784]
REG_NEURONS_BLOCK_12 = [42, 183, 983, 1571, 1816, 2687, 3002, 3008, 3868]
PUMP_CONDITIONS = ("intact", "zero_b11", "zero_b12", "zero_b11_b12")


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_strs(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def iter_batches(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield list(range(start, min(n, start + batch_size)))


def stable_slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text)).strip("._") or "x"


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cosine(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=dim, eps=eps)


def pearson_rows(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Pearson r along the final dimension."""
    af = a.float() - a.float().mean(dim=-1, keepdim=True)
    bf = b.float() - b.float().mean(dim=-1, keepdim=True)
    num = (af * bf).sum(dim=-1)
    den = af.square().sum(dim=-1).sqrt() * bf.square().sum(dim=-1).sqrt()
    return num / den.clamp_min(eps)


def group_mean(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    acc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    literals: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        k = tuple(row[x] for x in keys)
        literals[k] = {x: row[x] for x in keys}
        for name, value in row.items():
            if name in keys:
                continue
            if isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
                acc[k][name].append(float(value))
    out = []
    for k in sorted(acc, key=lambda x: tuple(str(y) for y in x)):
        row = dict(literals[k])
        for name, vals in acc[k].items():
            row[name] = float(np.mean(vals))
        out.append(row)
    return out


@dataclass
class ModelBundle:
    name: str
    model: torch.nn.Module
    preprocess: Any
    runner: VisualConditionRunner
    device: torch.device


def load_trained(checkpoint: str, module_root: str, device: str) -> tuple[ModelBundle, Any]:
    clip_mod = import_attnclip(module_root)
    model, preprocess = clip_mod.load(checkpoint, device=device, jit=False)
    model.eval()
    if getattr(model.visual, "read_null_token", None) is None:
        raise RuntimeError("Trained checkpoint does not contain visual.read_null_token")
    bundle = ModelBundle(
        name="trained",
        model=model,
        preprocess=preprocess,
        runner=VisualConditionRunner(model),
        device=torch.device(device),
    )
    return bundle, clip_mod


def load_pretrained_with_same_rn(
    clip_mod: Any,
    *,
    pretrained_rn_checkpoint: str,
    trained: ModelBundle,
    device: str,
) -> ModelBundle:
    """Load the explicit vanilla+trained-RN variant; never initialize RN/bridge randomly."""
    path = Path(pretrained_rn_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"Vanilla+RN variant checkpoint not found: {path}. "
            "Use reproduce.py so the explicit oai_vanilla_rn_from_xattn variant is materialized."
        )
    model, preprocess = clip_mod.load(str(path), device=device, jit=False)
    model.eval()
    if getattr(model.visual, "read_null_token", None) is None:
        raise RuntimeError("Vanilla+RN variant has no visual.read_null_token")
    if getattr(model, "read_implant", None) is not None:
        raise RuntimeError(
            "Vanilla+RN control unexpectedly contains a bridge/read_implant. "
            "This control must contain the trained RN only, with NO bridge."
        )
    implant_kind = getattr(model, "implant_kind", "none")
    if str(implant_kind) != "none":
        raise RuntimeError(f"Vanilla+RN control has implant_kind={implant_kind!r}; expected 'none'")

    donor_rn = trained.model.visual.read_null_token.detach().float().cpu()
    receiver_rn = model.visual.read_null_token.detach().float().cpu()
    if donor_rn.shape != receiver_rn.shape or not torch.equal(donor_rn, receiver_rn):
        raise RuntimeError(
            "Vanilla+RN control does not contain the exact trained donor RN token. "
            "Refusing a mismatched or randomly initialized RN."
        )
    donor_insert = int(trained.runner.insert_block)
    receiver_insert = int(getattr(model.visual, "read_null_insert_block", -1))
    if receiver_insert != donor_insert:
        raise RuntimeError(
            f"Vanilla+RN insert block mismatch: donor pre-B{donor_insert}, receiver pre-B{receiver_insert}"
        )

    return ModelBundle(
        name="pretrained",
        model=model,
        preprocess=preprocess,
        runner=VisualConditionRunner(model),
        device=torch.device(device),
    )


def pump_zero_map(condition: str) -> dict[int, list[int]]:
    if condition == "intact":
        return {}
    if condition == "zero_b11":
        return {11: REG_NEURONS_BLOCK_11}
    if condition == "zero_b12":
        return {12: REG_NEURONS_BLOCK_12}
    if condition == "zero_b11_b12":
        return {11: REG_NEURONS_BLOCK_11, 12: REG_NEURONS_BLOCK_12}
    raise ValueError(condition)


@contextlib.contextmanager
def zero_mlp_neurons_context(model: torch.nn.Module, condition: str):
    """Zero selected post-QuickGELU MLP units during ordinary model forwards."""
    mapping = pump_zero_map(condition)
    handles = []
    try:
        for block_idx, units in mapping.items():
            gelu = model.visual.transformer.resblocks[block_idx].mlp.gelu
            idx = torch.as_tensor(units, dtype=torch.long)

            def hook(_module, _inputs, output, idx_cpu=idx):
                y = output.clone()
                local = idx_cpu.to(device=y.device)
                y[..., local] = 0
                return y

            handles.append(gelu.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def forward_block_probe(
    blk: torch.nn.Module,
    x_tbc: torch.Tensor,
    *,
    measure_units: Optional[list[int]] = None,
    zero_units: Optional[list[int]] = None,
    capture_attention: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact ResidualAttentionBlock forward with compact MLP diagnostics."""
    ln1 = blk.ln_1(x_tbc)
    attn_out, _ = blk.attention(ln1, capture=capture_attention)
    mid = x_tbc + attn_out
    ln2 = blk.ln_2(mid)
    hidden = blk.mlp.c_fc(ln2)
    hidden = blk.mlp.gelu(hidden)

    selected_abs = torch.zeros(hidden.shape[:-1], device=hidden.device, dtype=torch.float32)
    selected_contrib = torch.zeros((*hidden.shape[:-1], x_tbc.shape[-1]), device=hidden.device, dtype=torch.float32)
    measured_idx = None
    if measure_units:
        measured_idx = torch.as_tensor(measure_units, device=hidden.device, dtype=torch.long)
        selected = hidden.index_select(-1, measured_idx)
        selected_abs = selected.float().abs().mean(dim=-1)
        weight = blk.mlp.c_proj.weight.index_select(1, measured_idx)
        # Diagnostic-only projection: execute explicitly in FP32.  The mechinterp
        # model intentionally mixes FP32 MLP islands with FP16 attention/storage,
        # and this helper may be called with tensors whose storage dtypes differ.
        selected_contrib = F.linear(selected.float(), weight.float(), bias=None)

    mlp_full = blk.mlp.c_proj(hidden)
    hidden_used = hidden
    if zero_units:
        zero_idx = torch.as_tensor(zero_units, device=hidden.device, dtype=torch.long)
        hidden_used = hidden.clone()
        hidden_used[..., zero_idx] = 0
    mlp_used = blk.mlp.c_proj(hidden_used)
    post = mid + mlp_used

    return post, {
        "pre": x_tbc.detach(),
        "mid": mid.detach(),
        "post": post.detach(),
        "attn_out": attn_out.detach(),
        "mlp_full": mlp_full.detach(),
        "mlp_used": mlp_used.detach(),
        "selected_abs": selected_abs.detach(),
        "selected_contrib": selected_contrib.detach(),
    }


def clone_attn_cache(attn: torch.nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for name in ("last_q", "last_k", "last_v", "last_logits", "last_probs", "last_z"):
        value = getattr(attn, name, None)
        if value is None:
            raise RuntimeError(f"Attention cache missing {name}; capture=True was expected")
        out[name] = value.detach().clone()
    return out


def register_mask_from_pre_b13(
    x_tbc: torch.Tensor,
    *,
    threshold: float,
    max_registers: int,
    min_registers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return [B,P] mask, [B] primary patch index, [B,P] norms."""
    patches = x_tbc[1:].permute(1, 0, 2).float()
    norms = patches.norm(dim=-1)
    bsz, patches_n = norms.shape
    mask = norms > float(threshold)
    out = torch.zeros_like(mask)
    primary = norms.argmax(dim=1)

    for b in range(bsz):
        idx = torch.where(mask[b])[0]
        if idx.numel() < min_registers:
            k = min(min_registers, patches_n)
            idx = torch.topk(norms[b], k=k, largest=True).indices
        elif max_registers > 0 and idx.numel() > max_registers:
            vals = norms[b].index_select(0, idx)
            keep = torch.topk(vals, k=max_registers, largest=True).indices
            idx = idx.index_select(0, keep)
        out[b, idx] = True
    return out, primary, norms


def gather_token_by_patch_index(tbc: torch.Tensor, patch_index_b: torch.Tensor) -> torch.Tensor:
    # tbc [T,B,C], patch idx [B] (0-based spatial patch)
    bsz = tbc.shape[1]
    src = 1 + patch_index_b.to(device=tbc.device)
    batch = torch.arange(bsz, device=tbc.device)
    btd = tbc.permute(1, 0, 2)
    return btd[batch, src]


def gather_map_value_bt(x_bt: torch.Tensor, patch_index_b: torch.Tensor) -> torch.Tensor:
    # x [T,B] -> [B] at source token 1+patch index
    bsz = x_bt.shape[1]
    src = 1 + patch_index_b.to(device=x_bt.device)
    batch = torch.arange(bsz, device=x_bt.device)
    return x_bt.permute(1, 0)[batch, src]


def gather_selected_register_mean(cache_bhtd: torch.Tensor, reg_mask_bp: torch.Tensor) -> torch.Tensor:
    """cache [B,H,T,D], patch register mask [B,P] -> mean [B,H,D]."""
    bsz, heads, _tokens, dim = cache_bhtd.shape
    out = torch.zeros((bsz, heads, dim), device=cache_bhtd.device, dtype=cache_bhtd.dtype)
    for b in range(bsz):
        patch_idx = torch.where(reg_mask_bp[b])[0]
        src = patch_idx + 1
        out[b] = cache_bhtd[b, :, src, :].mean(dim=1)
    return out


def attention_to_registers(probs_bhts: torch.Tensor, reg_mask_bp: torch.Tensor) -> torch.Tensor:
    """Sum attention to per-image native registers -> [B,H,Tquery]."""
    bsz, heads, tq, _ts = probs_bhts.shape
    out = torch.zeros((bsz, heads, tq), device=probs_bhts.device, dtype=probs_bhts.dtype)
    for b in range(bsz):
        src = torch.where(reg_mask_bp[b])[0] + 1
        out[b] = probs_bhts[b, :, :, src].sum(dim=-1)
    return out


def register_value_contribution(
    probs_bhts: torch.Tensor,
    values_bhsd: torch.Tensor,
    reg_mask_bp: torch.Tensor,
) -> torch.Tensor:
    """Per-head register-only weighted V contribution [B,H,T,D]."""
    bsz, heads, tq, _ts = probs_bhts.shape
    dim = values_bhsd.shape[-1]
    out = torch.zeros((bsz, heads, tq, dim), device=probs_bhts.device, dtype=values_bhsd.dtype)
    for b in range(bsz):
        src = torch.where(reg_mask_bp[b])[0] + 1
        p = probs_bhts[b, :, :, src]          # [H,T,R]
        v = values_bhsd[b, :, src, :]         # [H,R,D]
        out[b] = torch.einsum("htr,hrd->htd", p, v)
    return out


def head_out_payload(blk: torch.nn.Module, value_bhd: torch.Tensor) -> torch.Tensor:
    """Apply each head's corresponding out_proj slice independently: [B,H,D] -> [B,H,C]."""
    bsz, heads, dim = value_bhd.shape
    width = heads * dim
    weight = blk.attn.out_proj.weight
    chunks = []
    for h in range(heads):
        w = weight[:, h * dim:(h + 1) * dim]
        # This is an analysis projection, not a model forward.  Keep it in FP32
        # so cached V vectors (promoted to float for diagnostics) can be compared
        # against half-precision out_proj weights without dtype-dependent failure.
        chunks.append(F.linear(value_bhd[:, h].float(), w.float(), bias=None))
    return torch.stack(chunks, dim=1)


def region_patch_mask(specs: list[Any], grid: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(specs), grid * grid), dtype=torch.bool, device=device)
    for b, spec in enumerate(specs):
        x0, y0, x1, y1 = spec.bbox_patch
        for y in range(y0, y1):
            mask[b, y * grid + x0:y * grid + x1] = True
    return mask


def query_group_means(values_bht: torch.Tensor, region_bp: torch.Tensor) -> dict[str, torch.Tensor]:
    """values [B,H,T], T=CLS+patches (+RN possibly). Only ordinary queries used."""
    ordinary = values_bht[:, :, :1 + region_bp.shape[1]]
    patch = ordinary[:, :, 1:]
    result = {
        "cls": ordinary[:, :, 0],
        "patch": patch.mean(dim=-1),
    }
    # Per-image region/outside means -> [B,H]
    reg_vals, out_vals = [], []
    for b in range(patch.shape[0]):
        reg_vals.append(patch[b, :, region_bp[b]].mean(dim=-1))
        out_vals.append(patch[b, :, ~region_bp[b]].mean(dim=-1))
    result["region"] = torch.stack(reg_vals, dim=0)
    result["outside"] = torch.stack(out_vals, dim=0)
    return result


def token_group_norm_means(values_bhtd: torch.Tensor, region_bp: torch.Tensor) -> dict[str, torch.Tensor]:
    norms = values_bhtd.float().norm(dim=-1)
    return query_group_means(norms, region_bp)


def run_pre_b13_with_pump_probe(
    bundle: ModelBundle,
    images: torch.Tensor,
    *,
    pump_condition: str,
) -> dict[str, Any]:
    model = bundle.model
    visual = model.visual
    zero_map = pump_zero_map(pump_condition)
    states: dict[int, torch.Tensor] = {}
    pump: dict[int, dict[str, torch.Tensor]] = {}

    with torch.no_grad(), model_autocast_context(model):
        x = visual._prepare_tokens(images.type(model.dtype))
        for i, blk in enumerate(visual.transformer.resblocks[:13]):
            if i in (11, 12):
                measure = REG_NEURONS_BLOCK_11 if i == 11 else REG_NEURONS_BLOCK_12
                x, info = forward_block_probe(blk, x, measure_units=measure, zero_units=zero_map.get(i))
                pump[i] = info
            else:
                x = blk(x)
            if i in (10, 11, 12):
                states[i] = x.detach().clone()

    return {"pre_b13": x.detach().clone(), "states": states, "pump": pump}


def run_b13_pair_from_pre(bundle: ModelBundle, x_pre: torch.Tensor) -> dict[str, Any]:
    model = bundle.model
    blk = model.visual.transformer.resblocks[13]
    rn = model.visual.read_null_token

    with torch.no_grad(), model_autocast_context(model):
        base_post, base_info = forward_block_probe(blk, x_pre, capture_attention=True)
        base_cache = clone_attn_cache(blk.attn)

        token = rn.to(device=x_pre.device, dtype=x_pre.dtype).view(1, 1, -1).expand(1, x_pre.shape[1], -1)
        x_rn = torch.cat([x_pre, token], dim=0)
        rn_post_full, rn_info = forward_block_probe(blk, x_rn, capture_attention=True)
        rn_cache = clone_attn_cache(blk.attn)
        rn_post = rn_post_full[:-1]

    return {
        "base_post": base_post,
        "rn_post": rn_post,
        "base_info": base_info,
        "rn_info": rn_info,
        "base_cache": base_cache,
        "rn_cache": rn_cache,
    }


def primary_register_trajectory_rows(
    *,
    bundle_name: str,
    variant: str,
    pump_condition: str,
    pre_result: dict[str, Any],
    b13_pair: dict[str, Any],
    primary_patch_b: torch.Tensor,
    reg_mask_bp: torch.Tensor,
    patch_norms_bp: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    stages = {
        "B10": pre_result["states"][10],
        "B11": pre_result["states"][11],
        "B12": pre_result["states"][12],
        "B13_base": b13_pair["base_post"],
        "B13_RN": b13_pair["rn_post"],
    }
    for stage, state in stages.items():
        primary = gather_token_by_patch_index(state, primary_patch_b).float().norm(dim=-1)
        patch = state[1:].permute(1, 0, 2).float().norm(dim=-1)
        rows.append({
            "model": bundle_name,
            "variant": variant,
            "pump_condition": pump_condition,
            "stage": stage,
            "primary_reg_norm": float(primary.mean().cpu()),
            "max_patch_norm": float(patch.max(dim=1).values.mean().cpu()),
            "mean_patch_norm": float(patch.mean().cpu()),
            "register_count": float(reg_mask_bp.sum(dim=1).float().mean().cpu()) if stage == "B12" else float("nan"),
            "b12_primary_norm_reference": float(patch_norms_bp.max(dim=1).values.mean().cpu()),
        })
    return rows


def pump_metric_rows(
    *,
    bundle_name: str,
    variant: str,
    pump_condition: str,
    pre_result: dict[str, Any],
    primary_patch_b: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for block_idx in (11, 12):
        info = pre_result["pump"][block_idx]
        contrib_norm_tb = info["selected_contrib"].float().norm(dim=-1)
        mlp_norm_tb = info["mlp_full"].float().norm(dim=-1)
        abs_tb = info["selected_abs"].float()
        src = 1 + primary_patch_b.to(device=contrib_norm_tb.device)
        batch = torch.arange(contrib_norm_tb.shape[1], device=contrib_norm_tb.device)
        c_bt = contrib_norm_tb.permute(1, 0)
        m_bt = mlp_norm_tb.permute(1, 0)
        a_bt = abs_tb.permute(1, 0)
        primary_c = c_bt[batch, src]
        primary_m = m_bt[batch, src]
        primary_a = a_bt[batch, src]

        rows.append({
            "model": bundle_name,
            "variant": variant,
            "pump_condition": pump_condition,
            "block": block_idx,
            "token_group": "primary_register",
            "selected_contrib_norm": float(primary_c.mean().cpu()),
            "full_mlp_norm": float(primary_m.mean().cpu()),
            "selected_over_mlp": float((primary_c / primary_m.clamp_min(1e-8)).mean().cpu()),
            "selected_activation_abs": float(primary_a.mean().cpu()),
        })
        rows.append({
            "model": bundle_name,
            "variant": variant,
            "pump_condition": pump_condition,
            "block": block_idx,
            "token_group": "patch_mean",
            "selected_contrib_norm": float(c_bt[:, 1:].mean().cpu()),
            "full_mlp_norm": float(m_bt[:, 1:].mean().cpu()),
            "selected_over_mlp": float((c_bt[:, 1:] / m_bt[:, 1:].clamp_min(1e-8)).mean().cpu()),
            "selected_activation_abs": float(a_bt[:, 1:].mean().cpu()),
        })
        rows.append({
            "model": bundle_name,
            "variant": variant,
            "pump_condition": pump_condition,
            "block": block_idx,
            "token_group": "cls",
            "selected_contrib_norm": float(c_bt[:, 0].mean().cpu()),
            "full_mlp_norm": float(m_bt[:, 0].mean().cpu()),
            "selected_over_mlp": float((c_bt[:, 0] / m_bt[:, 0].clamp_min(1e-8)).mean().cpu()),
            "selected_activation_abs": float(a_bt[:, 0].mean().cpu()),
        })
    return rows


def b13_stash_rows_and_maps(
    *,
    bundle: ModelBundle,
    variant: str,
    pump_condition: str,
    x_pre: torch.Tensor,
    b13_pair: dict[str, Any],
    reg_mask_bp: torch.Tensor,
    primary_patch_b: torch.Tensor,
    region_bp: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    blk = bundle.model.visual.transformer.resblocks[13]
    base = b13_pair["base_cache"]
    rn = b13_pair["rn_cache"]

    q = base["last_q"].float()                   # [B,H,T,D], scaled q
    k_base = base["last_k"].float()
    v_base = base["last_v"].float()
    probs_base = base["last_probs"].float()
    probs_rn = rn["last_probs"].float()
    k_rn = rn["last_k"][:, :, -1, :].float()
    v_rn = rn["last_v"][:, :, -1, :].float()

    k_reg = gather_selected_register_mean(k_base, reg_mask_bp).float()
    v_reg = gather_selected_register_mean(v_base, reg_mask_bp).float()

    # Post-LN comparison in residual feature space.
    with torch.no_grad(), model_autocast_context(bundle.model):
        ln_pre = blk.ln_1(x_pre)
        u_reg = []
        for b in range(x_pre.shape[1]):
            src = torch.where(reg_mask_bp[b])[0] + 1
            u_reg.append(ln_pre[src, b].mean(dim=0))
        u_reg = torch.stack(u_reg, dim=0).float()
        raw_rn = bundle.model.visual.read_null_token.to(device=x_pre.device, dtype=x_pre.dtype)
        u_rn_single = blk.ln_1(raw_rn.view(1, 1, -1)).view(-1).float()
        u_rn = u_rn_single.view(1, -1).expand(x_pre.shape[1], -1)

    # Query-addressability profiles, using the exact scaled q from the block.
    rn_logits = torch.einsum("bhtd,bhd->bht", q, k_rn)
    reg_logits = torch.einsum("bhtd,bhd->bht", q, k_reg)
    # Ordinary queries only; RN's own query is not part of the tag-and-run mechanism.
    rn_logits_patch = rn_logits[:, :, 1:]
    reg_logits_patch = reg_logits[:, :, 1:]
    profile_corr = pearson_rows(rn_logits_patch, reg_logits_patch)  # [B,H]
    profile_cos = cosine(rn_logits_patch, reg_logits_patch, dim=-1)

    attn_reg_base = attention_to_registers(probs_base, reg_mask_bp)
    attn_reg_rn = attention_to_registers(probs_rn, reg_mask_bp)
    attn_rn = probs_rn[:, :, :1 + region_bp.shape[1], -1]

    rn_write = attn_rn.unsqueeze(-1) * v_rn.unsqueeze(2)
    reg_write_base = register_value_contribution(probs_base, v_base, reg_mask_bp)
    reg_write_rn = register_value_contribution(probs_rn, rn["last_v"].float(), reg_mask_bp)

    out_rn = head_out_payload(blk, v_rn)
    out_reg = head_out_payload(blk, v_reg)

    kcos = cosine(k_rn, k_reg, dim=-1)
    vcos = cosine(v_rn, v_reg, dim=-1)
    outcos = cosine(out_rn, out_reg, dim=-1)
    postln_cos = cosine(u_rn, u_reg, dim=-1)

    groups_rn = query_group_means(attn_rn, region_bp)
    groups_reg_base = query_group_means(attn_reg_base, region_bp)
    groups_reg_rn = query_group_means(attn_reg_rn, region_bp)
    groups_rn_write = token_group_norm_means(rn_write, region_bp)
    groups_reg_write_base = token_group_norm_means(reg_write_base, region_bp)
    groups_reg_write_rn = token_group_norm_means(reg_write_rn, region_bp)

    rows = []
    heads = q.shape[1]
    for h in range(heads):
        for group in ("cls", "patch", "region", "outside"):
            rows.append({
                "model": bundle.name,
                "variant": variant,
                "pump_condition": pump_condition,
                "head": h,
                "query_group": group,
                "postln_rn_vs_reg_cos": float(postln_cos.mean().cpu()),
                "k_rn_vs_reg_cos": float(kcos[:, h].mean().cpu()),
                "v_rn_vs_reg_cos": float(vcos[:, h].mean().cpu()),
                "wo_v_rn_vs_reg_cos": float(outcos[:, h].mean().cpu()),
                "k_rn_norm": float(k_rn[:, h].norm(dim=-1).mean().cpu()),
                "k_reg_norm": float(k_reg[:, h].norm(dim=-1).mean().cpu()),
                "v_rn_norm": float(v_rn[:, h].norm(dim=-1).mean().cpu()),
                "v_reg_norm": float(v_reg[:, h].norm(dim=-1).mean().cpu()),
                "query_logit_profile_corr": float(profile_corr[:, h].mean().cpu()),
                "query_logit_profile_cos": float(profile_cos[:, h].mean().cpu()),
                "attn_to_rn": float(groups_rn[group][:, h].mean().cpu()),
                "attn_to_reg_base": float(groups_reg_base[group][:, h].mean().cpu()),
                "attn_to_reg_with_rn": float(groups_reg_rn[group][:, h].mean().cpu()),
                "rn_value_write_norm": float(groups_rn_write[group][:, h].mean().cpu()),
                "reg_value_write_norm_base": float(groups_reg_write_base[group][:, h].mean().cpu()),
                "reg_value_write_norm_with_rn": float(groups_reg_write_rn[group][:, h].mean().cpu()),
            })

    # Maps are sums over the batch; caller will average over all scenes.
    maps = {
        "rn": attn_rn[:, :, 1:].sum(dim=0).detach().cpu().numpy(),             # [H,P]
        "reg_base": attn_reg_base[:, :, 1:].sum(dim=0).detach().cpu().numpy(),
        "reg_with_rn": attn_reg_rn[:, :, 1:1 + region_bp.shape[1]].sum(dim=0).detach().cpu().numpy(),
        "rn_logit": rn_logits[:, :, 1:].sum(dim=0).detach().cpu().numpy(),
        "reg_logit": reg_logits[:, :, 1:].sum(dim=0).detach().cpu().numpy(),
        "count": int(x_pre.shape[1]),
    }

    # Compact post-LN/K/V arrays for later cross-model comparison.
    compact = {
        "u_rn": u_rn_single.detach().cpu().numpy(),
        "k_rn_mean": k_rn.mean(dim=0).detach().cpu().numpy(),
        "v_rn_mean": v_rn.mean(dim=0).detach().cpu().numpy(),
        "out_rn_mean": out_rn.mean(dim=0).detach().cpu().numpy(),
    }
    return rows, maps, compact


def merge_map_accumulator(acc: dict[str, np.ndarray], maps: dict[str, np.ndarray]) -> None:
    for key in ("rn", "reg_base", "reg_with_rn", "rn_logit", "reg_logit"):
        if key not in acc:
            acc[key] = np.zeros_like(maps[key], dtype=np.float64)
        acc[key] += maps[key]
    acc["count"] = acc.get("count", 0.0) + float(maps["count"])


def plot_head_lines(rows: list[dict[str, Any]], out_path: Path, *, title: str, metrics: list[tuple[str, str]]) -> None:
    if not rows:
        return
    heads = sorted({int(r["head"]) for r in rows})
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for key, label in metrics:
        ys = []
        for h in heads:
            vals = [float(r[key]) for r in rows if int(r["head"]) == h and key in r]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(heads, ys, marker="o", label=label)
    ax.set_xlabel("B13 head")
    ax.set_title(title)
    ax.set_xticks(heads)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def plot_head_x_query_matrix(array_hp: np.ndarray, out_path: Path, *, title: str, grid: int) -> None:
    fig, ax = plt.subplots(figsize=(15, 5.2))
    im = ax.imshow(array_hp, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(f"query patch index (row-major {grid}x{grid})")
    ax.set_ylabel("B13 head")
    ax.set_yticks(np.arange(array_hp.shape[0]))
    ticks = np.arange(0, grid * grid, grid)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(x)) for x in ticks])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def draw_region_rect(ax, bbox_patch: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bbox_patch
    rect = plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, linewidth=1.5)
    ax.add_patch(rect)


def plot_all_head_spatial_maps(
    array_hp: np.ndarray,
    out_path: Path,
    *,
    title: str,
    grid: int,
    bbox_patch: Optional[tuple[int, int, int, int]],
) -> None:
    heads = array_hp.shape[0]
    ncol = int(math.ceil(math.sqrt(heads)))
    nrow = int(math.ceil(heads / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow), squeeze=False)
    vmax = float(np.nanmax(np.abs(array_hp)))
    vmin = float(np.nanmin(array_hp))
    for h in range(nrow * ncol):
        ax = axes[h // ncol][h % ncol]
        if h >= heads:
            ax.axis("off")
            continue
        im = ax.imshow(array_hp[h].reshape(grid, grid), interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"H{h}")
        ax.set_xticks([]); ax.set_yticks([])
        if bbox_patch is not None:
            draw_region_rect(ax, bbox_patch)
    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015)
    fig.subplots_adjust(top=0.92, wspace=0.08, hspace=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def plot_dynamite_norms(rows: list[dict[str, Any]], out_dir: Path) -> None:
    stage_order = ["B10", "B11", "B12", "B13_base", "B13_RN"]
    for model in sorted({r["model"] for r in rows}):
        for variant in sorted({r["variant"] for r in rows if r["model"] == model}):
            fig, ax = plt.subplots(figsize=(9.5, 5.2))
            for cond in PUMP_CONDITIONS:
                sub = [r for r in rows if r["model"] == model and r["variant"] == variant and r["pump_condition"] == cond]
                if not sub:
                    continue
                lookup = {r["stage"]: float(r["primary_reg_norm"]) for r in sub}
                ax.plot(stage_order, [lookup.get(s, np.nan) for s in stage_order], marker="o", label=cond)
            ax.set_ylabel("primary B12-register token norm")
            ax.set_title(f"B11→B12→B13 high-norm stash trajectory — {model} / {variant}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"01_dynamite_norm__{stable_slug(model)}__{stable_slug(variant)}.png", dpi=190)
            plt.close(fig)


def plot_pump_contrib(rows: list[dict[str, Any]], out_dir: Path) -> None:
    for model in sorted({r["model"] for r in rows}):
        for variant in sorted({r["variant"] for r in rows if r["model"] == model}):
            fig, ax = plt.subplots(figsize=(10.5, 5.2))
            labels, values = [], []
            for cond in PUMP_CONDITIONS:
                for block in (11, 12):
                    vals = [float(r["selected_over_mlp"]) for r in rows if r["model"] == model and r["variant"] == variant and r["pump_condition"] == cond and int(r["block"]) == block and r["token_group"] == "primary_register"]
                    if vals:
                        labels.append(f"{cond}\nB{block}")
                        values.append(float(np.mean(vals)))
            ax.bar(np.arange(len(values)), values)
            ax.set_xticks(np.arange(len(values)))
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_ylabel("||selected pump-neuron contribution|| / ||full MLP||")
            ax.set_title(f"Known register-neuron contribution on the primary stash token — {model} / {variant}")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"02_pump_contribution__{stable_slug(model)}__{stable_slug(variant)}.png", dpi=190)
            plt.close(fig)


def plot_stash_metrics(rows: list[dict[str, Any]], out_dir: Path) -> None:
    for model in sorted({r["model"] for r in rows}):
        for variant in sorted({r["variant"] for r in rows if r["model"] == model}):
            for cond in PUMP_CONDITIONS:
                sub = [r for r in rows if r["model"] == model and r["variant"] == variant and r["pump_condition"] == cond and r["query_group"] == "cls"]
                if not sub:
                    continue
                stem = f"{stable_slug(model)}__{stable_slug(variant)}__{stable_slug(cond)}"
                plot_head_lines(
                    sub, out_dir / f"03_stash_similarity__{stem}.png",
                    title=f"RN vs native B12 stash geometry — {model}/{variant}/{cond}",
                    metrics=[
                        ("k_rn_vs_reg_cos", "K cosine"),
                        ("v_rn_vs_reg_cos", "V cosine"),
                        ("wo_v_rn_vs_reg_cos", "W_O V cosine"),
                        ("query_logit_profile_corr", "query-logit profile r"),
                    ],
                )
                plot_head_lines(
                    sub, out_dir / f"04_source_substitution_cls__{stem}.png",
                    title=f"Who CLS reads at B13 — {model}/{variant}/{cond}",
                    metrics=[
                        ("attn_to_reg_base", "native register, RN off"),
                        ("attn_to_reg_with_rn", "native register, RN on"),
                        ("attn_to_rn", "RN"),
                    ],
                )
                plot_head_lines(
                    sub, out_dir / f"05_value_write_cls__{stem}.png",
                    title=f"CLS value payload norm at B13 — {model}/{variant}/{cond}",
                    metrics=[
                        ("reg_value_write_norm_base", "native register write, RN off"),
                        ("reg_value_write_norm_with_rn", "native register write, RN on"),
                        ("rn_value_write_norm", "RN write"),
                    ],
                )


def plot_cross_model_postln(compact: dict[tuple[str, str, str], dict[str, Any]], out_dir: Path) -> None:
    rows = []
    for key, trained in compact.items():
        model, variant, cond = key
        if model != "trained":
            continue
        other = compact.get(("pretrained", variant, cond))
        if other is None:
            continue
        ucos = float(F.cosine_similarity(torch.from_numpy(trained["u_rn"]).float(), torch.from_numpy(other["u_rn"]).float(), dim=0))
        kr = torch.from_numpy(trained["k_rn_mean"]).float(); kp = torch.from_numpy(other["k_rn_mean"]).float()
        vr = torch.from_numpy(trained["v_rn_mean"]).float(); vp = torch.from_numpy(other["v_rn_mean"]).float()
        orr = torch.from_numpy(trained["out_rn_mean"]).float(); op = torch.from_numpy(other["out_rn_mean"]).float()
        for h in range(kr.shape[0]):
            rows.append({
                "variant": variant,
                "pump_condition": cond,
                "head": h,
                "postln_rn_cos_trained_vs_pretrained": ucos,
                "k_rn_cos_trained_vs_pretrained": float(cosine(kr[h], kp[h], dim=0)),
                "v_rn_cos_trained_vs_pretrained": float(cosine(vr[h], vp[h], dim=0)),
                "wo_v_rn_cos_trained_vs_pretrained": float(cosine(orr[h], op[h], dim=0)),
            })
    save_rows(out_dir.parent / "cross_model_rn_geometry.csv", rows)
    for variant in sorted({r["variant"] for r in rows}):
        for cond in PUMP_CONDITIONS:
            sub = [r for r in rows if r["variant"] == variant and r["pump_condition"] == cond]
            if not sub:
                continue
            plot_head_lines(
                sub, out_dir / f"06_cross_model_rn_geometry__{stable_slug(variant)}__{stable_slug(cond)}.png",
                title=f"Same RN through trained vs pretrained B13 — {variant}/{cond}",
                metrics=[
                    ("k_rn_cos_trained_vs_pretrained", "K cosine"),
                    ("v_rn_cos_trained_vs_pretrained", "V cosine"),
                    ("wo_v_rn_cos_trained_vs_pretrained", "W_O V cosine"),
                ],
            )


def fit_b13_basis(
    bundle: ModelBundle,
    bank: PatchAlignedSyntheticBank,
    *,
    variant: str,
    n_scenes: int,
    batch_size: int,
    max_rows: int,
    rank: int,
) -> np.ndarray:
    rows: list[torch.Tensor] = []
    retained = 0
    for ids in iter_batches(n_scenes, batch_size):
        pil = []
        for sid in ids:
            _spec, imgs = bank.scene(sid)
            pil.append(imgs[variant])
        batch = preprocess_pil_batch(pil, bundle.preprocess, bundle.device)
        base = bundle.runner.run(batch, condition="base", capture_blocks=(13,))
        tag = bundle.runner.run(batch, condition="rn_tag", capture_blocks=(13,))
        delta = (tag.states[13][:, 1:] - base.states[13][:, 1:]).reshape(-1, base.states[13].shape[-1]).float()
        remain = max_rows - retained if max_rows > 0 else delta.shape[0]
        if remain > 0:
            take = delta[:remain].cpu()
            rows.append(take)
            retained += take.shape[0]
        del batch, base, tag, delta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    matrix = torch.cat(rows, dim=0)
    svd = randomized_svd_rows(matrix, k=rank, center=False)
    basis = np.asarray(svd.right_vectors[:rank], dtype=np.float32)
    # Orient signs so average B13 projection is positive.
    proj_mean = matrix @ torch.from_numpy(basis).T
    signs = torch.sign(proj_mean.mean(dim=0)); signs[signs == 0] = 1
    basis = basis * signs.numpy()[:, None]
    return basis


def track_b13_basis_spatially(
    bundle: ModelBundle,
    bank: PatchAlignedSyntheticBank,
    *,
    variant: str,
    basis_kc: np.ndarray,
    blocks: list[int],
    n_scenes: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    basis = torch.from_numpy(basis_kc).float()
    grid = bank.grid_size
    p = grid * grid
    k = basis.shape[0]
    signed = np.zeros((len(blocks), k, p), dtype=np.float64)
    absolute = np.zeros_like(signed)
    delta_norm = np.zeros((len(blocks), p), dtype=np.float64)
    n_total = 0

    for ids in iter_batches(n_scenes, batch_size):
        pil = []
        for sid in ids:
            _spec, imgs = bank.scene(sid)
            pil.append(imgs[variant])
        batch = preprocess_pil_batch(pil, bundle.preprocess, bundle.device)
        base = bundle.runner.run(batch, condition="base", capture_blocks=blocks)
        tag = bundle.runner.run(batch, condition="rn_tag", capture_blocks=blocks)
        bsz = len(ids)
        for bi, block in enumerate(blocks):
            delta = (tag.states[block][:, 1:] - base.states[block][:, 1:]).float().cpu()
            scores = torch.einsum("bpc,kc->bpk", delta, basis)
            signed[bi] += scores.sum(dim=0).T.numpy()
            absolute[bi] += scores.abs().sum(dim=0).T.numpy()
            delta_norm[bi] += delta.norm(dim=-1).sum(dim=0).numpy()
        n_total += bsz
        del batch, base, tag
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "signed": signed / max(1, n_total),
        "absolute": absolute / max(1, n_total),
        "delta_norm": delta_norm / max(1, n_total),
        "blocks": np.asarray(blocks, dtype=np.int64),
    }


def plot_lowrank_spatial(
    maps: dict[str, np.ndarray],
    out_dir: Path,
    *,
    model_name: str,
    variant: str,
    grid: int,
    bbox_patch: tuple[int, int, int, int],
) -> None:
    blocks = [int(x) for x in maps["blocks"]]
    signed = maps["signed"]
    absolute = maps["absolute"]
    delta_norm = maps["delta_norm"]
    k = signed.shape[1]

    for d in range(k):
        ncol = min(4, len(blocks))
        nrow = int(math.ceil(len(blocks) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.3 * nrow), squeeze=False)
        vmax = float(np.max(np.abs(signed[:, d])))
        for i, block in enumerate(blocks):
            ax = axes[i // ncol][i % ncol]
            im = ax.imshow(signed[i, d].reshape(grid, grid), interpolation="nearest", vmin=-vmax, vmax=vmax)
            draw_region_rect(ax, bbox_patch)
            ax.set_title(f"B{block}")
            ax.set_xticks([]); ax.set_yticks([])
        for j in range(len(blocks), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"B13 RN low-rank direction {d+1}: signed spatial projection — {model_name}/{variant}")
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015)
        fig.subplots_adjust(top=0.90, wspace=0.08, hspace=0.18)
        fig.savefig(out_dir / f"07_lowrank_dir{d+1}_signed__{stable_slug(model_name)}__{stable_slug(variant)}.png", dpi=190)
        plt.close(fig)

    # Matrix requested explicitly: WHERE is the B13 PC1 over time? Rows=blocks, cols=spatial patch index.
    fig, ax = plt.subplots(figsize=(15, 4.8))
    vmax = float(np.max(np.abs(signed[:, 0])))
    im = ax.imshow(signed[:, 0, :], aspect="auto", interpolation="nearest", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(blocks))); ax.set_yticklabels([f"B{x}" for x in blocks])
    ax.set_xlabel(f"patch index (row-major {grid}x{grid})")
    ax.set_title(f"WHERE the B13 RN PC1 lives downstream — {model_name}/{variant}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"08_lowrank_where_matrix__{stable_slug(model_name)}__{stable_slug(variant)}.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 4.8))
    im = ax.imshow(delta_norm, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(blocks))); ax.set_yticklabels([f"B{x}" for x in blocks])
    ax.set_xlabel(f"patch index (row-major {grid}x{grid})")
    ax.set_title(f"RN delta norm by downstream patch — {model_name}/{variant}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"09_delta_norm_where_matrix__{stable_slug(model_name)}__{stable_slug(variant)}.png", dpi=190)
    plt.close(fig)


def projected_concat_from_cache(cache_bhtd: torch.Tensor, reg_mask_bp: torch.Tensor) -> torch.Tensor:
    """Mean native register projected K/V across selected sources -> [B,E]."""
    mean = gather_selected_register_mean(cache_bhtd, reg_mask_bp)  # [B,H,D]
    return mean.reshape(mean.shape[0], -1)


@contextlib.contextmanager
def replace_last_projection(linear: torch.nn.Module, replacement_be: Optional[torch.Tensor]):
    if replacement_be is None:
        yield
        return

    def hook(_module, _inputs, output):
        y = output.clone()
        repl = replacement_be.to(device=y.device, dtype=y.dtype)
        if repl.shape != (y.shape[1], y.shape[2]):
            raise RuntimeError(f"replacement {tuple(repl.shape)} != expected {(y.shape[1], y.shape[2])}")
        y[-1] = repl
        return y

    handle = linear.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def hybrid_kv_embedding_rows(
    bundle: ModelBundle,
    bank: PatchAlignedSyntheticBank,
    *,
    variant: str,
    n_scenes: int,
    batch_size: int,
    register_threshold: float,
    max_registers: int,
    min_registers: int,
) -> list[dict[str, Any]]:
    rows = []
    blk = bundle.model.visual.transformer.resblocks[13]
    for ids in iter_batches(n_scenes, batch_size):
        pil = []
        for sid in ids:
            _spec, imgs = bank.scene(sid)
            pil.append(imgs[variant])
        images = preprocess_pil_batch(pil, bundle.preprocess, bundle.device)
        pre = run_pre_b13_with_pump_probe(bundle, images, pump_condition="intact")
        x_pre = pre["pre_b13"]
        reg_mask, _primary, _norms = register_mask_from_pre_b13(
            x_pre, threshold=register_threshold, max_registers=max_registers, min_registers=min_registers
        )

        with torch.no_grad(), model_autocast_context(bundle.model):
            base_post, _ = forward_block_probe(blk, x_pre, capture_attention=True)
            base_cache = clone_attn_cache(blk.attn)
            base_emb = F.normalize(bundle.runner.continue_from_post_b13(base_post.permute(1, 0, 2)), dim=-1)

            rn_token = bundle.model.visual.read_null_token.to(device=x_pre.device, dtype=x_pre.dtype)
            x_rn = torch.cat([x_pre, rn_token.view(1, 1, -1).expand(1, x_pre.shape[1], -1)], dim=0)

            k_reg_be = projected_concat_from_cache(base_cache["last_k"].float(), reg_mask)
            v_reg_be = projected_concat_from_cache(base_cache["last_v"].float(), reg_mask)

            embeddings = {}
            for name, replace_k, replace_v in (
                ("RN", None, None),
                ("RN_K__REG_V", None, v_reg_be),
                ("REG_K__RN_V", k_reg_be, None),
                ("REG_KV", k_reg_be, v_reg_be),
            ):
                with replace_last_projection(blk.attn.k_proj, replace_k), replace_last_projection(blk.attn.v_proj, replace_v):
                    post_full, _ = forward_block_probe(blk, x_rn, capture_attention=False)
                post = post_full[:-1].permute(1, 0, 2)
                embeddings[name] = F.normalize(bundle.runner.continue_from_post_b13(post), dim=-1)

        ref = embeddings["RN"] - base_emb
        ref_norm2 = ref.square().sum(dim=-1).clamp_min(1e-12)
        for name, emb in embeddings.items():
            d = emb - base_emb
            recovery = (d * ref).sum(dim=-1) / ref_norm2
            direction_cos = cosine(d, ref, dim=-1)
            norm_ratio = d.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)
            rows.append({
                "model": bundle.name,
                "variant": variant,
                "condition": name,
                "recovery_projection": float(recovery.mean().cpu()),
                "direction_cos_to_RN": float(direction_cos.mean().cpu()),
                "shift_norm_over_RN": float(norm_ratio.mean().cpu()),
            })

        del images, pre, x_pre, base_post, base_cache, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def plot_hybrid_kv(rows: list[dict[str, Any]], out_dir: Path) -> None:
    rows = group_mean(rows, ("model", "variant", "condition"))
    for model in sorted({r["model"] for r in rows}):
        for variant in sorted({r["variant"] for r in rows if r["model"] == model}):
            sub = [r for r in rows if r["model"] == model and r["variant"] == variant]
            labels = [r["condition"] for r in sub]
            vals = [float(r["recovery_projection"]) for r in sub]
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.bar(np.arange(len(vals)), vals)
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.axhline(0.0, linewidth=1)
            ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("RN displacement recovered (projection)")
            ax.set_title(f"B13 K/V source swap: address vs payload — {model}/{variant}")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"10_kv_hybrid__{stable_slug(model)}__{stable_slug(variant)}.png", dpi=190)
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="RN B13 stash impersonation + B11/B12 dynamite follow-up")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--module-root", default=".")
    ap.add_argument("--output-dir", default=r"rn_stash_followup")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pretrained-rn-checkpoint", default=None,
                    help="Explicit vanilla OpenAI CLIP + exact trained RN variant; bridge must be absent.")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--positions", default="center", help="center is recommended so spatial maps do not smear")
    ap.add_argument("--stash-variants", default="text,sine,blank")
    ap.add_argument("--stash-scenes", type=int, default=64)
    ap.add_argument("--lowrank-variants", default="text,sine,blank")
    ap.add_argument("--lowrank-fit-scenes", type=int, default=96)
    ap.add_argument("--lowrank-scenes", type=int, default=192)
    ap.add_argument("--lowrank-rank", type=int, default=4)
    ap.add_argument("--lowrank-max-rows", type=int, default=8192)
    ap.add_argument("--track-blocks", default="13,14,15,16,17,18,19,20,21,22,23")
    ap.add_argument("--swap-scenes", type=int, default=48)
    ap.add_argument("--swap-variants", default="text,sine")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--register-threshold", type=float, default=70.0)
    ap.add_argument("--max-registers", type=int, default=4)
    ap.add_argument("--min-registers", type=int, default=1)
    ap.add_argument("--font", action="append", default=[])
    args = ap.parse_args()

    out = Path(args.output_dir)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    trained, clip_mod = load_trained(args.checkpoint, args.module_root, args.device)
    bundles = [trained]
    if not args.no_pretrained:
        if not args.pretrained_rn_checkpoint:
            raise RuntimeError(
                "--pretrained-rn-checkpoint is required for the vanilla+RN control. "
                "Randomly initialized RN/bridge state is scientifically invalid."
            )
        bundles.append(load_pretrained_with_same_rn(
            clip_mod,
            pretrained_rn_checkpoint=args.pretrained_rn_checkpoint,
            trained=trained,
            device=args.device,
        ))

    # Geometry guardrail: this follow-up intentionally compares spatial maps directly.
    trained_res = int(trained.model.visual.input_resolution)
    trained_patch = int(trained.model.visual.conv1.kernel_size[0])
    for b in bundles:
        if int(b.model.visual.input_resolution) != trained_res or int(b.model.visual.conv1.kernel_size[0]) != trained_patch:
            raise RuntimeError(
                "Direct spatial comparison requires matching resolution/patch size in this script. "
                f"trained={trained_res}/{trained_patch}, {b.name}={b.model.visual.input_resolution}/{b.model.visual.conv1.kernel_size[0]}"
            )

    grid = trained_res // trained_patch
    bank = PatchAlignedSyntheticBank(
        resolution=trained_res,
        patch_size=trained_patch,
        seed=args.seed,
        positions=parse_strs(args.positions),
        font_paths=args.font,
    )
    preview_spec, _ = bank.scene(0)
    bank.preview(out / "synthetic_preview.png", n_scenes=5, variants=parse_strs(args.stash_variants))

    settings = vars(args).copy()
    settings.update({
        "resolution": trained_res,
        "patch_size": trained_patch,
        "grid": grid,
        "rn_insert_block": trained.runner.insert_block,
        "trained_rn_raw_norm": float(trained.model.visual.read_null_token.detach().float().norm().cpu()),
        "reg_neurons_block_11": REG_NEURONS_BLOCK_11,
        "reg_neurons_block_12": REG_NEURONS_BLOCK_12,
        "models": [b.name for b in bundles],
    })
    save_json(out / "settings.json", settings)

    # ---------------------------------------------------------------------
    # PASS 1: B11/B12 pump + B13 stash/RN source substitution.
    # ---------------------------------------------------------------------
    trajectory_rows: list[dict[str, Any]] = []
    pump_rows: list[dict[str, Any]] = []
    stash_rows: list[dict[str, Any]] = []
    map_acc: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    compact_acc: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for bundle in bundles:
        print(f"\n[model] {bundle.name}")
        for variant in parse_strs(args.stash_variants):
            for pump_condition in PUMP_CONDITIONS:
                print(f"  [stash] {variant} / {pump_condition}")
                key = (bundle.name, variant, pump_condition)
                map_acc[key] = {}
                for ids in iter_batches(args.stash_scenes, args.batch_size):
                    specs, pil = [], []
                    for sid in ids:
                        spec, imgs = bank.scene(sid)
                        specs.append(spec); pil.append(imgs[variant])
                    images = preprocess_pil_batch(pil, bundle.preprocess, bundle.device)
                    region = region_patch_mask(specs, grid, bundle.device)
                    pre = run_pre_b13_with_pump_probe(bundle, images, pump_condition=pump_condition)
                    x_pre = pre["pre_b13"]
                    reg_mask, primary, patch_norms = register_mask_from_pre_b13(
                        x_pre,
                        threshold=args.register_threshold,
                        max_registers=args.max_registers,
                        min_registers=args.min_registers,
                    )
                    pair = run_b13_pair_from_pre(bundle, x_pre)

                    trajectory_rows.extend(primary_register_trajectory_rows(
                        bundle_name=bundle.name,
                        variant=variant,
                        pump_condition=pump_condition,
                        pre_result=pre,
                        b13_pair=pair,
                        primary_patch_b=primary,
                        reg_mask_bp=reg_mask,
                        patch_norms_bp=patch_norms,
                    ))
                    pump_rows.extend(pump_metric_rows(
                        bundle_name=bundle.name,
                        variant=variant,
                        pump_condition=pump_condition,
                        pre_result=pre,
                        primary_patch_b=primary,
                    ))
                    rows, maps, compact = b13_stash_rows_and_maps(
                        bundle=bundle,
                        variant=variant,
                        pump_condition=pump_condition,
                        x_pre=x_pre,
                        b13_pair=pair,
                        reg_mask_bp=reg_mask,
                        primary_patch_b=primary,
                        region_bp=region,
                    )
                    stash_rows.extend(rows)
                    merge_map_accumulator(map_acc[key], maps)
                    compact_acc[key].append(compact)

                    del images, pre, x_pre, pair
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    trajectory_mean = group_mean(trajectory_rows, ("model", "variant", "pump_condition", "stage"))
    pump_mean = group_mean(pump_rows, ("model", "variant", "pump_condition", "block", "token_group"))
    stash_mean = group_mean(stash_rows, ("model", "variant", "pump_condition", "head", "query_group"))
    save_rows(out / "dynamite_register_trajectory.csv", trajectory_mean)
    save_rows(out / "pump_neuron_contribution.csv", pump_mean)
    save_rows(out / "b13_stash_impersonation.csv", stash_mean)

    compact_mean: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, entries in compact_acc.items():
        compact_mean[key] = {
            name: np.mean(np.stack([e[name] for e in entries], axis=0), axis=0)
            for name in entries[0]
        }

    # Spatial attention/query maps.
    for (model_name, variant, cond), acc in map_acc.items():
        count = max(float(acc.get("count", 0.0)), 1.0)
        stem = f"{stable_slug(model_name)}__{stable_slug(variant)}__{stable_slug(cond)}"
        rn_map = acc["rn"] / count
        reg_base_map = acc["reg_base"] / count
        reg_rn_map = acc["reg_with_rn"] / count
        rn_logit = acc["rn_logit"] / count
        reg_logit = acc["reg_logit"] / count
        np.savez_compressed(
            out / f"b13_query_maps__{stem}.npz",
            rn_attention=rn_map,
            register_attention_base=reg_base_map,
            register_attention_with_rn=reg_rn_map,
            rn_logits=rn_logit,
            register_logits=reg_logit,
        )
        plot_head_x_query_matrix(rn_map, plots / f"11_who_looks_at_RN_matrix__{stem}.png", title=f"WHO looks at RN — {model_name}/{variant}/{cond}", grid=grid)
        plot_head_x_query_matrix(reg_base_map, plots / f"12_who_looks_at_register_matrix__{stem}.png", title=f"WHO looks at native B12 stash (RN off) — {model_name}/{variant}/{cond}", grid=grid)
        plot_all_head_spatial_maps(rn_map, plots / f"13_where_queries_look_RN__{stem}.png", title=f"Spatial query map: attention to RN — {model_name}/{variant}/{cond}", grid=grid, bbox_patch=preview_spec.bbox_patch)
        plot_all_head_spatial_maps(reg_base_map, plots / f"14_where_queries_look_register__{stem}.png", title=f"Spatial query map: attention to native stash — {model_name}/{variant}/{cond}", grid=grid, bbox_patch=preview_spec.bbox_patch)
        # Pre-softmax addressability maps are useful when softmax competition changes.
        plot_all_head_spatial_maps(rn_logit, plots / f"15_where_RN_key_matches_queries__{stem}.png", title=f"Pre-softmax q·k_RN map — {model_name}/{variant}/{cond}", grid=grid, bbox_patch=preview_spec.bbox_patch)
        plot_all_head_spatial_maps(reg_logit, plots / f"16_where_register_key_matches_queries__{stem}.png", title=f"Pre-softmax q·k_register map — {model_name}/{variant}/{cond}", grid=grid, bbox_patch=preview_spec.bbox_patch)

    plot_dynamite_norms(trajectory_mean, plots)
    plot_pump_contrib(pump_mean, plots)
    plot_stash_metrics(stash_mean, plots)
    plot_cross_model_postln(compact_mean, plots)

    # ---------------------------------------------------------------------
    # PASS 2: track B13 low-rank RN directions spatially downstream.
    # Intact pump only, because the four-way pump grid is already covered above.
    # ---------------------------------------------------------------------
    lowrank_bases: dict[tuple[str, str], np.ndarray] = {}
    lowrank_maps: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    track_blocks = parse_ints(args.track_blocks)
    for bundle in bundles:
        for variant in parse_strs(args.lowrank_variants):
            print(f"[lowrank] fit {bundle.name}/{variant}")
            basis = fit_b13_basis(
                bundle, bank,
                variant=variant,
                n_scenes=args.lowrank_fit_scenes,
                batch_size=args.batch_size,
                max_rows=args.lowrank_max_rows,
                rank=args.lowrank_rank,
            )
            lowrank_bases[(bundle.name, variant)] = basis
            np.save(out / f"b13_rn_patch_basis__{stable_slug(bundle.name)}__{stable_slug(variant)}.npy", basis)
            print(f"[lowrank] track {bundle.name}/{variant}")
            maps = track_b13_basis_spatially(
                bundle, bank,
                variant=variant,
                basis_kc=basis,
                blocks=track_blocks,
                n_scenes=args.lowrank_scenes,
                batch_size=args.batch_size,
            )
            lowrank_maps[(bundle.name, variant)] = maps
            np.savez_compressed(
                out / f"lowrank_spatial__{stable_slug(bundle.name)}__{stable_slug(variant)}.npz",
                **maps,
            )
            plot_lowrank_spatial(
                maps, plots,
                model_name=bundle.name,
                variant=variant,
                grid=grid,
                bbox_patch=preview_spec.bbox_patch,
            )

    # Cross-model B13 low-rank subspace preservation.
    alignment_rows = []
    if {b.name for b in bundles} >= {"trained", "pretrained"}:
        for variant in parse_strs(args.lowrank_variants):
            bt = lowrank_bases.get(("trained", variant))
            bp = lowrank_bases.get(("pretrained", variant))
            if bt is None or bp is None:
                continue
            pa = principal_angles(bt, bp, k=min(args.lowrank_rank, bt.shape[0], bp.shape[0]))
            for i, (c, a) in enumerate(zip(pa["cosines"], pa["angles_deg"]), start=1):
                alignment_rows.append({"variant": variant, "component": i, "principal_cosine": float(c), "angle_deg": float(a)})
        save_rows(out / "cross_model_lowrank_alignment.csv", alignment_rows)
        for variant in sorted({r["variant"] for r in alignment_rows}):
            sub = [r for r in alignment_rows if r["variant"] == variant]
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.bar([int(r["component"]) for r in sub], [float(r["principal_cosine"]) for r in sub])
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("principal component")
            ax.set_ylabel("trained ↔ pretrained subspace cosine")
            ax.set_title(f"B13 RN low-rank subspace conservation — {variant}")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots / f"17_cross_model_lowrank_alignment__{stable_slug(variant)}.png", dpi=190)
            plt.close(fig)

    # ---------------------------------------------------------------------
    # PASS 3: K/V hybrid source experiment — same road vs different cargo.
    # ---------------------------------------------------------------------
    hybrid_rows: list[dict[str, Any]] = []
    if args.swap_scenes > 0:
        for bundle in bundles:
            for variant in parse_strs(args.swap_variants):
                print(f"[K/V hybrid] {bundle.name}/{variant}")
                hybrid_rows.extend(hybrid_kv_embedding_rows(
                    bundle, bank,
                    variant=variant,
                    n_scenes=args.swap_scenes,
                    batch_size=args.batch_size,
                    register_threshold=args.register_threshold,
                    max_registers=args.max_registers,
                    min_registers=args.min_registers,
                ))
        hybrid_mean = group_mean(hybrid_rows, ("model", "variant", "condition"))
        save_rows(out / "b13_kv_hybrid_causal.csv", hybrid_mean)
        plot_hybrid_kv(hybrid_rows, plots)

    (plots / "PLOTS_GENERATED.txt").write_text(
        "RN stash follow-up plots generated successfully.\n"
        "Key families: dynamite norm trajectory, pump-neuron contribution, stash K/V/addressability,\n"
        "source substitution, WHO/WHERE spatial matrices, downstream low-rank maps, trained-vs-pretrained\n"
        "geometry, and K/V hybrid causal swaps.\n",
        encoding="utf-8",
    )
    print(f"\n[done] numerical outputs: {out}")
    print(f"[done] plots: {plots}")




if __name__ == "__main__":
    main()
