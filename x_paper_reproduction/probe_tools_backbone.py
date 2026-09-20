#!/usr/bin/env python3
r"""Shared CLIP backbone loading, explicit QKV normalization, register-pump ablation, and sink measurements.
Imported by multiple probes; contains no benchmark entry point.
"""
from __future__ import annotations
from probe_tools_analysis import (clear_attn_cache)

# SIGNATURE
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =============================================================================
# HARDCODED LOCAL DEFAULTS -- exactly the paths supplied by the user
# =============================================================================

DEFAULT_CLIP_MODULE = "attnclip_mechinterp_sae"
DEFAULT_MODEL_SPEC = "ViT-L/14"


# =============================================================================
# Canonical names / metadata / hard distractors
# =============================================================================

# Normalize filenames to these display names before any evaluation.


# Extra known-name distractors, deliberately close in profession / era / region where possible.


# =============================================================================
# Records / helpers
# =============================================================================


# =============================================================================
# Image compositing -- preserve transparency correctly
# =============================================================================


# =============================================================================
# CLIP loader and text banks
# =============================================================================

class ClipBundle:
    def __init__(self,model,preprocess,tokenize,device,source):
        self.model=model;self.preprocess=preprocess;self.tokenize=tokenize;self.device=device;self.source=source


def load_clip(args) -> ClipBundle:
    """Load ordinary CLIP through the capture-enabled vanilla mechinterp module.

    Workspace probes require private instrumentation such as ``_prepare_tokens``
    and captured Q/K/V attention state.  Stock ``openai/clip`` does not expose
    that API, so falling back to it would only defer the failure to the first
    mechanistic probe.  Fail at the loader boundary instead.
    """
    device=torch.device("cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    try:
        clip_mod=importlib.import_module(args.clip_module)
        from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything
        model,preprocess,_=load_openai_clip_anything(clip_mod,args.model_spec,device=device,jit=False,strict=True)
    except Exception as e:
        raise RuntimeError(
            "Could not load the capture-enabled CLIP backbone through "
            f"{args.clip_module!r}. These paper probes require the vanilla "
            "attnclip_mechinterp_sae instrumentation and cannot safely fall "
            "back to stock openai/clip."
        ) from e

    visual=getattr(model,"visual",None)
    if visual is None or not callable(getattr(visual,"_prepare_tokens",None)):
        raise RuntimeError(
            f"Loaded {args.clip_module!r}, but its visual tower does not expose "
            "_prepare_tokens(). Use the bundled attnclip_mechinterp_sae module."
        )
    blocks=getattr(getattr(visual,"transformer",None),"resblocks",None)
    if not blocks:
        raise RuntimeError("Loaded visual tower has no transformer residual blocks")
    first_attn=getattr(blocks[0],"attn",None)
    if first_attn is None or not all(hasattr(first_attn,name) for name in ("q_proj","k_proj","v_proj")):
        raise RuntimeError(
            f"Loaded {args.clip_module!r}, but attention is not the capture-enabled "
            "explicit-QKV implementation required by the workspace probes."
        )

    model=model.eval().float()
    for p in model.parameters(): p.requires_grad_(False)
    tok=getattr(clip_mod,"tokenize",None)
    if tok is None:
        raise RuntimeError(f"{args.clip_module!r} does not expose tokenize()")
    print(f"[model] loaded pretrained {args.model_spec} through {args.clip_module} + load_openai_clip_anything")
    return ClipBundle(model,preprocess,tok,device,f"{args.clip_module}:{args.model_spec}")


# =============================================================================
# ViT hooks / diagnostics
# =============================================================================


# =============================================================================
# Task rendering / evaluation
# =============================================================================


# =============================================================================
# Outputs / plots
# =============================================================================


# =============================================================================
# Main
# =============================================================================


# BACKBONE
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Mapping, Tuple


import matplotlib
matplotlib.use("Agg")


# The only non-ordinary visual state known in the x-attn ViT.
VISUAL_CUSTOM_EXACT = {
    "visual.read_null_token",
    "visual.read_null_insert_block_config",
}
VISUAL_CUSTOM_SUBSTRINGS = (
    "read_null",
    "read_implant",
)
ROOT_CUSTOM_PREFIXES = (
    "read_implant.",
)
ROOT_CUSTOM_EXACT = {
    "hard_text_embedding",
    "null_text_embedding",
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def safe_torch_load(path: Path) -> Any:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    try:
        kwargs["weights_only"] = False
        return torch.load(str(path), **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(str(path), **kwargs)


def strip_common_prefixes(state: Mapping[str, Any]) -> Dict[str, Any]:
    out = {str(k): v for k, v in state.items()}
    # Strip wrapper prefixes only when they are global/common.
    for prefix in ("module.", "model.", "clip."):
        keys = list(out)
        if keys and sum(k.startswith(prefix) for k in keys) >= max(1, int(.9 * len(keys))):
            out = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in out.items()}
    return out


def _looks_like_clip_state(obj: Mapping[str, Any]) -> bool:
    keys = set(map(str, obj.keys()))
    return "visual.conv1.weight" in keys and (
        "token_embedding.weight" in keys or any(k.startswith("transformer.resblocks.") for k in keys)
    )


def extract_state_dict(obj: Any, source: str = "checkpoint") -> Dict[str, Any]:
    if isinstance(obj, torch.nn.Module):
        return strip_common_prefixes(obj.state_dict())
    if isinstance(obj, Mapping):
        if _looks_like_clip_state(obj):
            return strip_common_prefixes(obj)
        for key in ("state_dict", "model_state_dict", "model", "clip", "module"):
            if key not in obj:
                continue
            value = obj[key]
            if isinstance(value, torch.nn.Module):
                return strip_common_prefixes(value.state_dict())
            if isinstance(value, Mapping) and _looks_like_clip_state(value):
                return strip_common_prefixes(value)
    raise TypeError(f"Could not extract a CLIP state_dict from {source}: {type(obj)}")


def freeze_eval(model: torch.nn.Module) -> torch.nn.Module:
    model.eval().float()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def block_group_from_key(key: str) -> str:
    m = re.search(r"visual\.transformer\.resblocks\.(\d+)\.", key)
    if m:
        return f"B{int(m.group(1)):02d}"
    if key.startswith("visual.conv1"):
        return "conv1"
    if key.startswith("visual.ln_pre"):
        return "ln_pre"
    if key.startswith("visual.ln_post"):
        return "ln_post"
    if key.startswith("visual.positional_embedding"):
        return "positional_embedding"
    if key.startswith("visual.class_embedding"):
        return "class_embedding"
    if key.startswith("visual.proj"):
        return "visual_proj"
    return "other_visual"


# -----------------------------------------------------------------------------
# Model loading / stripping
# -----------------------------------------------------------------------------

def load_pretrained_bundle(args) -> S.ClipBundle:
    return S.load_clip(args)


def transplant_stripped_xattn_visual(bundle: S.ClipBundle, args, audit_dir: Path) -> Tuple[S.ClipBundle, pd.DataFrame]:
    ckpt_path = Path(args.xattn_checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"x-attn checkpoint not found: {ckpt_path}")

    # Full-model pickle may require the original class path to exist before torch.load.
    if args.xattn_module:
        importlib.import_module(args.xattn_module)

    print(f"[xattn] loading checkpoint: {ckpt_path}")
    obj = safe_torch_load(ckpt_path)
    state = extract_state_dict(obj, str(ckpt_path))
    del obj

    if any(k.endswith(".theta") or k.endswith(".r") for k in state):
        raise RuntimeError(
            "Checkpoint still contains GmP theta/r parameters. Use the materialized "
            "_ungmp_oaiclip_fullmodel.pt checkpoint for this comparison."
        )

    # Record every custom key we deliberately refuse to transplant.
    ignored_custom = sorted(
        k for k in state
        if k in ROOT_CUSTOM_EXACT
        or any(k.startswith(p) for p in ROOT_CUSTOM_PREFIXES)
        or k in VISUAL_CUSTOM_EXACT
        or any(s in k.lower() for s in VISUAL_CUSTOM_SUBSTRINGS)
    )

    source_visual = {
        k: v for k, v in state.items()
        if k.startswith("visual.")
        and k not in VISUAL_CUSTOM_EXACT
        and not any(s in k.lower() for s in VISUAL_CUSTOM_SUBSTRINGS)
        and torch.is_tensor(v)
    }

    # oaiclip uses torch.nn.MultiheadAttention in_proj tensors; the attached
    # attnclip_mechinterp_sae uses explicit q/k/v linears.  Convert source keys
    # with the normal module's own converter before matching.
    clip_model_mod = importlib.import_module(f"{args.clip_module}.model")
    convert = getattr(clip_model_mod, "convert_state_dict_inproj_to_qkv", None)
    if convert is not None:
        source_visual = convert(dict(source_visual))

    target_state = bundle.model.state_dict()
    target_visual_keys = [k for k in target_state if k.startswith("visual.")]
    load_state = {}
    missing = []
    shape_mismatch = []
    drift_rows = []

    for key in target_visual_keys:
        if key not in source_visual:
            missing.append(key)
            continue
        src = source_visual[key].detach().cpu()
        tgt = target_state[key].detach().cpu()
        if tuple(src.shape) != tuple(tgt.shape):
            shape_mismatch.append({"key": key, "source_shape": list(src.shape), "target_shape": list(tgt.shape)})
            continue
        load_state[key] = src.to(dtype=target_state[key].dtype)
        if src.is_floating_point() and src.numel() > 0:
            sf = src.float().reshape(-1)
            tf = tgt.float().reshape(-1)
            sn = float(sf.norm())
            tn = float(tf.norm())
            cos = float(F.cosine_similarity(sf, tf, dim=0, eps=1e-12)) if sn > 0 and tn > 0 else float("nan")
            rel = float((sf - tf).norm() / max(tn, 1e-12))
            drift_rows.append({
                "key": key,
                "group": block_group_from_key(key),
                "numel": int(src.numel()),
                "cosine_to_pretrained": cos,
                "relative_l2_to_pretrained": rel,
                "max_abs_delta": float((sf - tf).abs().max()),
            })

    source_extra = sorted(k for k in source_visual if k not in target_visual_keys)
    audit = {
        "checkpoint": str(ckpt_path),
        "checkpoint_state_keys": len(state),
        "ignored_custom_key_count": len(ignored_custom),
        "ignored_custom_keys": ignored_custom,
        "ordinary_source_visual_keys": len(source_visual),
        "target_visual_keys": len(target_visual_keys),
        "loaded_visual_keys": len(load_state),
        "missing_target_visual_keys": missing,
        "shape_mismatch": shape_mismatch,
        "ignored_extra_visual_keys": source_extra,
        "text_tower_loaded_from_xattn": False,
        "read_null_loaded": False,
        "bridge_loaded": False,
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "checkpoint_strip_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if missing or shape_mismatch:
        raise RuntimeError(
            "Cannot make a strict vanilla visual transplant. "
            f"missing={len(missing)} shape_mismatch={len(shape_mismatch)}. "
            f"See {audit_dir/'checkpoint_strip_audit.json'}"
        )

    # Only visual keys are ever loaded.  The pretrained text tower remains byte-for-byte
    # the model we just loaded through attnclip_mechinterp_sae.
    current = bundle.model.state_dict()
    current.update(load_state)
    bundle.model.load_state_dict(current, strict=True)
    bundle.model = freeze_eval(bundle.model)
    bundle.source = f"stripped_visual:{ckpt_path} + pretrained_text:{args.clip_module}:{args.model_spec}"

    drift = pd.DataFrame(drift_rows)
    if len(drift):
        drift.to_csv(audit_dir / "visual_parameter_drift_per_tensor.csv", index=False)
        sm_rows = []
        for group, q in drift.groupby("group"):
            sm_rows.append({
                "group": group,
                "tensors": len(q),
                "numel": int(q.numel.sum()),
                "cosine_mean": float(np.average(q.cosine_to_pretrained.fillna(1.0), weights=q.numel)),
                "relative_l2_rms": float(np.sqrt(np.average(np.square(q.relative_l2_to_pretrained), weights=q.numel))),
                "max_abs_delta": float(q.max_abs_delta.max()),
            })
        sm = pd.DataFrame(sm_rows)
        sm.to_csv(audit_dir / "visual_parameter_drift_summary.csv", index=False)
    return bundle, drift


# -----------------------------------------------------------------------------
# Shared text bank
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Model evaluation
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# SAE compatibility audit
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Comparison summaries and plots
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


# SINK
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple


# =============================================================================
# Paper-adapted constants / our CLIP controls
# =============================================================================

MODEL_PRE = "pretrained"
MODEL_GMP = "gmp"
MODEL_REG = "regression"
MODEL_BRUT = "brut_regression"
MODEL_SAE = "sae_hinge"
MODEL_FT = "finetune_stripped"

MODE_NOPUMP = "no_pump"

MODEL_ORDER = (MODEL_PRE, MODEL_GMP, MODEL_REG, MODEL_BRUT, MODEL_SAE, MODEL_FT)

DEFAULT_CHECKPOINTS = {
    MODEL_GMP: r"ViT-L-14-BEST-smooth-GmP-ft-pickle-OpenAI.pt",
    MODEL_REG: r"ViT-L-14-Regression-FULL-model-pickle-OpenAI.pt",
    MODEL_BRUT: r"ViT-L-14-BRUT-Regression-FULL-model-pickle-OpenAI.pt",
    MODEL_SAE: r"CLIP-SAE-TypoAttack-Robust-ViT-L-14-FULL-model-pickle-OpenAI.pt",
}

DEFAULT_XATTN_CHECKPOINT = "REPLACE_WITH_CHECKPOINT.pt"

# Complete functional B11/B12 pump list after the neuron-paranoia detour.
REGISTER_PUMP_UNITS: Dict[int, Tuple[int, ...]] = {
    11: (9, 987, 1100, 1967, 2555, 3661, 3784),
    12: (42, 183, 983, 1571, 1816, 2687, 3002, 3008, 3868),
}

PAPER_NOP_VALUE_RATIO = 0.20
PAPER_RANK1_STABLE_RANK = 1.10
PAPER_HEAD_SINK_THRESHOLD = 0.50
DEFAULT_MASSIVE_TOKEN_NORM = 70.0
B13_PRE = 13
B20 = 20


def stable_rank(x: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.float()
    if x.numel() == 0:
        return float("nan")
    gram = x.T @ x
    evals = torch.linalg.eigvalsh(gram)
    top = float(evals[-1].clamp_min(eps).item())
    frob2 = float(torch.trace(gram).item())
    return frob2 / top


def projected_stable_rank(z_th: torch.Tensor, out_proj_weight: torch.Tensor, head: int,
                          eps: float = 1e-12) -> float:
    """Exact stable rank after this head's W_O slice without forming T x d_model."""
    z = z_th.float()
    dh = z.shape[-1]
    W = out_proj_weight[:, head * dh:(head + 1) * dh].float()
    Gz = z.T @ z
    Gw = W.T @ W
    frob2 = float(torch.trace(Gz @ Gw).item())
    ez, U = torch.linalg.eigh(Gz)
    ez = ez.clamp_min(0)
    sqrtGz = (U * ez.sqrt().unsqueeze(0)) @ U.T
    M = sqrtGz @ Gw @ sqrtGz
    top = float(torch.linalg.eigvalsh(M)[-1].clamp_min(eps).item())
    return frob2 / top


def projected_row_norms(v_th: torch.Tensor, out_proj_weight: torch.Tensor, head: int) -> torch.Tensor:
    v = v_th.float()
    dh = v.shape[-1]
    W = out_proj_weight[:, head * dh:(head + 1) * dh].float()
    G = W.T @ W
    return torch.sqrt(torch.einsum("td,df,tf->t", v, G, v).clamp_min(0))


def projected_row_norms_batch(v_btd: torch.Tensor, out_proj_weight: torch.Tensor, head: int) -> torch.Tensor:
    v = v_btd.float()
    dh = v.shape[-1]
    W = out_proj_weight[:, head * dh:(head + 1) * dh].float()
    G = W.T @ W
    return torch.sqrt(torch.einsum("btd,df,btf->bt", v, G, v).clamp_min(0))


def normalize_qkv_shape(t: torch.Tensor, batch: int, heads: int, tokens: int) -> torch.Tensor:
    if t is None:
        raise RuntimeError("Q/K/V capture missing. Use capture-enabled attnclip_mechinterp_sae.")
    if t.ndim != 4:
        raise RuntimeError(f"Expected 4D Q/K/V cache, got {tuple(t.shape)}")
    if t.shape[0] == batch and t.shape[1] == heads and t.shape[2] == tokens:
        return t
    if t.shape[0] == heads and t.shape[1] == batch and t.shape[2] == tokens:
        return t.permute(1, 0, 2, 3)
    if t.shape[0] == tokens and t.shape[1] == batch and t.shape[2] == heads:
        return t.permute(1, 2, 0, 3)
    raise RuntimeError(f"Cannot normalize Q/K/V cache shape {tuple(t.shape)} for B={batch},H={heads},T={tokens}")


def normalize_probs_shape(p: torch.Tensor, batch: int, heads: int, tokens: int) -> torch.Tensor:
    if p.ndim == 3 and batch == 1 and p.shape[0] == heads:
        return p.unsqueeze(0)
    if p.ndim != 4:
        raise RuntimeError(f"Expected [B,H,T,T] probs, got {tuple(p.shape)}")
    if p.shape[:3] == (batch, heads, tokens):
        return p
    if p.shape[0] == heads and p.shape[1] == batch:
        return p.permute(1, 0, 2, 3)
    raise RuntimeError(f"Cannot normalize probs shape {tuple(p.shape)}")


def patch_set_from_norms(x_tbd: torch.Tensor, threshold: float) -> List[set]:
    n = x_tbd[1:].float().norm(dim=-1).T
    return [set(torch.nonzero(n[i] >= threshold, as_tuple=False).flatten().cpu().tolist()) for i in range(n.shape[0])]


def event_regime(value_ratio: float, sr: float) -> str:
    if np.isfinite(value_ratio) and value_ratio < PAPER_NOP_VALUE_RATIO:
        return "NOP"
    if np.isfinite(value_ratio) and value_ratio >= PAPER_NOP_VALUE_RATIO and np.isfinite(sr) and sr <= PAPER_RANK1_STABLE_RANK:
        return "BROADCAST"
    return "OTHER"


def apply_pump_ablation(gelu: torch.Tensor, block: int, mode: str) -> torch.Tensor:
    if mode != MODE_NOPUMP or block not in REGISTER_PUMP_UNITS:
        return gelu
    idx = torch.as_tensor(REGISTER_PUMP_UNITS[block], device=gelu.device, dtype=torch.long)
    out = gelu.clone()
    out.index_fill_(-1, idx, 0)
    return out


# =============================================================================
# Model loading
# =============================================================================

def _materialize_gmp_weights(state: Mapping[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Convert theta/r pairs only when the checkpoint actually contains them.

    A model/checkpoint being *named* GmP never selects a GmP runtime.  This is a
    legacy checkpoint-format adapter used before loading the resulting ordinary
    weights into the architecture required by the probe.
    """
    out = dict(state)
    converted = 0
    theta_keys = sorted(k for k in state if str(k).endswith(".theta"))
    for theta_key in theta_keys:
        base = theta_key[:-len("theta")]
        radius_key = base + "r"
        weight_key = base + "weight"
        if radius_key not in state:
            raise KeyError(f"Found {theta_key} without matching {radius_key}")
        if weight_key in state:
            raise KeyError(
                f"Checkpoint contains both {weight_key} and ({theta_key}, {radius_key}); "
                "cannot determine a single authoritative tensor"
            )
        theta = state[theta_key]
        radius = state[radius_key]
        if not torch.is_tensor(theta) or not torch.is_tensor(radius):
            raise TypeError(f"Non-tensor GmP pair: {theta_key}, {radius_key}")
        if theta.ndim != 2 or radius.numel() != theta.shape[0]:
            raise ValueError(
                f"Invalid GmP pair: {theta_key}={tuple(theta.shape)}, "
                f"{radius_key}={tuple(radius.shape)}"
            )
        out[weight_key] = (
            radius.reshape(-1, 1).to(theta.dtype)
            * F.normalize(theta, p=2, dim=1)
        ).contiguous()
        del out[theta_key]
        del out[radius_key]
        converted += 1
    leftovers = [k for k in out if str(k).endswith((".theta", ".r"))]
    if leftovers:
        raise RuntimeError(f"Unconverted GmP tensors remain: {leftovers[:20]}")
    return out, converted


def _strict_visual_transplant(bundle, checkpoint: Path, args, audit_dir: Path, label: str):
    """Transplant ordinary visual tensors into a fresh capture-enabled CLIP.

    This keeps the capture-enabled architecture and pretrained text tower, while
    making the visual forward byte-for-byte use the supplied ordinary ViT weights.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint}")

    # Common pickle class paths. Ignore modules not installed; torch.load will give
    # the real error if the checkpoint actually needs one of them.
    for mod in (args.pickle_module, "clip", "oaiclip"):
        if not mod:
            continue
        try:
            importlib.import_module(mod)
        except Exception:
            pass

    print(f"[{label}] loading ordinary checkpoint: {checkpoint}")
    obj = D.safe_torch_load(checkpoint)
    state = D.extract_state_dict(obj, str(checkpoint))
    del obj

    gmp_materialized = 0
    if any(str(k).endswith((".theta", ".r")) for k in state):
        state, gmp_materialized = _materialize_gmp_weights(state)
        print(f"[{label}] materialized {gmp_materialized} GeometricLinear matrices into ordinary weights")

    source_visual = {k: v for k, v in state.items() if k.startswith("visual.") and torch.is_tensor(v)}
    clip_model_mod = importlib.import_module(f"{args.clip_module}.model")
    convert = getattr(clip_model_mod, "convert_state_dict_inproj_to_qkv", None)
    if convert is not None:
        source_visual = convert(dict(source_visual))

    target_state = bundle.model.state_dict()
    target_visual_keys = [k for k in target_state if k.startswith("visual.")]
    load_state = {}
    missing, mismatch = [], []
    drift_rows = []
    for key in target_visual_keys:
        if key not in source_visual:
            missing.append(key)
            continue
        src = source_visual[key].detach().cpu()
        tgt = target_state[key].detach().cpu()
        if tuple(src.shape) != tuple(tgt.shape):
            mismatch.append({"key": key, "source_shape": list(src.shape), "target_shape": list(tgt.shape)})
            continue
        load_state[key] = src.to(dtype=target_state[key].dtype)
        if src.is_floating_point() and src.numel():
            sf, tf = src.float().reshape(-1), tgt.float().reshape(-1)
            tn = float(tf.norm())
            sn = float(sf.norm())
            drift_rows.append({
                "key": key,
                "group": D.block_group_from_key(key),
                "numel": int(src.numel()),
                "cosine_to_pretrained": float(F.cosine_similarity(sf, tf, dim=0, eps=1e-12)) if sn > 0 and tn > 0 else np.nan,
                "relative_l2_to_pretrained": float((sf - tf).norm() / max(tn, 1e-12)),
                "max_abs_delta": float((sf - tf).abs().max()),
            })

    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_state_keys": len(state),
        "source_visual_keys": len(source_visual),
        "target_visual_keys": len(target_visual_keys),
        "loaded_visual_keys": len(load_state),
        "missing_target_visual_keys": missing,
        "shape_mismatch": mismatch,
        "ignored_extra_visual_keys": sorted(k for k in source_visual if k not in target_visual_keys),
        "text_tower_loaded_from_checkpoint": False,
        "gmp_geometric_matrices_materialized": int(gmp_materialized),
    }
    (audit_dir / "checkpoint_visual_transplant_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if missing or mismatch:
        raise RuntimeError(f"Strict visual transplant failed for {label}: missing={len(missing)} mismatch={len(mismatch)}. See audit.")

    current = bundle.model.state_dict()
    current.update(load_state)
    bundle.model.load_state_dict(current, strict=True)
    bundle.model = D.freeze_eval(bundle.model)
    bundle.source = f"ordinary_visual:{checkpoint} + pretrained_text:{args.clip_module}:{args.model_spec}"

    drift = pd.DataFrame(drift_rows)
    if len(drift):
        drift.to_csv(audit_dir / "visual_parameter_drift_per_tensor.csv", index=False)
        sm = []
        for group, q in drift.groupby("group"):
            sm.append({
                "group": group,
                "tensors": len(q),
                "numel": int(q.numel.sum()),
                "cosine_mean": float(np.average(q.cosine_to_pretrained.fillna(1.0), weights=q.numel)),
                "relative_l2_rms": float(np.sqrt(np.average(np.square(q.relative_l2_to_pretrained), weights=q.numel))),
                "max_abs_delta": float(q.max_abs_delta.max()),
            })
        pd.DataFrame(sm).to_csv(audit_dir / "visual_parameter_drift_summary.csv", index=False)
    return bundle


def load_bundle(model_name: str, args, audit_dir: Path):
    b = D.load_pretrained_bundle(args)
    if model_name == MODEL_PRE:
        return b
    if model_name == MODEL_FT:
        b, drift = D.transplant_stripped_xattn_visual(b, args, audit_dir)
        if len(drift):
            drift.to_csv(audit_dir / "visual_parameter_drift.csv", index=False)
        return b
    attr = {
        MODEL_GMP: "gmp_checkpoint",
        MODEL_REG: "regression_checkpoint",
        MODEL_BRUT: "brut_checkpoint",
        MODEL_SAE: "sae_hinge_checkpoint",
    }.get(model_name)
    if attr is None:
        raise ValueError(f"Unknown model: {model_name}")
    return _strict_visual_transplant(b, Path(getattr(args, attr)), args, audit_dir, model_name)


# =============================================================================
# Oracle and extraction
# =============================================================================

@dataclass
class BatchOracle:
    b13_regs: List[set]
    b13_primary_reg: List[int]
    b13_primary_patch_sink: List[int]
    probe_patch: List[int]
    b20_newnorm: List[set] = field(default_factory=list)
    intact_refs: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)


@torch.no_grad()
def get_b13_oracle(bundle, images: torch.Tensor, args) -> BatchOracle:
    """Cheap intact pilot through B13 attention.

    This lets every earlier block be annotated using *future B13 register
    addresses*, so CLS-query spatial plots genuinely cover the entire model.
    """
    v = bundle.model.visual
    x = v._prepare_tokens(images.to(bundle.device, dtype=bundle.model.dtype))
    B = images.shape[0]
    H = int(v.transformer.resblocks[0].attn.num_heads)
    T = int(x.shape[0])

    for b, blk in enumerate(v.transformer.resblocks):
        if b == B13_PRE:
            regs = patch_set_from_norms(x, args.massive_threshold)
            ln1 = blk.ln_1(x)
            _, probs0 = blk.attention(ln1, need_weights=True, capture=True)
            probs = normalize_probs_shape(probs0, B, H, T).float()
            inflow = probs.mean(dim=1).mean(dim=1)  # [B,T]
            primary_patch, primary_reg, probe = [], [], []
            for bi in range(B):
                # strongest spatial sink irrespective of visible-register status
                pp = int(torch.argmax(inflow[bi, 1:]).item())
                primary_patch.append(pp)
                if regs[bi]:
                    cand = torch.as_tensor([p + 1 for p in sorted(regs[bi])], device=inflow.device)
                    rr = int(cand[torch.argmax(inflow[bi, cand])].item()) - 1
                else:
                    rr = -1
                primary_reg.append(rr)
                probe.append(rr if rr >= 0 else pp)
            clear_attn_cache(blk)
            return BatchOracle(regs, primary_reg, primary_patch, probe, [set() for _ in range(B)], {})

        ln1 = blk.ln_1(x)
        attn_out, _ = blk.attention(ln1, need_weights=True, capture=True)
        x_attn = x + attn_out
        ln2 = blk.ln_2(x_attn)
        x = x_attn + blk.mlp.c_proj(blk.mlp.gelu(blk.mlp.c_fc(ln2)))
        clear_attn_cache(blk)

    raise RuntimeError("B13 oracle failed: model has fewer than 14 visual blocks")


# =============================================================================
# Summaries
# =============================================================================


# =============================================================================
# Plotting
# =============================================================================


# =============================================================================
# Reporting / compact handoff bundle
# =============================================================================


# =============================================================================
# CLI / tests
# =============================================================================


from types import SimpleNamespace
S = SimpleNamespace(ClipBundle=ClipBundle, DEFAULT_CLIP_MODULE=DEFAULT_CLIP_MODULE, DEFAULT_MODEL_SPEC=DEFAULT_MODEL_SPEC, load_clip=load_clip)
D = SimpleNamespace(ROOT_CUSTOM_EXACT=ROOT_CUSTOM_EXACT, ROOT_CUSTOM_PREFIXES=ROOT_CUSTOM_PREFIXES, VISUAL_CUSTOM_EXACT=VISUAL_CUSTOM_EXACT, VISUAL_CUSTOM_SUBSTRINGS=VISUAL_CUSTOM_SUBSTRINGS, _looks_like_clip_state=_looks_like_clip_state, block_group_from_key=block_group_from_key, extract_state_dict=extract_state_dict, freeze_eval=freeze_eval, load_pretrained_bundle=load_pretrained_bundle, safe_torch_load=safe_torch_load, strip_common_prefixes=strip_common_prefixes, transplant_stripped_xattn_visual=transplant_stripped_xattn_visual)
