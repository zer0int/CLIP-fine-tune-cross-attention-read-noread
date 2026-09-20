#!/usr/bin/env python3
'\nPIECES bridge / text-tower / visual-tower transplant atlas\n==========================================================\n\nPurpose\n-------\nSeparate four questions that have become entangled in the RN results:\n\n    A. Does the learned RN token / B13 entry subspace transfer to pretrained CLIP?\n    B. Does the full learned cross-attention bridge transfer?\n    C. Is the bridge\'s Q-side interface co-adapted to the trained text encoder?\n    D. What changes when the native B11/B12 register-pump neurons are removed?\n\nThe four encoder/bridge conditions are a strict 2x2:\n\n    A_native\n        trained ViT + trained text tower + trained bridge/RN\n\n    B_preV_preT\n        pretrained ViT + pretrained text tower + trained bridge/RN\n\n    C_preV_trainT\n        pretrained ViT + trained text tower + trained bridge/RN\n\n    D_trainV_preT\n        trained ViT + pretrained text tower + trained bridge/RN\n\nEvery condition is evaluated under:\n    intact\n    no_pump\n\n`no_pump` zeros the COMPLETE currently established B11/B12 register-pump unit\nsets POST-QuickGELU and BEFORE c_proj, on every visual token:\n\n    B11: 9,987,1100,1967,2555,3661,3784\n    B12: 42,183,983,1571,1816,2687,3002,3008,3868\n\nThe ablation is instrumented.  The output contains call counts, number of values\nzeroed, pre-zero absmean/RMS/max, and post-zero max (must be exactly 0).\n\nTransplant policy\n-----------------\nAll variants are instantiated from the trained PIECES checkpoint, therefore\ntheir custom architecture is identical.  We then replace ONLY canonical\nordinary CLIP weights from a stock OpenAI ViT-L/14 source.\n\nProtected trained state remains identical in ALL four conditions:\n    - read_implant.*   (SOURCE / ORTHO / READ / CONTENT / router etc.)\n    - visual.read_null_token and READ_NULL configuration\n    - PIECES-only root state\n    - special token embedding rows >= the stock OpenAI vocab size\n    - trained logit_scale (kept fixed to avoid an irrelevant scalar confound)\n\nFor pretrained-text variants, token_embedding rows shared with stock OpenAI\nCLIP are copied from pretrained; extra PIECES special rows are retained from\nthe trained model.  This is deliberate: leaving <text>/<null>/etc. random would\nmake the comparison meaningless.\n\nMain science\n------------\n1) B13 RN entry-basis universality\n   - fit a local RN basis separately in every model x pump condition\n   - principal cosines to the trained-intact reference basis\n   - reference-rank4 capture of each model\'s actual RN-induced B13 delta\n\n2) SAME-COORDINATE control-surface atlas\n   - all eight conditions are poked in the TRAINED-INTACT reference coordinates\n   - PC1xPC2 and PC1xPC4 17x17 grids\n   - en,de,ar,zh,ru; English and native queries\n   - SynthRTA + exact paired NoRTA\n   - B21 high-D state surfaces\n   - local tangent / bending geometry\n   - scalar READ surfaces\n   - cross-language geometry\n   - cross-MODEL alignment to A_native/intact\n\n3) 4-D ridge / basin search\n   - Sobol points in the same trained-intact PC1..PC4 coordinate chart\n   - top high/low READ loci and alpha-space PCA\n\n4) 4-D B21 state Jacobian\n   - anchor + discovered max/min\n   - singular spectrum / participation-ratio effective rank\n\n5) Text-tower coordinate-contract audit\n   Same forced-<text> candidate bank is encoded through:\n       trained text tower\n       pretrained text tower + trained PIECES special rows\n   and compared at:\n       - classic/bare pre-ln_final EOT hidden\n       - classic/bare final CLIP text embedding\n       - forced-<text> pre-ln_final EOT hidden\n       - READ bridge text_ln output\n       - READ bridge Q projection\n       - ORTHO bridge text_ln output\n       - ORTHO bridge Q projection\n       - forced-<text> final CLIP text embedding\n\n   Metrics:\n       - paired row cosine distribution\n       - centered linear CKA\n       - orthogonal Procrustes train fit + held-out residual/cosine\n       - top principal cosines between centered feature subspaces\n\nInterpretation guardrail\n------------------------\n"RN universality" is deliberately split into:\n    token / B13 address universality\n    entry-subspace universality\n    downstream trajectory universality\n    text-conditioned readout universality\n\nThis script does NOT assume those are the same thing.\n\nRequired beside this script\n---------------------------\n    probe_tools_rn_control.py\n    probe_tools_rn_manifold.py\n\nDefault output\n--------------\n    rn_bridge_transplant_universality/\n        data/\n        plots/\n        ply/\n        audits/\n        SUMMARY.txt\n        compact_summary_rn_bridge_transplant.zip\n'
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()


import argparse
import contextlib
import csv
import gc
import hashlib
import importlib
import json
import math
import random
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import probe_tools_rn_control as base
import probe_tools_rn_manifold as atlas


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_CHECKPOINT = base.DEFAULT_CHECKPOINT
DEFAULT_PRETRAINED_SPEC = "ViT-L/14"
DEFAULT_PRETRAINED_MODULE = "attnclip_mechinterp_sae"
DEFAULT_DATASET_REPO = "zer0int/RTA-100-Multilingual"
DEFAULT_LANGUAGES = "en,de,es,fr,ar,ko,zh,ru"
DEFAULT_QUERY_MODES = "english,native"
DEFAULT_PLANES = "1x2,1x4"
DEFAULT_GRID_POINTS = 17
DEFAULT_GRID_BOUND = 2.0
DEFAULT_ATLAS_KEYS = 10
DEFAULT_BASIS_PAIRS_PER_LANGUAGE = 8
DEFAULT_SURFACE_BATCH = 48
DEFAULT_RIDGE_POINTS = 768
DEFAULT_RIDGE_BOUND = 2.0
DEFAULT_RIDGE_RADIUS = 3.0
DEFAULT_RIDGE_BATCH = 64
DEFAULT_RIDGE_TOPK = 16
DEFAULT_RIDGE_REFINE_STEPS = 2
DEFAULT_RIDGE_REFINE_RANDOM = 32
DEFAULT_STATE_JAC_EPS = 0.20
DEFAULT_TEXT_ALIGN_VOCAB = "vocab_deduped.txt"
DEFAULT_TEXT_ALIGN_MAX = 4096
DEFAULT_TEXT_ALIGN_BATCH = 512
DEFAULT_TEXT_ALIGN_HOLDOUT = 0.20
DEFAULT_SEED = 20260902

OPENAI_VOCAB_SIZE = 49408

REG_NEURONS: dict[int, list[int]] = {
    11: [9, 987, 1100, 1967, 2555, 3661, 3784],
    12: [42, 183, 983, 1571, 1816, 2687, 3002, 3008, 3868],
}

MODEL_SPECS = (
    ("A_native", False, False,
     "trained ViT + trained text tower + trained bridge/RN"),
    ("B_preV_preT", True, True,
     "pretrained ViT + pretrained text tower + trained bridge/RN"),
    #("C_preV_trainT", True, False,
    # "pretrained ViT + trained text tower + trained bridge/RN"),
    #("D_trainV_preT", False, True,
    #"trained ViT + pretrained text tower + trained bridge/RN"),
)

#PUMP_MODES = ("intact", "no_pump")
PUMP_MODES = ("intact",)

STATE_BLOCK = 21

TEXT_STAGE_ORDER = (
    "classic_eot_pre_ln",
    "classic_final_text_embedding",
    "forced_eot_pre_ln",
    "read_text_ln",
    "read_q",
    "ortho_text_ln",
    "ortho_q",
    "forced_final_text_embedding",
)


# =============================================================================
# Generic helpers
# =============================================================================

def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    base.save_rows(path, rows)


def save_json(path: Path, payload: Any) -> None:
    base.save_json(path, payload)


def safe_float(x: Any) -> float:
    return base.safe_float(x)


def stable_slug(x: str) -> str:
    return base.stable_slug(x)


def parse_strs(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_planes(text: str) -> list[tuple[int, int]]:
    return atlas.parse_planes(text)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tensor_digest(t: torch.Tensor, n: int = 4096) -> str:
    x = t.detach().cpu().contiguous().reshape(-1)
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    if x.numel():
        h.update(x[: min(n, x.numel())].numpy().tobytes())
    return h.hexdigest()[:16]


def state_digest(state: Mapping[str, torch.Tensor], keys: Sequence[str]) -> str:
    h = hashlib.sha256()
    for k in sorted(keys):
        if k not in state:
            continue
        v = state[k]
        if not torch.is_tensor(v):
            continue
        h.update(k.encode())
        h.update(tensor_digest(v).encode())
    return h.hexdigest()[:20]


def principal_cosines(a_kd: torch.Tensor, b_kd: torch.Tensor) -> np.ndarray:
    qa = torch.linalg.qr(a_kd.float().T, mode="reduced").Q
    qb = torch.linalg.qr(b_kd.float().T, mode="reduced").Q
    s = torch.linalg.svdvals(qa.T @ qb)
    return s.detach().cpu().numpy().astype(np.float64)


def reference_capture_fraction(delta_btd: torch.Tensor, basis_kd: torch.Tensor, rank: int = 4) -> float:
    return base.feature_subspace_fraction(delta_btd, basis_kd, rank)


def parameter_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    den = x.norm() * y.norm()
    if float(den) < 1e-12:
        return float("nan")
    return float((x @ y / den).cpu())


# =============================================================================
# Canonical pretrained source
# =============================================================================

def _maybe_convert_source_state(module_name: str, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Most recent attnclip_mechinterp_sae already stores explicit q/k/v.
    If a loader returns stock packed in_proj tensors, use the module's converter
    when available.
    """
    if any(".attn.q_proj.weight" in k for k in state):
        return state
    try:
        model_mod = importlib.import_module(f"{module_name}.model")
        convert = getattr(model_mod, "convert_state_dict_inproj_to_qkv", None)
        if callable(convert):
            return convert(dict(state))
    except Exception:
        pass
    return state


def load_pretrained_cpu_state(
    module_name: str,
    spec: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    print(f"[pretrained source] module={module_name} spec={spec} device=cpu")
    mod = importlib.import_module(module_name)
    load_fn = getattr(mod, "load")
    try:
        model, preprocess = load_fn(
            spec, device="cpu", jit=False, read_null_enabled=False
        )
    except TypeError:
        model, preprocess = load_fn(spec, device="cpu", jit=False)

    model = model.float().eval()
    sd = {
        str(k): v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if torch.is_tensor(v)
    }
    sd = _maybe_convert_source_state(module_name, sd)

    meta = {
        "module": module_name,
        "spec": spec,
        "state_tensor_count": len(sd),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "visual_depth": int(len(model.visual.transformer.resblocks)),
        "visual_width": int(model.visual.class_embedding.numel()),
        "input_resolution": int(model.visual.input_resolution),
        "token_embedding_rows": int(sd["token_embedding.weight"].shape[0]),
        "token_embedding_dim": int(sd["token_embedding.weight"].shape[1]),
        "text_projection_shape": list(sd["text_projection"].shape),
    }
    del model
    cleanup()
    return sd, meta


# =============================================================================
# Transplant construction
# =============================================================================

TEXT_EXACT_KEYS = {
    "positional_embedding",
    "text_projection",
}
TEXT_PREFIXES = (
    "transformer.",
    "ln_final.",
)
PROTECTED_PREFIXES = (
    "read_implant.",
)
PROTECTED_EXACT = {
    "visual.read_null_token",
    "visual.read_null_insert_block_config",
    "hard_text_embedding",
    "null_text_embedding",
}
VISUAL_CUSTOM_SUBSTR = ("read_null",)


def is_canonical_text_key(key: str) -> bool:
    if key == "token_embedding.weight":
        return True
    if key in TEXT_EXACT_KEYS:
        return True
    return any(key.startswith(p) for p in TEXT_PREFIXES)


def protected_keys(state: Mapping[str, torch.Tensor]) -> list[str]:
    out = []
    for k in state:
        if k in PROTECTED_EXACT or any(k.startswith(p) for p in PROTECTED_PREFIXES):
            out.append(k)
        elif k.startswith("visual.") and any(s in k.lower() for s in VISUAL_CUSTOM_SUBSTR):
            out.append(k)
    return sorted(set(out))


def apply_pretrained_transplant(
    model: torch.nn.Module,
    pretrained_state: Mapping[str, torch.Tensor],
    *,
    replace_visual: bool,
    replace_text: bool,
    audit_path: Path,
) -> dict[str, Any]:
    target = model.state_dict()
    before_protected = {k: target[k].detach().cpu().clone() for k in protected_keys(target)}
    trained_special_rows_before = None
    if (
        "token_embedding.weight" in target
        and target["token_embedding.weight"].shape[0] > OPENAI_VOCAB_SIZE
    ):
        trained_special_rows_before = (
            target["token_embedding.weight"][OPENAI_VOCAB_SIZE:]
            .detach().cpu().clone()
        )

    rows = []
    load = {}

    # Ordinary visual tower.
    if replace_visual:
        target_visual = [
            k for k in target
            if k.startswith("visual.")
            and not any(s in k.lower() for s in VISUAL_CUSTOM_SUBSTR)
        ]
        for k in target_visual:
            if k not in pretrained_state:
                rows.append({"section":"visual","key":k,"status":"missing_source"})
                continue
            src = pretrained_state[k]
            if tuple(src.shape) != tuple(target[k].shape):
                rows.append({
                    "section":"visual","key":k,"status":"shape_mismatch",
                    "source_shape":list(src.shape),"target_shape":list(target[k].shape),
                })
                continue
            load[k] = src.to(dtype=target[k].dtype)
            rows.append({
                "section":"visual","key":k,"status":"loaded",
                "cosine_before": parameter_cosine(target[k].cpu(), src),
            })

    # Text tower.  Keep trained logit_scale.  Token embedding gets a special
    # row-wise transplant so PIECES-only token rows remain trained.
    if replace_text:
        for k in target:
            if not is_canonical_text_key(k):
                continue

            if k == "token_embedding.weight":
                if k not in pretrained_state:
                    rows.append({"section":"text","key":k,"status":"missing_source"})
                    continue
                src = pretrained_state[k]
                dst = target[k].detach().cpu().clone()
                if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != dst.shape[1]:
                    rows.append({
                        "section":"text","key":k,"status":"shape_mismatch",
                        "source_shape":list(src.shape),"target_shape":list(dst.shape),
                    })
                    continue
                ncopy = min(int(src.shape[0]), int(dst.shape[0]), OPENAI_VOCAB_SIZE)
                merged = dst.clone()
                merged[:ncopy].copy_(src[:ncopy].to(dtype=merged.dtype))
                load[k] = merged
                rows.append({
                    "section":"text","key":k,"status":"loaded_rows",
                    "rows_copied":ncopy,
                    "rows_protected":int(dst.shape[0]-ncopy),
                    "source_rows":int(src.shape[0]),
                    "target_rows":int(dst.shape[0]),
                })
                continue

            if k not in pretrained_state:
                rows.append({"section":"text","key":k,"status":"missing_source"})
                continue
            src = pretrained_state[k]
            if tuple(src.shape) != tuple(target[k].shape):
                rows.append({
                    "section":"text","key":k,"status":"shape_mismatch",
                    "source_shape":list(src.shape),"target_shape":list(target[k].shape),
                })
                continue
            load[k] = src.to(dtype=target[k].dtype)
            rows.append({
                "section":"text","key":k,"status":"loaded",
                "cosine_before": parameter_cosine(target[k].cpu(), src),
            })

    # Strictness: if a requested canonical block has missing/shape mismatches,
    # fail.  Silent partial transplants are scientifically useless here.
    bad = [r for r in rows if r["status"] in {"missing_source","shape_mismatch"}]
    if bad:
        save_rows(audit_path.with_suffix(".csv"), rows)
        raise RuntimeError(
            f"Transplant incomplete: {len(bad)} missing/shape mismatches. "
            f"See {audit_path.with_suffix('.csv')}"
        )

    current = model.state_dict()
    current.update(load)
    model.load_state_dict(current, strict=True)

    # Protected custom bridge/RN state must remain bit-identical.
    after = model.state_dict()
    protected_mismatch = []
    for k, ref in before_protected.items():
        got = after[k].detach().cpu()
        if got.shape != ref.shape or got.dtype != ref.dtype or not torch.equal(got, ref):
            protected_mismatch.append(k)

    # Special token rows must remain exact if target has them.
    special_ok = True
    special_rows = 0
    t_after = after["token_embedding.weight"].detach().cpu()
    if t_after.shape[0] > OPENAI_VOCAB_SIZE:
        special_rows = int(t_after.shape[0] - OPENAI_VOCAB_SIZE)
        if trained_special_rows_before is None:
            special_ok = False
        else:
            special_ok = bool(torch.equal(
                trained_special_rows_before,
                t_after[OPENAI_VOCAB_SIZE:],
            ))

    audit = {
        "replace_visual": bool(replace_visual),
        "replace_text": bool(replace_text),
        "loaded_tensor_count": len(load),
        "protected_key_count": len(before_protected),
        "protected_state_digest_before": state_digest(before_protected, list(before_protected)),
        "protected_state_digest_after": state_digest(
            {k: after[k].detach().cpu() for k in before_protected},
            list(before_protected),
        ),
        "protected_mismatch": protected_mismatch,
        "special_rows": special_rows,
        "special_rows_bit_identical": special_ok,
        "logit_scale_kept_trained": True,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    save_rows(audit_path.with_suffix(".csv"), rows)
    save_json(audit_path.with_suffix(".json"), audit)

    if protected_mismatch or not special_ok:
        raise RuntimeError(
            f"Protected PIECES state changed during transplant: "
            f"protected_mismatch={protected_mismatch[:5]} special_ok={special_ok}"
        )
    return audit


def _trained_custom_digest(model: torch.nn.Module) -> str:
    state = model.state_dict()
    required = {
        "visual.read_null_token",
        "visual.read_null_insert_block_config",
        "hard_text_embedding",
        "null_text_embedding",
        "read_implant.read_bridge.q_proj.weight",
        "read_implant.content_pool.query",
        "read_implant.orthographic_bridge.q_proj.weight",
        "read_implant.source_head.patch_out.weight",
        "read_implant.trust_router.fc1.weight",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(
            "Bridge-transplant donor is not a complete trained x-attn model; "
            f"missing custom tensors={missing}. Random/partial bridge state is invalid."
        )
    keys = protected_keys(state)
    if not keys:
        raise RuntimeError("Bridge-transplant donor exposes no protected trained custom state")
    return state_digest({k: state[k].detach().cpu() for k in keys}, keys)


def load_variant(
    args: argparse.Namespace,
    pretrained_state: Mapping[str, torch.Tensor],
    model_name: str,
    replace_visual: bool,
    replace_text: bool,
    audit_dir: Path,
    *,
    expected_donor_custom_digest: str | None = None,
):
    loaded = base.load_model(
        args.checkpoint,
        package_root=args.module_root,
        device=args.device,
    )
    model = loaded.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    donor_digest = _trained_custom_digest(model)
    if (
        expected_donor_custom_digest is not None
        and donor_digest != expected_donor_custom_digest
    ):
        raise RuntimeError(
            f"{model_name}: trained bridge/RN donor digest differs from the reference donor. "
            "Refusing to run a mixed or randomly initialized transplant."
        )

    audit = apply_pretrained_transplant(
        model,
        pretrained_state,
        replace_visual=replace_visual,
        replace_text=replace_text,
        audit_path=audit_dir / model_name / "transplant_audit",
    )
    audit["trained_donor_custom_digest"] = donor_digest
    audit["matches_reference_donor"] = (
        expected_donor_custom_digest is None
        or donor_digest == expected_donor_custom_digest
    )
    save_json(audit_dir / model_name / "transplant_audit.json", audit)
    return loaded, audit


# =============================================================================
# Proper register-pump post-QuickGELU intervention
# =============================================================================

def _find_mlp_activation_module(block: nn.Module) -> nn.Module:
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        raise AttributeError(f"Block has no .mlp: {block}")

    for name in ("gelu", "quick_gelu", "act", "activation"):
        mod = getattr(mlp, name, None)
        if isinstance(mod, nn.Module):
            return mod

    if isinstance(mlp, nn.Sequential):
        for mod in mlp:
            if "gelu" in mod.__class__.__name__.lower():
                return mod

    # OpenAI-style MLP sometimes exposes children c_fc / gelu / c_proj.
    for _name, mod in mlp.named_children():
        if "gelu" in mod.__class__.__name__.lower():
            return mod

    raise RuntimeError(
        "Could not locate visual MLP GELU/QuickGELU module; refusing to zero "
        f"register neurons at an ambiguous location. MLP={mlp}"
    )


@dataclass
class PumpStats:
    block: int
    units: list[int]
    calls: int = 0
    selected_values: int = 0
    pre_abs_sum: float = 0.0
    pre_sq_sum: float = 0.0
    pre_max_abs: float = 0.0
    post_max_abs: float = 0.0
    output_width: int = 0


class RegisterPumpContext:
    """
    Logs the selected post-QuickGELU units in both modes.
    In no_pump mode, zeros them before c_proj.
    """
    def __init__(self, model: torch.nn.Module, mode: str, verbose: bool = True):
        if mode not in {"intact","no_pump"}:
            raise ValueError(mode)
        self.model = model
        self.mode = mode
        self.verbose = verbose
        self.handles = []
        self.stats = {
            b: PumpStats(block=b, units=list(units))
            for b, units in REG_NEURONS.items()
        }

    def __enter__(self):
        blocks = self.model.visual.transformer.resblocks
        for b, units in REG_NEURONS.items():
            act = _find_mlp_activation_module(blocks[b])
            idx_cpu = torch.tensor(units, dtype=torch.long)
            st = self.stats[b]

            def hook(_module, _inp, output, b=b, idx_cpu=idx_cpu, st=st):
                if not torch.is_tensor(output):
                    raise TypeError(f"B{b} activation output is {type(output)}")
                if output.ndim < 2:
                    raise RuntimeError(f"B{b} activation output shape={tuple(output.shape)}")
                if max(st.units) >= output.shape[-1]:
                    raise IndexError(
                        f"B{b}: activation width={output.shape[-1]}, "
                        f"cannot access units={st.units}"
                    )
                idx = idx_cpu.to(output.device)
                selected = output.index_select(-1, idx).detach().float()
                st.calls += 1
                st.selected_values += int(selected.numel())
                st.pre_abs_sum += float(selected.abs().sum().cpu())
                st.pre_sq_sum += float(selected.square().sum().cpu())
                st.pre_max_abs = max(st.pre_max_abs, float(selected.abs().max().cpu()))
                st.output_width = int(output.shape[-1])

                if st.calls == 1 and self.verbose:
                    print(
                        f"[register pump {self.mode}] B{b} units={st.units} "
                        f"first-call absmean={float(selected.abs().mean()):.6g} "
                        f"rms={float(selected.square().mean().sqrt()):.6g} "
                        f"max={float(selected.abs().max()):.6g}"
                    )

                if self.mode == "no_pump":
                    out = output.clone()
                    out[..., idx] = 0
                    post = out.index_select(-1, idx).detach()
                    pm = float(post.abs().max().float().cpu())
                    st.post_max_abs = max(st.post_max_abs, pm)
                    if pm != 0.0:
                        raise RuntimeError(f"B{b} pump zero failed: post max={pm}")
                    return out
                else:
                    st.post_max_abs = max(
                        st.post_max_abs,
                        float(selected.abs().max().cpu())
                    )
                    return output

            self.handles.append(act.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def rows(self, model_name: str) -> list[dict[str, Any]]:
        rows = []
        for b, st in sorted(self.stats.items()):
            n = max(st.selected_values, 1)
            rows.append({
                "model_name": model_name,
                "pump_mode": self.mode,
                "block": b,
                "units": ",".join(map(str, st.units)),
                "n_units": len(st.units),
                "calls": st.calls,
                "selected_values": st.selected_values,
                "activation_width": st.output_width,
                "pre_absmean": st.pre_abs_sum / n,
                "pre_rms": math.sqrt(st.pre_sq_sum / n),
                "pre_max_abs": st.pre_max_abs,
                "post_selected_max_abs": st.post_max_abs,
                "zero_verified": bool(self.mode == "intact" or st.post_max_abs == 0.0),
            })
        return rows


# =============================================================================
# Text coordinate-contract audit
# =============================================================================

def load_text_alignment_bank(
    vocab_path: Path,
    max_vocab: int,
    seed: int,
    mandatory: Sequence[str],
) -> list[str]:
    mandatory_clean = []
    seen = set()
    for x in mandatory:
        w = str(x).strip()
        if w and w not in seen:
            seen.add(w)
            mandatory_clean.append(w)

    words = []
    if vocab_path.is_file():
        for raw in vocab_path.read_text(encoding="utf-8", errors="replace").splitlines():
            w = raw.strip()
            if w and w not in seen:
                words.append(w)

    if max_vocab > 0 and len(words) > max_vocab:
        rng = random.Random(seed)
        words = rng.sample(words, max_vocab)

    return mandatory_clean + words


def _bridge_stage_modules(model: torch.nn.Module):
    implant = model.read_implant
    read = getattr(implant, "read_bridge")
    ortho = getattr(implant, "orthographic_bridge")
    return read, ortho


def collect_text_stages(
    model: torch.nn.Module,
    clip_mod: Any,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    out = {stage: [] for stage in TEXT_STAGE_ORDER}
    read_bridge, ortho_bridge = _bridge_stage_modules(model)

    for st in tqdm(range(0, len(texts), batch_size), desc="text stages", leave=False):
        batch = texts[st:st+batch_size]
        payload = [f"<text> {x}" for x in batch]
        try:
            toks = clip_mod.tokenize(payload, truncate=True).to(device)
            bare_toks = clip_mod.tokenize(batch, truncate=True).to(device)
        except TypeError:
            toks = clip_mod.tokenize(payload).to(device)
            bare_toks = clip_mod.tokenize(batch).to(device)

        with torch.no_grad(), base.model_autocast_context(model):
            # Ordinary vanilla CLIP path, useful because older encoder transplants
            # often remained compatible at the final embedding despite internal drift.
            classic = model._encode_text_hidden(bare_toks)

            # Exact PIECES forced-<text> query path.
            prepared = model.prepare_mode_tokens(toks)
            info = model._encode_text_hidden(prepared["read_tokens"])
            h = info["eot_hidden_pre_ln"]

            rln = read_bridge.text_ln(h)
            rq = read_bridge.q_proj(rln)

            oln = ortho_bridge.text_ln(h)
            oq = ortho_bridge.q_proj(oln)

        arrays = {
            "classic_eot_pre_ln": classic["eot_hidden_pre_ln"],
            "classic_final_text_embedding": classic["text_embedding"],
            "forced_eot_pre_ln": h,
            "read_text_ln": rln,
            "read_q": rq,
            "ortho_text_ln": oln,
            "ortho_q": oq,
            "forced_final_text_embedding": info["text_embedding"],
        }
        for name, val in arrays.items():
            out[name].append(val.detach().float().cpu().numpy().astype(np.float32))

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def centered_linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    X = np.asarray(x, np.float64)
    Y = np.asarray(y, np.float64)
    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)
    cross = X.T @ Y
    xx = X.T @ X
    yy = Y.T @ Y
    num = float(np.sum(cross * cross))
    den = math.sqrt(float(np.sum(xx * xx)) * float(np.sum(yy * yy)))
    return num / den if den > 1e-20 else float("nan")


def paired_cosines(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(x, np.float64)
    Y = np.asarray(y, np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return np.sum(X * Y, axis=1)


def fit_orthogonal_procrustes(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(x_train, np.float64)
    Y = np.asarray(y_train, np.float64)
    mx = X.mean(axis=0, keepdims=True)
    my = Y.mean(axis=0, keepdims=True)
    Xc = X - mx
    Yc = Y - my
    # R solves min ||X R - Y||, X and Y same width.
    U, _s, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    return R, mx, my


def procrustes_metrics(
    x: np.ndarray,
    y: np.ndarray,
    holdout: float,
    seed: int,
) -> dict[str, float]:
    X = np.asarray(x, np.float64)
    Y = np.asarray(y, np.float64)
    if X.shape != Y.shape:
        raise ValueError(f"Procrustes requires same shapes, got {X.shape}, {Y.shape}")
    n = len(X)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    ntest = max(1, int(round(n * holdout)))
    test_idx = order[:ntest]
    train_idx = order[ntest:]
    if len(train_idx) < 2:
        train_idx = order[:-1]
        test_idx = order[-1:]

    R, mx, my = fit_orthogonal_procrustes(X[train_idx], Y[train_idx])

    def eval_idx(idx):
        xp = (X[idx] - mx) @ R + my
        yt = Y[idx]
        resid = np.linalg.norm(xp - yt) / max(np.linalg.norm(yt - my), 1e-12)
        cos = paired_cosines(xp, yt)
        return float(resid), float(np.mean(cos)), float(np.median(cos))

    tr = eval_idx(train_idx)
    te = eval_idx(test_idx)
    return {
        "procrustes_train_relative_residual": tr[0],
        "procrustes_train_cos_mean": tr[1],
        "procrustes_train_cos_median": tr[2],
        "procrustes_test_relative_residual": te[0],
        "procrustes_test_cos_mean": te[1],
        "procrustes_test_cos_median": te[2],
        "procrustes_train_n": len(train_idx),
        "procrustes_test_n": len(test_idx),
    }


def centered_subspace_cosines(x: np.ndarray, y: np.ndarray, rank: int = 8) -> np.ndarray:
    X = np.asarray(x, np.float64)
    Y = np.asarray(y, np.float64)
    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)
    _, _, Vx = np.linalg.svd(X, full_matrices=False)
    _, _, Vy = np.linalg.svd(Y, full_matrices=False)
    k = min(rank, Vx.shape[0], Vy.shape[0])
    s = np.linalg.svd(Vx[:k] @ Vy[:k].T, compute_uv=False)
    return s


def text_alignment_rows(
    trained: dict[str, np.ndarray],
    pretrained: dict[str, np.ndarray],
    holdout: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for si, stage in enumerate(TEXT_STAGE_ORDER):
        X = pretrained[stage]
        Y = trained[stage]
        if X.shape != Y.shape:
            rows.append({
                "stage": stage,
                "status": "shape_mismatch",
                "pretrained_shape": str(X.shape),
                "trained_shape": str(Y.shape),
            })
            continue

        cos = paired_cosines(X,Y)
        proc = procrustes_metrics(X,Y,holdout,seed+si)
        pcs = centered_subspace_cosines(X,Y,rank=8)
        row = {
            "stage": stage,
            "status": "ok",
            "n": len(X),
            "dim": X.shape[1],
            "paired_cos_mean": float(np.mean(cos)),
            "paired_cos_median": float(np.median(cos)),
            "paired_cos_p05": float(np.quantile(cos,0.05)),
            "paired_cos_p95": float(np.quantile(cos,0.95)),
            "linear_cka": centered_linear_cka(X,Y),
            **proc,
        }
        for i, v in enumerate(pcs,1):
            row[f"principal_cos_{i}"] = float(v)
        rows.append(row)
    return rows


# =============================================================================
# Basis fit / audit per model condition
# =============================================================================

def basis_audit_rows(
    model_name: str,
    pump_mode: str,
    basis_info: dict[str, Any],
    reference_basis: torch.Tensor,
) -> list[dict[str, Any]]:
    b = basis_info["basis_kd"][:4].float()
    pcs = principal_cosines(reference_basis[:4], b)
    rows = []
    for i in range(4):
        rows.append({
            "model_name": model_name,
            "pump_mode": pump_mode,
            "pc": i+1,
            "singular_value": float(basis_info["singular_values"][i].cpu()),
            "explained_energy": float(basis_info["explained_energy"][i]),
            "individual_abs_cos_to_reference_pc": abs(parameter_cosine(reference_basis[i], b[i])),
            "subspace_principal_cos": float(pcs[i]),
        })
    return rows


# =============================================================================
# Atlas state accumulation / geometry
# =============================================================================

@dataclass
class SurfaceAcc:
    state_sum: np.ndarray
    emb_sum: np.ndarray
    count: int


def add_surface_acc(
    acc: dict[tuple, SurfaceAcc],
    key: tuple,
    state: np.ndarray,
    emb: np.ndarray,
) -> None:
    if key not in acc:
        acc[key] = SurfaceAcc(
            np.zeros_like(state,dtype=np.float64),
            np.zeros_like(emb,dtype=np.float64),
            0,
        )
    acc[key].state_sum += state
    acc[key].emb_sum += emb
    acc[key].count += 1


def choose_surface_scalar(
    surface_summary: list[dict[str,Any]],
    model_name: str,
    pump_mode: str,
    condition: str,
    language: str,
    query_mode: str,
    plane: tuple[int,int],
    basis_kind: str = "rn",
) -> np.ndarray:
    q = [
        r for r in surface_summary
        if r["model_name"]==model_name
        and r["pump_mode"]==pump_mode
        and r["condition"]==condition
        and r["language"]==language
        and r["query_mode"]==query_mode
        and r["basis_kind"]==basis_kind
        and int(r["pc_a"])==plane[0]
        and int(r["pc_b"])==plane[1]
    ]
    q=sorted(q,key=lambda r:int(r["grid_index"]))
    return np.asarray([safe_float(r["relative_calibrated_mean"]) for r in q],dtype=np.float64)


def surface_cross_model_rows(
    mean_states: dict[tuple,np.ndarray],
    tangent_fields: dict[tuple,np.ndarray],
    surface_summary: list[dict[str,Any]],
    reference_model: str,
    reference_pump: str,
    model_names: Sequence[str],
    pump_modes: Sequence[str],
    languages: Sequence[str],
    query_modes: Sequence[str],
    planes: Sequence[tuple[int,int]],
) -> list[dict[str,Any]]:
    rows=[]
    for model_name in model_names:
        for pump in pump_modes:
            for lang in languages:
                for plane in planes:
                    ref_key=(reference_model,reference_pump,"synth",lang,"rn",plane)
                    key=(model_name,pump,"synth",lang,"rn",plane)
                    if ref_key not in mean_states or key not in mean_states:
                        continue
                    A=mean_states[ref_key]
                    B=mean_states[key]
                    dA=atlas.pairwise_distances(A)
                    dB=atlas.pairwise_distances(B)
                    t1,t2=atlas.mean_tangent_principal_cosines(
                        tangent_fields[ref_key],tangent_fields[key]
                    )
                    an=np.linalg.norm(A,axis=1)
                    bn=np.linalg.norm(B,axis=1)
                    valid=(an*bn)>1e-10
                    if np.any(valid):
                        point_cos=np.sum(A[valid]*B[valid],axis=1)/(an[valid]*bn[valid])
                    else:
                        point_cos=np.asarray([np.nan],dtype=np.float64)
                    base_row={
                        "model_name":model_name,
                        "pump_mode":pump,
                        "language":lang,
                        "pc_a":plane[0],
                        "pc_b":plane[1],
                        "b21_pointwise_cos_mean":float(np.mean(point_cos)),
                        "b21_pointwise_cos_p05":float(np.quantile(point_cos,0.05)),
                        "state_distance_spearman_to_reference":atlas.spearman(
                            atlas.upper_triangle_values(dA),
                            atlas.upper_triangle_values(dB),
                        ),
                        "state_distance_pearson_to_reference":atlas.pearson(
                            atlas.upper_triangle_values(dA),
                            atlas.upper_triangle_values(dB),
                        ),
                        "knn8_to_reference":atlas.knn_overlap(A,B,k=8),
                        "tangent_cos1_to_reference":t1,
                        "tangent_cos2_to_reference":t2,
                        "procrustes_residual_32d_to_reference":atlas.procrustes_residual_lowd(A,B,dim=32),
                    }
                    for qm in query_modes:
                        ra=choose_surface_scalar(
                            surface_summary,reference_model,reference_pump,
                            "synth",lang,qm,plane
                        )
                        rb=choose_surface_scalar(
                            surface_summary,model_name,pump,
                            "synth",lang,qm,plane
                        )
                        row=dict(base_row)
                        row["query_mode"]=qm
                        if len(ra)==len(rb) and len(ra)>0:
                            row["read_surface_pearson_to_reference"]=atlas.pearson(ra,rb)
                            row["read_surface_spearman_to_reference"]=atlas.spearman(ra,rb)
                            row["read_surface_rmse_to_reference"]=float(np.sqrt(np.mean((ra-rb)**2)))
                        rows.append(row)
    return rows


def within_model_cross_language_rows(
    mean_states: dict[tuple,np.ndarray],
    tangent_fields: dict[tuple,np.ndarray],
    surface_summary: list[dict[str,Any]],
    model_names: Sequence[str],
    pump_modes: Sequence[str],
    languages: Sequence[str],
    query_modes: Sequence[str],
    planes: Sequence[tuple[int,int]],
) -> list[dict[str,Any]]:
    rows=[]
    for model_name in model_names:
        for pump in pump_modes:
            for plane in planes:
                for ia,la in enumerate(languages):
                    for lb in languages[ia+1:]:
                        ka=(model_name,pump,"synth",la,"rn",plane)
                        kb=(model_name,pump,"synth",lb,"rn",plane)
                        if ka not in mean_states or kb not in mean_states:
                            continue
                        A=mean_states[ka];B=mean_states[kb]
                        dA=atlas.pairwise_distances(A);dB=atlas.pairwise_distances(B)
                        t1,t2=atlas.mean_tangent_principal_cosines(
                            tangent_fields[ka],tangent_fields[kb]
                        )
                        for qm in query_modes:
                            va=choose_surface_scalar(surface_summary,model_name,pump,"synth",la,qm,plane)
                            vb=choose_surface_scalar(surface_summary,model_name,pump,"synth",lb,qm,plane)
                            rows.append({
                                "model_name":model_name,
                                "pump_mode":pump,
                                "pc_a":plane[0],"pc_b":plane[1],
                                "language_a":la,"language_b":lb,
                                "query_mode":qm,
                                "state_distance_spearman":atlas.spearman(
                                    atlas.upper_triangle_values(dA),
                                    atlas.upper_triangle_values(dB),
                                ),
                                "knn8_jaccard":atlas.knn_overlap(A,B,k=8),
                                "tangent_cos1":t1,
                                "tangent_cos2":t2,
                                "procrustes_residual_32d":atlas.procrustes_residual_lowd(A,B,dim=32),
                                "read_surface_spearman":atlas.spearman(va,vb) if len(va)==len(vb) and len(va) else float("nan"),
                                "read_surface_pearson":atlas.pearson(va,vb) if len(va)==len(vb) and len(va) else float("nan"),
                            })
    return rows


# =============================================================================
# Plotting
# =============================================================================

def plot_text_alignment(rows:list[dict[str,Any]], out:Path)->None:
    good=[r for r in rows if r.get("status")=="ok"]
    if not good:return
    stages=[r["stage"] for r in good]
    x=np.arange(len(stages))
    fig,ax=plt.subplots(figsize=(10,4.8))
    ax.plot(x,[r["paired_cos_mean"] for r in good],marker="o",label="paired cosine")
    ax.plot(x,[r["linear_cka"] for r in good],marker="o",label="linear CKA")
    ax.plot(x,[r["procrustes_test_cos_mean"] for r in good],marker="o",label="held-out cosine after Procrustes")
    ax.set_xticks(x);ax.set_xticklabels(stages,rotation=25,ha="right")
    ax.set_ylim(-0.05,1.05)
    ax.set_ylabel("alignment")
    ax.set_title("Trained vs pretrained text-coordinate alignment")
    ax.legend()
    fig.tight_layout();out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=190,bbox_inches="tight");plt.close(fig)


def plot_basis_principal(rows:list[dict[str,Any]], out:Path)->None:
    groups=defaultdict(list)
    for r in rows:
        groups[(r["model_name"],r["pump_mode"])].append(r)
    fig,ax=plt.subplots(figsize=(10,5))
    for (m,p),q in sorted(groups.items()):
        q=sorted(q,key=lambda r:int(r["pc"]))
        ax.plot([r["pc"] for r in q],[r["subspace_principal_cos"] for r in q],
                marker="o",label=f"{m}/{p}")
    ax.set_xticks([1,2,3,4]);ax.set_ylim(0,1.02)
    ax.set_xlabel("principal direction index")
    ax.set_ylabel("principal cosine to A_native/intact rank-4")
    ax.set_title("RN B13 entry-subspace conservation")
    ax.legend(fontsize=7,ncol=2)
    fig.tight_layout();out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=190,bbox_inches="tight");plt.close(fig)


def plot_surface_alignment(rows:list[dict[str,Any]], out:Path)->None:
    q=[r for r in rows if r["query_mode"]=="native" and int(r["pc_a"])==1 and int(r["pc_b"])==2]
    if not q:return
    cats=[];state=[];read=[]
    for r in q:
        cats.append(f"{r['model_name']}\n{r['pump_mode']}\n{r['language']}")
        state.append(r["state_distance_spearman_to_reference"])
        read.append(r["read_surface_spearman_to_reference"])
    x=np.arange(len(q))
    fig,ax=plt.subplots(figsize=(max(12,len(q)*0.28),5.4))
    ax.plot(x,state,marker="o",label="B21 state-distance geometry")
    ax.plot(x,read,marker="o",label="READ surface")
    ax.axhline(0,linewidth=1)
    ax.set_ylim(-1.05,1.05)
    ax.set_xticks(x);ax.set_xticklabels(cats,rotation=90,fontsize=6)
    ax.set_ylabel("Spearman to A_native/intact")
    ax.set_title("What transfers: state geometry vs text-conditioned readout")
    ax.legend()
    fig.tight_layout();out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=190,bbox_inches="tight");plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main()->None:
    ap=argparse.ArgumentParser(description="PIECES bridge/text/ViT transplant universality atlas")
    ap.add_argument("--checkpoint",default=DEFAULT_CHECKPOINT)
    ap.add_argument("--module-root",default=".")
    ap.add_argument("--pretrained-module",default=DEFAULT_PRETRAINED_MODULE)
    ap.add_argument("--pretrained-spec",default=DEFAULT_PRETRAINED_SPEC)
    ap.add_argument("--dataset-repo",default=DEFAULT_DATASET_REPO)
    ap.add_argument("--dataset-root",default="")
    ap.add_argument("--languages",default=DEFAULT_LANGUAGES)
    ap.add_argument("--query-modes",default=DEFAULT_QUERY_MODES)
    ap.add_argument("--atlas-keys",type=int,default=DEFAULT_ATLAS_KEYS)
    ap.add_argument("--basis-pairs-per-language",type=int,default=DEFAULT_BASIS_PAIRS_PER_LANGUAGE)
    ap.add_argument("--planes",default=DEFAULT_PLANES)
    ap.add_argument("--grid-points",type=int,default=DEFAULT_GRID_POINTS)
    ap.add_argument("--grid-bound",type=float,default=DEFAULT_GRID_BOUND)
    ap.add_argument("--surface-batch",type=int,default=DEFAULT_SURFACE_BATCH)
    ap.add_argument("--ridge-points",type=int,default=DEFAULT_RIDGE_POINTS)
    ap.add_argument("--ridge-bound",type=float,default=DEFAULT_RIDGE_BOUND)
    ap.add_argument("--ridge-radius",type=float,default=DEFAULT_RIDGE_RADIUS)
    ap.add_argument("--ridge-batch",type=int,default=DEFAULT_RIDGE_BATCH)
    ap.add_argument("--ridge-topk",type=int,default=DEFAULT_RIDGE_TOPK)
    ap.add_argument("--ridge-refine-steps",type=int,default=DEFAULT_RIDGE_REFINE_STEPS)
    ap.add_argument("--ridge-refine-random",type=int,default=DEFAULT_RIDGE_REFINE_RANDOM)
    ap.add_argument("--state-jac-eps",type=float,default=DEFAULT_STATE_JAC_EPS)
    ap.add_argument("--text-align-vocab",default=DEFAULT_TEXT_ALIGN_VOCAB)
    ap.add_argument("--text-align-max",type=int,default=DEFAULT_TEXT_ALIGN_MAX)
    ap.add_argument("--text-align-batch",type=int,default=DEFAULT_TEXT_ALIGN_BATCH)
    ap.add_argument("--text-align-holdout",type=float,default=DEFAULT_TEXT_ALIGN_HOLDOUT)
    ap.add_argument("--skip-ridge",action="store_true")
    ap.add_argument("--seed",type=int,default=DEFAULT_SEED)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--output-dir",default="rn_bridge_transplant_universality")
    args=ap.parse_args()

    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)

    languages=parse_strs(args.languages)
    query_modes=parse_strs(args.query_modes)
    planes=parse_planes(args.planes)
    model_names=[x[0] for x in MODEL_SPECS]

    out=Path(args.output_dir)
    data_dir=out/"data";plot_dir=out/"plots";audit_dir=out/"audits";ply_dir=out/"ply"
    for d in (data_dir,plot_dir,audit_dir,ply_dir):d.mkdir(parents=True,exist_ok=True)

    # Dataset once.
    norta,attacks,norta_idx,attack_idx,shared_keys=base.prepare_datasets(
        args.dataset_repo,args.dataset_root,languages,0,args.seed
    )
    if args.atlas_keys>0 and len(shared_keys)>args.atlas_keys:
        rng=random.Random(args.seed)
        atlas_keys=sorted(rng.sample(shared_keys,args.atlas_keys))
    else:
        atlas_keys=list(shared_keys)

    mandatory=[]
    for lang in languages:
        ds=attacks[lang]
        for key in atlas_keys:
            row=dict(ds[attack_idx[lang][key]])
            mandatory += [
                str(row["attack_word_en"]),str(row["object_label_en"]),
                str(row["attack_word"]),str(row["object_label"]),
            ]
    text_bank=load_text_alignment_bank(
        Path(args.text_align_vocab),args.text_align_max,args.seed,mandatory
    )
    (data_dir/"text_alignment_bank.txt").write_text("\n".join(text_bank),encoding="utf-8")

    # Stock source CPU state once.
    pretrained_state,pretrained_meta=load_pretrained_cpu_state(
        args.pretrained_module,args.pretrained_spec
    )
    save_json(audit_dir/"pretrained_source.json",pretrained_meta)

    # ----------------------------------------------------------------------
    # Text alignment arrays: trained text vs pretrained text+trained special rows.
    # ----------------------------------------------------------------------
    text_stage_npz={}
    text_kind_models=(
        ("trained",False),
        ("pretrained",True),
    )
    for text_kind,replace_text in text_kind_models:
        loaded,_audit=load_variant(
            args,pretrained_state,
            f"_textstage_{text_kind}",
            replace_visual=False,
            replace_text=replace_text,
            audit_dir=audit_dir/"text_stage_models",
        )
        stages=collect_text_stages(
            loaded.model,loaded.clip_module,text_bank,args.text_align_batch,loaded.device
        )
        for k,v in stages.items():
            text_stage_npz[f"{text_kind}__{k}"]=v.astype(np.float32)
        del loaded
        cleanup()

    np.savez_compressed(data_dir/"text_stage_arrays.npz",**text_stage_npz)
    trained_stages={k:text_stage_npz[f"trained__{k}"] for k in TEXT_STAGE_ORDER}
    pretrained_stages={k:text_stage_npz[f"pretrained__{k}"] for k in TEXT_STAGE_ORDER}
    text_align_rows=text_alignment_rows(
        trained_stages,pretrained_stages,args.text_align_holdout,args.seed
    )
    save_rows(data_dir/"text_tower_alignment.csv",text_align_rows)
    plot_text_alignment(text_align_rows,plot_dir/"01_text_tower_coordinate_alignment.png")

    # ----------------------------------------------------------------------
    # Reference basis = A_native / intact.
    # ----------------------------------------------------------------------
    print("\n[reference] fitting A_native/intact RN basis")
    ref_loaded,_=load_variant(
        args,pretrained_state,"A_native",
        replace_visual=False,replace_text=False,audit_dir=audit_dir
    )
    ref_model=ref_loaded.model
    reference_donor_custom_digest=_trained_custom_digest(ref_model)
    insert_block=int(ref_model.visual.read_null_insert_block)
    capture_list=[int(x) for x in ref_model.read_implant.capture_block_list()]
    early_needed={b for b in capture_list if b<insert_block}
    surface_capture=set(capture_list)|{STATE_BLOCK}
    target_res=int(ref_model.visual.input_resolution)
    patch=int(ref_model.visual.conv1.kernel_size[0])

    with RegisterPumpContext(ref_model,"intact",verbose=True) as ref_pump_ctx:
        ref_basis_info=base.fit_basis_from_dataset(
            ref_model,ref_loaded.preprocess,norta,attacks,norta_idx,attack_idx,
            shared_keys,languages,args.basis_pairs_per_language,
            early_needed,max_rank=4,device=ref_loaded.device
        )
    reference_basis=ref_basis_info["basis_kd"][:4].detach().float().to(ref_loaded.device)
    random_basis=atlas.random_basis_orthogonal_to(
        reference_basis,4,args.seed+991
    )
    save_rows(audit_dir/"A_native"/"intact_reference_pump_audit.csv",
              ref_pump_ctx.rows("A_native"))
    np.savez_compressed(
        data_dir/"reference_basis.npz",
        reference_basis=reference_basis.cpu().numpy(),
        random_basis=random_basis.cpu().numpy(),
        singular_values=ref_basis_info["singular_values"].cpu().numpy(),
        explained_energy=np.asarray(ref_basis_info["explained_energy"]),
    )
    del ref_loaded,ref_model
    cleanup()

    # ----------------------------------------------------------------------
    # All 8 conditions.
    # ----------------------------------------------------------------------
    basis_rows=[]
    pump_rows=[]
    b13_delta_rows=[]
    register_role_rows=[]
    surface_raw=[]
    ridge_raw=[]
    jac_raw=[]
    state_acc:dict[tuple,SurfaceAcc]={}

    grid_vals,grid_pairs=atlas.grid_alpha_pairs(args.grid_points,args.grid_bound)
    h=float(grid_vals[1]-grid_vals[0])

    for model_name,replace_visual,replace_text,description in MODEL_SPECS:
        for pump_mode in PUMP_MODES:
            print("\n"+"="*84)
            print(f"[condition] {model_name} / {pump_mode}")
            print(f"[meaning] {description}")
            print("="*84)

            loaded,transplant_audit=load_variant(
                args,pretrained_state,model_name,
                replace_visual=replace_visual,
                replace_text=replace_text,
                audit_dir=audit_dir,
                expected_donor_custom_digest=reference_donor_custom_digest,
            )
            model=loaded.model
            device=loaded.device

            query_cache={}
            sem_cache={}
            def get_query(label:str):
                if label not in query_cache:
                    query_cache[label]=base.prepare_read_queries(model,loaded.clip_module,label,device)
                return query_cache[label]
            def get_sem(label:str):
                if label not in sem_cache:
                    sem_cache[label]=base.prompt_bank_embedding(
                        model,loaded.clip_module,label,
                        ["{label}","a photo of {label}","the image depicts {label}",
                         "there is {label}","a picture of {label}"],device
                    )
                return sem_cache[label]

            with RegisterPumpContext(model,pump_mode,verbose=True) as pump_ctx:
                # Local basis fit: universality of actual RN entry geometry.
                local_basis_info=base.fit_basis_from_dataset(
                    model,loaded.preprocess,norta,attacks,norta_idx,attack_idx,
                    shared_keys,languages,args.basis_pairs_per_language,
                    early_needed,max_rank=4,device=device
                )
                basis_rows.extend(
                    basis_audit_rows(model_name,pump_mode,local_basis_info,reference_basis)
                )

                # Main atlas.
                progress=tqdm(atlas_keys,desc=f"{model_name}/{pump_mode}",unit="key")
                for key_index,key in enumerate(progress):
                    lang_rows={lang:dict(attacks[lang][attack_idx[lang][key]]) for lang in languages}

                    # Shared NoRTA visual bundle.
                    nrow=dict(norta[norta_idx[key]])
                    npil=base.ensure_pil(nrow["image"])
                    nimg=base.preprocess_pil(loaded.preprocess,npil,device)
                    npair=base.build_b13_pair(model,nimg,key,"norta",early_needed)
                    n_rn,n_rand=atlas.per_pc_components(
                        npair.delta_ord_btd,reference_basis,random_basis,4
                    )
                    n_anchor=atlas.anchor_run(model,npair,surface_capture)

                    # RN-delta reference capture / register stats.
                    b13_delta_rows.append({
                        "model_name":model_name,"pump_mode":pump_mode,
                        "sample_key":key,"condition":"norta","language":"shared",
                        "reference_rank4_capture":reference_capture_fraction(
                            npair.delta_ord_btd,reference_basis,4
                        ),
                        "delta_norm_mean":float(npair.delta_ord_btd.norm(dim=-1).mean().cpu()),
                    })
                    pre_all=npair.x_pre_tbc.permute(1,0,2).float()
                    pre_sp=pre_all[:,1:,:]
                    norms=pre_sp.norm(dim=-1)
                    reg_mask=npair.b12_register_mask[0].bool()
                    cls_vec=pre_all[0,0]
                    if int(reg_mask.sum())>0:
                        reg_vecs=pre_sp[0,reg_mask]
                        cls_reg_cos=F.cosine_similarity(
                            reg_vecs,cls_vec[None,:].expand_as(reg_vecs),dim=-1
                        )
                        cls_reg_cos_mean=float(cls_reg_cos.mean().cpu())
                        cls_reg_cos_absmean=float(cls_reg_cos.abs().mean().cpu())
                        reg_norm_mean=float(reg_vecs.norm(dim=-1).mean().cpu())
                    else:
                        cls_reg_cos_mean=float("nan")
                        cls_reg_cos_absmean=float("nan")
                        reg_norm_mean=float("nan")
                    register_role_rows.append({
                        "model_name":model_name,"pump_mode":pump_mode,
                        "sample_key":key,"condition":"norta","language":"shared",
                        "b12_register_count":int(npair.b12_register_mask.sum().cpu()),
                        "b12_spatial_norm_mean":float(norms.mean().cpu()),
                        "b12_spatial_norm_max":float(norms.max().cpu()),
                        "b12_register_norm_mean":reg_norm_mean,
                        "b12_cls_to_register_cos_mean":cls_reg_cos_mean,
                        "b12_cls_to_register_cos_absmean":cls_reg_cos_absmean,
                    })

                    n_specs=[]
                    for lang,srow in lang_rows.items():
                        ae=str(srow["attack_word_en"]);oe=str(srow["object_label_en"]);an=str(srow["attack_word"])
                        for qm in query_modes:
                            ql=ae if qm=="english" else an
                            n_specs.append({
                                "language":lang,"query_mode":qm,"query_candidate":ql,
                                "queries":get_query(ql),
                                "attack_sem_en":get_sem(ae),"object_sem_en":get_sem(oe),
                                "attack_word_en":ae,"object_label_en":oe,
                            })

                    for basis_kind,comps in (("rn",n_rn),("random",n_rand)):
                        for plane in planes:
                            rows,states,embs=atlas.evaluate_surface_visual(
                                model,npair,comps,plane,grid_pairs,n_specs,
                                surface_capture,args.surface_batch,n_anchor,
                                {
                                    "model_name":model_name,"pump_mode":pump_mode,
                                    "sample_key":key,"condition":"norta",
                                    "basis_kind":basis_kind,
                                },
                            )
                            surface_raw.extend(rows)
                            add_surface_acc(
                                state_acc,
                                (model_name,pump_mode,"norta","shared",basis_kind,plane),
                                states,embs
                            )

                    # Per-language SynthRTA.
                    for lang in languages:
                        srow=lang_rows[lang]
                        spil=base.ensure_pil(srow["image"])
                        simg=base.preprocess_pil(loaded.preprocess,spil,device)
                        spair=base.build_b13_pair(model,simg,key,"synth",early_needed)
                        srn,srand=atlas.per_pc_components(
                            spair.delta_ord_btd,reference_basis,random_basis,4
                        )
                        s_anchor=atlas.anchor_run(model,spair,surface_capture)

                        ae=str(srow["attack_word_en"]);oe=str(srow["object_label_en"])
                        an=str(srow["attack_word"])
                        specs=[]
                        for qm in query_modes:
                            ql=ae if qm=="english" else an
                            specs.append({
                                "language":lang,"query_mode":qm,"query_candidate":ql,
                                "queries":get_query(ql),
                                "attack_sem_en":get_sem(ae),"object_sem_en":get_sem(oe),
                                "attack_word_en":ae,"object_label_en":oe,
                            })

                        b13_delta_rows.append({
                            "model_name":model_name,"pump_mode":pump_mode,
                            "sample_key":key,"condition":"synth","language":lang,
                            "reference_rank4_capture":reference_capture_fraction(
                                spair.delta_ord_btd,reference_basis,4
                            ),
                            "delta_norm_mean":float(spair.delta_ord_btd.norm(dim=-1).mean().cpu()),
                        })
                        pre_all=spair.x_pre_tbc.permute(1,0,2).float()
                        pre_sp=pre_all[:,1:,:]
                        norms=pre_sp.norm(dim=-1)
                        reg_mask=spair.b12_register_mask[0].bool()
                        cls_vec=pre_all[0,0]
                        if int(reg_mask.sum())>0:
                            reg_vecs=pre_sp[0,reg_mask]
                            cls_reg_cos=F.cosine_similarity(
                                reg_vecs,cls_vec[None,:].expand_as(reg_vecs),dim=-1
                            )
                            cls_reg_cos_mean=float(cls_reg_cos.mean().cpu())
                            cls_reg_cos_absmean=float(cls_reg_cos.abs().mean().cpu())
                            reg_norm_mean=float(reg_vecs.norm(dim=-1).mean().cpu())
                        else:
                            cls_reg_cos_mean=float("nan")
                            cls_reg_cos_absmean=float("nan")
                            reg_norm_mean=float("nan")
                        register_role_rows.append({
                            "model_name":model_name,"pump_mode":pump_mode,
                            "sample_key":key,"condition":"synth","language":lang,
                            "b12_register_count":int(spair.b12_register_mask.sum().cpu()),
                            "b12_spatial_norm_mean":float(norms.mean().cpu()),
                            "b12_spatial_norm_max":float(norms.max().cpu()),
                            "b12_register_norm_mean":reg_norm_mean,
                            "b12_cls_to_register_cos_mean":cls_reg_cos_mean,
                            "b12_cls_to_register_cos_absmean":cls_reg_cos_absmean,
                        })

                        for basis_kind,comps in (("rn",srn),("random",srand)):
                            for plane in planes:
                                rows,states,embs=atlas.evaluate_surface_visual(
                                    model,spair,comps,plane,grid_pairs,specs,
                                    surface_capture,args.surface_batch,s_anchor,
                                    {
                                        "model_name":model_name,"pump_mode":pump_mode,
                                        "sample_key":key,"condition":"synth",
                                        "basis_kind":basis_kind,
                                    },
                                )
                                surface_raw.extend(rows)
                                add_surface_acc(
                                    state_acc,
                                    (model_name,pump_mode,"synth",lang,basis_kind,plane),
                                    states,embs
                                )

                            if not args.skip_ridge:
                                seed=args.seed+100000*model_names.index(model_name)+10000*PUMP_MODES.index(pump_mode)+1000*key_index+137*(languages.index(lang)+1)+(0 if basis_kind=="rn" else 700000)
                                pts=atlas.sobol_points_4d(
                                    args.ridge_points,args.ridge_bound,args.ridge_radius,seed
                                )
                                rr,_states=atlas.evaluate_alpha_cloud(
                                    model,spair,comps,pts,specs,surface_capture,
                                    args.ridge_batch,
                                    {
                                        "model_name":model_name,"pump_mode":pump_mode,
                                        "sample_key":key,"condition":"synth",
                                        "basis_kind":basis_kind,
                                    },
                                )
                                all_rr=list(rr);ridge_raw.extend(rr)
                                for step in range(args.ridge_refine_steps):
                                    scale=args.ridge_bound*(0.30/(2**step))
                                    refine=atlas.refine_points_around_extrema(
                                        all_rr,query_modes,scale,args.ridge_refine_random,
                                        args.ridge_bound,args.ridge_radius,seed+9000+step
                                    )
                                    if len(refine):
                                        new_rr,_=atlas.evaluate_alpha_cloud(
                                            model,spair,comps,refine,specs,surface_capture,
                                            args.ridge_batch,
                                            {
                                                "model_name":model_name,"pump_mode":pump_mode,
                                                "sample_key":key,"condition":"synth",
                                                "basis_kind":basis_kind,
                                            },
                                        )
                                        ridge_raw.extend(new_rr);all_rr.extend(new_rr)

                                centers=[("anchor",np.zeros(4,np.float32),"visual")]
                                for qm in query_modes:
                                    qrows=[r for r in all_rr if r["query_mode"]==qm]
                                    if qrows:
                                        mx=max(qrows,key=lambda r:safe_float(r["relative_calibrated"]))
                                        mn=min(qrows,key=lambda r:safe_float(r["relative_calibrated"]))
                                        for name,rrr in (("max",mx),("min",mn)):
                                            c=np.array([rrr[f"alpha{i}"] for i in range(1,5)],np.float32)
                                            centers.append((name,c,qm))
                                seen=set()
                                for landmark,c,qmlabel in centers:
                                    ck=(landmark,qmlabel,*np.round(c,5).tolist())
                                    if ck in seen:continue
                                    seen.add(ck)
                                    _J,svals=atlas.state_jacobian_at(
                                        model,spair,comps,c,args.state_jac_eps,surface_capture
                                    )
                                    row={
                                        "model_name":model_name,"pump_mode":pump_mode,
                                        "sample_key":key,"condition":"synth","language":lang,
                                        "basis_kind":basis_kind,"landmark":landmark,
                                        "query_mode":qmlabel,
                                        "effective_rank":atlas.effective_rank_from_singular_values(svals),
                                    }
                                    for i in range(4):
                                        row[f"alpha{i+1}"]=float(c[i])
                                        row[f"sigma{i+1}"]=float(svals[i]) if i<len(svals) else 0.0
                                    jac_raw.append(row)

                        del spair,simg,spil,s_anchor
                        cleanup()

                    del npair,nimg,npil,n_anchor
                    cleanup()

            pump_rows.extend(pump_ctx.rows(model_name))

            # Persist each condition's audit immediately.
            save_rows(
                audit_dir/model_name/f"{pump_mode}_register_pump_audit.csv",
                pump_ctx.rows(model_name)
            )

            del model,loaded
            cleanup()

    # ----------------------------------------------------------------------
    # Save raw + summarize.
    # ----------------------------------------------------------------------
    save_rows(data_dir/"basis_audit.csv",basis_rows)
    save_rows(data_dir/"register_pump_audit.csv",pump_rows)
    save_rows(data_dir/"b13_rn_delta_reference_capture.csv",b13_delta_rows)
    save_rows(data_dir/"register_role_stats.csv",register_role_rows)
    save_rows(data_dir/"surface_raw.csv",surface_raw)
    save_rows(data_dir/"ridge_raw.csv",ridge_raw)
    save_rows(data_dir/"state_jacobian_raw.csv",jac_raw)

    surface_summary=base.aggregate_rows(
        surface_raw,
        ["model_name","pump_mode","condition","language","query_mode",
         "basis_kind","pc_a","pc_b","alpha_a","alpha_b","grid_index"],
        [
            "relative_calibrated","candidate_read_null_B20",
            "candidate_read_null_B21","null_read_null_B21","early_ortho",
            "image_attack_en_logit","image_object_en_logit","attack_minus_object_en",
        ]
    )
    save_rows(data_dir/"surface_summary.csv",surface_summary)

    delta_summary=base.aggregate_rows(
        b13_delta_rows,
        ["model_name","pump_mode","condition","language"],
        ["reference_rank4_capture","delta_norm_mean"]
    )
    save_rows(data_dir/"b13_rn_delta_reference_capture_summary.csv",delta_summary)

    reg_summary=base.aggregate_rows(
        register_role_rows,
        ["model_name","pump_mode","condition","language"],
        ["b12_register_count","b12_spatial_norm_mean","b12_spatial_norm_max",
         "b12_register_norm_mean","b12_cls_to_register_cos_mean",
         "b12_cls_to_register_cos_absmean"]
    )
    save_rows(data_dir/"register_role_summary.csv",reg_summary)

    if ridge_raw:
        # atlas helper groups do not know model_name/pump_mode, so add a wrapper grouping.
        ridge_extrema=[]
        ridge_locus=[]
        groups=defaultdict(list)
        for r in ridge_raw:
            groups[(r["model_name"],r["pump_mode"])].append(r)
        for (m,p),rr in groups.items():
            ex,lo=atlas.ridge_extrema_and_loci(rr,args.ridge_topk)
            for x in ex:x.update({"model_name":m,"pump_mode":p})
            for x in lo:x.update({"model_name":m,"pump_mode":p})
            ridge_extrema.extend(ex);ridge_locus.extend(lo)
        save_rows(data_dir/"ridge_extrema.csv",ridge_extrema)
        save_rows(data_dir/"ridge_locus_summary.csv",ridge_locus)
    else:
        ridge_extrema=[];ridge_locus=[]

    jac_summary=base.aggregate_rows(
        jac_raw,
        ["model_name","pump_mode","condition","language","basis_kind","landmark","query_mode"],
        ["effective_rank","sigma1","sigma2","sigma3","sigma4"]
    ) if jac_raw else []
    save_rows(data_dir/"state_jacobian_summary.csv",jac_summary)

    # ----------------------------------------------------------------------
    # Mean B21 state surfaces + local geometry / PLY.
    # ----------------------------------------------------------------------
    mean_states={}
    tangent_fields={}
    local_geom=[]
    pca_rows=[]
    npz_arrays={}

    for key,acc in state_acc.items():
        model_name,pump,condition,lang,basis_kind,plane=key
        mean=(acc.state_sum/acc.count).astype(np.float32)
        mean_states[key]=mean
        G=args.grid_points
        state_grid=mean.reshape(G,G,-1)
        tangent_fields[key]=atlas.tangent_fields_from_surface(state_grid,h)

        # native query coloring for synth; English query on one language's NoRTA.
        qlang=lang if condition=="synth" else languages[0]
        qm="native" if "native" in query_modes else query_modes[0]
        scalar=choose_surface_scalar(
            surface_summary,model_name,pump,condition,qlang,qm,plane,basis_kind
        )
        if len(scalar)!=G*G:
            continue
        scalar_grid=scalar.reshape(G,G)
        geom,_aux=atlas.surface_local_geometry(state_grid,scalar_grid,h)
        for r in geom:
            rr={
                "model_name":model_name,"pump_mode":pump,
                "condition":condition,"language":lang,"basis_kind":basis_kind,
                "pc_a":plane[0],"pc_b":plane[1],
                "alpha_a":float(grid_vals[r["grid_i"]]),
                "alpha_b":float(grid_vals[r["grid_j"]]),
            }
            rr.update(r);local_geom.append(rr)

        coords,frac=atlas.pca3_surface(mean)
        for i,f in enumerate(frac,1):
            pca_rows.append({
                "model_name":model_name,"pump_mode":pump,
                "condition":condition,"language":lang,"basis_kind":basis_kind,
                "pc_a":plane[0],"pc_b":plane[1],
                "component":i,"explained_fraction":float(f),
            })
        arrname=stable_slug(
            f"{model_name}__{pump}__{condition}__{lang}__{basis_kind}__PC{plane[0]}xPC{plane[1]}"
        )
        npz_arrays[arrname]=mean
        if condition=="synth" and basis_kind=="rn":
            atlas.write_ply_surface(
                ply_dir/f"{arrname}__B21_PCA3.ply",
                coords,G,grid_pairs,scalar
            )

    np.savez_compressed(data_dir/"mean_b21_surfaces.npz",**npz_arrays)
    save_rows(data_dir/"surface_local_geometry.csv",local_geom)
    save_rows(data_dir/"surface_pca_spectrum.csv",pca_rows)

    cross_model=surface_cross_model_rows(
        mean_states,tangent_fields,surface_summary,
        "A_native","intact",model_names,PUMP_MODES,languages,query_modes,planes
    )
    save_rows(data_dir/"cross_model_surface_alignment.csv",cross_model)

    cross_lang=within_model_cross_language_rows(
        mean_states,tangent_fields,surface_summary,
        model_names,PUMP_MODES,languages,query_modes,planes
    )
    save_rows(data_dir/"cross_language_geometry.csv",cross_lang)

    # ----------------------------------------------------------------------
    # Selected plots.
    # ----------------------------------------------------------------------
    plot_basis_principal(basis_rows,plot_dir/"02_b13_entry_subspace_conservation.png")
    plot_surface_alignment(cross_model,plot_dir/"03_state_geometry_vs_readout_transfer.png")

    # ----------------------------------------------------------------------
    # Config / summary / compact handoff.
    # ----------------------------------------------------------------------
    config={
        "checkpoint":args.checkpoint,
        "pretrained_module":args.pretrained_module,
        "pretrained_spec":args.pretrained_spec,
        "dataset_repo":args.dataset_repo,
        "languages":languages,
        "query_modes":query_modes,
        "atlas_keys":atlas_keys,
        "planes":planes,
        "grid_points":args.grid_points,
        "grid_bound":args.grid_bound,
        "ridge_points":0 if args.skip_ridge else args.ridge_points,
        "ridge_bound":args.ridge_bound,
        "ridge_radius":args.ridge_radius,
        "text_alignment_bank_size":len(text_bank),
        "text_alignment_holdout":args.text_align_holdout,
        "openai_vocab_size_assumption":OPENAI_VOCAB_SIZE,
        "register_pump_units":REG_NEURONS,
        "model_conditions":[
            {
                "name":m,"replace_visual_with_pretrained":v,
                "replace_text_with_pretrained":t,"description":d,
            }
            for m,v,t,d in MODEL_SPECS
        ],
        "pump_modes":list(PUMP_MODES),
        "protected_trained_state":[
            "read_implant.*","visual.read_null*","PIECES root custom state",
            "special token_embedding rows >= 49408","logit_scale",
        ],
        "surface_coordinates":"A_native/intact RN PC1..PC4 reference basis for ALL variants",
    }
    save_json(data_dir/"config.json",config)

    summary=[
        "PIECES BRIDGE / TEXT / ViT TRANSPLANT UNIVERSALITY",
        "="*76,"",
        "Conditions:",
    ]
    for m,v,t,d in MODEL_SPECS:
        summary.append(f"  {m}: {d}")
    summary += [
        "",
        "Every condition runs intact + exact post-QuickGELU B11/B12 no_pump.",
        f"Register units: {REG_NEURONS}",
        "",
        "Text-tower alignment:",
    ]
    for r in text_align_rows:
        if r.get("status")=="ok":
            summary.append(
                f"  {r['stage']}: cos={r['paired_cos_mean']:.5f}, "
                f"CKA={r['linear_cka']:.5f}, "
                f"Procrustes heldout cos={r['procrustes_test_cos_mean']:.5f}, "
                f"resid={r['procrustes_test_relative_residual']:.5f}"
            )
    summary += ["","B13 reference-subspace conservation (principal cosines):"]
    for (m,p) in [(m,p) for m in model_names for p in PUMP_MODES]:
        q=sorted([r for r in basis_rows if r["model_name"]==m and r["pump_mode"]==p],
                 key=lambda r:r["pc"])
        if q:
            summary.append(
                f"  {m}/{p}: "+", ".join(f"{r['subspace_principal_cos']:.4f}" for r in q)
            )

    summary += ["","Cross-model RN PC1xPC2 native-query surface alignment to A_native/intact:"]
    for m in model_names:
        for p in PUMP_MODES:
            q=[
                r for r in cross_model
                if r["model_name"]==m and r["pump_mode"]==p
                and r["query_mode"]=="native"
                and int(r["pc_a"])==1 and int(r["pc_b"])==2
            ]
            if q:
                summary.append(
                    f"  {m}/{p}: state-distance rho={np.mean([r['state_distance_spearman_to_reference'] for r in q]):.4f}, "
                    f"READ rho={np.mean([r['read_surface_spearman_to_reference'] for r in q]):.4f}, "
                    f"tangent cos2={np.mean([r['tangent_cos2_to_reference'] for r in q]):.4f}"
                )
    summary += [
        "",
        "Interpretation guardrail:",
        "  Same RN token / same B13 entry address does NOT imply identical downstream",
        "  trajectory or identical text-conditioned READ field.  The outputs are split",
        "  explicitly so 'universality' can be stated at the correct level.",
        "",
    ]
    (out/"SUMMARY.txt").write_text("\n".join(summary),encoding="utf-8")

    include=[
        data_dir/"config.json",
        data_dir/"text_tower_alignment.csv",
        data_dir/"basis_audit.csv",
        data_dir/"register_pump_audit.csv",
        data_dir/"b13_rn_delta_reference_capture_summary.csv",
        data_dir/"register_role_summary.csv",
        data_dir/"surface_summary.csv",
        data_dir/"surface_local_geometry.csv",
        data_dir/"surface_pca_spectrum.csv",
        data_dir/"cross_model_surface_alignment.csv",
        data_dir/"cross_language_geometry.csv",
        data_dir/"state_jacobian_summary.csv",
        data_dir/"ridge_extrema.csv",
        data_dir/"ridge_locus_summary.csv",
        data_dir/"reference_basis.npz",
        out/"SUMMARY.txt",
    ]
    include += sorted(plot_dir.glob("*.png"))
    # PLY can be many but are compact and useful for literal surface inspection.
    include += sorted(ply_dir.glob("*.ply"))

    zpath=out/"compact_summary_rn_bridge_transplant.zip"
    if zpath.exists():zpath.unlink()
    with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in include:
            if p.exists() and p.is_file():
                z.write(p,arcname=p.relative_to(out).as_posix())

    print("\n[done]")
    print("  data:       ",data_dir.resolve())
    print("  audits:     ",audit_dir.resolve())
    print("  plots:      ",plot_dir.resolve())
    print("  PLY:        ",ply_dir.resolve())
    print("  compact summary:",zpath.resolve())




if __name__ == "__main__":
    main()
