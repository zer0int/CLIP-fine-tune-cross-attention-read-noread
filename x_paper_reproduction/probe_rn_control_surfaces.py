#!/usr/bin/env python3
r"""Native multilingual RN control surfaces with register-pump and block controls.
Commands: analyze (surface extraction), flow_maps (reuse saved surfaces; gradient maps and PLY).
External analysis dependencies: probe_tools_rn_control and probe_tools_rn_manifold.
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_analysis import (ensure_dir, parse_strs)

# ANALYZE
import argparse
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
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm



# =============================================================================
# Defaults
# =============================================================================

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
DEFAULT_SURFACE_BATCH = 96
DEFAULT_RIDGE_POINTS = 768
DEFAULT_SUPPORT_PCS = 4
DEFAULT_SUPPORT_QUANTILE = 0.95
DEFAULT_SUPPORT_DRAW_MULTIPLIER = 32
DEFAULT_SUPPORT_MIN_LOCAL_SPAN = 0.05
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
)

# IMPORTANT: the trailing comma is required. ("intact") is just a string and
# iterates as "i", "n", "t", ...
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

# Extraction dependencies are initialized by analyze_main.
base = atlas = None
DEFAULT_CHECKPOINT = None

def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    base.save_rows(path, rows)


def save_json(path: Path, payload: Any) -> None:
    base.save_json(path, payload)


def safe_float(x: Any) -> float:
    return base.safe_float(x)


def analyze_stable_slug(x: str) -> str:
    return base.stable_slug(x)


def analyze_parse_planes(text: str) -> list[tuple[int, int]]:
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


def load_variant(
    args: argparse.Namespace,
    pretrained_state: Mapping[str, torch.Tensor],
    model_name: str,
    replace_visual: bool,
    replace_text: bool,
    audit_dir: Path,
):
    loaded = base.load_model(
        args.checkpoint,
        package_root=args.module_root,
        device=args.device,
    )
    model = loaded.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    audit = apply_pretrained_transplant(
        model,
        pretrained_state,
        replace_visual=replace_visual,
        replace_text=replace_text,
        audit_path=audit_dir / model_name / "transplant_audit",
    )
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
# Native-only fast paper pass
# =============================================================================


def load_native_model(args: argparse.Namespace):
    """Load the actual trained checkpoint once; no tower transplantation."""
    loaded = base.load_model(
        args.checkpoint,
        package_root=args.module_root,
        device=args.device,
    )
    model = loaded.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return loaded


def _paper_cleanup(*, empty_cache: bool = False) -> None:
    """Cheap cleanup by default; CUDA cache purging is intentionally rare."""
    gc.collect()
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_surface_visual_adaptive(
    model: torch.nn.Module,
    pair: Any,
    components4: Sequence[torch.Tensor],
    plane: tuple[int, int],
    grid_pairs: np.ndarray,
    query_specs: list[dict[str, Any]],
    capture: set[int],
    batch_size: int,
    anchor: dict[str, Any],
    common_visual_meta: dict[str, Any],
    *,
    min_batch: int = 12,
):
    """
    Same atlas math, with CUDA-OOM backoff only around batching.

    The surface coordinates/forward/scoring are unchanged. If a larger batch does
    not fit, the whole surface is retried with half the batch size.
    """
    bs = max(1, int(batch_size))
    floor = max(1, int(min_batch))
    while True:
        try:
            return atlas.evaluate_surface_visual(
                model, pair, components4, plane, grid_pairs, query_specs,
                capture, bs, anchor, common_visual_meta,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = str(exc).lower()
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg
            if not is_oom or bs <= floor:
                raise
            new_bs = max(floor, bs // 2)
            if new_bs == bs:
                raise
            print(f"[surface batch] CUDA OOM at batch={bs}; retrying whole surface at batch={new_bs}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bs = new_bs


def _basis_options(
    rn_components: Sequence[torch.Tensor],
    random_components: Sequence[torch.Tensor],
    include_random: bool,
):
    out = [("rn", rn_components)]
    if include_random:
        out.append(("random", random_components))
    return out


@dataclass
class EmpiricalSupportModel:
    # Natural operating envelope in the four per-PC RMS amplitudes.  Candidate
    # interventions preserve each image's tokenwise PC field shape exactly and
    # only scale the four PC components, so amplitude-space is the appropriate
    # low-dimensional support coordinate.
    mean: np.ndarray                    # [4] population mean PC RMS amplitude
    cov_inv: np.ndarray                 # [4,4] regularized inverse covariance
    pca_v: np.ndarray                   # [4,4] covariance eigenvectors (plot/audit)
    pca_scale: np.ndarray               # [4] sqrt covariance eigenvalues
    empirical_md: np.ndarray            # [N] leave-one-out natural-scale distances
    empirical_residual_rms: np.ndarray  # compatibility/audit; zeros in v3
    explained_fraction: np.ndarray      # [4] covariance PCA fractions
    md_q95: float
    md_q99: float
    residual_q95: float
    residual_q99: float


def _rn_coeff_kt(delta_btd: torch.Tensor, basis_kd: torch.Tensor) -> np.ndarray:
    """Tokenwise RN coefficients in the fixed feature basis, returned [K,T]."""
    d = delta_btd[0].detach().float()
    b = basis_kd.detach().float().to(d.device)
    coeff_tk = d @ b.T
    return coeff_tk.T.cpu().numpy().astype(np.float32)


def _coeff_descriptor(coeff_kt: np.ndarray) -> np.ndarray:
    """PC-major flattened tokenwise coefficient field [K*T]."""
    return np.asarray(coeff_kt, np.float32).reshape(-1)


def fit_reference_basis_and_support_records(
    model: torch.nn.Module,
    preprocess: Any,
    norta: Any,
    attacks: dict[str, Any],
    norta_idx: dict[str, int],
    attack_idx: dict[str, dict[str, int]],
    keys: list[str],
    languages: list[str],
    pairs_per_language: int,
    needed_early: set[int],
    max_rank: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Same rank-basis fit as base.fit_basis_from_dataset(), but retain the per-image
    un-intervened RN deltas on CPU so empirical support costs no second visual pass.
    """
    if pairs_per_language <= 0:
        basis_keys = keys
    else:
        basis_keys = keys[: min(len(keys), pairs_per_language)]

    rows: list[torch.Tensor] = []
    templates: list[torch.Tensor] = []
    used: list[str] = []
    retained: list[dict[str, Any]] = []

    def add_one(key: str, condition: str, language: str, image_row: Any) -> None:
        pil = base.ensure_pil(image_row["image"])
        img = base.preprocess_pil(preprocess, pil, device)
        pair = base.build_b13_pair(model, img, key, condition, needed_early)
        d = pair.delta_ord_btd[0].float()
        rows.append(d)
        templates.append(d)
        used.append(f"{language}::{key}")
        retained.append({
            "sample_key": key,
            "condition": condition,
            "language": language,
            "delta_td": d.detach().cpu(),
        })
        del pair, img, pil

    for key in basis_keys:
        add_one(key, "norta", "shared", base.get_row(norta, norta_idx[key]))

    for lang in languages:
        for key in basis_keys:
            add_one(key, "synth", lang, base.get_row(attacks[lang], attack_idx[lang][key]))

    x = torch.cat(rows, dim=0)
    q = min(max(int(max_rank) + 8, 12), x.shape[0], x.shape[1])
    _, sing, vec = torch.pca_lowrank(x, q=q, center=False, niter=6)
    basis = vec[:, :max_rank].T.contiguous().float()
    total = float(x.square().sum().detach().cpu())
    explained = (sing[:max_rank].square() / max(total, 1e-12)).detach().cpu().numpy()
    mean_template = torch.stack(templates, dim=0).mean(dim=0).float()

    result = {
        "basis_kd": basis.detach(),
        "singular_values": sing[:max_rank].detach(),
        "explained_energy": explained,
        "mean_template_td": mean_template.detach(),
        "fit_items": used,
        "n_rows": int(x.shape[0]),
    }

    # Project retained natural deltas only after the basis has been fitted.
    for rec in retained:
        rec["coeff_kt"] = _rn_coeff_kt(rec.pop("delta_td")[None], basis.cpu())

    del rows, templates, x
    return result, retained


def fit_empirical_support_model(
    support_records: list[dict[str, Any]],
    pca_k: int,
) -> tuple[EmpiricalSupportModel, list[dict[str, Any]]]:
    """
    Natural-variation scale for causal 4-D RN interventions.

    Each intervention keeps the target image's tokenwise PC coefficient field
    fixed and only multiplies PC1..PC4 by alpha.  Therefore the only new degrees
    of freedom are the four component amplitudes.  Modeling the original
    K*T flattened field made held-out *natural* images look out-of-support because
    exact spatial coefficient patterns are image-specific.  v3 instead fits the
    4-D vector of per-PC RMS amplitudes and calibrates distance thresholds with
    leave-one-out natural samples.

    Search support is LOCAL around the actual image's alpha=(1,1,1,1) point: a
    candidate is allowed when its amplitude displacement is no larger (in the
    empirical covariance metric) than ordinary natural cross-image variation.
    This guarantees that the observed natural point is always distance zero while
    still preventing extrapolative steering.
    """
    if len(support_records) < 8:
        raise RuntimeError(f"Need >=8 empirical RN samples, got {len(support_records)}")

    X = np.stack([
        np.sqrt(np.mean(np.asarray(r["coeff_kt"], np.float64) ** 2, axis=1))
        for r in support_records
    ]).astype(np.float64)  # [N,4]

    def fit_cov(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu = Y.mean(axis=0)
        cov = np.cov(Y, rowvar=False, ddof=1)
        cov = np.atleast_2d(cov).astype(np.float64)
        # Mild isotropic shrinkage: with D=4,N~72 this is only a numerical guard,
        # not a dominant prior.
        ridge = max(0.05 * float(np.trace(cov)) / max(cov.shape[0], 1), 1e-10)
        cov_reg = cov + ridge * np.eye(cov.shape[0], dtype=np.float64)
        inv = np.linalg.inv(cov_reg)
        evals, evecs = np.linalg.eigh(cov_reg)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 1e-20)
        evecs = evecs[:, order]
        return mu, inv, evals, evecs

    mean, cov_inv, evals, evecs = fit_cov(X)

    # Out-of-sample calibration: each natural example is scored against a model
    # that did not contain that example.  This avoids the v2 failure where PCA
    # reconstruction residual thresholds were training-set optimistic.
    loo_md = []
    for i in range(len(X)):
        Y = np.delete(X, i, axis=0)
        mu_i, inv_i, _ev_i, _vec_i = fit_cov(Y)
        d = X[i] - mu_i
        loo_md.append(math.sqrt(max(float(d @ inv_i @ d), 0.0)))
    loo_md = np.asarray(loo_md, dtype=np.float64)

    frac = evals / max(float(evals.sum()), 1e-20)
    model = EmpiricalSupportModel(
        mean=mean.astype(np.float32),
        cov_inv=cov_inv.astype(np.float32),
        pca_v=evecs.T.astype(np.float32),
        pca_scale=np.sqrt(evals).astype(np.float32),
        empirical_md=loo_md,
        empirical_residual_rms=np.zeros(len(X), dtype=np.float64),
        explained_fraction=frac.astype(np.float64),
        md_q95=float(np.quantile(loo_md, 0.95)),
        md_q99=float(np.quantile(loo_md, 0.99)),
        residual_q95=0.0,
        residual_q99=0.0,
    )

    md_sorted = np.sort(model.empirical_md)
    out_rows = []
    for i, rec in enumerate(support_records):
        c = np.asarray(rec["coeff_kt"], np.float64)
        comp_rms = np.sqrt(np.mean(c * c, axis=1))
        total_rms = float(np.sqrt(np.mean(c * c)))
        md_pct = float(np.searchsorted(md_sorted, loo_md[i], side="right") / len(md_sorted))
        row = {
            "sample_key": rec["sample_key"],
            "condition": rec["condition"],
            "language": rec["language"],
            "support_mahalanobis": float(loo_md[i]),
            "support_md_percentile": md_pct,
            "support_residual_rms": 0.0,
            "support_residual_percentile": 0.0,
            "support_joint_percentile": md_pct,
            "rank4_coeff_rms": total_rms,
        }
        for j in range(c.shape[0]):
            row[f"pc{j+1}_coeff_rms"] = float(comp_rms[j])
        out_rows.append(row)
    return model, out_rows


def save_empirical_support_model(path: Path, model: EmpiricalSupportModel) -> None:
    np.savez_compressed(
        path,
        mean=model.mean,
        cov_inv=model.cov_inv,
        pca_v=model.pca_v,
        pca_scale=model.pca_scale,
        empirical_md=model.empirical_md,
        empirical_residual_rms=model.empirical_residual_rms,
        explained_fraction=model.explained_fraction,
        md_q95=np.asarray(model.md_q95),
        md_q99=np.asarray(model.md_q99),
        residual_q95=np.asarray(model.residual_q95),
        residual_q99=np.asarray(model.residual_q99),
    )


def _support_linear_terms(
    support: EmpiricalSupportModel,
    coeff_kt: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Precompute activation-distance geometry for one image."""
    coeff = np.asarray(coeff_kt, np.float64)
    K, T = coeff.shape
    D = K * T
    A = np.zeros((K, D), dtype=np.float64)
    for j in range(K):
        A[j, j*T:(j+1)*T] = coeff[j]
    Gfull = A @ A.T
    natural = np.ones(K, dtype=np.float64)
    natural_norm2 = float(natural @ Gfull @ natural)
    natural_amp = np.sqrt(np.mean(coeff * coeff, axis=1))
    return {
        "Gfull": Gfull,
        "natural_norm2": natural_norm2,
        "natural_amp": natural_amp,
    }


def support_metrics_for_alphas(
    support: EmpiricalSupportModel,
    coeff_kt: np.ndarray,
    alphas: np.ndarray,
) -> dict[str, np.ndarray]:
    a = np.asarray(alphas, np.float64).reshape(-1, 4)
    terms = _support_linear_terms(support, coeff_kt)
    natural_amp = np.asarray(terms["natural_amp"], np.float64)

    # Candidate amplitude vector.  Signed alpha is deliberate: a global sign
    # reversal of one learned PC is not treated as natural merely because its RMS
    # magnitude matches.
    amp = a * natural_amp[None, :]
    global_diff = amp - support.mean.astype(np.float64)[None, :]
    local_diff = amp - natural_amp[None, :]
    inv = support.cov_inv.astype(np.float64)
    global_md = np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", global_diff, inv, global_diff), 0.0))
    local_md = np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", local_diff, inv, local_diff), 0.0))

    md_sorted = np.sort(support.empirical_md)
    md_pct = np.searchsorted(md_sorted, local_md, side="right") / len(md_sorted)
    sign_ok = np.all(a > 0.0, axis=1)

    diff = a - 1.0
    Gfull = terms["Gfull"]
    d2 = np.einsum("bi,ij,bj->b", diff, Gfull, diff)
    rel_natural = np.sqrt(np.maximum(d2, 0.0) / max(float(terms["natural_norm2"]), 1e-20))

    return {
        # Backward-compatible headline field now means LOCAL natural-scale MD.
        "support_mahalanobis": local_md,
        "support_global_mahalanobis": global_md,
        "support_md_percentile": md_pct,
        "support_residual_rms": np.zeros(len(a), dtype=np.float64),
        "support_residual_percentile": np.zeros(len(a), dtype=np.float64),
        "support_joint_percentile": md_pct,
        "support_natural_sign_ok": sign_ok,
        "inside_support_95": (local_md <= support.md_q95) & sign_ok,
        "inside_support_99": (local_md <= support.md_q99) & sign_ok,
        "distance_to_natural_rank4": rel_natural,
        "alpha_l2_to_natural": np.linalg.norm(diff, axis=1),
        "alpha_negative_count": np.sum(a < 0.0, axis=1),
    }


def _append_support_metrics(rows: list[dict[str, Any]], metrics: dict[str, np.ndarray]) -> None:
    """Rows from evaluate_alpha_cloud repeat point_index for each query spec."""
    for r in rows:
        i = int(r["point_index"])
        for k, arr in metrics.items():
            v = arr[i]
            r[k] = bool(v) if np.issubdtype(np.asarray(arr).dtype, np.bool_) else float(v)


def _support_quantile_mask(
    support: EmpiricalSupportModel,
    metrics: dict[str, np.ndarray],
    quantile: float,
) -> np.ndarray:
    q=float(np.clip(quantile,0.0,1.0))
    md_thr=float(np.quantile(support.empirical_md,q))
    sign_ok=np.asarray(metrics.get("support_natural_sign_ok", np.ones_like(metrics["support_mahalanobis"],dtype=bool)),dtype=bool)
    return (metrics["support_mahalanobis"] <= md_thr) & sign_ok


def support_constrained_sobol_points(
    support: EmpiricalSupportModel,
    coeff_kt: np.ndarray,
    n: int,
    bound: float,
    radius: float,
    seed: int,
    quantile: float,
    draw_multiplier: int,
    min_local_span: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Draw a broad Sobol cloud, but retain only points whose *actual reconstructed
    tokenwise coefficient field* lies inside the empirical RN operating envelope.

    If broad sampling is too sparse, progressively concentrate around alpha=1,
    the image's natural rank-4 RN point.  No extra model forwards are spent on
    rejected points.
    """
    n = max(1, int(n))
    target_q = float(np.clip(quantile, 0.50, 1.0))
    keep_points: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()

    def accept(cand: np.ndarray) -> None:
        if len(cand) == 0:
            return
        m = support_metrics_for_alphas(support, coeff_kt, cand)
        mask = _support_quantile_mask(support,m,target_q)
        for x in cand[mask]:
            key = tuple(np.round(x, 6).tolist())
            if key not in seen:
                seen.add(key); keep_points.append(x.astype(np.float32))

    # Always consider the true rank-4 operating point first.
    accept(np.ones((1, 4), dtype=np.float32))

    broad_n = max(n * max(4, int(draw_multiplier)), 4096)
    accept(atlas.sobol_points_4d(broad_n, bound, radius, seed))

    # Adaptive local fill around natural alpha=1 if empirical support is narrow.
    span = min(float(bound), 1.5)
    round_id = 0
    while len(keep_points) < n and span >= float(min_local_span) - 1e-12:
        eng = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed + 1009 + round_id)
        draw = eng.draw(max(n * 8, 2048)).cpu().numpy().astype(np.float32)
        cand = 1.0 + (draw * 2.0 - 1.0) * span
        cand = atlas.clip_alpha_points(cand, bound, radius)
        accept(cand)
        span *= 0.65
        round_id += 1

    if not keep_points:
        # This can only happen if the sample's own natural point is outside the
        # requested population quantile. Preserve it as an explicit empirical outlier.
        keep_points = [np.ones(4, dtype=np.float32)]

    pts = np.stack(keep_points[:n]).astype(np.float32)
    metrics = support_metrics_for_alphas(support, coeff_kt, pts)
    return pts, metrics


def filter_points_to_empirical_support(
    support: EmpiricalSupportModel,
    coeff_kt: np.ndarray,
    points: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if len(points) == 0:
        return np.empty((0, 4), np.float32), support_metrics_for_alphas(
            support, coeff_kt, np.empty((0, 4), np.float32)
        )
    metrics = support_metrics_for_alphas(support, coeff_kt, points)
    mask = _support_quantile_mask(support,metrics,float(quantile))
    pts = np.asarray(points, np.float32)[mask]
    kept = {k: np.asarray(v)[mask] for k, v in metrics.items()}
    return pts, kept


def annotate_extrema_support(
    extrema: list[dict[str, Any]],
    coeff_lookup: dict[tuple[str, str], np.ndarray],
    support: EmpiricalSupportModel,
) -> None:
    for r in extrema:
        key = (str(r["sample_key"]), str(r["language"]))
        coeff = coeff_lookup.get(key)
        if coeff is None:
            continue
        a = np.asarray([[safe_float(r[f"alpha{i}"]) for i in range(1, 5)]], np.float32)
        m = support_metrics_for_alphas(support, coeff, a)
        for name, arr in m.items():
            v = arr[0]
            r[name] = bool(v) if np.issubdtype(np.asarray(arr).dtype, np.bool_) else float(v)


def plot_empirical_support_ridge_summary(
    support_rows: list[dict[str, Any]],
    extrema: list[dict[str, Any]],
    jac_rows: list[dict[str, Any]],
    support: EmpiricalSupportModel,
    out: Path,
) -> None:
    """Compact causal-support figure; descriptive only, all source tables are saved."""
    if not extrema:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))

    ax = axes[0, 0]
    cum = np.cumsum(support.explained_fraction)
    ax.plot(np.arange(1, len(cum)+1), cum, marker="o")
    ax.set_ylim(0, 1.02); ax.set_xlabel("empirical support PCA component")
    ax.set_ylabel("cumulative explained fraction")
    ax.set_title("A. Natural RN coefficient-field support")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    q = [r for r in extrema if r.get("basis_kind") == "rn" and r.get("query_mode") == "native"]
    for mode, marker in (("max", "^"), ("min", "v")):
        z = [r for r in q if r.get("extremum") == mode]
        if z:
            ax.scatter(
                [100.0*safe_float(r.get("support_joint_percentile")) for r in z],
                [safe_float(r.get("relative_calibrated")) for r in z],
                marker=marker, label=mode,
            )
    ax.axvline(95, linestyle="--", linewidth=1); ax.axvline(99, linestyle=":", linewidth=1)
    ax.set_xlabel("empirical support percentile")
    ax.set_ylabel("relative calibrated READ")
    ax.set_title("B. Support-constrained READ extrema")
    ax.legend(); ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    langs = sorted({str(r.get("language")) for r in q})
    x = np.arange(len(langs))
    for mode, marker in (("max", "^"), ("min", "v")):
        vals=[]
        for lang in langs:
            z=[r for r in q if str(r.get("language"))==lang and r.get("extremum")==mode]
            vals.append(np.nanmean([safe_float(r.get("distance_to_natural_rank4")) for r in z]) if z else np.nan)
        ax.plot(x, vals, marker=marker, label=mode)
    ax.set_xticks(x); ax.set_xticklabels(langs)
    ax.set_ylabel("activation distance / natural rank-4 norm")
    ax.set_title("C. How far extrema move from normal RN operation")
    ax.legend(); ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    jq=[r for r in jac_rows if r.get("basis_kind")=="rn" and r.get("query_mode") in {"native","visual"}]
    labels=[]; vals=[]
    for landmark in ("natural_rank4", "max", "min", "anchor"):
        z=[r for r in jq if r.get("landmark")==landmark]
        if z:
            labels.append(landmark); vals.append(np.nanmean([safe_float(r.get("effective_rank")) for r in z]))
    if vals:
        ax.bar(np.arange(len(vals)), vals)
        ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("B21 local Jacobian effective rank")
    ax.set_title("D. Local downstream dimensionality")
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("RN causal control within empirical operating support", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Multi-block state-sheet helpers (v5)
# =============================================================================

SCRIPT_GROUP = {
    "en": "Latin", "de": "Latin", "es": "Latin", "fr": "Latin", "it": "Latin", "pt": "Latin",
    "ru": "Cyrillic", "uk": "Cyrillic", "bg": "Cyrillic", "sr": "Cyrillic",
    "ar": "Arabic", "fa": "Arabic", "ur": "Arabic",
    "ko": "Hangul", "zh": "Hanzi", "ja": "KanaKanji",
}


def parse_block_spec(text: str) -> list[int]:
    """Parse '13-23', '13,14,20-23', etc."""
    out: list[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a = int(a); b = int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(tok))
    return sorted(set(out))


def _cli_supplied(flag: str) -> bool:
    return any(x == flag or x.startswith(flag + "=") for x in sys.argv[1:])


def state_vector_from_tbc(x_tbc: torch.Tensor, has_rn: bool = True) -> torch.Tensor:
    """
    Query-independent visual sheet state: mean ordinary spatial-patch residual.
    Excludes CLS and, when present, the RN token.
    """
    btd = x_tbc.permute(1, 0, 2)
    ordinary = base.aligned_ordinary_state(btd, has_rn).float()
    return ordinary[:, 1:, :].mean(dim=1).detach()


def run_downstream_multiblock(
    model: torch.nn.Module,
    post_b13_tbc: torch.Tensor,
    early_states: dict[int, torch.Tensor],
    bridge_capture_blocks: set[int],
    analysis_blocks: Sequence[int],
    has_rn: bool,
) -> dict[str, Any]:
    """
    Run B13->final once.

    Full token tensors are retained only for actual bridge capture blocks.
    For analysis-only blocks, retain only the query-independent spatial-patch
    mean [B,D]. This avoids cloning full [B,T,D] states at B13..B23 merely to
    screen manifold geometry.
    """
    visual = model.visual
    ib = int(visual.read_null_insert_block)
    batch = int(post_b13_tbc.shape[1])
    states = base.repeat_early_states(early_states, batch)
    analysis_set = set(map(int, analysis_blocks))
    analysis_states: dict[int, torch.Tensor] = {}

    x = post_b13_tbc
    if ib in bridge_capture_blocks:
        states[ib] = x.permute(1, 0, 2).detach().clone()
    if ib in analysis_set:
        analysis_states[ib] = state_vector_from_tbc(x, has_rn=has_rn)

    with torch.no_grad(), base.model_autocast_context(model):
        for i in range(ib + 1, len(visual.transformer.resblocks)):
            x = visual.transformer.resblocks[i](x)
            if i in bridge_capture_blocks:
                states[i] = x.permute(1, 0, 2).detach().clone()
            if i in analysis_set:
                analysis_states[i] = state_vector_from_tbc(x, has_rn=has_rn)
        embedding = visual._finalize_cls(x).float().detach()

    final = x.permute(1, 0, 2).detach()
    spatial = final[:, 1:-1, :] if has_rn else final[:, 1:, :]
    norms = spatial.float().norm(dim=-1)
    reg = visual._make_implicit_register_mask(
        norms.detach(),
        register_threshold=70.0,
        max_registers=8,
        min_registers=1,
    ).to(device=final.device)

    return {
        "states": states,
        "analysis_states": analysis_states,
        "embedding": embedding,
        "final_tokens": final,
        "register_mask": reg,
        "has_rn": has_rn,
    }


def anchor_run_multiblock(
    model: torch.nn.Module,
    pair: Any,
    bridge_capture_blocks: set[int],
    analysis_blocks: Sequence[int],
) -> dict[str, Any]:
    zeros = np.zeros((1, 1), dtype=np.float32)
    post = atlas.make_alpha_post_batch(pair, [torch.zeros_like(pair.delta_ord_btd)], zeros)
    return run_downstream_multiblock(
        model, post, pair.early_states, bridge_capture_blocks, analysis_blocks, has_rn=True
    )


def evaluate_surface_visual_multiblock(
    model: torch.nn.Module,
    pair: Any,
    components4: Sequence[torch.Tensor],
    plane: tuple[int, int],
    grid_pairs: np.ndarray,
    query_specs: list[dict[str, Any]],
    bridge_capture_blocks: set[int],
    analysis_blocks: Sequence[int],
    batch_size: int,
    anchor: dict[str, Any],
    common_visual_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], np.ndarray]:
    """
    Evaluate one visual intervention sheet once while collecting compact state
    vectors at all requested analysis blocks.
    """
    i0, j0 = plane[0] - 1, plane[1] - 1
    anchor_states = anchor["analysis_states"]
    anchor_emb = anchor["embedding"].float()

    scalar_rows: list[dict[str, Any]] = []
    state_chunks: dict[int, list[np.ndarray]] = {int(b): [] for b in analysis_blocks}
    emb_chunks: list[np.ndarray] = []

    for st in range(0, len(grid_pairs), batch_size):
        gp = grid_pairs[st:st + batch_size]
        alpha4 = np.zeros((len(gp), 4), dtype=np.float32)
        alpha4[:, i0] = gp[:, 0]
        alpha4[:, j0] = gp[:, 1]

        post = atlas.make_alpha_post_batch(pair, components4, alpha4)
        run = run_downstream_multiblock(
            model, post, pair.early_states, bridge_capture_blocks, analysis_blocks, has_rn=True
        )
        for b in analysis_blocks:
            sv = run["analysis_states"][int(b)] - anchor_states[int(b)]
            state_chunks[int(b)].append(sv.cpu().numpy().astype(np.float32))
        ev = run["embedding"].float() - anchor_emb
        emb_chunks.append(ev.cpu().numpy().astype(np.float32))

        for spec in query_specs:
            obs = base.score_run_batch(
                model, run, spec["queries"], spec["attack_sem_en"], spec["object_sem_en"]
            )
            for bi, o in enumerate(obs):
                row = dict(common_visual_meta)
                row.update({
                    "language": spec["language"],
                    "query_mode": spec["query_mode"],
                    "query_candidate": spec["query_candidate"],
                    "point_index": st + bi,
                    "grid_index": st + bi,
                    "pc_a": plane[0],
                    "pc_b": plane[1],
                    "alpha_a": float(gp[bi, 0]),
                    "alpha_b": float(gp[bi, 1]),
                })
                row.update(o)
                scalar_rows.append(row)

        del run, post

    states_out = {
        int(b): np.concatenate(chunks, axis=0)
        for b, chunks in state_chunks.items()
    }
    return scalar_rows, states_out, np.concatenate(emb_chunks, axis=0)


def evaluate_surface_visual_multiblock_adaptive(
    model: torch.nn.Module,
    pair: Any,
    components4: Sequence[torch.Tensor],
    plane: tuple[int, int],
    grid_pairs: np.ndarray,
    query_specs: list[dict[str, Any]],
    bridge_capture_blocks: set[int],
    analysis_blocks: Sequence[int],
    batch_size: int,
    anchor: dict[str, Any],
    common_visual_meta: dict[str, Any],
    *,
    min_batch: int = 12,
):
    bs = max(1, int(batch_size))
    floor = max(1, int(min_batch))
    while True:
        try:
            return evaluate_surface_visual_multiblock(
                model, pair, components4, plane, grid_pairs, query_specs,
                bridge_capture_blocks, analysis_blocks, bs, anchor, common_visual_meta,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = str(exc).lower()
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg
            if not is_oom or bs <= floor:
                raise
            new_bs = max(floor, bs // 2)
            if new_bs == bs:
                raise
            print(f"[surface batch] CUDA OOM at batch={bs}; retrying at batch={new_bs}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bs = new_bs


def state_jacobian_all_blocks_at(
    model: torch.nn.Module,
    pair: Any,
    components4: Sequence[torch.Tensor],
    center: np.ndarray,
    eps: float,
    bridge_capture_blocks: set[int],
    analysis_blocks: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Eight finite-difference forwards total for ALL requested blocks.
    """
    pts = []
    for j in range(4):
        m = center.copy(); m[j] -= eps
        p = center.copy(); p[j] += eps
        pts.extend([m, p])
    pts = np.asarray(pts, np.float32)

    post = atlas.make_alpha_post_batch(pair, components4, pts)
    run = run_downstream_multiblock(
        model, post, pair.early_states, bridge_capture_blocks, analysis_blocks, has_rn=True
    )
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for block in analysis_blocks:
        state = run["analysis_states"][int(block)].cpu().numpy()
        J = []
        for j in range(4):
            J.append((state[2*j+1] - state[2*j]) / (2.0 * eps))
        J = np.stack(J, axis=0)
        s = np.linalg.svd(J, compute_uv=False)
        out[int(block)] = (J, s)
    return out


def _surface_metric_and_intrinsic(
    state_grid: np.ndarray,
    scalar_grid: np.ndarray,
    a_vals: np.ndarray,
    b_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dfa, dfb = np.gradient(scalar_grid, a_vals, b_vals, edge_order=2)
    dXa = np.gradient(state_grid, a_vals, axis=0, edge_order=2)
    dXb = np.gradient(state_grid, b_vals, axis=1, edge_order=2)
    G11 = np.sum(dXa * dXa, axis=-1)
    G12 = np.sum(dXa * dXb, axis=-1)
    G22 = np.sum(dXb * dXb, axis=-1)
    det = G11 * G22 - G12 * G12
    det = np.where(det < 1e-8, det + 1e-8, det)
    iga = (G22 * dfa - G12 * dfb) / det
    igb = (-G12 * dfa + G11 * dfb) / det
    return iga, igb


def _field_cosine(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    na = np.sqrt(a0*a0 + a1*a1)
    nb = np.sqrt(b0*b0 + b1*b1)
    return (a0*b0 + a1*b1) / (na*nb + 1e-12)


def intrinsic_alignment_rows_for_block(
    mean_states: dict[tuple, np.ndarray],
    surface_summary: list[dict[str, Any]],
    model_name: str,
    pump_mode: str,
    languages: Sequence[str],
    query_modes: Sequence[str],
    planes: Sequence[tuple[int, int]],
    grid_vals: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    G = len(grid_vals)
    fields: dict[tuple[str, str, tuple[int,int]], tuple[np.ndarray,np.ndarray]] = {}
    for lang in languages:
        for qm in query_modes:
            for plane in planes:
                key = (model_name, pump_mode, "synth", lang, "rn", plane)
                if key not in mean_states:
                    continue
                scalar = choose_surface_scalar(
                    surface_summary, model_name, pump_mode, "synth", lang, qm, plane
                )
                if len(scalar) != G*G:
                    continue
                state_grid = mean_states[key].reshape(G, G, -1)
                scalar_grid = scalar.reshape(G, G)
                fields[(lang, qm, plane)] = _surface_metric_and_intrinsic(
                    state_grid, scalar_grid, grid_vals, grid_vals
                )

    for plane in planes:
        for qm in query_modes:
            for ia, la in enumerate(languages):
                for lb in languages[ia+1:]:
                    ka=(la,qm,plane); kb=(lb,qm,plane)
                    if ka not in fields or kb not in fields:
                        continue
                    a0,a1=fields[ka]; b0,b1=fields[kb]
                    c=_field_cosine(a0,a1,b0,b1)
                    sa=SCRIPT_GROUP.get(la, la)
                    sb=SCRIPT_GROUP.get(lb, lb)
                    rows.append({
                        "model_name":model_name,
                        "pump_mode":pump_mode,
                        "pc_a":plane[0],"pc_b":plane[1],
                        "query_mode":qm,
                        "language_a":la,"language_b":lb,
                        "script_a":sa,"script_b":sb,
                        "same_script":bool(sa==sb),
                        "within_latin":bool(sa=="Latin" and sb=="Latin"),
                        "intrinsic_gradient_cos_mean":float(np.nanmean(c)),
                        "intrinsic_gradient_cos_median":float(np.nanmedian(c)),
                        "intrinsic_gradient_negative_fraction":float(np.nanmean(c < 0.0)),
                    })
    return rows


def centered_surface_pca_fractions(mean: np.ndarray, max_k: int = 8) -> np.ndarray:
    X=np.asarray(mean,np.float64)
    X=X-X.mean(axis=0,keepdims=True)
    if X.shape[0] <= 1:
        return np.zeros(max_k,dtype=np.float64)
    s=np.linalg.svd(X,full_matrices=False,compute_uv=False)
    e=s*s
    den=max(float(e.sum()),1e-20)
    frac=e/den
    out=np.zeros(max_k,dtype=np.float64)
    out[:min(max_k,len(frac))]=frac[:max_k]
    return out


def consecutive_block_transition_rows(
    means_by_block: dict[int, dict[tuple,np.ndarray]],
    tangents_by_block: dict[int, dict[tuple,np.ndarray]],
    languages: Sequence[str],
    planes: Sequence[tuple[int,int]],
    model_name: str,
    pump_mode: str,
) -> list[dict[str,Any]]:
    rows=[]
    blocks=sorted(means_by_block)
    for ba,bb in zip(blocks[:-1],blocks[1:]):
        if bb != ba+1:
            continue
        for lang in languages:
            for plane in planes:
                k=(model_name,pump_mode,"synth",lang,"rn",plane)
                if k not in means_by_block[ba] or k not in means_by_block[bb]:
                    continue
                A=means_by_block[ba][k];B=means_by_block[bb][k]
                dA=atlas.pairwise_distances(A);dB=atlas.pairwise_distances(B)
                t1,t2=atlas.mean_tangent_principal_cosines(
                    tangents_by_block[ba][k],tangents_by_block[bb][k]
                )
                den=max(float(np.linalg.norm(A)),1e-12)
                rows.append({
                    "block_from":int(ba),"block_to":int(bb),"language":lang,
                    "pc_a":plane[0],"pc_b":plane[1],
                    "distance_geometry_spearman":atlas.spearman(
                        atlas.upper_triangle_values(dA),atlas.upper_triangle_values(dB)
                    ),
                    "tangent_cos1":float(t1),"tangent_cos2":float(t2),
                    "procrustes_residual_32d":atlas.procrustes_residual_lowd(A,B,dim=32),
                    "relative_sheet_change_fro":float(np.linalg.norm(B-A)/den),
                })
    return rows


def plot_block_transition_overview(rows:list[dict[str,Any]],out:Path)->None:
    if not rows:return
    fig,axes=plt.subplots(1,3,figsize=(14,4.4))
    planes=sorted({(int(r["pc_a"]),int(r["pc_b"])) for r in rows})
    for plane in planes:
        q=[r for r in rows if (int(r["pc_a"]),int(r["pc_b"]))==plane]
        tos=sorted({int(r["block_to"]) for r in q})
        def avg(field,to):
            z=[safe_float(r[field]) for r in q if int(r["block_to"])==to]
            return float(np.nanmean(z)) if z else np.nan
        axes[0].plot(tos,[avg("distance_geometry_spearman",b) for b in tos],marker="o",label=f"PC{plane[0]}xPC{plane[1]}")
        axes[1].plot(tos,[avg("relative_sheet_change_fro",b) for b in tos],marker="o",label=f"PC{plane[0]}xPC{plane[1]}")
        axes[2].plot(tos,[avg("tangent_cos1",b) for b in tos],marker="o",label=f"PC{plane[0]}xPC{plane[1]} t1")
        axes[2].plot(tos,[avg("tangent_cos2",b) for b in tos],marker="x",linestyle="--",label=f"PC{plane[0]}xPC{plane[1]} t2")
    axes[0].set_title("Consecutive sheet-geometry conservation")
    axes[0].set_ylabel("distance-matrix Spearman")
    axes[1].set_title("Consecutive sheet displacement")
    axes[1].set_ylabel(r"$\|X_b-X_{b-1}\|_F/\|X_{b-1}\|_F$")
    axes[2].set_title("Consecutive tangent alignment")
    axes[2].set_ylabel("principal cosine")
    for ax in axes:
        ax.set_xlabel("destination block")
        ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=190,bbox_inches="tight");plt.close(fig)


def make_block_screen_rows(
    block: int,
    cross_lang: list[dict[str, Any]],
    intrinsic_rows: list[dict[str, Any]],
    jac_rows: list[dict[str, Any]],
    pca_rows: list[dict[str, Any]],
    planes: Sequence[tuple[int,int]],
    query_modes: Sequence[str],
) -> list[dict[str, Any]]:
    rows=[]
    for plane in planes:
        for qm in query_modes:
            cg=[r for r in cross_lang
                if int(r["pc_a"])==plane[0] and int(r["pc_b"])==plane[1]
                and r["query_mode"]==qm]
            ig=[r for r in intrinsic_rows
                if int(r["pc_a"])==plane[0] and int(r["pc_b"])==plane[1]
                and r["query_mode"]==qm]
            within=[r for r in ig if bool(r.get("within_latin"))]
            cross=[r for r in ig if not bool(r.get("same_script"))]
            jnat=[r for r in jac_rows if int(r.get("state_block",-1))==int(block)
                  and r.get("landmark")=="natural_rank4"
                  and r.get("query_mode")=="visual"]
            janc=[r for r in jac_rows if int(r.get("state_block",-1))==int(block)
                  and r.get("landmark")=="anchor"
                  and r.get("query_mode")=="visual"]
            def nm(rs,k):
                vals=[safe_float(r.get(k)) for r in rs]
                return float(np.nanmean(vals)) if vals else float("nan")
            def sigma1_frac(rs):
                vals=[]
                for r in rs:
                    s=np.array([safe_float(r.get(f"sigma{i}")) for i in range(1,5)],np.float64)
                    den=float(np.sum(s*s))
                    if den>0: vals.append(float(s[0]*s[0]/den))
                return float(np.nanmean(vals)) if vals else float("nan")
            pp=[r for r in pca_rows if int(r["pc_a"])==plane[0] and int(r["pc_b"])==plane[1] and r.get("condition")=="synth" and r.get("basis_kind")=="rn"]
            def cum_pca(k):
                by_case=defaultdict(float)
                for r in pp:
                    if int(r.get("component",99))<=k:
                        by_case[(r.get("language"),r.get("model_name"),r.get("pump_mode"))]+=safe_float(r.get("explained_fraction"))
                return float(np.nanmean(list(by_case.values()))) if by_case else float("nan")
            rows.append({
                "state_block":int(block),
                "pc_a":plane[0],"pc_b":plane[1],
                "query_mode":qm,
                "state_distance_spearman_mean":nm(cg,"state_distance_spearman"),
                "knn8_jaccard_mean":nm(cg,"knn8_jaccard"),
                "tangent_cos1_mean":nm(cg,"tangent_cos1"),
                "tangent_cos2_mean":nm(cg,"tangent_cos2"),
                "read_surface_spearman_mean":nm(cg,"read_surface_spearman"),
                "state_sheet_pca2_cumulative_mean":cum_pca(2),
                "state_sheet_pca4_cumulative_mean":cum_pca(4),
                "state_sheet_pca8_cumulative_mean":cum_pca(8),
                "intrinsic_within_latin_cos_mean":nm(within,"intrinsic_gradient_cos_mean"),
                "intrinsic_cross_script_cos_mean":nm(cross,"intrinsic_gradient_cos_mean"),
                "intrinsic_within_latin_negative_fraction":nm(within,"intrinsic_gradient_negative_fraction"),
                "intrinsic_cross_script_negative_fraction":nm(cross,"intrinsic_gradient_negative_fraction"),
                "jac_anchor_effective_rank":nm(janc,"effective_rank"),
                "jac_natural_effective_rank":nm(jnat,"effective_rank"),
                "jac_anchor_sigma1_energy_fraction":sigma1_frac(janc),
                "jac_natural_sigma1_energy_fraction":sigma1_frac(jnat),
            })
    return rows


def plot_block_sweep_overview(rows: list[dict[str,Any]], out: Path) -> None:
    if not rows:
        return
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    # State geometry: average query duplicates away.
    for plane in sorted({(int(r["pc_a"]),int(r["pc_b"])) for r in rows}):
        q=[r for r in rows if (int(r["pc_a"]),int(r["pc_b"]))==plane]
        blocks=sorted({int(r["state_block"]) for r in q})
        vals=[]
        for b in blocks:
            z=[r for r in q if int(r["state_block"])==b]
            vals.append(float(np.nanmean([safe_float(r["state_distance_spearman_mean"]) for r in z])))
        axes[0,0].plot(blocks,vals,marker="o",label=f"PC{plane[0]}xPC{plane[1]}")
    axes[0,0].set_title("Cross-language state-geometry conservation")
    axes[0,0].set_ylabel("mean distance Spearman");axes[0,0].legend();axes[0,0].grid(alpha=.25)

    for qm in sorted({str(r["query_mode"]) for r in rows}):
        q=[r for r in rows if r["query_mode"]==qm and int(r["pc_a"])==1 and int(r["pc_b"])==4]
        blocks=[int(r["state_block"]) for r in q]
        axes[0,1].plot(blocks,[safe_float(r["intrinsic_within_latin_cos_mean"]) for r in q],marker="o",label=f"{qm}: within Latin")
        axes[0,1].plot(blocks,[safe_float(r["intrinsic_cross_script_cos_mean"]) for r in q],marker="x",linestyle="--",label=f"{qm}: cross script")
    axes[0,1].set_title("Intrinsic READ-field alignment (PC1xPC4)")
    axes[0,1].set_ylabel("mean local gradient cosine");axes[0,1].legend(fontsize=8);axes[0,1].grid(alpha=.25)

    # Jacobian ranks; identical across plane/query, use first row per block.
    blocks=sorted({int(r["state_block"]) for r in rows})
    an=[];nat=[]
    for b in blocks:
        z=[r for r in rows if int(r["state_block"])==b]
        an.append(safe_float(z[0]["jac_anchor_effective_rank"]))
        nat.append(safe_float(z[0]["jac_natural_effective_rank"]))
    axes[1,0].plot(blocks,an,marker="o",label="zero-pulse anchor")
    axes[1,0].plot(blocks,nat,marker="o",label="natural rank-4")
    axes[1,0].set_title("Local 4D→state Jacobian effective rank")
    axes[1,0].set_ylabel("effective rank");axes[1,0].legend();axes[1,0].grid(alpha=.25)

    af=[];nf=[]
    for b in blocks:
        z=[r for r in rows if int(r["state_block"])==b]
        af.append(safe_float(z[0]["jac_anchor_sigma1_energy_fraction"]))
        nf.append(safe_float(z[0]["jac_natural_sigma1_energy_fraction"]))
    axes[1,1].plot(blocks,af,marker="o",label="zero-pulse anchor")
    axes[1,1].plot(blocks,nf,marker="o",label="natural rank-4")
    axes[1,1].set_title("Dominant local Jacobian direction")
    axes[1,1].set_ylabel(r"$\sigma_1^2/\sum_i\sigma_i^2$");axes[1,1].legend();axes[1,1].grid(alpha=.25)

    for ax in axes.flat:
        ax.set_xlabel("ViT block")
        ax.set_xticks(blocks)
    fig.suptitle("RN control-sheet block sweep",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97))
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=190,bbox_inches="tight")
    plt.close(fig)


def _save_block_outputs(
    profile_root: Path,
    block: int,
    mean_states: dict[tuple,np.ndarray],
    tangent_fields: dict[tuple,np.ndarray],
    local_geom: list[dict[str,Any]],
    pca_rows: list[dict[str,Any]],
    cross_lang: list[dict[str,Any]],
    intrinsic_rows: list[dict[str,Any]],
    jac_block_raw: list[dict[str,Any]],
    jac_block_summary: list[dict[str,Any]],
    npz_arrays: dict[str,np.ndarray],
) -> None:
    bdir=profile_root/f"B{int(block)}"
    ddir=bdir/"data"
    ddir.mkdir(parents=True,exist_ok=True)
    save_rows(ddir/"surface_local_geometry.csv",local_geom)
    save_rows(ddir/"surface_pca_spectrum.csv",pca_rows)
    save_rows(ddir/"cross_language_geometry.csv",cross_lang)
    save_rows(ddir/"intrinsic_gradient_alignment.csv",intrinsic_rows)
    save_rows(ddir/"state_jacobian_raw.csv",jac_block_raw)
    save_rows(ddir/"state_jacobian_summary.csv",jac_block_summary)
    if npz_arrays:
        np.savez_compressed(ddir/f"mean_B{int(block)}_surfaces.npz",**npz_arrays)


# =============================================================================
# Register-pump ablation extension
# =============================================================================

# Exact bank for this experiment.  These are post-QuickGELU MLP units and are
# zeroed BEFORE c_proj; do not silently substitute older supersets.
REG_NEURONS = {
    11: [9, 987, 1967, 2555, 3661, 3784],
    12: [42, 183, 983, 1571, 1816, 2687, 3002, 3008, 3868],
}

RAIL_CHANNELS = (437, 565, 650)


@dataclass
class PumpAuditStats:
    block: int
    units: list[int]
    calls: int = 0
    selected_values: int = 0
    pre_signed_sum: float = 0.0
    pre_abs_sum: float = 0.0
    pre_sq_sum: float = 0.0
    pre_positive: int = 0
    pre_max_abs: float = 0.0
    post_max_abs: float = 0.0
    output_width: int = 0


class AuditedRegisterPumpContext:
    """
    Exact B11/B12 register-pump intervention with both cumulative and per-call audit.

    Hook location is the MLP GELU/QuickGELU output.  no_pump zeros only the
    requested 4096-D units in that post-activation tensor, before c_proj.
    """
    def __init__(self, model: torch.nn.Module, mode: str, verbose: bool = True):
        if mode not in {"intact", "no_pump"}:
            raise ValueError(mode)
        self.model=model; self.mode=mode; self.verbose=verbose
        self.handles=[]
        self.stats={b:PumpAuditStats(block=b,units=list(u)) for b,u in REG_NEURONS.items()}
        self.call_rows:list[dict[str,Any]]=[]
        self.context:dict[str,Any]={}

    def set_context(self, **kwargs)->None:
        self.context=dict(kwargs)

    def __enter__(self):
        blocks=self.model.visual.transformer.resblocks
        for b,units in REG_NEURONS.items():
            act=_find_mlp_activation_module(blocks[b])
            idx_cpu=torch.tensor(units,dtype=torch.long)
            st=self.stats[b]
            def hook(_module,_inp,output,b=b,idx_cpu=idx_cpu,st=st):
                if not torch.is_tensor(output):
                    raise TypeError(f"B{b} activation output is {type(output)}")
                if output.ndim<2:
                    raise RuntimeError(f"B{b} activation output shape={tuple(output.shape)}")
                if max(st.units)>=output.shape[-1]:
                    raise IndexError(f"B{b}: activation width={output.shape[-1]}, cannot access units={st.units}")
                idx=idx_cpu.to(output.device)
                selected=output.index_select(-1,idx).detach().float()
                n=int(selected.numel())
                signed=float(selected.sum().cpu())
                abss=float(selected.abs().sum().cpu())
                sq=float(selected.square().sum().cpu())
                mx=float(selected.abs().max().cpu()) if n else 0.0
                pos=int((selected>0).sum().cpu())
                st.calls+=1; st.selected_values+=n
                st.pre_signed_sum+=signed; st.pre_abs_sum+=abss; st.pre_sq_sum+=sq
                st.pre_positive+=pos; st.pre_max_abs=max(st.pre_max_abs,mx)
                st.output_width=int(output.shape[-1])
                post_max=mx
                if self.mode=="no_pump":
                    out=output.clone(); out[...,idx]=0
                    post=out.index_select(-1,idx).detach().float()
                    post_max=float(post.abs().max().cpu()) if post.numel() else 0.0
                    st.post_max_abs=max(st.post_max_abs,post_max)
                    if post_max!=0.0:
                        raise RuntimeError(f"B{b} pump zero failed: post max={post_max}")
                else:
                    out=output
                    st.post_max_abs=max(st.post_max_abs,post_max)
                row=dict(self.context)
                row.update({
                    "pump_mode":self.mode,"block":int(b),"units":",".join(map(str,st.units)),
                    "n_units":len(st.units),"activation_width":int(output.shape[-1]),
                    "selected_values":n,
                    "pre_signed_mean":signed/max(n,1),"pre_absmean":abss/max(n,1),
                    "pre_rms":math.sqrt(sq/max(n,1)),"pre_positive_fraction":pos/max(n,1),
                    "pre_max_abs":mx,"post_selected_max_abs":post_max,
                    "zero_verified":bool(self.mode=="intact" or post_max==0.0),
                })
                self.call_rows.append(row)
                if st.calls==1 and self.verbose:
                    print(f"[register pump {self.mode}] B{b} units={st.units} absmean={abss/max(n,1):.6g} rms={math.sqrt(sq/max(n,1)):.6g} max={mx:.6g}")
                return out
            self.handles.append(act.register_forward_hook(hook))
        return self

    def __exit__(self,exc_type,exc,tb):
        for h in self.handles:h.remove()
        self.handles.clear()

    def rows(self,model_name:str)->list[dict[str,Any]]:
        rows=[]
        for b,st in sorted(self.stats.items()):
            n=max(st.selected_values,1)
            rows.append({
                "model_name":model_name,"pump_mode":self.mode,"block":b,
                "units":",".join(map(str,st.units)),"n_units":len(st.units),
                "calls":st.calls,"selected_values":st.selected_values,"activation_width":st.output_width,
                "pre_signed_mean":st.pre_signed_sum/n,"pre_absmean":st.pre_abs_sum/n,
                "pre_rms":math.sqrt(st.pre_sq_sum/n),"pre_positive_fraction":st.pre_positive/n,
                "pre_max_abs":st.pre_max_abs,"post_selected_max_abs":st.post_max_abs,
                "zero_verified":bool(self.mode=="intact" or st.post_max_abs==0.0),
            })
        return rows


def parse_modes(text:str)->list[str]:
    modes=[x.strip() for x in str(text).split(",") if x.strip()]
    bad=[x for x in modes if x not in {"intact","no_pump"}]
    if bad:raise ValueError(f"Unknown pump modes: {bad}")
    # Always process intact first when present: it is the oracle/reference condition.
    return sorted(set(modes),key=lambda x:0 if x=="intact" else 1)


def _mask_key(condition:str,key:str,language:str)->tuple[str,str,str]:
    return (str(condition),str(key),str(language))


def _tensor_stats_1d(x:torch.Tensor)->dict[str,float]:
    y=x.detach().float().reshape(-1)
    if y.numel()==0:
        return {"mean":float("nan"),"absmean":float("nan"),"rms":float("nan"),"max_abs":float("nan")}
    return {"mean":float(y.mean().cpu()),"absmean":float(y.abs().mean().cpu()),
            "rms":float(y.square().mean().sqrt().cpu()),"max_abs":float(y.abs().max().cpu())}


def collect_operational_lineage_diagnostics(
    model:torch.nn.Module,
    pair:Any,
    fixed_reg_mask:torch.Tensor,
    analysis_blocks:Sequence[int],
    meta:dict[str,Any],
)->tuple[list[dict[str,Any]],dict[str,np.ndarray]]:
    """
    Follow exact intact-defined register patch identities through the ordinary
    natural RN forward.  Also record passive residual/LN1 rails 437/565/650.
    No channel intervention is performed here.
    """
    visual=model.visual
    ib=int(visual.read_null_insert_block)
    blocks=visual.transformer.resblocks
    mask=fixed_reg_mask.detach().bool().reshape(-1).to(pair.x_pre_tbc.device)
    vecs:dict[str,np.ndarray]={}
    rows=[]
    requested=set(map(int,analysis_blocks))

    # Pre-B13 includes the real RN token; post-B13 is the exact natural run stored in pair.
    x_pre13=base.append_rn(model,pair.x_pre_tbc)
    x=pair.full_post_tbc

    def record(block:int,pre_tbc:torch.Tensor,post_tbc:torch.Tensor):
        with torch.no_grad(),base.model_autocast_context(model):
            ln1=blocks[block].ln_1(pre_tbc).float()
        post=post_tbc.float()
        # T,B,D; diagnostic pairs are batch-1.
        ln_sp=ln1[1:-1,0,:]; post_sp=post[1:-1,0,:]
        cls_ln=ln1[0,0,:]; rn_ln=ln1[-1,0,:]
        cls_post=post[0,0,:]; rn_post=post[-1,0,:]
        if len(mask)!=ln_sp.shape[0]:
            raise RuntimeError(f"fixed register mask length {len(mask)} != patch count {ln_sp.shape[0]}")
        ordmask=~mask
        reg_post=post_sp[mask]; ord_post=post_sp[ordmask]
        reg_ln=ln_sp[mask]; ord_ln=ln_sp[ordmask]
        reg_norm=reg_post.norm(dim=-1); ord_norm=ord_post.norm(dim=-1)
        if reg_post.numel():
            c=F.cosine_similarity(reg_post,cls_post[None,:].expand_as(reg_post),dim=-1)
            reg_mean=reg_post.mean(dim=0)
        else:
            c=torch.empty(0,device=post.device);reg_mean=torch.zeros_like(cls_post)
        vid=analyze_stable_slug(f"{meta.get('pump_mode')}__{meta.get('condition')}__{meta.get('sample_key')}__{meta.get('language')}__B{block}")
        vecs[vid]=reg_mean.detach().cpu().numpy().astype(np.float32)
        row=dict(meta)
        row.update({
            "state_block":int(block),"fixed_register_count":int(mask.sum().item()),"vector_id":vid,
            "fixed_reg_norm_mean":float(reg_norm.mean().cpu()) if reg_norm.numel() else float("nan"),
            "fixed_reg_norm_max":float(reg_norm.max().cpu()) if reg_norm.numel() else float("nan"),
            "ordinary_norm_mean":float(ord_norm.mean().cpu()) if ord_norm.numel() else float("nan"),
            "ordinary_norm_max":float(ord_norm.max().cpu()) if ord_norm.numel() else float("nan"),
            "fixed_reg_to_ordinary_norm_ratio":float((reg_norm.mean()/ord_norm.mean()).cpu()) if reg_norm.numel() and ord_norm.numel() else float("nan"),
            "fixed_reg_cls_cos_mean":float(c.mean().cpu()) if c.numel() else float("nan"),
            "fixed_reg_cls_cos_absmean":float(c.abs().mean().cpu()) if c.numel() else float("nan"),
            "cls_post_norm":float(cls_post.norm().cpu()),"rn_post_norm":float(rn_post.norm().cpu()),
        })
        for ch in RAIL_CHANNELS:
            row[f"pre_ln1_cls_ch{ch}"]=float(cls_ln[ch].cpu())
            row[f"pre_ln1_rn_ch{ch}"]=float(rn_ln[ch].cpu())
            row[f"pre_ln1_fixed_reg_ch{ch}_mean"]=float(reg_ln[:,ch].mean().cpu()) if reg_ln.numel() else float("nan")
            row[f"pre_ln1_fixed_reg_ch{ch}_absmean"]=float(reg_ln[:,ch].abs().mean().cpu()) if reg_ln.numel() else float("nan")
            row[f"pre_ln1_ordinary_ch{ch}_mean"]=float(ord_ln[:,ch].mean().cpu()) if ord_ln.numel() else float("nan")
            row[f"pre_ln1_ordinary_ch{ch}_absmean"]=float(ord_ln[:,ch].abs().mean().cpu()) if ord_ln.numel() else float("nan")
            row[f"post_cls_ch{ch}"]=float(cls_post[ch].cpu())
            row[f"post_rn_ch{ch}"]=float(rn_post[ch].cpu())
            row[f"post_fixed_reg_ch{ch}_mean"]=float(reg_post[:,ch].mean().cpu()) if reg_post.numel() else float("nan")
            row[f"post_fixed_reg_ch{ch}_absmean"]=float(reg_post[:,ch].abs().mean().cpu()) if reg_post.numel() else float("nan")
            row[f"post_ordinary_ch{ch}_mean"]=float(ord_post[:,ch].mean().cpu()) if ord_post.numel() else float("nan")
            row[f"post_ordinary_ch{ch}_absmean"]=float(ord_post[:,ch].abs().mean().cpu()) if ord_post.numel() else float("nan")
        rows.append(row)

    if ib in requested:
        record(ib,x_pre13,x)
    with torch.no_grad(),base.model_autocast_context(model):
        for b in range(ib+1,len(blocks)):
            pre=x
            x=blocks[b](x)
            if b in requested:
                record(b,pre,x)
    # Audit the model's own late norm-based register selection separately from
    # the frozen intact lineage used above.  This catches fallback/reallocation
    # under no_pump instead of silently calling it "register removal".
    final_sp=x[1:-1,0,:].float();final_norm=final_sp.norm(dim=-1)
    dyn=visual._make_implicit_register_mask(
        final_norm[None].detach(),register_threshold=70.0,max_registers=8,min_registers=1
    )[0].bool()
    inter=int((dyn & mask).sum().item());union=int((dyn | mask).sum().item())
    jacc=float(inter/max(union,1))
    for r in rows:
        r["final_dynamic_register_count"]=int(dyn.sum().item())
        r["final_dynamic_vs_fixed_intersection"]=inter
        r["final_dynamic_vs_fixed_jaccard"]=jacc
        r["final_dynamic_norm_mean"]=float(final_norm[dyn].mean().cpu()) if int(dyn.sum()) else float("nan")
    return rows,vecs


def annotate_lineage_mu_geometry(rows:list[dict[str,Any]],vectors:dict[str,np.ndarray])->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    """Fit intact per-block uncentered rank-2 register basis and reuse it for no_pump."""
    basis_rows=[];basis_by_block={}
    blocks=sorted({int(r["state_block"]) for r in rows})
    for b in blocks:
        ids=[r["vector_id"] for r in rows if int(r["state_block"])==b and r["pump_mode"]=="intact" and r["condition"]=="synth"]
        X=np.stack([vectors[i] for i in ids if i in vectors],axis=0).astype(np.float64) if ids else np.zeros((0,1))
        if len(X)<2:continue
        _u,s,vt=np.linalg.svd(X,full_matrices=False)
        k=min(2,vt.shape[0]);B=vt[:k]
        mean=X.mean(axis=0);mean_norm=np.linalg.norm(mean)
        for j in range(k):
            if np.dot(B[j],mean)<0:B[j]*=-1
        basis_by_block[b]=(B,mean/(mean_norm+1e-12),s)
        energy=s*s;den=float(energy.sum())
        basis_rows.append({
            "state_block":b,"n_intact_samples":len(X),
            "pc1_energy_fraction":float(energy[0]/den) if den else float("nan"),
            "rank2_energy_fraction":float(energy[:2].sum()/den) if den else float("nan"),
            "sigma1":float(s[0]),"sigma2":float(s[1]) if len(s)>1 else 0.0,
        })
    for r in rows:
        b=int(r["state_block"]);vid=r["vector_id"]
        if b not in basis_by_block or vid not in vectors:continue
        B,mu,s=basis_by_block[b];v=vectors[vid].astype(np.float64);vn=np.linalg.norm(v)+1e-12
        coeff=B@v;proj=B.T@coeff
        r["cos_intact_reg_mean"]=float(np.dot(v,mu)/vn)
        r["cos_intact_mu1"]=float(coeff[0]/vn) if B.shape[0]>0 else float("nan")
        r["cos_intact_mu2"]=float(coeff[1]/vn) if B.shape[0]>1 else float("nan")
        r["intact_rank2_fraction"]=float(np.dot(proj,proj)/(np.dot(v,v)+1e-12))
    return rows,basis_rows


def cross_pump_lineage_rows(rows:list[dict[str,Any]],vectors:dict[str,np.ndarray])->list[dict[str,Any]]:
    lookup={(r["condition"],r["sample_key"],r["language"],int(r["state_block"]),r["pump_mode"]):r for r in rows}
    out=[]
    bases=sorted({(r["condition"],r["sample_key"],r["language"],int(r["state_block"])) for r in rows})
    for condition,key,lang,b in bases:
        a=lookup.get((condition,key,lang,b,"intact"));n=lookup.get((condition,key,lang,b,"no_pump"))
        if not a or not n:continue
        va=vectors.get(a["vector_id"]);vn=vectors.get(n["vector_id"])
        cos=float(np.dot(va,vn)/(np.linalg.norm(va)*np.linalg.norm(vn)+1e-12)) if va is not None and vn is not None else float("nan")
        out.append({
            "condition":condition,"sample_key":key,"language":lang,"state_block":b,
            "fixed_register_count":a.get("fixed_register_count"),
            "reg_mean_cos_intact_vs_no_pump":cos,
            "fixed_reg_norm_intact":a.get("fixed_reg_norm_mean"),"fixed_reg_norm_no_pump":n.get("fixed_reg_norm_mean"),
            "fixed_reg_norm_ratio_no_pump_over_intact":safe_float(n.get("fixed_reg_norm_mean"))/(safe_float(a.get("fixed_reg_norm_mean"))+1e-12),
            "cls_cos_intact":a.get("fixed_reg_cls_cos_mean"),"cls_cos_no_pump":n.get("fixed_reg_cls_cos_mean"),
            "rank2_fraction_intact":a.get("intact_rank2_fraction"),"rank2_fraction_no_pump":n.get("intact_rank2_fraction"),
        })
    return out


def _field_grad_from_scalar(v:np.ndarray,G:int,vals:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    z=np.asarray(v,np.float64).reshape(G,G)
    return np.gradient(z,vals,vals,edge_order=2)


def _metric_intrinsic_and_stats(state_grid:np.ndarray,scalar_grid:np.ndarray,vals:np.ndarray):
    dfa,dfb=np.gradient(scalar_grid,vals,vals,edge_order=2)
    dXa=np.gradient(state_grid,vals,axis=0,edge_order=2)
    dXb=np.gradient(state_grid,vals,axis=1,edge_order=2)
    G11=np.sum(dXa*dXa,axis=-1);G12=np.sum(dXa*dXb,axis=-1);G22=np.sum(dXb*dXb,axis=-1)
    tr=G11+G22;disc=np.sqrt(np.maximum((G11-G22)**2+4*G12*G12,0.0))
    l1=np.maximum((tr+disc)/2,1e-12);l2=np.maximum((tr-disc)/2,1e-12)
    s1=np.sqrt(l1);s2=np.sqrt(l2);cond=s1/s2;area=s1*s2
    det=np.maximum(G11*G22-G12*G12,1e-10)
    iga=(G22*dfa-G12*dfb)/det;igb=(-G12*dfa+G11*dfb)/det
    return iga,igb,cond,area


def _neighbor_field_stats(v0:np.ndarray,v1:np.ndarray)->dict[str,float]:
    mag=np.sqrt(v0*v0+v1*v1)
    corrs=[];coss=[]
    for axis in (0,1):
        if axis==0:
            a0,a1=v0[:-1,:],v1[:-1,:];b0,b1=v0[1:,:],v1[1:,:];ma,mb=mag[:-1,:],mag[1:,:]
        else:
            a0,a1=v0[:,:-1],v1[:,:-1];b0,b1=v0[:,1:],v1[:,1:];ma,mb=mag[:,:-1],mag[:,1:]
        corrs.append(atlas.pearson(ma.reshape(-1),mb.reshape(-1)))
        c=_field_cosine(a0,a1,b0,b1);coss.append(float(np.nanmean(c)))
    return {"magnitude_mean":float(np.nanmean(mag)),"magnitude_std":float(np.nanstd(mag)),
            "neighbor_magnitude_pearson":float(np.nanmean(corrs)),"neighbor_direction_cosine":float(np.nanmean(coss))}


def cross_pump_field_rows_for_block(
    block:int,mean_states:dict[tuple,np.ndarray],surface_summary:list[dict[str,Any]],
    model_name:str,languages:Sequence[str],query_modes:Sequence[str],planes:Sequence[tuple[int,int]],grid_vals:np.ndarray,
)->list[dict[str,Any]]:
    rows=[];G=len(grid_vals)
    for lang in languages:
        for plane in planes:
            ki=(model_name,"intact","synth",lang,"rn",plane);kn=(model_name,"no_pump","synth",lang,"rn",plane)
            if ki not in mean_states or kn not in mean_states:continue
            Si=mean_states[ki].reshape(G,G,-1);Sn=mean_states[kn].reshape(G,G,-1)
            for qm in query_modes:
                vi=choose_surface_scalar(surface_summary,model_name,"intact","synth",lang,qm,plane)
                vn=choose_surface_scalar(surface_summary,model_name,"no_pump","synth",lang,qm,plane)
                if len(vi)!=G*G or len(vn)!=G*G:continue
                zi=vi.reshape(G,G);zn=vn.reshape(G,G)
                eia,eib=_field_grad_from_scalar(vi,G,grid_vals);ena,enb=_field_grad_from_scalar(vn,G,grid_vals)
                iia,iib,ci,ai=_metric_intrinsic_and_stats(Si,zi,grid_vals)
                ina,inb,cn,an=_metric_intrinsic_and_stats(Sn,zn,grid_vals)
                ecos=_field_cosine(eia,eib,ena,enb);icos=_field_cosine(iia,iib,ina,inb)
                esi=_neighbor_field_stats(eia,eib);esn=_neighbor_field_stats(ena,enb)
                isi=_neighbor_field_stats(iia,iib);isn=_neighbor_field_stats(ina,inb)
                rows.append({
                    "state_block":int(block),"language":lang,"pc_a":plane[0],"pc_b":plane[1],"query_mode":qm,
                    "read_surface_spearman_intact_vs_no_pump":atlas.spearman(vi,vn),
                    "euclidean_gradient_cos_mean":float(np.nanmean(ecos)),"euclidean_gradient_cos_median":float(np.nanmedian(ecos)),
                    "euclidean_gradient_negative_fraction":float(np.nanmean(ecos<0)),
                    "intrinsic_gradient_cos_mean":float(np.nanmean(icos)),"intrinsic_gradient_cos_median":float(np.nanmedian(icos)),
                    "intrinsic_gradient_negative_fraction":float(np.nanmean(icos<0)),
                    "metric_condition_intact_mean":float(np.nanmean(ci)),"metric_condition_no_pump_mean":float(np.nanmean(cn)),
                    "metric_condition_intact_p95":float(np.nanquantile(ci,.95)),"metric_condition_no_pump_p95":float(np.nanquantile(cn,.95)),
                    "metric_area_intact_mean":float(np.nanmean(ai)),"metric_area_no_pump_mean":float(np.nanmean(an)),
                    **{f"euclidean_intact_{k}":v for k,v in esi.items()},
                    **{f"euclidean_no_pump_{k}":v for k,v in esn.items()},
                    **{f"intrinsic_intact_{k}":v for k,v in isi.items()},
                    **{f"intrinsic_no_pump_{k}":v for k,v in isn.items()},
                })
    return rows


def cross_pump_geometry_rows_for_block(
    block:int,mean_states:dict[tuple,np.ndarray],tangent_fields:dict[tuple,np.ndarray],surface_summary:list[dict[str,Any]],
    model_name:str,languages:Sequence[str],query_modes:Sequence[str],planes:Sequence[tuple[int,int]],
)->list[dict[str,Any]]:
    rows=[]
    for lang in languages:
        for plane in planes:
            ki=(model_name,"intact","synth",lang,"rn",plane);kn=(model_name,"no_pump","synth",lang,"rn",plane)
            if ki not in mean_states or kn not in mean_states:continue
            A=mean_states[ki];B=mean_states[kn]
            dA=atlas.pairwise_distances(A);dB=atlas.pairwise_distances(B)
            t1,t2=atlas.mean_tangent_principal_cosines(tangent_fields[ki],tangent_fields[kn])
            displacement=float(np.linalg.norm(B-A)/(np.linalg.norm(A)+1e-12))
            an=np.linalg.norm(A,axis=1);bn=np.linalg.norm(B,axis=1);valid=(an*bn)>1e-10
            pc=float(np.mean(np.sum(A[valid]*B[valid],axis=1)/(an[valid]*bn[valid]))) if np.any(valid) else float("nan")
            base_row={
                "state_block":int(block),"language":lang,"pc_a":plane[0],"pc_b":plane[1],
                "state_distance_spearman_intact_vs_no_pump":atlas.spearman(atlas.upper_triangle_values(dA),atlas.upper_triangle_values(dB)),
                "state_distance_pearson_intact_vs_no_pump":atlas.pearson(atlas.upper_triangle_values(dA),atlas.upper_triangle_values(dB)),
                "knn8_intact_vs_no_pump":atlas.knn_overlap(A,B,k=8),
                "tangent_cos1_intact_vs_no_pump":t1,"tangent_cos2_intact_vs_no_pump":t2,
                "procrustes_residual_32d_intact_vs_no_pump":atlas.procrustes_residual_lowd(A,B,dim=32),
                "relative_sheet_displacement_no_pump_vs_intact":displacement,
                "pointwise_state_cos_mean_intact_vs_no_pump":pc,
            }
            for qm in query_modes:
                r=dict(base_row);r["query_mode"]=qm
                va=choose_surface_scalar(surface_summary,model_name,"intact","synth",lang,qm,plane)
                vb=choose_surface_scalar(surface_summary,model_name,"no_pump","synth",lang,qm,plane)
                if len(va)==len(vb) and len(va):
                    r["read_surface_spearman_intact_vs_no_pump"]=atlas.spearman(va,vb)
                    r["read_surface_rmse_intact_vs_no_pump"]=float(np.sqrt(np.mean((va-vb)**2)))
                rows.append(r)
    return rows


def annotate_extrema_support_pump(rows:list[dict[str,Any]],coeff_lookup:dict[tuple[str,str,str],np.ndarray],support_model:EmpiricalSupportModel)->None:
    for r in rows:
        key=(str(r.get("pump_mode")),str(r.get("sample_key")),str(r.get("language")))
        coeff=coeff_lookup.get(key)
        if coeff is None:continue
        a=np.array([[safe_float(r.get(f"alpha{i}")) for i in range(1,5)]],dtype=np.float32)
        sm=support_metrics_for_alphas(support_model,coeff,a)
        for name,arr in sm.items():
            v=arr[0];r[name]=bool(v) if np.issubdtype(np.asarray(arr).dtype,np.bool_) else float(v)


def plot_pump_ablation_overview(block_rows:list[dict[str,Any]],lineage_cross:list[dict[str,Any]],out:Path)->None:
    if not block_rows:return
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    for plane in ((1,2),(1,4)):
        q=[r for r in block_rows if (int(r["pc_a"]),int(r["pc_b"]))==plane and r["query_mode"]=="native"]
        by=defaultdict(list)
        for r in q:by[int(r["state_block"])].append(r)
        bs=sorted(by);ys=[float(np.nanmean([safe_float(x.get("state_distance_spearman_intact_vs_no_pump")) for x in by[b]])) for b in bs]
        axes[0,0].plot(bs,ys,marker="o",label=f"PC{plane[0]}xPC{plane[1]}")
    axes[0,0].set_title("Intact vs no-pump state geometry");axes[0,0].set_ylabel("distance Spearman");axes[0,0].legend();axes[0,0].grid(alpha=.25)
    for qm in ("english","native"):
        q=[r for r in block_rows if int(r["pc_a"])==1 and int(r["pc_b"])==4 and r["query_mode"]==qm]
        by=defaultdict(list)
        for r in q:by[int(r["state_block"])].append(r)
        bs=sorted(by);ys=[float(np.nanmean([safe_float(x.get("intrinsic_gradient_cos_mean")) for x in by[b]])) for b in bs]
        axes[0,1].plot(bs,ys,marker="o",label=qm)
    axes[0,1].set_title("PC1xPC4 intrinsic READ field: intact vs no-pump");axes[0,1].set_ylabel("mean gradient cosine");axes[0,1].legend();axes[0,1].grid(alpha=.25)
    by=defaultdict(list)
    for r in lineage_cross:
        if r.get("condition")=="synth":by[int(r["state_block"])].append(r)
    bs=sorted(by);ys=[float(np.nanmean([safe_float(x.get("fixed_reg_norm_ratio_no_pump_over_intact")) for x in by[b]])) for b in bs]
    axes[1,0].plot(bs,ys,marker="o")
    axes[1,0].axhline(1.0,color="k",lw=.7,alpha=.5);axes[1,0].set_title("Fixed intact-register lineage norm under no-pump");axes[1,0].set_ylabel("no-pump / intact norm");axes[1,0].grid(alpha=.25)
    q=[r for r in block_rows if int(r["pc_a"])==1 and int(r["pc_b"])==4 and r["query_mode"]=="native"]
    by=defaultdict(list)
    for r in q:by[int(r["state_block"])].append(r)
    bs=sorted(by)
    disp=[float(np.nanmean([safe_float(x.get("relative_sheet_displacement_no_pump_vs_intact")) for x in by[b]])) for b in bs]
    cond=[float(np.nanmean([safe_float(x.get("metric_condition_no_pump_mean"))/max(safe_float(x.get("metric_condition_intact_mean")),1e-12) for x in by[b]])) for b in bs]
    axes[1,1].plot(bs,disp,marker="o",label="sheet displacement")
    axes[1,1].plot(bs,cond,marker="x",linestyle="--",label="metric cond. ratio")
    axes[1,1].set_title("PC1xPC4 deformation diagnostics");axes[1,1].legend();axes[1,1].grid(alpha=.25)
    for ax in axes.flat:
        ax.set_xlabel("ViT block");ax.set_xticks(sorted({int(r["state_block"]) for r in block_rows}))
    fig.suptitle("Register-pump ablation: multiblock control-sheet audit",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97));out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=190,bbox_inches="tight");plt.close(fig)


def analyze_main()->None:
    # Load extraction dependencies only for this command; cached plots do not use them.
    global base, atlas, DEFAULT_CHECKPOINT
    import probe_tools_rn_control as base
    import probe_tools_rn_manifold as atlas
    DEFAULT_CHECKPOINT = base.DEFAULT_CHECKPOINT
    ap=argparse.ArgumentParser(description="Intact vs B11/B12 register-pump ablation over the RN multilingual control sheet")
    ap.add_argument("--checkpoint",default=DEFAULT_CHECKPOINT);ap.add_argument("--module-root",default=".")
    ap.add_argument("--dataset-repo",default=DEFAULT_DATASET_REPO);ap.add_argument("--dataset-root",default="")
    ap.add_argument("--languages",default=DEFAULT_LANGUAGES);ap.add_argument("--query-modes",default=DEFAULT_QUERY_MODES)
    ap.add_argument("--pump-modes",default="intact,no_pump",help="Comma-separated subset of intact,no_pump")
    ap.add_argument("--ridge-modes",default="intact",help="Pump modes receiving full support-constrained ridge/extrema search. Default intentionally excludes no_pump.")
    ap.add_argument("--atlas-keys",type=int,default=DEFAULT_ATLAS_KEYS);ap.add_argument("--basis-pairs-per-language",type=int,default=DEFAULT_BASIS_PAIRS_PER_LANGUAGE)
    ap.add_argument("--planes",default=DEFAULT_PLANES);ap.add_argument("--grid-points",type=int,default=DEFAULT_GRID_POINTS);ap.add_argument("--grid-bound",type=float,default=DEFAULT_GRID_BOUND)
    ap.add_argument("--surface-batch",type=int,default=DEFAULT_SURFACE_BATCH)
    ap.add_argument("--state-blocks",default="",help="Default B13-B23 in fast mode; B19-B21 in full mode")
    ap.add_argument("--fast-ridge","--fast_ridge",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--ridge",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--state-jacobians",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--include-random-control",action=argparse.BooleanOptionalAction,default=False)
    ap.add_argument("--include-norta-surfaces",action=argparse.BooleanOptionalAction,default=False)
    ap.add_argument("--support-pcs",type=int,default=DEFAULT_SUPPORT_PCS);ap.add_argument("--ridge-support-quantile",type=float,default=DEFAULT_SUPPORT_QUANTILE)
    ap.add_argument("--ridge-support-draw-multiplier",type=int,default=DEFAULT_SUPPORT_DRAW_MULTIPLIER);ap.add_argument("--ridge-support-min-local-span",type=float,default=DEFAULT_SUPPORT_MIN_LOCAL_SPAN)
    ap.add_argument("--ridge-points",type=int,default=DEFAULT_RIDGE_POINTS);ap.add_argument("--ridge-bound",type=float,default=DEFAULT_RIDGE_BOUND);ap.add_argument("--ridge-radius",type=float,default=DEFAULT_RIDGE_RADIUS)
    ap.add_argument("--ridge-batch",type=int,default=DEFAULT_RIDGE_BATCH);ap.add_argument("--ridge-topk",type=int,default=DEFAULT_RIDGE_TOPK);ap.add_argument("--ridge-refine-steps",type=int,default=DEFAULT_RIDGE_REFINE_STEPS);ap.add_argument("--ridge-refine-random",type=int,default=DEFAULT_RIDGE_REFINE_RANDOM)
    ap.add_argument("--state-jac-eps",type=float,default=DEFAULT_STATE_JAC_EPS);ap.add_argument("--export-ply",action=argparse.BooleanOptionalAction,default=False)
    ap.add_argument("--tf32-fast",action=argparse.BooleanOptionalAction,default=False);ap.add_argument("--aggressive-cleanup",action=argparse.BooleanOptionalAction,default=False)
    ap.add_argument("--seed",type=int,default=DEFAULT_SEED);ap.add_argument("--device",default="cuda")
    ap.add_argument("--output-dir",default="rn_register_pump_multiblock_surface")
    args=ap.parse_args()

    if args.fast_ridge:
        if not _cli_supplied("--grid-points"):args.grid_points=9
        if not _cli_supplied("--ridge-points"):args.ridge_points=192
        if not _cli_supplied("--ridge-refine-steps"):args.ridge_refine_steps=1
        if not _cli_supplied("--ridge-refine-random"):args.ridge_refine_random=8
        if not _cli_supplied("--ridge-topk"):args.ridge_topk=8
        if not _cli_supplied("--ridge-batch"):args.ridge_batch=max(DEFAULT_RIDGE_BATCH,96)
    analysis_blocks=parse_block_spec(args.state_blocks) if args.state_blocks.strip() else (list(range(13,24)) if args.fast_ridge else [19,20,21])
    pump_modes=parse_modes(args.pump_modes);ridge_modes=set(parse_modes(args.ridge_modes)) if args.ridge_modes.strip() else set()
    if not set(pump_modes)<=set(("intact","no_pump")):raise ValueError(pump_modes)
    if not ridge_modes<=set(pump_modes):
        print(f"[note] ridge modes {sorted(ridge_modes-set(pump_modes))} are not evaluated pump modes and will be ignored")
        ridge_modes &= set(pump_modes)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=bool(args.tf32_fast)
        if hasattr(torch.backends,"cudnn"):torch.backends.cudnn.allow_tf32=bool(args.tf32_fast)
        try:torch.set_float32_matmul_precision("high" if args.tf32_fast else "highest")
        except Exception:pass
    original_cleanup=base.cleanup_cuda
    if not args.aggressive_cleanup:base.cleanup_cuda=lambda:None

    languages=parse_strs(args.languages);query_modes=parse_strs(args.query_modes);planes=analyze_parse_planes(args.planes)
    model_name="A_native";model_names=[model_name]
    out=Path(args.output_dir);profile_name="_register_pump_fast" if args.fast_ridge else "_register_pump_full"
    profile=out/profile_name;shared=profile/"shared";plots=profile/"plots";ply_dir=profile/"ply"
    for d in (shared,plots):d.mkdir(parents=True,exist_ok=True)
    if args.export_ply:ply_dir.mkdir(parents=True,exist_ok=True)

    norta,attacks,norta_idx,attack_idx,shared_keys=base.prepare_datasets(args.dataset_repo,args.dataset_root,languages,0,args.seed)
    if args.atlas_keys>0 and len(shared_keys)>args.atlas_keys:
        rng=random.Random(args.seed);atlas_keys=sorted(rng.sample(shared_keys,args.atlas_keys))
    else:atlas_keys=list(shared_keys)
    loaded=load_native_model(args);model=loaded.model;device=loaded.device
    insert_block=int(model.visual.read_null_insert_block);bridge_capture_list=[int(x) for x in model.read_implant.capture_block_list()];bridge_capture=set(bridge_capture_list);early_needed={b for b in bridge_capture_list if b<insert_block}

    # Intact basis and support are the common coordinate contract for ALL pump modes.
    print("\n[reference] fitting intact RN basis/support; this basis is frozen for no_pump")
    ref_basis_info,support_records=fit_reference_basis_and_support_records(model,loaded.preprocess,norta,attacks,norta_idx,attack_idx,shared_keys,languages,args.basis_pairs_per_language,early_needed,max_rank=4,device=device)
    reference_basis=ref_basis_info["basis_kd"][:4].detach().float().to(device);random_basis=atlas.random_basis_orthogonal_to(reference_basis,4,args.seed+991)
    support_model,support_rows=fit_empirical_support_model(support_records,args.support_pcs)
    save_rows(shared/"empirical_support_samples.csv",support_rows);save_empirical_support_model(shared/"empirical_support_model.npz",support_model)
    save_json(shared/"empirical_support_summary.json",{"n_samples":len(support_rows),"natural_alpha":[1,1,1,1],"anchor_alpha":[0,0,0,0],"md_q95":support_model.md_q95,"md_q99":support_model.md_q99,"coordinate_contract":"intact RN rank-4 basis reused for intact and no_pump"})
    np.savez_compressed(shared/"reference_basis.npz",reference_basis=reference_basis.cpu().numpy(),random_basis=random_basis.cpu().numpy(),singular_values=ref_basis_info["singular_values"].cpu().numpy(),explained_energy=np.asarray(ref_basis_info["explained_energy"]))

    # Intact oracle patch identities.  no_pump never re-thresholds its collapsed norms.
    print("\n[oracle] caching intact B12 register identities for the atlas images")
    fixed_masks:dict[tuple[str,str,str],torch.Tensor]={}
    for key in tqdm(atlas_keys,desc="intact register-mask oracle",unit="key"):
        nrow=dict(norta[norta_idx[key]]);img=base.preprocess_pil(loaded.preprocess,base.ensure_pil(nrow["image"]),device);p=base.build_b13_pair(model,img,key,"norta",early_needed)
        fixed_masks[_mask_key("norta",key,"shared")]=p.b12_register_mask[0].detach().cpu().bool();del p,img
        for lang in languages:
            srow=dict(attacks[lang][attack_idx[lang][key]]);img=base.preprocess_pil(loaded.preprocess,base.ensure_pil(srow["image"]),device);p=base.build_b13_pair(model,img,key,"synth",early_needed)
            fixed_masks[_mask_key("synth",key,lang)]=p.b12_register_mask[0].detach().cpu().bool();del p,img
    gc.collect()

    query_cache={};sem_cache={}
    def get_query(label:str):
        if label not in query_cache:query_cache[label]=base.prepare_read_queries(model,loaded.clip_module,label,device)
        return query_cache[label]
    def get_sem(label:str):
        if label not in sem_cache:sem_cache[label]=base.prompt_bank_embedding(model,loaded.clip_module,label,["{label}","a photo of {label}","the image depicts {label}","there is {label}","a picture of {label}"],device)
        return sem_cache[label]

    grid_vals,grid_pairs=atlas.grid_alpha_pairs(args.grid_points,args.grid_bound);h=float(grid_vals[1]-grid_vals[0])
    b13_delta_rows=[];surface_raw=[];ridge_raw=[];jac_raw=[];state_acc={};pump_summary_rows=[];pump_call_rows=[];lineage_rows=[];lineage_vectors={};ridge_coeff_lookup={}

    for pump_mode in pump_modes:
        print(f"\n{'='*80}\n[pump condition] {pump_mode}\n{'='*80}")
        with AuditedRegisterPumpContext(model,pump_mode,verbose=True) as pump_ctx:
            for key_index,key in enumerate(tqdm(atlas_keys,desc=pump_mode,unit="key")):
                lang_rows={lang:dict(attacks[lang][attack_idx[lang][key]]) for lang in languages}
                nrow=dict(norta[norta_idx[key]]);npil=base.ensure_pil(nrow["image"]);nimg=base.preprocess_pil(loaded.preprocess,npil,device)
                pump_ctx.set_context(sample_key=key,condition="norta",language="shared",phase="build_b13_pair")
                npair=base.build_b13_pair(model,nimg,key,"norta",early_needed)
                n_rn,n_rand=atlas.per_pc_components(npair.delta_ord_btd,reference_basis,random_basis,4)
                b13_delta_rows.append({"model_name":model_name,"pump_mode":pump_mode,"sample_key":key,"condition":"norta","language":"shared","reference_rank4_capture":reference_capture_fraction(npair.delta_ord_btd,reference_basis,4),"delta_norm_mean":float(npair.delta_ord_btd.norm(dim=-1).mean().cpu())})
                lr,lv=collect_operational_lineage_diagnostics(model,npair,fixed_masks[_mask_key("norta",key,"shared")],analysis_blocks,{"model_name":model_name,"pump_mode":pump_mode,"sample_key":key,"condition":"norta","language":"shared"});lineage_rows.extend(lr);lineage_vectors.update(lv)
                if args.include_norta_surfaces:
                    n_anchor=anchor_run_multiblock(model,npair,bridge_capture,analysis_blocks);n_specs=[]
                    for lang,srow in lang_rows.items():
                        ae=str(srow["attack_word_en"]);oe=str(srow["object_label_en"]);an=str(srow["attack_word"])
                        for qm in query_modes:
                            ql=ae if qm=="english" else an;n_specs.append({"language":lang,"query_mode":qm,"query_candidate":ql,"queries":get_query(ql),"attack_sem_en":get_sem(ae),"object_sem_en":get_sem(oe),"attack_word_en":ae,"object_label_en":oe})
                    for basis_kind,comps in _basis_options(n_rn,n_rand,args.include_random_control):
                        for plane in planes:
                            rr,states_by_block,embs=evaluate_surface_visual_multiblock_adaptive(model,npair,comps,plane,grid_pairs,n_specs,bridge_capture,analysis_blocks,args.surface_batch,n_anchor,{"model_name":model_name,"pump_mode":pump_mode,"sample_key":key,"condition":"norta","basis_kind":basis_kind});surface_raw.extend(rr)
                            for block,states in states_by_block.items():add_surface_acc(state_acc,(block,model_name,pump_mode,"norta","shared",basis_kind,plane),states,embs)
                    del n_anchor

                for lang in languages:
                    srow=lang_rows[lang];spil=base.ensure_pil(srow["image"]);simg=base.preprocess_pil(loaded.preprocess,spil,device)
                    pump_ctx.set_context(sample_key=key,condition="synth",language=lang,phase="build_b13_pair")
                    spair=base.build_b13_pair(model,simg,key,"synth",early_needed)
                    srn,srand=atlas.per_pc_components(spair.delta_ord_btd,reference_basis,random_basis,4);coeff_kt=_rn_coeff_kt(spair.delta_ord_btd,reference_basis);ridge_coeff_lookup[(pump_mode,str(key),str(lang))]=coeff_kt
                    b13_delta_rows.append({"model_name":model_name,"pump_mode":pump_mode,"sample_key":key,"condition":"synth","language":lang,"reference_rank4_capture":reference_capture_fraction(spair.delta_ord_btd,reference_basis,4),"delta_norm_mean":float(spair.delta_ord_btd.norm(dim=-1).mean().cpu())})
                    lr,lv=collect_operational_lineage_diagnostics(model,spair,fixed_masks[_mask_key("synth",key,lang)],analysis_blocks,{"model_name":model_name,"pump_mode":pump_mode,"sample_key":key,"condition":"synth","language":lang});lineage_rows.extend(lr);lineage_vectors.update(lv)
                    s_anchor=anchor_run_multiblock(model,spair,bridge_capture,analysis_blocks)
                    ae=str(srow["attack_word_en"]);oe=str(srow["object_label_en"]);an=str(srow["attack_word"]);specs=[]
                    for qm in query_modes:
                        ql=ae if qm=="english" else an;specs.append({"language":lang,"query_mode":qm,"query_candidate":ql,"queries":get_query(ql),"attack_sem_en":get_sem(ae),"object_sem_en":get_sem(oe),"attack_word_en":ae,"object_label_en":oe})
                    for basis_kind,comps in _basis_options(srn,srand,args.include_random_control):
                        for plane in planes:
                            rr,states_by_block,embs=evaluate_surface_visual_multiblock_adaptive(model,spair,comps,plane,grid_pairs,specs,bridge_capture,analysis_blocks,args.surface_batch,s_anchor,{"model_name":model_name,"pump_mode":pump_mode,"sample_key":key,"condition":"synth","basis_kind":basis_kind});surface_raw.extend(rr)
                            for block,states in states_by_block.items():add_surface_acc(state_acc,(block,model_name,pump_mode,"synth",lang,basis_kind,plane),states,embs)

                        # Anchor/natural local Jacobians are causal and cheap; compute for every pump mode.
                        if basis_kind=="rn" and args.state_jacobians:
                            for landmark,c in (("anchor",np.zeros(4,np.float32)),("natural_rank4",np.ones(4,np.float32))):
                                bj=state_jacobian_all_blocks_at(model,spair,comps,c,args.state_jac_eps,bridge_capture,analysis_blocks)
                                sm=support_metrics_for_alphas(support_model,coeff_kt,c[None])
                                for block,(_J,svals) in bj.items():
                                    row={"model_name":model_name,"pump_mode":pump_mode,"sample_key":key,"condition":"synth","language":lang,"basis_kind":basis_kind,"landmark":landmark,"query_mode":"visual","state_block":int(block),"effective_rank":atlas.effective_rank_from_singular_values(svals)}
                                    for name,arr in sm.items():
                                        v=arr[0];row[name]=bool(v) if np.issubdtype(np.asarray(arr).dtype,np.bool_) else float(v)
                                    for i in range(4):row[f"alpha{i+1}"]=float(c[i]);row[f"sigma{i+1}"]=float(svals[i]) if i<len(svals) else 0.0
                                    jac_raw.append(row)

                        # Full extrema search is intentionally opt-in per pump mode.
                        if basis_kind=="rn" and args.ridge and pump_mode in ridge_modes:
                            seed=args.seed+1000*key_index+137*(languages.index(lang)+1)+(0 if pump_mode=="intact" else 500000)
                            pts,ptm=support_constrained_sobol_points(support_model,coeff_kt,args.ridge_points,args.ridge_bound,args.ridge_radius,seed,args.ridge_support_quantile,args.ridge_support_draw_multiplier,args.ridge_support_min_local_span)
                            rr,_=atlas.evaluate_alpha_cloud(model,spair,comps,pts,specs,bridge_capture,args.ridge_batch,{"model_name":model_name,"pump_mode":pump_mode,"sample_key":key,"condition":"synth","basis_kind":basis_kind,"search_stage":"sobol_support"});_append_support_metrics(rr,ptm)
                            for r in rr:r["search_eligible"]=True
                            all_rr=list(rr);ridge_raw.extend(rr)
                            for step in range(args.ridge_refine_steps):
                                scale=args.ridge_bound*(0.30/(2**step));refine=atlas.refine_points_around_extrema(all_rr,query_modes,scale,args.ridge_refine_random,args.ridge_bound,args.ridge_radius,seed+9000+step);refine,rm=filter_points_to_empirical_support(support_model,coeff_kt,refine,args.ridge_support_quantile)
                                if len(refine):
                                    nr,_=atlas.evaluate_alpha_cloud(model,spair,comps,refine,specs,bridge_capture,args.ridge_batch,{"model_name":model_name,"pump_mode":pump_mode,"sample_key":key,"condition":"synth","basis_kind":basis_kind,"search_stage":f"refine_{step+1}"});_append_support_metrics(nr,rm)
                                    for r in nr:r["search_eligible"]=True
                                    ridge_raw.extend(nr);all_rr.extend(nr)
                            if args.state_jacobians:
                                for qm in query_modes:
                                    qrows=[r for r in all_rr if r["query_mode"]==qm and bool(r.get("search_eligible",True))]
                                    if not qrows:continue
                                    for landmark,rrr in (("max",max(qrows,key=lambda r:safe_float(r["relative_calibrated"]))),("min",min(qrows,key=lambda r:safe_float(r["relative_calibrated"])))):
                                        c=np.array([rrr[f"alpha{i}"] for i in range(1,5)],np.float32);bj=state_jacobian_all_blocks_at(model,spair,comps,c,args.state_jac_eps,bridge_capture,analysis_blocks);sm=support_metrics_for_alphas(support_model,coeff_kt,c[None])
                                        for block,(_J,svals) in bj.items():
                                            row={"model_name":model_name,"pump_mode":pump_mode,"sample_key":key,"condition":"synth","language":lang,"basis_kind":basis_kind,"landmark":landmark,"query_mode":qm,"state_block":int(block),"effective_rank":atlas.effective_rank_from_singular_values(svals)}
                                            for name,arr in sm.items():
                                                v=arr[0];row[name]=bool(v) if np.issubdtype(np.asarray(arr).dtype,np.bool_) else float(v)
                                            for i in range(4):row[f"alpha{i+1}"]=float(c[i]);row[f"sigma{i+1}"]=float(svals[i]) if i<len(svals) else 0.0
                                            jac_raw.append(row)
                    del spair,simg,spil,s_anchor
                del npair,nimg,npil
                if args.aggressive_cleanup:_paper_cleanup(empty_cache=True)
                elif (key_index+1)%4==0:gc.collect()
            pump_summary_rows.extend(pump_ctx.rows(model_name));pump_call_rows.extend(pump_ctx.call_rows)

    base.cleanup_cuda=original_cleanup
    lineage_rows,mu_basis_rows=annotate_lineage_mu_geometry(lineage_rows,lineage_vectors);lineage_cross=cross_pump_lineage_rows(lineage_rows,lineage_vectors)
    save_rows(shared/"register_pump_intervention_audit.csv",pump_summary_rows);save_rows(shared/"register_pump_call_log.csv",pump_call_rows)
    save_rows(shared/"fixed_register_lineage.csv",lineage_rows);save_rows(shared/"fixed_register_mu_basis_summary.csv",mu_basis_rows);save_rows(shared/"fixed_register_intact_vs_no_pump.csv",lineage_cross)
    lineage_summary=base.aggregate_rows(
        lineage_rows,["model_name","pump_mode","condition","state_block"],
        ["fixed_reg_norm_mean","ordinary_norm_mean","fixed_reg_to_ordinary_norm_ratio",
         "fixed_reg_cls_cos_mean","fixed_reg_cls_cos_absmean","cos_intact_reg_mean",
         "cos_intact_mu1","cos_intact_mu2","intact_rank2_fraction",
         "pre_ln1_cls_ch437","pre_ln1_cls_ch565","pre_ln1_cls_ch650",
         "pre_ln1_fixed_reg_ch650_absmean","pre_ln1_ordinary_ch650_absmean",
         "pre_ln1_fixed_reg_ch437_absmean","pre_ln1_fixed_reg_ch565_absmean",
         "final_dynamic_register_count","final_dynamic_vs_fixed_jaccard","final_dynamic_norm_mean"]
    )
    save_rows(shared/"fixed_register_lineage_summary.csv",lineage_summary)
    save_rows(shared/"b13_rn_delta_reference_capture.csv",b13_delta_rows)
    delta_summary=base.aggregate_rows(b13_delta_rows,["model_name","pump_mode","condition","language"],["reference_rank4_capture","delta_norm_mean"])
    save_rows(shared/"b13_rn_delta_reference_capture_summary.csv",delta_summary)
    save_rows(shared/"surface_raw.csv",surface_raw);save_rows(shared/"ridge_raw.csv",ridge_raw);save_rows(shared/"state_jacobian_all_blocks_raw.csv",jac_raw)
    surface_summary=base.aggregate_rows(surface_raw,["model_name","pump_mode","condition","language","query_mode","basis_kind","pc_a","pc_b","alpha_a","alpha_b","grid_index"],["relative_calibrated","candidate_read_null_B20","candidate_read_null_B21","null_read_null_B21","early_ortho","image_attack_en_logit","image_object_en_logit","attack_minus_object_en"]);save_rows(shared/"surface_summary.csv",surface_summary)
    delta_summary=base.aggregate_rows(b13_delta_rows,["model_name","pump_mode","condition","language"],["reference_rank4_capture","delta_norm_mean"]);save_rows(shared/"b13_rn_delta_reference_capture_summary.csv",delta_summary)
    if ridge_raw:
        ridge_extrema=[];ridge_locus=[];groups=defaultdict(list)
        for r in ridge_raw:groups[(r["model_name"],r["pump_mode"])].append(r)
        for (m,p),rr in groups.items():
            eligible=[r for r in rr if bool(r.get("search_eligible",True))];ex,lo=atlas.ridge_extrema_and_loci(eligible,args.ridge_topk)
            for x in ex:x.update({"model_name":m,"pump_mode":p})
            for x in lo:x.update({"model_name":m,"pump_mode":p})
            ridge_extrema.extend(ex);ridge_locus.extend(lo)
        annotate_extrema_support_pump(ridge_extrema,ridge_coeff_lookup,support_model)
    else:ridge_extrema=[];ridge_locus=[]
    save_rows(shared/"ridge_extrema.csv",ridge_extrema);save_rows(shared/"ridge_locus_summary.csv",ridge_locus)

    means_by_block={};tangents_by_block={};all_cross_geom=[];all_cross_field=[];all_screen=[]
    for block in analysis_blocks:
        mean_states={};tangent_fields={};local_geom=[];pca_rows=[];npz_arrays={}
        for key,acc in state_acc.items():
            b,mname,pump,condition,lang,basis_kind,plane=key
            if int(b)!=int(block):continue
            mean=(acc.state_sum/acc.count).astype(np.float32);k=(mname,pump,condition,lang,basis_kind,plane);mean_states[k]=mean;G=args.grid_points;state_grid=mean.reshape(G,G,-1);tangent_fields[k]=atlas.tangent_fields_from_surface(state_grid,h)
            qlang=lang if condition=="synth" else languages[0];qm="native" if "native" in query_modes else query_modes[0];scalar=choose_surface_scalar(surface_summary,mname,pump,condition,qlang,qm,plane,basis_kind)
            if len(scalar)==G*G:
                geom,_aux=atlas.surface_local_geometry(state_grid,scalar.reshape(G,G),h)
                for r in geom:
                    rr={"state_block":int(block),"model_name":mname,"pump_mode":pump,"condition":condition,"language":lang,"basis_kind":basis_kind,"pc_a":plane[0],"pc_b":plane[1],"alpha_a":float(grid_vals[r["grid_i"]]),"alpha_b":float(grid_vals[r["grid_j"]])};rr.update(r);local_geom.append(rr)
            coords,frac=atlas.pca3_surface(mean)
            # Preserve existing 3-D viz summary; also save rank-8 centered spectrum below if available.
            for i,f in enumerate(frac,1):pca_rows.append({"state_block":int(block),"model_name":mname,"pump_mode":pump,"condition":condition,"language":lang,"basis_kind":basis_kind,"pc_a":plane[0],"pc_b":plane[1],"component":i,"explained_fraction":float(f)})
            arrname=analyze_stable_slug(f"{mname}__{pump}__{condition}__{lang}__{basis_kind}__PC{plane[0]}xPC{plane[1]}");npz_arrays[arrname]=mean
            if args.export_ply and condition=="synth" and basis_kind=="rn":atlas.write_ply_surface(ply_dir/f"B{block}__{arrname}__PCA3.ply",coords,G,grid_pairs,scalar)
        means_by_block[int(block)]=mean_states;tangents_by_block[int(block)]=tangent_fields
        cross_lang=within_model_cross_language_rows(mean_states,tangent_fields,surface_summary,model_names,pump_modes,languages,query_modes,planes)
        for r in cross_lang:r["state_block"]=int(block)
        intrinsic=[]
        for pm in pump_modes:
            ir=intrinsic_alignment_rows_for_block(mean_states,surface_summary,model_name,pm,languages,query_modes,planes,grid_vals)
            for r in ir:r["state_block"]=int(block)
            intrinsic.extend(ir)
        cross_geom=cross_pump_geometry_rows_for_block(block,mean_states,tangent_fields,surface_summary,model_name,languages,query_modes,planes) if {"intact","no_pump"}<=set(pump_modes) else []
        cross_field=cross_pump_field_rows_for_block(block,mean_states,surface_summary,model_name,languages,query_modes,planes,grid_vals) if {"intact","no_pump"}<=set(pump_modes) else []
        all_cross_geom.extend(cross_geom);all_cross_field.extend(cross_field)
        jac_block=[r for r in jac_raw if int(r.get("state_block",-1))==int(block)];jac_summary=base.aggregate_rows(jac_block,["state_block","model_name","pump_mode","condition","language","basis_kind","landmark","query_mode"],["effective_rank","sigma1","sigma2","sigma3","sigma4"]) if jac_block else []
        bdir=profile/f"B{block}"/"data";bdir.mkdir(parents=True,exist_ok=True)
        save_rows(bdir/"cross_language_geometry.csv",cross_lang);save_rows(bdir/"intrinsic_gradient_alignment.csv",intrinsic);save_rows(bdir/"intact_vs_no_pump_surface_geometry.csv",cross_geom);save_rows(bdir/"intact_vs_no_pump_read_fields.csv",cross_field);save_rows(bdir/"surface_local_geometry.csv",local_geom);save_rows(bdir/"surface_pca_spectrum.csv",pca_rows);save_rows(bdir/"state_jacobian_raw.csv",jac_block);save_rows(bdir/"state_jacobian_summary.csv",jac_summary);np.savez_compressed(bdir/f"mean_B{block}_surfaces.npz",**npz_arrays)
        # Compact screen: merge geometry and field rows by identity.
        fmap={(r["language"],int(r["pc_a"]),int(r["pc_b"]),r["query_mode"]):r for r in cross_field}
        for g in cross_geom:
            k=(g["language"],int(g["pc_a"]),int(g["pc_b"]),g["query_mode"]);r=dict(g);r.update(fmap.get(k,{}));all_screen.append(r)
    save_rows(profile/"intact_vs_no_pump_block_screen.csv",all_screen);save_rows(profile/"intact_vs_no_pump_surface_geometry_all_blocks.csv",all_cross_geom);save_rows(profile/"intact_vs_no_pump_read_fields_all_blocks.csv",all_cross_field)
    plot_pump_ablation_overview(all_screen,lineage_cross,plots/"register_pump_ablation_block_overview.png")

    # Existing within-condition temporal transitions are still useful; save separately per pump mode.
    for pm in pump_modes:
        tr=consecutive_block_transition_rows(means_by_block,tangents_by_block,languages,planes,model_name,pm);save_rows(profile/f"block_transition_summary__{pm}.csv",tr);plot_block_transition_overview(tr,plots/f"block_transition_overview__{pm}.png")

    config={"checkpoint":args.checkpoint,"languages":languages,"query_modes":query_modes,"pump_modes":pump_modes,"ridge_modes":sorted(ridge_modes),"analysis_blocks":analysis_blocks,"profile":profile_name,"fast_ridge":bool(args.fast_ridge),"grid_points":args.grid_points,"ridge_points":args.ridge_points,"register_neurons":REG_NEURONS,"fixed_register_policy":"intact B12 identities frozen and reused under no_pump","basis_policy":"intact B13 RN rank-4 basis frozen and reused under no_pump","no_pump_definition":"selected post-QuickGELU MLP units zeroed before c_proj","rail_channels_passive_only":list(RAIL_CHANNELS),"export_ply":bool(args.export_ply)};save_json(profile/"config.json",config)
    (profile/"SUMMARY.txt").write_text("\n".join(["RN REGISTER-PUMP MULTIBLOCK CONTROL-SHEET AUDIT","="*72,"",f"Modes: {pump_modes}",f"Blocks: {analysis_blocks}",f"Grid: {args.grid_points}x{args.grid_points}",f"Ridge modes: {sorted(ridge_modes)}","Fixed coordinate contract: intact B13 RN rank-4 basis.","Fixed register lineage: intact B12 register identities reused under no_pump.","Passive rails: residual/LN1 channels 437,565,650 only; no channel ablations.","Primary comparison: intact_vs_no_pump_block_screen.csv"])+"\n",encoding="utf-8")

    zpath=out/("compact_summary_rn_control_surfaces_fast.zip" if args.fast_ridge else "compact_summary_rn_control_surfaces_full.zip")
    if zpath.exists():zpath.unlink()
    compact=[profile/"config.json",profile/"SUMMARY.txt",profile/"intact_vs_no_pump_block_screen.csv",profile/"intact_vs_no_pump_surface_geometry_all_blocks.csv",profile/"intact_vs_no_pump_read_fields_all_blocks.csv",shared/"register_pump_intervention_audit.csv",shared/"register_pump_call_log.csv",shared/"fixed_register_lineage.csv",shared/"fixed_register_lineage_summary.csv",shared/"fixed_register_mu_basis_summary.csv",shared/"fixed_register_intact_vs_no_pump.csv",shared/"b13_rn_delta_reference_capture_summary.csv",shared/"surface_summary.csv",shared/"ridge_extrema.csv",shared/"ridge_locus_summary.csv",shared/"empirical_support_summary.json",shared/"reference_basis.npz"]+sorted(plots.glob("*.png"))
    for b in analysis_blocks:
        d=profile/f"B{b}"/"data";compact += [d/"cross_language_geometry.csv",d/"intrinsic_gradient_alignment.csv",d/"intact_vs_no_pump_surface_geometry.csv",d/"intact_vs_no_pump_read_fields.csv",d/"surface_local_geometry.csv",d/"surface_pca_spectrum.csv",d/"state_jacobian_summary.csv",d/f"mean_B{b}_surfaces.npz"]
    with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in compact:
            if p.exists() and p.is_file():z.write(p,arcname=p.relative_to(out).as_posix())
    del model,loaded;_paper_cleanup(empty_cache=True)
    print("\n[done]",profile.resolve());print("[compact summary]",zpath.resolve())


# FLOW MAPS
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Ellipse


SCRIPT_MAP = {
    "en": "Latin",
    "de": "Latin",
    "es": "Latin",
    "fr": "Latin",
    "it": "Latin",
    "pt": "Latin",
    "ru": "Cyrillic",
    "uk": "Cyrillic",
    "bg": "Cyrillic",
    "sr": "Cyrillic",
    "ar": "Arabic",
    "fa": "Arabic",
    "ur": "Arabic",
    "ko": "Hangul",
    "zh": "Hanzi",
    "ja": "KanaKanji",
}


def flow_maps_stable_slug(x: str) -> str:
    x = str(x).strip().replace(" ", "_")
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "item"


def maybe_extract_input(inp: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if inp.is_dir():
        return inp, None
    if inp.is_file() and inp.suffix.lower() == ".zip":
        td = tempfile.TemporaryDirectory(prefix="rn_read_flow_")
        with zipfile.ZipFile(inp, "r") as zf:
            zf.extractall(td.name)
        root = Path(td.name)
        # Common layouts: older single-block root contains data/; newer multi-block
        # archives contain _ridge_fast/ or _ridge_full/ with shared/ + B*/data/.
        if (root / "data").is_dir() or (root / "shared" / "surface_summary.csv").is_file():
            return root, td
        kids = [p for p in root.iterdir() if p.is_dir()]
        for kid in kids:
            if (kid / "data").is_dir() or (kid / "shared" / "surface_summary.csv").is_file():
                return kid, td
        # Keep extraction root if a nested profile is discoverable recursively.
        if any(root.rglob("shared/surface_summary.csv")):
            return root, td
        raise FileNotFoundError(f"Could not find supported result layout inside extracted archive: {inp}")
    raise FileNotFoundError(f"Input path is neither directory nor zip: {inp}")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    """Read a postprocess table that is allowed to have zero rows.

    Several cross-language tables are structurally empty for one-language smoke
    runs.  ``save_rows([])`` produces an empty file, which pandas reports as
    ``EmptyDataError``.  That is a valid "no pairs" result, not corruption.
    """
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_required_tables(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    surface = pd.read_csv(root / "data" / "surface_summary.csv")
    cross_lang = _read_optional_csv(root / "data" / "cross_language_geometry.csv")
    return surface, cross_lang


def parse_languages(df: pd.DataFrame) -> List[str]:
    return sorted(map(str, df["language"].dropna().unique().tolist()))


def flow_maps_parse_planes(df: pd.DataFrame) -> List[Tuple[int, int]]:
    keys = sorted({(int(a), int(b)) for a, b in zip(df["pc_a"], df["pc_b"])})
    return keys


def parse_queries(df: pd.DataFrame) -> List[str]:
    return sorted(map(str, df["query_mode"].dropna().unique().tolist()))


def infer_scripts(languages: Sequence[str]) -> Dict[str, str]:
    return {lang: SCRIPT_MAP.get(lang, f"OTHER:{lang}") for lang in languages}


class SurfaceBundle:
    def __init__(self, a_vals: np.ndarray, b_vals: np.ndarray, scalar: np.ndarray):
        self.a_vals = np.asarray(a_vals, np.float64)
        self.b_vals = np.asarray(b_vals, np.float64)
        self.scalar = np.asarray(scalar, np.float64)
        self.ga: Optional[np.ndarray] = None
        self.gb: Optional[np.ndarray] = None
        self.mag: Optional[np.ndarray] = None
        self.iga: Optional[np.ndarray] = None
        self.igb: Optional[np.ndarray] = None
        self.imag: Optional[np.ndarray] = None
        self.G11: Optional[np.ndarray] = None
        self.G12: Optional[np.ndarray] = None
        self.G22: Optional[np.ndarray] = None

    @property
    def X(self) -> np.ndarray:
        # Horizontal axis = alpha_b
        return np.meshgrid(self.b_vals, self.a_vals)[0]

    @property
    def Y(self) -> np.ndarray:
        return np.meshgrid(self.b_vals, self.a_vals)[1]


def build_surface_grid(
    df: pd.DataFrame,
    language: str,
    query_mode: str,
    plane: Tuple[int, int],
    model_name: Optional[str],
    pump_mode: Optional[str],
    condition: str = "synth",
    basis_kind: str = "rn",
    scalar_col: str = "relative_calibrated_mean",
) -> SurfaceBundle:
    sub = df.copy()
    sub = sub[sub["condition"].astype(str) == condition]
    sub = sub[sub["basis_kind"].astype(str) == basis_kind]
    sub = sub[sub["language"].astype(str) == language]
    sub = sub[sub["query_mode"].astype(str) == query_mode]
    sub = sub[(sub["pc_a"].astype(int) == int(plane[0])) & (sub["pc_b"].astype(int) == int(plane[1]))]
    if model_name is not None:
        sub = sub[sub["model_name"].astype(str) == model_name]
    if pump_mode is not None:
        sub = sub[sub["pump_mode"].astype(str) == pump_mode]
    if sub.empty:
        raise KeyError(f"No rows for language={language}, query={query_mode}, plane={plane}")

    a_vals = np.sort(sub["alpha_a"].astype(float).unique())
    b_vals = np.sort(sub["alpha_b"].astype(float).unique())
    piv = sub.pivot_table(index="alpha_a", columns="alpha_b", values=scalar_col, aggfunc="mean")
    piv = piv.reindex(index=a_vals, columns=b_vals)
    scalar = piv.to_numpy(dtype=np.float64)
    if np.isnan(scalar).any():
        raise ValueError(f"NaN surface grid for {language=} {query_mode=} {plane=}")
    return SurfaceBundle(a_vals=a_vals, b_vals=b_vals, scalar=scalar)


def compute_euclidean_gradient(bundle: SurfaceBundle, edge_order: int = 2) -> None:
    ga, gb = np.gradient(bundle.scalar, bundle.a_vals, bundle.b_vals, edge_order=edge_order)
    bundle.ga = ga
    bundle.gb = gb
    bundle.mag = np.sqrt(ga * ga + gb * gb)


def parse_mean_state_npz(npz_path: Path) -> Dict[Tuple[str, str, str, str, str, int, int], np.ndarray]:
    arrays = np.load(npz_path)
    out: Dict[Tuple[str, str, str, str, str, int, int], np.ndarray] = {}
    pat = re.compile(r"^(.*?)__(.*?)__(.*?)__(.*?)__(.*?)__PC(\d+)x(?:PC)?(\d+)$")
    for key in arrays.files:
        m = pat.match(key)
        if not m:
            # ignore unrecognized names
            continue
        model_name, pump_mode, condition, language, basis_kind, pa, pb = m.groups()
        out[(model_name, pump_mode, condition, language, basis_kind, int(pa), int(pb))] = arrays[key]
    return out


def compute_intrinsic_geometry_and_gradient(
    bundle: SurfaceBundle,
    state_surface: np.ndarray,
    edge_order: int = 2,
    ridge_eps: float = 1e-6,
) -> None:
    state_surface = np.asarray(state_surface, np.float64)
    H, W, D = state_surface.shape
    if H != len(bundle.a_vals) or W != len(bundle.b_vals):
        raise ValueError(
            f"State surface shape {state_surface.shape} incompatible with scalar grid {(len(bundle.a_vals), len(bundle.b_vals))}"
        )
    if bundle.ga is None or bundle.gb is None:
        compute_euclidean_gradient(bundle, edge_order=edge_order)

    dXa = np.gradient(state_surface, bundle.a_vals, axis=0, edge_order=edge_order)
    dXb = np.gradient(state_surface, bundle.b_vals, axis=1, edge_order=edge_order)
    G11 = np.sum(dXa * dXa, axis=-1)
    G12 = np.sum(dXa * dXb, axis=-1)
    G22 = np.sum(dXb * dXb, axis=-1)
    det = G11 * G22 - G12 * G12
    det = np.where(det < ridge_eps, det + ridge_eps, det)

    # Intrinsic gradient: G^{-1} [d/d alpha_a, d/d alpha_b]^T.
    iga = (G22 * bundle.ga - G12 * bundle.gb) / det
    igb = (-G12 * bundle.ga + G11 * bundle.gb) / det

    bundle.G11 = G11
    bundle.G12 = G12
    bundle.G22 = G22
    bundle.iga = iga
    bundle.igb = igb
    bundle.imag = np.sqrt(iga * iga + igb * igb)


def normalized_components(v0: np.ndarray, v1: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag = np.sqrt(v0 * v0 + v1 * v1)
    return v0 / (mag + eps), v1 / (mag + eps), mag


def pairwise_gradient_cos(
    v0a: np.ndarray, v1a: np.ndarray, v0b: np.ndarray, v1b: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    na = np.sqrt(v0a * v0a + v1a * v1a)
    nb = np.sqrt(v0b * v0b + v1b * v1b)
    return (v0a * v0b + v1a * v1b) / (na * nb + eps)


def draw_metric_ellipses(ax, bundle: SurfaceBundle, step: int = 2, scale: float = 0.28, alpha: float = 0.45):
    if bundle.G11 is None or bundle.G12 is None or bundle.G22 is None:
        return
    db = float(np.median(np.diff(bundle.b_vals))) if len(bundle.b_vals) > 1 else 1.0
    da = float(np.median(np.diff(bundle.a_vals))) if len(bundle.a_vals) > 1 else 1.0
    base = min(abs(da), abs(db))
    for i in range(0, len(bundle.a_vals), step):
        for j in range(0, len(bundle.b_vals), step):
            G = np.array([[bundle.G11[i, j], bundle.G12[i, j]], [bundle.G12[i, j], bundle.G22[i, j]]], dtype=np.float64)
            try:
                evals, evecs = np.linalg.eigh(G)
            except np.linalg.LinAlgError:
                continue
            evals = np.clip(evals, 1e-12, None)
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
            # Use sqrt(eigenvalue) for local tangent length scale.
            width = scale * base * float(np.sqrt(evals[0]))
            height = scale * base * float(np.sqrt(evals[1]))
            angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
            ell = Ellipse(
                xy=(bundle.b_vals[j], bundle.a_vals[i]),
                width=width,
                height=height,
                angle=angle,
                fill=False,
                lw=0.55,
                alpha=alpha,
                edgecolor="black",
            )
            ax.add_patch(ell)


def common_axes_setup(ax, bundle: SurfaceBundle, plane: Tuple[int, int], query_mode: str, title: str):
    ax.set_xlabel(f"PC{plane[1]} coefficient α{plane[1]}")
    ax.set_ylabel(f"PC{plane[0]} coefficient α{plane[0]}")
    ax.set_title(title)
    ax.set_xlim(float(bundle.b_vals.min()), float(bundle.b_vals.max()))
    ax.set_ylim(float(bundle.a_vals.min()), float(bundle.a_vals.max()))


def _map_to_rgb(values: np.ndarray, cmap_name: str = "coolwarm", vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    arr = np.asarray(values, np.float64)
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        norm = np.zeros_like(arr, dtype=np.float64)
    else:
        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm)
    rgb = np.round(255.0 * rgba[..., :3]).astype(np.uint8)
    return rgb


def write_ascii_ply(
    out_path: Path,
    vertices: np.ndarray,
    faces: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    extra_vertex_props: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    ensure_dir(out_path.parent)
    V = np.asarray(vertices, np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"vertices must be [N,3], got {V.shape}")
    F = None if faces is None else np.asarray(faces, np.int64)
    C = None if colors is None else np.asarray(colors)
    if C is not None and (C.shape[0] != V.shape[0] or C.shape[1] != 3):
        raise ValueError(f"colors must be [N,3], got {C.shape}")
    extras = extra_vertex_props or {}
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {V.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        for name in extras.keys():
            f.write(f"property float {name}\n")
        if C is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        if F is not None:
            f.write(f"element face {F.shape[0]}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(V.shape[0]):
            row = [f"{float(V[i,0]):.8f}", f"{float(V[i,1]):.8f}", f"{float(V[i,2]):.8f}"]
            for name, arr in extras.items():
                row.append(f"{float(np.asarray(arr)[i]):.8f}")
            if C is not None:
                row.extend([str(int(C[i,0])), str(int(C[i,1])), str(int(C[i,2]))])
            f.write(" ".join(row) + "\n")
        if F is not None:
            for tri in F:
                f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def export_surface_ply(out_path: Path, bundle: SurfaceBundle) -> None:
    H = len(bundle.a_vals)
    W = len(bundle.b_vals)
    X = bundle.X
    Y = bundle.Y
    Z = bundle.scalar
    vertices = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
    colors = _map_to_rgb(Z.reshape(-1), cmap_name='coolwarm')
    alpha_a = Y.reshape(-1)
    alpha_b = X.reshape(-1)
    relative_read = Z.reshape(-1)
    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            idx = i * W + j
            v00 = idx
            v01 = idx + 1
            v10 = idx + W
            v11 = idx + W + 1
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.asarray(faces, np.int64)
    write_ascii_ply(
        out_path,
        vertices,
        faces=faces,
        colors=colors,
        extra_vertex_props={'alpha_a': alpha_a, 'alpha_b': alpha_b, 'relative_read': relative_read},
    )


def _surface_tangent_frame(bundle: SurfaceBundle, i: int, j: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Surface parameterization: S(alpha_a, alpha_b) = [alpha_b, alpha_a, READ].
    if bundle.ga is None or bundle.gb is None:
        raise ValueError("Euclidean gradient must be computed before tangent export")
    dS_da = np.array([0.0, 1.0, float(bundle.ga[i, j])], dtype=np.float64)
    dS_db = np.array([1.0, 0.0, float(bundle.gb[i, j])], dtype=np.float64)
    n = np.cross(dS_db, dS_da)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = n / n_norm
    return dS_da, dS_db, n


def export_field_tangent_ply(
    out_path: Path,
    bundle: SurfaceBundle,
    comp0: np.ndarray,
    comp1: np.ndarray,
    normalized: bool,
    constant_size_scale: float = 0.22,
    width_ratio: float = 0.40,
    normal_lift: float = 1e-3,
) -> None:
    H = len(bundle.a_vals)
    W = len(bundle.b_vals)
    U = np.asarray(comp1, np.float64)  # horizontal / alpha_b
    V = np.asarray(comp0, np.float64)  # vertical / alpha_a
    mag = np.sqrt(U * U + V * V)
    max_mag = max(float(np.nanmax(mag)), 1e-12)
    db = float(np.median(np.diff(bundle.b_vals))) if W > 1 else 1.0
    da = float(np.median(np.diff(bundle.a_vals))) if H > 1 else 1.0
    base = min(abs(da), abs(db))
    base_len = constant_size_scale * base

    verts = []
    cols = []
    faces = []
    v_alpha_a = []
    v_alpha_b = []
    v_relative_read = []
    v_vec_a = []
    v_vec_b = []
    v_vec_mag = []
    face_idx = 0
    field_colors = _map_to_rgb(mag, cmap_name='viridis', vmin=0.0, vmax=max_mag)
    for i in range(H):
        for j in range(W):
            m = float(mag[i, j])
            if m <= 1e-14:
                continue
            x = float(bundle.b_vals[j])
            y = float(bundle.a_vals[i])
            z = float(bundle.scalar[i, j])
            dS_da, dS_db, n = _surface_tangent_frame(bundle, i, j)
            v3 = float(V[i, j]) * dS_da + float(U[i, j]) * dS_db
            v3_norm = np.linalg.norm(v3)
            if v3_norm <= 1e-14:
                continue
            direction = v3 / v3_norm
            perp = np.cross(n, direction)
            perp_norm = np.linalg.norm(perp)
            if perp_norm <= 1e-14:
                # fallback: any vector orthogonal to direction
                fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                perp = np.cross(fallback, direction)
                perp_norm = np.linalg.norm(perp)
                if perp_norm <= 1e-14:
                    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    perp = np.cross(fallback, direction)
                    perp_norm = np.linalg.norm(perp)
            perp = perp / max(perp_norm, 1e-14)
            if normalized:
                scale = 1.0
            else:
                scale = 0.20 + 0.80 * (m / max_mag)
            length = base_len * scale
            width = width_ratio * length
            center = np.array([x, y, z], dtype=np.float64) + normal_lift * n
            tip = center + 0.55 * length * direction
            back = center - 0.25 * length * direction
            left = back + 0.5 * width * perp
            right = back - 0.5 * width * perp
            tri = np.stack([tip, left, right], axis=0)
            color = np.tile(field_colors[i, j][None, :], (3, 1))
            verts.append(tri)
            cols.append(color)
            faces.append([face_idx, face_idx + 1, face_idx + 2])
            for _ in range(3):
                v_alpha_a.append(y)
                v_alpha_b.append(x)
                v_relative_read.append(float(bundle.scalar[i, j]))
                v_vec_a.append(float(V[i, j]))
                v_vec_b.append(float(U[i, j]))
                v_vec_mag.append(m)
            face_idx += 3
    if not verts:
        return
    vertices = np.concatenate(verts, axis=0)
    colors = np.concatenate(cols, axis=0)
    face_arr = np.asarray(faces, np.int64)
    write_ascii_ply(
        out_path,
        vertices,
        faces=face_arr,
        colors=colors,
        extra_vertex_props={
            'alpha_a': np.asarray(v_alpha_a, np.float64),
            'alpha_b': np.asarray(v_alpha_b, np.float64),
            'relative_read': np.asarray(v_relative_read, np.float64),
            'vector_alpha_a': np.asarray(v_vec_a, np.float64),
            'vector_alpha_b': np.asarray(v_vec_b, np.float64),
            'vector_magnitude': np.asarray(v_vec_mag, np.float64),
        },
    )


def export_field_ply(
    out_path: Path,
    bundle: SurfaceBundle,
    comp0: np.ndarray,
    comp1: np.ndarray,
    normalized: bool,
    height_offset: float = 0.0,
    constant_size_scale: float = 0.22,
    width_ratio: float = 0.40,
) -> None:
    H = len(bundle.a_vals)
    W = len(bundle.b_vals)
    U = np.asarray(comp1, np.float64)  # horizontal / alpha_b
    V = np.asarray(comp0, np.float64)  # vertical / alpha_a
    mag = np.sqrt(U * U + V * V)
    max_mag = max(float(np.nanmax(mag)), 1e-12)
    db = float(np.median(np.diff(bundle.b_vals))) if W > 1 else 1.0
    da = float(np.median(np.diff(bundle.a_vals))) if H > 1 else 1.0
    base = min(abs(da), abs(db))
    base_len = constant_size_scale * base

    verts = []
    cols = []
    faces = []
    v_alpha_a = []
    v_alpha_b = []
    v_relative_read = []
    v_vec_a = []
    v_vec_b = []
    v_vec_mag = []
    face_idx = 0
    field_colors = _map_to_rgb(mag, cmap_name='viridis', vmin=0.0, vmax=max_mag)
    for i in range(H):
        for j in range(W):
            m = float(mag[i, j])
            if m <= 1e-14:
                continue
            x = float(bundle.b_vals[j])
            y = float(bundle.a_vals[i])
            z = float(bundle.scalar[i, j] + height_offset)
            du = float(U[i, j]) / m
            dv = float(V[i, j]) / m
            if normalized:
                scale = 1.0
            else:
                scale = 0.20 + 0.80 * (m / max_mag)
            length = base_len * scale
            width = width_ratio * length
            tip = np.array([x + 0.55 * length * du, y + 0.55 * length * dv, z], dtype=np.float64)
            back = np.array([x - 0.25 * length * du, y - 0.25 * length * dv, z], dtype=np.float64)
            perp = np.array([-dv, du, 0.0], dtype=np.float64)
            left = back + 0.5 * width * perp
            right = back - 0.5 * width * perp
            tri = np.stack([tip, left, right], axis=0)
            color = np.tile(field_colors[i, j][None, :], (3, 1))
            verts.append(tri)
            cols.append(color)
            faces.append([face_idx, face_idx + 1, face_idx + 2])
            for _ in range(3):
                v_alpha_a.append(y)
                v_alpha_b.append(x)
                v_relative_read.append(float(bundle.scalar[i, j]))
                v_vec_a.append(float(V[i, j]))
                v_vec_b.append(float(U[i, j]))
                v_vec_mag.append(m)
            face_idx += 3
    if not verts:
        return
    vertices = np.concatenate(verts, axis=0)
    colors = np.concatenate(cols, axis=0)
    face_arr = np.asarray(faces, np.int64)
    write_ascii_ply(
        out_path,
        vertices,
        faces=face_arr,
        colors=colors,
        extra_vertex_props={
            'alpha_a': np.asarray(v_alpha_a, np.float64),
            'alpha_b': np.asarray(v_alpha_b, np.float64),
            'relative_read': np.asarray(v_relative_read, np.float64),
            'vector_alpha_a': np.asarray(v_vec_a, np.float64),
            'vector_alpha_b': np.asarray(v_vec_b, np.float64),
            'vector_magnitude': np.asarray(v_vec_mag, np.float64),
        },
    )


def export_surface_and_field_ply(
    out_prefix: Path,
    bundle: SurfaceBundle,
    comp0: np.ndarray,
    comp1: np.ndarray,
    normalized: bool,
) -> None:
    surface_path = out_prefix.with_suffix('.ply')
    field_path = out_prefix.with_name(out_prefix.stem + '_field').with_suffix('.ply')
    tangent_field_path = out_prefix.with_name(out_prefix.stem + '_field_tangent').with_suffix('.ply')
    export_surface_ply(surface_path, bundle)
    export_field_ply(field_path, bundle, comp0=comp0, comp1=comp1, normalized=normalized)
    export_field_tangent_ply(tangent_field_path, bundle, comp0=comp0, comp1=comp1, normalized=normalized)


def plot_surface_stream(
    out_path: Path,
    bundle: SurfaceBundle,
    comp0: np.ndarray,
    comp1: np.ndarray,
    plane: Tuple[int, int],
    title: str,
    normalized: bool = True,
    metric_ellipses: bool = False,
):
    ensure_dir(out_path.parent)
    U = comp1  # horizontal is alpha_b
    V = comp0  # vertical is alpha_a
    if normalized:
        Vn, Un, mag = normalized_components(V, U)
    else:
        Vn, Un = V, U
        mag = np.sqrt(V * V + U * U)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    cf = ax.contourf(bundle.X, bundle.Y, bundle.scalar, levels=16, cmap="coolwarm")
    ax.contour(bundle.X, bundle.Y, bundle.scalar, levels=10, colors="k", linewidths=0.35, alpha=0.35)
    if metric_ellipses:
        draw_metric_ellipses(ax, bundle, step=2, scale=0.30, alpha=0.35)
    lw = 0.6 + 1.7 * (mag / max(float(np.nanmax(mag)), 1e-8))
    ax.streamplot(bundle.X, bundle.Y, Un, Vn, density=1.05, color="black", linewidth=lw, arrowsize=0.9)
    cb = fig.colorbar(cf, ax=ax, fraction=0.047, pad=0.02)
    cb.set_label("relative READ")
    common_axes_setup(ax, bundle, plane, "", title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_surface_quiver(
    out_path: Path,
    bundle: SurfaceBundle,
    comp0: np.ndarray,
    comp1: np.ndarray,
    plane: Tuple[int, int],
    title: str,
    normalized: bool = True,
    stride: int = 1,
    metric_ellipses: bool = False,
):
    ensure_dir(out_path.parent)
    U = comp1  # horizontal
    V = comp0  # vertical
    if normalized:
        Vp, Up, mag = normalized_components(V, U)
        scale = 18
    else:
        Vp, Up = V, U
        mag = np.sqrt(V * V + U * U)
        scale = None

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    im = ax.imshow(
        bundle.scalar,
        origin="lower",
        extent=[bundle.b_vals.min(), bundle.b_vals.max(), bundle.a_vals.min(), bundle.a_vals.max()],
        aspect="auto",
        cmap="coolwarm",
        alpha=0.92,
    )
    ax.contour(bundle.X, bundle.Y, bundle.scalar, levels=10, colors="k", linewidths=0.3, alpha=0.35)
    if metric_ellipses:
        draw_metric_ellipses(ax, bundle, step=2, scale=0.30, alpha=0.35)
    ax.quiver(
        bundle.X[::stride, ::stride],
        bundle.Y[::stride, ::stride],
        Up[::stride, ::stride],
        Vp[::stride, ::stride],
        mag[::stride, ::stride],
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=0.006,
        pivot="mid",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.02)
    cb.set_label("relative READ")
    common_axes_setup(ax, bundle, plane, "", title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_heatmaps(
    out_path: Path,
    bundle_lookup: Dict[str, SurfaceBundle],
    languages: Sequence[str],
    scripts: Dict[str, str],
    plane: Tuple[int, int],
    query_mode: str,
    intrinsic: bool = False,
):
    ensure_dir(out_path.parent)
    langs = [l for l in languages if l in bundle_lookup]
    same_pairs = [(a, b) for i, a in enumerate(langs) for b in langs[i+1:] if scripts[a] == scripts[b]]
    cross_pairs = [(a, b) for i, a in enumerate(langs) for b in langs[i+1:] if scripts[a] != scripts[b]]
    if not same_pairs or not cross_pairs:
        return

    def comps(lang: str) -> Tuple[np.ndarray, np.ndarray]:
        b = bundle_lookup[lang]
        if intrinsic:
            assert b.iga is not None and b.igb is not None
            return b.iga, b.igb
        assert b.ga is not None and b.gb is not None
        return b.ga, b.gb

    same_maps = []
    for a, b in same_pairs:
        a0, a1 = comps(a)
        b0, b1 = comps(b)
        same_maps.append(pairwise_gradient_cos(a0, a1, b0, b1))
    cross_maps = []
    for a, b in cross_pairs:
        a0, a1 = comps(a)
        b0, b1 = comps(b)
        cross_maps.append(pairwise_gradient_cos(a0, a1, b0, b1))

    same_mean = np.mean(np.stack(same_maps, axis=0), axis=0)
    cross_mean = np.mean(np.stack(cross_maps, axis=0), axis=0)
    diff = same_mean - cross_mean

    # Group-average surfaces for descriptive flow overlays.
    latin_langs = [l for l in langs if scripts.get(l) == "Latin"]
    other_langs = [l for l in langs if scripts.get(l) != "Latin"]
    base = bundle_lookup[langs[0]]

    def mean_surface_and_grad(group_langs: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gs = []
        v0s = []
        v1s = []
        for l in group_langs:
            b = bundle_lookup[l]
            gs.append(b.scalar)
            if intrinsic:
                v0s.append(b.iga)
                v1s.append(b.igb)
            else:
                v0s.append(b.ga)
                v1s.append(b.gb)
        return np.mean(np.stack(gs, axis=0), axis=0), np.mean(np.stack(v0s, axis=0), axis=0), np.mean(np.stack(v1s, axis=0), axis=0)

    latin_scalar, latin_v0, latin_v1 = mean_surface_and_grad(latin_langs)
    other_scalar, other_v0, other_v1 = mean_surface_and_grad(other_langs)

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.2), dpi=150)

    for ax, scalar, v0, v1, ttl in [
        (axes[0, 0], latin_scalar, latin_v0, latin_v1, "mean same-script (Latin) READ flow"),
        (axes[0, 1], other_scalar, other_v0, other_v1, "mean cross-script-side (non-Latin) READ flow"),
    ]:
        U = v1
        V = v0
        Vn, Un, mag = normalized_components(V, U)
        cf = ax.contourf(base.X, base.Y, scalar, levels=16, cmap="coolwarm")
        ax.contour(base.X, base.Y, scalar, levels=10, colors="k", linewidths=0.35, alpha=0.35)
        lw = 0.6 + 1.7 * (mag / max(float(np.nanmax(mag)), 1e-8))
        ax.streamplot(base.X, base.Y, Un, Vn, density=1.0, color="black", linewidth=lw, arrowsize=0.9)
        common_axes_setup(ax, base, plane, query_mode, ttl)
    fig.colorbar(cf, ax=[axes[0, 0], axes[0, 1]], fraction=0.026, pad=0.02, label="relative READ")

    for ax, arr, ttl in [
        (axes[0, 2], same_mean, "within same-script gradient cosine"),
        (axes[1, 0], cross_mean, "cross-script gradient cosine"),
        (axes[1, 1], diff, "same-script minus cross-script"),
    ]:
        im = ax.imshow(
            arr,
            origin="lower",
            extent=[base.b_vals.min(), base.b_vals.max(), base.a_vals.min(), base.a_vals.max()],
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        common_axes_setup(ax, base, plane, query_mode, ttl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Panel: pairwise mean alignment matrix.
    mat = np.full((len(langs), len(langs)), np.nan, dtype=np.float64)
    for i, a in enumerate(langs):
        a0, a1 = comps(a)
        for j, b in enumerate(langs):
            b0, b1 = comps(b)
            mat[i, j] = float(np.nanmean(pairwise_gradient_cos(a0, a1, b0, b1)))
    ax = axes[1, 2]
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs, rotation=45, ha="right")
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs)
    ax.set_title("pairwise mean gradient cosine")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    kind = "intrinsic" if intrinsic else "euclidean"
    fig.suptitle(f"{kind} READ-flow comparison | PC{plane[0]}×PC{plane[1]} | {query_mode}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_matrices(
    out_path: Path,
    bundle_lookup: Dict[str, SurfaceBundle],
    languages: Sequence[str],
    plane: Tuple[int, int],
    query_mode: str,
    intrinsic: bool,
    cross_lang_df: pd.DataFrame,
):
    ensure_dir(out_path.parent)
    langs = [l for l in languages if l in bundle_lookup]
    N = len(langs)
    grad_mat = np.full((N, N), np.nan, dtype=np.float64)

    def comps(lang: str) -> Tuple[np.ndarray, np.ndarray]:
        b = bundle_lookup[lang]
        if intrinsic:
            assert b.iga is not None and b.igb is not None
            return b.iga, b.igb
        assert b.ga is not None and b.gb is not None
        return b.ga, b.gb

    for i, a in enumerate(langs):
        a0, a1 = comps(a)
        for j, b in enumerate(langs):
            b0, b1 = comps(b)
            grad_mat[i, j] = float(np.nanmean(pairwise_gradient_cos(a0, a1, b0, b1)))

    scalar_mat = np.full((N, N), np.nan, dtype=np.float64)
    required_cross_cols = {
        "pc_a", "pc_b", "query_mode", "language_a", "language_b", "read_surface_spearman"
    }
    if cross_lang_df.empty or not required_cross_cols.issubset(cross_lang_df.columns):
        # A one-language run has no cross-language pairs by definition.  Keep an
        # empty typed table so the diagonal can still be rendered as identity.
        sub = pd.DataFrame(columns=sorted(required_cross_cols))
    else:
        sub = cross_lang_df.copy()
        sub = sub[(sub["pc_a"].astype(int) == int(plane[0])) & (sub["pc_b"].astype(int) == int(plane[1]))]
        sub = sub[sub["query_mode"].astype(str) == query_mode]
    for i, a in enumerate(langs):
        scalar_mat[i, i] = 1.0
        for j, b in enumerate(langs):
            if i == j:
                continue
            row = sub[(sub["language_a"].astype(str) == a) & (sub["language_b"].astype(str) == b)]
            if row.empty:
                row = sub[(sub["language_a"].astype(str) == b) & (sub["language_b"].astype(str) == a)]
            if not row.empty:
                scalar_mat[i, j] = float(row["read_surface_spearman"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=150)
    for ax, arr, ttl in [
        (axes[0], scalar_mat, "READ-surface Spearman"),
        (axes[1], grad_mat, f"{'intrinsic' if intrinsic else 'euclidean'} gradient mean cosine"),
    ]:
        im = ax.imshow(arr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(N)); ax.set_xticklabels(langs, rotation=45, ha="right")
        ax.set_yticks(range(N)); ax.set_yticklabels(langs)
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Pairwise READ similarity | PC{plane[0]}×PC{plane[1]} | {query_mode}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_ellipse_sheet(
    out_path: Path,
    bundle: SurfaceBundle,
    plane: Tuple[int, int],
    title: str,
    use_intrinsic: bool = True,
):
    if bundle.G11 is None or bundle.G12 is None or bundle.G22 is None:
        return
    ensure_dir(out_path.parent)
    if use_intrinsic:
        comp0, comp1 = bundle.iga, bundle.igb
        kind = "intrinsic"
    else:
        comp0, comp1 = bundle.ga, bundle.gb
        kind = "euclidean"
    assert comp0 is not None and comp1 is not None
    U = comp1
    V = comp0
    Vn, Un, mag = normalized_components(V, U)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    cf = ax.contourf(bundle.X, bundle.Y, bundle.scalar, levels=16, cmap="coolwarm", alpha=0.88)
    draw_metric_ellipses(ax, bundle, step=2, scale=0.30, alpha=0.42)
    lw = 0.6 + 1.7 * (mag / max(float(np.nanmax(mag)), 1e-8))
    ax.streamplot(bundle.X, bundle.Y, Un, Vn, density=1.0, color="black", linewidth=lw, arrowsize=0.9)
    fig.colorbar(cf, ax=ax, fraction=0.047, pad=0.02, label="relative READ")
    common_axes_setup(ax, bundle, plane, "", f"{title} ({kind} READ flow + metric ellipses)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_readme(root: Path, has_intrinsic: bool, languages: Sequence[str], planes: Sequence[Tuple[int, int]], queries: Sequence[str]):
    text = []
    text.append("READ flow figure pack\n")
    text.append("Main output uses normalized vector fields. Curiosity/raw-length versions live under you_asked_for_it/.\n")
    text.append(f"Languages: {', '.join(languages)}\n")
    text.append(f"Planes: {', '.join([f'PC{a}xPC{b}' for a,b in planes])}\n")
    text.append(f"Query modes: {', '.join(queries)}\n")
    text.append("\nNormal output:\n")
    text.append("  euclidean/: normalized READ gradient over control coordinates\n")
    text.append("    blender_ply/: overlapping surface PLY + _field.ply + _field_tangent.ply vector glyphs for Blender\n")
    if has_intrinsic:
        text.append("  intrinsic/: normalized G^{-1}∇READ using the B21 pullback metric\n")
        text.append("    blender_ply/: overlapping surface PLY + _field.ply + _field_tangent.ply vector glyphs for Blender\n")
        text.append("  metric_ellipse_views/: local metric ellipses under READ flow\n")
    else:
        text.append("  intrinsic/: unavailable because data/mean_b21_surfaces.npz was not found\n")
    text.append("\nCuriosity output (you_asked_for_it):\n")
    text.append("  raw vector-length quivers for both Euclidean and intrinsic definitions (when available)\n")
    text.append("  each raw folder also gets blender_ply/ with size-varying triangle glyphs\n")
    (root / "README_FIRST.txt").write_text("".join(text), encoding="utf-8")


# =============================================================================
# Multi-block result discovery / plotting
# =============================================================================

def _discover_multiblock_profiles(root: Path) -> list[dict]:
    """
    Discover v5 multi-block profiles and retain backwards compatibility with
    the older single-B21 atlas layout.
    """
    profiles=[]
    seen=set()

    candidates=[]
    if (root/'shared'/'surface_summary.csv').is_file():
        candidates.append(root)
    for p in root.rglob('shared/surface_summary.csv'):
        candidates.append(p.parent.parent)

    for prof in candidates:
        key=str(prof.resolve())
        if key in seen:
            continue
        seen.add(key)
        blocks={}
        for bdir in sorted(prof.glob('B*')):
            m=re.fullmatch(r'B(\d+)',bdir.name)
            if not m or not (bdir/'data').is_dir():
                continue
            block=int(m.group(1))
            npzs=sorted((bdir/'data').glob(f'mean_B{block}_surfaces.npz'))
            if not npzs:
                npzs=sorted((bdir/'data').glob('mean_B*_surfaces.npz'))
            if not npzs:
                continue
            blocks[block]={
                'dir':bdir,
                'npz':npzs[0],
                'cross_lang':bdir/'data'/'cross_language_geometry.csv',
            }
        if blocks:
            profiles.append({
                'name':prof.name,
                'root':prof,
                'surface':prof/'shared'/'surface_summary.csv',
                'blocks':blocks,
                'kind':'multiblock',
            })

    # Older single-block atlas: data/surface_summary.csv + data/mean_b21_surfaces.npz.
    if not profiles and (root/'data'/'surface_summary.csv').is_file():
        npz=root/'data'/'mean_b21_surfaces.npz'
        if npz.is_file():
            profiles.append({
                'name':'single_block',
                'root':root,
                'surface':root/'data'/'surface_summary.csv',
                'blocks':{
                    21:{
                        'dir':root,
                        'npz':npz,
                        'cross_lang':root/'data'/'cross_language_geometry.csv',
                    }
                },
                'kind':'single',
            })
    return profiles


def _parse_block_filter(text: str) -> Optional[set[int]]:
    text=str(text).strip()
    if not text:
        return None
    out=set()
    for tok in text.split(','):
        tok=tok.strip().upper().replace('B','')
        if not tok:
            continue
        if '-' in tok:
            a,b=tok.split('-',1)
            a=int(a);b=int(b)
            step=1 if b>=a else -1
            out.update(range(a,b+step,step))
        else:
            out.add(int(tok))
    return out


def _profile_output_root(base_out: Path, profile: dict, n_profiles: int) -> Path:
    # Preserve a simple output tree for the common one-profile case while keeping
    # _ridge_fast and _ridge_full separate if an input directory contains both.
    if n_profiles==1:
        return base_out
    return base_out/profile['name']


def _write_multiblock_readme(
    root: Path,
    blocks: Sequence[int],
    languages: Sequence[str],
    planes: Sequence[Tuple[int,int]],
    queries: Sequence[str],
    pump_modes: Sequence[str],
) -> None:
    txt=[]
    txt.append('RN READ-flow multi-block figure pack\n')
    txt.append('===================================\n\n')
    txt.append(f"Blocks discovered: {', '.join('B'+str(b) for b in blocks)}\n")
    txt.append(f"Languages: {', '.join(languages)}\n")
    txt.append(f"Planes: {', '.join(f'PC{a}xPC{b}' for a,b in planes)}\n")
    txt.append(f"Query modes: {', '.join(queries)}\n")
    txt.append(f"Register-pump modes: {', '.join(pump_modes)}\n\n")
    txt.append('Important geometry convention:\n')
    txt.append('  The scalar READ surface and Euclidean gradient are block-independent: the same\n')
    txt.append('  final READ response is laid over the common intervention coordinates. They are\n')
    txt.append('  therefore written once PER pump condition under _shared_read/{pump_mode}/.\n')
    txt.append('  B{idx}/{pump_mode}/ contains the block-specific intrinsic field G_B^{-1} grad READ, where\n')
    txt.append('  G_B is the pullback metric of that block\'s high-dimensional visual state sheet.\n\n')
    txt.append('Blender exports:\n')
    txt.append('  *.ply                 scalar READ surface\n')
    txt.append('  *_field.ply          flat control-coordinate glyphs\n')
    txt.append('  *_field_tangent.ply  glyphs constructed in the local tangent plane of the rendered READ surface\n\n')
    txt.append('Normal output uses normalized vectors. Raw-length variants live under each pump condition\'s you_asked_for_it/.\n')
    txt.append('If intact and no_pump are both present, B{idx}/comparisons/ contains direct paired diagnostic figures.\n')
    (root/'README_FIRST.txt').write_text(''.join(txt),encoding='utf-8')


def _plot_shared_euclidean(
    out_root: Path,
    surface_df: pd.DataFrame,
    cross_lang_df: pd.DataFrame,
    model_name: str,
    pump_mode: str,
    condition: str,
    basis_kind: str,
    scalar_col: str,
    languages: Sequence[str],
    queries: Sequence[str],
    planes: Sequence[Tuple[int,int]],
    scripts: Dict[str,str],
) -> None:
    shared_root=out_root/'_shared_read'/flow_maps_stable_slug(pump_mode)
    for query_mode in queries:
        for plane in planes:
            lang_bundles={}
            for lang in languages:
                try:
                    bundle=build_surface_grid(
                        surface_df,lang,query_mode,plane,model_name,pump_mode,
                        condition=condition,basis_kind=basis_kind,scalar_col=scalar_col,
                    )
                except KeyError:
                    continue
                compute_euclidean_gradient(bundle)
                lang_bundles[lang]=bundle

            for lang,bundle in lang_bundles.items():
                slug=flow_maps_stable_slug(f'{lang}_PC{plane[0]}xPC{plane[1]}_{query_mode}__{pump_mode}')
                title=f'{pump_mode} | {lang} | PC{plane[0]}×PC{plane[1]} | {query_mode} | normalized READ flow (shared across blocks)'
                plot_surface_stream(
                    shared_root/'main'/'euclidean'/'views_stream'/f'{slug}.png',
                    bundle,bundle.ga,bundle.gb,plane,title,True,False,
                )
                plot_surface_quiver(
                    shared_root/'main'/'euclidean'/'views_quiver'/f'{slug}.png',
                    bundle,bundle.ga,bundle.gb,plane,title,True,1,False,
                )
                plot_surface_quiver(
                    shared_root/'main'/'euclidean'/'camera_angles'/f'{slug}__contour_quiver.png',
                    bundle,bundle.ga,bundle.gb,plane,
                    f'{lang} | PC{plane[0]}×PC{plane[1]} | {query_mode} | shared Euclidean alt view',
                    True,2,False,
                )
                plot_surface_quiver(
                    shared_root/'you_asked_for_it'/'euclidean_raw'/f'{slug}.png',
                    bundle,bundle.ga,bundle.gb,plane,
                    f'{lang} | PC{plane[0]}×PC{plane[1]} | {query_mode} | raw-length READ gradient',
                    False,1,False,
                )
                export_surface_and_field_ply(
                    shared_root/'main'/'euclidean'/'blender_ply'/slug,
                    bundle,bundle.ga,bundle.gb,True,
                )
                export_surface_and_field_ply(
                    shared_root/'you_asked_for_it'/'euclidean_raw'/'blender_ply'/slug,
                    bundle,bundle.ga,bundle.gb,False,
                )

            if lang_bundles:
                plot_alignment_heatmaps(
                    shared_root/'main'/'euclidean'/'group_comparisons'/f'PC{plane[0]}xPC{plane[1]}__{query_mode}.png',
                    lang_bundles,languages,scripts,plane,query_mode,False,
                )
                plot_pairwise_matrices(
                    shared_root/'main'/'euclidean'/'matrices'/f'PC{plane[0]}xPC{plane[1]}__{query_mode}.png',
                    lang_bundles,languages,plane,query_mode,False,cross_lang_df,
                )


def _plot_one_intrinsic_block(
    block: int,
    out_root: Path,
    surface_df: pd.DataFrame,
    cross_lang_df: pd.DataFrame,
    state_surfaces: dict,
    model_name: str,
    pump_mode: str,
    condition: str,
    basis_kind: str,
    scalar_col: str,
    languages: Sequence[str],
    queries: Sequence[str],
    planes: Sequence[Tuple[int,int]],
    scripts: Dict[str,str],
) -> dict:
    block_root=out_root/f'B{block}'/flow_maps_stable_slug(pump_mode)
    produced=0
    for query_mode in queries:
        for plane in planes:
            lang_bundles={}
            for lang in languages:
                try:
                    bundle=build_surface_grid(
                        surface_df,lang,query_mode,plane,model_name,pump_mode,
                        condition=condition,basis_kind=basis_kind,scalar_col=scalar_col,
                    )
                except KeyError:
                    continue
                compute_euclidean_gradient(bundle)
                key=(model_name,pump_mode,condition,lang,basis_kind,plane[0],plane[1])
                if key not in state_surfaces:
                    continue
                mean=state_surfaces[key]
                H=len(bundle.a_vals);W=len(bundle.b_vals)
                if mean.ndim==2:
                    if mean.shape[0] != H*W:
                        raise ValueError(
                            f'B{block} {lang} PC{plane[0]}xPC{plane[1]}: state rows={mean.shape[0]} but grid={H}x{W}'
                        )
                    mean=mean.reshape(H,W,-1)
                compute_intrinsic_geometry_and_gradient(bundle,mean)
                lang_bundles[lang]=bundle

            for lang,bundle in lang_bundles.items():
                slug=flow_maps_stable_slug(f'{lang}_PC{plane[0]}xPC{plane[1]}_{query_mode}__{pump_mode}')
                prefix=f'B{block} | {pump_mode} | {lang} | PC{plane[0]}×PC{plane[1]} | {query_mode}'
                plot_surface_stream(
                    block_root/'main'/'intrinsic'/'views_stream'/f'{slug}.png',
                    bundle,bundle.iga,bundle.igb,plane,
                    prefix+' | normalized intrinsic READ flow',True,False,
                )
                plot_metric_ellipse_sheet(
                    block_root/'main'/'intrinsic'/'metric_ellipse_views'/f'{slug}.png',
                    bundle,plane,prefix,use_intrinsic=True,
                )
                plot_surface_quiver(
                    block_root/'main'/'intrinsic'/'camera_angles'/f'{slug}__quiver.png',
                    bundle,bundle.iga,bundle.igb,plane,
                    prefix+' | intrinsic alt view',True,2,True,
                )
                plot_surface_quiver(
                    block_root/'you_asked_for_it'/'intrinsic_raw'/f'{slug}.png',
                    bundle,bundle.iga,bundle.igb,plane,
                    prefix+' | raw-length intrinsic READ gradient',False,1,True,
                )
                export_surface_and_field_ply(
                    block_root/'main'/'intrinsic'/'blender_ply'/slug,
                    bundle,bundle.iga,bundle.igb,True,
                )
                export_surface_and_field_ply(
                    block_root/'you_asked_for_it'/'intrinsic_raw'/'blender_ply'/slug,
                    bundle,bundle.iga,bundle.igb,False,
                )
                produced+=1

            if lang_bundles:
                plot_alignment_heatmaps(
                    block_root/'main'/'intrinsic'/'group_comparisons'/f'PC{plane[0]}xPC{plane[1]}__{query_mode}.png',
                    lang_bundles,languages,scripts,plane,query_mode,True,
                )
                plot_pairwise_matrices(
                    block_root/'main'/'intrinsic'/'matrices'/f'PC{plane[0]}xPC{plane[1]}__{query_mode}.png',
                    lang_bundles,languages,plane,query_mode,True,cross_lang_df,
                )
    return {'block':block,'n_language_plane_query_surfaces':produced}


def _draw_flow_on_axis(
    ax,
    bundle: SurfaceBundle,
    comp0: np.ndarray,
    comp1: np.ndarray,
    title: str,
    normalized: bool = True,
):
    U = np.asarray(comp1, np.float64)
    V = np.asarray(comp0, np.float64)
    if normalized:
        Vp, Up, mag = normalized_components(V, U)
    else:
        Vp, Up = V, U
        mag = np.sqrt(V*V + U*U)
    im = ax.imshow(
        bundle.scalar,
        origin="lower",
        extent=[bundle.b_vals.min(), bundle.b_vals.max(), bundle.a_vals.min(), bundle.a_vals.max()],
        aspect="auto",
        cmap="coolwarm",
        alpha=0.92,
    )
    ax.contour(bundle.X, bundle.Y, bundle.scalar, levels=10, colors="k", linewidths=0.28, alpha=0.32)
    stride = 1 if len(bundle.a_vals) <= 17 else 2
    ax.quiver(
        bundle.X[::stride, ::stride],
        bundle.Y[::stride, ::stride],
        Up[::stride, ::stride],
        Vp[::stride, ::stride],
        mag[::stride, ::stride],
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=18 if normalized else None,
        width=0.006,
        pivot="mid",
    )
    ax.set_title(title)
    ax.set_xlabel("α_b")
    ax.set_ylabel("α_a")
    return im


def _plot_intact_vs_no_pump_block(
    block: int,
    out_root: Path,
    surface_df: pd.DataFrame,
    state_surfaces: dict,
    model_name: str,
    condition: str,
    basis_kind: str,
    scalar_col: str,
    languages: Sequence[str],
    queries: Sequence[str],
    planes: Sequence[Tuple[int,int]],
) -> int:
    """
    Direct paired diagnostic: same coordinates/basis, intact vs no_pump.

    Top: intrinsic READ fields over each condition's scalar surface.
    Bottom-left: pointwise cosine between intrinsic fields.
    Bottom-right: scalar READ difference (no_pump - intact).
    """
    produced = 0
    cmp_root = out_root / f"B{block}" / "comparisons" / "intact_vs_no_pump"
    for query_mode in queries:
        for plane in planes:
            for lang in languages:
                bundles = {}
                ok = True
                for pump_mode in ("intact", "no_pump"):
                    try:
                        b = build_surface_grid(
                            surface_df, lang, query_mode, plane, model_name, pump_mode,
                            condition=condition, basis_kind=basis_kind, scalar_col=scalar_col,
                        )
                    except KeyError:
                        ok = False
                        break
                    compute_euclidean_gradient(b)
                    key = (model_name, pump_mode, condition, lang, basis_kind, plane[0], plane[1])
                    if key not in state_surfaces:
                        ok = False
                        break
                    mean = state_surfaces[key]
                    H, W = len(b.a_vals), len(b.b_vals)
                    if mean.ndim == 2:
                        if mean.shape[0] != H * W:
                            ok = False
                            break
                        mean = mean.reshape(H, W, -1)
                    compute_intrinsic_geometry_and_gradient(b, mean)
                    bundles[pump_mode] = b
                if not ok:
                    continue

                bi = bundles["intact"]
                bn = bundles["no_pump"]
                cosmap = pairwise_gradient_cos(bi.iga, bi.igb, bn.iga, bn.igb)
                scalar_delta = bn.scalar - bi.scalar

                fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.6), dpi=150)
                im0 = _draw_flow_on_axis(
                    axes[0,0], bi, bi.iga, bi.igb,
                    f"intact | intrinsic READ flow", normalized=True,
                )
                im1 = _draw_flow_on_axis(
                    axes[0,1], bn, bn.iga, bn.igb,
                    f"no_pump | intrinsic READ flow", normalized=True,
                )
                fig.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.02, label="relative READ")
                fig.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.02, label="relative READ")

                im2 = axes[1,0].imshow(
                    cosmap, origin="lower",
                    extent=[bi.b_vals.min(), bi.b_vals.max(), bi.a_vals.min(), bi.a_vals.max()],
                    aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0,
                )
                axes[1,0].set_title(
                    f"intrinsic field cosine | mean={float(np.nanmean(cosmap)):.3f}"
                )
                axes[1,0].set_xlabel("α_b"); axes[1,0].set_ylabel("α_a")
                fig.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.02)

                lim = max(float(np.nanmax(np.abs(scalar_delta))), 1e-8)
                im3 = axes[1,1].imshow(
                    scalar_delta, origin="lower",
                    extent=[bi.b_vals.min(), bi.b_vals.max(), bi.a_vals.min(), bi.a_vals.max()],
                    aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim,
                )
                axes[1,1].set_title("relative READ difference: no_pump - intact")
                axes[1,1].set_xlabel("α_b"); axes[1,1].set_ylabel("α_a")
                fig.colorbar(im3, ax=axes[1,1], fraction=0.046, pad=0.02)

                fig.suptitle(
                    f"B{block} | {lang} | PC{plane[0]}×PC{plane[1]} | {query_mode} | register-pump ablation",
                    y=0.985,
                )
                fig.tight_layout(rect=[0,0,1,0.965])
                slug = flow_maps_stable_slug(f"{lang}_PC{plane[0]}xPC{plane[1]}_{query_mode}__intact_vs_no_pump")
                ensure_dir(cmp_root)
                fig.savefig(cmp_root / f"{slug}.png", bbox_inches="tight")
                plt.close(fig)
                produced += 1
    return produced

def flow_maps_main():
    ap=argparse.ArgumentParser(
        description='Plot Euclidean and block-specific intrinsic READ fields from native RN control-surface outputs.'
    )
    ap.add_argument('--input',type=str,default='rn_register_pump_multiblock_surface/compact_summary_rn_control_surfaces_full.zip',
                    help='Multi-block result directory/zip from v5, or older single-block native atlas.')
    ap.add_argument('--output-dir',type=str,default='rn_read_flow_figures_multiblock_pump_nopump')
    ap.add_argument('--blocks',type=str,default='',
                    help='Optional block filter, e.g. 13,19-21. Default: every B*/data/mean_B*_surfaces.npz discovered.')
    ap.add_argument('--model-name',type=str,default=None)
    ap.add_argument('--pump-mode',type=str,default=None,
                    help='Backwards-compatible single pump condition selector.')
    ap.add_argument('--pump-modes',type=str,default='',
                    help='Comma-separated pump conditions. Default: all discovered, e.g. intact,no_pump.')
    ap.add_argument('--condition',type=str,default='synth')
    ap.add_argument('--basis-kind',type=str,default='rn')
    ap.add_argument('--scalar-col',type=str,default='relative_calibrated_mean')
    ap.add_argument('--languages',type=str,default='')
    ap.add_argument('--queries',type=str,default='')
    ap.add_argument('--planes',type=str,default='')
    ap.add_argument('--skip-shared-euclidean',action='store_true',
                    help='Skip the block-independent scalar/Euclidean READ figures if already rendered.')
    args=ap.parse_args()

    inp=Path(args.input)
    root,tmpdir=maybe_extract_input(inp)
    try:
        profiles=_discover_multiblock_profiles(root)
        if not profiles:
            raise FileNotFoundError(
                'Could not discover a multi-block profile (shared/surface_summary.csv + B*/data/mean_B*_surfaces.npz) '
                'or older data/mean_b21_surfaces.npz layout.'
            )
        block_filter=_parse_block_filter(args.blocks)
        base_out=ensure_dir(Path(args.output_dir))
        global_manifest={'input':str(inp),'resolved_root':str(root),'profiles':[]}

        for profile in profiles:
            surface_df=pd.read_csv(profile['surface'])
            model_names=sorted(surface_df['model_name'].astype(str).unique().tolist())
            discovered_pump_modes=sorted(surface_df['pump_mode'].astype(str).unique().tolist())
            model_name=args.model_name or (model_names[0] if model_names else None)
            if args.pump_mode and args.pump_modes.strip():
                raise ValueError("Use either --pump-mode or --pump-modes, not both.")
            if args.pump_mode:
                selected_pump_modes=[args.pump_mode]
            elif args.pump_modes.strip():
                selected_pump_modes=[x.strip() for x in args.pump_modes.split(',') if x.strip()]
            else:
                # Prefer semantically useful order over alphabetical no_pump,intact.
                selected_pump_modes=[x for x in ("intact","no_pump") if x in discovered_pump_modes]
                selected_pump_modes += [x for x in discovered_pump_modes if x not in selected_pump_modes]
            missing=[x for x in selected_pump_modes if x not in discovered_pump_modes]
            if missing:
                raise ValueError(f"Requested pump modes not present: {missing}; found={discovered_pump_modes}")
            languages=[x.strip() for x in args.languages.split(',') if x.strip()] or parse_languages(surface_df)
            queries=[x.strip() for x in args.queries.split(',') if x.strip()] or parse_queries(surface_df)
            if args.planes.strip():
                planes=[]
                for tok in args.planes.split(','):
                    tok=tok.strip().lower().replace('pc','')
                    a,b=tok.split('x');planes.append((int(a),int(b)))
            else:
                planes=flow_maps_parse_planes(surface_df)
            scripts=infer_scripts(languages)
            blocks=sorted(profile['blocks'])
            if block_filter is not None:
                blocks=[b for b in blocks if b in block_filter]
            if not blocks:
                print(f"[skip] {profile['name']}: no blocks survive --blocks filter")
                continue

            prof_out=_profile_output_root(base_out,profile,len(profiles))
            ensure_dir(prof_out)
            _write_multiblock_readme(prof_out,blocks,languages,planes,queries,selected_pump_modes)
            print(f"[profile] {profile['name']} | blocks={blocks} | pump_modes={selected_pump_modes}")

            # Any block's cross-language table suffices for the block-independent
            # scalar READ values. Filter it by pump condition before plotting matrices.
            first_cross=profile['blocks'][blocks[0]]['cross_lang']
            cross_shared_all=_read_optional_csv(first_cross)
            if not args.skip_shared_euclidean:
                for pump_mode in selected_pump_modes:
                    cross_shared = cross_shared_all
                    if not cross_shared.empty and "pump_mode" in cross_shared.columns:
                        cross_shared = cross_shared[cross_shared["pump_mode"].astype(str)==pump_mode].copy()
                    print(f'[shared/{pump_mode}] plotting scalar READ + Euclidean field once across state blocks')
                    _plot_shared_euclidean(
                        prof_out,surface_df,cross_shared,model_name,pump_mode,args.condition,
                        args.basis_kind,args.scalar_col,languages,queries,planes,scripts,
                    )

            block_manifest=[]
            both_register_conditions = {"intact","no_pump"}.issubset(set(selected_pump_modes))
            for block in blocks:
                info=profile['blocks'][block]
                print(f'[B{block}] loading {info["npz"].name} and plotting intrinsic blinkenlights')
                state_surfaces=parse_mean_state_npz(info['npz'])
                cross_lang_all=_read_optional_csv(info['cross_lang']) if info['cross_lang'].is_file() else cross_shared_all
                mode_outputs=[]
                for pump_mode in selected_pump_modes:
                    cross_lang = cross_lang_all
                    if not cross_lang.empty and "pump_mode" in cross_lang.columns:
                        cross_lang = cross_lang[cross_lang["pump_mode"].astype(str)==pump_mode].copy()
                    print(f'  [{pump_mode}] intrinsic figures + PLY')
                    got=_plot_one_intrinsic_block(
                        block,prof_out,surface_df,cross_lang,state_surfaces,model_name,pump_mode,
                        args.condition,args.basis_kind,args.scalar_col,languages,queries,planes,scripts,
                    )
                    got['pump_mode']=pump_mode
                    mode_outputs.append(got)
                comparison_count=0
                if both_register_conditions:
                    print('  [comparison] intact vs no_pump paired intrinsic fields')
                    comparison_count=_plot_intact_vs_no_pump_block(
                        block,prof_out,surface_df,state_surfaces,model_name,args.condition,
                        args.basis_kind,args.scalar_col,languages,queries,planes,
                    )
                block_manifest.append({
                    'block':block,
                    'state_npz':str(info['npz']),
                    'pump_outputs':mode_outputs,
                    'n_intact_vs_no_pump_comparisons':comparison_count,
                })

            manifest={
                'profile_name':profile['name'],
                'profile_kind':profile['kind'],
                'profile_root':str(profile['root']),
                'output_root':str(prof_out),
                'model_name':model_name,'pump_modes':selected_pump_modes,
                'condition':args.condition,'basis_kind':args.basis_kind,'scalar_col':args.scalar_col,
                'languages':languages,'queries':queries,'planes':[list(x) for x in planes],
                'blocks':blocks,
                'shared_euclidean_rendered':not args.skip_shared_euclidean,
                'block_outputs':block_manifest,
                'note':'Euclidean READ gradient is block-independent but pump-condition dependent; B*/{pump_mode}/ intrinsic outputs use each block-specific visual-state pullback metric. Paired intact/no_pump figures are under B*/comparisons/.',
            }
            (prof_out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            global_manifest['profiles'].append(manifest)

        (base_out/'multiblock_manifest.json').write_text(json.dumps(global_manifest,indent=2),encoding='utf-8')
        print(f'[done] multi-block READ-flow figures saved to {base_out}')
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def main(argv=None):
    """Dispatch a workflow; each subcommand retains its original CLI options."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {'analyze': analyze_main, 'flow_maps': flow_maps_main}
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

