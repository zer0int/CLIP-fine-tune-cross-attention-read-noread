#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual-axis lineage and coordinate-takeover atlas for CLIP ViT-L/14 variants.

Purpose
-------
Distinguish two very different species of special residual coordinate:

  1. persistent inherited axes: an unusual Conv1/initial coordinate remains a
     recognizable signal and retains downstream causal leverage;
  2. late constructed buses: the Conv1 coordinate is ordinary or loses its
     retinal ancestry, then attention/MLP writes repurpose the same residual
     wire into a high-gain workspace/control variable.

The default special set includes the user-curated Conv1/residual culprits plus
650/565, whose late exchange/QK behavior motivates this probe. Deterministic
random residual coordinates are added as controls.

Baseline lineage measurements
-----------------------------
For every selected residual coordinate, image, and ViT block, capture:
  * raw Conv1 patch response;
  * Conv1 + positional embedding (pre-ln_pre);
  * actual transformer input after ln_pre;
  * block pre-attention residual;
  * post-attention residual and exact attention write;
  * post-MLP residual and exact MLP write;
  * CLS coordinate, spatial coordinate, frozen-register enrichment;
  * correlation to own raw Conv1 / initial residual;
  * multivariate R^2 against ALL selected initial coordinates;
  * standardized own-source and strongest-other-source coefficients;
  * full current-axis x initial-axis correlation matrix;
  * optional correlations with a supplied (mu1, mu2) role basis.

B22 650<->565 QK audit
----------------------
At selected QK blocks (default B22), decompose the CLS-query / patch-key logit
into coordinate-only terms for the selected pair:
  650Q*650K, 650Q*565K, 565Q*650K, 565Q*565K,
separately for frozen-register and ordinary-patch destinations, per head/image.
This uses the actual ln_1 state and q_proj/k_proj columns; biases are excluded
from the coordinate attribution but exact full QK logits are saved alongside.

650<->565 causal cross-wiring
----------------------------
The swap is an ACTIVATION coordinate swap; weights are untouched.
  full:    swap once before B0 and let the swapped state propagate;
  onset-k: swap once immediately before block k and propagate thereafter;
  window-k: swap immediately before block k, execute that block, then swap the
            two coordinates back in the block output. Only that block sees the
            cross-wired coordinate assignment.

Outputs include final embedding displacement, clean/adv split, same-scene vs
unrelated-scene manifold margin, and safetensors for all swap embeddings.

Supported model variants
------------------------
  pretrained      openai/clip-vit-large-patch14 via attnclip_mechinterp_sae
  gmp             zer0int/CLIP-GmP-ViT-L-14 via attnclip_mechinterp_sae
  bare_xattn      x-attn visual weights transplanted into vanilla ViT-L/14
  full_xattn      full ModeMUX visual backbone via attnclip_mechinterp_xattn
  pretrained_rn   pretrained + transplanted x-attn READ_NULL token only
  gmp_rn          GmP + transplanted READ_NULL token only
  bare_xattn_rn   bare x-attn + transplanted READ_NULL token only

No pickle output is produced. Raw vector/state artifacts use safetensors.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import random
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from safetensors.torch import save_file as _save_st, load_file as _load_st
except Exception as exc:
    raise RuntimeError("safetensors is required: pip install safetensors") from exc

EPS = 1e-12
SEED = 20260917
DEFAULT_IMAGE_DIR = r"image_sets/special_natural"
DEFAULT_OUT_ROOT = r"out_paper_reproduction/conv1/residual_axis_lineage"
DEFAULT_PRETRAINED = "openai/clip-vit-large-patch14"
DEFAULT_GMP = "zer0int/CLIP-GmP-ViT-L-14"
DEFAULT_XATTN = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_MODELS = "pretrained,gmp,bare_xattn,full_xattn"
DEFAULT_SPECIAL_AXES = (
    499, 250, 908, 953, 779, 196, 350, 139, 468, 469,
    951, 211, 1021, 866, 151, 720, 656, 400, 565, 650,
)
DEFAULT_SWAP_PAIR = (650, 565)

USER_EXPLICIT = {499,250,908,953,779,196,350,139,468,469,951,211,1021,866,151,565,650}
PRIOR_FOCUS = {720,779,866,151,499,250,468,908,656,400}
BUS_PAIR = {650,565}

# -----------------------------------------------------------------------------
# generic helpers
# -----------------------------------------------------------------------------

def save_safetensors(tensors: Mapping[str, torch.Tensor], filename: str, metadata: Optional[Mapping[str, str]] = None):
    packed = {str(k): v.detach().contiguous().clone() for k, v in tensors.items()}
    md = None if metadata is None else {str(k): str(v) for k, v in metadata.items()}
    _save_st(packed, filename, metadata=md)


def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(x: str) -> List[int]:
    return [int(v.strip()) for v in str(x).split(",") if v.strip()]


def parse_str_list(x: str) -> List[str]:
    return [v.strip().lower() for v in str(x).split(",") if v.strip()]


def safe_name(x: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x)).strip("_") or "item"


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def write_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".gz":
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)



# -----------------------------------------------------------------------------
# output layout / legacy recovery
# -----------------------------------------------------------------------------

LINEAGE_LEGACY_FILES = (
    "lineage_states.safetensors", "qk_pair_terms.csv.gz", "lineage_config.json",
    "axis_lineage_metrics.csv", "axis_cross_lineage.csv.gz", "axis_classification.csv",
    "SPECIAL_AXIS_CLASSIFICATION.txt",
)
SWAP_LEGACY_FILES = (
    "swap_embeddings.safetensors", "swap_event_metadata.csv", "swap_per_image.csv",
    "swap_summary.csv", "swap_config.json",
)


def stage_output_roots(base_root: Path, args):
    """Collision-proof output roots. Lineage and swap never share a model directory."""
    lineage_root = base_root / "lineage"
    a, b = map(int, args.swap_pair)
    swap_root = base_root / f"swap_{a}_{b}_{args.swap_token_scope}"
    return lineage_root, swap_root


def _copy_file_if_missing(src: Path, dst: Path, log: List[str]):
    if src.is_file() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log.append(f"FILE {src} -> {dst}")


def _copy_tree_merge(src: Path, dst: Path, log: List[str]):
    if src.is_dir():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        log.append(f"DIR  {src} -> {dst}")


def migrate_legacy_stage(base_root: Path, stage_root: Path, stage: str, models: Sequence[str]) -> List[str]:
    """COPY recoverable v2.3 flat-layout artifacts into the new stage directory.

    This is intentionally non-destructive: nothing in the legacy folder is moved,
    deleted, or overwritten. Stage-specific expensive artifacts are copied only when
    the new destination is absent. Shared files that a different suite could have
    overwritten are regenerated instead of trusted.
    """
    log: List[str] = []
    for name in models:
        legacy = base_root / name
        if not legacy.is_dir():
            continue
        dst = stage_root / name
        if stage == "lineage":
            for fn in LINEAGE_LEGACY_FILES:
                _copy_file_if_missing(legacy / fn, dst / fn, log)
            _copy_tree_merge(legacy / "plots" / "axis_focus", dst / "plots" / "axis_focus", log)
            _copy_tree_merge(legacy / "plots" / "qk", dst / "plots" / "qk", log)
            _copy_file_if_missing(legacy / "plots" / "axis_classification_scatter.png", dst / "plots" / "axis_classification_scatter.png", log)
            plots = legacy / "plots"
            if plots.is_dir():
                for q in plots.glob("pair_*_lineage_matrix.png"):
                    _copy_file_if_missing(q, dst / "plots" / q.name, log)
        elif stage == "swap":
            for fn in SWAP_LEGACY_FILES:
                _copy_file_if_missing(legacy / fn, dst / fn, log)
            _copy_tree_merge(legacy / "swap_event_cache", dst / "swap_event_cache", log)
            _copy_tree_merge(legacy / "plots" / "swap", dst / "plots" / "swap", log)
        else:
            raise ValueError(stage)
    if log:
        stage_root.mkdir(parents=True, exist_ok=True)
        (stage_root / "LEGACY_MIGRATION_LOG.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
        print(f"[{stage}] recovered {len(log)} legacy artifact(s) into {stage_root}")
    return log


def corr(a, b) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a = a[m] - np.mean(a[m])
    b = b[m] - np.mean(b[m])
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > EPS else 0.0


def rms(x) -> float:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x*x))) if x.size else float("nan")


def robust_z(x):
    x = np.asarray(x, np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x-med))
    s = 1.4826 * mad
    if not np.isfinite(s) or s < EPS:
        s = np.nanstd(x)
    if not np.isfinite(s) or s < EPS:
        s = 1.0
    return (x-med)/s


def normalize_t(x: torch.Tensor) -> torch.Tensor:
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(EPS)


def cosine_distance_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1, eps=EPS)


def amp_context(device: str, enabled: bool = True):
    """Match the audited vanilla probes: CLIP backbone QKV/proj may be FP16
    while the residual stream and some backbone modules are FP32. CUDA autocast
    is therefore part of the intended manual-forward precision policy.
    """
    dev = torch.device(device)
    if enabled and dev.type == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def source_from_stem(stem: str) -> str:
    return "adv" if stem.endswith("_adv") else "clean"


def pair_from_stem(stem: str) -> str:
    return stem[:-4] if stem.endswith("_adv") else stem


def scan_images(image_dir: Path, recursive: bool = False) -> pd.DataFrame:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    it = image_dir.rglob("*") if recursive else image_dir.iterdir()
    fs = sorted([p.resolve() for p in it if p.is_file() and p.suffix.lower() in exts], key=lambda p: p.name.lower())
    if not fs:
        raise RuntimeError(f"No images found in {image_dir}")
    rows = []
    seen = {}
    for p in fs:
        base = p.stem
        n = seen.get(base, 0)
        seen[base] = n + 1
        sid = base if n == 0 else f"{base}__dup{n}"
        rows.append({"stim_id": sid, "filename": p.name, "path": str(p), "source": source_from_stem(base), "pair": pair_from_stem(base)})
    return pd.DataFrame(rows)


def load_batch(preprocess, rows: pd.DataFrame, device: str):
    ims = []
    for r in rows.itertuples(index=False):
        with Image.open(r.path) as im:
            ims.append(preprocess(ImageOps.exif_transpose(im).convert("RGB")))
    return torch.stack(ims, 0).to(device, non_blocking=True)


def find_repo_root(start: Optional[Path] = None) -> Path:
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    if start is not None:
        starts.insert(0, start.resolve())
    seen = set()
    for r0 in starts:
        for p in [r0, *r0.parents]:
            if p in seen:
                continue
            seen.add(p)
            if (p / "attnclip_mechinterp_sae").is_dir() and (p / "attnclip_mechinterp_xattn").is_dir() and (p / "utils_clip_loader").is_dir():
                return p
    raise FileNotFoundError("Could not locate repo root with attnclip_mechinterp_sae, attnclip_mechinterp_xattn, utils_clip_loader; pass --repo_root")


# -----------------------------------------------------------------------------
# model loading
# -----------------------------------------------------------------------------

@dataclass
class Variant:
    name: str
    model: Any
    preprocess: Any
    rn_token: Optional[torch.Tensor]
    rn_insert_block: Optional[int]
    is_full_xattn: bool
    source_info: Dict[str, Any]

    @property
    def visual(self):
        return self.model.visual

    def maybe_insert_rn(self, block_idx: int, x_tbc: torch.Tensor) -> torch.Tensor:
        if self.is_full_xattn:
            return self.visual._maybe_insert_read_null(block_idx, x_tbc)
        if self.rn_token is not None and int(block_idx) == int(self.rn_insert_block):
            null = self.rn_token.to(dtype=x_tbc.dtype, device=x_tbc.device).view(1,1,-1).expand(1,x_tbc.shape[1],-1)
            return torch.cat([x_tbc, null], dim=0)
        return x_tbc


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
    sd, info = resolve_fn(args.xattn_model, cache_dir=(args.hf_cache_dir or None), revision=(args.xattn_revision or None), allow_unsafe_hf_pickle=False)
    conv = getattr(getattr(clip_sae, "model", None), "convert_state_dict_inproj_to_qkv", None)
    if callable(conv):
        sd = conv(sd)
    return sd, info


def _extract_rn(clip_sae, resolve_fn, args):
    sd, info = _resolve_xattn_state(clip_sae, resolve_fn, args)
    if "visual.read_null_token" not in sd:
        raise KeyError("x-attn checkpoint has no visual.read_null_token")
    token = sd["visual.read_null_token"].detach().float().cpu().reshape(-1).contiguous()
    block = int(sd.get("visual.read_null_insert_block_config", torch.tensor(args.rn_insert_block)).item())
    return token, block, info


def _load_gmp(clip_sae, load_any, args, device):
    try:
        model, preprocess, li = load_any(clip_sae, args.gmp_checkpoint, device=device, jit=False, strict=True, reuse_full_model_pickle=False)
        return _freeze(model), preprocess, {"source":args.gmp_checkpoint,"mode":"state_dict_rebuild","loader":str(li)}
    except Exception as first_error:
        src, _pp, li = load_any(clip_sae, args.gmp_checkpoint, device="cpu", jit=False, strict=True, reuse_full_model_pickle=True)
        model, preprocess, _ = load_any(clip_sae, args.pretrained_model, device=device, jit=False, strict=True, allow_unsafe_hf_pickle=False)
        srcsd = src.state_dict()
        conv = getattr(getattr(clip_sae,"model",None), "convert_state_dict_inproj_to_qkv", None)
        if callable(conv):
            srcsd = conv(srcsd)
        tgt = model.state_dict()
        filt = {k:v.to(dtype=tgt[k].dtype) for k,v in srcsd.items() if k.startswith("visual.") and k in tgt and tuple(v.shape)==tuple(tgt[k].shape)}
        miss = sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
        if miss:
            raise RuntimeError(f"GmP fallback missing {len(miss)} visual keys: {miss[:12]}") from first_error
        model.load_state_dict(filt, strict=False)
        del src
        gc.collect()
        return _freeze(model), preprocess, {"source":args.gmp_checkpoint,"mode":"trusted_pickle_visual_transplant","loader":str(li),"first_error":repr(first_error)}


def _load_bare_xattn(clip_sae, load_any, resolve_fn, args, device):
    model, preprocess, _ = load_any(clip_sae, args.pretrained_model, device=device, jit=False, strict=True, allow_unsafe_hf_pickle=False)
    sd, info = _resolve_xattn_state(clip_sae, resolve_fn, args)
    tgt = model.state_dict()
    filt = {}
    ignored = []
    for k,v in sd.items():
        if not k.startswith("visual.") or k in {"visual.read_null_token","visual.read_null_insert_block_config"} or k not in tgt or tuple(v.shape)!=tuple(tgt[k].shape):
            ignored.append(k)
            continue
        filt[k] = v.to(dtype=tgt[k].dtype)
    missing = sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
    if missing:
        raise RuntimeError(f"bare_xattn missing {len(missing)} vanilla visual keys: {missing[:12]}")
    inc = model.load_state_dict(filt, strict=False)
    return _freeze(model), preprocess, {"source":args.xattn_model,"loaded_visual_keys":len(filt),"ignored_key_count":len(ignored),"load_missing":list(inc.missing_keys),"load_unexpected":list(inc.unexpected_keys),"loader":str(info)}


def load_variant(name: str, args) -> Variant:
    repo, clip_sae, clip_x, load_any, resolve_fn = _import_runtime(args)
    base = name.lower()
    transplant_rn = base.endswith("_rn")
    if transplant_rn:
        base = base[:-3]
    if base == "pretrained":
        model, preprocess, li = load_any(clip_sae, args.pretrained_model, device=args.device, jit=False, strict=True, allow_unsafe_hf_pickle=False)
        model = _freeze(model)
        info = {"source":args.pretrained_model,"mode":"pretrained","loader":str(li)}
        is_full = False
    elif base == "gmp":
        model, preprocess, info = _load_gmp(clip_sae, load_any, args, args.device)
        is_full = False
    elif base == "bare_xattn":
        model, preprocess, info = _load_bare_xattn(clip_sae, load_any, resolve_fn, args, args.device)
        is_full = False
    elif base == "full_xattn":
        if transplant_rn:
            raise ValueError("Use full_xattn, not full_xattn_rn; RN is already architectural")
        model, preprocess, li = load_any(clip_x, args.xattn_model, device=args.device, jit=False, cache_dir=(args.hf_cache_dir or None), revision=(args.xattn_revision or None), strict=True, allow_unsafe_hf_pickle=False)
        model = _freeze(model)
        info = {"source":args.xattn_model,"mode":"full_xattn","loader":str(li)}
        is_full = True
    else:
        raise ValueError(f"Unknown model variant {name}")

    rn_token = None
    rn_block = None
    if transplant_rn:
        rn_token, rn_block, _ = _extract_rn(clip_sae, resolve_fn, args)
        if rn_token.numel() != model.visual.conv1.weight.shape[0]:
            raise RuntimeError("RN token width mismatch")
    elif is_full:
        rn_block = int(getattr(model.visual, "read_null_insert_block", args.rn_insert_block))

    return Variant(name=name, model=model, preprocess=preprocess, rn_token=rn_token, rn_insert_block=rn_block, is_full_xattn=is_full, source_info=info)


# -----------------------------------------------------------------------------
# axes / role basis
# -----------------------------------------------------------------------------

def select_axes(args, width: int):
    special = []
    for x in args.axes:
        if x < 0 or x >= width:
            raise ValueError(f"axis {x} outside width {width}")
        if x not in special:
            special.append(x)
    rng = np.random.default_rng(args.seed)
    pool = np.array([i for i in range(width) if i not in set(special)], dtype=int)
    controls = rng.choice(pool, size=min(args.control_count, len(pool)), replace=False).tolist() if args.control_count > 0 else []
    axes = special + controls
    rows=[]
    for i,c in enumerate(axes):
        tags=[]
        if c in USER_EXPLICIT: tags.append("user_explicit")
        if c in PRIOR_FOCUS: tags.append("prior_culprit")
        if c in BUS_PAIR: tags.append("bus_pair")
        if c in controls: tags.append("random_control")
        rows.append({"axis_index":i,"channel":c,"is_special":c in special,"is_control":c in controls,"tags":"|".join(tags)})
    return axes, special, controls, pd.DataFrame(rows)


def load_role_basis(path: str, width: int):
    if not path:
        return None
    p=Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix.lower()==".safetensors":
        z=_load_st(str(p), device="cpu")
        keys={k.lower():k for k in z}
        k1=keys.get("mu1") or keys.get("mu_1")
        k2=keys.get("mu2") or keys.get("mu_2")
        if not k1 or not k2: raise KeyError("role basis safetensors needs mu1/mu2 keys")
        mu1=z[k1].float().reshape(-1).numpy();mu2=z[k2].float().reshape(-1).numpy()
    elif p.suffix.lower()==".npz":
        z=np.load(p);mu1=np.asarray(z["mu1"],np.float32).reshape(-1);mu2=np.asarray(z["mu2"],np.float32).reshape(-1)
    elif p.suffix.lower()==".csv":
        d=pd.read_csv(p);mu1=d["mu1"].to_numpy(np.float32);mu2=d["mu2"].to_numpy(np.float32)
    else:
        raise ValueError("role basis must be .safetensors, .npz, or .csv")
    if len(mu1)!=width or len(mu2)!=width:
        raise ValueError(f"role basis width mismatch {len(mu1)},{len(mu2)} vs {width}")
    mu1=mu1/(np.linalg.norm(mu1)+EPS);mu2=mu2-mu1*np.dot(mu1,mu2);mu2=mu2/(np.linalg.norm(mu2)+EPS)
    return np.stack([mu1,mu2],axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# residual capture
# -----------------------------------------------------------------------------

def frozen_register_mask(pre13_tbc: torch.Tensor, P: int, threshold: float, max_registers: int, min_registers: int):
    n=pre13_tbc[1:1+P].float().norm(dim=-1).T  # [B,P]
    B=n.shape[0];mask=torch.zeros_like(n,dtype=torch.bool)
    for i in range(B):
        idx=torch.nonzero(n[i]>=threshold,as_tuple=False).flatten()
        if idx.numel()<min_registers:
            idx=torch.topk(n[i],k=min(min_registers,P)).indices
        if max_registers>0 and idx.numel()>max_registers:
            vals=n[i,idx];idx=idx[torch.topk(vals,k=max_registers).indices]
        mask[i,idx]=True
    return mask,n


def _swap_axes(x: torch.Tensor, a: int, b: int, scope: str, P: int, rn_present: bool):
    y=x.clone()
    if scope=="all":
        rows=torch.arange(y.shape[0],device=y.device)
    elif scope=="no_rn" or scope=="ordinary":
        rows=torch.arange(min(P+1,y.shape[0]),device=y.device)
    elif scope=="spatial":
        rows=torch.arange(1,min(P+1,y.shape[0]),device=y.device)
    elif scope=="cls":
        rows=torch.tensor([0],device=y.device)
    else:
        raise ValueError(scope)
    tmp=y[rows,:,a].clone();y[rows,:,a]=y[rows,:,b];y[rows,:,b]=tmp
    return y


def qk_pair_terms(blk, ln1_tbc: torch.Tensor, pair: Tuple[int,int], regmask: torch.Tensor, P: int, block: int, stim_ids: Sequence[str]):
    a,b=pair;attn=blk.attn;Wq=attn.q_proj.weight.detach().float();Wk=attn.k_proj.weight.detach().float();H=attn.num_heads;D=attn.head_dim;scale=1.0/math.sqrt(D)
    x=ln1_tbc.permute(1,0,2).float()  # [B,T,E]
    # Attribution/audit math is intentionally recomputed in FP32. Calling the
    # module Linear directly is unsafe here because attn q/k weights are FP16
    # in the OpenAI-style mechinterp module while ln_1/residual may be FP32.
    bq = attn.q_proj.bias.detach().float() if attn.q_proj.bias is not None else None
    bk = attn.k_proj.bias.detach().float() if attn.k_proj.bias is not None else None
    qfull=F.linear(x[:,0],Wq,bq).view(x.shape[0],H,D)
    kfull=F.linear(x[:,1:1+P],Wk,bk).view(x.shape[0],P,H,D).permute(0,2,1,3)
    qcomp={c:(x[:,0,c,None]*Wq[:,c][None,:]).view(x.shape[0],H,D) for c in (a,b)}
    kcomp={c:(x[:,1:1+P,c,None]*Wk[:,c][None,None,:]).view(x.shape[0],P,H,D).permute(0,2,1,3) for c in (a,b)}
    rows=[]
    for i,sid in enumerate(stim_ids):
        groups={"register":regmask[i],"ordinary":~regmask[i],"all_patches":torch.ones(P,dtype=torch.bool,device=regmask.device)}
        for gname,m in groups.items():
            if int(m.sum())==0: continue
            kf=kfull[i,:,m,:].mean(1)
            exact=(qfull[i]*kf).sum(-1)*scale
            terms={}
            for qc in (a,b):
                for kc in (a,b):
                    kk=kcomp[kc][i,:,m,:].mean(1)
                    terms[(qc,kc)]=(qcomp[qc][i]*kk).sum(-1)*scale
            for h in range(H):
                rows.append({
                    "stim_id":sid,"block":block,"head":h,"source_group":gname,
                    "q_axis_a":a,"q_axis_b":b,
                    "exact_full_qk_logit":float(exact[h].cpu()),
                    f"q{a}_k{a}":float(terms[(a,a)][h].cpu()),
                    f"q{a}_k{b}":float(terms[(a,b)][h].cpu()),
                    f"q{b}_k{a}":float(terms[(b,a)][h].cpu()),
                    f"q{b}_k{b}":float(terms[(b,b)][h].cpu()),
                    "cross_pair_sum":float((terms[(a,b)][h]+terms[(b,a)][h]).cpu()),
                })
    return rows


@torch.inference_mode()
def capture_batch(variant: Variant, images: torch.Tensor, stim_ids: Sequence[str], axes: Sequence[int], role_basis: Optional[np.ndarray], args):
    v=variant.visual;blocks=list(v.transformer.resblocks);L=len(blocks);A=len(axes);axis_t=torch.tensor(axes,device=args.device,dtype=torch.long)
    images=images.to(dtype=v.conv1.weight.dtype)
    conv=v.conv1(images);B,C,Hg,Wg=conv.shape;P=Hg*Wg
    conv_bpc=conv.reshape(B,C,P).permute(0,2,1).float()
    pos=v.positional_embedding.to(dtype=conv.dtype,device=conv.device)
    if pos.shape[0]!=P+1: raise RuntimeError(f"positional tokens {pos.shape[0]} != patches+CLS {P+1}")
    preln=(conv_bpc + pos[1:].float().unsqueeze(0))
    x=v._prepare_tokens(images)
    init_postln=x[1:1+P].permute(1,0,2).float()

    shape_sp=(L,B,P,A);shape_cls=(L,B,A)
    pre_sp=torch.empty(shape_sp,dtype=torch.float32,device="cpu");pa_sp=torch.empty_like(pre_sp);pm_sp=torch.empty_like(pre_sp)
    pre_cls=torch.empty(shape_cls,dtype=torch.float32,device="cpu");pa_cls=torch.empty_like(pre_cls);pm_cls=torch.empty_like(pre_cls)
    pre_rn=torch.full(shape_cls,float("nan"),dtype=torch.float32);pa_rn=torch.full_like(pre_rn,float("nan"));pm_rn=torch.full_like(pre_rn,float("nan"))
    role_pre=role_pa=role_pm=None
    role_t=None
    if role_basis is not None:
        role_t=torch.from_numpy(role_basis).to(device=args.device,dtype=torch.float32)  # [2,C]
        role_pre=torch.empty((L,B,P,2),dtype=torch.float32);role_pa=torch.empty_like(role_pre);role_pm=torch.empty_like(role_pre)

    regmask=None;regnorm=None;qkrows=[]
    for li,blk in enumerate(blocks):
        x=variant.maybe_insert_rn(li,x)
        if li==args.register_block:
            regmask,regnorm=frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
        cur=x
        pre_sp[li]=cur[1:1+P].permute(1,0,2).index_select(-1,axis_t).float().cpu()
        pre_cls[li]=cur[0].index_select(-1,axis_t).float().cpu()
        if cur.shape[0]>P+1:
            pre_rn[li]=cur[-1].index_select(-1,axis_t).float().cpu()
        if role_t is not None:
            role_pre[li]=torch.einsum("bpc,kc->bpk",cur[1:1+P].permute(1,0,2).float(),role_t).cpu()
        ln1=blk.ln_1(cur)
        if li in args.qk_blocks:
            if regmask is None:
                tmpmask,_=frozen_register_mask(cur,P,args.register_threshold,args.max_registers,args.min_registers)
            else: tmpmask=regmask
            qkrows.extend(qk_pair_terms(blk,ln1,args.swap_pair,tmpmask,P,li,stim_ids))
        with amp_context(args.device, args.amp):
            attn_out,_=blk.attention(ln1,need_weights=False,capture=False)
        post_attn=cur+attn_out
        pa_sp[li]=post_attn[1:1+P].permute(1,0,2).index_select(-1,axis_t).float().cpu();pa_cls[li]=post_attn[0].index_select(-1,axis_t).float().cpu()
        if post_attn.shape[0]>P+1: pa_rn[li]=post_attn[-1].index_select(-1,axis_t).float().cpu()
        if role_t is not None: role_pa[li]=torch.einsum("bpc,kc->bpk",post_attn[1:1+P].permute(1,0,2).float(),role_t).cpu()
        with amp_context(args.device, args.amp):
            mlp_out=blk.mlp(blk.ln_2(post_attn))
        post=post_attn+mlp_out
        pm_sp[li]=post[1:1+P].permute(1,0,2).index_select(-1,axis_t).float().cpu();pm_cls[li]=post[0].index_select(-1,axis_t).float().cpu()
        if post.shape[0]>P+1: pm_rn[li]=post[-1].index_select(-1,axis_t).float().cpu()
        if role_t is not None: role_pm[li]=torch.einsum("bpc,kc->bpk",post[1:1+P].permute(1,0,2).float(),role_t).cpu()
        x=post
    if regmask is None:
        regmask,regnorm=frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
    with amp_context(args.device, args.amp):
        final_cls=v._finalize_cls(x)
    emb=normalize_t(final_cls).cpu()
    out={
        "conv1_raw":conv_bpc.index_select(-1,axis_t).cpu(),
        "init_preln":preln.index_select(-1,axis_t).cpu(),
        "init_postln":init_postln.index_select(-1,axis_t).cpu(),
        "pre_spatial":pre_sp,"post_attn_spatial":pa_sp,"post_mlp_spatial":pm_sp,
        "pre_cls":pre_cls,"post_attn_cls":pa_cls,"post_mlp_cls":pm_cls,
        "pre_rn":pre_rn,"post_attn_rn":pa_rn,"post_mlp_rn":pm_rn,
        "register_mask":regmask.cpu().to(torch.uint8),"register_pre_norm":regnorm.cpu(),"embedding":emb,
    }
    if role_t is not None:
        out.update({"role_pre":role_pre,"role_post_attn":role_pa,"role_post_mlp":role_pm})
    return out,pd.DataFrame(qkrows),P


def concat_capture(parts: Sequence[Dict[str,torch.Tensor]]):
    out={}
    for k in parts[0]:
        if k.startswith("role_") and parts[0][k] is None: continue
        dim=1 if k.startswith(("pre_spatial","post_attn_spatial","post_mlp_spatial","pre_cls","post_attn_cls","post_mlp_cls","pre_rn","post_attn_rn","post_mlp_rn","role_")) else 0
        out[k]=torch.cat([p[k] for p in parts],dim=dim)
    return out


def conv1_static(variant: Variant, axes: Sequence[int]):
    W=variant.visual.conv1.weight.detach().float().cpu().numpy();norm=np.linalg.norm(W.reshape(W.shape[0],-1),axis=1);z=robust_z(np.log(np.maximum(norm,EPS)));rows=[]
    rank_order=np.argsort(np.argsort(norm))+1
    for c in axes:
        pop="sparse_tail" if z[c]<-3 else ("transition_band" if z[c]<-1 else ("high_weight" if z[c]>=1 else "ordinary"))
        rows.append({"channel":c,"conv1_weight_l2":norm[c],"conv1_log_weight_robust_z":z[c],"conv1_weight_rank_low_to_high":int(rank_order[c]),"conv1_population":pop})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# lineage analysis / classification
# -----------------------------------------------------------------------------

def ridge_r2_beta(X: np.ndarray, y: np.ndarray, lam: float = 1e-3):
    X=np.asarray(X,np.float64);y=np.asarray(y,np.float64);m=np.isfinite(y)&np.all(np.isfinite(X),axis=1);X=X[m];y=y[m]
    if len(y)<max(10,X.shape[1]+2): return float("nan"),np.full(X.shape[1],np.nan)
    xm=X.mean(0);xs=X.std(0);xs[xs<EPS]=1;Xz=(X-xm)/xs;ym=y.mean();ys=y.std();
    if ys<EPS:return 0.0,np.zeros(X.shape[1])
    yz=(y-ym)/ys;G=Xz.T@Xz+lam*np.eye(Xz.shape[1]);beta=np.linalg.solve(G,Xz.T@yz);pred=Xz@beta;ss=np.sum((yz-pred)**2);tot=np.sum(yz**2);return float(1-ss/max(tot,EPS)),beta


def _stage_arrays(z, stage: str):
    if stage=="pre": return z["pre_spatial"].numpy(), z["pre_cls"].numpy(), z.get("role_pre")
    if stage=="post_attn": return z["post_attn_spatial"].numpy(), z["post_attn_cls"].numpy(), z.get("role_post_attn")
    if stage=="post_mlp": return z["post_mlp_spatial"].numpy(), z["post_mlp_cls"].numpy(), z.get("role_post_mlp")
    raise ValueError(stage)


def analyze_lineage(model_name: str, z: Dict[str,torch.Tensor], axes_df: pd.DataFrame, convstatic: pd.DataFrame, out: Path, args):
    axes=axes_df.channel.astype(int).tolist();A=len(axes);L=z["pre_spatial"].shape[0];N=z["conv1_raw"].shape[0];P=z["conv1_raw"].shape[1]
    conv=z["conv1_raw"].numpy();ipre=z["init_preln"].numpy();ipost=z["init_postln"].numpy();mask=z["register_mask"].numpy().astype(bool)
    X=ipost.reshape(-1,A);rows=[];cross=[]
    initial_rms=np.sqrt(np.mean(ipost.astype(np.float64)**2,axis=(0,1)))
    for stage in ("pre","post_attn","post_mlp"):
        arr,cls,role_t=_stage_arrays(z,stage);role=role_t.numpy() if role_t is not None else None
        for li in range(L):
            cur=arr[li]
            for j,c in enumerate(axes):
                y=cur[:,:,j];r2,beta=ridge_r2_beta(X,y.reshape(-1),args.ridge_lambda)
                own=float(beta[j]) if np.isfinite(beta[j]) else np.nan
                tmp=np.abs(beta).copy();tmp[j]=-np.inf;oi=int(np.nanargmax(tmp)) if np.isfinite(tmp).any() else j
                regv=y[mask];ordv=y[~mask];absreg=np.mean(np.abs(regv)) if regv.size else np.nan;absord=np.mean(np.abs(ordv)) if ordv.size else np.nan
                pb=corr(np.abs(y).reshape(-1),mask.reshape(-1).astype(float))
                rec={
                    "model":model_name,"stage":stage,"block":li,"channel":c,
                    "corr_conv1_raw":corr(y,conv[:,:,j]),"corr_init_preln":corr(y,ipre[:,:,j]),"corr_init_postln":corr(y,ipost[:,:,j]),
                    "selected_initial_r2":r2,"own_source_beta":own,"strongest_other_source_channel":axes[oi],"strongest_other_source_beta":float(beta[oi]) if np.isfinite(beta[oi]) else np.nan,
                    "spatial_rms":rms(y),"cls_rms":rms(cls[li,:,j]),"register_abs_mean":float(absreg),"ordinary_abs_mean":float(absord),
                    "register_abs_enrichment":float(absreg/max(absord,EPS)) if np.isfinite(absreg) and np.isfinite(absord) else np.nan,
                    "register_signed_minus_ordinary":float(np.mean(regv)-np.mean(ordv)) if regv.size and ordv.size else np.nan,
                    "register_abs_pointbiserial":pb,
                    "initial_rms":float(initial_rms[j]),"residual_gain_over_initial":float(rms(y)/max(initial_rms[j],EPS)),
                }
                if stage=="post_attn":
                    w=arr[li]-z["pre_spatial"][li].numpy();rec["attn_write_rms"]=rms(w[:,:,j]);rec["mlp_write_rms"]=np.nan
                elif stage=="post_mlp":
                    aw=z["post_attn_spatial"][li].numpy()-z["pre_spatial"][li].numpy();mw=arr[li]-z["post_attn_spatial"][li].numpy();rec["attn_write_rms"]=rms(aw[:,:,j]);rec["mlp_write_rms"]=rms(mw[:,:,j])
                else:rec["attn_write_rms"]=np.nan;rec["mlp_write_rms"]=np.nan
                rec["total_block_write_gain"]=(float(np.nan_to_num(rec["attn_write_rms"]))+float(np.nan_to_num(rec["mlp_write_rms"])))/max(initial_rms[j],EPS)
                if role is not None:
                    rec["corr_role_mu1"]=corr(y,role[li,:,:,0]);rec["corr_role_mu2"]=corr(y,role[li,:,:,1])
                else:rec["corr_role_mu1"]=np.nan;rec["corr_role_mu2"]=np.nan
                rows.append(rec)
            # full cross-lineage correlation matrix for this stage/block
            for j,c in enumerate(axes):
                y=cur[:,:,j]
                for k,src in enumerate(axes):
                    cross.append({"model":model_name,"stage":stage,"block":li,"channel":c,"source_initial_channel":src,"corr":corr(y,ipost[:,:,k])})
    df=pd.DataFrame(rows);xdf=pd.DataFrame(cross)
    write_df(df,out/"axis_lineage_metrics.csv");write_df(xdf,out/"axis_cross_lineage.csv.gz")

    # classification on block-boundary/post-MLP states
    post=df[df.stage.eq("post_mlp")].copy();summ=[]
    for c,g in post.groupby("channel"):
        g=g.sort_values("block");late=g[g.block>=max(0,L-4)];
        peak_write=float(g.total_block_write_gain.max());late_r2=float(late.selected_initial_r2.mean());late_corr=float(late.corr_init_postln.abs().mean());final_gain=float(g.iloc[-1].residual_gain_over_initial)
        takeover=None
        for r in g.itertuples(index=False):
            if abs(r.corr_init_postln)<args.takeover_corr and r.selected_initial_r2<args.takeover_r2 and r.total_block_write_gain>args.takeover_write_gain and r.residual_gain_over_initial>args.takeover_residual_gain:
                takeover=int(r.block);break
        summ.append({
            "model":model_name,"channel":int(c),"early_init_corr":float(abs(g.iloc[0].corr_init_postln)),"late_init_corr":late_corr,"late_selected_initial_r2":late_r2,
            "peak_total_write_gain":peak_write,"final_residual_gain":final_gain,"takeover_block":takeover,
            "peak_register_enrichment":float(g.register_abs_enrichment.max()),"peak_abs_mu1_corr":float(g.corr_role_mu1.abs().max()) if g.corr_role_mu1.notna().any() else np.nan,"peak_abs_mu2_corr":float(g.corr_role_mu2.abs().max()) if g.corr_role_mu2.notna().any() else np.nan,
        })
    s=pd.DataFrame(summ)
    for col in ("peak_total_write_gain","final_residual_gain","peak_register_enrichment"):
        s[col+"_z"]=robust_z(s[col].to_numpy())
    labels=[]
    for r in s.itertuples(index=False):
        outlier=max(r.peak_total_write_gain_z,r.final_residual_gain_z,r.peak_register_enrichment_z)
        if r.late_init_corr>=0.55:
            lab="persistent_inherited_axis"
        elif r.late_selected_initial_r2>=0.60:
            lab="rotated_or_mixed_inherited"
        elif r.late_selected_initial_r2<=0.35 and r.peak_total_write_gain>=1.5 and r.final_residual_gain>=1.2 and outlier>=1.0:
            lab="late_constructed_bus"
        elif r.late_selected_initial_r2<=0.35 and r.peak_total_write_gain<1.5:
            lab="faded_low_lineage"
        else:
            lab="hybrid_transition"
        labels.append(lab)
    s["residual_role_class"]=labels
    s=s.merge(convstatic,on="channel",how="left").merge(axes_df[["channel","is_special","is_control","tags"]],on="channel",how="left")
    s["combined_class"]=s.conv1_population.astype(str)+" -> "+s.residual_role_class.astype(str)
    write_df(s,out/"axis_classification.csv")
    return df,xdf,s


# -----------------------------------------------------------------------------
# plots / text summary
# -----------------------------------------------------------------------------

def plot_axis_focus(metrics: pd.DataFrame, classification: pd.DataFrame, axes_df: pd.DataFrame, out: Path):
    pdir=out/"plots"/"axis_focus";pdir.mkdir(parents=True,exist_ok=True)
    specials=axes_df[axes_df.is_special.astype(bool)].channel.astype(int).tolist()
    for c in specials:
        g=metrics[(metrics.channel==c)&(metrics.stage=="post_mlp")].sort_values("block")
        if g.empty:continue
        fig,axs=plt.subplots(2,2,figsize=(13,9));x=g.block
        axs[0,0].plot(x,g.corr_conv1_raw,label="corr raw Conv1");axs[0,0].plot(x,g.corr_init_postln,label="corr transformer input");axs[0,0].plot(x,g.selected_initial_r2,label="R² all selected initial axes");axs[0,0].axhline(0,color="0.5",lw=.8);axs[0,0].set_ylim(-1.05,1.05);axs[0,0].set_title("lineage / inherited-source fit");axs[0,0].legend(fontsize=8)
        axs[0,1].plot(x,g.spatial_rms,label="residual RMS");axs[0,1].plot(x,g.attn_write_rms,label="attention write RMS");axs[0,1].plot(x,g.mlp_write_rms,label="MLP write RMS");axs[0,1].set_title("wire amplitude and new writes");axs[0,1].legend(fontsize=8)
        axs[1,0].plot(x,g.register_abs_enrichment,label="|axis| REG/ordinary");axs[1,0].plot(x,g.register_abs_pointbiserial,label="corr(|axis|, REG mask)");axs[1,0].axhline(1,color="0.6",ls="--",lw=.8);axs[1,0].set_title("register/workspace association");axs[1,0].legend(fontsize=8)
        if g.corr_role_mu1.notna().any():
            axs[1,1].plot(x,g.corr_role_mu1,label="corr role μ1");axs[1,1].plot(x,g.corr_role_mu2,label="corr role μ2")
        axs[1,1].plot(x,g.own_source_beta,label="own-source β");axs[1,1].plot(x,g.strongest_other_source_beta,label="strongest other β");axs[1,1].set_title("role / source reassignment");axs[1,1].legend(fontsize=8)
        for ax in axs.flat:ax.grid(alpha=.15);ax.set_xlabel("ViT block")
        cc=classification[classification.channel==c].iloc[0]
        fig.suptitle(f"ch{c:04d}: {cc.combined_class} | takeover={cc.takeover_block}",fontsize=14);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(pdir/f"ch{c:04d}_lineage.png",dpi=190);plt.close(fig)


def plot_classification(classification: pd.DataFrame, out: Path):
    fig,ax=plt.subplots(figsize=(11,8));ctrl=classification.is_control.astype(bool)
    ax.scatter(classification.loc[ctrl,"late_selected_initial_r2"],classification.loc[ctrl,"peak_total_write_gain"],alpha=.45,label="random controls")
    ax.scatter(classification.loc[~ctrl,"late_selected_initial_r2"],classification.loc[~ctrl,"peak_total_write_gain"],marker="x",s=65,label="special axes")
    for r in classification[~ctrl].itertuples(index=False):ax.annotate(str(r.channel),(r.late_selected_initial_r2,r.peak_total_write_gain),fontsize=8,xytext=(3,3),textcoords="offset points")
    ax.set_xlabel("late R² from selected initial axes");ax.set_ylabel("peak block-write gain / initial RMS");ax.set_title("Inherited-axis vs constructed-bus diagnostic");ax.grid(alpha=.15);ax.legend();fig.tight_layout();fig.savefig(out/"plots"/"axis_classification_scatter.png",dpi=200);plt.close(fig)


def plot_pair_lineage(cross: pd.DataFrame, pair: Tuple[int,int], out: Path):
    a,b=pair;g=cross[(cross.stage=="post_mlp") & (cross.channel.isin([a,b])) & (cross.source_initial_channel.isin([a,b]))]
    if g.empty:return
    fig,ax=plt.subplots(figsize=(10,6))
    for (c,s),q in g.groupby(["channel","source_initial_channel"]):ax.plot(q.block,q["corr"],marker="o",label=f"current {c} vs init {s}")
    ax.axhline(0,color="0.5",lw=.8);ax.set_ylim(-1.05,1.05);ax.set_xlabel("ViT block");ax.set_ylabel("Pearson correlation");ax.set_title(f"{a}<->{b}: 2x2 retinal-lineage matrix");ax.grid(alpha=.15);ax.legend();fig.tight_layout();fig.savefig(out/"plots"/f"pair_{a}_{b}_lineage_matrix.png",dpi=200);plt.close(fig)


def plot_qk(qk: pd.DataFrame, pair: Tuple[int,int], out: Path):
    if qk.empty:return
    a,b=pair;pdir=out/"plots"/"qk";pdir.mkdir(parents=True,exist_ok=True)
    for block in sorted(qk.block.unique()):
        for group in ("register","ordinary"):
            g=qk[(qk.block==block)&(qk.source_group==group)]
            if g.empty:continue
            m=g.groupby("head")[[f"q{a}_k{a}",f"q{a}_k{b}",f"q{b}_k{a}",f"q{b}_k{b}","cross_pair_sum"]].mean()
            fig,ax=plt.subplots(figsize=(12,6))
            for col in m.columns:ax.plot(m.index,m[col],marker="o",label=col)
            ax.axhline(0,color="0.5",lw=.8);ax.set_xlabel("attention head");ax.set_ylabel("mean coordinate QK logit contribution");ax.set_title(f"B{block} {group}: {a}/{b} coordinate QK terms");ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(pdir/f"B{block}_{group}_qk_{a}_{b}.png",dpi=190);plt.close(fig)


def write_classification_report(classification: pd.DataFrame, out: Path):
    sp=classification[classification.is_special].sort_values(["residual_role_class","channel"])
    lines=["SPECIAL RESIDUAL AXIS CLASSIFICATION","="*40,"","Heuristic labels are descriptive, not ontological ground truth.","Conv1 provenance and late residual role are intentionally reported separately.",""]
    for cls,g in sp.groupby("residual_role_class",sort=False):
        lines += [cls.upper(),"-"*len(cls)]
        for r in g.itertuples(index=False):
            lines.append(f"ch{r.channel:04d}  {r.conv1_population:16s}  lateR2={r.late_selected_initial_r2:.3f} lateOwnCorr={r.late_init_corr:.3f} peakWrite={r.peak_total_write_gain:.2f} finalGain={r.final_residual_gain:.2f} takeover={r.takeover_block}  tags={r.tags}")
        lines.append("")
    (out/"SPECIAL_AXIS_CLASSIFICATION.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


# -----------------------------------------------------------------------------
# swap causal sweep
# -----------------------------------------------------------------------------

@torch.inference_mode()
def forward_swap(variant: Variant, images: torch.Tensor, pair: Tuple[int,int], mode: str, block_k: Optional[int], scope: str, amp: bool = True):
    v=variant.visual;images=images.to(dtype=v.conv1.weight.dtype);x=v._prepare_tokens(images);P=x.shape[0]-1;a,b=pair
    if mode=="full":x=_swap_axes(x,a,b,scope,P,False)
    for li,blk in enumerate(v.transformer.resblocks):
        x=variant.maybe_insert_rn(li,x);rn=x.shape[0]>P+1
        do_onset=(mode=="onset" and li==block_k);do_window=(mode=="window" and li==block_k)
        if do_onset or do_window:x=_swap_axes(x,a,b,scope,P,rn)
        ln1=blk.ln_1(x)
        with amp_context(str(images.device), amp):
            attn_out,_=blk.attention(ln1,need_weights=False,capture=False)
        pa=x+attn_out
        with amp_context(str(images.device), amp):
            mlp_out=blk.mlp(blk.ln_2(pa))
        post=pa+mlp_out
        if do_window:post=_swap_axes(post,a,b,scope,P,rn)
        x=post
    with amp_context(str(images.device), amp):
        final_cls=v._finalize_cls(x)
    return normalize_t(final_cls).cpu()


def manifold_margin(emb: np.ndarray, manifest: pd.DataFrame):
    em=emb/np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),EPS);idx={str(r.stim_id):i for i,r in manifest.iterrows()};same=[];unrel=[]
    for pair,g in manifest.groupby("pair"):
        if len(g)>=2:
            ids=g.stim_id.astype(str).tolist();
            for i in range(len(ids)):
                for j in range(i+1,len(ids)):same.append(float(np.dot(em[idx[ids[i]]],em[idx[ids[j]]])))
    pairs=manifest.pair.astype(str).to_numpy()
    for i in range(len(manifest)):
        for j in range(i+1,len(manifest)):
            if pairs[i]!=pairs[j]:unrel.append(float(np.dot(em[i],em[j])))
    sm=float(np.mean(same)) if same else np.nan;um=float(np.mean(unrel)) if unrel else np.nan
    return sm,um,sm-um if np.isfinite(sm) and np.isfinite(um) else np.nan


@torch.inference_mode()
def run_swap_sweep(variant: Variant, manifest: pd.DataFrame, args, out: Path):
    emb_path=out/"swap_embeddings.safetensors";meta_path=out/"swap_event_metadata.csv";rows_path=out/"swap_per_image.csv";sweep_cfg=out/"swap_config.json"
    L=len(variant.visual.transformer.resblocks);events=[("baseline",None),("full",0)]+[("onset",k) for k in range(L)]+[("window",k) for k in range(L)]

    # Event-level cache: each expensive forward group is committed immediately.
    # This keeps a late plotting/serialization failure from discarding a completed sweep.
    swap_sig={
        "model":variant.name,
        "n_blocks":L,
        "pair":[int(args.swap_pair[0]),int(args.swap_pair[1])],
        "scope":args.swap_token_scope,
        "amp":bool(args.amp),
        "n_images":len(manifest),
        "stim_ids":manifest.stim_id.astype(str).tolist(),
    }
    if args.resume and emb_path.is_file() and meta_path.is_file() and rows_path.is_file() and sweep_cfg.is_file():
        try:
            old_cfg=json.loads(sweep_cfg.read_text(encoding="utf-8"))
        except Exception:
            old_cfg=None
        if old_cfg==swap_sig:
            print(f"[{variant.name} swap] resume: loading validated saved sweep")
            return pd.read_csv(rows_path),pd.read_csv(meta_path),_load_st(str(emb_path),device="cpu")
        print(f"[{variant.name} swap] consolidated files exist but signature is missing/mismatched; rebuilding from event cache")

    sig_json=json.dumps(swap_sig,sort_keys=True,separators=(",",":"))
    sig_hash=hashlib.sha256(sig_json.encode("utf-8")).hexdigest()[:16]
    cache_dir=out/"swap_event_cache"/sig_hash;cache_dir.mkdir(parents=True,exist_ok=True)
    cache_cfg=cache_dir/"cache_signature.json"
    if not cache_cfg.is_file():cache_cfg.write_text(json.dumps(swap_sig,indent=2),encoding="utf-8")

    per_event=[];event_rows=[];all_rows=[]
    for ei,(mode,bk) in enumerate(events):
        btag="none" if bk is None else f"{int(bk):02d}"
        event_path=cache_dir/f"event_{ei:03d}_{mode}_b{btag}.safetensors"
        E=None
        if args.resume and event_path.is_file():
            got=_load_st(str(event_path),device="cpu")
            candidate=got.get("embeddings")
            if candidate is not None and candidate.ndim==2 and candidate.shape[0]==len(manifest):
                E=candidate.float().contiguous()
                print(f"[{variant.name} swap {ei+1}/{len(events)}] resume {mode} block={bk}")
        if E is None:
            parts=[]
            for st in range(0,len(manifest),args.batch_size):
                meta=manifest.iloc[st:st+args.batch_size];batch=load_batch(variant.preprocess,meta,args.device)
                if mode=="baseline": emb=forward_swap(variant,batch,args.swap_pair,"onset",-999,args.swap_token_scope,args.amp)
                else:emb=forward_swap(variant,batch,args.swap_pair,mode,bk,args.swap_token_scope,args.amp)
                parts.append(emb);del batch
            E=torch.cat(parts,0).float().cpu().contiguous()
            # Save before moving to the next event.
            save_safetensors({"embeddings":E},str(event_path),metadata={"event_index":str(ei),"mode":mode,"block":str(bk),"signature":sig_hash})
            print(f"[{variant.name} swap {ei+1}/{len(events)}] {mode} block={bk}")
        per_event.append(E);event_rows.append({"event_index":ei,"mode":mode,"block":bk})

    stack=torch.stack(per_event,0).contiguous();base=stack[0].clone().contiguous()
    for ei,(mode,bk) in enumerate(events):
        dist=cosine_distance_rows(stack[ei],base).numpy();sm,um,margin=manifold_margin(stack[ei].numpy(),manifest)
        for i,r in enumerate(manifest.itertuples(index=False)):
            all_rows.append({"model":variant.name,"event_index":ei,"mode":mode,"block":bk,"stim_id":r.stim_id,"source":r.source,"pair":r.pair,"final_cosine_distance":float(dist[i]),"same_scene_mean_cos":sm,"unrelated_scene_mean_cos":um,"same_minus_unrelated_margin":margin})
    save_safetensors({"embeddings":stack,"baseline":base},str(emb_path),metadata={"pair":f"{args.swap_pair[0]},{args.swap_pair[1]}","scope":args.swap_token_scope,"cache_signature":sig_hash})
    write_df(pd.DataFrame(event_rows),meta_path);df=pd.DataFrame(all_rows);write_df(df,rows_path)
    sweep_cfg.write_text(json.dumps(swap_sig,indent=2),encoding="utf-8")
    return df,pd.DataFrame(event_rows),{"embeddings":stack,"baseline":base}


def summarize_swap(df: pd.DataFrame, out: Path):
    rows=[]
    for (model,mode,bk),g in df.groupby(["model","mode","block"],dropna=False):
        rec={"model":model,"mode":mode,"block":bk,"n":len(g),"mean_final_cosine_distance":g.final_cosine_distance.mean(),"max_final_cosine_distance":g.final_cosine_distance.max(),"same_scene_mean_cos":g.same_scene_mean_cos.iloc[0],"unrelated_scene_mean_cos":g.unrelated_scene_mean_cos.iloc[0],"same_minus_unrelated_margin":g.same_minus_unrelated_margin.iloc[0]}
        for src in ("clean","adv"):
            q=g[g.source==src];rec[f"mean_dist_{src}"]=q.final_cosine_distance.mean() if len(q) else np.nan
        rows.append(rec)
    s=pd.DataFrame(rows);write_df(s,out/"swap_summary.csv");return s


def plot_swap(summary: pd.DataFrame, out: Path, pair: Tuple[int,int]):
    pdir=out/"plots"/"swap";pdir.mkdir(parents=True,exist_ok=True);a,b=pair
    for mode in ("onset","window"):
        g=summary[summary["mode"].eq(mode)].sort_values("block")
        if g.empty:continue
        fig,ax=plt.subplots(figsize=(11,6));ax.plot(g.block,g.mean_dist_clean,marker="o",label="clean");ax.plot(g.block,g.mean_dist_adv,marker="o",label="adv");ax.plot(g.block,g.mean_final_cosine_distance,marker="o",label="all")
        ax.set_xlabel("swap block");ax.set_ylabel("final embedding cosine distance");ax.set_title(f"{a}<->{b} {mode} cross-wiring");ax.grid(alpha=.15);ax.legend();fig.tight_layout();fig.savefig(pdir/f"{mode}_{a}_{b}_final_distance.png",dpi=200);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,6));ax.plot(g.block,g.same_minus_unrelated_margin,marker="o");ax.set_xlabel("swap block");ax.set_ylabel("same-scene - unrelated cosine margin");ax.set_title(f"{a}<->{b} {mode}: manifold margin");ax.grid(alpha=.15);fig.tight_layout();fig.savefig(pdir/f"{mode}_{a}_{b}_margin.png",dpi=200);plt.close(fig)


# -----------------------------------------------------------------------------
# orchestration
# -----------------------------------------------------------------------------

def capture_model(variant: Variant, manifest: pd.DataFrame, axes: Sequence[int], axes_df: pd.DataFrame, role_basis, args, out: Path):
    raw_path=out/"lineage_states.safetensors";qk_path=out/"qk_pair_terms.csv.gz";cfg_path=out/"lineage_config.json"
    signature={"model":variant.name,"axes":list(map(int,axes)),"n_images":len(manifest),"stim_ids":manifest.stim_id.astype(str).tolist(),"qk_blocks":args.qk_blocks,"register_block":args.register_block,"role_basis":args.role_basis or None,"amp":bool(args.amp)}
    if args.resume and raw_path.is_file() and cfg_path.is_file():
        old=json.loads(cfg_path.read_text(encoding="utf-8"))
        if old==signature:
            print(f"[{variant.name}] lineage resume: loading saved states")
            z=_load_st(str(raw_path),device="cpu");qk=pd.read_csv(qk_path) if qk_path.is_file() else pd.DataFrame();return z,qk
    parts=[];qks=[];P=None
    for st in range(0,len(manifest),args.batch_size):
        meta=manifest.iloc[st:st+args.batch_size];batch=load_batch(variant.preprocess,meta,args.device);cap,qk,p=capture_batch(variant,batch,meta.stim_id.astype(str).tolist(),axes,role_basis,args);parts.append(cap);qks.append(qk);P=p;del batch;gc.collect()
        print(f"[{variant.name} lineage] {min(st+args.batch_size,len(manifest))}/{len(manifest)}")
    z=concat_capture(parts);save_safetensors(z,str(raw_path),metadata={"model":variant.name,"patches":str(P),"axes":",".join(map(str,axes))});qk=pd.concat(qks,ignore_index=True) if qks else pd.DataFrame();write_df(qk,qk_path);cfg_path.write_text(json.dumps(signature,indent=2),encoding="utf-8");return z,qk


def variant_audit(variant: Variant, args):
    v=variant.visual;return {"variant":variant.name,"source_info":variant.source_info,"vision_width":int(v.conv1.weight.shape[0]),"input_resolution":int(v.input_resolution),"n_blocks":len(v.transformer.resblocks),"n_heads":int(v.transformer.resblocks[0].attn.num_heads),"rn_transplant":variant.rn_token is not None,"rn_insert_block":variant.rn_insert_block,"full_xattn":variant.is_full_xattn}


def self_test():
    x=torch.randn(4,3,8);y=_swap_axes(x,2,5,"all",2,False);assert torch.allclose(y[:,:,2],x[:,:,5]) and torch.allclose(y[:,:,5],x[:,:,2]);z=_swap_axes(y,2,5,"all",2,False);assert torch.allclose(z,x)
    rng=np.random.default_rng(1);X=rng.normal(size=(1000,4));y=1.5*X[:,0]-.7*X[:,2]+.05*rng.normal(size=1000);r2,b=ridge_r2_beta(X,y);assert r2>.99 and abs(b[0])>abs(b[1])
    # Regression test for safetensors shared-storage rejection: base is a view into stack.
    stack=torch.randn(3,4,5);base=stack[0]
    with tempfile.TemporaryDirectory() as td:
        sp=Path(td)/"shared.safetensors";save_safetensors({"embeddings":stack,"baseline":base},str(sp));got=_load_st(str(sp),device="cpu")
        assert torch.allclose(got["embeddings"],stack) and torch.allclose(got["baseline"],base)
    # Regression test: DataFrame.mode is a method, so column access must be explicit.
    swap_summary=pd.DataFrame({"mode":["onset","window"],"block":[0,1]})
    assert len(swap_summary[swap_summary["mode"].eq("onset")])==1
    with tempfile.TemporaryDirectory() as td:
        base=Path(td);legacy=base/"pretrained";legacy.mkdir()
        (legacy/"lineage_states.safetensors").write_bytes(b"L")
        ec=legacy/"swap_event_cache"/"abc";ec.mkdir(parents=True);(ec/"event.safetensors").write_bytes(b"S")
        lr=base/"lineage";sr=base/"swap_650_565_all"
        migrate_legacy_stage(base,lr,"lineage",["pretrained"]);migrate_legacy_stage(base,sr,"swap",["pretrained"])
        assert (lr/"pretrained"/"lineage_states.safetensors").read_bytes()==b"L"
        assert (sr/"pretrained"/"swap_event_cache"/"abc"/"event.safetensors").read_bytes()==b"S"
        assert (legacy/"lineage_states.safetensors").exists() and (legacy/"swap_event_cache").exists()
    print("self-test passed")


def build_parser():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--suite",choices=["lineage","swap","all"],default="all")
    p.add_argument("--models",default=DEFAULT_MODELS)
    p.add_argument("--image_dir",default=DEFAULT_IMAGE_DIR);p.add_argument("--recursive",action="store_true")
    p.add_argument("--out_root",default=DEFAULT_OUT_ROOT);p.add_argument("--repo_root",default="")
    p.add_argument("--pretrained_model",default=DEFAULT_PRETRAINED);p.add_argument("--gmp_checkpoint",default=DEFAULT_GMP);p.add_argument("--xattn_model",default=DEFAULT_XATTN);p.add_argument("--xattn_revision",default="");p.add_argument("--hf_cache_dir",default="")
    p.add_argument("--device",default="cuda");p.add_argument("--batch_size",type=int,default=4);p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True,help="use CUDA FP16 autocast for manual CLIP block forwards; required by mixed-precision OpenAI-style QKV/proj weights")
    p.add_argument("--axes",type=parse_int_list,default=list(DEFAULT_SPECIAL_AXES));p.add_argument("--control_count",type=int,default=24)
    p.add_argument("--register_block",type=int,default=13);p.add_argument("--register_threshold",type=float,default=70.0);p.add_argument("--max_registers",type=int,default=4);p.add_argument("--min_registers",type=int,default=1)
    p.add_argument("--qk_blocks",type=parse_int_list,default=[22]);p.add_argument("--swap_pair",type=parse_int_list,default=list(DEFAULT_SWAP_PAIR));p.add_argument("--swap_token_scope",choices=["all","no_rn","ordinary","spatial","cls"],default="all")
    p.add_argument("--rn_insert_block",type=int,default=13)
    p.add_argument("--role_basis",default="",help="optional mu1/mu2 basis: safetensors, npz, or csv")
    p.add_argument("--ridge_lambda",type=float,default=1e-3)
    p.add_argument("--takeover_corr",type=float,default=.25);p.add_argument("--takeover_r2",type=float,default=.40);p.add_argument("--takeover_write_gain",type=float,default=1.0);p.add_argument("--takeover_residual_gain",type=float,default=1.2)
    p.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--migrate_legacy",action=argparse.BooleanOptionalAction,default=True,help="copy reusable v2.3 flat-layout artifacts into separate lineage/swap folders; never deletes legacy data")
    p.add_argument("--self_test",action="store_true")
    return p


def main():
    args=build_parser().parse_args();seed_all(args.seed)
    if args.self_test:self_test();return 0
    if len(args.swap_pair)!=2:raise ValueError("--swap_pair requires exactly two axes")
    args.swap_pair=(int(args.swap_pair[0]),int(args.swap_pair[1]));args.models=parse_str_list(args.models)

    base_root=Path(args.out_root);base_root.mkdir(parents=True,exist_ok=True)
    lineage_root,swap_root=stage_output_roots(base_root,args)
    do_lineage=args.suite in {"lineage","all"};do_swap=args.suite in {"swap","all"}
    if do_lineage:lineage_root.mkdir(parents=True,exist_ok=True)
    if do_swap:swap_root.mkdir(parents=True,exist_ok=True)
    if args.migrate_legacy:
        if do_lineage:migrate_legacy_stage(base_root,lineage_root,"lineage",args.models)
        if do_swap:migrate_legacy_stage(base_root,swap_root,"swap",args.models)

    manifest=scan_images(Path(args.image_dir),args.recursive);print(f"[manifest] {len(manifest)} images")
    if do_lineage:write_df(manifest,lineage_root/"image_manifest.csv")
    if do_swap:write_df(manifest,swap_root/"image_manifest.csv")
    (base_root/"OUTPUT_LAYOUT.txt").write_text(
        f"lineage={lineage_root}\nswap={swap_root}\n"
        "The two stage directories are intentionally disjoint. No swap run writes into lineage outputs.\n",
        encoding="utf-8",
    )

    combined_cls=[];combined_metrics=[];combined_swap=[]
    for mi,name in enumerate(args.models):
        print(f"\n=== MODEL {name} ===");variant=load_variant(name,args);audit=variant_audit(variant,args)
        width=int(variant.visual.conv1.weight.shape[0]);axes,special,controls,axes_df=select_axes(args,width);convstatic=conv1_static(variant,axes);role_basis=load_role_basis(args.role_basis,width)
        if do_lineage:
            mdir=lineage_root/name;mdir.mkdir(parents=True,exist_ok=True)
            (mdir/"model_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
            write_df(axes_df,mdir/"axis_manifest.csv");write_df(convstatic,mdir/"conv1_axis_static.csv")
            z,qk=capture_model(variant,manifest,axes,axes_df,role_basis,args,mdir);metrics,cross,cls=analyze_lineage(name,z,axes_df,convstatic,mdir,args);plot_axis_focus(metrics,cls,axes_df,mdir);plot_classification(cls,mdir);plot_pair_lineage(cross,args.swap_pair,mdir);plot_qk(qk,args.swap_pair,mdir);write_classification_report(cls,mdir);combined_cls.append(cls);combined_metrics.append(metrics)
        if do_swap:
            smdir=swap_root/name;smdir.mkdir(parents=True,exist_ok=True)
            (smdir/"model_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
            write_df(axes_df,smdir/"axis_manifest.csv");write_df(convstatic,smdir/"conv1_axis_static.csv")
            sdf,emeta,st=run_swap_sweep(variant,manifest,args,smdir);ss=summarize_swap(sdf,smdir);plot_swap(ss,smdir,args.swap_pair);combined_swap.append(ss)
        del variant;gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if combined_cls:
        allc=pd.concat(combined_cls,ignore_index=True);write_df(allc,lineage_root/"ALL_MODELS_axis_classification.csv")
        allm=pd.concat(combined_metrics,ignore_index=True);write_df(allm,lineage_root/"ALL_MODELS_axis_lineage_metrics.csv.gz")
        piv=allc[allc.is_special].pivot_table(index="channel",columns="model",values=["late_init_corr","late_selected_initial_r2","peak_total_write_gain","final_residual_gain"],aggfunc="first");piv.to_csv(lineage_root/"ALL_MODELS_special_axis_comparison.csv")
        (lineage_root/"RUN_CONFIG.json").write_text(json.dumps({**vars(args),"resolved_stage_root":str(lineage_root)},indent=2,default=str),encoding="utf-8")
    if combined_swap:
        write_df(pd.concat(combined_swap,ignore_index=True),swap_root/"ALL_MODELS_swap_summary.csv")
        (swap_root/"RUN_CONFIG.json").write_text(json.dumps({**vars(args),"resolved_stage_root":str(swap_root)},indent=2,default=str),encoding="utf-8")
    print(f"\n[done] base={base_root}")
    if do_lineage:print(f"[done] lineage={lineage_root}")
    if do_swap:print(f"[done] swap={swap_root}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
