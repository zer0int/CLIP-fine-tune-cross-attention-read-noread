#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModeMUX/x-attention Conv1 functional atlas, outlier census, and synergy screen.

Consolidates the useful analysis from the historical low-tail functional phase/
synergy and severe-manifold cluster scripts into one release-facing runner that
loads the current full x-attention ModeMUX checkpoint directly.

Main stages
-----------
1. Static Conv1 morphology census + natural-response / positional statistics.
2. Optional same-channel pretrained Conv1 drift audit.
3. Cheap global screen over ALL Conv1 channels (default FLIP + SHUFFLE on a
   deterministic image subset) so reorganized/"retired" axes are not missed.
4. Full seven-condition causal scan on an automatically selected candidate pool:
      ZERO, FLIP, ABS, SIGN, DC_ONLY, CENTERED, SHUFFLE
5. Legacy-compatible low-tail phase diagrams + severe/outlier functional clusters.
6. BACKBONE and corrected CONTENT embedding manifolds, displacement manifolds,
   and stack nonlinear-residual manifolds.
7. Same-image stacks, condition-family stacks, and a cross-condition pair screen.
8. Human-readable CULPRIT_SHORTLIST.txt + experiment_candidates.json that can be
   pasted into the Conv1 GPIC manifold explorer without combing through logs.

The x-attention-specific additions are intentional:
* every intervention is measured in BOTH raw BACKBONE and corrected CONTENT space;
* ``repair = backbone_distance - content_distance`` identifies perturbations the
  learned content correction suppresses versus amplifies;
* final implicit-register count/Jaccard, frozen pre-B13 register geometry,
  SOURCE gate/logits, READ_NULL score/attention, and register-coupling head
  signatures are recorded;
* candidate discovery starts globally, rather than assuming that the current
  model's interesting coordinates remain in the pretrained low-weight tail.

All vector artifacts are safetensors. CSV/CSV.GZ files contain the rich logs.
No pickle serialization is used.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    linear_sum_assignment = None
    HAVE_SCIPY = False

try:
    import umap  # type: ignore
    HAVE_UMAP = True
except Exception:
    umap = None
    HAVE_UMAP = False

try:
    from safetensors.torch import save_file as _save_safetensors_raw
except Exception as exc:  # pragma: no cover
    raise RuntimeError("safetensors is required: pip install safetensors") from exc


def save_safetensors(tensors: Mapping[str, torch.Tensor], filename: str, metadata: Optional[Mapping[str, str]] = None) -> None:
    """Save tensors safely even when inputs are views/non-contiguous.

    safetensors intentionally rejects non-contiguous tensors.  Analysis arrays
    such as positional maps can naturally be transposed/sliced views, so make
    serialization ownership explicit at the boundary instead of requiring every
    producer to remember to pack its output.
    """
    packed = {}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"safetensors value {name!r} is not a torch.Tensor: {type(tensor).__name__}")
        packed[name] = tensor.detach().contiguous()
    meta = None if metadata is None else {str(k): str(v) for k, v in metadata.items()}
    _save_safetensors_raw(packed, filename, metadata=meta)


# -----------------------------------------------------------------------------
# Defaults / constants
# -----------------------------------------------------------------------------

EPS = 1e-12
SEED = 20260916
DEFAULT_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_PRETRAINED = "openai/clip-vit-large-patch14"
DEFAULT_IMAGE_DIR = r"image_sets/special_natural"
DEFAULT_OUT = r"outputs/conv1_xattn_functional_atlas"

ALL_CONDITIONS = ("ZERO", "FLIP", "ABS", "SIGN", "DC_ONLY", "CENTERED", "SHUFFLE")
DEFAULT_SCREEN_CONDITIONS = ("FLIP", "SHUFFLE")
LOW_POPS = ("sparse_tail", "transition_band")

COND_COLORS = {
    "ZERO": "#7f7f7f",
    "FLIP": "#d62728",
    "ABS": "#ff7f0e",
    "SIGN": "#9467bd",
    "DC_ONLY": "#2ca02c",
    "CENTERED": "#1f77b4",
    "SHUFFLE": "#17becf",
    "BASELINE": "#111111",
}

# Morphology-only features. Weight magnitude is intentionally excluded from
# clustering and twin matching, matching the old atlas logic.
MORPH_FEATURES = (
    "lum_energy_frac", "rg_energy_frac", "by_energy_frac",
    "dc_frac", "low_frac", "mid_frac", "high_frac",
    "spectral_centroid", "spectral_bandwidth", "spectral_entropy",
    "radial_peak", "ring_frac", "orientation_anisotropy",
    "freq_axis_cos2", "freq_axis_sin2",
    "vertical_axis_frac", "horizontal_axis_frac", "corner_freq_frac",
    "symmetry180", "symmetry_lr", "symmetry_ud",
    "center_spatial_frac", "edge_spatial_frac",
    "dog_abs_corr", "gabor_abs_corr",
)

TWIN_FEATURE_DOMAINS: Mapping[str, Tuple[str, ...]] = {
    "color": ("lum_energy_frac", "rg_energy_frac", "by_energy_frac"),
    "frequency": (
        "dc_frac", "low_frac", "mid_frac", "high_frac",
        "spectral_centroid", "spectral_bandwidth", "spectral_entropy",
        "radial_peak", "ring_frac", "orientation_anisotropy",
        "freq_axis_cos2", "freq_axis_sin2",
        "vertical_axis_frac", "horizontal_axis_frac", "corner_freq_frac",
    ),
    "symmetry": ("symmetry180", "symmetry_lr", "symmetry_ud"),
    "spatial": ("center_spatial_frac", "edge_spatial_frac"),
    "shape": ("dog_abs_corr", "gabor_abs_corr"),
}


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(x: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x)).strip("_") or "item"


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def parse_float_list(spec: str) -> List[float]:
    return [float(x.strip()) for x in str(spec).split(",") if x.strip()]


def parse_str_list(spec: str) -> List[str]:
    return [x.strip().upper() for x in str(spec).split(",") if x.strip()]


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > EPS else float("nan")


def cosine_distance_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine_np(a, b))


def cosine_dist_rows_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1, eps=EPS)


def normalize_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), EPS)


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.nanstd(x)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return (x - med) / scale


def robust_log_z(norms: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y = np.log(np.asarray(norms, np.float64).clip(min=EPS))
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    sigma = 1.4826 * mad
    if sigma < 1e-12:
        sigma = float(np.std(y, ddof=0))
    if sigma < 1e-12:
        sigma = 1.0
    return (y - med) / sigma, med, sigma


def percentile01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() <= 1:
        return pd.Series(np.zeros(len(x), dtype=float), index=s.index)
    return x.rank(method="average", pct=True).fillna(0.0)


def population_group(z: float) -> str:
    if z < -3.0:
        return "sparse_tail"
    if z < -1.0:
        return "transition_band"
    return "other"


def write_df(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gz":
        df.to_csv(path, index=index, compression="gzip")
    else:
        df.to_csv(path, index=index)


def choose_stride_subset(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.copy().reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, n, dtype=int)
    return df.iloc[np.unique(idx)].copy().reset_index(drop=True)


def source_from_stem(stem: str) -> str:
    return "adv" if str(stem).lower().endswith("_adv") else "clean"


def pair_from_stem(stem: str) -> str:
    s = str(stem)
    return s[:-4] if s.lower().endswith("_adv") else s


def scan_images(image_dir: Path, recursive: bool = False) -> pd.DataFrame:
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    it = image_dir.rglob("*") if recursive else image_dir.iterdir()
    files = sorted([p.resolve() for p in it if p.is_file() and p.suffix.lower() in exts], key=lambda p: p.name.lower())
    if not files:
        raise RuntimeError(f"No images found in {image_dir}")
    seen: Dict[str, int] = {}
    rows = []
    for p in files:
        base = p.stem
        k = seen.get(base, 0); seen[base] = k + 1
        sid = base if k == 0 else f"{base}__dup{k}"
        pair = pair_from_stem(base)
        rows.append({
            "stim_id": sid,
            "filename": p.name,
            "path": str(p),
            "source": source_from_stem(base),
            "pair": pair,
        })
    return pd.DataFrame(rows)


def find_repo_root(start: Optional[Path] = None) -> Path:
    starts = []
    if start is not None:
        starts.append(start.resolve())
    starts += [Path.cwd().resolve(), Path(__file__).resolve().parent]
    checked = set()
    for root0 in starts:
        for p in [root0, *root0.parents]:
            if p in checked:
                continue
            checked.add(p)
            if (p / "attnclip_mechinterp_xattn").is_dir() and (p / "utils_clip_loader").is_dir():
                return p
    raise FileNotFoundError(
        "Could not locate repo root containing attnclip_mechinterp_xattn and utils_clip_loader. "
        "Run from the repository root or pass --repo_root."
    )


# -----------------------------------------------------------------------------
# Model loading / audit
# -----------------------------------------------------------------------------

def load_modemux(args):
    repo = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import attnclip_mechinterp_xattn as clip  # noqa
    from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

    model, preprocess, info = load_openai_clip_anything(
        clip,
        args.model,
        device=args.device,
        jit=False,
        cache_dir=(args.hf_cache_dir or None),
        revision=(args.model_revision or None),
        strict=True,
        allow_unsafe_hf_pickle=False,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if str(getattr(model, "implant_kind", "")) != "full":
        raise RuntimeError(f"Expected full ModeMUX/x-attention checkpoint; implant_kind={getattr(model, 'implant_kind', None)!r}")
    if getattr(model, "read_implant", None) is None:
        raise RuntimeError("Loaded model has no read_implant")
    if not hasattr(model, "encode_image_states"):
        raise RuntimeError("Loaded model lacks encode_image_states()")
    return repo, clip, model, preprocess, info


def model_audit(model, model_source: str, loader_info: Any) -> Dict[str, Any]:
    v = model.visual
    blocks = list(v.transformer.resblocks)
    implant = model.read_implant
    out = {
        "model_source": model_source,
        "dtype": str(model.dtype),
        "conv1_shape": list(v.conv1.weight.shape),
        "input_resolution": int(getattr(v, "input_resolution", -1)),
        "n_blocks": len(blocks),
        "n_heads": int(blocks[0].attn.num_heads),
        "vision_width": int(v.conv1.weight.shape[0]),
        "output_dim": int(getattr(v, "output_dim", -1)),
        "read_null_enabled": bool(getattr(model, "read_null_enabled", False)),
        "read_null_insert_block": int(getattr(model, "read_null_insert_block", -1)),
        "read_attention_architecture": str(getattr(model, "read_attention_architecture", "unknown")),
        "content_correction_default": bool(getattr(model, "_clip_apply_content_correction_by_default", False)),
        "loader_info": str(loader_info),
    }
    for name, fn in (("read_tap_blocks", "tap_block_list"), ("source_tap_blocks", "source_block_list"), ("ortho_tap_blocks", "ortho_block_list"), ("capture_blocks", "capture_block_list")):
        f = getattr(implant, fn, None)
        if callable(f):
            try:
                out[name] = [int(x) for x in f()]
            except Exception:
                pass
    return out


def amp_context(device: str, enabled: bool):
    if enabled and str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def load_batch(preprocess, rows: pd.DataFrame, device: str) -> Tuple[torch.Tensor, List[str], List[str]]:
    ims, ids, src = [], [], []
    for r in rows.itertuples(index=False):
        with Image.open(r.path) as im:
            ims.append(preprocess(ImageOps.exif_transpose(im).convert("RGB")))
        ids.append(str(r.stim_id)); src.append(str(r.source))
    return torch.stack(ims, 0).to(device, non_blocking=True), ids, src


# -----------------------------------------------------------------------------
# Conv1 morphology
# -----------------------------------------------------------------------------

def opponent_components(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, G, B = w[0], w[1], w[2]
    return (R + G + B) / math.sqrt(3.0), (R - G) / math.sqrt(2.0), (R + G - 2.0 * B) / math.sqrt(6.0)


def frequency_grid(h: int, w: int):
    fy = np.fft.fftshift(np.fft.fftfreq(h)); fx = np.fft.fftshift(np.fft.fftfreq(w))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    return yy, xx, rr / max(float(rr.max()), EPS)


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > EPS else 0.0


def dog_template(h: int, w: int, s1: float, s2: float) -> np.ndarray:
    y = np.linspace(-1, 1, h); x = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y, x, indexing="ij"); r2 = xx * xx + yy * yy
    g1 = np.exp(-r2 / (2*s1*s1)); g1 /= g1.sum() + EPS
    g2 = np.exp(-r2 / (2*s2*s2)); g2 /= g2.sum() + EPS
    t = g1 - g2; t -= t.mean(); t /= np.linalg.norm(t) + EPS
    return t


def gabor_template(h: int, w: int, angle: float, cycles: float, phase: float) -> np.ndarray:
    y = np.linspace(-1, 1, h); x = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    th = math.radians(angle); xr = xx * math.cos(th) + yy * math.sin(th)
    env = np.exp(-(xx*xx + yy*yy)/(2*0.55*0.55))
    t = env * np.cos(math.pi * cycles * xr + phase); t -= t.mean(); t /= np.linalg.norm(t) + EPS
    return t


def radial_profile(power: np.ndarray, nbins: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    h, w = power.shape; _, _, rr = frequency_grid(h, w)
    edges = np.linspace(0, 1 + 1e-9, nbins + 1); vals=[]; centers=[]
    for i in range(nbins):
        m = (rr >= edges[i]) & (rr < edges[i+1])
        vals.append(float(power[m].mean()) if m.any() else 0.0)
        centers.append(float((edges[i]+edges[i+1])/2))
    return np.asarray(centers), np.asarray(vals)


def filter_metrics(w: np.ndarray, idx: int) -> Dict[str, float]:
    w = np.asarray(w, np.float64); h, ww = w.shape[-2:]
    L, RG, BY = opponent_components(w)
    eL=float(np.sum(L*L)); eRG=float(np.sum(RG*RG)); eBY=float(np.sum(BY*BY)); et=eL+eRG+eBY+EPS
    Fw=np.fft.fftshift(np.fft.fft2(w,axes=(-2,-1)),axes=(-2,-1)); power=np.sum(np.abs(Fw)**2,axis=0); psum=float(power.sum())+EPS
    yy,xx,rr=frequency_grid(h,ww); dc=power[h//2,ww//2]; low=rr<=.25; mid=(rr>.25)&(rr<=.60); high=rr>.60
    pnorm=power/psum; centroid=float(np.sum(pnorm*rr)); bw=float(np.sqrt(np.sum(pnorm*(rr-centroid)**2)))
    pp=pnorm.ravel(); pp=pp[pp>0]; sent=float(-(pp*np.log(pp+EPS)).sum()/math.log(power.size))
    rc,rv=radial_profile(power,nbins=max(7,h//2)); peak_i=int(np.argmax(rv[1:])+1) if len(rv)>1 else 0; peak_r=float(rc[peak_i])
    ring_mask=np.abs(rr-peak_r)<=max(1.5/max(h,ww),.10); ring_frac=float(power[ring_mask].sum()/psum)
    mx=float(np.sum(pnorm*xx)); my=float(np.sum(pnorm*yy)); dx=xx-mx; dy=yy-my
    cov=np.array([[np.sum(pnorm*dx*dx),np.sum(pnorm*dx*dy)],[np.sum(pnorm*dx*dy),np.sum(pnorm*dy*dy)]],float)
    evals,evecs=np.linalg.eigh(cov); order=np.argsort(evals)[::-1]; l1,l2=float(evals[order[0]]),float(evals[order[1]]); v1=evecs[:,order[0]]
    anis=(l1-l2)/(l1+l2+EPS); angle=math.degrees(math.atan2(v1[1],v1[0]))%180.0; ar=math.radians(2*angle)
    axis_band=.09; vertical=np.abs(xx)<=axis_band*(np.max(np.abs(xx))+EPS); horizontal=np.abs(yy)<=axis_band*(np.max(np.abs(yy))+EPS)
    corner=(np.abs(xx)>=.55*np.max(np.abs(xx)))&(np.abs(yy)>=.55*np.max(np.abs(yy)))
    se=np.sum(w*w,axis=0); ssum=float(se.sum())+EPS; cy0,cy1=h//3,h-h//3; cx0,cx1=ww//3,ww-ww//3
    center=np.zeros((h,ww),bool); center[cy0:cy1,cx0:cx1]=True; edge=np.zeros((h,ww),bool); edge[[0,-1],:]=True; edge[:,[0,-1]]=True
    dog_best=0.0
    for s1,s2 in ((.18,.42),(.25,.55),(.32,.72)):
        dog_best=max(dog_best,abs(corr_flat(L,dog_template(h,ww,s1,s2))))
    gabor_best=0.0
    for a0 in (0,30,60,90,120,150):
        for cyc in (1.,2.,3.,4.):
            for ph in (0.,math.pi/2):
                t=gabor_template(h,ww,a0,cyc,ph); gabor_best=max(gabor_best,abs(corr_flat(L,t)),abs(corr_flat(RG,t)),abs(corr_flat(BY,t)))
    return {
        "channel":int(idx),"weight_l2":float(np.linalg.norm(w)),"weight_absmean":float(np.mean(np.abs(w))),
        "lum_energy_frac":eL/et,"rg_energy_frac":eRG/et,"by_energy_frac":eBY/et,
        "dc_frac":float(dc/psum),"low_frac":float(power[low].sum()/psum),"mid_frac":float(power[mid].sum()/psum),"high_frac":float(power[high].sum()/psum),
        "spectral_centroid":centroid,"spectral_bandwidth":bw,"spectral_entropy":sent,"radial_peak":peak_r,"ring_frac":ring_frac,
        "orientation_anisotropy":float(anis),"freq_axis_angle_deg":float(angle),"freq_axis_cos2":math.cos(ar),"freq_axis_sin2":math.sin(ar),
        "vertical_axis_frac":float(power[vertical].sum()/psum),"horizontal_axis_frac":float(power[horizontal].sum()/psum),"corner_freq_frac":float(power[corner].sum()/psum),
        "symmetry180":corr_flat(w,w[:,::-1,::-1]),"symmetry_lr":corr_flat(w,w[:,:,::-1]),"symmetry_ud":corr_flat(w,w[:,::-1,:]),
        "center_spatial_frac":float(se[center].sum()/ssum),"edge_spatial_frac":float(se[edge].sum()/ssum),
        "dog_abs_corr":float(dog_best),"gabor_abs_corr":float(gabor_best),
    }


def descriptive_cluster_label(g: pd.DataFrame, all_df: pd.DataFrame) -> str:
    m=g[list(MORPH_FEATURES)].mean(numeric_only=True); q=all_df[list(MORPH_FEATURES)].quantile([.25,.5,.75])
    if m.dog_abs_corr>=q.loc[.75,"dog_abs_corr"] and m.orientation_anisotropy<=q.loc[.5,"orientation_anisotropy"]: shape="center_surroundish"
    elif m.gabor_abs_corr>=q.loc[.75,"gabor_abs_corr"] and m.orientation_anisotropy>=q.loc[.5,"orientation_anisotropy"]: shape="oriented_gaborish"
    elif m.spectral_entropy>=q.loc[.75,"spectral_entropy"]: shape="broadband"
    elif m.ring_frac>=q.loc[.75,"ring_frac"]: shape="ring_bandpass"
    else: shape="mixed"
    if m.spectral_centroid<=q.loc[.25,"spectral_centroid"]: band="lowfreq"
    elif m.spectral_centroid>=q.loc[.75,"spectral_centroid"]: band="highfreq"
    else: band="midfreq"
    lum=float(m.lum_energy_frac); chrom=float(m.rg_energy_frac+m.by_energy_frac)
    col="luminance" if lum>.67 else ("chromatic" if chrom>.67 else "mixedcolor")
    orient="anisotropic" if m.orientation_anisotropy>=q.loc[.75,"orientation_anisotropy"] else "isotropicish"
    return f"{band}__{shape}__{col}__{orient}"


def cluster_filter_metrics(df: pd.DataFrame, kmin: int, kmax: int, seed: int) -> Tuple[pd.DataFrame,pd.DataFrame,dict]:
    out=df.copy(); Xraw=out[list(MORPH_FEATURES)].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
    Xs=StandardScaler().fit_transform(Xraw); ncomp=min(12,Xs.shape[1],max(1,Xs.shape[0]-1)); X=PCA(n_components=ncomp,random_state=seed).fit_transform(Xs)
    best=None; trials=[]
    for k in range(max(2,kmin),min(kmax,len(out)-1)+1):
        km=KMeans(n_clusters=k,n_init=30,random_state=seed); lab=km.fit_predict(X); sil=float(silhouette_score(X,lab)); trials.append({"k":k,"silhouette":sil,"inertia":float(km.inertia_)})
        if best is None or sil>best[0]: best=(sil,k,km,lab)
    if best is None:
        out["cluster_id"]=0; out["cluster_label"]="single_cluster"; out["cluster_distance"]=0.; out["cluster_outlier_z"]=0.
        return out,pd.DataFrame([{"cluster_id":0,"cluster_label":"single_cluster","n":len(out)}]),{"selected_k":1}
    _,kbest,km,labels=best; out["cluster_id"]=labels.astype(int); dist=np.linalg.norm(X-km.cluster_centers_[labels],axis=1); out["cluster_distance"]=dist; out["cluster_outlier_z"]=0.
    rows=[]; label_map={}
    for cid in sorted(out.cluster_id.unique()):
        idx=out.index[out.cluster_id==cid]; out.loc[idx,"cluster_outlier_z"]=robust_z(out.loc[idx,"cluster_distance"].to_numpy(float)); g=out.loc[idx]
        label=descriptive_cluster_label(g,out); label_map[int(cid)]=label
        rows.append({"cluster_id":int(cid),"cluster_label":label,"n":len(g),"representative_channels":",".join(map(str,g.sort_values("cluster_distance").head(8).channel.astype(int))),"top_outlier_channels":",".join(map(str,g.sort_values("cluster_outlier_z",ascending=False).head(8).channel.astype(int)))})
    out["cluster_label"]=out.cluster_id.map(label_map)
    return out,pd.DataFrame(rows),{"selected_k":int(kbest),"k_trials":trials,"features":list(MORPH_FEATURES),"note":"weight_l2 excluded"}


def kernel_rgb(w: np.ndarray) -> np.ndarray:
    x=np.transpose(w,(1,2,0)).astype(float); s=np.quantile(np.abs(x),.995)+EPS; return np.clip(.5+.48*x/s,0,1)


def fft_power(w: np.ndarray) -> np.ndarray:
    Fw=np.fft.fftshift(np.fft.fft2(w,axes=(-2,-1)),axes=(-2,-1)); return np.log1p(np.sum(np.abs(Fw)**2,axis=0))


# -----------------------------------------------------------------------------
# Natural Conv1 response + position statistics
# -----------------------------------------------------------------------------

@torch.inference_mode()
def empirical_activation_and_position(model, preprocess, manifest: pd.DataFrame, batch_size: int, device: str, amp: bool) -> Tuple[pd.DataFrame,np.ndarray,np.ndarray,np.ndarray]:
    v=model.visual; C=int(v.conv1.weight.shape[0]); sum_v=np.zeros(C,np.float64); sum_sq=np.zeros(C,np.float64); sum_abs=np.zeros(C,np.float64); pos_count=np.zeros(C,np.float64); n_scalar=0; mean_map_sum=None
    for st in range(0,len(manifest),batch_size):
        meta=manifest.iloc[st:st+batch_size]; batch,_ids,_src=load_batch(preprocess,meta,device)
        with amp_context(device,amp): y=v.conv1(batch.to(dtype=model.dtype)).float().detach().cpu().numpy()
        B,C2,H,W=y.shape; flat=y.transpose(1,0,2,3).reshape(C,-1); sum_v+=flat.sum(1); sum_sq+=(flat*flat).sum(1); sum_abs+=np.abs(flat).sum(1); pos_count+=(flat>0).sum(1); n_scalar+=flat.shape[1]
        mm=y.sum(0); mean_map_sum=mm if mean_map_sum is None else mean_map_sum+mm
        del batch,y; gc.collect()
    mean=sum_v/max(n_scalar,1); msq=sum_sq/max(n_scalar,1); std=np.sqrt(np.maximum(msq-mean*mean,0)); rms=np.sqrt(np.maximum(msq,0)); mean_abs=sum_abs/max(n_scalar,1); positive=pos_count/max(n_scalar,1); mean_map=mean_map_sum/max(len(manifest),1)
    pos=v.positional_embedding.detach().float().cpu().numpy(); patch_pos=pos[1:]; side=int(round(math.sqrt(patch_pos.shape[0])))
    if side*side!=patch_pos.shape[0]: raise RuntimeError(f"Non-square positional grid P={patch_pos.shape[0]}")
    pos_map=patch_pos.reshape(side,side,C).transpose(2,0,1).astype(np.float32)
    if mean_map.shape[-2:]!=pos_map.shape[-2:]: raise RuntimeError(f"Conv1 grid {mean_map.shape[-2:]} != position grid {pos_map.shape[-2:]}")
    rows=[]
    for c in range(C):
        pm=pos_map[c].astype(float); am=mean_map[c].astype(float); pos_std=float(pm.std()); pos_rms=float(np.sqrt(np.mean(pm*pm)))
        rows.append({"channel":c,"activation_mean":float(mean[c]),"activation_std":float(std[c]),"activation_rms":float(rms[c]),"activation_mean_abs":float(mean_abs[c]),"activation_positive_frac":float(positive[c]),"pos_mean":float(pm.mean()),"pos_std":pos_std,"pos_rms":pos_rms,"pos_range":float(pm.max()-pm.min()),"pos_to_pixel_std_ratio":float(pos_std/max(float(std[c]),EPS)),"pos_to_pixel_rms_ratio":float(pos_rms/max(float(rms[c]),EPS)),"mean_activation_pos_corr":corr_flat(am,pm)})
    init=mean_map.astype(np.float32)+pos_map.astype(np.float32)
    return pd.DataFrame(rows),mean_map.astype(np.float32),pos_map,init


def build_twins(master: pd.DataFrame, control_z_min: float) -> Tuple[pd.DataFrame,pd.DataFrame]:
    targets=master[master.population.isin(LOW_POPS)].copy().reset_index(drop=True); controls=master[master.robust_log_z>=float(control_z_min)].copy().reset_index(drop=True)
    if targets.empty or controls.empty: return pd.DataFrame(),pd.DataFrame()
    # Equal aggregate domain influence.
    Xt=[]; Xc=[]
    for _domain,cols in TWIN_FEATURE_DOMAINS.items():
        allv=pd.concat([targets[list(cols)],controls[list(cols)]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float); mu=allv.mean(0); sd=allv.std(0); sd[sd<1e-8]=1
        Xt.append((targets[list(cols)].to_numpy(float)-mu)/sd/math.sqrt(len(cols))); Xc.append((controls[list(cols)].to_numpy(float)-mu)/sd/math.sqrt(len(cols)))
    A=np.concatenate(Xt,1); B=np.concatenate(Xc,1); D=((A[:,None,:]-B[None,:,:])**2).sum(-1)
    if HAVE_SCIPY and len(controls)>=len(targets): rr,cc=linear_sum_assignment(D)
    else:
        rr=[];cc=[];used=set()
        for i in range(len(targets)):
            for j in np.argsort(D[i]):
                if int(j) not in used: rr.append(i);cc.append(int(j));used.add(int(j));break
    pairs=[]; tops=[]
    for ri,cj in zip(rr,cc):
        tr=targets.iloc[int(ri)]; cr=controls.iloc[int(cj)]
        pairs.append({"target_channel":int(tr.channel),"target_population":tr.population,"target_weight_l2":float(tr.weight_l2),"target_robust_z":float(tr.robust_log_z),"control_channel":int(cr.channel),"control_weight_l2":float(cr.weight_l2),"control_robust_z":float(cr.robust_log_z),"weight_ratio_control_over_target":float(cr.weight_l2/max(float(tr.weight_l2),EPS)),"morphology_distance_sq":float(D[int(ri),int(cj)]),"target_guess":str(tr.cluster_label),"control_guess":str(cr.cluster_label)})
        for rank,j in enumerate(np.argsort(D[int(ri)])[:5],1):
            alt=controls.iloc[int(j)]; tops.append({"target_channel":int(tr.channel),"rank":rank,"candidate_control_channel":int(alt.channel),"morphology_distance_sq":float(D[int(ri),int(j)]),"candidate_weight_l2":float(alt.weight_l2)})
    return pd.DataFrame(pairs),pd.DataFrame(tops)


# -----------------------------------------------------------------------------
# Conv1 interventions
# -----------------------------------------------------------------------------

def summarize_response_map(r: torch.Tensor) -> Dict[str, torch.Tensor]:
    f=r.flatten(1).float(); return {"mean":f.mean(1),"mean_abs":f.abs().mean(1),"std":f.std(1,unbiased=False),"positive_frac":(f>0).float().mean(1),"rms":f.square().mean(1).sqrt()}


class MultiConv1Transform:
    """Full-strength arbitrary Conv1 output interventions; one spec per channel."""
    def __init__(self, model, specs: Sequence[Tuple[int,str]], stim_ids: Sequence[str]):
        self.model=model; self.specs=[(int(c),str(cond).upper()) for c,cond in specs]; self.stim_ids=list(map(str,stim_ids)); self.handle=None; self.audit_cache={}
    def __enter__(self):
        specs=self.specs; ids=self.stim_ids
        def hook(_m,_inp,out):
            y=out.clone()
            for c,cond in specs:
                if c>=y.shape[1]: raise RuntimeError(f"Conv1 has {y.shape[1]} channels; requested {c}")
                r=y[:,c].clone(); before=summarize_response_map(r)
                if cond=="ZERO": z=torch.zeros_like(r)
                elif cond=="FLIP": z=-r
                elif cond=="ABS": z=r.abs()
                elif cond=="SIGN":
                    amp=r.abs().flatten(1).mean(1)[:,None,None]; z=torch.sign(r)*amp
                elif cond=="DC_ONLY":
                    mu=r.flatten(1).mean(1)[:,None,None]; z=mu.expand_as(r)
                elif cond=="CENTERED":
                    mu=r.flatten(1).mean(1)[:,None,None]; z=r-mu
                elif cond=="SHUFFLE":
                    B,H,W=r.shape; rf=r.flatten(1); zz=[]
                    for bi in range(B):
                        g=torch.Generator(device="cpu"); g.manual_seed(stable_seed(f"ch{c}:{cond}:{ids[bi]}")); perm=torch.randperm(H*W,generator=g).to(r.device); zz.append(rf[bi].index_select(0,perm).reshape(H,W))
                    z=torch.stack(zz,0)
                else: raise ValueError(cond)
                y[:,c]=z; after=summarize_response_map(z); self.audit_cache[(c,cond)]=(before,after)
            return y
        self.handle=self.model.visual.conv1.register_forward_hook(hook); return self
    def audits(self, ids: Sequence[str], sources: Sequence[str]) -> Dict[Tuple[int,str,str],dict]:
        out={}
        for (c,cond),(before,after) in self.audit_cache.items():
            b={k:v.detach().float().cpu().numpy() for k,v in before.items()}; a={k:v.detach().float().cpu().numpy() for k,v in after.items()}
            for i,sid in enumerate(ids):
                rec={"channel":c,"condition":cond,"stim_id":str(sid),"source":str(sources[i])}
                for k in b: rec[f"before_{k}"]=float(b[k][i]);rec[f"after_{k}"]=float(a[k][i])
                out[(c,cond,str(sid))]=rec
        return out
    def __exit__(self,*exc):
        if self.handle is not None: self.handle.remove(); self.handle=None


class PreBlockCapture:
    def __init__(self, model, blocks: Sequence[int]): self.model=model;self.blocks=sorted(set(map(int,blocks)));self.handles=[];self.pre={}
    def __enter__(self):
        allb=self.model.visual.transformer.resblocks
        for b in self.blocks:
            if 0<=b<len(allb):
                def hook(_m,inp,b=b): self.pre[b]=inp[0].detach()
                self.handles.append(allb[b].register_forward_pre_hook(hook))
        return self
    def __exit__(self,*exc):
        for h in self.handles: h.remove()
        self.handles=[]


def canonical_btc(x: torch.Tensor, B: int) -> torch.Tensor:
    if x.ndim!=3: raise RuntimeError(f"Expected 3D residual, got {tuple(x.shape)}")
    if x.shape[1]==B: return x.permute(1,0,2).contiguous()
    if x.shape[0]==B: return x.contiguous()
    raise RuntimeError(f"Cannot infer residual layout {tuple(x.shape)} B={B}")


def q_or_k_for_pre(block, pre: torch.Tensor, B: int, which: str) -> torch.Tensor:
    # Returns [B,H,T,Dh]. Residual block expects T,B,C.
    if pre.shape[1]!=B and pre.shape[0]==B: pre=pre.permute(1,0,2).contiguous()
    z=block.ln_1(pre)
    proj=block.attn.q_proj if which=="q" else block.attn.k_proj
    y=proj(z); T,BB,C=y.shape; H=int(block.attn.num_heads); dh=C//H
    return y.view(T,BB,H,dh).permute(1,2,0,3).contiguous()


@dataclass
class SampleProbe:
    backbone: np.ndarray
    content: np.ndarray
    correction: np.ndarray
    early_maps: Dict[int,np.ndarray]
    mid_maps: Dict[int,np.ndarray]
    b12_k_ratio_mean: float
    b12_k_ratio_max: float
    b12_reg_attn_mass: float
    pre13_regs: set
    pre13_max_norm: float
    pre13_frozen_reg_mean_norm: float
    b22_q: np.ndarray
    final_reg_set: set
    source_present_logit: float
    source_readable_logit: float
    source_gate: float
    glyph_mean: float
    glyph_max: float
    null_read_logit: float
    null_read_attention: float
    head_signature: Optional[np.ndarray]  # [blocks, heads, 3]


def _spatial_patch_count(model) -> int:
    return int(model.visual.positional_embedding.shape[0]-1)


def _attention_spatial(p: torch.Tensor, P: int) -> torch.Tensor:
    return p[...,1:1+P]


def _set_from_mask(mask: torch.Tensor) -> set:
    return set(torch.nonzero(mask,as_tuple=False).flatten().cpu().tolist())


@torch.inference_mode()
def probe_batch(model, images: torch.Tensor, ids: Sequence[str], args, *, frozen_regs: Optional[Sequence[set]]=None) -> List[SampleProbe]:
    blocks=model.visual.transformer.resblocks; nblocks=len(blocks); B=images.shape[0]; P=_spatial_patch_count(model)
    early=[b for b in args.early_blocks if 0<=b<nblocks]; routing=[b for b in args.routing_blocks if 0<=b<nblocks]
    special=[args.register_k_block,args.register_address_block,args.late_q_block]
    with PreBlockCapture(model,special) as cap:
        with amp_context(args.device,args.amp):
            info=model.encode_image_states(images,return_final_tokens=False)
            backbone_raw=info["image_embedding"]
            content_raw=model._content_image_from_info(info,apply_content_correction=True)
            backbone=F.normalize(backbone_raw.float(),dim=-1); content=F.normalize(content_raw.float(),dim=-1); correction=content_raw.float()-backbone_raw.float()
            source_logits,_glyph_logits,source_stats=model.read_implant.source_outputs(info["states"],return_details=False)
            source_probs=source_logits.sigmoid(); source_gate=source_probs[:,0]*source_probs[:,1]
            null_logits=torch.full((B,),float("nan"),device=images.device); null_att=None
            if args.reader_telemetry:
                nt=model._null_read_tokens(images.device); ni=model._encode_text_hidden(nt); ntxt=F.normalize(ni["text_embedding"].float(),dim=-1)
                if bool(getattr(model,"read_null_enabled",False)):
                    nf,details=model.read_implant.read_features(info["states"],ni["eot_hidden_pre_ln"],register_mask=info["register_mask"],return_details=True)
                    try: null_att=details["read_null_attention"][:,0].float()
                    except Exception: null_att=None
                else:
                    nf=model.read_implant.read_features(info["states"],ni["eot_hidden_pre_ln"],register_mask=info["register_mask"])
                nf=F.normalize(nf[:,0,:].float(),dim=-1); scale=model.logit_scale.float().exp(); raw=scale*torch.sum(nf*ntxt[0],dim=-1)
                null_logits=model.read_implant.calibrate_read_logits(raw[:,None],source_logits,null_mask=torch.ones(1,dtype=torch.bool,device=images.device))[:,0]

    # Current pre-B13 register sets.
    pre13=canonical_btc(cap.pre[args.register_address_block],B).float(); spatial13=pre13[:,1:1+P]; norms13=spatial13.norm(dim=-1)
    cur_regs=[]
    for i in range(B): cur_regs.append(set(torch.nonzero(norms13[i]>=args.register_threshold,as_tuple=False).flatten().cpu().tolist()))
    regs_for_metrics=list(frozen_regs) if frozen_regs is not None else cur_regs

    # Attention maps are already captured by the mechinterp attention implementation.
    early_maps={}; mid_maps={}
    for b in early:
        p=blocks[b].attn_probs.float(); early_maps[b]=p[:,:,0,1:1+P].numpy()
    for b in routing:
        p=blocks[b].attn_probs.float(); mid_maps[b]=p.mean(dim=2).mean(dim=1)[:,1:1+P].numpy()

    # B12 frozen-register K ratio + incoming mass.
    kblk=blocks[args.register_k_block]; k=q_or_k_for_pre(kblk,cap.pre[args.register_k_block],B,"k").float().cpu(); p12=kblk.attn_probs.float(); b12mean=[];b12max=[];b12mass=[]
    for i in range(B):
        regtok=[int(x)+1 for x in sorted(regs_for_metrics[i]) if 0<=int(x)<P]; other=[t for t in range(1,1+P) if t not in set(regtok)]
        if regtok:
            rk=k[i,:,regtok].norm(dim=-1).mean(dim=-1); ok=k[i,:,other].norm(dim=-1).mean(dim=-1) if other else torch.ones_like(rk); ratio=rk/ok.clamp_min(EPS)
            b12mean.append(float(ratio.mean()));b12max.append(float(ratio.max()));b12mass.append(float(p12[i,:,:,regtok].sum(dim=-1).mean()))
        else: b12mean.append(np.nan);b12max.append(np.nan);b12mass.append(np.nan)

    # Late CLS Q.
    q22=q_or_k_for_pre(blocks[args.late_q_block],cap.pre[args.late_q_block],B,"q").float().cpu(); q22=q22[:,:,0,:].reshape(B,-1).numpy()

    # Register-coupling head signature: [block,head,(CLS->REG, REG->CLS, ALL->REG)].
    head_sig=None
    if args.head_signature:
        H=int(blocks[0].attn.num_heads); head_sig=np.zeros((B,nblocks,H,3),np.float32)
        for b,blk in enumerate(blocks):
            p=blk.attn_probs.float(); T=p.shape[-1]
            for i in range(B):
                regtok=[int(x)+1 for x in sorted(regs_for_metrics[i]) if 0<=int(x)<P and int(x)+1<T]
                if not regtok: continue
                head_sig[i,b,:,0]=p[i,:,0,regtok].sum(-1).numpy()
                head_sig[i,b,:,1]=p[i,:,regtok,0].mean(-1).numpy()
                head_sig[i,b,:,2]=p[i,:,:,regtok].sum(-1).mean(-1).numpy()

    out=[]; frm=info["register_mask"].detach().cpu()
    for i in range(B):
        regs0=regs_for_metrics[i]; regidx=sorted(int(x) for x in regs0 if 0<=int(x)<P); regmean=float(norms13[i,regidx].mean()) if regidx else float("nan")
        out.append(SampleProbe(
            backbone=backbone[i].detach().cpu().numpy(),content=content[i].detach().cpu().numpy(),correction=correction[i].detach().cpu().numpy(),
            early_maps={b:early_maps[b][i].copy() for b in early},mid_maps={b:mid_maps[b][i].copy() for b in routing},
            b12_k_ratio_mean=b12mean[i],b12_k_ratio_max=b12max[i],b12_reg_attn_mass=b12mass[i],pre13_regs=set(cur_regs[i]),pre13_max_norm=float(norms13[i].max().cpu()),pre13_frozen_reg_mean_norm=regmean,b22_q=q22[i].copy(),final_reg_set=_set_from_mask(frm[i]),
            source_present_logit=float(source_logits[i,0].detach().cpu()),source_readable_logit=float(source_logits[i,1].detach().cpu()),source_gate=float(source_gate[i].detach().cpu()),glyph_mean=float(source_stats[i,0].detach().cpu()),glyph_max=float(source_stats[i,1].detach().cpu()),
            null_read_logit=float(null_logits[i].detach().cpu()) if torch.isfinite(null_logits[i]) else float("nan"),null_read_attention=float(null_att[i].detach().cpu()) if null_att is not None else float("nan"),head_signature=(head_sig[i].copy() if head_sig is not None else None)
        ))
    # Prevent one full probability tensor per block from hanging around between cells.
    for blk in blocks: blk.attn_probs=None
    return out


def compare_probe(base: SampleProbe, cur: SampleProbe, args) -> Dict[str,float]:
    bd=cosine_distance_np(base.backbone,cur.backbone); cd=cosine_distance_np(base.content,cur.content)
    early=[]
    for b in base.early_maps:
        if b in cur.early_maps:
            x=torch.from_numpy(base.early_maps[b]);y=torch.from_numpy(cur.early_maps[b]); early.append(float(cosine_dist_rows_torch(x,y).mean()))
    mid=[]
    for b in base.mid_maps:
        if b in cur.mid_maps: mid.append(cosine_distance_np(base.mid_maps[b],cur.mid_maps[b]))
    a=set(base.pre13_regs);c=set(cur.pre13_regs);u=a|c;inter=a&c; fa=set(base.final_reg_set);fc=set(cur.final_reg_set);fu=fa|fc;fi=fa&fc
    corr_delta=float(np.linalg.norm(cur.correction-base.correction)); corr_cos=cosine_np(base.correction,cur.correction)
    return {
        "backbone_cosine_distance":bd,"content_cosine_distance":cd,"max_embedding_distance":max(bd,cd),"content_repair":bd-cd,
        "content_correction_delta_norm":corr_delta,"content_correction_cosine":corr_cos,
        "early_cls_scanner_cosdist":float(np.nanmean(early)) if early else np.nan,"mid_incoming_cosdist":float(np.nanmean(mid)) if mid else np.nan,
        "b12_reg_k_ratio_delta":float(cur.b12_k_ratio_mean-base.b12_k_ratio_mean),"b12_reg_attn_mass_delta":float(cur.b12_reg_attn_mass-base.b12_reg_attn_mass),
        "b13_reg_jaccard":len(inter)/len(u) if u else 1.0,"b13_reg_recall":len(inter)/len(a) if a else 1.0,"b13_n_regs_baseline":len(a),"b13_n_regs_condition":len(c),"b13_max_spatial_norm_ratio":float(cur.pre13_max_norm/max(base.pre13_max_norm,EPS)),"b13_frozen_reg_mean_norm_ratio":float(cur.pre13_frozen_reg_mean_norm/max(base.pre13_frozen_reg_mean_norm,EPS)) if np.isfinite(cur.pre13_frozen_reg_mean_norm) and np.isfinite(base.pre13_frozen_reg_mean_norm) else np.nan,
        "b22_cls_q_cosine_distance":cosine_distance_np(base.b22_q,cur.b22_q),
        "final_reg_jaccard":len(fi)/len(fu) if fu else 1.0,"final_reg_count_baseline":len(fa),"final_reg_count_condition":len(fc),"final_reg_count_delta":len(fc)-len(fa),
        "source_present_logit_delta":cur.source_present_logit-base.source_present_logit,"source_readable_logit_delta":cur.source_readable_logit-base.source_readable_logit,"source_gate_delta":cur.source_gate-base.source_gate,"abs_source_gate_delta":abs(cur.source_gate-base.source_gate),"glyph_mean_delta":cur.glyph_mean-base.glyph_mean,"glyph_max_delta":cur.glyph_max-base.glyph_max,"null_read_logit_delta":cur.null_read_logit-base.null_read_logit if np.isfinite(cur.null_read_logit) and np.isfinite(base.null_read_logit) else np.nan,"abs_null_read_logit_delta":abs(cur.null_read_logit-base.null_read_logit) if np.isfinite(cur.null_read_logit) and np.isfinite(base.null_read_logit) else np.nan,"null_read_attention_delta":cur.null_read_attention-base.null_read_attention if np.isfinite(cur.null_read_attention) and np.isfinite(base.null_read_attention) else np.nan,"abs_final_reg_count_delta":abs(len(fc)-len(fa)),"abs_content_repair":abs(bd-cd),
    }


# -----------------------------------------------------------------------------
# Baseline cache and causal scan
# -----------------------------------------------------------------------------

@torch.inference_mode()
def build_baseline_cache(model,preprocess,manifest:pd.DataFrame,args)->Dict[str,SampleProbe]:
    cache={}
    for st in range(0,len(manifest),args.batch_size):
        meta=manifest.iloc[st:st+args.batch_size];batch,ids,_src=load_batch(preprocess,meta,args.device); probes=probe_batch(model,batch,ids,args,frozen_regs=None)
        for sid,p in zip(ids,probes): cache[str(sid)]=p
        print(f"[baseline] {min(st+len(meta),len(manifest))}/{len(manifest)}")
        del batch,probes;gc.collect()
    return cache


def _head_accumulate(acc: Optional[Dict[str,np.ndarray]], base: SampleProbe, cur: SampleProbe) -> None:
    if acc is None or base.head_signature is None or cur.head_signature is None: return
    d=cur.head_signature-base.head_signature
    if "sum" not in acc:
        acc["sum"]=np.zeros_like(d,np.float64);acc["sumabs"]=np.zeros_like(d,np.float64);acc["n"]=np.zeros(d.shape[:2],np.float64)
    acc["sum"]+=d;acc["sumabs"]+=np.abs(d);acc["n"]+=1


def head_acc_to_rows(channel:int,condition:str,acc:Dict[str,np.ndarray])->List[dict]:
    if not acc or "sum" not in acc: return []
    s=acc["sum"];a=acc["sumabs"];n=acc["n"];rows=[]
    for b in range(s.shape[0]):
        for h in range(s.shape[1]):
            den=max(float(n[b,h]),1.0)
            rows.append({"channel":int(channel),"condition":str(condition),"block":b,"head":h,"cls_to_reg_delta":float(s[b,h,0]/den),"reg_to_cls_delta":float(s[b,h,1]/den),"all_to_reg_delta":float(s[b,h,2]/den),"cls_to_reg_abs_delta":float(a[b,h,0]/den),"reg_to_cls_abs_delta":float(a[b,h,1]/den),"all_to_reg_abs_delta":float(a[b,h,2]/den)})
    return rows


@torch.inference_mode()
def run_scan(model,preprocess,manifest:pd.DataFrame,channels:Sequence[int],conditions:Sequence[str],args,out_path:Path, *, stage_name:str,save_heads:bool)->Tuple[pd.DataFrame,pd.DataFrame]:
    out_path.parent.mkdir(parents=True,exist_ok=True); existing=pd.read_csv(out_path) if args.resume and out_path.is_file() else pd.DataFrame(); rows=existing.to_dict("records") if len(existing) else []
    done=set()
    if len(existing):
        counts=existing.groupby(["channel","condition"]).stim_id.nunique()
        for (c,cond),n in counts.items():
            if int(n)>=len(manifest):done.add((int(c),str(cond)))
    base=build_baseline_cache(model,preprocess,manifest,args)
    if stage_name == "causal":
        srcmap={str(r.stim_id):str(r.source) for r in manifest.itertuples(index=False)}
        brows=[]
        for sid,b in base.items():
            brows.append({
                "stim_id":sid,"source":srcmap.get(sid,""),
                "pre_b13_register_count":len(b.pre13_regs),
                "final_implicit_register_count":len(b.final_reg_set),
                "content_correction_norm":float(np.linalg.norm(b.correction)),
                "backbone_content_cosine":cosine_np(b.backbone,b.content),
                "source_present_logit":b.source_present_logit,
                "source_readable_logit":b.source_readable_logit,
                "source_gate":b.source_gate,
                "glyph_mean":b.glyph_mean,"glyph_max":b.glyph_max,
                "null_read_logit":b.null_read_logit,
                "null_read_attention":b.null_read_attention,
            })
        write_df(pd.DataFrame(brows),out_path.parent/"baseline_xattn_metrics.csv")
    head_path=out_path.parent/"head_register_coupling_signature.csv.gz"
    if save_heads and args.head_signature and args.resume and head_path.is_file():
        head_rows=pd.read_csv(head_path).to_dict("records")
    else:
        head_rows=[]
    total=len(channels)*len(conditions);ci=0
    for c in map(int,channels):
        for cond in map(str,conditions):
            ci+=1
            if (c,cond) in done:
                print(f"[{stage_name} {ci}/{total}] ch{c:04d} {cond} [resume]");continue
            cell=[];hacc={} if (save_heads and args.head_signature) else None
            for st in range(0,len(manifest),args.batch_size):
                meta=manifest.iloc[st:st+args.batch_size];batch,ids,sources=load_batch(preprocess,meta,args.device); frozen=[set(base[s].pre13_regs) for s in ids]
                with MultiConv1Transform(model,[(c,cond)],ids) as iv:
                    cur=probe_batch(model,batch,ids,args,frozen_regs=frozen); audits=iv.audits(ids,sources)
                for sid,src,p in zip(ids,sources,cur):
                    rec={"channel":c,"condition":cond,"stim_id":sid,"source":src};rec.update(compare_probe(base[sid],p,args));rec.update(audits.get((c,cond,sid),{}));cell.append(rec);_head_accumulate(hacc,base[sid],p)
                del batch,cur;gc.collect()
            rows=[r for r in rows if not (int(r.get("channel",-1))==c and str(r.get("condition",""))==cond)];rows.extend(cell);write_df(pd.DataFrame(rows),out_path)
            if hacc is not None:
                head_rows=[r for r in head_rows if not (int(r.get("channel",-1))==c and str(r.get("condition",""))==cond)]
                head_rows.extend(head_acc_to_rows(c,cond,hacc));write_df(pd.DataFrame(head_rows),head_path)
            print(f"[{stage_name} {ci}/{total}] ch{c:04d} {cond} n={len(cell)}")
    return pd.DataFrame(rows),pd.DataFrame(head_rows)


def summarize_causal(per:pd.DataFrame,out:Path,name:str="causal_summary.csv")->pd.DataFrame:
    metrics=[c for c in per.columns if c not in {"channel","condition","stim_id","source"} and pd.api.types.is_numeric_dtype(per[c])]
    rows=[]
    for keys,g in list(per.groupby(["channel","condition","source"],dropna=False))+list(per.groupby(["channel","condition"],dropna=False)):
        if len(keys)==3: c,cond,src=keys
        else: c,cond=keys;src="all"
        rec={"channel":int(c),"condition":str(cond),"source":str(src),"n":len(g)}
        for m in metrics:
            v=pd.to_numeric(g[m],errors="coerce")
            rec[f"mean_{m}"]=float(v.mean());rec[f"max_{m}"]=float(v.max());rec[f"min_{m}"]=float(v.min());rec[f"p90_{m}"]=float(v.quantile(.90))
        rows.append(rec)
    df=pd.DataFrame(rows);write_df(df,out/name);return df


# -----------------------------------------------------------------------------
# Optional pretrained Conv1 drift
# -----------------------------------------------------------------------------

@torch.inference_mode()
def pretrained_drift(clip,current_model,master:pd.DataFrame,args,out:Path)->pd.DataFrame:
    print(f"[pretrained comparison] resolving {args.pretrained_model} on CPU for Conv1-only audit")
    from utils_clip_loader.clip_anything_to_openai import resolve_to_openai_state_dict
    sd,_=resolve_to_openai_state_dict(args.pretrained_model,cache_dir=(args.hf_cache_dir or None),allow_unsafe_hf_pickle=False)
    if "visual.conv1.weight" not in sd: raise KeyError(f"{args.pretrained_model} has no visual.conv1.weight")
    W0=sd["visual.conv1.weight"].detach().float().cpu().numpy();W1=current_model.visual.conv1.weight.detach().float().cpu().numpy();del sd;gc.collect()
    if W0.shape!=W1.shape: raise RuntimeError(f"Conv1 shape mismatch current={W1.shape} pretrained={W0.shape}")
    rows=[]
    for c in range(W1.shape[0]):
        a=W0[c].ravel();b=W1[c].ravel();na=np.linalg.norm(a);nb=np.linalg.norm(b)
        rows.append({"channel":c,"pretrained_weight_l2":float(na),"current_weight_l2":float(nb),"weight_norm_ratio_current_over_pretrained":float(nb/max(na,EPS)),"same_channel_kernel_cosine":float(np.dot(a,b)/max(na*nb,EPS)),"kernel_delta_l2":float(np.linalg.norm(b-a))})
    d=pd.DataFrame(rows);d["kernel_drift_score"]=1-d.same_channel_kernel_cosine;write_df(d,out/"conv1_pretrained_drift.csv")
    fig,ax=plt.subplots(figsize=(10,8));ax.scatter(d.pretrained_weight_l2,d.current_weight_l2,s=12,alpha=.55);lo=min(d.pretrained_weight_l2.min(),d.current_weight_l2.min());hi=max(d.pretrained_weight_l2.max(),d.current_weight_l2.max());ax.plot([lo,hi],[lo,hi],ls="--",lw=1,color="0.4");ax.set_xlabel("pretrained Conv1 weight L2");ax.set_ylabel("ModeMUX Conv1 weight L2");ax.set_title("Same-channel Conv1 norm drift");ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/"plots"/"00_CONV1_PRETRAINED_NORM_DRIFT.png",dpi=200);plt.close(fig)
    return d


# -----------------------------------------------------------------------------
# Candidate selection and phase metrics
# -----------------------------------------------------------------------------

def select_candidates(master:pd.DataFrame,screen_summary:pd.DataFrame,args,out:Path)->pd.DataFrame:
    s=screen_summary[screen_summary.source.eq("all")].copy(); agg=s.groupby("channel").agg(screen_max_effect=("max_max_embedding_distance","max"),screen_max_backbone=("max_backbone_cosine_distance","max"),screen_max_content=("max_content_cosine_distance","max"),screen_max_correction=("max_content_correction_delta_norm","max"),screen_max_routing=("max_mid_incoming_cosdist","max"),screen_max_b22=("max_b22_cls_q_cosine_distance","max"),screen_max_reg_disrupt=("min_b13_reg_jaccard","min") if "min_b13_reg_jaccard" in s.columns else ("min_final_reg_jaccard","min")).reset_index()
    z=master.merge(agg,on="channel",how="left");z["screen_max_reg_disrupt"]=1-pd.to_numeric(z.screen_max_reg_disrupt,errors="coerce")
    for col in ["screen_max_effect","screen_max_correction","screen_max_routing","screen_max_b22","screen_max_reg_disrupt","pos_to_pixel_std_ratio","cluster_outlier_z"]:
        z[f"pct_{col}"]=percentile01(z[col].fillna(z[col].median() if z[col].notna().any() else 0))
    include=set(z[z.population.isin(LOW_POPS)].channel.astype(int))
    rankcols=["screen_max_effect","screen_max_correction","screen_max_routing","screen_max_b22","screen_max_reg_disrupt","pos_to_pixel_std_ratio","cluster_outlier_z"]
    for col in rankcols: include.update(z.nlargest(min(args.candidate_top_per_axis,len(z)),col).channel.astype(int).tolist())
    if "kernel_drift_score" in z.columns: include.update(z.nlargest(min(args.candidate_top_per_axis,len(z)),"kernel_drift_score").channel.astype(int).tolist())
    q=z[z.channel.isin(include)].copy(); q["candidate_reason"]=""
    def reasons(r):
        rr=[]
        if r.population in LOW_POPS:rr.append(r.population)
        for col,label in [("pct_screen_max_effect","global_effect"),("pct_screen_max_correction","correction"),("pct_screen_max_routing","routing"),("pct_screen_max_b22","late_q"),("pct_screen_max_reg_disrupt","register"),("pct_pos_to_pixel_std_ratio","positional"),("pct_cluster_outlier_z","morph_outlier")]:
            if float(getattr(r,col,0) or 0)>=.97:rr.append(label)
        if hasattr(r,"kernel_drift_score") and pd.notna(r.kernel_drift_score) and float(r.kernel_drift_score)>=z.kernel_drift_score.quantile(.97):rr.append("pretrained_drift")
        return ";".join(rr) or "screen_union"
    q["candidate_reason"]=[reasons(r) for r in q.itertuples(index=False)]
    # Never delete low-tail channels. If union is huge, cap only non-low-tail additions.
    low=q[q.population.isin(LOW_POPS)];other=q[~q.population.isin(LOW_POPS)].copy();other["pre_score"]=np.sqrt(np.mean(np.stack([other.get(f"pct_{c}",pd.Series(0,index=other.index)).to_numpy(float)**2 for c in rankcols],1),1));other=other.sort_values("pre_score",ascending=False)
    room=max(0,int(args.max_candidates)-len(low)) if args.max_candidates>0 else len(other);q=pd.concat([low,other.head(room)],ignore_index=True).drop_duplicates("channel").sort_values("channel")
    write_df(q,out/"candidate_channels.csv");return q


def build_phase_metrics(master:pd.DataFrame,summary:pd.DataFrame,candidates:pd.DataFrame,out:Path)->pd.DataFrame:
    q=summary[summary.source.eq("all") & summary.channel.isin(candidates.channel)].copy()
    ag=[]
    mapping={"scanner":"max_early_cls_scanner_cosdist","routing":"max_mid_incoming_cosdist","b22q":"max_b22_cls_q_cosine_distance","effect":"max_max_embedding_distance","correction":"max_content_correction_delta_norm","register":"min_final_reg_jaccard","router":"max_abs_source_gate_delta"}
    for c,g in q.groupby("channel"):
        rec={"channel":int(c)}
        for short,col in mapping.items():
            if col in g:
                vals=pd.to_numeric(g[col],errors="coerce")
                if short=="register": vals=1-vals
                rec[f"max_{short}"]=float(vals.max());rec[f"argmax_{short}"]=str(g.iloc[int(np.nanargmax(vals.to_numpy(float)))].condition) if vals.notna().any() else ""
            else: rec[f"max_{short}"]=np.nan;rec[f"argmax_{short}"]=""
        ag.append(rec)
    df=candidates.merge(pd.DataFrame(ag),on="channel",how="left")
    for s in mapping:df[f"pct_{s}"]=percentile01(df[f"max_{s}"].fillna(0))
    df["causal_combined_legacy"]=np.sqrt((df.pct_scanner**2+df.pct_routing**2+df.pct_b22q**2)/3)
    df["xattn_oddity_score"]=np.sqrt((df.pct_effect**2+df.pct_correction**2+df.pct_register**2+df.pct_router**2+df.causal_combined_legacy**2)/5)
    arr=df[["pct_scanner","pct_routing","pct_b22q"]].to_numpy(float);names=np.array(["scanner","routing","b22q"],object);df["dominant_causal_axis"]=names[np.argmax(arr,axis=1)]
    feat=pd.DataFrame({"log_pos":np.log10(pd.to_numeric(df.pos_to_pixel_std_ratio,errors="coerce").clip(lower=1e-6)),"log_rms":np.log10(pd.to_numeric(df.activation_rms,errors="coerce").clip(lower=1e-6)),"scanner":df.pct_scanner,"routing":df.pct_routing,"b22q":df.pct_b22q,"effect":df.pct_effect,"correction":df.pct_correction,"register":df.pct_register,"router":df.pct_router}).replace([np.inf,-np.inf],np.nan).fillna(0)
    X=StandardScaler().fit_transform(feat);best=(None,-1,None)
    for k in range(2,min(8,len(df)-1)+1):
        km=KMeans(n_clusters=k,n_init=30,random_state=SEED);lab=km.fit_predict(X)
        try:sc=float(silhouette_score(X,lab))
        except Exception:sc=-1
        if sc>best[1]:best=(k,sc,lab.copy())
    df["phase_cluster"]=best[2] if best[2] is not None else 0;df["phase_cluster_k"]=best[0] or 1;df["phase_cluster_silhouette"]=best[1]
    write_df(df,out/"functional_phase_metrics.csv");return df


def culprit_ranking(phase:pd.DataFrame,summary:pd.DataFrame,out:Path)->pd.DataFrame:
    q=summary[summary.source.eq("all")].copy(); q["cell_effect"]=q[[c for c in ["max_backbone_cosine_distance","max_content_cosine_distance"] if c in q]].max(axis=1)
    best=q.sort_values(["cell_effect","max_content_correction_delta_norm"],ascending=False).groupby("channel",sort=False).head(1)[["channel","condition","cell_effect"]].rename(columns={"condition":"best_condition"})
    d=phase.merge(best,on="channel",how="left"); d["primary_family"]="other"
    scorecols={"strong_embedding_steerer":"pct_effect","routing_bus_like":"pct_routing","late_cls_q_sensitive":"pct_b22q","content_correction_sensitive":"pct_correction","register_allocator_like":"pct_register","reader_router_sensitive":"pct_router"}
    vals=np.stack([d[c].fillna(0).to_numpy(float) for c in scorecols.values()],1);keys=list(scorecols);d["primary_family"]=[keys[i] for i in np.argmax(vals,axis=1)]
    tags=[]
    for r in d.itertuples(index=False):
        t=[]
        for label,col in scorecols.items():
            if float(getattr(r,col,0) or 0)>=.90:t.append(label)
        if float(getattr(r,"pos_to_pixel_std_ratio",0) or 0)>=float(phase.pos_to_pixel_std_ratio.quantile(.90)):t.append("positional_scaffold_like")
        if float(getattr(r,"cluster_outlier_z",0) or 0)>=2.5:t.append("morphology_outlier")
        if hasattr(r,"kernel_drift_score") and pd.notna(r.kernel_drift_score) and float(r.kernel_drift_score)>=float(phase.kernel_drift_score.quantile(.90)):t.append("weight_drift_outlier")
        tags.append(";".join(dict.fromkeys(t)) or "moderate")
    d["family_tags"]=tags;d=d.sort_values(["xattn_oddity_score","cell_effect"],ascending=False).reset_index(drop=True);d["overall_rank"]=np.arange(1,len(d)+1);write_df(d,out/"culprit_channel_ranking.csv");return d


def legacy_severe_functional_clusters(master: pd.DataFrame, causal_per: pd.DataFrame, culprit: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Legacy-compatible three-way severe/outlier clustering.

    The historical names are retained for continuity, but are descriptive labels
    assigned from cluster medians, not mechanistic ground truth. Statistical
    outliers are accepted in addition to absolute severe events so a cleaner model
    does not produce an empty analysis merely because it no longer crosses 0.97.
    """
    if culprit.empty:
        empty=pd.DataFrame(); write_df(empty,out/"severe_only_cluster_assignments.csv"); write_df(empty,out/"severe_only_cluster_summary.csv"); return empty
    keys=culprit[["stim_id","channel","condition"]].drop_duplicates()
    q=causal_per.merge(keys,on=["stim_id","channel","condition"],how="inner")
    agg=q.groupby("channel").agg(
        severe_event_count=("stim_id","count"),
        max_final_dist=("max_embedding_distance","max"),
        mean_final_dist=("max_embedding_distance","mean"),
        max_scanner=("early_cls_scanner_cosdist","max"),
        mean_scanner=("early_cls_scanner_cosdist","mean"),
        max_routing=("mid_incoming_cosdist","max"),
        mean_routing=("mid_incoming_cosdist","mean"),
        max_b22=("b22_cls_q_cosine_distance","max"),
        mean_b22=("b22_cls_q_cosine_distance","mean"),
        max_correction=("content_correction_delta_norm","max"),
        max_register_disruption=("final_reg_jaccard","min"),
        max_router=("abs_source_gate_delta","max"),
    ).reset_index()
    agg["max_register_disruption"]=1-agg.max_register_disruption
    feat=master.merge(agg,on="channel",how="inner")
    if len(feat)<3:
        feat["functional_cluster_id"]=0; feat["functional_cluster"]="insufficient_for_three_way_clustering"; write_df(feat,out/"severe_only_cluster_assignments.csv"); write_df(pd.DataFrame([{"functional_cluster":"insufficient_for_three_way_clustering","n":len(feat)}]),out/"severe_only_cluster_summary.csv"); return feat
    X=pd.DataFrame({
        "log_pos_ratio":np.log10(feat.pos_to_pixel_std_ratio.clip(lower=1e-6)),
        "log_activation_rms":np.log10(feat.activation_rms.clip(lower=1e-6)),
        "scanner":feat.max_scanner.fillna(0),
        "routing":feat.max_routing.fillna(0),
        "b22":feat.max_b22.fillna(0),
    }).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
    Xz=StandardScaler().fit_transform(X); km=KMeans(n_clusters=3,n_init=64,random_state=SEED); labels=km.fit_predict(Xz); feat["functional_cluster_id"]=labels
    med=feat.groupby("functional_cluster_id").agg(pos=("pos_to_pixel_std_ratio","median"),rms=("activation_rms","median"),routing=("max_routing","median"),b22=("max_b22","median"))
    positional_id=int(med.pos.idxmax()); rem=[int(c) for c in med.index if int(c)!=positional_id]; routing_score=med.loc[rem,"routing"]+.6*med.loc[rem,"b22"]+.25*med.loc[rem,"rms"]; active_id=int(routing_score.idxmax()); quiet_id=int([c for c in rem if c!=active_id][0]); mapping={positional_id:"positional_scaffold",active_id:"active_routing_buses",quiet_id:"quiet_bridge_control"}; feat["functional_cluster"]=feat.functional_cluster_id.map(mapping)
    sil=float(silhouette_score(Xz,labels)) if len(feat)>=4 else np.nan
    summ=feat.groupby("functional_cluster").agg(n=("channel","count"),pos_to_pixel_median=("pos_to_pixel_std_ratio","median"),activation_rms_median=("activation_rms","median"),max_scanner_median=("max_scanner","median"),max_routing_median=("max_routing","median"),max_b22_median=("max_b22","median"),max_correction_median=("max_correction","median"),max_register_disruption_median=("max_register_disruption","median"),max_router_median=("max_router","median"),culprit_event_count_total=("severe_event_count","sum")).reset_index(); summ["silhouette"]=sil
    write_df(feat,out/"severe_only_cluster_assignments.csv"); write_df(summ,out/"severe_only_cluster_summary.csv"); return feat


# -----------------------------------------------------------------------------
# Plots: phase / morphology / culprit frequencies
# -----------------------------------------------------------------------------

def plot_phase(phase:pd.DataFrame,master:pd.DataFrame,out:Path)->None:
    pdir=out/"plots";pdir.mkdir(parents=True,exist_ok=True);x=pd.to_numeric(phase.pos_to_pixel_std_ratio,errors="coerce").clip(lower=1e-5);y=pd.to_numeric(phase.activation_rms,errors="coerce").clip(lower=1e-5);size=28+320*np.square(phase.causal_combined_legacy.to_numpy(float));rgb=np.stack([.15+.8*phase.pct_scanner,.15+.8*phase.pct_routing,.15+.8*phase.pct_b22q],1).clip(0,1);top=set(phase.head(min(0,len(phase))).channel.tolist())|set(phase.nlargest(min(24,len(phase)),"xattn_oddity_score").channel.astype(int))
    fig,ax=plt.subplots(figsize=(12.8,9.2));ctrl=master[master.population.eq("controls")]
    if len(ctrl):ax.scatter(ctrl.pos_to_pixel_std_ratio.clip(lower=1e-5),ctrl.activation_rms.clip(lower=1e-5),s=15,facecolors="none",edgecolors="0.75",alpha=.5,label="high-weight twins")
    for pop,mark in (("sparse_tail","o"),("transition_band","^"),("other","s")):
        m=phase.population.eq(pop);ax.scatter(x[m],y[m],s=size[m],c=rgb[m],marker=mark,edgecolors="black",linewidths=.4,alpha=.85,label=pop)
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("positional std / natural Conv1 pixel std");ax.set_ylabel("natural Conv1 activation RMS");ax.set_title("Conv1 functional phase diagram\nRGB = scanner / routing / B22-Q; size = legacy causal leverage");ax.grid(alpha=.15,which="both");ax.legend(fontsize=8)
    for r in phase[phase.channel.isin(top)].itertuples(index=False):ax.annotate(str(int(r.channel)),(max(r.pos_to_pixel_std_ratio,1e-5),max(r.activation_rms,1e-5)),xytext=(3,3),textcoords="offset points",fontsize=7)
    fig.tight_layout();fig.savefig(pdir/"01_LOW_TAIL_PHASE_DIAGRAM_RGB_CAUSALITY.png",dpi=220);plt.close(fig)
    domc={"scanner":"#d62728","routing":"#2ca02c","b22q":"#1f77b4"};fig,ax=plt.subplots(figsize=(12.8,9.2));ax.scatter(x,y,s=size,c=[domc.get(v,"0.5") for v in phase.dominant_causal_axis],edgecolors="black",linewidths=.4);ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("positional std / natural Conv1 pixel std");ax.set_ylabel("natural Conv1 activation RMS");ax.set_title("Dominant legacy causal axis");ax.grid(alpha=.15,which="both");fig.tight_layout();fig.savefig(pdir/"02_LOW_TAIL_PHASE_DIAGRAM_DOMINANT_AXIS.png",dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12.8,9.2));ax.scatter(x,y,s=size,c=phase.phase_cluster,cmap="tab10",edgecolors="black",linewidths=.4);ax.set_xscale("log");ax.set_yscale("log");ax.set_title(f"Descriptive x-attn functional clusters (k={int(phase.phase_cluster_k.iloc[0]) if len(phase) else 0}, silhouette={float(phase.phase_cluster_silhouette.iloc[0]) if len(phase) else np.nan:.3f})");ax.set_xlabel("positional std / natural Conv1 pixel std");ax.set_ylabel("natural Conv1 activation RMS");ax.grid(alpha=.15,which="both");fig.tight_layout();fig.savefig(pdir/"03_LOW_TAIL_PHASE_DIAGRAM_UNSUPERVISED.png",dpi=220);plt.close(fig)
    # Added decomposition: whether CONTENT repairs or amplifies Conv1 displacement.
    fig,ax=plt.subplots(figsize=(10.5,9));sc=ax.scatter(phase.max_effect*0+phase.get("max_effect",0),phase.get("max_effect",0),alpha=0) if False else None
    # Derive from phase if individual maxima were retained.
    if "max_effect" in phase.columns: pass
    # Use candidate summary columns if present.
    bx=phase.get("screen_max_backbone",pd.Series(np.zeros(len(phase)),index=phase.index));cy=phase.get("screen_max_content",pd.Series(np.zeros(len(phase)),index=phase.index));repair=bx-cy;pts=ax.scatter(bx,cy,c=repair,cmap="coolwarm",s=35+220*phase.xattn_oddity_score**2,edgecolors="black",linewidths=.35);lim=max(float(np.nanmax(bx)) if len(bx) else .01,float(np.nanmax(cy)) if len(cy) else .01,.01)*1.05;ax.plot([0,lim],[0,lim],ls="--",color="0.4",lw=1);ax.set_xlabel("global-screen max BACKBONE cosine distance");ax.set_ylabel("global-screen max CONTENT cosine distance");ax.set_title("CONTENT correction: perturbation amplification vs repair\nabove diagonal = CONTENT amplifies; below = CONTENT repairs");fig.colorbar(pts,ax=ax,label="BACKBONE dist - CONTENT dist (repair > 0)");ax.grid(alpha=.15);fig.tight_layout();fig.savefig(pdir/"04_XATTN_BACKBONE_VS_CONTENT_REPAIR.png",dpi=220);plt.close(fig)


def plot_culprit_families(ranking:pd.DataFrame,out:Path,culprit:Optional[pd.DataFrame]=None)->None:
    pdir=out/"plots";pdir.mkdir(parents=True,exist_ok=True)
    counts=ranking.primary_family.value_counts().sort_values(); write_df(counts.rename("n").reset_index().rename(columns={"index":"family"}),out/"culprit_family_frequency.csv")
    fig,ax=plt.subplots(figsize=(11,max(5,.45*len(counts)+2)));ax.barh(np.arange(len(counts)),counts.values);ax.set_yticks(np.arange(len(counts)));ax.set_yticklabels(counts.index);ax.set_xlabel("candidate channels");ax.set_title("Causal/x-attn culprit-family frequency");ax.grid(axis="x",alpha=.15);fig.tight_layout();fig.savefig(pdir/"05_CULPRIT_FAMILY_FREQUENCY.png",dpi=220);plt.close(fig)
    # Preserve the old FFT/morphology-family enrichment view as a separate artifact.
    morph=ranking.cluster_label.value_counts().sort_values(); write_df(morph.rename("n").reset_index().rename(columns={"index":"morphology_family"}),out/"culprit_morphology_family_frequency.csv")
    fig,ax=plt.subplots(figsize=(12,max(5.5,.38*len(morph)+2)));ax.barh(np.arange(len(morph)),morph.values);ax.set_yticks(np.arange(len(morph)));ax.set_yticklabels(morph.index,fontsize=8);ax.set_xlabel("candidate channels");ax.set_title("FFT/morphology family frequency among candidate culprits");ax.grid(axis="x",alpha=.15);fig.tight_layout();fig.savefig(pdir/"06_CULPRIT_MORPHOLOGY_FAMILY_FREQUENCY.png",dpi=220);plt.close(fig)
    if culprit is not None and len(culprit):
        z=culprit.merge(ranking[["channel","cluster_label"]].drop_duplicates("channel"),on="channel",how="left")
        tmp=z.groupby(["cluster_label","condition"]).size().unstack(fill_value=0)
        write_df(tmp.reset_index(),out/"culprit_morphology_family_x_intervention.csv")
        fig,ax=plt.subplots(figsize=(11,max(5.5,.36*len(tmp)+2)));im=ax.imshow(tmp.to_numpy(),aspect="auto",interpolation="nearest",cmap="magma");ax.set_xticks(range(len(tmp.columns)));ax.set_xticklabels(tmp.columns,rotation=35,ha="right");ax.set_yticks(range(len(tmp.index)));ax.set_yticklabels(tmp.index,fontsize=8);ax.set_title("Culprit events by morphology family and intervention");fig.colorbar(im,ax=ax,label="image-channel events");fig.tight_layout();fig.savefig(pdir/"06b_CULPRIT_MORPHOLOGY_X_INTERVENTION.png",dpi=220);plt.close(fig)


def render_filter_gallery(W:np.ndarray,master:pd.DataFrame,ranking:pd.DataFrame,out:Path,n:int=24)->None:
    chans=ranking.head(min(n,len(ranking))).channel.astype(int).tolist();pdir=out/"plots"/"TOP_FILTERS";pdir.mkdir(parents=True,exist_ok=True)
    for st in range(0,len(chans),6):
        cc=chans[st:st+6];fig,axes=plt.subplots(len(cc),3,figsize=(11,3.1*len(cc)),squeeze=False)
        for i,c in enumerate(cc):
            r=master[master.channel.eq(c)].iloc[0];axes[i,0].imshow(kernel_rgb(W[c]));axes[i,0].set_title(f"ch{c:04d} kernel\n{r.cluster_label}");axes[i,1].imshow(fft_power(W[c]),cmap="magma");axes[i,1].set_title(f"FFT\ncent={r.spectral_centroid:.2f} aniso={r.orientation_anisotropy:.2f}");axes[i,2].axis("off");rr=ranking[ranking.channel.eq(c)].iloc[0];axes[i,2].text(0,.95,f"rank {int(rr.overall_rank)}  score={rr.xattn_oddity_score:.3f}\nbest={rr.best_condition}\npop={r.population}\nweight z={r.robust_log_z:.2f}\npos/pixel={r.pos_to_pixel_std_ratio:.3g}\n{rr.family_tags}",va="top",family="monospace",fontsize=9)
            axes[i,0].axis("off");axes[i,1].axis("off")
        fig.tight_layout();fig.savefig(pdir/f"top_filters_{st//6:02d}.png",dpi=190);plt.close(fig)


# -----------------------------------------------------------------------------
# Severe/outlier events and embedding manifolds
# -----------------------------------------------------------------------------

def identify_culprit_events(per:pd.DataFrame,args,out:Path)->Tuple[pd.DataFrame,pd.DataFrame]:
    p=per.copy();p["severe_backbone"]=p.backbone_cosine_distance>(1-args.severe_cos_sim);p["severe_content"]=p.content_cosine_distance>(1-args.severe_cos_sim);p["severe_any"]=p.severe_backbone|p.severe_content
    severe=p[p.severe_any].copy();write_df(severe,out/"severe_single_events.csv")
    # Statistical outliers remain meaningful when a cleaner model never crosses the old absolute threshold.
    p["effect_robust_z"]=p.groupby("condition").max_embedding_distance.transform(lambda x:robust_z(x.to_numpy(float)));p["effect_pct"]=p.groupby("condition").max_embedding_distance.rank(pct=True);outlier=p[(p.effect_robust_z>=args.event_outlier_z)|(p.effect_pct>=args.event_outlier_pct)].copy();write_df(outlier,out/"outlier_single_events.csv")
    culprit=pd.concat([severe,outlier],ignore_index=True).drop_duplicates(["stim_id","channel","condition"]);write_df(culprit,out/"culprit_single_events.csv");return severe,culprit


@torch.inference_mode()
def collect_culprit_vectors(model,preprocess,manifest:pd.DataFrame,culprit:pd.DataFrame,args,out:Path)->Tuple[pd.DataFrame,Dict[str,np.ndarray],Dict[str,np.ndarray],Dict[Tuple[str,int,str],Tuple[np.ndarray,np.ndarray]]]:
    meta_path=out/"embedding_event_metadata.csv"; vec_path=out/"embedding_event_vectors.safetensors"
    if args.resume and meta_path.is_file() and vec_path.is_file():
        mdf=pd.read_csv(meta_path)
        tensors=__import__("safetensors.torch",fromlist=["load_file"]).load_file(str(vec_path))
        if "backbone" not in tensors or "content" not in tensors:
            raise RuntimeError(f"Resume vector file missing required tensors: {vec_path}")
        if int(tensors["backbone"].shape[0])!=len(mdf) or int(tensors["content"].shape[0])!=len(mdf):
            raise RuntimeError(f"Resume vector/metadata row mismatch: metadata={len(mdf)} backbone={tensors['backbone'].shape[0]} content={tensors['content'].shape[0]}")
        print(f"[culprit vectors] resume: loading {len(mdf)} saved vectors; no model rerun")
        return mdf,{}, {}, {}
    baseB={};baseC={};meta_lookup={str(r.stim_id):r for r in manifest.itertuples(index=False)}
    for st in range(0,len(manifest),args.batch_size):
        m=manifest.iloc[st:st+args.batch_size];batch,ids,_=load_batch(preprocess,m,args.device);probes=probe_batch(model,batch,ids,args)
        for sid,p in zip(ids,probes):baseB[sid]=p.backbone.copy();baseC[sid]=p.content.copy()
        del batch,probes
    altered={};rows=[]
    cells=culprit[["channel","condition"]].drop_duplicates().sort_values(["condition","channel"])
    for ci,r in enumerate(cells.itertuples(index=False),1):
        c=int(r.channel);cond=str(r.condition);ids_need=set(culprit[(culprit.channel==c)&(culprit.condition==cond)].stim_id.astype(str));mm=manifest[manifest.stim_id.astype(str).isin(ids_need)]
        for st in range(0,len(mm),args.batch_size):
            sub=mm.iloc[st:st+args.batch_size];batch,ids,_=load_batch(preprocess,sub,args.device)
            with MultiConv1Transform(model,[(c,cond)],ids):probes=probe_batch(model,batch,ids,args)
            for sid,p in zip(ids,probes):altered[(sid,c,cond)]=(p.backbone.copy(),p.content.copy());mr=meta_lookup[sid];rows.append({"kind":"altered","stim_id":sid,"pair":mr.pair,"source":mr.source,"channel":c,"condition":cond})
            del batch,probes
        print(f"[culprit vectors {ci}/{len(cells)}] ch{c:04d} {cond}")
    # Safetensor ordering: baseline then altered, same metadata rows for both spaces.
    meta=[];VB=[];VC=[]
    for r in manifest.itertuples(index=False):meta.append({"kind":"baseline","stim_id":str(r.stim_id),"pair":str(r.pair),"source":str(r.source),"channel":-1,"condition":"BASELINE"});VB.append(baseB[str(r.stim_id)]);VC.append(baseC[str(r.stim_id)])
    for rec in rows:
        k=(rec["stim_id"],rec["channel"],rec["condition"]);b,c=altered[k];meta.append(rec);VB.append(b);VC.append(c)
    mdf=pd.DataFrame(meta);write_df(mdf,out/"embedding_event_metadata.csv");save_safetensors({"backbone":torch.from_numpy(np.stack(VB).astype(np.float16)),"content":torch.from_numpy(np.stack(VC).astype(np.float16))},str(out/"embedding_event_vectors.safetensors"),metadata={"format":"ModeMUX Conv1 culprit vectors","normalized":"true"})
    return mdf,baseB,baseC,altered


def _fit_coords(X:np.ndarray,do_umap:bool,do_tsne:bool)->Dict[str,np.ndarray]:
    out={};n=len(X)
    if n>=3:
        nc=min(3,X.shape[1],n);out["pca"]=PCA(n_components=nc,random_state=SEED).fit_transform(X)
    if do_umap and HAVE_UMAP and n>=4:
        out["umap"]=umap.UMAP(n_components=2,n_neighbors=min(15,n-1),min_dist=.15,metric="cosine",random_state=SEED).fit_transform(X)
    if do_tsne and n>=5:
        perp=max(2,min(30,(n-1)//3));out["tsne"]=TSNE(n_components=2,perplexity=perp,init="pca",learning_rate="auto",random_state=SEED).fit_transform(X)
    return out


def plot_manifold(meta:pd.DataFrame,X:np.ndarray,out:Path,prefix:str,space:str,args)->None:
    if len(meta)!=len(X):
        raise RuntimeError(f"Manifold metadata/vector row mismatch for {prefix}/{space}: metadata={len(meta)} vectors={len(X)}")
    c=_fit_coords(X,not args.skip_umap,not args.skip_tsne);pdir=out/"plots"/"MANIFOLDS";pdir.mkdir(parents=True,exist_ok=True)
    for name,P in c.items():
        cols={f"{name}{j+1}":P[:,j] for j in range(P.shape[1])};cdf=pd.concat([meta.reset_index(drop=True),pd.DataFrame(cols)],axis=1);write_df(cdf,out/f"{prefix}_{space}_{name}_coordinates.csv")
        has_kind="kind" in cdf.columns; has_source="source" in cdf.columns; has_condition="condition" in cdf.columns
        fig,ax=plt.subplots(figsize=(12,9))
        if has_kind:
            base=cdf[cdf["kind"].astype(str).eq("baseline")]
            if len(base):
                if has_source:
                    for src,col in (("clean","black"),("adv","0.32")):
                        bb=base[base["source"].astype(str).eq(src)];ax.scatter(bb[f"{name}1"],bb[f"{name}2"],s=38,c=col,marker="o",alpha=.78,label=f"baseline {src}")
                else:
                    ax.scatter(base[f"{name}1"],base[f"{name}2"],s=38,c="black",marker="o",alpha=.78,label="baseline")
            alt=cdf[~cdf["kind"].astype(str).eq("baseline")]
            bxy={str(r.stim_id):(float(getattr(r,f"{name}1")),float(getattr(r,f"{name}2"))) for r in base.itertuples(index=False)} if "stim_id" in base.columns else {}
        else:
            # Displacement/residual datasets need no artificial baseline rows.
            base=cdf.iloc[:0]; alt=cdf; bxy={}
        groups=alt.groupby("condition") if has_condition else [(prefix,alt)]
        for cond,g in groups:
            color=COND_COLORS.get(str(cond),"tab:pink");ax.scatter(g[f"{name}1"],g[f"{name}2"],s=48,c=color,marker="x",label=str(cond),alpha=.8)
            if prefix=="raw" and bxy and "stim_id" in g.columns:
                for rr in g.itertuples(index=False):
                    if str(rr.stim_id) in bxy:
                        bx,by=bxy[str(rr.stim_id)];ax.annotate("",xy=(float(getattr(rr,f"{name}1")),float(getattr(rr,f"{name}2"))),xytext=(bx,by),arrowprops=dict(arrowstyle="->",lw=.5,color=color,alpha=.25))
        ax.set_title(f"{space} {prefix} {name.upper()} manifold");ax.grid(alpha=.12);handles,labels=ax.get_legend_handles_labels();uniq={l:h for h,l in zip(handles,labels)}
        if uniq: ax.legend(uniq.values(),uniq.keys(),fontsize=8,ncol=2,loc="upper right")
        fig.tight_layout();fig.savefig(pdir/f"{prefix}_{space}_{name}_2D.png",dpi=210);plt.close(fig)
        # Historical same-pair views only make sense when pair identity is available.
        if "pair" in cdf.columns:
            pairdir=pdir/f"{prefix}_{space}_{name}_PER_PAIR";pairdir.mkdir(parents=True,exist_ok=True)
            for pair,g in cdf.groupby("pair",dropna=False):
                if len(g)<2: continue
                fig,ax=plt.subplots(figsize=(8.5,7))
                if has_kind:
                    b=g[g["kind"].astype(str).eq("baseline")];a=g[~g["kind"].astype(str).eq("baseline")]
                else:
                    b=g.iloc[:0];a=g
                if len(b):
                    bc=["black" if x=="clean" else "0.35" for x in b["source"]] if has_source else "black"
                    ax.scatter(b[f"{name}1"],b[f"{name}2"],s=55,c=bc,marker="o",label="baseline")
                agroups=a.groupby("condition") if has_condition else [(prefix,a)]
                for cond,gg in agroups:ax.scatter(gg[f"{name}1"],gg[f"{name}2"],s=60,c=COND_COLORS.get(str(cond),"tab:pink"),marker="x",label=str(cond))
                ax.set_title(f"{space} {prefix} {name.upper()} — {pair}");ax.grid(alpha=.12);handles,labels=ax.get_legend_handles_labels();uniq={l:h for h,l in zip(handles,labels)}
                if uniq: ax.legend(uniq.values(),uniq.keys(),fontsize=8,loc="upper right")
                fig.tight_layout();fig.savefig(pairdir/f"{safe_name(pair)}.png",dpi=190);plt.close(fig)
        if P.shape[1]>=3:
            fig=plt.figure(figsize=(11,9));ax=fig.add_subplot(111,projection="3d");ax.scatter(P[:,0],P[:,1],P[:,2],s=30,alpha=.7);ax.set_title(f"{space} {prefix} PCA3");fig.tight_layout();fig.savefig(pdir/f"{prefix}_{space}_{name}_3D.png",dpi=190);plt.close(fig)


def build_displacement_meta(meta:pd.DataFrame,X:np.ndarray)->Tuple[pd.DataFrame,np.ndarray]:
    base={str(r.stim_id):X[i] for i,r in enumerate(meta.itertuples(index=False)) if r.kind=="baseline"};rows=[];vec=[]
    for i,r in enumerate(meta.itertuples(index=False)):
        if r.kind=="baseline" or str(r.stim_id) not in base:continue
        rows.append(r._asdict());vec.append(X[i]-base[str(r.stim_id)])
    return pd.DataFrame(rows),np.stack(vec) if vec else np.zeros((0,X.shape[1]),np.float32)


# -----------------------------------------------------------------------------
# Stack / pair synergy
# -----------------------------------------------------------------------------

def stack_metrics(base:np.ndarray,combo:np.ndarray,singles:Sequence[np.ndarray])->Dict[str,float]:
    actual=combo-base;linear=np.sum(np.stack([s-base for s in singles]),axis=0);res=actual-linear;ad=float(np.linalg.norm(actual));ld=float(np.linalg.norm(linear));nr=float(np.linalg.norm(res));single_d=[cosine_distance_np(base,s) for s in singles]
    return {"combined_cosine_distance":cosine_distance_np(base,combo),"sum_single_cosine_distance":float(np.sum(single_d)),"distance_excess_over_sum":float(cosine_distance_np(base,combo)-np.sum(single_d)),"distance_synergy_ratio":float(cosine_distance_np(base,combo)/max(float(np.sum(single_d)),EPS)),"actual_vs_linear_delta_cosine":cosine_np(actual,linear),"actual_delta_norm":ad,"linear_delta_norm":ld,"actual_over_linear_delta_norm":ad/max(ld,EPS),"nonlinear_residual_norm":nr,"nonlinear_residual_over_actual":nr/max(ad,EPS)}


@torch.inference_mode()
def run_specs(model,preprocess,manifest:pd.DataFrame,specs:Sequence[Tuple[int,str]],args)->Dict[str,Tuple[np.ndarray,np.ndarray]]:
    out={}
    for st in range(0,len(manifest),args.batch_size):
        m=manifest.iloc[st:st+args.batch_size];batch,ids,_=load_batch(preprocess,m,args.device)
        if specs:
            with MultiConv1Transform(model,specs,ids):probes=probe_batch(model,batch,ids,args)
        else:probes=probe_batch(model,batch,ids,args)
        for sid,p in zip(ids,probes):out[sid]=(p.backbone.copy(),p.content.copy())
        del batch,probes
    return out


@torch.inference_mode()
def run_synergy(model,preprocess,manifest:pd.DataFrame,culprit:pd.DataFrame,ranking:pd.DataFrame,summary:pd.DataFrame,args,out:Path)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,np.ndarray,np.ndarray]:
    same_path=out/"stack_synergy_same_image.csv"; fam_path=out/"stack_synergy_condition_family.csv"; pair_path=out/"pair_screen_summary.csv"; res_path=out/"stack_nonlinear_residuals.safetensors"; res_meta_path=out/"nonlinear_residual_metadata.csv"
    if args.resume and same_path.is_file() and fam_path.is_file() and pair_path.is_file():
        def _read(path:Path)->pd.DataFrame:
            try:return pd.read_csv(path)
            except pd.errors.EmptyDataError:return pd.DataFrame()
        same=_read(same_path); fam=_read(fam_path); pair=_read(pair_path); RB=np.zeros((0,1),np.float32);RC=np.zeros((0,1),np.float32)
        if res_path.is_file() and res_meta_path.is_file():
            t=__import__("safetensors.torch",fromlist=["load_file"]).load_file(str(res_path))
            if "backbone_residual" in t and "content_residual" in t:
                RB=t["backbone_residual"].float().numpy();RC=t["content_residual"].float().numpy();rm=_read(res_meta_path)
                if len(rm)!=len(RB) or len(rm)!=len(RC):raise RuntimeError(f"Resume residual row mismatch: metadata={len(rm)} backbone={len(RB)} content={len(RC)}")
        print(f"[synergy] resume: same={len(same)} family={len(fam)} pairs={len(pair)} residuals={len(RB)}; no model rerun")
        return same,fam,pair,RB,RC
    base=run_specs(model,preprocess,manifest,[],args);same_rows=[];fam_rows=[];resB=[];resC=[];res_meta=[]
    # Same-image stacks: exact image/condition top channels from culprit events.
    for (sid,cond),g in culprit.groupby(["stim_id","condition"]):
        g=g.sort_values("max_embedding_distance",ascending=False);chs=list(dict.fromkeys(g.channel.astype(int).tolist()));mr=manifest[manifest.stim_id.astype(str).eq(str(sid))]
        if mr.empty:continue
        one=mr.iloc[:1]
        for depth in args.same_image_depths:
            if len(chs)<depth:continue
            sel=chs[:depth];sing=[]
            for c in sel:sing.append(run_specs(model,preprocess,one,[(c,str(cond))],args)[str(sid)])
            combo=run_specs(model,preprocess,one,[(c,str(cond)) for c in sel],args)[str(sid)];bB,bC=base[str(sid)];mB=stack_metrics(bB,combo[0],[x[0] for x in sing]);mC=stack_metrics(bC,combo[1],[x[1] for x in sing]);mr0=one.iloc[0];rec={"stack_scope":"same_image","stim_id":str(sid),"pair":str(mr0.pair),"source":str(mr0.source),"condition":str(cond),"channels":";".join(map(str,sel)),"depth":depth};rec.update({f"backbone_{k}":v for k,v in mB.items()});rec.update({f"content_{k}":v for k,v in mC.items()});same_rows.append(rec);resB.append((combo[0]-bB)-np.sum(np.stack([x[0]-bB for x in sing]),0));resC.append((combo[1]-bC)-np.sum(np.stack([x[1]-bC for x in sing]),0));res_meta.append({**rec,"kind":"nonlinear_residual"})
    same=pd.DataFrame(same_rows);write_df(same,out/"stack_synergy_same_image.csv")
    # Condition-family stacks across all images.
    ss=summary[summary.source.eq("all")].copy();ss["global_score"]=ss.get("max_max_embedding_distance",0)+.5*ss.get("max_mid_incoming_cosdist",0)+.25*ss.get("max_b22_cls_q_cosine_distance",0)
    for cond,g in ss.groupby("condition"):
        chans=g.sort_values("global_score",ascending=False).channel.astype(int).tolist()
        for depth in args.global_stack_depths:
            if len(chans)<depth:continue
            sel=chans[:depth];singmap={c:run_specs(model,preprocess,manifest,[(c,str(cond))],args) for c in sel};combo=run_specs(model,preprocess,manifest,[(c,str(cond)) for c in sel],args)
            for r in manifest.itertuples(index=False):
                sid=str(r.stim_id);bB,bC=base[sid];sg=[singmap[c][sid] for c in sel];mB=stack_metrics(bB,combo[sid][0],[x[0] for x in sg]);mC=stack_metrics(bC,combo[sid][1],[x[1] for x in sg]);rec={"stack_scope":"condition_family","stim_id":sid,"condition":str(cond),"channels":";".join(map(str,sel)),"depth":depth};rec.update({f"backbone_{k}":v for k,v in mB.items()});rec.update({f"content_{k}":v for k,v in mC.items()});fam_rows.append(rec)
    fam=pd.DataFrame(fam_rows);write_df(fam,out/"stack_synergy_condition_family.csv")
    # Cross-condition pair screen: one best condition per top-ranked channel.
    top=ranking.dropna(subset=["best_condition"]).head(args.pair_screen_top_specs);specs=[(int(r.channel),str(r.best_condition)) for r in top.itertuples(index=False)];sub=choose_stride_subset(manifest,args.pair_screen_images);singlemaps={sp:run_specs(model,preprocess,sub,[sp],args) for sp in specs};pair_rows=[]
    for i in range(len(specs)):
        for j in range(i+1,len(specs)):
            a=specs[i];b=specs[j]
            if a[0]==b[0]:continue
            combo=run_specs(model,preprocess,sub,[a,b],args)
            for r in sub.itertuples(index=False):
                sid=str(r.stim_id);bB,bC=base[sid];sa=singlemaps[a][sid];sb=singlemaps[b][sid];mB=stack_metrics(bB,combo[sid][0],[sa[0],sb[0]]);mC=stack_metrics(bC,combo[sid][1],[sa[1],sb[1]]);rec={"stim_id":sid,"ch1":a[0],"cond1":a[1],"ch2":b[0],"cond2":b[1]};rec.update({f"backbone_{k}":v for k,v in mB.items()});rec.update({f"content_{k}":v for k,v in mC.items()});pair_rows.append(rec)
    pair=pd.DataFrame(pair_rows)
    if len(pair):
        agg=pair.groupby(["ch1","cond1","ch2","cond2"]).agg(n=("stim_id","size"),mean_content_residual_over_actual=("content_nonlinear_residual_over_actual","mean"),max_content_residual_over_actual=("content_nonlinear_residual_over_actual","max"),mean_content_actual_vs_linear=("content_actual_vs_linear_delta_cosine","mean"),mean_content_effect=("content_combined_cosine_distance","mean"),mean_backbone_residual_over_actual=("backbone_nonlinear_residual_over_actual","mean")).reset_index();agg["pair_score"]=agg.mean_content_residual_over_actual*agg.mean_content_effect;agg=agg.sort_values("pair_score",ascending=False);write_df(pair,out/"pair_screen_per_image.csv");write_df(agg,out/"pair_screen_summary.csv");pair=agg
    else:write_df(pair,out/"pair_screen_summary.csv")
    if resB:
        RB=np.stack(resB).astype(np.float32);RC=np.stack(resC).astype(np.float32);save_safetensors({"backbone_residual":torch.from_numpy(RB.astype(np.float16)),"content_residual":torch.from_numpy(RC.astype(np.float16))},str(out/"stack_nonlinear_residuals.safetensors"));write_df(pd.DataFrame(res_meta),out/"nonlinear_residual_metadata.csv")
    else:RB=np.zeros((0,1),np.float32);RC=np.zeros((0,1),np.float32)
    return same,fam,pair,RB,RC


def plot_synergy(same:pd.DataFrame,fam:pd.DataFrame,pair:pd.DataFrame,out:Path)->None:
    pdir=out/"plots"/"STACK_SYNERGY";pdir.mkdir(parents=True,exist_ok=True)
    if len(same):
        fig,ax=plt.subplots(figsize=(11,8));ax.scatter(same.content_sum_single_cosine_distance,same.content_combined_cosine_distance,s=48,alpha=.75);lim=max(same.content_sum_single_cosine_distance.max(),same.content_combined_cosine_distance.max(),.01)*1.05;ax.plot([0,lim],[0,lim],ls="--",color=".4");ax.set_xlabel("sum single CONTENT cosine distances");ax.set_ylabel("stacked CONTENT cosine distance");ax.set_title("Same-image stacks: geometric 1+1=5 screen");ax.grid(alpha=.15);fig.tight_layout();fig.savefig(pdir/"07_SAME_IMAGE_STACK_DISTANCE_SYNERGY.png",dpi=220);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,8));ax.scatter(same.content_actual_vs_linear_delta_cosine,same.content_actual_over_linear_delta_norm,s=48,alpha=.75);ax.axvline(1,ls="--",color=".5");ax.axhline(1,ls="--",color=".5");ax.set_xlabel("cos(actual CONTENT Δ, sum single Δ)");ax.set_ylabel("||actual CONTENT Δ|| / ||sum single Δ||");ax.set_title("Same-image stack vector nonlinearity");ax.grid(alpha=.15);fig.tight_layout();fig.savefig(pdir/"08_SAME_IMAGE_STACK_VECTOR_NONLINEARITY.png",dpi=220);plt.close(fig)
    if len(fam):
        summ=fam.groupby(["condition","depth"]).content_combined_cosine_distance.mean().reset_index();write_df(summ,out/"stack_synergy_condition_family_summary.csv");fig,ax=plt.subplots(figsize=(12,7));x=np.arange(len(summ));ax.bar(x,summ.content_combined_cosine_distance);ax.set_xticks(x);ax.set_xticklabels([f"{r.condition}\ntop{int(r.depth)}" for r in summ.itertuples(index=False)],rotation=45,ha="right");ax.set_ylabel("mean CONTENT cosine distance");ax.set_title("Condition-family culprit stacks");fig.tight_layout();fig.savefig(pdir/"09_CONDITION_FAMILY_STACK_MEAN_DISTANCE.png",dpi=220);plt.close(fig)
    if len(pair):
        top=pair.head(min(30,len(pair)));fig,ax=plt.subplots(figsize=(11,8));ax.scatter(top.mean_content_effect,top.mean_content_residual_over_actual,s=55);ax.set_xlabel("mean pair CONTENT displacement");ax.set_ylabel("mean nonlinear residual / actual");ax.set_title("Cross-condition pair screen");ax.grid(alpha=.15)
        for r in top.head(12).itertuples(index=False):ax.annotate(f"{r.ch1}:{r.cond1}+{r.ch2}:{r.cond2}",(r.mean_content_effect,r.mean_content_residual_over_actual),fontsize=7,xytext=(3,3),textcoords="offset points")
        fig.tight_layout();fig.savefig(pdir/"10_CROSS_CONDITION_PAIR_SCREEN.png",dpi=220);plt.close(fig)


# -----------------------------------------------------------------------------
# Head-signature similarities and shortlist report
# -----------------------------------------------------------------------------

def head_signature_pair_similarity(head:pd.DataFrame)->pd.DataFrame:
    if head.empty:return pd.DataFrame()
    cols=["cls_to_reg_delta","reg_to_cls_delta","all_to_reg_delta"]
    vecs={}
    for (c,cond),g in head.groupby(["channel","condition"]):
        g=g.sort_values(["block","head"]);vecs[(int(c),str(cond))]=g[cols].to_numpy(float).reshape(-1)
    keys=list(vecs);rows=[]
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            a,b=keys[i],keys[j]
            if a[0]==b[0]:continue
            rows.append({"ch1":a[0],"cond1":a[1],"ch2":b[0],"cond2":b[1],"head_signature_cosine":cosine_np(vecs[a],vecs[b])})
    return pd.DataFrame(rows)


def write_shortlist(ranking:pd.DataFrame,pair:pd.DataFrame,headsim:pd.DataFrame,args,out:Path)->None:
    lines=["ModeMUX/x-attention Conv1 culprit shortlist","="*48,"",f"Model: {args.model}",f"Severe threshold: cosine similarity < {args.severe_cos_sim:.4f}",f"Candidates scanned fully: {len(ranking)}","", "Interpretation guardrail: family names below are descriptive evidence tags, not claims that one Conv1 channel implements a standalone semantic feature.",""]
    lines += ["TOP CHANNELS","------------"]
    for r in ranking.head(min(args.shortlist_channels,len(ranking))).itertuples(index=False):
        lines.append(f"#{int(r.overall_rank):02d} ch{int(r.channel):04d}  {str(r.best_condition):8s}  score={float(r.xattn_oddity_score):.3f}  effect={float(r.cell_effect):.4f}  {r.family_tags}")
    lines += ["", "BY PRIMARY FAMILY", "-----------------"]
    for fam,g in ranking.groupby("primary_family",sort=False):
        vals=", ".join(f"{int(r.channel)}:{r.best_condition}" for r in g.head(8).itertuples(index=False));lines.append(f"{fam}: {vals}")
    lines += ["", "READY FOR MANIFOLD EXPLORER — SINGLES", "-------------------------------------", '"singles": [']
    for i,r in enumerate(ranking.head(min(args.shortlist_channels,len(ranking))).itertuples(index=False)):
        comma="," if i<min(args.shortlist_channels,len(ranking))-1 else "";lines.append(f'  {{"channel": {int(r.channel)}, "condition": "{r.best_condition}"}}{comma}')
    lines += ["]", "", "READY FOR MANIFOLD EXPLORER — PAIRS", "-----------------------------------", '"pairs": [']
    if len(pair):
        pr=pair.head(min(args.shortlist_pairs,len(pair))).copy()
        if len(headsim):pr=pr.merge(headsim,on=["ch1","cond1","ch2","cond2"],how="left")
        for i,r in enumerate(pr.itertuples(index=False)):
            comma="," if i<len(pr)-1 else ""; hs=getattr(r,"head_signature_cosine",np.nan);lines.append(f'  {{"ch1": {int(r.ch1)}, "cond1": "{r.cond1}", "ch2": {int(r.ch2)}, "cond2": "{r.cond2}"}}{comma}  # pair_score={float(r.pair_score):.4g}, head_sig_cos={hs:.3f}')
    lines += ["]", "", "NOTES", "-----", "Positive content_repair means the CONTENT correction reduced the Conv1-induced BACKBONE displacement; negative means it amplified it.", "The pair screen is full-strength alpha=1 only; feed its best pairs into the GPIC manifold explorer for dense alpha surfaces."]
    (out/"CULPRIT_SHORTLIST.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    singles=[{"channel":int(r.channel),"condition":str(r.best_condition)} for r in ranking.head(min(args.shortlist_channels,len(ranking))).itertuples(index=False)];pairs=[]
    if len(pair):pairs=[{"ch1":int(r.ch1),"cond1":str(r.cond1),"ch2":int(r.ch2),"cond2":str(r.cond2),"pair_score":float(r.pair_score)} for r in pair.head(min(args.shortlist_pairs,len(pair))).itertuples(index=False)]
    (out/"experiment_candidates.json").write_text(json.dumps({"singles":singles,"pairs":pairs},indent=2),encoding="utf-8")


# -----------------------------------------------------------------------------
# Morphology stage + static sheet support
# -----------------------------------------------------------------------------

def run_morphology(model,preprocess,manifest:pd.DataFrame,args,out:Path,clip)->Tuple[pd.DataFrame,np.ndarray]:
    master_path=out/"filter_master.csv";maps_path=out/"conv1_static_maps.safetensors";W=model.visual.conv1.weight.detach().float().cpu().numpy()
    if args.resume and master_path.is_file() and maps_path.is_file():
        master=pd.read_csv(master_path);return master,W
    if args.resume and master_path.is_file() and not maps_path.is_file():
        # Partial-resume path: a previous run may have completed the morphology,
        # activation statistics, twin assignment, and pretrained drift audit but
        # failed only while serializing the static maps. Rebuild just those maps
        # from the 20-image manifest instead of repeating the full static census
        # and pretrained comparison.
        print("[morphology] partial resume: filter_master.csv exists; rebuilding missing static maps only")
        master=pd.read_csv(master_path)
        act,mean_map,pos_map,init_map=empirical_activation_and_position(model,preprocess,manifest,args.batch_size,args.device,args.amp)
        write_df(act,out/"activation_position_stats.csv")
        save_safetensors({"mean_map":torch.from_numpy(mean_map),"pos_map":torch.from_numpy(pos_map),"init_map":torch.from_numpy(init_map)},str(maps_path),metadata={"format":"Conv1 static maps"})
        return master,W
    print("[morphology] Conv1 static census")
    rows=[filter_metrics(W[c],c) for c in range(W.shape[0])];master0=pd.DataFrame(rows);master,clusters,audit=cluster_filter_metrics(master0,args.cluster_kmin,args.cluster_kmax,SEED)
    rz,med,sig=robust_log_z(master.weight_l2.to_numpy(float));master["robust_log_z"]=rz;master["population"]=[population_group(float(x)) for x in rz]
    act,mean_map,pos_map,init_map=empirical_activation_and_position(model,preprocess,manifest,args.batch_size,args.device,args.amp);master=master.merge(act,on="channel",how="left",validate="one_to_one")
    if args.compare_pretrained:
        try:
            drift=pretrained_drift(clip,model,master,args,out);master=master.merge(drift.drop(columns=["current_weight_l2"]),on="channel",how="left")
        except Exception as exc:
            if args.require_pretrained_compare:raise
            print(f"[warning] pretrained Conv1 comparison skipped: {type(exc).__name__}: {exc}")
    pairs,tops=build_twins(master,args.control_z_min);master["twin_channel"]=np.nan
    for r in pairs.itertuples(index=False):master.loc[master.channel.eq(int(r.target_channel)),"twin_channel"]=int(r.control_channel);master.loc[master.channel.eq(int(r.control_channel)),"population"]="controls"
    write_df(master,master_path);write_df(clusters,out/"morphology_cluster_summary.csv");(out/"morphology_cluster_audit.json").write_text(json.dumps({**audit,"log_weight_median":med,"log_weight_sigma":sig},indent=2),encoding="utf-8");write_df(act,out/"activation_position_stats.csv");write_df(pairs,out/"twin_pairs.csv");write_df(tops,out/"twin_candidates_top5.csv");save_safetensors({"mean_map":torch.from_numpy(mean_map),"pos_map":torch.from_numpy(pos_map),"init_map":torch.from_numpy(init_map)},str(maps_path),metadata={"format":"Conv1 static maps"})
    print(f"[population] sparse_tail={(master.population=='sparse_tail').sum()} transition={(master.population=='transition_band').sum()} controls={(master.population=='controls').sum()}")
    return master,W


# -----------------------------------------------------------------------------
# CLI / self-test / main
# -----------------------------------------------------------------------------

def parse_args(argv=None):
    ap=argparse.ArgumentParser(description="ModeMUX/x-attention Conv1 functional atlas + culprit/pair discovery")
    ap.add_argument("--model",default=DEFAULT_MODEL);ap.add_argument("--pretrained_model",default=DEFAULT_PRETRAINED);ap.add_argument("--model_revision",default="");ap.add_argument("--hf_cache_dir",default="");ap.add_argument("--repo_root",default="")
    ap.add_argument("--image_dir",default=DEFAULT_IMAGE_DIR);ap.add_argument("--recursive_images",action="store_true");ap.add_argument("--out_dir",default=DEFAULT_OUT);ap.add_argument("--device",default="cuda");ap.add_argument("--batch_size",type=int,default=8);ap.add_argument("--amp",action="store_true");ap.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--suite",choices=["morphology","screen","causal","all"],default="all")
    ap.add_argument("--screen_conditions",default=",".join(DEFAULT_SCREEN_CONDITIONS));ap.add_argument("--screen_images",type=int,default=8);ap.add_argument("--max_candidates",type=int,default=160);ap.add_argument("--candidate_top_per_axis",type=int,default=24)
    ap.add_argument("--severe_cos_sim",type=float,default=.97);ap.add_argument("--event_outlier_z",type=float,default=3.0);ap.add_argument("--event_outlier_pct",type=float,default=.99)
    ap.add_argument("--register_threshold",type=float,default=70.0);ap.add_argument("--reader_telemetry",action=argparse.BooleanOptionalAction,default=True);ap.add_argument("--head_signature",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--early_blocks",type=lambda s:parse_int_list(s),default=parse_int_list("0,1,2,3"));ap.add_argument("--routing_blocks",type=lambda s:parse_int_list(s),default=parse_int_list("7,8,9"));ap.add_argument("--register_k_block",type=int,default=12);ap.add_argument("--register_address_block",type=int,default=13);ap.add_argument("--late_q_block",type=int,default=22)
    ap.add_argument("--cluster_kmin",type=int,default=4);ap.add_argument("--cluster_kmax",type=int,default=14);ap.add_argument("--control_z_min",type=float,default=1.0)
    ap.add_argument("--compare_pretrained",action="store_true");ap.add_argument("--require_pretrained_compare",action="store_true")
    ap.add_argument("--same_image_depths",type=lambda s:parse_int_list(s),default=parse_int_list("2,3"));ap.add_argument("--global_stack_depths",type=lambda s:parse_int_list(s),default=parse_int_list("2,3,5"));ap.add_argument("--pair_screen_top_specs",type=int,default=12);ap.add_argument("--pair_screen_images",type=int,default=8)
    ap.add_argument("--skip_umap",action="store_true");ap.add_argument("--skip_tsne",action="store_true");ap.add_argument("--shortlist_channels",type=int,default=20);ap.add_argument("--shortlist_pairs",type=int,default=12);ap.add_argument("--self_test",action="store_true")
    return ap.parse_args(argv)


def self_test()->None:
    seed_all(123);x=np.array([1,1,1,1,.01,.02],float);z,_,_=robust_log_z(x);assert np.isfinite(z).all();base=np.array([1.,0.,0.]);s1=np.array([.98,.1,0.]);s2=np.array([.98,0,.1]);combo=base+(s1-base)+(s2-base);m=stack_metrics(base,combo,[s1,s2]);assert m["nonlinear_residual_norm"]<1e-7
    W=np.random.default_rng(0).normal(size=(8,3,14,14)).astype(np.float32);df=pd.DataFrame([filter_metrics(W[i],i) for i in range(8)]);assert len(df)==8 and set(MORPH_FEATURES).issubset(df.columns)
    # Regression: static positional maps can be transposed/non-contiguous views.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        noncontig=torch.arange(24,dtype=torch.float32).reshape(4,6).T
        assert not noncontig.is_contiguous()
        tp=str(Path(td)/"noncontiguous.safetensors")
        save_safetensors({"pos_map":noncontig},tp,metadata={"test":"noncontiguous"})
        loaded=__import__("safetensors.torch",fromlist=["load_file"]).load_file(tp)["pos_map"]
        assert torch.equal(loaded,noncontig)
    # Regression: residual metadata has no baseline/altered semantics and older runs
    # did not include kind/pair/source columns. It must still be renderable.
    rm=pd.DataFrame([{"stim_id":"x","condition":"FLIP","depth":2},{"stim_id":"x_adv","condition":"ZERO","depth":2}])
    mm=pd.DataFrame([{"stim_id":"x","pair":"x","source":"clean"},{"stim_id":"x_adv","pair":"x","source":"adv"}])
    rm2=prepare_residual_metadata(rm,mm);assert {"kind","pair","source"}.issubset(rm2.columns) and len(rm2)==2
    print("self-test OK")


def prepare_residual_metadata(meta:pd.DataFrame,manifest:pd.DataFrame)->pd.DataFrame:
    """Normalize old/new residual metadata for manifold rendering without recomputation."""
    z=meta.copy()
    if "kind" not in z.columns:z["kind"]="nonlinear_residual"
    if "stim_id" in z.columns and ("pair" not in z.columns or "source" not in z.columns):
        mm=manifest[["stim_id","pair","source"]].copy();mm["stim_id"]=mm["stim_id"].astype(str);z["stim_id"]=z["stim_id"].astype(str)
        z=z.merge(mm,on="stim_id",how="left",suffixes=("","_manifest"))
        if "pair_manifest" in z.columns:
            z["pair"]=z.get("pair",pd.Series(index=z.index,dtype=object)).fillna(z["pair_manifest"]);z=z.drop(columns=["pair_manifest"])
        if "source_manifest" in z.columns:
            z["source"]=z.get("source",pd.Series(index=z.index,dtype=object)).fillna(z["source_manifest"]);z=z.drop(columns=["source_manifest"])
    return z


def main(argv=None)->int:
    args=parse_args(argv);seed_all(SEED);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);(out/"plots").mkdir(exist_ok=True)
    if args.self_test:self_test();return 0
    repo,clip,model,preprocess,loader_info=load_modemux(args);audit=model_audit(model,args.model,loader_info);(out/"model_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8");print(json.dumps(audit,indent=2))
    nblocks=audit["n_blocks"]
    for b in [*args.early_blocks,*args.routing_blocks,args.register_k_block,args.register_address_block,args.late_q_block]:
        if not 0<=int(b)<nblocks:raise ValueError(f"Requested block {b} outside n_blocks={nblocks}")
    manifest=scan_images(Path(args.image_dir),args.recursive_images);write_df(manifest,out/"manifest.csv");print(f"[manifest] {len(manifest)} images from {args.image_dir}")
    master,W=run_morphology(model,preprocess,manifest,args,out,clip)
    if args.suite=="morphology":print(f"DONE: {out}");return 0
    # Global screen, intentionally all channels.
    screen_manifest=choose_stride_subset(manifest,args.screen_images);screen_path=out/"global_screen_per_image.csv";screen_conds=parse_str_list(args.screen_conditions);screen_per,_=run_scan(model,preprocess,screen_manifest,range(W.shape[0]),screen_conds,args,screen_path,stage_name="global screen",save_heads=False);screen_summary=summarize_causal(screen_per,out,"global_screen_summary.csv")
    candidates=select_candidates(master,screen_summary,args,out)
    # Merge screen features onto master for plots/ranking downstream.
    sc=screen_summary[screen_summary.source.eq("all")].groupby("channel").agg(screen_max_backbone=("max_backbone_cosine_distance","max"),screen_max_content=("max_content_cosine_distance","max")).reset_index()
    # filter_master.csv is reused on resume; replace prior derived screen columns
    # rather than accumulating _x/_y suffixes on every rerun.
    master=master.drop(columns=[c for c in ("screen_max_backbone","screen_max_content") if c in master.columns],errors="ignore").merge(sc,on="channel",how="left");write_df(master,out/"filter_master.csv")
    if args.suite=="screen":print(f"DONE: {out}");return 0
    causal_path=out/"causal_per_image.csv.gz";causal_per,head=run_scan(model,preprocess,manifest,candidates.channel.astype(int).tolist(),ALL_CONDITIONS,args,causal_path,stage_name="causal",save_heads=True);causal_summary=summarize_causal(causal_per,out,"causal_summary.csv");
    if len(head):write_df(head,out/"head_register_coupling_signature.csv.gz")
    phase=build_phase_metrics(master,causal_summary,candidates,out);phase=phase.merge(master[[c for c in master.columns if c in {"channel","screen_max_backbone","screen_max_content"}]],on="channel",how="left",suffixes=("","_m"));write_df(phase,out/"functional_phase_metrics.csv")
    ranking=culprit_ranking(phase,causal_summary,out);plot_phase(phase,master,out);render_filter_gallery(W,master,ranking,out)
    severe,culprit=identify_culprit_events(causal_per,args,out)
    legacy_clusters=legacy_severe_functional_clusters(master,causal_per,culprit,out)
    plot_culprit_families(ranking,out,culprit)
    if len(legacy_clusters):
        ranking=ranking.merge(legacy_clusters[["channel","functional_cluster"]].drop_duplicates("channel"),on="channel",how="left");write_df(ranking,out/"culprit_channel_ranking.csv")
    if args.suite=="causal":
        write_shortlist(ranking,pd.DataFrame(),head_signature_pair_similarity(head),args,out);print(f"DONE: {out}");return 0
    # Manifolds from exact culprit events.
    emeta,baseB,baseC,altered=collect_culprit_vectors(model,preprocess,manifest,culprit,args,out);vec=__import__("safetensors.torch",fromlist=["load_file"]).load_file(str(out/"embedding_event_vectors.safetensors"));XB=vec["backbone"].float().numpy();XC=vec["content"].float().numpy();plot_manifold(emeta,XB,out,"raw","backbone",args);plot_manifold(emeta,XC,out,"raw","content",args);dmeta,dB=build_displacement_meta(emeta,XB);_,dC=build_displacement_meta(emeta,XC);write_df(dmeta,out/"displacement_metadata.csv");
    if len(dB):save_safetensors({"backbone_delta":torch.from_numpy(dB.astype(np.float16)),"content_delta":torch.from_numpy(dC.astype(np.float16))},str(out/"displacement_vectors.safetensors"));plot_manifold(dmeta,dB,out,"delta","backbone",args);plot_manifold(dmeta,dC,out,"delta","content",args)
    same,fam,pair,RB,RC=run_synergy(model,preprocess,manifest,culprit,ranking,causal_summary,args,out);plot_synergy(same,fam,pair,out)
    if len(RB)>=3:
        rmeta=prepare_residual_metadata(pd.read_csv(out/"nonlinear_residual_metadata.csv"),manifest);write_df(rmeta,out/"nonlinear_residual_metadata.csv");plot_manifold(rmeta,RB,out,"nonlinear_residual","backbone",args);plot_manifold(rmeta,RC,out,"nonlinear_residual","content",args)
    hsim=head_signature_pair_similarity(head);write_df(hsim,out/"head_signature_pair_similarity.csv") if len(hsim) else None;write_shortlist(ranking,pair,hsim,args,out)
    report=["# ModeMUX Conv1 functional atlas","",f"Model: `{args.model}`",f"Images: {len(manifest)}",f"Full causal candidates: {len(candidates)}",f"Absolute severe events (sim < {args.severe_cos_sim}): {len(severe)}",f"Culprit events incl. statistical outliers: {len(culprit)}",f"Cross-condition pair surfaces screened: {len(pair)}","","Primary handoff: `CULPRIT_SHORTLIST.txt` and `experiment_candidates.json`."]
    (out/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8");print(f"DONE: {out}");return 0


if __name__=="__main__":
    raise SystemExit(main())
