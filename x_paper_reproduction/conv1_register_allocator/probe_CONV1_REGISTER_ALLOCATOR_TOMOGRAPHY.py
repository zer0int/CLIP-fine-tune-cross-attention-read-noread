#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONV1 / POSITION -> EARLY SCANNERS -> REGISTER ALLOCATOR TOMOGRAPHY
===================================================================

Native pretrained OpenAI CLIP ViT-L/14 only, deliberately.

Question
--------
Can the "register Pac-Man" phenomenon be turned into a causal probe of what
Conv1 / the same residual-coordinate positional embedding are doing?

This experiment keeps four things separate:

1) POSITIONAL PRIOR
   Feed many mathematically generated / stationary images and estimate the
   empirical register-address distribution. This tests whether the native
   model has preferred sacrificial/register sites independent of semantics.

2) EARLY-HEAD PHENOTYPE
   Move one salient synthetic target over a blank canvas. For B0--B3 CLS->patch
   attention, distinguish:
       * visual trackers: attention/argmax follows the target broadly;
       * rigid/gated positional scanners: a global field stays fixed, and local
         capture happens mainly when the target lies inside the field.
   Labels are descriptive; all continuous metrics are saved.

3) CONV1-vs-POSITION FACTORIZATION
   For selected residual coordinates, independently perturb:
       * the Conv1 kernel row,
       * the matching learned positional-embedding coordinate.
   Conv1 kernel tests are WITHIN-PATCH tests:
       KERNEL_DC_ONLY      keep only the spatial mean of each RGB kernel plane
       KERNEL_AC_ONLY      subtract that spatial mean
       *_NORMMATCH         same, rescaled back to the original row norm
       CONV1_ZERO
   Positional tests are GLOBAL PATCH-GRID tests:
       POS_ZERO            zero only patch-position values of coordinate c
       POS_FLAT            replace them by their patch-grid mean
       POS_SHUFFLE         deterministically permute them across patch positions
       BOTH_ZERO           CONV1_ZERO + POS_ZERO

   This is intentionally different from the older Conv1 activation-map
   DC_ONLY/CENTERED interventions, which acted ACROSS patch locations.

4) OPTIONAL PAC-MAN CHASE
   Repeatedly locate late high-norm register sites on a blank canvas and stamp
   unique protected motifs into newly selected sites. The trace estimates the
   model's revealed preference ordering over sacrificial patch locations and
   the point at which it begins reusing protected locations.

No training. No gradients. No rollout.

Outputs
-------
<out_dir>/
  config.json
  model_audit.json
  conv1_dc_census.csv
  register_samples.csv.gz
  register_position_prior.csv
  register_family_stability.csv
  positional_coordinate_prior_alignment.csv
  early_head_probe_metrics.csv
  early_head_register_prior_alignment.csv
  channel_static_metrics.csv
  channel_causal_summary.csv
  channel_register_prior.csv.gz
  channel_early_head_effects.csv.gz
  chase_trace.csv                         (if enabled)
  chase_summary.json                      (if enabled)
  plots/*.png
  REPORT.md
  compact_summary_conv1_register_allocator_tomography.zip

The script uses the bundled ``oaicliporg`` vanilla runtime with weights
resolved from Hugging Face through ``utils_clip_loader``. The manual visual forward
is validated against model.encode_image() before the experiment starts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
SEED = 20260917

DEFAULT_OUT = "out_paper_reproduction/conv1/register_allocator_tomography"
DEFAULT_CHANNELS = "199,499,120,469,227,350"
EARLY_BLOCKS = (0, 1, 2, 3)
REGISTER_STAGES = (11, 12, 13, 20, 23)

DEFAULT_INTERVENTIONS = (
    "CONV1_ZERO,"
    "KERNEL_DC_ONLY,KERNEL_AC_ONLY,"
    "KERNEL_DC_ONLY_NORMMATCH,KERNEL_AC_ONLY_NORMMATCH,"
    "POS_ZERO,POS_FLAT,POS_SHUFFLE,BOTH_ZERO"
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") & 0x7FFFFFFF


def parse_ints(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(s).split(",") if x.strip())


def parse_strs(s: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in str(s).split(",") if x.strip())


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in str(s))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > EPS else float("nan")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > EPS else float("nan")


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, np.float64).clip(min=0)
    q = np.asarray(q, np.float64).clip(min=0)
    p = (p + EPS) / (p.sum() + EPS * p.size)
    q = (q + EPS) / (q.sum() + EPS * q.size)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def robust_log_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64).clip(min=EPS)
    y = np.log(x)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    den = 1.4826 * mad
    if den < EPS:
        den = np.std(y) + EPS
    return (y - med) / den


def rank_desc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    order = np.argsort(-values, kind="mergesort")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(values) + 1)
    return rank


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Model loading / direct visual forward
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    import sys
    repo = next((p for p in [Path.cwd().resolve(), *Path(__file__).resolve().parents] if (p / "oaicliporg").is_dir() and (p / "utils_clip_loader").is_dir()), None)
    if repo is None:
        raise FileNotFoundError("Could not locate repo root containing oaicliporg and utils_clip_loader")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import oaicliporg as clip
    from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything
    model, preprocess, _ = load_openai_clip_anything(clip, model_name, device=device, jit=False, strict=True, allow_unsafe_hf_pickle=False)
    model.eval()
    return model, preprocess


def model_geometry(model) -> Tuple[int, int, int, int]:
    pe = model.visual.positional_embedding
    P = int(pe.shape[0] - 1)
    G = int(round(math.sqrt(P)))
    if G * G != P:
        raise RuntimeError(f"Expected square ViT patch grid, positional tokens={P}")
    k = model.visual.conv1.kernel_size
    patch = int(k[0] if isinstance(k, tuple) else k)
    image_size = G * patch
    width = int(pe.shape[1])
    return image_size, G, patch, width


def _linear_qkv_from_attention(attn, z_lbd: torch.Tensor):
    """Return q,k,v as [B,H,L,dh] for OpenAI nn.MultiheadAttention or explicit q_proj."""
    L, B, D = z_lbd.shape
    H = int(attn.num_heads)
    dh = D // H

    if hasattr(attn, "q_proj") and hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
        q = attn.q_proj(z_lbd)
        k = attn.k_proj(z_lbd)
        v = attn.v_proj(z_lbd)
    elif hasattr(attn, "in_proj_weight") and attn.in_proj_weight is not None:
        W = attn.in_proj_weight
        b = attn.in_proj_bias
        qw, kw, vw = W.chunk(3, dim=0)
        if b is None:
            qb = kb = vb = None
        else:
            qb, kb, vb = b.chunk(3, dim=0)
        q = F.linear(z_lbd, qw, qb)
        k = F.linear(z_lbd, kw, kb)
        v = F.linear(z_lbd, vw, vb)
    else:
        raise RuntimeError("Unsupported attention module: no explicit q/k/v projections or in_proj_weight")

    def reshape(t):
        return t.permute(1, 0, 2).reshape(B, L, H, dh).permute(0, 2, 1, 3).contiguous()

    return reshape(q), reshape(k), reshape(v)


def cls_patch_attention(block, x_pre_lbd: torch.Tensor, P: int) -> torch.Tensor:
    """Actual pre-softmax geometry reconstructed from the block's LN1 input.
    Returns CLS->spatial patch probabilities [B,H,P], excluding CLS from the output
    but retaining CLS in the softmax denominator.
    """
    z = block.ln_1(x_pre_lbd)
    q, k, _v = _linear_qkv_from_attention(block.attn, z)
    dh = q.shape[-1]
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(float(dh))
    probs = logits.softmax(dim=-1)
    return probs[:, :, 0, 1:1 + P]


@dataclass
class ForwardResult:
    embedding: torch.Tensor
    conv: Optional[torch.Tensor]
    pre: Dict[int, torch.Tensor]
    post23: Optional[torch.Tensor]
    early_attn: Dict[int, torch.Tensor]


def visual_forward_capture(
    model,
    images_cpu: torch.Tensor,
    *,
    capture_stages: Sequence[int] = REGISTER_STAGES,
    early_blocks: Sequence[int] = EARLY_BLOCKS,
    capture_conv: bool = False,
) -> ForwardResult:
    """Auditable OpenAI CLIP visual forward, using the model's own residual blocks."""
    device = next(model.parameters()).device
    dtype = model.visual.conv1.weight.dtype
    x_img = images_cpu.to(device=device, dtype=dtype)

    with torch.inference_mode():
        conv = model.visual.conv1(x_img)  # [B,C,G,G]
        B, C, G, _ = conv.shape
        P = G * G
        x = conv.reshape(B, C, P).permute(0, 2, 1)  # [B,P,C]

        cls = model.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros((B, 1, C), dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        pe = model.visual.positional_embedding.to(x.dtype)
        if pe.shape[0] != x.shape[1]:
            raise RuntimeError(
                f"Input grid tokens={x.shape[1]-1}, positional grid tokens={pe.shape[0]-1}. "
                "Generate inputs at the model's native resolution."
            )
        x = x + pe
        x = model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # L,B,D

        pre: Dict[int, torch.Tensor] = {}
        early: Dict[int, torch.Tensor] = {}
        wanted = set(int(b) for b in capture_stages)
        early_set = set(int(b) for b in early_blocks)

        for bi, block in enumerate(model.visual.transformer.resblocks):
            if bi in wanted:
                pre[bi] = x.detach().float().cpu()
            if bi in early_set:
                early[bi] = cls_patch_attention(block, x, P).detach().float().cpu()
            x = block(x)

        post23 = x.detach().float().cpu()
        x_bld = x.permute(1, 0, 2)
        cls_out = model.visual.ln_post(x_bld[:, 0, :])
        if model.visual.proj is not None:
            cls_out = cls_out @ model.visual.proj
        emb = cls_out.detach().float().cpu()

    return ForwardResult(
        embedding=emb,
        conv=conv.detach().float().cpu() if capture_conv else None,
        pre=pre,
        post23=post23,
        early_attn=early,
    )


def validate_manual_forward(model, preprocess, image_size: int, device: str) -> Dict[str, float]:
    rng = np.random.default_rng(123)
    arr = np.clip(rng.normal(0.5, 0.16, size=(image_size, image_size, 3)), 0, 1)
    im = Image.fromarray((arr * 255).astype(np.uint8), "RGB")
    batch = preprocess(im).unsqueeze(0)

    got = visual_forward_capture(model, batch, capture_stages=(), early_blocks=()).embedding
    with torch.inference_mode():
        ref = model.encode_image(batch.to(device)).detach().float().cpu()
    cos = F.cosine_similarity(got, ref, dim=-1).item()
    max_abs = (got - ref).abs().max().item()
    rel = (got - ref).norm().item() / max(ref.norm().item(), EPS)
    if cos < 0.99999 or rel > 5e-4:
        raise RuntimeError(
            f"Manual forward parity failed: cosine={cos:.8f}, rel_l2={rel:.3e}, max_abs={max_abs:.3e}"
        )
    return {"cosine": float(cos), "relative_l2": float(rel), "max_abs": float(max_abs)}


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------

class Conv1KernelIntervention:
    def __init__(self, model, channel: int, mode: str):
        self.model = model
        self.channel = int(channel)
        self.mode = str(mode)
        self.orig = None

    def __enter__(self):
        w = self.model.visual.conv1.weight
        c = self.channel
        if not (0 <= c < w.shape[0]):
            raise IndexError(c)
        self.orig = w[c].detach().clone()
        x = self.orig.float()
        dc = x.mean(dim=(-2, -1), keepdim=True).expand_as(x)
        ac = x - dc
        if self.mode == "CONV1_ZERO":
            z = torch.zeros_like(x)
        elif self.mode == "KERNEL_DC_ONLY":
            z = dc
        elif self.mode == "KERNEL_AC_ONLY":
            z = ac
        elif self.mode == "KERNEL_DC_ONLY_NORMMATCH":
            z = dc * (x.norm() / dc.norm().clamp_min(EPS))
        elif self.mode == "KERNEL_AC_ONLY_NORMMATCH":
            z = ac * (x.norm() / ac.norm().clamp_min(EPS))
        else:
            raise ValueError(self.mode)
        with torch.no_grad():
            w[c].copy_(z.to(device=w.device, dtype=w.dtype))
        return self

    def __exit__(self, *exc):
        if self.orig is not None:
            with torch.no_grad():
                self.model.visual.conv1.weight[self.channel].copy_(
                    self.orig.to(self.model.visual.conv1.weight)
                )


class PositionalCoordinateIntervention:
    def __init__(self, model, channel: int, mode: str, seed: int):
        self.model = model
        self.channel = int(channel)
        self.mode = str(mode)
        self.seed = int(seed)
        self.orig = None

    def __enter__(self):
        pe = self.model.visual.positional_embedding
        c = self.channel
        if not (0 <= c < pe.shape[1]):
            raise IndexError(c)
        self.orig = pe[:, c].detach().clone()
        patch = self.orig[1:].float()
        if self.mode == "POS_ZERO":
            z = torch.zeros_like(patch)
        elif self.mode == "POS_FLAT":
            z = patch.mean().expand_as(patch)
        elif self.mode == "POS_SHUFFLE":
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed)
            perm = torch.randperm(len(patch), generator=g).to(patch.device)
            z = patch.index_select(0, perm)
        else:
            raise ValueError(self.mode)
        with torch.no_grad():
            pe[1:, c].copy_(z.to(device=pe.device, dtype=pe.dtype))
        return self

    def __exit__(self, *exc):
        if self.orig is not None:
            with torch.no_grad():
                self.model.visual.positional_embedding[:, self.channel].copy_(
                    self.orig.to(self.model.visual.positional_embedding)
                )


@contextlib.contextmanager
def apply_intervention(model, channel: int, condition: str):
    condition = str(condition)
    seed = stable_seed(f"pos:{channel}:{condition}")
    if condition in {
        "CONV1_ZERO",
        "KERNEL_DC_ONLY", "KERNEL_AC_ONLY",
        "KERNEL_DC_ONLY_NORMMATCH", "KERNEL_AC_ONLY_NORMMATCH",
    }:
        with Conv1KernelIntervention(model, channel, condition):
            yield
    elif condition in {"POS_ZERO", "POS_FLAT", "POS_SHUFFLE"}:
        with PositionalCoordinateIntervention(model, channel, condition, seed):
            yield
    elif condition == "BOTH_ZERO":
        with Conv1KernelIntervention(model, channel, "CONV1_ZERO"):
            with PositionalCoordinateIntervention(model, channel, "POS_ZERO", seed):
                yield
    else:
        raise ValueError(condition)


# ---------------------------------------------------------------------------
# Synthetic stimuli
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StimSpec:
    stim_id: str
    family: str
    seed: int


def _rgb_from_wave(v: np.ndarray, phase2: float) -> np.ndarray:
    v = (v - v.min()) / (v.max() - v.min() + EPS)
    r = 0.18 + 0.72 * v
    g = 0.18 + 0.72 * (0.5 + 0.5 * np.sin(2 * np.pi * v + phase2))
    b = 0.18 + 0.72 * (1.0 - v)
    return np.stack([r, g, b], axis=-1)


def render_stationary(spec: StimSpec, size: int, patch: int) -> Image.Image:
    rng = np.random.default_rng(spec.seed)
    fam = spec.family
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    xn = xx / max(1, size - 1)
    yn = yy / max(1, size - 1)

    if fam == "solid":
        rgb = rng.uniform(0.16, 0.90, size=3)
        arr = np.broadcast_to(rgb[None, None, :], (size, size, 3)).copy()

    elif fam == "tiled_patch":
        tile = rng.uniform(0.05, 0.95, size=(patch, patch, 3))
        # low-pass the tile but keep it exactly repeated patch-for-patch
        pil = Image.fromarray((tile * 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(radius=1.2))
        tile = np.asarray(pil, np.float64) / 255.0
        reps = (math.ceil(size / patch), math.ceil(size / patch), 1)
        arr = np.tile(tile, reps)[:size, :size, :]

    elif fam == "checker_tile":
        q = max(1, patch // 4)
        base = (((xx // q) + (yy // q)) % 2).astype(np.float64)
        c0 = rng.uniform(0.10, 0.40, size=3)
        c1 = rng.uniform(0.62, 0.94, size=3)
        arr = c0[None, None, :] * (1 - base[..., None]) + c1[None, None, :] * base[..., None]

    elif fam == "sine":
        theta = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(1.0, 7.0)
        phase = rng.uniform(0, 2 * np.pi)
        v = np.sin(2 * np.pi * freq * (np.cos(theta) * xn + np.sin(theta) * yn) + phase)
        arr = _rgb_from_wave(v, rng.uniform(0, 2 * np.pi))

    elif fam == "fractal_sine":
        v = np.zeros((size, size), np.float64)
        amp = 1.0
        for octave in range(5):
            theta = rng.uniform(0, 2 * np.pi)
            freq = (1.25 * (2 ** octave)) * rng.uniform(0.8, 1.2)
            phase = rng.uniform(0, 2 * np.pi)
            v += amp * np.sin(2 * np.pi * freq * (np.cos(theta) * xn + np.sin(theta) * yn) + phase)
            amp *= 0.55
        arr = _rgb_from_wave(v, rng.uniform(0, 2 * np.pi))

    else:
        raise ValueError(fam)

    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


def make_stationary_specs(n_per_family: int, seed: int) -> List[StimSpec]:
    fams = ("solid", "tiled_patch", "checker_tile", "sine", "fractal_sine")
    rows = []
    for fi, fam in enumerate(fams):
        for i in range(int(n_per_family)):
            s = stable_seed(f"{seed}:{fam}:{i}")
            rows.append(StimSpec(f"{fam}_{i:04d}", fam, s))
    return rows


def draw_salient_target(size: int, patch: int, row: int, col: int) -> Image.Image:
    im = Image.new("RGB", (size, size), (246, 246, 246))
    d = ImageDraw.Draw(im)
    cx = int((col + 0.5) * patch)
    cy = int((row + 0.5) * patch)
    r = max(5, int(1.10 * patch))
    # A non-text, high-contrast, multi-frequency target.
    d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(255, 205, 20), outline=(10, 10, 10), width=max(2, patch//7))
    d.ellipse((cx-r//2, cy-r//2, cx+r//2, cy+r//2), fill=(20, 190, 235), outline=(20, 20, 20), width=max(1, patch//9))
    d.line((cx-r, cy, cx+r, cy), fill=(225, 20, 95), width=max(2, patch//5))
    d.line((cx, cy-r, cx, cy+r), fill=(45, 30, 220), width=max(2, patch//5))
    return im


def probe_positions(G: int, n_axis: int) -> List[Tuple[int, int]]:
    if n_axis <= 1:
        vals = [G // 2]
    else:
        vals = np.linspace(1, G - 2, n_axis)
        vals = sorted(set(int(round(x)) for x in vals))
    return [(r, c) for r in vals for c in vals]


def preprocess_images(preprocess, images: Sequence[Image.Image]) -> torch.Tensor:
    return torch.stack([preprocess(im) for im in images], dim=0)


# ---------------------------------------------------------------------------
# Register prior
# ---------------------------------------------------------------------------

def spatial_norms_from_state(state_lbd: torch.Tensor, B: int, P: int) -> torch.Tensor:
    # state: L,B,D -> B,L,D
    x = state_lbd.permute(1, 0, 2).float()
    if x.shape[1] < 1 + P:
        raise RuntimeError(f"Captured token count {x.shape[1]} < 1+P={1+P}")
    return x[:, 1:1 + P].norm(dim=-1)


def aggregate_prior(sample_df: pd.DataFrame, G: int) -> pd.DataFrame:
    rows = []
    group_cols = ["stage", "family"]
    for (stage, family), g in sample_df.groupby(group_cols, dropna=False):
        n = len(g)
        arg_counts = np.zeros(G * G, dtype=np.float64)
        reg_counts = np.zeros(G * G, dtype=np.float64)
        norm_sum = np.zeros(G * G, dtype=np.float64)
        for r in g.itertuples(index=False):
            arg_counts[int(r.argmax_patch0)] += 1
            regs = [int(x) for x in str(r.register_patches0).split("|") if str(x) != ""]
            for p in regs:
                if 0 <= p < G * G:
                    reg_counts[p] += 1
            nv = np.asarray(json.loads(r.spatial_norms_json), np.float64)
            norm_sum += nv
        for p in range(G * G):
            rows.append({
                "stage": str(stage),
                "family": str(family),
                "patch_index0": p,
                "patch_index1": p + 1,
                "row0": p // G,
                "col0": p % G,
                "argmax_frequency": arg_counts[p] / max(1, n),
                "register_frequency": reg_counts[p] / max(1, n),
                "mean_spatial_norm": norm_sum[p] / max(1, n),
                "n_images": n,
            })
    return pd.DataFrame(rows)


def register_prior_stage_vector(prior: pd.DataFrame, stage: str, family: str, metric: str, P: int) -> np.ndarray:
    q = prior[(prior["stage"].eq(stage)) & (prior["family"].eq(family))].sort_values("patch_index0")
    if len(q) != P:
        raise RuntimeError(f"Prior rows missing for {stage}/{family}: {len(q)} vs {P}")
    return q[metric].to_numpy(float)


def run_bank(
    model,
    preprocess,
    specs: Sequence[StimSpec],
    *,
    size: int,
    G: int,
    patch: int,
    batch_size: int,
    register_threshold: float,
    capture_early: bool,
    capture_conv_channels: Sequence[int] = (),
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
    P = G * G
    sample_rows: List[dict] = []
    early_sum: Dict[int, Optional[np.ndarray]] = {b: None for b in EARLY_BLOCKS}
    early_n = 0
    conv_stats = {
        int(c): {"sum": 0.0, "sum2": 0.0, "n": 0, "sum_abs": 0.0}
        for c in capture_conv_channels
    }

    for st in range(0, len(specs), batch_size):
        batch_specs = list(specs[st:st + batch_size])
        ims = [render_stationary(s, size, patch) for s in batch_specs]
        batch = preprocess_images(preprocess, ims)

        fr = visual_forward_capture(
            model, batch,
            capture_stages=REGISTER_STAGES,
            early_blocks=EARLY_BLOCKS if capture_early else (),
            capture_conv=bool(capture_conv_channels),
        )
        B = len(batch_specs)

        if capture_early:
            for b in EARLY_BLOCKS:
                a = fr.early_attn[b].numpy()  # B,H,P
                sm = a.sum(axis=0)
                early_sum[b] = sm if early_sum[b] is None else early_sum[b] + sm
            early_n += B

        if fr.conv is not None:
            for c in capture_conv_channels:
                x = fr.conv[:, int(c)].numpy().astype(np.float64)
                rec = conv_stats[int(c)]
                rec["sum"] += float(x.sum())
                rec["sum2"] += float((x * x).sum())
                rec["sum_abs"] += float(np.abs(x).sum())
                rec["n"] += int(x.size)

        stage_states = {f"pre{b}": fr.pre[b] for b in REGISTER_STAGES}
        stage_states["post23"] = fr.post23
        for stage, state in stage_states.items():
            norms = spatial_norms_from_state(state, B, P).numpy()
            for bi, spec in enumerate(batch_specs):
                nv = norms[bi]
                regs = np.flatnonzero(nv > register_threshold).astype(int).tolist()
                top4 = np.argsort(-nv)[:4].astype(int).tolist()
                sample_rows.append({
                    "stim_id": spec.stim_id,
                    "family": spec.family,
                    "seed": spec.seed,
                    "stage": stage,
                    "argmax_patch0": int(np.argmax(nv)),
                    "argmax_patch1": int(np.argmax(nv)) + 1,
                    "register_count": len(regs),
                    "register_patches0": "|".join(map(str, regs)),
                    "top4_patches0": "|".join(map(str, top4)),
                    "max_spatial_norm": float(nv.max()),
                    "spatial_norms_json": json.dumps([float(x) for x in nv]),
                })

        print(f"[bank] {min(st + B, len(specs))}/{len(specs)}")

    early_mean = {}
    if capture_early:
        early_mean = {b: early_sum[b] / max(1, early_n) for b in EARLY_BLOCKS}

    conv_out: Dict[int, Dict[str, float]] = {}
    for c, rec in conv_stats.items():
        n = max(1, int(rec["n"]))
        mu = rec["sum"] / n
        var = max(0.0, rec["sum2"] / n - mu * mu)
        conv_out[c] = {
            "activation_mean": mu,
            "activation_std": math.sqrt(var),
            "activation_mean_abs": rec["sum_abs"] / n,
            "activation_rms": math.sqrt(rec["sum2"] / n),
        }

    return pd.DataFrame(sample_rows), early_mean, conv_out


def family_stability(prior: pd.DataFrame, stage: str, P: int) -> pd.DataFrame:
    fams = sorted(x for x in prior["family"].unique() if x != "all")
    rows = []
    for i, a in enumerate(fams):
        for b in fams[i + 1:]:
            pa = register_prior_stage_vector(prior, stage, a, "argmax_frequency", P)
            pb = register_prior_stage_vector(prior, stage, b, "argmax_frequency", P)
            rows.append({
                "stage": stage,
                "family_a": a,
                "family_b": b,
                "pearson_argmax_prior": pearson(pa, pb),
                "jsd_argmax_prior": js_divergence(pa, pb),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Early head moving-target phenotype
# ---------------------------------------------------------------------------

def neighborhood_indices(row: int, col: int, G: int, radius: int = 1) -> np.ndarray:
    idx = []
    for rr in range(max(0, row - radius), min(G, row + radius + 1)):
        for cc in range(max(0, col - radius), min(G, col + radius + 1)):
            idx.append(rr * G + cc)
    return np.asarray(idx, np.int64)


def _corr_or_zero(a, b) -> float:
    v = pearson(np.asarray(a), np.asarray(b))
    return 0.0 if not np.isfinite(v) else float(v)


def moving_probe_analysis(
    model,
    preprocess,
    *,
    size: int,
    G: int,
    patch: int,
    n_axis: int,
    batch_size: int,
    out: Path,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], List[Tuple[int, int]]]:
    pos = probe_positions(G, n_axis)
    blank = Image.new("RGB", (size, size), (246, 246, 246))
    images = [blank] + [draw_salient_target(size, patch, r, c) for r, c in pos]

    maps_by_block = {b: [] for b in EARLY_BLOCKS}
    for st in range(0, len(images), batch_size):
        batch = preprocess_images(preprocess, images[st:st + batch_size])
        fr = visual_forward_capture(model, batch, capture_stages=(), early_blocks=EARLY_BLOCKS)
        for b in EARLY_BLOCKS:
            maps_by_block[b].append(fr.early_attn[b].numpy())
    maps_by_block = {b: np.concatenate(v, axis=0) for b, v in maps_by_block.items()}

    rows = []
    for b in EARLY_BLOCKS:
        A = maps_by_block[b]  # [1+n,H,P]
        H = A.shape[1]
        for h in range(H):
            base = A[0, h].astype(np.float64)
            base_n = base / (base.sum() + EPS)
            base_cv = float(base_n.std() / (base_n.mean() + EPS))
            vals = []
            for j, (r, c) in enumerate(pos):
                cur = A[1 + j, h].astype(np.float64)
                cur_n = cur / (cur.sum() + EPS)
                neigh = neighborhood_indices(r, c, G, 1)
                arg = int(np.argmax(cur_n))
                ar, ac = divmod(arg, G)
                captured = int(max(abs(ar - r), abs(ac - c)) <= 1)
                local = float(cur_n[neigh].sum())
                base_local = float(base_n[neigh].sum())
                rr = np.repeat(np.arange(G), G)
                cc = np.tile(np.arange(G), G)
                com_r = float(np.sum(cur_n * rr))
                com_c = float(np.sum(cur_n * cc))
                outside = np.ones(G * G, dtype=bool)
                outside[neigh] = False
                vals.append({
                    "r": r, "c": c, "arg_r": ar, "arg_c": ac, "captured": captured,
                    "local_gain": local - base_local,
                    "base_support": base_local,
                    "com_r": com_r, "com_c": com_c,
                    "template_cos": cosine(cur_n, base_n),
                    "outside_template_cos": cosine(cur_n[outside], base_n[outside]),
                })
            v = pd.DataFrame(vals)
            tracking_r = _corr_or_zero(v["r"], v["com_r"])
            tracking_c = _corr_or_zero(v["c"], v["com_c"])
            tracking = 0.5 * (tracking_r + tracking_c)
            capture_rate = float(v["captured"].mean())
            qlo = float(v["base_support"].quantile(1 / 3))
            qhi = float(v["base_support"].quantile(2 / 3))
            lo = v[v["base_support"] <= qlo]
            hi = v[v["base_support"] >= qhi]
            cap_lo = float(lo["captured"].mean()) if len(lo) else float("nan")
            cap_hi = float(hi["captured"].mean()) if len(hi) else float("nan")
            gap = cap_hi - cap_lo if np.isfinite(cap_hi) and np.isfinite(cap_lo) else float("nan")
            gain_support_corr = pearson(v["base_support"], v["local_gain"])
            template = float(v["template_cos"].mean())
            outside_template = float(v["outside_template_cos"].mean())
            local_gain = float(v["local_gain"].mean())

            if base_cv < 0.10 and abs(local_gain) < 0.02:
                phenotype = "uniform_or_weak"
            elif tracking > 0.55 and capture_rate > 0.60 and (not np.isfinite(gap) or gap < 0.35):
                phenotype = "visual_tracker"
            elif base_cv > 0.25 and template > 0.90 and (
                (np.isfinite(gap) and gap > 0.30) or tracking < 0.30
            ):
                phenotype = "rigid_or_gated_positional"
            else:
                phenotype = "hybrid"

            rigid_score = base_cv * template * (1.0 + max(0.0, gap if np.isfinite(gap) else 0.0))
            tracker_score = max(0.0, tracking) * capture_rate

            rows.append({
                "block": b, "head": h,
                "phenotype": phenotype,
                "baseline_spatial_cv": base_cv,
                "template_cosine_mean": template,
                "outside_template_cosine_mean": outside_template,
                "tracking_row_r": tracking_r,
                "tracking_col_r": tracking_c,
                "tracking_score": tracking,
                "capture_rate_r1": capture_rate,
                "capture_rate_low_baseline_support": cap_lo,
                "capture_rate_high_baseline_support": cap_hi,
                "support_capture_gap": gap,
                "gain_support_corr": gain_support_corr,
                "local_gain_mean": local_gain,
                "rigid_score": rigid_score,
                "tracker_score": tracker_score,
            })

    df = pd.DataFrame(rows)
    np.savez_compressed(
        out / "early_head_probe_maps.npz",
        positions=np.asarray(pos, np.int16),
        **{f"block_{b}": maps_by_block[b].astype(np.float32) for b in EARLY_BLOCKS},
    )
    return df, maps_by_block, pos


# ---------------------------------------------------------------------------
# Static Conv1 / positional metrics
# ---------------------------------------------------------------------------

def conv1_positional_census(model, G: int) -> pd.DataFrame:
    W = model.visual.conv1.weight.detach().float().cpu().numpy()
    pe = model.visual.positional_embedding.detach().float().cpu().numpy()
    rows = []
    norms = np.linalg.norm(W.reshape(W.shape[0], -1), axis=1)
    rz = robust_log_z(norms)
    for c in range(W.shape[0]):
        w = W[c].astype(np.float64)
        dc = np.broadcast_to(w.mean(axis=(-2, -1), keepdims=True), w.shape)
        total = float(np.sum(w * w))
        dce = float(np.sum(dc * dc))
        ac = w - dc
        p = pe[1:, c]
        rows.append({
            "channel": c,
            "weight_l2": norms[c],
            "robust_log_weight_z": rz[c],
            "kernel_dc_energy_frac": dce / max(total, EPS),
            "kernel_ac_energy_frac": float(np.sum(ac * ac)) / max(total, EPS),
            "kernel_dc_rgb_r": float(w[0].mean()),
            "kernel_dc_rgb_g": float(w[1].mean()),
            "kernel_dc_rgb_b": float(w[2].mean()),
            "pos_mean": float(p.mean()),
            "pos_std": float(p.std()),
            "pos_rms": float(np.sqrt(np.mean(p * p))),
            "pos_range": float(p.max() - p.min()),
        })
    df = pd.DataFrame(rows)
    df["dc_rank_desc"] = rank_desc(df["kernel_dc_energy_frac"].to_numpy(float))
    df["weight_rank_low_to_high"] = pd.Series(df["weight_l2"]).rank(method="min").astype(int)
    return df


def positional_prior_alignment(model, prior_vec: np.ndarray) -> pd.DataFrame:
    pe = model.visual.positional_embedding.detach().float().cpu().numpy()[1:]
    rows = []
    for c in range(pe.shape[1]):
        v = pe[:, c]
        rows.append({
            "channel": c,
            "pos_signed_corr_register_prior": pearson(v, prior_vec),
            "pos_abs_corr_register_prior": pearson(np.abs(v), prior_vec),
            "pos_sq_corr_register_prior": pearson(v * v, prior_vec),
        })
    df = pd.DataFrame(rows)
    df["max_abs_prior_corr"] = df[
        ["pos_signed_corr_register_prior", "pos_abs_corr_register_prior", "pos_sq_corr_register_prior"]
    ].abs().max(axis=1)
    df = df.sort_values("max_abs_prior_corr", ascending=False).reset_index(drop=True)
    df["prior_alignment_rank"] = np.arange(1, len(df) + 1)
    return df


def early_head_prior_alignment(
    probe_maps: Dict[int, np.ndarray],
    prior_vec: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for b, A in probe_maps.items():
        base = A[0]  # H,P
        for h in range(base.shape[0]):
            v = base[h] / (base[h].sum() + EPS)
            rows.append({
                "block": int(b),
                "head": int(h),
                "baseline_attn_corr_register_prior": pearson(v, prior_vec),
                "baseline_attn_abs_corr_register_prior": abs(pearson(v, prior_vec)),
            })
    return pd.DataFrame(rows).sort_values("baseline_attn_abs_corr_register_prior", ascending=False)


# ---------------------------------------------------------------------------
# Channel causal factorization
# ---------------------------------------------------------------------------

def distribution_from_samples(df: pd.DataFrame, stage: str, P: int) -> Tuple[np.ndarray, np.ndarray]:
    q = df[df["stage"].eq(stage)]
    arg = np.zeros(P, np.float64)
    reg = np.zeros(P, np.float64)
    for r in q.itertuples(index=False):
        arg[int(r.argmax_patch0)] += 1
        for x in str(r.register_patches0).split("|"):
            if x != "":
                reg[int(x)] += 1
    if len(q):
        arg /= len(q)
        reg /= len(q)
    return arg, reg


def early_map_effects(base: Dict[int, np.ndarray], cur: Dict[int, np.ndarray]) -> List[dict]:
    rows = []
    for b in EARLY_BLOCKS:
        A = base[b]  # H,P
        C = cur[b]
        for h in range(A.shape[0]):
            a = A[h] / (A[h].sum() + EPS)
            c = C[h] / (C[h].sum() + EPS)
            rows.append({
                "block": b,
                "head": h,
                "cosine_distance": 1.0 - cosine(a, c),
                "jsd": js_divergence(a, c),
                "pearson": pearson(a, c),
            })
    return rows


def run_channel_factorization(
    model,
    preprocess,
    specs: Sequence[StimSpec],
    channels: Sequence[int],
    interventions: Sequence[str],
    *,
    size: int,
    G: int,
    patch: int,
    batch_size: int,
    register_threshold: float,
    watch_patch: int,
    static_df: pd.DataFrame,
    out: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, float]]]:
    P = G * G

    base_samples, base_early, base_conv = run_bank(
        model, preprocess, specs,
        size=size, G=G, patch=patch, batch_size=batch_size,
        register_threshold=register_threshold, capture_early=True,
        capture_conv_channels=channels,
    )
    base_arg23, base_reg23 = distribution_from_samples(base_samples, "pre23", P)
    base_arg13, base_reg13 = distribution_from_samples(base_samples, "pre13", P)

    summary_rows = []
    prior_rows = []
    head_rows = []

    for c in channels:
        sm = static_df[static_df["channel"].eq(int(c))]
        if sm.empty:
            raise RuntimeError(f"Channel {c} not in static census")
        for cond in interventions:
            print(f"\n[channel {c}] {cond}")
            with apply_intervention(model, int(c), str(cond)):
                cur_samples, cur_early, _ = run_bank(
                    model, preprocess, specs,
                    size=size, G=G, patch=patch, batch_size=batch_size,
                    register_threshold=register_threshold, capture_early=True,
                    capture_conv_channels=(),
                )

            a23, r23 = distribution_from_samples(cur_samples, "pre23", P)
            a13, r13 = distribution_from_samples(cur_samples, "pre13", P)

            top_base = int(np.argmax(base_arg23))
            top_cur = int(np.argmax(a23))
            rec = {
                "channel": int(c),
                "condition": str(cond),
                "kernel_dc_energy_frac": float(sm.iloc[0]["kernel_dc_energy_frac"]),
                "weight_l2": float(sm.iloc[0]["weight_l2"]),
                "robust_log_weight_z": float(sm.iloc[0]["robust_log_weight_z"]),
                "pos_std": float(sm.iloc[0]["pos_std"]),
                "pre23_argmax_jsd": js_divergence(base_arg23, a23),
                "pre23_register_jsd": js_divergence(base_reg23, r23),
                "pre13_argmax_jsd": js_divergence(base_arg13, a13),
                "pre13_register_jsd": js_divergence(base_reg13, r13),
                "pre23_argmax_prior_pearson": pearson(base_arg23, a23),
                "pre13_argmax_prior_pearson": pearson(base_arg13, a13),
                "baseline_top_patch0": top_base,
                "condition_top_patch0": top_cur,
                "top_patch_changed": int(top_base != top_cur),
                "baseline_watch_patch_argmax_freq": float(base_arg23[watch_patch]) if 0 <= watch_patch < P else np.nan,
                "condition_watch_patch_argmax_freq": float(a23[watch_patch]) if 0 <= watch_patch < P else np.nan,
                "watch_patch_argmax_delta": float(a23[watch_patch] - base_arg23[watch_patch]) if 0 <= watch_patch < P else np.nan,
                "conv1_activation_std_baseline_synth": float(base_conv[int(c)]["activation_std"]),
                "conv1_activation_rms_baseline_synth": float(base_conv[int(c)]["activation_rms"]),
                "pos_std_over_synth_conv1_std": float(sm.iloc[0]["pos_std"]) / max(base_conv[int(c)]["activation_std"], EPS),
            }

            he = early_map_effects(base_early, cur_early)
            for rr in he:
                rr.update({"channel": int(c), "condition": str(cond)})
                head_rows.append(rr)
            rec["early_head_max_cosdist"] = float(max(rr["cosine_distance"] for rr in he))
            rec["early_head_mean_cosdist"] = float(np.mean([rr["cosine_distance"] for rr in he]))
            rec["early_head_max_jsd"] = float(max(rr["jsd"] for rr in he))
            summary_rows.append(rec)

            for stage in ("pre13", "pre23"):
                aa, rg = distribution_from_samples(cur_samples, stage, P)
                for p in range(P):
                    prior_rows.append({
                        "channel": int(c), "condition": str(cond), "stage": stage,
                        "patch_index0": p, "row0": p // G, "col0": p % G,
                        "argmax_frequency": float(aa[p]),
                        "register_frequency": float(rg[p]),
                    })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(prior_rows),
        pd.DataFrame(head_rows),
        base_conv,
    )


# ---------------------------------------------------------------------------
# Pac-Man chase
# ---------------------------------------------------------------------------

def load_optional_font(path: str, size: int):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    return ImageFont.truetype(str(p), size=size)


def read_glyphs(path: str) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip("\n\r")
        if s:
            out.append(s)
    return out


def pseudo_glyph_patch(patch: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (patch, patch), (245, 245, 242))
    d = ImageDraw.Draw(im)
    palette = [(15,15,15), (180,20,60), (20,90,200), (20,150,70), (110,30,170)]
    col = palette[seed % len(palette)]
    w = max(1, patch // 7)
    anchors = [
        (2,2), (patch//2,2), (patch-3,2),
        (2,patch//2), (patch//2,patch//2), (patch-3,patch//2),
        (2,patch-3), (patch//2,patch-3), (patch-3,patch-3),
    ]
    # connected, glyph-like strokes rather than pure noise
    ids = rng.choice(len(anchors), size=5, replace=False)
    for a, b in zip(ids[:-1], ids[1:]):
        d.line((*anchors[int(a)], *anchors[int(b)]), fill=col, width=w)
    if seed % 2:
        d.rectangle((patch//3, patch//3, 2*patch//3, 2*patch//3), outline=(0,0,0), width=1)
    else:
        d.ellipse((patch//3, patch//3, 2*patch//3, 2*patch//3), outline=(0,0,0), width=1)
    return im


def glyph_patch(patch: int, glyph: str, font) -> Image.Image:
    im = Image.new("RGB", (patch, patch), (245, 245, 242))
    d = ImageDraw.Draw(im)
    bbox = d.textbbox((0,0), glyph, font=font)
    tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
    x = (patch - tw)//2 - bbox[0]
    y = (patch - th)//2 - bbox[1]
    d.text((x,y), glyph, font=font, fill=(5,5,5))
    return im


def stamp_patch(canvas: Image.Image, p: int, G: int, patch: int, motif: Image.Image, radius: int) -> None:
    r, c = divmod(p, G)
    for rr in range(max(0, r-radius), min(G, r+radius+1)):
        for cc in range(max(0, c-radius), min(G, c+radius+1)):
            # center gets original motif; neighbors get deterministic rotations
            m = motif
            if radius > 0 and (rr != r or cc != c):
                rot = ((rr-r+1) * 2 + (cc-c+1)) % 4
                m = motif.rotate(90 * rot)
            canvas.paste(m.resize((patch,patch), Image.Resampling.NEAREST), (cc*patch, rr*patch))


def chase_registers(
    model,
    preprocess,
    *,
    size: int,
    G: int,
    patch: int,
    register_threshold: float,
    rounds: int,
    protect_radius: int,
    font_path: str,
    glyph_file: str,
    save_every: int,
    out: Path,
) -> Tuple[pd.DataFrame, dict]:
    canvas = Image.new("RGB", (size, size), (246,246,246))
    font = load_optional_font(font_path, max(8, int(patch * 0.95))) if font_path else None
    glyphs = read_glyphs(glyph_file)
    protected: Dict[int, int] = {}
    collision_sites: List[int] = []
    rows = []
    frame_dir = ensure_dir(out / "chase_frames")

    first_collision = None
    for step in range(rounds):
        batch = preprocess_images(preprocess, [canvas])
        fr = visual_forward_capture(model, batch, capture_stages=(23,), early_blocks=())
        norms = spatial_norms_from_state(fr.pre[23], 1, G*G)[0].numpy()
        regs = np.flatnonzero(norms > register_threshold).astype(int).tolist()
        if not regs:
            regs = np.argsort(-norms)[:4].astype(int).tolist()

        collisions = [p for p in regs if p in protected]
        new_sites = [p for p in regs if p not in protected]
        if collisions and first_collision is None:
            first_collision = step
        collision_sites.extend(collisions)

        rows.append({
            "step": step,
            "register_sites0": "|".join(map(str, regs)),
            "new_sites0": "|".join(map(str, new_sites)),
            "collision_sites0": "|".join(map(str, collisions)),
            "n_protected_before": len(protected),
            "max_norm": float(norms.max()),
            "argmax_patch0": int(np.argmax(norms)),
        })

        if save_every > 0 and (step % save_every == 0 or step == rounds - 1):
            canvas.save(frame_dir / f"step_{step:03d}.png")

        if not new_sites:
            print(f"[chase] allocator has no new unprotected site at step {step}; stopping")
            break

        for p in new_sites:
            idx = len(protected)
            if glyphs and idx < len(glyphs) and font is not None:
                motif = glyph_patch(patch, glyphs[idx], font)
            else:
                motif = pseudo_glyph_patch(patch, stable_seed(f"chase:{idx}:{p}"))
            stamp_patch(canvas, p, G, patch, motif, protect_radius)
            protected[p] = step

        print(f"[chase] step={step} regs={regs} new={new_sites} collisions={collisions} protected={len(protected)}")
        if len(protected) >= G * G:
            break

    trace = pd.DataFrame(rows)
    summary = {
        "n_protected": len(protected),
        "first_collision_step": first_collision,
        "n_collision_events": len(collision_sites),
        "unique_collision_sites": sorted(set(collision_sites)),
        "protection_order": {str(k): int(v) for k, v in protected.items()},
    }
    canvas.save(out / "chase_final_canvas.png")
    return trace, summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heat(ax, vec: np.ndarray, G: int, title: str):
    im = ax.imshow(np.asarray(vec).reshape(G, G), interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    return im


def plot_register_prior(prior: pd.DataFrame, G: int, out: Path):
    allq = prior[prior["family"].eq("all")]
    stages = ["pre13", "pre23", "post23"]
    fig, axes = plt.subplots(2, len(stages), figsize=(4.2*len(stages), 8.0))
    for j, stage in enumerate(stages):
        q = allq[allq["stage"].eq(stage)].sort_values("patch_index0")
        if len(q) != G*G:
            axes[0,j].axis("off"); axes[1,j].axis("off"); continue
        im = plot_heat(axes[0,j], q["argmax_frequency"].to_numpy(), G, f"{stage}: top-norm site frequency")
        fig.colorbar(im, ax=axes[0,j], fraction=.045, pad=.03)
        im = plot_heat(axes[1,j], q["register_frequency"].to_numpy(), G, f"{stage}: norm>{q['n_images'].iloc[0]*0+0:g} site frequency")
        # overwrite title with neutral threshold wording; threshold printed in report/config
        axes[1,j].set_title(f"{stage}: threshold-register frequency")
        fig.colorbar(im, ax=axes[1,j], fraction=.045, pad=.03)
    fig.suptitle("Register-address prior on stationary / mathematical images")
    fig.tight_layout()
    fig.savefig(out / "01_REGISTER_ATTRACTOR_PRIOR.png", dpi=220)
    plt.close(fig)


def plot_family_prior(prior: pd.DataFrame, G: int, out: Path):
    fams = [x for x in sorted(prior["family"].unique()) if x != "all"]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.6*len(fams), 3.8), squeeze=False)
    for j, fam in enumerate(fams):
        q = prior[(prior["family"].eq(fam)) & (prior["stage"].eq("pre23"))].sort_values("patch_index0")
        im = plot_heat(axes[0,j], q["argmax_frequency"].to_numpy(), G, fam)
        fig.colorbar(im, ax=axes[0,j], fraction=.045, pad=.03)
    fig.suptitle("Pre-B23 top-norm position prior by synthetic family")
    fig.tight_layout()
    fig.savefig(out / "02_REGISTER_PRIOR_BY_FAMILY.png", dpi=220)
    plt.close(fig)


def plot_head_scatter(metrics: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    for label, marker in [
        ("visual_tracker", "o"),
        ("rigid_or_gated_positional", "s"),
        ("hybrid", "^"),
        ("uniform_or_weak", "x"),
    ]:
        q = metrics[metrics["phenotype"].eq(label)]
        if len(q):
            ax.scatter(q["tracking_score"], q["baseline_spatial_cv"], s=50, marker=marker, label=label, alpha=.8)
    for r in metrics.nlargest(8, "rigid_score").itertuples(index=False):
        ax.annotate(f"B{r.block}H{r.head}", (r.tracking_score, r.baseline_spatial_cv), xytext=(3,3), textcoords="offset points", fontsize=8)
    for r in metrics.nlargest(8, "tracker_score").itertuples(index=False):
        ax.annotate(f"B{r.block}H{r.head}", (r.tracking_score, r.baseline_spatial_cv), xytext=(3,-10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("target tracking score (COM row/col correlation)")
    ax.set_ylabel("blank-canvas spatial CV")
    ax.set_title("Early CLS->patch heads: movable visual tracking vs rigid spatial fields")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "03_EARLY_HEAD_RIGID_VS_TRACKING.png", dpi=220)
    plt.close(fig)


def plot_head_contact(metrics: pd.DataFrame, maps: Dict[int, np.ndarray], pos: List[Tuple[int,int]], G: int, out: Path):
    rigid = metrics.nlargest(4, "rigid_score")[["block","head"]].itertuples(index=False)
    tracker = metrics.nlargest(4, "tracker_score")[["block","head"]].itertuples(index=False)
    chosen = [("rigid", int(r.block), int(r.head)) for r in rigid] + [("tracker", int(r.block), int(r.head)) for r in tracker]
    if not chosen:
        return
    # blank + 4 spread placements
    idxs = [0]
    if pos:
        picks = np.linspace(0, len(pos)-1, min(4, len(pos))).round().astype(int).tolist()
        idxs += [1+i for i in picks]
    fig, axes = plt.subplots(len(chosen), len(idxs), figsize=(3.0*len(idxs), 2.7*len(chosen)), squeeze=False)
    for ri, (lab,b,h) in enumerate(chosen):
        A = maps[b][:,h]
        for cj, ii in enumerate(idxs):
            ax = axes[ri,cj]
            z = A[ii].reshape(G,G)
            ax.imshow(z, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if cj == 0:
                ax.set_ylabel(f"{lab}\nB{b}H{h}")
                ax.set_title("blank")
            else:
                pr, pc = pos[ii-1]
                ax.scatter([pc],[pr], marker="x", s=55)
                ax.set_title(f"probe ({pr},{pc})")
    fig.suptitle("Early-head response sheets: fixed global fields vs movable visual attention")
    fig.tight_layout()
    fig.savefig(out / "04_EARLY_HEAD_CONTACT_SHEET.png", dpi=190)
    plt.close(fig)


def plot_dc_census(census: pd.DataFrame, channels: Sequence[int], out: Path):
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    ax.scatter(census["robust_log_weight_z"], census["kernel_dc_energy_frac"], s=16, alpha=.45)
    q = census[census["channel"].isin(channels)]
    ax.scatter(q["robust_log_weight_z"], q["kernel_dc_energy_frac"], s=65)
    for r in q.itertuples(index=False):
        ax.annotate(str(int(r.channel)), (r.robust_log_weight_z, r.kernel_dc_energy_frac), xytext=(3,3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("robust log Conv1 weight-norm z")
    ax.set_ylabel("within-kernel DC energy fraction")
    ax.set_title("Conv1 morphology census: tiny filters are not automatically DC filters")
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(out / "05_CONV1_DC_CENSUS.png", dpi=220)
    plt.close(fig)


def plot_channel_causal(summary: pd.DataFrame, out: Path):
    if summary.empty:
        return
    piv = summary.pivot(index="channel", columns="condition", values="pre23_argmax_jsd")
    fig, ax = plt.subplots(figsize=(max(9, .85*len(piv.columns)+4), max(4, .45*len(piv)+2)))
    im = ax.imshow(piv.to_numpy(float), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"ch{x}" for x in piv.index])
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=35, ha="right")
    ax.set_title("Causal change in pre-B23 register-site prior (Jensen-Shannon divergence)")
    fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
    fig.tight_layout()
    fig.savefig(out / "06_CHANNEL_REGISTER_PRIOR_CAUSALITY.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for cond in sorted(summary["condition"].unique()):
        q = summary[summary["condition"].eq(cond)]
        ax.scatter(q["kernel_dc_energy_frac"], q["pre23_argmax_jsd"], label=cond, s=45, alpha=.8)
    for r in summary.sort_values("pre23_argmax_jsd", ascending=False).head(8).itertuples(index=False):
        ax.annotate(f"{r.channel}:{r.condition}", (r.kernel_dc_energy_frac, r.pre23_argmax_jsd), xytext=(3,3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("native Conv1 kernel DC energy fraction")
    ax.set_ylabel("pre-B23 register-prior JSD")
    ax.set_title("Does near-DC Conv1 morphology predict allocator leverage?")
    ax.grid(alpha=.15)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "07_DC_FRACTION_VS_ALLOCATOR_EFFECT.png", dpi=220)
    plt.close(fig)


def plot_chase(trace: pd.DataFrame, summary: dict, G: int, out: Path):
    order = np.full(G*G, np.nan)
    for k,v in summary.get("protection_order", {}).items():
        order[int(k)] = int(v)
    fig, ax = plt.subplots(figsize=(6.4,5.8))
    im = ax.imshow(order.reshape(G,G), interpolation="nearest")
    col = summary.get("unique_collision_sites", [])
    if col:
        rr = [p//G for p in col]; cc = [p%G for p in col]
        ax.scatter(cc, rr, marker="x", s=80, linewidths=2)
    ax.set_title("Pac-Man chase: first round each patch was protected\nx = allocator later reused a protected site")
    ax.set_xlabel("col"); ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, label="first protection round")
    fig.tight_layout()
    fig.savefig(out / "08_REGISTER_PACMAN_CHASE.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report / compact handoff
# ---------------------------------------------------------------------------

def make_report(
    *,
    out: Path,
    G: int,
    register_threshold: float,
    watch_patch: int,
    prior: pd.DataFrame,
    head_metrics: pd.DataFrame,
    head_prior: pd.DataFrame,
    pos_align: pd.DataFrame,
    causal: pd.DataFrame,
    chase_summary: Optional[dict],
):
    P = G*G
    lines = [
        "# Conv1 / positional register-allocator tomography",
        "",
        f"Grid: {G}x{G} ({P} spatial patches).",
        f"Register norm threshold used for threshold-frequency maps: {register_threshold:g}.",
        "",
        "## Stationary-input register prior",
    ]
    q = prior[(prior["family"].eq("all")) & (prior["stage"].eq("pre23"))].sort_values("argmax_frequency", ascending=False)
    for r in q.head(12).itertuples(index=False):
        lines.append(
            f"- patch0={int(r.patch_index0)} (patch1={int(r.patch_index1)}, row={int(r.row0)}, col={int(r.col0)}): "
            f"top-norm freq={r.argmax_frequency:.4f}, threshold-reg freq={r.register_frequency:.4f}"
        )
    if 0 <= watch_patch < P:
        w = q[q["patch_index0"].eq(watch_patch)]
        if len(w):
            rank = int((q["argmax_frequency"].to_numpy() > float(w.iloc[0]["argmax_frequency"])).sum() + 1)
            lines += [
                "",
                f"Watch patch {watch_patch} (0-based) rank by pre-B23 top-norm frequency: {rank}/{P}; "
                f"frequency={float(w.iloc[0]['argmax_frequency']):.4f}.",
            ]

    lines += ["", "## Early head phenotypes", "Top rigid/gated candidates:"]
    for r in head_metrics.nlargest(8, "rigid_score").itertuples(index=False):
        lines.append(
            f"- B{r.block}H{r.head}: phenotype={r.phenotype}, rigid={r.rigid_score:.3f}, "
            f"tracking={r.tracking_score:.3f}, capture={r.capture_rate_r1:.3f}, support-gap={r.support_capture_gap:.3f}"
        )
    lines += ["", "Top movable visual trackers:"]
    for r in head_metrics.nlargest(8, "tracker_score").itertuples(index=False):
        lines.append(
            f"- B{r.block}H{r.head}: phenotype={r.phenotype}, tracker={r.tracker_score:.3f}, "
            f"tracking={r.tracking_score:.3f}, capture={r.capture_rate_r1:.3f}"
        )

    lines += ["", "Early heads whose blank-canvas field best matches the late register prior:"]
    for r in head_prior.head(8).itertuples(index=False):
        lines.append(f"- B{r.block}H{r.head}: |r|={r.baseline_attn_abs_corr_register_prior:.3f}")

    lines += ["", "Positional coordinates most aligned with the stationary register prior:"]
    for r in pos_align.head(10).itertuples(index=False):
        lines.append(
            f"- ch{int(r.channel)}: max |corr|={r.max_abs_prior_corr:.3f} "
            f"(signed={r.pos_signed_corr_register_prior:+.3f}, abs={r.pos_abs_corr_register_prior:+.3f})"
        )

    if not causal.empty:
        lines += ["", "## Conv1-vs-position causal factorization", "Largest pre-B23 register-prior changes:"]
        for r in causal.sort_values("pre23_argmax_jsd", ascending=False).head(16).itertuples(index=False):
            lines.append(
                f"- ch{r.channel} / {r.condition}: JSD={r.pre23_argmax_jsd:.5f}, "
                f"early-max-cosdist={r.early_head_max_cosdist:.5f}, DC-energy={r.kernel_dc_energy_frac:.4f}"
            )

    if chase_summary is not None:
        lines += [
            "",
            "## Pac-Man chase",
            f"- protected unique patches: {chase_summary.get('n_protected')}",
            f"- first register reuse/collision round: {chase_summary.get('first_collision_step')}",
            f"- collision events: {chase_summary.get('n_collision_events')}",
        ]

    lines += [
        "",
        "## Interpretation guardrails",
        "- The early-head phenotype labels are descriptive summaries of movable-probe behavior, not semantic head names.",
        "- A Fourier DC bin is within-patch frequency and is not a global patch-grid location.",
        "- Register-address preference is measured behavior; 'allocator' / 'sacrifice cost' is a mechanistic hypothesis to be tested by the interventions.",
        "- KERNEL_DC_ONLY/AC_ONLY manipulate the Conv1 WEIGHT ROW within each 14x14 patch, unlike the older activation-map DC_ONLY/CENTERED manipulations across patch locations.",
        "- Positional-coordinate interventions leave the CLS positional value untouched and modify only spatial patch positions.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_zip(out: Path) -> Path:
    zpath = out / "compact_summary_conv1_register_allocator_tomography.zip"
    include = [
        "config.json", "model_audit.json", "REPORT.md",
        "conv1_dc_census.csv",
        "register_position_prior.csv",
        "register_family_stability.csv",
        "positional_coordinate_prior_alignment.csv",
        "early_head_probe_metrics.csv",
        "early_head_register_prior_alignment.csv",
        "channel_static_metrics.csv",
        "channel_causal_summary.csv",
        "chase_trace.csv", "chase_summary.json",
    ]
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in include:
            p = out / name
            if p.is_file():
                z.write(p, arcname=name)
        plotdir = out / "plots"
        if plotdir.is_dir():
            for p in sorted(plotdir.glob("*.png")):
                z.write(p, arcname=f"plots/{p.name}")
    return zpath


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def self_test() -> None:
    seed_all(123)

    # JSD
    p = np.array([.5,.5,0,0])
    assert js_divergence(p,p) < 1e-10
    assert js_divergence(p,np.array([0,0,.5,.5])) > .5

    # Synthetic generator exact geometry
    spec = StimSpec("x", "tiled_patch", 1)
    im = render_stationary(spec, 224, 14)
    assert im.size == (224,224)

    # Dummy Conv1 intervention restores exactly.
    class V:
        pass
    class M:
        pass
    m = M(); m.visual = V()
    m.visual.conv1 = torch.nn.Conv2d(3, 8, 4, stride=4, bias=False)
    orig = m.visual.conv1.weight.detach().clone()
    with Conv1KernelIntervention(m, 3, "KERNEL_DC_ONLY"):
        w = m.visual.conv1.weight[3].detach()
        assert torch.allclose(w, w.mean(dim=(-2,-1), keepdim=True).expand_as(w))
    assert torch.equal(m.visual.conv1.weight, orig)

    # Dummy positional intervention leaves CLS value intact and restores.
    m.visual.positional_embedding = torch.nn.Parameter(torch.randn(17,8))
    pe0 = m.visual.positional_embedding.detach().clone()
    with PositionalCoordinateIntervention(m, 2, "POS_ZERO", 1):
        assert torch.all(m.visual.positional_embedding[1:,2] == 0)
        assert torch.equal(m.visual.positional_embedding[0,2], pe0[0,2])
    assert torch.equal(m.visual.positional_embedding, pe0)

    # Probe positions all inside border.
    pp = probe_positions(16,5)
    assert pp and all(1 <= r <= 14 and 1 <= c <= 14 for r,c in pp)

    print("self-test OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Conv1 / positional register allocator tomography")
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_per_family", type=int, default=64,
                    help="Stationary prior images per synthetic family; 5 families total.")
    ap.add_argument("--causal_n_per_family", type=int, default=16,
                    help="Subset per family used for repeated channel interventions.")
    ap.add_argument("--register_threshold", type=float, default=60.0)
    ap.add_argument("--watch_patch", type=int, default=45,
                    help="0-based spatial patch index to explicitly report.")
    ap.add_argument("--probe_axis_positions", type=int, default=5,
                    help="Moving-target grid positions per axis (5 => ~25 probe placements).")
    ap.add_argument("--channels", default=DEFAULT_CHANNELS)
    ap.add_argument("--interventions", default=DEFAULT_INTERVENTIONS)
    ap.add_argument("--run_channel_causality", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--run_chase", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--chase_rounds", type=int, default=64)
    ap.add_argument("--chase_protect_radius", type=int, default=0,
                    help="0 protects only selected patch; 1 stamps a 3x3 patch neighborhood.")
    ap.add_argument("--chase_font_path", default="")
    ap.add_argument("--chase_glyph_file", default="",
                    help="Optional UTF-8 file, one unique glyph/string per line. Requires --chase_font_path.")
    ap.add_argument("--save_chase_frames_every", type=int, default=8)
    ap.add_argument("--self_test", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    seed_all(args.seed)
    out = ensure_dir(Path(args.out_dir))
    plotdir = ensure_dir(out / "plots")

    model, preprocess = load_model(args.clip_model, args.device)
    image_size, G, patch, width = model_geometry(model)
    P = G*G
    channels = tuple(dict.fromkeys(parse_ints(args.channels)))
    interventions = parse_strs(args.interventions)
    for c in channels:
        if not (0 <= c < width):
            raise ValueError(f"Channel {c} outside width={width}")

    parity = validate_manual_forward(model, preprocess, image_size, args.device)
    print(f"[manual forward parity] {parity}")

    config = vars(args).copy()
    config.update({
        "image_size": image_size, "grid": G, "patch_size": patch, "width": width,
        "n_spatial_tokens": P,
        "channels_resolved": list(channels),
        "interventions_resolved": list(interventions),
    })
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "model_audit.json").write_text(json.dumps({
        "clip_model": args.clip_model,
        "visual_blocks": len(model.visual.transformer.resblocks),
        "grid": G, "patch_size": patch, "width": width,
        "conv1_dtype": str(model.visual.conv1.weight.dtype),
        "device": str(next(model.parameters()).device),
        "manual_forward_parity": parity,
    }, indent=2), encoding="utf-8")

    # Static all-channel census.
    census = conv1_positional_census(model, G)
    census.to_csv(out / "conv1_dc_census.csv", index=False)
    plot_dc_census(census, channels, plotdir)

    # Stationary prior.
    specs = make_stationary_specs(args.n_per_family, args.seed)
    samples, stationary_early, _ = run_bank(
        model, preprocess, specs,
        size=image_size, G=G, patch=patch, batch_size=args.batch_size,
        register_threshold=args.register_threshold,
        capture_early=True, capture_conv_channels=(),
    )
    samples.to_csv(out / "register_samples.csv.gz", index=False, compression="gzip")

    # Add pooled "all" family rows before aggregating.
    pooled = samples.copy()
    pooled["family"] = "all"
    prior = aggregate_prior(pd.concat([samples, pooled], ignore_index=True), G)
    prior.to_csv(out / "register_position_prior.csv", index=False)
    stab = family_stability(prior, "pre23", P)
    stab.to_csv(out / "register_family_stability.csv", index=False)
    plot_register_prior(prior, G, plotdir)
    plot_family_prior(prior, G, plotdir)

    # Moving probe: distinguish rigid/gated positional fields from image trackers.
    head_metrics, probe_maps, positions = moving_probe_analysis(
        model, preprocess,
        size=image_size, G=G, patch=patch,
        n_axis=args.probe_axis_positions,
        batch_size=args.batch_size,
        out=out,
    )
    head_metrics.to_csv(out / "early_head_probe_metrics.csv", index=False)
    plot_head_scatter(head_metrics, plotdir)
    plot_head_contact(head_metrics, probe_maps, positions, G, plotdir)

    # Align native positional coordinates and early blank-canvas fields to the empirical prior.
    prior23 = register_prior_stage_vector(prior, "pre23", "all", "argmax_frequency", P)
    pos_align = positional_prior_alignment(model, prior23)
    pos_align.to_csv(out / "positional_coordinate_prior_alignment.csv", index=False)
    head_prior = early_head_prior_alignment(probe_maps, prior23)
    head_prior.to_csv(out / "early_head_register_prior_alignment.csv", index=False)

    # Merge static + prior-alignment metrics for selected channels.
    static_sel = census[census["channel"].isin(channels)].merge(
        pos_align.drop(columns=["prior_alignment_rank"], errors="ignore"), on="channel", how="left"
    )
    static_sel.to_csv(out / "channel_static_metrics.csv", index=False)

    causal = pd.DataFrame()
    causal_prior = pd.DataFrame()
    causal_heads = pd.DataFrame()
    if args.run_channel_causality and channels:
        # deterministic balanced subset by family
        causal_specs = []
        byfam: Dict[str, int] = {}
        for s in specs:
            n = byfam.get(s.family, 0)
            if n < args.causal_n_per_family:
                causal_specs.append(s)
                byfam[s.family] = n + 1
        causal, causal_prior, causal_heads, synth_conv = run_channel_factorization(
            model, preprocess, causal_specs, channels, interventions,
            size=image_size, G=G, patch=patch, batch_size=args.batch_size,
            register_threshold=args.register_threshold,
            watch_patch=args.watch_patch,
            static_df=census,
            out=out,
        )
        causal.to_csv(out / "channel_causal_summary.csv", index=False)
        causal_prior.to_csv(out / "channel_register_prior.csv.gz", index=False, compression="gzip")
        causal_heads.to_csv(out / "channel_early_head_effects.csv.gz", index=False, compression="gzip")
        plot_channel_causal(causal, plotdir)

    chase_trace = pd.DataFrame()
    chase_summary = None
    if args.run_chase:
        chase_trace, chase_summary = chase_registers(
            model, preprocess,
            size=image_size, G=G, patch=patch,
            register_threshold=args.register_threshold,
            rounds=args.chase_rounds,
            protect_radius=args.chase_protect_radius,
            font_path=args.chase_font_path,
            glyph_file=args.chase_glyph_file,
            save_every=args.save_chase_frames_every,
            out=out,
        )
        chase_trace.to_csv(out / "chase_trace.csv", index=False)
        (out / "chase_summary.json").write_text(json.dumps(chase_summary, indent=2), encoding="utf-8")
        plot_chase(chase_trace, chase_summary, G, plotdir)

    make_report(
        out=out, G=G, register_threshold=args.register_threshold,
        watch_patch=args.watch_patch, prior=prior,
        head_metrics=head_metrics, head_prior=head_prior,
        pos_align=pos_align, causal=causal,
        chase_summary=chase_summary,
    )

    z = compact_zip(out)
    print(f"\nDONE\nReport: {out/'REPORT.md'}\nCompact handoff: {z}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
