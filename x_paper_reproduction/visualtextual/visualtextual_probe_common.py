#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib, gc, hashlib, json, math, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

EPS = 1e-12
IMAGE_RE = re.compile(r"^(vis|txt|mix)_(.+?)(_bw)?$", re.IGNORECASE)
EXCLUDED_CONCEPTS = {"car", "mus"}

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") & 0x7FFFFFFF

def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cosine_np(a, b) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > EPS else float("nan")

def pearson(a, b) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > EPS else float("nan")

def normalize_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), EPS)

def robust_z_rows(x: torch.Tensor) -> torch.Tensor:
    med = x.median(dim=-1, keepdim=True).values
    mad = (x - med).abs().median(dim=-1, keepdim=True).values
    scale = 1.4826 * mad
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    scale = torch.where(scale > 1e-6, scale, std.clamp_min(1e-6))
    return (x - med) / scale

def parse_models(s: str) -> List[str]:
    out = [x.strip() for x in str(s).split(",") if x.strip()]
    valid = {"pretrained", "gmp", "full_xattn"}
    bad = [x for x in out if x not in valid]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}; valid={sorted(valid)}")
    return out

def amp_context(device: str, enabled: bool = True):
    if enabled and str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()

# ---------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------

def parse_dataset(root: Path) -> pd.DataFrame:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for p in sorted(root.rglob("*.png")):
        m = IMAGE_RE.match(p.stem)
        if not m:
            continue
        condition = m.group(1).lower()
        literal_label = m.group(2)
        bw = bool(m.group(3))
        if literal_label.lower() in EXCLUDED_CONCEPTS:
            continue
        rows.append({
            "path": str(p.resolve()),
            "filename": p.name,
            "condition": condition,
            "bw": bw,
            "category": f"{condition}_{'bw' if bw else 'rgb'}",
            "concept": literal_label,
            "prompt": literal_label.replace("_", " "),
            "stim_id": p.stem,
        })
    d = pd.DataFrame(rows)
    if d.empty:
        raise FileNotFoundError(f"No vis_*/txt_*/mix_*.png files under {root}")
    dup = d.groupby(["condition", "concept", "bw"]).size()
    if (dup > 1).any():
        raise RuntimeError(f"Duplicate condition/concept/bw cells: {dup[dup>1].to_dict()}")
    return d.sort_values(["concept", "bw", "condition"]).reset_index(drop=True)

def complete_concepts(manifest: pd.DataFrame, bw: bool) -> List[str]:
    q = manifest[manifest["bw"].eq(bool(bw))]
    need = {"vis", "txt", "mix"}
    out = []
    for c, g in q.groupby("concept"):
        if set(g["condition"]) >= need:
            out.append(str(c))
    return sorted(out)

def manifest_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for concept, g in manifest.groupby("concept"):
        row = {"concept": concept}
        for bw in (False, True):
            for cond in ("vis", "txt", "mix"):
                row[f"{cond}_{'bw' if bw else 'rgb'}"] = int(((g.condition == cond) & (g.bw == bw)).sum())
        row["complete_rgb"] = int(all(row[f"{c}_rgb"] == 1 for c in ("vis","txt","mix")))
        row["complete_bw"] = int(all(row[f"{c}_bw"] == 1 for c in ("vis","txt","mix")))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("concept").reset_index(drop=True)

# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def find_repo_root(start: Optional[Path] = None) -> Path:
    starts = []
    if start is not None:
        starts.append(Path(start).resolve())
    starts += [Path.cwd().resolve(), Path(__file__).resolve().parent]
    checked = set()
    for root0 in starts:
        for p in [root0, *root0.parents]:
            if p in checked:
                continue
            checked.add(p)
            if (
                (p / "attnclip_mechinterp_sae").is_dir()
                and (p / "attnclip_mechinterp_xattn").is_dir()
                and (p / "utils_clip_loader").is_dir()
            ):
                return p
    raise FileNotFoundError("Could not locate repo root; pass --repo_root.")

def import_runtime(repo_root: Optional[str] = None):
    repo = find_repo_root(Path(repo_root)) if repo_root else find_repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import attnclip_mechinterp_sae as clip_sae
    import attnclip_mechinterp_xattn as clip_x
    from utils_clip_loader.clip_anything_to_openai import (
        load_openai_clip_anything,
        resolve_to_openai_state_dict,
    )
    return repo, clip_sae, clip_x, load_openai_clip_anything, resolve_to_openai_state_dict

def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def _load_gmp(clip_sae, load_any, checkpoint: str, device: str):
    try:
        m, p, li = load_any(
            clip_sae, checkpoint, device=device, jit=False, strict=True,
            reuse_full_model_pickle=False,
        )
        return freeze_model(m), p, {"source": checkpoint, "mode": "state_dict_rebuild", "loader": str(li)}
    except Exception as first_error:
        src, _pp, li = load_any(
            clip_sae, checkpoint, device="cpu", jit=False, strict=True,
            reuse_full_model_pickle=True,
        )
        m, p, _ = load_any(clip_sae, "openai/clip-vit-large-patch14", device=device, jit=False, strict=True, allow_unsafe_hf_pickle=False)
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
        return freeze_model(m), p, {
            "source": checkpoint,
            "mode": "trusted_pickle_visual_transplant",
            "loader": str(li),
            "first_error": repr(first_error),
        }

@dataclass
class Variant:
    name: str
    model: Any
    preprocess: Any
    clip_module: Any
    source_info: Dict[str, Any]
    is_full_xattn: bool = False

    @property
    def visual(self):
        return self.model.visual

    def maybe_insert_rn(self, block_idx: int, x: torch.Tensor) -> torch.Tensor:
        if self.is_full_xattn:
            return self.visual._maybe_insert_read_null(block_idx, x)
        return x

def load_variant(name: str, args, runtime=None) -> Variant:
    if runtime is None:
        runtime = import_runtime(args.repo_root)
    _repo, clip_sae, clip_x, load_any, _resolve = runtime
    if name == "pretrained":
        m, p, li = load_any(clip_sae, args.pretrained_model, device=args.device, jit=False, strict=True, allow_unsafe_hf_pickle=False)
        return Variant(name, freeze_model(m), p, clip_sae, {"source": args.pretrained_model, "loader": str(li)}, False)
    if name == "gmp":
        m, p, info = _load_gmp(clip_sae, load_any, args.gmp_checkpoint, args.device)
        return Variant(name, m, p, clip_sae, info, False)
    if name == "full_xattn":
        m, p, li = load_any(
            clip_x, args.xattn_model, device=args.device, jit=False,
            cache_dir=(args.hf_cache_dir or None),
            revision=(args.xattn_revision or None),
            strict=True, allow_unsafe_hf_pickle=False,
        )
        return Variant(
            name, freeze_model(m), p, clip_x,
            {"source": args.xattn_model, "mode": "full_xattn", "loader": str(li)},
            True,
        )
    raise ValueError(name)

# ---------------------------------------------------------------------
# Visual forward and role geometry
# ---------------------------------------------------------------------

def model_geometry(variant: Variant) -> Tuple[int,int,int,int,int,int]:
    v = variant.visual
    P = int(v.positional_embedding.shape[0] - 1)
    G = int(round(math.sqrt(P)))
    if G * G != P:
        raise RuntimeError(f"Non-square patch grid: {P}")
    ks = v.conv1.kernel_size
    patch = int(ks[0] if isinstance(ks, tuple) else ks)
    size = G * patch
    width = int(v.positional_embedding.shape[1])
    blocks = len(v.transformer.resblocks)
    heads = int(v.transformer.resblocks[0].attn.num_heads)
    return size, G, patch, width, blocks, heads

def preprocess_rows(variant: Variant, rows: pd.DataFrame) -> torch.Tensor:
    ims = [Image.open(p).convert("RGB") for p in rows["path"]]
    return torch.stack([variant.preprocess(im) for im in ims], dim=0)

def prepare_tokens(variant: Variant, images_cpu: torch.Tensor, conv1_intervention: Optional[Tuple[int,str]] = None):
    model = variant.model
    v = variant.visual
    device = next(model.parameters()).device
    dtype = v.conv1.weight.dtype
    images = images_cpu.to(device=device, dtype=dtype)
    conv = v.conv1(images)
    if conv1_intervention is not None:
        channel, mode = conv1_intervention
        if mode == "ZERO":
            conv[:, int(channel)] = 0
        elif mode == "FLIP":
            conv[:, int(channel)] = -conv[:, int(channel)]
        else:
            raise ValueError(mode)
    B,C,G,_ = conv.shape
    x = conv.reshape(B,C,G*G).permute(0,2,1)
    cls = v.class_embedding.to(x.dtype) + torch.zeros((B,1,C), device=x.device, dtype=x.dtype)
    x = torch.cat([cls,x], dim=1)
    x = x + v.positional_embedding.to(x.dtype)
    x = v.ln_pre(x)
    return x.permute(1,0,2), conv

def visual_final_embedding(variant: Variant, x_lbd: torch.Tensor) -> torch.Tensor:
    v = variant.visual
    x = x_lbd.permute(1,0,2)
    cls = v.ln_post(x[:,0,:])
    if v.proj is not None:
        cls = cls @ v.proj
    return cls

@torch.inference_mode()
def manual_forward(
    variant: Variant,
    images_cpu: torch.Tensor,
    *,
    capture_blocks: Sequence[int] = (),
    capture_post: bool = True,
    conv1_intervention: Optional[Tuple[int,str]] = None,
    amp: bool = True,
):
    with amp_context(str(next(variant.model.parameters()).device), amp):
        x, conv = prepare_tokens(variant, images_cpu, conv1_intervention)
        pre = {}
        for bi, block in enumerate(variant.visual.transformer.resblocks):
            x = variant.maybe_insert_rn(bi, x)
            if bi in set(capture_blocks):
                pre[int(bi)] = x.detach().float().cpu()
            x = block(x)
        emb = visual_final_embedding(variant, x).detach().float().cpu()
        post = x.detach().float().cpu() if capture_post else None
        return {"embedding": emb, "pre": pre, "post": post, "conv": conv.detach().float().cpu()}

@torch.inference_mode()
def official_embeddings(variant: Variant, images_cpu: torch.Tensor, amp: bool = True):
    device = next(variant.model.parameters()).device
    images = images_cpu.to(device)
    with amp_context(str(device), amp):
        if variant.is_full_xattn:
            info = variant.model.encode_image_states(images, return_final_tokens=False)
            backbone = info["image_embedding"].float()
            content = variant.model._content_image_from_info(info, apply_content_correction=True).float()
            telemetry = {}
            try:
                source_logits, _glyph, source_stats = variant.model.read_implant.source_outputs(
                    info["states"], return_details=False
                )
                source_probs = source_logits.sigmoid()
                telemetry = {
                    "source_present_prob": source_probs[:,0].float().cpu().numpy(),
                    "source_readable_prob": source_probs[:,1].float().cpu().numpy(),
                    "source_gate": (source_probs[:,0]*source_probs[:,1]).float().cpu().numpy(),
                    "glyph_mean": source_stats[:,0].float().cpu().numpy(),
                    "glyph_max": source_stats[:,1].float().cpu().numpy(),
                }
            except Exception:
                telemetry = {}
            return {
                "backbone": backbone.cpu(),
                "content": content.cpu(),
                "telemetry": telemetry,
            }
        emb = variant.model.encode_image(images).float().cpu()
        return {"backbone": emb, "telemetry": {}}

@torch.inference_mode()
def encode_prompts(variant: Variant, prompts: Sequence[str], amp: bool = True) -> torch.Tensor:
    tok = variant.clip_module.tokenize(list(prompts)).to(next(variant.model.parameters()).device)
    with amp_context(str(next(variant.model.parameters()).device), amp):
        e = variant.model.encode_text(tok).float()
    return F.normalize(e, dim=-1).cpu()

def validate_manual_backbone(variant: Variant, images_cpu: torch.Tensor, amp: bool = True) -> dict:
    a = manual_forward(variant, images_cpu[:1], capture_blocks=(), capture_post=False, amp=amp)["embedding"]
    b = official_embeddings(variant, images_cpu[:1], amp=amp)["backbone"]
    cos = float(F.cosine_similarity(a.float(), b.float(), dim=-1).item())
    rel = float((a-b).norm().item() / max(float(b.norm().item()), EPS))
    if cos < 0.9999 or rel > 2e-3:
        raise RuntimeError(f"{variant.name}: manual/official backbone mismatch cos={cos:.8f} rel={rel:.3e}")
    return {"cosine": cos, "relative_l2": rel, "max_abs": float((a-b).abs().max())}

def spatial_tokens(state_lbd: torch.Tensor, P: int) -> torch.Tensor:
    x = state_lbd.permute(1,0,2).float()
    return x[:,1:1+P,:]

def collect_register_means(
    variant: Variant,
    manifest: pd.DataFrame,
    batch_size: int,
    threshold: float,
    amp: bool = True,
):
    size,G,patch,width,nblocks,nheads = model_geometry(variant)
    P = G*G
    vecs, meta = [], []
    for st in range(0, len(manifest), batch_size):
        q = manifest.iloc[st:st+batch_size]
        batch = preprocess_rows(variant, q)
        fr = manual_forward(variant, batch, capture_blocks=(), capture_post=True, amp=amp)
        sp = spatial_tokens(fr["post"], P)
        norms = sp.norm(dim=-1)
        for i, row in enumerate(q.itertuples(index=False)):
            mask = norms[i] > threshold
            if not bool(mask.any()):
                mask = torch.zeros_like(mask, dtype=torch.bool)
                mask[int(norms[i].argmax())] = True
            mean = sp[i,mask].mean(dim=0)
            vecs.append(mean)
            meta.append({"stim_id": row.stim_id, "n_registers": int(mask.sum()), "max_norm": float(norms[i].max())})
        print(f"[{variant.name} mu calibration] {min(st+len(q),len(manifest))}/{len(manifest)}")
    return torch.stack(vecs).float(), pd.DataFrame(meta)

def discover_mu_from_image_means(X: torch.Tensor):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xd = X.to(dev).float()
    _u, s, vh = torch.linalg.svd(xd, full_matrices=False)
    mu1 = vh[0].float()
    mu2 = vh[1].float()
    if torch.dot(mu1, xd.mean(dim=0)) < 0:
        mu1 = -mu1
    j = int(mu2.abs().argmax())
    if mu2[j] < 0:
        mu2 = -mu2
    e = s.square()
    raw = e / e.sum().clamp_min(EPS)
    rep = {
        "n_image_register_means": int(X.shape[0]),
        "width": int(X.shape[1]),
        "mu1_raw_uncentered_energy_fraction": float(raw[0]),
        "mu2_raw_uncentered_energy_fraction": float(raw[1]),
        "mu2_fraction_after_removing_mu1": float(e[1] / e[1:].sum().clamp_min(EPS)),
        "first_8_raw_energy_fractions": [float(x) for x in raw[:8].cpu()],
        "mu1_top_coordinates": [
            [int(i), float(mu1[i])] for i in torch.argsort(mu1.abs(), descending=True)[:16].cpu()
        ],
        "mu2_top_coordinates": [
            [int(i), float(mu2[i])] for i in torch.argsort(mu2.abs(), descending=True)[:16].cpu()
        ],
    }
    return mu1.cpu(), mu2.cpu(), rep

def distant_similarity_scores(spatial_bpd: torch.Tensor, G: int, local_radius: int = 1, topk: int = 8):
    B,P,D = spatial_bpd.shape
    # This analysis must stay FP32 even when the surrounding visual forward runs
    # under CUDA autocast.  Otherwise matmul is autocast back to FP16 and the
    # old -1e9 masking sentinel overflows Half before top-k is reached.
    if spatial_bpd.is_cuda:
        ctx = torch.autocast("cuda", enabled=False)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        n = F.normalize(spatial_bpd.float(), dim=-1)
        sim = torch.matmul(n, n.transpose(-2,-1)).float()
        mask = torch.ones((P,P), dtype=torch.bool, device=sim.device)
        for p in range(P):
            r,c = divmod(p,G)
            for rr in range(max(0,r-local_radius),min(G,r+local_radius+1)):
                for cc in range(max(0,c-local_radius),min(G,c+local_radius+1)):
                    mask[p,rr*G+cc] = False
        # -inf is representable in every floating dtype; unlike -1e9 it cannot
        # overflow if this function is ever changed to a lower-precision sim.
        sim = sim.masked_fill(~mask[None], float("-inf"))
        k = min(int(topk), int(mask.sum(dim=-1).min().item()))
        return sim.topk(k=k, dim=-1).values.mean(dim=-1)

def role_metrics(
    state_lbd: torch.Tensor,
    mu1: torch.Tensor,
    mu2: torch.Tensor,
    G: int,
    norm_thr: float,
    mu_reg_z: float,
    mu_hidden_z: float,
    scratch_z: float,
    scratch_topk: int,
):
    # Role classification is analysis, not part of the model forward.  Keep it
    # in FP32 so AMP cannot change mu projections, robust-z thresholds, or the
    # distant-patch cosine classifier.
    if state_lbd.is_cuda:
        ctx = torch.autocast("cuda", enabled=False)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        xb = state_lbd.permute(1,0,2).float()
        P = G*G
        sp = xb[:,1:1+P,:]
        cls = xb[:,0,:]
        m1 = mu1.to(device=sp.device, dtype=torch.float32)
        m2 = mu2.to(device=sp.device, dtype=torch.float32)
        p1 = torch.matmul(sp, m1)
        p2 = torch.matmul(sp, m2)
        z1 = robust_z_rows(p1)
        norm = sp.norm(dim=-1)
        legacy = norm > norm_thr
        reg = legacy & (z1 > mu_reg_z)
        hidden = (~legacy) & (z1 > mu_hidden_z)
        ds = distant_similarity_scores(sp,G,1,scratch_topk)
        dsz = robust_z_rows(ds)
        # Keep the exact prior scratchpad definition.
        scratch = (~legacy) & (~hidden) & (z1 < mu_reg_z) & (dsz > scratch_z)
        ordinary = ~(reg | hidden | scratch)
        return {
            "spatial": sp,
            "cls": cls,
            "mu1": p1,
            "mu2": p2,
            "mu1_z": z1,
            "norm": norm,
            "legacy_reg": legacy,
            "role_reg": reg,
            "hidden_mu": hidden,
            "scratchpad": scratch,
            "ordinary": ordinary,
            "distant_score": ds,
            "distant_z": dsz,
            "cls_mu1": torch.matmul(cls, m1),
            "cls_mu2": torch.matmul(cls, m2),
        }

def remove_role_plane(x: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor) -> torch.Tensor:
    m1 = F.normalize(mu1.to(x.device).float(), dim=0)
    m2 = mu2.to(x.device).float()
    m2 = m2 - torch.dot(m2,m1)*m1
    m2 = F.normalize(m2, dim=0)
    return x - (x@m1)[...,None]*m1 - (x@m2)[...,None]*m2

@torch.inference_mode()
def project_tokens_to_joint(variant: Variant, tokens_bpd: torch.Tensor) -> torch.Tensor:
    v = variant.visual
    B,P,D = tokens_bpd.shape
    z = v.ln_post(tokens_bpd.to(next(variant.model.parameters()).device))
    if v.proj is not None:
        z = z @ v.proj
    return z.float().cpu()

# ---------------------------------------------------------------------
# Input-derived text/object masks
# ---------------------------------------------------------------------

def _resize_rgb(path: str, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((size,size), Image.Resampling.BICUBIC)
    return np.asarray(im, np.float32) / 255.0

def _patch_coverage(pixel_mask: np.ndarray, G: int) -> np.ndarray:
    H,W = pixel_mask.shape
    if H % G or W % G:
        raise ValueError((H,W,G))
    ph,pw = H//G, W//G
    return pixel_mask.reshape(G,ph,G,pw).mean(axis=(1,3)).reshape(-1)

def build_input_region_masks(
    manifest: pd.DataFrame,
    image_size: int,
    G: int,
    *,
    text_pixel_threshold: float = 0.02,
    text_patch_coverage: float = 0.005,
    object_pixel_threshold: float = 0.08,
    object_patch_coverage: float = 0.03,
):
    lookup = {(r.condition,r.concept,bool(r.bw)):r for r in manifest.itertuples(index=False)}
    rows = []
    arrays = {}
    white = np.ones((image_size,image_size,3),np.float32)
    for r in manifest.itertuples(index=False):
        key = (r.condition,r.concept,bool(r.bw))
        img = _resize_rgb(r.path,image_size)
        text_px = np.zeros((image_size,image_size),bool)
        object_px = np.zeros_like(text_px)

        if r.condition == "mix" and ("vis",r.concept,bool(r.bw)) in lookup:
            vis = _resize_rgb(lookup[("vis",r.concept,bool(r.bw))].path,image_size)
            text_px = np.abs(img-vis).mean(axis=-1) > text_pixel_threshold
            object_px = np.abs(vis-white).mean(axis=-1) > object_pixel_threshold
        elif r.condition == "txt":
            text_px = np.abs(img-white).mean(axis=-1) > text_pixel_threshold
        elif r.condition == "vis":
            object_px = np.abs(img-white).mean(axis=-1) > object_pixel_threshold

        tc = _patch_coverage(text_px,G)
        oc = _patch_coverage(object_px,G)
        tm = tc > text_patch_coverage
        om = oc > object_patch_coverage
        bg = ~(tm | om)
        arrays[r.stim_id] = {
            "text_coverage": tc.astype(np.float32),
            "object_coverage": oc.astype(np.float32),
            "text_mask": tm,
            "object_mask": om,
            "background_mask": bg,
        }
        for p in range(G*G):
            rows.append({
                "stim_id": r.stim_id, "concept": r.concept, "condition": r.condition,
                "bw": bool(r.bw), "category": r.category,
                "patch_index0": p, "row0": p//G, "col0": p%G,
                "text_coverage": float(tc[p]), "object_coverage": float(oc[p]),
                "is_text_patch": int(tm[p]), "is_object_patch": int(om[p]),
                "is_background_patch": int(bg[p]),
            })
    return arrays, pd.DataFrame(rows)

# ---------------------------------------------------------------------
# PCA / geometric helpers
# ---------------------------------------------------------------------

def pca_2d(x: np.ndarray):
    x = np.asarray(x,np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x-mean
    _u,s,vh = np.linalg.svd(xc, full_matrices=False)
    z = xc @ vh[:2].T
    ev = (s*s) / max(float(np.sum(s*s)), EPS)
    return z, vh[:2], mean[0], ev[:2]

def direction_stats(D: np.ndarray) -> dict:
    D = np.asarray(D,np.float64)
    ok = np.isfinite(D).all(axis=1) & (np.linalg.norm(D,axis=1)>EPS)
    D = D[ok]
    if len(D)==0:
        return {"n":0}
    U = normalize_np(D)
    centroid = normalize_np(U.mean(axis=0))[0] if U.mean(axis=0).ndim>1 else normalize_np(U.mean(axis=0))
    cos_to = U @ centroid
    if len(U)>1:
        C = U @ U.T
        tri = C[np.triu_indices(len(U),1)]
        pair_mean = float(tri.mean())
        loo = []
        for i in range(len(U)):
            w = normalize_np(U[np.arange(len(U))!=i].mean(axis=0))
            loo.append(float(np.dot(U[i],w)))
        loo = np.asarray(loo)
    else:
        pair_mean = float("nan"); loo = np.array([np.nan])
    _u,s,_vh = np.linalg.svd(D, full_matrices=False)
    e = s*s
    return {
        "n": int(len(D)),
        "mean_norm": float(np.linalg.norm(D,axis=1).mean()),
        "mean_pairwise_cosine": pair_mean,
        "mean_cosine_to_centroid": float(np.mean(cos_to)),
        "min_cosine_to_centroid": float(np.min(cos_to)),
        "loo_positive_fraction": float(np.nanmean(loo>0)),
        "loo_mean_cosine": float(np.nanmean(loo)),
        "direction_pc1_energy_fraction": float(e[0]/max(e.sum(),EPS)),
        "direction_pc2_energy_fraction": float(e[1]/max(e.sum(),EPS)) if len(e)>1 else 0.0,
    }

# ---------------------------------------------------------------------
# Conv1 morphology
# ---------------------------------------------------------------------

def conv1_morphology(variant: Variant) -> pd.DataFrame:
    W = variant.visual.conv1.weight.detach().float().cpu().numpy()
    rows=[]
    H,Wk = W.shape[-2:]
    fy=np.fft.fftfreq(H); fx=np.fft.fftfreq(Wk)
    yy,xx=np.meshgrid(fy,fx,indexing="ij")
    rad=np.sqrt(xx*xx+yy*yy)
    for c,w in enumerate(W):
        ww=w.astype(np.float64)
        Fw=np.fft.fft2(ww,axes=(-2,-1))
        p=np.abs(Fw)**2
        total=p.sum()+EPS
        dc=float(p[:,0,0].sum()/total)
        hf=float(p[:,rad>=0.25].sum()/total)
        lf=float(p[:,rad<=0.12].sum()/total)
        rgb=ww.mean(axis=(-2,-1))
        chrom=float(np.std(rgb))
        rows.append({
            "channel":c,"weight_l2":float(np.linalg.norm(ww)),
            "kernel_dc_energy_frac":dc,"kernel_lowfreq_energy_frac":lf,
            "kernel_highfreq_energy_frac":hf,
            "dc_r":float(rgb[0]),"dc_g":float(rgb[1]),"dc_b":float(rgb[2]),
            "dc_rgb_std":chrom,
        })
    return pd.DataFrame(rows)

def self_test_common():
    names = [
        "mix_trout.png","mix_trout_bw.png","txt_trout.png","txt_trout_bw.png",
        "vis_trout.png","vis_trout_bw.png","txt_mouse.png",
    ]
    got=[]
    for n in names:
        m=IMAGE_RE.match(Path(n).stem)
        assert m
        got.append((m.group(1),m.group(2),bool(m.group(3))))
    assert got[0]==("mix","trout",False)
    assert got[1]==("mix","trout",True)
    assert got[-1]==("txt","mouse",False)

    x=torch.randn(2,17,23)
    assert tuple(x.norm(dim=-1).shape)==(2,17)

    # Regression for the AMP/FP16 scratchpad crash: input may be Half but
    # cosine similarity/masking must remain finite FP32.
    sp=torch.randn(2,16,23).half()
    ds=distant_similarity_scores(sp,4,1,4)
    assert tuple(ds.shape)==(2,16)
    assert ds.dtype==torch.float32
    assert torch.isfinite(ds).all()

    D=np.eye(4)
    st=direction_stats(D)
    assert st["n"]==4

    a=np.ones((224,224),bool)
    c=_patch_coverage(a,16)
    assert c.shape==(256,) and np.allclose(c,1)
