#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISUAL/TEXTUAL TEXT-DIRECTION TRAJECTORY ATLAS
==============================================

Purpose
-------
Turn the qualitative "combed PCA" observation into direct, layerwise and
statistically testable geometry.

The core controlled contrast is, for each concept c:

    add_text(c) = mix_c - vis_c

The script asks whether those 25 vectors share a direction, whether a direction
learned from OTHER concepts predicts txt_c - vis_c, where that direction first
appears inside the ViT, and whether it generalizes to unrelated synthetic words
rendered at random locations/styles.

Models
------
    pretrained, gmp, full_xattn

For full_xattn, the native/RN backbone is tracked through blocks and CONTENT is
also analyzed as a final 768-D endpoint.

Main outputs
------------
* layer_text_direction_summary.csv
* layer_crosscontrast_transfer.csv
* layer_axis_relationships.csv
* layer_direction_transport.csv
* synthetic_text_transfer.csv
* white_reference_alignment.csv
* conv1_white_reference.csv
* final_direction_angles.csv
* plots/*.png
* interactive/*.html
* compact_summary_conv1_visualtextual_text_direction.zip

Important design choices
------------------------
1. "Direction" is not inferred from PCA. PCA is only visualization.
2. Primary statistics operate in the original representation space.
3. Angles use leave-one-concept-out axes, so a held-out concept never helps
   define the direction against which it is evaluated.
4. A Monte-Carlo random-sign null preserves every vector exactly while
   destroying common orientation.
5. Synthetic overlays use unrelated words/pseudowords plus a tile-shuffled
   glyph control. This distinguishes a shared text/glyph direction from mere
   concept matching and gives a high-N verification bank.
6. White-reference tests use txt - white and compare it to mix - vis, directly
   removing the blank/sticker background contribution.
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
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_support.font_discovery import find_font_files
from visualtextual_probe_common import (
    EPS, amp_context, build_input_region_masks, cosine_np, import_runtime,
    load_variant, model_geometry, normalize_np, official_embeddings,
    parse_dataset, parse_models, pca_2d, prepare_tokens, preprocess_rows,
    seed_all, stable_seed, visual_final_embedding,
)

# =============================================================================
# Constants / helpers
# =============================================================================

COLORS = {
    "vis_rgb": "#1f77b4", "txt_rgb": "#d62728", "mix_rgb": "#9467bd",
    "vis_bw": "#17becf", "txt_bw": "#ff7f0e", "mix_bw": "#e377c2",
}

COMMON_WORDS = [
    "river","castle","window","forest","silver","orange","paper","cloud","garden","planet",
    "camera","pencil","mirror","bottle","rabbit","bridge","coffee","rocket","piano","winter",
    "summer","purple","circle","square","hammer","spoon","jacket","flower","beach","mountain",
    "street","train","apple","lemon","peach","cherry","whale","eagle","zebra","panda",
    "candle","clock","button","basket","helmet","violin","drum","ocean","island","desert",
    "storm","shadow","bright","quiet","small","large","round","sharp","soft","heavy",
    "metal","wooden","plastic","green","yellow","violet","black","white","brown","scarlet",
    "table","sofa","lamp","book","phone","screen","truck","bicycle","boat","plane",
    "bread","cheese","onion","tomato","pepper","berry","melon","cookie","pizza","soup",
    "fox","wolf","lion","tiger","otter","seal","owl","crow","duck","goat",
    "tree","leaf","grass","stone","sand","snow","rain","fire","smoke","water",
    "music","signal","vector","matrix","token","pixel","letter","word","image","object",
    "north","south","east","west","left","right","upper","lower","center","corner",
    "happy","strange","simple","complex","magic","normal","fuzzy","striped","spotted","smooth",
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def complete_concepts(meta: pd.DataFrame, bw: bool) -> List[str]:
    out=[]
    q=meta[meta["bw"].eq(bool(bw))]
    for c,g in q.groupby("concept"):
        if set(g["condition"]) >= {"vis","txt","mix"}:
            out.append(str(c))
    return sorted(out)


def complete_pair_manifest(meta: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Keep only concept/appearance cells with a complete vis/txt/mix triplet.

    The text-direction analysis is defined on paired ``mix-vis`` and ``txt-vis``
    contrasts and its canonical masks are shared across those three cells.
    Unpaired extras (for example legacy ``mix_adv_*`` images) therefore cannot
    participate in this experiment and must not enter the forward cache.
    """
    keep=set()
    for bw in (False,True):
        for c in complete_concepts(meta,bw):
            keep.add((str(c),bool(bw)))
    mask=np.asarray([
        (str(r.concept),bool(r.bw)) in keep
        for r in meta.itertuples(index=False)
    ],dtype=bool)
    paired=meta.loc[mask].copy().reset_index(drop=True)
    excluded=meta.loc[~mask].copy().reset_index(drop=True)
    if paired.empty:
        raise RuntimeError("No complete vis/txt/mix concept triplets found for text-direction analysis")
    return paired,excluded


def cell_lookup(meta: pd.DataFrame):
    return {(str(r.condition),str(r.concept),bool(r.bw)):str(r.stim_id) for r in meta.itertuples(index=False)}


def embedding_lookup(meta: pd.DataFrame, E: np.ndarray):
    return {str(r.stim_id):np.asarray(E[i]) for i,r in enumerate(meta.itertuples(index=False))}


def safe_unit(v: np.ndarray) -> np.ndarray:
    v=np.asarray(v,np.float64)
    n=float(np.linalg.norm(v))
    return v/max(n,EPS)


def angle_deg(a,b) -> float:
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    if (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
        return float("nan")
    if np.linalg.norm(a)<=EPS or np.linalg.norm(b)<=EPS:
        return float("nan")
    a=safe_unit(a); b=safe_unit(b)
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a,b)),-1,1))))


def cosine_rows(a,b):
    a=normalize_np(a); b=normalize_np(b)
    return np.sum(a*b,axis=-1)


def stage_order_label(nblocks: int):
    labels=["ln_pre"]+[f"B{i:02d}" for i in range(nblocks)]
    return labels


def rep_as_vectors(arr: np.ndarray, projected: bool=False) -> np.ndarray:
    """State vectors used for differences. Final/joint states are unit-normalized."""
    arr=np.asarray(arr,np.float64)
    if projected:
        return normalize_np(arr)
    return arr


def _loo_axes(U: np.ndarray) -> np.ndarray:
    U=normalize_np(U)
    total=U.sum(axis=0)
    axes=[]
    for i in range(len(U)):
        axes.append(safe_unit(total-U[i]))
    return np.stack(axes)


def direction_statistics(D: np.ndarray, seed: int, n_boot: int=1000, n_sign: int=4000) -> Tuple[dict,np.ndarray,np.ndarray]:
    D=np.asarray(D,np.float64)
    ok=np.isfinite(D).all(axis=1)&(np.linalg.norm(D,axis=1)>EPS)
    D=D[ok]
    if len(D)<2:
        return {"n":int(len(D))},np.array([]),np.array([])
    U=normalize_np(D)
    mean_axis=safe_unit(U.mean(axis=0))
    R=float(np.linalg.norm(U.mean(axis=0)))
    C=U@U.T
    tri=C[np.triu_indices(len(U),1)]
    axes=_loo_axes(U)
    loo_cos=np.sum(U*axes,axis=1)
    loo_angles=np.degrees(np.arccos(np.clip(loo_cos,-1,1)))
    s=np.linalg.svd(U,compute_uv=False); e=s*s

    rng=np.random.default_rng(seed)
    # Work in the N x N Gram matrix so resampling/null statistics do not scale
    # with representation width (768/1024). For coefficient vector a,
    # ||sum_i a_i u_i / N||^2 = a^T C a / N^2.
    boots=[]
    for _ in range(int(n_boot)):
        ii=rng.integers(0,len(U),len(U))
        counts=np.bincount(ii,minlength=len(U)).astype(np.float64)
        r2=float(counts @ C @ counts)/(len(U)**2)
        boots.append(math.sqrt(max(r2,0.0)))
    boots=np.asarray(boots)

    # Random-sign null: preserve every observed vector exactly and randomize
    # only common orientation. Gram evaluation makes thousands of draws cheap.
    signs=rng.choice(np.array([-1.,1.]),size=(int(n_sign),len(U)))
    r2=np.einsum("bi,ij,bj->b",signs,C,signs,optimize=True)/(len(U)**2)
    null=np.sqrt(np.maximum(r2,0.0))
    p=float((1+np.sum(null>=R))/(1+len(null)))

    out={
        "n":int(len(U)),
        "mean_norm":float(np.linalg.norm(D,axis=1).mean()),
        "resultant_length":R,
        "mean_pairwise_cosine":float(tri.mean()),
        "median_loo_angle_deg":float(np.median(loo_angles)),
        "mean_loo_angle_deg":float(np.mean(loo_angles)),
        "q90_loo_angle_deg":float(np.quantile(loo_angles,.90)),
        "max_loo_angle_deg":float(np.max(loo_angles)),
        "loo_positive_fraction":float(np.mean(loo_cos>0)),
        "direction_pc1_energy_fraction":float(e[0]/max(e.sum(),EPS)),
        "bootstrap_R_lo":float(np.quantile(boots,.025)),
        "bootstrap_R_hi":float(np.quantile(boots,.975)),
        "signflip_null_R_mean":float(null.mean()),
        "signflip_null_R_q995":float(np.quantile(null,.995)),
        "signflip_p":p,
    }
    return out,loo_angles,null


def transfer_statistics(train_D: np.ndarray, test_D: np.ndarray) -> Tuple[dict,np.ndarray]:
    train=np.asarray(train_D,np.float64); test=np.asarray(test_D,np.float64)
    n=min(len(train),len(test))
    train=train[:n]; test=test[:n]
    ok=(np.isfinite(train).all(axis=1)&np.isfinite(test).all(axis=1)&
        (np.linalg.norm(train,axis=1)>EPS)&(np.linalg.norm(test,axis=1)>EPS))
    train=train[ok]; test=test[ok]
    n=len(train)
    empty={
        "n":int(n),"positive_fraction":float("nan"),"mean_cosine":float("nan"),
        "median_angle_deg":float("nan"),"mean_angle_deg":float("nan"),
        "q90_angle_deg":float("nan"),"max_angle_deg":float("nan"),
    }
    # Leave-one-out axis requires at least two nonzero training directions.
    if n<2:
        return empty,np.array([],dtype=np.float64)
    U=normalize_np(train); V=normalize_np(test)
    axes=_loo_axes(U)
    valid_axes=np.linalg.norm(axes,axis=1)>EPS
    if not valid_axes.any():
        return empty,np.array([],dtype=np.float64)
    U=U[valid_axes]; V=V[valid_axes]; axes=axes[valid_axes]
    cos=np.sum(V*axes,axis=1)
    good=np.isfinite(cos)
    cos=cos[good]
    if len(cos)==0:
        return empty,np.array([],dtype=np.float64)
    ang=np.degrees(np.arccos(np.clip(cos,-1,1)))
    return {
        "n":int(len(cos)),
        "positive_fraction":float(np.mean(cos>0)),
        "mean_cosine":float(np.mean(cos)),
        "median_angle_deg":float(np.median(ang)),
        "mean_angle_deg":float(np.mean(ang)),
        "q90_angle_deg":float(np.quantile(ang,.90)),
        "max_angle_deg":float(np.max(ang)),
    },ang


def pca_3d(x: np.ndarray):
    x=np.asarray(x,np.float64); mean=x.mean(0,keepdims=True); xc=x-mean
    _u,s,vh=np.linalg.svd(xc,full_matrices=False)
    z=xc@vh[:3].T; ev=(s*s)/max(np.sum(s*s),EPS)
    return z,vh[:3],mean[0],ev[:3]

# =============================================================================
# Canonical paired masks
# =============================================================================

def canonical_region_masks(meta: pd.DataFrame, input_masks: Mapping[str,dict]) -> Dict[str,dict]:
    """Use the MIX mask for all three cells of a concept/appearance pair.

    This lets vis/txt/mix region means refer to exactly the same patch addresses.
    """
    lk=cell_lookup(meta); out={}
    for bw in (False,True):
        for c in complete_concepts(meta,bw):
            mix_id=lk[("mix",c,bw)]
            m=input_masks[mix_id]
            text=np.asarray(m["text_mask"],bool)
            obj=np.asarray(m["object_mask"],bool)
            bg=~(text|obj)
            # Never permit empty means to become NaN if a weird mask is absent.
            if not text.any():
                text=np.asarray(m["text_coverage"])>0
            if not obj.any():
                obj=np.asarray(m["object_coverage"])>0
            if not bg.any():
                bg=~text
            for cond in ("vis","txt","mix"):
                sid=lk[(cond,c,bw)]
                out[sid]={"text":text,"object":obj,"background":bg}
    return out


def masked_means(spatial_bpd: torch.Tensor, rows: pd.DataFrame, canonical: Mapping[str,dict], region: str) -> torch.Tensor:
    vals=[]
    for i,r in enumerate(rows.itertuples(index=False)):
        mask=torch.from_numpy(np.asarray(canonical[str(r.stim_id)][region],bool)).to(spatial_bpd.device)
        if bool(mask.any()): vals.append(spatial_bpd[i,mask].mean(0))
        else: vals.append(spatial_bpd[i].mean(0))
    return torch.stack(vals)

# =============================================================================
# Stage capture
# =============================================================================

def stage_cache_signature(variant,meta,args):
    payload={
        "model":variant.name,"source":variant.source_info,
        "stim_ids":list(meta.stim_id.astype(str)),
        "sizes":[int(Path(p).stat().st_size) for p in meta.path],
        "amp":bool(args.amp),"version":3,
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()[:20]


@torch.inference_mode()
def _capture_batch(variant,rows,canonical,args):
    size,G,patch,width,nblocks,nheads=model_geometry(variant); P=G*G
    batch=preprocess_rows(variant,rows)
    stage_labels=stage_order_label(nblocks)
    cls_raw=[]; cls_proj=[]; spatial=[]; textloc=[]; objectloc=[]; background=[]
    with amp_context(args.device,args.amp):
        x,conv=prepare_tokens(variant,batch)
        convsp=conv.permute(0,2,3,1).reshape(len(rows),P,width).float()
        conv_spatial=convsp.mean(1)
        conv_text=masked_means(convsp,rows,canonical,"text")
        conv_obj=masked_means(convsp,rows,canonical,"object")
        conv_bg=masked_means(convsp,rows,canonical,"background")

        def snap(xx):
            xb=xx.permute(1,0,2).float(); c=xb[:,0]; sp=xb[:,1:1+P]
            jp=variant.visual.ln_post(c.to(next(variant.model.parameters()).device))
            if variant.visual.proj is not None: jp=jp@variant.visual.proj
            cls_raw.append(c.detach().cpu())
            cls_proj.append(jp.float().detach().cpu())
            spatial.append(sp.mean(1).detach().cpu())
            textloc.append(masked_means(sp,rows,canonical,"text").detach().cpu())
            objectloc.append(masked_means(sp,rows,canonical,"object").detach().cpu())
            background.append(masked_means(sp,rows,canonical,"background").detach().cpu())

        snap(x)  # ln_pre
        for bi,blk in enumerate(variant.visual.transformer.resblocks):
            x=variant.maybe_insert_rn(bi,x)
            x=blk(x)
            snap(x)

        final_back=visual_final_embedding(variant,x).float().detach().cpu()

    out={
        "stim_ids":np.asarray(list(rows.stim_id.astype(str))),
        "stage_labels":np.asarray(stage_labels),
        "cls_raw":torch.stack(cls_raw,1).numpy().astype(np.float16),
        "cls_proj":torch.stack(cls_proj,1).numpy().astype(np.float16),
        "spatial_mean":torch.stack(spatial,1).numpy().astype(np.float16),
        "textloc_mean":torch.stack(textloc,1).numpy().astype(np.float16),
        "objectloc_mean":torch.stack(objectloc,1).numpy().astype(np.float16),
        "background_mean":torch.stack(background,1).numpy().astype(np.float16),
        "conv_spatial":conv_spatial.cpu().numpy().astype(np.float16),
        "conv_textloc":conv_text.cpu().numpy().astype(np.float16),
        "conv_objectloc":conv_obj.cpu().numpy().astype(np.float16),
        "conv_background":conv_bg.cpu().numpy().astype(np.float16),
        "final_backbone":final_back.numpy().astype(np.float32),
    }
    return out


def collect_real_stage_cache(variant,meta,canonical,args,model_dir):
    sig=stage_cache_signature(variant,meta,args); final=model_dir/"real_stage_cache.npz"; sigp=model_dir/"real_stage_signature.txt"
    if final.is_file() and sigp.is_file() and sigp.read_text().strip()==sig:
        print(f"[{variant.name}] resume real stage cache")
        z=np.load(final,allow_pickle=False); return {k:z[k] for k in z.files}
    bdir=ensure_dir(model_dir/"real_stage_batches"/sig)
    chunks=[]
    for st in range(0,len(meta),args.batch_size):
        q=meta.iloc[st:st+args.batch_size]; bp=bdir/f"{st:04d}_{st+len(q):04d}.npz"
        if bp.is_file():
            z=np.load(bp,allow_pickle=False); d={k:z[k] for k in z.files};
            if list(d["stim_ids"].astype(str))!=list(q.stim_id.astype(str)): raise RuntimeError("stage batch ID mismatch")
            print(f"[{variant.name} real stages] resume {st}:{st+len(q)}")
        else:
            d=_capture_batch(variant,q,canonical,args); np.savez_compressed(bp,**d)
            print(f"[{variant.name} real stages] {min(st+len(q),len(meta))}/{len(meta)}")
        chunks.append(d)
    keys=[k for k in chunks[0] if k not in ("stage_labels",)]
    merged={"stage_labels":chunks[0]["stage_labels"]}
    for k in keys: merged[k]=np.concatenate([d[k] for d in chunks],axis=0)

    # CONTENT is a genuine post-backbone endpoint; collect only for real images.
    if variant.is_full_xattn:
        content=[]
        for st in range(0,len(meta),args.batch_size):
            q=meta.iloc[st:st+args.batch_size]; batch=preprocess_rows(variant,q)
            off=official_embeddings(variant,batch,args.amp); content.append(off["content"].numpy().astype(np.float32))
        merged["content"]=np.concatenate(content,0)

    np.savez_compressed(final,**merged); sigp.write_text(sig); shutil.rmtree(bdir,ignore_errors=True)
    return merged


@torch.inference_mode()
def collect_white_stage(variant,args,model_dir):
    p=model_dir/"white_stage_cache.npz"
    if p.is_file():
        z=np.load(p,allow_pickle=False); return {k:z[k] for k in z.files}
    size,G,patch,width,nblocks,nheads=model_geometry(variant); P=G*G
    white=Image.new("RGB",(size,size),(255,255,255)); batch=torch.stack([variant.preprocess(white)])
    cls_raw=[]; cls_proj=[]; spatial_tokens=[]
    with amp_context(args.device,args.amp):
        x,conv=prepare_tokens(variant,batch)
        convsp=conv.permute(0,2,3,1).reshape(1,P,width).float()[0]
        def snap(xx):
            xb=xx.permute(1,0,2).float(); c=xb[:,0]
            jp=variant.visual.ln_post(c.to(next(variant.model.parameters()).device))
            if variant.visual.proj is not None: jp=jp@variant.visual.proj
            cls_raw.append(c[0].detach().cpu()); cls_proj.append(jp[0].float().detach().cpu())
            spatial_tokens.append(xb[0,1:1+P].detach().cpu())
        snap(x)
        for bi,blk in enumerate(variant.visual.transformer.resblocks):
            x=variant.maybe_insert_rn(bi,x); x=blk(x); snap(x)
        final=visual_final_embedding(variant,x).float().cpu()[0]
    out={
        "stage_labels":np.asarray(stage_order_label(nblocks)),
        "cls_raw":torch.stack(cls_raw).numpy().astype(np.float16),
        "cls_proj":torch.stack(cls_proj).numpy().astype(np.float16),
        "spatial_tokens":torch.stack(spatial_tokens).numpy().astype(np.float16),
        "conv_tokens":convsp.cpu().numpy().astype(np.float16),
        "final_backbone":final.numpy().astype(np.float32),
    }
    if variant.is_full_xattn:
        out["content"]=official_embeddings(variant,batch,args.amp)["content"][0].numpy().astype(np.float32)
    np.savez_compressed(p,**out); return out

# =============================================================================
# Real paired vector extraction
# =============================================================================

def _white_region_vectors(white,meta,canonical,key):
    """Return white reference region vector per stimulus for stage representation key."""
    if key in ("cls_raw","cls_proj"):
        W=np.asarray(white[key],np.float64)
        return {sid:W for sid in meta.stim_id.astype(str)}
    map_key={"spatial_mean":None,"textloc_mean":"text","objectloc_mean":"object","background_mean":"background"}[key]
    wt=np.asarray(white["spatial_tokens"],np.float64)  # [S,P,D]
    out={}
    for r in meta.itertuples(index=False):
        if map_key is None: out[str(r.stim_id)]=wt.mean(axis=1)
        else:
            m=np.asarray(canonical[str(r.stim_id)][map_key],bool)
            out[str(r.stim_id)]=wt[:,m].mean(axis=1) if m.any() else wt.mean(axis=1)
    return out


def _real_vectors_for_stage(meta,cache,white,canonical,key,stage_i,projected):
    ids={str(s):i for i,s in enumerate(cache["stim_ids"].astype(str))}; lk=cell_lookup(meta)
    X=rep_as_vectors(cache[key][:,stage_i,:].astype(np.float64),projected)
    lookup={sid:X[i] for sid,i in ids.items()}
    Wdict=_white_region_vectors(white,meta,canonical,key)
    if projected:
        Wdict={k:rep_as_vectors(v.astype(np.float64),True) for k,v in Wdict.items()}
    out={"rgb":{},"bw":{}}
    concepts={"rgb":complete_concepts(meta,False),"bw":complete_concepts(meta,True)}
    for app,bw in (("rgb",False),("bw",True)):
        vals={"add_text":[],"add_visual":[],"txt_minus_vis":[],"txt_white":[],"vis_white":[],"concepts":[]}
        for c in concepts[app]:
            vi=lk[("vis",c,bw)]; ti=lk[("txt",c,bw)]; mi=lk[("mix",c,bw)]
            v,t,m=lookup[vi],lookup[ti],lookup[mi]
            # White CLS is common. Region white uses this concept's canonical patch addresses.
            wv=Wdict[ti][stage_i] if Wdict[ti].ndim==2 else Wdict[ti]
            vals["add_text"].append(m-v); vals["add_visual"].append(m-t); vals["txt_minus_vis"].append(t-v)
            vals["txt_white"].append(t-wv); vals["vis_white"].append(v-wv); vals["concepts"].append(c)
        for k in list(vals):
            if k!="concepts": vals[k]=np.stack(vals[k])
        out[app]=vals
    return out


def analyze_real_layers(model_name,meta,cache,white,canonical,args,model_dir):
    stages=list(cache["stage_labels"].astype(str))
    # Keep a human-facing representation name separate from the serialized
    # cache key.  The cache stores projected CLS as ``cls_proj``; earlier code
    # accidentally tried to index it with the report label ``cls_projected``.
    reps=[
        ("cls_projected","cls_proj",True),("cls_raw","cls_raw",False),
        ("spatial_mean","spatial_mean",False),("textloc_mean","textloc_mean",False),
        ("objectloc_mean","objectloc_mean",False),("background_mean","background_mean",False),
    ]
    rows=[]; transfer=[]; axisrows=[]; angle_rows=[]; null_examples={}
    axis_store={}
    for rep,cache_key,projected in reps:
        if cache_key not in cache:
            raise KeyError(f"Missing real-stage cache key {cache_key!r} for representation {rep!r}; available={sorted(cache)}")
        for app in ("rgb","bw"):
            stage_axes=[]
            perstage=[]
            for si,stage in enumerate(stages):
                vv=_real_vectors_for_stage(meta,cache,white,canonical,cache_key,si,projected)[app]
                stats,angles,null=direction_statistics(vv["add_text"],stable_seed(f"{model_name}:{rep}:{app}:{stage}"),args.bootstrap,args.signflip)
                rows.append({"model":model_name,"representation":rep,"appearance":app,"stage":stage,"stage_index":si,"contrast":"add_text",**stats})
                # Other contrasts without expensive sign/boot null.
                for name in ("add_visual","txt_minus_vis","txt_white","vis_white"):
                    st,ang,_null=direction_statistics(vv[name],stable_seed(f"cheap:{model_name}:{rep}:{app}:{stage}:{name}"),max(50,args.bootstrap//10),max(100,args.signflip//20))
                    rows.append({"model":model_name,"representation":rep,"appearance":app,"stage":stage,"stage_index":si,"contrast":name,**st})
                for test_name in ("txt_minus_vis","txt_white"):
                    tr,ang=transfer_statistics(vv["add_text"],vv[test_name])
                    transfer.append({"model":model_name,"representation":rep,"appearance":app,"stage":stage,"stage_index":si,"train_contrast":"add_text","test_contrast":test_name,**tr})
                    if rep=="cls_projected" and test_name=="txt_minus_vis":
                        for c,a in zip(vv["concepts"],ang): angle_rows.append({"model":model_name,"appearance":app,"stage":stage,"stage_index":si,"concept":c,"angle_deg":float(a)})
                wa=safe_unit(normalize_np(vv["add_text"]).mean(0)); wv=safe_unit(normalize_np(vv["add_visual"]).mean(0))
                axisrows.append({"model":model_name,"representation":rep,"appearance":app,"stage":stage,"stage_index":si,"angle_text_vs_visual_deg":angle_deg(wa,wv),"cos_text_vs_visual":float(np.dot(wa,wv))})
                stage_axes.append(wa); perstage.append((stats,null))
            final_axis=stage_axes[-1]
            for si,(stage,w) in enumerate(zip(stages,stage_axes)):
                axis_store[(rep,app,stage)]=w
                axisrows.append({"model":model_name,"representation":rep,"appearance":app,"stage":stage,"stage_index":si,"angle_to_final_text_axis_deg":angle_deg(w,final_axis),"cos_to_final_text_axis":float(np.dot(w,final_axis))})
            if rep=="cls_projected": null_examples[app]=perstage[-1]

    rdf=pd.DataFrame(rows); tdf=pd.DataFrame(transfer); adf=pd.DataFrame(axisrows); angledf=pd.DataFrame(angle_rows)
    rdf.to_csv(model_dir/"layer_text_direction_summary.csv",index=False)
    tdf.to_csv(model_dir/"layer_crosscontrast_transfer.csv",index=False)
    adf.to_csv(model_dir/"layer_axis_relationships.csv",index=False)
    angledf.to_csv(model_dir/"final_direction_angles_by_concept.csv",index=False)

    # Stage-to-stage transport matrix in projected CLS space.
    trans=[]
    for app in ("rgb","bw"):
        A=np.stack([axis_store[("cls_projected",app,s)] for s in stages])
        norms=np.linalg.norm(A,axis=1)
        for i,sa in enumerate(stages):
            for j,sb in enumerate(stages):
                if norms[i]<=EPS or norms[j]<=EPS or (not np.isfinite(A[i]).all()) or (not np.isfinite(A[j]).all()):
                    cc=float("nan"); aa=float("nan")
                else:
                    cc=float(np.dot(A[i],A[j])); aa=float(np.degrees(np.arccos(np.clip(cc,-1,1))))
                trans.append({"model":model_name,"appearance":app,"stage_a":sa,"stage_b":sb,"i":i,"j":j,"cosine":cc,"angle_deg":aa})
    trdf=pd.DataFrame(trans); trdf.to_csv(model_dir/"layer_direction_transport.csv",index=False)
    return rdf,tdf,adf,angledf,trdf,axis_store,null_examples

# =============================================================================
# Conv1 white-reference
# =============================================================================

def analyze_conv1_white_reference(model_name,meta,cache,white,canonical,args,model_dir):
    ids={str(s):i for i,s in enumerate(cache["stim_ids"].astype(str))}; lk=cell_lookup(meta)
    rows=[]
    for rep,white_key in (("conv_textloc","conv_tokens"),("conv_spatial","conv_tokens")):
        for app,bw in (("rgb",False),("bw",True)):
            add=[]; txtw=[]; concepts=[]
            for c in complete_concepts(meta,bw):
                vi,ti,mi=[lk[(cond,c,bw)] for cond in ("vis","txt","mix")]
                v=cache[rep][ids[vi]].astype(np.float64); t=cache[rep][ids[ti]].astype(np.float64); m=cache[rep][ids[mi]].astype(np.float64)
                if rep=="conv_textloc":
                    mask=np.asarray(canonical[ti]["text"],bool); wt=white["conv_tokens"].astype(np.float64); w=wt[mask].mean(0) if mask.any() else wt.mean(0)
                else: w=white["conv_tokens"].astype(np.float64).mean(0)
                add.append(m-v); txtw.append(t-w); concepts.append(c)
            add=np.stack(add); txtw=np.stack(txtw)
            tr,ang=transfer_statistics(add,txtw)
            s1,_,_=direction_statistics(add,stable_seed(f"conv:{model_name}:{rep}:{app}:add"),args.bootstrap,args.signflip)
            s2,_,_=direction_statistics(txtw,stable_seed(f"conv:{model_name}:{rep}:{app}:txtw"),max(50,args.bootstrap//10),max(100,args.signflip//20))
            rows.append({"model":model_name,"representation":rep,"appearance":app,"measurement":"add_text",**s1})
            rows.append({"model":model_name,"representation":rep,"appearance":app,"measurement":"txt_white",**s2})
            rows.append({"model":model_name,"representation":rep,"appearance":app,"measurement":"add_text_to_txt_white_transfer",**tr})
    df=pd.DataFrame(rows); df.to_csv(model_dir/"conv1_white_reference.csv",index=False); return df

# =============================================================================
# Synthetic unrelated-word bank
# =============================================================================

def find_fonts() -> List[str]:
    preferred = (
        "arial.ttf",
        "segoeui.ttf",
        "calibri.ttf",
        "times.ttf",
        "consola.ttf",
        "DejaVuSans.ttf",
        "DejaVuSerif.ttf",
        "Arial.ttf",
        "Helvetica.ttc",
    )
    return find_font_files(preferred_names=preferred, fallback_limit=12)


def pseudo_word(rng,minlen=5,maxlen=10):
    vowels="aeiou"; cons="bcdfghjklmnprstvwxyz"
    n=int(rng.integers(minlen,maxlen+1)); out=[]
    start=bool(rng.integers(0,2))
    for i in range(n): out.append(rng.choice(list(vowels if ((i%2==0)==start) else cons)))
    return "".join(out)


def _load_font(fonts,size):
    for p in fonts:
        try:return ImageFont.truetype(p,int(size))
        except Exception:pass
    return ImageFont.load_default()


def render_text_layer(size:int, words:Sequence[str], rng, fonts):
    layer=Image.new("RGBA",(size,size),(0,0,0,0)); d=ImageDraw.Draw(layer)
    placements=[]
    for word in words:
        fs=int(rng.integers(max(13,size//16),max(20,size//6)))
        font=_load_font(fonts,fs)
        color=tuple(int(x) for x in rng.integers(0,235,size=3))+(int(rng.integers(190,256)),)
        # Draw each word on its own temporary layer, then rotate.
        bbox=d.textbbox((0,0),word,font=font,stroke_width=1); ww=max(2,bbox[2]-bbox[0]+8); hh=max(2,bbox[3]-bbox[1]+8)
        tmp=Image.new("RGBA",(ww,hh),(0,0,0,0)); td=ImageDraw.Draw(tmp)
        td.text((4-bbox[0],4-bbox[1]),word,font=font,fill=color,stroke_width=int(rng.integers(0,2)),stroke_fill=(255-color[0],255-color[1],255-color[2],color[3]))
        ang=float(rng.uniform(-32,32)); tmp=tmp.rotate(ang,resample=Image.Resampling.BICUBIC,expand=True)
        if tmp.width>size-4 or tmp.height>size-4:
            scale=min((size-4)/max(tmp.width,1),(size-4)/max(tmp.height,1))
            tmp=tmp.resize((max(1,int(tmp.width*scale)),max(1,int(tmp.height*scale))),Image.Resampling.LANCZOS)
        x=int(rng.integers(0,max(1,size-tmp.width+1))); y=int(rng.integers(0,max(1,size-tmp.height+1)))
        layer.alpha_composite(tmp,(x,y)); placements.append((word,fs,ang,x,y))
    return layer,placements


def tile_shuffle_layer(layer:Image.Image,rng,tile:int=8):
    a=np.asarray(layer).copy(); H,W,C=a.shape
    ph=math.ceil(H/tile); pw=math.ceil(W/tile)
    pad=np.zeros((ph*tile,pw*tile,C),dtype=a.dtype); pad[:H,:W]=a
    blocks=pad.reshape(ph,tile,pw,tile,C).transpose(0,2,1,3,4).reshape(ph*pw,tile,tile,C)
    perm=rng.permutation(len(blocks)); blocks=blocks[perm]
    out=blocks.reshape(ph,pw,tile,tile,C).transpose(0,2,1,3,4).reshape(ph*tile,pw*tile,C)[:H,:W]
    return Image.fromarray(out,"RGBA")


def generate_synthetic_bank(meta:pd.DataFrame,size:int,args,root:Path):
    syn_dir=ensure_dir(root/"SYNTHETIC_INPUTS"); manp=root/"synthetic_manifest.csv"; cfgp=root/"synthetic_config.json"
    bases=meta[(meta.condition.eq("vis"))&(meta.bw.eq(False))].sort_values("concept")
    cfg={"seed":int(args.seed),"synthetic_per_base":int(args.synthetic_per_base),"shuffle_tile":int(args.synthetic_shuffle_tile),"size":int(size),"base_ids":list(bases.stim_id.astype(str))}
    expected=len(bases)*int(args.synthetic_per_base)*3
    if manp.is_file() and cfgp.is_file():
        try:
            oldcfg=json.loads(cfgp.read_text(encoding="utf-8")); d=pd.read_csv(manp)
            if oldcfg==cfg and len(d)==expected and all(Path(p).is_file() for p in d.path): return d
        except Exception:
            pass
    shutil.rmtree(syn_dir,ignore_errors=True); syn_dir=ensure_dir(syn_dir)
    fonts=find_fonts(); rows=[]
    all_concepts=set(meta.concept.astype(str))
    for base in bases.itertuples(index=False):
        bim=Image.open(base.path).convert("RGB").resize((size,size),Image.Resampling.BICUBIC)
        for rep in range(args.synthetic_per_base):
            rr=np.random.default_rng(stable_seed(f"syn:{args.seed}:{base.concept}:{rep}"))
            nw=int(rr.integers(1,4))
            words=[]
            for _ in range(nw):
                pool=[w for w in COMMON_WORDS if w not in all_concepts and w!=base.concept]
                words.append(str(rr.choice(pool)))
            pseudo=[pseudo_word(rr) for _ in range(nw)]
            real_layer,placements=render_text_layer(size,words,rr,fonts)
            pseudo_layer,_=render_text_layer(size,pseudo,rr,fonts)
            scramble_layer=tile_shuffle_layer(real_layer,rr,tile=args.synthetic_shuffle_tile)
            gid=f"{base.concept}_{rep:02d}"
            for variant_name,layer,text in (("realword",real_layer," ".join(words)),("pseudoword",pseudo_layer," ".join(pseudo)),("glyphscramble",scramble_layer,"<shuffled-glyph-bitmap>")):
                out=bim.convert("RGBA"); out.alpha_composite(layer); out=out.convert("RGB")
                fp=syn_dir/f"{gid}_{variant_name}.png"; out.save(fp,optimize=True)
                diff=np.abs(np.asarray(out,np.float32)-np.asarray(bim,np.float32)).mean(-1)>1.0
                # Store mask compactly as pipe-separated flat indices.
                # Later conversion to patch coverage uses the pixel mask rebuilt from PNG/base difference.
                rows.append({"stim_id":f"syn_{gid}_{variant_name}","group_id":gid,"base_stim_id":base.stim_id,"base_concept":base.concept,
                             "variant":variant_name,"text":text,"path":str(fp.resolve())})
    d=pd.DataFrame(rows); d.to_csv(manp,index=False); cfgp.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    make_synthetic_gallery(d,bases,size,root/"synthetic_examples.png")
    return d


def make_synthetic_gallery(syn,bases,size,path):
    q=syn.groupby("group_id").head(3).head(12)
    if q.empty:return
    groups=list(q.group_id.unique())[:4]; cell=size//2
    canvas=Image.new("RGB",(cell*4,cell*len(groups)),(245,245,245))
    bl={str(r.stim_id):r for r in bases.itertuples(index=False)}
    for ri,gid in enumerate(groups):
        gg=syn[syn.group_id.eq(gid)]
        base_id=str(gg.base_stim_id.iloc[0]); b=Image.open(bl[base_id].path).convert("RGB").resize((cell,cell))
        canvas.paste(b,(0,ri*cell))
        for ci,var in enumerate(("realword","pseudoword","glyphscramble"),1):
            r=gg[gg.variant.eq(var)].iloc[0]; im=Image.open(r.path).convert("RGB").resize((cell,cell)); canvas.paste(im,(ci*cell,ri*cell))
    canvas.save(path)


def _synthetic_masks(syn:pd.DataFrame,base_meta:pd.DataFrame,size:int,G:int):
    base_by_id={str(r.stim_id):r for r in base_meta.itertuples(index=False)}; out={}
    for r in syn.itertuples(index=False):
        a=np.asarray(Image.open(r.path).convert("RGB").resize((size,size),Image.Resampling.BICUBIC),np.float32)
        b=np.asarray(Image.open(base_by_id[str(r.base_stim_id)].path).convert("RGB").resize((size,size),Image.Resampling.BICUBIC),np.float32)
        pix=np.abs(a-b).mean(-1)>.5
        ph=size//G; cov=pix.reshape(G,ph,G,ph).mean(axis=(1,3)).reshape(-1); tm=cov>.002
        if not tm.any():tm[cov.argmax()]=True
        out[str(r.stim_id)]={"text":tm,"object":~tm,"background":~tm}
    return out


def preprocess_synthetic_rows(variant,rows):
    ims=[Image.open(p).convert("RGB") for p in rows.path]
    return torch.stack([variant.preprocess(im) for im in ims])


@torch.inference_mode()
def _capture_synthetic_batch(variant,rows,maskmap,args):
    size,G,patch,width,nblocks,nheads=model_geometry(variant); P=G*G
    batch=preprocess_synthetic_rows(variant,rows)
    cls_raw=[]; cls_proj=[]; textloc=[]
    with amp_context(args.device,args.amp):
        x,conv=prepare_tokens(variant,batch)
        convsp=conv.permute(0,2,3,1).reshape(len(rows),P,width).float(); conv_text=masked_means(convsp,rows.rename(columns={"stim_id":"stim_id"}),maskmap,"text")
        def snap(xx):
            xb=xx.permute(1,0,2).float(); c=xb[:,0]; sp=xb[:,1:1+P]
            jp=variant.visual.ln_post(c.to(next(variant.model.parameters()).device));
            if variant.visual.proj is not None:jp=jp@variant.visual.proj
            cls_raw.append(c.cpu()); cls_proj.append(jp.float().cpu()); textloc.append(masked_means(sp,rows,maskmap,"text").cpu())
        snap(x)
        for bi,blk in enumerate(variant.visual.transformer.resblocks): x=variant.maybe_insert_rn(bi,x); x=blk(x); snap(x)
    return {"stim_ids":np.asarray(list(rows.stim_id.astype(str))),"stage_labels":np.asarray(stage_order_label(nblocks)),
            "cls_raw":torch.stack(cls_raw,1).numpy().astype(np.float16),"cls_proj":torch.stack(cls_proj,1).numpy().astype(np.float16),
            "textloc_mean":torch.stack(textloc,1).numpy().astype(np.float16),"conv_textloc":conv_text.cpu().numpy().astype(np.float16)}


def collect_synthetic_stage_cache(variant,syn,maskmap,args,model_dir):
    syn_file_sig="|".join(f"{sid}:{Path(p).stat().st_size}" for sid,p in zip(syn.stim_id.astype(str),syn.path.astype(str)))
    final=model_dir/"synthetic_stage_cache.npz"; sig=hashlib.sha256((variant.name+json.dumps(variant.source_info,sort_keys=True,default=str)+"|"+syn_file_sig+f"|{args.amp}").encode()).hexdigest()[:20]
    sigp=model_dir/"synthetic_stage_signature.txt"
    if final.is_file() and sigp.is_file() and sigp.read_text().strip()==sig:
        print(f"[{variant.name}] resume synthetic stage cache"); z=np.load(final,allow_pickle=False); return {k:z[k] for k in z.files}
    bdir=ensure_dir(model_dir/"synthetic_stage_batches"/sig); chunks=[]
    for st in range(0,len(syn),args.batch_size):
        q=syn.iloc[st:st+args.batch_size]; bp=bdir/f"{st:04d}_{st+len(q):04d}.npz"
        if bp.is_file(): z=np.load(bp,allow_pickle=False); d={k:z[k] for k in z.files}; print(f"[{variant.name} synthetic] resume {st}:{st+len(q)}")
        else: d=_capture_synthetic_batch(variant,q,maskmap,args); np.savez_compressed(bp,**d); print(f"[{variant.name} synthetic] {min(st+len(q),len(syn))}/{len(syn)}")
        chunks.append(d)
    merged={"stage_labels":chunks[0]["stage_labels"]}
    for k in chunks[0]:
        if k!="stage_labels":merged[k]=np.concatenate([d[k] for d in chunks],0)
    np.savez_compressed(final,**merged); sigp.write_text(sig); shutil.rmtree(bdir,ignore_errors=True); return merged


def _angle_summary_from_cosines(cos, projections=None, norms=None):
    cos=np.asarray(cos,np.float64)
    cos=cos[np.isfinite(cos)]
    if len(cos)==0:
        return {
            "n":0,"positive_fraction":float("nan"),"mean_cosine":float("nan"),
            "median_angle_deg":float("nan"),"q90_angle_deg":float("nan"),
            "mean_projection":float("nan"),"mean_norm":float("nan"),
        }
    ang=np.degrees(np.arccos(np.clip(cos,-1,1)))
    p=np.asarray([] if projections is None else projections,np.float64)
    p=p[np.isfinite(p)]
    nn=np.asarray([] if norms is None else norms,np.float64)
    nn=nn[np.isfinite(nn)]
    return {
        "n":int(len(cos)),"positive_fraction":float(np.mean(cos>0)),
        "mean_cosine":float(np.mean(cos)),"median_angle_deg":float(np.median(ang)),
        "q90_angle_deg":float(np.quantile(ang,.90)),
        "mean_projection":float(np.mean(p)) if len(p) else float("nan"),
        "mean_norm":float(np.mean(nn)) if len(nn) else float("nan"),
    }


def analyze_synthetic_transfer(model_name,meta,real_cache,syn,syn_cache,real_axis_store,args,model_dir):
    real_ids={str(s):i for i,s in enumerate(real_cache["stim_ids"].astype(str))}; syn_ids={str(s):i for i,s in enumerate(syn_cache["stim_ids"].astype(str))}
    stages=list(syn_cache["stage_labels"].astype(str)); rows=[]
    for rep,projected in (("cls_projected",True),("cls_raw",False),("textloc_mean",False)):
        real_key="cls_proj" if rep=="cls_projected" else rep
        syn_key="cls_proj" if rep=="cls_projected" else rep
        for si,stage in enumerate(stages):
            axis=real_axis_store[("cls_projected" if rep=="cls_projected" else rep,"rgb",stage)]
            axis_valid=np.isfinite(axis).all() and np.linalg.norm(axis)>EPS
            for variant_name,g in syn.groupby("variant"):
                cos=[]; proj=[]; norms=[]
                if not axis_valid:
                    stats=_angle_summary_from_cosines([],[],[])
                    rows.append({"model":model_name,"representation":rep,"stage":stage,"stage_index":si,"variant":variant_name,**stats})
                    continue
                for r in g.itertuples(index=False):
                    si2=syn_ids[str(r.stim_id)]; bi=real_ids[str(r.base_stim_id)]
                    a=syn_cache[syn_key][si2,si].astype(np.float64); b=real_cache[real_key][bi,si].astype(np.float64)
                    if projected: a=normalize_np(a); b=normalize_np(b)
                    d=a-b; n=np.linalg.norm(d)
                    if n<=EPS:continue
                    u=d/n; cos.append(float(np.dot(u,axis))); proj.append(float(np.dot(d,axis))); norms.append(float(n))
                stats=_angle_summary_from_cosines(cos,proj,norms)
                rows.append({"model":model_name,"representation":rep,"stage":stage,"stage_index":si,"variant":variant_name,**stats})
    # Conv1 text-location uses the real final RGB conv text axis learned from mix-vis.
    lk=cell_lookup(meta); realD=[]
    for c in complete_concepts(meta,False):
        vi,mi=lk[("vis",c,False)],lk[("mix",c,False)]; realD.append(real_cache["conv_textloc"][real_ids[mi]].astype(float)-real_cache["conv_textloc"][real_ids[vi]].astype(float))
    wax=safe_unit(normalize_np(np.stack(realD)).mean(0))
    for variant_name,g in syn.groupby("variant"):
        cs=[]
        for r in g.itertuples(index=False):
            d=syn_cache["conv_textloc"][syn_ids[str(r.stim_id)]].astype(float)-real_cache["conv_textloc"][real_ids[str(r.base_stim_id)]].astype(float)
            if np.linalg.norm(d)>EPS:cs.append(float(np.dot(safe_unit(d),wax)))
        stats=_angle_summary_from_cosines(cs)
        rows.append({"model":model_name,"representation":"conv_textloc","stage":"conv1","stage_index":-1,"variant":variant_name,**stats})
    df=pd.DataFrame(rows); df.to_csv(model_dir/"synthetic_text_transfer.csv",index=False); return df

# =============================================================================
# CONTENT endpoint and final static/interactive visualizations
# =============================================================================

def final_content_metrics(model_name,meta,cache,args,model_dir):
    if "content" not in cache:return pd.DataFrame(),pd.DataFrame(),{}
    E=normalize_np(cache["content"].astype(np.float64)); ids={str(s):i for i,s in enumerate(cache["stim_ids"].astype(str))}; lk=cell_lookup(meta)
    rows=[]; transfers=[]; vecs={}
    for app,bw in (("rgb",False),("bw",True)):
        add=[]; txtvis=[]; concepts=[]
        for c in complete_concepts(meta,bw):
            v,t,m=[E[ids[lk[(cond,c,bw)]]] for cond in ("vis","txt","mix")]
            add.append(m-v); txtvis.append(t-v); concepts.append(c)
        add=np.stack(add); txtvis=np.stack(txtvis); st,ang,null=direction_statistics(add,stable_seed(f"content:{model_name}:{app}"),args.bootstrap,args.signflip)
        rows.append({"model":model_name,"representation":"content","appearance":app,"stage":"CONTENT","stage_index":25,"contrast":"add_text",**st})
        tr,tang=transfer_statistics(add,txtvis); transfers.append({"model":model_name,"representation":"content","appearance":app,"stage":"CONTENT","stage_index":25,"train_contrast":"add_text","test_contrast":"txt_minus_vis",**tr})
        vecs[app]={"add_text":add,"txt_minus_vis":txtvis,"concepts":concepts,"null":null,"angles":tang}
    return pd.DataFrame(rows),pd.DataFrame(transfers),vecs


def plot_layer_summary(all_summary,all_transfer,all_axis,root):
    plots=ensure_dir(root/"plots")
    q=all_summary[(all_summary.representation.eq("cls_projected"))&(all_summary.appearance.eq("rgb"))&(all_summary.contrast.eq("add_text"))&(all_summary.stage.ne("CONTENT"))]
    fig,ax=plt.subplots(figsize=(10,5.8))
    for model,g in q.groupby("model"):
        ax.plot(g.stage_index,g.median_loo_angle_deg,marker="o",label=model)
    ax.axhline(90,ls="--",lw=1,color=".5"); ax.set(xlabel="stage (0=ln_pre, 1=B0, ..., 24=B23)",ylabel="median LOO angle (degrees)",title="Emergence of a shared added-text direction in CLS")
    ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(plots/"01_LAYER_TEXT_DIRECTION_LOO_ANGLE.png",dpi=230); plt.close(fig)

    q=all_transfer[(all_transfer.representation.eq("cls_projected"))&(all_transfer.appearance.eq("rgb"))&(all_transfer.test_contrast.eq("txt_minus_vis"))&(all_transfer.stage.ne("CONTENT"))]
    fig,ax=plt.subplots(figsize=(10,5.8))
    for model,g in q.groupby("model"):ax.plot(g.stage_index,g.median_angle_deg,marker="o",label=model)
    ax.axhline(90,ls="--",lw=1,color=".5"); ax.set(xlabel="stage",ylabel="median held-out txt-vis angle to add-text axis",title="Cross-contrast text-provenance transfer backwards through the ViT")
    ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(plots/"02_LAYER_TXTVIS_CROSSCONTRAST_ANGLE.png",dpi=230); plt.close(fig)

    q=all_summary[(all_summary.representation.eq("cls_projected"))&(all_summary.appearance.eq("rgb"))&(all_summary.stage.ne("CONTENT"))&all_summary.contrast.isin(["add_text","add_visual"])]
    fig,ax=plt.subplots(figsize=(10,5.8))
    for (model,con),g in q.groupby(["model","contrast"]):ax.plot(g.stage_index,g.mean_norm,marker="o",label=f"{model}:{con}",ls="-" if con=="add_text" else "--")
    ax.set(xlabel="stage",ylabel="mean controlled displacement norm",title="Text gain vs visual gain across blocks"); ax.legend(fontsize=8,ncol=2); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(plots/"03_LAYER_TEXT_VS_VISUAL_GAIN.png",dpi=230); plt.close(fig)

    q=all_axis[(all_axis.representation.eq("cls_projected"))&(all_axis.appearance.eq("rgb"))&all_axis.angle_text_vs_visual_deg.notna()]
    if len(q):
        fig,ax=plt.subplots(figsize=(10,5.8))
        for model,g in q.groupby("model"):ax.plot(g.stage_index,g.angle_text_vs_visual_deg,marker="o",label=model)
        ax.axhline(90,ls="--",lw=1,color=".5"); ax.set(xlabel="stage",ylabel="angle(add-text, add-visual) degrees",title="Separation of textual and visual evidence axes")
        ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(plots/"04_LAYER_TEXT_VISUAL_AXIS_ANGLE.png",dpi=230); plt.close(fig)


def plot_synthetic(all_syn,root):
    plots=ensure_dir(root/"plots")
    q=all_syn[(all_syn.representation.eq("cls_projected"))&(all_syn.stage.ne("conv1"))]
    fig,axes=plt.subplots(len(q.model.unique()),1,figsize=(10,4.3*max(1,len(q.model.unique()))),squeeze=False)
    for ax,(model,g) in zip(axes[:,0],q.groupby("model")):
        for var,h in g.groupby("variant"):ax.plot(h.stage_index,h.median_angle_deg,marker="o",label=var)
        ax.axhline(90,ls="--",lw=1,color=".5"); ax.set_title(model); ax.set_ylabel("median angle to real add-text axis"); ax.grid(alpha=.15); ax.legend()
    axes[-1,0].set_xlabel("stage"); fig.suptitle("Generalization to unrelated synthetic words / pseudo-words / shuffled glyphs"); fig.tight_layout(); fig.savefig(plots/"05_SYNTHETIC_TEXT_DIRECTION_GENERALIZATION.png",dpi=230); plt.close(fig)


def plot_white_reference(all_transfer,root):
    plots=ensure_dir(root/"plots")
    q=all_transfer[(all_transfer.representation.eq("cls_projected"))&(all_transfer.appearance.eq("rgb"))&(all_transfer.test_contrast.eq("txt_white"))]
    fig,ax=plt.subplots(figsize=(10,5.8))
    for model,g in q.groupby("model"):ax.plot(g.stage_index,g.median_angle_deg,marker="o",label=model)
    ax.axhline(90,ls="--",lw=1,color=".5"); ax.set(xlabel="stage",ylabel="median angle: txt-white to held-out mix-vis axis",title="White-reference test: text direction without the blank/sticker background")
    ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(plots/"06_WHITE_REFERENCE_TEXT_DIRECTION.png",dpi=230); plt.close(fig)


def plot_final_angle_strip(angle_df,content_angles,root):
    plots=ensure_dir(root/"plots"); q=angle_df[(angle_df.appearance.eq("rgb"))&(angle_df.stage.eq("B23"))].copy()
    if content_angles is not None and len(content_angles):q=pd.concat([q,content_angles],ignore_index=True)
    if q.empty:return
    order=list(dict.fromkeys(q.model.astype(str)))
    fig,ax=plt.subplots(figsize=(9,5.8)); rng=np.random.default_rng(123)
    for yi,name in enumerate(order):
        vals=q[q.model.eq(name)].angle_deg.to_numpy(float); x=vals; y=np.full(len(vals),yi)+rng.normal(0,.045,len(vals)); ax.scatter(x,y,s=38,alpha=.75); ax.plot([np.median(vals),np.median(vals)],[yi-.22,yi+.22],lw=3)
    ax.axvline(90,ls="--",lw=1,color=".5"); ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.set_xlabel("held-out txt-vis angle to text axis (degrees; lower = more aligned)")
    ax.set_title("A shared text-provenance direction, concept by concept"); ax.grid(axis="x",alpha=.15); fig.tight_layout(); fig.savefig(plots/"07_FINAL_TEXT_DIRECTION_ANGLE_STRIP.png",dpi=240); plt.close(fig)


def plot_transport(trdf,root):
    plots=ensure_dir(root/"plots")
    for (model,app),g in trdf.groupby(["model","appearance"]):
        if app!="rgb":continue
        p=g.pivot(index="i",columns="j",values="cosine"); fig,ax=plt.subplots(figsize=(7.2,6.5)); im=ax.imshow(p.to_numpy(),vmin=-1,vmax=1,aspect="auto")
        ax.set(xlabel="stage",ylabel="stage",title=f"{model}: text-axis transport across blocks (RGB)"); fig.colorbar(im,ax=ax,label="axis cosine"); fig.tight_layout(); fig.savefig(plots/f"TRANSPORT_{model}.png",dpi=220); plt.close(fig)

# =============================================================================
# Interactive HTML
# =============================================================================

def plotly_available():
    try:
        import plotly.graph_objects as go
        return True
    except Exception:return False


def write_embedding_pca3d(model_name,space,meta,E,outpath):
    try:import plotly.graph_objects as go
    except Exception:return False
    E=normalize_np(E); z,_c,_m,ev=pca_3d(E); d=meta.copy(); d[["pc1","pc2","pc3"]]=z
    fig=go.Figure()
    for cat,g in d.groupby("category"):
        fig.add_trace(go.Scatter3d(x=g.pc1,y=g.pc2,z=g.pc3,mode="markers",name=cat,text=[f"{r.concept} / {r.stim_id}" for r in g.itertuples()],marker=dict(size=4)))
    for concept,g in d.groupby("concept"):
        dd={(r.condition,bool(r.bw)):(r.pc1,r.pc2,r.pc3) for r in g.itertuples()}
        for bw in (False,True):
            if ("mix",bw) in dd:
                for src in ("vis","txt"):
                    if (src,bw) in dd:
                        a=dd[(src,bw)]; b=dd[("mix",bw)]
                        fig.add_trace(go.Scatter3d(x=[a[0],b[0]],y=[a[1],b[1]],z=[a[2],b[2]],mode="lines",showlegend=False,line=dict(width=2),hoverinfo="skip"))
    fig.update_layout(title=f"{model_name}/{space}: final embedding PCA3D — PC variance {ev[0]*100:.1f}%/{ev[1]*100:.1f}%/{ev[2]*100:.1f}%",scene=dict(xaxis_title="PC1",yaxis_title="PC2",zaxis_title="PC3"))
    fig.write_html(str(outpath),include_plotlyjs=True,full_html=True); return True

"""
def write_direction_cone3d(model_name, meta, cache, syn, syn_cache, outpath):
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    # --- modified: plot-size/style controls for readability ---
    REAL_MARKER_SIZE = 10
    SYN_MARKER_SIZE = 5
    AXIS_MARKER_SIZE = 13
    REAL_OPACITY = 0.95
    SYN_OPACITY = 0.72
    AXIS_LINE_WIDTH = 8
    MARKER_LINE_WIDTH = 0.5

    ids = {str(s): i for i, s in enumerate(cache["stim_ids"].astype(str))}
    lk = cell_lookup(meta)
    stage = -1
    real = []
    labels = []

    X = normalize_np(cache["cls_proj"][:, stage, :].astype(np.float64))
    for c in complete_concepts(meta, False):
        d = X[ids[lk[("mix", c, False)]]] - X[ids[lk[("vis", c, False)]]]
        real.append(safe_unit(d))
        labels.append(c)

    real = np.stack(real)
    w = safe_unit(real.mean(0))
    all_u = [real]
    syn_groups = {}

    if syn_cache is not None:
        sid = {str(s): i for i, s in enumerate(syn_cache["stim_ids"].astype(str))}
        SX = normalize_np(syn_cache["cls_proj"][:, stage, :].astype(np.float64))
        for var, g in syn.groupby("variant"):
            uu = []
            labs = []
            for r in g.itertuples(index=False):
                d = SX[sid[str(r.stim_id)]] - X[ids[str(r.base_stim_id)]]
                if np.linalg.norm(d) > EPS:
                    uu.append(safe_unit(d))
                    labs.append(f"{r.base_concept}: {r.text}")
            if uu:
                syn_groups[var] = (np.stack(uu), labs)
                all_u.append(np.stack(uu))

    U = np.concatenate(all_u, 0)
    residual = U - (U @ w)[:, None] * w[None, :]
    _u, _s, vh = np.linalg.svd(residual, full_matrices=False)
    e2 = vh[0]
    e2 = safe_unit(e2 - np.dot(e2, w) * w)
    e3 = vh[1]
    e3 = safe_unit(e3 - np.dot(e3, w) * w - np.dot(e3, e2) * e2)

    def coords(A):
        return np.stack([A @ w, A @ e2, A @ e3], 1)

    fig = go.Figure()

    R = coords(real)
    fig.add_trace(
        go.Scatter3d(
            x=R[:, 0],
            y=R[:, 1],
            z=R[:, 2],
            mode="markers",
            name="matched mix-vis",
            text=labels,
            # --- modified: bigger, clearer real points ---
            marker=dict(
                size=REAL_MARKER_SIZE,
                opacity=REAL_OPACITY,
                line=dict(width=MARKER_LINE_WIDTH),
            ),
            hovertemplate="%{text}<extra></extra>",
        )
    )

    for var, (A, labs) in syn_groups.items():
        C = coords(A)
        fig.add_trace(
            go.Scatter3d(
                x=C[:, 0],
                y=C[:, 1],
                z=C[:, 2],
                mode="markers",
                name=var,
                text=labs,
                # --- modified: synthetic points no longer microscopic ---
                marker=dict(
                    size=SYN_MARKER_SIZE,
                    opacity=SYN_OPACITY,
                    line=dict(width=MARKER_LINE_WIDTH),
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0, 1],
            y=[0, 0],
            z=[0, 0],
            mode="lines+markers",
            name="mean text axis",
            # --- modified: larger axis endpoints for visibility ---
            marker=dict(size=AXIS_MARKER_SIZE, opacity=1.0),
            line=dict(width=AXIS_LINE_WIDTH),
            hovertemplate="mean text axis<extra></extra>",
        )
    )

    # --- modified: slightly cleaner layout / legend ---
    fig.update_layout(
        title=f"{model_name}: unit-vector text-direction cone (x = exact cosine to real mean text axis)",
        scene=dict(
            xaxis_title="cos(angle to text axis)",
            yaxis_title="residual PC1",
            zaxis_title="residual PC2",
        ),
        legend=dict(itemsizing="constant"),
    )

    fig.write_html(str(outpath), include_plotlyjs=True, full_html=True)
    return True
"""

def write_direction_cone3d(model_name, meta, cache, syn, syn_cache, outpath):
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    # --- modified: readability controls ---
    REAL_MARKER_SIZE = 10
    SYN_MARKER_SIZE = 5
    AXIS_MARKER_SIZE = 13
    CENTROID_MARKER_SIZE = 14
    REAL_OPACITY = 0.95
    SYN_OPACITY = 0.72
    CONE_OPACITY = 0.16
    AXIS_LINE_WIDTH = 8
    MARKER_LINE_WIDTH = 0.5

    # --- modified: helper for the translucent cone surface ---
    def build_cone_surface(R, n_x=18, n_theta=48, radius_q=0.85, min_per_bin=2):
        if R.shape[0] < 4:
            return None

        x = R[:, 0]
        rho = np.sqrt(R[:, 1] ** 2 + R[:, 2] ** 2)

        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmax <= xmin + EPS:
            return None

        edges = np.linspace(xmin, xmax, n_x + 1)
        xc = []
        rc = []

        for i in range(n_x):
            lo = edges[i]
            hi = edges[i + 1]
            if i == n_x - 1:
                m = (x >= lo) & (x <= hi)
            else:
                m = (x >= lo) & (x < hi)

            if int(m.sum()) >= min_per_bin:
                xc.append(float(np.mean(x[m])))
                rc.append(float(np.quantile(rho[m], radius_q)))

        if len(xc) < 3:
            return None

        xc = np.asarray(xc, dtype=np.float64)
        rc = np.asarray(rc, dtype=np.float64)

        # --- modified: light smoothing to avoid ugly jagged shells ---
        if len(rc) >= 5:
            rc = np.convolve(rc, np.array([0.25, 0.5, 0.25]), mode="same")
            rc[0] = rc[1]
            rc[-1] = rc[-2]

        # --- modified: anchor the envelope at the origin to make the cone legible ---
        xc = np.concatenate([[0.0], xc])
        rc = np.concatenate([[0.0], rc])

        theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
        ct = np.cos(theta)
        st = np.sin(theta)

        Xs = np.repeat(xc[:, None], n_theta, axis=1)
        Ys = rc[:, None] * ct[None, :]
        Zs = rc[:, None] * st[None, :]

        return Xs, Ys, Zs

    ids = {str(s): i for i, s in enumerate(cache["stim_ids"].astype(str))}
    lk = cell_lookup(meta)
    stage = -1
    real = []
    labels = []

    X = normalize_np(cache["cls_proj"][:, stage, :].astype(np.float64))
    for c in complete_concepts(meta, False):
        d = X[ids[lk[("mix", c, False)]]] - X[ids[lk[("vis", c, False)]]]
        real.append(safe_unit(d))
        labels.append(c)

    real = np.stack(real)
    w = safe_unit(real.mean(0))
    all_u = [real]
    syn_groups = {}

    if syn_cache is not None:
        sid = {str(s): i for i, s in enumerate(syn_cache["stim_ids"].astype(str))}
        SX = normalize_np(syn_cache["cls_proj"][:, stage, :].astype(np.float64))
        for var, g in syn.groupby("variant"):
            uu = []
            labs = []
            for r in g.itertuples(index=False):
                d = SX[sid[str(r.stim_id)]] - X[ids[str(r.base_stim_id)]]
                if np.linalg.norm(d) > EPS:
                    uu.append(safe_unit(d))
                    labs.append(f"{r.base_concept}: {r.text}")
            if uu:
                syn_groups[var] = (np.stack(uu), labs)
                all_u.append(np.stack(uu))

    U = np.concatenate(all_u, 0)
    residual = U - (U @ w)[:, None] * w[None, :]
    _u, _s, vh = np.linalg.svd(residual, full_matrices=False)
    e2 = vh[0]
    e2 = safe_unit(e2 - np.dot(e2, w) * w)
    e3 = vh[1]
    e3 = safe_unit(e3 - np.dot(e3, w) * w - np.dot(e3, e2) * e2)

    def coords(A):
        return np.stack([A @ w, A @ e2, A @ e3], 1)

    fig = go.Figure()

    R = coords(real)

    # --- modified: translucent cone/envelope around the real text-direction cloud ---
    cone = build_cone_surface(R)
    if cone is not None:
        Xs, Ys, Zs = cone
        fig.add_trace(
            go.Surface(
                x=Xs,
                y=Ys,
                z=Zs,
                name="text-direction envelope",
                showscale=False,
                opacity=CONE_OPACITY,
                colorscale=[[0.0, "orange"], [1.0, "orange"]],
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=R[:, 0],
            y=R[:, 1],
            z=R[:, 2],
            mode="markers",
            name="matched mix-vis",
            text=labels,
            marker=dict(
                size=REAL_MARKER_SIZE,
                opacity=REAL_OPACITY,
                line=dict(width=MARKER_LINE_WIDTH),
            ),
            hovertemplate="%{text}<extra></extra>",
        )
    )

    syn_coords = {}
    for var, (A, labs) in syn_groups.items():
        C = coords(A)
        syn_coords[var] = (C, labs)
        fig.add_trace(
            go.Scatter3d(
                x=C[:, 0],
                y=C[:, 1],
                z=C[:, 2],
                mode="markers",
                name=var,
                text=labs,
                marker=dict(
                    size=SYN_MARKER_SIZE,
                    opacity=SYN_OPACITY,
                    line=dict(width=MARKER_LINE_WIDTH),
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # --- modified: highlighted concept centroids ---
    concept_points = {}
    for p, c in zip(R, labels):
        concept_points.setdefault(c, []).append(p)

    for var, (C, labs) in syn_coords.items():
        for p, lab in zip(C, labs):
            base_concept = lab.split(":", 1)[0]
            concept_points.setdefault(base_concept, []).append(p)

    centroid_xyz = []
    centroid_labels = []
    for c in sorted(concept_points.keys()):
        pts = np.stack(concept_points[c], 0)
        centroid_xyz.append(np.mean(pts, axis=0))
        centroid_labels.append(c)

    centroid_xyz = np.stack(centroid_xyz, 0)
    fig.add_trace(
        go.Scatter3d(
            x=centroid_xyz[:, 0],
            y=centroid_xyz[:, 1],
            z=centroid_xyz[:, 2],
            mode="markers",
            name="concept centroids",
            text=centroid_labels,
            marker=dict(
                size=CENTROID_MARKER_SIZE,
                symbol="diamond",
                opacity=0.98,
                color="white",
                line=dict(width=3, color="black"),
            ),
            hovertemplate="concept centroid: %{text}<extra></extra>",
        )
    )

    # --- modified: overall centroid of the real cloud ---
    real_centroid = np.mean(R, axis=0)
    fig.add_trace(
        go.Scatter3d(
            x=[real_centroid[0]],
            y=[real_centroid[1]],
            z=[real_centroid[2]],
            mode="markers",
            name="real-cloud centroid",
            marker=dict(
                size=CENTROID_MARKER_SIZE + 3,
                symbol="x",
                opacity=1.0,
                color="black",
                line=dict(width=2, color="black"),
            ),
            hovertemplate="real matched mix-vis centroid<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[0, 1],
            y=[0, 0],
            z=[0, 0],
            mode="lines+markers",
            name="mean text axis",
            marker=dict(size=AXIS_MARKER_SIZE, opacity=1.0),
            line=dict(width=AXIS_LINE_WIDTH),
            hovertemplate="mean text axis<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{model_name}: unit-vector text-direction cone (x = exact cosine to real mean text axis)",
        scene=dict(
            xaxis_title="cos(angle to text axis)",
            yaxis_title="residual PC1",
            zaxis_title="residual PC2",
        ),
        legend=dict(itemsizing="constant"),
    )

    fig.write_html(str(outpath), include_plotlyjs=True, full_html=True)
    return True


# =============================================================================
# Reports / compact handoff
# =============================================================================

def write_report(root,summary,transfer,syn,conv1):
    lines=["# Visual/textual text-direction trajectory atlas","",
           "The primary paper-facing statistic is a leave-one-concept-out angular test:",
           "the text axis is learned from mix-vis vectors of the other concepts, then tested on the held-out txt-vis vector.",""]
    q=transfer[(transfer.representation.eq("cls_projected"))&(transfer.appearance.eq("rgb"))&(transfer.stage.isin(["B23","CONTENT"]))&(transfer.test_contrast.eq("txt_minus_vis"))]
    lines += ["## Final cross-contrast angles"]
    for r in q.itertuples(index=False):lines.append(f"- {r.model}/{r.stage}: positive={r.positive_fraction:.3f}, median={r.median_angle_deg:.2f} deg, q90={r.q90_angle_deg:.2f} deg")
    lines += ["","## Final add-text directional concentration"]
    q=summary[(summary.representation.eq("cls_projected"))&(summary.appearance.eq("rgb"))&(summary.stage.isin(["B23","CONTENT"]))&(summary.contrast.eq("add_text"))]
    for r in q.itertuples(index=False):lines.append(f"- {r.model}/{r.stage}: R={r.resultant_length:.4f} [{r.bootstrap_R_lo:.4f},{r.bootstrap_R_hi:.4f}], signflip p={r.signflip_p:.6g}, median LOO angle={r.median_loo_angle_deg:.2f} deg")
    if len(syn):
        lines += ["","## Synthetic validation at B23"]
        q=syn[(syn.representation.eq("cls_projected"))&(syn.stage.eq("B23"))]
        for r in q.itertuples(index=False):lines.append(f"- {r.model}/{r.variant}: positive={r.positive_fraction:.3f}, median angle={r.median_angle_deg:.2f} deg")
    if len(conv1):
        lines += ["","## Conv1 white-reference transfer"]
        q=conv1[conv1.measurement.eq("add_text_to_txt_white_transfer")]
        for r in q.itertuples(index=False):lines.append(f"- {r.model}/{r.representation}/{r.appearance}: positive={r.positive_fraction:.3f}, median angle={r.median_angle_deg:.2f} deg")
    lines += ["","## Interpretation guardrails",
              "- PCA/3D HTML is visualization only; the angular tests use the original dimensions.",
              "- 0 degrees means parallel, 90 degrees means orthogonal/no signed transfer, >90 means opposite.",
              "- The sign-flip null preserves every observed vector and randomizes only common orientation.",
              "- White-reference txt-white removes the blank canvas contribution, but nonlinear transformer interactions can still differ from mix-vis.",
              "- Synthetic pseudo-words test glyph/text provenance independently of object semantics; shuffled glyph tiles are a high-frequency visual control, not a perfect matched psychophysical control."]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def compact_zip(root):
    zpath=root/"compact_summary_conv1_visualtextual_text_direction.zip"
    exclude={"real_stage_cache.npz","synthetic_stage_cache.npz","white_stage_cache.npz"}
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p == zpath or "SYNTHETIC_INPUTS" in p.parts:continue
            if p.name in exclude or p.stat().st_size>30_000_000:continue
            z.write(p,arcname=str(p.relative_to(root)))
    return zpath

# =============================================================================
# Main
# =============================================================================

def parse_args(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--image_dir",default="image_sets/visualtextual")
    ap.add_argument("--output_dir",default="out_paper_reproduction/conv1/visualtextual_text_direction")
    ap.add_argument("--models",default="pretrained,gmp,full_xattn")
    ap.add_argument("--repo_root",default="")
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--batch_size",type=int,default=12)
    ap.add_argument("--seed",type=int,default=20260918)
    ap.add_argument("--bootstrap",type=int,default=1000)
    ap.add_argument("--signflip",type=int,default=4000)
    ap.add_argument("--synthetic_per_base",type=int,default=8,help="unrelated overlay groups per RGB visual base (25 bases by default)")
    ap.add_argument("--synthetic_shuffle_tile",type=int,default=8)
    ap.add_argument("--pretrained_model",default="openai/clip-vit-large-patch14")
    ap.add_argument("--gmp_checkpoint",default="zer0int/CLIP-GmP-ViT-L-14")
    ap.add_argument("--xattn_model",default="zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX")
    ap.add_argument("--xattn_revision",default="")
    ap.add_argument("--hf_cache_dir",default="")
    ap.add_argument("--self_test",action="store_true")
    return ap.parse_args(argv)


def self_test():
    D=np.asarray([[1,.1,0],[1,-.1,0],[1,.05,.1],[1,-.02,-.1]],float)
    st,ang,null=direction_statistics(D,1,100,300)
    assert st["resultant_length"]>.98
    assert st["median_loo_angle_deg"]<15
    T=np.asarray([[1,0,0],[1,.1,0],[1,-.1,0],[1,.05,.05]],float)
    tr,a=transfer_statistics(D,T); assert tr["positive_fraction"]==1
    rng=np.random.default_rng(1); pw=pseudo_word(rng); assert isinstance(pw,str) and 5<=len(pw)<=10
    # Tile shuffle preserves RGBA dimensions and total pixel sum.
    layer=Image.new("RGBA",(32,32),(0,0,0,0)); ImageDraw.Draw(layer).text((2,2),"TEST",fill=(255,0,0,255))
    sh=tile_shuffle_layer(layer,rng,8); assert sh.size==layer.size; assert np.asarray(sh).sum()==np.asarray(layer).sum()

    # Dataset regression: legacy/unpaired extras such as mix_adv_* must not
    # enter the paired text-direction forward cache or canonical-mask lookup.
    pair_rows=[]
    for cond in ("vis","txt","mix"):
        pair_rows.append({"stim_id":f"{cond}_avocado","concept":"avocado","condition":cond,"bw":False})
    pair_rows.append({"stim_id":"mix_adv_avocado","concept":"adv_avocado","condition":"mix","bw":False})
    pm,px=complete_pair_manifest(pd.DataFrame(pair_rows))
    assert list(pm.stim_id)==["vis_avocado","txt_avocado","mix_avocado"]
    assert list(px.stim_id)==["mix_adv_avocado"]

    # Integration regression: run the real-layer analyzer against the exact
    # serialized cache schema, including the ``cls_proj`` key that previously
    # failed only after the expensive forward cache had completed.
    import tempfile
    from types import SimpleNamespace
    concepts=("cat","trout","rose")
    rows=[]
    for c in concepts:
        for bw in (False,True):
            for cond in ("vis","txt","mix"):
                rows.append({"stim_id":f"{cond}_{c}{'_bw' if bw else ''}","concept":c,"condition":cond,"bw":bw,"category":f"{cond}_{'bw' if bw else 'rgb'}"})
    meta=pd.DataFrame(rows)
    N=len(meta); S=2; Draw=7; Dproj=5; P=4
    rr=np.random.default_rng(7)
    # Give the paired contrasts a shared direction so directional statistics are defined.
    raw=rr.normal(scale=.05,size=(N,S,Draw)); proj=rr.normal(scale=.05,size=(N,S,Dproj))
    spatial=rr.normal(scale=.05,size=(N,S,Draw)); textloc=spatial.copy(); objloc=spatial.copy(); bg=spatial.copy()
    idx={sid:i for i,sid in enumerate(meta.stim_id.astype(str))}
    traw=np.zeros(Draw); traw[0]=1.; tproj=np.zeros(Dproj); tproj[0]=1.
    for r in meta.itertuples(index=False):
        if r.condition in ("txt","mix"):
            raw[idx[r.stim_id]] += .4*traw; spatial[idx[r.stim_id]] += .3*traw
            textloc[idx[r.stim_id]] += .5*traw; bg[idx[r.stim_id]] += .2*traw
            proj[idx[r.stim_id]] += .4*tproj
    cache={"stim_ids":np.asarray(meta.stim_id.astype(str)),"stage_labels":np.asarray(["ln_pre","B00"]),
           "cls_raw":raw.astype(np.float16),"cls_proj":proj.astype(np.float16),
           "spatial_mean":spatial.astype(np.float16),"textloc_mean":textloc.astype(np.float16),
           "objectloc_mean":objloc.astype(np.float16),"background_mean":bg.astype(np.float16)}
    white={"cls_raw":np.zeros((S,Draw),np.float16),"cls_proj":np.zeros((S,Dproj),np.float16),
           "spatial_tokens":np.zeros((S,P,Draw),np.float16)}
    canonical={sid:{"text":np.array([1,0,0,0],bool),"object":np.array([0,1,0,0],bool),"background":np.array([0,0,1,1],bool)} for sid in meta.stim_id.astype(str)}
    aargs=SimpleNamespace(bootstrap=20,signflip=40)
    with tempfile.TemporaryDirectory() as td:
        out=analyze_real_layers("selftest",meta,cache,white,canonical,aargs,Path(td))
        assert len(out[0])>0 and (out[0]["representation"]=="cls_projected").any()
        assert (Path(td)/"layer_text_direction_summary.csv").is_file()

    # Regression: synthetic CLS at ln_pre is image-independent, so the
    # synthetic-minus-base displacement is exactly zero.  That is an undefined
    # direction, not a 90-degree direction and certainly not a reason to crash.
    ez=_angle_summary_from_cosines([],[],[])
    assert ez["n"]==0 and np.isnan(ez["median_angle_deg"]) and np.isnan(ez["q90_angle_deg"])
    ztrain=np.zeros((3,5)); ztest=np.ones((3,5))
    ztr,zang=transfer_statistics(ztrain,ztest)
    assert ztr["n"]==0 and len(zang)==0 and np.isnan(ztr["median_angle_deg"])
    print("self-test OK")


def main(argv=None):
    args=parse_args(argv)
    if args.self_test:self_test(); return 0
    seed_all(args.seed); root=ensure_dir(Path(args.output_dir)); ensure_dir(root/"plots"); ensure_dir(root/"interactive")
    parsed_meta=parse_dataset(Path(args.image_dir))
    parsed_meta.to_csv(root/"image_manifest_all.csv",index=False)
    meta,excluded_meta=complete_pair_manifest(parsed_meta)
    meta.to_csv(root/"image_manifest.csv",index=False)
    if len(excluded_meta):
        excluded_meta.to_csv(root/"excluded_unpaired_images.csv",index=False)
        excluded_ids=list(excluded_meta.stim_id.astype(str))
        preview=", ".join(excluded_ids[:8]) + (" ..." if len(excluded_ids)>8 else "")
        print(f"[dataset] ignoring {len(excluded_ids)} unpaired image(s) not part of a complete vis/txt/mix triplet: {preview}")
    print(f"[dataset] paired text-direction manifest: {len(meta)}/{len(parsed_meta)} images")
    runtime=import_runtime(args.repo_root)
    model_names=parse_models(args.models)

    all_summary=[]; all_transfer=[]; all_axis=[]; all_angles=[]; all_transport=[]; all_syn=[]; all_conv1=[]; content_angle_rows=[]
    syn_manifest=None
    for model_name in model_names:
        print("\n"+"="*88+f"\nMODEL {model_name}\n"+"="*88)
        md=ensure_dir(root/model_name); ensure_dir(md/"plots"); ensure_dir(md/"interactive")
        variant=load_variant(model_name,args,runtime); size,G,patch,width,nblocks,nheads=model_geometry(variant)
        input_masks,_maskdf=build_input_region_masks(meta,size,G)
        canonical=canonical_region_masks(meta,input_masks)
        cache=collect_real_stage_cache(variant,meta,canonical,args,md); white=collect_white_stage(variant,args,md)

        summary,transfer,axis,angles,transport,axis_store,nulls=analyze_real_layers(model_name,meta,cache,white,canonical,args,md)
        conv1=analyze_conv1_white_reference(model_name,meta,cache,white,canonical,args,md)
        all_summary.append(summary); all_transfer.append(transfer); all_axis.append(axis); all_angles.append(angles); all_transport.append(transport); all_conv1.append(conv1)

        # CONTENT endpoint for full x-attn.
        csum,ctrans,cvec=final_content_metrics(model_name,meta,cache,args,md)
        if len(csum): all_summary.append(csum)
        if len(ctrans):
            all_transfer.append(ctrans)
            for app,d in cvec.items():
                if app=="rgb":
                    for c,a in zip(d["concepts"],d["angles"]):content_angle_rows.append({"model":f"{model_name}/CONTENT","appearance":app,"stage":"CONTENT","stage_index":25,"concept":c,"angle_deg":float(a)})

        # Synthetic bank is shared across models and built from RGB visual cutouts.
        if syn_manifest is None:
            syn_manifest=generate_synthetic_bank(meta,size,args,root)
        syn_masks=_synthetic_masks(syn_manifest,meta,size,G)
        syn_cache=collect_synthetic_stage_cache(variant,syn_manifest,syn_masks,args,md)
        synres=analyze_synthetic_transfer(model_name,meta,cache,syn_manifest,syn_cache,axis_store,args,md); all_syn.append(synres)

        # Model-specific interactive visualizations.
        write_embedding_pca3d(model_name,"backbone",meta,cache["final_backbone"],md/"interactive"/"FINAL_EMBEDDING_PCA3D_BACKBONE.html")
        write_direction_cone3d(model_name,meta,cache,syn_manifest,syn_cache,md/"interactive"/"TEXT_DIRECTION_CONE3D.html")
        if "content" in cache:write_embedding_pca3d(model_name,"content",meta,cache["content"],md/"interactive"/"FINAL_EMBEDDING_PCA3D_CONTENT.html")

        # Copy model result tables into a short report directory already implied by md.
        del variant,cache,syn_cache; gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

    S=pd.concat(all_summary,ignore_index=True); T=pd.concat(all_transfer,ignore_index=True); A=pd.concat(all_axis,ignore_index=True); ANG=pd.concat(all_angles,ignore_index=True); TR=pd.concat(all_transport,ignore_index=True); SYN=pd.concat(all_syn,ignore_index=True); C1=pd.concat(all_conv1,ignore_index=True)
    S.to_csv(root/"ALL_MODELS_layer_text_direction_summary.csv",index=False)
    T.to_csv(root/"ALL_MODELS_layer_crosscontrast_transfer.csv",index=False)
    A.to_csv(root/"ALL_MODELS_layer_axis_relationships.csv",index=False)
    ANG.to_csv(root/"ALL_MODELS_final_direction_angles.csv",index=False)
    TR.to_csv(root/"ALL_MODELS_layer_direction_transport.csv",index=False)
    SYN.to_csv(root/"ALL_MODELS_synthetic_text_transfer.csv",index=False)
    C1.to_csv(root/"ALL_MODELS_conv1_white_reference.csv",index=False)
    content_angles=pd.DataFrame(content_angle_rows)

    plot_layer_summary(S,T,A,root); plot_synthetic(SYN,root); plot_white_reference(T,root); plot_final_angle_strip(ANG,content_angles,root); plot_transport(TR,root)
    write_report(root,S,T,SYN,C1)
    for n in ("probe_VISUALTEXTUAL_TEXT_DIRECTION_TRAJECTORY.py","visualtextual_probe_common.py"):
        p=Path(__file__).resolve().parent/n
        if p.is_file():shutil.copy2(p,root/n)
    z=compact_zip(root); print("\nDONE\nCompact handoff:",z)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
