#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vanilla CLIP Conv1 functional atlas for GmP / bare-xattn backbones, with optional RN token.

Purpose
-------
Repeat the ModeMUX Conv1 functional-atlas analysis on vanilla attnclip_mechinterp_sae
vision backbones so the source of the x-attn robustness change can be localized.

Model variants
--------------
  pretrained   openai/clip-vit-large-patch14 loaded through attnclip_mechinterp_sae.
  gmp          zer0int/CLIP-GmP-ViT-L-14 (configurable HF/local source).
  bare_xattn   A vanilla ViT-L/14 whose *visual* state is loaded from the full
               ModeMUX/x-attn checkpoint; bridge, CONTENT correction, hard-text
               controls, and READ_NULL architecture keys are intentionally ignored.

Optional RN-only intervention
-----------------------------
With --use_rn_token, the learned visual.read_null_token from the x-attn checkpoint
is appended immediately before block --rn_insert_block (default 13) and remains the
last token thereafter. Nothing else from the bridge is attached. This isolates the
causal effect of the trained RN token on the vanilla backbone.

Default causal interventions are FLIP and SHUFFLE. They are configurable with
--conditions. Outputs are automatically separated as:
  <out_root>/normal/<model>/
  <out_root>/rn_token/<model>/

The analysis mirrors the x-attn functional atlas where applicable:
  * static Conv1 morphology, natural activation, positional dominance, low-tail bins;
  * global all-channel screen -> candidate union -> full candidate scan;
  * early CLS scanner, B7-9 incoming routing, frozen pre-B13 register geometry,
    B22 CLS-Q, final implicit-register geometry, per-head CLS<->REG signatures;
  * exact severe/statistical culprit events, raw/delta/nonlinear-residual manifolds;
  * same-image stacks, condition-family stacks, cross-condition pair screen;
  * focus plots for channels 779,720,866,151 by default;
  * CULPRIT_SHORTLIST.txt + experiment_candidates.json.

RN-only runs additionally save per-head RN attention and projected V-write telemetry:
  CLS->RN, RN->CLS, all-query->RN, projected RN V norm, and attention-weighted
  RN->CLS/all-query write norms. These are intended as diagnostics, not proof that
  the token is the causal explanation by themselves.

All vector artifacts use safetensors. No pickle output is written.
"""
from __future__ import annotations

import argparse, contextlib, gc, hashlib, json, math, os, random, re, sys
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

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY=True
except Exception:
    linear_sum_assignment=None; HAVE_SCIPY=False
try:
    import umap  # type: ignore
    HAVE_UMAP=True
except Exception:
    umap=None; HAVE_UMAP=False
try:
    from safetensors.torch import save_file as _save_st, load_file as _load_st
except Exception as exc:
    raise RuntimeError("safetensors is required: pip install safetensors") from exc

EPS=1e-12
SEED=20260917
DEFAULT_IMAGE_DIR=r"image_sets/special_natural"
DEFAULT_OUT_ROOT=r"out_paper_reproduction/conv1/vanilla_functional_atlas"
DEFAULT_PRETRAINED="openai/clip-vit-large-patch14"
DEFAULT_GMP="zer0int/CLIP-GmP-ViT-L-14"
DEFAULT_XATTN="zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_CONDITIONS=("FLIP","SHUFFLE")
LOW_POPS=("sparse_tail","transition_band")
FOCUS_DEFAULT=(779,720,866,151)

MORPH_FEATURES=(
    "lum_energy_frac","rg_energy_frac","by_energy_frac","dc_frac","low_frac","mid_frac","high_frac",
    "spectral_centroid","spectral_bandwidth","spectral_entropy","radial_peak","ring_frac",
    "orientation_anisotropy","freq_axis_cos2","freq_axis_sin2","vertical_axis_frac","horizontal_axis_frac",
    "corner_freq_frac","symmetry180","symmetry_lr","symmetry_ud","center_spatial_frac","edge_spatial_frac",
    "dog_abs_corr","gabor_abs_corr",
)
TWIN_FEATURE_DOMAINS={
    "color":("lum_energy_frac","rg_energy_frac","by_energy_frac"),
    "frequency":("dc_frac","low_frac","mid_frac","high_frac","spectral_centroid","spectral_bandwidth","spectral_entropy","radial_peak","ring_frac","orientation_anisotropy","freq_axis_cos2","freq_axis_sin2","vertical_axis_frac","horizontal_axis_frac","corner_freq_frac"),
    "symmetry":("symmetry180","symmetry_lr","symmetry_ud"),
    "spatial":("center_spatial_frac","edge_spatial_frac"),
    "shape":("dog_abs_corr","gabor_abs_corr"),
}

# ----------------------------- utilities ------------------------------------
def save_safetensors(tensors:Mapping[str,torch.Tensor],filename:str,metadata:Optional[Mapping[str,str]]=None):
    packed={k:v.detach().contiguous() for k,v in tensors.items()}
    _save_st(packed,filename,metadata=None if metadata is None else {str(k):str(v) for k,v in metadata.items()})

def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed%(2**32-1));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def safe_name(x):return re.sub(r"[^A-Za-z0-9._-]+","_",str(x)).strip("_") or "item"
def stable_seed(text):return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8],16)
def parse_int_list(s):return [int(x.strip()) for x in str(s).split(",") if x.strip()]
def parse_str_list(s):return [x.strip().upper() for x in str(s).split(",") if x.strip()]
def normalize_np(x,axis=-1):
    x=np.asarray(x,np.float32);return x/np.maximum(np.linalg.norm(x,axis=axis,keepdims=True),EPS)
def cosine_np(a,b):
    a=np.asarray(a,np.float64).reshape(-1);b=np.asarray(b,np.float64).reshape(-1);d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d>EPS else float("nan")
def cosine_distance_np(a,b):return 1.0-cosine_np(a,b)
def cosine_dist_rows_torch(a,b):return 1-F.cosine_similarity(a.float(),b.float(),dim=-1,eps=EPS)
def robust_z(x):
    x=np.asarray(x,float);m=np.nanmedian(x);mad=np.nanmedian(np.abs(x-m));s=1.4826*mad
    return (x-m)/(s if s>EPS else (np.nanstd(x) if np.nanstd(x)>EPS else 1.0))
def robust_log_z(norms):
    q=np.log(np.maximum(np.asarray(norms,float),EPS));m=float(np.median(q));s=float(1.4826*np.median(np.abs(q-m)));s=s if s>EPS else float(np.std(q)+EPS);return (q-m)/s,m,s
def percentile01(s):
    x=pd.to_numeric(s,errors="coerce");return x.rank(method="average",pct=True).fillna(0.0) if x.notna().sum()>1 else pd.Series(np.zeros(len(x)),index=s.index)
def population_group(z):return "sparse_tail" if z<-3 else ("transition_band" if z<-1 else ("controls" if z>=1 else "middle"))
def write_df(df,path,index=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix==".gz":df.to_csv(path,index=index,compression="gzip")
    elif path.suffix==".parquet":df.to_parquet(path,index=index)
    else:df.to_csv(path,index=index)
def choose_stride_subset(df,n):
    if n<=0 or n>=len(df):return df.copy().reset_index(drop=True)
    idx=np.linspace(0,len(df)-1,n,dtype=int);return df.iloc[np.unique(idx)].reset_index(drop=True)
def source_from_stem(stem):return "adv" if stem.endswith("_adv") else "clean"
def pair_from_stem(stem):return stem[:-4] if stem.endswith("_adv") else stem

def scan_images(image_dir:Path,recursive=False):
    exts={".png",".jpg",".jpeg",".webp",".bmp"};it=image_dir.rglob("*") if recursive else image_dir.iterdir();fs=sorted([p.resolve() for p in it if p.is_file() and p.suffix.lower() in exts],key=lambda p:p.name.lower())
    if not fs:raise RuntimeError(f"No images found in {image_dir}")
    rows=[];seen={}
    for p in fs:
        base=p.stem;n=seen.get(base,0);seen[base]=n+1;sid=base if n==0 else f"{base}__dup{n}";rows.append({"stim_id":sid,"filename":p.name,"path":str(p),"source":source_from_stem(base),"pair":pair_from_stem(base)})
    return pd.DataFrame(rows)

def find_repo_root(start:Optional[Path]=None):
    starts=[Path.cwd().resolve(),Path(__file__).resolve().parent]
    if start is not None:starts.insert(0,start.resolve())
    for r0 in starts:
        for p in [r0,*r0.parents]:
            if (p/"attnclip_mechinterp_sae").is_dir() and (p/"utils_clip_loader").is_dir():return p
    raise FileNotFoundError("Could not locate repo root containing attnclip_mechinterp_sae and utils_clip_loader; pass --repo_root")

def amp_context(device,enabled):
    return torch.autocast("cuda",dtype=torch.float16) if enabled and str(device).startswith("cuda") and torch.cuda.is_available() else contextlib.nullcontext()

def load_batch(preprocess,rows,device):
    ims=[];ids=[];src=[]
    for r in rows.itertuples(index=False):
        with Image.open(r.path) as im:ims.append(preprocess(ImageOps.exif_transpose(im).convert("RGB")))
        ids.append(str(r.stim_id));src.append(str(r.source))
    return torch.stack(ims,0).to(device,non_blocking=True),ids,src

# ----------------------------- model loading --------------------------------
def _import_runtime(args):
    repo=Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    if str(repo) not in sys.path:sys.path.insert(0,str(repo))
    import attnclip_mechinterp_sae as clip
    from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything,resolve_to_openai_state_dict
    return repo,clip,load_openai_clip_anything,resolve_to_openai_state_dict

def _freeze(model):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

def _resolve_xattn_visual_state(clip,resolve_fn,args):
    sd,info=resolve_fn(args.xattn_model,cache_dir=(args.hf_cache_dir or None),revision=(args.xattn_revision or None),allow_unsafe_hf_pickle=False)
    conv=getattr(getattr(clip,"model",None),"convert_state_dict_inproj_to_qkv",None)
    if callable(conv):sd=conv(sd)
    return sd,info

def _extract_rn(sd,args):
    key="visual.read_null_token"
    if key not in sd:raise KeyError(f"x-attn checkpoint has no {key}")
    token=sd[key].detach().float().cpu().reshape(-1).contiguous()
    source_block=None
    if "visual.read_null_insert_block_config" in sd:
        source_block=int(sd["visual.read_null_insert_block_config"].item())
    block=int(args.rn_insert_block)
    return token,source_block,block

def _load_bare_xattn(clip,load_any,resolve_fn,args,out):
    # Intentionally instantiate a vanilla model, then copy only exact-shape visual keys.
    model,preprocess,_=load_any(clip,args.pretrained_model,device=args.device,jit=False,strict=True,allow_unsafe_hf_pickle=False)
    sd,info=_resolve_xattn_visual_state(clip,resolve_fn,args)
    target=model.state_dict();filtered={};ignored=[];shape_mismatch=[]
    for k,v in sd.items():
        if not k.startswith("visual."):
            ignored.append(k);continue
        if k in {"visual.read_null_token","visual.read_null_insert_block_config"}:
            ignored.append(k);continue
        if k not in target:
            ignored.append(k);continue
        if tuple(v.shape)!=tuple(target[k].shape):
            shape_mismatch.append({"key":k,"source_shape":list(v.shape),"target_shape":list(target[k].shape)});continue
        filtered[k]=v.to(dtype=target[k].dtype)
    missing_visual=sorted(k for k in target if k.startswith("visual.") and k not in filtered)
    incompatible=model.load_state_dict(filtered,strict=False)
    audit={"source":args.xattn_model,"loaded_visual_keys":len(filtered),"ignored_keys":sorted(ignored),"shape_mismatch":shape_mismatch,"missing_visual_keys":missing_visual,"load_missing":list(incompatible.missing_keys),"load_unexpected":list(incompatible.unexpected_keys),"loader_info":str(info)}
    (out/"bare_xattn_load_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    (out/"bare_xattn_unexpected_keys.txt").write_text("\n".join(sorted(ignored))+"\n",encoding="utf-8")
    # Vision-only analysis: a missing visual key is a hard failure.
    if missing_visual:raise RuntimeError(f"bare_xattn missing {len(missing_visual)} vanilla visual keys; see bare_xattn_load_audit.json")
    return _freeze(model),preprocess,{"kind":"bare_xattn","source":args.xattn_model,"loaded_visual_keys":len(filtered)}

def load_model_variant(name,args,out):
    repo,clip,load_any,resolve_fn=_import_runtime(args);name=str(name).lower()
    if name=="pretrained":
        model,preprocess,li=load_any(clip,args.pretrained_model,device=args.device,jit=False,strict=True,allow_unsafe_hf_pickle=False);info={"kind":"pretrained","source":args.pretrained_model,"loader_info":str(li)}
    elif name=="gmp":
        try:
            model,preprocess,li=load_any(clip,args.gmp_checkpoint,device=args.device,jit=False,strict=True,reuse_full_model_pickle=False)
            info={"kind":"gmp","source":args.gmp_checkpoint,"loader_info":str(li),"load_mode":"state_dict_rebuild"}
        except Exception as first_error:
            # Trusted local checkpoint fallback: reuse the pickle only as a source of visual
            # weights, then transplant those weights into a fresh attnclip_mechinterp_sae
            # ViT-L/14.  This still guarantees the analysis runtime is the vanilla module.
            src,_pp,li=load_any(clip,args.gmp_checkpoint,device="cpu",jit=False,strict=True,reuse_full_model_pickle=True)
            model,preprocess,_=load_any(clip,args.pretrained_model,device=args.device,jit=False,strict=True,allow_unsafe_hf_pickle=False)
            srcsd=src.state_dict();conv=getattr(getattr(clip,"model",None),"convert_state_dict_inproj_to_qkv",None)
            if callable(conv):srcsd=conv(srcsd)
            tgt=model.state_dict();filt={k:v.to(dtype=tgt[k].dtype) for k,v in srcsd.items() if k.startswith("visual.") and k in tgt and tuple(v.shape)==tuple(tgt[k].shape)}
            miss=sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
            if miss:raise RuntimeError(f"GmP fallback could not populate {len(miss)} visual keys: {miss[:12]}") from first_error
            model.load_state_dict(filt,strict=False)
            (out/"gmp_load_fallback_audit.json").write_text(json.dumps({"first_error":repr(first_error),"loaded_visual_keys":len(filt),"loader_info":str(li)},indent=2),encoding="utf-8")
            del src;gc.collect()
            info={"kind":"gmp","source":args.gmp_checkpoint,"loader_info":str(li),"load_mode":"trusted_pickle_visual_transplant"}
    elif name=="bare_xattn":
        model,preprocess,info=_load_bare_xattn(clip,load_any,resolve_fn,args,out)
    else:raise ValueError(f"Unknown model {name!r}; choose pretrained,gmp,bare_xattn")
    model=_freeze(model)
    rn_token=None;rn_source_block=None
    if args.use_rn_token:
        sd,_=_resolve_xattn_visual_state(clip,resolve_fn,args);rn_token,rn_source_block,rn_block=_extract_rn(sd,args)
        if rn_token.numel()!=model.visual.conv1.weight.shape[0]:raise RuntimeError(f"RN width {rn_token.numel()} != visual width {model.visual.conv1.weight.shape[0]}")
        aud={"source_model":args.xattn_model,"token_norm":float(rn_token.norm()),"source_insert_block":rn_source_block,"requested_insert_block":rn_block,"width":int(rn_token.numel())}
        (out/"rn_token_audit.json").write_text(json.dumps(aud,indent=2),encoding="utf-8")
        if rn_source_block is not None and rn_source_block!=rn_block:print(f"[RN] WARNING source checkpoint says insert block {rn_source_block}, requested {rn_block}")
    return repo,clip,model,preprocess,info,rn_token

def model_audit(model,name,args,info,rn_token):
    b=model.visual.transformer.resblocks
    return {"variant":name,"source":info,"dtype":str(model.dtype),"conv1_shape":list(model.visual.conv1.weight.shape),"input_resolution":int(model.visual.input_resolution),"n_blocks":len(b),"n_heads":int(b[0].attn.num_heads),"vision_width":int(model.visual.conv1.weight.shape[0]),"output_dim":int(model.visual.output_dim),"use_rn_token":bool(args.use_rn_token),"rn_insert_block":int(args.rn_insert_block) if args.use_rn_token else None,"rn_token_norm":float(rn_token.norm()) if rn_token is not None else None,"conditions":list(args.conditions)}

# ----------------------------- morphology -----------------------------------
def opponent_components(w):
    R,G,B=w[0],w[1],w[2];return (R+G+B)/math.sqrt(3),(R-G)/math.sqrt(2),(R+G-2*B)/math.sqrt(6)
def frequency_grid(h,w):
    fy=np.fft.fftshift(np.fft.fftfreq(h));fx=np.fft.fftshift(np.fft.fftfreq(w));yy,xx=np.meshgrid(fy,fx,indexing="ij");rr=np.sqrt(xx*xx+yy*yy);return yy,xx,rr/max(float(rr.max()),EPS)
def corr_flat(a,b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel();a-=a.mean();b-=b.mean();d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d>EPS else 0.0
def dog_template(h,w,s1,s2):
    y=np.linspace(-1,1,h);x=np.linspace(-1,1,w);yy,xx=np.meshgrid(y,x,indexing="ij");r2=xx*xx+yy*yy;g1=np.exp(-r2/(2*s1*s1));g1/=g1.sum()+EPS;g2=np.exp(-r2/(2*s2*s2));g2/=g2.sum()+EPS;t=g1-g2;t-=t.mean();return t/(np.linalg.norm(t)+EPS)
def gabor_template(h,w,angle,cycles,phase):
    y=np.linspace(-1,1,h);x=np.linspace(-1,1,w);yy,xx=np.meshgrid(y,x,indexing="ij");th=math.radians(angle);xr=xx*math.cos(th)+yy*math.sin(th);env=np.exp(-(xx*xx+yy*yy)/(2*.55*.55));t=env*np.cos(math.pi*cycles*xr+phase);t-=t.mean();return t/(np.linalg.norm(t)+EPS)
def radial_profile(power,nbins=12):
    h,w=power.shape;_,_,rr=frequency_grid(h,w);edges=np.linspace(0,1+1e-9,nbins+1);vals=[];cent=[]
    for i in range(nbins):m=(rr>=edges[i])&(rr<edges[i+1]);vals.append(float(power[m].mean()) if m.any() else 0);cent.append(float((edges[i]+edges[i+1])/2))
    return np.asarray(cent),np.asarray(vals)
def filter_metrics(w,idx):
    w=np.asarray(w,np.float64);h,ww=w.shape[-2:];L,RG,BY=opponent_components(w);eL=np.sum(L*L);eRG=np.sum(RG*RG);eBY=np.sum(BY*BY);et=eL+eRG+eBY+EPS
    Fw=np.fft.fftshift(np.fft.fft2(w,axes=(-2,-1)),axes=(-2,-1));power=np.sum(np.abs(Fw)**2,axis=0);psum=float(power.sum())+EPS;yy,xx,rr=frequency_grid(h,ww);dc=power[h//2,ww//2];low=rr<=.25;mid=(rr>.25)&(rr<=.60);high=rr>.60;pn=power/psum;cent=float(np.sum(pn*rr));bw=float(np.sqrt(np.sum(pn*(rr-cent)**2)));pp=pn.ravel();pp=pp[pp>0];sent=float(-(pp*np.log(pp+EPS)).sum()/math.log(power.size));rc,rv=radial_profile(power,max(7,h//2));pi=int(np.argmax(rv[1:])+1) if len(rv)>1 else 0;pr=float(rc[pi]);ring=np.abs(rr-pr)<=max(1.5/max(h,ww),.10);mx=float(np.sum(pn*xx));my=float(np.sum(pn*yy));dx=xx-mx;dy=yy-my;cov=np.array([[np.sum(pn*dx*dx),np.sum(pn*dx*dy)],[np.sum(pn*dx*dy),np.sum(pn*dy*dy)]]);ev,evec=np.linalg.eigh(cov);o=np.argsort(ev)[::-1];l1,l2=float(ev[o[0]]),float(ev[o[1]]);v1=evec[:,o[0]];anis=(l1-l2)/(l1+l2+EPS);angle=math.degrees(math.atan2(v1[1],v1[0]))%180;ar=math.radians(2*angle);axis=.09;vertical=np.abs(xx)<=axis*(np.max(np.abs(xx))+EPS);horizontal=np.abs(yy)<=axis*(np.max(np.abs(yy))+EPS);corner=(np.abs(xx)>=.55*np.max(np.abs(xx)))&(np.abs(yy)>=.55*np.max(np.abs(yy)));se=np.sum(w*w,axis=0);ss=float(se.sum())+EPS;cy0,cy1=h//3,h-h//3;cx0,cx1=ww//3,ww-ww//3;cm=np.zeros((h,ww),bool);cm[cy0:cy1,cx0:cx1]=True;edge=np.zeros((h,ww),bool);edge[[0,-1],:]=True;edge[:,[0,-1]]=True
    dog=max(abs(corr_flat(L,dog_template(h,ww,a,b))) for a,b in ((.18,.42),(.25,.55),(.32,.72)));gab=0
    for a0 in (0,30,60,90,120,150):
        for cyc in (1.,2.,3.,4.):
            for ph in (0.,math.pi/2):
                t=gabor_template(h,ww,a0,cyc,ph);gab=max(gab,abs(corr_flat(L,t)),abs(corr_flat(RG,t)),abs(corr_flat(BY,t)))
    return {"channel":int(idx),"weight_l2":float(np.linalg.norm(w)),"weight_absmean":float(np.mean(np.abs(w))),"lum_energy_frac":eL/et,"rg_energy_frac":eRG/et,"by_energy_frac":eBY/et,"dc_frac":float(dc/psum),"low_frac":float(power[low].sum()/psum),"mid_frac":float(power[mid].sum()/psum),"high_frac":float(power[high].sum()/psum),"spectral_centroid":cent,"spectral_bandwidth":bw,"spectral_entropy":sent,"radial_peak":pr,"ring_frac":float(power[ring].sum()/psum),"orientation_anisotropy":float(anis),"freq_axis_angle_deg":float(angle),"freq_axis_cos2":math.cos(ar),"freq_axis_sin2":math.sin(ar),"vertical_axis_frac":float(power[vertical].sum()/psum),"horizontal_axis_frac":float(power[horizontal].sum()/psum),"corner_freq_frac":float(power[corner].sum()/psum),"symmetry180":corr_flat(w,w[:,::-1,::-1]),"symmetry_lr":corr_flat(w,w[:,:,::-1]),"symmetry_ud":corr_flat(w,w[:,::-1,:]),"center_spatial_frac":float(se[cm].sum()/ss),"edge_spatial_frac":float(se[edge].sum()/ss),"dog_abs_corr":float(dog),"gabor_abs_corr":float(gab)}
def descriptive_cluster_label(g,all_df):
    m=g[list(MORPH_FEATURES)].mean(numeric_only=True);q=all_df[list(MORPH_FEATURES)].quantile([.25,.5,.75]);shape="center_surroundish" if m.dog_abs_corr>=q.loc[.75,"dog_abs_corr"] and m.orientation_anisotropy<=q.loc[.5,"orientation_anisotropy"] else ("oriented_gaborish" if m.gabor_abs_corr>=q.loc[.75,"gabor_abs_corr"] and m.orientation_anisotropy>=q.loc[.5,"orientation_anisotropy"] else ("broadband" if m.spectral_entropy>=q.loc[.75,"spectral_entropy"] else ("ring_bandpass" if m.ring_frac>=q.loc[.75,"ring_frac"] else "mixed")));band="lowfreq" if m.spectral_centroid<=q.loc[.25,"spectral_centroid"] else ("highfreq" if m.spectral_centroid>=q.loc[.75,"spectral_centroid"] else "midfreq");chrom=float(m.rg_energy_frac+m.by_energy_frac);col="luminance" if m.lum_energy_frac>.67 else ("chromatic" if chrom>.67 else "mixedcolor");ori="anisotropic" if m.orientation_anisotropy>=q.loc[.75,"orientation_anisotropy"] else "isotropicish";return f"{band}__{shape}__{col}__{ori}"
def cluster_filter_metrics(df,kmin,kmax,seed):
    out=df.copy();Xraw=out[list(MORPH_FEATURES)].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float);Xs=StandardScaler().fit_transform(Xraw);nc=min(12,Xs.shape[1],max(1,Xs.shape[0]-1));X=PCA(n_components=nc,random_state=seed).fit_transform(Xs);best=None;tr=[]
    for k in range(max(2,kmin),min(kmax,len(out)-1)+1):
        km=KMeans(n_clusters=k,n_init=30,random_state=seed);lab=km.fit_predict(X);sil=float(silhouette_score(X,lab));tr.append({"k":k,"silhouette":sil,"inertia":float(km.inertia_)})
        if best is None or sil>best[0]:best=(sil,k,km,lab)
    if best is None:out["cluster_id"]=0;out["cluster_label"]="single_cluster";out["cluster_distance"]=0.;out["cluster_outlier_z"]=0.;return out,pd.DataFrame(),{"selected_k":1}
    _,kb,km,lab=best;out["cluster_id"]=lab.astype(int);dist=np.linalg.norm(X-km.cluster_centers_[lab],axis=1);out["cluster_distance"]=dist;out["cluster_outlier_z"]=0.;rows=[];lm={}
    for cid in sorted(out.cluster_id.unique()):
        idx=out.index[out.cluster_id==cid];out.loc[idx,"cluster_outlier_z"]=robust_z(out.loc[idx,"cluster_distance"].to_numpy());g=out.loc[idx];label=descriptive_cluster_label(g,out);lm[int(cid)]=label;rows.append({"cluster_id":int(cid),"cluster_label":label,"n":len(g),"representative_channels":",".join(map(str,g.sort_values("cluster_distance").head(8).channel.astype(int))),"top_outlier_channels":",".join(map(str,g.sort_values("cluster_outlier_z",ascending=False).head(8).channel.astype(int)))})
    out["cluster_label"]=out.cluster_id.map(lm);return out,pd.DataFrame(rows),{"selected_k":int(kb),"k_trials":tr,"features":list(MORPH_FEATURES),"note":"weight_l2 excluded"}

@torch.inference_mode()
def empirical_activation_and_position(model,preprocess,manifest,batch_size,device,amp):
    v=model.visual;C=int(v.conv1.weight.shape[0]);sv=np.zeros(C);ssq=np.zeros(C);sa=np.zeros(C);pc=np.zeros(C);n=0;ms=None
    for st in range(0,len(manifest),batch_size):
        meta=manifest.iloc[st:st+batch_size];batch,_,_=load_batch(preprocess,meta,device)
        with amp_context(device,amp):y=v.conv1(batch.to(dtype=model.dtype)).float().cpu().numpy()
        B,C2,H,W=y.shape;f=y.transpose(1,0,2,3).reshape(C,-1);sv+=f.sum(1);ssq+=(f*f).sum(1);sa+=np.abs(f).sum(1);pc+=(f>0).sum(1);n+=f.shape[1];mm=y.sum(0);ms=mm if ms is None else ms+mm;del batch,y
    mean=sv/max(n,1);msq=ssq/max(n,1);std=np.sqrt(np.maximum(msq-mean*mean,0));rms=np.sqrt(np.maximum(msq,0));ma=sa/max(n,1);posf=pc/max(n,1);mean_map=ms/max(len(manifest),1);pos=v.positional_embedding.detach().float().cpu().numpy()[1:];side=round(math.sqrt(len(pos)));pos_map=pos.reshape(side,side,C).transpose(2,0,1).astype(np.float32);rows=[]
    for c in range(C):
        pm=pos_map[c].astype(float);am=mean_map[c].astype(float);ps=float(pm.std());pr=float(np.sqrt(np.mean(pm*pm)));rows.append({"channel":c,"activation_mean":float(mean[c]),"activation_std":float(std[c]),"activation_rms":float(rms[c]),"activation_mean_abs":float(ma[c]),"activation_positive_frac":float(posf[c]),"pos_mean":float(pm.mean()),"pos_std":ps,"pos_rms":pr,"pos_range":float(pm.max()-pm.min()),"pos_to_pixel_std_ratio":ps/max(float(std[c]),EPS),"pos_to_pixel_rms_ratio":pr/max(float(rms[c]),EPS),"mean_activation_pos_corr":corr_flat(am,pm)})
    return pd.DataFrame(rows),mean_map.astype(np.float32),pos_map,(mean_map+pos_map).astype(np.float32)
def build_twins(master,control_z_min):
    targets=master[master.population.isin(LOW_POPS)].reset_index(drop=True);controls=master[master.robust_log_z>=control_z_min].reset_index(drop=True)
    if targets.empty or controls.empty:return pd.DataFrame(),pd.DataFrame()
    Xt=[];Xc=[]
    for cols in TWIN_FEATURE_DOMAINS.values():
        allv=pd.concat([targets[list(cols)],controls[list(cols)]],ignore_index=True).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float);mu=allv.mean(0);sd=allv.std(0);sd[sd<1e-8]=1;Xt.append((targets[list(cols)].to_numpy(float)-mu)/sd/math.sqrt(len(cols)));Xc.append((controls[list(cols)].to_numpy(float)-mu)/sd/math.sqrt(len(cols)))
    A=np.concatenate(Xt,1);B=np.concatenate(Xc,1);D=((A[:,None,:]-B[None,:,:])**2).sum(-1)
    if HAVE_SCIPY and len(controls)>=len(targets):rr,cc=linear_sum_assignment(D)
    else:
        rr=[];cc=[];used=set()
        for i in range(len(targets)):
            for j in np.argsort(D[i]):
                if int(j) not in used:rr.append(i);cc.append(int(j));used.add(int(j));break
    pairs=[];tops=[]
    for ri,cj in zip(rr,cc):
        tr=targets.iloc[int(ri)];cr=controls.iloc[int(cj)];pairs.append({"target_channel":int(tr.channel),"target_population":tr.population,"control_channel":int(cr.channel),"morphology_distance_sq":float(D[int(ri),int(cj)]),"target_guess":tr.cluster_label,"control_guess":cr.cluster_label})
        for rank,j in enumerate(np.argsort(D[int(ri)])[:5],1):tops.append({"target_channel":int(tr.channel),"rank":rank,"candidate_control_channel":int(controls.iloc[int(j)].channel),"morphology_distance_sq":float(D[int(ri),int(j)])})
    return pd.DataFrame(pairs),pd.DataFrame(tops)

def run_morphology(model,preprocess,manifest,args,out):
    mp=out/"filter_master.csv";mapsp=out/"conv1_static_maps.safetensors"
    if args.resume and mp.is_file() and mapsp.is_file():return pd.read_csv(mp),model.visual.conv1.weight.detach().float().cpu().numpy()
    W=model.visual.conv1.weight.detach().float().cpu().numpy();df=pd.DataFrame([filter_metrics(W[c],c) for c in range(len(W))]);rz,med,sig=robust_log_z(df.weight_l2.to_numpy());df["robust_log_z"]=rz;df["population"]=[population_group(z) for z in rz];df,clusters,audit=cluster_filter_metrics(df,args.cluster_kmin,args.cluster_kmax,SEED);act,mean_map,pos_map,init_map=empirical_activation_and_position(model,preprocess,manifest,args.batch_size,args.device,args.amp);df=df.merge(act,on="channel",validate="one_to_one");pairs,tops=build_twins(df,args.control_z_min);write_df(df,mp);write_df(clusters,out/"morphology_cluster_summary.csv");write_df(act,out/"activation_position_stats.csv");write_df(pairs,out/"twin_pairs.csv");write_df(tops,out/"twin_candidates_top5.csv");(out/"morphology_cluster_audit.json").write_text(json.dumps({**audit,"log_weight_median":med,"log_weight_sigma":sig},indent=2),encoding="utf-8");save_safetensors({"mean_map":torch.from_numpy(mean_map),"pos_map":torch.from_numpy(pos_map),"init_map":torch.from_numpy(init_map)},str(mapsp),{"format":"Conv1 static maps"});return df,W

# --------------------------- interventions/probing ---------------------------
def summarize_response_map(r):
    f=r.flatten(1).float();return {"mean":f.mean(1),"mean_abs":f.abs().mean(1),"std":f.std(1,unbiased=False),"positive_frac":(f>0).float().mean(1),"rms":f.square().mean(1).sqrt()}
class MultiConv1Transform:
    def __init__(self,model,specs,ids):self.model=model;self.specs=[(int(c),str(q).upper()) for c,q in specs];self.ids=list(map(str,ids));self.handle=None;self.cache={}
    def __enter__(self):
        def hook(_m,_inp,out):
            y=out.clone()
            for c,cond in self.specs:
                r=y[:,c].clone();before=summarize_response_map(r)
                if cond=="ZERO":z=torch.zeros_like(r)
                elif cond=="FLIP":z=-r
                elif cond=="ABS":z=r.abs()
                elif cond=="SIGN":z=torch.sign(r)*r.abs().flatten(1).mean(1)[:,None,None]
                elif cond=="DC_ONLY":z=r.flatten(1).mean(1)[:,None,None].expand_as(r)
                elif cond=="CENTERED":z=r-r.flatten(1).mean(1)[:,None,None]
                elif cond=="SHUFFLE":
                    B,H,W=r.shape;rf=r.flatten(1);zz=[]
                    for bi in range(B):
                        g=torch.Generator(device="cpu");g.manual_seed(stable_seed(f"ch{c}:{cond}:{self.ids[bi]}"));p=torch.randperm(H*W,generator=g).to(r.device);zz.append(rf[bi].index_select(0,p).reshape(H,W))
                    z=torch.stack(zz)
                else:raise ValueError(cond)
                y[:,c]=z;self.cache[(c,cond)]=(before,summarize_response_map(z))
            return y
        self.handle=self.model.visual.conv1.register_forward_hook(hook);return self
    def audits(self,ids,sources):
        out={}
        for (c,cond),(b,a) in self.cache.items():
            bb={k:v.detach().cpu().numpy() for k,v in b.items()};aa={k:v.detach().cpu().numpy() for k,v in a.items()}
            for i,sid in enumerate(ids):
                rec={"channel":c,"condition":cond,"stim_id":sid,"source":sources[i]}
                for k in bb:rec[f"before_{k}"]=float(bb[k][i]);rec[f"after_{k}"]=float(aa[k][i])
                out[(c,cond,sid)]=rec
        return out
    def __exit__(self,*exc):
        if self.handle:self.handle.remove()

def q_or_k_for_pre(block,pre,B,which):
    if pre.shape[1]!=B and pre.shape[0]==B:pre=pre.permute(1,0,2).contiguous()
    z=block.ln_1(pre);proj=block.attn.q_proj if which=="q" else block.attn.k_proj;y=proj(z);T,BB,C=y.shape;H=int(block.attn.num_heads);dh=C//H;return y.view(T,BB,H,dh).permute(1,2,0,3).contiguous()
def _spatial_patch_count(model):return int(model.visual.positional_embedding.shape[0]-1)
def _reg_mask(norms,threshold,max_regs,min_regs):
    B,P=norms.shape;out=torch.zeros_like(norms,dtype=torch.bool)
    for i in range(B):
        idx=torch.nonzero(norms[i]>=threshold,as_tuple=False).flatten()
        if max_regs>0 and idx.numel()>max_regs:idx=idx[torch.topk(norms[i,idx],max_regs).indices]
        if idx.numel()<min_regs:
            k=min(min_regs,P);idx=torch.unique(torch.cat([idx,torch.topk(norms[i],k).indices]))
        out[i,idx]=True
    return out

def _projected_v_norms(v_bhd,out_w,head):
    # v [B,Dh], W slice [C,Dh], result norm [B]
    dh=v_bhd.shape[-1];W=out_w[:,head*dh:(head+1)*dh].float();return torch.linalg.vector_norm(v_bhd.float()@W.T,dim=-1)

@dataclass
class Probe:
    emb:np.ndarray;early_maps:Dict[int,np.ndarray];mid_maps:Dict[int,np.ndarray];b12_k_ratio_mean:float;b12_reg_attn_mass:float;pre13_regs:set;pre13_max_norm:float;pre13_frozen_reg_mean_norm:float;b22_q:np.ndarray;final_reg_set:set;head_signature:Optional[np.ndarray];rn_signature:Optional[np.ndarray];rn_final_norm:float

@torch.inference_mode()
def probe_batch(model,images,ids,args,rn_token=None,frozen_regs=None):
    v=model.visual;blocks=v.transformer.resblocks;B=images.shape[0];P=_spatial_patch_count(model);nblocks=len(blocks);early=[b for b in args.early_blocks if 0<=b<nblocks];routing=[b for b in args.routing_blocks if 0<=b<nblocks];want={args.register_k_block,args.register_address_block,args.late_q_block};pre={}
    # Manual vanilla visual forward so RN-only insertion has no bridge side effects.
    with amp_context(args.device,args.amp):
        x=v.conv1(images.to(dtype=model.dtype));x=x.reshape(B,x.shape[1],-1).permute(0,2,1);x=torch.cat([v.class_embedding.to(x.dtype)+torch.zeros(B,1,x.shape[-1],dtype=x.dtype,device=x.device),x],1);x=x+v.positional_embedding.to(x.dtype);x=v.ln_pre(x).permute(1,0,2)
        for bi,blk in enumerate(blocks):
            if rn_token is not None and bi==args.rn_insert_block:
                rn=rn_token.to(device=x.device,dtype=x.dtype).view(1,1,-1).expand(1,B,-1);x=torch.cat([x,rn],dim=0)
            if bi in want:pre[bi]=x.detach()
            x=blk(x,capture=True)
        xbt=x.permute(1,0,2);cls=v.ln_post(xbt[:,0,:]);emb=cls@v.proj if v.proj is not None else cls;emb=F.normalize(emb.float(),dim=-1)
    spatial_final=xbt[:,1:1+P,:].float();fnorm=spatial_final.norm(dim=-1);fmask=_reg_mask(fnorm,args.register_threshold,args.register_max,args.register_min).cpu()
    pre13=pre[args.register_address_block];p13=pre13.permute(1,0,2).float() if pre13.shape[1]==B else pre13.float();n13=p13[:,1:1+P].norm(dim=-1);cur=[]
    for i in range(B):cur.append(set(torch.nonzero(n13[i]>=args.register_threshold,as_tuple=False).flatten().cpu().tolist()))
    regs=list(frozen_regs) if frozen_regs is not None else cur
    early_maps={};mid_maps={}
    for b in early:early_maps[b]=blocks[b].attn.last_probs.float()[:,:,0,1:1+P].cpu().numpy()
    for b in routing:mid_maps[b]=blocks[b].attn.last_probs.float().mean(2).mean(1)[:,1:1+P].cpu().numpy()
    kb=blocks[args.register_k_block];k=q_or_k_for_pre(kb,pre[args.register_k_block],B,"k").float().cpu();p12=kb.attn.last_probs.float().cpu();b12ratio=[];b12mass=[]
    for i in range(B):
        regtok=[r+1 for r in sorted(regs[i]) if 0<=r<P];other=[t for t in range(1,P+1) if t not in set(regtok)]
        if regtok:
            rk=k[i,:,regtok].norm(dim=-1).mean(-1);ok=k[i,:,other].norm(dim=-1).mean(-1) if other else torch.ones_like(rk);b12ratio.append(float((rk/ok.clamp_min(EPS)).mean()));b12mass.append(float(p12[i,:,:,regtok].sum(-1).mean()))
        else:b12ratio.append(np.nan);b12mass.append(np.nan)
    q22=q_or_k_for_pre(blocks[args.late_q_block],pre[args.late_q_block],B,"q").float().cpu()[:,:,0,:].reshape(B,-1).numpy()
    H=int(blocks[0].attn.num_heads);head=np.zeros((B,nblocks,H,3),np.float32) if args.head_signature else None;rn_sig=np.zeros((B,nblocks,H,6),np.float32) if rn_token is not None and args.rn_head_signature else None;rn_idx=P+1
    for b,blk in enumerate(blocks):
        p=blk.attn.last_probs.float().cpu();T=p.shape[-1]
        if head is not None:
            for i in range(B):
                regtok=[r+1 for r in sorted(regs[i]) if 0<=r<P and r+1<T]
                if regtok:
                    head[i,b,:,0]=p[i,:,0,regtok].sum(-1).numpy();head[i,b,:,1]=p[i,:,regtok,0].mean(-1).numpy();head[i,b,:,2]=p[i,:,:,regtok].sum(-1).mean(-1).numpy()
        if rn_sig is not None and b>=args.rn_insert_block and rn_idx<T:
            vv=blk.attn.last_v.float().cpu() if blk.attn.last_v is not None else None
            for h in range(H):
                rn_sig[:,b,h,0]=p[:,h,0,rn_idx].numpy();rn_sig[:,b,h,1]=p[:,h,rn_idx,0].numpy();rn_sig[:,b,h,2]=p[:,h,:,rn_idx].mean(-1).numpy()
                if vv is not None:
                    vn=_projected_v_norms(vv[:,h,rn_idx,:],blk.attn.out_proj.weight.detach().float().cpu(),h);rn_sig[:,b,h,3]=vn.numpy();rn_sig[:,b,h,4]=(vn*p[:,h,0,rn_idx]).numpy();rn_sig[:,b,h,5]=(vn*p[:,h,:,rn_idx].mean(-1)).numpy()
    out=[]
    for i in range(B):
        ridx=sorted(r for r in regs[i] if 0<=r<P);rmean=float(n13[i,ridx].mean()) if ridx else np.nan;fr=set(torch.nonzero(fmask[i],as_tuple=False).flatten().tolist());rnorm=float(xbt[i,rn_idx].float().norm().cpu()) if rn_token is not None and rn_idx<xbt.shape[1] else np.nan
        out.append(Probe(emb[i].cpu().numpy(),{b:early_maps[b][i].copy() for b in early},{b:mid_maps[b][i].copy() for b in routing},b12ratio[i],b12mass[i],set(cur[i]),float(n13[i].max().cpu()),rmean,q22[i].copy(),fr,head[i].copy() if head is not None else None,rn_sig[i].copy() if rn_sig is not None else None,rnorm))
    for blk in blocks:
        for attr in ("last_q","last_k","last_v","last_z","last_logits","last_probs","last_xin"):
            if hasattr(blk.attn,attr):setattr(blk.attn,attr,None)
    return out

def compare_probe(base,cur,args):
    early=[float(cosine_dist_rows_torch(torch.from_numpy(base.early_maps[b]),torch.from_numpy(cur.early_maps[b])).mean()) for b in base.early_maps if b in cur.early_maps];mid=[cosine_distance_np(base.mid_maps[b],cur.mid_maps[b]) for b in base.mid_maps if b in cur.mid_maps];a,c=set(base.pre13_regs),set(cur.pre13_regs);u=a|c;inter=a&c;fa,fc=set(base.final_reg_set),set(cur.final_reg_set);fu=fa|fc;fi=fa&fc
    return {"backbone_cosine_distance":cosine_distance_np(base.emb,cur.emb),"max_embedding_distance":cosine_distance_np(base.emb,cur.emb),"early_cls_scanner_cosdist":float(np.nanmean(early)) if early else np.nan,"mid_incoming_cosdist":float(np.nanmean(mid)) if mid else np.nan,"b12_reg_k_ratio_delta":cur.b12_k_ratio_mean-base.b12_k_ratio_mean,"b12_reg_attn_mass_delta":cur.b12_reg_attn_mass-base.b12_reg_attn_mass,"b13_reg_jaccard":len(inter)/len(u) if u else 1.,"b13_reg_recall":len(inter)/len(a) if a else 1.,"b13_n_regs_baseline":len(a),"b13_n_regs_condition":len(c),"b13_max_spatial_norm_ratio":cur.pre13_max_norm/max(base.pre13_max_norm,EPS),"b13_frozen_reg_mean_norm_ratio":cur.pre13_frozen_reg_mean_norm/max(base.pre13_frozen_reg_mean_norm,EPS) if np.isfinite(cur.pre13_frozen_reg_mean_norm) and np.isfinite(base.pre13_frozen_reg_mean_norm) else np.nan,"b22_cls_q_cosine_distance":cosine_distance_np(base.b22_q,cur.b22_q),"final_reg_jaccard":len(fi)/len(fu) if fu else 1.,"final_reg_count_baseline":len(fa),"final_reg_count_condition":len(fc),"final_reg_count_delta":len(fc)-len(fa),"abs_final_reg_count_delta":abs(len(fc)-len(fa)),"rn_final_norm_delta":cur.rn_final_norm-base.rn_final_norm if np.isfinite(cur.rn_final_norm) and np.isfinite(base.rn_final_norm) else np.nan}

def _sig_acc(acc,base,cur,field):
    x=getattr(base,field);y=getattr(cur,field)
    if acc is None or x is None or y is None:return
    d=y-x
    if "sum" not in acc:acc["sum"]=np.zeros_like(d,np.float64);acc["sumabs"]=np.zeros_like(d,np.float64);acc["n"]=0
    acc["sum"]+=d;acc["sumabs"]+=np.abs(d);acc["n"]+=1
def head_rows(channel,cond,acc):
    if not acc or "sum" not in acc:return []
    s=acc["sum"]/max(acc["n"],1);a=acc["sumabs"]/max(acc["n"],1);rows=[]
    for b in range(s.shape[0]):
        for h in range(s.shape[1]):rows.append({"channel":channel,"condition":cond,"block":b,"head":h,"cls_to_reg_delta":s[b,h,0],"reg_to_cls_delta":s[b,h,1],"all_to_reg_delta":s[b,h,2],"cls_to_reg_abs_delta":a[b,h,0],"reg_to_cls_abs_delta":a[b,h,1],"all_to_reg_abs_delta":a[b,h,2]})
    return rows
def rn_rows(channel,cond,acc):
    if not acc or "sum" not in acc:return []
    s=acc["sum"]/max(acc["n"],1);a=acc["sumabs"]/max(acc["n"],1);names=["cls_to_rn","rn_to_cls","all_to_rn","rn_projected_v_norm","cls_rn_write_norm","all_rn_write_norm"];rows=[]
    for b in range(s.shape[0]):
        for h in range(s.shape[1]):
            r={"channel":channel,"condition":cond,"block":b,"head":h}
            for j,n in enumerate(names):r[n+"_delta"]=s[b,h,j];r[n+"_abs_delta"]=a[b,h,j]
            rows.append(r)
    return rows

@torch.inference_mode()
def build_baseline(model,preprocess,manifest,args,rn_token):
    cache={}
    for st in range(0,len(manifest),args.batch_size):
        meta=manifest.iloc[st:st+args.batch_size];batch,ids,_=load_batch(preprocess,meta,args.device);ps=probe_batch(model,batch,ids,args,rn_token=rn_token)
        for sid,p in zip(ids,ps):cache[sid]=p
        del batch,ps
    return cache

@torch.inference_mode()
def run_scan(model,preprocess,manifest,channels,conditions,args,out_path,stage_name,rn_token,save_heads):
    existing=pd.read_csv(out_path) if args.resume and out_path.is_file() else pd.DataFrame();rows=existing.to_dict("records") if len(existing) else [];done=set()
    if len(existing):
        ct=existing.groupby(["channel","condition"]).stim_id.nunique()
        for (c,q),n in ct.items():
            if int(n)>=len(manifest):done.add((int(c),str(q)))
    base=build_baseline(model,preprocess,manifest,args,rn_token)
    hp=out_path.parent/"head_register_coupling_signature.csv.gz";rp=out_path.parent/"rn_head_signature.csv.gz";hrows=pd.read_csv(hp).to_dict("records") if save_heads and args.resume and hp.is_file() else [];rrows=pd.read_csv(rp).to_dict("records") if save_heads and args.use_rn_token and args.resume and rp.is_file() else []
    total=len(channels)*len(conditions);ci=0
    for c in map(int,channels):
        for cond in map(str,conditions):
            ci+=1
            if (c,cond) in done:print(f"[{stage_name} {ci}/{total}] ch{c:04d} {cond} [resume]");continue
            cell=[];ha={} if save_heads and args.head_signature else None;ra={} if save_heads and args.use_rn_token and args.rn_head_signature else None
            for st in range(0,len(manifest),args.batch_size):
                meta=manifest.iloc[st:st+args.batch_size];batch,ids,sources=load_batch(preprocess,meta,args.device);frozen=[set(base[s].pre13_regs) for s in ids]
                with MultiConv1Transform(model,[(c,cond)],ids) as iv:cur=probe_batch(model,batch,ids,args,rn_token=rn_token,frozen_regs=frozen);aud=iv.audits(ids,sources)
                for sid,src,p in zip(ids,sources,cur):
                    rec={"channel":c,"condition":cond,"stim_id":sid,"source":src};rec.update(compare_probe(base[sid],p,args));rec.update(aud.get((c,cond,sid),{}));cell.append(rec);_sig_acc(ha,base[sid],p,"head_signature");_sig_acc(ra,base[sid],p,"rn_signature")
                del batch,cur
            rows=[r for r in rows if not (int(r.get("channel",-1))==c and str(r.get("condition",""))==cond)]+cell;write_df(pd.DataFrame(rows),out_path)
            if ha is not None:hrows=[r for r in hrows if not (int(r.get("channel",-1))==c and str(r.get("condition",""))==cond)]+head_rows(c,cond,ha);write_df(pd.DataFrame(hrows),hp)
            if ra is not None:rrows=[r for r in rrows if not (int(r.get("channel",-1))==c and str(r.get("condition",""))==cond)]+rn_rows(c,cond,ra);write_df(pd.DataFrame(rrows),rp)
            print(f"[{stage_name} {ci}/{total}] ch{c:04d} {cond} n={len(cell)}")
    return pd.DataFrame(rows),pd.DataFrame(hrows),pd.DataFrame(rrows)

def summarize_causal(per,out,name):
    metrics=[c for c in per.columns if c not in {"channel","condition","stim_id","source"} and pd.api.types.is_numeric_dtype(per[c])];rows=[]
    groups=list(per.groupby(["channel","condition","source"],dropna=False))+list(per.groupby(["channel","condition"],dropna=False))
    for keys,g in groups:
        if len(keys)==3:c,q,src=keys
        else:c,q=keys;src="all"
        rec={"channel":int(c),"condition":str(q),"source":str(src),"n":len(g)}
        for m in metrics:
            v=pd.to_numeric(g[m],errors="coerce");rec[f"mean_{m}"]=float(v.mean());rec[f"max_{m}"]=float(v.max());rec[f"min_{m}"]=float(v.min());rec[f"p90_{m}"]=float(v.quantile(.9))
        rows.append(rec)
    df=pd.DataFrame(rows);write_df(df,out/name);return df

# ----------------------- candidate/culprit analysis --------------------------
def select_candidates(master,summary,args,out):
    s=summary[summary.source.eq("all")];agg=s.groupby("channel").agg(screen_max_effect=("max_backbone_cosine_distance","max"),screen_max_routing=("max_mid_incoming_cosdist","max"),screen_max_b22=("max_b22_cls_q_cosine_distance","max"),screen_min_reg=("min_b13_reg_jaccard","min")).reset_index();z=master.merge(agg,on="channel",how="left");z["screen_max_reg_disrupt"]=1-z.screen_min_reg
    rank=["screen_max_effect","screen_max_routing","screen_max_b22","screen_max_reg_disrupt","pos_to_pixel_std_ratio","cluster_outlier_z"]
    for c in rank:z["pct_"+c]=percentile01(z[c].fillna(0))
    include=set(z[z.population.isin(LOW_POPS)].channel.astype(int))
    for c in rank:include.update(z.nlargest(min(args.candidate_top_per_axis,len(z)),c).channel.astype(int))
    include.update(int(c) for c in args.focus_channels if int(c)<len(z));q=z[z.channel.isin(include)].copy();low=q[q.population.isin(LOW_POPS)];oth=q[~q.population.isin(LOW_POPS)].copy();oth["pre_score"]=np.sqrt(np.mean(np.stack([oth["pct_"+c].to_numpy()**2 for c in rank],1),1));room=max(0,args.max_candidates-len(low)) if args.max_candidates>0 else len(oth);q=pd.concat([low,oth.sort_values("pre_score",ascending=False).head(room)]).drop_duplicates("channel").sort_values("channel");write_df(q,out/"candidate_channels.csv");return q

def build_phase(master,summary,candidates,out):
    q=summary[summary.source.eq("all") & summary.channel.isin(candidates.channel)];rows=[];mapping={"scanner":"max_early_cls_scanner_cosdist","routing":"max_mid_incoming_cosdist","b22q":"max_b22_cls_q_cosine_distance","effect":"max_backbone_cosine_distance","register":"min_final_reg_jaccard"}
    for c,g in q.groupby("channel"):
        r={"channel":int(c)}
        for short,col in mapping.items():
            vals=pd.to_numeric(g[col],errors="coerce");vals=1-vals if short=="register" else vals;r["max_"+short]=float(vals.max());r["argmax_"+short]=str(g.iloc[int(np.nanargmax(vals.to_numpy()))].condition) if vals.notna().any() else ""
        rows.append(r)
    df=candidates.merge(pd.DataFrame(rows),on="channel",how="left")
    for s in mapping:df["pct_"+s]=percentile01(df["max_"+s].fillna(0))
    df["causal_combined_legacy"]=np.sqrt((df.pct_scanner**2+df.pct_routing**2+df.pct_b22q**2)/3);df["vanilla_oddity_score"]=np.sqrt((df.pct_effect**2+df.pct_register**2+df.causal_combined_legacy**2)/3);arr=df[["pct_scanner","pct_routing","pct_b22q"]].to_numpy();names=np.array(["scanner","routing","b22q"],object);df["dominant_causal_axis"]=names[np.argmax(arr,1)];feat=pd.DataFrame({"log_pos":np.log10(df.pos_to_pixel_std_ratio.clip(lower=1e-6)),"log_rms":np.log10(df.activation_rms.clip(lower=1e-6)),"scanner":df.pct_scanner,"routing":df.pct_routing,"b22q":df.pct_b22q,"effect":df.pct_effect,"register":df.pct_register}).replace([np.inf,-np.inf],np.nan).fillna(0);X=StandardScaler().fit_transform(feat);best=(1,-1,np.zeros(len(df),int))
    for k in range(2,min(8,len(df)-1)+1):
        lab=KMeans(k,n_init=30,random_state=SEED).fit_predict(X)
        try:sc=float(silhouette_score(X,lab))
        except:sc=-1
        if sc>best[1]:best=(k,sc,lab)
    df["phase_cluster"]=best[2];df["phase_cluster_k"]=best[0];df["phase_cluster_silhouette"]=best[1];write_df(df,out/"functional_phase_metrics.csv");return df

def culprit_ranking(phase,summary,out):
    q=summary[summary.source.eq("all")].copy();q["cell_effect"]=q.max_backbone_cosine_distance;best=q.sort_values("cell_effect",ascending=False).groupby("channel").head(1)[["channel","condition","cell_effect"]].rename(columns={"condition":"best_condition"});d=phase.merge(best,on="channel",how="left");scorecols={"strong_embedding_steerer":"pct_effect","routing_bus_like":"pct_routing","late_cls_q_sensitive":"pct_b22q","register_allocator_like":"pct_register"};vals=np.stack([d[c].fillna(0) for c in scorecols.values()],1);keys=list(scorecols);d["primary_family"]=[keys[i] for i in np.argmax(vals,1)];d["family_tags"]=[";".join([k for k,c in scorecols.items() if float(getattr(r,c,0))>=.90]+(["positional_scaffold_like"] if float(r.pos_to_pixel_std_ratio)>=float(phase.pos_to_pixel_std_ratio.quantile(.9)) else [])+(["morphology_outlier"] if float(r.cluster_outlier_z)>=2.5 else [])) or "moderate" for r in d.itertuples()];d=d.sort_values(["vanilla_oddity_score","cell_effect"],ascending=False).reset_index(drop=True);d["overall_rank"]=np.arange(1,len(d)+1);write_df(d,out/"culprit_channel_ranking.csv");return d

def identify_culprit_events(per,args,out):
    z=per.copy();dist=pd.to_numeric(z.backbone_cosine_distance,errors="coerce");z["event_robust_z"]=robust_z(dist.to_numpy());z["event_percentile"]=dist.rank(pct=True);z["absolute_severe"]=dist>(1-args.severe_cos_sim);z["statistical_outlier"]=(z.event_robust_z>=args.event_outlier_z)|(z.event_percentile>=args.event_outlier_pct);cul=z[z.absolute_severe|z.statistical_outlier].copy();sev=z[z.absolute_severe].copy();write_df(sev,out/"severe_single_events.csv");write_df(cul,out/"culprit_single_events.csv");return sev,cul

# ------------------------------ plotting ------------------------------------
def plot_phase(phase,master,out):
    p=out/"plots"/"PHASE";p.mkdir(parents=True,exist_ok=True);x=phase.pos_to_pixel_std_ratio.clip(lower=1e-5);y=phase.activation_rms.clip(lower=1e-5);size=25+240*phase.vanilla_oddity_score**2;rgb=np.clip(np.stack([.15+.8*phase.pct_scanner,.15+.8*phase.pct_routing,.15+.8*phase.pct_b22q],1),0,1);fig,ax=plt.subplots(figsize=(12,9));ax.scatter(x,y,s=size,c=rgb,edgecolors="black",linewidths=.35);ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("positional std / natural Conv1 std");ax.set_ylabel("natural Conv1 activation RMS");ax.set_title("Vanilla Conv1 functional phase: RGB scanner/routing/B22-Q");ax.grid(alpha=.15,which="both")
    top=set(phase.nlargest(min(20,len(phase)),"vanilla_oddity_score").channel.astype(int))|set(FOCUS_DEFAULT)
    for r in phase[phase.channel.isin(top)].itertuples():ax.annotate(str(int(r.channel)),(max(r.pos_to_pixel_std_ratio,1e-5),max(r.activation_rms,1e-5)),fontsize=8,xytext=(3,3),textcoords="offset points")
    fig.tight_layout();fig.savefig(p/"01_LOW_TAIL_PHASE_DIAGRAM_RGB_CAUSALITY.png",dpi=220);plt.close(fig)

def _fit_coords(X,args):
    X=np.asarray(X,np.float32);out={}
    if len(X)>=2:out["pca"]=PCA(n_components=min(3,X.shape[1],len(X)),random_state=SEED).fit_transform(X)
    if len(X)>=4 and HAVE_UMAP and not args.skip_umap:out["umap"]=umap.UMAP(n_components=2,n_neighbors=min(15,len(X)-1),min_dist=.1,random_state=SEED).fit_transform(X)
    if len(X)>=5 and not args.skip_tsne:out["tsne"]=TSNE(n_components=2,perplexity=min(30,max(2,(len(X)-1)//3)),init="pca",learning_rate="auto",random_state=SEED).fit_transform(X)
    return out

def plot_manifold(meta,X,out,prefix,args):
    if len(meta)!=len(X) or len(X)<2:return
    coords=_fit_coords(X,args);p=out/"plots"/"MANIFOLDS";p.mkdir(parents=True,exist_ok=True)
    for name,xy in coords.items():
        if xy.shape[1]<2:continue
        fig,ax=plt.subplots(figsize=(12,9));kind=meta["kind"] if "kind" in meta else pd.Series(["point"]*len(meta));base=kind.eq("baseline");ax.scatter(xy[base,0],xy[base,1],s=34,c="black",alpha=.7,label="baseline")
        alt=~base;conds=meta.loc[alt,"condition"] if "condition" in meta else pd.Series(["point"]*alt.sum(),index=meta.index[alt])
        for cond in pd.unique(conds):
            m=alt & meta.get("condition",pd.Series("point",index=meta.index)).eq(cond);ax.scatter(xy[m,0],xy[m,1],s=34,marker="x",label=str(cond),alpha=.7)
        ax.set_title(f"{prefix} {name.upper()} manifold");ax.grid(alpha=.12);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(p/f"{prefix}_{name}_2D.png",dpi=210);plt.close(fig)

def plot_focus(summary,head,rnhead,args,out):
    root=out/"plots"/"FOCUS_CHANNELS";root.mkdir(parents=True,exist_ok=True);q=summary[summary.source.eq("all")]
    for c in args.focus_channels:
        g=q[q.channel.eq(int(c))]
        if g.empty:continue
        fig,ax=plt.subplots(figsize=(10,6));metrics=["max_backbone_cosine_distance","max_early_cls_scanner_cosdist","max_mid_incoming_cosdist","max_b22_cls_q_cosine_distance"]
        xx=np.arange(len(g));w=.18
        for j,m in enumerate(metrics):ax.bar(xx+(j-1.5)*w,g[m],width=w,label=m.replace("max_","").replace("_cosine_distance","").replace("_cosdist",""))
        ax.set_xticks(xx);ax.set_xticklabels(g.condition);ax.set_title(f"ch{int(c):04d}: causal profile");ax.legend(fontsize=7);ax.grid(axis="y",alpha=.15);fig.tight_layout();fig.savefig(root/f"ch{int(c):04d}__causal_profile.png",dpi=210);plt.close(fig)
        for cond in g.condition:
            h=head[(head.channel==c)&(head.condition==cond)] if len(head) else pd.DataFrame()
            if len(h):
                pv=h.pivot(index="block",columns="head",values="cls_to_reg_abs_delta");fig,ax=plt.subplots(figsize=(11,7));im=ax.imshow(pv.to_numpy(),aspect="auto",interpolation="nearest");ax.set_title(f"ch{c:04d} {cond}: |Δ CLS→REG attention|");ax.set_xlabel("head");ax.set_ylabel("block");fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(root/f"ch{c:04d}__{cond}__CLS_TO_REG_HEADMAP.png",dpi=210);plt.close(fig)
            rh=rnhead[(rnhead.channel==c)&(rnhead.condition==cond)] if len(rnhead) else pd.DataFrame()
            if len(rh):
                for col,title in [("cls_to_rn_abs_delta","|Δ CLS→RN attention|"),("cls_rn_write_norm_abs_delta","|Δ RN→CLS projected write norm|")]:
                    pv=rh.pivot(index="block",columns="head",values=col);fig,ax=plt.subplots(figsize=(11,7));im=ax.imshow(pv.to_numpy(),aspect="auto",interpolation="nearest");ax.set_title(f"ch{c:04d} {cond}: {title}");ax.set_xlabel("head");ax.set_ylabel("block");fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(root/f"ch{c:04d}__{cond}__{col}.png",dpi=210);plt.close(fig)

# ------------------------ vectors + synergy ---------------------------------
@torch.inference_mode()
def run_specs(model,preprocess,manifest,specs,args,rn_token):
    out={}
    for st in range(0,len(manifest),args.batch_size):
        meta=manifest.iloc[st:st+args.batch_size];batch,ids,_=load_batch(preprocess,meta,args.device)
        with MultiConv1Transform(model,specs,ids):ps=probe_batch(model,batch,ids,args,rn_token=rn_token)
        for sid,p in zip(ids,ps):out[sid]=p.emb.copy()
    return out

@torch.inference_mode()
def collect_vectors(model,preprocess,manifest,culprit,args,out,rn_token):
    mp=out/"embedding_event_metadata.csv";vp=out/"embedding_event_vectors.safetensors"
    if args.resume and mp.is_file() and vp.is_file():
        m=pd.read_csv(mp);X=_load_st(str(vp))["embedding"].float().numpy()
        if len(m)==len(X):print(f"[culprit vectors] resume: {len(m)} rows");return m,X
    base=run_specs(model,preprocess,manifest,[],args,rn_token);meta=[];vec=[]
    for r in manifest.itertuples():meta.append({"kind":"baseline","stim_id":str(r.stim_id),"pair":str(r.pair),"source":str(r.source),"channel":-1,"condition":"BASELINE"});vec.append(base[str(r.stim_id)])
    cells=culprit[["channel","condition"]].drop_duplicates().sort_values(["condition","channel"]);lookup=manifest.set_index("stim_id");n=len(cells)
    for ci,r in enumerate(cells.itertuples(),1):
        ids=set(culprit[(culprit.channel==r.channel)&(culprit.condition==r.condition)].stim_id.astype(str));sub=manifest[manifest.stim_id.astype(str).isin(ids)];got=run_specs(model,preprocess,sub,[(int(r.channel),str(r.condition))],args,rn_token)
        for sid,z in got.items():mr=lookup.loc[sid];meta.append({"kind":"altered","stim_id":sid,"pair":mr.pair,"source":mr.source,"channel":int(r.channel),"condition":str(r.condition)});vec.append(z)
        print(f"[culprit vectors {ci}/{n}] ch{int(r.channel):04d} {r.condition}")
    m=pd.DataFrame(meta);X=np.stack(vec).astype(np.float32);write_df(m,mp);save_safetensors({"embedding":torch.from_numpy(X.astype(np.float16))},str(vp),{"normalized":"true"});return m,X

def build_displacement(meta,X):
    base={str(r.stim_id):X[i] for i,r in enumerate(meta.itertuples()) if r.kind=="baseline"};rows=[];vec=[]
    for i,r in enumerate(meta.itertuples()):
        if r.kind=="baseline":continue
        if str(r.stim_id) in base:rows.append(r._asdict());vec.append(X[i]-base[str(r.stim_id)])
    return pd.DataFrame(rows),np.stack(vec).astype(np.float32) if vec else np.zeros((0,X.shape[1]),np.float32)
def stack_metrics(base,combo,singles):
    d=combo-base;ds=[x-base for x in singles];lin=np.sum(ds,0);res=d-lin;ad=np.linalg.norm(d);ln=np.linalg.norm(lin);return {"combined_cosine_distance":cosine_distance_np(base,combo),"sum_single_cosine_distance":float(sum(cosine_distance_np(base,x) for x in singles)),"actual_vs_linear_delta_cosine":cosine_np(d,lin),"actual_delta_norm":float(ad),"linear_delta_norm":float(ln),"actual_over_linear_delta_norm":float(ad/(ln+EPS)),"nonlinear_residual_norm":float(np.linalg.norm(res)),"nonlinear_residual_over_actual":float(np.linalg.norm(res)/(ad+EPS))}

@torch.inference_mode()
def run_synergy(model,preprocess,manifest,culprit,ranking,args,out,rn_token):
    samep=out/"stack_synergy_same_image.csv";famp=out/"stack_synergy_condition_family.csv";pairp=out/"pair_screen_summary.csv";resp=out/"stack_nonlinear_residuals.safetensors";rmp=out/"nonlinear_residual_metadata.csv"
    if args.resume and all(p.is_file() for p in [samep,famp,pairp,resp,rmp]):
        t=_load_st(str(resp));print("[synergy] resume: loading saved artifacts");return pd.read_csv(samep),pd.read_csv(famp),pd.read_csv(pairp),t["residual"].float().numpy(),pd.read_csv(rmp)
    base=run_specs(model,preprocess,manifest,[],args,rn_token);same=[];res=[];rmeta=[]
    # same-image top 2/3 culprits by effect order from culprit table
    for (sid,cond),g in culprit.groupby(["stim_id","condition"]):
        chans=list(dict.fromkeys(g.sort_values("backbone_cosine_distance",ascending=False).channel.astype(int)))
        for depth in args.same_image_depths:
            if len(chans)<depth:continue
            sel=chans[:depth];one=manifest[manifest.stim_id.astype(str).eq(str(sid))];sing=[run_specs(model,preprocess,one,[(c,str(cond))],args,rn_token)[str(sid)] for c in sel];combo=run_specs(model,preprocess,one,[(c,str(cond)) for c in sel],args,rn_token)[str(sid)];m=stack_metrics(base[str(sid)],combo,sing);rec={"stack_scope":"same_image","stim_id":str(sid),"condition":str(cond),"channels":";".join(map(str,sel)),"depth":depth,**m};same.append(rec);res.append((combo-base[str(sid)])-np.sum(np.stack([x-base[str(sid)] for x in sing]),0));rmeta.append({**rec,"kind":"nonlinear_residual"})
    same=pd.DataFrame(same);write_df(same,samep)
    fam=[]
    for cond,g in ranking.dropna(subset=["best_condition"]).groupby("best_condition"):
        gg=g[g.best_condition.eq(cond)].sort_values("cell_effect",ascending=False);ch=gg.channel.astype(int).tolist()
        for depth in args.global_stack_depths:
            if len(ch)<depth:continue
            sel=ch[:depth];singmaps={c:run_specs(model,preprocess,manifest,[(c,str(cond))],args,rn_token) for c in sel};combo=run_specs(model,preprocess,manifest,[(c,str(cond)) for c in sel],args,rn_token)
            for r in manifest.itertuples():
                sid=str(r.stim_id);m=stack_metrics(base[sid],combo[sid],[singmaps[c][sid] for c in sel]);fam.append({"stack_scope":"condition_family","stim_id":sid,"condition":str(cond),"channels":";".join(map(str,sel)),"depth":depth,**m})
    fam=pd.DataFrame(fam);write_df(fam,famp)
    top=ranking.dropna(subset=["best_condition"]).head(args.pair_screen_top_specs);specs=[(int(r.channel),str(r.best_condition)) for r in top.itertuples()];sub=choose_stride_subset(manifest,args.pair_screen_images);single={sp:run_specs(model,preprocess,sub,[sp],args,rn_token) for sp in specs};pr=[]
    for i in range(len(specs)):
        for j in range(i+1,len(specs)):
            a,b=specs[i],specs[j];combo=run_specs(model,preprocess,sub,[a,b],args,rn_token)
            for r in sub.itertuples():
                sid=str(r.stim_id);m=stack_metrics(base[sid],combo[sid],[single[a][sid],single[b][sid]]);pr.append({"stim_id":sid,"ch1":a[0],"cond1":a[1],"ch2":b[0],"cond2":b[1],**m})
    per=pd.DataFrame(pr);write_df(per,out/"pair_screen_per_image.csv")
    if len(per):
        agg=per.groupby(["ch1","cond1","ch2","cond2"]).agg(n=("stim_id","size"),mean_residual_over_actual=("nonlinear_residual_over_actual","mean"),max_residual_over_actual=("nonlinear_residual_over_actual","max"),mean_actual_vs_linear=("actual_vs_linear_delta_cosine","mean"),mean_effect=("combined_cosine_distance","mean")).reset_index();agg["pair_score"]=agg.mean_residual_over_actual*agg.mean_effect;agg=agg.sort_values("pair_score",ascending=False)
    else:agg=pd.DataFrame()
    write_df(agg,pairp)
    R=np.stack(res).astype(np.float32) if res else np.zeros((0,1),np.float32);save_safetensors({"residual":torch.from_numpy(R.astype(np.float16))},str(resp));rm=pd.DataFrame(rmeta);write_df(rm,rmp);return same,fam,agg,R,rm

def plot_synergy(same,fam,pair,out):
    p=out/"plots"/"STACK_SYNERGY";p.mkdir(parents=True,exist_ok=True)
    if len(same):
        fig,ax=plt.subplots(figsize=(11,8));ax.scatter(same.sum_single_cosine_distance,same.combined_cosine_distance,s=45,alpha=.7);lim=max(same.sum_single_cosine_distance.max(),same.combined_cosine_distance.max(),.01)*1.05;ax.plot([0,lim],[0,lim],"--");ax.set_xlabel("sum single cosine distances");ax.set_ylabel("stacked cosine distance");ax.set_title("Same-image stacks: geometric 1+1=5 screen");ax.grid(alpha=.15);fig.tight_layout();fig.savefig(p/"07_SAME_IMAGE_STACK_DISTANCE_SYNERGY.png",dpi=220);plt.close(fig)
    if len(fam):
        s=fam.groupby(["condition","depth"]).combined_cosine_distance.mean().reset_index();write_df(s,out/"stack_synergy_condition_family_summary.csv");fig,ax=plt.subplots(figsize=(12,7));x=np.arange(len(s));ax.bar(x,s.combined_cosine_distance);ax.set_xticks(x);ax.set_xticklabels([f"{r.condition}\ntop{int(r.depth)}" for r in s.itertuples()],rotation=45,ha="right");ax.set_ylabel("mean final cosine distance");ax.set_title("Condition-family culprit stacks");fig.tight_layout();fig.savefig(p/"09_CONDITION_FAMILY_STACK_MEAN_DISTANCE.png",dpi=220);plt.close(fig)
    if len(pair):
        top=pair.head(30);fig,ax=plt.subplots(figsize=(11,8));ax.scatter(top.mean_effect,top.mean_residual_over_actual,s=50);ax.set_xlabel("mean pair displacement");ax.set_ylabel("mean nonlinear residual / actual");ax.set_title("Cross-condition pair screen");ax.grid(alpha=.15)
        for r in top.head(12).itertuples():ax.annotate(f"{r.ch1}:{r.cond1}+{r.ch2}:{r.cond2}",(r.mean_effect,r.mean_residual_over_actual),fontsize=7,xytext=(3,3),textcoords="offset points")
        fig.tight_layout();fig.savefig(p/"10_CROSS_CONDITION_PAIR_SCREEN.png",dpi=220);plt.close(fig)

def head_similarity(head):
    if head.empty:return pd.DataFrame()
    cols=["cls_to_reg_delta","reg_to_cls_delta","all_to_reg_delta"];vec={}
    for (c,q),g in head.groupby(["channel","condition"]):vec[(int(c),str(q))]=g.sort_values(["block","head"])[cols].to_numpy().reshape(-1)
    rows=[];keys=list(vec)
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            a,b=keys[i],keys[j]
            if a[0]!=b[0]:rows.append({"ch1":a[0],"cond1":a[1],"ch2":b[0],"cond2":b[1],"head_signature_cosine":cosine_np(vec[a],vec[b])})
    return pd.DataFrame(rows)

def write_shortlist(ranking,pair,headsim,args,out,variant):
    lines=[f"Vanilla Conv1 culprit shortlist — {variant}","="*56,"",f"RN token: {args.use_rn_token}",f"Conditions: {','.join(args.conditions)}","","TOP CHANNELS","------------"]
    for r in ranking.head(args.shortlist_channels).itertuples():lines.append(f"#{int(r.overall_rank):02d} ch{int(r.channel):04d} {str(r.best_condition):8s} score={float(r.vanilla_oddity_score):.3f} effect={float(r.cell_effect):.4f} {r.family_tags}")
    lines += ["","FOCUS CHANNELS","--------------",", ".join(str(x) for x in args.focus_channels),"","READY FOR MANIFOLD EXPLORER — SINGLES",'"singles": [']
    top=ranking.head(args.shortlist_channels)
    for i,r in enumerate(top.itertuples()):lines.append(f'  {{"channel": {int(r.channel)}, "condition": "{r.best_condition}"}}'+("," if i<len(top)-1 else ""))
    lines += ["]","","READY FOR MANIFOLD EXPLORER — PAIRS",'"pairs": [']
    pr=pair.head(args.shortlist_pairs) if len(pair) else pd.DataFrame()
    if len(pr) and len(headsim):pr=pr.merge(headsim,on=["ch1","cond1","ch2","cond2"],how="left")
    for i,r in enumerate(pr.itertuples()):lines.append(f'  {{"ch1": {int(r.ch1)}, "cond1": "{r.cond1}", "ch2": {int(r.ch2)}, "cond2": "{r.cond2}"}}'+("," if i<len(pr)-1 else ""))
    lines += ["]","","NOTE","----","RN-only mode appends exactly one learned x-attn READ_NULL token before the configured block; no bridge/correction/router module is present."]
    (out/"CULPRIT_SHORTLIST.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"experiment_candidates.json").write_text(json.dumps({"singles":[{"channel":int(r.channel),"condition":str(r.best_condition)} for r in top.itertuples()],"pairs":[{"ch1":int(r.ch1),"cond1":str(r.cond1),"ch2":int(r.ch2),"cond2":str(r.cond2),"pair_score":float(r.pair_score)} for r in pair.head(args.shortlist_pairs).itertuples()] if len(pair) else []},indent=2),encoding="utf-8")

# ------------------------------ one model run --------------------------------
def run_one_model(variant,args,manifest):
    mode="rn_token" if args.use_rn_token else "normal";out=Path(args.out_root)/mode/safe_name(variant);out.mkdir(parents=True,exist_ok=True);(out/"plots").mkdir(exist_ok=True);print(f"\n=== {variant} | {mode} -> {out} ===")
    repo,clip,model,preprocess,info,rn_token=load_model_variant(variant,args,out);audit=model_audit(model,variant,args,info,rn_token);(out/"model_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8");write_df(manifest,out/"manifest.csv");nblocks=audit["n_blocks"]
    for b in [*args.early_blocks,*args.routing_blocks,args.register_k_block,args.register_address_block,args.late_q_block,args.rn_insert_block if args.use_rn_token else 0]:
        if not 0<=int(b)<nblocks:raise ValueError(f"block {b} outside n_blocks={nblocks}")
    master,W=run_morphology(model,preprocess,manifest,args,out)
    if args.suite=="morphology":return
    sub=choose_stride_subset(manifest,args.screen_images);screen,_,_=run_scan(model,preprocess,sub,range(W.shape[0]),args.conditions,args,out/"global_screen_per_image.csv","global screen",rn_token,False);ss=summarize_causal(screen,out,"global_screen_summary.csv");cands=select_candidates(master,ss,args,out)
    if args.suite=="screen":return
    per,head,rnhead=run_scan(model,preprocess,manifest,cands.channel.astype(int).tolist(),args.conditions,args,out/"causal_per_image.csv.gz","causal",rn_token,True);summ=summarize_causal(per,out,"causal_summary.csv");phase=build_phase(master,summ,cands,out);rank=culprit_ranking(phase,summ,out);plot_phase(phase,master,out);sev,cul=identify_culprit_events(per,args,out);plot_focus(summ,head,rnhead,args,out)
    if args.suite=="causal":write_shortlist(rank,pd.DataFrame(),head_similarity(head),args,out,variant);return
    meta,X=collect_vectors(model,preprocess,manifest,cul,args,out,rn_token);plot_manifold(meta,X,out,"raw_embedding",args);dm,dX=build_displacement(meta,X);write_df(dm,out/"displacement_metadata.csv");save_safetensors({"delta":torch.from_numpy(dX.astype(np.float16))},str(out/"displacement_vectors.safetensors"));plot_manifold(dm,dX,out,"delta",args);same,fam,pair,R,rm=run_synergy(model,preprocess,manifest,cul,rank,args,out,rn_token);plot_synergy(same,fam,pair,out)
    if len(R)>=3:plot_manifold(rm,R,out,"nonlinear_residual",args)
    hs=head_similarity(head);write_df(hs,out/"head_signature_pair_similarity.csv") if len(hs) else None;write_shortlist(rank,pair,hs,args,out,variant);(out/"REPORT.md").write_text(f"# Vanilla Conv1 functional atlas — {variant}\n\nRN token: {args.use_rn_token}\nConditions: {args.conditions}\nCandidates: {len(cands)}\nAbsolute severe events: {len(sev)}\nCulprit events: {len(cul)}\n",encoding="utf-8");print(f"DONE: {out}")
    del model;gc.collect();
    if torch.cuda.is_available():torch.cuda.empty_cache()

# -------------------------------- CLI ----------------------------------------
def parse_args(argv=None):
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--models",type=lambda s:[x.strip().lower() for x in s.split(",") if x.strip()],default=["gmp","bare_xattn"])
    ap.add_argument("--conditions",type=parse_str_list,default=list(DEFAULT_CONDITIONS));ap.add_argument("--use_rn_token",action="store_true");ap.add_argument("--rn_insert_block",type=int,default=13)
    ap.add_argument("--pretrained_model",default=DEFAULT_PRETRAINED);ap.add_argument("--gmp_checkpoint",default=DEFAULT_GMP);ap.add_argument("--xattn_model",default=DEFAULT_XATTN);ap.add_argument("--xattn_revision",default="");ap.add_argument("--hf_cache_dir",default="");ap.add_argument("--repo_root",default="")
    ap.add_argument("--image_dir",default=DEFAULT_IMAGE_DIR);ap.add_argument("--recursive_images",action="store_true");ap.add_argument("--out_root",default=DEFAULT_OUT_ROOT);ap.add_argument("--device",default="cuda");ap.add_argument("--batch_size",type=int,default=8);ap.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);ap.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True);ap.add_argument("--suite",choices=["morphology","screen","causal","all"],default="all")
    ap.add_argument("--screen_images",type=int,default=8);ap.add_argument("--max_candidates",type=int,default=160);ap.add_argument("--candidate_top_per_axis",type=int,default=24);ap.add_argument("--severe_cos_sim",type=float,default=.97);ap.add_argument("--event_outlier_z",type=float,default=3.0);ap.add_argument("--event_outlier_pct",type=float,default=.99)
    ap.add_argument("--register_threshold",type=float,default=70.0);ap.add_argument("--register_max",type=int,default=4);ap.add_argument("--register_min",type=int,default=1);ap.add_argument("--head_signature",action=argparse.BooleanOptionalAction,default=True);ap.add_argument("--rn_head_signature",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--early_blocks",type=parse_int_list,default=[0,1,2,3]);ap.add_argument("--routing_blocks",type=parse_int_list,default=[7,8,9]);ap.add_argument("--register_k_block",type=int,default=12);ap.add_argument("--register_address_block",type=int,default=13);ap.add_argument("--late_q_block",type=int,default=22)
    ap.add_argument("--cluster_kmin",type=int,default=4);ap.add_argument("--cluster_kmax",type=int,default=14);ap.add_argument("--control_z_min",type=float,default=1.0);ap.add_argument("--same_image_depths",type=parse_int_list,default=[2,3]);ap.add_argument("--global_stack_depths",type=parse_int_list,default=[2,3,5]);ap.add_argument("--pair_screen_top_specs",type=int,default=12);ap.add_argument("--pair_screen_images",type=int,default=8);ap.add_argument("--focus_channels",type=parse_int_list,default=list(FOCUS_DEFAULT));ap.add_argument("--shortlist_channels",type=int,default=20);ap.add_argument("--shortlist_pairs",type=int,default=12);ap.add_argument("--skip_umap",action="store_true");ap.add_argument("--skip_tsne",action="store_true");ap.add_argument("--self_test",action="store_true")
    return ap.parse_args(argv)

def self_test():
    seed_all(1);base=np.array([1.,0,0]);a=np.array([.98,.1,0]);b=np.array([.98,0,.1]);combo=base+(a-base)+(b-base);m=stack_metrics(base,combo,[a,b]);assert m["nonlinear_residual_norm"]<1e-7;W=np.random.default_rng(0).normal(size=(12,3,14,14));df=pd.DataFrame([filter_metrics(W[i],i) for i in range(12)]);assert set(MORPH_FEATURES).issubset(df.columns);x=torch.arange(24).reshape(4,6).T;import tempfile
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"x.safetensors";save_safetensors({"x":x},str(p));assert torch.equal(_load_st(str(p))["x"],x)
    print("self-test OK")

def main(argv=None):
    args=parse_args(argv);seed_all()
    if args.self_test:self_test();return 0
    allowed={"ZERO","FLIP","ABS","SIGN","DC_ONLY","CENTERED","SHUFFLE"};bad=[x for x in args.conditions if x not in allowed]
    if bad:raise ValueError(f"unknown conditions: {bad}")
    if args.use_rn_token and not args.models:raise ValueError("No models selected")
    manifest=scan_images(Path(args.image_dir),args.recursive_images);print(f"[manifest] {len(manifest)} images from {args.image_dir}")
    for variant in args.models:run_one_model(variant,args,manifest)
    return 0

if __name__=="__main__":raise SystemExit(main())
