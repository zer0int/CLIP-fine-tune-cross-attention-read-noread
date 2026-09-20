#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blind B20 writeback-neuron discovery for OpenAI CLIP ViT-L/14 variants.

Scientific question
-------------------
B11/B12 contain a tiny, coherent register-construction population. B20 appears
qualitatively different: native register/workspace state is read out and written
back into ordinary patch tokens, after which B21 reformats the state for the
late B22/CLS machinery.

This probe deliberately does NOT contain any known B20 hidden-neuron IDs.
It discovers B20 units from three data-derived signals:

  1. register-conditioned PATCH WRITEBACK
     B20 ordinary patch queries are scored by how strongly they attend to the
     frozen pre-B13 register tokens. The residual direction written by the B20
     MLP into high-register-read patches versus low-register-read patches is
     derived directly from the model. Hidden units are scored by how much their
     activation contrast and c_proj output column explain that direction.

  2. REGISTER-READ COUPLING
     For every hidden unit, ordinary-patch activation is correlated with the
     amount of B20 attention that the same query places on frozen registers.
     Both an all-head mean and the automatically identified sharp register-read
     head are reported.

  3. B21 SOFTMAX SHARPNESS ATTRIBUTION
     The script identifies the B21 head whose attention becomes most sharply
     peaked because of the B20 MLP write (actual vs. counterfactual B20-MLP=0).
     It then computes activation*gradient attribution from every B20 hidden unit
     into that head's fixed-winner top-1 probability, separately for ordinary
     patches, registers, CLS, and READ_NULL when present.

The combined ranking is blind. The historical ~220 population size is NOT used
for scoring or thresholding. A top-220 list is exported only as a post-hoc
comparison convenience.

Exact group ablations of the discovered population (plus deterministic random
same-size controls) validate effects on B21 attention sharpness and the final
backbone embedding.

Default models: pretrained, gmp, bare_xattn, full_xattn.

No pickle outputs; vector artifacts use safetensors.
"""
from __future__ import annotations

import argparse, contextlib, gc, hashlib, json, math, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from safetensors.torch import save_file as _save_st, load_file as _load_st
except Exception as e:
    raise RuntimeError("safetensors is required") from e

EPS=1e-8
DEFAULT_AXES=[499,250,908,953,779,196,350,139,468,469,951,211,1021,866,151,720,656,400,565,650]
FORMAT_VERSION=2


def amp_context(device:str, enabled:bool):
    if enabled and str(device).startswith("cuda"):
        return torch.autocast("cuda",dtype=torch.float16)
    return contextlib.nullcontext()


def save_st(tensors:Dict[str,torch.Tensor],filename:Path,metadata:Optional[Dict[str,Any]]=None):
    packed={k:v.detach().cpu().contiguous().clone() for k,v in tensors.items()}
    filename.parent.mkdir(parents=True,exist_ok=True)
    _save_st(packed,str(filename),metadata={str(k):str(v) for k,v in (metadata or {}).items()})


def robust_z(x):
    x=np.asarray(x,dtype=np.float64); med=np.nanmedian(x); mad=np.nanmedian(np.abs(x-med))*1.4826
    if not np.isfinite(mad) or mad<EPS:
        sd=np.nanstd(x)
        if not np.isfinite(sd) or sd<EPS: return np.zeros_like(x)
        return (x-np.nanmean(x))/sd
    return (x-med)/mad


def normalize(v):
    v=np.asarray(v,dtype=np.float64); return v/(np.linalg.norm(v)+EPS)


def parse_ints(s): return [int(x.strip()) for x in str(s).split(",") if x.strip()]
def parse_strs(s): return [x.strip() for x in str(s).split(",") if x.strip()]


def find_repo_root()->Path:
    here=Path(__file__).resolve()
    for p in [Path.cwd(),here.parent,*here.parents]:
        if (p/"attnclip_mechinterp_sae").is_dir() and (p/"attnclip_mechinterp_xattn").is_dir() and (p/"utils_clip_loader").is_dir():
            return p
    raise FileNotFoundError("Could not locate repo root; pass --repo_root")


def _import_runtime(args):
    repo=Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    if str(repo) not in sys.path: sys.path.insert(0,str(repo))
    import attnclip_mechinterp_sae as clip_sae
    import attnclip_mechinterp_xattn as clip_x
    from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything, resolve_to_openai_state_dict
    return repo,clip_sae,clip_x,load_openai_clip_anything,resolve_to_openai_state_dict


def _freeze(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def _resolve_xattn_state(clip_sae,resolve_fn,args):
    sd,info=resolve_fn(args.xattn_model,cache_dir=(args.hf_cache_dir or None),revision=(args.xattn_revision or None),allow_unsafe_hf_pickle=False)
    conv=getattr(getattr(clip_sae,"model",None),"convert_state_dict_inproj_to_qkv",None)
    if callable(conv): sd=conv(sd)
    return sd,info


def _load_gmp(clip_sae,load_any,args,device):
    try:
        m,p,li=load_any(clip_sae,args.gmp_checkpoint,device=device,jit=False,strict=True,reuse_full_model_pickle=False)
        return _freeze(m),p,{"source":args.gmp_checkpoint,"mode":"state_dict_rebuild","loader":str(li)}
    except Exception as first_error:
        src,_pp,li=load_any(clip_sae,args.gmp_checkpoint,device="cpu",jit=False,strict=True,reuse_full_model_pickle=True)
        m,p,_=load_any(clip_sae,args.pretrained_model,device=device,jit=False,strict=True,allow_unsafe_hf_pickle=False)
        srcsd=src.state_dict(); conv=getattr(getattr(clip_sae,"model",None),"convert_state_dict_inproj_to_qkv",None)
        if callable(conv): srcsd=conv(srcsd)
        tgt=m.state_dict(); filt={k:v.to(dtype=tgt[k].dtype) for k,v in srcsd.items() if k.startswith("visual.") and k in tgt and tuple(v.shape)==tuple(tgt[k].shape)}
        miss=sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
        if miss: raise RuntimeError(f"GmP fallback missing visual keys: {miss[:12]}") from first_error
        m.load_state_dict(filt,strict=False); del src; gc.collect()
        return _freeze(m),p,{"source":args.gmp_checkpoint,"mode":"trusted_pickle_visual_transplant","loader":str(li),"first_error":repr(first_error)}


def _load_bare_xattn(clip_sae,load_any,resolve_fn,args,device):
    m,p,_=load_any(clip_sae,args.pretrained_model,device=device,jit=False,strict=True,allow_unsafe_hf_pickle=False); sd,info=_resolve_xattn_state(clip_sae,resolve_fn,args); tgt=m.state_dict(); filt={}; ignored=[]
    for k,v in sd.items():
        if not k.startswith("visual.") or k in {"visual.read_null_token","visual.read_null_insert_block_config"} or k not in tgt or tuple(v.shape)!=tuple(tgt[k].shape):
            ignored.append(k); continue
        filt[k]=v.to(dtype=tgt[k].dtype)
    miss=sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
    if miss: raise RuntimeError(f"bare_xattn missing visual keys: {miss[:12]}")
    inc=m.load_state_dict(filt,strict=False)
    return _freeze(m),p,{"source":args.xattn_model,"mode":"bare_xattn","loaded_visual_keys":len(filt),"ignored_key_count":len(ignored),"load_missing":list(inc.missing_keys),"load_unexpected":list(inc.unexpected_keys),"loader":str(info)}


@dataclass
class Variant:
    name:str; model:Any; preprocess:Any; source_info:Dict[str,Any]; is_full_xattn:bool=False
    @property
    def visual(self): return self.model.visual
    def maybe_insert_rn(self,block_idx:int,x:torch.Tensor)->torch.Tensor:
        if self.is_full_xattn: return self.visual._maybe_insert_read_null(block_idx,x)
        return x


def load_variant(name,args)->Variant:
    repo,clip_sae,clip_x,load_any,resolve_fn=_import_runtime(args)
    if name=="pretrained":
        m,p,li=load_any(clip_sae,args.pretrained_model,device=args.device,jit=False,strict=True,allow_unsafe_hf_pickle=False); return Variant(name,_freeze(m),p,{"source":args.pretrained_model,"loader":str(li)},False)
    if name=="gmp":
        m,p,i=_load_gmp(clip_sae,load_any,args,args.device); return Variant(name,m,p,i,False)
    if name=="bare_xattn":
        m,p,i=_load_bare_xattn(clip_sae,load_any,resolve_fn,args,args.device); return Variant(name,m,p,i,False)
    if name=="full_xattn":
        m,p,li=load_any(clip_x,args.xattn_model,device=args.device,jit=False,cache_dir=(args.hf_cache_dir or None),revision=(args.xattn_revision or None),strict=True,allow_unsafe_hf_pickle=False)
        return Variant(name,_freeze(m),p,{"source":args.xattn_model,"mode":"full_xattn","loader":str(li)},True)
    raise ValueError(name)


def build_manifest(image_dir,max_images):
    root=Path(image_dir); exts={".png",".jpg",".jpeg",".webp",".bmp"}; fs=sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if max_images>0: fs=fs[:max_images]
    if not fs: raise FileNotFoundError(f"No images under {root}")
    return pd.DataFrame({"stim_id":[p.stem for p in fs],"path":[str(p) for p in fs]})


def load_batch(preprocess,rows,device):
    from PIL import Image
    xs=[]
    for p in rows.path:
        with Image.open(p) as im: xs.append(preprocess(im.convert("RGB")))
    return torch.stack(xs,0).to(device,non_blocking=True)


def resolve_c_proj(blk):
    cp=getattr(blk.mlp,"c_proj",None)
    if cp is not None and hasattr(cp,"weight") and getattr(cp,"weight").ndim==2: return cp
    weighted=[m for m in blk.mlp.modules() if m is not blk.mlp and hasattr(m,"weight") and isinstance(getattr(m,"weight"),torch.Tensor) and getattr(m,"weight").ndim==2]
    if len(weighted)<2: raise RuntimeError("Could not identify MLP c_proj")
    return weighted[-1]


def frozen_register_mask(pre13_tbc,P,threshold,max_registers,min_registers):
    n=pre13_tbc[1:1+P].float().norm(dim=-1).T; mask=torch.zeros_like(n,dtype=torch.bool)
    for i in range(n.shape[0]):
        idx=torch.nonzero(n[i]>=threshold,as_tuple=False).flatten()
        if idx.numel()<min_registers: idx=torch.topk(n[i],k=min(min_registers,P)).indices
        if max_registers>0 and idx.numel()>max_registers:
            vals=n[i,idx]; idx=idx[torch.topk(vals,k=max_registers).indices]
        mask[i,idx]=True
    return mask,n


def attention_metrics(probs:torch.Tensor,regmask:torch.Tensor,P:int):
    """probs [B,H,T,S]. Return per-head sums over ordinary patch queries."""
    B,H,T,S=probs.shape; p=probs.float(); rows=[]
    ent=-(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum(-1)/math.log(max(S,2))
    top1=p.max(-1).values
    top2=torch.topk(p,k=min(2,S),dim=-1).values
    margin=top2[...,0]-(top2[...,1] if S>1 else 0.0)
    reg_mass=torch.zeros((B,H,P),dtype=torch.float32,device=p.device)
    for i in range(B):
        src=torch.zeros(S,dtype=torch.bool,device=p.device); src[1:1+P]=regmask[i].to(p.device)
        reg_mass[i]=p[i,:,1:1+P,src].sum(-1)
    ordmask=(~regmask).to(p.device)
    for h in range(H):
        vals={}
        for name,t in [("top1",top1[:,h,1:1+P]),("entropy",ent[:,h,1:1+P]),("margin",margin[:,h,1:1+P]),("reg_mass",reg_mass[:,h])]:
            vv=t[ordmask]; vals[name]=float(vv.mean().item()) if vv.numel() else float("nan")
        rows.append(vals)
    return rows,reg_mass


def init_corr_acc(M):
    return {"n":0,"sx":np.zeros(M,np.float64),"sxx":np.zeros(M,np.float64),"sy":0.0,"syy":0.0,"sxy":np.zeros(M,np.float64)}


def corr_update(acc,x:np.ndarray,y:np.ndarray):
    if x.size==0: return
    y=y.reshape(-1).astype(np.float64); x=x.astype(np.float64)
    acc["n"]+=len(y);acc["sx"]+=x.sum(0);acc["sxx"]+=(x*x).sum(0);acc["sy"]+=y.sum();acc["syy"]+=(y*y).sum();acc["sxy"]+=(x*y[:,None]).sum(0)


def corr_finalize(acc):
    n=max(acc["n"],1); num=acc["sxy"]-acc["sx"]*acc["sy"]/n; vx=acc["sxx"]-acc["sx"]**2/n; vy=acc["syy"]-acc["sy"]**2/n
    return num/np.sqrt(np.maximum(vx,0)*max(vy,0)+EPS)


def hidden_role_stats_init(M):
    z=np.zeros(M,np.float64)
    return {r+q:z.copy() for r in ["ord","reg","cls","rn"] for q in ["_sum","_abs","_max"]} | {r+"_n":0 for r in ["ord","reg","cls","rn"]}


def role_update(acc,h:torch.Tensor,regmask:torch.Tensor,P:int,has_rn:bool):
    # h [T,B,M]
    hf=h.detach().float().cpu(); B=hf.shape[1]
    roles={"cls":hf[0].reshape(-1,hf.shape[-1])}
    hp=hf[1:1+P].permute(1,0,2)
    roles["reg"]=hp[regmask.cpu()]; roles["ord"]=hp[(~regmask.cpu())]
    roles["rn"]=hf[-1].reshape(-1,hf.shape[-1]) if has_rn and hf.shape[0]>1+P else hf.new_zeros((0,hf.shape[-1]))
    for r,v in roles.items():
        if v.numel()==0: continue
        a=v.numpy().astype(np.float64); acc[r+"_sum"]+=a.sum(0);acc[r+"_abs"]+=np.abs(a).sum(0);acc[r+"_max"]=np.maximum(acc[r+"_max"],np.abs(a).max(0));acc[r+"_n"]+=a.shape[0]


def role_finalize(acc):
    out={}
    for r in ["ord","reg","cls","rn"]:
        n=max(acc[r+"_n"],1);out[r+"_mean"]=acc[r+"_sum"]/n;out[r+"_mean_abs"]=acc[r+"_abs"]/n;out[r+"_max_abs"]=acc[r+"_max"];out[r+"_n"]=acc[r+"_n"]
    return out


def forward_to_b20(variant:Variant,images,args,need_b20_probs=True):
    """Returns pa20, h20, mlp20, probs20, register mask, P, and current x20."""
    v=variant.visual; blocks=list(v.transformer.resblocks); images=images.to(dtype=v.conv1.weight.dtype); x=v._prepare_tokens(images); P=x.shape[0]-1; regmask=None
    for li,blk in enumerate(blocks[:args.block+1]):
        x=variant.maybe_insert_rn(li,x)
        if li==args.register_block:
            regmask,_=frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
        ln1=blk.ln_1(x)
        with amp_context(args.device,args.amp): attn_out,probs=blk.attention(ln1,need_weights=(li==args.block and need_b20_probs),capture=False)
        pa=x+attn_out
        if li==args.block:
            cp=resolve_c_proj(blk); box={}
            def hook(mod,inp): box["h"]=inp[0]
            hh=cp.register_forward_pre_hook(hook)
            try:
                with amp_context(args.device,args.amp): mlp=blk.mlp(blk.ln_2(pa))
            finally: hh.remove()
            x20=pa+mlp
            if regmask is None: regmask,_=frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
            return pa,box["h"],mlp,probs,regmask,P,x20
        with amp_context(args.device,args.amp): mlp=blk.mlp(blk.ln_2(pa))
        x=pa+mlp
    raise RuntimeError("B20 not reached")


def b21_probs(variant,x20,args,capture=False):
    blk=variant.visual.transformer.resblocks[args.block+1]; ln=blk.ln_1(x20)
    with amp_context(args.device,args.amp): _out,p=blk.attention(ln,need_weights=True,capture=capture)
    return p


def compute_first_pass(variant,manifest,args,out:Path):
    sig={"version":FORMAT_VERSION,"model":variant.name,"n_images":len(manifest),"paths_hash":hashlib.sha256("\n".join(manifest.path).encode()).hexdigest(),"block":args.block,"register_block":args.register_block,"threshold":args.register_threshold}
    sigp=out/"pass1_signature.json"; statp=out/"pass1_neuron_stats.csv.gz"; headp=out/"head_audit.csv"; vecp=out/"writeback_vectors.safetensors"
    if sigp.exists() and statp.exists() and headp.exists() and vecp.exists():
        if json.loads(sigp.read_text())==sig:
            print(f"[{variant.name} B20 pass1] resume")
            return pd.read_csv(statp),pd.read_csv(headp),_load_st(str(vecp))
    blk=variant.visual.transformer.resblocks[args.block]; cp=resolve_c_proj(blk); C,M=map(int,cp.weight.shape)
    roleacc=hidden_role_stats_init(M); corrmean=init_corr_acc(M)
    delta_act=[]; dirs=[]; head20_sum=None;head21_sum=None;head21cf_sum=None;n_batches=0
    for st in range(0,len(manifest),args.batch_size):
        rows=manifest.iloc[st:st+args.batch_size]; images=load_batch(variant.preprocess,rows,args.device)
        with torch.inference_mode():
            pa,h,mlp,p20,mask,P,x20=forward_to_b20(variant,images,args,True); p21=b21_probs(variant,x20,args,False); p21cf=b21_probs(variant,pa,args,False)
            has_rn=variant.is_full_xattn and h.shape[0]>1+P; role_update(roleacc,h,mask,P,has_rn)
            m20,regmass=attention_metrics(p20,mask,P); m21,_=attention_metrics(p21,mask,P); m21cf,_=attention_metrics(p21cf,mask,P)
            a20=np.array([[r[k] for k in ["top1","entropy","margin","reg_mass"]] for r in m20]); a21=np.array([[r[k] for k in ["top1","entropy","margin","reg_mass"]] for r in m21]); a21c=np.array([[r[k] for k in ["top1","entropy","margin","reg_mass"]] for r in m21cf])
            head20_sum=a20 if head20_sum is None else head20_sum+a20;head21_sum=a21 if head21_sum is None else head21_sum+a21;head21cf_sum=a21c if head21cf_sum is None else head21cf_sum+a21c;n_batches+=1
            hp=h[1:1+P].permute(1,0,2).float(); mp=mlp[1:1+P].permute(1,0,2).float(); y=regmass.mean(1).float() # all-head register-read mass
            for i in range(hp.shape[0]):
                om=~mask[i]
                xo=hp[i][om]; yo=y[i][om]; mo=mp[i][om]
                if xo.numel()==0: continue
                corr_update(corrmean,xo.cpu().numpy(),yo.cpu().numpy())
                if len(yo)>=4:
                    q1=torch.quantile(yo,.25);q3=torch.quantile(yo,.75);lo=yo<=q1;hi=yo>=q3
                    if hi.any() and lo.any():
                        delta_act.append((xo[hi].mean(0)-xo[lo].mean(0)).cpu().numpy()); dirs.append((mo[hi].mean(0)-mo[lo].mean(0)).cpu().numpy())
        print(f"[{variant.name} B20 pass1] {min(st+len(rows),len(manifest))}/{len(manifest)}")
        del images;gc.collect();
        if str(args.device).startswith("cuda"): torch.cuda.empty_cache()
    role=role_finalize(roleacc); dact=np.mean(delta_act,0) if delta_act else np.zeros(M); D=np.stack(dirs,0) if dirs else np.zeros((1,C)); meanD=D.mean(0); unitD=normalize(meanD).astype(np.float32)
    Dc=D-D.mean(0,keepdims=True)
    if len(D)>=2 and np.linalg.norm(Dc)>EPS:
        _u,s,vh=np.linalg.svd(Dc,full_matrices=False); pc1=vh[0].astype(np.float32); pc1*=1 if np.dot(pc1,meanD)>=0 else -1; pcfrac=float(s[0]**2/(np.square(s).sum()+EPS))
    else: pc1=unitD.copy();pcfrac=float("nan")
    W=cp.weight.detach().float().cpu().numpy(); colnorm=np.linalg.norm(W,axis=0)+EPS; dot=W.T@unitD; expected=dact*dot; corr=corr_finalize(corrmean)
    df=pd.DataFrame({"model":variant.name,"neuron":np.arange(M),"ord_mean_act":role["ord_mean"],"ord_mean_abs":role["ord_mean_abs"],"ord_max_abs":role["ord_max_abs"],"reg_mean_abs":role["reg_mean_abs"],"cls_mean_abs":role["cls_mean_abs"],"rn_mean_abs":role["rn_mean_abs"],"highlow_delta_act":dact,"regread_corr_allheads":corr,"cproj_col_norm":colnorm,"align_writeback_direction":dot/colnorm,"expected_writeback_contribution":expected,"abs_expected_writeback_contribution":np.abs(expected)})
    H=head20_sum.shape[0];hrows=[];h20=head20_sum/n_batches;h21=head21_sum/n_batches;h21c=head21cf_sum/n_batches
    for hidx in range(H):
        row={"model":variant.name,"head":hidx}
        for pref,a in [("b20",h20),("b21",h21),("b21_nomlp20",h21c)]:
            for j,k in enumerate(["top1","entropy","margin","reg_mass"]): row[f"{pref}_{k}"]=float(a[hidx,j])
        row["b21_delta_top1_from_mlp20"]=row["b21_top1"]-row["b21_nomlp20_top1"]
        row["b21_entropy_drop_from_mlp20"]=row["b21_nomlp20_entropy"]-row["b21_entropy"]
        row["b21_delta_margin_from_mlp20"]=row["b21_margin"]-row["b21_nomlp20_margin"]
        hrows.append(row)
    hdf=pd.DataFrame(hrows)
    # data-derived head choices
    hdf["b20_register_read_score"]=robust_z(hdf["b20_reg_mass"])+.6*robust_z(hdf["b20_top1"])-.4*robust_z(hdf["b20_entropy"])
    hdf["b21_mlp20_sharpness_score"]=robust_z(hdf["b21_delta_top1_from_mlp20"])+robust_z(hdf["b21_entropy_drop_from_mlp20"])+.5*robust_z(hdf["b21_delta_margin_from_mlp20"])+.25*robust_z(hdf["b21_top1"])
    read_head=int(hdf.sort_values("b20_register_read_score",ascending=False).iloc[0]["head"]); sharp_head=int(hdf.sort_values("b21_mlp20_sharpness_score",ascending=False).iloc[0]["head"])
    hdf["is_auto_b20_read_head"]=hdf["head"].eq(read_head);hdf["is_auto_b21_sharp_head"]=hdf["head"].eq(sharp_head)
    df.to_csv(statp,index=False,compression="gzip");hdf.to_csv(headp,index=False);save_st({"mean_writeback_direction":torch.from_numpy(unitD),"pc1_writeback_direction":torch.from_numpy(pc1),"mean_writeback_raw":torch.from_numpy(meanD.astype(np.float32))},vecp,{"model":variant.name,"pc1_fraction":pcfrac,"b20_read_head":read_head,"b21_sharp_head":sharp_head,"known_b20_neuron_ids_used":False});sigp.write_text(json.dumps(sig,indent=2))
    return df,hdf,_load_st(str(vecp))


def corr_head_second_pass_update(acc,hpatch,mask,yhead):
    # hpatch [B,P,M], yhead [B,P]
    for i in range(hpatch.shape[0]):
        om=~mask[i];corr_update(acc,hpatch[i][om].detach().float().cpu().numpy(),yhead[i][om].detach().float().cpu().numpy())


def attribution_pass(variant,manifest,args,out:Path,sharp_head:int,read_head:int,M:int):
    cdir=out/"attribution_batches";cdir.mkdir(exist_ok=True); blk=variant.visual.transformer.resblocks[args.block]; cp=resolve_c_proj(blk)
    corrcrit=init_corr_acc(M)
    # correlation must be recomputed globally; caches contain sufficient sums per batch
    attr_keys=["ord_signed","ord_abs","reg_signed","reg_abs","cls_signed","cls_abs","rn_signed","rn_abs"]
    totals={k:np.zeros(M,np.float64) for k in attr_keys};counts={r:0 for r in ["ord","reg","cls","rn"]}
    for st in range(0,len(manifest),args.batch_size):
        rows=manifest.iloc[st:st+args.batch_size];cache=cdir/f"batch_{st:06d}.safetensors"
        if cache.exists():
            z=_load_st(str(cache));
            for k in attr_keys: totals[k]+=z[k].double().numpy()
            for r in counts: counts[r]+=int(z[f"count_{r}"].item())
            # crit-head corr sums
            if "corr_n" in z:
                tmp={"n":int(z["corr_n"].item()),"sx":z["corr_sx"].double().numpy(),"sxx":z["corr_sxx"].double().numpy(),"sy":float(z["corr_sy"].item()),"syy":float(z["corr_syy"].item()),"sxy":z["corr_sxy"].double().numpy()}
                corrcrit["n"]+=tmp["n"];corrcrit["sx"]+=tmp["sx"];corrcrit["sxx"]+=tmp["sxx"];corrcrit["sy"]+=tmp["sy"];corrcrit["syy"]+=tmp["syy"];corrcrit["sxy"]+=tmp["sxy"]
            print(f"[{variant.name} B20 attr] resume {min(st+len(rows),len(manifest))}/{len(manifest)}");continue
        images=load_batch(variant.preprocess,rows,args.device)
        # forward to B20 under no-grad to get the real hidden state and B20 probs
        with torch.no_grad(): pa,h0,_mlp,p20,mask,P,_x20=forward_to_b20(variant,images,args,True)
        # critical B20 register-read coupling
        _,regmass=attention_metrics(p20,mask,P); ycrit=regmass[:,read_head]
        localcorr=init_corr_acc(M);corr_head_second_pass_update(localcorr,h0[1:1+P].permute(1,0,2),mask,ycrit)
        # exact differentiable c_proj from the captured hidden state -> B21 attention
        h=h0.detach().clone().requires_grad_(True)
        with amp_context(args.device,args.amp): mlp=cp(h); x20=pa.detach()+mlp; p21=b21_probs(variant,x20,args,True)
        logits=variant.visual.transformer.resblocks[args.block+1].attn.last_logits[:,sharp_head,1:1+P,:]
        probs=p21[:,sharp_head,1:1+P,:]
        om=(~mask).to(probs.device); winner=probs.detach().argmax(-1); pwin=probs.gather(-1,winner.unsqueeze(-1)).squeeze(-1)
        scalar=pwin[om].mean()
        grad=torch.autograd.grad(scalar,h,retain_graph=False,create_graph=False)[0];attr=(h*grad).detach().float().cpu(); hc=h.detach().float().cpu(); mc=mask.cpu()
        batch_sums={k:torch.zeros(M,dtype=torch.float64) for k in attr_keys};batch_counts={r:0 for r in counts}
        hp=attr[1:1+P].permute(1,0,2)
        for r,vals in [("ord",hp[~mc]),("reg",hp[mc]),("cls",attr[0])]:
            if vals.numel(): batch_sums[r+"_signed"]+=vals.double().sum(0);batch_sums[r+"_abs"]+=vals.abs().double().sum(0);batch_counts[r]+=vals.shape[0]
        if variant.is_full_xattn and attr.shape[0]>1+P:
            vals=attr[-1];batch_sums["rn_signed"]+=vals.double().sum(0);batch_sums["rn_abs"]+=vals.abs().double().sum(0);batch_counts["rn"]+=vals.shape[0]
        save_payload={k:v for k,v in batch_sums.items()}|{f"count_{r}":torch.tensor(batch_counts[r]) for r in counts}|{"corr_n":torch.tensor(localcorr["n"]),"corr_sx":torch.from_numpy(localcorr["sx"]),"corr_sxx":torch.from_numpy(localcorr["sxx"]),"corr_sy":torch.tensor(localcorr["sy"],dtype=torch.float64),"corr_syy":torch.tensor(localcorr["syy"],dtype=torch.float64),"corr_sxy":torch.from_numpy(localcorr["sxy"]),"sharpness_scalar":scalar.detach().cpu().float()}
        save_st(save_payload,cache,{"model":variant.name,"start":st,"sharp_head":sharp_head,"read_head":read_head})
        for k in attr_keys: totals[k]+=batch_sums[k].numpy()
        for r in counts: counts[r]+=batch_counts[r]
        corrcrit["n"]+=localcorr["n"];corrcrit["sx"]+=localcorr["sx"];corrcrit["sxx"]+=localcorr["sxx"];corrcrit["sy"]+=localcorr["sy"];corrcrit["syy"]+=localcorr["syy"];corrcrit["sxy"]+=localcorr["sxy"]
        print(f"[{variant.name} B20 attr] {min(st+len(rows),len(manifest))}/{len(manifest)}")
        del images,h,grad,attr;gc.collect();
        if str(args.device).startswith("cuda"): torch.cuda.empty_cache()
    outd={}
    for r in counts:
        n=max(counts[r],1);outd[r+"_attr_signed_mean"]=totals[r+"_signed"]/n;outd[r+"_attr_abs_mean"]=totals[r+"_abs"]/n
    outd["regread_corr_autohead"]=corr_finalize(corrcrit);return pd.DataFrame(outd)


def build_ranking(pass1:pd.DataFrame,attr:pd.DataFrame,args):
    df=pass1.merge(attr,left_index=True,right_index=True); 
    # three independent evidence streams
    df["z_writeback"]=robust_z(np.log1p(df["abs_expected_writeback_contribution"].to_numpy()*1000.0))
    df["z_regread"]=robust_z(np.abs(df["regread_corr_autohead"].to_numpy()))
    df["z_sharpness"]=robust_z(np.log1p(df["ord_attr_abs_mean"].to_numpy()*1e7))
    df["combined_score"]=df["z_writeback"]+0.8*df["z_regread"]+1.0*df["z_sharpness"]+0.10*robust_z(np.log1p(df["ord_max_abs"].to_numpy()))
    for c in ["combined_score","abs_expected_writeback_contribution","ord_attr_abs_mean","regread_corr_autohead"]:
        df["rank_"+c]=df[c].abs().rank(ascending=False,method="min").astype(int)
    # Data-derived families: no hard-coded size.
    df["evidence_votes"]=(df["z_writeback"]>=2).astype(int)+(df["z_regread"]>=2).astype(int)+(df["z_sharpness"]>=2).astype(int)
    df["consensus_family"]=(df["combined_score"]>=3.0)&(df["evidence_votes"]>=2)
    order=df.sort_values("ord_attr_abs_mean",ascending=False).index.to_numpy();vals=df.loc[order,"ord_attr_abs_mean"].to_numpy();cum=np.cumsum(vals)/(vals.sum()+EPS)
    core=set(order[cum<=.80].tolist());extended=set(order[cum<=.95].tolist())
    if len(order): core.add(int(order[0]));extended.add(int(order[0]))
    df["sharpness_core80"]=df.index.isin(core);df["sharpness_extended95"]=df.index.isin(extended)
    df["top220_posthoc"]=df["rank_combined_score"]<=220
    return df


def plot_outputs(df,hdf,out:Path):
    try:
        import matplotlib.pyplot as plt
    except Exception: return
    p=out/"plots";p.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,6));x=hdf["head"]
    ax.plot(x,hdf["b21_top1"],marker="o",label="B21 actual top1");ax.plot(x,hdf["b21_nomlp20_top1"],marker="o",label="B21 with B20 MLP=0")
    ax.set_xlabel("head");ax.set_ylabel("mean top-1 attention mass (ordinary queries)");ax.set_title("B20 MLP-induced B21 softmax sharpening");ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(p/"01_B21_HEAD_SHARPNESS.png",dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6));ax.plot(hdf["head"],hdf["b20_reg_mass"],marker="o",label="ordinary query -> frozen REG mass");ax.plot(hdf["head"],hdf["b20_top1"],marker="o",label="top1 attention mass");ax.set_xlabel("B20 head");ax.set_title("B20 native register-read heads");ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(p/"02_B20_REGISTER_READ_HEADS.png",dpi=190);plt.close(fig)
    g=df.sort_values("combined_score",ascending=False).reset_index(drop=True);fig,ax=plt.subplots(figsize=(10,6));ax.plot(np.arange(1,len(g)+1),g["combined_score"]);ax.axvline(220,ls="--",alpha=.5,label="220 post-hoc reference");ax.set_xlabel("rank");ax.set_ylabel("blind combined score");ax.set_title("B20 neuron discovery rank curve");ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(p/"03_B20_NEURON_RANK_CURVE.png",dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,7));ax.scatter(df["regread_corr_autohead"],df["ord_attr_abs_mean"],s=10,alpha=.5);top=df.nsmallest(24,"rank_combined_score")
    for _,r in top.iterrows(): ax.annotate(str(int(r.neuron)),(r.regread_corr_autohead,r.ord_attr_abs_mean),fontsize=7)
    ax.set_xlabel("corr(hidden activation, B20 REG-read mass)");ax.set_ylabel("|act×grad| into B21 sharpness");ax.set_title("B20 register-read coupling vs B21 softmax control");ax.grid(alpha=.2);fig.tight_layout();fig.savefig(p/"04_B20_NEURON_MECHANISM_SCATTER.png",dpi=190);plt.close(fig)


def special_axis_table(variant,ranking,args):
    cp=resolve_c_proj(variant.visual.transformer.resblocks[args.block]);W=cp.weight.detach().float().cpu().numpy();rows=[]
    for ax in args.target_axes:
        if not (0<=ax<W.shape[0]): continue
        contrib=ranking["ord_mean_act"].to_numpy()*W[ax]
        ids=np.argsort(-np.abs(contrib))[:args.axis_topk]
        for j in ids: rows.append({"model":variant.name,"axis":ax,"neuron":int(j),"weight_to_axis":float(W[ax,j]),"ordinary_mean_contribution":float(contrib[j]),"combined_rank":int(ranking.loc[j,"rank_combined_score"])})
    return pd.DataFrame(rows)


def apply_hidden_ablation(cp,idx:Sequence[int]):
    ids=torch.as_tensor(list(idx),dtype=torch.long)
    def hook(mod,inp):
        h=inp[0].clone();h.index_fill_(-1,ids.to(h.device),0);return (h,)
    return cp.register_forward_pre_hook(hook)


def full_forward_ablation(variant,images,args,idx:Sequence[int],sharp_head:int):
    v=variant.visual;blocks=list(v.transformer.resblocks);images=images.to(dtype=v.conv1.weight.dtype);x=v._prepare_tokens(images);P=x.shape[0]-1;regmask=None;p21_saved=None
    for li,blk in enumerate(blocks):
        x=variant.maybe_insert_rn(li,x)
        if li==args.register_block:regmask,_=frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
        with amp_context(args.device,args.amp): attn,probs=blk.attention(blk.ln_1(x),need_weights=(li==args.block+1),capture=False)
        pa=x+attn;hh=None
        if li==args.block and idx:hh=apply_hidden_ablation(resolve_c_proj(blk),idx)
        try:
            with amp_context(args.device,args.amp): mlp=blk.mlp(blk.ln_2(pa))
        finally:
            if hh is not None:hh.remove()
        x=pa+mlp
        if li==args.block+1:p21_saved=probs.detach()
    with amp_context(args.device,args.amp):emb=v._finalize_cls(x)
    emb=F.normalize(emb.float(),dim=-1)
    # sharpness summary
    if regmask is None:regmask,_=frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
    m,_=attention_metrics(p21_saved,regmask,P);return emb,float(m[sharp_head]["top1"]),float(m[sharp_head]["entropy"]),float(m[sharp_head]["margin"])


def ablation_validation(variant,manifest,ranking,args,out,sharp_head):
    ad=out/"ablation_cache";ad.mkdir(exist_ok=True);sizes=[x for x in args.ablation_sizes if x>0]
    rows=[];ranked=ranking.sort_values("combined_score",ascending=False).neuron.astype(int).tolist()
    # Baseline is expensive; compute exactly once per model and persist it.
    bp=ad/"baseline_embeddings.safetensors";bmp=ad/"baseline_metrics.json"
    if bp.exists() and bmp.exists():
        base_all=_load_st(str(bp))["embeddings"].float();bmet=json.loads(bmp.read_text())
        print(f"[{variant.name} ablate] resume baseline")
    else:
        emb=[];tops=[];ents=[];margs=[]
        for st in range(0,len(manifest),args.batch_size):
            ims=load_batch(variant.preprocess,manifest.iloc[st:st+args.batch_size],args.device)
            with torch.inference_mode(): b,bt,be,bm=full_forward_ablation(variant,ims,args,[],sharp_head)
            emb.append(b.cpu());tops.append(bt);ents.append(be);margs.append(bm);del ims
        base_all=torch.cat(emb,0);bmet={"mean_b21_top1":float(np.mean(tops)),"mean_b21_entropy":float(np.mean(ents)),"mean_b21_margin":float(np.mean(margs))};save_st({"embeddings":base_all},bp,{"model":variant.name});bmp.write_text(json.dumps(bmet,indent=2))
    jobs=[("discovered",s,0) for s in sizes]+[("random",s,r) for s in sizes for r in range(args.random_repeats)]
    for kind,size,rep in jobs:
        size=min(size,len(ranked));key=f"{kind}_k{size}_r{rep}";jp=ad/f"{key}.json"
        if jp.exists():rows.append(json.loads(jp.read_text()));continue
        if kind=="discovered": ids=ranked[:size]
        else:
            rng=np.random.default_rng(int(hashlib.sha256(f"{variant.name}:{size}:{rep}".encode()).hexdigest()[:16],16));pool=np.array(ranked[size:],dtype=int) if len(ranked)-size>=size else np.arange(len(ranked));ids=rng.choice(pool,size=size,replace=False).tolist()
        dists=[];tops=[];ents=[];margs=[];off=0
        for st in range(0,len(manifest),args.batch_size):
            ims=load_batch(variant.preprocess,manifest.iloc[st:st+args.batch_size],args.device)
            with torch.inference_mode(): z,t,e,m=full_forward_ablation(variant,ims,args,ids,sharp_head)
            bb=base_all[off:off+len(z)].to(z.device);off+=len(z);dists.extend((1-(bb*z).sum(-1)).cpu().tolist());tops.append(t);ents.append(e);margs.append(m);del ims
        row={"model":variant.name,"kind":kind,"size":size,"repeat":rep,"mean_final_cosine_distance":float(np.mean(dists)),"mean_b21_top1":float(np.mean(tops)),"delta_b21_top1_vs_baseline":float(np.mean(tops)-bmet["mean_b21_top1"]),"mean_b21_entropy":float(np.mean(ents)),"delta_b21_entropy_vs_baseline":float(np.mean(ents)-bmet["mean_b21_entropy"]),"mean_b21_margin":float(np.mean(margs)),"delta_b21_margin_vs_baseline":float(np.mean(margs)-bmet["mean_b21_margin"]),"neuron_ids":",".join(map(str,ids))};jp.write_text(json.dumps(row,indent=2));rows.append(row);print(f"[{variant.name} ablate] {kind} k={size} r={rep}")
    df=pd.DataFrame(rows);df.to_csv(out/"ablation_validation.csv",index=False);return df

def write_report(ranking,hdf,abl,out:Path,variant,args):
    read_head=int(hdf.loc[hdf["is_auto_b20_read_head"],"head"].iloc[0]);sharp_head=int(hdf.loc[hdf["is_auto_b21_sharp_head"],"head"].iloc[0]);top=ranking.sort_values("combined_score",ascending=False)
    core=top[top["sharpness_core80"]].neuron.astype(int).tolist();ext=top[top["sharpness_extended95"]].neuron.astype(int).tolist();cons=top[top["consensus_family"]].neuron.astype(int).tolist();t220=top.head(220).neuron.astype(int).tolist()
    lines=["B20 BLIND WRITEBACK-NEURON DISCOVERY","="*40,"",f"model: {variant.name}","known B20 hidden-neuron IDs used for ranking: NO",f"automatic B20 register-read head: H{read_head}",f"automatic B21 MLP20-sharpened head: H{sharp_head}","",f"consensus-family size: {len(cons)}",f"80% B21-sharpness attribution core size: {len(core)}",f"95% B21-sharpness attribution extended size: {len(ext)}","", "TOP 32", "rank neuron score writeback_rank regread_corr sharp_attr"]
    for rank,(_,r) in enumerate(top.head(32).iterrows(),1):lines.append(f"{rank:>4} {int(r.neuron):>6} {r.combined_score:>8.3f} {int(r.rank_abs_expected_writeback_contribution):>8} {r.regread_corr_autohead:>10.4f} {r.ord_attr_abs_mean:>11.6g}")
    lines += ["","CONSENSUS FAMILY IDS",",".join(map(str,cons)),"","SHARPNESS CORE80 IDS",",".join(map(str,core)),"","SHARPNESS EXTENDED95 IDS",",".join(map(str,ext)),"","POST-HOC TOP220 IDS (220 WAS NOT USED IN RANKING)",",".join(map(str,t220))]
    (out/"B20_POPULATION.txt").write_text("\n".join(lines),encoding="utf-8")
    (out/"B20_population_indices.json").write_text(json.dumps({"model":variant.name,"known_ids_used":False,"auto_b20_read_head":read_head,"auto_b21_sharp_head":sharp_head,"consensus_family":cons,"sharpness_core80":core,"sharpness_extended95":ext,"top220_posthoc":t220},indent=2))


def combine(models,out,args):
    tabs=[]
    for m in models:
        p=out/m/"b20_neuron_candidates.csv.gz"
        if p.exists():
            d=pd.read_csv(p);tabs.append(d.nsmallest(args.report_topn,"rank_combined_score")[["model","neuron","rank_combined_score","combined_score","regread_corr_autohead","ord_attr_abs_mean","abs_expected_writeback_contribution"]])
    if tabs:pd.concat(tabs,ignore_index=True).to_csv(out/"ALL_MODELS_B20_top_neurons.csv",index=False)
    ov=[]
    loaded={m:pd.read_csv(out/m/"b20_neuron_candidates.csv.gz") for m in models if (out/m/"b20_neuron_candidates.csv.gz").exists()}
    for i,a in enumerate(sorted(loaded)):
        A=set(loaded[a].nsmallest(args.overlap_topn,"rank_combined_score").neuron.astype(int))
        for b in sorted(loaded)[i+1:]:
            B=set(loaded[b].nsmallest(args.overlap_topn,"rank_combined_score").neuron.astype(int));inter=A&B;ov.append({"model_a":a,"model_b":b,"topn":args.overlap_topn,"intersection":len(inter),"jaccard":len(inter)/max(len(A|B),1),"shared_neurons":",".join(map(str,sorted(inter)))})
    if ov:pd.DataFrame(ov).to_csv(out/"ALL_MODELS_B20_overlap.csv",index=False)


def self_test():
    rng=np.random.default_rng(1);M=4096;acc=init_corr_acc(M);x=rng.normal(size=(100,M));y=x[:,7]*.8+rng.normal(size=100)*.1;corr_update(acc,x,y);r=corr_finalize(acc);assert np.argmax(np.abs(r))==7
    # pandas .mode footgun regression: explicit column indexing only
    d=pd.DataFrame({"mode":["a","b"]});assert len(d[d["mode"].eq("a")])==1
    # shared-storage safetensors regression
    t=torch.randn(3,4);tmp=Path("_b20_selftest.safetensors");save_st({"a":t,"b":t[0]},tmp);tmp.unlink()
    print("self-test passed")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo_root",default="");p.add_argument("--image_dir",default="image_sets/special_natural");p.add_argument("--output_dir",default="out_paper_reproduction/conv1/b20_writeback_neurons")
    p.add_argument("--models",default="pretrained,gmp,bare_xattn,full_xattn");p.add_argument("--pretrained_model",default="openai/clip-vit-large-patch14");p.add_argument("--gmp_checkpoint",default="zer0int/CLIP-GmP-ViT-L-14");p.add_argument("--xattn_model",default="zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX");p.add_argument("--xattn_revision",default="");p.add_argument("--hf_cache_dir",default="")
    p.add_argument("--block",type=int,default=20);p.add_argument("--register_block",type=int,default=13);p.add_argument("--register_threshold",type=float,default=70.0);p.add_argument("--max_registers",type=int,default=4);p.add_argument("--min_registers",type=int,default=1)
    p.add_argument("--target_axes",default=",".join(map(str,DEFAULT_AXES)));p.add_argument("--axis_topk",type=int,default=32)
    p.add_argument("--batch_size",type=int,default=4);p.add_argument("--max_images",type=int,default=0);p.add_argument("--device",default="cuda");p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--ablation_sizes",default="16,32,64,128,220");p.add_argument("--random_repeats",type=int,default=2);p.add_argument("--report_topn",type=int,default=320);p.add_argument("--overlap_topn",type=int,default=220);p.add_argument("--skip_ablation",action="store_true");p.add_argument("--self_test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test();return 0
    args.models=parse_strs(args.models);args.target_axes=parse_ints(args.target_axes);args.ablation_sizes=parse_ints(args.ablation_sizes)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);manifest=build_manifest(args.image_dir,args.max_images);manifest.to_csv(out/"image_manifest.csv",index=False)
    for name in args.models:
        print(f"\n=== MODEL {name} ===");variant=load_variant(name,args);md=out/name;md.mkdir(parents=True,exist_ok=True)
        pass1,hdf,vec=compute_first_pass(variant,manifest,args,md);M=len(pass1);read_head=int(hdf.loc[hdf["is_auto_b20_read_head"],"head"].iloc[0]);sharp_head=int(hdf.loc[hdf["is_auto_b21_sharp_head"],"head"].iloc[0]);attr=attribution_pass(variant,manifest,args,md,sharp_head,read_head,M);ranking=build_ranking(pass1,attr,args);ranking.to_csv(md/"b20_neuron_candidates.csv.gz",index=False,compression="gzip");special_axis_table(variant,ranking,args).to_csv(md/"b20_special_axis_contributions.csv.gz",index=False,compression="gzip");plot_outputs(ranking,hdf,md)
        abl=pd.DataFrame()
        if not args.skip_ablation:abl=ablation_validation(variant,manifest,ranking,args,md,sharp_head)
        write_report(ranking,hdf,abl,md,variant,args);(md/"audit.json").write_text(json.dumps({"format_version":FORMAT_VERSION,"model":name,"source_info":variant.source_info,"n_images":len(manifest),"known_b20_hidden_neuron_ids_used":False,"historical_population_size_used_for_ranking":False,"posthoc_top220_exported":True,"block":args.block,"auto_b20_read_head":read_head,"auto_b21_sharp_head":sharp_head},indent=2));del variant;gc.collect();
        if str(args.device).startswith("cuda"):torch.cuda.empty_cache()
    combine(args.models,out,args);(out/"DISCOVERY_PROTOCOL.txt").write_text("B20 hidden-neuron discovery used no known B20 neuron IDs and did not use 220 as a ranking threshold. top220 is exported only for post-hoc comparison.\n",encoding="utf-8");print(f"\nDone -> {out}");return 0

if __name__=="__main__":raise SystemExit(main())
