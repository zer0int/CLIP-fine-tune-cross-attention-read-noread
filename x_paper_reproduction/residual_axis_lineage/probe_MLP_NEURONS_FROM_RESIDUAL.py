#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blind MLP-neuron discovery from residual-side behavior.

Purpose
-------
Discover 4096-D MLP units *without* supplying known neuron indices.
The probe first derives residual-space directions from actual MLP writes on
frozen implicit-register tokens, then asks which hidden units explain those
directions through their activation statistics and c_proj output vectors.

The known register-neuron lists are intentionally NOT embedded in this file.
Use them only after the run as a holdout cross-check.

Default models:
  pretrained, gmp, bare_xattn, full_xattn
Default blocks:
  11, 12, 23
Default target residual axes:
  the special axes from the residual-axis lineage experiment, including 650/565.

Outputs per model
-----------------
  residual_directions.csv
  neuron_candidates.csv.gz
  neuron_axis_contributions.csv.gz
  top_neurons.txt
  blind_direction_vectors.safetensors
  audit.json

Combined outputs
----------------
  ALL_MODELS_top_neuron_comparison.csv
  ALL_MODELS_top_neuron_overlap.csv

No known hidden-neuron IDs are used for discovery or ranking.
"""

from __future__ import annotations

import argparse, contextlib, gc, hashlib, json, math, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from safetensors.torch import save_file as _save_st
except Exception as e:
    raise RuntimeError("safetensors is required") from e

EPS = 1e-8
DEFAULT_AXES = [499,250,908,953,779,196,350,139,468,469,951,211,1021,866,151,720,656,400,565,650]


def amp_context(device: str, enabled: bool):
    if enabled and str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def save_safetensors(tensors: Dict[str, torch.Tensor], filename: str, metadata: Optional[Dict[str,str]]=None):
    packed={k:v.detach().cpu().contiguous().clone() for k,v in tensors.items()}
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    _save_st(packed, filename, metadata={str(k):str(v) for k,v in (metadata or {}).items()})


def normalize(v: np.ndarray) -> np.ndarray:
    n=np.linalg.norm(v)
    return v/(n+EPS)


def zscore(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=np.float64)
    med=np.nanmedian(x); mad=np.nanmedian(np.abs(x-med))*1.4826
    if not np.isfinite(mad) or mad<EPS:
        sd=np.nanstd(x)
        if not np.isfinite(sd) or sd<EPS: return np.zeros_like(x)
        return (x-np.nanmean(x))/sd
    return (x-med)/mad


def find_repo_root() -> Path:
    here=Path(__file__).resolve()
    for p in [Path.cwd(), here.parent, *here.parents]:
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


def _resolve_xattn_state(clip_sae, resolve_fn, args):
    sd,info=resolve_fn(args.xattn_model,cache_dir=(args.hf_cache_dir or None),revision=(args.xattn_revision or None),allow_unsafe_hf_pickle=False)
    conv=getattr(getattr(clip_sae,"model",None),"convert_state_dict_inproj_to_qkv",None)
    if callable(conv): sd=conv(sd)
    return sd,info


def _load_gmp(clip_sae,load_any,args,device):
    try:
        model,preprocess,li=load_any(clip_sae,args.gmp_checkpoint,device=device,jit=False,strict=True,reuse_full_model_pickle=False)
        return _freeze(model),preprocess,{"source":args.gmp_checkpoint,"mode":"state_dict_rebuild","loader":str(li)}
    except Exception as first_error:
        src,_pp,li=load_any(clip_sae,args.gmp_checkpoint,device="cpu",jit=False,strict=True,reuse_full_model_pickle=True)
        model,preprocess,_=load_any(clip_sae,args.pretrained_model,device=device,jit=False,strict=True,allow_unsafe_hf_pickle=False)
        srcsd=src.state_dict(); conv=getattr(getattr(clip_sae,"model",None),"convert_state_dict_inproj_to_qkv",None)
        if callable(conv): srcsd=conv(srcsd)
        tgt=model.state_dict(); filt={k:v.to(dtype=tgt[k].dtype) for k,v in srcsd.items() if k.startswith("visual.") and k in tgt and tuple(v.shape)==tuple(tgt[k].shape)}
        miss=sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
        if miss: raise RuntimeError(f"GmP fallback missing visual keys: {miss[:12]}") from first_error
        model.load_state_dict(filt,strict=False); del src; gc.collect()
        return _freeze(model),preprocess,{"source":args.gmp_checkpoint,"mode":"trusted_pickle_visual_transplant","loader":str(li),"first_error":repr(first_error)}


def _load_bare_xattn(clip_sae,load_any,resolve_fn,args,device):
    model,preprocess,_=load_any(clip_sae,args.pretrained_model,device=device,jit=False,strict=True,allow_unsafe_hf_pickle=False)
    sd,info=_resolve_xattn_state(clip_sae,resolve_fn,args); tgt=model.state_dict(); filt={}; ignored=[]
    for k,v in sd.items():
        if not k.startswith("visual.") or k in {"visual.read_null_token","visual.read_null_insert_block_config"} or k not in tgt or tuple(v.shape)!=tuple(tgt[k].shape):
            ignored.append(k); continue
        filt[k]=v.to(dtype=tgt[k].dtype)
    miss=sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
    if miss: raise RuntimeError(f"bare_xattn missing visual keys: {miss[:12]}")
    inc=model.load_state_dict(filt,strict=False)
    return _freeze(model),preprocess,{"source":args.xattn_model,"mode":"bare_xattn","loaded_visual_keys":len(filt),"ignored_key_count":len(ignored),"load_missing":list(inc.missing_keys),"load_unexpected":list(inc.unexpected_keys),"loader":str(info)}


@dataclass
class Variant:
    name: str
    model: Any
    preprocess: Any
    source_info: Dict[str,Any]
    is_full_xattn: bool=False
    @property
    def visual(self): return self.model.visual
    def maybe_insert_rn(self,block_idx:int,x:torch.Tensor)->torch.Tensor:
        if self.is_full_xattn: return self.visual._maybe_insert_read_null(block_idx,x)
        return x


def load_variant(name:str,args)->Variant:
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


def build_manifest(image_dir:str,max_images:int)->pd.DataFrame:
    root=Path(image_dir)
    exts={".png",".jpg",".jpeg",".webp",".bmp"}
    files=sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if max_images>0: files=files[:max_images]
    if not files: raise FileNotFoundError(f"No images under {root}")
    return pd.DataFrame({"stim_id":[p.stem for p in files],"path":[str(p) for p in files]})


def load_batch(preprocess,rows:pd.DataFrame,device:str)->torch.Tensor:
    from PIL import Image
    xs=[]
    for p in rows.path:
        with Image.open(p) as im: xs.append(preprocess(im.convert("RGB")))
    return torch.stack(xs,0).to(device,non_blocking=True)


def frozen_register_mask(pre13_tbc: torch.Tensor,P:int,threshold:float,max_registers:int,min_registers:int):
    n=pre13_tbc[1:1+P].float().norm(dim=-1).T
    mask=torch.zeros_like(n,dtype=torch.bool)
    for i in range(n.shape[0]):
        idx=torch.nonzero(n[i]>=threshold,as_tuple=False).flatten()
        if idx.numel()<min_registers: idx=torch.topk(n[i],k=min(min_registers,P)).indices
        if max_registers>0 and idx.numel()>max_registers:
            vals=n[i,idx]; idx=idx[torch.topk(vals,k=max_registers).indices]
        mask[i,idx]=True
    return mask,n


def resolve_c_proj(blk):
    mlp=blk.mlp
    cp=getattr(mlp,"c_proj",None)
    if cp is not None and hasattr(cp,"weight") and getattr(cp,"weight").ndim==2:
        return cp
    # Fallback for custom Linear-like modules (e.g. GeometricLinear): choose the
    # last submodule exposing a 2-D weight matrix.
    weighted=[m for m in mlp.modules() if m is not mlp and hasattr(m,"weight") and isinstance(getattr(m,"weight"),torch.Tensor) and getattr(m,"weight").ndim==2]
    if len(weighted)<2: raise RuntimeError("Could not identify MLP c_proj / final Linear-like module")
    return weighted[-1]


def _stats_update(acc:Dict[str,torch.Tensor],h:torch.Tensor,mask:torch.Tensor):
    # h [B,P,M] float32 CPU/GPU; mask [B,P]
    reg=h[mask]; ordv=h[~mask]
    for prefix,v in (("reg",reg),("ord",ordv)):
        if v.numel()==0: continue
        acc[prefix+"_sum"] += v.sum(0).double().cpu()
        acc[prefix+"_abs_sum"] += v.abs().sum(0).double().cpu()
        acc[prefix+"_sq_sum"] += v.square().sum(0).double().cpu()
        acc[prefix+"_max_abs"] = torch.maximum(acc[prefix+"_max_abs"],v.abs().amax(0).double().cpu())
        acc[prefix+"_n"] += int(v.shape[0])


def init_hidden_acc(M:int):
    z=torch.zeros(M,dtype=torch.float64)
    return {"reg_sum":z.clone(),"reg_abs_sum":z.clone(),"reg_sq_sum":z.clone(),"reg_max_abs":z.clone(),"reg_n":0,
            "ord_sum":z.clone(),"ord_abs_sum":z.clone(),"ord_sq_sum":z.clone(),"ord_max_abs":z.clone(),"ord_n":0}


def finalize_hidden_acc(acc):
    out={}
    for p in ("reg","ord"):
        n=max(int(acc[p+"_n"]),1)
        mean=(acc[p+"_sum"]/n).numpy(); mean_abs=(acc[p+"_abs_sum"]/n).numpy(); rms=torch.sqrt(acc[p+"_sq_sum"]/n).numpy(); mx=acc[p+"_max_abs"].numpy()
        out[p+"_mean"]=mean;out[p+"_mean_abs"]=mean_abs;out[p+"_rms"]=rms;out[p+"_max_abs"]=mx;out[p+"_n"]=int(acc[p+"_n"])
    return out


@torch.inference_mode()
def capture_batch(variant:Variant,images:torch.Tensor,blocks_sel:Sequence[int],args):
    v=variant.visual; blocks=list(v.transformer.resblocks)
    images=images.to(dtype=v.conv1.weight.dtype); x=v._prepare_tokens(images)
    P=x.shape[0]-1
    saved={}; regmask=None; regnorm=None
    for li,blk in enumerate(blocks):
        x=variant.maybe_insert_rn(li,x)
        if li==args.register_block:
            regmask,regnorm=frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
        ln1=blk.ln_1(x)
        with amp_context(args.device,args.amp):
            attn_out,_=blk.attention(ln1,need_weights=False,capture=False)
        pa=x+attn_out
        hidden_box={}
        if li in blocks_sel:
            cp=resolve_c_proj(blk)
            def prehook(mod,inp): hidden_box["h"]=inp[0].detach()
            hh=cp.register_forward_pre_hook(prehook)
            try:
                with amp_context(args.device,args.amp): mlp_out=blk.mlp(blk.ln_2(pa))
            finally: hh.remove()
            h=hidden_box["h"][1:1+P].permute(1,0,2).float().cpu()
            m=mlp_out[1:1+P].permute(1,0,2).float().cpu()
            saved[li]=(h,m)
        else:
            with amp_context(args.device,args.amp): mlp_out=blk.mlp(blk.ln_2(pa))
        x=pa+mlp_out
    if regmask is None: regmask,regnorm=frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
    return saved,regmask.cpu(),regnorm.cpu()


def discover_model(variant:Variant,manifest:pd.DataFrame,args,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    blocks_sel=sorted(set(args.blocks)); v=variant.visual; blocks=list(v.transformer.resblocks)
    for b in blocks_sel:
        if b<0 or b>=len(blocks): raise ValueError(f"block {b} invalid")
    w0=resolve_c_proj(blocks[blocks_sel[0]]).weight
    C,M=map(int,w0.shape)
    hidden_acc={b:init_hidden_acc(M) for b in blocks_sel}
    per_image_contrast={b:[] for b in blocks_sel}; per_image_regwrite={b:[] for b in blocks_sel}
    weights={b:resolve_c_proj(blocks[b]).weight.detach().float().cpu().numpy() for b in blocks_sel}

    for st in range(0,len(manifest),args.batch_size):
        rows=manifest.iloc[st:st+args.batch_size]; images=load_batch(variant.preprocess,rows,args.device)
        saved,mask,_=capture_batch(variant,images,blocks_sel,args)
        for b,(h,m) in saved.items():
            _stats_update(hidden_acc[b],h,mask)
            # per-image residual write directions
            for i in range(h.shape[0]):
                mi=mask[i]
                if mi.any(): r=m[i][mi].mean(0)
                else: r=torch.zeros(C)
                if (~mi).any(): o=m[i][~mi].mean(0)
                else: o=torch.zeros(C)
                per_image_regwrite[b].append(r.numpy())
                per_image_contrast[b].append((r-o).numpy())
        print(f"[{variant.name}] {min(st+len(rows),len(manifest))}/{len(manifest)} images")
        del images,saved; gc.collect();
        if str(args.device).startswith("cuda"): torch.cuda.empty_cache()

    dir_rows=[]; cand_parts=[]; axis_parts=[]; dir_tensors={}
    for b in blocks_sel:
        hs=finalize_hidden_acc(hidden_acc[b]); D=np.stack(per_image_contrast[b],0).astype(np.float32); R=np.stack(per_image_regwrite[b],0).astype(np.float32)
        mean_contrast=D.mean(0); mean_reg=R.mean(0)
        Dc=D-D.mean(0,keepdims=True)
        if len(D)>=2:
            _u,_s,vh=np.linalg.svd(Dc,full_matrices=False); pc1=vh[0].astype(np.float32)
            # orient toward mean contrast when possible
            if np.dot(pc1,mean_contrast)<0: pc1=-pc1
            pc1_var=float((_s[0]**2)/(np.square(_s).sum()+EPS))
        else: pc1=normalize(mean_contrast).astype(np.float32);pc1_var=float("nan")
        d=normalize(mean_contrast); r=normalize(mean_reg); p=normalize(pc1)
        dir_tensors[f"block{b}_mean_register_contrast"]=torch.from_numpy(d.astype(np.float32))
        dir_tensors[f"block{b}_mean_register_write"]=torch.from_numpy(r.astype(np.float32))
        dir_tensors[f"block{b}_pc1_register_contrast"]=torch.from_numpy(p.astype(np.float32))
        dir_rows.append({"model":variant.name,"block":b,"contrast_norm":float(np.linalg.norm(mean_contrast)),"reg_write_norm":float(np.linalg.norm(mean_reg)),"pc1_explained_fraction":pc1_var,
                         "top_abs_axes_contrast":",".join(map(str,np.argsort(-np.abs(mean_contrast))[:16].tolist())),"top_abs_axes_regwrite":",".join(map(str,np.argsort(-np.abs(mean_reg))[:16].tolist()))})
        W=weights[b]  # [C,M]
        colnorm=np.linalg.norm(W,axis=0)+EPS
        dot_d=W.T@d; dot_r=W.T@r; dot_p=W.T@p
        align_d=dot_d/colnorm;align_r=dot_r/colnorm;align_p=dot_p/colnorm
        reg_mean=hs["reg_mean"];ord_mean=hs["ord_mean"];delta=reg_mean-ord_mean
        reg_abs=hs["reg_mean_abs"];ord_abs=hs["ord_mean_abs"];ratio=(reg_abs+1e-7)/(ord_abs+1e-7)
        expected=delta*dot_d
        abs_expected=np.abs(expected)
        # Blind score: no target residual coordinate and no known hidden index.
        s1=zscore(np.log1p(abs_expected*1000.0));s2=zscore(np.log1p(ratio));s3=zscore(np.log1p(hs["reg_max_abs"]))
        blind=1.0*s1+0.55*s2+0.25*s3
        df=pd.DataFrame({"model":variant.name,"block":b,"neuron":np.arange(M,dtype=int),
                         "reg_mean_act":reg_mean,"ord_mean_act":ord_mean,"delta_mean_act":delta,
                         "reg_mean_abs_act":reg_abs,"ord_mean_abs_act":ord_abs,"reg_abs_ratio":ratio,
                         "reg_rms_act":hs["reg_rms"],"ord_rms_act":hs["ord_rms"],"reg_max_abs_act":hs["reg_max_abs"],"ord_max_abs_act":hs["ord_max_abs"],
                         "cproj_col_norm":colnorm,"align_blind_contrast":align_d,"align_reg_write":align_r,"align_contrast_pc1":align_p,
                         "expected_blind_contribution":expected,"abs_expected_blind_contribution":abs_expected,"blind_score":blind})
        # exact explanatory fraction of mean contrast along the blind direction
        denom=np.sum(abs_expected)+EPS; df["abs_directional_share"]=abs_expected/denom
        df["rank_blind_score"]=df.blind_score.rank(ascending=False,method="min").astype(int)
        df["rank_directional_contribution"]=df.abs_expected_blind_contribution.rank(ascending=False,method="min").astype(int)
        df["rank_reg_selectivity"]=df.reg_abs_ratio.rank(ascending=False,method="min").astype(int)
        df["rank_reg_max_activation"]=df.reg_max_abs_act.rank(ascending=False,method="min").astype(int)
        cand_parts.append(df)
        # Axis-specific contributions are a SECONDARY cross-check, not used in blind ranking.
        ar=[]
        for ax in args.target_axes:
            if ax<0 or ax>=C: continue
            w=W[ax]
            regc=reg_mean*w; ordc=ord_mean*w; deltac=delta*w
            order=np.argsort(-np.abs(deltac))[:args.axis_topk]
            for j in order:
                ar.append({"model":variant.name,"block":b,"axis":int(ax),"neuron":int(j),"weight_to_axis":float(w[j]),"reg_mean_contribution":float(regc[j]),"ord_mean_contribution":float(ordc[j]),"register_contrast_contribution":float(deltac[j]),"abs_register_contrast_contribution":float(abs(deltac[j]))})
        axis_parts.append(pd.DataFrame(ar))

    dirs=pd.DataFrame(dir_rows); cands=pd.concat(cand_parts,ignore_index=True); axisdf=pd.concat(axis_parts,ignore_index=True) if axis_parts else pd.DataFrame()
    dirs.to_csv(out/"residual_directions.csv",index=False); cands.to_csv(out/"neuron_candidates.csv.gz",index=False,compression="gzip"); axisdf.to_csv(out/"neuron_axis_contributions.csv.gz",index=False,compression="gzip")
    save_safetensors(dir_tensors,str(out/"blind_direction_vectors.safetensors"),metadata={"model":variant.name,"known_neuron_ids_used":"false"})
    with (out/"top_neurons.txt").open("w",encoding="utf-8") as f:
        f.write("BLIND MLP NEURON DISCOVERY\nKnown hidden-neuron IDs are NOT inputs to this ranking.\n\n")
        for b in blocks_sel:
            g=cands[cands.block.eq(b)].sort_values(["rank_blind_score","rank_directional_contribution"]).head(args.topk)
            f.write(f"BLOCK {b}\n")
            f.write("rank  neuron  blind_score  dir_rank  reg_ratio  reg_max  align_d  abs_share\n")
            for rank,(_,r0) in enumerate(g.iterrows(),1):
                f.write(f"{rank:>4}  {int(r0.neuron):>6}  {r0.blind_score:>10.3f}  {int(r0.rank_directional_contribution):>8}  {r0.reg_abs_ratio:>9.2f}  {r0.reg_max_abs_act:>8.2f}  {r0.align_blind_contrast:>7.3f}  {r0.abs_directional_share:>9.5f}\n")
            f.write("\n")
    audit={"model":variant.name,"source_info":variant.source_info,"n_images":len(manifest),"blocks":blocks_sel,"hidden_width":M,"residual_width":C,"target_axes_secondary_only":args.target_axes,"known_hidden_neuron_ids_used":False,"register_block":args.register_block,"register_threshold":args.register_threshold}
    (out/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    return dirs,cands,axisdf


def combine_outputs(all_cands:List[pd.DataFrame],out:Path,args):
    c=pd.concat(all_cands,ignore_index=True)
    rows=[]
    for (model,b),g in c.groupby(["model","block"]):
        gg=g.sort_values(["rank_blind_score","rank_directional_contribution"]).head(args.topk)
        for rank,(_,r) in enumerate(gg.iterrows(),1): rows.append({"model":model,"block":int(b),"rank":rank,"neuron":int(r.neuron),"blind_score":float(r.blind_score),"directional_rank":int(r.rank_directional_contribution),"reg_ratio":float(r.reg_abs_ratio),"reg_max":float(r.reg_max_abs_act)})
    top=pd.DataFrame(rows);top.to_csv(out/"ALL_MODELS_top_neuron_comparison.csv",index=False)
    ov=[]
    for b in sorted(c.block.unique()):
        models=sorted(c.model.unique()); sets={m:set(c[(c.model.eq(m))&(c.block.eq(b))].nsmallest(args.overlap_topn,"rank_blind_score").neuron.astype(int)) for m in models}
        for i,a in enumerate(models):
            for bb in models[i+1:]:
                inter=len(sets[a]&sets[bb]);union=len(sets[a]|sets[bb]);ov.append({"block":int(b),"model_a":a,"model_b":bb,"topn":args.overlap_topn,"intersection":inter,"jaccard":inter/max(union,1),"shared_neurons":",".join(map(str,sorted(sets[a]&sets[bb])))})
    pd.DataFrame(ov).to_csv(out/"ALL_MODELS_top_neuron_overlap.csv",index=False)


def parse_csv_ints(s:str)->List[int]: return [int(x.strip()) for x in s.split(",") if x.strip()]
def parse_csv_strs(s:str)->List[str]: return [x.strip() for x in s.split(",") if x.strip()]


def self_test():
    rng=np.random.default_rng(0); C=8;M=16
    W=rng.normal(size=(C,M));d=normalize(rng.normal(size=C));delta=rng.normal(size=M)
    ex=delta*(W.T@d)
    assert ex.shape==(M,) and np.isfinite(ex).all()
    # ensure no known neuron list accidentally exists in globals
    forbidden=["REG_NEURONS","known_reg_neurons","known_neurons"]
    for k in forbidden: assert k not in globals()
    print("self-test passed")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo_root",default="");p.add_argument("--image_dir",default="image_sets/special_natural");p.add_argument("--output_dir",default="out_paper_reproduction/conv1/mlp_neuron_discovery")
    p.add_argument("--models",default="pretrained,gmp,bare_xattn,full_xattn");p.add_argument("--pretrained_model",default="openai/clip-vit-large-patch14");p.add_argument("--gmp_checkpoint",default="zer0int/CLIP-GmP-ViT-L-14");p.add_argument("--xattn_model",default="zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX");p.add_argument("--xattn_revision",default="");p.add_argument("--hf_cache_dir",default="")
    p.add_argument("--blocks",default="11,12,23");p.add_argument("--target_axes",default=",".join(map(str,DEFAULT_AXES)));p.add_argument("--axis_topk",type=int,default=32)
    p.add_argument("--register_block",type=int,default=13);p.add_argument("--register_threshold",type=float,default=70.0);p.add_argument("--max_registers",type=int,default=4);p.add_argument("--min_registers",type=int,default=1)
    p.add_argument("--batch_size",type=int,default=4);p.add_argument("--max_images",type=int,default=0);p.add_argument("--device",default="cuda");p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--topk",type=int,default=32);p.add_argument("--overlap_topn",type=int,default=32);p.add_argument("--self_test",action="store_true")
    args=p.parse_args()
    if args.self_test: self_test(); return 0
    args.models=parse_csv_strs(args.models);args.blocks=parse_csv_ints(args.blocks);args.target_axes=parse_csv_ints(args.target_axes)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    manifest=build_manifest(args.image_dir,args.max_images);manifest.to_csv(out/"image_manifest.csv",index=False)
    all_c=[]
    for name in args.models:
        print(f"\n=== MODEL {name} ===")
        var=load_variant(name,args);md=out/name
        dirs,cands,axisdf=discover_model(var,manifest,args,md);all_c.append(cands)
        del var;gc.collect();
        if str(args.device).startswith("cuda"): torch.cuda.empty_cache()
    combine_outputs(all_c,out,args)
    (out/"DISCOVERY_PROTOCOL.txt").write_text("Blind ranking used no known 4096-D neuron indices. Cross-check any prior neuron list only after this run.\n",encoding="utf-8")
    print(f"\nDone -> {out}")
    return 0

if __name__=="__main__": raise SystemExit(main())
