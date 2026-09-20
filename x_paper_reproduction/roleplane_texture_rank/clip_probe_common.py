#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib, json, math, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

EPS = 1e-12
SEED = 20260917

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed%(2**32)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def stable_seed(text:str)->int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"little") & 0x7fffffff

def pearson(a,b):
    a=np.asarray(a,np.float64).reshape(-1); b=np.asarray(b,np.float64).reshape(-1)
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<2: return float("nan")
    a=a[ok]-a[ok].mean(); b=b[ok]-b[ok].mean()
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d>EPS else float("nan")

def cosine_np(a,b):
    a=np.asarray(a,np.float64).reshape(-1); b=np.asarray(b,np.float64).reshape(-1)
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d>EPS else float("nan")

def robust_z_rows(x:torch.Tensor):
    med=x.median(-1,keepdim=True).values
    mad=(x-med).abs().median(-1,keepdim=True).values
    sc=1.4826*mad
    std=x.std(-1,keepdim=True,unbiased=False)
    sc=torch.where(sc>1e-6,sc,std.clamp_min(1e-6))
    return (x-med)/sc

def effective_rank_sv(s):
    p=s/s.sum(-1,keepdim=True).clamp_min(EPS)
    return torch.exp(-(p*torch.log(p.clamp_min(EPS))).sum(-1))

def effective_rank_energy(s):
    e=s.square(); p=e/e.sum(-1,keepdim=True).clamp_min(EPS)
    return torch.exp(-(p*torch.log(p.clamp_min(EPS))).sum(-1))

def stable_rank(s):
    return s.square().sum(-1)/s[...,0].square().clamp_min(EPS)

def rel_rank(s,rel:float):
    return (s > rel*s[...,:1]).sum(-1)

def load_model(name="openai/clip-vit-large-patch14",device="cuda"):
    repo=next((p for p in [Path.cwd().resolve(), *Path(__file__).resolve().parents] if (p/"oaicliporg").is_dir() and (p/"utils_clip_loader").is_dir()),None)
    if repo is None: raise FileNotFoundError("Could not locate repo root containing oaicliporg and utils_clip_loader")
    if str(repo) not in sys.path: sys.path.insert(0,str(repo))
    import oaicliporg as clip
    from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything
    model,preprocess,_=load_openai_clip_anything(clip,name,device=device,jit=False,strict=True,allow_unsafe_hf_pickle=False); model.eval()
    return model,preprocess

def model_geometry(model):
    P=int(model.visual.positional_embedding.shape[0]-1); G=round(math.sqrt(P))
    if G*G!=P: raise RuntimeError("non-square patch grid")
    ks=model.visual.conv1.kernel_size; patch=int(ks[0] if isinstance(ks,tuple) else ks)
    return G*patch,G,patch,int(model.visual.positional_embedding.shape[1]),len(model.visual.transformer.resblocks),int(model.visual.transformer.resblocks[0].attn.num_heads)

def qkv_from_attn(attn,z):
    L,B,D=z.shape; H=int(attn.num_heads); dh=D//H
    if hasattr(attn,"q_proj") and hasattr(attn,"k_proj"):
        q,k,v=attn.q_proj(z),attn.k_proj(z),attn.v_proj(z)
    else:
        W=attn.in_proj_weight; b=attn.in_proj_bias
        qw,kw,vw=W.chunk(3,0); qb,kb,vb=(b.chunk(3,0) if b is not None else (None,None,None))
        q,k,v=F.linear(z,qw,qb),F.linear(z,kw,kb),F.linear(z,vw,vb)
    def rs(t): return t.permute(1,0,2).reshape(B,L,H,dh).permute(0,2,1,3).contiguous()
    return rs(q),rs(k),rs(v)

def prepare_tokens(model,images_cpu):
    device=next(model.parameters()).device; dtype=model.visual.conv1.weight.dtype
    im=images_cpu.to(device=device,dtype=dtype)
    conv=model.visual.conv1(im); B,C,G,_=conv.shape
    x=conv.reshape(B,C,G*G).permute(0,2,1)
    cls=model.visual.class_embedding.to(x.dtype)+torch.zeros((B,1,C),device=x.device,dtype=x.dtype)
    x=torch.cat([cls,x],1)+model.visual.positional_embedding.to(x.dtype)
    x=model.visual.ln_pre(x).permute(1,0,2)
    return x,conv

def final_embedding(model,x):
    y=model.visual.ln_post(x.permute(1,0,2)[:,0,:])
    return y@model.visual.proj if model.visual.proj is not None else y

def manual_encode(model,batch):
    with torch.inference_mode():
        x,_=prepare_tokens(model,batch)
        for blk in model.visual.transformer.resblocks: x=blk(x)
        return final_embedding(model,x).float().cpu()

def validate_forward(model,preprocess,size,device):
    rng=np.random.default_rng(11); a=np.clip(rng.normal(.5,.18,(size,size,3)),0,1)
    im=Image.fromarray((a*255).astype(np.uint8),"RGB"); batch=preprocess(im).unsqueeze(0)
    got=manual_encode(model,batch)
    with torch.inference_mode(): ref=model.encode_image(batch.to(device)).float().cpu()
    cos=float(F.cosine_similarity(got,ref,dim=-1).item()); rel=float((got-ref).norm()/ref.norm().clamp_min(EPS))
    if cos<.99999 or rel>5e-4: raise RuntimeError(f"manual forward parity failed cos={cos} rel={rel}")
    return {"cosine":cos,"relative_l2":rel,"max_abs":float((got-ref).abs().max())}

def preprocess_items(preprocess,items):
    return torch.stack([preprocess(x["image"].convert("RGB")) for x in items])

def load_dtd():
    from datasets import load_dataset
    ds=load_dataset("tanganke/dtd",split="train")
    names=list(ds.features["label"].names)
    return ds,names

def balanced_dtd_indices(ds,names,basis_per_class,eval_per_class,seed):
    rng=np.random.default_rng(seed); labels=np.asarray(ds["label"],np.int64)
    basis=[]; ev=[]; audit=[]
    for y,name in enumerate(names):
        ids=np.flatnonzero(labels==y); rng.shuffle(ids)
        nb=min(basis_per_class,len(ids)); ne=min(eval_per_class,len(ids)-nb)
        basis+=ids[:nb].tolist(); ev+=ids[nb:nb+ne].tolist()
        audit.append({"label":y,"class_name":name,"available":len(ids),"basis_n":nb,"eval_n":ne})
    return basis,ev,audit

def dtd_items(ds,names,indices):
    out=[]
    for idx in indices:
        r=ds[int(idx)]; y=int(r["label"])
        out.append({"image":r["image"].convert("RGB"),"source":"dtd","group":names[y],"item_id":f"dtd_{idx}",
                    "params":{"dataset_index":int(idx),"label":y,"class_name":names[y]}})
    return out

@dataclass(frozen=True)
class PatternSpec:
    family:str; item_id:str; params:dict; seed:int

def _colorize(v,seed):
    v=np.asarray(v,np.float64); v=(v-v.min())/(v.max()-v.min()+EPS)
    ph=np.random.default_rng(seed).uniform(0,2*np.pi)
    return np.clip(np.stack([.12+.78*v,.12+.78*(.5+.5*np.sin(2*np.pi*v+ph)),.12+.78*(1-v)],-1),0,1)

def make_pattern_specs(seed):
    out=[]
    for a in range(0,180,15):
        for f in [1,2,3,4,6,8,12]:
            for pi,ph in enumerate([0.,math.pi/2]):
                out.append(PatternSpec("sine",f"sine_a{a:03d}_f{f:02d}_p{pi}",{"angle_deg":a,"frequency":f,"phase":ph},stable_seed(f"{seed}:s:{a}:{f}:{pi}")))
    for i in range(48):
        out.append(PatternSpec("fractal_sine",f"fractal_{i:03d}",{"base_frequency":1.2+(i%6)*.4,"octaves":5},stable_seed(f"{seed}:f:{i}")))
    for i in range(40):
        out.append(PatternSpec("tiled_smooth_noise",f"perlin_{i:03d}",{"tile":56,"blur":1+(i%5)*.6},stable_seed(f"{seed}:p:{i}")))
    for i in range(40):
        out.append(PatternSpec("tiled_voronoi",f"vor_{i:03d}",{"tile":56,"points":4+(i%10)},stable_seed(f"{seed}:v:{i}")))
    for i in range(32):
        out.append(PatternSpec("tiled_checker_ripple",f"checker_{i:03d}",{"tile":56,"period":2+(i%8)},stable_seed(f"{seed}:c:{i}")))
    return out

def _smooth_tile(tile,seed,blur):
    rng=np.random.default_rng(seed); a=rng.normal(size=(tile,tile))
    for sh in (1,2,4): a=(a+np.roll(a,sh,0)+np.roll(a,-sh,0)+np.roll(a,sh,1)+np.roll(a,-sh,1))/5
    a=(a-a.min())/(a.max()-a.min()+EPS)
    return np.asarray(Image.fromarray((a*255).astype(np.uint8),"L").filter(ImageFilter.GaussianBlur(blur)),np.float64)/255

def _vor(tile,npts,rng):
    pts=rng.uniform(0,tile,(npts,2)); yy,xx=np.mgrid[0:tile,0:tile].astype(np.float64)
    best=np.full((tile,tile),np.inf); second=np.full_like(best,np.inf)
    for py,px in pts:
        dy=np.minimum(np.abs(yy-py),tile-np.abs(yy-py)); dx=np.minimum(np.abs(xx-px),tile-np.abs(xx-px))
        d=np.sqrt(dx*dx+dy*dy); sw=d<best; second=np.where(sw,best,np.minimum(second,d)); best=np.where(sw,d,best)
    return second-best

def render_pattern(s,size):
    rng=np.random.default_rng(s.seed); yy,xx=np.mgrid[0:size,0:size].astype(np.float64); xn=xx/(size-1); yn=yy/(size-1)
    if s.family=="sine":
        th=math.radians(s.params["angle_deg"]); v=np.sin(2*np.pi*s.params["frequency"]*(np.cos(th)*xn+np.sin(th)*yn)+s.params["phase"])
    elif s.family=="fractal_sine":
        v=np.zeros((size,size)); amp=1.; base=s.params["base_frequency"]
        for o in range(s.params["octaves"]):
            th=rng.uniform(0,2*np.pi); fr=base*(2**o)*rng.uniform(.9,1.1); ph=rng.uniform(0,2*np.pi)
            v+=amp*np.sin(2*np.pi*fr*(np.cos(th)*xn+np.sin(th)*yn)+ph); amp*=.52
    elif s.family=="tiled_smooth_noise":
        t=s.params["tile"]; a=_smooth_tile(t,s.seed,s.params["blur"]); v=np.tile(a,(math.ceil(size/t),math.ceil(size/t)))[:size,:size]
    elif s.family=="tiled_voronoi":
        t=s.params["tile"]; a=_vor(t,s.params["points"],rng); v=np.tile(a,(math.ceil(size/t),math.ceil(size/t)))[:size,:size]
    else:
        t=s.params["tile"]; per=s.params["period"]; ty,tx=np.mgrid[0:t,0:t]
        chk=((tx//per+ty//per)%2).astype(float); rad=np.sin(2*np.pi*np.sqrt((tx-t/2)**2+(ty-t/2)**2)/max(3,2*per))
        a=.65*chk+.35*(.5+.5*rad); v=np.tile(a,(math.ceil(size/t),math.ceil(size/t)))[:size,:size]
    return Image.fromarray((_colorize(v,s.seed)*255).astype(np.uint8),"RGB")

def synthetic_items(specs,size):
    return [{"image":render_pattern(s,size),"source":"synthetic","group":s.family,"item_id":s.item_id,"params":dict(s.params,seed=s.seed)} for s in specs]

def distant_similarity(spatial,G,topk=8):
    B,P,D=spatial.shape; n=F.normalize(spatial.float(),dim=-1); sim=n@n.transpose(-2,-1)
    mask=torch.ones((P,P),dtype=torch.bool,device=sim.device)
    for p in range(P):
        r,c=divmod(p,G)
        for rr in range(max(0,r-1),min(G,r+2)):
            for cc in range(max(0,c-1),min(G,c+2)): mask[p,rr*G+cc]=False
    sim=sim.masked_fill(~mask[None],-1e9); k=min(topk,int(mask.sum(-1).min()))
    return sim.topk(k,-1).values.mean(-1)

def role_metrics(x,mu1,mu2,G,norm_thr,mu_reg_z,mu_hidden_z,scratch_z,scratch_topk):
    xb=x.permute(1,0,2).float(); sp=xb[:,1:1+G*G]; cls=xb[:,0]
    m1=mu1.to(sp.device); m2=mu2.to(sp.device)
    p1=sp@m1; p2=sp@m2; z1=robust_z_rows(p1); norm=sp.norm(dim=-1)
    legacy=norm>norm_thr; reg=legacy&(z1>mu_reg_z); hidden=(~legacy)&(z1>mu_hidden_z)
    ds=distant_similarity(sp,G,scratch_topk); dsz=robust_z_rows(ds)
    scratch=(~legacy)&(~hidden)&(z1<mu_reg_z)&(dsz>scratch_z)
    return {"spatial":sp,"cls":cls,"mu1":p1,"mu2":p2,"mu1_z":z1,"norm":norm,"legacy_reg":legacy,
            "role_reg":reg,"hidden_mu":hidden,"scratchpad":scratch,"distant_score":ds,"distant_z":dsz,
            "cls_mu1":cls@m1,"cls_mu2":cls@m2}

def collect_register_vectors(model,preprocess,items,batch_size,threshold):
    vecs=[]; counts=[]
    for st in range(0,len(items),batch_size):
        q=items[st:st+batch_size]; batch=preprocess_items(preprocess,q)
        with torch.inference_mode():
            x,_=prepare_tokens(model,batch)
            for blk in model.visual.transformer.resblocks: x=blk(x)
            sp=x.permute(1,0,2).float()[:,1:]
            norms=sp.norm(dim=-1)
            for i in range(len(q)):
                m=norms[i]>threshold
                if not m.any():
                    m=torch.zeros_like(m,dtype=torch.bool); m[int(norms[i].argmax())]=True
                vecs.append(sp[i,m].cpu()); counts.append(int(m.sum()))
        print(f"[mu basis] {min(st+len(q),len(items))}/{len(items)}")
    return torch.cat(vecs),counts

def discover_mu(X):
    dev="cuda" if torch.cuda.is_available() else "cpu"; X=X.to(dev).float()
    U,S,Vh=torch.linalg.svd(X,full_matrices=False); mu1=Vh[0]; mu2=Vh[1]
    if torch.dot(mu1,X.mean(0))<0: mu1=-mu1
    j=int(mu2.abs().argmax())
    if mu2[j]<0: mu2=-mu2
    e=S.square(); raw=e/e.sum()
    rep={"n_register_vectors":int(X.shape[0]),"width":int(X.shape[1]),
         "mu1_raw_uncentered_energy_fraction":float(raw[0]),"mu2_raw_uncentered_energy_fraction":float(raw[1]),
         "mu2_fraction_of_energy_after_removing_mu1":float(e[1]/e[1:].sum()),
         "first_8_raw_energy_fractions":[float(v) for v in raw[:8]]}
    return mu1.cpu(),mu2.cpu(),rep

def _self_test_register_norm_shape():
    sp=torch.randn(2,17,23)
    norms=sp.norm(dim=-1)
    assert tuple(norms.shape)==(2,17), tuple(norms.shape)

def conv1_color_census(model):
    W=model.visual.conv1.weight.detach().float().cpu().numpy(); pe=model.visual.positional_embedding.detach().float().cpu().numpy()[1:]
    lum=np.array([1,1,1],float); lum/=np.linalg.norm(lum); rg=np.array([1,-1,0],float); rg/=np.linalg.norm(rg); by=np.array([.5,.5,-1],float); by/=np.linalg.norm(by)
    out=[]
    for c,w in enumerate(W):
        w=w.astype(np.float64); dc=w.mean((-2,-1)); n=np.linalg.norm(dc); u=dc/(n+EPS); total=np.sum(w*w)
        dcgrid=np.broadcast_to(dc[:,None,None],w.shape); dce=np.sum(dcgrid*dcgrid)/(total+EPS)
        out.append({"channel":c,"conv1_weight_l2":float(np.sqrt(total)),"kernel_dc_energy_frac":float(dce),
                    "dc_r":dc[0],"dc_g":dc[1],"dc_b":dc[2],"dc_axis_luminance":float(u@lum),
                    "dc_axis_red_green":float(u@rg),"dc_axis_blue_yellow":float(u@by),
                    "dc_chromatic_strength":float(np.sqrt((u@rg)**2+(u@by)**2)),
                    "pos_std":float(pe[:,c].std()),"pos_rms":float(np.sqrt(np.mean(pe[:,c]**2)))})
    return out
