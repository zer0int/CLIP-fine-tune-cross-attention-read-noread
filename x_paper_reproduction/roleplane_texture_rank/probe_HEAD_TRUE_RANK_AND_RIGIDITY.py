#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from clip_probe_common import (
    seed_all, load_model, model_geometry, validate_forward, prepare_tokens, qkv_from_attn,
    preprocess_items, pearson, cosine_np, effective_rank_sv, effective_rank_energy,
    stable_rank, rel_rank, EPS
)

def salient_target(size,patch,row,col):
    im=Image.new("RGB",(size,size),(246,246,246)); d=ImageDraw.Draw(im)
    cx=int((col+.5)*patch); cy=int((row+.5)*patch); r=max(6,int(1.15*patch))
    d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=(252,202,25),outline=(8,8,8),width=max(2,patch//6))
    d.rectangle((cx-r//2,cy-r//2,cx+r//2,cy+r//2),fill=(24,185,225),outline=(10,10,10),width=max(1,patch//9))
    d.line((cx-r,cy,cx+r,cy),fill=(230,15,95),width=max(2,patch//5))
    d.line((cx,cy-r,cx,cy+r),fill=(45,25,215),width=max(2,patch//5))
    return im

def probe_positions(G,n_axis):
    vals=sorted(set(int(round(v)) for v in np.linspace(1,G-2,n_axis)))
    return [(r,c) for r in vals for c in vals]

def qk_singulars_centered(q,k):
    q=q.float(); k=k.float()
    q=q-q.mean(2,keepdim=True); k=k-k.mean(2,keepdim=True)
    B,H,L,d=q.shape
    qf=q.reshape(B*H,L,d); kf=k.reshape(B*H,L,d)
    _,Rq=torch.linalg.qr(qf,mode="reduced"); _,Rk=torch.linalg.qr(kf,mode="reduced")
    core=Rq@Rk.transpose(-2,-1)/math.sqrt(float(d))
    return torch.linalg.svdvals(core).reshape(B,H,d)

def softmax_singulars_blank(q,k):
    logits=q.float()@k.float().transpose(-2,-1)/math.sqrt(float(q.shape[-1]))
    return torch.linalg.svdvals(logits.softmax(-1)[0])

def classify_head(base,curmaps,pos,G):
    base=np.asarray(base,np.float64); base=base/(base.sum()+EPS)
    rows=[]
    grid_r=np.repeat(np.arange(G),G); grid_c=np.tile(np.arange(G),G)
    for cur,(pr,pc) in zip(curmaps,pos):
        cur=np.asarray(cur,np.float64); cur=cur/(cur.sum()+EPS)
        ar,ac=divmod(int(np.argmax(cur)),G)
        neigh=[]
        for rr in range(max(0,pr-1),min(G,pr+2)):
            for cc in range(max(0,pc-1),min(G,pc+2)): neigh.append(rr*G+cc)
        neigh=np.asarray(neigh,int)
        rows.append({
            "pr":pr,"pc":pc,
            "comr":float(np.sum(cur*grid_r)),"comc":float(np.sum(cur*grid_c)),
            "capture":int(max(abs(ar-pr),abs(ac-pc))<=1),
            "support":float(base[neigh].sum()),
            "gain":float(cur[neigh].sum()-base[neigh].sum()),
            "template":cosine_np(base,cur),
        })
    a=pd.DataFrame(rows)
    tr=.5*(pearson(a.pr,a.comr)+pearson(a.pc,a.comc)); cap=float(a.capture.mean())
    qlo=a.support.quantile(1/3); qhi=a.support.quantile(2/3)
    caplo=float(a[a.support<=qlo].capture.mean()); caphi=float(a[a.support>=qhi].capture.mean())
    gap=caphi-caplo; template=float(a.template.mean()); cv=float(base.std()/(base.mean()+EPS)); gain=float(a.gain.mean())
    if cv<.10 and abs(gain)<.02: ph="uniform_or_weak"
    elif tr>.55 and cap>.60 and gap<.35: ph="visual_tracker"
    elif cv>.25 and template>.90 and (gap>.30 or tr<.30): ph="rigid_or_gated_positional"
    else: ph="hybrid"
    rigid=cv*template*(1+max(0,gap)); tracker=max(0,tr)*cap
    return dict(phenotype=ph,baseline_spatial_cv=cv,template_cosine_mean=template,
                tracking_score=tr,capture_rate_r1=cap,support_capture_gap=gap,
                local_gain_mean=gain,rigid_score=rigid,tracker_score=tracker)

def plot_results(df,out):
    p=out/"plots"; p.mkdir(exist_ok=True)
    phenos=["rigid_or_gated_positional","visual_tracker","hybrid","uniform_or_weak"]
    fig,ax=plt.subplots(figsize=(10,5.8))
    for ph in phenos:
        q=df[df.phenotype.eq(ph)].groupby("block").qk_effrank_energy_blank.mean()
        if len(q): ax.plot(q.index,q.values,marker="o",label=ph)
    ax.set(xlabel="block",ylabel="centered-QK energy effective rank",title="Pre-softmax QK rank by head behavior")
    ax.grid(alpha=.15); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(p/"01_QK_TRUE_RANK_BY_PHENOTYPE.png",dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.8))
    ct=df.groupby(["block","phenotype"]).size().unstack(fill_value=0)
    for ph in ct.columns: ax.plot(ct.index,ct[ph],marker="o",label=ph)
    ax.set(xlabel="block",ylabel="number of heads",title="Rigid / moving-target phenotypes across all blocks")
    ax.grid(alpha=.15); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(p/"02_HEAD_PHENOTYPES_ALL_BLOCKS.png",dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,6))
    for ph in phenos:
        q=df[df.phenotype.eq(ph)]
        if len(q): ax.scatter(q.rigid_score,q.qk_effrank_energy_blank,s=38,alpha=.7,label=ph)
    ax.set(xlabel="rigid/position-gated score",ylabel="centered-QK energy effective rank",title="Head routing complexity vs positional rigidity")
    ax.grid(alpha=.15); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(p/"03_RIGIDITY_VS_TRUE_RANK.png",dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.8))
    q=df.groupby("block").agg(qk=("qk_rank_rel1e4_blank","mean"),soft=("attn_rank_rel1e4_blank","mean")).reset_index()
    ax.plot(q.block,q.qk,marker="o",label="centered pre-softmax QK")
    ax.plot(q.block,q.soft,marker="o",label="softmax attention")
    ax.set(xlabel="block",ylabel="mean numerical rank @ relative 1e-4",title="Softmax rank inflation vs pre-softmax QK rank")
    ax.grid(alpha=.15); ax.legend(); fig.tight_layout(); fig.savefig(p/"04_QK_VS_SOFTMAX_RANK.png",dpi=220); plt.close(fig)

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out_dir",default="out_paper_reproduction/conv1/roleplane_texture_rank"); ap.add_argument("--clip_model",default="openai/clip-vit-large-patch14")
    ap.add_argument("--device",default="cuda"); ap.add_argument("--seed",type=int,default=20260917)
    ap.add_argument("--probe_axis_positions",type=int,default=5); ap.add_argument("--self_test",action="store_true")
    return ap.parse_args()

def self_test():
    B,H,L,d=2,3,11,5
    a=torch.randn(B,H,L,1); u=torch.randn(B,H,1,d); q=a*u; k=torch.randn(B,H,L,d)
    s=qk_singulars_centered(q,k)
    assert int(rel_rank(s,1e-4).max())<=1
    print("self-test OK")

def main():
    args=parse_args()
    if args.self_test: self_test(); return
    seed_all(args.seed); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    model,preprocess=load_model(args.clip_model,args.device)
    size,G,patch,width,nblocks,nheads=model_geometry(model)
    parity=validate_forward(model,preprocess,size,args.device)

    pos=probe_positions(G,args.probe_axis_positions)
    ims=[Image.new("RGB",(size,size),(246,246,246))]+[salient_target(size,patch,r,c) for r,c in pos]
    batch=preprocess_items(preprocess,[{"image":im} for im in ims])
    rows=[]; maps={}
    with torch.inference_mode():
        x,_=prepare_tokens(model,batch)
        for b,blk in enumerate(model.visual.transformer.resblocks):
            z=blk.ln_1(x); q,k,_=qkv_from_attn(blk.attn,z)
            s=qk_singulars_centered(q,k).cpu()
            logits=q.float()@k.float().transpose(-2,-1)/math.sqrt(float(q.shape[-1]))
            A=logits.softmax(-1)
            cls=A[:,:,0,1:1+G*G].cpu().numpy(); maps[b]=cls.astype(np.float32)
            sb=softmax_singulars_blank(q[:1],k[:1]).cpu()
            for h in range(nheads):
                beh=classify_head(cls[0,h],cls[1:,h],pos,G)
                blank=s[0,h]; move=s[1:,h].mean(0); sa=sb[h]
                rows.append(dict(block=b,head=h,**beh,
                    qk_rank_rel1e4_blank=int(rel_rank(blank[None],1e-4)[0]),
                    qk_rank_rel1e6_blank=int(rel_rank(blank[None],1e-6)[0]),
                    qk_effrank_sv_blank=float(effective_rank_sv(blank[None])[0]),
                    qk_effrank_energy_blank=float(effective_rank_energy(blank[None])[0]),
                    qk_stable_rank_blank=float(stable_rank(blank[None])[0]),
                    qk_rank_rel1e4_moving_mean_spectrum=int(rel_rank(move[None],1e-4)[0]),
                    qk_effrank_energy_moving_mean_spectrum=float(effective_rank_energy(move[None])[0]),
                    attn_rank_rel1e4_blank=int(rel_rank(sa[None],1e-4)[0]),
                    attn_rank_rel1e6_blank=int(rel_rank(sa[None],1e-6)[0]),
                    attn_effrank_sv_blank=float(effective_rank_sv(sa[None])[0]),
                    attn_effrank_energy_blank=float(effective_rank_energy(sa[None])[0]),
                    attn_stable_rank_blank=float(stable_rank(sa[None])[0])))
            x=blk(x)
            print(f"[head probe] block {b+1}/{nblocks}")

    df=pd.DataFrame(rows); df.to_csv(out/"head_rank_rigidity_all_blocks.csv",index=False)
    block=df.groupby(["block","phenotype"]).agg(n=("head","size"),
        qk_rank_rel1e4=("qk_rank_rel1e4_blank","mean"),qk_effrank=("qk_effrank_energy_blank","mean"),
        attn_rank_rel1e4=("attn_rank_rel1e4_blank","mean"),attn_effrank=("attn_effrank_energy_blank","mean"),
        rigid_score=("rigid_score","mean"),tracker_score=("tracker_score","mean")).reset_index()
    block.to_csv(out/"head_rank_rigidity_block_summary.csv",index=False)
    cor=[]
    for b,g in df.groupby("block"):
        for m in ["qk_rank_rel1e4_blank","qk_effrank_energy_blank","attn_effrank_energy_blank"]:
            cor.append({"block":b,"rank_metric":m,"corr_rank_rigid":pearson(g[m],g.rigid_score),"corr_rank_tracker":pearson(g[m],g.tracker_score)})
    pd.DataFrame(cor).to_csv(out/"head_rank_rigidity_correlations.csv",index=False)
    np.savez_compressed(out/"head_probe_cls_maps_all_blocks.npz",positions=np.asarray(pos,np.int16),**{f"block_{b}":v for b,v in maps.items()})
    plot_results(df,out)
    (out/"head_probe_audit.json").write_text(json.dumps({"model":args.clip_model,"grid":G,"blocks":nblocks,"heads":nheads,"manual_forward_parity":parity},indent=2))
    print("DONE head rank / rigidity")

if __name__=="__main__": main()
