#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, zipfile, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clip_probe_common import (
    seed_all, load_model, model_geometry, validate_forward, prepare_tokens, preprocess_items,
    load_dtd, balanced_dtd_indices, dtd_items, make_pattern_specs, synthetic_items,
    collect_register_vectors, discover_mu, role_metrics, conv1_color_census, EPS
)

class LastInverseAccumulator:
    def __init__(self,width,sigma_frac):
        self.width=width; self.sigma_frac=sigma_frac
        roles=("register","hidden_mu","scratchpad","ordinary")
        self.freq_sum={r:np.zeros(width,np.float64) for r in roles}; self.freq_n={r:0 for r in roles}
        self.sum={r:np.zeros(width,np.float64) for r in roles}; self.sum2={r:np.zeros(width,np.float64) for r in roles}; self.n={r:0 for r in roles}
        self.rows=[]

    def update(self,x,roles,mu1,source,groups):
        # x [B,P,D] after final block
        x=x.float(); B,P,D=x.shape
        Fch=torch.fft.fft(x,dim=-1); Fs=torch.fft.fftshift(Fch,dim=-1)
        freq=torch.arange(D,device=x.device,dtype=torch.float32)-D/2
        sig=max(1.,self.sigma_frac*D); g=torch.exp(-.5*(freq/sig)**2)
        low=torch.fft.ifft(torch.fft.ifftshift(Fs*g[None,None,:],dim=-1),dim=-1).real
        high=x-low; power=Fs.abs().square(); pn=power/power.sum(-1,keepdim=True).clamp_min(EPS)
        entropy=-(pn*torch.log(pn.clamp_min(EPS))).sum(-1)/math.log(D)
        lowf=low.square().sum(-1)/x.square().sum(-1).clamp_min(EPS)
        highf=high.square().sum(-1)/x.square().sum(-1).clamp_min(EPS)
        tv=(x[:,:,1:]-x[:,:,:-1]).abs().mean(-1)

        # LAST-inspired patch votes. Paper code ranks patches channel-wise after channel FFT smoothing.
        ratio_signed=x/(high.abs()+1e-6)
        ratio_abs=x.abs()/(high.abs()+1e-6)
        ws=ratio_signed.argmax(1); wa=ratio_abs.argmax(1)
        vote_s=torch.zeros((B,P),device=x.device); vote_a=torch.zeros((B,P),device=x.device)
        vote_s.scatter_add_(1,ws,torch.ones_like(ws,dtype=torch.float32))
        vote_a.scatter_add_(1,wa,torch.ones_like(wa,dtype=torch.float32))

        mu=mu1.to(x.device); proj=x@mu; xr=x-proj[:,:,None]*mu[None,None,:]
        masks={"register":roles["role_reg"],"hidden_mu":roles["hidden_mu"],"scratchpad":roles["scratchpad"]}
        masks["ordinary"]=~(masks["register"]|masks["hidden_mu"]|masks["scratchpad"])
        for role,m in masks.items():
            if not m.any(): continue
            xx=xr[m]; pp=power[m]
            self.freq_sum[role]+=pp.sum(0).cpu().numpy(); self.freq_n[role]+=int(pp.shape[0])
            self.sum[role]+=xx.sum(0).cpu().numpy(); self.sum2[role]+=(xx*xx).sum(0).cpu().numpy(); self.n[role]+=int(xx.shape[0])
            for bi in range(B):
                mb=m[bi]
                if mb.any():
                    self.rows.append({"source":source,"group":groups[bi],"role":role,"n_patches":int(mb.sum()),
                        "channel_spectral_entropy":float(entropy[bi,mb].mean()),
                        "channel_lowpass_energy_frac":float(lowf[bi,mb].mean()),
                        "channel_highpass_energy_frac":float(highf[bi,mb].mean()),
                        "channel_total_variation":float(tv[bi,mb].mean()),
                        "last_signed_vote_fraction":float(vote_s[bi,mb].sum()/D),
                        "last_abs_vote_fraction":float(vote_a[bi,mb].sum()/D)})

    def finalize(self):
        role=pd.DataFrame(self.rows)
        frows=[]
        for r in self.freq_sum:
            n=max(1,self.freq_n[r]); m=self.freq_sum[r]/n; total=m.sum()+EPS
            for k,v in enumerate(m): frows.append({"role":r,"fftshift_channel_frequency_bin":k,"mean_power":v,"normalized_power":v/total,"n_patches":self.freq_n[r]})
        coords={}
        for r in self.sum:
            n=max(1,self.n[r]); mu=self.sum[r]/n; var=np.maximum(0,self.sum2[r]/n-mu*mu); coords[r]=(mu,var,self.n[r])
        return role,pd.DataFrame(frows),coords


ROLE_ORDER=("register","hidden_mu","scratchpad","ordinary")

def save_last_accumulator(acc,path:Path):
    np.savez_compressed(
        path,
        freq_sum=np.stack([acc.freq_sum[r] for r in ROLE_ORDER]),
        freq_n=np.asarray([acc.freq_n[r] for r in ROLE_ORDER],np.int64),
        coord_sum=np.stack([acc.sum[r] for r in ROLE_ORDER]),
        coord_sum2=np.stack([acc.sum2[r] for r in ROLE_ORDER]),
        coord_n=np.asarray([acc.n[r] for r in ROLE_ORDER],np.int64),
    )

def merge_last_accumulator_npz(acc,path:Path):
    z=np.load(path)
    for i,r in enumerate(ROLE_ORDER):
        acc.freq_sum[r]+=z["freq_sum"][i]; acc.freq_n[r]+=int(z["freq_n"][i])
        acc.sum[r]+=z["coord_sum"][i]; acc.sum2[r]+=z["coord_sum2"][i]; acc.n[r]+=int(z["coord_n"][i])

def merge_last_accumulator_obj(dst,src):
    for r in ROLE_ORDER:
        dst.freq_sum[r]+=src.freq_sum[r]; dst.freq_n[r]+=src.freq_n[r]
        dst.sum[r]+=src.sum[r]; dst.sum2[r]+=src.sum2[r]; dst.n[r]+=src.n[r]
    dst.rows.extend(src.rows)

def analyze_items(model,preprocess,items,mu1,mu2,args,G,last_acc,cache_dir=None):
    rows=[]; births=[]
    for st in range(0,len(items),args.batch_size):
        q=items[st:st+args.batch_size]; B=len(q)
        cache_base=None
        if cache_dir is not None:
            cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
            key=f"{st:06d}_{st+B:06d}"
            cache_base=cache_dir/key
            rows_p=Path(str(cache_base)+".rows.csv.gz")
            births_p=Path(str(cache_base)+".births.csv.gz")
            lastrows_p=Path(str(cache_base)+".lastrows.csv.gz")
            acc_p=Path(str(cache_base)+".acc.npz")
            if rows_p.is_file() and births_p.is_file() and lastrows_p.is_file() and acc_p.is_file():
                rows.extend(pd.read_csv(rows_p).to_dict("records"))
                btmp=pd.read_csv(births_p)
                if len(btmp): births.extend(btmp.to_dict("records"))
                ltmp=pd.read_csv(lastrows_p)
                if len(ltmp): last_acc.rows.extend(ltmp.to_dict("records"))
                merge_last_accumulator_npz(last_acc,acc_p)
                print(f"[{q[0]['source']}] resume batch {st}:{st+B}")
                continue
        batch=preprocess_items(preprocess,q)
        rows_before=len(rows); births_before=len(births)
        hist=[[] for _ in range(B)]; pre23=None
        with torch.inference_mode():
            x,_=prepare_tokens(model,batch)
            for b,blk in enumerate(model.visual.transformer.resblocks):
                m=role_metrics(x,mu1,mu2,G,args.register_norm_threshold,args.mu_reg_z,args.mu_hidden_z,args.scratch_z,args.scratch_topk)
                for i in range(B):
                    hist[i].append({"block":b,
                        "legacy":m["legacy_reg"][i].cpu().numpy(),"reg":m["role_reg"][i].cpu().numpy(),
                        "hidden":m["hidden_mu"][i].cpu().numpy(),"scratch":m["scratchpad"][i].cpu().numpy(),
                        "z1":m["mu1_z"][i].cpu().numpy(),"norm":m["norm"][i].cpu().numpy(),
                        "cls1":float(m["cls_mu1"][i]),"cls2":float(m["cls_mu2"][i]),
                        "p1":float(m["mu1"][i].mean()),"p2":float(m["mu2"][i].mean())})
                if b==23: pre23={k:v.detach() if torch.is_tensor(v) else v for k,v in m.items()}
                x=blk(x)
            post=role_metrics(x,mu1,mu2,G,args.register_norm_threshold,args.mu_reg_z,args.mu_hidden_z,args.scratch_z,args.scratch_topk)
            local_acc=LastInverseAccumulator(last_acc.width,last_acc.sigma_frac)
            local_acc.update(x.permute(1,0,2).float()[:,1:1+G*G],pre23,mu1,q[0]["source"],[r["group"] for r in q])
            merge_last_accumulator_obj(last_acc,local_acc)

        for i,item in enumerate(q):
            for z in hist[i]:
                rows.append({"source":item["source"],"group":item["group"],"item_id":item["item_id"],"params_json":json.dumps(item["params"],sort_keys=True),
                    "stage":f"pre{z['block']}","block":z["block"],"legacy_reg_count":int(z["legacy"].sum()),"role_reg_count":int(z["reg"].sum()),
                    "hidden_mu_count":int(z["hidden"].sum()),"scratchpad_count":int(z["scratch"].sum()),
                    "cls_mu1":z["cls1"],"cls_mu2":z["cls2"],"mu1_patch_mean":z["p1"],"mu2_patch_mean":z["p2"],
                    "mean_spatial_norm":float(z["norm"].mean()),"max_spatial_norm":float(z["norm"].max())})
            rows.append({"source":item["source"],"group":item["group"],"item_id":item["item_id"],"params_json":json.dumps(item["params"],sort_keys=True),
                "stage":"post23","block":24,"legacy_reg_count":int(post["legacy_reg"][i].sum()),"role_reg_count":int(post["role_reg"][i].sum()),
                "hidden_mu_count":int(post["hidden_mu"][i].sum()),"scratchpad_count":int(post["scratchpad"][i].sum()),
                "cls_mu1":float(post["cls_mu1"][i]),"cls_mu2":float(post["cls_mu2"][i]),"mu1_patch_mean":float(post["mu1"][i].mean()),
                "mu2_patch_mean":float(post["mu2"][i].mean()),"mean_spatial_norm":float(post["norm"][i].mean()),"max_spatial_norm":float(post["norm"][i].max())})

            final=hist[i][23]["reg"]; ids=np.flatnonzero(final)
            for p in ids:
                zseq=np.asarray([h["z1"][p] for h in hist[i]],float); nseq=np.asarray([h["norm"][p] for h in hist[i]],float)
                first_mu=next((b for b,v in enumerate(zseq) if v>args.mu_hidden_z),None)
                first_reg=next((b for b,(z,n) in enumerate(zip(zseq,nseq)) if z>args.mu_reg_z and n>args.register_norm_threshold),None)
                persist=float(np.mean(zseq[first_mu:]>args.mu_reg_z)) if first_mu is not None else np.nan
                births.append({"source":item["source"],"group":item["group"],"item_id":item["item_id"],"patch_index0":int(p),
                    "row0":int(p//G),"col0":int(p%G),"first_mu1_outlier_block":first_mu,"first_role_register_block":first_reg,
                    "mu1_persistence_after_birth":persist,"pre23_norm":float(nseq[23]),"pre23_mu1_z":float(zseq[23])})
        if cache_base is not None:
            pd.DataFrame(rows[rows_before:]).to_csv(rows_p,index=False,compression="gzip")
            pd.DataFrame(births[births_before:]).to_csv(births_p,index=False,compression="gzip")
            pd.DataFrame(local_acc.rows).to_csv(lastrows_p,index=False,compression="gzip")
            save_last_accumulator(local_acc,acc_p)
        print(f"[{items[0]['source']}] {min(st+B,len(items))}/{len(items)}")
    return pd.DataFrame(rows),pd.DataFrame(births)

def class_summary(df,source):
    q=df[(df.source.eq(source))&(df.stage.eq("pre23"))]
    return q.groupby("group").agg(n=("item_id","nunique"),role_reg_mean=("role_reg_count","mean"),role_reg_std=("role_reg_count","std"),
        hidden_mu_mean=("hidden_mu_count","mean"),hidden_mu_std=("hidden_mu_count","std"),scratchpad_mean=("scratchpad_count","mean"),
        legacy_reg_mean=("legacy_reg_count","mean"),max_norm_mean=("max_spatial_norm","mean")).reset_index().sort_values("role_reg_mean",ascending=False)

def sine_summary(df):
    q=df[(df.source.eq("synthetic"))&(df.stage.eq("pre23"))&(df.group.eq("sine"))].copy()
    ps=q.params_json.map(json.loads); q["angle_deg"]=[p["angle_deg"] for p in ps]; q["frequency"]=[p["frequency"] for p in ps]
    return q.groupby(["angle_deg","frequency"]).agg(n=("item_id","size"),role_reg_mean=("role_reg_count","mean"),
        hidden_mu_mean=("hidden_mu_count","mean"),scratchpad_mean=("scratchpad_count","mean"),max_norm_mean=("max_spatial_norm","mean")).reset_index().sort_values(["role_reg_mean","hidden_mu_mean"],ascending=False)

def stage_summary(df):
    return df.groupby(["source","stage"]).agg(n_images=("item_id","nunique"),legacy_reg_mean=("legacy_reg_count","mean"),
        role_reg_mean=("role_reg_count","mean"),hidden_mu_mean=("hidden_mu_count","mean"),scratchpad_mean=("scratchpad_count","mean"),
        cls_mu1_mean=("cls_mu1","mean"),cls_mu2_mean=("cls_mu2","mean"),mu1_patch_mean=("mu1_patch_mean","mean")).reset_index()

def birth_summary(df):
    if df.empty: return pd.DataFrame()
    return df.groupby(["source","group"]).agg(n_final_registers=("patch_index0","size"),first_mu1_mean=("first_mu1_outlier_block","mean"),
        first_mu1_median=("first_mu1_outlier_block","median"),first_role_reg_mean=("first_role_register_block","mean"),
        persistence_mean=("mu1_persistence_after_birth","mean")).reset_index()

def position_frequency(df):
    if df.empty: return pd.DataFrame()
    q=df.groupby(["source","group","patch_index0","row0","col0"]).size().reset_index(name="final_register_count")
    n=df.groupby(["source","group"]).item_id.nunique().reset_index(name="n_images"); q=q.merge(n,on=["source","group"])
    q["register_per_image_frequency"]=q.final_register_count/q.n_images
    return q

def coordinate_candidates(coords,color):
    om,ov,on=coords["ordinary"]; rows=[]
    for role in ("register","hidden_mu","scratchpad"):
        m,v,n=coords[role]; eff=(m-om)/np.sqrt(.5*(v+ov)+1e-8)
        for c,e in enumerate(eff): rows.append({"role":role,"channel":c,"residualized_mu1_effect":float(e),"abs_effect":abs(float(e)),"role_n_patches":n,"ordinary_n_patches":on})
    d=pd.DataFrame(rows).merge(color,on="channel",how="left").sort_values(["role","abs_effect"],ascending=[True,False]).reset_index(drop=True)
    d["rank_within_role"]=d.groupby("role").cumcount()+1
    return d

def plots(out,dtdsum,synsum,sine,stages,role,cands):
    p=out/"plots"; p.mkdir(exist_ok=True)
    if not dtdsum.empty:
        q=dtdsum.sort_values("role_reg_mean"); y=np.arange(len(q))
        fig,ax=plt.subplots(figsize=(10,max(8,.26*len(q)))); ax.barh(y,q.role_reg_mean,label="manifest REG"); ax.barh(y,q.hidden_mu_mean,left=q.role_reg_mean,label="hidden mu")
        ax.set_yticks(y); ax.set_yticklabels(q.group,fontsize=7); ax.set_xlabel("mean count @ pre23"); ax.set_title("DTD register + hidden-mu populations by class"); ax.legend(); fig.tight_layout(); fig.savefig(p/"05_DTD_REG_HIDDENMU_BY_CLASS.png",dpi=220); plt.close(fig)
    if not dtdsum.empty and not synsum.empty:
        fig,ax=plt.subplots(figsize=(9,6)); ax.scatter(dtdsum.role_reg_mean,dtdsum.hidden_mu_mean,s=35,alpha=.6,label="DTD classes")
        ax.scatter(synsum.role_reg_mean,synsum.hidden_mu_mean,s=90,marker="s",label="synthetic families")
        for r in synsum.itertuples(): ax.annotate(r.group,(r.role_reg_mean,r.hidden_mu_mean),xytext=(3,3),textcoords="offset points",fontsize=8)
        ax.set(xlabel="manifest REG mean @ pre23",ylabel="hidden-mu mean @ pre23",title="Natural textures vs mathematical repetition"); ax.grid(alpha=.15); ax.legend(); fig.tight_layout(); fig.savefig(p/"06_NATURAL_VS_SYNTH.png",dpi=220); plt.close(fig)
    if not sine.empty:
        for metric,name in [("role_reg_mean","07_SINE_REGISTERMAXXING.png"),("hidden_mu_mean","08_SINE_HIDDENMU.png")]:
            a=sine.pivot(index="angle_deg",columns="frequency",values=metric)
            fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(a.to_numpy(),aspect="auto"); ax.set_xticks(range(len(a.columns))); ax.set_xticklabels(a.columns); ax.set_yticks(range(len(a.index))); ax.set_yticklabels(a.index)
            ax.set(xlabel="sine frequency",ylabel="angle (deg)",title=metric.replace("_"," ")); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(p/name,dpi=220); plt.close(fig)
    if not stages.empty:
        fig,ax=plt.subplots(figsize=(10,5.8))
        for src in stages.source.unique():
            q=stages[stages.source.eq(src)].copy(); q["ord"]=q.stage.map(lambda s:24 if s=="post23" else int(s.replace("pre",""))); q=q.sort_values("ord")
            ax.plot(q.ord,q.role_reg_mean,marker="o",label=f"{src}:REG"); ax.plot(q.ord,q.hidden_mu_mean,marker="x",ls="--",label=f"{src}:hidden mu")
        ax.set(xlabel="block input (24=post23)",ylabel="mean token count",title="Role populations across depth"); ax.grid(alpha=.15); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(p/"09_ROLE_POPULATIONS_DEPTH.png",dpi=220); plt.close(fig)
    if not role.empty:
        q=role.groupby("role").agg(low=("channel_lowpass_energy_frac","mean"),high=("channel_highpass_energy_frac","mean")).reset_index(); x=np.arange(len(q))
        fig,ax=plt.subplots(figsize=(8,5.5)); ax.bar(x-.18,q.low,.36,label="low-pass"); ax.bar(x+.18,q.high,.36,label="high-pass"); ax.set_xticks(x); ax.set_xticklabels(q.role); ax.set_title("Inverse LAST-style channel-frequency decomposition"); ax.legend(); fig.tight_layout(); fig.savefig(p/"10_LAST_INVERSE_ROLE_FREQ.png",dpi=220); plt.close(fig)
    if not cands.empty:
        q=cands[cands.rank_within_role<=40]; fig,ax=plt.subplots(figsize=(8.5,6))
        for ro in ("register","hidden_mu","scratchpad"):
            g=q[q.role.eq(ro)]; ax.scatter(g.dc_chromatic_strength,g.residualized_mu1_effect,s=45,alpha=.7,label=ro)
        ax.axhline(0,lw=1); ax.set(xlabel="Conv1 DC chromatic/opponent strength",ylabel="late role enrichment after removing mu1",title="Candidate cache/address coordinates vs retinal color axis"); ax.grid(alpha=.15); ax.legend(); fig.tight_layout(); fig.savefig(p/"11_ADDRESS_CANDIDATES_RGB.png",dpi=220); plt.close(fig)

def report(out,mu_rep,dtdsum,synsum,sine,birth,cands):
    lines=["# Role-plane / texture / inverse-LAST atlas","","## Mu basis",
        f"- vectors: {mu_rep.get('n_register_vectors')}","- mu1 uncentered energy: %.8f"%mu_rep.get("mu1_raw_uncentered_energy_fraction",float("nan")),
        "- mu2 raw energy: %.8f"%mu_rep.get("mu2_raw_uncentered_energy_fraction",float("nan")),
        "- mu2 / residual-after-mu1 energy: %.8f"%mu_rep.get("mu2_fraction_of_energy_after_removing_mu1",float("nan")),"",
        "## DTD highest-register classes"]
    for r in dtdsum.head(12).itertuples(): lines.append(f"- {r.group}: REG={r.role_reg_mean:.3f}, hidden={r.hidden_mu_mean:.3f}, scratch={r.scratchpad_mean:.3f}")
    lines+=["","## Synthetic family means"]
    for r in synsum.sort_values("role_reg_mean",ascending=False).itertuples(): lines.append(f"- {r.group}: REG={r.role_reg_mean:.3f}, hidden={r.hidden_mu_mean:.3f}, scratch={r.scratchpad_mean:.3f}")
    lines+=["","## Top sine REGISTERMAXXING"]
    for r in sine.head(12).itertuples(): lines.append(f"- angle={r.angle_deg:g}, freq={r.frequency:g}: REG={r.role_reg_mean:.3f}, hidden={r.hidden_mu_mean:.3f}")
    if not birth.empty:
        lines+=["","## Mu/register birth"]
        for r in birth.groupby("source").agg(first_mu=("first_mu1_mean","mean"),first_reg=("first_role_reg_mean","mean"),persist=("persistence_mean","mean")).reset_index().itertuples():
            lines.append(f"- {r.source}: first mu1={r.first_mu:.3f}, first manifest REG={r.first_reg:.3f}, persistence={r.persist:.4f}")
    lines+=["","## Top residualized cache/address candidates"]
    for ro in ("register","hidden_mu","scratchpad"):
        lines.append(f"### {ro}")
        for r in cands[(cands.role.eq(ro))&(cands.rank_within_role<=8)].itertuples():
            lines.append(f"- ch{r.channel}: effect={r.residualized_mu1_effect:+.3f}, DCfrac={r.kernel_dc_energy_frac:.3f}, chrom={r.dc_chromatic_strength:.3f}, posstd={r.pos_std:.4f}")
    lines+=["","## Guardrails","- mu1/mu2 are discovered by uncentered SVD unless --mu_basis_in is supplied.","- hidden-mu and scratchpad are operational threshold definitions; raw per-image counts are saved.","- DTD is used as a texture class benchmark, not as pixel foreground/background ground truth.","- LAST-inspired analysis uses channel-axis FFT smoothing diagnostically; it does not reproduce LAST-ViT training."]
    (out/"REPORT_TEXTURE_ROLEPLANE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out_dir",default="out_paper_reproduction/conv1/roleplane_texture_rank"); ap.add_argument("--clip_model",default="openai/clip-vit-large-patch14"); ap.add_argument("--device",default="cuda"); ap.add_argument("--seed",type=int,default=20260917); ap.add_argument("--batch_size",type=int,default=12)
    ap.add_argument("--dtd_basis_per_class",type=int,default=4); ap.add_argument("--dtd_eval_per_class",type=int,default=20); ap.add_argument("--mu_basis_in",default="")
    ap.add_argument("--register_norm_threshold",type=float,default=60.); ap.add_argument("--mu_reg_z",type=float,default=3.); ap.add_argument("--mu_hidden_z",type=float,default=5.); ap.add_argument("--scratch_z",type=float,default=4.); ap.add_argument("--scratch_topk",type=int,default=8); ap.add_argument("--last_sigma_frac",type=float,default=.12)
    ap.add_argument("--self_test",action="store_true"); return ap.parse_args()

def self_test():
    from clip_probe_common import make_pattern_specs,render_pattern,robust_z_rows,_self_test_register_norm_shape
    s=make_pattern_specs(1); assert render_pattern(s[0],224).size==(224,224); assert float(robust_z_rows(torch.tensor([[0.,0.,0.,10.]]))[0,-1])>1; _self_test_register_norm_shape(); print("self-test OK")


def cache_signature(args,mu1,mu2,item_ids):
    h=hashlib.sha256()
    cfg={
        "norm":args.register_norm_threshold,"mu_reg_z":args.mu_reg_z,"mu_hidden_z":args.mu_hidden_z,
        "scratch_z":args.scratch_z,"scratch_topk":args.scratch_topk,"last_sigma_frac":args.last_sigma_frac,
        "batch_size":args.batch_size,
    }
    h.update(json.dumps(cfg,sort_keys=True).encode())
    h.update(mu1.numpy().tobytes()); h.update(mu2.numpy().tobytes())
    h.update("\\n".join(item_ids).encode())
    return h.hexdigest()[:16]

def main():
    args=parse_args()
    if args.self_test: self_test(); return
    seed_all(args.seed); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    model,preprocess=load_model(args.clip_model,args.device); size,G,patch,width,nblocks,nheads=model_geometry(model); parity=validate_forward(model,preprocess,size,args.device)
    ds,names=load_dtd(); basis_idx,eval_idx,audit=balanced_dtd_indices(ds,names,args.dtd_basis_per_class,args.dtd_eval_per_class,args.seed); pd.DataFrame(audit).to_csv(out/"dtd_sampling_audit.csv",index=False)

    basis_signature=hashlib.sha256(json.dumps({
        "basis_idx":[int(x) for x in basis_idx],
        "register_norm_threshold":float(args.register_norm_threshold),
        "clip_model":args.clip_model,
        "seed":int(args.seed),
    },sort_keys=True).encode()).hexdigest()[:16]
    basis_npz=out/"mu_basis.npz"; basis_json=out/"mu_basis_report.json"

    if args.mu_basis_in:
        z=np.load(args.mu_basis_in); mu1=torch.from_numpy(z["mu1"]).float(); mu2=torch.from_numpy(z["mu2"]).float()
        mu_rep={"source":"loaded","path":args.mu_basis_in,"n_register_vectors":-1,"mu1_raw_uncentered_energy_fraction":np.nan,"mu2_raw_uncentered_energy_fraction":np.nan,"mu2_fraction_of_energy_after_removing_mu1":np.nan}
    elif basis_npz.is_file() and basis_json.is_file():
        prev=json.loads(basis_json.read_text(encoding="utf-8"))
        if prev.get("basis_signature")==basis_signature:
            z=np.load(basis_npz); mu1=torch.from_numpy(z["mu1"]).float(); mu2=torch.from_numpy(z["mu2"]).float()
            mu_rep=prev
            print(f"[mu basis] resume: loading {basis_npz}")
        else:
            X,c=collect_register_vectors(model,preprocess,dtd_items(ds,names,basis_idx),args.batch_size,args.register_norm_threshold)
            mu1,mu2,mu_rep=discover_mu(X)
            mu_rep.update({"source":"DTD calibration","basis_images":len(basis_idx),"mean_registers_per_image":float(np.mean(c)),"basis_signature":basis_signature})
    else:
        X,c=collect_register_vectors(model,preprocess,dtd_items(ds,names,basis_idx),args.batch_size,args.register_norm_threshold)
        mu1,mu2,mu_rep=discover_mu(X)
        mu_rep.update({"source":"DTD calibration","basis_images":len(basis_idx),"mean_registers_per_image":float(np.mean(c)),"basis_signature":basis_signature})

    np.savez_compressed(basis_npz,mu1=mu1.numpy(),mu2=mu2.numpy())
    basis_json.write_text(json.dumps(mu_rep,indent=2),encoding="utf-8")

    color=pd.DataFrame(conv1_color_census(model)); color.to_csv(out/"conv1_color_coordinate_census.csv",index=False)
    acc=LastInverseAccumulator(width,args.last_sigma_frac)
    dtd_eval_items=dtd_items(ds,names,eval_idx)
    dtd_sig=cache_signature(args,mu1,mu2,[x["item_id"] for x in dtd_eval_items])
    dtd_df,dtd_birth=analyze_items(model,preprocess,dtd_eval_items,mu1,mu2,args,G,acc,out/"batch_cache"/f"dtd_{dtd_sig}"); dtd_df.to_csv(out/"dtd_per_image.csv.gz",index=False,compression="gzip")
    dtdsum=class_summary(dtd_df,"dtd"); dtdsum.to_csv(out/"dtd_per_class_summary.csv",index=False)

    specs=make_pattern_specs(args.seed); syn_items=synthetic_items(specs,size)
    syn_sig=cache_signature(args,mu1,mu2,[x["item_id"] for x in syn_items])
    syn_df,syn_birth=analyze_items(model,preprocess,syn_items,mu1,mu2,args,G,acc,out/"batch_cache"/f"synthetic_{syn_sig}"); syn_df.to_csv(out/"synthetic_per_image.csv.gz",index=False,compression="gzip")
    synsum=class_summary(syn_df,"synthetic"); synsum.to_csv(out/"synthetic_family_summary.csv",index=False); sine=sine_summary(syn_df); sine.to_csv(out/"sine_registermaxxing.csv",index=False)
    (out/"top_registermaxxing.txt").write_text("\n".join([f"angle={r.angle_deg:g} freq={r.frequency:g} REG={r.role_reg_mean:.4f} hidden={r.hidden_mu_mean:.4f}" for r in sine.head(30).itertuples()])+"\n")

    all_df=pd.concat([dtd_df,syn_df],ignore_index=True); stages=stage_summary(all_df); stages.to_csv(out/"role_stage_summary.csv",index=False)
    births=birth_summary(pd.concat([dtd_birth,syn_birth],ignore_index=True)); births.to_csv(out/"role_birth_summary.csv",index=False)
    pos=position_frequency(pd.concat([dtd_birth,syn_birth],ignore_index=True)); pos.to_csv(out/"role_position_frequency.csv.gz",index=False,compression="gzip")
    role,freq,coords=acc.finalize(); role.to_csv(out/"last_inverse_patch_role_summary.csv",index=False); freq.to_csv(out/"last_inverse_frequency_profiles.csv.gz",index=False,compression="gzip")
    cands=coordinate_candidates(coords,color); cands.to_csv(out/"address_bus_channel_candidates.csv",index=False)

    plots(out,dtdsum,synsum,sine,stages,role,cands); report(out,mu_rep,dtdsum,synsum,sine,births,cands)
    (out/"texture_probe_audit.json").write_text(json.dumps({"model":args.clip_model,"grid":G,"width":width,"blocks":nblocks,"heads":nheads,"dtd_classes":len(names),"manual_forward_parity":parity},indent=2))
    print("DONE role-plane / DTD / synthetic / inverse-LAST")

if __name__=="__main__": main()
