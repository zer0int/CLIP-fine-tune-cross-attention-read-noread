#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISUAL / TEXTUAL PROVENANCE + ROLE-PLANE ATLAS
==============================================

Dataset expected under:
    image_sets/visualtextual

Filename grammar:
    vis_<literal>.png
    vis_<literal>_bw.png
    txt_<literal>.png
    txt_<literal>_bw.png
    mix_<literal>.png
    mix_<literal>_bw.png

The literal filename label is also used verbatim as the raw text prompt
(underscores are converted to spaces). No "a photo of a ..." wrapper is used.

Models:
    pretrained
    gmp
    full_xattn

For full_xattn, both the visual BACKBONE embedding and the candidate-independent
CONTENT-corrected embedding are analyzed. Internal token analyses always refer
to the visual backbone / RN-augmented transformer state.

Major measurements:
  * final 768-D embedding PCA, category colors, paired arrows;
  * controlled added-text direction   mix - vis;
  * controlled added-visual direction mix - txt;
  * direct textual-vs-visual provenance direction txt - vis;
  * leave-one-concept-out text-provenance transfer;
  * linear removal of a text-add direction learned from OTHER concepts, testing
    whether txt and vis versions of the same concept become closer while the
    inter-concept margin survives;
  * RGB -> BW and BW -> RGB transfer;
  * prompt alignment using the raw literal label;
  * per-model mu1/mu2 role plane discovered from VISUAL-ONLY images using
    per-image mean register vectors + uncentered SVD;
  * manifest REG / hidden-mu / scratchpad / ordinary roles at every block;
  * role placement relative to INPUT-DERIVED text/object/background masks;
  * role stability/Jaccard across vis/txt/mix and RGB/BW;
  * role mean representations with mu1/mu2 removed, then the SAME provenance
    direction analysis inside REG / hidden-mu / scratchpad / ordinary states;
  * patchwise spread of the final text-add direction into text/object/background
    locations, including background role populations;
  * inverse-LAST-style channel-frequency summaries by internal token role;
  * all-1024 Conv1 text-tagging screen using controlled mix-vis differences at
    the actual text patches;
  * optional causal ZERO screen for top Conv1 text-tagging channels plus
    matched-weight controls.

The prior scratchpad definition is deliberately kept unchanged.
"""

from __future__ import annotations

import argparse, gc, hashlib, json, math, os, shutil, zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualtextual_probe_common import (
    EPS, amp_context, build_input_region_masks, collect_register_means,
    complete_concepts, conv1_morphology, direction_stats, discover_mu_from_image_means,
    encode_prompts, import_runtime, load_variant, manifest_audit, model_geometry,
    normalize_np, official_embeddings, parse_dataset, parse_models, pca_2d,
    pearson, cosine_np, prepare_tokens, preprocess_rows, project_tokens_to_joint, remove_role_plane,
    role_metrics, seed_all, spatial_tokens, stable_seed, validate_manual_backbone,
    visual_final_embedding, self_test_common
)

COLORS = {
    "vis_rgb": "#1f77b4",
    "txt_rgb": "#d62728",
    "mix_rgb": "#9467bd",
    "vis_bw": "#17becf",
    "txt_bw": "#ff7f0e",
    "mix_bw": "#e377c2",
}
MARKERS = {"vis":"o","txt":"s","mix":"^"}
ROLES = ("register","hidden_mu","scratchpad","ordinary")


# =============================================================================
# Small helpers
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def embedding_lookup(meta: pd.DataFrame, E: np.ndarray):
    return {str(r.stim_id): E[i] for i,r in enumerate(meta.itertuples(index=False))}

def cell_lookup(meta: pd.DataFrame):
    return {(str(r.condition),str(r.concept),bool(r.bw)): str(r.stim_id) for r in meta.itertuples(index=False)}

def pair_cell_ids(meta: pd.DataFrame, concept: str, bw: bool):
    lk=cell_lookup(meta)
    need={}
    for c in ("vis","txt","mix"):
        k=(c,concept,bool(bw))
        if k not in lk: return None
        need[c]=lk[k]
    return need

def safe_norm(v):
    return np.asarray(v,np.float64)/max(float(np.linalg.norm(v)),EPS)

def remove_direction_np(x: np.ndarray, w: np.ndarray):
    w=safe_norm(w)
    x=np.asarray(x,np.float64)
    return x-np.outer(x@w,w)

def cosine_rows(a,b):
    a=normalize_np(a); b=normalize_np(b)
    return np.sum(a*b,axis=-1)

def jaccard(a: np.ndarray,b: np.ndarray):
    a=np.asarray(a,bool); b=np.asarray(b,bool)
    u=np.logical_or(a,b).sum()
    return float(np.logical_and(a,b).sum()/u) if u else 1.0

def mask_for_role(cache, idx:int, role:str):
    key={"register":"role_reg_mask","hidden_mu":"hidden_mu_mask","scratchpad":"scratchpad_mask","ordinary":"ordinary_mask"}[role]
    return cache[key][idx].astype(bool)

def make_category(meta):
    return np.asarray(meta["category"].astype(str))

def explained_pc1(D):
    D=np.asarray(D,np.float64)
    if len(D)==0:return np.nan
    s=np.linalg.svd(D,compute_uv=False)
    e=s*s
    return float(e[0]/max(e.sum(),EPS))

# =============================================================================
# Model-level baseline cache
# =============================================================================

def model_cache_signature(variant, manifest, args, mu1, mu2):
    payload={
        "model":variant.name,
        "source":variant.source_info,
        "files":[Path(x).name for x in manifest["path"]],
        "sizes":[int(Path(x).stat().st_size) for x in manifest["path"]],
        "norm_thr":args.register_norm_threshold,
        "mu_reg_z":args.mu_reg_z,
        "mu_hidden_z":args.mu_hidden_z,
        "scratch_z":args.scratch_z,
        "scratch_topk":args.scratch_topk,
    }
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode())
    h.update(mu1.numpy().tobytes()); h.update(mu2.numpy().tobytes())
    return h.hexdigest()[:20]

def run_model_baseline(variant, manifest, mu1, mu2, args, model_dir:Path):
    size,G,patch,width,nblocks,nheads=model_geometry(variant)
    P=G*G
    cache_path=model_dir/"baseline_cache.npz"
    counts_path=model_dir/"role_stage_per_image.csv.gz"
    telemetry_path=model_dir/"xattn_telemetry.csv"
    signature=model_cache_signature(variant,manifest,args,mu1,mu2)
    sig_path=model_dir/"baseline_cache_signature.txt"
    batch_cache_dir=ensure_dir(model_dir/"baseline_batch_cache"/signature)

    if cache_path.is_file() and counts_path.is_file() and sig_path.is_file() and sig_path.read_text().strip()==signature:
        print(f"[{variant.name}] resume baseline cache")
        z=np.load(cache_path,allow_pickle=False)
        cache={k:z[k] for k in z.files}
        counts=pd.read_csv(counts_path)
        telemetry=pd.read_csv(telemetry_path) if telemetry_path.is_file() else pd.DataFrame()
        return cache,counts,telemetry

    N=len(manifest)
    ids=list(manifest["stim_id"].astype(str))
    backbone=[]
    content=[]
    conv_all=[]
    post_all=[]
    reg_all=[]; hidden_all=[]; scratch_all=[]; ordinary_all=[]
    role_rows=[]; telemetry_rows=[]

    # Official/manual parity before expensive work.
    first=preprocess_rows(variant,manifest.iloc[:1])
    parity=validate_manual_backbone(variant,first,args.amp)
    (model_dir/"manual_backbone_parity.json").write_text(json.dumps(parity,indent=2))

    for st in range(0,N,args.batch_size):
        q=manifest.iloc[st:st+args.batch_size]
        B=len(q)
        stem=f"{st:04d}_{st+B:04d}"
        bnpz=batch_cache_dir/f"{stem}.npz"
        broles=batch_cache_dir/f"{stem}.roles.csv.gz"
        btel=batch_cache_dir/f"{stem}.telemetry.csv"

        if bnpz.is_file() and broles.is_file():
            z=np.load(bnpz,allow_pickle=False)
            cached_ids=list(z["stim_ids"].astype(str))
            expected_ids=list(q["stim_id"].astype(str))
            if cached_ids!=expected_ids:
                raise RuntimeError(f"{variant.name}: batch cache ID mismatch at {stem}")
            backbone.append(z["backbone"])
            conv_all.append(z["conv"])
            post_all.append(z["post_tokens"])
            reg_all.append(z["role_reg_mask"])
            hidden_all.append(z["hidden_mu_mask"])
            scratch_all.append(z["scratchpad_mask"])
            ordinary_all.append(z["ordinary_mask"])
            if "content" in z.files:
                content.append(z["content"])
            role_rows.extend(pd.read_csv(broles).to_dict("records"))
            if btel.is_file() and btel.stat().st_size:
                try:
                    telemetry_rows.extend(pd.read_csv(btel).to_dict("records"))
                except pd.errors.EmptyDataError:
                    pass
            print(f"[{variant.name} baseline] resume {st}:{st+B}")
            continue

        role_start=len(role_rows)
        telemetry_start=len(telemetry_rows)
        batch=preprocess_rows(variant,q)
        official=None
        if variant.is_full_xattn:
            official=official_embeddings(variant,batch,args.amp)
        with torch.inference_mode(), amp_context(args.device,args.amp):
            x,conv=prepare_tokens(variant,batch)
            conv_all.append(conv.detach().float().cpu().numpy().astype(np.float16))
            pre23_roles=None
            for bi,blk in enumerate(variant.visual.transformer.resblocks):
                x=variant.maybe_insert_rn(bi,x)
                m=role_metrics(
                    x,mu1,mu2,G,args.register_norm_threshold,args.mu_reg_z,
                    args.mu_hidden_z,args.scratch_z,args.scratch_topk
                )
                for i,row in enumerate(q.itertuples(index=False)):
                    role_rows.append({
                        "stim_id":row.stim_id,"concept":row.concept,"condition":row.condition,
                        "bw":bool(row.bw),"category":row.category,"stage":f"pre{bi}","block":bi,
                        "legacy_reg_count":int(m["legacy_reg"][i].sum()),
                        "role_reg_count":int(m["role_reg"][i].sum()),
                        "hidden_mu_count":int(m["hidden_mu"][i].sum()),
                        "scratchpad_count":int(m["scratchpad"][i].sum()),
                        "ordinary_count":int(m["ordinary"][i].sum()),
                        "cls_mu1":float(m["cls_mu1"][i]),"cls_mu2":float(m["cls_mu2"][i]),
                        "mean_norm":float(m["norm"][i].mean()),"max_norm":float(m["norm"][i].max()),
                    })
                if bi==23:
                    pre23_roles={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in m.items()}
                x=blk(x)

            postm=role_metrics(
                x,mu1,mu2,G,args.register_norm_threshold,args.mu_reg_z,
                args.mu_hidden_z,args.scratch_z,args.scratch_topk
            )
            for i,row in enumerate(q.itertuples(index=False)):
                role_rows.append({
                    "stim_id":row.stim_id,"concept":row.concept,"condition":row.condition,
                    "bw":bool(row.bw),"category":row.category,"stage":"post23","block":24,
                    "legacy_reg_count":int(postm["legacy_reg"][i].sum()),
                    "role_reg_count":int(postm["role_reg"][i].sum()),
                    "hidden_mu_count":int(postm["hidden_mu"][i].sum()),
                    "scratchpad_count":int(postm["scratchpad"][i].sum()),
                    "ordinary_count":int(postm["ordinary"][i].sum()),
                    "cls_mu1":float(postm["cls_mu1"][i]),"cls_mu2":float(postm["cls_mu2"][i]),
                    "mean_norm":float(postm["norm"][i].mean()),"max_norm":float(postm["norm"][i].max()),
                })

            back=visual_final_embedding(variant,x).detach().float().cpu()
            backbone.append(back.numpy().astype(np.float32))
            sp=spatial_tokens(x.detach().float().cpu(),P)
            post_all.append(sp.numpy().astype(np.float16))
            reg_all.append(pre23_roles["role_reg"].numpy().astype(np.uint8))
            hidden_all.append(pre23_roles["hidden_mu"].numpy().astype(np.uint8))
            scratch_all.append(pre23_roles["scratchpad"].numpy().astype(np.uint8))
            ordinary_all.append(pre23_roles["ordinary"].numpy().astype(np.uint8))

        if variant.is_full_xattn and official is not None:
            content.append(official["content"].numpy().astype(np.float32))
            tel=official.get("telemetry",{})
            for i,row in enumerate(q.itertuples(index=False)):
                rec={"stim_id":row.stim_id,"concept":row.concept,"condition":row.condition,"bw":bool(row.bw),"category":row.category}
                for k,v in tel.items(): rec[k]=float(v[i])
                telemetry_rows.append(rec)

        batch_payload={
            "stim_ids":np.asarray(list(q["stim_id"].astype(str))),
            "backbone":backbone[-1],
            "conv":conv_all[-1],
            "post_tokens":post_all[-1],
            "role_reg_mask":reg_all[-1],
            "hidden_mu_mask":hidden_all[-1],
            "scratchpad_mask":scratch_all[-1],
            "ordinary_mask":ordinary_all[-1],
        }
        if variant.is_full_xattn and content:
            batch_payload["content"]=content[-1]
        np.savez_compressed(bnpz,**batch_payload)
        pd.DataFrame(role_rows[role_start:]).to_csv(broles,index=False,compression="gzip")
        if telemetry_start < len(telemetry_rows):
            pd.DataFrame(telemetry_rows[telemetry_start:]).to_csv(btel,index=False)
        print(f"[{variant.name} baseline] {min(st+B,N)}/{N}")

    cache={
        "stim_ids":np.asarray(ids,dtype=f"<U{max(8,max(map(len,ids)))}"),
        "backbone":np.concatenate(backbone,axis=0),
        "conv":np.concatenate(conv_all,axis=0),
        "post_tokens":np.concatenate(post_all,axis=0),
        "role_reg_mask":np.concatenate(reg_all,axis=0),
        "hidden_mu_mask":np.concatenate(hidden_all,axis=0),
        "scratchpad_mask":np.concatenate(scratch_all,axis=0),
        "ordinary_mask":np.concatenate(ordinary_all,axis=0),
    }
    if content:
        cache["content"]=np.concatenate(content,axis=0)
    np.savez(cache_path,**cache)
    counts=pd.DataFrame(role_rows)
    counts.to_csv(counts_path,index=False,compression="gzip")
    telemetry=pd.DataFrame(telemetry_rows)
    if len(telemetry): telemetry.to_csv(telemetry_path,index=False)
    sig_path.write_text(signature)
    # Consolidated cache is now sufficient for all later reruns.
    shutil.rmtree(batch_cache_dir,ignore_errors=True)
    return cache,counts,telemetry

# =============================================================================
# Final embedding geometry
# =============================================================================

def build_direction_rows(meta, E):
    lk=embedding_lookup(meta,E)
    rows=[]; vectors={}
    for bw in (False,True):
        for concept in complete_concepts(meta,bw):
            ids=pair_cell_ids(meta,concept,bw)
            v=lk[ids["vis"]]; t=lk[ids["txt"]]; m=lk[ids["mix"]]
            for name,d in [
                ("add_text",m-v),
                ("add_visual",m-t),
                ("txt_minus_vis",t-v),
            ]:
                key=f"{name}_{'bw' if bw else 'rgb'}_{concept}"
                vectors[key]=d.astype(np.float32)
                rows.append({
                    "concept":concept,"bw":bw,"appearance":"bw" if bw else "rgb",
                    "direction":name,"norm":float(np.linalg.norm(d)),
                })
    return pd.DataFrame(rows),vectors

def loo_provenance_test(meta,E,prompt_labels=None,prompt_E=None):
    """Learn text-add direction from mix-vis OTHER concepts; test on txt-vis."""
    lk=embedding_lookup(meta,E)
    rows=[]
    for train_bw in (False,True):
        train_concepts=complete_concepts(meta,train_bw)
        for test_bw in (False,True):
            test_concepts=complete_concepts(meta,test_bw)
            for concept in test_concepts:
                ds=[]
                for c in train_concepts:
                    if c==concept: continue
                    ids=pair_cell_ids(meta,c,train_bw)
                    ds.append(lk[ids["mix"]]-lk[ids["vis"]])
                if not ds: continue
                w=safe_norm(np.mean([safe_norm(d) for d in ds],axis=0))
                ids=pair_cell_ids(meta,concept,test_bw)
                ev=lk[ids["vis"]]; et=lk[ids["txt"]]; em=lk[ids["mix"]]
                txt_margin=float(np.dot(et-ev,w))
                mix_margin=float(np.dot(em-ev,w))
                before=cosine_np(et,ev)
                pair=np.stack([et,ev])
                pp=normalize_np(remove_direction_np(pair,w))
                after=cosine_np(pp[0],pp[1])

                others=[]
                for c2 in test_concepts:
                    if c2==concept: continue
                    ids2=pair_cell_ids(meta,c2,test_bw)
                    others.append(cosine_np(et,lk[ids2["vis"]]))
                others_perp=[]
                etp=pp[0]
                for c2 in test_concepts:
                    if c2==concept: continue
                    ids2=pair_cell_ids(meta,c2,test_bw)
                    vv=normalize_np(remove_direction_np(lk[ids2["vis"]][None],w))[0]
                    others_perp.append(cosine_np(etp,vv))
                rec={
                    "train_appearance":"bw" if train_bw else "rgb",
                    "test_appearance":"bw" if test_bw else "rgb",
                    "concept":concept,
                    "txt_minus_vis_projection":txt_margin,
                    "mix_minus_vis_projection":mix_margin,
                    "txt_vis_cos_before":before,
                    "txt_vis_cos_after_remove_textdir":after,
                    "same_minus_unrelated_before":before-float(np.mean(others)) if others else np.nan,
                    "same_minus_unrelated_after":after-float(np.mean(others_perp)) if others_perp else np.nan,
                }
                if prompt_labels is not None and prompt_E is not None and concept in set(prompt_labels):
                    T=normalize_np(prompt_E)
                    label_to_i={x:i for i,x in enumerate(prompt_labels)}
                    j=label_to_i[concept]
                    Tp=normalize_np(remove_direction_np(T,w))
                    before_txt=normalize_np(et[None])[0]@T.T
                    before_vis=normalize_np(ev[None])[0]@T.T
                    after_txt=pp[0]@Tp.T
                    after_vis=pp[1]@Tp.T
                    rec.update({
                        "txt_prompt_rank_before":int(np.where(np.argsort(-before_txt)==j)[0][0])+1,
                        "txt_prompt_rank_after":int(np.where(np.argsort(-after_txt)==j)[0][0])+1,
                        "vis_prompt_rank_before":int(np.where(np.argsort(-before_vis)==j)[0][0])+1,
                        "vis_prompt_rank_after":int(np.where(np.argsort(-after_vis)==j)[0][0])+1,
                        "txt_prompt_cos_before":float(before_txt[j]),
                        "txt_prompt_cos_after":float(after_txt[j]),
                        "vis_prompt_cos_before":float(before_vis[j]),
                        "vis_prompt_cos_after":float(after_vis[j]),
                    })
                rows.append(rec)
    return pd.DataFrame(rows)

def prompt_metrics(meta,E,prompt_labels,prompt_E):
    E=normalize_np(E); T=normalize_np(prompt_E)
    label_to_i={x:i for i,x in enumerate(prompt_labels)}
    sims=E@T.T
    rows=[]
    for i,r in enumerate(meta.itertuples(index=False)):
        if r.concept not in label_to_i: continue
        j=label_to_i[r.concept]
        order=np.argsort(-sims[i])
        rank=int(np.where(order==j)[0][0])+1
        top=int(order[0])
        second=float(np.partition(sims[i],-2)[-2]) if len(prompt_labels)>1 else np.nan
        rows.append({
            "stim_id":r.stim_id,"concept":r.concept,"condition":r.condition,"bw":bool(r.bw),"category":r.category,
            "own_prompt_cosine":float(sims[i,j]),"own_prompt_rank":rank,
            "top1_prompt":prompt_labels[top],"top1_correct":int(top==j),
            "top1_cosine":float(sims[i,top]),"own_vs_best_other_margin":float(sims[i,j]-np.max(np.delete(sims[i],j))) if len(prompt_labels)>1 else np.nan,
        })
    return pd.DataFrame(rows)

def analyze_embedding_space(model_name,space,meta,E,prompt_labels,prompt_E,out:Path):
    out=ensure_dir(out)
    E=normalize_np(E)
    np.savez(out/"final_embeddings.npz",embeddings=E.astype(np.float32),stim_ids=np.asarray(meta.stim_id.astype(str)))
    pm=prompt_metrics(meta,E,prompt_labels,prompt_E)
    pm.to_csv(out/"prompt_alignment.csv",index=False)

    # PCA.
    z,components,mean,ev=pca_2d(E)
    pca=meta[["stim_id","concept","condition","bw","category"]].copy()
    pca["pc1"]=z[:,0]; pca["pc2"]=z[:,1]
    pca.to_csv(out/"pca_points.csv",index=False)
    plot_pca(pca,ev,out/"PCA_FINAL_EMBEDDINGS.png",f"{model_name} / {space}")
    plot_pca_triplets(pca,out/"PCA_RGB_TRIPLETS.png",f"{model_name} / {space} — RGB",False)
    plot_pca_triplets(pca,out/"PCA_BW_TRIPLETS.png",f"{model_name} / {space} — BW",True)
    if "trout" in set(meta.concept):
        plot_trout(pca,out/"PCA_TROUT_ZOOM.png",f"{model_name} / {space} — trout")

    dr,dv=build_direction_rows(meta,E)
    dr.to_csv(out/"paired_direction_rows.csv",index=False)
    np.savez(out/"paired_direction_vectors.npz",**dv)

    summary=[]
    for bw in (False,True):
        for direction in ("add_text","add_visual","txt_minus_vis"):
            arr=[v for k,v in dv.items() if k.startswith(f"{direction}_{'bw' if bw else 'rgb'}_")]
            if arr:
                rec={"appearance":"bw" if bw else "rgb","direction":direction,**direction_stats(np.stack(arr))}
                summary.append(rec)
        # Mean text-vs-visual angle.
        dt=[v for k,v in dv.items() if k.startswith(f"add_text_{'bw' if bw else 'rgb'}_")]
        dvv=[v for k,v in dv.items() if k.startswith(f"add_visual_{'bw' if bw else 'rgb'}_")]
        if dt and dvv:
            wt=safe_norm(np.mean([safe_norm(x) for x in dt],axis=0))
            wv=safe_norm(np.mean([safe_norm(x) for x in dvv],axis=0))
            summary.append({
                "appearance":"bw" if bw else "rgb","direction":"mean_text_vs_visual_axis",
                "n":min(len(dt),len(dvv)),"mean_pairwise_cosine":float(np.dot(wt,wv)),
            })
    sdf=pd.DataFrame(summary)
    sdf.to_csv(out/"paired_direction_summary.csv",index=False)

    loo=loo_provenance_test(meta,E,prompt_labels,prompt_E)
    loo.to_csv(out/"loo_text_provenance_test.csv",index=False)
    lsum=loo.groupby(["train_appearance","test_appearance"]).agg(
        n=("concept","size"),
        positive_txt_projection=("txt_minus_vis_projection",lambda s:float((s>0).mean())),
        mean_txt_projection=("txt_minus_vis_projection","mean"),
        positive_mix_projection=("mix_minus_vis_projection",lambda s:float((s>0).mean())),
        mean_mix_projection=("mix_minus_vis_projection","mean"),
        txt_vis_cos_before=("txt_vis_cos_before","mean"),
        txt_vis_cos_after=("txt_vis_cos_after_remove_textdir","mean"),
        same_minus_unrelated_before=("same_minus_unrelated_before","mean"),
        same_minus_unrelated_after=("same_minus_unrelated_after","mean"),
        txt_prompt_top1_before=("txt_prompt_rank_before",lambda x:float((x==1).mean()) if len(x) else np.nan),
        txt_prompt_top1_after=("txt_prompt_rank_after",lambda x:float((x==1).mean()) if len(x) else np.nan),
        vis_prompt_top1_before=("vis_prompt_rank_before",lambda x:float((x==1).mean()) if len(x) else np.nan),
        vis_prompt_top1_after=("vis_prompt_rank_after",lambda x:float((x==1).mean()) if len(x) else np.nan),
    ).reset_index()
    lsum.to_csv(out/"loo_text_provenance_summary.csv",index=False)
    plot_provenance_ablation(lsum,out/"TEXT_PROVENANCE_LOO_ABLATION.png",f"{model_name} / {space}")

    # Explicit text/visual axes.
    axis_rows=[]
    for bw in (False,True):
        dt=[v for k,v in dv.items() if k.startswith(f"add_text_{'bw' if bw else 'rgb'}_")]
        dvis=[v for k,v in dv.items() if k.startswith(f"add_visual_{'bw' if bw else 'rgb'}_")]
        if not dt or not dvis: continue
        wt=safe_norm(np.mean([safe_norm(x) for x in dt],axis=0))
        wv=safe_norm(np.mean([safe_norm(x) for x in dvis],axis=0))
        wvo=safe_norm(wv-np.dot(wv,wt)*wt)
        # Center per concept triplet to remove semantic location before plotting provenance.
        coords=[]
        qmeta=meta[meta.bw.eq(bw)].copy()
        lk=embedding_lookup(meta,E)
        for concept in complete_concepts(meta,bw):
            ids=pair_cell_ids(meta,concept,bw)
            trio=np.stack([lk[ids[c]] for c in ("vis","txt","mix")])
            center=trio.mean(0)
            for cond,e in zip(("vis","txt","mix"),trio):
                coords.append({"concept":concept,"condition":cond,"x":float(np.dot(e-center,wt)),"y":float(np.dot(e-center,wvo))})
        ad=pd.DataFrame(coords)
        ad.to_csv(out/f"TEXT_VISUAL_AXES_{'BW' if bw else 'RGB'}.csv",index=False)
        plot_axes(ad,out/f"TEXT_VISUAL_AXES_{'BW' if bw else 'RGB'}.png",f"{model_name} / {space} — {'BW' if bw else 'RGB'}")
        axis_rows.append({"appearance":"bw" if bw else "rgb","text_visual_axis_cosine":float(np.dot(wt,wv))})
    pd.DataFrame(axis_rows).to_csv(out/"text_visual_axis_summary.csv",index=False)

    # Trout arithmetic.
    trout=[]
    for bw in (False,True):
        if "trout" not in complete_concepts(meta,bw): continue
        ids=pair_cell_ids(meta,"trout",bw); lk=embedding_lookup(meta,E)
        v,t,m=lk[ids["vis"]],lk[ids["txt"]],lk[ids["mix"]]
        trout.append({
            "appearance":"bw" if bw else "rgb",
            "cos_vis_txt":cosine_np(v,t),"cos_vis_mix":cosine_np(v,m),"cos_txt_mix":cosine_np(t,m),
            "norm_add_text":float(np.linalg.norm(m-v)),"norm_add_visual":float(np.linalg.norm(m-t)),
            "cos_addtext_addvisual":cosine_np(m-v,m-t),
        })
    pd.DataFrame(trout).to_csv(out/"trout_vector_arithmetic.csv",index=False)

    return {
        "E":E,"direction_vectors":dv,"direction_summary":sdf,
        "loo":loo,"loo_summary":lsum,"prompt":pm,
    }

def plot_pca(pca,ev,path,title):
    fig,ax=plt.subplots(figsize=(10,7))
    for cat,g in pca.groupby("category"):
        ax.scatter(g.pc1,g.pc2,s=46,alpha=.82,label=cat,color=COLORS.get(cat))
    # Thin concept-linked graph: vis->mix and txt->mix, plus RGB<->BW for same condition.
    for concept,gc in pca.groupby("concept"):
        d={(r.condition,bool(r.bw)):(r.pc1,r.pc2) for r in gc.itertuples()}
        for bw in (False,True):
            if ("mix",bw) in d:
                for src in ("vis","txt"):
                    if (src,bw) in d:
                        a=d[(src,bw)]; b=d[("mix",bw)]
                        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=.45,alpha=.23,color="0.25"))
        for cond in ("vis","txt","mix"):
            if (cond,False) in d and (cond,True) in d:
                a=d[(cond,False)]; b=d[(cond,True)]
                ax.plot([a[0],b[0]],[a[1],b[1]],lw=.35,alpha=.18,color="0.45",ls=":")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title(title+"\nfinal normalized image embeddings")
    ax.grid(alpha=.12); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(path,dpi=240); plt.close(fig)

def plot_pca_triplets(pca,path,title,bw):
    q=pca[pca.bw.eq(bool(bw))]
    fig,ax=plt.subplots(figsize=(9,7))
    for cat,g in q.groupby("category"):
        ax.scatter(g.pc1,g.pc2,s=55,alpha=.85,label=cat,color=COLORS.get(cat))
    for concept,gc in q.groupby("concept"):
        d={r.condition:(r.pc1,r.pc2) for r in gc.itertuples()}
        if "mix" in d:
            for src in ("vis","txt"):
                if src in d:
                    ax.annotate("",xy=d["mix"],xytext=d[src],arrowprops=dict(arrowstyle="->",lw=.55,alpha=.28,color="0.2"))
    ax.set_title(title); ax.grid(alpha=.12); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=240); plt.close(fig)

def plot_trout(pca,path,title):
    q=pca[pca.concept.eq("trout")]
    if q.empty:return
    pad=max(q.pc1.max()-q.pc1.min(),q.pc2.max()-q.pc2.min(),.05)*.35
    fig,ax=plt.subplots(figsize=(7,6))
    for r in q.itertuples():
        ax.scatter(r.pc1,r.pc2,s=90,color=COLORS.get(r.category),marker=MARKERS.get(r.condition,"o"))
        ax.annotate(r.category,(r.pc1,r.pc2),xytext=(4,4),textcoords="offset points",fontsize=9)
    ax.set_xlim(q.pc1.min()-pad,q.pc1.max()+pad); ax.set_ylim(q.pc2.min()-pad,q.pc2.max()+pad)
    ax.set_title(title); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(path,dpi=240); plt.close(fig)

def plot_provenance_ablation(s,path,title):
    if s.empty:return
    labels=[f"{r.train_appearance}->{r.test_appearance}" for r in s.itertuples()]
    x=np.arange(len(s)); w=.35
    fig,ax=plt.subplots(figsize=(8.8,5.5))
    ax.bar(x-w/2,s.txt_vis_cos_before,w,label="before")
    ax.bar(x+w/2,s.txt_vis_cos_after,w,label="after removing held-out text axis")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("same-concept txt↔vis cosine")
    ax.set_title(title+"\ntext direction learned from mix-vis on other concepts")
    ax.legend(); ax.grid(axis="y",alpha=.12); fig.tight_layout(); fig.savefig(path,dpi=230); plt.close(fig)

def plot_axes(df,path,title):
    fig,ax=plt.subplots(figsize=(8,6.5))
    cmap={"vis":"#1f77b4","txt":"#d62728","mix":"#9467bd"}
    for cond,g in df.groupby("condition"):
        ax.scatter(g.x,g.y,s=55,alpha=.8,label=cond,color=cmap[cond])
    for concept,g in df.groupby("concept"):
        d={r.condition:(r.x,r.y) for r in g.itertuples()}
        if "mix" in d:
            for src in ("vis","txt"):
                if src in d: ax.annotate("",xy=d["mix"],xytext=d[src],arrowprops=dict(arrowstyle="->",lw=.5,alpha=.25,color="0.2"))
    ax.axvline(0,lw=.8,color=".5"); ax.axhline(0,lw=.8,color=".5")
    ax.set_xlabel("paired added-text direction"); ax.set_ylabel("visual-add orthogonal component")
    ax.set_title(title+"\nconcept-centered final embedding coordinates")
    ax.legend(); ax.grid(alpha=.12); fig.tight_layout(); fig.savefig(path,dpi=230); plt.close(fig)

# =============================================================================
# Role analyses
# =============================================================================

def role_localization(meta,cache,input_masks):
    id_to_i={str(s):i for i,s in enumerate(cache["stim_ids"].astype(str))}
    rows=[]
    for r in meta.itertuples(index=False):
        i=id_to_i[r.stim_id]; im=input_masks[r.stim_id]
        regions={"text":im["text_mask"],"object":im["object_mask"],"background":im["background_mask"]}
        for role in ROLES:
            rm=mask_for_role(cache,i,role)
            nr=int(rm.sum())
            for region,gm in regions.items():
                overlap=int(np.logical_and(rm,gm).sum())
                region_frac=float(gm.mean())
                role_frac=float(overlap/nr) if nr else np.nan
                enrich=float(role_frac/region_frac) if nr and region_frac>0 else np.nan
                rows.append({
                    "stim_id":r.stim_id,"concept":r.concept,"condition":r.condition,"bw":bool(r.bw),"category":r.category,
                    "role":role,"region":region,"role_count":nr,"region_patch_fraction":region_frac,
                    "role_count_in_region":overlap,"role_fraction_in_region":role_frac,"enrichment":enrich,
                })
    return pd.DataFrame(rows)


def plot_role_stage_by_category(counts, model_dir):
    q=counts[counts["stage"].eq("pre23")].groupby("category").agg(
        register=("role_reg_count","mean"),
        hidden_mu=("hidden_mu_count","mean"),
        scratchpad=("scratchpad_count","mean"),
    ).reset_index()
    order=["vis_rgb","txt_rgb","mix_rgb","vis_bw","txt_bw","mix_bw"]
    q=q.set_index("category").reindex(order)
    x=np.arange(len(order)); w=.25
    fig,ax=plt.subplots(figsize=(9.5,5.7))
    ax.bar(x-w,q["register"],w,label="REG")
    ax.bar(x,q["hidden_mu"],w,label="hidden mu")
    ax.bar(x+w,q["scratchpad"],w,label="scratchpad")
    ax.set_xticks(x); ax.set_xticklabels(order,rotation=25,ha="right")
    ax.set_ylabel("mean pre23 token count")
    ax.set_title("Internal role populations by visual/textual condition")
    ax.legend(); ax.grid(axis="y",alpha=.12); fig.tight_layout()
    fig.savefig(model_dir/"plots"/"ROLE_COUNTS_BY_CATEGORY.png",dpi=230); plt.close(fig)

def plot_role_region_localization(localization, model_dir):
    q=localization[np.isfinite(localization["enrichment"])].groupby(["role","region"]).enrichment.mean().reset_index()
    roles=[r for r in ROLES if r in set(q.role)]
    regions=["text","object","background"]
    x=np.arange(len(roles)); w=.24
    fig,ax=plt.subplots(figsize=(9,5.6))
    for j,region in enumerate(regions):
        vals=[]
        for role in roles:
            g=q[(q.role.eq(role))&(q.region.eq(region))]
            vals.append(float(g.enrichment.iloc[0]) if len(g) else np.nan)
        ax.bar(x+(j-1)*w,vals,w,label=region)
    ax.axhline(1,ls="--",lw=1,color=".4")
    ax.set_xticks(x); ax.set_xticklabels(roles)
    ax.set_ylabel("role occupancy enrichment vs region area")
    ax.set_title("Where internal token roles live relative to input text/object/background")
    ax.legend(); ax.grid(axis="y",alpha=.12); fig.tight_layout()
    fig.savefig(model_dir/"plots"/"ROLE_REGION_ENRICHMENT.png",dpi=230); plt.close(fig)

def plot_role_direction_summary(summary, model_dir):
    if summary.empty:return
    q=summary[summary.direction.eq("add_text")].copy()
    fig,ax=plt.subplots(figsize=(9,5.6))
    for app,marker in [("rgb","o"),("bw","s")]:
        g=q[q.appearance.eq(app)]
        ax.scatter(g.role,g.mean_pairwise_cosine,s=75,marker=marker,label=app)
    ax.axhline(0,lw=1,color=".5")
    ax.set_ylabel("mean pairwise cosine of residualized mix-vis directions")
    ax.set_title("Shared text-add direction inside token roles after removing mu1/mu2")
    ax.legend(); ax.grid(axis="y",alpha=.12); fig.tight_layout()
    fig.savefig(model_dir/"plots"/"ROLE_TEXT_DIRECTION_COHERENCE.png",dpi=230); plt.close(fig)

def plot_xattn_telemetry(telemetry, model_dir):
    if telemetry.empty:return
    cols=[c for c in ["source_present_prob","source_readable_prob","source_gate","glyph_mean","glyph_max"] if c in telemetry.columns]
    if not cols:return
    q=telemetry.groupby("category")[cols].mean().reset_index()
    q.to_csv(model_dir/"xattn_telemetry_by_category.csv",index=False)
    order=["vis_rgb","txt_rgb","mix_rgb","vis_bw","txt_bw","mix_bw"]
    qq=q.set_index("category").reindex(order)
    fig,ax=plt.subplots(figsize=(9.5,5.6))
    for c in [x for x in ("source_present_prob","source_readable_prob","source_gate") if x in qq.columns]:
        ax.plot(range(len(order)),qq[c],marker="o",label=c)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=25,ha="right")
    ax.set_ylim(-.03,1.03); ax.set_ylabel("probability / gate")
    ax.set_title("x-attn source/readability router telemetry")
    ax.legend(); ax.grid(alpha=.12); fig.tight_layout()
    fig.savefig(model_dir/"plots"/"XATTN_ROUTER_BY_CATEGORY.png",dpi=230); plt.close(fig)

def role_pair_jaccard(meta,cache):
    ids={str(s):i for i,s in enumerate(cache["stim_ids"].astype(str))}
    rows=[]
    pairs=[("vis","mix"),("txt","mix"),("vis","txt")]
    lk=cell_lookup(meta)
    for bw in (False,True):
        for concept in sorted(set(meta.concept)):
            for a,b in pairs:
                ka=(a,concept,bw); kb=(b,concept,bw)
                if ka not in lk or kb not in lk:continue
                ia,ib=ids[lk[ka]],ids[lk[kb]]
                for role in ROLES:
                    rows.append({
                        "concept":concept,"bw":bw,"pair":f"{a}_vs_{b}","role":role,
                        "jaccard":jaccard(mask_for_role(cache,ia,role),mask_for_role(cache,ib,role)),
                    })
    return pd.DataFrame(rows)

def build_role_representations(variant,meta,cache,mu1,mu2,prompt_labels,prompt_E,model_dir,args):
    ids={str(s):i for i,s in enumerate(cache["stim_ids"].astype(str))}
    records=[]; raw=[]; perp=[]
    for r in meta.itertuples(index=False):
        i=ids[r.stim_id]
        tok=torch.from_numpy(cache["post_tokens"][i].astype(np.float32))
        tp=remove_role_plane(tok,mu1,mu2)
        for role in ROLES:
            m=mask_for_role(cache,i,role)
            if not m.any(): continue
            records.append({
                "stim_id":r.stim_id,"concept":r.concept,"condition":r.condition,"bw":bool(r.bw),"category":r.category,
                "role":role,"n_patches":int(m.sum()),
            })
            raw.append(tok[m].mean(0).numpy())
            perp.append(tp[m].mean(0).numpy())
    mdf=pd.DataFrame(records)
    if mdf.empty:
        return mdf,{},pd.DataFrame(),pd.DataFrame()

    raw=np.stack(raw).astype(np.float32); perp=np.stack(perp).astype(np.float32)
    # Joint-space diagnostic: final LN/proj applied to the role-mean token.
    chunks=[]
    for st in range(0,len(raw),args.batch_size*4):
        x=torch.from_numpy(raw[st:st+args.batch_size*4])[:,None,:]
        with torch.inference_mode(), amp_context(args.device,args.amp):
            y=project_tokens_to_joint(variant,x)[:,0,:]
        chunks.append(y.numpy())
    joint=normalize_np(np.concatenate(chunks,0)).astype(np.float32)

    np.savez(model_dir/"role_representations.npz",raw=raw,perp_mu12=perp,joint=joint)
    mdf.to_csv(model_dir/"role_representation_metadata.csv",index=False)

    # Prompt retrieval diagnostic.
    T=normalize_np(prompt_E); lab_to_i={x:i for i,x in enumerate(prompt_labels)}
    sims=joint@T.T; rr=[]
    for i,r in enumerate(mdf.itertuples(index=False)):
        if r.concept not in lab_to_i:continue
        j=lab_to_i[r.concept]; order=np.argsort(-sims[i]); top=int(order[0])
        rr.append({
            **r._asdict(),"own_prompt_cosine":float(sims[i,j]),
            "own_prompt_rank":int(np.where(order==j)[0][0])+1,
            "top1_correct":int(top==j),"top1_prompt":prompt_labels[top],
        })
    retrieval=pd.DataFrame(rr)
    retrieval.to_csv(model_dir/"role_prompt_retrieval.csv",index=False)

    # Provenance directions inside role representations AFTER mu1/mu2 removal.
    role_lookup={(r.stim_id,r.role):perp[i] for i,r in enumerate(mdf.itertuples(index=False))}
    vecs={}; drows=[]
    for role in ROLES:
        for bw in (False,True):
            for concept in complete_concepts(meta,bw):
                ids3=pair_cell_ids(meta,concept,bw)
                keys=[(ids3[c],role) for c in ("vis","txt","mix")]
                if not all(k in role_lookup for k in keys):continue
                v,t,m=[role_lookup[k] for k in keys]
                for name,d in [("add_text",m-v),("add_visual",m-t),("txt_minus_vis",t-v)]:
                    vecs[f"{role}__{name}__{'bw' if bw else 'rgb'}__{concept}"]=d.astype(np.float32)
                    drows.append({"role":role,"appearance":"bw" if bw else "rgb","concept":concept,"direction":name,"norm":float(np.linalg.norm(d))})
    pd.DataFrame(drows).to_csv(model_dir/"role_paired_direction_rows.csv",index=False)
    np.savez(model_dir/"role_paired_direction_vectors.npz",**vecs)
    sums=[]
    for role in ROLES:
        for app in ("rgb","bw"):
            for name in ("add_text","add_visual","txt_minus_vis"):
                arr=[v for k,v in vecs.items() if k.startswith(f"{role}__{name}__{app}__")]
                if arr:sums.append({"role":role,"appearance":app,"direction":name,**direction_stats(np.stack(arr))})
    summary=pd.DataFrame(sums)
    summary.to_csv(model_dir/"role_paired_direction_summary.csv",index=False)
    return mdf,vecs,retrieval,summary

def text_spatial_spread(variant,meta,cache,input_masks,emb_result,model_dir,args):
    """Project mix-vis patch deltas onto the model's final added-text direction."""
    ids={str(s):i for i,s in enumerate(cache["stim_ids"].astype(str))}
    lk=cell_lookup(meta)
    rows=[]; map_rows=[]
    plots=ensure_dir(model_dir/"plots")
    size,G,patch,width,nblocks,nheads=model_geometry(variant)

    for bw in (False,True):
        prefix=f"add_text_{'bw' if bw else 'rgb'}_"
        ds=[v for k,v in emb_result["direction_vectors"].items() if k.startswith(prefix)]
        if not ds: continue
        w=safe_norm(np.mean([safe_norm(d) for d in ds],axis=0))
        score_maps=[]; text_cov=[]
        for concept in complete_concepts(meta,bw):
            id3=pair_cell_ids(meta,concept,bw)
            if id3 is None: continue
            imix=ids[id3["mix"]]; ivis=ids[id3["vis"]]
            mix=torch.from_numpy(cache["post_tokens"][imix].astype(np.float32))[None]
            vis=torch.from_numpy(cache["post_tokens"][ivis].astype(np.float32))[None]
            with torch.inference_mode(), amp_context(args.device,args.amp):
                jm=project_tokens_to_joint(variant,mix)[0].numpy()
                jv=project_tokens_to_joint(variant,vis)[0].numpy()
            jm=normalize_np(jm); jv=normalize_np(jv)
            delta=jm-jv
            score=delta@w
            score_maps.append(score)
            masks=input_masks[id3["mix"]]
            text_cov.append(masks["text_coverage"])
            regions={"text":masks["text_mask"],"object":masks["object_mask"],"background":masks["background_mask"]}
            # Regions independent of internal role.
            for region,gm in regions.items():
                if gm.any():
                    rows.append({"appearance":"bw" if bw else "rgb","concept":concept,"role":"ALL","region":region,
                                 "n_patches":int(gm.sum()),"mean_text_projection":float(score[gm].mean()),
                                 "mean_abs_text_projection":float(np.abs(score[gm]).mean())})
            # Role x region, using mix-image role assignment.
            for role in ROLES:
                rm=mask_for_role(cache,imix,role)
                for region,gm in regions.items():
                    mm=rm&gm
                    if mm.any():
                        rows.append({"appearance":"bw" if bw else "rgb","concept":concept,"role":role,"region":region,
                                     "n_patches":int(mm.sum()),"mean_text_projection":float(score[mm].mean()),
                                     "mean_abs_text_projection":float(np.abs(score[mm]).mean())})
            for p,s in enumerate(score):
                map_rows.append({"appearance":"bw" if bw else "rgb","concept":concept,"patch_index0":p,
                                 "row0":p//G,"col0":p%G,"text_projection":float(s),
                                 "text_coverage":float(masks["text_coverage"][p]),
                                 "object_coverage":float(masks["object_coverage"][p])})
        if score_maps:
            mean_score=np.mean(score_maps,axis=0).reshape(G,G)
            mean_text=np.mean(text_cov,axis=0).reshape(G,G)
            fig,ax=plt.subplots(figsize=(6.5,5.8))
            im=ax.imshow(mean_score,interpolation="nearest")
            # The contour is only a guide to where input text tended to occur.
            try: ax.contour(mean_text,levels=[0.01],linewidths=.8)
            except Exception: pass
            ax.set_title(f"{variant.name}: {'BW' if bw else 'RGB'}\nmean patch projection onto final added-text direction")
            ax.set_xlabel("patch col"); ax.set_ylabel("patch row"); fig.colorbar(im,ax=ax)
            fig.tight_layout(); fig.savefig(plots/f"TEXT_DIRECTION_SPREAD_{'BW' if bw else 'RGB'}.png",dpi=230); plt.close(fig)

    df=pd.DataFrame(rows); maps=pd.DataFrame(map_rows)
    df.to_csv(model_dir/"text_direction_spatial_spread.csv",index=False)
    maps.to_csv(model_dir/"text_direction_patch_maps.csv.gz",index=False,compression="gzip")
    return df,maps

# =============================================================================
# Inverse-LAST role spectrum
# =============================================================================

def inverse_last_role_spectrum(cache,mu1,mu2,model_dir,sigma_frac=.12):
    X=torch.from_numpy(cache["post_tokens"].astype(np.float32))
    N,P,D=X.shape
    # Avoid giant temporary tensors: image-by-image accumulation.
    role_sum={r:np.zeros(D,np.float64) for r in ROLES}
    role_n={r:0 for r in ROLES}
    scalar=[]
    freq=torch.arange(D,dtype=torch.float32)-D/2
    sig=max(1.,sigma_frac*D)
    g=torch.exp(-.5*(freq/sig)**2)
    m1=F.normalize(mu1.float(),dim=0); m2=mu2.float()-torch.dot(mu2.float(),m1)*m1; m2=F.normalize(m2,dim=0)
    for i in range(N):
        x=X[i]
        fs=torch.fft.fftshift(torch.fft.fft(x,dim=-1),dim=-1)
        power=fs.abs().square()
        pn=power/power.sum(-1,keepdim=True).clamp_min(EPS)
        ent=-(pn*torch.log(pn.clamp_min(EPS))).sum(-1)/math.log(D)
        low=torch.fft.ifft(torch.fft.ifftshift(fs*g[None,:],dim=-1),dim=-1).real
        high=x-low
        lowf=low.square().sum(-1)/x.square().sum(-1).clamp_min(EPS)
        tv=(x[:,1:]-x[:,:-1]).abs().mean(-1)
        xr=x-(x@m1)[:,None]*m1-(x@m2)[:,None]*m2
        for role in ROLES:
            mask=mask_for_role(cache,i,role)
            if not mask.any():continue
            role_sum[role]+=power[mask].sum(0).numpy()
            role_n[role]+=int(mask.sum())
            scalar.append({"stim_id":str(cache["stim_ids"][i]),"role":role,"n_patches":int(mask.sum()),
                           "channel_spectral_entropy":float(ent[mask].mean()),
                           "channel_lowpass_energy_frac":float(lowf[mask].mean()),
                           "channel_total_variation":float(tv[mask].mean()),
                           "mu12_residual_rms":float(xr[mask].square().mean().sqrt())})
    fr=[]
    for role in ROLES:
        total=role_sum[role].sum()+EPS
        for k,v in enumerate(role_sum[role]):
            fr.append({"role":role,"fftshift_channel_frequency_bin":k,"power":v/max(role_n[role],1),
                       "normalized_power":v/total,"n_patches":role_n[role]})
    sdf=pd.DataFrame(scalar); fdf=pd.DataFrame(fr)
    sdf.to_csv(model_dir/"inverse_last_role_summary.csv",index=False)
    fdf.to_csv(model_dir/"inverse_last_frequency_profiles.csv.gz",index=False,compression="gzip")
    # Simple plot.
    if len(sdf):
        q=sdf.groupby("role").agg(entropy=("channel_spectral_entropy","mean"),low=("channel_lowpass_energy_frac","mean"),
                                  tv=("channel_total_variation","mean")).reset_index()
        fig,ax=plt.subplots(figsize=(8,5.5))
        x=np.arange(len(q)); ax.bar(x-.18,q.low,.36,label="channel low-pass energy")
        ax.bar(x+.18,q.entropy,.36,label="spectral entropy")
        ax.set_xticks(x); ax.set_xticklabels(q.role); ax.set_title("Inverse LAST-style channel-frequency phenotype")
        ax.legend(); fig.tight_layout(); fig.savefig(model_dir/"plots"/"INVERSE_LAST_ROLE_SPECTRUM.png",dpi=220); plt.close(fig)
    return sdf,fdf

# =============================================================================
# Conv1 text-tagging
# =============================================================================

def conv1_text_screen(variant,meta,cache,input_masks,emb_result,model_dir):
    ids={str(s):i for i,s in enumerate(cache["stim_ids"].astype(str))}
    C=cache["conv"].shape[1]
    per=[]
    # Final controlled text margin per concept from backbone.
    E=emb_result["E"]; elk=embedding_lookup(meta,E)
    final_margin={}
    for bw in (False,True):
        ds=[v for k,v in emb_result["direction_vectors"].items() if k.startswith(f"add_text_{'bw' if bw else 'rgb'}_")]
        if not ds:continue
        w=safe_norm(np.mean([safe_norm(d) for d in ds],axis=0))
        for concept in complete_concepts(meta,bw):
            id3=pair_cell_ids(meta,concept,bw)
            final_margin[(concept,bw)]=float(np.dot(elk[id3["mix"]]-elk[id3["vis"]],w))

    for bw in (False,True):
        for concept in complete_concepts(meta,bw):
            id3=pair_cell_ids(meta,concept,bw)
            imix,ivis=ids[id3["mix"]],ids[id3["vis"]]
            mix=cache["conv"][imix].astype(np.float32).reshape(C,-1)
            vis=cache["conv"][ivis].astype(np.float32).reshape(C,-1)
            delta=mix-vis
            masks=input_masks[id3["mix"]]
            t=masks["text_mask"]; bg=masks["background_mask"]; obj=masks["object_mask"]
            if not t.any():continue
            text_abs=np.abs(delta[:,t]).mean(1)
            text_signed=delta[:,t].mean(1)
            bg_abs=np.abs(delta[:,bg]).mean(1) if bg.any() else np.full(C,np.nan)
            obj_abs=np.abs(delta[:,obj]).mean(1) if obj.any() else np.full(C,np.nan)
            for c in range(C):
                per.append({
                    "appearance":"bw" if bw else "rgb","concept":concept,"channel":c,
                    "text_abs_delta":float(text_abs[c]),"text_signed_delta":float(text_signed[c]),
                    "background_abs_delta":float(bg_abs[c]),"object_abs_delta":float(obj_abs[c]),
                    "final_text_margin":final_margin.get((concept,bw),np.nan),
                })
    pdf=pd.DataFrame(per)
    pdf.to_csv(model_dir/"conv1_text_tagging_per_concept.csv.gz",index=False,compression="gzip")
    if pdf.empty:return pdf,pd.DataFrame()
    rows=[]
    for (app,c),g in pdf.groupby(["appearance","channel"]):
        ts=g.text_signed_delta.to_numpy(float)
        sign=np.sign(np.nanmean(ts))
        rows.append({
            "appearance":app,"channel":int(c),"n":len(g),
            "text_abs_delta_mean":float(g.text_abs_delta.mean()),
            "background_abs_delta_mean":float(g.background_abs_delta.mean()),
            "object_abs_delta_mean":float(g.object_abs_delta.mean()),
            "text_specificity_abs":float(g.text_abs_delta.mean()-g.background_abs_delta.mean()),
            "text_to_background_ratio":float(g.text_abs_delta.mean()/max(g.background_abs_delta.mean(),1e-8)),
            "signed_delta_mean":float(g.text_signed_delta.mean()),
            "signed_consistency":float(np.mean(np.sign(ts)==sign)) if sign!=0 else np.nan,
            "corr_text_activation_vs_final_margin":pearson(g.text_abs_delta,g.final_text_margin),
        })
    s=pd.DataFrame(rows)
    morph=conv1_morphology(variant)
    s=s.merge(morph,on="channel",how="left")
    # Combined rank is descriptive only.
    for app in s.appearance.unique():
        idx=s.appearance.eq(app)
        z=s.loc[idx,"text_specificity_abs"].to_numpy(float)
        mu=np.nanmean(z); sd=np.nanstd(z)+1e-8
        s.loc[idx,"text_tag_score"]=(z-mu)/sd
    s=s.sort_values(["appearance","text_tag_score"],ascending=[True,False]).reset_index(drop=True)
    s["rank_within_appearance"]=s.groupby("appearance").cumcount()+1
    s.to_csv(model_dir/"conv1_text_tagging_summary.csv",index=False)

    fig,ax=plt.subplots(figsize=(8.5,6))
    for app,g in s.groupby("appearance"):
        ax.scatter(g.kernel_highfreq_energy_frac,g.text_specificity_abs,s=18,alpha=.55,label=app)
        for r in g.head(8).itertuples():
            ax.annotate(str(r.channel),(r.kernel_highfreq_energy_frac,r.text_specificity_abs),xytext=(2,2),textcoords="offset points",fontsize=7)
    ax.set_xlabel("Conv1 kernel high-frequency energy fraction")
    ax.set_ylabel("controlled text-patch specificity |mix-vis| - background")
    ax.set_title(f"{variant.name}: Conv1 morphology vs later text-tagging input response")
    ax.grid(alpha=.15); ax.legend(); fig.tight_layout(); fig.savefig(model_dir/"plots"/"CONV1_TEXT_TAGGING_VS_FREQUENCY.png",dpi=230); plt.close(fig)
    return pdf,s

def matched_weight_controls(summary,morph,top_channels,seed):
    pool=[c for c in morph.channel.astype(int) if c not in set(top_channels)]
    used=set(); out=[]
    lm={int(r.channel):math.log(max(float(r.weight_l2),1e-12)) for r in morph.itertuples()}
    for c in top_channels:
        candidates=[x for x in pool if x not in used]
        if not candidates:break
        best=min(candidates,key=lambda x:abs(lm[x]-lm[c]))
        used.add(best); out.append(best)
    return out

def causal_conv1_screen(variant,meta,cache,emb_result,screen,args,model_dir):
    if args.conv1_causal_topk<=0 or screen.empty:return pd.DataFrame()
    # Pick top channels by mean RGB/BW rank score, then weight-matched controls.
    mean_score=screen.groupby("channel").text_tag_score.mean().sort_values(ascending=False)
    top=[int(x) for x in mean_score.head(args.conv1_causal_topk).index]
    morph=conv1_morphology(variant)
    controls=matched_weight_controls(screen,morph,top,args.seed)
    jobs=[("selected",c) for c in top]+[("weight_matched_control",c) for c in controls]

    # Only complete controlled vis/mix pairs.
    pair_rows=[]
    for bw in (False,True):
        for concept in complete_concepts(meta,bw):
            ids=pair_cell_ids(meta,concept,bw)
            for cond in ("vis","mix"):
                row=meta[meta.stim_id.eq(ids[cond])].iloc[0]
                pair_rows.append(row)
    pair_meta=pd.DataFrame(pair_rows).drop_duplicates("stim_id").reset_index(drop=True)
    base_lk=embedding_lookup(meta,emb_result["E"])
    d_all=[v for k,v in emb_result["direction_vectors"].items() if "__" not in k and k.startswith("add_text_")]
    # Separate axes per appearance.
    axes={}
    for bw in (False,True):
        ds=[v for k,v in emb_result["direction_vectors"].items() if k.startswith(f"add_text_{'bw' if bw else 'rgb'}_")]
        if ds:axes[bw]=safe_norm(np.mean([safe_norm(d) for d in ds],axis=0))
    baseline={}
    for bw,w in axes.items():
        vals=[]
        for concept in complete_concepts(meta,bw):
            ids=pair_cell_ids(meta,concept,bw); vals.append(np.dot(base_lk[ids["mix"]]-base_lk[ids["vis"]],w))
        baseline[bw]=float(np.mean(vals))

    out=[]
    cdir=ensure_dir(model_dir/"conv1_causal_cache")
    for kind,c in jobs:
        cp=cdir/f"{kind}_ch{int(c):04d}.csv"
        if cp.is_file():
            cached=pd.read_csv(cp)
            out.extend(cached.to_dict("records"))
            print(f"[{variant.name} Conv1 causal] resume {kind} ch{c}")
            continue
        pert={}
        for st in range(0,len(pair_meta),args.batch_size):
            q=pair_meta.iloc[st:st+args.batch_size]
            batch=preprocess_rows(variant,q)
            with torch.inference_mode(),amp_context(args.device,args.amp):
                x,_=prepare_tokens(variant,batch,(c,"ZERO"))
                for bi,blk in enumerate(variant.visual.transformer.resblocks):
                    x=variant.maybe_insert_rn(bi,x); x=blk(x)
                e=F.normalize(visual_final_embedding(variant,x).float(),dim=-1).cpu().numpy()
            for sid,ee in zip(q.stim_id,e):pert[str(sid)]=ee
        jobrows=[]
        for bw,w in axes.items():
            vals=[]
            for concept in complete_concepts(meta,bw):
                ids=pair_cell_ids(meta,concept,bw)
                vals.append(float(np.dot(pert[ids["mix"]]-pert[ids["vis"]],w)))
            mean=float(np.mean(vals))
            rec={"channel":c,"kind":kind,"appearance":"bw" if bw else "rgb","mode":"ZERO",
                 "baseline_text_margin":baseline[bw],"perturbed_text_margin":mean,
                 "fraction_remaining":mean/max(abs(baseline[bw]),1e-9),
                 "margin_change":mean-baseline[bw]}
            out.append(rec); jobrows.append(rec)
        pd.DataFrame(jobrows).to_csv(cp,index=False)
        print(f"[{variant.name} Conv1 causal] {kind} ch{c}")
    d=pd.DataFrame(out); d.to_csv(model_dir/"conv1_causal_text_direction.csv",index=False)
    return d

# =============================================================================
# Model report / cross-model report
# =============================================================================

def write_model_report(name,model_dir,mu_report,counts,localization,role_dir,emb_results,conv_screen,causal):
    lines=[f"# {name}: visual/textual provenance atlas","",
           "## Role plane",
           f"- mu1 raw uncentered energy: {mu_report['mu1_raw_uncentered_energy_fraction']:.8f}",
           f"- mu2 raw uncentered energy: {mu_report['mu2_raw_uncentered_energy_fraction']:.8f}",
           f"- mu2 residual fraction after mu1: {mu_report['mu2_fraction_after_removing_mu1']:.8f}",
           f"- mu1 top coords: {mu_report['mu1_top_coordinates'][:8]}",
           f"- mu2 top coords: {mu_report['mu2_top_coordinates'][:8]}","",
           "## Final embedding provenance"]
    for space,res in emb_results.items():
        lines.append(f"### {space}")
        if len(res["loo_summary"]):
            for r in res["loo_summary"].itertuples():
                lines.append(
                    f"- train {r.train_appearance} -> test {r.test_appearance}: "
                    f"txt projection positive={r.positive_txt_projection:.3f}, "
                    f"txt/vis cos {r.txt_vis_cos_before:.4f}->{r.txt_vis_cos_after:.4f}, "
                    f"same-unrelated {r.same_minus_unrelated_before:.4f}->{r.same_minus_unrelated_after:.4f}"
                )
    q=counts[counts.stage.eq("pre23")].groupby("category").agg(
        reg=("role_reg_count","mean"),hidden=("hidden_mu_count","mean"),scratch=("scratchpad_count","mean")).reset_index()
    lines+=["","## pre23 role counts by category"]
    for r in q.itertuples(): lines.append(f"- {r.category}: REG={r.reg:.3f}, hidden-mu={r.hidden:.3f}, scratch={r.scratch:.3f}")
    if len(localization):
        q=localization[(localization.region.eq("background"))].groupby(["category","role"]).enrichment.mean().reset_index()
        lines+=["","## Background enrichment"]
        for r in q.itertuples(): lines.append(f"- {r.category}/{r.role}: {r.enrichment:.3f}")
    if len(role_dir):
        lines+=["","## Role-plane-residualized provenance directions"]
        for r in role_dir[(role_dir.direction.eq("add_text"))].itertuples():
            lines.append(f"- {r.role}/{r.appearance}: n={r.n}, paircos={r.mean_pairwise_cosine:.3f}, pc1={r.direction_pc1_energy_fraction:.3f}")
    if len(conv_screen):
        lines+=["","## Top Conv1 controlled text-tagging channels"]
        for app in sorted(conv_screen.appearance.unique()):
            lines.append(f"### {app}")
            for r in conv_screen[conv_screen.appearance.eq(app)].head(10).itertuples():
                lines.append(f"- ch{r.channel}: score={r.text_tag_score:+.3f}, specificity={r.text_specificity_abs:.5f}, HF={r.kernel_highfreq_energy_frac:.3f}, DC={r.kernel_dc_energy_frac:.3f}")
    if len(causal):
        lines+=["","## Conv1 causal single-channel ZERO"]
        for r in causal[causal.kind.eq("selected")].sort_values("fraction_remaining").head(10).itertuples():
            lines.append(f"- ch{r.channel}/{r.appearance}: text-margin remaining={r.fraction_remaining:.3f}")
    lines+=["","## Guardrails",
            "- PCA is descriptive; paired/leave-one-concept-out directions are the stronger geometry test.",
            "- The text-add direction is learned from mix-vis, then tested on txt-vis; this avoids defining and testing provenance on the same pair.",
            "- Applying final ln_post/proj to non-CLS role means is a diagnostic readout, not a claim that CLIP explicitly decodes patches that way.",
            "- Internal role masks are backbone roles; x-attn CONTENT is a post-backbone candidate-independent correction.",
            "- The scratchpad definition is unchanged from the previous role-plane experiment."]
    (model_dir/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def cross_model_outputs(root,model_summaries):
    cross=ensure_dir(root/"CROSS_MODEL")
    # Embedding LOO summary.
    emb=[]
    roles=[]
    conv=[]
    causal=[]
    for name,s in model_summaries.items():
        for space,res in s["emb"].items():
            q=res["loo_summary"].copy(); q.insert(0,"embedding_space",space); q.insert(0,"model",name); emb.append(q)
        q=s["counts"]; q=q[q.stage.eq("pre23")].groupby("category").agg(
            role_reg_mean=("role_reg_count","mean"),hidden_mu_mean=("hidden_mu_count","mean"),scratchpad_mean=("scratchpad_count","mean")).reset_index()
        q.insert(0,"model",name); roles.append(q)
        if len(s["conv"]):
            q=s["conv"].copy(); q.insert(0,"model",name); conv.append(q)
        if len(s["causal"]):
            q=s["causal"].copy(); q.insert(0,"model",name); causal.append(q)
    edf=pd.concat(emb,ignore_index=True) if emb else pd.DataFrame()
    rdf=pd.concat(roles,ignore_index=True) if roles else pd.DataFrame()
    cdf=pd.concat(conv,ignore_index=True) if conv else pd.DataFrame()
    cadf=pd.concat(causal,ignore_index=True) if causal else pd.DataFrame()
    edf.to_csv(cross/"embedding_provenance_comparison.csv",index=False)
    rdf.to_csv(cross/"role_count_comparison.csv",index=False)
    if len(cdf):cdf.to_csv(cross/"conv1_text_tagging_comparison.csv",index=False)
    if len(cadf):cadf.to_csv(cross/"conv1_causal_comparison.csv",index=False)

    if len(rdf):
        fig,ax=plt.subplots(figsize=(10,5.8))
        for model,g in rdf.groupby("model"):
            order=["vis_rgb","txt_rgb","mix_rgb","vis_bw","txt_bw","mix_bw"]
            gg=g.set_index("category").reindex(order)
            ax.plot(range(len(order)),gg.scratchpad_mean,marker="o",label=model)
        ax.set_xticks(range(6)); ax.set_xticklabels(order,rotation=25,ha="right")
        ax.set_ylabel("mean pre23 scratchpad count")
        ax.set_title("Scratchpad population on real visual/textual stimuli")
        ax.legend(); ax.grid(alpha=.12); fig.tight_layout(); fig.savefig(cross/"SCRATCHPADS_BY_CATEGORY_AND_MODEL.png",dpi=230); plt.close(fig)

    if len(edf):
        q=edf[(edf.train_appearance.eq("rgb"))&(edf.test_appearance.eq("rgb"))]
        fig,ax=plt.subplots(figsize=(9,5.5))
        labels=[f"{r.model}/{r.embedding_space}" for r in q.itertuples()]
        x=np.arange(len(q)); ax.bar(x,q.positive_txt_projection)
        ax.set_xticks(x); ax.set_xticklabels(labels,rotation=30,ha="right")
        ax.set_ylim(0,1.05); ax.set_ylabel("held-out concepts with positive txt-vis projection")
        ax.set_title("Cross-concept recovery of a shared text-provenance direction")
        ax.grid(axis="y",alpha=.12); fig.tight_layout(); fig.savefig(cross/"TEXT_PROVENANCE_TRANSFER_MODELS.png",dpi=230); plt.close(fig)

def compact_zip(root:Path):
    zpath=root/"compact_summary_conv1_visualtextual_provenance.zip"
    exclude={"baseline_cache.npz","role_representations.npz","final_embeddings.npz","paired_direction_vectors.npz","role_paired_direction_vectors.npz"}
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob("*")):
            if not p.is_file():continue
            if p == zpath: continue
            if p.name in exclude:continue
            if p.stat().st_size>35_000_000:continue
            z.write(p,arcname=str(p.relative_to(root)))
    return zpath

# =============================================================================
# Main
# =============================================================================

def parse_args(argv=None):
    ap=argparse.ArgumentParser(description="Visual/textual provenance + role-plane atlas")
    ap.add_argument("--image_dir",default="image_sets/visualtextual")
    ap.add_argument("--output_dir",default="out_paper_reproduction/conv1/visualtextual_provenance")
    ap.add_argument("--models",default="pretrained,gmp,full_xattn")
    ap.add_argument("--repo_root",default="")
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--batch_size",type=int,default=12)
    ap.add_argument("--seed",type=int,default=20260918)

    ap.add_argument("--pretrained_model",default="openai/clip-vit-large-patch14")
    ap.add_argument("--gmp_checkpoint",default="zer0int/CLIP-GmP-ViT-L-14")
    ap.add_argument("--xattn_model",default="zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX")
    ap.add_argument("--xattn_revision",default="")
    ap.add_argument("--hf_cache_dir",default="")

    ap.add_argument("--register_norm_threshold",type=float,default=60.0)
    ap.add_argument("--mu_reg_z",type=float,default=3.0)
    ap.add_argument("--mu_hidden_z",type=float,default=5.0)
    ap.add_argument("--scratch_z",type=float,default=4.0)
    ap.add_argument("--scratch_topk",type=int,default=8)

    ap.add_argument("--text_pixel_threshold",type=float,default=.02)
    ap.add_argument("--text_patch_coverage",type=float,default=.005)
    ap.add_argument("--object_pixel_threshold",type=float,default=.08)
    ap.add_argument("--object_patch_coverage",type=float,default=.03)

    ap.add_argument("--conv1_causal_topk",type=int,default=8)
    ap.add_argument("--self_test",action="store_true")
    return ap.parse_args(argv)

def self_test():
    self_test_common()
    # Controlled direction sanity.
    v=np.array([1.,0.,0.]); t=np.array([0.,1.,0.]); m=np.array([1.,1.,0.])
    D=np.stack([m-v,m-v])
    assert direction_stats(D)["mean_cosine_to_centroid"]>.99
    # Direction removal.
    x=np.stack([np.array([1.,2.,0.]),np.array([1.,-2.,0.])])
    y=remove_direction_np(x,np.array([0.,1.,0.]))
    assert np.allclose(y[:,1],0)

    # Regression: exercise the leave-one-concept-out provenance path itself,
    # so missing geometry helpers cannot survive py_compile unnoticed.
    rows=[]
    for concept in ("cat","trout"):
        for cond in ("vis","txt","mix"):
            rows.append({
                "stim_id":f"{cond}_{concept}",
                "concept":concept,
                "condition":cond,
                "bw":False,
                "category":f"{cond}_rgb",
            })
    meta=pd.DataFrame(rows)
    # Shared semantic axis per concept + shared text-provenance offset.
    E=np.asarray([
        [1.0,0.0,0.0,0.0],   # vis cat
        [1.0,0.5,0.0,0.0],   # txt cat
        [1.0,0.5,0.2,0.0],   # mix cat
        [0.0,1.0,0.0,0.0],   # vis trout
        [0.0,1.0,0.0,0.5],   # txt trout
        [0.2,1.0,0.0,0.5],   # mix trout
    ],dtype=np.float64)
    # Reorder to match rows: cat vis/txt/mix, trout vis/txt/mix.
    prompts=np.eye(2,4,dtype=np.float64)
    loo=loo_provenance_test(meta,E,["cat","trout"],prompts)
    assert len(loo)==2
    assert np.isfinite(loo["txt_vis_cos_before"]).all()
    print("self-test OK")

def main(argv=None):
    args=parse_args(argv)
    if args.self_test:
        self_test(); return 0
    seed_all(args.seed)
    root=ensure_dir(Path(args.output_dir))
    manifest=parse_dataset(Path(args.image_dir))
    manifest.to_csv(root/"image_manifest.csv",index=False)
    audit=manifest_audit(manifest); audit.to_csv(root/"dataset_pairing_audit.csv",index=False)
    print("[dataset]",len(manifest),"images")
    print("[dataset] complete RGB concepts:",int(audit.complete_rgb.sum()),"complete BW:",int(audit.complete_bw.sum()))
    print("[dataset] incomplete concepts:",audit[(audit.complete_rgb==0)|(audit.complete_bw==0)].concept.tolist())

    runtime=import_runtime(args.repo_root)
    model_names=parse_models(args.models)
    model_summaries={}
    masks_written=False; input_masks=None

    for model_name in model_names:
        print("\n"+"="*88+f"\nMODEL {model_name}\n"+"="*88)
        md=ensure_dir(root/model_name); ensure_dir(md/"plots")
        variant=load_variant(model_name,args,runtime)
        size,G,patch,width,nblocks,nheads=model_geometry(variant)
        if not masks_written:
            input_masks,maskdf=build_input_region_masks(
                manifest,size,G,
                text_pixel_threshold=args.text_pixel_threshold,
                text_patch_coverage=args.text_patch_coverage,
                object_pixel_threshold=args.object_pixel_threshold,
                object_patch_coverage=args.object_patch_coverage,
            )
            maskdf.to_csv(root/"input_region_masks.csv.gz",index=False,compression="gzip")
            masks_written=True

        # Role plane: discover only from VISUAL-ONLY images, both RGB and BW.
        cal=manifest[manifest.condition.eq("vis")].copy()
        basis_path=md/"mu_basis.npz"; basis_report_path=md/"mu_basis_report.json"
        basis_sig=hashlib.sha256(("\n".join(cal.stim_id.astype(str))+
            f"|{args.register_norm_threshold}|{variant.source_info}").encode()).hexdigest()[:20]
        if basis_path.is_file() and basis_report_path.is_file():
            rep=json.loads(basis_report_path.read_text())
            if rep.get("basis_signature")==basis_sig:
                z=np.load(basis_path); mu1=torch.from_numpy(z["mu1"]).float(); mu2=torch.from_numpy(z["mu2"]).float()
                mu_report=rep; print(f"[{model_name} mu basis] resume")
            else:
                X,bmeta=collect_register_means(variant,cal,args.batch_size,args.register_norm_threshold,args.amp)
                mu1,mu2,mu_report=discover_mu_from_image_means(X); mu_report["basis_signature"]=basis_sig
                np.savez_compressed(basis_path,mu1=mu1.numpy(),mu2=mu2.numpy()); basis_report_path.write_text(json.dumps(mu_report,indent=2))
                bmeta.to_csv(md/"mu_basis_calibration.csv",index=False)
        else:
            X,bmeta=collect_register_means(variant,cal,args.batch_size,args.register_norm_threshold,args.amp)
            mu1,mu2,mu_report=discover_mu_from_image_means(X); mu_report["basis_signature"]=basis_sig
            np.savez_compressed(basis_path,mu1=mu1.numpy(),mu2=mu2.numpy()); basis_report_path.write_text(json.dumps(mu_report,indent=2))
            bmeta.to_csv(md/"mu_basis_calibration.csv",index=False)

        # Prompt embeddings, raw literal labels.
        prompt_labels=sorted(set(manifest.concept.astype(str)))
        prompt_text=[x.replace("_"," ") for x in prompt_labels]
        prompt_E=encode_prompts(variant,prompt_text,args.amp).numpy()
        np.savez(md/"text_prompt_embeddings.npz",embeddings=prompt_E.astype(np.float32),labels=np.asarray(prompt_labels))
        pd.DataFrame({"concept":prompt_labels,"prompt":prompt_text}).to_csv(md/"text_prompts.csv",index=False)

        cache,counts,telemetry=run_model_baseline(variant,manifest,mu1,mu2,args,md)
        counts.groupby(["stage","category"]).agg(
            n=("stim_id","size"),role_reg_mean=("role_reg_count","mean"),hidden_mu_mean=("hidden_mu_count","mean"),
            scratchpad_mean=("scratchpad_count","mean"),ordinary_mean=("ordinary_count","mean")
        ).reset_index().to_csv(md/"role_stage_by_category.csv",index=False)
        plot_role_stage_by_category(counts,md)
        plot_xattn_telemetry(telemetry,md)

        # Embedding spaces.
        emb_results={}
        emb_results["backbone"]=analyze_embedding_space(
            model_name,"backbone",manifest,cache["backbone"],prompt_labels,prompt_E,ensure_dir(md/"embedding_backbone")
        )
        if "content" in cache:
            emb_results["content"]=analyze_embedding_space(
                model_name,"content",manifest,cache["content"],prompt_labels,prompt_E,ensure_dir(md/"embedding_content")
            )

        # Internal roles.
        loc=role_localization(manifest,cache,input_masks); loc.to_csv(md/"role_input_region_localization.csv",index=False)
        plot_role_region_localization(loc,md)
        loc.groupby(["category","role","region"]).agg(
            n=("stim_id","size"),mean_enrichment=("enrichment","mean"),mean_role_count=("role_count","mean"),
            mean_role_fraction=("role_fraction_in_region","mean")
        ).reset_index().to_csv(md/"role_input_region_summary.csv",index=False)
        jac=role_pair_jaccard(manifest,cache); jac.to_csv(md/"role_pair_jaccard.csv",index=False)

        role_meta,role_vecs,role_ret,role_dir=build_role_representations(
            variant,manifest,cache,mu1,mu2,prompt_labels,prompt_E,md,args
        )
        plot_role_direction_summary(role_dir,md)
        # Spatial spread uses backbone final direction, since patch tokens are backbone states.
        spread,spreadmap=text_spatial_spread(variant,manifest,cache,input_masks,emb_results["backbone"],md,args)
        inv,invfreq=inverse_last_role_spectrum(cache,mu1,mu2,md)

        # Conv1 -> text tagging.
        conv_per,conv_screen=conv1_text_screen(variant,manifest,cache,input_masks,emb_results["backbone"],md)
        causal=causal_conv1_screen(variant,manifest,cache,emb_results["backbone"],conv_screen,args,md)

        write_model_report(model_name,md,mu_report,counts,loc,role_dir,emb_results,conv_screen,causal)
        (md/"model_audit.json").write_text(json.dumps({
            "name":model_name,"source_info":variant.source_info,"geometry":{"image_size":size,"grid":G,"patch":patch,"width":width,"blocks":nblocks,"heads":nheads},
            "mu_basis_visual_only":True,
        },indent=2,default=str))

        model_summaries[model_name]={
            "emb":emb_results,"counts":counts,"conv":conv_screen,"causal":causal
        }
        del variant,cache
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    cross_model_outputs(root,model_summaries)
    # Copy source scripts into output for provenance when running from package.
    for n in ["probe_VISUALTEXTUAL_PROVENANCE_ROLE_ATLAS.py","visualtextual_probe_common.py"]:
        p=Path(__file__).resolve().parent/n
        if p.is_file(): shutil.copy2(p,root/n)
    z=compact_zip(root)
    print("\nDONE")
    print("Compact handoff:",z)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
