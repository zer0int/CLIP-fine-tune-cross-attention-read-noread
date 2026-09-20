#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B20 sharpener-vs-flattener subfamily analysis for OpenAI CLIP ViT-L/14 variants.

This probe is a follow-up to probe_B20_WRITEBACK_NEURONS.py. It does NOT redo
B20 population discovery. Instead it reads each model's saved B20 discovery
artifacts and asks which discovered hidden units push B21 attention heads toward
or away from a sharp fixed-winner softmax state.

Per model it automatically chooses:
  * sharp_head: B21 head with largest positive B20-MLP top1 effect
  * flat_head:  B21 head with largest negative B20-MLP top1 effect

For each chosen head, exact local activation*gradient attribution is computed
for every B20 hidden unit on ordinary patch queries. Positive attribution means
that unit locally increases the current winner probability (sharpener); negative
means it decreases it (flattener).

The analysis then forms:
  * Hsharp sharpeners / Hsharp flatteners
  * Hflat sharpeners / Hflat flatteners
  * push-pull units: Hsharp sharpener AND Hflat flattener
  * reverse push-pull: Hsharp flattener AND Hflat sharpener

Groups are defined only within a configurable previously-discovered B20 family
(default: top220_posthoc). The old ~220 population therefore constrains the
follow-up scope but does not determine sign or subfamily membership.

Exact group ablations validate causal sign on all B21 heads and final embeddings.
Aggregate group write vectors and contributions into selected residual axes are
also exported. No pickle outputs are written.
"""
from __future__ import annotations

import argparse, contextlib, gc, hashlib, json, math, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from safetensors.torch import save_file as _save_st
except Exception as e:
    raise RuntimeError("safetensors is required") from e

EPS = 1e-12
FORMAT_VERSION = 1
DEFAULT_AXES = [499,250,908,953,779,196,350,139,468,469,951,211,1021,866,151,720,656,400,565,650]


def amp_context(device: str, enabled: bool):
    if enabled and str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def save_st(tensors: Dict[str, torch.Tensor], filename: Path, metadata: Optional[Dict[str, Any]] = None):
    packed = {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}
    filename.parent.mkdir(parents=True, exist_ok=True)
    _save_st(packed, str(filename), metadata={str(k): str(v) for k, v in (metadata or {}).items()})


def parse_ints(s): return [int(x.strip()) for x in str(s).split(",") if x.strip()]
def parse_strs(s): return [x.strip() for x in str(s).split(",") if x.strip()]


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [Path.cwd(), here.parent, *here.parents]:
        if (p/"attnclip_mechinterp_sae").is_dir() and (p/"attnclip_mechinterp_xattn").is_dir() and (p/"utils_clip_loader").is_dir():
            return p
    raise FileNotFoundError("Could not locate repo root; pass --repo_root")


def _import_runtime(args):
    repo = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    import attnclip_mechinterp_sae as clip_sae
    import attnclip_mechinterp_xattn as clip_x
    from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything, resolve_to_openai_state_dict
    return repo, clip_sae, clip_x, load_openai_clip_anything, resolve_to_openai_state_dict


def _freeze(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def _resolve_xattn_state(clip_sae, resolve_fn, args):
    sd, info = resolve_fn(args.xattn_model, cache_dir=(args.hf_cache_dir or None), revision=(args.xattn_revision or None), allow_unsafe_hf_pickle=False)
    conv = getattr(getattr(clip_sae, "model", None), "convert_state_dict_inproj_to_qkv", None)
    if callable(conv): sd = conv(sd)
    return sd, info


def _load_gmp(clip_sae, load_any, args, device):
    try:
        m, p, li = load_any(clip_sae, args.gmp_checkpoint, device=device, jit=False, strict=True, reuse_full_model_pickle=False)
        return _freeze(m), p, {"source": args.gmp_checkpoint, "mode": "state_dict_rebuild", "loader": str(li)}
    except Exception as first_error:
        src, _pp, li = load_any(clip_sae, args.gmp_checkpoint, device="cpu", jit=False, strict=True, reuse_full_model_pickle=True)
        m, p, _ = load_any(clip_sae, args.pretrained_model, device=device, jit=False, strict=True, allow_unsafe_hf_pickle=False)
        srcsd = src.state_dict(); conv = getattr(getattr(clip_sae, "model", None), "convert_state_dict_inproj_to_qkv", None)
        if callable(conv): srcsd = conv(srcsd)
        tgt = m.state_dict(); filt = {k: v.to(dtype=tgt[k].dtype) for k, v in srcsd.items() if k.startswith("visual.") and k in tgt and tuple(v.shape) == tuple(tgt[k].shape)}
        miss = sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
        if miss: raise RuntimeError(f"GmP fallback missing visual keys: {miss[:12]}") from first_error
        m.load_state_dict(filt, strict=False); del src; gc.collect()
        return _freeze(m), p, {"source": args.gmp_checkpoint, "mode": "trusted_pickle_visual_transplant", "loader": str(li), "first_error": repr(first_error)}


def _load_bare_xattn(clip_sae, load_any, resolve_fn, args, device):
    m, p, _ = load_any(clip_sae, args.pretrained_model, device=device, jit=False, strict=True, allow_unsafe_hf_pickle=False); sd, info = _resolve_xattn_state(clip_sae, resolve_fn, args); tgt = m.state_dict(); filt = {}; ignored = []
    for k, v in sd.items():
        if not k.startswith("visual.") or k in {"visual.read_null_token", "visual.read_null_insert_block_config"} or k not in tgt or tuple(v.shape) != tuple(tgt[k].shape):
            ignored.append(k); continue
        filt[k] = v.to(dtype=tgt[k].dtype)
    miss = sorted(k for k in tgt if k.startswith("visual.") and k not in filt)
    if miss: raise RuntimeError(f"bare_xattn missing visual keys: {miss[:12]}")
    inc = m.load_state_dict(filt, strict=False)
    return _freeze(m), p, {"source": args.xattn_model, "mode": "bare_xattn", "loaded_visual_keys": len(filt), "ignored_key_count": len(ignored), "load_missing": list(inc.missing_keys), "load_unexpected": list(inc.unexpected_keys), "loader": str(info)}


@dataclass
class Variant:
    name: str; model: Any; preprocess: Any; source_info: Dict[str, Any]; is_full_xattn: bool = False
    @property
    def visual(self): return self.model.visual
    def maybe_insert_rn(self, block_idx: int, x: torch.Tensor) -> torch.Tensor:
        if self.is_full_xattn: return self.visual._maybe_insert_read_null(block_idx, x)
        return x


def load_variant(name, args) -> Variant:
    repo, clip_sae, clip_x, load_any, resolve_fn = _import_runtime(args)
    if name == "pretrained":
        m, p, li = load_any(clip_sae, args.pretrained_model, device=args.device, jit=False, strict=True, allow_unsafe_hf_pickle=False); return Variant(name, _freeze(m), p, {"source": args.pretrained_model, "loader": str(li)}, False)
    if name == "gmp":
        m, p, i = _load_gmp(clip_sae, load_any, args, args.device); return Variant(name, m, p, i, False)
    if name == "bare_xattn":
        m, p, i = _load_bare_xattn(clip_sae, load_any, resolve_fn, args, args.device); return Variant(name, m, p, i, False)
    if name == "full_xattn":
        m, p, li = load_any(clip_x, args.xattn_model, device=args.device, jit=False, cache_dir=(args.hf_cache_dir or None), revision=(args.xattn_revision or None), strict=True, allow_unsafe_hf_pickle=False)
        return Variant(name, _freeze(m), p, {"source": args.xattn_model, "mode": "full_xattn", "loader": str(li)}, True)
    raise ValueError(name)


def load_manifest(discovery_root: Path, image_dir: str, max_images: int):
    p = discovery_root/"image_manifest.csv"
    if p.exists():
        d = pd.read_csv(p)
        if max_images > 0: d = d.iloc[:max_images].copy()
        return d
    root = Path(image_dir); exts = {".png",".jpg",".jpeg",".webp",".bmp"}; fs = sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in exts)
    if max_images > 0: fs = fs[:max_images]
    if not fs: raise FileNotFoundError(f"No images under {root} and no discovery manifest at {p}")
    return pd.DataFrame({"stim_id": [x.stem for x in fs], "path": [str(x) for x in fs]})


def load_batch(preprocess, rows, device):
    from PIL import Image
    xs = []
    for p in rows.path:
        with Image.open(p) as im: xs.append(preprocess(im.convert("RGB")))
    return torch.stack(xs, 0).to(device, non_blocking=True)


def resolve_c_proj(blk):
    cp = getattr(blk.mlp, "c_proj", None)
    if cp is not None and hasattr(cp, "weight") and getattr(cp, "weight").ndim == 2: return cp
    weighted = [m for m in blk.mlp.modules() if m is not blk.mlp and hasattr(m, "weight") and isinstance(getattr(m, "weight"), torch.Tensor) and getattr(m, "weight").ndim == 2]
    if len(weighted) < 2: raise RuntimeError("Could not identify MLP c_proj")
    return weighted[-1]


def frozen_register_mask(pre13_tbc, P, threshold, max_registers, min_registers):
    n = pre13_tbc[1:1+P].float().norm(dim=-1).T; mask = torch.zeros_like(n, dtype=torch.bool)
    for i in range(n.shape[0]):
        idx = torch.nonzero(n[i] >= threshold, as_tuple=False).flatten()
        if idx.numel() < min_registers: idx = torch.topk(n[i], k=min(min_registers, P)).indices
        if max_registers > 0 and idx.numel() > max_registers:
            vals = n[i, idx]; idx = idx[torch.topk(vals, k=max_registers).indices]
        mask[i, idx] = True
    return mask, n


def attention_metrics_all_heads(probs: torch.Tensor, regmask: torch.Tensor, P: int):
    B,H,T,S = probs.shape; p = probs.float(); ent = -(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum(-1)/math.log(max(S,2)); top1 = p.max(-1).values; top2 = torch.topk(p,k=min(2,S),dim=-1).values; margin = top2[...,0]-(top2[...,1] if S>1 else 0.0)
    out = []
    om = (~regmask).to(p.device)
    for h in range(H):
        vals = {}
        for name,t in [("top1",top1[:,h,1:1+P]),("entropy",ent[:,h,1:1+P]),("margin",margin[:,h,1:1+P])]:
            vv = t[om]; vals[name] = float(vv.mean().item()) if vv.numel() else float("nan")
        out.append(vals)
    return out


def forward_to_b20(variant: Variant, images, args):
    v = variant.visual; blocks = list(v.transformer.resblocks); images = images.to(dtype=v.conv1.weight.dtype); x = v._prepare_tokens(images); P = x.shape[0]-1; regmask = None
    for li, blk in enumerate(blocks[:args.block+1]):
        x = variant.maybe_insert_rn(li, x)
        if li == args.register_block: regmask,_ = frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
        ln1 = blk.ln_1(x)
        with amp_context(args.device,args.amp): attn_out,_ = blk.attention(ln1,need_weights=False,capture=False)
        pa = x + attn_out
        if li == args.block:
            cp = resolve_c_proj(blk); box = {}
            def hook(mod, inp): box["h"] = inp[0]
            hh = cp.register_forward_pre_hook(hook)
            try:
                with amp_context(args.device,args.amp): mlp = blk.mlp(blk.ln_2(pa))
            finally: hh.remove()
            if regmask is None: regmask,_ = frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
            return pa, box["h"], mlp, regmask, P
        with amp_context(args.device,args.amp): mlp = blk.mlp(blk.ln_2(pa))
        x = pa + mlp
    raise RuntimeError("B20 not reached")


def b21_probs(variant, x20, args):
    blk = variant.visual.transformer.resblocks[args.block+1]; ln = blk.ln_1(x20)
    with amp_context(args.device,args.amp): _out,p = blk.attention(ln,need_weights=True,capture=False)
    return p


def choose_heads(head_audit: pd.DataFrame, sharp_override: int, flat_override: int):
    if sharp_override >= 0: sharp = int(sharp_override)
    else: sharp = int(head_audit.loc[head_audit["b21_delta_top1_from_mlp20"].idxmax(),"head"])
    if flat_override >= 0: flat = int(flat_override)
    else: flat = int(head_audit.loc[head_audit["b21_delta_top1_from_mlp20"].idxmin(),"head"])
    return sharp, flat


def signed_attr_pass(variant, manifest, args, out: Path, heads: Sequence[int], M: int):
    cdir = out/"signed_attribution_batches"; cdir.mkdir(parents=True,exist_ok=True); cp = resolve_c_proj(variant.visual.transformer.resblocks[args.block])
    totals = {h: np.zeros(M,np.float64) for h in heads}; abs_totals = {h: np.zeros(M,np.float64) for h in heads}; counts = {h:0 for h in heads}
    for st in range(0,len(manifest),args.batch_size):
        rows = manifest.iloc[st:st+args.batch_size]; cache = cdir/f"batch_{st:06d}.safetensors"
        # Cache intentionally stores every requested head together; signature sidecar prevents stale reuse.
        sig = {"version":FORMAT_VERSION,"model":variant.name,"heads":list(map(int,heads)),"start":int(st),"n":len(rows),"block":args.block}
        sp = cdir/f"batch_{st:06d}.json"
        if cache.exists() and sp.exists() and json.loads(sp.read_text()) == sig:
            from safetensors.torch import load_file
            z = load_file(str(cache))
            for h in heads:
                totals[h] += z[f"h{h}_signed"].double().numpy(); abs_totals[h] += z[f"h{h}_abs"].double().numpy(); counts[h] += int(z[f"h{h}_count"].item())
            print(f"[{variant.name} signed attr] resume {min(st+len(rows),len(manifest))}/{len(manifest)}"); continue
        images = load_batch(variant.preprocess, rows, args.device)
        with torch.no_grad(): pa,h0,_mlp,mask,P = forward_to_b20(variant,images,args)
        h = h0.detach().clone().requires_grad_(True)
        with amp_context(args.device,args.amp): mlp = cp(h); x20 = pa.detach()+mlp; p21 = b21_probs(variant,x20,args)
        om = (~mask).to(p21.device); payload = {}
        for hi,hidx in enumerate(heads):
            probs = p21[:,hidx,1:1+P,:]
            winner = probs.detach().argmax(-1); pwin = probs.gather(-1,winner.unsqueeze(-1)).squeeze(-1); scalar = pwin[om].mean()
            grad = torch.autograd.grad(scalar,h,retain_graph=(hi < len(heads)-1),create_graph=False)[0]
            attr = (h*grad)[1:1+P].permute(1,0,2); vals = attr[om].detach().float().cpu()
            signed = vals.double().sum(0); absv = vals.abs().double().sum(0); n = vals.shape[0]
            payload[f"h{hidx}_signed"] = signed; payload[f"h{hidx}_abs"] = absv; payload[f"h{hidx}_count"] = torch.tensor(n); payload[f"h{hidx}_scalar"] = scalar.detach().cpu().float()
            totals[hidx] += signed.numpy(); abs_totals[hidx] += absv.numpy(); counts[hidx] += n
        save_st(payload,cache,{"model":variant.name,"heads":",".join(map(str,heads)),"start":st}); sp.write_text(json.dumps(sig,indent=2))
        print(f"[{variant.name} signed attr] {min(st+len(rows),len(manifest))}/{len(manifest)}")
        del images,h,p21; gc.collect();
        if str(args.device).startswith("cuda"): torch.cuda.empty_cache()
    d = {"neuron":np.arange(M)}
    for h in heads:
        n = max(counts[h],1); d[f"h{h}_signed"] = totals[h]/n; d[f"h{h}_abs"] = abs_totals[h]/n
    return pd.DataFrame(d)


def sign_core(ids: np.ndarray, vals: np.ndarray, positive: bool, frac: float):
    mask = vals > 0 if positive else vals < 0
    ii = ids[mask]; vv = np.abs(vals[mask])
    if len(ii)==0: return []
    order = np.argsort(-vv); ii=ii[order]; vv=vv[order]; cum=np.cumsum(vv)/(vv.sum()+EPS)
    keep = ii[cum <= frac].tolist()
    if not keep: keep=[int(ii[0])]
    # include the first item that crosses frac for a true >= coverage set
    if len(keep)<len(ii): keep.append(int(ii[len(keep)]))
    return list(map(int,keep))


def build_groups(candidates: pd.DataFrame, attrs: pd.DataFrame, sharp: int, flat: int, args):
    df = candidates.merge(attrs,on="neuron",how="left")
    if args.family_scope == "top220": fam = df["top220_posthoc"].astype(bool).to_numpy()
    elif args.family_scope == "consensus": fam = df["consensus_family"].astype(bool).to_numpy()
    elif args.family_scope == "extended95": fam = df["sharpness_extended95"].astype(bool).to_numpy()
    elif args.family_scope == "all": fam = np.ones(len(df),dtype=bool)
    else: raise ValueError(args.family_scope)
    ids = df.loc[fam,"neuron"].astype(int).to_numpy(); sv = df.loc[fam,f"h{sharp}_signed"].to_numpy(float); fv = df.loc[fam,f"h{flat}_signed"].to_numpy(float)
    groups = {
        "sharp_head_sharpeners_core80": sign_core(ids,sv,True,.80),
        "sharp_head_flatteners_core80": sign_core(ids,sv,False,.80),
        "flat_head_sharpeners_core80": sign_core(ids,fv,True,.80),
        "flat_head_flatteners_core80": sign_core(ids,fv,False,.80),
    }
    # Cross-head push-pull: rank by the weaker of the two normalized signed magnitudes.
    sscale=np.median(np.abs(sv))+EPS; fscale=np.median(np.abs(fv))+EPS
    push_mask=(sv>0)&(fv<0); rev_mask=(sv<0)&(fv>0)
    push_score=np.minimum(sv/sscale,-fv/fscale); rev_score=np.minimum(-sv/sscale,fv/fscale)
    for name,mask,score in [("pushpull_sharpH_up_flatH_down",push_mask,push_score),("reverse_sharpH_down_flatH_up",rev_mask,rev_score)]:
        jj=ids[mask]; ss=score[mask]; order=np.argsort(-ss); groups[name]=list(map(int,jj[order]))
    # Add per-neuron labels and ranks.
    df["in_family_scope"] = fam
    df["sharp_head_role"] = np.where(df[f"h{sharp}_signed"]>0,"sharpener",np.where(df[f"h{sharp}_signed"]<0,"flattener","neutral"))
    df["flat_head_role"] = np.where(df[f"h{flat}_signed"]>0,"sharpener",np.where(df[f"h{flat}_signed"]<0,"flattener","neutral"))
    df["pushpull_role"] = np.where((df[f"h{sharp}_signed"]>0)&(df[f"h{flat}_signed"]<0),"sharpH_up_flatH_down",np.where((df[f"h{sharp}_signed"]<0)&(df[f"h{flat}_signed"]>0),"sharpH_down_flatH_up","other"))
    df["abs_dual_signed"] = np.abs(df[f"h{sharp}_signed"])+np.abs(df[f"h{flat}_signed"])
    return df,groups


def apply_hidden_ablation(cp, idx: Sequence[int]):
    ids = torch.as_tensor(list(idx),dtype=torch.long)
    def hook(mod,inp):
        h=inp[0].clone(); h.index_fill_(-1,ids.to(h.device),0); return (h,)
    return cp.register_forward_pre_hook(hook)


def full_forward_ablation_all_heads(variant, images, args, idx: Sequence[int]):
    v=variant.visual; blocks=list(v.transformer.resblocks); images=images.to(dtype=v.conv1.weight.dtype); x=v._prepare_tokens(images); P=x.shape[0]-1; regmask=None; p21_saved=None
    for li,blk in enumerate(blocks):
        x=variant.maybe_insert_rn(li,x)
        if li==args.register_block: regmask,_=frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
        with amp_context(args.device,args.amp): attn,probs=blk.attention(blk.ln_1(x),need_weights=(li==args.block+1),capture=False)
        pa=x+attn; hh=None
        if li==args.block and idx: hh=apply_hidden_ablation(resolve_c_proj(blk),idx)
        try:
            with amp_context(args.device,args.amp): mlp=blk.mlp(blk.ln_2(pa))
        finally:
            if hh is not None: hh.remove()
        x=pa+mlp
        if li==args.block+1: p21_saved=probs.detach()
    with amp_context(args.device,args.amp): emb=v._finalize_cls(x)
    emb=F.normalize(emb.float(),dim=-1)
    if regmask is None: regmask,_=frozen_register_mask(x,P,args.register_threshold,args.max_registers,args.min_registers)
    metrics=attention_metrics_all_heads(p21_saved,regmask,P)
    return emb,metrics


def group_jobs(groups,args):
    jobs=[]
    for g,ids in groups.items():
        if not ids: continue
        sizes=args.group_sizes if "pushpull" in g or "reverse" in g else []
        if sizes:
            for k in sizes:
                if k>0 and len(ids)>=1: jobs.append((f"{g}_top{k}",ids[:min(k,len(ids))]))
        jobs.append((f"{g}_all",ids))
    return jobs


def ablation_validation(variant,manifest,args,out:Path,groups,sharp,flat):
    ad=out/"group_ablation_cache";ad.mkdir(parents=True,exist_ok=True); bp=ad/"baseline.safetensors"; bm=ad/"baseline.json"
    from safetensors.torch import load_file
    if bp.exists() and bm.exists():
        base=load_file(str(bp))["embeddings"].float(); base_meta=json.loads(bm.read_text()); print(f"[{variant.name} groups] resume baseline")
    else:
        em=[]; hs=[]
        for st in range(0,len(manifest),args.batch_size):
            ims=load_batch(variant.preprocess,manifest.iloc[st:st+args.batch_size],args.device)
            with torch.inference_mode(): z,m=full_forward_ablation_all_heads(variant,ims,args,[])
            em.append(z.cpu()); hs.append(m); del ims
        base=torch.cat(em,0); H=len(hs[0]); meanm=[{k:float(np.mean([b[h][k] for b in hs])) for k in ["top1","entropy","margin"]} for h in range(H)]
        save_st({"embeddings":base},bp,{"model":variant.name}); base_meta={"heads":meanm}; bm.write_text(json.dumps(base_meta,indent=2))
    rows=[]; headrows=[]
    rng=np.random.default_rng(int(hashlib.sha256(variant.name.encode()).hexdigest()[:16],16)); family_pool=sorted(set(sum([v for v in groups.values()],[])))
    for job,ids in group_jobs(groups,args):
        for kind,use in [("group",ids),("random",None)]:
            key=f"{job}__{kind}"; jp=ad/f"{key}.json"
            if jp.exists(): rec=json.loads(jp.read_text()); rows.append(rec["summary"]); headrows.extend(rec["heads"]); continue
            if kind=="random":
                pool=np.array([i for i in range(4096) if i not in set(ids)],dtype=int); use=rng.choice(pool,size=len(ids),replace=False).tolist()
            dists=[]; hm=[]; off=0
            for st in range(0,len(manifest),args.batch_size):
                ims=load_batch(variant.preprocess,manifest.iloc[st:st+args.batch_size],args.device)
                with torch.inference_mode(): z,m=full_forward_ablation_all_heads(variant,ims,args,use)
                bb=base[off:off+len(z)].to(z.device);off+=len(z);dists.extend((1-(bb*z).sum(-1)).cpu().tolist());hm.append(m);del ims
            H=len(hm[0]); meanm=[{k:float(np.mean([b[h][k] for b in hm])) for k in ["top1","entropy","margin"]} for h in range(H)]
            summary={"model":variant.name,"job":job,"kind":kind,"n_neurons":len(use),"mean_final_cosine_distance":float(np.mean(dists)),"neuron_ids":",".join(map(str,use)),"sharp_head":sharp,"flat_head":flat,"delta_sharp_head_top1":meanm[sharp]["top1"]-base_meta["heads"][sharp]["top1"],"delta_flat_head_top1":meanm[flat]["top1"]-base_meta["heads"][flat]["top1"],"delta_sharp_head_entropy":meanm[sharp]["entropy"]-base_meta["heads"][sharp]["entropy"],"delta_flat_head_entropy":meanm[flat]["entropy"]-base_meta["heads"][flat]["entropy"]}
            hrows=[]
            for h in range(H):
                rr={"model":variant.name,"job":job,"kind":kind,"head":h,"n_neurons":len(use)}
                for k in ["top1","entropy","margin"]: rr[f"baseline_{k}"]=base_meta["heads"][h][k];rr[f"ablated_{k}"]=meanm[h][k];rr[f"delta_{k}"]=meanm[h][k]-base_meta["heads"][h][k]
                hrows.append(rr)
            jp.write_text(json.dumps({"summary":summary,"heads":hrows},indent=2)); rows.append(summary);headrows.extend(hrows);print(f"[{variant.name} groups] {job} {kind} n={len(use)}")
    pd.DataFrame(rows).to_csv(out/"group_ablation_summary.csv",index=False);pd.DataFrame(headrows).to_csv(out/"group_ablation_head_metrics.csv",index=False)
    return pd.DataFrame(rows),pd.DataFrame(headrows)


def aggregate_group_vectors(variant,candidates:pd.DataFrame,groups,args,out:Path):
    cp=resolve_c_proj(variant.visual.transformer.resblocks[args.block]);W=cp.weight.detach().float().cpu().numpy(); meanact=candidates.set_index("neuron")["ord_mean_act"]
    tensors={}; rows=[]
    for g,ids in groups.items():
        if not ids: continue
        ids2=[i for i in ids if i in meanact.index]
        if not ids2: continue
        a=meanact.loc[ids2].to_numpy(float); vec=W[:,ids2]@a; tensors[g]=torch.from_numpy(vec.astype(np.float32))
        norm=float(np.linalg.norm(vec))
        for ax in args.target_axes:
            if 0<=ax<len(vec): rows.append({"model":variant.name,"group":g,"n_neurons":len(ids2),"axis":ax,"group_write_axis_value":float(vec[ax]),"group_write_axis_fraction_abs":float(abs(vec[ax])/(np.abs(vec).sum()+EPS)),"group_write_norm":norm})
    if tensors: save_st(tensors,out/"group_write_vectors.safetensors",{"model":variant.name,"block":args.block})
    pd.DataFrame(rows).to_csv(out/"group_special_axis_contributions.csv",index=False)


def plot_outputs(df,heads,abl,out:Path,sharp,flat):
    try: import matplotlib.pyplot as plt
    except Exception: return
    p=out/"plots";p.mkdir(exist_ok=True)
    fam=df[df["in_family_scope"]].copy()
    fig,ax=plt.subplots(figsize=(9,8));ax.scatter(fam[f"h{sharp}_signed"],fam[f"h{flat}_signed"],s=16,alpha=.55)
    ax.axhline(0,lw=1);ax.axvline(0,lw=1);top=fam.nlargest(28,"abs_dual_signed")
    for _,r in top.iterrows():ax.annotate(str(int(r.neuron)),(r[f"h{sharp}_signed"],r[f"h{flat}_signed"]),fontsize=7)
    ax.set_xlabel(f"signed act×grad into B21 H{sharp} winner p");ax.set_ylabel(f"signed act×grad into B21 H{flat} winner p");ax.set_title("B20 sharpener / flattener subfamilies");ax.grid(alpha=.15);fig.tight_layout();fig.savefig(p/"01_SIGNED_SUBFAMILY_QUADRANTS.png",dpi=200);plt.close(fig)
    if not abl.empty:
        g=abl[abl["kind"].eq("group")].copy();x=np.arange(len(g));fig,ax=plt.subplots(figsize=(12,7));w=.38;ax.bar(x-w/2,g["delta_sharp_head_top1"],width=w,label=f"Δ H{sharp} top1");ax.bar(x+w/2,g["delta_flat_head_top1"],width=w,label=f"Δ H{flat} top1");ax.axhline(0,lw=1);ax.set_xticks(x);ax.set_xticklabels(g["job"],rotation=45,ha="right");ax.set_ylabel("change after B20 neuron ablation");ax.set_title("Causal sign validation of B20 subfamilies");ax.legend();fig.tight_layout();fig.savefig(p/"02_GROUP_ABLATION_TARGET_HEADS.png",dpi=200);plt.close(fig)
    h=heads.copy().sort_values("head");fig,ax=plt.subplots(figsize=(10,6));ax.bar(h["head"],h["b21_delta_top1_from_mlp20"]);ax.axhline(0,lw=1);ax.axvline(sharp,ls="--",alpha=.5);ax.axvline(flat,ls="--",alpha=.5);ax.set_xlabel("B21 head");ax.set_ylabel("B20 MLP-induced Δ top1");ax.set_title(f"Auto targets: H{sharp} sharpener, H{flat} flattener");fig.tight_layout();fig.savefig(p/"03_TARGET_HEAD_SELECTION.png",dpi=200);plt.close(fig)


def write_report(df,groups,heads,abl,out:Path,variant,sharp,flat,args):
    sharp_delta = float(
        heads.loc[heads["head"].eq(sharp), "b21_delta_top1_from_mlp20"].iloc[0]
    )
    flat_delta = float(
        heads.loc[heads["head"].eq(flat), "b21_delta_top1_from_mlp20"].iloc[0]
    )
    lines=[
        "B20 SHARPENER / FLATTENER SUBFAMILIES",
        "="*42,
        "",
        f"model: {variant.name}",
        f"family scope: {args.family_scope}",
        f"auto/selected sharp head: H{sharp}",
        f"auto/selected flat head: H{flat}",
        "",
        f"B20 MLP Δtop1 H{sharp}: {sharp_delta:+.6f}",
        f"B20 MLP Δtop1 H{flat}: {flat_delta:+.6f}",
        "",
    ]
    for g,ids in groups.items(): lines += [f"{g}: n={len(ids)}", ",".join(map(str,ids[:256])),""]
    if not abl.empty:
        lines += ["CAUSAL ABLATIONS","-"]
        for _,r in abl[abl["kind"].eq("group")].iterrows():
            lines.append(f"{str(r['job']):48s} n={int(r['n_neurons']):3d}  ΔH{sharp}={r['delta_sharp_head_top1']:+.6f}  ΔH{flat}={r['delta_flat_head_top1']:+.6f}  final_d={r['mean_final_cosine_distance']:.6f}")
    (out/"B20_SHARPENERS_FLATTENERS.txt").write_text("\n".join(lines),encoding="utf-8")
    (out/"subfamily_indices.json").write_text(json.dumps({"model":variant.name,"family_scope":args.family_scope,"sharp_head":sharp,"flat_head":flat,"groups":groups},indent=2))


def combine(models,out:Path):
    tabs=[];groups={}
    for m in models:
        p=out/m/"b20_signed_subfamilies.csv.gz";j=out/m/"subfamily_indices.json"
        if p.exists():
            d=pd.read_csv(p);tabs.append(d[d["in_family_scope"]].nlargest(256,"abs_dual_signed"))
        if j.exists():groups[m]=json.loads(j.read_text())
    if tabs:pd.concat(tabs,ignore_index=True).to_csv(out/"ALL_MODELS_B20_SIGNED_TOP.csv",index=False)
    rows=[]
    names=sorted(groups)
    keys=sorted(set(k for m in names for k in groups[m]["groups"]))
    for key in keys:
        for i,a in enumerate(names):
            A=set(groups[a]["groups"].get(key,[]))
            for b in names[i+1:]:
                B=set(groups[b]["groups"].get(key,[])); inter=A&B
                rows.append({"group":key,"model_a":a,"model_b":b,"n_a":len(A),"n_b":len(B),"intersection":len(inter),"jaccard":len(inter)/max(len(A|B),1),"shared_neurons":",".join(map(str,sorted(inter)))})
    if rows:pd.DataFrame(rows).to_csv(out/"ALL_MODELS_B20_SIGNED_GROUP_OVERLAP.csv",index=False)


def self_test():
    ids=np.arange(6);v=np.array([5,3,1,-4,-2,-1],float);pos=sign_core(ids,v,True,.8);neg=sign_core(ids,v,False,.8);assert 0 in pos and 3 in neg
    t=torch.randn(4,8);tmp=Path("_b20sf_test.safetensors");save_st({"a":t,"b":t[0]},tmp);tmp.unlink()
    # Regression: DataFrame.head is a method; a column literally named "head" must use [] access.
    hdf=pd.DataFrame({"head":[8,12],"b21_delta_top1_from_mlp20":[-.2,.1]})
    assert float(hdf.loc[hdf["head"].eq(12),"b21_delta_top1_from_mlp20"].iloc[0]) == .1
    print("self-test passed")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo_root",default="");p.add_argument("--discovery_root",default="out_paper_reproduction/conv1/b20_writeback_neurons");p.add_argument("--output_dir",default="out_paper_reproduction/conv1/b20_sharpeners_flatteners")
    p.add_argument("--image_dir",default="image_sets/special_natural");p.add_argument("--models",default="pretrained,gmp,bare_xattn,full_xattn");p.add_argument("--pretrained_model",default="openai/clip-vit-large-patch14");p.add_argument("--gmp_checkpoint",default="zer0int/CLIP-GmP-ViT-L-14");p.add_argument("--xattn_model",default="zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX");p.add_argument("--xattn_revision",default="");p.add_argument("--hf_cache_dir",default="")
    p.add_argument("--block",type=int,default=20);p.add_argument("--register_block",type=int,default=13);p.add_argument("--register_threshold",type=float,default=70.0);p.add_argument("--max_registers",type=int,default=4);p.add_argument("--min_registers",type=int,default=1)
    p.add_argument("--family_scope",choices=["top220","consensus","extended95","all"],default="top220");p.add_argument("--sharp_head",type=int,default=-1);p.add_argument("--flat_head",type=int,default=-1);p.add_argument("--group_sizes",default="16,32,64")
    p.add_argument("--target_axes",default=",".join(map(str,DEFAULT_AXES)));p.add_argument("--batch_size",type=int,default=4);p.add_argument("--max_images",type=int,default=0);p.add_argument("--device",default="cuda");p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);p.add_argument("--skip_ablation",action="store_true");p.add_argument("--self_test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test();return 0
    args.models=parse_strs(args.models);args.target_axes=parse_ints(args.target_axes);args.group_sizes=parse_ints(args.group_sizes)
    discovery=Path(args.discovery_root);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);manifest=load_manifest(discovery,args.image_dir,args.max_images);manifest.to_csv(out/"image_manifest.csv",index=False)
    for name in args.models:
        print(f"\n=== MODEL {name} ===")
        cpath=discovery/name/"b20_neuron_candidates.csv.gz";hpath=discovery/name/"head_audit.csv"
        if not cpath.exists() or not hpath.exists(): raise FileNotFoundError(f"Missing prior B20 discovery outputs for {name}: {cpath} / {hpath}")
        cand=pd.read_csv(cpath);heads=pd.read_csv(hpath);sharp,flat=choose_heads(heads,args.sharp_head,args.flat_head);print(f"[{name}] sharp head H{sharp}; flat head H{flat}")
        variant=load_variant(name,args);md=out/name;md.mkdir(parents=True,exist_ok=True);attrs=signed_attr_pass(variant,manifest,args,md,[sharp,flat],len(cand));df,groups=build_groups(cand,attrs,sharp,flat,args);df.to_csv(md/"b20_signed_subfamilies.csv.gz",index=False,compression="gzip");aggregate_group_vectors(variant,df,groups,args,md)
        abl=pd.DataFrame();
        if not args.skip_ablation: abl,_=ablation_validation(variant,manifest,args,md,groups,sharp,flat)
        plot_outputs(df,heads,abl,md,sharp,flat);write_report(df,groups,heads,abl,md,variant,sharp,flat,args);(md/"audit.json").write_text(json.dumps({"format_version":FORMAT_VERSION,"model":name,"source_info":variant.source_info,"discovery_root":str(discovery),"family_scope":args.family_scope,"sharp_head":sharp,"flat_head":flat,"n_images":len(manifest)},indent=2));del variant;gc.collect();
        if str(args.device).startswith("cuda"):torch.cuda.empty_cache()
    combine(args.models,out);print(f"\nDone -> {out}");return 0

if __name__=="__main__":raise SystemExit(main())
