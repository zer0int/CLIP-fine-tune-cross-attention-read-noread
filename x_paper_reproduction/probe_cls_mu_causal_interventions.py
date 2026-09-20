#!/usr/bin/env python3
r"""CLS <-> mu causal 2x2, RN repeat, and RN/CLS slot-swap probe (v3 phase-separated)
=============================================================

Purpose
-------
Test the causal loop suggested by the no-RN CLS/GIPU measurements:

    early CLS spatial read  ->  CLS state
                             ->  CLS-as-value broadcast
                             ->  patchwise mu1 write
                             ->  register precursor / lineage organization
                             ->  B11/B12 register isolation
                             ->  B12/B13 register interrogation

The experiment is deliberately surgical.  It does NOT run the PIECES bridge,
router, correction, or text-conditioned READ.  It operates on the ordinary ViT
self-attention computation only.

Models
------
The same three capture-enabled ordinary visual towers as the NOP/BROADCAST and
CLS/GIPU runs:

    pretrained
    gmp
    finetune_stripped

The trained x-attn checkpoint contributes ONLY:
    * ordinary visual weights for finetune_stripped (through the supplied base loader)
    * the learned visual.read_null_token vector for the RN conditions

No bridge weights are executed.

Main experiment: phase-separated CLS 2x2
-------------------------------------------
The two factors deliberately target the chronology suggested by the maps:

    Qread blocks default to B0--B5
        the positional/global scan phase.

    Vbroadcast blocks default to B6--B10
        the broad mu-puff / handoff phase.

Both windows are CLI-configurable and may overlap if desired.

Two binary factors:

    Qread ON
        CLS query row is untouched.

    Qread OFF
        on --qread_blocks, for Q=CLS all spatial source columns are zeroed
        and the row is renormalized onto CLS itself.  Thus the early CLS state
        cannot be built by reading patches during the scan window.

    Vbroadcast ON
        spatial query rows may attend to CLS normally.

    Vbroadcast OFF
        on --vbroadcast_blocks, for every spatial query row the CLS source
        column is zeroed and the row is renormalized over spatial sources.
        Thus patches cannot consume CLS as K/V source during the mu-puff window.

The four combinations are run once through B6--B12.  Each pre-B13 state is then
forked into:

    no_rn
        ordinary vanilla ViT continuation

    rn
        append the exact trained x-attn RN vector immediately before B13,
        as in the native model, then continue B13--B23

Therefore the RN and no-RN 2x2 share an IDENTICAL B0--B12 computation for each
early intervention arm.  We do not redundantly recompute it.

Third experiment: slot swap / "math sphere sacrilege"
------------------------------------------------------
Only for Qread=ON, Vbroadcast=ON:

At the normal pre-B13 RN insertion point:

    old_cls = x[0]
    token 0 <- learned RN vector
    append old_cls as the final token (the position RN normally occupies)

Then leave the token order untouched for B13--B23.

The standard image embedding is deliberately produced from token 0 after B23:
i.e. from the mutated RN trajectory through ln_post + visual.proj.

For diagnosis, the appended copied-CLS token is ALSO passed through the same
ln_post + visual.proj and compared with the normal image embedding.  This tells
us whether semantic state merely moved to the "wrong" slot even if the official
slot-0 image embedding is wrecked.

Fixed coordinate system / no moving goalposts
----------------------------------------------
For every model, mu1/mu2 and register-lineage addresses are defined by the
ordinary no-RN Qread=ON/Vbroadcast=ON baseline and then held FIXED for every
intervention, RN condition, and slot swap.

By default the script reuses the already-computed oracle from:

    cls_gipu_exchange_no_rn/<model>/mu_basis.npz

when its stim_id order and dimensions exactly match the current manifest.
If unavailable, a baseline oracle pass is recomputed.

Measurements
------------
1) Exact B6--B10 attention-stage mu decomposition
   The explicit attention algebra decomposes each patch's signed mu1/mu2 update
   into the CLS source and fixed spatial source groups.

   Main causal quantities:
       patch |Delta mu1|
       CLS-source |mu1| / exact patch |Delta mu1|
       fixed register-lineage source ratio
       corr_spatial(total Delta mu1, CLS-source mu1)

2) CLS state
       pre/post-attention cosine and coefficient on fixed mu1/mu2

3) Early MLP register-precursor population signature
   For B6--B10 we accumulate the 4096-D post-QuickGELU mean at:
       fixed future-register-lineage positions
       ordinary positions
   The baseline difference vector is the model-local precursor signature.
   Every intervention is compared to that SAME baseline signature:
       cosine
       norm ratio
       projection amplitude
   This is population-level and data-driven; it does not assume a hand-picked
   early neuron list.

4) Register formation
       fixed-lineage norms at pre-B13 / final
       own final high-norm register count
       own-vs-baseline register-mask Jaccard
       fixed register mean cosine / coefficient on mu1,mu2

5) Native B12/B13 register interrogation
   Focus heads H5,H11,H12,H14,H15:
       slot0 -> fixed B13 visible-register set
       register-lineage queries -> slot0
   With an extra token:
       slot0 -> extra
       extra -> registers
       register queries -> extra
   Token labels are explicit:
       normal RN: slot0=CLS, extra=RN
       swap:      slot0=RN,  extra=CLS_COPY

6) B20/B21/B22 exchange
   Same directed metrics, plus a loose "mu field":
       non-register patches with signed cos(mu1) >= --mu_field_cos
   This is deliberately called MU_FIELD, not the stricter prior MU_CACHE role.

7) Embedding / token wreckage
       official slot0 image embedding cosine to:
           no-RN baseline
           same-token-mode Qon/Bon baseline
       raw embedding norm
       final slot0 / extra residual norms and mu1/mu2 cosines
       swap-only copied-CLS projected embedding cosine to normal baseline

8) Paired 2x2 factorial effects
   For selected per-image metrics the script reports:
       Qread-off main effect
       Vbroadcast-off main effect
       interaction
   with paired bootstrap 95% CIs, separately for no_rn and rn.

Outputs
-------
<out>/
    config.json
    rn_token_audit.json

    <model>/oracle_audit.json
    <model>/baseline_oracle.npz   # if recomputed; reused oracle is never modified

    early_per_image.csv
    early_summary.csv
    mlp_precursor_signature_summary.csv

    late_focus_per_image.csv
    final_per_image.csv
    final_summary.csv
    factorial_effects.csv

    plots/
      01_B9_mu1_2x2.png
      02_preB13_register_lineage_norm_2x2.png
      03_B13_H5_register_vs_RN.png
      04_B20_H14_exchange.png
      05_final_register_jaccard_2x2.png
      06_embedding_cosine_2x2.png
      07_swap_token_embeddings.png
      08_mlp_precursor_signature.png

    SUMMARY.txt
    compact_summary_workspace_cls_mu_causal.zip

Runtime design
--------------
For each batch/model the four early branches are run only through pre-B13,
then each is forked into no_rn and rn tails.  The swap tail is run only from
Qread=ON/Vbroadcast=ON.  The script does not run nine independent full forwards.

The phase windows are independent, so the default 2x2 asks the causal chain:
    B0--B5 CLS read history  x  B6--B10 CLS broadcast availability.

Important
---------
DataFrame column access in this file uses bracket syntax only.  The Great
Pandas Head Incident is not invited back.
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
import probe_tools_backbone as _backbone_tools
from probe_tools_analysis import (cosine_rows, projected_direction_per_head, savefig)


import argparse
import csv
import gc
import importlib
import importlib.util
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# =============================================================================
# Constants
# =============================================================================

MODEL_ORDER = ("pretrained", "gmp", "finetune_stripped")

QREAD_ON = "Qon"
QREAD_OFF = "Qoff"
VBROADCAST_ON = "Bon"
VBROADCAST_OFF = "Boff"

TOKEN_NO_RN = "no_rn"
TOKEN_RN = "rn"
TOKEN_SWAP = "swap_rn0_cls_last"

EARLY_BRANCHES = (
    (QREAD_ON,  VBROADCAST_ON),
    (QREAD_OFF, VBROADCAST_ON),
    (QREAD_ON,  VBROADCAST_OFF),
    (QREAD_OFF, VBROADCAST_OFF),
)

DEFAULT_MANIFEST = "nop_bc/fixed_sink_manifest.csv"
DEFAULT_OLD_ORACLE_ROOT = r"cls_gipu_exchange_no_rn"
DEFAULT_OUT = r"cls_mu_causal_2x2_rn_swap"

DEFAULT_XATTN_CHECKPOINT = "REPLACE_WITH_CHECKPOINT.pt"

EARLY_BLOCKS_DEFAULT = (6, 7, 8, 9, 10)
MID_FOCUS_BLOCKS = (11, 12)
LATE_FOCUS_BLOCKS = (13, 20, 21, 22)
FOCUS_HEADS_DEFAULT = (5, 11, 12, 14, 15)

EPS = 1e-12


# =============================================================================
# Generic helpers
# =============================================================================

def parse_ints(text: str) -> list[int]:
    out: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(token))
    return sorted(set(out))


def resolve_local(path_text: str, beside_script: bool = True) -> Path:
    path = Path(path_text)
    if path.is_file() or path.is_dir():
        return path
    if beside_script:
        alt = Path(__file__).resolve().parent / path_text
        if alt.is_file() or alt.is_dir():
            return alt
    return path


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def qfinite(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def meanfinite(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def pearson_rows(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    af = a.float() - a.float().mean(dim=-1, keepdim=True)
    bf = b.float() - b.float().mean(dim=-1, keepdim=True)
    num = (af * bf).sum(dim=-1)
    den = af.square().sum(dim=-1).sqrt() * bf.square().sum(dim=-1).sqrt()
    result = num / den.clamp_min(eps)
    return torch.where(den > eps, result, torch.full_like(result, float("nan")))


def fit_uncentered_basis(x_nd: np.ndarray, rank: int = 2) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_nd, np.float64)
    n, d = x.shape
    k = min(rank, n, d)
    if n <= d:
        gram = x @ x.T
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1][:k]
        evals = np.clip(evals[order], 0.0, None)
        singular = np.sqrt(evals)
        left = evecs[:, order]
        basis = []
        for j in range(k):
            if singular[j] <= 1e-12:
                basis.append(np.zeros(d, np.float64))
            else:
                basis.append((left[:, j].T @ x) / singular[j])
        basis = np.stack(basis)
    else:
        _u, singular, vt = np.linalg.svd(x, full_matrices=False)
        basis = vt[:k]
        singular = singular[:k]

    basis /= np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)

    mean = x.mean(axis=0)
    if float(basis[0] @ mean) < 0:
        basis[0] *= -1

    if len(basis) > 1:
        residual = x - (x @ basis[0:1].T) @ basis[0:1]
        residual_mean = residual.mean(axis=0)
        if float(basis[1] @ residual_mean) < 0:
            basis[1] *= -1

    return basis.astype(np.float32), np.asarray(singular[:k], np.float32)


def select_register_mask(
    norms_bp: torch.Tensor,
    threshold: float,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    batch, patches = norms_bp.shape
    result = torch.zeros(batch, patches, dtype=torch.bool, device=norms_bp.device)
    for bi in range(batch):
        idx = torch.nonzero(norms_bp[bi] >= threshold, as_tuple=False).flatten()
        if maximum > 0 and idx.numel() > maximum:
            idx = idx[torch.topk(norms_bp[bi, idx], k=maximum).indices]
        if idx.numel() < minimum:
            idx = torch.topk(norms_bp[bi], k=min(max(1, minimum), patches)).indices
        result[bi, idx] = True
    return result


def projected_v_row_norms_batch(
    v_bhtd: torch.Tensor,
    out_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    v: [B,H,T,dh]
    return projected row norms [B,H,T] after each head's W_O slice.
    """
    batch, heads, tokens, dh = v_bhtd.shape
    result = torch.empty(batch, heads, tokens, device=v_bhtd.device, dtype=torch.float32)
    weight = out_proj_weight.detach().float()
    for head in range(heads):
        wh = weight[:, head * dh:(head + 1) * dh]
        gram = wh.T @ wh
        vh = v_bhtd[:, head].float()
        result[:, head] = torch.sqrt(
            torch.einsum("btd,df,btf->bt", vh, gram, vh).clamp_min(0)
        )
    return result


def patch_mask_sum(x_bht: torch.Tensor, mask_bp: torch.Tensor) -> torch.Tensor:
    return (x_bht[:, :, 1:1 + mask_bp.shape[1]] * mask_bp[:, None, :].float()).sum(dim=-1)


def patch_mask_mean(x_bht: torch.Tensor, mask_bp: torch.Tensor) -> torch.Tensor:
    spatial = x_bht[:, :, 1:1 + mask_bp.shape[1]]
    mask = mask_bp[:, None, :].float()
    denominator = mask.sum(dim=-1)
    numerator = (spatial * mask).sum(dim=-1)
    mean = numerator / denominator.clamp_min(1.0)
    return torch.where(denominator > 0, mean, torch.full_like(mean, float("nan")))


def jaccard_rows(a_bp: torch.Tensor, b_bp: torch.Tensor) -> torch.Tensor:
    intersection = (a_bp & b_bp).sum(dim=-1).float()
    union = (a_bp | b_bp).sum(dim=-1).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def normalize_embedding(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-8)


def condition_name(token_mode: str, qread: str, vbroadcast: str) -> str:
    return f"{token_mode}__{qread}__{vbroadcast}"


# =============================================================================
# RN extraction
# =============================================================================

def load_trained_rn_token(base, args) -> tuple[torch.Tensor, dict[str, Any]]:
    checkpoint = Path(args.xattn_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"x-attn checkpoint not found: {checkpoint}")

    for module_name in (args.xattn_module, args.pickle_module, "oaiclip", "clip"):
        if not module_name:
            continue
        try:
            importlib.import_module(module_name)
        except Exception:
            pass

    obj = base.D.safe_torch_load(checkpoint)
    state = base.D.extract_state_dict(obj, str(checkpoint))
    del obj

    key = "visual.read_null_token"
    if key not in state:
        candidates = [name for name in state if name.endswith("visual.read_null_token")]
        if len(candidates) != 1:
            raise KeyError(
                f"Could not find unique {key}; candidates={candidates[:10]}"
            )
        key = candidates[0]

    token = state[key].detach().float().cpu().clone()
    if token.ndim != 1:
        raise RuntimeError(f"RN token must be 1-D, got {tuple(token.shape)}")

    insert_key = "visual.read_null_insert_block_config"
    insert_block = 13
    if insert_key in state:
        insert_block = int(state[insert_key].detach().cpu().item())

    audit = {
        "checkpoint": str(checkpoint),
        "state_key": key,
        "shape": list(token.shape),
        "norm": float(token.norm()),
        "mean": float(token.mean()),
        "std": float(token.std(unbiased=False)),
        "checkpoint_insert_block": int(insert_block),
        "experiment_insert_block": int(args.rn_insert_block),
        "insert_block_matches": bool(insert_block == args.rn_insert_block),
    }
    if insert_block != args.rn_insert_block:
        raise RuntimeError(
            f"RN checkpoint says insert before B{insert_block}, "
            f"but experiment requested B{args.rn_insert_block}"
        )
    return token, audit


# =============================================================================
# Baseline oracle
# =============================================================================

@dataclass
class Oracle:
    mu_basis: np.ndarray
    singular_values: np.ndarray
    b23_reg_mask: np.ndarray
    b13_reg_mask: np.ndarray
    stim_id: np.ndarray
    source: str


def try_load_old_oracle(
    root: Path,
    model_name: str,
    manifest: pd.DataFrame,
    width: int,
    patches: int,
) -> Optional[Oracle]:
    path = root / model_name / "mu_basis.npz"
    if not path.is_file():
        return None

    data = np.load(path, allow_pickle=True)
    needed = {"mu_basis", "singular_values", "b23_reg_mask", "b13_reg_mask", "stim_id"}
    if not needed.issubset(set(data.files)):
        return None

    ids = np.asarray(data["stim_id"], dtype=object).astype(str)
    current = manifest["stim_id"].astype(str).to_numpy()
    if ids.shape != current.shape or not np.array_equal(ids, current):
        return None

    basis = np.asarray(data["mu_basis"], np.float32)
    b23 = np.asarray(data["b23_reg_mask"], np.uint8)
    b13 = np.asarray(data["b13_reg_mask"], np.uint8)
    if basis.shape[1] != width or b23.shape != (len(manifest), patches):
        return None
    if b13.shape != (len(manifest), patches):
        return None

    return Oracle(
        mu_basis=basis[:2],
        singular_values=np.asarray(data["singular_values"], np.float32)[:2],
        b23_reg_mask=b23,
        b13_reg_mask=b13,
        stim_id=ids,
        source=str(path),
    )


@torch.no_grad()
def recompute_oracle(
    bundle,
    manifest: pd.DataFrame,
    args,
    model_dir: Path,
) -> Oracle:
    visual = bundle.model.visual
    n_images = len(manifest)
    patches = int(visual.positional_embedding.shape[0] - 1)
    width = int(visual.positional_embedding.shape[1])

    b23_all = np.zeros((n_images, patches), np.uint8)
    b13_all = np.zeros((n_images, patches), np.uint8)
    register_means = np.zeros((n_images, width), np.float32)

    for start in tqdm(
        range(0, n_images, args.batch_size),
        desc="oracle baseline",
        unit="batch",
    ):
        chunk = manifest.iloc[start:start + args.batch_size]
        tensors = []
        for row in chunk.itertuples(index=False):
            with Image.open(str(row.path)) as image:
                tensors.append(bundle.preprocess(image.convert("RGB")))
        images = torch.stack(tensors).to(bundle.device, dtype=bundle.model.dtype)

        x = visual._prepare_tokens(images)
        batch = images.shape[0]
        b13_mask = torch.zeros(batch, patches, dtype=torch.bool, device=bundle.device)

        for block_index, block in enumerate(visual.transformer.resblocks):
            if block_index == args.rn_insert_block:
                pre_norm = x[1:].float().norm(dim=-1).T
                b13_mask = pre_norm >= float(args.b13_visible_threshold)
            x = block(x)

        patch = x[1:].permute(1, 0, 2).float()
        norms = patch.norm(dim=-1)
        b23_mask = select_register_mask(
            norms,
            args.final_register_threshold,
            args.final_register_min,
            args.final_register_max,
        )

        for bi in range(batch):
            gi = start + bi
            b23_all[gi] = b23_mask[bi].cpu().numpy().astype(np.uint8)
            b13_all[gi] = b13_mask[bi].cpu().numpy().astype(np.uint8)
            register_means[gi] = patch[bi, b23_mask[bi]].mean(dim=0).cpu().numpy()

        del images, x, patch
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()

    basis, singular = fit_uncentered_basis(register_means, rank=2)
    ids = manifest["stim_id"].astype(str).to_numpy(dtype=object)

    out_path = model_dir / "baseline_oracle.npz"
    np.savez_compressed(
        out_path,
        mu_basis=basis,
        singular_values=singular,
        b23_reg_mask=b23_all,
        b13_reg_mask=b13_all,
        register_means=register_means,
        stim_id=ids,
    )

    return Oracle(
        mu_basis=basis,
        singular_values=singular,
        b23_reg_mask=b23_all,
        b13_reg_mask=b13_all,
        stim_id=ids,
        source=str(out_path),
    )


# =============================================================================
# Explicit attention with simultaneous Qread / Vbroadcast interventions
# =============================================================================

@dataclass
class AttentionData:
    output_tbd: torch.Tensor
    probs_bhts: torch.Tensor
    v_bhsd: torch.Tensor
    z_bhtd: torch.Tensor


def apply_early_2x2(
    probs_bhts: torch.Tensor,
    qread_on: bool,
    vbroadcast_on: bool,
    patch_count: int,
) -> torch.Tensor:
    """
    Apply the requested Qread/Vbroadcast interventions to already-softmaxed probabilities.

    These interventions are used only before B13; token layout is exactly:
        0 = CLS
        1..P = spatial
    """
    probs = probs_bhts.clone()
    spatial_end = 1 + patch_count

    if not qread_on:
        # Q=CLS can keep only source CLS.  This is exactly a normalized self-only row.
        probs[:, :, 0, :] = 0
        probs[:, :, 0, 0] = 1

    if not vbroadcast_on:
        # Spatial queries cannot use CLS source.
        rows = probs[:, :, 1:spatial_end, :]
        rows[:, :, :, 0] = 0
        denominator = rows.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs[:, :, 1:spatial_end, :] = rows / denominator

    return probs


def explicit_attention(
    block,
    ln1_tbd: torch.Tensor,
    qread_on: bool,
    vbroadcast_on: bool,
    patch_count: int,
) -> AttentionData:
    attn = block.attn
    if getattr(attn, "_prefix_k", None) is not None:
        raise RuntimeError("Unexpected RegCache prefix K/V in explicit early attention")
    if getattr(attn, "head_mask", None) is not None:
        raise RuntimeError("Unexpected persistent head_mask in explicit early attention")
    if getattr(attn, "head_scale", None):
        raise RuntimeError("Unexpected persistent head_scale in explicit early attention")

    tokens, batch, width = ln1_tbd.shape
    heads = int(attn.num_heads)
    dh = width // heads

    q = attn.q_proj(ln1_tbd)
    k = attn.k_proj(ln1_tbd)
    v = attn.v_proj(ln1_tbd)

    def split(x_tbd: torch.Tensor) -> torch.Tensor:
        return (
            x_tbd.permute(1, 0, 2)
            .reshape(batch, tokens, heads, dh)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    qh = split(q) * (dh ** -0.5)
    kh = split(k)
    vh = split(v)

    logits = torch.matmul(qh, kh.transpose(-1, -2))
    probs = F.softmax(logits, dim=-1)
    probs = apply_early_2x2(
        probs,
        qread_on=qread_on,
        vbroadcast_on=vbroadcast_on,
        patch_count=patch_count,
    )

    z = torch.matmul(probs, vh)  # [B,H,T,dh]
    merged = (
        z.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(batch, tokens, width)
        .permute(1, 0, 2)
        .contiguous()
    )
    output = F.linear(merged, attn.out_proj.weight, attn.out_proj.bias)

    return AttentionData(
        output_tbd=output,
        probs_bhts=probs,
        v_bhsd=vh,
        z_bhtd=z,
    )


@torch.no_grad()
def explicit_attention_parity_check(block, ln1_tbd: torch.Tensor, patch_count: int) -> dict[str, float]:
    native, _weights = block.attention(ln1_tbd, need_weights=False, capture=False)
    explicit = explicit_attention(
        block,
        ln1_tbd,
        qread_on=True,
        vbroadcast_on=True,
        patch_count=patch_count,
    ).output_tbd

    max_abs = float((native.float() - explicit.float()).abs().max())
    flat_native = native.float().reshape(-1)
    flat_explicit = explicit.float().reshape(-1)
    cosine = float(
        F.cosine_similarity(flat_native[None], flat_explicit[None], dim=-1, eps=1e-12)[0]
    )
    return {"max_abs": max_abs, "cosine": cosine}


# =============================================================================
# Population MLP precursor accumulators
# =============================================================================

@dataclass
class VectorPairAcc:
    reg_sum: torch.Tensor
    reg_count: int
    other_sum: torch.Tensor
    other_count: int


class PrecursorAccumulator:
    def __init__(self, width_mlp: int):
        self.width_mlp = int(width_mlp)
        self.data: dict[tuple[str, str, str, int], VectorPairAcc] = {}

    def add(
        self,
        model_name: str,
        qread: str,
        vbroadcast: str,
        block: int,
        gelu_tbd: torch.Tensor,
        fixed_reg_bp: torch.Tensor,
    ) -> None:
        key = (model_name, qread, vbroadcast, int(block))
        if key not in self.data:
            self.data[key] = VectorPairAcc(
                reg_sum=torch.zeros(self.width_mlp, dtype=torch.float64),
                reg_count=0,
                other_sum=torch.zeros(self.width_mlp, dtype=torch.float64),
                other_count=0,
            )
        acc = self.data[key]

        gelu_bpd = gelu_tbd[1:].permute(1, 0, 2).detach().float().cpu()
        reg_cpu = fixed_reg_bp.detach().cpu().bool()

        if int(reg_cpu.sum()) > 0:
            acc.reg_sum += gelu_bpd[reg_cpu].double().sum(dim=0)
            acc.reg_count += int(reg_cpu.sum())

        other = ~reg_cpu
        if int(other.sum()) > 0:
            acc.other_sum += gelu_bpd[other].double().sum(dim=0)
            acc.other_count += int(other.sum())

    def summarize(self) -> list[dict[str, Any]]:
        baseline: dict[tuple[str, int], torch.Tensor] = {}
        for (model, qread, vbroadcast, block), acc in self.data.items():
            if qread == QREAD_ON and vbroadcast == VBROADCAST_ON:
                reg_mean = acc.reg_sum / max(1, acc.reg_count)
                other_mean = acc.other_sum / max(1, acc.other_count)
                baseline[(model, block)] = (reg_mean - other_mean).float()

        rows: list[dict[str, Any]] = []
        for (model, qread, vbroadcast, block), acc in sorted(self.data.items()):
            reg_mean = acc.reg_sum / max(1, acc.reg_count)
            other_mean = acc.other_sum / max(1, acc.other_count)
            gap = (reg_mean - other_mean).float()
            ref = baseline[(model, block)]

            ref_norm = float(ref.norm())
            gap_norm = float(gap.norm())
            cosine = float(F.cosine_similarity(gap[None], ref[None], dim=-1, eps=1e-12)[0])
            projection = float(torch.dot(gap, ref) / ref.square().sum().clamp_min(1e-12))

            rows.append({
                "model_name": model,
                "qread": qread,
                "vbroadcast": vbroadcast,
                "block": int(block),
                "reg_token_count": int(acc.reg_count),
                "other_token_count": int(acc.other_count),
                "baseline_gap_norm": ref_norm,
                "condition_gap_norm": gap_norm,
                "gap_norm_ratio_to_baseline": gap_norm / max(ref_norm, 1e-12),
                "gap_cosine_to_baseline": cosine,
                "gap_projection_amplitude_on_baseline": projection,
            })
        return rows


# =============================================================================
# Per-block measurements
# =============================================================================

def source_mu_contribution(
    probs_bhts: torch.Tensor,
    v_bhsd: torch.Tensor,
    out_proj_weight: torch.Tensor,
    direction_d: torch.Tensor,
    source_mask_bp: torch.Tensor,
) -> torch.Tensor:
    """
    Sum signed residual contribution along direction from selected patch sources.
    Return [B,T_query].
    """
    batch, heads, tokens, _ = probs_bhts.shape
    patch_count = source_mask_bp.shape[1]
    w = projected_direction_per_head(out_proj_weight, direction_d, heads)
    vdir = torch.einsum("bhsd,hd->bhs", v_bhsd.float(), w)  # [B,H,S]

    source_mask_t = torch.cat(
        [
            torch.zeros(batch, 1, dtype=torch.bool, device=source_mask_bp.device),
            source_mask_bp,
        ],
        dim=1,
    )
    if tokens > 1 + patch_count:
        extra = torch.zeros(
            batch,
            tokens - (1 + patch_count),
            dtype=torch.bool,
            device=source_mask_bp.device,
        )
        source_mask_t = torch.cat([source_mask_t, extra], dim=1)

    return (
        probs_bhts
        * vdir[:, :, None, :]
        * source_mask_t[:, None, None, :].float()
    ).sum(dim=-1).sum(dim=1)


def cls_source_mu_contribution(
    probs_bhts: torch.Tensor,
    v_bhsd: torch.Tensor,
    out_proj_weight: torch.Tensor,
    direction_d: torch.Tensor,
) -> torch.Tensor:
    heads = probs_bhts.shape[1]
    w = projected_direction_per_head(out_proj_weight, direction_d, heads)
    vdir = torch.einsum("bhsd,hd->bhs", v_bhsd.float(), w)
    return (probs_bhts[:, :, :, 0] * vdir[:, :, 0][:, :, None]).sum(dim=1)


def exact_attention_direction(
    attention_output_tbd: torch.Tensor,
    direction_d: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("tbd,d->bt", attention_output_tbd.float(), direction_d.float())


def state_projection_rows(
    state_bd: torch.Tensor,
    mu1_d: torch.Tensor,
    mu2_d: torch.Tensor,
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}_mu1_coef": state_bd.float() @ mu1_d.float(),
        f"{prefix}_mu1_cos": cosine_rows(state_bd, mu1_d.view(1, -1).expand_as(state_bd)),
        f"{prefix}_mu2_coef": state_bd.float() @ mu2_d.float(),
        f"{prefix}_mu2_cos": cosine_rows(state_bd, mu2_d.view(1, -1).expand_as(state_bd)),
    }


def final_embedding(visual, x_tbd: torch.Tensor, token_index: int = 0) -> torch.Tensor:
    state = x_tbd[token_index].float()
    normalized = visual.ln_post(state)
    if visual.proj is not None:
        normalized = normalized @ visual.proj.float()
    return normalized.float()


def loose_mu_field_mask(
    patch_bpd: torch.Tensor,
    fixed_reg_bp: torch.Tensor,
    mu1_d: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    cos = F.cosine_similarity(
        patch_bpd.float(),
        mu1_d.view(1, 1, -1),
        dim=-1,
        eps=1e-8,
    )
    return (~fixed_reg_bp.bool()) & (cos >= float(threshold))


def focus_attention_rows(
    *,
    model_name: str,
    token_mode: str,
    qread: str,
    vbroadcast: str,
    block_index: int,
    stim_ids: Sequence[str],
    sources: Sequence[str],
    probs_bhts: torch.Tensor,
    v_bhsd: torch.Tensor,
    out_proj_weight: torch.Tensor,
    patch_count: int,
    fixed_reg_bp: torch.Tensor,
    fixed_b13_bp: torch.Tensor,
    patch_bpd: torch.Tensor,
    mu1_d: torch.Tensor,
    focus_heads: Sequence[int],
    mu_field_cos: float,
    slot0_label: str,
    extra_label: Optional[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch, heads, tokens, _ = probs_bhts.shape
    vproj = projected_v_row_norms_batch(v_bhsd, out_proj_weight)
    av = probs_bhts * vproj[:, :, None, :]
    av_den = av.sum(dim=-1).clamp_min(1e-12)

    reg_t = torch.cat(
        [torch.zeros(batch, 1, dtype=torch.bool, device=fixed_reg_bp.device), fixed_reg_bp],
        dim=1,
    )
    b13_t = torch.cat(
        [torch.zeros(batch, 1, dtype=torch.bool, device=fixed_b13_bp.device), fixed_b13_bp],
        dim=1,
    )
    if tokens > 1 + patch_count:
        extra_zeros = torch.zeros(
            batch,
            tokens - (1 + patch_count),
            dtype=torch.bool,
            device=fixed_reg_bp.device,
        )
        reg_t = torch.cat([reg_t, extra_zeros], dim=1)
        b13_t = torch.cat([b13_t, extra_zeros], dim=1)

    mu_field = loose_mu_field_mask(
        patch_bpd,
        fixed_reg_bp,
        mu1_d,
        threshold=mu_field_cos,
    )
    mu_field_t = torch.cat(
        [torch.zeros(batch, 1, dtype=torch.bool, device=mu_field.device), mu_field],
        dim=1,
    )
    if tokens > 1 + patch_count:
        extra_zeros = torch.zeros(
            batch,
            tokens - (1 + patch_count),
            dtype=torch.bool,
            device=mu_field.device,
        )
        mu_field_t = torch.cat([mu_field_t, extra_zeros], dim=1)

    extra_index = tokens - 1 if extra_label is not None else None

    for head in focus_heads:
        if head >= heads:
            continue

        p = probs_bhts[:, head]
        a = av[:, head]
        den = av_den[:, head]

        slot0_to_reg_attn = (p[:, 0] * reg_t.float()).sum(dim=-1)
        slot0_to_b13_attn = (p[:, 0] * b13_t.float()).sum(dim=-1)
        slot0_to_mu_field_attn = (p[:, 0] * mu_field_t.float()).sum(dim=-1)

        slot0_to_reg_av = (a[:, 0] * reg_t.float()).sum(dim=-1) / den[:, 0]
        slot0_to_b13_av = (a[:, 0] * b13_t.float()).sum(dim=-1) / den[:, 0]

        reg_query_to_slot0 = patch_mask_mean(
            p[:, None, :, 0], fixed_reg_bp
        )[:, 0]
        mu_query_to_slot0 = patch_mask_mean(
            p[:, None, :, 0], mu_field
        )[:, 0]

        reg_query_to_slot0_av = patch_mask_mean(
            (a[:, :, 0] / den).unsqueeze(1), fixed_reg_bp
        )[:, 0]

        if extra_index is not None:
            slot0_to_extra_attn = p[:, 0, extra_index]
            slot0_to_extra_av = a[:, 0, extra_index] / den[:, 0]
            extra_to_reg_attn = (p[:, extra_index] * reg_t.float()).sum(dim=-1)
            extra_to_b13_attn = (p[:, extra_index] * b13_t.float()).sum(dim=-1)
            extra_to_slot0_attn = p[:, extra_index, 0]
            reg_query_to_extra = patch_mask_mean(
                p[:, None, :, extra_index], fixed_reg_bp
            )[:, 0]
            mu_query_to_extra = patch_mask_mean(
                p[:, None, :, extra_index], mu_field
            )[:, 0]
        else:
            nan = torch.full((batch,), float("nan"), device=probs_bhts.device)
            slot0_to_extra_attn = nan
            slot0_to_extra_av = nan
            extra_to_reg_attn = nan
            extra_to_b13_attn = nan
            extra_to_slot0_attn = nan
            reg_query_to_extra = nan
            mu_query_to_extra = nan

        for bi in range(batch):
            rows.append({
                "model_name": model_name,
                "token_mode": token_mode,
                "qread": qread,
                "vbroadcast": vbroadcast,
                "condition": condition_name(token_mode, qread, vbroadcast),
                "stim_id": str(stim_ids[bi]),
                "source": str(sources[bi]),
                "block": int(block_index),
                "head": int(head),
                "slot0_label": slot0_label,
                "extra_label": extra_label if extra_label is not None else "",
                "slot0_to_reg_attn": float(slot0_to_reg_attn[bi]),
                "slot0_to_b13reg_attn": float(slot0_to_b13_attn[bi]),
                "slot0_to_mu_field_attn": float(slot0_to_mu_field_attn[bi]),
                "slot0_to_reg_avfrac": float(slot0_to_reg_av[bi]),
                "slot0_to_b13reg_avfrac": float(slot0_to_b13_av[bi]),
                "reg_query_to_slot0_attn_mean": float(reg_query_to_slot0[bi]),
                "reg_query_to_slot0_avfrac_mean": float(reg_query_to_slot0_av[bi]),
                "mu_field_query_to_slot0_attn_mean": float(mu_query_to_slot0[bi]),
                "slot0_to_extra_attn": float(slot0_to_extra_attn[bi]),
                "slot0_to_extra_avfrac": float(slot0_to_extra_av[bi]),
                "extra_to_reg_attn": float(extra_to_reg_attn[bi]),
                "extra_to_b13reg_attn": float(extra_to_b13_attn[bi]),
                "extra_to_slot0_attn": float(extra_to_slot0_attn[bi]),
                "reg_query_to_extra_attn_mean": float(reg_query_to_extra[bi]),
                "mu_field_query_to_extra_attn_mean": float(mu_query_to_extra[bi]),
                "mu_field_count": int(mu_field[bi].sum()),
            })
    return rows


# =============================================================================
# Mid branch B6--B12
# =============================================================================

@dataclass
class MidResult:
    pre_b13_tbd: torch.Tensor
    pre_b13_fixed_reg_norm: torch.Tensor
    pre_b13_fixed_reg_mu1_cos: torch.Tensor
    pre_b13_fixed_reg_mu2_cos: torch.Tensor


@torch.no_grad()
def run_mid_branch(
    *,
    base,
    bundle,
    x_pre_b6: torch.Tensor,
    model_name: str,
    qread: str,
    vbroadcast: str,
    stim_ids: Sequence[str],
    sources: Sequence[str],
    fixed_reg_bp: torch.Tensor,
    fixed_b13_bp: torch.Tensor,
    mu1_d: torch.Tensor,
    mu2_d: torch.Tensor,
    args,
    precursor_acc: PrecursorAccumulator,
    early_rows: list[dict[str, Any]],
    late_focus_rows: list[dict[str, Any]],
) -> MidResult:
    visual = bundle.model.visual
    patch_count = fixed_reg_bp.shape[1]
    q_on = qread == QREAD_ON
    b_on = vbroadcast == VBROADCAST_ON

    x = x_pre_b6.clone()

    active_or_measured = sorted(
        set(args.qread_blocks)
        | set(args.vbroadcast_blocks)
        | set(args.measure_blocks)
        | set(args.precursor_blocks)
    )
    first_active = min(active_or_measured)

    for block_index in range(first_active, args.rn_insert_block):
        block = visual.transformer.resblocks[block_index]
        pre = x
        ln1 = block.ln_1(x)

        use_explicit = (
            block_index in args.qread_blocks
            or block_index in args.vbroadcast_blocks
            or block_index in args.measure_blocks
        )

        if use_explicit:
            # A factor is active only in its own causal window.
            effective_qread_on = q_on if block_index in args.qread_blocks else True
            effective_vbroadcast_on = (
                b_on if block_index in args.vbroadcast_blocks else True
            )
            attn = explicit_attention(
                block,
                ln1,
                qread_on=effective_qread_on,
                vbroadcast_on=effective_vbroadcast_on,
                patch_count=patch_count,
            )
            attention_output = attn.output_tbd
            probs = attn.probs_bhts
            values = attn.v_bhsd
        else:
            attention_output, probs0 = block.attention(
                ln1,
                need_weights=True,
                capture=True,
            )
            probs = base.normalize_probs_shape(
                probs0,
                x.shape[1],
                int(block.attn.num_heads),
                x.shape[0],
            ).float()
            values = base.normalize_qkv_shape(
                block.attn.last_v,
                x.shape[1],
                int(block.attn.num_heads),
                x.shape[0],
            ).float()

        x_attn = x + attention_output
        ln2 = block.ln_2(x_attn)
        fc = block.mlp.c_fc(ln2)
        gelu = block.mlp.gelu(fc)
        x = x_attn + block.mlp.c_proj(gelu)

        if block_index in args.precursor_blocks:
            precursor_acc.add(
                model_name,
                qread,
                vbroadcast,
                block_index,
                gelu,
                fixed_reg_bp,
            )

        if block_index in args.measure_blocks:
            delta_mu1 = exact_attention_direction(attention_output, mu1_d)
            delta_mu2 = exact_attention_direction(attention_output, mu2_d)

            cls_mu1 = cls_source_mu_contribution(
                probs,
                values,
                block.attn.out_proj.weight,
                mu1_d,
            )
            cls_mu2 = cls_source_mu_contribution(
                probs,
                values,
                block.attn.out_proj.weight,
                mu2_d,
            )
            reg_mu1 = source_mu_contribution(
                probs,
                values,
                block.attn.out_proj.weight,
                mu1_d,
                fixed_reg_bp,
            )
            reg_mu2 = source_mu_contribution(
                probs,
                values,
                block.attn.out_proj.weight,
                mu2_d,
                fixed_reg_bp,
            )

            # Exact source-decomposition audit.  Early blocks contain only
            # CLS + P spatial tokens, so CLS-source + all-spatial-source + W_O
            # bias must reconstruct the full attention output along mu1/mu2.
            all_spatial_mask = torch.ones_like(fixed_reg_bp, dtype=torch.bool)
            spatial_mu1 = source_mu_contribution(
                probs,
                values,
                block.attn.out_proj.weight,
                mu1_d,
                all_spatial_mask,
            )
            spatial_mu2 = source_mu_contribution(
                probs,
                values,
                block.attn.out_proj.weight,
                mu2_d,
                all_spatial_mask,
            )
            bias = block.attn.out_proj.bias
            bias_mu1 = (
                torch.dot(bias.float(), mu1_d.float())
                if bias is not None else torch.tensor(0.0, device=mu1_d.device)
            )
            bias_mu2 = (
                torch.dot(bias.float(), mu2_d.float())
                if bias is not None else torch.tensor(0.0, device=mu2_d.device)
            )
            recon_mu1 = cls_mu1 + spatial_mu1 + bias_mu1
            recon_mu2 = cls_mu2 + spatial_mu2 + bias_mu2
            decomp_err1 = (recon_mu1 - delta_mu1).abs().max(dim=-1).values
            decomp_err2 = (recon_mu2 - delta_mu2).abs().max(dim=-1).values
            max_decomp_error = max(
                float(decomp_err1.max()),
                float(decomp_err2.max()),
            )
            if max_decomp_error > args.decomp_max_abs:
                raise RuntimeError(
                    f"Source decomposition audit failed at {model_name} "
                    f"B{block_index} {qread}/{vbroadcast}: "
                    f"max_abs={max_decomp_error:.3e} > {args.decomp_max_abs:.3e}"
                )

            pre_cls = pre[0].float()
            post_attn_cls = x_attn[0].float()
            pre_state = state_projection_rows(pre_cls, mu1_d, mu2_d, "cls_pre")
            post_state = state_projection_rows(
                post_attn_cls,
                mu1_d,
                mu2_d,
                "cls_postattn",
            )

            exact_patch1 = delta_mu1[:, 1:1 + patch_count]
            exact_patch2 = delta_mu2[:, 1:1 + patch_count]
            cls_patch1 = cls_mu1[:, 1:1 + patch_count]
            cls_patch2 = cls_mu2[:, 1:1 + patch_count]
            reg_patch1 = reg_mu1[:, 1:1 + patch_count]
            reg_patch2 = reg_mu2[:, 1:1 + patch_count]

            corr1 = pearson_rows(exact_patch1, cls_patch1)
            corr2 = pearson_rows(exact_patch2, cls_patch2)

            den1 = exact_patch1.abs().sum(dim=-1).clamp_min(EPS)
            den2 = exact_patch2.abs().sum(dim=-1).clamp_min(EPS)

            for bi in range(x.shape[1]):
                row = {
                    "model_name": model_name,
                    "qread": qread,
                    "vbroadcast": vbroadcast,
                    "early_condition": f"{qread}__{vbroadcast}",
                    "stim_id": str(stim_ids[bi]),
                    "source": str(sources[bi]),
                    "block": int(block_index),
                    "patch_delta_mu1_abs_mean": float(exact_patch1[bi].abs().mean()),
                    "patch_delta_mu2_abs_mean": float(exact_patch2[bi].abs().mean()),
                    "cls_source_mu1_abs_l1_ratio": float(
                        cls_patch1[bi].abs().sum() / den1[bi]
                    ),
                    "cls_source_mu2_abs_l1_ratio": float(
                        cls_patch2[bi].abs().sum() / den2[bi]
                    ),
                    "register_lineage_source_mu1_abs_l1_ratio": float(
                        reg_patch1[bi].abs().sum() / den1[bi]
                    ),
                    "register_lineage_source_mu2_abs_l1_ratio": float(
                        reg_patch2[bi].abs().sum() / den2[bi]
                    ),
                    "mu1_delta_corr_cls_source": float(corr1[bi]),
                    "mu2_delta_corr_cls_source": float(corr2[bi]),
                    "mu1_source_decomp_max_abs_error": float(decomp_err1[bi]),
                    "mu2_source_decomp_max_abs_error": float(decomp_err2[bi]),
                }
                for key, tensor in {**pre_state, **post_state}.items():
                    row[key] = float(tensor[bi])
                early_rows.append(row)

        if block_index in MID_FOCUS_BLOCKS:
            patch = pre[1:1 + patch_count].permute(1, 0, 2).float()
            late_focus_rows.extend(
                focus_attention_rows(
                    model_name=model_name,
                    token_mode="pre_b13_shared",
                    qread=qread,
                    vbroadcast=vbroadcast,
                    block_index=block_index,
                    stim_ids=stim_ids,
                    sources=sources,
                    probs_bhts=probs,
                    v_bhsd=values,
                    out_proj_weight=block.attn.out_proj.weight,
                    patch_count=patch_count,
                    fixed_reg_bp=fixed_reg_bp,
                    fixed_b13_bp=fixed_b13_bp,
                    patch_bpd=patch,
                    mu1_d=mu1_d,
                    focus_heads=args.focus_heads,
                    mu_field_cos=args.mu_field_cos,
                    slot0_label="CLS",
                    extra_label=None,
                )
            )

        base.clear_attn_cache(block)

    pre_b13_patch = x[1:1 + patch_count].permute(1, 0, 2).float()
    fixed_count = fixed_reg_bp.sum(dim=-1).float().clamp_min(1.0)
    fixed_norm = (
        pre_b13_patch.norm(dim=-1) * fixed_reg_bp.float()
    ).sum(dim=-1) / fixed_count

    reg_mean = (
        pre_b13_patch * fixed_reg_bp[:, :, None].float()
    ).sum(dim=1) / fixed_count[:, None]

    mu1_expand = mu1_d.view(1, -1).expand_as(reg_mean)
    mu2_expand = mu2_d.view(1, -1).expand_as(reg_mean)

    return MidResult(
        pre_b13_tbd=x.detach(),
        pre_b13_fixed_reg_norm=fixed_norm.detach(),
        pre_b13_fixed_reg_mu1_cos=cosine_rows(reg_mean, mu1_expand).detach(),
        pre_b13_fixed_reg_mu2_cos=cosine_rows(reg_mean, mu2_expand).detach(),
    )


# =============================================================================
# Tail B13--B23
# =============================================================================

@dataclass
class TailOutput:
    token_mode: str
    qread: str
    vbroadcast: str
    embedding: torch.Tensor
    extra_embedding: Optional[torch.Tensor]
    final_rows: list[dict[str, Any]]


@torch.no_grad()
def run_tail(
    *,
    base,
    bundle,
    pre_b13_tbd: torch.Tensor,
    model_name: str,
    token_mode: str,
    qread: str,
    vbroadcast: str,
    stim_ids: Sequence[str],
    sources: Sequence[str],
    fixed_reg_bp: torch.Tensor,
    fixed_b13_bp: torch.Tensor,
    mu1_d: torch.Tensor,
    mu2_d: torch.Tensor,
    rn_token_d: torch.Tensor,
    args,
    late_focus_rows: list[dict[str, Any]],
    mid_result: MidResult,
) -> TailOutput:
    visual = bundle.model.visual
    patch_count = fixed_reg_bp.shape[1]
    x = pre_b13_tbd.clone()
    batch = x.shape[1]

    if token_mode == TOKEN_NO_RN:
        slot0_label = "CLS"
        extra_label = None
    elif token_mode == TOKEN_RN:
        rn = rn_token_d.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        rn = rn.expand(1, batch, -1)
        x = torch.cat([x, rn], dim=0)
        slot0_label = "CLS"
        extra_label = "RN"
    elif token_mode == TOKEN_SWAP:
        old_cls = x[0:1].clone()
        rn = rn_token_d.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        rn = rn.expand(1, batch, -1)
        x = torch.cat([rn, x[1:], old_cls], dim=0)
        slot0_label = "RN"
        extra_label = "CLS_COPY"
    else:
        raise ValueError(token_mode)

    for block_index in range(args.rn_insert_block, 24):
        block = visual.transformer.resblocks[block_index]
        pre = x
        ln1 = block.ln_1(x)

        capture = block_index in LATE_FOCUS_BLOCKS
        attention_output, probs0 = block.attention(
            ln1,
            need_weights=capture,
            capture=capture,
        )
        x_attn = x + attention_output
        ln2 = block.ln_2(x_attn)
        gelu = block.mlp.gelu(block.mlp.c_fc(ln2))
        x = x_attn + block.mlp.c_proj(gelu)

        if capture:
            probs = base.normalize_probs_shape(
                probs0,
                batch,
                int(block.attn.num_heads),
                pre.shape[0],
            ).float()
            values = base.normalize_qkv_shape(
                block.attn.last_v,
                batch,
                int(block.attn.num_heads),
                pre.shape[0],
            ).float()
            patch = pre[1:1 + patch_count].permute(1, 0, 2).float()

            late_focus_rows.extend(
                focus_attention_rows(
                    model_name=model_name,
                    token_mode=token_mode,
                    qread=qread,
                    vbroadcast=vbroadcast,
                    block_index=block_index,
                    stim_ids=stim_ids,
                    sources=sources,
                    probs_bhts=probs,
                    v_bhsd=values,
                    out_proj_weight=block.attn.out_proj.weight,
                    patch_count=patch_count,
                    fixed_reg_bp=fixed_reg_bp,
                    fixed_b13_bp=fixed_b13_bp,
                    patch_bpd=patch,
                    mu1_d=mu1_d,
                    focus_heads=args.focus_heads,
                    mu_field_cos=args.mu_field_cos,
                    slot0_label=slot0_label,
                    extra_label=extra_label,
                )
            )

        base.clear_attn_cache(block)

    embedding = final_embedding(visual, x, token_index=0)
    extra_embedding = None
    if extra_label is not None:
        extra_embedding = final_embedding(visual, x, token_index=x.shape[0] - 1)

    patch_final = x[1:1 + patch_count].permute(1, 0, 2).float()
    norms = patch_final.norm(dim=-1)
    own_mask = select_register_mask(
        norms,
        args.final_register_threshold,
        args.final_register_min,
        args.final_register_max,
    )

    fixed_count = fixed_reg_bp.sum(dim=-1).float().clamp_min(1.0)
    fixed_norm = (
        norms * fixed_reg_bp.float()
    ).sum(dim=-1) / fixed_count
    fixed_mean = (
        patch_final * fixed_reg_bp[:, :, None].float()
    ).sum(dim=1) / fixed_count[:, None]

    fixed_mu1_cos = cosine_rows(
        fixed_mean,
        mu1_d.view(1, -1).expand_as(fixed_mean),
    )
    fixed_mu2_cos = cosine_rows(
        fixed_mean,
        mu2_d.view(1, -1).expand_as(fixed_mean),
    )
    fixed_mu1_coef = fixed_mean @ mu1_d
    fixed_mu2_coef = fixed_mean @ mu2_d

    own_jaccard = jaccard_rows(own_mask, fixed_reg_bp)
    own_count = own_mask.sum(dim=-1)

    slot0 = x[0].float()
    slot0_proj = state_projection_rows(slot0, mu1_d, mu2_d, "slot0_final")

    if extra_label is not None:
        extra = x[-1].float()
        extra_proj = state_projection_rows(extra, mu1_d, mu2_d, "extra_final")
        extra_norm = extra.norm(dim=-1)
    else:
        extra_proj = {}
        extra_norm = torch.full((batch,), float("nan"), device=x.device)

    rows: list[dict[str, Any]] = []
    for bi in range(batch):
        row = {
            "model_name": model_name,
            "token_mode": token_mode,
            "qread": qread,
            "vbroadcast": vbroadcast,
            "condition": condition_name(token_mode, qread, vbroadcast),
            "stim_id": str(stim_ids[bi]),
            "source": str(sources[bi]),
            "slot0_label": slot0_label,
            "extra_label": extra_label if extra_label is not None else "",
            "pre_b13_fixed_reg_norm": float(mid_result.pre_b13_fixed_reg_norm[bi]),
            "pre_b13_fixed_reg_mu1_cos": float(mid_result.pre_b13_fixed_reg_mu1_cos[bi]),
            "pre_b13_fixed_reg_mu2_cos": float(mid_result.pre_b13_fixed_reg_mu2_cos[bi]),
            "final_fixed_reg_norm_mean": float(fixed_norm[bi]),
            "final_fixed_reg_mu1_cos": float(fixed_mu1_cos[bi]),
            "final_fixed_reg_mu2_cos": float(fixed_mu2_cos[bi]),
            "final_fixed_reg_mu1_coef": float(fixed_mu1_coef[bi]),
            "final_fixed_reg_mu2_coef": float(fixed_mu2_coef[bi]),
            "final_own_register_count": int(own_count[bi]),
            "final_register_jaccard_to_baseline": float(own_jaccard[bi]),
            "final_patch_norm_mean": float(norms[bi].mean()),
            "final_patch_norm_p95": float(torch.quantile(norms[bi], 0.95)),
            "final_patch_norm_max": float(norms[bi].max()),
            "final_slot0_resid_norm": float(slot0[bi].norm()),
            "final_extra_resid_norm": float(extra_norm[bi]),
            "official_embedding_raw_norm": float(embedding[bi].norm()),
        }
        for key, tensor in slot0_proj.items():
            row[key] = float(tensor[bi])
        for key, tensor in extra_proj.items():
            row[key] = float(tensor[bi])
        rows.append(row)

    return TailOutput(
        token_mode=token_mode,
        qread=qread,
        vbroadcast=vbroadcast,
        embedding=embedding.detach(),
        extra_embedding=None if extra_embedding is None else extra_embedding.detach(),
        final_rows=rows,
    )


# =============================================================================
# Per-batch orchestration
# =============================================================================

@torch.no_grad()
def run_batch_model(
    *,
    base,
    bundle,
    model_name: str,
    manifest_chunk: pd.DataFrame,
    global_start: int,
    oracle: Oracle,
    rn_token_cpu: torch.Tensor,
    args,
    precursor_acc: PrecursorAccumulator,
    early_rows: list[dict[str, Any]],
    late_focus_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    parity_state: dict[str, bool],
) -> None:
    visual = bundle.model.visual
    tensors, ids, sources = [], [], []

    for row in manifest_chunk.itertuples(index=False):
        with Image.open(str(row.path)) as image:
            tensors.append(bundle.preprocess(image.convert("RGB")))
        ids.append(str(row.stim_id))
        sources.append(str(row.source))

    images = torch.stack(tensors).to(bundle.device, dtype=bundle.model.dtype)
    batch = images.shape[0]
    patch_count = int(visual.positional_embedding.shape[0] - 1)

    fixed_reg = torch.from_numpy(
        oracle.b23_reg_mask[global_start:global_start + batch].astype(bool)
    ).to(bundle.device)
    fixed_b13 = torch.from_numpy(
        oracle.b13_reg_mask[global_start:global_start + batch].astype(bool)
    ).to(bundle.device)

    mu1 = torch.from_numpy(oracle.mu_basis[0]).to(bundle.device).float()
    mu2 = torch.from_numpy(oracle.mu_basis[1]).to(bundle.device).float()
    rn_token = rn_token_cpu.to(bundle.device).float()

    # Shared prefix before the earliest intervention/measurement block.
    relevant_blocks = sorted(
        set(args.qread_blocks)
        | set(args.vbroadcast_blocks)
        | set(args.measure_blocks)
        | set(args.precursor_blocks)
    )
    first_relevant = min(relevant_blocks)
    x_pre_b6 = visual._prepare_tokens(images)
    for block_index in range(first_relevant):
        x_pre_b6 = visual.transformer.resblocks[block_index](x_pre_b6)

    # One runtime parity check per model.
    if not parity_state.get(model_name, False):
        block = visual.transformer.resblocks[first_relevant]
        ln1 = block.ln_1(x_pre_b6)
        audit = explicit_attention_parity_check(block, ln1, patch_count)
        print(
            f"[parity] {model_name} B{first_relevant}: "
            f"max_abs={audit['max_abs']:.3e} cosine={audit['cosine']:.9f}"
        )
        if audit["max_abs"] > args.parity_max_abs or audit["cosine"] < args.parity_min_cos:
            raise RuntimeError(
                f"Explicit attention parity failed for {model_name}: {audit}"
            )
        parity_state[model_name] = True

    outputs: dict[str, TailOutput] = {}

    # Four early branches.  Each is continued twice from the exact same pre-B13 state.
    for qread, vbroadcast in EARLY_BRANCHES:
        mid = run_mid_branch(
            base=base,
            bundle=bundle,
            x_pre_b6=x_pre_b6,
            model_name=model_name,
            qread=qread,
            vbroadcast=vbroadcast,
            stim_ids=ids,
            sources=sources,
            fixed_reg_bp=fixed_reg,
            fixed_b13_bp=fixed_b13,
            mu1_d=mu1,
            mu2_d=mu2,
            args=args,
            precursor_acc=precursor_acc,
            early_rows=early_rows,
            late_focus_rows=late_focus_rows,
        )

        for token_mode in (TOKEN_NO_RN, TOKEN_RN):
            out = run_tail(
                base=base,
                bundle=bundle,
                pre_b13_tbd=mid.pre_b13_tbd,
                model_name=model_name,
                token_mode=token_mode,
                qread=qread,
                vbroadcast=vbroadcast,
                stim_ids=ids,
                sources=sources,
                fixed_reg_bp=fixed_reg,
                fixed_b13_bp=fixed_b13,
                mu1_d=mu1,
                mu2_d=mu2,
                rn_token_d=rn_token,
                args=args,
                late_focus_rows=late_focus_rows,
                mid_result=mid,
            )
            outputs[out.condition if hasattr(out, "condition") else condition_name(token_mode, qread, vbroadcast)] = out

        if qread == QREAD_ON and vbroadcast == VBROADCAST_ON:
            swap = run_tail(
                base=base,
                bundle=bundle,
                pre_b13_tbd=mid.pre_b13_tbd,
                model_name=model_name,
                token_mode=TOKEN_SWAP,
                qread=qread,
                vbroadcast=vbroadcast,
                stim_ids=ids,
                sources=sources,
                fixed_reg_bp=fixed_reg,
                fixed_b13_bp=fixed_b13,
                mu1_d=mu1,
                mu2_d=mu2,
                rn_token_d=rn_token,
                args=args,
                late_focus_rows=late_focus_rows,
                mid_result=mid,
            )
            outputs[condition_name(TOKEN_SWAP, qread, vbroadcast)] = swap

    # Embedding comparisons are paired within this exact batch.
    no_rn_baseline = outputs[
        condition_name(TOKEN_NO_RN, QREAD_ON, VBROADCAST_ON)
    ]
    rn_baseline = outputs[
        condition_name(TOKEN_RN, QREAD_ON, VBROADCAST_ON)
    ]

    baseline_embedding = normalize_embedding(no_rn_baseline.embedding)
    rn_baseline_embedding = normalize_embedding(rn_baseline.embedding)

    for key, output in outputs.items():
        official = normalize_embedding(output.embedding)
        to_no_rn = cosine_rows(official, baseline_embedding)

        if output.token_mode == TOKEN_RN:
            mode_reference = rn_baseline_embedding
        else:
            mode_reference = baseline_embedding
        to_mode = cosine_rows(official, mode_reference)

        if output.extra_embedding is not None:
            extra_normed = normalize_embedding(output.extra_embedding)
            extra_to_no_rn = cosine_rows(extra_normed, baseline_embedding)
            extra_to_official = cosine_rows(extra_normed, official)
        else:
            extra_to_no_rn = torch.full((batch,), float("nan"), device=bundle.device)
            extra_to_official = torch.full((batch,), float("nan"), device=bundle.device)

        for bi, row in enumerate(output.final_rows):
            row["embedding_cos_to_no_rn_baseline"] = float(to_no_rn[bi])
            row["embedding_cos_to_same_token_mode_baseline"] = float(to_mode[bi])
            row["extra_projected_embedding_cos_to_no_rn_baseline"] = float(
                extra_to_no_rn[bi]
            )
            row["extra_projected_embedding_cos_to_official_slot0"] = float(
                extra_to_official[bi]
            )
            final_rows.append(row)

    del images, x_pre_b6, outputs
    if bundle.device.type == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# Summaries and factorial effects
# =============================================================================

def summarize_numeric(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    skip_cols: Sequence[str],
) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.columns
        if column not in set(group_cols) | set(skip_cols)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_cols), dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(group_cols, key_tuple)}
        row["n"] = len(group)
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(np.float64)
            finite = values[np.isfinite(values)]
            row[f"{column}_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[f"{column}_median"] = float(np.median(finite)) if finite.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, np.float64)
    chunk = 256
    for start in range(0, n_boot, chunk):
        count = min(chunk, n_boot - start)
        idx = rng.integers(0, values.size, size=(count, values.size))
        boot[start:start + count] = values[idx].mean(axis=1)
    return (
        mean,
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def factorial_rows(
    frame: pd.DataFrame,
    *,
    model_name: str,
    token_mode: str,
    metric: str,
    block: Optional[int],
    seed: int,
    n_boot: int,
) -> list[dict[str, Any]]:
    query = frame[
        (frame["model_name"] == model_name)
    ].copy()

    if "token_mode" in query.columns:
        query = query[query["token_mode"] == token_mode]

    if block is not None and "block" in query.columns:
        query = query[query["block"] == block]

    needed = {"stim_id", "qread", "vbroadcast", metric}
    if not needed.issubset(set(query.columns)):
        return []

    pivot = query.pivot_table(
        index="stim_id",
        columns=["qread", "vbroadcast"],
        values=metric,
        aggfunc="mean",
    )

    cols = [
        (QREAD_ON, VBROADCAST_ON),
        (QREAD_OFF, VBROADCAST_ON),
        (QREAD_ON, VBROADCAST_OFF),
        (QREAD_OFF, VBROADCAST_OFF),
    ]
    if not all(column in pivot.columns for column in cols):
        return []

    y00 = pivot[(QREAD_ON, VBROADCAST_ON)].to_numpy(np.float64)
    y10 = pivot[(QREAD_OFF, VBROADCAST_ON)].to_numpy(np.float64)
    y01 = pivot[(QREAD_ON, VBROADCAST_OFF)].to_numpy(np.float64)
    y11 = pivot[(QREAD_OFF, VBROADCAST_OFF)].to_numpy(np.float64)

    q_effect = 0.5 * ((y10 - y00) + (y11 - y01))
    b_effect = 0.5 * ((y01 - y00) + (y11 - y10))
    interaction = y11 - y10 - y01 + y00

    rows = []
    for effect_name, values, offset in (
        ("Qread_off_main_effect", q_effect, 1),
        ("Vbroadcast_off_main_effect", b_effect, 2),
        ("interaction", interaction, 3),
    ):
        mean, low, high = bootstrap_mean_ci(
            values,
            seed=seed + offset,
            n_boot=n_boot,
        )
        rows.append({
            "model_name": model_name,
            "token_mode": token_mode,
            "block": "" if block is None else int(block),
            "metric": metric,
            "effect": effect_name,
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
            "n_pairs": int(np.isfinite(values).sum()),
        })
    return rows


def build_factorial_effects(
    early: pd.DataFrame,
    final: pd.DataFrame,
    args,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    early_metrics = (
        "patch_delta_mu1_abs_mean",
        "cls_source_mu1_abs_l1_ratio",
        "mu1_delta_corr_cls_source",
        "cls_postattn_mu1_cos",
        "cls_postattn_mu2_cos",
    )
    final_metrics = (
        "pre_b13_fixed_reg_norm",
        "pre_b13_fixed_reg_mu1_cos",
        "final_fixed_reg_norm_mean",
        "final_fixed_reg_mu1_cos",
        "final_fixed_reg_mu2_cos",
        "final_register_jaccard_to_baseline",
        "embedding_cos_to_no_rn_baseline",
    )

    for mi, model_name in enumerate(args.models):
        for block in args.measure_blocks:
            for metric in early_metrics:
                rows.extend(
                    factorial_rows(
                        early,
                        model_name=model_name,
                        token_mode="pre_b13_shared",
                        metric=metric,
                        block=block,
                        seed=args.seed + 10000 * mi + 100 * block,
                        n_boot=args.bootstrap,
                    )
                )

        for ti, token_mode in enumerate((TOKEN_NO_RN, TOKEN_RN)):
            for metric in final_metrics:
                rows.extend(
                    factorial_rows(
                        final,
                        model_name=model_name,
                        token_mode=token_mode,
                        metric=metric,
                        block=None,
                        seed=args.seed + 50000 + 10000 * mi + 1000 * ti,
                        n_boot=args.bootstrap,
                    )
                )
    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================


def branch_label(qread: str, vbroadcast: str) -> str:
    return f"{qread}/{vbroadcast}"


def plot_early_2x2(early_summary: pd.DataFrame, pdir: Path) -> None:
    q = early_summary[early_summary["block"] == 9]
    if not len(q):
        return
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(16, 4.8), sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        z = q[q["model_name"] == model_name]
        labels, values = [], []
        for qread, vbroadcast in EARLY_BRANCHES:
            row = z[
                (z["qread"] == qread)
                & (z["vbroadcast"] == vbroadcast)
            ]
            if not len(row):
                continue
            labels.append(branch_label(qread, vbroadcast))
            values.append(float(row.iloc[0]["patch_delta_mu1_abs_mean_mean"]))
        ax.bar(labels, values)
        ax.set_title(model_name)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("B9 mean exact patch |Delta mu1|")
    fig.suptitle("B9 mu1 write under the early CLS 2x2")
    fig.tight_layout(rect=[0, 0, 1, .94])
    savefig(fig, pdir / "01_B9_mu1_2x2.png")


def plot_preb13_norm(final_summary: pd.DataFrame, pdir: Path) -> None:
    q = final_summary[final_summary["token_mode"] == TOKEN_NO_RN]
    if not len(q):
        return
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(16, 4.8), sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        z = q[q["model_name"] == model_name]
        labels, values = [], []
        for qread, vbroadcast in EARLY_BRANCHES:
            row = z[
                (z["qread"] == qread)
                & (z["vbroadcast"] == vbroadcast)
            ]
            if not len(row):
                continue
            labels.append(branch_label(qread, vbroadcast))
            values.append(float(row.iloc[0]["pre_b13_fixed_reg_norm_mean"]))
        ax.bar(labels, values)
        ax.set_title(model_name)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Pre-B13 fixed register-lineage norm")
    fig.suptitle("Does early CLS read/broadcast alter register isolation by B13?")
    fig.tight_layout(rect=[0, 0, 1, .94])
    savefig(fig, pdir / "02_preB13_register_lineage_norm_2x2.png")


def plot_b13_h5(late_summary: pd.DataFrame, pdir: Path) -> None:
    q = late_summary[
        (late_summary["block"] == 13)
        & (late_summary["head"] == 5)
        & (late_summary["qread"] == QREAD_ON)
        & (late_summary["vbroadcast"] == VBROADCAST_ON)
    ]
    if not len(q):
        return

    labels = []
    cls_reg = []
    cls_extra = []
    extra_reg = []
    for model_name in MODEL_ORDER:
        for token_mode in (TOKEN_NO_RN, TOKEN_RN, TOKEN_SWAP):
            row = q[
                (q["model_name"] == model_name)
                & (q["token_mode"] == token_mode)
            ]
            if not len(row):
                continue
            r = row.iloc[0]
            labels.append(f"{model_name}\n{token_mode}")
            cls_reg.append(float(r["slot0_to_b13reg_attn_mean"]))
            cls_extra.append(float(r["slot0_to_extra_attn_mean"]))
            extra_reg.append(float(r["extra_to_b13reg_attn_mean"]))

    x = np.arange(len(labels))
    width = .26
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.1), 5.4))
    ax.bar(x - width, cls_reg, width=width, label="slot0 -> B13 registers")
    ax.bar(x, cls_extra, width=width, label="slot0 -> extra token")
    ax.bar(x + width, extra_reg, width=width, label="extra token -> B13 registers")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean attention")
    ax.set_title("B13 H5: register interrogation, RN competition, and slot swap")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    savefig(fig, pdir / "03_B13_H5_register_vs_RN.png")


def plot_b20_h14(late_summary: pd.DataFrame, pdir: Path) -> None:
    q = late_summary[
        (late_summary["block"] == 20)
        & (late_summary["head"] == 14)
        & (late_summary["qread"] == QREAD_ON)
        & (late_summary["vbroadcast"] == VBROADCAST_ON)
    ]
    if not len(q):
        return

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(17, 5.2), sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        z = q[q["model_name"] == model_name]
        labels, reg_to_slot0, reg_to_extra, slot0_to_reg = [], [], [], []
        for token_mode in (TOKEN_NO_RN, TOKEN_RN, TOKEN_SWAP):
            row = z[z["token_mode"] == token_mode]
            if not len(row):
                continue
            r = row.iloc[0]
            labels.append(token_mode)
            reg_to_slot0.append(float(r["reg_query_to_slot0_attn_mean_mean"]))
            reg_to_extra.append(float(r["reg_query_to_extra_attn_mean_mean"]))
            slot0_to_reg.append(float(r["slot0_to_reg_attn_mean"]))
        x = np.arange(len(labels))
        width = .26
        ax.bar(x - width, slot0_to_reg, width=width, label="slot0 -> REG")
        ax.bar(x, reg_to_slot0, width=width, label="REG -> slot0")
        ax.bar(x + width, reg_to_extra, width=width, label="REG -> extra")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(model_name)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Mean attention")
    axes[-1].legend(fontsize=8)
    fig.suptitle("B20 H14 global exchange after RN insertion / slot swap")
    fig.tight_layout(rect=[0, 0, 1, .94])
    savefig(fig, pdir / "04_B20_H14_exchange.png")


def plot_final_2x2(final_summary: pd.DataFrame, pdir: Path, metric: str, title: str, filename: str) -> None:
    q = final_summary[final_summary["token_mode"].isin([TOKEN_NO_RN, TOKEN_RN])]
    if not len(q) or metric not in q.columns:
        return
    fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(17, 8.5), sharey="row")
    for row_index, token_mode in enumerate((TOKEN_NO_RN, TOKEN_RN)):
        for col_index, model_name in enumerate(MODEL_ORDER):
            ax = axes[row_index, col_index]
            z = q[
                (q["model_name"] == model_name)
                & (q["token_mode"] == token_mode)
            ]
            labels, values = [], []
            for qread, vbroadcast in EARLY_BRANCHES:
                one = z[
                    (z["qread"] == qread)
                    & (z["vbroadcast"] == vbroadcast)
                ]
                if not len(one):
                    continue
                labels.append(branch_label(qread, vbroadcast))
                values.append(float(one.iloc[0][metric]))
            ax.bar(labels, values)
            ax.tick_params(axis="x", rotation=25)
            ax.set_title(f"{model_name} / {token_mode}")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, .95])
    savefig(fig, pdir / filename)


def plot_swap(final_summary: pd.DataFrame, pdir: Path) -> None:
    q = final_summary[
        (final_summary["token_mode"] == TOKEN_SWAP)
        & (final_summary["qread"] == QREAD_ON)
        & (final_summary["vbroadcast"] == VBROADCAST_ON)
    ]
    if not len(q):
        return
    labels = q["model_name"].tolist()
    official = q["embedding_cos_to_no_rn_baseline_mean"].to_numpy(float)
    copied = q["extra_projected_embedding_cos_to_no_rn_baseline_mean"].to_numpy(float)
    x = np.arange(len(labels))
    width = .35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, official, width=width, label="official slot0 (RN trajectory)")
    ax.bar(x + width / 2, copied, width=width, label="appended copied-CLS trajectory")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Cosine to normal no-RN image embedding")
    ax.set_title("B13 RN/CLS slot swap: where did the semantic embedding go?")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    savefig(fig, pdir / "07_swap_token_embeddings.png")


def plot_precursor(precursor: pd.DataFrame, pdir: Path) -> None:
    if not len(precursor):
        return
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(17, 5), sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        z = precursor[precursor["model_name"] == model_name]
        for qread, vbroadcast in EARLY_BRANCHES:
            one = z[
                (z["qread"] == qread)
                & (z["vbroadcast"] == vbroadcast)
            ].sort_values("block")
            if len(one):
                ax.plot(
                    one["block"],
                    one["gap_projection_amplitude_on_baseline"],
                    marker="o",
                    label=branch_label(qread, vbroadcast),
                )
        ax.axhline(1.0, color=".5", lw=.8, ls="--")
        ax.set_title(model_name)
        ax.set_xlabel("Block")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Projection of register-vs-other MLP gap onto baseline gap")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Early 4096-D register-precursor signature under the CLS 2x2")
    fig.tight_layout(rect=[0, 0, 1, .94])
    savefig(fig, pdir / "08_mlp_precursor_signature.png")


# =============================================================================
# Report
# =============================================================================

def find_summary_row(
    frame: pd.DataFrame,
    **conditions,
) -> Optional[pd.Series]:
    query = frame
    for column, value in conditions.items():
        if column not in query.columns:
            return None
        query = query[query[column] == value]
    if not len(query):
        return None
    return query.iloc[0]


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return "nan"
    return f"{number:.{digits}f}" if np.isfinite(number) else "nan"


def write_summary(
    out: Path,
    early_summary: pd.DataFrame,
    late_summary: pd.DataFrame,
    final_summary: pd.DataFrame,
    precursor: pd.DataFrame,
    factorial: pd.DataFrame,
    oracle_audits: list[dict[str, Any]],
    rn_audit: dict[str, Any],
    args,
) -> None:
    lines = [
        "CLS <-> MU CAUSAL 2x2 + RN + SLOT SWAP",
        "=" * 78,
        "",
        "Intervention semantics:",
        f"  Qoff : CLS query may attend only to CLS on blocks {args.qread_blocks}.",
        f"  Boff : spatial queries may not use CLS as source on blocks {args.vbroadcast_blocks}.",
        "  Both masks are renormalized; all heads are affected.",
        "  RN is inserted only before B13, so RN/no-RN arms share B0--B12 exactly.",
        "",
        f"RN norm={rn_audit['norm']:.6f}, insert before B{rn_audit['experiment_insert_block']}.",
        "",
        "Fixed coordinate system:",
    ]
    for audit in oracle_audits:
        lines.append(
            f"  {audit['model_name']}: {audit['oracle_source']} "
            f"(sigma1={audit['sigma1']:.4g}, sigma2={audit['sigma2']:.4g})"
        )

    lines += [
        "",
        "B9 exact mu1 write:",
        "",
        "model                 Q/B              |Delta mu1|   CLS ratio   corr(total,CLS)",
        "-" * 82,
    ]
    for model_name in args.models:
        for qread, vbroadcast in EARLY_BRANCHES:
            row = find_summary_row(
                early_summary,
                model_name=model_name,
                qread=qread,
                vbroadcast=vbroadcast,
                block=9,
            )
            if row is None:
                continue
            lines.append(
                f"{model_name:<21} {branch_label(qread,vbroadcast):<16} "
                f"{fmt(row['patch_delta_mu1_abs_mean_mean']):>11} "
                f"{fmt(row['cls_source_mu1_abs_l1_ratio_mean']):>11} "
                f"{fmt(row['mu1_delta_corr_cls_source_mean']):>15}"
            )

    lines += [
        "",
        "Final register / embedding effects:",
        "",
        "model/token           Q/B              preB13 regN  final Jacc  emb cos->base",
        "-" * 86,
    ]
    for model_name in args.models:
        for token_mode in (TOKEN_NO_RN, TOKEN_RN):
            for qread, vbroadcast in EARLY_BRANCHES:
                row = find_summary_row(
                    final_summary,
                    model_name=model_name,
                    token_mode=token_mode,
                    qread=qread,
                    vbroadcast=vbroadcast,
                )
                if row is None:
                    continue
                lines.append(
                    f"{(model_name+'/'+token_mode):<21} {branch_label(qread,vbroadcast):<16} "
                    f"{fmt(row['pre_b13_fixed_reg_norm_mean']):>11} "
                    f"{fmt(row['final_register_jaccard_to_baseline_mean']):>11} "
                    f"{fmt(row['embedding_cos_to_no_rn_baseline_mean']):>13}"
                )

    lines += [
        "",
        "B13 H5 baseline branch:",
        "",
        "model/token           slot0 label  extra      slot0->B13REG  slot0->extra  extra->B13REG",
        "-" * 94,
    ]
    for model_name in args.models:
        for token_mode in (TOKEN_NO_RN, TOKEN_RN, TOKEN_SWAP):
            row = find_summary_row(
                late_summary,
                model_name=model_name,
                token_mode=token_mode,
                qread=QREAD_ON,
                vbroadcast=VBROADCAST_ON,
                block=13,
                head=5,
            )
            if row is None:
                continue
            lines.append(
                f"{(model_name+'/'+token_mode):<21} "
                f"{str(row['slot0_label']):<11} {str(row['extra_label']):<10} "
                f"{fmt(row['slot0_to_b13reg_attn_mean']):>15} "
                f"{fmt(row['slot0_to_extra_attn_mean']):>13} "
                f"{fmt(row['extra_to_b13reg_attn_mean']):>15}"
            )

    lines += [
        "",
        "Slot swap:",
        "",
        "model                 official RN-slot emb cos   copied-CLS emb cos   slot0 norm   extra norm",
        "-" * 93,
    ]
    for model_name in args.models:
        row = find_summary_row(
            final_summary,
            model_name=model_name,
            token_mode=TOKEN_SWAP,
            qread=QREAD_ON,
            vbroadcast=VBROADCAST_ON,
        )
        if row is None:
            continue
        lines.append(
            f"{model_name:<21} "
            f"{fmt(row['embedding_cos_to_no_rn_baseline_mean']):>24} "
            f"{fmt(row['extra_projected_embedding_cos_to_no_rn_baseline_mean']):>20} "
            f"{fmt(row['final_slot0_resid_norm_mean']):>12} "
            f"{fmt(row['final_extra_resid_norm_mean']):>12}"
        )

    lines += [
        "",
        "Interpretation guardrails:",
        "  * mu1/mu2 and fixed register addresses NEVER refit under intervention.",
        "  * Qoff and Boff are edge-routing interventions, not token deletion.",
        "  * RN is an appended B13 token only; bridge/router/correction are not executed.",
        "  * The slot-swap run is intentionally pathological and has no claim of preserving",
        "    the native architecture; both slot0 and copied-CLS projected embeddings are logged.",
        "  * The 4096-D precursor signature is a population difference vector, not a claim",
        "    that every coordinate in that vector is a dedicated register neuron.",
        "  * MU_FIELD is a loose signed-cos(mu1) role for late exchange diagnostics and is",
        "    not the stricter prior cache/scratchpad operational definition.",
        "",
    ]

    (out / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal CLS read/broadcast 2x2, RN repeat, and RN/CLS slot swap."
    )

    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--old_oracle_root", default=DEFAULT_OLD_ORACLE_ROOT)
    parser.add_argument(
        "--reuse_old_oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument(
        "--qread_blocks",
        default="0-5",
        help="Blocks where Qread OFF prevents CLS from reading spatial tokens.",
    )
    parser.add_argument(
        "--vbroadcast_blocks",
        default="6-10",
        help="Blocks where Vbroadcast OFF prevents spatial queries from consuming CLS.",
    )
    parser.add_argument(
        "--measure_blocks",
        default="0-10",
        help="Blocks receiving exact mu1/mu2 source-decomposition measurements.",
    )
    parser.add_argument(
        "--precursor_blocks",
        default="6-10",
        help="Blocks used for the 4096-D register-precursor MLP population signature.",
    )
    parser.add_argument("--focus_heads", default="5,11,12,14,15")
    parser.add_argument("--rn_insert_block", type=int, default=13)
    parser.add_argument("--mu_field_cos", type=float, default=0.50)

    parser.add_argument("--final_register_threshold", type=float, default=60.0)
    parser.add_argument("--final_register_min", type=int, default=1)
    parser.add_argument("--final_register_max", type=int, default=4)
    parser.add_argument("--b13_visible_threshold", type=float, default=70.0)

    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260909)

    parser.add_argument("--parity_max_abs", type=float, default=2e-5)
    parser.add_argument("--parity_min_cos", type=float, default=0.999999)
    parser.add_argument("--decomp_max_abs", type=float, default=5e-5)
    parser.add_argument(
        "--postprocess_only",
        action="store_true",
        help="Reuse completed extraction CSVs; do not load models or run the GPU.",
    )

    # Loader args expected by the supplied sink-script helpers.
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip_module", default="attnclip_mechinterp_sae")
    parser.add_argument("--model_spec", default="ViT-L/14")
    parser.add_argument(
        "--gmp_checkpoint",
        default=r"ViT-L-14-BEST-smooth-GmP-ft-pickle-OpenAI.pt",
    )
    parser.add_argument("--xattn_checkpoint", default=DEFAULT_XATTN_CHECKPOINT)
    parser.add_argument("--xattn_module", default="oaiclip")
    parser.add_argument("--pickle_module", default="clip")

    args = parser.parse_args()
    args.models = tuple(
        token.strip() for token in args.models.split(",") if token.strip()
    )
    args.qread_blocks = parse_ints(args.qread_blocks)
    args.vbroadcast_blocks = parse_ints(args.vbroadcast_blocks)
    args.measure_blocks = parse_ints(args.measure_blocks)
    args.precursor_blocks = parse_ints(args.precursor_blocks)
    args.focus_heads = parse_ints(args.focus_heads)

    unknown = set(args.models) - set(MODEL_ORDER)
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")

    for name, blocks in (
        ("qread_blocks", args.qread_blocks),
        ("vbroadcast_blocks", args.vbroadcast_blocks),
        ("measure_blocks", args.measure_blocks),
        ("precursor_blocks", args.precursor_blocks),
    ):
        if not blocks:
            parser.error(f"--{name} cannot be empty")
        if min(blocks) < 0:
            parser.error(f"--{name} contains a negative block")
        if max(blocks) >= args.rn_insert_block:
            parser.error(
                f"--{name} must be strictly before RN insertion B{args.rn_insert_block}"
            )
    if args.rn_insert_block != 13:
        print(
            f"[warning] user-requested/native experiment is B13; "
            f"you selected B{args.rn_insert_block}"
        )
    return args


def postprocess_existing_outputs(out: Path, args) -> None:
    """Rebuild summaries, figures, report, factorial tables and ZIP from completed CSVs."""
    required = {
        "early": out / "early_per_image.csv",
        "late": out / "late_focus_per_image.csv",
        "final": out / "final_per_image.csv",
        "precursor": out / "mlp_precursor_signature_summary.csv",
        "rn_audit": out / "rn_token_audit.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "--postprocess_only requires a completed extraction. Missing:\n  "
            + "\n  ".join(missing)
        )

    print("[postprocess] loading completed extraction outputs...")
    early = pd.read_csv(required["early"])
    late = pd.read_csv(required["late"])
    final = pd.read_csv(required["final"])
    precursor = pd.read_csv(required["precursor"])
    rn_audit = json.loads(required["rn_audit"].read_text(encoding="utf-8"))

    oracle_audits = []
    for model_name in args.models:
        path = out / model_name / "oracle_audit.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        oracle_audits.append(json.loads(path.read_text(encoding="utf-8")))

    print("[postprocess] rebuilding summaries...")
    early_summary = summarize_numeric(
        early,
        group_cols=("model_name", "qread", "vbroadcast", "block"),
        skip_cols=("stim_id", "source", "early_condition"),
    )
    late_summary = summarize_numeric(
        late,
        group_cols=(
            "model_name",
            "token_mode",
            "qread",
            "vbroadcast",
            "condition",
            "block",
            "head",
            "slot0_label",
            "extra_label",
        ),
        skip_cols=("stim_id", "source"),
    )
    final_summary = summarize_numeric(
        final,
        group_cols=(
            "model_name",
            "token_mode",
            "qread",
            "vbroadcast",
            "condition",
            "slot0_label",
            "extra_label",
        ),
        skip_cols=("stim_id", "source"),
    )

    early_summary.to_csv(out / "early_summary.csv", index=False)
    late_summary.to_csv(out / "late_focus_summary.csv", index=False)
    final_summary.to_csv(out / "final_summary.csv", index=False)

    factorial = build_factorial_effects(early, final, args)
    factorial.to_csv(out / "factorial_effects.csv", index=False)

    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("[postprocess] rebuilding plots...")
    plot_early_2x2(early_summary, plot_dir)
    plot_preb13_norm(final_summary, plot_dir)
    plot_b13_h5(late_summary, plot_dir)
    plot_b20_h14(late_summary, plot_dir)
    plot_final_2x2(
        final_summary,
        plot_dir,
        metric="final_register_jaccard_to_baseline_mean",
        title="Final register-address preservation under the CLS 2x2",
        filename="05_final_register_jaccard_2x2.png",
    )
    plot_final_2x2(
        final_summary,
        plot_dir,
        metric="embedding_cos_to_no_rn_baseline_mean",
        title="Final embedding preservation under the CLS 2x2",
        filename="06_embedding_cosine_2x2.png",
    )
    plot_swap(final_summary, plot_dir)
    plot_precursor(precursor, plot_dir)

    print("[postprocess] rebuilding report...")
    write_summary(
        out,
        early_summary,
        late_summary,
        final_summary,
        precursor,
        factorial,
        oracle_audits,
        rn_audit,
        args,
    )

    include = [
        out / "config.json",
        out / "rn_token_audit.json",
        out / "early_per_image.csv",
        out / "early_summary.csv",
        out / "late_focus_per_image.csv",
        out / "late_focus_summary.csv",
        out / "final_per_image.csv",
        out / "final_summary.csv",
        out / "mlp_precursor_signature_summary.csv",
        out / "factorial_effects.csv",
        out / "SUMMARY.txt",
    ]
    for model_name in args.models:
        include.append(out / model_name / "oracle_audit.json")
        recomputed = out / model_name / "baseline_oracle.npz"
        if recomputed.is_file():
            include.append(recomputed)
    include += sorted(plot_dir.glob("*.png"))

    zip_path = out / "compact_summary_workspace_cls_mu_causal.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=7,
    ) as archive:
        for path in include:
            if path.is_file():
                archive.write(path, arcname=path.relative_to(out).as_posix())

    print("[postprocess] done")
    print("[compact summary]", zip_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.postprocess_only:
        postprocess_existing_outputs(out, args)
        return

    base = _backbone_tools

    manifest_path = resolve_local(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    if args.max_images > 0:
        manifest = manifest.iloc[:args.max_images].copy().reset_index(drop=True)

    missing = [
        str(path)
        for path in manifest["path"].astype(str)
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest images are missing; first={missing[0]}"
        )

    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)

    save_json(out / "config.json", vars(args))

    rn_token, rn_audit = load_trained_rn_token(base, args)
    save_json(out / "rn_token_audit.json", rn_audit)
    print(
        f"[RN] shape={tuple(rn_token.shape)} norm={rn_audit['norm']:.6f} "
        f"insert_before_B{rn_audit['experiment_insert_block']}"
    )

    early_rows: list[dict[str, Any]] = []
    late_focus_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    oracle_audits: list[dict[str, Any]] = []
    parity_state: dict[str, bool] = {}

    precursor_acc: Optional[PrecursorAccumulator] = None

    for model_name in args.models:
        print(f"\n================ {model_name} ================")
        model_dir = out / model_name
        model_dir.mkdir(exist_ok=True)

        bundle = base.load_bundle(
            model_name,
            args,
            model_dir / "load_audit",
        )

        visual = bundle.model.visual
        width = int(visual.positional_embedding.shape[1])
        patches = int(visual.positional_embedding.shape[0] - 1)
        mlp_width = int(
            visual.transformer.resblocks[min(args.precursor_blocks)].mlp.c_fc.out_features
        )

        if rn_token.numel() != width:
            raise RuntimeError(
                f"RN width {rn_token.numel()} != {model_name} visual width {width}"
            )

        if precursor_acc is None:
            precursor_acc = PrecursorAccumulator(mlp_width)
        elif precursor_acc.width_mlp != mlp_width:
            raise RuntimeError("MLP width differs across models unexpectedly")

        oracle = None
        if args.reuse_old_oracle:
            oracle = try_load_old_oracle(
                Path(args.old_oracle_root),
                model_name,
                manifest,
                width,
                patches,
            )
            if oracle is not None:
                print(f"[oracle] reused {oracle.source}")

        if oracle is None:
            print("[oracle] previous oracle unavailable/mismatched; recomputing baseline")
            oracle = recompute_oracle(
                bundle,
                manifest,
                args,
                model_dir,
            )

        audit = {
            "model_name": model_name,
            "oracle_source": oracle.source,
            "sigma1": float(oracle.singular_values[0]),
            "sigma2": float(oracle.singular_values[1]),
            "mean_b23_register_count": float(
                np.asarray(oracle.b23_reg_mask).sum(axis=1).mean()
            ),
            "mean_b13_visible_register_count": float(
                np.asarray(oracle.b13_reg_mask).sum(axis=1).mean()
            ),
            "manifest_rows": int(len(manifest)),
        }
        oracle_audits.append(audit)
        save_json(model_dir / "oracle_audit.json", audit)

        for start in tqdm(
            range(0, len(manifest), args.batch_size),
            desc=model_name,
            unit="batch",
        ):
            chunk = manifest.iloc[start:start + args.batch_size]
            run_batch_model(
                base=base,
                bundle=bundle,
                model_name=model_name,
                manifest_chunk=chunk,
                global_start=start,
                oracle=oracle,
                rn_token_cpu=rn_token,
                args=args,
                precursor_acc=precursor_acc,
                early_rows=early_rows,
                late_focus_rows=late_focus_rows,
                final_rows=final_rows,
                parity_state=parity_state,
            )

        del bundle, oracle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if precursor_acc is None:
        raise RuntimeError("No models were processed")

    early = pd.DataFrame(early_rows)
    late = pd.DataFrame(late_focus_rows)
    final = pd.DataFrame(final_rows)
    precursor = pd.DataFrame(precursor_acc.summarize())

    early.to_csv(out / "early_per_image.csv", index=False)
    late.to_csv(out / "late_focus_per_image.csv", index=False)
    final.to_csv(out / "final_per_image.csv", index=False)
    precursor.to_csv(out / "mlp_precursor_signature_summary.csv", index=False)

    # Everything expensive is now on disk.  Use the same CPU-only postprocessor
    # for the normal path and for --postprocess_only recovery.
    postprocess_existing_outputs(out, args)


if __name__ == "__main__":
    main()
