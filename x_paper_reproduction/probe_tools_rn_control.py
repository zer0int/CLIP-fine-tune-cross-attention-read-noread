#!/usr/bin/env python3
'\nRN control-subspace knob suite — RTA-100 Multilingual\n======================================================\n\nDataset-level version of the B13 RN "control knob" experiment.\n\nDefault dataset:\n    zer0int/RTA-100-Multilingual\n\nDefault rendered languages:\n    en,de,ar,zh,ru\n\nThe analysis always pairs every rendered SynthRTA image with its exact NoRTA\ncontrol by `sample_key`.  This is important: NoRTA is *not* assumed to contain\nliterally zero incidental text anywhere in the scene.  The paired\nSynthRTA-minus-NoRTA contrast isolates the digitally rendered attack region.\n\nWhat this runs\n--------------\n1) alpha sweeps through paired/fixed rank-1 and rank-4 B13 RN components;\n2) token-population injection (CLS / native register / ordinary / bbox / outside);\n3) blockwise B13->late trajectory against each exact SynthRTA-NoRTA contrast;\n4) PC1..PC4 local observable response Jacobian.\n\nThe heavy visual intervention runs are reused across query modes.  By default\nwe probe:\n    english -> <text> canonical English attack_word_en\n    native  -> <text> rendered-language attack_word\n\nAll semantic image-vs-label dashboard logits use canonical English labels so\ncross-language comparisons are not confounded by the ordinary CLIP text tower.\n\nRuntime note\n------------\nThe HF repo contains 1000 rows per named config.  The default is a deterministic\n100-key paired mechanism sample per language.  Use --limit-per-language 0 for\nthe full shared-key intersection.\n\nOutputs\n-------\n  rn_rta_multilingual_control_knob/\n    data/\n      raw_*.csv                       # all tasty local data\n      *_summary.csv                   # aggregated mean/std/SEM/quantiles\n      paired_*_deltas.csv             # sample-level SynthRTA - NoRTA\n      basis.npz\n      config.json\n      sample_manifest.csv\n    plots/                            # deliberately small summary set\n    SUMMARY.txt\n    compact_summary_rn_control_knob.zip\n\nThe compact summary ZIP intentionally contains aggregate summaries + selected\nplots, not the full raw-row firehose.\n'
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()


import argparse
import csv
import gc
import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from x_paper_reproduction.rn_control_mechinterp.core import load_model, model_autocast_context


DEFAULT_CHECKPOINT = "REPLACE_WITH_CHECKPOINT.pt"
DEFAULT_DATASET_REPO = "zer0int/RTA-100-Multilingual"
DEFAULT_DATASET_ROOT = ""
DEFAULT_LANGUAGES = "en,de,ar,zh,ru"
DEFAULT_QUERY_MODES = "english,native"
DEFAULT_LIMIT = 100
DEFAULT_BASIS_PAIRS_PER_LANGUAGE = 6
DEFAULT_ALPHAS = "-2,-1.5,-1,-0.5,0,0.5,1,1.5,2"
DEFAULT_RANKS = "1,4"
DEFAULT_MODES = "paired,fixed"
DEFAULT_POPULATIONS = "all_ordinary,cls_only,register_only,ordinary_patches,nonregister_patches,text_region,outside_text"
DEFAULT_POPULATION_ALPHAS = "-1,1"
DEFAULT_TEMPLATES = "{label};a photo of {label};the image depicts {label};there is {label};a picture of {label}"
DEFAULT_SEED = 20260902

LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "ar": "Arabic",
    "zh": "Chinese",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
}


@dataclass
class B13Pair:
    sample_key: str
    condition: str
    early_states: dict[int, torch.Tensor]   # [B,T,D], pre-B13 taps
    x_pre_tbc: torch.Tensor                 # ordinary tokens entering B13
    base_post_tbc: torch.Tensor             # RN-off B13 output
    full_post_tbc: torch.Tensor             # RN-on B13 output incl RN
    delta_ord_btd: torch.Tensor             # [1,Tordinary,D]
    b12_register_mask: torch.Tensor         # [1,P]


@dataclass
class VisualBundle:
    pair: B13Pair
    bbox_patch_mask: torch.Tensor
    base_run: dict[str, Any]
    full_run: dict[str, Any]
    anchor_run: dict[str, Any]
    tag_run: dict[str, Any]
    alpha_runs: dict[tuple[str, int], tuple[dict[str, Any], list[dict[str, Any]]]]
    population_run: tuple[dict[str, Any], list[dict[str, Any]]]
    jacobian_run: tuple[dict[str, Any], list[dict[str, Any]]]


# =============================================================================
# IO / parsing
# =============================================================================

def stable_slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text)).strip("._") or "x"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Dataset
# =============================================================================

def load_subset(dataset_repo: str, dataset_root: str, subset: str) -> Dataset:
    if dataset_root.strip():
        path = Path(dataset_root) / "data" / f"{subset}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing local parquet: {path}")
        return load_dataset("parquet", data_files={"train": str(path)}, split="train")
    return load_dataset(dataset_repo, subset, split="train")


def index_by_sample_key(ds: Dataset, subset: str) -> dict[str, int]:
    keys = [str(x) for x in ds["sample_key"]]
    out = {k: i for i, k in enumerate(keys)}
    if len(out) != len(keys):
        raise RuntimeError(f"{subset}: duplicate sample_key values")
    return out


def prepare_datasets(
    dataset_repo: str,
    dataset_root: str,
    languages: list[str],
    limit: int,
    seed: int,
) -> tuple[Dataset, dict[str, Dataset], dict[str, int], dict[str, dict[str, int]], list[str]]:
    print("[data] loading norta")
    norta = load_subset(dataset_repo, dataset_root, "norta")
    attacks: dict[str, Dataset] = {}
    for lang in languages:
        print(f"[data] loading {lang}")
        attacks[lang] = load_subset(dataset_repo, dataset_root, lang)

    norta_idx = index_by_sample_key(norta, "norta")
    attack_idx = {lang: index_by_sample_key(ds, lang) for lang, ds in attacks.items()}

    shared = set(norta_idx)
    for lang in languages:
        shared &= set(attack_idx[lang])
    shared_keys = sorted(shared)

    if not shared_keys:
        raise RuntimeError("No sample_key intersection across norta and selected languages")

    if limit > 0 and len(shared_keys) > limit:
        rng = random.Random(seed)
        shared_keys = sorted(rng.sample(shared_keys, limit))

    print(f"[data] shared exact pairs: {len(shared_keys)}")
    print("[data] source rows:", {"norta": len(norta), **{k: len(v) for k, v in attacks.items()}})
    return norta, attacks, norta_idx, attack_idx, shared_keys


def get_row(ds: Dataset, index: int) -> dict[str, Any]:
    row = ds[int(index)]
    return dict(row)


def ensure_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        import io
        if image.get("bytes") is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
    raise TypeError(f"Unsupported image object: {type(image)!r}")


def preprocess_pil(preprocess: Any, pil: Image.Image, device: torch.device) -> torch.Tensor:
    return preprocess(pil.convert("RGB")).unsqueeze(0).to(device=device, non_blocking=True)


# =============================================================================
# Geometry: source bbox -> CLIP patch grid
# =============================================================================

def resized_center_crop_bbox(
    bbox_xyxy: Iterable[float],
    original_size: tuple[int, int],
    target: int,
) -> tuple[float, float, float, float]:
    """Map bbox through standard CLIP Resize(shorter-side=target)+CenterCrop(target)."""
    w, h = original_size
    if w <= 0 or h <= 0:
        raise ValueError(original_size)
    scale = float(target) / float(min(w, h))
    rw, rh = float(w) * scale, float(h) * scale
    crop_x = max(0.0, (rw - target) / 2.0)
    crop_y = max(0.0, (rh - target) / 2.0)

    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    x0 = x0 * scale - crop_x
    x1 = x1 * scale - crop_x
    y0 = y0 * scale - crop_y
    y1 = y1 * scale - crop_y
    x0, x1 = max(0.0, x0), min(float(target), x1)
    y0, y1 = max(0.0, y0), min(float(target), y1)
    return x0, y0, x1, y1


def bbox_patch_mask(
    bbox_xyxy: Iterable[float],
    original_size: tuple[int, int],
    target: int,
    patch: int,
    device: torch.device,
) -> torch.Tensor:
    side = target // patch
    x0, y0, x1, y1 = resized_center_crop_bbox(bbox_xyxy, original_size, target)
    mask = torch.zeros(side * side, dtype=torch.float32, device=device)
    if x1 <= x0 or y1 <= y0:
        return mask

    gx0 = max(0, min(side - 1, int(math.floor(x0 / patch))))
    gy0 = max(0, min(side - 1, int(math.floor(y0 / patch))))
    gx1 = max(0, min(side - 1, int(math.ceil(x1 / patch) - 1)))
    gy1 = max(0, min(side - 1, int(math.ceil(y1 / patch) - 1)))

    for gy in range(gy0, gy1 + 1):
        for gx in range(gx0, gx1 + 1):
            mask[gy * side + gx] = 1.0
    return mask


# =============================================================================
# Math
# =============================================================================

def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x.float(), dim=dim, eps=eps)


def safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    aa = a.float().reshape(-1)
    bb = b.float().reshape(-1)
    den = aa.norm() * bb.norm()
    if float(den) < eps:
        return float("nan")
    return float((aa @ bb / den).detach().cpu())


def safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < eps:
        return float("nan")
    return float(num / den)


def project_last_dim(x: torch.Tensor, basis_kd: torch.Tensor, rank: int) -> torch.Tensor:
    k = min(int(rank), int(basis_kd.shape[0]))
    b = basis_kd[:k].float()
    q = torch.linalg.qr(b.T, mode="reduced").Q
    xf = x.float()
    return (xf @ q) @ q.T


def feature_subspace_fraction(x: torch.Tensor, basis_kd: torch.Tensor, rank: int = 4) -> float:
    den = float(x.float().square().sum().detach().cpu())
    if den < 1e-12:
        return float("nan")
    proj = project_last_dim(x, basis_kd, rank)
    return float(proj.square().sum().detach().cpu()) / den


def embedding_effect_recovery_batch(
    anchor: torch.Tensor,
    exact: torch.Tensor,
    intervention: torch.Tensor,
) -> torch.Tensor:
    """Returns [B] recovery for batched interventions."""
    a = anchor.float()
    e = exact.float()
    y = intervention.float()
    if a.shape[0] == 1 and y.shape[0] != 1:
        a = a.expand(y.shape[0], -1)
    if e.shape[0] == 1 and y.shape[0] != 1:
        e = e.expand(y.shape[0], -1)
    target = e - a
    delta = y - a
    den = target.square().sum(dim=-1).clamp_min(1e-12)
    return (delta * target).sum(dim=-1) / den


# =============================================================================
# Text-side cached query and semantic banks
# =============================================================================

def prepare_read_queries(
    model: torch.nn.Module,
    clip_mod: Any,
    candidate: str,
    device: torch.device,
) -> dict[str, torch.Tensor | bool]:
    toks = clip_mod.tokenize([f"<text> {candidate}"]).to(device)
    with torch.no_grad(), model_autocast_context(model):
        prepared = model.prepare_mode_tokens(toks)
        read_tokens = prepared["read_tokens"]
        cand = model._encode_text_hidden(read_tokens)
        null_tokens = model._null_read_tokens(device)
        null = model._encode_text_hidden(null_tokens)
    return {
        "candidate_query": cand["eot_hidden_pre_ln"].detach(),
        "candidate_text_embedding": cand["text_embedding"].detach(),
        "candidate_is_null": bool(model.is_null_candidate(read_tokens)[0].item()) if hasattr(model, "is_null_candidate") else False,
        "null_query": null["eot_hidden_pre_ln"].detach(),
        "null_text_embedding": null["text_embedding"].detach(),
    }


def prompt_bank_embedding(
    model: torch.nn.Module,
    clip_mod: Any,
    label: str,
    templates: list[str],
    device: torch.device,
) -> torch.Tensor:
    prompts = [t.format(label=label) for t in templates]
    toks = clip_mod.tokenize(prompts).to(device)
    with torch.no_grad(), model_autocast_context(model):
        feat = model.encode_text(toks).float()
    return normalize(normalize(feat).mean(dim=0, keepdim=True))


# =============================================================================
# B13 pair and downstream
# =============================================================================

def append_rn(model: torch.nn.Module, x_tbc: torch.Tensor) -> torch.Tensor:
    rn = model.visual.read_null_token.to(device=x_tbc.device, dtype=x_tbc.dtype)
    rn = rn.view(1, 1, -1).expand(1, x_tbc.shape[1], -1)
    return torch.cat([x_tbc, rn], dim=0)


def build_b13_pair(
    model: torch.nn.Module,
    images: torch.Tensor,
    sample_key: str,
    condition: str,
    needed_early_blocks: set[int],
) -> B13Pair:
    visual = model.visual
    ib = int(visual.read_null_insert_block)

    with torch.no_grad(), model_autocast_context(model):
        x = visual._prepare_tokens(images.type(model.dtype))
        early: dict[int, torch.Tensor] = {}
        for i in range(ib):
            x = visual.transformer.resblocks[i](x)
            if i in needed_early_blocks:
                early[i] = x.permute(1, 0, 2).detach().clone()

        x_pre = x.detach().clone()
        blk = visual.transformer.resblocks[ib]
        base_post = blk(x_pre)
        full_post = blk(append_rn(model, x_pre))

    delta = (full_post[:-1] - base_post).permute(1, 0, 2).float()
    spatial_pre = x_pre.permute(1, 0, 2)[:, 1:, :]
    norms = spatial_pre.float().norm(dim=-1)
    reg = visual._make_implicit_register_mask(
        norms.detach(),
        register_threshold=70.0,
        max_registers=8,
        min_registers=1,
    ).to(device=images.device)

    return B13Pair(
        sample_key=sample_key,
        condition=condition,
        early_states=early,
        x_pre_tbc=x_pre,
        base_post_tbc=base_post.detach().clone(),
        full_post_tbc=full_post.detach().clone(),
        delta_ord_btd=delta.detach().clone(),
        b12_register_mask=reg.detach().clone(),
    )


def repeat_early_states(
    early_states: dict[int, torch.Tensor],
    batch: int,
) -> dict[int, torch.Tensor]:
    out = {}
    for b, s in early_states.items():
        if s.shape[0] == batch:
            out[b] = s
        elif s.shape[0] == 1:
            out[b] = s.expand(batch, -1, -1)
        else:
            raise ValueError(f"Cannot expand early state B={s.shape[0]} to B={batch}")
    return out


def run_downstream(
    model: torch.nn.Module,
    post_b13_tbc: torch.Tensor,
    early_states: dict[int, torch.Tensor],
    capture_blocks: set[int],
    has_rn: bool,
) -> dict[str, Any]:
    visual = model.visual
    ib = int(visual.read_null_insert_block)
    batch = int(post_b13_tbc.shape[1])
    states = repeat_early_states(early_states, batch)
    x = post_b13_tbc
    if ib in capture_blocks:
        states[ib] = x.permute(1, 0, 2).detach().clone()

    with torch.no_grad(), model_autocast_context(model):
        for i in range(ib + 1, len(visual.transformer.resblocks)):
            x = visual.transformer.resblocks[i](x)
            if i in capture_blocks:
                states[i] = x.permute(1, 0, 2).detach().clone()
        embedding = visual._finalize_cls(x).float().detach()

    final = x.permute(1, 0, 2).detach()
    spatial = final[:, 1:-1, :] if has_rn else final[:, 1:, :]
    norms = spatial.float().norm(dim=-1)
    reg = visual._make_implicit_register_mask(
        norms.detach(),
        register_threshold=70.0,
        max_registers=8,
        min_registers=1,
    ).to(device=final.device)

    return {
        "states": states,
        "embedding": embedding,
        "final_tokens": final,
        "register_mask": reg,
        "has_rn": has_rn,
    }


def aligned_ordinary_state(state_btd: torch.Tensor, has_rn: bool) -> torch.Tensor:
    return state_btd[:, :-1, :] if has_rn else state_btd


def make_custom_post_batch(
    pair: B13Pair,
    specs: list[dict[str, Any]],
) -> torch.Tensor:
    """Each spec: component [1,T,D], alpha, optional token_mask [T]."""
    base = pair.base_post_tbc.permute(1, 0, 2).float()
    ordinary_rows = []
    for spec in specs:
        comp = spec["component"].float()
        mask = spec.get("token_mask")
        if mask is not None:
            comp = comp * mask.to(comp.device, comp.dtype)[None, :, None]
        ordinary_rows.append(base + float(spec["alpha"]) * comp)

    ordinary_btd = torch.cat(ordinary_rows, dim=0)
    ordinary_tbc = ordinary_btd.to(pair.base_post_tbc.dtype).permute(1, 0, 2)
    rn = pair.full_post_tbc[-1:].expand(-1, len(specs), -1)
    return torch.cat([ordinary_tbc, rn], dim=0)


# =============================================================================
# Basis
# =============================================================================

def fit_basis_from_dataset(
    model: torch.nn.Module,
    preprocess: Any,
    norta: Dataset,
    attacks: dict[str, Dataset],
    norta_idx: dict[str, int],
    attack_idx: dict[str, dict[str, int]],
    keys: list[str],
    languages: list[str],
    pairs_per_language: int,
    needed_early: set[int],
    max_rank: int,
    device: torch.device,
) -> dict[str, Any]:
    if pairs_per_language <= 0:
        basis_keys = keys
    else:
        basis_keys = keys[: min(len(keys), pairs_per_language)]

    rows = []
    templates = []
    used = []

    # NoRTA once per key.
    for key in basis_keys:
        row = get_row(norta, norta_idx[key])
        pil = ensure_pil(row["image"])
        img = preprocess_pil(preprocess, pil, device)
        pair = build_b13_pair(model, img, key, "norta", needed_early)
        d = pair.delta_ord_btd[0].float()
        rows.append(d)
        templates.append(d)
        used.append(f"norta::{key}")

    # One attacked image per selected language/key.
    for lang in languages:
        for key in basis_keys:
            row = get_row(attacks[lang], attack_idx[lang][key])
            pil = ensure_pil(row["image"])
            img = preprocess_pil(preprocess, pil, device)
            pair = build_b13_pair(model, img, key, lang, needed_early)
            d = pair.delta_ord_btd[0].float()
            rows.append(d)
            templates.append(d)
            used.append(f"{lang}::{key}")

    x = torch.cat(rows, dim=0)
    q = min(max(int(max_rank) + 8, 12), x.shape[0], x.shape[1])
    _, s, v = torch.pca_lowrank(x, q=q, center=False, niter=6)
    basis = v[:, :max_rank].T.contiguous().float()
    total = float(x.square().sum().detach().cpu())
    explained = (s[:max_rank].square() / max(total, 1e-12)).detach().cpu().numpy()
    mean_template = torch.stack(templates, dim=0).mean(dim=0).float()

    return {
        "basis_kd": basis.detach(),
        "singular_values": s[:max_rank].detach(),
        "explained_energy": explained,
        "mean_template_td": mean_template.detach(),
        "fit_items": used,
        "n_rows": int(x.shape[0]),
    }


# =============================================================================
# Bridge observables
# =============================================================================

def mean_read_null_per_block_batch(details: dict[str, Any]) -> dict[int, torch.Tensor]:
    out = {}
    for block, info in details.get("per_block", {}).items():
        att = info.get("read_null_attention")
        if att is None:
            continue
        # [B,N,H] -> [B]
        dims = tuple(range(1, att.ndim))
        out[int(block)] = att.float().mean(dim=dims).detach()
    return out


def score_run_batch(
    model: torch.nn.Module,
    run: dict[str, Any],
    queries: dict[str, torch.Tensor | bool],
    candidate_semantic_en: torch.Tensor,
    object_semantic_en: torch.Tensor,
) -> list[dict[str, float]]:
    states = run["states"]
    register_mask = run["register_mask"]
    implant = model.read_implant

    with torch.no_grad():
        cand_feat, cand_details = implant.read_features(
            states,
            queries["candidate_query"],
            register_mask=register_mask,
            return_details=True,
        )
        null_feat, null_details = implant.read_features(
            states,
            queries["null_query"],
            register_mask=register_mask,
            return_details=True,
        )
        cand_text = normalize(queries["candidate_text_embedding"])
        null_text = normalize(queries["null_text_embedding"])
        scale = model.logit_scale.float().exp()

        cand_raw = scale * torch.einsum("bnd,nd->bn", normalize(cand_feat), cand_text)
        null_raw = scale * torch.einsum("bnd,nd->bn", normalize(null_feat), null_text)
        source_logits = implant.presence_logits(states)

        bsz = cand_raw.shape[0]
        cand_mask = torch.full(
            (1,),
            bool(queries["candidate_is_null"]),
            device=cand_raw.device,
            dtype=torch.bool,
        )
        null_mask = torch.ones(1, device=null_raw.device, dtype=torch.bool)
        cand_cal = implant.calibrate_read_logits(cand_raw, source_logits, null_mask=cand_mask)
        null_cal = implant.calibrate_read_logits(null_raw, source_logits, null_mask=null_mask)

        ortho = implant.orthographic_features(states, queries["candidate_query"])
        ortho_logit = scale * torch.einsum("bnd,nd->bn", normalize(ortho), cand_text)

        emb = normalize(run["embedding"])
        cand_img = scale * (emb @ candidate_semantic_en.T)
        obj_img = scale * (emb @ object_semantic_en.T)

        source_probs = source_logits.sigmoid()
        source_gate = source_probs[:, 0] * source_probs[:, 1]

    cand_rn = mean_read_null_per_block_batch(cand_details)
    null_rn = mean_read_null_per_block_batch(null_details)
    blocks = sorted(set(cand_rn) | set(null_rn))

    rows = []
    for i in range(int(cand_raw.shape[0])):
        row = {
            "candidate_raw": float(cand_raw[i, 0].cpu()),
            "null_raw": float(null_raw[i, 0].cpu()),
            "relative_raw": float((cand_raw - null_raw)[i, 0].cpu()),
            "candidate_calibrated": float(cand_cal[i, 0].cpu()),
            "null_calibrated": float(null_cal[i, 0].cpu()),
            "relative_calibrated": float((cand_cal - null_cal)[i, 0].cpu()),
            "early_ortho": float(ortho_logit[i, 0].cpu()),
            "source_gate": float(source_gate[i].cpu()),
            "image_attack_en_logit": float(cand_img[i, 0].cpu()),
            "image_object_en_logit": float(obj_img[i, 0].cpu()),
            "attack_minus_object_en": float((cand_img - obj_img)[i, 0].cpu()),
            "final_register_count": int(run["register_mask"][i].sum().cpu()),
            "candidate_read_null_mixed": float(cand_details["read_null_attention"][i, 0].cpu())
                if cand_details.get("read_null_attention") is not None else float("nan"),
            "null_read_null_mixed": float(null_details["read_null_attention"][i, 0].cpu())
                if null_details.get("read_null_attention") is not None else float("nan"),
        }
        for b in blocks:
            row[f"candidate_read_null_B{b}"] = float(cand_rn[b][i].cpu()) if b in cand_rn else float("nan")
            row[f"null_read_null_B{b}"] = float(null_rn[b][i].cpu()) if b in null_rn else float("nan")
        rows.append(row)
    return rows


# =============================================================================
# Population masks
# =============================================================================

def build_population_masks(
    pair: B13Pair,
    bbox_mask_p: torch.Tensor,
) -> dict[str, torch.Tensor]:
    t = int(pair.delta_ord_btd.shape[1])
    p = int(bbox_mask_p.numel())
    if t != p + 1:
        raise RuntimeError(f"Expected CLS+{p} patches = {p+1} ordinary tokens, got {t}")

    device = pair.delta_ord_btd.device
    all_ord = torch.ones(t, dtype=torch.float32, device=device)

    cls = torch.zeros_like(all_ord)
    cls[0] = 1.0

    patches = torch.zeros_like(all_ord)
    patches[1:] = 1.0

    reg = torch.zeros_like(all_ord)
    reg_idx = torch.nonzero(pair.b12_register_mask[0], as_tuple=False).flatten()
    for idx in reg_idx.tolist():
        reg[1 + int(idx)] = 1.0

    nonreg = patches * (1.0 - reg)

    text_region = torch.zeros_like(all_ord)
    text_region[1:] = bbox_mask_p.to(device=device, dtype=torch.float32)
    outside_text = patches * (1.0 - text_region)

    return {
        "all_ordinary": all_ord,
        "cls_only": cls,
        "register_only": reg,
        "ordinary_patches": patches,
        "nonregister_patches": nonreg,
        "text_region": text_region,
        "outside_text": outside_text,
    }


# =============================================================================
# Build all visual interventions once per image
# =============================================================================

def make_visual_bundle(
    model: torch.nn.Module,
    pair: B13Pair,
    bbox_mask_p: torch.Tensor,
    basis: torch.Tensor,
    mean_template: torch.Tensor,
    ranks: list[int],
    modes: list[str],
    alphas: list[float],
    populations: list[str],
    population_alphas: list[float],
    population_rank: int,
    pc_count: int,
    jac_eps: float,
    all_capture: set[int],
) -> VisualBundle:
    base_run = run_downstream(model, pair.base_post_tbc, pair.early_states, all_capture, has_rn=False)
    full_run = run_downstream(model, pair.full_post_tbc, pair.early_states, all_capture, has_rn=True)
    tag_run = run_downstream(model, pair.full_post_tbc[:-1], pair.early_states, all_capture, has_rn=False)

    zero = torch.zeros_like(pair.delta_ord_btd)
    anchor_post = make_custom_post_batch(pair, [{"component": zero, "alpha": 0.0}])
    anchor_run = run_downstream(model, anchor_post, pair.early_states, all_capture, has_rn=True)

    paired_components = {r: project_last_dim(pair.delta_ord_btd, basis, r) for r in ranks}
    fixed_btd = mean_template[None, :, :]
    fixed_components = {r: project_last_dim(fixed_btd, basis, r) for r in ranks}

    alpha_runs = {}
    for mode in modes:
        if mode not in {"paired", "fixed"}:
            raise ValueError(f"Unknown mode: {mode}")
        for rank in ranks:
            component = paired_components[rank] if mode == "paired" else fixed_components[rank]
            specs = [{"component": component, "alpha": a} for a in alphas]
            post = make_custom_post_batch(pair, specs)
            run = run_downstream(model, post, pair.early_states, all_capture, has_rn=True)
            meta = [{"mode": mode, "rank": rank, "alpha": a} for a in alphas]
            alpha_runs[(mode, rank)] = (run, meta)

    masks = build_population_masks(pair, bbox_mask_p)
    if population_rank not in paired_components:
        pop_component = project_last_dim(pair.delta_ord_btd, basis, population_rank)
    else:
        pop_component = paired_components[population_rank]

    pop_specs = []
    pop_meta = []
    for pop in populations:
        if pop not in masks:
            raise ValueError(f"Unknown population {pop!r}; available={sorted(masks)}")
        for a in population_alphas:
            pop_specs.append({
                "component": pop_component,
                "alpha": a,
                "token_mask": masks[pop],
            })
            pop_meta.append({
                "population": pop,
                "alpha": a,
                "rank": population_rank,
                "population_token_count": int(masks[pop].sum().item()),
            })
    pop_post = make_custom_post_batch(pair, pop_specs)
    pop_run = run_downstream(model, pop_post, pair.early_states, all_capture, has_rn=True)

    jac_specs = []
    jac_meta = []
    for pc0 in range(min(pc_count, int(basis.shape[0]))):
        u = basis[pc0:pc0+1]
        pc_comp = project_last_dim(pair.delta_ord_btd, u, 1)
        pc_norm = float(pc_comp.norm(dim=-1).mean().detach().cpu())
        for sign in (-1.0, 1.0):
            jac_specs.append({"component": pc_comp, "alpha": sign * jac_eps})
            jac_meta.append({
                "pc": pc0 + 1,
                "sign": sign,
                "alpha": sign * jac_eps,
                "pc_component_norm_mean": pc_norm,
            })
    jac_post = make_custom_post_batch(pair, jac_specs)
    jac_run = run_downstream(model, jac_post, pair.early_states, all_capture, has_rn=True)

    return VisualBundle(
        pair=pair,
        bbox_patch_mask=bbox_mask_p,
        base_run=base_run,
        full_run=full_run,
        anchor_run=anchor_run,
        tag_run=tag_run,
        alpha_runs=alpha_runs,
        population_run=(pop_run, pop_meta),
        jacobian_run=(jac_run, jac_meta),
    )


# =============================================================================
# Score a visual bundle for one query mode
# =============================================================================

JAC_OBSERVABLES = [
    "relative_calibrated",
    "candidate_read_null_B20",
    "candidate_read_null_B21",
    "null_read_null_B21",
    "early_ortho",
    "image_attack_en_logit",
    "image_object_en_logit",
    "attack_minus_object_en",
]


def score_visual_bundle(
    model: torch.nn.Module,
    bundle: VisualBundle,
    queries: dict[str, torch.Tensor | bool],
    attack_sem_en: torch.Tensor,
    object_sem_en: torch.Tensor,
    common_meta: dict[str, Any],
    ranks: list[int],
    modes: list[str],
    alphas: list[float],
    population_rank: int,
    jac_eps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    alpha_rows = []
    pop_rows = []
    jac_rows = []
    ref_rows = []

    full_obs = score_run_batch(model, bundle.full_run, queries, attack_sem_en, object_sem_en)[0]
    anchor_obs = score_run_batch(model, bundle.anchor_run, queries, attack_sem_en, object_sem_en)[0]

    for name, run in (
        ("persistent_rn", bundle.full_run),
        ("rn_context_alpha0", bundle.anchor_run),
    ):
        obs = full_obs if name == "persistent_rn" else anchor_obs
        rr = dict(common_meta)
        rr["reference"] = name
        rr.update(obs)
        ref_rows.append(rr)

    # no-RN/tag-only cannot be passed through READ with read_null_enabled, but
    # their embedding geometry is still valuable.
    for name, run in (("no_rn", bundle.base_run), ("b13_tag_only", bundle.tag_run)):
        rr = dict(common_meta)
        rr["reference"] = name
        rr["embedding_cos_to_no_rn"] = float(
            (normalize(run["embedding"]) * normalize(bundle.base_run["embedding"])).sum().cpu()
        )
        rr["embedding_cos_to_persistent_rn"] = float(
            (normalize(run["embedding"]) * normalize(bundle.full_run["embedding"])).sum().cpu()
        )
        ref_rows.append(rr)

    for mode in modes:
        for rank in ranks:
            run, meta = bundle.alpha_runs[(mode, rank)]
            obs_rows = score_run_batch(model, run, queries, attack_sem_en, object_sem_en)
            rec = embedding_effect_recovery_batch(
                bundle.anchor_run["embedding"],
                bundle.full_run["embedding"],
                run["embedding"],
            ).cpu().numpy()

            component = (
                project_last_dim(bundle.pair.delta_ord_btd, BASIS_GLOBAL, rank)
                if mode == "paired"
                else project_last_dim(MEAN_TEMPLATE_GLOBAL[None, :, :], BASIS_GLOBAL, rank)
            )
            comp_norm = float(component.norm(dim=-1).mean().detach().cpu())

            for i, (m, obs) in enumerate(zip(meta, obs_rows)):
                row = dict(common_meta)
                row.update(m)
                row["component_norm_mean"] = comp_norm
                row["injected_norm_mean"] = abs(float(m["alpha"])) * comp_norm
                row.update(obs)
                row["embedding_effect_recovery"] = float(rec[i])

                for key in ("relative_calibrated", "candidate_read_null_B21", "attack_minus_object_en"):
                    denom = float(full_obs.get(key, float("nan"))) - float(anchor_obs.get(key, float("nan")))
                    row[key + "_recovery"] = safe_ratio(
                        float(obs.get(key, float("nan"))) - float(anchor_obs.get(key, float("nan"))),
                        denom,
                    )
                alpha_rows.append(row)

    pop_run, pop_meta = bundle.population_run
    pop_obs = score_run_batch(model, pop_run, queries, attack_sem_en, object_sem_en)
    pop_rec = embedding_effect_recovery_batch(
        bundle.anchor_run["embedding"],
        bundle.full_run["embedding"],
        pop_run["embedding"],
    ).cpu().numpy()

    for i, (m, obs) in enumerate(zip(pop_meta, pop_obs)):
        row = dict(common_meta)
        row.update(m)
        row.update(obs)
        row["embedding_effect_recovery"] = float(pop_rec[i])
        for key in ("relative_calibrated", "candidate_read_null_B21", "attack_minus_object_en"):
            denom = float(full_obs.get(key, float("nan"))) - float(anchor_obs.get(key, float("nan")))
            row[key + "_recovery"] = safe_ratio(
                float(obs.get(key, float("nan"))) - float(anchor_obs.get(key, float("nan"))),
                denom,
            )
        pop_rows.append(row)

    jac_run, jac_meta = bundle.jacobian_run
    jac_obs = score_run_batch(model, jac_run, queries, attack_sem_en, object_sem_en)
    by_pc: dict[int, dict[float, tuple[dict[str, Any], dict[str, float]]]] = {}
    for m, obs in zip(jac_meta, jac_obs):
        by_pc.setdefault(int(m["pc"]), {})[float(m["sign"])] = (m, obs)

    for pc, pm in sorted(by_pc.items()):
        if -1.0 not in pm or 1.0 not in pm:
            continue
        m_minus, obs_minus = pm[-1.0]
        m_plus, obs_plus = pm[1.0]
        pc_norm = float(m_plus["pc_component_norm_mean"])
        for observable in JAC_OBSERVABLES:
            vm = float(obs_minus.get(observable, float("nan")))
            vp = float(obs_plus.get(observable, float("nan")))
            v0 = float(anchor_obs.get(observable, float("nan")))
            deriv = (vp - vm) / (2.0 * jac_eps)
            curv = (vp + vm - 2.0 * v0) / (jac_eps * jac_eps)
            row = dict(common_meta)
            row.update({
                "pc": pc,
                "eps": jac_eps,
                "observable": observable,
                "minus": vm,
                "anchor": v0,
                "plus": vp,
                "derivative": deriv,
                "curvature": curv,
                "pc_component_norm_mean": pc_norm,
                "derivative_per_component_norm": deriv / max(pc_norm, 1e-12),
            })
            jac_rows.append(row)

    return alpha_rows, pop_rows, jac_rows, ref_rows


# These globals are assigned in main solely to avoid re-materializing the basis
# for every score_visual_bundle call.
BASIS_GLOBAL: torch.Tensor
MEAN_TEMPLATE_GLOBAL: torch.Tensor


# =============================================================================
# Trajectory
# =============================================================================

def pick_alpha_run(
    bundle: VisualBundle,
    mode: str,
    rank: int,
    alpha: float,
) -> tuple[dict[str, Any], int]:
    run, meta = bundle.alpha_runs[(mode, rank)]
    idx = min(range(len(meta)), key=lambda i: abs(float(meta[i]["alpha"]) - float(alpha)))
    if abs(float(meta[idx]["alpha"]) - float(alpha)) > 1e-6:
        raise RuntimeError(f"Requested trajectory alpha {alpha} not present")
    return run, idx


def trajectory_rows_for_condition(
    bundle: VisualBundle,
    attack_contrast: dict[int, torch.Tensor],
    common_meta: dict[str, Any],
    basis: torch.Tensor,
    mode: str,
    rank: int,
    alpha: float,
    trajectory_blocks: set[int],
) -> list[dict[str, Any]]:
    run, idx = pick_alpha_run(bundle, mode, rank, alpha)
    rows = []

    for b in sorted(trajectory_blocks):
        if b not in run["states"] or b not in bundle.anchor_run["states"] or b not in bundle.full_run["states"]:
            continue

        s = aligned_ordinary_state(run["states"][b][idx:idx+1], True).float()
        a = aligned_ordinary_state(bundle.anchor_run["states"][b], True).float()
        f = aligned_ordinary_state(bundle.full_run["states"][b], True).float()
        d = s - a
        exact_d = f - a
        patchmean = d[:, 1:, :].mean(dim=1)

        row = dict(common_meta)
        row.update({
            "mode": mode,
            "rank": rank,
            "alpha": alpha,
            "block": b,
            "delta_norm_mean": float(d.norm(dim=-1).mean().cpu()),
            "flat_cos_exact_rn_delta": safe_cos(d, exact_d),
            "pc1_patchmean_cos": safe_cos(patchmean, basis[0][None, :]),
            "subspace_fraction_rank4": feature_subspace_fraction(
                d, basis, rank=min(4, basis.shape[0])
            ),
        })

        if b in attack_contrast:
            c = attack_contrast[b].float()
            cpatch = c[:, 1:, :].mean(dim=1)
            row["flat_cos_attack_contrast"] = safe_cos(d, c)
            row["patchmean_cos_attack_contrast"] = safe_cos(patchmean, cpatch)
        else:
            row["flat_cos_attack_contrast"] = float("nan")
            row["patchmean_cos_attack_contrast"] = float("nan")
        rows.append(row)

    return rows


# =============================================================================
# Pair rows SynthRTA - NoRTA
# =============================================================================

def pair_delta_rows(
    norta_rows: list[dict[str, Any]],
    synth_rows: list[dict[str, Any]],
    key_fields: list[str],
    identity_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    nmap = {tuple(r.get(k) for k in key_fields): r for r in norta_rows}
    smap = {tuple(r.get(k) for k in key_fields): r for r in synth_rows}
    common = sorted(set(nmap) & set(smap), key=str)

    out = []
    for key in common:
        n = nmap[key]
        s = smap[key]
        row = dict(identity_meta)
        for k, v in zip(key_fields, key):
            row[k] = v

        # Keep selected raw endpoints for interpretation.
        for name in (
            "relative_calibrated",
            "candidate_read_null_B20",
            "candidate_read_null_B21",
            "null_read_null_B21",
            "early_ortho",
            "image_attack_en_logit",
            "image_object_en_logit",
            "attack_minus_object_en",
            "embedding_effect_recovery",
            "relative_calibrated_recovery",
            "candidate_read_null_B21_recovery",
        ):
            nv = safe_float(n.get(name))
            sv = safe_float(s.get(name))
            if math.isfinite(nv):
                row["norta_" + name] = nv
            if math.isfinite(sv):
                row["synth_" + name] = sv
            if math.isfinite(nv) and math.isfinite(sv):
                row["delta_" + name] = sv - nv
        out.append(row)
    return out


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_rows(
    rows: list[dict[str, Any]],
    group_fields: list[str],
    measure_fields: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_fields)
        groups.setdefault(key, []).append(row)

    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = {k: v for k, v in zip(group_fields, key)}
        row["n"] = len(rs)
        for field in measure_fields:
            vals = np.asarray(
                [safe_float(r.get(field)) for r in rs],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            row[field + "_mean"] = float(vals.mean())
            row[field + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[field + "_sem"] = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            row[field + "_q10"] = float(np.quantile(vals, 0.10))
            row[field + "_median"] = float(np.quantile(vals, 0.50))
            row[field + "_q90"] = float(np.quantile(vals, 0.90))
        out.append(row)
    return out


ALPHA_MEASURES = [
    "relative_calibrated",
    "candidate_read_null_B20",
    "candidate_read_null_B21",
    "null_read_null_B21",
    "early_ortho",
    "image_attack_en_logit",
    "image_object_en_logit",
    "attack_minus_object_en",
    "embedding_effect_recovery",
    "relative_calibrated_recovery",
    "candidate_read_null_B21_recovery",
]

PAIRED_ALPHA_MEASURES = [
    "delta_relative_calibrated",
    "delta_candidate_read_null_B20",
    "delta_candidate_read_null_B21",
    "delta_null_read_null_B21",
    "delta_early_ortho",
    "delta_image_attack_en_logit",
    "delta_image_object_en_logit",
    "delta_attack_minus_object_en",
    "delta_embedding_effect_recovery",
    "delta_relative_calibrated_recovery",
]

POP_MEASURES = ALPHA_MEASURES
PAIRED_POP_MEASURES = PAIRED_ALPHA_MEASURES

TRAJ_MEASURES = [
    "delta_norm_mean",
    "flat_cos_exact_rn_delta",
    "pc1_patchmean_cos",
    "subspace_fraction_rank4",
    "flat_cos_attack_contrast",
    "patchmean_cos_attack_contrast",
]

JAC_MEASURES = [
    "derivative",
    "curvature",
    "derivative_per_component_norm",
]


# =============================================================================
# Plotting — deliberately selective
# =============================================================================

def auto_ylim(vals: list[float], include_zero: bool = True) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if math.isfinite(float(v))], dtype=float)
    if len(arr) == 0:
        return -1.0, 1.0
    lo, hi = float(arr.min()), float(arr.max())
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = max(hi - lo, 1e-6)
    pad = max(0.10 * span, 0.02 * max(abs(lo), abs(hi), 1.0))
    return lo - pad, hi + pad


def plot_alpha_summary(
    summary: list[dict[str, Any]],
    query_mode: str,
    rank: int,
    y_field: str,
    ylabel: str,
    title: str,
    out: Path,
) -> None:
    rs = [
        r for r in summary
        if r.get("query_mode") == query_mode
        and r.get("mode") == "paired"
        and int(r.get("rank", -1)) == int(rank)
    ]
    if not rs:
        return
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rs:
        groups.setdefault(str(r["language"]), []).append(r)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    all_y = []
    for lang, gr in sorted(groups.items()):
        gr = sorted(gr, key=lambda r: float(r["alpha"]))
        xs = [float(r["alpha"]) for r in gr]
        ys = [safe_float(r.get(y_field)) for r in gr]
        all_y.extend(ys)
        ax.plot(xs, ys, marker="o", label=lang)
    ax.axvline(0.0, linewidth=1.0)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("RN control strength α")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*auto_ylim(all_y, include_zero=True))
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_population_heatmap(
    summary: list[dict[str, Any]],
    query_mode: str,
    rank: int,
    alpha: float,
    measure: str,
    title: str,
    out: Path,
) -> None:
    rs = [
        r for r in summary
        if r.get("query_mode") == query_mode
        and int(r.get("rank", -1)) == rank
        and abs(float(r.get("alpha", 999)) - alpha) < 1e-6
    ]
    if not rs:
        return
    langs = sorted({str(r["language"]) for r in rs})
    pops = sorted({str(r["population"]) for r in rs})
    mat = np.full((len(langs), len(pops)), np.nan, dtype=float)
    for r in rs:
        i = langs.index(str(r["language"]))
        j = pops.index(str(r["population"]))
        mat[i, j] = safe_float(r.get(measure))

    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax < 1e-12:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    im = ax.imshow(mat, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(pops)))
    ax.set_xticklabels(pops, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(langs)))
    ax.set_yticklabels(langs)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_summary(
    summary: list[dict[str, Any]],
    query_mode: str,
    condition: str,
    measure: str,
    ylabel: str,
    title: str,
    out: Path,
) -> None:
    rs = [
        r for r in summary
        if r.get("query_mode") == query_mode
        and r.get("condition") == condition
    ]
    if not rs:
        return
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rs:
        groups.setdefault(str(r["language"]), []).append(r)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    all_y = []
    for lang, gr in sorted(groups.items()):
        gr = sorted(gr, key=lambda r: int(r["block"]))
        xs = [int(r["block"]) for r in gr]
        ys = [safe_float(r.get(measure)) for r in gr]
        all_y.extend(ys)
        ax.plot(xs, ys, marker="o", label=lang)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("block")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted({int(r["block"]) for r in rs}))
    ax.set_ylim(*auto_ylim(all_y, include_zero=True))
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_jac_heatmap(
    summary: list[dict[str, Any]],
    query_mode: str,
    condition: str,
    observable: str,
    title: str,
    out: Path,
) -> None:
    rs = [
        r for r in summary
        if r.get("query_mode") == query_mode
        and r.get("condition") == condition
        and r.get("observable") == observable
    ]
    if not rs:
        return
    langs = sorted({str(r["language"]) for r in rs})
    pcs = sorted({int(r["pc"]) for r in rs})
    mat = np.full((len(langs), len(pcs)), np.nan, dtype=float)
    for r in rs:
        mat[langs.index(str(r["language"])), pcs.index(int(r["pc"]))] = safe_float(r.get("derivative_mean"))

    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax < 1e-12:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(mat, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(pcs)))
    ax.set_xticklabels([f"PC{x}" for x in pcs])
    ax.set_yticks(np.arange(len(langs)))
    ax.set_yticklabels(langs)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Compact handoff
# =============================================================================

def build_summary_text(
    basis_rows: list[dict[str, Any]],
    paired_alpha_summary: list[dict[str, Any]],
    jac_summary: list[dict[str, Any]],
    languages: list[str],
    query_modes: list[str],
    limit: int,
) -> str:
    lines = []
    lines.append("RN × RTA-100-MULTILINGUAL CONTROL-KNOB — COMPACT HANDOFF")
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"languages: {','.join(languages)}")
    lines.append(f"query_modes: {','.join(query_modes)}")
    lines.append(f"paired keys per language: {'ALL' if limit <= 0 else limit}")
    lines.append("")
    lines.append("B13 RN basis:")
    for r in basis_rows:
        lines.append(
            f"  PC{r['pc']}: singular={float(r['singular_value']):.6g} "
            f"explained={float(r['explained_energy']):.6f}"
        )

    lines.append("")
    lines.append("Paired SynthRTA-NoRTA alpha sweep: paired rank-4, selected alphas")
    for qm in query_modes:
        lines.append(f"  query_mode={qm}")
        for lang in languages:
            rs = [
                r for r in paired_alpha_summary
                if r.get("query_mode") == qm
                and r.get("language") == lang
                and r.get("mode") == "paired"
                and int(r.get("rank", -1)) == 4
            ]
            bits = []
            for a in (-1.0, 0.0, 1.0):
                if not rs:
                    continue
                hit = min(rs, key=lambda r: abs(float(r["alpha"]) - a))
                bits.append(
                    f"a={float(hit['alpha']):+.1f}: "
                    f"dRel={safe_float(hit.get('delta_relative_calibrated_mean')):+.4f}, "
                    f"dB21RN={safe_float(hit.get('delta_candidate_read_null_B21_mean')):+.4f}"
                )
            if bits:
                lines.append(f"    {lang}: " + " | ".join(bits))

    lines.append("")
    lines.append("Largest mean |relative-read derivative| PC per language (SynthRTA):")
    for qm in query_modes:
        lines.append(f"  query_mode={qm}")
        for lang in languages:
            rs = [
                r for r in jac_summary
                if r.get("query_mode") == qm
                and r.get("condition") == "synth"
                and r.get("language") == lang
                and r.get("observable") == "relative_calibrated"
            ]
            if rs:
                best = max(rs, key=lambda r: abs(safe_float(r.get("derivative_mean"))))
                lines.append(
                    f"    {lang}: PC{best['pc']} d/dalpha={safe_float(best.get('derivative_mean')):+.5g} "
                    f"curvature={safe_float(best.get('curvature_mean')):+.5g}"
                )

    lines.append("")
    lines.append("Interpretation guardrail: NoRTA is an exact paired control for the rendered overlay,")
    lines.append("not a guarantee that the underlying natural image contains zero incidental text.")
    lines.append("")
    return "\n".join(lines)


def build_handoff_zip(output_dir: Path, files: list[Path]) -> Path:
    path = output_dir / "compact_summary_rn_control_knob.zip"
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            if p.exists() and p.is_file():
                z.write(p, arcname=p.relative_to(output_dir).as_posix())
    return path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset-level RN B13 control-knob suite on RTA-100 Multilingual")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--module-root", default=".")
    ap.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Optional local HF repo root containing data/*.parquet")
    ap.add_argument("--languages", default=DEFAULT_LANGUAGES)
    ap.add_argument("--query-modes", default=DEFAULT_QUERY_MODES, help="english,native")
    ap.add_argument("--limit-per-language", type=int, default=DEFAULT_LIMIT, help="0 = all shared keys")
    ap.add_argument("--basis-pairs-per-language", type=int, default=DEFAULT_BASIS_PAIRS_PER_LANGUAGE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alphas", default=DEFAULT_ALPHAS)
    ap.add_argument("--ranks", default=DEFAULT_RANKS)
    ap.add_argument("--modes", default=DEFAULT_MODES)
    ap.add_argument("--populations", default=DEFAULT_POPULATIONS)
    ap.add_argument("--population-alphas", default=DEFAULT_POPULATION_ALPHAS)
    ap.add_argument("--population-rank", type=int, default=4)
    ap.add_argument("--trajectory-alpha", type=float, default=1.0)
    ap.add_argument("--trajectory-rank", type=int, default=4)
    ap.add_argument("--trajectory-mode", default="paired")
    ap.add_argument("--jacobian-eps", type=float, default=0.25)
    ap.add_argument("--pc-count", type=int, default=4)
    ap.add_argument("--templates", default=DEFAULT_TEMPLATES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="rn_rta_multilingual_control_knob")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    languages = parse_str_list(args.languages)
    query_modes = parse_str_list(args.query_modes)
    alphas = parse_float_list(args.alphas)
    ranks = parse_int_list(args.ranks)
    modes = parse_str_list(args.modes)
    populations = parse_str_list(args.populations)
    population_alphas = parse_float_list(args.population_alphas)
    templates = [x.strip() for x in args.templates.split(";") if x.strip()]

    bad_q = [x for x in query_modes if x not in {"english", "native"}]
    if bad_q:
        raise ValueError(f"Unknown query modes: {bad_q}")

    out_dir = Path(args.output_dir)
    data_dir = out_dir / "data"
    plot_dir = out_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    norta, attacks, norta_idx, attack_idx, shared_keys = prepare_datasets(
        args.dataset_repo,
        args.dataset_root,
        languages,
        args.limit_per_language,
        args.seed,
    )

    loaded = load_model(args.checkpoint, package_root=args.module_root, device=args.device)
    model, preprocess, clip_mod, device = (
        loaded.model,
        loaded.preprocess,
        loaded.clip_module,
        loaded.device,
    )
    model.eval()

    target_res = int(model.visual.input_resolution)
    patch = int(model.visual.conv1.kernel_size[0])
    side = target_res // patch
    insert_block = int(model.visual.read_null_insert_block)
    read_blocks = [int(x) for x in model.read_implant.tap_block_list()]
    capture_list = [int(x) for x in model.read_implant.capture_block_list()]
    early_needed = {b for b in capture_list if b < insert_block}
    trajectory_blocks = set(range(insert_block, max(read_blocks) + 1))
    all_capture = set(capture_list) | trajectory_blocks

    max_rank = max(max(ranks), args.population_rank, args.trajectory_rank, args.pc_count, 4)
    print("[basis] fitting multilingual RN subspace")
    basis_info = fit_basis_from_dataset(
        model,
        preprocess,
        norta,
        attacks,
        norta_idx,
        attack_idx,
        shared_keys,
        languages,
        args.basis_pairs_per_language,
        early_needed,
        max_rank,
        device,
    )

    global BASIS_GLOBAL, MEAN_TEMPLATE_GLOBAL
    BASIS_GLOBAL = basis_info["basis_kd"].to(device)
    MEAN_TEMPLATE_GLOBAL = basis_info["mean_template_td"].to(device)

    np.savez_compressed(
        data_dir / "basis.npz",
        basis_kd=basis_info["basis_kd"].cpu().numpy(),
        singular_values=basis_info["singular_values"].cpu().numpy(),
        explained_energy=np.asarray(basis_info["explained_energy"]),
        mean_template_td=basis_info["mean_template_td"].cpu().numpy(),
        fit_items=np.asarray(basis_info["fit_items"], dtype=object),
    )
    basis_rows = [
        {
            "pc": i + 1,
            "singular_value": float(basis_info["singular_values"][i].cpu()),
            "explained_energy": float(basis_info["explained_energy"][i]),
        }
        for i in range(len(basis_info["explained_energy"]))
    ]
    save_rows(data_dir / "basis_summary.csv", basis_rows)

    query_cache: dict[str, dict[str, torch.Tensor | bool]] = {}
    semantic_cache: dict[str, torch.Tensor] = {}

    def get_query(label: str):
        if label not in query_cache:
            query_cache[label] = prepare_read_queries(model, clip_mod, label, device)
        return query_cache[label]

    def get_semantic_en(label: str):
        if label not in semantic_cache:
            semantic_cache[label] = prompt_bank_embedding(model, clip_mod, label, templates, device)
        return semantic_cache[label]

    raw_alpha: list[dict[str, Any]] = []
    raw_pop: list[dict[str, Any]] = []
    raw_jac: list[dict[str, Any]] = []
    raw_refs: list[dict[str, Any]] = []
    raw_traj: list[dict[str, Any]] = []
    paired_alpha: list[dict[str, Any]] = []
    paired_pop: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    progress = tqdm(shared_keys, desc="sample_key", unit="pair")

    for key in progress:
        # -------------------------
        # NoRTA visual bundle ONCE
        # -------------------------
        nrow = get_row(norta, norta_idx[key])
        npil = ensure_pil(nrow["image"])
        nimg = preprocess_pil(preprocess, npil, device)
        nbbox = bbox_patch_mask(
            nrow["bbox"],
            npil.size,
            target_res,
            patch,
            device,
        )
        npair = build_b13_pair(model, nimg, key, "norta", early_needed)
        nbundle = make_visual_bundle(
            model,
            npair,
            nbbox,
            BASIS_GLOBAL,
            MEAN_TEMPLATE_GLOBAL,
            ranks,
            modes,
            alphas,
            populations,
            population_alphas,
            args.population_rank,
            args.pc_count,
            args.jacobian_eps,
            all_capture,
        )

        for lang in languages:
            srow = get_row(attacks[lang], attack_idx[lang][key])
            spil = ensure_pil(srow["image"])
            simg = preprocess_pil(preprocess, spil, device)
            sbbox = bbox_patch_mask(
                srow["bbox"],
                spil.size,
                target_res,
                patch,
                device,
            )
            spair = build_b13_pair(model, simg, key, "synth", early_needed)
            sbundle = make_visual_bundle(
                model,
                spair,
                sbbox,
                BASIS_GLOBAL,
                MEAN_TEMPLATE_GLOBAL,
                ranks,
                modes,
                alphas,
                populations,
                population_alphas,
                args.population_rank,
                args.pc_count,
                args.jacobian_eps,
                all_capture,
            )

            attack_en = str(srow["attack_word_en"])
            object_en = str(srow["object_label_en"])
            attack_native = str(srow["attack_word"])
            object_native = str(srow["object_label"])

            manifest_rows.append({
                "sample_key": key,
                "language": lang,
                "language_name": LANGUAGE_NAMES.get(lang, lang),
                "object_label_en": object_en,
                "attack_word_en": attack_en,
                "object_label_native": object_native,
                "attack_word_native": attack_native,
                "bbox": json.dumps([float(x) for x in srow["bbox"]]),
                "norta_bbox_patch_count": int(nbbox.sum().item()),
                "synth_bbox_patch_count": int(sbbox.sum().item()),
            })

            attack_sem_en = get_semantic_en(attack_en)
            object_sem_en = get_semantic_en(object_en)

            # Exact attack-induced no-RN contrast for this sample/language.
            attack_contrast = {}
            for b in trajectory_blocks:
                if b in sbundle.base_run["states"] and b in nbundle.base_run["states"]:
                    attack_contrast[b] = (
                        sbundle.base_run["states"][b] - nbundle.base_run["states"][b]
                    ).float()

            for qm in query_modes:
                query_label = attack_en if qm == "english" else attack_native
                queries = get_query(query_label)

                base_meta = {
                    "sample_key": key,
                    "language": lang,
                    "language_name": LANGUAGE_NAMES.get(lang, lang),
                    "query_mode": qm,
                    "query_candidate": query_label,
                    "attack_word_en": attack_en,
                    "object_label_en": object_en,
                    "attack_word_native": attack_native,
                    "object_label_native": object_native,
                }

                nmeta = dict(base_meta)
                nmeta["condition"] = "norta"
                smeta = dict(base_meta)
                smeta["condition"] = "synth"

                na, npop, njac, nref = score_visual_bundle(
                    model, nbundle, queries, attack_sem_en, object_sem_en,
                    nmeta, ranks, modes, alphas, args.population_rank, args.jacobian_eps
                )
                sa, spop, sjac, sref = score_visual_bundle(
                    model, sbundle, queries, attack_sem_en, object_sem_en,
                    smeta, ranks, modes, alphas, args.population_rank, args.jacobian_eps
                )

                raw_alpha.extend(na); raw_alpha.extend(sa)
                raw_pop.extend(npop); raw_pop.extend(spop)
                raw_jac.extend(njac); raw_jac.extend(sjac)
                raw_refs.extend(nref); raw_refs.extend(sref)

                paired_alpha.extend(pair_delta_rows(
                    na,
                    sa,
                    key_fields=["mode", "rank", "alpha"],
                    identity_meta=base_meta,
                ))
                paired_pop.extend(pair_delta_rows(
                    npop,
                    spop,
                    key_fields=["population", "rank", "alpha"],
                    identity_meta=base_meta,
                ))

                # One canonical trajectory intervention, both clean and attacked.
                raw_traj.extend(trajectory_rows_for_condition(
                    nbundle,
                    attack_contrast,
                    nmeta,
                    BASIS_GLOBAL,
                    args.trajectory_mode,
                    args.trajectory_rank,
                    args.trajectory_alpha,
                    trajectory_blocks,
                ))
                raw_traj.extend(trajectory_rows_for_condition(
                    sbundle,
                    attack_contrast,
                    smeta,
                    BASIS_GLOBAL,
                    args.trajectory_mode,
                    args.trajectory_rank,
                    args.trajectory_alpha,
                    trajectory_blocks,
                ))

            del spair, sbundle, simg, spil
            cleanup_cuda()

        del npair, nbundle, nimg, npil
        cleanup_cuda()

    # -------------------------------------------------------------------------
    # Save raw local data
    # -------------------------------------------------------------------------
    save_rows(data_dir / "raw_references.csv", raw_refs)
    save_rows(data_dir / "raw_alpha_sweep.csv", raw_alpha)
    save_rows(data_dir / "raw_population_sweep.csv", raw_pop)
    save_rows(data_dir / "raw_trajectory.csv", raw_traj)
    save_rows(data_dir / "raw_pc_response_jacobian.csv", raw_jac)
    save_rows(data_dir / "paired_alpha_deltas.csv", paired_alpha)
    save_rows(data_dir / "paired_population_deltas.csv", paired_pop)
    save_rows(data_dir / "sample_manifest.csv", manifest_rows)

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------
    alpha_summary = aggregate_rows(
        raw_alpha,
        ["condition", "language", "query_mode", "mode", "rank", "alpha"],
        ALPHA_MEASURES,
    )
    paired_alpha_summary = aggregate_rows(
        paired_alpha,
        ["language", "query_mode", "mode", "rank", "alpha"],
        PAIRED_ALPHA_MEASURES,
    )
    pop_summary = aggregate_rows(
        raw_pop,
        ["condition", "language", "query_mode", "population", "rank", "alpha"],
        POP_MEASURES,
    )
    paired_pop_summary = aggregate_rows(
        paired_pop,
        ["language", "query_mode", "population", "rank", "alpha"],
        PAIRED_POP_MEASURES,
    )
    traj_summary = aggregate_rows(
        raw_traj,
        ["condition", "language", "query_mode", "mode", "rank", "alpha", "block"],
        TRAJ_MEASURES,
    )
    jac_summary = aggregate_rows(
        raw_jac,
        ["condition", "language", "query_mode", "pc", "observable"],
        JAC_MEASURES,
    )

    save_rows(data_dir / "alpha_summary.csv", alpha_summary)
    save_rows(data_dir / "paired_alpha_delta_summary.csv", paired_alpha_summary)
    save_rows(data_dir / "population_summary.csv", pop_summary)
    save_rows(data_dir / "paired_population_delta_summary.csv", paired_pop_summary)
    save_rows(data_dir / "trajectory_summary.csv", traj_summary)
    save_rows(data_dir / "pc_response_jacobian_summary.csv", jac_summary)

    config = {
        "checkpoint": args.checkpoint,
        "dataset_repo": args.dataset_repo,
        "dataset_root": args.dataset_root,
        "languages": languages,
        "query_modes": query_modes,
        "limit_per_language": args.limit_per_language,
        "shared_keys_used": len(shared_keys),
        "seed": args.seed,
        "basis_pairs_per_language": args.basis_pairs_per_language,
        "alphas": alphas,
        "ranks": ranks,
        "modes": modes,
        "populations": populations,
        "population_alphas": population_alphas,
        "population_rank": args.population_rank,
        "trajectory": {
            "mode": args.trajectory_mode,
            "rank": args.trajectory_rank,
            "alpha": args.trajectory_alpha,
        },
        "jacobian_eps": args.jacobian_eps,
        "pc_count": args.pc_count,
        "insert_block": insert_block,
        "read_blocks": read_blocks,
        "image_resolution": target_res,
        "patch_size": patch,
        "intervention_convention": (
            "RN-context hybrid: RN-off ordinary B13 state + alpha*RN-subspace "
            "component; exact real B13 RN output token retained downstream"
        ),
        "semantic_dashboard": (
            "canonical English object_label_en / attack_word_en for all languages"
        ),
        "pairing": "exact SynthRTA vs NoRTA by sample_key",
        "guardrail": (
            "NoRTA is treated as an exact rendered-overlay control, not as proof "
            "that the natural image contains zero incidental text"
        ),
    }
    save_json(data_dir / "config.json", config)

    # -------------------------------------------------------------------------
    # Selected plots
    # -------------------------------------------------------------------------
    for qm in query_modes:
        for rank in ranks:
            plot_alpha_summary(
                paired_alpha_summary,
                qm,
                rank,
                "delta_relative_calibrated_mean",
                "mean Δ relative READ (SynthRTA − NoRTA)",
                f"RN knob vs rendered-text effect — {qm} query — rank-{rank}",
                plot_dir / f"01_alpha_delta_relative_read__{qm}__rank{rank}.png",
            )

        plot_alpha_summary(
            paired_alpha_summary,
            qm,
            args.population_rank,
            "delta_candidate_read_null_B21_mean",
            "mean Δ B21 candidate→RN (SynthRTA − NoRTA)",
            f"B21 RN routing vs control strength — {qm} query",
            plot_dir / f"02_alpha_delta_B21_candidate_RN__{qm}.png",
        )

        # Population: show attacked absolute recovery, which is easiest to interpret.
        synth_pop = [
            r for r in pop_summary
            if r.get("condition") == "synth"
        ]
        plot_population_heatmap(
            synth_pop,
            qm,
            args.population_rank,
            1.0,
            "relative_calibrated_recovery_mean",
            f"Where must RN rank-{args.population_rank} be written? — SynthRTA — {qm}",
            plot_dir / f"03_population_read_recovery__synth__{qm}.png",
        )

        plot_trajectory_summary(
            traj_summary,
            qm,
            "synth",
            "patchmean_cos_attack_contrast_mean",
            "cos(control delta, exact SynthRTA−NoRTA contrast)",
            f"RN state unfolds toward rendered-text contrast — SynthRTA — {qm}",
            plot_dir / f"04_trajectory_attack_contrast__synth__{qm}.png",
        )

        plot_jac_heatmap(
            jac_summary,
            qm,
            "synth",
            "relative_calibrated",
            f"Local RN control axes: relative READ derivative — SynthRTA — {qm}",
            plot_dir / f"05_pc_relative_read_derivative__synth__{qm}.png",
        )

    summary_path = out_dir / "SUMMARY.txt"
    summary_path.write_text(
        build_summary_text(
            basis_rows,
            paired_alpha_summary,
            jac_summary,
            languages,
            query_modes,
            args.limit_per_language,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Small compact summary archive: aggregates + selected plots, not raw firehose.
    # -------------------------------------------------------------------------
    selected = [
        data_dir / "config.json",
        data_dir / "basis.npz",
        data_dir / "basis_summary.csv",
        data_dir / "sample_manifest.csv",
        data_dir / "alpha_summary.csv",
        data_dir / "paired_alpha_delta_summary.csv",
        data_dir / "population_summary.csv",
        data_dir / "paired_population_delta_summary.csv",
        data_dir / "trajectory_summary.csv",
        data_dir / "pc_response_jacobian_summary.csv",
        summary_path,
    ] + sorted(plot_dir.glob("*.png"))

    zip_path = build_handoff_zip(out_dir, selected)

    print("\n[done]")
    print(f"  raw + aggregate data: {data_dir.resolve()}")
    print(f"  selected plots:       {plot_dir.resolve()}")
    print(f"  summary:              {summary_path.resolve()}")
    print(f"  compact summary:         {zip_path.resolve()}")




if __name__ == "__main__":
    main()
