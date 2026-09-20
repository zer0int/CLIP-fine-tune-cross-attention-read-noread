#!/usr/bin/env python3
r"""Exact TEXT-to-CLS OV/write trajectory across models and RN states.
Commands: analyze (native model/RN factorial), plot (existing trajectory CSVs).
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_analysis import (RNPayload, clear_cuda)

# ANALYZE
import argparse
import importlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm.auto import tqdm

# The project-specific CLIP modules are imported lazily so --plot_only_summary
# can regenerate paper figures on a machine that does not have the mechinterp
# environment installed.
clip = None
orgclip = None


def ensure_clip_modules() -> None:
    global clip, orgclip

    if clip is None:
        import attnclip_mechinterp_xattn as _clip
        clip = _clip

    if orgclip is None:
        import attnclip_mechinterp_sae as _orgclip
        orgclip = _orgclip


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_DATASET = "zer0int/RTA-100-Triplet"

DEFAULT_GMP_CHECKPOINT = (
    r"GMP_CHECKPOINT.pt"
)

DEFAULT_XATTN_CHECKPOINT = "REPLACE_WITH_CHECKPOINT.pt"

DEFAULT_RN_CHECKPOINT = DEFAULT_XATTN_CHECKPOINT
DEFAULT_OUT_DIR = "figures/text_to_cls_multimodel_rn"

MODEL_ORDER = ("pretrained", "gmp", "xattn_stripped")
MODEL_LABELS = {
    "pretrained": "Pretrained",
    "gmp": "GmP",
    "xattn_stripped": "All-weights backbone",
}

RN_MODES = ("rn_off", "rn_on")
RN_LABELS = {
    "rn_off": "RN off",
    "rn_on": "RN on",
}

CONDITIONS = ("NoRTA", "RTA", "SynthRTA")
ATTACKS = ("RTA", "SynthRTA")
BLOCKS = tuple(range(12, 24))
RN_INSERT_BLOCK = 13
EPS = 1e-12

# Only used for a non-fatal sanity check of xattn_stripped / RN off.
REFERENCE_RTA = np.asarray(
    [
        -.0141, -.0246, -.0531, -.0397, -.0546, -.0017,
        -.0354, +.0052, +.0489, +.1054, +.1566, +.1545,
    ],
    dtype=np.float64,
)


# =============================================================================
# Dataclasses / basics
# =============================================================================


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path: Path) -> Any:
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(str(path), weights_only=False, **kwargs)
    except TypeError:
        return torch.load(str(path), **kwargs)


def strip_common_prefixes(state: Mapping[str, Any]) -> dict[str, Any]:
    out = {str(k): v for k, v in state.items()}
    for prefix in ("module.", "model.", "clip."):
        keys = list(out)
        if (
            keys
            and sum(k.startswith(prefix) for k in keys)
            >= max(1, int(0.9 * len(keys)))
        ):
            out = {
                k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in out.items()
            }
    return out


def looks_like_clip_state(obj: Mapping[str, Any]) -> bool:
    keys = set(map(str, obj.keys()))
    return (
        "visual.conv1.weight" in keys
        and (
            "token_embedding.weight" in keys
            or any(k.startswith("transformer.resblocks.") for k in keys)
            or any(k.startswith("visual.transformer.resblocks.") for k in keys)
        )
    )


def extract_state_dict(obj: Any, source: str) -> dict[str, Any]:
    if isinstance(obj, torch.nn.Module):
        return strip_common_prefixes(obj.state_dict())

    if isinstance(obj, Mapping):
        candidate = strip_common_prefixes(obj)
        if looks_like_clip_state(candidate):
            return candidate

        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "clip",
            "module",
        ):
            if key not in obj:
                continue
            value = obj[key]
            if isinstance(value, torch.nn.Module):
                return strip_common_prefixes(value.state_dict())
            if isinstance(value, Mapping):
                candidate = strip_common_prefixes(value)
                if looks_like_clip_state(candidate):
                    return candidate

    raise TypeError(
        f"Could not extract a CLIP state_dict from {source}: {type(obj)}"
    )


def freeze_eval(model):
    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def assert_explicit_qkv(model, label: str) -> None:
    attn = model.visual.transformer.resblocks[0].attn
    required = ("q_proj", "k_proj", "v_proj", "out_proj")
    missing = [name for name in required if not hasattr(attn, name)]
    if missing:
        raise RuntimeError(
            f"{label}: expected explicit-QKV attnclip architecture; "
            f"missing {missing}"
        )


# =============================================================================
# Model / RN loading
# =============================================================================

def load_rn_payload(
    checkpoint: str,
    requested_insert_block: int,
) -> RNPayload:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)

    obj = safe_torch_load(path)
    state = extract_state_dict(obj, str(path))
    del obj
    clear_cuda()

    key = "visual.read_null_token"
    if key not in state or not torch.is_tensor(state[key]):
        raise KeyError(f"{key!r} missing from {path}")

    token = (
        state[key]
        .detach()
        .float()
        .cpu()
        .reshape(-1)
        .contiguous()
    )

    cfg = state.get(
        "visual.read_null_insert_block_config",
        torch.tensor(requested_insert_block),
    )
    insert_block = int(cfg.item())

    if insert_block != requested_insert_block:
        raise RuntimeError(
            f"RN checkpoint says pre-B{insert_block}, "
            f"requested pre-B{requested_insert_block}"
        )

    print(
        f"[RN] token dim={token.numel()} "
        f"norm={float(token.norm()):.6f} "
        f"insert=pre-B{insert_block}"
    )

    return RNPayload(
        token=token,
        insert_block=insert_block,
        checkpoint=str(path),
    )


def load_pretrained_model(device: torch.device):
    print("[model] loading pretrained ViT-L/14")
    model, preprocess = orgclip.load(
        "ViT-L/14",
        device=device,
        jit=False,
    )
    model = freeze_eval(model)
    assert_explicit_qkv(model, "pretrained")
    return model, preprocess


def load_gmp_model(
    checkpoint: str,
    device: torch.device,
):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)

    print(f"[model] loading GmP: {path}")

    # attnclip_mechinterp_sae.load(path) is designed to extract ordinary
    # OpenAI-compatible checkpoint state and build the explicit-QKV model.
    try:
        model, preprocess = orgclip.load(
            str(path),
            device=device,
            jit=False,
        )
    except Exception as first_error:
        # Optional loader used elsewhere in this project.
        try:
            from utils_clip_loader.clip_anything_to_openai import (
                load_openai_clip_anything,
            )
            model, preprocess, _info = load_openai_clip_anything(
                orgclip,
                str(path),
                device=device,
                jit=False,
                strict=True,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"Could not load GmP checkpoint.\n"
                f"orgclip.load error: {first_error}\n"
                f"fallback error: {fallback_error}"
            ) from fallback_error

    model = freeze_eval(model)
    assert_explicit_qkv(model, "gmp")
    return model, preprocess


def load_xattn_stripped_model(
    checkpoint: str,
    device: torch.device,
    audit_dir: Path,
):
    """
    Load ONLY ordinary visual.* tensors from the final x-attn checkpoint
    into a fresh explicit-QKV vanilla ViT-L/14.
    """
    print("[model] loading fresh vanilla ViT for xattn_stripped")
    model, preprocess = orgclip.load(
        "ViT-L/14",
        device="cpu",
        jit=False,
    )
    model = freeze_eval(model)

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)

    print(f"[model] reading final x-attn checkpoint: {path}")
    obj = safe_torch_load(path)
    state = extract_state_dict(obj, str(path))
    del obj
    clear_cuda()

    if any(k.endswith(".theta") or k.endswith(".r") for k in state):
        raise RuntimeError(
            "Final x-attn checkpoint still contains GmP theta/r parameters; "
            "expected the materialized _ungmp_ checkpoint."
        )

    source = {
        k: v
        for k, v in state.items()
        if k.startswith("visual.")
        and "read_null" not in k.lower()
        and torch.is_tensor(v)
    }

    model_mod = importlib.import_module("attnclip_mechinterp_sae.model")
    convert = getattr(
        model_mod,
        "convert_state_dict_inproj_to_qkv",
        None,
    )
    if convert is not None:
        source = convert(dict(source))

    target = model.state_dict()
    target_visual_keys = [
        k for k in target
        if k.startswith("visual.")
    ]

    load_state = {}
    missing = []
    mismatched = []

    for key in target_visual_keys:
        if key not in source:
            missing.append(key)
            continue

        src = source[key].detach().cpu()
        dst = target[key].detach().cpu()

        if tuple(src.shape) != tuple(dst.shape):
            mismatched.append(
                {
                    "key": key,
                    "source_shape": list(src.shape),
                    "target_shape": list(dst.shape),
                }
            )
            continue

        load_state[key] = src.to(dtype=target[key].dtype)

    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "checkpoint": str(path),
        "source_ordinary_visual_keys": len(source),
        "target_visual_keys": len(target_visual_keys),
        "loaded_visual_keys": len(load_state),
        "missing_target_visual_keys": missing,
        "shape_mismatch": mismatched,
        "read_null_loaded": False,
        "bridge_router_correction_loaded": False,
        "text_tower_loaded_from_xattn": False,
        "note": (
            "Only ordinary visual.* tensors transplanted into fresh "
            "attnclip_mechinterp_sae ViT-L/14."
        ),
    }
    (audit_dir / "xattn_strip_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    if missing or mismatched:
        raise RuntimeError(
            f"Strict x-attn visual transplant failed: "
            f"missing={len(missing)}, mismatched={len(mismatched)}; "
            f"see {audit_dir / 'xattn_strip_audit.json'}"
        )

    merged = model.state_dict()
    merged.update(load_state)
    model.load_state_dict(merged, strict=True)

    model = freeze_eval(model.to(device))
    assert_explicit_qkv(model, "xattn_stripped")

    print(
        f"[model] transplanted {len(load_state)} ordinary visual tensors; "
        "RN/bridge/router/correction excluded"
    )
    return model, preprocess


def load_model_variant(
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
    audit_dir: Path,
):
    if model_name == "pretrained":
        return load_pretrained_model(device)

    if model_name == "gmp":
        return load_gmp_model(
            args.gmp_checkpoint,
            device,
        )

    if model_name == "xattn_stripped":
        return load_xattn_stripped_model(
            args.xattn_checkpoint,
            device,
            audit_dir,
        )

    raise ValueError(model_name)


# =============================================================================
# RTA-100 pairing
# =============================================================================

def canonical_pair_id(raw_id: Any, condition: str) -> str:
    value = str(raw_id).strip()
    lower = value.lower()
    prefix = condition.lower()

    if lower.startswith(prefix):
        value = value[len(condition):]
        value = value.lstrip(" _:-./")

    return value


def load_triplets(
    repo: str,
    split: str,
    limit: int,
    seed: int,
):
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)

    required = {"type", "image", "id"}
    missing = required - set(ds.column_names)
    if missing:
        raise KeyError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    by_condition: dict[str, dict[str, int]] = {
        condition: {}
        for condition in CONDITIONS
    }

    for index, row in enumerate(ds):
        condition = str(row["type"])
        if condition not in by_condition:
            continue

        pair_id = canonical_pair_id(
            row["id"],
            condition,
        )

        if pair_id in by_condition[condition]:
            raise RuntimeError(
                f"Duplicate pair id {pair_id!r} in {condition}"
            )

        by_condition[condition][pair_id] = index

    shared = set.intersection(
        *(set(by_condition[c]) for c in CONDITIONS)
    )
    if not shared:
        raise RuntimeError(
            "No paired NoRTA/RTA/SynthRTA IDs found."
        )

    pair_ids = sorted(shared)

    if 0 < limit < len(pair_ids):
        rng = random.Random(seed)
        pair_ids = sorted(
            rng.sample(pair_ids, limit)
        )

    rows = []
    for pair_id in pair_ids:
        row = {
            "pair_id": pair_id,
        }

        for condition in CONDITIONS:
            row[condition] = ds[
                by_condition[condition][pair_id]
            ]

        rows.append(row)

    print(
        f"[dataset] paired triplets: {len(rows)}"
    )
    return rows


def as_pil(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")

    if isinstance(image_obj, Mapping):
        if image_obj.get("path"):
            return Image.open(
                image_obj["path"]
            ).convert("RGB")

        if image_obj.get("bytes"):
            import io
            return Image.open(
                io.BytesIO(image_obj["bytes"])
            ).convert("RGB")

    raise TypeError(
        f"Unsupported dataset image object: {type(image_obj)}"
    )


# =============================================================================
# Exact paired TEXT masks
# =============================================================================

def patchify_score(
    score_hw: torch.Tensor,
    side: int,
) -> torch.Tensor:
    height, width = score_hw.shape

    if height % side != 0 or width % side != 0:
        raise ValueError(
            f"Preprocessed image {height}x{width} is not "
            f"divisible by {side}x{side} patch grid."
        )

    patch_h = height // side
    patch_w = width // side

    return (
        score_hw
        .reshape(
            side,
            patch_h,
            side,
            patch_w,
        )
        .permute(0, 2, 1, 3)
        .mean(dim=(2, 3))
        .reshape(-1)
    )


def text_mask_from_pair(
    clean_chw: torch.Tensor,
    attacked_chw: torch.Tensor,
    side: int,
    *,
    mad_mult: float,
    relmax: float,
    absolute_floor: float,
    min_patches: int,
    max_patches: int,
) -> np.ndarray:
    diff = (
        attacked_chw.float()
        - clean_chw.float()
    ).abs().mean(dim=0)

    score = (
        patchify_score(diff, side)
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    median = float(np.median(score))
    mad_sigma = (
        float(
            np.median(
                np.abs(score - median)
            )
        )
        * 1.4826
    )
    maximum = float(np.max(score))

    if maximum <= absolute_floor:
        raise RuntimeError(
            "Paired images appear identical after preprocess; "
            f"max diff={maximum:.6g}"
        )

    threshold = max(
        median + mad_mult * mad_sigma,
        relmax * maximum,
        absolute_floor,
    )

    mask = score >= threshold

    if int(mask.sum()) < min_patches:
        top = np.argsort(score)[
            -min(min_patches, len(score)):
        ]
        mask[:] = False
        mask[top] = True

    if (
        max_patches > 0
        and int(mask.sum()) > max_patches
    ):
        top = np.argsort(score)[
            -max_patches:
        ]
        mask[:] = False
        mask[top] = True

    return mask.astype(bool)


# =============================================================================
# Explicit-QKV helpers
# =============================================================================

def normalize_probs(
    probs: torch.Tensor,
    batch: int,
    heads: int,
    tokens: int,
) -> torch.Tensor:
    p = probs

    if p.ndim == 4:
        if tuple(p.shape) == (
            batch,
            heads,
            tokens,
            tokens,
        ):
            return p

        if tuple(p.shape) == (
            heads,
            batch,
            tokens,
            tokens,
        ):
            return (
                p
                .permute(1, 0, 2, 3)
                .contiguous()
            )

    if p.ndim == 3:
        if tuple(p.shape) == (
            batch * heads,
            tokens,
            tokens,
        ):
            return p.reshape(
                batch,
                heads,
                tokens,
                tokens,
            )

    raise RuntimeError(
        f"Unexpected attention-probability shape "
        f"{tuple(p.shape)} for B={batch}, H={heads}, T={tokens}"
    )


def normalize_qkv(
    qkv: torch.Tensor,
    batch: int,
    heads: int,
    tokens: int,
) -> torch.Tensor:
    x = qkv

    if x.ndim == 4:
        if (
            x.shape[0] == batch
            and x.shape[1] == heads
            and x.shape[2] == tokens
        ):
            return x

        if (
            x.shape[0] == heads
            and x.shape[1] == batch
            and x.shape[2] == tokens
        ):
            return (
                x
                .permute(1, 0, 2, 3)
                .contiguous()
            )

    if x.ndim == 3:
        if (
            x.shape[0] == batch * heads
            and x.shape[1] == tokens
        ):
            return x.reshape(
                batch,
                heads,
                tokens,
                x.shape[-1],
            )

    raise RuntimeError(
        f"Unexpected Q/K/V shape {tuple(x.shape)} "
        f"for B={batch}, H={heads}, T={tokens}"
    )


def projected_value_norm(
    values_bhtd: torch.Tensor,
    out_proj_weight: torch.Tensor,
    heads: int,
) -> torch.Tensor:
    """
    Exact per-head residual-write norm:

        ||W_O,h V_s||

    obtained from V^T W_O,h^T W_O,h V.
    """
    values = values_bhtd.float()
    wout = (
        out_proj_weight
        .detach()
        .float()
    )

    width = int(wout.shape[0])
    head_dim = width // heads

    grams = []

    for head in range(heads):
        w_head = wout[
            :,
            head * head_dim:
            (head + 1) * head_dim,
        ]
        grams.append(
            w_head.T @ w_head
        )

    gram_hdd = torch.stack(
        grams,
        dim=0,
    )

    norm2 = torch.einsum(
        "bhtd,hde,bhte->bht",
        values,
        gram_hdd,
        values,
    )

    return torch.sqrt(
        torch.clamp(
            norm2,
            min=0.0,
        )
    )


def clear_attn_capture(block) -> None:
    for attr in (
        "last_q",
        "last_k",
        "last_v",
        "last_logits",
        "last_probs",
        "last_z",
        "last_xin",
    ):
        if hasattr(block.attn, attr):
            setattr(
                block.attn,
                attr,
                None,
            )


# =============================================================================
# Direct TEXT -> CLS traffic
# =============================================================================

@torch.inference_mode()
def run_batch(
    model,
    images_bchw: torch.Tensor,
    rta_masks_bp: np.ndarray,
    synth_masks_bp: np.ndarray,
    batch_pairs: int,
    device: torch.device,
    *,
    rn_payload: RNPayload,
    rn_enabled: bool,
):
    """
    Image order:
        [NoRTA_0...B-1, RTA_0...B-1, SynthRTA_0...B-1]

    RN, when enabled, is appended immediately BEFORE B13 as the final token.
    Spatial-patch denominators remain ordinary patches only; RN is not part
    of the TEXT or ordinary-patch denominator.
    """
    visual = model.visual

    images = images_bchw.to(
        device=device,
        dtype=torch.float32,
    )

    x = visual._prepare_tokens(images)

    patch_count = int(
        visual.positional_embedding.shape[0]
        - 1
    )

    results = {
        attack: {}
        for attack in ATTACKS
    }

    rn_inserted = False

    for block_idx, block in enumerate(
        visual.transformer.resblocks
    ):
        if (
            rn_enabled
            and not rn_inserted
            and block_idx == rn_payload.insert_block
        ):
            token = rn_payload.token.to(
                device=x.device,
                dtype=x.dtype,
            )

            if token.numel() != x.shape[-1]:
                raise RuntimeError(
                    f"RN width={token.numel()} "
                    f"!= residual width={x.shape[-1]}"
                )

            rn = (
                token
                .view(1, 1, -1)
                .expand(
                    1,
                    x.shape[1],
                    x.shape[2],
                )
            )

            x = torch.cat(
                [x, rn],
                dim=0,
            )
            rn_inserted = True

        ln1 = block.ln_1(x)

        attn_out, probs0 = block.attention(
            ln1,
            need_weights=True,
            capture=True,
        )

        if block_idx in BLOCKS:
            physical_batch = int(
                x.shape[1]
            )
            tokens = int(
                x.shape[0]
            )
            heads = int(
                block.attn.num_heads
            )

            probs = normalize_probs(
                probs0,
                physical_batch,
                heads,
                tokens,
            ).float()

            values = normalize_qkv(
                block.attn.last_v,
                physical_batch,
                heads,
                tokens,
            ).float()

            value_norm = projected_value_norm(
                values,
                block.attn.out_proj.weight,
                heads,
            )

            # Ordinary spatial patches are always token indices 1..P.
            # RN is appended at the END, so this slice is invariant.
            spatial = slice(
                1,
                1 + patch_count,
            )

            p2c_attn = probs[
                :,
                :,
                0,
                spatial,
            ]

            p2c_write = (
                p2c_attn
                * value_norm[
                    :,
                    :,
                    spatial,
                ]
            )

            total_write = (
                p2c_write
                .sum(dim=-1)
                .clamp_min(EPS)
            )

            total_attn = (
                p2c_attn
                .sum(dim=-1)
                .clamp_min(EPS)
            )

            for (
                attack,
                offset,
                masks_np,
            ) in (
                (
                    "RTA",
                    batch_pairs,
                    rta_masks_bp,
                ),
                (
                    "SynthRTA",
                    2 * batch_pairs,
                    synth_masks_bp,
                ),
            ):
                masks = torch.as_tensor(
                    masks_np,
                    device=device,
                    dtype=p2c_write.dtype,
                )[:, None, :]

                clean_write_share = (
                    (
                        p2c_write[
                            :batch_pairs
                        ]
                        * masks
                    )
                    .sum(dim=-1)
                    / total_write[
                        :batch_pairs
                    ]
                )

                attack_write_share = (
                    (
                        p2c_write[
                            offset:
                            offset + batch_pairs
                        ]
                        * masks
                    )
                    .sum(dim=-1)
                    / total_write[
                        offset:
                        offset + batch_pairs
                    ]
                )

                clean_attn_share = (
                    (
                        p2c_attn[
                            :batch_pairs
                        ]
                        * masks
                    )
                    .sum(dim=-1)
                    / total_attn[
                        :batch_pairs
                    ]
                )

                attack_attn_share = (
                    (
                        p2c_attn[
                            offset:
                            offset + batch_pairs
                        ]
                        * masks
                    )
                    .sum(dim=-1)
                    / total_attn[
                        offset:
                        offset + batch_pairs
                    ]
                )

                results[
                    attack
                ][
                    block_idx
                ] = {
                    "write_delta": (
                        attack_write_share
                        - clean_write_share
                    )
                    .cpu()
                    .numpy(),
                    "attn_delta": (
                        attack_attn_share
                        - clean_attn_share
                    )
                    .cpu()
                    .numpy(),
                }

        # Exact ordinary residual-block forward.
        x_attn = x + attn_out
        ln2 = block.ln_2(x_attn)

        x = (
            x_attn
            + block.mlp.c_proj(
                block.mlp.gelu(
                    block.mlp.c_fc(
                        ln2
                    )
                )
            )
        )

        clear_attn_capture(block)

    if rn_enabled and not rn_inserted:
        raise RuntimeError(
            "RN-on run ended without inserting RN."
        )

    return results


# =============================================================================
# Aggregation
# =============================================================================

def sem(values: np.ndarray) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    values = values[
        np.isfinite(values)
    ]

    if len(values) <= 1:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1,
        )
        / math.sqrt(
            len(values)
        )
    )


def aggregate_condition(
    write_bh: np.ndarray,
    attn_bh: np.ndarray,
):
    # Heads are a mechanism population, not independent samples.
    # Average all heads within image first, then estimate SEM across images.
    write_image = (
        np.asarray(
            write_bh,
            dtype=np.float64,
        )
        .mean(axis=1)
    )

    attn_image = (
        np.asarray(
            attn_bh,
            dtype=np.float64,
        )
        .mean(axis=1)
    )

    return {
        "write_mean": float(
            write_image.mean()
        ),
        "write_sem": sem(
            write_image
        ),
        "attn_mean": float(
            attn_image.mean()
        ),
        "attn_sem": sem(
            attn_image
        ),
        "write_image": write_image,
        "attn_image": attn_image,
    }


# =============================================================================
# Plot helpers
# =============================================================================

def global_y_limits(
    summary: pd.DataFrame,
    metric: str = "write_delta_mean",
    sem_metric: str = "write_delta_sem",
):
    lo = (
        summary[metric]
        - summary[sem_metric]
    ).min()
    hi = (
        summary[metric]
        + summary[sem_metric]
    ).max()

    lo = min(
        float(lo),
        0.0,
    )
    hi = max(
        float(hi),
        0.0,
    )

    span = max(
        hi - lo,
        1e-4,
    )

    pad = 0.07 * span
    return (
        lo - pad,
        hi + pad,
    )


def plot_single(
    summary: pd.DataFrame,
    model_name: str,
    rn_mode: str,
    out_path: Path,
    *,
    attention: bool,
    ylim=None,
) -> None:
    z = summary[
        (summary["model"] == model_name)
        & (summary["rn_mode"] == rn_mode)
    ].copy()

    fig, ax = plt.subplots(
        figsize=(8.6, 4.8)
    )

    for attack in ATTACKS:
        q = (
            z[z["attack"] == attack]
            .sort_values("block")
        )

        line, = ax.plot(
            q["block"],
            q["write_delta_mean"],
            marker="o",
            linewidth=2.0,
            label=attack,
        )

        ax.fill_between(
            q["block"].to_numpy(),
            (
                q["write_delta_mean"]
                - q["write_delta_sem"]
            ).to_numpy(),
            (
                q["write_delta_mean"]
                + q["write_delta_sem"]
            ).to_numpy(),
            alpha=0.13,
            color=line.get_color(),
        )

        if attention:
            ax.plot(
                q["block"],
                q["attn_delta_mean"],
                marker=".",
                linestyle="--",
                linewidth=1.2,
                color=line.get_color(),
                label=f"{attack} attention",
            )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )
    ax.set_xticks(
        BLOCKS
    )
    ax.set_xlabel(
        "ViT block"
    )
    ax.set_ylabel(
        r"attack $-$ NoRTA TEXT$\rightarrow$CLS exact OV/write share"
    )
    ax.set_title(
        f"{MODEL_LABELS[model_name]} — {RN_LABELS[rn_mode]}"
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_models_fixed_rn(
    summary: pd.DataFrame,
    rn_mode: str,
    out_path: Path,
    *,
    models: Sequence[str],
    ylim=None,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.6),
        sharex=True,
        sharey=True,
    )

    for ax, attack in zip(
        axes,
        ATTACKS,
    ):
        for model_name in models:
            q = (
                summary[
                    (summary["model"] == model_name)
                    & (summary["rn_mode"] == rn_mode)
                    & (summary["attack"] == attack)
                ]
                .sort_values("block")
            )

            ax.plot(
                q["block"],
                q["write_delta_mean"],
                marker="o",
                linewidth=1.9,
                label=MODEL_LABELS[model_name],
            )

        ax.axhline(
            0.0,
            linewidth=1.0,
        )
        ax.set_xticks(
            BLOCKS
        )
        ax.set_xlabel(
            "ViT block"
        )
        ax.set_title(
            attack
        )
        ax.legend(
            frameon=False,
        )

        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel(
        r"attack $-$ NoRTA TEXT$\rightarrow$CLS exact OV/write share"
    )

    label = (
        "RN off"
        if rn_mode == "rn_off"
        else "RN on"
    )

    fig.suptitle(
        f"Backbone comparison — {label}",
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_rn_compare_model(
    summary: pd.DataFrame,
    model_name: str,
    out_path: Path,
    *,
    ylim=None,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.6),
        sharex=True,
        sharey=True,
    )

    for ax, attack in zip(
        axes,
        ATTACKS,
    ):
        for rn_mode in RN_MODES:
            q = (
                summary[
                    (summary["model"] == model_name)
                    & (summary["rn_mode"] == rn_mode)
                    & (summary["attack"] == attack)
                ]
                .sort_values("block")
            )

            linestyle = (
                "-"
                if rn_mode == "rn_off"
                else "--"
            )

            ax.plot(
                q["block"],
                q["write_delta_mean"],
                marker="o",
                linestyle=linestyle,
                linewidth=2.0,
                label=RN_LABELS[rn_mode],
            )

        ax.axhline(
            0.0,
            linewidth=1.0,
        )
        ax.set_xticks(
            BLOCKS
        )
        ax.set_xlabel(
            "ViT block"
        )
        ax.set_title(
            attack
        )
        ax.legend(
            frameon=False,
        )

        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel(
        r"attack $-$ NoRTA TEXT$\rightarrow$CLS exact OV/write share"
    )

    fig.suptitle(
        f"{MODEL_LABELS[model_name]} — RN intervention",
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_full_factorial(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    ylim=None,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.4, 8.4),
        sharex=True,
        sharey=True,
    )

    for row, attack in enumerate(
        ATTACKS
    ):
        for col, rn_mode in enumerate(
            RN_MODES
        ):
            ax = axes[row, col]

            for model_name in MODEL_ORDER:
                q = (
                    summary[
                        (summary["model"] == model_name)
                        & (summary["rn_mode"] == rn_mode)
                        & (summary["attack"] == attack)
                    ]
                    .sort_values("block")
                )

                ax.plot(
                    q["block"],
                    q["write_delta_mean"],
                    marker="o",
                    linewidth=1.8,
                    label=MODEL_LABELS[model_name],
                )

            ax.axhline(
                0.0,
                linewidth=1.0,
            )
            ax.set_xticks(
                BLOCKS
            )
            ax.set_title(
                f"{attack} — {RN_LABELS[rn_mode]}"
            )

            if row == 1:
                ax.set_xlabel(
                    "ViT block"
                )

            if col == 0:
                ax.set_ylabel(
                    r"attack $-$ NoRTA exact OV/write share"
                )

            if ylim is not None:
                ax.set_ylim(*ylim)

            ax.legend(
                frameon=False,
            )

    fig.suptitle(
        "Direct TEXT→CLS traffic: backbone × RN",
        y=0.995,
    )
    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_rn_effect_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Difference-of-differences:

        RN effect
          = (attack - NoRTA)_RN_on
            - (attack - NoRTA)_RN_off

    Negative values mean RN reduces attack-relative direct TEXT->CLS write.

    The SEM below is a conservative summary-level approximation because
    summary.csv does not preserve RN-on/RN-off covariance.  The paper plot
    intentionally shows the mean trajectories without uncertainty bands.
    The full run still saves per_image.csv if paired uncertainty is wanted.
    """
    required = {
        "model",
        "rn_mode",
        "attack",
        "block",
        "write_delta_mean",
        "write_delta_sem",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise KeyError(
            f"Summary is missing required columns: {missing}"
        )

    off = (
        summary[
            summary["rn_mode"] == "rn_off"
        ][
            [
                "model",
                "attack",
                "block",
                "write_delta_mean",
                "write_delta_sem",
            ]
        ]
        .rename(
            columns={
                "write_delta_mean": "write_off",
                "write_delta_sem": "sem_off",
            }
        )
    )

    on = (
        summary[
            summary["rn_mode"] == "rn_on"
        ][
            [
                "model",
                "attack",
                "block",
                "write_delta_mean",
                "write_delta_sem",
            ]
        ]
        .rename(
            columns={
                "write_delta_mean": "write_on",
                "write_delta_sem": "sem_on",
            }
        )
    )

    effect = off.merge(
        on,
        on=[
            "model",
            "attack",
            "block",
        ],
        how="inner",
        validate="one_to_one",
    )

    effect["rn_effect"] = (
        effect["write_on"]
        - effect["write_off"]
    )

    # Conservative only; RN-on/off are paired on identical images, so the
    # exact paired SEM from per_image.csv will generally be smaller.
    effect["rn_effect_sem_conservative"] = np.sqrt(
        effect["sem_on"] ** 2
        + effect["sem_off"] ** 2
    )

    return effect


def plot_rn_causal_effect(
    effect: pd.DataFrame,
    out_path: Path,
    *,
    models: Sequence[str],
    title: str,
    mark_insert: bool = True,
) -> None:
    """
    Paper-oriented plot:
      x = B12...B23
      y = [(attack-NoRTA)_RN] - [(attack-NoRTA)_noRN]
      panels = RTA, SynthRTA
      lines = visual backbones
    """
    z = effect[
        effect["model"].isin(models)
    ].copy()

    if z.empty:
        raise RuntimeError(
            f"No RN-effect rows for models={list(models)}"
        )

    y_min = min(
        0.0,
        float(z["rn_effect"].min()),
    )
    y_max = max(
        0.0,
        float(z["rn_effect"].max()),
    )
    span = max(
        y_max - y_min,
        1e-4,
    )
    pad = 0.08 * span

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.6),
        sharex=True,
        sharey=True,
    )

    for ax, attack in zip(
        axes,
        ATTACKS,
    ):
        for model_name in models:
            q = (
                z[
                    (z["model"] == model_name)
                    & (z["attack"] == attack)
                ]
                .sort_values("block")
            )

            ax.plot(
                q["block"],
                q["rn_effect"],
                marker="o",
                linewidth=2.0,
                label=MODEL_LABELS[model_name],
            )

        ax.axhline(
            0.0,
            linewidth=1.0,
        )

        if mark_insert:
            ax.axvline(
                RN_INSERT_BLOCK,
                linestyle=":",
                linewidth=1.0,
            )

        ax.set_xticks(
            BLOCKS
        )
        ax.set_xlabel(
            "ViT block"
        )
        ax.set_title(
            attack
        )
        ax.set_ylim(
            y_min - pad,
            y_max + pad,
        )
        ax.legend(
            frameon=False,
        )

    axes[0].set_ylabel(
        r"RN effect on $\Delta$ TEXT$\rightarrow$CLS exact OV/write share"
    )

    fig.suptitle(
        title,
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_rn_effect_plots(
    summary: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Save the direct causal RN plot proposed for the paper, plus a focused
    GmP-vs-all-weights variant.
    """
    effect = build_rn_effect_summary(
        summary
    )

    effect_csv = (
        out_dir
        / "rn_effect_difference_of_differences.csv"
    )
    effect.to_csv(
        effect_csv,
        index=False,
    )

    available = set(
        effect["model"].unique()
    )

    if set(MODEL_ORDER).issubset(
        available
    ):
        plot_rn_causal_effect(
            effect,
            out_dir
            / "paper_rn_effect__all_backbones.png",
            models=MODEL_ORDER,
            title=(
                "Causal effect of pre-B13 RN on direct TEXT→CLS traffic"
            ),
        )

    focused = (
        "gmp",
        "xattn_stripped",
    )
    if set(focused).issubset(
        available
    ):
        plot_rn_causal_effect(
            effect,
            out_dir
            / "paper_rn_effect__gmp_vs_allweights.png",
            models=focused,
            title=(
                "RN acts similarly before and after all-weights adaptation"
            ),
        )

    print(
        f"[saved] {effect_csv}"
    )

    return effect


# =============================================================================
# Main experiment
# =============================================================================

def analyze_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--split",
        default="train",
    )
    parser.add_argument(
        "--gmp_checkpoint",
        default=DEFAULT_GMP_CHECKPOINT,
    )
    parser.add_argument(
        "--xattn_checkpoint",
        default=DEFAULT_XATTN_CHECKPOINT,
    )
    parser.add_argument(
        "--rn_checkpoint",
        default=DEFAULT_RN_CHECKPOINT,
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--plot_only_summary",
        default="",
        help=(
            "Skip GPU inference and regenerate plots from an existing summary.csv. "
            "Useful after the expensive 3-backbone x RN run has already finished."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--batch_pairs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "0 = all paired triplets; positive = deterministic debug subset."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260910,
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_ORDER),
        help="Comma-separated subset of pretrained,gmp,xattn_stripped",
    )
    parser.add_argument(
        "--rn_modes",
        default=",".join(RN_MODES),
        help="Comma-separated subset of rn_off,rn_on",
    )
    parser.add_argument(
        "--attention_plots",
        action="store_true",
        help=(
            "Also overlay attention-share deltas on the six individual plots."
        ),
    )

    # Same mask policy as the single-model run.
    parser.add_argument(
        "--text_diff_mad_mult",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--text_diff_relmax",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--text_diff_absolute_floor",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--text_mask_min_patches",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--text_mask_max_patches",
        type=int,
        default=64,
    )

    return parser.parse_args()


def parse_csv_set(
    raw: str,
    allowed: Sequence[str],
    label: str,
):
    values = tuple(
        x.strip()
        for x in str(raw).split(",")
        if x.strip()
    )

    unknown = sorted(
        set(values)
        - set(allowed)
    )
    if unknown:
        raise ValueError(
            f"Unknown {label}: {unknown}; "
            f"allowed={list(allowed)}"
        )

    return values


def analyze_main() -> None:
    args = analyze_parse_args()
    seed_all(args.seed)

    out_dir = Path(
        args.out_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.plot_only_summary:
        summary_path = Path(
            args.plot_only_summary
        )
        if not summary_path.is_file():
            raise FileNotFoundError(
                summary_path
            )

        summary = pd.read_csv(
            summary_path
        )

        print(
            f"[plot-only] loaded {summary_path}"
        )

        save_rn_effect_plots(
            summary,
            out_dir,
        )

        print(
            f"[plot-only] saved paper RN-effect plots under {out_dir}"
        )
        return

    ensure_clip_modules()

    device = torch.device(
        args.device
    )

    models = parse_csv_set(
        args.models,
        MODEL_ORDER,
        "models",
    )

    rn_modes = parse_csv_set(
        args.rn_modes,
        RN_MODES,
        "rn modes",
    )

    if not models:
        raise ValueError(
            "No models selected."
        )
    if not rn_modes:
        raise ValueError(
            "No RN modes selected."
        )

    rn_payload = load_rn_payload(
        args.rn_checkpoint,
        RN_INSERT_BLOCK,
    )

    pairs = load_triplets(
        args.dataset,
        args.split,
        args.limit,
        args.seed,
    )

    summary_rows = []
    per_image_rows = []

    for model_name in models:
        print()
        print("=" * 88)
        print(
            f"[model] {MODEL_LABELS[model_name]}"
        )

        audit_dir = (
            out_dir
            / "load_audit"
            / model_name
        )

        model, preprocess = load_model_variant(
            model_name,
            args,
            device,
            audit_dir,
        )

        patch_count = int(
            model.visual.positional_embedding.shape[0]
            - 1
        )
        side = int(
            round(
                math.sqrt(
                    patch_count
                )
            )
        )

        if side * side != patch_count:
            raise RuntimeError(
                f"{model_name}: non-square patch grid "
                f"P={patch_count}"
            )

        # Accumulators:
        # [rn_mode][attack][block] -> list of [B,H] arrays.
        write_chunks = {
            rn_mode: {
                attack: defaultdict(list)
                for attack in ATTACKS
            }
            for rn_mode in rn_modes
        }

        attn_chunks = {
            rn_mode: {
                attack: defaultdict(list)
                for attack in ATTACKS
            }
            for rn_mode in rn_modes
        }

        for start in tqdm(
            range(
                0,
                len(pairs),
                args.batch_pairs,
            ),
            desc=MODEL_LABELS[model_name],
        ):
            batch = pairs[
                start:
                start + args.batch_pairs
            ]
            batch_pairs = len(batch)

            clean_tensors = []
            rta_tensors = []
            synth_tensors = []

            rta_masks = []
            synth_masks = []

            for item in batch:
                clean = preprocess(
                    as_pil(
                        item["NoRTA"]["image"]
                    )
                )
                rta = preprocess(
                    as_pil(
                        item["RTA"]["image"]
                    )
                )
                synth = preprocess(
                    as_pil(
                        item["SynthRTA"]["image"]
                    )
                )

                rta_mask = text_mask_from_pair(
                    clean,
                    rta,
                    side,
                    mad_mult=args.text_diff_mad_mult,
                    relmax=args.text_diff_relmax,
                    absolute_floor=args.text_diff_absolute_floor,
                    min_patches=args.text_mask_min_patches,
                    max_patches=args.text_mask_max_patches,
                )

                synth_mask = text_mask_from_pair(
                    clean,
                    synth,
                    side,
                    mad_mult=args.text_diff_mad_mult,
                    relmax=args.text_diff_relmax,
                    absolute_floor=args.text_diff_absolute_floor,
                    min_patches=args.text_mask_min_patches,
                    max_patches=args.text_mask_max_patches,
                )

                clean_tensors.append(
                    clean
                )
                rta_tensors.append(
                    rta
                )
                synth_tensors.append(
                    synth
                )

                rta_masks.append(
                    rta_mask
                )
                synth_masks.append(
                    synth_mask
                )

            images = torch.stack(
                (
                    clean_tensors
                    + rta_tensors
                    + synth_tensors
                ),
                dim=0,
            )

            rta_masks_np = np.stack(
                rta_masks,
                axis=0,
            )
            synth_masks_np = np.stack(
                synth_masks,
                axis=0,
            )

            for rn_mode in rn_modes:
                rn_enabled = (
                    rn_mode == "rn_on"
                )

                batch_result = run_batch(
                    model,
                    images,
                    rta_masks_np,
                    synth_masks_np,
                    batch_pairs,
                    device,
                    rn_payload=rn_payload,
                    rn_enabled=rn_enabled,
                )

                for attack in ATTACKS:
                    for block in BLOCKS:
                        write_chunks[
                            rn_mode
                        ][
                            attack
                        ][
                            block
                        ].append(
                            batch_result[
                                attack
                            ][
                                block
                            ][
                                "write_delta"
                            ]
                        )

                        attn_chunks[
                            rn_mode
                        ][
                            attack
                        ][
                            block
                        ].append(
                            batch_result[
                                attack
                            ][
                                block
                            ][
                                "attn_delta"
                            ]
                        )

            del (
                images,
                clean_tensors,
                rta_tensors,
                synth_tensors,
            )

        # Aggregate this model.
        for rn_mode in rn_modes:
            for attack in ATTACKS:
                for block in BLOCKS:
                    write_bh = np.concatenate(
                        write_chunks[
                            rn_mode
                        ][
                            attack
                        ][
                            block
                        ],
                        axis=0,
                    )

                    attn_bh = np.concatenate(
                        attn_chunks[
                            rn_mode
                        ][
                            attack
                        ][
                            block
                        ],
                        axis=0,
                    )

                    agg = aggregate_condition(
                        write_bh,
                        attn_bh,
                    )

                    summary_rows.append(
                        {
                            "model": model_name,
                            "rn_mode": rn_mode,
                            "attack": attack,
                            "block": block,
                            "n_pairs": int(
                                write_bh.shape[0]
                            ),
                            "n_heads": int(
                                write_bh.shape[1]
                            ),
                            "write_delta_mean": agg[
                                "write_mean"
                            ],
                            "write_delta_sem": agg[
                                "write_sem"
                            ],
                            "attn_delta_mean": agg[
                                "attn_mean"
                            ],
                            "attn_delta_sem": agg[
                                "attn_sem"
                            ],
                        }
                    )

                    for image_index, (
                        write_value,
                        attn_value,
                    ) in enumerate(
                        zip(
                            agg["write_image"],
                            agg["attn_image"],
                        )
                    ):
                        per_image_rows.append(
                            {
                                "model": model_name,
                                "rn_mode": rn_mode,
                                "attack": attack,
                                "block": block,
                                "pair_index": image_index,
                                "write_delta": float(
                                    write_value
                                ),
                                "attn_delta": float(
                                    attn_value
                                ),
                            }
                        )

        del model
        clear_cuda()

    summary = pd.DataFrame(
        summary_rows
    )
    per_image = pd.DataFrame(
        per_image_rows
    )

    summary_csv = (
        out_dir
        / "summary.csv"
    )
    per_image_csv = (
        out_dir
        / "per_image.csv"
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )
    per_image.to_csv(
        per_image_csv,
        index=False,
    )

    print()
    print(
        "All-head attack - NoRTA TEXT->CLS exact OV/write share"
    )
    print(
        "-------------------------------------------------------"
    )

    for model_name in models:
        for rn_mode in rn_modes:
            for attack in ATTACKS:
                q = (
                    summary[
                        (summary["model"] == model_name)
                        & (summary["rn_mode"] == rn_mode)
                        & (summary["attack"] == attack)
                    ]
                    .sort_values("block")
                )

                values = (
                    q["write_delta_mean"]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                print(
                    f"{model_name:<16} "
                    f"{rn_mode:<7} "
                    f"{attack:<8}: "
                    + ", ".join(
                        f"{value:+.4f}"
                        for value in values
                    )
                )

    # Non-fatal parity check against the already-computed dedicated curve.
    qref = (
        summary[
            (summary["model"] == "xattn_stripped")
            & (summary["rn_mode"] == "rn_off")
            & (summary["attack"] == "RTA")
        ]
        .sort_values("block")
    )

    if len(qref) == len(REFERENCE_RTA):
        observed = qref[
            "write_delta_mean"
        ].to_numpy(
            dtype=np.float64
        )

        max_error = float(
            np.max(
                np.abs(
                    observed
                    - REFERENCE_RTA
                )
            )
        )

        print()
        print(
            "[parity] xattn_stripped / rn_off / RTA "
            f"max |new - previous| = {max_error:.6f}"
        )

        if max_error > 0.01:
            print(
                "[parity] WARNING: larger drift than expected; "
                "check checkpoint/dataset/preprocess identity."
            )

    # -------------------------------------------------------------------------
    # Plot zoo.  We intentionally save several variants now and choose later.
    # -------------------------------------------------------------------------
    ylim = global_y_limits(
        summary
    )

    for model_name in models:
        for rn_mode in rn_modes:
            plot_single(
                summary,
                model_name,
                rn_mode,
                out_dir
                / f"{model_name}__{rn_mode}.png",
                attention=args.attention_plots,
                ylim=ylim,
            )

    if set(MODEL_ORDER).issubset(
        set(models)
    ):
        for rn_mode in rn_modes:
            plot_models_fixed_rn(
                summary,
                rn_mode,
                out_dir
                / f"models__{rn_mode}.png",
                models=MODEL_ORDER,
                ylim=ylim,
            )

            plot_models_fixed_rn(
                summary,
                rn_mode,
                out_dir
                / f"gmp_vs_xattn__{rn_mode}.png",
                models=(
                    "gmp",
                    "xattn_stripped",
                ),
                ylim=ylim,
            )

    if set(RN_MODES).issubset(
        set(rn_modes)
    ):
        for model_name in models:
            plot_rn_compare_model(
                summary,
                model_name,
                out_dir
                / f"{model_name}__rn_compare.png",
                ylim=ylim,
            )

    if (
        set(MODEL_ORDER).issubset(
            set(models)
        )
        and set(RN_MODES).issubset(
            set(rn_modes)
        )
    ):
        plot_full_factorial(
            summary,
            out_dir
            / "all_models_x_rn_factorial.png",
            ylim=ylim,
        )

    # -------------------------------------------------------------------------
    # Main-paper candidate: isolate the causal effect of RN itself.
    #
    # y = [(attack-NoRTA)_RN_on] - [(attack-NoRTA)_RN_off]
    #
    # This removes the native attack trajectory and makes the portable RN
    # intervention directly visible across backbones.
    # -------------------------------------------------------------------------
    if set(RN_MODES).issubset(
        set(rn_modes)
    ):
        save_rn_effect_plots(
            summary,
            out_dir,
        )

    print()
    print(
        f"[saved] {summary_csv}"
    )
    print(
        f"[saved] {per_image_csv}"
    )
    print(
        f"[saved] plot variants under {out_dir}"
    )


# PLOT
matplotlib.use("Agg")


BLOCK_MIN = 12
BLOCK_MAX = 23

WRITE_METRIC = "TEXT_P2C_write_share"
ATTN_METRIC = "TEXT_P2C_attn_share"

PAIR_KEYS = ("sample_key", "comparison", "block", "head")


def parse_run(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "--run must be LABEL=path/to/native_population_traffic.csv"
        )
    label, raw_path = text.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise argparse.ArgumentTypeError("Run label is empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"CSV not found: {path}")
    return label, path


def validate_traffic(df: pd.DataFrame, path: Path) -> None:
    required = {
        *PAIR_KEYS,
        "condition",
        WRITE_METRIC,
        ATTN_METRIC,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            f"{path} is missing columns {missing}\n"
            f"Available columns include: {list(df.columns)[:30]}"
        )


def paired_deltas(
    traffic: pd.DataFrame,
    comparison: str,
) -> pd.DataFrame:
    """
    Return one row per paired sample/block/head with attacked-clean deltas.
    """
    z = traffic[
        (traffic["comparison"] == comparison)
        & traffic["block"].between(BLOCK_MIN, BLOCK_MAX)
    ].copy()

    clean = z[z["condition"] == "NoRTA"][
        [*PAIR_KEYS, WRITE_METRIC, ATTN_METRIC]
    ].copy()
    attacked = z[z["condition"] == comparison][
        [*PAIR_KEYS, WRITE_METRIC, ATTN_METRIC]
    ].copy()

    merged = clean.merge(
        attacked,
        on=list(PAIR_KEYS),
        how="inner",
        suffixes=("_clean", "_attack"),
        validate="one_to_one",
    )

    if merged.empty:
        raise RuntimeError(
            f"No paired rows found for comparison={comparison!r}"
        )

    merged["delta_write"] = (
        merged[f"{WRITE_METRIC}_attack"]
        - merged[f"{WRITE_METRIC}_clean"]
    )
    merged["delta_attn"] = (
        merged[f"{ATTN_METRIC}_attack"]
        - merged[f"{ATTN_METRIC}_clean"]
    )

    return merged


def aggregate_run(
    label: str,
    traffic: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      summary:
        one row per attack/block, with mean + SEM across images after
        averaging heads within image.
      per_image:
        one row per sample/attack/block.
    """
    per_image_parts = []

    for attack in ATTACKS:
        delta = paired_deltas(traffic, attack)

        # Head population is a mechanism population, not an independent sample.
        # Collapse heads first, then estimate uncertainty across paired images.
        image = (
            delta.groupby(
                ["sample_key", "comparison", "block"],
                as_index=False,
                sort=True,
            )
            .agg(
                delta_write=("delta_write", "mean"),
                delta_attn=("delta_attn", "mean"),
                n_heads=("head", "nunique"),
            )
        )
        image["run"] = label
        per_image_parts.append(image)

    per_image = pd.concat(per_image_parts, ignore_index=True)

    rows = []
    for (comparison, block), z in per_image.groupby(
        ["comparison", "block"],
        sort=True,
    ):
        rows.append(
            {
                "run": label,
                "comparison": comparison,
                "block": int(block),
                "n_pairs": int(z["sample_key"].nunique()),
                "n_heads_min": int(z["n_heads"].min()),
                "n_heads_max": int(z["n_heads"].max()),
                "delta_write_mean": float(z["delta_write"].mean()),
                "delta_write_sem": sem(z["delta_write"].to_numpy()),
                "delta_attn_mean": float(z["delta_attn"].mean()),
                "delta_attn_sem": sem(z["delta_attn"].to_numpy()),
            }
        )

    summary = pd.DataFrame(rows)
    return summary, per_image


def plot_one_run(
    summary: pd.DataFrame,
    label: str,
    out_dir: Path,
    show_attention: bool,
) -> list[Path]:
    zrun = summary[summary["run"] == label].copy()
    blocks = np.arange(BLOCK_MIN, BLOCK_MAX + 1)

    made = []

    # ---------------------------------------------------------
    # Main paper version: exact OV/write only.
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    for attack in ATTACKS:
        z = zrun[zrun["comparison"] == attack].set_index("block").reindex(blocks)
        y = z["delta_write_mean"].to_numpy(np.float64)
        e = z["delta_write_sem"].to_numpy(np.float64)

        line, = ax.plot(
            blocks,
            y,
            marker="o",
            linewidth=2.0,
            label=attack,
        )
        ax.fill_between(
            blocks,
            y - e,
            y + e,
            alpha=0.15,
            color=line.get_color(),
        )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(blocks)
    ax.set_xlabel("ViT block")
    ax.set_ylabel(r"attack $-$ NoRTA TEXT$\rightarrow$CLS exact OV/write share")
    ax.set_title("Direct textual traffic into CLS")
    ax.legend(frameon=False)
    fig.tight_layout()

    path = out_dir / f"{label}__TEXT_to_CLS_exact_OV_write_trajectory.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    made.append(path)

    # ---------------------------------------------------------
    # Diagnostic version: write + attention.
    # ---------------------------------------------------------
    if show_attention:
        fig, ax = plt.subplots(figsize=(9.2, 5.2))

        for attack in ATTACKS:
            z = zrun[zrun["comparison"] == attack].set_index("block").reindex(blocks)

            yw = z["delta_write_mean"].to_numpy(np.float64)
            ew = z["delta_write_sem"].to_numpy(np.float64)
            ya = z["delta_attn_mean"].to_numpy(np.float64)
            ea = z["delta_attn_sem"].to_numpy(np.float64)

            line, = ax.plot(
                blocks,
                yw,
                marker="o",
                linewidth=2.0,
                label=f"{attack}: exact OV/write",
            )
            ax.fill_between(
                blocks,
                yw - ew,
                yw + ew,
                alpha=0.13,
                color=line.get_color(),
            )
            ax.plot(
                blocks,
                ya,
                marker=".",
                linestyle="--",
                linewidth=1.5,
                color=line.get_color(),
                label=f"{attack}: attention",
            )
            ax.fill_between(
                blocks,
                ya - ea,
                ya + ea,
                alpha=0.06,
                color=line.get_color(),
            )

        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(blocks)
        ax.set_xlabel("ViT block")
        ax.set_ylabel(r"attack $-$ NoRTA share")
        ax.set_title("TEXT→CLS: exact OV/write versus attention routing")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()

        path = out_dir / f"{label}__TEXT_to_CLS_write_vs_attention.png"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        made.append(path)

    return made


def plot_all_runs(
    summary: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """
    Compact small-multiple comparison. One panel per run, write trajectory only.
    """
    labels = list(dict.fromkeys(summary["run"].tolist()))
    n = len(labels)

    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 4.1 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    blocks = np.arange(BLOCK_MIN, BLOCK_MAX + 1)

    for ax, label in zip(axes.flat, labels):
        zrun = summary[summary["run"] == label]

        for attack in ATTACKS:
            z = zrun[zrun["comparison"] == attack].set_index("block").reindex(blocks)
            y = z["delta_write_mean"].to_numpy(np.float64)
            e = z["delta_write_sem"].to_numpy(np.float64)

            line, = ax.plot(
                blocks,
                y,
                marker="o",
                linewidth=1.8,
                label=attack,
            )
            ax.fill_between(
                blocks,
                y - e,
                y + e,
                alpha=0.12,
                color=line.get_color(),
            )

        ax.axhline(0.0, linewidth=0.9)
        ax.set_title(label)
        ax.set_xticks(blocks)
        ax.set_xlabel("ViT block")
        ax.legend(frameon=False)

    for ax in axes.flat[n:]:
        ax.axis("off")

    for row in axes:
        row[0].set_ylabel(
            r"attack $-$ NoRTA TEXT$\rightarrow$CLS exact OV/write share"
        )

    fig.suptitle("Direct textual traffic into CLS", y=1.01)
    fig.tight_layout()

    path = out_dir / "ALL_RUNS__TEXT_to_CLS_exact_OV_write_trajectory.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rn_pair(
    summary: pd.DataFrame,
    off_label: str,
    on_label: str,
    out_dir: Path,
) -> Path:
    """
    Two-panel RN-off / RN-on comparison for one backbone.
    """
    blocks = np.arange(BLOCK_MIN, BLOCK_MAX + 1)

    zoff = summary[summary["run"] == off_label]
    zon = summary[summary["run"] == on_label]

    # Shared y range.
    vals = pd.concat(
        [
            zoff["delta_write_mean"] - zoff["delta_write_sem"],
            zoff["delta_write_mean"] + zoff["delta_write_sem"],
            zon["delta_write_mean"] - zon["delta_write_sem"],
            zon["delta_write_mean"] + zon["delta_write_sem"],
        ],
        ignore_index=True,
    )
    ymin = float(vals.min())
    ymax = float(vals.max())
    pad = 0.08 * max(ymax - ymin, 1e-6)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.5),
        sharex=True,
        sharey=True,
    )

    for ax, label, title in (
        (axes[0], off_label, "RN off"),
        (axes[1], on_label, "RN on"),
    ):
        zrun = summary[summary["run"] == label]

        for attack in ATTACKS:
            z = zrun[zrun["comparison"] == attack].set_index("block").reindex(blocks)
            y = z["delta_write_mean"].to_numpy(np.float64)
            e = z["delta_write_sem"].to_numpy(np.float64)

            line, = ax.plot(
                blocks,
                y,
                marker="o",
                linewidth=2.0,
                label=attack,
            )
            ax.fill_between(
                blocks,
                y - e,
                y + e,
                alpha=0.13,
                color=line.get_color(),
            )

        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(blocks)
        ax.set_xlabel("ViT block")
        ax.set_title(title)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.legend(frameon=False)

    axes[0].set_ylabel(
        r"attack $-$ NoRTA TEXT$\rightarrow$CLS exact OV/write share"
    )
    fig.suptitle(f"{off_label} vs {on_label}", y=1.02)
    fig.tight_layout()

    safe_off = off_label.replace("/", "_").replace("\\", "_")
    safe_on = on_label.replace("/", "_").replace("\\", "_")
    path = out_dir / f"RN_COMPARE__{safe_off}__VS__{safe_on}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot direct TEXT->CLS traffic from existing RTA-100 probe CSVs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=path/to/native_population_traffic.csv; repeat as needed.",
    )
    parser.add_argument(
        "--out",
        default="text_to_cls_trajectory_plots",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Also save a diagnostic write-vs-attention variant.",
    )
    parser.add_argument(
        "--rn_pair",
        action="append",
        default=[],
        help=(
            "Optional OFF_LABEL,ON_LABEL pair; repeat for multiple backbones. "
            "Example: --rn_pair final_noRN,final_RN"
        ),
    )
    return parser.parse_args()


def plot_main() -> None:
    args = plot_parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    per_images = []

    parsed_runs = [parse_run(x) for x in args.run]

    for label, path in parsed_runs:
        print(f"[load] {label}: {path}")
        traffic = pd.read_csv(path)
        validate_traffic(traffic, path)

        summary, per_image = aggregate_run(label, traffic)
        summaries.append(summary)
        per_images.append(per_image)

        print(f"[summary] {label}")
        for attack in ATTACKS:
            z = summary[summary["comparison"] == attack].sort_values("block")
            bits = " ".join(
                f"B{int(r.block):02d}={float(r.delta_write_mean):+.4f}"
                for r in z.itertuples()
            )
            print(f"  {attack:<8} {bits}")

    summary_all = pd.concat(summaries, ignore_index=True)
    per_image_all = pd.concat(per_images, ignore_index=True)

    summary_path = out_dir / "TEXT_to_CLS_exact_OV_write_trajectory_summary.csv"
    per_image_path = out_dir / "TEXT_to_CLS_exact_OV_write_trajectory_per_image.csv"
    summary_all.to_csv(summary_path, index=False)
    per_image_all.to_csv(per_image_path, index=False)

    made = []
    for label, _path in parsed_runs:
        made.extend(
            plot_one_run(
                summary=summary_all,
                label=label,
                out_dir=out_dir,
                show_attention=args.attention,
            )
        )

    if len(parsed_runs) > 1:
        made.append(plot_all_runs(summary_all, out_dir))

    for spec in args.rn_pair:
        if "," not in spec:
            raise ValueError("--rn_pair must be OFF_LABEL,ON_LABEL")
        off_label, on_label = [x.strip() for x in spec.split(",", 1)]
        known = set(summary_all["run"])
        if off_label not in known or on_label not in known:
            raise KeyError(
                f"Unknown --rn_pair labels: {off_label!r}, {on_label!r}; "
                f"known={sorted(known)}"
            )
        made.append(
            plot_rn_pair(
                summary=summary_all,
                off_label=off_label,
                on_label=on_label,
                out_dir=out_dir,
            )
        )

    print(f"[save] {summary_path}")
    print(f"[save] {per_image_path}")
    for path in made:
        print(f"[plot] {path}")


def main(argv=None):
    """Dispatch a workflow; each subcommand retains its original CLI options."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {'analyze': analyze_main, 'plot': plot_main}
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("\nCommands: " + ", ".join(commands))
        print("Use: python " + __file__ + " COMMAND --help")
        return
    command = argv.pop(0)
    if command not in commands:
        raise SystemExit("Unknown command: " + command)
    previous = sys.argv
    sys.argv = [previous[0] + " " + command, *argv]
    try:
        return commands[command]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()

