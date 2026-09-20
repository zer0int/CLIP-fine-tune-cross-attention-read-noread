"""MSCOCO I2T/T2I retrieval for current PIECES OF CLIP.

Compares three meaningful lanes on the SAME moved backbone:
  classic  = raw final-CLS image encoder + ordinary text encoder; no PIECES correction/routing
  <notext> = PIECES semantic/content lane with CONTENT correction
  <any>    = PIECES automatic candidate-conditioned routing

With --full_corr_off, the full model additionally exposes <notext> and <any>
with only the separately trained CONTENT correction disabled. RN and the
SOURCE/ORTHO/READ/router machinery remain active.

Vanilla backbone storage follows the checkpoint; CUDA execution uses FP16
autocast while PIECES modules remain FP32. Cached embeddings are FP32.
No T2T, II, TF-IDF, null audit, CLI tuning jungle, or score memmaps.

Classic and <notext> retrieval are separable. PIECES <any> is candidate-conditioned,
so exact <any> retrieval still needs pairwise image-caption scoring. To keep that fast,
this script caches the image/text encoder work once and evaluates only the pairwise
PIECES tail. A preflight compares cached scores against the model's actual public/current
paths and aborts if they disagree.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow both `python -m benchmarks.mscoco` and direct script execution.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import oaiclip as clip
from utils_clip_loader.benchmark_runtime import (
    full_xattn_correction_variants,
    full_xattn_variant_mode_key,
    full_xattn_variant_mode_label,
    inference_autocast,
    is_full_xattn,
    normalized_image_features,
    normalized_text_features,
    separable_modes,
)
from benchmark_utils.models import (
    DEFAULT_MODEL_ALIAS,
    DEFAULT_MODEL_PATH,
    ModelSpec,
    load_model_spec,
    model_display_name,
    model_file_token,
)


# =============================================================================
# Fixed config
# =============================================================================

COCO_IMG_DIR = "path/to/COCO/val2014"
JSON_PATH = "utils_datasets/coco/coco_val_karpathy.json"
SPLIT = "val"
MAX_IMAGES = 0
BENCHMARK_NAME = "mscoco"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = True
IMAGE_BATCH = 32
TEXT_BATCH = 512
PAIR_TEXT_BLOCK = 512
NUM_WORKERS = 6
PREFLIGHT_ATOL = 2e-3



# =============================================================================
# COCO data
# =============================================================================

@dataclass
class CocoEntry:
    path: str
    captions: List[str]


def seed_everything(seed: int = 20260829):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _looks_like_coco_captions_json(data: dict) -> bool:
    return isinstance(data, dict) and ("images" in data) and ("annotations" in data)


def load_coco_entries_auto() -> List[CocoEntry]:
    """Match the original evaluator's COCO loading behavior exactly.

    Supports:
      1) Karpathy split JSON
      2) Official COCO captions annotations JSON

    For official annotations JSON, SPLIT is intentionally ignored.
    """
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[CocoEntry] = []

    # Official COCO captions annotations JSON.
    if _looks_like_coco_captions_json(data):
        id_to_file: Dict[int, str] = {}
        for image in data["images"]:
            if "id" in image and ("file_name" in image or "filename" in image):
                filename = image.get("file_name", image.get("filename"))
                id_to_file[int(image["id"])] = str(filename)

        id_to_caps: Dict[int, List[str]] = {image_id: [] for image_id in id_to_file}
        for annotation in data["annotations"]:
            image_id = int(annotation.get("image_id"))
            caption = annotation.get("caption")
            if caption is not None and image_id in id_to_caps:
                id_to_caps[image_id].append(str(caption))

        for image_id, filename in id_to_file.items():
            captions = id_to_caps.get(image_id, [])
            if not captions:
                continue
            path = os.path.join(COCO_IMG_DIR, filename)
            if not os.path.isfile(path):
                continue
            out.append(CocoEntry(path, captions))

    # Karpathy split JSON.
    else:
        rows = data["images"] if isinstance(data, dict) and "images" in data else data
        for row in rows:
            if not isinstance(row, dict):
                continue

            if SPLIT is not None and "split" in row:
                if str(row.get("split", "")).lower() != str(SPLIT).lower():
                    continue

            filename = row.get("filename") or row.get("file_name")
            if filename is None:
                continue

            path = os.path.join(COCO_IMG_DIR, str(filename))
            if not os.path.isfile(path):
                continue

            raw_sentences = row.get("sentences", row.get("captions", []))
            captions: List[str] = []
            for sentence in raw_sentences:
                if isinstance(sentence, dict):
                    raw = sentence.get("raw") or sentence.get("caption") or sentence.get("text")
                    if raw is not None:
                        captions.append(str(raw))
                elif isinstance(sentence, str):
                    captions.append(sentence)

            if captions:
                out.append(CocoEntry(path, captions))

    if MAX_IMAGES > 0:
        out = out[:MAX_IMAGES]
    return out


def caption_index(entries: Sequence[CocoEntry]):
    captions: List[str] = []
    img_to_caps: Dict[int, List[int]] = {}
    for image_idx, entry in enumerate(entries):
        img_to_caps[image_idx] = []
        for caption in entry.captions:
            idx = len(captions)
            captions.append(caption)
            img_to_caps[image_idx].append(idx)
    return captions, img_to_caps


class CocoImages(Dataset):
    def __init__(self, entries, preprocess):
        self.entries = entries
        self.preprocess = preprocess

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        with open(self.entries[idx].path, "rb") as f:
            image = Image.open(f).convert("RGB")
        return self.preprocess(image), idx


def collate(batch):
    images, indices = zip(*batch)
    return torch.stack(images), torch.tensor(indices, dtype=torch.long)


def loader_for(entries, preprocess):
    return DataLoader(
        CocoImages(entries, preprocess),
        batch_size=IMAGE_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate,
    )


# =============================================================================
# Current PIECES scoring, cached but verified against forward_modes()
# =============================================================================


def amp():
    if DEVICE == "cuda" and AMP:
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def helpers(model):
    mod = importlib.import_module(model.__class__.__module__)
    names = ("_fp32_normalize", "_fp32_scaled_matmul", "_fp32_scaled_einsum")
    if any(not hasattr(mod, name) for name in names):
        raise RuntimeError(f"{mod.__name__} is not the expected current PIECES implementation")
    return tuple(getattr(mod, name) for name in names)


@dataclass
class TextBank:
    classic: torch.Tensor   # CPU FP32, ordinary encode_text(caption)
    content: torch.Tensor   # CPU FP32, PIECES semantic/content target
    read: torch.Tensor      # CPU FP32, PIECES <text> target
    query: torch.Tensor     # CPU FP32, PIECES <text> EOT query


@dataclass
class ImageCache:
    info: Dict[str, Any]
    classic: torch.Tensor
    content_corrected: torch.Tensor
    content_base: torch.Tensor
    source_logits: torch.Tensor
    source_stats: torch.Tensor
    source_gate: torch.Tensor
    null: torch.Tensor
    injection_corrected: torch.Tensor
    injection_base: torch.Tensor


@torch.inference_mode()
def make_text_bank(model, captions: Sequence[str]) -> TextBank:
    normalize, _, _ = helpers(model)
    classic, content, read, query = [], [], [], []

    for start in tqdm(range(0, len(captions), TEXT_BATCH), desc="cache classic + PIECES text targets"):
        chunk = captions[start:start + TEXT_BATCH]

        # Classic target: exactly the ordinary text encoder on the untagged caption.
        plain_tokens = clip.tokenize(chunk, truncate=True).to(DEVICE)

        # PIECES target: let the current model parse <any>; do not invent token placement.
        any_tokens = clip.tokenize(["<any> " + x for x in chunk], truncate=True).to(DEVICE)
        prepared = model.prepare_mode_tokens(any_tokens)
        if not prepared["modes"].eq(0).all():
            raise RuntimeError("<any> did not parse as PIECES any mode")

        with amp():
            classic_embedding = model.encode_text(plain_tokens)
            c = model._encode_text_hidden(prepared["content_tokens"])
            r = model._encode_text_hidden(prepared["read_tokens"])

        classic.append(normalize(classic_embedding).float().cpu())
        content.append(normalize(c["text_embedding"]).float().cpu())
        read.append(normalize(r["text_embedding"]).float().cpu())
        query.append(r["eot_hidden_pre_ln"].float().cpu())

    return TextBank(
        torch.cat(classic),
        torch.cat(content),
        torch.cat(read),
        torch.cat(query),
    )


@torch.inference_mode()
def make_image_cache(model, images: torch.Tensor) -> ImageCache:
    normalize, _, einsum = helpers(model)
    images = images.to(DEVICE, non_blocking=(DEVICE == "cuda"))

    # Current models here are sigmoid_all. The old sigmoid_mass formula is different.
    if str(getattr(model, "read_attention_architecture", "")) == "sigmoid_mass":
        raise RuntimeError("This evaluator intentionally does not support old sigmoid_mass checkpoints")

    with amp():
        info = model.encode_image_states(images, return_final_tokens=False)

        # Classic image target: raw current-backbone final CLS, before PIECES CONTENT correction.
        classic = normalize(info["image_embedding"])

        # Full-model semantic image targets with and without the separately
        # trained CONTENT correction. Both reuse the same captured ViT states.
        content_corrected = normalize(
            model._content_image_from_info(info, apply_content_correction=True)
        )
        content_base = classic
        source_logits, _glyph_logits, source_stats = model.read_implant.source_outputs(
            info["states"], return_details=False
        )
        p = source_logits.sigmoid()
        source_gate = p[:, 0] * p[:, 1]

        null_tokens = model._null_read_tokens(images.device)
        null_info = model._encode_text_hidden(null_tokens)
        null_text = normalize(null_info["text_embedding"])
        null_feature = model.read_implant.read_features(
            info["states"], null_info["eot_hidden_pre_ln"], register_mask=info["register_mask"]
        )
        null_feature = normalize(null_feature[:, 0, :])
        scale = model.logit_scale.float().exp()
        raw_null = einsum("bd,d->b", scale, null_feature, null_text[0])
        null = model.read_implant.calibrate_read_logits(
            raw_null[:, None], source_logits,
            null_mask=torch.ones(1, dtype=torch.bool, device=DEVICE),
        )[:, 0]
        injection_corrected = model.read_implant.injection_feature(content_corrected)
        injection_base = model.read_implant.injection_feature(content_base)

    return ImageCache(
        info=info,
        classic=classic,
        content_corrected=content_corrected,
        content_base=content_base,
        source_logits=source_logits,
        source_stats=source_stats,
        source_gate=source_gate,
        null=null,
        injection_corrected=injection_corrected,
        injection_base=injection_base,
    )


@torch.inference_mode()
def score(
    model,
    cache: ImageCache,
    bank: TextBank,
    caption_ids: Sequence[int],
    *,
    include_corr_off: bool = False,
) -> Dict[str, torch.Tensor]:
    """Exact retrieval scores for one image batch x caption block.

    READ/ORTHO/source work is computed once. When requested, the correction-off
    semantic/routing lane reuses those exact states and differs only in the
    separately parameterized CONTENT residual and downstream router inputs that
    depend on that semantic image embedding.
    """
    normalize, matmul, einsum = helpers(model)
    ids = torch.as_tensor(caption_ids, dtype=torch.long)
    classic_text = bank.classic[ids].to(DEVICE, non_blocking=(DEVICE == "cuda"))
    content_text = bank.content[ids].to(DEVICE, non_blocking=(DEVICE == "cuda"))
    read_text = bank.read[ids].to(DEVICE, non_blocking=(DEVICE == "cuda"))
    query = bank.query[ids].to(DEVICE, non_blocking=(DEVICE == "cuda"))

    with amp():
        scale = model.logit_scale.float().exp()
        classic = matmul(scale, cache.classic, classic_text.t())

        read_feature = normalize(model.read_implant.read_features(
            cache.info["states"], query, register_mask=cache.info["register_mask"]
        ))
        ortho_feature = normalize(
            model.read_implant.orthographic_features(cache.info["states"], query)
        )
        raw_read = einsum("bnd,nd->bn", scale, read_feature, read_text)
        early = einsum("bnd,nd->bn", scale, ortho_feature, read_text)
        read_logits = model.read_implant.calibrate_read_logits(
            raw_read,
            cache.source_logits,
            null_mask=torch.zeros(
                len(caption_ids), dtype=torch.bool, device=DEVICE
            ),
        )
        positive = model.read_implant.positive_relative_read(
            read_logits - cache.null[:, None]
        ).detach()

        def semantic_and_any(content_image, injection):
            notext = matmul(scale, content_image, content_text.t())
            trust = model.read_implant.trust_gate(
                content_logits=notext,
                read_logits=read_logits,
                null_logits=cache.null,
                early_logits=early,
                content_image=content_image,
                read_image=read_feature,
                content_text=content_text,
                read_text=read_text,
                source_logits=cache.source_logits,
                source_stats=cache.source_stats,
                injection=injection,
            )
            route = trust * cache.source_gate[:, None].detach()
            any_logits = notext + (
                route.to(positive.dtype)
                * model.read_implant.auto_read_scale.to(positive.dtype)
                * positive
            ).to(notext.dtype)
            return notext.float(), any_logits.float()

        notext, any_logits = semantic_and_any(
            cache.content_corrected,
            cache.injection_corrected,
        )
        outputs: Dict[str, torch.Tensor] = {
            "classic": classic.float(),
            "notext": notext,
            "any": any_logits,
        }

        if include_corr_off:
            notext_off, any_off = semantic_and_any(
                cache.content_base,
                cache.injection_base,
            )
            outputs["notext_corr_off"] = notext_off
            outputs["any_corr_off"] = any_off

    return outputs


def full_retrieval_mode_labels(include_corr_off: bool = False) -> Dict[str, str]:
    labels = {
        "classic": "classic (RN backbone)",
        "notext": "<notext> / correction",
        "any": "<any>",
    }
    if include_corr_off:
        off = full_xattn_correction_variants(True)[-1]
        labels[full_xattn_variant_mode_key("notext", off)] = (
            full_xattn_variant_mode_label("<notext>", off)
        )
        labels[full_xattn_variant_mode_key("any", off)] = (
            full_xattn_variant_mode_label("<any>", off)
        )
    return labels


@torch.inference_mode()
def preflight(
    model,
    images,
    captions,
    bank,
    *,
    include_corr_off: bool = False,
):
    """Check the streamed scorer against public ``forward_modes`` paths."""
    n_img = min(2, images.shape[0])
    n_txt = min(TEXT_BATCH, PAIR_TEXT_BLOCK, len(captions))
    ids = list(range(n_txt))
    check_images = images[:n_img].to(DEVICE)

    cache = make_image_cache(model, images[:n_img])
    cached = score(
        model,
        cache,
        bank,
        ids,
        include_corr_off=include_corr_off,
    )

    classic_text = bank.classic[:n_txt].to(
        DEVICE, non_blocking=(DEVICE == "cuda")
    )
    normalize, matmul, _ = helpers(model)
    with amp():
        classic_image = normalize(model.encode_image_base(check_images))
        classic_ref = matmul(
            model.logit_scale.float().exp(),
            classic_image,
            classic_text.t(),
        )

    nt_tokens = clip.tokenize(
        ["<notext> " + captions[i] for i in ids], truncate=True
    ).to(DEVICE)
    any_tokens = clip.tokenize(
        ["<any> " + captions[i] for i in ids], truncate=True
    ).to(DEVICE)

    refs: Dict[str, torch.Tensor] = {"classic": classic_ref.float()}
    for variant in full_xattn_correction_variants(include_corr_off):
        for base_key, tokens in (("notext", nt_tokens), ("any", any_tokens)):
            with amp():
                output = model.forward_modes(
                    check_images,
                    tokens,
                    apply_content_correction=variant.apply_content_correction,
                    return_details=False,
                )
            key = full_xattn_variant_mode_key(base_key, variant)
            refs[key] = output[0].float()

    errors = {
        key: float((refs[key] - cached[key]).abs().max())
        for key in refs
    }
    print(
        f"[preflight matched block n={n_txt}] "
        + " | ".join(f"{key}={value:.6g}" for key, value in errors.items())
    )
    for name, err in errors.items():
        if err > PREFLIGHT_ATOL:
            raise RuntimeError(
                f"{name} matched-batch scoring mismatch "
                f"({err:.6g} > {PREFLIGHT_ATOL}); refusing to continue"
            )


# =============================================================================
# Exact I2T + T2I ranks, streamed without storing a 5000 x 25000 matrix
# =============================================================================

@torch.inference_mode()
def positive_thresholds(
    model,
    dl,
    bank,
    img_to_caps,
    n_images,
    n_caps,
    *,
    include_corr_off: bool = False,
):
    mode_keys = tuple(full_retrieval_mode_labels(include_corr_off))
    best = {key: torch.empty(n_images, dtype=torch.float32) for key in mode_keys}
    positive = {key: torch.empty(n_caps, dtype=torch.float32) for key in mode_keys}

    for images, image_ids_t in tqdm(dl, desc="positive-pair pass", unit="batch"):
        cache = make_image_cache(model, images)
        image_ids = [int(x) for x in image_ids_t.tolist()]
        flat, local = [], []
        for image_id in image_ids:
            own = img_to_caps[image_id]
            start = len(flat)
            flat.extend(own)
            local.append(list(range(start, start + len(own))))

        scores = score(
            model,
            cache,
            bank,
            flat,
            include_corr_off=include_corr_off,
        )
        scores = {key: value.cpu() for key, value in scores.items()}

        for row, image_id in enumerate(image_ids):
            cols = local[row]
            own = img_to_caps[image_id]
            for key in mode_keys:
                best[key][image_id] = scores[key][row, cols].max()
                for col, caption_id in zip(cols, own):
                    positive[key][caption_id] = scores[key][row, col]

        del cache
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {"best": best, "positive": positive}


@torch.inference_mode()
def exact_ranks(
    model,
    dl,
    bank,
    thresholds,
    img_to_caps,
    n_images,
    n_caps,
    *,
    include_corr_off: bool = False,
):
    """Exact ranks, counting only negatives that beat each positive."""
    mode_keys = tuple(full_retrieval_mode_labels(include_corr_off))
    best = thresholds["best"]
    positive = thresholds["positive"]

    better_i2t = {
        key: torch.zeros(n_images, dtype=torch.int64) for key in mode_keys
    }
    better_t2i = {
        key: torch.zeros(n_caps, dtype=torch.int64) for key in mode_keys
    }

    cap_to_img = torch.empty(n_caps, dtype=torch.long)
    for image_id, own_caps in img_to_caps.items():
        for caption_id in own_caps:
            cap_to_img[int(caption_id)] = int(image_id)

    for images, image_ids_t in tqdm(dl, desc="full retrieval pass", unit="batch"):
        cache = make_image_cache(model, images)
        image_ids = [int(x) for x in image_ids_t.tolist()]
        i2t_threshold = {
            key: best[key][image_ids].to(DEVICE)[:, None]
            for key in mode_keys
        }

        for start in range(0, n_caps, PAIR_TEXT_BLOCK):
            end = min(n_caps, start + PAIR_TEXT_BLOCK)
            ids = list(range(start, end))
            scores = score(
                model,
                cache,
                bank,
                ids,
                include_corr_off=include_corr_off,
            )

            gt_mask = torch.zeros(
                (len(image_ids), end - start),
                dtype=torch.bool,
                device=DEVICE,
            )
            for row, image_id in enumerate(image_ids):
                for caption_id in img_to_caps[image_id]:
                    if start <= caption_id < end:
                        gt_mask[row, caption_id - start] = True

            block_pos_images = cap_to_img[start:end].to(DEVICE)
            row_image_ids = torch.as_tensor(image_ids, device=DEVICE)[:, None]
            positive_image_mask = row_image_ids.eq(block_pos_images[None, :])

            for key in mode_keys:
                values = scores[key]
                better_i2t[key][image_ids] += (
                    (values > i2t_threshold[key]) & ~gt_mask
                ).sum(1).cpu()
                better_t2i[key][start:end] += (
                    (values > positive[key][start:end].to(DEVICE)[None, :])
                    & ~positive_image_mask
                ).sum(0).cpu()

        del cache
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        key: {
            "I2T": better_i2t[key] + 1,
            "T2I": better_t2i[key] + 1,
        }
        for key in mode_keys
    }


def metrics(ranks: torch.Tensor):
    r = ranks.double()
    return {
        "R@1": float((r <= 1).double().mean()),
        "R@5": float((r <= 5).double().mean()),
        "R@10": float((r <= 10).double().mean()),
        "MedR": float(r.median()),
        "MeanR": float(r.mean()),
        "n": int(r.numel()),
    }


def show(label, m):
    print(
        f"{label:<12} R@1={m['R@1']:.6f}  R@5={m['R@5']:.6f}  "
        f"R@10={m['R@10']:.6f}  MedR={m['MedR']:.2f}  MeanR={m['MeanR']:.2f}"
    )


@torch.inference_mode()
def make_generic_text_features(model, captions: Sequence[str]) -> torch.Tensor:
    chunks = []
    for start in tqdm(
        range(0, len(captions), TEXT_BATCH),
        desc="encode captions",
        unit="batch",
    ):
        tokens = clip.tokenize(
            captions[start : start + TEXT_BATCH], truncate=True
        ).to(DEVICE)
        with inference_autocast(DEVICE, AMP):
            chunks.append(normalized_text_features(model, tokens).cpu())
    return torch.cat(chunks)


@torch.inference_mode()
def make_generic_image_features(model, dl, mode) -> torch.Tensor:
    chunks = []
    for images, _image_ids in tqdm(dl, desc=f"encode images [{mode.key}]", unit="batch"):
        images = images.to(DEVICE, non_blocking=DEVICE == "cuda")
        with inference_autocast(DEVICE, AMP):
            chunks.append(normalized_image_features(model, images, mode).cpu())
    return torch.cat(chunks)


@torch.inference_mode()
def generic_exact_ranks(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    img_to_caps: Dict[int, List[int]],
):
    n_images, n_caps = image_features.shape[0], text_features.shape[0]
    cap_to_img = torch.empty(n_caps, dtype=torch.long)
    best_i2t = torch.empty(n_images, dtype=torch.float32)
    positive_t2i = torch.empty(n_caps, dtype=torch.float32)
    for image_id, caption_ids in img_to_caps.items():
        own = text_features[caption_ids]
        scores = image_features[image_id].float() @ own.float().t()
        best_i2t[image_id] = scores.max()
        for local_index, caption_id in enumerate(caption_ids):
            positive_t2i[caption_id] = scores[local_index]
            cap_to_img[caption_id] = image_id

    better_i2t = torch.zeros(n_images, dtype=torch.int64)
    better_t2i = torch.zeros(n_caps, dtype=torch.int64)
    for image_start in tqdm(
        range(0, n_images, IMAGE_BATCH), desc="exact retrieval ranks", unit="batch"
    ):
        image_end = min(n_images, image_start + IMAGE_BATCH)
        image_ids = torch.arange(image_start, image_end)
        image_block = image_features[image_start:image_end].to(DEVICE)
        i2t_threshold = best_i2t[image_start:image_end].to(DEVICE)[:, None]
        for caption_start in range(0, n_caps, PAIR_TEXT_BLOCK):
            caption_end = min(n_caps, caption_start + PAIR_TEXT_BLOCK)
            text_block = text_features[caption_start:caption_end].to(DEVICE)
            scores = image_block.float() @ text_block.float().t()
            caption_ids = torch.arange(caption_start, caption_end)

            gt_mask = torch.zeros_like(scores, dtype=torch.bool)
            for row, image_id in enumerate(image_ids.tolist()):
                for caption_id in img_to_caps[image_id]:
                    if caption_start <= caption_id < caption_end:
                        gt_mask[row, caption_id - caption_start] = True
            better_i2t[image_start:image_end] += (
                (scores > i2t_threshold) & ~gt_mask
            ).sum(dim=1).cpu()

            positive_image_mask = image_ids.to(DEVICE)[:, None].eq(
                cap_to_img[caption_start:caption_end].to(DEVICE)[None, :]
            )
            better_t2i[caption_start:caption_end] += (
                (
                    scores
                    > positive_t2i[caption_start:caption_end].to(DEVICE)[None, :]
                )
                & ~positive_image_mask
            ).sum(dim=0).cpu()

    return {"I2T": better_i2t + 1, "T2I": better_t2i + 1}


@torch.inference_mode()
def evaluate_generic(model, info, entries, captions, img_to_caps, preprocess):
    dl = loader_for(entries, preprocess)
    text_features = make_generic_text_features(model, captions)
    tasks = {}
    mode_labels = {}
    for mode in separable_modes(model, info):
        image_features = make_generic_image_features(model, dl, mode)
        ranks = generic_exact_ranks(image_features, text_features, img_to_caps)
        mode_labels[mode.key] = mode.label
        tasks[mode.key] = {}
        for task in ("I2T", "T2I"):
            tasks[mode.key][task] = metrics(ranks[task])
            show(f"{mode.key} {task}", tasks[mode.key][task])
    return tasks, mode_labels


@torch.inference_mode()
def evaluate(
    spec: ModelSpec,
    entries,
    captions,
    img_to_caps,
    *,
    include_corr_off: bool = False,
):
    alias, path = spec.alias, spec.path
    print("\n" + "=" * 100)
    print(f"[Model] {alias}\n{path}")
    print("=" * 100)

    model, preprocess, info = load_model_spec(clip, spec, device=DEVICE)
    print(f"[loader] family={info.model_family} source={info.source_kind}")
    out = {
        "alias": alias,
        "model": path,
        "base_model_or_path": spec.base_model_or_path,
        "model_family": info.model_family,
        "full_corr_off": bool(include_corr_off),
        "tasks": {},
        "mode_labels": {},
    }
    if is_full_xattn(model, info):
        bank = make_text_bank(model, captions)
        dl = loader_for(entries, preprocess)
        first_images, _ = next(iter(dl))
        preflight(
            model,
            first_images,
            captions,
            bank,
            include_corr_off=include_corr_off,
        )
        thresholds = positive_thresholds(
            model,
            dl,
            bank,
            img_to_caps,
            len(entries),
            len(captions),
            include_corr_off=include_corr_off,
        )
        ranks = exact_ranks(
            model,
            dl,
            bank,
            thresholds,
            img_to_caps,
            len(entries),
            len(captions),
            include_corr_off=include_corr_off,
        )
        out["mode_labels"] = full_retrieval_mode_labels(include_corr_off)
        for mode in out["mode_labels"]:
            out["tasks"][mode] = {}
            for task in ("I2T", "T2I"):
                result = metrics(ranks[mode][task])
                out["tasks"][mode][task] = result
                show(f"{mode} {task}", result)
        del bank
    else:
        out["tasks"], out["mode_labels"] = evaluate_generic(
            model, info, entries, captions, img_to_caps, preprocess
        )

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# =============================================================================
# Small, model-isolated outputs
# =============================================================================


def save_results(result, n_images, n_caps, output_dir, spec):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": n_images,
        "captions": n_caps,
        "amp": AMP,
        **result,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    rows = []
    for mode, tasks in result["tasks"].items():
        for task, values in tasks.items():
            rows.append(
                {
                    "alias": result["alias"],
                    "model": result["model"],
                    "model_family": result["model_family"],
                    "mode": mode,
                    "mode_label": result["mode_labels"][mode],
                    "task": task,
                    **values,
                }
            )
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"MSCOCO retrieval — {result['alias']}",
        "",
        "mode\ttask\tR@1\tR@5\tR@10\tMedR\tMeanR",
    ]
    for row in rows:
        lines.append(
            f"{row['mode_label']}\t{row['task']}\t{row['R@1']:.6f}\t"
            f"{row['R@5']:.6f}\t{row['R@10']:.6f}\t"
            f"{row['MedR']:.2f}\t{row['MeanR']:.2f}"
        )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = [f"{row['mode_label']} {row['task']}" for row in rows]
    x = range(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.4), 5.8))
    for offset, metric, color in (
        (-width, "R@1", "#8BCF68"),
        (0.0, "R@5", "#F2AD63"),
        (width, "R@10", "#EA7B7B"),
    ):
        ax.bar([value + offset for value in x], [row[metric] for row in rows], width, label=metric, color=color)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(list(x), labels, rotation=30, ha="right")
    ax.set_ylabel("Recall")
    ax.set_title(f"MSCOCO retrieval — {model_display_name(spec)}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"recall_{model_file_token(spec)}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"\n[Saved] {output_dir}")


def main():
    global COCO_IMG_DIR, JSON_PATH
    parser = argparse.ArgumentParser(description="MSCOCO retrieval benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--base-model-or-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("out_bench_results/mscoco"))
    parser.add_argument("--coco-img-dir", type=Path, default=Path(COCO_IMG_DIR))
    parser.add_argument("--json-path", type=Path, default=Path(JSON_PATH))
    parser.add_argument(
        "--full_corr_off",
        "--full-corr-off",
        dest="full_corr_off",
        action="store_true",
        help=(
            "For full x-attention models, also evaluate <notext> and <any> with "
            "only the separate CONTENT correction disabled. Classic is already "
            "the raw correction-free backbone lane."
        ),
    )
    args = parser.parse_args()
    COCO_IMG_DIR = str(args.coco_img_dir)
    JSON_PATH = str(args.json_path)

    seed_everything()
    entries = load_coco_entries_auto()
    if not entries:
        raise RuntimeError(
            "No COCO images found after parsing/filtering. "
            "Check COCO_IMG_DIR / JSON_PATH; SPLIT applies only to Karpathy JSON."
        )
    captions, img_to_caps = caption_index(entries)

    print("\n===================================================")
    print("MSCOCO retrieval — classic vs PIECES <notext> vs <any>")
    print("===================================================")
    print(f"[Device] {DEVICE} | native model storage, FP32 cache | AMP={AMP}")
    print(f"[Data]   images={len(entries):,} captions={len(captions):,}")
    mode_note = "classic raw backbone, <notext> CONTENT, <any> auto-route"
    if args.full_corr_off:
        mode_note += ", plus <notext>/<any> with CONTENT correction off"
    print(f"[Modes]  {mode_note}")
    print("[Tasks]  I2T, T2I")
    print(f"[Out]    {args.output_dir}")

    spec = ModelSpec(args.model_alias, args.model, args.base_model_or_path)
    for i, spec in enumerate((spec,), 1):
        print("\n[Run] 1/1")
        result = evaluate(
            spec,
            entries,
            captions,
            img_to_caps,
            include_corr_off=args.full_corr_off,
        )
        output_dir = args.output_dir
        save_results(result, len(entries), len(captions), output_dir, spec)


if __name__ == "__main__":
    main()
