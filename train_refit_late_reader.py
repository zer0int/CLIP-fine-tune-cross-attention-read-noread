#!/usr/bin/env python
"""
Native late READ / READ_NULL refit for the final HF model.

This script retrains the late reader exactly where it already exists.  It does not
change tap topology, substitute visual states, add blocks, remove blocks, or create a
special remote-code model class.

Native topology
---------------
* late READ taps: B20/B21
* SOURCE taps: B6/B7/B10
* ORTHO taps: B8/B12/B13
* RN insertion: before B13

Trainable
---------
* read_implant.read_bridge
* read_implant.read_tap_logits
* read_implant.glyph_bias_beta
* read_implant.null_abstain_weight in 1C (frozen during 1B.5, matching the original phase)

Frozen
------
* entire vision backbone
* RN / read_null_token
* text encoder
* hard_text_embedding and null_text_embedding
* SOURCE
* ORTHO
* CONTENT/correction
* trust router
* read_calibration_scale
* every other parameter

Data / schedule / losses
------------------------
Reads the prepared repository training config (``training_config.local.json`` by default,
falling back to ``training_config.json``) and imports the same packet builders and loss
helpers from ``main_hard_text_gate/train_gmp_anytext_stage1.py``.

It replays only the RN-active final-base curriculum:
    1B.5 = phase_1b5_epochs * steps_per_epoch
    1C   = phase_1c_epochs   * steps_per_epoch

The reader is optimized with the original SOURCE-ranking contribution and the original
READ_NULL attention objective.  SCAM, RTA, MVT, eval_bench, and image_sets/misc are never
training sources.

The result is saved as a complete ordinary full-xattn HF repo using the same
``configuration_xattn_clip.py`` / ``modeling_xattn_clip.py`` implementation as the input.
No topology wrapper is introduced.

Typical:
    python train_refit_late_reader.py --model PATH_TO_FULL_XATTN_HF

Smoke:
    python train_refit_late_reader.py --model PATH_TO_FULL_XATTN_HF --smoke

Blocks are zero-based.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoProcessor


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    ROOT / "training_config.local.json"
    if (ROOT / "training_config.local.json").is_file()
    else ROOT / "training_config.json"
)

EXPECTED_ARCH = "sigmoid_all"
EXPECTED_RN_INSERT = 13
EXPECTED_NATIVE_LATE = (20, 21)

# Hard firewall: the B20/B21 CONTENT/correction branch is NOT part of this refit.
# It must remain byte-for-byte identical to the input HF model.
CORRECTION_STATE_PREFIXES = (
    "read_implant.content_pool.",
)
CORRECTION_STATE_EXACT = {
    "read_implant.content_tap_logits",
}
EXPECTED_SOURCE = (6, 7, 10)
EXPECTED_ORTHO = (8, 12, 13)
CAPTURE_BLOCKS = tuple(
    sorted(set(EXPECTED_SOURCE) | set(EXPECTED_ORTHO) | set(EXPECTED_NATIVE_LATE))
)


# =================================================================================================
# Utilities
# =================================================================================================

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def safe_median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def resolve_repo_path(repo_root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    return path


def source_mix(final_cfg: Mapping[str, Any], phase: str) -> dict[str, float]:
    if phase == "1b5":
        return {
            "coco": float(final_cfg["mix_1b5_coco"]),
            "textcaps": float(final_cfg["mix_1b5_textcaps"]),
            "imagenet": float(final_cfg["mix_1b5_imagenet"]),
            "imagenet_math": (
                float(final_cfg["mix_1b5_imagenet_math"])
                if bool(final_cfg["math_curriculum_enabled"])
                else 0.0
            ),
        }
    if phase == "1c":
        return {
            "coco": float(final_cfg["mix_1c_coco"]),
            "textcaps": float(final_cfg["mix_1c_textcaps"]),
            "imagenet": float(final_cfg["mix_1c_imagenet"]),
            "imagenet_math": (
                float(final_cfg["mix_1c_imagenet_math"])
                if bool(final_cfg["math_curriculum_enabled"])
                else 0.0
            ),
        }
    raise ValueError(phase)


def phase_weights(final_cfg: Mapping[str, Any], phase: str) -> tuple[float, float]:
    if phase == "1b5":
        return (
            float(final_cfg["source_weight_1b5"]),
            float(final_cfg["read_null_weight_1b5"]),
        )
    if phase == "1c":
        return (
            float(final_cfg["source_weight_1c"]),
            float(final_cfg["read_null_weight_1c"]),
        )
    raise ValueError(phase)


def clevr_probability(clevr_cfg: Mapping[str, Any], phase: str) -> float:
    return float(clevr_cfg[f"additive_probability_{phase}"])


def late_tap_blocks(implant: Any) -> tuple[int, ...]:
    if hasattr(implant, "_block_list"):
        return tuple(int(x) for x in implant._block_list(implant.tap_blocks))
    return tuple(int(x) for x in implant.tap_blocks.detach().cpu().tolist())


def late_tap_weights(implant: Any) -> torch.Tensor:
    if hasattr(implant, "_mix_weights"):
        return implant._mix_weights(implant.read_tap_logits).detach().float().cpu()
    return implant.read_tap_logits.detach().float().softmax(dim=0).cpu()


# =================================================================================================
# Authoritative project data construction
# =================================================================================================

def import_training_core(repo_root: Path):
    training_dir = repo_root / "main_hard_text_gate"
    if not training_dir.is_dir():
        raise FileNotFoundError(
            f"Missing training module directory: {training_dir}"
        )

    # train_gmp_anytext_stage1.py is both a repository module and a directly
    # executable historical trainer.  It intentionally still has sibling-style
    # imports such as `from coco_trusted_reading_manifest import ...` and
    # `import router_auxiliary`.  Importing it as
    # `main_hard_text_gate.train_gmp_anytext_stage1` therefore requires BOTH the
    # repository root and main_hard_text_gate itself on sys.path.
    #
    # Keep the training directory first, matching Python's normal sys.path when
    # that trainer is executed directly as a script.
    for import_root in (repo_root, training_dir):
        value = str(import_root)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)

    required_siblings = (
        training_dir / "coco_trusted_reading_manifest.py",
        training_dir / "router_auxiliary.py",
    )
    missing = [str(path) for path in required_siblings if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The authoritative final trainer is missing sibling modules required "
            "by train_gmp_anytext_stage1.py:\n  - " + "\n  - ".join(missing)
        )

    try:
        from main_hard_text_gate import train_gmp_anytext_stage1 as core
    except Exception as exc:
        raise RuntimeError(
            "Could not import main_hard_text_gate/train_gmp_anytext_stage1.py "
            f"from repo root {repo_root}\n"
            f"training module dir={training_dir}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return core


def build_sources(
    *,
    core: Any,
    cfg: Mapping[str, Any],
    repo_root: Path,
    split_kind: str,
    patch_count: int,
    effective_handwriting_probability: float,
) -> dict[str, Any]:
    paths = cfg["paths"]
    global_cfg = cfg["global"]
    final_cfg = cfg["shared_args"]["final_anytext"]
    clevr_cfg = cfg.get("datasets", {}).get("clevr_property_binding", {})

    image_size = int(global_cfg["image_size"])
    fonts = [str(x) for x in paths.get("custom_fonts", [])]
    flip = float(final_cfg["flip_probability"]) if split_kind == "train" else 0.0

    if split_kind == "train":
        imagenet_split = "train"
        textcaps_split = "train"
        clevr_split = "train"
        coco_normal = resolve_repo_path(repo_root, paths["coco_train_json"])
        coco_gpt = resolve_repo_path(repo_root, paths["coco_train_gpt_json"])
        coco_reading = resolve_repo_path(repo_root, paths.get("coco_train_reading_json"))
        clevr_root = resolve_repo_path(repo_root, paths["clevr_train_root"])
    elif split_kind == "val":
        imagenet_split = "val"
        textcaps_split = "validation"
        clevr_split = "val"
        coco_normal = resolve_repo_path(repo_root, paths["coco_val_json"])
        coco_gpt = resolve_repo_path(repo_root, paths["coco_val_gpt_json"])
        coco_reading = resolve_repo_path(repo_root, paths.get("coco_val_reading_json"))
        clevr_root = resolve_repo_path(repo_root, paths["clevr_val_root"])
    else:
        raise ValueError(split_kind)

    imagenet_text_root = resolve_repo_path(repo_root, paths["imagenet_text_root"])
    imagenet_handwriting_root = resolve_repo_path(repo_root, paths["imagenet_handwriting_root"])
    textcaps_root = resolve_repo_path(repo_root, paths["textcaps_root"])
    coco_root = resolve_repo_path(repo_root, paths["coco_root"])
    wnid_json = resolve_repo_path(repo_root, paths["imagenet_wnid_json"])
    clevr_metadata = resolve_repo_path(repo_root, paths["clevr_metadata_jsonl"])

    sources: dict[str, Any] = {
        "imagenet": core.ImageNetPacketSource(
            imagenet_text_root,
            imagenet_handwriting_root,
            imagenet_split,
            image_size,
            patch_count,
            effective_handwriting_probability,
            flip,
            fonts,
        ),
        "imagenet_math": (
            core.ContinuousMathImageNetPacketSource(
                imagenet_text_root,
                wnid_json,
                imagenet_split,
                image_size,
                patch_count,
                fonts,
                final_cfg["math_curriculum_words"],
                final_cfg["math_curriculum_families"],
                final_cfg["math_curriculum_crop_probability"],
            )
            if bool(final_cfg["math_curriculum_enabled"])
            else None
        ),
        "textcaps": core.TextCapsPacketSource(
            textcaps_root,
            textcaps_split,
            image_size,
            patch_count,
            final_cfg["textcaps_mask_weight"],
            flip,
        ),
        "coco": core.CocoSprightPacketSource(
            coco_root,
            coco_normal,
            coco_gpt,
            image_size,
            patch_count,
            final_cfg["coco_changed_probability"],
            flip,
            coco_reading,
            final_cfg["coco_reading_probability"],
            final_cfg["coco_reading_mask_weight"],
        ),
    }

    if bool(clevr_cfg.get("include", False)):
        sources["clevr"] = core.ClevrPropertyPacketSource(
            clevr_root,
            clevr_metadata,
            clevr_split,
            image_size,
            patch_count,
            fonts,
            clevr_cfg["colored_text_probability"],
            clevr_cfg["ood_color_probability"],
            clevr_cfg["standalone_count_probability"],
            clevr_cfg["canny_low"],
            clevr_cfg["canny_high"],
            clevr_cfg["canny_dilate_px"],
            clevr_cfg["placement_margin_px"],
            clevr_cfg["max_obstacle_fraction"],
            clevr_cfg["placement_stride_px"],
            clevr_cfg["fill_contours"],
            clevr_cfg["min_contour_area_fraction"],
            clevr_cfg["min_font_size"],
            clevr_cfg["max_font_size"],
        )
    else:
        sources["clevr"] = None

    return sources


# =================================================================================================
# Frozen batch representation
# =================================================================================================

@dataclass
class PreparedBatch:
    states: dict[int, torch.Tensor]
    register_mask: torch.Tensor
    source_logits: torch.Tensor
    glyph_probabilities: torch.Tensor

    candidate_count: int
    selected_indices: torch.Tensor
    modes: torch.Tensor
    read_tokens: torch.Tensor
    selected_null_mask: torch.Tensor
    query: torch.Tensor
    read_text_norm: torch.Tensor

    null_query: torch.Tensor
    null_text_norm: torch.Tensor

    positive: torch.Tensor
    present_targets: torch.Tensor
    readable_targets: torch.Tensor
    source_triplets: list[tuple[int, int, list[int]]]

    def to_cpu(self) -> "PreparedBatch":
        return PreparedBatch(
            states={
                int(k): v.detach().to("cpu", dtype=torch.float16)
                for k, v in self.states.items()
            },
            register_mask=self.register_mask.detach().cpu(),
            source_logits=self.source_logits.detach().float().cpu(),
            glyph_probabilities=self.glyph_probabilities.detach().to("cpu", dtype=torch.float16),
            candidate_count=int(self.candidate_count),
            selected_indices=self.selected_indices.detach().cpu(),
            modes=self.modes.detach().cpu(),
            read_tokens=self.read_tokens.detach().cpu(),
            selected_null_mask=self.selected_null_mask.detach().cpu(),
            query=self.query.detach().to("cpu", dtype=torch.float16),
            read_text_norm=self.read_text_norm.detach().to("cpu", dtype=torch.float16),
            null_query=self.null_query.detach().to("cpu", dtype=torch.float16),
            null_text_norm=self.null_text_norm.detach().to("cpu", dtype=torch.float16),
            positive=self.positive.detach().cpu(),
            present_targets=self.present_targets.detach().cpu(),
            readable_targets=self.readable_targets.detach().cpu(),
            source_triplets=[
                (int(c), int(p), [int(n) for n in negs])
                for c, p, negs in self.source_triplets
            ],
        )

    def to_device(self, device: torch.device) -> "PreparedBatch":
        return PreparedBatch(
            states={int(k): v.to(device, non_blocking=True) for k, v in self.states.items()},
            register_mask=self.register_mask.to(device, non_blocking=True),
            source_logits=self.source_logits.to(device, non_blocking=True),
            glyph_probabilities=self.glyph_probabilities.to(device, non_blocking=True),
            candidate_count=int(self.candidate_count),
            selected_indices=self.selected_indices.to(device, non_blocking=True),
            modes=self.modes.to(device, non_blocking=True),
            read_tokens=self.read_tokens.to(device, non_blocking=True),
            selected_null_mask=self.selected_null_mask.to(device, non_blocking=True),
            query=self.query.to(device, non_blocking=True),
            read_text_norm=self.read_text_norm.to(device, non_blocking=True),
            null_query=self.null_query.to(device, non_blocking=True),
            null_text_norm=self.null_text_norm.to(device, non_blocking=True),
            positive=self.positive.to(device, non_blocking=True),
            present_targets=self.present_targets.to(device, non_blocking=True),
            readable_targets=self.readable_targets.to(device, non_blocking=True),
            source_triplets=self.source_triplets,
        )


@torch.no_grad()
def prepare_batch(
    *,
    model: Any,
    processor: Any,
    packed_batch: Any,
    device: torch.device,
    amp: bool,
) -> PreparedBatch:
    images = packed_batch.images.to(device, non_blocking=device.type == "cuda")

    with amp_context(device, amp):
        image_info = model._vision_with_intermediates(
            images,
            return_final_tokens=False,
        )

    missing = [b for b in CAPTURE_BLOCKS if b not in image_info["states"]]
    if missing:
        raise KeyError(
            f"Missing visual blocks {missing}; captured={sorted(image_info['states'])}"
        )

    source_logits, glyph_logits, _ = model.read_implant.source_outputs(
        image_info["states"],
        return_details=False,
    )
    glyph_probabilities = glyph_logits.sigmoid()

    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(
        list(packed_batch.captions),
        padding="max_length",
        max_length=int(model.config.text_config.max_position_embeddings),
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)

    # Respect explicit <text>/<null>/<notext> controls already in training captions.
    modes, _content_tokens, read_tokens = model._prepare_tokens(input_ids, "none")
    selected_mask = ~modes.eq(2)
    selected_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()

    if not bool(selected_mask.any()):
        raise RuntimeError("Packed batch has no READ candidates")

    selected_read_tokens = read_tokens[selected_mask]
    with amp_context(device, amp):
        read_info = model._encode_text_hidden(selected_read_tokens)
        null_info = model._encode_text_hidden(model._null_read_tokens(device))

    query = read_info["eot_hidden_pre_ln"].detach()
    read_text_norm = F.normalize(
        read_info["text_embedding"].detach().float(),
        dim=-1,
        eps=1.0e-12,
    )
    null_query = null_info["eot_hidden_pre_ln"].detach()
    null_text_norm = F.normalize(
        null_info["text_embedding"].detach().float(),
        dim=-1,
        eps=1.0e-12,
    )

    selected_null_mask = selected_read_tokens.eq(
        int(model.config.null_text_token_id)
    ).any(dim=-1)

    return PreparedBatch(
        states={
            int(block): image_info["states"][int(block)].detach()
            for block in CAPTURE_BLOCKS
        },
        register_mask=image_info["register_mask"].detach(),
        source_logits=source_logits.detach().float(),
        glyph_probabilities=glyph_probabilities.detach().float(),
        candidate_count=len(packed_batch.captions),
        selected_indices=selected_indices.detach(),
        modes=modes.detach(),
        read_tokens=read_tokens.detach(),
        selected_null_mask=selected_null_mask.detach(),
        query=query,
        read_text_norm=read_text_norm,
        null_query=null_query,
        null_text_norm=null_text_norm,
        positive=packed_batch.positive.to(device=device, dtype=torch.bool).detach(),
        present_targets=packed_batch.present_targets.to(device=device, dtype=torch.float32).detach(),
        readable_targets=packed_batch.readable_targets.to(device=device, dtype=torch.float32).detach(),
        source_triplets=[
            (int(c), int(p), [int(n) for n in negs])
            for c, p, negs in packed_batch.source_triplets
        ],
    )


# =================================================================================================
# Frozen CONTENT/correction integrity guard
# =================================================================================================

def snapshot_correction_state(model: Any) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    selected = {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
        if name in CORRECTION_STATE_EXACT
        or any(name.startswith(prefix) for prefix in CORRECTION_STATE_PREFIXES)
    }
    if "read_implant.content_tap_logits" not in selected:
        raise RuntimeError("missing read_implant.content_tap_logits; cannot guard correction branch")
    if not any(name.startswith("read_implant.content_pool.") for name in selected):
        raise RuntimeError("missing read_implant.content_pool.*; cannot guard correction branch")
    return selected


def assert_correction_unchanged(
    model: Any,
    reference: Mapping[str, torch.Tensor],
    *,
    where: str,
) -> None:
    current = model.state_dict()
    missing = [name for name in reference if name not in current]
    changed = [
        name
        for name, before in reference.items()
        if name in current and not torch.equal(current[name].detach().cpu(), before)
    ]
    if missing or changed:
        raise RuntimeError(
            f"CONTENT/correction integrity failure at {where}: "
            f"missing={missing[:10]} changed={changed[:10]}"
        )


def assert_base_model_frozen(model: Any) -> None:
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(
            "Base HF model must be completely frozen during late-reader refit; "
            f"found trainable parameters: {trainable[:20]}"
        )


# =================================================================================================
# Native reader refit state
# =================================================================================================

@dataclass
class ReaderRefit:
    read_bridge: nn.Module
    read_tap_logits: nn.Parameter
    null_abstain_weight: nn.Parameter
    glyph_bias_beta: nn.Parameter
    optimizer: torch.optim.Optimizer | None = None
    best_selection_loss: float = float("inf")
    best_state: dict[str, Any] | None = None

    def read_weights(self) -> torch.Tensor:
        return self.read_tap_logits.float().softmax(dim=0)

    def all_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.read_bridge.parameters())
            + [self.read_tap_logits, self.null_abstain_weight, self.glyph_bias_beta]
        )

    def state_dict_cpu(self) -> dict[str, Any]:
        return {
            "read_bridge": {
                k: v.detach().float().cpu().clone()
                for k, v in self.read_bridge.state_dict().items()
            },
            "read_tap_logits": self.read_tap_logits.detach().float().cpu().clone(),
            "null_abstain_weight": self.null_abstain_weight.detach().float().cpu().clone(),
            "glyph_bias_beta": self.glyph_bias_beta.detach().float().cpu().clone(),
        }

    def load_state_dict_cpu(self, payload: Mapping[str, Any]) -> None:
        self.read_bridge.load_state_dict(payload["read_bridge"], strict=True)
        with torch.no_grad():
            self.read_tap_logits.copy_(
                payload["read_tap_logits"].to(
                    self.read_tap_logits.device, dtype=torch.float32
                )
            )
            self.null_abstain_weight.copy_(
                payload["null_abstain_weight"].to(
                    self.null_abstain_weight.device, dtype=torch.float32
                )
            )
            self.glyph_bias_beta.copy_(
                payload["glyph_bias_beta"].to(
                    self.glyph_bias_beta.device, dtype=torch.float32
                )
            )


def make_refit(*, model: Any, device: torch.device) -> ReaderRefit:
    bridge = copy.deepcopy(model.read_implant.read_bridge).to(
        device=device, dtype=torch.float32
    )
    bridge.train()
    return ReaderRefit(
        read_bridge=bridge,
        read_tap_logits=nn.Parameter(
            model.read_implant.read_tap_logits.detach().float().to(device).clone()
        ),
        null_abstain_weight=nn.Parameter(
            model.read_implant.null_abstain_weight.detach().float().to(device).clone()
        ),
        glyph_bias_beta=nn.Parameter(
            model.read_implant.glyph_bias_beta.detach().float().to(device).clone()
        ),
    )


def refit_read_features(
    refit: ReaderRefit,
    batch: PreparedBatch,
    text_query: torch.Tensor,
    *,
    return_details: bool = False,
):
    weights = refit.read_weights()

    glyph = batch.glyph_probabilities.float()
    bridge_glyph = torch.cat(
        (glyph, torch.ones_like(glyph[:, :1])),
        dim=1,
    )
    bridge_register = torch.cat(
        (
            batch.register_mask.bool(),
            torch.zeros_like(batch.register_mask[:, :1], dtype=torch.bool),
        ),
        dim=1,
    )
    expected_tokens = int(glyph.shape[1]) + 2

    outputs = []
    per_block: dict[int, dict[str, Any]] = {}
    null_attention = []

    for block in EXPECTED_NATIVE_LATE:
        state = batch.states[int(block)]
        if int(state.shape[1]) != expected_tokens:
            raise RuntimeError(
                f"B{block} T={state.shape[1]}, expected {expected_tokens}"
            )

        if return_details:
            out, details = refit.read_bridge(
                text_query,
                state,
                register_mask=bridge_register,
                patch_textness=bridge_glyph,
                textness_beta=refit.glyph_bias_beta,
                read_null_index=expected_tokens - 1,
                return_attention=True,
            )
            per_block[int(block)] = details
            null_attention.append(details["read_null_attention"].mean(dim=-1))
        else:
            out = refit.read_bridge(
                text_query,
                state,
                register_mask=bridge_register,
                patch_textness=bridge_glyph,
                textness_beta=refit.glyph_bias_beta,
                read_null_index=expected_tokens - 1,
                return_attention=False,
            )
        outputs.append(out)

    mixed = torch.einsum(
        "k,kbnd->bnd",
        weights,
        torch.stack(outputs).float(),
    )
    if not return_details:
        return mixed

    mixed_null = torch.einsum(
        "k,kbn->bn",
        weights,
        torch.stack(null_attention).float(),
    ).clamp(0, 1)

    return mixed, {
        "tap_weights": weights,
        "per_block": per_block,
        "read_null_attention": mixed_null,
        "read_null_index": expected_tokens - 1,
        "tap_blocks": list(EXPECTED_NATIVE_LATE),
    }


def calibrate_refit(
    *,
    raw: torch.Tensor,
    source_logits: torch.Tensor,
    null_mask: torch.Tensor,
    fixed_read_scale: torch.Tensor,
    null_abstain_weight: torch.Tensor,
) -> torch.Tensor:
    out = fixed_read_scale.float() * raw.float()
    if bool(null_mask.any()):
        unavailable = -F.logsigmoid(source_logits[:, 1].detach().float())
        bonus = null_abstain_weight.float() * unavailable[:, None]
        out = out + bonus * null_mask.float()[None, :]
    return out


@dataclass
class ReaderRefitForward:
    full_read_logits: torch.Tensor
    internal_null_logits: torch.Tensor
    full_read_null_attention: torch.Tensor
    read_details: dict[str, Any]


def forward_refit(
    *,
    refit: ReaderRefit,
    batch: PreparedBatch,
    fixed_read_scale: torch.Tensor,
    logit_scale: torch.Tensor,
) -> ReaderRefitForward:
    read_feature, read_details = refit_read_features(
        refit,
        batch,
        batch.query.float(),
        return_details=True,
    )
    read_feature_norm = F.normalize(
        read_feature.float(), dim=-1, eps=1.0e-12
    )
    raw = (
        torch.einsum(
            "bnd,nd->bn",
            read_feature_norm,
            batch.read_text_norm.float(),
        )
        * logit_scale.float()
    )

    selected_read_logits = calibrate_refit(
        raw=raw,
        source_logits=batch.source_logits,
        null_mask=batch.selected_null_mask,
        fixed_read_scale=fixed_read_scale,
        null_abstain_weight=refit.null_abstain_weight,
    )

    full_read = torch.zeros(
        (batch.source_logits.shape[0], int(batch.candidate_count)),
        device=selected_read_logits.device,
        dtype=torch.float32,
    )
    full_read[:, batch.selected_indices] = selected_read_logits

    full_null_attn = torch.zeros_like(full_read)
    full_null_attn[:, batch.selected_indices] = read_details[
        "read_null_attention"
    ].float()

    null_feature = refit_read_features(
        refit,
        batch,
        batch.null_query.float(),
        return_details=False,
    )
    null_feature_norm = F.normalize(
        null_feature[:, 0, :].float(),
        dim=-1,
        eps=1.0e-12,
    )
    raw_null = (
        torch.einsum(
            "bd,d->b",
            null_feature_norm,
            batch.null_text_norm[0].float(),
        )
        * logit_scale.float()
    )
    internal_null = calibrate_refit(
        raw=raw_null[:, None],
        source_logits=batch.source_logits,
        null_mask=torch.ones(1, dtype=torch.bool, device=raw_null.device),
        fixed_read_scale=fixed_read_scale,
        null_abstain_weight=refit.null_abstain_weight,
    )[:, 0]

    return ReaderRefitForward(
        full_read_logits=full_read,
        internal_null_logits=internal_null,
        full_read_null_attention=full_null_attn,
        read_details=read_details,
    )


# =================================================================================================
# Loss / metrics
# =================================================================================================

def triplet_margin_stats(
    logits: torch.Tensor,
    triplets: Sequence[tuple[int, int, list[int]]],
) -> tuple[float, float]:
    margins = []
    correct = []
    for candidate, positive_image, negatives in triplets:
        for negative_image in negatives:
            margin = float(
                (
                    logits[int(positive_image), int(candidate)]
                    - logits[int(negative_image), int(candidate)]
                ).detach()
            )
            margins.append(margin)
            correct.append(float(margin > 0.0))
    return safe_mean(margins), safe_mean(correct)


def split_triplets_by_null(
    batch: PreparedBatch,
) -> tuple[list[tuple[int, int, list[int]]], list[tuple[int, int, list[int]]]]:
    original_is_null = torch.zeros(
        int(batch.candidate_count),
        dtype=torch.bool,
        device=batch.selected_indices.device,
    )
    original_is_null[batch.selected_indices] = batch.selected_null_mask
    null_flags = original_is_null.detach().cpu().tolist()

    word_triplets = []
    null_triplets = []
    for triplet in batch.source_triplets:
        if bool(null_flags[int(triplet[0])]):
            null_triplets.append(triplet)
        else:
            word_triplets.append(triplet)
    return word_triplets, null_triplets


def refit_loss_and_metrics(
    *,
    refit: ReaderRefit,
    batch: PreparedBatch,
    core: Any,
    phase: str,
    final_cfg: Mapping[str, Any],
    null_text_token_id: int,
    fixed_read_scale: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = forward_refit(
        refit=refit,
        batch=batch,
        fixed_read_scale=fixed_read_scale,
        logit_scale=logit_scale,
    )

    source_loss = core.source_ranking_loss(
        out.full_read_logits,
        batch.source_triplets,
        float(final_cfg["source_margin"]),
    )

    read_null_details = {
        "read_null_attention": out.full_read_null_attention,
        "mode_ids": batch.modes,
        "read_tokens": batch.read_tokens,
        "logits_per_image": out.full_read_logits,
        "read_details": out.read_details,
    }
    read_null_loss, read_null_metrics = core.read_null_attention_objective(
        read_null_details,
        batch.positive,
        int(null_text_token_id),
    )

    source_w, rn_w = phase_weights(final_cfg, phase)
    total = source_w * source_loss + rn_w * read_null_loss

    word_triplets, null_triplets = split_triplets_by_null(batch)
    zero = out.full_read_logits.sum() * 0.0
    word_loss = (
        core.source_ranking_loss(
            out.full_read_logits,
            word_triplets,
            float(final_cfg["source_margin"]),
        )
        if word_triplets
        else zero
    )
    explicit_null_loss = (
        core.source_ranking_loss(
            out.full_read_logits,
            null_triplets,
            float(final_cfg["source_margin"]),
        )
        if null_triplets
        else zero
    )

    source_margin, source_acc = triplet_margin_stats(
        out.full_read_logits, batch.source_triplets
    )
    word_margin, word_acc = triplet_margin_stats(
        out.full_read_logits, word_triplets
    )
    null_margin, null_acc = triplet_margin_stats(
        out.full_read_logits, null_triplets
    )

    nonnull_original = torch.zeros(
        int(batch.candidate_count),
        dtype=torch.bool,
        device=out.full_read_logits.device,
    )
    nonnull_original[batch.selected_indices] = ~batch.selected_null_mask
    if bool(nonnull_original.any()):
        best_word = out.full_read_logits[:, nonnull_original].amax(dim=1)
    else:
        best_word = torch.full_like(out.internal_null_logits, -1.0e9)

    clean = batch.present_targets <= 0.0
    readable = batch.readable_targets >= 1.0

    clean_null_acc = (
        float(
            (
                out.internal_null_logits[clean] > best_word[clean]
            ).float().mean().detach()
        )
        if bool(clean.any())
        else 1.0
    )
    readable_word_acc = (
        float(
            (
                best_word[readable] > out.internal_null_logits[readable]
            ).float().mean().detach()
        )
        if bool(readable.any())
        else 1.0
    )

    metrics = {
        "total": float(total.detach()),
        "source_loss": float(source_loss.detach()),
        "read_null_loss": float(read_null_loss.detach()),
        "word_source_loss": float(word_loss.detach()),
        "explicit_null_source_loss": float(explicit_null_loss.detach()),
        "source_margin": source_margin,
        "source_pair_acc": source_acc,
        "word_margin": word_margin,
        "word_pair_acc": word_acc,
        "explicit_null_margin": null_margin,
        "explicit_null_pair_acc": null_acc,
        "clean_internal_null_acc": clean_null_acc,
        "readable_internal_word_acc": readable_word_acc,
        "null_abstain_weight": float(refit.null_abstain_weight.detach()),
        "glyph_bias_beta": float(refit.glyph_bias_beta.detach()),
    }
    for key, value in read_null_metrics.items():
        if isinstance(value, (float, int)):
            metrics[key] = float(value)

    return total, metrics


# =================================================================================================
# Optimizer
# =================================================================================================

def make_optimizer(
    refit: ReaderRefit,
    *,
    final_cfg: Mapping[str, Any],
    phase: str,
) -> torch.optim.Optimizer:
    # Match the original phase trainability as closely as this surgical refit allows:
    #   1B.5: READ bridge/taps + glyph_bias_beta; null_abstain_weight frozen.
    #   1C:   READ bridge/taps + glyph_bias_beta + null_abstain_weight.
    # We intentionally keep text embeddings, RN, SOURCE/ORTHO/CONTENT/router, and
    # read_calibration_scale frozen in BOTH phases.
    if phase == "1b5":
        read_factor = float(final_cfg["read_lr_factor_1b5"])
        new_factor = float(final_cfg["new_lr_factor_1b5"])
        refit.null_abstain_weight.requires_grad_(False)
    elif phase == "1c":
        read_factor = float(final_cfg["read_lr_factor_1c"])
        new_factor = 1.0
        refit.null_abstain_weight.requires_grad_(True)
    else:
        raise ValueError(phase)

    for parameter in refit.read_bridge.parameters():
        parameter.requires_grad_(True)
    refit.read_tap_logits.requires_grad_(True)
    refit.glyph_bias_beta.requires_grad_(True)

    read_lr = float(final_cfg["lr_read"]) * read_factor
    scalar_lr = float(final_cfg["lr_new"]) * new_factor

    scalar_params = [refit.glyph_bias_beta]
    if phase == "1c":
        scalar_params.append(refit.null_abstain_weight)

    return torch.optim.AdamW(
        [
            {
                "params": list(refit.read_bridge.parameters()) + [refit.read_tap_logits],
                "lr": read_lr,
                "group_name": "read",
            },
            {
                "params": scalar_params,
                "lr": scalar_lr,
                "group_name": "readnull_scalars",
            },
        ],
        weight_decay=float(final_cfg["weight_decay"]),
    )


def cosine_lr(
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    warmup_fraction: float,
) -> dict[str, float]:
    warmup = max(1, round(total_steps * warmup_fraction))
    if step < warmup:
        multiplier = (step + 1) / warmup
    else:
        progress = (step - warmup) / max(1, total_steps - warmup)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))

    result = {}
    for index, group in enumerate(optimizer.param_groups):
        base = group.setdefault("initial_lr", group["lr"])
        group["lr"] = base * multiplier
        result[str(group.get("group_name", index))] = float(group["lr"])
    return result


# =================================================================================================
# Validation cache
# =================================================================================================

@dataclass
class CachedValidation:
    phase: str
    batch: PreparedBatch


@torch.no_grad()
def build_validation_cache(
    *,
    core: Any,
    model: Any,
    processor: Any,
    sources_val: Mapping[str, Any],
    cfg: Mapping[str, Any],
    batches_per_phase: int,
    logical_batch_images: int,
    device: torch.device,
    amp: bool,
    seed: int,
) -> list[CachedValidation]:
    final_cfg = cfg["shared_args"]["final_anytext"]
    clevr_cfg = cfg.get("datasets", {}).get("clevr_property_binding", {})
    clevr_enabled = bool(clevr_cfg.get("include", False))
    clevr_source = sources_val.get("clevr")
    clevr_packets = int(clevr_cfg.get("packets_per_batch", 1))

    cached: list[CachedValidation] = []

    for phase_index, phase in enumerate(("1b5", "1c")):
        mixer = core.SourceMixer(
            dict(sources_val),
            source_mix(final_cfg, phase),
            seed + 1000 + phase_index * 100,
        )
        p_clevr = (
            clevr_probability(clevr_cfg, phase)
            if clevr_enabled
            else 0.0
        )
        print(
            f"[val-cache] phase={phase} mix={source_mix(final_cfg, phase)} "
            f"clevr_add_p={p_clevr} batches={batches_per_phase}"
        )

        for _ in range(batches_per_phase):
            packed = core.sample_training_batch_with_additive_clevr(
                mixer,
                logical_batch_images,
                clevr_source,
                clevr_enabled,
                p_clevr,
                clevr_packets,
            )
            prepared = prepare_batch(
                model=model,
                processor=processor,
                packed_batch=packed,
                device=device,
                amp=amp,
            )
            cached.append(
                CachedValidation(
                    phase=phase,
                    batch=prepared.to_cpu(),
                )
            )

    return cached


@torch.no_grad()
def evaluate_refit(
    *,
    refit: ReaderRefit,
    cache: Sequence[CachedValidation],
    core: Any,
    final_cfg: Mapping[str, Any],
    null_text_token_id: int,
    fixed_read_scale: torch.Tensor,
    logit_scale: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    refit.read_bridge.eval()
    rows = []
    for item in cache:
        batch = item.batch.to_device(device)
        _loss, metrics = refit_loss_and_metrics(
            refit=refit,
            batch=batch,
            core=core,
            phase=item.phase,
            final_cfg=final_cfg,
            null_text_token_id=null_text_token_id,
            fixed_read_scale=fixed_read_scale,
            logit_scale=logit_scale,
        )
        rows.append({"phase": item.phase, **metrics})
        del batch

    refit.read_bridge.train()

    result = {
        key: safe_mean(r[key] for r in rows)
        for key in rows[0]
        if key != "phase" and isinstance(rows[0][key], (float, int))
    }

    for phase in ("1b5", "1c"):
        group = [r for r in rows if r["phase"] == phase]
        for key in (
            "total",
            "source_loss",
            "read_null_loss",
            "source_pair_acc",
            "explicit_null_pair_acc",
            "clean_internal_null_acc",
            "readable_internal_word_acc",
            "read_null_accuracy",
            "read_null_attention_supported",
            "read_null_attention_unsupported",
            "read_null_attention_gap",
        ):
            if group and key in group[0]:
                result[f"{phase}_{key}"] = safe_mean(r[key] for r in group)

    result["selection_loss"] = safe_mean(r["total"] for r in rows)
    return result


def maybe_save_best(refit: ReaderRefit, metrics: Mapping[str, float]) -> bool:
    score = float(metrics["selection_loss"])
    if not math.isfinite(score):
        raise RuntimeError("non-finite validation selection loss")
    if score < refit.best_selection_loss:
        refit.best_selection_loss = score
        refit.best_state = refit.state_dict_cpu()
        return True
    return False


def run_validation(
    *,
    refit: ReaderRefit,
    cache: Sequence[CachedValidation],
    core: Any,
    final_cfg: Mapping[str, Any],
    null_text_token_id: int,
    fixed_read_scale: torch.Tensor,
    logit_scale: torch.Tensor,
    device: torch.device,
    history: list[dict[str, Any]],
    phase: str,
    step_in_phase: int,
    global_step: int,
) -> None:
    print()
    print(f"[validation] phase={phase} step={step_in_phase} global={global_step}")
    metrics = evaluate_refit(
        refit=refit,
        cache=cache,
        core=core,
        final_cfg=final_cfg,
        null_text_token_id=null_text_token_id,
        fixed_read_scale=fixed_read_scale,
        logit_scale=logit_scale,
        device=device,
    )
    improved = maybe_save_best(refit, metrics)
    weights = refit.read_weights().detach().cpu().tolist()

    history.append(
        {
            "kind": "validation",
            "phase": phase,
            "step": step_in_phase,
            "global_step": global_step,
            "slot_B20_weight": float(weights[0]),
            "slot_B21_weight": float(weights[1]),
            "best": int(improved),
            **metrics,
        }
    )

    print(
        f"  sel={metrics['selection_loss']:.5f} "
        f"src={metrics['source_loss']:.5f} "
        f"RNloss={metrics['read_null_loss']:.5f} "
        f"srcAcc={metrics['source_pair_acc']:.4f} "
        f"nullTrip={metrics['explicit_null_pair_acc']:.4f} "
        f"cleanNull={metrics['clean_internal_null_acc']:.4f} "
        f"readable={metrics['readable_internal_word_acc']:.4f} "
        f"RNacc={metrics.get('read_null_accuracy', float('nan')):.4f} "
        f"{'*BEST*' if improved else ''}"
    )
    print(
        f"    slotW={weights[0]:.4f}/{weights[1]:.4f} "
        f"null_w={float(refit.null_abstain_weight.detach()):.4f} "
        f"glyph_b={float(refit.glyph_bias_beta.detach()):.4f}"
    )


# =================================================================================================
# Training
# =================================================================================================

def train_phase(
    *,
    phase: str,
    steps: int,
    refit: ReaderRefit,
    core: Any,
    model: Any,
    processor: Any,
    sources_train: Mapping[str, Any],
    cfg: Mapping[str, Any],
    validation_cache: Sequence[CachedValidation],
    logical_batch_images: int,
    device: torch.device,
    amp: bool,
    seed: int,
    eval_every: int,
    log_every: int,
    history: list[dict[str, Any]],
    global_step_start: int,
    null_text_token_id: int,
    fixed_read_scale: torch.Tensor,
    logit_scale: torch.Tensor,
) -> int:
    if steps <= 0:
        return global_step_start

    final_cfg = cfg["shared_args"]["final_anytext"]
    clevr_cfg = cfg.get("datasets", {}).get("clevr_property_binding", {})
    clevr_enabled = bool(clevr_cfg.get("include", False))
    clevr_source = sources_train.get("clevr")
    clevr_packets = int(clevr_cfg.get("packets_per_batch", 1))
    clevr_prob = clevr_probability(clevr_cfg, phase) if clevr_enabled else 0.0

    mix = source_mix(final_cfg, phase)
    mixer = core.SourceMixer(
        dict(sources_train),
        mix,
        seed + (150 if phase == "1b5" else 250),
    )

    source_w, rn_w = phase_weights(final_cfg, phase)
    refit.optimizer = make_optimizer(refit, final_cfg=final_cfg, phase=phase)

    print()
    print("=" * 118)
    print(
        f"[phase {phase}] steps={steps} batch={logical_batch_images} "
        f"mix={mix} clevr_add_p={clevr_prob} "
        f"source_w={source_w} read_null_w={rn_w}"
    )
    print(
        "  init_lrs="
        + repr({str(g.get("group_name")): float(g["lr"]) for g in refit.optimizer.param_groups})
    )
    print("=" * 118)

    global_step = global_step_start
    for step in range(steps):
        packed = core.sample_training_batch_with_additive_clevr(
            mixer,
            logical_batch_images,
            clevr_source,
            clevr_enabled,
            clevr_prob,
            clevr_packets,
        )

        prepared = prepare_batch(
            model=model,
            processor=processor,
            packed_batch=packed,
            device=device,
            amp=amp,
        )

        assert refit.optimizer is not None
        lr_now = cosine_lr(
            refit.optimizer,
            step=step,
            total_steps=steps,
            warmup_fraction=float(final_cfg["warmup_fraction"]),
        )
        refit.optimizer.zero_grad(set_to_none=True)

        loss, metrics = refit_loss_and_metrics(
            refit=refit,
            batch=prepared,
            core=core,
            phase=phase,
            final_cfg=final_cfg,
            null_text_token_id=null_text_token_id,
            fixed_read_scale=fixed_read_scale,
            logit_scale=logit_scale,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss phase={phase} step={step}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            refit.all_parameters(),
            float(final_cfg["grad_clip"]),
        )
        refit.optimizer.step()

        with torch.no_grad():
            refit.glyph_bias_beta.clamp_(0.0, float(final_cfg["glyph_beta_max"]))
            refit.null_abstain_weight.clamp_(
                0.0, float(final_cfg["null_abstain_max"])
            )

        global_step += 1

        if step % log_every == 0 or step == steps - 1:
            weights = refit.read_weights().detach().cpu().tolist()
            history.append(
                {
                    "kind": "train",
                    "phase": phase,
                    "step": step + 1,
                    "global_step": global_step,
                    "tap_slot_B20_weight": float(weights[0]),
                    "tap_slot_B21_weight": float(weights[1]),
                    "lr_read": lr_now.get("read", float("nan")),
                    "lr_scalars": lr_now.get("readnull_scalars", float("nan")),
                    **metrics,
                }
            )
            print(
                f"[{phase}] {step+1:5d}/{steps} | "
                f"loss={metrics['total']:.3f} RN={metrics['read_null_loss']:.3f}"
            )

        if (step + 1) % eval_every == 0 or step == steps - 1:
            run_validation(
                refit=refit,
                cache=validation_cache,
                core=core,
                final_cfg=final_cfg,
                null_text_token_id=null_text_token_id,
                fixed_read_scale=fixed_read_scale,
                logit_scale=logit_scale,
                device=device,
                history=history,
                phase=phase,
                step_in_phase=step + 1,
                global_step=global_step,
            )

        del prepared, packed
        if device.type == "cuda" and (step + 1) % 100 == 0:
            torch.cuda.empty_cache()

    return global_step


# =================================================================================================
# Save complete ordinary full-xattn HF model
# =================================================================================================

def save_refit_model(
    *,
    refit: ReaderRefit,
    base_model: Any,
    processor: Any,
    input_model_dir: Path,
    output_root: Path,
    metadata: Mapping[str, Any],
    correction_reference: Mapping[str, torch.Tensor],
) -> Path:
    if refit.best_state is None:
        raise RuntimeError("no best reader state")
    refit.load_state_dict_cpu(refit.best_state)

    output_dir = output_root / "refit_model"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(input_model_dir, output_dir)

    assert_base_model_frozen(base_model)
    assert_correction_unchanged(base_model, correction_reference, where="before save")

    implant = base_model.read_implant
    original_bridge = {
        k: v.detach().clone() for k, v in implant.read_bridge.state_dict().items()
    }
    original_tap = implant.read_tap_logits.detach().clone()
    original_null = implant.null_abstain_weight.detach().clone()
    original_glyph = implant.glyph_bias_beta.detach().clone()

    try:
        implant.read_bridge.load_state_dict(refit.read_bridge.state_dict(), strict=True)
        with torch.no_grad():
            implant.read_tap_logits.copy_(
                refit.read_tap_logits.detach().to(
                    implant.read_tap_logits.device, dtype=torch.float32
                )
            )
            implant.null_abstain_weight.copy_(
                refit.null_abstain_weight.detach().to(
                    implant.null_abstain_weight.device, dtype=torch.float32
                )
            )
            implant.glyph_bias_beta.copy_(
                refit.glyph_bias_beta.detach().to(
                    implant.glyph_bias_beta.device, dtype=torch.float32
                )
            )

        # Topology is immutable here.  Refuse to save anything except the native reader.
        if late_tap_blocks(implant) != EXPECTED_NATIVE_LATE:
            raise RuntimeError(
                f"reader topology changed unexpectedly: {late_tap_blocks(implant)}"
            )
        base_model.config.read_tap_blocks = list(EXPECTED_NATIVE_LATE)

        base_model.save_pretrained(output_dir, safe_serialization=True)
        processor.save_pretrained(output_dir)

        # Explicitly verify that the ordinary full-xattn remote-code mapping survived.
        config_path = output_dir / "config.json"
        saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if tuple(int(x) for x in saved_cfg.get("read_tap_blocks", [])) != EXPECTED_NATIVE_LATE:
            raise RuntimeError("saved config does not retain native late taps")
        auto_map = saved_cfg.get("auto_map", {})
        if "modeling_xattn_clip" not in str(auto_map.get("AutoModel", "")):
            raise RuntimeError(
                "saved refit repo no longer points at the ordinary full-xattn model code"
            )

        torch.save(
            {
                "format": "native-late-reader-refit-v1",
                "read_tap_blocks": list(EXPECTED_NATIVE_LATE),
                "read_bridge": {
                    k: v.detach().float().cpu()
                    for k, v in refit.read_bridge.state_dict().items()
                },
                "read_tap_logits": refit.read_tap_logits.detach().float().cpu(),
                "read_tap_weights": refit.read_weights().detach().float().cpu(),
                "null_abstain_weight": refit.null_abstain_weight.detach().float().cpu(),
                "glyph_bias_beta": refit.glyph_bias_beta.detach().float().cpu(),
                "best_selection_loss": float(refit.best_selection_loss),
            },
            output_dir / "late_reader_refit.pt",
        )

        refit_meta = dict(metadata)
        refit_meta.update(
            {
                "read_tap_blocks": list(EXPECTED_NATIVE_LATE),
                "read_tap_weights": refit.read_weights().detach().cpu().tolist(),
                "null_abstain_weight": float(refit.null_abstain_weight.detach()),
                "glyph_bias_beta": float(refit.glyph_bias_beta.detach()),
                "best_selection_loss": float(refit.best_selection_loss),
            }
        )
        write_json(output_dir / "late_reader_refit_metadata.json", refit_meta)
        assert_correction_unchanged(base_model, correction_reference, where="after save")
    finally:
        implant.read_bridge.load_state_dict(original_bridge, strict=True)
        with torch.no_grad():
            implant.read_tap_logits.copy_(original_tap)
            implant.null_abstain_weight.copy_(original_null)
            implant.glyph_bias_beta.copy_(original_glyph)
        base_model.config.read_tap_blocks = list(EXPECTED_NATIVE_LATE)

    return output_dir


# =================================================================================================
# Setup checks / summary
# =================================================================================================

def validate_setup(
    *,
    cfg: Mapping[str, Any],
    model: Any,
) -> None:
    g = cfg["global"]

    if str(g["read_attention_architecture"]) != EXPECTED_ARCH:
        raise RuntimeError(
            f"training_config architecture={g['read_attention_architecture']!r}, "
            f"expected {EXPECTED_ARCH!r}"
        )
    if not bool(g.get("read_null_enabled", False)):
        raise RuntimeError("training_config says READ_NULL is disabled")
    if int(g["read_null_insert_block"]) != EXPECTED_RN_INSERT:
        raise RuntimeError(
            f"training_config RN insert B{g['read_null_insert_block']}, "
            f"expected B{EXPECTED_RN_INSERT}"
        )
    if int(model.config.read_null_insert_block) != EXPECTED_RN_INSERT:
        raise RuntimeError(
            f"loaded model RN insert B{model.config.read_null_insert_block}, "
            f"expected B{EXPECTED_RN_INSERT}"
        )

    taps = g["tap_blocks"]
    if tuple(taps["late"]) != EXPECTED_NATIVE_LATE:
        raise RuntimeError(
            f"config late taps={taps['late']}, expected {EXPECTED_NATIVE_LATE}"
        )
    if tuple(taps["source"]) != EXPECTED_SOURCE:
        raise RuntimeError(
            f"config SOURCE taps={taps['source']}, expected {EXPECTED_SOURCE}"
        )
    if tuple(taps["orthographic"]) != EXPECTED_ORTHO:
        raise RuntimeError(
            f"config ORTHO taps={taps['orthographic']}, expected {EXPECTED_ORTHO}"
        )

    if late_tap_blocks(model.read_implant) != EXPECTED_NATIVE_LATE:
        raise RuntimeError(
            f"loaded HF late taps={late_tap_blocks(model.read_implant)}, "
            f"expected {EXPECTED_NATIVE_LATE}"
        )


    print(
        "[firewall] SCAM/RTA/MVT/eval_bench/image_sets/misc are NOT instantiated; "
        "post-training evaluation only."
    )


def print_config_summary(
    *,
    cfg: Mapping[str, Any],
    steps_1b5: int,
    steps_1c: int,
    batch_images: int,
    effective_handwriting_probability: float,
    model: Any,
) -> None:
    g = cfg["global"]
    f = cfg["shared_args"]["final_anytext"]
    clevr = cfg.get("datasets", {}).get("clevr_property_binding", {})

    print()
    print("=" * 118)
    print("LATE READ / READ_NULL REFIT — RESOLVED AUTHORITATIVE CONFIG")
    print("=" * 118)
    print(f"architecture:                  {g['read_attention_architecture']}")
    print(f"RN insertion:                 B{g['read_null_insert_block']} zero-based")
    print(f"native late taps:             {g['tap_blocks']['late']}")
    print(f"SOURCE taps:                  {g['tap_blocks']['source']}")
    print(f"ORTHO taps:                   {g['tap_blocks']['orthographic']}")
    print(f"handwriting_enabled:          {g.get('handwriting_enabled', True)}")
    print(f"configured handwriting p:     {g.get('handwriting_probability', 0.0)}")
    print(f"effective EXTRA handwriting p:{effective_handwriting_probability: .4f}")
    print("  (extra augmentation switch only; not 'no handwriting exists in training')")
    print(f"1B.5 mix:                     {source_mix(f, '1b5')}")
    print(f"1C mix:                       {source_mix(f, '1c')}")
    print(
        "loss weights source/RN:       "
        f"1b5={phase_weights(f, '1b5')} "
        f"1c={phase_weights(f, '1c')}"
    )
    print(
        "LRs:                          "
        f"read={f['lr_read']} "
        f"read_factor_1b5={f['read_lr_factor_1b5']} "
        f"read_factor_1c={f['read_lr_factor_1c']} "
        f"new={f['lr_new']} "
        f"new_factor_1b5={f['new_lr_factor_1b5']}"
    )
    print(
        "CLEVR additive:                "
        f"enabled={bool(clevr.get('include', False))} "
        f"p1b5={clevr.get('additive_probability_1b5')} "
        f"p1c={clevr.get('additive_probability_1c')} "
        f"packets={clevr.get('packets_per_batch')}"
    )
    print(
        "trusted COCO reading:           "
        f"p={f.get('coco_reading_probability')} "
        f"mask_w={f.get('coco_reading_mask_weight')}"
    )
    print(
        "ImageNet math:                  "
        f"enabled={f.get('math_curriculum_enabled')} "
        f"words={f.get('math_curriculum_words')} "
        f"families={f.get('math_curriculum_families')}"
    )
    print(
        f"steps/batch:                    1b5={steps_1b5}, "
        f"1c={steps_1c}, batch={batch_images}"
    )
    print(
        "frozen read_calibration_scale: "
        f"{float(model.read_implant.read_calibration_scale.detach()):.6f}"
    )
    print(
        "initial null/glyph scalars:     "
        f"null_w={float(model.read_implant.null_abstain_weight.detach()):.6f} "
        f"glyph_b={float(model.read_implant.glyph_bias_beta.detach()):.6f}"
    )
    print("=" * 118)


# =================================================================================================
# Main
# =================================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True,
        help="local full_xattn_model HF export directory",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="prepared training_config.local.json (falls back to training_config.json)",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--batch-images", type=int, default=None)
    parser.add_argument("--steps-1b5", type=int, default=None)
    parser.add_argument("--steps-1c", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-batches-per-phase", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="20 + 40 steps; two fixed validation batches per phase.",
    )
    parser.add_argument(
        "--force-handwriting",
        action="store_true",
        help=(
            "Override global.handwriting_enabled=false for the EXTRA handwriting "
            "augmentation path. OFF by default."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
    )
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "load the HF model/processor, validate topology/config, import the authoritative "
            "training core, and instantiate all train/val packet sources; do not train or write output"
        ),
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing training config: {config_path}\n"
            "Run from repository root or pass --config."
        )
    repo_root = config_path.parent
    cfg = load_json(config_path)
    core = import_training_core(repo_root)

    final_cfg = cfg["shared_args"]["final_anytext"]
    final_stage = cfg["stages"]["final_base"]["args"]

    steps_1b5 = (
        int(args.steps_1b5)
        if args.steps_1b5 is not None
        else int(final_stage["phase_1b5_epochs"])
        * int(final_stage["steps_per_epoch"])
    )
    steps_1c = (
        int(args.steps_1c)
        if args.steps_1c is not None
        else int(final_stage["phase_1c_epochs"])
        * int(final_stage["steps_per_epoch"])
    )
    batch_images = (
        int(args.batch_images)
        if args.batch_images is not None
        else int(final_stage["logical_batch_images"])
    )
    val_batches = int(args.val_batches_per_phase)

    if args.smoke:
        steps_1b5 = 20
        steps_1c = 40
        val_batches = 2
        args.eval_every = 20
        args.log_every = 5

    # Cheap-ish structural preflight before creating an output directory.  It loads
    # the real HF repo and instantiates the exact train/validation packet sources so
    # remote-code drift, topology mismatches, missing local data, and constructor
    # signature drift fail here instead of 20 minutes into a refit.
    if args.preflight_only:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        set_seed(args.seed)
        print(f"[preflight] model={args.model}")
        model = AutoModel.from_pretrained(
            args.model, trust_remote_code=True
        ).eval().to(device)
        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=True
        )
        validate_setup(cfg=cfg, model=model)
        global_cfg = cfg["global"]
        handwriting_enabled = bool(global_cfg.get("handwriting_enabled", True))
        configured_handwriting = float(
            global_cfg.get(
                "handwriting_probability",
                final_cfg.get("handwriting_probability", 0.0),
            )
        )
        effective_handwriting = (
            configured_handwriting
            if (handwriting_enabled or args.force_handwriting)
            else 0.0
        )
        patch_count = (
            int(model.config.vision_config.image_size)
            // int(model.config.vision_config.patch_size)
        ) ** 2
        print("[preflight] instantiate train packet sources ...")
        build_sources(
            core=core, cfg=cfg, repo_root=repo_root, split_kind="train",
            patch_count=patch_count,
            effective_handwriting_probability=effective_handwriting,
        )
        print("[preflight] instantiate validation packet sources ...")
        build_sources(
            core=core, cfg=cfg, repo_root=repo_root, split_kind="val",
            patch_count=patch_count,
            effective_handwriting_probability=effective_handwriting,
        )
        del processor, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            "[preflight] PASS: HF remote code, model topology, prepared config, "
            "training-core import, and exact train/val packet constructors are compatible"
        )
        return

    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite_output:
            raise FileExistsError(
                f"Output root already exists: {output_root}\n"
                "Use a new --output-root or pass --overwrite-output."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    (output_root / "TEMPORARY_READ_REFIT_EXPERIMENT_ROOT.txt").write_text(
        "TEMPORARY LATE READ / READ_NULL REFIT EXPERIMENT ROOT.\n"
        "Input model is untouched. Benchmark the saved refit before keeping it.\n",
        encoding="utf-8",
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed)

    print(f"[model] loading {args.model}")
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
    ).eval().to(device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert_base_model_frozen(model)
    correction_reference = snapshot_correction_state(model)
    print(
        f"[freeze] CONTENT/correction locked exactly: "
        f"{len(correction_reference)} tensors (content_pool + content_tap_logits)"
    )

    validate_setup(cfg=cfg, model=model)

    global_cfg = cfg["global"]
    handwriting_enabled = bool(
        global_cfg.get("handwriting_enabled", True)
    )
    configured_handwriting = float(
        global_cfg.get(
            "handwriting_probability",
            final_cfg.get("handwriting_probability", 0.0),
        )
    )
    effective_handwriting = (
        configured_handwriting
        if (handwriting_enabled or args.force_handwriting)
        else 0.0
    )

    print_config_summary(
        cfg=cfg,
        steps_1b5=steps_1b5,
        steps_1c=steps_1c,
        batch_images=batch_images,
        effective_handwriting_probability=effective_handwriting,
        model=model,
    )

    patch_count = (
        int(model.config.vision_config.image_size)
        // int(model.config.vision_config.patch_size)
    ) ** 2

    print("[data] constructing exact project packet sources ...")
    sources_train = build_sources(
        core=core,
        cfg=cfg,
        repo_root=repo_root,
        split_kind="train",
        patch_count=patch_count,
        effective_handwriting_probability=effective_handwriting,
    )
    sources_val = build_sources(
        core=core,
        cfg=cfg,
        repo_root=repo_root,
        split_kind="val",
        patch_count=patch_count,
        effective_handwriting_probability=effective_handwriting,
    )

    refit = make_refit(model=model, device=device)

    print(
        "[init] native trained late weights: "
        + " ".join(
            f"slot_B{b}={w:.6f}"
            for b, w in zip(
                EXPECTED_NATIVE_LATE,
                late_tap_weights(model.read_implant).tolist(),
            )
        )
    )

    fixed_read_scale = (
        model.read_implant.read_calibration_scale.detach().float().to(device)
    )
    logit_scale = model.logit_scale.detach().float().exp().to(device)
    null_text_token_id = int(model.config.null_text_token_id)

    history: list[dict[str, Any]] = []

    validation_cache = build_validation_cache(
        core=core,
        model=model,
        processor=processor,
        sources_val=sources_val,
        cfg=cfg,
        batches_per_phase=val_batches,
        logical_batch_images=batch_images,
        device=device,
        amp=args.amp,
        seed=args.seed,
    )

    # Initial current-reader state is eligible to remain best.
    run_validation(
        refit=refit,
        cache=validation_cache,
        core=core,
        final_cfg=final_cfg,
        null_text_token_id=null_text_token_id,
        fixed_read_scale=fixed_read_scale,
        logit_scale=logit_scale,
        device=device,
        history=history,
        phase="initial",
        step_in_phase=0,
        global_step=0,
    )

    global_step = 0
    global_step = train_phase(
        phase="1b5",
        steps=steps_1b5,
        refit=refit,
        core=core,
        model=model,
        processor=processor,
        sources_train=sources_train,
        cfg=cfg,
        validation_cache=validation_cache,
        logical_batch_images=batch_images,
        device=device,
        amp=args.amp,
        seed=args.seed,
        eval_every=args.eval_every,
        log_every=args.log_every,
        history=history,
        global_step_start=global_step,
        null_text_token_id=null_text_token_id,
        fixed_read_scale=fixed_read_scale,
        logit_scale=logit_scale,
    )
    global_step = train_phase(
        phase="1c",
        steps=steps_1c,
        refit=refit,
        core=core,
        model=model,
        processor=processor,
        sources_train=sources_train,
        cfg=cfg,
        validation_cache=validation_cache,
        logical_batch_images=batch_images,
        device=device,
        amp=args.amp,
        seed=args.seed,
        eval_every=args.eval_every,
        log_every=args.log_every,
        history=history,
        global_step_start=global_step,
        null_text_token_id=null_text_token_id,
        fixed_read_scale=fixed_read_scale,
        logit_scale=logit_scale,
    )
    if refit.best_state is None:
        raise RuntimeError("no best reader state")
    refit.load_state_dict_cpu(refit.best_state)
    metrics = evaluate_refit(
        refit=refit,
        cache=validation_cache,
        core=core,
        final_cfg=final_cfg,
        null_text_token_id=null_text_token_id,
        fixed_read_scale=fixed_read_scale,
        logit_scale=logit_scale,
        device=device,
    )
    weights = refit.read_weights().detach().cpu().tolist()
    final_row = {
        "slot_B20_weight": float(weights[0]),
        "slot_B21_weight": float(weights[1]),
        "null_abstain_weight": float(refit.null_abstain_weight.detach()),
        "glyph_bias_beta": float(refit.glyph_bias_beta.detach()),
        "best_selection_loss": float(refit.best_selection_loss),
        **metrics,
    }

    write_csv(output_root / "training_history.csv", history)
    write_json(output_root / "training_history.json", history)
    write_csv(output_root / "final_late_reader_refit_summary.csv", [final_row])
    write_json(output_root / "final_late_reader_refit_summary.json", final_row)

    metadata = {
        "format": "native-late-reader-refit-v1",
        "input_model": str(args.model),
        "authoritative_training_config": str(config_path),
        "output_root": str(output_root),
        "read_tap_blocks": list(EXPECTED_NATIVE_LATE),
        "topology_policy": "native B20/B21 READ tap topology is immutable; CONTENT/correction is never trained",
        "trainable": [
            "read_implant.read_bridge",
            "read_implant.read_tap_logits",
            "read_implant.glyph_bias_beta (1B.5 + 1C)",
            "read_implant.null_abstain_weight (1C only; frozen in 1B.5)",
        ],
        "frozen": [
            "entire CLIP vision backbone",
            "RN/read_null_token",
            "text encoder",
            "hard_text_embedding",
            "null_text_embedding",
            "SOURCE B6/B7/B10",
            "ORTHO bridge",
            "CONTENT/correction (content_pool + content_tap_logits; exact integrity-checked)",
            "trust router",
            "read_calibration_scale",
            "all other PIECES parameters",
        ],
        "loss": (
            "configured source_weight * original source_ranking_loss + "
            "configured read_null_weight * original read_null_attention_objective"
        ),
        "effective_extra_handwriting_probability": effective_handwriting,
        "configured_handwriting_probability": configured_handwriting,
        "handwriting_enabled_in_config": handwriting_enabled,
        "force_handwriting_override": bool(args.force_handwriting),
        "phase_1b5_mix": source_mix(final_cfg, "1b5"),
        "phase_1c_mix": source_mix(final_cfg, "1c"),
        "phase_1b5_loss_weights": phase_weights(final_cfg, "1b5"),
        "phase_1c_loss_weights": phase_weights(final_cfg, "1c"),
        "steps_1b5": steps_1b5,
        "steps_1c": steps_1c,
        "logical_batch_images": batch_images,
        "validation_batches_per_phase": val_batches,
        "benchmark_firewall": (
            "SCAM/RTA/MVT/eval_bench/image_sets/misc are not training sources."
        ),
    }
    write_json(output_root / "refit_metadata.json", metadata)

    print()
    print("=" * 118)
    print("FINAL NATIVE LATE READ / READ_NULL REFIT VALIDATION")
    print("=" * 118)
    print(
        f"sel={final_row['selection_loss']:.5f} "
        f"src={final_row['source_loss']:.5f} "
        f"RNloss={final_row['read_null_loss']:.5f} "
        f"srcAcc={final_row['source_pair_acc']:.4f} "
        f"nullTrip={final_row['explicit_null_pair_acc']:.4f} "
        f"cleanNull={final_row['clean_internal_null_acc']:.4f} "
        f"readable={final_row['readable_internal_word_acc']:.4f} "
        f"RNacc={final_row.get('read_null_accuracy', float('nan')):.4f}"
    )
    print(
        f"  slotW={final_row['slot_B20_weight']:.4f}/"
        f"{final_row['slot_B21_weight']:.4f} "
        f"null_w={final_row['null_abstain_weight']:.4f} "
        f"glyph_b={final_row['glyph_bias_beta']:.4f}"
    )

    if not args.no_save_models:
        path = save_refit_model(
            refit=refit,
            base_model=model,
            processor=processor,
            input_model_dir=args.model,
            output_root=output_root,
            metadata=metadata,
            correction_reference=correction_reference,
        )
        write_json(output_root / "saved_model_dir.json", {"path": str(path)})
        print(f"[save] complete HF refit model -> {path}")
    else:
        print("[save] --no-save-models set")

    print()
    print("[done] Native late reader refit complete; topology was not changed.")
    print("[done] No SCAM/RTA/MVT/image_sets/misc training occurred.")


if __name__ == "__main__":
    main()
