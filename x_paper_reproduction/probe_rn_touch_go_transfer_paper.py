"""
RN transplant: persistent vs B13 touch-and-go.

Focused paper evaluation derived from probe_rn_test_arena_v2.py.

Question
--------
Take exactly one learned 1024-D Read-Null (RN) token from the final B13-RN
checkpoint and transplant only that vector into otherwise unmodified OpenAI
CLIP ViT-L/14 and GmP ViT-L/14 visual towers.

Compare three receiver conditions:
    1) no_rn         : receiver as-is (reference)
    2) rn_b13_only   : append RN immediately before B13, execute B13, remove RN
                       before B14 ("touch-and-go")
    3) rn_persistent : append RN immediately before B13 and retain it through B23

No bridge, CONTENT correction, router, text-side control token, donor weights,
or receiver training is used. SCAM/RTA are evaluation only.

Outputs
-------
All outputs are written beside this script under ./rn_touch_go_paper_results/:
    benchmark_summary.csv
        Per receiver/subset/condition accuracy, Wilson CI, cosine margin, and
        CLIP-scaled logit margin.
    retention_summary.csv
        Attack-only causal effect retention for touch-and-go relative to
        persistent RN, per subset and weighted across all attacked subsets.
    trajectory_summary.csv
        Pooled attacked-image comparison of the RN-induced ordinary-token state
        for touch-and-go vs persistent RN from B13 through B23, plus the final
        normalized image embedding. Includes per-image delta cosine, norm ratio,
        relative error, and linear CKA of the induced delta matrices.
    paper_summary.txt
        Compact human-readable headline tables.
    run_config.json
        Exact donor/receiver paths, RN SHA256, deterministic settings, counts,
        software versions, and parity checks.
    rn_touch_go_benchmark.{png,pdf}
        Accuracy and mean logit-margin plots for all six benchmark subsets.
    rn_touch_go_trajectory.{png,pdf}
        Blockwise alignment and magnitude retention of the touch-and-go control
        state relative to persistent RN.

The script intentionally does NOT save per-image raw outputs.
"""

from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()

# Must be set before CUDA context creation for deterministic cuBLAS.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "260913")

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import oaicliporg as legacy_clip
from benchmark_final_clip import (
    _explicit_architecture,
    _extract_state_dict,
    torch_load_trusted,
    validate_final_state_dict,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# =============================================================================
# Fixed paper configuration
# =============================================================================

SEED = 260913
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INSERT_BLOCK = 13
TEXT_TEMPLATE = "a photo of a {}"

DONOR_MODELS: List[Tuple[str, str]] = [
    (
        "My-CLIP!wise-0p6-both__A4__sigmoid_b13RN",
        "REPLACE_WITH_CHECKPOINT.pt",
    ),
]

RECEIVERS: List[Tuple[str, str, str]] = [
    ("OAI", "OpenAI CLIP ViT-L/14", "ViT-L/14"),
    ("GmP", "GmP ViT-L/14", "REPLACE_WITH_CHECKPOINT.pt"),
]

SCAM_REPO = "BLISS-e-V/SCAM"
RTA_REPO = "zer0int/RTA-100-Triplet"
SUBSET_ORDER = ("NoSCAM", "SCAM", "SynthSCAM", "NoRTA", "RTA", "SynthRTA")
ATTACK_SUBSETS = ("SCAM", "SynthSCAM", "RTA", "SynthRTA")
CONDITION_ORDER = ("no_rn", "rn_b13_only", "rn_persistent")
CONDITION_LABELS = {
    "no_rn": "No RN",
    "rn_b13_only": "B13 only",
    "rn_persistent": "Persistent RN",
}

CAPTURE_BLOCKS = tuple(range(INSERT_BLOCK, 24))
DEFAULT_BATCH_SIZE = 16 if DEVICE == "cuda" else 2
DEFAULT_TRAJECTORY_SAMPLES_PER_ATTACK_SUBSET = 128
DEFAULT_NUM_WORKERS = 0  # Windows-safe and maximally reproducible.

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "rn_touch_go_paper_results"


# =============================================================================
# Determinism / precision
# =============================================================================


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.set_float32_matmul_precision("highest")
    except AttributeError:
        pass

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        # Prefer deterministic math SDPA over fused kernels when available.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    torch.use_deterministic_algorithms(True, warn_only=False)


configure_determinism(SEED)


# =============================================================================
# Dataset views
# =============================================================================


@dataclass(frozen=True)
class SubsetSpec:
    name: str
    dataset: Any
    indices: Tuple[int, ...]


class EvalSubsetDataset(Dataset):
    def __init__(self, spec: SubsetSpec, preprocess_fn):
        self.spec = spec
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.spec.indices)

    def __getitem__(self, local_index: int):
        row = self.spec.dataset[int(self.spec.indices[local_index])]
        image = row["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image, got {type(image)}")
        image = image.convert("RGB")
        return (
            self.preprocess_fn(image),
            str(row["object_label"]),
            str(row["attack_word"]),
            str(row.get("id", local_index)),
        )


class ImageOnlySubsetDataset(Dataset):
    def __init__(self, spec: SubsetSpec, preprocess_fn):
        self.spec = spec
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.spec.indices)

    def __getitem__(self, local_index: int):
        row = self.spec.dataset[int(self.spec.indices[local_index])]
        image = row["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image, got {type(image)}")
        return self.preprocess_fn(image.convert("RGB"))


def eval_collate(batch):
    images, object_labels, attack_words, ids = zip(*batch)
    return torch.stack(images), list(object_labels), list(attack_words), list(ids)


def deterministic_indices(indices: Sequence[int], limit: int, salt: str) -> Tuple[int, ...]:
    values = list(map(int, indices))
    if limit <= 0 or len(values) <= limit:
        return tuple(values)
    rng = random.Random(f"{SEED}:{salt}")
    chosen = sorted(rng.sample(values, int(limit)))
    return tuple(chosen)


def load_subset_specs(max_samples_per_subset: int = 0) -> Dict[str, SubsetSpec]:
    print("[dataset] loading SCAM and RTA-100-Triplet...")
    scam = load_dataset(SCAM_REPO, split="train")
    rta = load_dataset(RTA_REPO, split="train")

    # Strip the image column while discovering subset membership so indexing does
    # not eagerly decode every image.
    scam_meta = scam.remove_columns(["image"])
    rta_meta = rta.remove_columns(["image"])

    scam_idx: Dict[str, List[int]] = {k: [] for k in ("NoSCAM", "SCAM", "SynthSCAM")}
    for i, row in enumerate(scam_meta):
        sid = str(row["id"])
        subset = next((k for k in scam_idx if sid.startswith(k)), None)
        if subset is not None:
            scam_idx[subset].append(i)

    rta_idx: Dict[str, List[int]] = {k: [] for k in ("NoRTA", "RTA", "SynthRTA")}
    for i, row in enumerate(rta_meta):
        subset = str(row["type"])
        if subset in rta_idx:
            rta_idx[subset].append(i)

    specs: Dict[str, SubsetSpec] = {}
    for name in ("NoSCAM", "SCAM", "SynthSCAM"):
        idx = deterministic_indices(scam_idx[name], max_samples_per_subset, f"benchmark:{name}")
        specs[name] = SubsetSpec(name, scam, idx)
    for name in ("NoRTA", "RTA", "SynthRTA"):
        idx = deterministic_indices(rta_idx[name], max_samples_per_subset, f"benchmark:{name}")
        specs[name] = SubsetSpec(name, rta, idx)

    for name in SUBSET_ORDER:
        print(f"  {name:<10} n={len(specs[name].indices)}")
    return specs


def make_loader(dataset_obj: Dataset, batch_size: int, num_workers: int, *, collate_fn=None) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(DEVICE == "cuda"),
        "persistent_workers": bool(num_workers > 0),
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset_obj, **kwargs)


# =============================================================================
# Donor / receiver loading
# =============================================================================


@dataclass
class ReceiverBundle:
    alias: str
    display_name: str
    spec: str
    model: torch.nn.Module
    preprocess: Any


def load_donor_rn() -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Read the trained RN directly from the donor checkpoint state.

    This experiment never executes the donor model.  It needs only the learned
    RN vector and its insertion-block configuration, so constructing an x-attn
    runtime here would add an unnecessary architecture dependency and can
    accidentally route a split-QKV checkpoint through an incompatible model
    builder.  Validate that the file is a complete final x-attn checkpoint,
    then extract the two RN tensors as inert CPU state.
    """
    if len(DONOR_MODELS) != 1:
        raise RuntimeError(f"Expected exactly one RN donor, got {len(DONOR_MODELS)}")
    alias, path = DONOR_MODELS[0]
    print(f"[donor] {alias}: {path}")

    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RN donor checkpoint not found: {checkpoint}")

    loaded = torch_load_trusted(checkpoint)
    state, container = _extract_state_dict(loaded)
    explicit_architecture = _explicit_architecture(loaded)
    architecture = validate_final_state_dict(state, explicit_architecture)
    del loaded

    token_key = "visual.read_null_token"
    if token_key not in state:
        candidates = [key for key in state if key.endswith(token_key)]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Donor checkpoint has no unique {token_key}; candidates={candidates[:10]}"
            )
        token_key = candidates[0]

    insert_key = "visual.read_null_insert_block_config"
    if insert_key not in state:
        candidates = [key for key in state if key.endswith(insert_key)]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Donor checkpoint has no unique {insert_key}; candidates={candidates[:10]}"
            )
        insert_key = candidates[0]

    token = state[token_key]
    insert_tensor = state[insert_key]
    if not torch.is_tensor(token):
        raise RuntimeError(f"Donor {token_key} is not a tensor")
    if not torch.is_tensor(insert_tensor) or insert_tensor.numel() != 1:
        raise RuntimeError(
            f"Donor {insert_key} must be a scalar tensor, got {type(insert_tensor).__name__}"
        )

    insert_block = int(insert_tensor.detach().cpu().item())
    if insert_block != INSERT_BLOCK:
        raise RuntimeError(f"Donor RN inserts before B{insert_block}, expected B{INSERT_BLOCK}")

    rn = token.detach().float().cpu().contiguous().view(-1)
    if rn.numel() != 1024:
        raise RuntimeError(f"Expected 1024-D RN token, got {rn.numel()}")
    if not bool(torch.isfinite(rn).all()):
        raise FloatingPointError("Donor RN token contains non-finite values")

    sha = hashlib.sha256(rn.numpy().tobytes()).hexdigest()
    meta = {
        "alias": alias,
        "path": str(checkpoint),
        "checkpoint_container": container,
        "read_attention_architecture": architecture,
        "state_tensor_count": int(len(state)),
        "token_state_key": token_key,
        "insert_block_state_key": insert_key,
        "insert_block": insert_block,
        "width": int(rn.numel()),
        "norm": float(rn.norm().item()),
        "mean": float(rn.mean().item()),
        "std": float(rn.std(unbiased=False).item()),
        "sha256_fp32_bytes": sha,
        "donor_runtime_instantiated": False,
    }
    print(
        f"[donor] validated full x-attn state: container={container} "
        f"architecture={architecture}; donor runtime NOT instantiated"
    )
    print(
        f"[donor] RN width={meta['width']} norm={meta['norm']:.6f} "
        f"insert=pre-B{insert_block} sha256={sha[:16]}..."
    )
    del state
    return rn, meta


def load_receiver(alias: str, display_name: str, spec: str) -> ReceiverBundle:
    print(f"[receiver] {display_name}: {spec}")
    model, preprocess = legacy_clip.load(spec, device=DEVICE, jit=False)
    model = model.to(device=DEVICE, dtype=torch.float32).eval()
    if not hasattr(model, "visual") or not hasattr(model.visual, "transformer"):
        raise RuntimeError(f"Receiver {alias} is not an OpenAI-compatible ViT")
    if len(model.visual.transformer.resblocks) != 24:
        raise RuntimeError(f"Receiver {alias}: expected 24 ViT blocks")
    width = int(model.visual.class_embedding.numel())
    if width != 1024:
        raise RuntimeError(f"Receiver {alias}: expected width 1024, got {width}")
    return ReceiverBundle(alias, display_name, spec, model, preprocess)


# =============================================================================
# Native ViT forward with transplanted RN
# =============================================================================


def prepare_visual_tokens(visual: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Native-resolution OpenAI CLIP ViT token preparation; no interpolation."""
    x = visual.conv1(images.to(device=DEVICE, dtype=torch.float32))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    pos = visual.positional_embedding.to(device=x.device, dtype=x.dtype)
    if x.shape[1] != pos.shape[0]:
        raise RuntimeError(
            f"Native token count {x.shape[1]} != positional embedding {pos.shape[0]}; "
            "this focused paper script intentionally does not do multi-resolution interpolation."
        )
    x = visual.ln_pre(x + pos)
    return x.permute(1, 0, 2).contiguous()  # [T,B,C]


def finalize_cls(visual: torch.nn.Module, x_tbc: torch.Tensor) -> torch.Tensor:
    x_btc = x_tbc.permute(1, 0, 2)
    cls = visual.ln_post(x_btc[:, 0, :])
    if visual.proj is not None:
        cls = cls @ visual.proj
    return cls.float()


@dataclass
class VisualRun:
    embedding: torch.Tensor
    states: Dict[int, torch.Tensor]  # ordinary tokens only, [B,T,C]


@torch.inference_mode()
def run_receiver_visual(
    model: torch.nn.Module,
    images: torch.Tensor,
    rn_token_cpu: torch.Tensor,
    condition: str,
    capture_blocks: Sequence[int] = (),
) -> VisualRun:
    if condition not in CONDITION_ORDER:
        raise ValueError(condition)

    visual = model.visual
    x = prepare_visual_tokens(visual, images)
    capture = set(map(int, capture_blocks))
    states: Dict[int, torch.Tensor] = {}
    has_rn = False

    for block_idx, block in enumerate(visual.transformer.resblocks):
        if block_idx == INSERT_BLOCK and condition != "no_rn":
            rn = rn_token_cpu.to(device=x.device, dtype=x.dtype).reshape(1, 1, -1)
            x = torch.cat([x, rn.expand(1, x.shape[1], -1)], dim=0)
            has_rn = True

        x = block(x)

        if block_idx in capture:
            ordinary = x[:-1] if has_rn else x
            states[block_idx] = ordinary.permute(1, 0, 2).detach().float()

        if block_idx == INSERT_BLOCK and condition == "rn_b13_only":
            if not has_rn:
                raise RuntimeError("B13-only condition reached removal without RN")
            x = x[:-1]
            has_rn = False

    embedding = finalize_cls(visual, x)
    return VisualRun(embedding=embedding, states=states)


@torch.inference_mode()
def parity_check(bundle: ReceiverBundle, dataset_spec: SubsetSpec) -> float:
    ds = EvalSubsetDataset(
        SubsetSpec(dataset_spec.name, dataset_spec.dataset, dataset_spec.indices[:1]),
        bundle.preprocess,
    )
    image, _, _, _ = ds[0]
    image = image.unsqueeze(0).to(DEVICE).float()
    manual = run_receiver_visual(
        bundle.model, image, torch.zeros(1024), "no_rn"
    ).embedding
    native = bundle.model.encode_image(image).float()
    diff = float((manual - native).abs().max().item())
    print(f"[parity/{bundle.alias}] manual no-RN vs encode_image max_abs={diff:.3e}")
    if diff > 2.0e-5:
        raise RuntimeError(f"Receiver {bundle.alias} native parity failed: {diff:.3e}")
    return diff


# =============================================================================
# Text scoring / statistics
# =============================================================================


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1.0e-12)


@torch.inference_mode()
def build_pair_text_features(
    model: torch.nn.Module,
    object_labels: Sequence[str],
    attack_words: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompts = [
        *(TEXT_TEMPLATE.format(x) for x in object_labels),
        *(TEXT_TEMPLATE.format(x) for x in attack_words),
    ]
    tokens = legacy_clip.tokenize(prompts, truncate=True).to(DEVICE)
    text = normalize(model.encode_text(tokens))
    m = len(object_labels)
    return text[:m], text[m:]


def model_logit_scale(model: torch.nn.Module) -> float:
    value = getattr(model, "logit_scale", None)
    if torch.is_tensor(value):
        return float(value.detach().float().exp().cpu().item())
    return 1.0


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n)) / denom
    return center - half, center + half


def summarize_margins(margins: torch.Tensor, logit_scale: float) -> Dict[str, float]:
    m = margins.detach().float().cpu().numpy().astype(np.float64)
    n = int(m.size)
    correct = int((m > 0.0).sum())
    acc = correct / max(1, n)
    lo, hi = wilson_interval(correct, n)
    mean = float(m.mean()) if n else float("nan")
    std = float(m.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else float("nan")
    return {
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "accuracy_ci95_lo": lo,
        "accuracy_ci95_hi": hi,
        "mean_cosine_margin": mean,
        "cosine_margin_se": se,
        "mean_logit_margin": mean * logit_scale,
        "logit_margin_se": se * logit_scale,
    }


# =============================================================================
# Benchmark evaluation
# =============================================================================


@torch.inference_mode()
def evaluate_benchmark_receiver(
    bundle: ReceiverBundle,
    rn_token: torch.Tensor,
    specs: Mapping[str, SubsetSpec],
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    scale = model_logit_scale(bundle.model)
    print(f"[receiver/{bundle.alias}] logit_scale={scale:.6f}")

    for subset in SUBSET_ORDER:
        ds = EvalSubsetDataset(specs[subset], bundle.preprocess)
        loader = make_loader(ds, batch_size, num_workers, collate_fn=eval_collate)
        all_margins: Dict[str, List[torch.Tensor]] = {c: [] for c in CONDITION_ORDER}

        for images, object_labels, attack_words, _ids in tqdm(
            loader, desc=f"{bundle.alias} | {subset}", ncols=110
        ):
            images = images.to(DEVICE, non_blocking=DEVICE == "cuda").float()
            text_obj, text_atk = build_pair_text_features(bundle.model, object_labels, attack_words)
            for condition in CONDITION_ORDER:
                run = run_receiver_visual(bundle.model, images, rn_token, condition)
                z = normalize(run.embedding)
                margin = (z * text_obj).sum(dim=-1) - (z * text_atk).sum(dim=-1)
                all_margins[condition].append(margin.detach().cpu())

        for condition in CONDITION_ORDER:
            margins = torch.cat(all_margins[condition], dim=0)
            stats = summarize_margins(margins, scale)
            rows.append(
                {
                    "receiver": bundle.alias,
                    "receiver_name": bundle.display_name,
                    "subset": subset,
                    "condition": condition,
                    **stats,
                }
            )

    return pd.DataFrame(rows)


def weighted_metric(group: pd.DataFrame, col: str) -> float:
    w = group["n"].astype(float).to_numpy()
    x = group[col].astype(float).to_numpy()
    return float(np.sum(w * x) / max(1.0, np.sum(w)))


def safe_retention(no: float, touch: float, persistent: float) -> float:
    denom = persistent - no
    if abs(denom) < 1.0e-12:
        return float("nan")
    return (touch - no) / denom


def make_retention_summary(benchmark: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for receiver in [r[0] for r in RECEIVERS]:
        sub = benchmark[(benchmark.receiver == receiver) & benchmark.subset.isin(ATTACK_SUBSETS)]
        for subset in ATTACK_SUBSETS:
            g = sub[sub.subset == subset].set_index("condition")
            if not all(c in g.index for c in CONDITION_ORDER):
                continue
            no = g.loc["no_rn"]
            touch = g.loc["rn_b13_only"]
            persist = g.loc["rn_persistent"]
            rows.append(
                {
                    "receiver": receiver,
                    "subset": subset,
                    "n": int(no["n"]),
                    "no_rn_accuracy": float(no["accuracy"]),
                    "b13_only_accuracy": float(touch["accuracy"]),
                    "persistent_accuracy": float(persist["accuracy"]),
                    "persistent_gain_pp": 100.0 * (float(persist["accuracy"]) - float(no["accuracy"])),
                    "b13_only_gain_pp": 100.0 * (float(touch["accuracy"]) - float(no["accuracy"])),
                    "b13_only_minus_persistent_pp": 100.0 * (float(touch["accuracy"]) - float(persist["accuracy"])),
                    "accuracy_effect_retained": safe_retention(
                        float(no["accuracy"]), float(touch["accuracy"]), float(persist["accuracy"])
                    ),
                    "no_rn_logit_margin": float(no["mean_logit_margin"]),
                    "b13_only_logit_margin": float(touch["mean_logit_margin"]),
                    "persistent_logit_margin": float(persist["mean_logit_margin"]),
                    "logit_margin_effect_retained": safe_retention(
                        float(no["mean_logit_margin"]),
                        float(touch["mean_logit_margin"]),
                        float(persist["mean_logit_margin"]),
                    ),
                }
            )

        # Weighted aggregate across the four attacked subsets.
        agg: Dict[str, float] = {}
        for condition in CONDITION_ORDER:
            g = sub[sub.condition == condition]
            agg[f"{condition}_acc"] = weighted_metric(g, "accuracy")
            agg[f"{condition}_margin"] = weighted_metric(g, "mean_logit_margin")
        n_total = int(sub[sub.condition == "no_rn"]["n"].sum())
        rows.append(
            {
                "receiver": receiver,
                "subset": "ATTACK_ALL_WEIGHTED",
                "n": n_total,
                "no_rn_accuracy": agg["no_rn_acc"],
                "b13_only_accuracy": agg["rn_b13_only_acc"],
                "persistent_accuracy": agg["rn_persistent_acc"],
                "persistent_gain_pp": 100.0 * (agg["rn_persistent_acc"] - agg["no_rn_acc"]),
                "b13_only_gain_pp": 100.0 * (agg["rn_b13_only_acc"] - agg["no_rn_acc"]),
                "b13_only_minus_persistent_pp": 100.0 * (
                    agg["rn_b13_only_acc"] - agg["rn_persistent_acc"]
                ),
                "accuracy_effect_retained": safe_retention(
                    agg["no_rn_acc"], agg["rn_b13_only_acc"], agg["rn_persistent_acc"]
                ),
                "no_rn_logit_margin": agg["no_rn_margin"],
                "b13_only_logit_margin": agg["rn_b13_only_margin"],
                "persistent_logit_margin": agg["rn_persistent_margin"],
                "logit_margin_effect_retained": safe_retention(
                    agg["no_rn_margin"],
                    agg["rn_b13_only_margin"],
                    agg["rn_persistent_margin"],
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Intermediate-state trajectory: touch-and-go vs persistent RN
# =============================================================================


def per_row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1.0e-12)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA after centering rows; computed on GPU when available."""
    if x.shape[0] < 2:
        return float("nan")
    dev = torch.device(DEVICE)
    x = x.to(device=dev, dtype=torch.float32)
    y = y.to(device=dev, dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.T @ y
    hsic = (xty * xty).sum()
    xnorm = ((x.T @ x) ** 2).sum().sqrt()
    ynorm = ((y.T @ y) ** 2).sum().sqrt()
    denom = xnorm * ynorm
    if float(denom.detach().cpu()) <= 1.0e-20:
        return float("nan")
    return float((hsic / denom).detach().cpu().item())


def trajectory_metrics(
    persistent_delta: torch.Tensor,
    touch_delta: torch.Tensor,
) -> Dict[str, float]:
    """Metrics for [N,D] induced-delta matrices."""
    p = persistent_delta.float()
    t = touch_delta.float()
    pnorm = p.norm(dim=-1).clamp_min(1.0e-12)
    tnorm = t.norm(dim=-1)
    cos = per_row_cosine(t, p)
    ratio = tnorm / pnorm
    relerr = (t - p).norm(dim=-1) / pnorm
    return {
        "n": int(p.shape[0]),
        "mean_delta_cosine_touch_vs_persistent": float(cos.mean().item()),
        "mean_touch_over_persistent_delta_norm": float(ratio.mean().item()),
        "mean_relative_delta_error": float(relerr.mean().item()),
        "mean_persistent_delta_norm": float(pnorm.mean().item()),
        "mean_touch_delta_norm": float(tnorm.mean().item()),
        "linear_cka_touch_vs_persistent": linear_cka(t.cpu(), p.cpu()),
    }


@torch.inference_mode()
def evaluate_trajectory_receiver(
    bundle: ReceiverBundle,
    rn_token: torch.Tensor,
    specs: Mapping[str, SubsetSpec],
    samples_per_attack_subset: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    # Keep only summary matrices in CPU memory; nothing per-image is written out.
    store: Dict[Tuple[str, int], Dict[str, List[torch.Tensor]]] = {}
    for view in ("cls", "patch_mean"):
        for block in CAPTURE_BLOCKS:
            store[(view, block)] = {"persistent": [], "touch": []}
    final_store = {"persistent": [], "touch": []}

    for subset in ATTACK_SUBSETS:
        base_spec = specs[subset]
        chosen = deterministic_indices(
            base_spec.indices,
            samples_per_attack_subset,
            f"trajectory:{bundle.alias}:{subset}",
        )
        spec = SubsetSpec(subset, base_spec.dataset, chosen)
        ds = ImageOnlySubsetDataset(spec, bundle.preprocess)
        loader = make_loader(ds, batch_size, num_workers)

        for images in tqdm(loader, desc=f"{bundle.alias} | trajectory | {subset}", ncols=110):
            images = images.to(DEVICE, non_blocking=DEVICE == "cuda").float()
            base = run_receiver_visual(
                bundle.model, images, rn_token, "no_rn", capture_blocks=CAPTURE_BLOCKS
            )
            touch = run_receiver_visual(
                bundle.model, images, rn_token, "rn_b13_only", capture_blocks=CAPTURE_BLOCKS
            )
            persist = run_receiver_visual(
                bundle.model, images, rn_token, "rn_persistent", capture_blocks=CAPTURE_BLOCKS
            )

            for block in CAPTURE_BLOCKS:
                xb = base.states[block]
                xt = touch.states[block]
                xp = persist.states[block]
                if not (xb.shape == xt.shape == xp.shape):
                    raise RuntimeError(
                        f"Ordinary-token state mismatch at B{block}: "
                        f"base={tuple(xb.shape)} touch={tuple(xt.shape)} persistent={tuple(xp.shape)}"
                    )

                db_cls = xp[:, 0, :] - xb[:, 0, :]
                dt_cls = xt[:, 0, :] - xb[:, 0, :]
                db_patch = (xp[:, 1:, :] - xb[:, 1:, :]).mean(dim=1)
                dt_patch = (xt[:, 1:, :] - xb[:, 1:, :]).mean(dim=1)

                store[("cls", block)]["persistent"].append(db_cls.detach().cpu())
                store[("cls", block)]["touch"].append(dt_cls.detach().cpu())
                store[("patch_mean", block)]["persistent"].append(db_patch.detach().cpu())
                store[("patch_mean", block)]["touch"].append(dt_patch.detach().cpu())

            # Final normalized image embedding: the behavior-facing representation.
            zb = normalize(base.embedding)
            zt = normalize(touch.embedding)
            zp = normalize(persist.embedding)
            final_store["persistent"].append((zp - zb).detach().cpu())
            final_store["touch"].append((zt - zb).detach().cpu())

    rows: List[Dict[str, Any]] = []
    for (view, block), values in store.items():
        p = torch.cat(values["persistent"], dim=0)
        t = torch.cat(values["touch"], dim=0)
        rows.append(
            {
                "receiver": bundle.alias,
                "view": view,
                "stage": f"B{block}",
                "block": int(block),
                **trajectory_metrics(p, t),
            }
        )

    p = torch.cat(final_store["persistent"], dim=0)
    t = torch.cat(final_store["touch"], dim=0)
    rows.append(
        {
            "receiver": bundle.alias,
            "view": "image_embedding",
            "stage": "final_normalized_embedding",
            "block": 24,
            **trajectory_metrics(p, t),
        }
    )
    return pd.DataFrame(rows)


# =============================================================================
# Paper plots
# =============================================================================


def save_benchmark_plot(df: pd.DataFrame, out_dir: Path) -> None:
    receiver_order = [r[0] for r in RECEIVERS]
    x = np.arange(len(SUBSET_ORDER), dtype=float)
    width = 0.24
    offsets = {
        "no_rn": -width,
        "rn_b13_only": 0.0,
        "rn_persistent": width,
    }
    hatches = {"no_rn": "..", "rn_b13_only": "//", "rn_persistent": ""}

    fig, axes = plt.subplots(2, len(receiver_order), figsize=(15, 8.5), sharex="col")
    if len(receiver_order) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, receiver in enumerate(receiver_order):
        rname = next(r[1] for r in RECEIVERS if r[0] == receiver)
        rdf = df[df.receiver == receiver]
        for condition in CONDITION_ORDER:
            cdf = rdf[rdf.condition == condition].set_index("subset").loc[list(SUBSET_ORDER)]
            pos = x + offsets[condition]

            acc = cdf["accuracy"].to_numpy(float)
            lo = cdf["accuracy_ci95_lo"].to_numpy(float)
            hi = cdf["accuracy_ci95_hi"].to_numpy(float)
            yerr = np.vstack([acc - lo, hi - acc])
            axes[0, col].bar(
                pos,
                acc,
                width=width,
                label=CONDITION_LABELS[condition],
                hatch=hatches[condition],
                alpha=0.9,
                yerr=yerr,
                capsize=2,
            )

            margin = cdf["mean_logit_margin"].to_numpy(float)
            mse = cdf["logit_margin_se"].to_numpy(float)
            axes[1, col].bar(
                pos,
                margin,
                width=width,
                label=CONDITION_LABELS[condition],
                hatch=hatches[condition],
                alpha=0.9,
                yerr=1.96 * mse,
                capsize=2,
            )

        axes[0, col].set_title(rname)
        axes[0, col].set_ylim(0.0, 1.03)
        axes[0, col].set_ylabel("Binary accuracy")
        axes[1, col].set_ylabel("Object − attack logit margin")
        axes[1, col].axhline(0.0, linewidth=0.8)
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels(SUBSET_ORDER, rotation=25, ha="right")
        for row in range(2):
            axes[row, col].grid(axis="y", alpha=0.22)

    axes[0, 0].legend(frameon=True, fontsize=9)
    fig.suptitle("RN transplant: B13 touch-and-go vs persistent token", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "rn_touch_go_benchmark.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "rn_touch_go_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)


def save_trajectory_plot(df: pd.DataFrame, out_dir: Path) -> None:
    receiver_order = [r[0] for r in RECEIVERS]
    fig, axes = plt.subplots(2, len(receiver_order), figsize=(14, 8), sharex="col")
    if len(receiver_order) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, receiver in enumerate(receiver_order):
        rname = next(r[1] for r in RECEIVERS if r[0] == receiver)
        rdf = df[(df.receiver == receiver) & (df.block <= 23)]
        for view, label, marker in (
            ("cls", "CLS delta", "o"),
            ("patch_mean", "Patch-mean delta", "s"),
        ):
            v = rdf[rdf.view == view].sort_values("block")
            axes[0, col].plot(
                v["block"],
                v["mean_delta_cosine_touch_vs_persistent"],
                marker=marker,
                label=label,
            )
            axes[1, col].plot(
                v["block"],
                v["mean_touch_over_persistent_delta_norm"],
                marker=marker,
                label=label,
            )

        axes[0, col].set_title(rname)
        axes[0, col].set_ylabel("cos(Δ touch, Δ persistent)")
        axes[1, col].set_ylabel("||Δ touch|| / ||Δ persistent||")
        axes[1, col].set_xlabel("Block")
        axes[0, col].set_ylim(-0.05, 1.05)
        for row in range(2):
            axes[row, col].axvline(20, linestyle=":", linewidth=1.0, alpha=0.7)
            axes[row, col].grid(alpha=0.22)
            axes[row, col].set_xticks(list(range(13, 24)))
        axes[1, col].axhline(1.0, linestyle="--", linewidth=0.8, alpha=0.6)

    axes[0, 0].legend(frameon=True, fontsize=9)
    fig.suptitle(
        "Downstream state written by a B13-only RN vs persistent RN\n"
        "(pooled attacked images; vertical marker = B20)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "rn_touch_go_trajectory.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "rn_touch_go_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Compact text report
# =============================================================================


def make_text_report(benchmark: pd.DataFrame, retention: pd.DataFrame, trajectory: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("RN TRANSPLANT: B13 TOUCH-AND-GO VS PERSISTENT RN")
    lines.append("=" * 72)
    lines.append("")

    for receiver in [r[0] for r in RECEIVERS]:
        lines.append(f"[{receiver}] benchmark")
        rdf = benchmark[benchmark.receiver == receiver].copy()
        pivot_acc = rdf.pivot(index="subset", columns="condition", values="accuracy").loc[list(SUBSET_ORDER)]
        pivot_margin = rdf.pivot(index="subset", columns="condition", values="mean_logit_margin").loc[list(SUBSET_ORDER)]
        table = pd.DataFrame(
            {
                "NoRN acc": pivot_acc["no_rn"],
                "B13 acc": pivot_acc["rn_b13_only"],
                "Persist acc": pivot_acc["rn_persistent"],
                "NoRN margin": pivot_margin["no_rn"],
                "B13 margin": pivot_margin["rn_b13_only"],
                "Persist margin": pivot_margin["rn_persistent"],
            }
        )
        lines.append(table.to_string(float_format=lambda x: f"{x:.4f}"))
        lines.append("")

        rr = retention[
            (retention.receiver == receiver) & (retention.subset == "ATTACK_ALL_WEIGHTED")
        ]
        if len(rr):
            r = rr.iloc[0]
            lines.append(
                "Weighted attacked-subset retention: "
                f"accuracy={float(r['accuracy_effect_retained']):.4f}, "
                f"logit-margin={float(r['logit_margin_effect_retained']):.4f}, "
                f"B13-only minus persistent={float(r['b13_only_minus_persistent_pp']):+.3f} pp"
            )

        final = pd.DataFrame()
        if not trajectory.empty and {"receiver", "view"}.issubset(trajectory.columns):
            final = trajectory[
                (trajectory.receiver == receiver) & (trajectory.view == "image_embedding")
            ]
        if len(final):
            r = final.iloc[0]
            lines.append(
                "Final normalized-embedding RN-delta agreement: "
                f"cos={float(r['mean_delta_cosine_touch_vs_persistent']):.4f}, "
                f"norm-ratio={float(r['mean_touch_over_persistent_delta_norm']):.4f}, "
                f"CKA={float(r['linear_cka_touch_vs_persistent']):.4f}"
            )
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# CLI / main
# =============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Paper-focused RN transplant: persistent vs B13 touch-and-go."
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output folder. Default: subfolder beside this script.",
    )
    # Reproduction front-end override: preserve the historical defaults while
    # removing machine-specific checkpoint paths from the public workflow.
    ap.add_argument(
        "--donor-checkpoint",
        default=DONOR_MODELS[0][1],
        help="Full trained checkpoint containing the learned RN token.",
    )
    ap.add_argument(
        "--openai-spec",
        default=RECEIVERS[0][2],
        help="Canonical vanilla OpenAI CLIP receiver spec (normally ViT-L/14).",
    )
    ap.add_argument(
        "--gmp-checkpoint",
        default=RECEIVERS[1][2],
        help="GmP ViT-L/14 receiver checkpoint.",
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    ap.add_argument(
        "--max-samples-per-subset",
        type=int,
        default=0,
        help="0 = full six benchmark subsets; positive integer = deterministic debug subset.",
    )
    ap.add_argument(
        "--trajectory-samples-per-attack-subset",
        type=int,
        default=DEFAULT_TRAJECTORY_SAMPLES_PER_ATTACK_SUBSET,
        help="Deterministic attacked-image sample for intermediate-state summaries.",
    )
    ap.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Run only the full benchmark/retention experiment.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Keep all downstream reporting/plotting code unchanged by rebinding the
    # two historically hard-coded local checkpoint specs from CLI values.
    global DONOR_MODELS, RECEIVERS
    DONOR_MODELS = [
        ("My-CLIP!wise-0p6-both__A4__sigmoid_b13RN", str(args.donor_checkpoint)),
    ]
    RECEIVERS = [
        ("OAI", "OpenAI CLIP ViT-L/14", str(args.openai_spec)),
        ("GmP", "GmP ViT-L/14", str(args.gmp_checkpoint)),
    ]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("#" * 92)
    print("RN TRANSPLANT / B13 TOUCH-AND-GO PAPER RUN")
    print(f"device={DEVICE} | fp32=True | deterministic=True | seed={SEED}")
    print(f"outputs={out_dir}")
    print("#" * 92)

    rn_token, donor_meta = load_donor_rn()
    specs = load_subset_specs(args.max_samples_per_subset)

    benchmark_frames: List[pd.DataFrame] = []
    trajectory_frames: List[pd.DataFrame] = []
    parity: Dict[str, float] = {}
    receiver_meta: List[Dict[str, Any]] = []

    for alias, display_name, spec in RECEIVERS:
        print("\n" + "#" * 92)
        print(f"RECEIVER: {display_name}")
        print("#" * 92)
        bundle = load_receiver(alias, display_name, spec)
        parity[alias] = parity_check(bundle, specs["SCAM"])
        receiver_meta.append(
            {
                "alias": alias,
                "display_name": display_name,
                "spec": spec,
                "visual_width": int(bundle.model.visual.class_embedding.numel()),
                "logit_scale": model_logit_scale(bundle.model),
            }
        )

        bench = evaluate_benchmark_receiver(
            bundle,
            rn_token,
            specs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        benchmark_frames.append(bench)

        # Save partial benchmark progress after each receiver.
        pd.concat(benchmark_frames, ignore_index=True).to_csv(
            out_dir / "benchmark_summary.csv", index=False
        )

        if not args.skip_trajectory:
            traj = evaluate_trajectory_receiver(
                bundle,
                rn_token,
                specs,
                samples_per_attack_subset=args.trajectory_samples_per_attack_subset,
                batch_size=max(1, min(args.batch_size, 8)),
                num_workers=args.num_workers,
            )
            trajectory_frames.append(traj)
            pd.concat(trajectory_frames, ignore_index=True).to_csv(
                out_dir / "trajectory_summary.csv", index=False
            )

        del bundle.model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    benchmark = pd.concat(benchmark_frames, ignore_index=True)
    retention = make_retention_summary(benchmark)
    retention.to_csv(out_dir / "retention_summary.csv", index=False)

    if trajectory_frames:
        trajectory = pd.concat(trajectory_frames, ignore_index=True)
    else:
        trajectory = pd.DataFrame()

    save_benchmark_plot(benchmark, out_dir)
    if not trajectory.empty:
        save_trajectory_plot(trajectory, out_dir)

    report = make_text_report(benchmark, retention, trajectory)
    (out_dir / "paper_summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    run_config = {
        "seed": SEED,
        "device": DEVICE,
        "fp32": True,
        "autocast": False,
        "tf32": False,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "insert_before_block": INSERT_BLOCK,
        "conditions": {
            "no_rn": "receiver unchanged",
            "rn_b13_only": "RN inserted before B13 and removed immediately after B13",
            "rn_persistent": "RN inserted before B13 and retained through B23",
        },
        "donor": donor_meta,
        "receivers": receiver_meta,
        "datasets": {
            "SCAM": SCAM_REPO,
            "RTA": RTA_REPO,
            "subset_counts": {k: len(specs[k].indices) for k in SUBSET_ORDER},
            "max_samples_per_subset": int(args.max_samples_per_subset),
        },
        "trajectory": {
            "enabled": not bool(args.skip_trajectory),
            "samples_per_attack_subset": int(args.trajectory_samples_per_attack_subset),
            "capture_blocks": list(CAPTURE_BLOCKS),
            "views": ["cls", "patch_mean", "final_normalized_embedding"],
            "raw_per_image_saved": False,
        },
        "parity_max_abs": parity,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        },
        "output_dir": str(out_dir),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("\n" + "#" * 92)
    print("DONE")
    print(f"Outputs: {out_dir}")
    print("#" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
