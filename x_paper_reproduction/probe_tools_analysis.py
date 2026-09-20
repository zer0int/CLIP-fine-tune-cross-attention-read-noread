r"""Shared analysis helpers: paired data, numerical summaries, tensor geometry, and output I/O.

Only implementation-identical functions with compatible global dependencies are
shared. Different register masks, pooling rules and pairing policies stay local.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datasets import load_dataset
from pathlib import Path
from typing import Any, Iterable, Sequence
import gc
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import torch
import torch.nn.functional as F

DIGITAL_SUBSETS = ("SynthSCAM", "SynthRTA")
HANDWRITTEN_SUBSETS = ("SCAM", "RTA")
SUBSET_ORDER = (
    "NoSCAM",
    "SCAM",
    "SynthSCAM",
    "NoRTA",
    "RTA",
    "SynthRTA",
)
EPS = 1e-12

@dataclass(frozen=True)
class PairSample:
    image: Any
    correct_label: str
    distractor_label: str
    sample_id: str
    subset: str


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _batches(samples: Sequence[PairSample], batch_size: int):
    for start in range(0, len(samples), batch_size):
        yield start, samples[start : start + batch_size]


def _modality(subset: str) -> str:
    if subset in HANDWRITTEN_SUBSETS:
        return "handwritten"
    if subset in DIGITAL_SUBSETS:
        return "digital"
    return "no_attack"


def _family(subset: str) -> str:
    return "SCAM" if "SCAM" in subset else "RTA"


def _pair_key(sample_id: str) -> str:
    match = re.match(
        r"^(?:NoSCAM|SCAM|SynthSCAM|NoRTA|RTA|SynthRTA)_(.+)$",
        str(sample_id),
    )
    if not match:
        raise ValueError(f"Cannot derive pair key from id={sample_id!r}")
    return match.group(1)


def load_samples() -> dict[str, list[PairSample]]:
    result: dict[str, list[PairSample]] = {name: [] for name in SUBSET_ORDER}

    print("[dataset] Loading BLISS-e-V/SCAM ...")
    scam = load_dataset("BLISS-e-V/SCAM", split="train")
    for row in scam:
        sample_id = str(row["id"])
        subset = next(
            (name for name in SUBSET_ORDER[:3] if sample_id.startswith(name)),
            None,
        )
        if subset is None:
            continue
        result[subset].append(
            PairSample(
                image=row["image"],
                correct_label=str(row["object_label"]),
                distractor_label=str(row["attack_word"]),
                sample_id=sample_id,
                subset=subset,
            )
        )

    print("[dataset] Loading zer0int/RTA-100-Triplet ...")
    rta = load_dataset("zer0int/RTA-100-Triplet", split="train")
    for row in rta:
        subset = str(row["type"])
        if subset not in result:
            continue
        result[subset].append(
            PairSample(
                image=row["image"],
                correct_label=str(row["object_label"]),
                distractor_label=str(row["attack_word"]),
                sample_id=str(row["id"]),
                subset=subset,
            )
        )

    for subset in SUBSET_ORDER:
        print(f"  {subset:10s}: {len(result[subset])}")

    return result


def preprocess_images(
    processor: Any,
    samples: Sequence[PairSample],
    device: torch.device,
) -> torch.Tensor:
    encoded = processor(
        images=[sample.image.convert("RGB") for sample in samples],
        return_tensors="pt",
    )
    return encoded["pixel_values"].to(
        device,
        non_blocking=device.type == "cuda",
    )


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8)


def projected_direction_per_head(
    out_proj_weight: torch.Tensor,
    direction_d: torch.Tensor,
    heads: int,
) -> torch.Tensor:
    width = out_proj_weight.shape[0]
    dh = width // heads
    rows = []
    for head in range(heads):
        wh = out_proj_weight[:, head * dh:(head + 1) * dh].float()
        rows.append(wh.T @ direction_d.float())
    return torch.stack(rows, dim=0)


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def parse_strs(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


@dataclass(frozen=True)
class DemoSpec:
    filename: str
    present_words: tuple[str, ...]
    text_words: tuple[str, ...] = ()
    note: str = ""


def _normalize_rows(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1)


def normalize_probs(base, probs0, batch: int, heads: int, tokens: int) -> torch.Tensor:
    return base.normalize_probs_shape(probs0, batch, heads, tokens).float()


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= EPS:
        return 1.0 if ss_res <= EPS else 0.0
    return 1.0 - ss_res / ss_tot


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def clear_attn_cache(blk) -> None:
    for attr in ("last_logits", "last_probs", "last_v", "last_z", "last_q", "last_k", "last_xin"):
        if hasattr(blk.attn, attr):
            setattr(blk.attn, attr, None)


def select_register_mask(
    norms_bp: torch.Tensor,
    threshold: float,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    B, P = norms_bp.shape
    out = torch.zeros(B, P, dtype=torch.bool, device=norms_bp.device)
    for bi in range(B):
        idx = torch.nonzero(norms_bp[bi] >= threshold, as_tuple=False).flatten()
        if maximum > 0 and idx.numel() > maximum:
            idx = idx[torch.topk(norms_bp[bi, idx], k=maximum).indices]
        if idx.numel() < minimum:
            idx = torch.topk(norms_bp[bi], k=min(minimum, P)).indices
        out[bi, idx] = True
    return out


@dataclass
class RNPayload:
    token: torch.Tensor
    insert_block: int
    checkpoint: str


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
