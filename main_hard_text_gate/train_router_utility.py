#!/usr/bin/env python3
"""Router-only Experiment A4 for PIECES sigmoid-all.

Goal
----
Train ONLY ``read_implant.trust_router`` from the existing phase-1B.5 checkpoint
using the clean construction-known OPEN/CLOSE targets from A3, plus an explicit
paired ImageNet ranking loss.  A4 is designed to make discrimination begin at
step 1 instead of spending roughly half the schedule merely paying the z=-4
calibration tax.

1. Route labels remain construction-known only. Arbitrary contrastive negatives
   never become route labels.
2. Balanced BCEWithLogitsLoss still calibrates OPEN upward and CLOSE downward.
3. Each ImageNet packet also contributes a matched OPEN-vs-CLOSE pair:
      supportive primary OPEN z  >  adversarial written CLOSE z.
   The ranking loss acts on their difference and is invariant to the shared
   global z=-4 bias.
4. Counterfactual utility remains telemetry only. It cannot create, filter, flip,
   or mask route labels.
5. Validation uses one fixed held-out sample/seed, including a step-0 baseline.
6. The LR stays at its full value for a long plateau before cosine decay, so the
   router gets a real discrimination-learning window after escaping its prior.

Training sources remain fail-closed to:

  1. ImageNet counterfactual-text training packets; and
  2. ``image_sets/salt_n_pepper`` (186 images total).

No SCAM, RTA, MVT, eval_bench*, eval_benchmark*, TextCaps, COCO, CLEVR, or other
benchmark/evaluation data is imported or sampled here.

Construction-known routing labels
---------------------------------
ImageNet packet:

    supportive-text image + ordinary correct object candidate -> OPEN
    adversarial-text image + ordinary written-word candidate   -> CLOSE

Salt/pepper packet:

    generated SALT/PEPPER object + matching shaker candidate    -> OPEN

The frozen model still supplies counterfactual utility telemetry for trust=0 vs
trust=1, but utility has ZERO influence on supervision. Construction alone decides
OPEN/CLOSE. Utility is logged only to measure whether the current OPEN action helps
or hurts after the routing policy is learned.

The router is trained with balanced BCEWithLogitsLoss on its pre-sigmoid logit
plus a paired ImageNet ranking term ``softplus(margin - (z_open-z_close))``.
No task loss, route rent, semantic ontology loss, or backbone loss is
backpropagated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "main_hard_text_gate"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything
import train_gmp_anytext_stage1 as core


SALT_PEPPER_RE = re.compile(r"^(salt|pepper)_([a-z]+)_(\d+)$", re.IGNORECASE)
SALT_PEPPER_TYPES = {
    "chess", "colt", "frog", "grenade", "lighthouse",
    "matryoshka", "mushroom", "nuclear", "prescription", "tiki",
}

# The exact prompt variation requested by the user is preserved here.  The
# semantic label itself can also vary via label_variants in training_config.json.
DEFAULT_PROMPT_TEMPLATES = (
    "{label}",
    "a {label}",
    "a photo of a {label}",
    "there is a {label}",
    "the image depicts a {label}",
)

DEFAULT_DESCRIPTION_TEMPLATES = (
    '{object_label} with the text "{word}"',
    'a {object_label} with the text "{word}"',
    'a photo of a {object_label} with the text "{word}"',
    'there is a {object_label} with the text "{word}"',
    'the image depicts a {object_label} with the text "{word}"',
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        str(key): (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple, set))
            else value
        )
        for key, value in row.items()
    }
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(clean)


def stable_int(*parts: Any) -> int:
    text = "\0".join(str(x) for x in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def format_label_prompt(template: str, label: str) -> str:
    """Format prompt surfaces while fixing the common a/an case."""
    article = "an" if label[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    text = str(template).replace("a {label}", f"{article} {{label}}")
    return text.format(label=label)


def resolve_path(project_root: Path, value: str | Path) -> Path:
    text = str(value)
    path = Path(text).expanduser()
    if path.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", text):
        return path
    return (project_root / path).resolve()


def resolve_model_source(project_root: Path, value: str | Path) -> str:
    """Resolve local checkpoints while leaving HF repository identifiers intact."""
    text = str(value)
    local_candidate = Path(text).expanduser()
    if not local_candidate.is_absolute():
        local_candidate = project_root / local_candidate
    if (
        local_candidate.exists()
        or re.match(r"^[A-Za-z]:[\\/]", text)
        or text.startswith((".", "~", "/", "\\\\"))
        or Path(text).suffix.lower() in {".pt", ".pth", ".safetensors"}
        or text.count("/") != 1
    ):
        return str(resolve_path(project_root, text))
    return text


def tensor_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    y = labels.detach().cpu().numpy().astype(np.int64)
    s = scores.detach().cpu().numpy().astype(np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney AUC with tie handling.
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    sorted_vals = combined[order]
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    rank_sum_pos = ranks[:len(pos)].sum()
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def balanced_accuracy(labels: torch.Tensor, probs: torch.Tensor, threshold: float = 0.5) -> float:
    y = labels.detach().cpu().bool()
    p = (probs.detach().cpu() >= threshold)
    vals: List[float] = []
    for cls in (False, True):
        mask = y.eq(cls)
        if bool(mask.any()):
            vals.append(float((p[mask] == y[mask]).float().mean()))
    return float(sum(vals) / len(vals)) if vals else float("nan")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Salt/pepper source
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SaltPepperItem:
    path: Path
    word: str
    object_type: str
    index: int


class SaltPepperRouteSource:
    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int,
        patch_count: int,
        seed: int,
        val_fraction: float,
        expected_total: int,
        prompt_templates: Sequence[str],
        label_variants: Sequence[str],
    ):
        self.root = root
        self.split = split
        self.image_size = int(image_size)
        self.patch_count = int(patch_count)
        self.prompt_templates = tuple(str(x) for x in prompt_templates)
        self.label_variants = tuple(str(x) for x in label_variants)
        if not self.prompt_templates:
            raise ValueError("salt_n_pepper prompt_templates cannot be empty")
        if not self.label_variants:
            raise ValueError("salt_n_pepper label_variants cannot be empty")

        all_items: List[SaltPepperItem] = []
        if root.is_dir():
            for path in sorted(root.rglob("*.png")):
                match = SALT_PEPPER_RE.match(path.stem)
                if not match:
                    continue
                word = match.group(1).lower()
                object_type = match.group(2).lower()
                index = int(match.group(3))
                if object_type not in SALT_PEPPER_TYPES:
                    raise ValueError(f"Unknown salt_n_pepper type in {path.name}: {object_type}")
                all_items.append(SaltPepperItem(path, word, object_type, index))

        if expected_total > 0 and len(all_items) != int(expected_total):
            raise RuntimeError(
                f"Expected exactly {expected_total} salt+pepper images in {root}, found {len(all_items)}"
            )
        if not all_items:
            raise RuntimeError(f"No salt_n_pepper images found in {root}")

        # Stratify the deterministic holdout by word+object family so the held-out
        # utility evaluation contains both SALT/PEPPER and all morphology families.
        grouped: Dict[Tuple[str, str], List[SaltPepperItem]] = defaultdict(list)
        for item in all_items:
            grouped[(item.word, item.object_type)].append(item)
        train_items: List[SaltPepperItem] = []
        val_items: List[SaltPepperItem] = []
        for key, items in sorted(grouped.items()):
            items = sorted(items, key=lambda x: stable_int(seed, key, x.path.name))
            if len(items) <= 1:
                n_val = 0
            else:
                n_val = max(1, min(len(items) - 1, round(len(items) * float(val_fraction))))
            val_items.extend(items[:n_val])
            train_items.extend(items[n_val:])
        self.items = train_items if split == "train" else val_items
        if not self.items:
            raise RuntimeError(f"salt_n_pepper split {split!r} is empty")
        self.total_items = all_items

    def coverage(self) -> Dict[str, Any]:
        by_word: Dict[str, int] = defaultdict(int)
        by_type: Dict[str, int] = defaultdict(int)
        for item in self.items:
            by_word[item.word] += 1
            by_type[item.object_type] += 1
        return {
            "split": self.split,
            "root": str(self.root),
            "items": len(self.items),
            "total_items": len(self.total_items),
            "by_word": dict(sorted(by_word.items())),
            "by_type": dict(sorted(by_type.items())),
        }

    def sample(self, rng: random.Random) -> core.Packet:
        item = rng.choice(self.items)
        with Image.open(item.path) as image:
            tensor, _ = core.image_and_mask_transform(
                image.convert("RGB"), None, self.image_size, flip=False
            )
        label_template = rng.choice(self.label_variants)
        label = label_template.format(word=item.word, type=item.object_type)
        prompt = format_label_prompt(rng.choice(self.prompt_templates), label)
        return core.Packet(
            images=[tensor],
            patch_masks=[torch.zeros(self.patch_count, dtype=torch.float32)],
            mask_weights=[0.0],
            present_targets=[1.0],
            readable_targets=[1.0],
            captions=[prompt],
            positive=torch.ones((1, 1), dtype=torch.bool),
            metadata={
                "source": "salt_n_pepper",
                "path": str(item.path),
                "word": item.word,
                "object_type": item.object_type,
                "candidate": prompt,
            },
        )


# -----------------------------------------------------------------------------
# Prompt variation for ImageNet packets
# -----------------------------------------------------------------------------


def infer_primary_label(packet: core.Packet) -> str:
    value = str(packet.metadata.get("primary_label") or "").strip()
    if value:
        return value
    caption = str(packet.captions[0])
    prefix = "a photo of a "
    if caption.lower().startswith(prefix):
        return caption[len(prefix):]
    return caption


def vary_imagenet_packet_prompts(
    packet: core.Packet,
    rng: random.Random,
    prompt_templates: Sequence[str],
    description_templates: Sequence[str],
) -> core.Packet:
    """Vary the semantic surfaces without changing the packet's positive matrix."""
    primary_label = infer_primary_label(packet)
    support_text = str(packet.metadata.get("support_text") or "")
    adversarial_text = str(packet.metadata.get("adversarial_text") or "")

    captions = list(packet.captions)
    primary_prompt = format_label_prompt(rng.choice(prompt_templates), primary_label)
    adversarial_prompt = format_label_prompt(rng.choice(prompt_templates), adversarial_text)
    captions[0] = primary_prompt
    captions[1] = f"<notext> {primary_prompt}"
    captions[7] = adversarial_prompt

    if support_text:
        captions[5] = rng.choice(description_templates).format(
            object_label=primary_label, word=support_text
        )
    if adversarial_text:
        captions[6] = rng.choice(description_templates).format(
            object_label=primary_label, word=adversarial_text
        )

    metadata = dict(packet.metadata)
    metadata["primary_label"] = primary_label
    metadata["varied_primary_prompt"] = primary_prompt
    metadata["varied_adversarial_prompt"] = adversarial_prompt
    return core.Packet(
        images=packet.images,
        patch_masks=packet.patch_masks,
        mask_weights=packet.mask_weights,
        present_targets=packet.present_targets,
        readable_targets=packet.readable_targets,
        captions=captions,
        positive=packet.positive,
        source_triplets=packet.source_triplets,
        auto_triplets=packet.auto_triplets,
        invariance_pairs=packet.invariance_pairs,
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Exact counterfactual utility teacher
# -----------------------------------------------------------------------------


def exact_single_cell_loss_delta(
    closed_logits: torch.Tensor,
    positive: torch.Tensor,
    open_delta: torch.Tensor,
) -> torch.Tensor:
    """Exact Δ multi-positive CLIP loss for opening each cell independently.

    Returns a matrix ``delta_loss[i,j]`` such that a negative value means opening
    only cell (i,j) improves the current symmetric multi-positive objective.
    The derivation matches ``core.multi_positive_clip_loss`` exactly.
    """
    logits = closed_logits.float()
    positive = positive.to(device=logits.device, dtype=torch.bool)
    d = open_delta.float().clamp_min(0.0)
    expm1_d = torch.expm1(d)
    neg_inf = torch.finfo(logits.dtype).min

    row_valid = positive.any(dim=1)
    col_valid = positive.any(dim=0)
    have_row = bool(row_valid.any())
    have_col = bool(col_valid.any())
    if not have_row and not have_col:
        return torch.zeros_like(logits)

    total = torch.zeros_like(logits)
    components = 0

    if have_row:
        row_prob_all = torch.softmax(logits, dim=1)
        row_den_delta = torch.log1p(row_prob_all * expm1_d)

        row_pos_lse = torch.logsumexp(logits.masked_fill(~positive, neg_inf), dim=1)
        row_prob_pos = torch.zeros_like(logits)
        row_prob_pos[row_valid] = torch.exp(
            logits[row_valid] - row_pos_lse[row_valid, None]
        ) * positive[row_valid].to(logits.dtype)
        row_num_delta = torch.log1p(row_prob_pos * expm1_d)
        row_local = row_den_delta - row_num_delta
        row_local[~row_valid] = 0.0
        total = total + row_local / float(int(row_valid.sum()))
        components += 1

    if have_col:
        col_prob_all = torch.softmax(logits, dim=0)
        col_den_delta = torch.log1p(col_prob_all * expm1_d)

        col_pos_lse = torch.logsumexp(logits.masked_fill(~positive, neg_inf), dim=0)
        col_prob_pos = torch.zeros_like(logits)
        col_prob_pos[:, col_valid] = torch.exp(
            logits[:, col_valid] - col_pos_lse[None, col_valid]
        ) * positive[:, col_valid].to(logits.dtype)
        col_num_delta = torch.log1p(col_prob_pos * expm1_d)
        col_local = col_den_delta - col_num_delta
        col_local[:, ~col_valid] = 0.0
        total = total + col_local / float(int(col_valid.sum()))
        components += 1

    return total / float(components)


@dataclass
class TeacherSelection:
    z: torch.Tensor
    target: torch.Tensor
    delta_loss: torch.Tensor
    open_delta: torch.Tensor
    source_names: List[str]
    role_names: List[str]
    stats: Dict[str, Any]


@dataclass
class RankPairSelection:
    z_open: torch.Tensor
    z_close: torch.Tensor
    pair_ids: List[str]

    @property
    def gap(self) -> torch.Tensor:
        return self.z_open - self.z_close


def _top_quantile_indices(values: torch.Tensor, quantile: float, largest: bool) -> torch.Tensor:
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    if values.numel() == 1:
        return torch.zeros(1, dtype=torch.long, device=values.device)
    q = float(max(0.0, min(1.0, quantile)))
    threshold = torch.quantile(values.detach(), q if largest else (1.0 - q))
    if largest:
        return torch.nonzero(values >= threshold, as_tuple=False).flatten()
    return torch.nonzero(values <= threshold, as_tuple=False).flatten()


def select_balanced_teacher_targets(
    *,
    z_auto: torch.Tensor,
    delta_auto: torch.Tensor,
    open_delta_auto: torch.Tensor,
    construction_target_auto: torch.Tensor,
    construction_roles_auto: Mapping[Tuple[int, int], str],
    image_source_names: Sequence[str],
    targets_per_class: int,
    min_imagenet_open_fraction: float,
    rng: random.Random,
) -> TeacherSelection:
    """Select a balanced construction-known OPEN/CLOSE batch.

    A4 deliberately does *not* use counterfactual utility to create, filter, or
    flip route targets.  Construction is the answer key.  ``delta_auto`` and
    ``open_delta_auto`` are retained only as telemetry so we can separately ask
    whether the current OPEN action has useful downstream leverage.

    With the default batch construction there are 4 ImageNet OPEN cells,
    4 ImageNet CLOSE cells, and 3 salt/pepper OPEN cells.  We sample exactly
    ``targets_per_class`` from each side and require a minimum ImageNet fraction
    among OPEN examples so generated-domain identity cannot solve the task.
    """
    target = construction_target_auto.float()
    known = torch.isfinite(target)
    known_open = known & target.eq(1.0)
    known_close = known & target.eq(0.0)

    open_flat = torch.nonzero(known_open.reshape(-1), as_tuple=False).flatten()
    close_flat = torch.nonzero(known_close.reshape(-1), as_tuple=False).flatten()
    width = int(z_auto.shape[1])
    k = int(targets_per_class)
    if k <= 0:
        raise ValueError(f"targets_per_class must be positive, got {k}")
    if int(open_flat.numel()) < k or int(close_flat.numel()) < k:
        raise RuntimeError(
            "Construction-known route batch is too small; A4 requires an optimizer update every step. "
            f"OPEN={int(open_flat.numel())}, CLOSE={int(close_flat.numel())}, required_each={k}."
        )

    open_values = [int(x) for x in open_flat.detach().cpu().tolist()]
    close_values = [int(x) for x in close_flat.detach().cpu().tolist()]
    open_imagenet_values = [
        idx for idx in open_values
        if str(image_source_names[int(idx) // width]) == "imagenet"
    ]
    frac = float(max(0.0, min(1.0, min_imagenet_open_fraction)))
    required_imagenet = min(k, int(math.ceil(k * frac)))
    if len(open_imagenet_values) < required_imagenet:
        raise RuntimeError(
            "A4 domain-shortcut guard cannot be satisfied: "
            f"need {required_imagenet} ImageNet OPEN targets, found {len(open_imagenet_values)}."
        )

    def choose_values(values: Sequence[int], count: int) -> List[int]:
        values = list(int(x) for x in values)
        rng.shuffle(values)
        return values[:count]

    chosen_open = choose_values(open_imagenet_values, required_imagenet)
    chosen_set = set(chosen_open)
    remaining_open = [idx for idx in open_values if idx not in chosen_set]
    chosen_open.extend(choose_values(remaining_open, k - len(chosen_open)))
    chosen_close = choose_values(close_values, k)
    if len(chosen_open) != k or len(chosen_close) != k:
        raise RuntimeError(
            f"A4 failed to build exact balanced targets: OPEN={len(chosen_open)}, CLOSE={len(chosen_close)}, expected={k}."
        )

    selected_values = chosen_open + chosen_close
    selected = torch.tensor(selected_values, dtype=torch.long, device=z_auto.device)
    target_selected = torch.cat([
        torch.ones(k, device=z_auto.device, dtype=torch.float32),
        torch.zeros(k, device=z_auto.device, dtype=torch.float32),
    ])

    z_flat = z_auto.reshape(-1)
    delta_flat = delta_auto.reshape(-1)
    od_flat = open_delta_auto.reshape(-1)
    source_names: List[str] = []
    role_names: List[str] = []
    source_counts_open: Dict[str, int] = defaultdict(int)
    source_counts_close: Dict[str, int] = defaultdict(int)
    role_counts_open: Dict[str, int] = defaultdict(int)
    role_counts_close: Dict[str, int] = defaultdict(int)
    for label, flat_index in zip(target_selected.detach().cpu().tolist(), selected_values):
        image_index = int(flat_index) // width
        auto_index = int(flat_index) % width
        source = str(image_source_names[image_index])
        role = str(construction_roles_auto.get((image_index, auto_index), "unknown"))
        source_names.append(source)
        role_names.append(role)
        if label >= 0.5:
            source_counts_open[source] += 1
            role_counts_open[role] += 1
        else:
            source_counts_close[source] += 1
            role_counts_close[role] += 1

    return TeacherSelection(
        z=z_flat[selected],
        target=target_selected,
        delta_loss=delta_flat[selected].detach(),
        open_delta=od_flat[selected].detach(),
        source_names=source_names,
        role_names=role_names,
        stats={
            "construction_open_known": int(known_open.sum()),
            "construction_close_known": int(known_close.sum()),
            "candidate_open_pool": int(open_flat.numel()),
            "candidate_close_pool": int(close_flat.numel()),
            "selected_open": k,
            "selected_close": k,
            "open_source_counts": dict(source_counts_open),
            "close_source_counts": dict(source_counts_close),
            "open_role_counts": dict(role_counts_open),
            "close_role_counts": dict(role_counts_close),
        },
    )

# -----------------------------------------------------------------------------
# Model loading / compact checkpointing
# -----------------------------------------------------------------------------


def _state_blocks(state: Mapping[str, Any], key: str) -> Optional[List[int]]:
    value = state.get(key)
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return None


def configure_topology_from_compact(model: torch.nn.Module, compact: Mapping[str, Any]) -> None:
    state = compact.get("implant_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError("Starting checkpoint lacks implant_state_dict")
    for state_key, setter_name in (
        ("tap_blocks", "set_tap_blocks"),
        ("ortho_tap_blocks", "set_ortho_tap_blocks"),
        ("source_tap_blocks", "set_source_tap_blocks"),
    ):
        blocks = _state_blocks(state, state_key)
        if blocks is not None:
            getattr(model.read_implant, setter_name)(blocks, reset_uniform=False)


def load_start_model(
    config: Mapping[str, Any],
    cfg: Mapping[str, Any],
    start_checkpoint: Path,
    device: torch.device,
) -> Tuple[Any, torch.nn.Module, Mapping[str, Any], str]:
    compact = core.torch_load_trusted(start_checkpoint)
    if not isinstance(compact, Mapping) or "implant_state_dict" not in compact:
        raise RuntimeError(
            f"Router utility requires the compact phase_1b5_best.pt checkpoint; got unsupported {start_checkpoint}"
        )
    required_phase = str(config["router_utility"].get("require_checkpoint_phase", "1b5"))
    phase = str(compact.get("phase", ""))
    if required_phase and phase != required_phase:
        raise RuntimeError(
            f"Refusing starting checkpoint phase={phase!r}; expected {required_phase!r}. "
            "Experiment A4 must start before the 1C anti-reading collapse."
        )

    # Standard runs derive the seed model from this run's JOINT output. Explicit
    # base_model_override remains available for nonstandard continuation configs.
    base_source = resolve_model_source(PROJECT_ROOT, str(cfg["base_model_override"]))
    if Path(base_source).suffix and not Path(base_source).is_file():
        raise FileNotFoundError(
            f"Router base model is missing: {base_source}. "
            "Restore the current run's JOINT ordinary checkpoint or set router_utility.base_model_override."
        )

    global_cfg = config["global"]
    package_name = str(global_cfg.get("clip_package", "gmpclipattnamp"))
    clip_module = importlib.import_module(package_name)
    architecture = str(compact.get("read_attention_architecture") or global_cfg.get("read_attention_architecture", "sigmoid_all"))
    read_null_enabled = bool(global_cfg.get("read_null_enabled", True))
    read_null_insert_block = int(global_cfg.get("read_null_insert_block", 20))
    model, _, load_info = load_openai_clip_anything(
        clip_module,
        base_source,
        device=str(device),
        read_attention_architecture=architecture,
        read_null_enabled=read_null_enabled,
        read_null_insert_block=read_null_insert_block,
        reuse_full_model_pickle=False,
    )
    print(f"[model] source={load_info.source_kind} format={load_info.detected_format}")
    model.float()
    configure_topology_from_compact(model, compact)
    core.load_optional_implant(model, start_checkpoint)

    if not bool(getattr(model, "read_null_enabled", False)):
        raise RuntimeError("Experiment A4 expects the phase-1B.5 READ_NULL token to be active")
    if str(getattr(model, "read_attention_architecture", "")) != "sigmoid_all":
        raise RuntimeError("Experiment A4 package is specifically for the sigmoid_all branch")

    return clip_module, model, compact, base_source


def save_compact(
    path: Path,
    *,
    model: torch.nn.Module,
    start_compact: Mapping[str, Any],
    base_model_path: str,
    step: int,
    metrics: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> None:
    obj = {
        "format": "gmp_anytext_router_utility_v3",
        "phase": "router_utility_a4",
        "source_phase": str(start_compact.get("phase", "")),
        "global_step": int(step),
        "base_model_path": str(base_model_path),
        "read_attention_architecture": str(model.read_attention_architecture),
        "hard_text_embedding": model.hard_text_embedding.detach().cpu(),
        "null_text_embedding": model.null_text_embedding.detach().cpu(),
        "read_null_token": model.visual.read_null_token.detach().cpu(),
        "read_null_insert_block": int(getattr(model, "read_null_insert_block", 20)),
        "implant_state_dict": {
            key: value.detach().cpu() for key, value in model.read_implant.state_dict().items()
        },
        "metrics": dict(metrics),
        "router_utility_a4": dict(resolved_config),
        "training_policy": {
            "trainable": ["read_implant.trust_router.*"],
            "task_loss_backprop": False,
            "route_rent": 0.0,
            "benchmark_training": False,
            "label_policy": "construction_known_balanced_bce_plus_paired_imagenet_rank",
            "validation_policy": "fixed_heldout_seed",
            "rank_weight": float(resolved_config["rank_weight"]),
            "rank_margin": float(resolved_config["rank_margin"]),
            "data_sources": ["imagenet", "salt_n_pepper"],
        },
    }
    torch.save(obj, path)


# -----------------------------------------------------------------------------
# Batch creation / forwarding
# -----------------------------------------------------------------------------


def build_construction_route_targets(
    packets: Sequence[core.Packet],
    packed: core.PackedBatch,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[Tuple[int, int], str]]:
    """Build fail-closed route labels from packet construction, not CE negatives."""
    target = torch.full(packed.positive.shape, float("nan"), dtype=torch.float32)
    roles: Dict[Tuple[int, int], str] = {}
    caption_to_global = {caption: index for index, caption in enumerate(packed.captions)}
    image_offset = 0

    def set_target(local_image: int, packet: core.Packet, local_caption: int, value: float, role: str) -> None:
        global_image = image_offset + int(local_image)
        caption = str(packet.captions[int(local_caption)])
        if caption not in caption_to_global:
            raise RuntimeError(f"Packed caption disappeared: {caption!r}")
        global_caption = int(caption_to_global[caption])
        previous = target[global_image, global_caption]
        if torch.isfinite(previous) and float(previous) != float(value):
            raise RuntimeError(
                f"Conflicting construction route labels at image={global_image} caption={caption!r}: "
                f"{float(previous)} vs {value}"
            )
        target[global_image, global_caption] = float(value)
        roles[(global_image, global_caption)] = str(role)

    for packet in packets:
        source = str(packet.metadata.get("source", "unknown"))
        if source == "imagenet":
            if len(packet.images) < 3 or len(packet.captions) < 8:
                raise RuntimeError("Unexpected ImageNet packet layout for router A4")
            # Supportive overlay should be allowed to help the ordinary visual
            # target.  This is the construction-known pro-reading half.
            set_target(1, packet, 0, 1.0, "imagenet_support_primary_open")
            # On the adversarial overlay, the written-class ordinary candidate is
            # the explicit auto-triplet negative in ImageNetPacketSource.
            set_target(2, packet, 7, 0.0, "imagenet_adversarial_written_close")
        elif source == "salt_n_pepper":
            if len(packet.images) != 1 or len(packet.captions) != 1:
                raise RuntimeError("Unexpected salt_n_pepper packet layout for router A4")
            set_target(0, packet, 0, 1.0, "salt_pepper_matching_shaker_open")
        else:
            raise RuntimeError(f"Benchmark/data firewall: unexpected router A4 packet source {source!r}")
        image_offset += len(packet.images)

    if image_offset != int(packed.images.shape[0]):
        raise RuntimeError(f"Route target image alignment failed: {image_offset} != {int(packed.images.shape[0])}")

    # Strong semantic sanity check: every OPEN construction cell must be a true
    # positive in the packet objective; every CLOSE construction cell must be a
    # true negative.  This catches exactly the A1 salt-caption bookkeeping bug.
    known = torch.isfinite(target)
    open_known = known & target.eq(1.0)
    close_known = known & target.eq(0.0)
    if bool((open_known & ~packed.positive).any()):
        bad = torch.nonzero(open_known & ~packed.positive, as_tuple=False)[0].tolist()
        raise RuntimeError(f"A4 construction OPEN is not a packet positive at cell {bad}")
    if bool((close_known & packed.positive).any()):
        bad = torch.nonzero(close_known & packed.positive, as_tuple=False)[0].tolist()
        raise RuntimeError(f"A4 construction CLOSE is unexpectedly a packet positive at cell {bad}")

    return target.to(device), roles


def build_batch(
    *,
    imagenet_source: core.ImageNetPacketSource,
    salt_source: SaltPepperRouteSource,
    rng: random.Random,
    imagenet_packets: int,
    salt_images: int,
    prompt_templates: Sequence[str],
    description_templates: Sequence[str],
    device: torch.device,
) -> Tuple[core.PackedBatch, List[str], List[str], torch.Tensor, Dict[Tuple[int, int], str]]:
    packets: List[core.Packet] = []
    for _ in range(int(imagenet_packets)):
        packet = imagenet_source.sample(rng)
        packet = vary_imagenet_packet_prompts(packet, rng, prompt_templates, description_templates)
        packets.append(packet)
    for _ in range(int(salt_images)):
        packets.append(salt_source.sample(rng))
    rng.shuffle(packets)

    image_sources: List[str] = []
    image_packet_ids: List[str] = []
    for packet_i, packet in enumerate(packets):
        source = str(packet.metadata.get("source", "unknown"))
        packet_id = f"{source}:{packet_i}"
        image_sources.extend([source] * len(packet.images))
        image_packet_ids.extend([packet_id] * len(packet.images))

    packed_cpu = core.pack_packets(packets)
    construction_target, construction_roles = build_construction_route_targets(
        packets, packed_cpu, device
    )
    batch = core.batch_to_device(packed_cpu, device)
    return batch, image_sources, image_packet_ids, construction_target, construction_roles


class RouterLogitCapture:
    def __init__(self, model: torch.nn.Module):
        self.value: Optional[torch.Tensor] = None
        self.handle = model.read_implant.trust_router.fc3.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.value = output.squeeze(-1)

    def close(self) -> None:
        self.handle.remove()


def build_imagenet_rank_pairs(
    *,
    z_auto: torch.Tensor,
    construction_roles_auto: Mapping[Tuple[int, int], str],
    image_packet_ids: Sequence[str],
) -> RankPairSelection:
    """Collect exact per-packet ImageNet OPEN/CLOSE logits for ranking.

    Each ImageNet packet contributes one supportive-primary OPEN cell and one
    adversarial-written CLOSE cell.  They live on different image rows, so the
    packet id is used to match them after packet shuffling/packing.  Salt rows do
    not participate in the ranking loss because they intentionally have no
    construction-known CLOSE twin.
    """
    grouped: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    width = int(z_auto.shape[1])
    for (image_index, auto_index), role in construction_roles_auto.items():
        if image_index < 0 or image_index >= len(image_packet_ids):
            raise RuntimeError(f"Rank-pair image index out of range: {image_index}")
        packet_id = str(image_packet_ids[image_index])
        if not packet_id.startswith("imagenet:"):
            continue
        if role == "imagenet_support_primary_open":
            key = "open"
        elif role == "imagenet_adversarial_written_close":
            key = "close"
        else:
            continue
        if auto_index < 0 or auto_index >= width:
            raise RuntimeError(f"Rank-pair auto index out of range: {auto_index}")
        if key in grouped[packet_id]:
            raise RuntimeError(f"Duplicate ImageNet rank {key} cell for {packet_id}")
        grouped[packet_id][key] = z_auto[image_index, auto_index]

    pair_ids = sorted(grouped)
    opens: List[torch.Tensor] = []
    closes: List[torch.Tensor] = []
    valid_ids: List[str] = []
    for packet_id in pair_ids:
        pair = grouped[packet_id]
        if set(pair) != {"open", "close"}:
            raise RuntimeError(
                f"Incomplete ImageNet rank pair for {packet_id}: keys={sorted(pair)}"
            )
        opens.append(pair["open"])
        closes.append(pair["close"])
        valid_ids.append(packet_id)
    if not opens:
        raise RuntimeError("No ImageNet rank pairs were constructed")
    return RankPairSelection(
        z_open=torch.stack(opens),
        z_close=torch.stack(closes),
        pair_ids=valid_ids,
    )


def teacher_from_forward(
    *,
    model: torch.nn.Module,
    details: Mapping[str, torch.Tensor],
    raw_router_z: torch.Tensor,
    batch: core.PackedBatch,
    image_sources: Sequence[str],
    image_packet_ids: Sequence[str],
    construction_target: torch.Tensor,
    construction_roles: Mapping[Tuple[int, int], str],
    telemetry_open_scale: float,
    targets_per_class: int,
    min_imagenet_open_fraction: float,
    rng: random.Random,
) -> Tuple[TeacherSelection, RankPairSelection, Dict[str, torch.Tensor]]:
    modes = details["mode_ids"]
    read_candidate_mask = ~modes.eq(2)
    selected_modes = modes[read_candidate_mask]
    auto_within_read = selected_modes.eq(0)
    any_mask = modes.eq(0)

    z_auto = raw_router_z[:, auto_within_read]
    if z_auto.shape[1] != int(any_mask.sum()):
        raise RuntimeError(
            f"Router logit alignment failed: z_auto={tuple(z_auto.shape)}, any={int(any_mask.sum())}"
        )
    if tuple(construction_target.shape) != tuple(batch.positive.shape):
        raise RuntimeError(
            f"Construction target shape mismatch: {tuple(construction_target.shape)} vs {tuple(batch.positive.shape)}"
        )

    known = torch.isfinite(construction_target)
    outside_any = known & (~any_mask[None, :])
    if bool(outside_any.any()):
        bad = torch.nonzero(outside_any, as_tuple=False)[0].tolist()
        raise RuntimeError(f"Construction route target landed on non-<any> candidate at cell {bad}")

    current_logits = details["logits_per_image"].float()
    current_auto = details["auto_read_contribution"].float()
    closed_logits = current_logits - current_auto

    relative = details["relative_read_logits"].float()
    source_gate = details["source_gate"].float()[:, None]
    positive_read = model.read_implant.positive_relative_read(relative).detach().float()
    open_delta = torch.zeros_like(closed_logits)
    open_delta[:, any_mask] = (
        source_gate * float(telemetry_open_scale) * positive_read[:, any_mask]
    )
    open_logits = closed_logits + open_delta

    delta_loss = exact_single_cell_loss_delta(closed_logits, batch.positive, open_delta).detach()
    full_any_cols = torch.nonzero(any_mask, as_tuple=False).flatten().detach().cpu().tolist()
    full_to_auto = {int(full): auto for auto, full in enumerate(full_any_cols)}
    roles_auto: Dict[Tuple[int, int], str] = {}
    for (image_index, full_caption), role in construction_roles.items():
        if int(full_caption) not in full_to_auto:
            raise RuntimeError(
                f"Construction role {role!r} refers to non-<any> caption index {full_caption}"
            )
        roles_auto[(int(image_index), int(full_to_auto[int(full_caption)]))] = str(role)

    selection = select_balanced_teacher_targets(
        z_auto=z_auto,
        delta_auto=delta_loss[:, any_mask],
        open_delta_auto=open_delta[:, any_mask],
        construction_target_auto=construction_target[:, any_mask],
        construction_roles_auto=roles_auto,
        image_source_names=image_sources,
        targets_per_class=targets_per_class,
        min_imagenet_open_fraction=min_imagenet_open_fraction,
        rng=rng,
    )
    rank_pairs = build_imagenet_rank_pairs(
        z_auto=z_auto,
        construction_roles_auto=roles_auto,
        image_packet_ids=image_packet_ids,
    )
    telemetry = {
        "closed_logits": closed_logits.detach(),
        "open_logits": open_logits.detach(),
        "current_logits": current_logits.detach(),
        "open_delta": open_delta.detach(),
        "delta_loss": delta_loss.detach(),
        "construction_target": construction_target.detach(),
    }
    return selection, rank_pairs, telemetry


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    clip_module: Any,
    imagenet_source: core.ImageNetPacketSource,
    salt_source: SaltPepperRouteSource,
    cfg: Mapping[str, Any],
    device: torch.device,
    seed: int,
    batches: int,
) -> Dict[str, Any]:
    """Evaluate on one deterministic held-out sample reused at every step."""
    model.eval()
    rng = random.Random(int(seed))
    capture = RouterLogitCapture(model)
    labels_all: List[torch.Tensor] = []
    logits_all: List[torch.Tensor] = []
    delta_all: List[torch.Tensor] = []
    source_all: List[str] = []
    role_all: List[str] = []
    source_targets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"open": 0, "close": 0})
    role_targets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"open": 0, "close": 0})
    loss_current: List[float] = []
    loss_closed: List[float] = []
    loss_open: List[float] = []
    selected_batches = 0
    rank_gaps_all: List[torch.Tensor] = []
    rank_losses: List[float] = []
    try:
        for _batch_i in range(int(batches)):
            batch, image_sources, image_packet_ids, construction_target, construction_roles = build_batch(
                imagenet_source=imagenet_source,
                salt_source=salt_source,
                rng=rng,
                imagenet_packets=int(cfg["imagenet_packets_per_batch"]),
                salt_images=int(cfg["salt_images_per_batch"]),
                prompt_templates=cfg["prompt_templates"],
                description_templates=cfg["description_templates"],
                device=device,
            )
            tokens = clip_module.tokenize(batch.captions, truncate=True).to(device)
            capture.value = None
            with core.amp_context(device, str(cfg["amp_dtype"])):
                details = model.forward_modes(batch.images, tokens, return_details=True)
            if capture.value is None:
                raise RuntimeError("Failed to capture trust_router.fc3 logits during validation")
            selection, rank_pairs, telemetry = teacher_from_forward(
                model=model,
                details=details,
                raw_router_z=capture.value,
                batch=batch,
                image_sources=image_sources,
                image_packet_ids=image_packet_ids,
                construction_target=construction_target,
                construction_roles=construction_roles,
                telemetry_open_scale=float(cfg["telemetry_open_scale"]),
                targets_per_class=int(cfg["targets_per_class"]),
                min_imagenet_open_fraction=float(cfg["min_imagenet_open_fraction"]),
                rng=rng,
            )
            rank_gap = rank_pairs.gap.detach().float().cpu()
            rank_gaps_all.append(rank_gap)
            rank_losses.append(float(F.softplus(float(cfg["rank_margin"]) - rank_gap).mean()))
            loss_current.append(float(core.multi_positive_clip_loss(telemetry["current_logits"], batch.positive)))
            loss_closed.append(float(core.multi_positive_clip_loss(telemetry["closed_logits"], batch.positive)))
            loss_open.append(float(core.multi_positive_clip_loss(telemetry["open_logits"], batch.positive)))
            selected_batches += 1
            labels_all.append(selection.target.cpu())
            logits_all.append(selection.z.detach().float().cpu())
            delta_all.append(selection.delta_loss.float().cpu())
            source_all.extend(selection.source_names)
            role_all.extend(selection.role_names)
            for label, source, role in zip(
                selection.target.cpu().tolist(), selection.source_names, selection.role_names
            ):
                key = "open" if label >= 0.5 else "close"
                source_targets[source][key] += 1
                role_targets[role][key] += 1
    finally:
        capture.close()

    if not labels_all:
        return {
            "validation_seed": int(seed),
            "selected_batches": 0,
            "targets": 0,
            "bce": float("nan"),
            "auc": float("nan"),
            "balanced_accuracy": float("nan"),
            "trust_open_mean": float("nan"),
            "trust_close_mean": float("nan"),
            "loss_current": float(np.mean(loss_current)) if loss_current else float("nan"),
            "loss_closed": float(np.mean(loss_closed)) if loss_closed else float("nan"),
            "loss_fully_open": float(np.mean(loss_open)) if loss_open else float("nan"),
            "source_targets": {},
            "role_targets": {},
            "per_source": {},
            "per_role": {},
            "imagenet_pair_rank_loss": float("nan"),
            "imagenet_pair_gap_mean": float("nan"),
            "imagenet_pair_gap_median": float("nan"),
            "imagenet_pair_accuracy": float("nan"),
            "validation_objective": float("nan"),
        }

    labels = torch.cat(labels_all)
    logits = torch.cat(logits_all)
    probs = logits.sigmoid()
    delta = torch.cat(delta_all)
    rank_gaps = torch.cat(rank_gaps_all) if rank_gaps_all else torch.empty(0)
    rank_loss_mean = float(np.mean(rank_losses)) if rank_losses else float("nan")

    def subgroup_metrics(names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name in sorted(set(names)):
            idx = torch.tensor([i for i, value in enumerate(names) if value == name], dtype=torch.long)
            yy = labels[idx]
            zz = logits[idx]
            pp = zz.sigmoid()
            rec: Dict[str, Any] = {
                "n": int(idx.numel()),
                "open": int((yy == 1).sum()),
                "close": int((yy == 0).sum()),
                "bce": float(F.binary_cross_entropy_with_logits(zz, yy)),
                "trust_mean": float(pp.mean()),
            }
            if bool((yy == 1).any()):
                rec["trust_open_mean"] = float(pp[yy == 1].mean())
                rec["z_open_mean"] = float(zz[yy == 1].mean())
            if bool((yy == 0).any()):
                rec["trust_close_mean"] = float(pp[yy == 0].mean())
                rec["z_close_mean"] = float(zz[yy == 0].mean())
            if bool((yy == 1).any()) and bool((yy == 0).any()):
                rec["auc"] = tensor_auc(yy, zz)
                rec["balanced_accuracy"] = balanced_accuracy(yy, pp)
            else:
                rec["auc"] = float("nan")
                rec["balanced_accuracy"] = float("nan")
            out[name] = rec
        return out

    per_source = subgroup_metrics(source_all)
    per_role = subgroup_metrics(role_all)
    imagenet_metrics = per_source.get("imagenet", {})
    salt_metrics = per_source.get("salt_n_pepper", {})
    return {
        "validation_seed": int(seed),
        "selected_batches": selected_batches,
        "targets": int(labels.numel()),
        "open_targets": int((labels == 1).sum()),
        "close_targets": int((labels == 0).sum()),
        "bce": float(F.binary_cross_entropy_with_logits(logits, labels)),
        "auc": tensor_auc(labels, logits),
        "balanced_accuracy": balanced_accuracy(labels, probs),
        "router_z_open_mean": float(logits[labels == 1].mean()),
        "router_z_close_mean": float(logits[labels == 0].mean()),
        "trust_open_mean": float(probs[labels == 1].mean()),
        "trust_close_mean": float(probs[labels == 0].mean()),
        "counterfactual_delta_loss_open_mean": float(delta[labels == 1].mean()),
        "counterfactual_delta_loss_close_mean": float(delta[labels == 0].mean()),
        "counterfactual_open_help_fraction": float((delta[labels == 1] < 0).float().mean()),
        "counterfactual_close_hurt_fraction": float((delta[labels == 0] > 0).float().mean()),
        "loss_current": float(np.mean(loss_current)),
        "loss_closed": float(np.mean(loss_closed)),
        "loss_fully_open": float(np.mean(loss_open)),
        "imagenet_auc": float(imagenet_metrics.get("auc", float("nan"))),
        "imagenet_bce": float(imagenet_metrics.get("bce", float("nan"))),
        "imagenet_trust_open_mean": float(imagenet_metrics.get("trust_open_mean", float("nan"))),
        "imagenet_trust_close_mean": float(imagenet_metrics.get("trust_close_mean", float("nan"))),
        "salt_trust_open_mean": float(salt_metrics.get("trust_open_mean", float("nan"))),
        "imagenet_pair_rank_loss": rank_loss_mean,
        "imagenet_pair_gap_mean": float(rank_gaps.mean()) if rank_gaps.numel() else float("nan"),
        "imagenet_pair_gap_median": float(rank_gaps.median()) if rank_gaps.numel() else float("nan"),
        "imagenet_pair_accuracy": float((rank_gaps > 0).float().mean()) if rank_gaps.numel() else float("nan"),
        "validation_objective": (
            float(F.binary_cross_entropy_with_logits(logits, labels))
            + float(cfg["rank_weight"]) * rank_loss_mean
            if math.isfinite(rank_loss_mean) else float("nan")
        ),
        "source_targets": {k: dict(v) for k, v in sorted(source_targets.items())},
        "role_targets": {k: dict(v) for k, v in sorted(role_targets.items())},
        "per_source": per_source,
        "per_role": per_role,
    }


# -----------------------------------------------------------------------------
# Configuration / main training
# -----------------------------------------------------------------------------


def resolve_router_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    if "router_utility" not in config or not isinstance(config["router_utility"], Mapping):
        raise KeyError("training_config.json requires top-level router_utility object")
    raw = dict(config["router_utility"])
    global_cfg = config["global"]
    paths = config["paths"]
    stages = config.get("stages", {})
    if not isinstance(stages, Mapping):
        raise KeyError("training_config.json requires top-level stages object")
    final_base_stage = stages.get("final_base", {})
    joint_stage = stages.get("joint", {})
    if not isinstance(final_base_stage, Mapping) or not isinstance(joint_stage, Mapping):
        raise KeyError("training_config.json requires stages.final_base and stages.joint objects")

    train_root = resolve_path(PROJECT_ROOT, global_cfg["train_root"])
    expected_out = train_root / Path(str(final_base_stage.get("output_subdir", "final/sigmoid_all/base")))
    expected_joint_base = (
        train_root
        / Path(str(joint_stage.get("output_subdir", "main/joint")))
        / "best_merged_state_dict__ungmp_oaiclip_fullmodel.pt"
    )
    out_dir = resolve_path(PROJECT_ROOT, raw.get("output_dir") or expected_out)
    if str(out_dir).lower() != str(expected_out).lower():
        raise RuntimeError(
            "Router utility is intentionally required to use the EXISTING final/sigmoid_all/base folder. "
            f"Expected {expected_out}, got {out_dir}"
        )
    start_checkpoint = resolve_path(
        PROJECT_ROOT,
        raw.get("start_checkpoint") or (expected_out / "phase_1b5_best.pt"),
    )
    if start_checkpoint.parent != expected_out or start_checkpoint.name != "phase_1b5_best.pt":
        raise RuntimeError(
            "Experiment A4 must start from final/sigmoid_all/base/phase_1b5_best.pt in the current train_root"
        )

    base_model_override = raw.get("base_model_override")
    base_model_source = (
        resolve_path(PROJECT_ROOT, base_model_override)
        if base_model_override not in (None, "")
        else expected_joint_base
    )

    output_prefix = str(raw.get("output_prefix") or "router_utility_a4").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_prefix):
        raise ValueError(f"Unsafe router_utility.output_prefix: {output_prefix!r}")

    resolved = {
        **raw,
        "output_prefix": output_prefix,
        "train_root": str(train_root),
        "output_dir": str(out_dir),
        "start_checkpoint": str(start_checkpoint),
        "base_model_override": str(base_model_source),
        "imagenet_text_root": str(resolve_path(PROJECT_ROOT, paths["imagenet_text_root"])),
        "imagenet_handwriting_root": str(resolve_path(PROJECT_ROOT, paths["imagenet_handwriting_root"])),
        "salt_n_pepper_root": str(resolve_path(PROJECT_ROOT, paths.get("salt_n_pepper_root", "image_sets/salt_n_pepper"))),
        "image_size": int(global_cfg.get("image_size", 224)),
        "read_attention_architecture": str(global_cfg.get("read_attention_architecture", "sigmoid_all")),
        "read_null_enabled": bool(global_cfg.get("read_null_enabled", True)),
        "amp_dtype": str(raw.get("amp_dtype") or global_cfg.get("precision", {}).get("autocast", "fp16")),
        "prompt_templates": list(raw.get("prompt_templates") or DEFAULT_PROMPT_TEMPLATES),
        "description_templates": list(raw.get("description_templates") or DEFAULT_DESCRIPTION_TEMPLATES),
    }
    return resolved


def validate_config_and_paths(config: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    if cfg["read_attention_architecture"] != "sigmoid_all":
        raise RuntimeError("Router utility Experiment A4 requires sigmoid_all")
    if not cfg["read_null_enabled"]:
        raise RuntimeError("Router utility Experiment A4 requires READ_NULL from phase 1B.5")
    if float(cfg.get("rank_weight", 0.0)) <= 0.0:
        raise RuntimeError("Router utility Experiment A4 requires rank_weight > 0")
    if float(cfg.get("rank_margin", 0.0)) < 0.0:
        raise RuntimeError("Router utility Experiment A4 requires rank_margin >= 0")
    if int(cfg.get("lr_hold_steps", 0)) < int(cfg.get("warmup_steps", 0)):
        raise RuntimeError("lr_hold_steps must be >= warmup_steps")
    if int(cfg.get("lr_hold_steps", 0)) >= int(cfg.get("steps", 0)):
        raise RuntimeError("lr_hold_steps must be < total steps")
    for key in ("start_checkpoint", "imagenet_text_root", "salt_n_pepper_root"):
        path = Path(str(cfg[key]))
        if key == "start_checkpoint" and not path.is_file():
            raise FileNotFoundError(f"Missing required starting checkpoint: {path}")
        if key != "start_checkpoint" and not path.is_dir():
            raise FileNotFoundError(f"Missing required training source directory {key}: {path}")
    base_model = Path(str(cfg["base_model_override"]))
    if base_model.suffix and not base_model.is_file():
        raise FileNotFoundError(f"Missing required router base model: {base_model}")

    # Fail-closed training-source firewall.  The script has no generic dataset
    # hook: only the two named sources above are constructible.
    allowed = {"imagenet", "salt_n_pepper"}
    requested = set(str(x) for x in cfg.get("data_sources", ["imagenet", "salt_n_pepper"]))
    if requested != allowed:
        raise RuntimeError(
            f"Benchmark firewall: router utility data_sources must be exactly {sorted(allowed)}, got {sorted(requested)}"
        )


def make_sources(
    *,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> Tuple[core.ImageNetPacketSource, core.ImageNetPacketSource, SaltPepperRouteSource, SaltPepperRouteSource]:
    patch_count = int(model.visual.positional_embedding.shape[0] - 1)
    handwriting_probability = float(cfg.get("imagenet_handwriting_probability", 0.0))
    imagenet_train = core.ImageNetPacketSource(
        Path(cfg["imagenet_text_root"]),
        Path(cfg["imagenet_handwriting_root"]),
        "train",
        int(cfg["image_size"]),
        patch_count,
        handwriting_probability,
        0.0,
        (),
    )
    imagenet_val = core.ImageNetPacketSource(
        Path(cfg["imagenet_text_root"]),
        Path(cfg["imagenet_handwriting_root"]),
        "val",
        int(cfg["image_size"]),
        patch_count,
        handwriting_probability,
        0.0,
        (),
    )
    salt_train = SaltPepperRouteSource(
        Path(cfg["salt_n_pepper_root"]),
        "train",
        int(cfg["image_size"]),
        patch_count,
        int(cfg["seed"]),
        float(cfg["salt_val_fraction"]),
        int(cfg["expected_salt_n_pepper_images"]),
        cfg["prompt_templates"],
        cfg["salt_label_variants"],
    )
    salt_val = SaltPepperRouteSource(
        Path(cfg["salt_n_pepper_root"]),
        "val",
        int(cfg["image_size"]),
        patch_count,
        int(cfg["seed"]),
        float(cfg["salt_val_fraction"]),
        int(cfg["expected_salt_n_pepper_images"]),
        cfg["prompt_templates"],
        cfg["salt_label_variants"],
    )
    return imagenet_train, imagenet_val, salt_train, salt_val


def schedule_lr(optimizer: torch.optim.Optimizer, step: int, total_steps: int, cfg: Mapping[str, Any]) -> float:
    """Warm up, hold full LR, then cosine-decay to the floor."""
    base_lr = float(cfg["lr"])
    floor_lr = float(cfg.get("lr_floor", base_lr * 0.1))
    warmup = int(cfg.get("warmup_steps", 0))
    hold = int(cfg.get("lr_hold_steps", warmup))
    if hold < warmup:
        raise ValueError(f"lr_hold_steps ({hold}) must be >= warmup_steps ({warmup})")
    if hold >= total_steps:
        raise ValueError(f"lr_hold_steps ({hold}) must be < total steps ({total_steps})")
    if warmup > 0 and step <= warmup:
        lr = base_lr * float(step) / float(max(1, warmup))
    elif step <= hold:
        lr = base_lr
    else:
        progress = (step - hold) / float(max(1, total_steps - hold))
        progress = max(0.0, min(1.0, progress))
        lr = floor_lr + 0.5 * (base_lr - floor_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train(config_path: Path, validate_only: bool = False) -> None:
    config = load_json(config_path)
    cfg = resolve_router_config(config)
    validate_config_and_paths(config, cfg)

    out_dir = Path(cfg["output_dir"])
    start_checkpoint = Path(cfg["start_checkpoint"])
    prefix = str(cfg["output_prefix"])
    overwrite = bool(cfg.get("overwrite", False))

    def out(name: str) -> Path:
        return out_dir / f"{prefix}_{name}"

    outputs = [
        out("best.pt"),
        out("last.pt"),
        out("train.csv"),
        out("validation.csv"),
    ]
    if not overwrite and any(path.exists() for path in outputs):
        existing = [str(path) for path in outputs if path.exists()]
        raise RuntimeError(
            "Router utility A4 outputs already exist; set router_utility.overwrite=true to replace them:\n"
            + "\n".join(existing)
        )
    if overwrite:
        for path in outputs:
            if path.exists():
                path.unlink()

    print("[benchmark firewall] TRAINING SOURCES ARE EXACTLY: ImageNet + image_sets/salt_n_pepper")
    print("[benchmark firewall] no benchmark/eval dataset is imported, sampled, or backpropagated")
    print(f"[start checkpoint] {start_checkpoint}")
    print(f"[output folder]    {out_dir}")
    write_json(out("config_resolved.json"), cfg)

    if validate_only:
        # Cheap path audit stops before loading the large model.
        print("[validate-only] configuration and required paths are valid")
        return

    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device(str(config["global"].get("device", "cuda")) if torch.cuda.is_available() else "cpu")
    clip_module, model, start_compact, base_model_path = load_start_model(config, cfg, start_checkpoint, device)

    # Everything except the tiny candidate trust MLP is frozen.  auto_read_scale is
    # set to the forced-OPEN telemetry action size, but is NOT trained.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.read_implant.trust_router.parameters():
        parameter.requires_grad_(True)
    model.read_implant.auto_read_scale.data.fill_(float(cfg["telemetry_open_scale"]))
    model.read_implant.auto_read_scale.requires_grad_(False)
    model.eval()

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    expected_prefix = "read_implant.trust_router."
    if not trainable or any(not name.startswith(expected_prefix) for name in trainable):
        raise RuntimeError(f"Router-only trainable firewall failed: {trainable}")
    write_json(out("trainable.json"), trainable)
    print(f"[trainable] {len(trainable)} tensors, trust_router only")
    print(f"[auto_read_scale] fixed={float(model.read_implant.auto_read_scale.detach()):.6f}")

    imagenet_train, imagenet_val, salt_train, salt_val = make_sources(model=model, config=config, cfg=cfg)
    coverage = {
        "imagenet_train_groups": len(imagenet_train.group_ids),
        "imagenet_val_groups": len(imagenet_val.group_ids),
        "salt_train": salt_train.coverage(),
        "salt_val": salt_val.coverage(),
    }
    write_json(out("dataset_coverage.json"), coverage)
    print(f"[coverage] {coverage}")

    optimizer = torch.optim.AdamW(
        model.read_implant.trust_router.parameters(),
        lr=float(cfg["lr"]),
        betas=(float(cfg["beta1"]), float(cfg["beta2"])),
        eps=float(cfg["eps"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    steps = int(cfg["steps"])
    log_every = int(cfg["log_every"])
    eval_every = int(cfg["eval_every"])
    val_batches = int(cfg["val_batches"])
    rng = random.Random(seed + 17)
    capture = RouterLogitCapture(model)
    best_objective = float("inf")
    best_bce = float("inf")
    started = time.time()

    # Fixed held-out baseline is evaluated before any optimizer update.  This is
    # independent of target selection and gives an apples-to-apples z≈-4 anchor.
    baseline_metrics = evaluate(
        model=model,
        clip_module=clip_module,
        imagenet_source=imagenet_val,
        salt_source=salt_val,
        cfg=cfg,
        device=device,
        seed=int(cfg["validation_seed"]),
        batches=val_batches,
    )
    append_csv(out("validation.csv"), {"step": 0, **baseline_metrics})
    print(
        f"[val 0000] obj={baseline_metrics['validation_objective']:.4f} bce={baseline_metrics['bce']:.4f} "
        f"auc={baseline_metrics['auc']:.3f} imagenet_auc={baseline_metrics['imagenet_auc']:.3f} "
        f"pair_gap={baseline_metrics['imagenet_pair_gap_mean']:.3f} "
        f"pair_acc={baseline_metrics['imagenet_pair_accuracy']:.2f} "
        f"trust(open/close)={baseline_metrics['trust_open_mean']:.3f}/{baseline_metrics['trust_close_mean']:.3f}"
    )

    try:
        for step in range(1, steps + 1):
            lr = schedule_lr(optimizer, step, steps, cfg)
            batch, image_sources, image_packet_ids, construction_target, construction_roles = build_batch(
                imagenet_source=imagenet_train,
                salt_source=salt_train,
                rng=rng,
                imagenet_packets=int(cfg["imagenet_packets_per_batch"]),
                salt_images=int(cfg["salt_images_per_batch"]),
                prompt_templates=cfg["prompt_templates"],
                description_templates=cfg["description_templates"],
                device=device,
            )
            tokens = clip_module.tokenize(batch.captions, truncate=True).to(device)
            capture.value = None
            optimizer.zero_grad(set_to_none=True)
            with core.amp_context(device, str(cfg["amp_dtype"])):
                details = model.forward_modes(batch.images, tokens, return_details=True)
            if capture.value is None:
                raise RuntimeError("Failed to capture trust_router.fc3 logits")

            selection, rank_pairs, telemetry = teacher_from_forward(
                model=model,
                details=details,
                raw_router_z=capture.value,
                batch=batch,
                image_sources=image_sources,
                image_packet_ids=image_packet_ids,
                construction_target=construction_target,
                construction_roles=construction_roles,
                telemetry_open_scale=float(cfg["telemetry_open_scale"]),
                targets_per_class=int(cfg["targets_per_class"]),
                min_imagenet_open_fraction=float(cfg["min_imagenet_open_fraction"]),
                rng=rng,
            )
            loss_bce = F.binary_cross_entropy_with_logits(selection.z.float(), selection.target)
            rank_gap = rank_pairs.gap.float()
            loss_rank = F.softplus(float(cfg["rank_margin"]) - rank_gap).mean()
            loss = loss_bce + float(cfg["rank_weight"]) * loss_rank
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.read_implant.trust_router.parameters(), float(cfg["grad_clip"])
            ))
            optimizer.step()

            probs = selection.z.detach().sigmoid()
            open_mask = selection.target.eq(1)
            close_mask = selection.target.eq(0)
            current_loss = float(core.multi_positive_clip_loss(telemetry["current_logits"], batch.positive))
            closed_loss = float(core.multi_positive_clip_loss(telemetry["closed_logits"], batch.positive))
            full_open_loss = float(core.multi_positive_clip_loss(telemetry["open_logits"], batch.positive))
            row = {
                "step": step,
                "lr": lr,
                "loss_total": float(loss.detach()),
                "loss_router_bce": float(loss_bce.detach()),
                "loss_imagenet_rank": float(loss_rank.detach()),
                "rank_weight": float(cfg["rank_weight"]),
                "rank_margin": float(cfg["rank_margin"]),
                "imagenet_pair_gap_mean": float(rank_gap.detach().mean()),
                "imagenet_pair_gap_min": float(rank_gap.detach().min()),
                "imagenet_pair_accuracy": float((rank_gap.detach() > 0).float().mean()),
                "imagenet_rank_pairs": int(rank_gap.numel()),
                "auc": tensor_auc(selection.target, selection.z.detach()),
                "balanced_accuracy": balanced_accuracy(selection.target, probs),
                "targets": int(selection.target.numel()),
                "open_targets": int(open_mask.sum()),
                "close_targets": int(close_mask.sum()),
                "router_z_open_mean": float(selection.z.detach()[open_mask].mean()),
                "router_z_close_mean": float(selection.z.detach()[close_mask].mean()),
                "trust_open_mean": float(probs[open_mask].mean()),
                "trust_close_mean": float(probs[close_mask].mean()),
                "counterfactual_delta_loss_open_mean": float(selection.delta_loss[open_mask].mean()),
                "counterfactual_delta_loss_close_mean": float(selection.delta_loss[close_mask].mean()),
                "counterfactual_open_action_delta_mean": float(selection.open_delta.mean()),
                "counterfactual_open_help_fraction": float((selection.delta_loss[open_mask] < 0).float().mean()),
                "counterfactual_close_hurt_fraction": float((selection.delta_loss[close_mask] > 0).float().mean()),
                "grad_norm": grad_norm,
                "task_loss_current_no_backprop": current_loss,
                "task_loss_closed_no_backprop": closed_loss,
                "task_loss_fully_open_no_backprop": full_open_loss,
                "selected_open_sources": selection.stats["open_source_counts"],
                "selected_close_sources": selection.stats["close_source_counts"],
                "selected_open_roles": selection.stats["open_role_counts"],
                "selected_close_roles": selection.stats["close_role_counts"],
                "construction_open_known": selection.stats["construction_open_known"],
                "construction_close_known": selection.stats["construction_close_known"],
                "candidate_open_pool": selection.stats["candidate_open_pool"],
                "candidate_close_pool": selection.stats["candidate_close_pool"],
            }
            append_csv(out("train.csv"), row)

            if step == 1 or step % log_every == 0:
                print(
                    f"[step {step:04d}] total={row['loss_total']:.4f} bce={row['loss_router_bce']:.4f} "
                    f"rank={row['loss_imagenet_rank']:.4f} auc={row['auc']:.3f} "
                    f"gap={row['imagenet_pair_gap_mean']:.3f} pair_acc={row['imagenet_pair_accuracy']:.2f} "
                    f"z(open/close)={row['router_z_open_mean']:.3f}/{row['router_z_close_mean']:.3f} "
                    f"targets={row['open_targets']}+{row['close_targets']} lr={lr:.2e}"
                )

            if step % eval_every == 0 or step == steps:
                metrics = evaluate(
                    model=model,
                    clip_module=clip_module,
                    imagenet_source=imagenet_val,
                    salt_source=salt_val,
                    cfg=cfg,
                    device=device,
                    seed=int(cfg["validation_seed"]),
                    batches=val_batches,
                )
                metrics = {"step": step, **metrics}
                append_csv(out("validation.csv"), metrics)
                print(
                    f"[val {step:04d}] obj={metrics['validation_objective']:.4f} bce={metrics['bce']:.4f} "
                    f"auc={metrics['auc']:.3f} imagenet_auc={metrics['imagenet_auc']:.3f} "
                    f"pair_gap={metrics['imagenet_pair_gap_mean']:.3f} pair_acc={metrics['imagenet_pair_accuracy']:.2f} "
                    f"trust(open/close)={metrics['trust_open_mean']:.3f}/{metrics['trust_close_mean']:.3f} "
                    f"loss current/closed/open={metrics['loss_current']:.4f}/"
                    f"{metrics['loss_closed']:.4f}/{metrics['loss_fully_open']:.4f}"
                )
                save_compact(
                    out("last.pt"),
                    model=model,
                    start_compact=start_compact,
                    base_model_path=base_model_path,
                    step=step,
                    metrics=metrics,
                    resolved_config=cfg,
                )
                objective = float(metrics["validation_objective"])
                if math.isfinite(objective) and objective < best_objective:
                    best_objective = objective
                    best_bce = float(metrics["bce"])
                    save_compact(
                        out("best.pt"),
                        model=model,
                        start_compact=start_compact,
                        base_model_path=base_model_path,
                        step=step,
                        metrics=metrics,
                        resolved_config=cfg,
                    )
    finally:
        capture.close()

    elapsed = time.time() - started
    metadata = {
        "elapsed_seconds": elapsed,
        "steps_requested": steps,
        "skipped_steps": 0,
        "optimizer_steps": steps,
        "best_validation_objective": best_objective,
        "best_validation_bce_at_best_objective": best_bce,
        "validation_seed": int(cfg["validation_seed"]),
        "start_checkpoint": str(start_checkpoint),
        "base_model_path": str(base_model_path),
        "outputs": {
            "best": str(out("best.pt")),
            "last": str(out("last.pt")),
        },
    }
    write_json(out("run_metadata.json"), metadata)
    print(f"[done] router utility Experiment A4 -> {out_dir} (prefix={prefix})")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training_config.json")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train(args.config.expanduser().resolve(), validate_only=bool(args.validate_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
