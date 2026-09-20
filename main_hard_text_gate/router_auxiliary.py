#!/usr/bin/env python3
"""Shared OPEN/CLOSE router auxiliary for joint phase-1C and all-weights training.

This module deliberately keeps its supervision narrow and construction-known:

  ImageNet supportive overlay + ordinary correct candidate -> OPEN
  ImageNet adversarial overlay + ordinary written-word candidate -> CLOSE
  SALT/PEPPER generated image + matching shaker candidate -> OPEN

It never imports SCAM, RTA, MVT, eval_bench*, eval_benchmark*, or any benchmark
source.  It also never assigns semantic-ontology negatives to the SALT/PEPPER
images.  Prompt surfaces are varied, but construction is the answer key.

The auxiliary loss is meant to preserve the A4 router policy while ordinary 1C
and all-weights task training continues:

  lambda_bce  * BCEWithLogits(router_z, OPEN/CLOSE)
  lambda_rank * softplus(margin - (z_open_imagenet - z_close_imagenet))

CandidateTrustRouter detaches every upstream feature before its MLP, so these
auxiliary gradients affect the router only even while the full model is
trainable.  Separate gradient-contribution diagnostics can verify how the normal
task objective and the two auxiliary terms vote on the router.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

SALT_PEPPER_RE = re.compile(r"^(salt|pepper)_([a-z]+)_(\d+)$", re.IGNORECASE)
SALT_PEPPER_TYPES = {
    "chess", "colt", "frog", "grenade", "lighthouse",
    "matryoshka", "mushroom", "nuclear", "prescription", "tiki",
}

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

DEFAULT_SALT_LABEL_VARIANTS = (
    "{word} shaker",
    "ceramic {word} shaker",
    "novelty {word} shaker",
    "{word} dispenser",
)


def stable_int(*parts: Any) -> int:
    text = "\0".join(str(x) for x in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def format_label_prompt(template: str, label: str) -> str:
    article = "an" if label[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    text = str(template).replace("a {label}", f"{article} {{label}}")
    return text.format(label=label)


def tensor_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    y = labels.detach().cpu().numpy().astype(np.int64)
    s = scores.detach().cpu().numpy().astype(np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
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


@dataclass(frozen=True)
class SaltPepperItem:
    path: Path
    word: str
    object_type: str
    index: int


class SaltPepperRouteSource:
    def __init__(
        self,
        *,
        core: Any,
        root: Path,
        split: str,
        image_size: int,
        patch_count: int,
        seed: int,
        val_fraction: float,
        expected_total: int,
        prompt_templates: Sequence[str] = DEFAULT_PROMPT_TEMPLATES,
        label_variants: Sequence[str] = DEFAULT_SALT_LABEL_VARIANTS,
    ):
        self.core = core
        self.root = Path(root)
        self.split = str(split)
        self.image_size = int(image_size)
        self.patch_count = int(patch_count)
        self.prompt_templates = tuple(str(x) for x in prompt_templates)
        self.label_variants = tuple(str(x) for x in label_variants)
        if not self.prompt_templates or not self.label_variants:
            raise ValueError("router auxiliary prompt/label variants cannot be empty")

        all_items: List[SaltPepperItem] = []
        if self.root.is_dir():
            for path in sorted(self.root.rglob("*.png")):
                match = SALT_PEPPER_RE.match(path.stem)
                if not match:
                    continue
                word = match.group(1).lower()
                object_type = match.group(2).lower()
                index = int(match.group(3))
                if object_type not in SALT_PEPPER_TYPES:
                    raise ValueError(f"Unknown salt_n_pepper type in {path.name}: {object_type}")
                all_items.append(SaltPepperItem(path, word, object_type, index))
        if int(expected_total) > 0 and len(all_items) != int(expected_total):
            raise RuntimeError(
                f"Expected exactly {expected_total} salt+pepper images in {self.root}, found {len(all_items)}"
            )
        if not all_items:
            raise RuntimeError(f"No salt_n_pepper images found in {self.root}")

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
        self.items = train_items if self.split == "train" else val_items
        self.total_items = all_items
        if not self.items:
            raise RuntimeError(f"salt_n_pepper split {self.split!r} is empty")

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

    def sample(self, rng: random.Random):
        item = rng.choice(self.items)
        with Image.open(item.path) as image:
            tensor, _ = self.core.image_and_mask_transform(
                image.convert("RGB"), None, self.image_size, flip=False
            )
        label_template = rng.choice(self.label_variants)
        label = label_template.format(word=item.word, type=item.object_type)
        prompt = format_label_prompt(rng.choice(self.prompt_templates), label)
        return self.core.Packet(
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


def infer_primary_label(packet: Any) -> str:
    value = str(packet.metadata.get("primary_label") or "").strip()
    if value:
        return value
    caption = str(packet.captions[0])
    prefix = "a photo of a "
    if caption.lower().startswith(prefix):
        return caption[len(prefix):]
    return caption


def vary_imagenet_packet_prompts(
    *, core: Any, packet: Any, rng: random.Random,
    prompt_templates: Sequence[str], description_templates: Sequence[str],
):
    primary_label = infer_primary_label(packet)
    support_text = str(packet.metadata.get("support_text") or "")
    adversarial_text = str(packet.metadata.get("adversarial_text") or "")
    captions = list(packet.captions)
    primary_prompt = format_label_prompt(rng.choice(tuple(prompt_templates)), primary_label)
    adversarial_prompt = format_label_prompt(rng.choice(tuple(prompt_templates)), adversarial_text)
    captions[0] = primary_prompt
    captions[1] = f"<notext> {primary_prompt}"
    captions[7] = adversarial_prompt
    if support_text:
        captions[5] = rng.choice(tuple(description_templates)).format(
            object_label=primary_label, word=support_text
        )
    if adversarial_text:
        captions[6] = rng.choice(tuple(description_templates)).format(
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


def build_construction_route_targets(packets: Sequence[Any], packed: Any, device: torch.device):
    target = torch.full(packed.positive.shape, float("nan"), dtype=torch.float32)
    roles: Dict[Tuple[int, int], str] = {}
    caption_to_global = {caption: index for index, caption in enumerate(packed.captions)}
    image_offset = 0

    def set_target(local_image: int, packet: Any, local_caption: int, value: float, role: str) -> None:
        global_image = image_offset + int(local_image)
        caption = str(packet.captions[int(local_caption)])
        if caption not in caption_to_global:
            raise RuntimeError(f"Packed caption disappeared: {caption!r}")
        global_caption = int(caption_to_global[caption])
        previous = target[global_image, global_caption]
        if torch.isfinite(previous) and float(previous) != float(value):
            raise RuntimeError(
                f"Conflicting route labels at image={global_image} caption={caption!r}: {float(previous)} vs {value}"
            )
        target[global_image, global_caption] = float(value)
        roles[(global_image, global_caption)] = str(role)

    for packet in packets:
        source = str(packet.metadata.get("source", "unknown"))
        if source == "imagenet":
            if len(packet.images) < 3 or len(packet.captions) < 8:
                raise RuntimeError("Unexpected ImageNet packet layout for router auxiliary")
            set_target(1, packet, 0, 1.0, "imagenet_support_primary_open")
            set_target(2, packet, 7, 0.0, "imagenet_adversarial_written_close")
        elif source == "salt_n_pepper":
            if len(packet.images) != 1 or len(packet.captions) != 1:
                raise RuntimeError("Unexpected salt_n_pepper packet layout for router auxiliary")
            set_target(0, packet, 0, 1.0, "salt_pepper_matching_shaker_open")
        else:
            raise RuntimeError(
                f"Router auxiliary training-source firewall: unexpected packet source {source!r}"
            )
        image_offset += len(packet.images)

    if image_offset != int(packed.images.shape[0]):
        raise RuntimeError(f"Route target image alignment failed: {image_offset} != {int(packed.images.shape[0])}")

    known = torch.isfinite(target)
    open_known = known & target.eq(1.0)
    close_known = known & target.eq(0.0)
    if bool((open_known & ~packed.positive).any()):
        bad = torch.nonzero(open_known & ~packed.positive, as_tuple=False)[0].tolist()
        raise RuntimeError(f"Construction OPEN is not a packet positive at cell {bad}")
    if bool((close_known & packed.positive).any()):
        bad = torch.nonzero(close_known & packed.positive, as_tuple=False)[0].tolist()
        raise RuntimeError(f"Construction CLOSE is unexpectedly a packet positive at cell {bad}")
    return target.to(device), roles


def build_aux_batch(
    *, core: Any, imagenet_source: Any, salt_source: SaltPepperRouteSource,
    rng: random.Random, imagenet_packets: int, salt_images: int,
    prompt_templates: Sequence[str], description_templates: Sequence[str],
    device: torch.device,
):
    packets: List[Any] = []
    for _ in range(int(imagenet_packets)):
        packet = imagenet_source.sample(rng)
        packet = vary_imagenet_packet_prompts(
            core=core, packet=packet, rng=rng,
            prompt_templates=prompt_templates,
            description_templates=description_templates,
        )
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
    construction_target, roles = build_construction_route_targets(packets, packed_cpu, device)
    batch = core.batch_to_device(packed_cpu, device)
    return batch, image_sources, image_packet_ids, construction_target, roles


class RouterFeatureCapture:
    """Capture the exact 16-D tensor entering trust_router.fc1."""
    def __init__(self, model: torch.nn.Module):
        self.value: Optional[torch.Tensor] = None
        self.handle = model.read_implant.trust_router.fc1.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self.value = inputs[0].detach()

    def close(self) -> None:
        self.handle.remove()


def router_raw_logits_from_features(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Run only the tiny router MLP with gradients; upstream features stay detached.

    The normal CandidateTrustRouter forward is an fp32 custom island.  This manual
    replay preserves that policy even when the caller is inside CUDA autocast.
    """
    router = model.read_implant.trust_router
    device_type = features.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        x = features.float()
        hidden = F.gelu(router.fc1(x))
        hidden = F.gelu(router.fc2(hidden))
        return router.fc3(hidden).squeeze(-1)


@dataclass
class AuxSelection:
    z: torch.Tensor
    target: torch.Tensor
    source_names: List[str]
    role_names: List[str]


@dataclass
class RankPairs:
    z_open: torch.Tensor
    z_close: torch.Tensor

    @property
    def gap(self) -> torch.Tensor:
        return self.z_open - self.z_close


def _roles_to_auto_indices(
    *, roles_full: Mapping[Tuple[int, int], str], any_mask: torch.Tensor,
) -> Dict[Tuple[int, int], str]:
    full_any_cols = torch.nonzero(any_mask, as_tuple=False).flatten().detach().cpu().tolist()
    full_to_auto = {int(full): auto for auto, full in enumerate(full_any_cols)}
    out: Dict[Tuple[int, int], str] = {}
    for (image_index, full_caption), role in roles_full.items():
        if int(full_caption) not in full_to_auto:
            raise RuntimeError(f"Construction role {role!r} landed on a non-<any> caption")
        out[(int(image_index), int(full_to_auto[int(full_caption)]))] = str(role)
    return out


def select_balanced_targets(
    *, z_auto: torch.Tensor, target_auto: torch.Tensor,
    roles_auto: Mapping[Tuple[int, int], str], image_sources: Sequence[str],
    targets_per_class: int, min_imagenet_open_fraction: float, rng: random.Random,
) -> AuxSelection:
    known = torch.isfinite(target_auto)
    open_flat = torch.nonzero((known & target_auto.eq(1.0)).reshape(-1), as_tuple=False).flatten()
    close_flat = torch.nonzero((known & target_auto.eq(0.0)).reshape(-1), as_tuple=False).flatten()
    width = int(z_auto.shape[1])
    k = int(targets_per_class)
    if int(open_flat.numel()) < k or int(close_flat.numel()) < k:
        raise RuntimeError(
            f"Router auxiliary requires {k} OPEN + {k} CLOSE every step; "
            f"available OPEN={int(open_flat.numel())}, CLOSE={int(close_flat.numel())}"
        )
    open_values = [int(x) for x in open_flat.detach().cpu().tolist()]
    close_values = [int(x) for x in close_flat.detach().cpu().tolist()]
    open_imagenet = [idx for idx in open_values if str(image_sources[idx // width]) == "imagenet"]
    required_imagenet = min(k, int(math.ceil(k * float(min_imagenet_open_fraction))))
    if len(open_imagenet) < required_imagenet:
        raise RuntimeError(
            f"Router auxiliary domain guard needs {required_imagenet} ImageNet OPENs, found {len(open_imagenet)}"
        )
    rng.shuffle(open_imagenet)
    chosen_open = open_imagenet[:required_imagenet]
    chosen_set = set(chosen_open)
    remaining_open = [idx for idx in open_values if idx not in chosen_set]
    rng.shuffle(remaining_open)
    chosen_open.extend(remaining_open[: k - len(chosen_open)])
    rng.shuffle(close_values)
    chosen_close = close_values[:k]
    if len(chosen_open) != k or len(chosen_close) != k:
        raise RuntimeError("Failed to construct exact balanced router auxiliary batch")
    selected_values = chosen_open + chosen_close
    selected = torch.tensor(selected_values, dtype=torch.long, device=z_auto.device)
    target = torch.cat([
        torch.ones(k, device=z_auto.device, dtype=torch.float32),
        torch.zeros(k, device=z_auto.device, dtype=torch.float32),
    ])
    sources: List[str] = []
    roles: List[str] = []
    for flat_index in selected_values:
        image_index = int(flat_index) // width
        auto_index = int(flat_index) % width
        sources.append(str(image_sources[image_index]))
        roles.append(str(roles_auto.get((image_index, auto_index), "unknown")))
    return AuxSelection(z=z_auto.reshape(-1)[selected], target=target, source_names=sources, role_names=roles)


def build_rank_pairs(
    *, z_auto: torch.Tensor, roles_auto: Mapping[Tuple[int, int], str],
    image_packet_ids: Sequence[str],
) -> RankPairs:
    grouped: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    for (image_index, auto_index), role in roles_auto.items():
        packet_id = str(image_packet_ids[image_index])
        if not packet_id.startswith("imagenet:"):
            continue
        if role == "imagenet_support_primary_open":
            key = "open"
        elif role == "imagenet_adversarial_written_close":
            key = "close"
        else:
            continue
        grouped[packet_id][key] = z_auto[image_index, auto_index]
    opens: List[torch.Tensor] = []
    closes: List[torch.Tensor] = []
    for packet_id in sorted(grouped):
        pair = grouped[packet_id]
        if set(pair) != {"open", "close"}:
            raise RuntimeError(f"Incomplete ImageNet router rank pair for {packet_id}: {sorted(pair)}")
        opens.append(pair["open"])
        closes.append(pair["close"])
    if not opens:
        raise RuntimeError("No ImageNet router rank pairs were constructed")
    return RankPairs(z_open=torch.stack(opens), z_close=torch.stack(closes))


def compute_auxiliary_loss(
    *, core: Any, model: torch.nn.Module, clip_module: Any,
    imagenet_source: Any, salt_source: SaltPepperRouteSource,
    rng: random.Random, device: torch.device, args: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """One differentiable auxiliary forward; returns unweighted BCE and rank losses."""
    batch, image_sources, image_packet_ids, construction_target, roles_full = build_aux_batch(
        core=core,
        imagenet_source=imagenet_source,
        salt_source=salt_source,
        rng=rng,
        imagenet_packets=int(args.router_aux_imagenet_packets_per_batch),
        salt_images=int(args.router_aux_salt_images_per_batch),
        prompt_templates=args.router_aux_prompt_templates,
        description_templates=args.router_aux_description_templates,
        device=device,
    )
    tokens = clip_module.tokenize(batch.captions, truncate=True).to(device)
    capture = RouterFeatureCapture(model)
    try:
        capture.value = None
        # Full auxiliary feature extraction is inference-only. This keeps the extra
        # construction batch cheap in memory even during all-weights training.
        # CandidateTrustRouter itself already detaches these features semantically;
        # we replay only its tiny MLP below with gradients.
        with torch.no_grad():
            details = model.forward_modes(batch.images, tokens, return_details=True)
        if capture.value is None:
            raise RuntimeError("Failed to capture CandidateTrustRouter.fc1 features")
        raw_z = router_raw_logits_from_features(model, capture.value)
    finally:
        capture.close()

    modes = details["mode_ids"]
    read_candidate_mask = ~modes.eq(2)
    selected_modes = modes[read_candidate_mask]
    auto_within_read = selected_modes.eq(0)
    any_mask = modes.eq(0)
    z_auto = raw_z[:, auto_within_read]
    if z_auto.shape[1] != int(any_mask.sum()):
        raise RuntimeError("Router auxiliary candidate alignment failed")
    outside_any = torch.isfinite(construction_target) & (~any_mask[None, :])
    if bool(outside_any.any()):
        raise RuntimeError("Router auxiliary construction target landed outside <any>")
    roles_auto = _roles_to_auto_indices(roles_full=roles_full, any_mask=any_mask)
    selection = select_balanced_targets(
        z_auto=z_auto,
        target_auto=construction_target[:, any_mask],
        roles_auto=roles_auto,
        image_sources=image_sources,
        targets_per_class=int(args.router_aux_targets_per_class),
        min_imagenet_open_fraction=float(args.router_aux_min_imagenet_open_fraction),
        rng=rng,
    )
    pairs = build_rank_pairs(
        z_auto=z_auto, roles_auto=roles_auto, image_packet_ids=image_packet_ids
    )

    loss_bce = F.binary_cross_entropy_with_logits(selection.z, selection.target)
    gaps = pairs.gap
    loss_rank = F.softplus(float(args.router_aux_rank_margin) - gaps).mean()

    probs = selection.z.sigmoid()
    open_mask = selection.target.eq(1.0)
    close_mask = selection.target.eq(0.0)
    source_arr = np.asarray(selection.source_names, dtype=object)
    labels_np = selection.target.detach().cpu().numpy()
    imagenet_idx = np.nonzero(source_arr == "imagenet")[0]
    salt_idx = np.nonzero(source_arr == "salt_n_pepper")[0]

    def mean_for(indices: np.ndarray, label: Optional[int] = None, sigmoid: bool = True) -> float:
        if indices.size == 0:
            return float("nan")
        idx = torch.tensor(indices.tolist(), device=selection.z.device, dtype=torch.long)
        if label is not None:
            yy = selection.target[idx]
            idx = idx[yy.eq(float(label))]
            if idx.numel() == 0:
                return float("nan")
        values = selection.z[idx].sigmoid() if sigmoid else selection.z[idx]
        return float(values.detach().mean())

    imagenet_auc = float("nan")
    if imagenet_idx.size:
        ii = torch.tensor(imagenet_idx.tolist(), device=selection.z.device, dtype=torch.long)
        yy, zz = selection.target[ii], selection.z[ii]
        if bool(yy.eq(1).any()) and bool(yy.eq(0).any()):
            imagenet_auc = tensor_auc(yy, zz)

    metrics = {
        "router_aux_bce": float(loss_bce.detach()),
        "router_aux_rank": float(loss_rank.detach()),
        "router_aux_auc": tensor_auc(selection.target, selection.z),
        "router_aux_balanced_accuracy": balanced_accuracy(selection.target, probs),
        "router_aux_z_open": float(selection.z[open_mask].detach().mean()),
        "router_aux_z_close": float(selection.z[close_mask].detach().mean()),
        "router_aux_trust_open": float(probs[open_mask].detach().mean()),
        "router_aux_trust_close": float(probs[close_mask].detach().mean()),
        "router_aux_imagenet_auc": imagenet_auc,
        "router_aux_imagenet_trust_open": mean_for(imagenet_idx, label=1),
        "router_aux_imagenet_trust_close": mean_for(imagenet_idx, label=0),
        "router_aux_salt_trust_open": mean_for(salt_idx, label=1),
        "router_aux_pair_gap_mean": float(gaps.detach().mean()),
        "router_aux_pair_gap_min": float(gaps.detach().min()),
        "router_aux_pair_accuracy": float((gaps.detach() > 0).float().mean()),
    }
    return loss_bce, loss_rank, metrics


@torch.no_grad()
def evaluate_auxiliary(
    *, core: Any, model: torch.nn.Module, clip_module: Any,
    imagenet_source: Any, salt_source: SaltPepperRouteSource,
    device: torch.device, args: Any, seed: int, batches: int,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    rng = random.Random(int(seed))
    totals: Dict[str, float] = defaultdict(float)
    count = 0
    try:
        for _ in range(int(batches)):
            bce, rank, metrics = compute_auxiliary_loss(
                core=core, model=model, clip_module=clip_module,
                imagenet_source=imagenet_source, salt_source=salt_source,
                rng=rng, device=device, args=args,
            )
            metrics = dict(metrics)
            metrics["router_aux_bce"] = float(bce)
            metrics["router_aux_rank"] = float(rank)
            for key, value in metrics.items():
                if math.isfinite(float(value)):
                    totals[key] += float(value)
            count += 1
    finally:
        model.train(was_training)
    out = {key: value / max(1, count) for key, value in totals.items()}
    out["router_aux_validation_batches"] = float(count)
    return out


def make_sources(*, core: Any, model: torch.nn.Module, args: Any):
    """Create independent aux-only ImageNet and SALT/PEPPER train/val sources."""
    if not bool(args.router_aux_enabled):
        return None
    root = Path(args.router_aux_salt_n_pepper_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing router auxiliary SALT/PEPPER root: {root}")
    patch_count = int(model.visual.positional_embedding.shape[0] - 1)
    imagenet_train = core.ImageNetPacketSource(
        Path(args.imagenet_text_root), Path(args.imagenet_handwriting_root), "train",
        int(args.image_size), patch_count, float(args.router_aux_imagenet_handwriting_probability),
        0.0, (),
    )
    imagenet_val = core.ImageNetPacketSource(
        Path(args.imagenet_text_root), Path(args.imagenet_handwriting_root), "val",
        int(args.image_size), patch_count, float(args.router_aux_imagenet_handwriting_probability),
        0.0, (),
    )
    salt_train = SaltPepperRouteSource(
        core=core, root=root, split="train", image_size=int(args.image_size),
        patch_count=patch_count, seed=int(args.router_aux_seed),
        val_fraction=float(args.router_aux_salt_val_fraction),
        expected_total=int(args.router_aux_expected_salt_n_pepper_images),
        prompt_templates=args.router_aux_prompt_templates,
        label_variants=args.router_aux_salt_label_variants,
    )
    salt_val = SaltPepperRouteSource(
        core=core, root=root, split="val", image_size=int(args.image_size),
        patch_count=patch_count, seed=int(args.router_aux_seed),
        val_fraction=float(args.router_aux_salt_val_fraction),
        expected_total=int(args.router_aux_expected_salt_n_pepper_images),
        prompt_templates=args.router_aux_prompt_templates,
        label_variants=args.router_aux_salt_label_variants,
    )
    return {
        "imagenet_train": imagenet_train,
        "imagenet_val": imagenet_val,
        "salt_train": salt_train,
        "salt_val": salt_val,
        "train_rng": random.Random(int(args.router_aux_seed) + 17),
        "coverage": {
            "training_sources": ["imagenet", "salt_n_pepper"],
            "benchmark_training": False,
            "imagenet_train_groups": len(imagenet_train.group_ids),
            "imagenet_val_groups": len(imagenet_val.group_ids),
            "salt_train": salt_train.coverage(),
            "salt_val": salt_val.coverage(),
        },
    }


def load_router_init_checkpoint(model: torch.nn.Module, path: Optional[Path]) -> Dict[str, Any]:
    """Overlay only CandidateTrustRouter weights from the A4 router-only checkpoint."""
    if path is None:
        return {"loaded": False}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing router auxiliary init checkpoint: {path}")
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise RuntimeError(f"Unsupported router init checkpoint: {path}")
    implant = loaded.get("implant_state_dict")
    if not isinstance(implant, Mapping):
        raise RuntimeError(f"Router init checkpoint lacks implant_state_dict: {path}")
    prefix = "trust_router."
    state = {
        str(key)[len(prefix):]: value
        for key, value in implant.items()
        if str(key).startswith(prefix)
    }
    expected = model.read_implant.trust_router.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise RuntimeError(
            f"Router init state mismatch: missing={missing}, extra={extra}, path={path}"
        )
    model.read_implant.trust_router.load_state_dict(state, strict=True)
    with torch.no_grad():
        z_bias = float(model.read_implant.trust_router.fc3.bias.detach().mean())
        w_norm = float(model.read_implant.trust_router.fc3.weight.detach().float().norm())
    return {
        "loaded": True,
        "path": str(path),
        "fc3_bias": z_bias,
        "fc3_weight_norm": w_norm,
        "checkpoint_metrics": dict(loaded.get("metrics", {})) if isinstance(loaded.get("metrics"), Mapping) else {},
    }


def _flatten_grads(grads: Sequence[Optional[torch.Tensor]], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            chunks.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float32))
        else:
            chunks.append(grad.detach().float().reshape(-1))
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks)


def gradient_contribution_metrics(
    *, model: torch.nn.Module, task_loss: torch.Tensor,
    weighted_bce_loss: torch.Tensor, weighted_rank_loss: torch.Tensor,
) -> Dict[str, float]:
    """Exact router-only gradient votes before the ordinary backward/clip/step."""
    params = [p for p in model.read_implant.trust_router.parameters() if p.requires_grad]
    if not params:
        return {}
    g_task = torch.autograd.grad(task_loss, params, retain_graph=True, allow_unused=True)
    g_bce = torch.autograd.grad(weighted_bce_loss, params, retain_graph=True, allow_unused=True)
    g_rank = torch.autograd.grad(weighted_rank_loss, params, retain_graph=True, allow_unused=True)
    vt = _flatten_grads(g_task, params)
    vb = _flatten_grads(g_bce, params)
    vr = _flatten_grads(g_rank, params)
    va = vb + vr
    vn = vt + va

    def norm(v: torch.Tensor) -> float:
        return float(v.norm()) if v.numel() else 0.0

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = float(a.norm() * b.norm()) if a.numel() and b.numel() else 0.0
        return float(torch.dot(a, b) / denom) if denom > 0 else float("nan")

    return {
        "router_grad_task_norm": norm(vt),
        "router_grad_aux_bce_norm": norm(vb),
        "router_grad_aux_rank_norm": norm(vr),
        "router_grad_aux_total_norm": norm(va),
        "router_grad_net_norm": norm(vn),
        "router_grad_task_aux_cos": cosine(vt, va),
        "router_grad_bce_rank_cos": cosine(vb, vr),
        "router_grad_task_aux_dot": float(torch.dot(vt, va)) if vt.numel() else 0.0,
    }


def router_grad_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
    for p in model.read_implant.trust_router.parameters():
        if p.grad is not None:
            total = total + p.grad.detach().float().square().sum()
    return float(total.sqrt())
