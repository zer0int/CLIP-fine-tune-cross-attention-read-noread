"""In-memory SCAM/RTA typographic validation for the AnyText trainer.

SCAM and RTA are evaluation-only and are loaded directly from their Hugging Face
dataset repositories.  Scores go through ``model.forward_modes`` under
``torch.inference_mode()``.  They are monitoring metrics only and must never
participate in loss or checkpoint selection.

ObjectNet-MVT remains an optional local clean-image monitor because its images
are not redistributed with the training package.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image


@dataclass(frozen=True)
class TypoItem:
    image: Any
    variant: str
    object_label: str
    attack_word: str
    family: str


@dataclass(frozen=True)
class MvtItem:
    path: Path
    label: str


def _load_tensor(source: Any, image_size: int, transform_fn) -> torch.Tensor:
    if isinstance(source, Image.Image):
        image = source
        tensor, _ = transform_fn(image.convert("RGB"), None, image_size, flip=False)
        return tensor
    if isinstance(source, Mapping):
        payload = source.get("bytes")
        cached_path = source.get("path")
        if payload is not None:
            with Image.open(io.BytesIO(bytes(payload))) as image:
                tensor, _ = transform_fn(image.convert("RGB"), None, image_size, flip=False)
            return tensor
        if cached_path:
            # Hugging Face datasets may expose archive-backed images as fsspec
            # virtual URIs, e.g.
            #   zip://images/train/foo.jpg::somepath\...\images.zip
            # These are *not* filesystem paths.  Passing one through pathlib on
            # Windows rewrites '/' to '\\' and turns it into a bogus local
            # relative path.  Keep the original string intact.
            source = str(cached_path)
        else:
            raise RuntimeError("HF image record contains neither bytes nor a cache path")

    source_text = str(source)
    if "://" in source_text:
        try:
            import fsspec
        except Exception as exc:
            raise RuntimeError(
                "Opening archive-backed Hugging Face benchmark images requires fsspec. "
                "It is normally installed with datasets."
            ) from exc
        try:
            with fsspec.open(source_text, mode="rb") as handle:
                with Image.open(handle) as image:
                    tensor, _ = transform_fn(
                        image.convert("RGB"), None, image_size, flip=False
                    )
            return tensor
        except Exception as exc:
            raise RuntimeError(
                f"Could not open Hugging Face virtual image reference: {source_text!r}"
            ) from exc

    path = Path(source_text)
    with Image.open(path) as image:
        tensor, _ = transform_fn(image.convert("RGB"), None, image_size, flip=False)
    return tensor


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _component_metrics(model) -> Dict[str, float]:
    implant = model.read_implant
    out: Dict[str, float] = {}
    groups = (
        ("read", implant.tap_block_list(), implant.read_tap_logits),
        ("content", implant.tap_block_list(), implant.content_tap_logits),
        ("ortho", implant.ortho_block_list(), implant.ortho_tap_logits),
        ("source", implant.source_block_list(), implant.source_tap_logits),
    )
    for name, blocks, logits in groups:
        weights = logits.detach().float().softmax(dim=0).cpu().tolist()
        for block, weight in zip(blocks, weights):
            out[f"model/{name}_tap_B{block}"] = float(weight)
    out.update({
        "model/auto_read_scale": float(implant.auto_read_scale.detach().float().cpu()),
        "model/null_abstain_weight": float(implant.null_abstain_weight.detach().float().cpu()),
        "model/glyph_bias_beta": float(implant.glyph_bias_beta.detach().float().cpu()),
        "model/register_gate": float(implant.read_bridge.register_gate.detach().float().cpu()),
    })
    return out


def _print_important(out: Mapping[str, float]) -> None:
    parts: List[str] = []
    for family, clean_variants in (("scam", {"NoSCAM"}), ("rta", {"NoRTA"})):
        attacked = sorted({
            key.split("/")[1]
            for key in out
            if key.startswith(f"{family}/") and key.endswith("/any_acc")
            and key.split("/")[1] not in clean_variants
        })
        if not attacked:
            continue
        def mean(suffix: str) -> float:
            return sum(out[f"{family}/{variant}/{suffix}"] for variant in attacked) / len(attacked)
        parts.append(
            f"{family.upper()} attacked N/A/R+null="
            f"{mean('notext_acc'):.4f}/{mean('any_acc'):.4f}/{mean('read_with_null_acc'):.4f} "
            f"null_reject={mean('null_reject_acc'):.4f}"
        )
    if "mvt/any_acc" in out:
        parts.append(
            f"MVT N/A/null={out.get('mvt/notext_acc', float('nan')):.4f}/"
            f"{out.get('mvt/any_acc', float('nan')):.4f}/"
            f"{out.get('mvt/null_accept', float('nan')):.4f}"
        )
    if parts:
        print("[benchmark important] " + " | ".join(parts))

    tap_text: List[str] = []
    for name in ("read", "content", "ortho", "source"):
        tap_keys = sorted(key for key in out if key.startswith(f"model/{name}_tap_"))
        taps = "/".join(f"{key.rsplit('_', 1)[-1]}:{out[key]:.3f}" for key in tap_keys)
        tap_text.append(f"{name}={taps}")
    print(
        "[benchmark heads] "
        + " ".join(tap_text)
        + " "
        f"auto={out.get('model/auto_read_scale', float('nan')):.4f} "
        f"null={out.get('model/null_abstain_weight', float('nan')):.4f} "
        f"glyph_beta={out.get('model/glyph_bias_beta', float('nan')):.4f} "
        f"reg={out.get('model/register_gate', float('nan')):.4f}"
    )


def load_typo_items(
    repo_id: str,
    revision: Optional[str],
    family: str,
    max_items: int = 0,
) -> List[TypoItem]:
    try:
        from datasets import Image as HFImage, load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face datasets is required for SCAM/RTA validation. "
            "Install it with: pip install -U datasets"
        ) from exc

    dataset = load_dataset(repo_id, split="train", revision=revision)
    dataset = dataset.cast_column("image", HFImage(decode=False))
    items: List[TypoItem] = []
    per_variant: Dict[str, int] = {}
    expected = {"SCAM", "SynthSCAM", "NoSCAM"} if family == "SCAM" else {"RTA", "SynthRTA", "NoRTA"}
    for row in dataset:
        raw_id = str(row.get("id") or "").strip()
        variant = str(row.get("type") or "").strip()
        if variant not in expected:
            variant = next((name for name in expected if raw_id.startswith(name)), "")
        obj = str(row.get("object_label") or "").strip()
        atk = str(row.get("attack_word") or "").strip()
        image_value = row.get("image")
        if variant not in expected or not obj or not atk or image_value is None:
            continue
        if max_items > 0 and per_variant.get(variant, 0) >= max_items:
            continue
        per_variant[variant] = per_variant.get(variant, 0) + 1
        items.append(TypoItem(image_value, variant, obj, atk, family))
    missing = sorted(expected.difference(per_variant))
    if missing:
        raise RuntimeError(
            f"{family} benchmark repository {repo_id!r} is missing expected variants: {missing}"
        )
    return items


@torch.inference_mode()
def run_typo(
    model,
    clip_module,
    items: Sequence[TypoItem],
    device: torch.device,
    image_size: int,
    batch_size: int,
    transform_fn,
) -> Dict[str, float]:
    if not items:
        return {}
    totals: Dict[str, int] = {}
    hits: Dict[Tuple[str, str], int] = {}
    gate_sums: Dict[Tuple[str, str], float] = {}
    trust_sums: Dict[Tuple[str, str], float] = {}
    source_sums: Dict[str, float] = {}
    glyph_sums: Dict[Tuple[str, str], float] = {}

    for batch in _chunks(list(items), batch_size):
        images = torch.stack([_load_tensor(x.image, image_size, transform_fn) for x in batch]).to(device)
        m = len(batch)
        prompts: List[str] = []
        for prefix in ("<notext> ", ""):
            prompts.extend(prefix + f"a photo of a {x.object_label}" for x in batch)
            prompts.extend(prefix + f"a photo of a {x.attack_word}" for x in batch)
        prompts.extend(f"<text> {x.object_label}" for x in batch)
        prompts.extend(f"<text> {x.attack_word}" for x in batch)
        prompts.append("<text> <null>")
        tokens = clip_module.tokenize(prompts, truncate=True).to(device)
        details = model.forward_modes(images, tokens, return_details=True)
        logits = details["logits_per_image"].float()
        gates = details["route_gate"].float()
        trusts = details["trust_gate"].float()
        source_gate = details["source_gate"].float()
        glyph = details["glyph_probs"].float()
        row = torch.arange(m, device=device)
        columns = {
            "notext_object": row,
            "notext_attack": m + row,
            "any_object": 2 * m + row,
            "any_attack": 3 * m + row,
            "text_object": 4 * m + row,
            "text_attack": 5 * m + row,
        }
        values = {name: logits[row, col] for name, col in columns.items()}
        values["text_null"] = logits[:, 6 * m]
        gate_obj = gates[row, columns["any_object"]]
        gate_atk = gates[row, columns["any_attack"]]
        trust_obj = trusts[row, columns["any_object"]]
        trust_atk = trusts[row, columns["any_attack"]]
        glyph_mean = glyph.mean(dim=1)
        glyph_max = glyph.amax(dim=1)
        for i, item in enumerate(batch):
            variant = item.variant
            totals[variant] = totals.get(variant, 0) + 1
            attacked = variant not in {"NoSCAM", "NoRTA"}
            outcomes = {
                "notext_acc": values["notext_object"][i] > values["notext_attack"][i],
                "any_acc": values["any_object"][i] > values["any_attack"][i],
                "text_acc": values["text_attack"][i] > values["text_object"][i],
                "read_with_null_acc": (
                    values["text_attack"][i]
                    > torch.maximum(values["text_object"][i], values["text_null"][i])
                    if attacked else False
                ),
                "null_reject_acc": (
                    values["text_attack"][i] > values["text_null"][i]
                    if attacked else False
                ),
                "null_accept_acc": (
                    values["text_null"][i]
                    > torch.maximum(values["text_object"][i], values["text_attack"][i])
                    if not attacked else False
                ),
            }
            for key, result in outcomes.items():
                hits[(variant, key)] = hits.get((variant, key), 0) + int(bool(result))
            gate_sums[(variant, "object")] = gate_sums.get((variant, "object"), 0.0) + float(gate_obj[i])
            gate_sums[(variant, "attack")] = gate_sums.get((variant, "attack"), 0.0) + float(gate_atk[i])
            trust_sums[(variant, "object")] = trust_sums.get((variant, "object"), 0.0) + float(trust_obj[i])
            trust_sums[(variant, "attack")] = trust_sums.get((variant, "attack"), 0.0) + float(trust_atk[i])
            source_sums[variant] = source_sums.get(variant, 0.0) + float(source_gate[i])
            glyph_sums[(variant, "mean")] = glyph_sums.get((variant, "mean"), 0.0) + float(glyph_mean[i])
            glyph_sums[(variant, "max")] = glyph_sums.get((variant, "max"), 0.0) + float(glyph_max[i])

    family = items[0].family.casefold()
    if any(item.family.casefold() != family for item in items):
        raise RuntimeError("Mixed benchmark families passed to run_typo")
    out: Dict[str, float] = {}
    for variant, total in sorted(totals.items()):
        for key in (
            "notext_acc", "any_acc", "text_acc",
            "read_with_null_acc", "null_reject_acc", "null_accept_acc",
        ):
            out[f"{family}/{variant}/{key}"] = hits.get((variant, key), 0) / total
        out[f"{family}/{variant}/gate_object"] = gate_sums.get((variant, "object"), 0.0) / total
        out[f"{family}/{variant}/gate_attack"] = gate_sums.get((variant, "attack"), 0.0) / total
        out[f"{family}/{variant}/trust_object"] = trust_sums.get((variant, "object"), 0.0) / total
        out[f"{family}/{variant}/trust_attack"] = trust_sums.get((variant, "attack"), 0.0) / total
        out[f"{family}/{variant}/source_gate"] = source_sums.get(variant, 0.0) / total
        out[f"{family}/{variant}/glyph_mean"] = glyph_sums.get((variant, "mean"), 0.0) / total
        out[f"{family}/{variant}/glyph_max"] = glyph_sums.get((variant, "max"), 0.0) / total
        out[f"{family}/{variant}/n"] = float(total)
    return out


def load_mvt_items(csv_path: Path, image_root: Path, max_items: int = 0) -> List[MvtItem]:
    if not csv_path.is_file():
        return []
    items: List[MvtItem] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("image") or row.get("img_path") or "").strip()
            label = str(row.get("label") or row.get("object_label") or "").strip()
            if not name or not label:
                continue
            path = Path(name)
            if not path.is_file():
                path = image_root / name
            if path.is_file():
                items.append(MvtItem(path, label))
            if max_items > 0 and len(items) >= max_items:
                break
    return items


@torch.inference_mode()
def run_mvt(
    model,
    clip_module,
    items: Sequence[MvtItem],
    device: torch.device,
    image_size: int,
    batch_size: int,
    transform_fn,
) -> Dict[str, float]:
    if not items:
        return {}
    labels = sorted({item.label for item in items})
    label_to_index = {label: i for i, label in enumerate(labels)}
    prompts = (
        [f"<notext> {label}" for label in labels]
        + labels
        + ["<text> <null>"]
    )
    tokens = clip_module.tokenize(prompts, truncate=True).to(device)
    correct_notext = correct_any = total = 0
    target_gate_sum = max_gate_sum = target_trust_sum = source_gate_sum = glyph_mean_sum = 0.0
    null_accept = 0
    null_margin_sum = 0.0
    c = len(labels)
    for batch in _chunks(list(items), batch_size):
        images = torch.stack([_load_tensor(x.path, image_size, transform_fn) for x in batch]).to(device)
        details = model.forward_modes(images, tokens, return_details=True)
        logits = details["logits_per_image"].float()
        notext = logits[:, :c]
        any_scores = logits[:, c : 2 * c]
        null_score = logits[:, 2 * c]
        auto_slice = slice(c, 2 * c)
        gates = details["route_gate"][:, auto_slice].float()
        trusts = details["trust_gate"][:, auto_slice].float()
        source_gate = details["source_gate"].float().reshape(-1)

        if trusts.shape != gates.shape:
            raise RuntimeError(
                "MVT trust/route shape mismatch: "
                f"trust={tuple(trusts.shape)} route={tuple(gates.shape)}"
            )
        if source_gate.numel() != len(batch):
            raise RuntimeError(
                "MVT source-gate batch mismatch: "
                f"source={tuple(source_gate.shape)} batch={len(batch)}"
            )

        targets = torch.tensor([label_to_index[x.label] for x in batch], device=device)
        correct_notext += int((notext.argmax(dim=1) == targets).sum())
        correct_any += int((any_scores.argmax(dim=1) == targets).sum())
        target_gate_sum += float(gates.gather(1, targets[:, None]).sum())
        max_gate_sum += float(gates.amax(dim=1).sum())
        target_trust_sum += float(trusts.gather(1, targets[:, None]).sum())
        source_gate_sum += float(source_gate.sum())
        glyph_mean_sum += float(details["glyph_probs"].float().mean(dim=1).sum())
        # ObjectNet is treated as a clean no-readable-text abstention set.
        max_read_word = details["read_logits"][:, c : 2 * c].float().amax(dim=1)
        null_accept += int((null_score > max_read_word).sum())
        null_margin_sum += float((null_score - max_read_word).sum())
        total += len(batch)
    return {
        "mvt/notext_acc": correct_notext / max(1, total),
        "mvt/any_acc": correct_any / max(1, total),
        "mvt/target_gate": target_gate_sum / max(1, total),
        "mvt/max_gate": max_gate_sum / max(1, total),
        "mvt/target_trust": target_trust_sum / max(1, total),
        "mvt/source_gate": source_gate_sum / max(1, total),
        "mvt/glyph_mean": glyph_mean_sum / max(1, total),
        "mvt/null_accept": null_accept / max(1, total),
        "mvt/null_margin": null_margin_sum / max(1, total),
        "mvt/n": float(total),
        "mvt/classes": float(c),
    }


@torch.inference_mode()
def run_validation_benchmarks(model, clip_module, device, args, transform_fn) -> Dict[str, float]:
    model.eval()
    scam = load_typo_items(
        str(args.benchmark_scam_repo),
        getattr(args, "benchmark_scam_revision", None),
        "SCAM",
        args.benchmark_max_items,
    )
    rta = load_typo_items(
        str(args.benchmark_rta_repo),
        getattr(args, "benchmark_rta_revision", None),
        "RTA",
        args.benchmark_max_items,
    )
    out: Dict[str, float] = {}
    out.update(run_typo(
        model, clip_module, scam, device, args.image_size,
        args.benchmark_batch_size, transform_fn,
    ))
    out.update(run_typo(
        model, clip_module, rta, device, args.image_size,
        args.benchmark_batch_size, transform_fn,
    ))

    if bool(getattr(args, "benchmark_include_mvt", False)):
        if args.mvt_csv is None or args.mvt_image_root is None:
            raise RuntimeError(
                "--benchmark_include_mvt requires both --mvt_csv and --mvt_image_root"
            )
        mvt = load_mvt_items(
            Path(args.mvt_csv), Path(args.mvt_image_root), args.benchmark_max_items
        )
        out.update(run_mvt(
            model, clip_module, mvt, device, args.image_size,
            args.mvt_benchmark_batch_size, transform_fn,
        ))

    out.update(_component_metrics(model))
    _print_important(out)
    return out

