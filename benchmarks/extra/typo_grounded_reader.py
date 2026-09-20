#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Human-grounded SCAM/RTA PIECES reader benchmark.

Purpose
-------
Replace arbitrary all-label accuracy with three non-confounded evaluations:

1) semantic binary zero-shot:
       depicted object  vs  benchmark attack word
   scored with PIECES <any>; no global label bank, no canonicalization.

2) human-grounded TEXT_REAL on NoSCAM / NoRTA:
       model says some literal text is supported  vs  independent human text presence
   reported for:
       pair-only pool     = object label + attack word
       grounded pool      = pair-only + human WYSIWYG visible-text fragments
   The probe bank NEVER participates in this accuracy metric.

3) lexical / semantic diagnostics:
   every candidate gets:
       NOTEXT score
       ANY score
       forced READ score
       raw PIECES reader cosine
       calibrated READ-minus-NULL
       early ORTHO cosine
       trust / route gate
       READ->RN attention
   Candidate pools include the human WYSIWYG fragments plus a fixed probe bank.

Attacked SCAM/RTA/Synth* variants inherit the background annotations from their
paired No* source image. The rendered attack word is then an additional known
visible literal positive. We report preference, not arbitrary top-1 correctness:
attack word vs best known background word, and per-visible-word READ coverage.

No threshold is fitted. SCAM/RTA remain evaluation-only. Reference runs use FP32 by default; --amp is opt-in.

Expected repository context
---------------------------
Run from the project root containing:
    oaiclip/
    utils_clip_loader/

Example
-------
python eval_benchmark_zs_typo_grounded_reader.py ^
  --model "path/or/hf-repo-for-full-xattn-model" ^
  --human-attachment typo_noattack_human_text_annotations_v1.json ^
  --probe-bank reader_probe_bank_v1.json ^
  --output-dir out_bench_results/extra/grounded_typo_reader

Outputs
-------
summary.txt
summary_metrics.csv
image_summary.csv
candidate_scores.csv
pair_deltas.csv
probe_summary.csv
probe_top_cases.csv
results_manifest.json

There are intentionally no plots and no all-label accuracy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Direct-script bootstrap after moving this specialized benchmark under benchmarks/extra/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SEED = 20260911
PROMPT = "a photo of a {candidate}"
SUBSET_ORDER = ("NoSCAM", "SCAM", "SynthSCAM", "NoRTA", "RTA", "SynthRTA")
NO_SUBSETS = {"NoSCAM", "NoRTA"}
ATTACK_SUBSETS = {"SCAM", "SynthSCAM", "RTA", "SynthRTA"}
FAMILY_VARIANTS = {
    "SCAM": ("NoSCAM", "SCAM", "SynthSCAM"),
    "RTA": ("NoRTA", "RTA", "SynthRTA"),
}
SOURCE_STAT_NAMES = (
    "glyph_mean",
    "glyph_max",
    "glyph_top8_mean",
    "glyph_soft_area",
    "glyph_row_max_mean",
    "glyph_col_max_mean",
    "glyph_row_density_max",
    "glyph_logit_mean",
)


@dataclass(frozen=True)
class PairSample:
    image: Any
    sample_id: str
    subset: str
    dataset_family: str
    correct_label: str
    distractor_label: str
    pair_key: str


@dataclass(frozen=True)
class HumanRecord:
    sample_id: str
    dataset_family: str
    subset: str
    pair_key: str
    object_label: str
    attack_word: str
    text_real: bool
    visible_text_lines: tuple[str, ...]
    has_non_latin_text: bool
    has_unreadable_text: bool
    annotation_semantic_type: str


@dataclass
class EvalItem:
    sample: PairSample
    human: HumanRecord


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def normalized_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def compact_literal(value: str) -> str:
    # Unicode letters/numbers retained; punctuation/spacing removed.
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def literal_overlap(candidate: str, visible_strings: Sequence[str]) -> bool:
    c = compact_literal(candidate)
    if not c:
        return False
    return any(c in compact_literal(x) for x in visible_strings if compact_literal(x))


def dedupe_strings(values: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        value = str(value).strip()
        if not value:
            continue
        key = normalized_key(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def strip_variant_prefix(sample_id: str, variant: str) -> str:
    raw = str(sample_id).strip()
    stripped = re.sub(
        rf"^{re.escape(variant)}[\s_:\-./]*",
        "",
        raw,
        count=1,
        flags=re.IGNORECASE,
    )
    return stripped if stripped else raw


def safe_float(value: Any) -> float:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().item()
    return float(value)


def div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def fmt_pct(x: float) -> str:
    return "nan" if not math.isfinite(x) else f"{100.0*x:.2f}%"


# ---------------------------------------------------------------------------
# Human attachment + probe bank
# ---------------------------------------------------------------------------


def load_human_attachment(path: Path) -> tuple[dict[str, HumanRecord], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported human attachment schema: {payload.get('schema_version')!r}")

    records: dict[str, HumanRecord] = {}
    for raw in payload["records"]:
        rec = HumanRecord(
            sample_id=str(raw["id"]),
            dataset_family=str(raw["dataset_family"]),
            subset=str(raw["subset"]),
            pair_key=str(raw["pair_key"]),
            object_label=str(raw.get("dataset_object_label", "")),
            attack_word=str(raw.get("dataset_attack_word", "")),
            text_real=bool(raw["TEXT_REAL"]),
            visible_text_lines=tuple(str(x) for x in raw.get("visible_text_lines", [])),
            has_non_latin_text=bool(raw.get("has_non_latin_text")),
            has_unreadable_text=bool(raw.get("has_additional_or_unreadable_text")),
            annotation_semantic_type=str(raw.get("annotation_semantic_type", "")),
        )
        if rec.sample_id in records:
            raise ValueError(f"Duplicate human attachment id {rec.sample_id!r}")
        records[rec.sample_id] = rec

    if len(records) != 2162:
        raise ValueError(f"Human attachment expected 2162 No* records, got {len(records)}")
    return records, payload


def load_probe_bank(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported probe-bank schema: {payload.get('schema_version')!r}")
    categories = {
        str(category): dedupe_strings(values)
        for category, values in payload.get("categories", {}).items()
    }
    return categories, payload


# ---------------------------------------------------------------------------
# Dataset loading / pairing
# ---------------------------------------------------------------------------


def load_samples() -> dict[str, list[PairSample]]:
    from datasets import load_dataset

    result: dict[str, list[PairSample]] = {name: [] for name in SUBSET_ORDER}

    scam = load_dataset("BLISS-e-V/SCAM", split="train")
    for row in scam:
        sample_id = str(row["id"])
        subset = next((x for x in FAMILY_VARIANTS["SCAM"] if sample_id.startswith(x)), None)
        if subset is None:
            continue
        result[subset].append(
            PairSample(
                image=row["image"],
                sample_id=sample_id,
                subset=subset,
                dataset_family="SCAM",
                correct_label=str(row["object_label"]).strip(),
                distractor_label=str(row["attack_word"]).strip(),
                pair_key=strip_variant_prefix(sample_id, subset),
            )
        )

    rta = load_dataset("zer0int/RTA-100-Triplet", split="train")
    for row in rta:
        subset = str(row["type"])
        if subset not in FAMILY_VARIANTS["RTA"]:
            continue
        sample_id = str(row["id"])
        result[subset].append(
            PairSample(
                image=row["image"],
                sample_id=sample_id,
                subset=subset,
                dataset_family="RTA",
                correct_label=str(row["object_label"]).strip(),
                distractor_label=str(row["attack_word"]).strip(),
                pair_key=strip_variant_prefix(sample_id, subset),
            )
        )

    for subset in SUBSET_ORDER:
        if not result[subset]:
            raise RuntimeError(f"Dataset subset {subset} is empty.")
    return result


def validate_clean_attachment(
    samples: dict[str, list[PairSample]],
    human_by_id: Mapping[str, HumanRecord],
) -> dict[tuple[str, str], HumanRecord]:
    by_family_pair: dict[tuple[str, str], HumanRecord] = {}

    for subset in ("NoSCAM", "NoRTA"):
        seen_ids = set()
        for sample in samples[subset]:
            if sample.sample_id not in human_by_id:
                raise RuntimeError(f"Missing human annotation for {sample.sample_id}")
            h = human_by_id[sample.sample_id]
            if h.dataset_family != sample.dataset_family or h.subset != subset:
                raise RuntimeError(f"Human/dataset family mismatch for {sample.sample_id}")
            if h.pair_key != sample.pair_key:
                raise RuntimeError(
                    f"Pair-key mismatch for {sample.sample_id}: attachment={h.pair_key!r}, "
                    f"dataset={sample.pair_key!r}"
                )
            # Raw strings are checked case-insensitively/whitespace-normalized only.
            if h.object_label and normalized_key(h.object_label) != normalized_key(sample.correct_label):
                raise RuntimeError(
                    f"Object-label mismatch for {sample.sample_id}: "
                    f"{h.object_label!r} vs {sample.correct_label!r}"
                )
            if h.attack_word and normalized_key(h.attack_word) != normalized_key(sample.distractor_label):
                raise RuntimeError(
                    f"Attack-label mismatch for {sample.sample_id}: "
                    f"{h.attack_word!r} vs {sample.distractor_label!r}"
                )

            key = (sample.dataset_family, sample.pair_key)
            if key in by_family_pair:
                raise RuntimeError(f"Duplicate clean pair key {key}")
            by_family_pair[key] = h
            seen_ids.add(sample.sample_id)

        expected = 1162 if subset == "NoSCAM" else 1000
        if len(seen_ids) != expected:
            raise RuntimeError(f"{subset}: expected {expected} clean records, got {len(seen_ids)}")

    return by_family_pair


def attach_human_to_variants(
    samples: dict[str, list[PairSample]],
    clean_by_pair: Mapping[tuple[str, str], HumanRecord],
) -> dict[str, list[EvalItem]]:
    result: dict[str, list[EvalItem]] = {}

    for subset in SUBSET_ORDER:
        family = "SCAM" if "SCAM" in subset else "RTA"
        mapped: list[EvalItem] = []
        missing = []

        for sample in samples[subset]:
            h = clean_by_pair.get((family, sample.pair_key))
            if h is None:
                missing.append(sample)
                continue
            if normalized_key(h.object_label) != normalized_key(sample.correct_label):
                raise RuntimeError(
                    f"{subset}/{sample.sample_id}: paired object mismatch "
                    f"{h.object_label!r} vs {sample.correct_label!r}"
                )
            if normalized_key(h.attack_word) != normalized_key(sample.distractor_label):
                raise RuntimeError(
                    f"{subset}/{sample.sample_id}: paired attack mismatch "
                    f"{h.attack_word!r} vs {sample.distractor_label!r}"
                )
            mapped.append(EvalItem(sample=sample, human=h))

        if missing:
            # Explicit deterministic fallback: same clean dataset order, only when
            # size and labels prove alignment. Never silently zip unvalidated rows.
            clean_subset = "NoSCAM" if family == "SCAM" else "NoRTA"
            clean = samples[clean_subset]
            current = samples[subset]
            order_ok = len(clean) == len(current) and all(
                normalized_key(a.correct_label) == normalized_key(b.correct_label)
                and normalized_key(a.distractor_label) == normalized_key(b.distractor_label)
                for a, b in zip(clean, current)
            )
            if not order_ok:
                raise RuntimeError(
                    f"{subset}: {len(missing)} IDs could not pair by normalized ID and "
                    "validated order fallback is unavailable."
                )
            mapped = [
                EvalItem(
                    sample=variant,
                    human=clean_by_pair[(family, base.pair_key)],
                )
                for base, variant in zip(clean, current)
            ]
            print(
                f"[pairing] {subset}: normalized ID coverage incomplete; "
                f"using validated dataset order for {len(mapped)} pairs."
            )

        result[subset] = mapped
    return result


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------


def roles_for_item(
    item: EvalItem,
    probe_categories: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    s = item.sample
    h = item.human

    # Keyed by normalized candidate; preserve first surface form.
    entries: dict[str, dict[str, Any]] = {}

    def add(candidate: str, role: str, category: str | None = None) -> None:
        candidate = str(candidate).strip()
        if not candidate:
            return
        key = normalized_key(candidate)
        rec = entries.setdefault(
            key,
            {
                "candidate": candidate,
                "roles": [],
                "probe_categories": [],
            },
        )
        if role not in rec["roles"]:
            rec["roles"].append(role)
        if category and category not in rec["probe_categories"]:
            rec["probe_categories"].append(category)

    add(s.correct_label, "object_label")
    add(
        s.distractor_label,
        "rendered_attack_text" if s.subset in ATTACK_SUBSETS else "benchmark_attack_word_absent",
    )
    for line in h.visible_text_lines:
        add(line, "human_background_visible_text")

    for category, values in probe_categories.items():
        for candidate in values:
            add(candidate, "probe", category)

    known_visible = list(h.visible_text_lines)
    if s.subset in ATTACK_SUBSETS:
        known_visible.append(s.distractor_label)

    for rec in entries.values():
        roles = set(rec["roles"])
        rec["expected_literal_visible"] = (
            "human_background_visible_text" in roles
            or "rendered_attack_text" in roles
        )
        rec["probe_literal_overlap"] = (
            literal_overlap(rec["candidate"], known_visible)
            if rec["probe_categories"]
            else False
        )

    return entries


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------


class EvalDataset(Dataset):
    def __init__(self, items: Sequence[EvalItem], preprocess):
        self.items = list(items)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.preprocess(self.items[index].sample.image.convert("RGB")), index


def make_loader(items: Sequence[EvalItem], preprocess, batch_size: int, workers: int, device: str):
    return DataLoader(
        EvalDataset(items, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=workers > 0,
    )


# ---------------------------------------------------------------------------
# PIECES scoring
# ---------------------------------------------------------------------------


def tensor_scalar(matrix: torch.Tensor, row: int, col: int) -> float:
    return float(matrix[row, col].detach().float().cpu())


@torch.inference_mode()
def score_subset(
    *,
    model: Any,
    preprocess: Any,
    items: Sequence[EvalItem],
    probe_categories: Mapping[str, Sequence[str]],
    subset: str,
    clip_module: Any,
    inference_autocast: Any,
    device: str,
    amp: bool,
    batch_size: int,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loader = make_loader(items, preprocess, batch_size, workers, device)
    candidate_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []

    architecture = str(getattr(model, "read_attention_architecture", ""))
    if architecture != "sigmoid_all":
        raise RuntimeError(
            f"This benchmark defines reader cosine for current sigmoid_all PIECES; "
            f"loaded read_attention_architecture={architecture!r}"
        )

    scale = float(model.logit_scale.detach().float().exp().cpu())
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid CLIP logit scale {scale}")

    for images, indices in tqdm(loader, desc=f"{subset}: grounded PIECES", leave=False):
        batch_items = [items[int(i)] for i in indices.tolist()]
        role_maps = [roles_for_item(item, probe_categories) for item in batch_items]

        # Union only within this image batch: exact model forward, modest candidate bank.
        union_candidates = dedupe_strings(
            rec["candidate"]
            for role_map in role_maps
            for rec in role_map.values()
        )
        index_by_key = {normalized_key(x): i for i, x in enumerate(union_candidates)}
        n = len(union_candidates)

        prompts = (
            [f"<notext> {PROMPT.format(candidate=x)}" for x in union_candidates]
            + [f"<any> {PROMPT.format(candidate=x)}" for x in union_candidates]
            + [f"<text> {PROMPT.format(candidate=x)}" for x in union_candidates]
            + ["<text> <null>"]
        )
        tokens = clip_module.tokenize(prompts, truncate=True).to(device)
        images = images.to(device, non_blocking=device.startswith("cuda"))

        with inference_autocast(device, amp):
            output = model.forward_modes(
                images,
                tokens,
                apply_content_correction=True,
                return_details=True,
            )

        logits = output["logits_per_image"].detach().float()
        raw = output["raw_read_logits"].detach().float()
        calibrated = output["read_logits"].detach().float()
        relative = output["relative_read_logits"].detach().float()
        early = output["early_orthographic_logits"].detach().float()
        trust = output["trust_gate"].detach().float()
        route = output["route_gate"].detach().float()
        auto_contrib = output["auto_read_contribution"].detach().float()
        rn_attn = output["read_null_attention"].detach().float()
        source_logits = output["source_logits"].detach().float()
        source_probs = source_logits.sigmoid()
        source_gate = output["source_gate"].detach().float()
        source_stats = output["source_stats"].detach().float()
        glyph_probs = output["glyph_probs"].detach().float()

        null_col = 3 * n
        # Explicit <text><null> is required to agree with the model's internal null.
        explicit_null = logits[:, null_col]
        internal_null = output["null_read_logits"].detach().float()
        null_err = float((explicit_null - internal_null).abs().max().cpu())
        if null_err > 5e-4:
            raise RuntimeError(
                f"{subset}: exposed <null> != internal READ null baseline, max_abs={null_err:.6g}"
            )

        # <any> and <text> share the same literal reader candidate; raw values should match.
        if n:
            raw_any = raw[:, n : 2*n]
            raw_read = raw[:, 2*n : 3*n]
            raw_parity = float((raw_any - raw_read).abs().max().cpu())
            if raw_parity > 5e-4:
                raise RuntimeError(
                    f"{subset}: <any>/<text> raw READ parity failed, max_abs={raw_parity:.6g}"
                )

        for local_row, (item, role_map) in enumerate(zip(batch_items, role_maps)):
            s = item.sample
            h = item.human
            null_logit = float(explicit_null[local_row].cpu())
            null_raw_cos = tensor_scalar(raw, local_row, null_col) / scale

            glyph = glyph_probs[local_row]
            stats = source_stats[local_row]
            image_diag = {
                "id": s.sample_id,
                "base_human_id": h.sample_id,
                "pair_key": h.pair_key,
                "dataset_family": s.dataset_family,
                "subset": s.subset,
                "TEXT_REAL": h.text_real,
                "annotation_semantic_type": h.annotation_semantic_type,
                "human_visible_text_count": len(h.visible_text_lines),
                "human_visible_text_lines": " | ".join(h.visible_text_lines),
                "human_has_non_latin_text": h.has_non_latin_text,
                "human_has_unreadable_text": h.has_unreadable_text,
                "correct_label": s.correct_label,
                "attack_word": s.distractor_label,
                "null_read_logit": null_logit,
                "null_raw_reader_cosine": null_raw_cos,
                "source_present_logit": float(source_logits[local_row, 0].cpu()),
                "source_readable_logit": float(source_logits[local_row, 1].cpu()),
                "source_present_prob": float(source_probs[local_row, 0].cpu()),
                "source_readable_prob": float(source_probs[local_row, 1].cpu()),
                "source_gate": float(source_gate[local_row].cpu()),
                "glyph_prob_mean": float(glyph.mean().cpu()),
                "glyph_prob_max": float(glyph.max().cpu()),
                "glyph_prob_p95": float(torch.quantile(glyph, 0.95).cpu()),
                "glyph_prob_p99": float(torch.quantile(glyph, 0.99).cpu()),
                "register_count": int(output["register_mask"][local_row].detach().sum().cpu()),
            }
            for i, name in enumerate(SOURCE_STAT_NAMES):
                image_diag[f"source_stat_{name}"] = float(stats[i].cpu())

            local_candidate_records = []
            for rec in role_map.values():
                candidate = rec["candidate"]
                ci = index_by_key[normalized_key(candidate)]
                notext_col = ci
                any_col = n + ci
                read_col = 2*n + ci

                notext_logit = tensor_scalar(logits, local_row, notext_col)
                any_logit = tensor_scalar(logits, local_row, any_col)
                read_logit = tensor_scalar(logits, local_row, read_col)
                raw_read_logit = tensor_scalar(raw, local_row, read_col)
                rel = tensor_scalar(relative, local_row, read_col)
                early_logit = tensor_scalar(early, local_row, read_col)
                trust_v = tensor_scalar(trust, local_row, read_col)
                route_v = tensor_scalar(route, local_row, read_col)
                rn_v = tensor_scalar(rn_attn, local_row, read_col)
                auto_v = tensor_scalar(auto_contrib, local_row, any_col)

                row = {
                    **{k: image_diag[k] for k in (
                        "id", "base_human_id", "pair_key", "dataset_family", "subset",
                        "TEXT_REAL", "annotation_semantic_type", "human_visible_text_lines",
                        "human_has_non_latin_text", "human_has_unreadable_text",
                        "correct_label", "attack_word"
                    )},
                    "candidate": candidate,
                    "roles": "|".join(rec["roles"]),
                    "probe_categories": "|".join(rec["probe_categories"]),
                    "expected_literal_visible": bool(rec["expected_literal_visible"]),
                    "probe_literal_overlap": bool(rec["probe_literal_overlap"]),
                    "prompt": PROMPT.format(candidate=candidate),
                    "notext_logit": notext_logit,
                    "notext_cosine": notext_logit / scale,
                    "any_logit": any_logit,
                    "any_score_over_scale": any_logit / scale,
                    "any_minus_notext_logit": any_logit - notext_logit,
                    "auto_read_contribution": auto_v,
                    "forced_read_logit": read_logit,
                    "raw_reader_logit": raw_read_logit,
                    "raw_reader_cosine": raw_read_logit / scale,
                    "read_minus_null_logit": rel,
                    "beats_null": read_logit > null_logit,
                    "early_ortho_logit": early_logit,
                    "early_ortho_cosine": early_logit / scale,
                    "trust_gate": trust_v,
                    "route_gate": route_v,
                    "read_null_attention": rn_v,
                    "null_read_logit": null_logit,
                    "null_raw_reader_cosine": null_raw_cos,
                }
                # Parity: relative diagnostic should equal calibrated candidate - null.
                if abs((read_logit - null_logit) - rel) > 7e-4:
                    raise RuntimeError(
                        f"{s.sample_id}/{candidate}: relative READ parity failed: "
                        f"final-null={read_logit-null_logit:+.6f}, detail={rel:+.6f}"
                    )
                candidate_rows.append(row)
                local_candidate_records.append(row)

            # Index local candidate records by role.
            def with_role(role: str) -> list[dict[str, Any]]:
                return [r for r in local_candidate_records if role in r["roles"].split("|")]

            obj = with_role("object_label")
            attack = [
                r for r in local_candidate_records
                if "rendered_attack_text" in r["roles"].split("|")
                or "benchmark_attack_word_absent" in r["roles"].split("|")
            ]
            bg = with_role("human_background_visible_text")
            if len(obj) != 1 or len(attack) != 1:
                raise RuntimeError(f"{s.sample_id}: object/attack candidate dedupe ambiguity")

            object_row = obj[0]
            attack_row = attack[0]
            pair_rows = [object_row, attack_row]
            grounded_rows = dedupe_candidate_rows(pair_rows + bg)

            pair_max = max(pair_rows, key=lambda r: r["forced_read_logit"])
            grounded_max = max(grounded_rows, key=lambda r: r["forced_read_logit"])
            all_max = max(local_candidate_records, key=lambda r: r["forced_read_logit"])

            # Semantic binary: object-vs-attack only. No all-label accuracy.
            semantic_binary_correct = object_row["any_logit"] >= attack_row["any_logit"]
            semantic_binary_margin = object_row["any_logit"] - attack_row["any_logit"]

            # Original forced-reading binary on attacked variants:
            # attack word must beat object and NULL.
            attack_binary_read_correct = None
            attack_binary_read_margin = None
            if s.subset in ATTACK_SUBSETS:
                competitor = max(object_row["forced_read_logit"], null_logit)
                attack_binary_read_margin = attack_row["forced_read_logit"] - competitor
                attack_binary_read_correct = attack_binary_read_margin >= 0

            gt_seen = [bool(r["beats_null"]) for r in bg]
            attack_vs_best_bg_logit = None
            attack_vs_best_bg_cos = None
            attack_preferred_over_all_bg = None
            best_bg = None
            if s.subset in ATTACK_SUBSETS and bg:
                best_bg = max(bg, key=lambda r: r["forced_read_logit"])
                attack_vs_best_bg_logit = (
                    attack_row["forced_read_logit"] - best_bg["forced_read_logit"]
                )
                attack_vs_best_bg_cos = (
                    attack_row["raw_reader_cosine"] - best_bg["raw_reader_cosine"]
                )
                attack_preferred_over_all_bg = attack_vs_best_bg_logit >= 0

            # Did a diagnostic probe change the max away from the grounded pool?
            grounded_keys = {normalized_key(r["candidate"]) for r in grounded_rows}
            probe_caused_new_max = normalized_key(all_max["candidate"]) not in grounded_keys

            image_diag.update(
                {
                    "semantic_binary_any_correct": semantic_binary_correct,
                    "semantic_binary_any_margin": semantic_binary_margin,
                    "pair_only_pred_TEXT": pair_max["forced_read_logit"] > null_logit,
                    "pair_only_best_candidate": pair_max["candidate"],
                    "pair_only_best_minus_null": pair_max["forced_read_logit"] - null_logit,
                    "grounded_pred_TEXT": grounded_max["forced_read_logit"] > null_logit,
                    "grounded_best_candidate": grounded_max["candidate"],
                    "grounded_best_minus_null": grounded_max["forced_read_logit"] - null_logit,
                    "all_including_probes_best_candidate": all_max["candidate"],
                    "all_including_probes_best_minus_null": all_max["forced_read_logit"] - null_logit,
                    "probe_caused_new_max": probe_caused_new_max,
                    "gt_transcribed_candidate_count": len(bg),
                    "gt_candidate_seen_count": sum(gt_seen),
                    "gt_candidate_seen_fraction": div(sum(gt_seen), len(gt_seen)),
                    "gt_any_candidate_seen": any(gt_seen) if gt_seen else None,
                    "gt_all_candidates_seen": all(gt_seen) if gt_seen else None,
                    "attack_binary_read_correct": attack_binary_read_correct,
                    "attack_binary_read_margin": attack_binary_read_margin,
                    "attack_preferred_over_all_background_words": attack_preferred_over_all_bg,
                    "attack_minus_best_background_read_logit": attack_vs_best_bg_logit,
                    "attack_minus_best_background_reader_cosine": attack_vs_best_bg_cos,
                    "best_background_candidate": None if best_bg is None else best_bg["candidate"],
                }
            )
            image_rows.append(image_diag)

        del output, logits, raw, calibrated, relative, early, trust, route
        del auto_contrib, rn_attn, source_logits, source_probs, source_gate, source_stats, glyph_probs
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return image_rows, candidate_rows


def dedupe_candidate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        key = normalized_key(row["candidate"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def binary_confusion(rows: Sequence[Mapping[str, Any]], pred_field: str) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for row in rows:
        y = bool(row["TEXT_REAL"])
        p = bool(row[pred_field])
        if not y and not p:
            tn += 1
        elif not y and p:
            fp += 1
        elif y and not p:
            fn += 1
        else:
            tp += 1
    n = tn + fp + fn + tp
    accuracy = div(tp + tn, n)
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall) if math.isfinite(precision) and math.isfinite(recall) else float("nan")
    return {
        "n": n,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "pred_TEXT_rate": div(tp + fp, n),
        "TEXT_REAL_rate": div(tp + fn, n),
    }


def average(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(
    image_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    probe_categories: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []

    # Human-grounded No* detection.
    for subset in ("NoSCAM", "NoRTA"):
        rows = [r for r in image_rows if r["subset"] == subset]
        for pool, field in (
            ("pair_only", "pair_only_pred_TEXT"),
            ("grounded_plus_human_words", "grounded_pred_TEXT"),
        ):
            m = binary_confusion(rows, field)
            summary.append(
                {
                    "section": "TEXT_REAL_detection",
                    "subset": subset,
                    "condition": pool,
                    **m,
                }
            )

    combined = [r for r in image_rows if r["subset"] in NO_SUBSETS]
    for pool, field in (
        ("pair_only", "pair_only_pred_TEXT"),
        ("grounded_plus_human_words", "grounded_pred_TEXT"),
    ):
        summary.append(
            {
                "section": "TEXT_REAL_detection",
                "subset": "NoSCAM+NoRTA",
                "condition": pool,
                **binary_confusion(combined, field),
            }
        )

    # Semantic binary object-vs-attack in ANY.
    for subset in SUBSET_ORDER:
        rows = [r for r in image_rows if r["subset"] == subset]
        summary.append(
            {
                "section": "semantic_binary_ANY",
                "subset": subset,
                "condition": "object_vs_attack",
                "n": len(rows),
                "accuracy": div(sum(bool(r["semantic_binary_any_correct"]) for r in rows), len(rows)),
                "mean_margin": average([r["semantic_binary_any_margin"] for r in rows]),
            }
        )

    # Known rendered attack-word binary forced reading.
    for subset in ("SCAM", "SynthSCAM", "RTA", "SynthRTA"):
        rows = [r for r in image_rows if r["subset"] == subset]
        summary.append(
            {
                "section": "attack_word_binary_READ",
                "subset": subset,
                "condition": "attack_vs_object_and_null",
                "n": len(rows),
                "accuracy": div(sum(bool(r["attack_binary_read_correct"]) for r in rows), len(rows)),
                "mean_margin": average([r["attack_binary_read_margin"] for r in rows]),
            }
        )

    # Human transcribed-word coverage and post-it preference.
    for subset in SUBSET_ORDER:
        rows = [r for r in image_rows if r["subset"] == subset]
        evaluable = [r for r in rows if int(r["gt_transcribed_candidate_count"]) > 0]
        if evaluable:
            summary.append(
                {
                    "section": "background_word_READ_coverage",
                    "subset": subset,
                    "condition": "human_transcribed_fragments",
                    "n_images": len(evaluable),
                    "n_candidates": sum(int(r["gt_transcribed_candidate_count"]) for r in evaluable),
                    "candidate_seen_rate": div(
                        sum(int(r["gt_candidate_seen_count"]) for r in evaluable),
                        sum(int(r["gt_transcribed_candidate_count"]) for r in evaluable),
                    ),
                    "image_any_seen_rate": div(
                        sum(bool(r["gt_any_candidate_seen"]) for r in evaluable),
                        len(evaluable),
                    ),
                    "image_all_seen_rate": div(
                        sum(bool(r["gt_all_candidates_seen"]) for r in evaluable),
                        len(evaluable),
                    ),
                }
            )

        if subset in ATTACK_SUBSETS:
            with_bg = [
                r for r in rows
                if r["attack_preferred_over_all_background_words"] is not None
            ]
            if with_bg:
                summary.append(
                    {
                        "section": "postit_vs_background_preference",
                        "subset": subset,
                        "condition": "attack_word_vs_best_human_background_fragment",
                        "n": len(with_bg),
                        "attack_preference_rate": div(
                            sum(bool(r["attack_preferred_over_all_background_words"]) for r in with_bg),
                            len(with_bg),
                        ),
                        "mean_attack_minus_background_read_logit": average(
                            [r["attack_minus_best_background_read_logit"] for r in with_bg]
                        ),
                        "mean_attack_minus_background_reader_cosine": average(
                            [r["attack_minus_best_background_reader_cosine"] for r in with_bg]
                        ),
                    }
                )

    # Probe summaries. Exclude literal overlap from "negative" interpretation, but
    # retain every raw candidate row in candidate_scores.csv.
    probe_summary: list[dict[str, Any]] = []
    for category in probe_categories:
        for subset in SUBSET_ORDER:
            rows = [
                r for r in candidate_rows
                if category in str(r["probe_categories"]).split("|")
                and r["subset"] == subset
                and not bool(r["probe_literal_overlap"])
            ]
            if not rows:
                continue

            # For No* also split by actual human text presence.
            text_real_groups = [("all", rows)]
            if subset in NO_SUBSETS:
                text_real_groups.extend(
                    [
                        ("TEXT_REAL=false", [r for r in rows if not bool(r["TEXT_REAL"])]),
                        ("TEXT_REAL=true", [r for r in rows if bool(r["TEXT_REAL"])]),
                    ]
                )

            for gt_group, group in text_real_groups:
                if not group:
                    continue
                probe_summary.append(
                    {
                        "probe_category": category,
                        "subset": subset,
                        "TEXT_REAL_group": gt_group,
                        "n_scores": len(group),
                        "beats_null_rate": div(sum(bool(r["beats_null"]) for r in group), len(group)),
                        "mean_raw_reader_cosine": average([r["raw_reader_cosine"] for r in group]),
                        "mean_read_minus_null_logit": average([r["read_minus_null_logit"] for r in group]),
                        "mean_notext_cosine": average([r["notext_cosine"] for r in group]),
                        "mean_any_minus_notext_logit": average([r["any_minus_notext_logit"] for r in group]),
                        "mean_trust_gate": average([r["trust_gate"] for r in group]),
                        "mean_route_gate": average([r["route_gate"] for r in group]),
                    }
                )

    return summary, probe_summary


def probe_top_case_rows(
    candidate_rows: Sequence[dict[str, Any]],
    *,
    top_k: int = 25,
) -> list[dict[str, Any]]:
    """
    Rank diagnostic-probe outliers without turning probes into accuracy classes.

    raw_reader_cosine:
        literal-reader affinity before calibration
    any_minus_notext_logit:
        routed reading contribution added to semantic <any>

    Probe/image pairs that literally overlap known visible strings are excluded
    from this ranked trap view, but remain in candidate_scores.csv.
    """
    probes = [
        r for r in candidate_rows
        if str(r.get("probe_categories", "")).strip()
        and not bool(r.get("probe_literal_overlap"))
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in probes:
        for category in str(r["probe_categories"]).split("|"):
            category = category.strip()
            if category:
                grouped[(category, str(r["candidate"]), str(r["subset"]))].append(r)

    out: list[dict[str, Any]] = []
    for (category, candidate, subset), rows in sorted(grouped.items()):
        for metric in ("raw_reader_cosine", "any_minus_notext_logit"):
            ranked = sorted(rows, key=lambda r: float(r[metric]), reverse=True)[:top_k]
            for rank, r in enumerate(ranked, start=1):
                out.append(
                    {
                        "probe_category": category,
                        "probe_candidate": candidate,
                        "subset": subset,
                        "ranking_metric": metric,
                        "rank": rank,
                        "metric_value": r[metric],
                        "id": r["id"],
                        "base_human_id": r["base_human_id"],
                        "pair_key": r["pair_key"],
                        "TEXT_REAL": r["TEXT_REAL"],
                        "human_visible_text_lines": r["human_visible_text_lines"],
                        "human_has_non_latin_text": r["human_has_non_latin_text"],
                        "human_has_unreadable_text": r["human_has_unreadable_text"],
                        "correct_label": r["correct_label"],
                        "attack_word": r["attack_word"],
                        "notext_cosine": r["notext_cosine"],
                        "raw_reader_cosine": r["raw_reader_cosine"],
                        "read_minus_null_logit": r["read_minus_null_logit"],
                        "beats_null": r["beats_null"],
                        "any_minus_notext_logit": r["any_minus_notext_logit"],
                        "trust_gate": r["trust_gate"],
                        "route_gate": r["route_gate"],
                    }
                )
    return out


def pair_delta_rows(
    image_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_candidate = {
        (r["dataset_family"], r["pair_key"], r["subset"], normalized_key(r["candidate"])): r
        for r in candidate_rows
    }
    image_by = {
        (r["dataset_family"], r["pair_key"], r["subset"]): r
        for r in image_rows
    }
    out = []

    for family, variants in FAMILY_VARIANTS.items():
        clean_subset = variants[0]
        clean_images = [
            r for r in image_rows
            if r["dataset_family"] == family and r["subset"] == clean_subset
        ]
        for clean_img in clean_images:
            pair_key = clean_img["pair_key"]
            attack_word = clean_img["attack_word"]
            clean_attack = by_candidate.get(
                (family, pair_key, clean_subset, normalized_key(attack_word))
            )
            if clean_attack is None:
                continue

            for attacked_subset in variants[1:]:
                attacked_img = image_by.get((family, pair_key, attacked_subset))
                attacked_attack = by_candidate.get(
                    (family, pair_key, attacked_subset, normalized_key(attack_word))
                )
                if attacked_img is None or attacked_attack is None:
                    continue

                out.append(
                    {
                        "dataset_family": family,
                        "pair_key": pair_key,
                        "clean_id": clean_img["id"],
                        "attacked_id": attacked_img["id"],
                        "attacked_subset": attacked_subset,
                        "TEXT_REAL_background": clean_img["TEXT_REAL"],
                        "attack_word": attack_word,
                        "attack_delta_raw_reader_cosine": (
                            attacked_attack["raw_reader_cosine"] - clean_attack["raw_reader_cosine"]
                        ),
                        "attack_delta_read_minus_null_logit": (
                            attacked_attack["read_minus_null_logit"] - clean_attack["read_minus_null_logit"]
                        ),
                        "attack_delta_notext_cosine": (
                            attacked_attack["notext_cosine"] - clean_attack["notext_cosine"]
                        ),
                        "attack_delta_any_minus_notext_logit": (
                            attacked_attack["any_minus_notext_logit"] - clean_attack["any_minus_notext_logit"]
                        ),
                        "clean_attack_beats_null": clean_attack["beats_null"],
                        "attacked_attack_beats_null": attacked_attack["beats_null"],
                        "attacked_attack_preferred_over_all_background_words": attacked_img[
                            "attack_preferred_over_all_background_words"
                        ],
                    }
                )

    return out


# ---------------------------------------------------------------------------
# ASCII output
# ---------------------------------------------------------------------------


def ascii_summary(summary_rows: Sequence[dict[str, Any]], probe_rows: Sequence[dict[str, Any]]) -> str:
    lines = []
    lines.append("HUMAN-GROUNDED SCAM/RTA PIECES READER BENCHMARK")
    lines.append("=" * 110)
    lines.append("No all-label accuracy. No label canonicalization. No threshold fitting.")
    lines.append("")

    rows = [r for r in summary_rows if r["section"] == "TEXT_REAL_detection"]
    lines.append("TEXT_REAL on No* variants")
    lines.append(
        f"{'Subset':<16} {'Pool':<28} {'N':>5} {'Acc':>8} {'Prec':>8} "
        f"{'Recall':>8} {'Spec':>8} {'F1':>8} {'PredTEXT':>9}"
    )
    lines.append("-" * 110)
    for r in rows:
        lines.append(
            f"{r['subset']:<16} {r['condition']:<28} {r['n']:>5d} "
            f"{fmt_pct(r['accuracy']):>8} {fmt_pct(r['precision']):>8} "
            f"{fmt_pct(r['recall']):>8} {fmt_pct(r['specificity']):>8} "
            f"{r['f1']:>8.3f} {fmt_pct(r['pred_TEXT_rate']):>9}"
        )
    lines.append("")

    rows = [r for r in summary_rows if r["section"] == "semantic_binary_ANY"]
    lines.append("Semantic binary ZS: object vs benchmark attack word (<any>)")
    lines.append(f"{'Subset':<14} {'N':>6} {'Accuracy':>10} {'Mean margin':>14}")
    lines.append("-" * 50)
    for r in rows:
        lines.append(
            f"{r['subset']:<14} {r['n']:>6d} {fmt_pct(r['accuracy']):>10} "
            f"{r['mean_margin']:>+14.5f}"
        )
    lines.append("")

    rows = [r for r in summary_rows if r["section"] == "attack_word_binary_READ"]
    lines.append("Rendered attack-word forced READ: attack word vs object + NULL")
    lines.append(f"{'Subset':<14} {'N':>6} {'Accuracy':>10} {'Mean margin':>14}")
    lines.append("-" * 50)
    for r in rows:
        lines.append(
            f"{r['subset']:<14} {r['n']:>6d} {fmt_pct(r['accuracy']):>10} "
            f"{r['mean_margin']:>+14.5f}"
        )
    lines.append("")

    rows = [r for r in summary_rows if r["section"] == "background_word_READ_coverage"]
    lines.append("Human WYSIWYG background-word coverage")
    lines.append(
        f"{'Subset':<14} {'Imgs':>6} {'Words':>7} {'Word>NULL':>11} "
        f"{'Any seen':>10} {'All seen':>10}"
    )
    lines.append("-" * 68)
    for r in rows:
        lines.append(
            f"{r['subset']:<14} {r['n_images']:>6d} {r['n_candidates']:>7d} "
            f"{fmt_pct(r['candidate_seen_rate']):>11} "
            f"{fmt_pct(r['image_any_seen_rate']):>10} "
            f"{fmt_pct(r['image_all_seen_rate']):>10}"
        )
    lines.append("")

    rows = [r for r in summary_rows if r["section"] == "postit_vs_background_preference"]
    lines.append("Word-vs-word preference on attacked variants")
    lines.append(
        f"{'Subset':<14} {'N':>6} {'Post-it wins':>13} "
        f"{'ΔREAD logit':>13} {'Δreader cos':>13}"
    )
    lines.append("-" * 66)
    for r in rows:
        lines.append(
            f"{r['subset']:<14} {r['n']:>6d} {fmt_pct(r['attack_preference_rate']):>13} "
            f"{r['mean_attack_minus_background_read_logit']:>+13.5f} "
            f"{r['mean_attack_minus_background_reader_cosine']:>+13.5f}"
        )
    lines.append("")

    lines.append("Probe categories (raw detailed scores are in candidate_scores.csv)")
    lines.append(
        f"{'Category':<26} {'Subset':<12} {'GT group':<16} {'N':>6} "
        f"{'>NULL':>8} {'reader cos':>11} {'ANY-NT':>10}"
    )
    lines.append("-" * 100)
    for r in probe_rows:
        if r["TEXT_REAL_group"] != "all":
            continue
        lines.append(
            f"{r['probe_category']:<26} {r['subset']:<12} "
            f"{r['TEXT_REAL_group']:<16} {r['n_scores']:>6d} "
            f"{fmt_pct(r['beats_null_rate']):>8} "
            f"{r['mean_raw_reader_cosine']:>11.5f} "
            f"{r['mean_any_minus_notext_logit']:>+10.5f}"
        )
    lines.append("")
    lines.append(
        "Interpretation: probes do not participate in TEXT_REAL accuracy. "
        "For semantic derailment, ANY-NOTEXT is the bridge-added score contribution; "
        "NOTEXT cosine is the candidate-independent semantic/content baseline."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Full PIECES model repo ID or local path.")
    ap.add_argument("--human-attachment", type=Path, default="utils_datasets/typo_ref/typo_noattack_human_text_annotations_v1.json")
    ap.add_argument("--probe-bank", type=Path, default="utils_datasets/typo_ref/reader_probe_bank_v1.json")
    ap.add_argument("--output-dir", type=Path, default=Path("out_bench_results/extra/grounded_typo_reader"))
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable mixed-precision autocast. Default: False. The grounded reader "
            "benchmark runs FP32 by default because it parity-checks separately "
            "evaluated explicit and internal NULL paths."
        ),
    )
    ap.add_argument("--debug-limit", type=int, default=0, help="Per-subset smoke-test limit; 0=all.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()

    # Repository-local imports are deliberately delayed so --help and py_compile
    # work outside the project checkout.
    import oaiclip as clip
    from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything
    from utils_clip_loader.benchmark_runtime import inference_autocast, is_full_xattn

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp)

    # The benchmark's diagnostic parity checks compare equivalent reader/null
    # computations reached through different batching paths. Keep the default run
    # numerically strict rather than hiding mixed-precision differences behind a
    # looser tolerance.
    if device.startswith("cuda") and not amp:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    human_by_id, human_payload = load_human_attachment(args.human_attachment)
    probe_categories, probe_payload = load_probe_bank(args.probe_bank)

    print(f"[device] {device} amp={amp}")
    print(f"[human attachment] {len(human_by_id)} records")
    print("[probe categories]", ", ".join(probe_categories))

    samples = load_samples()
    clean_by_pair = validate_clean_attachment(samples, human_by_id)
    eval_sets = attach_human_to_variants(samples, clean_by_pair)

    if args.debug_limit > 0:
        eval_sets = {
            subset: items[: args.debug_limit]
            for subset, items in eval_sets.items()
        }

    model, preprocess, info = load_openai_clip_anything(
        clip, args.model, device=device
    )
    # Full FP32 is intentional for the public/reference benchmark. The exported
    # checkpoints are FP32-safe, and this avoids false parity failures between
    # mathematically equivalent explicit/internal NULL evaluations.
    if not amp:
        model = model.float()
    model = model.eval()

    if not is_full_xattn(model, info):
        raise RuntimeError(
            f"Grounded reader benchmark requires the full PIECES model; "
            f"loaded family={getattr(info, 'model_family', None)!r}"
        )
    if getattr(model, "read_implant", None) is None:
        raise RuntimeError("Loaded model has no read_implant.")

    architecture = str(getattr(model, "read_attention_architecture", ""))
    if architecture != "sigmoid_all":
        raise RuntimeError(
            f"Expected final sigmoid_all reader, got {architecture!r}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_image_rows: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, Any]] = []

    for subset in SUBSET_ORDER:
        items = eval_sets[subset]
        print(f"[subset] {subset}: {len(items)}")
        image_rows, candidate_rows = score_subset(
            model=model,
            preprocess=preprocess,
            items=items,
            probe_categories=probe_categories,
            subset=subset,
            clip_module=clip,
            inference_autocast=inference_autocast,
            device=device,
            amp=amp,
            batch_size=args.batch_size,
            workers=args.num_workers,
        )
        all_image_rows.extend(image_rows)
        all_candidate_rows.extend(candidate_rows)

    summary_rows, probe_rows = summarize(
        all_image_rows,
        all_candidate_rows,
        probe_categories,
    )
    deltas = pair_delta_rows(all_image_rows, all_candidate_rows)
    probe_top = probe_top_case_rows(all_candidate_rows)

    write_csv(args.output_dir / "summary_metrics.csv", summary_rows)
    write_csv(args.output_dir / "image_summary.csv", all_image_rows)
    write_csv(args.output_dir / "candidate_scores.csv", all_candidate_rows)
    write_csv(args.output_dir / "pair_deltas.csv", deltas)
    write_csv(args.output_dir / "probe_summary.csv", probe_rows)
    write_csv(args.output_dir / "probe_top_cases.csv", probe_top)

    summary_text = ascii_summary(summary_rows, probe_rows)
    (args.output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    manifest = {
        "benchmark": "human_grounded_scam_rta_pieces_reader",
        "seed": SEED,
        "model": args.model,
        "model_family": getattr(info, "model_family", None),
        "source_kind": getattr(info, "source_kind", None),
        "read_attention_architecture": architecture,
        "device": device,
        "amp": amp,
        "batch_size": args.batch_size,
        "human_attachment": {
            "path": str(args.human_attachment),
            "sha256": sha256_file(args.human_attachment),
            "schema_version": human_payload.get("schema_version"),
            "counts": human_payload.get("counts"),
        },
        "probe_bank": {
            "path": str(args.probe_bank),
            "sha256": sha256_file(args.probe_bank),
            "schema_version": probe_payload.get("schema_version"),
            "categories": probe_categories,
        },
        "dataset_sources": {
            "SCAM": "BLISS-e-V/SCAM",
            "RTA": "zer0int/RTA-100-Triplet",
        },
        "notes": [
            "No all-label accuracy.",
            "No label canonicalization.",
            "No threshold is selected or fitted on SCAM/RTA.",
            "Probe candidates never participate in TEXT_REAL accuracy.",
            "raw_reader_cosine = raw sigmoid_all PIECES READ similarity before calibration / logit_scale.",
            "ANY-NOTEXT is the bridge-added routed reading contribution.",
        ],
        "output_counts": {
            "image_rows": len(all_image_rows),
            "candidate_rows": len(all_candidate_rows),
            "pair_delta_rows": len(deltas),
            "probe_top_case_rows": len(probe_top),
        },
    }
    (args.output_dir / "results_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(summary_text)
    print(f"[saved] {args.output_dir}")

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
