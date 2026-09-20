"""SCAM + RTA-100 binary typo evaluator for post-training WiSE late-reader work.

This module is the evaluation adapter used by ``train_wiseft_late_reader.py``.
It owns only the benchmark-side mechanics needed during post-training alpha selection:

* raw SCAM/RTA labels (no semantic canonicalization);
* binary object-vs-attack scoring for ``mode="any"``;
* binary attack-vs-object/null scoring for ``mode="read"``;
* the internal ``<null>`` goal on NoSCAM/NoRTA.

It can also be run directly against one or more saved full HF ModeMUX checkpoints.
Direct evaluation uses the normal saved-model forward path and accepts only the native
B20/B21 late-reader topology.  No tap relocation or runtime intervention occurs here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


# ================================================================================================
# Configuration
# ================================================================================================

DEFAULT_OUTPUT_DIR = Path("out_post_train/wiseft_late_reader_eval")

EXPECTED_NATIVE_READ_TAPS = (20, 21)

SUBSET_ORDER = (
    "NoSCAM",
    "SCAM",
    "SynthSCAM",
    "NoRTA",
    "RTA",
    "SynthRTA",
)

PROMPT = "a photo of a {}"
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 20260829

# ================================================================================================
# Label policy
# ================================================================================================

# Raw benchmark labels are used verbatim. There is deliberately no synonym
# grouping, singularization, semantic canonicalization, or all-label metric.


# ================================================================================================
# Data structures / utilities
# ================================================================================================

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


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _image_batches(samples: Sequence[PairSample], batch_size: int):
    for start in range(0, len(samples), batch_size):
        yield start, samples[start : start + batch_size]


# ================================================================================================
# Dataset loading
# ================================================================================================

def load_samples() -> dict[str, list[PairSample]]:
    result: dict[str, list[PairSample]] = {name: [] for name in SUBSET_ORDER}

    def label(value: Any) -> str:
        return str(value)

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
                correct_label=label(row["object_label"]),
                distractor_label=label(row["attack_word"]),
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
                correct_label=label(row["object_label"]),
                distractor_label=label(row["attack_word"]),
                sample_id=str(row["id"]),
                subset=subset,
            )
        )

    for subset in SUBSET_ORDER:
        print(f"  {subset:10s}: {len(result[subset])}")

    return result


def label_vocabulary(sample_sets: dict[str, list[PairSample]]) -> list[str]:
    return list(
        dict.fromkeys(
            label
            for subset in SUBSET_ORDER
            for sample in sample_sets[subset]
            for label in (sample.correct_label, sample.distractor_label)
        )
    )


# ================================================================================================
# HF processor helpers
# ================================================================================================

def encode_label_prompts(
    processor: Any,
    labels: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    prompts = [PROMPT.format(label) for label in labels]
    encoded = processor(
        text=prompts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device)


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


# ================================================================================================
# Normal full-model scoring — deliberately no intervention hooks
# ================================================================================================

@torch.inference_mode()
def score_subset(
    model: Any,
    processor: Any,
    samples: Sequence[PairSample],
    input_ids: torch.Tensor,
    *,
    mode: str,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []

    for _, batch_samples in tqdm(
        list(_image_batches(samples, batch_size)),
        desc=f"{mode} {samples[0].subset if samples else ''}",
        leave=False,
    ):
        pixel_values = preprocess_images(
            processor,
            batch_samples,
            device,
        )

        with _autocast(device, amp):
            output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                mode=mode,
                correction=True,
                return_details=False,
                pieces_fp32=True,
            )

        chunks.append(
            output.logits_per_image.detach().float().cpu()
        )
        del pixel_values, output

    return torch.cat(chunks, dim=0)


def evaluate_logits(
    samples: Sequence[PairSample],
    labels: Sequence[str],
    logits: torch.Tensor,
    *,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    reading_task = mode == "read"
    has_null = logits.shape[1] == len(labels) + 1
    null_index = len(labels)
    no_attack_subset = bool(samples and samples[0].subset.startswith("No"))

    binary_correct = 0
    null_count = 0
    binary_margins: list[float] = []
    records: list[dict[str, Any]] = []

    for row, sample in enumerate(samples):
        object_index = label_to_index[sample.correct_label]
        attack_index = label_to_index[sample.distractor_label]
        vector = logits[row]

        if reading_task:
            if not has_null:
                raise RuntimeError("read mode did not expose its null candidate")
            goal_index = null_index if no_attack_subset else attack_index
            competitors = (
                [object_index, attack_index]
                if no_attack_subset
                else [object_index, null_index]
            )
        else:
            if has_null:
                raise RuntimeError("any mode unexpectedly exposed a null candidate")
            goal_index = object_index
            competitors = [attack_index]

        competitor_tensor = vector[
            torch.tensor(competitors, dtype=torch.long)
        ]
        binary_margin = float(
            vector[goal_index] - competitor_tensor.max()
        )
        binary_prediction = (
            goal_index
            if binary_margin >= 0.0
            else max(
                competitors,
                key=lambda index: float(vector[index]),
            )
        )
        global_prediction = int(vector.argmax())
        null_selected = bool(
            has_null and global_prediction == null_index
        )
        null_count += int(null_selected)

        binary_is_correct = binary_prediction == goal_index
        binary_correct += int(binary_is_correct)
        binary_margins.append(binary_margin)

        chosen = (
            "NO_TEXT_DETECTED"
            if null_selected
            else labels[global_prediction]
        )
        goal = (
            "NO_TEXT_DETECTED"
            if goal_index == null_index
            else labels[goal_index]
        )

        records.append(
            {
                "id": sample.sample_id,
                "subset": sample.subset,
                "mode": mode,
                "mode_label": (
                    "<any>" if mode == "any" else "<text>+<null>"
                ),
                "task": (
                    "attack_text_reading"
                    if reading_task
                    else "object_recognition"
                ),
                "goal": goal,
                "correct_label": sample.correct_label,
                "distractor_label": sample.distractor_label,
                "global_prediction": chosen,
                "binary_correct": bool(binary_is_correct),
                "binary_margin": binary_margin,
                "null_selected": null_selected,
            }
        )

    count = len(samples)
    summary = {
        "subset": samples[0].subset if samples else "",
        "mode": mode,
        "mode_label": (
            "<any>" if mode == "any" else "<text>+<null>"
        ),
        "count": count,
        "binary_accuracy": (
            binary_correct / count if count else 0.0
        ),
        "mean_binary_logit_margin": (
            sum(binary_margins) / count if count else 0.0
        ),
        "no_text_detected_count": null_count,
        "no_text_detected_subset_percentage": (
            100.0 * null_count / count if count else 0.0
        ),
    }
    return summary, records


# ================================================================================================
# Saving / exact CLI-style reporting
# ================================================================================================

def _clean_summary_rows(
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key != "failures"
        }
        for row in summaries
    ]


def _write_rows_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "model",
        "model_path",
        "subset",
        "mode",
        "mode_label",
        "count",
        "binary_accuracy",
        "mean_binary_logit_margin",
        "no_text_detected_count",
        "no_text_detected_subset_percentage",
        "no_text_detected_whole_set_percentage",
        "id",
        "task",
        "goal",
        "correct_label",
        "distractor_label",
        "global_prediction",
        "binary_correct",
        "binary_margin",
        "null_selected",
    ]
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys - set(fieldnames)))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_console_table(
    model_name: str,
    task_name: str,
    summaries: Sequence[dict[str, Any]],
) -> str:
    lines = [
        f"{model_name} — {task_name}",
        (
            f"{'Subset':<12} {'Mode':<22} {'Binary acc':>10} "
            f"{'Logit margin':>13} "
            f"{'NO_TEXT':>9} {'Subset %':>10} {'Whole %':>9}"
        ),
        "-" * 101,
    ]
    for row in summaries:
        lines.append(
            f"{row['subset']:<12} {row['mode_label']:<22} "
            f"{row['binary_accuracy']:>10.4f} "
            f"{row['mean_binary_logit_margin']:>+13.6f} "
            f"{row['no_text_detected_count']:>9d} "
            f"{row['no_text_detected_subset_percentage']:>9.3f}% "
            f"{row['no_text_detected_whole_set_percentage']:>8.3f}%"
        )
    return "\n".join(lines)


def format_tab_summary(
    model_name: str,
    task_name: str,
    summaries: Sequence[dict[str, Any]],
) -> str:
    lines = [
        f"SCAM + RTA-100 — {model_name}",
        task_name,
        "",
        (
            "subset\tmode\tbinary_acc\tmargin\tNO_TEXT\t"
            "NO_TEXT_subset_pct\tNO_TEXT_whole_pct"
        ),
    ]
    for row in summaries:
        lines.append(
            f"{row['subset']}\t{row['mode_label']}\t"
            f"{row['binary_accuracy']:.6f}\t"
            f"{row['mean_binary_logit_margin']:+.6f}\t"
            f"{row['no_text_detected_count']}\t"
            f"{row['no_text_detected_subset_percentage']:.3f}\t"
            f"{row['no_text_detected_whole_set_percentage']:.3f}"
        )
    return "\n".join(lines) + "\n"


def save_model_results(
    *,
    output_dir: Path,
    model_name: str,
    model_path: Path,
    labels: Sequence[str],
    topology: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    object_summaries = [
        row for row in summaries if row["mode"] == "any"
    ]
    read_summaries = [
        row for row in summaries if row["mode"] == "read"
    ]
    object_records = [
        row for row in records if row["mode"] == "any"
    ]
    read_records = [
        row for row in records if row["mode"] == "read"
    ]

    _write_rows_csv(
        output_dir / "summary.csv",
        _clean_summary_rows(object_summaries),
    )
    _write_rows_csv(
        output_dir / "read_summary.csv",
        _clean_summary_rows(read_summaries),
    )
    _write_rows_csv(
        output_dir / "benchmark_records.csv",
        records,
    )

    object_payload = {
        "model": model_name,
        "model_path": str(model_path),
        "labels": list(labels),
        "task": "object_recognition",
        "late_read_topology": topology,
        "summaries": object_summaries,
        "records": object_records,
    }
    read_payload = {
        "model": model_name,
        "model_path": str(model_path),
        "labels": list(labels),
        "task": "attack_text_reading",
        "late_read_topology": topology,
        "summaries": read_summaries,
        "records": read_records,
    }

    (output_dir / "results.json").write_text(
        json.dumps(object_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "read_results.json").write_text(
        json.dumps(read_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    object_text = format_tab_summary(
        model_name,
        "Object recognition",
        object_summaries,
    )
    read_text = format_tab_summary(
        model_name,
        "Attack-text reading",
        read_summaries,
    )

    for name in (
        "summary.txt",
        "benchmark_results_summary.txt",
    ):
        (output_dir / name).write_text(
            object_text,
            encoding="utf-8",
        )
    for name in (
        "read_summary.txt",
        "read_benchmark_results_summary.txt",
    ):
        (output_dir / name).write_text(
            read_text,
            encoding="utf-8",
        )


def _to_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if torch.is_tensor(value):
        return tuple(int(x) for x in value.detach().cpu().reshape(-1).tolist())
    return tuple(int(x) for x in value)


def audit_late_read_topology(
    model: Any,
    model_path: Path,
) -> dict[str, Any]:
    """Verify the production/native B20/B21 late-reader topology."""
    name = model_path.name or str(model_path)
    implant = model.read_implant

    tap_blocks = (
        tuple(int(x) for x in implant._block_list(implant.tap_blocks))
        if hasattr(implant, "_block_list")
        else _to_int_tuple(implant.tap_blocks)
    )
    if tap_blocks != EXPECTED_NATIVE_READ_TAPS:
        raise RuntimeError(
            f"{name}: read_tap_blocks={tap_blocks}, expected native {EXPECTED_NATIVE_READ_TAPS}"
        )

    # Current production models use the native taps directly and may not expose
    # the old experimental read_state_blocks field.  Historical native B20/B21
    # wrappers are also accepted when they explicitly record the same topology.
    config_state_blocks = _to_int_tuple(getattr(model.config, "read_state_blocks", None))
    implant_state_blocks = _to_int_tuple(getattr(implant, "read_state_blocks", None))
    effective_state_blocks = implant_state_blocks or config_state_blocks or tap_blocks

    for where, value in (
        ("config.read_state_blocks", config_state_blocks),
        ("read_implant.read_state_blocks", implant_state_blocks),
        ("effective READ states", effective_state_blocks),
    ):
        if value and value != EXPECTED_NATIVE_READ_TAPS:
            raise RuntimeError(
                f"{name}: {where}={value}, expected native {EXPECTED_NATIVE_READ_TAPS}"
            )

    read_weights = (
        implant._mix_weights(implant.read_tap_logits).detach().float().cpu().tolist()
        if hasattr(implant, "_mix_weights")
        else implant.read_tap_logits.detach().float().softmax(dim=0).cpu().tolist()
    )

    return {
        "read_tap_blocks": list(tap_blocks),
        "read_state_blocks": list(effective_state_blocks),
        "content_state_blocks": list(tap_blocks),
        "read_tap_weights": [float(x) for x in read_weights],
        "content_tap_weights": [float(x) for x in read_weights],
    }


# ================================================================================================
# Main
# ================================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Binary SCAM/RTA evaluation for saved post-training WiSE late-reader checkpoints. "
            "Normal saved-model forward only."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        required=True,
        help="One or more full HF ModeMUX model directories/repo ids.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use CUDA FP16 autocast for the ordinary backbone. "
            "PIECES remains FP32 via pieces_fp32=True."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configure_reproducibility(args.seed)

    print(f"[run] device={device}")
    print(f"[run] AMP={args.amp}")
    print(f"[run] output={output_root}")
    print("[run] NORMAL SAVED-MODEL FORWARD: native B20/B21; no runtime tap monkeypatch / no intervention")
    print("[run] models:")
    for model_path in args.models:
        print(f"  - {model_path}")

    print("[run] raw benchmark labels; binary metric only")
    sample_sets = load_samples()
    labels = label_vocabulary(sample_sets)
    whole_set_count = sum(
        len(values)
        for values in sample_sets.values()
    )
    print(
        f"[dataset] labels={len(labels)} images={whole_set_count}"
    )

    console_groups: list[
        tuple[str, str, list[dict[str, Any]]]
    ] = []
    all_model_summaries: list[dict[str, Any]] = []
    all_model_records: list[dict[str, Any]] = []
    root_metadata: list[dict[str, Any]] = []

    for model_path in args.models:
        model_path = Path(model_path)
        model_name = model_path.name
        model_output = output_root / model_name
        model_output.mkdir(parents=True, exist_ok=True)

        print()
        print("#" * 112)
        print(f"# MODEL: {model_name}")
        print(f"# PATH:  {model_path}")
        print("#" * 112)

        print(f"[model] Loading {model_path}")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).eval().to(device)
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        if not hasattr(model, "read_implant"):
            raise TypeError(
                f"{model_path} is not the expected full x-attention model: "
                "missing read_implant"
            )

        topology = audit_late_read_topology(model, model_path)
        print(
            "[model] late READ topology: "
            f"slots/content={topology['read_tap_blocks']} "
            f"READ states={topology['read_state_blocks']} "
            f"weights={topology['read_tap_weights']}"
        )
        print(
            "[model] correction/content remains native: "
            f"{topology['content_state_blocks']}"
        )

        source_blocks = [
            int(value)
            for value in model.read_implant.source_tap_blocks.detach().cpu().tolist()
        ]
        source_weights = [
            float(value)
            for value in (
                model.read_implant.source_tap_logits.detach()
                .float()
                .softmax(dim=0)
                .cpu()
                .tolist()
            )
        ]
        print(
            "[model] SOURCE unchanged: "
            + " ".join(
                f"B{block}={weight:.6f}"
                for block, weight in zip(source_blocks, source_weights)
            )
        )

        input_ids = encode_label_prompts(
            processor,
            labels,
            device,
        )

        summaries: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []

        for subset in SUBSET_ORDER:
            samples = sample_sets[subset]
            print(
                f"[benchmark] {model_name} {subset}: "
                f"{len(samples)} images"
            )

            for mode in ("any", "read"):
                logits = score_subset(
                    model,
                    processor,
                    samples,
                    input_ids,
                    mode=mode,
                    batch_size=args.batch_size,
                    device=device,
                    amp=args.amp,
                )
                summary, mode_records = evaluate_logits(
                    samples,
                    labels,
                    logits,
                    mode=mode,
                )
                summary["no_text_detected_whole_set_percentage"] = (
                    100.0
                    * summary["no_text_detected_count"]
                    / whole_set_count
                )
                summary["model"] = model_name
                summary["model_path"] = str(model_path)
                for record in mode_records:
                    record["model"] = model_name
                    record["model_path"] = str(model_path)

                summaries.append(summary)
                records.extend(mode_records)
                del logits

            # Incremental safety write after every subset.
            _write_rows_csv(
                model_output / "benchmark_summary_partial.csv",
                _clean_summary_rows(summaries),
            )
            _write_rows_csv(
                model_output / "benchmark_records_partial.csv",
                records,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

        object_summaries = [
            row for row in summaries if row["mode"] == "any"
        ]
        read_summaries = [
            row for row in summaries if row["mode"] == "read"
        ]

        metadata = {
            "model": model_name,
            "model_path": str(model_path),
            "device": str(device),
            "amp": bool(args.amp),
            "pieces_fp32": True,
            "correction": True,
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "label_policy": "raw benchmark labels; no canonicalization",
            "metric_policy": "binary only; raw benchmark labels",
            "late_read_topology": topology,
            "label_count": len(labels),
            "image_count": whole_set_count,
            "source_tap_blocks": source_blocks,
            "source_tap_weights": {
                str(block): weight
                for block, weight in zip(
                    source_blocks,
                    source_weights,
                )
            },
            "interventions": "NONE",
            "semantics": {
                "any": (
                    "Object recognition; object label is ground truth and attack word "
                    "is the binary competitor."
                ),
                "read_attack": (
                    "SCAM/SynthSCAM/RTA/SynthRTA: attack word is ground truth; "
                    "object and <null> are binary competitors."
                ),
                "read_no_attack": (
                    "For direct original-benchmark comparability, NoSCAM/NoRTA use "
                    "<null> as the forced-read goal with object and attack-word labels "
                    "as competitors."
                ),
            },
        }

        save_model_results(
            output_dir=model_output,
            model_name=model_name,
            model_path=model_path,
            labels=labels,
            topology=topology,
            summaries=summaries,
            records=records,
            metadata=metadata,
        )

        # Remove partial aliases after the complete files exist.
        for partial_name in (
            "benchmark_summary_partial.csv",
            "benchmark_records_partial.csv",
        ):
            partial_path = model_output / partial_name
            if partial_path.exists():
                partial_path.unlink()

        console_groups.append(
            (model_name, "Object recognition", object_summaries)
        )
        console_groups.append(
            (model_name, "Attack-text reading", read_summaries)
        )
        all_model_summaries.extend(summaries)
        all_model_records.extend(records)
        root_metadata.append(metadata)

        del input_ids, model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Consolidated machine-readable output.
    _write_rows_csv(
        output_root / "all_models_summary.csv",
        _clean_summary_rows(all_model_summaries),
    )
    _write_rows_csv(
        output_root / "all_models_records.csv",
        all_model_records,
    )
    (output_root / "all_models_results.json").write_text(
        json.dumps(
            {
                "models": root_metadata,
                "labels": labels,
                "summaries": all_model_summaries,
                "records": all_model_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Exact human-readable final tables, also saved verbatim.
    console_text = "\n\n".join(
        format_console_table(model_name, task_name, group)
        for model_name, task_name, group in console_groups
    ) + "\n"
    (output_root / "benchmark_results_summary.txt").write_text(
        console_text,
        encoding="utf-8",
    )

    print()
    print("=" * 112)
    print("ALL-DATASETS SUMMARY — NORMAL ZS, NO INTERVENTIONS")
    print("=" * 112)
    print(console_text, end="")

    print()
    print(f"[done] Results: {output_root}")
    print("[done] Consolidated files:")
    for name in (
        "benchmark_results_summary.txt",
        "all_models_summary.csv",
        "all_models_records.csv",
        "all_models_results.json",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
