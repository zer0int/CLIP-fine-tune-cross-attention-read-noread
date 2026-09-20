"""Unified SCAM + RTA-100 typo benchmark for HF and trusted pickle models.

All configured checkpoints are loaded through load_openai_clip_anything and
therefore expose one OpenAI/CLIP-style interface. Full PIECES checkpoints run
separate <any> and <text>+<null> lanes; --full_corr_off additionally repeats
those full-model lanes with only the separately trained CONTENT correction disabled.
RN correction checkpoints run correction off/on; RN-only checkpoints run their RN lane.

Evaluation is deliberately binary-only and uses the raw benchmark labels exactly as
provided. There is no semantic alias/canonicalization layer and no all-label accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow both `python -m benchmarks.typo` and direct script execution.
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


BENCHMARK_NAME = "typo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = True
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 20260829
PROMPT = "a photo of a {}"

SUBSET_ORDER = ("NoSCAM", "SCAM", "SynthSCAM", "NoRTA", "RTA", "SynthRTA")
SUBSET_COLORS = {
    "NoSCAM": "#8BCF68",
    "NoRTA": "#8BCF68",
    "SCAM": "#F2AD63",
    "RTA": "#F2AD63",
    "SynthSCAM": "#EA7B7B",
    "SynthRTA": "#EA7B7B",
}


@dataclass(frozen=True)
class PairSample:
    image: Any
    correct_label: str
    distractor_label: str
    sample_id: str
    subset: str


def configure_reproducibility() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_samples(hf_cache_dir: str | None = None) -> Dict[str, List[PairSample]]:
    result: Dict[str, List[PairSample]] = {name: [] for name in SUBSET_ORDER}
    cache_kwargs = {"cache_dir": hf_cache_dir} if hf_cache_dir else {}
    scam = load_dataset("BLISS-e-V/SCAM", split="train", **cache_kwargs)
    for row in scam:
        sample_id = str(row["id"])
        subset = next((name for name in SUBSET_ORDER[:3] if sample_id.startswith(name)), None)
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
    rta = load_dataset("zer0int/RTA-100-Triplet", split="train", **cache_kwargs)
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
    return result


class ImageDataset(Dataset):
    def __init__(self, samples: Sequence[PairSample], preprocess):
        self.samples = list(samples)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.preprocess(self.samples[index].image.convert("RGB")), index


def make_loader(samples: Sequence[PairSample], preprocess):
    return DataLoader(
        ImageDataset(samples, preprocess),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )


def label_vocabulary(sample_sets: Dict[str, List[PairSample]]) -> list[str]:
    return list(
        dict.fromkeys(
            label
            for subset in SUBSET_ORDER
            for sample in sample_sets[subset]
            for label in (sample.correct_label, sample.distractor_label)
        )
    )


def full_modes(
    include_corr_off: bool = False,
) -> tuple[tuple[str, str, str, bool], ...]:
    modes = []
    for variant in full_xattn_correction_variants(include_corr_off):
        for base_key, base_label in (("any", "<any>"), ("read", "<text>+<null>")):
            modes.append(
                (
                    full_xattn_variant_mode_key(base_key, variant),
                    full_xattn_variant_mode_label(base_label, variant),
                    base_key,
                    variant.apply_content_correction,
                )
            )
    return tuple(modes)


@torch.inference_mode()
def score_full(
    model,
    loader,
    labels: Sequence[str],
    base_mode: str,
    *,
    apply_content_correction: bool,
    description: str,
):
    prefix = "<any>" if base_mode == "any" else "<text>"
    prompts = [f"{prefix} {PROMPT.format(label)}" for label in labels]
    if base_mode == "read":
        prompts.append("<text> <null>")
    tokens = clip.tokenize(prompts, truncate=True).to(DEVICE)
    chunks = []
    for images, _indices in tqdm(loader, desc=f"score {description}", leave=False):
        images = images.to(DEVICE, non_blocking=DEVICE == "cuda")
        with inference_autocast(DEVICE, AMP):
            output = model.forward_modes(
                images,
                tokens,
                apply_content_correction=apply_content_correction,
                return_details=False,
            )
        logits = output[0] if isinstance(output, tuple) else output["logits_per_image"]
        chunks.append(logits.detach().float().cpu())
    return torch.cat(chunks)


@torch.inference_mode()
def score_separable(model, loader, labels: Sequence[str], mode):
    tokens = clip.tokenize([PROMPT.format(label) for label in labels], truncate=True).to(DEVICE)
    with inference_autocast(DEVICE, AMP):
        text = normalized_text_features(model, tokens)
    chunks = []
    for images, _indices in tqdm(loader, desc=f"score {mode.key}", leave=False):
        images = images.to(DEVICE, non_blocking=DEVICE == "cuda")
        with inference_autocast(DEVICE, AMP):
            image = normalized_image_features(model, images, mode)
            chunks.append((model.logit_scale.detach().float().exp() * image @ text.t()).cpu())
    return torch.cat(chunks)


def evaluate_logits(
    samples: Sequence[PairSample],
    labels: Sequence[str],
    logits: torch.Tensor,
    *,
    mode: str,
    mode_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate only the benchmark-defined binary decision.

    The shared raw-label bank is an efficient scoring cache. Correctness is never
    defined against that bank: each image is judged only against its paired object /
    attack labels (and explicit <null> where the forced-reader protocol requires it).
    """
    label_to_index = {label: index for index, label in enumerate(labels)}
    has_null = logits.shape[1] == len(labels) + 1
    null_index = len(labels)
    no_attack_subset = bool(samples and samples[0].subset.startswith("No"))
    reading_task = mode.startswith("read")
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
                raise RuntimeError("semantic mode unexpectedly exposed a null candidate")
            goal_index = object_index
            competitors = [attack_index]

        competitor_index = max(
            competitors,
            key=lambda index: float(vector[index]),
        )
        binary_margin = float(vector[goal_index] - vector[competitor_index])
        binary_is_correct = binary_margin >= 0.0
        binary_correct += int(binary_is_correct)
        binary_margins.append(binary_margin)

        # NULL top-1 is retained as reader telemetry only. It is not an all-label
        # correctness metric and never changes binary accuracy.
        global_prediction = int(vector.argmax())
        null_selected = bool(has_null and global_prediction == null_index)
        null_count += int(null_selected)

        goal = (
            "NO_TEXT_DETECTED"
            if goal_index == null_index
            else labels[goal_index]
        )
        binary_prediction = (
            goal
            if binary_is_correct
            else (
                "NO_TEXT_DETECTED"
                if competitor_index == null_index
                else labels[competitor_index]
            )
        )

        records.append(
            {
                "id": sample.sample_id,
                "subset": sample.subset,
                "mode": mode,
                "mode_label": mode_label,
                "task": "attack_text_reading" if reading_task else "object_recognition",
                "goal": goal,
                "correct_label": sample.correct_label,
                "distractor_label": sample.distractor_label,
                "binary_prediction": binary_prediction,
                "binary_correct": bool(binary_is_correct),
                "binary_margin": binary_margin,
                "null_selected": null_selected,
            }
        )

    count = len(samples)
    summary = {
        "mode": mode,
        "mode_label": mode_label,
        "count": count,
        "binary_accuracy": binary_correct / count if count else 0.0,
        "mean_binary_logit_margin": sum(binary_margins) / count if count else 0.0,
        "no_text_detected_count": null_count,
        "no_text_detected_subset_percentage": 100.0 * null_count / count if count else 0.0,
    }
    return summary, records


def save_plot(
    summary: pd.DataFrame,
    display_name: str,
    metric: str,
    path: Path,
    *,
    reading_task: bool = False,
) -> None:
    modes = list(dict.fromkeys(summary["mode"].tolist()))
    width = 0.8 / len(modes)
    x = np.arange(len(SUBSET_ORDER))
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for mode_index, mode in enumerate(modes):
        subset = summary[summary["mode"] == mode].set_index("subset")
        values = [float(subset.loc[name, metric]) for name in SUBSET_ORDER]
        offset = (mode_index - (len(modes) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=str(subset.iloc[0]["mode_label"]),
            color=("white" if mode_index == 0 and len(modes) > 1 else [SUBSET_COLORS[name] for name in SUBSET_ORDER]),
            edgecolor=[SUBSET_COLORS[name] for name in SUBSET_ORDER],
            linewidth=1.5,
        )
        for bar, name in zip(bars, SUBSET_ORDER):
            bar.set_hatch("o" if "SCAM" in name else "//")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x, SUBSET_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    if reading_task:
        label = "Binary attack-text"
    else:
        label = "Binary object-vs-attack"
    ax.set_title(f"{label} accuracy — {display_name}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified SCAM + RTA-100 benchmark for HF and trusted pickle models"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--base-model-or-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("out_bench_results/typo"))
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument(
        "--run_vanilla",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For RN-only models, also score the same backbone with RN removed.",
    )
    parser.add_argument(
        "--full_corr_off",
        "--full-corr-off",
        dest="full_corr_off",
        action="store_true",
        help=(
            "For full x-attention models, also score the same trained model with "
            "only the separate CONTENT correction disabled. RN and all reading/"
            "routing modules remain enabled."
        ),
    )
    args = parser.parse_args()
    configure_reproducibility()
    sample_sets = load_samples(args.hf_cache_dir)
    labels = label_vocabulary(sample_sets)
    whole_set_count = sum(len(values) for values in sample_sets.values())
    print(f"[dataset] raw_labels={len(labels)} images={whole_set_count} metric=binary-only canonicalization=OFF")
    console_groups = []

    spec = ModelSpec(args.model_alias, args.model, args.base_model_or_path)
    for spec in (spec,):
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        model, preprocess, info = load_model_spec(clip, spec, device=DEVICE)
        print(f"\n[model] {spec.alias}: family={info.model_family} source={info.source_kind}")
        summaries = []
        all_records = []
        for subset in SUBSET_ORDER:
            samples = sample_sets[subset]
            loader = make_loader(samples, preprocess)
            if is_full_xattn(model, info):
                modes = full_modes(args.full_corr_off)
                scored = [
                    (
                        key,
                        label,
                        score_full(
                            model,
                            loader,
                            labels,
                            base_mode,
                            apply_content_correction=apply_corr,
                            description=label,
                        ),
                    )
                    for key, label, base_mode, apply_corr in modes
                ]
            else:
                modes = list(separable_modes(model, info))
                if info.model_family == "rn_token" and not args.run_vanilla:
                    modes = [mode for mode in modes if mode.key != "vanilla"]
                scored = [
                    (mode.key, mode.label, score_separable(model, loader, labels, mode))
                    for mode in modes
                ]
            for mode, mode_label, logits in scored:
                summary, records = evaluate_logits(
                    samples, labels, logits, mode=mode, mode_label=mode_label
                )
                summary["subset"] = subset
                summary["no_text_detected_whole_set_percentage"] = (
                    100.0 * summary["no_text_detected_count"] / whole_set_count
                )
                summaries.append(summary)
                all_records.extend(records)
                print(
                    f"  {subset:10s} {mode_label:22s} "
                    f"binary={summary['binary_accuracy']:.4f} "
                    f"margin={summary['mean_binary_logit_margin']:+.4f} "
                    f"NO_TEXT={summary['no_text_detected_count']}"
                )

        groups = [("", [row for row in summaries if not row["mode"].startswith("read")])]
        read_summaries = [row for row in summaries if row["mode"].startswith("read")]
        if read_summaries:
            groups.append(("read_", read_summaries))
        display_name = model_display_name(spec)
        model_name = model_file_token(spec)
        for prefix, group_summaries in groups:
            if not group_summaries:
                continue
            console_groups.append(
                (display_name, "Attack-text reading" if prefix else "Object recognition", group_summaries)
            )
            group_modes = {row["mode"] for row in group_summaries}
            group_records = [row for row in all_records if row["mode"] in group_modes]
            frame = pd.DataFrame(
                [
                    {key: value for key, value in row.items() if key != "failures"}
                    for row in group_summaries
                ]
            )
            frame.to_csv(output_dir / f"{prefix}summary.csv", index=False)
            (output_dir / f"{prefix}results.json").write_text(
                json.dumps(
                    {
                        "model": spec.to_dict(),
                        "model_family": info.model_family,
                        "full_corr_off": bool(args.full_corr_off),
                        "metric_policy": "binary only; raw benchmark labels; no canonicalization",
                        "raw_label_bank": labels,
                        "task": (
                            "attack_text_reading" if prefix else "object_recognition"
                        ),
                        "summaries": group_summaries,
                        "records": group_records,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            lines = [
                f"SCAM + RTA-100 — {display_name}",
                ("Attack-text reading" if prefix else "Object recognition"),
                "",
                "subset\tmode\tbinary_acc\tmargin\tNO_TEXT\tNO_TEXT_subset_pct\tNO_TEXT_whole_pct",
            ]
            for row in group_summaries:
                lines.append(
                    f"{row['subset']}\t{row['mode_label']}\t{row['binary_accuracy']:.6f}\t"
                    f"{row['mean_binary_logit_margin']:+.6f}\t"
                    f"{row['no_text_detected_count']}\t"
                    f"{row['no_text_detected_subset_percentage']:.3f}\t"
                    f"{row['no_text_detected_whole_set_percentage']:.3f}"
                )
            summary_text = "\n".join(lines) + "\n"
            (output_dir / f"{prefix}summary.txt").write_text(
                summary_text, encoding="utf-8"
            )
            (output_dir / f"{prefix}benchmark_results_summary.txt").write_text(
                summary_text, encoding="utf-8"
            )
            save_plot(
                frame,
                display_name,
                "binary_accuracy",
                output_dir / f"{prefix}acc_binary_{model_name}.png",
                reading_task=bool(prefix),
            )
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 112)
    print("ALL-DATASETS SUMMARY")
    print("=" * 112)
    for display_name, task_name, group_summaries in console_groups:
        print(f"\n{display_name} — {task_name}")
        print(
            f"{'Subset':<12} {'Mode':<22} {'Binary acc':>10} "
            f"{'Logit margin':>13} {'NO_TEXT':>9} {'Subset %':>10} {'Whole %':>9}"
        )
        print("-" * 112)
        for row in group_summaries:
            print(
                f"{row['subset']:<12} {row['mode_label']:<22} "
                f"{row['binary_accuracy']:>10.4f} "
                f"{row['mean_binary_logit_margin']:>+13.6f} "
                f"{row['no_text_detected_count']:>9d} "
                f"{row['no_text_detected_subset_percentage']:>9.3f}% "
                f"{row['no_text_detected_whole_set_percentage']:>8.3f}%"
            )


if __name__ == "__main__":
    main()
