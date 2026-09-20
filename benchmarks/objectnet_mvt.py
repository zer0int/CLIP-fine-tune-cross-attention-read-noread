"""ObjectNet MVT zero-shot evaluation for HF and trusted pickle models.

Full PIECES models retain the four useful scoring views: <notext>, <any>,
<text>, and the R-N read-evidence score. --full_corr_off additionally repeats
those views with only the separately trained CONTENT correction disabled. Other model families expose only their
meaningful RN/correction modes. The script saves tabular logs and plots, never
copies dataset images. Dataset labels are used exactly as provided; no semantic label remapping is applied.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow both `python -m benchmarks.objectnet_mvt` and direct script execution.
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


CSV_FILE = "utils_datasets/mvt/human_responses_dedup.csv"
IMAGE_FOLDER = "path/to/objectnet.dev/MVT/data_release_2023/all/"
BENCHMARK_NAME = "objectnet_mvt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = True
BATCH_SIZE = 12 if DEVICE == "cuda" else 2
NUM_WORKERS = 4
SEED = 20260829

NOTEXT_TEMPLATE = "<notext> {}"
ANY_TEMPLATE = "<any> {}"
READ_TEMPLATE = "<text> {}"
NULL_PROMPT = "<text> <null>"


def configure_reproducibility() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CroppedImageCSVFileDataset(Dataset):
    def __init__(self, csv_file: str, image_folder: str, transform):
        frame = pd.read_csv(csv_file)
        self.image_names = frame["image"].astype(str).tolist()
        self.labels = frame["label"].astype(str).tolist()
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        path = Path(self.image_folder) / image_name
        delay = 0.02
        last_error: BaseException | None = None
        for _ in range(10):
            try:
                with path.open("rb") as handle:
                    image = Image.open(handle)
                    image.load()
                return self.transform(image.convert("RGB")), self.labels[index], image_name
            except (OSError, PermissionError) as error:
                last_error = error
                time.sleep(delay + random.random() * 0.01)
                delay = min(0.5, delay * 2)
        assert last_error is not None
        raise last_error


def collate_batch(batch):
    images, labels, names = zip(*batch)
    return torch.stack(images), list(labels), list(names)


def make_loader(preprocess) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE == "cuda",
        "collate_fn": collate_batch,
    }
    if NUM_WORKERS:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        CroppedImageCSVFileDataset(CSV_FILE, IMAGE_FOLDER, preprocess), **kwargs
    )


def build_labels() -> tuple[List[str], Dict[str, int]]:
    frame = pd.read_csv(CSV_FILE, usecols=["label"])
    labels = list(dict.fromkeys(frame["label"].astype(str).tolist()))
    return labels, {label: index for index, label in enumerate(labels)}


def full_tokens(labels: Sequence[str]) -> torch.Tensor:
    prompts = (
        [NOTEXT_TEMPLATE.format(label) for label in labels]
        + [ANY_TEMPLATE.format(label) for label in labels]
        + [READ_TEMPLATE.format(label) for label in labels]
        + [NULL_PROMPT]
    )
    return clip.tokenize(prompts, truncate=True).to(DEVICE)


def _score_rows(
    scores: torch.Tensor,
    mode: str,
    mode_label: str,
    batch_labels: Sequence[str],
    image_names: Sequence[str],
    labels: Sequence[str],
    label_to_index: Dict[str, int],
    extras: dict[str, torch.Tensor] | None = None,
    content_correction: bool | None = None,
) -> list[dict[str, Any]]:
    rows = []
    scores = scores.detach().float().cpu()
    extras = extras or {}
    for index, (target, image_name) in enumerate(zip(batch_labels, image_names)):
        target_index = label_to_index[target]
        vector = scores[index]
        prediction_index = int(vector.argmax())
        target_score = float(vector[target_index])
        runner_up = float(torch.cat((vector[:target_index], vector[target_index + 1 :])).max())
        row = {
            "image": image_name,
            "target": target,
            "prediction": labels[prediction_index],
            "mode": mode,
            "mode_label": mode_label,
            "content_correction": content_correction,
            "correct": prediction_index == target_index,
            "target_score": target_score,
            "target_margin": target_score - runner_up,
        }
        for name, values in extras.items():
            row[name] = float(values[index].detach().float().cpu())
        rows.append(row)
    return rows


@torch.inference_mode()
def evaluate_full(
    model,
    loader,
    labels,
    label_to_index,
    *,
    include_corr_off: bool = False,
):
    tokens = full_tokens(labels)
    count = len(labels)
    rows = []
    variants = full_xattn_correction_variants(include_corr_off)

    for images, batch_labels, image_names in tqdm(loader, desc="ObjectNet MVT", ncols=100):
        images = images.to(DEVICE, non_blocking=DEVICE == "cuda")
        for variant in variants:
            with inference_autocast(DEVICE, AMP):
                output = model.forward_modes(
                    images,
                    tokens,
                    apply_content_correction=variant.apply_content_correction,
                    return_details=True,
                )
            scale = model.logit_scale.detach().float().exp().clamp_min(1e-12)
            logits = output["logits_per_image"].float() / scale
            notext = logits[:, :count]
            any_scores = logits[:, count : 2 * count]
            read = logits[:, 2 * count : 3 * count]
            read_delta = read - notext
            null = logits[:, 3 * count]
            source = output["source_logits"].float().sigmoid()
            shared = {
                "text_present_probability": source[:, 0],
                "text_readable_probability": source[:, 1],
                "null_score": null,
            }
            for base_key, base_label, values in (
                ("notext", "<notext>", notext),
                ("any", "<any>", any_scores),
                ("read", "<text>", read),
                ("read_delta", "R-N", read_delta),
            ):
                key = full_xattn_variant_mode_key(base_key, variant)
                label = full_xattn_variant_mode_label(base_label, variant)
                rows.extend(
                    _score_rows(
                        values,
                        key,
                        label,
                        batch_labels,
                        image_names,
                        labels,
                        label_to_index,
                        shared,
                        content_correction=variant.apply_content_correction,
                    )
                )
    return pd.DataFrame(rows)


@torch.inference_mode()
def evaluate_separable(model, info, loader, labels, label_to_index):
    tokens = clip.tokenize(labels, truncate=True).to(DEVICE)
    with inference_autocast(DEVICE, AMP):
        text = normalized_text_features(model, tokens)
    modes = separable_modes(model, info)
    rows = []
    for images, batch_labels, image_names in tqdm(loader, desc="ObjectNet MVT", ncols=100):
        images = images.to(DEVICE, non_blocking=DEVICE == "cuda")
        with inference_autocast(DEVICE, AMP):
            for mode in modes:
                image = normalized_image_features(model, images, mode)
                scores = image @ text.t()
                rows.extend(
                    _score_rows(
                        scores,
                        mode.key,
                        mode.label,
                        batch_labels,
                        image_names,
                        labels,
                        label_to_index,
                    )
                )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, subset in frame.groupby("mode", sort=False):
        rows.append(
            {
                "mode": mode,
                "mode_label": subset.iloc[0]["mode_label"],
                "content_correction": subset.iloc[0].get("content_correction"),
                "n": len(subset),
                "accuracy": float(subset["correct"].mean()),
                "mean_target_margin": float(subset["target_margin"].mean()),
                "mean_text_present_probability": (
                    float(subset["text_present_probability"].mean())
                    if "text_present_probability" in subset
                    else float("nan")
                ),
                "mean_text_readable_probability": (
                    float(subset["text_readable_probability"].mean())
                    if "text_readable_probability" in subset
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def save_plot(summary: pd.DataFrame, alias: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(summary) * 1.5), 5.5))
    bars = ax.bar(
        summary["mode_label"], summary["accuracy"], color=plt.get_cmap("Set2").colors[: len(summary)]
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(f"ObjectNet MVT zero-shot — {alias}")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    global CSV_FILE, IMAGE_FOLDER
    parser = argparse.ArgumentParser(description="ObjectNet MVT zero-shot benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--base-model-or-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("out_bench_results/objectnet_mvt"))
    parser.add_argument("--csv-file", type=Path, default=Path(CSV_FILE))
    parser.add_argument("--image-folder", type=Path, default=Path(IMAGE_FOLDER))
    parser.add_argument(
        "--full_corr_off",
        "--full-corr-off",
        dest="full_corr_off",
        action="store_true",
        help=(
            "For full x-attention models, also run all scoring modes with only "
            "the separate CONTENT correction disabled. RN and reading/routing stay on."
        ),
    )
    args = parser.parse_args()
    CSV_FILE = str(args.csv_file)
    IMAGE_FOLDER = str(args.image_folder)

    configure_reproducibility()
    labels, label_to_index = build_labels()
    print(f"[dataset] raw labels={len(labels)} canonicalization=OFF")
    spec = ModelSpec(args.model_alias, args.model, args.base_model_or_path)
    for spec in (spec,):
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[model] {spec.alias}: {spec.path}")
        model, preprocess, info = load_model_spec(clip, spec, device=DEVICE)
        print(f"[loader] family={info.model_family} source={info.source_kind}")
        loader = make_loader(preprocess)
        if is_full_xattn(model, info):
            frame = evaluate_full(
                model,
                loader,
                labels,
                label_to_index,
                include_corr_off=args.full_corr_off,
            )
        else:
            frame = evaluate_separable(model, info, loader, labels, label_to_index)
        frame.insert(0, "model_alias", spec.alias)
        frame.insert(1, "model", spec.path)
        summary = summarize(frame)
        frame.to_csv(output_dir / "objectnet_mvt_all_samples.csv", index=False)
        summary.to_csv(output_dir / "summary.csv", index=False)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "model": spec.to_dict(),
                    "model_family": info.model_family,
                    "full_corr_off": bool(args.full_corr_off),
                    "labels": labels,
                    "metrics": summary.to_dict(orient="records"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        lines = [f"ObjectNet MVT — {spec.alias}", "", "mode\taccuracy\tmean_margin\tn"]
        for row in summary.itertuples(index=False):
            lines.append(
                f"{row.mode_label}\t{row.accuracy:.6f}\t{row.mean_target_margin:+.6f}\t{row.n}"
            )
            print(lines[-1])
        (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        save_plot(
            summary,
            model_display_name(spec),
            output_dir / f"accuracy_{model_file_token(spec)}.png",
        )
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
