"""SugarCrepe retrieval for HF exports and trusted OpenAI-format pickles.

Full PIECES models are evaluated in <notext>, <any>, <text>, and
<text>+<null> modes. --full_corr_off additionally repeats those modes with only
the separately trained CONTENT correction disabled. RN-correction exports are evaluated with correction off/on;
RN-only exports use their RN embedding. Outputs are isolated per model alias and
contain metrics/logs/plots only—dataset images are never copied.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow both `python -m benchmarks.sugarcrepe` and direct script execution.
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


COCO_IMAGE_ROOT = "path/to/COCO/val2017"
DATA_ROOT = "utils_datasets/sugar_crepe"
BENCHMARK_NAME = "sugarcrepe"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128 if DEVICE == "cuda" else 32
NUM_WORKERS = 4
AMP = True
SEED = 20260829

SPLITS: Dict[str, str] = {
    "add_obj": "add_obj.json",
    "add_att": "add_att.json",
    "replace_obj": "replace_obj.json",
    "replace_att": "replace_att.json",
    "replace_rel": "replace_rel.json",
    "swap_obj": "swap_obj.json",
    "swap_att": "swap_att.json",
}

FULL_MODE_LABELS = {
    "notext": "<notext>",
    "any": "<any>",
    "text": "<text>",
    "text_null": "<text>+<null>",
}


@dataclass(frozen=True)
class SugarCrepeItem:
    category: str
    key: str
    filename: str
    caption: str
    negative_caption: str


def load_sugar_crepe(data_root: str) -> Dict[str, List[SugarCrepeItem]]:
    buckets: Dict[str, List[SugarCrepeItem]] = {}
    for category, filename in SPLITS.items():
        path = Path(data_root) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing SugarCrepe JSON: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        buckets[category] = [
            SugarCrepeItem(
                category=category,
                key=str(key),
                filename=str(example["filename"]),
                caption=str(example["caption"]),
                negative_caption=str(example["negative_caption"]),
            )
            for key, example in raw.items()
        ]
    return buckets


def _open_image(path: Path, retries: int = 8) -> Image.Image:
    delay = 0.02
    last_error: BaseException | None = None
    for _ in range(retries):
        try:
            with path.open("rb") as handle:
                image = Image.open(handle)
                image.load()
            return image.convert("RGB")
        except (OSError, PermissionError) as error:
            last_error = error
            time.sleep(delay + random.random() * 0.01)
            delay *= 1.8
    assert last_error is not None
    raise last_error


class SugarCrepeDataset(Dataset):
    def __init__(self, items: Sequence[SugarCrepeItem], preprocess, image_root: Path):
        self.items = list(items)
        self.preprocess = preprocess
        # Store the resolved dataset path on the Dataset instance. On Windows,
        # DataLoader workers use spawn and re-import this module; runtime-mutated
        # module globals are therefore not inherited by worker processes.
        self.image_root = Path(image_root)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = _open_image(self.image_root / item.filename)
        return self.preprocess(image), item.caption, item.negative_caption


def _collate(batch):
    images, positives, negatives = zip(*batch)
    return torch.stack(images), list(positives), list(negatives)


def _loader(items: Sequence[SugarCrepeItem], preprocess, image_root: Path) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE == "cuda",
        "collate_fn": _collate,
    }
    if NUM_WORKERS:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(SugarCrepeDataset(items, preprocess, image_root), **kwargs)


def _empty_accumulator(mode_labels: Dict[str, str]) -> dict[str, dict[str, Any]]:
    return {
        key: {"label": label, "correct": 0, "count": 0, "margins": [], "null_top1": 0}
        for key, label in mode_labels.items()
    }


@torch.inference_mode()
def evaluate_full_xattn(
    model,
    loader: DataLoader,
    description: str,
    *,
    include_corr_off: bool = False,
):
    variants = full_xattn_correction_variants(include_corr_off)
    mode_specs: dict[str, tuple[str, bool]] = {}
    for variant in variants:
        for base_key, base_label in FULL_MODE_LABELS.items():
            key = full_xattn_variant_mode_key(base_key, variant)
            label = full_xattn_variant_mode_label(base_label, variant)
            mode_specs[key] = (label, variant.apply_content_correction)

    accumulator = _empty_accumulator(
        {key: label for key, (label, _corr) in mode_specs.items()}
    )
    for key, (_label, apply_corr) in mode_specs.items():
        accumulator[key]["content_correction"] = bool(apply_corr)

    for images, positives, negatives in tqdm(loader, desc=description, ncols=100):
        images = images.to(DEVICE, non_blocking=DEVICE == "cuda")
        batch = len(positives)
        prompts = (
            [f"<notext> {value}" for value in positives]
            + [f"<notext> {value}" for value in negatives]
            + [f"<any> {value}" for value in positives]
            + [f"<any> {value}" for value in negatives]
            + [f"<text> {value}" for value in positives]
            + [f"<text> {value}" for value in negatives]
            + ["<text> <null>"]
        )
        tokens = clip.tokenize(prompts, truncate=True).to(DEVICE)

        for variant in variants:
            with inference_autocast(DEVICE, AMP):
                output = model.forward_modes(
                    images,
                    tokens,
                    apply_content_correction=variant.apply_content_correction,
                    return_details=False,
                )
            logits = output[0] if isinstance(output, tuple) else output["logits_per_image"]
            logits = logits.detach().float().cpu()
            rows = torch.arange(batch)
            pairs = {
                "notext": (logits[rows, rows], logits[rows, batch + rows]),
                "any": (logits[rows, 2 * batch + rows], logits[rows, 3 * batch + rows]),
                "text": (logits[rows, 4 * batch + rows], logits[rows, 5 * batch + rows]),
            }
            null_score = logits[:, 6 * batch]

            for base_key, (positive, negative) in pairs.items():
                key = full_xattn_variant_mode_key(base_key, variant)
                margins = positive - negative
                accumulator[key]["correct"] += int((margins >= 0).sum())
                accumulator[key]["count"] += batch
                accumulator[key]["margins"].extend(margins.tolist())

            positive, negative = pairs["text"]
            key = full_xattn_variant_mode_key("text_null", variant)
            margins = positive - torch.maximum(negative, null_score)
            accumulator[key]["correct"] += int((margins >= 0).sum())
            accumulator[key]["count"] += batch
            accumulator[key]["margins"].extend(margins.tolist())
            accumulator[key]["null_top1"] += int(
                ((null_score > positive) & (null_score > negative)).sum()
            )

    return _summarize(accumulator)


@torch.inference_mode()
def evaluate_separable(model, info, loader: DataLoader, description: str):
    modes = separable_modes(model, info)
    accumulator = _empty_accumulator({mode.key: mode.label for mode in modes})
    for images, positives, negatives in tqdm(loader, desc=description, ncols=100):
        images = images.to(DEVICE, non_blocking=DEVICE == "cuda")
        tokens = clip.tokenize(positives + negatives, truncate=True).to(DEVICE)
        with inference_autocast(DEVICE, AMP):
            text = normalized_text_features(model, tokens)
            scale = model.logit_scale.detach().float().exp()
            for mode in modes:
                image = normalized_image_features(model, images, mode)
                logits = scale * image @ text.t()
                batch = len(positives)
                rows = torch.arange(batch, device=logits.device)
                margins = logits[rows, rows] - logits[rows, batch + rows]
                accumulator[mode.key]["correct"] += int((margins >= 0).sum())
                accumulator[mode.key]["count"] += batch
                accumulator[mode.key]["margins"].extend(margins.float().cpu().tolist())
    return _summarize(accumulator)


def _summarize(accumulator: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for key, values in accumulator.items():
        count = int(values["count"])
        margins = values["margins"]
        result[key] = {
            "mode_label": values["label"],
            "content_correction": values.get("content_correction"),
            "n_total": count,
            "n_correct": int(values["correct"]),
            "accuracy": values["correct"] / count if count else 0.0,
            "margin_mean": sum(margins) / len(margins) if margins else 0.0,
            "null_top1_count": int(values["null_top1"]),
            "null_top1_rate": values["null_top1"] / count if count else 0.0,
        }
    return result


def save_plot(rows: pd.DataFrame, alias: str, output_path: Path) -> None:
    modes = list(dict.fromkeys(rows["mode"].tolist()))
    categories = list(SPLITS)
    width = 0.8 / max(1, len(modes))
    x = torch.arange(len(categories)).numpy()
    fig, ax = plt.subplots(figsize=(max(11.0, 8.0 + 1.1 * len(modes)), 5.8))
    colors = plt.get_cmap("Set2")
    for index, mode in enumerate(modes):
        subset = rows[rows["mode"] == mode].set_index("category")
        values = [float(subset.loc[category, "accuracy"]) for category in categories]
        label = str(subset.iloc[0]["mode_label"])
        offset = (index - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label, color=colors(index))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Binary positive-vs-negative accuracy")
    ax.set_xticks(x, categories, rotation=25, ha="right")
    ax.set_title(f"SugarCrepe retrieval — {alias}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: pd.DataFrame, alias: str, path: Path) -> None:
    lines = [f"SugarCrepe retrieval — {alias}", "", "mode\tcategory\taccuracy\tmean_margin\tn"]
    for row in rows.itertuples(index=False):
        lines.append(
            f"{row.mode_label}\t{row.category}\t{row.accuracy:.6f}\t"
            f"{row.margin_mean:+.6f}\t{row.n_total}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SugarCrepe retrieval benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--base-model-or-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("out_bench_results/sugarcrepe"))
    parser.add_argument("--coco-image-root", type=Path, default=Path(COCO_IMAGE_ROOT))
    parser.add_argument("--data-root", type=Path, default=Path(DATA_ROOT))
    parser.add_argument(
        "--full_corr_off",
        "--full-corr-off",
        dest="full_corr_off",
        action="store_true",
        help=(
            "For full x-attention models, also run every mode with only the "
            "separate CONTENT correction disabled. RN and reading/routing stay on."
        ),
    )
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    image_root = args.coco_image_root.resolve()
    data_root = args.data_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"COCO val2017 image directory does not exist: {image_root}")
    buckets = load_sugar_crepe(str(data_root))
    # Fail in the parent process with the real configured path instead of
    # surfacing an opaque DataLoader-worker traceback after model loading.
    first_item = next((items[0] for items in buckets.values() if items), None)
    if first_item is None:
        raise RuntimeError(f"SugarCrepe annotations contain no examples: {data_root}")
    first_image = image_root / first_item.filename
    if not first_image.is_file():
        raise FileNotFoundError(
            f"SugarCrepe COCO image not found: {first_image}\n"
            f"Configured --coco-image-root: {image_root}"
        )
    print(f"[device] {DEVICE}; models=1")

    spec = ModelSpec(args.model_alias, args.model, args.base_model_or_path)
    for spec in (spec,):
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[model] {spec.alias}: {spec.path}")
        model, preprocess, info = load_model_spec(clip, spec, device=DEVICE)
        print(f"[loader] family={info.model_family} source={info.source_kind}")
        rows = []
        metrics: dict[str, Any] = {}
        for category, items in buckets.items():
            loader = _loader(items, preprocess, image_root)
            if is_full_xattn(model, info):
                summary = evaluate_full_xattn(
                    model,
                    loader,
                    f"{spec.alias} | {category}",
                    include_corr_off=args.full_corr_off,
                )
            else:
                summary = evaluate_separable(model, info, loader, f"{spec.alias} | {category}")
            metrics[category] = summary
            for mode, values in summary.items():
                row = {"model_alias": spec.alias, "model": spec.path, "category": category, "mode": mode, **values}
                rows.append(row)
                print(
                    f"  {category:12s} {values['mode_label']:22s} "
                    f"acc={values['accuracy']:.4f} margin={values['margin_mean']:+.4f}"
                )

        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "summary.csv", index=False)
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "model": spec.to_dict(),
                    "model_family": info.model_family,
                    "full_corr_off": bool(args.full_corr_off),
                    "dataset_paths": {"images": str(image_root), "annotations": str(data_root)},
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        display_name = model_display_name(spec)
        write_summary(frame, display_name, output_dir / "summary.txt")
        save_plot(
            frame,
            display_name,
            output_dir / f"accuracy_{model_file_token(spec)}.png",
        )
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
