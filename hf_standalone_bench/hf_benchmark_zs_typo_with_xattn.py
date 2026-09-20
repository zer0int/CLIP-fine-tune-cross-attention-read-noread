from __future__ import annotations

import argparse
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


# ==================================================================================================
# Models
# ==================================================================================================

OAI_MODEL = "openai/clip-vit-large-patch14"
GMP_MODEL = "zer0int/CLIP-GmP-ViT-L-14"
VPT_MODEL = "zer0int/CLIP-ViT-L-14-Universal-VPT-ReadNull-Token"
MUX_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"

MODEL_ALIASES = {
    "OAI": OAI_MODEL,
    "GmP": GMP_MODEL,
    "VPT": VPT_MODEL,
    "MUX": MUX_MODEL,
}

RN_FILENAME = "read_null_token.safetensors"
SUPPORTED_IMAGE_SIZES = (224, 336)

"""
NOTE!
The MUX checkpoint uses custom Hugging Face Transformers configuration/modeling code for
its cross-attention Read/NoRead architecture. Loading it therefore requires
``trust_remote_code=True``. This script enables that only for MUX; OAI, GmP, and VPT use
the stock CLIP classes. As usual with ``trust_remote_code=True``, review/pin the repository
revision if you need a fixed executable snapshot.
"""


# ==================================================================================================
# VPT / RN adapter
# ==================================================================================================


def _vision_model(model: Any):
    candidate = getattr(model, "vision_model", model)
    if hasattr(candidate, "vision_model"):
        candidate = candidate.vision_model
    if not all(hasattr(candidate, name) for name in ("embeddings", "encoder")):
        raise TypeError(
            "Expected a CLIP-like model exposing vision_model.embeddings and "
            "vision_model.encoder"
        )
    return candidate


def _architecture(vision_model: Any) -> dict[str, int]:
    config = vision_model.config
    image_size = config.image_size
    patch_size = config.patch_size
    if isinstance(image_size, (list, tuple)):
        if len(set(image_size)) != 1:
            raise ValueError(f"Only square CLIP inputs are supported, got {image_size}")
        image_size = image_size[0]
    if isinstance(patch_size, (list, tuple)):
        if len(set(patch_size)) != 1:
            raise ValueError(f"Only square patches are supported, got {patch_size}")
        patch_size = patch_size[0]
    return {
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "vision_width": int(config.hidden_size),
        "vision_layers": int(config.num_hidden_layers),
        "vision_heads": int(config.num_attention_heads),
    }


def validate_vit_l_14(vision_model: Any) -> dict[str, int]:
    """Require the OpenAI-style ViT-L/14 tensor architecture at 224 or 336 px."""
    actual = _architecture(vision_model)
    expected = {
        "patch_size": 14,
        "vision_width": 1024,
        "vision_layers": 24,
        "vision_heads": 16,
    }
    failures = [
        f"{key}={actual[key]} expected={value}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if actual["image_size"] not in SUPPORTED_IMAGE_SIZES:
        failures.append(
            f"image_size={actual['image_size']} expected one of {SUPPORTED_IMAGE_SIZES}"
        )
    if failures:
        raise ValueError("RN token requires ViT-L/14: " + "; ".join(failures))
    return actual


def _resolve_rn_file(path_or_repo_id: str | Path, revision: str | None = None) -> Path:
    path = Path(path_or_repo_id).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        candidate = path / RN_FILENAME
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate.resolve()
    return Path(
        hf_hub_download(
            repo_id=str(path_or_repo_id),
            filename=RN_FILENAME,
            revision=revision,
        )
    )


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _metadata_int(metadata: dict[str, str], key: str, fallback: int) -> int:
    value = metadata.get(key)
    return int(value) if value is not None else int(fallback)


def apply_read_null_token(
    model: Any,
    token_path_or_repo_id: str | Path,
    *,
    revision: str | None = None,
    debug: bool = False,
):
    """Append the learned VPT/RN token immediately before its checkpoint-defined ViT block."""
    vision = _vision_model(model)
    actual = validate_vit_l_14(vision)

    if getattr(vision, "_rn_adapter_handle", None) is not None:
        raise RuntimeError("RN/VPT token is already active")

    token_path = _resolve_rn_file(token_path_or_repo_id, revision)
    tensors = load_file(str(token_path), device="cpu")
    if "read_null_token" not in tensors:
        raise KeyError(f"{token_path} has no 'read_null_token' tensor")

    token = tensors["read_null_token"].reshape(-1)
    if token.numel() != actual["vision_width"]:
        raise ValueError(
            f"RN tensor shape {tuple(tensors['read_null_token'].shape)} does not match "
            f"vision width {actual['vision_width']}"
        )

    metadata = _metadata(token_path)
    insert_block = _metadata_int(metadata, "read_null_insert_block", 13)
    metadata_width = _metadata_int(metadata, "vision_width", actual["vision_width"])
    metadata_size = _metadata_int(metadata, "image_size", actual["image_size"])

    if metadata_width != actual["vision_width"]:
        raise ValueError(
            f"RN metadata vision_width={metadata_width} conflicts with model "
            f"vision_width={actual['vision_width']}"
        )
    if metadata_size not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(
            f"RN metadata image_size={metadata_size} is unsupported; expected one of "
            f"{SUPPORTED_IMAGE_SIZES}"
        )
    if not 0 <= insert_block < actual["vision_layers"]:
        raise ValueError(f"RN insertion block {insert_block} is outside the vision stack")

    device = vision.embeddings.class_embedding.device
    dtype = vision.embeddings.class_embedding.dtype
    vision.register_parameter(
        "read_null_token",
        torch.nn.Parameter(token.to(device=device, dtype=dtype)),
    )
    vision.read_null_insert_block = insert_block

    def append_rn(_module, args):
        if not args:
            raise RuntimeError("CLIP encoder layer received no hidden states")
        hidden_states = args[0]
        rn = vision.read_null_token.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).view(1, 1, -1)
        rn = rn.expand(hidden_states.shape[0], 1, -1)
        return (torch.cat((hidden_states, rn), dim=1), *args[1:])

    layer = vision.encoder.layers[insert_block]
    vision._rn_adapter_handle = layer.register_forward_pre_hook(append_rn)

    if debug:
        print("[VPT] model architecture:", json.dumps(actual, sort_keys=True))
        print(f"[VPT] token insertion: before zero-based block {insert_block}")
        print(
            f"[VPT] token: {token_path} | shape={tuple(token.shape)} | "
            f"fp32_l2={token.float().norm().item():.8f}"
        )
    return model


# ==================================================================================================
# Benchmark data
# ==================================================================================================

PROMPT = "a photo of a {word}"
DEFAULT_SEED = 20260829

SUBSET_ORDER = (
    ("SCAM", "NoSCAM"),
    ("SCAM", "SCAM"),
    ("SCAM", "SynthSCAM"),
    ("RTA", "NoRTA"),
    ("RTA", "RTA"),
    ("RTA", "SynthRTA"),
)

ATTACKED_SUBSETS = {"SCAM", "SynthSCAM", "RTA", "SynthRTA"}


@dataclass
class PairSample:
    image: Any
    correct_label: str
    distractor_label: str


@dataclass
class SubsetStats:
    count: int
    visual_accuracy: float
    read_accuracy: float | None


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_scam_samples() -> dict[str, list[PairSample]]:
    print("[benchmark] Loading BLISS-e-V/SCAM ...")
    dataset = load_dataset("BLISS-e-V/SCAM", split="train")
    buckets = {name: [] for name in ("NoSCAM", "SCAM", "SynthSCAM")}
    for entry in dataset:
        sample_id = str(entry["id"])
        variant = next((name for name in buckets if sample_id.startswith(name)), None)
        if variant is None:
            continue
        buckets[variant].append(
            PairSample(
                image=entry["image"],
                correct_label=str(entry["object_label"]),
                distractor_label=str(entry["attack_word"]),
            )
        )
    print(
        "[benchmark] SCAM subsets: "
        + ", ".join(f"{name}={len(samples)}" for name, samples in buckets.items())
    )
    return buckets


def load_rta_samples() -> dict[str, list[PairSample]]:
    print("[benchmark] Loading zer0int/RTA-100-Triplet ...")
    dataset = load_dataset("zer0int/RTA-100-Triplet", split="train")
    buckets = {name: [] for name in ("NoRTA", "RTA", "SynthRTA")}
    for entry in dataset:
        variant = str(entry["type"])
        if variant not in buckets:
            continue
        buckets[variant].append(
            PairSample(
                image=entry["image"],
                correct_label=str(entry["object_label"]),
                distractor_label=str(entry["attack_word"]),
            )
        )
    print(
        "[benchmark] RTA subsets: "
        + ", ".join(f"{name}={len(samples)}" for name, samples in buckets.items())
    )
    return buckets


# ==================================================================================================
# Stock CLIP / VPT evaluation
# ==================================================================================================


def _feature_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    for name in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Cannot locate feature tensor in {type(output)!r}")


@torch.inference_mode()
def _encode_stock_texts(
    model: Any,
    processor: Any,
    labels: list[str],
    device: torch.device,
) -> torch.Tensor:
    prompts = [PROMPT.format(word=label) for label in labels]
    inputs = processor(text=prompts, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    features = _feature_tensor(model.get_text_features(**inputs))
    return torch.nn.functional.normalize(features.float(), dim=-1).cpu()


@torch.inference_mode()
def _encode_stock_images(
    model: Any,
    processor: Any,
    images: list[Any],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        inputs = processor(images=images[start : start + batch_size], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        features = _feature_tensor(model.get_image_features(pixel_values=pixel_values))
        chunks.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
    return torch.cat(chunks) if chunks else torch.empty((0, 0), dtype=torch.float32)


def evaluate_stock_model(
    alias: str,
    model_reference: str,
    datasets: dict[str, dict[str, list[PairSample]]],
    labels: list[str],
    label_index: dict[str, int],
    device: torch.device,
    batch_size: int,
    *,
    attach_vpt: bool = False,
) -> dict[str, SubsetStats]:
    print()
    print("=" * 78)
    print(f"[benchmark] Loading {alias}: {model_reference}")
    model = CLIPModel.from_pretrained(model_reference).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_reference)

    if attach_vpt:
        print(f"[benchmark] {alias}: attaching VPT/RN token from {model_reference}")
        apply_read_null_token(model, model_reference, debug=True)

    print(
        f"[benchmark] {alias}: encoding {len(labels)} labels with prompt {PROMPT!r}"
    )
    text_features = _encode_stock_texts(model, processor, labels, device)

    results: dict[str, SubsetStats] = {}
    for dataset_name, subset_name in SUBSET_ORDER:
        samples = datasets[dataset_name][subset_name]
        print(f"[benchmark] {alias}: {dataset_name}/{subset_name} ({len(samples)} images) ...")
        image_features = _encode_stock_images(
            model,
            processor,
            [sample.image for sample in samples],
            device,
            batch_size,
        )

        visual_correct = 0
        attack_word_selected = 0
        for sample, image_feature in zip(samples, image_features):
            object_score = float(image_feature @ text_features[label_index[sample.correct_label]])
            attack_score = float(image_feature @ text_features[label_index[sample.distractor_label]])
            if object_score >= attack_score:
                visual_correct += 1
            else:
                attack_word_selected += 1

        count = len(samples)
        visual_acc = visual_correct / count if count else 0.0
        # For ordinary CLIP-like models, only an attacked image has a meaningful
        # "read the attack word" interpretation. It is simply the binary error rate.
        read_acc = (
            attack_word_selected / count
            if count and subset_name in ATTACKED_SUBSETS
            else None
        )
        results[subset_name] = SubsetStats(count, visual_acc, read_acc)

    del text_features, processor, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


# ==================================================================================================
# ModeMUX evaluation
# ==================================================================================================


def _looks_like_local_mux_repo(path: Path) -> bool:
    required = (
        "config.json",
        "model.safetensors",
        "configuration_xattn_clip.py",
        "modeling_xattn_clip.py",
    )
    return path.is_dir() and all((path / name).is_file() for name in required)


def resolve_mux_reference(explicit: str | None) -> str:
    if explicit:
        return explicit
    here = Path(__file__).resolve().parent
    if _looks_like_local_mux_repo(here):
        print(f"[benchmark] Found local ModeMUX clone beside benchmark: {here}")
        return str(here)
    return MUX_MODEL


def _display_model_reference(model_reference: str) -> str:
    raw = str(model_reference)
    path = Path(raw).expanduser()
    windows_path = "\\" in raw
    if path.is_dir() or raw in {".", ".."}:
        resolved = path.resolve()
        return f"{resolved.name} ({raw})"
    if windows_path:
        return PureWindowsPath(raw.rstrip("\\/")).name
    return raw.rstrip("/")


def _encode_mux_prompts(processor: Any, labels: list[str], device: torch.device) -> torch.Tensor:
    prompts = [PROMPT.format(word=label) for label in labels]
    encoded = processor(
        text=prompts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    if "input_ids" not in encoded:
        raise RuntimeError("ModeMUX processor did not return input_ids")
    return encoded["input_ids"].to(device)


def _validate_mux_output(
    output: Any,
    *,
    mode: str,
    batch_size: int,
    candidate_count: int,
) -> tuple[torch.Tensor, int | None]:
    logits = getattr(output, "logits_per_image", None)
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise RuntimeError(f"ModeMUX mode={mode!r} returned invalid logits_per_image")

    expected_columns = candidate_count + (1 if mode == "read" else 0)
    if tuple(logits.shape) != (batch_size, expected_columns):
        raise RuntimeError(
            f"ModeMUX mode={mode!r} returned logits shape {tuple(logits.shape)}, "
            f"expected {(batch_size, expected_columns)}"
        )

    null_index = getattr(output, "null_candidate_index", None)
    if mode == "read":
        if null_index is None or int(null_index) != candidate_count:
            raise RuntimeError(
                f"ModeMUX mode='read' must append NULL at index {candidate_count}; "
                f"got {null_index}"
            )
        return logits.detach().float().cpu(), int(null_index)

    if null_index is not None:
        raise RuntimeError(
            f"ModeMUX mode={mode!r} unexpectedly exposed null_candidate_index={null_index}"
        )
    return logits.detach().float().cpu(), None


@torch.inference_mode()
def evaluate_mux_model(
    model_reference: str,
    datasets: dict[str, dict[str, list[PairSample]]],
    labels: list[str],
    label_index: dict[str, int],
    device: torch.device,
    batch_size: int,
    revision: str | None,
) -> dict[str, SubsetStats]:
    print()
    print("=" * 78)
    print(f"[benchmark] Loading MUX: {_display_model_reference(model_reference)}")

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
        print(f"[benchmark] MUX revision: {revision}")

    model = AutoModel.from_pretrained(model_reference, **kwargs).float().eval().to(device)
    processor = AutoProcessor.from_pretrained(model_reference, **kwargs)

    model_type = str(getattr(model.config, "model_type", ""))
    if model_type != "xattn_clip":
        raise RuntimeError(
            f"Expected ModeMUX model_type='xattn_clip', got {model_type!r}"
        )

    input_ids = _encode_mux_prompts(processor, labels, device)
    candidate_count = len(labels)
    results: dict[str, SubsetStats] = {}

    for dataset_name, subset_name in SUBSET_ORDER:
        samples = datasets[dataset_name][subset_name]
        print(f"[benchmark] MUX: {dataset_name}/{subset_name} ({len(samples)} images) ...")
        visual_correct = 0
        read_correct = 0

        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            encoded = processor(
                images=[sample.image for sample in batch_samples],
                return_tensors="pt",
            )
            pixel_values = encoded["pixel_values"].to(device)
            batch_n = len(batch_samples)

            any_output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                mode="any",
                correction=True,
                return_details=False,
                pieces_fp32=True,
            )
            any_logits, _ = _validate_mux_output(
                any_output,
                mode="any",
                batch_size=batch_n,
                candidate_count=candidate_count,
            )

            read_output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                mode="read",
                correction=True,
                return_details=False,
                pieces_fp32=True,
            )
            read_logits, null_index = _validate_mux_output(
                read_output,
                mode="read",
                batch_size=batch_n,
                candidate_count=candidate_count,
            )
            assert null_index == candidate_count

            for row, sample in enumerate(batch_samples):
                object_index = label_index[sample.correct_label]
                attack_index = label_index[sample.distractor_label]

                # Normal/default ModeMUX evaluation: visual-semantic binary ZS.
                if float(any_logits[row, object_index]) >= float(any_logits[row, attack_index]):
                    visual_correct += 1

                # READ evaluation is deliberately controlled by mode='read'.
                # On attacked images, the attack word is correct; on clean controls,
                # correct behavior is abstention via the internally appended NULL.
                triplet = torch.stack(
                    (
                        read_logits[row, object_index],
                        read_logits[row, attack_index],
                        read_logits[row, null_index],
                    )
                )
                prediction = int(triplet.argmax().item())  # 0=object, 1=attack word, 2=NULL
                expected = 1 if subset_name in ATTACKED_SUBSETS else 2
                if prediction == expected:
                    read_correct += 1

        count = len(samples)
        results[subset_name] = SubsetStats(
            count=count,
            visual_accuracy=visual_correct / count if count else 0.0,
            read_accuracy=read_correct / count if count else 0.0,
        )

    del input_ids, processor, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


# ==================================================================================================
# Final ASCII report
# ==================================================================================================


def _ascii_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
) -> str:
    matrix = [["Subset", *headers], *rows]
    widths = [max(len(row[col]) for row in matrix) for col in range(len(matrix[0]))]

    def border() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def line(row: list[str], numeric: bool = False) -> str:
        cells = []
        for i, value in enumerate(row):
            if i == 0:
                cells.append(" " + value.ljust(widths[i]) + " ")
            else:
                cells.append(" " + value.rjust(widths[i]) + " ")
        return "|" + "|".join(cells) + "|"

    out = [title, border(), line(matrix[0]), border()]
    out.extend(line(row, numeric=True) for row in matrix[1:])
    out.append(border())
    return "\n".join(out)


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}"


def build_visual_table(all_results: dict[str, dict[str, SubsetStats]]) -> str:
    headers = ["OAI", "GmP", "VPT", "MUX"]
    rows = []
    for _, subset_name in SUBSET_ORDER:
        rows.append(
            [
                subset_name,
                *[_pct(all_results[alias][subset_name].visual_accuracy) for alias in headers],
            ]
        )
    return _ascii_table(
        "Visual-semantic zero-shot accuracy (%) -- normal/default mode (MUX = 'any')",
        headers,
        rows,
    )


def build_read_table(all_results: dict[str, dict[str, SubsetStats]]) -> str:
    headers = ["OAI**", "GmP**", "VPT**", "MUX"]
    source_alias = {"OAI**": "OAI", "GmP**": "GmP", "VPT**": "VPT", "MUX": "MUX"}
    rows = []
    for _, subset_name in SUBSET_ORDER:
        rows.append(
            [
                subset_name,
                *[
                    _pct(all_results[source_alias[header]][subset_name].read_accuracy)
                    for header in headers
                ],
            ]
        )
    return _ascii_table(
        "Reading accuracy (%) -- MUX = deliberate 'read'; ** = unintended attack-word selection",
        headers,
        rows,
    )


def ascii_box(title: str, paragraphs: list[str], width: int = 104) -> str:
    inner = width - 4
    top = "+" + "-" * (width - 2) + "+"
    rows = [top, f"| {title.center(inner)} |", top]
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=inner) if paragraph else [""]
        rows.extend(f"| {line.ljust(inner)} |" for line in wrapped)
    rows.append(top)
    return "\n".join(rows)


def print_final_report(all_results: dict[str, dict[str, SubsetStats]]) -> None:
    print()
    print("[benchmark] Evaluation complete.")
    print()
    print(build_visual_table(all_results))
    print()
    print(build_read_table(all_results))
    print()
    print(
        "** Typographic attack. This is NOT deliberate 'reading', but an unpredictable outcome - "
        "which nevertheless represents success in 'reading' the word, albeit unintended and "
        "unsteerable."
    )
    print(
        "   For NoSCAM/NoRTA there is no attack word to read, so OAI**/GmP**/VPT** are shown as --; "
        "MUX reports correct NULL abstention there."
    )
    print()
    print(
        ascii_box(
            "ModeMUX user-controlled Read / NoRead",
            [
                "For the MUX model, Read vs. NoRead is deliberate and controlled by the user:",
                "\"any\": Automatically controls (mainly suppresses) the influence of readable text and is the default mode for typographic-robustness. ZS for the visual-semantic object, IGNORING TEXT evidence.",
                "\"read\": Selects \"OCR-like\" mode for typographic-reading (or abstention, if no text detected). ZS for the word(s) in the image, IGNORING OBJECT evidence.",
            ],
        )
    )


# ==================================================================================================
# Main
# ==================================================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick HF SCAM/RTA zero-shot benchmark for OAI, GmP, VPT, and ModeMUX CLIP"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--mux-model",
        default=None,
        help=(
            "ModeMUX repo ID or local clone. By default, use the local directory when this script "
            "is inside a complete ModeMUX clone; otherwise use the public HF repo."
        ),
    )
    parser.add_argument(
        "--mux-revision",
        default=None,
        help="Optional pinned ModeMUX HF revision/tag/SHA.",
    )
    args = parser.parse_args()

    print("[benchmark] Starting quick binary zero-shot typographic-attack benchmark.")
    configure_reproducibility(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[benchmark] Using device: {device}")

    mux_reference = resolve_mux_reference(args.mux_model)
    print("[benchmark] Model aliases:")
    print(f"  OAI -> {OAI_MODEL}")
    print(f"  GmP -> {GMP_MODEL}")
    print(f"  VPT -> {VPT_MODEL}")
    print(f"  MUX -> {_display_model_reference(mux_reference)}")

    datasets = {
        "SCAM": load_scam_samples(),
        "RTA": load_rta_samples(),
    }
    all_samples = [
        sample
        for dataset_buckets in datasets.values()
        for samples in dataset_buckets.values()
        for sample in samples
    ]
    labels = sorted(
        {
            label
            for sample in all_samples
            for label in (sample.correct_label, sample.distractor_label)
        },
        key=str.casefold,
    )
    label_index = {label: index for index, label in enumerate(labels)}

    all_results: dict[str, dict[str, SubsetStats]] = {}
    all_results["OAI"] = evaluate_stock_model(
        "OAI", OAI_MODEL, datasets, labels, label_index, device, args.batch_size
    )
    all_results["GmP"] = evaluate_stock_model(
        "GmP", GMP_MODEL, datasets, labels, label_index, device, args.batch_size
    )
    all_results["VPT"] = evaluate_stock_model(
        "VPT",
        VPT_MODEL,
        datasets,
        labels,
        label_index,
        device,
        args.batch_size,
        attach_vpt=True,
    )
    all_results["MUX"] = evaluate_mux_model(
        mux_reference,
        datasets,
        labels,
        label_index,
        device,
        args.batch_size,
        args.mux_revision,
    )

    print_final_report(all_results)


if __name__ == "__main__":
    main()
