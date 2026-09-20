"""
CLIP Stroop Test — vanilla OpenAI CLIP vs ModeMUX CLIP
=======================================================

Evaluates the classic Stroop conflict: identify the ink color versus read the
written color word.

The baseline is official OpenAI CLIP ViT-L/14. ModeMUX is loaded through its
released Hugging Face remote-code API and evaluated through the public modes:

    notext   semantic/content lane with candidate reading disabled
    any      automatic robust semantic mode
    text     forced literal reading without abstention
    read     literal reading with the model's internal NULL candidate

Outcome composition is rendered as pie charts (ink color / written word /
NULL / other). Each model/mode also gets a prompt-ranking pie atlas plus an
automatic per-prompt analysis in the CLI and PROMPT_ANALYSIS.md. CLI rankings
now also print the literal expanded prompt for a fixed example color (red). Prompt
sensitivity across orientation and source diagnostics remain heatmaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

# Baseline stays genuine official OpenAI CLIP.
import clip as vanilla_clip


# Direct-script bootstrap after moving this specialized benchmark under benchmarks/extra/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))



def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(0)


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_MODEMUX_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_OUT_DIR = "out_bench_results/extra/stroop"
DEFAULT_VANILLA_MODEL = "ViT-L/14"

DEFAULT_STROOP_DIRS: Dict[str, Path] = {
    "upright": Path("image_sets/stroop/stroop"),
    "inverted": Path("image_sets/stroop/stroop_inv"),
    "mirrored": Path("image_sets/stroop/stroop_mir"),
}

# Preserve the user's full prompt zoo while fixing the duplicated key that
# previously overwrote the non-PNG trippy prompt.
PROMPT_TEMPLATES: Dict[str, str] = {
    "bare": "{color}",
    "article": "a {color}",
    "word_context": "an image of a {color} word",
    "color_context": "an image of a {color} color",
    "color_repeat": "hue is {color} in rgb {color}, saturation {color}, {color} dye",
    "color_art": "a {color} paint pigment",
    "color_percept": "a {color} hue",
    "color_tech": "rgb {color} palette",
    "color_spam": "{color} {color} {color} {color} {color} {color} {color} {color} {color} {color} {color}",
    "color_hug": "{color}word{color}",
    "color_oooo": "ooooooooocolor {color} mesmerizing enchanted",
    "color_4chan": "hahahaha {color} lmfao",
    "color_bad": "blurry {color} blob, {color} microwave, hallucin{color} radar, mathemat{color}, mandelmicroscope",
    "color_trippy": "hallucintrippy {color}",
    "color_trippy_png": "hallucintrippy {color} png",
    "emo_red": "🔴 {color}",
    "emo_orange": "🟠 {color}",
    "emo_yellow": "🟡 {color}",
    "emo_green": "🟢 {color}",
    "emo_blue": "🔵 {color}",
    "emo_purple": "🟣 {color}",
    "emo_brown": "🟤 {color}",
    "emo_black": "⚫ {color}",
    "emo_white": "⚪ {color}",
    "emo_palette": "🎨 {color}",
    "emo_brush": "🖌️ {color}",
    "emo_image": "🖼 {color}",
    "emo_lol": "🤣🤣🤣 {color}",
}

MODEMUX_MODE_ORDER = ["notext", "any", "text", "read"]
MODEL_MODE_ORDER = [
    ("pretrained", "vanilla"),
    ("modemux", "notext"),
    ("modemux", "any"),
    ("modemux", "text"),
    ("modemux", "read"),
]
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class StroopItem:
    path: Path
    condition: str
    ink_color: str
    word: str


@dataclass
class StroopResult:
    model: str
    mode: str
    condition: str
    prompt_key: str
    prompt_template: str
    filename: str
    ink_color: str
    word: str
    predicted: str
    outcome: str
    logit_color: float
    logit_word: float
    logit_null: float
    logit_margin: float
    top1_logit: float
    prob_color: float
    prob_word: float
    prob_null: float
    source_present_prob: float = float("nan")
    source_readable_prob: float = float("nan")
    source_gate: float = float("nan")
    glyph_mean: float = float("nan")
    glyph_max: float = float("nan")
    route_gate_color: float = float("nan")
    route_gate_word: float = float("nan")
    auto_contribution_color: float = float("nan")
    auto_contribution_word: float = float("nan")
    raw_read_color: float = float("nan")
    raw_read_word: float = float("nan")
    calibrated_read_color: float = float("nan")
    calibrated_read_word: float = float("nan")
    relative_read_color: float = float("nan")
    relative_read_word: float = float("nan")


# =============================================================================
# Utilities
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate official OpenAI CLIP and HF ModeMUX CLIP on Stroop images."
    )
    parser.add_argument(
        "--modemux_model", "--anytext_model", dest="modemux_model",
        default=DEFAULT_MODEMUX_MODEL,
        help="HF repo id or local ModeMUX model directory. --anytext_model is a legacy alias.",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Optional HF revision/tag/commit for the ModeMUX model.",
    )
    parser.add_argument("--vanilla_model", default=DEFAULT_VANILLA_MODEL)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--upright_dir", default=str(DEFAULT_STROOP_DIRS["upright"]))
    parser.add_argument("--inverted_dir", default=str(DEFAULT_STROOP_DIRS["inverted"]))
    parser.add_argument("--mirrored_dir", default=str(DEFAULT_STROOP_DIRS["mirrored"]))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--autocast_dtype",
        choices=["float16", "bfloat16", "none"],
        default="float16" if torch.cuda.is_available() else "none",
    )
    parser.add_argument(
        "--prompt_keys",
        default="",
        help="Optional comma-separated subset of prompt keys. Empty evaluates all prompts.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Optional global item cap for smoke tests. 0 means all images.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def finite_std(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(arr.std()) if arr.size else float("nan")


def finite_median(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def autocast_context(device: str, dtype_name: str):
    device_type = torch.device(device).type
    if device_type != "cuda" or dtype_name == "none":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def parse_stroop_files(dirs: Mapping[str, Path], max_items: int = 0) -> List[StroopItem]:
    items: List[StroopItem] = []
    for condition, directory in dirs.items():
        if not directory.exists():
            print(f"[WARN] Directory not found: {directory} — skipping {condition}")
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            parts = path.stem.lower().split("_")
            if len(parts) < 2:
                print(f"[WARN] Cannot parse filename: {path.name} — skipping")
                continue
            items.append(
                StroopItem(
                    path=path,
                    condition=condition,
                    ink_color=parts[0],
                    word=parts[1],
                )
            )
    if max_items > 0:
        items = items[:max_items]
    return items


def collect_color_vocab(items: Sequence[StroopItem]) -> List[str]:
    return sorted({x for item in items for x in (item.ink_color, item.word)})


def select_prompt_templates(prompt_keys: str) -> Dict[str, str]:
    if not prompt_keys.strip():
        return dict(PROMPT_TEMPLATES)
    requested = [x.strip() for x in prompt_keys.split(",") if x.strip()]
    missing = [x for x in requested if x not in PROMPT_TEMPLATES]
    if missing:
        raise KeyError(f"Unknown prompt keys: {missing}")
    return {key: PROMPT_TEMPLATES[key] for key in requested}


def preprocess_all_images(
    items: Sequence[StroopItem],
    preprocess: Any,
) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for index, item in enumerate(items, start=1):
        with Image.open(item.path) as image:
            tensors.append(preprocess(image.convert("RGB")))
        if index % 100 == 0:
            print(f"  preprocessed {index}/{len(items)} images")
    return torch.stack(tensors, dim=0)


def preprocess_all_images_hf(
    items: Sequence[StroopItem],
    processor: Any,
) -> torch.Tensor:
    """Preprocess once with the released HF processor, then reuse across prompt sweeps."""
    tensors: List[torch.Tensor] = []
    for index, item in enumerate(items, start=1):
        with Image.open(item.path) as image:
            encoded = processor(images=[image.convert("RGB")], return_tensors="pt")
        pixel_values = encoded.get("pixel_values")
        if pixel_values is None or pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
            raise RuntimeError(
                f"HF processor returned unexpected pixel_values for {item.path}: "
                f"{None if pixel_values is None else tuple(pixel_values.shape)}"
            )
        tensors.append(pixel_values[0])
        if index % 100 == 0:
            print(f"  preprocessed {index}/{len(items)} ModeMUX images")
    return torch.stack(tensors, dim=0)


def iter_batches(tensor: torch.Tensor, batch_size: int):
    for start in range(0, tensor.shape[0], batch_size):
        end = min(start + batch_size, tensor.shape[0])
        yield start, end, tensor[start:end]


def outcome_from_prediction(predicted: str, item: StroopItem) -> str:
    if predicted == item.ink_color:
        return "color"
    if predicted == item.word:
        return "word"
    if predicted == "<null>":
        return "null"
    return "mis"


def _as_block_list(value: Any) -> List[int]:
    if value is None:
        return []
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    return [int(x) for x in value]


def _tap_weight_map(blocks: Sequence[int], logits: Optional[torch.Tensor]) -> Dict[str, float]:
    if logits is None:
        return {}
    weights = logits.detach().float().cpu().softmax(dim=0).reshape(-1).tolist()
    return {f"B{int(block)}": float(weight) for block, weight in zip(blocks, weights)}


def learned_component_summary(model: torch.nn.Module) -> Dict[str, Any]:
    implant = getattr(model, "read_implant", None)
    if implant is None:
        return {}

    summary: Dict[str, Any] = {
        "model_type": str(getattr(model.config, "model_type", "")),
        "read_attention_architecture": str(
            getattr(getattr(implant, "read_bridge", None), "architecture", "")
        ),
    }
    try:
        late = _as_block_list(getattr(implant, "tap_blocks", None))
        read_states = _as_block_list(getattr(implant, "read_state_blocks", None)) or late
        ortho = _as_block_list(getattr(implant, "ortho_tap_blocks", None))
        source = _as_block_list(getattr(implant, "source_tap_blocks", None))
        summary["topology"] = {
            "ortho": ortho,
            "source": source,
            "content": late,
            "read_states": read_states,
        }
        summary["tap_weights"] = {
            "ortho": _tap_weight_map(ortho, getattr(implant, "ortho_tap_logits", None)),
            "source": _tap_weight_map(source, getattr(implant, "source_tap_logits", None)),
            "read": _tap_weight_map(late, getattr(implant, "read_tap_logits", None)),
            "content": _tap_weight_map(late, getattr(implant, "content_tap_logits", None)),
        }
    except Exception as exc:
        summary["tap_summary_error"] = repr(exc)

    for name in (
        "auto_read_scale",
        "null_abstain_weight",
        "glyph_bias_beta",
        "read_calibration_scale",
    ):
        value = getattr(implant, name, None)
        if torch.is_tensor(value):
            summary[name] = float(value.detach().float().cpu())

    register_gate = getattr(getattr(implant, "read_bridge", None), "register_gate", None)
    if torch.is_tensor(register_gate):
        summary["register_gate"] = float(register_gate.detach().float().cpu())
    return summary


# =============================================================================
# Model loading
# =============================================================================


def load_vanilla_model(model_name: str, device: str):
    print(f"\n[vanilla] loading official OpenAI CLIP: {model_name}")
    model, preprocess = vanilla_clip.load(model_name, device=device, jit=False)
    model.eval()
    return model, preprocess


def load_modemux_model(model_reference: str, device: str, revision: str = ""):
    print(f"\n[ModeMUX] loading HF model: {model_reference}")
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if revision.strip():
        kwargs["revision"] = revision.strip()
        print(f"[ModeMUX] revision={revision.strip()}")

    model = AutoModel.from_pretrained(model_reference, **kwargs)
    processor = AutoProcessor.from_pretrained(model_reference, **kwargs)

    # Reference benchmark path: checkpoint is released in FP32; autocast below may
    # accelerate the ordinary backbone while pieces_fp32=True protects ModeMUX extras.
    model = model.float().eval().to(device)

    model_type = str(getattr(model.config, "model_type", ""))
    if model_type != "xattn_clip":
        raise RuntimeError(
            f"Expected ModeMUX model_type='xattn_clip', got {model_type!r}."
        )
    implant = getattr(model, "read_implant", None)
    if implant is None:
        raise RuntimeError("Loaded HF checkpoint has no read_implant; not a full ModeMUX model.")
    architecture = str(getattr(getattr(implant, "read_bridge", None), "architecture", ""))
    if architecture and architecture != "sigmoid_all":
        raise RuntimeError(
            f"Expected final ModeMUX READ architecture 'sigmoid_all', got {architecture!r}."
        )

    print(f"[ModeMUX] model_type={model_type} READ={architecture or 'unknown'}")
    print("[ModeMUX] public modes: notext / any / text / read(NULL)")
    return model, processor


# =============================================================================
# Evaluation
# =============================================================================


@torch.inference_mode()
def evaluate_vanilla(
    model: torch.nn.Module,
    images_cpu: torch.Tensor,
    items: Sequence[StroopItem],
    color_vocab: Sequence[str],
    prompt_templates: Mapping[str, str],
    device: str,
    batch_size: int,
    autocast_dtype: str,
) -> List[StroopResult]:
    results: List[StroopResult] = []
    color_index = {color: i for i, color in enumerate(color_vocab)}
    scale = model.logit_scale.exp().detach()

    for prompt_index, (prompt_key, template) in enumerate(prompt_templates.items(), start=1):
        print(f"[vanilla] prompt {prompt_index}/{len(prompt_templates)}: {prompt_key}")
        prompts = [template.format(color=color) for color in color_vocab]
        tokens = vanilla_clip.tokenize(prompts, truncate=True).to(device)
        with autocast_context(device, autocast_dtype):
            text_features = F.normalize(model.encode_text(tokens), dim=-1)

        for start, end, batch_cpu in iter_batches(images_cpu, batch_size):
            batch = batch_cpu.to(device, non_blocking=True)
            with autocast_context(device, autocast_dtype):
                image_features = F.normalize(model.encode_image(batch), dim=-1)
                logits = (scale * image_features @ text_features.t()).float()
                probabilities = logits.softmax(dim=-1)

            logits_cpu = logits.cpu()
            probs_cpu = probabilities.cpu()
            for local_index, item_index in enumerate(range(start, end)):
                item = items[item_index]
                row = logits_cpu[local_index]
                prob_row = probs_cpu[local_index]
                predicted_index = int(row.argmax().item())
                predicted = color_vocab[predicted_index]
                ink_index = color_index[item.ink_color]
                word_index = color_index[item.word]
                logit_color = float(row[ink_index])
                logit_word = float(row[word_index])
                results.append(
                    StroopResult(
                        model="pretrained",
                        mode="vanilla",
                        condition=item.condition,
                        prompt_key=prompt_key,
                        prompt_template=template,
                        filename=item.path.name,
                        ink_color=item.ink_color,
                        word=item.word,
                        predicted=predicted,
                        outcome=outcome_from_prediction(predicted, item),
                        logit_color=logit_color,
                        logit_word=logit_word,
                        logit_null=float("nan"),
                        logit_margin=logit_color - logit_word,
                        top1_logit=float(row.max()),
                        prob_color=float(prob_row[ink_index]),
                        prob_word=float(prob_row[word_index]),
                        prob_null=float("nan"),
                    )
                )
    return results


def encode_modemux_prompts(
    processor: Any,
    prompts: Sequence[str],
    device: str,
) -> torch.Tensor:
    encoded = processor(
        text=list(prompts),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    if "input_ids" not in encoded:
        raise RuntimeError("HF processor did not return input_ids.")
    return encoded["input_ids"].to(device)


def _details_dict(output: Any) -> Mapping[str, Any]:
    details = getattr(output, "details", None)
    return details if isinstance(details, Mapping) else {}


def _details_tensor(details: Mapping[str, Any], key: str) -> Optional[torch.Tensor]:
    value = details.get(key)
    if not torch.is_tensor(value):
        return None
    return value.detach().float().cpu()


def _candidate_scalar(details: Mapping[str, Any], key: str, row: int, candidate: int) -> float:
    tensor = _details_tensor(details, key)
    if tensor is None:
        return float("nan")
    try:
        if tensor.ndim == 0:
            return float(tensor)
        if tensor.ndim == 1:
            value = tensor[candidate] if tensor.numel() > candidate else tensor[row]
        else:
            value = tensor[row, candidate]
        if value.numel() != 1:
            value = value.reshape(-1)[0]
        return float(value)
    except (IndexError, RuntimeError):
        return float("nan")


def _validate_modemux_output(
    output: Any,
    *,
    mode: str,
    batch_size: int,
    candidate_count: int,
) -> Tuple[torch.Tensor, Optional[int]]:
    logits = getattr(output, "logits_per_image", None)
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise RuntimeError(f"ModeMUX mode={mode!r} returned invalid logits_per_image.")
    expected_columns = candidate_count + (1 if mode == "read" else 0)
    if tuple(logits.shape) != (batch_size, expected_columns):
        raise RuntimeError(
            f"ModeMUX mode={mode!r} returned logits shape {tuple(logits.shape)}, "
            f"expected {(batch_size, expected_columns)}."
        )

    null_index = getattr(output, "null_candidate_index", None)
    if mode == "read":
        if null_index is None or int(null_index) != candidate_count:
            raise RuntimeError(
                f"mode='read' must append NULL at index {candidate_count}; got {null_index}."
            )
        return logits.detach().float().cpu(), int(null_index)
    if null_index is not None:
        raise RuntimeError(
            f"ModeMUX mode={mode!r} unexpectedly exposed null_candidate_index={null_index}."
        )
    return logits.detach().float().cpu(), None


@torch.inference_mode()
def evaluate_modemux(
    model: torch.nn.Module,
    processor: Any,
    images_cpu: torch.Tensor,
    items: Sequence[StroopItem],
    color_vocab: Sequence[str],
    prompt_templates: Mapping[str, str],
    device: str,
    batch_size: int,
    autocast_dtype: str,
) -> List[StroopResult]:
    """Evaluate the released HF API directly; no oaiclip compatibility layer."""
    results: List[StroopResult] = []
    color_index = {color: i for i, color in enumerate(color_vocab)}
    candidate_count = len(color_vocab)

    for prompt_index, (prompt_key, template) in enumerate(prompt_templates.items(), start=1):
        print(f"[ModeMUX] prompt {prompt_index}/{len(prompt_templates)}: {prompt_key}")
        prompts = [template.format(color=color) for color in color_vocab]
        input_ids = encode_modemux_prompts(processor, prompts, device)

        for start, end, batch_cpu in iter_batches(images_cpu, batch_size):
            batch = batch_cpu.to(device, non_blocking=True)
            batch_n = end - start
            mode_logits: Dict[str, torch.Tensor] = {}
            mode_details: Dict[str, Mapping[str, Any]] = {}
            read_null_index: Optional[int] = None

            for mode in MODEMUX_MODE_ORDER:
                with autocast_context(device, autocast_dtype):
                    output = model(
                        input_ids=input_ids,
                        pixel_values=batch,
                        mode=mode,
                        correction=True,
                        return_details=True,
                        pieces_fp32=True,
                    )
                logits_cpu, null_index = _validate_modemux_output(
                    output,
                    mode=mode,
                    batch_size=batch_n,
                    candidate_count=candidate_count,
                )
                mode_logits[mode] = logits_cpu
                mode_details[mode] = _details_dict(output)
                if mode == "read":
                    read_null_index = null_index

            if read_null_index != candidate_count:
                raise RuntimeError("ModeMUX READ NULL layout validation failed.")

            # Image-level source/glyph heads are mode-independent; take them from ANY.
            any_details = mode_details["any"]
            source_logits = _details_tensor(any_details, "source_logits")
            source_probs = source_logits.sigmoid() if source_logits is not None else None
            source_gate = _details_tensor(any_details, "source_gate")
            glyph_probs = _details_tensor(any_details, "glyph_probs")

            for local_index, item_index in enumerate(range(start, end)):
                item = items[item_index]
                ink_index = color_index[item.ink_color]
                word_index = color_index[item.word]

                if source_probs is not None and source_probs.ndim >= 2 and source_probs.shape[1] >= 2:
                    source_present = float(source_probs[local_index, 0])
                    source_readable = float(source_probs[local_index, 1])
                else:
                    source_present = source_readable = float("nan")
                source_gate_value = (
                    float(source_gate[local_index].reshape(-1)[0])
                    if source_gate is not None and source_gate.numel() > local_index
                    else float("nan")
                )
                glyph_mean = (
                    float(glyph_probs[local_index].mean())
                    if glyph_probs is not None and glyph_probs.shape[0] > local_index
                    else float("nan")
                )
                glyph_max = (
                    float(glyph_probs[local_index].max())
                    if glyph_probs is not None and glyph_probs.shape[0] > local_index
                    else float("nan")
                )

                for mode in MODEMUX_MODE_ORDER:
                    row_logits = mode_logits[mode][local_index]
                    candidate_logits = row_logits[:candidate_count]
                    details = mode_details[mode]

                    if mode == "read":
                        probs = row_logits.softmax(dim=-1)
                        winner = int(row_logits.argmax().item())
                        predicted = "<null>" if winner == read_null_index else color_vocab[winner]
                        logit_null = float(row_logits[read_null_index])
                        prob_null = float(probs[read_null_index])
                    else:
                        probs = candidate_logits.softmax(dim=-1)
                        winner = int(candidate_logits.argmax().item())
                        predicted = color_vocab[winner]
                        logit_null = float("nan")
                        prob_null = float("nan")

                    logit_color = float(candidate_logits[ink_index])
                    logit_word = float(candidate_logits[word_index])
                    results.append(
                        StroopResult(
                            model="modemux",
                            mode=mode,
                            condition=item.condition,
                            prompt_key=prompt_key,
                            prompt_template=template,
                            filename=item.path.name,
                            ink_color=item.ink_color,
                            word=item.word,
                            predicted=predicted,
                            outcome=outcome_from_prediction(predicted, item),
                            logit_color=logit_color,
                            logit_word=logit_word,
                            logit_null=logit_null,
                            logit_margin=logit_color - logit_word,
                            top1_logit=float(row_logits.max()),
                            prob_color=float(probs[ink_index]),
                            prob_word=float(probs[word_index]),
                            prob_null=prob_null,
                            source_present_prob=source_present,
                            source_readable_prob=source_readable,
                            source_gate=source_gate_value,
                            glyph_mean=glyph_mean,
                            glyph_max=glyph_max,
                            route_gate_color=_candidate_scalar(details, "route_gate", local_index, ink_index),
                            route_gate_word=_candidate_scalar(details, "route_gate", local_index, word_index),
                            auto_contribution_color=_candidate_scalar(details, "auto_read_contribution", local_index, ink_index),
                            auto_contribution_word=_candidate_scalar(details, "auto_read_contribution", local_index, word_index),
                            raw_read_color=_candidate_scalar(details, "raw_read_logits", local_index, ink_index),
                            raw_read_word=_candidate_scalar(details, "raw_read_logits", local_index, word_index),
                            calibrated_read_color=_candidate_scalar(details, "read_logits", local_index, ink_index),
                            calibrated_read_word=_candidate_scalar(details, "read_logits", local_index, word_index),
                            relative_read_color=_candidate_scalar(details, "relative_read_logits", local_index, ink_index),
                            relative_read_word=_candidate_scalar(details, "relative_read_logits", local_index, word_index),
                        )
                    )

    return results


# =============================================================================
# Aggregation and CSV output
# =============================================================================


def aggregate_results(
    results: Sequence[StroopResult],
    group_fields: Sequence[str],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[StroopResult]] = defaultdict(list)
    for result in results:
        key = tuple(getattr(result, field) for field in group_fields)
        groups[key].append(result)

    rows: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        n = len(group)
        counts = {name: sum(r.outcome == name for r in group) for name in ("color", "word", "null", "mis")}
        row: Dict[str, Any] = dict(zip(group_fields, key))
        row.update(
            {
                "n": n,
                "pct_color": 100.0 * counts["color"] / n,
                "pct_word": 100.0 * counts["word"] / n,
                "pct_null": 100.0 * counts["null"] / n,
                "pct_mis": 100.0 * counts["mis"] / n,
                "mean_margin": finite_mean(r.logit_margin for r in group),
                "median_margin": finite_median(r.logit_margin for r in group),
                "std_margin": finite_std(r.logit_margin for r in group),
                "mean_prob_color": finite_mean(r.prob_color for r in group),
                "mean_prob_word": finite_mean(r.prob_word for r in group),
                "mean_prob_null": finite_mean(r.prob_null for r in group),
                "source_present_mean": finite_mean(r.source_present_prob for r in group),
                "source_readable_mean": finite_mean(r.source_readable_prob for r in group),
                "source_gate_mean": finite_mean(r.source_gate for r in group),
                "glyph_mean": finite_mean(r.glyph_mean for r in group),
                "glyph_max_mean": finite_mean(r.glyph_max for r in group),
                "route_gate_color_mean": finite_mean(r.route_gate_color for r in group),
                "route_gate_word_mean": finite_mean(r.route_gate_word for r in group),
                "auto_contribution_color_mean": finite_mean(r.auto_contribution_color for r in group),
                "auto_contribution_word_mean": finite_mean(r.auto_contribution_word for r in group),
                "raw_read_margin_mean": finite_mean(r.raw_read_color - r.raw_read_word for r in group),
                "calibrated_read_margin_mean": finite_mean(
                    r.calibrated_read_color - r.calibrated_read_word for r in group
                ),
                "relative_read_color_mean": finite_mean(r.relative_read_color for r in group),
                "relative_read_word_mean": finite_mean(r.relative_read_word for r in group),
            }
        )
        rows.append(row)
    return rows


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in row.items()
                }
            )


def write_detailed_csv(path: Path, results: Sequence[StroopResult]) -> None:
    write_csv_rows(path, [asdict(result) for result in results])


def build_compact_rows(results: Sequence[StroopResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(aggregate_results(results, ["model", "mode"]))
    rows.extend(aggregate_results(results, ["model", "mode", "condition"]))
    for row in rows:
        row["scope"] = "condition" if "condition" in row else "overall"
        row.setdefault("condition", "ALL")
    return rows


# =============================================================================
# Plotting
# =============================================================================


PLOT_COLORS = {
    "color": "#4fc3f7",
    "word": "#ef5350",
    "null": "#ffca28",
    "mis": "#78909c",
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.size": 9,
            "axes.facecolor": "#0d0d0d",
            "figure.facecolor": "#0d0d0d",
            "savefig.facecolor": "#0d0d0d",
            "text.color": "#e0e0e0",
            "axes.labelcolor": "#e0e0e0",
            "xtick.color": "#aaaaaa",
            "ytick.color": "#aaaaaa",
            "axes.edgecolor": "#444444",
            "grid.color": "#333333",
        }
    )


def _mode_label(model: str, mode: str) -> str:
    if model == "pretrained":
        return "pretrained / vanilla"
    labels = {
        "notext": "ModeMUX / notext",
        "any": "ModeMUX / any",
        "text": "ModeMUX / text",
        "read": "ModeMUX / read + NULL",
    }
    return labels.get(mode, f"ModeMUX / {mode}")


def _mode_slug(model: str, mode: str) -> str:
    return f"{model}_{mode}".replace("<", "").replace(">", "")


def plot_mode_prompt_heatmaps(
    results: Sequence[StroopResult],
    model: str,
    mode: str,
    conditions: Sequence[str],
    prompt_keys: Sequence[str],
    out_path: Path,
) -> None:
    subset = [r for r in results if r.model == model and r.mode == mode]
    summary = aggregate_results(subset, ["condition", "prompt_key"])
    lookup = {(r["condition"], r["prompt_key"]): r for r in summary}

    metrics = [
        ("pct_color", "% ink color", "Blues", 0.0, 100.0),
        ("pct_word", "% written word", "Reds", 0.0, 100.0),
        ("mean_margin", "mean logit margin: color − word", "RdYlGn", None, None),
    ]
    fig, axes = plt.subplots(
        3, 1,
        figsize=(max(14, 0.65 * len(prompt_keys)), 7.8),
        constrained_layout=True,
    )
    margin_values = [
        safe_float(row.get("mean_margin"))
        for row in summary
        if math.isfinite(safe_float(row.get("mean_margin")))
    ]
    margin_abs = max([abs(x) for x in margin_values], default=1.0)

    for ax, (field, title, cmap, vmin, vmax) in zip(axes, metrics):
        data = np.full((len(conditions), len(prompt_keys)), np.nan, dtype=np.float64)
        for row_index, condition in enumerate(conditions):
            for col_index, prompt_key in enumerate(prompt_keys):
                row = lookup.get((condition, prompt_key))
                if row is not None:
                    data[row_index, col_index] = safe_float(row.get(field))
        if field == "mean_margin":
            vmin, vmax = -margin_abs, margin_abs
        image = ax.imshow(
            data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)
        ax.set_xticks(range(len(prompt_keys)))
        ax.set_xticklabels(prompt_keys, rotation=70, ha="right", fontsize=7)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01)
        for yi in range(data.shape[0]):
            for xi in range(data.shape[1]):
                value = data[yi, xi]
                if math.isfinite(value):
                    label = f"{value:.0f}" if field.startswith("pct_") else f"{value:.2f}"
                    ax.text(xi, yi, label, ha="center", va="center", fontsize=5.5, color="white")

    fig.suptitle(f"Stroop prompt response — {_mode_label(model, mode)}", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _pie_values(row: Mapping[str, Any]) -> List[float]:
    values = [
        max(0.0, safe_float(row.get("pct_color", 0.0))),
        max(0.0, safe_float(row.get("pct_word", 0.0))),
        max(0.0, safe_float(row.get("pct_null", 0.0))),
        max(0.0, safe_float(row.get("pct_mis", 0.0))),
    ]
    return [0.0 if not math.isfinite(v) else v for v in values]


def _autopct(pct: float) -> str:
    return f"{pct:.1f}%" if pct >= 2.0 else ""


def _prompt_target_spec(model: str, mode: str) -> Tuple[str, str]:
    """Return the intended Stroop target for this model/mode.

    Semantic modes are judged by ink color. Literal READ modes are judged by
    the written word. Stroop images always contain a word, so NULL is an
    abstention outcome rather than the target for mode=read.
    """
    if model == "modemux" and mode in {"text", "read"}:
        return "pct_word", "written word"
    return "pct_color", "ink color"


def _prompt_analysis_rows(
    prompt_overall_rows: Sequence[Mapping[str, Any]],
    prompt_condition_rows: Sequence[Mapping[str, Any]],
    model: str,
    mode: str,
) -> List[Dict[str, Any]]:
    target_field, target_label = _prompt_target_spec(model, mode)
    condition_lookup: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in prompt_condition_rows:
        if str(row.get("model")) == model and str(row.get("mode")) == mode:
            condition_lookup[str(row.get("prompt_key"))].append(row)

    enriched: List[Dict[str, Any]] = []
    for source in prompt_overall_rows:
        if str(source.get("model")) != model or str(source.get("mode")) != mode:
            continue
        row = dict(source)
        prompt_key = str(row.get("prompt_key"))
        condition_values = [
            safe_float(item.get(target_field))
            for item in condition_lookup.get(prompt_key, [])
            if math.isfinite(safe_float(item.get(target_field)))
        ]
        target_pct = safe_float(row.get(target_field))
        row["target_field"] = target_field
        row["target_label"] = target_label
        row["target_pct"] = target_pct
        row["target_worst_condition_pct"] = (
            min(condition_values) if condition_values else float("nan")
        )
        row["target_best_condition_pct"] = (
            max(condition_values) if condition_values else float("nan")
        )
        row["target_condition_range_pp"] = (
            max(condition_values) - min(condition_values)
            if len(condition_values) >= 2
            else 0.0 if len(condition_values) == 1
            else float("nan")
        )
        enriched.append(row)

    enriched.sort(
        key=lambda row: (
            -safe_float(row.get("target_pct")),
            -safe_float(row.get("target_worst_condition_pct")),
            str(row.get("prompt_key")),
        )
    )
    for rank, row in enumerate(enriched, start=1):
        row["rank"] = rank
    return enriched


def _expanded_prompt_example(prompt_key: str, color: str = "red") -> str:
    """Return the literal prompt produced by a named template for one example color."""
    template = PROMPT_TEMPLATES.get(str(prompt_key))
    if template is None:
        return ""
    return template.format(color=color)


def _format_prompt_analysis_line(row: Mapping[str, Any]) -> str:
    return (
        f"{int(row['rank']):>2d}  "
        f"{str(row['prompt_key']):<18.18s} "
        f"{safe_float(row['target_pct']):>6.1f}% "
        f"{safe_float(row['pct_color']):>6.1f}% "
        f"{safe_float(row['pct_word']):>6.1f}% "
        f"{safe_float(row['pct_null']):>6.1f}% "
        f"{safe_float(row['pct_mis']):>6.1f}% "
        f"{safe_float(row['mean_margin']):>+8.3f} "
        f"{safe_float(row['target_worst_condition_pct']):>6.1f}% "
        f"{safe_float(row['target_condition_range_pp']):>6.1f}pp"
    )


def print_prompt_analysis(
    prompt_overall_rows: Sequence[Mapping[str, Any]],
    prompt_condition_rows: Sequence[Mapping[str, Any]],
) -> None:
    print("\nPrompt analysis")
    print("===============")
    for model, mode in MODEL_MODE_ORDER:
        rows = _prompt_analysis_rows(
            prompt_overall_rows, prompt_condition_rows, model, mode
        )
        if not rows:
            continue
        target_label = str(rows[0]["target_label"])
        best = rows[0]
        worst = rows[-1]
        stable = min(
            rows,
            key=lambda row: (
                safe_float(row.get("target_condition_range_pp"))
                if math.isfinite(safe_float(row.get("target_condition_range_pp")))
                else float("inf"),
                -safe_float(row.get("target_pct")),
            ),
        )
        swing = max(
            rows,
            key=lambda row: (
                safe_float(row.get("target_condition_range_pp"))
                if math.isfinite(safe_float(row.get("target_condition_range_pp")))
                else -float("inf")
            ),
        )
        spread = safe_float(best["target_pct"]) - safe_float(worst["target_pct"])

        print(f"\n[{_mode_label(model, mode)}] target = {target_label}")
        print(
            f"best={best['prompt_key']} {safe_float(best['target_pct']):.1f}% | "
            f"worst={worst['prompt_key']} {safe_float(worst['target_pct']):.1f}% | "
            f"prompt spread={spread:.1f} pp"
        )
        print(
            f"most orientation-stable={stable['prompt_key']} "
            f"(range {safe_float(stable['target_condition_range_pp']):.1f} pp, "
            f"target {safe_float(stable['target_pct']):.1f}%) | "
            f"largest orientation swing={swing['prompt_key']} "
            f"({safe_float(swing['target_condition_range_pp']):.1f} pp)"
        )
        print(
            "rk  prompt             target   ink    word   NULL  other   margin   worst  range"
        )
        print("-" * 94)
        print('    expanded example uses color="red"')
        for row in rows:
            print(_format_prompt_analysis_line(row))
            example = _expanded_prompt_example(str(row["prompt_key"]), color="red")
            if example:
                print(f'    -> {example}')


def write_prompt_analysis_markdown(
    path: Path,
    prompt_overall_rows: Sequence[Mapping[str, Any]],
    prompt_condition_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Stroop prompt analysis",
        "",
        (
            "Prompts are ranked by the intended target of each model/mode: ink color "
            "for vanilla/notext/any, and written word for text/read."
        ),
    ]
    for model, mode in MODEL_MODE_ORDER:
        rows = _prompt_analysis_rows(
            prompt_overall_rows, prompt_condition_rows, model, mode
        )
        if not rows:
            continue
        target_label = str(rows[0]["target_label"])
        best, worst = rows[0], rows[-1]
        swing = max(
            rows,
            key=lambda row: safe_float(row.get("target_condition_range_pp"))
            if math.isfinite(safe_float(row.get("target_condition_range_pp")))
            else -float("inf"),
        )
        lines.extend(
            [
                "",
                f"## {_mode_label(model, mode)}",
                "",
                f"Target: **{target_label}**.",
                "",
                (
                    f"Best prompt: `{best['prompt_key']}` "
                    f"({safe_float(best['target_pct']):.2f}%). "
                    f"Worst: `{worst['prompt_key']}` "
                    f"({safe_float(worst['target_pct']):.2f}%). "
                    f"Largest orientation swing: `{swing['prompt_key']}` "
                    f"({safe_float(swing['target_condition_range_pp']):.2f} pp)."
                ),
                "",
                "| Rank | Prompt | Expanded example (`red`) | Target | Ink | Word | NULL | Other | Color−word margin | Worst orientation | Orientation range |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            expanded = _expanded_prompt_example(str(row["prompt_key"]), color="red")
            expanded_md = expanded.replace("|", "\\|")
            lines.append(
                f"| {int(row['rank'])} | `{row['prompt_key']}` | `{expanded_md}` | "
                f"{safe_float(row['target_pct']):.2f}% | "
                f"{safe_float(row['pct_color']):.2f}% | "
                f"{safe_float(row['pct_word']):.2f}% | "
                f"{safe_float(row['pct_null']):.2f}% | "
                f"{safe_float(row['pct_mis']):.2f}% | "
                f"{safe_float(row['mean_margin']):+.4f} | "
                f"{safe_float(row['target_worst_condition_pct']):.2f}% | "
                f"{safe_float(row['target_condition_range_pp']):.2f} pp |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_mode_prompt_pies(
    results: Sequence[StroopResult],
    model: str,
    mode: str,
    out_path: Path,
) -> None:
    """Small-multiple pie atlas: one overall choice composition per prompt."""
    subset = [r for r in results if r.model == model and r.mode == mode]
    summary = aggregate_results(subset, ["prompt_key"])
    target_field, target_label = _prompt_target_spec(model, mode)
    summary = sorted(
        summary,
        key=lambda row: (
            -safe_float(row.get(target_field)),
            str(row.get("prompt_key")),
        ),
    )
    if not summary:
        return

    ncols = min(6, len(summary))
    nrows = int(math.ceil(len(summary) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.15 * ncols, 2.75 * nrows),
        squeeze=False,
    )
    colors = [PLOT_COLORS[key] for key in ("color", "word", "null", "mis")]

    for index, ax in enumerate(axes.flat):
        if index >= len(summary):
            ax.axis("off")
            continue
        row = summary[index]
        values = _pie_values(row)
        if sum(values) > 0:
            ax.pie(
                values,
                colors=colors,
                startangle=90,
                counterclock=False,
                autopct=_autopct,
                pctdistance=0.69,
                textprops={"fontsize": 6.5},
                wedgeprops={"linewidth": 0.55, "edgecolor": "#0d0d0d"},
            )
        target = safe_float(row.get(target_field))
        ax.set_title(
            f"#{index + 1:02d} {row['prompt_key']}\n{target_label}: {target:.1f}%",
            fontsize=8,
        )

    handles = [
        Patch(facecolor=PLOT_COLORS[key], label=label)
        for key, label in (
            ("color", "ink color"),
            ("word", "written word"),
            ("null", "NULL"),
            ("mis", "other"),
        )
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, framealpha=0.2)
    fig.suptitle(
        f"Prompt ranking by {target_label} — {_mode_label(model, mode)}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mode_condition_pies(
    results: Sequence[StroopResult],
    model: str,
    mode: str,
    conditions: Sequence[str],
    out_path: Path,
) -> None:
    subset = [r for r in results if r.model == model and r.mode == mode]
    condition_rows = aggregate_results(subset, ["condition"])
    overall_rows = aggregate_results(subset, [])
    lookup = {str(row["condition"]): row for row in condition_rows}
    if overall_rows:
        lookup["ALL"] = overall_rows[0]

    scopes = ["ALL", *conditions]
    fig, axes = plt.subplots(1, len(scopes), figsize=(3.6 * len(scopes), 3.8))
    axes = np.atleast_1d(axes)
    colors = [PLOT_COLORS[k] for k in ("color", "word", "null", "mis")]

    for ax, scope in zip(axes, scopes):
        row = lookup.get(scope)
        values = _pie_values(row or {})
        if sum(values) <= 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.axis("off")
            continue
        ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct=_autopct,
            pctdistance=0.72,
            textprops={"fontsize": 8},
            wedgeprops={"linewidth": 0.8, "edgecolor": "#0d0d0d"},
        )
        ax.set_title(scope)

    handles = [
        Patch(facecolor=PLOT_COLORS[key], label=label)
        for key, label in (
            ("color", "ink color"),
            ("word", "written word"),
            ("null", "NULL"),
            ("mis", "other"),
        )
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, framealpha=0.2)
    fig.suptitle(f"Stroop outcome composition — {_mode_label(model, mode)}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_all_modes_condition_pies(
    results: Sequence[StroopResult],
    conditions: Sequence[str],
    out_path: Path,
) -> None:
    rows = build_compact_rows(results)
    lookup = {
        (str(row["model"]), str(row["mode"]), str(row.get("condition", "ALL"))): row
        for row in rows
    }
    scopes = ["ALL", *conditions]
    fig, axes = plt.subplots(
        len(MODEL_MODE_ORDER), len(scopes),
        figsize=(3.15 * len(scopes), 2.85 * len(MODEL_MODE_ORDER)),
        squeeze=False,
    )
    colors = [PLOT_COLORS[k] for k in ("color", "word", "null", "mis")]

    for row_index, (model, mode) in enumerate(MODEL_MODE_ORDER):
        for col_index, scope in enumerate(scopes):
            ax = axes[row_index, col_index]
            row = lookup.get((model, mode, scope))
            values = _pie_values(row or {})
            if sum(values) > 0:
                ax.pie(
                    values,
                    colors=colors,
                    startangle=90,
                    counterclock=False,
                    autopct=_autopct,
                    pctdistance=0.70,
                    textprops={"fontsize": 6.5},
                    wedgeprops={"linewidth": 0.6, "edgecolor": "#0d0d0d"},
                )
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.axis("off")
            if row_index == 0:
                ax.set_title(scope, fontsize=10)
            if col_index == 0:
                ax.text(
                    -1.45, 0.0, _mode_label(model, mode),
                    ha="right", va="center", fontsize=9, rotation=90,
                )

    handles = [
        Patch(facecolor=PLOT_COLORS[key], label=label)
        for key, label in (
            ("color", "ink color"),
            ("word", "written word"),
            ("null", "NULL"),
            ("mis", "other"),
        )
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, framealpha=0.2)
    fig.suptitle("Stroop choices — vanilla CLIP versus ModeMUX modes", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.06, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_modemux_source_diagnostics_heatmap(
    results: Sequence[StroopResult],
    conditions: Sequence[str],
    out_path: Path,
) -> None:
    any_rows = [r for r in results if r.model == "modemux" and r.mode == "any"]
    rows = aggregate_results(any_rows, ["condition"])
    lookup = {str(row["condition"]): row for row in rows}
    metrics = [
        ("source_present_mean", "present"),
        ("source_readable_mean", "readable"),
        ("source_gate_mean", "present × readable"),
        ("glyph_max_mean", "glyph max"),
    ]
    data = np.full((len(metrics), len(conditions)), np.nan, dtype=np.float64)
    for yi, (field, _) in enumerate(metrics):
        for xi, condition in enumerate(conditions):
            data[yi, xi] = safe_float(lookup.get(condition, {}).get(field))

    fig, ax = plt.subplots(figsize=(max(6.0, 1.7 * len(conditions)), 4.2))
    image = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([label for _, label in metrics])
    ax.set_title("ModeMUX source diagnostics")
    for yi in range(data.shape[0]):
        for xi in range(data.shape[1]):
            value = data[yi, xi]
            if math.isfinite(value):
                ax.text(xi, yi, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_all_plots(
    results: Sequence[StroopResult],
    plot_dir: Path,
    conditions: Sequence[str],
    prompt_keys: Sequence[str],
) -> List[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    paths: List[Path] = []

    for model, mode in MODEL_MODE_ORDER:
        slug = _mode_slug(model, mode)
        heatmap_path = plot_dir / f"{slug}__prompt_heatmaps.png"
        condition_pies_path = plot_dir / f"{slug}__condition_pies.png"
        prompt_pies_path = plot_dir / f"{slug}__prompt_pies.png"
        plot_mode_prompt_heatmaps(
            results, model, mode, conditions, prompt_keys, heatmap_path
        )
        plot_mode_condition_pies(
            results, model, mode, conditions, condition_pies_path
        )
        plot_mode_prompt_pies(results, model, mode, prompt_pies_path)
        paths.extend([heatmap_path, condition_pies_path, prompt_pies_path])

    comparison_path = plot_dir / "all_modes__condition_pies.png"
    source_path = plot_dir / "modemux__source_diagnostics_heatmap.png"
    plot_all_modes_condition_pies(results, conditions, comparison_path)
    plot_modemux_source_diagnostics_heatmap(results, conditions, source_path)
    paths.extend([comparison_path, source_path])
    return paths


# =============================================================================
# Compact report and share bundle
# =============================================================================


def result_order_key(row: Mapping[str, Any]) -> Tuple[int, int, str]:
    pair = (str(row.get("model")), str(row.get("mode")))
    try:
        model_index = MODEL_MODE_ORDER.index(pair)
    except ValueError:
        model_index = 999
    condition = str(row.get("condition", "ALL"))
    condition_order = {"ALL": 0, "upright": 1, "inverted": 2, "mirrored": 3}.get(condition, 99)
    return model_index, condition_order, condition


def write_compact_markdown(
    path: Path,
    compact_rows: Sequence[Mapping[str, Any]],
    prompt_rows: Sequence[Mapping[str, Any]],
    item_count: int,
    prompt_count: int,
    learned_components: Mapping[str, Any],
) -> None:
    rows = sorted(compact_rows, key=result_order_key)
    lines = [
        "# CLIP Stroop Test — Vanilla vs ModeMUX",
        "",
        f"Images: **{item_count}**  ",
        f"Prompt variants: **{prompt_count}**  ",
        "",
        "Modes:",
        "- `pretrained / vanilla`: official OpenAI CLIP.",
        "- `ModeMUX / notext`: semantic/content lane with candidate reading disabled.",
        "- `ModeMUX / any`: automatic robust semantic mode.",
        "- `ModeMUX / text`: forced literal reading without abstention.",
        "- `ModeMUX / read + NULL`: literal reading with the internal NULL candidate.",
        "",
        "## Aggregate outcomes",
        "",
        "| Model / mode | Scope | Ink color | Written word | Null | Other | Mean color−word margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = _mode_label(str(row["model"]), str(row["mode"]))
        scope = str(row.get("condition", "ALL"))
        lines.append(
            f"| {label} | {scope} | {safe_float(row['pct_color']):.2f}% | "
            f"{safe_float(row['pct_word']):.2f}% | {safe_float(row['pct_null']):.2f}% | "
            f"{safe_float(row['pct_mis']):.2f}% | {safe_float(row['mean_margin']):.4f} |"
        )

    lines.extend(["", "## Best and worst prompts by intended target", ""])
    for model, mode in MODEL_MODE_ORDER:
        ranked = _prompt_analysis_rows(prompt_rows, [], model, mode)
        if not ranked:
            continue
        best, worst = ranked[0], ranked[-1]
        lines.append(
            f"- **{_mode_label(model, mode)}** ({best['target_label']}): "
            f"best `{best['prompt_key']}` ({safe_float(best['target_pct']):.2f}%); "
            f"worst `{worst['prompt_key']}` ({safe_float(worst['target_pct']):.2f}%)."
        )

    if learned_components:
        lines.extend(["", "## ModeMUX configuration", "", "```json", json.dumps(to_jsonable(learned_components), indent=2), "```"])

    lines.extend(
        [
            "",
            "## Files in the share bundle",
            "",
            "- `COMPACT_SUMMARY.md`",
            "- `compact_summary.csv`",
            "- `compact_summary.json`",
            "- `summary_by_prompt.csv`",
            "- `summary_by_prompt_overall.csv`",
            "- `PROMPT_ANALYSIS.md`",
            "- `*_prompt_pies.png` (one ranked prompt atlas per model/mode)",
            "- `all_modes__condition_pies.png`",
            "- `modemux__source_diagnostics_heatmap.png`",
            "- `run_config.json`",
            "- `learned_components.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_share_bundle(
    zip_path: Path,
    files: Sequence[Path],
) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            if path.exists():
                archive.write(path, arcname=path.name)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    csv_dir = out_dir / "csv"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    stroop_dirs = {
        "upright": Path(args.upright_dir),
        "inverted": Path(args.inverted_dir),
        "mirrored": Path(args.mirrored_dir),
    }
    prompt_templates = select_prompt_templates(args.prompt_keys)
    items = parse_stroop_files(stroop_dirs, max_items=args.max_items)
    if not items:
        raise RuntimeError("No Stroop images found. Check the three input directories.")
    color_vocab = collect_color_vocab(items)
    conditions = [
        condition for condition in stroop_dirs
        if any(item.condition == condition for item in items)
    ]

    print("CLIP Stroop Test — vanilla vs ModeMUX")
    print("=======================================")
    print(f"device={args.device} autocast={args.autocast_dtype}")
    print(f"images={len(items)} colors={color_vocab}")
    print(f"prompts={len(prompt_templates)} output={out_dir}")
    for condition in conditions:
        print(f"  {condition}: {sum(item.condition == condition for item in items)}")

    run_config = {
        "modemux_model": args.modemux_model,
        "modemux_revision": args.revision or None,
        "vanilla_model": args.vanilla_model,
        "out_dir": str(out_dir),
        "device": args.device,
        "autocast_dtype": args.autocast_dtype,
        "batch_size": args.batch_size,
        "stroop_dirs": {key: str(value) for key, value in stroop_dirs.items()},
        "prompt_templates": prompt_templates,
        "color_vocab": color_vocab,
        "item_count": len(items),
        "condition_counts": {
            condition: sum(item.condition == condition for item in items)
            for condition in conditions
        },
        "modemux_modes": MODEMUX_MODE_ORDER,
        "hf_call": {
            "correction": True,
            "return_details": True,
            "pieces_fp32": True,
        },
    }
    run_config_path = out_dir / "run_config.json"
    run_config_path.write_text(
        json.dumps(to_jsonable(run_config), indent=2), encoding="utf-8"
    )

    all_results: List[StroopResult] = []

    # Official pretrained CLIP is a genuinely separate baseline.
    vanilla_model, vanilla_preprocess = load_vanilla_model(args.vanilla_model, args.device)
    vanilla_images = preprocess_all_images(items, vanilla_preprocess)
    all_results.extend(
        evaluate_vanilla(
            model=vanilla_model,
            images_cpu=vanilla_images,
            items=items,
            color_vocab=color_vocab,
            prompt_templates=prompt_templates,
            device=args.device,
            batch_size=args.batch_size,
            autocast_dtype=args.autocast_dtype,
        )
    )
    del vanilla_images, vanilla_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Released ModeMUX HF API: AutoModel/AutoProcessor + mode=...
    modemux_model, modemux_processor = load_modemux_model(
        args.modemux_model,
        args.device,
        revision=args.revision,
    )
    learned_components = learned_component_summary(modemux_model)
    learned_components_path = out_dir / "learned_components.json"
    learned_components_path.write_text(
        json.dumps(to_jsonable(learned_components), indent=2), encoding="utf-8"
    )
    modemux_images = preprocess_all_images_hf(items, modemux_processor)
    all_results.extend(
        evaluate_modemux(
            model=modemux_model,
            processor=modemux_processor,
            images_cpu=modemux_images,
            items=items,
            color_vocab=color_vocab,
            prompt_templates=prompt_templates,
            device=args.device,
            batch_size=args.batch_size,
            autocast_dtype=args.autocast_dtype,
        )
    )
    del modemux_images, modemux_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Tables.
    detail_path = csv_dir / "stroop_results_detailed.csv"
    prompt_summary_path = csv_dir / "summary_by_prompt.csv"
    prompt_overall_summary_path = csv_dir / "summary_by_prompt_overall.csv"
    condition_summary_path = csv_dir / "summary_by_condition.csv"
    overall_summary_path = csv_dir / "summary_overall.csv"
    compact_csv_path = out_dir / "compact_summary.csv"
    compact_json_path = out_dir / "compact_summary.json"

    write_detailed_csv(detail_path, all_results)
    prompt_rows = aggregate_results(all_results, ["model", "mode", "condition", "prompt_key"])
    prompt_overall_rows = aggregate_results(all_results, ["model", "mode", "prompt_key"])
    condition_rows = aggregate_results(all_results, ["model", "mode", "condition"])
    overall_rows = aggregate_results(all_results, ["model", "mode"])
    compact_rows = build_compact_rows(all_results)

    write_csv_rows(prompt_summary_path, prompt_rows)
    write_csv_rows(prompt_overall_summary_path, prompt_overall_rows)
    write_csv_rows(condition_summary_path, condition_rows)
    write_csv_rows(overall_summary_path, overall_rows)
    write_csv_rows(compact_csv_path, compact_rows)
    compact_json_path.write_text(
        json.dumps(to_jsonable(compact_rows), indent=2), encoding="utf-8"
    )

    # Human-readable prompt analysis: the CSV remains exhaustive, but these
    # tables and plots make the prompt-extortion story visible immediately.
    print_prompt_analysis(prompt_overall_rows, prompt_rows)
    prompt_analysis_path = out_dir / "PROMPT_ANALYSIS.md"
    write_prompt_analysis_markdown(
        prompt_analysis_path, prompt_overall_rows, prompt_rows
    )

    plot_paths = save_all_plots(
        results=all_results,
        plot_dir=plot_dir,
        conditions=conditions,
        prompt_keys=list(prompt_templates),
    )

    compact_md_path = out_dir / "COMPACT_SUMMARY.md"
    write_compact_markdown(
        path=compact_md_path,
        compact_rows=compact_rows,
        prompt_rows=prompt_overall_rows,
        item_count=len(items),
        prompt_count=len(prompt_templates),
        learned_components=learned_components,
    )

    share_plot_names = {
        "all_modes__condition_pies.png",
        "modemux__source_diagnostics_heatmap.png",
    }
    share_plots = [
        path for path in plot_paths
        if path.name in share_plot_names or path.name.endswith("__prompt_pies.png")
    ]
    share_bundle_path = out_dir / "stroop_compact_share_bundle.zip"
    build_share_bundle(
        share_bundle_path,
        [
            compact_md_path,
            compact_csv_path,
            compact_json_path,
            prompt_summary_path,
            prompt_overall_summary_path,
            prompt_analysis_path,
            run_config_path,
            learned_components_path,
            *share_plots,
        ],
    )

    print("\nCompleted.")
    print(f"  Detailed CSV:    {detail_path}")
    print(f"  Prompt summary:  {prompt_summary_path}")
    print(f"  Prompt analysis: {prompt_analysis_path}")
    print(f"  Compact summary: {compact_md_path}")
    print(f"  All plots:       {plot_dir}")
    print(f"  Share bundle:    {share_bundle_path}")

    print("\nCompact overall result:")
    for row in sorted(overall_rows, key=result_order_key):
        print(
            f"  {_mode_label(str(row['model']), str(row['mode'])):28s} "
            f"color={safe_float(row['pct_color']):6.2f}% "
            f"word={safe_float(row['pct_word']):6.2f}% "
            f"null={safe_float(row['pct_null']):6.2f}% "
            f"margin={safe_float(row['mean_margin']):9.4f}"
        )


if __name__ == "__main__":
    main()
