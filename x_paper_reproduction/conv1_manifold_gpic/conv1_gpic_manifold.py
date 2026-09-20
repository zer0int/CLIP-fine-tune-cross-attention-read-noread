#!/usr/bin/env python3
"""Conv1 semantic-manifold experiment against model-matched GPIC reference banks.

Scientific design
-----------------
For explicitly configured Conv1 channels / channel pairs, interpolate the Conv1
output activation map toward FLIP, SHUFFLE, or ABS targets and follow the model
through four synchronized views:

1. BACKBONE final image embedding (raw visual CLS projection).
2. CONTENT image embedding (BACKBONE + trained CONTENT correction).
3. GPIC nearest-image neighborhoods in every final embedding space exposed by the runtime.
4. For ModeMUX/x-attention models only, vocabulary behavior in CONTENT / forced READ / automatic ANY lanes.

Single-filter experiments produce one-dimensional alpha trajectories. Explicitly
listed pairs produce full alpha1 x alpha2 surfaces, including matched single-axis
controls at alpha1=0 / alpha2=0 and additive-vs-actual nonlinear geometry.

The runner is deliberately release-oriented:
* no pickle/PT outputs;
* trajectory embeddings are safetensors;
* tabular outputs are CSV + Parquet;
* reference banks may be a local folder or an HF dataset repo;
* per-forward-group parts make the expensive model stage resumable;
* cached vocabulary tensors are safetensors;
* a cached scorer is preflight-checked against public model.forward_modes().

The default vocabulary strategy is ``hybrid``. CONTENT top-k is globally exact.
READ/ANY are exact within a broad per-batch candidate pool formed from the union
of global CONTENT and READ-text-proxy neighbors. ``--vocab_strategy exact`` does
an exact candidate-conditioned scan of the entire vocabulary and is much slower.
Every vocabulary row records its search scope.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_REFERENCE_BANK = "zer0int/CLIP-GPIC-embeddings"
DEFAULT_OUTPUT_DIR = Path("out_paper_reproduction/conv1/gpic_manifold")
DEFAULT_VOCAB_RELATIVE = Path("utils_datasets/clipese/vocab_deduped.txt")
DEFAULT_IMAGE_DIR = Path("image_sets/retrieval")
DEFAULT_CONFIG_RELATIVE = Path("x_paper_reproduction/conv1_manifold_gpic/experiment_config.json")
RUN_FORMAT_VERSION = 1
TRAJECTORY_KEY_BACKBONE = "backbone"
TRAJECTORY_KEY_CONTENT = "content"
ALLOWED_CONDITIONS = {"FLIP", "SHUFFLE", "ABS"}
EPS = 1.0e-12


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (Path.cwd(), here.parents[2], here.parents[1]):
        if (candidate / "attnclip_mechinterp_xattn").is_dir() and (
            candidate / "utils_clip_loader"
        ).is_dir():
            return candidate.resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from x_paper_reproduction.conv1_embedding_banks.bank_io import (  # noqa: E402
    BankUnavailable,
    EmbeddingBank,
    gpic_access_probe,
    neighbor_image as bank_neighbor_image,
    print_gpic_access_warning,
    public_record,
    resolve_bank_descriptor,
    resolve_bank_source,
    search_collection,
)
from utils_clip_loader.mechinterp_auto import load_mechinterp_clip_anything  # noqa: E402


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {
            str(k): v
            for k, v in vars(value).items()
            if isinstance(v, (str, int, float, bool, type(None), dict, list, tuple))
        }
    return {"value": str(value)}


def model_architecture_summary(model: torch.nn.Module) -> dict[str, Any]:
    implant = getattr(model, "read_implant", None)
    summary: dict[str, Any] = {
        "implant_kind": str(getattr(model, "implant_kind", "vanilla")),
        "model_dtype": str(getattr(model, "dtype", "unknown")),
        "input_resolution": int(getattr(model.visual, "input_resolution", -1)),
        "output_dim": int(getattr(model.visual, "output_dim", -1)),
        "read_attention_architecture": str(getattr(model, "read_attention_architecture", "none")),
        "read_null_enabled": bool(getattr(model, "read_null_enabled", False)),
        "read_null_insert_block": int(getattr(model, "read_null_insert_block", -1)),
        "content_correction_default": bool(
            getattr(model, "_clip_apply_content_correction_by_default", False)
        ),
    }
    if implant is not None:
        for name, fn_name in (
            ("read_tap_blocks", "tap_block_list"),
            ("ortho_tap_blocks", "ortho_block_list"),
            ("source_tap_blocks", "source_block_list"),
            ("capture_blocks", "capture_block_list"),
        ):
            fn = getattr(implant, fn_name, None)
            if callable(fn):
                try:
                    summary[name] = [int(x) for x in fn()]
                except Exception:
                    pass
        auto_read_scale = getattr(implant, "auto_read_scale", None)
        if auto_read_scale is not None:
            try:
                summary["auto_read_scale"] = float(auto_read_scale.detach().cpu())
            except Exception:
                pass
    return summary


def amp_context(device: str, enabled: bool):
    if enabled and str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def thumb(image: Image.Image, size: tuple[int, int] = (336, 336)) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def _tokenize(texts: Sequence[str], *, truncate: bool = True) -> torch.Tensor:
    # Tokenization is needed only by the ModeMUX vocabulary analysis. Vanilla
    # CLIP runs disable that branch, so do not import the x-attn package merely
    # to parse CLI arguments or run GPIC retrieval with an OAI checkpoint.
    import attnclip_mechinterp_xattn as clip

    return clip.tokenize(list(texts), truncate=truncate)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def write_csv_and_parquet(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    atomic_write_parquet(frame, stem.with_suffix(".parquet"))


def safe_name(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return out or "item"


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image.copy()


def pca2(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return np.empty((0, 2), dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    if len(x) == 1:
        return np.zeros((1, 2), dtype=np.float64)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(2, vt.shape[0])]
    proj = centered @ components.T
    if proj.shape[1] == 1:
        proj = np.concatenate([proj, np.zeros((len(proj), 1))], axis=1)
    return proj[:, :2]


def normalize_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, EPS)


def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def scalar_or_nan(value: Optional[torch.Tensor], index: int) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value[index].detach().float().cpu())
    except Exception:
        return float("nan")


def get_model_math(model):
    mod = importlib.import_module(model.__class__.__module__)
    needed = ("_fp32_normalize", "_fp32_scaled_matmul", "_fp32_scaled_einsum")
    missing = [name for name in needed if not hasattr(mod, name)]
    if missing:
        raise RuntimeError(
            f"{mod.__name__} is not the expected current ModeMUX implementation; "
            f"missing {missing}"
        )
    return tuple(getattr(mod, name) for name in needed)


def has_xattn_content_path(model) -> bool:
    return bool(
        getattr(model, "read_implant", None) is not None
        and callable(getattr(model, "encode_image_states", None))
        and callable(getattr(model, "_content_image_from_info", None))
    )


def runtime_embedding_spaces(model) -> tuple[str, ...]:
    return ("backbone", "content") if has_xattn_content_path(model) else ("backbone",)


def load_reproduction_model(
    model_source: str,
    *,
    device: str,
    model_revision: str | None,
    hf_cache_dir: str | None,
):
    model, preprocess, info, auto_meta = load_mechinterp_clip_anything(
        model_source,
        device=device,
        jit=False,
        cache_dir=hf_cache_dir,
        revision=model_revision,
        strict=True,
        allow_unsafe_hf_pickle=False,
    )
    model.eval()
    return model, preprocess, info, auto_meta


# -----------------------------------------------------------------------------
# Experiment configuration: singles + explicitly selected pairs only
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Intervention:
    channel: int
    condition: str


@dataclass(frozen=True)
class ExperimentSpec:
    kind: str  # single | pair
    name: str
    first: Intervention
    second: Optional[Intervention]
    alphas1: Tuple[float, ...]
    alphas2: Tuple[float, ...]


@dataclass(frozen=True)
class ForwardGroup:
    group_index: int
    group_id: str
    kind: str
    experiment_name: str
    first: Intervention
    second: Optional[Intervention]
    alpha1: float
    alpha2: float
    event_start: int
    n_images: int

    @property
    def event_stop(self) -> int:
        return self.event_start + self.n_images

    def hook_specs(self) -> List[Tuple[int, str, float]]:
        out = [(self.first.channel, self.first.condition, float(self.alpha1))]
        if self.second is not None:
            out.append((self.second.channel, self.second.condition, float(self.alpha2)))
        return out


def _float_tuple(values: Sequence[Any], label: str) -> Tuple[float, ...]:
    out = tuple(float(x) for x in values)
    if not out:
        raise ValueError(f"{label} must be non-empty")
    if len(set(out)) != len(out):
        raise ValueError(f"{label} contains duplicates: {out}")
    if any(not np.isfinite(x) for x in out):
        raise ValueError(f"{label} contains non-finite values")
    return out


def _intervention(obj: Mapping[str, Any], label: str) -> Intervention:
    channel = int(obj["channel"])
    condition = str(obj["condition"]).upper().strip()
    if condition not in ALLOWED_CONDITIONS:
        raise ValueError(f"{label}: condition must be one of {sorted(ALLOWED_CONDITIONS)}")
    return Intervention(channel=channel, condition=condition)


def load_experiment_specs(path: Path) -> Tuple[dict[str, Any], List[ExperimentSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a JSON object")
    default_alphas = _float_tuple(raw.get("alphas", [0.0, 0.3, 0.6, 0.8, 1.0]), "alphas")
    specs: List[ExperimentSpec] = []
    names: set[str] = set()

    for idx, obj in enumerate(raw.get("singles", [])):
        if not bool(obj.get("enabled", True)):
            continue
        first = _intervention(obj, f"singles[{idx}]")
        alphas = _float_tuple(obj.get("alphas", default_alphas), f"singles[{idx}].alphas")
        if len(alphas) < 2:
            raise ValueError(f"single {idx} needs at least two alpha points")
        if 0.0 not in alphas:
            raise ValueError(f"single {idx} must include alpha=0 for matched baseline geometry")
        name = str(obj.get("name") or f"ch{first.channel:04d}_{first.condition.lower()}")
        if name in names:
            raise ValueError(f"Duplicate experiment name {name!r}")
        names.add(name)
        specs.append(
            ExperimentSpec(
                kind="single",
                name=name,
                first=first,
                second=None,
                alphas1=alphas,
                alphas2=(0.0,),
            )
        )

    for idx, obj in enumerate(raw.get("pairs", [])):
        if not bool(obj.get("enabled", True)):
            continue
        first = _intervention(obj["first"], f"pairs[{idx}].first")
        second = _intervention(obj["second"], f"pairs[{idx}].second")
        if first.channel == second.channel:
            raise ValueError(f"pairs[{idx}] uses the same Conv1 channel twice")
        alphas1 = _float_tuple(obj.get("alphas1", obj.get("alphas", default_alphas)), f"pairs[{idx}].alphas1")
        alphas2 = _float_tuple(obj.get("alphas2", obj.get("alphas", default_alphas)), f"pairs[{idx}].alphas2")
        if len(alphas1) < 2 or len(alphas2) < 2:
            raise ValueError(f"pair {idx} needs at least two points on each alpha axis")
        if 0.0 not in alphas1 or 0.0 not in alphas2:
            raise ValueError(f"pair {idx} must include alpha1=0 and alpha2=0")
        name = str(
            obj.get("name")
            or f"ch{first.channel:04d}_{first.condition.lower()}__ch{second.channel:04d}_{second.condition.lower()}"
        )
        if name in names:
            raise ValueError(f"Duplicate experiment name {name!r}")
        names.add(name)
        specs.append(
            ExperimentSpec(
                kind="pair",
                name=name,
                first=first,
                second=second,
                alphas1=alphas1,
                alphas2=alphas2,
            )
        )

    if not specs:
        raise RuntimeError("Experiment config contains no enabled singles or pairs")
    return raw, specs


def scan_source_images(image_dir: Path, recursive: bool) -> pd.DataFrame:
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    files = sorted(
        [p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in extensions],
        key=lambda p: (p.name.lower(), str(p).lower()),
    )
    if not files:
        raise RuntimeError(f"No source images in {image_dir}")
    seen: Dict[str, int] = {}
    rows = []
    for source_index, path in enumerate(files):
        base = path.stem
        count = seen.get(base, 0)
        seen[base] = count + 1
        stim_id = base if count == 0 else f"{base}__dup{count}"
        rows.append(
            {
                "source_index": source_index,
                "stim_id": stim_id,
                "filename": path.name,
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def build_events(
    source_manifest: pd.DataFrame,
    specs: Sequence[ExperimentSpec],
) -> Tuple[pd.DataFrame, List[ForwardGroup]]:
    rows: List[dict[str, Any]] = []
    groups: List[ForwardGroup] = []
    event_row = 0
    group_index = 0
    n_images = len(source_manifest)

    for experiment_order, spec in enumerate(specs):
        if spec.kind == "single":
            cells = [(a, 0.0) for a in spec.alphas1]
        else:
            cells = [(a1, a2) for a1 in spec.alphas1 for a2 in spec.alphas2]
        for cell_order, (alpha1, alpha2) in enumerate(cells):
            group_id = f"g{group_index:05d}_{safe_name(spec.name)}_a{alpha1:g}_b{alpha2:g}"
            group = ForwardGroup(
                group_index=group_index,
                group_id=group_id,
                kind=spec.kind,
                experiment_name=spec.name,
                first=spec.first,
                second=spec.second,
                alpha1=float(alpha1),
                alpha2=float(alpha2),
                event_start=event_row,
                n_images=n_images,
            )
            groups.append(group)
            for source in source_manifest.itertuples(index=False):
                rows.append(
                    {
                        "event_row": event_row,
                        "group_index": group_index,
                        "group_id": group_id,
                        "experiment_order": experiment_order,
                        "cell_order": cell_order,
                        "kind": spec.kind,
                        "experiment_name": spec.name,
                        "source_index": int(source.source_index),
                        "stim_id": str(source.stim_id),
                        "filename": str(source.filename),
                        "path": str(source.path),
                        "channel1": int(spec.first.channel),
                        "condition1": str(spec.first.condition),
                        "alpha1": float(alpha1),
                        "channel2": int(spec.second.channel) if spec.second is not None else pd.NA,
                        "condition2": str(spec.second.condition) if spec.second is not None else "",
                        "alpha2": float(alpha2) if spec.second is not None else np.nan,
                    }
                )
                event_row += 1
            group_index += 1
    return pd.DataFrame(rows), groups


# -----------------------------------------------------------------------------
# Image loading and deterministic Conv1 intervention
# -----------------------------------------------------------------------------


class SourceImageDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, preprocess):
        self.rows = manifest.to_dict("records")
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["path"])
        with Image.open(path) as image:
            tensor = self.preprocess(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, int(row["source_index"]), str(row["stim_id"])


def source_collate(batch):
    tensors, indices, stim_ids = zip(*batch)
    return torch.stack(tensors, dim=0), torch.tensor(indices, dtype=torch.long), list(stim_ids)


class MultiAlphaConv1OutputTransform:
    """Intervene on one or more distinct Conv1 output channels for one forward."""

    def __init__(
        self,
        model,
        specs: Sequence[Tuple[int, str, float]],
        stim_ids: Sequence[str],
    ):
        self.model = model
        self.specs = [(int(c), str(cond).upper(), float(a)) for c, cond, a in specs]
        self.stim_ids = list(map(str, stim_ids))
        self.handle = None

    def _target(self, r: torch.Tensor, channel: int, condition: str) -> torch.Tensor:
        if condition == "FLIP":
            return -r
        if condition == "ABS":
            return torch.abs(r)
        if condition == "SHUFFLE":
            batch, height, width = r.shape
            flat = r.flatten(1)
            out = []
            for batch_index in range(batch):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    stable_seed(f"ch{channel}:{condition}:{self.stim_ids[batch_index]}")
                )
                perm = torch.randperm(height * width, generator=generator).to(r.device)
                out.append(flat[batch_index].index_select(0, perm).reshape(height, width))
            return torch.stack(out, dim=0)
        raise ValueError(condition)

    def __enter__(self):
        specs = self.specs

        def hook(_module, _inputs, output):
            if output.ndim != 4:
                raise RuntimeError(f"Unexpected Conv1 output shape {tuple(output.shape)}")
            y = output.clone()
            channels = y.shape[1]
            for channel, condition, alpha in specs:
                if not (0 <= channel < channels):
                    raise IndexError(f"Conv1 channel {channel} outside [0,{channels})")
                if alpha == 0.0:
                    continue
                r = y[:, channel].clone()
                target = self._target(r, channel, condition)
                y[:, channel] = r + alpha * (target - r)
            return y

        self.handle = self.model.visual.conv1.register_forward_hook(hook)
        return self

    def __exit__(self, *_exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# -----------------------------------------------------------------------------
# ModeMUX cached image state and vocabulary bank
# -----------------------------------------------------------------------------


@dataclass
class ModeMuxImageCache:
    mode_mux: bool
    info: Dict[str, Any]
    backbone_raw: torch.Tensor
    content_raw: torch.Tensor
    backbone: torch.Tensor
    content: torch.Tensor
    correction: torch.Tensor
    source_logits: torch.Tensor
    source_stats: torch.Tensor
    source_gate: torch.Tensor
    null_read_logits: torch.Tensor
    null_query_read_null_attention: Optional[torch.Tensor]
    injection: torch.Tensor


@dataclass
class VocabBank:
    words: List[str]
    content: torch.Tensor
    read: torch.Tensor
    query: torch.Tensor
    vocab_sha256: str
    search_device: str


@torch.inference_mode()
def make_image_cache(model, images: torch.Tensor) -> ModeMuxImageCache:
    conv1 = getattr(getattr(model, "visual", None), "conv1", None)
    weight = getattr(conv1, "weight", None)
    if isinstance(weight, torch.Tensor) and images.dtype != weight.dtype:
        images = images.to(dtype=weight.dtype)

    if not has_xattn_content_path(model):
        backbone_raw = model.encode_image(images)
        backbone = F.normalize(backbone_raw.float(), dim=-1)
        batch = int(backbone.shape[0])
        device = backbone.device
        nan2 = torch.full((batch, 2), float("nan"), device=device)
        nan4 = torch.full((batch, 4), float("nan"), device=device)
        nan1 = torch.full((batch,), float("nan"), device=device)
        zeros = torch.zeros_like(backbone)
        return ModeMuxImageCache(
            mode_mux=False,
            info={},
            backbone_raw=backbone_raw,
            content_raw=backbone_raw,
            backbone=backbone,
            content=backbone,
            correction=zeros,
            source_logits=nan2,
            source_stats=nan4,
            source_gate=nan1,
            null_read_logits=nan1,
            null_query_read_null_attention=None,
            injection=zeros,
        )

    normalize, _matmul, einsum = get_model_math(model)
    if str(getattr(model, "read_attention_architecture", "")) == "sigmoid_mass":
        raise RuntimeError(
            "This experiment targets current sigmoid_all/softmax ModeMUX; old sigmoid_mass "
            "requires evidence-weighted scoring and is intentionally unsupported."
        )

    info = model.encode_image_states(images, return_final_tokens=False)
    backbone_raw = info["image_embedding"]
    content_raw = model._content_image_from_info(info, apply_content_correction=True)
    backbone = normalize(backbone_raw)
    content = normalize(content_raw)
    correction = content_raw.float() - backbone_raw.float()

    source_logits, _glyph_logits, source_stats = model.read_implant.source_outputs(
        info["states"], return_details=False
    )
    source_probs = source_logits.sigmoid()
    source_gate = source_probs[:, 0] * source_probs[:, 1]

    null_tokens = model._null_read_tokens(images.device)
    null_info = model._encode_text_hidden(null_tokens)
    null_text = normalize(null_info["text_embedding"])
    null_query_read_null_attention = None
    if bool(getattr(model, "read_null_enabled", False)):
        null_feature, null_details = model.read_implant.read_features(
            info["states"],
            null_info["eot_hidden_pre_ln"],
            register_mask=info["register_mask"],
            return_details=True,
        )
        try:
            null_query_read_null_attention = null_details["read_null_attention"][:, 0]
        except Exception:
            null_query_read_null_attention = None
    else:
        null_feature = model.read_implant.read_features(
            info["states"],
            null_info["eot_hidden_pre_ln"],
            register_mask=info["register_mask"],
        )
    null_feature = normalize(null_feature[:, 0, :])
    scale = model.logit_scale.float().exp()
    raw_null = einsum("bd,d->b", scale, null_feature, null_text[0])
    null_read_logits = model.read_implant.calibrate_read_logits(
        raw_null[:, None],
        source_logits,
        null_mask=torch.ones(1, dtype=torch.bool, device=images.device),
    )[:, 0]
    injection = model.read_implant.injection_feature(content)

    return ModeMuxImageCache(
        mode_mux=True,
        info=info,
        backbone_raw=backbone_raw,
        content_raw=content_raw,
        backbone=backbone,
        content=content,
        correction=correction,
        source_logits=source_logits,
        source_stats=source_stats,
        source_gate=source_gate,
        null_read_logits=null_read_logits,
        null_query_read_null_attention=null_query_read_null_attention,
        injection=injection,
    )


def read_vocab_lines(path: Path) -> List[str]:
    words: List[str] = []
    seen: set[str] = set()
    reserved = {"<text>", "<notext>", "<any>", "<null>"}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        word = raw.strip()
        if not word or word in seen or any(token in word for token in reserved):
            continue
        seen.add(word)
        words.append(word)
    if not words:
        raise RuntimeError(f"Vocabulary is empty: {path}")
    return words


def vocab_cache_signature(
    *, model_source: str, model_revision: Optional[str], vocab_sha256: str, n_words: int
) -> dict[str, Any]:
    return {
        # v2: text targets and READ queries are cached losslessly in FP32.
        # This intentionally matches the benchmark scorer and keeps the strict
        # forward_modes() numerical preflight meaningful.
        "format_version": 2,
        "tensor_dtype": "float32",
        "model_source": str(model_source),
        "model_revision": model_revision or "",
        "vocab_sha256": str(vocab_sha256),
        "n_words": int(n_words),
    }


@torch.inference_mode()
def build_or_load_vocab_bank(
    model,
    *,
    vocab_path: Path,
    cache_dir: Path,
    model_source: str,
    model_revision: Optional[str],
    device: str,
    batch_size: int,
    use_amp: bool,
    force_rebuild: bool,
) -> VocabBank:
    words = read_vocab_lines(vocab_path)
    vocab_sha = sha256_file(vocab_path)
    signature = vocab_cache_signature(
        model_source=model_source,
        model_revision=model_revision,
        vocab_sha256=vocab_sha,
        n_words=len(words),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = cache_dir / "modemux_vocab_cache.safetensors"
    meta_path = cache_dir / "modemux_vocab_cache.json"
    words_path = cache_dir / "vocab_entries.txt"

    reusable = False
    if not force_rebuild and tensor_path.is_file() and meta_path.is_file() and words_path.is_file():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            cached_words = words_path.read_text(encoding="utf-8", errors="replace").splitlines()
            reusable = all(old.get(k) == v for k, v in signature.items()) and cached_words == words
        except Exception:
            reusable = False

    if not reusable:
        normalize, _matmul, _einsum = get_model_math(model)
        content_parts: List[torch.Tensor] = []
        read_parts: List[torch.Tensor] = []
        query_parts: List[torch.Tensor] = []
        for start in tqdm(
            range(0, len(words), int(batch_size)),
            desc="ModeMUX vocab cache",
            unit="batch",
        ):
            chunk = words[start : start + int(batch_size)]
            tokens = _tokenize(["<any> " + word for word in chunk], truncate=True).to(device)
            prepared = model.prepare_mode_tokens(tokens)
            if not prepared["modes"].eq(0).all():
                raise RuntimeError("<any> vocabulary preparation did not parse as ANY mode")
            with amp_context(device, use_amp):
                content_info = model._encode_text_hidden(prepared["content_tokens"])
                read_info = model._encode_text_hidden(prepared["read_tokens"])
            # Keep this cache FP32. The current benchmark caches these same
            # quantities in FP32; FP16 here can move scaled CLIP logits by a
            # few 1e-3 and spuriously fail the strict forward_modes preflight.
            content_parts.append(normalize(content_info["text_embedding"]).float().cpu())
            read_parts.append(normalize(read_info["text_embedding"]).float().cpu())
            query_parts.append(read_info["eot_hidden_pre_ln"].float().cpu())

        tensors = {
            "content_text": torch.cat(content_parts, dim=0).contiguous(),
            "read_text": torch.cat(read_parts, dim=0).contiguous(),
            "read_query": torch.cat(query_parts, dim=0).contiguous(),
        }
        tmp = tensor_path.with_name(tensor_path.name + ".part")
        tmp.unlink(missing_ok=True)
        save_safetensors(
            tensors,
            str(tmp),
            metadata={k: str(v) for k, v in signature.items()},
        )
        os.replace(tmp, tensor_path)
        words_path.write_text("\n".join(words), encoding="utf-8")
        atomic_write_json(meta_path, {**signature, "tensor_file": tensor_path.name})

    tensors = load_safetensors(str(tensor_path), device="cpu")
    search_device = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    # v2 stores the three text-side tensors in FP32 (~0.5 GB for ~53k x 768).
    # Do not downcast: scorer/preflight parity is more important than ~250 MB.
    content = tensors["content_text"].to(search_device, dtype=torch.float32)
    read = tensors["read_text"].to(search_device, dtype=torch.float32)
    query = tensors["read_query"].to(search_device, dtype=torch.float32)
    if not (len(words) == len(content) == len(read) == len(query)):
        raise RuntimeError("Vocabulary cache row count mismatch")
    return VocabBank(words, content, read, query, vocab_sha, search_device)


@torch.inference_mode()
def score_candidate_ids(
    model,
    cache: ModeMuxImageCache,
    vocab: VocabBank,
    candidate_ids: torch.Tensor,
    *,
    return_details: bool,
) -> Dict[str, torch.Tensor]:
    normalize, matmul, einsum = get_model_math(model)
    device = cache.content.device
    ids = candidate_ids.to(vocab.content.device, dtype=torch.long)
    content_text = vocab.content.index_select(0, ids).to(device)
    read_text = vocab.read.index_select(0, ids).to(device)
    query = vocab.query.index_select(0, ids).to(device)
    scale = model.logit_scale.float().exp()

    content_cosine = cache.content.float() @ content_text.float().t()
    read_proxy_cosine = cache.content.float() @ read_text.float().t()
    content_logits = matmul(scale, cache.content, content_text.t())

    read_details = None
    if return_details and bool(getattr(model, "read_null_enabled", False)):
        read_feature, read_details = model.read_implant.read_features(
            cache.info["states"],
            query,
            register_mask=cache.info["register_mask"],
            return_details=True,
        )
    else:
        read_feature = model.read_implant.read_features(
            cache.info["states"],
            query,
            register_mask=cache.info["register_mask"],
        )
    read_feature = normalize(read_feature)
    ortho_feature = normalize(
        model.read_implant.orthographic_features(cache.info["states"], query)
    )
    raw_read_logits = einsum("bnd,nd->bn", scale, read_feature, read_text)
    early_logits = einsum("bnd,nd->bn", scale, ortho_feature, read_text)
    read_logits = model.read_implant.calibrate_read_logits(
        raw_read_logits,
        cache.source_logits,
        null_mask=torch.zeros(len(ids), dtype=torch.bool, device=device),
    )
    relative = read_logits - cache.null_read_logits[:, None]
    positive = model.read_implant.positive_relative_read(relative).detach()
    trust = model.read_implant.trust_gate(
        content_logits=content_logits,
        read_logits=read_logits,
        null_logits=cache.null_read_logits,
        early_logits=early_logits,
        content_image=cache.content,
        read_image=read_feature,
        content_text=content_text,
        read_text=read_text,
        source_logits=cache.source_logits,
        source_stats=cache.source_stats,
        injection=cache.injection,
    )
    route = trust * cache.source_gate[:, None].detach()
    auto_contribution = (
        route.to(positive.dtype)
        * model.read_implant.auto_read_scale.to(positive.dtype)
        * positive
    ).to(content_logits.dtype)
    any_logits = content_logits + auto_contribution

    read_null_attention = None
    if read_details is not None:
        value = read_details.get("read_null_attention")
        if value is not None:
            read_null_attention = value.to(content_logits.dtype)

    return {
        "content": content_logits.float(),
        "read": read_logits.float(),
        "any": any_logits.float(),
        "content_cosine": content_cosine.float(),
        "read_proxy_cosine": read_proxy_cosine.float(),
        "raw_read": raw_read_logits.float(),
        "raw_read_cosine": (raw_read_logits.float() / scale.float()).float(),
        "early": early_logits.float(),
        "early_cosine": (early_logits.float() / scale.float()).float(),
        "relative_read": relative.float(),
        "trust_gate": trust.float(),
        "route_gate": route.float(),
        "auto_read_contribution": auto_contribution.float(),
        "read_null_attention": read_null_attention,
    }


def _merge_topk(
    best_values: torch.Tensor,
    best_ids: torch.Tensor,
    values: torch.Tensor,
    ids: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = values.shape[0]
    candidate_ids = ids.view(1, -1).expand(batch, -1)
    all_values = torch.cat([best_values, values], dim=1)
    all_ids = torch.cat([best_ids, candidate_ids], dim=1)
    best_values, order = torch.topk(all_values, k=min(k, all_values.shape[1]), dim=1)
    best_ids = torch.gather(all_ids, 1, order)
    return best_values, best_ids


@torch.inference_mode()
def select_vocab_topk(
    model,
    cache: ModeMuxImageCache,
    vocab: VocabBank,
    *,
    strategy: str,
    topk: int,
    pool_per_proxy: int,
    exact_chunk_size: int,
) -> Tuple[Dict[str, torch.Tensor], str, int]:
    """Return top IDs for CONTENT/READ/ANY plus search-scope metadata."""
    B = cache.content.shape[0]
    N = len(vocab.words)
    K = min(int(topk), N)
    device = cache.content.device

    if strategy == "hybrid":
        # CONTENT is globally exact and also seeds the READ/ANY pool. A second
        # proxy uses the hard-READ text target against CONTENT image geometry.
        content_cos = cache.content.float() @ vocab.content.to(device).t()
        read_proxy = cache.content.float() @ vocab.read.to(device).t()
        pool_n = min(int(pool_per_proxy), N)
        _cv, content_pool = torch.topk(content_cos, pool_n, dim=1)
        _rv, read_pool = torch.topk(read_proxy, pool_n, dim=1)
        global_content_values, global_content_ids = torch.topk(
            model.logit_scale.float().exp() * content_cos, K, dim=1
        )
        union = torch.unique(torch.cat([content_pool.flatten(), read_pool.flatten(), global_content_ids.flatten()]))
        scored = score_candidate_ids(model, cache, vocab, union, return_details=False)
        read_values, read_pos = torch.topk(scored["read"], K, dim=1)
        any_values, any_pos = torch.topk(scored["any"], K, dim=1)
        result = {
            "content_ids": global_content_ids,
            "content_values": global_content_values,
            "read_ids": union[read_pos],
            "read_values": read_values,
            "any_ids": union[any_pos],
            "any_values": any_values,
        }
        return result, "hybrid_content+read_proxy_pool", int(union.numel())

    if strategy != "exact":
        raise ValueError(strategy)

    best = {
        lane: (
            torch.full((B, K), -float("inf"), device=device),
            torch.full((B, K), -1, dtype=torch.long, device=device),
        )
        for lane in ("content", "read", "any")
    }
    for start in range(0, N, int(exact_chunk_size)):
        end = min(start + int(exact_chunk_size), N)
        ids = torch.arange(start, end, dtype=torch.long, device=vocab.content.device)
        scored = score_candidate_ids(model, cache, vocab, ids, return_details=False)
        for lane in ("content", "read", "any"):
            best[lane] = _merge_topk(best[lane][0], best[lane][1], scored[lane], ids.to(device), K)

    return {
        "content_values": best["content"][0],
        "content_ids": best["content"][1],
        "read_values": best["read"][0],
        "read_ids": best["read"][1],
        "any_values": best["any"][0],
        "any_ids": best["any"][1],
    }, "exact_full_vocab", N


@torch.inference_mode()
def vocab_records_for_cache(
    model,
    cache: ModeMuxImageCache,
    vocab: VocabBank,
    *,
    event_rows: Sequence[int],
    event_meta: pd.DataFrame,
    strategy: str,
    topk: int,
    pool_per_proxy: int,
    exact_chunk_size: int,
) -> List[dict[str, Any]]:
    selected, scope, pool_size = select_vocab_topk(
        model,
        cache,
        vocab,
        strategy=strategy,
        topk=topk,
        pool_per_proxy=pool_per_proxy,
        exact_chunk_size=exact_chunk_size,
    )
    winner_ids = torch.unique(
        torch.cat(
            [
                selected["content_ids"].flatten(),
                selected["read_ids"].flatten(),
                selected["any_ids"].flatten(),
            ]
        )
    )
    detailed = score_candidate_ids(model, cache, vocab, winner_ids, return_details=True)
    id_to_col = {int(v): i for i, v in enumerate(winner_ids.detach().cpu().tolist())}
    source_probs = cache.source_logits.sigmoid()
    records: List[dict[str, Any]] = []

    for batch_index, event_row in enumerate(event_rows):
        meta = event_meta.iloc[int(event_row)]
        common = {
            "event_row": int(event_row),
            "experiment_name": str(meta.experiment_name),
            "kind": str(meta.kind),
            "stim_id": str(meta.stim_id),
            "channel1": int(meta.channel1),
            "condition1": str(meta.condition1),
            "alpha1": float(meta.alpha1),
            "channel2": int(meta.channel2) if pd.notna(meta.channel2) else pd.NA,
            "condition2": str(meta.condition2),
            "alpha2": float(meta.alpha2) if pd.notna(meta.alpha2) else np.nan,
            "vocab_search_scope": scope,
            "vocab_candidate_pool_size": int(pool_size),
            "source_present_logit": float(cache.source_logits[batch_index, 0].detach().cpu()),
            "source_ordered_logit": float(cache.source_logits[batch_index, 1].detach().cpu()),
            "source_present_prob": float(source_probs[batch_index, 0].detach().cpu()),
            "source_ordered_prob": float(source_probs[batch_index, 1].detach().cpu()),
            "source_gate": float(cache.source_gate[batch_index].detach().cpu()),
            "null_read_score": float(cache.null_read_logits[batch_index].detach().cpu()),
            "null_query_read_null_attention": scalar_or_nan(cache.null_query_read_null_attention, batch_index),
        }
        for lane in ("content", "read", "any"):
            ids = selected[f"{lane}_ids"][batch_index].detach().cpu().tolist()
            for rank, vocab_id in enumerate(ids, start=1):
                col = id_to_col[int(vocab_id)]
                read_null = detailed["read_null_attention"]
                record = {
                    **common,
                    "lane": lane,
                    "rank": rank,
                    "vocab_id": int(vocab_id),
                    "vocab_entry": vocab.words[int(vocab_id)],
                    "mode_score": float(detailed[lane][batch_index, col].detach().cpu()),
                    "content_score": float(detailed["content"][batch_index, col].detach().cpu()),
                    "content_cosine": float(detailed["content_cosine"][batch_index, col].detach().cpu()),
                    "read_proxy_cosine": float(detailed["read_proxy_cosine"][batch_index, col].detach().cpu()),
                    "raw_read_score": float(detailed["raw_read"][batch_index, col].detach().cpu()),
                    "raw_read_cosine": float(detailed["raw_read_cosine"][batch_index, col].detach().cpu()),
                    "read_score": float(detailed["read"][batch_index, col].detach().cpu()),
                    "relative_read_score": float(detailed["relative_read"][batch_index, col].detach().cpu()),
                    "early_orthographic_score": float(detailed["early"][batch_index, col].detach().cpu()),
                    "early_orthographic_cosine": float(detailed["early_cosine"][batch_index, col].detach().cpu()),
                    "trust_gate": float(detailed["trust_gate"][batch_index, col].detach().cpu()),
                    "route_gate": float(detailed["route_gate"][batch_index, col].detach().cpu()),
                    "auto_read_contribution": float(detailed["auto_read_contribution"][batch_index, col].detach().cpu()),
                    "read_null_attention": (
                        float(read_null[batch_index, col].detach().cpu()) if read_null is not None else np.nan
                    ),
                }
                records.append(record)
    return records


@torch.inference_mode()
def preflight_cached_mode_scorer(
    model,
    preprocess,
    image_path: Path,
    vocab: VocabBank,
    *,
    device: str,
    use_amp: bool,
    n_words: int,
    atol: float,
) -> None:
    n = min(int(n_words), len(vocab.words))
    if n <= 0:
        return
    with Image.open(image_path) as image:
        image_tensor = preprocess(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(device)
    with amp_context(device, use_amp):
        cache = make_image_cache(model, image_tensor)
        ids = torch.arange(n, dtype=torch.long, device=vocab.content.device)
        cached = score_candidate_ids(model, cache, vocab, ids, return_details=False)

    refs: Dict[str, torch.Tensor] = {}
    for lane, prefix in (("content", "<notext>"), ("read", "<text>"), ("any", "<any>")):
        tokens = _tokenize([f"{prefix} {word}" for word in vocab.words[:n]], truncate=True).to(device)
        with amp_context(device, use_amp):
            output = model.forward_modes(
                image_tensor,
                tokens,
                apply_content_correction=True,
                return_details=False,
            )
        refs[lane] = output[0].float()

    errors = {
        lane: float((cached[lane].float() - refs[lane]).abs().max().detach().cpu())
        for lane in refs
    }
    print("[preflight ModeMUX scorer] " + " | ".join(f"{k}={v:.6g}" for k, v in errors.items()))
    bad = {k: v for k, v in errors.items() if v > float(atol)}
    if bad:
        raise RuntimeError(
            f"Cached ModeMUX scorer disagrees with forward_modes() beyond atol={atol}: {bad}"
        )


# -----------------------------------------------------------------------------
# Resumable forward stage
# -----------------------------------------------------------------------------


def scalar_rows_for_cache(
    cache: ModeMuxImageCache,
    event_rows: Sequence[int],
    event_meta: pd.DataFrame,
) -> List[dict[str, Any]]:
    source_probs = cache.source_logits.sigmoid()
    register_mask = cache.info.get("register_mask")
    patch_norms = cache.info.get("patch_token_norms")
    read_null_norm = cache.info.get("read_null_token_norm")
    rows: List[dict[str, Any]] = []
    for batch_index, event_row in enumerate(event_rows):
        meta = event_meta.iloc[int(event_row)]
        base_raw = cache.backbone_raw[batch_index].float()
        content_raw = cache.content_raw[batch_index].float()
        correction = cache.correction[batch_index].float()
        base_norm = float(base_raw.norm().detach().cpu())
        if cache.mode_mux:
            correction_norm = float(correction.norm().detach().cpu())
            correction_cos = (
                float(F.cosine_similarity(base_raw[None], correction[None]).detach().cpu())
                if correction_norm > EPS and base_norm > EPS
                else np.nan
            )
            content_raw_norm = float(content_raw.norm().detach().cpu())
            backbone_content_cosine = float(
                (cache.backbone[batch_index] * cache.content[batch_index]).sum().detach().cpu()
            )
        else:
            correction_norm = np.nan
            correction_cos = np.nan
            content_raw_norm = np.nan
            backbone_content_cosine = np.nan
        stats = cache.source_stats[batch_index].detach().float().cpu()
        rows.append(
            {
                "event_row": int(event_row),
                "experiment_name": str(meta.experiment_name),
                "kind": str(meta.kind),
                "stim_id": str(meta.stim_id),
                "backbone_raw_norm": base_norm,
                "content_raw_norm": content_raw_norm,
                "correction_norm": correction_norm,
                "correction_cos_to_backbone_raw": correction_cos,
                "backbone_content_cosine": backbone_content_cosine,
                "register_count": (
                    int(register_mask[batch_index].sum().detach().cpu()) if register_mask is not None else -1
                ),
                "patch_norm_mean": (
                    float(patch_norms[batch_index].float().mean().detach().cpu()) if patch_norms is not None else np.nan
                ),
                "patch_norm_max": (
                    float(patch_norms[batch_index].float().max().detach().cpu()) if patch_norms is not None else np.nan
                ),
                "read_null_token_norm": scalar_or_nan(read_null_norm, batch_index),
                "source_present_logit": float(cache.source_logits[batch_index, 0].detach().cpu()),
                "source_ordered_logit": float(cache.source_logits[batch_index, 1].detach().cpu()),
                "source_present_prob": float(source_probs[batch_index, 0].detach().cpu()),
                "source_ordered_prob": float(source_probs[batch_index, 1].detach().cpu()),
                "source_gate": float(cache.source_gate[batch_index].detach().cpu()),
                "glyph_mean": float(stats[0]) if len(stats) > 0 else np.nan,
                "glyph_max": float(stats[1]) if len(stats) > 1 else np.nan,
                "glyph_topk": float(stats[2]) if len(stats) > 2 else np.nan,
                "glyph_area": float(stats[3]) if len(stats) > 3 else np.nan,
                "null_read_score": float(cache.null_read_logits[batch_index].detach().cpu()),
                "null_query_read_null_attention": scalar_or_nan(cache.null_query_read_null_attention, batch_index),
            }
        )
    return rows


def prepare_output_dir(out: Path, overwrite: bool) -> None:
    if overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "state" / "scalars").mkdir(parents=True, exist_ok=True)
    (out / "state" / "words").mkdir(parents=True, exist_ok=True)
    (out / "plots" / "singles").mkdir(parents=True, exist_ok=True)
    (out / "plots" / "pairs").mkdir(parents=True, exist_ok=True)


def trajectory_memmap_paths(out: Path, spaces: Sequence[str]) -> dict[str, Path]:
    return {space: out / "state" / f"{space}_f16.npy.part" for space in spaces}


def ensure_trajectory_memmaps(
    out: Path, n_events: int, dim: int, spaces: Sequence[str]
) -> dict[str, np.memmap]:
    paths = trajectory_memmap_paths(out, spaces)
    final_path = out / "trajectory_embeddings.safetensors"
    if final_path.is_file() and not any(path.is_file() for path in paths.values()):
        raise RuntimeError("Final trajectory safetensors already exists; forward generation should be skipped")
    arrays: dict[str, np.memmap] = {}
    for space, path in paths.items():
        if not path.is_file():
            np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=(n_events, dim)
            ).flush()
        arr = np.load(path, mmap_mode="r+")
        if tuple(arr.shape) != (n_events, dim):
            raise RuntimeError(f"Existing {space} trajectory memmap shape mismatch; use --overwrite")
        arrays[space] = arr
    return arrays


def export_trajectory_safetensors(
    out: Path, arrays: Mapping[str, np.ndarray], metadata: dict[str, str]
) -> Path:
    path = out / "trajectory_embeddings.safetensors"
    tmp = path.with_name(path.name + ".part")
    tmp.unlink(missing_ok=True)
    # Clone away from the NumPy memmap before safetensors sees the storage.
    # This prevents a Torch storage from retaining a Windows file handle after
    # the save completes, which would make cleanup of *.npy.part fail.
    tensors = {
        str(space): torch.from_numpy(np.asarray(array)).clone().contiguous()
        for space, array in arrays.items()
    }
    save_safetensors(
        tensors,
        str(tmp),
        metadata={str(k): str(v) for k, v in metadata.items()},
    )
    os.replace(tmp, path)
    return path


def close_trajectory_memmaps(arrays: Mapping[str, np.ndarray]) -> None:
    """Flush and explicitly close trajectory memmaps before temp-file cleanup.

    Relying on ``del`` is insufficient on Windows: a loop variable or NumPy/Torch
    view can keep the mapping alive and WinError 32 then prevents unlinking the
    temporary ``*.npy.part`` file.
    """
    for array in arrays.values():
        flush = getattr(array, "flush", None)
        if callable(flush):
            flush()
        mmap_obj = getattr(array, "_mmap", None)
        if mmap_obj is not None and not getattr(mmap_obj, "closed", False):
            mmap_obj.close()


def merge_group_parts(out: Path, folder: str, expected_events: int) -> pd.DataFrame:
    files = sorted((out / "state" / folder).glob("g*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)
    if "event_row" in merged.columns:
        merged = merged.sort_values(["event_row"] + (["lane", "rank"] if "lane" in merged.columns else [])).reset_index(drop=True)
    if folder == "scalars":
        unique = merged["event_row"].nunique()
        if unique != int(expected_events):
            raise RuntimeError(f"Scalar parts cover {unique} events, expected {expected_events}")
    return merged


@torch.inference_mode()
def run_forward_stage(
    args: argparse.Namespace,
    *,
    model,
    preprocess,
    source_manifest: pd.DataFrame,
    event_meta: pd.DataFrame,
    groups: Sequence[ForwardGroup],
    vocab: Optional[VocabBank],
    out: Path,
    dim: int,
    spaces: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    final_embeddings = out / "trajectory_embeddings.safetensors"
    all_scalar_parts = all((out / "state" / "scalars" / f"{g.group_id}.parquet").is_file() for g in groups)
    all_word_parts = (
        args.vocab_strategy == "off"
        or all((out / "state" / "words" / f"{g.group_id}.parquet").is_file() for g in groups)
    )
    if final_embeddings.is_file() and all_scalar_parts and all_word_parts:
        print("[forward] complete cached trajectory found; skipping model forwards")
        # A previous Windows run may have completed the safetensors export but
        # failed while unlinking an open mmap.  A fresh process no longer owns
        # those handles, so clean any stale temp arrays while accepting the
        # completed forward cache.
        if not args.keep_temp:
            for path in trajectory_memmap_paths(out, spaces).values():
                path.unlink(missing_ok=True)
        scalars = merge_group_parts(out, "scalars", len(event_meta))
        words = merge_group_parts(out, "words", len(event_meta)) if args.vocab_strategy != "off" else pd.DataFrame()
        write_csv_and_parquet(scalars, out / "event_telemetry")
        if args.vocab_strategy != "off":
            write_csv_and_parquet(words, out / "vocab_topk")
        return scalars, words

    temp_paths = trajectory_memmap_paths(out, spaces)
    if (
        not final_embeddings.is_file()
        and not any(path.is_file() for path in temp_paths.values())
        and any((out / "state" / "scalars").glob("g*.parquet"))
    ):
        # Scalar/word parts cannot reconstruct embeddings. If someone removed both
        # the final safetensors and temporary memmaps, rerun forward groups rather
        # than silently exporting zero-filled arrays.
        print("[forward] metadata parts exist but embedding arrays are gone; rerunning forward groups")
        shutil.rmtree(out / "state" / "scalars", ignore_errors=True)
        shutil.rmtree(out / "state" / "words", ignore_errors=True)
        (out / "state" / "scalars").mkdir(parents=True, exist_ok=True)
        (out / "state" / "words").mkdir(parents=True, exist_ok=True)

    trajectory_mm = ensure_trajectory_memmaps(out, len(event_meta), dim, spaces)
    loader = DataLoader(
        SourceImageDataset(source_manifest, preprocess),
        batch_size=int(args.image_batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=(int(args.workers) > 0),
        collate_fn=source_collate,
    )

    for group in groups:
        scalar_part = out / "state" / "scalars" / f"{group.group_id}.parquet"
        word_part = out / "state" / "words" / f"{group.group_id}.parquet"
        if scalar_part.is_file() and (args.vocab_strategy == "off" or word_part.is_file()):
            print(f"[forward] resume {group.group_index+1}/{len(groups)} {group.group_id}")
            continue

        print(
            f"[forward] {group.group_index+1}/{len(groups)} {group.experiment_name} "
            f"alpha=({group.alpha1:g},{group.alpha2:g}) specs={group.hook_specs()}"
        )
        scalar_rows: List[dict[str, Any]] = []
        word_rows: List[dict[str, Any]] = []

        for images, source_indices, stim_ids in loader:
            images = images.to(args.device, non_blocking=str(args.device).startswith("cuda"))
            source_idx_np = source_indices.numpy().astype(np.int64, copy=False)
            event_rows = (group.event_start + source_idx_np).astype(np.int64)
            with MultiAlphaConv1OutputTransform(model, group.hook_specs(), stim_ids):
                with amp_context(args.device, not args.no_amp):
                    cache = make_image_cache(model, images)

            trajectory_mm["backbone"][event_rows] = cache.backbone.detach().half().cpu().numpy()
            if "content" in trajectory_mm:
                trajectory_mm["content"][event_rows] = cache.content.detach().half().cpu().numpy()
            scalar_rows.extend(scalar_rows_for_cache(cache, event_rows.tolist(), event_meta))

            if vocab is not None:
                word_rows.extend(
                    vocab_records_for_cache(
                        model,
                        cache,
                        vocab,
                        event_rows=event_rows.tolist(),
                        event_meta=event_meta,
                        strategy=args.vocab_strategy,
                        topk=args.vocab_topk,
                        pool_per_proxy=args.vocab_pool_per_proxy,
                        exact_chunk_size=args.vocab_exact_chunk_size,
                    )
                )

            del cache, images

        for array in trajectory_mm.values():
            array.flush()
        atomic_write_parquet(pd.DataFrame(scalar_rows), scalar_part)
        if vocab is not None:
            atomic_write_parquet(pd.DataFrame(word_rows), word_part)

    metadata = {
        "format_version": str(RUN_FORMAT_VERSION),
        "rows": str(len(event_meta)),
        "dim": str(dim),
        "dtype": "float16",
        "normalized": "l2",
        "event_manifest": "events.parquet",
        "spaces": ",".join(spaces),
    }
    export_trajectory_safetensors(out, trajectory_mm, metadata)
    close_trajectory_memmaps(trajectory_mm)
    trajectory_mm.clear()
    del trajectory_mm
    if not args.keep_temp:
        for path in trajectory_memmap_paths(out, spaces).values():
            path.unlink(missing_ok=True)

    scalars = merge_group_parts(out, "scalars", len(event_meta))
    words = merge_group_parts(out, "words", len(event_meta)) if vocab is not None else pd.DataFrame()
    write_csv_and_parquet(scalars, out / "event_telemetry")
    if vocab is not None:
        write_csv_and_parquet(words, out / "vocab_topk")
    return scalars, words


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------


def load_trajectory_embeddings(out: Path) -> Dict[str, np.ndarray]:
    tensors = load_safetensors(str(out / "trajectory_embeddings.safetensors"), device="cpu")
    return {str(name): tensor.float().numpy() for name, tensor in tensors.items()}


def single_geometry(event_meta: pd.DataFrame, embeddings: Mapping[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict[str, Any]] = []
    summaries: List[dict[str, Any]] = []
    singles = event_meta[event_meta.kind == "single"]
    if singles.empty:
        return pd.DataFrame(), pd.DataFrame()

    for space, emb in embeddings.items():
        for (experiment_name, stim_id), group in singles.groupby(["experiment_name", "stim_id"], sort=False):
            group = group.sort_values("alpha1")
            base_rows = group[np.isclose(group.alpha1.astype(float), 0.0)]
            if len(base_rows) != 1:
                raise RuntimeError(f"Expected one alpha=0 baseline for {experiment_name}/{stim_id}")
            base_idx = int(base_rows.iloc[0].event_row)
            z0 = normalize_np(emb[base_idx])
            trajectory = []
            for row in group.itertuples(index=False):
                z = normalize_np(emb[int(row.event_row)])
                trajectory.append(z)
                rows.append(
                    {
                        "event_row": int(row.event_row),
                        "space": space,
                        "experiment_name": experiment_name,
                        "stim_id": stim_id,
                        "channel": int(row.channel1),
                        "condition": str(row.condition1),
                        "alpha": float(row.alpha1),
                        "self_cos": float(np.dot(z0, z)),
                        "self_dist": float(1.0 - np.dot(z0, z)),
                        "delta_norm": float(np.linalg.norm(z - z0)),
                    }
                )
            Z = np.stack(trajectory)
            path_length = float(np.linalg.norm(np.diff(Z, axis=0), axis=1).sum()) if len(Z) > 1 else 0.0
            chord = float(np.linalg.norm(Z[-1] - Z[0])) if len(Z) > 1 else 0.0
            summaries.append(
                {
                    "space": space,
                    "experiment_name": experiment_name,
                    "stim_id": stim_id,
                    "channel": int(group.iloc[0].channel1),
                    "condition": str(group.iloc[0].condition1),
                    "path_length": path_length,
                    "chord_length": chord,
                    "path_over_chord": path_length / max(chord, EPS),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def pair_geometry(event_meta: pd.DataFrame, embeddings: Mapping[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict[str, Any]] = []
    focus_rows: List[dict[str, Any]] = []
    pairs = event_meta[event_meta.kind == "pair"]
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()

    for space, emb in embeddings.items():
        for (experiment_name, stim_id), group in pairs.groupby(["experiment_name", "stim_id"], sort=False):
            lookup = {
                (float(row.alpha1), float(row.alpha2)): int(row.event_row)
                for row in group.itertuples(index=False)
            }
            if (0.0, 0.0) not in lookup:
                raise RuntimeError(f"Pair surface missing (0,0): {experiment_name}/{stim_id}")
            z00 = normalize_np(emb[lookup[(0.0, 0.0)]])
            local: List[dict[str, Any]] = []
            for row in group.itertuples(index=False):
                a1, a2 = float(row.alpha1), float(row.alpha2)
                if (a1, 0.0) not in lookup or (0.0, a2) not in lookup:
                    raise RuntimeError(f"Missing matched axis controls at ({a1},{a2})")
                z = normalize_np(emb[int(row.event_row)])
                za = normalize_np(emb[lookup[(a1, 0.0)]])
                zb = normalize_np(emb[lookup[(0.0, a2)]])
                d = z - z00
                d1 = za - z00
                d2 = zb - z00
                linear = d1 + d2
                residual = d - linear
                nd = float(np.linalg.norm(d))
                n1 = float(np.linalg.norm(d1))
                n2 = float(np.linalg.norm(d2))
                nl = float(np.linalg.norm(linear))
                nr = float(np.linalg.norm(residual))
                result = {
                    "event_row": int(row.event_row),
                    "space": space,
                    "experiment_name": experiment_name,
                    "stim_id": stim_id,
                    "channel1": int(row.channel1),
                    "condition1": str(row.condition1),
                    "channel2": int(row.channel2),
                    "condition2": str(row.condition2),
                    "alpha1": a1,
                    "alpha2": a2,
                    "self_cos": float(np.dot(z00, z)),
                    "self_dist": float(1.0 - np.dot(z00, z)),
                    "delta1_norm": n1,
                    "delta2_norm": n2,
                    "single_delta_cos": safe_cos(d1, d2),
                    "sum_single_norms": n1 + n2,
                    "actual_delta_norm": nd,
                    "linear_delta_norm": nl,
                    "actual_vs_linear_cos": safe_cos(d, linear),
                    "nonlinear_residual_norm": nr,
                    "nonlinear_residual_over_actual": nr / max(nd, EPS),
                    "actual_over_linear_norm": nd / max(nl, EPS),
                    "actual_over_sum_single_norms": nd / max(n1 + n2, EPS),
                }
                rows.append(result)
                local.append(result)

            local_df = pd.DataFrame(local)
            interior = local_df[(~np.isclose(local_df.alpha1.astype(float), 0.0)) & (~np.isclose(local_df.alpha2.astype(float), 0.0))].copy()
            if len(interior):
                fn = interior.sort_values(
                    ["nonlinear_residual_norm", "nonlinear_residual_over_actual"], ascending=[False, False]
                ).iloc[0]
                fr = interior.sort_values(
                    ["nonlinear_residual_over_actual", "nonlinear_residual_norm"], ascending=[False, False]
                ).iloc[0]
                focus_rows.append(
                    {
                        "space": space,
                        "experiment_name": experiment_name,
                        "stim_id": stim_id,
                        "focus_norm_event_row": int(fn.event_row),
                        "focus_norm_alpha1": float(fn.alpha1),
                        "focus_norm_alpha2": float(fn.alpha2),
                        "focus_norm_residual": float(fn.nonlinear_residual_norm),
                        "focus_norm_residual_over_actual": float(fn.nonlinear_residual_over_actual),
                        "focus_ratio_event_row": int(fr.event_row),
                        "focus_ratio_alpha1": float(fr.alpha1),
                        "focus_ratio_alpha2": float(fr.alpha2),
                        "focus_ratio_residual": float(fr.nonlinear_residual_norm),
                        "focus_ratio_residual_over_actual": float(fr.nonlinear_residual_over_actual),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(focus_rows)


# -----------------------------------------------------------------------------
# Exact GPIC retrieval for all trajectory events
# -----------------------------------------------------------------------------


@torch.inference_mode()
def exact_cosine_topk_query_batched(
    queries: torch.Tensor,
    bank: torch.Tensor,
    *,
    topk: int,
    bank_chunk_size: int,
    query_chunk_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    all_values: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []
    for start in tqdm(
        range(0, len(queries), int(query_chunk_size)),
        desc="GPIC query chunks",
        unit="qchunk",
    ):
        end = min(start + int(query_chunk_size), len(queries))
        values, indices = exact_cosine_topk(
            queries[start:end],
            bank,
            topk=topk,
            chunk_size=bank_chunk_size,
            device=device,
        )
        all_values.append(values)
        all_indices.append(indices)
    return np.concatenate(all_values, axis=0), np.concatenate(all_indices, axis=0)


def gpic_retrieval_records(
    event_meta: pd.DataFrame,
    values: np.ndarray,
    indices: np.ndarray,
    bank_manifest: pd.DataFrame,
    *,
    space: str,
) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    metadata_columns = [
        "embedding_row", "gpic_key", "sample_id", "shard", "shard_order", "member_name",
        "tar_offset", "encoded_size", "local_relpath", "caption_type", "caption", "license",
        "license_url", "attribution",
    ]
    for event_row in range(len(event_meta)):
        meta = event_meta.iloc[event_row]
        for rank in range(values.shape[1]):
            bank_row = int(indices[event_row, rank])
            ref = bank_manifest.iloc[bank_row]
            record = {
                "event_row": int(event_row),
                "space": space,
                "experiment_name": str(meta.experiment_name),
                "kind": str(meta.kind),
                "stim_id": str(meta.stim_id),
                "channel1": int(meta.channel1),
                "condition1": str(meta.condition1),
                "alpha1": float(meta.alpha1),
                "channel2": int(meta.channel2) if pd.notna(meta.channel2) else pd.NA,
                "condition2": str(meta.condition2),
                "alpha2": float(meta.alpha2) if pd.notna(meta.alpha2) else np.nan,
                "rank": rank + 1,
                "cosine": float(values[event_row, rank]),
                "bank_row": bank_row,
            }
            for column in metadata_columns:
                if column in bank_manifest.columns:
                    record[column] = ref[column]
            records.append(record)
    return records


@torch.inference_mode()
def run_gpic_retrieval(
    args: argparse.Namespace,
    *,
    banks: Sequence[EmbeddingBank],
    event_meta: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    out: Path,
) -> pd.DataFrame:
    existing = out / "gpic_topk.parquet"
    if existing.is_file() and not args.force_retrieval:
        print("[GPIC] cached retrieval table found; skipping")
        return pd.read_parquet(existing)

    records: List[dict[str, Any]] = []
    for space in embeddings:
        compatible = [bank for bank in banks if space in bank.spaces]
        if not compatible:
            print(f"[bank] no compatible {space} banks; skipping that retrieval space")
            continue
        print(
            f"[bank] exact {space.upper()} retrieval across "
            + ", ".join(f"{b.name}({len(b.manifest):,})" for b in compatible)
        )
        all_queries = torch.from_numpy(np.asarray(embeddings[space], dtype=np.float32))
        for start_row in range(0, len(all_queries), int(args.nn_query_chunk_size)):
            stop_row = min(start_row + int(args.nn_query_chunk_size), len(all_queries))
            chunk = all_queries[start_row:stop_row]
            merged = search_collection(
                chunk,
                compatible,
                space=space,
                topk=args.gpic_topk,
                chunk_size=args.nn_chunk_size,
                device=args.device,
            )
            for local_qi, neighbors in enumerate(merged):
                event_row = start_row + local_qi
                meta = event_meta.iloc[event_row]
                for neighbor in neighbors:
                    record = {
                        "event_row": int(event_row),
                        "space": str(space),
                        "experiment_name": str(meta.experiment_name),
                        "kind": str(meta.kind),
                        "stim_id": str(meta.stim_id),
                        "channel1": int(meta.channel1),
                        "condition1": str(meta.condition1),
                        "alpha1": float(meta.alpha1),
                        "channel2": int(meta.channel2) if pd.notna(meta.channel2) else pd.NA,
                        "condition2": str(meta.condition2),
                        "alpha2": float(meta.alpha2) if pd.notna(meta.alpha2) else np.nan,
                    }
                    record.update(public_record(neighbor))
                    records.append(record)
        del all_queries
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(records)
    write_csv_and_parquet(frame, out / "gpic_topk")
    return frame


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _top_words_map(words: pd.DataFrame) -> Dict[Tuple[int, str], List[Tuple[str, float]]]:
    mapping: Dict[Tuple[int, str], List[Tuple[str, float]]] = {}
    if words.empty:
        return mapping
    for (event_row, lane), group in words.groupby(["event_row", "lane"], sort=False):
        ordered = group.sort_values("rank")
        mapping[(int(event_row), str(lane))] = [
            (str(row.vocab_entry), float(row.mode_score)) for row in ordered.itertuples(index=False)
        ]
    return mapping


def _telemetry_map(scalars: pd.DataFrame) -> Dict[int, Any]:
    if scalars.empty:
        return {}
    return {int(row.event_row): row for row in scalars.itertuples(index=False)}


def _neighbor_group_map(gpic: pd.DataFrame) -> Dict[Tuple[int, str], pd.DataFrame]:
    mapping: Dict[Tuple[int, str], pd.DataFrame] = {}
    for (event_row, space), group in gpic.groupby(["event_row", "space"], sort=False):
        mapping[(int(event_row), str(space))] = group.sort_values("rank").reset_index(drop=True)
    return mapping


def neighbor_montage(
    group: pd.DataFrame,
    *,
    plot_topk: int,
    gpic_root: Optional[Path],
    banks_by_name: Mapping[str, EmbeddingBank],
    allow_remote: bool,
    cell_size: int = 250,
) -> Image.Image:
    count = min(int(plot_topk), len(group))
    cols = min(2, count) if count > 1 else 1
    rows = int(math.ceil(count / cols)) if count else 1
    canvas = Image.new("RGB", (cols * cell_size, rows * (cell_size + 34)), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        row = group.iloc[index]
        try:
            bank_name = str(row.get("bank_name", "gpic"))
            bank = banks_by_name.get(bank_name)
            if bank is None:
                raise KeyError(f"Unknown retrieval bank {bank_name!r}")
            record = row.to_dict()
            record["_bank"] = bank
            image = bank_neighbor_image(
                record,
                gpic_root=gpic_root,
                allow_remote=allow_remote,
            )
            image = thumb(image, (cell_size, cell_size))
        except Exception:
            image = Image.new("RGB", (cell_size, cell_size), (235, 235, 235))
            ImageDraw.Draw(image).text((12, cell_size // 2), "image unavailable", fill=(30, 30, 30))
        col = index % cols
        rr = index // cols
        x, y = col * cell_size, rr * (cell_size + 34)
        canvas.paste(image, (x, y))
        draw.text((x + 5, y + cell_size + 4), f"#{index+1} cos={float(row.cosine):.3f}", fill=(0, 0, 0))
    return canvas


def _heat(ax, matrix: np.ndarray, title: str, xlabels: Sequence[float], ylabels: Sequence[float], vmin=None, vmax=None, cmap="viridis"):
    import matplotlib.pyplot as plt
    im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)), [f"{x:g}" for x in xlabels])
    ax.set_yticks(range(len(ylabels)), [f"{y:g}" for y in ylabels])
    ax.set_xlabel("alpha2")
    ax.set_ylabel("alpha1")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _mark_cell(ax, alpha1: float, alpha2: float, alphas1: Sequence[float], alphas2: Sequence[float]) -> None:
    from matplotlib.patches import Rectangle
    i = int(np.argmin(np.abs(np.asarray(alphas1, float) - float(alpha1))))
    j = int(np.argmin(np.abs(np.asarray(alphas2, float) - float(alpha2))))
    ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor="yellow", linewidth=2.2, zorder=20))


def plot_single_sheets(
    args: argparse.Namespace,
    *,
    event_meta: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    single_geom: pd.DataFrame,
    words: pd.DataFrame,
    scalars: pd.DataFrame,
    gpic: pd.DataFrame,
    banks: Sequence[EmbeddingBank],
    out: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    word_map = _top_words_map(words)
    tele_map = _telemetry_map(scalars)
    neighbor_map = _neighbor_group_map(gpic)
    banks_by_name = {bank.name: bank for bank in banks}
    singles = event_meta[event_meta.kind == "single"]
    for space in embeddings:
        for (experiment_name, stim_id), group in singles.groupby(["experiment_name", "stim_id"], sort=False):
            group = group.sort_values("alpha1")
            alphas = group.alpha1.astype(float).tolist()
            ncols = len(group) + 1
            fig = plt.figure(figsize=(3.0 * ncols, 7.2))
            gs = fig.add_gridspec(2, ncols, height_ratios=[1.1, 1.0])
            source_path = Path(group.iloc[0].path)
            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(thumb(load_rgb(source_path), (336, 336)))
            ax.set_title(f"SOURCE\n{stim_id}")
            ax.axis("off")

            for col, row in enumerate(group.itertuples(index=False), start=1):
                ax = fig.add_subplot(gs[0, col])
                neighbors = neighbor_map.get((int(row.event_row), space))
                if neighbors is not None:
                    montage = neighbor_montage(
                        neighbors,
                        plot_topk=args.plot_topk,
                        gpic_root=args.gpic_root,
                        banks_by_name=banks_by_name,
                        allow_remote=not args.no_remote_image_fallback,
                        cell_size=220,
                    )
                    ax.imshow(montage)
                    top1 = neighbors.iloc[0]
                    title = f"α={float(row.alpha1):g}\nNN={float(top1.cosine):.3f}"
                else:
                    title = f"α={float(row.alpha1):g}"
                ax.set_title(title, fontsize=9)
                ax.axis("off")

            idx = group.event_row.to_numpy(np.int64)
            Z = normalize_np(np.asarray(embeddings[space][idx], dtype=np.float32))
            P = pca2(Z)
            axp = fig.add_subplot(gs[1, : max(2, ncols - 2)])
            axp.plot(P[:, 0], P[:, 1], marker="o")
            for i, alpha in enumerate(alphas):
                axp.text(P[i, 0], P[i, 1], f" {alpha:g}", fontsize=8)
            axp.set_title(f"local PCA2 — {space.upper()} trajectory")
            axp.grid(alpha=0.2)

            axt = fig.add_subplot(gs[1, max(2, ncols - 2) :])
            axt.axis("off")
            lines = ["ModeMUX vocabulary lanes:" if not words.empty else "Vocabulary lanes: N/A (vanilla CLIP)"]
            for row in group.itertuples(index=False):
                pieces = []
                for lane, tag in (("content", "C"), ("read", "R"), ("any", "A")):
                    vals = word_map.get((int(row.event_row), lane), [])[:3]
                    pieces.append(f"{tag}:" + "/".join(word for word, _ in vals) if vals else f"{tag}:—")
                tele = tele_map.get(int(row.event_row))
                gate = f" src={float(tele.source_gate):.2f}" if tele is not None else ""
                lines.append(f"α={float(row.alpha1):g}{gate}\n  " + " | ".join(pieces))
            axt.text(0.0, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=7.1)

            fig.suptitle(
                f"Conv1 semantic flight — {stim_id} — {experiment_name} — {space.upper()}",
                fontsize=13,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            path = out / "plots" / "singles" / space / f"{safe_name(stim_id)}__{safe_name(experiment_name)}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=165, bbox_inches="tight")
            plt.close(fig)


def plot_pair_sheets(
    args: argparse.Namespace,
    *,
    event_meta: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    pair_geom: pd.DataFrame,
    pair_focus: pd.DataFrame,
    words: pd.DataFrame,
    scalars: pd.DataFrame,
    gpic: pd.DataFrame,
    banks: Sequence[EmbeddingBank],
    out: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    word_map = _top_words_map(words)
    tele_map = _telemetry_map(scalars)
    neighbor_map = _neighbor_group_map(gpic)
    banks_by_name = {bank.name: bank for bank in banks}
    pairs = event_meta[event_meta.kind == "pair"]

    for space in embeddings:
        focus_space = pair_focus[pair_focus.space == space] if not pair_focus.empty else pd.DataFrame()
        focus_lookup = {
            (str(row.experiment_name), str(row.stim_id)): row for row in focus_space.itertuples(index=False)
        }
        for (experiment_name, stim_id), group in pairs.groupby(["experiment_name", "stim_id"], sort=False):
            geom = pair_geom[
                (pair_geom.space == space)
                & (pair_geom.experiment_name == experiment_name)
                & (pair_geom.stim_id == stim_id)
            ].copy()
            if geom.empty:
                continue
            alphas1 = sorted(geom.alpha1.astype(float).unique())
            alphas2 = sorted(geom.alpha2.astype(float).unique())
            lookup = {(float(r.alpha1), float(r.alpha2)): r for r in geom.itertuples(index=False)}
            event_lookup = {
                (float(r.alpha1), float(r.alpha2)): int(r.event_row) for r in group.itertuples(index=False)
            }
            n1, n2 = len(alphas1), len(alphas2)
            self_m = np.full((n1, n2), np.nan)
            linear_m = np.full((n1, n2), np.nan)
            residual_m = np.full((n1, n2), np.nan)
            for i, a1 in enumerate(alphas1):
                for j, a2 in enumerate(alphas2):
                    row = lookup[(a1, a2)]
                    self_m[i, j] = row.self_cos
                    linear_m[i, j] = row.actual_vs_linear_cos
                    residual_m[i, j] = row.nonlinear_residual_over_actual

            focus = focus_lookup.get((str(experiment_name), str(stim_id)))
            if focus is not None:
                focus_event = int(focus.focus_norm_event_row)
                fa1, fa2 = float(focus.focus_norm_alpha1), float(focus.focus_norm_alpha2)
            else:
                fa1, fa2 = max(alphas1), max(alphas2)
                focus_event = event_lookup[(fa1, fa2)]
            base_event = event_lookup[(0.0, 0.0)]
            corner_event = event_lookup[(max(alphas1), max(alphas2))]

            fig = plt.figure(figsize=(18.2, 10.0))
            gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.06], wspace=0.18, hspace=0.30)
            ax1 = fig.add_subplot(gs[0, 0])
            _heat(ax1, self_m, "self cosine", alphas2, alphas1, vmin=np.nanmin(self_m), vmax=1.0)
            _mark_cell(ax1, fa1, fa2, alphas1, alphas2)
            ax2 = fig.add_subplot(gs[0, 1])
            _heat(ax2, linear_m, "actual vs linear displacement cosine", alphas2, alphas1, vmin=-1, vmax=1, cmap="coolwarm")
            _mark_cell(ax2, fa1, fa2, alphas1, alphas2)
            ax3 = fig.add_subplot(gs[0, 2])
            vmax = max(1.0, float(np.nanpercentile(residual_m[np.isfinite(residual_m)], 95))) if np.isfinite(residual_m).any() else 1.0
            _heat(ax3, residual_m, "nonlinear residual / actual", alphas2, alphas1, vmin=0, vmax=vmax, cmap="magma")
            _mark_cell(ax3, fa1, fa2, alphas1, alphas2)

            source_path = Path(group.iloc[0].path)
            axsrc = fig.add_subplot(gs[0, 3])
            axsrc.imshow(thumb(load_rgb(source_path), (336, 336)))
            axsrc.set_title(f"SOURCE\n{stim_id}")
            axsrc.axis("off")

            axw = fig.add_subplot(gs[1, 0])
            axw.set_xlim(-0.5, n2 - 0.5)
            axw.set_ylim(-0.5, n1 - 0.5)
            axw.set_xticks(range(n2), [f"{x:g}" for x in alphas2])
            axw.set_yticks(range(n1), [f"{x:g}" for x in alphas1])
            axw.set_xlabel("alpha2")
            axw.set_ylabel("alpha1")
            axw.set_title("ModeMUX top words by cell (C/R/A)" if not words.empty else "Vocabulary lanes: N/A (vanilla CLIP)")
            axw.grid(alpha=0.2)
            for i, a1 in enumerate(alphas1):
                for j, a2 in enumerate(alphas2):
                    er = event_lookup[(a1, a2)]
                    labels = []
                    for lane, tag in (("content", "C"), ("read", "R"), ("any", "A")):
                        vals = word_map.get((er, lane), [])[:1]
                        labels.append(f"{tag}:{vals[0][0] if vals else '—'}")
                    axw.text(j, i, "\n".join(labels), ha="center", va="center", fontsize=5.4, rotation=15)
            _mark_cell(axw, fa1, fa2, alphas1, alphas2)

            ordered = group.sort_values(["alpha1", "alpha2"])
            idx = ordered.event_row.to_numpy(np.int64)
            Z = normalize_np(np.asarray(embeddings[space][idx], dtype=np.float32))
            P = pca2(Z)
            coords = {
                (float(row.alpha1), float(row.alpha2)): P[k]
                for k, row in enumerate(ordered.itertuples(index=False))
            }
            axp = fig.add_subplot(gs[1, 1])
            for a1 in alphas1:
                pts = np.stack([coords[(a1, a2)] for a2 in alphas2])
                axp.plot(pts[:, 0], pts[:, 1], "-o", alpha=0.65, lw=1, ms=3)
            for a2 in alphas2:
                pts = np.stack([coords[(a1, a2)] for a1 in alphas1])
                axp.plot(pts[:, 0], pts[:, 1], "-o", alpha=0.35, lw=1, ms=2)
            fp = coords[(fa1, fa2)]
            axp.scatter([fp[0]], [fp[1]], marker="*", s=160, edgecolors="black", linewidths=0.7, zorder=30)
            axp.text(fp[0], fp[1], f" FOCUS {fa1:g},{fa2:g}", fontsize=8, weight="bold")
            for point in ((0.0, 0.0), (max(alphas1), 0.0), (0.0, max(alphas2)), (max(alphas1), max(alphas2))):
                if point in coords:
                    p = coords[point]
                    axp.text(p[0], p[1], f" {point[0]:g},{point[1]:g}", fontsize=7)
            axp.set_title(f"local PCA2 — {space.upper()} pair surface")
            axp.grid(alpha=0.2)

            for slot, event_row, title in (
                (gs[1, 2], base_event, "BASELINE nearest GPIC"),
                (gs[1, 3], focus_event, "FOCUS nearest GPIC"),
            ):
                ax = fig.add_subplot(slot)
                neighbors = neighbor_map.get((event_row, space))
                if neighbors is not None:
                    montage = neighbor_montage(
                        neighbors,
                        plot_topk=args.plot_topk,
                        gpic_root=args.gpic_root,
                        banks_by_name=banks_by_name,
                        allow_remote=not args.no_remote_image_fallback,
                        cell_size=220,
                    )
                    ax.imshow(montage)
                    ax.set_title(f"{title}\ncos={float(neighbors.iloc[0].cosine):.3f}", fontsize=9)
                else:
                    ax.text(0.5, 0.5, "neighbors unavailable", ha="center", va="center")
                ax.axis("off")

            corner = lookup[(max(alphas1), max(alphas2))]
            frow = lookup[(fa1, fa2)]
            telemetry = tele_map.get(focus_event)
            gate_text = (
                f" source_gate={float(telemetry.source_gate):.3f} null_read={float(telemetry.null_read_score):.3f}"
                if telemetry is not None
                else ""
            )
            footer = []
            for label, er in (("base", base_event), ("focus", focus_event), ("corner", corner_event)):
                lane_text = []
                for lane, tag in (("content", "C"), ("read", "R"), ("any", "A")):
                    vals = word_map.get((er, lane), [])[:3]
                    lane_text.append(f"{tag}:" + "/".join(word for word, _ in vals) if vals else f"{tag}:—")
                footer.append(f"{label}: " + " | ".join(lane_text))
            fig.suptitle(
                f"ModeMUX two-filter semantic phase sweep — {space.upper()} — {stim_id} — {experiment_name}\n"
                f"corner self={corner.self_cos:.4f} actual↔linear={corner.actual_vs_linear_cos:.4f} residual/actual={corner.nonlinear_residual_over_actual:.4f}\n"
                f"focus α=({fa1:g},{fa2:g}) self={frow.self_cos:.4f} residual={frow.nonlinear_residual_norm:.4f} residual/actual={frow.nonlinear_residual_over_actual:.4f}{gate_text}",
                fontsize=12.5,
                y=0.985,
            )
            fig.text(
                0.995,
                0.012,
                "\n".join(footer),
                ha="right",
                va="bottom",
                fontsize=7.2,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.86, edgecolor="0.75"),
            )
            fig.subplots_adjust(left=0.045, right=0.985, bottom=0.095, top=0.885, wspace=0.30, hspace=0.34)
            path = out / "plots" / "pairs" / space / f"{safe_name(stim_id)}__{safe_name(experiment_name)}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def validate_channels(model, specs: Sequence[ExperimentSpec]) -> None:
    channels = int(model.visual.conv1.out_channels)
    used = sorted(
        {
            intervention.channel
            for spec in specs
            for intervention in ([spec.first] + ([spec.second] if spec.second is not None else []))
        }
    )
    bad = [channel for channel in used if not (0 <= channel < channels)]
    if bad:
        raise ValueError(f"Configured Conv1 channels outside [0,{channels}): {bad}")


def run_signature(
    *,
    args: argparse.Namespace,
    config_raw: dict[str, Any],
    source_manifest: pd.DataFrame,
    banks: Sequence[EmbeddingBank],
) -> dict[str, Any]:
    return {
        "format_version": RUN_FORMAT_VERSION,
        "model": args.model,
        "model_revision": args.model_revision or "",
        "config": config_raw,
        "source_images": [
            {
                "stim_id": str(row.stim_id),
                "path": str(row.path),
                "size": Path(row.path).stat().st_size,
            }
            for row in source_manifest.itertuples(index=False)
        ],
        "embedding_banks": [
            {
                "name": bank.name,
                "origin": bank.origin,
                "model_source": bank.model_source,
                "rows": len(bank.manifest),
                "spaces": sorted(bank.spaces),
                "selection_hash": bank.config.get("selection_hash"),
                "canonical_state_sha256": _bank_canonical_state_sha256(bank),
            }
            for bank in banks
        ],
        "vocab_strategy": args.vocab_strategy,
        "vocab_file": str(args.vocab_file),
        "vocab_sha256": (sha256_file(args.vocab_file) if args.vocab_strategy != "off" and args.vocab_file.is_file() else ""),
        "vocab_pool_per_proxy": int(args.vocab_pool_per_proxy),
        "vocab_topk": int(args.vocab_topk),
        "gpic_topk": int(args.gpic_topk),
    }


def snapshot_commit_from_path(value: Any) -> Optional[str]:
    text = re.sub(r"/+", "/", str(value or "").replace("\\", "/"))
    match = re.search(r"/snapshots/([0-9a-fA-F]{7,64})(?:/|$)", text)
    return match.group(1).lower() if match else None


def _bank_canonical_state_sha256(bank: EmbeddingBank) -> str:
    ident = bank.config.get("model_identity") or {}
    if isinstance(ident, Mapping) and ident.get("canonical_state_sha256"):
        return str(ident.get("canonical_state_sha256"))
    value = bank.config.get("canonical_state_sha256")
    return str(value or "")


def verify_model_matches_reference_bank(
    *,
    args: argparse.Namespace,
    auto_meta: Any,
    bank: EmbeddingBank,
) -> None:
    if bank.model_source and str(args.model) != bank.model_source:
        message = (
            f"Loaded model source does not match embedding bank: "
            f"runtime={args.model!r} bank={bank.model_source!r}."
        )
        if args.allow_model_bank_mismatch:
            print("[WARNING] " + message)
        else:
            raise RuntimeError(message + " Use --allow_model_bank_mismatch only intentionally.")

    bank_hash = _bank_canonical_state_sha256(bank)
    runtime_hash = str(getattr(auto_meta, "canonical_state_sha256", "") or "")
    if bank_hash and runtime_hash and bank_hash != runtime_hash:
        message = (
            "Loaded model weights do not match the embedding bank fingerprint: "
            f"runtime={runtime_hash} bank={bank_hash}."
        )
        if args.allow_model_bank_mismatch:
            print("[WARNING] " + message)
        else:
            raise RuntimeError(message + " Use --allow_model_bank_mismatch only intentionally.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conv1 single/pair semantic-manifold experiment against model-matched GPIC embedding banks."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--allow_model_bank_mismatch", action="store_true")

    parser.add_argument("--reference_bank", default=str(DEFAULT_REFERENCE_BANK), help="Primary GPIC embedding-bank root or HF dataset repo id")
    parser.add_argument("--bank_revision", default=None)
    parser.add_argument("--custom_bank", action="append", default=[], help="Optional additional local/HF bank root; repeatable. Missing local roots are skipped.")
    parser.add_argument("--gpic_root", type=Path, default=None, help="Optional local processed GPIC root; otherwise retrieved neighbors use sparse remote TAR range reads")
    parser.add_argument("--require_gpic_access", action="store_true", help="Fail instead of cleanly skipping when gated GPIC source access is unavailable")

    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--recursive_images", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_RELATIVE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit_source_images", type=int, default=0, help="0 = all source images; used by smoke tests")
    parser.add_argument("--smoke_grid", action="store_true", help="Use only alpha=0 and the final alpha for each axis")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_temp", action="store_true")

    parser.add_argument("--vocab_file", type=Path, default=DEFAULT_VOCAB_RELATIVE)
    parser.add_argument("--vocab_cache_dir", type=Path, default=None)
    parser.add_argument("--vocab_strategy", choices=("hybrid", "exact", "off"), default="hybrid")
    parser.add_argument("--vocab_topk", type=int, default=5)
    parser.add_argument("--vocab_pool_per_proxy", type=int, default=256)
    parser.add_argument("--vocab_exact_chunk_size", type=int, default=512)
    parser.add_argument("--text_batch_size", type=int, default=512)
    parser.add_argument("--rebuild_vocab_cache", action="store_true")
    parser.add_argument("--preflight_words", type=int, default=16)
    parser.add_argument("--preflight_atol", type=float, default=2.0e-3)
    parser.add_argument("--skip_preflight", action="store_true")

    parser.add_argument("--gpic_topk", type=int, default=10, help="Saved for every trajectory event in each space")
    parser.add_argument("--plot_topk", type=int, default=1, help="GPIC neighbors shown per plot cell; e.g. 4 makes a 2x2 montage")
    parser.add_argument("--nn_chunk_size", type=int, default=65536)
    parser.add_argument("--nn_query_chunk_size", type=int, default=128)
    parser.add_argument("--force_retrieval", action="store_true")
    parser.add_argument("--skip_retrieval", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--no_remote_image_fallback", action="store_true")
    return parser


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (PROJECT_ROOT / path).expanduser().resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seed_all(args.seed)
    args.image_dir = resolve_repo_path(args.image_dir)
    args.config = resolve_repo_path(args.config)
    args.vocab_file = resolve_repo_path(args.vocab_file)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.gpic_root = args.gpic_root.expanduser().resolve() if args.gpic_root is not None else None
    if args.plot_topk < 1 or args.gpic_topk < 1 or args.plot_topk > args.gpic_topk:
        raise ValueError("Require 1 <= --plot_topk <= --gpic_topk")
    if args.vocab_strategy != "off" and args.vocab_topk < 1:
        raise ValueError("--vocab_topk must be >=1")

    config_raw, specs = load_experiment_specs(args.config)
    if args.smoke_grid:
        compact: list[ExperimentSpec] = []
        for spec in specs:
            a1 = (0.0, float(spec.alphas1[-1]))
            a2 = (0.0,) if spec.kind == "single" else (0.0, float(spec.alphas2[-1]))
            compact.append(
                ExperimentSpec(
                    kind=spec.kind, name=spec.name, first=spec.first, second=spec.second,
                    alphas1=tuple(dict.fromkeys(a1)), alphas2=tuple(dict.fromkeys(a2)),
                )
            )
        specs = compact
    source_manifest = scan_source_images(args.image_dir, args.recursive_images)
    if int(args.limit_source_images) > 0:
        source_manifest = source_manifest.iloc[: int(args.limit_source_images)].copy().reset_index(drop=True)
        source_manifest["source_index"] = np.arange(len(source_manifest), dtype=np.int64)
    event_meta, groups = build_events(source_manifest, specs)

    # Resolve only the tiny bank_config.json first.  If the user has not accepted
    # the gated GPIC source or is not logged into Hugging Face, stop here as a
    # clean N/A before downloading the million-row manifest, multi-GiB reference
    # tensors, or even the runtime model.
    try:
        primary_descriptor = resolve_bank_descriptor(
            args.reference_bank,
            model_source=args.model,
            revision=args.bank_revision,
            cache_dir=args.hf_cache_dir,
            required_spaces=None,
            optional=False,
            is_primary=True,
        )
        assert primary_descriptor is not None
    except (BankUnavailable, FileNotFoundError) as exc:
        prepare_output_dir(args.output_dir, args.overwrite)
        print(f"[bank] no public GPIC embedding bank is available for {args.model!r}: {exc}")
        atomic_write_json(
            args.output_dir / "SKIPPED_GPIC_NA.json",
            {"reason": "embedding_bank_unavailable", "model": args.model, "detail": str(exc)},
        )
        return 0

    access_ok, access_detail = gpic_access_probe(primary_descriptor)
    if not access_ok:
        prepare_output_dir(args.output_dir, args.overwrite)
        print_gpic_access_warning(access_detail)
        atomic_write_json(
            args.output_dir / "SKIPPED_GPIC_NA.json",
            {
                "reason": "gpic_access_unavailable",
                "model": args.model,
                "reference_bank": str(args.reference_bank),
                "detail": access_detail,
            },
        )
        if args.require_gpic_access:
            return 2
        return 0

    print(f"[GPIC] gated source access OK: {access_detail}")

    # Only now resolve the runtime.  Its actual architecture determines whether
    # the final reference space is BACKBONE-only (vanilla CLIP) or additionally
    # CONTENT (the released x-attention model).
    print(f"[model] loading {args.model}")
    model, preprocess, load_info, auto_meta = load_reproduction_model(
        args.model,
        device=args.device,
        model_revision=args.model_revision,
        hf_cache_dir=args.hf_cache_dir,
    )
    spaces = runtime_embedding_spaces(model)
    if not has_xattn_content_path(model) and args.vocab_strategy != "off":
        print("[model] vanilla CLIP: CONTENT/READ/ANY lanes are unavailable; setting --vocab_strategy off")
        args.vocab_strategy = "off"
    if args.vocab_strategy != "off" and not args.vocab_file.is_file():
        raise FileNotFoundError(
            f"Vocabulary not found at release path: {args.vocab_file}. "
            "Expected utils_datasets/clipese/vocab_deduped.txt"
        )

    # Materialize only the model-matched manifest and final embedding spaces.
    # ``resolve_bank_source`` validates row alignment and safetensor dimensions.
    primary = resolve_bank_source(
        args.reference_bank,
        model_source=args.model,
        revision=args.bank_revision,
        cache_dir=args.hf_cache_dir,
        required_spaces=spaces,
        optional=False,
        is_primary=True,
        download_embeddings=True,
        name="gpic",
    )
    assert primary is not None
    verify_model_matches_reference_bank(args=args, auto_meta=auto_meta, bank=primary)

    banks: list[EmbeddingBank] = [primary]
    for custom_index, source in enumerate(args.custom_bank or [], start=1):
        bank = resolve_bank_source(
            source,
            model_source=args.model,
            revision=None,
            cache_dir=args.hf_cache_dir,
            required_spaces=spaces,
            optional=True,
            is_primary=False,
            download_embeddings=True,
            name=f"custom{custom_index}:{Path(str(source)).name or 'bank'}",
        )
        if bank is not None:
            verify_model_matches_reference_bank(args=args, auto_meta=auto_meta, bank=bank)
            banks.append(bank)
    print(
        "[banks] "
        + ", ".join(
            f"{bank.name}: rows={len(bank.manifest):,} spaces={','.join(sorted(bank.spaces))}"
            for bank in banks
        )
    )
    print(f"[sources] {len(source_manifest)} images | events={len(event_meta):,} | groups={len(groups):,}")

    prepare_output_dir(args.output_dir, args.overwrite)
    signature_obj = run_signature(
        args=args,
        config_raw=config_raw,
        source_manifest=source_manifest,
        banks=banks,
    )
    signature_hash = sha256_json(signature_obj)
    signature_path = args.output_dir / "run_signature.json"
    if signature_path.is_file() and not args.overwrite:
        old = json.loads(signature_path.read_text(encoding="utf-8"))
        if old.get("sha256") != signature_hash:
            raise RuntimeError(
                "Output directory contains a different experiment signature. "
                "Use a new --output_dir or --overwrite."
            )
    else:
        atomic_write_json(signature_path, {"sha256": signature_hash, "config": signature_obj})

    atomic_write_parquet(source_manifest, args.output_dir / "source_manifest.parquet")
    source_manifest.to_csv(args.output_dir / "source_manifest.csv", index=False)
    atomic_write_parquet(event_meta, args.output_dir / "events.parquet")
    event_meta.to_csv(args.output_dir / "events.csv", index=False)

    validate_channels(model, specs)
    arch = model_architecture_summary(model)
    dim = int(getattr(model.visual, "output_dim", arch.get("output_dim", -1)))
    if dim <= 0:
        raise RuntimeError(f"Could not determine visual output dimension: {dim}")
    model_info = {
        "model_source": args.model,
        "requested_revision": args.model_revision,
        "load_info": _as_plain_dict(load_info),
        "auto_mechinterp": _as_plain_dict(auto_meta),
        "architecture": arch,
        "embedding_spaces": list(spaces),
        "conv1_out_channels": int(model.visual.conv1.out_channels),
        "experiment_config": str(args.config),
        "vocab_file": str(args.vocab_file),
        "embedding_banks": [
            {"name": bank.name, "origin": bank.origin, "rows": len(bank.manifest), "spaces": sorted(bank.spaces)}
            for bank in banks
        ],
    }
    atomic_write_json(args.output_dir / "model_info.json", model_info)
    print("[model] " + json.dumps({**arch, "embedding_spaces": list(spaces)}, sort_keys=True))

    vocab: Optional[VocabBank] = None
    if args.vocab_strategy != "off":
        vocab_cache_dir = (
            args.vocab_cache_dir.expanduser().resolve()
            if args.vocab_cache_dir is not None
            else args.output_dir / "_vocab_cache"
        )
        vocab = build_or_load_vocab_bank(
            model,
            vocab_path=args.vocab_file,
            cache_dir=vocab_cache_dir,
            model_source=args.model,
            model_revision=args.model_revision,
            device=args.device,
            batch_size=args.text_batch_size,
            use_amp=not args.no_amp,
            force_rebuild=args.rebuild_vocab_cache,
        )
        print(f"[vocab] {len(vocab.words):,} entries strategy={args.vocab_strategy}")
        if not args.skip_preflight:
            preflight_cached_mode_scorer(
                model,
                preprocess,
                Path(source_manifest.iloc[0].path),
                vocab,
                device=args.device,
                use_amp=not args.no_amp,
                n_words=args.preflight_words,
                atol=args.preflight_atol,
            )

    scalars, words = run_forward_stage(
        args,
        model=model,
        preprocess=preprocess,
        source_manifest=source_manifest,
        event_meta=event_meta,
        groups=groups,
        vocab=vocab,
        out=args.output_dir,
        dim=dim,
        spaces=spaces,
    )
    embeddings = load_trajectory_embeddings(args.output_dir)

    single_geom, single_summary = single_geometry(event_meta, embeddings)
    pair_geom, pair_focus = pair_geometry(event_meta, embeddings)
    if not single_geom.empty:
        write_csv_and_parquet(single_geom, args.output_dir / "single_geometry")
        write_csv_and_parquet(single_summary, args.output_dir / "single_summary")
    if not pair_geom.empty:
        write_csv_and_parquet(pair_geom, args.output_dir / "pair_geometry")
        write_csv_and_parquet(pair_focus, args.output_dir / "pair_focus")

    if args.skip_retrieval:
        gpic = pd.DataFrame()
    else:
        gpic = run_gpic_retrieval(
            args,
            banks=banks,
            event_meta=event_meta,
            embeddings=embeddings,
            out=args.output_dir,
        )

    if not args.skip_plots and not gpic.empty:
        plot_single_sheets(
            args,
            event_meta=event_meta,
            embeddings=embeddings,
            single_geom=single_geom,
            words=words,
            scalars=scalars,
            gpic=gpic,
            banks=banks,
            out=args.output_dir,
        )
        plot_pair_sheets(
            args,
            event_meta=event_meta,
            embeddings=embeddings,
            pair_geom=pair_geom,
            pair_focus=pair_focus,
            words=words,
            scalars=scalars,
            gpic=gpic,
            banks=banks,
            out=args.output_dir,
        )

    summary = {
        "run_signature": signature_hash,
        "model": args.model,
        "embedding_spaces": list(embeddings),
        "sources": len(source_manifest),
        "events": len(event_meta),
        "groups": len(groups),
        "single_experiments": sum(spec.kind == "single" for spec in specs),
        "pair_experiments": sum(spec.kind == "pair" for spec in specs),
        "primary_gpic_rows": len(primary.manifest),
        "reference_rows_total": sum(len(bank.manifest) for bank in banks),
        "embedding_banks": [bank.name for bank in banks],
        "vocab_entries": len(vocab.words) if vocab is not None else 0,
        "vocab_strategy": args.vocab_strategy,
        "gpic_topk": args.gpic_topk,
        "plot_topk": args.plot_topk,
        "output_dir": str(args.output_dir),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print("[DONE] " + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
