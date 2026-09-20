#!/usr/bin/env python3
r"""Unified mechanistic analysis for the HF x-attention CLIP model.

This script replaces the one-off post-training probes with a single, non-interventional
analysis over image_sets/demoset.  It NEVER relocates/transplants bridge taps or copies
visual states between blocks/models.

Primary model:
    alpha_0_28

Reference reader:
    full_xattn_model

Output root:
    out_bridge_analysis_for_paper

The WiSE alpha checkpoint changes only the late READ branch parameters
(read_bridge.*, read_tap_logits, glyph_bias_beta, null_abstain_weight).  Therefore the
reference model is re-run only for READ / downstream reader-routing quantities; SOURCE,
ORTHO, CONTENT/correction, backbone attention, token diagnostics, and classic B23
rollout are not duplicated under _old_xattn.

Analysis families
-----------------
1. Scores / routing
   * classic RN-backbone cosine/logit
   * corrected content cosine/logit
   * robust any-mode logit
   * raw/calibrated/relative READ logits
   * ORTHO logit, trust/route gates, automatic READ contribution, READ_NULL attention

2. SOURCE taps
   * per-tap patch glyph probability
   * learned tap mixture
   * global source/readability logits and source statistics

3. ORTHO taps
   * per-tap, per-candidate attention (mean-head overlays + raw per-head arrays)
   * learned tap mixture
   * V/out-projection token contribution norm (bias excluded and reported separately)

4. READ taps
   * patch/non-register attention
   * register-only attention
   * signed effective attention = patch + register_gate * register
   * READ_NULL attention
   * V/out-projection token contribution norm (bias excluded)
   * per-tap and learned-mixture maps

5. CONTENT/correction
   * per-tap attention and learned tap mixture
   * V/out-projection token contribution norm (bias excluded)
   * correction-vector norm, relative norm, and effect on candidate similarities

6. Backbone / RN
   * every resblock: CLS->patch attention (mean-head overlays + per-head raw rows)
   * post-RN blocks: CLS->RN, RN->CLS, RN->RN, RN->patch attention
   * per-block patch residual norms, CLS norm, RN norm, final-register attention mass

7. Native demoset pair deltas (NO state/tap intervention)
   * condition-reference score deltas for all fixed prompts
   * SOURCE/correction/backbone/RN/register spatial deltas
   * candidate-specific ORTHO/READ and available B23 rollout deltas

8. Gradient-weighted classic attention rollout, LAST BLOCK B23 ONLY
   * candidates are selected from classic cosine similarity
   * all known-present prompts are retained
   * additional candidates are retained when cosine >= weakest known-present prompt
     (configurable with --good-threshold)
   * target is classic cosine similarity, not PIECES/bridge score
   * map = mean_heads(ReLU(attention * d(score)/d(attention))) for CLS->patch at B23
   * RN scalar from the same gradient-weighted CLS row is reported separately

The script saves raw arrays and figures into separate subfolders so that figures can be
regenerated without rerunning inference.
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_analysis import (DemoSpec, _normalize_rows)


import argparse
import csv
import gc
import hashlib
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoProcessor


# =================================================================================================
# Fixed experiment definition
# =================================================================================================

PROMPT_WORDS = (
    "banana",
    "apple",
    "granny smith",
    "ipod",
    "iphone",
    "pineapple",
    "bird",
    "goldfinch",
    "bee",
    "bumblebee",
    "shampoo",
    "shower gel",
    "detergent",
    "cat",
    "dog",
    "raccoon",
    "badger",
    "word",
    "text",
    "typography",
)

DEFAULT_PROMPT_TEMPLATE = "a photo of a {prompt_word}"

DEFAULT_IMAGE_DIR = Path("image_sets/demoset")
DEFAULT_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_OLD_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
DEFAULT_OUTPUT = Path("out_bridge_analysis_for_paper")


DEMO_SPECS = (
    DemoSpec(
        "apple_ipod.png",
        ("granny smith", "apple"),
        ("ipod",),
        "granny smith/apple; rendered text: ipod",
    ),
    DemoSpec(
        "apple_none.png",
        ("granny smith", "apple"),
        (),
        "granny smith/apple; no attack text",
    ),
    DemoSpec(
        "bananorange_pineapple.png",
        ("banana",),
        ("pineapple",),
        "banana/orange composite; rendered text: pineapple; orange is intentionally outside PROMPT_WORDS",
    ),
    DemoSpec(
        "bananorange_none.png",
        ("banana",),
        (),
        "banana/orange composite; no attack text; orange is intentionally outside PROMPT_WORDS",
    ),
    DemoSpec(
        "bottle_shampoo.png",
        ("shampoo",),
        ("shampoo",),
        "shampoo bottle; rendered text agrees with content",
    ),
    DemoSpec(
        "bottle_shower.png",
        ("shower gel",),
        ("shower gel",),
        "shower-gel bottle; rendered text agrees with content",
    ),
    DemoSpec(
        "cat_cat.png",
        ("cat",),
        ("cat",),
        "cat; rendered text agrees with content",
    ),
    DemoSpec(
        "dog_cat.png",
        ("cat",),
        ("dog",),
        "cat; rendered text: dog",
    ),
    DemoSpec(
        "goldfinch_bumblebee.png",
        ("bird", "goldfinch"),
        ("bumblebee",),
        "goldfinch/bird; rendered text: bumblebee",
    ),
    DemoSpec(
        "goldfinch_none.png",
        ("bird", "goldfinch"),
        (),
        "goldfinch/bird; no attack text",
    ),
    DemoSpec(
        "thecat_raccoon.png",
        ("cat", "raccoon"),
        (),
        "cat with raccoon face visible in mirror",
    ),
    DemoSpec(
        "thecat_none.png",
        ("cat",),
        (),
        "cat with normal cat mirror reflection",
    ),
)


@dataclass(frozen=True)
class PairSpec:
    name: str
    condition: str
    reference: str
    words: tuple[str, ...]
    note: str = ""


# Native, non-interventional contrasts within the fixed demoset.  `condition - reference`
# is used for every delta.  These are descriptive contrasts, not claims of causal isolation.
PAIR_SPECS = (
    PairSpec(
        "apple_ipod_vs_none",
        "apple_ipod.png",
        "apple_none.png",
        ("apple", "granny smith", "ipod", "iphone", "word", "text", "typography"),
        "same semantic apple family; rendered iPod text vs no attack text",
    ),
    PairSpec(
        "bananorange_pineapple_vs_none",
        "bananorange_pineapple.png",
        "bananorange_none.png",
        ("banana", "pineapple", "word", "text", "typography"),
        "banana/orange composite; rendered pineapple text vs no attack text",
    ),
    PairSpec(
        "goldfinch_bumblebee_vs_none",
        "goldfinch_bumblebee.png",
        "goldfinch_none.png",
        ("bird", "goldfinch", "bee", "bumblebee", "word", "text", "typography"),
        "goldfinch; rendered bumblebee text vs no attack text",
    ),
    PairSpec(
        "dog_text_vs_cat_text",
        "dog_cat.png",
        "cat_cat.png",
        ("cat", "dog", "word", "text", "typography"),
        "cat content with rendered dog vs cat text",
    ),
    PairSpec(
        "raccoon_mirror_vs_cat_mirror",
        "thecat_raccoon.png",
        "thecat_none.png",
        ("cat", "raccoon", "badger"),
        "raccoon-face mirror content vs normal cat reflection; not a pure text contrast",
    ),
    PairSpec(
        "shampoo_vs_shower_gel",
        "bottle_shampoo.png",
        "bottle_shower.png",
        ("shampoo", "shower gel", "detergent", "word", "text", "typography"),
        "semantic/text bottle contrast; not treated as an attack-control pair",
    ),
)

EXPECTED_WISE_DIFF_EXACT = {
    "read_implant.read_tap_logits",
    "read_implant.glyph_bias_beta",
    "read_implant.null_abstain_weight",
}
EXPECTED_WISE_DIFF_PREFIX = "read_implant.read_bridge."


@dataclass
class DemoSample:
    spec: DemoSpec
    path: Path
    image: Image.Image

    @property
    def image_name(self) -> str:
        return self.spec.filename

    @property
    def stem(self) -> str:
        return Path(self.spec.filename).stem


# =================================================================================================
# Generic utilities
# =================================================================================================


def _slug(text: str) -> str:
    text = text.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_.-]+", "_", text).strip("_") or "item"


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _layer_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    candidate = getattr(value, "last_hidden_state", None)
    if torch.is_tensor(candidate):
        return candidate
    raise TypeError(f"Cannot obtain tensor from layer output {type(value)!r}")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    preferred = [
        "image",
        "word",
        "block",
        "head",
        "is_known_present",
        "is_text_word",
        "model",
    ]
    union = set().union(*(row.keys() for row in rows))
    for key in preferred:
        if key in union and key not in seen:
            keys.append(key)
            seen.add(key)
    for key in sorted(union):
        if key not in seen:
            keys.append(key)
            seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _grid_side(patch_count: int) -> int:
    side = int(round(math.sqrt(int(patch_count))))
    if side * side != int(patch_count):
        raise ValueError(f"Patch count {patch_count} is not a square grid")
    return side


def _to_grid(values: torch.Tensor) -> torch.Tensor:
    side = _grid_side(values.shape[-1])
    return values.reshape(*values.shape[:-1], side, side)


def _numpy_grid(values: torch.Tensor) -> np.ndarray:
    return _to_grid(values.detach().float().cpu()).numpy()


def _open_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return image.convert("RGB")


def load_demo_samples(image_dir: Path) -> list[DemoSample]:
    allowed = set(PROMPT_WORDS)
    samples: list[DemoSample] = []
    missing: list[str] = []
    for spec in DEMO_SPECS:
        unknown = (set(spec.present_words) | set(spec.text_words)) - allowed
        if unknown:
            raise ValueError(f"Manifest {spec.filename} references words outside PROMPT_WORDS: {sorted(unknown)}")
        path = image_dir / spec.filename
        if not path.is_file():
            missing.append(str(path))
            continue
        samples.append(DemoSample(spec=spec, path=path, image=_open_rgb(path)))
    if missing:
        raise FileNotFoundError("Missing required demoset images:\n  " + "\n  ".join(missing))
    return samples


def _batches(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, list(items[start : start + batch_size])


def _tokenize_prompts(processor: Any, template: str, device: torch.device) -> torch.Tensor:
    text = [template.format(prompt_word=word) for word in PROMPT_WORDS]
    encoded = processor(
        text=text,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device)


def _preprocess_images(processor: Any, samples: Sequence[DemoSample], device: torch.device) -> torch.Tensor:
    encoded = processor(images=[sample.image for sample in samples], return_tensors="pt")
    return encoded["pixel_values"].to(device)


def _pixel_values_to_rgb(pixel_values: torch.Tensor, processor: Any) -> list[np.ndarray]:
    image_processor = getattr(processor, "image_processor", processor)
    mean = torch.tensor(
        getattr(image_processor, "image_mean", (0.48145466, 0.4578275, 0.40821073)),
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        getattr(image_processor, "image_std", (0.26862954, 0.26130258, 0.27577711)),
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    rgb = pixel_values.detach().float().cpu() * std + mean
    rgb = rgb.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return [array for array in rgb]


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    _ensure_dir(path.parent)
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255.0)).save(path)


def _finite_range(grid: np.ndarray, signed: bool) -> tuple[float, float]:
    arr = np.asarray(grid, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (-1.0, 1.0) if signed else (0.0, 1.0)
    if signed:
        vmax = float(np.percentile(np.abs(arr), 99.0))
        vmax = max(vmax, 1e-12)
        return -vmax, vmax
    positive = arr[arr >= 0]
    vmax = float(np.percentile(positive if positive.size else arr, 99.0))
    vmax = max(vmax, 1e-12)
    return 0.0, vmax


def save_overlay(
    rgb: np.ndarray,
    grid: np.ndarray,
    path: Path,
    *,
    title: str,
    signed: bool = False,
    cmap: str | None = None,
    alpha: float = 0.52,
) -> None:
    _ensure_dir(path.parent)
    if cmap is None:
        cmap = "coolwarm" if signed else "inferno"
    vmin, vmax = _finite_range(grid, signed=signed)
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=150)
    ax.imshow(rgb, interpolation="nearest")
    ax.imshow(
        grid,
        cmap=cmap,
        alpha=alpha,
        interpolation="nearest",
        extent=(0, rgb.shape[1], rgb.shape[0], 0),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    fig.tight_layout(pad=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_plain_map(
    grid: np.ndarray,
    path: Path,
    *,
    title: str,
    signed: bool = False,
    cmap: str | None = None,
) -> None:
    _ensure_dir(path.parent)
    if cmap is None:
        cmap = "coolwarm" if signed else "inferno"
    vmin, vmax = _finite_range(grid, signed=signed)
    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=150)
    im = ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_panel(
    rgb: np.ndarray,
    panels: Sequence[tuple[str, np.ndarray, bool]],
    path: Path,
    *,
    title: str,
) -> None:
    _ensure_dir(path.parent)
    columns = 1 + len(panels)
    fig, axes = plt.subplots(1, columns, figsize=(3.25 * columns, 3.45), dpi=150)
    if columns == 1:
        axes = [axes]
    axes[0].imshow(rgb)
    axes[0].set_title("input", fontsize=9)
    axes[0].set_axis_off()
    for ax, (panel_title, grid, signed) in zip(axes[1:], panels):
        vmin, vmax = _finite_range(grid, signed=signed)
        ax.imshow(rgb, interpolation="nearest")
        ax.imshow(
            grid,
            cmap="coolwarm" if signed else "inferno",
            alpha=0.52,
            interpolation="nearest",
            extent=(0, rgb.shape[1], rgb.shape[0], 0),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(panel_title, fontsize=9)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _npz(path: Path, **arrays: Any) -> None:
    _ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def _block_list(value: Any) -> list[int]:
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in value]


def _tap_weights(logits: torch.Tensor) -> torch.Tensor:
    return logits.detach().float().softmax(dim=0)


def _candidate_flags(sample: DemoSample, word: str) -> tuple[int, int]:
    return int(word in sample.spec.present_words), int(word in sample.spec.text_words)


def _model_config_dict(model: Any) -> dict[str, Any]:
    config = model.config
    if hasattr(config, "to_dict"):
        try:
            return config.to_dict()
        except Exception:
            pass
    return {key: value for key, value in vars(config).items() if not key.startswith("_")}


def _model_meta(model: Any, model_path: str | Path, prompt_template: str) -> dict[str, Any]:
    implant = model.read_implant
    read_blocks = _block_list(implant.tap_blocks)
    ortho_blocks = _block_list(implant.ortho_tap_blocks)
    source_blocks = _block_list(implant.source_tap_blocks)
    meta = {
        "model_path": str(model_path),
        "model_class": type(model).__name__,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "prompt_words": list(PROMPT_WORDS),
        "prompt_template": prompt_template,
        "read_null_insert_block": int(model.config.read_null_insert_block),
        "read_tap_blocks": read_blocks,
        "read_tap_weights": [_as_float(x) for x in _tap_weights(implant.read_tap_logits)],
        "ortho_tap_blocks": ortho_blocks,
        "ortho_tap_weights": [_as_float(x) for x in _tap_weights(implant.ortho_tap_logits)],
        "source_tap_blocks": source_blocks,
        "source_tap_weights": [_as_float(x) for x in _tap_weights(implant.source_tap_logits)],
        "content_tap_blocks": read_blocks,
        "content_tap_weights": [_as_float(x) for x in _tap_weights(implant.content_tap_logits)],
        "read_attention_architecture": str(implant.read_attention_architecture),
        "read_bridge_heads": int(implant.read_bridge.heads),
        "register_norm_threshold": float(model.config.register_norm_threshold),
        "register_min": int(model.config.register_min),
        "register_max": int(model.config.register_max),
        "read_register_gate": _as_float(implant.read_bridge.register_gate),
        "glyph_bias_beta": _as_float(implant.glyph_bias_beta),
        "read_calibration_scale": _as_float(implant.read_calibration_scale),
        "null_abstain_weight": _as_float(implant.null_abstain_weight),
        "auto_read_scale": _as_float(implant.auto_read_scale),
        "read_out_proj_bias_norm": _as_float(implant.read_bridge.out_proj.bias.float().norm())
        if implant.read_bridge.out_proj.bias is not None
        else 0.0,
        "ortho_out_proj_bias_norm": _as_float(implant.orthographic_bridge.out_proj.bias.float().norm())
        if implant.orthographic_bridge.out_proj.bias is not None
        else 0.0,
        "content_out_proj_bias_norm": _as_float(implant.content_pool.out_proj.bias.float().norm())
        if implant.content_pool.out_proj.bias is not None
        else 0.0,
        "transformers_version": None,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "config": _model_config_dict(model),
    }
    try:
        import transformers
        meta["transformers_version"] = transformers.__version__
    except Exception:
        pass
    # `model_path` may be either a local HF directory or a remote Hub repo ID.
    # Do not apply filesystem path semantics to repo IDs: on Windows that would
    # rewrite the required owner/model slash to a backslash.
    local_model_dir = Path(model_path).expanduser()
    if local_model_dir.is_dir():
        wise_meta = local_model_dir / "WISE_FT_METADATA.json"
        if wise_meta.is_file():
            try:
                meta["wise_ft_metadata"] = json.loads(wise_meta.read_text(encoding="utf-8"))
            except Exception as exc:
                meta["wise_ft_metadata_error"] = repr(exc)
    return meta


# =================================================================================================
# Exact bridge contribution decompositions (attention + V + output projection, bias excluded)
# =================================================================================================


def _fp32_layer_norm(module: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(
        x.float(),
        module.normalized_shape,
        module.weight.float(),
        module.bias.float() if module.bias is not None else None,
        module.eps,
    )


def _fp32_linear(module: torch.nn.Linear, x: torch.Tensor, *, bias: bool = True) -> torch.Tensor:
    b = module.bias.float() if bias and module.bias is not None else None
    return F.linear(x.float(), module.weight.float(), b)


def read_token_contribution_vectors(
    bridge: Any,
    visual_tokens: torch.Tensor,
    patch_attention: torch.Tensor,
    register_attention: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (token contribution vectors, effective attention).

    token contribution vectors: [B,N,T,Dproj], output-projection bias excluded.
    effective attention:        [B,N,H,T], patch + register_gate * register.
    """
    with torch.inference_mode():
        batch, token_count, _ = visual_tokens.shape
        vision = _fp32_layer_norm(bridge.vision_ln, visual_tokens)
        value = _fp32_linear(bridge.v_proj, vision)
        value = value.view(batch, token_count, bridge.heads, bridge.head_dim).permute(0, 2, 1, 3)
        effective = patch_attention.detach().float()
        if register_attention is not None:
            effective = effective + bridge.register_gate.detach().float() * register_attention.detach().float()
        z = effective.unsqueeze(-1) * value[:, None, :, :, :]
        z = z.permute(0, 1, 3, 2, 4).contiguous().reshape(
            batch, effective.shape[1], token_count, bridge.bridge_width
        )
        vectors = _fp32_linear(bridge.out_proj, z, bias=False)
        return vectors, effective


def ortho_token_contribution_vectors(
    bridge: Any,
    visual_tokens: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    """Return [B,N,P,Dproj] ORTHO token contribution vectors; output bias excluded."""
    with torch.inference_mode():
        patches = visual_tokens[:, 1:, :]
        batch, patch_count, _ = patches.shape
        hidden = F.gelu(_fp32_linear(bridge.patch_expand, _fp32_layer_norm(bridge.patch_ln, patches)))
        hidden = F.gelu(_fp32_linear(bridge.patch_contract, hidden))
        value = _fp32_linear(bridge.v_proj, hidden)
        value = value.view(batch, patch_count, bridge.heads, bridge.head_dim).permute(0, 2, 1, 3)
        z = attention.detach().float().unsqueeze(-1) * value[:, None, :, :, :]
        z = z.permute(0, 1, 3, 2, 4).contiguous().reshape(
            batch, attention.shape[1], patch_count, bridge.bridge_width
        )
        return _fp32_linear(bridge.out_proj, z, bias=False)


def content_token_contribution_vectors(
    pool: Any,
    visual_tokens: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    """Return [B,T,Dproj] CONTENT/correction token contribution vectors; bias excluded."""
    with torch.inference_mode():
        batch, token_count, _ = visual_tokens.shape
        vision = _fp32_layer_norm(pool.vision_ln, visual_tokens)
        value = _fp32_linear(pool.v_proj, vision)
        value = value.view(batch, token_count, pool.heads, pool.head_dim).permute(0, 2, 1, 3)
        z = attention.detach().float().unsqueeze(-1) * value
        z = z.permute(0, 2, 1, 3).contiguous().reshape(batch, token_count, pool.pool_width)
        return _fp32_linear(pool.out_proj, z, bias=False)


# =================================================================================================
# Backbone attention / token trace
# =================================================================================================


def validate_hf_clip_layer(layer: Any) -> None:
    for name in ("layer_norm1", "layer_norm2", "self_attn", "mlp"):
        if not hasattr(layer, name):
            raise TypeError(f"Expected HF CLIPEncoderLayer.{name}")
    for name in ("q_proj", "k_proj", "v_proj", "out_proj", "num_heads"):
        if not hasattr(layer.self_attn, name):
            raise TypeError(f"Expected HF CLIPAttention.{name}")


@torch.inference_mode()
def manual_backbone_attention(layer: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """Exact standard HF CLIP self-attention probabilities [B,H,T,T] in FP32."""
    validate_hf_clip_layer(layer)
    attn = layer.self_attn
    x = layer.layer_norm1(hidden_states).float()
    batch, token_count, _ = x.shape
    heads = int(attn.num_heads)

    q = _fp32_linear(attn.q_proj, x)
    k = _fp32_linear(attn.k_proj, x)
    head_dim = q.shape[-1] // heads
    scale = float(getattr(attn, "scale", head_dim ** -0.5))
    q = (q * scale).view(batch, token_count, heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(batch, token_count, heads, head_dim).transpose(1, 2).contiguous()
    return torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)


@torch.inference_mode()
def collect_backbone_trace(model: Any, pixel_values: torch.Tensor) -> dict[str, Any]:
    """Run the untouched RN backbone and collect per-block attention + post-block states."""
    vision = model.vision_model
    hidden = vision.embeddings(pixel_values, interpolate_pos_encoding=False)
    hidden = vision.pre_layrnorm(hidden)
    insert_block = int(model.config.read_null_insert_block)
    attentions: dict[int, torch.Tensor] = {}
    post_states: dict[int, torch.Tensor] = {}
    pre_states: dict[int, torch.Tensor] = {}
    inserted = False
    for block, layer in enumerate(vision.encoder.layers):
        if block == insert_block:
            rn = model.read_null_token.to(device=hidden.device, dtype=hidden.dtype)
            hidden = torch.cat((hidden, rn.view(1, 1, -1).expand(hidden.shape[0], 1, -1)), dim=1)
            inserted = True
        pre_states[block] = hidden
        attentions[block] = manual_backbone_attention(layer, hidden)
        hidden = _layer_tensor(layer(hidden, None))
        post_states[block] = hidden
    if not inserted:
        raise RuntimeError("READ_NULL was never inserted during backbone trace")
    return {
        "attentions": attentions,
        "pre_states": pre_states,
        "post_states": post_states,
        "final_tokens": hidden,
    }


def _patch_slice(has_rn: bool) -> slice:
    return slice(1, -1 if has_rn else None)


def save_backbone_analysis(
    root: Path,
    model: Any,
    batch_samples: Sequence[DemoSample],
    rgb_batch: Sequence[np.ndarray],
    trace: Mapping[str, Any],
    final_register_mask: torch.Tensor,
    rows: list[dict[str, Any]],
    paper_cache: dict[str, dict[str, Any]],
) -> None:
    insert_block = int(model.config.read_null_insert_block)
    last_block = len(model.vision_model.encoder.layers) - 1
    for block, attention in trace["attentions"].items():
        has_rn = block >= insert_block
        psl = _patch_slice(has_rn)
        state = trace["post_states"][block].detach().float()
        patch_state = state[:, psl, :]
        patch_norm = patch_state.norm(dim=-1)
        patch_norm_grid = _to_grid(patch_norm)
        cls_rows = attention[:, :, 0, :].detach().float()
        cls_patch = cls_rows[:, :, psl]
        cls_grid = _to_grid(cls_patch.mean(dim=1))
        if has_rn:
            rn_rows = attention[:, :, -1, :].detach().float()
            rn_patch = rn_rows[:, :, 1:-1]
            rn_grid = _to_grid(rn_patch.mean(dim=1))
        else:
            rn_rows = None
            rn_grid = None

        for i, (sample, rgb) in enumerate(zip(batch_samples, rgb_batch)):
            stem = sample.stem
            save_overlay(
                rgb,
                cls_grid[i].cpu().numpy(),
                root / "backbone" / "cls_to_patch" / f"B{block:02d}" / f"{stem}.png",
                title=f"{stem} | B{block:02d} CLS→patch mean-head",
            )
            save_plain_map(
                patch_norm_grid[i].cpu().numpy(),
                root / "token_diagnostics" / "patch_norms" / f"B{block:02d}" / f"{stem}.png",
                title=f"{stem} | B{block:02d} patch residual norm",
                cmap="viridis",
            )
            raw_payload: dict[str, Any] = {
                "cls_attention": cls_rows[i].cpu().numpy(),
                "cls_patch_attention": cls_patch[i].cpu().numpy(),
                "patch_norms": patch_norm[i].cpu().numpy(),
            }
            if rn_rows is not None and rn_grid is not None:
                save_overlay(
                    rgb,
                    rn_grid[i].cpu().numpy(),
                    root / "backbone" / "rn_to_patch" / f"B{block:02d}" / f"{stem}.png",
                    title=f"{stem} | B{block:02d} RN→patch mean-head",
                )
                raw_payload["rn_attention"] = rn_rows[i].cpu().numpy()
                raw_payload["rn_patch_attention"] = rn_rows[i, :, 1:-1].cpu().numpy()
            _npz(root / "backbone" / "raw" / f"B{block:02d}" / f"{stem}.npz", **raw_payload)

            reg = final_register_mask[i].bool()
            for head in range(attention.shape[1]):
                cp = cls_patch[i, head]
                row = {
                    "image": sample.image_name,
                    "block": block,
                    "head": head,
                    "cls_self": _as_float(cls_rows[i, head, 0]),
                    "cls_patch_mass": _as_float(cp.sum()),
                    "cls_patch_max": _as_float(cp.max()),
                    "cls_register_mass": _as_float(cp[reg].sum()) if bool(reg.any()) else 0.0,
                    "cls_norm_post": _as_float(state[i, 0].norm()),
                    "patch_norm_mean_post": _as_float(patch_norm[i].mean()),
                    "patch_norm_max_post": _as_float(patch_norm[i].max()),
                }
                if has_rn and rn_rows is not None:
                    rp = rn_rows[i, head, 1:-1]
                    row.update(
                        {
                            "cls_to_rn": _as_float(cls_rows[i, head, -1]),
                            "rn_to_cls": _as_float(rn_rows[i, head, 0]),
                            "rn_to_rn": _as_float(rn_rows[i, head, -1]),
                            "rn_patch_mass": _as_float(rp.sum()),
                            "rn_patch_max": _as_float(rp.max()),
                            "rn_register_mass": _as_float(rp[reg].sum()) if bool(reg.any()) else 0.0,
                            "rn_norm_post": _as_float(state[i, -1].norm()),
                        }
                    )
                rows.append(row)

            cache = paper_cache.setdefault(sample.image_name, {"rgb": rgb})
            if block == last_block:
                cache["backbone_b23_cls"] = cls_grid[i].cpu().numpy()
                if rn_grid is not None:
                    cache["backbone_b23_rn"] = rn_grid[i].cpu().numpy()


# =================================================================================================
# Bridge analysis
# =================================================================================================


def _source_metrics(details: Mapping[str, Any], row: int) -> dict[str, float]:
    source_logits = details["source_logits"].detach().float()
    probs = source_logits.sigmoid()
    stats = details["source_stats"].detach().float()
    names = (
        "glyph_mean",
        "glyph_max",
        "glyph_top8",
        "glyph_soft_area",
        "glyph_row_max_mean",
        "glyph_col_max_mean",
        "glyph_row_density_max",
        "glyph_logit_mean",
    )
    out = {
        "source_present_logit": _as_float(source_logits[row, 0]),
        "source_readable_logit": _as_float(source_logits[row, 1]),
        "source_present_prob": _as_float(probs[row, 0]),
        "source_readable_prob": _as_float(probs[row, 1]),
        "source_gate": _as_float(details["source_gate"][row]),
    }
    for index, name in enumerate(names):
        out[name] = _as_float(stats[row, index])
    return out


def analyze_source(
    root: Path,
    model: Any,
    batch_samples: Sequence[DemoSample],
    rgb_batch: Sequence[np.ndarray],
    details: Mapping[str, Any],
    rows: list[dict[str, Any]],
    paper_cache: dict[str, dict[str, Any]],
) -> None:
    source = details.get("source_details") or {}
    tap_weights = source.get("tap_weights")
    per_block = source.get("per_block_logits", {})
    mixed_probs = details["glyph_probs"].detach().float()
    mixed_grid = _to_grid(mixed_probs)
    blocks = [int(x) for x in per_block.keys()]

    for i, (sample, rgb) in enumerate(zip(batch_samples, rgb_batch)):
        row = {"image": sample.image_name}
        row.update(_source_metrics(details, i))
        for block, weight in zip(blocks, tap_weights if tap_weights is not None else []):
            row[f"tap_weight_B{block}"] = _as_float(weight)
        rows.append(row)
        grid = mixed_grid[i].cpu().numpy()
        save_overlay(
            rgb,
            grid,
            root / "source" / "mixed" / f"{sample.stem}.png",
            title=f"{sample.stem} | SOURCE mixed glyph probability",
        )
        _npz(
            root / "source" / "raw" / "mixed" / f"{sample.stem}.npz",
            mixed_glyph_logits=details["glyph_logits"][i].detach().float().cpu().numpy(),
            mixed_glyph_probs=mixed_probs[i].cpu().numpy(),
            source_logits=details["source_logits"][i].detach().float().cpu().numpy(),
            source_stats=details["source_stats"][i].detach().float().cpu().numpy(),
        )
        cache = paper_cache.setdefault(sample.image_name, {"rgb": rgb})
        cache["source_mixed"] = grid

    for block, logits in per_block.items():
        block = int(block)
        probs = logits.detach().float().sigmoid()
        grids = _to_grid(probs)
        for i, (sample, rgb) in enumerate(zip(batch_samples, rgb_batch)):
            save_overlay(
                rgb,
                grids[i].cpu().numpy(),
                root / "source" / "taps" / f"B{block:02d}" / f"{sample.stem}.png",
                title=f"{sample.stem} | SOURCE B{block:02d}",
            )
            _npz(
                root / "source" / "raw" / f"B{block:02d}" / f"{sample.stem}.npz",
                patch_logits=logits[i].detach().float().cpu().numpy(),
                patch_probs=probs[i].cpu().numpy(),
            )


def analyze_ortho(
    root: Path,
    model: Any,
    batch_samples: Sequence[DemoSample],
    rgb_batch: Sequence[np.ndarray],
    details: Mapping[str, Any],
    rows: list[dict[str, Any]],
    paper_cache: dict[str, dict[str, Any]],
    *,
    save_prompt_overlays: bool,
) -> None:
    component = details.get("orthographic_details") or {}
    per_block = component.get("per_block_attention", {})
    if not per_block:
        raise RuntimeError("ORTHO details missing; return_details=True is required")
    blocks = [int(x) for x in per_block.keys()]
    weights = component["tap_weights"].detach().float()
    stacked = torch.stack([per_block[block].detach().float() for block in blocks], dim=0)
    mixed_attention = torch.einsum("k,kbnhp->bnhp", weights, stacked)
    mixed_mean = mixed_attention.mean(dim=2)

    states = details["visual_states"]
    bridge = model.read_implant.orthographic_bridge
    mixed_contrib_vec: torch.Tensor | None = None
    per_block_contrib_norm: dict[int, torch.Tensor] = {}
    for slot, block in enumerate(blocks):
        visual = model.read_implant._spatial_state(states[block], block)
        vectors = ortho_token_contribution_vectors(bridge, visual, per_block[block])
        per_block_contrib_norm[block] = vectors.norm(dim=-1)
        weighted = weights[slot] * vectors
        mixed_contrib_vec = weighted if mixed_contrib_vec is None else mixed_contrib_vec + weighted
        del vectors
    if mixed_contrib_vec is None:
        raise RuntimeError("No ORTHO contribution vectors")
    mixed_contrib_norm = mixed_contrib_vec.norm(dim=-1)
    del mixed_contrib_vec

    for i, (sample, rgb) in enumerate(zip(batch_samples, rgb_batch)):
        raw_mixed_attention = mixed_attention[i].cpu().numpy()
        raw_mixed_contrib = mixed_contrib_norm[i].cpu().numpy()
        _npz(
            root / "ortho" / "raw" / "mixed" / f"{sample.stem}.npz",
            prompt_words=np.asarray(PROMPT_WORDS),
            attention_per_head=raw_mixed_attention,
            attention_mean_head=mixed_mean[i].cpu().numpy(),
            contribution_norm=raw_mixed_contrib,
        )
        for word_index, word in enumerate(PROMPT_WORDS):
            present, text = _candidate_flags(sample, word)
            amap = _to_grid(mixed_mean[i, word_index]).cpu().numpy()
            cmap = _to_grid(mixed_contrib_norm[i, word_index]).cpu().numpy()
            if save_prompt_overlays:
                wslug = _slug(word)
                save_overlay(
                    rgb,
                    amap,
                    root / "ortho" / "mixed_attention" / sample.stem / f"{wslug}.png",
                    title=f"{sample.stem} | ORTHO mixed attention | {word}",
                )
                save_overlay(
                    rgb,
                    cmap,
                    root / "ortho" / "mixed_contribution_norm" / sample.stem / f"{wslug}.png",
                    title=f"{sample.stem} | ORTHO mixed contribution norm | {word}",
                    cmap="magma",
                )
            cache = paper_cache.setdefault(sample.image_name, {"rgb": rgb})
            cache.setdefault("ortho", {})[word] = {"attention": amap, "contribution": cmap}
            row = {
                "image": sample.image_name,
                "word": word,
                "is_known_present": present,
                "is_text_word": text,
                "mixed_attention_mass": _as_float(mixed_mean[i, word_index].sum()),
                "mixed_attention_max": _as_float(mixed_mean[i, word_index].max()),
                "mixed_contribution_norm_sum": _as_float(mixed_contrib_norm[i, word_index].sum()),
                "mixed_contribution_norm_max": _as_float(mixed_contrib_norm[i, word_index].max()),
            }
            for head in range(mixed_attention.shape[2]):
                head_map = mixed_attention[i, word_index, head]
                row[f"mixed_H{head}_attention_mass"] = _as_float(head_map.sum())
                row[f"mixed_H{head}_attention_max"] = _as_float(head_map.max())
            for slot, block in enumerate(blocks):
                block_heads = per_block[block][i, word_index].detach().float()
                mean = block_heads.mean(dim=0)
                cn = per_block_contrib_norm[block][i, word_index]
                row[f"tap_weight_B{block}"] = _as_float(weights[slot])
                row[f"B{block}_attention_mass"] = _as_float(mean.sum())
                row[f"B{block}_attention_max"] = _as_float(mean.max())
                row[f"B{block}_contribution_norm_sum"] = _as_float(cn.sum())
                for head in range(block_heads.shape[0]):
                    row[f"B{block}_H{head}_attention_mass"] = _as_float(block_heads[head].sum())
                    row[f"B{block}_H{head}_attention_max"] = _as_float(block_heads[head].max())
            rows.append(row)

        for block in blocks:
            attn = per_block[block][i].detach().float()
            mean = attn.mean(dim=1)
            cn = per_block_contrib_norm[block][i]
            _npz(
                root / "ortho" / "raw" / "taps" / f"B{block:02d}" / f"{sample.stem}.npz",
                prompt_words=np.asarray(PROMPT_WORDS),
                attention_per_head=attn.cpu().numpy(),
                attention_mean_head=mean.cpu().numpy(),
                contribution_norm=cn.cpu().numpy(),
            )
            if save_prompt_overlays:
                for word_index, word in enumerate(PROMPT_WORDS):
                    wslug = _slug(word)
                    save_overlay(
                        rgb,
                        _to_grid(mean[word_index]).cpu().numpy(),
                        root / "ortho" / "taps" / f"B{block:02d}" / "attention" / sample.stem / f"{wslug}.png",
                        title=f"{sample.stem} | ORTHO B{block:02d} attention | {word}",
                    )
                    save_overlay(
                        rgb,
                        _to_grid(cn[word_index]).cpu().numpy(),
                        root / "ortho" / "taps" / f"B{block:02d}" / "contribution_norm" / sample.stem / f"{wslug}.png",
                        title=f"{sample.stem} | ORTHO B{block:02d} contribution | {word}",
                        cmap="magma",
                    )


def analyze_read(
    root: Path,
    model: Any,
    batch_samples: Sequence[DemoSample],
    rgb_batch: Sequence[np.ndarray],
    details: Mapping[str, Any],
    rows: list[dict[str, Any]],
    paper_cache: dict[str, dict[str, Any]],
    read_cache: dict[tuple[str, str], dict[str, Any]],
    *,
    save_prompt_overlays: bool,
) -> None:
    component = details.get("read_details") or {}
    per_block = component.get("per_block", {})
    if not per_block:
        raise RuntimeError("READ details missing; return_details=True is required")
    blocks = [int(x) for x in per_block.keys()]
    weights = component["tap_weights"].detach().float()
    bridge = model.read_implant.read_bridge
    states = details["visual_states"]

    patch_stack = torch.stack([per_block[b]["patch_attention"].detach().float() for b in blocks], dim=0)
    mixed_patch = torch.einsum("k,kbnht->bnht", weights, patch_stack)

    register_tensors: list[torch.Tensor] = []
    for block in blocks:
        reg = per_block[block].get("register_attention")
        if reg is None:
            reg = torch.zeros_like(per_block[block]["patch_attention"])
        register_tensors.append(reg.detach().float())
    register_stack = torch.stack(register_tensors, dim=0)
    mixed_register = torch.einsum("k,kbnht->bnht", weights, register_stack)
    mixed_effective = mixed_patch + bridge.register_gate.detach().float() * mixed_register

    mixed_contrib_vec: torch.Tensor | None = None
    per_block_contrib_norm: dict[int, torch.Tensor] = {}
    per_block_effective: dict[int, torch.Tensor] = {}
    for slot, block in enumerate(blocks):
        vectors, effective = read_token_contribution_vectors(
            bridge,
            states[block],
            per_block[block]["patch_attention"],
            per_block[block].get("register_attention"),
        )
        per_block_contrib_norm[block] = vectors.norm(dim=-1)
        per_block_effective[block] = effective
        weighted = weights[slot] * vectors
        mixed_contrib_vec = weighted if mixed_contrib_vec is None else mixed_contrib_vec + weighted
        del vectors
    if mixed_contrib_vec is None:
        raise RuntimeError("No READ contribution vectors")
    mixed_contrib_norm = mixed_contrib_vec.norm(dim=-1)
    del mixed_contrib_vec

    patch_mean = mixed_patch.mean(dim=2)
    register_mean = mixed_register.mean(dim=2)
    effective_mean = mixed_effective.mean(dim=2)
    rn_attention = component["read_null_attention"].detach().float()

    for i, (sample, rgb) in enumerate(zip(batch_samples, rgb_batch)):
        _npz(
            root / "read" / "raw" / "mixed" / f"{sample.stem}.npz",
            prompt_words=np.asarray(PROMPT_WORDS),
            patch_attention_per_head=mixed_patch[i].cpu().numpy(),
            register_attention_per_head=mixed_register[i].cpu().numpy(),
            effective_attention_per_head=mixed_effective[i].cpu().numpy(),
            contribution_norm=mixed_contrib_norm[i].cpu().numpy(),
            read_null_attention=rn_attention[i].cpu().numpy(),
            register_gate=np.asarray(_as_float(bridge.register_gate), dtype=np.float32),
        )
        for word_index, word in enumerate(PROMPT_WORDS):
            present, text = _candidate_flags(sample, word)
            spatial_patch = patch_mean[i, word_index, 1:-1]
            spatial_register = register_mean[i, word_index, 1:-1]
            spatial_effective = effective_mean[i, word_index, 1:-1]
            spatial_contrib = mixed_contrib_norm[i, word_index, 1:-1]
            patch_grid = _to_grid(spatial_patch).cpu().numpy()
            register_grid = _to_grid(spatial_register).cpu().numpy()
            effective_grid = _to_grid(spatial_effective).cpu().numpy()
            contrib_grid = _to_grid(spatial_contrib).cpu().numpy()
            wslug = _slug(word)
            if save_prompt_overlays:
                save_overlay(
                    rgb,
                    patch_grid,
                    root / "read" / "mixed_patch_attention" / sample.stem / f"{wslug}.png",
                    title=f"{sample.stem} | READ mixed patch/non-register | {word}",
                )
                save_overlay(
                    rgb,
                    register_grid,
                    root / "read" / "mixed_register_attention" / sample.stem / f"{wslug}.png",
                    title=f"{sample.stem} | READ mixed register-only | {word}",
                )
                save_overlay(
                    rgb,
                    effective_grid,
                    root / "read" / "mixed_effective_attention" / sample.stem / f"{wslug}.png",
                    title=f"{sample.stem} | READ effective attention | {word}",
                    signed=True,
                )
                save_overlay(
                    rgb,
                    contrib_grid,
                    root / "read" / "mixed_contribution_norm" / sample.stem / f"{wslug}.png",
                    title=f"{sample.stem} | READ contribution norm | {word}",
                    cmap="magma",
                )
            cache_entry = {
                "patch": patch_grid,
                "register": register_grid,
                "effective": effective_grid,
                "contribution": contrib_grid,
                "rn_attention": _as_float(rn_attention[i, word_index]),
            }
            read_cache[(sample.image_name, word)] = cache_entry
            cache = paper_cache.setdefault(sample.image_name, {"rgb": rgb})
            cache.setdefault("read", {})[word] = cache_entry

            patch_mass = patch_mean[i, word_index, 1:-1].sum()
            reg_mass = register_mean[i, word_index, 1:-1].sum()
            row = {
                "image": sample.image_name,
                "word": word,
                "is_known_present": present,
                "is_text_word": text,
                "register_gate": _as_float(bridge.register_gate),
                "mixed_patch_spatial_mass": _as_float(patch_mass),
                "mixed_register_spatial_mass": _as_float(reg_mass),
                "mixed_effective_spatial_mass": _as_float(spatial_effective.sum()),
                "mixed_patch_spatial_max": _as_float(spatial_patch.max()),
                "mixed_register_spatial_max": _as_float(spatial_register.max()),
                "mixed_effective_abs_max": _as_float(spatial_effective.abs().max()),
                "mixed_read_null_attention": _as_float(rn_attention[i, word_index]),
                "mixed_contribution_norm_sum": _as_float(spatial_contrib.sum()),
                "mixed_contribution_norm_max": _as_float(spatial_contrib.max()),
            }
            for head in range(mixed_patch.shape[2]):
                hp = mixed_patch[i, word_index, head]
                hr = mixed_register[i, word_index, head]
                he = mixed_effective[i, word_index, head]
                row[f"mixed_H{head}_patch_spatial_mass"] = _as_float(hp[1:-1].sum())
                row[f"mixed_H{head}_register_spatial_mass"] = _as_float(hr[1:-1].sum())
                row[f"mixed_H{head}_effective_spatial_sum"] = _as_float(he[1:-1].sum())
                row[f"mixed_H{head}_rn_attention"] = _as_float(hp[-1])
            for slot, block in enumerate(blocks):
                p_heads = per_block[block]["patch_attention"][i, word_index].detach().float()
                r_heads = register_tensors[slot][i, word_index].detach().float()
                p = p_heads.mean(dim=0)
                r = r_heads.mean(dim=0)
                c = per_block_contrib_norm[block][i, word_index]
                row[f"tap_weight_B{block}"] = _as_float(weights[slot])
                row[f"B{block}_patch_spatial_mass"] = _as_float(p[1:-1].sum())
                row[f"B{block}_register_spatial_mass"] = _as_float(r[1:-1].sum())
                row[f"B{block}_rn_attention"] = _as_float(p[-1])
                row[f"B{block}_contribution_norm_sum"] = _as_float(c[1:-1].sum())
                for head in range(p_heads.shape[0]):
                    row[f"B{block}_H{head}_patch_spatial_mass"] = _as_float(p_heads[head, 1:-1].sum())
                    row[f"B{block}_H{head}_register_spatial_mass"] = _as_float(r_heads[head, 1:-1].sum())
                    row[f"B{block}_H{head}_rn_attention"] = _as_float(p_heads[head, -1])
            rows.append(row)

        for slot, block in enumerate(blocks):
            p = per_block[block]["patch_attention"][i].detach().float()
            r = register_tensors[slot][i].detach().float()
            e = per_block_effective[block][i].detach().float()
            c = per_block_contrib_norm[block][i].detach().float()
            _npz(
                root / "read" / "raw" / "taps" / f"B{block:02d}" / f"{sample.stem}.npz",
                prompt_words=np.asarray(PROMPT_WORDS),
                patch_attention_per_head=p.cpu().numpy(),
                register_attention_per_head=r.cpu().numpy(),
                effective_attention_per_head=e.cpu().numpy(),
                contribution_norm=c.cpu().numpy(),
            )
            if save_prompt_overlays:
                for word_index, word in enumerate(PROMPT_WORDS):
                    wslug = _slug(word)
                    pm = p[word_index].mean(dim=0)[1:-1]
                    rm = r[word_index].mean(dim=0)[1:-1]
                    em = e[word_index].mean(dim=0)[1:-1]
                    cm = c[word_index][1:-1]
                    save_overlay(
                        rgb,
                        _to_grid(pm).cpu().numpy(),
                        root / "read" / "taps" / f"B{block:02d}" / "patch_attention" / sample.stem / f"{wslug}.png",
                        title=f"{sample.stem} | READ B{block:02d} patch | {word}",
                    )
                    save_overlay(
                        rgb,
                        _to_grid(rm).cpu().numpy(),
                        root / "read" / "taps" / f"B{block:02d}" / "register_attention" / sample.stem / f"{wslug}.png",
                        title=f"{sample.stem} | READ B{block:02d} register | {word}",
                    )
                    save_overlay(
                        rgb,
                        _to_grid(em).cpu().numpy(),
                        root / "read" / "taps" / f"B{block:02d}" / "effective_attention" / sample.stem / f"{wslug}.png",
                        title=f"{sample.stem} | READ B{block:02d} effective | {word}",
                        signed=True,
                    )
                    save_overlay(
                        rgb,
                        _to_grid(cm).cpu().numpy(),
                        root / "read" / "taps" / f"B{block:02d}" / "contribution_norm" / sample.stem / f"{wslug}.png",
                        title=f"{sample.stem} | READ B{block:02d} contribution | {word}",
                        cmap="magma",
                    )


def analyze_correction(
    root: Path,
    model: Any,
    batch_samples: Sequence[DemoSample],
    rgb_batch: Sequence[np.ndarray],
    details: Mapping[str, Any],
    rows: list[dict[str, Any]],
    paper_cache: dict[str, dict[str, Any]],
) -> None:
    component = details.get("content_details") or {}
    per_block = component.get("per_block_attention", {})
    if not per_block:
        raise RuntimeError("CONTENT/correction details missing; correction=True is required")
    blocks = [int(x) for x in per_block.keys()]
    weights = component["tap_weights"].detach().float()
    stacked = torch.stack([per_block[b].detach().float() for b in blocks], dim=0)
    mixed_attention = torch.einsum("k,kbht->bht", weights, stacked)
    states = details["visual_states"]
    pool = model.read_implant.content_pool

    mixed_contrib_vec: torch.Tensor | None = None
    per_block_contrib_norm: dict[int, torch.Tensor] = {}
    for slot, block in enumerate(blocks):
        vectors = content_token_contribution_vectors(pool, states[block], per_block[block])
        per_block_contrib_norm[block] = vectors.norm(dim=-1)
        weighted = weights[slot] * vectors
        mixed_contrib_vec = weighted if mixed_contrib_vec is None else mixed_contrib_vec + weighted
        del vectors
    if mixed_contrib_vec is None:
        raise RuntimeError("No CONTENT contribution vectors")
    mixed_contrib_norm = mixed_contrib_vec.norm(dim=-1)
    del mixed_contrib_vec

    mixed_mean = mixed_attention.mean(dim=1)
    correction = details["content_correction"].detach().float()
    base = details["base_image_embedding"].detach().float()
    corrected = details["content_image_embedding"].detach().float()
    correction_norm = correction.norm(dim=-1)
    relative = correction_norm / (base.norm(dim=-1) + 1e-12)
    base_corrected_cos = (_normalize_rows(base) * _normalize_rows(corrected)).sum(dim=-1)

    for i, (sample, rgb) in enumerate(zip(batch_samples, rgb_batch)):
        spatial_attn = mixed_mean[i, 1:-1]
        spatial_contrib = mixed_contrib_norm[i, 1:-1]
        attn_grid = _to_grid(spatial_attn).cpu().numpy()
        contrib_grid = _to_grid(spatial_contrib).cpu().numpy()
        save_overlay(
            rgb,
            attn_grid,
            root / "correction" / "mixed_attention" / f"{sample.stem}.png",
            title=f"{sample.stem} | correction mixed attention",
        )
        save_overlay(
            rgb,
            contrib_grid,
            root / "correction" / "mixed_contribution_norm" / f"{sample.stem}.png",
            title=f"{sample.stem} | correction mixed contribution norm",
            cmap="magma",
        )
        _npz(
            root / "correction" / "raw" / "mixed" / f"{sample.stem}.npz",
            attention_per_head=mixed_attention[i].cpu().numpy(),
            contribution_norm=mixed_contrib_norm[i].cpu().numpy(),
            correction_vector=correction[i].cpu().numpy(),
        )
        row = {
            "image": sample.image_name,
            "correction_vector_norm": _as_float(correction_norm[i]),
            "correction_relative_norm": _as_float(relative[i]),
            "base_corrected_cosine": _as_float(base_corrected_cos[i]),
            "mixed_attention_spatial_mass": _as_float(spatial_attn.sum()),
            "mixed_attention_spatial_max": _as_float(spatial_attn.max()),
            "mixed_contribution_norm_sum": _as_float(spatial_contrib.sum()),
            "mixed_contribution_norm_max": _as_float(spatial_contrib.max()),
        }
        for head in range(mixed_attention.shape[1]):
            ha = mixed_attention[i, head]
            row[f"mixed_H{head}_attention_spatial_mass"] = _as_float(ha[1:-1].sum())
            row[f"mixed_H{head}_attention_spatial_max"] = _as_float(ha[1:-1].max())
        for slot, block in enumerate(blocks):
            block_heads = per_block[block][i].detach().float()
            a = block_heads.mean(dim=0)
            c = per_block_contrib_norm[block][i]
            row[f"tap_weight_B{block}"] = _as_float(weights[slot])
            row[f"B{block}_attention_spatial_mass"] = _as_float(a[1:-1].sum())
            row[f"B{block}_contribution_norm_sum"] = _as_float(c[1:-1].sum())
            for head in range(block_heads.shape[0]):
                row[f"B{block}_H{head}_attention_spatial_mass"] = _as_float(block_heads[head, 1:-1].sum())
                row[f"B{block}_H{head}_attention_spatial_max"] = _as_float(block_heads[head, 1:-1].max())
        rows.append(row)
        cache = paper_cache.setdefault(sample.image_name, {"rgb": rgb})
        cache["correction_attention"] = attn_grid
        cache["correction_contribution"] = contrib_grid

        for block in blocks:
            a = per_block[block][i].detach().float()
            c = per_block_contrib_norm[block][i].detach().float()
            _npz(
                root / "correction" / "raw" / "taps" / f"B{block:02d}" / f"{sample.stem}.npz",
                attention_per_head=a.cpu().numpy(),
                contribution_norm=c.cpu().numpy(),
            )
            save_overlay(
                rgb,
                _to_grid(a.mean(dim=0)[1:-1]).cpu().numpy(),
                root / "correction" / "taps" / f"B{block:02d}" / "attention" / f"{sample.stem}.png",
                title=f"{sample.stem} | correction B{block:02d} attention",
            )
            save_overlay(
                rgb,
                _to_grid(c[1:-1]).cpu().numpy(),
                root / "correction" / "taps" / f"B{block:02d}" / "contribution_norm" / f"{sample.stem}.png",
                title=f"{sample.stem} | correction B{block:02d} contribution",
                cmap="magma",
            )


def save_register_diagnostics(
    root: Path,
    batch_samples: Sequence[DemoSample],
    rgb_batch: Sequence[np.ndarray],
    details: Mapping[str, Any],
    rows: list[dict[str, Any]],
    paper_cache: dict[str, dict[str, Any]],
) -> None:
    norms = details["patch_token_norms"].detach().float()
    mask = details["register_mask"].detach().bool()
    norm_grid = _to_grid(norms)
    mask_grid = _to_grid(mask.float())
    for i, (sample, rgb) in enumerate(zip(batch_samples, rgb_batch)):
        ng = norm_grid[i].cpu().numpy()
        mg = mask_grid[i].cpu().numpy()
        save_plain_map(
            ng,
            root / "token_diagnostics" / "final_patch_norms" / f"{sample.stem}.png",
            title=f"{sample.stem} | final patch norm",
            cmap="viridis",
        )
        save_overlay(
            rgb,
            mg,
            root / "token_diagnostics" / "register_mask" / f"{sample.stem}.png",
            title=f"{sample.stem} | final implicit-register mask",
            cmap="Reds",
            alpha=0.62,
        )
        indices = torch.nonzero(mask[i], as_tuple=False).flatten().cpu().tolist()
        rows.append(
            {
                "image": sample.image_name,
                "register_count": int(mask[i].sum().item()),
                "register_indices_json": json.dumps(indices),
                "patch_norm_mean": _as_float(norms[i].mean()),
                "patch_norm_max": _as_float(norms[i].max()),
                "register_norm_mean": _as_float(norms[i][mask[i]].mean()) if bool(mask[i].any()) else float("nan"),
            }
        )
        cache = paper_cache.setdefault(sample.image_name, {"rgb": rgb})
        cache["register_mask"] = mg
        cache["final_patch_norm"] = ng


# =================================================================================================
# Scores and routing
# =================================================================================================


def collect_score_rows(
    model: Any,
    batch_samples: Sequence[DemoSample],
    any_output: Any,
    classic_output: Any,
    rows: list[dict[str, Any]],
    paper_cache: dict[str, dict[str, Any]],
    *,
    model_label: str,
) -> None:
    details = any_output.details or {}
    scale = model.logit_scale.detach().float().exp()
    classic_logits = classic_output.logits_per_image.detach().float()
    classic_cos = classic_logits / scale
    content_image = _normalize_rows(details["content_image_embedding"])
    content_text = _normalize_rows(details["content_text_embedding"])
    content_cos = content_image @ content_text.t()
    content_logits = scale * content_cos
    any_logits = any_output.logits_per_image.detach().float()
    base_image = _normalize_rows(details["base_image_embedding"])
    base_content_cos = base_image @ content_text.t()

    for i, sample in enumerate(batch_samples):
        cache = paper_cache.setdefault(sample.image_name, {})
        cache.setdefault("scores", {})[model_label] = {}
        for word_index, word in enumerate(PROMPT_WORDS):
            present, text = _candidate_flags(sample, word)
            row = {
                "model": model_label,
                "image": sample.image_name,
                "word": word,
                "is_known_present": present,
                "is_text_word": text,
                "classic_cosine": _as_float(classic_cos[i, word_index]),
                "classic_logit": _as_float(classic_logits[i, word_index]),
                "base_content_cosine": _as_float(base_content_cos[i, word_index]),
                "corrected_content_cosine": _as_float(content_cos[i, word_index]),
                "correction_cosine_delta": _as_float(content_cos[i, word_index] - base_content_cos[i, word_index]),
                "content_logit": _as_float(content_logits[i, word_index]),
                "any_logit": _as_float(any_logits[i, word_index]),
                "raw_read_logit": _as_float(details["raw_read_logits"][i, word_index]),
                "read_logit": _as_float(details["read_logits"][i, word_index]),
                "null_read_logit": _as_float(details["null_read_logits"][i]),
                "relative_read_logit": _as_float(details["relative_read_logits"][i, word_index]),
                "early_orthographic_logit": _as_float(details["early_orthographic_logits"][i, word_index]),
                "trust_gate": _as_float(details["trust_gate"][i, word_index]),
                "source_gate": _as_float(details["source_gate"][i]),
                "route_gate": _as_float(details["route_gate"][i, word_index]),
                "auto_read_contribution": _as_float(details["auto_read_contribution"][i, word_index]),
                "read_null_attention": _as_float(details["read_null_attention"][i, word_index]),
            }
            rows.append(row)
            cache["scores"][model_label][word] = row


# =================================================================================================
# Gradient-weighted last-block classic attention rollout
# =================================================================================================


def _vision_to_block_input(model: Any, pixel_values: torch.Tensor, target_block: int) -> torch.Tensor:
    """Run untouched vision trunk under no_grad and return input to target block."""
    with torch.no_grad():
        vision = model.vision_model
        hidden = vision.embeddings(pixel_values, interpolate_pos_encoding=False)
        hidden = vision.pre_layrnorm(hidden)
        insert_block = int(model.config.read_null_insert_block)
        for block, layer in enumerate(vision.encoder.layers):
            if block == insert_block:
                rn = model.read_null_token.to(device=hidden.device, dtype=hidden.dtype)
                hidden = torch.cat((hidden, rn.view(1, 1, -1).expand(hidden.shape[0], 1, -1)), dim=1)
            if block == target_block:
                return hidden.detach()
            hidden = _layer_tensor(layer(hidden, None))
    raise ValueError(f"target_block B{target_block} not reached")


def manual_last_block_with_attention(layer: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable standard HF CLIPEncoderLayer forward returning attention probabilities."""
    validate_hf_clip_layer(layer)
    residual = hidden_states
    x = layer.layer_norm1(hidden_states)
    attn = layer.self_attn
    batch, token_count, _ = x.shape
    heads = int(attn.num_heads)
    q = F.linear(x, attn.q_proj.weight, attn.q_proj.bias)
    k = F.linear(x, attn.k_proj.weight, attn.k_proj.bias)
    v = F.linear(x, attn.v_proj.weight, attn.v_proj.bias)
    head_dim = q.shape[-1] // heads
    scale = float(getattr(attn, "scale", head_dim ** -0.5))
    q = (q * scale).view(batch, token_count, heads, head_dim).transpose(1, 2)
    k = k.view(batch, token_count, heads, head_dim).transpose(1, 2)
    v = v.view(batch, token_count, heads, head_dim).transpose(1, 2)
    probs = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
    dropout_p = float(getattr(attn, "dropout", 0.0))
    used_probs = F.dropout(probs, p=dropout_p, training=False) if dropout_p > 0 else probs
    context = torch.matmul(used_probs, v)
    context = context.transpose(1, 2).reshape(batch, token_count, heads * head_dim)
    attn_out = F.linear(context, attn.out_proj.weight, attn.out_proj.bias)
    hidden = residual + attn_out
    residual = hidden
    hidden = residual + layer.mlp(layer.layer_norm2(hidden))
    return hidden, probs


def validate_manual_last_block(model: Any, hidden_input: torch.Tensor, target_block: int) -> dict[str, float]:
    layer = model.vision_model.encoder.layers[target_block]
    with torch.no_grad():
        official = _layer_tensor(layer(hidden_input, None)).float()
        manual, _ = manual_last_block_with_attention(layer, hidden_input)
        manual = manual.float()
        diff = (official - manual).abs()
        cos = F.cosine_similarity(official.reshape(official.shape[0], -1), manual.reshape(manual.shape[0], -1), dim=-1)
    result = {
        "max_abs": _as_float(diff.max()),
        "mean_abs": _as_float(diff.mean()),
        "cosine": _as_float(cos.mean()),
    }
    if result["max_abs"] > 5e-4 or result["cosine"] < 0.999999:
        raise RuntimeError(
            "Manual B23 forward does not match the installed HF CLIPEncoderLayer closely enough for rollout: "
            + json.dumps(result)
        )
    return result


def _clean_classic_text_embedding(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ids = model._pad_context(input_ids)
        ids = model._compact_control_tokens(
            ids,
            (
                model.config.hard_text_token_id,
                model.config.no_text_token_id,
                model.config.any_text_token_id,
            ),
        )
        info = model._encode_text_hidden(ids)
        return _normalize_rows(info["text_embedding"]).detach()


def _good_threshold(values: list[float], mode: str) -> float:
    if not values:
        raise ValueError("No known-present prompt score is available")
    if mode == "present_min":
        return float(min(values))
    if mode == "present_mean":
        return float(np.mean(values))
    if mode == "present_max":
        return float(max(values))
    raise ValueError(f"Unknown good-threshold mode {mode!r}")


def select_rollout_candidates(
    samples: Sequence[DemoSample],
    score_rows: Sequence[dict[str, Any]],
    threshold_mode: str,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    main_rows = [row for row in score_rows if row["model"] == "xattn_refit_wiseft"]
    lookup = {(row["image"], row["word"]): row for row in main_rows}
    selections: dict[str, list[str]] = {}
    selection_rows: list[dict[str, Any]] = []
    for sample in samples:
        present_scores = [float(lookup[(sample.image_name, word)]["classic_cosine"]) for word in sample.spec.present_words]
        threshold = _good_threshold(present_scores, threshold_mode)
        selected: list[str] = []
        ranked = sorted(
            PROMPT_WORDS,
            key=lambda w: float(lookup[(sample.image_name, w)]["classic_cosine"]),
            reverse=True,
        )
        for rank, word in enumerate(ranked, 1):
            score = float(lookup[(sample.image_name, word)]["classic_cosine"])
            is_present = word in sample.spec.present_words
            keep = is_present or score >= threshold
            if keep:
                selected.append(word)
            selection_rows.append(
                {
                    "image": sample.image_name,
                    "word": word,
                    "rank": rank,
                    "classic_cosine": score,
                    "is_known_present": int(is_present),
                    "is_text_word": int(word in sample.spec.text_words),
                    "good_threshold_mode": threshold_mode,
                    "good_threshold_cosine": threshold,
                    "selected_for_rollout": int(keep),
                }
            )
        selections[sample.image_name] = selected
    return selections, selection_rows


def run_classic_b23_rollouts(
    root: Path,
    model: Any,
    processor: Any,
    samples: Sequence[DemoSample],
    input_ids: torch.Tensor,
    selections: Mapping[str, Sequence[str]],
    device: torch.device,
    paper_cache: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    layers = len(model.vision_model.encoder.layers)
    target_block = layers - 1
    if target_block != 23:
        raise RuntimeError(f"This experiment is explicitly B23-only, but model last block is B{target_block}")
    text_norm = _clean_classic_text_embedding(model, input_ids)
    rows: list[dict[str, Any]] = []
    rollout_cache: dict[tuple[str, str], np.ndarray] = {}
    validation: dict[str, Any] = {}

    # Parameter gradients are unnecessary; attention remains differentiable through the detached B23 input.
    original_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        for sample_index, sample in enumerate(samples):
            selected = list(selections.get(sample.image_name, ()))
            if not selected:
                continue
            pixel_values = _preprocess_images(processor, [sample], device)
            rgb = _pixel_values_to_rgb(pixel_values, processor)[0]
            hidden_input = _vision_to_block_input(model, pixel_values, target_block)
            if not validation:
                validation = validate_manual_last_block(model, hidden_input, target_block)
            hidden_input = hidden_input.detach().requires_grad_(True)
            final_hidden, probs = manual_last_block_with_attention(
                model.vision_model.encoder.layers[target_block], hidden_input
            )
            image_embedding = model.visual_projection(model.vision_model.post_layernorm(final_hidden[:, 0, :]))
            image_norm = _normalize_rows(image_embedding)
            cosines = image_norm @ text_norm.t()

            word_to_index = {word: idx for idx, word in enumerate(PROMPT_WORDS)}
            for local_number, word in enumerate(selected):
                word_index = word_to_index[word]
                score = cosines[0, word_index]
                grad = torch.autograd.grad(
                    score,
                    probs,
                    retain_graph=local_number < len(selected) - 1,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                signed = (probs * grad).mean(dim=1)[0, 0]
                positive = (probs * grad).clamp(min=0).mean(dim=1)[0, 0]
                spatial_signed = signed[1:-1]
                spatial_positive = positive[1:-1]
                grid_signed = _to_grid(spatial_signed).detach().cpu().numpy()
                grid_positive = _to_grid(spatial_positive).detach().cpu().numpy()
                rn_value = _as_float(positive[-1])
                rollout_cache[(sample.image_name, word)] = grid_positive
                paper_cache.setdefault(sample.image_name, {"rgb": rgb}).setdefault("rollout", {})[word] = grid_positive
                wslug = _slug(word)
                save_overlay(
                    rgb,
                    grid_positive,
                    root / "rollout_classic_b23" / "positive" / sample.stem / f"{wslug}.png",
                    title=f"{sample.stem} | classic B23 grad×attn CLS→patch | {word}",
                    cmap="magma",
                )
                save_overlay(
                    rgb,
                    grid_signed,
                    root / "rollout_classic_b23" / "signed" / sample.stem / f"{wslug}.png",
                    title=f"{sample.stem} | classic B23 signed grad×attn | {word}",
                    signed=True,
                )
                _npz(
                    root / "rollout_classic_b23" / "raw" / sample.stem / f"{wslug}.npz",
                    attention=probs[0].detach().float().cpu().numpy(),
                    gradient=grad[0].detach().float().cpu().numpy(),
                    grad_x_attention_signed=signed.detach().float().cpu().numpy(),
                    grad_x_attention_positive=positive.detach().float().cpu().numpy(),
                    spatial_signed=spatial_signed.detach().float().cpu().numpy(),
                    spatial_positive=spatial_positive.detach().float().cpu().numpy(),
                )
                rows.append(
                    {
                        "image": sample.image_name,
                        "word": word,
                        "classic_cosine_recomputed": _as_float(score),
                        "is_known_present": int(word in sample.spec.present_words),
                        "is_text_word": int(word in sample.spec.text_words),
                        "positive_spatial_mass": _as_float(spatial_positive.sum()),
                        "positive_spatial_max": _as_float(spatial_positive.max()),
                        "signed_spatial_sum": _as_float(spatial_signed.sum()),
                        "signed_spatial_abs_sum": _as_float(spatial_signed.abs().sum()),
                        "positive_cls_to_rn": rn_value,
                    }
                )
            del final_hidden, probs, hidden_input, cosines, pixel_values
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for p, flag in zip(model.parameters(), original_requires_grad):
            p.requires_grad_(flag)
    return rows, rollout_cache, validation


# =================================================================================================
# Paper-oriented summary panels
# =================================================================================================


def build_paper_panels(
    root: Path,
    samples: Sequence[DemoSample],
    paper_cache: Mapping[str, dict[str, Any]],
    score_rows: Sequence[dict[str, Any]],
    rollout_cache: Mapping[tuple[str, str], np.ndarray],
) -> None:
    score_lookup = {
        (row["image"], row["word"]): row
        for row in score_rows
        if row["model"] == "xattn_refit_wiseft"
    }
    for sample in samples:
        cache = paper_cache.get(sample.image_name, {})
        rgb = cache.get("rgb")
        if rgb is None:
            continue
        overview: list[tuple[str, np.ndarray, bool]] = []
        for key, title, signed in (
            ("source_mixed", "SOURCE mixed", False),
            ("correction_attention", "correction attn", False),
            ("backbone_b23_cls", "B23 CLS→patch", False),
            ("backbone_b23_rn", "B23 RN→patch", False),
            ("register_mask", "register mask", False),
        ):
            if key in cache:
                overview.append((title, cache[key], signed))
        if overview:
            save_panel(
                rgb,
                overview,
                root / "paper_panels" / "overview" / f"{sample.stem}.png",
                title=f"{sample.stem}: candidate-independent visual diagnostics",
            )

        ranked = sorted(
            PROMPT_WORDS,
            key=lambda w: float(score_lookup[(sample.image_name, w)]["classic_cosine"]),
            reverse=True,
        )
        interesting = list(dict.fromkeys((*sample.spec.present_words, *sample.spec.text_words, ranked[0])))
        for word in interesting:
            panels: list[tuple[str, np.ndarray, bool]] = []
            if word in cache.get("ortho", {}):
                panels.append(("ORTHO mixed", cache["ortho"][word]["attention"], False))
            if word in cache.get("read", {}):
                panels.append(("READ patch", cache["read"][word]["patch"], False))
                panels.append(("READ effective", cache["read"][word]["effective"], True))
                panels.append(("READ contribution", cache["read"][word]["contribution"], False))
            if (sample.image_name, word) in rollout_cache:
                panels.append(("classic B23 grad×attn", rollout_cache[(sample.image_name, word)], False))
            if panels:
                s = score_lookup[(sample.image_name, word)]
                label_bits = []
                if word in sample.spec.present_words:
                    label_bits.append("present")
                if word in sample.spec.text_words:
                    label_bits.append("text")
                labels = ", ".join(label_bits) if label_bits else "not annotated present"
                save_panel(
                    rgb,
                    panels,
                    root / "paper_panels" / "candidate" / sample.stem / f"{_slug(word)}.png",
                    title=(
                        f"{sample.stem} | {word} ({labels}) | "
                        f"classic cos={float(s['classic_cosine']):.4f}, any={float(s['any_logit']):.3f}"
                    ),
                )


# =================================================================================================
# Native demoset pair deltas (no intervention)
# =================================================================================================


def save_pair_delta_panel(
    condition_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    panels: Sequence[tuple[str, np.ndarray]],
    path: Path,
    *,
    title: str,
) -> None:
    """Show the two inputs followed by signed condition-reference spatial deltas."""
    _ensure_dir(path.parent)
    columns = 2 + len(panels)
    fig, axes = plt.subplots(1, columns, figsize=(3.2 * columns, 3.45), dpi=150)
    axes = np.atleast_1d(axes).tolist()
    axes[0].imshow(condition_rgb)
    axes[0].set_title("condition", fontsize=9)
    axes[0].set_axis_off()
    axes[1].imshow(reference_rgb)
    axes[1].set_title("reference", fontsize=9)
    axes[1].set_axis_off()
    for ax, (panel_title, grid) in zip(axes[2:], panels):
        vmin, vmax = _finite_range(grid, signed=True)
        # Use the condition image as spatial context; the map itself is condition-reference.
        ax.imshow(condition_rgb, interpolation="nearest")
        ax.imshow(
            grid,
            cmap="coolwarm",
            alpha=0.56,
            interpolation="nearest",
            extent=(0, condition_rgb.shape[1], condition_rgb.shape[0], 0),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(panel_title, fontsize=9)
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_pairwise_deltas(
    root: Path,
    paper_cache: Mapping[str, dict[str, Any]],
    score_rows: Sequence[dict[str, Any]],
    rollout_cache: Mapping[tuple[str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Save native condition-reference deltas for the fixed demoset pairs."""
    score_lookup = {
        (row["image"], row["word"]): row
        for row in score_rows
        if row["model"] == "xattn_refit_wiseft"
    }
    score_delta_rows: list[dict[str, Any]] = []
    map_delta_rows: list[dict[str, Any]] = []

    candidate_independent = (
        ("source_mixed", "SOURCE mixed"),
        ("correction_attention", "correction attention"),
        ("correction_contribution", "correction contribution"),
        ("backbone_b23_cls", "B23 CLS→patch"),
        ("backbone_b23_rn", "B23 RN→patch"),
        ("final_patch_norm", "final patch norm"),
        ("register_mask", "register mask"),
    )

    for pair in PAIR_SPECS:
        condition = paper_cache.get(pair.condition)
        reference = paper_cache.get(pair.reference)
        if condition is None or reference is None:
            raise KeyError(f"Pair cache missing for {pair.name}: {pair.condition}, {pair.reference}")
        condition_rgb = condition["rgb"]
        reference_rgb = reference["rgb"]
        pair_root = root / "pairwise_deltas" / pair.name

        independent_panels: list[tuple[str, np.ndarray]] = []
        for key, title in candidate_independent:
            if key not in condition or key not in reference:
                continue
            delta = np.asarray(condition[key], dtype=np.float32) - np.asarray(reference[key], dtype=np.float32)
            independent_panels.append((f"Δ {title}", delta))
            _npz(
                pair_root / "raw" / "candidate_independent" / f"{key}.npz",
                condition=np.asarray(condition[key], dtype=np.float32),
                reference=np.asarray(reference[key], dtype=np.float32),
                delta=delta,
            )
            map_delta_rows.append(
                {
                    "pair": pair.name,
                    "condition": pair.condition,
                    "reference": pair.reference,
                    "word": "",
                    "component": key,
                    "delta_l1": float(np.abs(delta).sum()),
                    "delta_l2": float(np.linalg.norm(delta.reshape(-1))),
                    "delta_linf": float(np.abs(delta).max()),
                    "delta_signed_sum": float(delta.sum()),
                    "note": pair.note,
                }
            )
        if independent_panels:
            save_pair_delta_panel(
                condition_rgb,
                reference_rgb,
                independent_panels,
                pair_root / "overview.png",
                title=f"{pair.name}: condition − reference (native forward passes)",
            )

        for word in pair.words:
            panels: list[tuple[str, np.ndarray]] = []
            raw_payload: dict[str, Any] = {}

            if word in condition.get("ortho", {}) and word in reference.get("ortho", {}):
                for subkey, title in (("attention", "ORTHO attention"), ("contribution", "ORTHO contribution")):
                    a = np.asarray(condition["ortho"][word][subkey], dtype=np.float32)
                    b = np.asarray(reference["ortho"][word][subkey], dtype=np.float32)
                    delta = a - b
                    panels.append((f"Δ {title}", delta))
                    raw_payload[f"ortho_{subkey}_condition"] = a
                    raw_payload[f"ortho_{subkey}_reference"] = b
                    raw_payload[f"ortho_{subkey}_delta"] = delta
                    map_delta_rows.append(
                        {
                            "pair": pair.name,
                            "condition": pair.condition,
                            "reference": pair.reference,
                            "word": word,
                            "component": f"ortho_{subkey}",
                            "delta_l1": float(np.abs(delta).sum()),
                            "delta_l2": float(np.linalg.norm(delta.reshape(-1))),
                            "delta_linf": float(np.abs(delta).max()),
                            "delta_signed_sum": float(delta.sum()),
                            "note": pair.note,
                        }
                    )

            if word in condition.get("read", {}) and word in reference.get("read", {}):
                for subkey, title in (
                    ("patch", "READ patch"),
                    ("register", "READ register"),
                    ("effective", "READ effective"),
                    ("contribution", "READ contribution"),
                ):
                    a = np.asarray(condition["read"][word][subkey], dtype=np.float32)
                    b = np.asarray(reference["read"][word][subkey], dtype=np.float32)
                    delta = a - b
                    panels.append((f"Δ {title}", delta))
                    raw_payload[f"read_{subkey}_condition"] = a
                    raw_payload[f"read_{subkey}_reference"] = b
                    raw_payload[f"read_{subkey}_delta"] = delta
                    map_delta_rows.append(
                        {
                            "pair": pair.name,
                            "condition": pair.condition,
                            "reference": pair.reference,
                            "word": word,
                            "component": f"read_{subkey}",
                            "delta_l1": float(np.abs(delta).sum()),
                            "delta_l2": float(np.linalg.norm(delta.reshape(-1))),
                            "delta_linf": float(np.abs(delta).max()),
                            "delta_signed_sum": float(delta.sum()),
                            "note": pair.note,
                        }
                    )

            rollout_a = rollout_cache.get((pair.condition, word))
            rollout_b = rollout_cache.get((pair.reference, word))
            if rollout_a is not None and rollout_b is not None:
                a = np.asarray(rollout_a, dtype=np.float32)
                b = np.asarray(rollout_b, dtype=np.float32)
                delta = a - b
                panels.append(("Δ classic B23 grad×attn", delta))
                raw_payload["rollout_condition"] = a
                raw_payload["rollout_reference"] = b
                raw_payload["rollout_delta"] = delta
                map_delta_rows.append(
                    {
                        "pair": pair.name,
                        "condition": pair.condition,
                        "reference": pair.reference,
                        "word": word,
                        "component": "classic_b23_grad_x_attention",
                        "delta_l1": float(np.abs(delta).sum()),
                        "delta_l2": float(np.linalg.norm(delta.reshape(-1))),
                        "delta_linf": float(np.abs(delta).max()),
                        "delta_signed_sum": float(delta.sum()),
                        "note": pair.note,
                    }
                )

            if raw_payload:
                _npz(pair_root / "raw" / "candidate" / f"{_slug(word)}.npz", **raw_payload)
            if panels:
                save_pair_delta_panel(
                    condition_rgb,
                    reference_rgb,
                    panels,
                    pair_root / "candidate" / f"{_slug(word)}.png",
                    title=f"{pair.name} | {word} | condition − reference",
                )

        for word in PROMPT_WORDS:
            a = score_lookup[(pair.condition, word)]
            b = score_lookup[(pair.reference, word)]
            row: dict[str, Any] = {
                "pair": pair.name,
                "condition": pair.condition,
                "reference": pair.reference,
                "word": word,
                "condition_is_known_present": a["is_known_present"],
                "reference_is_known_present": b["is_known_present"],
                "condition_is_text_word": a["is_text_word"],
                "reference_is_text_word": b["is_text_word"],
                "note": pair.note,
            }
            for key in (
                "classic_cosine",
                "classic_logit",
                "base_content_cosine",
                "corrected_content_cosine",
                "correction_cosine_delta",
                "content_logit",
                "any_logit",
                "raw_read_logit",
                "read_logit",
                "null_read_logit",
                "relative_read_logit",
                "early_orthographic_logit",
                "trust_gate",
                "source_gate",
                "route_gate",
                "auto_read_contribution",
                "read_null_attention",
            ):
                row[f"condition_{key}"] = a[key]
                row[f"reference_{key}"] = b[key]
                row[f"delta_{key}"] = float(a[key]) - float(b[key])
            score_delta_rows.append(row)

    _write_json(
        root / "pairwise_deltas" / "_meta" / "pairs.json",
        [
            {
                "name": pair.name,
                "condition": pair.condition,
                "reference": pair.reference,
                "words": list(pair.words),
                "note": pair.note,
                "delta_definition": "condition - reference",
                "causal_claim": False,
            }
            for pair in PAIR_SPECS
        ],
    )
    _write_csv(root / "pairwise_deltas" / "candidate_score_deltas.csv", score_delta_rows)
    _write_csv(root / "pairwise_deltas" / "map_delta_metrics.csv", map_delta_rows)
    return score_delta_rows, map_delta_rows


# =================================================================================================
# New-vs-old READ comparison
# =================================================================================================


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-20:
        return float("nan")
    return float(np.dot(x, y) / denom)


def compare_read_models(
    root: Path,
    samples: Sequence[DemoSample],
    paper_cache: Mapping[str, dict[str, Any]],
    main_read: Mapping[tuple[str, str], dict[str, Any]],
    old_read: Mapping[tuple[str, str], dict[str, Any]],
    main_scores: Sequence[dict[str, Any]],
    old_scores: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    main_lookup = {(r["image"], r["word"]): r for r in main_scores if r["model"] == "xattn_refit_wiseft"}
    old_lookup = {(r["image"], r["word"]): r for r in old_scores if r["model"] == "xattn_original"}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rgb = paper_cache[sample.image_name]["rgb"]
        for word in PROMPT_WORDS:
            key = (sample.image_name, word)
            if key not in main_read or key not in old_read:
                continue
            new = main_read[key]
            old = old_read[key]
            effective_delta = np.asarray(new["effective"]) - np.asarray(old["effective"])
            contribution_delta = np.asarray(new["contribution"]) - np.asarray(old["contribution"])
            wslug = _slug(word)
            save_overlay(
                rgb,
                effective_delta,
                root / "_comparison" / "read_effective_delta" / sample.stem / f"{wslug}.png",
                title=f"{sample.stem} | READ effective alpha0.28 - original | {word}",
                signed=True,
            )
            save_overlay(
                rgb,
                contribution_delta,
                root / "_comparison" / "read_contribution_norm_delta" / sample.stem / f"{wslug}.png",
                title=f"{sample.stem} | READ contribution norm alpha0.28 - original | {word}",
                signed=True,
            )
            a = main_lookup[key]
            b = old_lookup[key]
            present, text = _candidate_flags(sample, word)
            rows.append(
                {
                    "image": sample.image_name,
                    "word": word,
                    "is_known_present": present,
                    "is_text_word": text,
                    "effective_map_cosine": _cosine_np(new["effective"], old["effective"]),
                    "patch_map_cosine": _cosine_np(new["patch"], old["patch"]),
                    "register_map_cosine": _cosine_np(new["register"], old["register"]),
                    "contribution_map_cosine": _cosine_np(new["contribution"], old["contribution"]),
                    "effective_delta_l1": float(np.abs(effective_delta).sum()),
                    "effective_delta_linf": float(np.abs(effective_delta).max()),
                    "contribution_delta_l1": float(np.abs(contribution_delta).sum()),
                    "rn_attention_delta": float(new["rn_attention"] - old["rn_attention"]),
                    "read_logit_delta": float(a["read_logit"] - b["read_logit"]),
                    "relative_read_logit_delta": float(a["relative_read_logit"] - b["relative_read_logit"]),
                    "trust_gate_delta": float(a["trust_gate"] - b["trust_gate"]),
                    "route_gate_delta": float(a["route_gate"] - b["route_gate"]),
                    "auto_read_contribution_delta": float(a["auto_read_contribution"] - b["auto_read_contribution"]),
                    "any_logit_delta": float(a["any_logit"] - b["any_logit"]),
                }
            )
    return rows


# =================================================================================================
# Model analysis driver
# =================================================================================================


def analyze_model(
    *,
    model_path: str | Path,
    root: Path,
    samples: Sequence[DemoSample],
    prompt_template: str,
    device: torch.device,
    batch_size: int,
    model_label: str,
    full_components: bool,
    save_prompt_overlays: bool,
) -> tuple[Any, Any, list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    print(f"\n[load] {model_label}: {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    for required in ("read_null_token", "read_implant", "_vision_with_intermediates", "_encode_text_hidden"):
        if not hasattr(model, required):
            raise TypeError(f"{model_path} is not the expected full x-attention HF model: missing {required}")

    _write_json(root / "_meta" / "model.json", _model_meta(model, model_path, prompt_template))
    input_ids = _tokenize_prompts(processor, prompt_template, device)

    score_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    ortho_rows: list[dict[str, Any]] = []
    read_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    backbone_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    read_cache: dict[tuple[str, str], dict[str, Any]] = {}
    paper_cache: dict[str, dict[str, Any]] = {}

    for start, batch_samples in _batches(samples, batch_size):
        print(f"[{model_label}] images {start + 1}-{start + len(batch_samples)} / {len(samples)}")
        pixel_values = _preprocess_images(processor, batch_samples, device)
        rgb_batch = _pixel_values_to_rgb(pixel_values, processor)
        for sample, rgb in zip(batch_samples, rgb_batch):
            paper_cache.setdefault(sample.image_name, {})["rgb"] = rgb
            if full_components:
                _save_rgb(rgb, root / "_meta" / "processed_inputs" / sample.image_name)

        with torch.inference_mode():
            any_output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                mode="any",
                correction=True,
                return_details=True,
                pieces_fp32=True,
            )
            classic_output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                mode="classic",
                return_details=True,
                pieces_fp32=True,
            )
        details = any_output.details or {}
        if not details:
            raise RuntimeError("Model returned no details")

        collect_score_rows(
            model,
            batch_samples,
            any_output,
            classic_output,
            score_rows,
            paper_cache,
            model_label=model_label,
        )
        analyze_read(
            root,
            model,
            batch_samples,
            rgb_batch,
            details,
            read_rows,
            paper_cache,
            read_cache,
            save_prompt_overlays=save_prompt_overlays,
        )

        if full_components:
            analyze_source(root, model, batch_samples, rgb_batch, details, source_rows, paper_cache)
            analyze_ortho(
                root,
                model,
                batch_samples,
                rgb_batch,
                details,
                ortho_rows,
                paper_cache,
                save_prompt_overlays=save_prompt_overlays,
            )
            analyze_correction(root, model, batch_samples, rgb_batch, details, correction_rows, paper_cache)
            save_register_diagnostics(root, batch_samples, rgb_batch, details, token_rows, paper_cache)
            trace = collect_backbone_trace(model, pixel_values)
            save_backbone_analysis(
                root,
                model,
                batch_samples,
                rgb_batch,
                trace,
                details["register_mask"].detach().bool(),
                backbone_rows,
                paper_cache,
            )
            del trace

        del any_output, classic_output, details, pixel_values
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(root / "scores" / "candidate_scores.csv", score_rows)
    _write_csv(root / "read" / "metrics.csv", read_rows)
    if full_components:
        _write_csv(root / "source" / "metrics.csv", source_rows)
        _write_csv(root / "ortho" / "metrics.csv", ortho_rows)
        _write_csv(root / "correction" / "metrics.csv", correction_rows)
        _write_csv(root / "backbone" / "metrics.csv", backbone_rows)
        _write_csv(root / "token_diagnostics" / "metrics.csv", token_rows)

    tables = {
        "source": source_rows,
        "ortho": ortho_rows,
        "read": read_rows,
        "correction": correction_rows,
        "backbone": backbone_rows,
        "tokens": token_rows,
    }
    return model, processor, score_rows, read_cache, paper_cache, tables


# =================================================================================================
# CLI / run metadata
# =================================================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF repo id or local HF model directory")
    parser.add_argument("--old-model", type=str, default=DEFAULT_OLD_MODEL, help="HF repo id or local HF model directory")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--good-threshold",
        choices=("present_min", "present_mean", "present_max"),
        default="present_min",
        help="Classic-cosine threshold for additional B23 rollout candidates; all known-present words are always included.",
    )
    parser.add_argument(
        "--compact-figures",
        action="store_true",
        help="Skip exhaustive per-prompt ORTHO/READ tap overlays; raw arrays, metrics, paper panels, and rollout remain.",
    )
    parser.add_argument(
        "--skip-old-xattn",
        action="store_true",
        help="Skip the original READ-branch comparison.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before running.",
    )
    return parser.parse_args()


def write_run_manifest(root: Path, args: argparse.Namespace, samples: Sequence[DemoSample]) -> None:
    manifest_rows = []
    for sample in samples:
        manifest_rows.append(
            {
                "image": sample.image_name,
                "path": str(sample.path),
                "sha256": _sha256(sample.path),
                "present_words": json.dumps(sample.spec.present_words),
                "text_words": json.dumps(sample.spec.text_words),
                "note": sample.spec.note,
            }
        )
    _write_csv(root / "_meta" / "dataset_manifest.csv", manifest_rows)
    _write_json(
        root / "_meta" / "run.json",
        {
            "argv": sys.argv,
            "model": str(args.model),
            "old_model": str(args.old_model),
            "image_dir": str(args.image_dir),
            "output_dir": str(args.output_dir),
            "prompt_template": args.prompt_template,
            "prompt_words": list(PROMPT_WORDS),
            "good_threshold": args.good_threshold,
            "batch_size": args.batch_size,
            "device": args.device,
            "compact_figures": bool(args.compact_figures),
            "skip_old_xattn": bool(args.skip_old_xattn),
            "no_transplantation": True,
            "old_model_duplication_policy": {
                "duplicated": [
                    "READ tap attention/contribution",
                    "READ_NULL attention",
                    "read/null/relative logits",
                    "trust/route gate",
                    "auto READ contribution",
                    "final any-mode scores",
                ],
                "not_duplicated_because_wise_checkpoint_does_not_change_them": [
                    "vision backbone/resblocks",
                    "READ_NULL token parameter",
                    "SOURCE head/taps",
                    "ORTHO bridge/taps",
                    "CONTENT/correction pool/taps",
                    "classic B23 rollout",
                ],
                "expected_changed_parameters": {
                    "prefix": EXPECTED_WISE_DIFF_PREFIX,
                    "exact": sorted(EXPECTED_WISE_DIFF_EXACT),
                },
            },
        },
    )


# =================================================================================================
# Main
# =================================================================================================


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if "{prompt_word}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain {prompt_word}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    root = args.output_dir
    if root.exists():
        if args.overwrite:
            shutil.rmtree(root)
        elif any(root.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {root}\n"
                "Use --overwrite or choose a fresh --output-dir to avoid mixing stale figures/tables."
            )
    _ensure_dir(root)

    samples = load_demo_samples(args.image_dir)
    write_run_manifest(root, args, samples)

    print("=" * 100)
    print("Unified x-attention CLIP mechanistic paper probe")
    print("NO bridge-tap relocation/transplantation is performed.")
    print(f"images: {len(samples)} from {args.image_dir}")
    print(f"prompts: {len(PROMPT_WORDS)}")
    print(f"main: {args.model}")
    print(f"old READ reference: {args.old_model}")
    print(f"output: {root}")
    print("=" * 100)

    main_model, main_processor, main_scores, main_read, paper_cache, _ = analyze_model(
        model_path=args.model,
        root=root,
        samples=samples,
        prompt_template=args.prompt_template,
        device=device,
        batch_size=args.batch_size,
        model_label="xattn_refit_wiseft",
        full_components=True,
        save_prompt_overlays=not args.compact_figures,
    )

    selections, selection_rows = select_rollout_candidates(samples, main_scores, args.good_threshold)
    _write_csv(root / "rollout_classic_b23" / "selection.csv", selection_rows)
    rollout_rows, rollout_cache, manual_validation = run_classic_b23_rollouts(
        root,
        main_model,
        main_processor,
        samples,
        _tokenize_prompts(main_processor, args.prompt_template, device),
        selections,
        device,
        paper_cache,
    )
    _write_csv(root / "rollout_classic_b23" / "metrics.csv", rollout_rows)
    _write_json(root / "rollout_classic_b23" / "manual_b23_validation.json", manual_validation)

    build_paper_panels(root, samples, paper_cache, main_scores, rollout_cache)
    build_pairwise_deltas(root, paper_cache, main_scores, rollout_cache)

    old_scores: list[dict[str, Any]] = []
    old_read: dict[tuple[str, str], dict[str, Any]] = {}
    if not args.skip_old_xattn:
        # The old model is only duplicated where the WiSE refit can change the observable.
        old_root = root / "_old_xattn"
        old_model, old_processor, old_scores, old_read, _, _ = analyze_model(
            model_path=args.old_model,
            root=old_root,
            samples=samples,
            prompt_template=args.prompt_template,
            device=device,
            batch_size=args.batch_size,
            model_label="xattn_original",
            full_components=False,
            save_prompt_overlays=not args.compact_figures,
        )
        _write_json(
            old_root / "_meta" / "scope.json",
            {
                "reason": "alpha_0_28 WiSE checkpoint changes only late READ-branch parameters",
                "included": ["read", "scores/routing"],
                "intentionally_not_recomputed": ["source", "ortho", "correction", "backbone", "token diagnostics", "classic B23 rollout"],
                "expected_changed_parameter_prefix": EXPECTED_WISE_DIFF_PREFIX,
                "expected_changed_parameter_exact": sorted(EXPECTED_WISE_DIFF_EXACT),
            },
        )
        comparison_rows = compare_read_models(
            root,
            samples,
            paper_cache,
            main_read,
            old_read,
            main_scores,
            old_scores,
        )
        _write_csv(root / "_comparison" / "read_new_vs_old.csv", comparison_rows)
        del old_model, old_processor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n[done]")
    print(f"Main output: {root}")
    if not args.skip_old_xattn:
        print(f"Old READ reference: {root / '_old_xattn'}")
        print(f"READ deltas: {root / '_comparison'}")
    print(f"Classic B23 rollout: {root / 'rollout_classic_b23'}")
    print(f"Paper-oriented panels: {root / 'paper_panels'}")
    print(f"Native pair deltas: {root / 'pairwise_deltas'}")


if __name__ == "__main__":
    main()
