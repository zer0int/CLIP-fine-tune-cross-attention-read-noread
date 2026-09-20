#!/usr/bin/env python3
"""Export a GmP CLIP checkpoint as an ordinary oaiclip full-model pickle.

This script supports either:

1. A full checkpoint/model that already contains the AnyText implant; or
2. A base GmP model plus a compact Stage-1 AnyText Stage-1 compact checkpoint; or
3. The older compact hard-``<text>`` "PIECE OF CLIP" format.

It reconstructs every GeometricLinear weight as

    weight = r * normalize(theta, dim=1)

builds the model with the *modified* ``oaiclip`` package, and saves the entire
Python model object with ``torch.save(model, output_path)``.

Examples
--------

Compact AnyText Stage-1 checkpoint, using its embedded ungmp Stage-3 base::

    python export_ungmp_oaiclip_full_model.py ^
      --model_path "my/model/stage1_complete.pt" ^
      --save_dtype fp32 ^
      --also_save_state_dict

Compact Stage-1 checkpoint, explicitly rebuilding from the original GmP base::

    python export_ungmp_oaiclip_full_model.py ^
      --model_path "my/model/stage1_complete.pt" ^
      --base_model_override "path/to/trusted_base_gmp_model.pt" ^
      --save_dtype fp32 ^
      --also_save_state_dict

The first form is the safest exact merge because the compact checkpoint records
the already-ungmp Stage-3 base used for Stage-1.  The second form performs an
actual theta/r reconstruction from the original GmP backbone and should be
equivalent because Stage 1 trained only the detachable implant.

Already merged/full checkpoint::

    python export_ungmp_oaiclip_full_model.py ^
      --model_path "my_model/hard_text_stage3_joint/best_merged_state_dict.pt"

The default output is written beside ``--model_path``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Import both packages before torch.load so full-model pickles whose classes live
# in either package can be deserialized.  The exported object itself is built
# from oaiclip classes.
try:
    import gmpclipattnamp  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name == "gmpclipattnamp":
        raise ImportError(
            "Could not import gmpclipattnamp. Run this script from the directory "
            "containing your patched gmpclipattnamp package."
        ) from exc
    raise

try:
    import oaiclip  # noqa: F401
    from oaiclip.model import build_model as build_oaiclip_model
except ModuleNotFoundError as exc:
    if exc.name in {"oaiclip", "oaiclip.model"}:
        raise ImportError(
            "Could not import oaiclip. Run this script from the directory containing "
            "your patched oaiclip package."
        ) from exc
    raise


HARD_TEXT_TOKEN_ID = 49408
NO_TEXT_TOKEN_ID = 49409
ANY_TEXT_TOKEN_ID = 49410
NULL_TEXT_TOKEN_ID = 49411
STAGE1_FORMATS = {
    "gmp_anytext_stage1_v1",
    "gmp_anytext_stage1_v2_null_controls",
    "gmp_anytext_early_branch_v3",
}


def torch_load_full(path: Path) -> Any:
    """Load tensors or full Python objects across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only=...
        return torch.load(path, map_location="cpu")


def load_checkpoint_object(path: Path) -> Any:
    """Read TorchScript, state_dict, wrapped state_dict, or full module."""
    try:
        return torch.jit.load(str(path), map_location="cpu").eval()
    except Exception:
        return torch_load_full(path)


def strip_common_prefixes(state: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        out[key] = value
    return out


def looks_like_model_state(state: Mapping[str, Any]) -> bool:
    keys = set(map(str, state.keys()))
    required_any = {
        "visual.conv1.weight",
        "visual.class_embedding",
        "token_embedding.weight",
        "text_projection",
    }
    return bool(keys & required_any) and "token_embedding.weight" in keys


def is_compact_implant_checkpoint(obj: Any) -> bool:
    return (
        isinstance(obj, Mapping)
        and isinstance(obj.get("implant_state_dict"), Mapping)
        and torch.is_tensor(obj.get("hard_text_embedding"))
    )


def checkpoint_read_attention_architecture(
    obj: Any,
    state: Mapping[str, Any],
) -> Optional[str]:
    explicit: Optional[str] = None
    if isinstance(obj, Mapping):
        value = obj.get("read_attention_architecture")
        metadata = obj.get("metadata")
        if value is None and isinstance(metadata, Mapping):
            value = metadata.get("read_attention_architecture")
        args_metadata = obj.get("args")
        if value is None and isinstance(args_metadata, Mapping):
            value = args_metadata.get("read_attention_architecture")
        if value is not None:
            explicit = str(value)
    elif isinstance(obj, nn.Module):
        value = getattr(obj, "read_attention_architecture", None)
        if value is None:
            info = getattr(obj, "_ungmp_export_info", None)
            if isinstance(info, Mapping):
                value = info.get("read_attention_architecture")
        if value is not None:
            explicit = str(value)
    if explicit is not None and explicit not in {"softmax", "sigmoid_mass", "sigmoid_all"}:
        raise ValueError(f"Unknown read-attention architecture metadata: {explicit!r}")

    reader_keys = [str(key) for key in state if "read_bridge." in str(key)]
    inferred = None
    if reader_keys:
        inferred = (
            "sigmoid_all"
            if any(key.endswith("sigmoid_patch_head_bias") for key in reader_keys)
            else (
                "sigmoid_mass"
                if any(key.endswith("sigmoid_head_bias") for key in reader_keys)
                else "softmax"
            )
        )
    if explicit is not None and inferred is not None and explicit != inferred:
        raise ValueError(
            "Read-attention metadata disagrees with checkpoint parameters: "
            f"metadata={explicit!r}, inferred={inferred!r}"
        )
    return explicit or inferred



def classify_anytext_architecture(state: Mapping[str, Any]) -> str:
    keys = tuple(map(str, state.keys()))
    has_legacy = any(k.startswith("read_implant.presence_pool.") for k in keys)
    has_final = any(
        k.startswith((
            "read_implant.source_head.",
            "read_implant.orthographic_bridge.",
            "read_implant.trust_router.",
        ))
        for k in keys
    )
    if has_legacy and has_final:
        return "mixed-invalid"
    if has_legacy:
        return "legacy"
    if has_final:
        return "final"
    return "base-or-unknown"

def extract_model_state(obj: Any, source: Path) -> Dict[str, Any]:
    """Extract a full model state_dict from common checkpoint formats."""
    if isinstance(obj, nn.Module):
        return strip_common_prefixes(obj.state_dict())

    if not isinstance(obj, Mapping):
        raise TypeError(f"Unsupported checkpoint object in {source}: {type(obj)}")

    for key in ("state_dict", "model_state_dict"):
        value = obj.get(key)
        if isinstance(value, Mapping) and looks_like_model_state(value):
            return strip_common_prefixes(value)

    # Some training wrappers store a complete model under "model".
    value = obj.get("model")
    if isinstance(value, nn.Module):
        return strip_common_prefixes(value.state_dict())
    if isinstance(value, Mapping) and looks_like_model_state(value):
        return strip_common_prefixes(value)

    if looks_like_model_state(obj):
        return strip_common_prefixes(obj)

    if "implant_state_dict" in obj:
        raise ValueError(
            f"{source} is a compact implant checkpoint, not a full model. "
            "Pass the original/full CLIP through --model_path and this file "
            "through --implant_checkpoint."
        )

    raise ValueError(
        f"Could not find a full CLIP state_dict in {source}. "
        f"Top-level keys: {list(obj)[:20]}"
    )


def extract_implant_state(
    obj: Any,
    source: Path,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """Extract the detachable implant plus dedicated control embeddings.

    New ``gmp_anytext_stage1_v2_null_controls`` checkpoints contain both the
    learned hard ``<text>`` embedding and the dedicated ``<null>`` embedding.
    Older AnyText/PIECE checkpoints contain only ``hard_text_embedding``; those
    remain exportable with a zero-initialized null embedding supplied by the
    patched ``oaiclip.build_model`` migration.
    """
    if isinstance(obj, nn.Module):
        state: Mapping[str, Any] = obj.state_dict()
    elif isinstance(obj, Mapping):
        if "implant_state_dict" in obj:
            implant = obj["implant_state_dict"]
            hard_token = obj.get("hard_text_embedding")
            null_token = obj.get("null_text_embedding")
            if not isinstance(implant, Mapping) or not torch.is_tensor(hard_token):
                raise ValueError(f"Malformed compact implant checkpoint: {source}")
            if null_token is not None and not torch.is_tensor(null_token):
                raise TypeError(
                    f"null_text_embedding in {source} is not a tensor: {type(null_token)}"
                )

            tensor_implant = {
                str(k): v.detach().cpu()
                for k, v in implant.items()
                if torch.is_tensor(v)
            }
            non_tensor = [str(k) for k, v in implant.items() if not torch.is_tensor(v)]
            if non_tensor:
                raise TypeError(
                    f"Compact implant contains non-tensor entries: {non_tensor[:20]}"
                )
            if not tensor_implant:
                raise ValueError(f"Compact implant is empty: {source}")

            checkpoint_format = str(obj.get("format", "legacy_compact"))
            if checkpoint_format in STAGE1_FORMATS:
                if checkpoint_format == "gmp_anytext_early_branch_v3":
                    required = {
                        "orthographic_bridge.out_proj.weight",
                        "source_head.patch_out.weight",
                        "trust_router.fc3.weight",
                        "ortho_tap_blocks",
                        "source_tap_blocks",
                        "read_bridge.q_proj.weight",
                        "content_pool.out_proj.weight",
                        "auto_read_scale",
                        "null_abstain_weight",
                        "glyph_bias_beta",
                        "read_probe",
                    }
                else:
                    required = {
                        "glyph_head.fc2.weight",
                        "auto_router.fc2.weight",
                        "auto_read_scale",
                        "glyph_bias_beta",
                        "read_bridge.q_proj.weight",
                        "content_pool.out_proj.weight",
                    }
                    if checkpoint_format == "gmp_anytext_stage1_v2_null_controls":
                        required.update({"null_abstain_weight", "read_probe"})
                if checkpoint_format in {"gmp_anytext_stage1_v2_null_controls", "gmp_anytext_early_branch_v3"}:
                    if not torch.is_tensor(null_token):
                        raise KeyError(
                            f"{checkpoint_format} checkpoint is missing null_text_embedding"
                        )
                missing = sorted(required.difference(tensor_implant))
                if missing:
                    raise KeyError(
                        "Stage-1 checkpoint is missing required AnyText implant keys: "
                        + ", ".join(missing)
                    )
            print(
                f"[attach] compact format={checkpoint_format}; "
                f"implant tensors={len(tensor_implant)}; "
                f"null token={'yes' if torch.is_tensor(null_token) else 'legacy-default'}"
            )
            embedded_base = obj.get("base_model_path")
            if embedded_base:
                print(f"[attach] compact embedded base: {embedded_base}")
            return (
                tensor_implant,
                hard_token.detach().cpu(),
                null_token.detach().cpu() if torch.is_tensor(null_token) else None,
            )

        wrapped = obj.get("state_dict", obj.get("model_state_dict", obj))
        if not isinstance(wrapped, Mapping):
            raise TypeError(f"Unsupported implant checkpoint in {source}: {type(wrapped)}")
        state = wrapped
    else:
        raise TypeError(f"Unsupported implant checkpoint object in {source}: {type(obj)}")

    state = strip_common_prefixes(state)
    implant = {
        key[len("read_implant."):]: value
        for key, value in state.items()
        if key.startswith("read_implant.") and torch.is_tensor(value)
    }
    hard_token = state.get("hard_text_embedding")
    null_token = state.get("null_text_embedding")
    if not implant or not torch.is_tensor(hard_token):
        raise KeyError(
            f"Could not find read_implant.* plus hard_text_embedding in {source}"
        )
    return (
        implant,
        hard_token.detach().cpu(),
        null_token.detach().cpu() if torch.is_tensor(null_token) else None,
    )


def attach_implant_to_state(
    base_state: MutableMapping[str, Any],
    implant_state: Mapping[str, torch.Tensor],
    hard_text_embedding: torch.Tensor,
    null_text_embedding: Optional[torch.Tensor],
) -> None:
    """Replace the complete detachable implant; never overwrite backbone tensors."""
    for key in list(base_state):
        if key.startswith("read_implant."):
            del base_state[key]
    for key, value in implant_state.items():
        base_state[f"read_implant.{key}"] = value.detach().cpu()
    base_state["hard_text_embedding"] = hard_text_embedding.detach().cpu().reshape(-1)
    if null_text_embedding is not None:
        base_state["null_text_embedding"] = (
            null_text_embedding.detach().cpu().reshape(-1)
        )
    else:
        # Do not inherit a stale null embedding from the base when attaching an
        # old compact implant. build_model will insert its deterministic legacy
        # default, and the exporter reports that this was a compatibility path.
        base_state.pop("null_text_embedding", None)



def assert_legacy_hard_text_implant(state: Mapping[str, Any], source: str) -> None:
    """Require the known working late read/content/presence architecture."""
    required = {
        "read_implant.read_bridge.q_proj.weight",
        "read_implant.content_pool.out_proj.weight",
        "read_implant.presence_pool.out_proj.weight",
        "read_implant.read_tap_logits",
        "read_implant.content_tap_logits",
        "read_implant.presence_tap_logits",
        "hard_text_embedding",
    }
    forbidden_prefixes = (
        "read_implant.source_head.",
        "read_implant.orthographic_bridge.",
        "read_implant.trust_router.",
    )
    forbidden_exact = {
        "read_implant.source_tap_logits",
        "read_implant.ortho_tap_logits",
    }
    missing = sorted(key for key in required if key not in state)
    leaked = sorted(
        key for key in state
        if key in forbidden_exact or any(key.startswith(prefix) for prefix in forbidden_prefixes)
    )
    if missing or leaked:
        raise KeyError(
            f"{source} is not the isolated legacy hard-text checkpoint required at "
            f"the architecture boundary. missing={missing}, leaked_final_keys={leaked[:20]}"
        )


def verify_legacy_transfer(
    legacy_state: Mapping[str, Any],
    final_state: Mapping[str, torch.Tensor],
) -> Tuple[float, str]:
    """Verify every transferable legacy read/content tensor survived migration."""
    transferable_exact = {
        "read_implant.tap_blocks",
        "read_implant.bridge_heads_config",
        "read_implant.read_tap_logits",
        "read_implant.content_tap_logits",
        "hard_text_embedding",
    }
    transferable_prefixes = (
        "read_implant.read_bridge.",
        "read_implant.content_pool.",
    )
    keys = sorted(
        key for key, value in legacy_state.items()
        if torch.is_tensor(value)
        and (key in transferable_exact or any(key.startswith(prefix) for prefix in transferable_prefixes))
    )
    missing = [key for key in keys if key not in final_state]
    if missing:
        raise RuntimeError(f"Legacy migration lost transferable keys: {missing[:20]}")

    max_diff = 0.0
    worst = ""
    for key in keys:
        expected = legacy_state[key].detach().cpu().to(dtype=final_state[key].dtype)
        actual = final_state[key].detach().cpu()
        if expected.shape != actual.shape:
            raise RuntimeError(
                f"Legacy migration shape mismatch for {key}: "
                f"expected={tuple(expected.shape)} actual={tuple(actual.shape)}"
            )
        diff = float((actual - expected).abs().max())
        if diff > max_diff:
            max_diff = diff
            worst = key
    return max_diff, worst


def assert_complete_anytext_implant(state: Mapping[str, Any], source: str) -> None:
    """Reject accidental export with only the old partial implant."""
    required = {
        "read_implant.read_bridge.q_proj.weight",
        "read_implant.content_pool.out_proj.weight",
        "read_implant.orthographic_bridge.out_proj.weight",
        "read_implant.source_head.patch_out.weight",
        "read_implant.trust_router.fc3.weight",
        "read_implant.ortho_tap_blocks",
        "read_implant.source_tap_blocks",
        "read_implant.auto_read_scale",
        "read_implant.glyph_bias_beta",
        "read_implant.null_abstain_weight",
        "read_implant.read_probe",
        "hard_text_embedding",
        "null_text_embedding",
    }
    missing = sorted(key for key in required if key not in state)
    if missing:
        raise KeyError(
            f"{source} does not contain the complete AnyText null-controls implant: "
            + ", ".join(missing)
        )


def verify_implant_roundtrip(
    expected_implant: Mapping[str, torch.Tensor],
    expected_hard_token: torch.Tensor,
    expected_null_token: Optional[torch.Tensor],
    final_state: Mapping[str, torch.Tensor],
) -> Tuple[float, str]:
    """Verify every compact implant tensor survived oaiclip construction."""
    max_diff = 0.0
    worst_key = ""
    missing = []
    shape_mismatch = []

    def compare_tensor(key: str, expected: torch.Tensor) -> None:
        nonlocal max_diff, worst_key
        actual = final_state.get(key)
        if actual is None:
            missing.append(key)
            return
        if tuple(actual.shape) != tuple(expected.shape):
            shape_mismatch.append(
                f"{key}: expected={tuple(expected.shape)} actual={tuple(actual.shape)}"
            )
            return
        diff = float(
            (actual.detach().cpu().float() - expected.detach().cpu().float()).abs().max()
        )
        if diff > max_diff:
            max_diff = diff
            worst_key = key

    for local_key, expected in expected_implant.items():
        compare_tensor(f"read_implant.{local_key}", expected)
    compare_tensor("hard_text_embedding", expected_hard_token.reshape(-1))
    if expected_null_token is not None:
        compare_tensor("null_text_embedding", expected_null_token.reshape(-1))

    if missing or shape_mismatch:
        raise RuntimeError(
            "Implant round-trip verification failed. "
            f"missing={missing[:20]} shape_mismatch={shape_mismatch[:20]}"
        )
    return max_diff, worst_key

def ungmp_state_dict(
    state: Mapping[str, Any],
    math_dtype: str = "source",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Replace every matching .theta/.r pair with an ordinary .weight tensor."""
    out: Dict[str, Any] = dict(state)
    report: Dict[str, Dict[str, float]] = {}

    theta_keys = sorted(
        key for key, value in state.items()
        if key.endswith(".theta") and torch.is_tensor(value)
    )

    for theta_key in theta_keys:
        base = theta_key[:-len("theta")]
        r_key = base + "r"
        weight_key = base + "weight"
        if r_key not in state or not torch.is_tensor(state[r_key]):
            raise KeyError(f"Found {theta_key} without matching {r_key}")
        if weight_key in state:
            raise KeyError(
                f"Checkpoint contains both GmP pair ({theta_key}, {r_key}) and {weight_key}; "
                "refusing to guess which one is authoritative."
            )

        theta = state[theta_key].detach().cpu()
        r = state[r_key].detach().cpu()
        if theta.ndim != 2:
            raise ValueError(f"Expected 2-D theta at {theta_key}, got {tuple(theta.shape)}")
        if r.numel() != theta.shape[0]:
            raise ValueError(
                f"Shape mismatch for {theta_key}/{r_key}: theta={tuple(theta.shape)}, "
                f"r={tuple(r.shape)}"
            )
        r = r.reshape(theta.shape[0], 1)

        if math_dtype == "fp32":
            weight = r.float() * F.normalize(theta.float(), p=2, dim=1)
        elif math_dtype == "source":
            # This mirrors GeometricLinear.forward() most closely when theta/r
            # were stored in fp16.
            weight = r.to(theta.dtype) * F.normalize(theta, p=2, dim=1)
        else:
            raise ValueError(f"Unknown math_dtype: {math_dtype}")

        out[weight_key] = weight.contiguous()
        del out[theta_key]
        del out[r_key]

        row_norms = weight.float().norm(dim=1)
        report[weight_key] = {
            "out_features": float(theta.shape[0]),
            "in_features": float(theta.shape[1]),
            "weight_norm_mean": float(row_norms.mean()),
            "weight_norm_min": float(row_norms.min()),
            "weight_norm_max": float(row_norms.max()),
        }

    leftovers = [key for key in out if key.endswith(".theta") or key.endswith(".r")]
    if leftovers:
        raise RuntimeError(
            "Unconverted geometric parameters remain: " + ", ".join(leftovers[:20])
        )
    return out, report


def default_output_path(model_path: Path, implant_path: Optional[Path]) -> Path:
    piece_tag = ""
    if implant_path is not None:
        parent = implant_path.parent.name.strip()
        piece_tag = f"__{parent}" if parent else "__implant"
    return model_path.with_name(
        f"{model_path.stem}{piece_tag}__ungmp_oaiclip_fullmodel.pt"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Undo GmP, attach a complete AnyText null-controls implant, and save a full ordinary oaiclip model."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help=(
            "Full/base CLIP checkpoint, or a compact Stage-1 checkpoint. "
            "For a compact checkpoint the embedded base_model_path is used automatically."
        ),
    )
    parser.add_argument(
        "--base_model_override",
        type=Path,
        default=None,
        help=(
            "When --model_path itself is a compact Stage-1 checkpoint, use this "
            "full/base model instead of the checkpoint's embedded base_model_path. "
            "Point this at the original GmP model to force a real theta/r conversion."
        ),
    )
    parser.add_argument(
        "--implant_checkpoint",
        type=Path,
        default=None,
        help="Optional compact Stage-1/legacy implant checkpoint to attach to the base model.",
    )
    parser.add_argument(
        "--expect_legacy_hard_text",
        action="store_true",
        help=(
            "DEPRECATED/forbidden. Legacy migration must occur inside final-base loading "
            "after its training seed is set; use the dedicated legacy boundary exporter."
        ),
    )
    parser.add_argument(
        "--require_anytext_stage1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require glyph/router/calibration modules in the attached/final implant. "
            "Disable only when exporting an older hard-<text> checkpoint."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output full-model pickle. Default: beside --model_path with an ungmp suffix.",
    )
    parser.add_argument(
        "--conversion_math",
        choices=("source", "fp32"),
        default="source",
        help="Use source-dtype normalization (closest to GmP forward) or fp32 conversion math.",
    )
    parser.add_argument(
        "--save_dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="Dtype of the saved oaiclip model object.",
    )
    parser.add_argument(
        "--also_save_state_dict",
        action="store_true",
        help="Also save an ordinary-weight state_dict next to the full model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_model_path = args.model_path.expanduser().resolve()
    explicit_implant_path = (
        args.implant_checkpoint.expanduser().resolve()
        if args.implant_checkpoint
        else None
    )
    base_override = (
        args.base_model_override.expanduser().resolve()
        if args.base_model_override
        else None
    )

    if not requested_model_path.is_file():
        raise FileNotFoundError(requested_model_path)
    if explicit_implant_path is not None and not explicit_implant_path.is_file():
        raise FileNotFoundError(explicit_implant_path)
    if base_override is not None and not base_override.is_file():
        raise FileNotFoundError(base_override)

    requested_obj = load_checkpoint_object(requested_model_path)
    compact_as_model = is_compact_implant_checkpoint(requested_obj)

    if compact_as_model:
        if explicit_implant_path is not None:
            raise ValueError(
                "--model_path is already a compact implant checkpoint; do not also "
                "pass --implant_checkpoint."
            )
        implant_path = requested_model_path
        embedded_base = str(requested_obj.get("base_model_path") or "").strip()
        if base_override is not None:
            model_path = base_override
            print(f"[resolve] compact checkpoint base override: {model_path}")
        elif embedded_base:
            model_path = Path(embedded_base).expanduser().resolve()
            print(f"[resolve] compact checkpoint embedded base: {model_path}")
        else:
            raise ValueError(
                f"Compact checkpoint {requested_model_path} has no base_model_path. "
                "Pass --base_model_override."
            )
        piece_obj = requested_obj
    else:
        if base_override is not None:
            raise ValueError(
                "--base_model_override is only valid when --model_path is a compact checkpoint."
            )
        model_path = requested_model_path
        implant_path = explicit_implant_path
        piece_obj = None
        del requested_obj

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Resolved base model does not exist: {model_path}. "
            "Use --base_model_override if the embedded path moved."
        )

    if args.output_path:
        output_path = args.output_path.expanduser().resolve()
    elif compact_as_model:
        output_path = requested_model_path.with_name(
            f"{requested_model_path.stem}__ungmp_oaiclip_fullmodel.pt"
        )
    else:
        output_path = default_output_path(model_path, implant_path)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (use --overwrite)")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] full/base model: {model_path}")
    base_obj = load_checkpoint_object(model_path)
    state = extract_model_state(base_obj, model_path)
    base_read_architecture = checkpoint_read_attention_architecture(base_obj, state)
    architecture_stage = classify_anytext_architecture(state)
    if args.expect_legacy_hard_text:
        raise RuntimeError(
            "--expect_legacy_hard_text is no longer permitted in the final exporter. "
            "The historical boundary requires a LEGACY ordinary checkpoint to be saved first, "
            "then migrated by final-base clip.load() after set_seed()."
        )
    if architecture_stage in {"legacy", "mixed-invalid"}:
        raise RuntimeError(
            "Final oaiclip exporter refuses a legacy/mixed checkpoint: "
            f"architecture={architecture_stage}. Do not migrate during export."
        )
    del base_obj

    expected_implant: Optional[Dict[str, torch.Tensor]] = None
    expected_hard_token: Optional[torch.Tensor] = None
    expected_null_token: Optional[torch.Tensor] = None
    implant_read_architecture: Optional[str] = None

    if implant_path is not None:
        print(f"[attach] AnyText implant: {implant_path}")
        if piece_obj is None:
            piece_obj = load_checkpoint_object(implant_path)
        implant_state, hard_token, null_token = extract_implant_state(piece_obj, implant_path)
        implant_read_architecture = checkpoint_read_attention_architecture(piece_obj, implant_state)
        expected_implant = dict(implant_state)
        expected_hard_token = hard_token.detach().cpu().reshape(-1)
        expected_null_token = (
            null_token.detach().cpu().reshape(-1) if null_token is not None else None
        )
        attach_implant_to_state(state, implant_state, hard_token, null_token)
        del piece_obj
    else:
        has_piece = (
            torch.is_tensor(state.get("hard_text_embedding"))
            and any(key.startswith("read_implant.") for key in state)
        )
        print(f"[attach] separate implant: none; implant already in model={has_piece}")

    if args.expect_legacy_hard_text:
        if args.require_anytext_stage1:
            raise ValueError(
                "--expect_legacy_hard_text requires --no-require_anytext_stage1; "
                "the whole point is to cross the architecture boundary explicitly."
            )
        assert_legacy_hard_text_implant(state, source=str(implant_path or model_path))
        print("[migration] validated isolated legacy read/content/presence implant")
    elif args.require_anytext_stage1:
        assert_complete_anytext_implant(state, source=str(implant_path or model_path))

    final_state_architecture = checkpoint_read_attention_architecture({}, state)
    selected_read_architecture = (
        implant_read_architecture
        or final_state_architecture
        or base_read_architecture
        or "softmax"
    )
    if final_state_architecture is not None and final_state_architecture != selected_read_architecture:
        raise ValueError(
            "Final attached reader parameters do not match selected architecture: "
            f"selected={selected_read_architecture!r}, inferred={final_state_architecture!r}"
        )
    print(f"[reader] architecture={selected_read_architecture}")

    print(f"[ungmp] conversion math: {args.conversion_math}")
    ordinary_state, conversion_report = ungmp_state_dict(state, args.conversion_math)
    legacy_transfer_state = dict(ordinary_state) if args.expect_legacy_hard_text else None
    n_pairs = len(conversion_report)
    if n_pairs:
        print(f"[ungmp] reconstructed {n_pairs} ordinary .weight tensors")
    else:
        print("[ungmp] source already uses ordinary weights; no theta/r pairs found")

    # build_model mutates a few bookkeeping keys, so pass a fresh dict.
    print("[build] constructing model with oaiclip.model.build_model")
    model = build_oaiclip_model(
        dict(ordinary_state),
        read_attention_architecture=selected_read_architecture,
    )
    model.eval().cpu()
    if str(model.read_attention_architecture) != selected_read_architecture:
        raise RuntimeError(
            "Built model architecture mismatch: "
            f"model={model.read_attention_architecture!r}, selected={selected_read_architecture!r}"
        )

    required_model_attrs = (
        "forward_hard",
        "forward_modes",
        "read_implant",
        "hard_text_token_id",
        "no_text_token_id",
        "any_text_token_id",
        "null_text_token_id",
        "null_text_embedding",
    )
    missing_attrs = [name for name in required_model_attrs if not hasattr(model, name)]
    if missing_attrs:
        raise RuntimeError(
            "The imported oaiclip package is not the AnyText Stage-1 version. "
            f"Missing attributes: {missing_attrs}"
        )
    if int(model.hard_text_token_id) != HARD_TEXT_TOKEN_ID:
        raise RuntimeError(f"Unexpected <text> token id: {model.hard_text_token_id}")
    if int(model.no_text_token_id) != NO_TEXT_TOKEN_ID:
        raise RuntimeError(f"Unexpected <notext> token id: {model.no_text_token_id}")
    if int(model.any_text_token_id) != ANY_TEXT_TOKEN_ID:
        raise RuntimeError(f"Unexpected <any> token id: {model.any_text_token_id}")
    if int(model.null_text_token_id) != NULL_TEXT_TOKEN_ID:
        raise RuntimeError(f"Unexpected <null> token id: {model.null_text_token_id}")

    if args.save_dtype == "fp32":
        # oaiclip.build_model follows OpenAI CLIP and initially builds/loads in
        # fp16.  Reload the original ordinary state after converting the module
        # to fp32 so --save_dtype fp32 does not merely upcast already-rounded
        # fp16 tensors.
        model.float()
        reload_state: Dict[str, torch.Tensor] = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        for key, value in ordinary_state.items():
            if key not in reload_state or not torch.is_tensor(value):
                continue
            target = reload_state[key]
            source = value.detach().cpu()
            if key == "token_embedding.weight" and source.shape[0] < target.shape[0]:
                padded = target.clone()
                padded[:source.shape[0]].copy_(source.to(dtype=target.dtype))
                source = padded
            if source.shape == target.shape:
                reload_state[key] = source.to(dtype=target.dtype)
        model.load_state_dict(reload_state, strict=True)
    else:
        model.half()

    # Keep the reserved inspection row synchronized with the dedicated token.
    if hasattr(model, "hard_text_embedding") and hasattr(model, "set_hard_text_token_embedding"):
        model.set_hard_text_token_embedding(model.hard_text_embedding.detach())
    if hasattr(model, "null_text_embedding") and hasattr(model, "set_null_text_token_embedding"):
        model.set_null_text_token_embedding(model.null_text_embedding.detach())

    model.eval().cpu()

    final_state = model.state_dict()
    bad_keys = [key for key in final_state if key.endswith(".theta") or key.endswith(".r")]
    if bad_keys:
        raise RuntimeError(f"oaiclip model still contains GmP parameters: {bad_keys[:20]}")

    # Verify every reconstructed weight survived build/load, modulo destination dtype.
    max_diff = 0.0
    worst_key = ""
    for weight_key in conversion_report:
        expected = ordinary_state[weight_key].to(dtype=final_state[weight_key].dtype)
        diff = float((final_state[weight_key].cpu() - expected.cpu()).abs().max())
        if diff > max_diff:
            max_diff = diff
            worst_key = weight_key
    print(f"[verify] max reconstructed-weight diff={max_diff:.8g} ({worst_key or 'n/a'})")

    legacy_transfer_max_diff = 0.0
    legacy_transfer_worst_key = ""
    if legacy_transfer_state is not None:
        legacy_transfer_max_diff, legacy_transfer_worst_key = verify_legacy_transfer(
            legacy_transfer_state, final_state
        )
        if any(key.startswith("read_implant.presence_pool.") for key in final_state):
            raise RuntimeError("Final model still contains legacy presence_pool after migration")
        required_final = (
            "read_implant.source_head.patch_out.weight",
            "read_implant.orthographic_bridge.out_proj.weight",
            "read_implant.trust_router.fc3.weight",
        )
        missing_final = [key for key in required_final if key not in final_state]
        if missing_final:
            raise RuntimeError(
                f"Legacy migration did not initialize final architecture keys: {missing_final}"
            )
        print(
            f"[verify] max legacy transferable diff={legacy_transfer_max_diff:.8g} "
            f"({legacy_transfer_worst_key or 'n/a'})"
        )

    implant_max_diff = 0.0
    implant_worst_key = ""
    if expected_implant is not None and expected_hard_token is not None:
        implant_max_diff, implant_worst_key = verify_implant_roundtrip(
            expected_implant, expected_hard_token, expected_null_token, final_state
        )
        print(
            f"[verify] max Stage-1 implant diff={implant_max_diff:.8g} "
            f"({implant_worst_key or 'n/a'})"
        )

    # Small metadata attributes travel with the full model pickle and do not
    # affect its state_dict or inference behavior.
    model._ungmp_export_info = {
        "requested_model_path": str(requested_model_path),
        "source_model": str(model_path),
        "implant_checkpoint": str(implant_path) if implant_path else "",
        "compact_checkpoint_as_model": bool(compact_as_model),
        "converted_gmp_layers": n_pairs,
        "conversion_math": args.conversion_math,
        "save_dtype": args.save_dtype,
        "read_attention_architecture": selected_read_architecture,
        "hard_text_token_id": HARD_TEXT_TOKEN_ID,
        "no_text_token_id": NO_TEXT_TOKEN_ID,
        "any_text_token_id": ANY_TEXT_TOKEN_ID,
        "null_text_token_id": NULL_TEXT_TOKEN_ID,
        "complete_anytext_stage1_required": bool(args.require_anytext_stage1),
        "legacy_hard_text_migration": False,
        "architecture_stage": "final",
    }

    print(f"[save] full oaiclip model object: {output_path}")
    torch.save(model, output_path)

    # Verify the resulting pickle is actually loadable as a full module.
    reloaded = torch_load_full(output_path)
    if not isinstance(reloaded, nn.Module):
        raise RuntimeError(f"Saved object reloaded as {type(reloaded)}, not nn.Module")
    reload_bad = [
        key for key in reloaded.state_dict()
        if key.endswith(".theta") or key.endswith(".r")
    ]
    if reload_bad:
        raise RuntimeError(f"Reloaded model contains GmP keys: {reload_bad[:20]}")
    print(
        f"[verify] reloaded full model: {type(reloaded).__module__}.{type(reloaded).__name__}; "
        f"parameters={sum(p.numel() for p in reloaded.parameters()):,}"
    )
    del reloaded

    if args.also_save_state_dict:
        state_path = output_path.with_name(output_path.stem + "__state_dict.pt")
        if state_path.exists() and not args.overwrite:
            raise FileExistsError(f"State-dict output already exists: {state_path}")
        print(f"[save] ordinary state_dict: {state_path}")
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata": {
                    "read_attention_architecture": selected_read_architecture,
                    "format": "ungmp_oaiclip_state_dict_v2",
                },
            },
            state_path,
        )

    report_path = output_path.with_suffix(output_path.suffix + ".export.json")
    report = {
        "output_path": str(output_path),
        "requested_model_path": str(requested_model_path),
        "source_model": str(model_path),
        "implant_checkpoint": str(implant_path) if implant_path else "",
        "compact_checkpoint_as_model": bool(compact_as_model),
        "converted_gmp_layers": n_pairs,
        "conversion_math": args.conversion_math,
        "save_dtype": args.save_dtype,
        "read_attention_architecture": selected_read_architecture,
        "max_reconstructed_weight_diff": max_diff,
        "worst_weight_key": worst_key,
        "max_stage1_implant_diff": implant_max_diff,
        "worst_implant_key": implant_worst_key,
        "max_legacy_transfer_diff": legacy_transfer_max_diff,
        "worst_legacy_transfer_key": legacy_transfer_worst_key,
        "hard_text_token_id": HARD_TEXT_TOKEN_ID,
        "no_text_token_id": NO_TEXT_TOKEN_ID,
        "any_text_token_id": ANY_TEXT_TOKEN_ID,
        "null_text_token_id": NULL_TEXT_TOKEN_ID,
        "complete_anytext_stage1_required": bool(args.require_anytext_stage1),
        "legacy_hard_text_migration": False,
        "architecture_stage": "final",
        "full_model_class": f"{type(model).__module__}.{type(model).__name__}",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "output_bytes": output_path.stat().st_size,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] export report: {report_path}")


if __name__ == "__main__":
    main()
