#!/usr/bin/env python3
"""Undo GmP while preserving the *legacy* hard-text architecture.

This exporter intentionally does **not** migrate the legacy read/content/presence
implant into the final AnyText source/orthography/trust architecture.  It exists
to reproduce the historical scattercode handoff:

    legacy GmP joint state_dict
        -> ordinary-weight legacy oaiclip full model
        -> final-base process sets its seed
        -> final gmpclipattnamp.load/build_model performs migration

The output must contain ``read_implant.presence_pool.*`` and must not contain any
``source_head``, ``orthographic_bridge`` or ``trust_router`` parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# The local legacy oaiclip package is imported lazily in main(), *after* the
# input state has been classified as pure legacy.  This prevents importing a
# final package from another working directory from silently owning the export.

HARD_TEXT_TOKEN_ID = 49408
EOT_TOKEN_ID = 49407
FINAL_PREFIXES = (
    "read_implant.source_head.",
    "read_implant.orthographic_bridge.",
    "read_implant.trust_router.",
)


def torch_load_full(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_object(path: Path) -> Any:
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
    return "token_embedding.weight" in keys and any(
        key in keys for key in ("visual.conv1.weight", "visual.class_embedding")
    )


def extract_model_state(obj: Any, source: Path) -> Dict[str, Any]:
    if isinstance(obj, nn.Module):
        return strip_common_prefixes(obj.state_dict())
    if not isinstance(obj, Mapping):
        raise TypeError(f"Unsupported checkpoint object in {source}: {type(obj)}")
    for key in ("state_dict", "model_state_dict"):
        value = obj.get(key)
        if isinstance(value, Mapping) and looks_like_model_state(value):
            return strip_common_prefixes(value)
    if looks_like_model_state(obj):
        return strip_common_prefixes(obj)
    raise ValueError(f"Could not find a full model state_dict in {source}")


def read_architecture(obj: Any, state: Mapping[str, Any]) -> str:
    explicit: Optional[str] = None
    if isinstance(obj, Mapping):
        value = obj.get("read_attention_architecture")
        metadata = obj.get("metadata")
        if value is None and isinstance(metadata, Mapping):
            value = metadata.get("read_attention_architecture")
        if value is not None:
            explicit = str(value)
    elif isinstance(obj, nn.Module):
        value = getattr(obj, "read_attention_architecture", None)
        if value is not None:
            explicit = str(value)
    inferred = (
        "sigmoid_all"
        if "read_implant.read_bridge.sigmoid_patch_head_bias" in state
        else ("sigmoid_mass" if "read_implant.read_bridge.sigmoid_head_bias" in state else "softmax")
    )
    if explicit is not None and explicit not in {"softmax", "sigmoid_mass", "sigmoid_all"}:
        raise ValueError(f"Unknown reader architecture metadata: {explicit!r}")
    if explicit is not None and explicit != inferred:
        raise ValueError(
            "Reader architecture metadata disagrees with parameters: "
            f"metadata={explicit!r}, inferred={inferred!r}"
        )
    return explicit or inferred


def classify_architecture(state: Mapping[str, Any]) -> str:
    keys = tuple(map(str, state.keys()))
    has_legacy = any(key.startswith("read_implant.presence_pool.") for key in keys)
    has_final = any(key.startswith(prefix) for key in keys for prefix in FINAL_PREFIXES)
    if has_legacy and has_final:
        return "mixed-invalid"
    if has_legacy:
        return "legacy"
    if has_final:
        return "final"
    return "base-or-unknown"


def require_legacy(state: Mapping[str, Any], where: str) -> None:
    stage = classify_architecture(state)
    if stage != "legacy":
        raise RuntimeError(
            f"{where} must be a pure legacy hard-text architecture; found {stage}. "
            "Expected read_implant.presence_pool.* and no source/ortho/trust modules."
        )


def ungmp_state_dict(
    state: Mapping[str, Any], math_dtype: str
) -> Tuple[Dict[str, Any], int]:
    out: Dict[str, Any] = dict(state)
    converted = 0
    theta_keys = sorted(
        key for key, value in state.items()
        if str(key).endswith(".theta") and torch.is_tensor(value)
    )
    for theta_key in theta_keys:
        base = theta_key[:-len("theta")]
        r_key = base + "r"
        weight_key = base + "weight"
        if r_key not in state or not torch.is_tensor(state[r_key]):
            raise KeyError(f"Found {theta_key} without matching {r_key}")
        if weight_key in state:
            raise KeyError(
                f"Checkpoint contains both ({theta_key}, {r_key}) and {weight_key}"
            )
        theta = state[theta_key].detach().cpu()
        r = state[r_key].detach().cpu().reshape(theta.shape[0], 1)
        if math_dtype == "fp32":
            weight = r.float() * F.normalize(theta.float(), p=2, dim=1)
        elif math_dtype == "source":
            weight = r.to(theta.dtype) * F.normalize(theta, p=2, dim=1)
        else:
            raise ValueError(math_dtype)
        out[weight_key] = weight.contiguous()
        del out[theta_key]
        del out[r_key]
        converted += 1
    leftovers = [key for key in out if str(key).endswith((".theta", ".r"))]
    if leftovers:
        raise RuntimeError(f"Unconverted GmP parameters remain: {leftovers[:20]}")
    return out, converted


def max_state_diff(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> Tuple[float, str]:
    worst = 0.0
    worst_key = ""
    for key, value in expected.items():
        if not torch.is_tensor(value) or key not in actual or not torch.is_tensor(actual[key]):
            continue
        if tuple(value.shape) != tuple(actual[key].shape):
            raise RuntimeError(
                f"Shape mismatch after legacy export at {key}: "
                f"{tuple(value.shape)} vs {tuple(actual[key].shape)}"
            )
        diff = float((value.float() - actual[key].detach().cpu().float()).abs().max())
        if diff > worst:
            worst, worst_key = diff, key
    return worst, worst_key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", type=Path, required=True)
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument("--conversion_math", choices=("source", "fp32"), default="source")
    p.add_argument("--save_dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--also_save_state_dict", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--preserve_legacy_hard_text", action="store_true",
        help="Required safety acknowledgement; this exporter never performs migration.",
    )
    # Accepted only so a shared JSON export schema can keep null fields.
    p.add_argument("--base_model_override", type=Path, default=None)
    p.add_argument("--implant_checkpoint", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.preserve_legacy_hard_text:
        raise RuntimeError(
            "Refusing legacy boundary export without --preserve_legacy_hard_text"
        )
    if args.base_model_override is not None or args.implant_checkpoint is not None:
        raise ValueError(
            "Legacy boundary export expects the already-merged joint model; "
            "base_model_override/implant_checkpoint must be omitted."
        )
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    obj = load_checkpoint_object(model_path)
    state = extract_model_state(obj, model_path)
    require_legacy(state, "Joint checkpoint")
    architecture = read_architecture(obj, state)
    print(f"[boundary] input architecture=legacy reader={architecture}")
    ordinary_state, n_pairs = ungmp_state_dict(state, args.conversion_math)
    require_legacy(ordinary_state, "Ungmp state")
    print(f"[ungmp] reconstructed {n_pairs} ordinary .weight tensors")

    # IMPORTANT: local legacy oaiclip builder.  No final modules exist here.
    # Import only now, after the state guard above.  The runner executes this
    # script with cwd=legacy_hard_text_pre, reproducing the historical package
    # ownership of the joint export.
    import oaiclip  # noqa: F401
    from oaiclip.model import build_model as build_legacy_oaiclip_model

    model = build_legacy_oaiclip_model(
        dict(ordinary_state),
        hard_text_token_id=HARD_TEXT_TOKEN_ID,
        eot_token_id=EOT_TOKEN_ID,
        read_attention_architecture=architecture,
    )
    model_state = model.state_dict()
    require_legacy(model_state, "Built export model")
    model.read_attention_architecture = architecture
    model._ungmp_export_info = {
        "architecture_stage": "legacy",
        "migration_performed": False,
        "preserve_legacy_hard_text": True,
        "read_attention_architecture": architecture,
        "conversion_math": args.conversion_math,
        "converted_gmp_layers": n_pairs,
        "save_dtype": args.save_dtype,
    }
    if args.save_dtype == "fp32":
        # build_model follows OpenAI CLIP and initially converts its ordinary
        # linear/conv weights to fp16.  Reload the unrounded ordinary state after
        # switching the module to fp32, matching the final exporter semantics.
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
        # Historical scattercode boundary artifact was explicitly saved fp16
        # (about 860 MB for ViT-L/14).  Preserve that serialization behavior.
        model.half()

    model.eval().cpu()
    model_state = model.state_dict()

    # Conversion/build round trip should preserve every supplied legacy tensor,
    # modulo the requested destination serialization dtype.
    expected_for_verify: Dict[str, Any] = {}
    for key, value in ordinary_state.items():
        if not torch.is_tensor(value) or key not in model_state:
            expected_for_verify[key] = value
            continue
        target = model_state[key]
        source = value.detach().cpu()
        if key == "token_embedding.weight" and source.shape[0] < target.shape[0]:
            padded = target.detach().cpu().clone()
            padded[:source.shape[0]].copy_(source.to(dtype=target.dtype))
            source = padded
        expected_for_verify[key] = source.to(dtype=target.dtype)
    diff, worst = max_state_diff(expected_for_verify, model_state)
    print(f"[verify] max ordinary-state diff={diff:.8g} ({worst or 'n/a'})")

    print(f"[save] LEGACY oaiclip model object: {output_path}")
    torch.save(model, output_path)
    reloaded = torch_load_full(output_path)
    reloaded_state = reloaded.state_dict() if isinstance(reloaded, nn.Module) else extract_model_state(reloaded, output_path)
    require_legacy(reloaded_state, "Reloaded export")
    if classify_architecture(reloaded_state) != "legacy":
        raise AssertionError("Legacy architecture did not survive serialization")
    parameter_count = sum(int(p.numel()) for p in reloaded.parameters()) if isinstance(reloaded, nn.Module) else 0
    print(
        f"[verify] reloaded architecture=legacy class={type(reloaded).__module__}.{type(reloaded).__name__} "
        f"parameters={parameter_count:,}"
    )

    if args.also_save_state_dict:
        state_path = output_path.with_name(output_path.stem + "__state_dict.pt")
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in reloaded_state.items()},
                "metadata": {
                    "format": "ungmp_legacy_oaiclip_state_dict_v1",
                    "architecture_stage": "legacy",
                    "read_attention_architecture": architecture,
                },
            },
            state_path,
        )
        print(f"[save] ordinary legacy state_dict: {state_path}")

    report = {
        "output_path": str(output_path),
        "source_model": str(model_path),
        "converted_gmp_layers": n_pairs,
        "conversion_math": args.conversion_math,
        "save_dtype": args.save_dtype,
        "max_reconstructed_weight_diff": diff,
        "worst_weight_key": worst,
        "full_model_class": f"{type(reloaded).__module__}.{type(reloaded).__name__}",
        "parameter_count": parameter_count,
        "output_bytes": output_path.stat().st_size,
        "architecture_stage": "legacy",
        "migration_performed": False,
        "read_attention_architecture": architecture,
        "presence_pool_present": any(k.startswith("read_implant.presence_pool.") for k in reloaded_state),
        "final_source_head_present": any(k.startswith("read_implant.source_head.") for k in reloaded_state),
    }
    report_path = output_path.with_suffix(output_path.suffix + ".export.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[done] export report: {report_path}")


if __name__ == "__main__":
    main()
