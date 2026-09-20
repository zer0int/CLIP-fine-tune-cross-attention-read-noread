#!/usr/bin/env python3
"""Paper-reproduction front end for the CLIP ModeMUX mechanistic analyses.

Typical use::

    python reproduce.py setup
    python reproduce.py list
    python reproduce.py status
    python reproduce.py run bridge.cross_attention
    python reproduce.py run workspace.cls_mu_causal --with-deps

The underlying scientific scripts remain in ``x_paper_reproduction/``.  This
front end supplies stable task names, dependency ordering, portable output paths,
and deterministic ObjectNet-MVT populations for the natural-image workspace probes.
"""
from __future__ import annotations

import argparse
import py_compile
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from reproduction_utils.catalog import FAMILIES, TASKS, TASK_BY_ID, Task
from reproduction_utils.version import REPRODUCTION_SCHEMA, REPRODUCTION_RELEASE
from reproduction_utils.config import (
    DEFAULT_CONFIG_PATH,
    get_key,
    load_config,
    save_config,
    set_key,
)
from reproduction_utils.objectnet_mvt import (
    build_workspace_manifest,
    dedup_csv_path as objectnet_dedup_csv_path,
    is_complete_root as objectnet_root_complete,
    missing_names as objectnet_missing_names,
)
from reproduction_utils.model_identity import (
    fingerprint_model_spec,
    guard_and_merge_identities,
)
from reproduction_utils.model_variants import (
    VARIANTS,
    get_variant,
    task_variant_ids,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PROBE_ROOT = PROJECT_ROOT / "x_paper_reproduction"
SELECTION_SCHEMA = 1


def _local_path(value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _display(value: Any) -> str:
    return "not configured" if value is None or not str(value).strip() else str(value)


def _prompt(label: str, current: Any = None) -> str | None:
    suffix = f" [{current}]" if current not in (None, "") else ""
    raw = input(f"{label}{suffix}: ").strip()
    if not raw:
        return None if current in (None, "") else str(current)
    return raw


def setup(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, allow_missing=True)
    print("\n" + "=" * 88)
    print("PAPER REPRODUCTION SETUP")
    print("=" * 88)
    print("This does not download private/local checkpoints or source images.")
    print("Press Enter to keep an existing/default value; use '-' to clear a value.\n")

    fields = [
        ("output_root", "Output root"),
        ("models.final_hf", "Final released x-attn HF model / local HF directory"),
        ("models.vanilla_spec", "Canonical vanilla OpenAI CLIP spec"),
        ("models.vanilla_hf", "Vanilla OpenAI CLIP HF model / local HF directory"),
        ("models.gmp_hf", "GmP comparison HF model / local HF directory"),
        ("models.xattn_checkpoint", "Local full x-attn/RN OpenAI-format checkpoint (.pt)"),
        ("models.gmp_checkpoint", "Local GmP OpenAI-format checkpoint (.pt), for low-level probes"),
        ("datasets.demoset_dir", "Paper demoset image directory"),
        ("datasets.misc_image_dir", "READ/null misc control image directory"),
        ("datasets.special_delivery_dir", "Controlled SPECIAL DELIVERY image directory"),
        ("datasets.objectnet_mvt_root", "Existing ObjectNet-MVT image root (Enter = reuse benchmark install / prepare on first use)"),
        ("runtime.device", "Default device"),
        ("runtime.model_cache_policy", "Managed model cache policy (keep/task/run)"),
    ]

    explicit = {
        "output_root": args.output_root,
        "models.final_hf": args.final_model,
        "models.vanilla_spec": args.vanilla_model,
        "models.vanilla_hf": args.vanilla_hf_model,
        "models.gmp_hf": args.gmp_model,
        "models.xattn_checkpoint": args.xattn_checkpoint,
        "models.gmp_checkpoint": args.gmp_checkpoint,
        "datasets.demoset_dir": args.demoset_dir,
        "datasets.misc_image_dir": args.misc_image_dir,
        "datasets.special_delivery_dir": args.special_delivery_dir,
        "datasets.objectnet_mvt_root": args.objectnet_mvt_root,
        "runtime.device": args.device,
        "runtime.model_cache_policy": args.model_cache_policy,
    }

    for key, label in fields:
        current = get_key(cfg, key)
        value = explicit.get(key)
        if value is None and not args.non_interactive:
            value = _prompt(label, current)
        if value is None:
            continue
        if value == "-":
            value = None
        set_key(cfg, key, value)

    save_config(cfg, args.config)
    print(f"\n[saved] {args.config}")
    print("Run `python reproduce.py status` to see which task families are ready.")
    return 0


def _task_output(task: Task, cfg: dict[str, Any]) -> Path | None:
    if task.output_rel is None:
        return None
    root = _local_path(cfg["output_root"])
    assert root is not None
    # Smoke runs intentionally share the real model cache but never share task
    # outputs/markers with scientific runs.  This lets us exercise the full
    # dispatcher cheaply without poisoning or being skipped by real results.
    if bool(cfg.get("_smoke", False)):
        root = root / "_smoke"
    return root / task.output_rel


def _configured_task_args(task: Task, cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("task_args", {}).get(task.id, [])
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return list(raw)
    raise ValueError(f"task_args.{task.id} must be a string or list of strings")


def _workspace_manifest_path(cfg: dict[str, Any]) -> Path:
    out_root = _local_path(cfg["output_root"])
    assert out_root is not None
    sample_size = 8 if bool(cfg.get("_smoke", False)) else int(cfg["datasets"].get("objectnet_mvt_sample_size", 480))
    if bool(cfg.get("_smoke", False)):
        out_root = out_root / "_smoke"
    return out_root / "_manifests" / f"objectnet_mvt_workspace_{sample_size}.csv"


def _read_benchmark_config() -> dict[str, Any]:
    path = PROJECT_ROOT / "benchmark_config.json"
    if not path.is_file():
        return {}
    try:
        import json
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _benchmark_mvt_candidates(cfg: dict[str, Any]) -> list[Path]:
    """Return configured/shared canonical ObjectNet-MVT locations worth reusing."""
    candidates: list[Path] = []
    explicit = _local_path(cfg["datasets"].get("objectnet_mvt_root"))
    if explicit is not None:
        candidates.append(explicit)

    benchmark_cfg = _read_benchmark_config()
    value = benchmark_cfg.get("datasets", {}).get("objectnet_mvt_root") if isinstance(benchmark_cfg.get("datasets"), dict) else None
    if value:
        candidates.append(Path(str(value)).expanduser())
    data_root = benchmark_cfg.get("data_root")
    if data_root:
        candidates.append(Path(str(data_root)).expanduser() / "ObjectNet-MVT" / "all")

    out_root = _local_path(cfg["output_root"])
    assert out_root is not None
    candidates.append(out_root / "_datasets" / "ObjectNet-MVT" / "all")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _discover_objectnet_mvt_root(cfg: dict[str, Any]) -> Path | None:
    explicit_value = cfg["datasets"].get("objectnet_mvt_root")
    explicit = _local_path(explicit_value)
    if explicit_value not in (None, ""):
        return explicit if objectnet_root_complete(explicit, PROJECT_ROOT) else None
    for candidate in _benchmark_mvt_candidates(cfg):
        if objectnet_root_complete(candidate, PROJECT_ROOT):
            return candidate
    return None


def _objectnet_install_data_root(cfg: dict[str, Any]) -> Path:
    benchmark_cfg = _read_benchmark_config()
    data_root = benchmark_cfg.get("data_root")
    if data_root:
        return Path(str(data_root)).expanduser()
    out_root = _local_path(cfg["output_root"])
    assert out_root is not None
    return out_root / "_datasets"


def _objectnet_status(cfg: dict[str, Any]) -> tuple[bool, str]:
    dedup = objectnet_dedup_csv_path(PROJECT_ROOT)
    if not dedup.is_file():
        return False, f"bundled ObjectNet-MVT deduplicated index missing: {dedup}"

    explicit_value = cfg["datasets"].get("objectnet_mvt_root")
    explicit = _local_path(explicit_value)
    if explicit_value not in (None, "") and not objectnet_root_complete(explicit, PROJECT_ROOT):
        missing = objectnet_missing_names(explicit, PROJECT_ROOT)
        return False, f"configured ObjectNet-MVT root is incomplete/missing ({len(missing):,} canonical images absent): {explicit}"

    found = _discover_objectnet_mvt_root(cfg)
    if found is not None:
        return True, f"ObjectNet-MVT={found} (canonical deduplicated 4,771-image set)"
    if bool(cfg["datasets"].get("objectnet_mvt_auto_download", True)):
        target = _objectnet_install_data_root(cfg) / "ObjectNet-MVT" / "all"
        return True, f"ObjectNet-MVT will be prepared by the benchmark installer on first use -> {target}"
    return False, "ObjectNet-MVT not found and datasets.objectnet_mvt_auto_download=false"


def _ensure_objectnet_mvt_root(cfg: dict[str, Any]) -> Path:
    explicit_value = cfg["datasets"].get("objectnet_mvt_root")
    explicit = _local_path(explicit_value)
    if explicit_value not in (None, ""):
        if objectnet_root_complete(explicit, PROJECT_ROOT):
            assert explicit is not None
            return explicit
        missing = objectnet_missing_names(explicit, PROJECT_ROOT)
        raise FileNotFoundError(
            f"Configured ObjectNet-MVT root is incomplete/missing ({len(missing):,} canonical images absent): {explicit}"
        )

    found = _discover_objectnet_mvt_root(cfg)
    if found is not None:
        return found
    if not bool(cfg["datasets"].get("objectnet_mvt_auto_download", True)):
        raise FileNotFoundError("ObjectNet-MVT not found and automatic preparation is disabled")

    try:
        from benchmark_utils.data_setup import install_objectnet_mvt
    except Exception as exc:
        raise RuntimeError("Could not import the repository's canonical ObjectNet-MVT installer") from exc

    data_root = _objectnet_install_data_root(cfg)
    print(f"[dataset] preparing canonical deduplicated ObjectNet-MVT via benchmark installer -> {data_root}")
    image_root = install_objectnet_mvt(PROJECT_ROOT, data_root)
    if not objectnet_root_complete(image_root, PROJECT_ROOT):
        missing = objectnet_missing_names(image_root, PROJECT_ROOT)
        raise RuntimeError(
            f"ObjectNet-MVT installer returned an incomplete root ({len(missing):,} canonical images absent): {image_root}"
        )
    return image_root


def _materialize_workspace_manifest(cfg: dict[str, Any]) -> Path:
    image_root = _ensure_objectnet_mvt_root(cfg)
    sample_size = 8 if bool(cfg.get("_smoke", False)) else int(cfg["datasets"].get("objectnet_mvt_sample_size", 480))
    seed = int(cfg["datasets"].get("objectnet_mvt_seed", 20260915))
    out = _workspace_manifest_path(cfg)
    return build_workspace_manifest(
        project_root=PROJECT_ROOT,
        image_root=image_root,
        output_path=out,
        sample_size=sample_size,
        seed=seed,
    )



def _checkpoint_cache_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _checkpoint_state(path: Path) -> dict[str, Any]:
    import torch
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a state_dict mapping in {path}, got {type(obj).__name__}")
    # A few historical files wrap the actual mapping one level deep.
    for key in ("state_dict", "model_state_dict", "model"):
        value = obj.get(key)
        if isinstance(value, dict) and value:
            obj = value
            break
    return obj


def _checkpoint_family_from_keys(state: dict[str, Any]) -> tuple[str, list[str]]:
    """Classify the *runtime architecture* represented by a native state dict.

    Training provenance is intentionally irrelevant.  A GmP-trained checkpoint
    with effective ``.weight`` tensors is vanilla CLIP.  Conversely, a state
    dict is not considered a full x-attn model merely because it contains one
    custom key: the trained RN and the complete final bridge signature must be
    present.  This prevents partial/random custom architectures from passing a
    filename-level cache check.
    """
    keys = set(map(str, state))
    rn_keys = {
        "visual.read_null_token",
        "visual.read_null_insert_block_config",
    }
    control_roots = {"hard_text_embedding", "null_text_embedding"}
    bridge_required = {
        "read_implant.read_tap_logits",
        "read_implant.content_tap_logits",
        "read_implant.ortho_tap_logits",
        "read_implant.source_tap_logits",
        "read_implant.read_bridge.q_proj.weight",
        "read_implant.read_bridge.k_proj.weight",
        "read_implant.read_bridge.v_proj.weight",
        "read_implant.read_bridge.out_proj.weight",
        "read_implant.content_pool.query",
        "read_implant.content_pool.k_proj.weight",
        "read_implant.content_pool.v_proj.weight",
        "read_implant.content_pool.out_proj.weight",
        "read_implant.orthographic_bridge.q_proj.weight",
        "read_implant.source_head.patch_out.weight",
        "read_implant.trust_router.fc1.weight",
    }
    custom = sorted(
        k for k in keys
        if k in rn_keys or k in control_roots or k.startswith("read_implant.")
    )
    has_rn = rn_keys.issubset(keys)
    any_rn = bool(rn_keys & keys)
    any_bridge = any(k.startswith("read_implant.") for k in keys)
    any_control = bool(control_roots & keys)

    if not custom:
        return "vanilla", custom
    if has_rn and not any_bridge and not any_control and set(custom) == rn_keys:
        return "rn_only", custom
    if has_rn and control_roots.issubset(keys) and bridge_required.issubset(keys):
        return "xattn_full", custom
    if any_bridge and not any_control:
        return "correction_or_partial", custom
    if any_rn or any_bridge or any_control:
        return "partial_custom", custom
    return "unknown", custom


def _validate_state_family(state: dict[str, Any], expected: str) -> tuple[bool, str]:
    family, custom = _checkpoint_family_from_keys(state)
    core = {
        "visual.conv1.weight",
        "visual.class_embedding",
        "visual.positional_embedding",
        "token_embedding.weight",
        "positional_embedding",
    }
    missing_core = sorted(core - set(map(str, state)))
    if missing_core:
        return False, f"missing ordinary CLIP core keys: {missing_core}"
    if family != expected:
        return False, (
            f"architecture family is {family!r}, expected {expected!r}; "
            f"custom-key examples={custom[:10]}"
        )
    return True, f"family={family}, keys={len(state):,}, custom_keys={len(custom):,}"


def _validate_checkpoint_family(path: Path, expected: str) -> tuple[bool, str]:
    return _validate_state_family(_checkpoint_state(path), expected)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _state_keyset_sha256(state: dict[str, Any]) -> str:
    import hashlib
    payload = "\n".join(sorted(map(str, state))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    import hashlib
    import torch
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor, got {type(tensor).__name__}")
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("ascii"))
    h.update(str(tuple(t.shape)).encode("ascii"))
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def _model_cache_dir(cfg: dict[str, Any]) -> Path:
    out_root = _local_path(cfg["output_root"])
    assert out_root is not None
    return out_root / "_models"


def _variant_cache_path(cfg: dict[str, Any], variant_id: str) -> Path:
    from reproduction_utils.model_variants import get_variant
    variant = get_variant(variant_id)
    if variant.cache_filename is None:
        raise ValueError(f"Variant {variant_id!r} is not a persisted cache variant")
    return _model_cache_dir(cfg) / variant.cache_filename


def _write_checkpoint_cache_meta(
    path: Path,
    *,
    variant_id: str,
    expected_family: str,
    source_spec: str,
    verified_from_source: bool,
    conversion_info: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
) -> Path:
    import json
    state = _checkpoint_state(path)
    ok, detail = _validate_state_family(state, expected_family)
    if not ok:
        raise RuntimeError(f"Refusing to stamp invalid model cache {path}: {detail}")
    meta = {
        "schema_version": 2,
        "managed_by_reproduce": True,
        "variant_id": variant_id,
        "role": variant_id,  # compatibility alias; variant_id is authoritative
        "expected_runtime_family": expected_family,
        "source_spec": source_spec,
        "artifact_sha256": _sha256_file(path),
        "state_keyset_sha256": _state_keyset_sha256(state),
        "verified_from_source_this_run": bool(verified_from_source),
        "validation": detail,
    }
    if conversion_info:
        meta["conversion"] = conversion_info
    if lineage:
        meta["lineage"] = lineage
    out = _checkpoint_cache_meta_path(path)
    out.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _cache_sidecar_matches(
    path: Path,
    *,
    variant_id: str,
    expected_family: str,
    source_spec: str,
) -> bool:
    import json
    ok, detail = _validate_checkpoint_family(path, expected_family)
    if not ok:
        print(f"[model-cache] rejecting stale {variant_id} cache: {detail}")
        return False
    meta_path = _checkpoint_cache_meta_path(path)
    if not meta_path.is_file():
        print(f"[model-cache] {variant_id}: cache has no v14 provenance sidecar; rebuilding once")
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[model-cache] rejecting {variant_id} cache with unreadable metadata: {exc}")
        return False
    checks = {
        "variant_id": (meta.get("variant_id"), variant_id),
        "expected_runtime_family": (meta.get("expected_runtime_family"), expected_family),
        "source_spec": (meta.get("source_spec"), source_spec),
        "artifact_sha256": (meta.get("artifact_sha256"), _sha256_file(path)),
        "state_keyset_sha256": (meta.get("state_keyset_sha256"), _state_keyset_sha256(_checkpoint_state(path))),
    }
    bad = [f"{name}: recorded={got!r} current={want!r}" for name, (got, want) in checks.items() if got != want]
    if bad:
        print(f"[model-cache] rejecting {variant_id} cache because its provenance mismatches:")
        for line in bad:
            print(f"  - {line}")
        return False
    return True


def _try_migrate_v13_cache(
    cfg: dict[str, Any],
    *,
    legacy_name: str,
    new_path: Path,
    variant_id: str,
    expected_family: str,
    source_spec: str,
) -> bool:
    """Move a verified v13 managed cache to its explicit v14 variant name."""
    import json
    old = _model_cache_dir(cfg) / legacy_name
    if new_path.exists() or not old.is_file():
        return False
    old_meta = _checkpoint_cache_meta_path(old)
    ok, detail = _validate_checkpoint_family(old, expected_family)
    if not ok:
        print(f"[model-cache] legacy {old.name} is not {variant_id}: {detail}")
        return False
    if not old_meta.is_file():
        print(f"[model-cache] legacy {old.name} has no provenance sidecar; not adopting it")
        return False
    try:
        meta = json.loads(old_meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("source_spec") != source_spec or meta.get("artifact_sha256") != _sha256_file(old):
        print(f"[model-cache] legacy {old.name} provenance does not match canonical source; rebuilding")
        return False
    new_path.parent.mkdir(parents=True, exist_ok=True)
    old.replace(new_path)
    old_meta.unlink(missing_ok=True)
    _write_checkpoint_cache_meta(
        new_path,
        variant_id=variant_id,
        expected_family=expected_family,
        source_spec=source_spec,
        verified_from_source=bool(meta.get("verified_from_source_this_run", False)),
        conversion_info={"migrated_from_v13": legacy_name, **(meta.get("conversion") or {})},
    )
    print(f"[model-cache] migrated {legacy_name} -> {new_path.name}")
    return True


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _xattn_materialized_path(cfg: dict[str, Any]) -> Path:
    return _variant_cache_path(cfg, "xattn_full_trained")


def _ensure_xattn_checkpoint(cfg: dict[str, Any]) -> Path:
    """Return the canonical *full trained* x-attn/RN checkpoint."""
    configured = get_key(cfg, "models.xattn_checkpoint")
    if configured is not None and str(configured).strip():
        path = _local_path(configured)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"models.xattn_checkpoint={path}")
        ok, detail = _validate_checkpoint_family(path, "xattn_full")
        if not ok:
            raise RuntimeError(
                f"Configured models.xattn_checkpoint is not a complete trained x-attn checkpoint: {path}\n  {detail}"
            )
        return path

    spec = get_key(cfg, "models.final_hf")
    if spec is None or not str(spec).strip():
        raise FileNotFoundError("Neither models.xattn_checkpoint nor models.final_hf is configured")
    spec = str(spec)
    cached = _xattn_materialized_path(cfg)
    _try_migrate_v13_cache(
        cfg, legacy_name="final_xattn_openai_state_dict.pt", new_path=cached,
        variant_id="xattn_full_trained", expected_family="xattn_full", source_spec=spec,
    )
    if cached.is_file() and _cache_sidecar_matches(
        cached, variant_id="xattn_full_trained", expected_family="xattn_full", source_spec=spec
    ):
        return cached

    print(f"[model] reconstructing full trained x-attn from {spec} -> {cached}")
    try:
        import attnclip_mechinterp_xattn as xclip
        from utils_clip_loader import load_openai_clip_anything
    except Exception as exc:
        raise RuntimeError("Could not import the bundled HF->native x-attn loader") from exc
    model, _preprocess, info = load_openai_clip_anything(xclip, spec, device="cpu", jit=False, strict=True)
    state = dict(model.state_dict())
    del model
    ok, detail = _validate_state_family(state, "xattn_full")
    if not ok:
        raise RuntimeError(f"Reconstructed final x-attn state failed validation: {detail}")
    _atomic_torch_save(state, cached)
    _write_checkpoint_cache_meta(
        cached,
        variant_id="xattn_full_trained",
        expected_family="xattn_full",
        source_spec=spec,
        verified_from_source=True,
        conversion_info={
            "source_kind": getattr(info, "source_kind", None),
            "detected_format": getattr(info, "detected_format", None),
            "model_family": getattr(info, "model_family", None),
        },
    )
    return cached


def _materialize_gmp_weights(state: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Convert theta/r pairs only when an old checkpoint actually contains them."""
    import torch
    import torch.nn.functional as F

    out = dict(state)
    converted = 0
    theta_keys = sorted(k for k in state if str(k).endswith(".theta"))
    for theta_key in theta_keys:
        base = theta_key[:-len("theta")]
        radius_key = base + "r"
        weight_key = base + "weight"
        if radius_key not in state:
            raise KeyError(f"Found {theta_key} without matching {radius_key}")
        if weight_key in state:
            raise KeyError(f"Checkpoint contains both {weight_key} and ({theta_key}, {radius_key})")
        theta = state[theta_key]
        radius = state[radius_key]
        if not torch.is_tensor(theta) or not torch.is_tensor(radius):
            raise TypeError(f"Non-tensor GmP pair: {theta_key}, {radius_key}")
        if theta.ndim != 2 or radius.numel() != theta.shape[0]:
            raise ValueError(f"Invalid GmP pair: {theta_key}={tuple(theta.shape)}, {radius_key}={tuple(radius.shape)}")
        out[weight_key] = (radius.reshape(-1, 1).to(theta.dtype) * F.normalize(theta, p=2, dim=1)).contiguous()
        del out[theta_key]
        del out[radius_key]
        converted += 1
    leftovers = [k for k in out if str(k).endswith((".theta", ".r"))]
    if leftovers:
        raise RuntimeError(f"Unconverted GmP tensors remain: {leftovers[:20]}")
    return out, converted


def _canonicalize_cached_gmp_checkpoint(path: Path) -> Path:
    import torch
    state = _checkpoint_state(path)
    if not any(str(k).endswith((".theta", ".r")) for k in state):
        return path
    state, converted = _materialize_gmp_weights(state)
    _atomic_torch_save(state, path)
    _checkpoint_cache_meta_path(path).unlink(missing_ok=True)
    print(f"[model] upgraded legacy theta/r checkpoint to ordinary weights: {converted} matrices -> {path}")
    return path


def _gmp_materialized_path(cfg: dict[str, Any]) -> Path:
    return _variant_cache_path(cfg, "gmp_trained_vanilla")


def _ensure_gmp_checkpoint(cfg: dict[str, Any]) -> Path:
    """Return GmP-trained weights as an ordinary vanilla-CLIP state dict."""
    configured = get_key(cfg, "models.gmp_checkpoint")
    if configured is not None and str(configured).strip():
        path = _local_path(configured)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"models.gmp_checkpoint={path}")
        path = _canonicalize_cached_gmp_checkpoint(path)
        ok, detail = _validate_checkpoint_family(path, "vanilla")
        if not ok:
            raise RuntimeError(
                "Configured models.gmp_checkpoint is not ordinary vanilla CLIP.\n"
                f"  path: {path}\n  {detail}\n"
                "GmP is training provenance, not a runtime architecture."
            )
        return path

    spec = get_key(cfg, "models.gmp_hf")
    if spec is None or not str(spec).strip():
        raise FileNotFoundError("Neither models.gmp_checkpoint nor models.gmp_hf is configured")
    spec = str(spec)
    cached = _gmp_materialized_path(cfg)
    _try_migrate_v13_cache(
        cfg, legacy_name="gmp_openai_state_dict.pt", new_path=cached,
        variant_id="gmp_trained_vanilla", expected_family="vanilla", source_spec=spec,
    )
    if cached.is_file():
        cached = _canonicalize_cached_gmp_checkpoint(cached)
        if _cache_sidecar_matches(
            cached, variant_id="gmp_trained_vanilla", expected_family="vanilla", source_spec=spec
        ):
            return cached

    print(f"[model] resolving GmP-trained weights as ordinary CLIP: {spec} -> {cached}")
    try:
        from utils_clip_loader.clip_anything_to_openai import resolve_to_openai_state_dict
    except Exception as exc:
        raise RuntimeError("Could not import the generic HF->OpenAI CLIP converter") from exc
    state, info = resolve_to_openai_state_dict(spec)
    state, converted = _materialize_gmp_weights(state)
    ok, detail = _validate_state_family(state, "vanilla")
    if not ok:
        raise RuntimeError(f"Canonical GmP source converted to a non-vanilla state: {detail}")
    _atomic_torch_save(state, cached)
    _write_checkpoint_cache_meta(
        cached,
        variant_id="gmp_trained_vanilla",
        expected_family="vanilla",
        source_spec=spec,
        verified_from_source=True,
        conversion_info={
            "source_kind": getattr(info, "source_kind", None),
            "detected_format": getattr(info, "detected_format", None),
            "model_family": getattr(info, "model_family", None),
            "legacy_theta_r_pairs_materialized": converted,
        },
    )
    return cached


def _oai_rn_materialized_path(cfg: dict[str, Any]) -> Path:
    return _variant_cache_path(cfg, "oai_vanilla_rn_from_xattn")


def _ensure_oai_rn_variant(cfg: dict[str, Any]) -> Path:
    """Build vanilla OAI CLIP + the exact trained RN token, with no bridge at all."""
    import json
    vanilla_spec = str(get_key(cfg, "models.vanilla_spec"))
    donor = _ensure_xattn_checkpoint(cfg)
    donor_state = _checkpoint_state(donor)
    donor_rn = donor_state.get("visual.read_null_token")
    donor_insert = donor_state.get("visual.read_null_insert_block_config")
    if donor_rn is None or donor_insert is None:
        raise RuntimeError("Full x-attn donor is missing trained RN token/config")
    donor_rn_sha = _tensor_sha256(donor_rn)
    source_spec = f"{vanilla_spec} + RN@{_sha256_file(donor)}:{donor_rn_sha}"
    cached = _oai_rn_materialized_path(cfg)

    if cached.is_file() and _cache_sidecar_matches(
        cached,
        variant_id="oai_vanilla_rn_from_xattn",
        expected_family="rn_only",
        source_spec=source_spec,
    ):
        meta = json.loads(_checkpoint_cache_meta_path(cached).read_text(encoding="utf-8"))
        if (meta.get("lineage") or {}).get("donor_rn_sha256") == donor_rn_sha:
            return cached
        print("[model-cache] RN-only variant donor token digest changed; rebuilding")

    print(f"[model] building OAI vanilla + exact trained RN (NO bridge): {vanilla_spec} -> {cached}")
    try:
        import attnclip_mechinterp_sae as saeclip
    except Exception as exc:
        raise RuntimeError("Could not import attnclip_mechinterp_sae to build the vanilla RN receiver") from exc
    vanilla_model, _ = saeclip.load(vanilla_spec, device="cpu", jit=False)
    state = dict(vanilla_model.state_dict())
    del vanilla_model
    # The only custom tensors permitted in this reusable variant are the exact
    # donor RN token and its insertion-block config.  No bridge is instantiated
    # or saved, so random bridge parameters are impossible by construction.
    state["visual.read_null_token"] = donor_rn.detach().cpu().clone()
    state["visual.read_null_insert_block_config"] = donor_insert.detach().cpu().clone()
    ok, detail = _validate_state_family(state, "rn_only")
    if not ok:
        raise RuntimeError(f"Constructed vanilla+RN variant failed validation: {detail}")
    _atomic_torch_save(state, cached)
    _write_checkpoint_cache_meta(
        cached,
        variant_id="oai_vanilla_rn_from_xattn",
        expected_family="rn_only",
        source_spec=source_spec,
        verified_from_source=True,
        conversion_info={"construction": "OAI vanilla state + exact donor RN token/config; bridge absent"},
        lineage={
            "base_vanilla_spec": vanilla_spec,
            "donor_checkpoint": str(donor),
            "donor_checkpoint_sha256": _sha256_file(donor),
            "donor_rn_sha256": donor_rn_sha,
            "bridge_policy": "absent",
        },
    )
    return cached


def _cleanup_managed_model_caches(cfg: dict[str, Any]) -> list[Path]:
    """Delete only model files explicitly created and stamped by reproduce.py."""
    import json
    root = _model_cache_dir(cfg)
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for meta_path in sorted(root.glob("*.pt.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not meta.get("managed_by_reproduce", False):
            continue
        model_path = Path(str(meta_path)[:-len(".meta.json")])
        if model_path.is_file():
            model_path.unlink()
            removed.append(model_path)
        meta_path.unlink(missing_ok=True)
    return removed



def _conv1_gpic_models(cfg: dict[str, Any]) -> list[str]:
    """Ordered GPIC model set, with the legacy single-model override preserved."""
    conv = cfg.get("conv1", {})
    legacy = conv.get("gpic_model")
    raw = [legacy] if legacy is not None and str(legacy).strip() else conv.get("gpic_models")
    if not raw:
        raw = [
            get_key(cfg, "models.final_hf"),
            get_key(cfg, "models.vanilla_hf") or "openai/clip-vit-large-patch14",
        ]
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if value is None:
            continue
        model = str(value).strip()
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    return out


def _conv1_gpic_output_name(model: str) -> str:
    import re
    value = str(model).strip().replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not value:
        raise ValueError(f"Could not derive GPIC output name from {model!r}")
    return value

def _requirement_status(key: str, cfg: dict[str, Any]) -> tuple[bool, str]:
    if key == "assets.rn_vocab":
        vocab = PROBE_ROOT / "vocab_deduped.txt"
        return (vocab.is_file(), f"RN manifold vocabulary={vocab}" if vocab.is_file() else f"missing required RN manifold vocabulary: {vocab}")
    if key == "assets.single_image_demo":
        demo_root = _local_path(get_key(cfg, "datasets.demoset_dir"))
        image = (demo_root / "bottle_shower.png") if demo_root is not None else None
        return (bool(image and image.is_file()), f"single-image demo={image}" if image and image.is_file() else f"missing required single-image demo: {image}")
    if key == "datasets.objectnet_mvt":
        return _objectnet_status(cfg)
    if key == "models.conv1_gpic_models":
        models = _conv1_gpic_models(cfg)
        if not models:
            return False, "conv1.gpic_models contains no usable model ids"
        return True, "Conv1 GPIC models=" + ", ".join(models)
    if key == "datasets.conv1_image_dir":
        value = get_key(cfg, "conv1.image_dir")
        p = _local_path(value) if value is not None else None
        return (bool(p and p.is_dir()), f"Conv1 image_dir={p}" if p and p.is_dir() else f"missing Conv1 image_dir: {p}")
    if key == "models.oai_rn_variant":
        cached = _oai_rn_materialized_path(cfg)
        if cached.is_file():
            return True, f"vanilla+trained-RN variant={cached}"
        if get_key(cfg, "models.vanilla_spec") and (get_key(cfg, "models.xattn_checkpoint") or get_key(cfg, "models.final_hf")):
            return True, "vanilla+trained-RN variant will materialize from OAI vanilla + canonical x-attn donor"
        return False, "cannot build vanilla+trained-RN variant: vanilla or x-attn donor not configured"
    if key == "models.gmp_checkpoint":
        configured = get_key(cfg, key)
        if configured is not None and str(configured).strip():
            p = _local_path(configured)
            return (bool(p and p.is_file()), f"{key}={p}")
        cached = _gmp_materialized_path(cfg)
        if cached.is_file():
            return True, f"materialized GmP={cached}"
        spec = get_key(cfg, "models.gmp_hf")
        if spec is not None and str(spec).strip():
            return True, f"GmP will materialize from {spec}"
        return False, "models.gmp_checkpoint/models.gmp_hf not configured"
    if key == "models.xattn_checkpoint":
        configured = get_key(cfg, key)
        if configured is not None and str(configured).strip():
            p = _local_path(configured)
            return (bool(p and p.is_file()), f"{key}={p}")
        cached = _xattn_materialized_path(cfg)
        if cached.is_file():
            return True, f"reconstructed x-attn={cached}"
        spec = get_key(cfg, "models.final_hf")
        if spec is not None and str(spec).strip():
            return True, f"x-attn will reconstruct from {spec}"
        return False, "models.xattn_checkpoint/models.final_hf not configured"
    value = get_key(cfg, key)
    if value is None or not str(value).strip():
        return False, f"{key} not configured"
    if key in {"datasets.demoset_dir", "datasets.misc_image_dir", "datasets.special_delivery_dir", "datasets.special_natural_dir", "datasets.visualtextual_dir"}:
        p = _local_path(value)
        return (bool(p and p.is_dir()), f"{key}={p}")
    return True, f"{key}={value}"


def _task_ready(task: Task, cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    problems = []
    script = PROBE_ROOT / task.script
    if not script.is_file():
        problems.append(f"missing script {script}")
    for req in task.requirements:
        ok, note = _requirement_status(req, cfg)
        if not ok:
            problems.append(note)
    return not problems, problems


def _task_complete(task: Task, cfg: dict[str, Any]) -> bool:
    out = _task_output(task, cfg)
    if out is None or not out.exists():
        return False
    if task.id == "conv1.gpic_manifold":
        import json
        batch = out / "batch_summary.json"
        if not batch.is_file():
            return False
        try:
            payload = json.loads(batch.read_text(encoding="utf-8"))
        except Exception:
            return False
        models = _conv1_gpic_models(cfg)
        if payload.get("models") != models:
            return False
        return all((out / _conv1_gpic_output_name(model) / "summary.json").is_file() for model in models)
    if task.markers:
        return all((out / marker).exists() for marker in task.markers)
    if out.is_dir():
        return any(out.iterdir())
    return out.is_file()


def _load_task_selection(path: Path) -> list[Task]:
    """Load a selection-only JSON file without touching reproduction_config.json.

    Selection files deliberately cannot contain model/dataset/output settings. They
    contain exact task ids only, so the user's configured reproduction environment
    remains authoritative.
    """
    import json

    path = Path(path).expanduser()
    if not path.is_file():
        raise SystemExit(f"Selection file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not parse selection JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"Selection file must contain a JSON object: {path}")

    allowed = {"schema", "name", "tasks"}
    extra = sorted(set(raw) - allowed)
    if extra:
        raise SystemExit(
            "Selection files are task-only and cannot override the main reproduction config. "
            f"Unexpected key(s): {', '.join(extra)}"
        )
    schema = raw.get("schema", SELECTION_SCHEMA)
    if schema != SELECTION_SCHEMA:
        raise SystemExit(
            f"Unsupported selection schema {schema!r}; expected {SELECTION_SCHEMA}"
        )
    values = raw.get("tasks")
    if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
        raise SystemExit('Selection JSON requires a "tasks" array of exact task-id strings.')

    chosen: list[Task] = []
    seen: set[str] = set()
    for value in values:
        task_id = value.strip()
        if not task_id:
            continue
        if task_id not in TASK_BY_ID:
            raise SystemExit(f"Unknown task id in selection file {path}: {task_id}")
        if task_id not in seen:
            seen.add(task_id)
            chosen.append(TASK_BY_ID[task_id])
    if not chosen:
        raise SystemExit(f"Selection file contains no tasks: {path}")
    return chosen


def _selected_tasks_from_args(
    args: argparse.Namespace,
    *,
    default_selectors: list[str],
) -> list[Task]:
    selectors = list(getattr(args, "selectors", None) or [])
    selection = getattr(args, "selection", None)
    if selection is not None:
        if selectors:
            raise SystemExit("Use either positional selectors or --selection, not both.")
        return _load_task_selection(selection)
    if not selectors:
        selectors = list(default_selectors)
    if not selectors:
        raise SystemExit("No tasks selected. Pass task selectors, `all`, or --selection <file.json>.")
    return _resolve_selectors(selectors, include_controls=bool(getattr(args, "all", False)))


def _selector_help_text(*, include_task_ids: bool = True) -> str:
    lines = [
        "Legal selectors:",
        "  paper       canonical paper-facing analyses + cached figure tasks",
        "  all         every automatically runnable analysis/control/figure task",
        "              (manual audit utilities are excluded; select `audit` explicitly)",
        "  families    " + ", ".join(FAMILIES),
        "  prefixes    e.g. bridge.read_null, workspace.register_geometry",
    ]
    if include_task_ids:
        lines.append("  exact task IDs:")
        lines.extend(f"    {task.id}" for task in TASKS)
    return "\n".join(lines)


def _resolve_selectors(selectors: list[str], *, include_controls: bool) -> list[Task]:
    if not selectors:
        selectors = ["paper"]
    chosen: list[Task] = []
    seen: set[str] = set()
    for selector in selectors:
        if selector == "all":
            # `run all` must be genuinely unattended. Provenance/audit utilities that
            # require task-specific arguments remain available via explicit selection.
            pool = [t for t in TASKS if not t.manual_args and t.tier != "utility"]
        elif selector == "paper":
            if include_controls:
                pool = [t for t in TASKS if not t.manual_args and t.tier != "utility"]
            else:
                pool = [t for t in TASKS if t.tier in {"canonical", "figure"}]
        elif selector in FAMILIES:
            pool = [t for t in TASKS if t.family == selector and (include_controls or t.tier in {"canonical", "figure"})]
        elif selector in TASK_BY_ID:
            pool = [TASK_BY_ID[selector]]
        else:
            matches = [t for t in TASKS if t.id.startswith(selector + ".")]
            if not matches:
                raise SystemExit(f"Unknown task/family selector: {selector}\n\n{_selector_help_text()}")
            pool = [t for t in matches if include_controls or t.tier in {"canonical", "figure"}]
        for task in pool:
            if task.id not in seen:
                seen.add(task.id)
                chosen.append(task)
    return chosen


def _with_dependencies(tasks: Iterable[Task]) -> list[Task]:
    ordered: list[Task] = []
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(task: Task) -> None:
        if task.id in done:
            return
        if task.id in visiting:
            raise RuntimeError(f"Dependency cycle at {task.id}")
        visiting.add(task.id)
        for dep in task.deps:
            visit(TASK_BY_ID[dep])
        visiting.remove(task.id)
        done.add(task.id)
        ordered.append(task)

    for task in tasks:
        visit(task)
    return ordered


def _variant_summary(task: Task) -> str:
    ids = task_variant_ids(task.id)
    return ", ".join(ids) if ids else "—"


def _print_variant_plan(task: Task) -> None:
    ids = task_variant_ids(task.id)
    if not ids:
        return
    print(f"[models] {task.id}:")
    for variant_id in ids:
        v = get_variant(variant_id)
        persistence = v.persistence
        cache = f" cache={v.cache_filename}" if v.cache_filename else ""
        print(
            f"  - {variant_id}: family={v.runtime_family} persistence={persistence}"
            f" bridge={v.bridge_policy} rn={v.rn_policy}{cache}"
        )


def list_tasks(args: argparse.Namespace) -> int:
    # `list` is discovery-oriented, so no selector means show every auto-runnable task.
    tasks = _selected_tasks_from_args(args, default_selectors=["all"])
    print(f"{'TASK':43} {'TIER':10} {'KIND':6} TITLE")
    print("-" * 104)
    for t in tasks:
        print(f"{t.id:43} {t.tier:10} {t.kind:6} {t.title}")
        if getattr(args, "models", False):
            print(f"    models: {_variant_summary(t)}")
    print("\n" + _selector_help_text())
    return 0


def _load_command_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load config and apply command-line overrides without mutating the JSON file."""
    cfg = load_config(args.config)
    output_root = getattr(args, "output_root", None)
    if output_root is not None:
        cfg["output_root"] = str(output_root)
    policy = getattr(args, "model_cache_policy", None)
    if policy is not None:
        cfg.setdefault("runtime", {})["model_cache_policy"] = str(policy)
    return cfg


def status(args: argparse.Namespace) -> int:
    cfg = _load_command_config(args)
    tasks = _selected_tasks_from_args(args, default_selectors=["all"])
    print(f"Config: {args.config.resolve()}")
    print(f"Output root: {_local_path(cfg['output_root'])}\n")
    print(f"{'TASK':43} {'STATE':12} {'TIER':10} NOTE")
    print("-" * 112)
    blocked = 0
    for t in tasks:
        complete = _task_complete(t, cfg)
        ready, problems = _task_ready(t, cfg)
        dep_missing = [d for d in t.deps if not _task_complete(TASK_BY_ID[d], cfg)]
        if complete:
            state, note = "CACHED", str(_task_output(t, cfg))
        elif not ready:
            state, note = "BLOCKED", "; ".join(problems[:2])
            blocked += 1
        elif dep_missing:
            state, note = "WAITING", "deps: " + ", ".join(dep_missing)
        elif t.manual_args:
            state, note = "MANUAL", "pass task-specific args after --"
        else:
            state, note = "READY", str(_task_output(t, cfg))
        print(f"{t.id:43} {state:12} {t.tier:10} {note}")
    return 1 if blocked and args.strict else 0


def _common(cfg: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    final = str(cfg["models"]["final_hf"])
    vanilla = str(cfg["models"]["vanilla_spec"])
    gmp_hf = str(cfg["models"]["gmp_hf"])
    configured_xattn = _local_path(cfg["models"]["xattn_checkpoint"])
    xattn = str(configured_xattn if configured_xattn is not None else _xattn_materialized_path(cfg))
    configured_gmp = _local_path(cfg["models"]["gmp_checkpoint"])
    gmp = str(configured_gmp if configured_gmp is not None else _gmp_materialized_path(cfg))
    device = str(cfg["runtime"]["device"])
    return final, vanilla, gmp_hf, xattn, gmp, device


def _auto_args(task: Task, cfg: dict[str, Any]) -> list[str]:
    final, vanilla, gmp_hf, xattn, gmp, device = _common(cfg)
    out = _task_output(task, cfg)
    out_s = str(out) if out is not None else ""
    module_root = str(_local_path(cfg["runtime"]["module_root"]) or PROJECT_ROOT)
    demo = str(_local_path(cfg["datasets"]["demoset_dir"]))
    misc = str(_local_path(cfg["datasets"]["misc_image_dir"]))
    oracle = str(_task_output(TASK_BY_ID["workspace.cls_register_exchange"], cfg))

    if task.id == "bridge.cross_attention":
        return ["--model", final, "--old-model", final, "--image-dir", demo, "--output-dir", out_s, "--device", device, "--overwrite"]
    if task.id == "bridge.backbone_dynamics":
        return ["--xattn-model", final, "--pretrained", gmp_hf, "--image-dir", demo, "--output-dir", out_s, "--device", device, "--overwrite"]
    if task.id == "bridge.read_null.hallucinations":
        return ["--images", misc, "--correction-model", final, "--full-model", final, "--output-dir", out_s, "--device", device]
    if task.id in {"bridge.read_null.tap_transplants", "bridge.read_null.diagnostic"}:
        return ["--model", final, "--image-dir", misc, "--out", out_s, "--device", device]

    if task.id in {"workspace.broadcast_sinks", "workspace.cls_register_exchange", "workspace.cls_mu_causal"}:
        manifest = str(_workspace_manifest_path(cfg))
        base = [
            "--manifest", manifest,
            "--clip_module", "attnclip_mechinterp_sae", "--model_spec", vanilla,
            "--xattn_checkpoint", xattn, "--gmp_checkpoint", gmp, "--device", device,
        ]
        if task.id == "workspace.broadcast_sinks":
            return ["--out_dir", out_s, *base]
        if task.id == "workspace.cls_register_exchange":
            return ["--out_dir", out_s, *base]
        return ["--out_dir", out_s, "--old_oracle_root", oracle, *base]
    if task.id == "workspace.broadcast_channel_interventions":
        special = str(_local_path(cfg["datasets"]["special_delivery_dir"]))
        return [
            "--out_dir", out_s, "--image_dir", special,
            "--clip_module", "attnclip_mechinterp_sae", "--model_spec", vanilla,
            "--xattn_checkpoint", xattn, "--gmp_checkpoint", gmp, "--device", device,
        ]

    if task.id == "workspace.cls_role_surfaces":
        return ["--oracle_root", oracle, "--out_dir", out_s,
                "--clip_module", "attnclip_mechinterp_sae", "--model_spec", vanilla,
                "--xattn_checkpoint", xattn, "--gmp_checkpoint", gmp, "--device", device]
    if task.id.startswith("workspace.qk_role_gates."):
        return ["--oracle_root", oracle, "--out_dir", out_s,
                "--clip_module", "attnclip_mechinterp_sae", "--model_spec", vanilla,
                "--xattn_checkpoint", xattn, "--gmp_checkpoint", gmp, "--device", device]
    if task.id in {"workspace.register_geometry.secondary", "workspace.register_geometry.grad_attention"}:
        return ["--gmp", gmp, "--xattn", xattn, "--module-root", module_root, "--output-dir", out_s, "--device", device]
    if task.id == "workspace.register_geometry.principal_angles":
        src = _task_output(TASK_BY_ID["workspace.register_geometry.secondary"], cfg) / "data" / "register_means.npz"
        return ["--register-means", str(src), "--output-dir", out_s]
    if task.id == "workspace.register_cache_transport":
        return ["--full-model", final, "--output-dir", out_s, "--device", device]
    if task.id == "workspace.rta_head_population":
        return ["--oracle_root", oracle, "--out_dir", out_s,
                "--clip_module", "attnclip_mechinterp_sae", "--model_spec", vanilla,
                "--gmp_checkpoint", gmp, "--xattn_checkpoint", xattn, "--device", device,
                "--skip_legacy_population_compare"]
    if task.id == "workspace.rta_head_contact_sheets":
        src = _task_output(TASK_BY_ID["workspace.rta_head_population"], cfg)
        return ["--input_dir", str(src), "--output_dir", out_s]
    if task.id in {"workspace.single_image.example", "workspace.single_image.motifs"}:
        image = Path(demo) / "bottle_shower.png"
        return ["--image_path", str(image), "--oracle_root", oracle, "--out_dir", out_s,
                "--clip_module", "attnclip_mechinterp_sae", "--model_spec", vanilla,
                "--gmp_checkpoint", gmp, "--xattn_checkpoint", xattn, "--device", device]
    if task.id == "workspace.text_cls_trajectory":
        return ["--gmp_checkpoint", gmp, "--xattn_checkpoint", xattn, "--rn_checkpoint", xattn, "--out_dir", out_s, "--device", device]

    if task.id in {"rn.control_mechanism", "rn.control_knob", "rn.control_manifold", "rn.subspace_alignment"}:
        return ["--checkpoint", xattn, "--module-root", module_root, "--output-dir", out_s, "--device", device]
    if task.id == "rn.control_surfaces":
        # Paper reproduction always exports the lightweight Blender-friendly surface meshes.
        return ["--checkpoint", xattn, "--module-root", module_root, "--output-dir", out_s,
                "--device", device, "--export-ply"]
    if task.id == "rn.control_surface_flow_maps":
        base = _task_output(TASK_BY_ID["rn.control_surfaces"], cfg)
        fast = base / "compact_summary_rn_control_surfaces_fast.zip"
        full = base / "compact_summary_rn_control_surfaces_full.zip"
        src = fast if fast.exists() or not full.exists() else full
        return ["--input", str(src), "--output-dir", out_s]
    if task.id == "rn.stash_followup":
        rn_variant = str(_oai_rn_materialized_path(cfg))
        return ["--checkpoint", xattn, "--module-root", module_root, "--output-dir", out_s,
                "--pretrained-rn-checkpoint", rn_variant, "--device", device]
    if task.id == "rn.text_relocation":
        return ["--full-model", final, "--output-dir", out_s, "--device", device]
    if task.id == "rn.bridge_transplant":
        return ["--checkpoint", xattn, "--module-root", module_root,
                "--pretrained-module", "attnclip_mechinterp_sae", "--pretrained-spec", vanilla,
                "--output-dir", out_s, "--device", device]
    if task.id == "rn.touch_go_transfer":
        return ["--donor-checkpoint", xattn, "--openai-spec", vanilla,
                "--gmp-checkpoint", gmp, "--out-dir", out_s]

    if task.id == "conv1.gpic_manifold":
        conv = cfg.get("conv1", {})
        models = _conv1_gpic_models(cfg)
        bank_repo = str(conv.get("gpic_embedding_repo") or "zer0int/CLIP-GPIC-embeddings")
        image_dir = str(_local_path(conv.get("image_dir") or "image_sets/retrieval"))
        exp_cfg = str(_local_path(conv.get("experiment_config") or "x_paper_reproduction/conv1_manifold_gpic/experiment_config.json"))
        args = [
            "--reference_bank", bank_repo,
            "--image_dir", image_dir, "--config", exp_cfg,
            "--output_dir", out_s, "--device", device,
        ]
        for model in models:
            args.extend(["--model", model])
        bank_revision = conv.get("gpic_bank_revision")
        if bank_revision is not None and str(bank_revision).strip():
            args.extend(["--bank_revision", str(bank_revision)])
        for custom in conv.get("custom_embedding_banks") or []:
            if custom is not None and str(custom).strip():
                args.extend(["--custom_bank", str(custom)])
        return args

    vanilla_hf = str(cfg["models"].get("vanilla_hf") or "openai/clip-vit-large-patch14")
    special_natural = str(_local_path(cfg["datasets"].get("special_natural_dir") or "image_sets/special_natural"))
    visualtextual = str(_local_path(cfg["datasets"].get("visualtextual_dir") or "image_sets/visualtextual"))
    common_models = [
        "--pretrained_model", vanilla_hf,
        "--gmp_checkpoint", gmp_hf,
        "--xattn_model", final,
    ]

    if task.id == "conv1.xattn_functional_atlas":
        return [
            "--model", final, "--pretrained_model", vanilla_hf,
            "--image_dir", special_natural, "--out_dir", out_s, "--device", device,
            "--batch_size", "8", "--screen_conditions", "FLIP,SHUFFLE", "--screen_images", "8",
            "--max_candidates", "160", "--candidate_top_per_axis", "24",
            "--severe_cos_sim", "0.97", "--event_outlier_z", "3", "--event_outlier_pct", "0.99",
            "--register_threshold", "70", "--compare_pretrained",
            "--pair_screen_top_specs", "12", "--pair_screen_images", "8",
        ]
    if task.id in {"conv1.vanilla_functional_atlas", "conv1.vanilla_functional_atlas_rn"}:
        models = "pretrained,gmp,bare_xattn" if task.id.endswith("_rn") else "gmp,bare_xattn"
        args = [
            "--models", models, "--conditions", "FLIP,SHUFFLE", *common_models,
            "--image_dir", special_natural, "--out_root", out_s, "--device", device,
            "--batch_size", "8", "--screen_images", "8", "--max_candidates", "160",
            "--candidate_top_per_axis", "24", "--severe_cos_sim", "0.97",
            "--event_outlier_z", "3", "--event_outlier_pct", "0.99",
            "--register_threshold", "70", "--register_max", "4", "--register_min", "1",
            "--pair_screen_top_specs", "12", "--pair_screen_images", "8",
            "--focus_channels", "779,720,866,151",
        ]
        if task.id.endswith("_rn"):
            args.extend(["--use_rn_token", "--rn_insert_block", "13"])
        return args
    if task.id in {"conv1.residual_axis_lineage", "conv1.residual_axis_swap_650_565", "conv1.residual_axis_lineage_rn"}:
        is_swap = task.id == "conv1.residual_axis_swap_650_565"
        models = "pretrained_rn,gmp_rn,bare_xattn_rn,full_xattn" if task.id.endswith("_rn") else "pretrained,gmp,bare_xattn,full_xattn"
        return [
            "--suite", "swap" if is_swap else "lineage", "--models", models,
            "--image_dir", special_natural, "--out_root", out_s, *common_models,
            "--device", device, "--batch_size", "4",
            "--axes", "499,468,779,169,720,866,151,650,565,715",
            "--control_count", "24", "--qk_blocks", "22", "--swap_pair", "650,565",
            "--swap_token_scope", "all", "--resume",
        ]
    if task.id == "conv1.mlp_neuron_discovery":
        return [
            "--image_dir", special_natural, "--output_dir", out_s,
            "--models", "pretrained,gmp,bare_xattn,full_xattn", *common_models,
            "--blocks", "11,12,23", "--batch_size", "4", "--device", device, "--amp",
        ]
    if task.id == "conv1.b20_writeback_neurons":
        return [
            "--image_dir", special_natural, "--output_dir", out_s,
            "--models", "pretrained,gmp,bare_xattn,full_xattn", *common_models,
            "--block", "20", "--target_axes", "499,468,779,169,720,866,151,650,565,715",
            "--batch_size", "4", "--device", device, "--amp",
            "--ablation_sizes", "16,32,64,128,220", "--random_repeats", "2",
            "--report_topn", "320", "--overlap_topn", "220",
        ]
    if task.id == "conv1.b20_sharpeners_flatteners":
        discovery = _task_output(TASK_BY_ID["conv1.b20_writeback_neurons"], cfg)
        return [
            "--discovery_root", str(discovery), "--output_dir", out_s, "--image_dir", special_natural,
            "--models", "pretrained,gmp,bare_xattn,full_xattn", *common_models,
            "--family_scope", "top220", "--group_sizes", "16,32,64",
            "--batch_size", "4", "--device", device, "--amp",
        ]
    if task.id == "conv1.b20_pushpull_650_715":
        sharp = _task_output(TASK_BY_ID["conv1.b20_sharpeners_flatteners"], cfg)
        return [
            "--sharp_flat_root", str(sharp), "--output_dir", out_s, "--image_dir", special_natural,
            "--models", "pretrained,gmp,bare_xattn,full_xattn", *common_models,
            "--group_name", "pushpull_sharpH_up_flatH_down", "--group_top_k", "16",
            "--axis_a", "650", "--axis_b", "715", "--q_axis", "565",
            "--batch_size", "4", "--device", device, "--amp",
        ]
    if task.id == "conv1.register_allocator_tomography":
        return [
            "--out_dir", out_s, "--clip_model", vanilla_hf, "--device", device,
            "--batch_size", "16", "--n_per_family", "64", "--causal_n_per_family", "16",
            "--register_threshold", "60", "--watch_patch", "45", "--probe_axis_positions", "5",
            "--channels", "199,499,120,469,227,350", "--run_channel_causality", "--run_chase",
            "--chase_rounds", "64", "--chase_protect_radius", "0", "--save_chase_frames_every", "8",
        ]
    if task.id == "conv1.roleplane_texture_rank.head_rank":
        return ["--out_dir", out_s, "--clip_model", vanilla_hf, "--device", device, "--probe_axis_positions", "5"]
    if task.id == "conv1.roleplane_texture_rank.texture_inverse":
        return [
            "--out_dir", out_s, "--clip_model", vanilla_hf, "--device", device, "--batch_size", "12",
            "--dtd_basis_per_class", "4", "--dtd_eval_per_class", "20",
            "--register_norm_threshold", "60", "--mu_reg_z", "3", "--mu_hidden_z", "5",
            "--scratch_z", "4", "--scratch_topk", "8", "--last_sigma_frac", "0.12",
        ]
    if task.id == "conv1.roleplane_texture_rank.compact":
        return ["--out_dir", out_s]
    if task.id == "conv1.visualtextual_provenance":
        return [
            "--image_dir", visualtextual, "--output_dir", out_s,
            "--models", "pretrained,gmp,full_xattn", *common_models,
            "--device", device, "--amp", "--batch_size", "12",
            "--register_norm_threshold", "60", "--mu_reg_z", "3", "--mu_hidden_z", "5",
            "--scratch_z", "4", "--scratch_topk", "8", "--conv1_causal_topk", "8",
        ]
    if task.id == "conv1.visualtextual_text_direction":
        return [
            "--image_dir", visualtextual, "--output_dir", out_s,
            "--models", "pretrained,gmp,full_xattn", *common_models,
            "--device", device, "--amp", "--batch_size", "12",
            "--bootstrap", "1000", "--signflip", "4000",
            "--synthetic_per_base", "8", "--synthetic_shuffle_tile", "8",
        ]

    if task.id == "figures.role_text_atlas":
        src = _task_output(TASK_BY_ID["workspace.rta_head_population"], cfg)
        # The atlas predates the upstream RN off/on factorial.  Its paper-facing
        # semantics are the native/no-RN role-vs-site map, so select rn_off
        # explicitly rather than relying on postprocessor defaults.
        return ["--root", str(src), "--out_dir", out_s, "--rn-mode", "rn_off"]
    if task.id == "figures.rn_mechanism":
        control = _task_output(TASK_BY_ID["rn.control_mechanism"], cfg)
        stash = _task_output(TASK_BY_ID["rn.stash_followup"], cfg)
        universality = _task_output(TASK_BY_ID["rn.bridge_transplant"], cfg)
        return ["--control-dir", str(control), "--stash-dir", str(stash), "--universality-dir", str(universality), "--output-dir", out_s]

    return []


def _model_specs_for_tasks(tasks: Iterable[Task], cfg: dict[str, Any]) -> dict[str, str]:
    requirements = {req for task in tasks for req in task.requirements}
    specs: dict[str, str] = {}

    needs_vanilla = "models.vanilla_spec" in requirements
    if needs_vanilla:
        specs["vanilla_openai"] = str(get_key(cfg, "models.vanilla_spec"))

    if "models.vanilla_hf" in requirements:
        specs["vanilla_hf"] = str(get_key(cfg, "models.vanilla_hf"))

    needs_xattn_native = "models.xattn_checkpoint" in requirements
    needs_final_hf = "models.final_hf" in requirements
    configured_xattn = get_key(cfg, "models.xattn_checkpoint")
    if needs_xattn_native and configured_xattn is not None and str(configured_xattn).strip():
        specs["xattn_checkpoint"] = str(_local_path(configured_xattn))
    if needs_final_hf or (needs_xattn_native and not str(configured_xattn or "").strip()):
        specs["final_hf"] = str(get_key(cfg, "models.final_hf"))

    needs_gmp_native = "models.gmp_checkpoint" in requirements
    needs_gmp_hf = "models.gmp_hf" in requirements
    configured_gmp = get_key(cfg, "models.gmp_checkpoint")
    if needs_gmp_native and configured_gmp is not None and str(configured_gmp).strip():
        specs["gmp_checkpoint"] = str(_local_path(configured_gmp))
    if needs_gmp_hf or (needs_gmp_native and not str(configured_gmp or "").strip()):
        specs["gmp_hf"] = str(get_key(cfg, "models.gmp_hf"))
    return specs


def _guard_output_root_models(tasks: Iterable[Task], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out_root = _local_path(cfg["output_root"])
    assert out_root is not None
    out_root.mkdir(parents=True, exist_ok=True)
    specs = _model_specs_for_tasks(tasks, cfg)
    identities: dict[str, dict[str, Any]] = {}
    for slot, spec in specs.items():
        print(f"[model-id] resolving {slot}: {spec}")
        identities[slot] = fingerprint_model_spec(spec, project_root=PROJECT_ROOT)
        print(f"[model-id] {slot}: {identities[slot]['sha256']}")
    if identities:
        manifest, notes = guard_and_merge_identities(out_root, identities)
        for note in notes:
            print(f"[model-id] WARNING: {note}")
        print(f"[model-id] workspace identity: {manifest}")
    return identities


def _write_task_identity(task: Task, cfg: dict[str, Any], identities: dict[str, dict[str, Any]]) -> None:
    out = _task_output(task, cfg)
    if out is None or not identities:
        return
    import json
    path = out / "_meta" / "reproduction_model_identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"task": task.id, "workspace_models": identities}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _child_env() -> dict[str, str]:
    """Environment for probe subprocesses with repo-local packages importable.

    Executing ``python x_paper_reproduction/probe_*.py`` makes Python put the
    probe directory, not the repository root, at ``sys.path[0]``.  Preserve the
    caller environment but prepend the repository root to ``PYTHONPATH`` so
    sibling packages such as ``attnclip_mechinterp_xattn`` resolve reliably on
    Windows and POSIX alike.
    """
    env = os.environ.copy()
    root = str(PROJECT_ROOT)
    existing = env.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if root not in parts:
        env["PYTHONPATH"] = os.pathsep.join([root, *parts])
    return env


def _smoke_args(task: Task) -> list[str]:
    """Tiny-but-real integration settings for expensive probes.

    Smoke mode is not scientifically meaningful.  It exists to execute model
    loading, dataset plumbing, helper namespaces, forward/backward paths, and
    output serialization with the smallest practical workload.  Static assets
    are still required by preflight even when an expensive optional analysis is
    skipped in smoke mode.
    """
    tid = task.id
    if tid == "bridge.cross_attention":
        return ["--batch-size", "1"]
    if tid == "bridge.backbone_dynamics":
        return ["--batch-size", "1"]
    if tid in {"bridge.read_null.hallucinations", "bridge.read_null.tap_transplants"}:
        return ["--batch-size", "1"]
    if tid == "bridge.read_null.diagnostic":
        # Already a one/few-case microscope; this subcommand has no batch-size option.
        return []
    if tid in {"workspace.broadcast_sinks", "workspace.broadcast_channel_interventions", "workspace.cls_register_exchange"}:
        return ["--batch_size", "1"]
    if tid == "workspace.cls_mu_causal":
        return ["--batch_size", "1", "--max_images", "4"]
    if tid == "workspace.cls_role_surfaces":
        return ["--batch_size", "1", "--grid", "5", "--grid_chunk", "1",
                "--rta_triplets", "1", "--rta_compare_blocks", "20",
                "--rta_compare_planes", "1x2", "--rta_compare_grid", "5"]
    if tid.startswith("workspace.qk_role_gates."):
        return ["--batch_size", "1", "--n_images", "2", "--gradient_images", "1", "--gradient_batch_size", "1"]
    if tid == "workspace.register_geometry.secondary":
        return ["--limit", "10", "--batch-size", "2", "--folds", "2",
                "--bootstrap", "20", "--permutations", "20"]
    if tid == "workspace.register_geometry.grad_attention":
        return ["--limit", "4", "--batch-size", "2", "--text-batch-size", "2",
                "--bootstrap", "20", "--permutations", "20"]
    if tid == "workspace.register_cache_transport":
        return ["--batch-size", "1", "--calibration-per-subset", "1", "--overlay-n-per-modality", "1", "--skip-overlays"]
    if tid == "workspace.rta_head_population":
        return ["--limit", "4", "--batch_triplets", "1"]
    if tid == "workspace.text_cls_trajectory":
        return ["--limit", "4", "--batch_pairs", "1"]
    if tid == "rn.control_mechanism":
        return ["--n-scenes", "2", "--batch-size", "1", "--jacobian-scenes", "1", "--jacobian-iters", "1", "--jacobian-k", "1", "--patch-samples-per-image", "1", "--svd-max-rows-per-view", "16"]
    if tid == "rn.control_knob":
        return ["--languages", "en", "--query-modes", "english", "--limit-per-language", "1", "--basis-pairs-per-language", "1", "--population-alphas", "0"]
    if tid == "rn.control_manifold":
        return ["--languages", "en", "--query-modes", "native", "--planes", "1x2",
                "--atlas-keys", "1", "--basis-pairs-per-language", "1",
                "--grid-points", "3", "--surface-batch", "1",
                "--ridge-points", "3", "--ridge-batch", "1", "--ridge-topk", "1",
                "--ridge-refine-steps", "0", "--ridge-refine-random", "0",
                "--vocab-limit", "32", "--vocab-build-batch", "16",
                "--vocab-score-batch", "16", "--pair-beam", "2"]
    if tid == "rn.control_surfaces":
        return ["--languages", "en", "--query-modes", "english", "--planes", "1x2",
                "--atlas-keys", "1", "--basis-pairs-per-language", "4",
                "--grid-points", "3", "--surface-batch", "1",
                "--ridge-points", "3", "--ridge-batch", "1", "--ridge-topk", "1",
                "--ridge-refine-steps", "0", "--ridge-refine-random", "0", "--fast-ridge"]
    if tid == "rn.subspace_alignment":
        return ["--languages", "en", "--pairs-per-language", "1"]
    if tid == "rn.stash_followup":
        return ["--batch-size", "1", "--lowrank-max-rows", "16"]
    if tid == "rn.text_relocation":
        return ["--batch-size", "1", "--top-n-per-modality", "1"]
    if tid == "rn.bridge_transplant":
        return ["--languages", "en", "--atlas-keys", "1", "--basis-pairs-per-language", "1",
                "--grid-points", "3", "--surface-batch", "1", "--skip-ridge",
                "--text-align-max", "8", "--text-align-batch", "2"]
    if tid == "rn.touch_go_transfer":
        return ["--batch-size", "1", "--max-samples-per-subset", "1", "--num-workers", "0", "--skip-trajectory"]
    if tid == "conv1.gpic_manifold":
        return ["--image_batch_size", "1", "--workers", "0",
                "--limit_source_images", "1", "--smoke_grid",
                "--vocab_strategy", "off", "--gpic_topk", "2",
                "--plot_topk", "1", "--nn_query_chunk_size", "8"]
    if tid == "conv1.xattn_functional_atlas":
        return ["--suite", "morphology", "--batch_size", "1", "--skip_umap", "--skip_tsne"]
    if tid in {"conv1.vanilla_functional_atlas", "conv1.vanilla_functional_atlas_rn"}:
        return ["--suite", "morphology", "--batch_size", "1", "--screen_images", "1", "--skip_umap", "--skip_tsne"]
    if tid in {"conv1.residual_axis_lineage", "conv1.residual_axis_lineage_rn"}:
        return ["--models", "pretrained", "--axes", "650,565", "--control_count", "1", "--qk_blocks", "22", "--batch_size", "1"]
    if tid == "conv1.residual_axis_swap_650_565":
        return ["--models", "pretrained", "--axes", "650,565", "--control_count", "1", "--qk_blocks", "22", "--batch_size", "1", "--swap_token_scope", "all"]
    if tid == "conv1.mlp_neuron_discovery":
        return ["--models", "pretrained", "--blocks", "20", "--max_images", "2", "--batch_size", "1", "--topk", "2", "--overlap_topn", "2"]
    if tid == "conv1.b20_writeback_neurons":
        return ["--models", "pretrained", "--max_images", "2", "--batch_size", "1", "--skip_ablation", "--report_topn", "4", "--overlap_topn", "4"]
    if tid == "conv1.b20_sharpeners_flatteners":
        return ["--models", "pretrained", "--max_images", "2", "--batch_size", "1", "--group_sizes", "1", "--skip_ablation"]
    if tid == "conv1.b20_pushpull_650_715":
        return ["--models", "pretrained", "--max_images", "2", "--batch_size", "1", "--group_top_k", "1", "--modes", "baseline"]
    if tid == "conv1.register_allocator_tomography":
        return ["--batch_size", "1", "--n_per_family", "1", "--causal_n_per_family", "1",
                "--probe_axis_positions", "2", "--channels", "199", "--no-run_channel_causality", "--no-run_chase"]
    if tid == "conv1.roleplane_texture_rank.head_rank":
        return ["--probe_axis_positions", "2"]
    if tid == "conv1.roleplane_texture_rank.texture_inverse":
        return ["--batch_size", "1", "--dtd_basis_per_class", "1", "--dtd_eval_per_class", "1", "--scratch_topk", "2"]
    if tid == "conv1.visualtextual_provenance":
        return ["--models", "pretrained", "--batch_size", "1", "--conv1_causal_topk", "1"]
    if tid == "conv1.visualtextual_text_direction":
        return ["--models", "pretrained", "--batch_size", "1", "--bootstrap", "10", "--signflip", "20", "--synthetic_per_base", "1"]
    return []


_PARSE_ONLY_SENTINEL = 86
_PARSE_ONLY_MARKER = "__REPRO_PARSE_ONLY_OK__"


def _parse_only_validate_command(task: Task, cfg: dict[str, Any]) -> tuple[bool, str]:
    """Validate the exact task CLI without entering experiment computation.

    A subprocess imports the real entry point and runs its real argparse parser.
    ``ArgumentParser.parse_args`` is wrapped so that a successful parse exits
    immediately with a private sentinel before model/data work can begin. This
    catches subcommand-specific option drift that a file-wide grep cannot.
    """
    exact = _command(task, cfg, [])
    if len(exact) < 2:
        return False, "could not construct parse-only command"

    wrapper = r'''import argparse
from pathlib import Path
import runpy
import sys

_SENTINEL = 86
_MARKER = "__REPRO_PARSE_ONLY_OK__"
_orig_parse_args = argparse.ArgumentParser.parse_args


def _parse_then_stop(self, args=None, namespace=None):
    ns = _orig_parse_args(self, args=args, namespace=namespace)
    print(_MARKER)
    raise SystemExit(_SENTINEL)


argparse.ArgumentParser.parse_args = _parse_then_stop
script = str(Path(sys.argv[1]).resolve())
# Match normal ``python path/to/script.py`` import semantics. ``runpy.run_path``
# does not put the script directory on sys.path, but the real interpreter does.
script_dir = str(Path(script).parent)
if not sys.path or sys.path[0] != script_dir:
    sys.path.insert(0, script_dir)
sys.argv = [script, *sys.argv[2:]]
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exc:
    if exc.code == _SENTINEL:
        raise SystemExit(0)
    raise
raise SystemExit(87)
'''
    cmd = [sys.executable, "-c", wrapper, *exact[1:]]
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        return False, "parse-only CLI validation timed out after 45s"
    except Exception as exc:
        return False, f"parse-only CLI validation failed: {exc}"

    if result.returncode == 0 and _PARSE_ONLY_MARKER in (result.stdout or ""):
        return True, ""

    text = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
    lines = [line for line in text.splitlines() if line.strip()]
    tail = " | ".join(lines[-4:]) if lines else f"exit {result.returncode}"
    return False, tail


def _preflight_tasks(tasks: list[Task], cfg: dict[str, Any], *, import_help: bool) -> list[str]:
    """Return every detectable problem before any expensive experiment starts."""
    problems: list[str] = []
    seen_scripts: set[Path] = set()
    for task in tasks:
        ready, task_problems = _task_ready(task, cfg)
        if not ready:
            problems.extend(f"{task.id}: {p}" for p in task_problems)
        script = PROBE_ROOT / task.script
        if script.is_file() and script not in seen_scripts:
            seen_scripts.add(script)
            try:
                py_compile.compile(str(script), doraise=True)
            except Exception as exc:
                problems.append(f"{task.id}: compile failed for {script.name}: {exc}")
        try:
            # Constructing the exact command catches dispatcher/catalog drift.
            _command(task, cfg, [])
        except Exception as exc:
            problems.append(f"{task.id}: command construction failed: {exc}")

        if bool(cfg.get("_smoke", False)) and script.is_file():
            # Validate the exact subcommand and exact generated smoke command.
            # A file-wide option grep is insufficient because sibling
            # subcommands can expose different argparse contracts.
            ok, detail = _parse_only_validate_command(task, cfg)
            if not ok:
                problems.append(f"{task.id}: smoke CLI contract failed: {detail}")

    if import_help:
        # Run each unique entry point only to argparse help.  This imports the
        # real module stack but exits before model/dataset computation.
        checked: set[tuple[str, str | None]] = set()
        for task in tasks:
            key = (task.script, task.command)
            if key in checked:
                continue
            checked.add(key)
            cmd = [sys.executable, str(PROBE_ROOT / task.script)]
            if task.command:
                cmd.append(task.command)
            cmd.append("--help")
            try:
                result = subprocess.run(
                    cmd, cwd=PROJECT_ROOT, env=_child_env(),
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True, timeout=45,
                )
                if result.returncode != 0:
                    tail = (result.stderr or "").strip().splitlines()[-1:] or [f"exit {result.returncode}"]
                    problems.append(f"{task.id}: import/help smoke failed: {tail[0]}")
            except subprocess.TimeoutExpired:
                problems.append(f"{task.id}: import/help smoke timed out after 45s")
            except Exception as exc:
                problems.append(f"{task.id}: import/help smoke failed: {exc}")
    return problems


def preflight(args: argparse.Namespace) -> int:
    cfg = _load_command_config(args)
    if bool(getattr(args, "smoke", False)):
        cfg["_smoke"] = True
    selected = _selected_tasks_from_args(args, default_selectors=["all"])
    tasks = _with_dependencies(selected) if args.with_deps else selected
    problems = _preflight_tasks(tasks, cfg, import_help=not args.no_imports)
    if problems:
        print(f"[preflight] FAIL — {len(problems)} problem(s) found before compute:")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    print(f"[preflight] PASS — {len(tasks)} task(s), static assets/config/compile/command" + ("/imports" if not args.no_imports else "") + " OK")
    return 0


def _command(task: Task, cfg: dict[str, Any], passthrough: list[str]) -> list[str]:
    script = PROBE_ROOT / task.script
    cmd = [sys.executable, str(script)]
    if task.command:
        cmd.append(task.command)
    if not task.manual_args:
        cmd.extend(_auto_args(task, cfg))
    cmd.extend(_configured_task_args(task, cfg))
    if bool(cfg.get("_smoke", False)):
        cmd.extend(_smoke_args(task))
    cmd.extend(passthrough)
    return cmd


def run(args: argparse.Namespace) -> int:
    cfg = _load_command_config(args)
    if bool(getattr(args, "smoke", False)):
        cfg["_smoke"] = True
    selected = _selected_tasks_from_args(args, default_selectors=[])
    tasks = _with_dependencies(selected) if args.with_deps else selected

    # Fail fast on *all* known requirements before task 1 gets a GPU.  This is
    # intentionally global: missing vocab for task 24 must not be discovered
    # after task 23 has run for ninety minutes.
    upfront = _preflight_tasks(tasks, cfg, import_help=False)
    if upfront:
        print(f"[preflight] BLOCKED — {len(upfront)} problem(s) found before compute:")
        for problem in upfront:
            print(f"  - {problem}")
        return 2

    passthrough = list(args.extra or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if passthrough and len(tasks) != 1:
        raise SystemExit("Passthrough args after -- are allowed only when exactly one task is selected.")

    policy = str(cfg.get("runtime", {}).get("model_cache_policy", "keep"))
    if policy not in {"keep", "task", "run"}:
        raise SystemExit(f"Invalid runtime.model_cache_policy={policy!r}; expected keep/task/run")
    original_xattn_checkpoint = get_key(cfg, "models.xattn_checkpoint")
    original_gmp_checkpoint = get_key(cfg, "models.gmp_checkpoint")

    identities: dict[str, dict[str, Any]] = {}
    if not args.dry_run:
        try:
            identities = _guard_output_root_models(tasks, cfg)
        except RuntimeError as exc:
            print(f"[NOPE] {exc}")
            return 3

    for i, task in enumerate(tasks, 1):
        if _task_complete(task, cfg) and not args.force:
            print(f"[{i}/{len(tasks)}] CACHED {task.id} — use --force to rerun")
            continue
        ready, problems = _task_ready(task, cfg)
        if not ready:
            print(f"[{i}/{len(tasks)}] BLOCKED {task.id}")
            for p in problems:
                print(f"  - {p}")
            return 2
        dep_missing = [d for d in task.deps if not _task_complete(TASK_BY_ID[d], cfg)]
        if dep_missing and not args.with_deps:
            print(f"[{i}/{len(tasks)}] WAITING {task.id}: missing cached deps {', '.join(dep_missing)}")
            print("  Re-run with --with-deps, or run the prerequisite task(s) first.")
            return 2

        configured_args = _configured_task_args(task, cfg)
        if task.manual_args and not passthrough and not configured_args:
            print(f"[{i}/{len(tasks)}] MANUAL {task.id}")
            print("  This provenance/audit utility has no safe canonical argument set.")
            print("  Pass its script arguments after `--`, or set task_args for this task in the config.")
            return 2

        if "datasets.objectnet_mvt" in task.requirements and not args.dry_run:
            manifest = _materialize_workspace_manifest(cfg)
            print(f"[dataset] ObjectNet-MVT workspace manifest: {manifest}")

        if "models.xattn_checkpoint" in task.requirements and not args.dry_run:
            resolved_xattn = _ensure_xattn_checkpoint(cfg)
            cfg["models"]["xattn_checkpoint"] = str(resolved_xattn)
        if "models.gmp_checkpoint" in task.requirements and not args.dry_run:
            resolved_gmp = _ensure_gmp_checkpoint(cfg)
            cfg["models"]["gmp_checkpoint"] = str(resolved_gmp)
        if "models.oai_rn_variant" in task.requirements and not args.dry_run:
            _ensure_oai_rn_variant(cfg)

        out = _task_output(task, cfg)
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
        cmd = _command(task, cfg, passthrough)
        print("\n" + "=" * 96)
        print(f"[{i}/{len(tasks)}] {task.id} — {task.title}")
        print("=" * 96)
        _print_variant_plan(task)
        print(shlex.join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=_child_env())
        if result.returncode != 0:
            print(f"[failed] {task.id} exited with code {result.returncode}")
            return result.returncode
        _write_task_identity(task, cfg, identities)
        if policy == "task":
            removed = _cleanup_managed_model_caches(cfg)
            if removed:
                print(f"[model-cache] cleanup(task): removed {len(removed)} managed model file(s)")
            cfg["models"]["xattn_checkpoint"] = original_xattn_checkpoint
            cfg["models"]["gmp_checkpoint"] = original_gmp_checkpoint

    if not args.dry_run and policy == "run":
        removed = _cleanup_managed_model_caches(cfg)
        if removed:
            print(f"[model-cache] cleanup(run): removed {len(removed)} managed model file(s)")
    return 0


def graph(args: argparse.Namespace) -> int:
    tasks = _selected_tasks_from_args(args, default_selectors=["paper"])
    ids = {t.id for t in tasks}
    for t in tasks:
        deps = [d for d in t.deps if d in ids or args.external]
        rhs = ", ".join(deps) if deps else "—"
        print(f"{t.id} <- {rhs}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=_selector_help_text(include_task_ids=False),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {REPRODUCTION_RELEASE} (schema {REPRODUCTION_SCHEMA})")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="create/update reproduction_config.json")
    s.add_argument("--non-interactive", action="store_true")
    s.add_argument("--output-root")
    s.add_argument("--final-model")
    s.add_argument("--vanilla-model", help="canonical OpenAI CLIP spec, normally ViT-L/14")
    s.add_argument("--vanilla-hf-model", help="vanilla OpenAI CLIP HF source, normally openai/clip-vit-large-patch14")
    s.add_argument("--gmp-model", help="GmP comparison model; not used as generic vanilla CLIP")
    s.add_argument("--xattn-checkpoint")
    s.add_argument("--gmp-checkpoint")
    s.add_argument("--demoset-dir")
    s.add_argument("--misc-image-dir")
    s.add_argument("--special-delivery-dir")
    s.add_argument("--objectnet-mvt-root")
    s.add_argument("--device")
    s.add_argument("--model-cache-policy", choices=("keep", "task", "run"))
    s.set_defaults(func=setup)

    s = sub.add_parser("list", help="list stable reproduction tasks", epilog=_selector_help_text(), formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("selectors", nargs="*", help="task id, family, paper, or all")
    s.add_argument("--selection", type=Path, help="task-only selection JSON from reproduce_info/configurator.html; main config is unchanged")
    s.add_argument("--all", action="store_true", help="include optional controls when selecting a family/paper")
    s.add_argument("--models", action="store_true", help="show the scientific model variants each task instantiates")
    s.set_defaults(func=list_tasks)

    s = sub.add_parser("status", help="show cached/ready/blocked tasks", epilog=_selector_help_text(), formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("selectors", nargs="*", help="task id, family, paper, or all")
    s.add_argument("--selection", type=Path, help="task-only selection JSON; main reproduction_config.json remains authoritative")
    s.add_argument("--all", action="store_true")
    s.add_argument("--strict", action="store_true", help="exit nonzero if any selected task is blocked")
    s.add_argument("--output-root", type=Path, help="override output_root for this status check only")
    s.set_defaults(func=status)

    s = sub.add_parser("preflight", help="validate all selected tasks before any expensive compute", epilog=_selector_help_text(), formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("selectors", nargs="*", help="task id, family, paper, or all")
    s.add_argument("--selection", type=Path, help="task-only selection JSON; main reproduction_config.json remains authoritative")
    s.add_argument("--all", action="store_true")
    s.add_argument("--with-deps", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--no-imports", action="store_true", help="skip subprocess import/--help checks")
    s.add_argument("--smoke", action="store_true", help="also validate smoke-mode command overrides/output routing")
    s.add_argument("--output-root", type=Path, help="override output_root for this check only")
    s.set_defaults(func=preflight)

    s = sub.add_parser("graph", help="print task dependency edges", epilog=_selector_help_text(), formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("selectors", nargs="*", help="task id, family, paper, or all")
    s.add_argument("--selection", type=Path, help="task-only selection JSON; main reproduction_config.json remains authoritative")
    s.add_argument("--all", action="store_true")
    s.add_argument("--external", action="store_true", help="show dependency ids outside the selected set")
    s.set_defaults(func=graph)

    s = sub.add_parser("run", help="run task(s), a family, `paper`, or `all`", epilog=_selector_help_text(), formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("selectors", nargs="*", help="selector(s); use `all` to run every automatic task")
    s.add_argument("--selection", type=Path, help="task-only selection JSON from the configurator; models/datasets/output still come from the main config")
    s.add_argument("--all", action="store_true", help="include controls for family selectors")
    s.add_argument("--with-deps", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--force", action="store_true", help="rerun even when cached markers exist")
    s.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    s.add_argument("--smoke", action="store_true", help="run tiny real integration workloads under <output_root>/_smoke; not scientifically meaningful")
    s.add_argument("--output-root", type=Path, help="override output_root for this run only")
    s.add_argument("--model-cache-policy", choices=("keep", "task", "run"),
                   help="managed derived model caches: keep, delete after each successful task, or delete after a successful run")
    s.set_defaults(func=run)
    return p


def main() -> int:
    parser = build_parser()
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        extra = argv[split + 1:]
        argv = argv[:split]
    args = parser.parse_args(argv)
    args.extra = extra
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
