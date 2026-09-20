#!/usr/bin/env python3
"""Single-config launcher for the complete CLIP hard-text training curriculum.

Edit ``training_config.json`` and run::

    python train_clip_xattn_bridge.py

The launcher deliberately keeps the legacy 429.96M pretraining implementation
isolated from the final 440M AnyText implementation.  It also generates the
runtime branch config from the single user JSON, assembles all artifact paths,
and preserves the historical architecture boundary: the joint post-export only
undoes GmP while retaining the legacy read/content/presence implant.  The
final-base trainer owns the legacy-to-final migration after setting its seed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "training_config.json"
RUNTIME_DIR = PROJECT_ROOT / "_runtime"
CODE_DIR = PROJECT_ROOT / "main_hard_text_gate"
LEGACY_DIR = CODE_DIR / "legacy_hard_text_pre"

STAGE_ORDER = ("soft_token", "read", "content", "joint", "final_base", "all_weights")

ROUTER_UTILITY_SCRIPT = CODE_DIR / "train_router_utility.py"

SCRIPT_PATHS = {
    "soft_token": LEGACY_DIR / "pre1_train_clip_soft_text_token_imagenet.py",
    "read": LEGACY_DIR / "train_gmp_hard_text_implant_imagenet.py",
    "content": LEGACY_DIR / "train_gmp_hard_text_implant_imagenet.py",
    "joint": LEGACY_DIR / "train_gmp_hard_text_implant_imagenet.py",
    "final_base": CODE_DIR / "train_gmp_anytext_stage1.py",
    "all_weights": CODE_DIR / "train_gmp_anytext_all_weights_stage2.py",
}

LEGACY_PACKAGE_STAGES = frozenset(("soft_token", "read", "content", "joint"))
LEGACY_STAGES = frozenset(("read", "content", "joint"))
FINAL_STAGES = frozenset(("final_base", "all_weights"))

# Arguments assembled from global/path/artifact settings must never reappear in
# a stage's local args block.  This guard prevents the exact configuration drift
# that the single-config launcher is intended to eliminate.
DERIVED_ARGUMENTS = {
    "soft_token": frozenset((
        "imagenet_root", "train_root", "val_root", "class_index_json",
        "val_ground_truth", "word_bank_json", "download_root", "font_paths",
        "overlay_dir", "handwriting_overlay_root", "handwriting_probability",
        "model", "device", "out_dir", "image_size",
        "precision_diagnostics", "precision_log_every",
        "precision_example_values", "precision_parameter_limit",
        "precision_max_values", "precision_save_plots",
    )),
    "read": frozenset((
        "stage", "clip_package", "read_attention_architecture", "clip_package_root",
        "read_tap_blocks", "model_path", "implant_checkpoint", "soft_token_path", "resume",
        "imagenet_root", "train_root", "val_root", "wnid_json", "overlay_dir",
        "font_paths", "init_implant_checkpoint", "out_dir", "image_size",
        "handwriting_overlay_root", "handwriting_probability", "device", "amp_dtype",
        "precision_diagnostics", "precision_log_every",
        "precision_example_values", "precision_parameter_limit",
        "precision_max_values", "precision_save_plots",
    )),
    "content": frozenset((
        "stage", "clip_package", "read_attention_architecture", "clip_package_root",
        "read_tap_blocks", "model_path", "implant_checkpoint", "soft_token_path", "resume",
        "imagenet_root", "train_root", "val_root", "wnid_json", "overlay_dir",
        "font_paths", "init_implant_checkpoint", "out_dir", "image_size",
        "handwriting_overlay_root", "handwriting_probability", "device", "amp_dtype",
        "precision_diagnostics", "precision_log_every",
        "precision_example_values", "precision_parameter_limit",
        "precision_max_values", "precision_save_plots",
    )),
    "joint": frozenset((
        "stage", "clip_package", "read_attention_architecture", "clip_package_root",
        "read_tap_blocks", "model_path", "implant_checkpoint", "soft_token_path", "resume",
        "imagenet_root", "train_root", "val_root", "wnid_json", "overlay_dir",
        "font_paths", "init_implant_checkpoint", "out_dir", "image_size",
        "handwriting_overlay_root", "handwriting_probability", "device", "amp_dtype",
        "precision_diagnostics", "precision_log_every",
        "precision_example_values", "precision_parameter_limit",
        "precision_max_values", "precision_save_plots",
    )),
    "final_base": frozenset((
        "clip_package", "clip_package_root", "read_attention_architecture",
        "read_null_enabled", "read_null_insert_block", "read_null_start_phase",
        "model_path", "implant_checkpoint", "branch_config", "read_tap_blocks",
        "ortho_tap_blocks", "source_tap_blocks", "mini_fixed_ortho_blocks",
        "mini_late_anchor_block", "rank1_probe_npz",
        "rank1_probe_key", "out_dir", "imagenet_text_root",
        "imagenet_handwriting_root", "textcaps_root",
        "coco_root", "coco_train_json", "coco_train_gpt_json", "coco_train_reading_json",
        "coco_val_json", "coco_val_gpt_json", "coco_val_reading_json",
        "clevr_train_root", "clevr_val_root", "clevr_metadata_jsonl",
        "mvt_csv", "mvt_image_root", "control_font_paths", "mini_config_json", "image_size",
        "device", "amp_dtype",
        "precision_diagnostics", "precision_log_every",
        "precision_example_values", "precision_parameter_limit",
        "precision_max_values", "precision_save_plots", "require_legacy_migration_input",
    )),
    "all_weights": frozenset((
        "clip_package", "clip_package_root", "read_attention_architecture",
        "read_null_enabled", "read_null_insert_block", "read_null_start_phase",
        "model_path", "implant_checkpoint", "branch_config", "read_tap_blocks",
        "ortho_tap_blocks", "source_tap_blocks", "mini_fixed_ortho_blocks",
        "mini_late_anchor_block", "rank1_probe_npz",
        "rank1_probe_key", "out_dir", "imagenet_text_root",
        "imagenet_handwriting_root", "textcaps_root",
        "coco_root", "coco_train_json", "coco_train_gpt_json", "coco_train_reading_json",
        "coco_val_json", "coco_val_gpt_json", "coco_val_reading_json",
        "clevr_train_root", "clevr_val_root", "clevr_metadata_jsonl",
        "mvt_csv", "mvt_image_root", "control_font_paths", "mini_config_json", "image_size",
        "device", "amp_dtype",
        "precision_diagnostics", "precision_log_every",
        "precision_example_values", "precision_parameter_limit",
        "precision_max_values", "precision_save_plots",
    )),
}

BOOLEAN_OPTIONAL_FLAGS = {
    "precision_diagnostics": (
        "--precision_diagnostics", "--no-precision_diagnostics"
    ),
    "precision_save_plots": (
        "--precision_save_plots", "--no-precision_save_plots"
    ),
    "enforce_legacy_fingerprint": (
        "--enforce_legacy_fingerprint", "--no-enforce_legacy_fingerprint"
    ),
    "require_anytext_stage1": (
        "--require_anytext_stage1", "--no-require_anytext_stage1"
    ),
    "read_null_enabled": (
        "--read_null_enabled", "--no-read_null_enabled"
    ),
    "read_null_save_plots": (
        "--read_null_save_plots", "--no-read_null_save_plots"
    ),
    "save_stage_complete_checkpoint": (
        "--save_stage_complete_checkpoint", "--no-save_stage_complete_checkpoint"
    ),
    "math_curriculum_enabled": (
        "--math_curriculum_enabled", "--no-math_curriculum_enabled"
    ),
    "clevr_enabled": (
        "--clevr_enabled", "--no-clevr_enabled"
    ),
    "clevr_fill_contours": (
        "--clevr_fill_contours", "--no-clevr_fill_contours"
    ),
}

WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class ConfigError(RuntimeError):
    pass

FORBIDDEN_DERIVED_READING_KEYS = frozenset((
    "textcaps_reading_json",
))

# TextCaps derived OCR is the known gate-killing condition and remains forbidden.
FORBIDDEN_FINAL_SOURCE_MARKERS = (
    "textcaps_reading_json",
    "load_textcaps_reading_manifest",
    "derived_reading_conf999/textcaps",
)

REQUIRED_COCO_READING_KEYS = frozenset((
    "coco_train_reading_json",
    "coco_reading_probability",
    "coco_reading_mask_weight",
))

def _walk_config_keys(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.append((path, child))
            out.extend(_walk_config_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            out.extend(_walk_config_keys(child, f"{prefix}[{index}]"))
    return out


def enforce_historical_final_dataset_policy(config: Mapping[str, Any]) -> None:
    """Fail closed: historical TextCaps + literal-pseudoword ImageNet + enhanced COCO only."""
    offenders = [
        path for path, _value in _walk_config_keys(config)
        if path.rsplit(".", 1)[-1] in FORBIDDEN_DERIVED_READING_KEYS
    ]
    if offenders:
        joined = "\n".join(f"  - {item}" for item in offenders)
        raise ConfigError(
            "Forbidden derived TextCaps reading config key(s) detected. "
            "This branch intentionally restores COCO reading supervision ONLY:\n" + joined
        )

    found_keys = {
        path.rsplit(".", 1)[-1]
        for path, _value in _walk_config_keys(config)
    }
    missing = sorted(REQUIRED_COCO_READING_KEYS - found_keys)
    if missing:
        raise ConfigError(
            "COCO-restoration branch is missing required COCO reading config key(s): "
            + ", ".join(missing)
        )

    source_offenders: list[str] = []
    for stage in ("final_base", "all_weights"):
        path = SCRIPT_PATHS[stage]
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_FINAL_SOURCE_MARKERS:
            if marker in text:
                source_offenders.append(f"{path.name}: {marker}")
    if source_offenders:
        joined = "\n".join(f"  - {item}" for item in source_offenders)
        raise ConfigError(
            "Active final trainer contains forbidden derived TextCaps code:\n" + joined
        )

    # Conversely, both final trainers must explicitly wire the enhanced COCO path.
    required_markers = (
        "coco_train_reading_json",
        "coco_reading_probability",
        "load_coco_reading_manifest",
    )
    missing_source: list[str] = []
    for stage in ("final_base", "all_weights"):
        path = SCRIPT_PATHS[stage]
        text = path.read_text(encoding="utf-8")
        for marker in required_markers:
            if marker not in text:
                missing_source.append(f"{path.name}: {marker}")
    if missing_source:
        joined = "\n".join(f"  - {item}" for item in missing_source)
        raise ConfigError("Active final trainer lost required COCO restoration wiring:\n" + joined)


def historical_textcaps_manifest_paths(config: Mapping[str, Any]) -> list[str]:
    paths = resolved_paths(config)
    root_text = str(paths["textcaps_root"])
    if is_windows_absolute(root_text):
        base = PureWindowsPath(root_text)
        return [
            str(base / "manifests" / "train.jsonl"),
            str(base / "manifests" / "validation.jsonl"),
        ]
    base = Path(root_text)
    return [
        str(base / "manifests" / "train.jsonl"),
        str(base / "manifests" / "validation.jsonl"),
    ]


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Top-level configuration must be a JSON object")
    return data


def require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Expected object at {key!r}")
    return value


def is_windows_absolute(value: str) -> bool:
    return bool(WINDOWS_ABSOLUTE_RE.match(value)) or value.startswith("\\\\")


def resolve_project_path(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value)
    if is_windows_absolute(text):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def join_output_path(root: str, relative: str) -> str:
    if is_windows_absolute(root):
        return str(PureWindowsPath(root) / PureWindowsPath(relative))
    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path
    return str((root_path / Path(relative)).resolve())


def path_exists(path_text: str) -> bool:
    return Path(path_text).expanduser().exists()


def looks_like_model_identifier(value: str) -> bool:
    # OpenAI CLIP identifiers such as ViT-L/14 are not filesystem paths.
    if is_windows_absolute(value):
        return False
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.exists():
        return False
    return not value.startswith((".", "~", "/")) and value.count("/") == 1 and not Path(value).suffix


def value_to_cli(name: str, value: Any) -> list[str]:
    if name.startswith("_") or value is None:
        return []
    if name in BOOLEAN_OPTIONAL_FLAGS:
        positive, negative = BOOLEAN_OPTIONAL_FLAGS[name]
        if not isinstance(value, bool):
            raise ConfigError(f"{name} must be true or false")
        return [positive if value else negative]
    option = f"--{name}"
    if isinstance(value, bool):
        return [option] if value else []
    if isinstance(value, list):
        if not value:
            return []
        return [option, *(str(item) for item in value)]
    return [option, str(value)]


def args_to_cli(values: Mapping[str, Any]) -> list[str]:
    command: list[str] = []
    for name, value in values.items():
        command.extend(value_to_cli(name, value))
    return command


def display_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def config_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def stage_output_dir(config: Mapping[str, Any], stage: str) -> str:
    global_cfg = require_mapping(config, "global")
    stage_cfg = require_mapping(require_mapping(config, "stages"), stage)
    train_root = str(global_cfg["train_root"])
    output_subdir = str(stage_cfg["output_subdir"])
    return join_output_path(train_root, output_subdir)


def artifact_paths(config: Mapping[str, Any]) -> dict[str, str]:
    out = {stage: stage_output_dir(config, stage) for stage in STAGE_ORDER}
    return {
        "soft_token_final": join_output_path(out["soft_token"], "soft_text_token_final.pt"),
        "read_best": join_output_path(out["read"], "best_inference.pt"),
        "content_best": join_output_path(out["content"], "best_inference.pt"),
        "joint_merged": join_output_path(out["joint"], "best_merged_state_dict.pt"),
        "joint_ordinary": join_output_path(
            out["joint"], "best_merged_state_dict__ungmp_oaiclip_fullmodel.pt"
        ),
        "final_base_merged": join_output_path(
            out["final_base"], "stage1_complete_merged_state_dict.pt"
        ),
        "final_base_ordinary": join_output_path(
            out["final_base"],
            "stage1_complete_merged_state_dict__ungmp_oaiclip_fullmodel.pt",
        ),
        "all_weights_merged": join_output_path(
            out["all_weights"], "stage1_complete_merged_state_dict.pt"
        ),
        "all_weights_ordinary": join_output_path(
            out["all_weights"],
            "stage1_complete_merged_state_dict__ungmp_oaiclip_fullmodel.pt",
        ),
        "final_base_best_pieces": join_output_path(
            out["final_base"], "phase_1c_best.pt"
        ),
        "router_utility_a4": join_output_path(
            out["final_base"],
            f"{str(require_mapping(config, 'router_utility').get('output_prefix') or 'router_utility_a4')}_best.pt",
        ),
        "all_weights_best_merged": join_output_path(
            out["all_weights"], "phase_1c_best_merged_state_dict.pt"
        ),
        "all_weights_best_ordinary": join_output_path(
            out["all_weights"],
            "phase_1c_best_merged_state_dict__ungmp_oaiclip_fullmodel.pt",
        ),
    }


def generated_branch_config(config: Mapping[str, Any]) -> Path:
    global_cfg = require_mapping(config, "global")
    taps = require_mapping(global_cfg, "tap_blocks")
    payload = {
        "ortho_blocks": list(taps["orthographic"]),
        "source_blocks": list(taps["source"]),
        "late_blocks": list(taps["late"]),
        "early_expanded_width": int(taps["early_expanded_width"]),
    }
    RUNTIME_DIR.mkdir(exist_ok=True)
    path = RUNTIME_DIR / "branch_config.generated.json"
    write_json(path, payload)
    return path


def validate_topology(config: Mapping[str, Any]) -> None:
    global_cfg = require_mapping(config, "global")
    taps = require_mapping(global_cfg, "tap_blocks")
    ortho = list(taps.get("orthographic", []))
    source = list(taps.get("source", []))
    late = list(taps.get("late", []))
    if not ortho or not source or len(late) != 2:
        raise ConfigError(
            "tap_blocks requires nonempty orthographic/source lists and exactly two late blocks"
        )
    all_blocks = ortho + source + late
    if any(not isinstance(block, int) or block < 0 for block in all_blocks):
        raise ConfigError("All tap blocks must be nonnegative integers")
    architecture = global_cfg.get("read_attention_architecture")
    if architecture not in ("softmax", "sigmoid_mass", "sigmoid_all"):
        raise ConfigError("read_attention_architecture must be softmax, sigmoid_mass, or sigmoid_all")
    read_null_enabled = bool(global_cfg.get("read_null_enabled", False))
    read_null_insert_block = int(global_cfg.get("read_null_insert_block", 20))
    if read_null_enabled and architecture not in ("softmax", "sigmoid_all"):
        raise ConfigError("READ_NULL requires softmax or raw sigmoid_all PIECES attention")
    read_null_start_phase = str(global_cfg.get("read_null_start_phase", "1b5"))
    if read_null_start_phase not in ("1a", "1b", "1b5", "1c"):
        raise ConfigError("global.read_null_start_phase must be 1a, 1b, 1b5, or 1c")
    if read_null_enabled and ("1a", "1b", "1b5", "1c").index(read_null_start_phase) < 2:
        raise ConfigError("This branch intentionally introduces READ_NULL no earlier than phase 1b5")
    if read_null_enabled and read_null_insert_block > min(late):
        raise ConfigError(
            "READ_NULL must be inserted no later than the first late reader tap: "
            f"insert={read_null_insert_block}, late={late}"
        )
    precision = require_mapping(global_cfg, "precision")
    autocast = precision.get("autocast")
    if autocast not in ("auto", "bf16", "fp16", "none"):
        raise ConfigError("global.precision.autocast must be auto, bf16, fp16, or none")
    legacy_dtype = precision.get("legacy_dtype", "auto")
    if legacy_dtype not in ("auto", "bf16", "fp16", "none"):
        raise ConfigError("global.precision.legacy_dtype must be auto, bf16, fp16, or none")
    if not isinstance(global_cfg.get("handwriting_enabled", False), bool):
        raise ConfigError("global.handwriting_enabled must be true or false")
    if precision.get("trainable_parameters") != "fp32":
        raise ConfigError("global.precision.trainable_parameters is fixed to fp32")
    if precision.get("custom_implant_compute") != "fp32":
        raise ConfigError("global.precision.custom_implant_compute is fixed to fp32")
    diagnostics = require_mapping(precision, "diagnostics")
    if not isinstance(diagnostics.get("enabled"), bool):
        raise ConfigError("global.precision.diagnostics.enabled must be true or false")
    if int(diagnostics.get("log_every_optimizer_steps", 0)) <= 0:
        raise ConfigError("precision diagnostics log interval must be positive")
    if int(diagnostics.get("example_values_per_tensor", 0)) <= 0:
        raise ConfigError("precision diagnostics example count must be positive")
    if int(diagnostics.get("selected_parameter_limit", 0)) <= 0:
        raise ConfigError("precision diagnostics parameter limit must be positive")
    if int(diagnostics.get("max_values_for_statistics", 0)) <= 0:
        raise ConfigError("precision diagnostics statistic sample limit must be positive")
    if not isinstance(diagnostics.get("save_plots"), bool):
        raise ConfigError("global.precision.diagnostics.save_plots must be true or false")


def validate_local_args(config: Mapping[str, Any]) -> None:
    stages = require_mapping(config, "stages")
    shared_sections = require_mapping(config, "shared_args")
    legacy_shared = require_mapping(shared_sections, "legacy_hard_text")
    final_shared = require_mapping(shared_sections, "final_anytext")
    for key in ("benchmark_scam_repo", "benchmark_rta_repo"):
        value = final_shared.get(key)
        if not isinstance(value, str) or not value.strip() or value.count("/") != 1:
            raise ConfigError(f"shared_args.final_anytext.{key} must be a Hugging Face dataset repo id")
    if bool(final_shared.get("select_by_benchmark_score", False)):
        raise ConfigError(
            "SCAM/RTA are evaluation-only; shared_args.final_anytext.select_by_benchmark_score must remain false"
        )
    for stage in STAGE_ORDER:
        stage_cfg = require_mapping(stages, stage)
        local_args = require_mapping(stage_cfg, "args")
        shared_args: Mapping[str, Any]
        if stage in LEGACY_STAGES:
            shared_args = legacy_shared
        elif stage in FINAL_STAGES:
            shared_args = final_shared
        else:
            shared_args = {}
        duplicate = set(local_args).intersection(shared_args)
        if duplicate:
            raise ConfigError(
                f"stages.{stage}.args duplicates shared arguments: "
                + ", ".join(sorted(duplicate))
            )
        overlap = (set(local_args) | set(shared_args)).intersection(DERIVED_ARGUMENTS[stage])
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ConfigError(
                f"Stage {stage} redundantly defines launcher-derived arguments: {names}"
            )


def resolve_optional_config_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return resolve_project_path(str(value))


def resolve_stage_resume(stage_cfg: Mapping[str, Any], out_dir: str) -> str | None:
    resume = stage_cfg.get("resume")
    if resume in (None, ""):
        return None
    if resume == "auto":
        return join_output_path(out_dir, "last.pt")
    return resolve_project_path(str(resume))


def precision_diagnostic_args(global_cfg: Mapping[str, Any]) -> dict[str, Any]:
    precision = require_mapping(global_cfg, "precision")
    diagnostics = require_mapping(precision, "diagnostics")
    return {
        "precision_diagnostics": bool(diagnostics["enabled"]),
        "precision_log_every": int(diagnostics["log_every_optimizer_steps"]),
        "precision_example_values": int(diagnostics["example_values_per_tensor"]),
        "precision_parameter_limit": int(diagnostics["selected_parameter_limit"]),
        "precision_max_values": int(diagnostics["max_values_for_statistics"]),
        "precision_save_plots": bool(diagnostics["save_plots"]),
    }


def common_global_args(
    config: Mapping[str, Any], *, legacy: bool = False
) -> dict[str, Any]:
    global_cfg = require_mapping(config, "global")
    precision = require_mapping(global_cfg, "precision")
    handwriting_probability = (
        float(global_cfg["handwriting_probability"])
        if bool(global_cfg.get("handwriting_enabled", False))
        else 0.0
    )
    if legacy:
        # The full-from-scratch sigmoid branch teaches the historical
        # READ/CONTENT/JOINT stages exactly one newer concept: which PIECES
        # token-attention normalization to instantiate. Everything else on the
        # historical invocation surface remains unchanged.
        return {
            "clip_package": global_cfg["clip_package"],
            "read_attention_architecture": global_cfg["read_attention_architecture"],
            "image_size": global_cfg["image_size"],
            "device": global_cfg["device"],
            "amp_dtype": precision["legacy_dtype"],
            # Global precision diagnostics are part of the derived launcher
            # surface for legacy stages too. Keep historical optimizer/AMP math
            # unchanged while letting READ/CONTENT/JOINT audit actual updates.
            **precision_diagnostic_args(global_cfg),
        }
    return {
        "clip_package": global_cfg["clip_package"],
        "read_attention_architecture": global_cfg["read_attention_architecture"],
        "read_null_enabled": bool(global_cfg.get("read_null_enabled", False)),
        "read_null_insert_block": int(global_cfg.get("read_null_insert_block", 20)),
        "read_null_start_phase": str(global_cfg.get("read_null_start_phase", "1b5")),
        "image_size": global_cfg["image_size"],
        "device": global_cfg["device"],
        "amp_dtype": precision["autocast"],
        **precision_diagnostic_args(global_cfg),
    }


def resolved_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve every user-supplied path while preserving nulls and path lists."""
    paths = require_mapping(config, "paths")
    resolved: dict[str, Any] = {}
    for key, value in paths.items():
        if value in (None, ""):
            resolved[key] = None
        elif isinstance(value, list):
            resolved[key] = [resolve_project_path(str(item)) for item in value]
        else:
            resolved[key] = resolve_project_path(str(value))
    return resolved


def semicolon_path_list(values: Any, *, key: str) -> str | None:
    if values in (None, ""):
        return None
    if not isinstance(values, list):
        raise ConfigError(f"paths.{key} must be a JSON array")
    return ";".join(str(value) for value in values) or None


def clevr_dataset_cli_args(config: Mapping[str, Any]) -> dict[str, Any]:
    datasets = config.get("datasets", {})
    if datasets is None:
        datasets = {}
    if not isinstance(datasets, Mapping):
        raise ConfigError("datasets must be a JSON object when present")
    cfg = datasets.get("clevr_property_binding", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, Mapping):
        raise ConfigError("datasets.clevr_property_binding must be a JSON object")
    include = cfg.get("include", False)
    if not isinstance(include, bool):
        raise ConfigError("datasets.clevr_property_binding.include must be true or false")
    return {
        "clevr_enabled": include,
        "clevr_additive_probability_1b5": float(cfg.get("additive_probability_1b5", 0.50)),
        "clevr_additive_probability_1c": float(cfg.get("additive_probability_1c", 0.50)),
        "clevr_packets_per_batch": int(cfg.get("packets_per_batch", 1)),
        "clevr_val_batches": int(cfg.get("val_batches", 16)),
        "clevr_preview_count": int(cfg.get("preview_count", 8)),
        "clevr_colored_text_probability": float(cfg.get("colored_text_probability", 0.15)),
        "clevr_ood_color_probability": float(cfg.get("ood_color_probability", 0.08)),
        "clevr_standalone_count_probability": float(cfg.get("standalone_count_probability", 0.15)),
        "clevr_canny_low": int(cfg.get("canny_low", 45)),
        "clevr_canny_high": int(cfg.get("canny_high", 120)),
        "clevr_canny_dilate_px": int(cfg.get("canny_dilate_px", 3)),
        "clevr_placement_margin_px": int(cfg.get("placement_margin_px", 4)),
        "clevr_max_obstacle_fraction": float(cfg.get("max_obstacle_fraction", 0.025)),
        "clevr_placement_stride_px": int(cfg.get("placement_stride_px", 3)),
        "clevr_fill_contours": bool(cfg.get("fill_contours", True)),
        "clevr_min_contour_area_fraction": float(cfg.get("min_contour_area_fraction", 0.001)),
        "clevr_min_font_size": int(cfg.get("min_font_size", 8)),
        "clevr_max_font_size": int(cfg.get("max_font_size", 42)),
    }


def validate_clevr_dataset_config(config: Mapping[str, Any]) -> None:
    args = clevr_dataset_cli_args(config)
    for key in ("clevr_additive_probability_1b5", "clevr_additive_probability_1c",
                "clevr_colored_text_probability", "clevr_ood_color_probability",
                "clevr_standalone_count_probability", "clevr_max_obstacle_fraction"):
        value = float(args[key])
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"{key} must be in [0, 1], got {value}")
    if int(args["clevr_packets_per_batch"]) < 0:
        raise ConfigError("clevr_packets_per_batch must be >= 0")
    if int(args["clevr_val_batches"]) < 0 or int(args["clevr_preview_count"]) < 0:
        raise ConfigError("CLEVR val/preview counts must be >= 0")
    if int(args["clevr_canny_low"]) < 0 or int(args["clevr_canny_high"]) <= int(args["clevr_canny_low"]):
        raise ConfigError("CLEVR Canny thresholds must satisfy 0 <= low < high")
    if int(args["clevr_min_font_size"]) < 4 or int(args["clevr_max_font_size"]) < int(args["clevr_min_font_size"]):
        raise ConfigError("CLEVR font sizes are invalid")


def build_stage_args(
    config: Mapping[str, Any], stage: str, branch_config: Path
) -> tuple[dict[str, Any], list[str]]:
    global_cfg = require_mapping(config, "global")
    stages = require_mapping(config, "stages")
    stage_cfg = require_mapping(stages, stage)
    shared_sections = require_mapping(config, "shared_args")
    if stage in LEGACY_STAGES:
        shared = dict(require_mapping(shared_sections, "legacy_hard_text"))
    elif stage in FINAL_STAGES:
        shared = dict(require_mapping(shared_sections, "final_anytext"))
    else:
        shared = {}
    stage_local = dict(require_mapping(stage_cfg, "args"))
    overlap = set(shared).intersection(stage_local)
    if overlap:
        raise ConfigError(
            f"Stage {stage} redundantly overrides shared arguments: "
            + ", ".join(sorted(overlap))
        )
    local = {**shared, **stage_local}
    paths = resolved_paths(config)
    artifacts = artifact_paths(config)
    out_dir = stage_output_dir(config, stage)
    required_inputs: list[str] = []

    if stage == "soft_token":
        if stage_cfg.get("resume") not in (None, ""):
            raise ConfigError("soft_token has no optimizer resume argument; leave resume=null")
        model = stage_cfg.get("input_override") or global_cfg["base_gmp_model"]
        resolved_model = str(model)
        if not looks_like_model_identifier(resolved_model):
            resolved_model = resolve_project_path(resolved_model)
            required_inputs.append(resolved_model)
        local.update({
            "imagenet_root": paths["imagenet_root"],
            "train_root": paths["imagenet_train_root_override"],
            "val_root": paths["imagenet_val_root_override"],
            "class_index_json": paths["imagenet_class_index_json"],
            "val_ground_truth": paths["imagenet_val_ground_truth"],
            "word_bank_json": paths["soft_token_word_bank_json"],
            "download_root": paths["clip_download_root"],
            "font_paths": semicolon_path_list(paths["custom_fonts"], key="custom_fonts"),
            "overlay_dir": paths["overlay_selection"],
            # Historical soft-token phase is deliberately narrow. Handwriting is
            # reintroduced only in later hard/final stages, not in addressability.
            "handwriting_overlay_root": None,
            "handwriting_probability": 0.0,
            "model": resolved_model,
            "device": global_cfg["device"],
            "out_dir": out_dir,
            "image_size": global_cfg["image_size"],
            **precision_diagnostic_args(global_cfg),
        })
        return local, required_inputs

    if stage in LEGACY_STAGES:
        model_path = str(stage_cfg.get("input_override") or global_cfg["base_gmp_model"])
        if not looks_like_model_identifier(model_path):
            model_path = resolve_project_path(model_path)
            required_inputs.append(model_path)
        local.update(common_global_args(config, legacy=True))
        local.update({
            "stage": stage,
            "model_path": model_path,
            "imagenet_root": paths["imagenet_root"],
            "train_root": paths["imagenet_train_root_override"],
            "val_root": paths["imagenet_val_root_override"],
            "wnid_json": paths["imagenet_wnid_json"],
            "overlay_dir": paths["overlay_selection"],
            "font_paths": semicolon_path_list(paths["custom_fonts"], key="custom_fonts"),
            "init_implant_checkpoint": paths["legacy_init_implant_checkpoint"],
            "out_dir": out_dir,
            "clip_package_root": None,
        })
        if stage == "read":
            local["soft_token_path"] = artifacts["soft_token_final"]
            local["implant_checkpoint"] = None
            required_inputs.append(artifacts["soft_token_final"])
        elif stage == "content":
            local["soft_token_path"] = None
            local["implant_checkpoint"] = artifacts["read_best"]
            required_inputs.append(artifacts["read_best"])
        else:
            local["soft_token_path"] = None
            local["implant_checkpoint"] = artifacts["content_best"]
            required_inputs.append(artifacts["content_best"])
        resume = resolve_stage_resume(stage_cfg, out_dir)
        local["resume"] = resume
        if resume:
            required_inputs.append(resume)
        return local, required_inputs

    if stage in FINAL_STAGES:
        taps = require_mapping(global_cfg, "tap_blocks")
        rank1 = require_mapping(global_cfg, "rank1_probe")
        start_phase = str(local.get("start_phase", "1a"))
        default_input = artifacts["joint_ordinary"]
        default_implant: str | None = None
        # final/base freezes the original CLIP backbone. Therefore its handoff to
        # all-weights does NOT need a ~1.6 GB full merged checkpoint: reconstruct
        # the final/base best model from the SAME full seed model plus the compact
        # phase_<phase>_best PIECES checkpoint (implant + hard/null + READ_NULL).
        if stage == "final_base":
            if start_phase == "1b5" and not stage_cfg.get("input_override"):
                default_input = join_output_path(
                    stage_output_dir(config, "final_base"),
                    "phase_1b_best_merged_state_dict.pt",
                )
        else:
            final_base_cfg = require_mapping(stages, "final_base")
            final_base_args = require_mapping(final_base_cfg, "args")
            final_base_start = str(final_base_args.get("start_phase", "1a"))
            if final_base_cfg.get("input_override"):
                default_input = str(final_base_cfg["input_override"])
            elif final_base_start == "1b5":
                default_input = join_output_path(
                    stage_output_dir(config, "final_base"),
                    "phase_1b_best_merged_state_dict.pt",
                )
            else:
                default_input = artifacts["joint_ordinary"]

            final_phase = None
            for candidate in ("1c", "1b5", "1b", "1a"):
                if int(final_base_args.get(f"phase_{candidate}_epochs", 0)) > 0:
                    final_phase = candidate
                    break
            if final_phase is None:
                raise ConfigError("final_base has no trained phase to hand off to all_weights")
            default_implant = join_output_path(
                stage_output_dir(config, "final_base"),
                f"phase_{final_phase}_best.pt",
            )

        model_path = str(stage_cfg.get("input_override") or default_input)
        if not looks_like_model_identifier(model_path):
            model_path = resolve_project_path(model_path)
            required_inputs.append(model_path)
        implant_override = stage_cfg.get("implant_checkpoint_override")
        implant = resolve_optional_config_path(implant_override) if implant_override else (resolve_project_path(default_implant) if default_implant else None)
        if implant:
            required_inputs.append(implant)
        if (
            stage == "final_base"
            and start_phase != "1a"
            and implant is not None
            and bool(local.get("reset_tap_logits_uniform", False))
        ):
            raise ConfigError(
                "Unsafe final_base continuation: reset_tap_logits_uniform=true would "
                "erase tap weights loaded from implant_checkpoint_override. Set it to "
                "false, or restart from phase 1a without an implant checkpoint."
            )
        # Router A4 is an internal artifact of this training run.  A null config
        # value means "derive from the current train_root"; an explicit value is
        # retained only as an advanced continuation override.
        router_aux_checkpoint = None
        if stage == "final_base":
            router_aux_checkpoint = (
                paths.get("router_utility_a4_checkpoint")
                or resolve_project_path(artifacts["router_utility_a4"])
            )

        local.update(common_global_args(config, legacy=False))
        local.update(clevr_dataset_cli_args(config))
        local.update({
            "clip_package_root": str(PROJECT_ROOT),
            "model_path": model_path,
            "implant_checkpoint": implant,
            "require_legacy_migration_input": bool(stage == "final_base" and start_phase == "1a"),
            "branch_config": str(branch_config),
            "read_tap_blocks": list(taps["late"]),
            "ortho_tap_blocks": list(taps["orthographic"]),
            "source_tap_blocks": list(taps["source"]),
            "mini_fixed_ortho_blocks": list(taps["orthographic"]),
            "mini_late_anchor_block": int(list(taps["late"])[0]),
            "rank1_probe_npz": resolve_optional_config_path(rank1.get("npz")),
            "rank1_probe_key": rank1.get("key", "mean_dir"),
            "out_dir": out_dir,
            "imagenet_text_root": paths["imagenet_text_root"],
            "imagenet_wnid_json": paths["imagenet_wnid_json"],
            "imagenet_handwriting_root": paths["imagenet_handwriting_root"],
            "textcaps_root": paths["textcaps_root"],
            "coco_root": paths["coco_root"],
            "coco_train_json": paths["coco_train_json"],
            "coco_train_gpt_json": paths["coco_train_gpt_json"],
            "coco_train_reading_json": paths["coco_train_reading_json"],
            "coco_val_json": paths["coco_val_json"],
            "coco_val_gpt_json": paths["coco_val_gpt_json"],
            "coco_val_reading_json": paths["coco_val_reading_json"],
            "clevr_train_root": paths["clevr_train_root"],
            "clevr_val_root": paths["clevr_val_root"],
            "clevr_metadata_jsonl": paths["clevr_metadata_jsonl"],
            "mvt_csv": paths["mvt_csv"],
            "mvt_image_root": paths["mvt_image_root"],
            "control_font_paths": list(paths["custom_fonts"]),
            "mini_config_json": paths["mini_config_json"],
            "router_aux_salt_n_pepper_root": paths.get("salt_n_pepper_root"),
            "router_aux_init_checkpoint": router_aux_checkpoint,
        })
        if local["rank1_probe_npz"]:
            required_inputs.append(str(local["rank1_probe_npz"]))
        if bool(local.get("math_curriculum_enabled", False)):
            required_inputs.append(str(resolve_project_path(str(paths["imagenet_wnid_json"]))))
        if bool(local.get("clevr_enabled", False)):
            required_inputs.extend([
                str(paths["clevr_train_root"]),
                str(paths["clevr_val_root"]),
                str(paths["clevr_metadata_jsonl"]),
            ])
        if bool(local.get("router_aux_enabled", False)):
            required_inputs.append(str(paths["salt_n_pepper_root"]))
            if stage == "final_base":
                if not router_aux_checkpoint:
                    raise ConfigError("Could not derive router utility A4 checkpoint")
                required_inputs.append(str(router_aux_checkpoint))
        return local, required_inputs

    raise AssertionError(stage)


def python_executable(config: Mapping[str, Any]) -> str:
    configured = require_mapping(config, "global").get("python_executable")
    return str(configured) if configured else sys.executable


def stage_environment(stage: str) -> dict[str, str]:
    env = dict(os.environ)
    if stage not in LEGACY_PACKAGE_STAGES:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(PROJECT_ROOT) if not existing else str(PROJECT_ROOT) + os.pathsep + existing
        )
    return env


def stage_cwd(stage: str) -> Path:
    return LEGACY_DIR if stage in LEGACY_PACKAGE_STAGES else PROJECT_ROOT


def stage_script_argument(stage: str) -> str:
    script = SCRIPT_PATHS[stage]
    if stage in LEGACY_PACKAGE_STAGES:
        return script.name
    return str(script)



def required_dataset_config_keys(config: Mapping[str, Any], stage: str) -> list[str]:
    final_shared = require_mapping(require_mapping(config, "shared_args"), "final_anytext")
    if stage == "soft_token":
        keys = ["imagenet_root", "overlay_selection"]
        if bool(require_mapping(config, "global").get("handwriting_enabled", False)):
            keys.append("handwriting_overlays")
        return keys
    if stage in LEGACY_STAGES:
        keys = ["imagenet_root", "imagenet_wnid_json", "overlay_selection"]
        if bool(require_mapping(config, "global").get("handwriting_enabled", False)):
            keys.append("handwriting_overlays")
        return keys
    keys = [
        "imagenet_text_root", "imagenet_handwriting_root", "imagenet_wnid_json",
        "textcaps_root", "coco_root",
        "coco_train_json", "coco_train_gpt_json", "coco_train_reading_json",
        "coco_val_json", "coco_val_gpt_json",
    ]
    if bool(clevr_dataset_cli_args(config).get("clevr_enabled", False)):
        keys.extend(("clevr_train_root", "clevr_val_root", "clevr_metadata_jsonl"))
    if bool(final_shared.get("router_aux_enabled", False)):
        keys.append("salt_n_pepper_root")
    if (
        bool(require_mapping(require_mapping(require_mapping(config, "stages"), stage), "args").get("run_benchmark_validation", False))
        and bool(final_shared.get("benchmark_include_mvt", False))
    ):
        keys.extend(("mvt_csv", "mvt_image_root"))
    return keys


def validate_dataset_configured(config: Mapping[str, Any], stage: str) -> None:
    paths = require_mapping(config, "paths")
    missing = [key for key in required_dataset_config_keys(config, stage) if paths.get(key) in (None, "")]
    if missing:
        raise ConfigError(
            "Training dataset paths are not prepared for stage "
            f"{stage!r}: {', '.join(missing)}. Run prepare_training_data.py first "
            "and use the generated training_config.local.json."
        )

def validate_required_paths(paths: Iterable[str], *, label: str) -> None:
    missing = [path for path in paths if not path_exists(path)]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise ConfigError(f"Missing required {label} path(s):\n{formatted}")


def relevant_external_paths(config: Mapping[str, Any], stage: str) -> list[str]:
    paths = resolved_paths(config)
    if stage == "soft_token":
        keys = [
            "imagenet_root", "imagenet_train_root_override", "imagenet_val_root_override",
            "imagenet_class_index_json", "imagenet_val_ground_truth",
            "soft_token_word_bank_json", "overlay_selection", "custom_fonts",
        ]
        if bool(require_mapping(config, "global").get("handwriting_enabled", False)):
            keys.append("handwriting_overlays")
    elif stage in LEGACY_STAGES:
        keys = [
            "imagenet_root", "imagenet_train_root_override", "imagenet_val_root_override",
            "imagenet_wnid_json", "overlay_selection",
            "custom_fonts", "legacy_init_implant_checkpoint",
        ]
        if bool(require_mapping(config, "global").get("handwriting_enabled", False)):
            keys.append("handwriting_overlays")
    else:
        keys = [
            "imagenet_text_root", "imagenet_handwriting_root", "textcaps_root",
            "coco_root", "coco_train_json", "coco_train_gpt_json", "coco_train_reading_json",
            "coco_val_json", "coco_val_gpt_json", "coco_val_reading_json",
            "custom_fonts", "mini_config_json",
        ]
        final_shared = require_mapping(require_mapping(config, "shared_args"), "final_anytext")
        stage_args = require_mapping(require_mapping(require_mapping(config, "stages"), stage), "args")
        if (
            bool(stage_args.get("run_benchmark_validation", False))
            and bool(final_shared.get("benchmark_include_mvt", False))
        ):
            keys.extend(("mvt_csv", "mvt_image_root"))
        if bool(final_shared.get("router_aux_enabled", False)):
            keys.append("salt_n_pepper_root")
        if bool(clevr_dataset_cli_args(config).get("clevr_enabled", False)):
            keys.extend(["clevr_train_root", "clevr_val_root", "clevr_metadata_jsonl"])
    result: list[str] = []
    for key in keys:
        value = paths[key]
        if value in (None, ""):
            continue
        if isinstance(value, list):
            result.extend(str(item) for item in value)
        else:
            result.append(str(value))
    return result


def output_is_nonempty(path_text: str) -> bool:
    path = Path(path_text)
    return path.exists() and any(path.iterdir())


def completion_marker(out_dir: str) -> Path:
    return Path(out_dir) / ".pipeline_complete.json"


def enforce_output_policy(
    config: Mapping[str, Any], stage: str, out_dir: str, *, cli_allow_existing: bool
) -> bool:
    policy = str(require_mapping(config, "pipeline").get("existing_output_policy", "error"))
    if cli_allow_existing:
        policy = "allow"
    if policy not in ("error", "allow", "skip_if_complete"):
        raise ConfigError(f"Unknown existing_output_policy: {policy}")
    marker = completion_marker(out_dir)
    if policy == "skip_if_complete" and marker.is_file():
        print(f"[skip] {stage}: completion marker exists at {marker}")
        return False
    if policy == "error" and output_is_nonempty(out_dir):
        raise ConfigError(
            f"Output directory is nonempty for stage {stage}: {out_dir}\n"
            "Set pipeline.existing_output_policy='allow', use --allow-existing, "
            "or choose a new global.train_root."
        )
    return True


def command_record(
    config: Mapping[str, Any], stage: str, command: Sequence[str], cwd: Path,
    resolved_args: Mapping[str, Any], out_dir: str, *, write_to_output: bool,
) -> None:
    payload = {
        "stage": stage,
        "cwd": str(cwd),
        "command": list(command),
        "display_command": display_command(command),
        "resolved_args": dict(resolved_args),
        "config_sha256": config_digest(config),
    }
    if write_to_output:
        write_json(Path(out_dir) / "launcher_resolved_command.json", payload)
    write_json(RUNTIME_DIR / f"{stage}_last_command.json", payload)


def ensure_reconstructed_phase1b_seed(
    config: Mapping[str, Any], stage: str, resolved_args: Mapping[str, Any], *, dry_run: bool
) -> None:
    """If the historical phase-1B merged seed was pruned, rebuild it from compact PIECES."""
    if stage not in FINAL_STAGES:
        return
    model_path = str(resolved_args.get("model_path", ""))
    if not model_path.endswith("phase_1b_best_merged_state_dict.pt"):
        return
    if path_exists(model_path):
        print(f"[seed] reusing existing merged phase-1B seed: {model_path}")
        return
    compact_path = model_path[:-len("_merged_state_dict.pt")] + ".pt"
    if not dry_run and not path_exists(compact_path):
        raise ConfigError(
            "Missing both phase_1b_best_merged_state_dict.pt and its compact reconstruction source: "
            f"{compact_path}"
        )
    global_cfg = require_mapping(config, "global")
    command = [
        python_executable(config),
        str(CODE_DIR / "reconstruct_merged_from_compact.py"),
        "--compact_checkpoint", compact_path,
        "--output_path", model_path,
        "--clip_package_root", str(PROJECT_ROOT),
        "--read_attention_architecture", str(global_cfg.get("read_attention_architecture", "softmax")),
        "--read_null_insert_block", str(int(global_cfg.get("read_null_insert_block", 20))),
        "--read_null_enabled" if bool(global_cfg.get("read_null_enabled", False)) else "--no-read_null_enabled",
    ]
    print(f"[seed] merged phase-1B seed missing; reconstructing from compact PIECES: {compact_path}")
    run_command(command, cwd=PROJECT_ROOT, env=stage_environment("final_base"), dry_run=dry_run)


def run_command(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], dry_run: bool
) -> None:
    print(f"[cwd] {cwd}")
    print(f"[cmd] {display_command(command)}")
    if dry_run:
        return
    subprocess.run(list(command), cwd=str(cwd), env=dict(env), check=True)


def export_command(
    config: Mapping[str, Any], export_name: str, model_path: str, output_path: str
) -> tuple[list[str], Mapping[str, Any], Path, Path, Mapping[str, str]]:
    export_cfg = dict(require_mapping(require_mapping(config, "exports"), export_name))
    export_cfg = {key: value for key, value in export_cfg.items() if not key.startswith("_")}
    args: dict[str, Any] = {
        "model_path": model_path,
        "output_path": output_path,
        **export_cfg,
    }
    if export_name == "legacy_to_final":
        script = LEGACY_DIR / "export_ungmp_legacy_oaiclip_full_model.py"
        cwd = LEGACY_DIR
        env = stage_environment("joint")
    else:
        script = CODE_DIR / "export_ungmp_oaiclip_full_model.py"
        cwd = PROJECT_ROOT
        env = stage_environment("final_base")
    command = [
        python_executable(config),
        str(script if cwd == PROJECT_ROOT else script.name),
        *args_to_cli(args),
    ]
    return command, args, script, cwd, env


def run_post_export(config: Mapping[str, Any], stage: str, *, dry_run: bool) -> None:
    stages_cfg = require_mapping(config, "stages")
    stage_cfg = require_mapping(stages_cfg, stage)
    if not bool(stage_cfg.get("post_export", True)):
        print(f"[export] {stage}: skipped by stages.{stage}.post_export=false")
        return

    artifacts = artifact_paths(config)
    if stage == "joint":
        export_name = "legacy_to_final"
        model_path = artifacts["joint_merged"]
        output_path = artifacts["joint_ordinary"]
    elif stage == "final_base":
        export_name = "final_model"
        model_path = artifacts["final_base_merged"]
        output_path = artifacts["final_base_ordinary"]
    elif stage == "all_weights":
        # Convert BEST, not the redundant stage-complete/last copy. The trainer
        # keeps rolling full `best` and `last`; only best receives the fp16 OpenAI
        # checkpoint instantiation used by downstream evals.
        args_cfg = require_mapping(stage_cfg, "args")
        final_phase = None
        for candidate in ("1c", "1b5", "1b", "1a"):
            if int(args_cfg.get(f"phase_{candidate}_epochs", 0)) > 0:
                final_phase = candidate
                break
        if final_phase is None:
            raise ConfigError("all_weights has no trained phase to export")
        out_dir = stage_output_dir(config, "all_weights")
        export_name = "final_model"
        model_path = join_output_path(out_dir, f"phase_{final_phase}_best_merged_state_dict.pt")
        output_path = join_output_path(
            out_dir,
            f"phase_{final_phase}_best_merged_state_dict__ungmp_oaiclip_fullmodel.pt",
        )
    else:
        return
    if not dry_run:
        validate_required_paths([model_path], label=f"{stage} export input")
    command, _, _, export_cwd, export_env = export_command(
        config, export_name, model_path, output_path
    )
    print(f"[export] {stage} BEST -> {output_path}")
    run_command(
        command,
        cwd=export_cwd,
        env=export_env,
        dry_run=dry_run,
    )


def run_stage_process(
    config: MutableMapping[str, Any],
    cli: argparse.Namespace,
    stage: str,
    branch_path: Path,
    *,
    dry_run: bool,
    validate_paths: bool,
    allow_existing: bool = False,
    write_completion: bool = True,
) -> None:
    stages_cfg = require_mapping(config, "stages")
    stage_cfg = require_mapping(stages_cfg, stage)
    if not bool(stage_cfg.get("enabled", True)):
        print(f"[skip] {stage}: enabled=false")
        return
    out_dir = stage_output_dir(config, stage)
    if not enforce_output_policy(
        config,
        stage,
        out_dir,
        cli_allow_existing=bool(cli.allow_existing or allow_existing),
    ):
        return
    resolved_args, required_inputs = build_stage_args(config, stage, branch_path)
    ensure_reconstructed_phase1b_seed(config, stage, resolved_args, dry_run=dry_run)
    if validate_paths and not dry_run:
        validate_dataset_configured(config, stage)
        validate_required_paths(required_inputs, label=f"{stage} input")
        validate_required_paths(relevant_external_paths(config, stage), label=f"{stage} dataset")
        if stage in FINAL_STAGES:
            validate_required_paths(
                historical_textcaps_manifest_paths(config),
                label=f"{stage} original TextCaps manifest",
            )
    command = [
        python_executable(config),
        stage_script_argument(stage),
        *args_to_cli(resolved_args),
    ]
    if not dry_run:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    command_record(
        config,
        stage,
        command,
        stage_cwd(stage),
        resolved_args,
        out_dir,
        write_to_output=not dry_run,
    )
    print("\n" + "=" * 88)
    print(f"[stage] {stage}")
    print("=" * 88)
    run_command(
        command,
        cwd=stage_cwd(stage),
        env=stage_environment(stage),
        dry_run=dry_run,
    )
    run_post_export(config, stage, dry_run=dry_run)
    if not dry_run and write_completion:
        write_json(
            completion_marker(out_dir),
            {
                "stage": stage,
                "config_sha256": config_digest(config),
                "output_dir": out_dir,
            },
        )


def run_final_base_with_router_interleave(
    config: MutableMapping[str, Any],
    cli: argparse.Namespace,
    branch_path: Path,
    *,
    dry_run: bool,
    validate_paths: bool,
) -> None:
    """Replay final-base 1A→1B.5, router A4, then the A5-enabled 1C leg."""
    before_router = copy.deepcopy(config)
    before_shared = before_router["shared_args"]["final_anytext"]
    before_stage = before_router["stages"]["final_base"]
    before_shared["router_aux_enabled"] = False
    before_stage["args"]["phase_1c_epochs"] = 0
    before_stage["post_export"] = False
    run_stage_process(
        before_router,
        cli,
        "final_base",
        branch_path,
        dry_run=dry_run,
        validate_paths=validate_paths,
        write_completion=False,
    )

    print("\n" + "=" * 88)
    print("[stage] router_utility_a4")
    print("=" * 88)
    run_router_utility(config, cli, force_dry_run=dry_run)

    phase_1c = copy.deepcopy(config)
    phase_stage = phase_1c["stages"]["final_base"]
    phase_stage["args"].update(
        {
            "start_phase": "1c",
            "phase_1a_epochs": 0,
            "phase_1b_epochs": 0,
            "phase_1b5_epochs": 0,
            "reset_tap_logits_uniform": False,
        }
    )
    phase_stage["implant_checkpoint_override"] = join_output_path(
        stage_output_dir(config, "final_base"), "phase_1b5_best.pt"
    )
    run_stage_process(
        phase_1c,
        cli,
        "final_base",
        branch_path,
        dry_run=dry_run,
        validate_paths=validate_paths,
        allow_existing=True,
        write_completion=True,
    )


def selected_stages(config: Mapping[str, Any], cli: argparse.Namespace) -> list[str]:
    pipeline = require_mapping(config, "pipeline")
    start = cli.start_at or pipeline.get("start_at", STAGE_ORDER[0])
    stop = cli.stop_after or pipeline.get("stop_after", STAGE_ORDER[-1])
    if cli.only:
        start = stop = cli.only
    if start not in STAGE_ORDER or stop not in STAGE_ORDER:
        raise ConfigError(f"start/stop stages must be one of: {', '.join(STAGE_ORDER)}")
    start_index = STAGE_ORDER.index(str(start))
    stop_index = STAGE_ORDER.index(str(stop))
    if start_index > stop_index:
        raise ConfigError(f"start_at {start!r} occurs after stop_after {stop!r}")
    return list(STAGE_ORDER[start_index : stop_index + 1])


def run_training(config: MutableMapping[str, Any], cli: argparse.Namespace) -> None:
    enforce_historical_final_dataset_policy(config)
    validate_topology(config)
    validate_clevr_dataset_config(config)
    validate_local_args(config)
    branch_path = generated_branch_config(config)
    pipeline = require_mapping(config, "pipeline")
    dry_run = bool(cli.dry_run or pipeline.get("dry_run", False) or cli.validate_only)
    validate_paths = bool(pipeline.get("validate_external_paths", True))
    if cli.no_validate_paths:
        validate_paths = False

    selected = selected_stages(config, cli)
    print(f"[pipeline] stages: {' -> '.join(selected)}")
    global_cfg = require_mapping(config, "global")
    precision_cfg = require_mapping(global_cfg, "precision")
    print(f"[pipeline] architecture: {global_cfg['read_attention_architecture']}")
    print(
        f"[pipeline] READ_NULL={bool(global_cfg.get('read_null_enabled', False))} "
        f"insert_before_B{int(global_cfg.get('read_null_insert_block', 20))}"
    )
    print(
        f"[pipeline] legacy dtype={precision_cfg.get('legacy_dtype', 'auto')} "
        f"handwriting={bool(global_cfg.get('handwriting_enabled', False))} "
        f"final dtype={precision_cfg['autocast']}"
    )
    print(f"[pipeline] branch config: {branch_path}")
    print(f"[pipeline] config sha256: {config_digest(config)}")
    if any(stage in FINAL_STAGES for stage in selected):
        print(
            "[dataset policy] COCO-v2 + READ_NULL + continuous-text/math ImageNet + additive CLEVR property binding: "
            "historical TextCaps + enhanced trusted-reading COCO + ImageNet scattercode with continuous "
            "real-word scale/crop + procedural non-text twins; CLEVR is appended without renormalizing the "
            "base mix; source×trust multiplier preserved; derived TextCaps forbidden"
        )

    write_json(RUNTIME_DIR / "last_resolved_user_config.json", config)

    for stage in selected:
        if (
            stage == "final_base"
            and bool(pipeline.get("router_utility_between_final_phases", False))
        ):
            run_final_base_with_router_interleave(
                config,
                cli,
                branch_path,
                dry_run=dry_run,
                validate_paths=validate_paths,
            )
            continue
        run_stage_process(
            config,
            cli,
            stage,
            branch_path,
            dry_run=dry_run,
            validate_paths=validate_paths,
        )
    if cli.validate_only:
        print("[validate] configuration and selected command graph are valid")
    elif dry_run:
        print("[dry-run] no training or export process was launched")
    else:
        print("[done] selected pipeline completed")


def recover_all_weights(config: Mapping[str, Any], cli: argparse.Namespace) -> None:
    recovery_cfg = require_mapping(
        require_mapping(config, "recovery"), "all_weights_saved_best"
    )
    out_dir = stage_output_dir(config, "all_weights")
    args = {
        "out_dir": out_dir,
        "phase": recovery_cfg.get("phase", "1c"),
        "overwrite": bool(recovery_cfg.get("overwrite", True)),
    }
    command = [
        python_executable(config),
        str(CODE_DIR / "recover_final_train_step2_from_saved_best.py"),
        *args_to_cli(args),
    ]
    dry_run = bool(cli.dry_run or require_mapping(config, "pipeline").get("dry_run", False))
    run_command(
        command,
        cwd=PROJECT_ROOT,
        env=stage_environment("all_weights"),
        dry_run=dry_run,
    )
    artifacts = artifact_paths(config)
    export_cmd, _, _, export_cwd, export_env = export_command(
        config,
        "final_model",
        artifacts["all_weights_merged"],
        artifacts["all_weights_ordinary"],
    )
    run_command(
        export_cmd,
        cwd=export_cwd,
        env=export_env,
        dry_run=dry_run,
    )



def run_router_utility(
    config: Mapping[str, Any],
    cli: argparse.Namespace,
    *,
    force_dry_run: bool = False,
) -> None:
    """Run router-only Experiment A in the existing final/sigmoid_all/base folder."""
    config_path = cli.config.expanduser().resolve()
    command = [
        python_executable(config),
        str(ROUTER_UTILITY_SCRIPT),
        "--config",
        str(config_path),
    ]
    if cli.validate_only:
        command.append("--validate-only")
    if cli.dry_run or force_dry_run:
        print("[dry-run] " + shlex.join(command))
        return
    run_command(
        command,
        cwd=PROJECT_ROOT,
        env=stage_environment("final_base"),
        dry_run=False,
    )

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CLIP hard-text curriculum from one JSON configuration."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-at", choices=STAGE_ORDER)
    parser.add_argument("--stop-after", choices=STAGE_ORDER)
    parser.add_argument("--only", choices=STAGE_ORDER)
    parser.add_argument("--action", choices=("train", "router_utility", "recover_all_weights_saved_best"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--no-validate-paths", action="store_true")
    parser.add_argument("--list-stages", action="store_true")
    return parser


def main() -> int:
    cli = build_cli().parse_args()
    if cli.list_stages:
        print("\n".join(STAGE_ORDER))
        return 0
    config_path = cli.config.expanduser().resolve()
    config = load_json(config_path)
    action = cli.action or require_mapping(config, "pipeline").get("action", "train")
    try:
        if action == "train":
            run_training(config, cli)
        elif action == "router_utility":
            run_router_utility(config, cli)
        elif action == "recover_all_weights_saved_best":
            recover_all_weights(config, cli)
        else:
            raise ConfigError(f"Unknown pipeline action: {action}")
    except subprocess.CalledProcessError as exc:
        print(f"[failed] subprocess returned exit code {exc.returncode}", file=sys.stderr)
        return int(exc.returncode or 1)
    except ConfigError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
