#!/usr/bin/env python3
"""Create a ~25%-of-full CLIP Cross-Attn MUX integration-training config.

This is deliberately *not* the tiny gradient smoke test.  It keeps the user's
prepared dataset/model paths, full dataset breadth, real batch sizes, precision,
optimizer families, requires-grad policy, normal checkpoint handoffs, and normal
stage ordering.  It reduces optimization exposure to a configurable fraction
(default 0.25) and trims validation overhead proportionally.

The intended use is a medium-cost release confidence run: long enough for the
all-weights backbone to get through its known rough startup, but far shorter than
retraining the model.

Benchmark policy:
  * final_base: OFF, so SCAM/RTA/MVT are not repeatedly evaluated per phase
  * all_weights: ON only at the final all-weights epoch
  * benchmark checkpoint selection: always OFF

The input should be the prepared ``training_config.local.json`` produced by
``prepare_training_data.py`` and optionally patched by ``prepare_objectnet_mvt.py``.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, MutableMapping

WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON must be an object: {path}")
    return data


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def require_dict(parent: Mapping[str, Any], key: str) -> MutableMapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object at {key!r}")
    return value


def is_windows_path(text: str) -> bool:
    return bool(WINDOWS_ABS.match(text)) or text.startswith("\\\\")


def suffix_path(root: str, suffix: str) -> str:
    text = str(root).rstrip("/\\")
    if is_windows_path(text):
        p = PureWindowsPath(text)
        return str(p.with_name(p.name + suffix))
    p = PurePosixPath(text)
    return str(p.with_name(p.name + suffix))


def cfg_join(root: str, *parts: str) -> str:
    """Join config paths while preserving Windows path semantics."""
    text = str(root)
    if is_windows_path(text):
        p = PureWindowsPath(text)
        for part in parts:
            p = p / PureWindowsPath(str(part))
        return str(p)
    p = PurePosixPath(text)
    for part in parts:
        p = p / PurePosixPath(str(part))
    return str(p)


def scale_positive(value: int, fraction: float, *, minimum: int = 1) -> int:
    value = int(value)
    if value <= 0:
        return value
    return max(minimum, int(math.ceil(value * fraction)))


def scale_optional(mapping: MutableMapping[str, Any], key: str, fraction: float, *, minimum: int = 1) -> None:
    if key not in mapping or mapping[key] in (None, ""):
        return
    mapping[key] = scale_positive(int(mapping[key]), fraction, minimum=minimum)


def assert_prepared(config: Mapping[str, Any]) -> None:
    paths = require_dict(config, "paths")
    final = require_dict(require_dict(config, "shared_args"), "final_anytext")
    datasets = config.get("datasets", {})

    required = {
        "imagenet_root",
        "imagenet_wnid_json",
        "overlay_selection",
        "imagenet_text_root",
        "imagenet_handwriting_root",
        "textcaps_root",
        "coco_root",
        "coco_train_json",
        "coco_train_gpt_json",
        "coco_train_reading_json",
        "coco_val_json",
        "coco_val_gpt_json",
        "salt_n_pepper_root",
    }
    if isinstance(datasets, Mapping):
        clevr = datasets.get("clevr_property_binding")
        if isinstance(clevr, Mapping) and bool(clevr.get("include", False)):
            required.update({"clevr_train_root", "clevr_val_root", "clevr_metadata_jsonl"})
    if bool(final.get("benchmark_include_mvt", False)):
        required.update({"mvt_csv", "mvt_image_root"})

    missing = sorted(k for k in required if paths.get(k) in (None, ""))
    if missing:
        raise ValueError(
            "Input config is not fully prepared. Missing: " + ", ".join(missing)
            + ". Use training_config.local.json after data preparation."
        )


def make_intermediate_config(
    source: Mapping[str, Any],
    *,
    fraction: float,
    train_root: str | None,
    benchmark_items: int,
    diagnostic_every: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = copy.deepcopy(dict(source))
    assert_prepared(cfg)

    pct = fraction * 100.0
    notes = cfg.setdefault("notes", {})
    if isinstance(notes, dict):
        notes["intermediate_smoke_test"] = (
            f"Generated from the prepared local config at approximately {pct:.1f}% of full optimization exposure. "
            "This is an integration-training confidence run, not a quality-comparable training result."
        )
        notes["intermediate_smoke_test_invariants"] = (
            "Dataset/model paths, full dataset breadth, batch sizes, AMP/precision, optimizer families, learning rates, "
            "requires-grad policy, stage ordering, and checkpoint/export semantics are preserved. Optimization horizons "
            "and validation workload are reduced. Benchmark datasets remain evaluation-only."
        )

    pipeline = require_dict(cfg, "pipeline")
    pipeline.update({
        "action": "train",
        "start_at": "soft_token",
        "stop_after": "all_weights",
        "existing_output_policy": "error",
        "validate_external_paths": True,
        "dry_run": False,
        "router_utility_between_final_phases": True,
    })

    global_cfg = require_dict(cfg, "global")
    original_root = str(global_cfg.get("train_root") or "outputs/clip_xattn_training")
    tag = int(round(pct))
    global_cfg["train_root"] = str(train_root) if train_root else suffix_path(
        original_root, f"_intermediate_{tag}pct"
    )

    # Keep real precision policy; enable sparse update diagnostics for evidence of motion.
    precision = global_cfg.get("precision")
    if isinstance(precision, dict):
        diagnostics = precision.get("diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["enabled"] = True
            diagnostics["log_every_optimizer_steps"] = int(diagnostic_every)
            diagnostics["save_plots"] = False

    shared = require_dict(cfg, "shared_args")
    legacy = require_dict(shared, "legacy_hard_text")
    final = require_dict(shared, "final_anytext")

    # Full data breadth: do not set max_classes, max_train_batches, tiny image counts, etc.
    # Only trim validation/telemetry cost.
    scale_optional(legacy, "val_read_examples", fraction)
    legacy["smoke"] = False

    scale_optional(final, "val_batches", fraction)
    final["select_by_benchmark_score"] = False
    final["benchmark_max_items"] = int(benchmark_items)
    # benchmark_every_epochs is set after we know the all-weights epoch count.
    if "read_null_save_plots" in final:
        final["read_null_save_plots"] = False
    scale_optional(final, "math_curriculum_val_batches", fraction)
    scale_optional(final, "router_aux_val_batches", fraction)

    datasets = cfg.get("datasets")
    if isinstance(datasets, dict):
        clevr = datasets.get("clevr_property_binding")
        if isinstance(clevr, dict):
            scale_optional(clevr, "val_batches", fraction)
            # Previews are diagnostic, not training signal; one is enough here.
            if "preview_count" in clevr:
                clevr["preview_count"] = min(int(clevr["preview_count"]), 1)

    stages = require_dict(cfg, "stages")
    schedule_report: dict[str, Any] = {"fraction": fraction}

    # Soft token: scale actual optimizer steps and warmup proportionally.
    soft = require_dict(require_dict(stages, "soft_token"), "args")
    old_soft_steps = int(soft.get("steps", 0))
    old_soft_warmup = int(soft.get("warmup_steps", 0))
    soft["steps"] = scale_positive(old_soft_steps, fraction)
    soft["warmup_steps"] = min(
        soft["steps"], scale_positive(old_soft_warmup, fraction) if old_soft_warmup > 0 else 0
    )
    if int(soft.get("eval_every", 0) or 0) > soft["steps"]:
        soft["eval_every"] = soft["steps"]
    soft["smoke"] = False
    schedule_report["soft_token"] = {"steps": [old_soft_steps, soft["steps"]]}

    # Legacy READ/CONTENT/JOINT are epoch-oriented. Quarter the epoch count, rounded up,
    # while keeping the full normal per-epoch dataset construction and batch size.
    for stage_name in ("read", "content", "joint"):
        stage = require_dict(stages, stage_name)
        stage["input_override"] = None
        stage["resume"] = None
        args = require_dict(stage, "args")
        old_epochs = int(args.get("epochs", 1))
        args["epochs"] = scale_positive(old_epochs, fraction)
        schedule_report[stage_name] = {"epochs": [old_epochs, args["epochs"]]}

    # Final-base: retain every phase and its original epoch count, but quarter steps/epoch.
    # This preserves all phase transitions and checkpoint handoffs at ~fraction of total updates.
    final_base = require_dict(stages, "final_base")
    final_base["input_override"] = None
    final_base["implant_checkpoint_override"] = None
    fb_args = require_dict(final_base, "args")
    old_fb_steps = int(fb_args.get("steps_per_epoch", 0))
    fb_args["steps_per_epoch"] = scale_positive(old_fb_steps, fraction)
    fb_args["run_benchmark_validation"] = False
    final_base["post_export"] = bool(final_base.get("post_export", True))
    fb_epoch_keys = ("phase_1a_epochs", "phase_1b_epochs", "phase_1b5_epochs", "phase_1c_epochs")
    fb_epochs = {k: int(fb_args.get(k, 0)) for k in fb_epoch_keys}
    schedule_report["final_base"] = {
        "steps_per_epoch": [old_fb_steps, fb_args["steps_per_epoch"]],
        "phase_epochs_preserved": fb_epochs,
        "full_updates_estimate": old_fb_steps * sum(fb_epochs.values()),
        "intermediate_updates_estimate": fb_args["steps_per_epoch"] * sum(fb_epochs.values()),
    }

    # Router A4: scale training horizon and warmup/hold scheduler proportionally.
    #
    # Changing global.train_root invalidates any explicit router paths copied from
    # training_config.local.json. A4 is intentionally required to live in this
    # run's EXISTING final/sigmoid_all/base folder, so rebase all coupled paths.
    router = require_dict(cfg, "router_utility")
    paths = require_dict(cfg, "paths")
    current_train_root = str(global_cfg["train_root"])
    fb_out = cfg_join(current_train_root, str(final_base["output_subdir"]))
    joint_stage = require_dict(stages, "joint")
    joint_out = cfg_join(current_train_root, str(joint_stage["output_subdir"]))
    router_prefix = str(router.get("output_prefix") or "router_utility_a4").strip()
    router.update({
        "output_dir": fb_out,
        "start_checkpoint": cfg_join(fb_out, "phase_1b5_best.pt"),
        "base_model_override": cfg_join(
            joint_out, "best_merged_state_dict__ungmp_oaiclip_fullmodel.pt"
        ),
    })
    paths["router_utility_a4_checkpoint"] = cfg_join(
        fb_out, f"{router_prefix}_best.pt"
    )

    old_router_steps = int(router.get("steps", 0))
    new_router_steps = scale_positive(old_router_steps, fraction, minimum=2)
    router["steps"] = new_router_steps
    if int(router.get("warmup_steps", 0) or 0) > 0:
        router["warmup_steps"] = scale_positive(int(router["warmup_steps"]), fraction)
    if int(router.get("lr_hold_steps", 0) or 0) > 0:
        router["lr_hold_steps"] = scale_positive(int(router["lr_hold_steps"]), fraction)
        router["lr_hold_steps"] = max(int(router.get("warmup_steps", 1)), int(router["lr_hold_steps"]))
        router["lr_hold_steps"] = min(router["lr_hold_steps"], new_router_steps - 1)
    scale_optional(router, "val_batches", fraction)
    if int(router.get("eval_every", 0) or 0) > new_router_steps:
        router["eval_every"] = new_router_steps
    schedule_report["router_utility"] = {"steps": [old_router_steps, new_router_steps]}

    # All-weights: keep all epochs, quarter steps/epoch.  This gives ~25% of total full-model
    # optimizer attempts while still exercising epoch boundaries/checkpoint writes.
    all_weights = require_dict(stages, "all_weights")
    all_weights["input_override"] = None
    all_weights["implant_checkpoint_override"] = None
    aw_args = require_dict(all_weights, "args")
    old_aw_steps = int(aw_args.get("steps_per_epoch", 0))
    old_aw_warmup = int(aw_args.get("sched_warmup_batches", 0) or 0)
    aw_args["steps_per_epoch"] = scale_positive(old_aw_steps, fraction)
    if old_aw_warmup > 0:
        aw_args["sched_warmup_batches"] = scale_positive(old_aw_warmup, fraction)
    aw_args["run_benchmark_validation"] = True
    aw_epochs = int(aw_args.get("phase_1c_epochs", 0))
    if aw_epochs <= 0:
        raise ValueError("all_weights.phase_1c_epochs must be > 0 for the intermediate test")
    # run_validation condition is epoch % N == 0 OR epoch == epochs; N=epochs gives exactly one run.
    final["benchmark_every_epochs"] = aw_epochs
    schedule_report["all_weights"] = {
        "steps_per_epoch": [old_aw_steps, aw_args["steps_per_epoch"]],
        "epochs_preserved": aw_epochs,
        "full_updates_estimate": old_aw_steps * aw_epochs,
        "intermediate_updates_estimate": aw_args["steps_per_epoch"] * aw_epochs,
        "sched_warmup_batches": [old_aw_warmup, int(aw_args.get("sched_warmup_batches", 0))],
    }

    expected_fb = cfg_join(str(global_cfg["train_root"]), str(final_base["output_subdir"]))
    expected_joint = cfg_join(
        str(global_cfg["train_root"]), str(require_dict(stages, "joint")["output_subdir"])
    )
    router_contract = {
        "router_utility.output_dir": (router.get("output_dir"), expected_fb),
        "router_utility.start_checkpoint": (
            router.get("start_checkpoint"), cfg_join(expected_fb, "phase_1b5_best.pt")
        ),
        "router_utility.base_model_override": (
            router.get("base_model_override"),
            cfg_join(expected_joint, "best_merged_state_dict__ungmp_oaiclip_fullmodel.pt"),
        ),
        "paths.router_utility_a4_checkpoint": (
            paths.get("router_utility_a4_checkpoint"),
            cfg_join(expected_fb, f"{router_prefix}_best.pt"),
        ),
    }
    bad_router_paths = [
        f"{name}: got={got!r}, expected={expected!r}"
        for name, (got, expected) in router_contract.items()
        if str(got).lower() != str(expected).lower()
    ]
    if bad_router_paths:
        raise AssertionError("Router path rebase failed:\n  - " + "\n  - ".join(bad_router_paths))

    if bool(final.get("select_by_benchmark_score")):
        raise AssertionError("benchmark checkpoint selection must remain disabled")
    if bool(fb_args.get("run_benchmark_validation")):
        raise AssertionError("final_base benchmark must be disabled")
    if not bool(aw_args.get("run_benchmark_validation")):
        raise AssertionError("all_weights benchmark must be enabled")

    return cfg, schedule_report


def _parser_option_strings(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    options: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        long_args = [
            arg.value for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        ]
        options.update(long_args)
        boolean_optional = any(
            kw.arg == "action"
            and isinstance(kw.value, ast.Attribute)
            and kw.value.attr == "BooleanOptionalAction"
            for kw in node.keywords
        )
        if boolean_optional:
            for option in long_args:
                if not option.startswith("--no-"):
                    options.add("--no-" + option[2:])
    return options


def validate_stage_parser_contract(project_root: Path, config: Mapping[str, Any]) -> None:
    launcher_path = project_root / "train_clip_xattn_bridge.py"
    spec = importlib.util.spec_from_file_location("_intermediate_launcher_contract", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import launcher: {launcher_path}")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    branch = launcher.generated_branch_config(config)
    failures: list[str] = []
    for stage in launcher.STAGE_ORDER:
        resolved_args, _ = launcher.build_stage_args(config, stage, branch)
        emitted = {
            token for token in launcher.args_to_cli(resolved_args)
            if isinstance(token, str) and token.startswith("--")
        }
        script = Path(launcher.SCRIPT_PATHS[stage])
        accepted = _parser_option_strings(script)
        missing = sorted(emitted - accepted)
        if missing:
            failures.append(f"{stage}: {', '.join(missing)}")
    if failures:
        raise RuntimeError(
            "Launcher emits option(s) not accepted by target trainer parser:\n  - "
            + "\n  - ".join(failures)
        )
    print("[intermediate-config] launcher -> trainer parser option contract: OK")


def validate_with_launcher(project_root: Path, output: Path) -> None:
    launcher = project_root / "train_clip_xattn_bridge.py"
    command = [
        sys.executable, str(launcher), "--config", str(output),
        "--validate-only", "--allow-existing",
    ]
    print("[intermediate-config] validating launcher command graph:")
    print("  " + subprocess.list2cmdline(command))
    subprocess.run(command, cwd=str(project_root), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("training_config.local.json"),
        help="Prepared user config; all resolved paths/revisions are copied from here",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("intermediate_smoke_test_training_config.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--fraction", type=float, default=0.25,
        help="Optimization fraction of the full configured run (default: 0.25)",
    )
    parser.add_argument(
        "--train-root", default=None,
        help="Output root. Default: normal train_root + _intermediate_25pct",
    )
    parser.add_argument(
        "--benchmark-items", type=int, default=0,
        help="Max items per benchmark subset for the one benchmark pass; 0 = full benchmark",
    )
    parser.add_argument(
        "--diagnostic-every", type=int, default=25,
        help="Precision/update diagnostic cadence in optimizer steps when supported",
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        parser.error("--fraction must be > 0 and <= 1")
    if args.benchmark_items < 0:
        parser.error("--benchmark-items must be >= 0")
    if args.diagnostic_every <= 0:
        parser.error("--diagnostic-every must be > 0")

    project_root = args.project_root.expanduser().resolve()
    source_path = args.config if args.config.is_absolute() else project_root / args.config
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    source = load_json(source_path)
    cfg, report = make_intermediate_config(
        source,
        fraction=float(args.fraction),
        train_root=args.train_root,
        benchmark_items=int(args.benchmark_items),
        diagnostic_every=int(args.diagnostic_every),
    )
    write_json(output_path, cfg)

    global_cfg = require_dict(cfg, "global")
    shared = require_dict(cfg, "shared_args")
    final = require_dict(shared, "final_anytext")

    print(f"[intermediate-config] source : {source_path}")
    print(f"[intermediate-config] output : {output_path}")
    print(f"[intermediate-config] train  : {global_cfg['train_root']}")
    print(f"[intermediate-config] fraction: {args.fraction:.3f} ({args.fraction*100:.1f}%)")
    print("[intermediate-config] schedule:")
    for name in ("soft_token", "read", "content", "joint", "final_base", "router_utility", "all_weights"):
        print(f"  {name:14s} {json.dumps(report[name], sort_keys=True)}")
    print(
        "[intermediate-config] benchmark: final_base=OFF; all_weights=ON only at final epoch; "
        f"max_items={args.benchmark_items} (0=full); MVT={bool(final.get('benchmark_include_mvt', False))}"
    )
    print(
        "[intermediate-config] preserved: full dataset breadth, real batch sizes, precision, optimizers, "
        "requires-grad rules, stage transitions, checkpoint/export semantics"
    )
    print(
        f"[intermediate-config] diagnostics: enabled every {args.diagnostic_every} optimizer step(s) where supported"
    )
    router_cfg = require_dict(cfg, "router_utility")
    print(f"[intermediate-config] router output : {router_cfg['output_dir']}")
    print(f"[intermediate-config] router start  : {router_cfg['start_checkpoint']}")
    print(f"[intermediate-config] router base   : {router_cfg['base_model_override']}")

    if not args.no_validate:
        validate_stage_parser_contract(project_root, cfg)
        validate_with_launcher(project_root, output_path)

    print("[intermediate-config] ready")
    print(
        f"[intermediate-config] run: {sys.executable} train_clip_xattn_bridge.py --config {output_path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
