#!/usr/bin/env python3
"""Create a tiny but real CLIP-MUX training smoke configuration.

The input should be the *prepared* ``training_config.local.json`` produced by
``prepare_training_data.py`` (and optionally patched by ``prepare_objectnet_mvt.py``).
The output preserves the user's resolved dataset/model paths, real batch sizes,
precision policy, optimizer families, learning rates, trainable-parameter logic,
and normal stage ordering.  It only shortens dataset construction, epoch/step
counts, validation, warmup horizons, and expensive plotting/export work.

The resulting run is intentionally a gradient/update smoke test rather than a
quality experiment.  Precision diagnostics are emitted on every optimizer step,
including sampled FP32 parameter deltas after the optimizer update.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
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


def cfg_join(root: str, *parts: str) -> str:
    if is_windows_path(root):
        return str(PureWindowsPath(root).joinpath(*parts))
    return str(PurePosixPath(root).joinpath(*parts))


def default_smoke_root(source: str) -> str:
    text = str(source).rstrip("/\\")
    if is_windows_path(text):
        p = PureWindowsPath(text)
        return str(p.with_name(p.name + "_smoke_test"))
    p = PurePosixPath(text)
    return str(p.with_name(p.name + "_smoke_test"))


def set_if_present(mapping: MutableMapping[str, Any], key: str, value: Any) -> None:
    if key in mapping:
        mapping[key] = value


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
            "Input config is not fully prepared. Missing: " + ", ".join(missing) +
            ". Use training_config.local.json after prepare_training_data.py."
        )


def make_smoke_config(
    source: Mapping[str, Any],
    *,
    train_root: str | None,
    soft_steps: int,
    legacy_batches: int,
    final_steps: int,
    all_weights_steps: int,
    router_steps: int,
    benchmark_items: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(source))
    assert_prepared(cfg)

    notes = cfg.setdefault("notes", {})
    if isinstance(notes, dict):
        notes["smoke_test"] = (
            "Generated from the prepared local config. Quality is meaningless: this run exists only to "
            "exercise every optimizer path, checkpoint handoff, router interleave, and one benchmark pass."
        )
        notes["smoke_test_invariants"] = (
            "User dataset/model paths, trainable-parameter rules, AMP/precision, real batch sizes, optimizer "
            "families, and main learning rates are preserved. Counts/steps/validation and warmup horizons are reduced."
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
    old_root = str(global_cfg.get("train_root") or "outputs/clip_xattn_training")
    smoke_root = str(train_root) if train_root else default_smoke_root(old_root)
    global_cfg["train_root"] = smoke_root

    # Keep the actual precision choices.  Only make diagnostics fire every real optimizer step.
    precision = require_dict(global_cfg, "precision")
    diagnostics = require_dict(precision, "diagnostics")
    diagnostics["enabled"] = True
    diagnostics["log_every_optimizer_steps"] = 1
    diagnostics["save_plots"] = False

    shared = require_dict(cfg, "shared_args")
    legacy = require_dict(shared, "legacy_hard_text")
    final = require_dict(shared, "final_anytext")

    # Do NOT enable the trainers' built-in --smoke mode: it changes batch size.
    legacy["smoke"] = False
    legacy["max_classes"] = min(int(legacy.get("max_classes", 0) or 12), 12)
    legacy["max_dog_classes"] = min(int(legacy.get("max_dog_classes", 24)), 4)
    legacy["train_images_per_class"] = min(int(legacy.get("train_images_per_class", 6)), 2)
    legacy["val_images_per_class"] = 1
    legacy["max_train_batches"] = int(legacy_batches)
    legacy["val_read_examples"] = min(int(legacy.get("val_read_examples", 512)), 16)
    legacy["num_content_templates"] = min(int(legacy.get("num_content_templates", 20)), 4)
    legacy["log_every"] = 1
    # batch_size, grad_accum_steps, AMP dtype, LRs, etc. intentionally untouched.

    final["log_every"] = 1
    final["val_batches"] = 1
    final["benchmark_every_epochs"] = 1
    final["benchmark_max_items"] = int(benchmark_items)
    final["select_by_benchmark_score"] = False
    final["save_every_epoch"] = False
    final["read_null_save_plots"] = False
    set_if_present(final, "math_curriculum_preview_count", 1)
    set_if_present(final, "math_curriculum_val_batches", 1)
    set_if_present(final, "router_aux_val_batches", 1)
    set_if_present(final, "router_aux_grad_log_every", 1)

    stages = require_dict(cfg, "stages")

    soft = require_dict(require_dict(stages, "soft_token"), "args")
    soft["smoke"] = False
    soft["train_examples_per_word"] = 1
    soft["seen_val_examples_per_word"] = 1
    soft["heldout_examples_per_word"] = 1
    soft["render_specs"] = 1
    soft["steps"] = int(soft_steps)
    soft["warmup_steps"] = 1
    soft["eval_every"] = int(soft_steps)
    # Keep batch_words untouched so the real configured soft-token batch is exercised.

    for stage_name in ("read", "content", "joint"):
        stage = require_dict(stages, stage_name)
        stage["input_override"] = None
        stage["resume"] = None
        require_dict(stage, "args")["epochs"] = 1

    final_base = require_dict(stages, "final_base")
    final_base["input_override"] = None
    final_base["implant_checkpoint_override"] = None
    fb_args = require_dict(final_base, "args")
    fb_args.update({
        "start_phase": "1a",
        "reset_tap_logits_uniform": True,
        # A static multi-phase invocation would benchmark once PER phase. Keep it off here.
        "run_benchmark_validation": False,
        "phase_1a_epochs": 1,
        "phase_1b_epochs": 1,
        "phase_1b5_epochs": 1,
        "phase_1c_epochs": 1,
        "steps_per_epoch": int(final_steps),
        "export_merged_each_phase": False,
    })
    final_base["post_export"] = False

    all_weights = require_dict(stages, "all_weights")
    all_weights["input_override"] = None
    all_weights["implant_checkpoint_override"] = None
    aw_args = require_dict(all_weights, "args")
    aw_args.update({
        "start_phase": "1c",
        "run_benchmark_validation": True,  # exactly once: one all-weights epoch
        "phase_1a_epochs": 0,
        "phase_1b_epochs": 0,
        "phase_1b5_epochs": 0,
        "phase_1c_epochs": 1,
        "steps_per_epoch": int(all_weights_steps),
        "save_full_every_epoch": False,
        "grad_log_every": 1,
        # Compress only the horizon so the real peak LR is actually reached during the tiny run.
        "sched_warmup_batches": 1,
    })
    all_weights["post_export"] = False

    # Keep normal CLEVR training semantics but reduce previews/validation overhead.
    datasets = cfg.get("datasets")
    if isinstance(datasets, dict):
        clevr = datasets.get("clevr_property_binding")
        if isinstance(clevr, dict):
            set_if_present(clevr, "val_batches", 1)
            set_if_present(clevr, "preview_count", 1)

    # Router A4 is a real optimizer stage between 1B.5 and 1C. Repoint every checkpoint
    # reference into the smoke tree; otherwise a smoke config can accidentally read/write
    # the normal experiment directory.
    fb_out = cfg_join(smoke_root, str(final_base["output_subdir"]))
    paths = require_dict(cfg, "paths")
    paths["router_utility_a4_checkpoint"] = cfg_join(fb_out, "router_utility_a4_best.pt")

    router = require_dict(cfg, "router_utility")
    joint_out = cfg_join(smoke_root, str(require_dict(stages, "joint")["output_subdir"]))
    router.update({
        "start_checkpoint": cfg_join(fb_out, "phase_1b5_best.pt"),
        "output_dir": fb_out,
        "base_model_override": cfg_join(
            joint_out, "best_merged_state_dict__ungmp_oaiclip_fullmodel.pt"
        ),
        "overwrite": True,
        "steps": int(router_steps),
        "warmup_steps": 1,
        # Router scheduler requires hold >= warmup and hold < total steps.
        "lr_hold_steps": 1,
        "log_every": 1,
        "eval_every": int(router_steps),
        "val_batches": 1,
    })

    # Safety: smoke benchmarks are monitors only and occur only in all_weights.
    if bool(fb_args.get("run_benchmark_validation")):
        raise AssertionError("final_base benchmark must be disabled in smoke config")
    if not bool(aw_args.get("run_benchmark_validation")) or int(aw_args["phase_1c_epochs"]) != 1:
        raise AssertionError("all_weights must contain the single smoke benchmark epoch")
    if bool(final.get("select_by_benchmark_score")):
        raise AssertionError("benchmark checkpoint selection must remain disabled")

    return cfg



def _parser_option_strings(path: Path) -> set[str]:
    """Statically collect long argparse option strings, including BooleanOptionalAction negatives."""
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
    """Catch launcher→subprocess CLI drift without importing GPU/dependency-heavy trainers."""
    launcher_path = project_root / "train_clip_xattn_bridge.py"
    spec = importlib.util.spec_from_file_location("_smoke_launcher_contract", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import launcher for contract check: {launcher_path}")
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
            "Launcher emits option(s) not accepted by the target trainer parser:\n  - "
            + "\n  - ".join(failures)
        )
    print("[smoke-config] launcher -> trainer parser option contract: OK")

def validate_with_launcher(project_root: Path, output: Path) -> None:
    launcher = project_root / "train_clip_xattn_bridge.py"
    if not launcher.is_file():
        raise FileNotFoundError(f"Launcher not found: {launcher}")
    command = [sys.executable, str(launcher), "--config", str(output), "--validate-only", "--allow-existing"]
    print("[smoke-config] validating launcher command graph:")
    print("  " + subprocess.list2cmdline(command))
    subprocess.run(command, cwd=str(project_root), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training_config.local.json"),
                        help="Prepared user config; dataset/model paths are copied from here")
    parser.add_argument("--output", type=Path, default=Path("smoke_test_training_config.json"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--train-root", default=None,
                        help="Smoke output root. Default: normal train_root with _smoke_test suffix")
    parser.add_argument("--soft-steps", type=int, default=2)
    parser.add_argument("--legacy-batches", type=int, default=2)
    parser.add_argument("--final-steps", type=int, default=2)
    parser.add_argument("--all-weights-steps", type=int, default=2)
    parser.add_argument("--router-steps", type=int, default=2)
    parser.add_argument("--benchmark-items", type=int, default=8,
                        help="Maximum items per typo/MVT benchmark subset for the single smoke benchmark pass")
    parser.add_argument("--no-validate", action="store_true",
                        help="Write config without running launcher --validate-only")
    args = parser.parse_args()

    for name in ("soft_steps", "legacy_batches", "final_steps", "all_weights_steps", "benchmark_items"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if int(args.router_steps) < 2:
        parser.error("--router-steps must be >= 2 so the real warmup/hold scheduler can be exercised")

    project_root = args.project_root.expanduser().resolve()
    source_path = args.config if args.config.is_absolute() else project_root / args.config
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    source = load_json(source_path)
    smoke = make_smoke_config(
        source,
        train_root=args.train_root,
        soft_steps=args.soft_steps,
        legacy_batches=args.legacy_batches,
        final_steps=args.final_steps,
        all_weights_steps=args.all_weights_steps,
        router_steps=args.router_steps,
        benchmark_items=args.benchmark_items,
    )
    write_json(output_path, smoke)

    g = require_dict(smoke, "global")
    s = require_dict(smoke, "shared_args")
    stages = require_dict(smoke, "stages")
    print(f"[smoke-config] source : {source_path}")
    print(f"[smoke-config] output : {output_path}")
    print(f"[smoke-config] train  : {g['train_root']}")
    print(
        "[smoke-config] preserved real execution settings: "
        f"legacy batch={require_dict(s, 'legacy_hard_text').get('batch_size')} | "
        f"final logical batch={require_dict(require_dict(stages, 'final_base'), 'args').get('logical_batch_images')} | "
        f"all-weights logical batch={require_dict(require_dict(stages, 'all_weights'), 'args').get('logical_batch_images')} | "
        f"legacy AMP={require_dict(g, 'precision').get('legacy_dtype')} | "
        f"final AMP={require_dict(g, 'precision').get('autocast')}"
    )
    print(
        "[smoke-config] tiny schedule: soft="
        f"{args.soft_steps} steps; read/content/joint=1 epoch x <= {args.legacy_batches} batches; "
        f"final phases=1 epoch x {args.final_steps} steps; router={args.router_steps} steps; "
        f"all_weights=1 epoch x {args.all_weights_steps} steps"
    )
    print(
        "[smoke-config] benchmark: final_base=OFF; all_weights=ON for its single epoch; "
        f"max_items={args.benchmark_items}; MVT="
        f"{bool(require_dict(s, 'final_anytext').get('benchmark_include_mvt', False))}"
    )
    print("[smoke-config] precision parameter-update diagnostics: EVERY optimizer step")

    if not args.no_validate:
        validate_stage_parser_contract(project_root, smoke)
        validate_with_launcher(project_root, output_path)

    print("[smoke-config] ready")
    print(f"[smoke-config] run: {sys.executable} train_clip_xattn_bridge.py --config {output_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
