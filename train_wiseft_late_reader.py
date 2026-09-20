#!/usr/bin/env python3
"""WiSE-FT interpolation for the B20/B21 late READ / READ_NULL reader.

This file does exactly three things:

    pareto    adaptive alpha sweep + constrained Pareto selection
    boundary  monotonic boundary scan + one-step-back selection
    export    save explicit alpha checkpoints without evaluation

The simple zero-shot image benchmark is intentionally not embedded here; it lives in
``eval_wiseft_late_reader.py``.  The SCAM/RTA helper used for alpha selection remains an
external evaluation adapter because it owns the benchmark dataset/label logic.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


# -------------------------------------------------------------------------------------------------
# Project defaults: one name for each thing, once.
# -------------------------------------------------------------------------------------------------

DEFAULT_ORIGINAL_MODEL = Path("path/to/hf/exported/full_xattn_model")
DEFAULT_TRAINED_CONTROL = Path("path/to/refit/trained/model")
DEFAULT_EVAL_HELPER = Path("eval_wiseft_late_reader.py")

EXPECTED_TAPS = (20, 21)
SUBSET_ORDER = ("NoSCAM", "SCAM", "SynthSCAM", "NoRTA", "RTA", "SynthRTA")
NO_ATTACK_SUBSETS = ("NoSCAM", "NoRTA")
ATTACKED_READ_SUBSETS = ("SCAM", "SynthSCAM", "RTA", "SynthRTA")
BOUNDARY_SCAN_ORDER = ("SCAM", "RTA", "SynthSCAM", "SynthRTA")

WISE_EXACT_NAMES = {
    "read_implant.read_tap_logits",
    "read_implant.glyph_bias_beta",
    "read_implant.null_abstain_weight",
}
WISE_PREFIX = "read_implant.read_bridge."


# -------------------------------------------------------------------------------------------------
# Small utilities
# -------------------------------------------------------------------------------------------------


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def alpha_grid(start: float, stop: float, step: float, *, include_stop: bool = True) -> list[float]:
    if step <= 0:
        raise ValueError("alpha step must be > 0")
    if not 0.0 <= start <= stop <= 1.0:
        raise ValueError("require 0 <= start <= stop <= 1")

    values: list[float] = []
    i = 0
    while True:
        value = start + i * step
        if value > stop + 1.0e-10:
            break
        values.append(round(value, 10))
        i += 1

    if include_stop and (not values or abs(values[-1] - stop) > 1.0e-9):
        values.append(round(stop, 10))
    return sorted(set(values))


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Use --overwrite-output or choose another --output-root."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def load_eval_helper(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation helper not found: {path}\n"
            "Pass --eval-helper with the benchmark helper path."
        )

    spec = importlib.util.spec_from_file_location("late_read_eval_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import evaluation helper: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -------------------------------------------------------------------------------------------------
# WiSE endpoint handling
# -------------------------------------------------------------------------------------------------


def tap_tuple(model: Any) -> tuple[int, ...]:
    implant = model.read_implant
    if hasattr(implant, "_block_list"):
        return tuple(int(x) for x in implant._block_list(implant.tap_blocks))
    return tuple(int(x) for x in implant.tap_blocks.detach().cpu().reshape(-1).tolist())


def state_tuple(model: Any) -> tuple[int, ...]:
    implant = model.read_implant
    value = getattr(implant, "read_state_blocks", None)
    if value is None:
        return tap_tuple(model)
    if torch.is_tensor(value):
        return tuple(int(x) for x in value.detach().cpu().reshape(-1).tolist())
    return tuple(int(x) for x in value)


def validate_reader_topology(model: Any, label: str) -> None:
    taps = tap_tuple(model)
    states = state_tuple(model)
    if taps != EXPECTED_TAPS:
        raise RuntimeError(f"{label}: tap_blocks={taps}, expected {EXPECTED_TAPS}")
    if states != EXPECTED_TAPS:
        raise RuntimeError(f"{label}: READ states={states}, expected {EXPECTED_TAPS}")


def wise_parameter_names(model: Any) -> list[str]:
    names = sorted(
        name
        for name, _ in model.named_parameters()
        if name.startswith(WISE_PREFIX) or name in WISE_EXACT_NAMES
    )

    missing = sorted(WISE_EXACT_NAMES - set(names))
    if missing:
        raise RuntimeError(f"Missing expected WiSE parameters: {missing}")
    if not any(name.startswith(WISE_PREFIX) for name in names):
        raise RuntimeError("No read_bridge parameters found")
    return names


@dataclass
class Endpoint:
    path: Path
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any]


@torch.no_grad()
def load_endpoint(model_path: Path, expected_names: Sequence[str] | None = None) -> Endpoint:
    print(f"[endpoint] loading on CPU: {model_path}")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().cpu()
    validate_reader_topology(model, model_path.name or str(model_path))

    names = wise_parameter_names(model)
    if expected_names is not None and names != list(expected_names):
        expected = set(expected_names)
        actual = set(names)
        raise RuntimeError(
            "WiSE parameter-name mismatch between endpoints.\n"
            f"Missing here:    {sorted(expected - actual)}\n"
            f"Unexpected here: {sorted(actual - expected)}"
        )

    params = dict(model.named_parameters())
    tensors = {name: params[name].detach().float().cpu().clone() for name in names}
    metadata = {
        "model_path": str(model_path),
        "tap_blocks": list(tap_tuple(model)),
        "read_state_blocks": list(state_tuple(model)),
        "wise_parameter_count": len(names),
        "wise_scalar_count": int(sum(t.numel() for t in tensors.values())),
    }

    del params, model
    gc.collect()
    return Endpoint(path=model_path, tensors=tensors, metadata=metadata)


@dataclass
class WiseContext:
    original: Endpoint
    trained: Endpoint
    model: Any
    processor: Any
    parameter_names: list[str]
    device: torch.device


@torch.no_grad()
def load_wise_context(
    original_model: Path,
    trained_control: Path,
    device: torch.device,
) -> WiseContext:
    original = load_endpoint(original_model)
    parameter_names = sorted(original.tensors)

    print(f"[working] loading trained control on {device}: {trained_control}")
    model = AutoModel.from_pretrained(trained_control, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(trained_control, trust_remote_code=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    validate_reader_topology(model, "trained control")
    trained_names = wise_parameter_names(model)
    if trained_names != parameter_names:
        raise RuntimeError(
            "Original/trained WiSE parameter lists differ.\n"
            f"original-only={sorted(set(parameter_names) - set(trained_names))}\n"
            f"trained-only={sorted(set(trained_names) - set(parameter_names))}"
        )

    params = dict(model.named_parameters())
    trained_tensors = {
        name: params[name].detach().float().cpu().clone() for name in parameter_names
    }
    trained = Endpoint(
        path=trained_control,
        tensors=trained_tensors,
        metadata={
            "model_path": str(trained_control),
            "tap_blocks": list(tap_tuple(model)),
            "read_state_blocks": list(state_tuple(model)),
            "wise_parameter_count": len(parameter_names),
            "wise_scalar_count": int(sum(t.numel() for t in trained_tensors.values())),
        },
    )

    print(
        f"[WiSE] interpolating {len(parameter_names)} tensors / "
        f"{sum(t.numel() for t in original.tensors.values()):,} scalars"
    )
    return WiseContext(
        original=original,
        trained=trained,
        model=model,
        processor=processor,
        parameter_names=parameter_names,
        device=device,
    )


@torch.no_grad()
def apply_alpha(context: WiseContext, alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    original = context.original.tensors
    trained = context.trained.tensors
    if set(original) != set(trained):
        raise RuntimeError("Original/trained WiSE tensor-key mismatch")

    params = dict(context.model.named_parameters())
    missing = sorted(set(original) - set(params))
    if missing:
        raise RuntimeError(f"Working model is missing WiSE parameters: {missing}")

    for name in context.parameter_names:
        dst = params[name]
        a = original[name]
        b = trained[name]
        if a.shape != b.shape or tuple(dst.shape) != tuple(a.shape):
            raise RuntimeError(
                f"Shape mismatch for {name}: original={tuple(a.shape)} "
                f"trained={tuple(b.shape)} working={tuple(dst.shape)}"
            )
        dst.copy_(torch.lerp(a, b, float(alpha)).to(device=dst.device, dtype=dst.dtype))


def save_full_checkpoint(
    context: WiseContext,
    output_dir: Path,
    metadata: Mapping[str, Any],
    *,
    note: str,
) -> None:
    """Save a complete local HF repo, preserving the trained-control custom code/assets."""
    if output_dir.exists():
        raise FileExistsError(output_dir)

    if context.trained.path.is_dir():
        shutil.copytree(context.trained.path, output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    context.model.save_pretrained(output_dir, safe_serialization=True)
    context.processor.save_pretrained(output_dir)
    write_json(output_dir / "WISE_FT_METADATA.json", metadata)
    (output_dir / "TEMPORARY_WISE_FT_MODEL.txt").write_text(note, encoding="utf-8")


# -------------------------------------------------------------------------------------------------
# Benchmark adapter
# -------------------------------------------------------------------------------------------------


@dataclass
class BenchmarkContext:
    helper: Any
    sample_sets: Mapping[str, Sequence[Any]]
    labels: Sequence[str]
    input_ids: torch.Tensor
    batch_size: int
    amp: bool


def load_benchmark_context(
    context: WiseContext,
    eval_helper: Path,
    batch_size: int,
    amp: bool,
) -> BenchmarkContext:
    helper = load_eval_helper(eval_helper.resolve())
    sample_sets = helper.load_samples()
    labels = helper.label_vocabulary(sample_sets)
    input_ids = helper.encode_label_prompts(context.processor, labels, context.device)

    print(
        f"[dataset] labels={len(labels)} "
        + " ".join(f"{subset}={len(sample_sets[subset])}" for subset in SUBSET_ORDER)
    )
    return BenchmarkContext(
        helper=helper,
        sample_sets=sample_sets,
        labels=labels,
        input_ids=input_ids,
        batch_size=batch_size,
        amp=amp,
    )


@dataclass
class AlphaResult:
    alpha: float
    summaries: dict[str, dict[str, Any]]

    def subset_accuracy(self, subset: str) -> float:
        return float(self.summaries[subset]["binary_accuracy"])

    def weighted_accuracy(self, subsets: Sequence[str]) -> float:
        numerator = 0.0
        denominator = 0
        for subset in subsets:
            row = self.summaries[subset]
            count = int(row["count"])
            numerator += float(row["binary_accuracy"]) * count
            denominator += count
        return numerator / denominator if denominator else float("nan")

    def weighted_margin(self, subsets: Sequence[str]) -> float:
        numerator = 0.0
        denominator = 0
        for subset in subsets:
            row = self.summaries[subset]
            count = int(row["count"])
            numerator += float(row["mean_binary_logit_margin"]) * count
            denominator += count
        return numerator / denominator if denominator else float("nan")


@torch.inference_mode()
def score_subset(
    context: WiseContext,
    benchmark: BenchmarkContext,
    subset: str,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = benchmark.sample_sets[subset]
    logits = benchmark.helper.score_subset(
        context.model,
        context.processor,
        samples,
        benchmark.input_ids,
        mode=mode,
        batch_size=benchmark.batch_size,
        device=context.device,
        amp=benchmark.amp,
    )
    summary, records = benchmark.helper.evaluate_logits(
        samples,
        benchmark.labels,
        logits,
        mode=mode,
    )
    summary["subset"] = subset
    del logits
    return summary, records


@torch.inference_mode()
def evaluate_read_alpha(
    context: WiseContext,
    benchmark: BenchmarkContext,
    alpha: float,
    subsets: Sequence[str] = SUBSET_ORDER,
) -> AlphaResult:
    apply_alpha(context, alpha)
    summaries: dict[str, dict[str, Any]] = {}

    print()
    print(f"[alpha={alpha:.6f}] READ sweep")
    for subset in subsets:
        summary, _ = score_subset(context, benchmark, subset, mode="read")
        summaries[subset] = summary
        print(
            f"  {subset:<11s} acc={summary['binary_accuracy']:.4f} "
            f"margin={summary['mean_binary_logit_margin']:+.4f} "
            f"NO_TEXT={summary['no_text_detected_count']}"
        )
    return AlphaResult(alpha=float(alpha), summaries=summaries)


@torch.inference_mode()
def evaluate_full_selected(
    context: WiseContext,
    benchmark: BenchmarkContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for subset in SUBSET_ORDER:
        for mode in ("any", "read"):
            summary, subset_records = score_subset(context, benchmark, subset, mode)
            summaries.append(summary)
            records.extend(subset_records)
    return summaries, records


def print_full_benchmark(
    benchmark: BenchmarkContext,
    alpha: float,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    for mode, title in (("any", "Object recognition"), ("read", "Attack-text reading")):
        group = [row for row in summaries if row["mode"] == mode]
        print()
        print(benchmark.helper.format_console_table(f"WiSE alpha={alpha:.4f}", title, group))


# -------------------------------------------------------------------------------------------------
# Pareto selection
# -------------------------------------------------------------------------------------------------


def result_row(
    result: AlphaResult,
    baseline: AlphaResult,
    max_read_drop_pp: float,
) -> dict[str, Any]:
    attacked = result.weighted_accuracy(ATTACKED_READ_SUBSETS)
    nominal_null = result.weighted_accuracy(NO_ATTACK_SUBSETS)
    baseline_attacked = baseline.weighted_accuracy(ATTACKED_READ_SUBSETS)

    per_subset_drops = {
        subset: 100.0 * (baseline.subset_accuracy(subset) - result.subset_accuracy(subset))
        for subset in ATTACKED_READ_SUBSETS
    }
    worst_subset = max(per_subset_drops, key=per_subset_drops.get)
    worst_drop_pp = per_subset_drops[worst_subset]

    row: dict[str, Any] = {
        "alpha": float(result.alpha),
        "attacked_read_accuracy_weighted": attacked,
        "attacked_read_drop_weighted_pp": 100.0 * (baseline_attacked - attacked),
        "nominal_noattack_null_accuracy_weighted": nominal_null,
        "attacked_read_margin_weighted": result.weighted_margin(ATTACKED_READ_SUBSETS),
        "nominal_noattack_null_margin_weighted": result.weighted_margin(NO_ATTACK_SUBSETS),
        "max_attacked_subset_drop_pp": worst_drop_pp,
        "max_drop_subset": worst_subset,
        "feasible_max_drop": bool(worst_drop_pp <= max_read_drop_pp + 1.0e-9),
    }

    for subset in SUBSET_ORDER:
        row[f"{subset}_accuracy"] = result.subset_accuracy(subset)
        row[f"{subset}_margin"] = float(result.summaries[subset]["mean_binary_logit_margin"])
        row[f"{subset}_no_text"] = int(result.summaries[subset]["no_text_detected_count"])
    for subset in ATTACKED_READ_SUBSETS:
        row[f"{subset}_drop_pp_vs_alpha0"] = per_subset_drops[subset]
    return row


def rows_from_cache(
    cache: Mapping[float, AlphaResult],
    baseline: AlphaResult,
    max_read_drop_pp: float,
) -> list[dict[str, Any]]:
    return sorted(
        [result_row(result, baseline, max_read_drop_pp) for result in cache.values()],
        key=lambda row: float(row["alpha"]),
    )


def pareto_frontier(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points = [dict(row) for row in rows]
    frontier: list[dict[str, Any]] = []

    for i, row in enumerate(points):
        read_i = float(row["attacked_read_accuracy_weighted"])
        null_i = float(row["nominal_noattack_null_accuracy_weighted"])
        dominated = False

        for j, other in enumerate(points):
            if i == j:
                continue
            read_j = float(other["attacked_read_accuracy_weighted"])
            null_j = float(other["nominal_noattack_null_accuracy_weighted"])
            weakly_better = read_j >= read_i - 1.0e-12 and null_j >= null_i - 1.0e-12
            strictly_better = read_j > read_i + 1.0e-12 or null_j > null_i + 1.0e-12
            if weakly_better and strictly_better:
                dominated = True
                break

        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: float(row["alpha"]))


def select_feasible_pareto(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    feasible = [dict(row) for row in rows if bool(row["feasible_max_drop"])]
    if not feasible:
        raise RuntimeError("No feasible alpha found; alpha=0 should always be feasible")

    frontier = pareto_frontier(feasible)
    return max(
        frontier,
        key=lambda row: (
            float(row["nominal_noattack_null_accuracy_weighted"]),
            float(row["attacked_read_accuracy_weighted"]),
            -float(row["alpha"]),
        ),
    )


def fine_grid(
    coarse_rows: Sequence[Mapping[str, Any]],
    coarse_step: float,
    fine_step: float,
) -> list[float]:
    target: set[float] = set()

    for row in pareto_frontier(coarse_rows):
        if bool(row["feasible_max_drop"]):
            lo = max(0.0, float(row["alpha"]) - coarse_step)
            hi = min(1.0, float(row["alpha"]) + coarse_step)
            target.update(alpha_grid(lo, hi, fine_step))

    ordered = sorted(coarse_rows, key=lambda row: float(row["alpha"]))
    for left, right in zip(ordered[:-1], ordered[1:]):
        if bool(left["feasible_max_drop"]) != bool(right["feasible_max_drop"]):
            target.update(alpha_grid(float(left["alpha"]), float(right["alpha"]), fine_step))

    if not target:
        best = select_feasible_pareto(coarse_rows)
        center = float(best["alpha"])
        target.update(
            alpha_grid(max(0.0, center - coarse_step), min(1.0, center + coarse_step), fine_step)
        )
    return sorted(target)


def evaluate_alpha_set(
    context: WiseContext,
    benchmark: BenchmarkContext,
    cache: dict[float, AlphaResult],
    alphas: Iterable[float],
) -> None:
    for alpha in alphas:
        key = round(float(alpha), 10)
        if key not in cache:
            cache[key] = evaluate_read_alpha(context, benchmark, key)


def print_frontier(frontier: Sequence[Mapping[str, Any]], max_read_drop_pp: float) -> None:
    print()
    print("=" * 118)
    print(
        "WiSE-FT PARETO FRONTIER "
        f"(hard constraint: worst attacked subset drop <= {max_read_drop_pp:.3f} pp)"
    )
    print("=" * 118)
    print(
        f"{'alpha':>7} {'READ w.avg':>11} {'No* NULL':>11} "
        f"{'worst drop':>11} {'subset':>11} {'feasible':>9}"
    )
    print("-" * 118)
    for row in frontier:
        print(
            f"{float(row['alpha']):>7.4f} "
            f"{float(row['attacked_read_accuracy_weighted']):>11.4f} "
            f"{float(row['nominal_noattack_null_accuracy_weighted']):>11.4f} "
            f"{float(row['max_attacked_subset_drop_pp']):>+10.3f}pp "
            f"{str(row['max_drop_subset']):>11s} "
            f"{str(bool(row['feasible_max_drop'])):>9s}"
        )


def run_pareto(args: argparse.Namespace) -> None:
    validate_common_args(args)
    if not 0.0 < args.coarse_step <= 1.0:
        raise ValueError("--coarse-step must be in (0,1]")
    if not 0.0 < args.fine_step <= args.coarse_step:
        raise ValueError("--fine-step must be >0 and <= coarse-step")
    if not 0.0 < args.micro_step <= args.fine_step:
        raise ValueError("--micro-step must be >0 and <= fine-step")

    set_seed(args.seed)
    prepare_output_dir(args.output_root, args.overwrite_output)
    device = resolve_device(args.device)
    context = load_wise_context(args.original_model, args.trained_control, device)
    benchmark = load_benchmark_context(context, args.eval_helper, args.batch_size, args.amp)

    cache: dict[float, AlphaResult] = {}
    evaluate_alpha_set(context, benchmark, cache, alpha_grid(0.0, 1.0, args.coarse_step))
    baseline = cache[0.0]

    coarse_rows = rows_from_cache(cache, baseline, args.max_read_drop_pp)
    evaluate_alpha_set(
        context,
        benchmark,
        cache,
        fine_grid(coarse_rows, args.coarse_step, args.fine_step),
    )

    if not args.no_micro:
        fine_rows = rows_from_cache(cache, baseline, args.max_read_drop_pp)
        center = float(select_feasible_pareto(fine_rows)["alpha"])
        evaluate_alpha_set(
            context,
            benchmark,
            cache,
            alpha_grid(
                max(0.0, center - args.fine_step),
                min(1.0, center + args.fine_step),
                args.micro_step,
            ),
        )

    rows = rows_from_cache(cache, baseline, args.max_read_drop_pp)
    frontier = pareto_frontier(rows)
    feasible_frontier = [row for row in frontier if bool(row["feasible_max_drop"])]
    selected = select_feasible_pareto(rows)

    write_csv(args.output_root / "wise_sweep.csv", rows)
    write_csv(args.output_root / "pareto_frontier.csv", frontier)
    write_csv(args.output_root / "feasible_pareto.csv", feasible_frontier)
    print_frontier(frontier, args.max_read_drop_pp)

    selected_alpha = float(selected["alpha"])
    print()
    print(f"[selected] alpha={selected_alpha:.6f}")
    print(
        f"[selected] attacked READ w.avg={float(selected['attacked_read_accuracy_weighted']):.6f} | "
        f"No* NULL={float(selected['nominal_noattack_null_accuracy_weighted']):.6f} | "
        f"worst drop={float(selected['max_attacked_subset_drop_pp']):+.4f} pp "
        f"({selected['max_drop_subset']})"
    )

    apply_alpha(context, selected_alpha)
    full_summaries, full_records = evaluate_full_selected(context, benchmark)
    write_csv(args.output_root / "selected_full_benchmark.csv", full_summaries)
    write_csv(args.output_root / "selected_full_records.csv", full_records)
    print_full_benchmark(benchmark, selected_alpha, full_summaries)

    metadata = {
        "format": "wise-ft-late-read-b20-b21-v2",
        "formula": "theta(alpha)=(1-alpha)*theta_original+alpha*theta_trained_control",
        "original_model": str(args.original_model),
        "trained_control": str(args.trained_control),
        "eval_helper": str(args.eval_helper),
        "original_endpoint": context.original.metadata,
        "trained_endpoint": context.trained.metadata,
        "wise_parameter_names": context.parameter_names,
        "selection": selected,
        "constraint": {
            "type": "worst_attacked_subset_absolute_accuracy_drop",
            "max_drop_percentage_points": float(args.max_read_drop_pp),
            "baseline_alpha": 0.0,
            "attacked_subsets": list(ATTACKED_READ_SUBSETS),
        },
        "pareto_objectives": {
            "maximize_1": "weighted attacked READ binary accuracy",
            "maximize_2": "weighted nominal NoSCAM+NoRTA NULL binary accuracy",
            "selection_on_feasible_frontier": (
                "maximize nominal No* NULL accuracy; tie-break attacked READ accuracy"
            ),
        },
        "sweep": {
            "coarse_step": float(args.coarse_step),
            "fine_step": float(args.fine_step),
            "micro_step": None if args.no_micro else float(args.micro_step),
            "evaluated_alphas": sorted(float(alpha) for alpha in cache),
        },
        "training_firewall": (
            "No optimization occurs. SCAM/RTA are used only for post-training WiSE "
            "interpolation selection/evaluation."
        ),
    }
    write_json(args.output_root / "selected.json", metadata)

    if not args.no_save_model:
        save_full_checkpoint(
            context,
            args.output_root / "selected_model",
            metadata,
            note=(
                "TEMPORARY WiSE-FT late READ / READ_NULL interpolation.\n"
                f"alpha={selected_alpha:.8f}\n"
            ),
        )
        print(f"[save] selected model -> {args.output_root / 'selected_model'}")


# -------------------------------------------------------------------------------------------------
# Boundary scan selection
# -------------------------------------------------------------------------------------------------


def boundary_row(
    context: WiseContext,
    benchmark: BenchmarkContext,
    alpha: float,
    baseline: AlphaResult,
    max_drop_pp: float,
) -> tuple[dict[str, Any], bool]:
    apply_alpha(context, alpha)
    row: dict[str, Any] = {
        "alpha": float(alpha),
        "feasible": True,
        "failed_subset": "",
        "max_drop_pp": 0.0,
    }
    worst_drop = -float("inf")
    worst_subset = ""

    print()
    print(f"[alpha={alpha:.6f}] attacked READ constraint check")
    for subset in BOUNDARY_SCAN_ORDER:
        summary, _ = score_subset(context, benchmark, subset, mode="read")
        baseline_acc = baseline.subset_accuracy(subset)
        acc = float(summary["binary_accuracy"])
        drop_pp = 100.0 * (baseline_acc - acc)

        row[f"{subset}_accuracy"] = acc
        row[f"{subset}_drop_pp"] = drop_pp
        row[f"{subset}_margin"] = float(summary["mean_binary_logit_margin"])

        if drop_pp > worst_drop:
            worst_drop = drop_pp
            worst_subset = subset

        print(
            f"  {subset:<11s} acc={acc:.4f} drop={drop_pp:+.3f}pp "
            f"margin={summary['mean_binary_logit_margin']:+.4f}"
        )

        if drop_pp > max_drop_pp + 1.0e-9:
            row["feasible"] = False
            row["failed_subset"] = subset
            row["max_drop_pp"] = drop_pp
            row["max_drop_subset"] = subset
            print(f"  -> OVERSTEP: {drop_pp:.3f} pp > {max_drop_pp:.3f} pp")
            return row, False

    row["max_drop_pp"] = worst_drop
    row["max_drop_subset"] = worst_subset
    return row, True


def run_boundary(args: argparse.Namespace) -> None:
    validate_common_args(args)
    if not 0.0 <= args.alpha_start <= args.alpha_max <= 1.0:
        raise ValueError("Require 0 <= alpha-start <= alpha-max <= 1")
    if args.alpha_step <= 0:
        raise ValueError("--alpha-step must be >0")

    set_seed(args.seed)
    prepare_output_dir(args.output_root, args.overwrite_output)
    device = resolve_device(args.device)
    context = load_wise_context(args.original_model, args.trained_control, device)
    benchmark = load_benchmark_context(context, args.eval_helper, args.batch_size, args.amp)

    baseline = evaluate_read_alpha(context, benchmark, 0.0, ATTACKED_READ_SUBSETS)
    scan_rows: list[dict[str, Any]] = []
    last_feasible: float | None = None
    overstepped: float | None = None

    for alpha in alpha_grid(args.alpha_start, args.alpha_max, args.alpha_step):
        row, feasible = boundary_row(
            context,
            benchmark,
            alpha,
            baseline,
            args.max_read_drop_pp,
        )
        scan_rows.append(row)
        write_csv(args.output_root / "scan.csv", scan_rows)
        if feasible:
            last_feasible = float(alpha)
        else:
            overstepped = float(alpha)
            break

    if overstepped is not None:
        selected_alpha = round(overstepped - args.alpha_step, 10)
        if last_feasible is not None and abs(selected_alpha - last_feasible) > 1.0e-8:
            raise RuntimeError(
                f"Boundary bookkeeping mismatch: selected={selected_alpha}, last_feasible={last_feasible}"
            )
    elif last_feasible is not None:
        selected_alpha = last_feasible
    else:
        selected_alpha = round(args.alpha_start - args.alpha_step, 10)
        if selected_alpha < 0.0:
            selected_alpha = 0.0

    print(f"[select] alpha={selected_alpha:.6f}")
    apply_alpha(context, selected_alpha)
    full_summaries, full_records = evaluate_full_selected(context, benchmark)
    write_csv(args.output_root / "selected_full_benchmark.csv", full_summaries)
    write_csv(args.output_root / "selected_full_records.csv", full_records)
    print_full_benchmark(benchmark, selected_alpha, full_summaries)

    metadata = {
        "format": "wise-ft-b20-b21-simple-boundary-v2",
        "formula": "theta(alpha)=(1-alpha)*theta_original+alpha*theta_trained_control",
        "original_model": str(args.original_model),
        "trained_control": str(args.trained_control),
        "eval_helper": str(args.eval_helper),
        "selected_alpha": float(selected_alpha),
        "first_overstepped_alpha": None if overstepped is None else float(overstepped),
        "scan": {
            "alpha_start": float(args.alpha_start),
            "alpha_max": float(args.alpha_max),
            "alpha_step": float(args.alpha_step),
            "evaluated_alphas": [float(row["alpha"]) for row in scan_rows],
        },
        "constraint": {
            "max_individual_attacked_read_drop_pp": float(args.max_read_drop_pp),
            "baseline_alpha": 0.0,
            "subsets": list(ATTACKED_READ_SUBSETS),
        },
        "wise_parameter_names": context.parameter_names,
        "policy": (
            "scan upward; stop at first hard overstep; select exactly one alpha step back; "
            "full-evaluate selected alpha; save complete HF model"
        ),
    }
    write_json(args.output_root / "selected.json", metadata)
    save_full_checkpoint(
        context,
        args.output_root / "selected_model",
        metadata,
        note=(
            "TEMPORARY WiSE-FT late READ / READ_NULL interpolation.\n"
            f"alpha={selected_alpha:.8f}\n"
        ),
    )
    print(f"[saved] alpha={selected_alpha:.6f} -> {args.output_root / 'selected_model'}")


# -------------------------------------------------------------------------------------------------
# Explicit checkpoint export
# -------------------------------------------------------------------------------------------------


def alpha_dir_name(alpha: float) -> str:
    return f"alpha_{alpha:.2f}"


def run_export(args: argparse.Namespace) -> None:
    alphas = [float(alpha) for alpha in args.alphas]
    if not alphas:
        raise ValueError("No alphas supplied")
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("All alphas must be in [0,1]")

    prepare_output_dir(args.output_root, args.overwrite_output)
    context = load_wise_context(args.original_model, args.trained_control, torch.device("cpu"))

    saved: list[Path] = []
    for alpha in alphas:
        apply_alpha(context, alpha)
        metadata = {
            "format": "wise-ft-b20-b21-manual-checkpoint-v2",
            "alpha": float(alpha),
            "formula": "theta(alpha)=(1-alpha)*theta_original+alpha*theta_trained_control",
            "original_model": str(args.original_model),
            "trained_control": str(args.trained_control),
            "read_tap_blocks": list(EXPECTED_TAPS),
            "read_state_blocks": list(EXPECTED_TAPS),
            "interpolated_parameter_names": context.parameter_names,
            "evaluation": "NONE",
        }
        out_dir = args.output_root / alpha_dir_name(alpha)
        save_full_checkpoint(
            context,
            out_dir,
            metadata,
            note=(
                "TEMPORARY WiSE-FT B20/B21 late reader checkpoint.\n"
                f"alpha={alpha:.8f}\n"
                "No evaluation was run by the checkpoint-saving command.\n"
            ),
        )
        saved.append(out_dir)
        print(f"[saved] alpha={alpha:.2f} -> {out_dir}")

    write_json(
        args.output_root / "CHECKPOINTS.json",
        {
            "alphas": alphas,
            "directories": [str(path) for path in saved],
            "evaluation": "NONE",
        },
    )


# -------------------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------------------


def resolve_device(value: str | None) -> torch.device:
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def validate_common_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_read_drop_pp < 0:
        raise ValueError("--max-read-drop-pp must be >=0")


def add_endpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--original-model", type=Path, default=DEFAULT_ORIGINAL_MODEL)
    parser.add_argument("--trained-control", type=Path, default=DEFAULT_TRAINED_CONTROL)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval-helper", type=Path, default=DEFAULT_EVAL_HELPER)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-read-drop-pp",
        type=float,
        default=2.0,
        help="Hard constraint on the worst attacked subset relative to alpha=0.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pareto = subparsers.add_parser("pareto", help="adaptive constrained Pareto alpha sweep")
    add_endpoint_args(pareto)
    add_eval_args(pareto)
    pareto.add_argument("--output-root", type=Path, default=Path("wiseft_pareto"))
    pareto.add_argument("--coarse-step", type=float, default=0.10)
    pareto.add_argument("--fine-step", type=float, default=0.02)
    pareto.add_argument("--micro-step", type=float, default=0.005)
    pareto.add_argument("--no-micro", action="store_true")
    pareto.add_argument("--overwrite-output", action="store_true")
    pareto.add_argument("--no-save-model", action="store_true")
    pareto.set_defaults(handler=run_pareto)

    boundary = subparsers.add_parser("boundary", help="scan upward and step back once")
    add_endpoint_args(boundary)
    add_eval_args(boundary)
    boundary.set_defaults(max_read_drop_pp=1.8)
    boundary.add_argument("--output-root", type=Path, default=Path("wiseft_boundary"))
    boundary.add_argument("--alpha-start", type=float, default=0.24)
    boundary.add_argument("--alpha-max", type=float, default=0.35)
    boundary.add_argument("--alpha-step", type=float, default=0.01)
    boundary.add_argument("--overwrite-output", action="store_true")
    boundary.set_defaults(handler=run_boundary)

    export = subparsers.add_parser("export", help="save explicit alpha checkpoints; no evaluation")
    add_endpoint_args(export)
    export.add_argument("--output-root", type=Path, default=Path("wiseft_manual_checkpoints_v1"))
    export.add_argument("--alphas", nargs="+", type=float, default=[0.20, 0.23, 0.25, 0.28])
    export.add_argument("--overwrite-output", action="store_true")
    export.set_defaults(handler=run_export)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
