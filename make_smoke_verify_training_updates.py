#!/usr/bin/env python3
"""Verify that every tiny smoke-test optimizer step actually changed FP32 parameters.

This reads the precision diagnostics produced by the real trainers.  Those
snapshots only select parameters with ``requires_grad=True`` and compare sampled
FP32 values immediately before vs. after ``optimizer.step()``.  The check is
therefore stronger than merely observing a finite loss or gradient norm.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON must be an object: {path}")
    return data


def resolve_root(project_root: Path, text: str) -> Path:
    if WIN_ABS.match(text) and project_root.drive.lower() != PureWindowsPath(text).drive.lower():
        # On Windows, pathlib resolves the drive normally.  This branch mainly
        # keeps cross-OS inspection intelligible.
        return Path(text)
    p = Path(text)
    return p if p.is_absolute() else project_root / p


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def inum(value: Any) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def expected_updates(cfg: Mapping[str, Any], stage_name: str) -> set[tuple[str, int]]:
    stages = cfg["stages"]
    if stage_name == "soft_token":
        n = int(stages[stage_name]["args"]["steps"])
        return {("soft_token", i) for i in range(1, n + 1)}
    if stage_name in {"read", "content", "joint"}:
        shared = cfg["shared_args"]["legacy_hard_text"]
        args = stages[stage_name]["args"]
        epochs = int(args["epochs"])
        max_batches = int(shared.get("max_train_batches", 0))
        accum = max(1, int(shared.get("grad_accum_steps", 1)))
        if max_batches <= 0:
            # The generated smoke config intentionally always sets this.  Fail
            # closed rather than pretending we know the full loader length.
            raise ValueError("Smoke verifier requires legacy max_train_batches > 0")
        per_epoch = math.ceil(max_batches / accum)
        return {(stage_name, i) for i in range(1, epochs * per_epoch + 1)}
    if stage_name in {"final_base", "all_weights"}:
        args = stages[stage_name]["args"]
        steps = int(args["steps_per_epoch"])
        expected: set[tuple[str, int]] = set()
        for phase in ("1a", "1b", "1b5", "1c"):
            epochs = int(args.get(f"phase_{phase}_epochs", 0))
            for i in range(1, epochs * steps + 1):
                expected.add((phase, i))
        return expected
    raise KeyError(stage_name)


def changed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if math.isfinite(fnum(row.get("update_abs_max"))) and fnum(row.get("update_abs_max")) > 0.0
    ]


def short_name(name: str, limit: int = 82) -> str:
    return name if len(name) <= limit else "…" + name[-(limit - 1):]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("smoke_test_training_config.json"))
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    args = ap.parse_args()

    root = args.project_root.expanduser().resolve()
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    cfg = load_json(cfg_path)
    train_root = resolve_root(root, str(cfg["global"]["train_root"]))
    stages: Mapping[str, Any] = cfg["stages"]

    failures: list[str] = []
    print(f"[smoke-verify] train root: {train_root}")

    for stage_name in ("soft_token", "read", "content", "joint", "final_base", "all_weights"):
        out = train_root / Path(str(stages[stage_name]["output_subdir"]))
        updates = out / "precision_diagnostics" / "precision_parameter_updates.csv"
        if not updates.is_file():
            failures.append(f"{stage_name}: missing {updates}")
            print(f"[smoke-verify] {stage_name:12s} MISSING precision update CSV")
            continue

        rows = read_csv(updates)
        changed = changed_rows(rows)
        by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        changed_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            step = inum(row.get("optimizer_step"))
            if step is None:
                continue
            key = (str(row.get("phase", "")), step)
            by_key[key].append(row)
        for row in changed:
            step = inum(row.get("optimizer_step"))
            if step is None:
                continue
            changed_by_key[(str(row.get("phase", "")), step)].append(row)

        expected = expected_updates(cfg, stage_name)
        missing_snapshots = sorted(expected - set(by_key))
        no_change = sorted(key for key in expected if key in by_key and key not in changed_by_key)
        extras = sorted(set(by_key) - expected)
        phases = sorted({phase for phase, _ in by_key})
        print(
            f"[smoke-verify] {stage_name:12s} rows={len(rows):4d} changed={len(changed):4d} "
            f"expected_steps={len(expected):2d} observed_steps={len(by_key):2d} phases={phases}"
        )
        if missing_snapshots:
            failures.append(f"{stage_name}: missing precision snapshots for {missing_snapshots}")
        if no_change:
            failures.append(f"{stage_name}: optimizer step(s) had no sampled FP32 parameter change: {no_change}")
        if extras:
            print(f"[smoke-verify] {stage_name:12s} note: extra diagnostic step keys={extras}")

        # Print the strongest actual change per phase so the user can eyeball
        # which trainable family moved without opening the CSV.
        for phase in phases:
            candidates = [r for r in changed if str(r.get("phase", "")) == phase]
            if candidates:
                strongest = max(candidates, key=lambda r: fnum(r.get("update_abs_max")))
                print(
                    f"[smoke-verify]   {stage_name}/{phase}: strongest="
                    f"{short_name(str(strongest.get('name', '')))} "
                    f"max|Δ|={fnum(strongest.get('update_abs_max')):.3e}"
                )

        if stage_name == "all_weights":
            backbone = [
                r for r in changed
                if any(token in str(r.get("name", "")).lower() for token in (
                    "visual.conv1", "visual.positional_embedding", "resblocks.0.", "resblocks.23."
                ))
            ]
            if backbone:
                print(f"[smoke-verify] all_weights  backbone sampled updates: {len(backbone)}")
            else:
                failures.append("all_weights: no sampled backbone update among Conv1/position/B0/B23 tensors")

    fb_out = train_root / Path(str(stages["final_base"]["output_subdir"]))
    router_prefix = str(cfg.get("router_utility", {}).get("output_prefix", "router_utility_a4"))
    router_csv = fb_out / f"{router_prefix}_train.csv"
    if router_csv.is_file():
        rows = read_csv(router_csv)
        expected_router = int(cfg.get("router_utility", {}).get("steps", 0))
        finite_rows = [r for r in rows if math.isfinite(fnum(r.get("grad_norm")))]
        nonzero_grad = [r for r in finite_rows if fnum(r.get("grad_norm")) > 0.0]
        print(
            f"[smoke-verify] router_A4    rows={len(rows)} nonzero_grad_steps={len(nonzero_grad)} "
            f"expected={expected_router}"
        )
        if len(rows) < expected_router:
            failures.append(f"router A4: expected {expected_router} train rows, found {len(rows)}")
        if len(nonzero_grad) < expected_router:
            failures.append(
                f"router A4: expected nonzero grad_norm on {expected_router} updates, found {len(nonzero_grad)}"
            )
    else:
        failures.append(f"router A4: missing {router_csv}")
        print("[smoke-verify] router_A4    MISSING train CSV")

    aw_out = train_root / Path(str(stages["all_weights"]["output_subdir"]))
    benchmark_files = sorted(aw_out.glob("benchmarks_*_epoch_*.json"))
    fb_bench = sorted(fb_out.glob("benchmarks_*_epoch_*.json"))
    total_bench = len(benchmark_files) + len(fb_bench)
    print(
        f"[smoke-verify] benchmark JSONs: final_base={len(fb_bench)} "
        f"all_weights={len(benchmark_files)} total={total_bench}"
    )
    if total_bench != 1:
        failures.append(f"expected exactly one benchmark pass, found {total_bench}")

    if failures:
        print("\n[smoke-verify] FAIL")
        for item in failures:
            print(f"  - {item}")
        return 2

    print("\n[smoke-verify] PASS: CLIP touched grad. Every scheduled smoke optimizer step moved sampled FP32 weights. :P")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
