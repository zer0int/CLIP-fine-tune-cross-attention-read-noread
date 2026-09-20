#!/usr/bin/env python3
"""Public multi-model reproduction entry point for the Conv1 GPIC manifold probe.

Each requested model runs in an independent output subdirectory whose name matches
its embedding-bank lookup key (``repo/id`` -> ``repo__id``).  This lets one
reproduction workspace contain multiple GPIC model variants without weakening the
single-run model/bank identity checks inside ``conv1_gpic_manifold``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

from probe_tools_repo import ensure_repo_root

ensure_repo_root()

from x_paper_reproduction.conv1_manifold_gpic.conv1_gpic_manifold import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    build_parser as build_single_model_parser,
    main as run_single_model,
)

DEFAULT_MODELS = (
    DEFAULT_MODEL,
    "openai/clip-vit-large-patch14",
)


def model_output_name(model: str) -> str:
    """Use the same readable repo-id mapping as the public embedding bank."""
    value = str(model).strip().replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not value:
        raise ValueError(f"Could not derive output name from model source {model!r}")
    return value


def build_wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Conv1 GPIC manifold sequentially for one or more model-matched embedding banks.",
        epilog="All unrecognized options are forwarded unchanged to the single-model GPIC probe.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Model source; repeat for multiple models. Defaults to released x-attn then OpenAI CLIP.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory; each model is written below its own model-name subfolder.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_wrapper_parser()
    known, forwarded = parser.parse_known_args(argv)
    models = list(dict.fromkeys(known.models or DEFAULT_MODELS))
    base_output = known.output_dir.expanduser().resolve()

    # Validate every option forwarded by the multi-model wrapper with the real
    # single-model parser before touching caches or entering model/data work.
    # Besides catching typos early in normal runs, this is intentionally a
    # ``parse_args`` call: reproduce.py's parse-only preflight sentinel wraps
    # that method and can therefore stop here safely even though the outer
    # wrapper itself uses ``parse_known_args``.
    validation_model = models[0]
    validation_output = base_output / model_output_name(validation_model)
    build_single_model_parser().parse_args(
        [*forwarded, "--model", validation_model, "--output_dir", str(validation_output)]
    )

    base_output.mkdir(parents=True, exist_ok=True)

    overwrite = "--overwrite" in forwarded
    results: list[dict[str, object]] = []
    first_error = 0

    for index, model in enumerate(models, start=1):
        model_dir = base_output / model_output_name(model)
        summary_path = model_dir / "summary.json"
        skip_path = model_dir / "SKIPPED_GPIC_NA.json"

        if summary_path.is_file() and not overwrite:
            print(f"[GPIC {index}/{len(models)}] CACHED {model} -> {model_dir}")
            results.append({"model": model, "output_dir": str(model_dir), "status": "cached"})
            continue

        print(f"[GPIC {index}/{len(models)}] RUN {model} -> {model_dir}")
        child_args = [*forwarded, "--model", model, "--output_dir", str(model_dir)]
        rc = int(run_single_model(child_args))
        if rc != 0 and first_error == 0:
            first_error = rc

        if summary_path.is_file():
            status = "completed"
        elif skip_path.is_file():
            status = "gpic_unavailable"
        else:
            status = f"exit_{rc}"
        results.append({"model": model, "output_dir": str(model_dir), "status": status, "return_code": rc})

        if rc != 0:
            break

    batch_summary_path = base_output / "batch_summary.json"
    all_complete = len(results) == len(models) and all(
        (base_output / model_output_name(model) / "summary.json").is_file()
        for model in models
    )
    if all_complete:
        batch_summary_path.write_text(
            json.dumps({"models": models, "results": results}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[GPIC] all requested models complete: {batch_summary_path}")
    else:
        batch_summary_path.unlink(missing_ok=True)
        if first_error == 0:
            print("[GPIC] one or more models are N/A/incomplete; task remains retryable.")

    return first_error


if __name__ == "__main__":
    raise SystemExit(main())
