#!/usr/bin/env python3
"""Friendly setup/status/runner for the public CLIP ModeMUX benchmark suite.

Typical reproduction:

    python benchmark.py setup
    python benchmark.py status
    python benchmark.py run

If this checkout was used for training, reuse already-known dataset locations:

    python benchmark.py setup --from-training-config training_config.local.json

`benchmark_config.json` is standalone after setup; benchmark execution never imports
or depends on the training configuration.
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_utils.config import (
    BENCHMARK_ORDER,
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_PATH,
    load_config,
    optional_path,
    save_config,
    training_seed_paths,
)
from benchmark_utils.data_setup import (
    COCO_EXPECTED,
    ask_path,
    image_count,
    install_coco_split,
    install_objectnet_mvt,
    status_rows,
    validate_imagenet_root,
    warm_typo_cache,
    yn,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_BY_BENCHMARK = {
    "typo": PROJECT_ROOT / "benchmarks" / "typo.py",
    "objectnet_mvt": PROJECT_ROOT / "benchmarks" / "objectnet_mvt.py",
    "mscoco": PROJECT_ROOT / "benchmarks" / "mscoco.py",
    "sugarcrepe": PROJECT_ROOT / "benchmarks" / "sugarcrepe.py",
    "imagenet_linear_probe": PROJECT_ROOT / "benchmarks" / "imagenet_linear_probe.py",
}
OUTPUT_DIRNAME = {
    "typo": "typo",
    "objectnet_mvt": "objectnet_mvt",
    "mscoco": "mscoco",
    "sugarcrepe": "sugarcrepe",
    "imagenet_linear_probe": "imagenet_linear_probe",
}


def _display_path(value: Any) -> str:
    return "not configured" if value is None or not str(value).strip() else str(value)


def _default_download_root_from_training(seed: dict[str, Any]) -> Path | None:
    root = _seeded_imagenet_root(seed)
    if root is not None:
        return root.parent
    # Deliberately do not infer from SPRIGHT/COCO; that path can be several levels
    # below the user's dataset root and is not benchmark COCO anyway.
    return None


def _existing_dir(value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_dir() else None


def _seeded_imagenet_root(seed: dict[str, Any]) -> Path | None:
    root = _existing_dir(seed.get("imagenet_root"))
    if root is not None:
        return root
    train = _existing_dir(seed.get("imagenet_train_root_override"))
    val = _existing_dir(seed.get("imagenet_val_root_override"))
    if train is not None and val is not None and train.parent == val.parent:
        return train.parent
    return None


def _discover_downloaded_defaults(cfg: dict[str, Any], data_root: Path) -> None:
    """Reuse conventional benchmark download locations without another prompt."""
    candidates = {
        "objectnet_mvt_root": data_root / "ObjectNet-MVT" / "all",
        "coco_val2014_root": data_root / "COCO" / "val2014",
        "coco_val2017_root": data_root / "COCO" / "val2017",
    }
    for key, path in candidates.items():
        if cfg["datasets"].get(key):
            continue
        if path.is_dir():
            cfg["datasets"][key] = str(path)


def _prompt_optional_existing(prompt: str, current: Path | None = None) -> Path | None:
    while True:
        path = ask_path(prompt, current)
        if path is None:
            return None
        if path.is_dir():
            return path
        print(f"  not a directory: {path}")
        if not yn("Try another path?", default=True):
            return None


def setup(args: argparse.Namespace) -> None:
    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path, allow_missing=True)
    cfg = copy.deepcopy(cfg)

    print("\n" + "=" * 88)
    print("BENCHMARK SETUP")
    print("=" * 88)
    print(f"Model: {cfg['model']['path']}")
    print(f"Config: {cfg_path}")

    seed: dict[str, Any] | None = None
    if args.from_training_config is not None:
        training_path = args.from_training_config.resolve()
        seed = training_seed_paths(training_path)
        print(f"\n[seed] reading paths only from {training_path}")

        imagenet = _seeded_imagenet_root(seed)
        if imagenet is not None:
            cfg["datasets"]["imagenet_root"] = str(imagenet)
            ok, note = validate_imagenet_root(imagenet, PROJECT_ROOT)
            print(f"  ImageNet: {'FOUND' if ok else 'FOUND, NEEDS ATTENTION'} — {imagenet} ({note})")
        else:
            print(f"  ImageNet: not found — {_display_path(seed.get('imagenet_root'))}")

        mvt = _existing_dir(seed.get("mvt_image_root"))
        if mvt is not None and image_count(mvt) >= 4771:
            cfg["datasets"]["objectnet_mvt_root"] = str(mvt)
            print(f"  ObjectNet-MVT: FOUND — {mvt}")
        else:
            print(f"  ObjectNet-MVT: not found/complete — {_display_path(seed.get('mvt_image_root'))}")

        scam_images = _existing_dir(seed.get("scam_images_root"))
        scam_labels = Path(str(seed.get("scam_labels_csv"))).expanduser() if seed.get("scam_labels_csv") else None
        if scam_images is not None and scam_labels is not None and scam_labels.is_file():
            print(f"  SCAM training/eval copy: FOUND — {scam_images}")
            print("    (public typo benchmark still resolves SCAM/RTA by canonical HF dataset ID)")
        else:
            print("  SCAM training/eval copy: not found in training config; HF download remains available")

        training_coco = _existing_dir(seed.get("training_coco_root"))
        if training_coco is not None:
            print(f"  Training COCO/SPRIGHT: FOUND — {training_coco}")
            print("    (not reused: MSCOCO/SugarCrepe benchmarks require official COCO val2014/val2017)")
        else:
            print(f"  Training COCO/SPRIGHT: not found — {_display_path(seed.get('training_coco_root'))}")
        print("  Benchmark COCO val2014: " + ("FOUND" if _existing_dir(cfg['datasets'].get('coco_val2014_root')) else "MISSING"))
        print("  Benchmark COCO val2017: " + ("FOUND" if _existing_dir(cfg['datasets'].get('coco_val2017_root')) else "MISSING"))

    # Pick a benchmark data/download root.  A seeded ImageNet root gives the natural
    # default (e.g. AI_DATASET); otherwise retain config or use benchmark_data/.
    current_data_root = Path(str(cfg.get("data_root") or "benchmark_data")).expanduser()
    seed_default = _default_download_root_from_training(seed) if seed else None
    proposed = seed_default or current_data_root
    if args.data_root is not None:
        data_root = args.data_root.expanduser()
    elif seed is not None:
        data_root = proposed
        print(f"\n[data] default benchmark download folder from training config: {data_root}")
    else:
        data_root = ask_path("Folder for auto-downloaded benchmark data", proposed) or proposed
    cfg["data_root"] = str(data_root)
    _discover_downloaded_defaults(cfg, data_root)

    # HF datasets: no local path is required.  Optionally keep them under the chosen
    # data root instead of the user's normal Hugging Face cache.
    if args.hf_cache_dir is not None:
        hf_cache = args.hf_cache_dir.expanduser()
        cfg["hf_cache_dir"] = str(hf_cache)
    elif seed is None and not args.yes:
        current_cache = optional_path(cfg.get("hf_cache_dir"), PROJECT_ROOT)
        entered = ask_path("Optional Hugging Face cache for SCAM/RTA (Enter = normal HF cache)", current_cache)
        cfg["hf_cache_dir"] = None if entered is None else str(entered)
    hf_cache = optional_path(cfg.get("hf_cache_dir"), PROJECT_ROOT)

    # Standalone setup asks for reusable local paths before offering downloads.
    if seed is None:
        current_mvt = _existing_dir(cfg["datasets"].get("objectnet_mvt_root"))
        mvt = _prompt_optional_existing(
            "Existing ObjectNet-MVT image folder (Enter = auto-download)", current_mvt
        )
        if mvt is not None and image_count(mvt) >= 4771:
            cfg["datasets"]["objectnet_mvt_root"] = str(mvt)
        elif mvt is not None:
            print(f"  ObjectNet-MVT path has only {image_count(mvt):,} top-level image files; will offer download.")
            cfg["datasets"]["objectnet_mvt_root"] = None

        current_imagenet = _existing_dir(cfg["datasets"].get("imagenet_root"))
        imagenet = _prompt_optional_existing(
            "ILSVRC2012 root containing train/ and val/ (Enter = skip ImageNet probe)", current_imagenet
        )
        if imagenet is not None:
            ok, note = validate_imagenet_root(imagenet, PROJECT_ROOT)
            if ok:
                cfg["datasets"]["imagenet_root"] = str(imagenet)
                cfg["benchmarks"]["imagenet_linear_probe"] = True
                print(f"  ImageNet: READY — {note}")
            else:
                print(f"  ImageNet not enabled: {note}")
                cfg["datasets"]["imagenet_root"] = None
                cfg["benchmarks"]["imagenet_linear_probe"] = False
        else:
            cfg["datasets"]["imagenet_root"] = None
            cfg["benchmarks"]["imagenet_linear_probe"] = False

        for key, split in (("coco_val2014_root", "val2014"), ("coco_val2017_root", "val2017")):
            current = _existing_dir(cfg["datasets"].get(key))
            prompt = f"Existing COCO {split} image folder (Enter = auto-download)"
            path = _prompt_optional_existing(prompt, current)
            if path is not None and image_count(path) == COCO_EXPECTED[split]:
                cfg["datasets"][key] = str(path)
            elif path is not None:
                print(f"  COCO {split}: expected {COCO_EXPECTED[split]:,} images; will offer download.")
                cfg["datasets"][key] = None

    # Seeded setup: one confirmation for the stuff not already found.  Standalone setup
    # asks per source only where a usable path was not supplied.
    missing_mvt = not (
        (mvt_path := _existing_dir(cfg["datasets"].get("objectnet_mvt_root")))
        and image_count(mvt_path) >= 4771
    )
    missing_c14 = not (
        (c14 := _existing_dir(cfg["datasets"].get("coco_val2014_root")))
        and image_count(c14) == COCO_EXPECTED["val2014"]
    )
    missing_c17 = not (
        (c17 := _existing_dir(cfg["datasets"].get("coco_val2017_root")))
        and image_count(c17) == COCO_EXPECTED["val2017"]
    )

    if args.no_download:
        do_missing = False
    elif seed is not None and (missing_mvt or missing_c14 or missing_c17):
        names = [name for name, missing in (
            ("ObjectNet-MVT", missing_mvt), ("COCO val2014", missing_c14), ("COCO val2017", missing_c17)
        ) if missing]
        do_missing = yn(
            f"Download missing benchmark data ({', '.join(names)}) under {data_root}?",
            default=True,
            assume_yes=args.yes,
        )
    else:
        # Standalone setup asks separately for each missing downloadable source.
        do_missing = False

    if missing_mvt and do_missing:
        cfg["datasets"]["objectnet_mvt_root"] = str(install_objectnet_mvt(PROJECT_ROOT, data_root))
    elif missing_mvt and seed is None and not args.no_download:
        if yn("Auto-download ObjectNet-MVT?", default=True, assume_yes=args.yes):
            cfg["datasets"]["objectnet_mvt_root"] = str(install_objectnet_mvt(PROJECT_ROOT, data_root))

    for key, split, missing in (
        ("coco_val2014_root", "val2014", missing_c14),
        ("coco_val2017_root", "val2017", missing_c17),
    ):
        if not missing:
            continue
        if do_missing:
            cfg["datasets"][key] = str(install_coco_split(split, data_root))
        elif seed is None and not args.no_download:
            if yn(f"Auto-download official COCO {split}?", default=True, assume_yes=args.yes):
                cfg["datasets"][key] = str(install_coco_split(split, data_root))

    # Typo datasets are small enough to cache during setup, but users can decline and
    # let the actual benchmark download them on demand later.
    if not args.no_download and yn(
        "Cache SCAM and RTA-100 now?", default=True, assume_yes=args.yes
    ):
        warm_typo_cache(hf_cache)

    # Re-check ImageNet after seeding.  It is optional and never auto-downloaded.
    imagenet = _existing_dir(cfg["datasets"].get("imagenet_root"))
    if imagenet is not None:
        ok, note = validate_imagenet_root(imagenet, PROJECT_ROOT)
        cfg["benchmarks"]["imagenet_linear_probe"] = bool(ok)
        if not ok:
            print(f"[ImageNet] configured path is not benchmark-ready: {note}")
    else:
        cfg["benchmarks"]["imagenet_linear_probe"] = False

    # Missing optional datasets disable only their own benchmark; a bare `run`
    # therefore executes everything that setup actually made ready.
    mvt_ready = (
        (mvt_final := _existing_dir(cfg["datasets"].get("objectnet_mvt_root"))) is not None
        and image_count(mvt_final) >= 4771
    )
    c14_ready = (
        (c14_final := _existing_dir(cfg["datasets"].get("coco_val2014_root"))) is not None
        and image_count(c14_final) == COCO_EXPECTED["val2014"]
    )
    c17_ready = (
        (c17_final := _existing_dir(cfg["datasets"].get("coco_val2017_root"))) is not None
        and image_count(c17_final) == COCO_EXPECTED["val2017"]
    )
    cfg["benchmarks"]["objectnet_mvt"] = bool(mvt_ready)
    cfg["benchmarks"]["mscoco"] = bool(c14_ready)
    cfg["benchmarks"]["sugarcrepe"] = bool(c17_ready)
    cfg["benchmarks"]["typo"] = True

    save_config(cfg, cfg_path)
    print(f"\n[save] {cfg_path}")
    print_status(cfg)


def print_status(cfg: dict[str, Any]) -> None:
    print("\nBenchmark configuration")
    print("-" * 88)
    print(f"model       {cfg['model']['path']}")
    print(f"output_root {cfg['output_root']}")
    print(f"data_root   {cfg['data_root']}")
    print("\nDatasets")
    for name, state, detail in status_rows(cfg, PROJECT_ROOT):
        print(f"  {name:<22} {state:<18} {detail}")
    print("\nBenchmarks")
    for name in BENCHMARK_ORDER:
        enabled = bool(cfg["benchmarks"].get(name, False))
        print(f"  {name:<24} {'enabled' if enabled else 'disabled'}")


def status(args: argparse.Namespace) -> None:
    cfg = load_config(args.config.resolve())
    print_status(cfg)


def _resolved_output_root(cfg: dict[str, Any]) -> Path:
    path = Path(str(cfg["output_root"])).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _model_args(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    model = dict(cfg["model"])
    if args.model is not None:
        model["path"] = args.model
    if args.model_alias is not None:
        model["alias"] = args.model_alias
    result = ["--model", str(model["path"]), "--model-alias", str(model["alias"])]
    base = model.get("base_model_or_path")
    if base:
        result += ["--base-model-or-path", str(base)]
    return result


def _require_dir(path: Path | None, label: str) -> Path:
    if path is None or not path.is_dir():
        raise FileNotFoundError(
            f"{label} is not ready ({path or 'not configured'}). Run `python benchmark.py setup`."
        )
    return path


def command_for(name: str, cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    script = SCRIPT_BY_BENCHMARK[name]
    if not script.is_file():
        raise FileNotFoundError(script)
    out = _resolved_output_root(cfg) / OUTPUT_DIRNAME[name]
    data = cfg["datasets"]
    cmd = [sys.executable, str(script), *_model_args(cfg, args), "--output-dir", str(out)]

    if name == "typo":
        cache = optional_path(cfg.get("hf_cache_dir"), PROJECT_ROOT)
        if cache is not None:
            cmd += ["--hf-cache-dir", str(cache)]
    elif name == "objectnet_mvt":
        image_root = _require_dir(optional_path(data.get("objectnet_mvt_root"), PROJECT_ROOT), "ObjectNet-MVT")
        cmd += [
            "--csv-file", str(PROJECT_ROOT / "utils_datasets" / "mvt" / "human_responses_dedup.csv"),
            "--image-folder", str(image_root),
        ]
    elif name == "mscoco":
        image_root = _require_dir(optional_path(data.get("coco_val2014_root"), PROJECT_ROOT), "COCO val2014")
        cmd += [
            "--coco-img-dir", str(image_root),
            "--json-path", str(PROJECT_ROOT / "utils_datasets" / "coco" / "coco_val_karpathy.json"),
        ]
    elif name == "sugarcrepe":
        image_root = _require_dir(optional_path(data.get("coco_val2017_root"), PROJECT_ROOT), "COCO val2017")
        cmd += [
            "--coco-image-root", str(image_root),
            "--data-root", str(PROJECT_ROOT / "utils_datasets" / "sugar_crepe"),
        ]
    elif name == "imagenet_linear_probe":
        imagenet = _require_dir(optional_path(data.get("imagenet_root"), PROJECT_ROOT), "ImageNet-1k")
        ok, note = validate_imagenet_root(imagenet, PROJECT_ROOT)
        if not ok:
            raise RuntimeError(f"ImageNet root is not benchmark-ready: {note}")
        cmd += [
            "--train", str(imagenet / "train"),
            "--val", str(imagenet / "val"),
            "--devkit", str(PROJECT_ROOT / "utils_datasets" / "imagenet" / "ILSVRC2012_devkit_t12"),
        ]
    else:  # pragma: no cover
        raise KeyError(name)
    return cmd


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config.resolve())
    requested = list(args.benchmarks or [])
    if not requested or requested == ["all"]:
        selected = [name for name in BENCHMARK_ORDER if bool(cfg["benchmarks"].get(name, False))]
    else:
        unknown = [name for name in requested if name not in BENCHMARK_ORDER]
        if unknown:
            raise ValueError(f"Unknown benchmark(s): {unknown}; choices={BENCHMARK_ORDER}")
        selected = requested
    if not selected:
        raise RuntimeError("No benchmarks selected/enabled")

    print("\nExecution plan")
    print("-" * 88)
    for index, name in enumerate(selected, start=1):
        print(f"  {index}. {name} -> {_resolved_output_root(cfg) / OUTPUT_DIRNAME[name]}")
    print()

    for index, name in enumerate(selected, start=1):
        cmd = command_for(name, cfg, args)
        print("=" * 88)
        print(f"[{index}/{len(selected)}] {name}")
        print("[cmd] " + subprocess.list2cmdline(cmd))
        print("=" * 88, flush=True)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / DEFAULT_CONFIG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="interactive one-time dataset/config setup")
    p_setup.add_argument("--from-training-config", type=Path, default=None)
    p_setup.add_argument("--data-root", type=Path, default=None)
    p_setup.add_argument("--hf-cache-dir", type=Path, default=None)
    p_setup.add_argument("--yes", action="store_true", help="accept download confirmations")
    p_setup.add_argument("--no-download", action="store_true", help="configure/inspect only; download nothing")
    p_setup.set_defaults(handler=setup)

    p_status = sub.add_parser("status", help="show configured model, datasets, and enabled benchmarks")
    p_status.set_defaults(handler=status)

    p_run = sub.add_parser("run", help="run configured benchmarks")
    p_run.add_argument("benchmarks", nargs="*", help="benchmark names or `all`; default=enabled config entries")
    p_run.add_argument("--model", default=None, help="one-run model override (HF id or local path)")
    p_run.add_argument("--model-alias", default=None, help="one-run display alias override")
    p_run.set_defaults(handler=run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
