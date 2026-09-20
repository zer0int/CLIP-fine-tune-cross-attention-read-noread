"""Load, validate, seed, and save the standalone benchmark configuration."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .constants import DEFAULT_MODEL_ALIAS, DEFAULT_MODEL_PATH

DEFAULT_CONFIG_PATH = Path("benchmark_config.json")
DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "alias": DEFAULT_MODEL_ALIAS,
        "path": DEFAULT_MODEL_PATH,
        "base_model_or_path": None,
    },
    "output_root": "out_bench_results",
    "data_root": "benchmark_data",
    "hf_cache_dir": None,
    "benchmarks": {
        "typo": True,
        "objectnet_mvt": True,
        "mscoco": True,
        "sugarcrepe": True,
        "imagenet_linear_probe": False,
    },
    "datasets": {
        "imagenet_root": None,
        "objectnet_mvt_root": None,
        "coco_val2014_root": None,
        "coco_val2017_root": None,
    },
}

BENCHMARK_ORDER = (
    "typo",
    "objectnet_mvt",
    "mscoco",
    "sugarcrepe",
    "imagenet_linear_probe",
)


def _merge_defaults(default: Any, value: Any) -> Any:
    if isinstance(default, dict):
        result = copy.deepcopy(default)
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in result:
                    result[key] = _merge_defaults(result[key], item)
                else:
                    result[key] = copy.deepcopy(item)
        return result
    return copy.deepcopy(default if value is None else value)


def load_config(path: Path = DEFAULT_CONFIG_PATH, *, allow_missing: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if allow_missing:
            return copy.deepcopy(DEFAULT_CONFIG)
        raise FileNotFoundError(
            f"Benchmark config not found: {path}. Run `python benchmark.py setup` first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    cfg = _merge_defaults(DEFAULT_CONFIG, raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    model = cfg.get("model")
    if not isinstance(model, Mapping) or not str(model.get("path") or "").strip():
        raise ValueError("benchmark config requires model.path")
    if not str(model.get("alias") or "").strip():
        raise ValueError("benchmark config requires model.alias")
    benches = cfg.get("benchmarks")
    if not isinstance(benches, Mapping):
        raise ValueError("benchmark config requires benchmarks object")
    for name in BENCHMARK_ORDER:
        if name not in benches:
            raise ValueError(f"benchmark config missing benchmarks.{name}")
    datasets = cfg.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("benchmark config requires datasets object")


def save_config(cfg: Mapping[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> None:
    validate_config(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(cfg), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def optional_path(value: Any, project_root: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def required_path(value: Any, project_root: Path, label: str) -> Path:
    path = optional_path(value, project_root)
    if path is None:
        raise FileNotFoundError(f"{label} is not configured")
    return path


def training_seed_paths(training_config: Path) -> dict[str, Any]:
    raw = json.loads(training_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("paths"), dict):
        raise ValueError(f"Unexpected training config schema: {training_config}")
    paths = raw["paths"]
    return {
        "imagenet_root": paths.get("imagenet_root"),
        "imagenet_train_root_override": paths.get("imagenet_train_root_override"),
        "imagenet_val_root_override": paths.get("imagenet_val_root_override"),
        "mvt_image_root": paths.get("mvt_image_root"),
        "scam_labels_csv": paths.get("scam_labels_csv"),
        "scam_images_root": paths.get("scam_images_root"),
        # This is SPRIGHT/COCO training data, not the benchmark releases.  It is
        # returned only so setup can explain why it deliberately is not reused.
        "training_coco_root": paths.get("coco_root"),
    }
