from __future__ import annotations

import json
from pathlib import Path

from benchmark_utils.config import DEFAULT_CONFIG, load_config, save_config, training_seed_paths
from benchmark_utils.data_setup import validate_imagenet_root


def test_default_config_is_single_model_and_benchmark_first() -> None:
    assert DEFAULT_CONFIG["model"]["path"] == "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
    assert DEFAULT_CONFIG["output_root"] == "out_bench_results"
    assert DEFAULT_CONFIG["benchmarks"]["typo"] is True
    assert DEFAULT_CONFIG["benchmarks"]["imagenet_linear_probe"] is False


def test_training_seed_does_not_treat_training_coco_as_benchmark_coco(tmp_path: Path) -> None:
    p = tmp_path / "training_config.local.json"
    p.write_text(json.dumps({"paths": {
        "imagenet_root": "ILSVRC2012",
        "mvt_image_root": "MVT/all",
        "coco_root": "SPRIGHT/COCO/data-square",
    }}), encoding="utf-8")
    seed = training_seed_paths(p)
    assert seed["imagenet_root"] == "ILSVRC2012"
    assert seed["mvt_image_root"] == "MVT/all"
    assert seed["training_coco_root"] == "SPRIGHT/COCO/data-square"
    assert "coco_val2014_root" not in seed
    assert "coco_val2017_root" not in seed


def test_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "benchmark_config.json"
    save_config(DEFAULT_CONFIG, path)
    got = load_config(path)
    assert got == DEFAULT_CONFIG
