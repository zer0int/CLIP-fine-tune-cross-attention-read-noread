from __future__ import annotations

import importlib.util
import json

import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _launcher_module():
    spec = importlib.util.spec_from_file_location(
        "train_clip_xattn_bridge", ROOT / "train_clip_xattn_bridge.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authoritative_checkpoint_config_is_reconciled() -> None:
    saved_path = ROOT / "config.json"
    if not saved_path.is_file():
        pytest.skip("historical config.json is not shipped in this source package")
    active = json.loads((ROOT / "training_config.json").read_text(encoding="utf-8"))
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert active["global"]["base_gmp_model"] == "zer0int/CLIP-GmP-ViT-L-14"
    assert active["global"]["read_null_insert_block"] == 13
    assert saved["read_null_insert_block"] == 13
    assert active["global"]["tap_blocks"]["late"] == saved["read_tap_blocks"]
    assert active["global"]["tap_blocks"]["orthographic"] == saved["ortho_tap_blocks"]
    assert active["global"]["tap_blocks"]["source"] == saved["source_tap_blocks"]


def test_resolved_all_weights_hyperparameters_match_saved_config() -> None:
    saved_path = ROOT / "config.json"
    if not saved_path.is_file():
        pytest.skip("historical config.json is not shipped in this source package")
    launcher = _launcher_module()
    active = json.loads((ROOT / "training_config.json").read_text(encoding="utf-8"))
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    resolved, _ = launcher.build_stage_args(
        active, "all_weights", ROOT / "_runtime" / "branch_config.generated.json"
    )
    environment_fields = {
        "amp_dtype_requested",
        "branch_config",
        "clevr_metadata_jsonl",
        "clevr_train_root",
        "clevr_val_root",
        "clip_package_root",
        "coco_root",
        "coco_train_gpt_json",
        "coco_train_json",
        "coco_train_reading_json",
        "coco_val_gpt_json",
        "coco_val_json",
        "imagenet_handwriting_root",
        "imagenet_text_root",
        "imagenet_wnid_json",
        "implant_checkpoint",
        "model_path",
        "mvt_csv",
        "mvt_image_root",
        "out_dir",
        "require_legacy_migration_input",
        "router_aux_salt_n_pepper_root",
        "benchmark_scam_repo",
        "benchmark_scam_revision",
        "benchmark_rta_repo",
        "benchmark_rta_revision",
        "benchmark_include_mvt",
        "run_benchmark_validation",
        "textcaps_root",
    }
    differences = {
        key: (resolved.get(key), saved.get(key))
        for key in set(resolved) | set(saved)
        if key not in environment_fields and resolved.get(key) != saved.get(key)
    }
    assert differences == {}


def test_legacy_hard_token_stage_keeps_vanilla_clip_package() -> None:
    source = (
        ROOT
        / "main_hard_text_gate"
        / "legacy_hard_text_pre"
        / "pre1_train_clip_soft_text_token_imagenet.py"
    ).read_text(encoding="utf-8")
    assert "import oaicliporg as clip" in source
    assert '"reuse_full_model_pickle": False' in source


def test_precision_diagnostics_has_one_canonical_copy() -> None:
    matches = list(ROOT.rglob("training_precision_diagnostics.py"))
    assert matches == [
        ROOT / "training_support" / "diagnostics" / "training_precision_diagnostics.py"
    ]


def test_checked_in_training_config_has_no_maintainer_dataset_paths() -> None:
    active = json.loads((ROOT / "training_config.json").read_text(encoding="utf-8"))
    paths = active["paths"]
    # External datasets and pipeline-internal router artifacts are deliberately pathless.
    for key, value in paths.items():
        if key == "custom_fonts":
            assert value == []
            continue
        assert value is None, (key, value)
    router = active["router_utility"]
    assert router["start_checkpoint"] is None
    assert router["output_dir"] is None
    assert router["base_model_override"] is None


def test_typo_benchmarks_are_hf_backed_and_monitoring_only() -> None:
    active = json.loads((ROOT / "training_config.json").read_text(encoding="utf-8"))
    final = active["shared_args"]["final_anytext"]
    assert final["benchmark_scam_repo"] == "BLISS-e-V/SCAM"
    assert final["benchmark_rta_repo"] == "zer0int/RTA-100-Triplet"
    assert final["select_by_benchmark_score"] is False
    assert active["stages"]["final_base"]["args"]["run_benchmark_validation"] is True
    assert active["stages"]["all_weights"]["args"]["run_benchmark_validation"] is True


def test_router_artifacts_are_derived_from_current_train_root() -> None:
    launcher = _launcher_module()
    active = json.loads((ROOT / "training_config.json").read_text(encoding="utf-8"))
    active["global"]["train_root"] = "outputs/router_derivation_probe"
    artifacts = launcher.artifact_paths(active)
    expected_suffix = "outputs/router_derivation_probe/final/sigmoid_all/base/router_utility_a4_best.pt"
    assert str(artifacts["router_utility_a4"]).replace("\\", "/").endswith(expected_suffix)

    resolved, required = launcher.build_stage_args(
        active, "final_base", ROOT / "_runtime" / "branch_config.generated.json"
    )
    assert str(resolved["router_aux_init_checkpoint"]).replace("\\", "/").endswith(expected_suffix)
    assert any(str(item).replace("\\", "/").endswith(expected_suffix) for item in required)


def test_prepare_local_config_keeps_router_artifacts_derived(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "prepare_training_data_test", ROOT / "prepare_training_data.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    output = tmp_path / "training_config.local.json"
    module.generate_local_config(
        ROOT,
        ROOT / "training_config.json",
        output,
        tmp_path / "data",
        tmp_path / "imagenet",
        "scam-revision",
        "rta-revision",
    )
    local = json.loads(output.read_text(encoding="utf-8"))
    assert local["paths"]["router_utility_a4_checkpoint"] is None
    assert local["router_utility"]["start_checkpoint"] is None
    assert local["router_utility"]["output_dir"] is None
    assert local["router_utility"]["base_model_override"] is None
