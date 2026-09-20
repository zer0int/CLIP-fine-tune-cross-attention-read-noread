from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hf_export_reader_masking_is_autograd_safe():
    text = (ROOT / "hf_export" / "modeling_xattn_clip.py").read_text(encoding="utf-8")
    assert "patch_attention *=" not in text
    assert "register_attention *=" not in text
    assert "patch_attention = patch_attention * patch_keep" in text
    assert "register_attention = register_attention * register_keep" in text


def test_refit_uses_native_full_xattn_remote_code_without_topology_wrapper():
    text = (ROOT / "train_refit_late_reader.py").read_text(encoding="utf-8")
    assert "configuration_xattn_clip.py" in text
    assert "modeling_xattn_clip.py" in text
    assert "late_read_refit" not in text
    assert "EXPECTED_NATIVE_LATE = (20, 21)" in text
    assert "topology is immutable" in text


def test_wise_export_is_explicitly_no_evaluation():
    text = (ROOT / "train_wiseft_late_reader.py").read_text(encoding="utf-8")
    assert 'export = subparsers.add_parser("export", help="save explicit alpha checkpoints; no evaluation")' in text
    assert 'export.set_defaults(handler=run_export)' in text
    assert '"evaluation": "NONE"' in text
    assert "add_eval_args(export)" not in text


def test_current_single_config_json_is_audited_not_silently_ignored():
    path = ROOT / "hf_export" / "checkpoint_spec.py"
    spec = importlib.util.spec_from_file_location("checkpoint_spec_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = {
        "global": {
            "read_attention_architecture": "sigmoid_all",
            "read_null_enabled": True,
            "read_null_insert_block": 13,
            "image_size": 224,
            "tap_blocks": {
                "late": [20, 21],
                "orthographic": [8, 12, 13],
                "source": [6, 7, 10],
            },
        }
    }
    assert module._training_json_value(cfg, "read_attention_architecture") == "sigmoid_all"
    assert module._training_json_value(cfg, "read_tap_blocks") == [20, 21]
    assert module._training_json_value(cfg, "ortho_tap_blocks") == [8, 12, 13]
    assert module._training_json_value(cfg, "source_tap_blocks") == [6, 7, 10]


def test_refit_training_core_supports_current_sibling_imports():
    text = (ROOT / "train_refit_late_reader.py").read_text(encoding="utf-8")
    assert 'training_dir = repo_root / "main_hard_text_gate"' in text
    assert 'training_dir / "coco_trusted_reading_manifest.py"' in text
    assert 'training_dir / "router_auxiliary.py"' in text
    assert 'for import_root in (repo_root, training_dir):' in text
    assert 'sys.path.insert(0, value)' in text
    assert 'from main_hard_text_gate import train_gmp_anytext_stage1 as core' in text
