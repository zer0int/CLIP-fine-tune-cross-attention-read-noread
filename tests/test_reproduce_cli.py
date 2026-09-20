from __future__ import annotations

from pathlib import Path
import sys

import reproduce
from reproduction_utils.catalog import TASKS, TASK_BY_ID, Task
from reproduction_utils.config import DEFAULT_CONFIG
from reproduction_utils.objectnet_mvt import (
    EXPECTED_MVT_IMAGES,
    _balanced_sample,
    load_dedup_rows,
)


def test_reproduction_catalog_is_closed_and_acyclic():
    ids = [task.id for task in TASKS]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(TASK_BY_ID)

    outputs = [task.output_rel for task in TASKS if task.output_rel is not None]
    from collections import Counter
    duplicates = {name for name, count in Counter(outputs).items() if count > 1}
    assert duplicates == {"conv1/roleplane_texture_rank"}

    visiting: set[str] = set()
    done: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in done:
            return
        assert task_id not in visiting, f"dependency cycle at {task_id}"
        visiting.add(task_id)
        for dep in TASK_BY_ID[task_id].deps:
            assert dep in TASK_BY_ID
            visit(dep)
        visiting.remove(task_id)
        done.add(task_id)

    for task_id in ids:
        visit(task_id)


def test_every_catalog_script_exists():
    for task in TASKS:
        assert (reproduce.PROBE_ROOT / task.script).is_file(), task.id




def test_rn_control_surfaces_exports_ply_by_default(tmp_path: Path):
    cfg = {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"]},
        "datasets": {**DEFAULT_CONFIG["datasets"]},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }
    args = reproduce._auto_args(TASK_BY_ID["rn.control_surfaces"], cfg)
    assert "--export-ply" in args

def test_rn_touch_go_has_portable_checkpoint_overrides():
    text = (reproduce.PROBE_ROOT / "probe_rn_touch_go_transfer_paper.py").read_text(encoding="utf-8")
    assert '"--donor-checkpoint"' in text
    assert '"--openai-spec"' in text
    assert '"--gmp-checkpoint"' in text


def test_reproduction_debris_removed():
    root = reproduce.PROJECT_ROOT
    assert not (root / "rn_control_mechinterp").exists()
    assert not (reproduce.PROBE_ROOT / "precision_policy.py").exists()
    assert not (reproduce.PROBE_ROOT / "reference_lookup.txt").exists()
    assert not (reproduce.PROBE_ROOT / "probe_dataset_manifest.py").exists()


def test_objectnet_population_consumers_are_exactly_the_workspace_population_tasks():
    consumers = {
        task.id
        for task in TASKS
        if "datasets.objectnet_mvt" in task.requirements
    }
    assert consumers == {
        "workspace.broadcast_sinks",
        "workspace.cls_register_exchange",
        "workspace.cls_mu_causal",
    }
    assert "datasets.objectnet_mvt" not in TASK_BY_ID["workspace.broadcast_channel_interventions"].requirements


def test_bundled_mvt_index_is_unique_and_balanced_source_pool():
    rows = load_dedup_rows(reproduce.PROJECT_ROOT)
    assert len(rows) == EXPECTED_MVT_IMAGES == 4771
    assert len({row["image"] for row in rows}) == EXPECTED_MVT_IMAGES
    assert len({row["label"] for row in rows}) == 50
    domains = {domain: sum(row["domain"] == domain for row in rows) for domain in {"objectnet", "imagenet"}}
    assert domains == {"objectnet": 2415, "imagenet": 2356}


def test_workspace_mvt_sample_is_deterministic_label_domain_balanced():
    rows = load_dedup_rows(reproduce.PROJECT_ROOT)
    a = _balanced_sample(rows, 480, 20260915)
    b = _balanced_sample(rows, 480, 20260915)
    assert [row["image"] for row in a] == [row["image"] for row in b]
    assert len(a) == 480

    counts: dict[tuple[str, str], int] = {}
    domains = {"objectnet": 0, "imagenet": 0}
    for row in a:
        key = (row["label"], row["domain"])
        counts[key] = counts.get(key, 0) + 1
        domains[row["domain"]] += 1
    assert len(counts) == 100
    assert min(counts.values()) == 4
    assert max(counts.values()) == 5
    assert domains == {"objectnet": 240, "imagenet": 240}


def test_objectnet_reproduction_reuses_benchmark_installer_not_parallel_downloader():
    helper = (reproduce.PROJECT_ROOT / "reproduction_utils" / "objectnet_mvt.py").read_text(encoding="utf-8")
    front = (reproduce.PROJECT_ROOT / "reproduce.py").read_text(encoding="utf-8")
    assert "download_once" not in helper
    assert "STIMULI_URL" not in helper
    assert "from benchmark_utils.data_setup import install_objectnet_mvt" in front


def test_utils_datasets_mvt_is_bundled():
    assert (reproduce.PROJECT_ROOT / "utils_datasets" / "mvt" / "human_responses_dedup.csv").is_file()


def test_manual_utilities_have_no_implicit_auto_args():
    cfg = {
        **DEFAULT_CONFIG,
        "datasets": {**DEFAULT_CONFIG["datasets"]},
        "models": {**DEFAULT_CONFIG["models"]},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }
    for task in TASKS:
        if task.manual_args:
            assert reproduce._auto_args(task, cfg) == []


def test_every_top_level_reproduction_script_is_public_or_declared_helper():
    all_scripts = {p.name for p in reproduce.PROBE_ROOT.glob("*.py")}
    public_scripts = {task.script for task in TASKS if "/" not in task.script}
    helpers = {"__init__.py", "benchmark_final_clip.py", "probe_tools_analysis.py", "probe_tools_backbone.py", "probe_tools_repo.py"}
    assert all_scripts == public_scripts | helpers


def test_default_models_have_distinct_vanilla_xattn_and_gmp_roles():
    assert DEFAULT_CONFIG["models"]["final_hf"] == "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
    assert DEFAULT_CONFIG["models"]["vanilla_spec"] == "ViT-L/14"
    assert DEFAULT_CONFIG["models"]["vanilla_hf"] == "openai/clip-vit-large-patch14"
    assert DEFAULT_CONFIG["models"]["gmp_hf"] == "zer0int/CLIP-GmP-ViT-L-14"


def test_gmp_local_checkpoint_can_fall_back_to_hf_materialization(tmp_path: Path):
    cfg = {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"], "gmp_checkpoint": None},
        "datasets": {**DEFAULT_CONFIG["datasets"]},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }
    ok, note = reproduce._requirement_status("models.gmp_checkpoint", cfg)
    assert ok
    assert "zer0int/CLIP-GmP-ViT-L-14" in note
    assert reproduce._gmp_materialized_path(cfg) == tmp_path / "out" / "_models" / "gmp_trained_vanilla.pt"


def test_backbone_dynamics_uses_configured_gmp_hf_baseline(tmp_path: Path):
    cfg = {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"]},
        "datasets": {
            **DEFAULT_CONFIG["datasets"],
            "demoset_dir": str(tmp_path / "demo"),
        },
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }
    args = reproduce._auto_args(TASK_BY_ID["bridge.backbone_dynamics"], cfg)
    idx = args.index("--pretrained")
    assert args[idx + 1] == "zer0int/CLIP-GmP-ViT-L-14"


def test_output_root_cli_override_is_ephemeral(tmp_path: Path):
    config_path = tmp_path / "reproduction_config.json"
    original_root = tmp_path / "original"
    override_root = tmp_path / "override"
    config_path.write_text(
        __import__("json").dumps({"output_root": str(original_root)}), encoding="utf-8"
    )
    args = reproduce.build_parser().parse_args(
        ["--config", str(config_path), "status", "paper", "--output-root", str(override_root)]
    )
    cfg = reproduce._load_command_config(args)
    assert Path(cfg["output_root"]) == override_root
    assert __import__("json").loads(config_path.read_text(encoding="utf-8"))["output_root"] == str(original_root)


def test_xattn_local_checkpoint_can_fall_back_to_final_hf(tmp_path: Path):
    cfg = {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"], "xattn_checkpoint": None},
        "datasets": {**DEFAULT_CONFIG["datasets"]},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }
    ok, note = reproduce._requirement_status("models.xattn_checkpoint", cfg)
    assert ok
    assert DEFAULT_CONFIG["models"]["final_hf"] in note
    assert reproduce._xattn_materialized_path(cfg) == tmp_path / "out" / "_models" / "xattn_full_trained.pt"


def test_rn_native_loader_uses_merged_xattn_package():
    text = (reproduce.PROBE_ROOT / "rn_control_mechinterp" / "core.py").read_text(encoding="utf-8")
    assert 'importlib.import_module("attnclip_mechinterp_xattn.clip")' in text


def test_model_family_implementations_stay_separate():
    # Vanilla CLIP with Q/K/V capture uses the vanilla SAE/mechinterp implementation.
    assert (reproduce.PROJECT_ROOT / "attnclip_mechinterp_sae" / "model.py").is_file()
    backbone = (reproduce.PROBE_ROOT / "probe_tools_backbone.py").read_text(encoding="utf-8")
    assert 'DEFAULT_CLIP_MODULE = "attnclip_mechinterp_sae"' in backbone
    assert "back to stock openai/clip" in backbone

    # RN/x-attn native tooling uses the x-attn implementation.
    rn_core = (reproduce.PROBE_ROOT / "rn_control_mechinterp" / "core.py").read_text(encoding="utf-8")
    assert 'importlib.import_module("attnclip_mechinterp_xattn.clip")' in rn_core

    # Ordinary vanilla and concat-attn packages are intentionally distinct too.
    # Touch-and-go receivers are ordinary vanilla CLIP; the x-attn donor is now
    # state-dict-only and must not instantiate the concat/x-attn runtime at all.
    touch_go = (reproduce.PROBE_ROOT / "probe_rn_touch_go_transfer_paper.py").read_text(encoding="utf-8")
    assert "import oaicliporg as legacy_clip" in touch_go
    assert "import oaiclip as xclip" not in touch_go
    assert (reproduce.PROJECT_ROOT / "oaiclip" / "model.py").is_file()
    assert (reproduce.PROJECT_ROOT / "main_hard_text_gate" / "legacy_hard_text_pre" / "oaicliporg" / "model.py").is_file()


def test_gmp_is_only_required_by_explicit_comparison_tasks():
    actual = {
        task.id for task in TASKS
        if "models.gmp_checkpoint" in task.requirements or "models.gmp_hf" in task.requirements
    }
    expected = {
        "bridge.backbone_dynamics",
        "workspace.broadcast_sinks",
        "workspace.broadcast_channel_interventions",
        "workspace.cls_register_exchange",
        "workspace.cls_mu_causal",
        "workspace.cls_role_surfaces",
        "workspace.qk_role_gates.scan",
        "workspace.qk_role_gates.same_heads",
        "workspace.register_geometry.secondary",
        "workspace.register_geometry.grad_attention",
        "workspace.rta_head_population",
        "workspace.single_image.example",
        "workspace.single_image.motifs",
        "workspace.text_cls_trajectory",
        "rn.touch_go_transfer",
        "conv1.vanilla_functional_atlas",
        "conv1.vanilla_functional_atlas_rn",
        "conv1.residual_axis_lineage",
        "conv1.residual_axis_swap_650_565",
        "conv1.residual_axis_lineage_rn",
        "conv1.mlp_neuron_discovery",
        "conv1.b20_writeback_neurons",
        "conv1.b20_sharpeners_flatteners",
        "conv1.b20_pushpull_650_715",
        "conv1.visualtextual_provenance",
        "conv1.visualtextual_text_direction",
    }
    assert actual == expected


def test_generic_pretrained_rn_controls_use_openai_not_gmp(tmp_path: Path):
    cfg = {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"]},
        "datasets": {**DEFAULT_CONFIG["datasets"]},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }
    stash = reproduce._auto_args(TASK_BY_ID["rn.stash_followup"], cfg)
    rn_path = Path(stash[stash.index("--pretrained-rn-checkpoint") + 1])
    assert rn_path.name == "oai_vanilla_rn_from_xattn.pt"
    assert rn_path.parent.name == "_models"
    bridge = reproduce._auto_args(TASK_BY_ID["rn.bridge_transplant"], cfg)
    assert bridge[bridge.index("--pretrained-module") + 1] == "attnclip_mechinterp_sae"
    assert bridge[bridge.index("--pretrained-spec") + 1] == "ViT-L/14"


def test_openai_vitl14_has_official_artifact_fingerprint():
    from reproduction_utils.model_identity import fingerprint_model_spec
    identity = fingerprint_model_spec("ViT-L/14", project_root=reproduce.PROJECT_ROOT)
    assert identity["kind"] == "openai_clip"
    assert identity["sha256"] == "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd0bfd2b27167a4a06ec9aa"


def test_hf_repo_ids_are_not_parsed_as_filesystem_paths():
    remote_cli_args = {
        "probe_cross_attention_bridge.py": ("--model", "--old-model"),
        "probe_backbone_dynamics.py": ("--xattn-model",),
        "probe_read_null_controls.py": ("--correction-model", "--full-model", "--model"),
        "probe_register_cache_transport.py": ("--full-model",),
        "probe_rn_text_relocation.py": ("--full-model",),
    }
    for filename, option_names in remote_cli_args.items():
        text = (reproduce.PROBE_ROOT / filename).read_text(encoding="utf-8")
        for option_name in option_names:
            assert f'{option_name}", type=Path' not in text, (filename, option_name)
        assert 'Path("zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX")' not in text

    bridge_text = (reproduce.PROBE_ROOT / "probe_cross_attention_bridge.py").read_text(encoding="utf-8")
    assert 'model_path / "WISE_FT_METADATA.json"' not in bridge_text


def test_read_null_correction_control_supports_final_xattn_content_branch():
    text = (reproduce.PROBE_ROOT / "probe_read_null_controls.py").read_text(encoding="utf-8")
    assert '"xattn_content_branch"' in text
    assert 'model.read_implant.content_pool' in text
    assert 'model.read_implant.content_tap_logits' in text
    assert 'model._vision_with_intermediates(' in text
    assert 'return_final_tokens=True' in text
    assert 'model_meta["correction_backend"]' in text


def test_all_selector_runs_every_automatic_task_but_not_manual_audit():
    selected = reproduce._resolve_selectors(["all"], include_controls=False)
    ids = {task.id for task in selected}
    assert "bridge.read_null.diagnostic" in ids  # controls are included by `all`
    assert "workspace.register_cache_transport" in ids
    assert "audit.read_probe_router" not in ids
    assert all(not task.manual_args for task in selected)


def test_selector_help_advertises_all_families_and_exact_tasks():
    text = reproduce._selector_help_text()
    assert "paper" in text
    assert "all" in text
    for family in reproduce.FAMILIES:
        assert family in text
    for task in TASKS:
        assert task.id in text


def test_bridge_auto_args_allow_safe_same_directory_rerun(tmp_path: Path):
    cfg = {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"]},
        "datasets": {**DEFAULT_CONFIG["datasets"], "demoset_dir": str(tmp_path / "demo")},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }
    assert "--overwrite" in reproduce._auto_args(TASK_BY_ID["bridge.cross_attention"], cfg)
    assert "--overwrite" in reproduce._auto_args(TASK_BY_ID["bridge.backbone_dynamics"], cfg)


def test_model_identity_guard_allows_same_hash_and_rejects_mismatch(tmp_path: Path):
    from reproduction_utils.model_identity import guard_and_merge_identities, load_identity_manifest

    root = tmp_path / "out"
    root.mkdir()
    (root / "partial.txt").write_text("old partial output", encoding="utf-8")
    first = {"final_hf": {"source": "model-A", "sha256": "aaa", "hash_kind": "test", "kind": "test"}}
    path, notes = guard_and_merge_identities(root, first)
    assert path.is_file()
    assert notes  # legacy/partial roots are adopted rather than rejected

    second = {"final_hf": {"source": "model-A-again", "sha256": "aaa", "hash_kind": "test", "kind": "test"}}
    guard_and_merge_identities(root, second)
    assert load_identity_manifest(root)["models"]["final_hf"]["sha256"] == "aaa"

    mismatch = {"final_hf": {"source": "model-B", "sha256": "bbb", "hash_kind": "test", "kind": "test"}}
    import pytest
    with pytest.raises(RuntimeError, match="MODEL IDENTITY MISMATCH"):
        guard_and_merge_identities(root, mismatch)


def test_local_checkpoint_fingerprint_changes_with_bytes(tmp_path: Path):
    from reproduction_utils.model_identity import fingerprint_local

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"apple")
    a = fingerprint_local(checkpoint)
    checkpoint.write_bytes(b"banana")
    b = fingerprint_local(checkpoint)
    assert a["hash_kind"] == "sha256"
    assert a["sha256"] != b["sha256"]


def test_bridge_cache_markers_point_to_real_output_locations():
    bridge = TASK_BY_ID["bridge.cross_attention"]
    backbone = TASK_BY_ID["bridge.backbone_dynamics"]
    assert bridge.markers == ("rollout_classic_b23/metrics.csv", "pairwise_deltas/map_delta_metrics.csv")
    assert backbone.markers == ("summary.txt",)


def test_probe_subprocess_env_exposes_repository_root(monkeypatch):
    import os
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["already", "there"]))
    env = reproduce._child_env()
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(reproduce.PROJECT_ROOT)
    assert parts[1:] == ["already", "there"]


def test_all_reproduction_entrypoints_bootstrap_repository_root():
    helper = reproduce.PROBE_ROOT / "probe_tools_repo.py"
    assert helper.is_file()
    helper_text = helper.read_text(encoding="utf-8")
    assert "Path(__file__).resolve().parent.parent" in helper_text
    assert "sys.path.insert(0, root_text)" in helper_text

    for task in TASKS:
        text = (reproduce.PROBE_ROOT / task.script).read_text(encoding="utf-8")
        if "/" not in task.script:
            assert "from probe_tools_repo import ensure_repo_root" in text, task.script
            assert "ensure_repo_root()" in text, task.script
        else:
            assert ("find_repo_root" in text or "clip_probe_common" in text or
                    "visualtextual_probe_common" in text or
                    ("oaicliporg" in text and "utils_clip_loader" in text) or
                    task.id.endswith(".compact")), task.script


def test_gmp_materialization_recovers_effective_weight():
    import torch
    theta = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    radius = torch.tensor([[10.0], [6.0]])
    state = {
        "visual.transformer.resblocks.0.mlp.c_fc.theta": theta,
        "visual.transformer.resblocks.0.mlp.c_fc.r": radius,
        "visual.transformer.resblocks.0.mlp.c_fc.bias": torch.zeros(2),
    }
    out, converted = reproduce._materialize_gmp_weights(state)
    key = "visual.transformer.resblocks.0.mlp.c_fc.weight"
    assert converted == 1
    assert key in out
    assert not any(k.endswith((".theta", ".r")) for k in out)
    expected = torch.tensor([[6.0, 8.0], [0.0, 6.0]])
    assert torch.allclose(out[key], expected)


def test_existing_gmp_cache_is_upgraded_in_place(tmp_path: Path):
    import torch
    path = tmp_path / "gmp_openai_state_dict.pt"
    theta_key = "visual.transformer.resblocks.0.mlp.c_proj.theta"
    radius_key = "visual.transformer.resblocks.0.mlp.c_proj.r"
    torch.save({theta_key: torch.eye(2), radius_key: torch.tensor([[2.0], [3.0]])}, path)
    returned = reproduce._canonicalize_cached_gmp_checkpoint(path)
    assert returned == path
    try:
        out = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        out = torch.load(path, map_location="cpu")
    assert "visual.transformer.resblocks.0.mlp.c_proj.weight" in out
    assert theta_key not in out and radius_key not in out


def test_workspace_transplant_materializes_gmp_parameterization():
    text = (reproduce.PROBE_ROOT / "probe_tools_backbone.py").read_text(encoding="utf-8")
    assert "def _materialize_gmp_weights" in text
    assert "gmp_geometric_matrices_materialized" in text


def test_broadcast_sinks_reuses_completed_model_legs():
    text = (reproduce.PROBE_ROOT / "probe_broadcast_sinks.py").read_text(encoding="utf-8")
    assert "def _model_extraction_complete" in text
    assert "cached extraction complete" in text
    assert "incomplete prior model leg" in text
    assert 'default=False, help="force rerunning even fully completed per-model extraction legs"' in text


def test_gmp_model_name_does_not_select_gmp_runtime_module():
    root = Path(__file__).resolve().parents[1]
    text = (root / "reproduce.py").read_text(encoding="utf-8")
    assert "import gmpclipattnamp" not in text
    assert "resolve_to_openai_state_dict" in text
    block = text[text.index("def _ensure_gmp_checkpoint"):text.index("def _requirement_status")]
    assert "GmP is training provenance" in block or "``GmP`` is training provenance" in block


def _tiny_vanilla_clip_state():
    import torch
    return {
        "visual.conv1.weight": torch.zeros(1, 1, 1, 1),
        "visual.class_embedding": torch.zeros(1),
        "visual.positional_embedding": torch.zeros(2, 1),
        "token_embedding.weight": torch.zeros(2, 1),
        "positional_embedding": torch.zeros(2, 1),
    }


def test_gmp_cache_rejects_xattn_runtime_keys_and_rebuilds(tmp_path: Path, monkeypatch):
    import sys
    import types
    import torch
    fake_colorama = types.ModuleType("colorama")
    fake_colorama.Fore = types.SimpleNamespace(RED="", GREEN="", YELLOW="", CYAN="", MAGENTA="")
    fake_colorama.Style = types.SimpleNamespace(RESET_ALL="")
    monkeypatch.setitem(sys.modules, "colorama", fake_colorama)
    import utils_clip_loader.clip_anything_to_openai as conv

    cfg = {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"], "gmp_checkpoint": None},
        "datasets": {**DEFAULT_CONFIG["datasets"]},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }
    cache = reproduce._gmp_materialized_path(cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    poisoned = _tiny_vanilla_clip_state()
    poisoned["hard_text_embedding"] = torch.ones(1)
    poisoned["read_implant.read_probe"] = torch.ones(1)
    torch.save(poisoned, cache)

    clean = _tiny_vanilla_clip_state()
    info = types.SimpleNamespace(source_kind="hf_state_dict", detected_format="hf", model_family="vanilla")
    monkeypatch.setattr(conv, "resolve_to_openai_state_dict", lambda spec: (dict(clean), info))

    resolved = reproduce._ensure_gmp_checkpoint(cfg)
    assert resolved == cache
    out = reproduce._checkpoint_state(cache)
    family, custom = reproduce._checkpoint_family_from_keys(out)
    assert family == "vanilla"
    assert custom == []
    meta = reproduce._checkpoint_cache_meta_path(cache)
    assert meta.is_file()
    payload = __import__("json").loads(meta.read_text(encoding="utf-8"))
    assert payload["variant_id"] == "gmp_trained_vanilla"
    assert payload["role"] == "gmp_trained_vanilla"
    assert payload["verified_from_source_this_run"] is True
    assert payload["artifact_sha256"] == reproduce._sha256_file(cache)


def test_gmp_cache_family_detection_flags_read_implant_keys(tmp_path: Path):
    import torch
    path = tmp_path / "fake_gmp.pt"
    state = _tiny_vanilla_clip_state()
    state["read_implant.content_tap_logits"] = torch.ones(1)
    torch.save(state, path)
    ok, detail = reproduce._validate_checkpoint_family(path, "vanilla")
    assert not ok
    assert "expected 'vanilla'" in detail
    assert "read_implant.content_tap_logits" in detail


def test_model_variant_registry_is_closed_for_declared_task_plans():
    from reproduction_utils.model_variants import VARIANTS, TASK_MODEL_VARIANTS
    task_ids = {task.id for task in TASKS}
    assert set(TASK_MODEL_VARIANTS).issubset(task_ids)
    for task_id, variants in TASK_MODEL_VARIANTS.items():
        assert variants, task_id
        for variant_id in variants:
            assert variant_id in VARIANTS, (task_id, variant_id)


def test_xattn_family_requires_complete_bridge_not_just_one_custom_key():
    import torch
    state = _tiny_vanilla_clip_state()
    state["visual.read_null_token"] = torch.zeros(1)
    state["visual.read_null_insert_block_config"] = torch.tensor(13)
    state["hard_text_embedding"] = torch.zeros(1)
    family, _ = reproduce._checkpoint_family_from_keys(state)
    assert family == "partial_custom"
    ok, detail = reproduce._validate_state_family(state, "xattn_full")
    assert not ok
    assert "partial_custom" in detail


def test_rn_only_family_forbids_bridge_and_requires_exact_rn_config():
    import torch
    state = _tiny_vanilla_clip_state()
    state["visual.read_null_token"] = torch.zeros(1)
    state["visual.read_null_insert_block_config"] = torch.tensor(13)
    family, custom = reproduce._checkpoint_family_from_keys(state)
    assert family == "rn_only"
    assert set(custom) == {"visual.read_null_token", "visual.read_null_insert_block_config"}
    state["read_implant.read_probe"] = torch.zeros(1)
    family, _ = reproduce._checkpoint_family_from_keys(state)
    assert family == "correction_or_partial"


def test_stash_followup_refuses_random_rn_or_bridge_construction():
    text = (reproduce.PROBE_ROOT / "probe_rn_stash_followup.py").read_text(encoding="utf-8")
    assert '"--pretrained-rn-checkpoint"' in text
    assert "read_null_enabled=True" not in text
    assert "does not contain the exact trained donor RN token" in text
    assert "unexpectedly contains a bridge/read_implant" in text


def test_bridge_transplant_preserves_trained_custom_state_exactly():
    text = (reproduce.PROBE_ROOT / "probe_rn_bridge_transplant.py").read_text(encoding="utf-8")
    assert 'PROTECTED_PREFIXES = (' in text
    assert '"read_implant."' in text
    assert '"visual.read_null_token"' in text
    assert '"hard_text_embedding"' in text
    assert "protected_state_digest_before" in text
    assert "protected_state_digest_after" in text
    assert "Protected PIECES state changed during transplant" in text


def test_model_cache_policy_defaults_to_keep_and_has_cleanup_modes():
    assert DEFAULT_CONFIG["runtime"]["model_cache_policy"] == "keep"
    parser = reproduce.build_parser()
    for policy in ("keep", "task", "run"):
        args = parser.parse_args(["run", "bridge.cross_attention", "--model-cache-policy", policy, "--dry-run"])
        assert args.model_cache_policy == policy


def test_cleanup_only_deletes_reproduce_managed_model_caches(tmp_path: Path):
    import json
    import torch
    cfg = {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"]},
        "datasets": {**DEFAULT_CONFIG["datasets"]},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }
    root = tmp_path / "out" / "_models"
    root.mkdir(parents=True)
    managed = root / "xattn_full_trained.pt"
    torch.save(_tiny_vanilla_clip_state(), managed)
    (root / "xattn_full_trained.pt.meta.json").write_text(
        json.dumps({"managed_by_reproduce": True}), encoding="utf-8"
    )
    user = root / "user_checkpoint.pt"
    torch.save(_tiny_vanilla_clip_state(), user)
    (root / "user_checkpoint.pt.meta.json").write_text(
        json.dumps({"managed_by_reproduce": False}), encoding="utf-8"
    )
    removed = reproduce._cleanup_managed_model_caches(cfg)
    assert managed in removed and not managed.exists()
    assert user.exists()


def test_list_models_exposes_scientific_variant_plans(capsys):
    args = reproduce.build_parser().parse_args(["list", "rn.stash_followup", "--models"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "xattn_full_trained" in out
    assert "oai_vanilla_rn_from_xattn" in out


def test_oai_rn_variant_contains_exact_donor_rn_and_no_bridge(tmp_path: Path, monkeypatch):
    import sys
    import types
    import torch

    cfg = {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"], "xattn_checkpoint": None},
        "datasets": {**DEFAULT_CONFIG["datasets"]},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }

    donor = tmp_path / "donor.pt"
    donor_state = _tiny_vanilla_clip_state()
    donor_state["visual.read_null_token"] = torch.tensor([1.0, 2.0, 3.0])
    donor_state["visual.read_null_insert_block_config"] = torch.tensor(13)
    torch.save(donor_state, donor)
    monkeypatch.setattr(reproduce, "_ensure_xattn_checkpoint", lambda _cfg: donor)

    class FakeVanilla:
        def state_dict(self):
            return dict(_tiny_vanilla_clip_state())

    fake_sae = types.SimpleNamespace(load=lambda *a, **k: (FakeVanilla(), object()))
    monkeypatch.setitem(sys.modules, "attnclip_mechinterp_sae", fake_sae)

    path = reproduce._ensure_oai_rn_variant(cfg)
    state = reproduce._checkpoint_state(path)
    family, custom = reproduce._checkpoint_family_from_keys(state)
    assert family == "rn_only"
    assert set(custom) == {"visual.read_null_token", "visual.read_null_insert_block_config"}
    assert torch.equal(state["visual.read_null_token"], donor_state["visual.read_null_token"])
    assert not any(str(k).startswith("read_implant.") for k in state)
    meta = __import__("json").loads(reproduce._checkpoint_cache_meta_path(path).read_text(encoding="utf-8"))
    assert meta["variant_id"] == "oai_vanilla_rn_from_xattn"
    assert meta["lineage"]["bridge_policy"] == "absent"
    assert meta["lineage"]["donor_rn_sha256"] == reproduce._tensor_sha256(donor_state["visual.read_null_token"])


def test_internal_rn_helpers_are_namespaced_against_stale_repo_root_copy():
    # Old releases briefly left a duplicate repo-root rn_control_mechinterp package.
    # Reproduction scripts must resolve the bundled internal helper explicitly so
    # extracting a newer zip over an older tree cannot silently select stale code.
    offenders = []
    for path in reproduce.PROBE_ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from rn_control_mechinterp." in text or "import rn_control_mechinterp." in text:
            offenders.append(path.name)
    assert offenders == []
    assert (reproduce.PROBE_ROOT / "__init__.py").is_file()


def test_register_geometry_uses_xattn_as_inert_weight_source_only():
    text = (reproduce.PROBE_ROOT / "probe_register_geometry.py").read_text(encoding="utf-8")
    assert "def load_xattn_weight_source" in text
    assert "safe_torch_load(path)" in text
    assert "extract_state_dict(obj" in text
    assert "loaded_x = load_native_xattn" not in text
    assert "load_native_xattn(" not in text
    assert "read_implant.read_bridge.q_proj.weight" in text
    assert "visual.read_null_token" in text


def test_register_geometry_manual_attention_supports_split_mechinterp_qkv():
    text = (reproduce.PROBE_ROOT / "probe_register_geometry.py").read_text(encoding="utf-8")
    assert 'q_proj = getattr(attn, "q_proj", None)' in text
    assert 'k_proj = getattr(attn, "k_proj", None)' in text
    assert 'v_proj = getattr(attn, "v_proj", None)' in text
    assert 'q = q_proj(x_tbd)' in text
    assert 'Expected packed in_proj_weight on visual MultiheadAttention' not in text


def test_register_geometry_grad_attention_namespace_exposes_weight_source_loader():
    text = (reproduce.PROBE_ROOT / "probe_register_geometry.py").read_text(encoding="utf-8")
    assert "load_xattn_weight_source=load_xattn_weight_source" in text
    assert "xstate = analysis_mod.load_xattn_weight_source(args.xattn)" in text


def test_reproduction_bundle_schema_handshake_is_present():
    from reproduction_utils.version import REPRODUCTION_SCHEMA
    helper = (reproduce.PROBE_ROOT / "probe_tools_repo.py").read_text(encoding="utf-8")
    assert REPRODUCTION_SCHEMA == 30
    assert "PROBE_REPRODUCTION_SCHEMA = 30" in helper
    assert "Mixed paper-reproduction bundle detected" in helper
    assert "x_paper_reproduction/ from the same release" in helper


def test_reproduction_bundle_schema_mismatch_fails_fast(monkeypatch):
    import reproduction_utils.version as version
    from x_paper_reproduction import probe_tools_repo
    monkeypatch.setattr(version, "REPRODUCTION_SCHEMA", 14)
    try:
        probe_tools_repo._assert_bundle_schema(reproduce.PROJECT_ROOT)
    except RuntimeError as exc:
        assert "Mixed paper-reproduction bundle detected" in str(exc)
    else:
        raise AssertionError("mixed reproduction schema was not rejected")


def _minimal_cfg(tmp_path: Path):
    return {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"]},
        "datasets": {**DEFAULT_CONFIG["datasets"]},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }


def test_preflight_catches_missing_rn_vocab_before_compute(tmp_path: Path, monkeypatch):
    cfg = _minimal_cfg(tmp_path)
    monkeypatch.setattr(reproduce, "PROBE_ROOT", tmp_path / "probes")
    reproduce.PROBE_ROOT.mkdir(parents=True)
    task = TASK_BY_ID["rn.control_manifold"]
    ready, problems = reproduce._task_ready(task, cfg)
    assert not ready
    assert any("vocab_deduped.txt" in p for p in problems)


def test_smoke_outputs_are_isolated_from_scientific_outputs(tmp_path: Path):
    cfg = _minimal_cfg(tmp_path)
    task = TASK_BY_ID["workspace.register_geometry.grad_attention"]
    scientific = reproduce._task_output(task, cfg)
    cfg["_smoke"] = True
    smoke = reproduce._task_output(task, cfg)
    assert scientific == tmp_path / "out" / "workspace" / "register_geometry_grad_attention"
    assert smoke == tmp_path / "out" / "_smoke" / "workspace" / "register_geometry_grad_attention"


def test_smoke_manifold_exercises_vocab_path_instead_of_skipping_it():
    task = TASK_BY_ID["rn.control_manifold"]
    args = reproduce._smoke_args(task)
    assert "--vocab-limit" in args
    assert "--skip-vocab" not in args
    assert args[args.index("--vocab-limit") + 1] == "32"


def test_all_smoke_override_options_are_declared_by_target_scripts():
    import re
    offenders = []
    for task in TASKS:
        smoke_args = reproduce._smoke_args(task)
        if not smoke_args:
            continue
        script = reproduce.PROBE_ROOT / task.script
        text = script.read_text(encoding="utf-8")
        # Public wrappers may delegate parser construction to a nested implementation.
        if task.id == "conv1.gpic_manifold":
            text += (reproduce.PROBE_ROOT / "conv1_manifold_gpic" / "conv1_gpic_manifold.py").read_text(encoding="utf-8")
        declared = set(re.findall(r"add_argument\(\s*['\"](--[^'\"]+)", text))
        declared |= {"--no-" + flag[2:] for flag in list(declared) if flag.startswith("--")}
        missing = sorted({arg for arg in smoke_args if arg.startswith("--") and arg not in declared})
        if missing:
            offenders.append((task.id, missing))
    assert offenders == []


def test_single_image_preflight_checks_exact_hardcoded_demo_file(tmp_path: Path):
    cfg = _minimal_cfg(tmp_path)
    demo = tmp_path / "demoset"
    demo.mkdir()
    cfg["datasets"]["demoset_dir"] = str(demo)
    ok, note = reproduce._requirement_status("assets.single_image_demo", cfg)
    assert not ok
    assert "bottle_shower.png" in note
    (demo / "bottle_shower.png").write_bytes(b"x")
    ok, note = reproduce._requirement_status("assets.single_image_demo", cfg)
    assert ok
    assert "bottle_shower.png" in note


def test_preflight_parser_accepts_smoke_mode():
    args = reproduce.build_parser().parse_args(["preflight", "all", "--smoke", "--no-imports"])
    assert args.smoke is True


def test_rn_manifold_default_vocab_resolves_beside_script():
    text = (reproduce.PROBE_ROOT / "probe_tools_rn_manifold.py").read_text(encoding="utf-8")
    assert "beside_script = Path(__file__).resolve().parent / path" in text
    assert "if beside_script.is_file():" in text


def test_rn_manifold_can_resume_after_completed_atlas_without_recomputing_sweep():
    text = (reproduce.PROBE_ROOT / "probe_tools_rn_manifold.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--resume-after-atlas"' in text
    assert '[resume] reusing completed atlas stage' in text
    assert 'atlas_stage_complete.json' in text
    assert 'Saved surface_raw.csv has' in text
    assert 'legacy partial run has no atlas_stage_complete.json' in text


def test_read_null_diagnostic_smoke_does_not_inject_batch_size():
    task = TASK_BY_ID["bridge.read_null.diagnostic"]
    args = reproduce._smoke_args(task)
    assert "--batch-size" not in args


def test_parse_only_cli_validation_matches_script_directory_import_semantics(tmp_path: Path, monkeypatch):
    helper = tmp_path / "sibling_helper.py"
    helper.write_text("VALUE = 7\n", encoding="utf-8")
    script = tmp_path / "uses_sibling.py"
    script.write_text(
        """
import argparse
from sibling_helper import VALUE

p = argparse.ArgumentParser()
p.add_argument('--value', type=int, required=True)
args = p.parse_args()
assert VALUE == args.value
raise RuntimeError('parse-only harness failed to stop execution')
""",
        encoding="utf-8",
    )
    task = Task("dummy.sibling", "dummy", "dummy", script.name)
    monkeypatch.setattr(reproduce, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reproduce, "PROBE_ROOT", tmp_path)
    monkeypatch.setattr(
        reproduce,
        "_command",
        lambda _task, _cfg, _extra: [sys.executable, str(script), "--value", "7"],
    )
    ok, detail = reproduce._parse_only_validate_command(task, {})
    assert ok, detail


def test_parse_only_cli_validation_is_subcommand_specific(tmp_path: Path, monkeypatch):
    script = tmp_path / "multi.py"
    script.write_text(
        """
import argparse, sys

def main():
    command = sys.argv[1]
    sys.argv = [sys.argv[0] + ' ' + command, *sys.argv[2:]]
    p = argparse.ArgumentParser()
    if command == 'a':
        p.add_argument('--foo')
    elif command == 'b':
        p.add_argument('--bar')
    else:
        raise SystemExit(3)
    p.parse_args()
    raise RuntimeError('parse-only harness failed to stop execution')

if __name__ == '__main__':
    main()
""",
        encoding="utf-8",
    )
    task = Task("dummy.b", "dummy", "dummy", script.name, "b")
    monkeypatch.setattr(reproduce, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(reproduce, "PROBE_ROOT", tmp_path)

    monkeypatch.setattr(
        reproduce,
        "_command",
        lambda _task, _cfg, _extra: [sys.executable, str(script), "b", "--foo", "1"],
    )
    ok, detail = reproduce._parse_only_validate_command(task, {})
    assert not ok
    assert "unrecognized arguments" in detail

    monkeypatch.setattr(
        reproduce,
        "_command",
        lambda _task, _cfg, _extra: [sys.executable, str(script), "b", "--bar", "1"],
    )
    ok, detail = reproduce._parse_only_validate_command(task, {})
    assert ok
    assert detail == ""


def _smoke_option_value(args, option):
    i = args.index(option)
    return args[i + 1]


def test_register_geometry_secondary_smoke_meets_runtime_minimum():
    task = reproduce.TASK_BY_ID["workspace.register_geometry.secondary"]
    args = reproduce._smoke_args(task)
    limit = int(_smoke_option_value(args, "--limit"))
    folds = int(_smoke_option_value(args, "--folds"))
    assert limit >= max(folds * 2, 10)


def test_rn_control_surfaces_smoke_meets_empirical_support_minimum():
    task = reproduce.TASK_BY_ID["rn.control_surfaces"]
    args = reproduce._smoke_args(task)
    langs = [x for x in _smoke_option_value(args, "--languages").split(",") if x]
    pairs = int(_smoke_option_value(args, "--basis-pairs-per-language"))
    # fit_reference_basis_and_support_records retains one shared NoRTA record
    # per basis key plus one attacked record per language/key.
    support_records = pairs * (1 + len(langs))
    assert support_records >= 8


def test_rn_manifold_smoke_plane_uses_script_semantics():
    task = reproduce.TASK_BY_ID["rn.control_manifold"]
    args = reproduce._smoke_args(task)
    value = _smoke_option_value(args, "--planes")
    planes = [part.strip() for part in value.split(",") if part.strip()]
    assert planes
    for part in planes:
        bits = part.lower().replace("pc", "").split("x")
        assert len(bits) == 2
        a, b = map(int, bits)
        assert a != b and min(a, b) >= 1


def test_rn_control_surfaces_smoke_plane_uses_manifold_plane_semantics():
    task = reproduce.TASK_BY_ID["rn.control_surfaces"]
    args = reproduce._smoke_args(task)
    value = _smoke_option_value(args, "--planes")
    planes = [part.strip() for part in value.split(",") if part.strip()]
    assert planes
    for part in planes:
        bits = part.lower().replace("pc", "").split("x")
        assert len(bits) == 2
        a, b = map(int, bits)
        assert a != b and min(a, b) >= 1


def test_flow_maps_treats_empty_cross_language_csv_as_valid_no_pairs():
    text = (reproduce.PROBE_ROOT / "probe_rn_control_surfaces.py").read_text(encoding="utf-8")
    assert "def _read_optional_csv(path: Path)" in text
    assert "except pd.errors.EmptyDataError:" in text
    assert "cross_shared_all=_read_optional_csv(first_cross)" in text
    assert "required_cross_cols" in text
    assert "cross_lang_df.empty or not required_cross_cols.issubset" in text


def test_flow_maps_keeps_surface_summary_strict():
    text = (reproduce.PROBE_ROOT / "probe_rn_control_surfaces.py").read_text(encoding="utf-8")
    assert 'surface = pd.read_csv(root / "data" / "surface_summary.csv")' in text


def test_bridge_transplant_pump_modes_is_singleton_tuple():
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "x_paper_reproduction" / "probe_rn_bridge_transplant.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PUMP_MODES" for t in node.targets)
    ]
    assert len(matches) == 1
    value = ast.literal_eval(matches[0])
    assert value == ("intact",)
    assert isinstance(value, tuple)


def test_bridge_transplant_smoke_grid_supports_edge_order_2():
    import reproduce
    task = reproduce.TASK_BY_ID["rn.bridge_transplant"]
    args = reproduce._smoke_args(task)
    idx = args.index("--grid-points")
    assert int(args[idx + 1]) >= 3



def test_touch_go_extracts_rn_from_state_without_instantiating_donor():
    text = (reproduce.PROBE_ROOT / "probe_rn_touch_go_transfer_paper.py").read_text(encoding="utf-8")
    assert "torch_load_trusted(checkpoint)" in text
    assert "_extract_state_dict(loaded)" in text
    assert "validate_final_state_dict(state, explicit_architecture)" in text
    assert 'token_key = "visual.read_null_token"' in text
    assert 'insert_key = "visual.read_null_insert_block_config"' in text
    assert '"donor_runtime_instantiated": False' in text
    assert "load_final_clip_checkpoint(" not in text
    assert "import oaiclip as xclip" not in text



def test_role_text_atlas_selects_single_rn_mode():
    import importlib.util
    import pandas as pd

    path = reproduce.PROJECT_ROOT / "x_paper_reproduction" / "probe_role_text_atlas.py"
    script_dir = str(path.parent)
    sys.path.insert(0, script_dir)
    try:
        spec = importlib.util.spec_from_file_location("_role_text_atlas_test", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(script_dir)

    df = pd.DataFrame({
        "rn_mode": ["rn_off", "rn_on"],
        "condition": ["NoRTA", "NoRTA"],
        "block": [0, 0],
        "head": [0, 0],
    })
    off = mod.select_rn_mode(df, "rn_off", table_name="fits.csv")
    assert len(off) == 1
    assert off.iloc[0]["rn_mode"] == "rn_off"


def test_role_text_atlas_legacy_tables_only_support_rn_off():
    import importlib.util
    import pandas as pd
    import pytest

    path = reproduce.PROJECT_ROOT / "x_paper_reproduction" / "probe_role_text_atlas.py"
    script_dir = str(path.parent)
    sys.path.insert(0, script_dir)
    try:
        spec = importlib.util.spec_from_file_location("_role_text_atlas_legacy_test", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(script_dir)

    legacy = pd.DataFrame({"condition": ["NoRTA"], "block": [0], "head": [0]})
    assert len(mod.select_rn_mode(legacy, "rn_off", table_name="fits.csv")) == 1
    with pytest.raises(ValueError, match="legacy table without rn_mode"):
        mod.select_rn_mode(legacy, "rn_on", table_name="fits.csv")


def test_role_text_atlas_dispatcher_explicitly_selects_rn_off(tmp_path):
    cfg = dict(DEFAULT_CONFIG)
    # Deep-enough minimal copy for output root; task command only needs normal config defaults otherwise.
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["output_root"] = str(tmp_path / "out")
    args = reproduce._auto_args(TASK_BY_ID["figures.role_text_atlas"], cfg)
    assert args[-2:] == ["--rn-mode", "rn_off"]


def test_rn_mechanism_figures_tolerates_empty_optional_csv(tmp_path):
    import importlib.util

    path = reproduce.PROJECT_ROOT / "x_paper_reproduction" / "probe_rn_mechanism_figures.py"
    script_dir = str(path.parent)
    sys.path.insert(0, script_dir)
    try:
        spec = importlib.util.spec_from_file_location("_rn_mech_fig_test", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(script_dir)

    (tmp_path / "cross_language_geometry.csv").write_text("", encoding="utf-8")
    df = mod.read_csv(tmp_path, "cross_language_geometry.csv")
    assert df.empty


def test_conv1_gpic_windows_memmaps_are_explicitly_closed():
    source = (Path(__file__).resolve().parents[1] / "x_paper_reproduction" / "conv1_manifold_gpic" / "conv1_gpic_manifold.py").read_text(encoding="utf-8")
    assert "torch.from_numpy(np.asarray(array)).clone().contiguous()" in source
    assert "def close_trajectory_memmaps" in source
    assert "close_trajectory_memmaps(trajectory_mm)" in source
    cleanup_pos = source.index("close_trajectory_memmaps(trajectory_mm)")
    unlink_pos = source.index("path.unlink(missing_ok=True)", cleanup_pos)
    assert cleanup_pos < unlink_pos
    assert "A previous Windows run may have completed the safetensors export" in source
