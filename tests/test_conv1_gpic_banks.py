from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

import reproduce
from reproduction_utils.catalog import TASK_BY_ID
from reproduction_utils.config import DEFAULT_CONFIG
from x_paper_reproduction.conv1_embedding_banks import bank_io


MODEL = "acme/clip-test"


@pytest.fixture(autouse=True)
def _parquet_reader_without_optional_pyarrow(monkeypatch):
    monkeypatch.setattr(bank_io.pd, "read_parquet", lambda path, *a, **k: pd.read_pickle(path))



def _write_bank(
    root: Path,
    *,
    model: str = MODEL,
    vectors: dict[str, torch.Tensor] | None = None,
    format_version: int = 2,
    source_kind: str = "local_manifest",
    extra_declared_space_without_file: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    vectors = vectors or {"backbone": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16)}
    manifest = pd.DataFrame(
        {
            "bank_row": list(range(next(iter(vectors.values())).shape[0])),
            "local_path": [str(root / f"image_{i}.jpg") for i in range(next(iter(vectors.values())).shape[0])],
        }
    )
    # Store a pickle behind the .parquet name in this dependency-light test
    # environment; the fixture below redirects the bank reader accordingly.
    manifest.to_pickle(root / "manifest.parquet")
    spaces = {}
    for name, tensor in vectors.items():
        filename = f"{name}.safetensors"
        save_file({"embeddings": tensor.contiguous()}, root / filename)
        spaces[name] = filename
    if extra_declared_space_without_file:
        spaces["future_space"] = "future_space.safetensors"

    if format_version == 1:
        config = {
            "format_version": 1,
            "model_source": model,
            "rows": len(manifest),
            "dim": int(next(iter(vectors.values())).shape[1]),
            "embeddings": {
                "normalized": True,
                "stored_dtype": "float16",
                **{name: filename for name, filename in spaces.items()},
                "tensor_key": "embeddings",
            },
            "gpic_source_metadata": {"gpic_repo_id": "stanford-vision-lab/gpic", "gpic_revision": "deadbeef"},
        }
    else:
        config = {
            "format_version": 2,
            "model_source": model,
            "rows": len(manifest),
            "dim": int(next(iter(vectors.values())).shape[1]),
            "source": {"kind": source_kind},
            "embeddings": {
                "normalized": True,
                "stored_dtype": "float16",
                "tensor_key": "embeddings",
                "spaces": spaces,
            },
            "model_identity": {
                "source": model,
                "canonical_state_sha256": "abc123",
                "model_family": "vanilla",
            },
        }
    (root / "bank_config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def _cfg(tmp_path: Path) -> dict:
    return {
        **DEFAULT_CONFIG,
        "output_root": str(tmp_path / "out"),
        "models": {**DEFAULT_CONFIG["models"]},
        "datasets": {**DEFAULT_CONFIG["datasets"]},
        "conv1": {**DEFAULT_CONFIG["conv1"]},
        "runtime": {**DEFAULT_CONFIG["runtime"]},
        "task_args": {},
    }


def test_model_repo_id_maps_to_stable_bank_subdir():
    assert bank_io.model_repo_to_subdir("openai/clip-vit-large-patch14") == "openai__clip-vit-large-patch14"
    assert (
        bank_io.model_repo_to_subdir("zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX")
        == "zer0int__CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
    )


def test_v1_and_v2_bank_configs_are_both_accepted(tmp_path: Path):
    v1 = _write_bank(tmp_path / "v1", format_version=1)
    bank1 = bank_io.validate_bank_dir(v1, expected_model_source=MODEL, required_spaces=("backbone",))
    assert bank1.model_source == MODEL
    assert "backbone" in bank1.spaces

    v2 = _write_bank(tmp_path / "v2", format_version=2)
    bank2 = bank_io.validate_bank_dir(v2, expected_model_source=MODEL, required_spaces=("backbone",))
    assert bank2.model_source == MODEL
    assert bank2.source_kind == "local_manifest"


def test_validation_only_requires_requested_spaces(tmp_path: Path):
    root = _write_bank(tmp_path / "bank", extra_declared_space_without_file=True)
    bank = bank_io.validate_bank_dir(root, expected_model_source=MODEL, required_spaces=("backbone",))
    assert "future_space" in bank.spaces




def test_descriptor_resolves_from_bank_config_without_manifest(tmp_path: Path):
    root = tmp_path / "descriptor_only"
    root.mkdir()
    config = {
        "format_version": 2,
        "model_source": MODEL,
        "source": {
            "kind": "gpic",
            "gpic_source_metadata": {
                "gpic_repo_id": "stanford-vision-lab/gpic",
                "gpic_revision": "deadbeef",
                "gpic_tar_path_template": "test/gpic_test_{shard:05d}.tar",
            },
        },
        "embeddings": {
            "normalized": True,
            "spaces": {"backbone": "backbone.safetensors"},
            "tensor_key": "embeddings",
        },
    }
    (root / "bank_config.json").write_text(json.dumps(config), encoding="utf-8")
    descriptor = bank_io.resolve_bank_descriptor(
        root, model_source=MODEL, required_spaces=None, is_primary=True
    )
    assert descriptor is not None
    assert descriptor.source_kind == "gpic"
    assert descriptor.model_source == MODEL
    assert descriptor.spaces == {"backbone": "backbone.safetensors"}


def test_gpic_descriptor_access_probe_does_not_need_manifest(tmp_path: Path, monkeypatch):
    root = tmp_path / "descriptor_probe"
    root.mkdir()
    config = {
        "format_version": 2,
        "model_source": MODEL,
        "source": {
            "kind": "gpic",
            "gpic_source_metadata": {
                "gpic_repo_id": "stanford-vision-lab/gpic",
                "gpic_revision": "deadbeef",
                "gpic_tar_path_template": "test/gpic_test_{shard:05d}.tar",
            },
        },
        "embeddings": {"spaces": {"backbone": "backbone.safetensors"}},
    }
    (root / "bank_config.json").write_text(json.dumps(config), encoding="utf-8")
    descriptor = bank_io.resolve_bank_descriptor(root, model_source=MODEL, is_primary=True)
    assert descriptor is not None

    import huggingface_hub

    seen = {}
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "fake-token")

    class FakeFS:
        def __init__(self, token=None):
            seen["token"] = token
        def info(self, path):
            seen["path"] = path
            return {"name": path}

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FakeFS)
    ok, detail = bank_io.gpic_access_probe(descriptor)
    assert ok
    assert detail == "stanford-vision-lab/gpic@deadbeef"
    assert seen["token"] == "fake-token"
    assert seen["path"].endswith("/test/gpic_test_00000.tar")


def test_present_optional_bank_with_missing_required_space_is_rejected(tmp_path: Path):
    root = _write_bank(tmp_path / "present_bank")
    with pytest.raises(bank_io.BankUnavailable):
        bank_io.resolve_bank_source(
            root,
            model_source=MODEL,
            required_spaces=("backbone", "content"),
            optional=True,
        )


def test_missing_optional_custom_bank_is_skipped(tmp_path: Path):
    result = bank_io.resolve_bank_source(
        tmp_path / "does_not_exist",
        model_source=MODEL,
        required_spaces=("backbone",),
        optional=True,
    )
    assert result is None


def test_collection_search_merges_topk_across_compatible_banks(tmp_path: Path):
    a_root = _write_bank(
        tmp_path / "a",
        vectors={"backbone": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16)},
    )
    b_root = _write_bank(
        tmp_path / "b",
        vectors={"backbone": torch.tensor([[0.8, 0.6], [-1.0, 0.0]], dtype=torch.float16)},
    )
    a = bank_io.validate_bank_dir(a_root, expected_model_source=MODEL, name="a", required_spaces=("backbone",))
    b = bank_io.validate_bank_dir(b_root, expected_model_source=MODEL, name="b", required_spaces=("backbone",))
    rows = bank_io.search_collection(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        [a, b],
        space="backbone",
        topk=2,
        chunk_size=1,
        device="cpu",
    )[0]
    assert [row["bank_name"] for row in rows] == ["a", "b"]
    assert [row["rank"] for row in rows] == [1, 2]
    assert rows[0]["cosine"] > rows[1]["cosine"]


def test_gate_warning_contains_accept_and_login_instructions(capsys):
    bank_io.print_gpic_access_warning("no token")
    text = capsys.readouterr().out
    assert "SKIPPED (N/A)" in text
    assert "https://huggingface.co/datasets/stanford-vision-lab/gpic" in text
    assert "hf auth login" in text


def test_conv1_gpic_task_begins_appended_conv1_branch_after_original_33():
    automatic = [t for t in reproduce.TASKS if not t.manual_args and t.tier != "utility"]
    assert automatic[32].id == "figures.rn_mechanism"
    assert automatic[33].id == "conv1.gpic_manifold"
    assert automatic[34].id == "conv1.xattn_functional_atlas"
    selected = reproduce._resolve_selectors(["all"], include_controls=False)
    ids = [task.id for task in selected]
    assert ids.index("conv1.gpic_manifold") == 33
    assert ids.index("conv1.gpic_manifold") < ids.index("conv1.xattn_functional_atlas")


def test_conv1_gpic_auto_args_default_and_model_override(tmp_path: Path):
    cfg = _cfg(tmp_path)
    task = TASK_BY_ID["conv1.gpic_manifold"]
    args = reproduce._auto_args(task, cfg)
    model_positions = [i for i, value in enumerate(args) if value == "--model"]
    assert [args[i + 1] for i in model_positions] == [
        DEFAULT_CONFIG["models"]["final_hf"],
        DEFAULT_CONFIG["models"]["vanilla_hf"],
    ]
    assert args[args.index("--reference_bank") + 1] == "zer0int/CLIP-GPIC-embeddings"

    cfg["conv1"]["gpic_model"] = "openai/clip-vit-large-patch14"
    cfg["conv1"]["custom_embedding_banks"] = ["private/laion_bank", "another/private-bank"]
    args = reproduce._auto_args(task, cfg)
    assert args[args.index("--model") + 1] == "openai/clip-vit-large-patch14"
    positions = [i for i, value in enumerate(args) if value == "--custom_bank"]
    assert [args[i + 1] for i in positions] == ["private/laion_bank", "another/private-bank"]


def test_conv1_gpic_smoke_is_tiny_but_runs_real_retrieval_path():
    args = reproduce._smoke_args(TASK_BY_ID["conv1.gpic_manifold"])
    assert args[args.index("--limit_source_images") + 1] == "1"
    assert "--smoke_grid" in args
    assert args[args.index("--gpic_topk") + 1] == "2"
    assert "--skip_retrieval" not in args


def test_conv1_gpic_multimodel_wrapper_parse_only_preflight_stops_before_runtime(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg["_smoke"] = True
    task = TASK_BY_ID["conv1.gpic_manifold"]
    ok, detail = reproduce._parse_only_validate_command(task, cfg)
    assert ok, detail


def test_gpic_na_marker_does_not_make_task_complete(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg["_smoke"] = True
    task = TASK_BY_ID["conv1.gpic_manifold"]
    out = reproduce._task_output(task, cfg)
    assert out is not None
    out.mkdir(parents=True)
    (out / "SKIPPED_GPIC_NA.json").write_text("{}", encoding="utf-8")
    assert not reproduce._task_complete(task, cfg)
    models = reproduce._conv1_gpic_models(cfg)
    for model in models:
        child = out / reproduce._conv1_gpic_output_name(model)
        child.mkdir(parents=True, exist_ok=True)
        (child / "summary.json").write_text("{}", encoding="utf-8")
    (out / "batch_summary.json").write_text(json.dumps({"models": models}), encoding="utf-8")
    assert reproduce._task_complete(task, cfg)
    cfg["conv1"]["gpic_models"] = [*models, "acme/extra-model"]
    assert not reproduce._task_complete(task, cfg)




def test_gpic_na_exits_before_runtime_model_load(tmp_path: Path, monkeypatch):
    from PIL import Image
    from x_paper_reproduction.conv1_manifold_gpic import conv1_gpic_manifold as impl

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (16, 16), "white").save(image_dir / "x.png")
    experiment = tmp_path / "experiment.json"
    experiment.write_text(
        json.dumps(
            {
                "alphas": [0.0, 1.0],
                "singles": [{"name": "ch0", "channel": 0, "condition": "FLIP"}],
                "pairs": [],
            }
        ),
        encoding="utf-8",
    )
    descriptor = bank_io.BankDescriptor(
        name="gpic",
        root=tmp_path,
        config={
            "model_source": MODEL,
            "gpic_source_metadata": {
                "gpic_repo_id": "stanford-vision-lab/gpic",
                "gpic_revision": "deadbeef",
            },
            "embeddings": {"backbone": "backbone.safetensors"},
        },
        spaces={"backbone": "backbone.safetensors"},
        model_source=MODEL,
        source_kind="gpic",
        origin="fake",
        is_primary=True,
    )
    monkeypatch.setattr(impl, "resolve_bank_descriptor", lambda *a, **k: descriptor)
    monkeypatch.setattr(impl, "gpic_access_probe", lambda bank: (False, "no token"))

    def forbidden_model_load(*a, **k):
        raise AssertionError("runtime model must not be loaded before GPIC access succeeds")

    monkeypatch.setattr(impl, "load_reproduction_model", forbidden_model_load)
    out = tmp_path / "out"
    rc = impl.main(
        [
            "--model", MODEL,
            "--image_dir", str(image_dir),
            "--config", str(experiment),
            "--output_dir", str(out),
            "--vocab_strategy", "off",
        ]
    )
    assert rc == 0
    marker = json.loads((out / "SKIPPED_GPIC_NA.json").read_text(encoding="utf-8"))
    assert marker["reason"] == "gpic_access_unavailable"


def test_public_conv1_wrapper_and_nested_parser_import_cleanly():
    from x_paper_reproduction.conv1_manifold_gpic import conv1_gpic_manifold as impl

    parser = impl.build_parser()
    args = parser.parse_args([])
    assert args.reference_bank == "zer0int/CLIP-GPIC-embeddings"
    assert args.model == "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"


def test_paper_default_experiment_config_has_required_720_and_779_pair_surfaces():
    config_dir = Path(__file__).resolve().parents[1] / "x_paper_reproduction" / "conv1_manifold_gpic"
    expected_pairs = [
        (720, "FLIP", 499, "FLIP"),
        (720, "FLIP", 866, "FLIP"),
        (720, "SHUFFLE", 866, "FLIP"),
        (779, "FLIP", 499, "FLIP"),
        (779, "FLIP", 866, "FLIP"),
        (779, "SHUFFLE", 866, "FLIP"),
    ]
    for filename in ("experiment_config.json", "experiment_config.example.json"):
        raw = json.loads((config_dir / filename).read_text(encoding="utf-8"))
        pairs = [
            (
                int(row["first"]["channel"]),
                str(row["first"]["condition"]).upper(),
                int(row["second"]["channel"]),
                str(row["second"]["condition"]).upper(),
            )
            for row in raw["pairs"]
        ]
        assert pairs == expected_pairs
        singles = {(int(row["channel"]), str(row["condition"]).upper()) for row in raw["singles"]}
        assert {(720, "FLIP"), (720, "SHUFFLE"), (779, "FLIP"), (779, "SHUFFLE"), (499, "FLIP"), (866, "FLIP")} <= singles


def test_custom_bank_root_auto_selects_model_subfolder_and_skips_missing_sibling(tmp_path: Path):
    root = tmp_path / "laion_conv1_manifold_alpha_flight"
    xattn = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
    oai = "openai/clip-vit-large-patch14"

    xattn_dir = root / bank_io.model_repo_to_subdir(xattn)
    _write_bank(
        xattn_dir,
        model=xattn,
        vectors={
            "backbone": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16),
            "content": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16),
        },
    )

    selected = bank_io.resolve_bank_source(
        root,
        model_source=xattn,
        required_spaces=("backbone", "content"),
        optional=True,
    )
    assert selected is not None
    assert selected.root == xattn_dir.resolve()
    assert selected.model_source == xattn

    # Same optional root, but no OpenAI subfolder: this model simply runs
    # without the extra bank instead of failing the multi-model GPIC task.
    missing = bank_io.resolve_bank_source(
        root,
        model_source=oai,
        required_spaces=("backbone",),
        optional=True,
    )
    assert missing is None


def test_custom_bank_root_dispatches_two_model_specific_subfolders(tmp_path: Path):
    root = tmp_path / "laion_conv1_manifold_alpha_flight"
    xattn = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
    oai = "openai/clip-vit-large-patch14"

    _write_bank(
        root / bank_io.model_repo_to_subdir(xattn),
        model=xattn,
        vectors={
            "backbone": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16),
            "content": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16),
        },
    )
    _write_bank(
        root / bank_io.model_repo_to_subdir(oai),
        model=oai,
        vectors={"backbone": torch.tensor([[0.8, 0.6], [0.0, 1.0]], dtype=torch.float16)},
    )

    xattn_bank = bank_io.resolve_bank_source(
        root,
        model_source=xattn,
        required_spaces=("backbone", "content"),
        optional=True,
    )
    oai_bank = bank_io.resolve_bank_source(
        root,
        model_source=oai,
        required_spaces=("backbone",),
        optional=True,
    )
    assert xattn_bank is not None and xattn_bank.root.name == bank_io.model_repo_to_subdir(xattn)
    assert oai_bank is not None and oai_bank.root.name == bank_io.model_repo_to_subdir(oai)
