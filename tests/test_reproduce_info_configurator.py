from __future__ import annotations

import json
from pathlib import Path

import pytest
import reproduce
from reproduction_utils.catalog import PAPER_INFO, TASKS


def test_every_task_has_paper_facing_metadata():
    assert set(PAPER_INFO) == {task.id for task in TASKS}
    for task in TASKS:
        info = PAPER_INFO[task.id]
        assert info.experiment_type in {"M", "I", "M/I"}
        assert info.description.strip()


def test_generated_catalog_uses_stable_task_ids_and_excludes_manual_utility_from_checkbox():
    payload = json.loads((reproduce.PROJECT_ROOT / "reproduce_info" / "task_catalog.json").read_text(encoding="utf-8"))
    rows = payload["tasks"]
    assert [row["id"] for row in rows] == [task.id for task in TASKS]
    audit = next(row for row in rows if row["id"] == "audit.read_probe_router")
    assert audit["selectable"] is False
    gpic = next(row for row in rows if row["id"] == "conv1.gpic_manifold")
    assert gpic["thumbnail"] == "AA_GLOBAL_bullet5__ch0720_FLIP__ch0866_FLIP.png"
    assert gpic["intervention_family"] == "gpic"


def test_selection_file_is_task_only_and_preserves_order(tmp_path: Path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"schema": 1, "name": "test", "tasks": [
        "rn.touch_go_transfer", "conv1.gpic_manifold", "rn.touch_go_transfer"
    ]}), encoding="utf-8")
    tasks = reproduce._load_task_selection(path)
    assert [task.id for task in tasks] == ["rn.touch_go_transfer", "conv1.gpic_manifold"]


def test_selection_file_cannot_override_main_reproduction_config(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema": 1,
        "tasks": ["conv1.gpic_manifold"],
        "datasets": {"demoset_dir": "NOPE"},
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="cannot override the main reproduction config"):
        reproduce._load_task_selection(path)


def test_run_parser_accepts_selection_without_positional_selector(tmp_path: Path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"schema": 1, "tasks": ["conv1.gpic_manifold"]}), encoding="utf-8")
    args = reproduce.build_parser().parse_args(["run", "--selection", str(path), "--smoke"])
    chosen = reproduce._selected_tasks_from_args(args, default_selectors=[])
    assert [task.id for task in chosen] == ["conv1.gpic_manifold"]
    assert args.smoke


def test_selector_and_selection_are_mutually_exclusive(tmp_path: Path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"schema": 1, "tasks": ["conv1.gpic_manifold"]}), encoding="utf-8")
    args = reproduce.build_parser().parse_args(["run", "rn.touch_go_transfer", "--selection", str(path)])
    with pytest.raises(SystemExit, match="either positional selectors or --selection"):
        reproduce._selected_tasks_from_args(args, default_selectors=[])



def test_configurator_has_steering_favorites_filter_and_ply_tags():
    html = (reproduce.PROJECT_ROOT / "reproduce_info" / "configurator.html").read_text(encoding="utf-8")
    assert "Show steering favorites only" in html
    assert ".ply {" in html

    payload = json.loads((reproduce.PROJECT_ROOT / "reproduce_info" / "task_catalog.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in payload["tasks"]}
    expected_ply = {
        "workspace.cls_role_surfaces",
        "rn.control_manifold",
        "rn.control_surfaces",
        "rn.control_surface_flow_maps",
        "rn.bridge_transplant",
    }
    actual_ply = {task_id for task_id, row in rows.items() if "PLY" in row.get("special_tags", [])}
    assert actual_ply == expected_ply


def test_configurator_uses_static_thumbnail_assets_only():
    info_dir = reproduce.PROJECT_ROOT / "reproduce_info"
    assert not (info_dir / "make_figure_thumbnails.py").exists()
    assert "make_figure_thumbnails.py" not in (info_dir / "README.md").read_text(encoding="utf-8")
    assert "make_figure_thumbnails.py" not in (info_dir / "generate_reproduce_info.py").read_text(encoding="utf-8")
    assert (info_dir / "figures_thumbs").is_dir()
