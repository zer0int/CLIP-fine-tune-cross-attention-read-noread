from pathlib import Path

from reproduction_utils.catalog import TASK_BY_ID


ROOT = Path(__file__).resolve().parents[1]


def _text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".zip"}:
            continue
        try:
            yield path, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def test_no_legacy_handoff_branding_remains():
    hits = []
    for path, text in _text_files():
        for forbidden in ("SEND" + "_ME_THIS", "FOR" + "_" + "CHAT" + "GPT"):
            if forbidden in text:
                hits.append(f"{path.relative_to(ROOT)}: {forbidden}")
    assert hits == []


def test_conv1_compact_archives_use_task_named_root_files():
    expected = {
        "conv1.roleplane_texture_rank.compact": "compact_summary_conv1_roleplane_texture_rank_compact.zip",
        "conv1.visualtextual_provenance": "compact_summary_conv1_visualtextual_provenance.zip",
        "conv1.visualtextual_text_direction": "compact_summary_conv1_visualtextual_text_direction.zip",
    }
    catalog = (ROOT / "reproduction_utils" / "catalog.py").read_text(encoding="utf-8")
    for task_id, filename in expected.items():
        assert filename in catalog or task_id != "conv1.roleplane_texture_rank.compact"
        assert "/" not in filename and "\\" not in filename

    assert TASK_BY_ID["conv1.roleplane_texture_rank.compact"].markers == (
        expected["conv1.roleplane_texture_rank.compact"],
    )
    # The two GPU tasks cache on their substantive report so renaming the convenience
    # archive does not invalidate an already completed run.
    assert TASK_BY_ID["conv1.visualtextual_provenance"].markers == ("REPORT.md",)
    assert TASK_BY_ID["conv1.visualtextual_text_direction"].markers == ("REPORT.md",)


def test_visualtextual_compact_archives_are_written_at_task_root():
    provenance = (ROOT / "x_paper_reproduction" / "visualtextual" / "probe_VISUALTEXTUAL_PROVENANCE_ROLE_ATLAS.py").read_text(encoding="utf-8")
    direction = (ROOT / "x_paper_reproduction" / "visualtextual" / "probe_VISUALTEXTUAL_TEXT_DIRECTION_TRAJECTORY.py").read_text(encoding="utf-8")
    assert 'root/"compact_summary_conv1_visualtextual_provenance.zip"' in provenance
    assert 'root/"compact_summary_conv1_visualtextual_text_direction.zip"' in direction
    assert 'root/"compact_summary"' not in provenance
    assert 'root/"compact_summary"' not in direction
