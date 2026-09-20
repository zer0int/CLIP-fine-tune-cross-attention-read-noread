from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_obsolete_reader_remap_artifacts_are_absent():
    legacy_names = (
        "configuration_" + "late_read_" + "refit.py",
        "modeling_" + "late_read_" + "refit.py",
        "train_refit_late_read_b19_b22_readnull.py",
    )
    for name in legacy_names:
        assert not (ROOT / name).exists(), name

    # The old topology-rewrite experiment used explicit block-moving helpers.
    # ``read_state_blocks`` is intentionally NOT forbidden: current WiSE/eval
    # code may expose it as topology metadata while keeping native B20/B21 taps.
    forbidden_tokens = (
        "move_" + "B19_B21",
        "move_" + "B20_B22",
    )
    checked = [
        ROOT / "train_refit_late_reader.py",
        ROOT / "train_wiseft_late_reader.py",
        ROOT / "eval_wiseft_late_reader.py",
    ]
    for path in checked:
        assert path.is_file(), path
        source = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source, f"obsolete reader-remap token present in {path}"


def test_native_refit_is_fixed_to_existing_late_reader():
    text = (ROOT / "train_refit_late_reader.py").read_text(encoding="utf-8")
    assert "EXPECTED_NATIVE_LATE = (20, 21)" in text
    assert "for block in EXPECTED_NATIVE_LATE" in text
    assert "topology is immutable" in text
