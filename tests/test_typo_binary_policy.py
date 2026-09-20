from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_typo_is_binary_only_raw_labels():
    text = (ROOT / "benchmarks" / "typo.py").read_text(encoding="utf-8")
    assert "benchmark_label_canonicalization" not in text
    assert "canonicalize_benchmark_label" not in text
    assert "multi_accuracy" not in text
    assert "all_label" not in text.lower()
    assert "metric_policy" in text
    assert "binary only; raw benchmark labels; no canonicalization" in text


def test_objectnet_uses_raw_labels():
    text = (ROOT / "benchmarks" / "objectnet_mvt.py").read_text(encoding="utf-8")
    assert "benchmark_label_canonicalization" not in text
    assert "canonicalize_benchmark_label" not in text


def test_obsolete_canonicalizer_is_absent():
    assert not (ROOT / "benchmark_label_canonicalization.py").exists()
    assert not (ROOT / "benchmarks" / "benchmark_label_canonicalization.py").exists()


def test_all_public_typo_helpers_are_raw_label_only():
    paths = [
        ROOT / "eval_wiseft_late_reader.py",
        ROOT / "train_wiseft_late_reader.py",
    ]
    forbidden = (
        "canonicalize_labels",
        "canonicalize_benchmark_label",
        "SEMANTIC_GROUPS",
        "PLURAL_NORMALIZATION",
        "multi_accuracy",
        "all_label_accuracy",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path}: stale token {token}"


def test_post_training_helpers_do_not_live_in_paper_reproduction():
    paper = ROOT / "x_paper_reproduction"
    assert not (paper / "wiseft_late_reader.py").exists()
    assert not (paper / "eval_full_xattn_NORMAL_TYPO_EVAL_LATE_READ_REFITS.py").exists()

    train_text = (ROOT / "train_wiseft_late_reader.py").read_text(encoding="utf-8")
    assert 'DEFAULT_EVAL_HELPER = Path("eval_wiseft_late_reader.py")' in train_text
    assert "x_paper_reproduction/" not in train_text
