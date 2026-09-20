import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "x_paper_reproduction"
    / "b20_pushpull_mediation"
    / "probe_B20_PUSHPULL_MEDIATION_650_715.py"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("b20_pushpull_probe_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_combine_tolerates_legacy_headerless_mediation_csv(tmp_path):
    probe = _load_probe()
    model_dir = tmp_path / "pretrained"
    model_dir.mkdir()

    pd.DataFrame(
        [{"model": "pretrained", "intervention": "baseline", "sharp_top1": 0.5}]
    ).to_csv(model_dir / "intervention_summary.csv", index=False)

    # Exact artifact shape left by the old baseline-only smoke path:
    # DataFrame([]).to_csv(index=False) -> a single newline, no header.
    (model_dir / "mediation_ratios.csv").write_text("\n", encoding="utf-8")

    probe.combine(["pretrained"], tmp_path)

    combined = pd.read_csv(tmp_path / "ALL_MODELS_MEDIATION_SUMMARY.csv")
    assert combined["intervention"].tolist() == ["baseline"]
    assert not (tmp_path / "ALL_MODELS_MEDIATION_RATIOS.csv").exists()


def test_empty_mediation_schema_is_readable(tmp_path):
    probe = _load_probe()
    q = tmp_path / "mediation_ratios.csv"
    pd.DataFrame(columns=probe.MEDIATION_COLUMNS).to_csv(q, index=False)

    parsed = pd.read_csv(q)
    assert parsed.empty
    assert parsed.columns.tolist() == probe.MEDIATION_COLUMNS
