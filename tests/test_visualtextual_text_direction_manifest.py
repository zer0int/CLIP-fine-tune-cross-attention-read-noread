import importlib.util
import sys
from pathlib import Path

import pandas as pd


VIS_DIR = (
    Path(__file__).resolve().parents[1]
    / "x_paper_reproduction"
    / "visualtextual"
)
SCRIPT = VIS_DIR / "probe_VISUALTEXTUAL_TEXT_DIRECTION_TRAJECTORY.py"


def _load_probe():
    sys.path.insert(0, str(VIS_DIR))
    spec = importlib.util.spec_from_file_location("visualtextual_text_direction_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unpaired_mix_adv_is_excluded_from_text_direction_manifest():
    probe = _load_probe()
    rows = [
        {"stim_id": "vis_avocado", "concept": "avocado", "condition": "vis", "bw": False},
        {"stim_id": "txt_avocado", "concept": "avocado", "condition": "txt", "bw": False},
        {"stim_id": "mix_avocado", "concept": "avocado", "condition": "mix", "bw": False},
        {"stim_id": "mix_adv_avocado", "concept": "adv_avocado", "condition": "mix", "bw": False},
    ]

    paired, excluded = probe.complete_pair_manifest(pd.DataFrame(rows))

    assert paired["stim_id"].tolist() == ["vis_avocado", "txt_avocado", "mix_avocado"]
    assert excluded["stim_id"].tolist() == ["mix_adv_avocado"]
