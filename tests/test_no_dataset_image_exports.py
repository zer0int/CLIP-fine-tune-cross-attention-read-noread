from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_benchmarks_do_not_copy_or_save_dataset_images() -> None:
    for path in ROOT.glob("eval_benchmark_*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
        assert "copy2" not in calls, path.name
        assert "copyfile" not in calls, path.name
        assert "torch.save" not in source, path.name
        assert "numpy.save" not in source, path.name

