from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str, default=np.nan) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _int(row: dict, key: str, default=0) -> int:
    value = row.get(key, "")
    if value in (None, ""):
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt = _mpl()
    plt.close(fig)


def plot_svd_energy(run_dir: Path, plot_dir: Path) -> list[Path]:
    rows = _read_csv(run_dir / "svd_summary.csv")
    if not rows:
        return []
    plt = _mpl()
    made = []
    views = sorted({r.get("view", "") for r in rows if r.get("view")})
    variants = sorted({r.get("variant", "") for r in rows if r.get("variant")})
    for view in views:
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for variant in variants:
            sub = sorted(
                [r for r in rows if r.get("view") == view and r.get("variant") == variant],
                key=lambda r: _int(r, "block"),
            )
            if not sub:
                continue
            ax.plot([_int(r, "block") for r in sub], [_float(r, "energy_pc1") for r in sub], marker="o", label=variant)
        ax.set_title(f"RN delta: first singular-component energy ({view})")
        ax.set_xlabel("Block")
        ax.set_ylabel("Fraction of uncentered delta energy in component 1")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, ncol=2)
        path = plot_dir / f"01_svd_pc1_energy__{view}.png"
        _finish(fig, path); made.append(path)
    return made


def plot_universal_fraction(run_dir: Path, plot_dir: Path) -> list[Path]:
    rows = _read_csv(run_dir / "svd_summary.csv")
    if not rows:
        return []
    plt = _mpl(); made = []
    for view in ("cls", "region_mean", "outside_mean", "patch_mean"):
        suball = [r for r in rows if r.get("view") == view]
        if not suball:
            continue
        variants = sorted({r.get("variant", "") for r in suball if r.get("variant")})
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for variant in variants:
            sub = sorted([r for r in suball if r.get("variant") == variant], key=lambda r: _int(r, "block"))
            ax.plot([_int(r, "block") for r in sub], [_float(r, "universal_mean_template_fraction") for r in sub], marker="o", label=variant)
        ax.set_title(f"Image-independent RN delta fraction ({view})")
        ax.set_xlabel("Block")
        ax.set_ylabel("Universal mean-template fraction")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, ncol=2)
        path = plot_dir / f"02_universal_template_fraction__{view}.png"
        _finish(fig, path); made.append(path)
    return made


def plot_subspace_alignment(run_dir: Path, plot_dir: Path) -> list[Path]:
    rows = _read_csv(run_dir / "subspace_alignment.csv")
    if not rows:
        return []
    plt = _mpl(); made = []
    for view in ("cls", "region_mean", "outside_mean", "patch_mean"):
        suball = [r for r in rows if r.get("view") == view]
        if not suball:
            continue
        comparisons = sorted({r.get("comparison", "") for r in suball if r.get("comparison")})
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        for comp in comparisons:
            sub = sorted([r for r in suball if r.get("comparison") == comp], key=lambda r: _int(r, "block"))
            ax.plot([_int(r, "block") for r in sub], [_float(r, "mean_subspace_cos") for r in sub], marker="o", label=comp)
        ax.set_title(f"RN steering subspace alignment ({view})")
        ax.set_xlabel("Block")
        ax.set_ylabel("Mean cosine of principal subspace angles")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        path = plot_dir / f"03_subspace_alignment__{view}.png"
        _finish(fig, path); made.append(path)

        input_rows = [r for r in suball if "input(text-" in r.get("comparison", "") and r.get("mean_direction_cos", "") not in (None, "")]
        if input_rows:
            comps = sorted({r["comparison"] for r in input_rows})
            fig, ax = plt.subplots(figsize=(9.5, 5.3))
            for comp in comps:
                sub = sorted([r for r in input_rows if r.get("comparison") == comp], key=lambda r: _int(r, "block"))
                ax.plot([_int(r, "block") for r in sub], [_float(r, "mean_direction_cos") for r in sub], marker="o", label=comp)
            ax.axhline(0.0, linewidth=1.0)
            ax.set_title(f"RN mean direction vs readable-text contrast ({view})")
            ax.set_xlabel("Block")
            ax.set_ylabel("Cosine")
            ax.set_ylim(-1.02, 1.02)
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
            path = plot_dir / f"04_mean_direction_vs_text_contrast__{view}.png"
            _finish(fig, path); made.append(path)
    return made


def plot_embedding_conditions(run_dir: Path, plot_dir: Path) -> list[Path]:
    rows = _read_csv(run_dir / "embedding_condition_summary_per_scene.csv")
    if not rows:
        return []
    plt = _mpl(); made = []
    variants = sorted({r.get("variant", "") for r in rows if r.get("variant")})
    metrics = [
        ("cos_tag_vs_full", "Tag-only RN vs persistent RN cosine", "Cosine"),
        ("cos_base_vs_tag", "Baseline vs RN-tag embedding cosine", "Cosine"),
        ("nop_over_tag_shift", "Pure-NOP shift / full RN-tag shift", "Ratio"),
    ]
    for metric, title, ylabel in metrics:
        means, stds = [], []
        for variant in variants:
            xs = np.array([_float(r, metric) for r in rows if r.get("variant") == variant], dtype=np.float64)
            xs = xs[np.isfinite(xs)]
            means.append(float(xs.mean()) if xs.size else np.nan)
            stds.append(float(xs.std()) if xs.size else 0.0)
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        x = np.arange(len(variants))
        ax.bar(x, means, yerr=stds, capsize=3)
        ax.set_xticks(x, variants, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if metric.startswith("cos_"):
            ax.set_ylim(0, 1.01)
        ax.grid(True, axis="y", alpha=0.2)
        path = plot_dir / f"05_embedding__{metric}.png"
        _finish(fig, path); made.append(path)
    return made


def plot_b13_head_decomposition(run_dir: Path, plot_dir: Path) -> list[Path]:
    rows = _read_csv(run_dir / "b13_head_kv_decomposition.csv")
    if not rows:
        return []
    plt = _mpl(); made = []
    for group in ("cls", "patch"):
        sub = sorted([r for r in rows if r.get("token_group") == group], key=lambda r: _int(r, "head"))
        if not sub:
            continue
        heads = [_int(r, "head") for r in sub]
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.bar(heads, [_float(r, "rn_attention_mean") for r in sub])
        ax.set_title(f"B13 RN attention by head ({group})")
        ax.set_xlabel("Head")
        ax.set_ylabel("Mean attention to RN")
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.2)
        path = plot_dir / f"06_b13_rn_attention__{group}.png"
        _finish(fig, path); made.append(path)

        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.plot(heads, [_float(r, "steal_norm_mean") for r in sub], marker="o", label="Softmax steal")
        ax.plot(heads, [_float(r, "write_norm_mean") for r in sub], marker="o", label="Active RN write")
        ax.plot(heads, [_float(r, "delta_norm_mean") for r in sub], marker="o", label="Total head delta")
        ax.set_title(f"B13 RN steal vs active value write ({group})")
        ax.set_xlabel("Head")
        ax.set_ylabel("Mean vector norm")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
        path = plot_dir / f"07_b13_steal_vs_write__{group}.png"
        _finish(fig, path); made.append(path)
    return made


def plot_pathway(run_dir: Path, plot_dir: Path) -> list[Path]:
    path_json = run_dir / "b13_attention_vs_mlp_pathway.json"
    if not path_json.is_file():
        return []
    data = json.loads(path_json.read_text(encoding="utf-8"))
    keys = ["attn_delta_norm_mean", "mlp_response_delta_norm_mean", "post_delta_norm_mean"]
    if not all(k in data for k in keys):
        return []
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    labels = ["Attention delta", "MLP response delta", "Post-B13 delta"]
    ax.bar(np.arange(3), [float(data[k]) for k in keys])
    ax.set_xticks(np.arange(3), labels, rotation=15, ha="right")
    ax.set_title("B13 pathway decomposition")
    ax.set_ylabel("Mean state-change norm")
    ax.grid(True, axis="y", alpha=0.2)
    path = plot_dir / "08_b13_attention_vs_mlp_pathway.png"
    _finish(fig, path)
    return [path]


def plot_causal_low_rank(run_dir: Path, plot_dir: Path) -> list[Path]:
    rows = _read_csv(run_dir / "causal_low_rank_test.csv")
    if not rows:
        return []
    plt = _mpl(); made = []
    variants = sorted({r.get("variant", "") for r in rows if r.get("variant")})
    modes = sorted({r.get("mode", "") for r in rows if r.get("mode")})
    for variant in variants:
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for mode in modes:
            sub = sorted([r for r in rows if r.get("variant") == variant and r.get("mode") == mode], key=lambda r: _int(r, "rank"))
            if not sub:
                continue
            ax.plot([_int(r, "rank") for r in sub], [_float(r, "effect_recovery_mean") for r in sub], marker="o", label=mode)
        ax.axhline(0.0, linewidth=1.0)
        ax.axhline(1.0, linewidth=1.0)
        ax.set_title(f"Causal low-rank RN steering recovery ({variant})")
        ax.set_xlabel("Rank k")
        ax.set_ylabel("Effect recovery toward RN embedding shift")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        path = plot_dir / f"09_causal_low_rank_recovery__{variant}.png"
        _finish(fig, path); made.append(path)

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for mode in modes:
            sub = sorted([r for r in rows if r.get("variant") == variant and r.get("mode") == mode], key=lambda r: _int(r, "rank"))
            if not sub:
                continue
            ax.plot([_int(r, "rank") for r in sub], [_float(r, "target_residual_ratio_mean") for r in sub], marker="o", label=mode)
        ax.set_title(f"Residual distance from true RN state effect ({variant})")
        ax.set_xlabel("Rank k")
        ax.set_ylabel("Residual / RN displacement norm")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        path = plot_dir / f"10_causal_low_rank_residual__{variant}.png"
        _finish(fig, path); made.append(path)
    return made


def plot_jacobian(run_dir: Path, plot_dir: Path) -> list[Path]:
    plt = _mpl(); made = []
    for point in ("trained", "zero"):
        path_npz = run_dir / "jacobian" / f"b13_rn_input_spectrum__{point}.npz"
        if not path_npz.is_file():
            continue
        z = np.load(path_npz)
        # tolerate key naming variants
        singular = None
        for k in ("singular_values", "values", "s"):
            if k in z:
                singular = np.asarray(z[k], dtype=np.float64).reshape(-1)
                break
        if singular is None:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 4.9))
        idx = np.arange(1, len(singular) + 1)
        ax.plot(idx, singular, marker="o")
        ax.set_title(f"Frozen B13 RN-input Jacobian spectrum ({point} RN point)")
        ax.set_xlabel("Singular direction")
        ax.set_ylabel("Estimated singular value")
        ax.grid(True, alpha=0.2)
        path = plot_dir / f"11_jacobian_spectrum__{point}.png"
        _finish(fig, path); made.append(path)

        cos = None
        for k in ("rn_cosines", "cos_with_trained_rn", "trained_rn_cosine"):
            if k in z:
                cos = np.asarray(z[k], dtype=np.float64).reshape(-1)
                break
        if cos is not None and len(cos):
            fig, ax = plt.subplots(figsize=(7.6, 4.9))
            ax.bar(idx[:len(cos)], cos)
            ax.axhline(0.0, linewidth=1.0)
            ax.set_title(f"Learned RN alignment with B13 high-gain input directions ({point})")
            ax.set_xlabel("Singular direction")
            ax.set_ylabel("Cosine with trained RN")
            ax.set_ylim(-1.02, 1.02)
            ax.grid(True, axis="y", alpha=0.2)
            path = plot_dir / f"12_jacobian_rn_alignment__{point}.png"
            _finish(fig, path); made.append(path)
    return made


def render_all_plots(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    for fn in (
        plot_svd_energy,
        plot_universal_fraction,
        plot_subspace_alignment,
        plot_embedding_conditions,
        plot_b13_head_decomposition,
        plot_pathway,
        plot_causal_low_rank,
        plot_jacobian,
    ):
        made.extend(fn(run_dir, plot_dir))
    index = plot_dir / "PLOTS_GENERATED.txt"
    index.write_text("\n".join(p.name for p in made) + ("\n" if made else ""), encoding="utf-8")
    return made
