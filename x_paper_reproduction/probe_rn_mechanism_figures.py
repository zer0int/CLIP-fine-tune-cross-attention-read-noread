#!/usr/bin/env python3
r"""RN mechanism paper figures from cached results.
Commands: native (native multilingual surface schema), legacy (earlier universality schema).
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()

# NATIVE
import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CONTROL_DIR = Path(r"rn_control_mechinterp")
DEFAULT_STASH_DIR = Path(r"rn_stash_followup")
DEFAULT_OUTPUT_DIR = Path(r"rn_paper_figures_compact")

MODEL_LABELS = {
    "A_native": "trained V / trained T",
    "B_preV_preT": "pretrained V / pretrained T",
    "C_preV_trainT": "pretrained V / trained T",
    "D_trainV_preT": "trained V / pretrained T",
}
LANG_ORDER = ["en", "de", "ar", "zh", "ru"]


def native_parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    ap.add_argument("--stash-dir", type=Path, default=DEFAULT_STASH_DIR)
    ap.add_argument(
        "--universality-dir",
        type=Path,
        default=None,
        help="Result directory containing data/cross_language_geometry.csv; cross_model_surface_alignment.csv is optional.",
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--variant", default="text", help="Primary synthetic variant for Figs. 1-2.")
    ap.add_argument("--pump-condition", default="intact")
    ap.add_argument("--model", default="trained", help="Stash-followup model label for Fig. 1.")
    ap.add_argument("--query-mode", default="native", choices=["native", "english"])
    ap.add_argument("--pc-plane", default="1x2", help="Control-surface plane for Fig. 3, e.g. 1x2 or 1x4.")
    ap.add_argument("--dpi", type=int, default=220)
    return ap.parse_args()


def parse_plane(text: str) -> tuple[int, int]:
    bits = text.lower().replace("pc", "").split("x")
    if len(bits) != 2:
        raise ValueError(f"Bad plane {text!r}; expected e.g. 1x2")
    return int(bits[0]), int(bits[1])


def data_file(root: Optional[Path], name: str) -> Optional[Path]:
    if root is None:
        return None
    candidates = [root / name, root / "data" / name, root / "derived_tables" / name]
    for p in candidates:
        if p.is_file():
            return p
    return None


def read_csv(root: Optional[Path], name: str) -> pd.DataFrame:
    p = data_file(root, name)
    if p is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        # Single-language / reduced smoke runs can legitimately emit an
        # optional cross-language CSV with zero columns. Treat that as
        # "no optional rows" rather than a corrupt result. Required figure
        # inputs are checked by the figure builders themselves.
        return pd.DataFrame()


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def finite_mean(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.11, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def head_lines(ax: plt.Axes, df: pd.DataFrame, metrics: list[tuple[str, str]]) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return
    q = df.sort_values("head")
    x = safe_numeric(q, "head").to_numpy()
    for col, label in metrics:
        if col in q.columns:
            ax.plot(x, safe_numeric(q, col), marker="o", label=label)
    ax.set_xlabel("B13 attention head")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)


def choose_metric_column(df: pd.DataFrame) -> Optional[str]:
    """Find the displacement-recovery scalar written by displacement_metrics()."""
    preferred = [
        "recovery_projection",
        "projection_recovery",
        "displacement_recovery",
        "recovery_fraction",
        "recovery",
        "projection_fraction",
        "fraction_recovered",
        "delta_projection_fraction",
        "along_target_fraction",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    # Conservative lexical fallback: numeric column containing both a recovery-like
    # term and not merely metadata/cosine/error.
    bad = ("cos", "angle", "error", "norm", "rank", "scene", "fit", "test")
    for c in df.columns:
        low = c.lower()
        if ("recover" in low or "projection" in low or "fraction" in low) and not any(b in low for b in bad):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                return c
    return None


def choose_decomp_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Best-effort mapping for the exact B13 decomposition CSV across kit revisions."""
    aliases = {
        "attn": ["rn_attention", "attention_to_rn", "p_rn", "rn_attn", "attention_mass"],
        "steal": ["steal_norm", "softmax_steal_norm", "nop_norm", "attention_steal_norm"],
        "write": ["write_norm", "active_write_norm", "rn_write_norm", "value_write_norm"],
        "recon": ["reconstruction_error", "relative_reconstruction_error", "recon_error", "relative_error"],
    }
    out: dict[str, Optional[str]] = {}
    for key, names in aliases.items():
        out[key] = next((n for n in names if n in df.columns), None)
    return out


def native_find_universality_root(explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.exists() else None
    candidates = [
        Path("rn_bridge_transplant_universality"),
        Path("compact_summary_rn_bridge_transplant"),
    ]
    for p in candidates:
        if data_file(p, "cross_language_geometry.csv") is not None:
            return p
    return None


def figure1_b13_address_payload(
    stash_root: Path,
    control_root: Path,
    out: Path,
    *,
    model: str,
    variant: str,
    pump_condition: str,
    dpi: int,
    headline_rows: list[dict[str, Any]],
) -> Optional[Path]:
    stash = read_csv(stash_root, "b13_stash_impersonation.csv")
    hybrids = read_csv(stash_root, "b13_kv_hybrid_causal.csv")
    decomp = read_csv(control_root, "b13_head_kv_decomposition.csv")

    if stash.empty and hybrids.empty and decomp.empty:
        return None

    if not stash.empty:
        q = stash.copy()
        for c in ("model", "variant", "pump_condition", "query_group"):
            if c in q.columns:
                q[c] = q[c].astype(str)
        q = q[
            (q.get("model", "") == model)
            & (q.get("variant", "") == variant)
            & (q.get("pump_condition", "") == pump_condition)
            & (q.get("query_group", "") == "cls")
        ].copy()
    else:
        q = pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))

    ax = axes[0, 0]
    panel_letter(ax, "A")
    head_lines(
        ax,
        q,
        [
            ("k_rn_vs_reg_cos", "RN ↔ native register K"),
            ("v_rn_vs_reg_cos", "RN ↔ native register V"),
            ("wo_v_rn_vs_reg_cos", "RN ↔ native register $W_OV$"),
            ("query_logit_profile_cos", "ordinary-query qK profile"),
        ],
    )
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(-0.15, 1.02)
    ax.set_title("Address resembles the native source; payload does not")

    ax = axes[0, 1]
    panel_letter(ax, "B")
    head_lines(
        ax,
        q,
        [
            ("attn_to_reg_base", "native register · RN absent"),
            ("attn_to_reg_with_rn", "native register · RN present"),
            ("attn_to_rn", "READ_NULL"),
        ],
    )
    ax.set_ylabel("CLS attention probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("READ_NULL substitutes for the native B13 source")

    ax = axes[1, 0]
    panel_letter(ax, "C")
    decomp_cols = choose_decomp_columns(decomp)
    d = decomp.copy()
    if not d.empty and "token_group" in d.columns:
        # Prefer CLS; fall back to the first available group.
        groups = d["token_group"].astype(str)
        if (groups == "cls").any():
            d = d[groups == "cls"].copy()
    if not d.empty and decomp_cols["steal"] and decomp_cols["write"]:
        d = d.sort_values("head")
        x = safe_numeric(d, "head").to_numpy()
        ax.plot(x, safe_numeric(d, decomp_cols["steal"]), marker="o", label="softmax steal / NOP term")
        ax.plot(x, safe_numeric(d, decomp_cols["write"]), marker="o", label="active RN value write")
        if decomp_cols["attn"]:
            ax2 = ax.twinx()
            ax2.plot(x, safe_numeric(d, decomp_cols["attn"]), marker=".", linestyle="--", label="CLS→RN attention")
            ax2.set_ylabel("attention probability")
            ax2.set_ylim(-0.03, 1.03)
        ax.set_xlabel("B13 attention head")
        ax.set_ylabel("head-update norm")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title("High attention is value-active, not a passive sink")
    else:
        head_lines(
            ax,
            q,
            [
                ("reg_value_write_norm_base", "native register write · RN absent"),
                ("reg_value_write_norm_with_rn", "native register write · RN present"),
                ("rn_value_write_norm", "RN write"),
            ],
        )
        ax.set_ylabel("value contribution norm")
        ax.set_title("READ_NULL carries a large active value payload")

    ax = axes[1, 1]
    panel_letter(ax, "D")
    h = hybrids.copy()
    if not h.empty:
        for c in ("model", "variant", "condition"):
            if c in h.columns:
                h[c] = h[c].astype(str)
        h = h[(h.get("model", "") == model) & (h.get("variant", "") == variant)].copy()
    if not h.empty and "recovery_projection" in h.columns:
        preferred_order = ["RN", "RN_K__REG_V", "REG_K__RN_V", "REG_KV"]
        h["_order"] = h["condition"].map({x: i for i, x in enumerate(preferred_order)}).fillna(99)
        h = h.sort_values("_order")
        labels = {
            "RN": "RN K + RN V",
            "RN_K__REG_V": "RN K + register V",
            "REG_K__RN_V": "register K + RN V",
            "REG_KV": "register K + register V",
        }
        xs = np.arange(len(h))
        ax.bar(xs, safe_numeric(h, "recovery_projection"))
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([labels.get(x, x) for x in h["condition"].astype(str)], rotation=20, ha="right")
        ax.set_ylabel("fraction of RN displacement recovered")
        ax.set_title("K and V form a co-adapted control packet")
        ax.grid(True, axis="y", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "K/V hybrid causal table not found", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle("B13 READ_NULL control packet: native addressability, novel active payload", fontsize=15)
    path = out / "fig1_b13_address_and_payload.png"
    save_figure(fig, path, dpi)

    # Headline numeric audit from the strongest RN-attention head.
    if not q.empty and "attn_to_rn" in q.columns:
        qq = q.copy()
        qq["attn_to_rn"] = safe_numeric(qq, "attn_to_rn")
        if qq["attn_to_rn"].notna().any():
            r = qq.loc[qq["attn_to_rn"].idxmax()]
            for c in (
                "attn_to_rn", "attn_to_reg_base", "attn_to_reg_with_rn",
                "k_rn_vs_reg_cos", "v_rn_vs_reg_cos", "wo_v_rn_vs_reg_cos",
                "query_logit_profile_cos", "rn_value_write_norm",
            ):
                if c in r.index:
                    headline_rows.append({"figure": 1, "scope": f"strongest RN head H{int(r['head'])}", "metric": c, "value": float(r[c])})

    return path


def figure2_low_rank_pulse(
    control_root: Path,
    out: Path,
    *,
    variant: str,
    dpi: int,
    headline_rows: list[dict[str, Any]],
) -> Optional[Path]:
    svd = read_csv(control_root, "svd_summary.csv")
    embed = read_csv(control_root, "embedding_condition_summary_per_scene.csv")
    causal = read_csv(control_root, "causal_low_rank_test.csv")
    align = read_csv(control_root, "subspace_alignment.csv")

    if svd.empty and embed.empty and causal.empty and align.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))

    # A: PC1 energy trajectory, patch-mean view.
    ax = axes[0, 0]
    panel_letter(ax, "A")
    s = svd.copy()
    if not s.empty:
        s["variant"] = s["variant"].astype(str)
        s["view"] = s["view"].astype(str)
        view = "patch_mean" if (s["view"] == "patch_mean").any() else str(s["view"].iloc[0])
        keep_variants = [x for x in [variant, "sine", "blank"] if (s["variant"] == x).any()]
        for v in keep_variants:
            z = s[(s["variant"] == v) & (s["view"] == view)].sort_values("block")
            if not z.empty:
                ax.plot(safe_numeric(z, "block"), safe_numeric(z, "energy_pc1"), marker="o", label=v)
        ax.set_xlabel("visual block")
        ax.set_ylabel("uncentered RN-delta energy in PC1")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        ax.set_title(f"The B13 control pulse starts nearly rank-1 ({view})")
    else:
        ax.text(0.5, 0.5, "svd_summary.csv not found", ha="center", va="center"); ax.set_axis_off()

    # B: top-k energy trajectory.
    ax = axes[0, 1]
    panel_letter(ax, "B")
    if not s.empty:
        view = "patch_mean" if (s["view"] == "patch_mean").any() else str(s["view"].iloc[0])
        keep_variants = [x for x in [variant, "sine", "blank"] if (s["variant"] == x).any()]
        for v in keep_variants:
            z = s[(s["variant"] == v) & (s["view"] == view)].sort_values("block")
            if not z.empty:
                ax.plot(safe_numeric(z, "block"), safe_numeric(z, "energy_topk"), marker="o", label=v)
        ax.set_xlabel("visual block")
        ax.set_ylabel("energy in fitted top-k RN subspace")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        ax.set_title("The pulse fans out while remaining highly structured")
    else:
        ax.text(0.5, 0.5, "svd_summary.csv not found", ha="center", va="center"); ax.set_axis_off()

    # C: pure NOP / zero-V control.
    ax = axes[1, 0]
    panel_letter(ax, "C")
    if not embed.empty and "nop_over_tag_shift" in embed.columns:
        e = embed.copy()
        e["variant"] = e["variant"].astype(str)
        agg = e.groupby("variant", observed=True)["nop_over_tag_shift"].agg(["mean", "sem"]).reset_index()
        order = [x for x in [variant, "phase", "shred", "sine", "checker", "blank"] if x in set(agg["variant"])]
        agg["_o"] = agg["variant"].map({x: i for i, x in enumerate(order)}).fillna(99)
        agg = agg.sort_values("_o")
        xs = np.arange(len(agg))
        ax.bar(xs, safe_numeric(agg, "mean"), yerr=safe_numeric(agg, "sem"), capsize=3)
        ax.set_xticks(xs); ax.set_xticklabels(agg["variant"], rotation=25, ha="right")
        ax.set_ylabel("zero-V shift / normal RN shift")
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_title("Keeping the K destination but deleting V removes most steering")
        row = agg[agg["variant"] == variant]
        if not row.empty:
            headline_rows.append({"figure": 2, "scope": variant, "metric": "zero_v_over_full_tag_shift", "value": float(row.iloc[0]["mean"])})
    else:
        ax.text(0.5, 0.5, "zero-V summary not found", ha="center", va="center"); ax.set_axis_off()

    # D: held-out causal low-rank sufficiency / necessity.
    ax = axes[1, 1]
    panel_letter(ax, "D")
    c = causal.copy()
    metric = choose_metric_column(c)
    if not c.empty and metric is not None:
        c["variant"] = c["variant"].astype(str)
        c["mode"] = c["mode"].astype(str)
        c = c[c["variant"] == variant].copy()
        for mode, label in [
            ("paired_feature_projection", "retain rank-k RN component"),
            ("fixed_mean_template", "fixed rank-k mean template"),
            ("remove_feature_projection", "RN effect remaining after removal"),
        ]:
            z = c[c["mode"] == mode].sort_values("rank")
            if not z.empty:
                ax.plot(safe_numeric(z, "rank"), safe_numeric(z, metric), marker="o", label=label)
                for _, r in z.iterrows():
                    if int(r["rank"]) in (4, 8):
                        headline_rows.append({"figure": 2, "scope": f"{variant}/{mode}/rank{int(r['rank'])}", "metric": metric, "value": float(r[metric])})
        ax.axhline(0.0, linewidth=1)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("rank k")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        ax.set_title("A tiny feature subspace is causally sufficient and necessary")
    else:
        msg = "causal table unavailable"
        if not c.empty:
            msg += "\nmetric columns:\n" + ", ".join(c.columns)
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8); ax.set_axis_off()

    fig.suptitle("READ_NULL writes a low-dimensional B13 pulse that fans out through native CLIP", fontsize=15)
    path = out / "fig2_low_rank_control_pulse.png"
    save_figure(fig, path, dpi)
    return path


def symmetric_pair_matrix(
    df: pd.DataFrame,
    a_col: str,
    b_col: str,
    value_col: str,
    labels: list[str],
) -> np.ndarray:
    mat = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
    idx = {x: i for i, x in enumerate(labels)}
    np.fill_diagonal(mat, 1.0)
    for _, r in df.iterrows():
        a, b = str(r[a_col]), str(r[b_col])
        if a not in idx or b not in idx:
            continue
        try:
            v = float(r[value_col])
        except Exception:
            continue
        mat[idx[a], idx[b]] = v
        mat[idx[b], idx[a]] = v
    return mat


def heatmap(ax: plt.Axes, mat: np.ndarray, labels: list[str], title: str) -> None:
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels([x.upper() for x in labels])
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels([x.upper() for x in labels])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman")


def _language_order_from_crosslang(df: pd.DataFrame) -> list[str]:
    preferred = ["en", "de", "es", "fr", "it", "ar", "ru", "zh", "ja", "ko"]
    seen = set()
    for c in ("language_a", "language_b"):
        if c in df.columns:
            seen.update(str(x) for x in df[c].dropna().astype(str).tolist())
    ordered = [x for x in preferred if x in seen]
    ordered += sorted(seen - set(ordered))
    return ordered


def native_figure3_shared_geometry_language_readout(
    universality_root: Path,
    out: Path,
    *,
    query_mode: str,
    plane: tuple[int, int],
    dpi: int,
    headline_rows: list[dict[str, Any]],
) -> Optional[Path]:
    """
    Native-only multilingual paper figure.

    No transplant rows are required. The central comparison is within the actual
    trained A_native/intact model: B21 state-sheet geometry is compared with the
    language/query-conditioned READ scalar field over that same sheet.
    """
    crosslang = read_csv(universality_root, "cross_language_geometry.csv")
    if crosslang.empty:
        return None

    cl = crosslang.copy()
    for c in ("model_name", "pump_mode", "query_mode", "language_a", "language_b"):
        if c in cl.columns:
            cl[c] = cl[c].astype(str)

    base = cl[
        (cl["model_name"] == "A_native")
        & (cl["pump_mode"] == "intact")
        & (safe_numeric(cl, "pc_a") == plane[0])
        & (safe_numeric(cl, "pc_b") == plane[1])
    ].copy()
    if base.empty:
        return None

    langs = _language_order_from_crosslang(base)
    if len(langs) < 2:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.2))

    # A: every language pair, state geometry vs READ field.
    ax = axes[0, 0]
    panel_letter(ax, "A")
    rq = base[base["query_mode"] == query_mode].copy()
    x = safe_numeric(rq, "state_distance_spearman").to_numpy(float)
    y = safe_numeric(rq, "read_surface_spearman").to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[finite], y[finite], s=36)
    for (_, r), xx, yy in zip(rq.loc[finite].iterrows(), x[finite], y[finite]):
        ax.annotate(
            f"{str(r['language_a']).upper()}-{str(r['language_b']).upper()}",
            (xx, yy), fontsize=6, alpha=0.80,
        )
    lo = min(0.0, float(np.nanmin(np.r_[x[finite], y[finite]])) if finite.any() else 0.0)
    ax.plot([lo, 1.02], [lo, 1.02], linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlim(lo - 0.02, 1.03); ax.set_ylim(lo - 0.02, 1.03)
    ax.set_xlabel("cross-language B21 state-geometry Spearman")
    ax.set_ylabel(f"cross-language READ-field Spearman · {query_mode} Q")
    ax.grid(True, alpha=0.2)
    ax.set_title("Same visual control sheet; readout varies more across languages")

    # B: query-independent visual state geometry matrix.
    geom = base[base["query_mode"] == "english"] if (base["query_mode"] == "english").any() else base
    mat_g = symmetric_pair_matrix(geom, "language_a", "language_b", "state_distance_spearman", langs)
    ax = axes[0, 1]; panel_letter(ax, "B")
    heatmap(ax, mat_g, langs, "Cross-language B21 state geometry")

    # C: native-query READ field matrix.
    mat_r = symmetric_pair_matrix(rq, "language_a", "language_b", "read_surface_spearman", langs)
    ax = axes[1, 0]; panel_letter(ax, "C")
    heatmap(ax, mat_r, langs, f"Cross-language READ field · {query_mode} query")

    tri = np.triu_indices(len(langs), k=1)
    headline_rows.append({
        "figure": 3,
        "scope": f"A_native/intact/PC{plane[0]}xPC{plane[1]}",
        "metric": "mean_cross_language_state_geometry_spearman",
        "value": finite_mean(mat_g[tri]),
    })
    headline_rows.append({
        "figure": 3,
        "scope": f"A_native/intact/{query_mode}/PC{plane[0]}xPC{plane[1]}",
        "metric": "mean_cross_language_read_field_spearman",
        "value": finite_mean(mat_r[tri]),
    })

    # D: English as reference language. This is a compact script-family diagnostic:
    # Latin-vs-Latin should be visually comparable to Cyrillic/Hanzi/etc without
    # pretending the language ordering itself is a continuous variable.
    ax = axes[1, 1]
    panel_letter(ax, "D")
    if "en" in langs:
        others = [l for l in langs if l != "en"]
        state_vals = []
        read_vals = []
        for lang in others:
            pair = base[
                (((base["language_a"] == "en") & (base["language_b"] == lang))
                 | ((base["language_b"] == "en") & (base["language_a"] == lang)))
            ]
            # State metric is query independent; use first finite copy.
            sv = safe_numeric(pair, "state_distance_spearman").dropna()
            state_vals.append(float(sv.iloc[0]) if len(sv) else np.nan)
            pv = pair[pair["query_mode"] == query_mode]
            rv = safe_numeric(pv, "read_surface_spearman").dropna()
            read_vals.append(float(rv.iloc[0]) if len(rv) else np.nan)
        xs = np.arange(len(others))
        ax.plot(xs, state_vals, marker="o", label="B21 state geometry")
        ax.plot(xs, read_vals, marker="o", label=f"READ field ({query_mode})")
        ax.set_xticks(xs); ax.set_xticklabels([x.upper() for x in others])
        ax.set_ylim(-0.05, 1.03)
        ax.set_ylabel("Spearman similarity to EN surface")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        ax.set_title("Similarity to English: conserved sheet vs script/language-sensitive readout")
    else:
        ax.text(0.5, 0.5, "English not present in language set", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle("Native RN control surface: shared visual geometry, language-dependent READ field", fontsize=15)
    path = out / "fig3_shared_geometry_language_readout.png"
    save_figure(fig, path, dpi)
    return path


def native_write_readme(out: Path, made: list[Path], universality_root: Optional[Path]) -> None:
    lines = [
        "RN compact paper-figure pass",
        "============================",
        "",
        "The plotting script reads existing experiment outputs only; it does not run CLIP.",
        "",
        "Figure logic:",
        "  Fig. 1: B13 address + payload. K resembles the native register source more than V/W_OV do;",
        "          RN substitutes as an attended source and its active value write is functionally essential.",
        "  Fig. 2: Low-rank control pulse. RN-on minus RN-off is very low-dimensional at B13;",
        "          low-rank reconstruction/removal tests causal sufficiency/necessity; zero-V is the NOP control.",
        "  Fig. 3: Shared state geometry vs language-dependent READ field. Generated only when",
        "          the multilingual native control-surface result tables are available.",
        "",
        "Generated:",
    ]
    for p in made:
        lines.append(f"  - {p.name}")
    if universality_root is None:
        lines += [
            "",
            "Fig. 3 was skipped because no multilingual control-surface result root was found.",
            "Pass --universality-dir <path-to-native-control-surface-results> to enable it.",
        ]
    (out / "README_FIRST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def native_main() -> None:
    args = native_parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plane = parse_plane(args.pc_plane)
    universality_root = native_find_universality_root(args.universality_dir)

    headline_rows: list[dict[str, Any]] = []
    made: list[Path] = []

    p = figure1_b13_address_payload(
        args.stash_dir,
        args.control_dir,
        out,
        model=args.model,
        variant=args.variant,
        pump_condition=args.pump_condition,
        dpi=args.dpi,
        headline_rows=headline_rows,
    )
    if p is not None:
        made.append(p)

    p = figure2_low_rank_pulse(
        args.control_dir,
        out,
        variant=args.variant,
        dpi=args.dpi,
        headline_rows=headline_rows,
    )
    if p is not None:
        made.append(p)

    if universality_root is not None:
        p = native_figure3_shared_geometry_language_readout(
            universality_root,
            out,
            query_mode=args.query_mode,
            plane=plane,
            dpi=args.dpi,
            headline_rows=headline_rows,
        )
        if p is not None:
            made.append(p)

    pd.DataFrame(headline_rows).to_csv(out / "headline_numbers.csv", index=False)
    manifest = {
        "control_dir": str(args.control_dir),
        "stash_dir": str(args.stash_dir),
        "universality_dir": None if universality_root is None else str(universality_root),
        "variant": args.variant,
        "pump_condition": args.pump_condition,
        "stash_model": args.model,
        "query_mode": args.query_mode,
        "pc_plane": list(plane),
        "figures": [p.name for p in made],
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    native_write_readme(out, made, universality_root)

    print("[done]")
    for p in made:
        print("  ", p)
    print("  ", out / "headline_numbers.csv")
    print("  ", out / "README_FIRST.txt")


# LEGACY
def legacy_parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    ap.add_argument("--stash-dir", type=Path, default=DEFAULT_STASH_DIR)
    ap.add_argument(
        "--universality-dir",
        type=Path,
        default=None,
        help="Result directory containing data/cross_model_surface_alignment.csv and data/cross_language_geometry.csv.",
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--variant", default="text", help="Primary synthetic variant for Figs. 1-2.")
    ap.add_argument("--pump-condition", default="intact")
    ap.add_argument("--model", default="trained", help="Stash-followup model label for Fig. 1.")
    ap.add_argument("--query-mode", default="native", choices=["native", "english"])
    ap.add_argument("--pc-plane", default="1x2", help="Control-surface plane for Fig. 3, e.g. 1x2 or 1x4.")
    ap.add_argument("--dpi", type=int, default=220)
    return ap.parse_args()


def legacy_find_universality_root(explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.exists() else None
    candidates = [
        Path("rn_bridge_transplant_universality"),
        Path("compact_summary_rn_bridge_transplant"),
    ]
    for p in candidates:
        if data_file(p, "cross_model_surface_alignment.csv") is not None:
            return p
    return None


def legacy_figure3_shared_geometry_language_readout(
    universality_root: Path,
    out: Path,
    *,
    query_mode: str,
    plane: tuple[int, int],
    dpi: int,
    headline_rows: list[dict[str, Any]],
) -> Optional[Path]:
    crossmodel = read_csv(universality_root, "cross_model_surface_alignment.csv")
    crosslang = read_csv(universality_root, "cross_language_geometry.csv")
    if crossmodel.empty and crosslang.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))

    # A: visual geometry transfer vs READ-field transfer.
    ax = axes[0, 0]
    panel_letter(ax, "A")
    cm = crossmodel.copy()
    if not cm.empty:
        for c in ("model_name", "pump_mode", "query_mode", "language"):
            if c in cm.columns:
                cm[c] = cm[c].astype(str)
        z = cm[
            (safe_numeric(cm, "pc_a") == plane[0])
            & (safe_numeric(cm, "pc_b") == plane[1])
            & (cm["query_mode"] == query_mode)
        ].copy()
        for model in ["A_native", "B_preV_preT", "C_preV_trainT", "D_trainV_preT"]:
            a = z[(z["model_name"] == model) & (z["pump_mode"] == "intact")]
            if a.empty:
                continue
            ax.scatter(
                safe_numeric(a, "state_distance_spearman_to_reference"),
                safe_numeric(a, "read_surface_spearman_to_reference"),
                label=MODEL_LABELS.get(model, model),
            )
            for _, r in a.iterrows():
                ax.annotate(str(r["language"]).upper(), (float(r["state_distance_spearman_to_reference"]), float(r["read_surface_spearman_to_reference"])), fontsize=7)
        ax.set_xlabel("B21 state-geometry Spearman to trained V/T")
        ax.set_ylabel("READ-field Spearman to trained V/T")
        ax.set_xlim(0, 1.03); ax.set_ylim(-0.05, 1.03)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)
        ax.set_title("State geometry transfers more strongly than behavioral readout")
    else:
        ax.text(0.5, 0.5, "cross-model table not found", ha="center", va="center"); ax.set_axis_off()

    # B/C: matched state and READ matrices for native trained model.
    cl = crosslang.copy()
    if not cl.empty:
        for c in ("model_name", "pump_mode", "query_mode", "language_a", "language_b"):
            if c in cl.columns:
                cl[c] = cl[c].astype(str)
        base = cl[
            (cl["model_name"] == "A_native")
            & (cl["pump_mode"] == "intact")
            & (safe_numeric(cl, "pc_a") == plane[0])
            & (safe_numeric(cl, "pc_b") == plane[1])
        ].copy()

        geom = base[base["query_mode"] == "english"] if (base["query_mode"] == "english").any() else base
        mat_g = symmetric_pair_matrix(geom, "language_a", "language_b", "state_distance_spearman", LANG_ORDER)
        ax = axes[0, 1]; panel_letter(ax, "B")
        heatmap(ax, mat_g, LANG_ORDER, "Cross-language B21 state geometry")

        rq = base[base["query_mode"] == query_mode]
        mat_r = symmetric_pair_matrix(rq, "language_a", "language_b", "read_surface_spearman", LANG_ORDER)
        ax = axes[1, 0]; panel_letter(ax, "C")
        heatmap(ax, mat_r, LANG_ORDER, f"Cross-language READ field · {query_mode} query")

        # Audit means over off-diagonal entries.
        tri = np.triu_indices(len(LANG_ORDER), k=1)
        headline_rows.append({"figure": 3, "scope": f"A_native/intact/PC{plane[0]}xPC{plane[1]}", "metric": "mean_cross_language_state_geometry_spearman", "value": finite_mean(mat_g[tri])})
        headline_rows.append({"figure": 3, "scope": f"A_native/intact/{query_mode}/PC{plane[0]}xPC{plane[1]}", "metric": "mean_cross_language_read_field_spearman", "value": finite_mean(mat_r[tri])})
    else:
        for ax, letter, title in [(axes[0, 1], "B", "state geometry"), (axes[1, 0], "C", "READ field")]:
            panel_letter(ax, letter); ax.text(0.5, 0.5, f"cross-language {title} table not found", ha="center", va="center"); ax.set_axis_off()

    # D: text-tower contract on the same pretrained ViT.
    ax = axes[1, 1]
    panel_letter(ax, "D")
    if not cm.empty:
        z = cm[
            (safe_numeric(cm, "pc_a") == plane[0])
            & (safe_numeric(cm, "pc_b") == plane[1])
            & (cm["query_mode"] == query_mode)
            & (cm["pump_mode"] == "intact")
            & (cm["model_name"].isin(["B_preV_preT", "C_preV_trainT"]))
        ].copy()
        if not z.empty:
            xmap = {lang: i for i, lang in enumerate(LANG_ORDER)}
            for model, label in [
                ("B_preV_preT", "pretrained V + pretrained T"),
                ("C_preV_trainT", "pretrained V + trained T"),
            ]:
                a = z[z["model_name"] == model].copy()
                if a.empty:
                    continue
                a["_x"] = a["language"].map(xmap)
                a = a.sort_values("_x")
                ax.plot(a["_x"], safe_numeric(a, "read_surface_spearman_to_reference"), marker="o", label=label)
            ax.set_xticks(np.arange(len(LANG_ORDER))); ax.set_xticklabels([x.upper() for x in LANG_ORDER])
            ax.set_ylim(-0.05, 1.03)
            ax.set_ylabel("READ-field Spearman to trained V/T")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)
            ax.set_title("Same pretrained ViT; swapping the text tower changes the READ contract")
        else:
            ax.text(0.5, 0.5, "preV/preT vs preV/trainT rows not found", ha="center", va="center"); ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "cross-model table not found", ha="center", va="center"); ax.set_axis_off()

    fig.suptitle("Shared visual control sheet, language/query-dependent READ field", fontsize=15)
    path = out / "fig3_shared_geometry_language_readout.png"
    save_figure(fig, path, dpi)
    return path


def legacy_write_readme(out: Path, made: list[Path], universality_root: Optional[Path]) -> None:
    lines = [
        "RN compact paper-figure pass",
        "============================",
        "",
        "The plotting script reads existing experiment outputs only; it does not run CLIP.",
        "",
        "Figure logic:",
        "  Fig. 1: B13 address + payload. K resembles the native register source more than V/W_OV do;",
        "          RN substitutes as an attended source and its active value write is functionally essential.",
        "  Fig. 2: Low-rank control pulse. RN-on minus RN-off is very low-dimensional at B13;",
        "          low-rank reconstruction/removal tests causal sufficiency/necessity; zero-V is the NOP control.",
        "  Fig. 3: Shared state geometry vs language-dependent READ field. Generated only when",
        "          the universality result tables are available.",
        "",
        "Generated:",
    ]
    for p in made:
        lines.append(f"  - {p.name}")
    if universality_root is None:
        lines += [
            "",
            "Fig. 3 was skipped because no universality result root was found.",
            "Pass --universality-dir <path-to-rn_bridge_transplant_universality> to enable it.",
        ]
    (out / "README_FIRST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def legacy_main() -> None:
    args = legacy_parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plane = parse_plane(args.pc_plane)
    universality_root = legacy_find_universality_root(args.universality_dir)

    headline_rows: list[dict[str, Any]] = []
    made: list[Path] = []

    p = figure1_b13_address_payload(
        args.stash_dir,
        args.control_dir,
        out,
        model=args.model,
        variant=args.variant,
        pump_condition=args.pump_condition,
        dpi=args.dpi,
        headline_rows=headline_rows,
    )
    if p is not None:
        made.append(p)

    p = figure2_low_rank_pulse(
        args.control_dir,
        out,
        variant=args.variant,
        dpi=args.dpi,
        headline_rows=headline_rows,
    )
    if p is not None:
        made.append(p)

    if universality_root is not None:
        p = legacy_figure3_shared_geometry_language_readout(
            universality_root,
            out,
            query_mode=args.query_mode,
            plane=plane,
            dpi=args.dpi,
            headline_rows=headline_rows,
        )
        if p is not None:
            made.append(p)

    pd.DataFrame(headline_rows).to_csv(out / "headline_numbers.csv", index=False)
    manifest = {
        "control_dir": str(args.control_dir),
        "stash_dir": str(args.stash_dir),
        "universality_dir": None if universality_root is None else str(universality_root),
        "variant": args.variant,
        "pump_condition": args.pump_condition,
        "stash_model": args.model,
        "query_mode": args.query_mode,
        "pc_plane": list(plane),
        "figures": [p.name for p in made],
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    legacy_write_readme(out, made, universality_root)

    print("[done]")
    for p in made:
        print("  ", p)
    print("  ", out / "headline_numbers.csv")
    print("  ", out / "README_FIRST.txt")


def main(argv=None):
    """Dispatch a workflow; each subcommand retains its original CLI options."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {'native': native_main, 'legacy': legacy_main}
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("\nCommands: " + ", ".join(commands))
        print("Use: python " + __file__ + " COMMAND --help")
        return
    command = argv.pop(0)
    if command not in commands:
        raise SystemExit("Unknown command: " + command)
    previous = sys.argv
    sys.argv = [previous[0] + " " + command, *argv]
    try:
        return commands[command]()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()

