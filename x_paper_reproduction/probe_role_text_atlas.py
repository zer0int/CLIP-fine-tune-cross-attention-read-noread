#!/usr/bin/env python3
r"""Post-process the compact RTA-100 mu2/head outputs into a role-vs-site atlas (v3).

NO GPU. NO datasets download. NO raw multi-GB tables required.

Reads only:
    condition_mean_mu2_curves.csv
    condition_mean_mu2_head_fits.csv
    paired_significance_population.csv
    paired_significance_mu2.csv          (optional)
    config.json                          (optional)

Writes:
    compact_summary_figures_role_text_atlas.csv
    role_site_text_atlas.html
    role_site_text_top_candidates.csv
    ROLE_SITE_TEXT_SUMMARY.txt
    compact_summary_figures_role_text_atlas.zip

Definitions
-----------
ROLE CHANGE:
    attack-vs-clean change in the head's response to the condition's OWN
    register population.

SITE CHANGE:
    attack-vs-clean change in the head's response involving the clean NoRTA
    register addresses frozen across the triplet.

For each policy, the response vector is 10-dimensional:
    five C->R slider values + five R->C slider values.

curve_change:
    pooled-scale normalized RMS distance. Captures magnitude + shape change.

shape_change:
    RMS distance after independently normalizing clean and attack response
    vectors. Emphasizes shape rather than absolute gain.

A head with:
    low ROLE CHANGE + high SITE CHANGE + positive TEXT recruitment
is an operational role-preserving / population-switching candidate.

The generated HTML is self-contained and offline:
    * click-to-sort columns
    * comparison / category / block filters
    * text search
    * checkbox collection
    * select-visible
    * selected-row copy box (TSV)
    * download-selected CSV
    * clickable role-vs-site scatter
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()


import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12
COMPARISONS = ("RTA", "SynthRTA")


# =============================================================================
# Helpers
# =============================================================================

def read_required(root: Path, name: str) -> pd.DataFrame:
    p = root / name
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def read_optional(root: Path, name: str) -> pd.DataFrame | None:
    p = root / name
    return pd.read_csv(p) if p.is_file() else None


def select_rn_mode(
    df: pd.DataFrame | None,
    rn_mode: str,
    *,
    table_name: str,
) -> pd.DataFrame | None:
    """Select one RN state from factorial tables while preserving legacy inputs.

    The role-vs-site atlas predates the RN off/on factorial extension of
    ``probe_rta_head_population.py``.  New tables therefore contain duplicate
    (condition, block, head) rows distinguished by ``rn_mode``.  The atlas must
    choose one state explicitly rather than silently mixing them.

    Legacy compact tables without an ``rn_mode`` column are interpreted as the
    original no-RN analysis and are therefore only compatible with ``rn_off``.
    """
    if df is None:
        return None
    if "rn_mode" not in df.columns:
        if rn_mode != "rn_off":
            raise ValueError(
                f"{table_name} is a legacy table without rn_mode; cannot select {rn_mode!r}. "
                "Use --rn-mode rn_off or regenerate the upstream RN-factorial outputs."
            )
        return df.copy()

    out = df[df["rn_mode"].astype(str) == rn_mode].copy()
    if out.empty:
        available = sorted(str(x) for x in df["rn_mode"].dropna().unique())
        raise ValueError(
            f"{table_name} has no rows for rn_mode={rn_mode!r}; available={available}"
        )
    return out


def f(x: Any, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def best_q(row: Mapping[str, Any]) -> float:
    w = f(row.get("wilcoxon_q"))
    t = f(row.get("t_q"))
    return w if np.isfinite(w) else t


def curve_distance(clean: np.ndarray, attack: np.ndarray) -> tuple[float, float, float]:
    clean = np.asarray(clean, np.float64)
    attack = np.asarray(attack, np.float64)

    pooled = max(
        float(np.max(np.abs(clean))),
        float(np.max(np.abs(attack))),
        EPS,
    )
    change = float(np.sqrt(np.mean(((attack - clean) / pooled) ** 2)))

    cscale = max(float(np.max(np.abs(clean))), EPS)
    ascale = max(float(np.max(np.abs(attack))), EPS)
    shape = float(
        np.sqrt(np.mean((clean / cscale - attack / ascale) ** 2))
    )
    return change, shape, pooled


def descriptor_distance(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.sqrt(np.mean((b - a) ** 2)))


def classify_motif(
    sC: float,
    sR: float,
    cC: float,
    cR: float,
    *,
    slope_threshold: float,
    curvature_threshold: float,
    insensitive_threshold: float,
    dominance_ratio: float,
) -> str:
    if max(abs(sC), abs(sR), abs(cC), abs(cR)) < insensitive_threshold:
        return "insensitive"

    if sC <= -slope_threshold and sR >= slope_threshold:
        return "push_pull_exchange"
    if sC >= slope_threshold and sR <= -slope_threshold:
        return "opposite_push_pull"

    if sC <= -slope_threshold and sR <= -slope_threshold:
        return "common_mode_attenuation"
    if sC >= slope_threshold and sR >= slope_threshold:
        return "common_mode_gain"

    if (
        abs(sR) >= slope_threshold
        and abs(sR) >= dominance_ratio * max(abs(sC), EPS)
    ):
        return "read_gain"

    if (
        abs(sC) >= slope_threshold
        and abs(sC) >= dominance_ratio * max(abs(sR), EPS)
    ):
        return "broadcast_gain"

    if (
        cC >= curvature_threshold
        and cC >= 0.85 * max(abs(sC), slope_threshold)
    ):
        return "source_null_curvature"

    if (
        cR <= -curvature_threshold
        and abs(cR) >= 0.85 * max(abs(sR), slope_threshold)
    ):
        return "read_optimum"

    return "mixed"


# =============================================================================
# Lookup tables
# =============================================================================

def curve_vector(
    curves: pd.DataFrame,
    condition: str,
    block: int,
    head: int,
    policy: str,
) -> np.ndarray:
    z = curves[
        (curves["condition"] == condition)
        & (curves["block"] == block)
        & (curves["head"] == head)
    ].sort_values("factor_mu2")

    if z.empty:
        raise KeyError((condition, block, head, policy))

    return np.r_[
        z[f"{policy}_C2R_write"].to_numpy(np.float64),
        z[f"{policy}_R2C_write"].to_numpy(np.float64),
    ]


def get_fit(
    fits: pd.DataFrame,
    condition: str,
    block: int,
    head: int,
) -> pd.Series:
    z = fits[
        (fits["condition"] == condition)
        & (fits["block"] == block)
        & (fits["head"] == head)
    ]
    if len(z) != 1:
        raise KeyError((condition, block, head))
    return z.iloc[0]


def stat_map(df: pd.DataFrame | None) -> dict[tuple[str, str, int, int], dict[str, float]]:
    out = {}
    if df is None:
        return out
    for row in df.to_dict("records"):
        key = (
            str(row["comparison"]),
            str(row["metric"]),
            int(row["block"]),
            int(row["head"]),
        )
        out[key] = {
            "mean_delta": f(row.get("mean_delta")),
            "median_delta": f(row.get("median_delta")),
            "cohen_dz": f(row.get("cohen_dz")),
            "q": best_q(row),
        }
    return out


def sg(
    smap: Mapping[tuple[str, str, int, int], Mapping[str, float]],
    comparison: str,
    metric: str,
    block: int,
    head: int,
    field: str,
) -> float:
    return f(
        smap.get((comparison, metric, block, head), {}).get(field)
    )


# =============================================================================
# Atlas
# =============================================================================

def build_atlas(
    curves: pd.DataFrame,
    fits: pd.DataFrame,
    popstats: pd.DataFrame,
    mu2stats: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    pop = stat_map(popstats)
    mu2 = stat_map(mu2stats)

    slope_thr = float(config.get("motif_slope_threshold", 0.10))
    curve_thr = float(config.get("motif_curvature_threshold", 0.12))
    ins_thr = float(config.get("motif_insensitive_threshold", 0.06))
    dom = float(config.get("motif_dominance_ratio", 1.8))

    blocks = sorted(int(x) for x in fits["block"].unique())
    heads = sorted(int(x) for x in fits["head"].unique())
    rows = []

    for comp in COMPARISONS:
        for block in blocks:
            for head in heads:
                clean = get_fit(fits, "NoRTA", block, head)
                attack = get_fit(fits, comp, block, head)

                cf = curve_vector(curves, "NoRTA", block, head, "frozen")
                af = curve_vector(curves, comp, block, head, "frozen")
                co = curve_vector(curves, "NoRTA", block, head, "own")
                ao = curve_vector(curves, comp, block, head, "own")

                site_change, site_shape, frozen_scale = curve_distance(cf, af)
                role_change, role_shape, own_scale = curve_distance(co, ao)

                frozen_clean = [
                    clean["frozen_C2R_slope_rel"],
                    clean["frozen_R2C_slope_rel"],
                    clean["frozen_C2R_curvature_rel"],
                    clean["frozen_R2C_curvature_rel"],
                ]
                frozen_attack = [
                    attack["frozen_C2R_slope_rel"],
                    attack["frozen_R2C_slope_rel"],
                    attack["frozen_C2R_curvature_rel"],
                    attack["frozen_R2C_curvature_rel"],
                ]
                own_clean = [
                    clean["own_C2R_slope_rel"],
                    clean["own_R2C_slope_rel"],
                    clean["own_C2R_curvature_rel"],
                    clean["own_R2C_curvature_rel"],
                ]
                own_attack = [
                    attack["own_C2R_slope_rel"],
                    attack["own_R2C_slope_rel"],
                    attack["own_C2R_curvature_rel"],
                    attack["own_R2C_curvature_rel"],
                ]

                motif_kwargs = dict(
                    slope_threshold=slope_thr,
                    curvature_threshold=curve_thr,
                    insensitive_threshold=ins_thr,
                    dominance_ratio=dom,
                )
                frozen_motif_clean = classify_motif(
                    *map(float, frozen_clean), **motif_kwargs
                )
                frozen_motif_attack = classify_motif(
                    *map(float, frozen_attack), **motif_kwargs
                )
                own_motif_clean = classify_motif(
                    *map(float, own_clean), **motif_kwargs
                )
                own_motif_attack = classify_motif(
                    *map(float, own_attack), **motif_kwargs
                )

                t2c_w = sg(
                    pop, comp, "TEXT_P2C_write_share",
                    block, head, "mean_delta"
                )
                c2t_w = sg(
                    pop, comp, "TEXT_C2P_write_share",
                    block, head, "mean_delta"
                )
                t2c_a = sg(
                    pop, comp, "TEXT_P2C_attn_share",
                    block, head, "mean_delta"
                )
                c2t_a = sg(
                    pop, comp, "TEXT_C2P_attn_share",
                    block, head, "mean_delta"
                )

                text_positive = float(
                    math.sqrt(
                        max(t2c_w, 0.0) ** 2
                        + max(c2t_w, 0.0) ** 2
                    )
                )

                row = {
                    "comparison": comp,
                    "block": block,
                    "head": head,
                    "BH": f"B{block:02d} H{head:02d}",

                    "role_change": role_change,
                    "site_change": site_change,
                    "site_minus_role": site_change - role_change,
                    "site_role_ratio": site_change / max(role_change, 1e-6),

                    "role_shape_change": role_shape,
                    "site_shape_change": site_shape,
                    # Total curve changes include gain + shape.
                    "role_change_total": role_change,
                    "site_change_total": site_change,

                    # Grammar change is the distance in the normalized local
                    # [s_C2R, s_R2C, c_C2R, c_R2C] response descriptor.
                    "role_grammar_change": descriptor_distance(
                        own_clean, own_attack
                    ),
                    "site_grammar_change": descriptor_distance(
                        frozen_clean, frozen_attack
                    ),

                    # Keep the earlier explicit names too, for compatibility.
                    "own_descriptor_change": descriptor_distance(
                        own_clean, own_attack
                    ),
                    "frozen_descriptor_change": descriptor_distance(
                        frozen_clean, frozen_attack
                    ),

                    "own_response_scale": own_scale,
                    "frozen_response_scale": frozen_scale,
                    "role_gain_change": abs(
                        float(attack["own_response_scale"])
                        - float(clean["own_response_scale"])
                    ) / max(
                        float(attack["own_response_scale"]),
                        float(clean["own_response_scale"]),
                        EPS,
                    ),
                    "site_gain_change": abs(
                        float(attack["frozen_response_scale"])
                        - float(clean["frozen_response_scale"])
                    ) / max(
                        float(attack["frozen_response_scale"]),
                        float(clean["frozen_response_scale"]),
                        EPS,
                    ),

                    "own_motif_clean": own_motif_clean,
                    "own_motif_attack": own_motif_attack,
                    "own_motif_changed": own_motif_clean != own_motif_attack,

                    "frozen_motif_clean": frozen_motif_clean,
                    "frozen_motif_attack": frozen_motif_attack,
                    "frozen_motif_changed": (
                        frozen_motif_clean != frozen_motif_attack
                    ),

                    "own_s_C2R_clean": float(own_clean[0]),
                    "own_s_C2R_attack": float(own_attack[0]),
                    "own_s_C2R_delta": float(own_attack[0] - own_clean[0]),
                    "own_s_R2C_clean": float(own_clean[1]),
                    "own_s_R2C_attack": float(own_attack[1]),
                    "own_s_R2C_delta": float(own_attack[1] - own_clean[1]),
                    "own_c_C2R_delta": float(own_attack[2] - own_clean[2]),
                    "own_c_R2C_delta": float(own_attack[3] - own_clean[3]),

                    "frozen_s_C2R_delta": float(
                        frozen_attack[0] - frozen_clean[0]
                    ),
                    "frozen_s_R2C_delta": float(
                        frozen_attack[1] - frozen_clean[1]
                    ),
                    "frozen_c_C2R_delta": float(
                        frozen_attack[2] - frozen_clean[2]
                    ),
                    "frozen_c_R2C_delta": float(
                        frozen_attack[3] - frozen_clean[3]
                    ),

                    "TEXT_to_CLS_write_share_delta": t2c_w,
                    "CLS_to_TEXT_write_share_delta": c2t_w,
                    "TEXT_to_CLS_attn_share_delta": t2c_a,
                    "CLS_to_TEXT_attn_share_delta": c2t_a,
                    "TEXT_positive_bidirectional_write": text_positive,

                    "TEXT_to_CLS_write_dz": sg(
                        pop, comp, "TEXT_P2C_write_share",
                        block, head, "cohen_dz"
                    ),
                    "CLS_to_TEXT_write_dz": sg(
                        pop, comp, "TEXT_C2P_write_share",
                        block, head, "cohen_dz"
                    ),
                    "TEXT_to_CLS_write_q": sg(
                        pop, comp, "TEXT_P2C_write_share",
                        block, head, "q"
                    ),
                    "CLS_to_TEXT_write_q": sg(
                        pop, comp, "TEXT_C2P_write_share",
                        block, head, "q"
                    ),

                    "REG_FROZEN_to_CLS_write_share_delta": sg(
                        pop, comp, "REG_FROZEN_P2C_write_share",
                        block, head, "mean_delta"
                    ),
                    "REG_OWN_to_CLS_write_share_delta": sg(
                        pop, comp, "REG_OWN_P2C_write_share",
                        block, head, "mean_delta"
                    ),
                    "MU1_CACHE_to_CLS_write_share_delta": sg(
                        pop, comp, "MU1_CACHE_P2C_write_share",
                        block, head, "mean_delta"
                    ),
                    "SCRATCH_RESIDUAL_to_CLS_write_share_delta": sg(
                        pop, comp, "SCRATCH_RESIDUAL_P2C_write_share",
                        block, head, "mean_delta"
                    ),

                    "own_s_C2R_paired_q": sg(
                        mu2, comp, "own_C2R_slope_rel",
                        block, head, "q"
                    ),
                    "own_s_R2C_paired_q": sg(
                        mu2, comp, "own_R2C_slope_rel",
                        block, head, "q"
                    ),
                    "frozen_s_C2R_paired_q": sg(
                        mu2, comp, "frozen_C2R_slope_rel",
                        block, head, "q"
                    ),
                    "frozen_s_R2C_paired_q": sg(
                        mu2, comp, "frozen_R2C_slope_rel",
                        block, head, "q"
                    ),
                }
                rows.append(row)

    atlas = pd.DataFrame(rows).sort_values(
        ["comparison", "block", "head"]
    ).reset_index(drop=True)

    atlas["role_change_pct"] = np.nan
    atlas["site_change_pct"] = np.nan
    atlas["role_grammar_change_pct"] = np.nan
    atlas["site_response_change_pct"] = np.nan
    atlas["text_recruitment_pct"] = np.nan
    atlas["population_switch_score"] = np.nan
    atlas["controller_remap_score"] = np.nan
    atlas["operational_class"] = ""

    for comp, idx in atlas.groupby("comparison").groups.items():
        ii = np.asarray(list(idx), int)
        z = atlas.loc[ii]

        role_total_pct = z["role_change_total"].rank(
            method="average", pct=True
        )
        site_total_pct = z["site_change_total"].rank(
            method="average", pct=True
        )
        role_grammar_pct = z["role_grammar_change"].rank(
            method="average", pct=True
        )
        text_pct = z["TEXT_positive_bidirectional_write"].rank(
            method="average", pct=True
        )

        # Backward-compatible percentile names refer to total response change.
        atlas.loc[ii, "role_change_pct"] = role_total_pct.to_numpy()
        atlas.loc[ii, "site_change_pct"] = site_total_pct.to_numpy()

        atlas.loc[ii, "role_grammar_change_pct"] = (
            role_grammar_pct.to_numpy()
        )
        atlas.loc[ii, "site_response_change_pct"] = (
            site_total_pct.to_numpy()
        )
        atlas.loc[ii, "text_recruitment_pct"] = text_pct.to_numpy()

        # Population-switch score deliberately asks for:
        #   high frozen-site response change
        #   low OWN-register response-GRAMMAR change
        #   positive TEXT recruitment
        atlas.loc[ii, "population_switch_score"] = (
            site_total_pct.to_numpy()
            * (1.0 - role_grammar_pct.to_numpy())
            * text_pct.to_numpy()
        )

        # Controller remap asks for actual OWN-register grammar change plus
        # a nontrivial total OWN-register response change.
        atlas.loc[ii, "controller_remap_score"] = (
            role_grammar_pct.to_numpy()
            * role_total_pct.to_numpy()
        )

        classes = []
        for rgp, rtp, sp, tp, txt in zip(
            role_grammar_pct.to_numpy(),
            role_total_pct.to_numpy(),
            site_total_pct.to_numpy(),
            text_pct.to_numpy(),
            z["TEXT_positive_bidirectional_write"].to_numpy(np.float64),
        ):
            if sp >= .75 and rgp <= .25 and txt > 0:
                cls = "role-preserving / population-switching"
            elif rgp >= .75 and rtp >= .50:
                cls = "controller-remap candidate"
            elif rgp <= .25 and sp <= .25:
                cls = "stable role grammar + stable sites"
            elif sp >= .75 and rgp <= .25:
                cls = "site shift, little role-grammar change"
            elif tp >= .90 and txt > 0:
                cls = "strong TEXT recruitment"
            else:
                cls = "mixed / intermediate"
            classes.append(cls)

        atlas.loc[ii, "operational_class"] = classes

    return atlas


# =============================================================================
# Offline HTML
# =============================================================================

DISPLAY_COLUMNS = [
    "comparison", "BH", "operational_class",
    "role_grammar_change", "site_change_total",
    "role_change_total", "site_minus_role", "site_role_ratio",
    "role_shape_change", "site_shape_change",
    "role_gain_change", "site_gain_change",
    "own_descriptor_change", "frozen_descriptor_change",
    "population_switch_score", "controller_remap_score",
    "own_motif_clean", "own_motif_attack",
    "frozen_motif_clean", "frozen_motif_attack",
    "TEXT_to_CLS_write_share_delta", "CLS_to_TEXT_write_share_delta",
    "TEXT_to_CLS_attn_share_delta", "CLS_to_TEXT_attn_share_delta",
    "TEXT_positive_bidirectional_write",
    "TEXT_to_CLS_write_dz", "CLS_to_TEXT_write_dz",
    "TEXT_to_CLS_write_q", "CLS_to_TEXT_write_q",
    "REG_FROZEN_to_CLS_write_share_delta",
    "REG_OWN_to_CLS_write_share_delta",
    "MU1_CACHE_to_CLS_write_share_delta",
    "SCRATCH_RESIDUAL_to_CLS_write_share_delta",
    "own_s_C2R_delta", "own_s_R2C_delta",
    "frozen_s_C2R_delta", "frozen_s_R2C_delta",
]


def json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for row in df.to_dict("records"):
        clean = {}
        for k, v in row.items():
            if pd.isna(v):
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_, bool)):
                clean[k] = bool(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


def build_html(df: pd.DataFrame, path: Path) -> None:
    data_json = json.dumps(json_records(df), separators=(",", ":"))
    cols_json = json.dumps(DISPLAY_COLUMNS)

    page = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>RTA-100 μ2 role-vs-site head atlas</title>
<style>
:root { color-scheme: dark; }
body { font-family:Consolas,ui-monospace,monospace; margin:0; background:#111; color:#ddd; }
header { padding:14px 18px 10px; position:sticky; top:0; background:#111; z-index:10; border-bottom:1px solid #333; }
h1 { font-size:20px; margin:0 0 7px; }
.small { font-size:12px; color:#aaa; }
.toolbar { display:flex; flex-wrap:wrap; gap:7px; margin-top:9px; align-items:center; }
input, select, button, textarea { background:#1b1b1b; color:#ddd; border:1px solid #555; border-radius:4px; padding:5px; font:inherit; }
button { cursor:pointer; } button:hover { background:#292929; }
#scatterWrap { padding:12px 18px 3px; }
#scatter { width:100%; max-width:1050px; height:430px; border:1px solid #333; background:#151515; }
#scatter text { fill:#aaa; font-size:11px; }
.pt { cursor:pointer; opacity:.78; } .pt:hover { stroke:white; stroke-width:2; opacity:1; }
.selectedPt { stroke:white; stroke-width:2.5; opacity:1; }
.tableWrap { overflow:auto; max-height:63vh; border-top:1px solid #333; }
table { border-collapse:collapse; width:max-content; min-width:100%; font-size:11px; }
thead th { position:sticky; top:0; background:#202020; z-index:4; border-bottom:1px solid #555; cursor:pointer; }
th,td { padding:5px 7px; border-right:1px solid #2d2d2d; white-space:nowrap; }
tbody tr:nth-child(even) { background:#171717; }
tbody tr:hover { background:#252525; }
tbody tr.rowSelected { background:#243244 !important; }
td.num { text-align:right; }
.copyArea { padding:12px 18px 24px; }
textarea { width:calc(100% - 16px); min-height:170px; resize:vertical; }
</style>
</head>
<body>
<header>
<h1>RTA-100 μ2 role-vs-site head atlas</h1>
<div class="small">
ROLE GRAMMAR = attack change in normalized slope/curvature response to current OWN registers.
SITE = attack change at clean NoRTA register addresses frozen across the pair.
Low ROLE-GRAMMAR change + high SITE response change + TEXT recruitment ≈ role-preserving / population-switching candidate.
Click headers to sort; click points/checkboxes to collect interesting goobers.
</div>
<div class="toolbar">
<input id="search" placeholder="search B06 H09 / motif / class" size="31">
<select id="comparison"><option value="">all comparisons</option><option>RTA</option><option>SynthRTA</option></select>
<select id="clazz"><option value="">all classes</option></select>
<label>B min <input id="bmin" type="number" value="0" min="0" max="23" style="width:48px"></label>
<label>B max <input id="bmax" type="number" value="23" min="0" max="23" style="width:48px"></label>
<label><input id="sigOnly" type="checkbox"> TEXT→CLS q&lt;.05</label>
<button id="selectVisible">select visible</button>
<button id="clearSelected">clear selected</button>
<button id="copySelected">build copy box</button>
<button id="downloadSelected">download selected CSV</button>
<span id="status" class="small"></span>
</div>
</header>

<div id="scatterWrap">
<div class="small">
x = within-comparison ROLE-GRAMMAR percentile · y = SITE-RESPONSE percentile ·
point size = positive bidirectional TEXT-write recruitment.
Dashed guides mark x=0.25 and y=0.75; the upper-left corner is the operational
role-preserving / population-switching region.
</div>
<svg id="scatter" viewBox="0 0 1000 430"></svg>
</div>

<div class="tableWrap">
<table id="tbl"><thead><tr id="headrow"><th>pick</th></tr></thead><tbody></tbody></table>
</div>

<div class="copyArea">
<div class="small">Selected rows as TSV, ready to scoop into chat:</div>
<textarea id="copybox" spellcheck="false"></textarea>
</div>

<script>
const DATA=__DATA__;
const COLS=__COLS__;
let sortCol="population_switch_score", sortAsc=false;
const selected=new Set();
const key=r=>`${r.comparison}|${r.block}|${r.head}`;

function fmt(v){
  if(v===null||v===undefined) return "";
  if(typeof v==="number"){
    const a=Math.abs(v);
    if(a!==0&&(a<0.001||a>=1000)) return v.toExponential(3);
    return v.toFixed(4);
  }
  return String(v);
}
function init(){
  const hr=document.getElementById("headrow");
  COLS.forEach(c=>{
    const th=document.createElement("th"); th.textContent=c;
    th.onclick=()=>{ if(sortCol===c) sortAsc=!sortAsc; else {sortCol=c; sortAsc=false;} render(); };
    hr.appendChild(th);
  });
  const classes=[...new Set(DATA.map(r=>r.operational_class))].sort();
  const s=document.getElementById("clazz");
  classes.forEach(c=>{const o=document.createElement("option");o.value=c;o.textContent=c;s.appendChild(o);});
}
function filtered(){
  const q=document.getElementById("search").value.toLowerCase().trim();
  const comp=document.getElementById("comparison").value;
  const clazz=document.getElementById("clazz").value;
  const bmin=+document.getElementById("bmin").value, bmax=+document.getElementById("bmax").value;
  const sig=document.getElementById("sigOnly").checked;
  const rows=DATA.filter(r=>{
    if(comp&&r.comparison!==comp) return false;
    if(clazz&&r.operational_class!==clazz) return false;
    if(r.block<bmin||r.block>bmax) return false;
    if(sig&&!(r.TEXT_to_CLS_write_q!==null&&r.TEXT_to_CLS_write_q<0.05)) return false;
    if(q){
      const blob=`${r.comparison} ${r.BH} ${r.operational_class} ${r.own_motif_clean} ${r.own_motif_attack} ${r.frozen_motif_clean} ${r.frozen_motif_attack}`.toLowerCase();
      if(!blob.includes(q)) return false;
    }
    return true;
  });
  rows.sort((a,b)=>{
    const av=a[sortCol], bv=b[sortCol];
    if(av===null&&bv===null) return 0; if(av===null) return 1; if(bv===null) return -1;
    const d=(typeof av==="number"&&typeof bv==="number") ? av-bv : String(av).localeCompare(String(bv));
    return sortAsc?d:-d;
  });
  return rows;
}
function table(rows){
  const body=document.querySelector("#tbl tbody"); body.innerHTML="";
  rows.forEach(r=>{
    const tr=document.createElement("tr"); tr.id=`row-${r.comparison}-${r.block}-${r.head}`;
    if(selected.has(key(r))) tr.classList.add("rowSelected");
    const td0=document.createElement("td"), cb=document.createElement("input");
    cb.type="checkbox"; cb.checked=selected.has(key(r));
    cb.onchange=()=>{cb.checked?selected.add(key(r)):selected.delete(key(r));render();};
    td0.appendChild(cb); tr.appendChild(td0);
    COLS.forEach(c=>{
      const td=document.createElement("td"); td.textContent=fmt(r[c]);
      if(typeof r[c]==="number") td.classList.add("num"); tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}
function scatter(rows){
  const svg=document.getElementById("scatter"); svg.innerHTML="";
  const V=rows.filter(r=>Number.isFinite(r.role_grammar_change_pct)&&Number.isFinite(r.site_response_change_pct));
  if(!V.length) return;
  const W=1000,H=430,L=65,R=25,T=20,B=48;
  const xmax=1.0, ymax=1.0;
  const tmax=Math.max(...V.map(r=>Math.max(r.TEXT_positive_bidirectional_write||0,0)))||1;
  const sx=x=>L+(W-L-R)*x/xmax, sy=y=>H-B-(H-T-B)*y/ymax;
  function ln(x1,y1,x2,y2,stroke="#444"){
    const e=document.createElementNS("http://www.w3.org/2000/svg","line");
    [["x1",x1],["y1",y1],["x2",x2],["y2",y2],["stroke",stroke]].forEach(([k,v])=>e.setAttribute(k,v));svg.appendChild(e);
  }
  ln(L,H-B,W-R,H-B);ln(L,T,L,H-B);

  // Operational guide region: low role-grammar rank, high site-response rank.
  const guide=document.createElementNS("http://www.w3.org/2000/svg","rect");
  guide.setAttribute("x",sx(0));
  guide.setAttribute("y",sy(1.0));
  guide.setAttribute("width",sx(0.25)-sx(0));
  guide.setAttribute("height",sy(0.75)-sy(1.0));
  guide.setAttribute("fill","none");
  guide.setAttribute("stroke","#777");
  guide.setAttribute("stroke-dasharray","5,4");
  svg.appendChild(guide);
  ln(sx(0.25),T,sx(0.25),H-B,"#555");
  ln(L,sy(0.75),W-R,sy(0.75),"#555");
  function label(x,y,txt,rot){
    const e=document.createElementNS("http://www.w3.org/2000/svg","text");
    e.setAttribute("x",x);e.setAttribute("y",y);e.setAttribute("text-anchor","middle");
    if(rot)e.setAttribute("transform",`rotate(${rot} ${x} ${y})`);e.textContent=txt;svg.appendChild(e);
  }
  label((L+W-R)/2,H-10,"ROLE-GRAMMAR CHANGE percentile (own registers)",0);
  label(14,(T+H-B)/2,"SITE-RESPONSE CHANGE percentile (frozen clean addresses)",-90);
  V.forEach(r=>{
    const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
    const t=Math.max(r.TEXT_positive_bidirectional_write||0,0)/tmax;
    c.setAttribute("cx",sx(r.role_grammar_change_pct));c.setAttribute("cy",sy(r.site_response_change_pct));
    c.setAttribute("r",3+8*Math.sqrt(t));
    c.setAttribute("fill",r.comparison==="RTA"?"#5da5da":"#f17cb0");
    c.setAttribute("class","pt"+(selected.has(key(r))?" selectedPt":""));
    const title=document.createElementNS("http://www.w3.org/2000/svg","title");
    title.textContent=`${r.comparison} ${r.BH}\n${r.operational_class}\nroleGrammar=${fmt(r.role_grammar_change)} roleTotal=${fmt(r.role_change_total)} site=${fmt(r.site_change_total)}\nTEXT→CLS=${fmt(r.TEXT_to_CLS_write_share_delta)} CLS→TEXT=${fmt(r.CLS_to_TEXT_write_share_delta)}`;
    c.appendChild(title);
    c.onclick=()=>{selected.has(key(r))?selected.delete(key(r)):selected.add(key(r));render();setTimeout(()=>{const e=document.getElementById(`row-${r.comparison}-${r.block}-${r.head}`);if(e)e.scrollIntoView({behavior:"smooth",block:"center"});},25);};
    svg.appendChild(c);
  });
}
function render(){
  const rows=filtered();table(rows);scatter(rows);
  document.getElementById("status").textContent=`visible ${rows.length}/${DATA.length} · selected ${selected.size}`;
}
function buildBox(){
  const rows=DATA.filter(r=>selected.has(key(r))).sort((a,b)=>a.comparison.localeCompare(b.comparison)||a.block-b.block||a.head-b.head);
  const cc=["comparison","BH","operational_class","role_grammar_change","role_change_total","site_change_total","site_minus_role","site_role_ratio","population_switch_score","controller_remap_score","own_motif_clean","own_motif_attack","frozen_motif_clean","frozen_motif_attack","TEXT_to_CLS_write_share_delta","CLS_to_TEXT_write_share_delta","TEXT_to_CLS_attn_share_delta","CLS_to_TEXT_attn_share_delta","TEXT_to_CLS_write_dz","TEXT_to_CLS_write_q","own_s_C2R_delta","own_s_R2C_delta","frozen_s_C2R_delta","frozen_s_R2C_delta"];
  const lines=[cc.join("\t")]; rows.forEach(r=>lines.push(cc.map(c=>fmt(r[c])).join("\t")));
  const b=document.getElementById("copybox");b.value=lines.join("\n");b.focus();b.select();
}
function download(){
  const rows=DATA.filter(r=>selected.has(key(r)));
  if(!rows.length)return;const cols=Object.keys(rows[0]);
  const esc=v=>{if(v===null||v===undefined)return "";const s=String(v);return /[",\n]/.test(s)?`"${s.replaceAll('"','""')}"`:s;};
  const csv=[cols.join(",")].concat(rows.map(r=>cols.map(c=>esc(r[c])).join(","))).join("\n");
  const blob=new Blob([csv],{type:"text/csv"}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="selected_role_site_heads.csv";a.click();URL.revokeObjectURL(a.href);
}
["search","comparison","clazz","bmin","bmax","sigOnly"].forEach(id=>document.getElementById(id).addEventListener(id==="search"?"input":"change",render));
document.getElementById("selectVisible").onclick=()=>{filtered().forEach(r=>selected.add(key(r)));render();};
document.getElementById("clearSelected").onclick=()=>{selected.clear();render();};
document.getElementById("copySelected").onclick=buildBox;
document.getElementById("downloadSelected").onclick=download;
init();render();
</script>
</body>
</html>
"""
    page = page.replace("__DATA__", data_json).replace("__COLS__", cols_json)
    path.write_text(page, encoding="utf-8")


# =============================================================================
# Summary + CLI
# =============================================================================

def write_summary(atlas: pd.DataFrame, out: Path) -> None:
    lines = [
        "RTA-100 ROLE-vs-SITE + TEXT HEAD ATLAS",
        "=" * 78,
        "",
        "Low role_grammar_change + high site_change_total + positive TEXT recruitment",
        "  -> operational role-preserving / population-switching candidate.",
        "High role_change",
        "  -> candidate for actual current-register controller remapping.",
        "",
    ]
    for comp in COMPARISONS:
        z = atlas[atlas["comparison"] == comp]
        lines += [f"[{comp}]", "", "Top population-switch candidates:"]
        for r in z.sort_values("population_switch_score", ascending=False).head(15).itertuples(index=False):
            lines.append(
                f"  {r.BH:<8} score={r.population_switch_score:.3f} "
                f"roleGrammar={r.role_grammar_change:.4f} roleTotal={r.role_change_total:.4f} site={r.site_change_total:.4f} "
                f"T->C={r.TEXT_to_CLS_write_share_delta:+.4f} "
                f"C->T={r.CLS_to_TEXT_write_share_delta:+.4f}"
            )
        lines += ["", "Top controller-remap candidates:"]
        for r in z.sort_values("controller_remap_score", ascending=False).head(12).itertuples(index=False):
            lines.append(
                f"  {r.BH:<8} score={r.controller_remap_score:.3f} "
                f"roleGrammar={r.role_grammar_change:.4f} roleTotal={r.role_change_total:.4f} site={r.site_change_total:.4f} "
                f"own={r.own_motif_clean}->{r.own_motif_attack}"
            )
        lines.append("")
    (out / "ROLE_SITE_TEXT_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=r"rta100_mu2_head_population_grammar",
    )
    p.add_argument("--out_dir", default=None)
    p.add_argument(
        "--rn-mode",
        choices=("rn_off", "rn_on"),
        default="rn_off",
        help=(
            "RN state to atlas when the upstream head-population tables contain the "
            "RN off/on factorial. Default rn_off preserves the original native/no-RN "
            "role-vs-site atlas semantics. Legacy tables without rn_mode are accepted "
            "only for rn_off."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out_dir) if args.out_dir else root / "role_site_text_atlas"
    out.mkdir(parents=True, exist_ok=True)

    curves = read_required(root, "condition_mean_mu2_curves.csv")
    fits = read_required(root, "condition_mean_mu2_head_fits.csv")
    popstats = read_required(root, "paired_significance_population.csv")
    mu2stats = read_optional(root, "paired_significance_mu2.csv")

    curves = select_rn_mode(curves, args.rn_mode, table_name="condition_mean_mu2_curves.csv")
    fits = select_rn_mode(fits, args.rn_mode, table_name="condition_mean_mu2_head_fits.csv")
    popstats = select_rn_mode(popstats, args.rn_mode, table_name="paired_significance_population.csv")
    mu2stats = select_rn_mode(mu2stats, args.rn_mode, table_name="paired_significance_mu2.csv")
    print(f"[rn mode] {args.rn_mode}")

    config = {}
    cfg = root / "config.json"
    if cfg.is_file():
        try:
            config = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            config = {}

    atlas = build_atlas(curves, fits, popstats, mu2stats, config)
    atlas.insert(0, "rn_mode", args.rn_mode)

    csv_path = out / "compact_summary_figures_role_text_atlas.csv"
    atlas.to_csv(csv_path, index=False)

    top = (
        atlas.sort_values(
            ["comparison", "population_switch_score"],
            ascending=[True, False],
        )
        .groupby("comparison", as_index=False)
        .head(80)
        .reset_index(drop=True)
    )
    top.to_csv(out / "role_site_text_top_candidates.csv", index=False)

    html_path = out / "role_site_text_atlas.html"
    build_html(atlas, html_path)
    write_summary(atlas, out)

    zip_path = out / "compact_summary_figures_role_text_atlas.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in (
            csv_path,
            out / "role_site_text_top_candidates.csv",
            html_path,
            out / "ROLE_SITE_TEXT_SUMMARY.txt",
        ):
            z.write(p, arcname=p.name)

    print("[done]")
    print("CSV to send me:", csv_path)
    print("HTML for human gazing:", html_path)
    print("compact ZIP:", zip_path)


if __name__ == "__main__":
    main()
