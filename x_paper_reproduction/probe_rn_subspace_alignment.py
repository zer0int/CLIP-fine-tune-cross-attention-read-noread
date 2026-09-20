#!/usr/bin/env python3
r"""Measure alignment between the current model's B13 RN control basis and its
intact register invariant subspace across downstream ViT blocks.

Question
--------
Does the RN-control PC that is unusually fragile under register-pump ablation
(especially PC4) overlap more strongly with the register invariant subspace than
the comparatively stable PC2?

Definitions
-----------
* U_RN: uncentered rank-4 PCA basis of the RN-induced ordinary-token B13 delta,
  using the SAME NoRTA + multilingual SynthRTA fitting policy as the native
  control-surface runner.
* U_reg(B): uncentered rank-2 SVD basis of per-image mean register states at
  block B, using intact B12 register identities frozen as spatial addresses and
  followed forward through the natural RN run.

All vectors live in the same ViT residual-stream feature coordinates (D=1024).
The script reports:
  - |cos(RN PC_i, register mu_j(B))|, i=1..4, j=1..2
  - fraction of each RN PC lying in the register rank-2 subspace
  - fraction of each register mu_j lying in the RN rank-4 subspace
  - two principal cosines between U_RN(rank4) and U_reg(B)(rank2)
  - the explicit PC4-minus-PC2 register-subspace overlap contrast.

Outputs
-------
  rn_pc_register_alignment_long.csv
  rn_pc_register_alignment_by_block.csv
  rn_pc_register_bases.npz
  rn_pc_register_subspace_alignment.png
  rn_pc_register_mu_abs_cos_heatmap.png
  SUMMARY.txt

Required beside this script
---------------------------
  probe_tools_rn_control.py
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
from probe_tools_analysis import (parse_strs, unit_rows)


import argparse
import gc
import random
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import probe_tools_rn_control as base


DEFAULT_DATASET_REPO = "zer0int/RTA-100-Multilingual"
DEFAULT_LANGUAGES = "en,de,es,fr,ar,ko,zh,ru"
DEFAULT_PAIRS_PER_LANGUAGE = 8
DEFAULT_BLOCKS = "13-21"
DEFAULT_SEED = 20260902
DEFAULT_OUT = "rn_pc_register_subspace_alignment"


def parse_blocks(text: str) -> list[int]:
    out: list[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(tok))
    return sorted(set(out))


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def fit_uncentered_basis(x_nd: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal right-singular directions [rank,D] and singular values."""
    x = np.asarray(x_nd, np.float64)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(int(rank), vt.shape[0])
    basis = vt[:k].copy()
    # Deterministic sign convention: orient each direction toward the sample mean.
    mean = x.mean(axis=0)
    for i in range(k):
        if float(np.dot(basis[i], mean)) < 0:
            basis[i] *= -1.0
    return basis, s[:k]


def collect_register_means(
    model: torch.nn.Module,
    pair: Any,
    blocks_to_collect: Sequence[int],
) -> dict[int, np.ndarray]:
    """
    Use pair.b12_register_mask as fixed intact register addresses, and follow those
    same spatial positions through the natural RN forward.
    """
    requested = set(map(int, blocks_to_collect))
    insert_block = int(model.visual.read_null_insert_block)
    if min(requested) < insert_block:
        raise ValueError(
            f"Requested block before RN insertion: insert_block={insert_block}, blocks={sorted(requested)}"
        )

    mask = pair.b12_register_mask[0].detach().bool().reshape(-1).to(pair.full_post_tbc.device)
    if int(mask.sum()) == 0:
        raise RuntimeError("B12 fixed register mask is empty")

    out: dict[int, np.ndarray] = {}
    x = pair.full_post_tbc  # exact natural post-insertion-block state, T,B,D

    def record(block: int, state_tbd: torch.Tensor) -> None:
        spatial = state_tbd[1:-1, 0, :].float()
        if spatial.shape[0] != mask.numel():
            raise RuntimeError(
                f"B{block}: spatial token count {spatial.shape[0]} != mask length {mask.numel()}"
            )
        reg_mean = spatial[mask].mean(dim=0)
        out[int(block)] = reg_mean.detach().cpu().numpy().astype(np.float32)

    if insert_block in requested:
        record(insert_block, x)

    with torch.no_grad(), base.model_autocast_context(model):
        for b in range(insert_block + 1, len(model.visual.transformer.resblocks)):
            x = model.visual.transformer.resblocks[b](x)
            if b in requested:
                record(b, x)
            if b >= max(requested):
                break
    return out


def plot_alignment(
    block_rows: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
    out: Path,
) -> None:
    df = pd.DataFrame(block_rows).sort_values("block")
    long = pd.DataFrame(long_rows)
    blocks = df["block"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), dpi=180)

    # A. RN PC projection into register rank-2.
    for pc in range(1, 5):
        q = long[long["rn_pc"] == pc].sort_values("block")
        axes[0, 0].plot(
            q["block"], q["rn_pc_register_rank2_fraction"],
            marker="o", label=f"RN PC{pc}",
        )
    axes[0, 0].set_title("RN PC overlap with register rank-2 subspace")
    axes[0, 0].set_ylabel(r"$\|P_{\mathrm{reg},2} u_i^{\mathrm{RN}}\|_2^2$")
    axes[0, 0].legend()

    # B. Explicit PC2 vs PC4 hypothesis.
    axes[0, 1].plot(
        blocks, df["rn_pc2_register_rank2_fraction"],
        marker="o", label="RN PC2",
    )
    axes[0, 1].plot(
        blocks, df["rn_pc4_register_rank2_fraction"],
        marker="o", label="RN PC4",
    )
    axes[0, 1].plot(
        blocks, df["pc4_minus_pc2_register_rank2_fraction"],
        marker="x", linestyle="--", label="PC4 - PC2",
    )
    axes[0, 1].axhline(0.0, linewidth=0.8)
    axes[0, 1].set_title("Register-overlap contrast: PC4 vs PC2")
    axes[0, 1].set_ylabel("projection fraction / difference")
    axes[0, 1].legend()

    # C. Principal angles between RN rank-4 and register rank-2.
    axes[1, 0].plot(
        blocks, df["principal_cosine_1"],
        marker="o", label="principal cosine 1",
    )
    axes[1, 0].plot(
        blocks, df["principal_cosine_2"],
        marker="o", label="principal cosine 2",
    )
    axes[1, 0].set_title("RN rank-4 vs register rank-2 subspace")
    axes[1, 0].set_ylabel("principal cosine")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].legend()

    # D. Register directions captured by RN rank-4.
    axes[1, 1].plot(
        blocks, df["mu1_in_rn_rank4_fraction"],
        marker="o", label=r"$\mu_1$ in RN rank-4",
    )
    axes[1, 1].plot(
        blocks, df["mu2_in_rn_rank4_fraction"],
        marker="o", label=r"$\mu_2$ in RN rank-4",
    )
    axes[1, 1].set_title("Register directions represented in RN control subspace")
    axes[1, 1].set_ylabel("projection fraction")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("ViT block")
        ax.set_xticks(blocks)
        ax.grid(alpha=0.25)

    fig.suptitle("Current-checkpoint RN control basis vs intact register invariant subspace", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_heatmap(long_rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(long_rows)
    blocks = sorted(df["block"].unique())
    labels = []
    rows = []
    for pc in range(1, 5):
        for mu in (1, 2):
            labels.append(f"RN PC{pc} × μ{mu}")
            vals = []
            for b in blocks:
                q = df[(df["block"] == b) & (df["rn_pc"] == pc)]
                vals.append(float(q[f"abs_cos_mu{mu}"].iloc[0]))
            rows.append(vals)
    arr = np.asarray(rows, np.float64)

    fig, ax = plt.subplots(figsize=(10.6, 6.0), dpi=180)
    im = ax.imshow(arr, aspect="auto", origin="upper", vmin=0.0, vmax=max(0.25, float(np.nanmax(arr))))
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([f"B{b}" for b in blocks])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Absolute cosine: RN control PCs vs intact register directions")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("|cosine|")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure current-checkpoint RN-PC overlap with intact register invariant subspace."
    )
    ap.add_argument("--checkpoint", default=base.DEFAULT_CHECKPOINT)
    ap.add_argument("--module-root", default=".")
    ap.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    ap.add_argument("--dataset-root", default="")
    ap.add_argument("--languages", default=DEFAULT_LANGUAGES)
    ap.add_argument("--pairs-per-language", type=int, default=DEFAULT_PAIRS_PER_LANGUAGE)
    ap.add_argument("--blocks", default=DEFAULT_BLOCKS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    languages = parse_strs(args.languages)
    blocks = parse_blocks(args.blocks)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    loaded = base.load_model(
        args.checkpoint,
        package_root=args.module_root,
        device=args.device,
    )
    model = loaded.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = loaded.device

    insert_block = int(model.visual.read_null_insert_block)
    if insert_block not in blocks:
        print(f"[note] RN insertion block is B{insert_block}; requested blocks={blocks}")
    if min(blocks) < insert_block:
        raise ValueError(
            f"Blocks before RN insertion are not meaningful for this comparison: insert=B{insert_block}, blocks={blocks}"
        )

    capture_list = [int(x) for x in model.read_implant.capture_block_list()]
    early_needed = {b for b in capture_list if b < insert_block}

    norta, attacks, norta_idx, attack_idx, shared_keys = base.prepare_datasets(
        args.dataset_repo,
        args.dataset_root,
        languages,
        0,
        args.seed,
    )
    if args.pairs_per_language <= 0:
        basis_keys = list(shared_keys)
    else:
        basis_keys = list(shared_keys)[: min(len(shared_keys), args.pairs_per_language)]

    rn_delta_rows: list[torch.Tensor] = []
    register_means: dict[int, list[np.ndarray]] = {b: [] for b in blocks}
    sample_audit: list[dict[str, Any]] = []

    def process_one(key: str, condition: str, language: str, image_row: Any) -> None:
        pil = base.ensure_pil(image_row["image"])
        img = base.preprocess_pil(loaded.preprocess, pil, device)
        pair = base.build_b13_pair(model, img, key, condition, early_needed)

        rn_delta_rows.append(pair.delta_ord_btd[0].detach().float().cpu())
        reg = collect_register_means(model, pair, blocks)
        for b in blocks:
            if b not in reg:
                raise RuntimeError(f"Missing register mean for B{b}")
            register_means[b].append(reg[b])

        sample_audit.append({
            "sample_key": key,
            "condition": condition,
            "language": language,
            "fixed_b12_register_count": int(pair.b12_register_mask[0].sum().item()),
            "rn_delta_token_count": int(pair.delta_ord_btd.shape[1]),
            "feature_dim": int(pair.delta_ord_btd.shape[2]),
        })
        del pair, img, pil

    total = len(basis_keys) * (1 + len(languages))
    with tqdm(total=total, desc="RN/register basis bank", unit="image") as bar:
        for key in basis_keys:
            process_one(key, "norta", "shared", base.get_row(norta, norta_idx[key]))
            bar.update(1)
        for lang in languages:
            for key in basis_keys:
                process_one(
                    key, "synth", lang,
                    base.get_row(attacks[lang], attack_idx[lang][key]),
                )
                bar.update(1)

    # Fit the exact style of uncentered RN rank-4 feature basis used by the surface runner.
    x_cpu = torch.cat(rn_delta_rows, dim=0).float()
    x = x_cpu.to(device)
    q = min(12, x.shape[0], x.shape[1])
    _u, sing, vec = torch.pca_lowrank(x, q=q, center=False, niter=6)
    rn_basis = vec[:, :4].T.contiguous().float()
    rn_basis_np = rn_basis.detach().cpu().numpy().astype(np.float64)
    rn_basis_np = unit_rows(rn_basis_np)
    total_energy = float(x.square().sum().detach().cpu())
    rn_explained = (sing[:4].square() / max(total_energy, 1e-12)).detach().cpu().numpy()

    long_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    reg_basis_store: dict[int, np.ndarray] = {}
    reg_singular_store: dict[int, np.ndarray] = {}

    for b in blocks:
        X = np.stack(register_means[b], axis=0).astype(np.float64)
        reg_basis, reg_s = fit_uncentered_basis(X, rank=2)
        reg_basis = unit_rows(reg_basis)
        reg_basis_store[b] = reg_basis.astype(np.float32)
        reg_singular_store[b] = reg_s.astype(np.float32)

        cross = rn_basis_np @ reg_basis.T  # [4,2]
        principal = np.linalg.svd(cross, compute_uv=False)
        rn_proj = np.sum(cross * cross, axis=1)       # each RN PC into reg rank-2
        mu_proj = np.sum(cross * cross, axis=0)       # each reg mu into RN rank-4

        for pc in range(4):
            long_rows.append({
                "block": b,
                "rn_pc": pc + 1,
                "abs_cos_mu1": float(abs(cross[pc, 0])),
                "abs_cos_mu2": float(abs(cross[pc, 1])),
                "signed_cos_mu1": float(cross[pc, 0]),
                "signed_cos_mu2": float(cross[pc, 1]),
                "rn_pc_register_rank2_fraction": float(rn_proj[pc]),
                "rn_pc_explained_energy_at_B13_fit": float(rn_explained[pc]),
                "n_register_mean_samples": int(X.shape[0]),
            })

        block_rows.append({
            "block": b,
            "principal_cosine_1": float(principal[0]),
            "principal_cosine_2": float(principal[1]) if len(principal) > 1 else float("nan"),
            "rn_pc1_register_rank2_fraction": float(rn_proj[0]),
            "rn_pc2_register_rank2_fraction": float(rn_proj[1]),
            "rn_pc3_register_rank2_fraction": float(rn_proj[2]),
            "rn_pc4_register_rank2_fraction": float(rn_proj[3]),
            "pc4_minus_pc2_register_rank2_fraction": float(rn_proj[3] - rn_proj[1]),
            "mu1_in_rn_rank4_fraction": float(mu_proj[0]),
            "mu2_in_rn_rank4_fraction": float(mu_proj[1]),
            "register_pc1_energy_fraction": float(reg_s[0]**2 / np.sum(reg_s**2)) if len(reg_s) else float("nan"),
            "register_rank2_energy_fraction": float(np.sum(reg_s[:2]**2) / np.sum(reg_s**2)) if len(reg_s) else float("nan"),
            "n_register_mean_samples": int(X.shape[0]),
        })

    save_csv(out / "rn_pc_register_alignment_long.csv", long_rows)
    save_csv(out / "rn_pc_register_alignment_by_block.csv", block_rows)
    save_csv(out / "sample_audit.csv", sample_audit)

    npz_payload: dict[str, np.ndarray] = {
        "rn_rank4_basis": rn_basis_np.astype(np.float32),
        "rn_singular_values": sing[:4].detach().cpu().numpy().astype(np.float32),
        "rn_explained_energy": rn_explained.astype(np.float32),
    }
    for b in blocks:
        npz_payload[f"register_rank2_basis_B{b}"] = reg_basis_store[b]
        npz_payload[f"register_singular_values_B{b}"] = reg_singular_store[b]
    np.savez_compressed(out / "rn_pc_register_bases.npz", **npz_payload)

    plot_alignment(
        block_rows, long_rows,
        out / "rn_pc_register_subspace_alignment.png",
    )
    plot_cosine_heatmap(
        long_rows,
        out / "rn_pc_register_mu_abs_cos_heatmap.png",
    )

    bdf = pd.DataFrame(block_rows)
    peak_pc4 = bdf.iloc[int(np.nanargmax(bdf["rn_pc4_register_rank2_fraction"].to_numpy()))]
    peak_diff = bdf.iloc[int(np.nanargmax(bdf["pc4_minus_pc2_register_rank2_fraction"].to_numpy()))]
    lines = [
        "RN CONTROL PCs vs INTACT REGISTER INVARIANT SUBSPACE",
        "=" * 72,
        "",
        f"checkpoint: {args.checkpoint}",
        f"RN insertion block: B{insert_block}",
        f"blocks: {blocks}",
        f"languages: {languages}",
        f"basis keys per language/NoRTA: {len(basis_keys)}",
        f"basis images total: {len(sample_audit)}",
        "",
        "RN rank-4 explained energy fractions:",
        "  " + ", ".join(f"PC{i+1}={rn_explained[i]:.6f}" for i in range(4)),
        "",
        "Hypothesis diagnostic:",
        f"  max PC4 register-rank2 overlap at B{int(peak_pc4['block'])}: "
        f"{float(peak_pc4['rn_pc4_register_rank2_fraction']):.6f}",
        f"  max (PC4-PC2) overlap contrast at B{int(peak_diff['block'])}: "
        f"{float(peak_diff['pc4_minus_pc2_register_rank2_fraction']):+.6f}",
        "",
        "Interpretation guardrail:",
        "  Feature-space overlap does not by itself establish causal mediation.",
        "  A positive PC4>PC2 result would provide a geometric bridge between the",
        "  register-pump sensitivity of PC4 READ fields and the GIPU register",
        "  invariant subspace; mediation would still require intervention/path tests.",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done]", out.resolve())
    print("  figure:", (out / "rn_pc_register_subspace_alignment.png").resolve())
    print("  heatmap:", (out / "rn_pc_register_mu_abs_cos_heatmap.png").resolve())
    print("  data:", (out / "rn_pc_register_alignment_by_block.csv").resolve())

    del model, loaded, x, x_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
