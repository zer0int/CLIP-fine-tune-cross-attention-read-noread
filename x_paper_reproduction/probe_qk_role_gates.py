#!/usr/bin/env python3
r"""Late Q/K role-plane gates. Commands: scan (select high-gain heads), same_heads (fixed B22 head control).
"""
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()
import probe_tools_backbone as _backbone_tools
from probe_tools_analysis import (clear_attn_cache, select_register_mask)

# SCAN
import argparse
import gc
import json
import math
import random
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

MODEL_ORDER = ("pretrained", "gmp", "finetune_stripped")
DEFAULT_ORACLE_ROOT = r"cls_gipu_exchange_no_rn"
DEFAULT_OUT = r"late_qk_role_plane_heads"
DEFAULT_XATTN_CHECKPOINT = "REPLACE_WITH_CHECKPOINT.pt"
EPS = 1e-12


# =============================================================================
# helpers
# =============================================================================

def parse_ints(text: str) -> list[int]:
    out: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(token))
    return sorted(set(out))


def resolve_local(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    alt = Path(__file__).resolve().parent / path_text
    return alt if alt.exists() else path


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_qkv_shape(t: torch.Tensor, batch: int, heads: int, tokens: int) -> torch.Tensor:
    if t is None:
        raise RuntimeError("Q/K/V capture missing")
    if t.ndim != 4:
        raise RuntimeError(f"Expected 4D Q/K/V cache, got {tuple(t.shape)}")
    if t.shape[0] == batch and t.shape[1] == heads and t.shape[2] == tokens:
        return t
    if t.shape[0] == heads and t.shape[1] == batch and t.shape[2] == tokens:
        return t.permute(1, 0, 2, 3)
    if t.shape[0] == tokens and t.shape[1] == batch and t.shape[2] == heads:
        return t.permute(1, 2, 0, 3)
    raise RuntimeError(f"Cannot normalize Q/K/V cache shape {tuple(t.shape)} for B={batch},H={heads},T={tokens}")


def project_to_mu_basis(x_bd: torch.Tensor, mu1_d: torch.Tensor, mu2_d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu1_coef = x_bd.float() @ mu1_d.float()
    mu2_coef = x_bd.float() @ mu2_d.float()
    x_norm = x_bd.float().norm(dim=-1).clamp_min(EPS)
    mu1_cos = mu1_coef / x_norm
    mu2_cos = mu2_coef / x_norm
    return mu1_coef, mu2_coef, mu1_cos, mu2_cos


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, sem


# =============================================================================
# dataset / oracle
# =============================================================================

@dataclass
class MuBasis:
    mu1: np.ndarray
    mu2: np.ndarray
    source_path: str


def load_rta_rows(repo: str, split: str, subset_type: str, n_images: int) -> pd.DataFrame:
    ds = load_dataset(repo, split=split)
    required = {"type", "image", "id"}
    missing = sorted(required - set(ds.column_names))
    if missing:
        raise KeyError(f"Missing dataset columns: {missing}; have {ds.column_names}")

    rows = []
    for row in ds:
        if str(row["type"]) != str(subset_type):
            continue
        rows.append({
            "stim_id": str(row["id"]),
            "image": row["image"],
            "type": str(row["type"]),
        })
        if len(rows) >= n_images:
            break
    if not rows:
        raise RuntimeError(f"No rows found for type={subset_type!r} in {repo}")
    return pd.DataFrame(rows)


def load_mu_basis(
    oracle_root: Path,
    model_name: str,
    width: int,
) -> MuBasis:
    path = oracle_root / model_name / "mu_basis.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    basis = np.asarray(data["mu_basis"], np.float32)
    if basis.shape[1] != width:
        raise RuntimeError(f"mu basis width mismatch: {basis.shape} vs D={width}")
    return MuBasis(
        mu1=basis[0],
        mu2=basis[1],
        source_path=str(path),
    )


def compute_fixed_b23_register_masks(bundle, manifest: pd.DataFrame, args) -> np.ndarray:
    visual = bundle.model.visual
    masks = []
    stop_block = max(args.late_blocks)
    if stop_block < 23:
        stop_block = 23
    for start in tqdm(range(0, len(manifest), args.batch_size), desc=f"{bundle.name}/B23mask", unit="batch"):
        images, _ = build_images(bundle, manifest, start, args.batch_size)
        x = visual._prepare_tokens(images)
        for block_index in range(stop_block):
            x = visual.transformer.resblocks[block_index](x)
        norms = x[1:].float().norm(dim=-1).T
        reg = select_register_mask(
            norms,
            threshold=float(args.final_register_threshold),
            minimum=int(args.final_register_min),
            maximum=int(args.final_register_max),
        )
        masks.append(reg.detach().cpu().numpy().astype(np.uint8))
        del images, x, norms, reg
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(masks, axis=0)


# =============================================================================
# scanning
# =============================================================================

def build_images(bundle, manifest: pd.DataFrame, start: int, batch_size: int) -> tuple[torch.Tensor, list[str]]:
    chunk = manifest.iloc[start:start + batch_size]
    tensors, ids = [], []
    for row in chunk.itertuples(index=False):
        image = row.image
        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            pil = Image.open(str(image)).convert("RGB")
        tensors.append(bundle.preprocess(pil))
        ids.append(str(row.stim_id))
    images = torch.stack(tensors).to(bundle.device, dtype=bundle.model.dtype)
    return images, ids


@torch.no_grad()
def scan_model_heads(base, bundle, manifest: pd.DataFrame, fixed_b23_reg_mask: np.ndarray, args) -> pd.DataFrame:
    visual = bundle.model.visual
    late_blocks = set(args.late_blocks)
    max_block = max(late_blocks)
    rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(manifest), args.batch_size), desc=f"{bundle.name}/scan", unit="batch"):
        images, ids = build_images(bundle, manifest, start, args.batch_size)
        batch = images.shape[0]
        x = visual._prepare_tokens(images)
        patch_count = x.shape[0] - 1

        fixed_reg = torch.from_numpy(fixed_b23_reg_mask[start:start + batch].astype(bool)).to(bundle.device)

        for block_index in range(max_block + 1):
            block = visual.transformer.resblocks[block_index]
            heads = int(block.attn.num_heads)

            ln1 = block.ln_1(x)
            attn_out, _ = block.attention(ln1, need_weights=True, capture=True)

            if block_index in late_blocks:
                q = normalize_qkv_shape(block.attn.last_q, batch, heads, x.shape[0]).float()
                k = normalize_qkv_shape(block.attn.last_k, batch, heads, x.shape[0]).float()
                dh = q.shape[-1]
                scale = 1.0 / math.sqrt(dh)

                q_cls = q[:, :, 0, :]                                # [B,H,Dh]
                k_spatial = k[:, :, 1:1 + patch_count, :]            # [B,H,P,Dh]
                q_norm = q_cls.norm(dim=-1)                          # [B,H]
                k_norm = k_spatial.norm(dim=-1)                      # [B,H,P]
                logits = (q_cls[:, :, None, :] * k_spatial).sum(dim=-1) * scale

                reg_mask = fixed_reg[:, None, :].expand(batch, heads, patch_count)
                neg_inf = torch.full_like(logits, -1e9)
                k_sel_norm = torch.where(reg_mask, k_norm, neg_inf).max(dim=-1).values
                abs_logits = torch.where(reg_mask, logits.abs(), neg_inf)
                best_idx = abs_logits.argmax(dim=-1)                 # [B,H]
                best_logit_abs = abs_logits.gather(-1, best_idx[..., None]).squeeze(-1)
                best_logit_signed = logits.gather(-1, best_idx[..., None]).squeeze(-1)

                score = q_norm * k_sel_norm
                for h in range(heads):
                    rows.append({
                        "model_name": bundle.name,
                        "block": int(block_index),
                        "head": int(h),
                        "n_images": int(batch),
                        "q_cls_norm_mean": float(q_norm[:, h].mean()),
                        "k_selected_reg_norm_mean": float(k_sel_norm[:, h].mean()),
                        "qk_norm_product_mean": float(score[:, h].mean()),
                        "cls_reg_qk_abs_mean": float(best_logit_abs[:, h].mean()),
                        "cls_reg_qk_signed_mean": float(best_logit_signed[:, h].mean()),
                    })

            # continue the block manually so we do not run attention twice
            x_attn = x + attn_out
            ln2 = block.ln_2(x_attn)
            x = x_attn + block.mlp.c_proj(block.mlp.gelu(block.mlp.c_fc(ln2)))
            clear_attn_cache(block)

        del images, x
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    metrics = [
        "q_cls_norm_mean",
        "k_selected_reg_norm_mean",
        "qk_norm_product_mean",
        "cls_reg_qk_abs_mean",
        "cls_reg_qk_signed_mean",
    ]
    out_rows = []
    for key, group in frame.groupby(["model_name", "block", "head"], sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = {
            "model_name": key[0],
            "block": int(key[1]),
            "head": int(key[2]),
            "n_images": int(group["n_images"].sum()),
        }
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce").to_numpy(np.float64)
            vals = vals[np.isfinite(vals)]
            row[metric] = float(vals.mean()) if vals.size else float("nan")
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    out = out.sort_values(["model_name", "block", "qk_norm_product_mean"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def select_top_heads(scan_df: pd.DataFrame, top_heads_per_block: int) -> pd.DataFrame:
    rows = []
    for (model_name, block), group in scan_df.groupby(["model_name", "block"], sort=True):
        z = group.sort_values("qk_norm_product_mean", ascending=False).head(top_heads_per_block)
        for rank, row in enumerate(z.itertuples(index=False), start=1):
            rows.append({
                "model_name": row.model_name,
                "block": int(row.block),
                "head": int(row.head),
                "rank_within_block": int(rank),
                "q_cls_norm_mean": float(row.q_cls_norm_mean),
                "k_selected_reg_norm_mean": float(row.k_selected_reg_norm_mean),
                "qk_norm_product_mean": float(row.qk_norm_product_mean),
                "cls_reg_qk_abs_mean": float(row.cls_reg_qk_abs_mean),
                "cls_reg_qk_signed_mean": float(row.cls_reg_qk_signed_mean),
            })
    return pd.DataFrame(rows)


# =============================================================================
# gradients
# =============================================================================

def forward_to_pre_block(bundle, images: torch.Tensor, stop_block: int) -> torch.Tensor:
    visual = bundle.model.visual
    x = visual._prepare_tokens(images.to(bundle.device, dtype=bundle.model.dtype))
    for block_index in range(stop_block):
        x = visual.transformer.resblocks[block_index](x)
    return x


def gather_token_rows(x_tbd: torch.Tensor, token_index_b: torch.Tensor) -> torch.Tensor:
    # x_tbd: [T,B,D], token_index_b: [B]
    out = []
    for bi in range(x_tbd.shape[1]):
        out.append(x_tbd[int(token_index_b[bi].item()), bi].float())
    return torch.stack(out, dim=0)


def gradient_probe_for_head(base, bundle, manifest: pd.DataFrame, mu_basis: MuBasis, fixed_b23_reg_mask: np.ndarray, block_index: int, head: int, args) -> dict[str, Any]:
    visual = bundle.model.visual
    block = visual.transformer.resblocks[block_index]
    mu1 = torch.from_numpy(mu_basis.mu1).to(bundle.device).float()
    mu2 = torch.from_numpy(mu_basis.mu2).to(bundle.device).float()

    cls_mu1, cls_mu2 = [], []
    cls_cos1, cls_cos2 = [], []
    reg_mu1, reg_mu2 = [], []
    reg_cos1, reg_cos2 = [], []
    signed_logits, abs_logits = [], []

    total_images = min(len(manifest), args.gradient_images)
    for start in tqdm(range(0, total_images, args.gradient_batch_size), desc=f"{bundle.name}/B{block_index}/H{head} grad", unit="batch"):
        images, _ = build_images(bundle, manifest, start, args.gradient_batch_size)
        batch = images.shape[0]
        patch_count = bundle.model.visual.positional_embedding.shape[0] - 1
        reg_mask = torch.from_numpy(fixed_b23_reg_mask[start:start + batch].astype(bool)).to(bundle.device)

        pre = forward_to_pre_block(bundle, images, block_index).detach().clone().requires_grad_(True)
        ln1 = block.ln_1(pre)
        _, _ = block.attention(ln1, need_weights=True, capture=True)
        heads = int(block.attn.num_heads)
        q = normalize_qkv_shape(block.attn.last_q, batch, heads, pre.shape[0]).float()
        k = normalize_qkv_shape(block.attn.last_k, batch, heads, pre.shape[0]).float()
        dh = q.shape[-1]
        scale = 1.0 / math.sqrt(dh)

        q_cls = q[:, head, 0, :]                       # [B,Dh]
        k_spatial = k[:, head, 1:1 + patch_count, :]   # [B,P,Dh]
        logits = (q_cls[:, None, :] * k_spatial).sum(dim=-1) * scale
        neg_inf = torch.full_like(logits, -1e9)
        abs_score = torch.where(reg_mask, logits.abs(), neg_inf)
        best_idx = abs_score.argmax(dim=-1)            # spatial 0-based
        best_logit_signed = logits.gather(1, best_idx[:, None]).squeeze(1)
        sign = best_logit_signed.detach().sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        ell = (best_logit_signed * sign).mean()        # == mean absolute selected logit with fixed native sign

        grad = torch.autograd.grad(ell, pre, retain_graph=False, create_graph=False)[0].float()
        grad_cls = grad[0]                             # [B,D]
        token_index = best_idx + 1                     # convert spatial -> token index
        grad_reg = gather_token_rows(grad, token_index)

        c1, c2, cc1, cc2 = project_to_mu_basis(grad_cls, mu1, mu2)
        r1, r2, rc1, rc2 = project_to_mu_basis(grad_reg, mu1, mu2)
        cls_mu1.extend(c1.detach().cpu().numpy().tolist())
        cls_mu2.extend(c2.detach().cpu().numpy().tolist())
        cls_cos1.extend(cc1.detach().cpu().numpy().tolist())
        cls_cos2.extend(cc2.detach().cpu().numpy().tolist())
        reg_mu1.extend(r1.detach().cpu().numpy().tolist())
        reg_mu2.extend(r2.detach().cpu().numpy().tolist())
        reg_cos1.extend(rc1.detach().cpu().numpy().tolist())
        reg_cos2.extend(rc2.detach().cpu().numpy().tolist())
        signed_logits.extend(best_logit_signed.detach().cpu().numpy().tolist())
        abs_logits.extend((best_logit_signed.detach().abs()).cpu().numpy().tolist())

        clear_attn_cache(block)
        del images, pre, grad, q, k, logits
        if bundle.device.type == "cuda":
            torch.cuda.empty_cache()

    cm1, sem_cm1 = mean_sem(np.asarray(cls_mu1))
    cm2, sem_cm2 = mean_sem(np.asarray(cls_mu2))
    cc1m, sem_cc1 = mean_sem(np.asarray(cls_cos1))
    cc2m, sem_cc2 = mean_sem(np.asarray(cls_cos2))
    rm1, sem_rm1 = mean_sem(np.asarray(reg_mu1))
    rm2, sem_rm2 = mean_sem(np.asarray(reg_mu2))
    rc1m, sem_rc1 = mean_sem(np.asarray(reg_cos1))
    rc2m, sem_rc2 = mean_sem(np.asarray(reg_cos2))
    slm, sem_slm = mean_sem(np.asarray(signed_logits))
    alm, sem_alm = mean_sem(np.asarray(abs_logits))

    return {
        "model_name": bundle.name,
        "block": int(block_index),
        "head": int(head),
        "n_images": int(total_images),
        "gate_abs_mean": alm,
        "gate_abs_sem": sem_alm,
        "gate_signed_mean": slm,
        "gate_signed_sem": sem_slm,
        "grad_cls_mu1_coef_mean": cm1,
        "grad_cls_mu1_coef_sem": sem_cm1,
        "grad_cls_mu2_coef_mean": cm2,
        "grad_cls_mu2_coef_sem": sem_cm2,
        "grad_cls_mu1_cos_mean": cc1m,
        "grad_cls_mu1_cos_sem": sem_cc1,
        "grad_cls_mu2_cos_mean": cc2m,
        "grad_cls_mu2_cos_sem": sem_cc2,
        "grad_reg_mu1_coef_mean": rm1,
        "grad_reg_mu1_coef_sem": sem_rm1,
        "grad_reg_mu2_coef_mean": rm2,
        "grad_reg_mu2_coef_sem": sem_rm2,
        "grad_reg_mu1_cos_mean": rc1m,
        "grad_reg_mu1_cos_sem": sem_rc1,
        "grad_reg_mu2_cos_mean": rc2m,
        "grad_reg_mu2_cos_sem": sem_rc2,
        "grad_cls_plane_angle_deg": float(np.degrees(np.arctan2(cm2, cm1))),
        "grad_reg_plane_angle_deg": float(np.degrees(np.arctan2(rm2, rm1))),
        "grad_cls_plane_norm": float(math.sqrt(cm1 * cm1 + cm2 * cm2)),
        "grad_reg_plane_norm": float(math.sqrt(rm1 * rm1 + rm2 * rm2)),
    }


# =============================================================================
# plotting / report
# =============================================================================

def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(scan_df: pd.DataFrame, out_dir: Path) -> None:
    for model_name in MODEL_ORDER:
        z = scan_df[scan_df["model_name"] == model_name]
        if z.empty:
            continue
        piv = z.pivot(index="head", columns="block", values="qk_norm_product_mean").sort_index().sort_index(axis=1)
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        im = ax.imshow(piv.to_numpy(np.float64), aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"B{int(x)}" for x in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"H{int(x)}" for x in piv.index])
        ax.set_title(f"{model_name}: late Q*K product score")
        fig.colorbar(im, ax=ax, label="mean ||q_cls|| * ||k_reg||")
        savefig(fig, out_dir / f"qk_head_score_heatmap_{model_name}.png")

        piv = z.pivot(index="head", columns="block", values="cls_reg_qk_abs_mean").sort_index().sort_index(axis=1)
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        im = ax.imshow(piv.to_numpy(np.float64), aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"B{int(x)}" for x in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"H{int(x)}" for x in piv.index])
        ax.set_title(f"{model_name}: selected |CLS↔REG QK| gate")
        fig.colorbar(im, ax=ax, label="mean |selected QK logit|")
        savefig(fig, out_dir / f"qk_abs_gate_heatmap_{model_name}.png")


def plot_gradient_summary(grad_df: pd.DataFrame, out_dir: Path) -> None:
    for model_name in MODEL_ORDER:
        z = grad_df[grad_df["model_name"] == model_name].sort_values(["block", "head"])
        if z.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=False)
        axes[0].scatter(z["grad_cls_mu1_coef_mean"], z["grad_cls_mu2_coef_mean"])
        for row in z.itertuples(index=False):
            axes[0].annotate(f"B{int(row.block)}H{int(row.head)}", (float(row.grad_cls_mu1_coef_mean), float(row.grad_cls_mu2_coef_mean)), fontsize=8)
        axes[0].axhline(0, linewidth=.8)
        axes[0].axvline(0, linewidth=.8)
        axes[0].set_xlabel("grad wrt CLS · mu1")
        axes[0].set_ylabel("grad wrt CLS · mu2")
        axes[0].set_title("CLS-side gate gradient")
        axes[0].grid(alpha=.2)

        axes[1].scatter(z["grad_reg_mu1_coef_mean"], z["grad_reg_mu2_coef_mean"])
        for row in z.itertuples(index=False):
            axes[1].annotate(f"B{int(row.block)}H{int(row.head)}", (float(row.grad_reg_mu1_coef_mean), float(row.grad_reg_mu2_coef_mean)), fontsize=8)
        axes[1].axhline(0, linewidth=.8)
        axes[1].axvline(0, linewidth=.8)
        axes[1].set_xlabel("grad wrt selected REG · mu1")
        axes[1].set_ylabel("grad wrt selected REG · mu2")
        axes[1].set_title("REG-side gate gradient")
        axes[1].grid(alpha=.2)

        fig.suptitle(f"{model_name}: projected local gate gradients in the role plane")
        fig.tight_layout(rect=[0, 0, 1, .94])
        savefig(fig, out_dir / f"qk_gradient_plane_summary_{model_name}.png")


def write_summary(out_dir: Path, selected_df: pd.DataFrame, grad_df: pd.DataFrame) -> None:
    lines = [
        "LATE Q/K ROLE-PLANE HEAD PROBE",
        "=" * 76,
        "",
        "Selected heads (top by mean ||q_cls|| * ||k_reg|| within each late block):",
        "",
        "model                 block  head   q*k score   |QK| mean   signed QK mean",
        "-" * 76,
    ]
    for row in selected_df.itertuples(index=False):
        lines.append(
            f"{row.model_name:<21} B{int(row.block):<5d} H{int(row.head):<5d} "
            f"{float(row.qk_norm_product_mean):>10.4f} "
            f"{float(row.cls_reg_qk_abs_mean):>10.4f} "
            f"{float(row.cls_reg_qk_signed_mean):>15.4f}"
        )

    lines += [
        "",
        "Projected local gradients for the selected heads:",
        "",
        "model                 block  head   dgate/dCLS(mu1,mu2)     dgate/dREG(mu1,mu2)",
        "-" * 76,
    ]
    for row in grad_df.itertuples(index=False):
        lines.append(
            f"{row.model_name:<21} B{int(row.block):<5d} H{int(row.head):<5d} "
            f"({float(row.grad_cls_mu1_coef_mean):>8.4f}, {float(row.grad_cls_mu2_coef_mean):>8.4f})   "
            f"({float(row.grad_reg_mu1_coef_mean):>8.4f}, {float(row.grad_reg_mu2_coef_mean):>8.4f})"
        )

    lines += [
        "",
        "Interpretation guardrails:",
        "  * head selection is purely native and norm-based, not hinge-derived;",
        "  * gradients are local derivatives of the selected |CLS↔REG QK| gate;",
        "  * projecting gradients into (mu1,mu2) tests whether the late high-gain",
        "    gate is aligned with the same low-dimensional role plane seen earlier.",
        "",
    ]
    (out_dir / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI / main
# =============================================================================

def scan_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe late high-gain Q/K heads in the CLS/register role plane.")

    parser.add_argument("--oracle_root", default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))

    parser.add_argument("--dataset_repo", default="zer0int/RTA-100-Triplet")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--subset_type", default="RTA")
    parser.add_argument("--n_images", type=int, default=120)

    parser.add_argument("--late_blocks", default="20-23")
    parser.add_argument("--top_heads_per_block", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_images", type=int, default=48)
    parser.add_argument("--gradient_batch_size", type=int, default=4)
    parser.add_argument("--final_register_threshold", type=float, default=60.0)
    parser.add_argument("--final_register_min", type=int, default=1)
    parser.add_argument("--final_register_max", type=int, default=4)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip_module", default="attnclip_mechinterp_sae")
    parser.add_argument("--model_spec", default="ViT-L/14")
    parser.add_argument(
        "--gmp_checkpoint",
        default=r"ViT-L-14-BEST-smooth-GmP-ft-pickle-OpenAI.pt",
    )
    parser.add_argument(
        "--xattn_checkpoint",
        default=DEFAULT_XATTN_CHECKPOINT,
        help="Full x-attn checkpoint used only as the ordinary visual-weight source for finetune_stripped.",
    )
    parser.add_argument(
        "--xattn_module",
        default="oaiclip",
        help="Module to pre-import so x-attn pickle class paths resolve.",
    )
    parser.add_argument(
        "--pickle_module",
        default="clip",
        help="Module to pre-import before loading ordinary OpenAI CLIP pickle checkpoints.",
    )

    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    args.models = tuple(token.strip() for token in args.models.split(",") if token.strip())
    args.late_blocks = parse_ints(args.late_blocks)
    unknown = set(args.models) - set(MODEL_ORDER)
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")
    if not args.late_blocks:
        parser.error("--late_blocks must not be empty")
    return args


def scan_main() -> None:
    args = scan_parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", vars(args))

    base = _backbone_tools
    manifest = load_rta_rows(args.dataset_repo, args.dataset_split, args.subset_type, args.n_images)

    scan_frames = []
    selected_frames = []
    grad_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        print(f"\n================ {model_name} ================")
        model_dir = out_dir / model_name
        bundle = base.load_bundle(model_name, args, model_dir / "load_audit")
        bundle.name = model_name

        width = int(bundle.model.visual.positional_embedding.shape[1])
        mu_basis = load_mu_basis(Path(args.oracle_root), model_name, width)
        print(f"[mu basis] {mu_basis.source_path}")
        fixed_b23_reg_mask = compute_fixed_b23_register_masks(bundle, manifest, args)
        print(f"[B23 masks] shape={tuple(fixed_b23_reg_mask.shape)}")

        scan_df = scan_model_heads(base, bundle, manifest, fixed_b23_reg_mask, args)
        scan_frames.append(scan_df)

        selected_df = select_top_heads(scan_df, args.top_heads_per_block)
        selected_frames.append(selected_df)

        for row in selected_df.itertuples(index=False):
            grad_rows.append(
                gradient_probe_for_head(
                    base,
                    bundle,
                    manifest,
                    mu_basis,
                    fixed_b23_reg_mask,
                    block_index=int(row.block),
                    head=int(row.head),
                    args=args,
                )
            )

        del bundle, mu_basis, fixed_b23_reg_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scan_all = pd.concat(scan_frames, ignore_index=True)
    selected_all = pd.concat(selected_frames, ignore_index=True)
    grad_df = pd.DataFrame(grad_rows)

    scan_all.to_csv(out_dir / "qk_head_scan.csv", index=False)
    selected_all.to_csv(out_dir / "qk_selected_heads.csv", index=False)
    grad_df.to_csv(out_dir / "qk_role_plane_gradients.csv", index=False)

    plot_heatmaps(scan_all, out_dir)
    plot_gradient_summary(grad_df, out_dir)
    write_summary(out_dir, selected_all, grad_df)

    include = [
        out_dir / "config.json",
        out_dir / "qk_head_scan.csv",
        out_dir / "qk_selected_heads.csv",
        out_dir / "qk_role_plane_gradients.csv",
        out_dir / "SUMMARY.txt",
    ]
    include += sorted(out_dir.glob("qk_*.png"))

    zpath = out_dir / "compact_summary_workspace_qk_role_gates_scan.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for path in include:
            if path.is_file():
                archive.write(path, arcname=path.relative_to(out_dir).as_posix())

    print("[compact summary]", zpath)


# SAME HEADS
matplotlib.use("Agg")


# =============================================================================
# helpers
# =============================================================================


# =============================================================================
# dataset / oracle
# =============================================================================


# =============================================================================
# scanning
# =============================================================================


# =============================================================================
# gradients
# =============================================================================


# =============================================================================
# plotting / report
# =============================================================================


def pairwise_cos2(a1: float, a2: float, b1: float, b2: float) -> float:
    a = np.asarray([a1, a2], np.float64)
    b = np.asarray([b1, b2], np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def build_same_head_cross_model(scan_df: pd.DataFrame, grad_df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("pretrained", "gmp"),
        ("pretrained", "finetune_stripped"),
        ("gmp", "finetune_stripped"),
    ]
    rows = []
    for head in sorted(set(grad_df["head"].astype(int).tolist())):
        for model_a, model_b in pairs:
            sa = scan_df[
                (scan_df["model_name"] == model_a)
                & (scan_df["block"] == 22)
                & (scan_df["head"] == head)
            ]
            sb = scan_df[
                (scan_df["model_name"] == model_b)
                & (scan_df["block"] == 22)
                & (scan_df["head"] == head)
            ]
            ga = grad_df[
                (grad_df["model_name"] == model_a)
                & (grad_df["block"] == 22)
                & (grad_df["head"] == head)
            ]
            gb = grad_df[
                (grad_df["model_name"] == model_b)
                & (grad_df["block"] == 22)
                & (grad_df["head"] == head)
            ]
            if sa.empty or sb.empty or ga.empty or gb.empty:
                continue
            sa, sb, ga, gb = sa.iloc[0], sb.iloc[0], ga.iloc[0], gb.iloc[0]

            cls_cos = pairwise_cos2(
                ga["grad_cls_mu1_coef_mean"], ga["grad_cls_mu2_coef_mean"],
                gb["grad_cls_mu1_coef_mean"], gb["grad_cls_mu2_coef_mean"],
            )
            reg_cos = pairwise_cos2(
                ga["grad_reg_mu1_coef_mean"], ga["grad_reg_mu2_coef_mean"],
                gb["grad_reg_mu1_coef_mean"], gb["grad_reg_mu2_coef_mean"],
            )

            sign_a = np.sign(float(sa["cls_reg_qk_signed_mean"]))
            sign_b = np.sign(float(sb["cls_reg_qk_signed_mean"]))
            rows.append({
                "block": 22,
                "head": int(head),
                "model_a": model_a,
                "model_b": model_b,
                "q_norm_ratio_b_over_a": float(sb["q_cls_norm_mean"]) / max(float(sa["q_cls_norm_mean"]), 1e-12),
                "k_norm_ratio_b_over_a": float(sb["k_selected_reg_norm_mean"]) / max(float(sa["k_selected_reg_norm_mean"]), 1e-12),
                "abs_gate_ratio_b_over_a": float(sb["cls_reg_qk_abs_mean"]) / max(float(sa["cls_reg_qk_abs_mean"]), 1e-12),
                "signed_gate_a": float(sa["cls_reg_qk_signed_mean"]),
                "signed_gate_b": float(sb["cls_reg_qk_signed_mean"]),
                "signed_gate_sign_flip": bool(sign_a != 0 and sign_b != 0 and sign_a != sign_b),
                "cls_role_plane_gradient_cosine": cls_cos,
                "reg_role_plane_gradient_cosine": reg_cos,
                "cls_plane_angle_a_deg": float(ga["grad_cls_plane_angle_deg"]),
                "cls_plane_angle_b_deg": float(gb["grad_cls_plane_angle_deg"]),
                "reg_plane_angle_a_deg": float(ga["grad_reg_plane_angle_deg"]),
                "reg_plane_angle_b_deg": float(gb["grad_reg_plane_angle_deg"]),
            })
    return pd.DataFrame(rows)


def plot_same_head_comparison(scan_df: pd.DataFrame, grad_df: pd.DataFrame, out_dir: Path) -> None:
    heads = sorted(set(grad_df["head"].astype(int).tolist()))
    fig, axes = plt.subplots(2, len(heads), figsize=(4.2 * len(heads), 8.0), squeeze=False)

    for col, head in enumerate(heads):
        sg = scan_df[(scan_df["block"] == 22) & (scan_df["head"] == head)]
        gg = grad_df[(grad_df["block"] == 22) & (grad_df["head"] == head)]

        labels, q_vals, k_vals, gate_vals = [], [], [], []
        for model_name in MODEL_ORDER:
            row = sg[sg["model_name"] == model_name]
            if row.empty:
                continue
            r = row.iloc[0]
            labels.append(model_name)
            q_vals.append(float(r["q_cls_norm_mean"]))
            k_vals.append(float(r["k_selected_reg_norm_mean"]))
            gate_vals.append(float(r["cls_reg_qk_signed_mean"]))

        ax = axes[0, col]
        x = np.arange(len(labels))
        width = 0.24
        ax.bar(x - width, q_vals, width=width, label="||Q_CLS||")
        ax.bar(x, k_vals, width=width, label="||K_REG||")
        ax2 = ax.twinx()
        ax2.plot(x, gate_vals, marker="o", linewidth=1.6, label="signed QK")
        ax.axhline(0, linewidth=.6)
        ax2.axhline(0, linewidth=.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"B22 H{head}: norm scaffold + signed gate")
        ax.grid(axis="y", alpha=.18)
        if col == 0:
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)

        ax = axes[1, col]
        for model_name in MODEL_ORDER:
            row = gg[gg["model_name"] == model_name]
            if row.empty:
                continue
            r = row.iloc[0]
            ax.arrow(
                0, 0,
                float(r["grad_cls_mu1_coef_mean"]),
                float(r["grad_cls_mu2_coef_mean"]),
                length_includes_head=True,
                head_width=max(0.015, 0.04 * max(float(r["grad_cls_plane_norm"]), 1e-6)),
                alpha=.8,
            )
            ax.text(
                float(r["grad_cls_mu1_coef_mean"]),
                float(r["grad_cls_mu2_coef_mean"]),
                model_name,
                fontsize=8,
            )
        ax.axhline(0, linewidth=.7)
        ax.axvline(0, linewidth=.7)
        ax.set_xlabel("d gate / d CLS·mu1")
        if col == 0:
            ax.set_ylabel("d gate / d CLS·mu2")
        ax.set_title(f"B22 H{head}: CLS gate-gradient direction")
        ax.grid(alpha=.18)

    fig.suptitle("Forced same-head B22 control")
    fig.tight_layout(rect=[0, 0, 1, .96])
    savefig(fig, out_dir / "forced_b22_samehead_comparison.png")


def write_forced_summary(
    out_dir: Path,
    scan_df: pd.DataFrame,
    grad_df: pd.DataFrame,
    cross_df: pd.DataFrame,
) -> None:
    lines = [
        "FORCED SAME-HEAD B22 Q/K ROLE-PLANE CONTROL",
        "=" * 78,
        "",
        "Same physical heads are measured in every model.",
        "",
        "model                 head   ||Qcls||   ||Kreg||   |QK|      signed QK",
        "-" * 78,
    ]
    for row in scan_df.sort_values(["head", "model_name"]).itertuples(index=False):
        lines.append(
            f"{row.model_name:<21} H{int(row.head):<5d} "
            f"{float(row.q_cls_norm_mean):>9.4f} "
            f"{float(row.k_selected_reg_norm_mean):>10.4f} "
            f"{float(row.cls_reg_qk_abs_mean):>9.4f} "
            f"{float(row.cls_reg_qk_signed_mean):>12.4f}"
        )

    lines += [
        "",
        "Role-plane local gate gradients:",
        "",
        "model                 head   CLS (mu1,mu2)                  REG (mu1,mu2)",
        "-" * 86,
    ]
    for row in grad_df.sort_values(["head", "model_name"]).itertuples(index=False):
        lines.append(
            f"{row.model_name:<21} H{int(row.head):<5d} "
            f"({float(row.grad_cls_mu1_coef_mean):>9.5f}, {float(row.grad_cls_mu2_coef_mean):>9.5f})   "
            f"({float(row.grad_reg_mu1_coef_mean):>9.5f}, {float(row.grad_reg_mu2_coef_mean):>9.5f})"
        )

    lines += [
        "",
        "Cross-model same-head comparisons:",
        "",
        "head  pair                              signflip   CLS grad cos   REG grad cos",
        "-" * 82,
    ]
    for row in cross_df.sort_values(["head", "model_a", "model_b"]).itertuples(index=False):
        pair = f"{row.model_a}->{row.model_b}"
        lines.append(
            f"H{int(row.head):<4d} {pair:<33} "
            f"{str(bool(row.signed_gate_sign_flip)):<10} "
            f"{float(row.cls_role_plane_gradient_cosine):>12.4f} "
            f"{float(row.reg_role_plane_gradient_cosine):>12.4f}"
        )

    lines += [
        "",
        "Guardrail: a signed QK reversal is not by itself 'more' or 'less' reading.",
        "It is a change in the relational gate convention. The role-plane gradients",
        "test whether that same-head gate also rotates/re-polarizes in the conserved",
        "(mu1,mu2) interface geometry.",
        "",
    ]
    (out_dir / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI / main
# =============================================================================

def same_heads_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forced B22 same-head Q/K role-plane control."
    )

    parser.add_argument("--oracle_root", default=DEFAULT_ORACLE_ROOT)
    parser.add_argument(
        "--out_dir",
        default=r"b22_samehead_qk_role_plane",
    )
    parser.add_argument("--models", default=",".join(MODEL_ORDER))

    parser.add_argument("--dataset_repo", default="zer0int/RTA-100-Triplet")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--subset_type", default="RTA")
    parser.add_argument("--n_images", type=int, default=120)

    parser.add_argument("--block", type=int, default=22)
    parser.add_argument("--heads", default="3,8,9,10")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_images", type=int, default=48)
    parser.add_argument("--gradient_batch_size", type=int, default=4)
    parser.add_argument("--final_register_threshold", type=float, default=60.0)
    parser.add_argument("--final_register_min", type=int, default=1)
    parser.add_argument("--final_register_max", type=int, default=4)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip_module", default="attnclip_mechinterp_sae")
    parser.add_argument("--model_spec", default="ViT-L/14")
    parser.add_argument(
        "--gmp_checkpoint",
        default=r"ViT-L-14-BEST-smooth-GmP-ft-pickle-OpenAI.pt",
    )
    parser.add_argument(
        "--xattn_checkpoint",
        default=DEFAULT_XATTN_CHECKPOINT,
        help="Used only as ordinary visual-weight source for finetune_stripped.",
    )
    parser.add_argument("--xattn_module", default="oaiclip")
    parser.add_argument("--pickle_module", default="clip")
    parser.add_argument("--seed", type=int, default=20260909)

    args = parser.parse_args()
    args.models = tuple(
        token.strip() for token in args.models.split(",") if token.strip()
    )
    args.heads = parse_ints(args.heads)
    args.late_blocks = [int(args.block)]
    args.top_heads_per_block = len(args.heads)

    unknown = set(args.models) - set(MODEL_ORDER)
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")
    if not args.heads:
        parser.error("--heads cannot be empty")
    if args.block != 22:
        parser.error("same_heads compares B22 only; use scan --late_blocks for other blocks")
    return args


def same_heads_main() -> None:
    args = same_heads_parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", vars(args))

    base = _backbone_tools
    manifest = load_rta_rows(
        args.dataset_repo,
        args.dataset_split,
        args.subset_type,
        args.n_images,
    )

    scan_frames = []
    grad_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        print(f"\n================ {model_name} ================")
        model_dir = out_dir / model_name
        bundle = base.load_bundle(model_name, args, model_dir / "load_audit")
        bundle.name = model_name

        width = int(bundle.model.visual.positional_embedding.shape[1])
        mu_basis = load_mu_basis(Path(args.oracle_root), model_name, width)
        print(f"[mu basis] {mu_basis.source_path}")

        fixed_b23_reg_mask = compute_fixed_b23_register_masks(
            bundle, manifest, args
        )
        print(f"[B23 masks] shape={tuple(fixed_b23_reg_mask.shape)}")

        scan_df = scan_model_heads(
            base,
            bundle,
            manifest,
            fixed_b23_reg_mask,
            args,
        )
        scan_df = scan_df[
            (scan_df["block"] == args.block)
            & (scan_df["head"].isin(args.heads))
        ].copy()
        scan_frames.append(scan_df)

        for head in args.heads:
            print(f"[forced gradient] {model_name} B{args.block} H{head}")
            grad_rows.append(
                gradient_probe_for_head(
                    base,
                    bundle,
                    manifest,
                    mu_basis,
                    fixed_b23_reg_mask,
                    block_index=args.block,
                    head=head,
                    args=args,
                )
            )

        del bundle, mu_basis, fixed_b23_reg_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scan_all = pd.concat(scan_frames, ignore_index=True)
    grad_df = pd.DataFrame(grad_rows)
    cross_df = build_same_head_cross_model(scan_all, grad_df)

    scan_all.to_csv(out_dir / "forced_b22_head_scan.csv", index=False)
    pd.DataFrame({
        "block": [args.block] * len(args.heads),
        "head": args.heads,
    }).to_csv(out_dir / "forced_b22_heads.csv", index=False)
    grad_df.to_csv(out_dir / "forced_b22_role_plane_gradients.csv", index=False)
    cross_df.to_csv(out_dir / "forced_b22_cross_model_samehead.csv", index=False)

    plot_same_head_comparison(scan_all, grad_df, out_dir)
    plot_gradient_summary(grad_df, out_dir)
    write_forced_summary(out_dir, scan_all, grad_df, cross_df)

    include = [
        out_dir / "config.json",
        out_dir / "forced_b22_head_scan.csv",
        out_dir / "forced_b22_heads.csv",
        out_dir / "forced_b22_role_plane_gradients.csv",
        out_dir / "forced_b22_cross_model_samehead.csv",
        out_dir / "forced_b22_samehead_comparison.png",
        out_dir / "SUMMARY.txt",
    ]
    include += sorted(out_dir.glob("qk_gradient_plane_summary_*.png"))

    zpath = out_dir / "compact_summary_workspace_qk_role_gates_same_heads.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(
        zpath,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=7,
    ) as archive:
        for path in include:
            if path.is_file():
                archive.write(path, arcname=path.relative_to(out_dir).as_posix())

    print("[compact summary]", zpath)


def main(argv=None):
    """Dispatch a workflow; each subcommand retains its original CLI options."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {'scan': scan_main, 'same_heads': same_heads_main}
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

