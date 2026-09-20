from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import csv
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SVDResult:
    singular_values: np.ndarray
    right_vectors: np.ndarray
    explained_energy: np.ndarray
    mean: np.ndarray
    n_rows: int

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            singular_values=self.singular_values,
            right_vectors=self.right_vectors,
            explained_energy=self.explained_energy,
            mean=self.mean,
            n_rows=np.asarray([self.n_rows], dtype=np.int64),
        )


def randomized_svd_rows(rows: torch.Tensor, k: int = 8, center: bool = False) -> SVDResult:
    """Low-rank SVD of [N,D] rows using torch.pca_lowrank.

    `center=False` is deliberate for intervention deltas: the shared mean shift is
    part of the mechanism rather than nuisance variance.
    """
    x = rows.float()
    n, d = x.shape
    q = max(1, min(d, n, int(k) + 4))
    mean = x.mean(dim=0)
    u, s, v = torch.pca_lowrank(x, q=q, center=center, niter=4)
    s2 = s.square()
    total = x.square().sum().clamp_min(1e-20)
    explained = (s2 / total).detach().cpu().numpy()
    return SVDResult(
        singular_values=s.detach().cpu().numpy(),
        right_vectors=v[:, :k].T.detach().cpu().numpy(),
        explained_energy=explained[:k],
        mean=mean.detach().cpu().numpy(),
        n_rows=int(n),
    )


def principal_angles(v1: np.ndarray, v2: np.ndarray, k: Optional[int] = None) -> dict:
    """Principal angles between row-wise orthonormal bases [K,D]."""
    a = torch.from_numpy(np.asarray(v1)).float()
    b = torch.from_numpy(np.asarray(v2)).float()
    if k is not None:
        a, b = a[:k], b[:k]
    a = torch.linalg.qr(a.T, mode="reduced").Q
    b = torch.linalg.qr(b.T, mode="reduced").Q
    sigma = torch.linalg.svdvals(a.T @ b).clamp(0, 1)
    angles = torch.rad2deg(torch.acos(sigma))
    return {
        "cosines": sigma.cpu().numpy(),
        "angles_deg": angles.cpu().numpy(),
        "mean_cosine": float(sigma.mean()),
        "min_angle_deg": float(angles.min()),
        "max_angle_deg": float(angles.max()),
    }


class DeltaCollector:
    """Collect compact paired RN deltas without storing the full dataset tensor."""

    def __init__(self, *, width: int, tokens: int, patch_samples_per_image: int = 4,
                 seed: int = 20260901, max_rows_per_view: int | None = 8192):
        self.width = int(width)
        self.tokens = int(tokens)
        self.patch_samples_per_image = int(patch_samples_per_image)
        self.max_rows_per_view = None if max_rows_per_view is None or int(max_rows_per_view) <= 0 else int(max_rows_per_view)
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.mean_template_sum = torch.zeros(tokens, width, dtype=torch.float64)
        self.delta_sq_sum = 0.0
        self.cls_rows: list[torch.Tensor] = []
        self.patch_mean_rows: list[torch.Tensor] = []
        self.sampled_patch_rows: list[torch.Tensor] = []
        self.region_mean_rows: list[torch.Tensor] = []
        self.outside_mean_rows: list[torch.Tensor] = []

    def _append_bounded(self, bucket: list[torch.Tensor], rows: torch.Tensor) -> None:
        """Append CPU rows while enforcing a hard per-view row cap.

        The synthetic scenes are deterministic IID draws, so retaining the first
        N rows is sufficient for the low-rank diagnostic while the exact running
        mean/template and total energy still use every scene.  This prevents a
        large --n-scenes run from turning SVD sample storage into unbounded RAM.
        """
        rows = rows.detach().cpu()
        if self.max_rows_per_view is None:
            bucket.append(rows)
            return
        used = sum(int(x.shape[0]) for x in bucket)
        remain = self.max_rows_per_view - used
        if remain <= 0:
            return
        bucket.append(rows[:remain].clone())

    def add(self, delta_btd: torch.Tensor, region_masks_bp: Optional[torch.Tensor] = None) -> None:
        d = delta_btd.detach().float().cpu()
        b, t, w = d.shape
        if t != self.tokens or w != self.width:
            raise ValueError(f"delta shape {tuple(d.shape)} != expected (*,{self.tokens},{self.width})")
        self.n += b
        self.mean_template_sum += d.double().sum(dim=0)
        self.delta_sq_sum += float(d.square().sum())
        self._append_bounded(self.cls_rows, d[:, 0])
        patches = d[:, 1:]
        self._append_bounded(self.patch_mean_rows, patches.mean(dim=1))

        p = patches.shape[1]
        if self.patch_samples_per_image > 0:
            rows = []
            for bi in range(b):
                idx = self.rng.choice(p, size=min(p, self.patch_samples_per_image), replace=False)
                rows.append(patches[bi, torch.as_tensor(idx, dtype=torch.long)])
            self._append_bounded(self.sampled_patch_rows, torch.cat(rows, dim=0))

        if region_masks_bp is not None:
            mask = region_masks_bp.detach().cpu().bool()
            if mask.shape != (b, p):
                raise ValueError(f"region mask {tuple(mask.shape)} != {(b,p)}")
            rmean, omean = [], []
            for bi in range(b):
                r = patches[bi][mask[bi]]
                o = patches[bi][~mask[bi]]
                rmean.append(r.mean(dim=0) if len(r) else torch.zeros(w))
                omean.append(o.mean(dim=0) if len(o) else torch.zeros(w))
            self._append_bounded(self.region_mean_rows, torch.stack(rmean))
            self._append_bounded(self.outside_mean_rows, torch.stack(omean))

    def mean_template(self) -> torch.Tensor:
        if self.n == 0:
            raise RuntimeError("collector is empty")
        return (self.mean_template_sum / self.n).float()

    def universal_template_fraction(self) -> float:
        m = self.mean_template()
        # sum_n ||d_n - M||^2 = sum ||d_n||^2 - N||M||^2
        residual = max(0.0, self.delta_sq_sum - self.n * float(m.square().sum()))
        return 1.0 - residual / max(self.delta_sq_sum, 1e-20)

    def rows(self, name: str) -> torch.Tensor:
        mapping = {
            "cls": self.cls_rows,
            "patch_mean": self.patch_mean_rows,
            "patch_sample": self.sampled_patch_rows,
            "region_mean": self.region_mean_rows,
            "outside_mean": self.outside_mean_rows,
        }
        if name not in mapping or not mapping[name]:
            raise KeyError(f"No rows collected for {name}")
        return torch.cat(mapping[name], dim=0)


def rank_k_template(template_td: torch.Tensor, rank: int) -> torch.Tensor:
    x = template_td.float()
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    r = min(int(rank), s.numel())
    return (u[:, :r] * s[:r]) @ vh[:r]


def project_last_dim(x: torch.Tensor, basis_kd: torch.Tensor) -> torch.Tensor:
    """Project [...,D] onto the row-space of basis [K,D]."""
    b = basis_kd.float()
    q = torch.linalg.qr(b.T, mode="reduced").Q  # [D,K]
    return (x.float() @ q) @ q.T


def displacement_metrics(base: torch.Tensor, target: torch.Tensor, intervention: torch.Tensor) -> dict[str, float]:
    b = base.float()
    t = target.float()
    y = intervention.float()
    dt = t - b
    di = y - b
    denom = dt.square().sum(dim=-1).clamp_min(1e-12)
    recovery = (di * dt).sum(dim=-1) / denom
    residual_ratio = (y - t).norm(dim=-1) / dt.norm(dim=-1).clamp_min(1e-12)
    cos_target = F.cosine_similarity(y, t, dim=-1)
    cos_base = F.cosine_similarity(y, b, dim=-1)
    return {
        "effect_recovery_mean": float(recovery.mean()),
        "effect_recovery_median": float(recovery.median()),
        "target_residual_ratio_mean": float(residual_ratio.mean()),
        "cos_to_rn_mean": float(cos_target.mean()),
        "cos_to_base_mean": float(cos_base.mean()),
    }


def b13_head_decomposition(pair: dict, out_csv: str | Path | None = None) -> list[dict]:
    """Quantify exact RN softmax-steal vs active-value write at B13."""
    bc = pair["base_cache"]
    rc = pair["rn_cache"]
    base_z = bc["last_z"].float()          # [B,H,T,D]
    rn_z = rc["last_z"].float()[:, :, :-1]
    probs = rc["last_probs"].float()       # [B,H,T+1,T+1]
    rn_v = rc["last_v"].float()[:, :, -1] # [B,H,D]
    p = probs[:, :, :-1, -1]               # [B,H,T]

    predicted = (1.0 - p[..., None]) * base_z + p[..., None] * rn_v[:, :, None, :]
    err = (predicted - rn_z).norm(dim=-1)
    delta = rn_z - base_z
    steal = -p[..., None] * base_z
    write = p[..., None] * rn_v[:, :, None, :]

    rows = []
    hcount = base_z.shape[1]
    for h in range(hcount):
        for token_group, sl in (("cls", slice(0,1)), ("patch", slice(1,None))):
            pp = p[:, h, sl]
            dd = delta[:, h, sl]
            ss = steal[:, h, sl]
            ww = write[:, h, sl]
            ee = err[:, h, sl]
            flat_s = ss.reshape(-1, ss.shape[-1])
            flat_w = ww.reshape(-1, ww.shape[-1])
            cos_sw = F.cosine_similarity(flat_s, flat_w, dim=-1)
            rows.append({
                "head": h,
                "token_group": token_group,
                "rn_attention_mean": float(pp.mean()),
                "delta_norm_mean": float(dd.norm(dim=-1).mean()),
                "steal_norm_mean": float(ss.norm(dim=-1).mean()),
                "write_norm_mean": float(ww.norm(dim=-1).mean()),
                "steal_write_cos_mean": float(cos_sw.mean()),
                "formula_error_norm_mean": float(ee.mean()),
                "formula_rel_error": float(ee.mean() / dd.norm(dim=-1).mean().clamp_min(1e-12)),
            })

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    return rows


def pathway_decomposition(pair: dict) -> dict[str, float]:
    da = pair["delta_attn"].float()
    dm = pair["delta_mlp"].float()
    dp = pair["delta_post"].float()
    err = dp - (da + dm)
    def stats(x):
        return float(x.norm(dim=-1).mean())
    flat_a = da.reshape(-1, da.shape[-1])
    flat_m = dm.reshape(-1, dm.shape[-1])
    return {
        "attn_delta_norm_mean": stats(da),
        "mlp_response_delta_norm_mean": stats(dm),
        "post_delta_norm_mean": stats(dp),
        "attn_mlp_cos_mean": float(F.cosine_similarity(flat_a, flat_m, dim=-1).mean()),
        "reconstruction_error_norm_mean": stats(err),
    }


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
