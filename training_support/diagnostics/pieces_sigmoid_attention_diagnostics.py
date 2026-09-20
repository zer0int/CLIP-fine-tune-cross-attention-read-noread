"""Logging-only diagnostics for raw independent-sigmoid PIECES attention.

IMPORTANT: normalized distributions created here exist ONLY to compute an entropy
shape diagnostic. They are never returned to the model and never participate in
training. The actual PIECES attention remains raw independent sigmoid weights.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import torch


def _safe_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    x = x.detach().float()
    if mask is not None:
        mask = mask.to(device=x.device, dtype=torch.bool)
        while mask.ndim < x.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(x)
        if not bool(mask.any()):
            return float("nan")
        x = x[mask]
    if x.numel() == 0:
        return float("nan")
    return float(x.mean())


def _mass_entropy(weights: torch.Tensor):
    """Return raw total mass and normalized-shape entropy per head/query.

    `weights` are raw sigmoid edges. The division below is diagnostic ONLY.
    Entropy is normalized to [0,1] by log(number of nonzero eligible sources).
    """
    w = weights.detach().float().clamp_min(0.0)
    mass = w.sum(dim=-1)
    p = w / mass.unsqueeze(-1).clamp_min(1.0e-12)
    entropy = -(p * p.clamp_min(1.0e-12).log()).sum(dim=-1)
    count = (w > 0).sum(dim=-1).clamp(min=1).float()
    denom = count.log()
    entropy_norm = torch.where(denom > 0, entropy / denom, torch.zeros_like(entropy))
    return mass, entropy_norm


def _add_candidate_head_stats(
    out: Dict[str, float],
    prefix: str,
    weights: torch.Tensor,  # [B,N,H,S]
    positive: Optional[torch.Tensor],  # [B,N]
) -> None:
    mass, ent = _mass_entropy(weights)
    H = int(weights.shape[-2])
    for h in range(H):
        out[f"{prefix}_h{h}_mass_all"] = _safe_mean(mass[..., h])
        out[f"{prefix}_h{h}_entropy_norm_all"] = _safe_mean(ent[..., h])
        if positive is not None and tuple(positive.shape) == tuple(mass.shape[:2]):
            out[f"{prefix}_h{h}_mass_supported"] = _safe_mean(mass[..., h], positive)
            out[f"{prefix}_h{h}_mass_unsupported"] = _safe_mean(mass[..., h], ~positive)
            out[f"{prefix}_h{h}_entropy_norm_supported"] = _safe_mean(ent[..., h], positive)
            out[f"{prefix}_h{h}_entropy_norm_unsupported"] = _safe_mean(ent[..., h], ~positive)
    out[f"{prefix}_mass_mean"] = _safe_mean(mass)
    out[f"{prefix}_entropy_norm_mean"] = _safe_mean(ent)


def _add_candidate_scalar_edge_stats(
    out: Dict[str, float],
    prefix: str,
    edge: torch.Tensor,  # [B,N,H]
    positive: Optional[torch.Tensor],
) -> None:
    e = edge.detach().float()
    H = int(e.shape[-1])
    for h in range(H):
        out[f"{prefix}_h{h}_mean_all"] = _safe_mean(e[..., h])
        if positive is not None and tuple(positive.shape) == tuple(e.shape[:2]):
            out[f"{prefix}_h{h}_mean_supported"] = _safe_mean(e[..., h], positive)
            out[f"{prefix}_h{h}_mean_unsupported"] = _safe_mean(e[..., h], ~positive)


def _add_content_head_stats(
    out: Dict[str, float],
    prefix: str,
    weights: torch.Tensor,  # [B,H,S]
    present_targets: Optional[torch.Tensor],
    readable_targets: Optional[torch.Tensor],
) -> None:
    mass, ent = _mass_entropy(weights)
    H = int(weights.shape[-2])
    categories = {}
    if present_targets is not None:
        p = present_targets.detach().to(device=mass.device)
        known_p = p >= 0
        categories["no_text"] = known_p & (p < 0.5)
        if readable_targets is not None:
            r = readable_targets.detach().to(device=mass.device)
            known_r = r >= 0
            categories["readable_text"] = known_p & (p >= 0.5) & known_r & (r >= 0.5)
            categories["unreadable_text"] = known_p & (p >= 0.5) & known_r & (r < 0.5)
    for h in range(H):
        out[f"{prefix}_h{h}_mass_all"] = _safe_mean(mass[:, h])
        out[f"{prefix}_h{h}_entropy_norm_all"] = _safe_mean(ent[:, h])
        for name, mask in categories.items():
            out[f"{prefix}_h{h}_mass_{name}"] = _safe_mean(mass[:, h], mask)
            out[f"{prefix}_h{h}_entropy_norm_{name}"] = _safe_mean(ent[:, h], mask)
    out[f"{prefix}_mass_mean"] = _safe_mean(mass)
    out[f"{prefix}_entropy_norm_mean"] = _safe_mean(ent)


def collect_pieces_sigmoid_attention_metrics(
    model: torch.nn.Module,
    details: Mapping[str, Any],
    positive: Optional[torch.Tensor] = None,
    present_targets: Optional[torch.Tensor] = None,
    readable_targets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Collect logging-only mass/entropy diagnostics for `sigmoid_all` PIECES."""
    if str(getattr(model, "read_attention_architecture", "softmax")) != "sigmoid_all":
        return {}

    out: Dict[str, float] = {}
    read_candidate_positive = None
    if positive is not None:
        no_text_mode = details.get("no_text_mode_mask")
        if isinstance(no_text_mode, torch.Tensor) and positive.ndim == 2:
            read_candidate_positive = positive[:, ~no_text_mode.to(device=positive.device, dtype=torch.bool)]

    read_details = details.get("read_details")
    if isinstance(read_details, Mapping):
        per_block = read_details.get("per_block", {})
        if isinstance(per_block, Mapping):
            for block, block_details in per_block.items():
                if not isinstance(block_details, Mapping):
                    continue
                patch = block_details.get("patch_attention")
                if isinstance(patch, torch.Tensor):
                    _add_candidate_head_stats(
                        out, f"pieces_sig_read_b{int(block)}_patch", patch, read_candidate_positive
                    )
                reg = block_details.get("register_attention")
                if isinstance(reg, torch.Tensor):
                    _add_candidate_head_stats(
                        out, f"pieces_sig_read_b{int(block)}_register", reg, read_candidate_positive
                    )
                rn = block_details.get("read_null_attention")
                if isinstance(rn, torch.Tensor):
                    _add_candidate_scalar_edge_stats(
                        out, f"pieces_sig_read_b{int(block)}_read_null", rn, read_candidate_positive
                    )

    ortho_details = details.get("orthographic_details")
    if isinstance(ortho_details, Mapping):
        per_block = ortho_details.get("per_block_attention", {})
        if isinstance(per_block, Mapping):
            for block, weights in per_block.items():
                if isinstance(weights, torch.Tensor):
                    _add_candidate_head_stats(
                        out, f"pieces_sig_ortho_b{int(block)}", weights, read_candidate_positive
                    )

    content_details = details.get("content_details")
    if isinstance(content_details, Mapping):
        per_block = content_details.get("per_block_attention", {})
        if isinstance(per_block, Mapping):
            for block, weights in per_block.items():
                if isinstance(weights, torch.Tensor):
                    _add_content_head_stats(
                        out,
                        f"pieces_sig_content_b{int(block)}",
                        weights,
                        present_targets,
                        readable_targets,
                    )

    implant = getattr(model, "read_implant", None)
    if implant is not None:
        for component_name in ("read_bridge", "orthographic_bridge", "content_pool"):
            component = getattr(implant, component_name, None)
            if component is None:
                continue
            for bias_name in (
                "sigmoid_head_bias",
                "sigmoid_patch_head_bias",
                "sigmoid_register_head_bias",
            ):
                bias = getattr(component, bias_name, None)
                if isinstance(bias, torch.Tensor):
                    flat = bias.detach().float().reshape(-1)
                    for h, value in enumerate(flat):
                        out[f"pieces_sig_bias_{component_name}_{bias_name}_h{h}"] = float(value)

    # Compact console-friendly rollups. These are also useful for plots.
    def average_matching(fragment: str) -> float:
        vals = [v for k, v in out.items() if fragment in k and k.endswith("_mass_all") and math.isfinite(v)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    out["pieces_sig_read_mass_rollup"] = average_matching("pieces_sig_read_")
    out["pieces_sig_ortho_mass_rollup"] = average_matching("pieces_sig_ortho_")
    out["pieces_sig_content_mass_rollup"] = average_matching("pieces_sig_content_")
    return out
