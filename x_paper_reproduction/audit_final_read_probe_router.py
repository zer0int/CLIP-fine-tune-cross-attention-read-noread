#!/usr/bin/env python3
"""
Audit the final PIECES / x-attn HF checkpoint for the optional READ probe and
trust-router input width WITHOUT loading the model or executing remote code.

Default target:
    alpha_0_28

What this checks
----------------
1. config.json and WISE_FT_METADATA.json (if present)
2. all *.safetensors shards in the model directory
3. exact checkpoint tensors ending in:
       read_implant.read_probe
       read_implant.trust_router.fc1.weight / bias
       read_implant.auto_read_scale
       read_implant.read_calibration_scale
       read_implant.null_abstain_weight
       read_implant.glyph_bias_beta
       read_implant.read_tap_logits
4. local modeling_*.py source, if present, for the expected router features
       injection
       injection * tanh(relative / 20)

The central question is whether read_probe is exactly zero. If it is, then
injection_feature(content_image) is identically zero and the two optional
probe-derived router inputs are dead in the released checkpoint (assuming the
local modeling code uses the expected feature definition, which this script
also checks textually).

No datasets, GPU, transformers, or trust_remote_code are required.
Requires: safetensors, torch
"""

from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


DEFAULT_MODEL = Path("zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX")
DEFAULT_JSON = Path("alpha_0_28_read_probe_audit.json")
DEFAULT_TXT = Path("alpha_0_28_read_probe_audit.txt")

TARGET_SUFFIXES = (
    "read_implant.read_probe",
    "read_implant.trust_router.fc1.weight",
    "read_implant.trust_router.fc1.bias",
    "read_implant.auto_read_scale",
    "read_implant.read_calibration_scale",
    "read_implant.null_abstain_weight",
    "read_implant.glyph_bias_beta",
    "read_implant.read_tap_logits",
)


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach().float().cpu()
    flat = x.reshape(-1)
    if flat.numel() == 0:
        return {
            "shape": list(x.shape),
            "numel": 0,
            "dtype_in_checkpoint": str(t.dtype),
        }
    abs_x = flat.abs()
    return {
        "shape": list(x.shape),
        "numel": int(flat.numel()),
        "dtype_in_checkpoint": str(t.dtype),
        "l2_norm": float(torch.linalg.vector_norm(flat).item()),
        "max_abs": float(abs_x.max().item()),
        "mean_abs": float(abs_x.mean().item()),
        "exact_nonzero": int(torch.count_nonzero(flat).item()),
        "finite": bool(torch.isfinite(flat).all().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
    }


def find_weight_files(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        files = sorted(model_dir.rglob("*.safetensors"))
    return files


def scan_safetensors(model_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    weight_files = find_weight_files(model_dir)
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files found under: {model_dir}")

    selected: dict[str, dict[str, Any]] = {}
    all_matching_keys: list[str] = []
    files_scanned: list[str] = []

    for path in weight_files:
        files_scanned.append(str(path))
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                if any(key.endswith(suffix) for suffix in TARGET_SUFFIXES):
                    all_matching_keys.append(key)
                    t = f.get_tensor(key)
                    selected[key] = {
                        "file": str(path),
                        **tensor_stats(t),
                    }

    return selected, sorted(set(all_matching_keys)), files_scanned


def suffix_lookup(selected: dict[str, dict[str, Any]], suffix: str) -> list[tuple[str, dict[str, Any]]]:
    return sorted((k, v) for k, v in selected.items() if k.endswith(suffix))


def source_router_audit(model_dir: Path) -> dict[str, Any]:
    candidates = sorted(model_dir.glob("modeling_*.py")) + sorted(model_dir.glob("*.py"))
    # De-duplicate while preserving order.
    unique: list[Path] = []
    seen = set()
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    evidence: list[dict[str, Any]] = []
    for path in unique:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "trust_router" not in text and "CandidateTrustRouter" not in text:
            continue

        low = text.replace(" ", "").replace("\n", "")
        has_injection_feature = "injection_feature" in text
        has_inject_relative = (
            "inject*torch.tanh(relative/20.0)" in low
            or "injection*torch.tanh(relative/20.0)" in low
            or "inject*torch.tanh(relative/20)" in low
            or "injection*torch.tanh(relative/20)" in low
        )
        has_read_probe = "read_probe" in text

        # Collect a compact local snippet around the most informative occurrence.
        lines = text.splitlines()
        hit_lines = []
        for i, line in enumerate(lines):
            if "tanh(relative" in line.replace(" ", "") or "read_probe" in line or "injection_feature" in line:
                hit_lines.append(i)
        snippet = ""
        if hit_lines:
            i = hit_lines[0]
            lo_i = max(0, i - 6)
            hi_i = min(len(lines), i + 10)
            snippet = "\n".join(f"{j+1}: {lines[j]}" for j in range(lo_i, hi_i))

        evidence.append({
            "file": str(path),
            "has_read_probe_symbol": has_read_probe,
            "has_injection_feature_symbol": has_injection_feature,
            "has_injection_times_relative_tanh_feature": has_inject_relative,
            "snippet": snippet,
        })

    return {
        "files_checked": [str(p) for p in unique],
        "router_source_evidence": evidence,
    }


def interesting_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    keep = {}
    keywords = (
        "read", "null", "bridge", "router", "source", "ortho", "content",
        "tap", "register", "alpha", "wise", "attention",
    )
    for k, v in config.items():
        if any(word in str(k).lower() for word in keywords):
            keep[k] = v
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("model_dir", nargs="?", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--text-out", type=Path, default=DEFAULT_TXT)
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    config = read_json(model_dir / "config.json")
    wise = read_json(model_dir / "WISE_FT_METADATA.json")
    selected, matching_keys, files_scanned = scan_safetensors(model_dir)
    source = source_router_audit(model_dir)

    probe_hits = suffix_lookup(selected, "read_implant.read_probe")
    fc1_hits = suffix_lookup(selected, "read_implant.trust_router.fc1.weight")

    if len(probe_hits) == 1:
        probe_key, probe_stats = probe_hits[0]
        probe_exact_zero = probe_stats.get("exact_nonzero") == 0
    else:
        probe_key, probe_stats, probe_exact_zero = None, None, None

    if len(fc1_hits) == 1:
        fc1_key, fc1_stats = fc1_hits[0]
        shape = fc1_stats.get("shape", [])
        router_input_width = int(shape[1]) if len(shape) == 2 else None
    else:
        fc1_key, fc1_stats, router_input_width = None, None, None

    source_confirms_optional_pair = any(
        row.get("has_injection_times_relative_tanh_feature", False)
        for row in source["router_source_evidence"]
    )

    if probe_exact_zero is True and source_confirms_optional_pair:
        verdict = (
            "READ_PROBE_IS_EXACTLY_ZERO: the optional injection feature and its "
            "injection×relative-READ companion are identically zero in this checkpoint."
        )
    elif probe_exact_zero is True:
        verdict = (
            "READ_PROBE_IS_EXACTLY_ZERO. The local source scan did not independently "
            "confirm the expected injection×relative feature formula, so report the "
            "zero probe fact separately from router-slot interpretation."
        )
    elif probe_exact_zero is False:
        verdict = (
            "READ_PROBE_IS_ACTIVE_NONZERO: do NOT describe the final router's optional "
            "probe-derived inputs as disabled."
        )
    else:
        verdict = "READ_PROBE_NOT_UNIQUELY_FOUND: inspect matching keys / checkpoint layout."

    report = {
        "model_dir": str(model_dir),
        "config_file_present": (model_dir / "config.json").is_file(),
        "wise_metadata_present": (model_dir / "WISE_FT_METADATA.json").is_file(),
        "config_interesting_fields": interesting_config(config),
        "wise_metadata": wise,
        "safetensors_files_scanned": files_scanned,
        "matching_tensor_keys": matching_keys,
        "selected_tensor_stats": selected,
        "read_probe": {
            "key": probe_key,
            "stats": probe_stats,
            "exactly_zero": probe_exact_zero,
        },
        "trust_router_fc1": {
            "key": fc1_key,
            "stats": fc1_stats,
            "input_width": router_input_width,
        },
        "source_code_audit": source,
        "source_confirms_injection_times_relative_feature": source_confirms_optional_pair,
        "verdict": verdict,
    }

    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = []
    lines.append("FINAL READ-PROBE / TRUST-ROUTER AUDIT")
    lines.append("=" * 72)
    lines.append(f"model_dir: {model_dir}")
    if isinstance(wise, dict):
        lines.append(f"WiSE alpha: {wise.get('alpha', '<missing>')}")
        names = wise.get("interpolated_parameter_names", [])
        lines.append(f"WiSE interpolated tensors: {len(names) if isinstance(names, list) else '<unknown>'}")
        if isinstance(names, list):
            probe_wised = any(str(x).endswith("read_implant.read_probe") for x in names)
            router_wised = any("trust_router" in str(x) for x in names)
            lines.append(f"read_probe interpolated by WiSE: {probe_wised}")
            lines.append(f"trust_router interpolated by WiSE: {router_wised}")
    lines.append("")
    lines.append(f"read_probe key: {probe_key}")
    if probe_stats:
        lines.append(f"read_probe shape: {probe_stats['shape']}")
        lines.append(f"read_probe L2: {probe_stats['l2_norm']:.12g}")
        lines.append(f"read_probe max|x|: {probe_stats['max_abs']:.12g}")
        lines.append(f"read_probe exact_nonzero: {probe_stats['exact_nonzero']} / {probe_stats['numel']}")
        lines.append(f"read_probe EXACT ZERO: {probe_exact_zero}")
    lines.append("")
    lines.append(f"trust_router.fc1 key: {fc1_key}")
    if fc1_stats:
        lines.append(f"trust_router.fc1 shape: {fc1_stats['shape']}")
    lines.append(f"router input width: {router_input_width}")
    lines.append(f"local code confirms injection×tanh(relative/20) slot: {source_confirms_optional_pair}")
    lines.append("")
    lines.append("VERDICT")
    lines.append(verdict)
    lines.append("")
    lines.append(f"JSON report: {args.json_out}")

    text = "\n".join(lines) + "\n"
    args.text_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
