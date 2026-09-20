from __future__ import annotations

import csv
import json
import math
import platform
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PrecisionDiagnosticsConfig:
    enabled: bool = True
    log_every_optimizer_steps: int = 100
    example_values_per_tensor: int = 12
    selected_parameter_limit: int = 48
    max_values_for_statistics: int = 262_144
    save_plots: bool = True


_FIXED_STAT_FIELDS = (
    "stage", "phase", "epoch", "optimizer_step", "micro_step", "kind", "name",
    "dtype", "shape", "numel", "sampled_numel", "finite_fraction", "zero_fraction",
    "nonzero_abs_min", "abs_max", "mean", "std", "rms", "p01", "p50", "p99",
    "below_fp16_normal_fraction", "above_fp16_max_fraction",
    "below_bf16_normal_fraction", "above_bf16_max_fraction",
    "fp16_cast_abs_error_mean", "fp16_cast_abs_error_max", "fp16_cast_rel_rms",
    "bf16_cast_abs_error_mean", "bf16_cast_abs_error_max", "bf16_cast_rel_rms",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _append_csv(path: Path, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), allow_nan=True) + "\n")


def _strided_sample(flat: torch.Tensor, limit: int) -> torch.Tensor:
    flat = flat.reshape(-1)
    if flat.numel() <= limit:
        return flat
    # Deterministic coverage over the whole tensor, avoiding RNG interaction with training.
    indices = torch.linspace(
        0, flat.numel() - 1, steps=limit, device=flat.device, dtype=torch.float64
    ).round().to(torch.long)
    return flat.index_select(0, indices)


def _example_indices(numel: int, count: int) -> list[int]:
    if numel <= 0 or count <= 0:
        return []
    if numel <= count:
        return list(range(numel))
    return np.linspace(0, numel - 1, num=count, dtype=np.int64).tolist()


def _roundtrip(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    try:
        return values.to(dtype).float()
    except RuntimeError:
        # Some CPU builds have incomplete half kernels; conversion itself is normally available,
        # but keep diagnostics non-fatal on unusual installations.
        return torch.full_like(values, float("nan"))


def tensor_precision_stats(
    tensor: torch.Tensor,
    *,
    max_values: int,
) -> Dict[str, Any]:
    detached = tensor.detach()
    original_dtype = str(detached.dtype)
    shape = list(detached.shape)
    numel = int(detached.numel())
    if not detached.is_floating_point() or numel == 0:
        return {
            "dtype": original_dtype,
            "shape": "x".join(map(str, shape)),
            "numel": numel,
            "sampled_numel": 0,
            **{field: float("nan") for field in _FIXED_STAT_FIELDS[11:]},
        }

    values = _strided_sample(detached.float(), max(1, int(max_values))).cpu()
    finite = torch.isfinite(values)
    finite_fraction = float(finite.float().mean()) if values.numel() else float("nan")
    values = values[finite]
    if values.numel() == 0:
        return {
            "dtype": original_dtype,
            "shape": "x".join(map(str, shape)),
            "numel": numel,
            "sampled_numel": 0,
            "finite_fraction": finite_fraction,
            **{field: float("nan") for field in _FIXED_STAT_FIELDS[12:]},
        }

    absolute = values.abs()
    nonzero = absolute[absolute > 0]
    fp16 = torch.finfo(torch.float16)
    bf16 = torch.finfo(torch.bfloat16)
    fp16_rt = _roundtrip(values, torch.float16)
    bf16_rt = _roundtrip(values, torch.bfloat16)

    def cast_metrics(roundtrip: torch.Tensor) -> tuple[float, float, float]:
        error = (roundtrip - values).abs()
        rms = float(values.square().mean().sqrt())
        return (
            float(error.mean()),
            float(error.max()),
            float(error.square().mean().sqrt() / max(rms, 1.0e-30)),
        )

    fp16_mean, fp16_max, fp16_rel = cast_metrics(fp16_rt)
    bf16_mean, bf16_max, bf16_rel = cast_metrics(bf16_rt)
    quantiles = torch.quantile(values, torch.tensor([0.01, 0.50, 0.99]))
    return {
        "dtype": original_dtype,
        "shape": "x".join(map(str, shape)),
        "numel": numel,
        "sampled_numel": int(values.numel()),
        "finite_fraction": finite_fraction,
        "zero_fraction": float((values == 0).float().mean()),
        "nonzero_abs_min": float(nonzero.min()) if nonzero.numel() else 0.0,
        "abs_max": float(absolute.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "rms": float(values.square().mean().sqrt()),
        "p01": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "below_fp16_normal_fraction": float(((absolute > 0) & (absolute < fp16.tiny)).float().mean()),
        "above_fp16_max_fraction": float((absolute > fp16.max).float().mean()),
        "below_bf16_normal_fraction": float(((absolute > 0) & (absolute < bf16.tiny)).float().mean()),
        "above_bf16_max_fraction": float((absolute > bf16.max).float().mean()),
        "fp16_cast_abs_error_mean": fp16_mean,
        "fp16_cast_abs_error_max": fp16_max,
        "fp16_cast_rel_rms": fp16_rel,
        "bf16_cast_abs_error_mean": bf16_mean,
        "bf16_cast_abs_error_max": bf16_max,
        "bf16_cast_rel_rms": bf16_rel,
    }


def tensor_numerical_examples(tensor: torch.Tensor, count: int) -> Dict[str, Any]:
    detached = tensor.detach()
    if not detached.is_floating_point() or detached.numel() == 0:
        return {"indices": [], "fp32": [], "fp16_roundtrip": [], "bf16_roundtrip": []}
    flat = detached.float().reshape(-1).cpu()
    indices = _example_indices(flat.numel(), count)
    values = flat[indices]
    return {
        "indices": indices,
        "fp32": values.tolist(),
        "fp16_roundtrip": _roundtrip(values, torch.float16).tolist(),
        "bf16_roundtrip": _roundtrip(values, torch.bfloat16).tolist(),
    }


def _parameter_priority(name: str, parameter: torch.Tensor) -> tuple[int, str]:
    lower = name.lower()
    if parameter.numel() <= 16:
        return (0, name)
    if lower in {"hard_text_embedding", "null_text_embedding", "logit_scale"}:
        return (1, name)
    if "trust_router" in lower or "source_head" in lower:
        return (2, name)
    if lower.startswith("read_implant."):
        return (3, name)
    if lower.endswith("visual.conv1.weight") or "positional_embedding" in lower:
        return (4, name)
    if "resblocks.0." in lower or "resblocks.23." in lower:
        return (5, name)
    return (9, name)


class PrecisionDiagnostics:
    """Low-overhead, deterministic numerical logging for mixed-precision training.

    Snapshots are taken only on optimizer steps. Activations, unscaled pre-clip
    gradients, FP32 parameters, actual sampled updates, GradScaler state, and
    FP16/BF16 round-trip examples are all written without changing training RNG.
    """

    def __init__(
        self,
        out_dir: Path | str,
        stage: str,
        config: PrecisionDiagnosticsConfig,
        *,
        requested_precision: str,
        resolved_precision: str,
        device: torch.device | str,
    ) -> None:
        self.stage = str(stage)
        self.config = config
        self.root = Path(out_dir) / "precision_diagnostics"
        self.tensor_csv = self.root / "precision_tensor_stats.csv"
        self.snapshot_csv = self.root / "precision_snapshot_log.csv"
        self.examples_jsonl = self.root / "precision_numerical_examples.jsonl"
        self.update_csv = self.root / "precision_parameter_updates.csv"
        self._hook_tensors: Dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self.device = torch.device(device)
        if not config.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        environment = {
            "stage": self.stage,
            "config": asdict(config),
            "requested_precision": requested_precision,
            "resolved_precision": resolved_precision,
            "device": str(self.device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "fp16": {
                "tiny_normal": torch.finfo(torch.float16).tiny,
                "max": torch.finfo(torch.float16).max,
                "eps": torch.finfo(torch.float16).eps,
            },
            "bf16": {
                "tiny_normal": torch.finfo(torch.bfloat16).tiny,
                "max": torch.finfo(torch.bfloat16).max,
                "eps": torch.finfo(torch.bfloat16).eps,
            },
            "fp32": {
                "tiny_normal": torch.finfo(torch.float32).tiny,
                "max": torch.finfo(torch.float32).max,
                "eps": torch.finfo(torch.float32).eps,
            },
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            environment.update({
                "cuda_device_name": torch.cuda.get_device_name(self.device),
                "cuda_capability": list(torch.cuda.get_device_capability(self.device)),
                "native_bf16": bool(torch.cuda.is_bf16_supported()),
            })
        (self.root / "precision_environment.json").write_text(
            json.dumps(_jsonable(environment), indent=2) + "\n", encoding="utf-8"
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def should_log(self, optimizer_step: int, *, force: bool = False) -> bool:
        if not self.enabled:
            return False
        return bool(
            force
            or optimizer_step == 1
            or optimizer_step % max(1, self.config.log_every_optimizer_steps) == 0
        )

    @contextmanager
    def capture_model_hooks(self, model: torch.nn.Module, active: bool):
        if not self.enabled or not active:
            yield {}
            return
        self._hook_tensors = {}
        handles: list[Any] = []

        def pre(name: str):
            def hook(_module, inputs):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    self._hook_tensors[name] = inputs[0].detach()
            return hook

        def post(name: str):
            def hook(_module, _inputs, output):
                if isinstance(output, torch.Tensor):
                    self._hook_tensors[name] = output.detach()
            return hook

        implant = getattr(model, "read_implant", None)

        def capture_layers(owner: Any, owner_name: str, layer_names: Sequence[str]) -> None:
            if owner is None:
                return
            for layer_name in layer_names:
                layer = getattr(owner, layer_name, None)
                if isinstance(layer, torch.nn.Module):
                    full_name = f"{owner_name}.{layer_name}"
                    handles.append(layer.register_forward_pre_hook(pre(f"{full_name}.input")))
                    handles.append(layer.register_forward_hook(post(f"{full_name}.output")))

        # Candidate-conditioned router and source detector (final architecture).
        capture_layers(getattr(implant, "trust_router", None), "trust_router", ("fc1", "fc2", "fc3"))
        capture_layers(
            getattr(implant, "source_head", None), "source_head",
            ("patch_expand", "patch_contract", "patch_out", "stats_fc1", "stats_fc2"),
        )

        # Reader/pool projection boundaries exist in both the isolated legacy
        # curriculum and the final model.  Their true input/output dtypes are
        # especially useful when deciding whether FP16 or BF16 autocast is safer.
        capture_layers(
            getattr(implant, "read_bridge", None), "read_bridge",
            ("q_proj", "k_proj", "v_proj", "out_proj"),
        )
        for pool_name in ("content_pool", "presence_pool", "orthographic_pool"):
            pool = getattr(implant, pool_name, None)
            capture_layers(pool, pool_name, ("k_proj", "v_proj", "out_proj"))
        try:
            yield self._hook_tensors
        finally:
            for handle in handles:
                handle.remove()

    def _selected_parameters(self, model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
        """Choose a deterministic but architecture-diverse parameter sample.

        A plain global sort was liable to fill the entire budget with tiny tap
        logits/scalars.  Quotas guarantee that final runs also include router,
        reader, Conv1/position, and early/late backbone tensors.
        """
        rows = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        rows.sort(key=lambda item: _parameter_priority(item[0], item[1]))
        limit = max(1, int(self.config.selected_parameter_limit))
        buckets: Dict[int, list[tuple[str, torch.nn.Parameter]]] = {}
        for item in rows:
            buckets.setdefault(_parameter_priority(item[0], item[1])[0], []).append(item)

        # Maximums, not minimums.  Any unused capacity is filled below.
        quotas = {0: 8, 1: 4, 2: 12, 3: 12, 4: 6, 5: 6}
        selected: list[tuple[str, torch.nn.Parameter]] = []
        selected_names: set[str] = set()
        for priority in (0, 1, 2, 3, 4, 5):
            for item in buckets.get(priority, [])[: min(quotas[priority], limit - len(selected))]:
                selected.append(item)
                selected_names.add(item[0])
            if len(selected) >= limit:
                return selected

        # Fill with the remaining parameters in the original deterministic order.
        for item in rows:
            if item[0] in selected_names:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def capture_before_update(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        optimizer_step: int,
        micro_step: int,
        epoch: int,
        phase: str,
        tensors: Mapping[str, torch.Tensor],
        hook_tensors: Optional[Mapping[str, torch.Tensor]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        metadata = {
            "stage": self.stage,
            "phase": str(phase),
            "epoch": int(epoch),
            "optimizer_step": int(optimizer_step),
            "micro_step": int(micro_step),
        }
        all_tensors: Dict[str, tuple[str, torch.Tensor]] = {
            str(name): ("activation", value)
            for name, value in tensors.items()
            if isinstance(value, torch.Tensor)
        }
        for name, value in (hook_tensors or {}).items():
            if isinstance(value, torch.Tensor):
                all_tensors[str(name)] = ("hook", value)

        selected = self._selected_parameters(model)
        parameter_samples: Dict[str, Any] = {}
        for name, parameter in selected:
            all_tensors[f"parameter.{name}"] = ("parameter", parameter)
            if parameter.grad is not None:
                all_tensors[f"gradient.{name}"] = ("gradient_unscaled_preclip", parameter.grad)
            # Optimizer moments are expected to remain FP32; logging them makes
            # that policy visible and exposes moment underflow/dynamic-range issues.
            state = optimizer.state.get(parameter, {})
            for state_name, state_value in state.items():
                if isinstance(state_value, torch.Tensor):
                    all_tensors[f"optimizer_state.{name}.{state_name}"] = (
                        "optimizer_state", state_value
                    )
            indices = _example_indices(parameter.numel(), self.config.example_values_per_tensor)
            if indices:
                flat = parameter.detach().float().reshape(-1)
                parameter_samples[name] = {
                    "indices": indices,
                    "before": flat[indices].cpu().tolist(),
                }

        for name, (kind, tensor) in all_tensors.items():
            stats = tensor_precision_stats(
                tensor, max_values=self.config.max_values_for_statistics
            )
            row = {**metadata, "kind": kind, "name": name, **stats}
            _append_csv(self.tensor_csv, row, _FIXED_STAT_FIELDS)
            examples = tensor_numerical_examples(
                tensor, self.config.example_values_per_tensor
            )
            _append_jsonl(self.examples_jsonl, {
                **metadata,
                "kind": kind,
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                **examples,
            })

        lrs = [float(group.get("lr", float("nan"))) for group in optimizer.param_groups]
        scaler_enabled = bool(getattr(scaler, "is_enabled", lambda: False)()) if scaler is not None else False
        scaler_scale = (
            float(scaler.get_scale()) if scaler is not None and hasattr(scaler, "get_scale") else 1.0
        )
        snapshot_row = {
            **metadata,
            "timestamp": time.time(),
            "scaler_enabled": scaler_enabled,
            "scaler_scale": scaler_scale,
            "lr_min": min(lrs) if lrs else float("nan"),
            "lr_max": max(lrs) if lrs else float("nan"),
            "trainable_parameter_tensors": sum(1 for p in model.parameters() if p.requires_grad),
            "selected_parameter_tensors": len(selected),
            "extra_json": json.dumps(_jsonable(extra or {}), allow_nan=True),
        }
        _append_csv(
            self.snapshot_csv,
            snapshot_row,
            (
                "stage", "phase", "epoch", "optimizer_step", "micro_step", "timestamp",
                "scaler_enabled", "scaler_scale", "lr_min", "lr_max",
                "trainable_parameter_tensors", "selected_parameter_tensors", "extra_json",
            ),
        )
        return {"metadata": metadata, "parameter_samples": parameter_samples}

    def capture_after_update(
        self,
        snapshot: Mapping[str, Any],
        model: torch.nn.Module,
    ) -> None:
        if not self.enabled or not snapshot:
            return
        named = dict(model.named_parameters())
        metadata = dict(snapshot.get("metadata", {}))
        for name, record in snapshot.get("parameter_samples", {}).items():
            parameter = named.get(name)
            if parameter is None:
                continue
            indices = list(record["indices"])
            before = torch.tensor(record["before"], dtype=torch.float32)
            after = parameter.detach().float().reshape(-1)[indices].cpu()
            delta = after - before
            abs_before = before.abs()
            row = {
                **metadata,
                "name": name,
                "sample_count": len(indices),
                "update_nonzero_fraction": float((delta != 0).float().mean()) if delta.numel() else float("nan"),
                "update_abs_mean": float(delta.abs().mean()) if delta.numel() else float("nan"),
                "update_abs_max": float(delta.abs().max()) if delta.numel() else float("nan"),
                "update_rms": float(delta.square().mean().sqrt()) if delta.numel() else float("nan"),
                "parameter_rms_before": float(before.square().mean().sqrt()) if before.numel() else float("nan"),
                "update_to_parameter_rms": float(
                    delta.square().mean().sqrt() / max(float(before.square().mean().sqrt()), 1.0e-30)
                ) if delta.numel() else float("nan"),
                "updates_below_fp16_normal_fraction": float(
                    ((delta.abs() > 0) & (delta.abs() < torch.finfo(torch.float16).tiny)).float().mean()
                ) if delta.numel() else float("nan"),
                "updates_lost_if_parameter_fp16_fraction": float(
                    (_roundtrip(before + delta, torch.float16) == _roundtrip(before, torch.float16)).float().mean()
                ) if delta.numel() else float("nan"),
                "updates_lost_if_parameter_bf16_fraction": float(
                    (_roundtrip(before + delta, torch.bfloat16) == _roundtrip(before, torch.bfloat16)).float().mean()
                ) if delta.numel() else float("nan"),
            }
            _append_csv(
                self.update_csv,
                row,
                (
                    "stage", "phase", "epoch", "optimizer_step", "micro_step", "name",
                    "sample_count", "update_nonzero_fraction", "update_abs_mean", "update_abs_max",
                    "update_rms", "parameter_rms_before", "update_to_parameter_rms",
                    "updates_below_fp16_normal_fraction", "updates_lost_if_parameter_fp16_fraction",
                    "updates_lost_if_parameter_bf16_fraction",
                ),
            )
            _append_jsonl(self.examples_jsonl, {
                **metadata,
                "kind": "actual_parameter_update",
                "name": name,
                "indices": indices,
                "before_fp32": before.tolist(),
                "after_fp32": after.tolist(),
                "delta_fp32": delta.tolist(),
                "before_fp16_roundtrip": _roundtrip(before, torch.float16).tolist(),
                "after_fp16_roundtrip": _roundtrip(after, torch.float16).tolist(),
                "before_bf16_roundtrip": _roundtrip(before, torch.bfloat16).tolist(),
                "after_bf16_roundtrip": _roundtrip(after, torch.bfloat16).tolist(),
            })

    def finalize(self) -> None:
        if not self.enabled or not self.config.save_plots:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            (self.root / "plot_error.txt").write_text(str(exc), encoding="utf-8")
            return
        if not self.tensor_csv.exists():
            return
        with self.tensor_csv.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        plot_dir = self.root / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        def plot_metric(metric: str, filename: str, title: str, ylabel: str, kinds: set[str]):
            by_name: Dict[str, list[tuple[int, float]]] = {}
            for row in rows:
                if row.get("kind") not in kinds:
                    continue
                try:
                    value = float(row[metric])
                    step = int(float(row["optimizer_step"]))
                except (ValueError, KeyError):
                    continue
                if math.isfinite(value):
                    by_name.setdefault(row["name"], []).append((step, value))
            ranked = sorted(
                by_name.items(), key=lambda item: max((abs(v) for _, v in item[1]), default=0.0),
                reverse=True,
            )[:16]
            if not ranked:
                return
            fig, ax = plt.subplots(figsize=(13, 7))
            for name, values in ranked:
                values.sort()
                ax.plot([x for x, _ in values], [y for _, y in values], label=name[-70:])
            ax.set_xlabel("optimizer step")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=170)
            plt.close(fig)

        plot_metric(
            "rms", "gradient_rms_selected.png", "Selected unscaled pre-clip gradient RMS",
            "gradient RMS", {"gradient_unscaled_preclip"},
        )
        plot_metric(
            "below_fp16_normal_fraction", "gradient_fp16_underflow_risk.png",
            "Fraction of selected gradient values below FP16 normal range",
            "fraction", {"gradient_unscaled_preclip"},
        )
        plot_metric(
            "bf16_cast_rel_rms", "activation_bf16_roundtrip_error.png",
            "BF16 round-trip relative RMS error for critical activations",
            "relative RMS error", {"activation", "hook"},
        )
        plot_metric(
            "fp16_cast_rel_rms", "activation_fp16_roundtrip_error.png",
            "FP16 round-trip relative RMS error for critical activations",
            "relative RMS error", {"activation", "hook"},
        )
