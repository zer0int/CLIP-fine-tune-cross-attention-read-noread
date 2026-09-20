from __future__ import annotations

from contextlib import nullcontext
from functools import wraps
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch


AMP_CHOICES = ("auto", "bf16", "fp16", "none")


@dataclass(frozen=True)
class ResolvedPrecision:
    requested: str
    resolved: str
    native_bf16: bool
    autocast_dtype: torch.dtype | None
    grad_scaler_enabled: bool

    @property
    def autocast_enabled(self) -> bool:
        return self.autocast_dtype is not None


def native_bf16_supported(device: torch.device | str) -> bool:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return bool(torch.cuda.is_bf16_supported())


def resolve_precision(requested: str, device: torch.device | str) -> ResolvedPrecision:
    requested = str(requested).lower()
    if requested not in AMP_CHOICES:
        raise ValueError(f"Unknown amp dtype {requested!r}; expected one of {AMP_CHOICES}")

    device = torch.device(device)
    native_bf16 = native_bf16_supported(device)

    if device.type != "cuda":
        if requested in {"bf16", "fp16"}:
            raise RuntimeError(
                f"Explicit {requested} CUDA autocast requested on non-CUDA device {device}"
            )
        return ResolvedPrecision(
            requested=requested,
            resolved="none",
            native_bf16=native_bf16,
            autocast_dtype=None,
            grad_scaler_enabled=False,
        )

    if requested == "auto":
        resolved = "bf16" if native_bf16 else "fp16"
    elif requested == "bf16":
        if not native_bf16:
            name = torch.cuda.get_device_name(device)
            raise RuntimeError(
                f"Native BF16 is unavailable on {name}; use amp_dtype='auto', 'fp16', or 'none'"
            )
        resolved = "bf16"
    else:
        resolved = requested

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "none": None,
    }[resolved]
    return ResolvedPrecision(
        requested=requested,
        resolved=resolved,
        native_bf16=native_bf16,
        autocast_dtype=dtype,
        grad_scaler_enabled=(resolved == "fp16"),
    )


def autocast_context(device: torch.device | str, resolved: ResolvedPrecision | str):
    device = torch.device(device)
    if isinstance(resolved, str):
        if resolved == "auto":
            resolved = resolve_precision(resolved, device)
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[resolved]
            resolved = ResolvedPrecision(
                requested=resolved, resolved=resolved, native_bf16=False,
                autocast_dtype=dtype, grad_scaler_enabled=(resolved == "fp16"),
            )
    if device.type != "cuda" or not resolved.autocast_enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=resolved.autocast_dtype)


def make_grad_scaler(device: torch.device | str, resolved: ResolvedPrecision | str):
    device = torch.device(device)
    if isinstance(resolved, str):
        if resolved == "auto":
            resolved = resolve_precision(resolved, device)
        else:
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[resolved]
            resolved = ResolvedPrecision(
                requested=resolved, resolved=resolved, native_bf16=False,
                autocast_dtype=dtype, grad_scaler_enabled=(resolved == "fp16"),
            )
    enabled = device.type == "cuda" and resolved.grad_scaler_enabled
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def fp32_island(reference: torch.Tensor | torch.device | str | None = None):
    if reference is None:
        return nullcontext()
    if isinstance(reference, torch.Tensor):
        device_type = reference.device.type
    else:
        device_type = torch.device(reference).type
    if device_type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def _first_floating_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value if value.is_floating_point() else None
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_floating_tensor(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_floating_tensor(item)
            if found is not None:
                return found
    return None


def fp32_function(function):
    """Disable autocast and cast floating tensor arguments to FP32."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        reference = _first_floating_tensor((args, kwargs))
        with fp32_island(reference):
            return function(*to_fp32_tree(args), **to_fp32_tree(kwargs))
    return wrapped


def to_fp32_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(to_fp32_tree(item) for item in value)
    if isinstance(value, list):
        return [to_fp32_tree(item) for item in value]
    if isinstance(value, Mapping):
        return type(value)((key, to_fp32_tree(item)) for key, item in value.items())
    return value


def iter_trainable_parameters(module: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Parameter]]:
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            yield name, parameter


def assert_trainable_parameters_fp32(module: torch.nn.Module, *, label: str = "model") -> None:
    bad = [
        f"{name}:{parameter.dtype}"
        for name, parameter in iter_trainable_parameters(module)
        if parameter.dtype != torch.float32
    ]
    if bad:
        preview = ", ".join(bad[:20])
        suffix = " ..." if len(bad) > 20 else ""
        raise RuntimeError(
            f"{label} has trainable parameters outside FP32: {preview}{suffix}"
        )


def precision_summary(resolved: ResolvedPrecision) -> str:
    fallback = " (FP16 fallback + GradScaler)" if resolved.resolved == "fp16" else ""
    return (
        f"requested={resolved.requested} resolved={resolved.resolved} "
        f"native_bf16={resolved.native_bf16} trainable_params=fp32 "
        f"custom_implant=fp32{fallback}"
    )
