"""Mixed-precision helpers used by the PIECES bridge and router."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch


def fp32_island(reference: torch.Tensor | torch.device | str | None = None):
    """Disable autocast on the device associated with ``reference``."""
    if reference is None:
        return nullcontext()
    device_type = (
        reference.device.type
        if isinstance(reference, torch.Tensor)
        else torch.device(reference).type
    )
    if device_type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def to_fp32_tree(value: Any) -> Any:
    """Recursively cast floating tensors to FP32 without moving devices."""
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(to_fp32_tree(item) for item in value)
    if isinstance(value, list):
        return [to_fp32_tree(item) for item in value]
    if isinstance(value, Mapping):
        return type(value)((key, to_fp32_tree(item)) for key, item in value.items())
    return value
