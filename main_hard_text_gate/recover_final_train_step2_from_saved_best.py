#!/usr/bin/env python
"""Finalize the all-weights stage from its already-saved best phase checkpoint.

This is intended for runs that completed training but failed while restoring the
wrapped ``phase_<name>_best_merged_state_dict.pt`` checkpoint before final export.
No training is repeated and no model tensors are modified.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


def torch_load_trusted(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def checkpoint_architecture(checkpoint: Mapping[str, Any]) -> Optional[str]:
    explicit = checkpoint.get("read_attention_architecture")
    metadata = checkpoint.get("metadata")
    if explicit is None and isinstance(metadata, Mapping):
        explicit = metadata.get("read_attention_architecture")
    args_metadata = checkpoint.get("args")
    if explicit is None and isinstance(args_metadata, Mapping):
        explicit = args_metadata.get("read_attention_architecture")
    return str(explicit) if explicit is not None else None


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def recover(out_dir: Path, phase: str, overwrite: bool) -> None:
    best_compact_path = out_dir / f"phase_{phase}_best.pt"
    best_full_path = out_dir / f"phase_{phase}_best_merged_state_dict.pt"
    final_compact_path = out_dir / "stage1_complete.pt"
    final_full_path = out_dir / "stage1_complete_merged_state_dict.pt"

    for path in (best_compact_path, best_full_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved best checkpoint is missing: {path}")
    if not overwrite:
        existing = [path for path in (final_compact_path, final_full_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Final output already exists; pass --overwrite only after confirming it "
                f"may be replaced: {existing}"
            )

    compact = torch_load_trusted(best_compact_path)
    full = torch_load_trusted(best_full_path)
    if not isinstance(compact, Mapping):
        raise RuntimeError(f"Compact best checkpoint is not a mapping: {best_compact_path}")
    if not isinstance(full, Mapping) or not isinstance(full.get("state_dict"), Mapping):
        raise RuntimeError(
            "Full best checkpoint does not use the expected wrapped state_dict format: "
            f"{best_full_path}"
        )
    state_dict = full["state_dict"]
    if not state_dict or not all(torch.is_tensor(value) for value in state_dict.values()):
        raise RuntimeError(f"Full best state_dict is empty or contains non-tensors: {best_full_path}")

    compact_architecture = checkpoint_architecture(compact)
    full_architecture = checkpoint_architecture(full)
    if (
        compact_architecture is not None
        and full_architecture is not None
        and compact_architecture != full_architecture
    ):
        raise ValueError(
            "Compact and full best checkpoints disagree on read-attention architecture: "
            f"compact={compact_architecture!r}, full={full_architecture!r}"
        )

    complete = dict(compact)
    complete["phase"] = "complete"
    complete["epoch"] = 0
    complete["recovered_from_phase"] = str(phase)
    complete["recovered_from_full_checkpoint"] = str(best_full_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(complete, final_compact_path)
    # The best merged checkpoint is already the exact complete-model export format;
    # byte-copy it so recovery cannot alter or round any model tensor.
    atomic_copy(best_full_path, final_full_path)

    print(f"[recover] compact -> {final_compact_path}")
    print(f"[recover] full model -> {final_full_path}")
    print(f"[recover] tensors={len(state_dict)} architecture={full_architecture or 'unknown'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/clip_xattn_training/final/sigmoid_all/all_weights"),
    )
    parser.add_argument("--phase", default="1c")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    recover(args.out_dir, args.phase, args.overwrite)


if __name__ == "__main__":
    main()
