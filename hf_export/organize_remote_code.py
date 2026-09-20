#!/usr/bin/env python3
"""Move legacy Hugging Face remote-code files out of the repository root.

Dry-run by default.  With ``--apply``:
- exact duplicate root copies are removed;
- differing legacy root copies are moved to ``hf_export/legacy_root_remote_code``;
- canonical source files remain under ``hf_export/``.

This utility never overwrites a backup.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

HF_EXPORT = Path(__file__).resolve().parent
ROOT = HF_EXPORT.parent
BACKUP = HF_EXPORT / "legacy_root_remote_code"

CANONICAL = {
    "configuration_xattn_clip.py": HF_EXPORT / "configuration_xattn_clip.py",
    "modeling_xattn_clip.py": HF_EXPORT / "modeling_xattn_clip.py",
    "configuration_rn_clip.py": HF_EXPORT / "configuration_rn_clip.py",
    "modeling_rn_clip.py": HF_EXPORT / "modeling_rn_clip.py",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def backup_path(name: str) -> Path:
    candidate = BACKUP / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 1
    while True:
        candidate = BACKUP / f"{stem}.root_backup_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    actions = []
    for name, canonical in CANONICAL.items():
        root_copy = ROOT / name
        if not root_copy.exists():
            continue
        if not canonical.is_file():
            raise FileNotFoundError(
                f"Refusing to move {root_copy}: canonical destination is missing: {canonical}"
            )
        if digest(root_copy) == digest(canonical):
            actions.append(("remove duplicate", root_copy, None))
        else:
            actions.append(("archive differing legacy copy", root_copy, backup_path(name)))

    if not actions:
        print("[remote-code-layout] root already clean")
        return 0

    for action, source, destination in actions:
        if destination is None:
            print(f"[remote-code-layout] {action}: {source}")
        else:
            print(f"[remote-code-layout] {action}: {source} -> {destination}")

    if not args.apply:
        print("[remote-code-layout] dry run; pass --apply to perform these moves/removals")
        return 0

    BACKUP.mkdir(parents=True, exist_ok=True)
    for action, source, destination in actions:
        if destination is None:
            source.unlink()
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
    print("[remote-code-layout] applied; repository root remote-code clutter removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
