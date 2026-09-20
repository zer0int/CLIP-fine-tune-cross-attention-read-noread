"""Portable, deterministic discovery of fonts used by text-rendering curricula."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_FONT_NAMES = (
    "arial.ttf",
    "arialbd.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "segoeui.ttf",
    "verdana.ttf",
    "tahoma.ttf",
    "times.ttf",
    "georgia.ttf",
    "consola.ttf",
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSerif.ttf",
    "LiberationSans-Regular.ttf",
    "FreeSans.ttf",
    "Arial.ttf",
    "Helvetica.ttc",
    "Times New Roman.ttf",
)

_FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    output: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        expanded = path.expanduser()
        key = os.path.normcase(str(expanded))
        if key not in seen:
            seen.add(key)
            output.append(expanded)
    return tuple(output)


def _system_font_roots() -> tuple[Path, ...]:
    """Return font roots for the current OS only, in stable priority order."""
    if os.name == "nt":
        windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        local_app_data = os.environ.get("LOCALAPPDATA")
        user_font_root = (
            Path(local_app_data) / "Microsoft" / "Windows" / "Fonts"
            if local_app_data
            else Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts"
        )
        return _unique_paths((windows_dir / "Fonts", user_font_root))

    if sys.platform == "darwin":
        return _unique_paths(
            (
                Path("/System/Library/Fonts"),
                Path("/Library/Fonts"),
                Path.home() / "Library" / "Fonts",
                Path("/Network/Library/Fonts"),
            )
        )

    # Linux and other Unix-like systems. Honor XDG data roots in addition to
    # the traditional per-user ~/.fonts location.
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    user_data_root = (
        Path(xdg_data_home)
        if xdg_data_home
        else Path.home() / ".local" / "share"
    )
    xdg_data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")
    shared_roots = [Path(value) / "fonts" for value in xdg_data_dirs.split(os.pathsep) if value]
    return _unique_paths(
        (
            user_data_root / "fonts",
            Path.home() / ".fonts",
            *shared_roots,
        )
    )


def _font_files_under(root: Path) -> list[Path]:
    """Recursively enumerate font files without failing on unreadable subtrees."""
    files: list[Path] = []
    try:
        walker = os.walk(root, topdown=True, onerror=lambda _exc: None)
        for directory, dirnames, filenames in walker:
            dirnames.sort(key=str.casefold)
            for filename in sorted(filenames, key=str.casefold):
                if Path(filename).suffix.casefold() in _FONT_SUFFIXES:
                    files.append(Path(directory) / filename)
    except OSError:
        return files
    return files


def _deduplicate_existing(paths: Iterable[Path]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for path in paths:
        expanded = path.expanduser()
        try:
            resolved = expanded.resolve()
        except OSError:
            resolved = expanded
        key = os.path.normcase(str(resolved))
        try:
            exists = resolved.is_file()
        except OSError:
            exists = False
        if exists and key not in seen:
            seen.add(key)
            output.append(str(resolved))
    return output


def find_font_files(
    explicit: Sequence[str] = (),
    *,
    preferred_names: Sequence[str] = DEFAULT_FONT_NAMES,
    fallback_limit: int = 32,
) -> list[str]:
    """Return explicit fonts first, then stable cross-platform system choices.

    Only roots belonging to the current OS are inspected. Preferred font names
    have global priority across those roots; root order breaks ties when the
    same font name is installed in more than one location. A bounded generic
    fallback is used only when no explicit or preferred font exists.
    """
    roots = tuple(root for root in _system_font_roots() if root.is_dir())
    explicit_paths = [Path(value) for value in explicit if str(value).strip()]

    generic_paths: list[Path] = []
    by_name: dict[str, list[Path]] = {}
    for root in roots:
        files = _font_files_under(root)
        generic_paths.extend(files)
        for path in files:
            by_name.setdefault(path.name.casefold(), []).append(path)

    preferred_paths: list[Path] = []
    for name in preferred_names:
        preferred_paths.extend(by_name.get(str(name).casefold(), ()))

    preferred = _deduplicate_existing((*explicit_paths, *preferred_paths))
    if preferred:
        return preferred
    return _deduplicate_existing(generic_paths)[: max(0, int(fallback_limit))]
