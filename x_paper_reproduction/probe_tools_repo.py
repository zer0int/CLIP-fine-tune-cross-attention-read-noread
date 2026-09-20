"""Import bootstrap for standalone paper-reproduction probes.

Filename execution puts ``x_paper_reproduction`` on ``sys.path`` rather than
the repository root.  The project also keeps the unmodified vanilla OpenAI CLIP
implementation (``oaicliporg``) under
``main_hard_text_gate/legacy_hard_text_pre``.  Expose both locations while
keeping the repository root first so root-level ``oaiclip`` continues to mean
the concat-attention implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path


PROBE_REPRODUCTION_SCHEMA = 30


def _assert_bundle_schema(root: Path) -> None:
    try:
        from reproduction_utils.version import REPRODUCTION_SCHEMA, REPRODUCTION_RELEASE
    except Exception as exc:
        raise RuntimeError(
            "This x_paper_reproduction probe cannot find the matching reproduction_utils/version.py. "
            "The repository appears to contain a mixed/partial reproduction bundle. Copy reproduce.py, "
            "reproduction_utils/, and x_paper_reproduction/ from the same release. "
            f"Original error: {exc!r}"
        ) from exc
    if int(REPRODUCTION_SCHEMA) != PROBE_REPRODUCTION_SCHEMA:
        raise RuntimeError(
            "Mixed paper-reproduction bundle detected: "
            f"probe schema={PROBE_REPRODUCTION_SCHEMA}, front-end schema={REPRODUCTION_SCHEMA} "
            f"({REPRODUCTION_RELEASE}). Copy reproduce.py, reproduction_utils/, and "
            "x_paper_reproduction/ from the same release before running experiments."
        )


def ensure_repo_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    legacy_parent = root / "main_hard_text_gate" / "legacy_hard_text_pre"
    legacy_text = str(legacy_parent)
    if legacy_parent.is_dir() and legacy_text not in sys.path:
        # Append, do not prepend: root/oaiclip must win over the historical
        # legacy_hard_text_pre/oaiclip sibling, while oaicliporg becomes importable.
        sys.path.append(legacy_text)
    _assert_bundle_schema(root)
    return root


REPO_ROOT = ensure_repo_root()
