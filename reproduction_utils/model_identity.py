"""Model identity fingerprints for safe reuse of reproduction output roots.

The reproduction workspace is allowed to contain partial/rerun outputs.  What it
must not contain is results from different model weights under the same logical
model slot.  This module fingerprints local checkpoints/directories and Hugging
Face repositories without loading the model into GPU memory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")

OPENAI_CLIP_ARTIFACT_SHA256 = {
    "RN50": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
    "RN101": "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599",
    "RN50x4": "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd",
    "RN50x16": "52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa",
    "RN50x64": "be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c",
    "ViT-B/32": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    "ViT-B/16": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
    "ViT-L/14": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd0bfd2b27167a4a06ec9aa",
    "ViT-L/14@336px": "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _aggregate(entries: Iterable[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for name, digest in sorted(entries):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _weight_files(root: Path) -> list[Path]:
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES
    ]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def fingerprint_local(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_file():
        digest = _sha256_file(path)
        return {
            "kind": "local_file",
            "source": str(path),
            "hash_kind": "sha256",
            "sha256": digest,
        }
    if not path.is_dir():
        raise FileNotFoundError(path)
    weights = _weight_files(path)
    if not weights:
        raise FileNotFoundError(f"No model weight files found under local model directory: {path}")
    entries = [(p.relative_to(path).as_posix(), _sha256_file(p)) for p in weights]
    return {
        "kind": "local_directory",
        "source": str(path),
        "hash_kind": "aggregate_weight_sha256",
        "sha256": _aggregate(entries),
        "weight_files": [{"path": name, "sha256": digest} for name, digest in entries],
    }


def _lfs_sha256(lfs: Any) -> str | None:
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        for key in ("sha256", "oid"):
            value = lfs.get(key)
            if value:
                value = str(value)
                return value.removeprefix("sha256:")
        return None
    for key in ("sha256", "oid"):
        value = getattr(lfs, key, None)
        if value:
            return str(value).removeprefix("sha256:")
    return None


def fingerprint_hf(repo_id: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("huggingface_hub is required to fingerprint remote HF models") from exc

    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as exc:
        raise RuntimeError(f"Could not resolve Hugging Face model identity for {repo_id!r}: {exc}") from exc
    entries: list[tuple[str, str]] = []
    for sibling in getattr(info, "siblings", ()) or ():
        name = str(getattr(sibling, "rfilename", ""))
        if not name.lower().endswith(WEIGHT_SUFFIXES):
            continue
        digest = _lfs_sha256(getattr(sibling, "lfs", None))
        if digest:
            entries.append((name, digest))

    commit = str(getattr(info, "sha", "") or "") or None
    if entries:
        return {
            "kind": "huggingface",
            "source": repo_id,
            "hash_kind": "aggregate_remote_weight_sha256",
            "sha256": _aggregate(entries),
            "revision": commit,
            "weight_files": [{"path": name, "sha256": digest} for name, digest in sorted(entries)],
        }
    if commit:
        # Fallback for hubs/backends that do not expose per-file LFS digests.
        digest = hashlib.sha256(f"hf-commit\0{repo_id}\0{commit}".encode("utf-8")).hexdigest()
        return {
            "kind": "huggingface",
            "source": repo_id,
            "hash_kind": "resolved_hf_commit",
            "sha256": digest,
            "revision": commit,
            "note": "Per-weight SHA-256 metadata was unavailable; identity is pinned to the resolved HF commit.",
        }
    raise RuntimeError(f"Could not resolve a stable fingerprint for Hugging Face model: {repo_id}")


def fingerprint_model_spec(spec: str | Path, *, project_root: Path) -> dict[str, Any]:
    raw = str(spec).strip()
    if not raw:
        raise ValueError("Empty model specification")

    if raw in OPENAI_CLIP_ARTIFACT_SHA256:
        return {
            "kind": "openai_clip",
            "source": raw,
            "hash_kind": "official_artifact_sha256",
            "sha256": OPENAI_CLIP_ARTIFACT_SHA256[raw],
        }

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return fingerprint_local(candidate)
    project_candidate = project_root / candidate
    if project_candidate.exists():
        return fingerprint_local(project_candidate)
    if candidate.suffix.lower() in WEIGHT_SUFFIXES:
        raise FileNotFoundError(project_candidate)
    return fingerprint_hf(raw)


def identity_manifest_path(output_root: Path) -> Path:
    return output_root / "_meta" / "model_identity.json"


def load_identity_manifest(output_root: Path) -> dict[str, Any] | None:
    path = identity_manifest_path(output_root)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid model identity manifest: {path}")
    return raw


def write_identity_manifest(output_root: Path, models: dict[str, dict[str, Any]]) -> Path:
    path = identity_manifest_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "models": models}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def guard_and_merge_identities(
    output_root: Path,
    current: dict[str, dict[str, Any]],
) -> tuple[Path, list[str]]:
    """Verify overlapping model slots and merge newly-used slots.

    Existing unrelated/partial files are intentionally allowed.  A hard failure is
    raised only when the same logical model slot has a different fingerprint.
    """
    previous = load_identity_manifest(output_root)
    recorded: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    if previous is not None:
        raw_models = previous.get("models", {})
        if isinstance(raw_models, dict):
            recorded = dict(raw_models)
    elif output_root.exists() and any(output_root.iterdir()):
        notes.append(
            "Existing output root had no model-identity manifest; accepting it as a legacy/partial workspace and stamping current model identities."
        )

    for slot, identity in current.items():
        old = recorded.get(slot)
        if old is not None and old.get("sha256") != identity.get("sha256"):
            raise RuntimeError(
                "MODEL IDENTITY MISMATCH\n"
                f"  output root: {output_root}\n"
                f"  model slot: {slot}\n"
                f"  recorded: {old.get('sha256')} ({old.get('source')})\n"
                f"  current:  {identity.get('sha256')} ({identity.get('source')})\n"
                "Use a different --output-root for genuinely different model weights."
            )
        recorded[slot] = identity

    path = write_identity_manifest(output_root, recorded)
    return path, notes
