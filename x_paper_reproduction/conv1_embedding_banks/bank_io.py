from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

EMBEDDING_KEY = "embeddings"
DEFAULT_GPIC_REPO = "stanford-vision-lab/gpic"
DEFAULT_GPIC_BANK_REPO = "zer0int/CLIP-GPIC-embeddings"


class BankUnavailable(RuntimeError):
    """Optional embedding bank is not available for the selected model."""


@dataclass(frozen=True)
class EmbeddingBank:
    name: str
    root: Path
    config: dict[str, Any]
    manifest: pd.DataFrame
    spaces: dict[str, str]
    model_source: str
    source_kind: str
    origin: str
    is_primary: bool = False


@dataclass(frozen=True)
class BankDescriptor:
    """Tiny bank descriptor resolved from ``bank_config.json`` only.

    This is intentionally sufficient for GPIC gate probing before a model, the
    million-row manifest, or multi-GiB embedding tensors are downloaded.
    """

    name: str
    root: Path
    config: dict[str, Any]
    spaces: dict[str, str]
    model_source: str
    source_kind: str
    origin: str
    is_primary: bool = False


def looks_like_hf_repo_id(value: str) -> bool:
    if not isinstance(value, str) or value.count("/") != 1:
        return False
    if "\\" in value or value.startswith(("./", "../", "~/")):
        return False
    if len(value) >= 2 and value[1] == ":":
        return False
    return not Path(value).expanduser().exists()


def model_repo_to_subdir(model_source: str) -> str:
    if looks_like_hf_repo_id(model_source):
        return model_source.replace("/", "__")
    stem = Path(model_source).name or "local_model"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
    return "local__" + safe


def _schema_metadata(path: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    raw = pq.read_schema(path).metadata or {}
    return {
        k.decode("utf-8", errors="replace"): v.decode("utf-8", errors="replace")
        for k, v in raw.items()
    }


def _bank_model_source(config: Mapping[str, Any]) -> str:
    return str(
        config.get("model_source")
        or (config.get("model_identity") or {}).get("source")
        or (config.get("model_load_info") or {}).get("base_model_or_path")
        or ""
    )


def _bank_source_kind(config: Mapping[str, Any]) -> str:
    source = config.get("source")
    if isinstance(source, Mapping) and source.get("kind"):
        return str(source.get("kind"))
    if config.get("gpic_source_metadata"):
        return "gpic"
    return "unknown"


def _space_files(config: Mapping[str, Any]) -> dict[str, str]:
    emb = config.get("embeddings") or {}
    spaces = emb.get("spaces") if isinstance(emb, Mapping) else None
    if isinstance(spaces, Mapping) and spaces:
        return {str(k): str(v) for k, v in spaces.items()}
    out: dict[str, str] = {}
    if isinstance(emb, Mapping):
        for name in ("backbone", "content"):
            value = emb.get(name)
            if value:
                out[name] = str(value)
    return out


def _safetensor_shape(path: Path, tensor_key: str = EMBEDDING_KEY) -> tuple[int, ...]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if tensor_key not in keys:
            raise KeyError(f"{path.name} lacks tensor {tensor_key!r}")
        return tuple(int(x) for x in handle.get_slice(tensor_key).get_shape())


def validate_bank_dir(
    bank_dir: Path,
    *,
    expected_model_source: str | None = None,
    required_spaces: Sequence[str] | None = None,
    name: str | None = None,
    origin: str | None = None,
    is_primary: bool = False,
    require_embedding_files: bool = True,
) -> EmbeddingBank:
    bank_dir = bank_dir.expanduser().resolve()
    config_path = bank_dir / "bank_config.json"
    manifest_path = bank_dir / "manifest.parquet"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    spaces = _space_files(config)
    if not spaces:
        raise RuntimeError(f"No embedding spaces declared in {config_path}")
    if required_spaces is not None:
        missing = [s for s in required_spaces if s not in spaces]
        if missing:
            raise BankUnavailable(
                f"bank {bank_dir} does not provide required space(s) {missing}; available={sorted(spaces)}"
            )

    model_source = _bank_model_source(config)
    if expected_model_source and model_source and model_source != str(expected_model_source):
        raise RuntimeError(
            f"Embedding-bank model mismatch: bank={model_source!r} runtime={str(expected_model_source)!r}"
        )

    manifest = pd.read_parquet(manifest_path)
    if "bank_row" not in manifest.columns:
        raise KeyError(f"{manifest_path} missing bank_row")
    if len(manifest) == 0:
        raise RuntimeError(f"{manifest_path}: embedding bank manifest is empty")
    expected = np.arange(len(manifest), dtype=np.int64)
    got = manifest["bank_row"].to_numpy(np.int64)
    if not np.array_equal(got, expected):
        raise RuntimeError(f"{manifest_path}: bank_row is not contiguous 0..N-1")

    if require_embedding_files:
        dims: set[int] = set()
        spaces_to_check = (
            {space: spaces[space] for space in required_spaces}
            if required_spaces is not None
            else spaces
        )
        for space, filename in spaces_to_check.items():
            path = bank_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            shape = _safetensor_shape(path)
            if len(shape) != 2:
                raise RuntimeError(f"Bad {space} embedding shape {shape} in {path}")
            if shape[0] != len(manifest):
                raise RuntimeError(
                    f"{space} rows={shape[0]:,} but manifest rows={len(manifest):,} in {bank_dir}"
                )
            dims.add(int(shape[1]))
        if len(dims) != 1:
            raise RuntimeError(f"Embedding dimensions disagree across spaces in {bank_dir}: {sorted(dims)}")

    return EmbeddingBank(
        name=name or bank_dir.name,
        root=bank_dir,
        config=config,
        manifest=manifest,
        spaces=spaces,
        model_source=model_source,
        source_kind=_bank_source_kind(config),
        origin=origin or str(bank_dir),
        is_primary=bool(is_primary),
    )


def _validate_descriptor_config(
    config_path: Path,
    *,
    expected_model_source: str | None,
    required_spaces: Sequence[str] | None,
    name: str,
    origin: str,
    is_primary: bool,
) -> BankDescriptor:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spaces = _space_files(config)
    if not spaces:
        raise RuntimeError(f"No embedding spaces declared in {config_path}")
    if required_spaces is not None:
        missing = [space for space in required_spaces if space not in spaces]
        if missing:
            raise BankUnavailable(
                f"bank {config_path.parent} does not provide required space(s) {missing}; "
                f"available={sorted(spaces)}"
            )
    model_source = _bank_model_source(config)
    if expected_model_source and model_source and model_source != str(expected_model_source):
        raise RuntimeError(
            f"Embedding-bank model mismatch: bank={model_source!r} "
            f"runtime={str(expected_model_source)!r}"
        )
    return BankDescriptor(
        name=name,
        root=config_path.parent.resolve(),
        config=config,
        spaces=spaces,
        model_source=model_source,
        source_kind=_bank_source_kind(config),
        origin=origin,
        is_primary=bool(is_primary),
    )


def resolve_bank_descriptor(
    source: str | Path,
    *,
    model_source: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    required_spaces: Sequence[str] | None = None,
    optional: bool = False,
    is_primary: bool = False,
    name: str | None = None,
) -> BankDescriptor | None:
    """Resolve only ``bank_config.json`` for a model-matched bank.

    Remote sources download one tiny JSON file.  This is used to check the gated
    GPIC source before fetching the large manifest, embeddings, or runtime model.
    Missing *optional* local roots/model subdirectories are skipped; a present
    but malformed or incompatible bank is an error.
    """
    value = str(source)
    path = Path(value).expanduser()
    subdir = model_repo_to_subdir(model_source)

    if path.is_dir():
        bank_dir = path if (path / "bank_config.json").is_file() else path / subdir
        if not bank_dir.is_dir():
            if optional:
                print(f"[bank] skip optional source {value!r}: no {subdir}/ bank")
                return None
            raise BankUnavailable(f"No bank for model {model_source!r} under {path}")
        config_path = bank_dir / "bank_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        origin = str(path.resolve())
    elif looks_like_hf_repo_id(value):
        from huggingface_hub import hf_hub_download

        try:
            config_path = Path(
                hf_hub_download(
                    repo_id=value,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=cache_dir,
                    filename=f"{subdir}/bank_config.json",
                )
            )
        except Exception:
            # A configured optional HF source may simply not publish this model.
            # Only suppress an absent model subdirectory; malformed *present* banks
            # are handled below.  Avoid importing hub exception classes here so
            # older huggingface_hub versions remain usable.
            if optional:
                print(f"[bank] skip optional HF source {value!r}: no accessible {subdir}/bank_config.json")
                return None
            raise
        origin = f"hf://datasets/{value}"
    else:
        if optional:
            print(f"[bank] skip optional source {value!r}: path unavailable")
            return None
        raise FileNotFoundError(value)

    descriptor_name = name or ("gpic" if is_primary else config_path.parent.name)
    return _validate_descriptor_config(
        config_path,
        expected_model_source=model_source,
        required_spaces=required_spaces,
        name=descriptor_name,
        origin=origin,
        is_primary=is_primary,
    )


def _download_hf_bank(
    repo_id: str,
    *,
    model_source: str,
    revision: str | None,
    cache_dir: str | None,
    required_spaces: Sequence[str] | None,
    download_embeddings: bool = True,
) -> Path:
    from huggingface_hub import hf_hub_download

    subdir = model_repo_to_subdir(model_source)
    config_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
            filename=f"{subdir}/bank_config.json",
        )
    )
    bank_dir = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spaces = _space_files(config)
    wanted = list(required_spaces) if required_spaces is not None else list(spaces)
    for space in wanted:
        if space not in spaces:
            raise BankUnavailable(
                f"HF bank {repo_id}/{subdir} lacks {space!r}; available={sorted(spaces)}"
            )
    filenames = [f"{subdir}/manifest.parquet"]
    if download_embeddings:
        filenames.extend(f"{subdir}/{spaces[s]}" for s in wanted)
    for filename in filenames:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
            filename=filename,
        )
    return bank_dir.resolve()


def resolve_bank_source(
    source: str | Path,
    *,
    model_source: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    required_spaces: Sequence[str] | None = None,
    optional: bool = False,
    is_primary: bool = False,
    download_embeddings: bool = True,
    name: str | None = None,
) -> EmbeddingBank | None:
    value = str(source)
    path = Path(value).expanduser()
    subdir = model_repo_to_subdir(model_source)

    if path.is_dir():
        bank_dir = path if (path / "bank_config.json").is_file() else path / subdir
        if not bank_dir.is_dir():
            if optional:
                print(f"[bank] skip optional source {value!r}: no {subdir}/ bank")
                return None
            raise BankUnavailable(f"No bank for model {model_source!r} under {path}")
        origin = str(path.resolve())
    elif looks_like_hf_repo_id(value):
        bank_dir = _download_hf_bank(
            value,
            model_source=model_source,
            revision=revision,
            cache_dir=cache_dir,
            required_spaces=required_spaces,
            download_embeddings=download_embeddings,
        )
        origin = f"hf://datasets/{value}"
    else:
        if optional:
            print(f"[bank] skip optional source {value!r}: path unavailable")
            return None
        raise FileNotFoundError(value)

    return validate_bank_dir(
        bank_dir,
        expected_model_source=model_source,
        required_spaces=required_spaces,
        name=(name or ("gpic" if is_primary else bank_dir.parent.name + ":" + bank_dir.name)),
        origin=origin,
        is_primary=is_primary,
        require_embedding_files=download_embeddings,
    )

def resolve_bank_collection(
    primary_source: str | Path,
    *,
    model_source: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    required_spaces: Sequence[str] | None = None,
    custom_sources: Iterable[str | Path] = (),
) -> list[EmbeddingBank]:
    primary = resolve_bank_source(
        primary_source,
        model_source=model_source,
        revision=revision,
        cache_dir=cache_dir,
        required_spaces=required_spaces,
        optional=False,
        is_primary=True,
    )
    assert primary is not None
    banks = [primary]
    for source in custom_sources:
        if source is None or not str(source).strip():
            continue
        bank = resolve_bank_source(
            source,
            model_source=model_source,
            revision=None,
            cache_dir=cache_dir,
            required_spaces=required_spaces,
            optional=True,
            is_primary=False,
        )
        if bank is not None:
            banks.append(bank)
    return banks


def ensure_bank_embeddings(
    bank: EmbeddingBank,
    *,
    model_source: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    required_spaces: Sequence[str] | None = None,
) -> EmbeddingBank:
    """Materialize embedding tensors after a metadata-only access probe."""
    if bank.origin.startswith("hf://datasets/"):
        repo_id = bank.origin[len("hf://datasets/"):]
        _download_hf_bank(
            repo_id,
            model_source=model_source,
            revision=revision,
            cache_dir=cache_dir,
            required_spaces=required_spaces,
            download_embeddings=True,
        )
    return validate_bank_dir(
        bank.root,
        expected_model_source=model_source,
        required_spaces=required_spaces,
        name=bank.name,
        origin=bank.origin,
        is_primary=bank.is_primary,
        require_embedding_files=True,
    )

def load_space(bank: EmbeddingBank, space: str) -> torch.Tensor:
    if space not in bank.spaces:
        raise KeyError(f"Bank {bank.name!r} has no space {space!r}; available={sorted(bank.spaces)}")
    tensors = load_safetensors(str(bank.root / bank.spaces[space]), device="cpu")
    if EMBEDDING_KEY not in tensors:
        raise KeyError(f"{bank.spaces[space]} lacks tensor {EMBEDDING_KEY!r}")
    tensor = tensors[EMBEDDING_KEY]
    if tensor.ndim != 2:
        raise RuntimeError(f"Bad {bank.name}/{space} shape: {tuple(tensor.shape)}")
    return tensor


@torch.inference_mode()
def exact_cosine_topk(
    queries: torch.Tensor,
    bank: torch.Tensor,
    *,
    topk: int,
    chunk_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    q = F.normalize(queries.float().to(device), dim=-1)
    k = min(int(topk), int(bank.shape[0]))
    best_values = torch.full((len(q), k), -float("inf"), device=device)
    best_indices = torch.full((len(q), k), -1, dtype=torch.long, device=device)
    for start in range(0, int(bank.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(bank.shape[0]))
        chunk = F.normalize(bank[start:end].to(device=device, dtype=torch.float32), dim=-1)
        similarities = q @ chunk.t()
        kk = min(k, similarities.shape[1])
        values, indices = torch.topk(similarities, kk, dim=1)
        indices = indices + start
        candidate_values = torch.cat((best_values, values), dim=1)
        candidate_indices = torch.cat((best_indices, indices), dim=1)
        best_values, order = torch.topk(candidate_values, k, dim=1)
        best_indices = torch.gather(candidate_indices, 1, order)
    return best_values.cpu().numpy(), best_indices.cpu().numpy()


def search_collection(
    queries: torch.Tensor,
    banks: Sequence[EmbeddingBank],
    *,
    space: str,
    topk: int,
    chunk_size: int,
    device: str,
) -> list[list[dict[str, Any]]]:
    per_query: list[list[dict[str, Any]]] = [[] for _ in range(len(queries))]
    for bank in banks:
        if space not in bank.spaces:
            continue
        tensor = load_space(bank, space)
        if tensor.shape[1] != queries.shape[1]:
            raise RuntimeError(
                f"Dimension mismatch in {bank.name}/{space}: bank={tensor.shape[1]} query={queries.shape[1]}"
            )
        values, indices = exact_cosine_topk(
            queries, tensor, topk=topk, chunk_size=chunk_size, device=device
        )
        for qi in range(len(queries)):
            for local_rank in range(values.shape[1]):
                bank_row = int(indices[qi, local_rank])
                manifest_row = bank.manifest.iloc[bank_row]
                record: dict[str, Any] = {
                    "bank_name": bank.name,
                    "bank_origin": bank.origin,
                    "bank_row": bank_row,
                    "bank_local_rank": local_rank + 1,
                    "cosine": float(values[qi, local_rank]),
                    "_bank": bank,
                }
                for key, value in manifest_row.items():
                    if key not in record:
                        record[str(key)] = value
                per_query[qi].append(record)
        del tensor
    for qi in range(len(per_query)):
        per_query[qi].sort(key=lambda x: float(x["cosine"]), reverse=True)
        per_query[qi] = per_query[qi][: int(topk)]
        for rank, row in enumerate(per_query[qi], start=1):
            row["rank"] = rank
    return per_query


def _gpic_metadata_from_config(config: Mapping[str, Any]) -> dict[str, str]:
    meta: dict[str, str] = {}
    v1 = config.get("gpic_source_metadata") or {}
    if isinstance(v1, Mapping):
        for key, value in v1.items():
            meta[str(key)] = str(value)
    source = config.get("source") or {}
    if isinstance(source, Mapping):
        v2 = source.get("gpic_source_metadata") or {}
        if isinstance(v2, Mapping):
            for key, value in v2.items():
                meta.setdefault(str(key), str(value))
    return meta


def _gpic_metadata(bank: EmbeddingBank) -> dict[str, str]:
    # The Parquet schema is authoritative for row-layout metadata; the exported
    # JSON carries the same public GPIC provenance and remains a fallback for v1.
    meta = _schema_metadata(bank.root / "manifest.parquet")
    for key, value in _gpic_metadata_from_config(bank.config).items():
        meta.setdefault(key, value)
    return meta


def _ascii_box(lines: Sequence[str]) -> str:
    width = max(len(line) for line in lines)
    border = "+-" + "-" * width + "-+"
    body = ["| " + line.ljust(width) + " |" for line in lines]
    return "\n".join([border, *body, border])


def gpic_access_probe(bank: EmbeddingBank | BankDescriptor) -> tuple[bool, str]:
    """Check gated GPIC source access without downloading an image payload.

    For a :class:`BankDescriptor` this requires only ``bank_config.json``; the
    million-row manifest and embedding tensors are intentionally not needed yet.
    """
    if bank.source_kind != "gpic":
        return False, "primary bank is not marked as GPIC"
    try:
        from huggingface_hub import HfFileSystem, get_token
    except Exception as exc:
        return False, f"huggingface_hub unavailable: {exc}"
    token = get_token()
    if not token:
        return False, "no Hugging Face token found"
    meta = _gpic_metadata_from_config(bank.config)
    if isinstance(bank, EmbeddingBank):
        # Older bank configs may rely on metadata stored only in Parquet.
        for key, value in _gpic_metadata(bank).items():
            meta.setdefault(key, value)
    repo_id = meta.get("gpic_repo_id", DEFAULT_GPIC_REPO)
    revision = meta.get("gpic_revision")
    if not revision:
        return False, "bank has no pinned gpic_revision"
    template = meta.get("gpic_tar_path_template", "test/gpic_test_{shard:05d}.tar")
    try:
        remote_rel = str(template).format(shard=0)
    except Exception:
        remote_rel = "test/gpic_test_00000.tar"
    remote = f"hf://datasets/{repo_id}@{revision}/{remote_rel.lstrip('/')}"
    try:
        fs = HfFileSystem(token=token)
        fs.info(remote)
        return True, f"{repo_id}@{revision}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def print_gpic_access_warning(detail: str) -> None:
    print(
        _ascii_box(
            [
                "GPIC OPTIONAL SOURCE ACCESS IS UNAVAILABLE",
                "",
                "GPIC-dependent Conv1 reproduction will be SKIPPED (N/A).",
                "Accept the GPIC dataset terms in your browser:",
                "    https://huggingface.co/datasets/stanford-vision-lab/gpic",
                "Then authenticate this machine:",
                "    hf auth login",
                "and rerun the task.",
                "",
                f"Reason: {detail}",
            ]
        )
    )


def _to_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return image.convert("RGB")


def _gpic_display_transform(payload: bytes, bank: EmbeddingBank) -> Image.Image:
    meta = _gpic_metadata(bank)
    target_size = int(meta.get("target_size", 336) or 336)
    with Image.open(io.BytesIO(payload)) as opened:
        image = _to_rgb(opened)
        w, h = image.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize(
            (target_size, target_size),
            resample=Image.Resampling.BICUBIC,
            reducing_gap=2.0,
        )

        # GPIC bank generation stores a center-cropped/resized JPEG derivative.
        # Reproduce that final JPEG round-trip for remotely fetched source members
        # so displayed neighbors match the actual pixels that were embedded.
        quality = int(meta.get("jpeg_quality", 90) or 90)
        subsampling = int(meta.get("jpeg_subsampling", 2) or 2)
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=quality, subsampling=subsampling)
        encoded.seek(0)
        with Image.open(encoded) as roundtripped:
            roundtripped.load()
            return roundtripped.convert("RGB").copy()


def _fetch_gpic_range(row: Mapping[str, Any], bank: EmbeddingBank) -> Image.Image:
    from huggingface_hub import HfFileSystem, get_token

    token = get_token()
    if not token:
        raise PermissionError("No HF token available for gated GPIC retrieval")
    meta = _gpic_metadata(bank)
    repo_id = meta.get("gpic_repo_id", DEFAULT_GPIC_REPO)
    revision = meta.get("gpic_revision")
    if not revision:
        raise RuntimeError("Bank has no pinned gpic_revision")
    shard = int(row["shard"])
    remote = f"hf://datasets/{repo_id}@{revision}/test/gpic_test_{shard:05d}.tar"
    fs = HfFileSystem(token=token)
    with fs.open(remote, "rb", block_size=1 << 20) as handle:
        handle.seek(int(row["tar_offset"]))
        payload = handle.read(int(row["encoded_size"]))
    if len(payload) != int(row["encoded_size"]):
        raise IOError(
            f"Short GPIC range read: expected {int(row['encoded_size'])}, got {len(payload)}"
        )
    return _gpic_display_transform(payload, bank)


def neighbor_image(
    record: Mapping[str, Any],
    *,
    gpic_root: Path | None = None,
    allow_remote: bool = True,
) -> Image.Image:
    bank = record.get("_bank")
    if not isinstance(bank, EmbeddingBank):
        raise TypeError("neighbor record is missing its EmbeddingBank handle")

    if bank.source_kind == "gpic":
        rel = record.get("local_relpath")
        if gpic_root is not None and rel is not None and str(rel).strip():
            local = gpic_root / str(rel)
            if local.is_file():
                with Image.open(local) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image.load()
                    return image.copy()
        if allow_remote:
            return _fetch_gpic_range(record, bank)
        raise FileNotFoundError("GPIC neighbor unavailable locally and remote fallback disabled")

    local_path = record.get("local_path") or record.get("path")
    if local_path and Path(str(local_path)).expanduser().is_file():
        with Image.open(Path(str(local_path)).expanduser()) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.load()
            return image.copy()
    raise FileNotFoundError(f"Neighbor image unavailable for custom bank {bank.name!r}")


def public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Strip the private in-memory bank handle before writing tabular outputs."""
    return {str(k): v for k, v in record.items() if k != "_bank"}
