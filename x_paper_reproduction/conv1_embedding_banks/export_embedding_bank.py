#!/usr/bin/env python3
"""Export portable final-image-embedding banks for Conv1 reproduction.

Two source modes are supported:

* ``gpic``: consume the processed GPIC manifest used by the paper tooling. The
  output manifest retains sparse TAR lookup fields so public GPIC images can be
  fetched on demand later without downloading whole shards.
* ``local``: consume a CSV/Parquet manifest of local images, or recursively scan
  a local image directory. This is intended for private/custom augmentation
  banks (for example a local LAION sample).

Banks are stored under ``<output_root>/<model_repo_id with '/' -> '__'>`` by
default. A vanilla checkpoint is instantiated through ``attnclip_mechinterp_sae``;
RN/correction/x-attention checkpoints are instantiated through
``attnclip_mechinterp_xattn``. Detection is based on checkpoint parameters, not
on the model repo name.

Only final normalized image embeddings are exported:

* ``backbone`` for every supported CLIP model.
* ``content`` additionally when the loaded model exposes the trained CONTENT
  correction used by the x-attention model.

No patch tokens, residual streams, attention maps, or intermediate activations
are written.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (Path.cwd(), here.parents[2], here.parents[3]):
        if (candidate / "utils_clip_loader").is_dir() and (
            candidate / "attnclip_mechinterp_sae"
        ).is_dir():
            return candidate.resolve()
    return Path.cwd().resolve()


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_clip_loader.mechinterp_auto import load_mechinterp_clip_anything  # noqa: E402


FORMAT_VERSION = 2
EXPORTER_PATCH = "vanilla-sae-consistent-fp16-v3"
EMBEDDING_KEY = "embeddings"
DEFAULT_GPIC_MODEL = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def looks_like_hf_repo_id(value: str) -> bool:
    if not isinstance(value, str) or value.count("/") != 1:
        return False
    if "\\" in value or value.startswith(("./", "../", "~/")):
        return False
    if len(value) >= 2 and value[1] == ":":
        return False
    return not Path(value).exists()


def model_repo_to_subdir(model_source: str) -> str:
    """Portable bank subdirectory name.

    HF model ids use the exact convention requested for the release:
    ``namespace/repo`` -> ``namespace__repo``.
    """
    if looks_like_hf_repo_id(model_source):
        return model_source.replace("/", "__")
    stem = Path(model_source).name or "local_model"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
    return "local__" + safe


def schema_metadata(path: Path) -> dict[str, str]:
    raw = pq.read_schema(path).metadata or {}
    return {
        k.decode("utf-8", errors="replace"): v.decode("utf-8", errors="replace")
        for k, v in raw.items()
    }


def sha256_int64(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def amp_context(device: str, enabled: bool):
    if enabled and str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _architecture_summary(model: torch.nn.Module) -> dict[str, Any]:
    visual = getattr(model, "visual", None)
    return {
        "implant_kind": str(getattr(model, "implant_kind", "vanilla")),
        "model_dtype": str(getattr(model, "dtype", "unknown")),
        "input_resolution": int(getattr(visual, "input_resolution", -1)),
        "output_dim": int(getattr(visual, "output_dim", -1)),
        "read_attention_architecture": str(
            getattr(model, "read_attention_architecture", "none")
        ),
        "read_null_enabled": bool(getattr(model, "read_null_enabled", False)),
        "read_null_insert_block": int(getattr(model, "read_null_insert_block", -1)),
        "content_correction_available": bool(
            callable(getattr(model, "_content_image_from_info", None))
            and callable(getattr(model, "encode_image_states", None))
        ),
    }


def _normalize_vanilla_sae_precision(model: torch.nn.Module, device: str) -> str:
    """Repair the mixed-dtype state produced by the bundled SAE build_model.

    ``attnclip_mechinterp_sae.model.convert_weights`` converts its custom Q/K/V
    projections to fp16 but (unlike OpenAI CLIP's converter) does not convert
    Conv/Linear modules such as ``visual.conv1`` and the MLPs.  A vanilla HF
    checkpoint can therefore load as conv1=float32 while q_proj=float16, which
    fails at the first attention block regardless of the input image dtype.

    On CUDA we restore OpenAI-CLIP-style mixed precision: Conv/Linear weights
    are fp16 while LayerNorm parameters remain fp32.  On CPU we use fp32
    throughout, matching the usual OpenAI CLIP CPU path.
    """
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        model.float()
        return "fp32_cpu"

    def _convert(module: torch.nn.Module) -> None:
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            if getattr(module, "weight", None) is not None:
                module.weight.data = module.weight.data.half()
            if getattr(module, "bias", None) is not None:
                module.bias.data = module.bias.data.half()

    model.apply(_convert)

    # OpenAI CLIP also keeps these projection parameters in the compute dtype.
    for module in model.modules():
        for name in ("text_projection", "proj"):
            value = getattr(module, name, None)
            if isinstance(value, torch.nn.Parameter):
                value.data = value.data.half()

    return "openai_fp16_cuda"


def _vanilla_dtype_audit(model: torch.nn.Module) -> dict[str, str]:
    """Small audit for the exact dtype mismatch that motivated patch v3."""
    out: dict[str, str] = {}
    visual = getattr(model, "visual", None)
    conv1 = getattr(visual, "conv1", None)
    if getattr(conv1, "weight", None) is not None:
        out["conv1"] = str(conv1.weight.dtype)
    try:
        block0 = visual.transformer.resblocks[0]
        out["q_proj"] = str(block0.attn.q_proj.weight.dtype)
        out["mlp_c_fc"] = str(block0.mlp.c_fc.weight.dtype)
        out["ln_1"] = str(block0.ln_1.weight.dtype)
    except Exception:
        pass
    return out


def _visual_input_dtype(model: torch.nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
    """Return the dtype expected at the visual encoder input.

    OpenAI CLIP normally casts images inside ``encode_image``. Some bundled
    mechinterp variants expose lower-level visual forwards and do not perform
    that cast consistently, so the exporter owns this boundary explicitly.
    ``visual.conv1.weight`` is the authoritative dtype for ViT image input.
    """
    visual = getattr(model, "visual", None)
    conv1 = getattr(visual, "conv1", None)
    weight = getattr(conv1, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.is_floating_point():
        return weight.dtype

    dtype = getattr(model, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    return fallback


def _prepare_images_for_model(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    dtype = _visual_input_dtype(model, images.dtype)
    if images.dtype != dtype:
        images = images.to(dtype=dtype)
    return images


def _extract_final_spaces(model: torch.nn.Module, images: torch.Tensor) -> dict[str, torch.Tensor]:
    # Do not rely on each mechinterp wrapper to emulate OpenAI CLIP's
    # ``image.type(self.dtype)`` behavior.  In particular, split-QKV SAE
    # attention will otherwise receive float32 activations against fp16
    # projection weights when the checkpoint itself is fp16.
    images = _prepare_images_for_model(model, images)

    encode_states = getattr(model, "encode_image_states", None)
    content_fn = getattr(model, "_content_image_from_info", None)
    if callable(encode_states):
        info = encode_states(images, return_final_tokens=False)
        if "image_embedding" not in info:
            raise KeyError("encode_image_states() did not return image_embedding")
        backbone_raw = info["image_embedding"]
        result = {"backbone": F.normalize(backbone_raw.float(), dim=-1)}
        if callable(content_fn):
            content_raw = content_fn(info, apply_content_correction=True)
            result["content"] = F.normalize(content_raw.float(), dim=-1)
        return result

    encode_image = getattr(model, "encode_image", None)
    if not callable(encode_image):
        raise RuntimeError("Loaded CLIP model has neither encode_image_states nor encode_image")
    return {"backbone": F.normalize(encode_image(images).float(), dim=-1)}


def _available_spaces(model: torch.nn.Module, preprocess, device: str) -> tuple[list[str], int]:
    resolution = int(getattr(getattr(model, "visual", None), "input_resolution", 224))
    dummy = Image.new("RGB", (max(32, resolution), max(32, resolution)), (127, 127, 127))
    tensor = preprocess(dummy).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = _extract_final_spaces(model, tensor)
    dim = int(next(iter(outputs.values())).shape[-1])
    return list(outputs.keys()), dim


def _choose(frame: pd.DataFrame, key_col: str, max_images: int, selection: str, seed: int) -> pd.DataFrame:
    if key_col not in frame.columns:
        raise KeyError(f"source manifest missing selection key column {key_col!r}")
    frame = frame.sort_values(key_col).reset_index(drop=True)
    n = len(frame)
    if n == 0:
        raise RuntimeError("No usable source images")
    limit = int(max_images)
    if limit <= 0 or limit >= n:
        chosen = frame
    elif selection == "head":
        chosen = frame.iloc[:limit].copy()
    elif selection == "stride":
        idx = np.linspace(0, n - 1, num=limit, dtype=np.int64)
        chosen = frame.iloc[idx].copy()
    elif selection == "random":
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(n, size=limit, replace=False))
        chosen = frame.iloc[idx].copy()
    else:
        raise ValueError(selection)
    chosen = chosen.reset_index(drop=True)
    if "bank_row" in chosen.columns:
        chosen = chosen.drop(columns=["bank_row"])
    chosen.insert(0, "bank_row", np.arange(len(chosen), dtype=np.int64))
    return chosen


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported manifest format: {path}")


def prepare_gpic(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], Path, str]:
    root = args.gpic_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else root / "gpic_test_embedding_manifest.parquet"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    frame = pd.read_parquet(manifest_path)
    if "processing_ok" in frame.columns:
        frame = frame[frame["processing_ok"].astype(bool)].copy()
    if "embedding_row" not in frame.columns:
        raise KeyError("GPIC source manifest missing embedding_row")
    if "local_relpath" not in frame.columns:
        raise KeyError("GPIC source manifest missing local_relpath")
    selected = _choose(frame, "embedding_row", args.max_images, args.selection, args.seed)
    source_meta = schema_metadata(manifest_path)
    source_info = {
        "kind": "gpic",
        "source_manifest_name": manifest_path.name,
        "gpic_source_metadata": source_meta,
        "path_column": "local_relpath",
        "path_mode": "relative_to_runtime_gpic_root",
    }
    return selected, source_info, root, "embedding_row"


def _scan_images(root: Path, recursive: bool) -> pd.DataFrame:
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(
        (p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    return pd.DataFrame({"source_row": np.arange(len(paths), dtype=np.int64), "local_path": [str(p) for p in paths]})


def prepare_local(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], Path | None, str]:
    if args.input_manifest is None and args.image_root is None:
        raise ValueError("local mode requires --input-manifest or --image-root")
    if args.input_manifest is not None and args.image_root is not None and args.path_base is None:
        # image_root is allowed as an explicit path base only via --path-base; keeping
        # these concepts separate avoids surprising path rewrites.
        raise ValueError("Use either --input-manifest or --image-root; use --path-base for manifest-relative images")

    if args.image_root is not None:
        root = args.image_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        frame = _scan_images(root, args.recursive)
        source_name = root.name
        path_base = None
    else:
        manifest_path = args.input_manifest.expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        frame = _read_table(manifest_path)
        if args.path_column not in frame.columns:
            raise KeyError(f"Input manifest missing --path-column {args.path_column!r}")
        if "source_row" not in frame.columns:
            frame.insert(0, "source_row", np.arange(len(frame), dtype=np.int64))
        base = (
            args.path_base.expanduser().resolve()
            if args.path_base is not None
            else manifest_path.parent
        )
        resolved: list[str] = []
        for raw in frame[args.path_column].astype(str):
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = base / p
            resolved.append(str(p.resolve()))
        frame = frame.copy()
        frame["local_path"] = resolved
        source_name = manifest_path.name
        path_base = base

    exists = frame["local_path"].map(lambda value: Path(str(value)).is_file())
    missing = int((~exists).sum())
    if missing:
        sample = frame.loc[~exists, "local_path"].head(3).tolist()
        raise FileNotFoundError(
            f"{missing:,} local source images are missing; examples={sample}. "
            "Custom-bank export is strict so row identities cannot silently shift."
        )
    selected = _choose(frame, "source_row", args.max_images, args.selection, args.seed)
    source_info = {
        "kind": "local_manifest",
        "source_manifest_name": source_name,
        "path_column": "local_path",
        "path_mode": "local_only",
    }
    if path_base is not None:
        source_info["path_base_not_exported"] = True
    return selected, source_info, None, "source_row"


class BankImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, source_kind: str, source_root: Path | None, preprocess):
        self.entries: list[tuple[int, str]] = []
        self.preprocess = preprocess
        if source_kind == "gpic":
            assert source_root is not None
            for row in rows.itertuples(index=False):
                path = source_root / str(getattr(row, "local_relpath"))
                self.entries.append((int(getattr(row, "bank_row")), str(path.resolve())))
        else:
            for row in rows.itertuples(index=False):
                self.entries.append((int(getattr(row, "bank_row")), str(getattr(row, "local_path"))))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        bank_row, path_string = self.entries[index]
        path = Path(path_string)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.preprocess(image)
        return tensor, bank_row


def collate(batch):
    images, rows = zip(*batch)
    return torch.stack(images), torch.tensor(rows, dtype=torch.long)


def make_loader(rows, source_kind, source_root, preprocess, args):
    return DataLoader(
        BankImageDataset(rows, source_kind, source_root, preprocess),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=int(args.workers) > 0,
        collate_fn=collate,
    )


def write_manifest(frame: pd.DataFrame, path: Path, metadata: dict[str, Any]) -> None:
    encoded = {str(k).encode(): str(v).encode() for k, v in metadata.items()}
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(encoded)
    tmp = path.with_name(path.name + ".part")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def export_safetensors(npy_path: Path, out_path: Path, metadata: dict[str, str]) -> None:
    arr = np.load(npy_path, mmap_mode="c")  # writable copy-on-write mmap: zero-copy for export, no read-only Torch warning
    tensor = torch.from_numpy(arr)
    if not tensor.is_contiguous():
        raise RuntimeError(f"Non-contiguous embedding memmap: {npy_path}")
    tmp = out_path.with_name(out_path.name + ".part")
    tmp.unlink(missing_ok=True)
    save_safetensors({EMBEDDING_KEY: tensor}, str(tmp), metadata=metadata)
    os.replace(tmp, out_path)


def _state_signature(model_source, requested_revision, resolved_revision, model_hash, selection_hash, rows, dim, spaces):
    return {
        "format_version": FORMAT_VERSION,
        "model_source": model_source,
        "model_requested_revision": requested_revision,
        "model_resolved_revision": resolved_revision,
        "canonical_state_sha256": model_hash,
        "selection_hash": selection_hash,
        "rows": int(rows),
        "dim": int(dim),
        "spaces": list(spaces),
    }


def _validate_resume(state: dict[str, Any], signature: dict[str, Any]) -> None:
    for key, value in signature.items():
        if state.get(key) != value:
            raise RuntimeError(
                f"Existing generation state is incompatible at {key}: "
                f"{state.get(key)!r} != {value!r}. Use --overwrite-bank."
            )


def _prepare_bank_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for p in path.iterdir():
            if p.is_file() and (
                p.name in {"bank_config.json", "manifest.parquet", "generation_state.json", "self_test.csv"}
                or p.name.endswith(".safetensors")
                or p.name.endswith("_f16.npy.part")
            ):
                p.unlink()


def _image_path_for_row(row: pd.Series, source_kind: str, source_root: Path | None) -> Path:
    if source_kind == "gpic":
        assert source_root is not None
        return (source_root / str(row["local_relpath"])).resolve()
    return Path(str(row["local_path"])).expanduser().resolve()


def exact_top1(query: torch.Tensor, bank: torch.Tensor, device: str, chunk_size: int) -> tuple[float, int]:
    query = F.normalize(query.float().reshape(1, -1).to(device), dim=-1)
    best_value = -float("inf")
    best_index = -1
    for start in range(0, int(bank.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(bank.shape[0]))
        chunk = F.normalize(bank[start:end].to(device=device, dtype=torch.float32), dim=-1)
        sims = query @ chunk.t()
        value, idx = sims.max(dim=1)
        score = float(value.item())
        if score > best_value:
            best_value = score
            best_index = start + int(idx.item())
    return best_value, best_index


@torch.inference_mode()
def run_self_test(bank_dir: Path, manifest: pd.DataFrame, source_kind: str, source_root: Path | None, model, preprocess, spaces: Sequence[str], args) -> pd.DataFrame:
    count = min(int(args.self_test_count), len(manifest))
    if count <= 0:
        return pd.DataFrame()
    positions = np.linspace(0, len(manifest) - 1, num=count, dtype=np.int64)
    banks = {
        space: load_safetensors(str(bank_dir / f"{space}.safetensors"), device="cpu")[EMBEDDING_KEY]
        for space in spaces
    }
    rows_out: list[dict[str, Any]] = []
    for position in positions:
        row = manifest.iloc[int(position)]
        image_path = _image_path_for_row(row, source_kind, source_root)
        with Image.open(image_path) as image:
            tensor = preprocess(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(args.device)
        with amp_context(args.device, not args.no_amp):
            fresh = _extract_final_spaces(model, tensor)
        for space in spaces:
            score, found = exact_top1(fresh[space][0], banks[space], args.device, args.retrieval_chunk)
            rows_out.append({
                "space": space,
                "expected_bank_row": int(row["bank_row"]),
                "retrieved_bank_row": int(found),
                "cosine": float(score),
                "pass": bool(found == int(row["bank_row"])),
            })
    result = pd.DataFrame(rows_out)
    result.to_csv(bank_dir / "self_test.csv", index=False)
    if not bool(result["pass"].all()):
        raise RuntimeError("Reference-bank self-test failed; see self_test.csv")
    return result


@torch.inference_mode()
def export_bank(args: argparse.Namespace) -> Path:
    output_root = args.output_root.expanduser().resolve()
    subdir = args.bank_subdir or model_repo_to_subdir(args.model)
    bank_dir = output_root / subdir
    _prepare_bank_dir(bank_dir, args.overwrite_bank)

    if args.mode == "gpic":
        selected, source_info, source_root, source_key = prepare_gpic(args)
    else:
        selected, source_info, source_root, source_key = prepare_local(args)

    print(f"[model] resolving {args.model}")
    model, preprocess, load_info, auto_meta = load_mechinterp_clip_anything(
        args.model,
        device=args.device,
        cache_dir=args.hf_cache_dir,
        revision=args.model_revision,
        strict=True,
        allow_unsafe_hf_pickle=False,
    )
    model.eval()

    # The SAE mechinterp runtime historically had an incomplete convert_weights()
    # implementation: custom Q/K/V became fp16 while Conv/MLP Linear modules
    # remained fp32.  Casting only the input cannot fix that internal mismatch.
    # Normalize vanilla SAE precision before the first probe forward.
    precision_policy = "native"
    if auto_meta.clip_module == "attnclip_mechinterp_sae":
        precision_policy = _normalize_vanilla_sae_precision(model, args.device)
        print(
            f"[model] exporter_patch={EXPORTER_PATCH} precision_policy={precision_policy} "
            f"dtype_audit={_vanilla_dtype_audit(model)}"
        )

    spaces, dim = _available_spaces(model, preprocess, args.device)
    arch = _architecture_summary(model)
    print(
        f"[model] family={auto_meta.model_family} module={auto_meta.clip_module} "
        f"spaces={spaces} dim={dim} visual_input_dtype={_visual_input_dtype(model)}"
    )
    if auto_meta.resolved_revision:
        print(f"[model] resolved revision={auto_meta.resolved_revision}")
    print(f"[model] canonical state sha256={auto_meta.canonical_state_sha256}")

    selection_hash = sha256_int64(selected[source_key].to_numpy(np.int64))
    signature = _state_signature(
        args.model,
        args.model_revision,
        auto_meta.resolved_revision,
        auto_meta.canonical_state_sha256,
        selection_hash,
        len(selected),
        dim,
        spaces,
    )

    manifest_meta = {
        "embedding_bank_format_version": FORMAT_VERSION,
        "model_source": args.model,
        "model_requested_revision": args.model_revision or "",
        "model_resolved_revision": auto_meta.resolved_revision or "",
        "canonical_state_sha256": auto_meta.canonical_state_sha256,
        "selection": args.selection,
        "selection_hash": selection_hash,
        "rows": len(selected),
        "spaces": ",".join(spaces),
        "embeddings_normalized": "true",
        "source_kind": source_info["kind"],
    }
    if source_info["kind"] == "gpic":
        manifest_meta.update(source_info.get("gpic_source_metadata", {}))
    write_manifest(selected, bank_dir / "manifest.parquet", manifest_meta)

    state_path = bank_dir / "generation_state.json"
    temp_paths = {space: bank_dir / f"{space}_f16.npy.part" for space in spaces}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _validate_resume(state, signature)
        if state.get("complete") and all((bank_dir / f"{s}.safetensors").is_file() for s in spaces):
            print("[bank] complete bank already exists")
            return bank_dir
        for path in temp_paths.values():
            if not path.is_file():
                raise RuntimeError("Resume state exists but an embedding memmap is missing; use --overwrite-bank")
        start_row = int(state.get("next_bank_row", 0))
    else:
        for path in temp_paths.values():
            np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(selected), dim)).flush()
        state = {
            **signature,
            "created_at_utc": utc_now(),
            "next_bank_row": 0,
            "complete": False,
        }
        atomic_json(state_path, state)
        start_row = 0

    if start_row < len(selected):
        arrays = {space: np.load(path, mmap_mode="r+") for space, path in temp_paths.items()}
        remaining = selected.iloc[start_row:].copy()
        loader = make_loader(remaining, source_info["kind"], source_root, preprocess, args)
        bar = tqdm(total=len(selected), initial=start_row, desc=f"{subdir} bank", unit="img")
        for images, bank_rows in loader:
            images = images.to(args.device, non_blocking=str(args.device).startswith("cuda"))
            with amp_context(args.device, not args.no_amp):
                outputs = _extract_final_spaces(model, images)
            if set(outputs) != set(spaces):
                raise RuntimeError(f"Embedding spaces changed during generation: {list(outputs)} vs {spaces}")
            idx = bank_rows.numpy().astype(np.int64, copy=False)
            for space in spaces:
                arrays[space][idx] = outputs[space].cpu().to(torch.float16).numpy()
                arrays[space].flush()
            state["next_bank_row"] = int(idx[-1]) + 1
            state["updated_at_utc"] = utc_now()
            atomic_json(state_path, state)
            bar.update(len(idx))
        bar.close()
        del arrays

    common_meta = {
        "embedding_bank_format_version": str(FORMAT_VERSION),
        "model_source": args.model,
        "model_requested_revision": args.model_revision or "",
        "model_resolved_revision": auto_meta.resolved_revision or "",
        "canonical_state_sha256": auto_meta.canonical_state_sha256,
        "rows": str(len(selected)),
        "dim": str(dim),
        "dtype": "float16",
        "normalized": "l2",
        "selection_hash": selection_hash,
        "manifest": "manifest.parquet",
    }
    for space in spaces:
        print(f"[bank] exporting {space}.safetensors")
        export_safetensors(
            temp_paths[space],
            bank_dir / f"{space}.safetensors",
            {**common_meta, "space": space},
        )

    config = {
        **signature,
        "created_at_utc": state.get("created_at_utc", utc_now()),
        "completed_at_utc": utc_now(),
        "bank_subdir": subdir,
        "selection": args.selection,
        "seed": int(args.seed),
        "source": source_info,
        "embeddings": {
            "normalized": True,
            "stored_dtype": "float16",
            "tensor_key": EMBEDDING_KEY,
            "spaces": {space: f"{space}.safetensors" for space in spaces},
        },
        "model_identity": {
            "source": args.model,
            "requested_revision": args.model_revision,
            "resolved_revision": auto_meta.resolved_revision,
            "canonical_state_sha256": auto_meta.canonical_state_sha256,
            "model_family": auto_meta.model_family,
            "clip_module": auto_meta.clip_module,
            "hf_model_type": getattr(load_info, "hf_model_type", None),
            "inferred_openai_model": getattr(load_info, "inferred_openai_model", None),
        },
        "model_architecture": arch,
    }
    atomic_json(bank_dir / "bank_config.json", config)

    result = run_self_test(
        bank_dir,
        selected,
        source_info["kind"],
        source_root,
        model,
        preprocess,
        spaces,
        args,
    )
    if not result.empty:
        print(f"[self-test] {int(result['pass'].sum())}/{len(result)} passed")

    state["complete"] = True
    state["completed_at_utc"] = utc_now()
    atomic_json(state_path, state)
    if not args.keep_temp:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)

    print(f"[done] {bank_dir}")
    return bank_dir


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HF model repo id or supported local checkpoint")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--output-root", type=Path, required=True, help="Dataset-specific bank root; model subdir is created inside")
    parser.add_argument("--bank-subdir", default=None, help="Override auto subdir; default is HF repo id with '/' -> '__'")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument("--selection", choices=["head", "stride", "random"], default="stride")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite-bank", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--self-test-count", type=int, default=4)
    parser.add_argument("--retrieval-chunk", type=int, default=65536)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    gpic = sub.add_parser("gpic", help="Export from the processed GPIC test corpus")
    add_common(gpic)
    gpic.add_argument("--gpic-root", type=Path, required=True)
    gpic.add_argument("--manifest", type=Path, default=None, help="Defaults to <gpic-root>/gpic_test_embedding_manifest.parquet")

    local = sub.add_parser("local", help="Export a private/custom local-image bank")
    add_common(local)
    source = local.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-manifest", type=Path, default=None, help="CSV/Parquet with image paths")
    source.add_argument("--image-root", type=Path, default=None, help="Scan an image directory instead of a manifest")
    local.add_argument("--path-column", default="path", help="Path column in --input-manifest")
    local.add_argument("--path-base", type=Path, default=None, help="Base directory for relative manifest paths; defaults to manifest directory")
    local.add_argument("--recursive", action="store_true", help="Recursively scan --image-root")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_bank(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
