"""Configuration helpers for the paper reproduction front end."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG_PATH = Path("reproduction_config.json")
DEFAULT_CONFIG: dict[str, Any] = {
    "output_root": "out_paper_reproduction",
    "models": {
        # Final released x-attention model used by the HF-facing probes.
        "final_hf": "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX",
        # Historical OpenAI model-name spec retained for the older low-level
        # reproduction probes that still use the original OpenAI download table.
        "vanilla_spec": "ViT-L/14",
        # HF-safe vanilla source used by the Conv1/mechinterp cleanup branch.
        "vanilla_hf": "openai/clip-vit-large-patch14",
        # GmP exists only as an explicit comparison/training-derived condition.
        "gmp_hf": "zer0int/CLIP-GmP-ViT-L-14",
        # Low-level OpenAI-style probes still require the original local checkpoints.
        "xattn_checkpoint": None,
        "gmp_checkpoint": None,
    },
    "datasets": {
        "demoset_dir": "image_sets/demoset",
        "misc_image_dir": "image_sets/misc",
        "special_delivery_dir": "image_sets/special_delivery",
        "special_natural_dir": "image_sets/special_natural",
        "visualtextual_dir": "image_sets/visualtextual",
        # Shared natural-image population for sink / CLS-register workspace probes.
        # When unset, reproduce.py reuses the canonical benchmark MVT install when
        # possible and otherwise delegates preparation to the benchmark installer.
        # The bundled human_responses_dedup.csv is the one-row-per-image source of truth.
        "objectnet_mvt_root": None,
        "objectnet_mvt_auto_download": True,
        "objectnet_mvt_sample_size": 480,
        "objectnet_mvt_seed": 20260915,
    },
    "conv1": {
        # Public model-matched GPIC final-embedding bank.  The runtime model repo id
        # is mapped to a bank subfolder by replacing '/' with '__'.
        "gpic_embedding_repo": "zer0int/CLIP-GPIC-embeddings",
        "gpic_bank_revision": None,
        # Default public comparison set. Each model is written to its own
        # repo-id-derived output subfolder and matched to the same-named bank folder.
        "gpic_models": [
            "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX",
            "openai/clip-vit-large-patch14",
        ],
        # Backward-compatible single-model override; when set it replaces gpic_models.
        "gpic_model": None,
        "image_dir": "image_sets/retrieval",
        "experiment_config": "x_paper_reproduction/conv1_manifold_gpic/experiment_config.json",
        # Optional local/HF bank roots in the same exporter format. Missing local
        # paths are skipped cleanly; present-but-incompatible banks are rejected.
        "custom_embedding_banks": [],
    },
    "runtime": {
        "device": "cuda",
        "module_root": ".",
        # Managed derived model caches under <output_root>/_models.
        # keep = reuse across tasks/runs; task = delete after each successful task;
        # run = delete after the selected run completes successfully.
        "model_cache_policy": "keep",
    },
    "task_args": {},
}


def _merge(default: Any, value: Any) -> Any:
    if isinstance(default, dict):
        out = copy.deepcopy(default)
        if isinstance(value, Mapping):
            for k, v in value.items():
                out[k] = _merge(out[k], v) if k in out else copy.deepcopy(v)
        return out
    return copy.deepcopy(default if value is None else value)


def load_config(path: Path = DEFAULT_CONFIG_PATH, *, allow_missing: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if allow_missing:
            return copy.deepcopy(DEFAULT_CONFIG)
        raise FileNotFoundError(f"Reproduction config not found: {path}. Run `python reproduce.py setup` first.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return _merge(DEFAULT_CONFIG, raw)


def save_config(cfg: Mapping[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(cfg), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def get_key(cfg: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def set_key(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value
