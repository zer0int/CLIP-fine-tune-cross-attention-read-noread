from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path

from hf_export.checkpoint_spec import (
    extract_state_dict,
    infer_checkpoint_spec,
    load_trusted_checkpoint,
    read_training_json,
    warn_on_json_mismatches,
)
from hf_export.hf_conversion import (
    export_correction_model,
    export_full_xattn_model,
    export_stock_rn_model,
    materialize_gmp_weights,
    save_rn_token,
    save_vanilla_text_encoders,
    write_conversion_manifest,
)


ROOT = Path(__file__).resolve().parent
SUPPORT_ROOT = ROOT / "hf_export"
ASSETS = SUPPORT_ROOT / "assets" / "hf_clip_tokenizer"

_REQUIRED_SUPPORT_FILES = (
    "configuration_rn_clip.py",
    "modeling_rn_clip.py",
    "configuration_xattn_clip.py",
    "modeling_xattn_clip.py",
    "xattn_inference.py",
    "verify_xattn_conversion.py",
    "checkpoint_spec.py",
    "rn_adapter.py",
)
_REQUIRED_TOKENIZER_ASSETS = (
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _validate_support_tree() -> None:
    missing = [str(SUPPORT_ROOT / name) for name in _REQUIRED_SUPPORT_FILES if not (SUPPORT_ROOT / name).is_file()]
    missing += [str(ASSETS / name) for name in _REQUIRED_TOKENIZER_ASSETS if not (ASSETS / name).is_file()]
    if not (ROOT / "requirements.txt").is_file():
        missing.append(str(ROOT / "requirements.txt"))
    if missing:
        raise FileNotFoundError("HF export support tree is incomplete:\n  - " + "\n  - ".join(missing))


def _copy_files(destination: Path, names: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = ROOT / name if name == "requirements.txt" else SUPPORT_ROOT / name
        shutil.copy2(source, destination / name)


def _write_repo_readme(destination: Path, title: str, body: str) -> None:
    (destination / "README.md").write_text(
        f"# {title}\n\n{body.strip()}\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one x-attention CLIP pickle into complete HF exports"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--training-json",
        type=Path,
        default=None,
        help="optional saved training config; checkpoint tensors remain authoritative",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("trained_model_hf_export")
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="inspect checkpoint/config/support files and exit before materializing HF models",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="replace a non-empty output directory after preflight succeeds",
    )
    parser.add_argument(
        "--export-rn-token-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--export-rn-model-base", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--export-rn-model-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--export-full-xattn-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--export-vanilla-text-encoders",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = args.output_dir.expanduser().resolve()
    _validate_support_tree()

    enabled_exports = {
        "rn_token_only": bool(args.export_rn_token_only),
        "rn_model_base": bool(args.export_rn_model_base),
        "rn_model_correction": bool(args.export_rn_model_correction),
        "full_xattn_model": bool(args.export_full_xattn_model),
        "vanilla_text_encoders": bool(args.export_vanilla_text_encoders),
    }
    if not args.preflight_only and not any(enabled_exports.values()):
        raise ValueError("No export target is enabled")

    print(f"[load] trusted pickle: {args.checkpoint}")
    loaded = load_trusted_checkpoint(args.checkpoint)
    raw_state = extract_state_dict(loaded)
    state, gmp_report = materialize_gmp_weights(raw_state)
    spec = infer_checkpoint_spec(args.checkpoint, loaded, state)
    training_json = read_training_json(args.training_json)
    json_mismatches = warn_on_json_mismatches(spec, training_json)
    print(
        f"[checkpoint] ViT-L/14@{spec.image_size}; RN before B{spec.read_null_insert_block}; "
        f"late taps={list(spec.read_tap_blocks)}; attention={spec.read_attention_architecture}"
    )

    print(f"[checkpoint] SOURCE={list(spec.source_tap_blocks)} ORTHO={list(spec.ortho_tap_blocks)}")
    print(f"[checkpoint] vocab={spec.source_vocab_size} image={spec.image_size} patch={spec.patch_size}")
    print(f"[plan] output={output_root}")
    print("[plan] " + ", ".join(f"{name}={'ON' if enabled else 'off'}" for name, enabled in enabled_exports.items()))
    if args.preflight_only:
        print("[preflight] PASS: checkpoint architecture, optional training config, tokenizer assets, and export support files are compatible")
        return

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite_output:
            raise FileExistsError(
                f"HF export output is not empty: {output_root}\n"
                "Use a fresh --output-dir or pass --overwrite-output."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    # Re-write the manifest because an overwrite may have removed the preflight copy.
    write_conversion_manifest(output_root, spec, json_mismatches, gmp_report)

    if args.export_rn_token_only:
        destination = output_root / "rn_token_only"
        save_rn_token(destination, state, spec)
        _copy_files(
            destination,
            (
                "rn_adapter.py",
                "requirements.txt",
            ),
        )
        _write_repo_readme(
            destination,
            "Standalone RN token",
            """
Load an OpenAI-style ViT-L/14 model through `load_clip_with_rn`. Both 224 and
336 px checkpoints are supported; other architectures throw. The helper is
explicit and does not use `trust_remote_code`.

```python
from rn_adapter import load_clip_with_rn

model = load_clip_with_rn(
    "zer0int/CLIP-GmP-ViT-L-14", ".", debug=True
)
```

Use the repository benchmark scripts for evaluation.
""",
        )
        print(f"[saved] {destination}")

    if args.export_rn_model_base:
        destination = output_root / "rn_model_base"
        export_stock_rn_model(destination, state, spec, ASSETS)
        _copy_files(
            destination,
            (
                "rn_adapter.py",
                "requirements.txt",
            ),
        )
        _write_repo_readme(
            destination,
            "RN-only fine-tune",
            """
This is Model 4: the fine-tuned vanilla CLIP backbone plus RN, without content
correction. `model.safetensors` is a stock `CLIPModel`; the separate RN tensor
is applied by the explicit helper, so normal model loading has no extra key.

```python
from rn_adapter import load_clip_with_rn
model = load_clip_with_rn(".", ".", debug=True)
```

Use the repository benchmark scripts to compare RN-on and RN-removed inference.
""",
        )
        gc.collect()
        print(f"[saved] {destination}")

    if args.export_rn_model_correction:
        destination = output_root / "rn_model_correction"
        export_correction_model(destination, state, spec, ASSETS)
        _copy_files(
            destination,
            (
                "configuration_rn_clip.py",
                "modeling_rn_clip.py",
                "requirements.txt",
            ),
        )
        _write_repo_readme(
            destination,
            "RN fine-tune with content correction",
            """
Load with `trust_remote_code=True`. Correction defaults to `True`; passing
`correction=False` disables only the B20/B21 content correction and leaves RN active.
All exported floating tensors are stored in FP32. At load time the correction
module and RN parameter remain FP32 even when the CLIP backbone is requested as FP16/BF16.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained(".", trust_remote_code=True)
features = model.get_image_features(pixel_values=pixels, correction=True)
```

Use the repository benchmark scripts for correction-off/on comparisons.
""",
        )
        gc.collect()
        print(f"[saved] {destination}")

    if args.export_full_xattn_model:
        destination = output_root / "full_xattn_model"
        export_full_xattn_model(destination, state, spec, ASSETS)
        _copy_files(
            destination,
            (
                "configuration_xattn_clip.py",
                "modeling_xattn_clip.py",
                "xattn_inference.py",
                "verify_xattn_conversion.py",
                "checkpoint_spec.py",
                "requirements.txt",
            ),
        )
        _write_repo_readme(
            destination,
            "Full RN + PIECES x-attention CLIP",
            """
This is the full converted model: RN-modified ViT, late lexical read bridge,
B20/B21 content correction, early orthographic/source branches, trust router,
and internal `<text><null>` abstention candidate. Load it with custom code:

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(".", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
print(model.describe_modes())
```

Use `AutoProcessor`, not a direct `CLIPProcessor` load: Transformers otherwise
probes this custom config through its empty base config class and emits a bogus
`xattn_clip` → `` model-type warning.

`mode="any"` is the robust default. `mode="read"` appends one output column
whose label is `NO_TEXT_DETECTED`; use `xattn_inference.predict_candidates` to
map it safely. `text`, `notext`, `classic`, and mixed-control `none` modes are
also available. `classic` still includes the trained RN token, but skips all
PIECES branches and content correction.

All exported floating tensors are stored in FP32. When loading with an explicit
HF `dtype`, the stock CLIP backbone follows that dtype while RN + PIECES remain FP32.
The repository benchmarks may still use CUDA autocast for the backbone.

If a real-checkpoint parity question appears, run
`python verify_xattn_conversion.py --checkpoint PATH --model .`; the resulting
JSON localizes the first differing backbone or PIECES component.
""",
        )
        gc.collect()
        print(f"[saved] {destination}")

    if args.export_vanilla_text_encoders:
        destination = output_root / "xattn_text_encoder_vanilla"
        paths = save_vanilla_text_encoders(destination, state, spec, ASSETS)
        _write_repo_readme(
            destination,
            "Vanilla CLIP text encoder exports",
            f"""
The vocabulary is stripped to rows `0:49408`; EOT remains token 49407.

- `{paths[0].name}`: `CLIPTextModel`, named for text-to-image/genAI consumers.
- `{paths[1].name}`: `CLIPTextModelWithProjection`.

Each safetensors file has a same-stem `_config.json`. Instantiate the named
Transformers class with that config, then load the exact safetensors file.
""",
        )
        print(f"[saved] {destination}")

    print(f"[done] {output_root}")


if __name__ == "__main__":
    main()
