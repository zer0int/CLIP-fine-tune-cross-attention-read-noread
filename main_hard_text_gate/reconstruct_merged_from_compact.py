#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
from typing import Any, Mapping
import torch


def torch_load_trusted(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def migrate_implant_state(model: torch.nn.Module, state: Mapping[str, Any]):
    out = {str(k): v for k, v in state.items() if torch.is_tensor(v)}
    stale_prefixes = ('presence_pool.', 'glyph_head.', 'auto_router.')
    stale_exact = {'readability_log_weight','read_calibration_bias','presence_tap_logits','glyph_tap_logits'}
    for key in list(out):
        if key in stale_exact or any(key.startswith(p) for p in stale_prefixes):
            out.pop(key, None)
    defaults = model.read_implant.state_dict()
    for key, value in defaults.items():
        out.setdefault(key, value)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compact_checkpoint', type=Path, required=True)
    ap.add_argument('--output_path', type=Path, required=True)
    ap.add_argument('--clip_package_root', type=Path, required=True)
    ap.add_argument('--read_attention_architecture', default='softmax')
    ap.add_argument('--read_null_enabled', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--read_null_insert_block', type=int, default=13)
    args = ap.parse_args()

    compact = torch_load_trusted(args.compact_checkpoint)
    if not isinstance(compact, dict) or 'implant_state_dict' not in compact:
        raise RuntimeError(f'Not a compact AnyText checkpoint: {args.compact_checkpoint}')
    base = compact.get('base_model_path')
    if not base and isinstance(compact.get('args'), Mapping):
        base = compact['args'].get('model_path')
    if not base:
        raise RuntimeError('Compact checkpoint contains neither base_model_path nor args.model_path')
    base_model = str(base)
    looks_local = (
        bool(re.match(r"^[A-Za-z]:[\\/]", base_model))
        or base_model.startswith((".", "~", "/", "\\\\"))
        or os.path.splitext(base_model)[1].lower() in {".pt", ".pth", ".safetensors"}
    )
    if looks_local:
        base_model = str(Path(base_model).expanduser())
        if not Path(base_model).is_file() and not Path(base_model).is_dir():
            raise FileNotFoundError(
                f'Original backbone/base model recorded by compact checkpoint is missing: {base_model}'
            )

    root = args.clip_package_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import oaiclip as clip
    from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

    print(f'[reconstruct] base model: {base_model}')
    print(f'[reconstruct] compact PIECES: {args.compact_checkpoint}')
    model, _, _ = load_openai_clip_anything(
        clip, base_model, device='cpu',
        read_attention_architecture=args.read_attention_architecture,
        read_null_enabled=args.read_null_enabled,
        read_null_insert_block=args.read_null_insert_block,
        reuse_full_model_pickle=False,
    )
    model.float()
    implant = migrate_implant_state(model, compact['implant_state_dict'])
    missing, unexpected = model.read_implant.load_state_dict(implant, strict=False)
    if 'hard_text_embedding' in compact:
        model.set_hard_text_token_embedding(compact['hard_text_embedding'])
    if 'null_text_embedding' in compact:
        model.set_null_text_token_embedding(compact['null_text_embedding'])
    saved_rn = compact.get('read_null_token')
    if isinstance(saved_rn, torch.Tensor) and args.read_null_enabled:
        model.visual.read_null_token.data.copy_(saved_rn.to(dtype=model.visual.read_null_token.dtype))
    elif args.read_null_enabled:
        print('[reconstruct] compact checkpoint predates READ_NULL; keeping loader initialization for READ_NULL')

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': {k: v.detach().cpu() for k,v in model.state_dict().items()},
        'metadata': {
            'read_attention_architecture': str(model.read_attention_architecture),
            'read_null_enabled': bool(getattr(model,'read_null_enabled',False)),
            'read_null_insert_block': int(getattr(model,'read_null_insert_block',20)),
            'format': 'gmp_anytext_merged_v5_read_null',
            'reconstructed_from_compact': str(args.compact_checkpoint),
            'base_model_path': base_model,
        }
    }, args.output_path)
    print(f'[reconstruct] saved merged checkpoint -> {args.output_path}')
    print(f'[reconstruct] implant missing={len(missing)} unexpected={len(unexpected)}')

if __name__ == '__main__':
    main()
