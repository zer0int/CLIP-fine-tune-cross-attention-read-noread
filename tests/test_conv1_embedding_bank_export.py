from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "x_paper_reproduction" / "conv1_embedding_banks" / "export_embedding_bank.py"
spec = importlib.util.spec_from_file_location("conv1_embedding_export", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _preprocess(image):
    arr = np.asarray(image.resize((4, 4)).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class Vanilla(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = SimpleNamespace(input_resolution=4, output_dim=3)
        self.dtype = torch.float32

    def encode_image(self, images):
        return images.mean(dim=(2, 3))


class XAttn(Vanilla):
    implant_kind = "full"

    def encode_image_states(self, images, return_final_tokens=False):
        return {"image_embedding": images.mean(dim=(2, 3))}

    def _content_image_from_info(self, info, apply_content_correction=True):
        return info["image_embedding"] + torch.tensor([0.2, -0.1, 0.05], device=info["image_embedding"].device)


def _fake_loader(model, family, module):
    def fn(*args, **kwargs):
        info = SimpleNamespace(hf_model_type="clip", inferred_openai_model="ViT-L/14")
        meta = SimpleNamespace(
            model_family=family,
            clip_module=module,
            canonical_state_sha256="a" * 64,
            resolved_revision="b" * 40,
        )
        return model, _preprocess, info, meta
    return fn


def _args(tmp_path, model_source="openai/clip-vit-large-patch14"):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for i, rgb in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        p = image_dir / f"{i}.jpg"
        Image.new("RGB", (8, 8), rgb).save(p)
        rows.append({"index": i, "path": str(p)})
    csv = tmp_path / "paths.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    parser = mod.build_parser()
    return parser.parse_args([
        "local",
        "--model", model_source,
        "--output-root", str(tmp_path / "banks"),
        "--input-manifest", str(csv),
        "--path-column", "path",
        "--device", "cpu",
        "--workers", "0",
        "--batch-size", "2",
        "--self-test-count", "1",
        "--retrieval-chunk", "2",
    ])


def test_model_repo_to_subdir():
    assert mod.model_repo_to_subdir("openai/clip-vit-large-patch14") == "openai__clip-vit-large-patch14"
    assert mod.model_repo_to_subdir("zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX") == "zer0int__CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"


def test_vanilla_exports_backbone_only(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_mechinterp_clip_anything", _fake_loader(Vanilla(), "vanilla", "attnclip_mechinterp_sae"))
    out = mod.export_bank(_args(tmp_path))
    assert (out / "backbone.safetensors").is_file()
    assert not (out / "content.safetensors").exists()
    cfg = __import__("json").loads((out / "bank_config.json").read_text())
    assert cfg["embeddings"]["spaces"] == {"backbone": "backbone.safetensors"}
    assert cfg["model_identity"]["clip_module"] == "attnclip_mechinterp_sae"


def test_xattn_exports_backbone_and_content(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "load_mechinterp_clip_anything", _fake_loader(XAttn(), "full_xattn", "attnclip_mechinterp_xattn"))
    out = mod.export_bank(_args(tmp_path, "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"))
    assert (out / "backbone.safetensors").is_file()
    assert (out / "content.safetensors").is_file()
    cfg = __import__("json").loads((out / "bank_config.json").read_text())
    assert set(cfg["embeddings"]["spaces"]) == {"backbone", "content"}
    assert cfg["model_identity"]["clip_module"] == "attnclip_mechinterp_xattn"
