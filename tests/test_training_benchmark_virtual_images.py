from __future__ import annotations

import io
import zipfile
from pathlib import Path

import torch
from PIL import Image

from training_support.validation.benchmark_anytext_validation import _load_tensor


def _dummy_transform(image: Image.Image, _mask, image_size: int, *, flip: bool):
    assert image.mode == "RGB"
    assert image_size == 224
    assert flip is False
    pixel = image.getpixel((0, 0))
    return torch.tensor(pixel, dtype=torch.float32), None


def test_load_tensor_opens_hf_zip_virtual_uri_without_pathlib_mangling(tmp_path: Path) -> None:
    payload = io.BytesIO()
    Image.new("RGB", (3, 3), (10, 20, 30)).save(payload, format="PNG")
    archive = tmp_path / "images.zip"
    member = "images/train/example.png"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, payload.getvalue())

    value = {"bytes": None, "path": f"zip://{member}::{archive}"}
    tensor = _load_tensor(value, 224, _dummy_transform)
    assert tensor.shape == (3,)
    assert torch.equal(tensor, torch.tensor([10.0, 20.0, 30.0]))


def test_load_tensor_still_opens_normal_local_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "ordinary.png"
    Image.new("RGB", (2, 2), (40, 50, 60)).save(image_path)
    tensor = _load_tensor(image_path, 224, _dummy_transform)
    assert torch.equal(tensor, torch.tensor([40.0, 50.0, 60.0]))
