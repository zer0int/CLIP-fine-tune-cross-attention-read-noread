from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from hf_export.rn_adapter import apply_read_null_token, remove_read_null_token  # noqa: E402


class IdentityLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return hidden_states


class FakeVision(nn.Module):
    def __init__(self, image_size: int, patch_size: int = 14):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "image_size": image_size,
                "patch_size": patch_size,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
            },
        )()
        self.embeddings = nn.Module()
        self.embeddings.class_embedding = nn.Parameter(torch.zeros(1024))
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([IdentityLayer() for _ in range(24)])


class Wrapper(nn.Module):
    def __init__(self, image_size: int, patch_size: int = 14):
        super().__init__()
        self.vision_model = FakeVision(image_size, patch_size)


class RNAdapterTests(unittest.TestCase):
    def test_224_token_applies_to_336_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "read_null_token.safetensors"
            save_file(
                {"read_null_token": torch.ones(1024)},
                path,
                metadata={
                    "read_null_insert_block": "13",
                    "image_size": "224",
                    "vision_width": "1024",
                },
            )
            model = Wrapper(336)
            apply_read_null_token(model, path)
            hidden = torch.zeros(2, 577, 1024)
            for layer in model.vision_model.encoder.layers:
                hidden = layer(hidden, None)
            self.assertEqual(tuple(hidden.shape), (2, 578, 1024))
            self.assertEqual(float(hidden[:, -1].detach().sum()), 2048.0)
            remove_read_null_token(model)
            self.assertFalse(hasattr(model.vision_model, "read_null_token"))

    def test_non_l14_throws(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "read_null_token.safetensors"
            save_file({"read_null_token": torch.ones(1024)}, path)
            with self.assertRaisesRegex(ValueError, "requires ViT-L/14"):
                apply_read_null_token(Wrapper(224, patch_size=16), path)

    def test_remove_strips_parameter_without_hook(self):
        model = Wrapper(224)
        model.vision_model.register_parameter(
            "read_null_token", nn.Parameter(torch.ones(1024))
        )

        remove_read_null_token(model)

        self.assertFalse(hasattr(model.vision_model, "read_null_token"))
        self.assertIsNone(model.vision_model._rn_adapter_handle)


if __name__ == "__main__":
    unittest.main()
