from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from hf_export.hf_conversion import copy_clip_backbone  # noqa: E402


def source_state() -> dict[str, torch.Tensor]:
    dtype = torch.float16
    state = {
        "token_embedding.weight": torch.randn(49408, 32, dtype=dtype),
        "positional_embedding": torch.randn(8, 32, dtype=dtype),
        "ln_final.weight": torch.randn(32, dtype=dtype),
        "ln_final.bias": torch.randn(32, dtype=dtype),
        "text_projection": torch.randn(32, 16, dtype=dtype),
        "visual.conv1.weight": torch.randn(32, 3, 16, 16, dtype=dtype),
        "visual.class_embedding": torch.randn(32, dtype=dtype),
        "visual.positional_embedding": torch.randn(5, 32, dtype=dtype),
        "visual.ln_pre.weight": torch.randn(32, dtype=dtype),
        "visual.ln_pre.bias": torch.randn(32, dtype=dtype),
        "visual.ln_post.weight": torch.randn(32, dtype=dtype),
        "visual.ln_post.bias": torch.randn(32, dtype=dtype),
        "visual.proj": torch.randn(32, 16, dtype=dtype),
        "logit_scale": torch.tensor(2.5, dtype=dtype),
    }
    for prefix, count in (
        ("transformer.resblocks.", 2),
        ("visual.transformer.resblocks.", 3),
    ):
        for index in range(count):
            width = 32
            base = f"{prefix}{index}"
            for ln in ("ln_1", "ln_2"):
                state[f"{base}.{ln}.weight"] = torch.randn(width, dtype=dtype)
                state[f"{base}.{ln}.bias"] = torch.randn(width, dtype=dtype)
            state[f"{base}.attn.in_proj_weight"] = torch.randn(
                3 * width, width, dtype=dtype
            )
            state[f"{base}.attn.in_proj_bias"] = torch.randn(3 * width, dtype=dtype)
            state[f"{base}.attn.out_proj.weight"] = torch.randn(
                width, width, dtype=dtype
            )
            state[f"{base}.attn.out_proj.bias"] = torch.randn(width, dtype=dtype)
            state[f"{base}.mlp.c_fc.weight"] = torch.randn(64, width, dtype=dtype)
            state[f"{base}.mlp.c_fc.bias"] = torch.randn(64, dtype=dtype)
            state[f"{base}.mlp.c_proj.weight"] = torch.randn(width, 64, dtype=dtype)
            state[f"{base}.mlp.c_proj.bias"] = torch.randn(width, dtype=dtype)
    return state


class ConversionMappingTests(unittest.TestCase):
    def test_oai_to_hf_mapping_and_dtype(self):
        text = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=32,
            intermediate_size=64,
            projection_dim=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=8,
            pad_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
        )
        vision = CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            projection_dim=16,
            num_hidden_layers=3,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
        )
        model = CLIPModel(
            CLIPConfig(
                text_config=text.to_dict(),
                vision_config=vision.to_dict(),
                projection_dim=16,
            )
        )
        state = source_state()
        copy_clip_backbone(model, state)
        q, k, v = state["transformer.resblocks.0.attn.in_proj_weight"].chunk(3)
        layer = model.text_model.encoder.layers[0].self_attn
        self.assertTrue(torch.equal(layer.q_proj.weight, q))
        self.assertTrue(torch.equal(layer.k_proj.weight, k))
        self.assertTrue(torch.equal(layer.v_proj.weight, v))
        self.assertTrue(
            torch.equal(model.text_projection.weight, state["text_projection"].T)
        )
        self.assertTrue(
            torch.equal(model.visual_projection.weight, state["visual.proj"].T)
        )
        self.assertEqual(model.visual_projection.weight.dtype, torch.float32)
        self.assertEqual(layer.q_proj.weight.dtype, torch.float32)
        torch.testing.assert_close(layer.q_proj.weight, q.float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
