from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from hf_export.configuration_rn_clip import RNCLIPConfig  # noqa: E402
from hf_export.hf_conversion import copy_correction  # noqa: E402
from hf_export.modeling_rn_clip import RNCLIPModel  # noqa: E402


def tiny_model() -> RNCLIPModel:
    config = RNCLIPConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "projection_dim": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 8,
            "pad_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 3,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "projection_dim": 16,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 16,
        },
        projection_dim=16,
        read_null_insert_block=1,
        read_tap_blocks=[1, 2],
        read_bridge_width=16,
        read_bridge_heads=4,
    )
    return RNCLIPModel(config).eval()


def correction_state(model: RNCLIPModel) -> dict[str, torch.Tensor]:
    state = {
        "visual.read_null_token": model.read_null_token.detach().clone(),
        "read_implant.content_tap_logits": model.content_tap_logits.detach().clone(),
    }
    for name, parameter in model.content_pool.named_parameters():
        value = parameter.detach().clone()
        if name.startswith(("k_proj.", "v_proj.", "out_proj.")):
            value = value.half()
        state[f"read_implant.content_pool.{name}"] = value
    return state


class CustomModelTests(unittest.TestCase):
    def test_rn_and_correction_toggle(self):
        model = tiny_model()
        pixels = torch.randn(2, 3, 32, 32)
        input_ids = torch.tensor([[2, 5, 3, 1], [2, 6, 3, 1]])
        default = model(input_ids=input_ids, pixel_values=pixels)
        disabled = model(input_ids=input_ids, pixel_values=pixels, correction=False)
        self.assertEqual(default.vision_model_output.last_hidden_state.shape[1], 6)
        self.assertEqual(tuple(default.logits_per_image.shape), (2, 2))
        self.assertEqual(tuple(disabled.logits_per_image.shape), (2, 2))
        self.assertTrue(model.config.correction)

    def test_conversion_upcasts_correction_weights_to_fp32(self):
        source = correction_state(tiny_model())
        model = tiny_model()
        copy_correction(model, source)
        self.assertEqual(model.content_pool.k_proj.weight.dtype, torch.float32)
        self.assertEqual(model.content_pool.v_proj.weight.dtype, torch.float32)
        self.assertEqual(model.content_pool.out_proj.weight.dtype, torch.float32)
        output = model.get_image_features(torch.randn(1, 3, 32, 32))
        self.assertTrue(torch.isfinite(output.pooler_output).all())

    def test_hf_bfloat16_load_keeps_correction_fp32(self):
        import tempfile

        model = tiny_model()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            reloaded = RNCLIPModel.from_pretrained(directory, dtype=torch.bfloat16).eval()
        self.assertEqual(
            reloaded.vision_model.encoder.layers[0].self_attn.q_proj.weight.dtype,
            torch.bfloat16,
        )
        self.assertEqual(reloaded.read_null_token.dtype, torch.float32)
        self.assertEqual(reloaded.content_pool.k_proj.weight.dtype, torch.float32)
        self.assertEqual(reloaded.content_tap_logits.dtype, torch.float32)



if __name__ == "__main__":
    unittest.main()
