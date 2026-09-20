from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from hf_export.checkpoint_spec import infer_checkpoint_spec, warn_on_json_mismatches  # noqa: E402
from hf_export.hf_conversion import materialize_gmp_weights  # noqa: E402


def synthetic_state(image_size: int = 224) -> dict[str, torch.Tensor]:
    grid = image_size // 14

    def meta(*shape: int) -> torch.Tensor:
        return torch.empty(shape, device="meta")

    state = {
        "visual.conv1.weight": meta(1024, 3, 14, 14),
        "visual.class_embedding": meta(1024),
        "visual.positional_embedding": meta(grid * grid + 1, 1024),
        "visual.proj": meta(1024, 768),
        "visual.read_null_token": meta(1024),
        "visual.read_null_insert_block_config": torch.tensor(13),
        "token_embedding.weight": meta(49412, 768),
        "positional_embedding": meta(77, 768),
        "ln_final.weight": meta(768),
        "text_projection": meta(768, 768),
        "read_implant.tap_blocks": torch.tensor([20, 21]),
        "read_implant.ortho_tap_blocks": torch.tensor([8, 12, 13]),
        "read_implant.source_tap_blocks": torch.tensor([6, 7, 10]),
        "read_implant.bridge_heads_config": torch.tensor(4),
        "read_implant.early_expanded_width_config": torch.tensor(4096),
        "read_implant.read_bridge.q_proj.weight": meta(256, 768),
        "read_implant.read_bridge.k_proj.weight": meta(256, 1024),
        "read_implant.read_bridge.v_proj.weight": meta(256, 1024),
        "read_implant.read_bridge.sigmoid_patch_head_bias": meta(4),
        "read_implant.content_pool.out_proj.weight": meta(768, 256),
        "read_implant.content_tap_logits": meta(2),
        "read_implant.read_tap_logits": meta(2),
        "read_implant.ortho_tap_logits": meta(3),
        "read_implant.source_tap_logits": meta(3),
        "read_implant.read_probe": meta(768),
        "read_implant.orthographic_bridge.patch_expand.weight": meta(4096, 1024),
        "read_implant.orthographic_bridge.patch_contract.weight": meta(256, 4096),
        "read_implant.source_head.patch_expand.weight": meta(4096, 1024),
        "read_implant.source_head.patch_contract.weight": meta(256, 4096),
        "read_implant.source_head.stats_fc1.weight": meta(256, 8),
        "read_implant.source_head.stats_fc2.weight": meta(2, 256),
        "read_implant.trust_router.fc1.weight": meta(128, 16),
        "read_implant.trust_router.fc2.weight": meta(64, 128),
        "read_implant.trust_router.fc3.weight": meta(1, 64),
    }
    for index in range(24):
        state[f"visual.transformer.resblocks.{index}.ln_1.weight"] = meta(1024)
    for index in range(12):
        state[f"transformer.resblocks.{index}.ln_1.weight"] = meta(768)
    return state


class CheckpointTests(unittest.TestCase):
    def test_pickle_dimensions_win_over_json(self):
        spec = infer_checkpoint_spec(
            Path("future_checkpoint.pt"), {}, synthetic_state(image_size=336)
        )
        self.assertEqual(spec.image_size, 336)
        self.assertEqual(spec.read_null_insert_block, 13)
        self.assertEqual(spec.read_attention_architecture, "sigmoid_all")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = warn_on_json_mismatches(
                spec, {"image_size": 224, "read_null_insert_block": 20}
            )
        self.assertEqual(len(mismatches), 2)
        self.assertEqual(len(caught), 2)

    def test_internal_shape_conflict_throws(self):
        state = synthetic_state()
        state["visual.read_null_token"] = torch.empty(768, device="meta")
        with self.assertRaisesRegex(ValueError, "internally inconsistent"):
            infer_checkpoint_spec(Path("bad.pt"), {}, state)

    def test_gmp_materialization_preserves_source_dtype(self):
        theta = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float16)
        radius = torch.tensor([[10.0], [3.0]], dtype=torch.float16)
        output, report = materialize_gmp_weights(
            {"layer.theta": theta, "layer.r": radius}
        )
        self.assertEqual(output["layer.weight"].dtype, torch.float16)
        self.assertTrue(
            torch.allclose(
                output["layer.weight"].float(),
                torch.tensor([[6.0, 8.0], [0.0, 3.0]]),
                atol=1e-3,
            )
        )
        self.assertIn("layer.weight", report)


if __name__ == "__main__":
    unittest.main()
