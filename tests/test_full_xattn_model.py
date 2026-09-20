from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoProcessor


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HF_EXPORT_ROOT = PACKAGE_ROOT / "hf_export"
sys.path.insert(0, str(PACKAGE_ROOT))

from hf_export.configuration_xattn_clip import XAttnCLIPConfig  # noqa: E402
from hf_export.hf_conversion import copy_clip_backbone, copy_xattn, save_processor  # noqa: E402
from hf_export.modeling_xattn_clip import XAttnCLIPModel  # noqa: E402
from oaiclip.model import CLIP  # noqa: E402
from hf_export.xattn_inference import resolve_xattn_model_reference  # noqa: E402


def tiny_pair() -> tuple[CLIP, XAttnCLIPModel]:
    torch.manual_seed(5)
    source = CLIP(
        embed_dim=32,
        image_resolution=32,
        vision_layers=4,
        vision_width=64,
        vision_patch_size=16,
        context_length=8,
        vocab_size=49412,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=2,
        read_tap_blocks=(2, 3),
        ortho_tap_blocks=(0, 1),
        source_tap_blocks=(0, 1),
        early_expanded_width=128,
        read_bridge_width=32,
        read_bridge_heads=4,
        read_attention_architecture="sigmoid_all",
        read_null_enabled=True,
        read_null_insert_block=1,
    ).eval()
    config = XAttnCLIPConfig(
        text_config={
            "vocab_size": 49412,
            "hidden_size": 64,
            "intermediate_size": 256,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "max_position_embeddings": 8,
            "hidden_act": "quick_gelu",
            "layer_norm_eps": 1e-5,
            "attention_dropout": 0.0,
            "pad_token_id": 0,
            "bos_token_id": 49406,
            "eos_token_id": 49407,
        },
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 256,
            "projection_dim": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 1,
            "num_channels": 3,
            "image_size": 32,
            "patch_size": 16,
            "hidden_act": "quick_gelu",
            "layer_norm_eps": 1e-5,
            "attention_dropout": 0.0,
        },
        projection_dim=32,
        read_null_insert_block=1,
        read_tap_blocks=[2, 3],
        ortho_tap_blocks=[0, 1],
        source_tap_blocks=[0, 1],
        read_attention_architecture="sigmoid_all",
        read_bridge_width=32,
        read_bridge_heads=4,
        early_expanded_width=128,
        source_hidden_width=128,
        trust_hidden_width=128,
        hard_text_token_id=49408,
        no_text_token_id=49409,
        any_text_token_id=49410,
        null_text_token_id=49411,
        sot_token_id=49406,
        eot_token_id=49407,
        register_min=1,
    )
    target = XAttnCLIPModel(config).eval()
    copy_clip_backbone(target, source.state_dict())
    copy_xattn(target, source.state_dict())
    return source, target


class FullXAttnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.target = tiny_pair()
        cls.pixel_values = torch.randn(2, 3, 32, 32)
        cls.input_ids = torch.zeros(3, 8, dtype=torch.long)
        cls.input_ids[:, 0] = 49406
        cls.input_ids[:, 1] = torch.tensor([1, 2, 3])
        cls.input_ids[:, 2] = 49407

    def test_original_forward_modes_parity(self):
        with torch.inference_mode():
            source = self.source.forward_modes(
                self.pixel_values, self.input_ids, return_details=True
            )
            target = self.target(
                input_ids=self.input_ids,
                pixel_values=self.pixel_values,
                mode="none",
                return_details=True,
            )
        torch.testing.assert_close(
            source["logits_per_image"], target.logits_per_image, atol=2e-5, rtol=2e-5
        )
        for key in (
            "base_image_embedding",
            "content_correction",
            "content_text_embedding",
            "source_logits",
            "read_logits",
            "trust_gate",
            "auto_read_contribution",
        ):
            torch.testing.assert_close(
                source[key], target.details[key], atol=2e-5, rtol=2e-5
            )

    def test_modes_and_read_null_contract(self):
        expected_width = self.input_ids.shape[0]
        with torch.inference_mode():
            any_output = self.target(
                input_ids=self.input_ids[:, :3],
                pixel_values=self.pixel_values[:1],
                mode="any",
            )
            read_output = self.target(
                input_ids=self.input_ids[:, :3],
                pixel_values=self.pixel_values[:1],
                mode="read",
            )
            notext_output = self.target(
                input_ids=self.input_ids[:, :3],
                pixel_values=self.pixel_values[:1],
                mode="notext",
            )
        self.assertEqual(any_output.logits_per_image.shape[1], expected_width)
        self.assertEqual(read_output.logits_per_image.shape[1], expected_width + 1)
        self.assertEqual(read_output.null_candidate_index, expected_width)
        self.assertTrue(read_output.mode_ids.eq(1).all())
        self.assertTrue(notext_output.mode_ids.eq(2).all())

    def test_save_reload_through_auto_model(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self.target.save_pretrained(destination, safe_serialization=True)
            for name in ("configuration_xattn_clip.py", "modeling_xattn_clip.py"):
                shutil.copy2(HF_EXPORT_ROOT / name, destination / name)
            reloaded = AutoModel.from_pretrained(
                destination, trust_remote_code=True
            ).eval()
            with torch.inference_mode():
                expected = self.target(
                    input_ids=self.input_ids,
                    pixel_values=self.pixel_values[:1],
                    mode="any",
                ).logits_per_image
                actual = reloaded(
                    input_ids=self.input_ids,
                    pixel_values=self.pixel_values[:1],
                    mode="any",
                ).logits_per_image
            torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)

    def test_hf_bfloat16_load_keeps_pieces_fp32(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self.target.save_pretrained(destination, safe_serialization=True)
            for name in ("configuration_xattn_clip.py", "modeling_xattn_clip.py"):
                shutil.copy2(HF_EXPORT_ROOT / name, destination / name)
            reloaded = AutoModel.from_pretrained(
                destination,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            ).eval()
        self.assertEqual(
            reloaded.vision_model.encoder.layers[0].self_attn.q_proj.weight.dtype,
            torch.bfloat16,
        )
        self.assertEqual(reloaded.read_null_token.dtype, torch.float32)
        self.assertEqual(reloaded.read_implant.read_bridge.q_proj.weight.dtype, torch.float32)
        self.assertEqual(reloaded.read_implant.trust_router.fc1.weight.dtype, torch.float32)
        self.assertEqual(reloaded.hard_text_embedding.dtype, torch.float32)
        self.assertEqual(reloaded.null_text_embedding.dtype, torch.float32)

    def test_external_bfloat_autocast_keeps_routed_logits_fp32(self):
        for mode in ("any", "read", "text", "notext", "classic", "none"):
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                output = self.target(
                    input_ids=self.input_ids,
                    pixel_values=self.pixel_values[:1],
                    mode=mode,
                )
            self.assertEqual(output.logits_per_image.dtype, torch.float32)

    def test_pieces_can_follow_ambient_autocast_without_dtype_mismatch(self):
        for mode in ("any", "read", "text", "notext", "classic", "none"):
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.float16):
                output = self.target(
                    input_ids=self.input_ids,
                    pixel_values=self.pixel_values[:1],
                    mode=mode,
                    pieces_fp32=False,
                )
            self.assertEqual(output.logits_per_image.dtype, torch.float16)
            self.assertTrue(torch.isfinite(output.logits_per_image).all())

    def test_saved_processor_control_token_ids(self):
        assets = HF_EXPORT_ROOT / "assets" / "hf_clip_tokenizer"
        spec = SimpleNamespace(
            image_size=32,
            source_vocab_size=49412,
            hard_text_token_id=49408,
            no_text_token_id=49409,
            any_text_token_id=49410,
            null_text_token_id=49411,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            self.target.config.save_pretrained(destination)
            shutil.copy2(
                HF_EXPORT_ROOT / "configuration_xattn_clip.py",
                destination / "configuration_xattn_clip.py",
            )
            save_processor(destination, spec, assets, xattn_special_tokens=True)
            processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True)
            ids = [
                processor.tokenizer.convert_tokens_to_ids(token)
                for token in ("<text>", "<notext>", "<any>", "<null>")
            ]
            self.assertEqual(ids, [49408, 49409, 49410, 49411])
            self.assertEqual(len(processor.tokenizer), 49412)

    def test_model_reference_discovers_nested_full_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "named_output" / "full_xattn_model"
            export.mkdir(parents=True)
            (export / "config.json").write_text(
                '{"model_type": "xattn_clip"}\n', encoding="utf-8"
            )
            resolved = resolve_xattn_model_reference(
                None, script_dir=root, working_dir=root
            )
            self.assertEqual(resolved, export.resolve())

    def test_model_reference_rejects_wrong_local_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "model_type=None"):
                resolve_xattn_model_reference(root, script_dir=root, working_dir=root)


if __name__ == "__main__":
    unittest.main()
