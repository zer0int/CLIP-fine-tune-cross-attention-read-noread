from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPOutput

try:  # local package import and Hugging Face custom-code import are both supported
    from .configuration_rn_clip import RNCLIPConfig
except ImportError:  # pragma: no cover - used when this folder is on sys.path
    from configuration_rn_clip import RNCLIPConfig


class VisualContentPool(nn.Module):
    """Pool spatial visual tokens with checkpoint-selected attention normalization."""

    def __init__(self, config: RNCLIPConfig):
        super().__init__()
        vision_width = int(config.vision_config.hidden_size)
        pool_width = int(config.read_bridge_width)
        heads = int(config.read_bridge_heads)
        if pool_width % heads:
            raise ValueError("read_bridge_width must be divisible by read_bridge_heads")
        self.pool_width = pool_width
        self.heads = heads
        self.head_dim = pool_width // heads
        self.architecture = config.read_attention_architecture
        self.vision_ln = nn.LayerNorm(vision_width, eps=config.vision_config.layer_norm_eps)
        self.query = nn.Parameter(torch.empty(heads, self.head_dim))
        self.k_proj = nn.Linear(vision_width, pool_width)
        self.v_proj = nn.Linear(vision_width, pool_width)
        self.out_proj = nn.Linear(pool_width, config.projection_dim)
        if self.architecture == "sigmoid_all":
            self.sigmoid_head_bias = nn.Parameter(torch.zeros(heads))
        else:
            self.register_parameter("sigmoid_head_bias", None)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # Correction arithmetic is an FP32 island; the CLIP backbone keeps its HF dtype.
        device_type = visual_tokens.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            tokens = visual_tokens.float()
            batch, token_count, _ = tokens.shape
            x = F.layer_norm(
                tokens,
                self.vision_ln.normalized_shape,
                self.vision_ln.weight.float(),
                self.vision_ln.bias.float(),
                self.vision_ln.eps,
            )
            keys = F.linear(
                x, self.k_proj.weight.float(), self.k_proj.bias.float()
            ).view(batch, token_count, self.heads, self.head_dim)
            values = F.linear(
                x, self.v_proj.weight.float(), self.v_proj.bias.float()
            ).view(batch, token_count, self.heads, self.head_dim)
            keys = keys.permute(0, 2, 1, 3)
            values = values.permute(0, 2, 1, 3)
            logits = torch.einsum(
                "hd,bhtd->bht", self.query.float(), keys
            ) / math.sqrt(self.head_dim)

            keep = torch.ones((batch, token_count), dtype=torch.bool, device=tokens.device)
            keep[:, 0] = False  # CLS
            if token_count < 3:
                raise ValueError("Visual sequence is too short to exclude CLS and RN")
            keep[:, -1] = False  # RN
            if self.architecture == "sigmoid_all":
                valid_count = keep.sum(dim=-1).clamp(min=1).float()
                adjusted = logits + self.sigmoid_head_bias.float()[None, :, None]
                adjusted = adjusted - valid_count.log()[:, None, None]
                weights = torch.sigmoid(adjusted) * keep[:, None, :]
            else:
                weights = logits.masked_fill(~keep[:, None, :], torch.finfo(logits.dtype).min)
                weights = weights.softmax(dim=-1)
            pooled = torch.einsum("bht,bhtd->bhd", weights, values)
            return F.linear(
                pooled.reshape(batch, self.pool_width),
                self.out_proj.weight.float(),
                self.out_proj.bias.float(),
            )


class RNCLIPModel(CLIPModel):
    """HF CLIP with an inserted RN token and optional late visual correction."""

    config_class = RNCLIPConfig
    # Match oaiclip storage: RN/correction parameters remain FP32; RN is cast only at insertion.
    # Transformers 5.16.1 strict applies for both fp16 and bf16 loads.
    _keep_in_fp32_modules = ["read_null_token", "content_pool", "content_tap_logits"]
    _keep_in_fp32_modules_strict = [
        "read_null_token",
        "content_pool",
        "content_tap_logits",
    ]

    def __init__(self, config: RNCLIPConfig, debug: bool = False):
        super().__init__(config)
        width = int(config.vision_config.hidden_size)
        self.read_null_token = nn.Parameter(torch.zeros(width))
        self.content_pool = VisualContentPool(config)
        self.content_tap_logits = nn.Parameter(torch.zeros(len(config.read_tap_blocks)))
        self.post_init()
        if debug:
            self.debug_summary()

    def debug_summary(self) -> None:
        """Print the inference-relevant RN/correction configuration."""
        vision = self.config.vision_config
        print(
            "[RN correction] "
            f"ViT-L/14@{vision.image_size}; "
            f"RN before B{self.config.read_null_insert_block}; "
            f"taps={self.config.read_tap_blocks}; "
            f"attention={self.config.read_attention_architecture}; "
            f"correction_default={self.config.correction}; correction_compute=fp32"
        )
        print(
            "[RN correction] register config: "
            f"threshold={self.config.register_norm_threshold}, "
            f"min={self.config.register_min}, max={self.config.register_max}"
        )

    def _vision_with_rn(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False, **kwargs
    ) -> tuple[BaseModelOutputWithPooling, list[torch.Tensor]]:
        vision = self.vision_model
        hidden = vision.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )
        hidden = vision.pre_layrnorm(hidden)
        captured: list[torch.Tensor] = []
        taps = set(int(item) for item in self.config.read_tap_blocks)
        inserted = False
        for block_index, layer in enumerate(vision.encoder.layers):
            if block_index == int(self.config.read_null_insert_block):
                rn = self.read_null_token.to(device=hidden.device, dtype=hidden.dtype)
                rn = rn.view(1, 1, -1).expand(hidden.shape[0], 1, -1)
                hidden = torch.cat((hidden, rn), dim=1)
                inserted = True
            hidden = layer(hidden, None, **kwargs)
            if block_index in taps:
                captured.append(hidden)
        if not inserted:
            raise RuntimeError("RN token was never inserted")
        if len(captured) != len(self.config.read_tap_blocks):
            raise RuntimeError("Not all correction tap blocks were captured")
        pooled = vision.post_layernorm(hidden[:, 0, :])
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden, pooler_output=pooled
        ), captured

    def _corrected_image_output(
        self,
        pixel_values: torch.Tensor,
        *,
        interpolate_pos_encoding: bool = False,
        correction: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        """Pool B20/B21 spatial states, exclude RN, mix outputs, and add to final CLS.

        This further reduces the tendency to misclassify images as text in edge cases.
        """
        outputs, tapped = self._vision_with_rn(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs
        )
        base = self.visual_projection(outputs.pooler_output)
        enabled = self.config.correction if correction is None else bool(correction)
        if enabled:
            weights = self.content_tap_logits.float().softmax(dim=0)
            corrections = torch.stack([self.content_pool(state) for state in tapped])
            correction_vector = torch.einsum("k,kbd->bd", weights, corrections)
            base = base + correction_vector.to(base.dtype)
        outputs.pooler_output = base
        return outputs

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
        correction: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        return self._corrected_image_output(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            correction=correction,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_loss: bool | None = None,
        interpolate_pos_encoding: bool = False,
        correction: bool | None = None,
        **kwargs,
    ) -> CLIPOutput:
        vision_outputs = self.get_image_features(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            correction=correction,
            **kwargs,
        )
        text_outputs = self.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        logits_per_text = text_embeds @ image_embeds.t().to(text_embeds.device)
        logits_per_text = logits_per_text * self.logit_scale.exp().to(text_embeds.device)
        logits_per_image = logits_per_text.t()
        if return_loss:
            labels = torch.arange(logits_per_text.shape[0], device=logits_per_text.device)
            text_loss = torch.nn.functional.cross_entropy(logits_per_text, labels)
            image_loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss = (text_loss + image_loss) / 2
        else:
            loss = None
        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
