from __future__ import annotations

from transformers import CLIPConfig


class XAttnCLIPConfig(CLIPConfig):
    """Configuration for the final RN + PIECES x-attention CLIP model."""

    model_type = "xattn_clip"
    has_no_defaults_at_init = True

    def __init__(
        self,
        *args,
        read_null_insert_block: int = 0,
        read_tap_blocks: list[int] | tuple[int, ...] = (0,),
        ortho_tap_blocks: list[int] | tuple[int, ...] = (0,),
        source_tap_blocks: list[int] | tuple[int, ...] = (0,),
        read_attention_architecture: str = "sigmoid_all",
        read_bridge_width: int = 256,
        read_bridge_heads: int = 4,
        early_expanded_width: int = 4096,
        source_hidden_width: int = 256,
        trust_hidden_width: int = 128,
        hard_text_token_id: int = 49408,
        no_text_token_id: int = 49409,
        any_text_token_id: int = 49410,
        null_text_token_id: int = 49411,
        sot_token_id: int = 49406,
        eot_token_id: int = 49407,
        correction: bool = True,
        default_mode: str = "any",
        register_norm_threshold: float = 60.0,
        register_min: int = 1,
        register_max: int = 0,
        source_checkpoint_fingerprint: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.read_null_enabled = True
        self.read_null_insert_block = int(read_null_insert_block)
        self.read_tap_blocks = [int(item) for item in read_tap_blocks]
        self.ortho_tap_blocks = [int(item) for item in ortho_tap_blocks]
        self.source_tap_blocks = [int(item) for item in source_tap_blocks]
        self.read_attention_architecture = str(read_attention_architecture)
        self.read_bridge_width = int(read_bridge_width)
        self.read_bridge_heads = int(read_bridge_heads)
        self.early_expanded_width = int(early_expanded_width)
        self.source_hidden_width = int(source_hidden_width)
        self.trust_hidden_width = int(trust_hidden_width)
        self.hard_text_token_id = int(hard_text_token_id)
        self.no_text_token_id = int(no_text_token_id)
        self.any_text_token_id = int(any_text_token_id)
        self.null_text_token_id = int(null_text_token_id)
        self.sot_token_id = int(sot_token_id)
        self.eot_token_id = int(eot_token_id)
        self.correction = bool(correction)
        self.default_mode = str(default_mode)
        self.register_norm_threshold = float(register_norm_threshold)
        self.register_min = int(register_min)
        self.register_max = int(register_max)
        self.source_checkpoint_fingerprint = source_checkpoint_fingerprint
        self.auto_map = {
            "AutoConfig": "configuration_xattn_clip.XAttnCLIPConfig",
            "AutoModel": "modeling_xattn_clip.XAttnCLIPModel",
        }
        self.architectures = ["XAttnCLIPModel"]
        self._validate_xattn()

    def _validate_xattn(self) -> None:
        layers = int(self.vision_config.num_hidden_layers)
        if not 0 <= self.read_null_insert_block < layers:
            raise ValueError("read_null_insert_block is outside the vision stack")
        for name in ("read_tap_blocks", "ortho_tap_blocks", "source_tap_blocks"):
            blocks = getattr(self, name)
            if not blocks or any(item < 0 or item >= layers for item in blocks):
                raise ValueError(f"{name} contains invalid visual blocks")
        if any(item < self.read_null_insert_block for item in self.read_tap_blocks):
            raise ValueError("read_tap_blocks must follow RN insertion")
        if self.read_attention_architecture not in {
            "softmax",
            "sigmoid_mass",
            "sigmoid_all",
        }:
            raise ValueError("Unknown read_attention_architecture")
        if self.read_attention_architecture == "sigmoid_mass":
            raise ValueError("READ_NULL is incompatible with sigmoid_mass attention")
        if self.read_bridge_width % self.read_bridge_heads:
            raise ValueError("read_bridge_width must be divisible by read_bridge_heads")
        if self.trust_hidden_width < 2 or self.trust_hidden_width % 2:
            raise ValueError("trust_hidden_width must be an even value >= 2")
        if self.default_mode not in {
            "any",
            "read",
            "text",
            "notext",
            "classic",
            "none",
        }:
            raise ValueError("Unknown default_mode")
        if int(self.text_config.vocab_size) <= max(
            self.hard_text_token_id,
            self.no_text_token_id,
            self.any_text_token_id,
            self.null_text_token_id,
        ):
            raise ValueError("text vocabulary does not contain all control tokens")
