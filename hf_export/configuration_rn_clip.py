from __future__ import annotations

from transformers import CLIPConfig


class RNCLIPConfig(CLIPConfig):
    """CLIP config for RN insertion plus optional late content correction."""

    model_type = "rn_clip"

    def __init__(
        self,
        *args,
        read_null_insert_block: int = 0,
        read_tap_blocks: list[int] | tuple[int, ...] = (0,),
        read_attention_architecture: str = "sigmoid_all",
        read_bridge_width: int = 256,
        read_bridge_heads: int = 4,
        correction: bool = True,
        register_norm_threshold: float = 60.0,
        register_min: int = 1,
        register_max: int = 0,
        source_checkpoint_fingerprint: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.read_null_insert_block = int(read_null_insert_block)
        self.read_tap_blocks = [int(item) for item in read_tap_blocks]
        self.read_attention_architecture = str(read_attention_architecture)
        self.read_bridge_width = int(read_bridge_width)
        self.read_bridge_heads = int(read_bridge_heads)
        self.correction = bool(correction)
        self.register_norm_threshold = float(register_norm_threshold)
        self.register_min = int(register_min)
        self.register_max = int(register_max)
        self.source_checkpoint_fingerprint = source_checkpoint_fingerprint
        self.auto_map = {
            "AutoConfig": "configuration_rn_clip.RNCLIPConfig",
            "AutoModel": "modeling_rn_clip.RNCLIPModel",
        }
        self.architectures = ["RNCLIPModel"]

        layers = int(self.vision_config.num_hidden_layers)
        if not 0 <= self.read_null_insert_block < layers:
            raise ValueError("read_null_insert_block is outside the vision stack")
        if not self.read_tap_blocks or any(
            item < self.read_null_insert_block or item >= layers
            for item in self.read_tap_blocks
        ):
            raise ValueError("read_tap_blocks must be valid blocks after RN insertion")
        if self.read_attention_architecture not in {"softmax", "sigmoid_all"}:
            raise ValueError(
                "correction export supports read_attention_architecture "
                "'softmax' or 'sigmoid_all'"
            )
