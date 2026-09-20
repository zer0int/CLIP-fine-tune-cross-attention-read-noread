from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import contextlib
import importlib
import sys

import torch
import torch.nn.functional as F


def infer_model_autocast_dtype(model: torch.nn.Module) -> Optional[torch.dtype]:
    """Infer the mixed-precision compute dtype expected by the patched CLIP.

    The supplied mechinterp CLIP intentionally stores some modules (notably
    attention Q/K/V/O) in FP16 while conv/MLP parameters may remain FP32.
    Its normal training/inference path therefore relies on CUDA autocast.
    `model.dtype` is *not* sufficient here because that property is derived
    from visual.conv1 and can report FP32 while attention projections are FP16.
    """
    try:
        dtype = model.visual.transformer.resblocks[0].attn.q_proj.weight.dtype
    except Exception:
        return None
    return dtype if dtype in (torch.float16, torch.bfloat16) else None


def model_autocast_context(model: torch.nn.Module):
    """Return the inference autocast context required by the loaded model."""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return contextlib.nullcontext()
    dtype = infer_model_autocast_dtype(model)
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


@dataclass
class LoadedModel:
    model: torch.nn.Module
    preprocess: Any
    clip_module: Any
    device: torch.device
    checkpoint: str


def import_attnclip(package_root: str | Path):
    root = Path(package_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        clip_mod = importlib.import_module("attnclip_mechinterp_xattn.clip")
    except Exception as exc:
        raise RuntimeError(
            "Could not import attnclip_mechinterp_xattn from --module-root. "
            "Point --module-root at the repository root containing that package. Original error: " + repr(exc)
        ) from exc
    return clip_mod


def load_model(
    checkpoint: str,
    *,
    package_root: str | Path,
    device: str | torch.device = "cuda",
) -> LoadedModel:
    clip_mod = import_attnclip(package_root)
    device = torch.device(device)
    model, preprocess = clip_mod.load(checkpoint, device=device, jit=False)
    model.eval()
    return LoadedModel(model=model, preprocess=preprocess, clip_module=clip_mod, device=device, checkpoint=checkpoint)


def preprocess_pil_batch(images: Sequence[Any], preprocess: Any, device: torch.device) -> torch.Tensor:
    batch = torch.stack([preprocess(im) for im in images], dim=0)
    return batch.to(device=device, non_blocking=True)


def _clone_cache(attn) -> dict[str, Optional[torch.Tensor]]:
    out = {}
    for name in ("last_q", "last_k", "last_v", "last_logits", "last_probs", "last_z"):
        value = getattr(attn, name, None)
        out[name] = None if value is None else value.detach().clone()
    return out


@contextlib.contextmanager
def _zero_last_projected_token(linear: torch.nn.Module, enabled: bool):
    if not enabled:
        yield
        return

    def hook(_module, _inputs, output):
        y = output.clone()
        y[-1] = 0
        return y

    handle = linear.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@dataclass
class VisualRun:
    embedding: torch.Tensor
    states: dict[int, torch.Tensor]
    raw_states: dict[int, torch.Tensor]
    b13_cache: Optional[dict[str, Optional[torch.Tensor]]]
    read_null_insert_block: int
    condition: str


class VisualConditionRunner:
    """Run the frozen visual spine under controlled READ_NULL interventions.

    Conditions
    ----------
    base:
        Never insert READ_NULL.
    rn_full:
        Standard checkpoint behavior: insert at B13 and keep the token thereafter.
    rn_tag:
        Insert for the configured block only, then delete before the next block.
    rn_tag_zero_v:
        Same as rn_tag, but zero READ_NULL's projected V at the insertion block.
        This is the cleanest causal approximation to a pure attention sink/NOP.
    rn_tag_zero_kv:
        Zero both projected K and V for RN at the insertion block.

    `states` always contains *aligned surviving tokens*: RN is removed before the
    state is returned to analysis code. `raw_states` preserves the actual token
    sequence for diagnostics.
    """

    VALID_CONDITIONS = {"base", "rn_full", "rn_tag", "rn_tag_zero_v", "rn_tag_zero_kv"}

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.visual = model.visual
        self.insert_block = int(getattr(self.visual, "read_null_insert_block", 13))
        if getattr(self.visual, "read_null_token", None) is None:
            raise RuntimeError("Loaded checkpoint does not contain visual.read_null_token")

    def _append_rn(self, x_tbc: torch.Tensor, token_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        token = self.visual.read_null_token if token_override is None else token_override
        token = token.to(device=x_tbc.device, dtype=x_tbc.dtype)
        token = token.view(1, 1, -1).expand(1, x_tbc.shape[1], -1)
        return torch.cat([x_tbc, token], dim=0)

    @staticmethod
    def _aligned_state(x_tbc: torch.Tensor, has_rn: bool, *, to_cpu: bool = False) -> torch.Tensor:
        x = x_tbc[:-1] if has_rn else x_tbc
        y = x.permute(1, 0, 2).detach()
        return y.cpu() if to_cpu else y.clone()

    @torch.no_grad()
    def run(
        self,
        images: torch.Tensor,
        *,
        condition: str,
        capture_blocks: Iterable[int] = (12, 13, 14, 20, 21, 23),
        capture_b13: bool = False,
        rn_token_override: Optional[torch.Tensor] = None,
        capture_raw_states: bool = False,
        captured_states_to_cpu: bool = True,
    ) -> VisualRun:
        if condition not in self.VALID_CONDITIONS:
            raise ValueError(f"Unknown condition={condition!r}; expected {sorted(self.VALID_CONDITIONS)}")
        capture_blocks = set(int(x) for x in capture_blocks)
        states: dict[int, torch.Tensor] = {}
        raw_states: dict[int, torch.Tensor] = {}
        b13_cache = None
        has_rn = False

        # IMPORTANT: this mechinterp CLIP uses mixed storage precision.  In
        # particular q/k/v/out projections are FP16 while conv/MLP weights can
        # remain FP32.  The original model expects CUDA autocast; manually
        # stepping through blocks without it produces Float-vs-Half F.linear
        # failures even though `model.dtype` reports visual.conv1's FP32 dtype.
        with model_autocast_context(self.model):
            x = self.visual._prepare_tokens(images.type(self.model.dtype))

            for i, blk in enumerate(self.visual.transformer.resblocks):
                if i == self.insert_block and condition != "base":
                    x = self._append_rn(x, rn_token_override)
                    has_rn = True

                zero_v = i == self.insert_block and condition in {"rn_tag_zero_v", "rn_tag_zero_kv"}
                zero_k = i == self.insert_block and condition == "rn_tag_zero_kv"
                with _zero_last_projected_token(blk.attn.v_proj, zero_v), _zero_last_projected_token(blk.attn.k_proj, zero_k):
                    x = blk(x, capture=(capture_b13 and i == self.insert_block))

                if capture_b13 and i == self.insert_block:
                    b13_cache = _clone_cache(blk.attn)

                if i in capture_blocks:
                    if capture_raw_states:
                        raw = x.permute(1, 0, 2).detach()
                        raw_states[i] = raw.cpu().clone() if captured_states_to_cpu else raw.clone()
                    states[i] = self._aligned_state(x, has_rn, to_cpu=captured_states_to_cpu)

                if i == self.insert_block and condition in {"rn_tag", "rn_tag_zero_v", "rn_tag_zero_kv"}:
                    x = x[:-1]
                    has_rn = False

            embedding = self.visual._finalize_cls(x).float().detach()
        return VisualRun(
            embedding=embedding,
            states=states,
            raw_states=raw_states,
            b13_cache=b13_cache,
            read_null_insert_block=self.insert_block,
            condition=condition,
        )

    @torch.no_grad()
    def pre_b13_state(self, images: torch.Tensor) -> torch.Tensor:
        with model_autocast_context(self.model):
            x = self.visual._prepare_tokens(images.type(self.model.dtype))
            for i in range(self.insert_block):
                x = self.visual.transformer.resblocks[i](x)
        return x.detach().clone()

    @torch.no_grad()
    def block13_pair(self, images: torch.Tensor) -> dict[str, Any]:
        """Exact baseline-vs-RN B13 decomposition inputs/caches.

        Returns the common pre-B13 ordinary-token state, baseline and RN block
        outputs, attention caches, and attention/MLP pathway deltas.
        """
        x_pre = self.pre_b13_state(images)
        blk = self.visual.transformer.resblocks[self.insert_block]

        with model_autocast_context(self.model):
            x_base_in = x_pre
            x_base = blk(x_base_in, capture=True)
            base_cache = _clone_cache(blk.attn)

            x_rn_in = self._append_rn(x_pre)
            x_rn = blk(x_rn_in, capture=True)
            rn_cache = _clone_cache(blk.attn)

            # Reconstruct attention pathway explicitly from captured per-head z.
            def attn_out_from_z(z: torch.Tensor) -> torch.Tensor:
                # z [B,H,T,D] -> [T,B,E] -> out_proj
                b, h, t, d = z.shape
                merged = z.permute(0, 2, 1, 3).reshape(b, t, h * d).permute(1, 0, 2)
                return blk.attn.out_proj(merged)

            base_attn = attn_out_from_z(base_cache["last_z"])
            rn_attn = attn_out_from_z(rn_cache["last_z"])
            rn_attn_ord = rn_attn[:-1]

            base_mid = x_base_in + base_attn
            rn_mid = x_pre + rn_attn_ord
            base_mlp = blk.mlp(blk.ln_2(base_mid))
            rn_mlp = blk.mlp(blk.ln_2(rn_mid))

        return {
            "pre": x_pre.permute(1, 0, 2).float(),
            "base_post": x_base.permute(1, 0, 2).float(),
            "rn_post": x_rn[:-1].permute(1, 0, 2).float(),
            "base_cache": base_cache,
            "rn_cache": rn_cache,
            "base_attn_out": base_attn.permute(1, 0, 2).float(),
            "rn_attn_out": rn_attn_ord.permute(1, 0, 2).float(),
            "delta_attn": (rn_attn_ord - base_attn).permute(1, 0, 2).float(),
            "delta_mlp": (rn_mlp - base_mlp).permute(1, 0, 2).float(),
            "delta_post": (x_rn[:-1] - x_base).permute(1, 0, 2).float(),
        }

    @torch.no_grad()
    def continue_from_post_b13(self, state_btd: torch.Tensor) -> torch.Tensor:
        """Continue a RN-free aligned post-B13 state through B14..end."""
        with model_autocast_context(self.model):
            x = state_btd.to(device=self.visual.class_embedding.device, dtype=self.model.dtype).permute(1, 0, 2)
            for i in range(self.insert_block + 1, len(self.visual.transformer.resblocks)):
                x = self.visual.transformer.resblocks[i](x)
            out = self.visual._finalize_cls(x).float()
        return out


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=eps)
