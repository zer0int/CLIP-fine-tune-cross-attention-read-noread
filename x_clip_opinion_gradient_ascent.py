#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLIP gradient-ascent "opinion" for the source-conditioned AnyText model.

The experiment keeps the original wordword lexical optimization, but evaluates
three distinct groups:

    forced <notext>:
        corrected content image embedding <-> ordinary optimized text

    forced <text>:
        query-conditioned visual read embedding <-> optimized "<text> ..." text

    model-selected <any>:
        the trained candidate-conditioned router supplies a continuous read gate,
        and the score uses the model's actual inference equation:

            content + source_gate * trust_gate * auto_read_scale * positive(read - null)

Unlike the previous script, automatic samples do NOT optimize a separate Gumbel
mode bit.  Their routing is genuinely chosen by the frozen model.  For compact
terminal reporting only, a gate >= 0.5 is labelled as self-selected <text> and a
gate < 0.5 as self-selected <notext>; the optimized score still uses the full
continuous gate.

The current <null> abstention candidate is also evaluated as a fixed read
baseline.  This lets forced-read and automatic opinions report whether the
optimized phrase actually beats "no readable word".

The image and model parameters remain frozen.  Gradients flow only into the
optimized lexical token distributions.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import kornia
import kornia.augmentation as kaugs
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style, just_fix_windows_console
from torch.cuda.amp import GradScaler, autocast

import oaiclip as clip
from utils_clip_loader.clip_anything_to_openai import load_openai_clip_anything

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

OPINION_SCRIPT_REVISION = "current-pieces-v6-fp32-score-parity-r3"


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Get forced-<notext>, forced-<text>, and learned-<any> CLIP opinions."
    )
    parser.add_argument("--use_model", default="zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX", help="Current merged PIECES model (or compatible compact checkpoint).")
    parser.add_argument(
        "--use_image",
        type=str,
        default="image_sets/pink_elephant/savanna_eleword_p.png",
        help="Path to one image.",
    )
    parser.add_argument(
        "--img_folder",
        type=str,
        default="None",
        help="Image folder for batch processing. Literal 'None' uses --use_image.",
    )
    parser.add_argument(
        "--batch_size",
        default=12,
        type=int,
        help="Number of simultaneous candidate opinions.",
    )
    parser.add_argument(
        "--num_tokens",
        default=8,
        type=int,
        help="Number of optimized lexical BPE positions.",
    )
    parser.add_argument(
        "--iterations",
        default=301,
        type=int,
    )
    parser.add_argument(
        "--checkin_step",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--repeats",
        default=4,
        type=int,
        help="Augmented image views per candidate.",
    )
    parser.add_argument(
        "--lse_tau",
        default=0.07,
        type=float,
        help="LogSumExp temperature across image augmentations.",
    )
    parser.add_argument(
        "--token_gumbel_temp",
        default=1000.0,
        type=float,
        help="Original wordword lexical Gumbel temperature.",
    )
    parser.add_argument(
        "--mode_gumbel_temp",
        default=1.0,
        type=float,
        help="Deprecated compatibility option; learned <any> routing now comes from the model.",
    )
    parser.add_argument(
        "--mode_strategy",
        default="compare",
        choices=["compare", "split", "auto", "ordinary", "notext", "read"],
        help=(
            "compare = thirds forced <notext>/forced <text>/model <any>; "
            "split = halves forced <notext>/<text>; auto = all learned <any>; "
            "ordinary/notext/read force one lane."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional fixed prompt inserted before optimized tokens in both lanes.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5.0,
        type=float,
    )
    parser.add_argument(
        "--dump_embeds",
        action="store_true",
        help="Dump the best chosen text embeddings.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Reproducibility / image utilities
# -----------------------------------------------------------------------------

def fix_random_seed(seed: int = 6247423):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def load_image(img_path: str, side_x: int, side_y: int) -> torch.Tensor:
    image = Image.open(img_path).convert("RGB")
    tensor = torch.tensor(np.array(image), device="cuda")
    tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
    return F.interpolate(tensor, (side_x, side_y), mode="bilinear", align_corners=False)


# -----------------------------------------------------------------------------
# Candidate modes
# -----------------------------------------------------------------------------

MODE_NOTEXT = 0
MODE_READ = 1
MODE_AUTO = 2

MODE_NAMES = {
    MODE_NOTEXT: "forced-notext",
    MODE_READ: "forced-read",
    MODE_AUTO: "self-selected",
}

SECTION_HEADINGS = {
    MODE_NOTEXT: "----------------- forced <notext> below -----------------",
    MODE_READ: "----------------- forced <text> below -------------------",
    MODE_AUTO: "----------------- model self-selected <any> below -------",
}

# Orchid: deliberately between pink and purple, distinct from Colorama blue.
SELF_CHOSEN_COLOR = "\x1b[38;2;218;112;214m"
SECTION_END = "----------------------------------------------------------"


def make_mode_codes(batch_size: int, strategy: str) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if strategy == "ordinary":
        codes = [MODE_NOTEXT] * batch_size
    elif strategy == "notext":
        codes = [MODE_NOTEXT] * batch_size
    elif strategy == "read":
        codes = [MODE_READ] * batch_size
    elif strategy == "auto":
        codes = [MODE_AUTO] * batch_size
    elif strategy == "split":
        n_ordinary = batch_size // 2
        codes = [MODE_NOTEXT] * n_ordinary + [MODE_READ] * (batch_size - n_ordinary)
    elif strategy == "compare":
        # Keep all three groups non-empty whenever batch_size permits.
        n_ordinary = batch_size // 3
        n_read = batch_size // 3
        n_auto = batch_size - n_ordinary - n_read
        codes = (
            [MODE_NOTEXT] * n_ordinary
            + [MODE_READ] * n_read
            + [MODE_AUTO] * n_auto
        )
    else:
        raise ValueError(strategy)

    return torch.tensor(codes, dtype=torch.long)


# -----------------------------------------------------------------------------
# Differentiable optimized text
# -----------------------------------------------------------------------------

@dataclass
class TextSample:
    lexical_st: torch.Tensor
    lexical_ids: torch.Tensor
    ordinary_token_ids: torch.Tensor
    read_token_ids: torch.Tensor
    ordinary_eot_indices: torch.Tensor
    read_eot_indices: torch.Tensor


class Pars(nn.Module):
    """Optimized lexical tokens with routing delegated to the frozen model.

    Lexical tokens may not sample any routing or sequence-control token.  The
    forced lanes are assembled explicitly, while automatic samples are scored by
    the trained AnyText router.
    """

    def __init__(
        self,
        batch_size: int,
        text_pos_shape: int,
        many_tokens: int,
        prompt_ids: Sequence[int],
        vocab_size: int,
        sot_id: int,
        eot_id: int,
        hard_text_id: int,
        no_text_id: int,
        any_text_id: int,
        null_text_id: int,
        mode_strategy: str,
        token_gumbel_temp: float,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.text_pos_shape = int(text_pos_shape)
        self.many_tokens = int(many_tokens)
        self.prompt_ids = [int(x) for x in prompt_ids]
        self.vocab_size = int(vocab_size)
        self.sot_id = int(sot_id)
        self.eot_id = int(eot_id)
        self.hard_text_id = int(hard_text_id)
        self.no_text_id = int(no_text_id)
        self.any_text_id = int(any_text_id)
        self.null_text_id = int(null_text_id)
        self.token_gumbel_temp = float(token_gumbel_temp)

        self.register_buffer(
            "mode_codes",
            make_mode_codes(batch_size, mode_strategy),
            persistent=False,
        )

        ordinary_eot = 1 + len(self.prompt_ids) + self.many_tokens
        read_eot = ordinary_eot + 1
        if read_eot >= self.text_pos_shape:
            raise ValueError(
                f"Sequence too long: read EOT index {read_eot}, "
                f"context length {self.text_pos_shape}."
            )

        self.register_buffer(
            "ordinary_eot_indices",
            torch.full((batch_size,), ordinary_eot, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "read_eot_indices",
            torch.full((batch_size,), read_eot, dtype=torch.long),
            persistent=False,
        )

        token_logits = torch.zeros(
            batch_size,
            many_tokens,
            vocab_size,
        ).normal_()
        self.normu = nn.Parameter(token_logits.cuda())

        lexical_allowed = torch.ones(vocab_size, dtype=torch.bool)
        for token_id in (
            0,
            self.sot_id,
            self.eot_id,
            self.hard_text_id,
            self.no_text_id,
            self.any_text_id,
            self.null_text_id,
        ):
            if 0 <= token_id < vocab_size:
                lexical_allowed[token_id] = False
        self.register_buffer(
            "lexical_allowed",
            lexical_allowed,
            persistent=False,
        )

    def _sample_lexical(self) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.normu.masked_fill(
            ~self.lexical_allowed.view(1, 1, -1),
            -1e9,
        )
        lexical_st = F.gumbel_softmax(
            logits,
            tau=self.token_gumbel_temp,
            dim=-1,
            hard=True,
        )
        lexical_ids = lexical_st.argmax(dim=-1)
        return lexical_st, lexical_ids

    def _build_ids(
        self,
        lexical_ids: torch.Tensor,
        read_mode: bool,
    ) -> torch.Tensor:
        ids = torch.zeros(
            self.batch_size,
            self.text_pos_shape,
            dtype=torch.long,
            device=lexical_ids.device,
        )
        ids[:, 0] = self.sot_id

        position = 1
        if read_mode:
            ids[:, position] = self.hard_text_id
            position += 1

        if self.prompt_ids:
            prompt = torch.tensor(
                self.prompt_ids,
                dtype=torch.long,
                device=ids.device,
            )
            ids[:, position : position + len(self.prompt_ids)] = prompt
            position += len(self.prompt_ids)

        ids[:, position : position + self.many_tokens] = lexical_ids
        position += self.many_tokens
        ids[:, position] = self.eot_id
        return ids

    def forward(self) -> TextSample:
        lexical_st, lexical_ids = self._sample_lexical()
        return TextSample(
            lexical_st=lexical_st,
            lexical_ids=lexical_ids,
            ordinary_token_ids=self._build_ids(lexical_ids, read_mode=False),
            read_token_ids=self._build_ids(lexical_ids, read_mode=True),
            ordinary_eot_indices=self.ordinary_eot_indices,
            read_eot_indices=self.read_eot_indices,
        )


# -----------------------------------------------------------------------------
# Differentiable text encoder equivalent to the hard model
# -----------------------------------------------------------------------------

@dataclass
class EncodedSoftText:
    text_embedding: torch.Tensor
    eot_hidden_pre_ln: torch.Tensor


def _add_model_positional_embedding(model, x: torch.Tensor) -> torch.Tensor:
    if getattr(model, "use_positional_embedding_res", False):
        pos = model.positional_embedding.to(device=x.device, dtype=x.dtype)
        pos_res = model.positional_embedding_res.to(device=x.device, dtype=x.dtype)
        mask1 = model.mask1.to(device=x.device, dtype=x.dtype)
        mask2 = model.mask2.to(device=x.device, dtype=x.dtype)
        return x + pos * mask1 + pos_res * mask2

    return x + model.positional_embedding.to(device=x.device, dtype=x.dtype)


def clip_encode_soft_text(
    model,
    lexical_st: torch.Tensor,
    token_ids: torch.Tensor,
    eot_indices: torch.Tensor,
    prompt_len: int,
    read_mode: bool,
) -> EncodedSoftText:
    """
    Encode a straight-through one-hot lexical distribution.

    Fixed positions use normal integer token lookup. Optimized positions use a
    differentiable distribution over the token embedding matrix. The dedicated
    hard_text_embedding replaces the reserved token row exactly as in
    model._encode_text_hidden().
    """
    x = model.token_embedding(token_ids).type(model.dtype)

    # Replace fixed <text> lookup by the dedicated trained parameter.
    hard_fixed = token_ids.eq(model.hard_text_token_id).unsqueeze(-1)
    if hard_fixed.any():
        hard = model.hard_text_embedding.to(
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, -1)
        x = torch.where(hard_fixed, hard, x)

    null_fixed = token_ids.eq(model.null_text_token_id).unsqueeze(-1)
    if null_fixed.any():
        null = model.null_text_embedding.to(
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, -1)
        x = torch.where(null_fixed, null, x)

    lexical_start = 1 + prompt_len + (1 if read_mode else 0)
    lexical_end = lexical_start + lexical_st.shape[1]

    weight = model.token_embedding.weight

    # IMPORTANT: keep the forward value IDENTICAL to a normal integer token lookup,
    # while retaining the straight-through gradient through the lexical distribution.
    #
    # A raw one-hot matmul under CUDA autocast can differ slightly from
    # token_embedding(lexical_ids), despite being mathematically equivalent.  The
    # hard lookup below supplies the exact deployed forward value; the zero-valued
    # (soft - soft.detach()) term supplies the lexical gradient only.
    lexical_ids = lexical_st.argmax(dim=-1)
    hard_lexical_embed = F.embedding(lexical_ids, weight)
    with torch.autocast(device_type="cuda", enabled=False):
        soft_lexical_embed = lexical_st.float() @ weight.float()
    soft_lexical_embed = soft_lexical_embed.to(hard_lexical_embed.dtype)
    # Parenthesize the zero-valued straight-through term BEFORE adding it to
    # the hard lookup.  `(hard + soft) - soft.detach()` is NOT forward-exact in
    # floating point because the intermediate hard+soft addition rounds.
    # `hard + (soft - soft.detach())` is forward-identical to hard lookup while
    # retaining d/dsoft = 1 for the lexical optimizer.
    lexical_embed = hard_lexical_embed + (
        soft_lexical_embed - soft_lexical_embed.detach()
    )

    x = x.clone()
    x[:, lexical_start:lexical_end] = lexical_embed.to(x.dtype)
    x = _add_model_positional_embedding(model, x)

    hidden_pre_ln = model.transformer(
        x.permute(1, 0, 2)
    ).permute(1, 0, 2)
    hidden_post_ln = model.ln_final(hidden_pre_ln).type(model.dtype)

    batch = torch.arange(
        hidden_post_ln.shape[0],
        device=hidden_post_ln.device,
    )
    feature = (
        hidden_post_ln[batch, eot_indices]
        @ model.text_projection
    )

    return EncodedSoftText(
        text_embedding=feature,
        eot_hidden_pre_ln=hidden_pre_ln[batch, eot_indices],
    )


# -----------------------------------------------------------------------------
# Current PIECES FP32 score arithmetic
# -----------------------------------------------------------------------------

def _opinion_fp32_normalize(value: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Match oaiclip.model._fp32_normalize under the surrounding AMP context."""
    with torch.autocast(device_type="cuda", enabled=False):
        return F.normalize(value.float(), dim=dim, eps=1.0e-12)


def _opinion_fp32_scaled_matmul(
    scale: torch.Tensor,
    left: torch.Tensor,
    right_t: torch.Tensor,
) -> torch.Tensor:
    """Match oaiclip.model._fp32_scaled_matmul exactly in operation order."""
    with torch.autocast(device_type="cuda", enabled=False):
        return scale.float() * (left.float() @ right_t.float())


def _opinion_fp32_scaled_einsum(
    equation: str,
    scale: torch.Tensor,
    *operands: torch.Tensor,
) -> torch.Tensor:
    """Match oaiclip.model._fp32_scaled_einsum exactly under AMP."""
    with torch.autocast(device_type="cuda", enabled=False):
        return scale.float() * torch.einsum(
            equation,
            *(operand.float() for operand in operands),
        )


# -----------------------------------------------------------------------------
# Hard PIECE OF CLIP opinion objective
# -----------------------------------------------------------------------------

@dataclass
class OpinionDiagnostics:
    sim_content: torch.Tensor
    sim_read: torch.Tensor
    sim_auto: torch.Tensor
    sim_chosen: torch.Tensor
    route_gate: torch.Tensor
    trust_gate: torch.Tensor
    source_gate: torch.Tensor
    auto_contribution: torch.Tensor
    null_score: torch.Tensor
    presence_mean: torch.Tensor
    glyph_mean: torch.Tensor
    glyph_max: torch.Tensor



@torch.no_grad()
def _check_current_forward_value_parity(
    visual_input: torch.Tensor,
    model,
    sample: TextSample,
    content_logits_all: torch.Tensor,
    read_logits_all: torch.Tensor,
    auto_logits_all: torch.Tensor,
) -> None:
    """One-time runtime proof that the hand-differentiable scorer has current values.

    We compare whole image x candidate matrices for each lane separately, so the
    candidate batch shape matches the manual path exactly.  This catches stale
    source/read/null/router/content equations without changing the optimization.

    The automatic lane is VALUE-identical to deployed forward_modes().  Its search
    gradient is intentionally a straight-through surrogate through positive(read-null),
    because the deployed forward detaches that expert contribution.
    """
    if bool(getattr(model, "_clip_opinion_forward_parity_checked", False)):
        return

    # All three explicit controls occupy the same position immediately after SOT.
    read_tokens = sample.read_token_ids.detach()
    notext_tokens = read_tokens.clone()
    any_tokens = read_tokens.clone()
    notext_tokens[:, 1] = int(model.no_text_token_id)
    any_tokens[:, 1] = int(model.any_text_token_id)

    ref_notext = model.forward_modes(
        visual_input,
        notext_tokens,
        apply_content_correction=True,
        return_details=False,
    )[0].float()
    ref_read = model.forward_modes(
        visual_input,
        read_tokens,
        apply_content_correction=True,
        return_details=False,
    )[0].float()
    ref_any = model.forward_modes(
        visual_input,
        any_tokens,
        apply_content_correction=True,
        return_details=False,
    )[0].float()

    errors = {
        "<notext>": float((ref_notext - content_logits_all.float()).abs().max().item()),
        "<text>": float((ref_read - read_logits_all.float()).abs().max().item()),
        "<any>": float((ref_any - auto_logits_all.float()).abs().max().item()),
    }
    print(
        "[forward parity] "
        + " | ".join(f"{name} max_abs={value:.6g}" for name, value in errors.items())
    )

    # Same computational shapes are used on both sides.  A few 1e-3 of AMP
    # roundoff is acceptable; anything larger means this script and forward_modes()
    # have actually diverged.
    tolerance = 5.0e-3
    bad = {name: value for name, value in errors.items() if value > tolerance}
    if bad:
        raise RuntimeError(
            "Opinion scorer no longer matches current model.forward_modes(): "
            + ", ".join(f"{name}={value:.6g}" for name, value in bad.items())
            + f" (tolerance {tolerance})"
        )

    setattr(model, "_clip_opinion_forward_parity_checked", True)


def hard_piece_opinion(
    image: torch.Tensor,
    model,
    lats: Pars,
    normalizer: nn.Module,
    augment: nn.Module,
    repeats: int,
    lse_tau: float,
    iteration: int,
    null_text_info: Mapping[str, torch.Tensor],
):
    """Value-exact early-branch AnyText opinion objective.

    The score matches current ``forward_modes``:

        content + source_gate * trust_gate * auto_scale * positive(read - null)

    The model and both gates remain frozen.  The deployed <any> forward detaches
    positive(read-null), and CandidateTrustRouter also detaches all of its inputs.
    If reproduced literally, <any> gradient ascent would therefore search only the
    CONTENT term.  For this diagnostic only, positive relative-read keeps a
    straight-through lexical gradient so the optimizer can discover phrases that
    actually raise the frozen model's read contribution.  The FORWARD VALUE is
    unchanged and is checked once against model.forward_modes().
    """
    batch_size = lats.batch_size
    sample = lats()

    ordinary = clip_encode_soft_text(
        model=model,
        lexical_st=sample.lexical_st,
        token_ids=sample.ordinary_token_ids,
        eot_indices=sample.ordinary_eot_indices,
        prompt_len=len(lats.prompt_ids),
        read_mode=False,
    )
    read = clip_encode_soft_text(
        model=model,
        lexical_st=sample.lexical_st,
        token_ids=sample.read_token_ids,
        eot_indices=sample.read_eot_indices,
        prompt_len=len(lats.prompt_ids),
        read_mode=True,
    )

    # B candidates x R independently augmented views, ordered [R, B].
    image_rep = image[:, :3].expand(
        batch_size * repeats,
        -1,
        -1,
        -1,
    )
    visual_input = normalizer(augment(image_rep))

    # Frozen visual computation. Read/orthographic bridge autograd remains enabled
    # below so gradients can reach the optimized lexical query.
    with torch.no_grad():
        image_info = model.encode_image_states(
            visual_input,
            return_final_tokens=False,
        )
        correction = model.read_implant.content_correction(
            image_info["states"]
        )
        content_image_raw = image_info["image_embedding"] + correction
        source_logits, glyph_logits, source_stats = model.read_implant.source_outputs(
            image_info["states"], return_details=False
        )
        glyph_probs = glyph_logits.sigmoid()
        source_probs = source_logits.sigmoid()
        source_gate_per_image = source_probs[:, 0] * source_probs[:, 1]

    # Current forward_modes deliberately moves all CLIP similarity arithmetic
    # into FP32 islands, even when the surrounding evaluation uses CUDA AMP.
    content_image = _opinion_fp32_normalize(content_image_raw)
    ordinary_text = _opinion_fp32_normalize(ordinary.text_embedding)
    read_text = _opinion_fp32_normalize(read.text_embedding)
    logit_scale = model.logit_scale.float().exp()

    # All image/candidate pairs, matching current forward_modes operation order:
    #     scale * (normalized_image @ normalized_text.T)
    # Do NOT write `scale * image @ text.T`; @ and * are same-precedence,
    # left-associative operators, so that computes `(scale * image) @ text.T`.
    content_logits_all = _opinion_fp32_scaled_matmul(
        logit_scale,
        content_image,
        ordinary_text.t(),
    )

    read_feature_all = model.read_implant.read_features(
        image_info["states"],
        read.eot_hidden_pre_ln,
        register_mask=image_info["register_mask"],
        return_details=False,
    )
    read_image_all = _opinion_fp32_normalize(read_feature_all)
    raw_read_logits_all = _opinion_fp32_scaled_einsum(
        "bnd,nd->bn",
        logit_scale,
        read_image_all,
        read_text,
    )
    read_logits_all = model.read_implant.calibrate_read_logits(
        raw_read_logits_all,
        source_logits,
        null_mask=None,
    )

    ortho_feature_all = model.read_implant.orthographic_features(
        image_info["states"],
        read.eot_hidden_pre_ln,
        return_details=False,
    )
    ortho_image_all = _opinion_fp32_normalize(ortho_feature_all)
    early_logits_all = _opinion_fp32_scaled_einsum(
        "bnd,nd->bn",
        logit_scale,
        ortho_image_all,
        read_text,
    )

    # Fixed <text> <null> baseline for every independently augmented image.
    with torch.no_grad():
        null_query = null_text_info["eot_hidden_pre_ln"]
        null_text = _opinion_fp32_normalize(
            null_text_info["text_embedding"]
        )
        null_feature = model.read_implant.read_features(
            image_info["states"],
            null_query,
            register_mask=image_info["register_mask"],
            return_details=False,
        )
        null_image = _opinion_fp32_normalize(null_feature)
        null_raw = _opinion_fp32_scaled_einsum(
            "bnd,nd->bn",
            logit_scale,
            null_image,
            null_text,
        )
        null_logits_all = model.read_implant.calibrate_read_logits(
            null_raw,
            source_logits,
            null_mask=torch.ones(1, dtype=torch.bool, device=null_raw.device),
        )

    trust_gate_all = model.read_implant.trust_gate(
        content_logits=content_logits_all,
        read_logits=read_logits_all,
        null_logits=null_logits_all[:, 0],
        early_logits=early_logits_all,
        content_image=content_image,
        read_image=read_image_all,
        content_text=ordinary_text,
        read_text=read_text,
        source_logits=source_logits,
        source_stats=source_stats,
        injection=model.read_implant.injection_feature(content_image),
    )
    effective_gate_all = (
        trust_gate_all * source_gate_per_image[:, None].detach()
    )
    relative_read_all = read_logits_all - null_logits_all
    positive_read_all = model.read_implant.positive_relative_read(relative_read_all)
    auto_contribution_all = (
        effective_gate_all
        * model.read_implant.auto_read_scale.to(positive_read_all.dtype)
        * positive_read_all
    )
    auto_logits_all = content_logits_all + auto_contribution_all

    # Verify once per loaded model that these forward VALUES still match the real
    # current PIECES implementation.  The check is no-grad and does not affect the
    # lexical optimizer.
    if iteration == 0:
        _check_current_forward_value_parity(
            visual_input=visual_input.detach(),
            model=model,
            sample=sample,
            content_logits_all=content_logits_all.detach(),
            read_logits_all=read_logits_all.detach(),
            auto_logits_all=auto_logits_all.detach(),
        )

    flat_image_idx = torch.arange(
        batch_size * repeats,
        device=content_logits_all.device,
    )
    matched_query_idx = torch.arange(
        batch_size,
        device=content_logits_all.device,
    ).repeat(repeats)

    def matched(matrix: torch.Tensor) -> torch.Tensor:
        return matrix[flat_image_idx, matched_query_idx].view(repeats, batch_size)

    content_logits = matched(content_logits_all)
    read_logits = matched(read_logits_all)
    auto_logits = matched(auto_logits_all)
    trust_gate = matched(trust_gate_all)
    route_gate = matched(effective_gate_all)
    auto_contribution = matched(auto_contribution_all)
    null_logits = null_logits_all[:, 0].view(repeats, batch_size)
    source_gate = source_gate_per_image.view(repeats, batch_size)

    # Current PIECES similarities are explicitly FP32 even under AMP. Keep this
    # defensive promotion because it is harmless and protects compatible variants.
    score_dtype = torch.promote_types(
        content_logits.dtype,
        torch.promote_types(read_logits.dtype, auto_logits.dtype),
    )
    content_logits = content_logits.to(score_dtype)
    read_logits = read_logits.to(score_dtype)
    auto_logits = auto_logits.to(score_dtype)
    null_logits = null_logits.to(score_dtype)

    mode_codes = lats.mode_codes.to(content_logits.device)
    chosen_logits = content_logits.clone()
    forced_read = mode_codes.eq(MODE_READ)
    automatic = mode_codes.eq(MODE_AUTO)
    chosen_logits[:, forced_read] = read_logits[:, forced_read]
    chosen_logits[:, automatic] = auto_logits[:, automatic]

    # Keep the original cosine-like reporting/loss scale while preserving the
    # exact calibrated/fused score arithmetic.
    scale_detached = logit_scale.detach().clamp_min(1.0e-6)
    sim_content = content_logits / scale_detached
    sim_read = read_logits / scale_detached
    sim_auto = auto_logits / scale_detached
    sim_chosen = chosen_logits / scale_detached
    null_score = null_logits / scale_detached
    auto_contribution_sim = auto_contribution / scale_detached

    tau = float(lse_tau)
    sim_lse = (
        tau * torch.logsumexp(sim_chosen / tau, dim=0)
        - tau * math.log(repeats)
    )
    loss = -100.0 * sim_lse

    gate_mean = route_gate.mean(dim=0)
    if iteration % 50 == 0:
        auto_mask = mode_codes.eq(MODE_AUTO)
        auto_text_rate = (
            gate_mean[auto_mask].ge(0.5).float().mean().item()
            if auto_mask.any()
            else float("nan")
        )
        source_mean = source_probs.mean(dim=0)
        source_gate_mean = (
            source_gate[:, auto_mask].mean().item()
            if auto_mask.any() else float("nan")
        )
        trust_mean = (
            trust_gate[:, auto_mask].mean().item()
            if auto_mask.any() else float("nan")
        )
        print(
            "sim_content="
            f"{sim_content.mean().item():.4f} "
            "sim_read="
            f"{sim_read.mean().item():.4f} "
            "sim_auto="
            f"{sim_auto.mean().item():.4f} "
            "self_text_rate="
            f"{auto_text_rate:.3f} "
            "source="
            f"{source_mean[0].item():.3f}/{source_mean[1].item():.3f} "
            "source_gate="
            f"{source_gate_mean:.3f} "
            "trust="
            f"{trust_mean:.3f} "
            "null="
            f"{null_score.mean().item():.4f}"
        )

    # Proxy embedding for optional dumps. Automatic samples use their mean
    # effective gate rather than a fabricated hard mode bit.
    embedding_gate = torch.zeros_like(gate_mean)
    embedding_gate[forced_read] = 1.0
    embedding_gate[automatic] = gate_mean[automatic]
    chosen_embedding = (
        (1.0 - embedding_gate.unsqueeze(-1)) * ordinary.text_embedding
        + embedding_gate.unsqueeze(-1) * read.text_embedding
    )

    diagnostics = OpinionDiagnostics(
        sim_content=sim_content.detach(),
        sim_read=sim_read.detach(),
        sim_auto=sim_auto.detach(),
        sim_chosen=sim_chosen.detach(),
        route_gate=route_gate.detach(),
        trust_gate=trust_gate.detach(),
        source_gate=source_gate.detach(),
        auto_contribution=auto_contribution_sim.detach(),
        null_score=null_score.detach(),
        presence_mean=source_probs.mean(dim=0).detach(),
        glyph_mean=glyph_probs.mean().detach(),
        glyph_max=glyph_probs.amax().detach(),
    )
    return loss, chosen_embedding, sample, diagnostics


# -----------------------------------------------------------------------------
# Reporting / best samples
# -----------------------------------------------------------------------------

def make_best_store() -> Dict[str, List[Tuple[float, str]]]:
    return defaultdict(list)


def update_best(
    store: Dict[str, List[Tuple[float, str]]],
    bucket: str,
    loss_value: float,
    decoded: str,
    keep: int = 8,
) -> bool:
    old_entries = list(store[bucket])
    entries = old_entries + [(float(loss_value), decoded)]
    entries.sort(key=lambda item: item[0])

    # Deduplicate by exact decoded surface string while keeping the best loss.
    deduped: List[Tuple[float, str]] = []
    seen = set()
    for item in entries:
        if item[1] in seen:
            continue
        deduped.append(item)
        seen.add(item[1])
        if len(deduped) >= keep:
            break

    changed = deduped != old_entries
    store[bucket] = deduped
    return changed


def clean_decoded(text: str) -> str:
    text = text.replace("<|startoftext|>", "")
    text = text.replace("<|endoftext|>", "")
    text = "".join(c if c.isprintable() else " " for c in text)
    return " ".join(text.split())


def decode_sample(
    tokenizer,
    token_ids: torch.Tensor,
    eot_index: int,
) -> str:
    ids = token_ids[: eot_index + 1].detach().cpu().tolist()
    return clean_decoded(tokenizer.decode(ids))


def route_label(
    base_mode: int,
    route_gate: float,
) -> str:
    if base_mode == MODE_AUTO:
        return "auto-><text>" if route_gate >= 0.5 else "auto-><notext>"
    if base_mode == MODE_READ:
        return "forced-<text>"
    return "forced-<notext>"


def token_color(base_mode: int) -> str:
    if base_mode == MODE_AUTO:
        return SELF_CHOSEN_COLOR + Style.BRIGHT
    if base_mode == MODE_READ:
        return Fore.CYAN + Style.BRIGHT
    return Fore.BLUE + Style.BRIGHT


def checkin(
    loss: torch.Tensor,
    sample: TextSample,
    diagnostics: OpinionDiagnostics,
    lats: Pars,
    tokenizer,
    bests: Dict[str, List[Tuple[float, str]]],
    image_name: str,
):
    os.makedirs("out_ga_texts", exist_ok=True)

    gate_mean = diagnostics.route_gate.mean(dim=0)
    trust_mean = diagnostics.trust_gate.mean(dim=0)
    source_gate_mean = diagnostics.source_gate.mean(dim=0)
    sim_c_mean = diagnostics.sim_content.mean(dim=0)
    sim_r_mean = diagnostics.sim_read.mean(dim=0)
    sim_a_mean = diagnostics.sim_auto.mean(dim=0)
    null_mean = diagnostics.null_score.mean(dim=0)
    contribution_mean = diagnostics.auto_contribution.mean(dim=0)

    records_by_mode: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for index in range(lats.batch_size):
        base_mode = int(lats.mode_codes[index].item())
        gate = float(gate_mean[index].item())
        display_read = base_mode == MODE_READ or (
            base_mode == MODE_AUTO and gate >= 0.5
        )
        mode = route_label(base_mode, gate)

        token_ids = (
            sample.read_token_ids[index]
            if display_read
            else sample.ordinary_token_ids[index]
        )
        eot_index = int(
            sample.read_eot_indices[index].item()
            if display_read
            else sample.ordinary_eot_indices[index].item()
        )
        decoded = decode_sample(
            tokenizer,
            token_ids,
            eot_index,
        )

        changed = update_best(
            bests,
            MODE_NAMES[base_mode],
            float(loss[index].item()),
            decoded,
        )
        record = {
            "index": index,
            "mode": mode,
            "decoded": decoded,
            "changed": changed,
            "loss": float(loss[index].item()),
            "content": float(sim_c_mean[index].item()),
            "read": float(sim_r_mean[index].item()),
            "auto": float(sim_a_mean[index].item()),
            "gate": gate,
            "trust": float(trust_mean[index].item()),
            "source_gate": float(source_gate_mean[index].item()),
            "auto_contribution": float(contribution_mean[index].item()),
            "null": float(null_mean[index].item()),
            "read_beats_null": bool(sim_r_mean[index] > null_mean[index]),
        }
        records_by_mode[base_mode].append(record)

        if changed:
            with open(
                f"out_ga_texts/tokens_{image_name}_all.txt",
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    f"[{mode}] loss={record['loss']:.6f} "
                    f"content={record['content']:.6f} "
                    f"read={record['read']:.6f} "
                    f"auto={record['auto']:.6f} "
                    f"gate={record['gate']:.6f} "
                    f"trust={record['trust']:.6f} "
                    f"source_gate={record['source_gate']:.6f} "
                    f"auto_add={record['auto_contribution']:.6f} "
                    f"null={record['null']:.6f} :: {decoded}\n"
                )

    for base_mode in (MODE_NOTEXT, MODE_READ, MODE_AUTO):
        records = records_by_mode.get(base_mode, [])
        if not records:
            continue

        print(Fore.WHITE + Style.BRIGHT + SECTION_HEADINGS[base_mode] + Fore.RESET)
        changed_records = [record for record in records if record["changed"]]
        if not changed_records:
            # CLI convenience only: keep the latest terminal check-in useful.
            # The on-disk files retain the exact existing "write only on change"
            # behavior below.
            current_bests = bests.get(MODE_NAMES[base_mode], [])
            print(Fore.WHITE + Style.DIM + "[ current saved best tokens ]" + Style.RESET_ALL)
            for rank, (best_loss, best_decoded) in enumerate(current_bests, start=1):
                print(
                    Fore.WHITE
                    + f"Best {rank:02d} loss={best_loss:.6f} Tokens:"
                    + Fore.RESET
                )
                print(token_color(base_mode) + best_decoded + Style.RESET_ALL)
        else:
            for record in changed_records:
                print(Fore.WHITE + f"Sample {record['index']} [{record['mode']}] Tokens:")
                print(token_color(base_mode) + record["decoded"] + Style.RESET_ALL)
                print(
                    Fore.CYAN
                    + f"  content={record['content']:.4f} "
                    f"read={record['read']:.4f} "
                    f"auto={record['auto']:.4f} "
                    f"gate={record['gate']:.3f} "
                    f"trust={record['trust']:.3f} "
                    f"source={record['source_gate']:.3f} "
                    f"auto_add={record['auto_contribution']:+.4f} "
                    f"null={record['null']:.4f} "
                    f"read>null={record['read_beats_null']}"
                    + Fore.RESET
                )

        if base_mode == MODE_NOTEXT:
            mean_content = sum(r["content"] for r in records) / len(records)
            print(
                Fore.BLUE + Style.BRIGHT
                + f"[ forced <notext>: {len(records)} | mean content {mean_content:.4f} ]"
                + Style.RESET_ALL
            )
        elif base_mode == MODE_READ:
            n_beats = sum(int(r["read_beats_null"]) for r in records)
            mean_read = sum(r["read"] for r in records) / len(records)
            mean_null = sum(r["null"] for r in records) / len(records)
            print(
                Fore.CYAN + Style.BRIGHT
                + f"[ forced <text>: {len(records)} | read>null {n_beats}/{len(records)} "
                f"| mean read {mean_read:.4f} | mean null {mean_null:.4f} ]"
                + Style.RESET_ALL
            )
        else:
            n_text = sum(int(r["gate"] >= 0.5) for r in records)
            n_notext = len(records) - n_text
            n_beats = sum(int(r["read_beats_null"]) for r in records)
            mean_gate = sum(r["gate"] for r in records) / len(records)
            mean_trust = sum(r["trust"] for r in records) / len(records)
            mean_source = sum(r["source_gate"] for r in records) / len(records)
            mean_add = sum(r["auto_contribution"] for r in records) / len(records)
            print(
                SELF_CHOSEN_COLOR + Style.BRIGHT
                + f"[ self selected: <text> {n_text} | <notext> {n_notext} "
                f"| mean gate {mean_gate:.3f} "
                f"| source/trust {mean_source:.3f}/{mean_trust:.3f} "
                f"| read>null {n_beats}/{len(records)} "
                f"| mean auto add {mean_add:+.4f} ]"
                + Style.RESET_ALL
            )

        print(Fore.WHITE + SECTION_END + Fore.RESET)

    print(
        Fore.WHITE
        + f"[image heads] present={diagnostics.presence_mean[0].item():.3f} "
        f"readable={diagnostics.presence_mean[1].item():.3f} "
        f"glyph_mean={diagnostics.glyph_mean.item():.4f} "
        f"glyph_max={diagnostics.glyph_max.item():.4f}"
        + Fore.RESET
    )

    # Rewrite one concise best-opinion file per static experiment group.
    for bucket, entries in bests.items():
        with open(
            f"out_ga_texts/tokens_{image_name}_{bucket}.txt",
            "w",
            encoding="utf-8",
        ) as handle:
            for loss_value, decoded in entries:
                handle.write(f"{loss_value:.6f}\t{decoded}\n")


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train_step(
    image,
    model,
    lats,
    optimizer,
    normalizer,
    augment,
    repeats,
    lse_tau,
    iteration,
    scaler,
    null_text_info,
):
    with autocast():
        loss_vector, text_embedding, sample, diagnostics = hard_piece_opinion(
            image=image,
            model=model,
            lats=lats,
            normalizer=normalizer,
            augment=augment,
            repeats=repeats,
            lse_tau=lse_tau,
            iteration=iteration,
            null_text_info=null_text_info,
        )
        loss = loss_vector.mean()

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss_vector, text_embedding, sample, diagnostics


def generate_target_text_embeddings(
    img_path,
    model,
    lats,
    optimizer,
    args,
    normalizer,
    augment,
    tokenizer,
):
    image_name = os.path.splitext(
        os.path.basename(img_path)
    )[0]
    image = load_image(
        img_path,
        model.visual.input_resolution,
        model.visual.input_resolution,
    )

    print(
        Fore.YELLOW
        + Style.BRIGHT
        + f"\nRunning AnyText null-controls gradient ascent for {image_name}...\n"
        + Fore.RESET
    )
    print(
        f"[Modes] strategy={args.mode_strategy} "
        f"codes={[MODE_NAMES[int(x)] for x in lats.mode_codes.cpu().tolist()]}"
    )

    null_tokens = clip.tokenize(["<text> <null>"], truncate=True).to(image.device)
    # forward_modes encodes its internal null while the outer inference call is
    # under CUDA AMP. Match that text-tower precision here as well.
    with torch.no_grad(), autocast():
        null_text_info = model._encode_text_hidden(null_tokens)

    scaler = GradScaler()
    best_mean_loss = float("inf")
    best_text_embeddings = None
    bests = make_best_store()

    for iteration in range(args.iterations):
        (
            loss,
            text_embedding,
            sample,
            diagnostics,
        ) = train_step(
            image=image,
            model=model,
            lats=lats,
            optimizer=optimizer,
            normalizer=normalizer,
            augment=augment,
            repeats=args.repeats,
            lse_tau=args.lse_tau,
            iteration=iteration,
            scaler=scaler,
            null_text_info=null_text_info,
        )

        current_loss = float(loss.mean().item())
        if current_loss < best_mean_loss:
            best_mean_loss = current_loss
            best_text_embeddings = copy.deepcopy(
                text_embedding.detach()
            )
            print(
                Fore.RED
                + Style.BRIGHT
                + f"New best mean loss: {best_mean_loss:.3f}"
                + Fore.RESET
            )
            checkin(
                loss,
                sample,
                diagnostics,
                lats,
                tokenizer,
                bests,
                image_name,
            )
            print(
                Fore.RED
                + Style.BRIGHT
                + "-------------------"
                + Fore.RESET
            )

        if iteration % args.checkin_step == 0:
            print(
                Fore.GREEN
                + f"Iteration {iteration}: Average Loss: {current_loss:.3f}"
                + Fore.RESET
            )
            checkin(
                loss,
                sample,
                diagnostics,
                lats,
                tokenizer,
                bests,
                image_name,
            )

    if args.dump_embeds:
        os.makedirs(
            "out_ga_txtembeds",
            exist_ok=True,
        )
        torch.save(
            {
                "chosen_text_embeddings": best_text_embeddings,
                "mode_codes": lats.mode_codes.detach().cpu(),
                "token_logits": lats.normu.detach().cpu(),
                "mode_strategy": args.mode_strategy,
                "routing": "early_source_x_candidate_trust_router",
            },
            f"out_ga_txtembeds/{image_name}_anytext_null_opinion.pt",
        )
        print(
            Fore.MAGENTA
            + Style.BRIGHT
            + "\nBest AnyText-opinion state saved to out_ga_txtembeds."
            + Fore.RESET
        )

    print(
        Fore.MAGENTA
        + Style.BRIGHT
        + "\nOpinions saved to out_ga_texts.\n"
        + Fore.RESET
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def torch_load_trusted(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


STAGE1_COMPACT_FORMATS = {
    "gmp_anytext_stage1_v1",
    "gmp_anytext_stage1_v2_null_controls",
    "gmp_anytext_early_branch_v3",
}


def is_stage1_compact(obj: Any) -> bool:
    return (
        isinstance(obj, Mapping)
        and str(obj.get("format", "")) in STAGE1_COMPACT_FORMATS
        and isinstance(obj.get("implant_state_dict"), Mapping)
    )


def _state_blocks(state: Mapping[str, Any], key: str) -> Optional[List[int]]:
    value = state.get(key)
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return None


def _configure_implant_topology_from_compact(model, compact: Mapping[str, Any]) -> None:
    if not hasattr(model, "read_implant") or model.read_implant is None:
        raise AttributeError("Base model did not instantiate read_implant.")
    state = compact["implant_state_dict"]
    late = _state_blocks(state, "tap_blocks")
    ortho = _state_blocks(state, "ortho_tap_blocks")
    source = _state_blocks(state, "source_tap_blocks")
    if late is not None:
        model.read_implant.set_tap_blocks(late, reset_uniform=False)
    if ortho is not None:
        model.read_implant.set_ortho_tap_blocks(ortho, reset_uniform=False)
    if source is not None:
        model.read_implant.set_source_tap_blocks(source, reset_uniform=False)
    print(
        "[model] compact topology: "
        f"ortho={ortho or model.read_implant.ortho_block_list()} "
        f"source={source or model.read_implant.source_block_list()} "
        f"late={late or model.read_implant.tap_block_list()}"
    )



def _validate_early_branch_api(model) -> None:
    """Fail immediately when this script is paired with a stale AnyText package."""
    implant = getattr(model, "read_implant", None)
    if implant is None:
        raise AttributeError("Loaded model has no read_implant.")
    required = (
        "source_outputs",
        "orthographic_features",
        "read_features",
        "content_correction",
        "calibrate_read_logits",
        "trust_gate",
        "positive_relative_read",
        "injection_feature",
    )
    missing = [name for name in required if not hasattr(implant, name)]
    if missing:
        raise AttributeError(
            "Loaded oaiclip/implant is not the early-branch V3 API; missing: "
            + ", ".join(missing)
        )
    architecture = str(getattr(model, "read_attention_architecture", "softmax"))
    if architecture == "sigmoid_mass":
        raise RuntimeError(
            "This opinion script intentionally does not emulate the historical "
            "sigmoid_mass evidence-scaling branch. Use the current sigmoid_all model."
        )

    print(
        f"[opinion script] revision={OPINION_SCRIPT_REVISION} "
        f"read_arch={architecture} "
        "fusion=content+source_gate*trust_gate*scale*positive(read-null) "
        "auto_search_grad=straight-through-read (forward value exact)"
    )


def load_opinion_model(model_ref: str):
    resolved = str(model_ref)
    compact: Optional[Mapping[str, Any]] = None
    if os.path.isfile(resolved):
        try:
            candidate = torch_load_trusted(resolved)
        except Exception:
            candidate = None
        if is_stage1_compact(candidate):
            compact = candidate
            resolved = str(compact["base_model_path"])
            print(f"[model] compact AnyText checkpoint: {model_ref}")
            print(f"[model] loading base model:         {resolved}")

    model, preprocess, metadata = load_openai_clip_anything(
        clip,
        resolved,
        device="cuda",
        jit=False,
        strict=True,
    )
    if compact is not None:
        _configure_implant_topology_from_compact(model, compact)
        model.read_implant.load_state_dict(compact["implant_state_dict"], strict=True)
        model.set_hard_text_token_embedding(compact["hard_text_embedding"])
        if "null_text_embedding" not in compact:
            raise KeyError("Compact checkpoint predates the required <null> embedding.")
        model.set_null_text_token_embedding(compact["null_text_embedding"])

    _validate_early_branch_api(model)

    bad_keys = [
        key for key in model.state_dict()
        if key.endswith(".theta") or key.endswith(".r")
    ]
    if bad_keys:
        raise RuntimeError(
            "Opinion model still contains geometric parametrization; export ungmp first. "
            f"Examples: {bad_keys[:8]}"
        )
    return model, preprocess, metadata



def prompt_token_ids(prompt: str, model) -> List[int]:
    tokens = clip.tokenize(prompt, truncate=True)[0].tolist()
    excluded = {
        0,
        int(getattr(model, "sot_token_id", 49406)),
        int(model.eot_token_id),
        int(model.hard_text_token_id),
        int(model.no_text_token_id),
        int(model.any_text_token_id),
        int(model.null_text_token_id),
    }
    return [int(token) for token in tokens if int(token) not in excluded]


def process_one_image(
    img_path: str,
    model,
    args,
    normalizer,
    tokenizer,
):
    augment = nn.Sequential(
        kaugs.RandomAffine(
            degrees=10,
            translate=0.1,
            p=0.8,
        ).cuda()
    ).cuda()

    lats = Pars(
        batch_size=args.batch_size,
        text_pos_shape=model.positional_embedding.shape[0],
        many_tokens=args.num_tokens,
        prompt_ids=prompt_token_ids(args.prompt, model),
        vocab_size=model.token_embedding.num_embeddings,
        sot_id=getattr(model, "sot_token_id", 49406),
        eot_id=model.eot_token_id,
        hard_text_id=model.hard_text_token_id,
        no_text_id=model.no_text_token_id,
        any_text_id=model.any_text_token_id,
        null_text_id=model.null_text_token_id,
        mode_strategy=args.mode_strategy,
        token_gumbel_temp=args.token_gumbel_temp,
    ).cuda()

    optimizer = torch.optim.Adam(
        [{"params": [lats.normu], "lr": args.learning_rate}]
    )

    generate_target_text_embeddings(
        img_path=img_path,
        model=model,
        lats=lats,
        optimizer=optimizer,
        args=args,
        normalizer=normalizer,
        augment=augment,
        tokenizer=tokenizer,
    )


def main():
    args = parse_arguments()
    if args.deterministic:
        fix_random_seed()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This script currently expects CUDA, matching the original experiment."
        )

    just_fix_windows_console()

    model, _preprocess, _ = load_opinion_model(args.use_model)
    model = model.eval().float()

    required = [
        "forward_hard",
        "encode_image_states",
        "read_implant",
        "hard_text_embedding",
        "hard_text_token_id",
        "null_text_embedding",
        "null_text_token_id",
        "no_text_token_id",
        "any_text_token_id",
        "forward_modes",
    ]
    missing = [
        name for name in required
        if not hasattr(model, name)
    ]
    if missing:
        raise AttributeError(
            "The loaded model is not the hard PIECE OF CLIP. "
            f"Missing: {missing}"
        )

    # Freeze everything in the model. Gradients still flow through frozen
    # modules into the optimized text distributions.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    normalizer = Normalization(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    ).cuda()
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    if args.img_folder != "None":
        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
        )
        image_files = [
            os.path.join(args.img_folder, filename)
            for filename in sorted(os.listdir(args.img_folder))
            if filename.lower().endswith(valid_extensions)
        ]
        for image_path in image_files:
            process_one_image(
                image_path,
                model,
                args,
                normalizer,
                tokenizer,
            )
            print(f"Done processing image: {image_path}")
    else:
        process_one_image(
            args.use_image,
            model,
            args,
            normalizer,
            tokenizer,
        )
        print(f"Done processing image: {args.use_image}")


if __name__ == "__main__":
    main()
