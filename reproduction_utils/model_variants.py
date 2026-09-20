"""Scientific model-variant registry for paper reproduction.

A variant name describes the *runtime scientific object*, not merely the source
checkpoint name.  Training provenance (for example GmP) and runtime architecture
(for example vanilla CLIP vs full x-attn) are kept separate deliberately.

Only variants that are useful across tasks are persisted under ``_models``.
Experiment-local transplants remain ephemeral but are still declared here so the
front-end can print/audit what each task is supposed to instantiate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Persistence = Literal["source", "cached", "ephemeral"]
RuntimeFamily = Literal["vanilla", "rn_only", "xattn_full", "mixed_runtime"]
BridgePolicy = Literal["absent", "trained_donor", "bypassed", "not_applicable"]
RNPolicy = Literal["absent", "trained_donor", "manual_intervention", "not_applicable"]


@dataclass(frozen=True)
class ModelVariant:
    id: str
    title: str
    persistence: Persistence
    runtime_family: RuntimeFamily
    bridge_policy: BridgePolicy
    rn_policy: RNPolicy
    cache_filename: str | None = None
    description: str = ""


VARIANTS: dict[str, ModelVariant] = {
    "oai_vanilla": ModelVariant(
        "oai_vanilla", "OpenAI vanilla CLIP", "source", "vanilla", "absent", "absent",
        description="Canonical OpenAI CLIP weights in the architecture required by the probe.",
    ),
    "gmp_trained_vanilla": ModelVariant(
        "gmp_trained_vanilla", "GmP-trained ordinary CLIP", "cached", "vanilla", "absent", "absent",
        cache_filename="gmp_trained_vanilla.pt",
        description="Weights learned under GmP training, materialized into ordinary CLIP weights.",
    ),
    "xattn_full_trained": ModelVariant(
        "xattn_full_trained", "full trained x-attn/RN model", "cached", "xattn_full", "trained_donor", "trained_donor",
        cache_filename="xattn_full_trained.pt",
        description="Canonical released x-attn model including trained bridge, router, CONTENT and RN state.",
    ),
    "oai_vanilla_rn_from_xattn": ModelVariant(
        "oai_vanilla_rn_from_xattn", "OpenAI vanilla + transplanted trained RN", "cached", "rn_only", "absent", "trained_donor",
        cache_filename="oai_vanilla_rn_from_xattn.pt",
        description="Vanilla OpenAI CLIP plus only the trained RN token/config from x-attn; no bridge exists.",
    ),
    "xattn_visual_on_oai_text": ModelVariant(
        "xattn_visual_on_oai_text", "x-attn-trained visual tower on OAI text tower", "ephemeral", "vanilla", "absent", "absent",
        description="Ordinary x-attn visual weights transplanted into capture-enabled vanilla CLIP; custom x-attn state excluded.",
    ),
    "xattn_visual_stripped": ModelVariant(
        "xattn_visual_stripped", "x-attn-trained visual tower stripped to vanilla runtime", "ephemeral", "vanilla", "absent", "absent",
        description="Ordinary x-attn visual tensors are read directly from the donor state dict and copied into a vanilla visual shell; no x-attn/RN/bridge runtime is instantiated. Text tower is not used by this visual-state comparison.",
    ),
    "xattn_backbone_no_bridge": ModelVariant(
        "xattn_backbone_no_bridge", "jointly-trained x-attn backbone without bridge/RN", "ephemeral", "vanilla", "absent", "absent",
        description="Ordinary trained backbone reconstructed without custom bridge/control-token state.",
    ),
    "xattn_rn_classic": ModelVariant(
        "xattn_rn_classic", "jointly-trained backbone + trained RN, bridge bypassed", "ephemeral", "xattn_full", "bypassed", "trained_donor",
        description="Full trained model used only for the classic visual/RN computation; bridge outputs are not executed.",
    ),
    "rn_manual_from_xattn": ModelVariant(
        "rn_manual_from_xattn", "manual trained-RN intervention", "ephemeral", "mixed_runtime", "absent", "manual_intervention",
        description="Exact trained RN vector is inserted manually into an otherwise ordinary visual tower; no bridge is instantiated.",
    ),
    "bridge_on_oai_vit_oai_text": ModelVariant(
        "bridge_on_oai_vit_oai_text", "trained bridge/RN on OAI visual + OAI text", "ephemeral", "xattn_full", "trained_donor", "trained_donor",
        description="Starts from the trained x-attn donor; ordinary visual and text weights are replaced while trained custom state is preserved exactly.",
    ),
    "bridge_on_oai_vit_trained_text": ModelVariant(
        "bridge_on_oai_vit_trained_text", "trained bridge/RN on OAI visual + trained text", "ephemeral", "xattn_full", "trained_donor", "trained_donor",
        description="Starts from trained x-attn; only ordinary visual weights are replaced.",
    ),
    "bridge_on_trained_vit_oai_text": ModelVariant(
        "bridge_on_trained_vit_oai_text", "trained bridge/RN on trained visual + OAI text", "ephemeral", "xattn_full", "trained_donor", "trained_donor",
        description="Starts from trained x-attn; only ordinary text weights are replaced.",
    ),
    "rn_touch_go_oai": ModelVariant(
        "rn_touch_go_oai", "trained RN transplanted into OAI receiver", "ephemeral", "mixed_runtime", "absent", "manual_intervention",
        description="Ordinary OAI receiver; exact donor RN is inserted by the experiment's native visual forward.",
    ),
    "rn_touch_go_gmp": ModelVariant(
        "rn_touch_go_gmp", "trained RN transplanted into GmP-trained receiver", "ephemeral", "mixed_runtime", "absent", "manual_intervention",
        description="Ordinary GmP-trained receiver; exact donor RN is inserted by the experiment's native visual forward.",
    ),
}


def get_variant(variant_id: str) -> ModelVariant:
    try:
        return VARIANTS[variant_id]
    except KeyError as exc:
        raise KeyError(f"Unknown reproduction model variant {variant_id!r}") from exc

# Task -> scientific runtime variants.  Figure/postprocess tasks intentionally
# have no model variants because they consume cached tables only.
TASK_MODEL_VARIANTS: dict[str, tuple[str, ...]] = {
    "bridge.cross_attention": ("xattn_full_trained",),
    "bridge.backbone_dynamics": ("gmp_trained_vanilla", "xattn_backbone_no_bridge", "xattn_rn_classic"),
    "bridge.read_null.hallucinations": ("xattn_full_trained",),
    "bridge.read_null.tap_transplants": ("xattn_full_trained",),
    "bridge.read_null.diagnostic": ("xattn_full_trained",),

    "workspace.broadcast_sinks": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text"),
    "workspace.broadcast_channel_interventions": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text"),
    "workspace.cls_register_exchange": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text"),
    "workspace.cls_mu_causal": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text", "rn_manual_from_xattn"),
    "workspace.cls_role_surfaces": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text", "rn_manual_from_xattn"),
    "workspace.qk_role_gates.scan": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text"),
    "workspace.qk_role_gates.same_heads": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text"),
    "workspace.register_geometry.secondary": ("gmp_trained_vanilla", "xattn_visual_stripped"),
    "workspace.register_geometry.grad_attention": ("gmp_trained_vanilla", "xattn_visual_stripped"),
    "workspace.register_cache_transport": ("xattn_full_trained",),
    "workspace.rta_head_population": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text", "rn_manual_from_xattn"),
    "workspace.single_image.example": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text", "rn_manual_from_xattn"),
    "workspace.single_image.motifs": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text", "rn_manual_from_xattn"),
    "workspace.text_cls_trajectory": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_on_oai_text", "rn_manual_from_xattn"),

    "rn.control_mechanism": ("xattn_full_trained",),
    "rn.control_knob": ("xattn_full_trained",),
    "rn.control_manifold": ("xattn_full_trained",),
    "rn.control_surfaces": ("xattn_full_trained",),
    "rn.subspace_alignment": ("xattn_full_trained",),
    "rn.stash_followup": ("xattn_full_trained", "oai_vanilla_rn_from_xattn"),
    "rn.text_relocation": ("xattn_full_trained",),
    "rn.bridge_transplant": (
        "xattn_full_trained",
        "bridge_on_oai_vit_oai_text",
        "bridge_on_oai_vit_trained_text",
        "bridge_on_trained_vit_oai_text",
    ),
    "rn.touch_go_transfer": ("xattn_full_trained", "rn_touch_go_oai", "rn_touch_go_gmp"),

    # Conv1 / early-routing branch. These tasks now resolve all public comparison
    # checkpoints from HF; the variant labels describe the instantiated runtime.
    "conv1.gpic_manifold": ("xattn_full_trained", "oai_vanilla"),
    "conv1.xattn_functional_atlas": ("xattn_full_trained", "oai_vanilla"),
    "conv1.vanilla_functional_atlas": ("gmp_trained_vanilla", "xattn_visual_stripped"),
    "conv1.vanilla_functional_atlas_rn": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_stripped", "rn_manual_from_xattn"),
    "conv1.residual_axis_lineage": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_stripped", "xattn_full_trained"),
    "conv1.residual_axis_swap_650_565": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_stripped", "xattn_full_trained"),
    "conv1.residual_axis_lineage_rn": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_stripped", "xattn_full_trained", "rn_manual_from_xattn"),
    "conv1.mlp_neuron_discovery": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_stripped", "xattn_full_trained"),
    "conv1.b20_writeback_neurons": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_stripped", "xattn_full_trained"),
    "conv1.b20_sharpeners_flatteners": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_stripped", "xattn_full_trained"),
    "conv1.b20_pushpull_650_715": ("oai_vanilla", "gmp_trained_vanilla", "xattn_visual_stripped", "xattn_full_trained"),
    "conv1.register_allocator_tomography": ("oai_vanilla",),
    "conv1.roleplane_texture_rank.head_rank": ("oai_vanilla",),
    "conv1.roleplane_texture_rank.texture_inverse": ("oai_vanilla",),
    "conv1.visualtextual_provenance": ("oai_vanilla", "gmp_trained_vanilla", "xattn_full_trained"),
    "conv1.visualtextual_text_direction": ("oai_vanilla", "gmp_trained_vanilla", "xattn_full_trained"),
}


def task_variant_ids(task_id: str) -> tuple[str, ...]:
    return TASK_MODEL_VARIANTS.get(task_id, ())
