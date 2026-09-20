"""Declarative task catalog for ``reproduce.py``.

The probe implementations remain in ``x_paper_reproduction``.  This module only
assigns stable public task names, dependency edges, output locations, and enough
metadata for status/cached-output handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Tier = Literal["canonical", "control", "figure", "utility"]
Kind = Literal["gpu", "cpu", "audit", "data"]


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    title: str
    script: str
    command: str | None = None
    tier: Tier = "canonical"
    kind: Kind = "gpu"
    deps: tuple[str, ...] = ()
    output_rel: str | None = None
    markers: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    description: str = ""
    manual_args: bool = False


@dataclass(frozen=True)
class PaperInfo:
    """Stable paper-facing metadata used by the visual experiment catalog."""

    experiment_type: str
    intervention_family: str | None = None
    paper_sections: tuple[str, ...] = ()
    paper_artifacts: tuple[str, ...] = ()
    description: str = ""
    thumbnail: str | None = None
    special_tags: tuple[str, ...] = ()


TASKS: tuple[Task, ...] = (
    # Bridge / trained-model behavior -------------------------------------------------
    Task("bridge.cross_attention", "bridge", "SOURCE / ORTHO / READ / CONTENT bridge", "probe_cross_attention_bridge.py",
         output_rel="bridge/cross_attention", markers=("rollout_classic_b23/metrics.csv", "pairwise_deltas/map_delta_metrics.csv"), requirements=("models.final_hf", "datasets.demoset_dir")),
    Task("bridge.backbone_dynamics", "bridge", "trained-backbone / GIPU dynamics", "probe_backbone_dynamics.py",
         output_rel="bridge/backbone_dynamics", markers=("summary.txt",), requirements=("models.final_hf", "models.gmp_hf", "datasets.demoset_dir")),
    Task("bridge.read_null.hallucinations", "bridge", "READ/null hallucination controls", "probe_read_null_controls.py", "hallucinations",
         tier="control", output_rel="bridge/read_null_hallucinations", requirements=("models.final_hf", "datasets.misc_image_dir")),
    Task("bridge.read_null.tap_transplants", "bridge", "late-tap READ/null transplants", "probe_read_null_controls.py", "tap_transplants",
         tier="control", output_rel="bridge/read_null_tap_transplants", requirements=("models.final_hf", "datasets.misc_image_dir")),
    Task("bridge.read_null.diagnostic", "bridge", "raw/calibrated READ/null diagnostics", "probe_read_null_controls.py", "diagnostic",
         tier="control", output_rel="bridge/read_null_diagnostic", requirements=("models.final_hf", "datasets.misc_image_dir")),

    # Native workspace / register circuit --------------------------------------------
    Task("workspace.broadcast_sinks", "workspace", "NOP/broadcast sinks: OAI / GmP / trained-backbone comparison", "probe_broadcast_sinks.py",
         output_rel="workspace/broadcast_sinks", markers=("REPORT.md",),
         requirements=("datasets.objectnet_mvt", "models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.broadcast_channel_interventions", "workspace", "565/650/123 causal channel interventions", "probe_broadcast_channel_interventions.py",
         tier="control", output_rel="workspace/broadcast_channel_interventions",
         requirements=("datasets.special_delivery_dir", "models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.cls_register_exchange", "workspace", "CLS↔register/workspace exchange", "probe_cls_register_exchange.py",
         output_rel="workspace/cls_register_exchange", markers=("SUMMARY.txt",),
         requirements=("datasets.objectnet_mvt", "models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.cls_mu_causal", "workspace", "CLS↔mu causal 2×2 / RN repeat / slot swap", "probe_cls_mu_causal_interventions.py",
         deps=("workspace.cls_register_exchange",), output_rel="workspace/cls_mu_causal",
         markers=("SUMMARY.txt",),
         requirements=("datasets.objectnet_mvt", "models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.cls_role_surfaces", "workspace", "CLS/register role-plane surfaces", "probe_cls_role_surfaces.py",
         deps=("workspace.cls_register_exchange",), output_rel="workspace/cls_role_surfaces",
         markers=("SUMMARY.txt",),
         requirements=("models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.qk_role_gates.scan", "workspace", "late Q/K role-gate scan", "probe_qk_role_gates.py", "scan",
         tier="control", deps=("workspace.cls_register_exchange",), output_rel="workspace/qk_role_gates_scan",
         markers=("SUMMARY.txt",), requirements=("models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.qk_role_gates.same_heads", "workspace", "fixed B22 same-head Q/K control", "probe_qk_role_gates.py", "same_heads",
         tier="control", deps=("workspace.cls_register_exchange",), output_rel="workspace/qk_role_gates_same_heads",
         markers=("SUMMARY.txt",), requirements=("models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.register_geometry.secondary", "workspace", "paired secondary register/workspace geometry", "probe_register_geometry.py", "secondary",
         output_rel="workspace/register_geometry_secondary", markers=("data/register_means.npz",),
         requirements=("models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.register_geometry.grad_attention", "workspace", "late register attribution trajectory", "probe_register_geometry.py", "grad_attention",
         tier="control", output_rel="workspace/register_geometry_grad_attention",
         markers=("SUMMARY.txt",), requirements=("models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.register_geometry.principal_angles", "workspace", "B20–B22 residual-subspace principal angles", "probe_register_geometry.py", "principal_angles",
         kind="cpu", deps=("workspace.register_geometry.secondary",), output_rel="workspace/register_geometry_principal_angles",
         markers=("residual_subspace_principal_angles.csv",)),
    Task("workspace.register_cache_transport", "workspace", "B12→B13 register/cache transport vs RN", "probe_register_cache_transport.py",
         tier="control", output_rel="workspace/register_cache_transport", requirements=("models.final_hf",)),
    Task("workspace.rta_head_population", "workspace", "RTA head-population RN factorial / OV motifs", "probe_rta_head_population.py", "analyze",
         deps=("workspace.cls_register_exchange",), output_rel="workspace/rta_head_population",
         markers=("SUMMARY.txt",),
         requirements=("models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.rta_head_contact_sheets", "workspace", "cached RN motif contact sheets", "probe_rta_head_population.py", "contact_sheets",
         tier="figure", kind="cpu", deps=("workspace.rta_head_population",), output_rel="figures/rta_head_contact_sheets"),
    Task("workspace.single_image.example", "workspace", "single-image CLS/register atlas", "probe_single_image_head_motifs.py", "example",
         tier="control", deps=("workspace.cls_register_exchange",), output_rel="workspace/single_image_example",
         requirements=("datasets.demoset_dir", "assets.single_image_demo", "models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.single_image.motifs", "workspace", "single-image all-head motif catalogue", "probe_single_image_head_motifs.py", "motifs",
         tier="control", deps=("workspace.cls_register_exchange",), output_rel="workspace/single_image_motifs",
         requirements=("datasets.demoset_dir", "assets.single_image_demo", "models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),
    Task("workspace.text_cls_trajectory", "workspace", "exact TEXT→CLS OV/write trajectory", "probe_text_cls_trajectory.py", "analyze",
         output_rel="workspace/text_cls_trajectory", markers=("summary.csv",),
         requirements=("models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),

    # RN mechanism / universality -----------------------------------------------------
    Task("rn.control_mechanism", "rn", "synthetic single-block RN steering mechanism", "probe_rn_control_mechanism.py",
         output_rel="rn/control_mechanism", markers=("OVERVIEW.json",), requirements=("models.xattn_checkpoint",)),
    Task("rn.control_knob", "rn", "multilingual RN control-knob suite", "probe_tools_rn_control.py",
         tier="control", output_rel="rn/control_knob", markers=("SUMMARY.txt",),
         requirements=("models.xattn_checkpoint",)),
    Task("rn.control_manifold", "rn", "RN local control-manifold atlas", "probe_tools_rn_manifold.py",
         tier="control", output_rel="rn/control_manifold", markers=("SUMMARY.txt",),
         requirements=("models.xattn_checkpoint", "assets.rn_vocab")),
    Task("rn.control_surfaces", "rn", "native multilingual RN control surfaces", "probe_rn_control_surfaces.py", "analyze",
         output_rel="rn/control_surfaces", markers=("_register_pump_fast/SUMMARY.txt",),
         requirements=("models.xattn_checkpoint",)),
    Task("rn.control_surface_flow_maps", "rn", "cached RN surface flow maps / PLY", "probe_rn_control_surfaces.py", "flow_maps",
         tier="figure", kind="cpu", deps=("rn.control_surfaces",), output_rel="figures/rn_control_surface_flow_maps"),
    Task("rn.subspace_alignment", "rn", "RN-control PCs vs intact register subspaces", "probe_rn_subspace_alignment.py",
         tier="control", output_rel="rn/subspace_alignment", markers=("SUMMARY.txt",), requirements=("models.xattn_checkpoint",)),
    Task("rn.stash_followup", "rn", "RN stash impersonation / B11→B13 follow-up", "probe_rn_stash_followup.py",
         output_rel="rn/stash_followup", markers=("b13_stash_impersonation.csv",),
         requirements=("models.vanilla_spec", "models.xattn_checkpoint", "models.oai_rn_variant")),
    Task("rn.text_relocation", "rn", "SOURCE/PIECE text-evidence relocation after RN", "probe_rn_text_relocation.py",
         tier="control", output_rel="rn/text_relocation", markers=("rn_relocation_summary.csv",), requirements=("models.final_hf",)),
    Task("rn.bridge_transplant", "rn", "bridge/text/visual transplant universality", "probe_rn_bridge_transplant.py",
         output_rel="rn/bridge_transplant", markers=("SUMMARY.txt",),
         requirements=("models.vanilla_spec", "models.xattn_checkpoint")),
    Task("rn.touch_go_transfer", "rn", "RN transplant: persistent vs B13 touch-and-go", "probe_rn_touch_go_transfer_paper.py",
         output_rel="rn/touch_go_transfer", markers=("paper_summary.txt",),
         requirements=("models.vanilla_spec", "models.xattn_checkpoint", "models.gmp_checkpoint")),

    # Cached figures / postprocess -----------------------------------------------------
    Task("figures.role_text_atlas", "figures", "role-vs-site text atlas", "probe_role_text_atlas.py",
         tier="figure", kind="cpu", deps=("workspace.rta_head_population",), output_rel="figures/role_text_atlas",
         markers=("ROLE_SITE_TEXT_SUMMARY.txt",)),
    Task("figures.rn_mechanism", "figures", "compact RN mechanism paper figures", "probe_rn_mechanism_figures.py", "native",
         tier="figure", kind="cpu", deps=("rn.control_mechanism", "rn.stash_followup", "rn.bridge_transplant"),
         output_rel="figures/rn_mechanism", markers=("figure_manifest.json",)),

    # Conv1 / early-routing mechanistic branch ----------------------------------------
    Task("conv1.gpic_manifold", "conv1", "Conv1 semantic manifold with model-matched GPIC embedding banks",
         "probe_conv1_gpic_manifold.py", output_rel="conv1/gpic_manifold",
         markers=("batch_summary.json",), requirements=("models.conv1_gpic_models", "datasets.conv1_image_dir")),
    Task("conv1.xattn_functional_atlas", "conv1", "x-attn Conv1 functional atlas / culprit and pair discovery",
         "conv1_functional_atlas/probe_CONV1_XATTN_FUNCTIONAL_ATLAS.py", output_rel="conv1/xattn_functional_atlas",
         markers=("CULPRIT_SHORTLIST.txt", "experiment_candidates.json"),
         requirements=("models.final_hf", "models.vanilla_hf", "datasets.special_natural_dir")),
    Task("conv1.vanilla_functional_atlas", "conv1", "vanilla/GmP/bare-xattn Conv1 functional atlas",
         "conv1_functional_atlas/probe_CONV1_VANILLA_FUNCTIONAL_ATLAS.py", output_rel="conv1/vanilla_functional_atlas",
         markers=("normal/gmp/CULPRIT_SHORTLIST.txt", "normal/bare_xattn/CULPRIT_SHORTLIST.txt"),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.special_natural_dir")),
    Task("conv1.vanilla_functional_atlas_rn", "conv1", "vanilla/GmP/bare-xattn Conv1 atlas with transplanted RN",
         "conv1_functional_atlas/probe_CONV1_VANILLA_FUNCTIONAL_ATLAS.py", tier="control",
         output_rel="conv1/vanilla_functional_atlas_rn",
         markers=("rn_token/pretrained/CULPRIT_SHORTLIST.txt", "rn_token/gmp/CULPRIT_SHORTLIST.txt", "rn_token/bare_xattn/CULPRIT_SHORTLIST.txt"),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.special_natural_dir")),
    Task("conv1.residual_axis_lineage", "conv1", "Conv1 residual-axis lineage / classification",
         "residual_axis_lineage/probe_RESIDUAL_AXIS_LINEAGE.py", output_rel="conv1/residual_axis_lineage",
         markers=("lineage/RUN_CONFIG.json",),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.special_natural_dir")),
    Task("conv1.residual_axis_swap_650_565", "conv1", "650/565 residual-axis swap sweep",
         "residual_axis_lineage/probe_RESIDUAL_AXIS_LINEAGE.py", tier="control", output_rel="conv1/residual_axis_swap_650_565",
         markers=("swap_650_565_all/RUN_CONFIG.json",),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.special_natural_dir")),
    Task("conv1.residual_axis_lineage_rn", "conv1", "RN-variant residual-axis lineage / classification",
         "residual_axis_lineage/probe_RESIDUAL_AXIS_LINEAGE.py", tier="control", output_rel="conv1/residual_axis_lineage_rn",
         markers=("lineage/RUN_CONFIG.json",),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.special_natural_dir")),
    Task("conv1.mlp_neuron_discovery", "conv1", "blind MLP-neuron discovery from residual axes",
         "residual_axis_lineage/probe_MLP_NEURONS_FROM_RESIDUAL.py", output_rel="conv1/mlp_neuron_discovery",
         markers=("ALL_MODELS_top_neuron_comparison.csv",),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.special_natural_dir")),
    Task("conv1.b20_writeback_neurons", "conv1", "B20 writeback-neuron discovery",
         "residual_axis_lineage/probe_B20_WRITEBACK_NEURONS.py", output_rel="conv1/b20_writeback_neurons",
         markers=("ALL_MODELS_B20_top_neurons.csv",),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.special_natural_dir")),
    Task("conv1.b20_sharpeners_flatteners", "conv1", "B20 sharpener/flattener subfamilies",
         "b20_sharpeners_flatteners/probe_B20_SHARPENERS_FLATTENERS.py", deps=("conv1.b20_writeback_neurons",),
         output_rel="conv1/b20_sharpeners_flatteners", markers=("ALL_MODELS_B20_SIGNED_TOP.csv",),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.special_natural_dir")),
    Task("conv1.b20_pushpull_650_715", "conv1", "B20 650/715 push-pull mediation",
         "b20_pushpull_mediation/probe_B20_PUSHPULL_MEDIATION_650_715.py", deps=("conv1.b20_sharpeners_flatteners",),
         output_rel="conv1/b20_pushpull_650_715", markers=("ALL_MODELS_MEDIATION_SUMMARY.csv",),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.special_natural_dir")),
    Task("conv1.register_allocator_tomography", "conv1", "Conv1 / positional register-allocator tomography",
         "conv1_register_allocator/probe_CONV1_REGISTER_ALLOCATOR_TOMOGRAPHY.py", output_rel="conv1/register_allocator_tomography",
         markers=("REPORT.md",), requirements=("models.vanilla_hf",)),
    Task("conv1.roleplane_texture_rank.head_rank", "conv1", "all-block head true-rank / rigidity",
         "roleplane_texture_rank/probe_HEAD_TRUE_RANK_AND_RIGIDITY.py", output_rel="conv1/roleplane_texture_rank",
         markers=("head_rank_rigidity_all_blocks.csv",), requirements=("models.vanilla_hf",)),
    Task("conv1.roleplane_texture_rank.texture_inverse", "conv1", "DTD/math texture role-plane + inverse-LAST",
         "roleplane_texture_rank/probe_TEXTURE_ROLEPLANE_AND_LAST_INVERSE.py", output_rel="conv1/roleplane_texture_rank",
         markers=("REPORT_TEXTURE_ROLEPLANE.md",), requirements=("models.vanilla_hf",)),
    Task("conv1.roleplane_texture_rank.compact", "conv1", "compact role-plane / texture / rank handoff",
         "roleplane_texture_rank/make_COMPACT_ROLEPLANE_TEXTURE_RANK.py", tier="figure", kind="cpu",
         deps=("conv1.roleplane_texture_rank.head_rank", "conv1.roleplane_texture_rank.texture_inverse"),
         output_rel="conv1/roleplane_texture_rank", markers=("compact_summary_conv1_roleplane_texture_rank_compact.zip",)),
    Task("conv1.visualtextual_provenance", "conv1", "visual/textual provenance + role-plane atlas",
         "visualtextual/probe_VISUALTEXTUAL_PROVENANCE_ROLE_ATLAS.py", output_rel="conv1/visualtextual_provenance",
         markers=("REPORT.md",),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.visualtextual_dir")),
    Task("conv1.visualtextual_text_direction", "conv1", "visual/textual text-direction trajectory atlas",
         "visualtextual/probe_VISUALTEXTUAL_TEXT_DIRECTION_TRAJECTORY.py", output_rel="conv1/visualtextual_text_direction",
         markers=("REPORT.md",),
         requirements=("models.final_hf", "models.gmp_hf", "models.vanilla_hf", "datasets.visualtextual_dir")),

    # Utilities / audit ---------------------------------------------------------------
    Task("audit.read_probe_router", "audit", "static READ-probe/router checkpoint audit", "audit_final_read_probe_router.py",
         tier="utility", kind="audit", output_rel="audit/read_probe_router", manual_args=True),
)

PAPER_INFO: dict[str, PaperInfo] = {
    'bridge.cross_attention': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:bridge_architecture', 'sec:results_routing'), paper_artifacts=('fig:bridge_detailed', 'fig:bridge_candidate_example'),
        description='SOURCE/ORTHO/READ/CONTENT maps, contributions and rollouts.', thumbnail='bridge_detailed.png',
    ),
    'bridge.backbone_dynamics': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:mech_native_exchange', 'app:early_register_formation'), paper_artifacts=(),
        description='Non-interventional backbone/register/workspace dynamics underlying the bridge story.', thumbnail='01_native_exchange_by_block.png',
    ),
    'bridge.read_null.hallucinations': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:results_human_audit', 'sec:bridge_architecture'), paper_artifacts=('fig:grounded_reader_and_null (support)',),
        description='No-text/unsupported-query hallucination and READ/NULL behavior.', thumbnail='grounded_null_vs_source.png',
    ),
    'bridge.read_null.tap_transplants': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:bridge_architecture', 'sec:results_routing'), paper_artifacts=(),
        description='Frozen late-tap transplants test what READ/NULL depends on.', thumbnail='grounded_reader_discrimination.png',
    ),
    'bridge.read_null.diagnostic': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:results_human_audit', 'sec:bridge_architecture'), paper_artifacts=('fig:grounded_reader_and_null (support)',),
        description='Raw/calibrated READ, NULL and RN-attention diagnostics.', thumbnail='grounded_reader_discrimination.png',
    ),
    'workspace.broadcast_sinks': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:mech_native_exchange', 'app:early_register_formation'), paper_artifacts=(),
        description='NOP/broadcast-sink census and register-overlap measurements.', thumbnail='01_native_exchange_by_block.png',
    ),
    'workspace.broadcast_channel_interventions': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:mech_native_exchange', 'sec:conv1_residual_takeover'), paper_artifacts=(),
        description='Causal 565/650/123 residual-channel interventions on native broadcast/readout.', thumbnail='axis650_write_rms.png',
    ),
    'workspace.cls_register_exchange': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:mech_native_exchange', 'app:early_register_formation'), paper_artifacts=('fig:app_early_mu1_sources',),
        description='Exact CLS↔register/workspace exchange and early μ1 source attribution.', thumbnail='01_native_exchange_by_block.png',
    ),
    'workspace.cls_mu_causal': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:mech_native_exchange', 'app:early_register_formation', 'app:rn_routing_controls'), paper_artifacts=('fig:app_preb13_register_norm', 'fig:app_mlp_precursor', 'fig:app_b13_h5_substitution', 'fig:app_slot_swap_semantics'),
        description='Phase-separated CLS read/broadcast lesions, RN repeat and slot-swap causal controls.', thumbnail='02_preB13_register_lineage_norm_2x2.png',
    ),
    'workspace.cls_role_surfaces': PaperInfo(
        experiment_type='I', intervention_family='mu2',
        paper_sections=('sec:mech_role_plane', 'app:role_plane_controls'), paper_artifacts=('fig:natural_role_plane', 'fig:app_rn_role_plane_surfaces'),
        description='Fixed μ1/μ2 role-plane surfaces and controlled CLS/RN displacement.', thumbnail='natural_role_plane.png', special_tags=('PLY',),
    ),
    'workspace.qk_role_gates.scan': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('app:b22_qk_controls',), paper_artifacts=(),
        description='Late Q/K gate-gradient scan projected into the role plane.', thumbnail='forced_b22_samehead_comparison.png',
    ),
    'workspace.qk_role_gates.same_heads': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('app:b22_qk_controls',), paper_artifacts=('fig:app_b22_samehead',),
        description='Forced same-head B22 Q/K comparison across backbones.', thumbnail='forced_b22_samehead_comparison.png',
    ),
    'workspace.register_geometry.secondary': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:mech_training_reorganizes', 'app:late_workspace_geometry'), paper_artifacts=('fig:secondary_mode_crossfit_main',),
        description='Cross-fitted secondary register/workspace occupancy geometry.', thumbnail='secondary_mode_crossfit.png',
    ),
    'workspace.register_geometry.grad_attention': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:mech_training_reorganizes',), paper_artifacts=('fig:late_causal_trajectory',),
        description='Grad×attention late attribution trajectory; diagnostic rather than ablation.', thumbnail='06_paper_trajectory_summary.png',
    ),
    'workspace.register_geometry.principal_angles': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('app:late_workspace_geometry',), paper_artifacts=('fig:app_residual_principal_angles', 'fig:app_residual_temporal_alignment', 'fig:app_attack_factorization'),
        description='Post-PC1 B20–B22 residual-subspace alignment and factorization.', thumbnail='residual_subspace_principal_angles_b20_b22.png',
    ),
    'workspace.register_cache_transport': PaperInfo(
        experiment_type='M/I', intervention_family=None,
        paper_sections=('sec:mech_rn_address_payload', 'app:rn_routing_controls'), paper_artifacts=(),
        description='B12 register-tail → B13 cache transport measured with RN off/on source competition.', thumbnail='fig1_b13_address_and_payload.png',
    ),
    'workspace.rta_head_population': PaperInfo(
        experiment_type='I', intervention_family='mu2',
        paper_sections=('sec:mech_role_plane', 'app:role_plane_controls'), paper_artifacts=('fig:app_mu2_full_clean', 'fig:app_mu2_rta_rn_full', 'fig:app_mu2_synth_rn_full'),
        description='Dataset-scale all-head μ2 slider plus RN factorial and OV/write motifs.', thumbnail='B13_NoRTA_vs_RTA_vs_RTA_RN.png',
    ),
    'workspace.rta_head_contact_sheets': PaperInfo(
        experiment_type='M', intervention_family='mu2',
        paper_sections=('app:role_plane_controls',), paper_artifacts=('fig:app_changed_motif_rta', 'fig:app_changed_motif_synth'),
        description='Figure/postprocess task for μ2/RN head-motif changes.', thumbnail='changed_primary_motif_heads_RTA.png',
    ),
    'workspace.single_image.example': PaperInfo(
        experiment_type='M/I', intervention_family='mu2',
        paper_sections=('sec:mech_native_exchange', 'sec:mech_role_plane'), paper_artifacts=('fig:native_exchange_by_block',),
        description='Single-image native exchange atlas plus blockwise μ2 steering sweeps.', thumbnail='01_native_exchange_by_block.png',
    ),
    'workspace.single_image.motifs': PaperInfo(
        experiment_type='I', intervention_family='mu2',
        paper_sections=('sec:mech_role_plane', 'app:role_plane_controls'), paper_artifacts=('fig:mu2_b13_families', 'tab:mu2_sign_grammar (support)'),
        description='All-block/all-head μ2-slider response motif catalogue.', thumbnail='mu2_slider_response_b13.png',
    ),
    'workspace.text_cls_trajectory': PaperInfo(
        experiment_type='M/I', intervention_family=None,
        paper_sections=('sec:mech_native_exchange',), paper_artifacts=('fig:text_cls_write', 'fig:rn_text_cls_effect'),
        description='Exact TEXT→CLS OV/write trajectory across models and RN states.', thumbnail='text_to_cls_exact_ov_write_trajectory.png',
    ),
    'rn.control_mechanism': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:mech_rn_address_payload', 'sec:mech_rn_lowrank'), paper_artifacts=('fig:rn_address_payload', 'fig:rn_lowrank_pulse (via figures.rn_mechanism)'),
        description='Single-block RN source substitution, active-write and low-rank sufficiency/necessity mechanism.', thumbnail='fig1_b13_address_and_payload.png',
    ),
    'rn.control_knob': PaperInfo(
        experiment_type='I', intervention_family='rn_surface',
        paper_sections=('app:rn_lowrank_controls',), paper_artifacts=('fig:app_rn_control_knobs',),
        description='Continuous rank-1/rank-4 RN control-coordinate manipulations.', thumbnail='read_null_control_state_rank1_rank4.png',
    ),
    'rn.control_manifold': PaperInfo(
        experiment_type='I', intervention_family='rn_surface',
        paper_sections=('app:rn_lowrank_controls',), paper_artifacts=(),
        description='Local multidimensional RN control-manifold chart and reachable-state geometry.', thumbnail='read_null_control_state_trajectory.png', special_tags=('PLY',),
    ),
    'rn.control_surfaces': PaperInfo(
        experiment_type='I', intervention_family='rn_surface',
        paper_sections=('app:rn_lowrank_controls', 'app:multilingual_rn_surface'), paper_artifacts=('fig:app_rn_empirical_support', 'fig:app_multilingual_read_flow (support)'),
        description='Native multilingual RN causal surfaces, empirical-support restriction and register-pump controls.', thumbnail='empirical_support_ridge.png', special_tags=('PLY',),
    ),
    'rn.control_surface_flow_maps': PaperInfo(
        experiment_type='M', intervention_family='rn_surface',
        paper_sections=('app:multilingual_rn_surface',), paper_artifacts=('fig:app_multilingual_read_flow',),
        description='Cached surface gradients/flow maps and PLY export.', thumbnail='read_flow_gradinet.png', special_tags=('PLY',),
    ),
    'rn.subspace_alignment': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('app:rn_lowrank_controls',), paper_artifacts=('fig:app_rn_register_overlap',),
        description='Alignment between RN-control PCs and intact register invariant subspaces.', thumbnail='rn_pc_register_subspace_alignment.png',
    ),
    'rn.stash_followup': PaperInfo(
        experiment_type='M/I', intervention_family=None,
        paper_sections=('sec:mech_rn_address_payload', 'app:rn_routing_controls'), paper_artifacts=('fig:rn_address_payload (support)',),
        description='RN stash impersonation, native-source substitution and K/V addressing/payload follow-up.', thumbnail='fig1_b13_address_and_payload.png',
    ),
    'rn.text_relocation': PaperInfo(
        experiment_type='M/I', intervention_family=None,
        paper_sections=('sec:mech_native_exchange', 'sec:mech_training_reorganizes'), paper_artifacts=(),
        description='Where source-PIECE text evidence moves after RN insertion.', thumbnail='paper_rn_effect__all_backbones.png',
    ),
    'rn.bridge_transplant': PaperInfo(
        experiment_type='I', intervention_family='rn_surface',
        paper_sections=('sec:mech_rn_transfer', 'sec:mech_rn_address_payload', 'app:rn_lowrank_controls'), paper_artifacts=(),
        description='Visual/text/bridge transplants, pump ablations and same-coordinate RN control surfaces.', thumbnail='fig3_shared_geometry_language_readout.png', special_tags=('PLY',),
    ),
    'rn.touch_go_transfer': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:mech_rn_touch_go',), paper_artifacts=('fig:rn_touch_go_trajectory',),
        description='Persistent RN versus B13-only touch-and-go transplantation.', thumbnail='rn_touch_go_trajectory.png',
    ),
    'figures.role_text_atlas': PaperInfo(
        experiment_type='M', intervention_family='mu2',
        paper_sections=('sec:mech_role_plane', 'app:role_plane_controls'), paper_artifacts=('tab:mu2_sign_grammar', 'changed-motif sheets (support)'),
        description='CPU postprocess tying μ2 role grammar to text-site traffic.', thumbnail='changed_primary_motif_heads_RTA.png',
    ),
    'figures.rn_mechanism': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:mech_rn_address_payload', 'sec:mech_rn_lowrank'), paper_artifacts=('fig:rn_address_payload', 'fig:rn_lowrank_pulse'),
        description='Cached paper-figure compositor for RN address/payload and low-rank pulse.', thumbnail='fig1_b13_address_and_payload.png',
    ),
    'conv1.gpic_manifold': PaperInfo(
        experiment_type='I', intervention_family='gpic',
        paper_sections=('sec:conv1_manifold_surfing',), paper_artifacts=('fig:AA_GLOBAL_bullet5__ch0720_FLIP__ch0866_FLIP', 'fig:AA_LOCAL_bullet5__ch0720_FLIP__ch0499_FLIP', 'fig:AA_typo_exam__ch0720_SHUFFLE__ch0866_FLIP'),
        description="Conv1 perturbation 'manifold surfing' with model-matched GPIC nearest-neighbor retrieval.", thumbnail='AA_GLOBAL_bullet5__ch0720_FLIP__ch0866_FLIP.png',
    ),
    'conv1.xattn_functional_atlas': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:conv1_causal_atlas', 'app:conv1_excursion_controls'), paper_artifacts=('tab:conv1_matched_severe', 'fig:perturb-conv1-examples', 'fig:app_conv1_severe_counts (support)'),
        description='Seven-condition Conv1 causal atlas, culprit discovery and x-attn displacement manifolds.', thumbnail='perturb-conv1-examples.png',
    ),
    'conv1.vanilla_functional_atlas': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:conv1_causal_atlas', 'app:conv1_excursion_controls'), paper_artifacts=('fig:conv1_model_attenuation', 'tab:conv1_matched_severe', 'fig:app_conv1_severe_counts'),
        description='Matched Conv1 perturbation atlas for pretrained/GmP/bare-xattn backbones.', thumbnail='conv1_final_distance_models.png',
    ),
    'conv1.vanilla_functional_atlas_rn': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:conv1_causal_atlas', 'app:conv1_excursion_controls'), paper_artifacts=('tab:app_conv1_rn_portability',),
        description='Same fixed Conv1 events with transplanted RN across vanilla-family backbones.', thumbnail='conv1_severe_event_counts.png',
    ),
    'conv1.residual_axis_lineage': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:conv1_residual_takeover', 'app:conv1_excursion_controls'), paper_artifacts=('fig:conv1_axis_takeover', 'fig:app_conv1_650_write'),
        description='Depthwise ancestry/takeover of residual coordinates 650, 565 and controls.', thumbnail='axis650_register_enrichment.png',
    ),
    'conv1.residual_axis_swap_650_565': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:conv1_residual_takeover',), paper_artifacts=('fig:conv1_axis_takeover (swap panel)',),
        description='Block-local and persistent 650↔565 coordinate-swap interventions.', thumbnail='swap_650_565_localization.png',
    ),
    'conv1.residual_axis_lineage_rn': PaperInfo(
        experiment_type='M/I', intervention_family=None,
        paper_sections=('sec:conv1_residual_takeover',), paper_artifacts=(),
        description='Residual-axis lineage repeated with RN variants to locate RN relative to B12 construction.', thumbnail='axis650_register_enrichment.png',
    ),
    'conv1.mlp_neuron_discovery': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:conv1_residual_takeover', 'app:conv1_excursion_controls'), paper_artifacts=('fig:conv1_blind_register_neurons', 'tab:app_conv1_reg_neurons'),
        description='Blind residual-side recovery of B11/B12 register-neuron populations.', thumbnail='blind_register_neuron_recovery.png',
    ),
    'conv1.b20_writeback_neurons': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:conv1_b20_writeback', 'app:conv1_excursion_controls'), paper_artifacts=('fig:b20_register_neuron_3357 (support)',),
        description='Blind discovery of sparse B20 register/workspace writeback neurons.', thumbnail='b20_register_neuron_3357.png',
    ),
    'conv1.b20_sharpeners_flatteners': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:conv1_b20_writeback', 'app:conv1_excursion_controls'), paper_artifacts=('fig:app_b20_b21_program', 'tab:app_b20_signed_ablation'),
        description='Signed B20 sharpener/flattener neuron-family ablations and B21 routing effects.', thumbnail='b20_b21_head_program.png',
    ),
    'conv1.b20_pushpull_650_715': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:conv1_b20_writeback', 'app:conv1_excursion_controls'), paper_artifacts=('fig:app_b20_coordinate_mediation',),
        description='Coordinate add-back/removal tests for whether 650/715 mediate the B20 actuator packet.', thumbnail='b20_coordinate_mediation.png',
    ),
    'conv1.register_allocator_tomography': PaperInfo(
        experiment_type='I', intervention_family=None,
        paper_sections=('sec:conv1_causal_atlas', 'sec:conv1_role_allocation'), paper_artifacts=(),
        description='Causal Conv1/positional → early-scanner → register-allocation tomography.', thumbnail='stimulus-dependent-allocation.png',
    ),
    'conv1.roleplane_texture_rank.head_rank': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:conv1_role_allocation', 'app:role_allocation_controls'), paper_artifacts=('fig:app_rank_role_emergence',),
        description='All-block QK true-rank/rigidity and visual-tracker phenotype census.', thumbnail='true_rank_vs_role_emergence.png',
    ),
    'conv1.roleplane_texture_rank.texture_inverse': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:conv1_role_allocation', 'app:role_allocation_controls'), paper_artifacts=('fig:app_register_allocation_texture',),
        description='DTD/synthetic register allocation, role-plane occupancy and inverse-LAST spectral analysis.', thumbnail='dtd_reg_hiddenmu_by_class.png',
    ),
    'conv1.roleplane_texture_rank.compact': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:conv1_role_allocation', 'app:role_allocation_controls'), paper_artifacts=('fig:app_rank_role_emergence', 'fig:app_register_allocation_texture'),
        description='CPU postprocess/compact summary of rank, texture and role-allocation results.', thumbnail='true_rank_vs_role_emergence.png',
    ),
    'conv1.visualtextual_provenance': PaperInfo(
        experiment_type='M/I', intervention_family=None,
        paper_sections=('sec:text_provenance_geometry', 'app:text_provenance_controls'), paper_artifacts=('fig:stimulus-dependent-allocation', 'fig:app_b20_background_provenance', 'fig:app_role_text_direction', 'fig:app_conv1_text_frequency'),
        description='Visual/text/mix provenance atlas, role-plane removal and causal Conv1 text-channel controls.', thumbnail='stimulus-dependent-allocation.png',
    ),
    'conv1.visualtextual_text_direction': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:text_provenance_geometry', 'app:text_provenance_controls'), paper_artifacts=('fig:text_provenance_depth', 'fig:app_final_text_angle_strip', 'fig:app_synthetic_text_transfer'),
        description='Layerwise transferable text-direction trajectory and synthetic generalization.', thumbnail='text_provenance_crosscontrast_angle.png',
    ),
    'audit.read_probe_router': PaperInfo(
        experiment_type='M', intervention_family=None,
        paper_sections=('sec:bridge_architecture', 'methods trust-router equations'), paper_artifacts=(),
        description='Static checkpoint/config audit for optional READ probe and trust-router input width.', thumbnail=None,
    ),
}


TASK_BY_ID = {task.id: task for task in TASKS}
FAMILIES = tuple(dict.fromkeys(task.family for task in TASKS))


def paper_info_for(task_id: str) -> PaperInfo:
    return PAPER_INFO[task_id]


def tasks_for_family(family: str, *, include_controls: bool = True) -> list[Task]:
    out = [t for t in TASKS if t.family == family]
    if not include_controls:
        out = [t for t in out if t.tier in {"canonical", "figure"}]
    return out
