> Start with `python reproduce.py setup`, then open
> [`../reproduce_info/configurator.html`](../reproduce_info/configurator.html) locally to choose paper experiments.
> This document is the advanced implementation/reference guide.

> To use GPIC [gated], you'll need to accept the terms:\
> [stanford-vision-lab/gpic](https://huggingface.co/datasets/stanford-vision-lab/gpic) 🤗\
> ...and be logged in via HuggingFace CLI.

# Paper reproduction

This directory contains the scientific probe implementations used for the paper.
For normal reproduction, **do not start by calling these files directly**. Use the
repository-level front end:

```bash
python reproduce.py setup
python reproduce.py status
python reproduce.py list
python reproduce.py run bridge.cross_attention
python reproduce.py run workspace.cls_mu_causal --with-deps
python reproduce.py run all
```

`reproduce.py` keeps the existing probe code intact and supplies the missing
reproducibility layer around it: stable task names, dependency ordering, one output
root, cached-result detection, shared checkpoint/data configuration, and logical
references to outputs produced by prerequisite tasks.

Model implementations are kept separate by scientific role:

- ordinary vanilla OpenAI CLIP construction: `main_hard_text_gate/legacy_hard_text_pre/oaicliporg`;
- vanilla OpenAI CLIP when explicit Q/K/V or token-state capture is required: `attnclip_mechinterp_sae`;
- x-attn/RN CLIP instrumentation: `attnclip_mechinterp_xattn`;
- concat-attention standard CLIP construction: repository-root `oaiclip`.

GmP is **not** used as a synonym for vanilla CLIP. It is loaded only by experiments
whose stated model comparison explicitly includes GmP (or by training code outside
this reproduction front end). For capture-heavy vanilla probes, stock `openai/clip`
is deliberately not a fallback because it lacks the required Q/K/V instrumentation.

## Task classes

- **canonical** — paper-facing numerical/mechanistic reproduction.
- **control** — useful causal/control analyses that should not be presented as the
  first thing a reader has to run.
- **figure** — cached post-processing or rendering; no expensive extraction when its
  prerequisites already exist.
- **utility** — provenance/data/audit tooling, not a paper result by itself.

`paper` selects canonical + figure tasks. `all` selects every automatically runnable
analysis/control/figure task (manual audit utilities stay explicit). `python reproduce.py list`
now shows all automatic tasks plus the complete legal selector/task-ID list. `--all` remains
as a convenience for including controls when selecting `paper` or a family.

This first pass deliberately does **not** invent exact Figure/Table/Section numbers or
wall-clock estimates. The supplied repository snapshot does not contain the paper
source or recorded per-probe timings. The catalog therefore records the scientific
purpose, dependency graph, and whether a task is GPU extraction, CPU postprocess,
data utility, or audit. Exact paper anchors and measured runtimes can be added to the
registry once those two sources of truth are available.

## Configuration

`reproduction_config.json` is independent of `training_config.json` and
`benchmark_config.json`. The important shared settings are:

- `models.final_hf`: final released x-attention model (or a local HF directory).
  The default is `zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX`; there is
  deliberately no second HF x-attn setting for the same model.
- `models.vanilla_spec`: canonical ordinary OpenAI CLIP source. The default is
  `ViT-L/14`.
- `models.gmp_hf`: GmP **comparison** source, used only by tasks that explicitly
  compare against GmP. The default is `zer0int/CLIP-GmP-ViT-L-14`.
- `models.xattn_checkpoint`: optional local native/OpenAI-format x-attn checkpoint
  override. When unset, low-level probes reconstruct the same `models.final_hf`
  release once as `<output_root>/_models/xattn_full_trained.pt` and
  reuse it. Thus the final HF x-attn model remains the single source of truth.
- `models.gmp_checkpoint`: optional checkpoint for the **GmP-trained weight set**
  used by experiments that explicitly compare it with other learned weights. When
  unset, the front end converts `models.gmp_hf` with the same generic HF→OpenAI
  state-dict converter used for ordinary CLIP and caches it once as
  `<output_root>/_models/gmp_trained_vanilla.pt`. The string `GmP` is training
  provenance only; it never selects a GmP runtime/model implementation. If a legacy
  checkpoint actually contains `theta`/`r` GeometricLinear pairs, those tensors are
  detected from the state dict itself and converted to the exact effective ordinary
  weights (`r * normalize(theta)`). Older reproduction caches with those tensors are
  upgraded automatically. Auto-generated model caches are also validated by runtime
  architecture before reuse: the GmP-trained cache must be ordinary vanilla CLIP,
  while the final x-attn cache must contain the x-attn/RN runtime tensors. Each cache
  gets a neighboring `.meta.json` recording its logical role, source spec, expected
  runtime family, and SHA-256 of the cached artifact. A stale cache with the wrong
  architecture (for example, an old x-attn state accidentally stored under the GmP
  filename) is rebuilt atomically from its canonical source rather than trusted by name.
- reusable scientific variants are named by what they *are at runtime*, not by an
  ambiguous source filename. In particular, `oai_vanilla_rn_from_xattn.pt` is a
  vanilla OpenAI CLIP state plus the exact trained RN token/config from the canonical
  x-attn donor, and contains **no bridge parameters at all**. Experiment-local bridge
  transplants remain ephemeral and are required to preserve the trained donor bridge/RN
  state bit-for-bit. Run `python reproduce.py list all --models` to see the declared
  model variants for every task.
- stripped-backbone comparisons treat the trained x-attn donor as **inert checkpoint
  data**: ordinary `visual.*` tensors are copied into a vanilla mechinterp shell, while
  RN/READ/bridge/router state is never instantiated or executed. This makes the
  no-bridge/no-RN condition true by construction rather than by a runtime flag.
- reproduction-internal RN helpers are imported through the qualified
  `x_paper_reproduction.rn_control_mechinterp` namespace, so a stale repo-root copy
  left behind by overlaying an older package cannot shadow the current implementation.
- `runtime.model_cache_policy` controls only the derived files under
  `<output_root>/_models`: `keep` (default) reuses them, `task` deletes reproduce-managed
  model files after each successful experiment, and `run` deletes them after a successful
  selected run. User-supplied checkpoints and normal HF/OpenAI download caches are never
  deleted by this setting.
- `datasets.demoset_dir`, `datasets.misc_image_dir`: small local paper/control image
  sets where required.
- `datasets.objectnet_mvt_root`: optional existing canonical ObjectNet-MVT image
  directory. When unset, reproduction first reuses `benchmark_config.json` if a
  complete benchmark install is already configured; otherwise it delegates dataset
  preparation to the same benchmark installer used by `benchmark.py`.

### Shared natural-image population

The historical sink/CLS experiments used a fixed 480-image COCO / Visual Genome /
LAION mixture because those datasets were already available locally and gave a
reasonably varied natural-image population. Public reproduction now replaces that
machine-specific mixture with **ObjectNet-MVT**, which is already a benchmark
dependency of this repository.

`utils_datasets/mvt/human_responses_dedup.csv` is the canonical one-row-per-image
index: 4,771 unique MVT images across 50 labels, with both ObjectNet and ImageNet
stimuli. For the expensive workspace probes, `reproduce.py` deterministically selects
480 images by round-robin sampling over label × source-domain strata, with SHA-256
ordering inside each stratum. The generated manifest is written to
`<output_root>/_manifests/objectnet_mvt_workspace_480.csv` (plus a JSON audit file).

This replacement intentionally preserves the *role* of the historical population
(diverse, nontrivial natural images) rather than claiming sample-identical numerical
reproduction of the original COCO/VG/LAION run. The old
`nop_bc/fixed_sink_manifest.csv` remains only as an archival direct-script fallback;
the public `reproduce.py` path never asks for COCO/VG/LAION roots and never uses it.


### Output location

All automatically managed dumps live below one root. Set it persistently with:

```bash
python reproduce.py setup --non-interactive --output-root D:/clip-paper-reproduction
```

or override it for a single run/status check without changing the JSON config:

```bash
python reproduce.py run paper --output-root D:/clip-paper-reproduction
python reproduce.py status paper --output-root D:/clip-paper-reproduction
```

Every registered task receives a task-specific subdirectory below that root, and
prerequisite/cache lookups use the same overridden root. The output root is a reusable
**reproduction workspace**: partial runs and reruns are allowed in place. The front end
records model fingerprints in `<output_root>/_meta/model_identity.json`; local checkpoints
use streaming SHA-256, while HF repositories use the published weight-file SHA-256 metadata
(or the resolved HF commit when per-file hashes are unavailable). Reusing the root with the
same fingerprints is allowed. A conflicting fingerprint for an already-recorded logical
model slot is the one hard refusal, because that would mix different model weights.

The two legacy bridge probes that historically rejected any non-empty output directory are
automatically invoked with their own `--overwrite` switch *after* this model-identity guard,
so an interrupted/partial task can be rerun in its existing task directory safely.
Script-specific output flags passed after `--` still take precedence for deliberate one-off
experiments.

For unusual script-specific flags, add them under `task_args` in the JSON config,
or pass them to a single run after `--`:

```bash
python reproduce.py run rn.touch_go_transfer -- --max-samples-per-subset 32
```

To inspect model construction before running anything:

```bash
python reproduce.py list all --models
```

To conserve reproduction-workspace disk space while accepting rematerialization cost:

```bash
python reproduce.py run all --model-cache-policy task
```

## Preflight and smoke testing

Do **not** use the full paper experiments as the test suite. Before committing GPU
time, validate the entire selected graph:

```bash
python reproduce.py preflight all
```

This checks every selected task's declared static assets/configuration, compiles the
probe entry points, constructs the exact dispatcher commands, and imports each unique
entry point through its `--help` path. In particular, the RN control-manifold task
requires `x_paper_reproduction/vocab_deduped.txt`; a missing vocabulary now blocks the
run before task 1 instead of after the manifold atlas has already been computed.

Then run a tiny **real integration smoke test**:

```bash
python reproduce.py run all --smoke
```

Smoke mode uses the actual models, loaders, datasets, forward/backward paths, late
post-processing and serialization code, but injects very small batch/sample/grid/search
settings. Its outputs live under `<output_root>/_smoke/`, so smoke markers can neither
poison nor satisfy scientific runs. The shared derived model cache remains reusable.
For the RN manifold, smoke mode deliberately still enters the vocabulary/CLIP-ESE path
with a 32-entry prefix; it does **not** skip the code path that previously failed late.

`preflight` can also validate the smoke overrides themselves without running them:

```bash
python reproduce.py preflight all --smoke
```

If a full RN manifold run is interrupted **after the atlas sweep** but before the
CLIP-ESE/finalization stage, the expensive atlas products can be reused explicitly:

```bash
python reproduce.py run rn.control_manifold -- --resume-after-atlas
```

The resume path checks the saved sample keys, languages, surface-row cardinality and
basis files before reusing them. New runs also write `data/atlas_stage_complete.json`
for stronger compatibility checking. A legacy partial run without that marker can still
be resumed explicitly after structural validation, but prints a provenance warning.

## Cache and DAG behavior

A task is `CACHED` when its paper-facing marker artifact(s) already exist under the
configured output directory. With the default `--with-deps`, prerequisites are run
in topological order and cached prerequisites are skipped. `--force` reruns a completed
task in the same task directory; partial directories are also rerunnable without choosing a
new output root. To execute the entire automatic reproduction graph, including controls, use:

```bash
python reproduce.py run all
```

The dependency graph is inspectable without executing anything:

```bash
python reproduce.py graph paper --external
```

## Public task catalog

| Task | Tier | Kind | Depends on | Purpose |
|---|---|---|---|---|
| `bridge.cross_attention` | canonical | gpu | — | SOURCE / ORTHO / READ / CONTENT bridge |
| `bridge.backbone_dynamics` | canonical | gpu | — | trained-backbone / GIPU dynamics |
| `bridge.read_null.hallucinations` | control | gpu | — | READ/null hallucination controls |
| `bridge.read_null.tap_transplants` | control | gpu | — | late-tap READ/null transplants |
| `bridge.read_null.diagnostic` | control | gpu | — | raw/calibrated READ/null diagnostics |
| `workspace.broadcast_sinks` | canonical | gpu | — | NOP/broadcast sinks and register overlap |
| `workspace.broadcast_channel_interventions` | control | gpu | — | 565/650/123 causal channel interventions |
| `workspace.cls_register_exchange` | canonical | gpu | — | CLS↔register/workspace exchange |
| `workspace.cls_mu_causal` | canonical | gpu | workspace.cls_register_exchange | CLS↔mu causal 2×2 / RN repeat / slot swap |
| `workspace.cls_role_surfaces` | canonical | gpu | workspace.cls_register_exchange | CLS/register role-plane surfaces |
| `workspace.qk_role_gates.scan` | control | gpu | workspace.cls_register_exchange | late Q/K role-gate scan |
| `workspace.qk_role_gates.same_heads` | control | gpu | workspace.cls_register_exchange | fixed B22 same-head Q/K control |
| `workspace.register_geometry.secondary` | canonical | gpu | — | paired secondary register/workspace geometry |
| `workspace.register_geometry.grad_attention` | control | gpu | — | late register attribution trajectory |
| `workspace.register_geometry.principal_angles` | canonical | cpu | workspace.register_geometry.secondary | B20–B22 residual-subspace principal angles |
| `workspace.register_cache_transport` | control | gpu | — | B12→B13 register/cache transport vs RN |
| `workspace.rta_head_population` | canonical | gpu | workspace.cls_register_exchange | RTA head-population RN factorial / OV motifs |
| `workspace.rta_head_contact_sheets` | figure | cpu | workspace.rta_head_population | cached RN motif contact sheets |
| `workspace.single_image.example` | control | gpu | workspace.cls_register_exchange | single-image CLS/register atlas |
| `workspace.single_image.motifs` | control | gpu | workspace.cls_register_exchange | single-image all-head motif catalogue |
| `workspace.text_cls_trajectory` | canonical | gpu | — | exact TEXT→CLS OV/write trajectory |
| `rn.control_mechanism` | canonical | gpu | — | synthetic single-block RN steering mechanism |
| `rn.control_knob` | control | gpu | — | multilingual RN control-knob suite |
| `rn.control_manifold` | control | gpu | — | RN local control-manifold atlas |
| `rn.control_surfaces` | canonical | gpu | — | native multilingual RN control surfaces |
| `rn.control_surface_flow_maps` | figure | cpu | rn.control_surfaces | cached RN surface flow maps / PLY |
| `rn.subspace_alignment` | control | gpu | — | RN-control PCs vs intact register subspaces |
| `rn.stash_followup` | canonical | gpu | — | RN stash impersonation / B11→B13 follow-up |
| `rn.text_relocation` | control | gpu | — | SOURCE/PIECE text-evidence relocation after RN |
| `rn.bridge_transplant` | canonical | gpu | — | bridge/text/visual transplant universality |
| `rn.touch_go_transfer` | canonical | gpu | — | RN transplant: persistent vs B13 touch-and-go |
| `figures.role_text_atlas` | figure | cpu | workspace.rta_head_population | role-vs-site text atlas |
| `figures.rn_mechanism` | figure | cpu | rn.control_mechanism, rn.stash_followup, rn.bridge_transplant | compact RN mechanism paper figures |
| `conv1.gpic_manifold` | canonical | gpu | — | Conv1 semantic manifold with model-matched GPIC final-embedding banks |
| `conv1.xattn_functional_atlas` | canonical | gpu | — | x-attn Conv1 functional atlas / culprit and pair discovery |
| `conv1.vanilla_functional_atlas` | canonical | gpu | — | vanilla/GmP/bare-xattn Conv1 functional atlas |
| `conv1.vanilla_functional_atlas_rn` | control | gpu | — | same atlas with transplanted RN |
| `conv1.residual_axis_lineage` | canonical | gpu | — | residual-axis lineage and classification |
| `conv1.residual_axis_swap_650_565` | control | gpu | — | 650/565 residual-axis swap sweep |
| `conv1.residual_axis_lineage_rn` | control | gpu | — | RN-variant residual-axis lineage |
| `conv1.mlp_neuron_discovery` | canonical | gpu | — | blind MLP-neuron discovery |
| `conv1.b20_writeback_neurons` | canonical | gpu | — | B20 writeback-neuron discovery |
| `conv1.b20_sharpeners_flatteners` | canonical | gpu | conv1.b20_writeback_neurons | B20 sharpener/flattener families |
| `conv1.b20_pushpull_650_715` | canonical | gpu | conv1.b20_sharpeners_flatteners | B20 650/715 push-pull mediation |
| `conv1.register_allocator_tomography` | canonical | gpu | — | Conv1/positional register-allocator tomography |
| `conv1.roleplane_texture_rank.head_rank` | canonical | gpu | — | all-block head true-rank / rigidity |
| `conv1.roleplane_texture_rank.texture_inverse` | canonical | gpu | — | DTD/math texture role-plane + inverse-LAST |
| `conv1.roleplane_texture_rank.compact` | figure | cpu | head_rank, texture_inverse | compact role-plane/texture/rank handoff |
| `conv1.visualtextual_provenance` | canonical | gpu | — | visual/textual provenance + role-plane atlas |
| `conv1.visualtextual_text_direction` | canonical | gpu | — | visual/textual text-direction trajectory atlas |
| `audit.read_probe_router` | utility | audit | — | static READ-probe/router checkpoint audit |


## Conv1 GPIC reference-bank branch

`conv1.gpic_manifold` begins the appended Conv1/early-routing reproduction branch,
so the original automatic task ordering and caches remain intact. The default public
reference source is the model-matched bank under `zer0int/CLIP-GPIC-embeddings`;
Hugging Face model ids map to subdirectories by replacing `/` with `__`. The bank
contains final normalized image embeddings only.

The GPIC dispatcher defaults to the released x-attention model followed by
`openai/clip-vit-large-patch14`. Each model writes to its own model-name subfolder
under `conv1/gpic_manifold`, so several model/bank pairs can coexist in one output
root without weakening the normal model-identity guard. `batch_summary.json` is
written only after every configured model has a real `summary.json`; gated N/A
runs therefore remain retryable. Set `conv1.gpic_models` to change the ordered list,
or use legacy `conv1.gpic_model` as a single-model override.

GPIC source-image retrieval remains gated by the original
`stanford-vision-lab/gpic` dataset. If the gate has not been accepted or this
machine is not authenticated, the task prints an accept/login ASCII notice and
returns a clean N/A without downloading the large reference bank. Selected
neighbors are fetched from the pinned remote TARs by byte range rather than by
downloading whole shards.

Optional compatible custom banks can be listed under
`conv1.custom_embedding_banks`; unavailable local roots are skipped, while a
present bank with the wrong model/space/row geometry is rejected. See
`conv1_embedding_banks/README.md` for the shared exporter format.

## Internal modules

`probe_tools_analysis.py`, `probe_tools_backbone.py`, `probe_tools_repo.py`,
`benchmark_final_clip.py`, and `rn_control_mechinterp/` are implementation helpers. They are not public
reproduction tasks. `probe_tools_rn_control.py` and `probe_tools_rn_manifold.py` are
historical hybrid names: they remain executable because other RN probes import
them, but the public UI exposes them as `rn.control_knob` and
`rn.control_manifold`.

The old raw `reference_lookup.txt` audit and the duplicate repository-root
`rn_control_mechinterp/` copy were deliberately removed; neither was part of the
scientific runtime.


### Smoke/preflight validation

Before a full paper run, use `python reproduce.py preflight all --smoke` and then
`python reproduce.py run all --smoke`. Smoke mode uses a separate `_smoke/` output tree
and tiny real workloads. The front-end validates each exact generated smoke command with
the target subcommand's real argparse parser before any smoke experiment starts, so CLI
drift is reported globally rather than after earlier GPU tasks have run.
