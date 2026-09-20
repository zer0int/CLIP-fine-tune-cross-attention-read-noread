> To use GPIC [gated], you'll need to accept the terms:\
> HF: [stanford-vision-lab/gpic](https://huggingface.co/datasets/stanford-vision-lab/gpic) 🤗\
> ...and be logged in via HuggingFace CLI.

# Conv1 × GPIC semantic-manifold reproduction

This experiment perturbs selected Conv1 output channels and follows the resulting
trajectory in the model's **final image-embedding spaces**. Nearest-neighbor
interpretation uses model-matched reference banks from the public Hugging Face
dataset repo:

`zer0int/CLIP-GPIC-embeddings`

The bank repo is organized by model repo id with `/` replaced by `__`, e.g.

```text
openai__clip-vit-large-patch14/
zer0int__CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX/
```

`bank_config.json` remains authoritative; the subfolder name is only the lookup
convention.

## Reproduction CLI

The normal public entry point is:

```bash
python reproduce.py run conv1.gpic_manifold
```

By default the public task runs **both** released comparison models, in order:

1. `zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX`
2. `openai/clip-vit-large-patch14`

Each model gets an independent subdirectory below the task output root, using the
same `/` → `__` naming rule as the embedding-bank repo. For example:

```text
out_paper_reproduction/conv1/gpic_manifold/
  zer0int__CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX/
  openai__clip-vit-large-patch14/
  batch_summary.json
```

Configure any ordered comparison set with `conv1.gpic_models`. The legacy
`conv1.gpic_model` field remains a single-model override for old configs. A model
whose own `summary.json` already exists is skipped independently, so interrupted
multi-model runs resume without recomputing completed models.

Vanilla CLIP has the final `backbone` space only. The released x-attention model
also has the final `content` space. The runner never invents a CONTENT bank for a
model that does not expose one.

## GPIC access

The embedding bank itself is public, but plotting/retrieving the corresponding
source images requires access to the gated source dataset
`stanford-vision-lab/gpic`.

Accept its terms in the browser and authenticate the machine with:

```bash
hf auth login
```

If GPIC access is unavailable, this optional task prints an ASCII warning, writes
`SKIPPED_GPIC_NA.json`, exits successfully, and lets `reproduce.py run all`
continue. The access check uses only the tiny model-specific `bank_config.json`;
it happens before downloading the million-row manifest, multi-GiB embedding tensors,
or the runtime model. The skip marker deliberately does **not** count as a completed
result, so a later rerun after authentication can execute the task normally.

Retrieved source images are not bulk-downloaded. The manifest stores each TAR
member's byte offset and encoded size, so remote neighbors are fetched by seeking
into the pinned GPIC TAR and reading only the selected member bytes (HTTP Range
requests under Hugging Face's filesystem layer).

## Optional custom embedding banks

Additional private/local banks can use exactly the same exporter format. Configure
zero or more roots:

```json
{
  "conv1": {
    "custom_embedding_banks": [
      "D:/private/laion_embedding_banks"
    ]
  }
}
```

Each root may contain the same model subfolders as the public GPIC repo. For
example, one root can contain both:

```text
G:/laion_conv1_manifold_alpha_flight/
  zer0int__CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX/
  openai__clip-vit-large-patch14/
```

The runtime model selects only its matching subfolder. If the root exists but a
particular model subfolder is absent, that optional custom bank is skipped for
that model and the run continues with the remaining compatible banks. A present
model-specific bank with the wrong model identity, missing required final-embedding
space, row mismatch, or dimensional mismatch is rejected rather than silently mixed.

Search is exact cosine independently within each compatible bank; the per-bank
top-k candidates are merged into one global top-k. The million-row banks are not
physically concatenated.

To generate a custom bank, see `../conv1_embedding_banks/README.md` and
`../conv1_embedding_banks/export_embedding_bank.py`.

## Intervention config

`experiment_config.json` contains only explicitly selected single channels and
channel pairs. For activation map `r`:

```text
FLIP    target = -r
SHUFFLE target = deterministic spatial permutation of r
ABS     target = |r|

r_alpha = r + alpha * (target - r)
```

Singles trace one-dimensional alpha trajectories. Pairs form an alpha1 × alpha2
surface, including the corresponding single-axis controls.

The shipped paper-reproduction defaults are intentionally explicit in both
`experiment_config.json` and `experiment_config.example.json`. They run these six
pair surfaces:

```text
ch0720 FLIP    × ch0499 FLIP
ch0720 FLIP    × ch0866 FLIP
ch0720 SHUFFLE × ch0866 FLIP
ch0779 FLIP    × ch0499 FLIP
ch0779 FLIP    × ch0866 FLIP
ch0779 SHUFFLE × ch0866 FLIP
```

The corresponding single-channel controls for 499, 720, 779, and 866 are also
listed in the JSON. No GPIC Conv1 channel selection is hidden in Python code; edit
the JSON directly to add/remove channels, conditions, alpha grids, or pairings.

For the x-attention model the run may additionally record CONTENT/READ/ANY
vocabulary behavior and bridge telemetry. Vanilla CLIP automatically disables
those model-specific lanes.

## Main outputs

The run writes portable CSV/Parquet tables plus safetensors, including the source
manifest, event table, final trajectory embeddings, telemetry, nearest-neighbor
results, geometry summaries, plots, model/bank provenance, and `summary.json`.

`summary.json` is the reproduction completion marker.
