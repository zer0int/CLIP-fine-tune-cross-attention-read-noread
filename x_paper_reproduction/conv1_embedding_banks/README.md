# Conv1 final-embedding banks

This directory contains the bank exporter used by the Conv1 reproduction tools.
It writes **final normalized CLIP image embeddings**.

## Layout

A dataset-specific bank root may contain several models side by side:

```text
<bank-root>/
  openai__clip-vit-large-patch14/
    bank_config.json
    manifest.parquet
    backbone.safetensors
    self_test.csv
  zer0int__CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX/
    bank_config.json
    manifest.parquet
    backbone.safetensors
    content.safetensors
    self_test.csv
```

The default model subdirectory is the Hugging Face model repo id with `/`
replaced by `__`. `bank_config.json` contains the original repo id and an exact
canonical state-dict SHA-256, so the folder name is a convenience rather than
the sole model-identity check.

Vanilla checkpoints are loaded through `attnclip_mechinterp_sae`. Checkpoints
whose parameters contain RN/correction/cross-attention components are loaded through
`attnclip_mechinterp_xattn`. Detection is based on checkpoint contents, not the
repo name.

Vanilla CLIP exports `backbone.safetensors`. The trained x-attention model also
exports `content.safetensors` for the final BACKBONE+CONTENT corrected image
embedding.

## GPIC

> To use GPIC [gated], you'll need to accept the terms:\
> HF: [stanford-vision-lab/gpic](https://huggingface.co/datasets/stanford-vision-lab/gpic) 🤗\
> ...and be logged in via HuggingFace CLI.

Use `gpic` mode with the already-processed GPIC test corpus and
`gpic_test_embedding_manifest.parquet`. The output manifest retains the GPIC
shard/TAR offsets required for sparse on-demand image retrieval.

## Custom/private image banks

Use `local` mode either with:

- a CSV/Parquet manifest containing an image-path column, or
- `--image-root` to scan a directory.

The output uses the same bank format as GPIC, so reproduction code can append
compatible custom banks. Custom banks are expected to remain local unless the
underlying images and metadata are redistributable.

For a CSV such as a private LAION example:

```text
index,path,stem
0,D:\AI_DATASET\...\00000000.jpg,00000000
...
```

pass `--path-column path`. Missing source images are treated as an error during
export so row identities cannot silently shift.

## Resume behavior

Generation writes FP16 `.npy.part` memmaps plus `generation_state.json` and is
safe to resume. Re-running the same command resumes. If the model identity,
selection, dimensions, or embedding spaces change, the exporter refuses to
reuse the partial bank; use `--overwrite-bank` intentionally.
