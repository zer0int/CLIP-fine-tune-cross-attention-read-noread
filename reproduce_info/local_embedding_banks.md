# Adding local/custom embedding banks

`conv1.gpic_manifold` can retrieve against additional local image corpora using the same final-embedding-bank format as the public GPIC bank. This is optional and affects only the GPIC/manifold-surfing task.

## 1. Export one bank per model

The exporter automatically creates a model-specific subdirectory by replacing `/` in the model repo ID with `__`.

For a directory of local images:

```bash
python x_paper_reproduction/conv1_embedding_banks/export_embedding_bank.py local --model zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX --output-root path/to/embeds/out --image-root path/to/images --recursive
python x_paper_reproduction/conv1_embedding_banks/export_embedding_bank.py local --model openai/clip-vit-large-patch14 --output-root path/to/embeds/out --image-root path/to/images --recursive
```

Or use a CSV/Parquet manifest instead of scanning a directory:

```bash
python x_paper_reproduction/conv1_embedding_banks/export_embedding_bank.py local --model openai/clip-vit-large-patch14 --output-root path/to/embeds/out --input-manifest path/to/my_images.parquet --path-column path
```

The resulting layout is model matched:

```text
D:/my_embeddins_example/
  zer0int__CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX/
    bank_config.json
    manifest.parquet
    backbone.safetensors
    content.safetensors

  openai__clip-vit-large-patch14/
    bank_config.json
    manifest.parquet
    backbone.safetensors
```

## 2. Add the bank root to the main reproduction config

Edit only the `conv1.custom_embedding_banks` list in your existing `reproduction_config.json`:

```json
{
  "conv1": {
    "custom_embedding_banks": ["D:/my_embeddins_example"]
  }
}
```

Dataset/output/model settings created by `python reproduce.py setup` remain the main reproduction configuration.

For each runtime model, `conv1.gpic_manifold` looks for the matching `repo__model` subdirectory. Missing optional model subdirectories are skipped cleanly. A subdirectory that does exist but has the wrong model identity, embedding dimensionality, or required embedding spaces is rejected rather than silently mixed into retrieval.

See `x_paper_reproduction/conv1_embedding_banks/README.md` for exporter details, resume behavior, and the GPIC-specific export mode.
