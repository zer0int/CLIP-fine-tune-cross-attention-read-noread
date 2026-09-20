# Private legacy pretraining namespace

This directory intentionally contains the old working hard-text pretraining model and trainer.

Run `train_clip_xattn_bridge.py` rather than importing these modules from the
final trainers. The launcher changes into this directory before starting Python
so `import gmpclipattnamp` resolves here.

Expected softmax ViT-L/14 fingerprints:

- total: 429,961,738
- read trainable: 1,451,015
- content trainable: 724,482
- joint trainable: 2,175,497
- fresh tap logits: [0, -8]

Do not replace this package with the repository-root `gmpclipattnamp`; that root package is the later final architecture.

The soft-token pre1 script is also executed from this directory so its explicit
`import oaicliporg as clip` resolves to the bridge-free historical
implementation. The shared HF-state loader receives that module and rebuilds
the backbone through it; the final bridge classes are never instantiated in
this stage.
