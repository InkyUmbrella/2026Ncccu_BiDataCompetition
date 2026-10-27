# Original DINOv2 attribution

Copyright (c) Meta Platforms, Inc. and affiliates.

The standalone implementation in `src/glaucoma/models/dinov2.py` adapts the original
non-register ViT-S/14 architecture, parameter layout and positional interpolation
from https://github.com/facebookresearch/dinov2 at revision
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`:

- `dinov2/models/vision_transformer.py`
- `dinov2/hub/backbones.py`
- `dinov2/layers/attention.py`, `block.py`, `mlp.py`, `patch_embed.py`, `layer_scale.py`

Changes: only the fixed original ViT-S/14 architecture is retained; native PyTorch
scaled-dot-product attention is used without xFormers; masking, register variants,
stochastic depth, nested tensors and self-supervised training APIs are omitted.
An independent binary classifier head implements this project's `forward(image, age)` API.
The original `mask_token` parameter remains for strict upstream checkpoint loading.

See `LICENSE` for the complete Apache License 2.0 and `MODEL_CARD.md` for upstream
model/data documentation. This attribution does not cover medical variants or DINOv3.

Parity with the pinned upstream model and loading the actual official weights are
pending runtime validation. This source tree does not claim either check passed.
