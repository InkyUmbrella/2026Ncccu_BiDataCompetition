# SSL pretraining (MoCo-v2) on the provided train split

Status: implemented (task 08). **The 200/100-epoch run is not launched by this
change**; the orchestrator launches it on the free GPU (5).

## Why

The competition forbids pretrained weights and external data, but
self-supervised pretraining on the **provided train split** is allowed. The
primary arm is MoCo-v2 with a from-scratch ConvNeXt-Tiny encoder. The cheap
go/no-go signal is a frozen linear probe at 384 on `valid`: an SSL encoder must
beat a random-init encoder by **>= +0.010 AUROC** before any fine-tune GPU-hours
are spent (`+0.005..0.010` = candidate only, `< +0.005` = abandon).

## Data policy (hard)

- SSL reads **only** `train.csv` / `train/`. `load_train_rows()` raises if any
  sample's parent directory is not `train` or escapes the configured root.
- Valid and test pixels are **never** loaded by the SSL loader, not for colour
  statistics or sanity images. Normalisation stays fixed `mean=std=0.5`.
- No patient IDs and no fellow-eye positives: two views are augmentations of the
  **same image**; the fellow eye may appear as a negative.
- Labels are used only by the supervised train-probe read-out (and by
  `scripts/linear_probe_ssl.py` on `valid`). The SSL loss never sees a label.
- The residual linear probe inside `pretrain` is fitted and evaluated on
  **disjoint subsets of train**, never on valid.

## Recipe

| Item | Value |
|---|---|
| Encoder | `torchvision.models.convnext_tiny(weights=None)`, features + avgpool + final LayerNorm + flatten (768-d), stochastic depth 0 |
| Projector | `Linear(768, 2048) -> ReLU -> Linear(2048, 128)`, L2-normalised (pretraining only) |
| Views | 224 crops from the existing 384 uint8 cache: `RandomResizedCrop(224, scale=(0.2, 1.0))`, hflip, rotation 15, colour jitter (0.3/0.3/0.2/0.05), grayscale p=0.2, gaussian blur p=0.3, solarize p=0.2 |
| Momentum encoder | EMA of encoder + projector, `m=0.999` |
| Queue | 65,536 keys x 128, FIFO, per process |
| Loss | InfoNCE, temperature 0.2, FP32 logits |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 (no WD on bias/norm), lr 0.03 |
| Schedule | 10-epoch linear warmup, cosine to 0 |
| Precision | FP16 autocast + `torch.amp.GradScaler("cuda")`; momentum/queue updates are skipped when the scaler skips a step |
| Batch | 256 total (`batch_size % world_size == 0`) |
| Epochs | 200 full / 100 cut-down (identical otherwise) |

Augmentation uses torchvision/PIL only. `Augment` is applied to the cache that
already went through crop-black-border + letterbox exactly once, so the SSL
transform never re-crops or re-letterboxes.

## Files

| Path | Contents |
|---|---|
| `src/glaucoma/ssl/config.py` | `SSLConfig` / `SSLDataConfig` / `SSLTrainConfig` (strict JSON, separate from supervised `Config`) |
| `src/glaucoma/ssl/augment.py` | `TwoViewTransform`, `build_ssl_transform`, `build_probe_transform`, `AUGMENTATION_SPEC` |
| `src/glaucoma/ssl/model.py` | `build_encoder`, `build_projector`, `MoCoV2` (queue, EMA, `decode`-free forward) |
| `src/glaucoma/ssl/trainer.py` | `pretrain`, train-only loader, schedule, AMP skips, checkpoints, train-only probe |
| `src/glaucoma/ssl/probe.py` | frozen feature extraction, logistic probe, effective rank / feature std |
| `src/glaucoma/ssl/checkpoint.py` | `kind="ssl_encoder"` artifacts and `kind="ssl_training"` recovery files |
| `scripts/linear_probe_ssl.py` | SSL-init vs random-init frozen probe on `valid` (the gate) |
| `configs/ssl/moco_v2_224_200ep.json`, `..._100ep.json` | frozen configs |
| `tests/test_ssl.py` | CPU synthetic tests |

## CLI

```bash
# launch (single GPU, cut-down is the recommended default)
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src OMP_NUM_THREADS=2 \
python -m glaucoma ssl --config configs/ssl/moco_v2_224_100ep.json \
  --image-cache /var/tmp/ncccu-heyuhang/cache384

# `python -m glaucoma pretrain ...` is an alias; both accept
# --data-root/--output-root/--resume last.pt/--stop-after-epochs N/--image-cache

# 4-GPU (only when the four supervised arms have finished)
NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=2 PYTHONPATH=src \
torchrun --standalone --nnodes=1 --nproc-per-node=4 -m glaucoma ssl \
  --config configs/ssl/moco_v2_224_200ep.json \
  --image-cache /var/tmp/ncccu-heyuhang/cache384

# gate probe (feature extraction at the downstream 384)
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python scripts/linear_probe_ssl.py \
  --encoder outputs/ssl/<run>/encoder_last.pt \
  --config configs/ssl/moco_v2_224_100ep.json \
  --output outputs/ssl/<run>/probe_gate.json

# fine-tune from the SSL encoder (supervised defaults unchanged)
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python -m glaucoma train \
  --config configs/experiments/b0_crop384_control.json \
  --init-encoder outputs/ssl/<run>/encoder_last.pt \
  --image-cache /var/tmp/ncccu-heyuhang/cache384
```

`ssl` prints per epoch: loss, LR, queue fill, effective rank and per-dimension
std of the last momentum keys, plus the train-only linear-probe AUROC every
`probe_every` epochs.

## Run directory

`outputs/ssl/<experiment>/seed<seed>_<utc>_<uuid>/`:

- `config.json`, `metadata.json` (environment, train CSV sha256, cache manifest
  sha256, preprocessing, world size, batch per process)
- `history.json` (epoch, loss, LR, queue fill, effective rank, std, probe,
  seconds), `probe.json`, `manifest.json` (sha256 of every artifact)
- `encoder_last.pt` / `encoder_best.pt` (`kind="ssl_encoder"`, encoder weights
  only, ~114 MB; CUDA tensors are moved to CPU; DDP unwrapped; rank 0 only)
- `last.pt` (`kind="ssl_training"`: online + momentum branches, optimizer,
  scaler, queue, queue pointer, epoch, history, RNG)

`train --init-encoder` calls `read_ssl_encoder()` + `initialize_from_encoder()`
which use `strict=True` on the encoder only; the classifier and age modules stay
freshly initialised, and `metadata.json` records the encoder path + sha256. It
is mutually exclusive with `--resume`/`--init-from`; a mismatched backbone
(`model.name != convnext_tiny`) or shape fails loudly.

## Collapse / stop rules

- Abort if the effective rank of the key features is `< 10` or the mean
  per-dimension std is `< 1e-3` after epoch 20.
- Abort/fallback if more than ~5 % of optimizer steps are skipped by the AMP
  scaler.
- No advantage over the random-init frozen probe (`< +0.005`) after 100 epochs
  means abandon SSL; do not spend the fine-tune hours.

## Measured smoke (GPU 5, 100 real batches)

_Filled by the implementer after the smoke run; see `.w1mer/results/` for the
archived numbers._

## Deliberate deviations from `.w1mer/detail/08_ssl_plan.md`

- Projector output is **128** (queue 65,536 x 128, ~33 MB) as required by the
  task brief; the plan's table said 256.
- Two-view crop scale is **0.2–1.0** and solarize is enabled (task brief), while
  the plan's §2.3 suggested scale 0.55 without solarize. The full transform
  spec is stored in `ssl_config`, so a run is self-describing.
- The per-epoch probe inside `pretrain` is train-only (task requirement); the
  plan's valid-set probe is implemented by `scripts/linear_probe_ssl.py`.
