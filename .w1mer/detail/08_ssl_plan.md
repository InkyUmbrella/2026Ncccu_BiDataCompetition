---
id: 08
title: "Task 08 plan — self-supervised pretraining on the provided train images"
state: active
task: "08"
base: fafcf7dcc6a1bce6e5ca8f4922891baeac4a6cfc
date: 2026-10-22
---

# Task 08 — self-supervised pretraining on the provided train images

Status: **plan only** (explorer, read-only). Baseline `fafcf7d` (branch
`heyuhang/w1mer-opt`). Owner of implementation: next implementer. This document
is the change contract; no new design decisions should be needed.

> **Evidence labels.** `[M]` = measured earlier by the team (docs/logs quoted
> below); `[E]` = analytical estimate derived from `[M]`, not measured; `[T]` =
> must be measured by the implementer with the commands in §7.
> The explorer has read-only file tools and cannot run Python/ssh. All
> throughputs and wall-clocks are therefore `[E]` from the one measured
> supervised number (81.05 img/s per V100 at 384 with cache) plus FLOP scaling.

---

## 0. TL;DR

1. **Primary method: MoCo-v2**, ConvNeXt-Tiny encoder, 224×224, batch 256,
   queue 65,536, temperature 0.2, momentum 0.999, SGD 0.03, 200 epochs
   (cut-down: 100 epochs). **Fallback: SimCLR**, same encoder/resolution,
   batch 512, temperature 0.2, 100 epochs. Reject MoCo-v3, DINO, MAE, BYOL for
   this budget (§1).
2. **Data:** train split only, unlabelled. Never load valid/test pixels for SSL.
   No patient-pair construction; the fellow eye is **not** a positive view.
   Image-level sampling, two augmented views of the same image only.
3. **Code:** new `src/glaucoma/ssl/` package (config, augment, model, trainer,
   probe, checkpoint helpers), one CLI subcommand `pretrain`, one new
   `train --init-encoder` flag. The encoder is saved as a self-describing
   `kind="ssl_encoder"` torch file (~114 MB) and loaded into the existing
   `ConvNeXtClassifier.encoder` with `strict=True`. Supervised defaults are
   untouched.
4. **Budget:** one V100, 200 epochs @224 ≈ **16–19 h [E]**; 100 epochs ≈
   **8–10 h [E]**. A 60-epoch 384 fine-tune is ~**11 h [E]** on the same GPU.
   The cut-down arm is **≈ 20–21 h end-to-end on one GPU** (fits 24 h).
   4-GPU DDP (after the four current arms finish) is ≈ **5 h [E]** for 200
   epochs with `NCCL_P2P_DISABLE=1`; DDP efficiency 84–92 %, not 4×.
5. **Proof:** frozen linear probe must beat a random-init probe before any
   fine-tune is spent; then compare SSL-init 60-epoch fine-tune against
   (a) scratch 60 epochs, same seed, and (b) a **compute-matched scratch
   control** `60 + SSL_GPU_hours / 0.178 h` epochs (≈120 epochs for the
   cut-down arm), with paired bootstrap over the 5,849 valid images.
   Meaningful effect: **ΔAUROC ≥ 0.010 with 95 % paired-CI lower bound > 0**;
   0.005–0.010 is only a candidate needing a second seed; <0.005 is noise.
6. **Stop rule:** at 100 epochs, if the 384 frozen probe is not at least
   +0.005 AUROC over the random-init probe, abandon SSL and spend the remaining
   GPU budget on the ensemble/longer-supervised queue.

---

## 1. Method choice

### 1.1 Why this is a hard SSL setting

- 51,396 train images but only **~26k independent patients** (47 % of adjacent
  rows are same-age both-eye pairs; DATA_AUDIT). Effective sample count is
  smaller than it looks.
- 3.2 % positives: the label signal is subtle (optic-disc cupping/rim), and
  most of the image is background/retina texture. Reconstruction or generic
  instance discrimination can spend capacity on acquisition/device signatures.
- Multi-device domain shift is documented (same source mix but several camera
  geometries). A useful encoder must be **device-invariant**, not just
  patient-discriminative.
- 16 GB V100 without NVLink, FP16 only (no bf16 tensor cores), and 1–2 days of
  GPU budget. Literature-scale SSL (800–1600 epochs, batch 4096) is out.
- Downstream resolution is 384; the encoder must transfer across a 224→384
  scale change. ConvNeXt is convolutional and all-normalisation, so this is
  structurally safe, but it is still a distribution shift.

### 1.2 Comparison

| Method | Forward-equivalents / sample | Batch needed | Typical schedule | Fit to 50k fundus, 1–2 days | Verdict |
|---|---:|---:|---|---|---|
| **MoCo-v2** | ~4 (q fwd+bwd, k fwd) | 128–256 + queue | 200–800 ep | Queue decouples batch; strongest practical choice at this scale; simple InfoNCE | **Primary** |
| **SimCLR** | ~6 (2 × fwd+bwd) | 1024–4096 | 400–1000 ep | Needs large batch for enough negatives; 16 GB caps ~512; 50–100 % slower | **Fallback** |
| MoCo-v3 | ~5 (symmetric + predictor) | 1024+ | 300–800 ep | Tuned for ViT/large batch; extra predictor/symmetrisation risk without evidence at 50k | Reject |
| BYOL | ~4 (+EMA) | 256–4096 | 300–1000 ep | No negatives, collapse-prone; sensitive to batch/BN; less proven on small fundus | Reject |
| DINO | ~8 (2 global + 6 local) | 1024+ | 300–800 ep | Multi-crop compute, centering/teacher-student, collapse diagnostics; too much code/time | Reject |
| MAE | ~3 + decoder | 512+ | 800–1600 ep | ConvNeXt is not sparse; zero-masking does not save conv FLOPs; short schedules underfit; reconstruction wastes capacity on black border | Reject |

**Decision:** MoCo-v2 primary. It is the only method here whose negative pool
does not depend on a large batch, it has one loss term and one extra
no-gradient forward, and its failure modes (queue collapse, stale queue) are
easy to monitor. SimCLR is the fallback only because it is the simplest code;
it is expected to be weaker at batch 512/100 epochs.

### 1.3 Primary recipe (freeze these values)

| Item | Value |
|---|---|
| Encoder | `torchvision.models.convnext_tiny(weights=None)`, features + avgpool + final LayerNorm + flatten → 768-d; `stochastic_depth_prob=0.0` during SSL |
| Projector | `Linear(768, 2048) → ReLU → Linear(2048, 256)`, L2-normalised |
| Resolution | 224×224 (two views from the 384 cache) |
| Batch | 256 total (`256 / world_size` per GPU, must divide) |
| Queue | 65,536 keys × 256 dims, FIFO; per-rank queue in DDP |
| Temperature | 0.2 |
| Key momentum | 0.999 |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 (no WD on bias/norm), `lr = 0.03 × batch/256` |
| Schedule | 10-epoch linear warmup, cosine to 0 at the end |
| Precision | FP16 autocast + `GradScaler("cuda")`; loss/logits in FP32; skip momentum/queue update on scaler skip |
| Epochs | **200** full; **100** cut-down (same everything else) |
| Augmentation | see §2.3 |
| Checkpoint | every 10 epochs + last; probe at 50/100/150/200 (cut-down: 50/100) |

If OOM at 224: drop total batch to 128 and lr to 0.015; keep queue 65,536.
Do **not** reduce queue size — it is the main source of negatives.

### 1.4 Fallback recipe (only if MoCo code fails review or collapses)

SimCLR: same encoder/projector/224, batch 512, temperature 0.2, SGD 0.03
(`lr = 0.03 × batch/256`), 10-epoch warmup + cosine, 100 epochs, NT-Xent over
the 2×512 views. Expected cost ≈1.5× MoCo for the same epochs; cap at 12 GPU-h.
Do not run both in parallel: one arm only.

---

## 2. Data policy

### 2.1 Split rules (hard)

- SSL reads **only** `train.csv` / `train/`:
  `read_samples(root, "train", require_labels=False)`. Assert
  `len(rows) == 51,396` and every `row.path.parent.name == "train"`.
- **Valid pixels are never loaded during SSL**, not even for colour statistics
  or a sanity image. The lossless cache contains valid entries; instantiate
  `ImageCache` with the train samples only, so only `train/*.npy` is read.
- **Test is forbidden** for SSL, probes, normalisation, or debugging.
- Normalisation stays fixed `mean=std=0.5` (same as the supervised baseline), so
  there is no data-dependent statistic to leak.
- Labels from `train.csv` are used **only** by the linear probe (a supervised
  read-out of frozen features). The SSL loss never sees a label. Valid labels
  are used for probe evaluation and model selection only.

### 2.2 Patient / both-eye structure

DATA_AUDIT: 47 % of adjacent train rows share an exact age; 22,192 same-age
groups of size 2; adjacent positive-positive rate 7× chance. There is **no
patient ID** in the CSV. Therefore:

- Build **no patient IDs and no pair sampler**. Sample images i.i.d. from the
  train split, as the supervised runs already do.
- The two views of a positive pair are **augmentations of the same image**.
  Never use the fellow eye as a positive view. Reasons:
  1. Fellow eyes share patient identity and are strongly label-correlated:
     a positive eye's fellow eye is positive in 31 % of positive pairs
     (7× chance, DATA_AUDIT). Treating them as positive would inject a weak
     label proxy and turn "patient recognition" into the pretext task.
  2. It would force invariance to eye-specific anatomy/acquisition differences
     that may be diagnostic, and it makes the task easier in a way that does
     not match single-eye inference.
  3. The adjacency heuristic is not a reliable patient ID; a wrong pair creates
     a false positive that directly corrupts the contrastive objective.
- Consequence: the fellow eye can appear in the same batch/queue as a
  **negative**. This is accepted (it is at worst a hard negative; the
  supervised baseline has the same information). Do not "fix" it with the
  adjacency heuristic. Colour/blur augmentation (below) limits the model's
  ability to key on eye laterality/device.

### 2.3 Two-view augmentation (`fundus_moco_v2_v1`)

Views are produced from the **already cropped/letterboxed 384 cache**, so the
SSL transform must not re-apply crop-black-border or letterbox.

1. `RandomResizedCrop(224, scale=(0.55, 1.0), ratio=(0.75, 1.333), antialias=True)`.
   Deliberately conservative: the optic disc is only ~35–45 px at 384, and
   aggressive ImageNet crops can remove it.
2. `RandomHorizontalFlip(p=0.5)`. No vertical flip (anatomically implausible).
3. `RandomRotation(15°)`, bilinear, fill 0.
4. `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)`.
5. `RandomGrayscale(p=0.2)`.
6. `GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))` with p=0.3.
7. `ToTensor()`, `Normalize((0.5,)*3, (0.5,)*3)`.

No solarisation/equalisation (destroys clinical colour), no cutout (may erase
the disc). Photometric augmentation is intentionally stronger than the
supervised recipe to attack device shift.

Probe/fine-tune transforms are **not** changed: probes use deterministic
resize/letterbox to the probe size + normalise; fine-tune keeps
`build_transform(384, training=...)` exactly as today.

### 2.4 Reuse of the existing loader without changing supervised defaults

`FundusDataset.__getitem__` stores `self.transform(image)` under `"image"`.
Pass a `TwoViewTransform` that returns a **tuple of two tensors**; PyTorch
`default_collate` turns `[(v1,v2), ...]` into `[batch_v1, batch_v2]`. No change
to `FundusDataset`, `ImageCache`, or `build_transform` is required.

```python
rows = read_samples(cfg.data.root, "train", require_labels=False)
cache = ImageCache(cfg.data.cache, rows, 384, **preprocessing)   # 384 cache
dataset = FundusDataset(rows, AgeStats.fit(r.age for r in rows),
                        build_ssl_transform(cfg.data.ssl_size), image_cache=cache)
# batch["image"] == [view1: [B,3,224,224], view2: [B,3,224,224]]
```

The cache is the existing `/var/tmp/ncccu-heyuhang/cache384` (train+valid,
~25 GB) or any path with the same manifest. SSL must accept `--image-cache`;
without it, NFS decode at ~25 img/s would make 224 pointless.

---

## 3. Code design (exact change surface)

All new code under `src/glaucoma/ssl/`; minimal edits to existing files.
Follow current conventions: stdlib + torch/torchvision/numpy/sklearn/PIL only,
strict JSON configs, `atomic_write`/`save_json`, deterministic seeding,
`RuntimeError` on decode/transform failure, tests are stdlib `unittest`.

### 3.1 New files

| File | Contents |
|---|---|
| `src/glaucoma/ssl/__init__.py` | re-export `SSLConfig`, `pretrain` |
| `src/glaucoma/ssl/config.py` | `SSLDataConfig`, `SSLTrainConfig`, `SSLConfig` (`load`, `from_dict`, `to_dict`, `validate`). Fields: `root`, `cache`, `image_size=384` (cache), `ssl_size=224`, `num_workers=8`, `crop_black_border/threshold/margin`; `epochs`, `batch_size`, `learning_rate`, `momentum`, `weight_decay`, `warmup_epochs`, `temperature`, `queue_size`, `moco_momentum`, `amp`, `save_every`, `probe_every`, `grad_clip` (default 0 = off). Validate types/ranges; reject `batch_size % 256`-style mistakes only where needed (`queue_size>0`, `0<temp<1`, `0<moco_momentum<1`, `ssl_size` in [96,384], `batch_size` divisible by world size at runtime). |
| `src/glaucoma/ssl/augment.py` | `TwoViewTransform` (`__call__(PIL.Image) -> tuple[Tensor, Tensor]`), `build_ssl_transform(size)`, `build_probe_transform(size, crop_black_border, ...)`, `AUGMENTATION_SPEC = "fundus_moco_v2_v1"`. Uses torchvision transforms only; no new dependency. |
| `src/glaucoma/ssl/model.py` | `build_encoder(stochastic_depth_prob=0.0)` returning the same `nn.Sequential(backbone.features, backbone.avgpool, backbone.classifier[0], nn.Flatten(1))` used by `ConvNeXtClassifier`; `MoCoV2(nn.Module)` with online `encoder` + `projector` (trainable), `momentum_encoder` (`requires_grad_(False)`, EMA copy), `queue`/`queue_ptr` buffers (non-persistent), `forward(view_q, view_k) -> (logits, targets)`, `@torch.no_grad() momentum_update()`, `@torch.no_grad() dequeue_and_enqueue(keys)`, `encode(x)` for probes. No DDP-specific code in the module. |
| `src/glaucoma/ssl/trainer.py` | `pretrain(config, resume=None, stop_after_epochs=None)`; `make_ssl_loader(dataset, total_batch, world_size, rank, ...)`; DDP init/finalise; epoch loop; AMP/scaler skip handling (do not update momentum/queue on skip); cosine+warmup; history JSON; checkpoint save; calls probe every `probe_every`; prints one line/epoch. Rejects any split other than train. |
| `src/glaucoma/ssl/probe.py` | `extract_features(encoder, loader, device, amp) -> (features, labels)`; `fit_probe(train_x, train_y, C=1.0)` (StandardScaler + `LogisticRegression(class_weight="balanced", max_iter=2000)`); `evaluate_probe` returns `{auroc, ap, auprc_trapz, positives, count}`; `effective_rank(features)` for collapse monitoring; `probe_metrics(...)` convenience. Uses valid **features and labels only** at evaluation time. |
| `src/glaucoma/ssl/checkpoint.py` | `encoder_payload(model, config, metadata, epoch, metrics)`, `save_encoder(path, ...)`, `save_recovery/read_recovery`. **Unwrap DDP (`model.module`) and save on rank 0 only.** Encoder payload contains **only** `encoder_state` (keys of `model.encoder.state_dict()`) — never the projector, momentum encoder, or queue. Reading lives in `engine/checkpoint.py` (§3.2) so the fine-tune path has no dependency on the SSL trainer. |

**DDP detail (critical):** wrap with
`DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)`.
The queue is a 67 MB buffer; leaving `broadcast_buffers=True` would broadcast it
every forward and destroy throughput. Each rank therefore maintains its own
queue (more negatives; gradients are still averaged). Use
`DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=True)` and call
`sampler.set_epoch(epoch)`; total batch stays 256 so the LR schedule and
steps/epoch (201) are world-size independent. Broadcast the initial online
weights from rank 0 once so all ranks start identical even if seeding differs.
The existing supervised trainer is **not** converted to DDP.

### 3.2 Existing-file edits (small, default-preserving)

**`src/glaucoma/engine/checkpoint.py`**
```python
def read_ssl_encoder(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("kind") != "ssl_encoder":
        raise ValueError(f"Expected version 1 ssl_encoder checkpoint: {path}")
    return payload

def initialize_from_encoder(model, payload):
    """Load only the image encoder; head/age modules stay freshly initialised."""
    if payload["model"]["name"] != "convnext_tiny":
        raise ValueError("Only convnext_tiny SSL encoders are supported")
    model.encoder.load_state_dict(payload["encoder_state"], strict=True)
```
No change to `read_checkpoint`, `load_model`, `initialize_model`, or existing
formats.

**`src/glaucoma/engine/trainer.py`**
- `train(config, resume=None, stop_after_epochs=None, image_cache=None, init_from=None, init_encoder=None)`.
- Errors: `init_encoder` with `resume` or `init_from` (mutually exclusive).
- After `model = build_model(...)`: if `init_encoder`, `enc = read_ssl_encoder(...)`,
  `initialize_from_encoder(model, enc)`. Optimizer is built **after** init, so
  no optimizer state is inherited.
- Metadata: `initialized_from_encoder=str(path)` and
  `initialized_from_encoder_sha256=sha256_file(path)`.
- Do not change any default numerics; don't touch the EMA/mixup/sampling paths.

**`src/glaucoma/cli.py`**
- `train` mutually exclusive group gains
  `--init-encoder help="ssl_encoder .pt; initialise image encoder only"`; pass
  through to `train(...)`.
- New subcommand:
  ```
  glaucoma pretrain --config configs/ssl/moco_v2_224_200ep.json
                    [--data-root] [--output-root] [--resume last.pt]
                    [--stop-after-epochs N] [--image-cache PATH]
  ```
  Under `torchrun`, device = `cuda:{LOCAL_RANK}`; under plain `python`, `cuda:0`.
- `models` output is unchanged.

### 3.3 Encoder file format (the contract)

```python
{
  "format_version": 1,
  "kind": "ssl_encoder",
  "model": {"name": "convnext_tiny", "params": {"stochastic_depth_prob": 0.0}},
  "encoder_state": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
  "ssl_config": {...},                     # full validated JSON config
  "data": {
      "split": "train", "count": 51396,
      "csv_sha256": "<sha256 train.csv>",
      "cache_manifest_sha256": "<sha256 manifest.json>",
      "cache_preprocessing": {...}, "ssl_size": 224,
  },
  "augmentation": "fundus_moco_v2_v1",
  "epoch": 200,
  "metrics": {...},                        # last probe + loss, not used for selection downstream
  "environment": {...}                     # environment_metadata() output
}
```

Size ≈ 114 MB (28 M params fp32), within the 500 MB submission limit; it is
**not** a submission artifact, only an init file. Save with `atomic_write` +
`torch.save`. The fine-tune run dir records its path and sha256.

### 3.4 CLI workflows

```bash
# single GPU, primary
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src OMP_NUM_THREADS=2 \
python -m glaucoma pretrain --config configs/ssl/moco_v2_224_200ep.json \
  --image-cache /var/tmp/ncccu-heyuhang/cache384

# 4 GPUs (only after the four supervised arms finish)
NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=2 PYTHONPATH=src \
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m glaucoma pretrain --config configs/ssl/moco_v2_224_200ep.json \
  --image-cache /var/tmp/ncccu-heyuhang/cache384

# fine-tune: exactly the existing control recipe + encoder init
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python -m glaucoma train \
  --config configs/experiments/b0_crop384_control.json \
  --init-encoder outputs/ssl/.../encoder_best.pt \
  --image-cache /var/tmp/ncccu-heyuhang/cache384
```

Fine-tuning uses the **unmodified** control config (384, pos_weight 5, no
sampling, no EMA/mixup) so the only delta vs the existing scratch arm is the
initialisation.

### 3.5 Tests (CPU, no competition images)

`tests/test_ssl.py`:
1. `TwoViewTransform` returns 2 tensors `[3,224,224]`, values in [-1,1],
   deterministic after `torch.manual_seed`, and does not mutate the input.
2. `MoCoV2`: queue FIFO order and wrap-around; `momentum_update` changes
   `momentum_encoder` only; momentum params have `requires_grad=False`;
   forward returns `[B, Q+1]` logits and zero targets; loss finite.
3. `dequeue_and_enqueue` works for `B > queue_size` (raise or clamp; choose
   raise) and after resume.
4. `read_ssl_encoder` rejects wrong `format_version`/`kind`/model.
5. `initialize_from_encoder` loads encoder weights bit-exactly, leaves head and
   (for `convnext_tiny_age_residual`) `age_residual` untouched; rejects a
   mismatched encoder shape.
6. `train(..., init_encoder=...)` on the tiny synthetic fixture runs one epoch,
   writes `metadata.json` with the encoder hash, and `resume`+`init_encoder`
   raises.
7. `SSLConfig.validate` rejects bad temperature/momentum/queue/size/batch;
   JSON round-trip stable.
8. `pretrain` refuses rows whose parent split is not `train` (monkeypatch a
   valid row and expect `ValueError`).

GPU smoke (synthetic 64 px, 2 epochs, batch 8):
`scripts/smoke_ssl.py` — checks AMP, queue update, checkpoint round-trip,
`encoder_best.pt` → `train --init-encoder` one step. Use
`CUDA_VISIBLE_DEVICES=5`.

### 3.6 Config files

`configs/ssl/moco_v2_224_200ep.json`:
```json
{
  "experiment": "ssl_moco_v2_224_200ep",
  "seed": 42,
  "output_root": "outputs/ssl",
  "data": {
    "root": "/home/lichunyu/competition/dataset/glaucoma_first",
    "cache": "/var/tmp/ncccu-heyuhang/cache384",
    "image_size": 384, "ssl_size": 224, "num_workers": 8,
    "crop_black_border": true, "crop_threshold": 10, "crop_margin": 0.03
  },
  "train": {
    "epochs": 200, "batch_size": 256, "learning_rate": 0.03,
    "momentum": 0.9, "weight_decay": 0.0001, "warmup_epochs": 10,
    "temperature": 0.2, "queue_size": 65536, "moco_momentum": 0.999,
    "amp": true, "save_every": 10, "probe_every": 50
  }
}
```
`configs/ssl/moco_v2_224_100ep.json` is identical with `epochs: 100`.
`probe_every: 50` means the full 384 frozen probe runs at epochs 50/100/150/200
(full arm) or 50/100 (cut-down); the cheap effective-rank/feature-std monitor
runs every epoch. Probes use the **full train split** (51k×768 is small and
the probe is only run 2–4 times). If probe time becomes a bottleneck, use all
1,657 positives + 10,343 random negatives (seed fixed).

### 3.7 Reproducibility record (must be written by the run)

Run dir `outputs/ssl/<experiment>/seed42_<utc>_<uuid>/`:
- `config.json` (validated), `metadata.json` (`environment_metadata()` + train
  CSV sha256 + cache manifest sha256 + world size + git revision + source
  sha256), `history.json` (epoch, loss, lr, seconds, scaler skips, effective
  rank, probe metrics), `probe.json` (all probe results), `train.log`.
- `encoder_best.pt`, `encoder_last.pt`, `last.pt` (recovery: online+momentum+
  optimizer+scaler+queue+RNG+epoch), `manifest.json` with sha256 of every
  artifact.
- The fine-tune run's `metadata.json` must contain the encoder path + sha256;
  the reviewer must be able to trace a quoted AUROC to `best.pt` sha256 and
  `source_sha256`.

---

## 4. Compute budget

### 4.1 Throughput derivation

- Measured: **81.05 img/s per V100 at 384** (1,024-image cached short run,
  includes loader startup; `docs/TRAINING_SPEED.md`) `[M]`. Use 80 `[E]`.
- FLOP scaling to 224: `80 × (384/224)² = 235 img/s` for supervised
  forward+backward `[E]`.
- MoCo-v2 per step = online fwd+bwd (3 forward-equivalents) + key fwd
  (1) = 4/3 of supervised → **~176 source img/s per GPU** `[E]`.
- One epoch: `51,396 / 176 ≈ 292 s = 4.9 min` `[E]`.
- Add 15–20 % for two-view CPU transforms, probe passes and logging.

| Config | 100 epochs | 200 epochs |
|---|---:|---:|
| 1 V100 | **8.1 h ideal / 9–10 h realistic** `[E]` | **16.2 h / 18–20 h** `[E]` |
| 2 V100 DDP | 4.4 / 5–5.5 h `[E]` | 8.8 / 10–11 h `[E]` |
| 4 V100 DDP | 2.3 / 2.8–3 h `[E]` | 4.7 / 5.5–6 h `[E]` |

DDP assumptions: `NCCL_P2P_DISABLE=1`, host-SHM path 2.1 GB/s for 16 MiB
`[M]`. ConvNeXt-Tiny + projector ≈30 M fp32 gradients ≈120 MB per all-reduce
→ **~57 ms/step** `[E]` (2.1 GB/s). At 4 GPUs the step is ~0.36 s, so
communication is ~15 %: expect **84–92 % DDP efficiency**, not 4×. The
existing supervised trainer has no DDP; only SSL gets multi-GPU support.

### 4.2 Scheduling

The four 60-epoch 384 arms are about to run on 4 GPUs for ~11 h. Start SSL on
the **fifth GPU** immediately (GPU whitelist is `0,1,3,4,5`; GPU 2 is not ours,
see R_01 F6). Do not wait for a multi-GPU slot. When the arms finish, either
(a) let the 1-GPU run continue, or (b) if a 2–4 GPU slot exists and the run is
<50 % done, restart from `encoder_last.pt` with `--resume` at the larger world
size (optimizer/queue compatible; RNG differs → document that the resumed run
is not bit-identical). **Recommendation: do not restart for <2× speedup**;
keep the single-GPU 100-epoch cut-down as the default so it finishes alongside
the arms.

Fine-tune is single-GPU (existing trainer is not DDP): 60 epochs × 10.7 min
≈ **10.7 h + validation ≈ 11–12 h** `[E]`.

### 4.3 Cut-down variant that fits 24 h (one GPU, end-to-end)

- SSL: **100 epochs @224**, batch 256, queue 65,536 → **9–10 h**.
- Probe at 50/100 + decision + write encoder: **~0.5 h**.
- Fine-tune: 60 epochs @384, control recipe → **11–12 h**.
- **Total ≈ 21–22 h** on one GPU, plus ~1 h smoke/setup = fits 24 h. This
  cut-down is **SSL + fine-tune only**; the compute-matched control C is a
  separate ~21 h job and does not fit the same 24 h on one GPU.
- With 3 GPUs, run C concurrently with SSL and fine-tune B on the third GPU
  after SSL: wall-clock ≈ 10 h (SSL) / 21 h (C, overlapping) and B finishes
  ~21–22 h from launch. With 2 GPUs, run SSL then B, and C afterwards
  (total ≈ 32 h critical path).

**Full arm** (200 epochs) is only worth it if either (a) a 4-GPU slot is free
after the arms, or (b) the 100-epoch probe is clearly positive (≥+0.01) and we
want the best possible encoder for the final submission. Cap the full arm at
**20 GPU-h**; stop at the last checkpoint if exceeded.

### 4.4 Total GPU-hour envelope

| Item | GPU-h |
|---|---:|
| Primary SSL, 100 ep | 9–10 |
| Probe + smoke | 1 |
| Fine-tune 60 ep | 11–12 |
| Compute-matched scratch control (120 ep) | 21–22 |
| **Minimum go/no-go set** | **~32 (SSL+probe+FT) / ~42 with control** |
| Fallback SimCLR cap | 12 |
| Absolute SSL cap before abandonment | 20 |

On the 5-GPU server, launch C on a second free GPU at the same time as SSL;
then the go/no-go critical path is SSL (~10 h) → fine-tune B (~11 h) ≈ **21–22 h
wall-clock**, while C finishes in the background. On one GPU the same set is
≈42 h, so the cut-down 24 h plan runs SSL + B only and defers C.

---

## 5. Evaluation protocol — proving SSL helps, not just more epochs

### 5.1 Stage 0: cheap signal (no fine-tune)

1. Build a random-init `convnext_tiny` (seed 42) and run the frozen probe at
   **384×384** on the full train split, evaluate on valid. Record AUROC/AP as
   `probe_random`. This is the representation null model. `[T]` expected
   0.85–0.90 `[E]`.
2. Run SSL with probes at 50/100/150/200 epochs, features extracted at **384**
   (matches downstream; no fine-tuning of the encoder). Use the deterministic
   probe transform: resize the 384 cache to 384 (no-op), ToTensor, normalise.
3. Decision at 100 epochs (cut-down) or 100/200 (full):
   - `probe_ssl - probe_random ≥ +0.010` → proceed to fine-tune.
   - `+0.005 … +0.010` → proceed only if a GPU is idle; otherwise stop.
   - `< +0.005` → **abandon SSL** (write the negative result; do not spend the
     fine-tune GPU-hours).
4. Also report `effective_rank` of the key features (entropy of the singular
   values of the centred 256×256 batch Gram matrix) and per-dimension std.
   Collapse/abort if rank <10 or mean std <1e-3 after epoch 20.

### 5.2 Stage 1: matched fine-tune comparison

Three arms, all at 384, same seed 42, same control config (384, crop-black,
pos_weight 5, no sampling/EMA/mixup, 60 epochs unless stated), same cache:

| Arm | Init | Epochs | Purpose |
|---|---|---:|---|
| A | random | 60 | existing control (already queued) |
| B | SSL encoder | 60 | does SSL init help at equal fine-tune budget? |
| C | random | **60 + ceil(SSL_GPU_h / 0.178)** ≈ **120** for 100-ep SSL | compute-matched control: does SSL beat simply training longer? |

Arm C must use a cosine schedule set to its own epoch count (120), same
optimizer/augmentation. Do not compare to the 60-epoch arm and call it
compute-matched. If the SSL run uses 200 epochs, C is ≈168 epochs; only run it
if the 4-GPU/2-day budget allows.

### 5.3 Statistical test

- Paired bootstrap over the **5,849 valid images** (eye-level; no patient ID):
  10,000 resamples, same resample indices for both models, compute
  `ΔAUROC = AUROC_B - AUROC_A` (and B−C). Report mean, 2.5/97.5 percentiles,
  and `P(Δ>0)`.
- Reference uncertainty: `n1=191`, `A≈0.93` → Hanley–McNeil
  `SE(AUROC)≈0.0129`; with score correlation ρ≈0.90–0.95 the paired
  `SE(Δ)≈0.004–0.006`. Thus:
  - **ΔAUROC ≥ 0.010** is the smallest effect worth claiming; require the 95 %
    paired-CI lower bound >0.
  - 0.005–0.010: candidate only; repeat with seeds 43/44 (or a second SSL
    pretraining seed) and require sign agreement.
  - <0.005: inconclusive/noise; do not put it in the final recipe.
  - `ΔAP`/`ΔAUPRC` are noisier still (`n1=191`); report them but never gate on
    a PR difference <0.01.
- Selection bias: both A/B use the same `best.pt` rule, so the paired delta is
  partially protected, but the reported absolute AUROC is still max-of-60
  optimistic. Additionally evaluate a **pre-declared fixed epoch 60** checkpoint
  for A and B and require the same sign as the `best.pt` delta.
- State the limitation in the result doc: eye-level bootstrap CIs are too
  narrow because same-patient eyes can straddle train/valid; no patient ID
  exists.

### 5.4 What "success" looks like

- Primary claim: `B > C` with paired 95 % CI lower bound >0 and
  `ΔAUROC ≥ 0.010` from one seed, confirmed by a second seed (or a second SSL
  seed) with the same sign.
- Secondary: `B > A` by a similar margin; `probe_ssl > probe_random` by ≥0.01;
  no degradation in sens@spec95; AUPRC not worse.
- If `B ≈ C` (SSL only matches longer supervised training), SSL is **not**
  worth its complexity; record it and prefer longer supervised training +
  ensemble for submission. That is a valid, useful negative result.

### 5.5 Expected value (estimate, not a promise)

- Point estimate: **+0.005 to +0.015 valid AUROC** over the 60-epoch scratch
  control, most likely **+0.005 to +0.010**; AUPRC could move more but is too
  noisy to claim at 191 positives. `[E]`
- Probability of a null/inconclusive result is material (30–50 % `[E]`): 51k
  images, ~26k patients, strong device shift, and a short 100–200 epoch
  schedule are all below SSL-paper scale.
- Upside beyond the point estimate: an SSL encoder is a **diverse second
  backbone** for task 07's ensemble even if its single-model AUROC is only
  equal; but do not spend ensemble budget on it unless `B` is at least
  non-inferior to `A`.
- The cheap frozen probe (Stage 0) is the real value: it answers "does this
  representation contain more label signal than random?" in ~10 GPU-h, before
  the 11 h fine-tune.

---

## 6. Risks and stop criteria

### 6.1 Why SSL may not help here

| Risk | Why it bites | Mitigation / evidence |
|---|---|---|
| Small effective dataset | ~26k patients, 51k eyes; SSL papers use 1 M+ images and 800+ epochs | 100–200 epochs and a queue; linear-probe gate before fine-tune |
| Supervised baseline is already strong | 60 epochs = 3.1 M samples; 0.923 AUROC | compute-matched control C, not just A |
| Device/site shift | Instance discrimination can encode camera/black-border signature | strong colour/grayscale/blur aug; conservative crop; probe per-device not possible (no device ID) |
| Subtle label signal | CDR/rim needs the optic disc; random crops can remove it | crop scale floor 0.55, no cutout, moderate rotation |
| Resolution mismatch | 224 pretrain → 384 fine-tune | ConvNeXt is convolutional; probe at 384; optionally pretrain 256 if time |
| Fine-tune overwrites features | 60 epochs at lr 3e-4 is long | keep identical for the clean test; if B wins, try lr 1.5e-4 or frozen-head warmup next |
| Both-eye false negatives | Fellow eye in the queue as a negative | accepted; do not pseudo-pair |
| DDP host-SHM | P2P disabled, 2.1 GB/s; hangs possible | 100-step `torchrun` smoke with 30 s timeout before long run; single GPU is the default |
| NFS output writes | 114 MB checkpoints every 10 epochs | atomic writes; keep only best/last; write to local `/var/tmp` then copy if NFS is slow |

### 6.2 Failure modes to watch

- **Queue/representation collapse:** loss falls toward 0 or feature rank
  collapses. Abort if `effective_rank < 10` or mean per-dim std `< 1e-3` after
  epoch 20. (With Q=65,536, initial InfoNCE loss is ≈ln Q ≈11.1.)
- **Stale/noisy queue:** loss plateaus high (>10) after 20 epochs; check
  `dequeue_and_enqueue` order and that momentum update uses the online weights
  *after* the optimizer step, not before.
- **AMP skip storm:** monitor `skipped_optimizer_steps`; if >5 % of steps,
  lower lr or disable AMP for the projector (keep encoder AMP).
- **Data loader starvation:** GPU util <70 %; increase `num_workers` to 12 per
  GPU or move the cache to local NVMe; the 384→224 two-view crops are the CPU
  cost.
- **Encoder key mismatch:** `load_state_dict(strict=True)` must fail loudly;
  never silently load a partial encoder.
- **Wrong split:** assert train-only at dataset construction; a valid row must
  raise, not warn.

### 6.3 Stop criteria (hard)

1. **Smoke gate:** 100 steps, finite loss, queue changes, checkpoint→
   `--init-encoder` one-step works. If not fixed in 2 h, fall back to SimCLR or
   drop the task.
2. **Collapse gate:** effective rank <10 or std <1e-3 after epoch 20 → abort.
3. **Probe gate at 100 epochs:** `probe_ssl - probe_random < 0.005` → abandon
   SSL; do not run the 60-epoch fine-tune.
4. **Budget gate:** 20 GPU-h for SSL; 12 GPU-h for SimCLR fallback. Stop at the
   last checkpoint when the cap is reached.
5. **Fine-tune gate:** if the linear-probe gain is <0.01, fine-tune only on an
   otherwise-idle GPU; never delay the submission/ensemble queue for it.
6. **DDP gate:** if multi-GPU hangs once, do not debug >1 h; run single-GPU
   100 epochs.

---

## 7. Implementation checklist (ordered, acceptance-tested)

1. `SSLConfig` + JSON configs; unit tests for validation/round-trip. No GPU.
2. `TwoViewTransform` + `FundusDataset` reuse test (tuple collation, cache,
   train-only assertion). No GPU.
3. `build_encoder`/`MoCoV2` + FIFO/momentum/loss tests. CPU.
4. `pretrain` loop: 1-GPU first, synthetic smoke, then 20 real steps on the
   384 cache; log throughput and compare with the §4 estimate. `[T]`
5. Encoder payload + `--init-encoder` + `initialize_from_encoder` tests, then
   `train --init-encoder` one real epoch (small `--stop-after-epochs`).
6. Random-init frozen probe at 384 on the full train/valid; record `[T]`.
7. Launch the 100-epoch arm on GPU 5. Probe at 50/100; apply the probe gate.
8. If gate passes: launch fine-tune B (seed 42) and, on another GPU, matched
   control C (120 epochs). If no second GPU, run B first (submission value),
   C after.
9. Paired bootstrap + write `.w1mer/results/` with config sha256, encoder
   sha256, `best.pt` sha256, per-run metrics, and the bootstrap JSON.
10. If `B > C` with Δ≥0.010: propose the SSL-init recipe for task 09's
    full-data retrain; otherwise record the negative result and keep the
    baseline.

### 7.1 Verification commands (run on the server, not by the explorer)

```bash
cd ~/work/ncccu
PYTHONPATH=src python -m unittest discover -s tests -v          # CPU suite
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python scripts/smoke_ssl.py
# 20-step real throughput check (fresh output dir; does not touch the 60-ep arms)
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src python -m glaucoma pretrain \
  --config configs/ssl/moco_v2_224_100ep.json --stop-after-epochs 1 \
  --image-cache /var/tmp/ncccu-heyuhang/cache384 --output-root /var/tmp/ssl-smoke
```

Expected one-epoch time from the measured cache: **~5 min** at 224 on one V100;
if the 1-epoch check is >8 min, fix the loader before launching 100 epochs.

---

## 8. Open decisions (already resolved; do not re-litigate)

- **No pretrained weights, no external data, no test images.** The only data
  touched by SSL is `train/`.
- **No patient-pair positives.** Image-level sampling only.
- **No timm, no new dependencies.** torchvision ConvNeXt-Tiny, torch SGD/AMP,
  sklearn logistic regression (already a dependency).
- **No changes to supervised defaults.** `--init-encoder` is opt-in; every
  existing config/checkpoint loads exactly as before.
- **Single writer.** The implementer owns the contiguous commits below; the
  four 60-epoch arms must not be interrupted by this work.
