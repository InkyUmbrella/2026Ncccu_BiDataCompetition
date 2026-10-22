---
id: 03
title: "Task 03 plan — latency-safe decoding, decode parity, T4 latency harness"
state: active
task: "03"
base: a8ab0ff50ff07a19dc65c94643f3b38d6f01d5ef
date: 2026-10-22
---

# Task 03 — latency-safe decoding + parity + latency harness

Status: **plan only** (explorer, read-only). Baseline `a8ab0ff`.
Owner of implementation: next implementer. This document is the change contract.

> **Measurement caveat (important for the orchestrator).** The explorer agent
> runs with shell permissions limited to `w1mer`, `git log/diff/show`; remote
> `ssh`/`python` execution is denied by the agent profile, so this document
> could not produce fresh server numbers itself. Numbers below are marked
> `[M]` = measured earlier by the orchestrator, `[E]` = analytical estimate,
> `[T]` = must be measured by the implementer with the scripts specified here.
> The scripts are specified precisely enough to be written and run without
> further design work.

## 0. TL;DR

1. The 100 ms budget cannot be met with the current decode path. Full PIL
   decode is `[M] ~259 ms/img`; `Image.draft` + resize is `[M] ~34 ms/img`
   (7.6×). The model forward is *not* the only bottleneck.
2. Make decode policy **part of `DataConfig`** (`"full" | "draft" | "draft2x"`),
   saved into the checkpoint, used identically by cache build, training,
   evaluation and prediction. This makes train/inference pixel parity a
   property of the checkpoint instead of a CLI flag nobody remembers.
3. Default stays `"full"` in task 03 so the in-flight 60-epoch runs and every
   existing checkpoint keep exactly their current behaviour. Switching the
   canonical policy to `"draft"` is a one-line config change for the task 09
   full-data retrain, gated on the measured parity check below.
4. Ship two scripts: `scripts/check_decode_parity.py` (decode micro-benchmark +
   tensor/prediction parity, no GPU required) and `scripts/benchmark_latency.py`
   (single-image end-to-end harness with p50/p95/p99, `<100 ms` and `<500 MB`
   assertions, `--cpu-only` fallback, explicit V100→T4 projection).
5. Do **not** add cv2/turbojpeg as a dependency for the default path. Optional,
   import-guarded backends may be measured, but deployment must stay on
   Pillow-only so the judge environment (PyTorch 2.10/CUDA 12.8, no guarantee
   of OpenCV) reproduces our numbers.

Expected value: decode drops from ~259 ms to ~34 ms, leaving ~60–70 ms of the
T4 budget for preprocess + forward at 384. It also creates the hard latency
gate that task 07 (ensemble/TTA) must pass, and removes a silent
train/inference skew risk before the final submission.

---

## 1. Code findings (baseline `a8ab0ff`)

### 1.1 Current inference path (what the judge would time)

`cli.py::_inference` → `read_samples` → `FundusDataset` with
`build_transform(image_size, training=False, crop_black_border=...)` →
`make_loader(batch_size=16, num_workers=4)` →
`engine/evaluation.py::collect_predictions`.

Per image:

| stage | code | note |
|---|---|---|
| file open + full JPEG decode + EXIF + RGB | `data/dataset.py::load_rgb` (L58-62) | `Image.open`, `exif_transpose`, `convert("RGB")`, `load()` — full resolution |
| black-border crop | `data/transforms.py::CropBlackBorder` (L21-41) | `np.asarray(image)` on the **full-res** array, max-channel threshold 10, margin 3 % of the longer bbox side |
| letterbox to square | `transforms.py::Letterbox` (L6-18) | `Image.resize(..., BILINEAR)` then paste on black canvas |
| tensor + normalize | `build_transform` (L56-64) | `ToTensor`, mean=std=0.5 |
| batch, H2D, forward | `collect_predictions` (L15-24) | `torch.autocast` only when `config.train.amp`; batch 16 at submission time |
| sigmoid + CSV | `write_predictions` | `Probability`, `Pred_Label = p >= 0.5` |

Latency-relevant facts:

- `load_rgb` decodes *before* the crop, so a 4928×3264 image allocates a
  48 MB RGB array and pays full IDCT. This is the 259 ms `[M]`.
- `CropBlackBorder.bbox` is computed on the decoded array; a fast decode
  changes both pixel values and (slightly) the bbox. This is the parity risk
  that must be measured, not assumed.
- The deployment path uses `DataLoader` with 4 workers and batch 16. That is
  correct for a 5,855-row submission but wrong for single-image latency
  (worker startup + IPC + batching dominate). A separate in-process
  single-image harness is required.
- `predict` restores `config` from the checkpoint (`engine/checkpoint.py::
  load_model`), so any decode field added to `DataConfig` is automatically
  honoured at inference. This is the cheapest possible parity guarantee.

### 1.2 Cache path (training I/O)

`data/cache.py::prepare_cache` stores the **already letterboxed uint8** image
at `image_size` (`.npy`), manifest version 2, identity =
`preprocessing_spec(crop_black_border, crop_threshold, crop_margin)`.
`ImageCache.load` returns `Image.fromarray(pixels)`, and
`FundusDataset.__getitem__` then applies `build_transform(preprocessed=True)`.

Cache is the only place the running jobs read images. It was built from
**full** decode. Consequence: changing `load_rgb` today does **not** change the
running jobs' inputs, but a future draft-decoded cache would. The manifest
must therefore carry the decode policy (version 3) or the cache can silently
mix decode policies.

`tests/test_baseline.py::test_lossless_cache_preserves_augmentation_and_rejects_stale_sources`
asserts cache vs original tensors are **bitwise equal**. As long as both sides
use the same decode policy this test keeps passing; it is also the guard that
`decode="full"` remains the exact current behaviour.

### 1.3 Constraints recorded elsewhere

- `docs/BASELINE_DESIGN.md` L34, L114: T4 end-to-end (decode included) ≤100 ms;
  model files pre-compression ≤500 MB; V100 results do not substitute for T4.
- `docs/TRAINING_SPEED.md`: with 8 workers, full decode gave 25.15 img/s and a
  local lossless cache 81.05 img/s. Sanity check: 8 workers / 0.259 s ≈ 30.9
  img/s, consistent with the measured 25 img/s — the decode number is credible.
- Current server env per task brief: PyTorch 2.10, CUDA 12.8, Python env
  `/home/lichunyu/miniconda3/envs/glaucoma/bin/python`. `docs/SIX_GPU_FEASIBILITY.md`
  describes an older 2.5.1/cu121 environment; do not trust its versions,
  only the V100/no-NVLink hardware facts.

---

## 2. Numbers: measured, estimated, to-measure

### 2.1 Decode strategies

`[M]` orchestrator, single-image wall time on the shared server CPU, including
resize to the model input (exact protocol of the 259 ms number was: full
decode + `exif_transpose` + resize; the 34 ms number: `Image.draft("RGB", ...)`
+ resize).

| strategy | p50 | p95 | source | note |
|---|---:|---:|---|---|
| PIL full decode + EXIF + RGB | **~259 ms** | `[T]` | `[M]` | current `load_rgb` |
| PIL `draft` + resize | **~34 ms** | `[T]` | `[M]` | 7.6× faster |
| PIL `draft2x` (cap 2×size) | — | — | `[T]` | ~4× the pixels of `draft` for very large images; **no-op** for images where the short side < 2×cap |
| cv2 `IMREAD_REDUCED_COLOR_2` | — | — | `[T]` | 1/2 scale only; ~4× more pixels than PIL 1/8; optional, not a dependency |
| cv2 `IMREAD_REDUCED_COLOR_4` | — | — | `[T]` | closer to PIL draft |
| turbojpeg scaling 1/8 | — | — | `[T]` (if importable) | likely fastest; native dep, **not** allowed as default |

Why `draft` is the right default: Pillow's JPEG `draft(mode, (s,s))` chooses
the largest DCT scale `1/2/4/8` while keeping the decoded image **at least**
`s` in each dimension (it doubles the scale while `w/scale > 2*s` and
`h/scale > 2*s`). Therefore `draft("RGB", (image_size, image_size))`
guarantees decoded short side ≥ `image_size`, i.e. the letterbox never has to
upscale, and the IDCT work for a 16 MP image drops by 64×.

Examples (decoded size / scale chosen):

| source | `draft(384)` | `draft2x` = `draft(768)` |
|---|---|---|
| 4928×3264 (16 MP) | 616×408, scale 8 | 1232×816, scale 4 |
| 3000×2000 | 750×500, scale 4 | 1500×1000, scale 2 |
| 2000×1500 | 1000×750, scale 2 | 2000×1500, **scale 1 (no saving)** |
| ≤768 on both sides | unchanged | unchanged |

Consequence: `draft2x` is a quality reference, **not** a speed option. For
mid-size images it is nearly full decode. Deployment candidate is `draft`
with `s = image_size`; `draft2x` exists to quantify how much of the parity
delta comes from JPEG DCT scaling.

`draft` only affects JPEG. PNG/TIFF decode unchanged (no-op); the implementation
must guard with `if image.format == "JPEG"` and still work for non-JPEG inputs.

### 2.2 Model forward

`[E]` because no T4 is available and the explorer could not run the V100
measurement. ConvNeXt-Tiny: 28.6 M params, ~4.46 GFLOPs @224.

| input | GFLOPs | V100 fp16 batch-1 `[E]` | T4 fp16 batch-1 `[E]` |
|---|---:|---:|---:|
| 384 | ~13.1 | 12–15 ms | **20–33 ms** |
| 512 | ~23.3 | 21–26 ms | **36–57 ms** |

Basis: batch-1 convnets achieve roughly 8–15 % of tensor-core peak due to
launch overhead, LayerNorm/GELU and depthwise-conv memory traffic. T4 fp16
peak is ~65 TFLOPS vs V100 ~125 TFLOPS, and Turing is less efficient per
FLOP; empirical batch-1 scaling is ~1.7–2.2×. **This is explicitly an
estimate, not a T4 measurement.** The implementer must replace `[E]` with the
measured V100 numbers from `benchmark_latency.py`; the script prints the
2.0× projection and is the screening gate only.

### 2.3 Projected end-to-end budget at 384 (T4)

| stage | p50 estimate | note |
|---|---:|---|
| file read (warm page cache) | ~2–5 ms | judge storage unknown; measure separately |
| decode `draft(384)` | ~34 ms `[M]` server | judge CPU could be ±50 % |
| crop + letterbox | ~3–6 ms `[T]` | on ≤768 px array, numpy + PIL |
| uint8 HWC→GPU + fp16 normalize | ~1–2 ms | 0.44 MB H2D |
| forward 384 fp16 | 20–33 ms `[E]` | dominant risk |
| sigmoid + CSV row (in-process) | <1 ms | not disk I/O |
| **total** | **~60–78 ms** | 22–40 ms slack under 100 ms |

At 512 the same arithmetic gives ~80–105 ms — **too close to fail**. Task 06
must not ship 512 unless the measured forward on T4 is much better than the
estimate, or a distilled/smaller model is used.

### 2.4 Model artifact size

| artifact | size | note |
|---|---:|---|
| ConvNeXt-T fp32 `state_dict` | ~114.5 MB | 28.6 M × 4 B |
| `best.pt` (state + config + metrics) | ~115 MB | deployable, <500 MB |
| `last.pt` (optimizer + RNG + history) | ~350–400 MB | **must never be submitted** |
| 4-model fp32 ensemble only | ~458 MB | size fits, **latency does not** |

Ensemble/TTA implication for task 07: at ~20–33 ms per extra forward, a second
model takes the total to ~80–111 ms. A 2-model ensemble is only acceptable if
the measured T4 single forward is ≤20 ms; TTA (2–4 views) is almost certainly
out of budget at 384. Task 07 must gate on `benchmark_latency.py`, not assume.

---

## 3. Parity: what to measure and the acceptance gate

Parity is between the **deployment config** (crop_black_border=True,
image_size=384) using full decode and the same config using `draft`/`draft2x`.
No checkpoint exists at planning time; the script must support both modes:
tensor-only now, prediction delta as soon as `outputs/runs/*/best.pt` exists.

Per image, `scripts/check_decode_parity.py` computes:

1. decoded dimensions and chosen JPEG scale (from `image.size` after draft);
2. black-border bbox from each decode path (refactor `CropBlackBorder` to
   expose `bbox(image)`; behaviour unchanged);
3. uint8 letterboxed images → torch tensors (`ToTensor` + `(x-0.5)/0.5`);
4. tensor delta: per-image mean |Δ|, RMSE, max |Δ|, fraction of elements with
   |Δ| > 0.01, and p50/p95/p99 of per-image mean |Δ|;
5. bbox delta: |Δleft|, |Δtop|, |Δright|, |Δbottom| in decoded pixels and as
   a fraction of the decoded side;
6. if `--checkpoint` is given: run both tensors through the same model on GPU
   (or CPU with `--device cpu`), report per-image |Δlogit|, |Δp| (p50/p95/p99/
   max), Spearman rank correlation of probabilities, and AUROC on the full
   split (`--samples 0` = all valid rows; a 100-image subset AUROC is too noisy
   for the decision).

Proposed acceptance gate for switching the canonical policy to `draft`:

| metric | gate |
|---|---|
| per-image mean abs tensor Δ (normalized) | p95 ≤ 0.010, max ≤ 0.020 |
| element-wise |Δ| | p99 ≤ 0.05, max ≤ 0.30 (saturated edges) |
| bbox mismatch > 1 % of side | ≤ 2 % of images, and bbox area change ≤ 1 % |
| AUROC (full valid, 5,849 rows) | \|Δ\| ≤ 0.001 absolute (primary) |
| per-image probability Δ | p99 ≤ 0.02, max ≤ 0.05 |
| Spearman ρ | ≥ 0.9995 |

If only the tensor gate fails but AUROC passes, prefer `draft2x`; if AUROC
fails, `draft` must be trained, not inferred (Section 4).

Rationale for the thresholds: normalized 0.010 ≈ 1.3/255 gray levels mean
error. DCT-scaled decode is a mild low-pass plus chroma-resampling difference;
after bilinear letterbox to 384 this is normally below 1 gray level. If it is
not, the model sees a distribution shift and we must not guess.

---

## 4. Recommended design and migration

### 4.1 Canonical decode policy in config (recommended)

Add to `DataConfig`:

```python
decode: str = "full"          # "full" | "draft" | "draft2x"
```

- Validated in `Config.validate` (exact strings).
- Written into `config.json`, `metadata.json`, the cache manifest, and the
  inference checkpoint (already automatic via `inference_payload`).
- `predict`/`evaluate` therefore reproduce **exactly** the training decode
  policy without any flag; a checkpoint can never be silently evaluated with a
  different decoder.
- Old checkpoints load with the default `"full"`, and `_resume_compatible`
  keeps working for the in-flight runs (both sides load the new default).

Why not a CLI-only flag: `predict` already has no decode argument and the
submission process will be run by someone else; a flag is exactly the kind of
thing that drifts. Why not change the global default now: the running jobs were
started under full-decode cache; changing `load_rgb` would affect future
predictions but not the cache, creating a silent skew. Task 03 should land the
capability with `"full"` default and no behaviour change.

### 4.2 Loader API

```python
# data/dataset.py
def load_rgb(path, decode="full", image_size=384):
    with Image.open(path) as image:
        if decode != "full" and image.format == "JPEG":
            cap = image_size * (2 if decode == "draft2x" else 1)
            image.draft("RGB", (cap, cap))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image
```

Notes:

- `draft` must be called **before** `exif_transpose`: transpose may trigger
  `load()`, after which `draft` is a no-op. Square cap is orientation-agnostic.
- `draft` can raise `ValueError` for unusual JPEG modes; catch `(ValueError,
  OSError, AttributeError)` and fall back to full decode (record a counter in
  diagnostics; never fail an image because of the optimisation).
- Do not change `load_rgb(path)` default semantics for `audit.py`; audits
  should keep validating the full decode path.
- `FundusDataset` gets `decode` and `image_size` (or a bound `image_loader`)
  and passes them through. `trainer.py` builds both `ImageCache` and
  `FundusDataset` with `config.data.decode`.

### 4.3 Cache identity (manifest v3)

`preprocessing_spec` gains `decode`. Manifest version 3:

```json
{"version": 3, "decode": "draft", "preprocessing": {...}, ...}
```

Rules: version 1/2 manifests are read as `decode="full"`; a mismatch between
the manifest and `config.data.decode` raises
`"Image cache decode mismatch; rebuild from original images"`. Cache size is
unchanged (~24 GiB at 384). A draft cache must be rebuilt from source images,
not converted from the old cache (the old cache is already letterboxed and
would bake full-decode pixels into a "draft" cache).

### 4.4 Migration plan for the in-flight training

1. **Task 03 (now):** land capability only, default `"full"`. Running jobs,
   their caches, `resume`, and all current checkpoints are untouched.
   Verify with a fixed checkpoint that `decode="full"` logits are bitwise
   identical before/after the change.
2. **As soon as any `best.pt` exists:** run
   `check_decode_parity.py --samples 0 --decode draft --checkpoint <best.pt>`.
   This costs one valid-set pass (~2–4 min GPU) and answers the question
   empirically. No retraining is needed to *measure*.
3. **Decision:**
   - Gate passes → task 09 (full-data retrain) sets `"decode": "draft"` in its
     config, rebuilds the cache with `--decode draft`, trains once, and the
     checkpoint itself forces matching inference. This is the target state:
     train and inference use identical pixels.
   - Gate fails on AUROC but passes tensor-wise → same as above but fine-tune
     from the final checkpoint for 2–3 epochs with the draft cache
     (`--init-from best.pt`, fresh run dir, low LR), then re-run parity.
   - Gate fails badly → keep full decode and buy latency elsewhere
     (lower input size, distillation); do **not** ship inference-only draft.
4. **Do not switch the decode policy mid-run.** Besides the resume-config
   rejection, it would change the input distribution without validation and
   contaminate the B0/B1 comparison. The cost of doing it at task 09 is one
   cache rebuild (~1 h with 8 workers, 24 GiB) plus zero extra epochs in the
   passing case.

Inference-only fast decode is the fallback only if a checkpoint exists and the
parity gate passes; it must be recorded in the submission notes as
"trained on full decode, inference draft decode" with the measured AUROC delta.

---

## 5. `scripts/benchmark_latency.py` — specification

Purpose: the single authoritative end-to-end single-image latency harness, run
on an allocated V100 as a screening gate and re-run on T4 for acceptance.

### CLI

```
--checkpoint PATH        best.pt (config + age stats + weights). If omitted,
                         build a random model from --config (forward timing only;
                         parity/accuracy is meaningless then).
--config PATH            used only without --checkpoint (defaults to
                         configs/experiments/b0_crop384_control.json)
--data-root PATH         dataset root (test/valid images)
--split {valid,test}     default valid
--images N               default 200
--include-largest K      always include the K largest files (default 10)
--image-size S           default: config.data.image_size
--decode {full,draft,draft2x}   default: config.data.decode (override prints a warning)
--device cuda|cpu        default cuda
--cpu-only               shorthand for --device cpu --fp32 --no-enforce
--precision {fp16,fp32}  default fp16 (CUDA); fp32 on CPU
--warmup N               default 20 (excluded from timing)
--preload                decode/read all images first; times compute only
--max-latency-ms 100     assertion threshold on p95
--max-model-mb 500       assertion on the on-disk artifact size
--output PATH            JSON report; also prints a human table
--enforce / --no-enforce exit non-zero on gate failure (default enforce on cuda,
                         off on cpu)
--seed 42
```

### Timed loop (exactly the deployment path, batch=1, no DataLoader)

For each image, with `time.perf_counter_ns()` and `torch.cuda.synchronize()`
around GPU work:

1. `read` bytes from disk (measure separately; optional `--preload` excludes it);
2. decode (`load_rgb` with the configured policy) — stage `decode`;
3. `CropBlackBorder` + `Letterbox` (`build_preprocess`) — stage `preprocess`;
4. `np.asarray` uint8 HWC → `torch.from_numpy` → `.to(device)` →
   `permute(2,0,1).unsqueeze(0)` → fp16 normalize `(x/255 - 0.5)/0.5` on GPU —
   stage `transfer` (do **not** use `ToTensor`+`Normalize` on CPU in the timed
   path; report a CPU-normalize variant only as a comparison);
5. `model.eval()`, `model.half()`, `torch.inference_mode()`,
   `logits = model(x, age)` — stage `forward`;
6. `sigmoid`, `.item()`, format the CSV row in memory — stage `post`;
7. total = sum of stages (one row per image).

Also measure and report a `--model-only` mode (precomputed tensor reused) so
the forward number can be compared with the estimate table.

Warmup must include: CUDA context init, cuDNN autotune (see §7), first model
call, and one image through the whole pipeline. Report cold-start time
separately (one measurement before warmup) because the judge may include it.

### Report

```
GPU: Tesla V100-SXM2-16GB, torch 2.10.x, cuDNN ..., fp16
model: <path>, params 28.6 M, artifact 114.8 MB, input 384x384, decode draft
images: 200 (largest included: 4928x3264)

stage            p50     p95     p99     max
read             ...     ...     ...     ...
decode           ...     ...     ...     ...
preprocess       ...     ...     ...     ...
transfer         ...     ...     ...     ...
forward          ...     ...     ...     ...
post             ...     ...     ...     ...
TOTAL            ...     ...     ...     ...

projected T4 (2.0x forward+total, estimate): p50 ... p95 ...
GATE latency  p95 <= 100 ms: PASS/FAIL
GATE model    artifact <= 500 MB: PASS/FAIL
```

JSON: identical data plus per-image rows for later analysis, config, versions,
GPU name, `torch.backends.cudnn` flags, and the assertion results.

Percentiles: nearest-rank or linear interpolation, state which; use the same
function in tests.

### How to interpret on T4

- V100 is **screening only**. Rule of thumb (state it in the report and the
  docs): T4 ≈ 1.7–2.2× V100 batch-1 fp16 latency for ConvNeXt-class models.
  Screening gate on V100: total p95 ≤ 50 ms and forward p95 ≤ 25 ms, which
  projects to ≤100 ms on T4 at 2.0×.
- The projection is a prior, not a measurement. Final acceptance **must** be
  a real T4 run of the same script; the report must say so.
- If the judge's harness excludes decode, the forward-only line is the relevant
  number; keep both in the report.
- The server V100 host CPU differs from the judge host; decode is CPU-bound, so
  compare decode p50 to the 34 ms `[M]` server number and treat ±50 % as
  possible on unknown 8 vCPU hardware. Do not over-tune to the server CPU.

---

## 6. `scripts/check_decode_parity.py` — specification

One script covers the decode micro-benchmark (task item 2) and the parity
quantification (task item 3), because both need the same decode paths.

```
--data-root PATH --split valid --samples N (0 = all)
--image-size 384
--crop-black-border / --no-crop-black-border  (default: from --config if given, else true)
--config PATH --checkpoint PATH
--decode-compare draft,draft2x  (policies compared against full)
--timing            run the decode strategy micro-benchmark
--repeats 3         per-image timings, report median
--device cuda|cpu   for the optional checkpoint pass
--output PATH       JSON
--seed 42
```

Sampling: deterministic RNG seed, plus force-include the 10 largest files by
size; report the sampled size distribution and max dimensions.

Timing table: PIL full, PIL `draft`, PIL `draft2x`, cv2 REDUCED_2/4 (if
importable), turbojpeg 1/8 (if importable). Report file-read, decode, and
decode+preprocess separately, p50/p95. For each strategy report the chosen
scale histogram and the count of no-ops. These numbers replace every `[T]` in
§2.1.

Parity table: as defined in §3, per policy, with the gate results in JSON.
This is a **read-only** analysis script; it must not write caches or repo files.

---

## 7. Deployment risks in the current predict path (and mitigations)

1. **Decode is CPU-bound and dominates.** Mitigated by `draft`. Do not measure
   on a warm page cache only and call it end-to-end; report read separately.
2. **`exif_transpose` after `draft`** is fine; transpose before draft would
   force full decode. Document the order in the loader docstring.
3. **`draft` is JPEG-only.** Guard `image.format == "JPEG"` and fall back
   silently for PNG/TIFF. Add a pixel-bomb guard: after draft, if the decoded
   side is still >4× the model input (non-JPEG huge image), call
   `image.thumbnail((4*S, 4*S), BILINEAR)` after `load()` to bound memory.
4. **DataLoader in the deploy path.** `predict` with `num_workers=4` and batch
   16 is right for a 5,855-row submission but adds worker startup/IPC to a
   single-image measurement. The harness must be in-process; if the judge runs
   our CLI per image, add a `predict-one` path (or make `--num-workers 0`
   documented for single-image). Do not change submission batching.
5. **CUDA/cuDNN cold start.** First forward pays context creation and cuDNN
   autotune. Report `cold_start_ms`; if the judge times the first image and the
   budget includes it, warm up at process start (allowed; it is a fixed setup
   cost, but confirm the rules) or at least pre-create the CUDA context.
   `torch.backends.cudnn.benchmark = True` is safe for a fixed 384 input and
   can help batch 1; measure with and without.
6. **fp16 numerics.** `model.half()` is faster than `autocast` at batch 1 but
   changes logits slightly. Report `max |Δlogit|` fp32 vs fp16 on 200 valid
   images; gate at 1e-3 (probability 2.5e-4) and verify AUROC unchanged.
   Always `model.eval()`; never leave dropout active.
7. **`last.pt` is ~3× `best.pt`.** Only `best.pt` may be counted against the
   500 MB limit. Add an explicit artifact check and a submission note listing
   exactly the files shipped.
8. **Ensembles/TTA multiply the forward.** See §2.4. Task 07 must run the
   harness for the actual ensemble; the size limit alone is misleading.
9. **`torch.compile` / TensorRT.** Out of scope. `torch.compile` has a
   10–60 s first-call compile; if the judge includes cold start, it is fatal.
   TensorRT engines are hardware-specific and cannot be built on the judge
   reliably. Do not enable by default.
10. **Thread oversubscription.** `OMP_NUM_THREADS`/`torch.set_num_threads`
    affect CPU normalize and PIL; the harness should fix `OMP_NUM_THREADS=4`
    and document it. GPU normalize avoids the issue in the default path.
11. **CSV writing.** `write_predictions` opens exclusive-create and writes per
    row; in-process formatting is <1 ms/image. Do not fsync.
12. **`read_samples` directory scan** (5,855 `iterdir` + CSV) costs tens of ms
    and must not be inside a per-image timing loop if the judge calls us per
    image; the harness reads the manifest once.
13. **Judge input distribution.** Unknown resolutions. Always report the
    worst-case largest image, not just the mean; the 16 MP case is the budget
    driver.
14. **Pillow version drift.** `draft` semantics have been stable, but pin
    Pillow in the submission metadata and record the version in the harness
    report. If the judge's Pillow differs, decode timing changes, not math.

Pre-existing observations (report to orchestrator, not part of this plan):
`ROADMAP.md` points task 03 at `.w1mer/detail/latency.md` while this plan is
`.w1mer/detail/03_latency_plan.md`; a stray `.w1mer/detail/01_latency_plan.md`
was auto-created by `w1mer new detail` before the requested path was written.

---

## 8. Exact change surface

| file | change |
|---|---|
| `src/glaucoma/config.py` | `DataConfig.decode: str = "full"`; validate it in `Config.validate` |
| `src/glaucoma/data/dataset.py` | `load_rgb(path, decode="full", image_size=384)` with draft-then-transpose, JPEG guard, fallback; `FundusDataset(..., decode, image_size)` and pass-through |
| `src/glaucoma/data/transforms.py` | `preprocessing_spec(..., decode="full")`; extract `CropBlackBorder.bbox(image)` (behaviour unchanged); no other transform change |
| `src/glaucoma/data/cache.py` | `prepare_cache(..., decode="full")`; `_cache_one` task carries decode; manifest v3 with decode; `ImageCache(..., decode="full")` validates; v1/v2 ⇒ full |
| `src/glaucoma/data/audit.py` | no functional change; keep full decode for auditing |
| `src/glaucoma/engine/trainer.py` | pass `decode`/`image_size` to `ImageCache` and `FundusDataset`; add `decode` to `metadata.json` |
| `src/glaucoma/cli.py` | `cache-images --decode`; `train --decode` optional override (record in metadata, still validated by config); `evaluate`/`predict` inherit the checkpoint config, add `--decode` override only for parity experiments, print a warning |
| `src/glaucoma/engine/evaluation.py` | no change (keep `inference_mode`, sigmoid) |
| `scripts/check_decode_parity.py` | new (spec §6) |
| `scripts/benchmark_latency.py` | new (spec §5) |
| `tests/test_latency_decode.py` | new (below) |
| `docs/LATENCY.md` | new: measured decode table, T4 procedure, budget, ensemble gate; link from README |
| `README.md` | one-line link to `docs/LATENCY.md` and the `--decode` flag |

Tests (CPU, synthetic images only, no network/downloads):

- `draft` on a synthetic 1200×800 JPEG: result is RGB, both sides in
  `[image_size, 2*image_size]`, cached `image.format` path works; PNG input
  unchanged (no-op).
- full vs draft on a smooth large JPEG: per-image mean tensor Δ ≤ 0.02
  (loose, non-flaky); assert the script's percentile helper.
- config roundtrip: old payload without `decode` ⇒ `"full"`; invalid value
  rejected; resume with a changed decode rejected.
- cache: manifest records decode; `ImageCache` with the wrong decode raises;
  v2 manifest still loads as full; existing bitwise cache-vs-original test
  still passes with `decode="full"`.
- CLI: `cache-images --decode draft` writes manifest v3; `predict` with a
  checkpoint whose config has `decode="draft"` uses draft (assert via a
  monkeypatched loader or a decode counter in the script, not by re-reading
  pixels).
- `benchmark_latency.py` helper test: percentile function and argument
  validation only (do not run ConvNeXt on CPU in unit tests).

No new runtime dependency.

---

## 9. Acceptance criteria for task 03

1. `decode="full"` is byte-for-byte the current behaviour: the existing test
   suite passes unchanged, and a fixed checkpoint produces bitwise-identical
   logits on a fixed image before/after the change.
2. `check_decode_parity.py` runs on ≥100 valid images (or all valid with a
   checkpoint) and writes the JSON defined in §3, including bbox deltas,
   tensor deltas and (with a checkpoint) AUROC/probability deltas.
3. `benchmark_latency.py --cpu-only --images 3` runs to completion and prints
   the percentile table (used as an environment smoke test).
4. On an idle V100 of ours, `benchmark_latency.py` with `--decode draft`
   completes in <3 min and reports total p95 ≤ 50 ms at 384 (screening gate
   for the 2× T4 projection); the report includes the projection and the
   explicit "not a T4 measurement" statement.
5. The artifact-size check prints PASS for `best.pt` and FAIL for a bundle
   containing `last.pt` (negative test in the harness or docs).
6. `docs/LATENCY.md` contains the measured decode table (`[T]` rows filled),
   the T4 procedure, the end-to-end budget, and the ensemble/TTA gate.
7. No changes to training numerics when `decode="full"`; no new dependencies.

---

## 10. Open questions (for orchestrator / implementer)

1. **Which checkpoint to use for the first parity run?** Whichever of
   `outputs/runs/*/best.pt` is complete first; the script accepts any. If none
   is complete when task 03 is implemented, ship tensor-only parity and attach
   the checkpoint pass to task 09's acceptance.
2. **Judge harness semantics:** does the 100 ms include image file read,
   EXIF handling and CSV write? The plan assumes yes and reports all stages
   separately; if the judge feeds preprocessed tensors, only the forward line
   matters. Confirm from the official rules if possible.
3. **T4 access:** is a real T4 available anywhere for final acceptance? If
   not, the V100×2 projection must be labelled as unverified in the submission
   notes and the 384 config chosen conservatively.
4. **Task 06 (512):** keep 512 as an accuracy experiment, but it may not be
   deployable. Decide only after the measured 512 forward on T4 (or the
   projected number with the 2× caveat).
5. **Task 07 (ensemble/TTA):** needs a latency budget decision now: at most one
   extra forward at 384, or smaller input for ensemble members. Gate on the
   harness, not on the 500 MB size limit.
6. **Decoder backend:** if Pillow `draft` parity passes but the judge CPU is
   much slower than expected, the only remaining lever is a native decoder
   (cv2/turbojpeg), which adds an environment dependency. Decide only if the
   measured T4 total p95 is >100 ms with `draft`.

---

## 11. Ready-to-run measurement commands (for the implementer)

```bash
# on the remote clone (pull first; the branch may lag by a few commits)
ssh -i /home/niko/.ssh/id_ed25519_xianka -o StrictHostKeyChecking=no \
    heyuhang@100.68.97.6
cd /home/heyuhang/work/ncccu && git pull -q origin heyuhang/w1mer-opt
PY=/home/lichunyu/miniconda3/envs/glaucoma/bin/python
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv

# 1) decode micro-benchmark + tensor parity (CPU only, no GPU, <3 min)
PYTHONPATH=src $PY scripts/check_decode_parity.py \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  --split valid --samples 100 --image-size 384 \
  --timing --output /tmp/w1mer_decode_parity.json

# 2) prediction parity once a checkpoint exists (idle GPU from {0,1,3,4,5})
CKPT=$(ls -t outputs/runs/*/best.pt | head -1)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src $PY scripts/check_decode_parity.py \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  --split valid --samples 0 --image-size 384 --decode-compare draft \
  --checkpoint "$CKPT" --device cuda --output /tmp/w1mer_parity_ckpt.json

# 3) end-to-end latency, 384 and 512 (each <3 min, one idle GPU)
for S in 384 512; do
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src $PY scripts/benchmark_latency.py \
    --checkpoint "$CKPT" --data-root /home/lichunyu/competition/dataset/glaucoma_first \
    --split valid --images 200 --include-largest 10 --image-size $S \
    --decode draft --warmup 20 --output /tmp/w1mer_lat_$S.json --enforce
done
```

Use `/tmp` on the remote; do not write under `/home/lichunyu` or
`/home/zhenxiaojun`; keep GPU runs under ~3 min and check `nvidia-smi` first.
