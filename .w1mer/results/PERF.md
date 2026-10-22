# PERF.md — performance optimization effects

## 03 Latency-safe decode + parity check + latency harness

- Roadmap id: `03` (PERF)
- Commit: see ROADMAP row (single commit `data: add decode policy (full|draft) + parity/latency harness`)
- Date: 2026-10-22
- Environment: itgpu V100-SXM2-16GB (GPU 5), torch 2.5.1+cu121, torchvision 0.20.1, Pillow 13,
  shared CPU at load average 12–19. Data: `/home/lichunyu/competition/dataset/glaucoma_first`.

### What was measured

CPU suite (`scripts/dev_test.sh`, server mirror):

```
Ran 39 tests in 23.274s

OK (skipped=1)
```

The skip is the pre-existing opt-in CUDA smoke. `tests/test_latency_decode.py` adds 9 tests:
draft/no-upscale + JPEG-only, full-vs-draft tensor delta on synthetic smooth images, config and
checkpoint round-trip of the decode policy, resume rejection, dataset/cache plumbing, manifest v3
decode gate + legacy v2-as-full, draft-cache gate, percentile helper, CPU-only benchmark on a tiny
model, and fp16-on-CPU rejection.

Decode micro-benchmark (100 random valid images, 384, `--config b0_crop384_control.json --timing`,
per-image median of 2 repeats, ms):

| stage | full p50/p95 | draft p50/p95 |
|---|---:|---:|
| read | 0.63 / 9.96 | 0.63 / 9.96 |
| decode | 104.30 / 255.81 | 15.49 / 32.79 |
| crop + letterbox | 350.08 / 770.43 | 14.46 / 27.12 |
| decode + preprocess | 454 / 1026 | 30 / 60 |

Without crop (`--no-crop` default): full decode 99.70 / 253.76, draft 15.32 / 32.99;
letterbox-only preprocess full 38.38 / 76.55, draft 2.91 / 4.50.

End-to-end single image (384, fp16, batch=1, in-process, 20 valid images incl. 10 largest by bytes,
`scripts/benchmark_latency.py --decode draft --warmup 10`), ms:

| stage | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| read | 0.66 | 31.87 | 41.21 | 43.54 |
| decode | 24.74 | 34.06 | 34.38 | 34.46 |
| preprocess | 13.59 | 26.37 | 26.39 | 26.40 |
| transfer | 0.83 | 0.92 | 0.99 | 1.01 |
| forward | 7.22 | 7.49 | 7.56 | 7.58 |
| post | 0.07 | 0.10 | 0.11 | 0.11 |
| **total** | **55.28** | **82.15** | **84.87** | **85.55** |

`--decode full` same images: total p50 870.13 / p95 1038.35 (decode 201/250, preprocess 659/778).
Cold start (CUDA context + cuDNN autotune + first forward + first image): draft 2.79 s.

Decode parity (100 random valid, 384, `--config b0_crop384_control.json`): tensor per-image mean
|Δ| p50 0.00364 / p95 **0.01092** / max 0.01649; 6/100 images above 0.010; element-wise |Δ| max
2.0; bbox normalized max delta 0.82% (≤1% gate). Without crop: p50 0.00073 / p95 **0.00312** PASS.
Gate `tensor p95 <= 0.010` therefore **marginal FAIL with black-border crop**, PASS without.
No checkpoint existed on the server, so the AUROC gate (|Δ| ≤ 0.001 on 5,849 valid rows) and the
500 MB artifact gate were not measured.

T4 projection (×2.0 forward only, CPU stages unchanged, per-image): total p50 62.6 ms / p95 89.7 ms;
forward p50 14.4 ms. Not a T4 measurement.

### Conclusion

`significant`: decode + crop preprocessing drops from ~454 ms to ~30 ms (15×) and end-to-end from
870 ms to 55 ms p50 on V100, removing the CPU decode bottleneck. Defaults are unchanged
(`decode="full"`, bitwise-identical existing path; all 30 pre-existing tests pass).

### Leftover risks

- With `crop_black_border=True`, draft's tensor parity is marginal (p95 0.0109 vs 0.010 gate);
  the decision to train/serve draft must wait for the checkpoint AUROC parity measurement.
- T4 projected p95 89.7 ms assumes judge CPU ≈ server CPU; +50% CPU for decode/preprocess gives
  ≈108 ms. Real T4 + judge-host measurement is mandatory.
- Cold start is ~2.8 s; if the judge includes the first image, warmup must happen before timing.
- Pillow `draft` is JPEG-only (PNG/TIFF decode unchanged, bounded by a 4× thumbnail guard in draft
  mode only) and its DCT-scale choice depends on Pillow version (13 here).
- Pre-existing (not fixed): `utils.resolve_device("cuda")` fails on torch 2.5.1 (needs `cuda:0`);
  both scripts normalize the bare device name, the shared helper is unchanged.
