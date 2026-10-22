# FEATURE.md — feature implementation verification

## 02 Trainer upgrades: EMA / stochastic depth / Mixup-CutMix / label smoothing

- Roadmap id: `02` (FEATURE)
- Commit: `2ac17b0` — `trainer: add EMA/stochastic depth/mixup/label smoothing (default-off)`
- Date: 2026-10-22
- Environment: itgpu, `/home/lichunyu/miniconda3/envs/glaucoma` (torch 2.5.1, torchvision 0.20.1), CPU suite + one V100-SXM2-16GB smoke (GPU 0, idle).

### What was measured

CPU suite (`scripts/dev_test.sh`, server mirror `/home/heyuhang/work/ncccu`, branch `heyuhang/w1mer-opt`):

```
Ran 30 tests in 15.655s

OK (skipped=1)
```

The skip is the existing opt-in CUDA smoke (`test_cuda_smoke`). New file `tests/test_upgrades.py` adds 8 tests:

- EMA update math (`shadow = decay*shadow + (1-decay)*weights`), warmup `min(decay, (1+step)/(10+step))`, fp32 shadow, state round-trip, `apply_ema` swap/restore.
- AMP-skipped step: stub scaler that halves its scale after `update()` leaves `global_step` and EMA `num_updates` at 0; a non-skipping stub advances both.
- EMA + Mixup + label smoothing full train, 1-epoch partial run, then resume: `global_step`, `model_state`, EMA shadow and `num_updates` are bit-identical between full and resumed runs; `best.pt` equals the EMA shadow when the best epoch is the last epoch.
- Mixup/CutMix: soft convex targets, exact lambda/permutation formula, deterministic under re-seeded RNG, 50/50 choice when both enabled.
- Soft-target `pos_weight` semantics: `-(w*y*log σ(x) + (1-y)*log(1-σ(x)))` matches manual computation.
- Label smoothing target `y*(1-s)+0.5*s` and the loss reported by `train_one_epoch`.
- Stochastic depth: `stochastic_depth_prob` reaches torchvision `StochasticDepth` blocks with the requested `p`; explicit `0.0` and omitted produce identical state dict; eval-mode outputs identical.
- Defaults no-op: legacy config loads, defaults round-trip, and the default `train_one_epoch` matches a hand-written pre-upgrade accumulation loop bit-for-bit with an identical post-loop CPU RNG state (no extra RNG draws).

GPU smoke (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/smoke_upgrades.py`, synthetic fixture data, ConvNeXt-Tiny 64px, batch 4, accumulation 2, FP16, 2 epochs, EMA 0.9 + warmup 2, mixup 0.2 + cutmix 1.0, label smoothing 0.05, stochastic depth 0.1):

```
[epoch 1/2] train_loss=1.973277 valid_auc=1.000000 valid_ap=1.000000
[epoch 2/2] train_loss=4.837380 valid_auc=1.000000 valid_ap=1.000000
global_step 1 ema_updates 1
max|ema-live| 3.0137598514556885e-05
best epoch 0 best auroc 1.0
peak allocated MiB 657.7
```

`ema_updates == global_step` shows EMA advanced only on the single non-skipped AMP step (the other candidate steps were skipped by GradScaler while the loss scale settled); `max|ema-live| > 0` proves the shadow diverged from the live weights. The valid metrics are meaningless on 4 synthetic images; this is a plumbing check only.

### Conclusion

`significant` for training capability (EMA, stochastic depth, Mixup/CutMix and label smoothing are now available); `none` for default runs, verified bit-identical math and RNG consumption with all options off.

### Leftover risks

- `ema_warmup_steps` only applies when `ema_decay` is set; setting warmup alone is silently ignored (documented).
- Mixup/CutMix draws use the global CPU RNG; a CUDA run and a CPU run with the same seed may permute differently (cross-hardware bit-identity is not promised by this project).
- `pos_weight` under soft targets scales positive mass by `y`; effective positive weighting shrinks with Mixup, so `pos_weight` may need re-tuning when enabling mixing.

## 04 Offline evaluation & fusion toolkit

- Roadmap id: `04` (FEATURE)
- Commit: `3fe2743` — `eval: add fusion, run report, paired bootstrap and prediction fusion tooling`
- Date: 2026-10-23
- Environment: server mirror `/home/heyuhang/work/ncccu`, `/home/lichunyu/miniconda3/envs/glaucoma` (torch 2.5.1, sklearn/numpy), CPU only. GPU 3 was running `full4`; no GPU was used.

### What was measured

Full CPU suite (`scripts/dev_test.sh`, branch `heyuhang/w1mer-opt`):

```
Ran 80 tests in 55.916s

OK (skipped=1)
```

The skip is the pre-existing opt-in CUDA smoke. `tests/test_fusion.py` adds 25 tests covering:
average ranks/ties, standardizer fit/apply/reuse and constant-sample rejection, all three fusion
rules and weight normalisation, name-order/target/missing-name/duplicate errors, `align_by_name`
reordering and official-order override, paired-bootstrap determinism/CI sanity/null centring/
grouped mode, `group_same_age_pairs` adjacency, the report scanner (nested and flat metrics,
predictions-only fallback, malformed JSON), end-to-end offline fuse, rank/prob output columns,
test-mode submission ordering/labels, and the compare-script WIN/grouped paths.

Pre-existing failures repaired in the same commit (not caused by this task):

- `test_ssl.ConfigTests.test_shipped_configs_load`: expected batch 256 for both configs; the
  committed 100ep config is 128 (`12b1006`, 16 GB V100 OOM). The test now asserts 256/128.
- `test_age_residual.TransferTests.test_config_and_initialization_guards`: expected the old
  message `requires init_from or resume`; the guard now mentions `init_encoder` too.
- `test_ssl.ModelTests.test_infonce_loss_decreases_on_separable_views`: compared the last window
  against the *empty-queue* transient, which is not a learning signal (loss rises as the queue
  fills). It now uses queue=8 / lr=0.005 and compares steady-state windows (steps 9–13 vs the
  last 5) after the queue has filled; deterministic and stable across seeds 1,2,3,7,11,42.

Synthetic end-to-end demo (200 rows, 40 positives, `/tmp/demo_fusion`):

```
Wrote 200 fused rows to /tmp/demo_fusion/fused.csv; method=logit; AUROC=0.71296875
Paired bootstrap: image-level, stratified; reps=200 seed=42 rows=200 pos=40 neg=160
metric             delta   95% CI low  95% CI high  P(delta<=0)
ΔAUROC           0.00149     -0.00704      0.00907       0.3450
ΔAP             -0.00569     -0.02390      0.00621       0.7750
ΔAUPRC-trapz    -0.00631     -0.02736      0.00600       0.7850
Verdict: NO-EVIDENCE (requires ΔAUROC >= 0.003 and CI low > 0)
experiment  seed  epoch  auroc  auprc_trapz  ap     sens@s95  wall_s
demo        42    7      0.9000 0.5000       0.5500 0.4000    5
```

The compare demo is intentionally a null pair (two independent models with the same weak signal);
it exercises the NO-EVIDENCE verdict. No real run directory had `metrics.json` at measurement
time (the four 60-epoch arms were still training), so `report_runs.py` was validated against a
synthetic run tree and the tuple/epoch extraction is unit-tested.

### Conclusion

`significant` for tooling readiness: a 10k-replicate stratified paired bootstrap over 5,849 rows
takes seconds, and a fusion can be recomputed from saved CSVs with no GPU. Fusion is not yet
measured on real arms, so no ensemble claim is made. The default shipped rule remains
train-standardised logit averaging; rank fusion is clearly documented as transductive/analysis-only.

### Leftover risks

- The grouped bootstrap is a cluster sensitivity check only; the dataset has no patient IDs, so it
  cannot correct eye-level dependence.
- Without `--reference` train predictions, logit standardisation is fitted on the supplied valid
  file and the JSON report marks it `input_leaky`; do not quote that run as clean.
- `report_runs.py` reads `best_epoch` from `metrics.json`; a still-running arm has none and appears
  only once its final metrics/predictions exist.
- `scripts/fuse_predictions.py` does not run inference; per-checkpoint preprocessing/latency
  checks stay in the CLI/benchmark task.
