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
