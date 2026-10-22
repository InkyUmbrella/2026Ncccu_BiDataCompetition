---
id: CHANGES
title: Recent changes (architecture-impact deltas)
state: active
---

# Recent Changes

> Architecture-impact deltas recorded by reviewers during main batches.
> A periodic compact merges these into the stable `codebase/` docs and clears
> this log. One cache invalidation per compact — keep stable periods long.

## Deltas

> One bullet per delta, tagged with the target module doc:
> `- [gc] contract X moved`. Tags = module names from
> `codebase/INDEX.md`. Untagged → unassigned.
> `w1mer sync` shows a grouped preview; `w1mer sync --apply` writes + clears.

- [trainer][model][config] R_02 task-02 (2ac17b0): new `engine/ema.py`
  (`ModelEMA` fp32 shadow of all floating state; `apply_ema` swap/restore) changes
  the trainer contract: validation/best-selection/`best.pt` run under EMA weights,
  `last.pt` gains `{shadow,num_updates}` while keeping live `model_state` for exact
  resume (full config equality already covers the new `TrainConfig` keys);
  `train_one_epoch` takes an optional `ema` and updates it only on real optimizer
  steps (same GradScaler predicate as `global_step`); training loss now accepts
  label-smoothed/mixed soft targets (BCE with `pos_weight`, semantic caveat when
  `pos_weight!=1`); `model.params.stochastic_depth_prob` forwards to torchvision
  and adds no state keys. **Behavioural delta:** the old call used torchvision's
  SD default 0.1, the new default is 0.0, so old configs/resumes lose SD unless set
  explicitly (review F1, follow-up 02.1).
- [trainer][data][config] R_01 audit (a8ab0ff): confirmed the contracts —
  `train()` resume = config equality (data.root/output_root excepted) + CSV sha256
  + optimizer/scaler/RNG/history/best; `best.pt` inference-only
  (config+age_stats+weights+metrics), `last.pt` recovery-only; cache identity =
  preprocessing spec + image_size + CSV sha256 + per-file size/mtime; loader
  determinism via per-epoch `seed+epoch` generators + `seed_worker`. Gaps recorded:
  checkpoints carry neither `csv_hashes` nor a preprocessing version; experiment
  configs extend 384/512 at a constant 402 updates/epoch and effective batch 128.
