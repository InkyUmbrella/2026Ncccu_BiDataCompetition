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

- [trainer][data][config] R_01 audit (a8ab0ff): confirmed the contracts —
  `train()` resume = config equality (data.root/output_root excepted) + CSV sha256
  + optimizer/scaler/RNG/history/best; `best.pt` inference-only
  (config+age_stats+weights+metrics), `last.pt` recovery-only; cache identity =
  preprocessing spec + image_size + CSV sha256 + per-file size/mtime; loader
  determinism via per-epoch `seed+epoch` generators + `seed_worker`. Gaps recorded:
  checkpoints carry neither `csv_hashes` nor a preprocessing version; experiment
  configs extend 384/512 at a constant 402 updates/epoch and effective batch 128.
