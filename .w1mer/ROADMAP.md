# {project} — Roadmap

> Section order in the file = priority order (the template puts performance
> first, backlog last); custom sections work the same way — `list` and
> `re-rank` follow file order.
> Within a section, open rows are ranked by `Defer` (desc), then ID.
> `Defer` counts how many times the task was skipped (`w1mer defer <id>`).
> Each completed task: mark `[x]` + date + commit, add a one-line measured
> effect summary and `done`; full data goes to the results archive.
> Insert new items with derived sub-ids (`05.1`, `05.1.2`); never batch-renumber.

## ID rules

- Unbounded rolling hierarchical IDs: `01`, `05`, `05.1`, `05.1.1`, ...
- Children branch deeper on demand (sub-batch fixes, review follow-ups).
- Ordering: pre-order traversal (parent first, children right behind).
- Cross-references stay stable; derived ids link review/results/bug docs.

## Status states

`todo` `done` `reviewed` `issue`

## 一、Performance

| # | Task | Doc | Status | Defer | Effect |
|---|------|-----|--------|-------|--------|
<!-- w1mer:task:perf -->
| 03 | Latency-safe decode + parity check + T4 latency harness | .w1mer/detail/latency.md | done | 0 | decode policy + parity/latency harness (R_03 follow-ups open) |
| 04 | Eval tooling: run table, paired bootstrap, rank fusion, age stacking | .w1mer/detail/eval_tools.md | todo | 0 | |
| 05 | From-scratch 384: control vs positive sampling (60ep) | .w1mer/detail/scratch384.md | todo | 0 | 4 arms launched 2026-10-22T13:41Z on GPUs 0/1/3/4 (ctrl, ctrl_ema, samp_ema, full); ETA ~10h |
| 06 | From-scratch 512 sampled + EMA/long-schedule follow-up | .w1mer/detail/crop512.md | todo | 0 | |
| 07 | Seed ensemble + TTA with rank fusion | .w1mer/detail/ensemble.md | todo | 0 | |
| 08 | Self-supervised pretraining on provided train images | .w1mer/detail/ssl.md | done | 0 | MoCo-v2 implemented + tests green; smoke pending |
| 09 | Full-data retrain + submission + model-size/latency audit | .w1mer/detail/submission.md | todo | 0 | |







## 二、Bug fixes

| # | Task | Doc | Status | Defer | Effect |
|---|------|-----|--------|-------|--------|
<!-- w1mer:task:bug -->
| 02.1 | Fix stochastic-depth default parity (torchvision 0.1 -> 0.0) and document/resume impact | docs/TRAINER_UPGRADES.md | todo | 0 | |
| 02.2 | Harden upgrade tests: ConvNeXt default-path parity, best.pt EMA on disk, CutMix lambda/area | tests/test_upgrades.py | todo | 0 | |


## 三、Features

| # | Task | Doc | Status | Defer | Effect |
|---|------|-----|--------|-------|--------|
<!-- w1mer:task:feature -->
| 02 | Trainer upgrades: EMA, stochastic depth, mixup/cutmix, label smoothing (default-off) | .w1mer/detail/trainer_upgrades.md | done | 0 | 2026-10-22 2ac17b0: EMA/stochastic depth/mixup/label smoothing added default-off; 30 CPU tests pass, GPU smoke EMA verified |

## 四、Infrastructure

| # | Task | Doc | Status | Defer | Effect |
|---|------|-----|--------|-------|--------|
<!-- w1mer:task:infra -->
| 01 | Own run harness + caches + branch bridge (bootstrap) | docs/W1MER_SETUP.md | done | 0 | bootstrapped: branch + server bare bridge + run harness + own caches building |

## 五、Backlog

(Items awaiting re-prioritization)
<!-- w1mer:task:backlog -->
