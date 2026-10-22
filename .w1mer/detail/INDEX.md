# Detail Index (detail/)

> Dynamic layer: recent changes, field-level details, concerns — anything too
> volatile for the stable codebase layer. Each detail gets its own file; this
> INDEX is a build artifact (`w1mer build`).

## Details

| ID | Title | State | Doc |
|----|-------|-------|-----|
<!-- w1mer:rows -->
| 01 | "Superseded — see 03_latency_plan.md" | archived | [01_latency_plan.md](01_latency_plan.md) |
| 03 | "Task 03 plan — latency-safe decoding, decode parity, T4 latency harness" | active | [03_latency_plan.md](03_latency_plan.md) |
| 08 | "Task 08 plan — self-supervised pretraining on the provided train images" | active | [08_ssl_plan.md](08_ssl_plan.md) |
| 09 | "Task 07 plan — ensembling, TTA and score fusion under the limits" | active | [09_task_07_plan_ensembling_tta_and_score_fusion_under_the_limits.md](09_task_07_plan_ensembling_tta_and_score_fusion_under_the_limits.md) |
| "07" | "Task 07 plan — ensembling, TTA and score fusion under the latency/size limits" | active | [07_ensemble_plan.md](07_ensemble_plan.md) |
<!-- /w1mer:rows -->

## Lifecycle

- Created via `w1mer new detail --title "..."`.
- A periodic **compact** promotes settled details into the stable `codebase/`
  docs, then the detail is archived.
- `CHANGES.md` holds recent architecture-impact deltas written by reviewers.
