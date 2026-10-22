---
id: "07"
title: "Task 07 plan — ensembling, TTA and score fusion under the latency/size limits"
state: active
task: "07"
base: 9a2da75a4d858359b6073d7037d225b5ba892b1c
date: 2026-10-22
---

# Task 07 — ensembling, TTA and score fusion under the limits

Status: **plan only** (explorer, read-only). Baseline `9a2da75`.
Owner of implementation: next implementer. This document is the change contract.

> Numbers marked `[M]` are measured and live in `docs/LATENCY.md` /
> `.w1mer/results/PERF.md`. Numbers marked `[E]` are analytical projections from
> those measurements. Numbers marked `[T]` must be measured by the implementer on
> the exact configuration before it is shipped. No T4 exists in this repo's
> environment at planning time; the final gate is a real T4 run.

## 0. TL;DR — the decision

1. **Ship a weight-space soup, not a prediction ensemble.** Average the weights
   of two 384 arms into one checkpoint (`full_recipe` + `sampled_ema`), evaluate
   it, and ship it if it passes the accuracy gate. A soup costs **one forward**,
   **one 57 MB fp16 file** and therefore keeps the current latency budget.
   Prediction-level 2-model ensembles cost a second forward and **project to
   ~105 ms p95 on T4**, i.e. they fail the 100 ms limit unless the real T4
   measurement is much better than the V100×2 projection.
2. **Default fusion rule is equal-weight, train-standardised logit averaging.**
   It is pointwise (works for a single image), calibration-free for AUROC, and
   needs no valid-label fitting. Rank averaging is kept as an offline
   analysis/sensitivity check only; it is transductive and cannot be used by a
   single-image latency path or by a per-image judge that re-runs inference.
3. **TTA default is none.** The only predeclared view is horizontal flip, which
   is in-distribution (training already uses `RandomHorizontalFlip(0.5)`).
   Batched hflip projects to **97–105 ms p95 on T4**; ship it only if the
   measured T4 p95 for the exact configuration is ≤90 ms. Rotations/multi-scale
   are exploratory, not shipped by default.
4. **512 is not deployable and must not enter the shipped ensemble.** A 512
   forward alone projects to ~101–109 ms p95. 512 remains a valid-only
   diversity/teacher experiment and a distillation target.
5. **Age fusion is default-off.** The prior is +0.002 AUROC with a CI touching
   0. The plan specifies the exact OOF/grouped stacker and a strict gate; do not
   ship age unless the gate passes. Age is free at inference, so it is worth one
   cheap valid-CV check, not five retrainings.
6. **Recommended shipped configuration (expected):** single 384 fp16 model =
   soup of `full_recipe` + `sampled_ema`, no TTA, no age fusion, `decode=draft`
   only after task 09's parity gate. Projected: **AUROC +0.001…+0.004 over the
   best single; ~89.6 ms p95 T4; 57.3 MB**. If the soup fails its accuracy gate,
   ship the best single model unchanged. If the soup passes but T4 measurement
   fails, ship the best single; do not gamble the submission on a projection.

The rest of this document is the exact contract for getting there.

---

## 1. What the baseline actually contains (and does not)

Read at `9a2da75`:

- `src/glaucoma/cli.py::_inference` accepts **one** `--checkpoint`. There is no
  multi-checkpoint path, no `--fusion`, no TTA. The task brief's description of
  "multi-checkpoint fusion / averaging sigmoid probabilities" is not true at this
  baseline; this task must add it. (This is a planning fact, not a bug report.)
- `engine/evaluation.py::collect_predictions` returns `names`, `logits`,
  `probabilities`, `targets`, `ages`; it computes metrics from probabilities.
  This is the right seam: keep logits, fuse logits, compute metrics on the fused
  score.
- `engine/checkpoint.py::load_model` restores the checkpoint's own config,
  age stats and weights; `inference_payload` stores config + age stats + weights
  + epoch + metrics, but **not** a source hash or preprocessing version (R_01
  F4/F5; out of scope here, but artifacts must record hashes externally).
- `data/dataset.py::read_samples` preserves CSV order and names; fusion must
  align by `Image_name`, never by row index across different checkpoints.
- `data/transforms.py::build_transform` is deterministic at inference; training
  uses hflip + rotation ±10° + mild colour jitter. This defines the safe TTA
  set.
- The four 60-epoch arms at `configs/experiments/` all use seed 42 and the same
  ConvNeXt-Tiny architecture. They are **recipe diversity, not seed diversity**:
  - `b0_crop384_control`: pos_weight=5, no EMA, SD=0.1.
  - `b0_crop384_control_ema`: same recipe + EMA 0.999.
  - `b0_crop384_sampled_ema`: positive_fraction=0.1428, pos_weight=1, EMA.
  - `b0_crop384_full_recipe`: positive_fraction=0.1428, pos_weight=1, EMA,
    mixup 0.2, label smoothing 0.05, SD=0.1.
  `control` and `control_ema` follow the same data order and trajectory, so
  their predictions will be nearly collinear (ρ expected >0.99). They are a
  poor ensemble pair. `full_recipe` vs `sampled_ema` differ in sampling,
  loss weighting and mixup/LS and are the predeclared pair.
- The 512 arm (`b0_crop512_sampled`) exists as a config only; no result yet.
  The age-residual model is a separate 20-epoch fine-tune, not one of the four
  from-scratch arms.

### 1.1 Latency baseline `[M]`

V100-SXM2-16GB, 384 fp16, batch=1, in-process, `decode=draft`, 20 valid images
including the 10 largest (`docs/LATENCY.md` §2.2):

| stage | p50 ms | p95 ms |
|---|---:|---:|
| read | 0.66 | 31.87 |
| decode | 24.74 | 34.06 |
| preprocess (crop + letterbox) | 13.59 | 26.37 |
| transfer | 0.83 | 0.92 |
| forward | 7.22 | 7.49 |
| post | 0.07 | 0.10 |
| **total** | **55.28** | **82.15** |

The T4 projection used everywhere below is the conservative `docs/LATENCY.md`
rule: **forward ×2.0, CPU stages unchanged**. Therefore

```
T4_total_p95(F forwards) = (82.15 - 7.49) + 2 * 7.49 * F
                         = 74.66 + 14.98 F          [E]
```

This gives 89.6 ms for F=1, 104.6 ms for F=2, 119.6 ms for F=3, 134.6 ms for
F=4. The `p50` projection is 62.6, 77.0, 91.4, 105.8 ms respectively. **The p95
line is the binding one.**

Important caveats:

- The projection assumes the judge CPU is as fast as the server CPU. `[M]` shows
  a +50 % CPU slowdown already yields ~108 ms p95 for a **single** model. The
  ensemble/TTA gate must therefore be run on the real T4 host, not inferred.
- The current arms train on `decode=full`. Task 09 is the only place allowed to
  switch the canonical decode to `draft`; if the draft AUROC parity gate fails,
  no ensemble or TTA can ship (full decode alone is ~870 ms p50).
- The forward numbers are batch=1. TTA views of the **same** model can be
  concatenated into one batch-2 forward, which should be cheaper than two
  batch-1 forwards but is **not** free; measure it. Different models cannot be
  batched this way.

### 1.2 Size baseline `[M]`

| artifact | size |
|---|---:|
| ConvNeXt-T fp32 inference checkpoint (`best.pt`) | ~114.5–114.8 MB |
| ConvNeXt-T fp16 inference export (new) | ~57.3 MB |
| `last.pt` (optimizer + EMA + best copy) | ~350–400 MB — **never submit** |

The 500 MB limit is not the binding constraint: four fp32 checkpoints (~458 MB)
fit by size but not by latency; eight fp16 checkpoints (~458 MB) fit by size but
are hopeless by latency. **Latency chooses the ensemble size, size only caps the
export format.**

---

## 2. Which models to combine

### 2.1 Diversity argument

For two models with individual AUROC ≈0.93 and score correlation ρ, the
ensemble gain is roughly proportional to `sqrt((1-ρ)/2)` and is small when
ρ>0.97. Expected correlations for this queue:

| pair | expected ρ | expected 2-model ΔAUROC `[E]` | deployable? |
|---|---:|---:|---|
| control + control_ema | >0.99 | ~0.000–0.001 | soup only; not worth a slot |
| full_recipe + sampled_ema | 0.96–0.98 | +0.002–0.005 | **predeclared pair** |
| full_recipe + control_ema | 0.94–0.97 | +0.002–0.006 | alternate pair |
| any 384 + 512 | 0.90–0.94 | +0.003–0.008 | **no** (latency) |
| 384 + SSL-init fine-tune | 0.90–0.95 | +0.003–0.008 | if it exists and fits one forward |

Resolution diversity is the strongest lever but is exactly the lever the
deployment budget forbids: a 512 model already projects to ~101–109 ms p95
alone. Use 512 only as (a) a valid-only diversity measurement, (b) a teacher for
distillation, or (c) an init for a 384 student. Do not put it in the shipped
ensemble.

Seed diversity is not available: all four arms use seed 42. Do not call the four
arms a seed ensemble. If a true seed ensemble is wanted, it requires two extra
60-epoch runs of the winning recipe; that is a task 09/10 decision, not task 07.

### 2.2 Predeclared membership (no valid-label search)

To avoid selecting the pair by its valid AUROC (R_01 F2/F13), **predeclare**:

- **Primary pair:** `full_recipe` + `sampled_ema`.
- **Alternate pair (only if primary soup/fusion fails the accuracy gate):**
  `full_recipe` + `control_ema`.
- **Do not pair** `control` with `control_ema` (same trajectory), and **do not**
  include `control` alone in a prediction ensemble unless it passes the same
  single-model quality screen.
- **SSL replacement rule:** if task 08 produces a fine-tuned 384 model and its
  single-model valid AUROC is within 0.003 of the best 384 arm, it replaces
  `sampled_ema` as the second member (it is expected to be more decorrelated).
  This is a coarse, predeclared strength screen, not a fusion-gain search; record
  it as such.
- If a member is missing or fails to produce a complete 5,849-row prediction
  file, abort the fusion; **never** silently drop a member after seeing valid.

All pairwise Spearman correlations on valid predictions are recorded for the
archive (label-free, cannot overfit), but they do not change membership.

### 2.3 Weight-space soup vs prediction ensemble

**Primary: equal-weight model soup.** For same-architecture checkpoints with
identical parameter keys, average every floating tensor in `model_state` in
fp32, keep non-floating entries from the first checkpoint, and evaluate the
resulting single model. This is the only technique in this plan that gets
multi-model benefit at **F=1**, 57 MB and no extra latency.

Rules:
- Only average models with identical architecture/config-shape (all 384 arms
  satisfy this; 384 vs 512 are different input sizes but the same parameter
  shapes — do **not** soup across resolutions because the models were trained on
  different input distributions; keep soup within 384).
- Require identical `state_dict` keys and shapes; otherwise raise.
- Use equal weights. A greedy soup (add a member only if valid AUROC improves)
  is exploratory and uses valid labels; run it only for the archive, never ship
  it without the paired bootstrap and a confirm-half check.
- Evaluate the soup on valid with the same protocol as a single model. Gate:
  `ΔAUROC(soup − best single) ≥ +0.002` and paired-bootstrap 95 % CI LB > 0.
- If soup fails, fall back to the best single model. A prediction ensemble is
  only considered if the measured T4 latency gate passes (unlikely).

Why soup before prediction fusion: the soup has the same deployment cost as the
best single model, so it can never make the submission worse on latency or size.
Its risk is accuracy instability, which is measured, not assumed.

---

## 3. Fusion rule (prediction-level, only if used)

### 3.1 Default: standardised logit mean

Let model `i` produce logit `l_i(x)` for image `x`. Define a **label-free,
pointwise** standardiser fitted on a fixed train reference sample `R`:

```
R = 2,048 train images sampled deterministically (seed 42, uniform over rows)
mu_i  = mean_{r in R} l_i(r)
sd_i  = std_{r in R}  l_i(r)          # require sd_i > 1e-6
z_i(x) = (l_i(x) - mu_i) / sd_i
s(x)   = sum_i w_i z_i(x),   w_i = 1/M   (equal weights)
p(x)   = sigmoid(s(x))
```

Why train reference: it uses no valid/test labels, it is a fixed function
(so a single image can be scored), and it removes the scale differences caused
by `pos_weight`, mixup and label smoothing. The reference sample is not the
training loss set; it is only a scale estimate. Save `mu_i`, `sd_i`, the sample
spec and the seed in `fusion.json`.

- Do **not** fit the standardiser on valid or test: that leaks valid labels and
  makes the fused valid score transductive.
- Do **not** tune `w_i` on valid. With 191 positives, a 4-weight grid on 5,849
  images overfits; equal weights are the minimax choice. A valid-only weighted
  grid may be run for analysis and must be labelled "oracle/optimistic"; it is
  never shipped.
- If a model's logits are all constant (`sd_i` ≈ 0), drop it and renormalise the
  weights; if all are constant, abort.

### 3.2 Rank averaging (analysis / batch-only fallback)

For a fixed evaluation set of `N` images:

```
r_i = average_rank(l_i)          # 1..N, ties get the average rank
u_i = (r_i - 0.5) / N
s_rank = sum_i w_i u_i,  w_i = 1/M
```

Use this for offline sensitivity analysis and for a batch submission CSV only
if it beats the logit rule by ≥0.002 with CI LB > 0. It is **not** the shipped
default because:
- it is transductive (depends on the other images in the set), so a single-image
  latency path or a per-image judge cannot reproduce it;
- the natural output is uniform in `[0,1]`, which destroys any calibrated
  `Pred_Label` at 0.5.

If rank fusion is ever used for the submission, write `Probability` as the
fused rank score and choose `Pred_Label` from a valid-derived threshold; state
this explicitly in the run card. The default is **logit**.

### 3.3 Per-checkpoint preprocessing

Each checkpoint carries its own `Config` (`image_size`, crop, decode). The fusion
runner must:

1. Build one `FundusDataset`/loader per distinct preprocessing spec. Do not
   assume the same size or decode across members.
2. Align predictions by exact `Image_name` after inference; verify the name set
   equals the official split's set and that no name is duplicated. Never align
   by row index.
3. Store `(path, sha256, bytes, config_sha256, image_size, decode, epoch)` for
   every member in `checkpoints.json`.
4. If a member's `decode` differs (e.g. one full, one draft), the fusion is
   valid but the latency measurement must include both decode paths; prefer all
   members on the same decode policy for the shipped config.

### 3.4 Probability output and labels

`sigmoid(s)` is a ranking score, not a calibrated risk (R_01 F16). For the
competition, `Probability` is the ranked output; keep `Pred_Label = p >= 0.5`
for backward compatibility, but:
- report sensitivity/specificity at the valid-selected threshold in the run card;
- if the official metric scores labels, tune the threshold on valid (never test)
  and document it;
- do not isotonic-calibrate on valid and then report valid AUROC as if clean;
  use cross-fitting if calibration is needed.

---

## 4. TTA

### 4.1 Safe transforms for fundus

| transform | safe? | reason | cost |
|---|---|---|---|
| horizontal flip | **yes** | training uses `RandomHorizontalFlip(0.5)`; glaucoma cues (cup/disc ratio) are not left/right encoded in this dataset | one extra view, batchable |
| small rotation ±10° | yes, in-distribution | training uses `RandomRotation(10°, fill=0)`; black corners are in-distribution | one extra view; affine on 384 tensor is cheap |
| multi-scale 352/416 | risky | model trained only at 384; ConvNeXt is fully convolutional so it runs, but features are not scale-invariant | extra preprocess + forward |
| centre crop 90 % | risky | may cut the optic disc; training used the full field | extra preprocess |
| colour jitter | no | not a label-preserving view; changes appearance, not geometry | — |
| vertical flip | no | not in training; fundus has a vertical orientation | — |
| 90°/arbitrary rotations | no | not in training; disc position changes | — |

### 4.2 Aggregation

- Compute one logit per view; **average raw logits** for views of the same model
  (same scale), then apply the model's standardiser, then fuse across models.
- If a view is produced by a different input size, still average the logit; the
  model is the same and the scale is comparable.
- Batch the views into one forward when possible:
  `x2 = torch.cat([x, x.flip(-1)], dim=0)`, `age2 = torch.cat([age, age], 0)`,
  `logits2 = model(x2, age2)`, then `logit = 0.5*(logits2[:N] + logits2[N:])`.
  This is the only way hflip has a chance of fitting.

### 4.3 Concrete default and measurement plan

- **Default shipped TTA: `none`.**
- **Predeclared optional view: `hflip` only.** Ship it only if the exact
  configuration's T4 p95 is ≤90 ms (10 ms safety margin).
- **Exploratory, not shipped by default:** `hflip + rot10` (two extra views),
  `hflip + ms352 + ms416` (multi-scale). Run them on valid and record the
  accuracy/latency trade-off; only consider shipping if one of them passes the
  same T4 gate and adds ≥0.002 AUROC with CI LB > 0.
- Measurement matrix (V100 screening then T4 acceptance):
  1. single, no TTA (baseline);
  2. single, hflip batched (batch=2 forward);
  3. single, hflip + rot10 batched (batch=3/4);
  4. single, hflip + ms352 + ms416 (separate preprocess);
  5. two-model sequential, no TTA;
  6. two-model + hflip batched per model.
  For each: `n=200`, `--include-largest 10`, `--decode draft`, `--warmup 20`,
  repeat 3×, report median p95 and the projected T4 number. Only a real T4 run
  is acceptance.
- V100 screening rule for task 07: **projected T4 p95 ≤95 ms** (5 ms margin) and
  **forward p95 ≤15 ms per forward**. Do **not** use the stale 50 ms total
  target from `docs/LATENCY.md` §5: the measured single-model V100 p95 is already
  82.15 ms, so that target is unmet at baseline and would reject every
  configuration. The projection is the screen; the T4 measurement is the gate.

---

## 5. Age fusion

### 5.1 Form

Age alone has valid AUROC **0.696** (`docs/BASELINE_DESIGN.md`). The teammate's
age-residual fine-tune gave **+0.002** with a 95 % CI touching 0. Therefore the
post-hoc stacker is specified, but **default-off**.

Use a monotone, one-parameter rank stacker:

```
a(x) = clip((age - median_train) / IQR_train, -3, 3)
       missing/invalid -> 0.0, and a separate 0/1 indicator is NOT used
s(x) = s_img(x) + lambda * a(x)          # s_img is the standardised fused image logit
```

- `median_train`, `IQR_train` are fitted on valid ages in `(0,100]` from
  `train.csv` only.
- `lambda >= 0` is required (older → higher risk). A negative fitted value means
  the sign is unstable; reject age fusion.
- The indicator for missing/invalid is dropped: only 10 missing + 4 invalid in
  train and 2 invalid in valid, so it cannot be estimated and is not monotone.
  Those rows get `a=0` (median age) and are listed in the run card.

### 5.2 Fitting without leaking valid labels

**Protocol of record (gold): 5-fold grouped OOF from train.**

1. Build groups by scanning `train.csv` in order: if adjacent rows have the same
   finite age in `(0,100]`, put them in one group (max 2; DATA_AUDIT: 22,192
   groups of size 2, 47 % of adjacent rows). Do **not** group all rows sharing an
   age globally.
2. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on
   `(group, label)`.
3. For each fold, train the chosen image recipe on the other four folds for the
   full 60 epochs and predict the held-out fold. Five runs are required; on five
   free GPUs this is one wall-clock run, otherwise it is not worth the +0.002
   prior.
4. For each fold, fit `lambda` on the other folds' OOF scores by maximising
   AUROC on a grid `[0, 0.5]` step 0.01; predict the held-out fold. Concatenate
   the held-out scores → OOF stacked score.
5. Report `ΔAUROC(OOF stacked − OOF image-only)` and the paired bootstrap CI.

**Pragmatic fallback (what task 07 should actually run first): 5-fold grouped
CV on valid.** The image model never saw valid, so valid is out-of-sample for
the stacker; use the same group construction and `StratifiedGroupKFold` on valid
rows, fit `lambda` on 4 folds, predict the 5th. This costs seconds, not five
retrainings. It is not a substitute for train OOF for a publication claim
because checkpoint selection already used valid, but it is the right cheap
screen. If it passes, confirm with train OOF before shipping.

### 5.3 Acceptance gate

Ship age fusion only if **all** hold:

- OOF `ΔAUROC ≥ +0.003`;
- paired-bootstrap 95 % CI lower bound **> 0** (10k, stratified, grouped by
  same-age pairs);
- `lambda > 0` in at least 4 of 5 folds;
- `ΔAP ≥ -0.002` (non-inferiority);
- no latency/size change at inference (age is free; assert it).

Given the prior +0.002 and CI touching 0, **the expected outcome is FAIL and no
age fusion is shipped.** Do not relax the gate because the technique is cheap;
the risk is a valid-set overfit that does not transfer to test.

---

## 6. Verification protocol

### 6.1 Paired bootstrap

Implement `paired_bootstrap_auroc(y, new, ref, n=10000, seed=42, groups=None)`:

- image-level, **stratified**: each resample draws `n_pos=191` positives with
  replacement from the positive indices and `n_neg=5658` negatives with
  replacement from the negative indices; concatenate. This preserves prevalence
  and prevents a resample with no positives.
- Compute `ΔAUROC = AUROC(y_b, new_b) − AUROC(y_b, ref_b)` on the same indices.
- Return `delta_mean`, `ci_low` (2.5 %), `ci_high` (97.5 %), `p_win =
  mean(delta > 0)`, `n`, `n_pos`, `n_neg`.
- **Grouped sensitivity:** repeat with groups = same-age adjacent pairs
  (blocks); resample groups within label strata. Report both the image-level and
  grouped CI; the grouped CI is wider and is the honest one when a pair straddles
  the split.
- Use `float64` and `sklearn.metrics.roc_auc_score`; 10k × 5,849 is fast enough
  in a script (seconds to a minute). Fix the seed and test determinism.

### 6.2 Minimum effect to claim a win

| claim | minimum |
|---|---|
| "ensemble/soup/TTA/age helps" | `ΔAUROC ≥ +0.003` **and** 95 % CI LB > 0 |
| "strong win" | `ΔAUROC ≥ +0.005` and CI LB > 0 |
| ship a configuration | the above **and** exact T4 p95 ≤ 90 ms **and** size ≤ 500 MB |
| non-inferiority of AP | `ΔAP ≥ -0.002` |

Why +0.003: R_01 estimates `SE(AUROC) ≈ 0.013` and paired `SE(Δ) ≈ 0.004–0.006`
for correlated models. A paired bootstrap can resolve 0.003, but it cannot fix
selection bias; +0.003 with CI LB > 0 is the smallest claim worth acting on.
Anything below +0.003 is noise/selection and must not change the submission.

### 6.3 Seeds and selection

- We have **one seed per recipe**. The unit of diversity is recipe, not seed.
  Do not report a "seed ensemble"; report a "recipe ensemble".
- To claim a seed-ensemble effect, run ≥3 seeds of the winning recipe, report
  mean ± range, and bootstrap over the seed-mean. That is not task 07.
- Every checkpoint was selected on valid (best of 60 epochs), so all valid
  numbers are optimistically biased by roughly +0.005–0.013 (R_01 §3). State
  this in every run card.
- If a confirm-half is affordable: split valid **before** looking at fusion
  results into a selection half and a confirm half, stratified by label and
  grouped by same-age pairs. Choose the fusion config on the selection half and
  report the confirm half as a sign check (95 positives → wide CI; not a gate).
  If not affordable, use the predeclared configuration and say so.

### 6.4 Artifact set (must be enough to recompute every number)

For every candidate configuration, save to
`outputs/fusion/<config_id>/` (ignored by git, retained on the server):

| file | contents |
|---|---|
| `valid_predictions.csv` | one row per valid image in official order: `Image_name,True_Label,raw_age,logit__<member>,prob__<member>,fused_score,fused_prob` |
| `test_predictions.csv` | same, for the shipped config only, plus `Pred_Label`; official test order |
| `metrics.json` | per-member and fused AUROC/AP/`auprc_trapz`/sens@spec95, age-stratified groups, count/positives |
| `bootstrap.json` | 10k paired-bootstrap results per comparison, image-level and grouped |
| `fusion.json` | fusion kind, members, weights, `mu/sd` reference spec, TTA views, age `lambda`, code commit, config sha256 |
| `checkpoints.json` | path, sha256, bytes, dtype, config sha256, epoch, stored metrics |
| `latency.json` | V100 + T4 benchmark output, per-stage p50/p95/p99, projection, gate PASS/FAIL |
| `README.md` | one-paragraph run card: what, why, exact command, result, caveat |

Commit the small JSON/README files **and** `valid_predictions.csv` for every
candidate under `.w1mer/results/fusion/<config_id>/` (the CSV is ~1–2 MB and is
the only way to recompute a fusion without retraining; if it exceeds 2 MB,
gzip it and record the sha256). Checkpoints stay in `outputs/` and are referenced
by hash; never commit `*.pt` (gitignored). This satisfies "future comparisons
without retraining" even after the run directory is deleted on the server.

`config_id` format:
`YYYYMMDD_<fusion>_<member1>+<member2>[_tta]_<size>_<decode>`,
e.g. `20261023_soup_fullrecipe+sampledema_384_draft`.

---

## 7. Deployment consequence table

Projections use §1.1. All assume `decode=draft`; if task 09 keeps `full`, every
row is ~15× slower and no ensemble/TTA ships.

| # | configuration | forwards | size (fp16) | projected T4 p95 `[E]` | verdict |
|---|---|---:|---:|---:|---|
| 1 | single 384 | 1 | 57 MB | 89.6 ms | **SHIP baseline** (pending draft parity) |
| 2 | **soup of two 384 arms** | 1 | 57 MB | 89.6 ms | **SHIP if ΔAUROC ≥ +0.002** |
| 3 | single 384 + hflip, batched | 1×batch2 | 57 MB | 97–105 ms | **conditional**: only if T4 ≤90 ms |
| 4 | two 384 models, sequential | 2 | 115 MB | 104.6 ms | fail by projection; T4 measurement only |
| 5 | two 384 + hflip each | 2×batch2 | 115 MB | ~119.6 ms | **no** |
| 6 | four 384 models | 4 | 229 MB | 134.6 ms | **no** |
| 7 | single 512 | 1@512 | 57 MB | ~101–109 ms | **no** (marginal/fail) |
| 8 | two 320 models | 2@320 | 115 MB | ~88–95 ms | **conditional**: requires 320 accuracy retention |
| 9 | distilled 384 student | 1 | 57 MB | 89.6 ms | fallback if teacher gain ≥ +0.005 |

Notes:

- Row 2 (soup) is the only way to get multi-model benefit at F=1. It is the
  recommended path.
- Row 3: batched hflip is cheaper than two batch-1 forwards but not free. The
  97–105 ms range assumes batch-2 costs 1.5–2.0× a batch-1 forward. If the real
  T4 batch-2 forward is ≤12 ms, total ≈91 ms and it may pass; measure.
- Row 8: 320 forward scales ~`(320/384)^2 = 0.69`; CPU decode/preprocess is also
  slightly cheaper. This is the only resolution-diverse deployable option, but
  no 320 models exist; do not train them for task 07 unless the soup and hflip
  both fail and there is spare GPU time. A 320 model can also be obtained by
  evaluating a 384 checkpoint at 320 (cheap to test, distribution shift) before
  committing to training.
- Row 9: distillation is only justified if the teacher ensemble is clearly
  better (≥+0.005) and the T4 gate fails. It costs one 60-epoch run and may not
  transfer. See §8.3.

### 7.1 Distillation plan (fallback only)

- Teacher: the predeclared 2-model prediction ensemble (or the 4-model if its
  gain is ≥+0.005), frozen.
- Student: ConvNeXt-T 384, initialized from the best single checkpoint, same
  recipe, 30–40 epochs.
- Targets: train images only. Use `0.5 * BCE(student, hard_label) + 0.5 *
  KL(student/T || teacher/T)`, `T=2`, `pos_weight=1`, strong augmentation.
- Caveat: teachers were trained on all of train, so their train soft targets are
  overfit. Cross-fitted teachers (5 folds) are the clean version but cost 5×
  training. Do not run distillation unless the teacher gain is large enough to
  survive this risk.
- Gate: student valid AUROC ≥ best single +0.002 **and** ≥ teacher −0.002;
  single-model latency; 57 MB fp16. Otherwise ship the best single.

---

## 8. Code design

No new runtime dependency. `numpy`, `scikit-learn` and `Pillow` are already
declared. Do not import `scipy` directly (it is only a transitive dependency);
implement average ranks in numpy.

### 8.1 New module `src/glaucoma/fusion.py`

Pure functions + one dataclass; no torch, no file I/O except CSV helpers.

```python
def average_ranks(values) -> np.ndarray          # 1-based, ties averaged
def rank_scores(values) -> np.ndarray            # (rank-0.5)/n in (0,1)
def fit_standardizer(values) -> tuple[float, float]
def apply_standardizer(values, mean, std) -> np.ndarray
def fuse_logits(logits_by_model, weights=None, standardizers=None) -> np.ndarray
def fuse_ranks(logits_by_model, weights=None) -> np.ndarray
def align_by_name(predictions: list[dict]) -> list[dict]   # raises on missing/extra/duplicate
def paired_bootstrap_auroc(y, new, ref, n=10000, seed=42, groups=None) -> dict
def age_score(ages, median, iqr) -> np.ndarray
def fit_age_lambda(scores, y, ages, grid=np.arange(0, 0.501, 0.01)) -> float
def group_same_age_pairs(rows) -> list[int]      # group id per row
@dataclass
class FusionSpec:
    kind: str                     # "logit" | "rank" | "prob"
    members: list[str]
    weights: list[float] | None
    standardizers: list[tuple[float, float]] | None
    tta: str
    age_lambda: float | None
    def validate(self): ...
    def to_dict(self): ...
    @classmethod
    def from_dict(cls, values): ...
```

Validation rules: `kind in {"logit","rank","prob"}`, `len(members)==len(weights
or members)`, weights positive and sum to 1, standardizers length matches,
`tta in {"none","hflip","hflip+rot10","hflip+ms"}`, `age_lambda is None or >=0`.
`align_by_name` must return predictions in the official row order and raise if
the name sets differ.

### 8.2 New module `src/glaucoma/engine/inference.py`

Extract the inference loop from `cli.py::_inference` so the CLI and the fusion
script share one path:

```python
def infer_checkpoint(checkpoint, data_root, split, device, batch_size=16,
                     num_workers=4, tta="none") -> dict
def infer_checkpoints(checkpoints, data_root, split, device, **kwargs) -> list[dict]
```

`infer_checkpoint` uses the checkpoint's own `config.data` (size, crop, decode)
and `age_stats`; it returns the same dict as `collect_predictions` plus
`config_sha256`, `checkpoint_sha256`, `checkpoint_bytes`, `epoch`, `image_size`,
`decode`. `infer_checkpoints` runs each member independently and calls
`soup.align_by_name` before returning.

### 8.3 `engine/evaluation.py` TTA hook

Extend the signature (default preserves behaviour):

```python
def collect_predictions(model, loader, device, pos_weight=5.0, amp=False, tta="none"):
```

- `tta="none"`: exact current code path.
- `tta="hflip"`: build `images2 = torch.cat([images, images.flip(-1)], 0)`,
  `ages2 = torch.cat([ages, ages], 0)`, forward once, average the two halves.
- Any richer TTA lives in `scripts/fuse.py` or a future `engine/tta.py`; do not
  bloat `collect_predictions`.
- Add a unit test asserting `tta="none"` is bit-identical to the old function's
  output on a fixed batch.

### 8.4 `cli.py`

- Change `--checkpoint` from a string to `action="append", required=True`
  (one or more). Existing single-checkpoint invocations are unchanged.
- Add for `evaluate` and `predict`:
  - `--fusion {logit,rank,prob}` default `logit` (only affects >1 checkpoint);
  - `--tta {none,hflip}` default `none`;
  - `--weights "1,1"` optional, analysis only (validated positive, normalised);
  - `--save-per-model` optional, writes one CSV per member in `evaluate`.
- `_inference`: call `infer_checkpoints`; if one member, call the old
  `write_predictions` path with the old columns (byte-identical output); if
  multiple, write `fused.csv` plus `model_<i>.csv` under `--output-dir` for
  `evaluate`, and a single fused `submission.csv` for `predict`.
- `evaluate` writes `metrics.json` with `members: {...}` and `fused: {...}`;
  the top-level `auroc`/`ap` keys remain for backward compatibility.
- Do not change the checkpoint schema or `load_model`.

### 8.5 `scripts/fuse.py` (offline, no GPU)

Consumes saved per-member `predictions.csv` files and recomputes fusion,
bootstrap and age stacking without re-running inference. This is the tool used
for the artifact set and for reproducing a result after the server run dir is
gone.

```
--prediction NAME=PATH        repeatable; CSV with Image_name,True_Label,Logit,Probability
--fusion {logit,rank,prob}
--weights 1,1
--reference PATH              optional train-reference CSV for mu/sd; else fit on the
                              supplied valid CSV and mark the spec "leaky/analysis"
--tta {none,hflip}            only metadata; TTA must have been applied at inference
--age {none,rank} --age-lambda L
--bootstrap 10000 --groups {image,pair}
--output-dir DIR
```

It writes the §6.4 artifact set and refuses to overwrite an existing directory
(repo convention: exclusive creation).

### 8.6 `scripts/soup.py`

```
--checkpoint PATH              repeatable, >=2
--weights 0.5,0.5              optional; default equal
--output PATH                  new .pt, exclusive creation
--data-root --split valid --device cpu|cuda --evaluate   optional
--fp16                         save a half-precision export
```

Average every floating tensor in `model_state`; require identical keys/shapes;
keep non-floating entries from the first; config/age_stats from the first (they
must be architecture-identical); add `soup_members` (paths + sha256) and
`soup_weights` to the payload; evaluate and store `metrics` if requested.
`read_checkpoint` ignores the extra keys; `load_model` still works.

### 8.7 `scripts/export_fp16.py`

Convert `model_state` floating tensors to `torch.float16`, keep the rest,
add `"precision": "fp16"` and `"source_sha256"`, save with `atomic_write`.
Verify on 10 valid images: max `|Δlogit|` ≤ 1e-3 and AUROC delta ≤ 0.001.
`load_model` does not need a change: `load_state_dict` casts fp16 tensors into
fp32 parameters; the latency harness already calls `model.half()`. If a future
CLI path wants to run half without autocast, add an explicit `model.half()` and
cast the input there; do not change the default.

### 8.8 `scripts/benchmark_latency.py`

Extend without changing the default:

- `--checkpoint` `action="append"` (one or more); load each, time sequential
  forwards; report per-model and total `forward`.
- `--tta {none,hflip}`; for hflip, use a batch-2 tensor per model and time it as
  one forward.
- Artifact size = sum of all checkpoint files; gate ≤500 MB.
- Projection: `projected_t4_total = total - forward + 2*forward` (CPU stages
  unchanged). Print both per-model and total.
- V100 screening: `projected_t4_total p95 ≤95 ms` and `forward p95 ≤15 ms` per
  forward. Do not use the stale 50 ms V100 total target from `docs/LATENCY.md`
  §5 (the measured baseline is 82.15 ms; see §4.3). T4 acceptance uses
  `--max-latency-ms 90` (safety margin) and must be run on the real T4 host.
- Add per-image rows to JSON (already present) so the p95 can be recomputed.

### 8.9 Tests (CPU, no competition images, no GPU)

New `tests/test_fusion.py`:

- `average_ranks([0.1, 0.2, 0.2, 0.4]) == [1, 2.5, 2.5, 4]`; `rank_scores` in
  `(0,1)` and monotone; ties handled.
- `fit_standardizer` / `apply_standardizer` round-trip; constant input raises or
  is flagged.
- `fuse_logits` equal weights; weights normalised; wrong length raises.
- `align_by_name` reorders, raises on missing/extra/duplicate.
- `paired_bootstrap_auroc` deterministic under seed, CI brackets the point
  estimate, `p_win` in `[0,1]`.
- `age_score` monotone, missing/invalid → 0; `fit_age_lambda` picks 0 when age is
  uninformative on a synthetic set.
- `group_same_age_pairs` groups only adjacent equal-age rows.
- `FusionSpec` round-trip and validation errors.

New `tests/test_soup.py`:

- Two identical `_test_tiny` checkpoints → soup identical to the original.
- Different keys/shapes → raises.
- Soup of two different checkpoints → finite logits, `load_model` works, extra
  metadata preserved.

New `tests/test_cli_fusion.py` (or extend `WorkflowTests`):

- Single-checkpoint `evaluate`/`predict` output is byte-identical to the
  pre-change baseline for the existing fixture (regression).
- Two-checkpoint `evaluate --fusion logit` writes a fused CSV whose score is the
  mean of the two standardised logits; name alignment is correct.
- `--tta hflip` on a model whose output is invariant to horizontal flip gives the
  same logits as `--tta none`; on a non-invariant model it averages correctly.
- `scripts/fuse.py` reproduces the same fused metrics from the saved CSVs
  without torch/GPU.

New `tests/test_fp16_export.py`:

- Export a `_test_tiny` checkpoint; all floating state tensors are half; file is
  smaller; `load_model` loads it; logits are close to fp32 on a fixed input.
- `last.pt`-like payload with optimizer state is not accepted by the export
  script (only `kind="inference"`).

Do not run ConvNeXt in unit tests; use the registered `_test_tiny` fixture.

### 8.10 Backward compatibility checklist

- `collect_predictions(..., tta="none")` is bit-identical to the old path.
- Single `--checkpoint` CLI output columns, order and formatting are unchanged;
  `write_predictions` is untouched for the single-model submission.
- `best.pt`/`last.pt` schema unchanged; the fp16 export is an additional
  artifact, not a replacement. `load_model` reads both.
- `metrics.json` keeps the old top-level keys for `evaluate`; new member metrics
  go under a new key.
- No change to training, optimizer, LR schedule or data caching.
- No new dependency; no network; no test images.

---

## 9. Execution order (implementer)

1. **Land the reusable code first**, with no behaviour change: `fusion.py`,
   `inference.py`, `collect_predictions(..., tta="none")`, CLI append-only
   `--checkpoint`, `soup.py`, `export_fp16.py`, tests. Run the full CPU suite;
   assert single-checkpoint parity.
2. **Wait for the four arms.** For each arm, run `evaluate` on valid once and
   save `valid_predictions.csv` (5,849 rows each). Do not re-run inference per
   fusion candidate.
3. **Compute label-free diversity**: pairwise Spearman on valid predictions;
   confirm `full_recipe` vs `sampled_ema` is not ρ>0.99. Record all pairs.
4. **Evaluate the soup** of the predeclared pair on valid; run the paired
   bootstrap vs the best single. Apply the §6.2 gate.
5. **Evaluate prediction fusion** (logit default, rank analysis) from the saved
   CSVs. Apply the same gate. This is the accuracy ceiling; it is allowed to be
   better than the soup but is not allowed to ship if latency fails.
6. **Latency screen** the winners on V100: soup (F=1), single+hflip (F=1 batch2),
   two-model (F=2). Project to T4. Only then run T4.
7. **T4 acceptance**: exact configuration, 200 images, largest 10, draft,
   warmup 20, 3 repeats, p95 ≤90 ms and size ≤500 MB. If no T4 is available,
   ship the best single 384 and record "ensemble/TTA not shipped: no T4
   measurement".
8. **Export fp16** for the shipped checkpoint; re-run parity (max |Δlogit| ≤1e-3,
   AUROC delta ≤0.001) and the latency harness.
9. **Optional age CV** on valid (seconds); apply §5.3. Default expectation: no.
10. **Test inference**: `predict` on test, verify 5,855 rows, names/order match
    `test.csv`, probabilities in [0,1], no NaN; archive the §6.4 artifact set;
    write `.w1mer/results/FUSION.md` and a result row.

### 9.1 Commands (to be run on the server; fill `<...>`)

```bash
PY=/home/lichunyu/miniconda3/envs/glaucoma/bin/python
ROOT=/home/lichunyu/competition/dataset/glaucoma_first

# per-arm valid predictions (one GPU each, ~2-4 min)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY -m glaucoma evaluate \
  --checkpoint outputs/runs/<arm>/best.pt --data-root $ROOT --device cuda \
  --output-dir outputs/fusion/<arm>_valid

# soup (CPU or one GPU) + evaluation
PYTHONPATH=src $PY scripts/soup.py \
  --checkpoint outputs/runs/<full_recipe>/best.pt \
  --checkpoint outputs/runs/<sampled_ema>/best.pt \
  --output outputs/fusion/soup_full+sampled.pt \
  --data-root $ROOT --split valid --device cuda --evaluate

# offline fusion + bootstrap (no GPU)
PYTHONPATH=src $PY scripts/fuse.py \
  --prediction full=outputs/fusion/<full_recipe>_valid/predictions.csv \
  --prediction sampled=outputs/fusion/<sampled_ema>_valid/predictions.csv \
  --fusion logit --bootstrap 10000 --groups pair \
  --output-dir outputs/fusion/20261023_logit_full+sampled_384_full

# latency screen (V100): projected T4 p95, not the stale 50 ms total target
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src $PY scripts/benchmark_latency.py \
  --checkpoint outputs/fusion/soup_full+sampled.pt \
  --data-root $ROOT --split valid --images 200 --include-largest 10 \
  --decode draft --warmup 20 --max-latency-ms 95 --output /tmp/lat_soup.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/benchmark_latency.py \
  --checkpoint outputs/fusion/soup_full+sampled_fp16.pt \
  --data-root $ROOT --split valid --images 200 --include-largest 10 \
  --decode draft --warmup 20 --max-latency-ms 90 --output /tmp/lat_t4.json
```

Use `/tmp` on the remote; do not write under another user's home. Check
`nvidia-smi` before claiming a GPU; our allocation is `0,1,3,4,5` (R_01 F6).

---

## 10. Risks and mitigations

| # | risk | severity | mitigation |
|---|---|---|---|
| 1 | T4 projection is not a measurement; judge CPU may be slower | **blocker** for ensemble/TTA | ship single/soup; require real T4 p95 ≤90 ms; no T4 → no ensemble |
| 2 | Draft-decode parity gate fails; full decode is ~870 ms | **blocker** for everything | task 09 decides; if full, no ensemble/TTA can ship; document |
| 3 | 191 positives → ±0.02 AUROC; selection over 60 epochs | major | predeclared config; paired bootstrap; report bias; do not chase <0.003 |
| 4 | No patient ID; same-age pairs may straddle split | major | grouped bootstrap sensitivity; state eye-level limitation |
| 5 | Soup instability (different loss/sampling basins) | major | evaluate on valid; key/shape check; fall back to best single |
| 6 | Valid-label overfitting from choosing fusion/weights/TTA | major | equal weights; predeclared members/views; no grid on valid; confirm-half if possible |
| 7 | Rank fusion is transductive and not pointwise | major | ship logit; rank only for offline analysis/batch CSV with explicit note |
| 8 | Age fusion prior is +0.002 with CI touching 0 | major | default off; strict OOF gate; do not relax |
| 9 | 512 arm looks attractive but cannot ship | major | keep out of deployed ensemble; use as teacher/analysis only |
| 10 | fp16 export changes logits | minor | loose 1e-3 gate; re-evaluate AUROC; keep fp32 fallback |
| 11 | Size accounting includes `last.pt` or duplicate exports | minor | `checkpoints.json` lists exact files; harness sums them; never submit `last.pt` |
| 12 | `Pred_Label` is not calibrated | minor | report ranking metric; tune threshold on valid only if labels are scored |

---

## 11. Acceptance criteria for task 07

1. All new CPU tests pass, and the existing suite is unchanged (single-checkpoint
   output byte-identical).
2. `scripts/soup.py` produces a loadable checkpoint from two same-architecture
   members and evaluates it; `scripts/fuse.py` reproduces the fused valid metrics
   from saved CSVs without GPU.
3. `scripts/benchmark_latency.py` accepts multiple checkpoints and `--tta hflip`,
   reports per-model and total forward, and prints the T4 projection and gates.
4. For the shipped configuration: a real T4 run reports p95 ≤90 ms and the
   artifact sum ≤500 MB; `latency.json` is archived.
5. The §6.4 artifact set exists for every candidate, with checkpoint/config
   hashes; `valid_predictions.csv` is committed for each candidate.
6. The §6.2 accuracy gate is applied; the report states the point estimate,
   image-level and grouped bootstrap CIs, and the selection-bias caveat.
7. The decision tree in §0/§7 is executed and the chosen submission is the
   highest-lower-bound configuration that passes latency, or the best single if
   none does.
8. Age fusion is either absent (expected) or supported by a passing OOF gate;
   `lambda`, folds and CI are recorded.
9. No change to training numerics, checkpoint schema or existing CLI output for
   a single checkpoint; no new dependency.
