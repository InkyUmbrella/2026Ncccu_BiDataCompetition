# Data audit — independent checks (2026-10-22, heyuhang)

Scripts/results on the GPU server: `/tmp/scan2.log`, `/tmp/scan_report.json`,
`/tmp/dhash_an2.log`, `/tmp/dhash_report.json`.

## Leakage / split integrity

| Check | Result |
|---|---|
| Exact duplicate files across train/valid/test (md5, all size-collision candidates) | **0 duplicate groups** |
| dHash 256-bit nearest neighbour, valid → train | median 14 bits; kNN-10 label vote **AUROC 0.609** |
| Label agreement of the 2,009 closest valid→train matches (≤8 bits) | 96.4% (base rate 96.8%) |
| Positive rate among closest matches | 2.19% vs 2.09% for their neighbours → no signal |
| Metadata shortcuts: AUROC of file size / width / height / aspect / area | 0.495 / 0.520 / 0.533 / 0.503 / 0.533 (valid), age 0.695 |
| Image size distribution (train/valid/test medians) | 3088×2136 / 2736×2136 / 3008×2136 → same source mix |

Conclusion: the official split is i.i.d. at image level; there is no exploitable
duplicate or near-duplicate leakage, and metadata cannot shortcut the label.
The validation set is trustworthy as a split (its 191 positives are still a
precision problem — see review R_01).

## Patient/pair structure

- 47% of adjacent training rows share the exact same age; 22,192 of 27,231
  same-age groups are exactly size 2 → rows come in patient (both-eye) pairs.
- Adjacent "positive-positive" rate 0.0074 vs 0.0010 chance (7×) → patient pairs
  are usually both positive or both negative, but 745/1,081 positive pairs are
  mixed (31% of positives have a positive fellow eye).
- Pair-averaging predictions was simulated: at AUC≈0.885 (σ≈1) it is neutral to
  slightly negative, so do **not** average predictions across a pair.

## Image geometry / resolution

- Fundus mask fills a median 72% of the frame; content bbox width median 0.848.
  Cropping the black border before resize gains ≈1.18× linear resolution.
- Median source size 3088×2136; at 384 the optic disc is ≈35–45 px.
- Test time budget: full PIL decode + resize **259 ms** vs `Image.draft` decode
  **34 ms** (7.6×). The current inference path cannot meet 100 ms without the
  draft path; see `.w1mer/detail/03_latency_plan.md`.

## Provenance note

The reported crop384 valid AUROC (~0.923–0.930) comes from runs whose
`metrics.json`/`best.pt` live in another user's home and are not readable by us.
Our own reproduction and all future comparisons must be committed artifacts in
this repository (see R_01 F1).
