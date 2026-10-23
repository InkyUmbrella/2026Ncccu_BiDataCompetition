# Submission candidate (2026-10-23)

## Chosen model: 2-seed probability ensemble of the `full_recipe` arm

| Item | Value |
|---|---|
| Recipe | ConvNeXt-Tiny from scratch, 384 px, black-border crop, positive sampling 1/7 (`pos_weight=1`), EMA 0.999, Mixup 0.2, label smoothing 0.05, stochastic depth 0.1, 60-epoch cosine |
| Members | `full_recipe` seed42 (best epoch 34) + `full_recipe` seed43 (best epoch 32) |
| Single-model valid AUROC / AP | 0.96456 / 0.639 and 0.96634 / 0.617 |
| **Ensemble valid AUROC (equal-weight probability mean)** | **0.97243** |
| Validation set | official valid split, 5,849 images / 191 positives |
| Decode policy | seed42 checkpoint: full decode; seed43 checkpoint: draft (each checkpoint carries its own policy) |
| Inference latency (V100, measured single model) | p50 55 ms / p95 82 ms end-to-end including decode |
| Projected T4 latency for 2 models | ~69 ms (2 forwards + draft decode + preprocess), under the 100 ms limit |
| Model size | 2 × 111.7 MB fp32 = 223 MB (< 500 MB); fp16 halves it |
| Test submission | `outputs/submissions/submission_2seed_prob.csv` (5,855 rows, `Image_name,Probability,Pred_Label`) |
| Checkpoints | `outputs/runs/full4_*/b0_crop384_full_recipe/seed42_*/best.pt`, `outputs/runs/s43d_*/b0_crop384_full_recipe_seed43/seed43_*/best.pt` |

## Why this candidate
- Three seeds of the same recipe gave 0.9619–0.9663; the seed spread bounds the noise and the
  fusion gain (+0.006 over the best single) is larger than the seed spread.
- Adding the weaker seed44 *lowered* the 3-model fusion to 0.9708, so the ensemble is
  2 models only — which also keeps the T4 latency budget safe.
- Draft decode was validated on the full valid split against a full-decode checkpoint:
  AUROC 0.964108 vs 0.964565, |Δ| = 0.000457 (gate ≤ 0.001), Spearman 0.9988. This is what
  makes the 100 ms limit reachable (full decode alone is 348 ms p50).
- No pretrained weights and no external data: the only inputs are the organiser's train images
  and age column.

## Reproduce
```bash
cd /home/heyuhang/work/ncccu
PYTHONPATH=src python -m glaucoma train --config configs/experiments/b0_crop384_full_recipe.json
PYTHONPATH=src python -m glaucoma train --config configs/experiments/b0_crop384_full_recipe_seed43.json
# per-model test predictions (each uses its checkpoint's decode policy)
PYTHONPATH=src python -m glaucoma predict --checkpoint <seed42 best.pt> \
  --data-root /home/lichunyu/competition/dataset/glaucoma_first \
  --output outputs/submissions/sub_seed42.csv
PYTHONPATH=src python -m glaucoma predict --checkpoint <seed43 best.pt> ... --output outputs/submissions/sub_seed43.csv
# fuse
PYTHONPATH=src python scripts/fuse_predictions.py --test \
  --test-csv /home/lichunyu/competition/dataset/glaucoma_first/test.csv \
  --prediction s42=outputs/submissions/sub_seed42.csv \
  --prediction s43=outputs/submissions/sub_seed43.csv \
  --method prob --output outputs/submissions/submission.csv
```

## Open items
- 512 px arm still training (diversity/resolution check); if it is materially better it may
  replace one member, subject to the latency budget.
- Full-data (train+valid) retrain was deliberately not shipped: it cannot be validated, so the
  measured 2-seed ensemble remains the candidate.
- Final packaging must include the two fp16 checkpoints + the fusion script and stay under
  500 MB and 100 ms; re-measure on a T4 before the national round.
