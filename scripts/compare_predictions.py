"""Paired bootstrap comparison of two valid-prediction CSV files.

The unit of resampling is the image, stratified by label: each replicate draws
exactly ``n_pos`` positives and ``n_neg`` negatives with replacement and scores
both models on the same rows. The verdict encodes the pre-declared minimum
effect: a win requires ``Delta AUROC >= 0.003`` *and* a 95% percentile CI lower
bound above zero. Everything below that is noise, not a result.

``--group-by-age-pair`` is a sensitivity check. The dataset has no patient ID,
so same-age adjacent rows (likely fellow eyes) are resampled as clusters. This
reduces the false-confidence of the image-level CI but is **not** a corrected
patient-level interval; quote it as a sensitivity bound, never as the primary
result.

Usage:
    python scripts/compare_predictions.py new.csv ref.csv
    python scripts/compare_predictions.py new.csv ref.csv --reps 10000
    python scripts/compare_predictions.py new.csv ref.csv \
        --group-by-age-pair --data-root /path/to/glaucoma_first
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from glaucoma.fusion import align_by_name, group_same_age_pairs, paired_bootstrap

MINIMUM_EFFECT = 0.003
METRIC_LABELS = {"auroc": "ΔAUROC", "ap": "ΔAP", "auprc_trapz": "ΔAUPRC-trapz"}


def read_predictions(path):
    """Read Image_name/True_Label and prefer the raw Logit over Probability."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty prediction file: {path}")
    fields = rows[0].keys()
    if "Image_name" not in fields or "True_Label" not in fields:
        raise ValueError(f"{path} requires Image_name and True_Label")
    if "Logit" in fields:
        score_column, score_kind = "Logit", "logits"
    elif "Probability" in fields:
        score_column, score_kind = "Probability", "probabilities"
    else:
        raise ValueError(f"{path} requires a Logit or Probability column")
    names = [row["Image_name"] for row in rows]
    try:
        targets = [int(row["True_Label"]) for row in rows]
        scores = [float(row[score_column]) for row in rows]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: malformed label or score ({error})") from error
    return {"names": names, "targets": targets, score_kind: scores}


def read_valid_rows(root):
    """Read official valid.csv order and parsed ages (NaN when unparseable)."""
    path = Path(root) / "valid.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Image_name" not in rows[0] or "Age" not in rows[0]:
        raise ValueError(f"{path} requires Image_name and Age columns")
    names = [row["Image_name"] for row in rows]
    ages = []
    for row in rows:
        try:
            ages.append(float(row["Age"]))
        except (TypeError, ValueError):
            ages.append(float("nan"))
    return names, ages


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("new", help="New model valid_predictions.csv")
    parser.add_argument("ref", help="Reference model valid_predictions.csv")
    parser.add_argument("--reps", type=int, default=10000, help="Bootstrap replicates (default 10000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-by-age-pair", action="store_true",
                        help="Cluster-resample adjacent same-age rows (sensitivity only)")
    parser.add_argument("--data-root", help="Dataset root containing valid.csv (required with grouping)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.reps < 1:
        raise SystemExit("--reps must be positive")
    new = read_predictions(args.new)
    ref = read_predictions(args.ref)
    aligned = align_by_name([new, ref])
    new, ref = aligned
    if not np.array_equal(np.asarray(new["targets"]), np.asarray(ref["targets"])):
        raise SystemExit("The two prediction files disagree on True_Label")
    new_scores = np.asarray(new.get("logits", new.get("probabilities")), dtype=np.float64)
    ref_scores = np.asarray(ref.get("logits", ref.get("probabilities")), dtype=np.float64)
    groups = None
    if args.group_by_age_pair:
        if not args.data_root:
            raise SystemExit("--group-by-age-pair requires --data-root")
        # Group in valid.csv (official) order, then move ids into prediction
        # order by name so the clusters do not depend on file ordering.
        official_names, official_ages = read_valid_rows(args.data_root)
        official_groups = group_same_age_pairs(official_ages)
        group_by_name = dict(zip(official_names, official_groups))
        missing = [name for name in new["names"] if name not in group_by_name]
        if missing:
            raise SystemExit(f"valid.csv is missing prediction rows (e.g. {missing[0]})")
        groups = np.asarray([group_by_name[name] for name in new["names"]], dtype=np.int64)
    result = paired_bootstrap(new["targets"], new_scores, ref_scores,
                              n=args.reps, seed=args.seed, groups=groups)
    mode = "grouped by adjacent same-age rows (sensitivity)" if groups is not None else "image-level, stratified"
    print(f"Paired bootstrap: {mode}; reps={result['n']} seed={result['seed']} "
          f"rows={result['rows']} pos={result['n_pos']} neg={result['n_neg']}")
    print(f"{'metric':<14}{'delta':>10}{'95% CI low':>13}{'95% CI high':>13}{'P(delta<=0)':>13}")
    for name in ("auroc", "ap", "auprc_trapz"):
        block = result["metrics"][name]
        print(f"{METRIC_LABELS[name]:<14}{block['delta_mean']:>10.5f}{block['ci_low']:>13.5f}"
              f"{block['ci_high']:>13.5f}{block['p_le_zero']:>13.4f}")
    auroc = result["metrics"]["auroc"]
    win = auroc["ci_low"] > 0 and auroc["delta_mean"] >= MINIMUM_EFFECT
    verdict = "WIN" if win else "NO-EVIDENCE"
    print(f"Verdict: {verdict} (requires ΔAUROC >= {MINIMUM_EFFECT:.3f} and CI low > 0)")
    if groups is not None:
        print("Caveat: no patient IDs; the grouped CI is a sensitivity bound, not a corrected CI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
