"""Fuse 2..N saved prediction files offline (no GPU, no images).

Offline mode reads ``valid_predictions.csv``-shaped files
(``Image_name,True_Label,Logit,Probability``), aligns them by ``Image_name``,
fuses the scores and writes a fused predictions CSV plus a JSON report. The
default ``logit`` rule standardises each model with train-reference statistics
(``--reference``) so no valid/test label is used; without a reference the
standardiser is fitted on the supplied file and the report marks it
``input_leaky``.

Test mode (``--test``) reads a ``test.csv`` plus per-model test prediction
CSVs (``Image_name,Probability[,Pred_Label]``), fuses in ``test.csv`` order and
writes a submission-shaped CSV with ``Pred_Label = Probability >= 0.5``.

The rank rule is transductive: it ranks within the supplied set and must not be
used for a single-image judge.

Usage:
    python scripts/fuse_predictions.py \
        --prediction full=full.csv --prediction sampled=sampled.csv \
        --method logit --weights 1,1 --output fused.csv --report fused.json
    python scripts/fuse_predictions.py --test --test-csv test.csv \
        --prediction full=full_test.csv --prediction sampled=sampled_test.csv \
        --method logit --output submission.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from glaucoma.fusion import (LogitStandardizer, align_by_name, fuse, logit,
                             paired_bootstrap, ranking_metrics, sigmoid)


def parse_named_paths(values, option):
    parsed = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option} requires NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise SystemExit(f"{option} requires nonempty NAME and PATH: {value!r}")
        parsed.append((name, Path(path)))
    names = [name for name, _ in parsed]
    if len(set(names)) != len(names):
        raise SystemExit(f"Duplicate member name in {option}: {names}")
    return parsed


def _score_column(fields):
    if "Logit" in fields:
        return "Logit", "logits"
    if "Probability" in fields:
        return "Probability", "probabilities"
    raise ValueError("Prediction CSV requires a Logit or Probability column")


def read_offline(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty prediction file: {path}")
    if "Image_name" not in rows[0] or "True_Label" not in rows[0]:
        raise ValueError(f"{path} requires Image_name and True_Label")
    column, kind = _score_column(rows[0])
    names = [row["Image_name"] for row in rows]
    targets = [int(row["True_Label"]) for row in rows]
    values = [float(row[column]) for row in rows]
    if kind == "probabilities":
        values = logit(values).tolist()
    return {"names": names, "targets": targets, "logits": values}


def read_test(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty test prediction file: {path}")
    column, kind = _score_column(rows[0])
    values = [float(row[column]) for row in rows]
    if kind == "probabilities":
        values = logit(values).tolist()
    return {"names": [row["Image_name"] for row in rows], "logits": values}


def read_test_order(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "Image_name" not in (reader.fieldnames or []):
            raise ValueError(f"{path} requires an Image_name column")
        return [row["Image_name"] for row in reader]


def write_rows(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _parse_weights(text, count):
    if text is None:
        return None
    try:
        weights = [float(part) for part in text.split(",")]
    except ValueError as error:
        raise SystemExit(f"Bad --weights: {text!r}") from error
    if len(weights) != count:
        raise SystemExit(f"Expected {count} weights, got {len(weights)}")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise SystemExit("Weights must be nonnegative and not all zero")
    return weights


def _normalised(weights, count):
    values = np.asarray(weights if weights is not None else [1.0] * count, dtype=np.float64)
    return (values / values.sum()).tolist()


def _build_standardizers(members, references, scores, method):
    if method != "logit":
        return None, None
    fitted_on = "input_leaky"
    reference_scores = {}
    for name, path in references:
        reference_scores[name] = read_offline(path)["logits"]
    standardizers, metadata = [], {}
    for index, (name, _) in enumerate(members):
        if references and name not in reference_scores:
            raise SystemExit(f"--reference has no file for member {name!r}")
        sample = reference_scores[name] if references else scores[index]
        standardizer = LogitStandardizer.fit(sample)
        standardizers.append(standardizer)
        metadata[name] = {**standardizer.to_dict(),
                          "fitted_on": "reference" if references else fitted_on}
    if references:
        fitted_on = "reference"
    return standardizers, {"fitted_on": fitted_on, "members": metadata}


def _fused_output(method, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if method == "logit":
        return scores, sigmoid(scores)
    return logit(scores), scores


def run_offline(args, members, weights):
    predictions = [read_offline(path) for _, path in members]
    aligned = align_by_name(predictions)
    targets = np.asarray(aligned[0]["targets"], dtype=np.int64)
    names = aligned[0]["names"]
    scores = [np.asarray(prediction["logits"], dtype=np.float64) for prediction in aligned]
    standardizers, standardizer_report = _build_standardizers(members, args.reference, scores, args.method)
    fused = fuse([(names, score, targets) for score in scores],
                 weights=weights, method=args.method, standardizers=standardizers)
    fused_logits, fused_probabilities = _fused_output(args.method, fused)
    write_rows(args.output, ["Image_name", "True_Label", "Logit", "Probability"],
               [[name, int(target), format(float(score), ".10g"), format(float(probability), ".10g")]
                for name, target, score, probability in zip(names, targets, fused_logits, fused_probabilities)])
    per_model = {name: ranking_metrics(targets, score)
                 for (name, _), score in zip(members, scores)}
    fused_metrics = ranking_metrics(targets, fused)
    report = {
        "mode": "offline",
        "method": args.method,
        "members": [name for name, _ in members],
        "weights": dict(zip([name for name, _ in members], _normalised(weights, len(members)))),
        "standardizer": standardizer_report,
        "rows": int(len(names)),
        "positives": int(targets.sum()),
        "per_model": per_model,
        "fused": fused_metrics,
        "bootstrap_vs_best_single": None,
    }
    if args.bootstrap > 0:
        best_name, best_score = max(zip([name for name, _ in members], scores),
                                    key=lambda pair: ranking_metrics(targets, pair[1])["auroc"])
        report["bootstrap_vs_best_single"] = {
            "best_single": best_name,
            **paired_bootstrap(targets, fused, best_score,
                               n=args.bootstrap, seed=args.bootstrap_seed),
        }
    if args.report:
        write_json(args.report, report)
    print(f"Wrote {len(names)} fused rows to {args.output}; method={args.method}; "
          f"AUROC={fused_metrics['auroc']}")
    return 0


def run_test(args, members, weights):
    if not args.test_csv:
        raise SystemExit("--test requires --test-csv")
    predictions = [read_test(path) for _, path in members]
    order = read_test_order(args.test_csv)
    aligned = align_by_name(predictions, order=order)
    scores = [np.asarray(prediction["logits"], dtype=np.float64) for prediction in aligned]
    standardizers, standardizer_report = _build_standardizers(members, args.reference, scores, args.method)
    fused = fuse([(order, score, np.zeros(len(order), dtype=np.int64)) for score in scores],
                 weights=weights, method=args.method, standardizers=standardizers)
    _, probabilities = _fused_output(args.method, fused)
    write_rows(args.output, ["Image_name", "Probability", "Pred_Label"],
               [[name, format(float(probability), ".10g"), int(probability >= 0.5)]
                for name, probability in zip(order, probabilities)])
    if args.report:
        write_json(args.report, {
            "mode": "test",
            "method": args.method,
            "members": [name for name, _ in members],
            "weights": dict(zip([name for name, _ in members], _normalised(weights, len(members)))),
            "standardizer": standardizer_report,
            "rows": len(order),
        })
    print(f"Wrote {len(order)} submission rows to {args.output}; method={args.method}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=PATH",
                        help="Member prediction CSV; repeat for 2..N members")
    parser.add_argument("--method", choices=("prob", "logit", "rank"), default="logit")
    parser.add_argument("--weights", help="Comma-separated weights, normalised (default equal)")
    parser.add_argument("--reference", action="append", default=[], metavar="NAME=PATH",
                        help="Optional train-reference predictions for logit standardisation")
    parser.add_argument("--output", required=True, help="New fused CSV (exclusive creation)")
    parser.add_argument("--report", help="New JSON report (exclusive creation)")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Offline: paired-bootstrap reps of fused vs best single (0=off)")
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="Fuse test-split prediction CSVs")
    parser.add_argument("--test-csv", help="Official test.csv (defines row order)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    members = parse_named_paths(args.prediction, "--prediction")
    if len(members) < 2:
        raise SystemExit("Fusion requires at least two --prediction members")
    if args.bootstrap < 0:
        raise SystemExit("--bootstrap must be nonnegative")
    weights = _parse_weights(args.weights, len(members))
    if args.test:
        return run_test(args, members, weights)
    if args.bootstrap > 0 and not args.report:
        raise SystemExit("--bootstrap requires --report")
    return run_offline(args, members, weights)


if __name__ == "__main__":
    raise SystemExit(main())
