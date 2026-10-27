"""Read-only run table over ``outputs/**`` (metrics.json / valid predictions).

The scanner is deliberately tolerant: a run may be still training (no
``metrics.json`` yet), may have been produced by ``glaucoma evaluate`` (flat
metrics) or by ``glaucoma train`` (metrics nested under ``valid``), and may
predate a column. It never writes into a run directory and it never loads a
checkpoint.

Usage:
    python scripts/report_runs.py --root outputs
    python scripts/report_runs.py --root outputs --json
"""

import argparse
import csv
import json
from pathlib import Path
import sys

from glaucoma.fusion import ranking_metrics

PREDICTION_FILES = ("valid_predictions.csv", "predictions.csv")
TABLE_COLUMNS = ("experiment", "seed", "best_epoch", "auroc", "auprc_trapz", "ap",
                 "sens_at_spec95", "wall_time_s", "path")


def _read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _read_predictions(path):
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, ValueError):
        return None
    if not rows or "Image_name" not in rows[0] or "True_Label" not in rows[0]:
        return None
    score_column = "Logit" if "Logit" in rows[0] else "Probability"
    if score_column not in rows[0]:
        return None
    try:
        targets = [int(row["True_Label"]) for row in rows]
        scores = [float(row[score_column]) for row in rows]
    except (TypeError, ValueError):
        return None
    return targets, scores


def _metrics_block(metrics):
    if not isinstance(metrics, dict):
        return {}
    valid = metrics.get("valid")
    if isinstance(valid, dict):
        return valid
    return metrics


def _wall_time(run_dir, metrics):
    history = _read_json(run_dir / "history.json")
    if isinstance(history, list) and history:
        seconds = [row.get("seconds") for row in history if isinstance(row, dict)]
        seconds = [float(value) for value in seconds if isinstance(value, (int, float))]
        if seconds:
            return sum(seconds)
    for key in ("wall_time_s", "seconds"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _seed_from_name(name):
    if name.startswith("seed"):
        head = name[4:].split("_", 1)[0]
        try:
            return int(head)
        except ValueError:
            return None
    return None


def _row_for_run(run_dir):
    metrics = _read_json(run_dir / "metrics.json")
    if not isinstance(metrics, dict):
        metrics = {}
    block = _metrics_block(metrics)
    config = _read_json(run_dir / "config.json")
    if not isinstance(config, dict):
        config = {}
    predictions = None
    for name in PREDICTION_FILES:
        candidate = run_dir / name
        if candidate.is_file():
            predictions = _read_predictions(candidate)
            if predictions is not None:
                break
    if not block and predictions is None:
        return None
    if predictions is not None and not block:
        block = ranking_metrics(predictions[0], predictions[1])
    history = _read_json(run_dir / "history.json")
    history = [row for row in history if isinstance(row, dict)] if isinstance(history, list) else []
    candidates = []
    for epoch in history:
        if type(epoch.get("epoch")) is not int:
            continue
        for key in ("valid", "valid_raw"):
            value = epoch.get(key, {})
            if isinstance(value.get("auroc"), (int, float)) and isinstance(value.get("auprc_trapz"), (int, float)):
                candidates.append((value["auroc"], value["auprc_trapz"], epoch["epoch"], key))
    history_best = max(candidates, key=lambda item: item[:2]) if candidates else None
    best_epoch = metrics.get("best_epoch")
    if best_epoch is None and history_best is not None:
        # An evaluate run writes flat metrics, without the training epoch. Infer
        # it only when both selection metrics match the retained history.
        if all(isinstance(block.get(key), (int, float)) and abs(block[key] - value) < 1e-8
               for key, value in zip(("auroc", "auprc_trapz"), history_best[:2])):
            best_epoch = history_best[2]
    completed = metrics.get("completed_epochs")
    epochs = [row["epoch"] for row in history if type(row.get("epoch")) is int]
    if completed is None and epochs:
        completed = max(epochs) + 1
    planned = config.get("train", {}).get("epochs")
    complete = metrics.get("training_complete")
    if complete is None and completed is not None and planned is not None:
        complete = completed >= planned
    experiment = config.get("experiment") or run_dir.parent.name or run_dir.name
    seed = config.get("seed")
    if seed is None:
        seed = _seed_from_name(run_dir.name)
    return {
        "experiment": str(experiment),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_epoch_one_based": None if best_epoch is None else best_epoch + 1,
        "completed_epochs": completed,
        "planned_epochs": planned,
        "training_complete": complete,
        "history_best_auroc": None if history_best is None else history_best[0],
        "auroc": block.get("auroc"),
        "auprc_trapz": block.get("auprc_trapz"),
        "ap": block.get("ap"),
        "sens_at_spec95": block.get("sensitivity_at_specificity95"),
        "wall_time_s": _wall_time(run_dir, metrics),
        "path": str(run_dir),
    }


def scan_runs(root):
    root = Path(root)
    rows, seen = [], set()
    if not root.exists():
        return rows
    for metrics_path in sorted(root.rglob("metrics.json")):
        run_dir = metrics_path.parent.resolve()
        if run_dir in seen:
            continue
        seen.add(run_dir)
        row = _row_for_run(run_dir)
        if row is not None:
            rows.append(row)
    for name in PREDICTION_FILES:
        for prediction_path in sorted(root.rglob(name)):
            run_dir = prediction_path.parent.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            row = _row_for_run(run_dir)
            if row is not None:
                rows.append(row)
    return rows


def _as_float(value):
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def sort_rows(rows, key="auroc"):
    if key == "experiment":
        return sorted(rows, key=lambda row: (row["experiment"], row["seed"] if row["seed"] is not None else -1))
    if key == "ap":
        return sorted(rows, key=lambda row: (-_as_float(row["ap"]), row["experiment"]))
    return sorted(rows, key=lambda row: (-_as_float(row["auroc"]), row["experiment"],
                                         row["seed"] if row["seed"] is not None else -1))


def _format(row):
    def number(value, digits=4):
        return "—" if value is None else f"{value:.{digits}f}"

    wall = "—" if row["wall_time_s"] is None else f"{row['wall_time_s']:.0f}"
    return [row["experiment"], "—" if row["seed"] is None else str(row["seed"]),
            "—" if row["best_epoch_one_based"] is None else str(row["best_epoch_one_based"]),
            number(row["auroc"]), number(row["auprc_trapz"]), number(row["ap"]),
            number(row["sens_at_spec95"]), wall]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="outputs", help="Directory searched recursively (default: outputs)")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    parser.add_argument("--sort", choices=("auroc", "ap", "experiment"), default="auroc")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = sort_rows(scan_runs(args.root), args.sort)
    if args.json:
        print(json.dumps(rows, indent=2, allow_nan=False))
        return 0
    if not rows:
        print(f"No runs with metrics.json or valid predictions under {args.root}", file=sys.stderr)
        return 0
    widths = [max(len("experiment"), *(len(row["experiment"]) for row in rows))]
    widths += [len(column) for column in TABLE_COLUMNS[1:-1]]
    header = ["experiment", "seed", "epoch(1b)", "auroc", "auprc_trapz", "ap", "sens@s95", "wall_s"]
    print("  ".join(title.ljust(width) for title, width in zip(header, widths)))
    for row in rows:
        cells = _format(row)
        print("  ".join(cell.ljust(width) for cell, width in zip(cells, widths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
