"""Prediction-level fusion and paired comparison utilities (CPU only).

Fusion here is *offline*: it consumes per-model logits/probabilities that were
already produced by inference. Nothing in this module imports torch or touches
images, so a fusion can be reproduced from a saved CSV after the server run
directory is gone.

Ranking metrics and the fusion rules are chosen for AUROC/AUPRC, not for
calibration:

* ``logit`` -- standardise each model's logit with train-fitted statistics
  (``mu``/``sd``) and average. It is pointwise (one image can be scored on its
  own), label-free at evaluation time and removes the scale differences caused
  by ``pos_weight``, oversampling, mixup and label smoothing, so it is the
  default shipped rule.
* ``rank`` -- average per-model average-ranks. This is often slightly better
  for AUROC because it is invariant to any monotone rescaling, but it is
  *transductive*: the score of one image depends on the other images in the
  batch. It therefore cannot be used by a single-image judge and destroys
  calibrated probabilities; use it for offline sensitivity analysis only.
* ``prob`` -- average sigmoid(logit). Simple and calibration-friendly but the
  least robust when members were trained with different loss weighting.

All arithmetic is float64. 10k paired bootstrap resamples over ~5.8k images take
seconds on a laptop CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_auc_score)

METHODS = ("prob", "logit", "rank")
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def average_ranks(values):
    """1-based ranks with ties receiving their average rank.

    ``average_ranks([0.1, 0.2, 0.2, 0.4]) == [1, 2.5, 2.5, 4]``.  The stable
    sort plus the forward scan make tie handling deterministic.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("Expected a nonempty one-dimensional score vector")
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite")
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < ordered.size:
        stop = start + 1
        while stop < ordered.size and ordered[stop] == ordered[start]:
            stop += 1
        # Ranks start+1 .. stop (inclusive) share their mean.
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def rank_scores(values):
    """Map scores to ``(rank - 0.5) / n`` in ``(0, 1)``; ties stay tied."""
    values = np.asarray(values, dtype=np.float64)
    return (average_ranks(values) - 0.5) / values.size


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    # Numerically stable for large |x| without importing scipy.
    return np.where(values >= 0, 1.0 / (1.0 + np.exp(-values)),
                    np.exp(values) / (1.0 + np.exp(values)))


def logit(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

def fit_standardizer(values):
    """Return ``(mean, std)`` for a non-degenerate finite sample."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("Standardizer needs a nonempty one-dimensional sample")
    if not np.isfinite(values).all():
        raise ValueError("Standardizer sample must be finite")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not math.isfinite(std) or std <= 1e-6:
        raise ValueError("Standardizer sample is constant; the model cannot be fused")
    return mean, std


def apply_standardizer(values, mean, std):
    values = np.asarray(values, dtype=np.float64)
    mean, std = float(mean), float(std)
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 1e-6:
        raise ValueError("Invalid standardizer statistics")
    return (values - mean) / std


@dataclass(frozen=True)
class LogitStandardizer:
    """Train-fitted logit scale, reusable across valid/test inference."""

    mean: float
    std: float

    @classmethod
    def fit(cls, values):
        return cls(*fit_standardizer(values))

    def apply(self, values):
        return apply_standardizer(values, self.mean, self.std)

    transform = apply

    def to_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, values):
        if "mean" not in values or "std" not in values:
            raise ValueError("Standardizer requires mean and std")
        mean, std = float(values["mean"]), float(values["std"])
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 1e-6:
            raise ValueError("Invalid standardizer statistics")
        return cls(mean, std)


def standardise_logits(values, standardizer=None):
    """Fit on ``values`` when no standardizer is given, then apply.

    Returns ``(standardised, standardizer)``.  Passing a standardizer fitted on
    a *train* reference sample is what makes a valid/test score reproducible and
    label-free; fitting on valid is only acceptable for an explicitly
    label-leaking analysis run.
    """
    values = np.asarray(values, dtype=np.float64)
    if standardizer is None:
        standardizer = LogitStandardizer.fit(values)
    elif isinstance(standardizer, dict):
        standardizer = LogitStandardizer.from_dict(standardizer)
    elif isinstance(standardizer, (tuple, list)) and len(standardizer) == 2:
        standardizer = LogitStandardizer(float(standardizer[0]), float(standardizer[1]))
    if not isinstance(standardizer, LogitStandardizer):
        raise TypeError("standardizer must be a LogitStandardizer, (mean, std) or dict")
    return standardizer.apply(values), standardizer


# ---------------------------------------------------------------------------
# Alignment and fusion
# ---------------------------------------------------------------------------

def _validate_names(names, label="table"):
    names = list(names)
    if not names:
        raise ValueError(f"{label} has no image names")
    if len(set(names)) != len(names):
        seen, duplicates = set(), []
        for name in names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        raise ValueError(f"{label} has duplicate image names: {duplicates[:5]}")
    return names


def _validate_tables(tables):
    if not tables:
        raise ValueError("At least one prediction table is required")
    parsed = []
    for index, table in enumerate(tables):
        if not isinstance(table, (tuple, list)) or len(table) != 3:
            raise ValueError("Each prediction table must be (names, scores, targets)")
        names, scores, targets = table
        names = _validate_names(names, f"table {index}")
        scores = np.asarray(scores, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.int64)
        if scores.shape != (len(names),) or targets.shape != (len(names),):
            raise ValueError(f"Table {index} has mismatched lengths")
        if not np.isfinite(scores).all():
            raise ValueError(f"Table {index} contains non-finite scores")
        if not np.isin(targets, [0, 1]).all():
            raise ValueError(f"Table {index} targets must be binary")
        parsed.append((names, scores, targets))
    reference = parsed[0][0]
    for index, (names, _, targets) in enumerate(parsed[1:], start=1):
        if names != reference:
            missing = sorted(set(reference) - set(names))
            extra = sorted(set(names) - set(reference))
            if missing or extra:
                raise ValueError(
                    f"Table {index} name set differs (missing={missing[:5]}, extra={extra[:5]}); "
                    "align by Image_name before fusing")
            raise ValueError(
                f"Table {index} name order differs from table 0; the rows must be in the same order")
        if not np.array_equal(targets, parsed[0][2]):
            raise ValueError(f"Table {index} targets differ from table 0")
    return parsed


def _normalise_weights(weights, count):
    if weights is None:
        return np.full(count, 1.0 / count, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (count,):
        raise ValueError(f"Expected {count} weights, got {weights.shape}")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Weights must not sum to zero")
    return weights / total


def fuse_logits(logits_by_model, weights=None, standardizers=None):
    """Weighted mean of standardised logits; returns a logit-scale score."""
    tables = _validate_tables([(np.arange(len(scores)), scores, np.zeros(len(scores), dtype=int))
                               for scores in logits_by_model])
    scores = [table[1] for table in tables]
    weights = _normalise_weights(weights, len(scores))
    if standardizers is None:
        standardizers = [None] * len(scores)
    if len(standardizers) != len(scores):
        raise ValueError("One standardizer is required per model")
    fused = np.zeros_like(scores[0])
    for score, weight, standardizer in zip(scores, weights, standardizers):
        z, _ = standardise_logits(score, standardizer)
        fused += weight * z
    return fused


def fuse_ranks(logits_by_model, weights=None):
    """Weighted mean of per-model rank scores in ``(0, 1)``."""
    tables = _validate_tables([(np.arange(len(scores)), scores, np.zeros(len(scores), dtype=int))
                               for scores in logits_by_model])
    scores = [table[1] for table in tables]
    weights = _normalise_weights(weights, len(scores))
    fused = np.zeros_like(scores[0])
    for score, weight in zip(scores, weights):
        fused += weight * rank_scores(score)
    return fused


def fuse_probabilities(logits_by_model, weights=None):
    """Weighted mean of sigmoid(logit); returns a probability in [0, 1]."""
    tables = _validate_tables([(np.arange(len(scores)), scores, np.zeros(len(scores), dtype=int))
                               for scores in logits_by_model])
    scores = [table[1] for table in tables]
    weights = _normalise_weights(weights, len(scores))
    fused = np.zeros_like(scores[0])
    for score, weight in zip(scores, weights):
        fused += weight * sigmoid(score)
    return fused


def fuse(prediction_tables, weights=None, method="logit", standardizers=None,
         input_kind="logits"):
    """Fuse ``[(names, scores, targets), ...]`` tables.

    The tables must already be aligned by ``Image_name`` (same order, same
    targets). ``input_kind`` declares whether ``scores`` are logits (default) or
    probabilities; probabilities are converted to logits so all three methods
    share one input contract.

    Returns a fused score vector: logit scale for ``method="logit"``,
    probability/rank scale in ``(0, 1)`` for ``prob``/``rank``.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown fusion method: {method}")
    tables = _validate_tables(prediction_tables)
    if input_kind not in ("logits", "probabilities"):
        raise ValueError("input_kind must be 'logits' or 'probabilities'")
    scores = [table[1] for table in tables]
    if input_kind == "probabilities":
        if any(((s < 0) | (s > 1)).any() for s in scores):
            raise ValueError("input_kind='probabilities' requires scores in [0, 1]")
        scores = [logit(s) for s in scores]
    if method == "logit":
        return fuse_logits(scores, weights, standardizers)
    if method == "rank":
        return fuse_ranks(scores, weights)
    return fuse_probabilities(scores, weights)


def align_by_name(predictions, order=None):
    """Reorder prediction dicts to a common name order.

    Each dict needs a ``names`` list plus equally long array-like fields
    (``logits``, ``probabilities``, ``targets``, ``ages``). Missing, extra or
    duplicate names raise instead of silently dropping a member/row. With
    ``order=None`` the first prediction defines the order.
    """
    if not predictions:
        raise ValueError("At least one prediction table is required")
    for index, prediction in enumerate(predictions):
        if "names" not in prediction:
            raise ValueError(f"Prediction {index} has no 'names'")
        _validate_names(prediction["names"], f"prediction {index}")
    reference = list(order) if order is not None else list(predictions[0]["names"])
    _validate_names(reference, "reference order")
    reference_set = set(reference)
    for index, prediction in enumerate(predictions):
        seen = set(prediction["names"])
        missing = sorted(reference_set - seen)
        extra = sorted(seen - reference_set)
        if missing or extra:
            raise ValueError(
                f"Prediction {index} name set differs (missing={missing[:5]}, extra={extra[:5]})")
    aligned = []
    for prediction in predictions:
        position = {name: row for row, name in enumerate(prediction["names"])}
        indices = np.asarray([position[name] for name in reference], dtype=np.int64)
        new = {"names": reference}
        for key, value in prediction.items():
            if key == "names":
                continue
            if isinstance(value, (list, tuple)):
                value = list(value)
                if len(value) != len(indices):
                    raise ValueError(f"Field '{key}' length does not match names")
                new[key] = [value[int(index)] for index in indices]
            elif isinstance(value, np.ndarray):
                if value.shape[:1] != (len(indices),):
                    raise ValueError(f"Field '{key}' length does not match names")
                new[key] = value[indices]
            else:
                new[key] = value
        aligned.append(new)
    return aligned


# ---------------------------------------------------------------------------
# Metrics and paired bootstrap
# ---------------------------------------------------------------------------

def ranking_metrics(targets, scores):
    """AUROC, AP and trapezoidal AUPRC for a ranking score (no calibration)."""
    y = np.asarray(targets, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not y.size:
        raise ValueError("Expected nonempty, equally sized one-dimensional targets and scores")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Expected binary targets and finite scores")
    result = {"count": int(y.size), "positives": int(y.sum()),
              "auroc": None, "ap": None, "auprc_trapz": None}
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, s))
        prec, rec, _ = precision_recall_curve(y, s)
        result["auprc_trapz"] = float(auc(rec, prec))
        result["ap"] = float(average_precision_score(y, s))
    return result


def _metric_value(name, y, score):
    if name == "auroc":
        return float(roc_auc_score(y, score))
    if name == "ap":
        return float(average_precision_score(y, score))
    if name == "auprc_trapz":
        precision, recall, _ = precision_recall_curve(y, score)
        return float(auc(recall, precision))
    raise ValueError(f"Unknown metric: {name}")


def _draw_indices(y, groups, rng, max_attempts=100):
    if groups is None:
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        return np.concatenate((rng.choice(positives, size=positives.size, replace=True),
                               rng.choice(negatives, size=negatives.size, replace=True)))
    groups = np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError("groups must have one entry per row")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("Grouped bootstrap needs at least two groups")
    for _ in range(max_attempts):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if np.unique(y[indices]).size == 2:
            return indices
    raise ValueError("Grouped bootstrap could not draw a resample with both classes")


def paired_bootstrap(y, new, ref, n=10000, seed=42, groups=None,
                     metrics=("auroc", "ap", "auprc_trapz")):
    """Stratified image-level paired bootstrap of ``new - ref``.

    Every resample draws exactly ``n_pos`` positives and ``n_neg`` negatives
    with replacement (so prevalence is preserved and every replicate has both
    classes) and evaluates both models on the *same* rows. With ``groups``
    supplied, whole groups (e.g. adjacent same-age pairs) are resampled instead;
    that is a cluster bootstrap sensitivity check, not a patient-level CI,
    because the dataset has no patient IDs.
    """
    y = np.asarray(y, dtype=np.int64)
    new = np.asarray(new, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if y.ndim != 1 or new.shape != y.shape or ref.shape != y.shape or not y.size:
        raise ValueError("Expected nonempty, equally sized one-dimensional arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must be binary")
    if len(np.unique(y)) != 2:
        raise ValueError("Paired bootstrap needs both classes")
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise ValueError("groups must have one entry per row")
    rng = np.random.default_rng(seed)
    deltas = {name: np.empty(n, dtype=np.float64) for name in metrics}
    for replicate in range(n):
        indices = _draw_indices(y, groups, rng)
        y_b = y[indices]
        for name in metrics:
            deltas[name][replicate] = (_metric_value(name, y_b, new[indices])
                                       - _metric_value(name, y_b, ref[indices]))
    result = {
        "n": int(n),
        "rows": int(y.size),
        "n_pos": int(y.sum()),
        "n_neg": int(y.size - y.sum()),
        "seed": int(seed),
        "grouped": groups is not None,
        "metrics": {},
    }
    for name, values in deltas.items():
        result["metrics"][name] = {
            "delta_mean": float(values.mean()),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "p_win": float((values > 0).mean()),
            "p_le_zero": float((values <= 0).mean()),
        }
    return result


def paired_bootstrap_auroc(y, new, ref, n=10000, seed=42, groups=None):
    """Convenience wrapper returning the AUROC block (plan §6.1 contract)."""
    result = paired_bootstrap(y, new, ref, n=n, seed=seed, groups=groups, metrics=("auroc",))
    block = result["metrics"]["auroc"]
    return {"delta_mean": block["delta_mean"], "ci_low": block["ci_low"],
            "ci_high": block["ci_high"], "p_win": block["p_win"],
            "n": result["n"], "n_pos": result["n_pos"], "n_neg": result["n_neg"]}


# ---------------------------------------------------------------------------
# Grouping (for the grouped sensitivity bootstrap)
# ---------------------------------------------------------------------------

def group_same_age_pairs(ages, upper=100.0):
    """Group *adjacent* rows with the same finite age in ``(0, upper]``.

    Returns a list of integer group ids. Rows with a missing/invalid age are
    singletons. This mirrors the train-side pair construction (DATA_AUDIT:
    ~22k adjacent same-age pairs) and is used only for the grouped bootstrap
    sensitivity check.
    """
    ages = np.asarray(ages, dtype=np.float64)
    if ages.ndim != 1:
        raise ValueError("ages must be one-dimensional")
    groups = np.empty(ages.size, dtype=np.int64)
    current = -1
    previous = None
    for index, age in enumerate(ages):
        usable = math.isfinite(float(age)) and 0 < float(age) <= upper
        if usable and previous is not None and float(age) == previous:
            groups[index] = current
        else:
            current += 1
            groups[index] = current
            previous = float(age) if usable else None
    return groups.tolist()
