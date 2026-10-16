"""Ranking metrics; AP and trapezoidal PR area are deliberately separate."""

import numpy as np
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)


def binary_metrics(targets, probabilities):
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty, equally sized one-dimensional targets and scores")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1]")
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
    result = {"count": len(y), "positives": int(y.sum()), "auroc": None,
              "auprc_trapz": None, "ap": None,
              "confusion_at_0.5": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
              "sensitivity_at_0.5": float(tp / (tp + fn)) if tp + fn else None,
              "specificity_at_0.5": float(tn / (tn + fp)) if tn + fp else None,
              "sensitivity_at_specificity95": None, "threshold_at_specificity95": None}
    if y.sum():
        precision, recall, _ = precision_recall_curve(y, p)
        result.update(auprc_trapz=float(auc(recall, precision)), ap=float(average_precision_score(y, p)))
    if len(np.unique(y)) == 2:
        fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
        eligible = np.flatnonzero(fpr <= 0.05)
        index = eligible[np.argmax(tpr[eligible])]
        result.update(auroc=float(roc_auc_score(y, p)),
                      sensitivity_at_specificity95=float(tpr[index]),
                      threshold_at_specificity95=float(thresholds[index]) if np.isfinite(thresholds[index]) else None)
    return result


def age_stratified_metrics(targets, probabilities, ages):
    ages = np.asarray(ages, dtype=np.float64)
    valid = np.isfinite(ages) & (ages > 0) & (ages <= 100)
    masks = {"under50": valid & (ages < 50), "50to64": valid & (ages >= 50) & (ages < 65),
             "65plus": valid & (ages >= 65), "missing_or_invalid": ~valid}
    y, p = np.asarray(targets), np.asarray(probabilities)
    return {name: binary_metrics(y[mask], p[mask]) if mask.any() else {"count": 0}
            for name, mask in masks.items()}
