"""Frozen-feature extraction, linear probe and collapse diagnostics."""

import numpy as np
import torch

from ..metrics import binary_metrics


@torch.no_grad()
def extract_features(encoder, loader, device, amp=True):
    """Run a frozen encoder over a loader; returns (features, labels, names)."""
    encoder.eval()
    features, labels, names = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            output = encoder(images)
        features.append(output.float().cpu().numpy())
        if "label" in batch:
            labels.append(batch["label"].cpu().numpy())
        names.extend(batch["name"])
    if not features:
        raise ValueError("Empty loader passed to extract_features")
    result = np.concatenate(features, axis=0)
    target = np.concatenate(labels, axis=0).astype(np.int64) if labels else None
    return result, target, names


def fit_probe(features, labels, C=1.0):
    """StandardScaler + class-balanced logistic regression on frozen features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("Expected [N, D] features and [N] labels")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The probe requires both classes")
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(class_weight="balanced", max_iter=2000, C=C)
    classifier.fit(scaler.transform(x), y)
    return scaler, classifier


def evaluate_probe(scaler, classifier, features, labels):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    probabilities = classifier.predict_proba(scaler.transform(x))[:, 1]
    return binary_metrics(y, probabilities)


def probe_metrics(train_features, train_labels, eval_features, eval_labels, C=1.0):
    scaler, classifier = fit_probe(train_features, train_labels, C=C)
    return evaluate_probe(scaler, classifier, eval_features, eval_labels)


def effective_rank(features):
    """Entropy of the centred singular-value spectrum (collapse monitor)."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("effective_rank requires a [N, D] matrix with N >= 2")
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    total = singular.sum()
    if not total:
        return 0.0
    probabilities = singular[singular > 0] / total
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def feature_std(features):
    """Mean per-dimension standard deviation (second collapse monitor)."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("feature_std requires a [N, D] matrix with N >= 2")
    return float(x.std(axis=0).mean())
