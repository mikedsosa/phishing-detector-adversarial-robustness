"""Detection and robustness metrics (phishing = positive class)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def detection_metrics(y_true, p_phish, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_phish, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auc = roc_auc_score(y, p) if len(set(y.tolist())) == 2 else float("nan")
    return {
        "n": int(len(y)),
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": float(auc),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def evasion_rate(success) -> float:
    """Fraction of attacked (initially detected) phishing URLs that evade."""
    s = np.asarray(success, dtype=bool)
    return float(s.mean()) if len(s) else float("nan")


def detection_under_attack(initially_detected, success) -> float:
    """Fraction of *all* sampled phishing URLs still detected after the attack.

    ``initially_detected`` is a bool mask over the full sample and ``success``
    is the per-URL evasion flag for the detected subset.
    """
    det = np.asarray(initially_detected, dtype=bool)
    s = np.asarray(success, dtype=bool)
    if det.sum() != len(s):
        raise ValueError("success must be aligned with the detected subset")
    return float((det.sum() - s.sum()) / max(len(det), 1))
