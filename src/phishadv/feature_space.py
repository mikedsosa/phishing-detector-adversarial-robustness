"""Unconstrained feature-space attack baseline using ART's HopSkipJump.

This is the classic "edit the feature vector" attack. It is included to show
the gap between feature-space and problem-space robustness (Pierazzi et al.,
2020): we measure how often the adversarial feature vectors could correspond
to *any* real URL at all, and how often they change features tied to the
registrable domain, which the attacker does not control.
"""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier

from .features import BINARY_FEATURES, INTEGER_FEATURES, LOCKED_FEATURES
from .models import GBMDetector


def hopskipjump_feature_space(model: GBMDetector, urls: list[str], x_train: np.ndarray,
                              x_benign: np.ndarray, max_iter: int = 20, max_eval: int = 1000,
                              init_eval: int = 100, seed: int = 0) -> dict:
    """``x_benign``: feature vectors of legitimate URLs; HopSkipJump starts from
    the nearest one (L2, min-max scaled) to each phishing vector."""
    names = model.featurizer.feature_names
    x = model.features(urls).astype(np.float32)
    lo = x_train.min(axis=0).astype(np.float32)
    hi = x_train.max(axis=0).astype(np.float32)
    hi = np.where(hi > lo, hi, lo + 1.0)

    def predict(z):
        p = model.clf.predict_proba(np.asarray(z, dtype=np.float32))
        return p

    clf = BlackBoxClassifier(predict, input_shape=(x.shape[1],), nb_classes=2,
                             clip_values=(lo, hi))
    attack = HopSkipJump(clf, targeted=False, max_iter=max_iter, max_eval=max_eval,
                         init_eval=init_eval, verbose=False)
    scale0 = np.where(hi > lo, hi - lo, 1.0)
    xb = x_benign[model.clf.predict_proba(x_benign)[:, 1] < 0.5].astype(np.float32)
    d = (((x[:, None, :] - xb[None, :, :]) / scale0) ** 2).sum(-1)
    x_init = xb[d.argmin(1)]
    np.random.seed(seed)
    x_adv = attack.generate(x=x, x_adv_init=x_init)
    p_adv = model.clf.predict_proba(x_adv)[:, 1]
    success = p_adv < 0.5
    changed = ~np.isclose(x_adv, x, atol=1e-6)
    int_idx = [i for i, n in enumerate(names) if n in INTEGER_FEATURES or n.startswith("tld=")]
    bin_idx = [i for i, n in enumerate(names) if n in BINARY_FEATURES or n.startswith("tld=")]
    lock_idx = [i for i, n in enumerate(names) if n in LOCKED_FEATURES or n.startswith("tld=")]
    integral = np.all(np.isclose(x_adv[:, int_idx], np.round(x_adv[:, int_idx]), atol=1e-6), axis=1)
    binary = np.all(np.isin(np.round(x_adv[:, bin_idx], 6), [0.0, 1.0]), axis=1)
    onehot_ok = np.isclose(x_adv[:, [i for i, n in enumerate(names) if n.startswith("tld=")]].sum(1), 1)
    realizable_types = integral & binary & onehot_ok
    touches_locked = changed[:, lock_idx].any(axis=1)
    s = success
    scale = np.where(hi > lo, hi - lo, 1.0)
    l2 = np.linalg.norm((x_adv - x) / scale, axis=1)
    nan = float("nan")

    def frac(mask: np.ndarray) -> float:
        return float(mask[s].mean()) if s.any() else nan

    return {
        "n_attacked": int(len(urls)),
        "evasion_rate": float(s.mean()),
        "median_features_changed_successful": float(np.median(changed[s].sum(1))) if s.any() else nan,
        "median_normalized_l2_successful": float(np.median(l2[s])) if s.any() else nan,
        "frac_successful_type_consistent": frac(realizable_types),
        "frac_successful_touching_locked_features": frac(touches_locked),
        "frac_successful_realizable_under_threat_model": frac(realizable_types & ~touches_locked),
    }
