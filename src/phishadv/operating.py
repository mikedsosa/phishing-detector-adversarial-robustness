"""Low base-rate evaluation and a low-FPR operating point -> results/operating/<dataset>/.

Uses saved seed-42 models (``models/phreshphish`` or ``models/phiusiil-grouped``).

* **Base rate.** Real traffic is mostly benign. Phishing test URLs are
  re-weighted so they make up 5%, 1%, 0.5% or 0.05% of the test set, and we
  report average precision (AP) and precision at 90% recall at each rate.
  Re-weighting keeps every URL, so the ROC is unchanged and only precision moves.
* **Operating point.** The threshold is chosen on the validation split so that
  validation FPR is at most 0.1% (and 1%), then the test FPR / recall are
  reported and the greedy, held-out and beam attacks are re-run at that threshold
  on the same 1,000-URL attack sample as the main experiment.

Run ``python -m phishadv.operating --dataset phreshphish``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

from .attacks import beam_blackbox, greedy_blackbox
from .data import PHRESH_CSV, RAW_CSV, load_dataset, split
from .experiment import CONFIGS
from .metrics import wilson_interval
from .models import load_detector

BASE_RATES = (0.05, 0.01, 0.005, 0.0005)
TARGET_FPRS = (0.01, 0.001)
MODELS = ("gbm_raw", "cnn_raw", "gbm_raw_at", "cnn_raw_at")
SETUPS = {"phreshphish": ("phresh", PHRESH_CSV, "models/phreshphish"),
          "phiusiil-grouped": ("full-grouped", RAW_CSV, "models/phiusiil-grouped")}


def weights_for_base_rate(y: np.ndarray, rate: float) -> np.ndarray:
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    w1 = rate / (1 - rate) * n0 / n1
    return np.where(y == 1, w1, 1.0)


def precision_at_recall(y, p, rate: float, recall: float = 0.9) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    i = int(np.searchsorted(tpr, recall))  # first point with TPR >= recall
    t, f = tpr[i], fpr[i]
    return float(rate * t / (rate * t + (1 - rate) * f)) if t + f else float("nan")


def threshold_at_fpr(y_val, p_val, target: float) -> float:
    """Smallest threshold whose validation FPR (pred = p >= t) is <= target."""
    neg = np.sort(np.asarray(p_val)[np.asarray(y_val) == 0])[::-1]
    k = int(np.floor(target * len(neg)))  # number of negatives allowed above the threshold
    return float(np.nextafter(neg[k], np.inf)) if k < len(neg) else 0.0


def run(dataset: str, root: Path = Path(".")) -> None:
    cfg_name, data, mdir = SETUPS[dataset]
    cfg = CONFIGS[cfg_name]
    out = root / "results/operating" / dataset
    out.mkdir(parents=True, exist_ok=True)
    sp = split(load_dataset(cfg.dataset, data), cfg.seed, max_train=cfg.max_train, max_test=cfg.max_test,
               group_by_domain=cfg.group_split)
    te_u, te_y = sp.test.url.tolist(), sp.test.y.to_numpy()
    va_u, va_y = sp.val.url.tolist(), sp.val.y.to_numpy()
    rng = np.random.default_rng(cfg.seed)
    phish_test = sp.test[sp.test.y == 1].url.to_numpy()
    sample = phish_test[rng.choice(len(phish_test), size=min(cfg.n_attack, len(phish_test)),
                                   replace=False)].tolist()
    base_rows, op_rows = [], []
    for name in MODELS:
        m, _ = load_detector(root / mdir / name)
        p_te, p_va = m.predict_proba(te_u), m.predict_proba(va_u)
        for r in BASE_RATES:
            base_rows.append({"model": name, "base_rate": r,
                              "average_precision": float(average_precision_score(
                                  te_y, p_te, sample_weight=weights_for_base_rate(te_y, r))),
                              "precision_at_90_recall": precision_at_recall(te_y, p_te, r)})
        p_s = m.predict_proba(sample)
        for target in (None, *TARGET_FPRS):
            t = 0.5 if target is None else threshold_at_fpr(va_y, p_va, target)
            pred = p_te >= t
            row = {"model": name, "target_val_fpr": target if target is not None else "default (t=0.5)",
                   "threshold": t, "test_fpr": float(pred[te_y == 0].mean()),
                   "test_recall": float(pred[te_y == 1].mean())}
            targets = [u for u, s in zip(sample, p_s) if s >= t]
            row["n_attacked"] = len(targets)
            for aname, fn in (
                ("greedy_blackbox", lambda: greedy_blackbox(m, targets, max_steps=cfg.budget, threshold=t,
                                                            seed=cfg.seed)),
                ("greedy_heldout", lambda: greedy_blackbox(m, targets, max_steps=cfg.budget, threshold=t,
                                                           seed=cfg.seed, tier="heldout")),
                ("beam_extended", lambda: beam_blackbox(m, targets, max_steps=cfg.beam_budget,
                                                        beam=cfg.beam_width, threshold=t, seed=cfg.seed)),
            ):
                res = fn()
                k, n = int(res.success.sum()), len(res.success)
                lo, hi = wilson_interval(k, n)
                row.update({f"{aname}_evasion": k / n if n else float("nan"),
                            f"{aname}_ci95_low": lo, f"{aname}_ci95_high": hi})
            op_rows.append(row)
            print(json.dumps(row, default=float), flush=True)
    pd.DataFrame(base_rows).to_csv(out / "base_rate.csv", index=False)
    pd.DataFrame(op_rows).to_csv(out / "operating_point.csv", index=False)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(SETUPS), default="phreshphish")
    run(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
