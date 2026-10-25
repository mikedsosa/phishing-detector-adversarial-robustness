"""Fast GBM-only preview of the domain-grouped PhiUSIIL split (one seed).

Same data, split, sample and attack settings as ``phishadv.experiment --config
full-grouped``; trains only the GBM baseline and its adversarially trained
version. The full experiment (all models) supersedes this when it finishes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phishadv.attacks import beam_blackbox, greedy_blackbox
from phishadv.data import load_dataset, split
from phishadv.experiment import CONFIGS, adversarial_training, fit, make_model, seed_everything
from phishadv.metrics import detection_metrics, wilson_interval

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
cfg = CONFIGS["full-grouped"]
from dataclasses import replace  # noqa: E402
cfg = replace(cfg, seed=seed)
out = Path(f"results/phiusiil-grouped-gbm-preview/seed{seed}")
out.mkdir(parents=True, exist_ok=True)
seed_everything(seed)
df = load_dataset(cfg.dataset, cfg.data)
sp = split(df, seed, max_train=cfg.max_train, max_test=cfg.max_test, group_by_domain=True)
models = {"gbm_raw": fit(make_model("gbm", "raw", cfg), sp.train.url.tolist(), sp.train.y.to_numpy())}
models["gbm_raw_at"], _ = adversarial_training("gbm", "raw", models["gbm_raw"], sp.train, cfg)
te_u, te_y = sp.test.url.tolist(), sp.test.y.to_numpy()
clean = pd.DataFrame([{"model": n, **detection_metrics(te_y, m.predict_proba(te_u))} for n, m in models.items()])
clean.to_csv(out / "clean_metrics.csv", index=False)
rng = np.random.default_rng(seed)
phish = sp.test[sp.test.y == 1].url.to_numpy()
sample = phish[rng.choice(len(phish), size=min(cfg.n_attack, len(phish)), replace=False)].tolist()
rows = []
for n, m in models.items():
    targets = [u for u, p in zip(sample, m.predict_proba(sample)) if p >= 0.5]
    for a, fn in {"greedy_blackbox": lambda: greedy_blackbox(m, targets, max_steps=cfg.budget, seed=seed),
                  "greedy_heldout": lambda: greedy_blackbox(m, targets, max_steps=cfg.budget, seed=seed,
                                                            tier="heldout"),
                  "beam_extended": lambda: beam_blackbox(m, targets, max_steps=cfg.beam_budget,
                                                         beam=cfg.beam_width, seed=seed)}.items():
        r = fn()
        k = int(r.success.sum())
        lo, hi = wilson_interval(k, len(r.success))
        rows.append({"model": n, "attack": a, "n_sample": len(sample), "n_initially_detected": len(targets),
                     "evaded": k, "evasion_rate": k / max(1, len(targets)), "ci95_low": lo, "ci95_high": hi})
        print(json.dumps(rows[-1]), flush=True)
pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)
print("PREVIEW_DONE", flush=True)
