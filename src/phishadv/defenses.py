"""Round-2 defenses: do any of them generalize beyond the attack they were trained on?

For one dataset and seed this trains, from scratch:

* the baselines (``gbm_raw``, ``cnn_raw``) and the domain-only models (the
  "drop every feature the phisher controls" extreme),
* standard adversarial training (``*_raw_at``: greedy trajectories, plus
  HotFlip for the CNN), the same recipe as :mod:`phishadv.experiment`,
* *mixed* adversarial training (``*_raw_mix``): each round also adds beam-search
  trajectories (width ``beam_width``, up to ``beam_budget`` edits). Training
  attacks only use the ``owner`` edit types, so the held-out edits stay held
  out at evaluation time,
* score-averaging GBM+CNN ensembles of the baseline, AT and mixed-AT pairs.

Every model is then attacked directly (adaptive black-box) with greedy (owner
edits, <=10), greedy with the held-out edits (<=10) and beam search over all 19
edit types (<=20). Run ``python -m phishadv.defenses --config phresh --seed 42``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .attacks import beam_blackbox, greedy_blackbox, hotflip_whitebox
from .data import load_dataset, split
from .experiment import CONFIGS, Config, fit, log, make_model, seed_everything
from .metrics import detection_metrics, evasion_rate, wilson_interval
from .models import CNNDetector

ATTACKS = ("greedy_blackbox", "greedy_heldout", "beam_extended")


class Ensemble:
    """Mean of the members' phishing probabilities (attacked as one black box)."""

    name = "ensemble"

    def __init__(self, *members):
        self.members = members
        self.view = "raw"

    def predict_proba(self, urls: list[str]) -> np.ndarray:
        return np.mean([m.predict_proba(urls) for m in self.members], axis=0)


def train_with_attacks(kind: str, base, train: pd.DataFrame, cfg: Config, mixed: bool):
    """Problem-space adversarial training. ``mixed=False`` reproduces
    :func:`phishadv.experiment.adversarial_training` for the raw view;
    ``mixed=True`` also adds owner-tier beam-search trajectories each round."""
    rng = np.random.default_rng(cfg.seed + 7)
    phish = train[train.y == 1].url.to_numpy()
    order = rng.permutation(len(phish))
    urls, y = train.url.tolist(), train.y.to_numpy()
    model, added, stats = base, [], []
    for r in range(cfg.at_rounds):
        chunk = phish[order[r * cfg.at_n_per_round: (r + 1) * cfg.at_n_per_round]].tolist()
        det = model.predict_proba(chunk) >= 0.5
        chunk = [u for u, d in zip(chunk, det) if d]
        res = greedy_blackbox(model, chunk, max_steps=cfg.budget, seed=cfg.seed + 100 + r)
        new = [u for tr in res.trace for u in tr[1:]]
        if isinstance(model, CNNDetector):
            hres = hotflip_whitebox(model, chunk[: len(chunk) // 2], max_flips=cfg.budget,
                                    structural=True, seed=cfg.seed + 200 + r)
            new += [a for a, o in zip(hres.adversarial, hres.original) if a != o]
        row = {"round": r + 1, "attacked": len(chunk), "train_greedy_evasion": evasion_rate(res.success)}
        if mixed:
            bres = beam_blackbox(model, chunk, max_steps=cfg.beam_budget, beam=cfg.beam_width,
                                 seed=cfg.seed + 300 + r, tier="owner")
            bnew = [u for tr in bres.trace for u in tr[1:]]
            new += bnew
            row.update(train_beam_evasion=evasion_rate(bres.success), added_beam=len(bnew))
        added.extend(new)
        row["added"] = len(new)
        stats.append(row)
        log(f"  {'mixed ' if mixed else ''}AT {kind} round {r + 1}: {row}")
        model = fit(make_model(kind, "raw", cfg), urls + added, np.concatenate([y, np.ones(len(added), int)]))
    return model, stats


def run(cfg: Config, out: Path) -> dict:
    t0 = time.time()
    seed_everything(cfg.seed)
    torch.set_num_threads(int(os.environ.get("PHISHADV_THREADS", min(8, os.cpu_count() or 1))))
    out.mkdir(parents=True, exist_ok=True)
    sp = split(load_dataset(cfg.dataset, cfg.data), cfg.seed, max_train=cfg.max_train, max_test=cfg.max_test,
               group_by_domain=cfg.group_split)
    tr_u, tr_y = sp.train.url.tolist(), sp.train.y.to_numpy()
    log(f"{cfg.name} seed {cfg.seed}: train={len(sp.train)} test={len(sp.test)}")

    models: dict[str, object] = {}
    for kind in ("gbm", "cnn"):
        for view in ("raw", "domain"):
            log(f"train {kind}_{view}")
            models[f"{kind}_{view}"] = fit(make_model(kind, view, cfg), tr_u, tr_y)
    at_stats = {}
    for kind in ("gbm", "cnn"):
        for suffix, mixed in (("at", False), ("mix", True)):
            name = f"{kind}_raw_{suffix}"
            log(f"adversarial training {name}")
            models[name], at_stats[name] = train_with_attacks(kind, models[f"{kind}_raw"], sp.train, cfg, mixed)
    for suffix in ("raw", "raw_at", "raw_mix"):
        models[f"ens_{suffix}"] = Ensemble(models[f"gbm_{suffix}"], models[f"cnn_{suffix}"])

    te_u, te_y = sp.test.url.tolist(), sp.test.y.to_numpy()
    va_u, va_y = sp.val.url.tolist(), sp.val.y.to_numpy()
    clean = []
    for name, m in models.items():
        row = {"model": name, **detection_metrics(te_y, m.predict_proba(te_u))}
        row["val_f1"] = detection_metrics(va_y, m.predict_proba(va_u))["f1"]
        clean.append(row)
        log(f"  clean {name}: f1={row['f1']:.4f} fpr={row['fpr']:.4f}")
    pd.DataFrame(clean).to_csv(out / "clean_metrics.csv", index=False)

    # Same 1,000-URL attack sample as phishadv.experiment for this seed.
    rng = np.random.default_rng(cfg.seed)
    phish_test = sp.test[sp.test.y == 1].url.to_numpy()
    sample = phish_test[rng.choice(len(phish_test), size=min(cfg.n_attack, len(phish_test)),
                                   replace=False)].tolist()
    rob, per_url = [], []
    for name, m in models.items():
        det = m.predict_proba(sample) >= 0.5
        idx = np.flatnonzero(det)
        targets = [sample[i] for i in idx]
        runs = {"greedy_blackbox": lambda: greedy_blackbox(m, targets, max_steps=cfg.budget, seed=cfg.seed)}
        if getattr(m, "view", "raw") != "domain":
            runs["greedy_heldout"] = lambda: greedy_blackbox(m, targets, max_steps=cfg.budget, seed=cfg.seed,
                                                             tier="heldout")
            runs["beam_extended"] = lambda: beam_blackbox(m, targets, max_steps=cfg.beam_budget,
                                                          beam=cfg.beam_width, seed=cfg.seed)
        for aname, fn in runs.items():
            ta = time.time()
            res = fn()
            k, n = int(res.success.sum()), len(res.success)
            lo, hi = wilson_interval(k, n)
            rob.append({"model": name, "attack": aname, "n_sample": len(sample), "n_initially_detected": n,
                        "evaded": k, "evasion_rate": evasion_rate(res.success), "evasion_ci95_low": lo,
                        "evasion_ci95_high": hi, "seconds": round(time.time() - ta, 1)})
            # Per-URL outcomes (index into the attack sample, no URLs) for paired tests between defenses.
            per_url += [{"model": name, "attack": aname, "i": int(i), "evaded": int(e), "steps": float(st)}
                        for i, e, st in zip(idx, res.success, res.steps)]
            log(f"  {name:12s} {aname:16s} evasion={evasion_rate(res.success):.3f} (n={n})")
    pd.DataFrame(rob).to_csv(out / "robustness.csv", index=False)
    pd.DataFrame(per_url).to_csv(out / "per_url.csv.gz", index=False)
    summary = {"config": asdict(cfg), "torch_threads": torch.get_num_threads(), "clean": clean,
               "robustness": rob, "adversarial_training": at_stats, "runtime_seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done in {time.time() - t0:.0f}s -> {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=sorted(CONFIGS), default="phresh")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = replace(CONFIGS[args.config], seed=args.seed)
    run(cfg, Path(args.out or f"results/defenses/{cfg.name}-seed{cfg.seed}"))


if __name__ == "__main__":
    main()
