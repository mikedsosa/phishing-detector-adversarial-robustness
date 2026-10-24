"""End-to-end experiment: train detectors, attack them, train defenses, report.

Run ``python -m phishadv.experiment --config full`` (or ``make reproduce``).
All randomness is seeded; outputs go to ``results/`` (or ``--out``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import __version__
from .attacks import greedy_blackbox, hotflip_whitebox, random_blackbox, shortener_wrap
from .data import RAW_CSV, SAMPLE_CSV, load_urls, split
from .feature_space import hopskipjump_feature_space
from .metrics import detection_metrics, detection_under_attack, evasion_rate
from .models import CNNDetector, GBMDetector


@dataclass
class Config:
    name: str
    data: str
    seed: int = 42
    max_train: int | None = None
    cnn_epochs: int = 3
    n_attack: int = 1000  # phishing test URLs sampled for attacks
    budget: int = 10  # max perturbations per URL
    at_rounds: int = 2
    at_n_per_round: int = 4000  # training phishing URLs attacked per AT round
    n_hsj: int = 200
    hsj_max_iter: int = 20


CONFIGS = {
    "full": Config(name="full", data=str(RAW_CSV)),
    "smoke": Config(name="smoke", data=str(SAMPLE_CSV), max_train=None, cnn_epochs=1,
                    n_attack=40, budget=4, at_rounds=1, at_n_per_round=60, n_hsj=8,
                    hsj_max_iter=3),
}
BUDGETS = list(range(0, 11))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_model(kind: str, view: str, cfg: Config):
    if kind == "gbm":
        return GBMDetector(view=view, seed=cfg.seed)
    return CNNDetector(view=view, seed=cfg.seed, epochs=cfg.cnn_epochs)


def fit(model, urls, y):
    if isinstance(model, CNNDetector):
        return model.fit(urls, y, log=log)
    return model.fit(urls, y)


def adversarial_training(kind: str, view: str, base, train: pd.DataFrame, cfg: Config):
    """Iterative adversarial training (Goodfellow et al. 2014; Madry et al. 2017),
    adapted to the problem space: attack the current model on fresh training
    phishing URLs, add every URL on the attack trajectories (labelled phishing)
    to the training set, retrain from scratch, repeat."""
    rng = np.random.default_rng(cfg.seed + 7)
    phish = train[train.y == 1].url.to_numpy()
    order = rng.permutation(len(phish))
    urls, y = train.url.tolist(), train.y.to_numpy()
    model, added, stats = base, [], []
    for r in range(cfg.at_rounds):
        chunk = phish[order[r * cfg.at_n_per_round : (r + 1) * cfg.at_n_per_round]].tolist()
        det = model.predict_proba(chunk) >= 0.5
        chunk = [u for u, d in zip(chunk, det) if d]
        res = greedy_blackbox(model, chunk, max_steps=cfg.budget, seed=cfg.seed + 100 + r)
        new = [u for tr in res.trace for u in tr[1:]]
        if isinstance(model, CNNDetector) and view == "raw":
            hres = hotflip_whitebox(model, chunk[: len(chunk) // 2], max_flips=cfg.budget,
                                    structural=True, seed=cfg.seed + 200 + r)
            new += [a for a, o in zip(hres.adversarial, hres.original) if a != o]
        added.extend(new)
        stats.append({"round": r + 1, "attacked": len(chunk),
                      "train_evasion_rate": evasion_rate(res.success), "added": len(new)})
        log(f"  AT {kind}/{view} round {r + 1}: train evasion {evasion_rate(res.success):.3f}, "
            f"+{len(new)} adversarial URLs")
        model = fit(make_model(kind, view, cfg), urls + added, np.concatenate([y, np.ones(len(added), int)]))
    return model, stats


def run(cfg: Config, out: Path) -> dict:
    t0 = time.time()
    seed_everything(cfg.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    out.mkdir(parents=True, exist_ok=True)
    df = load_urls(cfg.data)
    sp = split(df, cfg.seed, max_train=cfg.max_train)
    log(f"data: {len(df)} unique URLs ({df.y.mean():.3f} phishing); "
        f"train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}")
    tr_u, tr_y = sp.train.url.tolist(), sp.train.y.to_numpy()
    structure = dataset_structure(df)
    log(f"dataset structure: {structure}")

    # ---------------- detectors (baselines + normalization/hardening defenses)
    models: dict[str, object] = {}
    for kind in ("gbm", "cnn"):
        for view in ("raw", "canon", "domain"):
            name = f"{kind}_{view}"
            log(f"train {name}")
            models[name] = fit(make_model(kind, view, cfg), tr_u, tr_y)

    # ---------------- adversarial training defenses
    at_stats = {}
    for kind, view in (("gbm", "raw"), ("cnn", "raw"), ("gbm", "canon")):
        name = f"{kind}_{view}_at"
        log(f"adversarial training {name}")
        models[name], at_stats[name] = adversarial_training(kind, view, models[f"{kind}_{view}"],
                                                            sp.train, cfg)

    # ---------------- clean metrics
    te_u, te_y = sp.test.url.tolist(), sp.test.y.to_numpy()
    va_u, va_y = sp.val.url.tolist(), sp.val.y.to_numpy()
    clean_rows = []
    for name, m in models.items():
        row = {"model": name, **detection_metrics(te_y, m.predict_proba(te_u))}
        row["val_f1"] = detection_metrics(va_y, m.predict_proba(va_u))["f1"]
        clean_rows.append(row)
    # A one-line "URL format" rule, to quantify how much of the task is a dataset shortcut.
    clean_rows.append({"model": "format_rule", **detection_metrics(te_y, [float(not is_bare_https_www(u))
                                                                           for u in te_u])})
    clean = pd.DataFrame(clean_rows)
    clean.to_csv(out / "clean_metrics.csv", index=False)
    log("clean metrics:\n" + clean[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr"]]
        .to_string(index=False))

    # ---------------- attacks
    rng = np.random.default_rng(cfg.seed)
    phish_test = sp.test[sp.test.y == 1].url.to_numpy()
    sample = phish_test[rng.choice(len(phish_test), size=min(cfg.n_attack, len(phish_test)),
                                   replace=False)].tolist()
    pd.DataFrame({"url": sample}).to_csv(out / "attack_sample_urls.csv", index=False)

    rob_rows, curves, q_curves, examples = [], [], [], []
    shortened = shortener_wrap(sample, seed=cfg.seed)
    adv_by_model = {}
    for name, m in models.items():
        p0 = m.predict_proba(sample)
        det = p0 >= 0.5
        targets = [u for u, d in zip(sample, det) if d]
        attacks = {
            "greedy_blackbox": lambda: greedy_blackbox(m, targets, max_steps=cfg.budget, seed=cfg.seed),
            "random_blackbox": lambda: random_blackbox(m, targets, max_steps=cfg.budget, seed=cfg.seed),
            "greedy_append_only": lambda: greedy_blackbox(m, targets, max_steps=cfg.budget, seed=cfg.seed,
                                                          tier="append_only"),
        }
        if isinstance(m, CNNDetector) and m.view == "raw":
            attacks["hotflip_whitebox"] = lambda: hotflip_whitebox(m, targets, max_flips=cfg.budget)
            attacks["hybrid_whitebox"] = lambda: hotflip_whitebox(m, targets, max_flips=cfg.budget,
                                                                  structural=True, seed=cfg.seed)
        for aname, fn in attacks.items():
            ta = time.time()
            res = fn()
            dt = time.time() - ta
            if aname == "greedy_blackbox":
                adv_by_model[name] = res
                for o, a, sc, st in zip(res.original, res.adversarial, res.final_score, res.steps):
                    if o != a:
                        examples.append({"model": name, "original": o, "adversarial": a,
                                         "final_p_phish": float(sc), "steps": float(st)})
                qb = [1, 5, 10, 20, 50, 100, 200, 500]
                q_curves.append({"model": name, "attack": aname,
                                 **{f"q<={b}": v for b, v in zip(qb, res.evasion_curve(qb, by="queries"))}})
            succ_q = res.queries[res.success] if aname == "greedy_blackbox" else np.array([])
            rob_rows.append({
                "model": name, "attack": aname, "n_sample": len(sample),
                "n_initially_detected": int(det.sum()),
                "evasion_rate": evasion_rate(res.success),
                "detection_under_attack": detection_under_attack(det, res.success),
                "median_steps_successful": (float(np.median(res.steps[res.success]))
                                            if res.success.any() else float("nan")),
                "median_queries_successful": float(np.median(succ_q)) if len(succ_q) else float("nan"),
                "seconds": round(dt, 1),
            })
            curves.append({"model": name, "attack": aname,
                           **{f"k={b}": v for b, v in zip(BUDGETS, res.evasion_curve(BUDGETS))}})
            log(f"  {name:14s} {aname:17s} evasion={evasion_rate(res.success):.3f} "
                f"(n={len(targets)}, {dt:.0f}s)")
        # Out-of-threat-model baseline: wrap the URL in a link shortener.
        ps = m.predict_proba([s for s, d in zip(shortened, det) if d])
        rob_rows.append({"model": name, "attack": "shortener_wrap", "n_sample": len(sample),
                         "n_initially_detected": int(det.sum()),
                         "evasion_rate": evasion_rate(ps < 0.5),
                         "detection_under_attack": detection_under_attack(det, ps < 0.5),
                         "median_steps_successful": 1.0 if (ps < 0.5).any() else float("nan"),
                         "median_queries_successful": float("nan"), "seconds": 0.0})
    rob = pd.DataFrame(rob_rows)
    rob.to_csv(out / "robustness.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "evasion_vs_perturbations.csv", index=False)
    pd.DataFrame(q_curves).to_csv(out / "evasion_vs_queries.csv", index=False)
    pd.DataFrame(examples).to_csv(out / "adversarial_examples_greedy.csv", index=False)

    # ---------------- transfer of black-box adversarial URLs between model families
    transfer = []
    for src in ("gbm_raw", "cnn_raw"):
        res = adv_by_model[src]
        adv = [a for a, s in zip(res.adversarial, res.success) if s]
        for dst in models:
            if dst == src or not adv:
                continue
            transfer.append({"source": src, "target": dst, "n": len(adv),
                             "target_evasion_rate": float((models[dst].predict_proba(adv) < 0.5).mean())})
    pd.DataFrame(transfer).to_csv(out / "transfer.csv", index=False)

    # ---------------- unconstrained feature-space baseline (ART HopSkipJump)
    gbm = models["gbm_raw"]
    hs_targets = [u for u in sample[: cfg.n_hsj * 2] if gbm.predict_proba([u])[0] >= 0.5][: cfg.n_hsj]
    legit_train = sp.train[sp.train.y == 0].url.to_numpy()
    xb = gbm.features(rng.choice(legit_train, size=min(3000, len(legit_train)), replace=False).tolist())
    log(f"HopSkipJump feature-space attack on gbm_raw (n={len(hs_targets)})")
    hsj = hopskipjump_feature_space(gbm, hs_targets, gbm.features(tr_u), xb,
                                    max_iter=cfg.hsj_max_iter, seed=cfg.seed)
    log(f"  {hsj}")

    # ---------------- greedy attack: which transformations were used
    op_counts = {}
    for name, res in adv_by_model.items():
        if name in ("gbm_raw", "cnn_raw"):
            op_counts[name] = transformation_usage(res)

    summary = {
        "config": asdict(cfg),
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "data": {"n_unique_urls": int(len(df)), "phishing_fraction": float(df.y.mean()),
                 "n_train": len(sp.train), "n_val": len(sp.val), "n_test": len(sp.test),
                 "structure": structure},
        "clean": clean.set_index("model").to_dict(orient="index"),
        "robustness": rob_rows,
        "adversarial_training": at_stats,
        "transfer": transfer,
        "hopskipjump_feature_space_gbm_raw": hsj,
        "greedy_transformation_usage": op_counts,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    from .report import write_report

    write_report(out)
    log(f"done in {time.time() - t0:.0f}s -> {out}")
    return summary


def is_bare_https_www(url: str) -> bool:
    """True for URLs of the form https://www.<domain>[/] with no path or query."""
    from .urls import parse

    p = parse(url)
    return (p.scheme == "https" and p.sub_labels[:1] == ["www"] and p.path in ("", "/")
            and not p.query)


def dataset_structure(df: pd.DataFrame) -> dict:
    """Per-class share of URLs with https, a leading www label, a path or a query."""
    from .urls import parse

    out = {}
    for label, name in ((0, "legitimate"), (1, "phishing")):
        parts = [parse(u) for u in df[df.y == label].url]
        n = max(len(parts), 1)
        out[name] = {
            "n": len(parts),
            "https": sum(p.scheme == "https" for p in parts) / n,
            "leading_www": sum(p.sub_labels[:1] == ["www"] for p in parts) / n,
            "non_root_path": sum(p.path not in ("", "/") for p in parts) / n,
            "query": sum(bool(p.query) for p in parts) / n,
            "bare_https_www": sum(p.scheme == "https" and p.sub_labels[:1] == ["www"]
                                  and p.path in ("", "/") and not p.query for p in parts) / n,
        }
    return out


def transformation_usage(res) -> dict[str, int]:
    """Classify each greedy step (successful attacks only) by what changed."""
    from .urls import parse

    counts: dict[str, int] = {}
    for tr, ok in zip(res.trace, res.success):
        if not ok:
            continue
        for a, b in zip(tr, tr[1:]):
            pa, pb = parse(a), parse(b)
            if pa.scheme != pb.scheme:
                k = "scheme->https"
            elif pa.subdomain != pb.subdomain:
                k = "subdomain edit" if "xn--" not in pb.subdomain or "xn--" in pa.subdomain else "homoglyph (IDN)"
            elif pa.path != pb.path:
                k = "path edit"
            else:
                k = "query/fragment edit"
            counts[k] = counts.get(k, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=sorted(CONFIGS), default="full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    out = Path(args.out or ("results" if cfg.name == "full" else f"results-{cfg.name}"))
    run(cfg, out)


if __name__ == "__main__":
    main()
