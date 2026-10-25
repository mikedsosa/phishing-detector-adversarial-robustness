"""Aggregate results across seeds and datasets -> results/SEEDS.md, results/seeds_*.csv, results/headline.png.

Run after ``make reproduce`` (``python -m phishadv.aggregate``). Seed 42 runs live in
``results/`` (PhiUSIIL) and ``results/phreshphish/``; extra seeds in ``results/seeds/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import wilson_interval  # noqa: E402
from .report import ATTACK_LABELS, DATASET_LABELS, LABELS  # noqa: E402

RUNS = {"phiusiil": ("results", "full"), "phreshphish": ("results/phreshphish", "phresh"),
        "phiusiil-grouped": ("results/phiusiil-grouped", "full-grouped")}
KEY_MODELS = ["gbm_raw", "cnn_raw", "gbm_raw_at", "cnn_raw_at", "gbm_canon", "cnn_canon", "gbm_canon_at"]
KEY_ATTACKS = ["greedy_blackbox", "greedy_heldout", "beam_extended", "hybrid_whitebox", "greedy_append_only"]


def find_runs(root: Path = Path(".")) -> dict[str, list[Path]]:
    out = {}
    for ds, (main, cfg) in RUNS.items():
        dirs = [root / main] + sorted((root / "results/seeds").glob(f"{cfg}-seed*"))
        out[ds] = [d for d in dirs if (d / "summary.json").exists()]
    return out


def collect(runs: dict[str, list[Path]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rob, clean, cross = [], [], []
    for ds, dirs in runs.items():
        for d in dirs:
            s = json.loads((d / "summary.json").read_text())
            seed = s["config"]["seed"]
            for r in s["robustness"]:
                n = int(r["n_initially_detected"])
                rob.append({"dataset": ds, "seed": seed, "model": r["model"], "attack": r["attack"], "n": n,
                            "k": int(round(r["evasion_rate"] * n)), "evasion_rate": r["evasion_rate"]})
            for m, c in s["clean"].items():
                clean.append({"dataset": ds, "seed": seed, "model": m, "f1": c["f1"], "fpr": c["fpr"],
                              "recall": c["recall"]})
            for c in s.get("cross_dataset", []):
                cross.append({"train_dataset": ds, "eval_dataset": c["eval_dataset"], "seed": seed,
                              "model": c["model"], "f1": c["f1"], "fpr": c["fpr"], "recall": c["recall"]})
    return pd.DataFrame(rob), pd.DataFrame(clean), pd.DataFrame(cross)


def summarize_rob(rob: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, m, a), g in rob.groupby(["dataset", "model", "attack"], sort=False):
        lo, hi = wilson_interval(int(g.k.sum()), int(g.n.sum()))
        rows.append({"dataset": ds, "model": m, "attack": a, "n_seeds": len(g),
                     "seeds": " ".join(str(x) for x in g.seed),
                     "per_seed": " ".join(f"{x:.3f}" for x in g.evasion_rate),
                     "mean": g.evasion_rate.mean(), "min": g.evasion_rate.min(), "max": g.evasion_rate.max(),
                     "pooled_k": int(g.k.sum()), "pooled_n": int(g.n.sum()), "pooled_ci95_low": lo,
                     "pooled_ci95_high": hi})
    return pd.DataFrame(rows)


def summarize_metric(df: pd.DataFrame, keys: list[str], col: str = "f1") -> pd.DataFrame:
    g = df.groupby(keys, sort=False)[col]
    return g.agg(n_seeds="count", mean="mean", min="min", max="max").reset_index()


def headline_chart(srob: pd.DataFrame, out: Path) -> None:
    models = ["gbm_raw", "cnn_raw", "gbm_raw_at", "cnn_raw_at"]
    attacks = ["greedy_blackbox", "greedy_heldout", "beam_extended"]
    colors = ["#4a6fa5", "#9db4d3", "#1f3b5e"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, ds in zip(axes, ["phiusiil", "phreshphish"]):
        d = srob[srob.dataset == ds].set_index(["model", "attack"])
        x = np.arange(len(models))
        for j, (a, c) in enumerate(zip(attacks, colors)):
            vals, err_lo, err_hi = [], [], []
            for m in models:
                if (m, a) in d.index:
                    r = d.loc[(m, a)]
                    vals.append(100 * r["mean"])
                    err_lo.append(100 * (r["mean"] - r["min"]))
                    err_hi.append(100 * (r["max"] - r["mean"]))
                else:
                    vals.append(np.nan)
                    err_lo.append(0)
                    err_hi.append(0)
            ax.bar(x + (j - 1) * 0.26, vals, 0.26, color=c, label=ATTACK_LABELS[a],
                   yerr=[err_lo, err_hi], capsize=3, error_kw={"lw": 0.8})
        ax.set_xticks(x, [LABELS[m].replace(" (baseline)", "\n(baseline)").replace(" + ", "\n+ ") for m in models],
                      fontsize=8)
        nseeds = int(d["n_seeds"].max()) if len(d) else 0
        sub = f"mean of {nseeds} seeds; bars = min-max" if nseeds > 1 else "1 seed"
        ax.set_title(f"{DATASET_LABELS[ds]} ({sub})")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Evasion rate (% of detected phishing URLs)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "headline.png", dpi=150)
    plt.close(fig)


def write(root: Path = Path(".")) -> None:
    out = root / "results"
    runs = find_runs(root)
    rob, clean, cross = collect(runs)
    srob = summarize_rob(rob)
    sclean = summarize_metric(clean, ["dataset", "model"])
    scross = summarize_metric(cross, ["train_dataset", "eval_dataset", "model"]) if len(cross) else pd.DataFrame()
    srob.to_csv(out / "seeds_robustness.csv", index=False)
    sclean.to_csv(out / "seeds_clean.csv", index=False)
    if len(scross):
        scross.to_csv(out / "seeds_cross_dataset.csv", index=False)
    headline_chart(srob, out)

    lines = ["# Results across seeds and datasets", "",
             "Generated by `python -m phishadv.aggregate` from these runs:", ""]
    for ds, dirs in runs.items():
        lines.append(f"- {DATASET_LABELS[ds]}: " + ", ".join(f"`{d}`" for d in dirs))
    lines += ["", "Each seed changes the data split (PhiUSIIL) or the train/validation split and training-set "
              "subsample (PhreshPhish, whose test split is fixed by time), model initialisation, the 1,000 "
              "attacked phishing URLs and the attack's random choices. *Mean [min-max]* is over seeds. The "
              "95% CI is a Wilson interval on the evasions pooled over all seeds; it reflects sampling error "
              "only, not the seed-to-seed spread.", "",
              "## Clean test F1", "", "| Model | " + " | ".join(DATASET_LABELS[d] for d in runs) + " |",
              "|---|" + "---|" * len(runs)]
    for m in KEY_MODELS + ["format_rule", "query_rule"]:
        cells = []
        for ds in runs:
            r = sclean[(sclean.dataset == ds) & (sclean.model == m)]
            cells.append(f"{r['mean'].iloc[0]:.4f} [{r['min'].iloc[0]:.4f}-{r['max'].iloc[0]:.4f}]" if len(r)
                         else "-")
        lines.append(f"| {LABELS.get(m, m)} | " + " | ".join(cells) + " |")
    for ds in runs:
        lines += ["", f"## Evasion rates: {DATASET_LABELS[ds]}", "",
                  "| Model | Attack | Mean [min-max] over seeds | Pooled 95% CI | Per seed |", "|---|---|---|---|---|"]
        d = srob[srob.dataset == ds]
        for m in KEY_MODELS:
            for a in KEY_ATTACKS:
                r = d[(d.model == m) & (d.attack == a)]
                if not len(r):
                    continue
                r = r.iloc[0]
                lines.append(f"| {LABELS.get(m, m)} | {ATTACK_LABELS[a]} | {r['mean']:.3f} [{r['min']:.3f}-"
                             f"{r['max']:.3f}] | [{r.pooled_ci95_low:.3f}, {r.pooled_ci95_high:.3f}] | "
                             f"{r.per_seed} |")
    if len(scross):
        lines += ["", "## Cross-dataset clean F1 (train on one dataset, test on the other's test split)", "",
                  "| Trained on | Tested on | Model | F1 mean [min-max] |", "|---|---|---|---|"]
        for _, r in scross.iterrows():
            if r.model in KEY_MODELS[:4] + ["format_rule", "query_rule"]:
                lines.append(f"| {DATASET_LABELS[r.train_dataset]} | {DATASET_LABELS[r.eval_dataset]} | "
                             f"{LABELS.get(r.model, r.model)} | {r['mean']:.4f} [{r['min']:.4f}-{r['max']:.4f}] |")
    lines += ["", "![Headline evasion rates](headline.png)", ""]
    (out / "SEEDS.md").write_text("\n".join(lines))
    print(f"wrote {out / 'SEEDS.md'}")


if __name__ == "__main__":
    write()
