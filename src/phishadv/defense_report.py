"""Round-2 defense tables, paired tests and the hero figure.

Reads ``results/defenses/<config>-seed*/`` (from :mod:`phishadv.defenses`) and
writes ``results/defenses/DEFENSES.md``, ``defenses_summary.csv``,
``paired_tests.csv``, plus ``results/hero.png`` / ``results/hero_data.csv`` from
the main experiment runs (baseline and adversarially trained models, every seed).

*Caught under attack* is the share of the 1,000 sampled phishing URLs that the
model flags before the attack and still flags after it. Unlike the evasion
rate, its denominator is the same for every model. Paired tests compare two
models on the same URLs (same seed): an exact McNemar test on the URLs that
one model still catches and the other doesn't, pooled over seeds.

Run ``python -m phishadv.defense_report``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from .metrics import wilson_interval  # noqa: E402
from .report import DATASET_LABELS, LABELS  # noqa: E402

CONFIG_DATASET = {"phresh": "phreshphish", "full-grouped": "phiusiil-grouped", "full": "phiusiil"}
ATTACKS = {"none": "No attack", "greedy_blackbox": "Greedy (training edits), ≤10",
           "greedy_heldout": "Held-out edits, ≤10", "beam_extended": "Beam search, all edits, ≤20"}
ORDER = ["gbm_raw", "gbm_raw_at", "gbm_raw_mix", "gbm_domain", "cnn_raw", "cnn_raw_at", "cnn_raw_mix", "cnn_domain",
         "ens_raw", "ens_raw_at", "ens_raw_mix"]
PAIRS = [("gbm_raw_at", "gbm_raw_mix"), ("cnn_raw_at", "cnn_raw_mix"), ("cnn_raw_at", "ens_raw_at"),
         ("gbm_raw_at", "ens_raw_at"), ("ens_raw_at", "ens_raw_mix"), ("gbm_raw", "gbm_raw_at"),
         ("cnn_raw", "cnn_raw_at"), ("cnn_raw_mix", "ens_raw_mix")]


def load(root: Path = Path("results/defenses")):
    rob, clean, per = [], [], []
    for d in sorted(root.glob("*-seed*")):
        if not (d / "summary.json").exists():
            continue
        cfg, seed = d.name.rsplit("-seed", 1)
        tag = {"dataset": CONFIG_DATASET[cfg], "seed": int(seed)}
        rob.append(pd.read_csv(d / "robustness.csv").assign(**tag))
        clean.append(pd.read_csv(d / "clean_metrics.csv").assign(**tag))
        per.append(pd.read_csv(d / "per_url.csv.gz").assign(**tag))
    if not rob:
        raise SystemExit("no finished runs in results/defenses/")
    return pd.concat(rob), pd.concat(clean), pd.concat(per)


def caught(per: pd.DataFrame, n_sample: dict) -> pd.DataFrame:
    """Per (dataset, seed, model, attack, url index): 1 if still flagged after the attack."""
    rows = []
    for (ds, seed, m, a), g in per.groupby(["dataset", "seed", "model", "attack"]):
        v = np.zeros(n_sample[(ds, seed)], dtype=int)
        v[g.i.to_numpy()] = 1 - g.evaded.to_numpy()
        rows.append(pd.DataFrame({"dataset": ds, "seed": seed, "model": m, "attack": a,
                                  "i": np.arange(len(v)), "caught": v}))
        if a == "greedy_blackbox":  # clean detection on the sample
            v0 = np.zeros_like(v)
            v0[g.i.to_numpy()] = 1
            rows.append(pd.DataFrame({"dataset": ds, "seed": seed, "model": m, "attack": "none",
                                      "i": np.arange(len(v)), "caught": v0}))
    return pd.concat(rows)


def summarize(rob, clean, cau) -> pd.DataFrame:
    rows = []
    for (ds, m), g in clean.groupby(["dataset", "model"]):
        base = {"dataset": ds, "model": m, "n_seeds": g.seed.nunique(), "seeds": " ".join(map(str, sorted(g.seed))),
                "clean_f1": g.f1.mean(), "clean_f1_min": g.f1.min(), "clean_f1_max": g.f1.max(),
                "clean_fpr": g.fpr.mean(), "clean_fpr_min": g.fpr.min(), "clean_fpr_max": g.fpr.max()}
        for a in ATTACKS:
            c = cau[(cau.dataset == ds) & (cau.model == m) & (cau.attack == a)]
            if len(c):
                per_seed = c.groupby("seed").caught.mean()
                lo, hi = wilson_interval(int(c.caught.sum()), len(c))
                base.update({f"{a}_caught": c.caught.mean(), f"{a}_caught_lo": lo, f"{a}_caught_hi": hi,
                             f"{a}_caught_min": per_seed.min(), f"{a}_caught_max": per_seed.max()})
            r = rob[(rob.dataset == ds) & (rob.model == m) & (rob.attack == a)]
            if len(r):
                lo, hi = wilson_interval(int(r.evaded.sum()), int(r.n_initially_detected.sum()))
                base.update({f"{a}_evasion": r.evaded.sum() / r.n_initially_detected.sum(),
                             f"{a}_evasion_lo": lo, f"{a}_evasion_hi": hi})
        rows.append(base)
    s = pd.DataFrame(rows)
    s["order"] = s.model.map({m: i for i, m in enumerate(ORDER)})
    return s.sort_values(["dataset", "order"]).drop(columns="order")


def paired(cau: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in cau.dataset.unique():
        for a in [x for x in ATTACKS if x != "none"]:
            for m1, m2 in PAIRS:
                x = cau[(cau.dataset == ds) & (cau.attack == a)]
                p1 = x[x.model == m1].set_index(["seed", "i"]).caught
                p2 = x[x.model == m2].set_index(["seed", "i"]).caught
                j = p1.to_frame("a").join(p2.to_frame("b"), how="inner")
                if not len(j):
                    continue
                b = int(((j.a == 1) & (j.b == 0)).sum())
                c = int(((j.a == 0) & (j.b == 1)).sum())
                pv = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
                rows.append({"dataset": ds, "attack": a, "model_a": m1, "model_b": m2,
                             "n_pairs": len(j), "seeds": j.index.get_level_values(0).nunique(),
                             "caught_a": j.a.mean(), "caught_b": j.b.mean(),
                             "only_a_catches": b, "only_b_catches": c, "mcnemar_exact_p": pv})
    return pd.DataFrame(rows)


def _pct(v, lo=None, hi=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{100 * v:.1f}%" + (f" [{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else "")


def hero(summary: pd.DataFrame, out: Path, datasets=None) -> None:
    """Clean vs attacked vs defended, both datasets, 95% CIs. Sized for a 1200x627 LinkedIn image."""
    conds = [("none", "No attack"), ("greedy_blackbox", "Greedy\nattack"),
             ("greedy_heldout", "Held-out\nedits"), ("beam_extended", "Beam\nsearch")]
    series = [("gbm_raw", "#b9cde5"), ("gbm_raw_at", "#5b8fc7"), ("cnn_raw", "#f3c9a1"), ("cnn_raw_at", "#d9822b")]
    have = set(summary.dataset)
    if datasets is None:  # prefer the leak-free PhiUSIIL split once it exists
        datasets = ("phreshphish", "phiusiil-grouped" if "phiusiil-grouped" in have else "phiusiil")
    datasets = [d for d in datasets if d in have]
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 6.27), sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = summary[summary.dataset == ds].set_index("model")
        x = np.arange(len(conds))
        w = 0.2
        for j, (m, col) in enumerate(series):
            if m not in d.index:
                continue
            r = d.loc[m]
            vals = np.array([100 * r.get(f"{a}_caught", np.nan) for a, _ in conds])
            lo = np.array([100 * r.get(f"{a}_caught_lo", np.nan) for a, _ in conds])
            hi = np.array([100 * r.get(f"{a}_caught_hi", np.nan) for a, _ in conds])
            ax.bar(x + (j - 1.5) * w, vals, w, color=col, label=LABELS[m], yerr=[vals - lo, hi - vals],
                   capsize=2.5, error_kw={"lw": 1})
            for xi, v, h in zip(x + (j - 1.5) * w, vals, hi):
                if np.isfinite(v):
                    ax.text(xi, h + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(x, [c for _, c in conds], fontsize=13)
        n = int(d["n_seeds"].max())
        ax.set_title(f"{DATASET_LABELS[ds]}\n({n} seed{'s' if n > 1 else ''}, 95% CI)", fontsize=14)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Phishing URLs still caught (%)", fontsize=13)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=12, frameon=False)
    fig.suptitle("URL phishing detectors under cheap, realistic URL edits", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def summary_from_experiments(root: Path = Path(".")) -> pd.DataFrame:
    """Caught-under-attack per model from the main experiment runs (all seeds), for the hero figure."""
    import json

    from .aggregate import find_runs

    rows = []
    for ds, dirs in find_runs(root).items():
        for d in dirs:
            s = json.loads((d / "summary.json").read_text())
            for r in s["robustness"]:
                n = int(r["n_sample"])
                rows.append({"dataset": ds, "seed": s["config"]["seed"], "model": r["model"], "attack": r["attack"],
                             "n": n, "k": int(round(r["detection_under_attack"] * n))})
                if r["attack"] == "greedy_blackbox":
                    rows.append({**rows[-1], "attack": "none", "k": int(r["n_initially_detected"])})
    df = pd.DataFrame(rows)
    out = []
    for (ds, m), g in df.groupby(["dataset", "model"]):
        row = {"dataset": ds, "model": m, "n_seeds": g.seed.nunique(), "seeds": " ".join(map(str, sorted(set(g.seed))))}
        for a, h in g.groupby("attack"):
            lo, hi = wilson_interval(int(h.k.sum()), int(h.n.sum()))
            per = h.k / h.n
            row.update({f"{a}_caught": h.k.sum() / h.n.sum(), f"{a}_caught_lo": lo, f"{a}_caught_hi": hi,
                        f"{a}_caught_min": per.min(), f"{a}_caught_max": per.max()})
        out.append(row)
    return pd.DataFrame(out)


def write_hero(root: Path = Path(".")) -> pd.DataFrame:
    s = summary_from_experiments(root)
    s.to_csv(root / "results/hero_data.csv", index=False)
    hero(s, root / "results/hero.png")
    return s


def write(root: Path = Path(".")) -> None:
    out = root / "results/defenses"
    rob, clean, per = load(out)
    n_sample = {(ds, s): int(g.n_sample.iloc[0]) for (ds, s), g in rob.groupby(["dataset", "seed"])}
    cau = caught(per, n_sample)
    summ = summarize(rob, clean, cau)
    pt = paired(cau)
    summ.to_csv(out / "defenses_summary.csv", index=False)
    pt.to_csv(out / "paired_tests.csv", index=False)

    lines = ["# Round-2 defenses", "", "Generated by `python -m phishadv.defense_report` from "
             + ", ".join(f"`{d.name}`" for d in sorted(out.glob('*-seed*')) if (d / 'summary.json').exists()) + ".",
             "", "*Caught* = share of the 1,000 sampled phishing URLs per seed that the model flags and still "
             "flags after the attack (same denominator for every model). *Evasion* = share of the initially "
             "flagged URLs that the attack gets below the threshold (denominator differs by model). Brackets are "
             "95% Wilson intervals pooled over seeds (sampling error only); clean F1 and FPR are means over seeds "
             "on the full test split. Domain-only models are only attacked with greedy: no allowed edit changes "
             "their input.", ""]
    for ds in summ.dataset.unique():
        d = summ[summ.dataset == ds]
        lines += [f"## {DATASET_LABELS[ds]} (seeds: {d.seeds.iloc[0]})", "",
                  "| Model | Clean F1 | Clean FPR | Caught, no attack | Caught: greedy | Caught: held-out | "
                  "Caught: beam | Evasion: greedy | Evasion: held-out | Evasion: beam |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            cells = [f"{r.clean_f1:.3f}", f"{100 * r.clean_fpr:.1f}%"]
            for a in ATTACKS:
                cells.append(_pct(r.get(f"{a}_caught"), r.get(f"{a}_caught_lo"), r.get(f"{a}_caught_hi")))
            for a in list(ATTACKS)[1:]:
                cells.append(_pct(r.get(f"{a}_evasion"), r.get(f"{a}_evasion_lo"), r.get(f"{a}_evasion_hi")))
            lines.append(f"| {LABELS.get(r.model, r.model)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Paired tests (exact McNemar on *caught under attack*, same URLs, pooled over seeds)", "",
              "| Dataset | Attack | A | B | Caught A | Caught B | Only A | Only B | p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in pt.iterrows():
        lines.append(f"| {DATASET_LABELS[r.dataset]} | {ATTACKS[r.attack]} | {LABELS.get(r.model_a)} | "
                     f"{LABELS.get(r.model_b)} | {100 * r.caught_a:.1f}% | {100 * r.caught_b:.1f}% | "
                     f"{r.only_a_catches} | {r.only_b_catches} | {r.mcnemar_exact_p:.2g} |")
    lines.append("")
    (out / "DEFENSES.md").write_text("\n".join(lines))
    print(f"wrote {out / 'DEFENSES.md'}")


if __name__ == "__main__":
    write_hero()
    if any(Path("results/defenses").glob("*-seed*/summary.json")):
        write()
