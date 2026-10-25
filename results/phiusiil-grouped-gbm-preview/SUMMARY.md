# PhiUSIIL: random split vs. domain-grouped split (GBM only, seeds 42–44)

Generated from `scripts/grouped_gbm_preview.py` (grouped) and the round-1 runs
(`results/`, `results/seeds/full-seed4x/`, random split). Same 1,000-URL attack
sample size, budgets and settings. Cells: mean over seeds, pooled 95% Wilson
interval, per-seed range. The full grouped experiment (all models, including
the CNN) is `python -m phishadv.experiment --config full-grouped`.

| Model | Attack | Random split | Domain-grouped split |
|---|---|---|---|
| GBM | Clean test F1 (FPR) | 0.9978 (0.04%) | 0.9977 (0.05%) |
| GBM | Greedy, ≤10 edits | 63.5% [61.8, 65.2] (61.7–64.5) | 61.7% [60.0, 63.5] (58.8–65.5) |
| GBM | Held-out edits | 0.1% [0.0, 0.3] | 0.0% [0.0, 0.1] |
| GBM | Beam, ≤20 | 79.8% [78.3, 81.2] (77.4–81.2) | 82.8% [81.4, 84.1] (78.8–84.9) |
| GBM + adv. training | Clean test F1 (FPR) | 0.9960 (0.38%) | 0.9958 (0.39%) |
| GBM + adv. training | Greedy | 34.3% [32.6, 36.0] (32.7–35.8) | 34.8% [33.1, 36.6] (33.9–35.4) |
| GBM + adv. training | Beam | 41.2% [39.4, 43.0] (38.3–43.2) | 41.4% [39.7, 43.2] (39.0–43.6) |

Denominators: 2,989–2,992 initially flagged URLs pooled over three seeds in
every cell.
