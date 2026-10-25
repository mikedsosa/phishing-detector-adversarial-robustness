"""Share of test URLs whose registrable domain also appears, with the same label, in training.

Writes results/domain_leakage.csv. Usage: PYTHONPATH=src python scripts/domain_leakage.py
"""
from __future__ import annotations

import pandas as pd

from phishadv.data import PHRESH_CSV, RAW_CSV, load_dataset, registrable_domains, split

rows = []
for ds, path, grouped, seeds in [("phiusiil", RAW_CSV, False, (42, 43, 44)), ("phiusiil", RAW_CSV, True, (42,)),
                                 ("phreshphish", PHRESH_CSV, False, (42,))]:
    df = load_dataset(ds, str(path))
    for seed in seeds:
        sp = split(df, seed, max_train=164759 if ds == "phreshphish" else None, group_by_domain=grouped)
        tr = sp.train.assign(d=registrable_domains(sp.train.url))
        te = sp.test.assign(d=registrable_domains(sp.test.url))
        for y, lab in ((1, "phishing"), (0, "legitimate")):
            seen = set(tr.d[tr.y == y])
            t = te[te.y == y]
            rows.append({"dataset": ds, "split": "domain-grouped" if grouped else ("time" if ds == "phreshphish"
                                                                                  else "random"),
                         "seed": seed, "label": lab, "n_test": len(t),
                         "share_domain_seen_in_train_same_label": float(t.d.isin(seen).mean())})
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/domain_leakage.csv", index=False)
