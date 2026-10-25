"""Tests for the round-2 defenses (ensemble, mixed adversarial training) and the smoke run."""

from dataclasses import replace

import numpy as np
import pandas as pd

from phishadv.data import PHRESH_SAMPLE_CSV, load_phreshphish, split
from phishadv.defenses import Ensemble, run, train_with_attacks
from phishadv.experiment import CONFIGS
from phishadv.models import GBMDetector


class Const:
    view = "raw"

    def __init__(self, p):
        self.p = p

    def predict_proba(self, urls):
        return np.full(len(urls), self.p)


def test_ensemble_averages_member_scores():
    e = Ensemble(Const(0.2), Const(0.8), Const(0.5))
    assert e.view == "raw"
    assert np.allclose(e.predict_proba(["a", "b"]), [0.5, 0.5])


def test_mixed_training_adds_beam_examples():
    cfg = replace(CONFIGS["smoke-phresh"], at_rounds=1, at_n_per_round=30)
    sp = split(load_phreshphish(PHRESH_SAMPLE_CSV), cfg.seed)
    base = GBMDetector(seed=cfg.seed).fit(sp.train.url.tolist(), sp.train.y.to_numpy())
    _, plain = train_with_attacks("gbm", base, sp.train, cfg, mixed=False)
    _, mixed = train_with_attacks("gbm", base, sp.train, cfg, mixed=True)
    assert "added_beam" not in plain[0]
    assert mixed[0]["added"] == plain[0]["added"] + mixed[0]["added_beam"]


def test_defenses_smoke_run(tmp_path):
    s = run(CONFIGS["smoke-phresh"], tmp_path)
    rob = pd.read_csv(tmp_path / "robustness.csv")
    assert {"gbm_raw_mix", "cnn_raw_mix", "ens_raw", "ens_raw_at", "ens_raw_mix"} <= set(rob.model)
    # domain-only models are attacked with greedy only: no allowed edit can change their input.
    assert set(rob[rob.model == "gbm_domain"].attack) == {"greedy_blackbox"}
    assert (rob[rob.model.str.endswith("domain")].evasion_rate == 0).all()
    assert len(s["clean"]) == 11
