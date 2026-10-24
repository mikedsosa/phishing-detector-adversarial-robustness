"""Tests for the held-out transformations, beam search, minimal flip, PhreshPhish loading,
model persistence and the single-URL demo CLI."""

import random

import numpy as np
import pytest

from phishadv.attacks import HELDOUT_TRANSFORMS, TIERS, beam_blackbox, labeled_neighbors, minimal_flip, neighbors
from phishadv.cli import main as cli_main
from phishadv.data import PHRESH_SAMPLE_CSV, SAMPLE_CSV, load_phreshphish, load_urls, split
from phishadv.metrics import wilson_interval
from phishadv.models import CNNDetector, GBMDetector, load_detector, save_detector
from phishadv.urls import is_valid_url, parse

URLS = [
    "http://www.worldmedicsky.info",
    "http://membershiplienquangarenavn.xyz/ui/login/",
    "https://asb--authorise.firebaseapp.com/",
    "http://secure.paypal-account.example.com/webscr/cmd.php?login=1&id=2#top",
    "http://192.168.1.20/bank/login.html",
]


class ToyModel:
    """P(phishing) = 0.9 minus 0.3 per 'benign-looking' feature present."""

    def predict_proba(self, urls):
        out = []
        for u in urls:
            p = parse(u)
            s = 0.9 - 0.3 * (p.scheme == "https") - 0.3 * ("blog" in p.path) - 0.3 * (not p.query)
            out.append(max(s, 0.0))
        return np.array(out)


def test_heldout_transforms_valid_and_disjoint_from_training_tier():
    assert not set(TIERS["heldout"]) & set(TIERS["owner"])
    assert set(TIERS["extended"]) == set(TIERS["owner"]) | set(TIERS["heldout"])
    rng = random.Random(0)
    phish = load_urls(SAMPLE_CSV).query("y == 1").url.tolist()[:100]
    for url in URLS + phish:
        reg = parse(url).registrable
        for name, t in HELDOUT_TRANSFORMS.items():
            for c in t(url, rng):
                if is_valid_url(c, url):
                    assert parse(c).registrable == reg, (name, url, c)
        for c, name in labeled_neighbors(url, url, rng, "heldout"):
            assert name in HELDOUT_TRANSFORMS
            assert is_valid_url(c, url)


def test_owner_tier_neighbors_unchanged_by_extension():
    # The training-time tier must produce exactly the transformations it always did.
    for url in URLS:
        a = neighbors(url, url, random.Random(3), "owner")
        b = [c for c, n in labeled_neighbors(url, url, random.Random(3), "owner")]
        assert a == b


def test_beam_search_succeeds_with_valid_trajectory():
    url = "http://secure.example-login.com/acct/verify.php?id=7"
    res = beam_blackbox(ToyModel(), [url], max_steps=6, beam=3, seed=0)
    assert res.success[0]
    assert res.final_score[0] < 0.5
    assert res.trace[0][0] == url and res.trace[0][-1] == res.adversarial[0]
    for u in res.trace[0]:
        assert is_valid_url(u, url)
    assert res.steps[0] == len(res.trace[0]) - 1


def test_minimal_flip_finds_fewest_steps():
    url = "http://secure.example-login.com/acct/verify.php?id=7"
    res = minimal_flip(ToyModel(), url, max_depth=3)
    assert res["flagged"] and res["found"]
    # https + drop query = 0.3 < 0.5 needs two edits; one edit can never get below 0.6.
    assert len(res["steps"]) == 2
    assert res["p_after"] < 0.5
    assert minimal_flip(ToyModel(), "https://example.com/blog/x", max_depth=2)["flagged"] is False


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0
    assert np.isnan(wilson_interval(0, 0)[0])


def test_phreshphish_loader_keeps_time_split():
    df = load_phreshphish(PHRESH_SAMPLE_CSV)
    assert set(df.y.unique()) == {0, 1}
    s = split(df, seed=0)
    assert (s.test.split == "test").all() and (s.train.split == "train").all()
    assert s.train.date.max() <= s.test.date.min()
    assert not set(s.train.url) & set(s.test.url)


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    df = load_urls(SAMPLE_CSV)
    urls, y = df.url.tolist(), df.y.to_numpy()
    d = tmp_path_factory.mktemp("models")
    gbm = GBMDetector(seed=0).fit(urls, y)
    cnn = CNNDetector(seed=0, epochs=1).fit(urls, y, log=lambda m: None)
    save_detector(gbm, d / "gbm_raw", meta={"dataset": "phiusiil", "seed": 0})
    save_detector(cnn, d / "cnn_raw", meta={"dataset": "phiusiil", "seed": 0})
    return d, gbm, cnn, urls[:50]


def test_save_load_roundtrip(tiny_models):
    d, gbm, cnn, urls = tiny_models
    g2, info = load_detector(d / "gbm_raw")
    c2, _ = load_detector(d / "cnn_raw")
    assert info["dataset"] == "phiusiil"
    np.testing.assert_allclose(g2.predict_proba(urls), gbm.predict_proba(urls))
    np.testing.assert_allclose(c2.predict_proba(urls), cnn.predict_proba(urls), rtol=1e-5)


def test_cli_scores_one_url_and_defangs(tiny_models, capsys):
    d = tiny_models[0]
    url = "http://secure-login.paypa1-verify.xyz/webscr/cmd.php?login=1"
    assert cli_main(["score", url, "--models-dir", str(d), "--model", "gbm_raw"]) == 0
    out = capsys.readouterr().out
    assert "P(phishing)" in out
    assert "http://" not in out and "https://" not in out and "paypa1-verify.xyz" not in out
    with pytest.raises(SystemExit):  # exactly one URL, no batch mode
        cli_main(["score", url, url, "--models-dir", str(d)])


def test_hopskipjump_call_cap_counts_as_failure(tiny_models):
    from phishadv.feature_space import hopskipjump_feature_space

    _, gbm, _, _ = tiny_models
    df = load_urls(SAMPLE_CSV)
    phish = [u for u in df.url[df.y == 1] if gbm.predict_proba([u])[0] >= 0.5][:3]
    x_tr = gbm.features(df.url.tolist())
    xb = gbm.features(df.url[df.y == 0].tolist())
    capped = hopskipjump_feature_space(gbm, phish, x_tr, xb, max_iter=2, max_calls=1)
    assert capped["n_aborted"] == 3 and capped["evasion_rate"] == 0.0
    normal = hopskipjump_feature_space(gbm, phish, x_tr, xb, max_iter=2)
    assert normal["n_aborted"] == 0
