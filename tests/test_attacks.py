import random

import numpy as np
import pytest

from phishadv.attacks import (
    TRANSFORMS,
    controllable_positions,
    greedy_blackbox,
    hotflip_whitebox,
    neighbors,
    random_blackbox,
    shortener_wrap,
)
from phishadv.data import SAMPLE_CSV, load_urls
from phishadv.models import CNNDetector
from phishadv.urls import canonicalize, is_valid_url, parse

URLS = [
    "http://www.worldmedicsky.info",
    "http://membershiplienquangarenavn.xyz/ui/login/",
    "https://asb--authorise.firebaseapp.com/",
    "http://secure.paypal-account.example.com/webscr/cmd.php?login=1&id=2#top",
    "http://192.168.1.20/bank/login.html",
]


@pytest.fixture(scope="module")
def sample_phish():
    df = load_urls(SAMPLE_CSV)
    return df[df.y == 1].url.tolist()[:150]


def test_every_transformation_keeps_urls_valid_and_domain_fixed(sample_phish):
    rng = random.Random(0)
    for url in URLS + sample_phish:
        reg = parse(url).registrable
        for name, t in TRANSFORMS.items():
            for c in t(url, rng):
                if is_valid_url(c, url):
                    assert parse(c).registrable == reg, (name, url, c)
        for c in neighbors(url, url, rng):
            assert is_valid_url(c, url)
            assert parse(c).registrable == reg
            assert c.isascii()


def test_multi_step_trajectories_stay_valid(sample_phish):
    rng = random.Random(1)
    for url in sample_phish[:60]:
        cur = url
        for _ in range(8):
            nb = neighbors(cur, url, rng)
            if not nb:
                break
            cur = rng.choice(nb)
            assert is_valid_url(cur, url)


def test_homoglyph_is_idna_and_canonicalizes_back():
    rng = random.Random(0)
    out = TRANSFORMS["homoglyph_sub_label"]("http://secure.example.com/a", rng)
    assert out and "xn--" in out[0]
    assert canonicalize(out[0]) == canonicalize("http://secure.example.com/a")


class HttpsLover:
    """Toy detector: everything without https is phishing."""

    def predict_proba(self, urls):
        return np.array([0.1 if u.startswith("https://") else 0.9 for u in urls])


def test_greedy_finds_one_step_evasion():
    res = greedy_blackbox(HttpsLover(), ["http://evil.example.com/x"], max_steps=3)
    assert res.success.all()
    assert res.steps[0] == 1
    assert res.adversarial[0].startswith("https://")
    assert res.queries[0] >= 2


def test_greedy_respects_query_budget():
    class Never:
        def predict_proba(self, urls):
            return np.full(len(urls), 0.9)

    res = greedy_blackbox(Never(), ["http://a.b.example.com/x/y?z=1"], max_steps=5, max_queries=3)
    assert not res.success.any()


def test_random_baseline_runs():
    res = random_blackbox(HttpsLover(), URLS, max_steps=3, seed=0)
    assert len(res.adversarial) == len(URLS)
    assert all(is_valid_url(a, o) for a, o in zip(res.adversarial, res.original))


def test_controllable_positions_exclude_registrable_domain():
    url = "http://login.example.com/acc?x=1"
    pos = {j for j, _ in controllable_positions(url)}
    dom_start = url.index("example")
    assert not any(dom_start <= j < dom_start + len("example.com") for j in pos)
    assert url.index("login") in pos
    assert url.index("acc") in pos


@pytest.mark.parametrize("structural", [False, True])
def test_hotflip_outputs_valid_urls(structural):
    m = CNNDetector(view="raw", seed=0)  # untrained network is fine for a constraint test
    res = hotflip_whitebox(m, URLS, max_flips=3, top_k=4, threshold=-1.0, structural=structural)
    for o, a in zip(res.original, res.adversarial):
        assert is_valid_url(a, o), (o, a)
        assert parse(a).registrable == parse(o).registrable


def test_shortener_wrap_is_valid():
    for u in shortener_wrap(URLS, seed=0):
        assert is_valid_url(u)


def test_anchor_label_not_in_controllable_positions():
    url = "http://a1.tenant.example.com/p"
    pos = {j for j, _ in controllable_positions(url, url)}
    t = url.index("tenant")
    assert not any(t <= j < t + len("tenant") for j in pos)
    assert url.index("a1") in pos


def test_append_only_tier_never_changes_host_or_path(sample_phish):
    rng = random.Random(3)
    for url in URLS + sample_phish[:80]:
        o = parse(url)
        for c in neighbors(url, url, rng, tier="append_only"):
            p = parse(c)
            assert (p.scheme, p.host, p.path) == (o.scheme, o.host, o.path)
            assert is_valid_url(c, url)
