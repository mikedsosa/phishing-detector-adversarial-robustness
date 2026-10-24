import numpy as np
import pytest

from phishadv.data import SAMPLE_CSV, load_urls, split, to_ascii
from phishadv.metrics import detection_metrics, detection_under_attack, evasion_rate


def test_detection_metrics_known_confusion():
    y = [1, 1, 1, 0, 0, 0, 0, 1]
    p = [0.9, 0.8, 0.2, 0.1, 0.6, 0.3, 0.4, 0.7]
    m = detection_metrics(y, p)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 3)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(0.25)
    assert m["accuracy"] == pytest.approx(0.75)


def test_robustness_metrics():
    det = np.array([True, True, False, True])
    succ = np.array([True, False, False])
    assert evasion_rate(succ) == pytest.approx(1 / 3)
    assert detection_under_attack(det, succ) == pytest.approx(2 / 4)
    with pytest.raises(ValueError):
        detection_under_attack(det, np.array([True]))


def test_label_flip_and_split():
    df = load_urls(SAMPLE_CSV)
    assert set(df.y.unique()) == {0, 1}
    # PhiUSIIL legitimate URLs are all https://www.<domain>; after the flip they are y=0.
    assert df[df.y == 0].url.str.startswith("https://").all()
    s = split(df, seed=1)
    assert len(s.train) + len(s.val) + len(s.test) == len(df)
    assert not set(s.train.url) & set(s.test.url)


def test_to_ascii():
    assert to_ascii("http://b\u00fccher.example/\u00e4") == "http://xn--bcher-kva.example/%C3%A4"
