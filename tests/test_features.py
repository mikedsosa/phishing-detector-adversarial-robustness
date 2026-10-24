import numpy as np

from phishadv.features import NUMERIC_FEATURES, LexicalFeaturizer, url_numeric_features


def feat(url):
    return dict(zip(NUMERIC_FEATURES, url_numeric_features(url)))


def test_numeric_feature_values():
    f = feat("https://www.secure-login.example.com/verify/index.php?id=12&x=3")
    assert f["is_https"] == 1.0
    assert f["has_www"] == 1.0
    assert f["n_sub_labels"] == 2
    assert f["n_hyphens_host"] == 1
    assert f["n_path_segments"] == 2
    assert f["n_query_params"] == 2
    assert f["has_file_ext"] == 1.0
    assert f["n_suspicious_tokens"] >= 3  # secure, login, verify
    assert f["domain_len"] == len("example")
    assert f["url_len"] == len("https://www.secure-login.example.com/verify/index.php?id=12&x=3")


def test_punycode_and_ip_flags():
    assert feat("http://xn--pypal-4ve.example.com")["has_punycode"] == 1.0
    assert feat("http://10.0.0.1/a")["is_ip"] == 1.0
    assert feat("http://example.com")["is_https"] == 0.0


def test_featurizer_shape_and_onehot():
    urls = ["https://www.a.com", "http://b.net/x", "http://c.org", "http://d.xyz", "http://e.com"]
    fz = LexicalFeaturizer(top_k_suffixes=2).fit(urls)
    x = fz.transform(urls + ["http://f.unseen"])
    assert x.shape == (6, len(NUMERIC_FEATURES) + 3)
    onehot = x[:, len(NUMERIC_FEATURES):]
    np.testing.assert_array_equal(onehot.sum(1), np.ones(6))
    assert fz.feature_names[-1] == "tld=other"
    assert onehot[-1, -1] == 1.0
