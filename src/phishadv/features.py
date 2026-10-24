"""Hand-crafted lexical URL features for the gradient-boosting detector.

Features are computed from the URL string only (no page content, no WHOIS,
no network lookups), so every feature is something a mail or web gateway can
compute offline at line rate.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .urls import is_ip, parse

SUSPICIOUS_TOKENS = (
    "login", "signin", "sign-in", "verify", "verification", "account", "update",
    "secure", "security", "bank", "confirm", "password", "webscr", "wallet",
    "support", "billing", "unlock", "auth", "recover", "suspend",
)
FILE_EXTS = (".php", ".html", ".htm", ".asp", ".aspx", ".js", ".cgi", ".exe", ".zip")

NUMERIC_FEATURES = [
    "url_len", "host_len", "path_len", "query_len", "n_sub_labels", "n_dots_host",
    "n_hyphens_host", "n_digits_host", "n_digits_url", "digit_ratio_url",
    "letter_ratio_url", "n_special_url", "is_https", "has_www", "is_ip", "has_port",
    "has_at", "has_double_slash_path", "n_path_segments", "n_query_params",
    "has_punycode", "host_entropy", "max_label_len", "domain_len", "suffix_len",
    "n_suspicious_tokens", "has_file_ext", "n_percent", "domain_digit_ratio",
    "domain_vowel_ratio",
]

# Features whose value is necessarily an integer or a 0/1 flag. Used to check
# whether a feature-space adversarial example corresponds to any real URL.
BINARY_FEATURES = {
    "is_https", "has_www", "is_ip", "has_port", "has_at", "has_double_slash_path",
    "has_punycode", "has_file_ext",
}
INTEGER_FEATURES = {
    "url_len", "host_len", "path_len", "query_len", "n_sub_labels", "n_dots_host",
    "n_hyphens_host", "n_digits_host", "n_digits_url", "n_special_url",
    "n_path_segments", "n_query_params", "max_label_len", "domain_len", "suffix_len",
    "n_suspicious_tokens", "n_percent",
} | BINARY_FEATURES
# Features that depend only on the registrable domain, which the attacker in
# our threat model does not change.
LOCKED_FEATURES = {"domain_len", "suffix_len", "is_ip", "domain_digit_ratio", "domain_vowel_ratio"}


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def url_numeric_features(url: str) -> list[float]:
    p = parse(url)
    host = p.host
    low = url.lower()
    n = max(len(url), 1)
    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)
    special = sum((not ch.isalnum()) and ch not in "./:" for ch in url)
    labels = host.split(".") if host else [""]
    dom = p.domain
    path_q = (p.path + "?" + p.query).lower()
    return [
        len(url),
        len(host),
        len(p.path),
        len(p.query),
        len(p.sub_labels),
        host.count("."),
        host.count("-"),
        sum(ch.isdigit() for ch in host),
        digits,
        digits / n,
        letters / n,
        special,
        float(p.scheme == "https"),
        float(bool(p.sub_labels) and p.sub_labels[0] == "www"),
        float(is_ip(host)),
        float(bool(p.port)),
        float("@" in url),
        float("//" in p.path),
        len(p.path_segments),
        len([q for q in p.query.split("&") if q]) if p.query else 0,
        float("xn--" in host),
        _entropy(host),
        max(len(lab) for lab in labels),
        len(dom),
        len(p.suffix),
        sum(tok in low for tok in SUSPICIOUS_TOKENS),
        float(any(ext in path_q for ext in FILE_EXTS)),
        url.count("%"),
        sum(ch.isdigit() for ch in dom) / max(len(dom), 1),
        sum(ch in "aeiou" for ch in dom) / max(len(dom), 1),
    ]


class LexicalFeaturizer:
    """Numeric lexical features + one-hot of the most frequent public suffixes."""

    def __init__(self, top_k_suffixes: int = 30):
        self.top_k = top_k_suffixes
        self.suffixes: list[str] = []

    def fit(self, urls: list[str]) -> LexicalFeaturizer:
        c = Counter(parse(u).suffix for u in urls)
        self.suffixes = [s for s, _ in c.most_common(self.top_k)]
        return self

    @property
    def feature_names(self) -> list[str]:
        return NUMERIC_FEATURES + [f"tld={s}" for s in self.suffixes] + ["tld=other"]

    def transform(self, urls: list[str]) -> np.ndarray:
        idx = {s: i for i, s in enumerate(self.suffixes)}
        k = len(self.suffixes)
        out = np.zeros((len(urls), len(NUMERIC_FEATURES) + k + 1), dtype=np.float32)
        for r, u in enumerate(urls):
            out[r, : len(NUMERIC_FEATURES)] = url_numeric_features(u)
            out[r, len(NUMERIC_FEATURES) + idx.get(parse(u).suffix, k)] = 1.0
        return out
