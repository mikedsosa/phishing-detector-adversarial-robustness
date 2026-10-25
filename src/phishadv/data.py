"""Dataset loading and splitting.

Two public datasets are supported (see ``data/README.md``):

* **PhiUSIIL** (UCI, CC BY 4.0). It encodes ``label = 1`` for *legitimate*
  and ``label = 0`` for *phishing*; :func:`load_urls` flips that so that
  ``y = 1`` means phishing, the positive class for every metric here.
* **PhreshPhish** (Hugging Face, CC BY 4.0). Only the URL columns are used.
  It ships a time-based train/test split, which :func:`split` keeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
SAMPLE_CSV = Path("tests/fixtures/phiusiil_sample.csv")
PHRESH_CSV = Path("data/raw/phreshphish_urls.csv")
PHRESH_SAMPLE_CSV = Path("tests/fixtures/phreshphish_sample.csv")


def to_ascii(url: str) -> str:
    """IDNA-encode the host and percent-encode any non-ASCII path/query chars."""
    url = url.strip()
    if url.isascii():
        return url
    s = urlsplit(url)
    host = s.hostname or ""
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        host = quote(host, safe=".-")
    netloc = host + (f":{s.port}" if s.port else "")
    safe = "/:@!$&'()*+,;=-._~%?#[]"
    return urlunsplit((s.scheme, netloc, quote(s.path, safe=safe),
                       quote(s.query, safe=safe), quote(s.fragment, safe=safe)))


def _safe_ascii(url: str) -> str | None:
    try:
        return to_ascii(url)
    except (UnicodeError, ValueError):
        return None


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop URLs that appear with both labels, then exact duplicates."""
    conflicted = df.groupby("url")["y"].nunique()
    df = df[~df["url"].isin(conflicted[conflicted > 1].index)]
    return df.drop_duplicates("url").reset_index(drop=True)


def load_phreshphish(path: Path | str = PHRESH_CSV) -> pd.DataFrame:
    """Return de-duplicated ``url``, ``y`` (1 = phishing), ``split`` and ``date`` columns.

    URLs that cannot be converted to ASCII (one in the full dataset) are dropped.
    """
    df = pd.read_csv(path, usecols=["url", "label", "split", "date"])
    df["url"] = df["url"].astype(str).map(_safe_ascii)
    df = df.dropna(subset=["url"])
    df["y"] = (df["label"] == "phish").astype(np.int64)
    return _dedupe(df[["url", "y", "split", "date"]])


def load_dataset(name: str, path: Path | str) -> pd.DataFrame:
    if name == "phiusiil":
        return load_urls(path)
    if name == "phreshphish":
        return load_phreshphish(path)
    raise ValueError(f"unknown dataset {name!r}")


def load_urls(path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Return a de-duplicated DataFrame with columns ``url`` and ``y`` (1 = phishing)."""
    df = pd.read_csv(path, usecols=["URL", "label"], encoding="utf-8-sig")
    df = df.rename(columns={"URL": "url"})
    df["url"] = df["url"].astype(str).map(to_ascii)
    df["y"] = (1 - df["label"].astype(int)).astype(np.int64)
    return _dedupe(df[["url", "y"]])


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def registrable_domains(urls) -> np.ndarray:
    from .urls import parse

    return np.array([parse(u).registrable for u in urls], dtype=object)


def _group_split(df: pd.DataFrame, size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that no registrable domain appears on both sides (``size`` is the share of domains)."""
    from sklearn.model_selection import GroupShuffleSplit

    groups = registrable_domains(df.url)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed).split(df, groups=groups))
    return df.iloc[a], df.iloc[b]


def split(df: pd.DataFrame, seed: int, test_size: float = 0.2, val_size: float = 0.1,
          max_train: int | None = None, max_test: int | None = None, group_by_domain: bool = False) -> Split:
    """Train/val/test split; optionally subsample the training and test sets.

    If ``df`` has a ``split`` column (PhreshPhish), its time-based train/test
    assignment is kept and the validation set is a stratified ``val_size``
    share of the training rows. Otherwise the split is stratified at random, or,
    with ``group_by_domain``, grouped by registrable domain so that no domain is
    in more than one of train, validation and test (not stratified; the class
    balance stays close because most domains have one URL).
    """
    if "split" in df.columns:
        trval, test = df[df["split"] == "train"], df[df["split"] == "test"]
        train, val = train_test_split(trval, test_size=val_size, stratify=trval["y"], random_state=seed)
    elif group_by_domain:
        trval, test = _group_split(df, test_size, seed)
        train, val = _group_split(trval, val_size / (1 - test_size), seed)
    else:
        trval, test = train_test_split(df, test_size=test_size, stratify=df["y"], random_state=seed)
        rel = val_size / (1 - test_size)
        train, val = train_test_split(trval, test_size=rel, stratify=trval["y"], random_state=seed)
    if max_test is not None and len(test) > max_test:
        test, _ = train_test_split(test, train_size=max_test, stratify=test["y"], random_state=seed)
    if max_train is not None and len(train) > max_train:
        train, _ = train_test_split(train, train_size=max_train, stratify=train["y"],
                                    random_state=seed)
    return Split(*(d.reset_index(drop=True) for d in (train, val, test)))
