"""Phishing URL detectors.

Every detector consumes raw URL strings and returns P(phishing). A detector
is configured with a *view* (``raw``, ``canon`` or ``domain``, see
:mod:`phishadv.urls`), the transformation applied to a URL before the model
sees it. The view is part of the deployed pipeline, so attacks always operate
on the raw URL and pass through the view, as a real attacker's URL would.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn

from .features import LexicalFeaturizer
from .urls import VIEWS

MAX_LEN = 200
PAD, UNK = 0, 1
VOCAB_SIZE = 2 + (127 - 32)  # printable ASCII


def encode_chars(urls: list[str], max_len: int = MAX_LEN) -> torch.Tensor:
    """Encode URLs as padded int64 character-index tensors (first ``max_len`` chars)."""
    longest = min(max((len(u) for u in urls), default=1), max_len)
    out = np.zeros((len(urls), max(longest, 1)), dtype=np.int64)
    for i, u in enumerate(urls):
        codes = [o - 30 if 32 <= (o := ord(ch)) < 127 else UNK for ch in u[:max_len]]
        out[i, : len(codes)] = codes
    return torch.from_numpy(out)


class Detector:
    name = "detector"

    def __init__(self, view: str = "raw"):
        self.view = view
        self._view_fn = VIEWS[view]

    def views(self, urls: list[str]) -> list[str]:
        return [self._view_fn(u) for u in urls]

    def predict_proba(self, urls: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class GBMDetector(Detector):
    """Histogram gradient boosting on lexical features."""

    name = "gbm"

    def __init__(self, view: str = "raw", seed: int = 0):
        super().__init__(view)
        self.featurizer = LexicalFeaturizer()
        self.clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=63, early_stopping=False,
            random_state=seed,
        )

    def fit(self, urls: list[str], y: np.ndarray) -> GBMDetector:
        v = self.views(urls)
        self.featurizer.fit(v)
        self.clf.fit(self.featurizer.transform(v), y)
        return self

    def features(self, urls: list[str]) -> np.ndarray:
        return self.featurizer.transform(self.views(urls))

    def predict_proba(self, urls: list[str]) -> np.ndarray:
        if not urls:
            return np.zeros(0)
        return self.clf.predict_proba(self.features(urls))[:, 1]


class CharCNN(nn.Module):
    """Character-level CNN in the spirit of URLNet / eXpose (multi-width convs + max-pool)."""

    def __init__(self, emb_dim: int = 32, n_filters: int = 128, widths=(3, 5, 7)):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, w, padding=w // 2) for w in widths)
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(n_filters * len(widths), 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward_embedded(self, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = e.transpose(1, 2)
        neg = (~mask).unsqueeze(1) * -1e4
        pooled = [torch.relu(c(x)).add(neg).amax(dim=2) for c in self.convs]
        return self.head(torch.cat(pooled, dim=1)).squeeze(1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.forward_embedded(self.emb(idx), idx != PAD)


class CNNDetector(Detector):
    name = "cnn"

    def __init__(self, view: str = "raw", seed: int = 0, epochs: int = 3, batch_size: int = 256,
                 lr: float = 1e-3):
        super().__init__(view)
        self.seed, self.epochs, self.batch_size, self.lr = seed, epochs, batch_size, lr
        torch.manual_seed(seed)
        self.net = CharCNN()

    def fit(self, urls: list[str], y: np.ndarray, log=print) -> CNNDetector:
        torch.manual_seed(self.seed)
        self.net = CharCNN()
        v = self.views(urls)
        y_t = torch.tensor(np.array(y), dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(self.seed)
        n = len(v)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=g).tolist()
            total = 0.0
            for b in range(0, n, self.batch_size):
                ids = perm[b : b + self.batch_size]
                x = encode_chars([v[i] for i in ids])
                opt.zero_grad()
                loss = loss_fn(self.net(x), y_t[ids])
                loss.backward()
                opt.step()
                total += loss.item() * len(ids)
            log(f"  [{self.name}/{self.view}] epoch {ep + 1}/{self.epochs} loss={total / n:.4f}")
        self.net.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, urls: list[str], batch: int = 2048) -> np.ndarray:
        if not urls:
            return np.zeros(0)
        self.net.eval()
        v = self.views(urls)
        out = [torch.sigmoid(self.net(encode_chars(v[i : i + batch]))) for i in range(0, len(v), batch)]
        return torch.cat(out).numpy().astype(np.float64)
