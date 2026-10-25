"""Download the URL columns of the PhreshPhish dataset (Hugging Face) and verify them.

Source:  https://huggingface.co/datasets/phreshphish/phreshphish (revision pinned below)
License: Creative Commons Attribution 4.0 International (CC BY 4.0); the dataset
         authors ask that it be used only for anti-phishing research.

The full dataset is ~37 GB because every row carries the page HTML. This script
uses HTTP range requests to read only the ``sha256``, ``url``, ``label`` and
``date`` columns of each Parquet file (~21 MB in total). It never resolves or
visits any URL in the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = "phreshphish/phreshphish"
REVISION = "eca603e0ecd4f698f8d4d226c088659e72681239"  # v1.0.1 (2026-02-07)
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data"
FILES = {"train": 56, "test": 21}
COLUMNS = ["sha256", "url", "label", "date"]
OUT = Path("data/raw/phreshphish_urls.csv")
# SHA-256 over the sorted "sha256<TAB>label<TAB>split" lines: independent of CSV formatting.
CONTENT_SHA256 = "09601865e26d4c172103587779aa122bd4cb0111b1e8c5a6552d0da9bfa1d728"


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP range requests."""

    def __init__(self, url: str, retries: int = 5):
        self.retries, self.pos = retries, 0
        r = self._open(urllib.request.Request(url, method="HEAD"))
        self.size, self.url = int(r.headers["Content-Length"]), r.url

    def _open(self, req):
        for attempt in range(self.retries):
            try:
                return urllib.request.urlopen(req, timeout=60)
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        data = self._open(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})).read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def content_hash(df: pd.DataFrame) -> str:
    lines = sorted(f"{s}\t{lab}\t{sp}" for s, lab, sp in zip(df.sha256, df.label, df.split))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if OUT.exists():
        df = pd.read_csv(OUT)
        if content_hash(df) == CONTENT_SHA256:
            print(f"OK  {OUT} already present ({len(df)} rows)")
            return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for split, n in FILES.items():
        for i in range(n):
            name = f"{split}-{i:03d}.parquet"
            t = pq.ParquetFile(HTTPRangeFile(f"{BASE}/{name}")).read(columns=COLUMNS).to_pandas()
            t["split"] = split
            parts.append(t)
            print(f"  {name}: {len(t)} rows", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["date"] = df["date"].astype(str)
    got = content_hash(df)
    if got != CONTENT_SHA256:
        sys.exit(f"Content checksum mismatch: got {got}, expected {CONTENT_SHA256}")
    df.to_csv(OUT, index=False)
    print(f"OK  {OUT}  rows={len(df)}  content_sha256={got}")


if __name__ == "__main__":
    main()
