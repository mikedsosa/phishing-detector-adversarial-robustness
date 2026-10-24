"""Download the PhiUSIIL Phishing URL dataset from UCI and verify checksums.

Source:  https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
RAW = Path("data/raw")
ZIP = RAW / "phiusiil.zip"
CSV = RAW / "PhiUSIIL_Phishing_URL_Dataset.csv"
SHA256 = {
    ZIP.name: "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a",
    CSV.name: "a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> None:
    got = sha256(path)
    want = SHA256[path.name]
    if got != want:
        sys.exit(f"Checksum mismatch for {path}: got {got}, expected {want}")
    print(f"OK  {path}  sha256={got}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if CSV.exists() and sha256(CSV) == SHA256[CSV.name]:
        print(f"OK  {CSV} already present")
        return
    if not ZIP.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, ZIP)
    verify(ZIP)
    with zipfile.ZipFile(ZIP) as z:
        z.extract(CSV.name, RAW)
    verify(CSV)


if __name__ == "__main__":
    main()
