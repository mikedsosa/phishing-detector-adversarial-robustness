# Data

This project uses the **PhiUSIIL Phishing URL Dataset**.

| | |
|---|---|
| Source | UCI Machine Learning Repository, dataset 967: <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset> |
| Creators | Arvind Prasad and Shalini Chandra (Babasaheb Bhimrao Ambedkar University) |
| License | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| Size | 235,795 rows (134,850 legitimate, 100,945 phishing), 54.2 MB CSV |
| Labels | In the raw file, `label = 1` means **legitimate** and `label = 0` means **phishing**. `phishadv.data.load_urls` flips this so that `y = 1` means phishing. |

**Citation**

> Prasad, A. & Chandra, S. (2024). PhiUSIIL: A diverse security profile
> empowered phishing URL detection framework based on similarity index and
> incremental learning. *Computers & Security*, 136, 103545.
> https://doi.org/10.1016/j.cose.2023.103545

## Download

```bash
make data          # or: python scripts/download_data.py
```

This downloads the zip from UCI into `data/raw/` (git-ignored) and verifies
these SHA-256 checksums:

```
0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a  phiusiil.zip
a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6  PhiUSIIL_Phishing_URL_Dataset.csv
```

## What we use

Only the `URL` and `label` columns. The dataset's 50+ precomputed features
include page-content features (HTML, title, forms) and derived scores such as
`URLSimilarityIndex`. We deliberately don't use any of them, because a
URL-only detector must work without fetching the page. All features are
recomputed from the URL string in `src/phishadv/features.py`.

After de-duplication there are 235,370 unique URLs. URLs that appear with both
labels are dropped, and so are exact duplicates.

## Known dataset artifact (important)

Every legitimate URL in PhiUSIIL has the form `https://www.<domain>`: HTTPS,
a `www` label, and no path or query. Phishing URLs are far more varied (see
`results/RESULTS.md`, "Dataset structure"). A model trained on this data can
therefore score highly by learning URL *format*, and that is exactly what our
cheapest attacks exploit. See `docs/paper.md`.

## Bundled test sample

`tests/fixtures/phiusiil_sample.csv` holds 1,500 rows (750 per class, random
seed 42) of the `URL` and `label` columns. It is redistributed under CC BY 4.0
with the attribution above, so CI can run without downloading the dataset.
**These are real phishing URLs. Don't visit them.**
