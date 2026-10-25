# Data

This project uses two public datasets. Raw files are not committed; download
scripts fetch them into `data/raw/` (git-ignored) and verify checksums.

## 1. PhiUSIIL Phishing URL Dataset

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

Dataset citation (UCI):

> Prasad, A. & Chandra, S. (2024). PhiUSIIL Phishing URL (Website)
> [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.1016/j.cose.2023.103545

### Download

```bash
make data          # or: python scripts/download_data.py
```

This downloads the zip from UCI into `data/raw/` (git-ignored) and verifies
these SHA-256 checksums:

```
0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a  phiusiil.zip
a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6  PhiUSIIL_Phishing_URL_Dataset.csv
```

### What we use

Only the `URL` and `label` columns. The dataset's 50+ precomputed features
include page-content features (HTML, title, forms) and derived scores such as
`URLSimilarityIndex`. We deliberately don't use any of them, because a
URL-only detector must work without fetching the page. All features are
recomputed from the URL string in `src/phishadv/features.py`.

After de-duplication there are 235,370 unique URLs. URLs that appear with both
labels are dropped, and so are exact duplicates.

### Known dataset artifact (important)

99.97% of legitimate URLs in PhiUSIIL have the form `https://www.<domain>`: HTTPS,
a `www` label, and no path or query. Phishing URLs are far more varied (see
`results/RESULTS.md`, "Dataset structure"). A model trained on this data can
therefore score highly by learning URL *format*, and that is exactly what our
cheapest attacks exploit. See `docs/paper.md`.

### Bundled test sample

`tests/fixtures/phiusiil_sample.csv` holds 1,500 rows (750 per class, random
seed 42) of the `URL` and `label` columns. It is redistributed under CC BY 4.0
with the attribution above, so CI can run without downloading the dataset.
**These are real phishing URLs. Don't visit them.** License, attribution and warning
notes are in [`tests/fixtures/README.md`](../tests/fixtures/README.md).

## 2. PhreshPhish

Added as a second, harder dataset because PhiUSIIL's legitimate URLs are all
homepages.

| | |
|---|---|
| Source | Hugging Face, [`phreshphish/phreshphish`](https://huggingface.co/datasets/phreshphish/phreshphish), pinned to revision `eca603e0ecd4f698f8d4d226c088659e72681239` (v1.0.1) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The authors ask that the data be used only for anti-phishing research, which is what this project does. |
| Size used | 666,315 rows (train 498,255, test 168,060); 666,313 after de-duplication and dropping one URL that can't be converted to ASCII |
| Split | Time-based, as published: train URLs are dated 2024-07-02 to 2025-09-08, test URLs 2025-09-08 to 2025-12-16 |
| Labels | `label` is `phish` or `benign`; `y = 1` means phishing |

**Citation**

> Dalton, T., Gowda, H., Rao, G., Pargi, S., Khodabakhshi, A. H., Rombs, J.,
> Jou, S. & Marwah, M. (2025). PhreshPhish: A Real-World, High-Quality,
> Large-Scale Phishing Website Dataset and Benchmark. arXiv:2507.10854.

### Download

```bash
python scripts/download_phreshphish.py   # also part of `make data`
```

The full dataset is about 37 GB because it includes page HTML. The script
reads only the `sha256`, `url`, `label` and `date` columns from the Parquet
files with HTTP range requests (about 21 MB transferred) and writes
`data/raw/phreshphish_urls.csv`. It never requests any of the URLs. It checks
a SHA-256 over the sorted `sha256<TAB>label<TAB>split` lines:

```
09601865e26d4c172103587779aa122bd4cb0111b1e8c5a6552d0da9bfa1d728
```

### How we use it

We keep the published time split. Validation is a stratified 10% of the
training split. To keep compute comparable with PhiUSIIL, models train on a
stratified sample of 164,759 training URLs (the size of the PhiUSIIL training
split); the full 168,059-URL test split is used for clean metrics.

### Known artifact

PhreshPhish legitimate URLs look like real browsing: 92.5% have a non-root
path. But almost none carry a query string (0.018%), while 11.6% of phishing
URLs do. The dataset paper says query parameters were removed from benign
URLs as part of PII scrubbing, so this is a collection artifact. "Has a query
string" is therefore a near-perfect-precision rule here. We report it as the
`query_rule` baseline so the shortcut is visible.

### Bundled test sample

`tests/fixtures/phreshphish_sample.csv` holds 1,500 rows (600 per class from
train, 150 per class from test, seed 42) with `sha256`, `url`, `label`,
`split` and `date`. It is redistributed under CC BY 4.0 with the attribution
above so CI runs offline. **These are real phishing URLs. Don't visit them.** License, attribution and warning
notes are in [`tests/fixtures/README.md`](../tests/fixtures/README.md).
