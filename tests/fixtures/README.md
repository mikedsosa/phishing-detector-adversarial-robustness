# Test fixtures

> **Warning: these files contain real, possibly still live, phishing URLs.
> Don't visit, resolve or share them.** The code in this repo only treats
> them as strings. They are here for research and for testing this code only.

| File | Source | License |
|---|---|---|
| `phiusiil_sample.csv` | 1,500 rows (750 per class, seed 42) of the `URL` and `label` columns of PhiUSIIL (Prasad & Chandra, 2024; [UCI dataset 967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset), doi:10.1016/j.cose.2023.103545) | CC BY 4.0 |
| `phreshphish_sample.csv` | 1,500 rows (600 per class from train, 150 per class from test, seed 42) of the `sha256`, `url`, `label`, `split` and `date` columns of PhreshPhish v1.0.1 (Dalton et al., 2025; [Hugging Face](https://huggingface.co/datasets/phreshphish/phreshphish), arXiv:2507.10854) | CC BY 4.0. The dataset authors ask that it be used only for anti-phishing research. |

Both samples are redistributed unchanged (apart from row selection) under
CC BY 4.0 with the attribution above, so CI can run without downloading the
full datasets. See [`data/README.md`](../../data/README.md) for checksums and
how the full data is fetched.
