# Model card: URL-only phishing detectors (research models)

These are the four demo models the CLI and the local Streamlit app load from
`models/phreshphish/`. They are produced by `make reproduce-main` (config
`phresh`, seed 42) or, without the full experiment, by `make models`. Model
files are not committed; this card describes what that training step makes.

**These are research baselines for studying evasion, not a production
phishing filter.** Don't use them to block or allow real traffic.

## Models

| Name | Type | Input | Training |
|---|---|---|---|
| `gbm_raw` | scikit-learn `HistGradientBoostingClassifier` (300 iterations, 63 leaves) | 30 lexical features + one-hot of the 30 most frequent public suffixes, computed from the URL string | Clean training data |
| `cnn_raw` | PyTorch character-level CNN (embedding 32, convolutions of width 3/5/7 with 128 filters each, max length 200) | Raw URL characters | Clean training data, 3 epochs |
| `gbm_raw_at` | Same as `gbm_raw` | Same | Plus 2 rounds of adversarial training with the greedy black-box attack |
| `cnn_raw_at` | Same as `cnn_raw` | Same | Plus 2 rounds of adversarial training with the greedy black-box and hybrid white-box attacks |

Output: P(phishing). A URL is flagged at P ≥ 0.5. The models never fetch the
page, resolve the host or look up reputation; they only read the string.

## Training data

PhreshPhish (Dalton et al., 2025; CC BY 4.0; the authors ask that it be used
only for anti-phishing research), pinned Hugging Face revision
`eca603e0`. Only the URL and label columns are used. Models train on a
stratified sample of 164,759 URLs from the published training split (dated
2024-07-02 to 2025-09-08) and are tested on the full later test split
(168,059 URLs, dated 2025-09-08 to 2025-12-16). See
[`data/README.md`](data/README.md).

## Performance (seed 42, PhreshPhish test split, n = 168,059)

From [`results/phreshphish/clean_metrics.csv`](results/phreshphish/clean_metrics.csv):

| Model | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|
| `gbm_raw` | 0.9611 | 0.8987 | 0.9289 | 0.9811 | 0.0306 |
| `cnn_raw` | 0.9723 | 0.9221 | 0.9465 | 0.9888 | 0.0221 |
| `gbm_raw_at` | 0.9520 | 0.9135 | 0.9323 | 0.9814 | 0.0388 |
| `cnn_raw_at` | 0.9437 | 0.9562 | 0.9499 | 0.9904 | 0.0480 |

A false-positive rate of 2–5% is far too high for blocking real traffic.
Adversarial training raises it on seed 42: `cnn_raw_at` flags 4.8% of
legitimate URLs against 2.2% for `cnn_raw`, and `gbm_raw_at` 3.9% against
3.1%. The GBM's rise repeats on seeds 43 and 44; the CNN's does not
(`cnn_raw_at` 2.3% on both, vs. 2.4% and 3.1% for `cnn_raw`).

## Robustness (seed 42, 1,000 phishing test URLs)

Evasion rate among URLs each model flagged before the attack, from
[`results/phreshphish/robustness.csv`](results/phreshphish/robustness.csv).
Brackets are 95% Wilson intervals. The denominators differ by model (910
`gbm_raw`, 931 `cnn_raw`, 922 `gbm_raw_at`, 962 `cnn_raw_at` of the 1,000
sampled phishing URLs).

| Model | Greedy, budget 10 | Greedy, held-out edits only | Beam search, all edits, budget 20 |
|---|---|---|---|
| `gbm_raw` | 0.832 [0.806, 0.855] | 0.482 [0.450, 0.515] | 0.959 [0.944, 0.970] |
| `cnn_raw` | 0.956 [0.941, 0.967] | 0.755 [0.726, 0.782] | 1.000 [0.996, 1.000] |
| `gbm_raw_at` | 0.347 [0.317, 0.378] | 0.352 [0.322, 0.384] | 0.700 [0.669, 0.728] |
| `cnn_raw_at` | 0.176 [0.153, 0.201] | 0.551 [0.519, 0.582] | 0.976 [0.964, 0.984] |

Adversarial training helps against the attack it was trained on and much less
against edits it never saw or a stronger search. Seeds 43 and 44 give
the same picture. Averaged over seeds 42–44 (pooled 95% Wilson intervals),
greedy evasion is 0.849 [0.835, 0.862] for `gbm_raw`, 0.951 [0.942, 0.958]
for `cnn_raw`, 0.367 [0.350, 0.386] for `gbm_raw_at` and 0.229
[0.214, 0.244] for `cnn_raw_at` (per-seed 0.176–0.269); see
[`results/SEEDS.md`](results/SEEDS.md). The published models are the seed-42
ones.

## Known limitations

* **Out-of-distribution failure.** Scored on the PhiUSIIL test split, whose
  legitimate URLs are bare `https://www.<domain>` homepages, the PhreshPhish
  models' F1 falls to 0.57–0.76 and `gbm_raw` flags almost every legitimate
  homepage (FPR 0.9987; `results/phreshphish/cross_dataset.csv`).
* **Drift over time.** Phishing recall on the September 2025 test URLs (the
  first weeks after the training data ends) is 58.7% for `gbm_raw` and 85.0%
  for `cnn_raw`, against 95.9% and 95.7% on same-period validation data. It
  recovers to 91–93% in October and November. On December's 1,017 benign URLs
  the FPR rises to 8.4% (`gbm_raw`), 8.8% (`cnn_raw`) and 15.6%
  (`cnn_raw_at`). See [`results/drift/DRIFT.md`](results/drift/DRIFT.md).
* **Low base rates.** At a 1% phishing base rate, precision at 90% recall is
  22.5% (`gbm_raw`) to 52.1% (`cnn_raw_at`); at 0.05% it is 1.4–5.1%. At a
  threshold tuned for 0.1% validation FPR, greedy evasion rises (e.g.
  `cnn_raw` 98.4%, `cnn_raw_at` 38.5%). See
  [`results/operating/phreshphish/OPERATING.md`](results/operating/phreshphish/OPERATING.md).
* **Query-string shortcut.** The dataset authors stripped query strings from
  benign URLs for privacy. As a result only 0.018% of legitimate URLs have
  one, against 11.6% of phishing URLs, and the models can learn "has a query"
  as a phishing cue.
* **Easy to evade.** See the table above. One to three cheap edits that keep
  the registrable domain are usually enough.
* URL-only: no page content, certificate, WHOIS or reputation signals.
* English-centric benign tokens in the attack, ASCII-only inputs (IDN labels
  are converted to `xn--` form).

## Intended use

* Reproducing and extending the robustness study in [`docs/paper.md`](docs/paper.md).
* Explaining one URL at a time with `python -m phishadv score URL`, which
  shows the smallest set of edits that flips the verdict. The tool takes a
  single URL, has no batch mode, prints only defanged URLs and never visits
  anything.

**Out of scope:** production filtering, generating evasive URLs in bulk,
scoring URLs you aren't authorized to analyze.

## Ethics

The training data contains real phishing URLs. Don't visit them. The attacks
and the demo only rewrite strings offline. The transformations are ones
phishers already use; the point is to show defenders which cheap edits these
models over-trust.
