# Phishing Detector Adversarial Robustness

[![CI](https://github.com/mikedsosa/phishing-detector-adversarial-robustness/actions/workflows/ci.yml/badge.svg)](https://github.com/mikedsosa/phishing-detector-adversarial-robustness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

**How robust are URL-based phishing detectors to the cheap, realistic edits a phisher can actually make?**

This research project trains two URL-only phishing detectors, a scikit-learn
gradient-boosting model (GBM) and a PyTorch character-level CNN, and attacks
them in the **problem space**: every adversarial example passes our URL
validity and ownership checks (syntax, unchanged registrable domain). We don't
check that the rewritten page still serves. It then measures what several
defenses buy and cost, on two public datasets: PhiUSIIL and the newer,
time-split PhreshPhish.

> **Background:** this project builds on my earlier university research
> interest in adversarial machine learning and phishing detection. It is a new,
> independent project using only public data.

📄 **Paper:** [`docs/paper.md`](docs/paper.md) · ✍️ **Short write-up:** [`docs/writeup.md`](docs/writeup.md) · 🎯 **Threat model:** [`docs/threat-model.md`](docs/threat-model.md) · 🗂️ **Model card:** [`MODEL_CARD.md`](MODEL_CARD.md) · 📊 **Results:** [`results/SEEDS.md`](results/SEEDS.md), [`results/defenses/DEFENSES.md`](results/defenses/DEFENSES.md), [`results/drift/DRIFT.md`](results/drift/DRIFT.md), [`results/operating/`](results/operating/)

![Phishing URLs still caught: no attack vs. greedy, held-out-edit and beam-search attacks, baseline vs. adversarially trained, both datasets, 95% CIs](results/hero.png)

*Share of 1,000 sampled phishing test URLs per seed that each model flags and
still flags after the attack (same denominator for every model), pooled over
seeds 42–44, 95% Wilson intervals. The PhiUSIIL panel uses the domain-grouped
split (no registrable domain in more than one split); random-split results are
compared [below](#phiusiil-random-split-vs-domain-grouped-split). Data: [`results/hero_data.csv`](results/hero_data.csv).*

## Quickstart (about 10 minutes, offline after install)

```bash
git clone https://github.com/mikedsosa/phishing-detector-adversarial-robustness.git
cd phishing-detector-adversarial-robustness
python3.12 -m venv .venv && source .venv/bin/activate   # or: uv venv -p 3.12 .venv
make install        # pinned requirements, CPU-only PyTorch
make test           # unit tests, including small end-to-end runs on the bundled samples
make demo-sample    # trains small models on the bundled PhiUSIIL sample and attacks one example URL
```

`make demo-sample` prints the model's score for an example URL and the
smallest set of edits that flips it, defanged. The full study needs the
datasets (`make data` downloads about 36 MB) and several CPU hours; see
[Reproduce](#reproduce).

## Threat model and limitations (short version)

* **The attacker owns their URL.** In the main `owner` tier the phisher runs
  the server and DNS for their domain, so they can change the scheme,
  subdomain labels, path, query and fragment, but not the registrable domain.
  Under a stricter `append_only` tier (roughly a compromised site),
  evasion stays at or below 1.2%. Attacks get unlimited score queries unless
  stated; query-budget and label-only results are [below](#attacker-knowledge-query-budgets-and-label-only-access).
* **URL-only detectors.** Real filters also use page content, hosting and
  reputation, so these are upper bounds for one layer, not for a mail or
  browser pipeline.
* **Dataset artifacts.** 99.97% of PhiUSIIL's legitimate URLs are bare
  `https://www.<domain>` homepages, and PhreshPhish stripped benign query
  strings for privacy. Both are shortcuts a model can learn. PhiUSIIL's random
  split also puts 38.6% of phishing test URLs (mean of seeds 42–44; 38.5–38.7%
  per seed) on a registrable domain already seen in phishing training data, so
  we add a **domain-grouped split**.
* **Seed instability.** On PhiUSIIL (random split) the baseline CNN's greedy
  evasion ranges from 50.6% to 63.0% across seeds 42–44. Three seeds show
  spread; they are not a variance study.
* **Defenses are upper bounds on robustness.** Adversarial training runs two
  rounds against hand-written edits; stronger or LLM-written rewrites could do
  better than our attacks.

Full details: [`docs/threat-model.md`](docs/threat-model.md) and
[`docs/paper.md` §5](docs/paper.md#5-limitations-and-honest-negatives).

## Key findings

Three seeds (42–44) per dataset unless stated. Brackets are 95% Wilson
intervals on evasions pooled over seeds.

1. **Fairer data made the detectors easier to evade, not harder.** Greedy
   black-box evasion with at most 10 edits is 63.5% (GBM) and 57.9% (CNN) on
   PhiUSIIL's random split, but 84.9% and 95.1% on PhreshPhish, usually within
   two edits.
2. **Adversarial training mostly teaches the model the attacker it saw.** On
   PhreshPhish it cuts greedy evasion to 36.7% (GBM) and 22.9% (CNN). Against
   edit types left out of training it helps less (39.5% and 62.1%), a wider
   beam search still gets through 72.5% and 98.7% of the time, and the GBM's
   false-positive rate goes up on every seed (the CNN's only on seed 42).
3. **Both benchmarks have shortcuts.** On PhiUSIIL a one-line "is it a bare
   `https://www.` homepage" rule scores 0.9948 F1. On PhreshPhish, benign
   URLs had their query strings stripped for privacy, so "has a query string"
   flags phishing with 99.7% precision.
4. **PhiUSIIL's random split leaks domains.** 38.6% of phishing test URLs
   (mean of seeds 42–44; 38.5–38.7% per seed) share a registrable domain with a phishing
   training URL. Re-running on a split grouped by domain leaves the
   GBM's results within sampling error (greedy evasion 61.7% vs. 63.5%) and
   the CNN's slightly lower (54.5% vs. 57.9%), so the leak is not what makes
   PhiUSIIL easy.
5. **Models don't carry over between datasets or months.** PhiUSIIL-trained
   models flag essentially all of PhreshPhish's legitimate URLs, and
   the PhreshPhish GBM's phishing recall falls from 95.9% on validation data to
   58.7% in the first month after its training data ends (seed 42).
6. **Stricter thresholds make evasion easier.** Tuning the PhreshPhish GBM to
   a 1% validation false-positive rate (seed 42) lets the greedy attack evade
   91.7% [89.7, 93.4] of flagged URLs, up from 83.2% at the default
   threshold.

## Results

### PhreshPhish (time split), seeds 42–44

Evasion rate = share of the phishing URLs a model flags at p ≥ 0.5 that the
attack pushes below 0.5. Cells: mean over seeds, pooled 95% Wilson interval
[brackets] and per-seed range. Denominators are the initially flagged URLs,
which differ by model (2,729–2,834 pooled over the three seeds for these four
models; on seed 42 alone, 910 for the GBM vs. 962 for the CNN + adv.
training), so a lower evasion rate can partly reflect a model flagging more
URLs to begin with. The hero figure above uses the same 1,000-URL denominator
for every model and is the fairer comparison. Clean F1 and FPR cells show the
mean and per-seed range; we don't compute intervals for them (test n = 168,059).

| | GBM | Char-CNN | GBM + adv. training | Char-CNN + adv. training |
|---|---|---|---|---|
| Clean test F1 | 0.931 (0.929–0.931) | 0.950 (0.947–0.951) | 0.932 (0.932–0.932) | 0.951 (0.949–0.955) |
| Clean test FPR | 3.06% (2.99–3.12) | 2.57% (2.21–3.13) | 3.86% (3.84–3.88) | 3.17% (2.35–4.80) |
| Greedy, ≤10 edits | 84.9% [83.5, 86.2] (83.2–86.1) | 95.1% [94.2, 95.8] (94.2–95.6) | 36.7% [35.0, 38.6] (34.7–40.0) | 22.9% [21.4, 24.4] (17.6–26.9) |
| Held-out edits, ≤10 | 49.3% [47.4, 51.2] (48.2–49.8) | 73.2% [71.5, 74.8] (69.3–75.5) | 39.5% [37.7, 41.4] (35.2–41.8) | 62.1% [60.3, 63.8] (55.1–66.8) |
| Beam search, all 19 edits, ≤20 | 96.1% [95.3, 96.7] (95.4–96.9) | 99.96% [99.8, 99.99] (99.9–100) | 72.5% [70.8, 74.1] (70.0–75.0) | 98.7% [98.2, 99.1] (97.6–99.4) |
| Random edits, label-only, ≤10 | 27.0% [25.4, 28.7] (26.0–28.9) | 25.9% [24.3, 27.6] (20.4–28.9) | 11.8% [10.6, 13.0] (10.7–13.6) | 7.9% [6.9, 8.9] (4.6–10.1) |

Adversarial training raises the GBM's false-positive rate on every seed
(3.1% to 3.9% on seed 42; 3.84–3.88% vs. 2.99–3.12% across seeds). For the CNN
it rose from 2.2% to 4.8% on seed 42, but not on seeds 43 and 44 (2.3% for
the adversarially trained CNN vs. 2.4% and 3.1% for the baseline), so the
CNN's FPR cost is not consistent across seeds.

### PhiUSIIL: random split vs. domain-grouped split

PhiUSIIL's random split leaks domains: 38.6% of phishing test URLs (mean of
seeds 42–44; 38.5–38.7% per seed) share a registrable domain with a phishing
training URL, against about 2.3% (2.3–2.4%) of legitimate ones ([`results/domain_leakage.csv`](results/domain_leakage.csv)). Round 2 adds a 70/10/20 split
grouped by registrable domain, so no domain appears in more than one split.
For the GBM it changes little: clean scores, greedy and adversarially
trained results stay within sampling error, and beam-search evasion rises by
3 points (seeds 42–44, pooled 95% Wilson
intervals, per-seed range; [`results/phiusiil-grouped-gbm-preview/SUMMARY.md`](results/phiusiil-grouped-gbm-preview/SUMMARY.md)):

| | Random split | Domain-grouped split |
|---|---|---|
| GBM clean F1 (FPR) | 0.9978 (0.04%) | 0.9977 (0.05%) |
| GBM greedy, ≤10 edits | 63.5% [61.8, 65.2] (61.7–64.5) | 61.7% [60.0, 63.5] (58.8–65.5) |
| GBM beam, ≤20 | 79.8% [78.3, 81.2] (77.4–81.2) | 82.8% [81.4, 84.1] (78.8–84.9) |
| GBM + adv. training, greedy | 34.3% [32.6, 36.0] (32.7–35.8) | 34.8% [33.1, 36.6] (33.9–35.4) |
| GBM + adv. training, beam | 41.2% [39.4, 43.0] (38.3–43.2) | 41.4% [39.7, 43.2] (39.0–43.6) |

Clean F1 (FPR) is the mean over seeds, without an interval. Evasion
denominators are 2,989–2,992 initially flagged URLs pooled over three seeds
in every cell.

For the GBM, the leak isn't what makes PhiUSIIL easy; the bare-homepage
format shortcut is. The char-CNN points the same way (seeds 42–44, same
cell format; [`results/SEEDS.md`](results/SEEDS.md)):

| | Random split | Domain-grouped split |
|---|---|---|
| CNN clean F1 | 0.9980 (0.9980–0.9981) | 0.9978 (0.9973–0.9981) |
| CNN greedy, ≤10 edits | 57.9% [56.1, 59.6] (50.6–63.0) | 54.5% [52.7, 56.3] (49.1–64.4) |
| CNN beam, ≤20 | 82.7% [81.3, 84.0] (79.7–84.4) | 82.1% [80.6, 83.4] (77.5–86.6) |
| CNN + adv. training, greedy | 29.8% [28.2, 31.5] (28.9–30.9) | 32.4% [30.8, 34.1] (29.5–35.2) |
| CNN + adv. training, beam | 41.2% [39.4, 43.0] (37.4–44.3) | 46.9% [45.1, 48.7] (43.8–50.4) |

The baseline CNN is no easier to evade on the grouped split, so the leak
wasn't inflating its results. The adversarially trained CNN does somewhat
worse there under beam search (46.9% vs. 41.2%, non-overlapping intervals),
so for that defense the random split was slightly flattering.

Round-1 random-split PhiUSIIL table, seeds 42–44 (full results in
[`results/SEEDS.md`](results/SEEDS.md)). Same cell format as the PhreshPhish
table; denominators are 2,989–2,993 flagged URLs pooled over seeds for each
model:

| | GBM | Char-CNN | GBM + adv. training | Char-CNN + adv. training |
|---|---|---|---|---|
| Clean test F1 | 0.9978 (0.9977–0.9980) | 0.9980 (0.9980–0.9981) | 0.9960 (0.9957–0.9961) | 0.9966 (0.9961–0.9971) |
| Greedy, ≤10 edits | 63.5% [61.8, 65.2] (61.7–64.5) | 57.9% [56.1, 59.6] (50.6–63.0) | 34.3% [32.6, 36.0] (32.7–35.8) | 29.8% [28.2, 31.5] (28.9–30.9) |
| Held-out edits, ≤10 | 0.1% [0.0, 0.3] (0.0–0.2) | 0.0% [0.0, 0.1] (0.0–0.0) | 0.0% [0.0, 0.1] (0.0–0.0) | 0.0% [0.0, 0.1] (0.0–0.0) |
| Beam, ≤20 | 79.8% [78.3, 81.2] (77.4–81.2) | 82.7% [81.3, 84.0] (79.7–84.4) | 41.2% [39.4, 43.0] (38.3–43.2) | 41.2% [39.4, 43.0] (37.4–44.3) |

### Drift

* **Over time (PhreshPhish, seed-42 models).** Training data runs to
  2025-09-08, testing from 2025-09-08 to 2025-12-16. Phishing detection drops
  from 95.9% (GBM) and 95.7% (CNN) on same-period validation rows to 58.7% and 85.0% on
  September 2025 test URLs, then recovers to 91–93% in October and November.
  False positives stay near 2–3% until December, when they rise to 8.4% (GBM)
  and 8.8% (CNN) on 1,017 benign URLs. Full table with CIs:
  [`results/drift/DRIFT.md`](results/drift/DRIFT.md).
* **Across datasets (all seeds).** PhiUSIIL-trained baselines flag 99.99–100%
  of PhreshPhish's legitimate URLs, and the PhreshPhish-trained GBM flags
  97.9–99.9% of PhiUSIIL's (the CNN 37.2–53.4%). Neither dataset alone
  predicts performance on the other.

### Low base rate and low false-positive thresholds (PhreshPhish, seed 42)

Real traffic is mostly benign. Reweighting the test set so phishing is 1% of
URLs, the baseline CNN's precision at 90% recall is 40.0% (GBM: 22.5%); at
0.05% it is 3.2% (GBM: 1.4%). Average precision falls from 0.92 to 0.39 (CNN)
between a 5% and a 0.05% base rate.

A lower threshold doesn't help against the attacks. Choosing the threshold on
validation data for at most 1% or 0.1% FPR and rerunning the attacks there
(95% Wilson intervals; denominators are the URLs flagged at that threshold,
so they shrink as the threshold rises). This is a single-seed (42) analysis;
test FPR and recall are point estimates on the full test split:

| Model | Target val. FPR | Test FPR | Test recall | Greedy | Held-out | Beam |
|---|---|---|---|---|---|---|
| GBM | default (t = 0.5) | 3.06% | 89.9% | 83.2% [80.6, 85.5] | 48.2% [45.0, 51.5] | 95.9% [94.4, 97.0] |
| GBM | 1% | 1.16% | 85.3% | 91.7% [89.7, 93.4] | 51.4% [48.1, 54.7] | 98.2% [97.0, 98.9] |
| GBM | 0.1% | 0.12% | 72.7% | 99.2% [98.2, 99.6] | 76.3% [73.0, 79.2] | 99.9% [99.2, 100.0] |
| Char-CNN | default | 2.21% | 92.2% | 95.6% [94.1, 96.7] | 75.5% [72.6, 78.2] | 100.0% [99.6, 100.0] |
| Char-CNN | 0.1% | 0.15% | 70.3% | 98.4% [97.2, 99.1] | 74.1% [70.7, 77.2] | 100.0% [99.4, 100.0] |
| GBM + adv. training | 0.1% | 0.12% | 75.0% | 73.8% [70.6, 76.9] | 67.7% [64.3, 70.9] | 93.7% [91.8, 95.2] |
| Char-CNN + adv. training | 0.1% | 0.14% | 75.5% | 38.5% [35.1, 42.0] | 71.7% [68.4, 74.8] | 100.0% [99.5, 100.0] |

A strict threshold flags fewer phishing URLs to begin with, and the attacker
only has to push a score below a higher bar, so evasion goes up. Full tables:
[`results/operating/phreshphish/OPERATING.md`](results/operating/phreshphish/OPERATING.md).

PhiUSIIL's domain-grouped split (seed 42) looks different, because of the
format shortcut. Precision at 90% recall stays at 100.0% even at a 0.05%
base rate (average precision 0.995–0.997 for all four models), and the
default threshold already gives test FPRs of 0.00–0.34%. The thresholds
picked for 1% or 0.1% validation FPR are mostly *below* 0.5 there, so this
isn't a stricter-threshold test; evasion stays in the same range or drops
(baseline CNN greedy: 64.4% [61.4, 67.3] at t = 0.5, 41.3% [38.2, 44.4] at
the 1% target, 49.7% [46.6, 52.8] at 0.1%). Full table:
[`results/operating/phiusiil-grouped/OPERATING.md`](results/operating/phiusiil-grouped/OPERATING.md).

### Attacker knowledge: query budgets and label-only access

The greedy attack needs few queries on PhreshPhish. With at most 50 score
queries per URL it already evades 81.7% (GBM, 79.2–83.5% across seeds) and
90.8% (CNN, 86.7–93.1%) of flagged URLs; with 10 queries, 17.0% (16.1–18.4%)
and 18.1% (15.7–19.9%). These are means of the per-seed rates; we don't
compute pooled intervals for the query curves.
An attacker who only sees the verdict, never the score, and applies random
valid edits until the URL passes still evades 27.0% (GBM) and 25.9% (CNN)
within 10 edits (table above). Per-budget curves: `evasion_vs_queries.csv` in
each results folder.

### Round-2 defenses: mixed adversarial training and ensembles

Round 2 adds two defenses. *Mixed* adversarial training also adds
beam-search trajectories (owner-tier edits only, so held-out edits stay held
out) to each training round. The *ensemble* averages the GBM's and CNN's
scores. Every model was retrained from scratch for these runs, so baseline
numbers differ a little from the tables above.

PhreshPhish, seeds 42–44. Cells are the share of the 1,000 sampled phishing
URLs per seed that the model flags and still flags after the attack (same
denominator for every model), pooled over seeds with 95% Wilson intervals.
Clean FPR is the mean over seeds on the full test split:

| Model | Clean FPR | No attack | Greedy, ≤10 | Held-out, ≤10 | Beam, ≤20 |
|---|---|---|---|---|---|
| GBM (baseline) | 3.1% | 91.0% | 13.7% [12.5, 15.0] | 46.1% [44.4, 47.9] | 3.6% [3.0, 4.3] |
| GBM + adv. training | 3.9% | 92.0% | 58.2% [56.4, 60.0] | 55.6% [53.8, 57.4] | 25.3% [23.8, 26.9] |
| GBM + mixed adv. training | 4.5% | 92.2% | 65.5% [63.7, 67.1] | 58.7% [56.9, 60.4] | 25.9% [24.4, 27.5] |
| Char-CNN (baseline) | 2.7% | 93.6% | 3.9% [3.3, 4.7] | 25.2% [23.6, 26.8] | 0.0% [0.0, 0.1] |
| Char-CNN + adv. training | 3.8% | 95.4% | 77.1% [75.6, 78.6] | 36.6% [34.9, 38.3] | 1.1% [0.8, 1.5] |
| Char-CNN + mixed adv. training | 3.5% | 95.0% | 76.8% [75.3, 78.3] | 29.8% [28.2, 31.5] | 0.6% [0.4, 0.9] |
| GBM+CNN ensemble (baseline) | 2.1% | 93.7% | 4.9% [4.2, 5.8] | 35.4% [33.7, 37.2] | 0.3% [0.2, 0.6] |
| GBM+CNN ensemble + adv. training | 2.9% | 95.2% | 74.1% [72.5, 75.6] | 53.7% [51.9, 55.5] | 18.6% [17.2, 20.0] |
| GBM+CNN ensemble + mixed adv. training | 2.9% | 95.2% | 76.7% [75.1, 78.1] | 52.5% [50.7, 54.2] | 15.2% [14.0, 16.5] |

Paired comparisons are exact McNemar tests on the same attacked URLs, pooled
over seeds:

* **Mixed adversarial training helps the GBM against the attack it saw, not
  against a stronger one.** Greedy: 65.5% vs. 58.2% caught (p = 1.5e-33);
  held-out edits: 58.7% vs. 55.6% (p = 6.6e-11); beam: 25.9% vs. 25.3%
  (p = 0.27). Clean FPR goes from 3.9% to 4.5%.
* **For the CNN it doesn't help.** Greedy is unchanged (76.8% vs. 77.1%,
  p = 0.64), and it catches fewer URLs under held-out edits (29.8% vs. 36.6%,
  p = 1.7e-25) and beam search (0.6% vs. 1.1%, p = 0.00027).
* **The ensemble doesn't beat its best member under attack.** Against beam
  search the adversarially trained ensemble catches 18.6% vs. 25.3% for the
  adversarially trained GBM alone (p = 1e-36); on held-out edits, 53.7% vs.
  55.6% (p = 0.0021). It does have a lower clean FPR (2.9% vs. 3.9%).
* **Nothing holds up to beam search on PhreshPhish.** The best defense,
  GBM + mixed adversarial training, still lets about three in four flagged
  URLs through.
* **PhiUSIIL (domain-grouped split):** held-out edits evade essentially none (0.0%) of the
  models, and mixed training gives small but significant gains under beam
  search (GBM 60.7% vs. 58.4% caught, CNN 57.8% vs. 52.9%, ensemble 61.7% vs.
  56.0%; all p < 1e-8), with clean FPRs of 0.2–0.7%.

Full tables for both datasets and every paired test:
[`results/defenses/DEFENSES.md`](results/defenses/DEFENSES.md).

### Other findings carried over from round 1

* **Feature-space attacks mislead.** ART's HopSkipJump "evades" almost every
  time in feature space, but only 0% (PhiUSIIL) and 6% (PhreshPhish) of its
  outputs map to a realizable URL (seed 42, 200 URLs).
* **Domain-only models** can't be evaded by the edits in our threat model
  but reach only 0.75–0.86 F1. Wrapping the URL in a link shortener, which is
  outside the threat model, still evades the domain-only GBM about 24% of the
  time.
* **Canonicalization** (stripping attacker-controlled formatting before
  scoring) helps the PhreshPhish GBM (greedy 83.2% → 56.9%, seed 42) far more
  than the CNN. We did not build an attack aimed specifically at it.

![Evasion vs. perturbation budget (PhreshPhish, seed 42)](results/phreshphish/evasion_vs_perturbations.png)

## Related work

Full references are in [`docs/paper.md`](docs/paper.md#references). The
closest prior work is Sabir, Gaire & Abuadbba, "Reliability and Robustness
analysis of Machine Learning based Phishing URL Detectors" (IEEE TDSC 2022,
doi:10.1109/TDSC.2022.3218043), whose URLBUG attack shows the same weakness
to cheap URL edits. The feature-space vs. problem-space framing comes from
Pierazzi et al. (IEEE S&P 2020, doi:10.1109/SP40000.2020.00073) and, for
phishing, SpacePhish (ACSAC 2022, doi:10.1145/3564625.3567980) and
Multi-SpacePhish (ACM DTRAP 2024, doi:10.1145/3638253). Earlier URL-evasion
work includes Shirazi et al. (DBSec 2019), AlEroud & Karabatis (IWSPA 2020)
and DeepPhish: A. Correa Bahnsen, I. Torroledo, L. D. Camacho and
S. Villegas, "DeepPhish: Simulating Malicious AI," an author-hosted preprint
presented as a talk at APWG eCrime 2018 and Black Hat Europe 2018
([PDF](https://albahnsen.wordpress.com/wp-content/uploads/2018/05/deepphish-simulating-malicious-ai_submitted.pdf)).
We found no peer-reviewed version of DeepPhish, so we cite it as a preprint
and talk. It trains an LSTM to generate phishing URLs that slip past an ML
detector. For defenses and evaluation practice we follow Tramèr & Boneh
(NeurIPS 2019) and Kang et al. (arXiv:1908.08016) on robustness to unseen
perturbations, Tramèr et al. (NeurIPS 2020) and Carlini et al.
(arXiv:1902.06705) on adaptive attacks, and Arp et al. (USENIX Security
2022) and TESSERACT (USENIX Security 2019) on pitfalls in security ML.

## How it works

```mermaid
flowchart LR
    D[(PhiUSIIL + PhreshPhish<br/>CC BY 4.0)] -->|download + SHA-256| P[Parse / dedupe /<br/>random or time split]
    P --> M1[GBM on 30 lexical features]
    P --> M2[Char-CNN on raw URL]
    M1 & M2 --> A{Problem-space attacks}
    A --> A1[Greedy black-box,<br/>held-out edits, beam search]
    A --> A2[HotFlip + hybrid<br/>white-box]
    A --> A3[ART HopSkipJump<br/>feature-space baseline]
    A1 & A2 --> R[Defenses:<br/>adv. training,<br/>canonicalization,<br/>domain-only]
    R --> O[results/*.csv, *.json, *.png]
```

| Component | Where |
|---|---|
| URL parsing, validity constraints, canonicalization | `src/phishadv/urls.py` |
| Lexical features | `src/phishadv/features.py` |
| GBM and Char-CNN detectors | `src/phishadv/models.py` |
| Attacks (greedy, held-out, beam, random, HotFlip, hybrid, append-only, shortener) | `src/phishadv/attacks.py` |
| ART HopSkipJump feature-space baseline | `src/phishadv/feature_space.py` |
| Experiment driver, report, cross-seed aggregation | `src/phishadv/experiment.py`, `report.py`, `aggregate.py` |
| Single-URL demo CLI and local Streamlit app | `src/phishadv/cli.py`, `app/streamlit_app.py` |

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
make install      # pinned requirements (CPU PyTorch)
make reproduce    # downloads both datasets, verifies checksums, runs seeds 42-44 on both, aggregates (about 3 h on 8 CPU cores; each of the six runs takes 28–31 min)
```

Round 2 (domain-grouped PhiUSIIL split, new defenses, drift, operating
points; several more CPU hours):

```bash
make round2       # grouped PhiUSIIL seeds 42-44, defense runs on both datasets, then aggregate, drift, operating, defense_report
```

`make reproduce` runs `reproduce-main` (seed 42 on both datasets, which also
saves the demo models to `models/`), then `seeds` (43 and 44), then
`aggregate`. Other targets:

```bash
make smoke        # fast end-to-end runs on the bundled samples -> results-smoke/
make test         # pytest
make lint         # ruff
make scan         # gitleaks secret scan
make models       # train and save the demo models only
```

To keep compute modest, models train on the full training split (PhreshPhish:
a fixed 164,759-URL subsample of it), but attacks are subsampled: 1,000 test
phishing URLs (HopSkipJump: 200), and adversarial training uses 4,000
training phishing URLs per round. Seeds are fixed and PyTorch deterministic
mode is on.

## Demo (local only)

The CLI scores **one** URL and searches for the smallest set of edits that
flips the verdict. There is no batch mode on purpose. Every URL it prints is
defanged (`hxxp`, `[.]`), and it never resolves or visits anything.

```bash
make models                                   # or make reproduce-main
make demo URL='http://secure-login.example-account.xyz/verify.php?id=1'
pip install -r requirements-demo.txt && make app   # Streamlit on localhost:8501
```

The demo is not deployed anywhere.

## Docker

```bash
docker build -t phishadv . && docker run --rm phishadv
```

The image runs the unit tests and the smoke test, then trains small models on
the bundled sample and scores one example URL.
Docker itself wasn't available where this was developed. The image was built
with rootless podman, using a copy of the Dockerfile without `--chown` and
`USER` (the build sandbox couldn't change file owners). A full container run
hasn't been completed yet, so treat the Dockerfile as lightly tested.

## Limitations and future work

* URL-only detectors. Real filters also see page content, hosting and
  reputation signals, so these evasion rates are an upper bound for a
  URL-only layer, not for a full email or browser pipeline.
* Both benchmarks have collection artifacts (homepage-only benign URLs in
  PhiUSIIL; stripped benign query strings in PhreshPhish) that the models can
  exploit. The `format_rule` and `query_rule` baselines make them visible.
* PhiUSIIL's random split puts many same-domain phishing URLs in both train
  and test. Round-1 PhiUSIIL numbers use that split; the domain-grouped
  numbers above are the ones to compare against.
* The main `owner` threat tier assumes the attacker runs the server and DNS.
  Under the conservative `append_only` tier (compromised sites), evasion is
  at most 1.2%. Validity is checked syntactically: we do not check that a
  rewritten URL still serves the phishing page.
* Attacks get unlimited score queries unless stated. The label-only attack
  is a simple random search, not a state-of-the-art decision-based attack.
* Three seeds (42–44) per dataset: enough to show run-to-run spread, not a
  variance study. Some round-2 analyses use seed 42 only and say so.
* Adversarial training and the round-2 defenses are evaluated against our
  own hand-written attacks, so their robustness is an upper bound.

**Future work:** URLNet- and transformer-based detectors; an adaptive
attack aimed at canonicalization; LLM-written URL rewrites; the gray-box
defense of Apruzzese & Subrahmanian (paper ref. [17]); checking that
rewritten URLs still resolve to the same content.

See [`docs/paper.md` §5](docs/paper.md#5-limitations-and-honest-negatives) and
[`MODEL_CARD.md`](MODEL_CARD.md) for the full list.

## Data and ethics

| Dataset | Source | License | Split |
|---|---|---|---|
| PhiUSIIL (Prasad & Chandra, 2024) | [UCI](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) | CC BY 4.0 | random 70/10/20, and 70/10/20 grouped by registrable domain (round 2) |
| PhreshPhish v1.0.1 (Dalton et al., 2025) | [Hugging Face](https://huggingface.co/datasets/phreshphish/phreshphish) | CC BY 4.0 (authors ask that it be used for anti-phishing research) | the dataset's own time split |

Raw data is not committed. `make data` downloads only the dataset files (for
PhreshPhish, just the URL and label columns via HTTP range requests) and
checks SHA-256 hashes; see [`data/README.md`](data/README.md). The attacks
only rewrite strings offline. No URL is ever resolved or visited, and example
URLs in the results are defanged. The CI samples in `tests/fixtures/`
contain real phishing URLs, so **don't visit them**.

## License

[MIT](LICENSE) © 2026 mikedsosa. Datasets © their creators, CC BY 4.0.
