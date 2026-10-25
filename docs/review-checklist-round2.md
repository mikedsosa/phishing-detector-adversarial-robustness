# Round-2 review checklist

Status of each item in the round-1 external review (commit c848f23). "Waiting"
means the docs carry a marked TODO until the listed results exist; no numbers
were filled in ahead of the results.

## 1. Prior work

| Item | Status | Where |
|---|---|---|
| Add Sabir et al. 2022 (closest prior work) | Done | paper §1, ref [15]; README Related work |
| Add Multi-SpacePhish, Apruzzese & Subrahmanian, Tramèr & Boneh, Kang et al., Tramèr et al. 2020, Carlini et al. 2019, Arp et al., TESSERACT, Real Attackers | Done | paper refs [6], [16]–[23] |
| DOIs for Pierazzi, SpacePhish, Shirazi, AlEroud, ref [6] | Done | paper refs [5]–[9] |
| DeepPhish cited as preprint/talk, not peer-reviewed | Done | paper ref [25]; README Related work |
| Drop Lee et al. ESORICS 2023 framing credit | Done | framing credited to Pierazzi [5] and SpacePhish [7] |
| PhishGAN tangential at most | Done | not cited |

## 2. Claims audit

| Item | Status |
|---|---|
| 1. 92.6% → 92.5% non-root path | Done (writeup, paper, data/README) |
| 2. "Every" → "99.97% of" | Done (data/README) |
| 3. Domain-only "can't be evaded": add shortener caveat | Done (abstract, §4, writeup, README) |
| 4. "valid URL the attacker could deploy" wording | Done (README intro) |
| 5. "defeat" → "evade" | Done (paper) |
| 6. CIs or single-seed footnotes on seed-42 tables | Done (paper §4.2, §4.4, §4.7; README) |
| 7. Differing denominators; AT CNN FPR 2.2% → 4.8% | Done (README, paper, MODEL_CARD; also notes the CNN rise does not repeat on seeds 43–44) |
| 8. Python version alignment | Done (README, pyproject and Dockerfile all 3.12) |

## 3. Datasets

| Item | Status |
|---|---|
| PhiUSIIL UCI dataset citation | Done (data/README, paper ref [24]) |
| PhreshPhish license and research-use request | Done (README, data/README, fixtures README) |
| Fixtures license / research-only / do-not-visit note | Done (tests/fixtures/README.md) |

## 4. Gaps

| Item | Status |
|---|---|
| 1. Domain-grouped PhiUSIIL split | GBM: done (README, paper §4.10). Char-CNN: **waiting** on aggregation of the grouped seeds (TODO at the ROUND2-GROUPED-CNN markers) |
| 2. Base rate and low-FPR operating point | PhreshPhish: done (paper §4.11, README). PhiUSIIL grouped: **waiting** (TODO at ROUND2-PAPER) |
| 3. Query budget / label-only | Done (paper §4.9, README) |
| 4. Liveness and owner-tier assumption in abstract | Done |
| 5. Baselines (URLNet, transformer, deployed detector, A&S defense) | Stated as limitations and future work (paper §5, README) |
| 6. Adaptive attacks; AT as upper bound | Done (paper §1 and §5, README); LLM rewrites and canonicalization attack listed as not done |
| 7. Paired tests between defenses | Limitation stated for round-1 tables. McNemar results: **waiting** on grouped PhiUSIIL defense runs (seeds 43, 44) and `phishadv.defense_report` (TODO at ROUND2-DEFENSES / ROUND2-PAPER) |
| 8. Single source per dataset; PSL private section | Done (paper §5) |
| 9. Reproducibility | Not applicable (verified good by the review) |
