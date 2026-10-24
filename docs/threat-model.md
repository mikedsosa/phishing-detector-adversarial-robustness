# Threat model

This document defines who the attacker is, what they want, what they know, and,
most importantly, what they can actually change. Every attack in
`src/phishadv/attacks.py` is implemented against this model, and the
constraints are enforced in code (`phishadv.urls.is_valid_url`) and in tests
(`tests/test_attacks.py`).

## System under attack

A URL-only phishing classifier of the kind used in mail gateways, secure web
gateways, browser extensions and SOC triage pipelines. It receives a URL string
and returns P(phishing). It sees no page content, no WHOIS data and no
reputation feeds. A URL is blocked or flagged if P(phishing) >= 0.5.

We study two model families:

| Detector | Input | Notes |
|---|---|---|
| Gradient boosting (scikit-learn `HistGradientBoostingClassifier`) | 30 hand-crafted lexical features plus a one-hot of the 30 most common public suffixes | Not differentiable, so it can only be attacked with queries |
| Character-level CNN (PyTorch) | First 200 characters of the URL | Differentiable, so it can also be attacked with gradients |

Each detector can be put behind a preprocessing *view* (see "Defenses" below).
The view is part of the deployed system, so the attacker's URL always goes
through it.

## Attacker goal

**Evasion at test time.** The attacker starts from a phishing URL that the
detector currently flags and wants a *functionally equivalent* URL that the
detector scores below the threshold. Poisoning the training data, stealing the
model and inferring membership are all out of scope.

## Attacker knowledge

| Setting | What the attacker knows | Attacks |
|---|---|---|
| Black-box, score-based | Can submit URLs and read back P(phishing), for example through a "check this link" API or by probing a gateway. Knows nothing about the model's internals. | `greedy_blackbox`, `random_blackbox` |
| White-box | Full access to the character CNN's weights and gradients, which is the worst case (an insider or a leaked model) | `hotflip_whitebox`, `hybrid_whitebox` |
| Adaptive | Knows which defense is deployed (canonicalization, adversarial training) and attacks the full defended pipeline | Every attack is re-run against every defended model. No result is computed with attacks crafted against a different model, except the separately reported transfer experiment. |

## Attacker capabilities (the problem-space constraint)

The attacker controls **their own infrastructure**, not the detector's feature
vector. So we allow only changes that a phisher can make to a live URL while
keeping it working and resolving to the same phishing kit. We define two
capability tiers.

### Tier 1: `owner` (main results)

The attacker runs the phishing site's web server and DNS. That covers an
attacker-registered domain, or their own tenant on a hosting platform (for
example `<tenant>.firebaseapp.com`).

| Part of the URL | Who controls it | Allowed transformations |
|---|---|---|
| Scheme | Attacker, since TLS certificates are free (e.g. ACME/Let's Encrypt) | `http` -> `https` |
| Subdomain labels *to the left of* the anchor label | Attacker, through DNS records | Add or remove `www`, drop a label, replace a label with a common one (`m`, `app`, `web`, ...), prepend a label, remove digits and hyphens, swap one Latin letter for a Unicode confusable (emitted as an IDNA `xn--` label) |
| Path, query and fragment | Attacker, who controls their web server's routing | Drop the query or fragment, move the kit to a parent directory or the web root, rename a path segment to a benign token (`index`, `home`, `en`, ...), append a benign token, remove digits and hyphens from a segment, append an unused query parameter (`ref=home`, `utm_source=newsletter`, ...) or a fragment |
| **Registrable domain (eTLD+1, PSL private section included)** | **Fixed.** Changing it means registering a new domain, which costs money and resets domain reputation. On platforms listed in the Public Suffix List's private section (`firebaseapp.com`, `pantheonsite.io`, `github.io`, ...), the tenant's label *is* the registrable domain, so it is fixed too. | Not allowed |
| **Anchor label**: the rightmost subdomain label, unless it is `www` | **Fixed.** On hosting platforms that aren't in the PSL, this is usually the tenant's site name, and removing or changing it would point the URL at someone else's content. | Not allowed |
| Port | Fixed | Not allowed |

### Tier 2: `append_only` (conservative lower bound)

This tier covers compromised legitimate sites and shared infrastructure
(IPFS gateways, document-sharing links, open redirects). There, the existing
host labels, path and query may *identify the content*, and the attacker
doesn't control TLS or DNS. The only allowed changes are additions that can't
change what the server returns: **append an unused query parameter** and
**append a fragment**.

### Validity checks applied to every candidate

Every candidate URL must:

* use `http` or `https` and be pure ASCII (internationalized labels in A-label form),
* use only valid LDH host labels (letters, digits and hyphens, 1 to 63 characters, no leading or trailing hyphen), with a host of 253 characters or fewer,
* use only RFC 3986 characters in the path, query and fragment,
* keep the original's registrable domain, port and anchor label.

The white-box character attack is stricter still. It may substitute only
alphanumeric characters inside the non-anchor subdomain labels and the path,
query or fragment, so separators, the anchor label and the registrable domain
never change.

**One perturbation** is one transformation from the table, or one character
substitution in the white-box attack. Budgets are counted in perturbations and,
for the black-box attack, in model queries.

### What is deliberately *not* allowed

* **Arbitrary feature-vector edits.** Changing `domain_len` from 14 to 9.37
  doesn't correspond to any URL. The feature-space HopSkipJump baseline (ART)
  is included only to *measure* this gap.
* **Changing the registrable domain or TLD.** In the real world this is a
  legitimate and common attacker move, but it costs money, so it is a separate
  cost tier and is not modelled here.
* **Link shorteners and open redirects.** These replace the visible URL with
  someone else's domain. We report a one-step "shortener wrap" separately,
  outside the main threat model, because a URL-only model can't judge it.

### Realism caveats

* Rehosting a kit at the web root or renaming its paths assumes the attacker
  runs the server. That holds for attacker-registered domains and most free
  hosting, but often not for **compromised** legitimate sites, where the
  attacker may control only one directory. That case is what the
  `append_only` tier is for.
* We can't tell automatically which URLs in the dataset are attacker-owned
  and which are compromised, so the `owner` tier is an *upper bound* on what
  cheap edits achieve and `append_only` is a *lower bound*.
* Homoglyph subdomain labels resolve fine, but modern browsers may display
  them as `xn--...`. That weakens the visual deception aimed at the *human*,
  although it still works against the *model*.
* Free-hosting platforms (e.g. `*.firebaseapp.com`) let the attacker pick
  their tenant name but not the platform domain. The PSL-private-section rule
  and the anchor-label rule together stop the attack from "evading" by
  pointing the URL at the platform's own homepage, a bug we found and fixed
  during development.

## Defenses evaluated

| Defense | Idea | Where |
|---|---|---|
| Canonicalization (input normalization) | Before the model sees the URL, decode IDNA, map confusables to an ASCII skeleton, force the scheme to `http`, strip a leading `www` and trailing slashes, and drop the fragment. These are things an attacker gets for free, so the model shouldn't rely on them. | `phishadv.urls.canonicalize`, view `canon` |
| Feature hardening: domain-only | The model sees only `http://<registrable domain>`, the part the attacker can't change in this threat model | view `domain` |
| Adversarial training | Two rounds of: attack the current model on 4,000 fresh *training* phishing URLs, add every URL on the attack trajectories (labelled phishing) to the training set, and retrain from scratch. For the CNN, hybrid white-box examples are added too. | `experiment.adversarial_training` |
| Canonicalization + adversarial training | Both of the above, for the GBM | model `gbm_canon_at` |

## Assets and impact

A successful evasion means a phishing link reaches the user unflagged, which
can lead to credential theft, session hijacking or malware delivery. For a
security team the key question is: **how much effort does evasion take, and
which cheap moves does the model over-trust?** The evasion-vs-budget curves
answer that.
