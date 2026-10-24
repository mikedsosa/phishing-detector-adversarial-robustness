"""Problem-space evasion attacks on phishing URL detectors.

The attacker starts from a phishing URL and may only apply *transformations*
that keep the URL valid and functional and that touch parts of the URL the
attacker actually controls (see ``docs/threat-model.md``):

* the scheme (free TLS certificates),
* subdomain labels under the attacker's registrable domain,
* path segments, query string and fragment on the attacker's server.

The registrable domain (eTLD+1) and port are never changed. Every candidate is
checked with :func:`phishadv.urls.is_valid_url` against the original URL.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from .models import PAD, CNNDetector, encode_chars
from .urls import compose, is_valid_url, parse

BENIGN_PATH_TOKENS = ("index", "home", "en", "about", "docs", "news", "blog", "help")
BENIGN_SUB_LABELS = ("www", "m", "app", "web", "mail", "shop", "blog", "cdn")
# Latin -> visually confusable Cyrillic/Greek letters (Unicode confusables).
HOMOGLYPHS = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441",
              "x": "\u0445", "y": "\u0443", "i": "\u0456", "s": "\u0455", "j": "\u0458"}

Transform = Callable[[str, random.Random], list[str]]


def _set_https(u, rng):
    p = parse(u)
    return [compose(p.with_(scheme="https"))] if p.scheme != "https" else []


def _toggle_www(u, rng):
    p = parse(u)
    labs = p.sub_labels
    if labs and labs[0] == "www":
        return [compose(p.with_(subdomain=".".join(labs[1:])))]
    if not labs:
        return [compose(p.with_(subdomain="www"))]
    return []


def _drop_query_fragment(u, rng):
    p = parse(u)
    out = []
    if p.query:
        out.append(compose(p.with_(query="")))
    if p.fragment:
        out.append(compose(p.with_(fragment="")))
    return out


def _shorten_path(u, rng):
    """Re-host the kit higher in the directory tree (attacker controls the server)."""
    p = parse(u)
    segs = p.path_segments
    out = []
    if segs:
        out.append(compose(p.with_(path="/" + "/".join(segs[:-1]))))
        out.append(compose(p.with_(path="/", query="")))
    if p.path == "/":
        out.append(compose(p.with_(path="")))
    return out


def _rename_path_segment(u, rng):
    p = parse(u)
    segs = p.path_segments
    out = []
    for i in range(len(segs)):
        new = list(segs)
        new[i] = rng.choice(BENIGN_PATH_TOKENS)
        out.append(compose(p.with_(path="/" + "/".join(new))))
    return out


def _append_path_token(u, rng):
    p = parse(u)
    segs = p.path_segments
    if len(segs) >= 6:
        return []
    return [compose(p.with_(path="/" + "/".join(segs + [rng.choice(BENIGN_PATH_TOKENS)])))]


def _drop_sub_label(u, rng):
    p = parse(u)
    labs = p.sub_labels
    return [compose(p.with_(subdomain=".".join(labs[:i] + labs[i + 1 :]))) for i in range(len(labs))]


def _replace_sub_label(u, rng):
    p = parse(u)
    labs = p.sub_labels
    out = []
    for i, lab in enumerate(labs):
        new = list(labs)
        new[i] = rng.choice([b for b in BENIGN_SUB_LABELS if b != lab])
        out.append(compose(p.with_(subdomain=".".join(new))))
    return out


def _prepend_sub_label(u, rng):
    p = parse(u)
    labs = p.sub_labels
    if len(labs) >= 3:
        return []
    return [compose(p.with_(subdomain=".".join([rng.choice(BENIGN_SUB_LABELS)] + labs)))]


def _homoglyph_sub_label(u, rng):
    """Swap one Latin letter of a subdomain label for a confusable; emit IDNA A-label."""
    p = parse(u)
    labs = p.sub_labels
    out = []
    for i, lab in enumerate(labs):
        if lab.startswith("xn--"):
            continue
        pos = [j for j, ch in enumerate(lab) if ch in HOMOGLYPHS]
        if not pos:
            continue
        j = rng.choice(pos)
        uni = lab[:j] + HOMOGLYPHS[lab[j]] + lab[j + 1 :]
        try:
            alabel = uni.encode("idna").decode("ascii")
        except UnicodeError:
            continue
        new = list(labs)
        new[i] = alabel
        out.append(compose(p.with_(subdomain=".".join(new))))
    return out


def _strip_chars(u, rng):
    """Remove hyphens/digits from one attacker-controlled label or path segment."""
    p = parse(u)
    out = []
    labs = p.sub_labels
    for i, lab in enumerate(labs):
        cleaned = "".join(ch for ch in lab if ch.isalpha())
        if cleaned and cleaned != lab and not lab.startswith("xn--"):
            new = list(labs)
            new[i] = cleaned
            out.append(compose(p.with_(subdomain=".".join(new))))
    segs = p.path_segments
    for i, s in enumerate(segs):
        cleaned = "".join(ch for ch in s if ch.isalpha() or ch == ".")
        if cleaned and cleaned != s:
            new = list(segs)
            new[i] = cleaned
            out.append(compose(p.with_(path="/" + "/".join(new))))
    return out


BENIGN_QUERY_PARAMS = ("ref=home", "lang=en", "utm_source=newsletter", "utm_medium=email", "src=web", "v=2")
BENIGN_FRAGMENTS = ("top", "main", "content", "home")


def _append_query_param(u, rng):
    """Add an unused query parameter; virtually every web server ignores it."""
    p = parse(u)
    if p.query.count("&") >= 5:
        return []
    kv = rng.choice(BENIGN_QUERY_PARAMS)
    return [compose(p.with_(query=f"{p.query}&{kv}" if p.query else kv))]


def _append_fragment(u, rng):
    """Add a fragment; it is never sent to the server, so it cannot break the page."""
    p = parse(u)
    return [] if p.fragment else [compose(p.with_(fragment=rng.choice(BENIGN_FRAGMENTS)))]


TRANSFORMS: dict[str, Transform] = {
    "set_https": _set_https,
    "toggle_www": _toggle_www,
    "drop_query_fragment": _drop_query_fragment,
    "shorten_path": _shorten_path,
    "rename_path_segment": _rename_path_segment,
    "append_path_token": _append_path_token,
    "drop_sub_label": _drop_sub_label,
    "replace_sub_label": _replace_sub_label,
    "prepend_sub_label": _prepend_sub_label,
    "homoglyph_sub_label": _homoglyph_sub_label,
    "strip_chars": _strip_chars,
    "append_query_param": _append_query_param,
    "append_fragment": _append_fragment,
}

# Capability tiers (see docs/threat-model.md).
#  * "owner": the attacker operates the server and DNS of the phishing site
#    (attacker-registered domain or their own tenant on a hosting platform).
#  * "append_only": a conservative lower bound for compromised sites and shared
#    infrastructure, where existing host labels and paths may identify the
#    content and TLS/DNS are not under the attacker's control. Only additions
#    that cannot change what the server returns are allowed.
TIERS: dict[str, tuple[str, ...]] = {
    "owner": tuple(TRANSFORMS),
    "append_only": ("append_query_param", "append_fragment"),
}


def neighbors(url: str, original: str, rng: random.Random, tier: str = "owner") -> list[str]:
    """All valid one-transformation neighbors of ``url`` (deduplicated, original-checked)."""
    seen, out = {url}, []
    for name in TIERS[tier]:
        for c in TRANSFORMS[name](url, rng):
            if c not in seen and is_valid_url(c, original):
                seen.add(c)
                out.append(c)
    return out


@dataclass
class AttackResult:
    original: list[str]
    adversarial: list[str]
    success: np.ndarray  # bool per URL
    steps: np.ndarray  # perturbations used (np.inf if failed)
    queries: np.ndarray  # model queries used (np.inf if failed)
    final_score: np.ndarray
    trace: list[list[str]] = field(default_factory=list)

    def evasion_curve(self, budgets, by: str = "steps") -> list[float]:
        arr = self.steps if by == "steps" else self.queries
        return [float(np.mean(arr <= b)) for b in budgets]


def greedy_blackbox(model, urls: list[str], max_steps: int = 10, threshold: float = 0.5,
                    seed: int = 0, max_queries: int | None = None, tier: str = "owner") -> AttackResult:
    """Score-based black-box greedy search over problem-space transformations.

    At each step every valid one-transformation neighbor of the current URL is
    scored by the target model (one query each) and the one with the lowest
    P(phishing) is kept. Stops on evasion, when no neighbor improves the score,
    or when the step/query budget is spent.
    """
    rng = random.Random(seed)
    n = len(urls)
    cur = list(urls)
    score = model.predict_proba(cur)
    queries = np.ones(n)
    steps_at = np.full(n, np.inf)
    q_at = np.full(n, np.inf)
    done = score < threshold
    steps_at[done], q_at[done] = 0, 1
    active = ~done
    traces = [[u] for u in urls]
    for step in range(1, max_steps + 1):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        cands, owner = [], []
        for i in idx:
            nb = neighbors(cur[i], urls[i], rng, tier)
            if max_queries is not None:
                nb = nb[: max(0, int(max_queries - queries[i]))]
            cands.extend(nb)
            owner.extend([i] * len(nb))
        if not cands:
            break
        s = model.predict_proba(cands)
        owner = np.asarray(owner)
        for i in idx:
            m = np.flatnonzero(owner == i)
            if len(m) == 0:
                active[i] = False
                continue
            queries[i] += len(m)
            best = m[np.argmin(s[m])]
            if s[best] >= score[i]:
                active[i] = False  # local optimum
                continue
            cur[i], score[i] = cands[best], s[best]
            traces[i].append(cur[i])
            if score[i] < threshold:
                steps_at[i], q_at[i] = step, queries[i]
                active[i] = False
    return AttackResult(list(urls), cur, score < threshold, steps_at, q_at, score, traces)


def random_blackbox(model, urls: list[str], max_steps: int = 10, threshold: float = 0.5,
                    seed: int = 0) -> AttackResult:
    """Baseline: apply random valid transformations, no model feedback during search.

    ``steps`` records the first number of random transformations after which
    the URL evades (the model is queried once per step only to log the curve).
    """
    rng = random.Random(seed)
    cur = list(urls)
    n = len(urls)
    score = model.predict_proba(cur)
    steps_at = np.where(score < threshold, 0, np.inf)
    for step in range(1, max_steps + 1):
        for i in range(n):
            nb = neighbors(cur[i], urls[i], rng)
            if nb:
                cur[i] = rng.choice(nb)
        score = model.predict_proba(cur)
        newly = (score < threshold) & np.isinf(steps_at)
        steps_at[newly] = step
    succ = np.isfinite(steps_at)
    return AttackResult(list(urls), cur, succ, steps_at, steps_at + 1, score)


# --------------------------------------------------------------------------
# White-box gradient-guided attack (HotFlip-style) on the character CNN.
# --------------------------------------------------------------------------

_HOST_CHARS = string.ascii_lowercase + string.digits
_PATH_CHARS = string.ascii_letters + string.digits


def _char_idx(ch: str) -> int:
    return ord(ch) - 30


_HOST_IDS = torch.tensor([_char_idx(c) for c in _HOST_CHARS])
_PATH_IDS = torch.tensor([_char_idx(c) for c in _PATH_CHARS])


def controllable_positions(url: str, original: str | None = None) -> list[tuple[int, str]]:
    """Character positions the attacker may substitute, with their allowed alphabet.

    Only alphanumeric characters inside subdomain labels (excluding IDNA
    labels) and inside path/query/fragment are editable; separators are kept,
    so the URL structure and validity are preserved by construction. If the
    ``original`` URL has an anchor label (see ``URLParts.anchor_label``), that
    label is not editable either.
    """
    p = parse(url)
    fixed_last = original is not None and parse(original).anchor_label is not None
    out: list[tuple[int, str]] = []
    start = len(p.scheme) + 3
    if p.subdomain:
        pos = start
        labs = p.sub_labels
        for li, lab in enumerate(labs):
            if fixed_last and li == len(labs) - 1:
                pos += len(lab) + 1
                continue
            if not lab.startswith("xn--"):
                out.extend((pos + j, "host") for j, ch in enumerate(lab) if ch.isalnum())
            pos += len(lab) + 1
    tail_start = start + len(p.host) + len(p.port)
    for j, ch in enumerate(url[tail_start:]):
        if ch.isalnum():
            out.append((tail_start + j, "path"))
    return out


def _slot_inits(url: str) -> list[tuple[str, int]]:
    """Starting points: the URL itself, or with a fresh attacker-owned label/segment
    of placeholder characters (counts as one extra perturbation)."""
    p = parse(url)
    inits = [(url, 0)]
    if len(p.sub_labels) < 3:
        inits.append((compose(p.with_(subdomain=".".join(["aaaaaa"] + p.sub_labels))), 1))
    if len(p.path_segments) < 6 and not p.query and not p.fragment:
        inits.append((compose(p.with_(path="/" + "/".join(p.path_segments + ["aaaaaa"]))), 1))
    return inits


def hotflip_whitebox(model: CNNDetector, urls: list[str], max_flips: int = 10, top_k: int = 16,
                     threshold: float = 0.5, structural: bool = False,
                     seed: int = 0) -> AttackResult:
    """Gradient-guided character substitution restricted to attacker-controlled spans.

    For each step, the first-order change in the phishing logit for replacing
    character ``x_i`` by ``c`` is estimated as ``(E[c] - E[x_i]) . dL/de_i``
    (Ebrahimi et al., 2018). The ``top_k`` most promising flips are verified
    with an exact forward pass and the best is applied. Only applies to
    detectors using the ``raw`` view (positions map 1:1 to the model input).

    With ``structural=True`` (the *hybrid* attack) every step also considers all
    problem-space transformations from :data:`TRANSFORMS`, scored exactly with
    the white-box model, so one step is either one character flip or one
    transformation. Without it, the attack may additionally start from a fresh
    attacker-owned subdomain label or path segment (one extra perturbation).
    """
    rng = random.Random(seed)
    assert model.view == "raw", "white-box attack is defined on the raw view"
    net = model.net
    net.eval()
    emb_w = net.emb.weight.detach()
    n = len(urls)
    steps_at = np.full(n, np.inf)
    adv, final = list(urls), model.predict_proba(urls)
    for i, url in enumerate(urls):
        if final[i] < threshold:
            steps_at[i] = 0
            continue
        best_url, best_score, best_steps = url, final[i], np.inf
        inits = [(url, 0)] if structural else _slot_inits(url)
        for init_url, init_cost in inits:
            cur = init_url
            for flip in range(max_flips - init_cost + 1):
                s = model.predict_proba([cur])[0]
                if s < best_score:
                    best_url, best_score = cur, s
                if s < threshold:
                    best_steps = min(best_steps, init_cost + flip)
                    break
                if flip == max_flips - init_cost:
                    break
                pos = [(j, kind) for j, kind in controllable_positions(cur, url) if j < 200]
                if not pos and not structural:
                    break
                x = encode_chars([cur])
                e = net.emb(x).detach().requires_grad_(True)
                logit = net.forward_embedded(e, x != PAD)[0]
                logit.backward()
                g = e.grad[0]  # (L, D)
                cand = []
                for j, kind in pos:
                    alphabet = _HOST_CHARS if kind == "host" else _PATH_CHARS
                    ids = _HOST_IDS if kind == "host" else _PATH_IDS
                    delta = (emb_w[ids] - emb_w[x[0, j]]) @ g[j]
                    k = int(torch.argmin(delta))
                    if alphabet[k] != cur[j]:
                        cand.append((float(delta[k]), j, alphabet[k]))
                cand.sort()
                trials = [cur[:j] + c + cur[j + 1 :] for _, j, c in cand[:top_k]]
                trials = [t for t in trials if is_valid_url(t, url)]
                if structural:
                    trials += [t for t in neighbors(cur, url, rng) if t not in trials]
                if not trials:
                    break
                ts = model.predict_proba(trials)
                b = int(np.argmin(ts))
                if ts[b] >= s:
                    break
                cur = trials[b]
            if np.isfinite(best_steps):
                break
        adv[i], final[i], steps_at[i] = best_url, best_score, best_steps
    return AttackResult(list(urls), adv, np.isfinite(steps_at), steps_at, np.full(n, np.nan), final)


def shortener_wrap(urls: list[str], seed: int = 0) -> list[str]:
    """Replace each URL with a plausible link-shortener URL (7-char random slug).

    This changes the registrable domain, so it is outside the main threat model;
    it is reported separately as a single-step evasion baseline.
    """
    rng = random.Random(seed)
    hosts = ("bit.ly", "tinyurl.com", "t.co", "is.gd")
    alpha = string.ascii_letters + string.digits
    return [f"https://{rng.choice(hosts)}/{''.join(rng.choices(alpha, k=7))}" for _ in urls]
