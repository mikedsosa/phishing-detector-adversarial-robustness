"""URL parsing, recomposition, validity checks and canonicalization.

All URLs handled by the models are ASCII. Internationalized host labels are
kept in their IDNA A-label form (``xn--...``), which is what a browser puts on
the wire and what most mail and web gateways log.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from urllib.parse import urlsplit

import tldextract
from confusable_homoglyphs import confusables

# Offline Public Suffix List snapshot bundled with tldextract (no network).
# Private-section suffixes (e.g. firebaseapp.com, pantheonsite.io, github.io) are
# treated as public suffixes, so on shared hosting platforms the tenant's own
# label is the registrable domain, which the attacker cannot swap for free.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)

_LDH_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
# RFC 3986 unreserved + reserved + percent sign.
_PATH_QUERY_CHARS = re.compile(r"^[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*$")


@dataclass(frozen=True)
class URLParts:
    scheme: str
    subdomain: str  # dot-separated labels, may be ""
    domain: str  # registrable label (or IP address)
    suffix: str  # public suffix, may be "" (e.g. IP hosts)
    port: str  # "" or ":8080"
    path: str  # "" or starts with "/"
    query: str  # without leading "?"
    fragment: str  # without leading "#"

    @property
    def host(self) -> str:
        return ".".join(p for p in (self.subdomain, self.domain, self.suffix) if p)

    @property
    def registrable(self) -> str:
        return ".".join(p for p in (self.domain, self.suffix) if p)

    @property
    def sub_labels(self) -> list[str]:
        return self.subdomain.split(".") if self.subdomain else []

    @property
    def anchor_label(self) -> str | None:
        """Rightmost subdomain label unless it is ``www``.

        Treated as fixed: on hosting platforms not listed in the PSL, this
        label is typically the tenant's site name, so removing or changing it
        would point the URL at someone else's content.
        """
        labs = self.sub_labels
        return labs[-1] if labs and labs[-1] != "www" else None

    @property
    def path_segments(self) -> list[str]:
        return [s for s in self.path.split("/") if s]

    def with_(self, **kw) -> URLParts:
        return replace(self, **kw)


def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _ensure_scheme(url: str) -> str:
    url = url.strip()
    if "://" not in url[:12]:
        url = "http://" + url
    return url


@lru_cache(maxsize=500_000)
def parse(url: str) -> URLParts:
    """Split a URL into the parts an attacker may or may not control."""
    s = urlsplit(_ensure_scheme(url))
    host = (s.hostname or "").lower().rstrip(".")
    port = ""
    try:
        if s.port is not None:
            port = f":{s.port}"
    except ValueError:
        port = ""
    if is_ip(host):
        sub, dom, suf = "", host, ""
    else:
        ext = _EXTRACT(host)
        sub, dom, suf = ext.subdomain, ext.domain, ext.suffix
        if not dom:  # host is itself a public suffix or unparseable
            sub, dom, suf = "", host, ""
    return URLParts(
        scheme=(s.scheme or "http").lower(),
        subdomain=sub,
        domain=dom,
        suffix=suf,
        port=port,
        path=s.path,
        query=s.query,
        fragment=s.fragment,
    )


def compose(p: URLParts) -> str:
    url = f"{p.scheme}://{p.host}{p.port}{p.path}"
    if p.query:
        url += "?" + p.query
    if p.fragment:
        url += "#" + p.fragment
    return url


def is_valid_url(url: str, original: str | None = None) -> bool:
    """Return True if ``url`` is a syntactically valid, resolvable-looking URL.

    When ``original`` is given, also enforce the threat-model constraint that
    the registrable domain (eTLD+1), port and anchor subdomain label are unchanged and
    that every host label is either a valid LDH label or was already present
    in the original URL (some real-world URLs contain underscores).
    """
    if not url or any(ch.isspace() for ch in url) or not url.isascii():
        return False
    try:
        s = urlsplit(url)
    except ValueError:
        return False
    if s.scheme not in ("http", "https") or not s.hostname:
        return False
    p = parse(url)
    host = p.host
    if len(host) > 253:
        return False
    orig_labels: set[str] = set()
    if original is not None:
        o = parse(original)
        if o.registrable != p.registrable or o.port != p.port:
            return False
        if o.anchor_label is not None and (not p.sub_labels or p.sub_labels[-1] != o.anchor_label):
            return False
        orig_labels = set(o.host.split("."))
    if not is_ip(host):
        for label in host.split("."):
            if label in orig_labels:
                continue
            if not _LDH_LABEL.match(label):
                return False
    rest = url.split(host, 1)[1] if host in url else ""
    return bool(_PATH_QUERY_CHARS.match(rest))


# --------------------------------------------------------------------------
# Canonicalization (defense): undo cheap, attacker-controlled cosmetics.
# --------------------------------------------------------------------------


def _label_to_unicode(label: str) -> str:
    if label.startswith("xn--"):
        try:
            return label.encode("ascii").decode("idna")
        except UnicodeError:
            return label
    return label


@lru_cache(maxsize=200_000)
def skeleton(text: str) -> str:
    """Map confusable characters to a Latin/ASCII skeleton (best effort)."""
    text = unicodedata.normalize("NFKC", text)
    if text.isascii():
        return text
    out = []
    for ch in text:
        if ch.isascii():
            out.append(ch)
            continue
        hits = confusables.is_confusable(ch, preferred_aliases=["latin"], greedy=True)
        repl = None
        if hits:
            for h in hits[0]["homoglyphs"]:
                if h["c"].isascii():
                    repl = h["c"].lower()
                    break
        out.append(repl if repl is not None else "-")
    return "".join(out)


@lru_cache(maxsize=500_000)
def canonicalize(url: str) -> str:
    """Canonical view used by the normalization defense.

    * IDNA A-labels are decoded and confusables are mapped to ASCII
      (homoglyph/punycode canonicalization);
    * the scheme is fixed to ``http`` (TLS is free for attackers);
    * a leading ``www`` label and trailing slashes are removed;
    * the fragment is dropped (never sent to the server).
    """
    p = parse(url)
    labels = [skeleton(_label_to_unicode(lab)).lower() for lab in p.sub_labels]
    if labels and labels[0] == "www":
        labels = labels[1:]
    dom = skeleton(_label_to_unicode(p.domain)).lower()
    path = p.path.rstrip("/")
    return compose(
        p.with_(scheme="http", subdomain=".".join(labels), domain=dom, path=path, fragment="")
    )


def domain_view(url: str) -> str:
    """Only the registrable domain: the part the attacker cannot cheaply change."""
    p = parse(url)
    return f"http://{p.registrable}"


VIEWS = {
    "raw": lambda u: u,
    "canon": canonicalize,
    "domain": domain_view,
}
