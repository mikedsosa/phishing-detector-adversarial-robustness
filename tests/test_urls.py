from phishadv.urls import canonicalize, compose, domain_view, is_valid_url, parse


def test_parse_compose_roundtrip():
    u = "http://login.secure.example.co.uk:8080/a/b.php?x=1#frag"
    p = parse(u)
    assert p.scheme == "http"
    assert p.subdomain == "login.secure"
    assert p.domain == "example"
    assert p.suffix == "co.uk"
    assert p.port == ":8080"
    assert p.path_segments == ["a", "b.php"]
    assert compose(p) == u


def test_ip_host():
    p = parse("http://192.168.10.5/login")
    assert p.domain == "192.168.10.5" and p.suffix == "" and p.subdomain == ""


def test_validity_rejects_registrable_domain_change():
    orig = "http://www.evil-site.com/login"
    assert is_valid_url("https://www.evil-site.com/home", orig)
    assert not is_valid_url("https://www.good-site.com/login", orig)
    assert not is_valid_url("https://www.evil-site.net/login", orig)


def test_validity_rejects_malformed():
    assert not is_valid_url("http://bad_label!.example.com/")
    assert not is_valid_url("http://exa mple.com/")
    assert not is_valid_url("ftp://example.com/")
    assert not is_valid_url("http://-lead.example.com/")
    assert not is_valid_url("http://ex\u0430mple.com/")  # non-ASCII must be IDNA-encoded
    assert is_valid_url("http://xn--exmple-4nf.example.com/a")


def test_canonicalize_undoes_cosmetics():
    a = canonicalize("https://www.example.com/")
    b = canonicalize("http://example.com")
    assert a == b == "http://example.com"
    # Cyrillic 'а' in a subdomain label, IDNA-encoded, maps back to Latin.
    homoglyph = "http://" + "p\u0430ypal".encode("idna").decode() + ".example.com/x"
    assert canonicalize(homoglyph) == "http://paypal.example.com/x"


def test_domain_view():
    assert domain_view("https://a.b.example.org/x?y=1") == "http://example.org"


def test_private_suffix_tenant_is_registrable():
    p = parse("https://my-kit-123.firebaseapp.com/login")
    assert p.registrable == "my-kit-123.firebaseapp.com"
    # Swapping the tenant name or landing on the platform itself is not allowed.
    assert not is_valid_url("https://www.firebaseapp.com/login", "https://my-kit-123.firebaseapp.com/login")
    assert is_valid_url("https://www.my-kit-123.firebaseapp.com/", "https://my-kit-123.firebaseapp.com/login")


def test_anchor_label_is_fixed():
    orig = "https://storefront.someplatform.com/x"
    assert parse(orig).anchor_label == "storefront"
    assert not is_valid_url("https://www.someplatform.com/x", orig)
    assert not is_valid_url("https://shop.someplatform.com/x", orig)
    assert is_valid_url("https://www.storefront.someplatform.com/", orig)
    assert parse("http://www.example.com").anchor_label is None
