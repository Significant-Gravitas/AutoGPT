"""Tests for the HTTP client component."""

import pytest

from forge.utils.exceptions import HTTPError

from .http_client import HTTPClientComponent, HTTPClientConfiguration


@pytest.fixture
def allowlisted_component():
    """Create an HTTPClientComponent restricted to whitelist.com."""
    config = HTTPClientConfiguration(allowed_domains=["whitelist.com"])
    return HTTPClientComponent(config)


def test_allowed_domain_passes(allowlisted_component):
    assert allowlisted_component._is_domain_allowed("https://whitelist.com/path")


def test_subdomain_of_allowed_domain_passes(allowlisted_component):
    assert allowlisted_component._is_domain_allowed("https://api.whitelist.com/path")


def test_no_allowlist_allows_everything():
    component = HTTPClientComponent(HTTPClientConfiguration())
    assert component._is_domain_allowed("http://127.0.0.1/")


def test_disallowed_domain_rejected(allowlisted_component):
    assert not allowlisted_component._is_domain_allowed("http://127.0.0.1/")


def test_userinfo_backslash_bypass_rejected(allowlisted_component):
    # Regression: netloc-based check passed but request hit 127.0.0.1.
    assert not allowlisted_component._is_domain_allowed(
        "http://127.0.0.1:6666\\@.whitelist.com"
    )


def test_userinfo_bypass_rejected(allowlisted_component):
    assert not allowlisted_component._is_domain_allowed(
        "http://127.0.0.1@whitelist.com.evil.com/"
    )


def test_make_request_blocks_internal_address_without_allowlist():
    # Even with the default empty allowlist, _make_request must reject
    # internal/private targets before issuing the request.
    component = HTTPClientComponent(HTTPClientConfiguration())
    with pytest.raises(HTTPError):
        component._make_request("GET", "http://127.0.0.1/")
