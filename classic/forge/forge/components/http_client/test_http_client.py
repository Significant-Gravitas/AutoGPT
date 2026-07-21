"""Tests for the HTTP client component."""

from unittest.mock import MagicMock

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


def test_backslash_url_rejected_without_allowlist():
    # Backslash authority trick must be rejected even with the default empty
    # allowlist (requests normalizes "\" and would reach an unintended host).
    component = HTTPClientComponent(HTTPClientConfiguration())
    assert not component._is_domain_allowed("http://127.0.0.1:6666\\@example.com/")
    with pytest.raises(HTTPError):
        component._make_request("GET", "http://127.0.0.1:6666\\@example.com/")


def _redirect_response(location: str) -> MagicMock:
    response = MagicMock()
    response.is_redirect = True
    response.status_code = 302
    response.headers = {"Location": location}
    return response


def _final_response() -> MagicMock:
    response = MagicMock()
    response.is_redirect = False
    response.status_code = 200
    response.headers = {}
    response.content = b"ok"
    return response


def test_cross_host_redirect_strips_authorization_header():
    # requests strips Authorization when following a cross-host redirect
    # (Session.rebuild_auth); the manual redirect loop must do the same.
    component = HTTPClientComponent(HTTPClientConfiguration())
    component.session.request = MagicMock(
        side_effect=[_redirect_response("https://example.org/"), _final_response()]
    )

    component._make_request(
        "GET", "https://example.com/", headers={"Authorization": "Bearer token"}
    )

    calls = component.session.request.call_args_list
    assert calls[0].kwargs["headers"] == {"Authorization": "Bearer token"}
    assert calls[1].kwargs["headers"] == {}


def test_same_host_redirect_keeps_authorization_header():
    component = HTTPClientComponent(HTTPClientConfiguration())
    component.session.request = MagicMock(
        side_effect=[_redirect_response("/moved"), _final_response()]
    )

    component._make_request(
        "GET", "https://example.com/", headers={"Authorization": "Bearer token"}
    )

    calls = component.session.request.call_args_list
    assert calls[1].kwargs["headers"] == {"Authorization": "Bearer token"}


def test_make_request_blocks_redirect_to_internal():
    # A public URL that redirects to an internal host must be rejected: redirects
    # are followed manually and each hop is re-validated.
    component = HTTPClientComponent(HTTPClientConfiguration())
    redirect = MagicMock()
    redirect.is_redirect = True
    redirect.status_code = 302
    redirect.headers = {"Location": "http://127.0.0.1/"}
    component.session.request = MagicMock(return_value=redirect)
    with pytest.raises(HTTPError):
        component._make_request("GET", "https://example.com/")
