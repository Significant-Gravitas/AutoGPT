"""Unit tests for the shared MCP helpers."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.mcp.helpers import (
    auto_lookup_mcp_credential,
    is_mcp_credential_for_server,
    mcp_auth_token,
    normalize_mcp_url,
    parse_mcp_content,
    server_host,
)
from backend.data.model import APIKeyCredentials, OAuth2Credentials

# ---------------------------------------------------------------------------
# normalize_mcp_url
# ---------------------------------------------------------------------------


def test_normalize_trailing_slash():
    assert normalize_mcp_url("https://mcp.example.com/") == "https://mcp.example.com"


def test_normalize_whitespace():
    assert normalize_mcp_url("  https://mcp.example.com  ") == "https://mcp.example.com"


def test_normalize_both():
    assert (
        normalize_mcp_url("  https://mcp.example.com/  ") == "https://mcp.example.com"
    )


def test_normalize_noop():
    assert normalize_mcp_url("https://mcp.example.com") == "https://mcp.example.com"


def test_normalize_path_with_trailing_slash():
    assert (
        normalize_mcp_url("https://mcp.example.com/path/")
        == "https://mcp.example.com/path"
    )


# ---------------------------------------------------------------------------
# server_host
# ---------------------------------------------------------------------------


def test_server_host_standard_url():
    assert server_host("https://mcp.example.com/mcp") == "mcp.example.com"


def test_server_host_strips_credentials():
    """hostname must not expose user:pass."""
    assert server_host("https://user:secret@mcp.example.com/mcp") == "mcp.example.com"


def test_server_host_with_port():
    """Port should not appear in hostname (hostname strips it)."""
    assert server_host("https://mcp.example.com:8080/mcp") == "mcp.example.com"


def test_server_host_fallback():
    """Falls back to the raw string for un-parseable URLs."""
    assert server_host("not-a-url") == "not-a-url"


# ---------------------------------------------------------------------------
# parse_mcp_content
# ---------------------------------------------------------------------------


def test_parse_text_plain():
    assert parse_mcp_content([{"type": "text", "text": "hello world"}]) == "hello world"


def test_parse_text_json():
    content = [{"type": "text", "text": '{"status": "ok", "count": 42}'}]
    assert parse_mcp_content(content) == {"status": "ok", "count": 42}


def test_parse_image():
    content = [{"type": "image", "data": "abc123==", "mimeType": "image/png"}]
    assert parse_mcp_content(content) == {
        "type": "image",
        "data": "abc123==",
        "mimeType": "image/png",
    }


def test_parse_resource():
    content = [
        {"type": "resource", "resource": {"uri": "file:///tmp/out.txt", "text": "hi"}}
    ]
    assert parse_mcp_content(content) == {"uri": "file:///tmp/out.txt", "text": "hi"}


def test_parse_multi_item():
    content = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert parse_mcp_content(content) == ["first", "second"]


def test_parse_empty():
    assert parse_mcp_content([]) is None


# ---------------------------------------------------------------------------
# mcp_auth_token / is_mcp_credential_for_server
# ---------------------------------------------------------------------------


def _oauth_cred(url: str) -> OAuth2Credentials:
    return OAuth2Credentials(
        provider="mcp",
        access_token=SecretStr("oauth-token"),
        scopes=[],
        title="MCP",
        metadata={"mcp_server_url": url},
    )


def _api_key_cred(url: str) -> APIKeyCredentials:
    return APIKeyCredentials(
        provider="mcp",
        api_key=SecretStr("static-token"),
        title="MCP",
        metadata={"mcp_server_url": url},
    )


def test_mcp_auth_token_oauth():
    assert mcp_auth_token(_oauth_cred("https://mcp.example.com/mcp")) == "oauth-token"


def test_mcp_auth_token_api_key():
    assert (
        mcp_auth_token(_api_key_cred("https://mcp.example.com/mcp")) == "static-token"
    )


def test_is_mcp_credential_for_server_matches_both_types():
    url = "https://mcp.example.com/mcp"
    assert is_mcp_credential_for_server(_oauth_cred(url), url)
    assert is_mcp_credential_for_server(_api_key_cred(url), url)


def test_is_mcp_credential_for_server_normalizes_trailing_slash():
    # Stored with a trailing slash, looked up without — must still match.
    cred = _api_key_cred("https://mcp.example.com/mcp/")
    assert is_mcp_credential_for_server(cred, "https://mcp.example.com/mcp")


def test_is_mcp_credential_for_server_rejects_other_server():
    cred = _api_key_cred("https://other.example.com/mcp")
    assert not is_mcp_credential_for_server(cred, "https://mcp.example.com/mcp")


def test_is_mcp_credential_for_server_rejects_non_oauth_non_apikey():
    """The TypeGuard must reject credential types that aren't OAuth2/APIKey
    even when the provider is MCP (e.g. a host-scoped credential)."""
    from backend.data.model import HostScopedCredentials

    cred = HostScopedCredentials(
        provider="mcp",
        host="mcp.example.com",
        headers={"Authorization": SecretStr("secret")},
    )
    assert not is_mcp_credential_for_server(cred, "https://mcp.example.com/mcp")


# ---------------------------------------------------------------------------
# auto_lookup_mcp_credential — mixed-type selection + OAuth-only refresh guard
# ---------------------------------------------------------------------------

_MCP_URL = "https://mcp.example.com/mcp"


def _patched_manager(stored: list, refreshed=None):
    """Patch IntegrationCredentialsManager used by auto_lookup with a fake store."""
    mgr = AsyncMock()
    mgr.store.get_creds_by_provider = AsyncMock(return_value=stored)
    mgr.refresh_if_needed = AsyncMock(side_effect=lambda uid, cred: refreshed or cred)
    return (
        patch(
            "backend.blocks.mcp.helpers.IntegrationCredentialsManager",
            return_value=mgr,
        ),
        mgr,
    )


@pytest.mark.asyncio
async def test_auto_lookup_prefers_non_expiring_api_key_over_expired_oauth():
    """A valid non-expiring bearer token must win over a stale OAuth row that
    once had an expiry — and the api-key path must NOT attempt a refresh."""
    expired_oauth = OAuth2Credentials(
        provider="mcp",
        access_token=SecretStr("old"),
        scopes=[],
        access_token_expires_at=1000,  # long past
        title="MCP",
        metadata={"mcp_server_url": _MCP_URL},
    )
    api_key = _api_key_cred(_MCP_URL)

    ctx, mgr = _patched_manager([expired_oauth, api_key])
    with ctx:
        result = await auto_lookup_mcp_credential("user", _MCP_URL)

    assert result is api_key
    mgr.refresh_if_needed.assert_not_called()


@pytest.mark.asyncio
async def test_auto_lookup_refreshes_selected_oauth_credential():
    """When the best match is an OAuth2 credential, it is refreshed."""
    oauth = OAuth2Credentials(
        provider="mcp",
        access_token=SecretStr("live"),
        scopes=[],
        access_token_expires_at=9999999999,
        title="MCP",
        metadata={"mcp_server_url": _MCP_URL},
    )
    refreshed = OAuth2Credentials(
        provider="mcp",
        access_token=SecretStr("rotated"),
        scopes=[],
        title="MCP",
        metadata={"mcp_server_url": _MCP_URL},
    )
    ctx, mgr = _patched_manager([oauth], refreshed=refreshed)
    with ctx:
        result = await auto_lookup_mcp_credential("user", _MCP_URL)

    assert result is refreshed
    mgr.refresh_if_needed.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_lookup_returns_none_when_no_match():
    other = _api_key_cred("https://other.example.com/mcp")
    ctx, mgr = _patched_manager([other])
    with ctx:
        result = await auto_lookup_mcp_credential("user", _MCP_URL)

    assert result is None
    mgr.refresh_if_needed.assert_not_called()


@pytest.mark.asyncio
async def test_auto_lookup_tiebreaks_equal_rank_by_recency():
    """Among equally-ranked (non-expiring) credentials for the same server,
    the most recently created — last in iteration order — wins."""
    older = _api_key_cred(_MCP_URL)
    newer = _api_key_cred(_MCP_URL)
    ctx, mgr = _patched_manager([older, newer])
    with ctx:
        result = await auto_lookup_mcp_credential("user", _MCP_URL)

    assert result is newer
