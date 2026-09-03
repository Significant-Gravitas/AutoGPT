"""Unit tests for the shared MCP helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import SecretStr

from backend.blocks.mcp.helpers import (
    auto_lookup_mcp_credential,
    normalize_mcp_url,
    parse_mcp_content,
    server_host,
)
from backend.data.model import OAuth2Credentials

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
# auto_lookup_mcp_credential
# ---------------------------------------------------------------------------

_SERVER_URL = "https://mcp.example.com/mcp"


def _static_token_credential() -> OAuth2Credentials:
    """An MCP credential exactly as ``POST /mcp/token`` stores it: a static
    bearer token with no expiry, no refresh token, and no OAuth metadata."""
    return OAuth2Credentials(
        provider="mcp",
        title="MCP: mcp.example.com",
        access_token=SecretStr("dft_static_token"),
        scopes=[],
        metadata={"mcp_server_url": _SERVER_URL},
    )


def _patched_manager(creds: list[OAuth2Credentials], refresh: AsyncMock | None = None):
    manager = MagicMock()
    manager.store.get_creds_by_provider = AsyncMock(return_value=creds)
    manager.refresh_if_needed = refresh or AsyncMock(side_effect=lambda _user, c: c)
    return patch(
        "backend.blocks.mcp.helpers.IntegrationCredentialsManager",
        return_value=manager,
    )


async def test_auto_lookup_resolves_static_token_credential():
    """Regression for SECRT-2592.

    Resolving a manually-entered API token used to raise inside the refresh
    path, and the blanket ``except`` reported that as ``None`` — so the agent
    said "not connected" while the UI still showed a green Connected pill.
    """
    cred = _static_token_credential()
    with _patched_manager([cred]):
        assert await auto_lookup_mcp_credential("user-1", _SERVER_URL) is cred


async def test_auto_lookup_keeps_stored_token_when_refresh_fails():
    """A failed refresh must not be indistinguishable from "no credential".

    Returning the stored token lets the MCP server be the one to reject it,
    which is what triggers ``invalidate_mcp_credential`` and clears the row.
    """
    cred = _static_token_credential()
    with _patched_manager(
        [cred], refresh=AsyncMock(side_effect=ValueError("cannot refresh tokens"))
    ):
        assert await auto_lookup_mcp_credential("user-1", _SERVER_URL) is cred


async def test_auto_lookup_returns_none_for_unmatched_server():
    with _patched_manager([_static_token_credential()]):
        assert (
            await auto_lookup_mcp_credential("user-1", "https://other.example.com/mcp")
            is None
        )


async def test_auto_lookup_returns_none_when_store_fails():
    manager = MagicMock()
    manager.store.get_creds_by_provider = AsyncMock(side_effect=RuntimeError("db down"))
    with patch(
        "backend.blocks.mcp.helpers.IntegrationCredentialsManager",
        return_value=manager,
    ):
        assert await auto_lookup_mcp_credential("user-1", _SERVER_URL) is None
