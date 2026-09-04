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


def _oauth_credential() -> OAuth2Credentials:
    """A real OAuth MCP credential: expired access token, good refresh token."""
    return OAuth2Credentials(
        provider="mcp",
        title="MCP: mcp.example.com",
        access_token=SecretStr("expired-access-token"),
        refresh_token=SecretStr("still-good-refresh-token"),
        access_token_expires_at=1,
        scopes=[],
        metadata={
            "mcp_server_url": _SERVER_URL,
            "mcp_token_url": "https://auth.example.com/token",
        },
    )


def _real_manager_over(creds: list[OAuth2Credentials]):
    """Patch only the *store*, so the real ``IntegrationCredentialsManager``
    and its real ``refresh_if_needed`` run.

    Mocking ``refresh_if_needed`` here would make these tests pass against the
    unfixed code — the bug lived inside it.
    """
    store = MagicMock()
    store.get_creds_by_provider = AsyncMock(return_value=creds)
    return patch(
        "backend.integrations.creds_manager.IntegrationCredentialsStore",
        return_value=store,
    )


async def test_auto_lookup_resolves_static_token_credential():
    """Regression for SECRT-2592.

    Resolving a manually-entered API token used to raise inside the real
    refresh path (``create_mcp_oauth_handler`` has no ``mcp_token_url`` to work
    with), and the blanket ``except`` reported that as ``None`` — so the agent
    said "not connected" while the UI still showed a green Connected pill.
    """
    cred = _static_token_credential()
    with _real_manager_over([cred]):
        assert await auto_lookup_mcp_credential("user-1", _SERVER_URL) is cred


async def test_auto_lookup_returns_none_when_a_real_refresh_fails():
    """A transient outage at the provider's token endpoint must not hand back
    the stale access token.

    The caller reads a 401 on a credential it *has* as proof the token is dead
    and deletes the row — taking a still-valid refresh token with it.
    """
    cred = _oauth_credential()
    with (
        _real_manager_over([cred]),
        patch(
            "backend.integrations.creds_manager.IntegrationCredentialsManager"
            "._get_oauth_handler",
            new_callable=AsyncMock,
            side_effect=RuntimeError("token endpoint 503"),
        ),
    ):
        assert await auto_lookup_mcp_credential("user-1", _SERVER_URL) is None


async def test_auto_lookup_returns_none_for_unmatched_server():
    with _real_manager_over([_static_token_credential()]):
        assert (
            await auto_lookup_mcp_credential("user-1", "https://other.example.com/mcp")
            is None
        )


async def test_auto_lookup_returns_none_when_store_fails():
    store = MagicMock()
    store.get_creds_by_provider = AsyncMock(side_effect=RuntimeError("db down"))
    with patch(
        "backend.integrations.creds_manager.IntegrationCredentialsStore",
        return_value=store,
    ):
        assert await auto_lookup_mcp_credential("user-1", _SERVER_URL) is None
