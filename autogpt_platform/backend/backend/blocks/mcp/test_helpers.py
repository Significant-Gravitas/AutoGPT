"""Unit tests for the shared MCP helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.mcp.helpers import (
    auto_lookup_mcp_credential,
    is_manual_mcp_credential,
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
# is_manual_mcp_credential / auto_lookup_mcp_credential refresh gate
# ---------------------------------------------------------------------------


def _mcp_credential(**overrides) -> OAuth2Credentials:
    fields: dict = {
        "provider": "mcp",
        "title": "MCP: mcp.example.com",
        "access_token": SecretStr("token"),
        "scopes": [],
        "metadata": {"mcp_server_url": "https://mcp.example.com/mcp"},
    }
    fields.update(overrides)
    return OAuth2Credentials(**fields)


def test_manual_credential_is_distinguished_from_oauth():
    manual = _mcp_credential()
    oauth = _mcp_credential(
        refresh_token=SecretStr("refresh"),
        metadata={
            "mcp_server_url": "https://mcp.example.com/mcp",
            "mcp_token_url": "https://mcp.example.com/token",
            "mcp_client_id": "client-abc",
        },
    )
    assert is_manual_mcp_credential(manual) is True
    assert is_manual_mcp_credential(oauth) is False


def test_oauth_metadata_alone_marks_a_credential_as_non_manual():
    """A row with client registration but no refresh token is still OAuth's."""
    assert (
        is_manual_mcp_credential(
            _mcp_credential(
                metadata={
                    "mcp_server_url": "https://mcp.example.com/mcp",
                    "mcp_token_url": "https://mcp.example.com/token",
                }
            )
        )
        is False
    )


@pytest.mark.asyncio
async def test_expiring_credential_is_refreshed():
    """Positive guard for the refresh gate.

    Only the negative case ("a manual credential is never refreshed") was
    covered, so deleting the ``refresh_if_needed`` call entirely left the
    whole backend suite green.
    """
    expiring = _mcp_credential(
        access_token_expires_at=1,
        refresh_token=SecretStr("refresh"),
        metadata={
            "mcp_server_url": "https://mcp.example.com/mcp",
            "mcp_token_url": "https://mcp.example.com/token",
        },
    )
    refreshed = _mcp_credential(access_token=SecretStr("refreshed-token"))

    with patch(
        "backend.blocks.mcp.helpers.IntegrationCredentialsManager"
    ) as manager_cls:
        manager = manager_cls.return_value
        manager.store.get_creds_by_provider = AsyncMock(return_value=[expiring])
        manager.refresh_if_needed = AsyncMock(return_value=refreshed)

        result = await auto_lookup_mcp_credential(
            "test-user-id", "https://mcp.example.com/mcp"
        )

    manager.refresh_if_needed.assert_awaited_once()
    assert result is not None
    assert result.access_token.get_secret_value() == "refreshed-token"


@pytest.mark.asyncio
async def test_a_manual_credential_outranks_a_surviving_oauth_row():
    """Ranking by expiry alone pinned execution to a stale OAuth grant.

    A manual credential never has an ``access_token_expires_at``, so
    ``cred.access_token_expires_at or 0`` made it lose to *any* OAuth row for
    the same server.  Storing a manual credential leaves the OAuth row in place
    (deleting it there would orphan the refresh token at the provider), and the
    supersede loop swallows delete failures, so this is reachable rather than
    theoretical: the user pastes a working credential, all three UIs probe it
    and report "Connected", and every execution still goes out on the old token.
    """
    oauth = _mcp_credential(
        access_token=SecretStr("oauth-token"),
        access_token_expires_at=2_000_000_000,
        refresh_token=SecretStr("refresh"),
        metadata={
            "mcp_server_url": "https://mcp.example.com/mcp",
            "mcp_token_url": "https://mcp.example.com/token",
            "mcp_client_id": "client-abc",
        },
    )
    manual = _mcp_credential(
        access_token=SecretStr("Basic encoded-value"),
        metadata={
            "mcp_server_url": "https://mcp.example.com/mcp",
            "mcp_auth_scheme": "basic",
        },
    )

    with patch(
        "backend.blocks.mcp.helpers.IntegrationCredentialsManager"
    ) as manager_cls:
        manager = manager_cls.return_value
        manager.store.get_creds_by_provider = AsyncMock(return_value=[oauth, manual])
        manager.refresh_if_needed = AsyncMock()

        result = await auto_lookup_mcp_credential(
            "test-user-id", "https://mcp.example.com/mcp"
        )

    assert result is not None
    assert result.id == manual.id
    # And a manual credential is never put through the OAuth refresh path.
    manager.refresh_if_needed.assert_not_awaited()


# ---------------------------------------------------------------------------
# auto_lookup_mcp_credential
# ---------------------------------------------------------------------------


_SERVER_URL = "https://mcp.example.com/mcp"


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
