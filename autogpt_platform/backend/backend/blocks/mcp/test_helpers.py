"""Unit tests for the shared MCP helpers."""

from unittest.mock import AsyncMock, patch

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
