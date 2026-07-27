"""Unit tests for the shared MCP helpers."""

from pydantic import SecretStr

from backend.blocks.mcp.helpers import (
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
