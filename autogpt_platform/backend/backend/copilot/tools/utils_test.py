"""Tests for chat tools utility functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.model import CredentialsFieldInfo


def _make_regular_field() -> CredentialsFieldInfo:
    return CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["github"],
            "credentials_types": ["api_key"],
            "is_auto_credential": False,
        },
        by_alias=True,
    )


def _make_mcp_field(server_url: str) -> CredentialsFieldInfo:
    return CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["mcp"],
            "credentials_types": ["oauth2", "api_key"],
            "discriminator": "server_url",
            "discriminator_values": [server_url],
            "is_auto_credential": False,
        },
        by_alias=True,
    )


def test_credential_is_for_mcp_server_matches_both_credential_types():
    """A static API-key MCP token must match the server URL just like an
    OAuth2 token — otherwise copilot reports the credential as missing."""
    from pydantic import SecretStr

    from backend.copilot.tools.utils import _credential_is_for_mcp_server
    from backend.data.model import APIKeyCredentials, OAuth2Credentials

    server_url = "https://mcp.example.com/mcp"
    field = _make_mcp_field(server_url)

    oauth_cred = OAuth2Credentials(
        provider="mcp",
        access_token=SecretStr("t"),
        scopes=[],
        title="MCP",
        metadata={"mcp_server_url": server_url},
    )
    api_key_cred = APIKeyCredentials(
        provider="mcp",
        api_key=SecretStr("t"),
        title="MCP",
        metadata={"mcp_server_url": server_url},
    )
    other_server = APIKeyCredentials(
        provider="mcp",
        api_key=SecretStr("t"),
        title="MCP",
        metadata={"mcp_server_url": "https://other.example.com/mcp"},
    )

    assert _credential_is_for_mcp_server(oauth_cred, field)
    assert _credential_is_for_mcp_server(api_key_cred, field)
    assert not _credential_is_for_mcp_server(other_server, field)


def test_build_missing_credentials_excludes_auto_creds():
    """
    build_missing_credentials_from_graph() should use regular_credentials_inputs
    and thus exclude auto_credentials from the "missing" set.
    """
    from backend.copilot.tools.utils import build_missing_credentials_from_graph

    regular_field = _make_regular_field()

    mock_graph = MagicMock()
    # regular_credentials_inputs should only return the non-auto field
    mock_graph.regular_credentials_inputs = {
        "github_api_key": (regular_field, {("node-1", "credentials")}, True),
    }

    result = build_missing_credentials_from_graph(mock_graph, matched_credentials=None)

    # Should include the regular credential
    assert "github_api_key" in result
    # Should NOT include the auto_credential (not in regular_credentials_inputs)
    assert "google_oauth2" not in result


@pytest.mark.asyncio
async def test_match_user_credentials_excludes_auto_creds():
    """
    match_user_credentials_to_graph() should use regular_credentials_inputs
    and thus exclude auto_credentials from matching.
    """
    from backend.copilot.tools.utils import match_user_credentials_to_graph

    regular_field = _make_regular_field()

    mock_graph = MagicMock()
    mock_graph.id = "test-graph"
    # regular_credentials_inputs returns only non-auto fields
    mock_graph.regular_credentials_inputs = {
        "github_api_key": (regular_field, {("node-1", "credentials")}, True),
    }

    # Mock the credentials manager to return no credentials
    with patch(
        "backend.copilot.tools.utils.IntegrationCredentialsManager"
    ) as MockCredsMgr:
        mock_store = AsyncMock()
        mock_store.get_all_creds.return_value = []
        MockCredsMgr.return_value.store = mock_store

        matched, missing = await match_user_credentials_to_graph(
            user_id="test-user", graph=mock_graph
        )

    # No credentials available, so github should be missing
    assert len(matched) == 0
    assert len(missing) == 1
    assert "github_api_key" in missing[0]
