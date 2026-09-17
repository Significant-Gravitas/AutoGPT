"""Connection state feeds ranking, so a credential that reads as absent
quietly demotes every capability the user actually connected."""

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, OAuth2Credentials

from .resolve import connection_state_from


def _oauth(provider: str, url: str | None = None) -> OAuth2Credentials:
    return OAuth2Credentials(
        id="cred-1",
        provider=provider,
        title="Linear",
        username=None,
        access_token=SecretStr("at"),
        refresh_token=None,
        access_token_expires_at=None,
        refresh_token_expires_at=None,
        scopes=[],
        metadata={"mcp_server_url": url} if url else {},
    )


def test_a_legacy_provider_string_still_counts_as_connected():
    """Credentials written under Python 3.13 carry ``"ProviderName.MCP"``.

    Ranking asks for the canonical ``"mcp"``, so without normalising here the
    user's own connected server ranks as one they have never signed into.
    """
    state = connection_state_from(
        [_oauth("ProviderName.MCP", "https://mcp.linear.app/mcp")]
    )
    assert "mcp" in state.providers
    assert "https://mcp.linear.app/mcp" in state.server_urls


def test_a_canonical_provider_is_left_alone():
    state = connection_state_from([_oauth("mcp", "https://mcp.linear.app/mcp")])
    assert state.providers == frozenset({"mcp"})
    assert state.server_urls == frozenset({"https://mcp.linear.app/mcp"})


def test_an_unknown_legacy_member_survives_as_itself():
    """A provider we have since retired must not vanish from the set."""
    state = connection_state_from([_oauth("ProviderName.NOT_A_PROVIDER")])
    assert state.providers == frozenset({"ProviderName.NOT_A_PROVIDER"})


def test_a_non_mcp_credential_contributes_no_server_url():
    state = connection_state_from(
        [
            APIKeyCredentials(
                id="cred-2",
                provider="openai",
                title="OpenAI",
                api_key=SecretStr("sk"),
                expires_at=None,
            )
        ]
    )
    assert state.providers == frozenset({"openai"})
    assert state.server_urls == frozenset()


def test_no_credentials_is_an_empty_state():
    state = connection_state_from([])
    assert not state.providers and not state.server_urls and not state.hosts
