"""Connection state feeds ranking, so a credential that reads as absent
quietly demotes every capability the user actually connected — and a
catalog URL that fails to resolve sends trusted writes to human review."""

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, OAuth2Credentials

from .index import CapabilityIndex
from .models import CapabilityEntry, Connection, Implementation
from .resolve import connection_state_from, resolve_entry


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


def _mcp_entry(slug: str, server_url: str) -> CapabilityEntry:
    return CapabilityEntry(
        id=f"mcp:{slug}",
        kind="mcp_server",
        name=slug,
        purpose=f"{slug} server",
        implementations=[Implementation(kind="mcp_server", ref=server_url)],
        connection=Connection(required=True, key_type="server_url", key=server_url),
    )


def _index(*entries: CapabilityEntry) -> CapabilityIndex:
    return CapabilityIndex(list(entries))


def test_a_shared_host_resolves_however_the_url_is_written():
    """Two presets on one host fall through to a full-URL match.

    Missing there does not read as "unknown id" — it reads as "not a catalog
    server", which sends a trusted write to human review and drops the
    catalog's own setup hints.
    """
    index = _index(
        _mcp_entry("atlassian-jira", "https://mcp.atlassian.com/v1/sse"),
        _mcp_entry("atlassian-conf", "https://mcp.atlassian.com/v1/confluence"),
    )
    for written in (
        "https://mcp.atlassian.com/v1/sse",
        "https://mcp.atlassian.com/v1/sse/",
        "  https://MCP.Atlassian.com/v1/SSE  ",
    ):
        entry = resolve_entry(index, written)
        assert entry is not None and entry.id == "mcp:atlassian-jira", written


def test_a_catalog_key_stored_with_a_slash_matches_a_url_without_one():
    """Catalog keys are stored both ways, so normalise both sides."""
    index = _index(
        _mcp_entry("miro-a", "https://mcp.miro.com/"),
        _mcp_entry("miro-b", "https://mcp.miro.com/other"),
    )
    entry = resolve_entry(index, "https://mcp.miro.com")
    assert entry is not None and entry.id == "mcp:miro-a"


def test_a_url_on_no_catalog_host_still_resolves_to_nothing():
    index = _index(_mcp_entry("miro-a", "https://mcp.miro.com/"))
    assert resolve_entry(index, "https://mcp.example.com/mcp") is None
