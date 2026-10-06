"""One grouping key for a provider, a catalog server and a stored credential."""

from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.mcp_catalog import get_mcp_catalog
from backend.integrations.service_identity import (
    service_for_catalog_entry,
    service_for_credential,
    service_for_provider,
)


def _entry(name: str):
    return next(e for e in get_mcp_catalog() if e.name == name)


def _mcp_credential(url: str | None, provider: str = "mcp") -> OAuth2Credentials:
    return OAuth2Credentials(
        id="cred-mcp",
        provider=provider,
        title=f"MCP: {url}",
        access_token=SecretStr("t"),
        scopes=[],
        metadata={"mcp_server_url": url} if url is not None else {},
    )


def test_block_provider_is_its_own_service():
    identity = service_for_provider("github")
    assert identity.service == "github"
    assert identity.name is None
    assert identity.icon == "github"


def test_catalog_entry_with_a_block_provider_uses_that_provider():
    identity = service_for_catalog_entry(_entry("mcp_linear"))
    assert identity.service == "linear"
    assert identity.name == "Linear"
    assert identity.icon == _entry("mcp_linear").mcp_server.icon_id


def test_catalog_entry_without_a_block_provider_uses_its_slug():
    identity = service_for_catalog_entry(_entry("mcp_sentry"))
    assert identity.service == "sentry"
    assert identity.name == "Sentry"


def test_api_key_credential_resolves_to_its_provider():
    credential = APIKeyCredentials(
        id="c", provider="notion", title="Team Notion", api_key=SecretStr("k")
    )
    assert service_for_credential(credential).service == "notion"


def test_mcp_credential_for_a_catalog_url_resolves_to_the_catalog_service():
    url = _entry("mcp_linear").mcp_server.server_url
    assert url
    identity = service_for_credential(_mcp_credential(url))
    assert identity.service == "linear"
    assert identity.name == "Linear"


def test_mcp_credential_matches_the_catalog_url_without_a_trailing_slash():
    url = _entry("mcp_notion").mcp_server.server_url
    assert url
    identity = service_for_credential(_mcp_credential(url.rstrip("/") + "/"))
    assert identity.service == "notion"


def test_mcp_credential_for_a_custom_url_is_keyed_by_host():
    identity = service_for_credential(
        _mcp_credential("https://mcp.internal.example/mcp")
    )
    assert identity.service == "mcp:mcp.internal.example"
    assert identity.name == "mcp.internal.example"
    assert identity.icon is None


def test_mcp_credential_with_a_legacy_provider_spelling_still_resolves():
    url = _entry("mcp_linear").mcp_server.server_url
    assert url
    identity = service_for_credential(_mcp_credential(url, provider="ProviderName.MCP"))
    assert identity.service == "linear"


def test_mcp_credential_with_a_malformed_url_never_raises():
    identity = service_for_credential(_mcp_credential("not a url"))
    assert identity.service == "mcp:unknown"
    identity = service_for_credential(_mcp_credential(None))
    assert identity.service == "mcp:unknown"
