"""The service behind a provider, a catalog server or a stored credential.

A vendor can be reachable two ways, as a block provider ("linear") and as an
MCP server ("mcp_linear"), and a stored MCP credential names neither: it is
``provider="mcp"`` plus a server URL.  Every listing that groups, de-dupes or
draws a logo needs the same answer to "which service is this", so it is
computed here and nowhere else.
"""

import logging
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from backend.data.model import Credentials, OAuth2Credentials
from backend.integrations.credentials_store import canonical_provider
from backend.integrations.mcp_catalog import (
    MCPCatalogEntry,
    get_mcp_catalog_entry_for_url,
)
from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)

UNKNOWN_MCP_SERVICE = "mcp:unknown"


class ServiceIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    service: str
    name: str | None = None
    icon: str | None = None


def catalog_slug(entry: MCPCatalogEntry) -> str:
    return entry.name.removeprefix("mcp_")


def service_for_provider(provider: str) -> ServiceIdentity:
    """A block provider is its own service; its logo is filed under its slug."""
    slug = canonical_provider(provider)
    return ServiceIdentity(service=slug, icon=slug)


def service_for_catalog_entry(entry: MCPCatalogEntry) -> ServiceIdentity:
    server = entry.mcp_server
    return ServiceIdentity(
        service=server.provider or catalog_slug(entry),
        name=entry.display_name,
        icon=server.icon_id,
    )


def service_for_credential(credential: Credentials) -> ServiceIdentity:
    provider = canonical_provider(credential.provider)
    if provider != ProviderName.MCP.value or not isinstance(
        credential, OAuth2Credentials
    ):
        return service_for_provider(provider)
    url = (credential.metadata or {}).get("mcp_server_url")
    if not isinstance(url, str) or not url.strip():
        logger.warning("MCP credential %s has no server URL", credential.id)
        return ServiceIdentity(service=UNKNOWN_MCP_SERVICE)
    entry = get_mcp_catalog_entry_for_url(url)
    if entry is not None:
        return service_for_catalog_entry(entry)
    host = _hostname(url)
    if host is None:
        logger.warning("MCP credential %s has an unparseable server URL", credential.id)
        return ServiceIdentity(service=UNKNOWN_MCP_SERVICE)
    return ServiceIdentity(service=f"mcp:{host}", name=host)


def _hostname(url: str) -> str | None:
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    return parsed.hostname or None
