"""User-connected MCP servers that are absent from the shared catalog."""

import re
from urllib.parse import urlsplit

from .index import CapabilityIndex
from .models import CapabilityEntry, Connection, Implementation, clip_purpose
from .ranking import ConnectionState, normalize_server_url

_NON_SERVICE_LABELS = frozenset(
    {"mcp", "api", "www", "com", "org", "net", "io", "ai", "dev", "co", "uk", "app"}
)


def connected_mcp_entries(
    index: CapabilityIndex, connections: ConnectionState
) -> list[CapabilityEntry]:
    """Build session-scoped entries for stored MCP endpoints outside the catalog."""
    catalog_urls = {
        normalize_server_url(entry.connection.key)
        for entry in index.entries
        if entry.kind == "mcp_server" and entry.connection.key
    }
    urls = connections.server_urls
    if connections.ungranted is not None:
        urls |= connections.ungranted.server_urls
    by_url = {
        normalize_server_url(url): url.strip() for url in sorted(urls, reverse=True)
    }
    return [
        entry
        for key, url in by_url.items()
        if key not in catalog_urls
        if (entry := custom_mcp_entry(url)) is not None
    ]


def custom_mcp_entry(server_url: str) -> CapabilityEntry | None:
    """Use an HTTPS endpoint's hostname as metadata and its raw URL as the ID."""
    try:
        parsed = urlsplit(server_url)
    except ValueError:
        return None
    host = parsed.hostname
    if parsed.scheme != "https" or not host:
        return None
    description = f"MCP server connected at {host}."
    service_names = [
        label
        for label in re.split(r"[^a-z0-9]+", host)
        if label and label not in _NON_SERVICE_LABELS
    ]
    return CapabilityEntry(
        id=server_url,
        kind="mcp_server",
        name=host,
        purpose=clip_purpose(description),
        description=description,
        tags=["mcp", host, *service_names],
        context="direct",
        implementations=[Implementation(kind="mcp_server", ref=server_url, name=host)],
        connection=Connection(required=True, key_type="server_url", key=server_url),
    )
