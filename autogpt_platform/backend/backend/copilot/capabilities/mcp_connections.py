"""User-connected MCP servers that are absent from the shared catalog."""

import re
from urllib.parse import urlsplit

from backend.integrations.mcp_catalog import get_mcp_catalog_entry_for_url

from .index import CapabilityIndex
from .models import CapabilityEntry, Connection, Implementation, clip_purpose
from .ranking import ConnectionState, normalize_server_url

_NON_SERVICE_LABELS = frozenset(
    {"mcp", "api", "www", "com", "org", "net", "io", "ai", "dev", "co", "uk", "app"}
)


def connected_mcp_entries(
    index: CapabilityIndex, connections: ConnectionState
) -> list[CapabilityEntry]:
    """Layer custom endpoints and bind catalog presets to stored URL options."""
    catalog_urls = {
        normalize_server_url(entry.connection.key)
        for entry in index.entries
        if entry.kind == "mcp_server" and entry.connection.key
    }
    urls = stored_mcp_urls(connections)
    by_url = {
        normalize_server_url(url): url.strip() for url in sorted(urls, reverse=True)
    }
    granted = {normalize_server_url(url) for url in connections.server_urls}
    entries = [
        catalog_mcp_entry(index, url) or entry
        for url in sorted(
            by_url.values(),
            key=lambda url: (normalize_server_url(url) not in granted, url),
        )
        if normalize_server_url(url) not in catalog_urls
        if (entry := custom_mcp_entry(url)) is not None
    ]
    return _unique_preset_ids(entries)


def catalog_mcp_entry(
    index: CapabilityIndex, server_url: str
) -> CapabilityEntry | None:
    """Bind a catalog URL option to its preset's metadata and stable ID."""
    preset = get_mcp_catalog_entry_for_url(server_url)
    if preset is None:
        return None
    catalog = next(
        (
            entry
            for entry in index.entries
            if entry.kind == "mcp_server" and entry.schema_ref == f"mcp:{preset.name}"
        ),
        None,
    )
    if catalog is None:
        return None
    if catalog.connection.key and normalize_server_url(
        catalog.connection.key
    ) == normalize_server_url(server_url):
        return catalog
    return catalog.model_copy(
        update={
            "connection": Connection(
                required=True, key_type="server_url", key=server_url
            ),
            "implementations": [
                Implementation(kind="mcp_server", ref=server_url, name=catalog.name)
            ],
        }
    )


def custom_mcp_entry(server_url: str) -> CapabilityEntry | None:
    """Use an HTTPS endpoint's hostname as metadata and its raw URL as the ID."""
    try:
        parsed = urlsplit(server_url)
    except ValueError:
        return None
    host = parsed.hostname
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
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
        tags=[*service_names, "mcp", host],
        context="direct",
        implementations=[Implementation(kind="mcp_server", ref=server_url, name=host)],
        connection=Connection(required=True, key_type="server_url", key=server_url),
    )


def stored_mcp_urls(connections: ConnectionState) -> frozenset[str]:
    """Include stored endpoints an expert has not yet been granted."""
    return connections.server_urls | (
        connections.ungranted.server_urls
        if connections.ungranted is not None
        else frozenset()
    )


def _unique_preset_ids(entries: list[CapabilityEntry]) -> list[CapabilityEntry]:
    """Bind the preset ID to the first usable option; address other options by URL."""
    first_urls = {entry.id: entry.implementations[0].ref for entry in reversed(entries)}
    return [
        (
            entry
            if entry.implementations[0].ref == first_urls[entry.id]
            else entry.model_copy(update={"id": entry.implementations[0].ref})
        )
        for entry in entries
    ]
