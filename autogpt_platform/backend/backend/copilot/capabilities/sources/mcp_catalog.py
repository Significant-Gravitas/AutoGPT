"""The official MCP catalog (``integrations/mcp_catalog.json``) as entries.

One entry per catalogued server.  Its tools are not enumerated here: the
per-user source lists them once the user has connected (listing needs auth),
so a catalog entry is the "connect to Linear" capability itself.
"""

from collections import Counter
from urllib.parse import urlsplit

from backend.copilot.capabilities.models import (
    CapabilityEntry,
    Connection,
    Implementation,
    clip_purpose,
)
from backend.copilot.capabilities.text import tokenize
from backend.integrations.mcp_catalog import MCPCatalogEntry, get_mcp_catalog


def mcp_catalog_entries() -> list[CapabilityEntry]:
    presets = get_mcp_catalog()
    hosts = Counter(_preset_host(preset) for preset in presets)
    return [
        # A host shared by two presets (Atlassian ships a v2 server and a
        # Forge server on ``mcp.atlassian.com``) cannot key both entries, so
        # those fall back to the slug and the resolver matches them by URL.
        _catalog_entry(preset, keyed_by_host=hosts[_preset_host(preset)] == 1)
        for preset in presets
    ]


def _preset_host(preset: MCPCatalogEntry) -> str | None:
    url = preset.mcp_server.server_url
    return urlsplit(url).hostname if url else None


def _catalog_entry(preset: MCPCatalogEntry, keyed_by_host: bool) -> CapabilityEntry:
    server = preset.mcp_server
    host = _preset_host(preset)
    slug = preset.name.removeprefix("mcp_")
    tags = sorted(set(tokenize(preset.display_name)) | set(tokenize(slug)))
    # Service names come last so ``index._service_tags`` can read them back:
    # the slug, the host, and the display name when it is a single word.
    tags += ["mcp", slug]
    if host:
        tags.append(host)
    if " " not in preset.display_name.strip():
        tags.append(preset.display_name.strip().lower())
    return CapabilityEntry(
        id=f"mcp:{host}" if keyed_by_host and host else f"mcp:{slug}",
        kind="mcp_server",
        klass="service",
        name=preset.display_name,
        purpose=clip_purpose(preset.description),
        tags=tags,
        context="direct",
        implementations=[
            Implementation(
                kind="mcp_server",
                ref=server.server_url or preset.name,
                name=preset.display_name,
            )
        ],
        connection=Connection(
            required=True, key_type="server_url", key=server.server_url
        ),
        schema_ref=f"mcp:{preset.name}",
    )
