"""The official MCP catalog (``integrations/mcp_catalog.json``) as entries.

One entry per catalogued server.  Its tools are not enumerated here: the
per-user source lists them once the user has connected (listing needs auth),
so a catalog entry is the "connect to Linear" capability itself.
"""

from collections import Counter
from functools import cache
from urllib.parse import urlsplit

from backend.copilot.capabilities.models import (
    CapabilityEntry,
    Connection,
    Implementation,
    clip_purpose,
    normalize_text,
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
    brand = _brand(slug)
    if brand:
        tags.append(brand)
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
        description=normalize_text(preset.description),
        tags=tags,
        context="direct",
        implementations=[
            # A ``custom`` preset ships no URL: the user supplies their own
            # tenant endpoint.  Leaving the ref empty keeps the preset name
            # out of URL position, where it used to be parsed as a hostname
            # ("Blocked server URL: Hostname 'mcp_amplitude' ...").
            Implementation(
                kind="mcp_server",
                ref=server.server_url or "",
                name=preset.display_name,
            )
        ],
        connection=Connection(
            required=True, key_type="server_url", key=server.server_url
        ),
        schema_ref=f"mcp:{preset.name}",
    )


# Suffixes that are part of a company's domain rather than its name, so
# "apollo_io" is the service "apollo" and a query for "apollo" must reach it.
_BRAND_SUFFIXES = frozenset({"io", "com", "dev", "ai", "co", "app", "so", "to", "sh"})


def _brand(slug: str) -> str | None:
    """The brand inside a domain-shaped slug, e.g. ``apollo_io`` -> ``apollo``."""
    head, _, tail = slug.rpartition("_")
    if head and tail in _BRAND_SUFFIXES and "_" not in head:
        return head
    return None


def setup_hint(schema_ref: str) -> str:
    """How to reach a catalog server that ships no URL of its own."""
    preset = _presets_by_schema_ref().get(schema_ref)
    if preset is None:
        return (
            "This MCP server needs a server URL. The user adds it under "
            "Settings -> Integrations, then call run_capability with that "
            "https:// URL as the id."
        )
    options = ", ".join(
        f"{option.label} ({option.url})"
        for option in preset.mcp_server.server_url_options
    )
    lines = [
        f"{preset.display_name} has no shared endpoint: it runs at a URL "
        "specific to the user's account, so the platform cannot call it until "
        "they add one under Settings -> Integrations.",
        preset.mcp_server.setup_instructions,
        f"Known endpoints: {options}." if options else "",
        "Once the user has added their URL, call run_capability with that "
        "https:// URL as the id.",
    ]
    return " ".join(line for line in lines if line)


@cache
def _presets_by_schema_ref() -> dict[str, MCPCatalogEntry]:
    return {f"mcp:{preset.name}": preset for preset in get_mcp_catalog()}
