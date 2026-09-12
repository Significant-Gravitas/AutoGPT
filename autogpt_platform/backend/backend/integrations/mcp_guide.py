import re
from itertools import groupby

from backend.integrations.mcp_catalog import (
    MCPCatalogEntry,
    get_connectable_mcp_catalog,
)

MCP_CATALOG_MARKER = "<!-- official-mcp-catalog -->"


def render_mcp_guide(content: str) -> str:
    if MCP_CATALOG_MARKER not in content:
        return content
    existing_urls = set(re.findall(r"https://[^\s\x60|]+", content))
    entries = [
        entry
        for entry in get_connectable_mcp_catalog()
        if entry.mcp_server.connection_mode == "hosted"
        and entry.mcp_server.server_url not in existing_urls
    ]
    groups = groupby(sorted(entries, key=_server_url), key=_server_url)
    rows = [_render_server_group(list(group)) for _, group in groups]
    table = "\n".join(
        [
            "Additional official connections from the integration catalog:",
            "",
            "Respect each entry's purpose. Public documentation servers do not "
            "provide private account access. Local and setup-only entries are omitted.",
            "",
            "| Service | URL | Authentication | Purpose | Setup requirements |",
            "|---|---|---|---|---|",
            *rows,
        ]
    )
    return content.replace(MCP_CATALOG_MARKER, table)


def _server_url(entry: MCPCatalogEntry) -> str:
    return entry.mcp_server.server_url or ""


def _render_server_group(entries: list[MCPCatalogEntry]) -> str:
    names = " / ".join(dict.fromkeys(entry.display_name for entry in entries))
    purposes = " ".join(dict.fromkeys(entry.description for entry in entries))
    auth_modes = " / ".join(
        dict.fromkeys(entry.mcp_server.auth_mode for entry in entries)
    )
    requirements = " ".join(
        dict.fromkeys(entry.mcp_server.setup_instructions for entry in entries)
    )
    cells = [names, f"`{_server_url(entries[0])}`", auth_modes, purposes, requirements]
    return "| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |"
