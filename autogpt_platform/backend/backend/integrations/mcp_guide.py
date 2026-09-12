import re
from itertools import groupby

from backend.integrations.mcp_catalog import (
    MCPCatalogEntry,
    get_connectable_mcp_catalog,
)
from backend.util.settings import Settings

MCP_CATALOG_MARKER = "<!-- official-mcp-catalog -->"
settings = Settings()


def render_mcp_guide(content: str) -> str:
    if MCP_CATALOG_MARKER not in content:
        return content
    existing_urls = set(re.findall(r"https://[^\s\x60|]+", content))
    entries = [
        entry
        for entry in get_connectable_mcp_catalog(settings.config.frontend_base_url)
        if entry.mcp_server.server_url not in existing_urls
    ]
    hosted = [entry for entry in entries if entry.mcp_server.server_url]
    groups = groupby(sorted(hosted, key=_server_url), key=_server_url)
    rows = [_render_server_group(list(group)) for _, group in groups]
    rows.extend(
        _render_server_group([entry])
        for entry in entries
        if not entry.mcp_server.server_url
    )
    table = "\n".join(
        [
            "Additional official connections from the integration catalog:",
            "",
            "Respect each entry's purpose. Public documentation servers do not "
            "provide private account access. For regional or tenant-specific servers, "
            "confirm the user's region or obtain their endpoint before connecting. "
            "Entries unsupported by this deployment are omitted.",
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
    auth_modes = " / ".join(dict.fromkeys(_authentication(entry) for entry in entries))
    requirements = " ".join(
        dict.fromkeys(entry.mcp_server.setup_instructions for entry in entries)
    )
    locations = " / ".join(dict.fromkeys(_server_location(entry) for entry in entries))
    cells = [names, locations, auth_modes, purposes, requirements]
    return "| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |"


def _server_location(entry: MCPCatalogEntry) -> str:
    server = entry.mcp_server
    if server.server_url_options:
        options = "; ".join(
            f"{option.label}: `{option.url}`" for option in server.server_url_options
        )
        return f"Choose the account region: {options}"
    if server.server_url:
        return f"`{server.server_url}`"
    return f"User-provided endpoint; [setup instructions]({server.documentation_url})"


def _authentication(entry: MCPCatalogEntry) -> str:
    server = entry.mcp_server
    methods = " / ".join(server.auth_methods)
    if server.oauth_server_url:
        methods += f"; OAuth uses `{server.oauth_server_url}`"
    if server.oauth_scopes is not None:
        methods += "; default OAuth scopes: " + (
            ", ".join(server.oauth_scopes) or "none"
        )
    if server.oauth_write_scopes:
        methods += "; optional grants require explicit selection in Integrations"
    return methods
