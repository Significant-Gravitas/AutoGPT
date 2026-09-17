from backend.integrations.mcp_catalog import MCPCatalogEntry, get_mcp_catalog

MCP_CATALOG_MARKER = "<!-- official-mcp-catalog -->"


def render_mcp_guide(content: str) -> str:
    if MCP_CATALOG_MARKER not in content:
        return content
    table = "\n".join(
        [
            "Additional official connections from the integration catalog:",
            "",
            "Respect each entry's purpose. Public documentation servers do not "
            "provide private account access. For regional or tenant-specific servers, "
            "confirm the user's region or obtain their endpoint before connecting.",
            "",
            "| Service | URL | Authentication | Purpose | Setup requirements |",
            "|---|---|---|---|---|",
            *(_render_entry(entry) for entry in get_mcp_catalog()),
        ]
    )
    return content.replace(MCP_CATALOG_MARKER, table)


def _render_entry(entry: MCPCatalogEntry) -> str:
    cells = [
        entry.display_name,
        _server_location(entry),
        _authentication(entry),
        entry.description,
        entry.mcp_server.setup_instructions,
    ]
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
