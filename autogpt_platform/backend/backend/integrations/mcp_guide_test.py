from unittest.mock import patch

from backend.integrations.mcp_catalog import (
    MCPCatalogEntry,
    MCPServerMetadata,
    get_mcp_catalog,
)
from backend.integrations.mcp_guide import MCP_CATALOG_MARKER, render_mcp_guide


def test_guide_distinguishes_oauth_url_and_optional_permissions():
    entry = MCPCatalogEntry(
        name="mcp_vendor",
        display_name="Vendor",
        description="Search public and private content.",
        mcp_server=MCPServerMetadata(
            server_url="https://mcp.vendor.com/mcp",
            oauth_server_url="https://mcp.vendor.com/mcp/oauth",
            documentation_url="https://docs.vendor.com/mcp",
            setup_instructions="Select your workspace.",
            connection_mode="hosted",
            auth_methods=["none", "oauth", "bearer"],
            oauth_scopes=["read"],
            oauth_write_scopes=["write"],
        ),
    )
    with patch("backend.integrations.mcp_guide.get_mcp_catalog", return_value=(entry,)):
        guide = render_mcp_guide(MCP_CATALOG_MARKER)

    assert "none / oauth / bearer" in guide
    assert "OAuth uses `https://mcp.vendor.com/mcp/oauth`" in guide
    assert "default OAuth scopes: read" in guide
    assert "optional grants require explicit selection" in guide


def test_guide_includes_connections_and_endpoint_choices():
    guide = render_mcp_guide(MCP_CATALOG_MARKER)

    for entry in get_mcp_catalog():
        assert entry.display_name in guide
        assert entry.mcp_server.setup_instructions in guide
        server = entry.mcp_server
        if server.server_url:
            assert f"`{server.server_url}`" in guide
        elif not server.server_url_options:
            assert f"[setup instructions]({server.documentation_url})" in guide
        for option in server.server_url_options:
            assert f"{option.label}: `{option.url}`" in guide
