from unittest.mock import patch

from backend.integrations.mcp_catalog import MCPCatalogEntry, MCPServerMetadata
from backend.integrations.mcp_guide import MCP_CATALOG_MARKER, render_mcp_guide


def test_guide_preserves_authentication_and_requirements_for_shared_urls():
    entry = MCPCatalogEntry(
        name="mcp_vendor",
        display_name="Vendor",
        description="Search workspace content.",
        official=True,
        mcp_server=MCPServerMetadata(
            server_url="https://mcp.vendor.com/mcp",
            documentation_url="https://docs.vendor.com/mcp",
            setup_instructions="Ask an admin to enable the integration.",
            connection_mode="hosted",
            auth_mode="oauth",
        ),
    )
    token_entry = entry.model_copy(
        update={
            "name": "mcp_vendor_token",
            "mcp_server": entry.mcp_server.model_copy(
                update={
                    "auth_mode": "token",
                    "setup_instructions": "Generate a dedicated integration token.",
                }
            ),
        }
    )
    with patch(
        "backend.integrations.mcp_guide.get_connectable_mcp_catalog",
        return_value=(entry, token_entry),
    ):
        guide = render_mcp_guide(MCP_CATALOG_MARKER)

    assert guide.count("`https://mcp.vendor.com/mcp`") == 1
    assert "| oauth / token |" in guide
    assert "Ask an admin to enable the integration." in guide
    assert "Generate a dedicated integration token." in guide
