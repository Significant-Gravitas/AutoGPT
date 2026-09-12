from unittest.mock import patch

from backend.integrations.mcp_catalog import (
    MCPCatalogEntry,
    MCPServerMetadata,
    get_connectable_mcp_catalog,
)
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
            auth_methods=["oauth"],
        ),
    )
    token_entry = entry.model_copy(
        update={
            "name": "mcp_vendor_token",
            "mcp_server": entry.mcp_server.model_copy(
                update={
                    "auth_mode": "token",
                    "auth_methods": ["bearer"],
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
    assert "| oauth / bearer |" in guide
    assert "Ask an admin to enable the integration." in guide
    assert "Generate a dedicated integration token." in guide


def test_guide_distinguishes_oauth_url_and_optional_permissions():
    entry = MCPCatalogEntry(
        name="mcp_vendor",
        display_name="Vendor",
        description="Search public and private content.",
        official=True,
        mcp_server=MCPServerMetadata(
            server_url="https://mcp.vendor.com/mcp",
            oauth_server_url="https://mcp.vendor.com/mcp/oauth",
            documentation_url="https://docs.vendor.com/mcp",
            setup_instructions="Select your workspace.",
            connection_mode="hosted",
            auth_mode="none",
            auth_methods=["none", "oauth", "bearer"],
            oauth_scopes=["read"],
            oauth_write_scopes=["write"],
        ),
    )
    with patch(
        "backend.integrations.mcp_guide.get_connectable_mcp_catalog",
        return_value=(entry,),
    ):
        guide = render_mcp_guide(MCP_CATALOG_MARKER)

    assert "none / oauth / bearer" in guide
    assert "OAuth uses `https://mcp.vendor.com/mcp/oauth`" in guide
    assert "default OAuth scopes: read" in guide
    assert "optional grants require explicit selection" in guide


def test_guide_includes_distinct_custom_connections_and_region_choices():
    hosted_url = "https://platform.agpt.co"
    with patch(
        "backend.integrations.mcp_guide.settings.config.frontend_base_url",
        hosted_url,
    ):
        guide = render_mcp_guide(MCP_CATALOG_MARKER)

    for entry in get_connectable_mcp_catalog(hosted_url):
        assert entry.display_name in guide
        assert entry.mcp_server.setup_instructions in guide
        for option in entry.mcp_server.server_url_options:
            assert f"{option.label}: `{option.url}`" in guide
    assert "| Langfuse | Choose the account region:" in guide
    assert "| Amplitude | Choose the account region:" in guide
    assert "| Chargebee | User-provided endpoint;" in guide
    assert "| AWS Core |" not in guide
    assert "| 1Password Environments |" not in guide
