import pytest

from .models import CapabilityEntry, Connection
from .ranking import ConnectionState, normalize_server_url, resolve_connected


@pytest.mark.parametrize(
    "url, expected",
    [
        ("  HTTPS://MCP.PayPal.COM/TenantMCP/  ", "https://mcp.paypal.com/TenantMCP"),
        ("https://MCP.PayPal.COM/MCP", "https://mcp.paypal.com/MCP"),
        ("https://MCP.PayPal.COM/", "https://mcp.paypal.com"),
        (
            "https://User:Pass@MCP.PayPal.COM/MCP?Tenant=ABC/#Tools",
            "https://User:Pass@mcp.paypal.com/MCP?Tenant=ABC/#Tools",
        ),
        ("https://[invalid/MCP", "https://[invalid/MCP"),
    ],
)
def test_server_url_normalization_preserves_case_sensitive_parts(
    url: str, expected: str
):
    """Normalize host spelling without changing the selected endpoint."""
    assert normalize_server_url(url) == expected


def test_a_connection_to_one_path_does_not_connect_a_different_case_path():
    """Credentials for a lowercase path must not mark another endpoint connected."""
    entry = CapabilityEntry(
        id="https://mcp.example.com/TenantMCP",
        kind="mcp_server",
        name="Example",
        purpose="Example MCP server.",
        connection=Connection(
            required=True,
            key_type="server_url",
            key="https://mcp.example.com/TenantMCP",
        ),
    )
    state = ConnectionState(
        server_urls=frozenset({"https://mcp.example.com/tenantmcp"})
    )

    assert resolve_connected(entry, state) is False
