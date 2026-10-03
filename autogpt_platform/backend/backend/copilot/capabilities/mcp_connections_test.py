from .index import CapabilityIndex
from .mcp_connections import connected_mcp_entries
from .ranking import ConnectionState


def test_custom_connections_preserve_distinct_case_sensitive_paths():
    """Two connected endpoints that differ by path case must both be discoverable."""
    urls = {"https://mcp.example.com/TenantMCP", "https://mcp.example.com/tenantmcp"}
    state = ConnectionState(server_urls=frozenset(urls))

    entries = connected_mcp_entries(CapabilityIndex([]), state)

    assert {entry.id for entry in entries} == urls
    assert {entry.implementations[0].ref for entry in entries} == urls
