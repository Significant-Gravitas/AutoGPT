import pytest

from .index import CapabilityIndex
from .mcp_connections import connected_mcp_entries, custom_mcp_entry
from .models import CapabilityEntry, Connection
from .ranking import ConnectionState


def test_custom_connections_preserve_distinct_case_sensitive_paths():
    """Two connected endpoints that differ by path case must both be discoverable."""
    urls = {"https://mcp.example.com/TenantMCP", "https://mcp.example.com/tenantmcp"}
    state = ConnectionState(server_urls=frozenset(urls))

    entries = connected_mcp_entries(CapabilityIndex([]), state)

    assert {entry.id for entry in entries} == urls
    assert {entry.implementations[0].ref for entry in entries} == urls


@pytest.mark.parametrize(
    "host, word",
    [
        ("docs.mcp.cloudflare.com", "docs"),
        ("browser.example.com", "browser"),
        ("cloud.langfuse.com", "cloud"),
        ("mcp.eu.example.com", "eu"),
    ],
)
def test_hostname_labels_do_not_hide_unrelated_tools(host: str, word: str):
    """Hostname words are lexical tags, not service names that exclude tools."""
    server = custom_mcp_entry(f"https://{host}/mcp")
    assert server is not None
    tool = CapabilityEntry(
        id=f"tool:search_{word}",
        kind="tool",
        name=f"search_{word}",
        purpose=f"Search the {word}.",
    )
    unrelated = [
        CapabilityEntry(id=f"tool:{name}", kind="tool", name=name, purpose=name)
        for name in ("read_file", "send_email", "list_schedules", "calculate")
    ]
    index = CapabilityIndex([tool, *unrelated, server])

    result = index.search(f"search the {word}")

    assert tool.id in result.ids
    assert result.service is None
    assert index.search(host).service == host


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@mcp.example.com/mcp",
        "https://token@mcp.example.com/mcp",
        "https://mcp.example.com/mcp?api_key=test-secret",
        "https://mcp.example.com/mcp#test-secret",
    ],
)
def test_urls_with_credentials_query_or_fragment_are_not_listed(url: str):
    """Omit endpoints the MCP runner rejects before exposing their IDs to chat."""
    assert custom_mcp_entry(url) is None
    state = ConnectionState(server_urls=frozenset({url}))
    assert connected_mcp_entries(CapabilityIndex([]), state) == []


def test_custom_servers_still_match_a_service_already_in_the_catalog():
    """Plain hostname tags must remain searchable alongside a named catalog service."""
    server = custom_mcp_entry("https://mcp.paypal.com/TenantMCP")
    assert server is not None
    block = CapabilityEntry(
        id="block:paypal",
        kind="block",
        name="PayPalInvoice",
        purpose="PayPal invoices.",
        connection=Connection(required=True, key_type="provider", key="paypal"),
    )
    unrelated = [
        CapabilityEntry(id=f"tool:{name}", kind="tool", name=name, purpose=name)
        for name in ("read_file", "send_email", "list_schedules", "calculate")
    ]
    result = CapabilityIndex([block, *unrelated, server]).search("paypal")
    assert server.id in result.ids
