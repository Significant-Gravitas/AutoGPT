from unittest.mock import AsyncMock

import pytest

from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.models import (
    CapabilityEntry,
    Connection,
    Implementation,
)
from backend.copilot.capabilities.ranking import ConnectionState
from backend.copilot.context import set_execution_context
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.describe_capability import DescribeCapabilityTool
from backend.copilot.tools.find_capability import NEEDS_EXPERT_GRANT, FindCapabilityTool
from backend.copilot.tools.models import (
    CapabilityDetailsResponse,
    CapabilityListResponse,
    MCPToolsDiscoveredResponse,
    NoResultsResponse,
)
from backend.copilot.tools.run_capability import RunCapabilityTool

USER = "user-custom-mcp-search"
SERVER_URL = "https://mcp.custom-payments.example.com/mcp"


@pytest.fixture(autouse=True)
def clean_context():
    """Keep the execution context local to each test's session."""
    set_execution_context(USER, make_session(USER))
    yield
    set_execution_context(None, None)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> CapabilityIndex:
    """Provide an unrelated catalog without external registry dependencies."""
    index = CapabilityIndex(
        [
            CapabilityEntry(
                id="tool:web_search",
                kind="tool",
                name="web_search",
                purpose="Search the web.",
            ),
            CapabilityEntry(
                id="tool:read_file",
                kind="tool",
                name="read_file",
                purpose="Read a workspace file.",
            ),
            CapabilityEntry(
                id="tool:list_schedules",
                kind="tool",
                name="list_schedules",
                purpose="List scheduled tasks.",
            ),
        ]
    )
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.session_registry",
        AsyncMock(return_value=index),
    )
    return index


@pytest.mark.parametrize(
    "query", ["custom-payments", "mcp.custom-payments.example.com", SERVER_URL]
)
async def test_find_connected_custom_mcp_server(
    query: str, registry: CapabilityIndex, monkeypatch: pytest.MonkeyPatch
):
    """Find a connected custom endpoint by service name, hostname, or URL."""
    load_state = AsyncMock(
        return_value=ConnectionState(server_urls=frozenset({SERVER_URL}))
    )
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.load_connection_state", load_state
    )

    result = await FindCapabilityTool()._execute(USER, make_session(USER), query=query)

    assert isinstance(result, CapabilityListResponse)
    assert result.capabilities[0] == {
        "id": SERVER_URL,
        "name": "mcp.custom-payments.example.com",
        "purpose": "MCP server connected at mcp.custom-payments.example.com.",
        "kind": "mcp_server",
        "connected": True,
    }
    assert sum(c["id"] == SERVER_URL for c in result.capabilities) == 1
    assert len(registry) == 3
    load_state.assert_awaited_once_with(USER, None)


async def test_custom_servers_are_isolated_and_connections_refresh(
    registry: CapabilityIndex, monkeypatch: pytest.MonkeyPatch
):
    """Refresh per-user connections without mutating the shared registry."""
    connected = ConnectionState(server_urls=frozenset({SERVER_URL}))
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(side_effect=[ConnectionState(), connected, ConnectionState()]),
    )
    tool = FindCapabilityTool()
    before = await tool._execute(USER, make_session(USER), query="custom-payments")
    after = await tool._execute(USER, make_session(USER), query="custom-payments")
    other_user = await tool._execute(
        "another-user", make_session("another-user"), query="custom-payments"
    )

    assert isinstance(before, NoResultsResponse)
    assert isinstance(after, CapabilityListResponse)
    assert after.capabilities[0]["id"] == SERVER_URL
    assert isinstance(other_user, NoResultsResponse)
    assert registry.get(SERVER_URL) is None


async def test_catalog_metadata_is_preserved_without_duplicate_results(
    registry: CapabilityIndex, monkeypatch: pytest.MonkeyPatch
):
    """Equivalent host spelling keeps the catalog entry and its metadata."""
    catalog = _catalog_entry()
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.session_registry",
        AsyncMock(return_value=registry.with_entries([catalog])),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(
            return_value=ConnectionState(
                server_urls=frozenset({"HTTPS://MCP.Custom-Payments.Example.COM/mcp/"})
            )
        ),
    )

    result = await FindCapabilityTool()._execute(
        USER, make_session(USER), query="custom-payments"
    )

    assert isinstance(result, CapabilityListResponse)
    assert result.capabilities == [{**catalog.listing(), "connected": True}]


@pytest.mark.parametrize("path", ["TenantMCP", "MCP"])
async def test_distinct_endpoints_on_one_host_are_not_deduplicated(
    path: str, registry: CapabilityIndex, monkeypatch: pytest.MonkeyPatch
):
    """A custom path stays discoverable and runnable beside a catalog endpoint."""
    tenant_url = f"https://mcp.custom-payments.example.com/{path}"
    catalog_index = registry.with_entries([_catalog_entry()])
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.session_registry",
        AsyncMock(return_value=catalog_index),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(
            return_value=ConnectionState(
                server_urls=frozenset({tenant_url, tenant_url + "/"})
            )
        ),
    )

    result = await FindCapabilityTool()._execute(
        USER, make_session(USER), query="mcp.custom-payments.example.com"
    )

    assert isinstance(result, CapabilityListResponse)
    assert {c["id"] for c in result.capabilities} == {
        "mcp:mcp.custom-payments.example.com",
        tenant_url,
    }
    custom = next(c for c in result.capabilities if c["id"] == tenant_url)
    assert custom["connected"] is True
    monkeypatch.setattr(
        "backend.copilot.tools.session_registry.get_registry", lambda: catalog_index
    )
    monkeypatch.setattr(
        "backend.copilot.tools.session_registry.load_connection_state",
        AsyncMock(return_value=ConnectionState(server_urls=frozenset({tenant_url}))),
    )
    validated = await RunCapabilityTool()._execute(
        USER, make_session(USER), id=custom["id"], input={}, validate_only=True
    )
    assert isinstance(validated, CapabilityDetailsResponse)
    assert validated.capability["id"] == tenant_url


@pytest.mark.parametrize("granted", [False, True])
async def test_custom_server_connection_respects_expert_grants(
    granted: bool, registry: CapabilityIndex, monkeypatch: pytest.MonkeyPatch
):
    """Distinguish granted credentials from stored credentials needing a grant."""
    connected = ConnectionState(server_urls=frozenset({SERVER_URL}))
    state = (
        connected.model_copy(update={"ungranted": connected})
        if granted
        else ConnectionState(ungranted=connected)
    )
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=state),
    )

    result = await FindCapabilityTool()._execute(
        USER, make_session(USER, expert_id="expert-1"), query="custom-payments"
    )

    assert isinstance(result, CapabilityListResponse)
    assert len(result.capabilities) == 1
    assert result.capabilities[0]["connected"] == (
        True if granted else NEEDS_EXPERT_GRANT
    )


@pytest.mark.parametrize(
    "options", [{"context": "graph"}, {"kind": "skill"}, {"kind": "block"}]
)
async def test_custom_servers_respect_search_filters(
    options: dict[str, str], registry: CapabilityIndex, monkeypatch: pytest.MonkeyPatch
):
    """Custom MCP entries obey capability kind and execution-context filters."""
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=ConnectionState(server_urls=frozenset({SERVER_URL}))),
    )

    result = await FindCapabilityTool()._execute(
        USER, make_session(USER), query="custom-payments", **options
    )

    assert isinstance(result, NoResultsResponse)


@pytest.mark.parametrize(
    "url",
    [
        "invalid-custom-payments-url",
        "http://mcp.custom-payments.example.com/mcp",
        "https://[invalid-custom-payments/mcp",
    ],
)
async def test_invalid_stored_urls_do_not_break_search(
    url: str, registry: CapabilityIndex, monkeypatch: pytest.MonkeyPatch
):
    """Reject unusable stored endpoints without breaking capability search."""
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=ConnectionState(server_urls=frozenset({url}))),
    )
    result = await FindCapabilityTool()._execute(
        USER, make_session(USER), query="custom-payments"
    )
    assert isinstance(result, NoResultsResponse)


async def test_discovered_custom_server_id_can_be_described_and_run(
    registry: CapabilityIndex, monkeypatch: pytest.MonkeyPatch
):
    """Use a discovered raw URL throughout the find/describe/run workflow."""
    url = "https://mcp.custom-payments.example.com/mcp"
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=ConnectionState(server_urls=frozenset({url}))),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.session_registry.get_registry", lambda: registry
    )
    discovery = MCPToolsDiscoveredResponse(message="Tools", server_url=url, tools=[])
    run_mcp = AsyncMock(return_value=discovery)
    monkeypatch.setattr(
        "backend.copilot.tools.describe_capability.RunMCPToolTool._execute", run_mcp
    )
    session = make_session(USER)

    result = await FindCapabilityTool()._execute(USER, session, query="custom-payments")
    assert isinstance(result, CapabilityListResponse)
    capability_id = result.capabilities[0]["id"]
    described = await DescribeCapabilityTool()._execute(USER, session, id=capability_id)
    assert described is discovery
    run_mcp.assert_awaited_once_with(USER, session, server_url=url)
    validated = await RunCapabilityTool()._execute(
        USER, session, id=capability_id, input={}, validate_only=True
    )
    assert isinstance(validated, CapabilityDetailsResponse)
    assert validated.capability["id"] == url


def _catalog_entry() -> CapabilityEntry:
    """Build a catalog server with richer metadata than a custom fallback."""
    return CapabilityEntry(
        id="mcp:mcp.custom-payments.example.com",
        kind="mcp_server",
        name="Custom Payments",
        purpose="Manage custom invoices and payments.",
        tags=["mcp", "custom-payments", "mcp.custom-payments.example.com"],
        context="direct",
        connection=Connection(required=True, key_type="server_url", key=SERVER_URL),
        implementations=[Implementation(kind="mcp_server", ref=SERVER_URL)],
    )
