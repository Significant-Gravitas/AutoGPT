from unittest.mock import AsyncMock

import pytest

import backend.copilot.tools.session_registry as session_registry_module
from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.ranking import ConnectionState
from backend.copilot.capabilities.sources.mcp_catalog import mcp_catalog_entries
from backend.copilot.context import set_execution_context

from ._test_data import make_session
from .describe_capability import DescribeCapabilityTool
from .find_capability import NEEDS_EXPERT_GRANT, FindCapabilityTool
from .models import (
    CapabilityDetailsResponse,
    CapabilityListResponse,
    MCPToolOutputResponse,
    MCPToolsDiscoveredResponse,
    ReviewRequiredResponse,
)
from .run_capability import RunCapabilityTool

USER = "user-mcp-review"


@pytest.fixture(autouse=True)
def context():
    """Keep permission checks scoped to the test's personal session."""
    set_execution_context(USER, make_session(USER))
    yield
    set_execution_context(None, None)


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> CapabilityIndex:
    """Use real catalog metadata while isolating registry and credential storage."""
    index = CapabilityIndex(mcp_catalog_entries())
    monkeypatch.setattr(session_registry_module, "get_registry", lambda: index)
    monkeypatch.setattr(
        "backend.copilot.tools.run_capability.get_registry", lambda: index
    )
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.session_registry",
        AsyncMock(return_value=index),
    )
    return index


@pytest.fixture
def connections(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Return the same credential state to search and session URL resolution."""
    loader = AsyncMock(return_value=ConnectionState())
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.load_connection_state", loader
    )
    monkeypatch.setattr(session_registry_module, "load_connection_state", loader)
    return loader


@pytest.mark.parametrize(
    "slug, url, label",
    [
        ("paypal", "https://mcp.paypal.com/mcp", "Production"),
        ("langfuse", "https://cloud.langfuse.com/api/public/mcp", "EU"),
        ("amplitude", "https://mcp.eu.amplitude.com/mcp", "EU"),
    ],
)
async def test_catalog_options_are_connected_and_work_through_describe_run(
    slug: str,
    url: str,
    label: str,
    catalog: CapabilityIndex,
    connections: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Bind option-only catalog presets to stored endpoints without duplicate rows."""
    connections.return_value = ConnectionState(server_urls=frozenset({url}))
    session = make_session(USER)
    result = await FindCapabilityTool()._execute(USER, session, query=slug)
    assert isinstance(result, CapabilityListResponse)
    preset = catalog.get(f"mcp:{slug}")
    assert preset is not None
    assert result.capabilities == [
        {**preset.listing(), "name": f"{preset.name} ({label})", "connected": True}
    ]
    assert preset.connection.key is None
    discovered = MCPToolsDiscoveredResponse(message="Tools", server_url=url, tools=[])
    runner = AsyncMock(return_value=discovered)
    monkeypatch.setattr(
        "backend.copilot.tools.run_capability.RunMCPToolTool._execute", runner
    )

    described = await DescribeCapabilityTool()._execute(USER, session, id=preset.id)
    assert described is discovered
    ran = await RunCapabilityTool()._execute(USER, session, id=preset.id, input={})
    assert ran is discovered
    assert [call.kwargs["server_url"] for call in runner.await_args_list] == [url, url]


async def test_catalog_option_connection_changes_refresh_and_do_not_leak(
    catalog: CapabilityIndex,
    connections: AsyncMock,
):
    """Refresh a stable preset ID when the connected endpoint changes or disappears."""
    urls = ["https://mcp.paypal.com/mcp", "https://mcp.sandbox.paypal.com/mcp"]
    session = make_session(USER)
    for url in urls:
        connections.return_value = ConnectionState(server_urls=frozenset({url}))
        result = await FindCapabilityTool()._execute(USER, session, query="paypal")
        assert isinstance(result, CapabilityListResponse)
        assert len(result.capabilities) == 1
        assert result.capabilities[0]["connected"] is True
        resolved = await session_registry_module.resolve_session_entry(
            USER, session, result.capabilities[0]["id"]
        )
        assert resolved is not None and resolved.implementations[0].ref == url
    connections.return_value = ConnectionState()
    result = await FindCapabilityTool()._execute(
        "another-user", make_session("another-user"), query="paypal"
    )
    assert isinstance(result, CapabilityListResponse)
    assert result.capabilities[0]["connected"] is None
    assert catalog.get("mcp:paypal").connection.key is None


async def test_multiple_catalog_options_remain_distinct_and_prefer_granted_url(
    catalog: CapabilityIndex,
    connections: AsyncMock,
):
    """Keep every endpoint and bind the preset ID to an option the expert can use."""
    allowed = "https://mcp.sandbox.paypal.com/mcp"
    ungranted = "https://mcp.paypal.com/mcp"
    connections.return_value = ConnectionState(
        server_urls=frozenset({allowed}),
        ungranted=ConnectionState(server_urls=frozenset({ungranted})),
    )
    session = make_session(USER, expert_id="expert-1")
    result = await FindCapabilityTool()._execute(USER, session, query="paypal")
    assert isinstance(result, CapabilityListResponse)
    assert {c["id"]: c["connected"] for c in result.capabilities} == {
        "mcp:paypal": True,
        ungranted: NEEDS_EXPERT_GRANT,
    }
    assert {c["id"]: c["name"] for c in result.capabilities} == {
        "mcp:paypal": "PayPal (Sandbox)",
        ungranted: "PayPal (Production)",
    }
    resolved = await session_registry_module.resolve_session_entry(
        USER, session, "mcp:paypal"
    )
    assert resolved is not None and resolved.implementations[0].ref == allowed


@pytest.mark.parametrize("host", ["mcp.linear.app", "mcp.airtable.com"])
async def test_bare_catalog_urls_keep_the_default_endpoint(
    host: str,
    catalog: CapabilityIndex,
    connections: AsyncMock,
):
    """Existing bare-host shortcuts retain their catalog URL and stored credentials."""
    entry = await session_registry_module.resolve_session_entry(
        USER, make_session(USER), f"https://{host}"
    )
    assert entry is not None and entry.id == f"mcp:{host}"
    assert entry.implementations[0].ref == f"https://{host}/mcp"


@pytest.mark.parametrize("suffix", ["", "/TenantMCP", "/MCP"])
@pytest.mark.parametrize("granted", [False, True])
async def test_connected_custom_url_on_catalog_host_keeps_its_endpoint(
    suffix: str,
    granted: bool,
    catalog: CapabilityIndex,
    connections: AsyncMock,
):
    """Stored custom endpoints keep their own credentials and custom write review."""
    url = f"https://mcp.linear.app{suffix}"
    state = ConnectionState(server_urls=frozenset({url}))
    connections.return_value = state if granted else ConnectionState(ungranted=state)
    resolved = await session_registry_module.resolve_session_entry(
        USER, make_session(USER), url
    )
    assert resolved is None
    validated = await RunCapabilityTool()._execute(
        USER, make_session(USER), id=url, input={}, validate_only=True
    )
    assert isinstance(validated, CapabilityDetailsResponse)
    assert validated.capability["id"] == url


async def test_connected_custom_url_keeps_the_write_review_gate(
    catalog: CapabilityIndex,
    connections: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """A stored custom endpoint on a catalog host still requires write approval."""
    url = "https://mcp.linear.app/TenantMCP"
    connections.return_value = ConnectionState(server_urls=frozenset({url}))
    review = AsyncMock(return_value="custom-review")
    runner = AsyncMock()
    monkeypatch.setattr("backend.copilot.tools.run_capability.open_mcp_review", review)
    monkeypatch.setattr(
        "backend.copilot.tools.run_capability.gate_active",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.run_capability.RunMCPToolTool._execute", runner
    )
    result = await RunCapabilityTool()._execute(
        USER,
        make_session(USER),
        id=url,
        input={"tool": "create_issue", "arguments": {}},
    )
    assert isinstance(result, ReviewRequiredResponse)
    assert review.await_args.kwargs["payload"].server_url == url
    runner.assert_not_awaited()


@pytest.mark.parametrize(
    "slug, url",
    [
        ("paypal", "https://mcp.paypal.com/mcp"),
        ("langfuse", "https://cloud.langfuse.com/api/public/mcp"),
        ("amplitude", "https://mcp.eu.amplitude.com/mcp"),
    ],
)
@pytest.mark.parametrize("use_preset_id", [False, True])
@pytest.mark.parametrize(
    "tool_name, gate_on, dry_run",
    [
        ("create_invoice", False, False),
        ("list_invoices", False, False),
        ("create_invoice", True, False),
        ("create_invoice", False, True),
    ],
)
async def test_catalog_option_write_review_respects_execution_mode(
    slug: str,
    url: str,
    use_preset_id: bool,
    tool_name: str,
    gate_on: bool,
    dry_run: bool,
    catalog: CapabilityIndex,
    connections: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
):
    connections.return_value = ConnectionState(server_urls=frozenset({url}))
    review = AsyncMock(return_value="option-review")
    output = MCPToolOutputResponse(message="Done", server_url=url, tool_name=tool_name)
    runner = AsyncMock(return_value=output)
    monkeypatch.setattr("backend.copilot.tools.run_capability.open_mcp_review", review)
    monkeypatch.setattr(
        "backend.copilot.tools.run_capability.gate_active",
        AsyncMock(return_value=gate_on),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.run_capability.RunMCPToolTool._execute", runner
    )
    session = make_session(USER)
    session.metadata.dry_run = dry_run
    arguments = {"amount": "10.00"}

    result = await RunCapabilityTool()._execute(
        USER,
        session,
        id=f"mcp:{slug}" if use_preset_id else url,
        input={"tool": tool_name, "arguments": arguments},
    )

    if tool_name == "list_invoices" or gate_on or dry_run:
        assert result is output
        review.assert_not_awaited()
        runner.assert_awaited_once()
        assert runner.await_args.kwargs["server_url"] == url
        assert runner.await_args.kwargs["tool_name"] == tool_name
        assert runner.await_args.kwargs["tool_arguments"] == arguments
        return

    assert isinstance(result, ReviewRequiredResponse)
    assert result.review_id == "option-review"
    review.assert_awaited_once()
    assert review.await_args.kwargs["user_id"] == USER
    assert review.await_args.kwargs["session_id"] == session.session_id
    assert review.await_args.kwargs["payload"].server_url == url
    assert review.await_args.kwargs["payload"].arguments == arguments
    runner.assert_not_awaited()
