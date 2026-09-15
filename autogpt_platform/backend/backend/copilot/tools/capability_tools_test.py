"""find/describe/run/resume_capability over the real registry with the
execution paths mocked at their boundaries."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.capabilities.mcp_review import COPILOT_MCP_NODE_PREFIX
from backend.copilot.capabilities.ranking import ConnectionState
from backend.copilot.capabilities.sources import EAGER_CORE
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.copilot.context import set_execution_context
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.prompting import SHARED_TOOL_NOTES
from backend.copilot.tools import (
    DEFERRED_TOOL_NAMES,
    TOOL_REGISTRY,
    get_available_tools,
)

from ._test_data import make_session
from .describe_capability import DescribeCapabilityTool
from .find_capability import FindCapabilityTool
from .models import (
    BlockDetails,
    BlockDetailsResponse,
    BlockOutputResponse,
    CapabilityDetailsResponse,
    CapabilityListResponse,
    ErrorResponse,
    MCPToolOutputResponse,
    NoResultsResponse,
    ReviewRequiredResponse,
)
from .resume_capability import ResumeCapabilityTool
from .run_capability import RunCapabilityTool

USER = "user-cap-tools"
LINEAR_STATE = ConnectionState(providers=frozenset({"linear"}))


@pytest.fixture(autouse=True)
def _clean_context():
    session = make_session(USER)
    set_execution_context(USER, session)
    yield
    set_execution_context(None, None)


# ---------------------------------------------------------------- registry


def test_eager_and_deferred_split_the_registry():
    eager = set(TOOL_REGISTRY) & EAGER_CORE
    assert eager.isdisjoint(DEFERRED_TOOL_NAMES)
    assert eager | DEFERRED_TOOL_NAMES == set(TOOL_REGISTRY)
    shown = {t["function"]["name"] for t in get_available_tools()}
    assert shown <= eager
    assert {
        "find_capability",
        "describe_capability",
        "run_capability",
        "resume_capability",
    } <= shown
    assert len(get_available_tools(include_deferred=True)) > len(shown)


def test_prompt_names_only_registry_tools():
    for legacy in (
        "find_block",
        "run_block",
        "run_mcp_tool",
        "get_mcp_guide",
        "continue_run_block",
    ):
        assert legacy not in SHARED_TOOL_NOTES, legacy
    assert (
        "find_capability" in SHARED_TOOL_NOTES
        and "resume_capability" in SHARED_TOOL_NOTES
    )


# ------------------------------------------------------------ find_capability


async def test_find_capability_ranks_connected_service_first():
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=LINEAR_STATE),
    ):
        result = await FindCapabilityTool()._execute(
            USER, make_session(USER), query="linear issue"
        )
    assert isinstance(result, CapabilityListResponse)
    assert result.service == "linear"
    assert result.capabilities[0]["name"].startswith("Linear")
    assert result.capabilities[0]["connected"] is True
    assert all(c.get("class") != "primitive" for c in result.capabilities)


async def test_find_capability_no_results_points_to_open_world():
    with patch(
        "backend.copilot.tools.find_capability.load_connection_state",
        AsyncMock(return_value=ConnectionState()),
    ):
        result = await FindCapabilityTool()._execute(
            USER, make_session(USER), query="zzqx"
        )
    assert isinstance(result, NoResultsResponse)
    assert any("MCP server" in s for s in result.suggestions)


# -------------------------------------------------------- describe_capability


async def test_describe_tool_returns_parameters():
    result = await DescribeCapabilityTool()._execute(
        USER, make_session(USER), id="tool:list_schedules"
    )
    assert isinstance(result, CapabilityDetailsResponse)
    assert result.capability["id"] == "tool:list_schedules"
    assert result.parameters == TOOL_REGISTRY["list_schedules"].parameters


async def test_describe_block_collapses_large_enums_unless_expanded():
    details = BlockDetailsResponse(
        message="Block 'X' details.",
        block=BlockDetails(
            id="b",
            name="X",
            description="",
            inputs={"properties": {"model": {"enum": list(range(40))}}},
        ),
    )
    with patch(
        "backend.copilot.tools.describe_capability.RunBlockTool._execute",
        AsyncMock(return_value=details.model_copy(deep=True)),
    ):
        collapsed = await DescribeCapabilityTool()._execute(
            USER, make_session(USER), id="SendWebRequestBlock"
        )
    assert isinstance(collapsed, BlockDetailsResponse)
    assert collapsed.block.inputs["properties"]["model"]["enum_count"] == 40
    assert "run_capability" in collapsed.message
    with patch(
        "backend.copilot.tools.describe_capability.RunBlockTool._execute",
        AsyncMock(return_value=details.model_copy(deep=True)),
    ):
        full = await DescribeCapabilityTool()._execute(
            USER, make_session(USER), id="SendWebRequestBlock", expand=True
        )
    assert isinstance(full, BlockDetailsResponse)
    assert len(full.block.inputs["properties"]["model"]["enum"]) == 40


async def test_describe_unknown_id():
    result = await DescribeCapabilityTool()._execute(
        USER, make_session(USER), id="tool:nope"
    )
    assert isinstance(result, ErrorResponse) and "find_capability" in result.message


# ------------------------------------------------------------ run_capability


def _stub_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = "stub"
    tool.parameters = {"type": "object", "properties": {"x": {"type": "string"}}}
    tool._execute = AsyncMock(return_value=ErrorResponse(message="ran", session_id="s"))
    return tool


async def test_run_tool_dispatches_to_the_deferred_tool():
    stub = _stub_tool("list_schedules")
    with patch(
        "backend.copilot.tools.run_capability.configured_tool", return_value=stub
    ):
        result = await RunCapabilityTool()._execute(
            USER, make_session(USER), id="tool:list_schedules", input={"x": "1"}
        )
    assert result.message == "ran"
    stub._execute.assert_awaited_once()
    assert stub._execute.await_args.kwargs == {"x": "1"}


async def test_run_tool_respects_turn_hidden_tools():
    session = make_session(USER)
    set_execution_context(USER, session, hidden_tools=frozenset({"list_schedules"}))
    stub = _stub_tool("list_schedules")
    with patch(
        "backend.copilot.tools.run_capability.configured_tool", return_value=stub
    ):
        result = await RunCapabilityTool()._execute(
            USER, session, id="tool:list_schedules", input={}
        )
    assert isinstance(result, ErrorResponse) and result.error == "tool_disabled"
    stub._execute.assert_not_awaited()


async def test_run_block_honours_the_run_block_gate():
    session = make_session(USER)
    set_execution_context(
        USER, session, permissions=CopilotPermissions(tools=["run_block"])
    )
    with patch(
        "backend.copilot.tools.run_capability.RunBlockTool._execute", AsyncMock()
    ) as run:
        result = await RunCapabilityTool()._execute(
            USER, session, id="SendWebRequestBlock", input={}
        )
    assert isinstance(result, ErrorResponse) and result.error == "tool_disabled"
    run.assert_not_awaited()


async def test_run_block_forwards_to_run_block_tool():
    output = BlockOutputResponse(
        message="ok", block_id="b", block_name="SendWebRequestBlock", outputs={}
    )
    with patch(
        "backend.copilot.tools.run_capability.RunBlockTool._execute",
        AsyncMock(return_value=output),
    ) as run:
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="SendWebRequestBlock",
            input={"url": "https://x"},
            validate_only=True,
        )
    assert result is output
    kwargs = run.await_args.kwargs
    assert (
        kwargs["input_data"] == {"url": "https://x"} and kwargs["validate_only"] is True
    )
    assert kwargs["block_id"]


async def test_run_mcp_catalog_write_runs_without_review():
    out = MCPToolOutputResponse(
        message="done", server_url="u", tool_name="create_issue"
    )
    with patch(
        "backend.copilot.tools.run_capability.RunMCPToolTool._execute",
        AsyncMock(return_value=out),
    ) as run, patch(
        "backend.copilot.tools.run_capability.open_mcp_review", AsyncMock()
    ) as review:
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="mcp:mcp.linear.app",
            input={"tool": "create_issue", "arguments": {"a": 1}},
        )
    assert result is out
    review.assert_not_awaited()
    assert run.await_args.kwargs["tool_name"] == "create_issue"
    assert run.await_args.kwargs["tool_arguments"] == {"a": 1}


async def test_run_mcp_open_world_write_pauses_for_review():
    with patch(
        "backend.copilot.tools.run_capability.RunMCPToolTool._execute", AsyncMock()
    ) as run, patch(
        "backend.copilot.tools.run_capability.open_mcp_review",
        AsyncMock(return_value="copilot-mcp-x:1"),
    ):
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="https://mcp.example.com/mcp",
            input={"tool": "delete_thing", "arguments": {}},
        )
    assert isinstance(result, ReviewRequiredResponse)
    assert (
        result.review_id == "copilot-mcp-x:1" and "resume_capability" in result.message
    )
    run.assert_not_awaited()


async def test_run_mcp_open_world_read_runs():
    out = MCPToolOutputResponse(message="done", server_url="u", tool_name="list_things")
    with patch(
        "backend.copilot.tools.run_capability.RunMCPToolTool._execute",
        AsyncMock(return_value=out),
    ):
        result = await RunCapabilityTool()._execute(
            USER,
            make_session(USER),
            id="https://mcp.example.com/mcp",
            input={"tool": "list_things"},
        )
    assert result is out


async def test_run_mcp_validate_only_describes_input_shape():
    result = await RunCapabilityTool()._execute(
        USER, make_session(USER), id="mcp:mcp.linear.app", input={}, validate_only=True
    )
    assert isinstance(result, CapabilityDetailsResponse)
    assert set(result.parameters["properties"]) == {"tool", "arguments", "connect"}


async def test_run_unknown_id():
    result = await RunCapabilityTool()._execute(
        USER, make_session(USER), id="block:nope", input={}
    )
    assert isinstance(result, ErrorResponse) and "find_capability" in result.message


# --------------------------------------------------------- resume_capability


def _review(
    review_id: str, status: ReviewStatus, payload: dict[str, Any], session_id: str
) -> MagicMock:
    review = MagicMock()
    review.node_exec_id = review_id
    review.status = status
    review.payload = payload
    review.graph_exec_id = f"{COPILOT_SESSION_PREFIX}{session_id}"
    return review


async def test_resume_block_review_delegates_to_continue_run_block():
    with patch(
        "backend.copilot.tools.resume_capability.ContinueRunBlockTool._execute",
        AsyncMock(return_value=ErrorResponse(message="cont")),
    ) as cont:
        result = await ResumeCapabilityTool()._execute(
            USER, make_session(USER), review_id="copilot-node-b:1"
        )
    assert result.message == "cont"
    assert cont.await_args.kwargs == {"review_id": "copilot-node-b:1"}


async def test_resume_mcp_review_replays_call_with_overrides():
    session = make_session(USER)
    review_id = f"{COPILOT_MCP_NODE_PREFIX}mcp.example.com:ab12"
    review = _review(
        review_id,
        ReviewStatus.APPROVED,
        {
            "server_url": "https://mcp.example.com/mcp",
            "tool": "delete_thing",
            "arguments": {"id": 1, "force": False},
        },
        session.session_id,
    )
    db = MagicMock()
    db.get_reviews_by_node_exec_ids = AsyncMock(return_value={review_id: review})
    db.delete_review_by_node_exec_id = AsyncMock()
    out = MCPToolOutputResponse(
        message="done", server_url="u", tool_name="delete_thing"
    )
    with patch(
        "backend.copilot.tools.resume_capability.review_db", return_value=db
    ), patch(
        "backend.copilot.tools.resume_capability.RunMCPToolTool._execute",
        AsyncMock(return_value=out),
    ) as run:
        result = await ResumeCapabilityTool()._execute(
            USER, session, review_id=review_id, input_overrides={"force": True}
        )
    assert result is out
    assert run.await_args.kwargs["tool_arguments"] == {"id": 1, "force": True}
    db.delete_review_by_node_exec_id.assert_awaited_once_with(review_id, USER)


async def test_resume_mcp_review_waits_for_approval():
    session = make_session(USER)
    review_id = f"{COPILOT_MCP_NODE_PREFIX}mcp.example.com:cd34"
    review = _review(review_id, ReviewStatus.WAITING, {}, session.session_id)
    db = MagicMock()
    db.get_reviews_by_node_exec_ids = AsyncMock(return_value={review_id: review})
    with patch("backend.copilot.tools.resume_capability.review_db", return_value=db):
        result = await ResumeCapabilityTool()._execute(
            USER, session, review_id=review_id
        )
    assert isinstance(result, ErrorResponse) and "not been approved" in result.message
