from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.mcp.block import MCPToolBlock
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.graph import Link, Node
from backend.executor.manager import execute_node
from backend.executor.utils import validate_exec

_NESTED = {"type": "object"}


@pytest.mark.parametrize(
    "sink_name,linked,saved,expected,extra_props",
    [
        ("query", "linked", {}, {"query": "linked"}, {}),
        ("tool_arguments_#_query", "linked", {}, {"query": "linked"}, {}),
        (
            "tool_arguments_#_query",
            "linked",
            {"query": "saved", "limit": 3},
            {"query": "linked", "limit": 3},
            {},
        ),
        ("tool_arguments", {"query": "linked"}, {}, {"query": "linked"}, {}),
        (
            "limit",
            4,
            {"query": "saved", "tool_arguments": {"query": "nested"}},
            {"query": "saved", "limit": 4, "tool_arguments": {"query": "nested"}},
            {"tool_arguments": _NESTED},
        ),
    ],
)
async def test_a_value_linked_into_an_mcp_tool_argument_reaches_the_tool(
    sink_name: str,
    linked: Any,
    saved: dict[str, Any],
    expected: dict[str, Any],
    extra_props: dict[str, Any],
):
    """The value goes through `validate_exec`, as when the upstream node
    enqueues this one, then `execute_node`. A `tool_arguments_#_query` link's
    value arrived nested and was dropped: the tool failed for a missing
    argument, or ran with the saved one. A tool's own `tool_arguments`
    parameter stays one argument."""
    node = _mcp_tool_node(sink_name, saved, extra_props)

    assert await _arguments_sent(node, {sink_name: linked}) == expected


async def test_an_argument_the_orchestrator_supplies_beats_the_saved_one():
    """The Orchestrator block enqueues a tool node with its saved inputs,
    `tool_arguments` among them, plus the arguments the model chose."""
    node = _mcp_tool_node("query", {"query": "saved", "limit": 3}, {})

    sent = await _arguments_sent(node, {**node.input_default, "query": "the model's"})

    assert sent == {"query": "the model's", "limit": 3}


async def test_a_nested_link_beats_a_bare_one_whatever_is_saved():
    """Both links feed `query`; the nested one wins even when its value is
    the saved one, so the outcome does not depend on what was saved."""
    node = _mcp_tool_node("query", {"query": "saved"}, {})
    node.input_links.append(
        node.input_links[0].model_copy(update={"sink_name": "tool_arguments_#_query"})
    )

    sent = await _arguments_sent(
        node, {"query": "bare", "tool_arguments_#_query": "saved"}
    )

    assert sent == {"query": "saved"}


async def _arguments_sent(node: Node, enqueued: dict[str, Any]) -> dict[str, Any]:
    inputs, error = validate_exec(node, enqueued)
    assert inputs is not None, error
    call_tool = AsyncMock(return_value="ok")

    with patch.object(MCPToolBlock, "_call_mcp_tool", call_tool):
        outputs = [
            output
            async for output in execute_node(
                node, _entry(node, inputs), SimpleNamespace(creds_manager=MagicMock())
            )
        ]

    assert outputs == [("result", "ok")]
    return call_tool.await_args.kwargs["arguments"]


def _mcp_tool_node(
    sink_name: str, saved: dict[str, Any], extra_props: dict[str, Any]
) -> Node:
    return Node(
        id="tool",
        block_id=MCPToolBlock().id,
        input_default={
            "server_url": "https://mcp.example.com/mcp",
            "selected_tool": "search",
            "tool_input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    **extra_props,
                },
                "required": ["query"],
            },
            "tool_arguments": saved,
        },
        input_links=[
            Link(
                source_id="value",
                sink_id="tool",
                source_name="output",
                sink_name=sink_name,
            )
        ],
    )


def _entry(node: Node, inputs: dict[str, Any]) -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id=node.id,
        block_id=node.block_id,
        inputs=inputs,
        execution_context=ExecutionContext(),
    )
