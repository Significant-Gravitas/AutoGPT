from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.mcp.block import MCPToolBlock
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.graph import Link, Node
from backend.executor.manager import execute_node
from backend.executor.utils import validate_exec


@pytest.mark.parametrize(
    "sink_name,saved,expected",
    [
        ("query", {}, {"query": "linked"}),
        ("tool_arguments_#_query", {}, {"query": "linked"}),
        (
            "tool_arguments_#_query",
            {"query": "saved", "limit": 3},
            {"query": "linked", "limit": 3},
        ),
    ],
)
async def test_a_value_linked_into_an_mcp_tool_argument_reaches_the_tool(
    sink_name: str, saved: dict[str, Any], expected: dict[str, Any]
):
    """The value goes through `validate_exec`, as when the upstream node
    enqueues this one, then `execute_node`. A `tool_arguments_#_query` link's
    value arrived nested and was dropped: the tool failed for a missing
    argument, or ran with the saved one."""
    node = _mcp_tool_node(sink_name, saved)
    inputs, error = validate_exec(node, {sink_name: "linked"})
    assert inputs is not None, error
    call_tool = AsyncMock(return_value="ok")

    with patch.object(MCPToolBlock, "_call_mcp_tool", call_tool), patch.object(
        MCPToolBlock, "_auto_lookup_credential", AsyncMock(return_value=None)
    ):
        outputs = [
            output
            async for output in execute_node(
                node, _entry(node, inputs), SimpleNamespace(creds_manager=MagicMock())
            )
        ]

    assert outputs == [("result", "ok")]
    assert call_tool.await_args.kwargs["arguments"] == expected


def _mcp_tool_node(sink_name: str, saved: dict[str, Any]) -> Node:
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
