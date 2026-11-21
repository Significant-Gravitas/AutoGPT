from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.agent import AgentExecutorBlock
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.graph import Node
from backend.executor.manager import execute_node
from backend.executor.utils import validate_exec

SAVED_MASKS = {"sub-node": {"method": "POST"}}
RUN_MASKS = {"agent-node": {"credentials": {"id": "picked-for-this-run"}}}


@pytest.mark.parametrize("run_masks", [None, {}, RUN_MASKS])
async def test_a_sub_graph_run_gets_the_masks_of_the_run_not_of_the_saved_node(
    run_masks: dict[str, Any] | None,
):
    """`nodes_input_masks` carries the running graph's own masks (credentials
    picked for the run, a webhook payload) into the sub-graph run. Saved on the
    node it would rewrite any sub-graph input at run time, while every check
    that reads the sub-graph as stored still reads the saved value."""
    node = _agent_executor_node(SAVED_MASKS)

    started = await _sub_graph_run_started(node, run_masks)

    assert started["user_id"] == "user-1"
    assert (started["nodes_input_masks"] or {}) == (run_masks or {})


async def _sub_graph_run_started(
    node: Node, run_masks: dict[str, Any] | None
) -> dict[str, Any]:
    """Runs the node through the real `execute_node` and returns the keyword
    arguments the sub-graph run was started with."""
    inputs, error = validate_exec(node, {})
    assert inputs is not None, error
    add_graph_execution = AsyncMock(return_value=SimpleNamespace(id="sub-exec-1"))

    with (
        patch("backend.executor.utils.add_graph_execution", add_graph_execution),
        patch.object(AgentExecutorBlock, "_run", _one_output),
    ):
        outputs = [
            output
            async for output in execute_node(
                node,
                _entry(node, inputs),
                SimpleNamespace(creds_manager=MagicMock()),
                nodes_input_masks=run_masks,
            )
        ]

    assert outputs == [("result", "ok")]
    assert add_graph_execution.await_args is not None
    return add_graph_execution.await_args.kwargs


async def _one_output(self, **kwargs):
    yield "result", "ok"


def _agent_executor_node(saved_masks: dict[str, Any]) -> Node:
    return Node(
        id="agent-node",
        block_id=AgentExecutorBlock().id,
        input_default={
            "user_id": "someone-else",
            "graph_id": "sub-graph",
            "graph_version": 1,
            "inputs": {},
            "input_schema": {},
            "output_schema": {},
            "nodes_input_masks": saved_masks,
        },
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
