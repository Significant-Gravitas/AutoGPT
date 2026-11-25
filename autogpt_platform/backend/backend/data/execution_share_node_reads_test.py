"""The shared-run read fetches each node once, not once per node execution.

A run can hold many executions of one node. These tests count the node reads:
one query for the distinct nodes, and no node in the per-execution include.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import prisma.models
import pytest
from prisma.enums import AgentExecutionStatus

from backend.blocks.io import AgentOutputBlock
from backend.data.execution import get_graph_execution_by_share_token

OUTPUT_BLOCK_ID = AgentOutputBlock().id
SHARED_NODE_ID = "node-shared"
_T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
_EXECUTIONS_OF_ONE_NODE = 5


def _io(name: str, value: Any, n: int) -> dict:
    return {
        "id": f"io-{name}-{n}",
        "name": name,
        "data": json.dumps(value),
        "time": _T0,
    }


def _node_exec(n: int) -> dict:
    """A node execution with no `Node` attached: the read fetches it separately."""
    return {
        "id": f"exec-{n}",
        "agentGraphExecutionId": "ge-1",
        "agentNodeId": SHARED_NODE_ID,
        "executionStatus": AgentExecutionStatus.COMPLETED,
        "addedTime": _T0,
        "queuedTime": _T0 + timedelta(seconds=n),
        "executionData": json.dumps({"name": "note", "value": f"value {n}"}),
        "Input": [_io("name", "note", 0), _io("value", f"value {n}", 1)],
        "Output": [],
    }


def _execution() -> prisma.models.AgentGraphExecution:
    return prisma.models.AgentGraphExecution.model_validate(
        {
            "id": "ge-1",
            "agentGraphId": "g-1",
            "agentGraphVersion": 1,
            "userId": "owner-1",
            "createdAt": _T0,
            "executionStatus": AgentExecutionStatus.COMPLETED,
            "isDeleted": False,
            "isShared": True,
            "shareToken": "tok-1",
            "visibility": "PRIVATE",
            "AgentGraph": {
                "id": "g-1",
                "version": 1,
                "name": "Shared graph",
                "userId": "owner-1",
                "createdAt": _T0,
                "isActive": True,
                "visibility": "PRIVATE",
            },
            "NodeExecutions": [_node_exec(n) for n in range(_EXECUTIONS_OF_ONE_NODE)],
        }
    )


def _node_row() -> prisma.models.AgentNode:
    return prisma.models.AgentNode.model_validate(
        {
            "id": SHARED_NODE_ID,
            "agentBlockId": OUTPUT_BLOCK_ID,
            "agentGraphId": "g-1",
            "agentGraphVersion": 1,
            "constantInput": "{}",
            "metadata": "{}",
        }
    )


@pytest.fixture
def clients(mocker):
    execution_client = mocker.AsyncMock()
    execution_client.find_first.return_value = _execution()
    node_client = mocker.AsyncMock()
    node_client.find_many.return_value = [_node_row()]
    mocker.patch.object(
        prisma.models.AgentGraphExecution, "prisma", return_value=execution_client
    )
    mocker.patch.object(prisma.models.AgentNode, "prisma", return_value=node_client)
    return execution_client, node_client


@pytest.mark.asyncio
async def test_the_nodes_are_read_once_for_the_whole_run(clients):
    execution_client, node_client = clients

    await get_graph_execution_by_share_token("tok-1")

    node_client.find_many.assert_awaited_once()
    assert node_client.find_many.await_args.kwargs["where"] == {
        "id": {"in": [SHARED_NODE_ID]}
    }


@pytest.mark.asyncio
async def test_the_node_is_not_joined_to_every_execution(clients):
    execution_client, _ = clients

    await get_graph_execution_by_share_token("tok-1")

    node_executions = execution_client.find_first.await_args.kwargs["include"][
        "NodeExecutions"
    ]
    assert "Node" not in node_executions["include"]


@pytest.mark.asyncio
async def test_the_outputs_are_unchanged_by_reading_the_node_separately(clients):
    shared = await get_graph_execution_by_share_token("tok-1")

    assert shared is not None
    assert dict(shared.outputs) == {
        "note": [f"value {n}" for n in range(_EXECUTIONS_OF_ONE_NODE)]
    }
