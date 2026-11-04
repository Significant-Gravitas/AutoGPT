import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import prisma
import pytest
from prisma.enums import AgentExecutionStatus, ResourceVisibility
from prisma.models import AgentGraphExecution, AgentNodeExecution

from backend.data.execution import create_graph_execution


def _created_run(node_count: int) -> AgentGraphExecution:
    """A new run as read back from the database: its node executions are
    nested under it and carry no copy of it."""
    return AgentGraphExecution(
        id="exec-1",
        createdAt=datetime.now(timezone.utc),
        isDeleted=False,
        executionStatus=AgentExecutionStatus.INCOMPLETE,
        agentGraphId="graph-1",
        agentGraphVersion=4,
        userId="owner",
        inputs=json.dumps({"x": "y" * 1000}),
        isShared=False,
        visibility=ResourceVisibility.PRIVATE,
        NodeExecutions=[
            AgentNodeExecution(
                id=f"node-exec-{i}",
                agentGraphExecutionId="exec-1",
                agentNodeId=f"node-{i}",
                executionStatus=AgentExecutionStatus.QUEUED,
                addedTime=datetime.now(timezone.utc),
            )
            for i in range(node_count)
        ],
    )


@pytest.mark.asyncio
async def test_create_graph_execution_reads_the_run_back_once(mocker):
    execution_client = AsyncMock()
    execution_client.create.return_value = _created_run(3)
    mocker.patch.object(
        prisma.models.AgentGraphExecution, "prisma", return_value=execution_client
    )
    mocker.patch.object(prisma.models.LibraryAgent, "prisma", return_value=AsyncMock())

    run = await create_graph_execution(
        graph_id="graph-1",
        graph_version=4,
        starting_nodes_input=[(f"node-{i}", {"value": "y" * 1000}) for i in range(3)],
        inputs={"x": "y" * 1000},
        user_id="owner",
    )

    include = execution_client.create.call_args.kwargs["include"]
    assert "GraphExecution" not in include["NodeExecutions"]["include"]
    assert len(run.node_executions) == 3
    for node_execution in run.node_executions:
        assert node_execution.user_id == "owner"
        assert node_execution.graph_id == "graph-1"
        assert node_execution.graph_version == 4
