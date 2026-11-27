"""A finished run's results come back through the real External API app, with each
output value exactly as the run produced it, whatever its JSON type.

Real key verification and route. Only storage is stood in for: the run lookup
answers the key owner's query with the rows Postgres would return, as JSON text,
and the workflow check finds the workflow.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from prisma.enums import AgentExecutionStatus, ResourceVisibility
from prisma.models import (
    AgentGraphExecution,
    AgentNode,
    AgentNodeExecution,
    AgentNodeExecutionInputOutput,
)

from backend.blocks.io import AgentOutputBlock
from backend.blocks.maths import CalculatorBlock
from backend.data import execution as execution_db
from backend.data import graph as graph_db

GRAPH_ID = "graph-1"
PERSONAL_ORG = "personal-org"
RUN_ID = "run-1"
STARTED = datetime(2026, 11, 19, 0, 51, tzinfo=UTC)
CALCULATOR_INPUT = {"operation": "Add", "a": 5.5, "b": 6.5, "round_result": False}
OUTPUTS: list[tuple[str, Any]] = [
    ("Sum", 12.0),
    ("Count", 3),
    ("Passed", True),
    ("Rows", [1, "two", None]),
    ("Summary", {"total": 12.0, "tags": ["a", "b"], "checked": False}),
    ("Nothing", None),
]


# Keeps the rate limiter's request counter in memory instead of Redis.
@pytest.mark.usefixtures("v1_counter")
async def test_a_completed_runs_outputs_keep_their_json_types(
    mocker, v1_client, v1_generated_keys, v1_key_rows
):
    owner = v1_key_rows[0].user_id
    mocker.patch(
        "backend.api.external.middleware.get_user_default_team",
        AsyncMock(return_value=(PERSONAL_ORG, None)),
    )
    mocker.patch.object(execution_db, "get_user_team_ids", AsyncMock(return_value=[]))
    owners_query = {
        "id": RUN_ID,
        "isDeleted": False,
        "AND": execution_db._execution_visibility_filters(owner, PERSONAL_ORG, []),
    }
    runs = AsyncMock()
    runs.find_first.side_effect = lambda *, where, include: (
        _completed_run(owner) if where == owners_query else None
    )
    mocker.patch.object(AgentGraphExecution, "prisma", return_value=runs)
    mocker.patch.object(graph_db, "get_graph", AsyncMock(return_value=object()))

    response = await v1_client.get(
        f"/v1/graphs/{GRAPH_ID}/executions/{RUN_ID}/results",
        headers={"X-API-Key": v1_generated_keys[0].key},
    )

    assert response.status_code == 200, response.text
    assert _json_text(response.json()) == _json_text(
        {
            "execution_id": RUN_ID,
            "status": "COMPLETED",
            "nodes": [
                {
                    "node_id": "calculator",
                    "input": CALCULATOR_INPUT,
                    "output": {"result": [12.0]},
                },
                *(
                    {
                        "node_id": f"output-{name}",
                        "input": value,
                        "output": {"output": [value], "name": [name]},
                    }
                    for name, value in OUTPUTS
                ),
            ],
            "output": [{name: value} for name, value in OUTPUTS],
        }
    )


def _json_text(value: Any) -> str:
    """Python compares 12.0 equal to 12 and True equal to 1; JSON text does not."""
    return json.dumps(value, sort_keys=True)


def _completed_run(owner: str) -> AgentGraphExecution:
    """A calculator's sum, then one output block per value, as stored."""
    return AgentGraphExecution(
        id=RUN_ID,
        createdAt=STARTED,
        isDeleted=False,
        executionStatus=AgentExecutionStatus.COMPLETED,
        agentGraphId=GRAPH_ID,
        agentGraphVersion=1,
        userId=owner,
        isShared=False,
        visibility=ResourceVisibility.PRIVATE,
        NodeExecutions=[
            _node_execution(
                0,
                "calculator",
                CalculatorBlock().id,
                CALCULATOR_INPUT,
                {"result": 12.0},
            ),
            *(
                _node_execution(
                    order,
                    f"output-{name}",
                    AgentOutputBlock().id,
                    {"name": name, "value": value},
                    {"output": value, "name": name},
                )
                for order, (name, value) in enumerate(OUTPUTS, start=1)
            ),
        ],
    )


def _node_execution(
    order: int,
    node_id: str,
    block_id: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> AgentNodeExecution:
    queued = STARTED + timedelta(seconds=order)
    return AgentNodeExecution(
        id=f"{node_id}-execution",
        agentGraphExecutionId=RUN_ID,
        agentNodeId=node_id,
        executionStatus=AgentExecutionStatus.COMPLETED,
        addedTime=queued,
        queuedTime=queued,
        executionData=json.dumps(inputs),
        Node=AgentNode(
            id=node_id,
            agentBlockId=block_id,
            agentGraphId=GRAPH_ID,
            agentGraphVersion=1,
            constantInput="{}",
            metadata="{}",
        ),
        Input=[],
        Output=[
            AgentNodeExecutionInputOutput(
                id=f"{node_id}-{pin}", name=pin, data=json.dumps(value), time=queued
            )
            for pin, value in outputs.items()
        ],
    )
