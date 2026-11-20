"""upsert_execution_input against the real database: which waiting run gets the input."""

from datetime import datetime, timedelta, timezone

import prisma
import pytest

from backend.api.model import CreateGraph
from backend.data import execution
from backend.data.db import query_raw_with_schema
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_the_waiting_run_lookup_is_an_anti_join(server: SpinTestServer):
    plan = await query_raw_with_schema(
        "EXPLAIN " + execution.WAITING_NODE_EXEC_QUERY,
        "graph-exec",
        "node",
        "input1",
        None,
    )
    plan_text = "\n".join(row["QUERY PLAN"] for row in plan)

    assert "Anti Join" in plan_text, plan_text
    assert "SubPlan" not in plan_text, plan_text


@pytest.mark.asyncio(loop_scope="session")
async def test_the_input_goes_to_the_earliest_waiting_run_of_this_node_in_this_run(
    server: SpinTestServer,
):
    node_id, other_node_id, run_id, other_run_id = await _graph_with_two_runs(server)
    other_node = await _waiting_run(other_node_id, run_id, "input", hours_ago=6)
    has_pin = await _waiting_run(node_id, run_id, "data", hours_ago=5)
    not_incomplete = await _waiting_run(node_id, run_id, "input", hours_ago=4)
    await prisma.models.AgentNodeExecution.prisma().update(
        where={"id": not_incomplete},
        data={"executionStatus": execution.ExecutionStatus.COMPLETED},
    )
    other_run = await _waiting_run(node_id, other_run_id, "input", hours_ago=4)
    later = await _waiting_run(node_id, run_id, "input", hours_ago=1)
    earliest = await _waiting_run(node_id, run_id, "input", hours_ago=2)

    result, inputs = await execution.upsert_execution_input(
        node_id, run_id, "data", "value"
    )

    assert result.node_exec_id == earliest
    assert result.node_exec_id not in (
        other_node,
        has_pin,
        not_incomplete,
        other_run,
        later,
    )
    assert inputs == {"input": "seed", "data": "value"}


@pytest.mark.asyncio(loop_scope="session")
async def test_node_exec_id_picks_that_run_and_refuses_it_once_it_has_the_input(
    server: SpinTestServer,
):
    node_id, _, run_id, _ = await _graph_with_two_runs(server)
    await _waiting_run(node_id, run_id, "input", hours_ago=2)
    named = await _waiting_run(node_id, run_id, "input", hours_ago=1)

    result, _ = await execution.upsert_execution_input(
        node_id, run_id, "data", "first", node_exec_id=named
    )
    assert result.node_exec_id == named

    with pytest.raises(ValueError, match="already has input data"):
        await execution.upsert_execution_input(
            node_id, run_id, "data", "second", node_exec_id=named
        )


async def _graph_with_two_runs(server: SpinTestServer) -> tuple[str, str, str, str]:
    user = await create_test_user()
    graph = await server.agent_server.test_create_graph(
        CreateGraph(graph=create_test_graph()), user.id
    )
    run_ids = []
    for _ in range(2):
        run = await execution.create_graph_execution(
            graph_id=graph.id,
            graph_version=graph.version,
            starting_nodes_input=[],
            inputs={},
            user_id=user.id,
        )
        run_ids.append(run.id)
    return graph.nodes[-1].id, graph.nodes[0].id, run_ids[0], run_ids[1]


async def _waiting_run(node_id: str, run_id: str, pin: str, hours_ago: int) -> str:
    """Create an INCOMPLETE run of the node holding only `pin`, added `hours_ago`."""
    created = await prisma.models.AgentNodeExecution.prisma().create(
        data={
            "agentNodeId": node_id,
            "agentGraphExecutionId": run_id,
            "executionStatus": execution.ExecutionStatus.INCOMPLETE,
            "addedTime": datetime.now(timezone.utc) - timedelta(hours=hours_ago),
            "Input": {"create": {"name": pin, "data": prisma.Json("seed")}},
        }
    )
    return created.id
