"""DB-backed: a moderation fail clears the flagged content (#15907).

Output moderation runs after the nodes COMPLETED. COMPLETED -> FAILED is not in
VALID_STATUS_TRANSITIONS, so the guarded write dropped status and cleared_*
together and the run kept showing the flagged content.
"""

import uuid
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest
import pytest_asyncio
from prisma import Json

from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.data import db
from backend.data.execution import (
    ExecutionStatus,
    create_graph_execution,
    get_graph_execution,
    get_node_execution,
    get_node_executions,
    mark_node_executions_moderated,
    update_graph_execution_stats,
    update_node_execution_status,
    upsert_execution_output,
)

from .manager import AutoModManager

MSG = "Failed due to content moderation (Moderation ID: cid-1)"
PRIOR_STATS = {"walltime": 1.5, "cost": 3}


@pytest_asyncio.fixture(loop_scope="session")
async def user_and_graph() -> AsyncIterator[tuple[str, str]]:
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    for block in (AgentInputBlock(), AgentOutputBlock()):
        await prisma.models.AgentBlock.prisma().upsert(
            where={"id": block.id},
            data={
                "create": {
                    "id": block.id,
                    "name": block.name,
                    "inputSchema": "{}",
                    "outputSchema": "{}",
                },
                "update": {},
            },
        )
    user_id, graph_id = str(uuid.uuid4()), str(uuid.uuid4())
    await prisma.models.User.prisma().create(
        data={"id": user_id, "email": f"automod-{user_id}@example.com"}
    )
    await prisma.models.AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": 1,
            "name": "automod-clear-test",
            "userId": user_id,
            "isActive": True,
        }
    )
    try:
        yield user_id, graph_id
    finally:
        await prisma.models.AgentGraphExecution.prisma().delete_many(
            where={"userId": user_id}
        )
        await prisma.models.AgentGraph.prisma().delete_many(where={"id": graph_id})
        await prisma.models.User.prisma().delete_many(where={"id": user_id})
        if owned_connection:
            await db.disconnect()


async def _run_with_node(
    user_id: str, graph_id: str, block_id: str, name: str
) -> tuple[str, str]:
    """A graph execution with one QUEUED node execution that has Input rows."""
    node = await prisma.models.AgentNode.prisma().create(
        data={
            "agentBlockId": block_id,
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
        }
    )
    execution = await create_graph_execution(
        graph_id=graph_id,
        graph_version=1,
        starting_nodes_input=[(node.id, {"name": name, "value": "flagged"})],
        inputs={},
        user_id=user_id,
    )
    [node_exec] = await get_node_executions(execution.id)
    await prisma.models.AgentNodeExecution.prisma().update(
        where={"id": node_exec.node_exec_id}, data={"stats": Json(PRIOR_STATS)}
    )
    return execution.id, node_exec.node_exec_id


async def _moderate(graph_exec_id: str, moderation_type) -> AsyncMock:
    db_client = SimpleNamespace(
        get_node_executions=get_node_executions,
        mark_node_executions_moderated=mark_node_executions_moderated,
    )
    with patch.object(AutoModManager, "_load_config"), patch(
        "backend.executor.manager.send_async_execution_update", new_callable=AsyncMock
    ) as send:
        await AutoModManager()._update_failed_nodes_for_moderation(
            db_client, graph_exec_id, moderation_type, "cid-1"  # type: ignore[arg-type]
        )
    return send


@pytest.mark.asyncio(loop_scope="session")
async def test_output_moderation_clears_a_completed_output_node(user_and_graph):
    user_id, graph_id = user_and_graph
    graph_exec_id, node_exec_id = await _run_with_node(
        user_id, graph_id, AgentOutputBlock().id, "report"
    )
    await prisma.models.AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data={"executionStatus": ExecutionStatus.COMPLETED},
    )
    await upsert_execution_output(node_exec_id, "output", "flagged")

    send = await _moderate(graph_exec_id, "output")

    row = await prisma.models.AgentNodeExecution.prisma().find_unique_or_raise(
        where={"id": node_exec_id}
    )
    assert row.executionStatus == ExecutionStatus.FAILED
    assert row.stats == {
        **PRIOR_STATS,
        "error": MSG,
        "cleared_inputs": {"name": ["report"], "value": [MSG]},
        "cleared_outputs": {"output": [MSG]},
    }
    node = await get_node_execution(node_exec_id)
    assert node is not None
    assert node.input_data["value"] == MSG
    assert dict(node.output_data) == {"output": [MSG]}
    [sent] = [call.args[0] for call in send.await_args_list]
    assert sent.status == ExecutionStatus.FAILED
    assert dict(sent.output_data) == {"output": [MSG]}

    run = await get_graph_execution(
        user_id, graph_exec_id, include_node_executions=True
    )
    assert run is not None
    assert dict(run.outputs) == {"report": [MSG]}
    assert "flagged" not in run.model_dump_json()


@pytest.mark.asyncio(loop_scope="session")
async def test_input_moderation_clears_a_queued_input_node(user_and_graph):
    user_id, graph_id = user_and_graph
    graph_exec_id, node_exec_id = await _run_with_node(
        user_id, graph_id, AgentInputBlock().id, "payload"
    )

    await _moderate(graph_exec_id, "input")

    node = await get_node_execution(node_exec_id)
    assert node is not None
    assert node.status == ExecutionStatus.FAILED
    assert node.input_data == {"name": "payload", "value": MSG}
    row = await prisma.models.AgentNodeExecution.prisma().find_unique_or_raise(
        where={"id": node_exec_id}
    )
    assert row.stats is not None
    assert (row.stats["walltime"], row.stats["cost"]) == (1.5, 3)
    assert row.stats["cleared_inputs"]["name"] == ["payload"]


@pytest.mark.asyncio(loop_scope="session")
async def test_global_map_still_refuses_completed_to_failed(user_and_graph):
    """The moderation write must not loosen VALID_STATUS_TRANSITIONS: it is what
    rejects a second terminal write to a finished node or run."""
    user_id, graph_id = user_and_graph
    graph_exec_id, node_exec_id = await _run_with_node(
        user_id, graph_id, AgentOutputBlock().id, "report"
    )
    await prisma.models.AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data={"executionStatus": ExecutionStatus.COMPLETED},
    )
    await prisma.models.AgentGraphExecution.prisma().update(
        where={"id": graph_exec_id},
        data={"executionStatus": ExecutionStatus.COMPLETED},
    )

    node = await update_node_execution_status(node_exec_id, ExecutionStatus.FAILED)
    run = await update_graph_execution_stats(graph_exec_id, ExecutionStatus.FAILED)

    assert node.status == ExecutionStatus.COMPLETED
    assert run is None


@pytest.mark.asyncio(loop_scope="session")
async def test_rows_moderation_could_not_clear_are_logged_as_errors(user_and_graph):
    user_id, graph_id = user_and_graph
    graph_exec_id, node_exec_id = await _run_with_node(
        user_id, graph_id, AgentOutputBlock().id, "report"
    )

    # The node is QUEUED, not COMPLETED, so the output-path write must skip it.
    with patch("backend.data.execution.logger") as logger:
        updated = await mark_node_executions_moderated(
            graph_exec_id,
            {node_exec_id: {"error": MSG}},
            [ExecutionStatus.COMPLETED],
        )

    assert updated == []
    logger.error.assert_called_once()
    assert node_exec_id in logger.error.call_args.args[0]
    node = await get_node_execution(node_exec_id)
    assert node is not None and node.status == ExecutionStatus.QUEUED
