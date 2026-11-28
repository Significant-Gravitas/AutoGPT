"""DB-backed: a moderation fail clears the flagged content (#15907).

Output moderation runs after the nodes COMPLETED. COMPLETED -> FAILED is not in
VALID_STATUS_TRANSITIONS, so the guarded write dropped status and cleared_*
together and the run kept showing the flagged content.
"""

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.models
import pytest
import pytest_asyncio
from prisma import Json

from backend.blocks._base import Block
from backend.blocks.human_in_the_loop import HumanInTheLoopBlock
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
    user_id, graph_id = str(uuid.uuid4()), str(uuid.uuid4())
    try:
        for block in (AgentInputBlock(), AgentOutputBlock(), HumanInTheLoopBlock()):
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
        yield user_id, graph_id
    finally:
        try:
            await prisma.models.AgentGraphExecution.prisma().delete_many(
                where={"userId": user_id}
            )
            await prisma.models.AgentGraph.prisma().delete_many(where={"id": graph_id})
            await prisma.models.User.prisma().delete_many(where={"id": user_id})
        finally:
            if owned_connection:
                await db.disconnect()


async def _run_with_nodes(
    user_id: str, graph_id: str, block_id: str, *names: str
) -> tuple[str, list[str]]:
    """A graph execution with one QUEUED node execution per name, each with
    Input rows. Returns the run's ID and the node executions' IDs, in order."""
    nodes = [
        await prisma.models.AgentNode.prisma().create(
            data={
                "agentBlockId": block_id,
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
            }
        )
        for _ in names
    ]
    run = await create_graph_execution(
        graph_id=graph_id,
        graph_version=1,
        starting_nodes_input=[
            (node.id, {"name": name, "value": "flagged"})
            for node, name in zip(nodes, names)
        ],
        inputs={},
        user_id=user_id,
    )
    by_node = {
        node_exec.node_id: node_exec.node_exec_id
        for node_exec in await get_node_executions(run.id)
    }
    node_exec_ids = [by_node[node.id] for node in nodes]
    for node_exec_id in node_exec_ids:
        await prisma.models.AgentNodeExecution.prisma().update(
            where={"id": node_exec_id}, data={"stats": Json(PRIOR_STATS)}
        )
    return run.id, node_exec_ids


async def _moderate(graph_exec_id: str, moderation_type) -> AsyncMock:
    db_client = MagicMock(
        get_node_executions=get_node_executions,
        mark_node_executions_moderated=mark_node_executions_moderated,
    )
    with patch.object(AutoModManager, "_load_config"), patch(
        "backend.executor.manager.send_async_execution_update", new_callable=AsyncMock
    ) as send:
        await AutoModManager().update_failed_nodes_for_moderation(
            db_client, graph_exec_id, moderation_type, "cid-1"
        )
    return send


@pytest.mark.asyncio(loop_scope="session")
async def test_output_moderation_clears_a_completed_output_node(user_and_graph):
    user_id, graph_id = user_and_graph
    graph_exec_id, [node_exec_id] = await _run_with_nodes(
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
    assert node.input_data["name"] == "report"
    assert node.input_data["value"] == MSG
    assert dict(node.output_data) == {"output": [MSG]}
    [sent] = [call.args[0] for call in send.await_args_list]
    assert sent.status == ExecutionStatus.FAILED
    assert dict(sent.output_data) == {"output": [MSG]}

    run = await get_graph_execution(
        user_id, graph_exec_id, include_node_executions=True
    )
    assert run is not None
    # Run outputs are keyed on the declared Agent Output name.
    assert dict(run.outputs) == {"report": [MSG]}
    assert "flagged" not in run.model_dump_json()


@pytest.mark.asyncio(loop_scope="session")
async def test_old_stats_the_reader_cant_validate_dont_unmask_the_node(
    user_and_graph,
):
    """from_db drops stats it can't validate, moderation fields included, so
    stats from an older schema must not be kept next to the mask. Dropping the
    rest of them is logged."""
    user_id, graph_id = user_and_graph
    graph_exec_id, [node_exec_id] = await _run_with_nodes(
        user_id, graph_id, AgentOutputBlock().id, "report"
    )
    await prisma.models.AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data={
            "executionStatus": ExecutionStatus.COMPLETED,
            "stats": Json({**PRIOR_STATS, "provider_cost_type": "retired_type"}),
        },
    )
    await upsert_execution_output(node_exec_id, "output", "flagged")

    with patch("backend.data.execution.logger") as logger:
        await _moderate(graph_exec_id, "output")

    node = await get_node_execution(node_exec_id)
    assert node is not None and node.status == ExecutionStatus.FAILED
    assert dict(node.output_data) == {"output": [MSG]}
    assert node.input_data["value"] == MSG
    logger.warning.assert_called_once()
    assert node_exec_id in logger.warning.call_args.args[0]
    assert "provider_cost_type" in logger.warning.call_args.args[0]


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "block, name_shown",
    [
        (AgentInputBlock(), "payload"),
        (HumanInTheLoopBlock(), MSG),
    ],
    ids=[
        "an Agent Input keeps its declared name",
        "the name input of any other block is content",
    ],
)
async def test_input_moderation_clears_a_queued_node(
    user_and_graph, block: Block, name_shown: str
):
    user_id, graph_id = user_and_graph
    graph_exec_id, [node_exec_id] = await _run_with_nodes(
        user_id, graph_id, block.id, "payload"
    )

    await _moderate(graph_exec_id, "input")

    node = await get_node_execution(node_exec_id)
    assert node is not None
    assert node.status == ExecutionStatus.FAILED
    assert node.input_data == {"name": name_shown, "value": MSG}
    row = await prisma.models.AgentNodeExecution.prisma().find_unique_or_raise(
        where={"id": node_exec_id}
    )
    assert row.stats is not None
    assert (row.stats["walltime"], row.stats["cost"]) == (1.5, 3)
    assert row.stats["cleared_inputs"]["name"] == [name_shown]


@pytest.mark.asyncio(loop_scope="session")
async def test_outputs_written_after_moderation_stay_hidden(user_and_graph):
    """A node failed by input moderation while RUNNING has no outputs yet. Its
    mask must still hide the ones it writes later: readers take an empty mask
    for no mask."""
    user_id, graph_id = user_and_graph
    graph_exec_id, [node_exec_id] = await _run_with_nodes(
        user_id, graph_id, AgentOutputBlock().id, "report"
    )
    await prisma.models.AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data={"executionStatus": ExecutionStatus.RUNNING},
    )

    await _moderate(graph_exec_id, "input")
    await upsert_execution_output(node_exec_id, "output", "flagged")

    node = await get_node_execution(node_exec_id)
    assert node is not None and node.status == ExecutionStatus.FAILED
    assert dict(node.output_data) == {"error": [MSG]}
    run = await get_graph_execution(
        user_id, graph_exec_id, include_node_executions=True
    )
    assert run is not None
    assert dict(run.outputs) == {"report": [MSG]}
    assert "flagged" not in run.model_dump_json()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_run_with_more_nodes_than_writes_in_flight_is_masked_whole(
    user_and_graph, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("backend.data.execution.MODERATION_WRITES_IN_FLIGHT", 2)
    user_id, graph_id = user_and_graph
    graph_exec_id, node_exec_ids = await _run_with_nodes(
        user_id, graph_id, AgentOutputBlock().id, "a", "b", "c", "d", "e"
    )
    for node_exec_id in node_exec_ids:
        await prisma.models.AgentNodeExecution.prisma().update(
            where={"id": node_exec_id},
            data={"executionStatus": ExecutionStatus.COMPLETED},
        )
        await upsert_execution_output(node_exec_id, "output", "flagged")

    send = await _moderate(graph_exec_id, "output")

    sent = [call.args[0] for call in send.await_args_list]
    assert sorted(node.node_exec_id for node in sent) == sorted(node_exec_ids)
    for node_exec_id in node_exec_ids:
        node = await get_node_execution(node_exec_id)
        assert node is not None and node.status == ExecutionStatus.FAILED
        assert dict(node.output_data) == {"output": [MSG]}


@pytest.mark.asyncio(loop_scope="session")
async def test_a_replayed_moderation_returns_the_rows_it_masked(user_and_graph):
    """The RPC client retries, so the same call can arrive twice. The replay
    finds its rows FAILED already and must still return them masked, for the
    live update, but never a row that failed for another reason."""
    user_id, graph_id = user_and_graph
    graph_exec_id, [moderated_id, failed_id] = await _run_with_nodes(
        user_id, graph_id, AgentOutputBlock().id, "report", "summary"
    )
    await prisma.models.AgentNodeExecution.prisma().update(
        where={"id": moderated_id},
        data={"executionStatus": ExecutionStatus.COMPLETED},
    )
    await upsert_execution_output(moderated_id, "output", "flagged")
    await prisma.models.AgentNodeExecution.prisma().update(
        where={"id": failed_id},
        data={
            "executionStatus": ExecutionStatus.FAILED,
            "stats": Json({"error": "the block raised"}),
        },
    )
    patches = {
        node_exec_id: {
            "error": MSG,
            "cleared_inputs": {"name": ["report"], "value": [MSG]},
            "cleared_outputs": {"output": [MSG]},
        }
        for node_exec_id in (moderated_id, failed_id)
    }

    first = await mark_node_executions_moderated(
        graph_exec_id, patches, [ExecutionStatus.COMPLETED]
    )
    replay = await mark_node_executions_moderated(
        graph_exec_id, patches, [ExecutionStatus.COMPLETED]
    )

    assert [node.node_exec_id for node in first] == [moderated_id]
    assert [node.node_exec_id for node in replay] == [moderated_id]
    assert replay[0].status == ExecutionStatus.FAILED
    assert dict(replay[0].output_data) == {"output": [MSG]}
    row = await prisma.models.AgentNodeExecution.prisma().find_unique_or_raise(
        where={"id": failed_id}
    )
    assert row.stats == {"error": "the block raised"}


@pytest.mark.asyncio(loop_scope="session")
async def test_global_map_still_refuses_completed_to_failed(user_and_graph):
    """The moderation write must not loosen VALID_STATUS_TRANSITIONS: it is what
    rejects a second terminal write to a finished node or run."""
    user_id, graph_id = user_and_graph
    graph_exec_id, [node_exec_id] = await _run_with_nodes(
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
    graph_exec_id, [node_exec_id] = await _run_with_nodes(
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


@pytest.mark.asyncio(loop_scope="session")
async def test_moderation_never_clears_another_runs_node(user_and_graph):
    user_id, graph_id = user_and_graph
    graph_exec_id, [_] = await _run_with_nodes(
        user_id, graph_id, AgentOutputBlock().id, "report"
    )
    _, [other_node_exec_id] = await _run_with_nodes(
        user_id, graph_id, AgentOutputBlock().id, "report"
    )
    await prisma.models.AgentNodeExecution.prisma().update(
        where={"id": other_node_exec_id},
        data={"executionStatus": ExecutionStatus.COMPLETED},
    )

    updated = await mark_node_executions_moderated(
        graph_exec_id, {other_node_exec_id: {"error": MSG}}, [ExecutionStatus.COMPLETED]
    )

    assert updated == []
    row = await prisma.models.AgentNodeExecution.prisma().find_unique_or_raise(
        where={"id": other_node_exec_id}
    )
    assert row.executionStatus == ExecutionStatus.COMPLETED
    assert row.stats == PRIOR_STATS
