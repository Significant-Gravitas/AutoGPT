"""DB-backed: the IDs of a run's child runs that have not finished."""

import uuid

import prisma.models
import pytest

from backend.data.execution import (
    ExecutionStatus,
    get_unfinished_child_graph_execution_ids,
)


@pytest.mark.asyncio(loop_scope="session")
async def test_only_a_runs_unfinished_live_children_are_listed(server):
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    await prisma.models.User.prisma().create(
        data={"id": user_id, "email": f"child-ids-{user_id}@example.com"}
    )
    await prisma.models.AgentGraph.prisma().create(
        data={"id": graph_id, "version": 1, "name": "child-ids", "userId": user_id}
    )
    try:
        parent = await _run(user_id, graph_id, ExecutionStatus.RUNNING)
        other_parent = await _run(user_id, graph_id, ExecutionStatus.RUNNING)
        unfinished = [
            await _run(user_id, graph_id, status, parent=parent)
            for status in (
                ExecutionStatus.QUEUED,
                ExecutionStatus.INCOMPLETE,
                ExecutionStatus.RUNNING,
                ExecutionStatus.REVIEW,
            )
        ]
        for status in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.TERMINATED,
            ExecutionStatus.FAILED,
        ):
            await _run(user_id, graph_id, status, parent=parent)
        await _run(
            user_id, graph_id, ExecutionStatus.QUEUED, parent=parent, deleted=True
        )
        await _run(user_id, graph_id, ExecutionStatus.QUEUED, parent=other_parent)

        child_ids = await get_unfinished_child_graph_execution_ids(parent)

        assert sorted(child_ids) == sorted(unfinished)
        # Running children are stopped first.
        assert child_ids[0] == unfinished[2]
    finally:
        # Deleting a user deletes their graphs and runs with them.
        await prisma.models.User.prisma().delete_many(where={"id": user_id})


async def _run(
    user_id: str,
    graph_id: str,
    status: ExecutionStatus,
    parent: str | None = None,
    deleted: bool = False,
) -> str:
    run = await prisma.models.AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "userId": user_id,
            "executionStatus": status,
            "isDeleted": deleted,
            **({"parentGraphExecutionId": parent} if parent else {}),
        }
    )
    return run.id
