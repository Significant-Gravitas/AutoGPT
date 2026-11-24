"""DB-backed: the stop path's status-only reads and write, and the narrow read
that scores a finished run."""

import uuid

import prisma.models
import pytest
from prisma import Json

from backend.data.execution import (
    ExecutionStatus,
    get_graph_execution_owners,
    get_graph_execution_status,
    update_graph_execution_status,
)
from backend.notifications.scoring import compute_score, score_completed_run


@pytest.mark.asyncio(loop_scope="session")
async def test_a_runs_status_and_owner_are_read_from_the_visible_runs(server):
    user_id, graph_id = await _user_with_graph()
    other_user_id, other_graph_id = await _user_with_graph()
    try:
        running = await _run(user_id, graph_id, ExecutionStatus.RUNNING)
        hidden = await _run(user_id, graph_id, ExecutionStatus.QUEUED, deleted=True)
        others = await _run(other_user_id, other_graph_id, ExecutionStatus.QUEUED)

        assert await get_graph_execution_status(user_id, running) == (
            ExecutionStatus.RUNNING
        )
        assert await get_graph_execution_status(user_id, others) is None
        assert await get_graph_execution_status(user_id, hidden) is None

        unknown = str(uuid.uuid4())
        assert await get_graph_execution_owners([running, others, hidden, unknown]) == {
            running: user_id,
            others: other_user_id,
        }
        assert await get_graph_execution_owners([]) == {}
    finally:
        await _delete_users(user_id, other_user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_the_status_write_applies_only_from_a_status_it_may_follow(server):
    user_id, graph_id = await _user_with_graph()
    try:
        queued = await _run(user_id, graph_id, ExecutionStatus.QUEUED)
        completed = await _run(user_id, graph_id, ExecutionStatus.COMPLETED)

        assert await update_graph_execution_status(queued, ExecutionStatus.TERMINATED)
        assert not await update_graph_execution_status(
            completed, ExecutionStatus.TERMINATED
        )

        stopped = await prisma.models.AgentGraphExecution.prisma().find_unique(
            where={"id": queued}
        )
        finished = await prisma.models.AgentGraphExecution.prisma().find_unique(
            where={"id": completed}
        )
        assert stopped and stopped.executionStatus == ExecutionStatus.TERMINATED
        assert stopped.endedAt is not None
        assert finished and finished.executionStatus == ExecutionStatus.COMPLETED
    finally:
        await _delete_users(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_finished_run_is_scored_from_its_status_and_stats(server):
    user_id, graph_id = await _user_with_graph()
    try:
        run_id = await _run(
            user_id,
            graph_id,
            ExecutionStatus.TERMINATED,
            stats={"cost": 5, "node_error_count": 1, "activity_status": "sent it"},
        )

        score = await score_completed_run(run_id)

        assert score == compute_score(
            status=ExecutionStatus.TERMINATED,
            cost_cents=5,
            node_error_count=1,
            has_activity=True,
            first_success=False,
            cost_baseline=None,
        )
        row = await prisma.models.AgentGraphExecution.prisma().find_unique(
            where={"id": run_id}
        )
        assert row and row.interestingness == score
    finally:
        await _delete_users(user_id)


async def _user_with_graph() -> tuple[str, str]:
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    await prisma.models.User.prisma().create(
        data={"id": user_id, "email": f"execution-status-{user_id}@example.com"}
    )
    await prisma.models.AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": 1,
            "name": "execution-status",
            "userId": user_id,
        }
    )
    return user_id, graph_id


async def _run(
    user_id: str,
    graph_id: str,
    status: ExecutionStatus,
    deleted: bool = False,
    stats: dict | None = None,
) -> str:
    run = await prisma.models.AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "userId": user_id,
            "executionStatus": status,
            "isDeleted": deleted,
            **({"stats": Json(stats)} if stats else {}),
        }
    )
    return run.id


async def _delete_users(*user_ids: str) -> None:
    # Deleting a user deletes their graphs and runs with them.
    await prisma.models.User.prisma().delete_many(where={"id": {"in": list(user_ids)}})
