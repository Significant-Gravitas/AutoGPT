"""DB-backed: the IDs of a user's runs in an org, filtered by status."""

import uuid
from datetime import datetime, timedelta, timezone

import prisma.models
import pytest

from backend.data.execution import ExecutionStatus, get_graph_execution_ids

UNFINISHED = [ExecutionStatus.QUEUED, ExecutionStatus.RUNNING]
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.asyncio(loop_scope="session")
async def test_ids_of_a_users_runs_in_one_org_by_status(server):
    org_id, other_org_id = str(uuid.uuid4()), str(uuid.uuid4())
    user_id, graph_id = await _user_with_graph()
    other_user_id, other_graph_id = await _user_with_graph()
    try:
        queued = await _run(user_id, graph_id, org_id, ExecutionStatus.QUEUED, 1)
        running = await _run(user_id, graph_id, org_id, ExecutionStatus.RUNNING, 2)
        hidden = await _run(
            user_id, graph_id, org_id, ExecutionStatus.RUNNING, 3, deleted=True
        )
        await _run(user_id, graph_id, org_id, ExecutionStatus.COMPLETED, 4)
        await _run(user_id, graph_id, other_org_id, ExecutionStatus.QUEUED, 5)
        await _run(other_user_id, other_graph_id, org_id, ExecutionStatus.QUEUED, 6)

        visible = await get_graph_execution_ids(
            user_id=user_id, organization_id=org_id, statuses=UNFINISHED
        )
        deleted = await get_graph_execution_ids(
            user_id=user_id, organization_id=org_id, statuses=UNFINISHED, deleted=True
        )

        # Newest first, as `get_graph_executions` lists them.
        assert visible == [running, queued]
        assert deleted == [hidden]
    finally:
        # Deleting a user deletes their graphs and runs with them.
        await prisma.models.User.prisma().delete_many(
            where={"id": {"in": [user_id, other_user_id]}}
        )


async def _user_with_graph() -> tuple[str, str]:
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    await prisma.models.User.prisma().create(
        data={"id": user_id, "email": f"execution-ids-{user_id}@example.com"}
    )
    await prisma.models.AgentGraph.prisma().create(
        data={"id": graph_id, "version": 1, "name": "execution-ids", "userId": user_id}
    )
    return user_id, graph_id


async def _run(
    user_id: str,
    graph_id: str,
    org_id: str,
    status: ExecutionStatus,
    minute: int,
    deleted: bool = False,
) -> str:
    run = await prisma.models.AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": graph_id,
            "agentGraphVersion": 1,
            "userId": user_id,
            "organizationId": org_id,
            "executionStatus": status,
            "isDeleted": deleted,
            "createdAt": START + timedelta(minutes=minute),
        }
    )
    return run.id
