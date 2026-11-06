"""Failure/completion time in admin diagnostics comes from endedAt, not updatedAt.

DB-backed: a FAILED run that ended 30h ago but was written again just now
(e.g. shared, which bumps @updatedAt) must not count as a recent failure.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from prisma.enums import AgentExecutionStatus
from prisma.models import AgentGraph, AgentGraphExecution, User

from backend.data import db
from backend.data import diagnostics as diag


@pytest_asyncio.fixture(loop_scope="session")
async def graph() -> AsyncIterator[tuple[str, str]]:
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"diag-ended-at-{user_id}@example.com"}
    )
    await AgentGraph.prisma().create(
        data={"id": graph_id, "version": 1, "userId": user_id, "name": "diag"}
    )
    try:
        yield user_id, graph_id
    finally:
        await AgentGraphExecution.prisma().delete_many(where={"userId": user_id})
        await AgentGraph.prisma().delete_many(where={"id": graph_id})
        await User.prisma().delete(where={"id": user_id})
        if owned_connection:
            await db.disconnect()


async def _execution(
    graph: tuple[str, str],
    status: AgentExecutionStatus,
    ended_at: datetime | None,
    created_at: datetime | None = None,
) -> AgentGraphExecution:
    user_id, graph_id = graph
    data: dict = {
        "agentGraphId": graph_id,
        "agentGraphVersion": 1,
        "userId": user_id,
        "executionStatus": status,
        "endedAt": ended_at,
    }
    if created_at:
        data["createdAt"] = created_at
    return await AgentGraphExecution.prisma().create(data=data)  # type: ignore[arg-type]


async def _summary() -> diag.ExecutionDiagnosticsSummary:
    with (
        patch.object(diag, "get_rabbitmq_queue_depth", return_value=0),
        patch.object(diag, "get_rabbitmq_cancel_queue_depth", return_value=0),
    ):
        return await diag.get_execution_diagnostics()


@pytest.mark.asyncio(loop_scope="session")
async def test_old_failure_with_fresh_updated_at_is_not_recent(graph):
    now = datetime.now(timezone.utc)
    before_summary = await _summary()
    before_count = await diag.get_failed_executions_count(hours=24)

    # updatedAt defaults to now: same state as an old failure that was just shared.
    old = await _execution(
        graph, AgentExecutionStatus.FAILED, now - timedelta(hours=30)
    )
    recent = await _execution(graph, AgentExecutionStatus.FAILED, now)

    summary = await _summary()
    assert summary.failed_count_1h - before_summary.failed_count_1h == 1
    assert summary.failed_count_24h - before_summary.failed_count_24h == 1
    assert await diag.get_failed_executions_count(hours=24) - before_count == 1

    listed = {
        e.execution_id: e
        for e in await diag.get_failed_executions_details(limit=1000, hours=24)
    }
    assert old.id not in listed
    assert recent.id in listed

    listed_48h = {
        e.execution_id: e
        for e in await diag.get_failed_executions_details(limit=1000, hours=48)
    }
    assert old.id in listed_48h
    assert old.endedAt is not None
    assert listed_48h[old.id].failed_at == old.endedAt


async def _assert_failed_with_ended_at(execution_id: str) -> None:
    row = await AgentGraphExecution.prisma().find_unique(where={"id": execution_id})
    assert row is not None
    assert row.executionStatus == AgentExecutionStatus.FAILED
    assert row.endedAt is not None
    listed = await diag.get_failed_executions_details(limit=1000, hours=1)
    assert execution_id in {e.execution_id for e in listed}


@pytest.mark.asyncio(loop_scope="session")
async def test_bulk_cleanup_sets_ended_at(graph):
    # Used by both the single and the all-orphaned cleanup routes.
    old = datetime.now(timezone.utc) - timedelta(hours=30)
    stuck = await _execution(graph, AgentExecutionStatus.RUNNING, None, created_at=old)

    assert await diag.cleanup_orphaned_executions_bulk([stuck.id], "admin") == 1

    await _assert_failed_with_ended_at(stuck.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_stop_all_long_running_sets_ended_at(graph):
    stuck = await _execution(graph, AgentExecutionStatus.RUNNING, None)
    await AgentGraphExecution.prisma().update(
        where={"id": stuck.id},
        data={"startedAt": datetime.now(timezone.utc) - timedelta(hours=30)},
    )
    queue = MagicMock(publish_message=AsyncMock())

    with patch.object(
        diag, "get_async_execution_queue", new_callable=AsyncMock, return_value=queue
    ):
        assert await diag.stop_all_long_running_executions("admin") >= 1

    await _assert_failed_with_ended_at(stuck.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_cleanup_all_stuck_queued_sets_ended_at(graph):
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    stuck = await _execution(graph, AgentExecutionStatus.QUEUED, None, created_at=old)

    assert await diag.cleanup_all_stuck_queued_executions("admin") >= 1

    await _assert_failed_with_ended_at(stuck.id)
