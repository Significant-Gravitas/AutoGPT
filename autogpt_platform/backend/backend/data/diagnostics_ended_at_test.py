"""Failure/completion time in admin diagnostics comes from endedAt, not updatedAt.

DB-backed: a FAILED run that ended 30h ago but was written again just now
(e.g. shared, which bumps @updatedAt) must not count as a recent failure.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from prisma.enums import AgentExecutionStatus
from prisma.models import AgentGraph, AgentGraphExecution, User
from prisma.types import AgentGraphExecutionCreateInput

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
    data: AgentGraphExecutionCreateInput = {
        "agentGraphId": graph_id,
        "agentGraphVersion": 1,
        "userId": user_id,
        "executionStatus": status,
        "endedAt": ended_at,
    }
    if created_at:
        data["createdAt"] = created_at
    return await AgentGraphExecution.prisma().create(data=data)


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

    recent = await _execution(graph, AgentExecutionStatus.FAILED, now)
    # Written last, so it has the newest updatedAt: the same state as an old
    # failure that was just shared.
    old = await _execution(
        graph, AgentExecutionStatus.FAILED, now - timedelta(hours=30)
    )

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

    listed_48h = await diag.get_failed_executions_details(limit=1000, hours=48)
    order = [e.execution_id for e in listed_48h]
    assert old.id in order
    # Newest failure first, even though the old one was written last.
    assert order.index(recent.id) < order.index(old.id)
    assert old.endedAt is not None
    assert listed_48h[order.index(old.id)].failed_at == old.endedAt


@pytest.mark.asyncio(loop_scope="session")
async def test_old_completion_with_fresh_updated_at_is_not_recent(graph):
    now = datetime.now(timezone.utc)
    before_summary = await _summary()

    await _execution(graph, AgentExecutionStatus.COMPLETED, now - timedelta(hours=30))
    await _execution(graph, AgentExecutionStatus.COMPLETED, now)

    summary = await _summary()
    assert summary.completed_1h - before_summary.completed_1h == 1
    assert summary.completed_24h - before_summary.completed_24h == 1


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
        async with _restoring_other_users_runs(graph[0]):
            assert await diag.stop_all_long_running_executions("admin") >= 1

    await _assert_failed_with_ended_at(stuck.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_cleanup_all_stuck_queued_sets_ended_at(graph):
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    stuck = await _execution(graph, AgentExecutionStatus.QUEUED, None, created_at=old)

    async with _restoring_other_users_runs(graph[0]):
        assert await diag.cleanup_all_stuck_queued_executions("admin") >= 1

    await _assert_failed_with_ended_at(stuck.id)


@asynccontextmanager
async def _restoring_other_users_runs(user_id: str) -> AsyncIterator[None]:
    # The two cleanups above change every matching run in the database, so put
    # back any other test's RUNNING or QUEUED run they mark FAILED.
    others = await AgentGraphExecution.prisma().find_many(
        where={
            "userId": {"not": user_id},
            "executionStatus": {
                "in": [AgentExecutionStatus.RUNNING, AgentExecutionStatus.QUEUED]
            },
        }
    )
    try:
        yield
    finally:
        for run in others:
            await AgentGraphExecution.prisma().update_many(
                where={"id": run.id, "executionStatus": AgentExecutionStatus.FAILED},
                data={
                    "executionStatus": run.executionStatus,
                    "updatedAt": run.updatedAt,
                    "endedAt": run.endedAt,
                },
            )
