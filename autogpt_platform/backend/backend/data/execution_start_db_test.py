"""The executor's start write, against the real database (#17178).

Only a run that has not started becomes RUNNING, and the check is part of the
write. A stop that wrote TERMINATED after the executor read the run as QUEUED
must not be undone by the start that follows. A mocked Prisma client takes an
unconditional update as readily as a conditional one, so this runs on Postgres.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Coroutine
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from prisma import Prisma
from prisma.models import AgentGraph, AgentGraphExecution, User

from backend.data import db
from backend.data.execution import (
    ExecutionStatus,
    create_graph_execution,
    update_graph_execution_start_time,
)
from backend.util.test import SpinTestServer

STARTED_BEFORE = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
WAIT_SECONDS = 30


@pytest.fixture
async def owner(server: SpinTestServer) -> AsyncIterator[tuple[str, str]]:
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"start-write-{user_id}@example.com",
            "name": "Start Write Test",
        }
    )
    await AgentGraph.prisma().create(
        data={
            "id": graph_id,
            "version": 1,
            "name": "start-write-test",
            "description": "start-write-test",
            "userId": user_id,
            "isActive": True,
        }
    )
    yield user_id, graph_id
    await AgentGraphExecution.prisma().delete_many(where={"userId": user_id})
    await AgentGraph.prisma().delete_many(where={"id": graph_id})
    await User.prisma().delete_many(where={"id": user_id})


async def _run_in(
    owner: tuple[str, str],
    status: ExecutionStatus,
    started_at: datetime | None = None,
) -> str:
    user_id, graph_id = owner
    run = await create_graph_execution(
        graph_id=graph_id,
        graph_version=1,
        starting_nodes_input=[],
        inputs={},
        user_id=user_id,
    )
    # Written directly: the point is to find a run in each status, however
    # the transition table would have taken it there.
    await AgentGraphExecution.prisma().update(
        where={"id": run.id},
        data={
            "executionStatus": status,
            **({"startedAt": started_at} if started_at else {}),
        },
    )
    return run.id


async def _stored(run_id: str) -> AgentGraphExecution:
    return await AgentGraphExecution.prisma().find_unique_or_raise(where={"id": run_id})


@pytest.mark.parametrize("status", [ExecutionStatus.QUEUED, ExecutionStatus.INCOMPLETE])
async def test_a_run_waiting_to_start_becomes_running(
    owner: tuple[str, str], status: ExecutionStatus
):
    run_id = await _run_in(owner, status)

    started = await update_graph_execution_start_time(run_id)

    assert started is not None
    assert started.id == run_id
    assert started.status == ExecutionStatus.RUNNING
    assert started.started_at is not None
    stored = await _stored(run_id)
    assert stored.executionStatus == ExecutionStatus.RUNNING
    assert stored.startedAt is not None


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.TERMINATED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.REVIEW,
        ExecutionStatus.RUNNING,
    ],
)
async def test_a_run_not_waiting_to_start_is_left_as_it_is(
    owner: tuple[str, str], status: ExecutionStatus
):
    run_id = await _run_in(owner, status, started_at=STARTED_BEFORE)

    assert await update_graph_execution_start_time(run_id) is None

    stored = await _stored(run_id)
    assert (stored.executionStatus, stored.startedAt) == (status, STARTED_BEFORE)


async def test_a_run_that_does_not_exist_is_not_started(server: SpinTestServer):
    assert await update_graph_execution_start_time(str(uuid.uuid4())) is None


async def test_a_start_that_waits_behind_a_stop_does_not_undo_it(
    owner: tuple[str, str],
):
    """The bad order is forced, not hoped for: the stop's TERMINATED write is held
    uncommitted until the database reports the start's statement waiting on the
    run's row lock. The start has then looked at the run while it still read
    QUEUED, so only a check that is part of the write, made again once the stop
    commits, leaves the run TERMINATED."""
    run_id = await _run_in(owner, ExecutionStatus.QUEUED)
    # The stop runs on a client of its own, i.e. its own engine and connection,
    # as another API pod's would: on one client the engine runs the two
    # statements one after the other.
    stopper = Prisma(auto_register=False, datasource={"url": db.DATABASE_URL})
    await stopper.connect()
    start: asyncio.Task[Any] | None = None
    try:
        # Longer than the wait inside it, so a test that fails says no race
        # was forced rather than that this transaction expired.
        async with stopper.tx(timeout=timedelta(seconds=WAIT_SECONDS * 2)) as stop:
            await AgentGraphExecution.prisma(stop).update_many(
                where={"id": run_id},
                data={"executionStatus": ExecutionStatus.TERMINATED},
            )
            start = asyncio.create_task(update_graph_execution_start_time(run_id))
            await _or_fail(
                _until_a_statement_is_blocked_by(stop),
                "the start never waited behind the stop's write",
            )
        started = await asyncio.wait_for(start, timeout=WAIT_SECONDS)
    finally:
        if start is not None:
            await asyncio.gather(start, return_exceptions=True)
        await stopper.disconnect()

    assert started is None
    stored = await _stored(run_id)
    assert (stored.executionStatus, stored.startedAt) == (
        ExecutionStatus.TERMINATED,
        None,
    )


async def _or_fail(waiting: Coroutine[Any, Any, None], reason: str) -> None:
    try:
        await asyncio.wait_for(waiting, timeout=WAIT_SECONDS)
    except TimeoutError:
        pytest.fail(f"{reason} within {WAIT_SECONDS} s, so no race was forced")


async def _until_a_statement_is_blocked_by(stop: Prisma) -> None:
    """Until the database reports a backend waiting on a lock the stop's own
    transaction holds, not just any statement waiting on some lock."""
    while True:
        # Inside a transaction, pg_stat_activity is frozen at first access
        # until the snapshot is cleared, so clear it before every look.
        await stop.execute_raw("SELECT pg_stat_clear_snapshot()")
        rows = await stop.query_raw(
            "SELECT count(*)::int AS waiting FROM pg_stat_activity "
            "WHERE datname = current_database() "
            "AND pg_backend_pid() = ANY(pg_blocking_pids(pid))"
        )
        if rows and rows[0]["waiting"] >= 1:
            return
        await asyncio.sleep(0.05)
