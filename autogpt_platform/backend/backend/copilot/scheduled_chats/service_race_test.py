"""Exercise scheduled-chat pause and resume races at the storage boundaries."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fakeredis import FakeAsyncRedis
from prisma.models import ExpertRoutine, ScheduledChatStreak
from pytest_mock import MockerFixture

from backend.copilot import scheduler_write_lock
from backend.copilot.at_reset_lock_fixtures import FakeAtResetRedis
from backend.copilot.scheduled_chats import service
from backend.copilot.scheduled_chats.outcomes import RunOutcome
from backend.copilot.scheduled_chats.service import (
    ScheduledRun,
    record_scheduled_run,
    resume_scheduled_chat,
)
from backend.copilot.scheduled_chats.service_fixtures import (
    DISCORD_FAILURE,
    scheduled_run,
)
from backend.copilot.scheduled_chats.service_fixtures import (
    scheduler_fixture as scheduler_fixture,
)
from backend.copilot.scheduled_chats.service_fixtures import (
    watch_scheduler_lock_fixture as watch_scheduler_lock_fixture,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lease_expires",
    [
        pytest.param(False, id="lease-held"),
        pytest.param(True, id="lease-expired"),
    ],
)
async def test_concurrent_pause_writes_while_the_lease_is_held_or_expired(
    scheduler: MagicMock,
    mocker: MockerFixture,
    watch_scheduler_lock: Callable[[str, str], asyncio.Event],
    lease_expires: bool,
):
    """Check contention with a live lease and successor writes after expiry."""
    watchdog_seconds = 30
    lock_key = "copilot:scheduled_chat_lock:5:owner:routine-1"
    contention = watch_scheduler_lock("owner", "routine-1")
    if lease_expires:
        # Simulate a missed renewal. Redis expires the key while the first
        # writer's scheduler call is still in flight.
        mocker.patch.object(
            scheduler_write_lock, "_LOCK_RENEW_EVERY_SECONDS", 2 * watchdog_seconds
        )
    active = {"job-a", "job-b"}
    first_paused_a = asyncio.Event()
    finish_first_pause = asyncio.Event()

    async def pause(schedule_id: str, *, user_id: str) -> bool:
        if schedule_id not in active:
            return False
        active.remove(schedule_id)
        if schedule_id == "job-a":
            first_paused_a.set()
            await finish_first_pause.wait()
        return True

    async def resume(schedule_id: str, *, user_id: str) -> bool:
        active.add(schedule_id)
        return True

    scheduler.pause_schedule.side_effect = pause
    scheduler.resume_schedule.side_effect = resume
    scheduler.get_execution_schedules.side_effect = lambda **_: [
        MagicMock(id=job_id) for job_id in active
    ]

    row: ScheduledChatStreak | None = ScheduledChatStreak.model_construct(
        id="streak-1",
        userId="owner",
        scheduleKey="routine-1",
        scheduleIds=[],
        title="Morning and noon sweep",
        sessionId="session-1",
        failureKey=DISCORD_FAILURE.key,
        failureKind=DISCORD_FAILURE.kind,
        provider=DISCORD_FAILURE.provider,
        tool=DISCORD_FAILURE.tool,
        failures=2,
        wholeRunFailures=2,
        pausedAt=None,
        notifiedAt=None,
    )

    async def update_many(*, where, data) -> int:
        nonlocal row
        assert row is not None
        row = row.model_copy(update=data)
        return 1

    async def delete(*, where) -> None:
        nonlocal row
        row = None

    rows = MagicMock(
        find_unique=AsyncMock(side_effect=lambda **_: row),
        update_many=AsyncMock(side_effect=update_many),
        delete=AsyncMock(side_effect=delete),
        count=AsyncMock(return_value=1),
    )
    mocker.patch.object(ScheduledChatStreak, "prisma", return_value=rows)
    mocker.patch.object(
        ExpertRoutine,
        "prisma",
        return_value=MagicMock(
            find_unique=AsyncMock(
                return_value=MagicMock(
                    title="Morning and noon sweep",
                    scheduleIds=["job-a", "job-b", "owner-paused"],
                )
            )
        ),
    )
    mocker.patch.object(service.alerts, "raise_paused", new_callable=AsyncMock)
    mocker.patch.object(service, "resolve_alert", new_callable=AsyncMock)
    mocker.patch.object(service, "post_notice", new_callable=AsyncMock)

    failed = RunOutcome(failure=DISCORD_FAILURE, whole_run=True)

    def routine_run(job_id: str) -> ScheduledRun:
        return scheduled_run(
            "owner",
            "session-1",
            failed,
            schedule_key="routine-1",
            schedule_id=job_id,
            routine_id="routine-1",
        )

    async with FakeAsyncRedis() as redis:
        mocker.patch.object(scheduler_write_lock, "get_redis_async", return_value=redis)
        async with asyncio.timeout(watchdog_seconds), asyncio.TaskGroup() as tasks:
            first = tasks.create_task(record_scheduled_run(routine_run("job-a")))
            await first_paused_a.wait()
            if lease_expires:
                assert await redis.pexpire(lock_key, 0)
                assert not await redis.exists(lock_key)
                await record_scheduled_run(routine_run("job-b"))
                assert not first.done()
                assert row is not None and "job-b" in row.scheduleIds
            else:
                tasks.create_task(record_scheduled_run(routine_run("job-b")))
                await contention.wait()
                rows.find_unique.assert_awaited_once()
            finish_first_pause.set()

        # Following expiry, the first writer can still reach storage. The
        # expired-lease case checks the successor's write before that happens.
        if not lease_expires:
            assert await resume_scheduled_chat("owner", "routine-1") is True
            assert row is None
            assert active == {"job-a", "job-b"}


@pytest.mark.asyncio
async def test_scheduler_chat_lock_failure_leaves_the_jobs_untouched(
    scheduler: MagicMock, mocker: MockerFixture
):
    mocker.patch.object(
        scheduler_write_lock, "get_redis_async", side_effect=ConnectionError("down")
    )
    rows = MagicMock(find_unique=AsyncMock(return_value=None))
    mocker.patch.object(ScheduledChatStreak, "prisma", return_value=rows)
    run = scheduled_run(
        "owner", "session-1", RunOutcome(failure=DISCORD_FAILURE, whole_run=True)
    )

    with pytest.raises(RuntimeError, match="scheduled chat write lock"):
        await record_scheduled_run(run)
    rows.find_unique.assert_not_awaited()

    with pytest.raises(RuntimeError, match="scheduled chat write lock"):
        await resume_scheduled_chat("owner", "sched-1")

    scheduler.pause_schedule.assert_not_awaited()
    scheduler.resume_schedule.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("already_paused", [False, True])
async def test_resume_checks_which_pause_it_observed(
    scheduler: MagicMock, mocker: MockerFixture, already_paused: bool
):
    """A waiting Resume can act on its first pause but leaves a later one alone."""
    mocker.patch.object(
        scheduler_write_lock, "get_redis_async", return_value=FakeAtResetRedis()
    )
    paused_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = ScheduledChatStreak.model_construct(
        id="streak-1",
        scheduleIds=["sched-1"],
        sessionId=None,
        pausedAt=paused_at,
    )
    observed = row.model_copy(
        update={
            "pausedAt": paused_at - timedelta(seconds=1) if already_paused else None
        }
    )
    rows = MagicMock(
        find_unique=AsyncMock(side_effect=[observed, row]), delete=AsyncMock()
    )
    mocker.patch.object(ScheduledChatStreak, "prisma", return_value=rows)
    resolved = mocker.patch.object(service, "resolve_alert", new_callable=AsyncMock)

    assert await resume_scheduled_chat("owner", "sched-1") is True

    if already_paused:
        scheduler.resume_schedule.assert_not_awaited()
        rows.delete.assert_not_awaited()
        resolved.assert_not_awaited()
    else:
        scheduler.resume_schedule.assert_awaited_once_with("sched-1", user_id="owner")
        rows.delete.assert_awaited_once_with(where={"id": "streak-1"})
        resolved.assert_awaited_once_with("owner", "sched-1")
