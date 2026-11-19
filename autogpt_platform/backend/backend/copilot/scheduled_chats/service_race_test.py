"""Exercise scheduled-chat pause and resume races at the storage boundaries."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
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


@pytest.mark.asyncio
async def test_concurrent_failures_resume_every_job_they_paused(
    scheduler: MagicMock, mocker: MockerFixture
):
    """Resume restores both fire times and leaves an owner-paused job alone."""
    watchdog_seconds = 30
    # The outer watchdog bounds the race. Lock expiry is tested separately.
    mocker.patch.object(
        scheduler_write_lock, "_LOCK_WAIT_SECONDS", 2 * watchdog_seconds
    )
    redis = FakeAtResetRedis()
    mocker.patch.object(scheduler_write_lock, "get_redis_async", return_value=redis)
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

    async with asyncio.timeout(watchdog_seconds), asyncio.TaskGroup() as tasks:
        tasks.create_task(record_scheduled_run(routine_run("job-a")))
        await first_paused_a.wait()
        tasks.create_task(record_scheduled_run(routine_run("job-b")))
        await redis.lock_contended.wait()
        rows.find_unique.assert_awaited_once()
        finish_first_pause.set()

    assert await resume_scheduled_chat("owner", "routine-1") is True
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
