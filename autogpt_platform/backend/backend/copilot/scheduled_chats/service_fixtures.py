"""Scheduler and inputs shared by the scheduled-chat service tests."""

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture
from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.copilot import scheduler_write_lock
from backend.copilot.scheduled_chats import jobs
from backend.copilot.scheduled_chats.outcomes import RunFailure, RunOutcome
from backend.copilot.scheduled_chats.service import ScheduledRun

DISCORD_FAILURE = RunFailure(
    key="connection:discord", kind="connection", provider="discord", tool="run_block"
)


@pytest.fixture(name="scheduler")
def scheduler_fixture(mocker: MockerFixture) -> MagicMock:
    client = MagicMock(
        pause_schedule=AsyncMock(return_value=True),
        resume_schedule=AsyncMock(return_value=True),
        # Paused jobs drop out of this list, so empty means none fires.
        get_execution_schedules=AsyncMock(return_value=[]),
    )
    mocker.patch.object(jobs, "get_scheduler_client", return_value=client)
    return client


@pytest.fixture(name="watch_scheduler_lock")
def watch_scheduler_lock_fixture(
    mocker: MockerFixture,
) -> Callable[[str, str], asyncio.Event]:
    contention: dict[str, asyncio.Event] = {}
    do_acquire = AsyncRedisLock.do_acquire

    async def observe_attempt(lock: AsyncRedisLock, token: str | bytes) -> bool:
        acquired = await do_acquire(lock, token)
        if not acquired and isinstance(lock.name, str):
            if (event := contention.get(lock.name)) is not None:
                event.set()
        return acquired

    mocker.patch.object(AsyncRedisLock, "do_acquire", new=observe_attempt)
    mocker.patch.object(scheduler_write_lock, "_LOCK_WAIT_SECONDS", 60)

    def watch(user_id: str, schedule_key: str) -> asyncio.Event:
        key = f"copilot:scheduled_chat_lock:{len(user_id)}:{user_id}:{schedule_key}"
        event = asyncio.Event()
        contention[key] = event
        return event

    return watch


def scheduled_run(
    user_id: str,
    session_id: str,
    outcome: RunOutcome,
    *,
    schedule_key: str = "sched-1",
    schedule_id: str | None = "sched-1",
    routine_id: str | None = None,
    title: str = "Discord monitor",
) -> ScheduledRun:
    return ScheduledRun(
        user_id=user_id,
        schedule_key=schedule_key,
        schedule_id=schedule_id,
        routine_id=routine_id,
        session_id=session_id,
        title=title,
        outcome=outcome,
    )
