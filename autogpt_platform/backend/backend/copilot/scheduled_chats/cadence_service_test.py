"""Running a scheduled chat less often, against a real database (SECRT-2879).

The scheduler is mocked; the routine rows, their ownership and the streak are
real.
"""

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import ExpertRoutine, ScheduledChatStreak, User

from backend.copilot.scheduled_chats.cadence_service import (
    CadenceNotFound,
    CadenceNotOffered,
    ScheduledChatPaused,
    get_cadence,
    set_cadence,
)
from backend.executor.scheduler import CopilotTurnJobInfo
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)

_SERVICE = "backend.copilot.scheduled_chats.cadence_service"
_ROUTINE_JOBS = "backend.api.features.experts.routine_jobs"
# The service reads the real clock; a zone without daylight saving keeps an
# hourly beat even in the week the clocks change.
_TOKYO = "Asia/Tokyo"


@pytest.fixture
def scheduler(mocker) -> MagicMock:
    client = MagicMock(
        get_execution_schedules=AsyncMock(return_value=[]),
        update_copilot_turn_cron=AsyncMock(),
        add_copilot_turn_schedule=AsyncMock(return_value=MagicMock(id="job-new")),
        delete_schedule=AsyncMock(),
    )
    mocker.patch(f"{_SERVICE}.get_scheduler_client", return_value=client)
    mocker.patch(f"{_ROUTINE_JOBS}.get_scheduler_client", return_value=client)
    return client


async def _owner() -> str:
    user_id = f"cadence-{uuid4()}"
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"{user_id}@example.com",
                "name": "Cadence",
                "timezone": _TOKYO,
            }
        )
    except UniqueViolationError:
        pass
    return user_id


async def _cleanup(user_id: str) -> None:
    try:
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("cleanup for %s failed: %s", user_id, exc)


async def _hourly_routine(user_id: str, *, enabled: bool = True) -> ExpertRoutine:
    return await ExpertRoutine.prisma().create(
        data={
            "userId": user_id,
            "title": "Inbox sweep",
            "prompt": "Sweep the inbox.",
            "crons": ["17 * * * *"],
            "asks": [],
            "scheduleIds": ["job-old"],
            "enabledAt": datetime.now(timezone.utc) if enabled else None,
        }
    )


def _every_minute_follow_up(user_id: str, **overrides) -> CopilotTurnJobInfo:
    fields = dict(
        id="job-1",
        name="copilot turn",
        next_run_time="2026-10-12T09:00:00+00:00",
        timezone=_TOKYO,
        schedule_id="job-1",
        user_id=user_id,
        message="Check Discord for new messages.",
        cron="* * * * *",
        user_timezone=_TOKYO,
    )
    fields.update(overrides)
    return CopilotTurnJobInfo(**fields)


@pytest.mark.asyncio(loop_scope="session")
async def test_an_hourly_routine_runs_daily_from_now_on(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id = await _owner()
    try:
        routine = await _hourly_routine(user_id)

        offered = await get_cadence(user_id, routine.id)
        assert offered.title == "Inbox sweep"
        assert offered.current == "Every hour"
        assert offered.timezone == _TOKYO
        assert [option.label for option in offered.options] == [
            "Every 4 hours",
            "Every day at 09:17",
            "Weekdays at 09:17",
            "Every Monday at 09:17",
        ]

        changed = await set_cadence(user_id, routine.id, "17 9 * * *")

        assert changed.current == "Every day at 09:17"
        row = await ExpertRoutine.prisma().find_unique(where={"id": routine.id})
        assert row is not None
        assert row.crons == ["17 9 * * *"] and row.scheduleIds == ["job-new"]
        # The new job is armed before the old one goes.
        calls = [name for name, _, _ in scheduler.mock_calls]
        assert calls == ["add_copilot_turn_schedule", "delete_schedule"]
        scheduler.delete_schedule.assert_awaited_once_with("job-old", user_id=user_id)
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_routine_that_keeps_one_thread_keeps_it_at_its_new_cadence(
    server: SpinTestServer, scheduler: MagicMock
):
    """A THREAD routine's first fire mints the chat every later fire goes
    to. Running it less often must not start that conversation over."""
    user_id = await _owner()
    try:
        routine = await _hourly_routine(user_id)
        await ExpertRoutine.prisma().update(
            where={"id": routine.id}, data={"sessionId": "thread-1"}
        )

        await set_cadence(user_id, routine.id, "17 9 * * *")

        row = await ExpertRoutine.prisma().find_unique(where={"id": routine.id})
        assert row is not None and row.sessionId == "thread-1"
        # Its jobs still name no chat: each fire reads the row's.
        assert (
            scheduler.add_copilot_turn_schedule.await_args.kwargs["session_id"] is None
        )
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_follow_up_is_changed_in_place(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id = await _owner()
    scheduler.get_execution_schedules.return_value = [_every_minute_follow_up(user_id)]
    try:
        offered = await get_cadence(user_id, "job-1")
        assert offered.title == "Check Discord for new messages."
        assert offered.current == "Every minute"
        assert offered.options[0].label == "Every 15 minutes"

        await set_cadence(user_id, "job-1", "0 * * * *")

        scheduler.update_copilot_turn_cron.assert_awaited_once_with(
            "job-1", user_id=user_id, cron="0 * * * *"
        )
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_follow_up_deleted_since_it_was_read_reads_as_not_found(
    server: SpinTestServer, scheduler: MagicMock
):
    """Not a 503 that says to try again: trying again finds nothing."""
    user_id = await _owner()
    scheduler.get_execution_schedules.return_value = [_every_minute_follow_up(user_id)]
    scheduler.update_copilot_turn_cron.side_effect = NotFoundError("gone")
    try:
        with pytest.raises(CadenceNotFound):
            await set_cadence(user_id, "job-1", "0 * * * *")
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_only_an_offered_cadence_is_taken(
    server: SpinTestServer, scheduler: MagicMock
):
    """Faster is not "less often", and neither is anything made up."""
    user_id = await _owner()
    try:
        routine = await _hourly_routine(user_id)

        for cron in ("* * * * *", "17 9 * * sat"):
            with pytest.raises(CadenceNotOffered):
                await set_cadence(user_id, routine.id, cron)

        scheduler.add_copilot_turn_schedule.assert_not_awaited()
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_nothing_to_change_reads_as_not_found(
    server: SpinTestServer, scheduler: MagicMock
):
    """Someone else's routine, one switched off, a routine's own job, a
    one-shot, and a key that names nothing."""
    user_id = await _owner()
    other = await _owner()
    try:
        theirs = await _hourly_routine(other)
        off = await _hourly_routine(user_id, enabled=False)
        scheduler.get_execution_schedules.return_value = [
            _every_minute_follow_up(user_id, id="routine-job", routine_id=off.id),
            _every_minute_follow_up(
                user_id, id="one-shot", cron=None, run_at=datetime.now(timezone.utc)
            ),
        ]

        for key in (theirs.id, off.id, "routine-job", "one-shot", "nothing"):
            with pytest.raises(CadenceNotFound):
                await get_cadence(user_id, key)
    finally:
        await _cleanup(user_id)
        await _cleanup(other)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_paused_chat_is_resumed_first(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id = await _owner()
    try:
        routine = await _hourly_routine(user_id)
        await ScheduledChatStreak.prisma().create(
            data={
                "userId": user_id,
                "scheduleKey": routine.id,
                "scheduleIds": ["job-old"],
                "title": "Inbox sweep",
                "failureKey": "connection:gmail",
                "failureKind": "connection",
                "provider": "gmail",
                "failures": 3,
                "wholeRunFailures": 3,
                "pausedAt": datetime.now(timezone.utc),
            }
        )

        with pytest.raises(ScheduledChatPaused):
            await set_cadence(user_id, routine.id, "17 9 * * *")
    finally:
        await _cleanup(user_id)
