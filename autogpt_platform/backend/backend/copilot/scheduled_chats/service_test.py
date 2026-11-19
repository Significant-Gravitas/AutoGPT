"""The failure streak against a real database (SECRT-2879).

The scheduler is mocked; the notes are really written, into a real chat, the
way the DatabaseManager writes them.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from prisma.actions import ScheduledChatStreakActions
from prisma.enums import AlertCause, AlertConditionStatus
from prisma.errors import UniqueViolationError
from prisma.models import (
    AlertCondition,
    ChatMessage,
    ExpertRoutine,
    ScheduledChatStreak,
    User,
)

from backend.api.features.experts import routines
from backend.copilot import db as copilot_db
from backend.copilot.scheduled_chats import forget, reconnect, service
from backend.copilot.scheduled_chats.outcomes import RunFailure, RunOutcome
from backend.copilot.scheduled_chats.reconnect import resume_after_reconnect
from backend.copilot.scheduled_chats.service import (
    list_paused_scheduled_chats,
    record_scheduled_run,
    resume_scheduled_chat,
)
from backend.copilot.scheduled_chats.service_fixtures import DISCORD_FAILURE as _DISCORD
from backend.copilot.scheduled_chats.service_fixtures import scheduled_run as _run
from backend.copilot.scheduled_chats.service_fixtures import (
    scheduler_fixture as scheduler_fixture,
)
from backend.copilot.scheduled_chats.service_fixtures import (
    watch_scheduler_lock_fixture as watch_scheduler_lock_fixture,
)
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)

_SERVICE = "backend.copilot.scheduled_chats.service"

_TWITTER = RunFailure(
    key="connection:twitter", kind="connection", provider="twitter", tool="run_block"
)
_TWITTER_OUTRIGHT = RunOutcome(failure=_TWITTER, whole_run=True)


async def _owner() -> tuple[str, str]:
    user_id = f"streak-{uuid4()}"
    try:
        await User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.com", "name": "Streak"}
        )
    except UniqueViolationError:
        pass
    session = await copilot_db.create_chat_session(
        session_id=str(uuid4()), user_id=user_id
    )
    return user_id, session.session_id


async def _cleanup(user_id: str) -> None:
    try:
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("cleanup for %s failed: %s", user_id, exc)


async def _notes(session_id: str) -> list[ChatMessage]:
    rows = await ChatMessage.prisma().find_many(
        where={"sessionId": session_id}, order={"sequence": "asc"}
    )
    return [
        r
        for r in rows
        if isinstance(r.metadata, dict)
        and r.metadata.get("kind") == "scheduled_chat_notice"
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_the_monitor_is_paused_and_told_on_the_third_run(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    try:
        for outcome in (
            RunOutcome(failure=_DISCORD, whole_run=True),
            RunOutcome(idle=True),
            RunOutcome(idle=True),
        ):
            await record_scheduled_run(_run(user_id, session_id, outcome))

        scheduler.pause_schedule.assert_awaited_once_with("sched-1", user_id=user_id)
        row = await ScheduledChatStreak.prisma().find_first(where={"userId": user_id})
        assert row is not None
        assert row.pausedAt is not None and row.notifiedAt is not None
        assert row.failures == 3
        notes = await _notes(session_id)
        assert [n.metadata["headline"] for n in notes] == [
            "Paused: “Discord monitor” can't use Discord"
        ]
        assert notes[0].metadata["hidden_from_model"] is True
        assert notes[0].metadata["schedule_key"] == "sched-1"

        paused = await list_paused_scheduled_chats(user_id)
        assert [p.summary for p in paused] == ["Paused: can't use Discord"]
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_failing_step_is_told_once_and_never_paused(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    try:
        for _ in range(4):
            await record_scheduled_run(
                _run(
                    user_id,
                    session_id,
                    RunOutcome(failure=_TWITTER, whole_run=False),
                    title="Daily briefing",
                )
            )

        scheduler.pause_schedule.assert_not_awaited()
        notes = await _notes(session_id)
        assert [n.metadata["headline"] for n in notes] == [
            "“Daily briefing” can't use Twitter"
        ]
        row = await ScheduledChatStreak.prisma().find_first(where={"userId": user_id})
        assert row is not None and row.failures == 4 and row.pausedAt is None
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_good_run_clears_the_streak(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        await record_scheduled_run(_run(user_id, session_id, failed))
        await record_scheduled_run(_run(user_id, session_id, RunOutcome()))

        assert await ScheduledChatStreak.prisma().count(where={"userId": user_id}) == 0
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_routine_pauses_every_fire_time_and_is_named_by_its_title(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    try:
        routine = await ExpertRoutine.prisma().create(
            data={
                "userId": user_id,
                "title": "Morning and noon sweep",
                "prompt": "Check the inbox.",
                "crons": ["30 8 * * *", "0 13 * * *"],
                "asks": [],
                "scheduleIds": ["job-a", "job-b"],
            }
        )
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for job in ("job-a", "job-b", "job-a"):
            await record_scheduled_run(
                _run(
                    user_id,
                    session_id,
                    failed,
                    schedule_key=routine.id,
                    schedule_id=job,
                    routine_id=routine.id,
                    title="check the inbox",
                )
            )

        paused = {c.args[0] for c in scheduler.pause_schedule.await_args_list}
        assert paused == {"job-a", "job-b"}
        notes = await _notes(session_id)
        assert notes[0].metadata["title"] == "Morning and noon sweep"
        # Deleted with its expert, so the paused list offers no Delete.
        assert [p.routine for p in await list_paused_scheduled_chats(user_id)] == [True]
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_switching_a_paused_routine_off_forgets_its_streak(
    server: SpinTestServer, scheduler: MagicMock, mocker
):
    """Its jobs are deleted as it is switched off, and switching it on again
    makes new ones that fire. Kept, the streak would still list it as
    paused, with a Resume that says it runs again."""
    scheduler.delete_schedule = AsyncMock()
    mocker.patch(
        "backend.api.features.experts.routine_jobs.get_scheduler_client",
        return_value=scheduler,
    )
    user_id, session_id = await _owner()
    try:
        routine = await ExpertRoutine.prisma().create(
            data={
                "userId": user_id,
                "title": "Morning sweep",
                "prompt": "Check the inbox.",
                "crons": ["30 8 * * *"],
                "asks": [],
                "scheduleIds": ["job-a"],
            }
        )
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(
                _run(
                    user_id,
                    session_id,
                    failed,
                    schedule_key=routine.id,
                    schedule_id="job-a",
                    routine_id=routine.id,
                )
            )
        assert len(await list_paused_scheduled_chats(user_id)) == 1

        await routines.disable_routine(user_id, None, routine.id)

        assert await list_paused_scheduled_chats(user_id) == []
        assert not await resume_scheduled_chat(user_id, routine.id)
        # Nor is its owner emailed about a routine they switched off.
        alert = await AlertCondition.prisma().find_first(
            where={"userId": user_id, "causeKey": f"scheduled_chat:{routine.id}"}
        )
        assert alert is not None and alert.status == AlertConditionStatus.RESOLVED
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_routine_is_switched_off_even_when_its_streak_cannot_be_cleared(
    server: SpinTestServer, scheduler: MagicMock, mocker
):
    """Its jobs are deleted and its row says off by then: a 500 would say the
    switch failed."""
    scheduler.delete_schedule = AsyncMock()
    mocker.patch(
        "backend.api.features.experts.routine_jobs.get_scheduler_client",
        return_value=scheduler,
    )
    mocker.patch.object(
        routines, "forget_routine", AsyncMock(side_effect=RuntimeError("db blip"))
    )
    user_id, _ = await _owner()
    try:
        routine = await _routine(user_id)

        await routines.disable_routine(user_id, None, routine.id)

        row = await ExpertRoutine.prisma().find_unique(where={"id": routine.id})
        assert row is not None and row.scheduleIds == [] and row.enabledAt is None
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_schedule_that_could_not_be_paused_does_not_say_it_was(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    scheduler.pause_schedule.side_effect = RuntimeError("scheduler down")
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(4):
            await record_scheduled_run(_run(user_id, session_id, failed))

        # Every run from the third tries the pause again.
        assert scheduler.pause_schedule.await_count == 2
        row = await ScheduledChatStreak.prisma().find_first(where={"userId": user_id})
        assert row is not None and row.pausedAt is None
        # The owner is still told, once, without being told it is paused.
        assert [n.metadata["headline"] for n in await _notes(session_id)] == [
            "“Discord monitor” can't use Discord"
        ]
    finally:
        await _cleanup(user_id)


async def _routine(user_id: str) -> ExpertRoutine:
    return await ExpertRoutine.prisma().create(
        data={
            "userId": user_id,
            "title": "Morning and noon sweep",
            "prompt": "Check the inbox.",
            "crons": ["30 8 * * *", "0 13 * * *"],
            "asks": [],
            "scheduleIds": ["job-a", "job-b"],
        }
    )


def _routine_run(user_id: str, session_id: str, routine: ExpertRoutine, job: str):
    return _run(
        user_id,
        session_id,
        RunOutcome(failure=_DISCORD, whole_run=True),
        schedule_key=routine.id,
        schedule_id=job,
        routine_id=routine.id,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_pause_that_fails_partway_puts_back_what_it_paused(
    server: SpinTestServer, scheduler: MagicMock
):
    async def pause(schedule_id: str, *, user_id: str) -> bool:
        if schedule_id == "job-b":
            raise RuntimeError("scheduler refused")
        return True

    scheduler.pause_schedule.side_effect = pause
    user_id, session_id = await _owner()
    try:
        routine = await _routine(user_id)
        for job in ("job-a", "job-b", "job-a"):
            await record_scheduled_run(_routine_run(user_id, session_id, routine, job))

        scheduler.resume_schedule.assert_awaited_once_with("job-a", user_id=user_id)
        assert await list_paused_scheduled_chats(user_id) == []
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_pause_whose_streak_cannot_be_saved_is_put_back(
    server: SpinTestServer, scheduler: MagicMock, mocker
):
    """Paused jobs with no row behind them would never be listed or resumed:
    the chat would stop without a word."""
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(2):
            await record_scheduled_run(_run(user_id, session_id, failed))
        mocker.patch(f"{_SERVICE}._save", AsyncMock(side_effect=RuntimeError("db")))

        with pytest.raises(RuntimeError):
            await record_scheduled_run(_run(user_id, session_id, failed))

        scheduler.pause_schedule.assert_awaited_once_with("sched-1", user_id=user_id)
        scheduler.resume_schedule.assert_awaited_once_with("sched-1", user_id=user_id)
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_run_that_fired_before_the_pause_changes_nothing(
    server: SpinTestServer, scheduler: MagicMock
):
    """An every-minute chat can have its next run going when the third pauses
    it. That run finishes after the pause, and must not undo it."""
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))

        for straggler in (failed, RunOutcome()):
            await record_scheduled_run(_run(user_id, session_id, straggler))

        assert len(await list_paused_scheduled_chats(user_id)) == 1
        assert len(await _notes(session_id)) == 1
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_two_runs_finishing_together_keep_the_pause(
    server: SpinTestServer, scheduler: MagicMock, watch_scheduler_lock, mocker
):
    """The second failure waits for the first pause and leaves its row alone."""
    pause_started = asyncio.Event()
    finish_pause = asyncio.Event()

    async def pause(schedule_id: str, *, user_id: str) -> bool:
        pause_started.set()
        await finish_pause.wait()
        return True

    scheduler.pause_schedule.side_effect = pause
    user_id, session_id = await _owner()
    run = _run(user_id, session_id, RunOutcome(failure=_DISCORD, whole_run=True))
    contend = watch_scheduler_lock(user_id, run.schedule_key)
    read_streak = mocker.spy(ScheduledChatStreakActions, "find_unique")

    try:
        for _ in range(2):
            await record_scheduled_run(run)

        async with asyncio.timeout(30), asyncio.TaskGroup() as tasks:
            tasks.create_task(record_scheduled_run(run))
            await pause_started.wait()
            reads = read_streak.await_count
            tasks.create_task(record_scheduled_run(run))
            await contend.wait()
            assert read_streak.await_count == reads
            finish_pause.set()

        scheduler.pause_schedule.assert_awaited_once()
        assert len(await list_paused_scheduled_chats(user_id)) == 1
        assert len(await _notes(session_id)) == 1
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_good_run_that_read_the_streak_before_the_pause_keeps_it(
    server: SpinTestServer, scheduler: MagicMock
):
    """It ends the streak it read. The row another run paused since must
    stay, or its jobs stay paused with nothing to list or resume them."""
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))
        streak_of = service._streak_of

        def before_the_pause(row):
            streak = streak_of(row)
            return streak.model_copy(update={"paused": False}) if streak else None

        with patch.object(service, "_streak_of", new=before_the_pause):
            await record_scheduled_run(_run(user_id, session_id, RunOutcome()))

        assert len(await list_paused_scheduled_chats(user_id)) == 1
        # And its alert: withdrawn only with the streak.
        alert = await _alert(user_id)
        assert alert is not None and alert.status == AlertConditionStatus.PENDING
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_new_failure_finishing_with_the_pause_keeps_it(
    server: SpinTestServer, scheduler: MagicMock, watch_scheduler_lock, mocker
):
    """A different failure reported during a pause keeps that pause."""
    pause_started = asyncio.Event()
    finish_pause = asyncio.Event()

    async def pause(schedule_id: str, *, user_id: str) -> bool:
        pause_started.set()
        await finish_pause.wait()
        return True

    scheduler.pause_schedule.side_effect = pause
    user_id, session_id = await _owner()
    run = _run(user_id, session_id, RunOutcome(failure=_DISCORD, whole_run=True))
    contend = watch_scheduler_lock(user_id, run.schedule_key)
    read_streak = mocker.spy(ScheduledChatStreakActions, "find_unique")

    try:
        for _ in range(2):
            await record_scheduled_run(run)

        async with asyncio.timeout(30), asyncio.TaskGroup() as tasks:
            tasks.create_task(record_scheduled_run(run))
            await pause_started.wait()
            reads = read_streak.await_count
            tasks.create_task(
                record_scheduled_run(_run(user_id, session_id, _TWITTER_OUTRIGHT))
            )
            await contend.wait()
            assert read_streak.await_count == reads
            finish_pause.set()

        paused = await list_paused_scheduled_chats(user_id)
        assert [chat.provider for chat in paused] == ["discord"]
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_chat_resumed_some_other_way_starts_over(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))
        scheduler.get_execution_schedules.return_value = [MagicMock(id="sched-1")]

        await record_scheduled_run(_run(user_id, session_id, RunOutcome()))

        assert await ScheduledChatStreak.prisma().count(where={"userId": user_id}) == 0
        # The run that worked is the fix, so the pause alert is withdrawn too.
        alert = await _alert(user_id)
        assert alert is not None and alert.status == AlertConditionStatus.RESOLVED
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_chat_resumed_some_other_way_that_runs_idle_is_not_called_paused(
    server: SpinTestServer, scheduler: MagicMock
):
    """Its next run only repeated that Discord needs connecting. The chat runs
    again, so neither the Scheduled page nor its alert may call it paused."""
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))
        scheduler.get_execution_schedules.return_value = [MagicMock(id="sched-1")]

        await record_scheduled_run(_run(user_id, session_id, RunOutcome(idle=True)))

        assert await list_paused_scheduled_chats(user_id) == []
        row = await ScheduledChatStreak.prisma().find_first(where={"userId": user_id})
        assert row is not None and row.failures == 1 and row.pausedAt is None
        alert = await _alert(user_id)
        assert alert is not None and alert.status == AlertConditionStatus.RESOLVED
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_pause_whose_card_failed_has_its_alert_withdrawn_when_it_ends(
    server: SpinTestServer, scheduler: MagicMock, mocker
):
    """The pause raised its alert, but its card didn't post, so the streak
    reads as never told. Resumed some other way, the run that ends it must
    still withdraw the alert."""
    mocker.patch(f"{_SERVICE}.post_notice", side_effect=RuntimeError("db"))
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))
        row = await ScheduledChatStreak.prisma().find_first(where={"userId": user_id})
        assert row is not None and row.pausedAt is not None and row.notifiedAt is None
        scheduler.get_execution_schedules.return_value = [MagicMock(id="sched-1")]

        await record_scheduled_run(_run(user_id, session_id, RunOutcome()))

        alert = await _alert(user_id)
        assert alert is not None and alert.status == AlertConditionStatus.RESOLVED
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_note_that_could_not_be_posted_is_told_on_the_next_run(
    server: SpinTestServer, scheduler: MagicMock, mocker
):
    post = mocker.patch(
        f"{_SERVICE}.post_notice", side_effect=[RuntimeError("db"), True]
    )
    user_id, session_id = await _owner()
    try:
        failing = RunOutcome(failure=_TWITTER, whole_run=False)
        for _ in range(4):
            await record_scheduled_run(_run(user_id, session_id, failing))

        assert post.await_count == 2
        row = await ScheduledChatStreak.prisma().find_first(where={"userId": user_id})
        assert row is not None and row.notifiedAt is not None
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_resume_restarts_the_jobs_clears_the_streak_and_says_so(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))

        assert await resume_scheduled_chat(user_id, "sched-1") is True
        # A second click on the same card finds nothing paused.
        assert await resume_scheduled_chat(user_id, "sched-1") is False
        # And nobody else's chat can be resumed.
        assert await resume_scheduled_chat("someone-else", "sched-1") is False

        scheduler.resume_schedule.assert_awaited_once_with("sched-1", user_id=user_id)
        assert await ScheduledChatStreak.prisma().count(where={"userId": user_id}) == 0
        headlines = [n.metadata["headline"] for n in await _notes(session_id)]
        assert headlines[-1] == "Resumed “Discord monitor”"
    finally:
        await _cleanup(user_id)


async def _alert(user_id: str) -> AlertCondition | None:
    return await AlertCondition.prisma().find_first(
        where={"userId": user_id, "causeKey": "scheduled_chat:sched-1"}
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_chat_resumed_as_its_pause_raises_the_alert_is_left_without_one(
    server: SpinTestServer, scheduler: MagicMock, watch_scheduler_lock, mocker
):
    """Resuming while the pause raises its alert clears the alert."""
    alert_started = asyncio.Event()
    finish_alert = asyncio.Event()
    raise_paused = service.alerts.raise_paused
    user_id, session_id = await _owner()
    run = _run(user_id, session_id, RunOutcome(failure=_DISCORD, whole_run=True))
    contend = watch_scheduler_lock(user_id, run.schedule_key)

    async def held_alert(*args, **kwargs):
        alert_started.set()
        await finish_alert.wait()
        await raise_paused(*args, **kwargs)

    mocker.patch.object(service.alerts, "raise_paused", new=held_alert)
    try:
        for _ in range(2):
            await record_scheduled_run(run)

        async with asyncio.timeout(30), asyncio.TaskGroup() as tasks:
            tasks.create_task(record_scheduled_run(run))
            await alert_started.wait()
            result = tasks.create_task(resume_scheduled_chat(user_id, run.schedule_key))
            await contend.wait()
            scheduler.resume_schedule.assert_not_awaited()
            finish_alert.set()

        assert result.result() is True
        alert = await _alert(user_id)
        assert alert is not None and alert.status == AlertConditionStatus.RESOLVED
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_pause_is_told_even_when_its_alert_cannot_be_checked(
    server: SpinTestServer, scheduler: MagicMock
):
    """The check after the alert is only a guard: failing, it must not keep
    the card from the chat, which has no next run to try again on."""
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(2):
            await record_scheduled_run(_run(user_id, session_id, failed))
        failing = AsyncMock(side_effect=RuntimeError("db blip"))
        with patch.object(ScheduledChatStreakActions, "count", failing):
            await record_scheduled_run(_run(user_id, session_id, failed))
        failing.assert_awaited_once()

        headlines = [n.metadata["headline"] for n in await _notes(session_id)]
        assert headlines[-1].startswith("Paused: ")
        alert = await _alert(user_id)
        assert alert is not None and alert.status == AlertConditionStatus.PENDING
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_pause_raises_the_alert_and_resume_resolves_it(
    server: SpinTestServer, scheduler: MagicMock
):
    """The alert engine turns it into the email (where user email is on) or
    the Briefing card; resuming cancels a send still in its debounce."""
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))

        alert = await _alert(user_id)
        assert alert is not None
        assert alert.cause == AlertCause.PAUSED_FAILURES
        assert alert.status == AlertConditionStatus.PENDING
        assert alert.data["agent"] == "Discord monitor"
        assert alert.data["step"] == "Discord connection"

        assert await resume_scheduled_chat(user_id, "sched-1")

        alert = await _alert(user_id)
        assert alert is not None and alert.status == AlertConditionStatus.RESOLVED
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_reconnecting_the_provider_resumes_what_it_paused(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))
        # Another chat only told about Twitter, and one about Discord's twin.
        twitter = RunOutcome(failure=_TWITTER, whole_run=False)
        for _ in range(3):
            await record_scheduled_run(
                _run(user_id, session_id, twitter, schedule_key="briefing")
            )

        await resume_after_reconnect(user_id, "discord")

        scheduler.resume_schedule.assert_awaited_once_with("sched-1", user_id=user_id)
        alert = await _alert(user_id)
        assert alert is not None and alert.status == AlertConditionStatus.RESOLVED
        keys = {
            row.scheduleKey
            for row in await ScheduledChatStreak.prisma().find_many(
                where={"userId": user_id}
            )
        }
        assert keys == {"briefing"}

        await resume_after_reconnect(user_id, "twitter")

        assert await ScheduledChatStreak.prisma().count(where={"userId": user_id}) == 0
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_chat_paused_while_a_reconnect_reads_it_is_resumed(
    server: SpinTestServer, scheduler: MagicMock
):
    """The reconnect read the chat as only told, while a run still failing
    on the old connection went on to pause it. Deleting from that stale read
    would leave its jobs paused with nothing to list or resume them."""
    user_id, session_id = await _owner()
    try:
        told = RunOutcome(failure=_DISCORD, whole_run=False)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, told))
        read = await ScheduledChatStreak.prisma().find_first(where={"userId": user_id})
        assert read is not None and read.pausedAt is None
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))
        assert len(await list_paused_scheduled_chats(user_id)) == 1

        await reconnect._clear(user_id, read)

        scheduler.resume_schedule.assert_awaited_once_with("sched-1", user_id=user_id)
        assert await list_paused_scheduled_chats(user_id) == []
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_two_resumes_at_once_both_say_it_resumed(
    server: SpinTestServer, scheduler: MagicMock
):
    """Two callers who observe the same pause both get a successful result."""
    both_read_the_row = asyncio.Barrier(2)
    find_unique = ScheduledChatStreakActions.find_unique
    remaining_observations = 2

    async def read_initial_pause(actions, **kwargs):
        nonlocal remaining_observations
        row = await find_unique(actions, **kwargs)
        if remaining_observations:
            remaining_observations -= 1
            await both_read_the_row.wait()
        return row

    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))

        with patch.object(
            ScheduledChatStreakActions, "find_unique", new=read_initial_pause
        ):
            async with asyncio.timeout(10), asyncio.TaskGroup() as tasks:
                first = tasks.create_task(resume_scheduled_chat(user_id, "sched-1"))
                second = tasks.create_task(resume_scheduled_chat(user_id, "sched-1"))

        assert first.result() is second.result() is True
        scheduler.resume_schedule.assert_awaited_once_with("sched-1", user_id=user_id)
        assert await list_paused_scheduled_chats(user_id) == []
        headlines = [n.metadata["headline"] for n in await _notes(session_id)]
        assert headlines.count("Resumed “Discord monitor”") == 1
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_one_chat_that_cannot_resume_does_not_hold_up_the_rest(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for key in ("first", "second"):
            for _ in range(3):
                await record_scheduled_run(
                    _run(user_id, session_id, failed, schedule_key=key, schedule_id=key)
                )
        scheduler.resume_schedule.side_effect = [RuntimeError("scheduler down"), True]

        await resume_after_reconnect(user_id, "discord")

        assert scheduler.resume_schedule.await_count == 2
        assert len(await list_paused_scheduled_chats(user_id)) == 1
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_resume_the_scheduler_refuses_keeps_the_chat_paused(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))
        scheduler.resume_schedule.side_effect = RuntimeError("scheduler down")

        with pytest.raises(RuntimeError):
            await resume_scheduled_chat(user_id, "sched-1")

        assert len(await list_paused_scheduled_chats(user_id)) == 1
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_resume_that_fails_partway_pauses_the_rest_again(
    server: SpinTestServer, scheduler: MagicMock
):
    async def resume(schedule_id: str, *, user_id: str) -> bool:
        if schedule_id == "job-b":
            raise RuntimeError("scheduler refused")
        return True

    user_id, session_id = await _owner()
    try:
        routine = await _routine(user_id)
        for job in ("job-a", "job-b", "job-a"):
            await record_scheduled_run(_routine_run(user_id, session_id, routine, job))
        scheduler.pause_schedule.reset_mock()
        scheduler.resume_schedule.side_effect = resume

        with pytest.raises(RuntimeError):
            await resume_scheduled_chat(user_id, routine.id)

        # job-b too: its resume may have landed with the answer lost.
        paused = [call.args[0] for call in scheduler.pause_schedule.await_args_list]
        assert paused == ["job-a", "job-b"]
        assert len(await list_paused_scheduled_chats(user_id)) == 1
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_resume_leaves_a_job_its_owner_paused_alone(
    server: SpinTestServer, scheduler: MagicMock
):
    """The owner had paused the noon run; the streak paused the morning one.
    Resume brings back the morning run only."""

    async def pause(schedule_id: str, *, user_id: str) -> bool:
        return schedule_id == "job-a"

    scheduler.pause_schedule.side_effect = pause
    user_id, session_id = await _owner()
    try:
        routine = await _routine(user_id)
        for _ in range(3):
            await record_scheduled_run(
                _routine_run(user_id, session_id, routine, "job-a")
            )
        assert len(await list_paused_scheduled_chats(user_id)) == 1

        assert await resume_scheduled_chat(user_id, routine.id) is True

        scheduler.resume_schedule.assert_awaited_once_with("job-a", user_id=user_id)
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_chat_whose_jobs_its_owner_paused_is_not_called_paused(
    server: SpinTestServer, scheduler: MagicMock
):
    scheduler.pause_schedule.return_value = False
    user_id, session_id = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))

        assert await list_paused_scheduled_chats(user_id) == []
        assert [n.metadata["headline"] for n in await _notes(session_id)] == [
            "\u201cDiscord monitor\u201d can't use Discord"
        ]
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_job_deleted_since_the_pause_does_not_block_the_resume(
    server: SpinTestServer, scheduler: MagicMock
):
    user_id, session_id = await _owner()
    try:
        routine = await _routine(user_id)
        for job in ("job-a", "job-b", "job-a"):
            await record_scheduled_run(_routine_run(user_id, session_id, routine, job))
        scheduler.resume_schedule.side_effect = [NotFoundError("gone"), True]

        assert await resume_scheduled_chat(user_id, routine.id) is True

        assert await list_paused_scheduled_chats(user_id) == []
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_deleted_chat_leaves_no_paused_ghost(
    server: SpinTestServer, scheduler: MagicMock
):
    """Deleting a paused chat used to leave its streak: still listed as
    paused, and a Resume that said it would run again."""
    user_id, session_id = await _owner()
    other_id, other_session = await _owner()
    try:
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(_run(user_id, session_id, failed))
            await record_scheduled_run(_run(other_id, other_session, failed))

        await forget.forget_scheduled_chat(user_id, "sched-1")

        assert await ScheduledChatStreak.prisma().count(where={"userId": user_id}) == 0
        assert await list_paused_scheduled_chats(user_id) == []
        assert await resume_scheduled_chat(user_id, "sched-1") is False
        # Nor does its alert email about a chat that is gone.
        alert = await _alert(user_id)
        assert alert is not None and alert.status == AlertConditionStatus.RESOLVED
        other_alert = await _alert(other_id)
        assert other_alert is not None
        assert other_alert.status == AlertConditionStatus.PENDING
        # Only the owner's own chat goes.
        [left] = await list_paused_scheduled_chats(other_id)
        # A follow-up, which the paused list can delete.
        assert left.routine is False
    finally:
        await _cleanup(user_id)
        await _cleanup(other_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_forgetting_a_routines_id_leaves_its_pause_alone(
    server: SpinTestServer, scheduler: MagicMock
):
    """A delete names a schedule id; a routine's streak outlives any one of
    its jobs, so its paused row stays listed with Resume."""
    user_id, session_id = await _owner()
    try:
        routine = await _routine(user_id)
        for job in ("job-a", "job-b", "job-a"):
            await record_scheduled_run(_routine_run(user_id, session_id, routine, job))

        await forget.forget_scheduled_chat(user_id, routine.id)

        paused = await list_paused_scheduled_chats(user_id)
        assert [p.schedule_key for p in paused] == [routine.id]
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_key_naming_someone_elses_routine_is_a_follow_up(
    server: SpinTestServer, scheduler: MagicMock
):
    """Only the owner's routines are kept and listed as routines: a key that
    names another account's routine is a follow-up, which a delete clears."""
    user_id, session_id = await _owner()
    other_id, _ = await _owner()
    try:
        foreign = await _routine(other_id)
        failed = RunOutcome(failure=_DISCORD, whole_run=True)
        for _ in range(3):
            await record_scheduled_run(
                _run(
                    user_id,
                    session_id,
                    failed,
                    schedule_key=foreign.id,
                    schedule_id=foreign.id,
                )
            )
        [paused] = await list_paused_scheduled_chats(user_id)
        assert paused.routine is False

        await forget.forget_scheduled_chat(user_id, foreign.id)

        assert await list_paused_scheduled_chats(user_id) == []
    finally:
        await _cleanup(user_id)
        await _cleanup(other_id)
