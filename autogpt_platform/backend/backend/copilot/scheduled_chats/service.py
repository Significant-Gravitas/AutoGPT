"""A scheduled chat's failure streak, kept in the database (SECRT-2879).

Runs in the DatabaseManager: the copilot executor records each scheduled run's
outcome through one RPC (``record_scheduled_run``), and the REST API resumes a
paused chat (``resume_scheduled_chat``). This module loads the streak, asks
``streaks.advance`` what the run changes, and carries it out: pause the
schedule's jobs, persist the streak, and tell the owner in the chat.
"""

import logging
from datetime import datetime, timezone

from prisma.errors import UniqueViolationError
from prisma.models import ExpertRoutine, ScheduledChatStreak
from prisma.types import (
    ScheduledChatStreakWhereInput,
    ScheduledChatStreakWhereUniqueInput,
)
from pydantic import BaseModel, ValidationError

from backend.copilot.scheduled_chats import alerts
from backend.copilot.scheduled_chats.failure_notices import (
    failing_notice,
    pause_summary,
    paused_notice,
    resumed_notice,
)
from backend.copilot.scheduled_chats.forget import owned_routines, resolve_alert
from backend.copilot.scheduled_chats.jobs import (
    jobs_paused,
    pause_jobs,
    resume_jobs,
    unpause,
)
from backend.copilot.scheduled_chats.notices import ScheduledChatNotice, post_notice
from backend.copilot.scheduled_chats.outcomes import RunFailure, RunOutcome
from backend.copilot.scheduled_chats.streaks import Streak, StreakStep, advance

logger = logging.getLogger(__name__)


class ScheduledRun(BaseModel):
    """One scheduled run's outcome, as the copilot executor saw it."""

    user_id: str
    # The routine's id, or the follow-up's schedule id.
    schedule_key: str
    schedule_id: str | None = None
    routine_id: str | None = None
    # The chat the run went to.
    session_id: str
    # A follow-up's name; a routine's comes from its row.
    title: str
    outcome: RunOutcome


class PausedScheduledChat(BaseModel):
    """A scheduled chat its failure streak paused, as the lists show it."""

    # What Resume takes: the routine's id, or the follow-up's schedule id.
    schedule_key: str
    title: str
    # The chat its last run went to, where the note with the fix is.
    session_id: str | None
    reason: str
    provider: str | None
    # "Paused: can't use Twitter", and so on.
    summary: str
    paused_at: datetime
    # A routine is deleted with its expert, not from this list.
    routine: bool = False


async def record_scheduled_run(run: ScheduledRun) -> None:
    where = _key(run.user_id, run.schedule_key)
    row = await ScheduledChatStreak.prisma().find_unique(where=where)
    current = _streak_of(row)
    saw_pause = current is not None and current.paused
    if saw_pause and await _still_paused(run):
        # It fired before the pause and finished after it, so it changes
        # nothing. One that fired after was resumed some other way.
        return
    step = advance(current, run.outcome)
    # A pause raised the alert whether or not its card got through, so the
    # pause ending withdraws it, as the end of a streak the owner was told of.
    withdraw = step.ended_notified or saw_pause
    if step.streak is None:
        # Withdrawn only with the streak: a pause this run didn't see stands,
        # and so does its alert.
        dropped = row is not None and await _drop(row, saw_pause=saw_pause)
        if dropped and withdraw:
            await resolve_alert(run.user_id, run.schedule_key)
        return
    if step.streak == current:
        return
    streak = step.streak
    schedule_ids, title = await _schedule_of(run)
    changed = await pause_jobs(run.user_id, schedule_ids) if step.pause else None
    # Only a pause that stopped something counts, and only what it
    # stopped comes back on Resume.
    paused = bool(changed)
    if step.pause and not paused:
        # Nothing was paused, so nothing may say it was. Each later run tries
        # again; meanwhile the owner is told once, without that claim.
        streak = streak.model_copy(update={"paused": False})
    told = streak.notified and not (current is not None and current.notified)
    try:
        row = await _save(
            run,
            streak,
            changed or [],
            title,
            restarted=row is None or streak.failures == 1,
            told=told,
            replaces_pause=saw_pause,
        )
    except Exception:
        # A pause with no row behind it is never listed or resumed: the chat
        # would stop without a word. Put the jobs back; the next run retries.
        if changed:
            await unpause(run.user_id, changed)
        raise
    if row is None:
        # Another run of this chat paused it after this one read the streak,
        # and told the owner. That pause stands.
        return
    if withdraw:
        # A new failure replaced the streak the owner was told about, or the
        # pause. Before any pause it raises: both are this chat's one alert.
        await resolve_alert(run.user_id, run.schedule_key)
    if paused:
        await _raise_paused(run, row, streak, title)
    notice = _notice(step, streak, paused=paused, told=told, title=title, row=row)
    if notice is not None:
        await _tell(run, row, notice, told=told)


async def resume_scheduled_chat(user_id: str, schedule_key: str) -> bool:
    """Resume a scheduled chat its streak paused, and say whether it was.

    Its jobs are resumed before the streak is cleared, all of them or none:
    a failure keeps the row, so the chat still reads as paused and Resume can
    be tried again.
    """
    row = await ScheduledChatStreak.prisma().find_unique(
        where=_key(user_id, schedule_key)
    )
    if row is None or row.pausedAt is None:
        return False
    await resume_jobs(user_id, row.scheduleIds)
    await ScheduledChatStreak.prisma().delete(where={"id": row.id})
    await resolve_alert(user_id, schedule_key)
    if row.sessionId is not None:
        try:
            await post_notice(
                resumed_notice(title=row.title, streak_id=row.id),
                user_id=user_id,
                schedule_key=schedule_key,
                title=row.title,
                session_id=row.sessionId,
                expert_id=None,
            )
        except Exception:
            logger.warning(
                "Could not post the resumed note for scheduled chat %s",
                schedule_key,
                exc_info=True,
            )
    return True


async def list_paused_scheduled_chats(user_id: str) -> list[PausedScheduledChat]:
    rows = await ScheduledChatStreak.prisma().find_many(
        where={"userId": user_id, "NOT": [{"pausedAt": None}]},
        order={"pausedAt": "desc"},
    )
    if not rows:
        # Home asks on every load; most have nothing paused.
        return []
    routines = await ExpertRoutine.prisma().find_many(
        where={
            "id": {"in": [row.scheduleKey for row in rows]},
            **owned_routines(user_id),
        }
    )
    routine_ids = {routine.id for routine in routines}
    paused: list[PausedScheduledChat] = []
    for row in rows:
        streak = _streak_of(row)
        if streak is None or row.pausedAt is None:
            continue
        paused.append(
            PausedScheduledChat(
                schedule_key=row.scheduleKey,
                title=row.title,
                session_id=row.sessionId,
                reason=streak.failure.kind,
                provider=streak.failure.provider,
                summary=pause_summary(streak.failure),
                paused_at=row.pausedAt,
                routine=row.scheduleKey in routine_ids,
            )
        )
    return paused


async def _still_paused(run: ScheduledRun) -> bool:
    """Whether the chat's jobs are paused yet. A routine's are its current
    ones: editing it while paused replaces them with jobs that fire."""
    schedule_ids, _ = await _schedule_of(run)
    return await jobs_paused(run.user_id, schedule_ids)


def _notice(
    step: StreakStep,
    streak: Streak,
    *,
    paused: bool,
    told: bool,
    title: str,
    row: ScheduledChatStreak,
) -> ScheduledChatNotice | None:
    failure, failures = streak.failure, streak.failures
    if paused:
        return paused_notice(failure, title=title, failures=failures, streak_id=row.id)
    if step.pause and told:
        return paused_notice(
            failure, title=title, failures=failures, streak_id=row.id, paused=False
        )
    if step.tell:
        return failing_notice(failure, title=title, failures=failures, streak_id=row.id)
    return None


async def _tell(
    run: ScheduledRun,
    row: ScheduledChatStreak,
    notice: ScheduledChatNotice,
    *,
    told: bool,
) -> None:
    try:
        await post_notice(
            notice,
            user_id=run.user_id,
            schedule_key=run.schedule_key,
            title=row.title,
            session_id=run.session_id,
            expert_id=None,
        )
    except Exception:
        logger.warning(
            "Could not post the note for scheduled chat %s",
            run.schedule_key,
            exc_info=True,
        )
        if told:
            # Not told after all: a later run that fails this way tells
            # again. A paused chat has no later run, but the Scheduled page
            # still lists it, with Resume.
            await ScheduledChatStreak.prisma().update(
                where={"id": row.id}, data={"notifiedAt": None}
            )


async def _schedule_of(run: ScheduledRun) -> tuple[list[str], str]:
    """The jobs to pause and the name to use: a routine's from its row, which
    knows every fire time, a follow-up's from the run."""
    fallback = [run.schedule_id] if run.schedule_id else []
    if run.routine_id is None:
        return fallback, run.title
    routine = await ExpertRoutine.prisma().find_unique(where={"id": run.routine_id})
    if routine is None:
        return fallback, run.title
    return list(routine.scheduleIds) or fallback, routine.title


async def _save(
    run: ScheduledRun,
    streak: Streak,
    paused_ids: list[str],
    title: str,
    *,
    restarted: bool,
    told: bool,
    replaces_pause: bool,
) -> ScheduledChatStreak | None:
    """A streak that restarted gets a new row, so its id names this streak
    alone: the notes it posts are deduplicated on it. *told* stamps when the
    owner was told, once.

    Two runs of an every-minute chat can finish together. A view of the
    streak from before a pause never overwrites that pause: its jobs are
    paused, and a row that says otherwise could never be resumed. Only a run
    that saw the pause, and found the jobs firing again, replaces it
    (*replaces_pause*). Returns None, having written nothing, when the pause
    stands."""
    now = datetime.now(tz=timezone.utc)
    fields = {
        "scheduleIds": paused_ids,
        "title": title,
        "sessionId": run.session_id,
        "failureKey": streak.failure.key,
        "failureKind": streak.failure.kind,
        "provider": streak.failure.provider,
        "tool": streak.failure.tool,
        "failures": streak.failures,
        "wholeRunFailures": streak.whole_run_failures,
        # Only the run that pauses a schedule writes here: a paused one fires
        # no runs until it is resumed, and resuming removes the row.
        "pausedAt": now if streak.paused else None,
    }
    where: ScheduledChatStreakWhereInput = {
        "userId": run.user_id,
        "scheduleKey": run.schedule_key,
    }
    if not (streak.paused or replaces_pause):
        where["pausedAt"] = None
    if not restarted:
        updated = await ScheduledChatStreak.prisma().update_many(
            where=where, data={**fields, **({"notifiedAt": now} if told else {})}
        )
        if updated:
            return await ScheduledChatStreak.prisma().find_unique(
                where=_key(run.user_id, run.schedule_key)
            )
    await ScheduledChatStreak.prisma().delete_many(where=where)
    try:
        return await ScheduledChatStreak.prisma().create(
            data={
                **fields,
                "userId": run.user_id,
                "scheduleKey": run.schedule_key,
                "notifiedAt": now if told else None,
            }
        )
    except UniqueViolationError:
        # The paused row the delete left in place.
        return None


async def _raise_paused(
    run: ScheduledRun, row: ScheduledChatStreak, streak: Streak, title: str
) -> None:
    """Raise the chat's pause alert. A resume can land between the pause's row
    and this, its own withdrawal before the alert existed, so the alert stays
    only if the row is still paused once it is raised."""
    await alerts.raise_paused(
        run.user_id,
        run.schedule_key,
        alerts.paused_cause(
            streak.failure,
            title=title,
            failures=streak.failures,
            session_id=run.session_id,
        ),
    )
    try:
        still_paused = await ScheduledChatStreak.prisma().count(
            where={"id": row.id, "NOT": [{"pausedAt": None}]}
        )
    except Exception:
        # Only a guard, and the pause most likely stands: keep the alert, and
        # let the chat still be told.
        logger.warning(
            "Could not check scheduled chat %s is still paused",
            run.schedule_key,
            exc_info=True,
        )
        return
    if not still_paused:
        await resolve_alert(run.user_id, run.schedule_key)


async def _drop(row: ScheduledChatStreak, *, saw_pause: bool) -> bool:
    """End the streak a run read, and say whether it did. As in ``_save``, a
    view from before a pause never ends that pause: its jobs are paused, and
    without the row nothing lists or resumes them. Only a run that saw the
    pause, and found the jobs firing again, ends one."""
    where: ScheduledChatStreakWhereInput = {"id": row.id}
    if not saw_pause:
        where["pausedAt"] = None
    return bool(await ScheduledChatStreak.prisma().delete_many(where=where))


def _streak_of(row: ScheduledChatStreak | None) -> Streak | None:
    if row is None:
        return None
    try:
        failure = RunFailure.model_validate(
            {
                "key": row.failureKey,
                "kind": row.failureKind,
                "provider": row.provider,
                "tool": row.tool,
            }
        )
    except ValidationError:
        # A kind this code no longer knows: start the count over.
        logger.warning("Dropping unreadable streak %s", row.id)
        return None
    return Streak(
        failure=failure,
        failures=row.failures,
        whole_run_failures=row.wholeRunFailures,
        notified=row.notifiedAt is not None,
        paused=row.pausedAt is not None,
    )


def _key(user_id: str, schedule_key: str) -> ScheduledChatStreakWhereUniqueInput:
    return {"userId_scheduleKey": {"userId": user_id, "scheduleKey": schedule_key}}
