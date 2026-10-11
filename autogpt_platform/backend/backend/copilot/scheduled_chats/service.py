"""A scheduled chat's failure streak, kept in the database (SECRT-2879).

Runs in the DatabaseManager: the copilot executor records each scheduled run's
outcome through one RPC (``record_scheduled_run``), and the REST API resumes a
paused chat (``resume_scheduled_chat``). This module loads the streak, asks
``streaks.advance`` what the run changes, and carries it out: pause the
schedule's jobs, persist the streak, and tell the owner in the chat.
"""

import logging
from datetime import datetime, timezone

from prisma.models import ExpertRoutine, ScheduledChatStreak
from prisma.types import ScheduledChatStreakWhereUniqueInput
from pydantic import BaseModel, ValidationError

from backend.copilot.scheduled_chats.failure_notices import (
    failing_notice,
    pause_summary,
    paused_notice,
    resumed_notice,
)
from backend.copilot.scheduled_chats.notices import post_notice
from backend.copilot.scheduled_chats.outcomes import RunFailure, RunOutcome
from backend.copilot.scheduled_chats.streaks import Streak, advance
from backend.util.clients import get_scheduler_client

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


async def record_scheduled_run(run: ScheduledRun) -> None:
    where = _key(run.user_id, run.schedule_key)
    row = await ScheduledChatStreak.prisma().find_unique(where=where)
    current = _streak_of(row)
    step = advance(current, run.outcome)
    if step.streak is None:
        if row is not None:
            await ScheduledChatStreak.prisma().delete(where={"id": row.id})
        return
    if step.streak == current:
        return
    streak = step.streak
    schedule_ids, title = await _schedule_of(run)
    if step.pause and not await _pause(run.user_id, schedule_ids):
        # Nothing was paused, so nothing may say it was: the next run counts
        # again and tries once more.
        streak = streak.model_copy(update={"paused": False, "notified": False})
    row = await _save(
        run,
        streak,
        schedule_ids,
        title,
        restarted=row is None or streak.failures == 1,
        told=streak.notified and not (current is not None and current.notified),
    )
    if streak.paused:
        notice = paused_notice(
            streak.failure, title=title, failures=streak.failures, streak_id=row.id
        )
    elif step.tell:
        notice = failing_notice(
            streak.failure, title=title, failures=streak.failures, streak_id=row.id
        )
    else:
        return
    await post_notice(
        notice,
        user_id=run.user_id,
        schedule_key=run.schedule_key,
        title=title,
        session_id=run.session_id,
        expert_id=None,
    )


async def resume_scheduled_chat(user_id: str, schedule_key: str) -> bool:
    """Resume a scheduled chat its streak paused, and say whether it was.

    Every job is resumed before the streak is cleared: one that fails leaves
    the row, so the chat still reads as paused and Resume can be tried again.
    """
    row = await ScheduledChatStreak.prisma().find_unique(
        where=_key(user_id, schedule_key)
    )
    if row is None or row.pausedAt is None:
        return False
    scheduler = get_scheduler_client()
    for schedule_id in row.scheduleIds:
        await scheduler.resume_schedule(schedule_id, user_id=user_id)
    await ScheduledChatStreak.prisma().delete(where={"id": row.id})
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
            )
        )
    return paused


async def _pause(user_id: str, schedule_ids: list[str]) -> bool:
    """Pause every job, and say whether any is paused now."""
    scheduler = get_scheduler_client()
    paused = False
    for schedule_id in schedule_ids:
        try:
            await scheduler.pause_schedule(schedule_id, user_id=user_id)
            paused = True
        except Exception:
            logger.warning(
                "Could not pause scheduled chat job %s", schedule_id, exc_info=True
            )
    return paused


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
    schedule_ids: list[str],
    title: str,
    *,
    restarted: bool,
    told: bool,
) -> ScheduledChatStreak:
    """A streak that restarted gets a new row, so its id names this streak
    alone: the notes it posts are deduplicated on it. *told* stamps when the
    owner was told, once."""
    now = datetime.now(tz=timezone.utc)
    fields = {
        "scheduleIds": schedule_ids,
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
    if not restarted:
        updated = await ScheduledChatStreak.prisma().update(
            where=_key(run.user_id, run.schedule_key),
            data={**fields, **({"notifiedAt": now} if told else {})},
        )
        if updated is not None:
            return updated
    await ScheduledChatStreak.prisma().delete_many(
        where={"userId": run.user_id, "scheduleKey": run.schedule_key}
    )
    return await ScheduledChatStreak.prisma().create(
        data={
            **fields,
            "userId": run.user_id,
            "scheduleKey": run.schedule_key,
            "notifiedAt": now if told else None,
        }
    )


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
