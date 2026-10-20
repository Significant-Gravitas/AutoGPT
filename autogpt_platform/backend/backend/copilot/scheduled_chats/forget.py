"""Dropping a scheduled chat's failure streak once its owner deletes it or
switches it off, and withdrawing its pause alert.

Apart from ``service`` and ``alerts`` so the routines layer and the copilot's
own tools can call it: both reach the copilot's tools, and the tools reach the
routines layer.
"""

import logging

from prisma.models import ExpertRoutine, ScheduledChatStreak
from prisma.types import ExpertRoutineWhereInput

from backend.data.alerts import resolve_alert_condition

logger = logging.getLogger(__name__)


def alert_key(schedule_key: str) -> str:
    return f"scheduled_chat:{schedule_key}"


async def resolve_alert(user_id: str, schedule_key: str) -> None:
    """Withdraw the chat's pause alert. Best effort: one left open only means
    an email about a chat that runs again."""
    try:
        await resolve_alert_condition(user_id, alert_key(schedule_key))
    except Exception:
        logger.warning(
            "Could not resolve the alert for scheduled chat %s",
            schedule_key,
            exc_info=True,
        )


def owned_routines(user_id: str) -> ExpertRoutineWhereInput:
    """The account's own routines and its experts'. A streak's key is the
    owner's routine only if it names one of these."""
    return {"OR": [{"userId": user_id}, {"Expert": {"is": {"ownerUserId": user_id}}}]}


async def forget_scheduled_chat(user_id: str, schedule_key: str) -> None:
    """Drop the streak of a follow-up its owner deleted. Kept, a paused one
    would stay listed as paused and under Needs you, with a Resume that says
    it runs again, and its alert could still email about it.

    A routine's streak is left alone: its paused jobs outlive any one
    schedule id, and a streak dropped from under them would stop the routine
    without a word."""
    if await ExpertRoutine.prisma().find_first(
        where={"id": schedule_key, **owned_routines(user_id)}
    ):
        return
    await ScheduledChatStreak.prisma().delete_many(
        where={"userId": user_id, "scheduleKey": schedule_key}
    )
    await resolve_alert(user_id, schedule_key)


async def forget_routine(user_id: str, routine_id: str) -> None:
    """Its jobs are deleted as it is switched off, and switching it on again
    makes new ones that fire. Kept, a paused one would stay listed as paused,
    with a Resume that says it runs again, and its alert could still email
    about it."""
    await ScheduledChatStreak.prisma().delete_many(
        where={"userId": user_id, "scheduleKey": routine_id}
    )
    await resolve_alert(user_id, routine_id)
