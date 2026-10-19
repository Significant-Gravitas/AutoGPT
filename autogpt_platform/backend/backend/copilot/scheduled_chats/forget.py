"""Dropping a routine's failure streak once its owner switches it off, and
withdrawing a scheduled chat's pause alert.

Apart from ``service`` and ``alerts`` so the routines layer can call it: both
reach the copilot's tools, and the tools reach the routines layer.
"""

import logging

from prisma.models import ScheduledChatStreak

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


async def forget_routine(user_id: str, routine_id: str) -> None:
    """Its jobs are deleted as it is switched off, and switching it on again
    makes new ones that fire. Kept, a paused one would stay listed as paused,
    with a Resume that says it runs again, and its alert could still email
    about it."""
    await ScheduledChatStreak.prisma().delete_many(
        where={"userId": user_id, "scheduleKey": routine_id}
    )
    await resolve_alert(user_id, routine_id)
