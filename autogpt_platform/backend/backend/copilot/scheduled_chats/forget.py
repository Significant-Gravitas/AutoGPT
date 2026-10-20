"""Dropping a routine's failure streak once its owner switches it off.

Apart from ``service`` so the routines layer can call it: ``service`` reaches
the copilot's tools, and the tools reach the routines layer.
"""

from prisma.models import ScheduledChatStreak


async def forget_routine(user_id: str, routine_id: str) -> None:
    """Its jobs are deleted as it is switched off, and switching it on again
    makes new ones that fire. Kept, a paused one would stay listed as paused,
    with a Resume that says it runs again."""
    await ScheduledChatStreak.prisma().delete_many(
        where={"userId": user_id, "scheduleKey": routine_id}
    )
