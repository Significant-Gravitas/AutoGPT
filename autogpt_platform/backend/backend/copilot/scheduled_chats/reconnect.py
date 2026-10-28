"""Connecting a provider is the fix for a scheduled chat failing on it
(SECRT-2879).

The integrations routes call ``resume_after_reconnect_in_background`` after
every successful connect, so the fix counts wherever the owner made it: the
card's Reconnect button, the integrations page, or a chat's connect prompt.
"""

import asyncio
import logging

from prisma.models import ScheduledChatStreak

from backend.copilot.scheduled_chats.service import resume_scheduled_chat

logger = logging.getLogger(__name__)

# Held so a resume that is still running is not garbage-collected mid-flight.
_resumes: set[asyncio.Task[None]] = set()


def resume_after_reconnect_in_background(
    user_id: str, provider: str
) -> asyncio.Task[None]:
    """Start ``resume_after_reconnect`` without making the connect wait on it.

    Resuming goes through the scheduler, whose calls can take minutes when it
    is unwell, and the credential is saved either way.
    """
    task = asyncio.create_task(resume_after_reconnect(user_id, provider))
    _resumes.add(task)
    task.add_done_callback(_resumes.discard)
    return task


async def resume_after_reconnect(user_id: str, provider: str) -> None:
    """Resume every scheduled chat paused on *provider*, and forget the ones
    only told about it. Never raises."""
    try:
        # *provider* is the connect route's ProviderName value, and a streak's
        # is what the failing tool put in ErrorResponse.details: the same
        # vocabulary, which must stay the same. A failure that named no
        # provider is not resumed here; Resume or a good run clears it.
        rows = await ScheduledChatStreak.prisma().find_many(
            where={"userId": user_id, "provider": provider, "failureKind": "connection"}
        )
    except Exception:
        logger.warning(
            "Could not load the scheduled chats waiting on %s", provider, exc_info=True
        )
        return
    for row in rows:
        try:
            await _clear(user_id, row)
        except Exception:
            logger.warning(
                "Could not resume scheduled chat %s after %s was reconnected",
                row.scheduleKey,
                provider,
                exc_info=True,
            )


async def _clear(user_id: str, row: ScheduledChatStreak) -> None:
    if row.pausedAt is None:
        # Only told: its next run proves the fix. Until then nothing says the
        # chat is failing, and a failure that persists counts afresh.
        if await ScheduledChatStreak.prisma().delete_many(
            where={"id": row.id, "pausedAt": None}
        ):
            return
        # A run paused it after this read it: that run failed before the fix,
        # and deleting the row would leave its jobs paused with nothing to
        # list or resume them.
    await resume_scheduled_chat(user_id, row.scheduleKey)
