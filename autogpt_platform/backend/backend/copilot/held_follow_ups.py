"""Follow-ups a turn ended without reading.

A message typed while a turn runs waits in the session's pending queue
(``backend.copilot.pending_messages``) for that turn to read it. Only a
running turn reads the queue, so a turn that ends first (an error, a Stop, a
pause at a limit, or simply before it got to them) leaves its follow-ups
there with nothing left to answer them. The turn's end records that here and
tells the client, which shows each one as not sent, with a Retry. A retry is
an ordinary send of the same text: it takes the waiting copy out first, so the
turn it starts reads the message once rather than twice.
"""

import logging
from typing import get_args

from backend.copilot.pending_messages import (
    PENDING_TTL_SECONDS,
    PendingMessage,
    peek_pending_count,
    push_pending_message,
    take_pending_message,
)
from backend.copilot.response_model import FollowUpHeldReason, StreamFollowUpHeld
from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import as_str

logger = logging.getLogger(__name__)

# Same slot and lifetime as the pending queue it describes.
_HELD_KEY_PREFIX = "copilot:pending:held:"


def _held_key(session_id: str) -> str:
    return f"{_HELD_KEY_PREFIX}{{{session_id}}}"


def held_reason_for(
    *, failed: bool, cancelled: bool, paused: bool
) -> FollowUpHeldReason:
    """How the turn ended, as the client explains a held follow-up."""
    if cancelled:
        return "stopped"
    if failed:
        return "error"
    return "paused" if paused else "ended"


async def hold_follow_ups_at_turn_end(
    session_id: str, turn_id: str, reason: FollowUpHeldReason
) -> StreamFollowUpHeld | None:
    """Record whether the turn that just ended left follow-ups unread.

    Called once the session no longer reads as running, when no new
    follow-up can join the queue. Returns the stream entry to publish before
    the turn's finish, or None when nothing waits. Never raises: it must not
    break the turn's completion.
    """
    try:
        count = await peek_pending_count(session_id)
        redis = await get_redis_async()
        if count == 0:
            await redis.delete(_held_key(session_id))
            return None
        await redis.set(_held_key(session_id), reason, ex=PENDING_TTL_SECONDS)
    except Exception:
        logger.warning(
            f"Could not check session {session_id} for unread follow-ups",
            exc_info=True,
        )
        return None
    logger.warning(
        f"Turn ended with {count} follow-up(s) unread; they wait for a resend",
        extra={
            "json_fields": {
                "session_id": session_id,
                "turn_id": turn_id,
                "reason": reason,
                "count": count,
                "path": "turn_end_held",
            }
        },
    )
    return StreamFollowUpHeld(reason=reason, count=count)


async def held_follow_up_reason(session_id: str) -> FollowUpHeldReason | None:
    """Why the queued follow-ups were left unread, or None if they were not.

    Only meaningful while no turn runs: a running turn reads the queue.
    """
    redis = await get_redis_async()
    stored = as_str(await redis.get(_held_key(session_id)))
    return next((r for r in get_args(FollowUpHeldReason) if r == stored), None)


async def take_held_follow_up(session_id: str, message: str) -> PendingMessage | None:
    """Take the held copy of *message* out of the queue as it is sent again.

    Only while follow-ups are held: then the send starts a turn, whose start
    would otherwise read the waiting copy as well. Returns the copy so a send
    that is refused can put it back; None when nothing was taken.
    """
    try:
        if await held_follow_up_reason(session_id) is None:
            return None
        taken = await take_pending_message(session_id, message)
    except Exception:
        logger.warning(
            f"Could not take a resent follow-up out of session {session_id}",
            exc_info=True,
        )
        return None
    if taken is not None:
        logger.info(
            "Resent follow-up taken out of the queue",
            extra={"json_fields": {"session_id": session_id, "path": "resend"}},
        )
    return taken


async def restore_held_follow_up(session_id: str, taken: PendingMessage | None) -> None:
    """Put back a held follow-up whose resend was refused."""
    if taken is None:
        return
    try:
        await push_pending_message(session_id, taken)
    except Exception:
        logger.error(
            f"Could not put a refused resend back in session {session_id}",
            exc_info=True,
        )
