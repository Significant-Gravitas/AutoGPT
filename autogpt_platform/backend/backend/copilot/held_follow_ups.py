"""Follow-ups a turn ended without reading.

A message typed while a turn runs waits in the session's pending queue
(``backend.copilot.pending_messages``) for that turn to read it. Only a
running turn reads the queue, so a turn that ends first (an error, a Stop, a
pause at a limit, or simply before it got to them) leaves its follow-ups
there with nothing left to answer them. The turn's end records that here and
tells the client, which shows each one as not sent, with a Retry. A retry
names the follow-up by its id: the server takes that very message off the
queue and runs it as a turn, or, if another turn read it meanwhile, sends
nothing, so Otto never reads one message twice.
"""

import logging
import math
import time
from typing import Literal, get_args

from backend.copilot.pending_messages import (
    PENDING_TTL_SECONDS,
    PendingMessage,
    held_marker_key,
    peek_pending_count,
    peek_pending_messages,
    put_back_pending_message,
    take_pending_message_by_id,
    was_pending_message_read,
)
from backend.copilot.response_model import FollowUpHeldReason, StreamFollowUpHeld
from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import as_str

logger = logging.getLogger(__name__)

# The hold covers what was queued before the turn ended. The API pod stamps a
# follow-up's enqueue time and the executor stamps the hold, so their clocks
# may differ a little; a resend comes long after either.
_CLOCK_SLACK_SECONDS = 2.0


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
            await redis.delete(held_marker_key(session_id))
            return None
        await redis.set(
            held_marker_key(session_id),
            f"{reason}|{time.time()}",
            ex=PENDING_TTL_SECONDS,
        )
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


async def held_follow_up_reason(
    session_id: str, *, covering: list[PendingMessage] | None = None
) -> FollowUpHeldReason | None:
    """Why the queued follow-ups were left unread, or None if they were not.

    With *covering*, only when the hold covers each of those messages: one
    queued after the turn ended is not held, and offering it a Retry would
    send it twice. Only meaningful while no turn runs: a running turn reads
    the queue.
    """
    hold = await _read_hold(session_id)
    if hold is None:
        return None
    if covering is not None and any(m.enqueued_at > hold[1] for m in covering):
        return None
    return hold[0]


async def _read_hold(session_id: str) -> tuple[FollowUpHeldReason, float] | None:
    """The hold's reason, and the latest enqueue time a message it covers has."""
    redis = await get_redis_async()
    stored = as_str(await redis.get(held_marker_key(session_id)))
    if stored is None:
        return None
    reason_text, _, held_at = stored.partition("|")
    reason = next((r for r in get_args(FollowUpHeldReason) if r == reason_text), None)
    if reason is None:
        return None
    try:
        return reason, float(held_at) + _CLOCK_SLACK_SECONDS
    except ValueError:
        return reason, math.inf


FollowUpState = Literal["waiting", "read", "unknown"]


async def follow_up_state(session_id: str, pending_id: str) -> FollowUpState:
    """Where the follow-up *pending_id* stands: still ``waiting`` in the queue,
    already ``read`` by a turn, or ``unknown`` (expired unread, or never
    queued here). Only an unknown one can be sent again without risking Otto
    reading it twice."""
    waiting = await peek_pending_messages(session_id)
    if any(pending.id == pending_id for pending in waiting):
        return "waiting"
    if await was_pending_message_read(session_id, pending_id):
        return "read"
    return "unknown"


async def take_follow_up(session_id: str, pending_id: str) -> PendingMessage | None:
    """Take the follow-up *pending_id* off the queue to send it as a turn of
    its own. Atomic: of two resends of it, only one gets it. Returns it so a
    send that is refused can put it back; None when it no longer waits."""
    taken = await take_pending_message_by_id(session_id, pending_id)
    if taken is not None:
        logger.info(
            "Resent follow-up taken out of the queue",
            extra={
                "json_fields": {
                    "session_id": session_id,
                    "pending_id": pending_id,
                    "path": "resend",
                }
            },
        )
    return taken


async def restore_held_follow_up(session_id: str, taken: PendingMessage | None) -> None:
    """Put back a held follow-up whose resend was refused."""
    if taken is None:
        return
    try:
        await put_back_pending_message(session_id, taken)
    except Exception:
        logger.error(
            f"Could not put a refused resend back in session {session_id}",
            exc_info=True,
            extra={
                "json_fields": {
                    "session_id": session_id,
                    "pending_id": taken.id,
                    "path": "resend_put_back_failed",
                }
            },
        )
