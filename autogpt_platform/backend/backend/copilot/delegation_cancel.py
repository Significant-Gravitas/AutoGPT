"""Stop the threads a cancelled turn handed work to.

A turn that delegates, or opens a sub-session, waits on another chat's turn.
Stopping the waiting turn used to leave that other turn running with nobody
left to read its result. The executor calls this once a turn ends cancelled,
so any cancel, whichever path sent it, reaches the children too. Each child's
own cancel then reaches its children the same way.
"""

import logging
from datetime import datetime, timedelta

from backend.copilot.executor.utils import enqueue_cancel_task
from backend.copilot.model import CHAT_STATUS_QUEUED, CHAT_STATUS_RUNNING
from backend.copilot.turn_queue import cancel_queued_turn
from backend.data.db_accessors import chat_db

logger = logging.getLogger(__name__)

# The turn's start is the executor's clock, a message's creation the
# database's: allow for the two drifting apart a little.
_CLOCK_SKEW = timedelta(seconds=5)


async def cancel_delegated_children(
    session_id: str, user_id: str | None, since: datetime
) -> list[str]:
    """Cancel the running or queued threads the turn *session_id* started at
    *since* set working; their ids.

    A thread counts when this chat sent the message its current turn runs on
    (``from_session_id``) during this turn. A thread an earlier turn left
    working, or a turn the user started in the expert's own thread, keeps
    going. A hand-off is left alone: it transferred the task for good, so
    nobody is waiting on it and the receiving expert owns it now.
    """
    if not user_id:
        return []
    children = [
        child_id
        for child_id in await _live_children(session_id, user_id)
        if await _sent_this_turn(child_id, session_id, since)
    ]
    for child_id in children:
        # A queued turn has no executor to signal: flipping it back to idle
        # is the whole cancel. Otherwise the fanout reaches its worker.
        if not await cancel_queued_turn(user_id=user_id, session_id=child_id):
            await enqueue_cancel_task(child_id)
    if children:
        logger.info(f"Cancelled {len(children)} child thread(s) of {session_id}")
    return children


async def _live_children(session_id: str, user_id: str) -> list[str]:
    db = chat_db()
    sessions = [
        *await db.list_chat_sessions_by_status(
            user_id=user_id, status=CHAT_STATUS_RUNNING
        ),
        *await db.list_chat_sessions_by_status(
            user_id=user_id, status=CHAT_STATUS_QUEUED
        ),
    ]
    return [
        s.session_id
        for s in sessions
        if s.metadata.delegated_by_session_id == session_id
        and s.metadata.handed_off_from_expert_id is None
    ]


async def _sent_this_turn(child_id: str, session_id: str, since: datetime) -> bool:
    """Whether *session_id* sent the message *child_id*'s turn runs on, at or
    after *since*."""
    message = await chat_db().get_latest_user_message_in_session(child_id)
    if message is None or (message.metadata or {}).get("from_session_id") != (
        session_id
    ):
        return False
    return message.created_at is None or message.created_at >= since - _CLOCK_SKEW
