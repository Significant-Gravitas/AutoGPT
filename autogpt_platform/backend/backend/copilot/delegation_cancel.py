"""Stop the threads a cancelled turn handed work to.

A turn that delegates, or opens a sub-session, waits on another chat's turn.
Stopping the waiting turn used to leave that other turn running with nobody
left to read its result. The executor calls this once a turn ends cancelled,
so any cancel, whichever path sent it, reaches the children too. Each child's
own cancel then reaches its children the same way.
"""

import logging

from backend.copilot.executor.utils import enqueue_cancel_task
from backend.copilot.model import CHAT_STATUS_QUEUED, CHAT_STATUS_RUNNING
from backend.copilot.turn_queue import cancel_queued_turn
from backend.data.db_accessors import chat_db

logger = logging.getLogger(__name__)


async def cancel_delegated_children(session_id: str, user_id: str | None) -> list[str]:
    """Cancel every running or queued thread *session_id* opened; their ids.

    A hand-off is left alone: it transferred the task for good, so nobody is
    waiting on it and the receiving expert owns it now.
    """
    if not user_id:
        return []
    children = await _live_children(session_id, user_id)
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
