"""Settle a task run from its page when a turn in its chat ends.

A chat opened by ``POST /tasks/{id}/run`` has nobody waiting on it, unlike a
background worker whose parent chat is woken (``background_delivery``), so
the executor calls this at the end of every turn. A turn that handed work off
to background subtasks leaves the task open: the wake that brings their
results back is a turn of this chat too, and settles it.
"""

import logging

from backend.copilot.background_task import BACKGROUND_TIMEOUT_MESSAGE
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.model import ChatSession, get_chat_session
from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.data.db_accessors import background_task_db, task_run_db
from backend.data.task_run import TurnEnding

logger = logging.getLogger(__name__)


async def settle_task_run_turn(entry: CoPilotExecutionEntry, error: str | None) -> None:
    """Best-effort: a turn's end must not fail on the task's record."""
    if not entry.user_id:
        return
    try:
        session = await get_chat_session(entry.session_id, entry.user_id)
        task_id = session.metadata.work_task_id if session else None
        if not task_id:
            return
        ending = turn_ending(error)
        if ending != "cancelled" and await background_task_db().has_open_handoffs(
            entry.user_id, entry.session_id
        ):
            return
        await task_run_db().settle_task_run(
            task_id, entry.session_id, entry.user_id, ending, last_reply(session)
        )
    except Exception:
        logger.exception(f"Could not settle the task run of chat {entry.session_id}")


async def note_task_run_started(session: ChatSession, user_id: str) -> None:
    """A turn in a task's run chat is starting: a run that waited for a free
    slot is worked on now, so the task stops saying it is queued.
    Best-effort, and one call only for a run chat."""
    task_id = session.metadata.work_task_id
    if not task_id:
        return
    try:
        await task_run_db().clear_run_queued(task_id, user_id)
    except Exception:
        logger.warning(f"Could not clear the queued mark of task {task_id}")


def turn_ending(error: str | None) -> TurnEnding:
    if error is None:
        return "completed"
    if error == CANCELLED_MESSAGE:
        return "cancelled"
    if error == BACKGROUND_TIMEOUT_MESSAGE:
        return "timed_out"
    return "failed"


# Enough for the reason's sentence or two; the rest never crosses the RPC.
_REPLY_MAX_CHARS = 2_000


def last_reply(session: ChatSession | None) -> str | None:
    """What the worker said last in this turn: the newest assistant text
    after the newest user message, or None when it said nothing."""
    for message in reversed(session.messages if session else []):
        if message.role == "user":
            return None
        if message.role == "assistant" and (message.content or "").strip():
            return (message.content or "")[:_REPLY_MAX_CHARS]
    return None
