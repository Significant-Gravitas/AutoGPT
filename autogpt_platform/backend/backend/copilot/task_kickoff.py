"""Put an owner to work on a task that nobody is driving yet.

Every other path into a DelegatedTask opens the receipt *for* a turn that is
already running — ``task_spine`` for a chat's own run, ``delegate_to_expert``
for a teammate's. A task created outside a conversation (an office pack's
intro task) has no such turn, so it lands QUEUED with no
``originSessionId``: no thread to run in, and none for the overseer's stall
retry to nudge. This module is the missing half — it opens the worker
session, claims the task into it, and dispatches the first turn.

Claim-before-dispatch is deliberate. The claim is the QUEUED→WORKING
compare-and-set, so two callers racing the same task (a hire and the
overseer sweep) can only produce one worker; and if the dispatch then fails,
the task is already bound to a session the overseer can retry into, rather
than being stranded again.
"""

import logging

from backend.copilot.executor.utils import schedule_chat_turn
from backend.copilot.model import create_chat_session, delete_chat_session
from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)


async def start_task_in_new_session(
    user_id: str,
    *,
    task_id: str,
    title: str,
    expert_id: str | None,
) -> str | None:
    """Open a session for *task_id*'s owner and dispatch its first turn.

    Returns the session id once the task is claimed, or None when the task
    was already claimed by someone else or the kickoff could not be opened.
    The spec is not repeated in the message: the session carries
    ``delegated_task_id``, so the per-turn ``<current_task>`` block
    (:func:`task_spine.build_task_context`) already hands the model the
    title, spec and any mid-task instructions.
    """
    try:
        session = await create_chat_session(
            user_id,
            dry_run=False,
            expert_id=expert_id,
            delegated_task_id=task_id,
            origin="automation",
        )
    except Exception:
        logger.warning(
            "Could not open a worker session for task #%s", task_id, exc_info=True
        )
        return None

    client = get_database_manager_async_client()
    if not await client.claim_task_for_session(user_id, task_id, session.session_id):
        logger.info("Task #%s was already claimed; dropping the kickoff", task_id)
        await _discard_session(session.session_id, user_id)
        return None

    try:
        await schedule_chat_turn(
            session_id=session.session_id,
            user_id=user_id,
            message=_kickoff_message(task_id, title),
        )
    except Exception:
        logger.warning(
            "Could not dispatch the kickoff turn for task #%s in session #%s",
            task_id,
            session.session_id,
            exc_info=True,
        )
    return session.session_id


async def _discard_session(session_id: str, user_id: str) -> None:
    """A session that lost the claim has no task and never ran a turn —
    leaving it would show the user an empty thread. Best-effort."""
    try:
        await delete_chat_session(session_id, user_id)
    except Exception:
        logger.warning(
            "Could not discard unused worker session #%s", session_id, exc_info=True
        )


def _kickoff_message(task_id: str, title: str) -> str:
    return (
        f"[Starting task '{title}' (task_id: {task_id})]\n\n"
        "This task is yours and it starts now — the full spec is in your "
        "current-task briefing. Do the work, using your own workflows and "
        "integrations. If something only the user can answer blocks you, ask "
        "with escalate_task; when the work is done, close it out with "
        f"report_task (task_id: {task_id})."
    )
