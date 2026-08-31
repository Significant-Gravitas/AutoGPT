"""Open a DelegatedTask receipt when a chat turn starts a run.

Runs inside the copilot executor, which has no Prisma client, so the write
goes through the DatabaseManager RPC. Best-effort throughout: a receipt that
fails to open must never stop the run the user actually asked for — the run
simply lands without a task, exactly as it did before the spine existed.
"""

import logging

from backend.copilot.model import ChatSession, ChatSessionInfo, get_chat_session
from backend.util.clients import get_database_manager_async_client
from backend.util.exceptions import TaskDelegationRefusedError

logger = logging.getLogger(__name__)

# The spec is the model's own description of the run, so it is bounded here
# rather than trusted; the column is 20k but a chat turn has no business
# writing that much.
_MAX_SPEC_LENGTH = 4_000


async def open_task_for_run(
    user_id: str,
    session: ChatSession,
    *,
    agent_name: str,
    inputs: dict,
) -> str | None:
    """Open a QUEUED receipt for a run about to start. Returns its id, or
    None when the receipt could not be created.

    ``ownerId`` is the session's expert — null for Autopilot, matching
    ``ChatSession.expertId``, so an Autopilot run still gets a receipt and
    simply has no expert to attribute it to.
    """
    try:
        task = await get_database_manager_async_client().create_delegated_task(
            user_id=user_id,
            title=agent_name,
            spec=_describe_run(agent_name, inputs),
            owner_id=session.expert_id,
            origin_session_id=session.session_id,
            created_by_type="USER",
            created_by_id=user_id,
        )
    except Exception:
        logger.warning(
            "Failed to open a delegated task for session #%s",
            session.session_id,
            exc_info=True,
        )
        return None
    return task.id


async def mark_task_working(user_id: str, task_id: str) -> None:
    """Flip QUEUED → WORKING once the executor has accepted the run."""
    try:
        await get_database_manager_async_client().mark_delegated_task_working(
            user_id=user_id, task_id=task_id
        )
    except Exception:
        logger.warning("Failed to mark task #%s working", task_id, exc_info=True)


async def fail_task(user_id: str, task_id: str, reason: str) -> None:
    """Close a receipt whose run never started. Without this the task would
    sit in QUEUED forever and the Tasks tab would show phantom active work."""
    try:
        await get_database_manager_async_client().close_delegated_task(
            user_id=user_id,
            task_id=task_id,
            succeeded=False,
            outcome_summary=reason,
        )
    except Exception:
        logger.warning("Failed to close task #%s", task_id, exc_info=True)


async def settle_task_for_turn(
    user_id: str | None,
    session: ChatSession,
    *,
    error_message: str | None,
) -> None:
    """Close a delegated worker session's receipt when its turn ends.

    ``delegate_to_expert`` settles the receipt only while its waiter is still
    listening; a turn that outlives the wait window (or is never polled)
    would otherwise sit WORKING forever. This runs in the executor at turn
    completion, so the receipt closes no matter who is watching.

    Skips anything that is not this turn's to settle: a receipt already
    closed or parked WAITING_USER by an escalation, and one handed off to a
    different owner mid-turn. Cancellation is owned by the cancel route.
    """
    task_id = session.metadata.delegated_task_id
    if not task_id or user_id is None:
        return
    if error_message == "Operation cancelled":
        return
    client = get_database_manager_async_client()
    try:
        detail = await client.get_delegated_task(user_id, task_id)
        task = detail.task if detail else None
        if task is None or task.status not in ("QUEUED", "WORKING"):
            return
        if (task.owner.id if task.owner else None) != session.expert_id:
            return
        if error_message:
            await client.close_delegated_task(
                user_id=user_id,
                task_id=task_id,
                succeeded=False,
                outcome_summary=f"The delegated turn failed: {error_message}",
            )
            return
        await client.report_delegated_task(
            user_id, task_id, outcome_summary=await _final_answer(session, user_id)
        )
    except TaskDelegationRefusedError as e:
        logger.info("Leaving task #%s open: %s", task_id, e)
    except Exception:
        logger.warning("Failed to settle task #%s", task_id, exc_info=True)


async def _final_answer(session: ChatSession, user_id: str) -> str:
    """The turn's closing assistant message, re-read from storage — the
    session object the executor holds predates the turn's writes."""
    fresh = await get_chat_session(session.session_id, user_id)
    for message in reversed(fresh.messages if fresh else []):
        if message.role == "assistant" and message.content and message.content.strip():
            return " ".join(message.content.split())
    return "The delegated work finished."


async def record_mid_task_instruction(
    user_id: str, session: ChatSession | ChatSessionInfo, text: str
) -> None:
    """A user message into a session working a delegated task is a mid-task
    instruction — append it to the task's timeline so the next model turn
    (via ``build_task_context``) and the task drawer both see it.
    Best-effort; the message itself still reaches the model either way."""
    task_id = session.metadata.delegated_task_id
    note = " ".join(text.split()) if text else ""
    if not task_id or not note:
        return
    try:
        await get_database_manager_async_client().append_task_amendment(
            user_id, task_id, note=note, by="user", kind="note"
        )
    except Exception:
        logger.warning(
            "Failed to record mid-task instruction on task #%s",
            task_id,
            exc_info=True,
        )


async def build_task_context(user_id: str | None, session: ChatSession) -> str:
    """The delegated-task briefing injected as ``<current_task>`` into a
    worker session's turn. Empty string when the session has no open task —
    the block is simply omitted."""
    task_id = session.metadata.delegated_task_id
    if not task_id or user_id is None:
        return ""
    try:
        detail = await get_database_manager_async_client().get_delegated_task(
            user_id, task_id
        )
    except Exception:
        logger.warning(
            "Failed to load task #%s for context injection", task_id, exc_info=True
        )
        return ""
    task = detail.task if detail else None
    if task is None or task.status not in ("QUEUED", "WORKING"):
        return ""
    lines = [
        f"You are working delegated task '{task.title}' "
        f"(task_id: {task.id}, status: {task.status}).",
        f"Spec: {task.spec}",
    ]
    instructions = [a for a in task.amendments if a.kind == "note" and a.by == "user"]
    if instructions:
        lines.append("The user added instructions mid-task:")
        lines.extend(
            f"- [{a.at.isoformat(timespec='minutes')}] {a.note}" for a in instructions
        )
        lines.append(
            "Fold these into the work in progress — they refine the spec above."
        )
    return "\n".join(lines)


def _describe_run(agent_name: str, inputs: dict) -> str:
    """A human-readable record of what was asked for, so the receipt still
    means something after the agent's inputs have moved on."""
    if not inputs:
        return f"Run {agent_name}."
    listed = "\n".join(f"- {name}: {value!r}" for name, value in inputs.items())
    return f"Run {agent_name} with:\n{listed}"[:_MAX_SPEC_LENGTH]
