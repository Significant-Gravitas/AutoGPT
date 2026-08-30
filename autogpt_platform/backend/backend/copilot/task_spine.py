"""Open a DelegatedTask receipt when a chat turn starts a run.

Runs inside the copilot executor, which has no Prisma client, so the write
goes through the DatabaseManager RPC. Best-effort throughout: a receipt that
fails to open must never stop the run the user actually asked for — the run
simply lands without a task, exactly as it did before the spine existed.
"""

import logging

from backend.copilot.model import ChatSession
from backend.util.clients import get_database_manager_async_client

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


def _describe_run(agent_name: str, inputs: dict) -> str:
    """A human-readable record of what was asked for, so the receipt still
    means something after the agent's inputs have moved on."""
    if not inputs:
        return f"Run {agent_name}."
    listed = "\n".join(f"- {name}: {value!r}" for name, value in inputs.items())
    return f"Run {agent_name} with:\n{listed}"[:_MAX_SPEC_LENGTH]
