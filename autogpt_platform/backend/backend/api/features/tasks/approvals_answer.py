"""Answer a task's hire and raise cards from the board (``approvals.py``).

A hire proposed outside any task's chat is its own task
(``copilot/tools/hire_task.py``) with no work chat: its card lives in the
user's chat, remembered here, so the answer goes to that chat. Declining
cancels the hire's task; approving leaves it hiring until the confirm in the
chat completes it.
"""

import logging
from datetime import UTC, datetime

from backend.api.features.tasks import approvals, contract_db, db
from backend.api.features.tasks.events import publish_contract_update
from backend.api.features.tasks.models import Actor, CardDecision, Task, TaskStatus
from backend.copilot.gate.card_approval import record_card_decisions
from backend.copilot.sdk.session_waiter import run_copilot_turn_via_queue
from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_HIRE_CHAT_KEY = "tasks:hire_chat:"
_HIRE_CHAT_TTL_SECONDS = 7 * 24 * 3600


async def answer_cards(
    user_id: str, task_id: str, decisions: list[CardDecision]
) -> str:
    """Record the user's decisions on the task's cards and close its open
    question. Returns the message that resumes the worker.

    Raises LookupError for an unknown task, ValueError when none of the
    decisions names a card the task is still waiting on.
    """
    task = await db.get_task(user_id, task_id)
    if task is None:
        raise LookupError("Task not found")
    chat = task.work_session_id or await hire_chat(task_id)
    if not chat:
        raise ValueError("This task has no chat to approve the cards in.")
    lines = approvals.decision_lines(task.approvals, decisions)
    if not lines:
        raise ValueError("These cards are no longer waiting on you.")
    decided = "\n".join(lines)
    # The same record the chat route keeps when the card is clicked there:
    # this request is the user's own, so the confirm it unlocks is theirs.
    await record_card_decisions(user_id, chat, decided)
    update = await contract_db.drop_approvals(
        user_id, task_id, [d.confirmation_id for d in decisions]
    )
    await publish_contract_update(user_id, update)
    text = approvals.answer_text(task.approvals, decisions)
    await db.answer_input(user_id, task_id, text, None, Actor.user(user_id))
    if task.work_session_id is None:
        await _settle_hire(user_id, task, decisions, chat, decided)
    return f"The user answered the cards on the task board:\n{decided}"


async def remember_hire_chat(task_id: str, session_id: str) -> None:
    """Note the chat holding a hire task's card (``hire_task``)."""
    redis = await get_redis_async()
    await redis.set(_HIRE_CHAT_KEY + task_id, session_id, ex=_HIRE_CHAT_TTL_SECONDS)


async def hire_chat(task_id: str) -> str | None:
    redis = await get_redis_async()
    session_id = await redis.get(_HIRE_CHAT_KEY + task_id)
    if isinstance(session_id, bytes):
        return session_id.decode()
    return session_id or None


async def _settle_hire(
    user_id: str,
    task: Task,
    decisions: list[CardDecision],
    chat: str,
    decided: str,
) -> None:
    """A hire task's own card was answered: declined cancels it, approved
    shows it hiring (live, so nothing dispatches it) until the confirm.
    Either way the chat gets the line its card would have sent."""
    answered = {d.confirmation_id: d.approved for d in decisions}
    held = [a for a in task.approvals if a.confirmation_id in answered]
    approved = [a for a in held if answered[a.confirmation_id]]
    if approved:
        await db.set_live(
            task.id, True, f"Hiring {approved[0].name}", datetime.now(UTC)
        )
    else:
        await db.set_task_status(
            user_id,
            task.id,
            TaskStatus.CANCELLED,
            Actor.user(user_id),
            note="Declined on the task board",
        )
    try:
        await run_copilot_turn_via_queue(
            session_id=chat,
            user_id=user_id,
            message=decided,
            timeout=0,
            tool_call_id=f"task_approvals:{task.id}",
            tool_name="task_approvals",
            allow_queue=True,
        )
    except Exception:
        logger.warning(
            f"Could not send the board's card answer to chat {chat}", exc_info=True
        )
