"""Answer a task's hire and raise cards from the board (``approvals.py``)."""

from backend.api.features.tasks import approvals, contract_db, db
from backend.api.features.tasks.events import publish_contract_update
from backend.api.features.tasks.models import Actor, CardDecision
from backend.copilot.gate.card_approval import record_card_decisions


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
    if not task.work_session_id:
        raise ValueError("This task has no chat to approve the cards in.")
    lines = approvals.decision_lines(task.approvals, decisions)
    if not lines:
        raise ValueError("These cards are no longer waiting on you.")
    decided = "\n".join(lines)
    # The same record the chat route keeps when the card is clicked there:
    # this request is the user's own, so the confirm it unlocks is theirs.
    await record_card_decisions(user_id, task.work_session_id, decided)
    update = await contract_db.drop_approvals(
        user_id, task_id, [d.confirmation_id for d in decisions]
    )
    await publish_contract_update(user_id, update)
    text = approvals.answer_text(task.approvals, decisions)
    await db.answer_input(user_id, task_id, text, None, Actor.user(user_id))
    return f"The user answered the cards on the task board:\n{decided}"
