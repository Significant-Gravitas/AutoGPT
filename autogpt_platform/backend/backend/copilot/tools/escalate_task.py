"""Escalate a blocked DelegatedTask up the chain.

Two routes up:

- ``target="user"`` flips the task to WAITING_USER and records the question —
  and this session's id — on the task's timeline. Home's "Needs You" surfaces
  the question as a card; the user's answer is delivered back into this
  session as a pending message and the task resumes WORKING.
- ``target="manager"`` keeps the task WORKING and drops the question into the
  delegator's working session (the parent task's origin session) as a pending
  message, so the expert or AutoPilot who delegated the task can answer from
  context instead of interrupting the user. Refused on root tasks.
"""

import logging
from typing import Any

from backend.api.features.tasks.models import DelegatedTask, TaskEscalationTarget
from backend.copilot.model import ChatSession
from backend.copilot.pending_message_helpers import queue_user_message
from backend.util.clients import get_database_manager_async_client
from backend.util.exceptions import TaskDelegationRefusedError, TaskUpdateConflictError

from .base import BaseTool
from .models import ErrorResponse, TaskUpdateResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_MANAGER_QUESTION_MAX = 800


class EscalateTaskTool(BaseTool):
    """Ask the user or the delegating manager a blocking question."""

    @property
    def name(self) -> str:
        return "escalate_task"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Escalate a delegated task when you are blocked. "
            'target="user" (default) pauses the task (WAITING_USER) and puts '
            "your question on the user's Home; their answer comes back into "
            'this conversation. target="manager" sends the question to '
            "whoever delegated this task to you — use it when they likely "
            "hold the answer (scope, stack, decisions already made upstream) "
            "so the user is not interrupted; the task keeps working. Ask ONE "
            "clear question; offer options when the choice is enumerable."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The delegated task that is blocked.",
                },
                "question": {
                    "type": "string",
                    "description": (
                        "The single decision you need, as one short direct "
                        "question answerable without opening the task — it "
                        "is shown verbatim on the user's activity feed."
                    ),
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional short answers the user can pick with one "
                        'click instead of typing (target="user" only).'
                    ),
                    "default": [],
                },
                "target": {
                    "type": "string",
                    "enum": ["user", "manager"],
                    "description": (
                        "Who to escalate to: the user (pauses the task) or "
                        "the teammate/AutoPilot session that delegated this "
                        "task (task keeps working)."
                    ),
                    "default": "user",
                },
            },
            "required": ["task_id", "question"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        task_id: str = "",
        question: str = "",
        options: list[str] | None = None,
        target: TaskEscalationTarget = "user",
        **kwargs,
    ) -> ToolResponseBase:
        if user_id is None:
            return self._error("Authentication required", session)
        if not task_id.strip():
            return self._error("task_id is required", session)
        if not question.strip():
            return self._error("question is required", session)
        if target not in ("user", "manager"):
            return self._error('target must be "user" or "manager"', session)

        try:
            task = await get_database_manager_async_client().escalate_delegated_task(
                user_id,
                task_id.strip(),
                question=question.strip(),
                options=options or [],
                session_id=session.session_id,
                target=target,
            )
        except (TaskDelegationRefusedError, TaskUpdateConflictError) as e:
            return self._error(str(e), session)

        if target == "manager":
            await _notify_manager(user_id, task, question.strip(), session)
            logger.info(
                "Routing: session %s escalated task %s to its delegator",
                session.session_id,
                task.id,
            )
            return TaskUpdateResponse.from_task(
                task,
                action="escalation",
                message=(
                    f"Your question about '{task.title}' was sent to the "
                    "session that delegated it. The task stays WORKING; "
                    "their guidance arrives here as a follow-up message. "
                    "Continue any unblocked work and end your turn."
                ),
                session_id=session.session_id,
            )

        logger.info(
            "Routing: session %s escalated task %s to the user",
            session.session_id,
            task.id,
        )
        return TaskUpdateResponse.from_task(
            task,
            action="escalation",
            message=(
                f"Task '{task.title}' is now waiting on the user. Your "
                "question is on their Home screen; their answer will arrive "
                "in this conversation. End your turn after any remaining "
                "unblocked work."
            ),
            session_id=session.session_id,
        )

    def _error(self, message: str, session: ChatSession) -> ErrorResponse:
        return ErrorResponse(message=message, session_id=session.session_id)


async def _notify_manager(
    user_id: str,
    task: DelegatedTask,
    question: str,
    session: ChatSession,
) -> None:
    """Deliver the question into the delegator's working session.

    The parent task's ``origin_session_id`` is where the delegator was
    working when they opened this subtask, so the question belongs there.
    Best-effort — a lost notification still leaves the escalation entry on
    the task timeline, which the overseer and Team page can surface.
    """
    if task.parent_task_id is None:
        return
    try:
        detail = await get_database_manager_async_client().get_delegated_task(
            user_id, task.parent_task_id
        )
        parent_session_id = detail.task.origin_session_id if detail else None
        if parent_session_id is None or parent_session_id == session.session_id:
            return
        owner = task.owner.name if task.owner else "A teammate"
        trimmed = " ".join(question.split())
        if len(trimmed) > _MANAGER_QUESTION_MAX:
            trimmed = f"{trimmed[:_MANAGER_QUESTION_MAX]}…"
        await queue_user_message(
            session_id=parent_session_id,
            message=(
                f"[Task escalation — not the user speaking] {owner} is "
                f"blocked on the subtask '{task.title}' (task_id: {task.id}) "
                f"and asks: {trimmed} You delegated this task. Answer from "
                "context you already hold with a follow-up "
                "`delegate_to_expert` into their session, or escalate your "
                'own task to the user with `escalate_task(target="user")` '
                "if only they can decide."
            ),
        )
    except Exception:
        logger.warning(
            "Failed to notify delegator about escalated task #%s",
            task.id,
            exc_info=True,
        )
