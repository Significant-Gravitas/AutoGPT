"""Park a DelegatedTask on the user with a question.

The escalation flips the task to WAITING_USER and records the question — and
this session's id — on the task's timeline. Home's "Needs You" surfaces the
question as a card; the user's answer is delivered back into this session as
a pending message and the task resumes WORKING.
"""

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.util.clients import get_database_manager_async_client
from backend.util.exceptions import TaskDelegationRefusedError, TaskUpdateConflictError

from .base import BaseTool
from .models import ErrorResponse, TaskUpdateResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class EscalateTaskTool(BaseTool):
    """Ask the user a blocking question about a delegated task."""

    @property
    def name(self) -> str:
        return "escalate_task"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Escalate a delegated task to the user when you are blocked on "
            "something only they can decide. The task pauses (WAITING_USER) "
            "and your question appears on their Home; their answer comes "
            "back into this conversation and the task resumes. Ask ONE "
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
                        "The single decision you need from the user, with "
                        "enough context to answer without opening the task."
                    ),
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional short answers the user can pick with one "
                        "click instead of typing."
                    ),
                    "default": [],
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
        **kwargs,
    ) -> ToolResponseBase:
        if user_id is None:
            return self._error("Authentication required", session)
        if not task_id.strip():
            return self._error("task_id is required", session)
        if not question.strip():
            return self._error("question is required", session)

        try:
            task = await get_database_manager_async_client().escalate_delegated_task(
                user_id,
                task_id.strip(),
                question=question.strip(),
                options=options or [],
                session_id=session.session_id,
            )
        except (TaskDelegationRefusedError, TaskUpdateConflictError) as e:
            return self._error(str(e), session)

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
