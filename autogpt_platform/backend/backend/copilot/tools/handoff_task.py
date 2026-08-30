"""Transfer ownership of a DelegatedTask to another hired expert.

Where ``handoff_to_expert`` moves a *conversation*, this moves the *receipt*:
the task row's owner swaps, the hop lands in its amendments timeline, and the
task keeps its tree, spend, and history. The write is optimistically locked
on ``updatedAt`` — two experts racing to hand the same task off cannot both
win, and the loser is told to re-read and retry.
"""

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.util.clients import get_database_manager_async_client
from backend.util.exceptions import TaskDelegationRefusedError, TaskUpdateConflictError

from .base import BaseTool
from .expert_delegation import resolve_target_expert, unknown_target_message
from .models import ErrorResponse, TaskUpdateResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class HandoffTaskTool(BaseTool):
    """Swap a delegated task's owner, keeping its receipt and history."""

    @property
    def name(self) -> str:
        return "handoff_task"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Transfer a delegated task you own to a DIFFERENT expert on the "
            "user's team, when the remaining work needs their skills rather "
            "than yours. The task keeps its history and spend; only the "
            "owner changes. Refused after 5 handoffs — finish or "
            "escalate_task instead. On a conflict error, the task changed "
            "under you: re-check it and retry."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The delegated task to hand over.",
                },
                "to_expert_id": {
                    "type": "string",
                    "description": (
                        "Teammate to hand the task to: their expert id from "
                        "<team_context>, or their exact name."
                    ),
                },
                "note": {
                    "type": "string",
                    "description": (
                        "Why the task is changing hands and where the work "
                        "stands — recorded on the task's timeline."
                    ),
                },
            },
            "required": ["task_id", "to_expert_id", "note"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        task_id: str = "",
        to_expert_id: str = "",
        note: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        if user_id is None:
            return self._error("Authentication required", session)
        if not task_id.strip() or not to_expert_id.strip():
            return self._error("task_id and to_expert_id are required", session)

        try:
            target = await resolve_target_expert(user_id, to_expert_id.strip())
        except Exception as e:
            logger.warning(f"Handoff target lookup failed for {to_expert_id}: {e}")
            return self._error(
                "Could not reach that expert right now. Try again.", session
            )
        if target is None or target.is_archived:
            return self._error(
                await unknown_target_message(
                    user_id, to_expert_id.strip(), session.expert_id
                ),
                session,
            )
        if target.schedules_paused_at is not None:
            return self._error(
                f"{target.name} is paused and cannot take over work until "
                "the user resumes them.",
                session,
            )

        client = get_database_manager_async_client()
        try:
            detail = await client.get_delegated_task(user_id, task_id.strip())
            if detail is None:
                return self._error("No such task on this account.", session)
            task = await client.handoff_delegated_task(
                user_id,
                task_id.strip(),
                to_expert_id=target.id,
                note=note.strip() or f"Handed off to {target.name}.",
                expected_updated_at=detail.task.updated_at,
            )
        except (TaskDelegationRefusedError, TaskUpdateConflictError) as e:
            return self._error(str(e), session)

        logger.info(
            "Routing: session %s handed task %s to expert %s",
            session.session_id,
            task.id,
            target.id,
        )
        return TaskUpdateResponse.from_task(
            task,
            action="handoff",
            message=(
                f"Task '{task.title}' is now owned by {target.name} "
                f"(handoff {task.handoff_count} of 5)."
            ),
            session_id=session.session_id,
        )

    def _error(self, message: str, session: ChatSession) -> ErrorResponse:
        return ErrorResponse(message=message, session_id=session.session_id)
