"""Report a DelegatedTask finished.

Marks the receipt DONE — refused while any subtask is still open, so a
parent can't be reported finished out from under its children. When the
closed task is itself a subtask, the parent's working session is notified
with a pending message so the parent owner picks the result up on its next
round (or next turn) instead of polling.
"""

import logging
from typing import Any

from backend.api.features.tasks.models import DelegatedTask
from backend.copilot.model import ChatSession
from backend.copilot.pending_message_helpers import queue_user_message
from backend.util.clients import get_database_manager_async_client
from backend.util.exceptions import TaskDelegationRefusedError, TaskUpdateConflictError

from .base import BaseTool
from .models import ErrorResponse, TaskUpdateResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_NOTIFY_SUMMARY_MAX = 800


class ReportTaskTool(BaseTool):
    """Close a delegated task with its outcome."""

    @property
    def name(self) -> str:
        return "report_task"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Mark a delegated task DONE with a short outcome summary once "
            "the work is finished. Refused while the task still has open "
            "subtasks — wait for them or cancel the ones no longer needed. "
            "If this task was delegated by a teammate, they are notified "
            "automatically."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The delegated task that is finished.",
                },
                "outcome_summary": {
                    "type": "string",
                    "description": (
                        "The receipt shown on the task's activity feed: 1-2 "
                        "short sentences leading with the deliverable and "
                        "where it lives (a name or link). Markdown allowed. "
                        "No process narration — the user wants the result, "
                        "not the journey."
                    ),
                },
            },
            "required": ["task_id", "outcome_summary"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        task_id: str = "",
        outcome_summary: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        if user_id is None:
            return self._error("Authentication required", session)
        if not task_id.strip():
            return self._error("task_id is required", session)
        if not outcome_summary.strip():
            return self._error("outcome_summary is required", session)

        try:
            task = await get_database_manager_async_client().report_delegated_task(
                user_id,
                task_id.strip(),
                outcome_summary=outcome_summary.strip(),
            )
        except (TaskDelegationRefusedError, TaskUpdateConflictError) as e:
            return self._error(str(e), session)

        await _notify_parent(task, session)
        return TaskUpdateResponse.from_task(
            task,
            action="report",
            message=f"Task '{task.title}' is marked done.",
            session_id=session.session_id,
        )

    def _error(self, message: str, session: ChatSession) -> ErrorResponse:
        return ErrorResponse(message=message, session_id=session.session_id)


async def _notify_parent(task: DelegatedTask, session: ChatSession) -> None:
    """Drop the outcome into the parent task's working context.

    A subtask's ``origin_session_id`` is the session its parent owner was
    working in when they delegated, so that session is where the result
    belongs. Skipped when the report happens *in* that session (the owner
    already has the answer in-thread). Best-effort — a lost notification
    still leaves the receipt DONE and pollable.
    """
    if task.parent_task_id is None or task.origin_session_id is None:
        return
    if task.origin_session_id == session.session_id:
        return
    owner = task.owner.name if task.owner else "A teammate"
    summary = " ".join((task.outcome_summary or "").split())
    if len(summary) > _NOTIFY_SUMMARY_MAX:
        summary = f"{summary[:_NOTIFY_SUMMARY_MAX]}…"
    try:
        await queue_user_message(
            session_id=task.origin_session_id,
            message=(
                f"[Task update — not the user speaking] {owner} finished "
                f"the subtask '{task.title}' (task_id: {task.id}). "
                f"Outcome: {summary or 'done.'}"
            ),
        )
    except Exception:
        logger.warning(
            "Failed to notify parent session #%s about task #%s",
            task.origin_session_id,
            task.id,
            exc_info=True,
        )
