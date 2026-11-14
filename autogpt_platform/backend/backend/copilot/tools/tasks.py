"""Tools for keeping a record of each piece of work as a task.

A chat can carry several tasks and a task can span several chats. Tasks are
the user's bookkeeping, so these tools run without asking (gate/policy.py).
"""

from typing import Any

from prisma.enums import TaskCreator, TaskStatus

from backend.copilot.model import ChatSession
from backend.data.db_accessors import task_db
from backend.data.task import TaskPatch
from backend.data.task_text import DONE_WHEN_MAX_CHARS, GOAL_MAX_CHARS, clip

from .base import BaseTool
from .external_scope import error_detail
from .models import ErrorResponse, ToolResponseBase
from .task_models import (
    TaskArchivedResponse,
    TaskCreatedResponse,
    TaskDetailsResponse,
    TaskInfo,
    TaskListResponse,
    TaskUpdatedResponse,
)

_STATUSES = [status.value for status in TaskStatus]
_TASK_ID = {
    "type": "string",
    "description": "The task's id, from create_task or list_tasks.",
    "entity": None,
}
_EXPERT_ID = {
    "type": "string",
    "description": "Id of the hired expert who does the work; omit for you.",
    "entity": "expert",
}


class CreateTaskTool(BaseTool):
    """Open a task for a piece of work, linked to this chat."""

    @property
    def name(self) -> str:
        return "create_task"

    @property
    def description(self) -> str:
        return (
            "Record a new piece of work as a task, linked to this chat. Open one "
            "per distinct goal so each has its own record and status."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short task title."},
                "description": {
                    "type": "string",
                    "description": "What done looks like, in a few lines.",
                },
                "expert_id": _EXPERT_ID,
            },
            "required": ["title"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        title: str = "",
        description: str | None = None,
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        assert user_id is not None  # guaranteed by requires_auth
        try:
            task = await task_db().create_task(
                user_id=user_id,
                title=title,
                description=description,
                owner_expert_id=expert_id or None,
                created_by=(
                    TaskCreator.EXPERT if session.expert_id else TaskCreator.OTTO
                ),
                created_by_expert_id=session.expert_id,
                session_id=session.session_id,
                organization_id=session.organization_id,
                team_id=session.team_id,
            )
        except ValueError as e:  # includes NotFoundError
            return _error("create_task_failed", str(e), session)
        except Exception as e:
            return _error("create_task_failed", error_detail(e, session), session)
        return TaskCreatedResponse(
            message=f"Created task '{task.title}'.",
            task=TaskInfo.from_task(task),
            session_id=session.session_id,
        )


class ListTasksTool(BaseTool):
    """List the user's tasks, this chat's or all of them."""

    @property
    def name(self) -> str:
        return "list_tasks"

    @property
    def description(self) -> str:
        return "List the user's tasks, newest first; archived ones are left out."

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "this_chat_only": {
                    "type": "boolean",
                    "description": "Only the tasks linked to this chat.",
                },
                "status": {
                    "type": "string",
                    "enum": _STATUSES,
                    "description": "Only tasks in this status.",
                },
                "expert_id": _EXPERT_ID,
            },
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        this_chat_only: bool = False,
        status: str | None = None,
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        assert user_id is not None  # guaranteed by requires_auth
        try:
            tasks = await task_db().list_tasks(
                user_id=user_id,
                status=TaskStatus(status) if status else None,
                owner_expert_id=expert_id or None,
                session_id=session.session_id if this_chat_only else None,
            )
        except ValueError as e:  # includes NotFoundError
            return _error("list_tasks_failed", str(e), session)
        except Exception as e:
            return _error("list_tasks_failed", error_detail(e, session), session)
        return TaskListResponse(
            message=f"Found {len(tasks)} task(s).",
            tasks=[TaskInfo.from_task(task) for task in tasks],
            count=len(tasks),
            session_id=session.session_id,
        )


class GetTaskTool(BaseTool):
    """Read one task."""

    @property
    def name(self) -> str:
        return "get_task"

    @property
    def description(self) -> str:
        return "Read one task: its status, owner and the chats it was worked in."

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"task_id": _TASK_ID},
            "required": ["task_id"],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, task_id: str = "", **kwargs
    ) -> ToolResponseBase:
        assert user_id is not None  # guaranteed by requires_auth
        try:
            task = await task_db().get_task(task_id, user_id)
        except ValueError as e:  # includes NotFoundError
            return _error("get_task_failed", str(e), session)
        except Exception as e:
            return _error("get_task_failed", error_detail(e, session), session)
        if task is None:
            return _error("task_not_found", f"Task {task_id} not found", session)
        return TaskDetailsResponse(
            message=f"Task '{task.title}' is {task.status.value}.",
            task=TaskInfo.from_task(task),
            session_id=session.session_id,
        )


class UpdateTaskTool(BaseTool):
    """Change a task's title, description, status or owner."""

    @property
    def name(self) -> str:
        return "update_task"

    @property
    def description(self) -> str:
        return (
            "Update a task: move its status as the work goes (IN_PROGRESS, DONE, "
            "FAILED, CANCELLED), rename it, or hand it to another owner."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "title": {"type": "string", "description": "New title."},
                "description": {
                    "type": "string",
                    "description": "An empty string clears it.",
                },
                "status": {
                    "type": "string",
                    "enum": _STATUSES,
                    "description": "New status.",
                },
                "expert_id": _EXPERT_ID,
                "assign_to_otto": {
                    "type": "boolean",
                    "description": "Take the task back from its expert.",
                },
                "goal": {"type": "string", "description": "What it is for."},
                "done_when": {
                    "type": "string",
                    "description": "What done looks like, so it can be checked.",
                },
            },
            "required": ["task_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        task_id: str = "",
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        expert_id: str | None = None,
        assign_to_otto: bool = False,
        goal: str | None = None,
        done_when: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        assert user_id is not None  # guaranteed by requires_auth
        try:
            patch = TaskPatch(
                title=title,
                description=description,
                status=TaskStatus(status) if status else None,
                owner_expert_id=expert_id or None,
                assign_to_otto=assign_to_otto,
                goal=clip(goal, GOAL_MAX_CHARS),
                done_when=clip(done_when, DONE_WHEN_MAX_CHARS),
            )
            task = await task_db().update_task(task_id, user_id, patch)
        except ValueError as e:  # includes NotFoundError
            return _error("update_task_failed", str(e), session)
        except Exception as e:
            return _error("update_task_failed", error_detail(e, session), session)
        return TaskUpdatedResponse(
            message=f"Updated task '{task.title}' ({task.status.value}).",
            task=TaskInfo.from_task(task),
            session_id=session.session_id,
        )


class ArchiveTaskTool(BaseTool):
    """Archive a task the user no longer wants listed."""

    @property
    def name(self) -> str:
        return "archive_task"

    @property
    def description(self) -> str:
        return "Archive a task so it is no longer listed. Its history is kept."

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"task_id": _TASK_ID},
            "required": ["task_id"],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, task_id: str = "", **kwargs
    ) -> ToolResponseBase:
        assert user_id is not None  # guaranteed by requires_auth
        try:
            archived = await task_db().archive_task(task_id, user_id)
        except ValueError as e:  # includes NotFoundError
            return _error("archive_task_failed", str(e), session)
        except Exception as e:
            return _error("archive_task_failed", error_detail(e, session), session)
        if not archived:
            return _error("task_not_found", f"Task {task_id} not found", session)
        return TaskArchivedResponse(
            message="Archived the task.",
            task_id=task_id,
            session_id=session.session_id,
        )


def _error(code: str, message: str, session: ChatSession) -> ErrorResponse:
    return ErrorResponse(message=message, error=code, session_id=session.session_id)
