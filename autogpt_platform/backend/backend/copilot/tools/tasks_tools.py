"""Tasks v1 tools an expert (or Otto) uses while working a task."""

from typing import Any

from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.events import publish_task_event
from backend.api.features.tasks.models import (
    AssigneeType,
    AttachmentCreate,
    TaskFilters,
    TaskStatus,
)
from backend.copilot.model import ChatSession

from .models import TaskListToolResponse, TaskUpdatedToolResponse, ToolResponseBase
from .tasks_helpers import (
    TaskToolBase,
    TaskToolError,
    caller_actor,
    dump,
    resolve_task,
)

_TASK_ID = {
    "type": "string",
    "description": "Task id; omit to use the task this chat is working.",
}
_SETTABLE = ["todo", "in_progress", "in_review", "done", "blocked", "cancelled"]


class ListMyTasksTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "list_my_tasks"

    @property
    def description(self) -> str:
        return (
            "List the tasks on the user's task board assigned to you (this "
            "expert, or Otto in a plain chat). Optionally filter by status."
        )

    @property
    def search_keywords(self) -> tuple[str, ...]:
        return ("tasks", "task board", "assigned", "todo")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "array", "items": {"type": "string", "enum": _SETTABLE}},
                "include_done": {"type": "boolean", "default": False},
            },
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        statuses = kwargs.get("status") or None
        filters = TaskFilters(
            status=[TaskStatus(s) for s in statuses] if statuses else None,
            assignee_type=AssigneeType.EXPERT if session.expert_id else AssigneeType.OTTO,
            assignee_id=session.expert_id,
        )
        tasks = await tasks_store().list_tasks(user_id, filters)
        if not kwargs.get("include_done") and not statuses:
            tasks = [t for t in tasks if t.status != TaskStatus.DONE]
        lines = [f"{t.key} [{t.status.value}] {t.title} (id {t.id})" for t in tasks]
        return TaskListToolResponse(
            message="\n".join(lines) or "No tasks assigned to you.",
            session_id=session.session_id,
            tasks=[t.model_dump(mode="json") for t in tasks],
        )


class UpdateTaskStatusTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "update_task_status"

    @property
    def description(self) -> str:
        return (
            "Move a task to a new status. When the work is finished use "
            "status='done' with outcome_summary (markdown: what you did and "
            "where the result is)."
        )

    @property
    def search_keywords(self) -> tuple[str, ...]:
        return ("task status", "mark done", "complete task", "outcome")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "status": {"type": "string", "enum": _SETTABLE},
                "outcome_summary": {"type": "string"},
                "note": {"type": "string", "description": "Optional short reason."},
            },
            "required": ["status"],
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        task = await resolve_task(user_id, session, kwargs.get("task_id"))
        status = TaskStatus(kwargs["status"])
        summary = kwargs.get("outcome_summary")
        if status == TaskStatus.DONE and not summary:
            raise TaskToolError("Pass outcome_summary when marking a task done.")
        actor = await caller_actor(user_id, session)
        updated = await tasks_store().set_task_status(
            user_id, task.id, status, actor, note=kwargs.get("note"), outcome_summary=summary
        )
        event = "task_completed" if status == TaskStatus.DONE else "task_updated"
        await publish_task_event(user_id, event, updated, task_id=task.id)
        return TaskUpdatedToolResponse(
            message=f"{updated.key} is now {status.value}.",
            session_id=session.session_id,
            task=dump(updated),
        )


class CommentOnTaskTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "comment_on_task"

    @property
    def description(self) -> str:
        return "Post a progress comment on a task's activity feed (markdown)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"task_id": _TASK_ID, "body": {"type": "string"}},
            "required": ["body"],
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        task = await resolve_task(user_id, session, kwargs.get("task_id"))
        actor = await caller_actor(user_id, session)
        activity = await tasks_store().add_comment(user_id, task.id, kwargs["body"], actor)
        await publish_task_event(user_id, "task_activity", task, activity, task.id)
        return TaskUpdatedToolResponse(
            message=f"Comment posted on {task.key}.",
            session_id=session.session_id,
            task=dump(task),
            activity=dump(activity),
        )


class RequestInputTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "request_input"

    @property
    def description(self) -> str:
        return (
            "Ask the user a question you cannot proceed without. The task moves "
            "to 'Needs you'. After calling this, STOP and end your turn: the "
            "answer arrives as a new message in this chat."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "question": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suggested answers; [] for free text.",
                },
            },
            "required": ["question"],
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        store = tasks_store()
        task = await resolve_task(user_id, session, kwargs.get("task_id"))
        actor = await caller_actor(user_id, session)
        await store.request_input(
            user_id, task.id, kwargs["question"], kwargs.get("options") or [], actor
        )
        events = await store.list_activity(user_id, task.id)
        activity = events[-1] if events else None
        fresh = await store.get_task(user_id, task.id)
        await publish_task_event(user_id, "task_needs_input", fresh, activity, task.id)
        return TaskUpdatedToolResponse(
            message=(
                f"Question sent; {task.key} now waits for the user. Stop here "
                "and end your turn. Their answer will arrive as a new message."
            ),
            session_id=session.session_id,
            task=dump(fresh),
        )


class LinkArtifactTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "link_artifact"

    @property
    def description(self) -> str:
        return (
            "Attach a result to a task: a workspace file (kind=file, ref_id=file "
            "id), a chat (kind=chat), a workflow run (kind=run, ref_id=graph exec "
            "id), a link (kind=url, url=...) or a text note (kind=text)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "kind": {"type": "string", "enum": ["file", "chat", "run", "url", "text"]},
                "title": {"type": "string"},
                "ref_id": {"type": "string"},
                "url": {"type": "string"},
            },
            "required": ["kind", "title"],
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        task = await resolve_task(user_id, session, kwargs.get("task_id"))
        actor = await caller_actor(user_id, session)
        data = AttachmentCreate(
            kind=kwargs["kind"],
            title=kwargs["title"],
            ref_id=kwargs.get("ref_id"),
            url=kwargs.get("url"),
        )
        attachment = await tasks_store().add_attachment(user_id, task.id, data, actor)
        await publish_task_event(user_id, "task_updated", task, task_id=task.id)
        return TaskUpdatedToolResponse(
            message=f"Attached '{attachment.title}' to {task.key}.",
            session_id=session.session_id,
            task=dump(task),
            attachment=dump(attachment),
        )
