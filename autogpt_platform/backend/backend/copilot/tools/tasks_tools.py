"""Tasks v1 tools an expert (or Otto) uses while working a task."""

from typing import Any

from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.events import (
    publish_contract_update,
    publish_task_event,
)
from backend.api.features.tasks.models import (
    AssigneeType,
    AttachmentCreate,
    Task,
    TaskFilters,
    TaskStatus,
)
from backend.api.features.tasks.outcome_text import reads_as_failure
from backend.copilot.model import ChatSession

from .models import TaskListToolResponse, TaskUpdatedToolResponse, ToolResponseBase
from .tasks_helpers import TaskToolBase, TaskToolError, caller_actor, dump, resolve_task

_TASK_ID = {
    "type": "string",
    "description": "Task id; omit to use the task this chat is working.",
}
_SETTABLE = ["todo", "in_progress", "in_review", "done", "blocked", "cancelled"]
_EVIDENCE_KIND = {
    "file": "file",
    "url": "link",
    "run": "record",
    "chat": "record",
    "text": "quote",
}


class ListMyTasksTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "list_my_tasks"

    @property
    def description(self) -> str:
        return (
            "List the tasks on the user's task board assigned to you (this "
            "expert, or Otto in a plain chat). Optionally filter by status. "
            "Pass include_others to see everyone's, e.g. to find the open "
            "task a message is about before creating a new one."
        )

    @property
    def search_keywords(self) -> tuple[str, ...]:
        return ("tasks", "task board", "assigned", "todo")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "array",
                    "items": {"type": "string", "enum": _SETTABLE},
                    "description": "Only these statuses.",
                },
                "include_done": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also list done tasks.",
                },
                "include_others": {
                    "type": "boolean",
                    "default": False,
                    "description": "Also list tasks assigned to others or no one.",
                },
            },
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        statuses = kwargs.get("status") or None
        everyone = bool(kwargs.get("include_others"))
        filters = TaskFilters(
            status=[TaskStatus(s) for s in statuses] if statuses else None,
            assignee_type=None if everyone else _own_assignee_type(session),
            assignee_id=None if everyone else session.expert_id,
        )
        tasks = await tasks_store().list_tasks(user_id, filters)
        if not kwargs.get("include_done") and not statuses:
            tasks = [t for t in tasks if t.status != TaskStatus.DONE]
        lines = [_task_line(t, with_assignee=everyone) for t in tasks]
        empty = "No open tasks." if everyone else "No tasks assigned to you."
        return TaskListToolResponse(
            message="\n".join(lines) or empty,
            session_id=session.session_id,
            tasks=[t.model_dump(mode="json") for t in tasks],
        )


def _own_assignee_type(session: ChatSession) -> AssigneeType:
    return AssigneeType.EXPERT if session.expert_id else AssigneeType.OTTO


def _task_line(task: Task, *, with_assignee: bool) -> str:
    line = f"{task.key} [{task.status.value}] {task.title} (id {task.id})"
    if not with_assignee:
        return line
    return f"{line}, assigned to {task.assignee.name if task.assignee else 'no one'}"


def _stated_failure(
    status: TaskStatus, summary: str | None, note: str | None
) -> str | None:
    """The blocker a status change amounts to: the reason a task is moved to
    blocked, or a done summary that says the work wasn't done."""
    if status == TaskStatus.BLOCKED:
        return (note or "").strip() or "Stopped before it finished."
    if status == TaskStatus.DONE and reads_as_failure(summary):
        return (summary or "").strip()
    return None


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
                "status": {
                    "type": "string",
                    "enum": _SETTABLE,
                    "description": "The new status.",
                },
                "outcome_summary": {
                    "type": "string",
                    "description": "Required for done.",
                },
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
        reason = _stated_failure(status, summary, kwargs.get("note"))
        if reason is not None:
            # A stop says why, and a "done" that says the work couldn't be
            # done fails instead, so the card never reads green or blank.
            update = await tasks_store().contract_update_task(
                user_id, task.id, {"blocker": {"what_stopped_it": reason}}, actor
            )
            await publish_contract_update(user_id, update)
            return TaskUpdatedToolResponse(
                message=f"{task.key} is blocked: {reason} End your turn.",
                session_id=session.session_id,
                task=dump(update.task),
            )
        updated = await tasks_store().set_task_status(
            user_id,
            task.id,
            status,
            actor,
            note=kwargs.get("note"),
            outcome_summary=summary,
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
            "properties": {
                "task_id": _TASK_ID,
                "body": {"type": "string", "description": "Markdown comment."},
            },
            "required": ["body"],
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        task = await resolve_task(user_id, session, kwargs.get("task_id"))
        actor = await caller_actor(user_id, session)
        activity = await tasks_store().add_comment(
            user_id, task.id, kwargs["body"], actor
        )
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
                "question": {"type": "string", "description": "One question."},
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
        task = await resolve_task(user_id, session, kwargs.get("task_id"))
        actor = await caller_actor(user_id, session)
        update = await tasks_store().contract_ask(
            user_id,
            task.id,
            kwargs["question"],
            actor,
            options=kwargs.get("options") or [],
        )
        await publish_contract_update(user_id, update)
        return TaskUpdatedToolResponse(
            message=(
                f"Question sent; {task.key} now waits for the user. Stop here "
                "and end your turn. Their answer will arrive as a new message."
            ),
            session_id=session.session_id,
            task=dump(update.task),
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
                "kind": {
                    "type": "string",
                    "enum": ["file", "chat", "run", "url", "text"],
                    "description": "What the result is.",
                },
                "title": {"type": "string", "description": "Shown on the task."},
                "ref_id": {"type": "string", "description": "File, chat or run id."},
                "url": {"type": "string", "description": "For kind=url."},
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
        store = tasks_store()
        attachment = await store.add_attachment(user_id, task.id, data, actor)
        update = await store.contract_add_evidence(
            user_id,
            task.id,
            _EVIDENCE_KIND[data.kind],
            data.title,
            data.title,
            actor,
            ref=data.ref_id,
            url=data.url,
            attachment_id=attachment.id,
        )
        await publish_contract_update(user_id, update)
        await publish_task_event(user_id, "task_updated", task, task_id=task.id)
        return TaskUpdatedToolResponse(
            message=f"Attached '{attachment.title}' to {task.key}.",
            session_id=session.session_id,
            task=dump(task),
            attachment=dump(attachment),
        )
