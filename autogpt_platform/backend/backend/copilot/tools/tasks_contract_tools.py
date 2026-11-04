"""Tasks v1: the tools a task's worker keeps the task's record (its contract)
with. Each call is one small patch plus an activity row and a live event; none
calls a model. What the worker forgets, the harness fills in when the turn
ends (``contract_db.fill_from_turn``)."""

from datetime import datetime
from typing import Any

from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.contract import (
    SUGGESTED_CAUSES,
    SUGGESTED_EVIDENCE_KINDS,
)
from backend.api.features.tasks.events import publish_contract_update
from backend.copilot.model import ChatSession

from .models import TaskContractToolResponse, ToolResponseBase
from .tasks_helpers import TaskToolBase, TaskToolError, caller_actor, resolve_task

_TASK_ID = {"type": "string", "description": "Omit for this chat's task."}
_STRINGS = {"type": "array", "items": {"type": "string"}}
_STEP_STATUS = {
    "type": "string",
    "enum": ["pending", "running", "done", "failed", "skipped"],
}


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


class _ContractTool(TaskToolBase):
    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        task = await resolve_task(user_id, session, kwargs.pop("task_id", None))
        actor = await caller_actor(user_id, session)
        try:
            update = await self._patch(user_id, task.id, actor, **kwargs)
        except (ValueError, LookupError) as e:
            raise TaskToolError(str(e)) from e
        await publish_contract_update(user_id, update)
        return TaskContractToolResponse(
            message=self._message(update),
            session_id=session.session_id,
            task_id=task.id,
            version=update.version,
            state=update.task.state.value,
            contract=update.contract.model_dump(mode="json", exclude_none=True),
        )

    async def _patch(self, user_id: str, task_id: str, actor: Any, **kwargs: Any):
        raise NotImplementedError

    def _message(self, update: Any) -> str:
        return f"Saved (v{update.version})."


class UpdateTaskTool(_ContractTool):
    @property
    def name(self) -> str:
        return "update_task"

    @property
    def description(self) -> str:
        return (
            "Keep this task's record; send only what changed. goal, done_when "
            "and plan replace those lists. steps and criteria set statuses by "
            "id (s1, d1). blocker: why it can't go on. result: finishes it, "
            "done if every check is met, else the user accepts it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "goal": {"type": "string", "description": "In the user's words."},
                "done_when": {**_STRINGS, "description": "Checks that prove it."},
                "plan": {**_STRINGS, "description": "Step titles."},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "status": _STEP_STATUS,
                            "rows": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "status": _STEP_STATUS,
                                    },
                                    "required": ["title"],
                                },
                                "description": "Parallel runs of this step, "
                                "by title (one per hire).",
                            },
                        },
                        "required": ["id"],
                    },
                    "description": "Step changes.",
                },
                "criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "met", "unmet", "skipped"],
                            },
                            "evidence_ids": _STRINGS,
                        },
                        "required": ["id", "status"],
                    },
                    "description": "done_when changes, with proof ids.",
                },
                "blocker": {
                    "type": "object",
                    "properties": {
                        "what_stopped_it": {"type": "string"},
                        "cause": {
                            "type": "string",
                            "description": "e.g. " + ", ".join(SUGGESTED_CAUSES),
                        },
                        "done_so_far": {"type": "string"},
                        "next_actions": {
                            **_STRINGS,
                            "description": "Keys: allow, skip",
                        },
                    },
                    "required": ["what_stopped_it"],
                    "description": "Rest auto-filled",
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "primary_evidence_id": {"type": "string"},
                    },
                    "required": ["summary"],
                    "description": "1-2 lines.",
                },
            },
        }

    async def _patch(self, user_id: str, task_id: str, actor: Any, **kwargs: Any):
        patch = {k: v for k, v in kwargs.items() if v is not None}
        if not patch:
            raise ValueError("Send at least one field to change.")
        return await tasks_store().contract_update_task(user_id, task_id, patch, actor)

    def _message(self, update: Any) -> str:
        if update.status_event == "task_completed":
            return "Done."
        if update.task.state.value == "needs_you" and update.contract.result:
            return "Sent to the user to accept: not every check is met."
        if update.contract.blocker is not None:
            return "Blocker recorded. End your turn."
        return f"Saved (v{update.version})."


class AddEvidenceTool(_ContractTool):
    @property
    def name(self) -> str:
        return "add_evidence"

    @property
    def description(self) -> str:
        return (
            "Record proof of the work with a one-line summary. Raw output "
            "(HTML, CSS, tables) goes in content, shown only when opened. Files "
            "you write are recorded for you."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "kind": {
                    "type": "string",
                    "description": "e.g. " + ", ".join(SUGGESTED_EVIDENCE_KINDS),
                },
                "title": {"type": "string", "description": "Short name."},
                "summary": {"type": "string", "description": "One line."},
                "url": {"type": "string", "description": "For a link."},
                "ref": {"type": "string", "description": "File, run or record id."},
                "criterion_ids": {**_STRINGS, "description": "Checks it proves."},
                "content": {"type": "string", "description": "Raw output."},
            },
            "required": ["kind", "title", "summary"],
        }

    async def _patch(self, user_id: str, task_id: str, actor: Any, **kwargs: Any):
        return await tasks_store().contract_add_evidence(
            user_id,
            task_id,
            kwargs["kind"],
            kwargs["title"],
            kwargs.get("summary") or "",
            actor,
            ref=kwargs.get("ref"),
            url=kwargs.get("url"),
            criterion_ids=_strings(kwargs.get("criterion_ids")),
            content=kwargs.get("content"),
        )

    def _message(self, update: Any) -> str:
        return f"Evidence {update.evidence.id} saved (v{update.version})."


class AskUserTool(_ContractTool):
    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Ask the user one question; the task waits in Needs you and a "
            "second ask joins the open one. Finish work that doesn't need the "
            "answer, then end your turn; it arrives as a new message."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": _TASK_ID,
                "question": {"type": "string", "description": "One question."},
                "why": {"type": "string", "description": "What it decides."},
                "options": {**_STRINGS, "description": "Suggested answers."},
                "default_option": {"type": "string", "description": "Your pick."},
                "blocking_step_id": {
                    "type": "string",
                    "description": "The step that waits on it.",
                },
                "expires_at": {
                    "type": "string",
                    "description": "ISO time it's needed by.",
                },
            },
            "required": ["question"],
        }

    async def _patch(self, user_id: str, task_id: str, actor: Any, **kwargs: Any):
        expires = kwargs.get("expires_at")
        return await tasks_store().contract_ask(
            user_id,
            task_id,
            kwargs["question"],
            actor,
            why=kwargs.get("why"),
            options=_strings(kwargs.get("options")),
            default_option=kwargs.get("default_option"),
            blocking_step_id=kwargs.get("blocking_step_id"),
            expires_at=datetime.fromisoformat(expires) if expires else None,
        )

    def _message(self, update: Any) -> str:
        return (
            "Asked. Finish any step that doesn't need the answer, then end your "
            "turn; the answer arrives as a new message."
        )
