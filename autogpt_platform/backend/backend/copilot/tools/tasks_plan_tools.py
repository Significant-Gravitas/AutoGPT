"""Tasks v1 tools that create work: sub-tasks and Otto's plans."""

from typing import Any

from backend.api.features.tasks import dispatch
from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.events import publish_task_event
from backend.api.features.tasks.models import PlanStep, TaskCreate
from backend.copilot.model import ChatSession

from .models import (
    TaskPlanConfirmedToolResponse,
    TaskPlanToolResponse,
    TaskUpdatedToolResponse,
    ToolResponseBase,
)
from .tasks_helpers import (
    TaskToolBase,
    TaskToolError,
    caller_actor,
    dump,
    resolve_assignee_name,
    resolve_task,
)


class CreateTaskTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "create_task"

    @property
    def description(self) -> str:
        return (
            "Create a sub-task of the current task (one level only) and "
            "optionally assign it to a teammate by name, 'otto' or 'me'. An "
            "expert or Otto assignee starts working it right away."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "parent_task_id": {
                    "type": "string",
                    "description": "Parent task id; omit for this chat's task.",
                },
                "title": {"type": "string"},
                "description": {"type": "string"},
                "assignee": {"type": "string", "description": "Expert name/id, 'otto' or 'me'."},
                "priority": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high", "urgent"],
                },
            },
            "required": ["title"],
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        parent = await resolve_task(user_id, session, kwargs.get("parent_task_id"))
        if parent.parent_id:
            raise TaskToolError(
                f"{parent.key} is already a sub-task; sub-tasks cannot have sub-tasks."
            )
        assignee = await resolve_assignee_name(user_id, kwargs.get("assignee"))
        data = TaskCreate(
            title=kwargs["title"],
            description=kwargs.get("description"),
            parent_id=parent.id,
            assignee_type=assignee.type if assignee else None,
            assignee_id=assignee.id if assignee else None,
        )
        if kwargs.get("priority"):
            data.priority = kwargs["priority"]
        actor = await caller_actor(user_id, session)
        detail = await tasks_store().create_task(user_id, data, actor)
        await publish_task_event(user_id, "task_created", detail, task_id=detail.id)
        await dispatch.on_task_assigned(user_id, detail.id)
        return TaskUpdatedToolResponse(
            message=f"Created {detail.key} under {parent.key}.",
            session_id=session.session_id,
            task=dump(detail),
        )


class ProposePlanTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "propose_plan"

    @property
    def description(self) -> str:
        return (
            "Propose a multi-step plan for a goal: each step becomes a sub-task "
            "assigned to an expert (by name), 'otto' or 'me'. The user sees an "
            "approval card; nothing starts until they approve it (or you call "
            "confirm_task_plan after they explicitly say yes in chat)."
        )

    @property
    def search_keywords(self) -> tuple[str, ...]:
        return ("plan", "project", "break down", "tasks", "assign team")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "assignee": {"type": "string"},
                        },
                        "required": ["title"],
                    },
                },
                "budget_usd": {"type": "number"},
            },
            "required": ["goal", "steps"],
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        raw_steps = kwargs.get("steps") or []
        if not raw_steps:
            raise TaskToolError("A plan needs at least one step.")
        steps = [
            PlanStep(
                title=s["title"],
                description=s.get("description"),
                assignee=await resolve_assignee_name(user_id, s.get("assignee")),
            )
            for s in raw_steps
        ]
        plan = await tasks_store().create_plan(
            user_id, kwargs["goal"], steps, kwargs.get("budget_usd"), session.session_id
        )
        return TaskPlanToolResponse(
            message=(
                f"Plan proposed with {len(steps)} steps. The user approves it "
                "on the plan card; wait for them before starting anything."
            ),
            session_id=session.session_id,
            plan=plan.model_dump(mode="json"),
        )


class ConfirmTaskPlanTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "confirm_task_plan"

    @property
    def description(self) -> str:
        return (
            "Start a proposed plan after the user explicitly approved it: "
            "creates the parent task (owned by Otto) and one sub-task per step, "
            "and the assignees start working."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"plan_id": {"type": "string"}},
            "required": ["plan_id"],
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        store = tasks_store()
        plan, parent = await store.confirm_plan(user_id, kwargs["plan_id"], None)
        await publish_task_event(user_id, "task_created", parent, task_id=parent.id)
        for sub in parent.subtasks:
            await publish_task_event(user_id, "task_created", sub, task_id=sub.id)
            await dispatch.on_task_assigned(user_id, sub.id)
        return TaskPlanConfirmedToolResponse(
            message=f"Plan started as {parent.key} with {len(parent.subtasks)} sub-tasks.",
            session_id=session.session_id,
            plan=plan.model_dump(mode="json"),
            parent_task=parent.model_dump(mode="json"),
        )
