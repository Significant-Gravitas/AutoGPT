"""Tasks v1 tools that create work: sub-tasks and Otto's plans."""

from typing import Any

from backend.api.features.tasks import dispatch
from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.events import publish_task_event
from backend.api.features.tasks.models import PlanStep, Task, TaskCreate
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
    distinct_titles,
    dump,
    find_open_duplicate,
    find_work_task,
    resolve_assignee_name,
    resolve_task,
)

MAX_SUBTASKS = 10


class CreateTaskTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "create_task"

    @property
    def description(self) -> str:
        return (
            "Put one piece of work on the user's task board, optionally "
            "assigned to an expert or 'otto', who starts it right away. In a "
            "chat that is working a task, or with parent_task_id, it is a "
            "sub-task (one level only). Several clearly different "
            "deliverables (one per channel, language or stage) are a split "
            "propose_plan, not one create_task. For a tracked outcome that "
            "outlives this turn, or when the user asks to track something "
            "('add to my tasks'). Never for an answer, a lookup, a short "
            "draft or one setting change. A message about an open task "
            "updates that task instead. Work asked for 'every…' or 'when…' "
            "is a routine (tool:schedule_routine). A hire is never a task: "
            "tool:hire_expert's card puts it on the board."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "parent_task_id": {
                    "type": "string",
                    "description": (
                        "Parent task id. Omit for this chat's task, or for a "
                        "new top-level task in a chat not working one."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Short outcome in the user's words, never an invented name.",
                },
                "description": {
                    "type": "string",
                    "description": "Details, including anything still unknown.",
                },
                "assignee": {
                    "type": "string",
                    "description": "Expert name/id or 'otto'.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high", "urgent"],
                    "description": "Default none.",
                },
            },
            "required": ["title"],
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        parent = await _parent_for(user_id, session, kwargs.get("parent_task_id"))
        if parent is not None and parent.parent_id:
            raise TaskToolError(
                f"{parent.key} is already a sub-task; sub-tasks cannot have sub-tasks."
            )
        assignee = await resolve_assignee_name(user_id, kwargs.get("assignee"))
        if parent is None:
            existing = await find_open_duplicate(user_id, kwargs["title"], assignee)
            if existing is not None:
                raise TaskToolError(
                    f'{existing.key} "{existing.title}" is already open. Update '
                    "it with update_task_status or comment_on_task instead of "
                    "creating another task."
                )
        data = TaskCreate(
            title=kwargs["title"],
            description=kwargs.get("description"),
            parent_id=parent.id if parent else None,
            assignee_type=assignee.type if assignee else None,
            assignee_id=assignee.id if assignee else None,
        )
        if kwargs.get("priority"):
            data.priority = kwargs["priority"]
        actor = await caller_actor(user_id, session)
        detail = await tasks_store().create_task(user_id, data, actor)
        await publish_task_event(user_id, "task_created", detail, task_id=detail.id)
        await dispatch.on_task_assigned(user_id, detail.id)
        created = (
            f"Created {detail.key} under {parent.key}."
            if parent
            else f"Created {detail.key}."
        )
        return TaskUpdatedToolResponse(
            message=(
                f"{created} The user sees a link to it in the chat: confirm "
                f'in one line by its title "{detail.title}", without its key '
                "or fields."
            ),
            session_id=session.session_id,
            task=dump(detail),
        )


async def _parent_for(
    user_id: str, session: ChatSession, parent_task_id: str | None
) -> Task | None:
    if parent_task_id:
        return await resolve_task(user_id, session, parent_task_id)
    return await find_work_task(user_id, session)


class ProposePlanTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "propose_plan"

    @property
    def description(self) -> str:
        return (
            "Propose a plan for a goal; steps are assigned to an expert (by "
            "name) or 'otto'. By default it runs as ONE task with the "
            "steps as its checklist: use that for batch work ('write 5 "
            "emails', 'reply to 30 emails') and the steps of one deliverable. "
            "Set split only for clearly different pieces of work with their "
            "own outcome, owner or approval; each step then becomes a "
            "sub-task (2 to 10, one level). Never split a hire. The user sees "
            "an approval card; nothing starts until they approve it (or you "
            "call confirm_task_plan after they explicitly say yes in chat)."
        )

    @property
    def search_keywords(self) -> tuple[str, ...]:
        return ("plan", "project", "break down", "tasks", "assign team")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The overall outcome."},
                "steps": {
                    "type": "array",
                    "description": "Checklist items, or sub-tasks when split.",
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
                "split": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "True only when the steps are clearly different pieces "
                        "of work with their own outcome, owner or approval: "
                        "each becomes a sub-task (2 to 10). False: one task "
                        "with the steps as its checklist."
                    ),
                },
                "budget_usd": {"type": "number", "description": "Optional cap."},
            },
            "required": ["goal", "steps"],
        }

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        raw_steps = kwargs.get("steps") or []
        split = bool(kwargs.get("split"))
        check_plan_shape([s["title"] for s in raw_steps], split)
        steps = [
            PlanStep(
                title=s["title"],
                description=s.get("description"),
                assignee=await resolve_assignee_name(user_id, s.get("assignee")),
            )
            for s in raw_steps
        ]
        plan = await tasks_store().create_plan(
            user_id,
            kwargs["goal"],
            steps,
            kwargs.get("budget_usd"),
            session.session_id,
            split,
        )
        shape = (
            f"{len(steps)} sub-tasks" if split else f"one task with {len(steps)} items"
        )
        return TaskPlanToolResponse(
            message=(
                f"Plan proposed as {shape}. The user approves it on the plan "
                "card; wait for them before starting anything. Any older plan "
                "in this chat is replaced and can't be approved, so describe "
                "only this one, without repeating the steps the card shows."
            ),
            session_id=session.session_id,
            plan=plan.model_dump(mode="json"),
        )


def check_plan_shape(titles: list[str], split: bool) -> None:
    if not titles:
        raise TaskToolError("A plan needs at least one step.")
    if not split:
        return
    if distinct_titles(titles) < 2:
        raise TaskToolError(
            "Sub-tasks are only for clearly different pieces of work. Propose "
            "this with split=false so it runs as one task."
        )
    if len(titles) > MAX_SUBTASKS:
        raise TaskToolError(
            f"A plan can split into at most {MAX_SUBTASKS} sub-tasks. Batch the "
            "rest into fewer steps (e.g. 'Reply to emails 1-10')."
        )


class ConfirmTaskPlanTool(TaskToolBase):
    @property
    def name(self) -> str:
        return "confirm_task_plan"

    @property
    def description(self) -> str:
        return (
            "Start a proposed plan after the user explicitly approved it: "
            "creates its one task, or for a split plan the parent task (owned "
            "by Otto) and one sub-task per step, and the assignees start "
            "working."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "From propose_plan."}
            },
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
        if not plan.split:
            await dispatch.on_task_assigned(user_id, parent.id)
        return TaskPlanConfirmedToolResponse(
            message=(
                f"Plan started as {parent.key} with {len(parent.subtasks)} sub-tasks."
                if plan.split
                else f"Plan started as {parent.key}."
            ),
            session_id=session.session_id,
            plan=plan.model_dump(mode="json"),
            parent_task=parent.model_dump(mode="json"),
        )
