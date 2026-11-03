import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import prisma.errors
import prisma.models
import prisma.types
from prisma import Json

from backend.api.features.tasks.converters import (
    TASK_INCLUDE,
    expert_assignee,
    row_to_activity,
    row_to_attachment,
    row_to_detail,
    row_to_plan,
    row_to_task,
    task_key,
    to_datetime,
)
from backend.api.features.tasks.models import (
    HIDDEN_STATUSES,
    ActivityEvent,
    ActivityKind,
    Actor,
    Assignee,
    AssigneeRef,
    AssigneeType,
    Attachment,
    AttachmentCreate,
    ExpertOption,
    PendingInput,
    Plan,
    PlanStatus,
    PlanStep,
    Task,
    TaskCreate,
    TaskDetail,
    TaskFilters,
    TaskStatus,
    TaskUpdate,
)
from backend.api.features.tasks.plans_db import cancel_plan, create_plan, get_plan

logger = logging.getLogger(__name__)

__all__ = ["cancel_plan", "create_plan", "get_plan"]

_ACTIVE_STATUSES = [TaskStatus.IN_PROGRESS]


async def _find_row(user_id: str, task_id: str) -> Optional[prisma.models.Task]:
    return await prisma.models.Task.prisma().find_first(
        where={"id": task_id, "userId": user_id}, include=TASK_INCLUDE
    )


async def get_task(user_id: str, task_id: str) -> Optional[Task]:
    row = await _find_row(user_id, task_id)
    return row_to_task(row) if row else None


async def get_task_detail(user_id: str, task_id: str) -> Optional[TaskDetail]:
    row = await _find_row(user_id, task_id)
    if row is None:
        return None
    subtasks = await prisma.models.Task.prisma().find_many(
        where={"parentId": row.id, "userId": user_id},
        include=TASK_INCLUDE,
        order={"number": "asc"},
    )
    attachments = await prisma.models.TaskAttachment.prisma().find_many(
        where={"taskId": row.id}, order={"createdAt": "asc"}
    )
    return row_to_detail(row, subtasks, attachments)


async def list_tasks(user_id: str, filters: TaskFilters) -> list[Task]:
    where: dict[str, Any] = {"userId": user_id}
    if filters.status:
        where["status"] = {"in": [s.value for s in filters.status]}
    elif not filters.include_hidden:
        where["status"] = {"not_in": [s.value for s in HIDDEN_STATUSES]}
    if filters.mine:
        where["assigneeType"] = AssigneeType.USER.value
    elif filters.assignee_type:
        where["assigneeType"] = filters.assignee_type.value
    if filters.assignee_id:
        where["assigneeExpertId"] = filters.assignee_id
    if filters.priority:
        where["priority"] = {"in": [p.value for p in filters.priority]}
    if filters.parent_id:
        where["parentId"] = filters.parent_id
    elif filters.top_level:
        where["parentId"] = None
    if filters.q:
        where["title"] = {"contains": filters.q, "mode": "insensitive"}
    if filters.due == "none":
        where["dueDate"] = None
    elif filters.due:
        where["dueDate"] = _due_filter(filters.due)
        if filters.due == "overdue" and not filters.status:
            where["status"] = {"not_in": ["done", "cancelled", "backlog"]}
    rows = await prisma.models.Task.prisma().find_many(
        where=where,
        include=TASK_INCLUDE,
        order={"updatedAt": "desc"},
        take=max(1, min(filters.limit, 500)),
    )
    return [row_to_task(r) for r in rows]


def _due_filter(due: str) -> dict[str, datetime]:
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if due == "overdue":
        return {"lt": today}
    if due == "today":
        return {"gte": today, "lt": today + timedelta(days=1)}
    if due == "week":
        return {"gte": today, "lt": today + timedelta(days=7)}
    return {"gte": today, "lt": today + timedelta(days=7)}


async def list_tasks_needing_user(user_id: str) -> list[Task]:
    rows = await prisma.models.Task.prisma().find_many(
        where={"userId": user_id, "status": {"in": ["blocked", "in_review"]}},
        include=TASK_INCLUDE,
        order={"updatedAt": "desc"},
    )
    return [row_to_task(r) for r in rows]


async def resolve_assignee(
    user_id: str, assignee_type: Optional[AssigneeType], assignee_id: Optional[str]
) -> Optional[Assignee]:
    if assignee_type is None:
        return None
    if assignee_type == AssigneeType.USER:
        return Assignee(type=AssigneeType.USER, id=user_id, name="Me")
    if assignee_type == AssigneeType.OTTO:
        return Assignee(type=AssigneeType.OTTO, id=None, name="Otto")
    if not assignee_id:
        raise ValueError("Pick an expert to assign this task to")
    expert = await prisma.models.Expert.prisma().find_first(
        where={"id": assignee_id, "ownerUserId": user_id, "isArchived": False}
    )
    if expert is None:
        raise ValueError("That expert isn't on your team")
    return expert_assignee(expert)


async def _next_number(user_id: str) -> int:
    last = await prisma.models.Task.prisma().find_first(
        where={"userId": user_id}, order={"number": "desc"}
    )
    return (last.number if last else 0) + 1


async def create_task(user_id: str, data: TaskCreate, actor: Actor) -> TaskDetail:
    parent = None
    if data.parent_id:
        parent = await _find_row(user_id, data.parent_id)
        if parent is None:
            raise ValueError("Parent task not found")
        if parent.parentId:
            raise ValueError("Sub-tasks can't have their own sub-tasks")
    assignee = await resolve_assignee(user_id, data.assignee_type, data.assignee_id)
    status = data.status or TaskStatus.TODO
    payload: dict[str, Any] = {
        "userId": user_id,
        "title": data.title.strip(),
        "description": data.description,
        "status": status.value,
        "priority": (data.priority.value if data.priority else "none"),
        "assigneeType": assignee.type.value if assignee else None,
        "assigneeExpertId": (
            assignee.id if assignee and assignee.type == AssigneeType.EXPERT else None
        ),
        "dueDate": to_datetime(data.due_date),
        "labels": data.labels or [],
        "parentId": parent.id if parent else None,
        "budgetCapUsd": data.budget_cap_usd,
        "reviewRequired": bool(data.review_required),
    }
    if status == TaskStatus.IN_PROGRESS:
        payload["startedAt"] = datetime.now(timezone.utc)
    row = await _insert_with_number(user_id, payload)
    await add_activity(user_id, row.id, ActivityKind.CREATED, actor)
    if parent is not None:
        await add_activity(
            user_id,
            parent.id,
            ActivityKind.SUBTASK_CREATED,
            actor,
            data={
                "subtask_id": row.id,
                "subtask_key": task_key(row.number),
                "title": row.title,
            },
        )
    detail = await get_task_detail(user_id, row.id)
    assert detail is not None
    return detail


async def _insert_with_number(
    user_id: str, payload: dict[str, Any]
) -> prisma.models.Task:
    for attempt in range(2):
        payload["number"] = await _next_number(user_id)
        try:
            return await prisma.models.Task.prisma().create(data=payload)
        except prisma.errors.UniqueViolationError:
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")


async def update_task(
    user_id: str, task_id: str, data: TaskUpdate, actor: Actor
) -> TaskDetail:
    row = await _find_row(user_id, task_id)
    if row is None:
        raise LookupError("Task not found")
    fields = data.model_fields_set
    patch: dict[str, Any] = {}
    for name, column in (
        ("title", "title"),
        ("description", "description"),
        ("outcome_summary", "outcomeSummary"),
        ("budget_cap_usd", "budgetCapUsd"),
    ):
        if name in fields:
            patch[column] = getattr(data, name)
    if "priority" in fields and data.priority:
        patch["priority"] = data.priority.value
    if "labels" in fields:
        patch["labels"] = data.labels or []
    if "review_required" in fields and data.review_required is not None:
        patch["reviewRequired"] = data.review_required
    if "due_date" in fields:
        patch["dueDate"] = to_datetime(data.due_date)
    if "parent_id" in fields and data.parent_id != row.parentId:
        patch["parentId"] = await _validated_parent(user_id, task_id, data.parent_id)
    if "assignee_type" in fields or "assignee_id" in fields:
        await _apply_assignee_change(user_id, row, data, actor, patch)
    if patch:
        await prisma.models.Task.prisma().update(where={"id": task_id}, data=patch)
    if "status" in fields and data.status and data.status.value != str(getattr(row.status, "value", row.status)):
        await set_task_status(user_id, task_id, data.status, actor)
    detail = await get_task_detail(user_id, task_id)
    assert detail is not None
    return detail


async def _validated_parent(
    user_id: str, task_id: str, parent_id: Optional[str]
) -> Optional[str]:
    if parent_id is None:
        return None
    if parent_id == task_id:
        raise ValueError("A task can't be its own parent")
    parent = await _find_row(user_id, parent_id)
    if parent is None:
        raise ValueError("Parent task not found")
    if parent.parentId:
        raise ValueError("Sub-tasks can't have their own sub-tasks")
    return parent.id


async def _apply_assignee_change(
    user_id: str,
    row: prisma.models.Task,
    data: TaskUpdate,
    actor: Actor,
    patch: dict[str, Any],
) -> None:
    before = row_to_task(row).assignee
    after = await resolve_assignee(user_id, data.assignee_type, data.assignee_id)
    before_ref = (before.type, before.id) if before else None
    after_ref = (after.type, after.id) if after else None
    if before_ref == after_ref:
        return
    patch["assigneeType"] = after.type.value if after else None
    patch["assigneeExpertId"] = (
        after.id if after and after.type == AssigneeType.EXPERT else None
    )
    await add_activity(
        user_id,
        row.id,
        ActivityKind.ASSIGNED,
        actor,
        data={
            "from": before.model_dump() if before else None,
            "to": after.model_dump() if after else None,
        },
    )


async def delete_task(user_id: str, task_id: str) -> bool:
    deleted = await prisma.models.Task.prisma().delete_many(
        where={"id": task_id, "userId": user_id}
    )
    return deleted > 0


async def set_task_status(
    user_id: str,
    task_id: str,
    status: TaskStatus,
    actor: Actor,
    note: Optional[str] = None,
    outcome_summary: Optional[str] = None,
) -> Task:
    row = await _find_row(user_id, task_id)
    if row is None:
        raise LookupError("Task not found")
    previous = str(getattr(row.status, "value", row.status))
    now = datetime.now(timezone.utc)
    patch: dict[str, Any] = {"status": status.value}
    if status == TaskStatus.IN_PROGRESS and row.startedAt is None:
        patch["startedAt"] = now
    if status == TaskStatus.DONE:
        patch["completedAt"] = now
    elif row.completedAt is not None:
        patch["completedAt"] = None
    if status in (TaskStatus.DONE, TaskStatus.CANCELLED):
        patch["liveRunning"] = False
        patch["liveStep"] = None
    if outcome_summary is not None:
        patch["outcomeSummary"] = outcome_summary
    await prisma.models.Task.prisma().update(where={"id": task_id}, data=patch)
    if previous != status.value:
        await add_activity(
            user_id,
            task_id,
            ActivityKind.STATUS_CHANGED,
            actor,
            body=note,
            data={"from": previous, "to": status.value},
        )
    if outcome_summary:
        await add_activity(
            user_id, task_id, ActivityKind.OUTCOME, actor, body=outcome_summary
        )
    task = await get_task(user_id, task_id)
    assert task is not None
    return task


async def add_activity(
    user_id: str,
    task_id: str,
    kind: ActivityKind,
    actor: Actor,
    body: Optional[str] = None,
    data: Optional[dict] = None,
) -> ActivityEvent:
    row = await prisma.models.TaskActivity.prisma().create(
        data={
            "taskId": task_id,
            "userId": user_id,
            "kind": kind.value,
            "actorType": actor.type.value,
            "actorId": actor.id,
            "actorName": actor.name,
            "body": body,
            "data": Json(data or {}),
        }
    )
    return row_to_activity(row)


async def update_activity(
    event_id: str, data: Optional[dict] = None, body: Optional[str] = None
) -> ActivityEvent:
    row = await prisma.models.TaskActivity.prisma().find_unique(where={"id": event_id})
    if row is None:
        raise LookupError("Activity not found")
    patch: dict[str, Any] = {}
    if data is not None:
        merged = dict(row.data) if isinstance(row.data, dict) else {}
        merged.update(data)
        patch["data"] = Json(merged)
    if body is not None:
        patch["body"] = body
    if not patch:
        return row_to_activity(row)
    updated = await prisma.models.TaskActivity.prisma().update(
        where={"id": event_id}, data=patch
    )
    assert updated is not None
    return row_to_activity(updated)


async def list_activity(user_id: str, task_id: str) -> list[ActivityEvent]:
    rows = await prisma.models.TaskActivity.prisma().find_many(
        where={"taskId": task_id, "userId": user_id}, order={"createdAt": "asc"}
    )
    return [row_to_activity(r) for r in rows]


async def add_comment(
    user_id: str, task_id: str, body: str, actor: Actor
) -> ActivityEvent:
    experts = await _user_experts(user_id)
    mentions = parse_mentions(body, [(e.id, e.name) for e in experts])
    return await add_activity(
        user_id,
        task_id,
        ActivityKind.COMMENT,
        actor,
        body=body,
        data={"mentions": mentions},
    )


def parse_mentions(body: str, experts: list[tuple[str, str]]) -> list[dict[str, str]]:
    lowered = body.lower()
    found: list[dict[str, str]] = []
    for expert_id, name in experts:
        candidates = {name.lower(), name.split()[0].lower()} if name.strip() else set()
        for candidate in candidates:
            pattern = r"@" + re.escape(candidate) + r"(?![\w])"
            if re.search(pattern, lowered):
                found.append({"expert_id": expert_id, "name": name})
                break
    return found


async def request_input(
    user_id: str, task_id: str, question: str, options: list[str], actor: Actor
) -> PendingInput:
    pending = PendingInput(
        id=str(uuid.uuid4()),
        question=question,
        options=options,
        asked_by=actor,
        asked_at=datetime.now(timezone.utc),
    )
    await prisma.models.Task.prisma().update(
        where={"id": task_id},
        data={"pendingInput": Json(pending.model_dump(mode="json"))},
    )
    await set_task_status(user_id, task_id, TaskStatus.BLOCKED, actor)
    await add_activity(
        user_id,
        task_id,
        ActivityKind.INPUT_REQUESTED,
        actor,
        body=question,
        data={"input_id": pending.id, "options": options},
    )
    return pending


async def answer_input(
    user_id: str, task_id: str, answer: str, input_id: Optional[str], actor: Actor
) -> Optional[PendingInput]:
    task = await get_task(user_id, task_id)
    if task is None:
        raise LookupError("Task not found")
    pending = task.pending_input
    if pending is not None and input_id and input_id != pending.id:
        raise ValueError("That question has already been answered")
    await prisma.models.Task.prisma().update(
        where={"id": task_id}, data={"pendingInput": Json({})}
    )
    await set_task_status(user_id, task_id, TaskStatus.IN_PROGRESS, actor)
    await add_activity(
        user_id,
        task_id,
        ActivityKind.INPUT_ANSWERED,
        actor,
        body=answer,
        data={"input_id": pending.id if pending else input_id},
    )
    return pending


async def add_attachment(
    user_id: str, task_id: str, data: AttachmentCreate, actor: Actor
) -> Attachment:
    if await get_task(user_id, task_id) is None:
        raise LookupError("Task not found")
    row = await prisma.models.TaskAttachment.prisma().create(
        data={
            "taskId": task_id,
            "kind": data.kind,
            "title": data.title,
            "refId": data.ref_id,
            "url": data.url,
            "addedBy": Json(actor.model_dump(mode="json")),
        }
    )
    attachment = row_to_attachment(row)
    await add_activity(
        user_id,
        task_id,
        ActivityKind.ATTACHMENT_ADDED,
        actor,
        data={"attachment": attachment.model_dump(mode="json")},
    )
    return attachment


async def set_work_session(task_id: str, session_id: str) -> None:
    await prisma.models.Task.prisma().update(
        where={"id": task_id}, data={"workSessionId": session_id}
    )


async def get_task_for_work_session(session_id: str) -> Optional[tuple[str, Task]]:
    row = await prisma.models.Task.prisma().find_first(
        where={"workSessionId": session_id}, include=TASK_INCLUDE
    )
    if row is None:
        return None
    return row.userId, row_to_task(row)


async def set_live(
    task_id: str,
    running: bool,
    current_step: Optional[str] = None,
    turn_started_at: Optional[datetime] = None,
) -> None:
    await prisma.models.Task.prisma().update(
        where={"id": task_id},
        data={
            "liveRunning": running,
            "liveStep": current_step,
            "liveTurnStartedAt": turn_started_at,
        },
    )


async def add_turn_metrics(
    task_id: str, tool_calls: int, tokens: int, cost_usd: float, duration_ms: int
) -> Task:
    row = await prisma.models.Task.prisma().update(
        where={"id": task_id},
        data={
            "turns": {"increment": 1},
            "toolCalls": {"increment": max(0, tool_calls)},
            "tokens": {"increment": max(0, tokens)},
            "costUsd": {"increment": max(0.0, cost_usd)},
            "durationMs": {"increment": max(0, duration_ms)},
        },
        include=TASK_INCLUDE,
    )
    if row is None:
        raise LookupError("Task not found")
    return row_to_task(row)


async def get_busy_task_for_expert(
    user_id: str, expert_id: str, exclude_task_id: Optional[str] = None
) -> Optional[Task]:
    where: dict[str, Any] = {
        "userId": user_id,
        "assigneeType": "expert",
        "assigneeExpertId": expert_id,
        "status": {"in": [s.value for s in _ACTIVE_STATUSES]},
    }
    if exclude_task_id:
        where["id"] = {"not": exclude_task_id}
    row = await prisma.models.Task.prisma().find_first(
        where=where,
        include=TASK_INCLUDE,
        order={"startedAt": "asc"},
    )
    return row_to_task(row) if row else None


async def list_waiting_tasks_for_expert(user_id: str, expert_id: str) -> list[Task]:
    rows = await prisma.models.Task.prisma().find_many(
        where={
            "userId": user_id,
            "assigneeType": "expert",
            "assigneeExpertId": expert_id,
            "status": "todo",
        },
        include=TASK_INCLUDE,
        order={"createdAt": "asc"},
    )
    return [row_to_task(r) for r in rows]


async def _user_experts(user_id: str) -> list[prisma.models.Expert]:
    return await prisma.models.Expert.prisma().find_many(
        where={"ownerUserId": user_id, "isArchived": False, "isTemplate": False},
        order={"createdAt": "asc"},
    )


async def list_experts_for_user(user_id: str) -> list[ExpertOption]:
    experts = await _user_experts(user_id)
    busy_rows = await prisma.models.Task.prisma().find_many(
        where={
            "userId": user_id,
            "assigneeType": "expert",
            "status": {"in": [s.value for s in _ACTIVE_STATUSES]},
        },
        order={"startedAt": "asc"},
    )
    busy: dict[str, prisma.models.Task] = {}
    for row in busy_rows:
        if row.assigneeExpertId and row.assigneeExpertId not in busy:
            busy[row.assigneeExpertId] = row
    return [
        ExpertOption(
            id=e.id,
            name=e.name,
            role=e.jobTitle or e.role,
            avatar_url=e.avatarUrl,
            color=e.color or None,
            busy_task_id=busy[e.id].id if e.id in busy else None,
            busy_task_key=task_key(busy[e.id].number) if e.id in busy else None,
        )
        for e in experts
    ]


async def confirm_plan(
    user_id: str, plan_id: str, steps: Optional[list[PlanStep]]
) -> tuple[Plan, TaskDetail]:
    plan = await get_plan(user_id, plan_id)
    if plan is None:
        raise LookupError("Plan not found")
    if plan.status != PlanStatus.PROPOSED:
        raise ValueError(f"This plan is already {plan.status.value}")
    final_steps = steps if steps else plan.steps
    if not final_steps:
        raise ValueError("A plan needs at least one step")
    otto = Actor.otto()
    parent = await create_task(
        user_id,
        TaskCreate(
            title=plan.goal,
            status=TaskStatus.IN_PROGRESS,
            assignee=AssigneeRef(type="otto"),
            budget_cap_usd=plan.budget_usd,
        ),
        otto,
    )
    task_ids = [
        (await create_task(user_id, _step_task(step, parent.id), otto)).id
        for step in final_steps
    ]
    await prisma.models.Task.prisma().update_many(
        where={"id": {"in": [parent.id, *task_ids]}, "userId": user_id},
        data={"planId": plan_id},
    )
    await add_activity(
        user_id,
        parent.id,
        ActivityKind.PLAN_CONFIRMED,
        Actor.user(user_id),
        body=f"Plan approved with {len(task_ids)} steps",
        data={"plan_id": plan_id, "task_ids": task_ids},
    )
    row = await prisma.models.TaskPlan.prisma().update(
        where={"id": plan_id},
        data={
            "status": PlanStatus.CONFIRMED.value,
            "steps": Json([s.model_dump(mode="json") for s in final_steps]),
            "parentTaskId": parent.id,
            "taskIds": task_ids,
        },
    )
    assert row is not None
    detail = await get_task_detail(user_id, parent.id)
    assert detail is not None
    return row_to_plan(row), detail


def _step_task(step: PlanStep, parent_id: str) -> TaskCreate:
    assignee = step.assignee
    return TaskCreate(
        title=step.title,
        description=step.description,
        parent_id=parent_id,
        assignee=(
            AssigneeRef(type=assignee.type, id=assignee.id) if assignee else None
        ),
    )
