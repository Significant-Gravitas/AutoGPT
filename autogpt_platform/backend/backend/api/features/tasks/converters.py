from datetime import date, datetime, time, timezone
from typing import Any, Optional

import prisma.models
import prisma.types

from backend.api.features.tasks.models import (
    ActivityEvent,
    ActivityKind,
    Actor,
    ActorType,
    Assignee,
    AssigneeType,
    Attachment,
    PendingInput,
    Plan,
    PlanStatus,
    PlanStep,
    SubtaskCounts,
    Task,
    TaskDetail,
    TaskLive,
    TaskMetrics,
    TaskParentRef,
    TaskPriority,
    TaskStatus,
)

TASK_INCLUDE: prisma.types.TaskInclude = {
    "AssigneeExpert": True,
    "Parent": True,
    "Subtasks": True,
}


def task_key(number: int) -> str:
    return f"T-{number}"


def to_datetime(value: Optional[date]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def row_assignee(row: prisma.models.Task) -> Optional[Assignee]:
    if row.assigneeType is None:
        return None
    kind = AssigneeType(str(getattr(row.assigneeType, "value", row.assigneeType)))
    if kind == AssigneeType.USER:
        return Assignee(type=kind, id=row.userId, name="Me")
    if kind == AssigneeType.OTTO:
        return Assignee(type=kind, id=None, name="Otto")
    expert = row.AssigneeExpert
    if expert is None:
        return Assignee(type=kind, id=row.assigneeExpertId, name="Expert")
    return expert_assignee(expert)


def expert_assignee(expert: prisma.models.Expert) -> Assignee:
    return Assignee(
        type=AssigneeType.EXPERT,
        id=expert.id,
        name=expert.name,
        avatar_url=expert.avatarUrl,
        color=expert.color or None,
    )


def row_to_task(row: prisma.models.Task) -> Task:
    subtasks = row.Subtasks or []
    parent = row.Parent
    return Task(
        id=row.id,
        number=row.number,
        key=task_key(row.number),
        title=row.title,
        description=row.description,
        status=TaskStatus(str(getattr(row.status, "value", row.status))),
        priority=TaskPriority(str(getattr(row.priority, "value", row.priority))),
        assignee=row_assignee(row),
        due_date=row.dueDate.date() if row.dueDate else None,
        labels=list(row.labels or []),
        parent_id=row.parentId,
        parent_key=task_key(parent.number) if parent else None,
        outcome_summary=row.outcomeSummary,
        work_session_id=row.workSessionId,
        budget_cap_usd=row.budgetCapUsd,
        review_required=row.reviewRequired,
        metrics=TaskMetrics(
            turns=row.turns,
            tool_calls=row.toolCalls,
            tokens=row.tokens,
            cost_usd=row.costUsd,
            duration_ms=row.durationMs,
        ),
        subtask_counts=SubtaskCounts(
            total=len(subtasks),
            done=sum(1 for s in subtasks if str(getattr(s.status, "value", s.status)) == "done"),
        ),
        pending_input=_pending_input(row.pendingInput),
        live=TaskLive(
            running=row.liveRunning,
            current_step=row.liveStep,
            turn_started_at=row.liveTurnStartedAt,
        ),
        created_at=row.createdAt,
        updated_at=row.updatedAt,
        started_at=row.startedAt,
        completed_at=row.completedAt,
        plan_id=row.planId,
    )


def row_to_detail(
    row: prisma.models.Task,
    subtasks: list[prisma.models.Task],
    attachments: list[prisma.models.TaskAttachment],
) -> TaskDetail:
    base = row_to_task(row)
    parent = row.Parent
    return TaskDetail(
        **base.model_dump(),
        subtasks=[row_to_task(s) for s in subtasks],
        attachments=[row_to_attachment(a) for a in attachments],
        parent=(
            TaskParentRef(id=parent.id, key=task_key(parent.number), title=parent.title)
            if parent
            else None
        ),
    )


def _pending_input(raw: Any) -> Optional[PendingInput]:
    if not raw or not isinstance(raw, dict):
        return None
    return PendingInput.model_validate(raw)


def _actor_from_json(raw: Any) -> Actor:
    if isinstance(raw, dict) and raw.get("type"):
        return Actor.model_validate(raw)
    return Actor.system()


def row_to_attachment(row: prisma.models.TaskAttachment) -> Attachment:
    return Attachment(
        id=row.id,
        kind=row.kind,
        title=row.title,
        ref_id=row.refId,
        url=row.url,
        created_at=row.createdAt,
        added_by=_actor_from_json(row.addedBy),
    )


def row_to_activity(row: prisma.models.TaskActivity) -> ActivityEvent:
    return ActivityEvent(
        id=row.id,
        task_id=row.taskId,
        created_at=row.createdAt,
        actor=Actor(
            type=ActorType(row.actorType),
            id=row.actorId,
            name=row.actorName,
        ),
        kind=ActivityKind(row.kind),
        body=row.body,
        data=row.data if isinstance(row.data, dict) else {},
    )


def row_to_plan(row: prisma.models.TaskPlan) -> Plan:
    steps = row.steps if isinstance(row.steps, list) else []
    return Plan(
        id=row.id,
        status=PlanStatus(row.status),
        goal=row.goal,
        budget_usd=row.budgetUsd,
        steps=[PlanStep.model_validate(s) for s in steps],
        session_id=row.sessionId,
        parent_task_id=row.parentTaskId,
        task_ids=list(row.taskIds or []),
        created_at=row.createdAt,
    )
