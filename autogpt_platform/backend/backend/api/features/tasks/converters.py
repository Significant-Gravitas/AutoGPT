from datetime import UTC, date, datetime, time
from typing import Any

import prisma.models
import prisma.types

from backend.api.features.tasks.contract import (
    card_blocker,
    now_line,
    parse_contract,
    raw_status,
    state_of,
)
from backend.api.features.tasks.models import (
    ActivityEvent,
    ActivityKind,
    Actor,
    ActorType,
    Ask,
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


def to_datetime(value: date | None) -> datetime | None:
    if value is None:
        return None
    return datetime.combine(value, time.min, tzinfo=UTC)


def row_assignee(row: prisma.models.Task) -> Assignee | None:
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
    status = TaskStatus(str(getattr(row.status, "value", row.status)))
    pending = _pending_input(row.pendingInput)
    sub_statuses = [str(getattr(s.status, "value", s.status)) for s in subtasks]
    sub_done = sum(1 for s in sub_statuses if s == "done")
    open_subtasks = sum(1 for s in sub_statuses if s not in ("done", "cancelled"))
    contract = parse_contract(row.contract)
    updated = row.updatedAt
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    raw = raw_status(
        status,
        contract,
        pending,
        running=row.liveRunning,
        quiet_minutes=(datetime.now(UTC) - updated).total_seconds() / 60,
        open_subtasks=open_subtasks,
    )
    return Task(
        id=row.id,
        number=row.number,
        key=task_key(row.number),
        title=row.title,
        description=row.description,
        status=status,
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
        subtask_counts=SubtaskCounts(total=len(subtasks), done=sub_done),
        pending_input=pending,
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
        state=state_of(raw),
        raw_status=raw,
        now=now_line(raw, contract, row.liveStep, (sub_done, len(subtasks))),
        ask=contract.ask if contract else _pending_ask(pending),
        blocker=card_blocker(contract, raw, status),
        contract_version=row.contractVersion,
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


def _pending_input(raw: Any) -> PendingInput | None:
    if not raw or not isinstance(raw, dict):
        return None
    return PendingInput.model_validate(raw)


def _pending_ask(pending: PendingInput | None) -> Ask | None:
    if pending is None:
        return None
    return Ask(
        id=pending.id,
        question=pending.question,
        options=pending.options,
        default_option=pending.options[0] if pending.options else None,
    )


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
        split=row.split,
        session_id=row.sessionId,
        parent_task_id=row.parentTaskId,
        task_ids=list(row.taskIds or []),
        created_at=row.createdAt,
    )
