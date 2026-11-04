"""Read and patch a task's contract.

Each write is one ``UPDATE ... WHERE contractVersion = <read version>`` that
bumps the version; a write that lost the race re-reads and re-applies, so
parallel tool calls (five hires at once) never drop each other's patch. Every
patch also lands in the activity log, which keeps the full record (raw
evidence content included) while the contract stays small.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import prisma.models
from prisma import Json
from pydantic import BaseModel

from backend.api.features.tasks import approvals as card_approvals
from backend.api.features.tasks import contract as c
from backend.api.features.tasks import db
from backend.api.features.tasks.converters import (
    TASK_INCLUDE,
    row_to_activity,
    row_to_attachment,
    row_to_plan,
    row_to_task,
)
from backend.api.features.tasks.models import (
    ActivityEvent,
    ActivityKind,
    Actor,
    Assignee,
    Attachment,
    CardApproval,
    ContractState,
    PendingInput,
    Plan,
    SubtaskCounts,
    Task,
    TaskLive,
    TaskMetrics,
    TaskStatus,
)

MAX_CONTENT_CHARS = 200_000
MAX_AUTO_EVIDENCE = 20
_WRITE_ATTEMPTS = 8
_ATTACHABLE = {"link": "url", "file": "file", "screenshot": "file"}
_SETTLED = (TaskStatus.DONE, TaskStatus.CANCELLED, TaskStatus.IN_REVIEW)


class ContractUpdate(BaseModel):
    """A patched contract, what changed, and the event to publish for it."""

    task: Task
    contract: c.Contract
    version: int
    patches: list[c.ContractPatch]
    activity: ActivityEvent | None = None
    evidence: c.Evidence | None = None
    status_event: str | None = None


class TaskSummary(BaseModel):
    """The contract with what the UI needs around it, and nothing else."""

    id: str
    key: str
    title: str
    status: TaskStatus
    state: ContractState
    raw_status: str
    now: str
    work_session_id: str | None = None
    version: int
    derived: bool
    contract: c.Contract
    assignee: Assignee | None = None
    live: TaskLive
    metrics: TaskMetrics
    subtask_counts: SubtaskCounts
    parent_id: str | None = None
    updated_at: datetime


class EvidenceDetail(BaseModel):
    evidence: c.Evidence
    attachment: Attachment | None = None
    activity: ActivityEvent | None = None


@dataclass
class _Written:
    contract: c.Contract
    version: int
    patches: list[c.ContractPatch]
    task: Task
    previous: c.Contract
    derived: bool


async def _row(user_id: str, task_id: str) -> prisma.models.Task:
    row = await prisma.models.Task.prisma().find_first(
        where={"id": task_id, "userId": user_id}, include=TASK_INCLUDE
    )
    if row is None:
        raise LookupError("Task not found")
    return row


async def _base(user_id: str, row: prisma.models.Task) -> tuple[c.Contract, bool]:
    """The stored contract, or the one the task implies when it has none."""
    stored = c.parse_contract(row.contract)
    if stored is not None:
        return stored, False
    task = row_to_task(row)
    attachments, plan, note = await _fallback_inputs(user_id, task)
    return c.fallback_contract(task, attachments, plan, note), True


async def _fallback_inputs(
    user_id: str, task: Task
) -> tuple[list[Attachment], Plan | None, str | None]:
    rows = await prisma.models.TaskAttachment.prisma().find_many(
        where={"taskId": task.id}, order={"createdAt": "asc"}
    )
    plan = None
    if task.plan_id:
        plan_row = await prisma.models.TaskPlan.prisma().find_first(
            where={"id": task.plan_id, "userId": user_id}
        )
        plan = row_to_plan(plan_row) if plan_row else None
    note = None
    if task.status == TaskStatus.BLOCKED and task.pending_input is None:
        last = await prisma.models.TaskActivity.prisma().find_first(
            where={
                "taskId": task.id,
                "kind": {
                    "in": [
                        ActivityKind.SYSTEM.value,
                        ActivityKind.STATUS_CHANGED.value,
                    ]
                },
                "body": {"not": None},
            },
            order={"createdAt": "desc"},
        )
        note = last.body if last else None
    return [row_to_attachment(r) for r in rows], plan, note


Apply = Callable[[c.Contract, Task], tuple[c.Contract, list[c.ContractPatch]]]


async def _patch(
    user_id: str,
    task_id: str,
    apply: Apply,
    extra: Callable[[c.Contract], dict[str, Any]] | None = None,
) -> _Written:
    """Apply *apply* to the current contract and write it, retrying when a
    parallel write got there first. *extra* adds column writes to the same
    update."""
    for _ in range(_WRITE_ATTEMPTS):
        row = await _row(user_id, task_id)
        base, derived = await _base(user_id, row)
        task = row_to_task(row)
        contract, patches = apply(base, task)
        if not patches:
            return _Written(contract, row.contractVersion, [], task, base, derived)
        version = row.contractVersion + 1
        data: dict[str, Any] = {
            "contract": Json(contract.model_dump(mode="json")),
            "contractVersion": version,
            **(extra(contract) if extra else {}),
        }
        written = await prisma.models.Task.prisma().update_many(
            where={"id": task_id, "contractVersion": row.contractVersion},
            data=data,  # type: ignore[arg-type]
        )
        if written:
            fresh = await db.get_task(user_id, task_id)
            assert fresh is not None
            return _Written(contract, version, patches, fresh, base, derived)
    raise c.ContractError("The task changed too often to save this; try again.")


async def _log(
    user_id: str,
    task_id: str,
    kind: ActivityKind,
    actor: Actor,
    body: str | None = None,
    data: dict[str, Any] | None = None,
    activity_id: str | None = None,
) -> ActivityEvent:
    row = await prisma.models.TaskActivity.prisma().create(
        data={
            **({"id": activity_id} if activity_id else {}),
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


async def _set_status(
    user_id: str, task: Task, status: TaskStatus, actor: Actor
) -> Task:
    if task.status == status:
        return task
    return await db.set_task_status(user_id, task.id, status, actor)


def _update(written: _Written, **kwargs: Any) -> ContractUpdate:
    return ContractUpdate(
        task=written.task,
        contract=written.contract,
        version=written.version,
        patches=written.patches,
        **kwargs,
    )


async def update_task(
    user_id: str, task_id: str, patch: dict[str, Any], actor: Actor
) -> ContractUpdate:
    """One ``update_task`` call: any of goal, done_when, plan, step and
    criterion changes, a blocker or the result, in one versioned write."""
    change = c.TaskPatch.model_validate(patch)
    running = {s.id for s in change.steps if s.status == c.StepStatus.RUNNING}

    def extra(contract: c.Contract) -> dict[str, Any]:
        data: dict[str, Any] = {}
        title = next((s.title for s in contract.plan if s.id in running), None)
        if title:
            data["liveStep"] = title
        if change.result is not None:
            data["pendingInput"] = Json({})
        return data

    written = await _patch(
        user_id, task_id, lambda base, _t: c.apply_patch(base, change), extra
    )
    contract, version, previous = written.contract, written.version, written.previous
    activity: ActivityEvent | None = None
    status_event = None
    revised = {"goal", "done_when", "plan"} & {p.field for p in written.patches}
    if revised:
        goal_changed = (
            not written.derived
            and bool(previous.done_when or previous.plan)
            and previous.goal != contract.goal
        )
        activity = await _log(
            user_id,
            task_id,
            ActivityKind.GOAL_UPDATED if goal_changed else ActivityKind.CONTRACT_SET,
            actor,
            body=contract.goal,
            data={
                "version": version,
                "previous_goal": previous.goal if goal_changed else None,
                "done_when": [x.text for x in contract.done_when],
                "plan": [s.title for s in contract.plan],
            },
        )
    steps = {s.id: s for s in contract.plan}
    for step in change.steps:
        now = steps[step.id]
        activity = await _log(
            user_id,
            task_id,
            ActivityKind.STEP_UPDATED,
            actor,
            body=now.title,
            data={
                "step_id": step.id,
                "status": now.status.value,
                "rows": [r.title for r in step.rows or []],
                "version": version,
            },
        )
    texts = {x.id: x.text for x in contract.done_when}
    for criterion in change.criteria:
        activity = await _log(
            user_id,
            task_id,
            ActivityKind.CRITERION_MARKED,
            actor,
            body=texts.get(criterion.id),
            data={
                "criterion_id": criterion.id,
                "status": criterion.status.value,
                "evidence_ids": criterion.evidence_ids,
                "version": version,
            },
        )
    task = written.task
    if change.result is not None and contract.result is not None:
        finished = c.all_met(contract) or not contract.done_when
        task = await db.set_task_status(
            user_id,
            task_id,
            TaskStatus.DONE if finished else TaskStatus.IN_REVIEW,
            actor,
            outcome_summary=contract.result.summary,
        )
        status_event = "task_completed" if finished else "task_needs_input"
    elif change.blocker is not None and contract.blocker is not None:
        task = await _set_status(user_id, task, TaskStatus.BLOCKED, actor)
        activity = await _log(
            user_id,
            task_id,
            ActivityKind.BLOCKER_REPORTED,
            actor,
            body=contract.blocker.what_stopped_it,
            data={**contract.blocker.model_dump(mode="json"), "version": version},
        )
        status_event = "task_needs_input"
    written.task = task
    return _update(written, activity=activity, status_event=status_event)


async def add_evidence(
    user_id: str,
    task_id: str,
    kind: str,
    title: str,
    summary: str,
    actor: Actor,
    ref: str | None = None,
    url: str | None = None,
    criterion_ids: list[str] | None = None,
    content: str | None = None,
    attachment_id: str | None = None,
) -> ContractUpdate:
    """Add one proof item. Raw output (HTML, a table) goes in *content*,
    which is kept on the activity row and fetched on demand; the contract
    holds only the one-line summary. A link or file also becomes an
    attachment unless *attachment_id* names the one it already is."""
    kind = c.one_line(kind, 40).lower() or "record"
    activity_id = str(uuid.uuid4())
    if attachment_id is None and kind in _ATTACHABLE and (url or ref):
        attachment = await prisma.models.TaskAttachment.prisma().create(
            data={
                "taskId": task_id,
                "kind": _ATTACHABLE[kind],
                "title": c.one_line(title, 120),
                "refId": ref,
                "url": url,
                "addedBy": Json(actor.model_dump(mode="json")),
            }
        )
        attachment_id = attachment.id
    added: dict[str, c.Evidence] = {}

    def apply(base: c.Contract, _task: Task):
        updated, item, patches = c.add_evidence(
            base,
            kind,
            title,
            summary,
            ref=ref,
            url=url,
            criterion_ids=criterion_ids,
            attachment_id=attachment_id,
            source_activity_id=activity_id,
        )
        added["item"] = item
        return updated, patches

    written = await _patch(user_id, task_id, apply)
    item = added["item"]
    activity = await _log(
        user_id,
        task_id,
        ActivityKind.EVIDENCE_ADDED,
        actor,
        body=item.summary,
        data={
            **item.model_dump(mode="json"),
            "evidence_id": item.id,
            "version": written.version,
            "content": content[:MAX_CONTENT_CHARS] if content else None,
            "content_chars": len(content) if content else 0,
        },
        activity_id=activity_id,
    )
    return _update(written, activity=activity, evidence=item)


async def ask(
    user_id: str,
    task_id: str,
    question: str,
    actor: Actor,
    why: str | None = None,
    options: list[str] | None = None,
    default_option: str | None = None,
    blocking_step_id: str | None = None,
    expires_at: datetime | None = None,
) -> ContractUpdate:
    """Open the task's one question (or fold this one into it) and move the
    task to Needs you. Work that doesn't depend on the answer goes on."""
    opened: dict[str, c.Ask] = {}

    def apply(base: c.Contract, _task: Task):
        updated, current, patches = c.open_ask(
            base,
            question,
            why=why,
            options=options,
            default_option=default_option,
            blocking_step_id=blocking_step_id,
            expires_at=expires_at,
        )
        opened["ask"] = current
        return updated, patches

    def extra(contract: c.Contract) -> dict[str, Any]:
        current = contract.ask
        assert current is not None
        pending = PendingInput(
            id=current.id,
            question=current.question,
            options=current.options,
            asked_by=actor,
            asked_at=datetime.now(UTC),
        )
        return {"pendingInput": Json(pending.model_dump(mode="json"))}

    written = await _patch(user_id, task_id, apply, extra)
    current = opened["ask"]
    written.task = await _set_status(user_id, written.task, TaskStatus.BLOCKED, actor)
    activity = await _log(
        user_id,
        task_id,
        ActivityKind.INPUT_REQUESTED,
        actor,
        body=question,
        data={
            "input_id": current.id,
            "options": current.options,
            "why": current.why,
            "default_option": current.default_option,
            "blocking_step_id": current.blocking_step_id,
            "merged": current.question != question.strip(),
            "version": written.version,
        },
    )
    return _update(written, activity=activity, status_event="task_needs_input")


async def add_approval(
    user_id: str, task_id: str, card: CardApproval, actor: Actor
) -> ContractUpdate:
    """Keep a hire or raise card proposed in the task's chat on the task and
    move it to Needs you, so the user can answer it from the board."""
    written = await _patch(
        user_id, task_id, lambda base, _task: card_approvals.add_approval(base, card)
    )
    if not written.patches:
        return _update(written)
    written.task = await _set_status(user_id, written.task, TaskStatus.BLOCKED, actor)
    verb = "Hire" if card.kind == "hire" else "Create"
    activity = await _log(
        user_id,
        task_id,
        ActivityKind.INPUT_REQUESTED,
        actor,
        body=f"{verb} {card.name}? Approve the card to go on.",
        data={"confirmation_id": card.confirmation_id, "version": written.version},
    )
    return _update(written, activity=activity, status_event="task_needs_input")


async def drop_approvals(
    user_id: str, task_id: str, confirmation_ids: list[str]
) -> ContractUpdate:
    """Forget cards that were answered (on the board or in the chat)."""
    ids = set(confirmation_ids)
    written = await _patch(
        user_id, task_id, lambda base, _task: card_approvals.drop_approvals(base, ids)
    )
    return _update(written)


async def close_ask(user_id: str, task_id: str) -> None:
    """Clear the open question once it is answered (``db.answer_input``)."""
    row = await _row(user_id, task_id)
    if c.parse_contract(row.contract) is None:
        return
    await _patch(user_id, task_id, lambda base, _task: c.close_ask(base))


async def fill_from_turn(
    user_id: str,
    task_id: str,
    session_id: str,
    since: datetime,
    blocker_reason: str | None = None,
    blocker_tool: str | None = None,
) -> list[ContractUpdate]:
    """What the harness adds when a work turn ends, so the record stays
    current even when the worker forgot a call: each durable effect the turn
    logged (a file written, a run, an integration call) as evidence, and a
    failure it never got past as the blocker."""
    events = await prisma.models.ActivityEvent.prisma().find_many(
        where={"userId": user_id, "sessionId": session_id, "createdAt": {"gte": since}},
        order={"createdAt": "asc"},
        take=MAX_AUTO_EVIDENCE,
    )
    found = [
        item
        for e in events
        if (
            item := c.evidence_from_event(
                str(getattr(e.category, "value", e.category)),
                e.eventType,
                e.title,
                e.objectId,
                dict(e.data) if isinstance(e.data, dict) else {},
            )
        )
    ]
    updates: list[ContractUpdate] = []
    if found:
        update = await _add_auto_evidence(user_id, task_id, found)
        if update is not None:
            updates.append(update)
    if blocker_reason:
        update = await _auto_block(user_id, task_id, blocker_reason, blocker_tool)
        if update is not None:
            updates.append(update)
    return updates


async def _add_auto_evidence(
    user_id: str, task_id: str, found: list[dict[str, Any]]
) -> ContractUpdate | None:
    added: list[c.Evidence] = []

    def apply(base: c.Contract, _task: Task):
        added.clear()
        seen = {(e.ref or e.title) for e in base.evidence}
        contract, patches = base, []
        for raw in found:
            key = raw["ref"] or raw["title"]
            if key in seen:
                continue
            seen.add(key)
            contract, item, more = c.add_evidence(
                contract,
                raw["kind"],
                raw["title"],
                raw["summary"],
                ref=raw["ref"],
                auto=True,
            )
            added.append(item)
            patches.extend(more)
        return contract, patches

    written = await _patch(user_id, task_id, apply)
    if not added:
        return None
    activity = None
    for item in added:
        activity = await _log(
            user_id,
            task_id,
            ActivityKind.EVIDENCE_ADDED,
            Actor.system(),
            body=item.summary,
            data={
                **item.model_dump(mode="json"),
                "evidence_id": item.id,
                "version": written.version,
            },
        )
    return _update(written, activity=activity)


async def _auto_block(
    user_id: str, task_id: str, reason: str, tool: str | None
) -> ContractUpdate | None:
    """A failed turn's blocker, unless the task is settled, already stuck, or
    waiting on the user's answer."""
    skipped: dict[str, bool] = {}

    def apply(base: c.Contract, task: Task):
        if task.status in _SETTLED or base.blocker is not None or base.ask:
            skipped["yes"] = True
            return base, []
        return c.set_blocker(base, c.make_blocker(base, reason, tool_name=tool))

    written = await _patch(user_id, task_id, apply)
    if skipped or written.contract.blocker is None:
        return None
    blocker = written.contract.blocker
    written.task = await _set_status(
        user_id, written.task, TaskStatus.BLOCKED, Actor.system()
    )
    activity = await _log(
        user_id,
        task_id,
        ActivityKind.BLOCKER_REPORTED,
        Actor.system(),
        body=blocker.what_stopped_it,
        data={
            **blocker.model_dump(mode="json"),
            "auto": True,
            "tool": tool,
            "version": written.version,
        },
    )
    return _update(written, activity=activity, status_event="task_needs_input")


async def clear_blocker(user_id: str, task_id: str) -> ContractUpdate:
    written = await _patch(user_id, task_id, lambda base, _t: c.clear_blocker(base))
    return _update(written)


async def change_goal(
    user_id: str, task_id: str, goal: str, actor: Actor
) -> ContractUpdate:
    def apply(base: c.Contract, _task: Task):
        updated, patches = c.revise(base, goal=goal)
        cleared, more = c.clear_blocker(updated)
        return cleared, [*patches, *more]

    written = await _patch(user_id, task_id, apply)
    activity = await _log(
        user_id,
        task_id,
        ActivityKind.GOAL_UPDATED,
        actor,
        body=written.contract.goal,
        data={"previous_goal": written.previous.goal, "version": written.version},
    )
    return _update(written, activity=activity)


async def log_action(
    user_id: str, task_id: str, action: str, actor: Actor, body: str
) -> ActivityEvent:
    return await _log(
        user_id,
        task_id,
        ActivityKind.ACTION_TAKEN,
        actor,
        body=body,
        data={"action": action},
    )


async def get_contract(user_id: str, task_id: str) -> c.Contract:
    row = await _row(user_id, task_id)
    contract, _derived = await _base(user_id, row)
    return contract


async def get_summary(user_id: str, task_id: str) -> TaskSummary | None:
    """One task read; three more only for a task without a stored contract."""
    try:
        row = await _row(user_id, task_id)
    except LookupError:
        return None
    contract, derived = await _base(user_id, row)
    task = row_to_task(row)
    if contract.result is None and task.outcome_summary:
        # Finished with update_task_status rather than update_task.
        contract = contract.model_copy(
            update={
                "result": c.Result(
                    summary=c.one_line(task.outcome_summary, c.RESULT_CHARS),
                    primary_evidence_id=(
                        contract.evidence[-1].id if contract.evidence else None
                    ),
                )
            }
        )
    if contract.blocker is None and task.blocker is not None:
        contract = contract.model_copy(update={"blocker": task.blocker})
    return TaskSummary(
        id=task.id,
        key=task.key,
        title=task.title,
        status=task.status,
        state=task.state,
        raw_status=task.raw_status,
        now=c.now_line(
            task.raw_status,
            contract,
            task.live.current_step,
            (task.subtask_counts.done, task.subtask_counts.total),
        ),
        work_session_id=task.work_session_id,
        version=row.contractVersion,
        derived=derived,
        contract=contract,
        assignee=task.assignee,
        live=task.live,
        metrics=task.metrics,
        subtask_counts=task.subtask_counts,
        parent_id=task.parent_id,
        updated_at=task.updated_at,
    )


async def get_evidence(
    user_id: str, task_id: str, evidence_id: str
) -> EvidenceDetail | None:
    try:
        contract = await get_contract(user_id, task_id)
    except LookupError:
        return None
    item = next((e for e in contract.evidence if e.id == evidence_id), None)
    if item is None:
        return None
    attachment = None
    if item.attachment_id:
        row = await prisma.models.TaskAttachment.prisma().find_first(
            where={"id": item.attachment_id, "taskId": task_id}
        )
        attachment = row_to_attachment(row) if row else None
    activity = None
    if item.source_activity_id:
        found = await prisma.models.TaskActivity.prisma().find_first(
            where={
                "id": item.source_activity_id,
                "taskId": task_id,
                "userId": user_id,
            }
        )
        activity = row_to_activity(found) if found else None
    return EvidenceDetail(evidence=item, attachment=attachment, activity=activity)


async def list_activity_page(
    user_id: str, task_id: str, cursor: str | None, limit: int
) -> tuple[list[ActivityEvent], str | None]:
    """Oldest first, *limit* at a time; pass the returned cursor for more."""
    rows = await prisma.models.TaskActivity.prisma().find_many(
        where={"taskId": task_id, "userId": user_id},
        order=[{"createdAt": "asc"}, {"id": "asc"}],
        take=limit + 1,
        **({"cursor": {"id": cursor}, "skip": 1} if cursor else {}),
    )
    page = rows[:limit]
    next_cursor = page[-1].id if len(rows) > limit and page else None
    return [row_to_activity(r) for r in page], next_cursor
