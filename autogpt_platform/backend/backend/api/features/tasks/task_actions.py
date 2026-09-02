"""Phase-2 writers for the task spine: handoff, escalation, report, answer.

Same tenancy rule as ``tasks_db``: every statement filters on ``userId``.
These are called over the DatabaseManager RPC by Prisma-less copilot tools,
so refusals raise the two exception types ``backend.util.exceptions`` maps
across that boundary — ``TaskDelegationRefusedError`` (act on the message)
and ``TaskUpdateConflictError`` (re-read and retry).
"""

import logging
from datetime import UTC, datetime

import prisma.enums
import prisma.models
import prisma.types

from backend.util.exceptions import TaskDelegationRefusedError, TaskUpdateConflictError
from backend.util.json import SafeJson

from .errors import DelegatedTaskNotFoundError
from .mapping import TASK_INCLUDE as _TASK_INCLUDE
from .mapping import library_agents_by_graph as _library_agents_by_graph
from .mapping import to_model as _to_model
from .models import (
    MAX_TASK_HANDOFFS,
    MAX_TASK_QUESTION_OPTIONS,
    OPEN_TASK_STATUSES,
    TASK_ANSWER_MAX_LENGTH,
    TASK_OUTCOME_MAX_LENGTH,
    TASK_QUESTION_MAX_LENGTH,
    DelegatedTask,
    TaskAmendment,
    TaskEscalationTarget,
)

logger = logging.getLogger(__name__)

_OPEN_STATUSES = [prisma.enums.DelegatedTaskStatus(s) for s in OPEN_TASK_STATUSES]


async def handoff_delegated_task(
    user_id: str,
    task_id: str,
    *,
    to_expert_id: str,
    note: str,
    expected_updated_at: datetime,
) -> DelegatedTask:
    """Swap the task's owner, recording the hop in the amendments timeline.

    Optimistic concurrency: the write only lands if ``updatedAt`` still
    matches what the caller read, so two experts handing the same task off
    cannot both win — the loser gets a retryable conflict.
    """
    row = await _open_task(user_id, task_id)
    if row.handoffCount >= MAX_TASK_HANDOFFS:
        raise TaskDelegationRefusedError(
            f"Handoff refused: this task has already changed hands "
            f"{MAX_TASK_HANDOFFS} times. Finish it or escalate to the user "
            "with escalate_task instead."
        )
    if row.ownerId == to_expert_id:
        raise TaskDelegationRefusedError(
            "Handoff refused: that expert already owns this task."
        )

    trail = list(row.ancestorExpertIds)
    if to_expert_id not in trail:
        trail.append(to_expert_id)
    amendments = _appended(
        row,
        TaskAmendment(
            at=datetime.now(UTC),
            by=row.ownerId or "autopilot",
            note=note,
            kind="handoff",
            from_expert_id=row.ownerId,
            to_expert_id=to_expert_id,
        ),
    )
    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={"id": task_id, "userId": user_id, "updatedAt": expected_updated_at},
        data={
            "ownerId": to_expert_id,
            "handoffCount": row.handoffCount + 1,
            "ancestorExpertIds": trail,
            "amendments": amendments,
        },
    )
    if updated == 0:
        raise TaskUpdateConflictError(
            "The task changed while you were handing it off. Fetch it again "
            "and retry."
        )
    return await _reload(user_id, task_id)


async def escalate_delegated_task(
    user_id: str,
    task_id: str,
    *,
    question: str,
    options: list[str] | None = None,
    session_id: str | None = None,
    target: TaskEscalationTarget = "user",
) -> DelegatedTask:
    """Route a blocked task's question up.

    ``target="user"`` parks the task WAITING_USER; Home surfaces the
    question and the answer resumes the recorded session.
    ``target="manager"`` keeps the task WORKING and only records the
    escalation entry — the caller delivers the question into the
    delegator's session. Refused on a root task: there is no delegator
    above it, so the user is the only way up.
    """
    row = await _open_task(user_id, task_id)
    if target == "manager" and row.parentTaskId is None:
        raise TaskDelegationRefusedError(
            "This task has no delegator above it — nobody is managing it "
            'but the user. Escalate with target="user" instead.'
        )
    amendments = _appended(
        row,
        TaskAmendment(
            at=datetime.now(UTC),
            by=row.ownerId or "autopilot",
            note=question[:TASK_QUESTION_MAX_LENGTH],
            kind="escalation",
            question=question[:TASK_QUESTION_MAX_LENGTH],
            options=[
                option[:TASK_QUESTION_MAX_LENGTH]
                for option in (options or [])[:MAX_TASK_QUESTION_OPTIONS]
                if option.strip()
            ],
            session_id=session_id,
            target=target,
        ),
    )
    data: prisma.types.DelegatedTaskUpdateManyMutationInput = {"amendments": amendments}
    if target == "user":
        data["status"] = prisma.enums.DelegatedTaskStatus.WAITING_USER
    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={"id": task_id, "userId": user_id, "status": {"in": _OPEN_STATUSES}},
        data=data,
    )
    if updated == 0:
        raise TaskUpdateConflictError(
            "The task closed while you were escalating it. Fetch it again."
        )
    return await _reload(user_id, task_id)


async def report_delegated_task(
    user_id: str,
    task_id: str,
    *,
    outcome_summary: str,
) -> DelegatedTask:
    """Close the task DONE — refused while any subtask is still open, so a
    parent can never report a tree finished out from under its children."""
    await _open_task(user_id, task_id)
    open_children = await prisma.models.DelegatedTask.prisma().count(
        where={
            "parentTaskId": task_id,
            "userId": user_id,
            "status": {"in": _OPEN_STATUSES},
        }
    )
    if open_children > 0:
        raise TaskDelegationRefusedError(
            f"Cannot mark this task done: {open_children} open subtask"
            f"{'s' if open_children != 1 else ''} remain. Wait for them to "
            "finish, or cancel the ones no longer needed."
        )
    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={"id": task_id, "userId": user_id, "status": {"in": _OPEN_STATUSES}},
        data={
            "status": prisma.enums.DelegatedTaskStatus.DONE,
            "outcomeSummary": outcome_summary[:TASK_OUTCOME_MAX_LENGTH],
        },
    )
    if updated == 0:
        raise TaskUpdateConflictError("The task closed while you were reporting it.")
    return await _reload(user_id, task_id)


async def answer_delegated_task(
    user_id: str,
    task_id: str,
    *,
    answer: str,
) -> tuple[DelegatedTask, str | None]:
    """Record the user's answer to an escalation and put the task back to
    WORKING. Returns the task and the session the escalating expert was
    working in, so the caller can deliver the answer there and resume it."""
    row = await prisma.models.DelegatedTask.prisma().find_first(
        where={"id": task_id, "userId": user_id}
    )
    if row is None:
        # Route-facing (not RPC): the missing row maps to a plain 404.
        raise DelegatedTaskNotFoundError(task_id)
    if row.status != prisma.enums.DelegatedTaskStatus.WAITING_USER:
        raise TaskDelegationRefusedError("This task is not waiting on an answer.")

    worker_session_id = _latest_escalation_session(row)
    amendments = _appended(
        row,
        TaskAmendment(
            at=datetime.now(UTC),
            by="user",
            note=answer[:TASK_ANSWER_MAX_LENGTH],
            kind="answer",
        ),
    )
    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={
            "id": task_id,
            "userId": user_id,
            "status": prisma.enums.DelegatedTaskStatus.WAITING_USER,
        },
        data={
            "status": prisma.enums.DelegatedTaskStatus.WORKING,
            "amendments": amendments,
        },
    )
    if updated == 0:
        raise TaskUpdateConflictError("This escalation was already answered.")
    return await _reload(user_id, task_id), worker_session_id


async def _open_task(user_id: str, task_id: str) -> prisma.models.DelegatedTask:
    row = await prisma.models.DelegatedTask.prisma().find_first(
        where={"id": task_id, "userId": user_id}
    )
    if row is None:
        raise TaskDelegationRefusedError("No such task on this account.")
    if row.status not in _OPEN_STATUSES:
        raise TaskDelegationRefusedError(
            f"This task is already {row.status} and can no longer change."
        )
    return row


async def _reload(user_id: str, task_id: str) -> DelegatedTask:
    row = await prisma.models.DelegatedTask.prisma().find_first(
        where={"id": task_id, "userId": user_id},
        include=_TASK_INCLUDE,
    )
    if row is None:
        raise TaskDelegationRefusedError("No such task on this account.")
    return _to_model(row, await _library_agents_by_graph(user_id, [row]))


def _appended(row: prisma.models.DelegatedTask, entry: TaskAmendment) -> SafeJson:
    """The amendments column with *entry* appended, ready to write back.

    Malformed stored entries are preserved verbatim rather than validated
    away: the column is append-only history, and a write must not silently
    rewrite what it did not author.
    """
    existing = row.amendments if isinstance(row.amendments, list) else []
    return SafeJson([*existing, entry.model_dump(mode="json")])


def _latest_escalation_session(row: prisma.models.DelegatedTask) -> str | None:
    if not isinstance(row.amendments, list):
        return None
    for entry in reversed(row.amendments):
        if not isinstance(entry, dict):
            continue
        # A handoff after the escalation moved ownership: the escalating
        # session belongs to the previous owner, and delivering the answer
        # there would resume the wrong expert. Degrade to timeline-only.
        if entry.get("kind") == "handoff":
            return None
        if entry.get("kind") == "escalation":
            # A manager escalation can land after the user one; the answer
            # resumes only the session that asked the user. Rows written
            # before ``target`` existed default to "user".
            if (entry.get("target") or "user") != "user":
                continue
            session_id = entry.get("session_id")
            return session_id if isinstance(session_id, str) else None
    return None
