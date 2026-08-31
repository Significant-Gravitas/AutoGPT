"""Phase-3 writers: outcome accept/reject with bounded revisions, plus the
generic amendment append the mid-task instruction hook and the overseer use.

Same tenancy rule as ``tasks_db``: every statement filters on ``userId``.
``append_task_amendment`` crosses the DatabaseManager RPC (the copilot
executor and scheduler are Prisma-less); the review writers are called from
the REST API only.
"""

import logging
from datetime import UTC, datetime

import prisma.enums
import prisma.models

from backend.util.exceptions import TaskDelegationRefusedError, TaskUpdateConflictError

from .errors import DelegatedTaskNotFoundError
from .models import (
    MAX_TASK_REVISIONS,
    TASK_ANSWER_MAX_LENGTH,
    TASK_SPEC_MAX_LENGTH,
    TASK_TITLE_MAX_LENGTH,
    DelegatedTask,
    TaskAmendment,
    TaskAmendmentKind,
)
from .task_actions import _appended, _reload

logger = logging.getLogger(__name__)


async def append_task_amendment(
    user_id: str,
    task_id: str,
    *,
    note: str,
    by: str,
    kind: TaskAmendmentKind = "note",
) -> DelegatedTask | None:
    """Append one timeline entry to a still-open task. Returns the task, or
    None when it is closed or missing — callers treat both as "nothing to
    record". Last-write-wins on a concurrent append: this is a best-effort
    log line, not a state transition, so it takes no optimistic lock."""
    row = await prisma.models.DelegatedTask.prisma().find_first(
        where={"id": task_id, "userId": user_id}
    )
    if row is None or row.status not in (
        prisma.enums.DelegatedTaskStatus.QUEUED,
        prisma.enums.DelegatedTaskStatus.WORKING,
        prisma.enums.DelegatedTaskStatus.WAITING_USER,
    ):
        return None
    amendments = _appended(
        row,
        TaskAmendment(
            at=datetime.now(UTC),
            by=by,
            note=note[:TASK_ANSWER_MAX_LENGTH],
            kind=kind,
        ),
    )
    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={"id": task_id, "userId": user_id},
        data={"amendments": amendments},
    )
    if updated == 0:
        return None
    return await _reload(user_id, task_id)


async def accept_delegated_task(user_id: str, task_id: str) -> DelegatedTask:
    """Stamp the user's approval on a finished task's outcome."""
    row = await _finished_task(user_id, task_id)
    if row.acceptance == prisma.enums.DelegatedTaskAcceptance.ACCEPTED:
        return await _reload(user_id, task_id)
    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={
            "id": task_id,
            "userId": user_id,
            "status": prisma.enums.DelegatedTaskStatus.DONE,
        },
        data={"acceptance": prisma.enums.DelegatedTaskAcceptance.ACCEPTED},
    )
    if updated == 0:
        raise TaskUpdateConflictError("The task changed while you were accepting it.")
    return await _reload(user_id, task_id)


async def reject_delegated_task(
    user_id: str,
    task_id: str,
    *,
    note: str,
) -> tuple[DelegatedTask, DelegatedTask | None]:
    """Reject a finished task's outcome with what to change.

    Under the cap this opens a revision subtask beneath the task (same
    owner) and bumps ``revisionCount``; the caller then nudges the owner's
    session to work it. At the cap no subtask is opened — the task is marked
    REJECTED and the caller tells the user to clarify in chat instead of
    looping a third time. Returns ``(task, revision_task | None)``.
    """
    row = await _finished_task(user_id, task_id)
    revision_note = note[:TASK_ANSWER_MAX_LENGTH]
    amendments = _appended(
        row,
        TaskAmendment(
            at=datetime.now(UTC),
            by="user",
            note=revision_note,
            kind="revision",
        ),
    )

    if row.revisionCount >= MAX_TASK_REVISIONS:
        updated = await prisma.models.DelegatedTask.prisma().update_many(
            where={
                "id": task_id,
                "userId": user_id,
                "status": prisma.enums.DelegatedTaskStatus.DONE,
                "revisionCount": row.revisionCount,
            },
            data={
                "acceptance": prisma.enums.DelegatedTaskAcceptance.REJECTED,
                "amendments": amendments,
            },
        )
        if updated == 0:
            raise TaskUpdateConflictError(
                "The task changed while you were rejecting it. Try again."
            )
        return await _reload(user_id, task_id), None

    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={
            "id": task_id,
            "userId": user_id,
            "status": prisma.enums.DelegatedTaskStatus.DONE,
            "revisionCount": row.revisionCount,
        },
        data={
            "acceptance": prisma.enums.DelegatedTaskAcceptance.REJECTED,
            "revisionCount": row.revisionCount + 1,
            "amendments": amendments,
        },
    )
    if updated == 0:
        raise TaskUpdateConflictError(
            "The task changed while you were rejecting it. Try again."
        )

    # Deliberately NOT create_delegated_task: the revision keeps the same
    # owner, which the delegation loop-check would refuse (the owner is on
    # their own ancestor trail by construction).
    revision_row = await prisma.models.DelegatedTask.prisma().create(
        data={
            "userId": user_id,
            "ownerId": row.ownerId,
            "originSessionId": row.originSessionId,
            "createdByType": prisma.enums.TaskCreatedByType.USER,
            "createdById": user_id,
            "title": f"Revision {row.revisionCount + 1}: {row.title}"[
                :TASK_TITLE_MAX_LENGTH
            ],
            "spec": _revision_spec(row, revision_note),
            "status": prisma.enums.DelegatedTaskStatus.QUEUED,
            "ancestorExpertIds": list(row.ancestorExpertIds),
            "parentTaskId": row.id,
            "rootTaskId": row.rootTaskId or row.id,
        }
    )
    return await _reload(user_id, task_id), await _reload(user_id, revision_row.id)


async def _finished_task(user_id: str, task_id: str) -> prisma.models.DelegatedTask:
    row = await prisma.models.DelegatedTask.prisma().find_first(
        where={"id": task_id, "userId": user_id}
    )
    if row is None:
        raise DelegatedTaskNotFoundError(task_id)
    if row.status != prisma.enums.DelegatedTaskStatus.DONE:
        raise TaskDelegationRefusedError(
            "Only a finished task's outcome can be accepted or rejected."
        )
    return row


def _revision_spec(row: prisma.models.DelegatedTask, note: str) -> str:
    outcome = " ".join((row.outcomeSummary or "").split())
    return (
        f"Revise the outcome of task '{row.title}'.\n"
        f"Previous outcome: {outcome or 'not recorded.'}\n"
        f"User feedback: {note}"
    )[:TASK_SPEC_MAX_LENGTH]
