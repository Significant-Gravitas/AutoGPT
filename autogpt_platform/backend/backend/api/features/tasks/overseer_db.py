"""Prisma reads and writes the overseer pass and its surfaces lean on.

Same tenancy rule as ``tasks_db``: every statement filters on ``userId``.
The pass runs in the scheduler process (Prisma-less), so each function here
is exposed over the DatabaseManager RPC — names must stay unique across
that surface and match their registered attribute.
"""

import logging
from datetime import UTC, datetime

import prisma.enums
import prisma.models

from backend.util.json import SafeJson

from .mapping import RUNNING_EXECUTION_STATUSES, to_model
from .models import MAX_TASKS_PER_PAGE, OPEN_TASK_STATUSES, DelegatedTask, TaskAmendment

logger = logging.getLogger(__name__)

_OPEN_STATUSES = [prisma.enums.DelegatedTaskStatus(s) for s in OPEN_TASK_STATUSES]


async def has_running_executions(user_id: str, task_ids: list[str]) -> dict[str, bool]:
    """Which of *task_ids* still have a live execution attached — the signal
    that a WORKING task is progressing rather than stalled."""
    if not task_ids:
        return {}
    rows = await prisma.models.AgentGraphExecution.prisma().find_many(
        where={
            "userId": user_id,
            "delegatedTaskId": {"in": task_ids},
            "executionStatus": {"in": RUNNING_EXECUTION_STATUSES},
        }
    )
    running = {row.delegatedTaskId for row in rows if row.delegatedTaskId}
    return {task_id: task_id in running for task_id in task_ids}


async def mark_task_stale(user_id: str, task_id: str, stale_at: datetime) -> bool:
    """Stamp ``staleAt`` on a still-waiting task. Never changes status —
    stale is a nag, not a cancellation."""
    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={
            "id": task_id,
            "userId": user_id,
            "status": prisma.enums.DelegatedTaskStatus.WAITING_USER,
            "staleAt": None,
        },
        data={"staleAt": stale_at},
    )
    return updated > 0


async def count_recent_failed_tasks_by_expert(
    user_id: str, since: datetime
) -> dict[str, int]:
    """FAILED tasks per owning expert since *since* — the expert-health
    signal that trips an automatic pause."""
    rows = await prisma.models.DelegatedTask.prisma().find_many(
        where={
            "userId": user_id,
            "status": prisma.enums.DelegatedTaskStatus.FAILED,
            "updatedAt": {"gte": since},
            "ownerId": {"not": None},
        },
        take=500,
    )
    counts: dict[str, int] = {}
    for row in rows:
        if row.ownerId:
            counts[row.ownerId] = counts.get(row.ownerId, 0) + 1
    return counts


async def list_recent_failed_tasks(
    user_id: str, since: datetime, limit: int = MAX_TASKS_PER_PAGE
) -> list[DelegatedTask]:
    """Recently FAILED tasks, newest first — Home's attention list shows the
    ones the overseer gave up on after a retry."""
    rows = await prisma.models.DelegatedTask.prisma().find_many(
        where={
            "userId": user_id,
            "status": prisma.enums.DelegatedTaskStatus.FAILED,
            "updatedAt": {"gte": since},
        },
        order={"updatedAt": "desc"},
        take=min(limit, MAX_TASKS_PER_PAGE),
    )
    return [to_model(row, {}) for row in rows]


async def list_recent_autopilot_tasks(
    user_id: str, since: datetime, limit: int = 200
) -> list[DelegatedTask]:
    """Tasks Autopilot finished itself (no owning expert) — the recruiter's
    raw material for spotting a category worth hiring for."""
    rows = await prisma.models.DelegatedTask.prisma().find_many(
        where={
            "userId": user_id,
            "ownerId": None,
            "status": prisma.enums.DelegatedTaskStatus.DONE,
            "createdAt": {"gte": since},
        },
        order={"createdAt": "desc"},
        take=limit,
    )
    return [to_model(row, {}) for row in rows]


async def count_open_tasks_for_expert(user_id: str, expert_id: str) -> int:
    """Open tasks an expert still holds — the fire dialog warns these will
    move to Autopilot."""
    return await prisma.models.DelegatedTask.prisma().count(
        where={
            "userId": user_id,
            "ownerId": expert_id,
            "status": {"in": _OPEN_STATUSES},
        }
    )


async def reassign_open_tasks_to_autopilot(user_id: str, expert_id: str) -> int:
    """Move an archived expert's open tasks to Autopilot (null owner),
    recording the swap on each task's timeline. Row-by-row because the
    amendment append is per-task Json; open task counts are small."""
    rows = await prisma.models.DelegatedTask.prisma().find_many(
        where={
            "userId": user_id,
            "ownerId": expert_id,
            "status": {"in": _OPEN_STATUSES},
        }
    )
    reassigned = 0
    for row in rows:
        entry = TaskAmendment(
            at=datetime.now(UTC),
            by="system",
            note="Owner was archived — reassigned to Autopilot.",
            kind="handoff",
            from_expert_id=expert_id,
            to_expert_id=None,
        )
        existing = row.amendments if isinstance(row.amendments, list) else []
        updated = await prisma.models.DelegatedTask.prisma().update_many(
            where={"id": row.id, "userId": user_id, "ownerId": expert_id},
            data={
                "ownerId": None,
                "amendments": SafeJson([*existing, entry.model_dump(mode="json")]),
            },
        )
        reassigned += updated
    if reassigned:
        logger.info(
            "Reassigned %d open task(s) from archived expert %s to Autopilot",
            reassigned,
            expert_id,
        )
    return reassigned
