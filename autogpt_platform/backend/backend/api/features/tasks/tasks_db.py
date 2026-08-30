"""Prisma access for the task spine.

Every read and write filters on ``userId``. The owning expert is not a
sufficient guard on its own: experts are soft-deleted, so an archived
expert's id stays readable and would otherwise leak its receipts.
"""

import logging
from typing import cast

import prisma.enums
import prisma.models
import prisma.types

from backend.copilot.briefing.outcome import DEFAULT_AGENT_NAME, run_link
from backend.data.db import transaction
from backend.executor import utils as execution_utils

from .errors import DelegatedTaskNotFoundError
from .models import (
    MAX_TASKS_PER_PAGE,
    OPEN_TASK_STATUSES,
    TASK_OUTCOME_MAX_LENGTH,
    TASK_SPEC_MAX_LENGTH,
    TASK_TITLE_MAX_LENGTH,
    DelegatedTask,
    DelegatedTaskDetail,
    TaskAcceptance,
    TaskAmendment,
    TaskCreatedBy,
    TaskExpertRef,
    TaskRunRef,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# The owner join is always needed (the card shows who did the work) and the
# execution join is what makes a task a receipt rather than a label.
_TASK_INCLUDE: prisma.types.DelegatedTaskInclude = {
    "Owner": True,
    "Executions": {"include": {"AgentGraph": True}},
}

_RUNNING_EXECUTION_STATUSES = [
    prisma.enums.AgentExecutionStatus.QUEUED,
    prisma.enums.AgentExecutionStatus.RUNNING,
    prisma.enums.AgentExecutionStatus.INCOMPLETE,
]


async def list_tasks(
    user_id: str,
    *,
    expert_id: str | None = None,
    status: TaskStatus | None = None,
    limit: int = MAX_TASKS_PER_PAGE,
) -> list[DelegatedTask]:
    """The user's tasks, newest first, optionally narrowed to one expert or
    one status."""
    where: prisma.types.DelegatedTaskWhereInput = {"userId": user_id}
    if expert_id is not None:
        where["ownerId"] = expert_id
    if status is not None:
        where["status"] = prisma.enums.DelegatedTaskStatus(status)

    rows = await prisma.models.DelegatedTask.prisma().find_many(
        where=where,
        include=_TASK_INCLUDE,
        order={"createdAt": "desc"},
        take=min(limit, MAX_TASKS_PER_PAGE),
    )
    library_agents = await _library_agents_by_graph(user_id, rows)
    return [_to_model(row, library_agents) for row in rows]


async def list_open_tasks(
    user_id: str, limit: int = MAX_TASKS_PER_PAGE
) -> list[DelegatedTask]:
    """Tasks that can still change, newest first — what Home's active list
    and the Tasks tab's "active" split both read. Uses the
    ``[userId, status]`` index."""
    rows = await prisma.models.DelegatedTask.prisma().find_many(
        where={
            "userId": user_id,
            "status": {
                "in": [prisma.enums.DelegatedTaskStatus(s) for s in OPEN_TASK_STATUSES]
            },
        },
        include=_TASK_INCLUDE,
        order={"createdAt": "desc"},
        take=min(limit, MAX_TASKS_PER_PAGE),
    )
    library_agents = await _library_agents_by_graph(user_id, rows)
    return [_to_model(row, library_agents) for row in rows]


async def get_task(user_id: str, task_id: str) -> DelegatedTaskDetail | None:
    """One task plus its direct children. Returns None when the task does not
    exist or belongs to someone else — callers turn that into a 404 so the two
    cases are indistinguishable from outside."""
    row = await prisma.models.DelegatedTask.prisma().find_first(
        where={"id": task_id, "userId": user_id},
        include=_TASK_INCLUDE,
    )
    if row is None:
        return None

    children = await prisma.models.DelegatedTask.prisma().find_many(
        where={"parentTaskId": task_id, "userId": user_id},
        include=_TASK_INCLUDE,
        order={"createdAt": "asc"},
        take=MAX_TASKS_PER_PAGE,
    )
    library_agents = await _library_agents_by_graph(user_id, [row, *children])
    return DelegatedTaskDetail(
        task=_to_model(row, library_agents),
        children=[_to_model(child, library_agents) for child in children],
    )


# The three RPC-exposed writers below MUST keep the same name as their
# DatabaseManager attribute: the AppService route is the attribute name but
# the generated client calls the *function* name, so a mismatch 404s.
async def create_delegated_task(
    user_id: str,
    *,
    title: str,
    spec: str,
    owner_id: str | None = None,
    origin_session_id: str | None = None,
    created_by_type: TaskCreatedBy = "USER",
    created_by_id: str | None = None,
) -> DelegatedTask:
    """Open a receipt for a delegation.

    Phase 1 only ever creates roots, so ``rootTaskId`` is stamped with the
    row's own id — a tree read is then one indexed lookup even before
    handoff exists. Takes the literal ``created_by_type`` rather than the
    Prisma enum so copilot (which calls this over RPC, Prisma-less) never
    has to import ``prisma``.
    """
    row = await prisma.models.DelegatedTask.prisma().create(
        data={
            "userId": user_id,
            "ownerId": owner_id,
            "originSessionId": origin_session_id,
            "createdByType": prisma.enums.TaskCreatedByType(created_by_type),
            "createdById": created_by_id,
            "title": title[:TASK_TITLE_MAX_LENGTH],
            "spec": spec[:TASK_SPEC_MAX_LENGTH],
            "status": prisma.enums.DelegatedTaskStatus.QUEUED,
            "ancestorExpertIds": [owner_id] if owner_id else [],
        }
    )
    stamped = await prisma.models.DelegatedTask.prisma().update(
        where={"id": row.id}, data={"rootTaskId": row.id}
    )
    return _to_model(stamped or row, {})


async def mark_delegated_task_working(user_id: str, task_id: str) -> bool:
    """QUEUED → WORKING once the run is actually accepted by the executor."""
    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={
            "id": task_id,
            "userId": user_id,
            "status": prisma.enums.DelegatedTaskStatus.QUEUED,
        },
        data={"status": prisma.enums.DelegatedTaskStatus.WORKING},
    )
    return updated > 0


async def close_delegated_task(
    user_id: str,
    task_id: str,
    *,
    succeeded: bool,
    outcome_summary: str | None,
    spend: int = 0,
) -> str | None:
    """Close the receipt and hand back the session its outcome belongs in.

    Only open tasks are closed: a CANCELLED task whose run finished anyway
    must keep its cancellation, and a re-fired completion must not overwrite
    an outcome that already landed. Returns the origin session id when the
    task was closed by this call, else None — so the caller posts exactly
    one outcome message per task.
    """
    status = (
        prisma.enums.DelegatedTaskStatus.DONE
        if succeeded
        else prisma.enums.DelegatedTaskStatus.FAILED
    )
    data: prisma.types.DelegatedTaskUpdateManyMutationInput = {"status": status}
    if outcome_summary is not None:
        data["outcomeSummary"] = outcome_summary[:TASK_OUTCOME_MAX_LENGTH]
    if spend > 0:
        data["spendTotal"] = {"increment": spend}

    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={
            "id": task_id,
            "userId": user_id,
            "status": {
                "in": [prisma.enums.DelegatedTaskStatus(s) for s in OPEN_TASK_STATUSES]
            },
        },
        data=data,
    )
    if updated == 0:
        return None
    row = await prisma.models.DelegatedTask.prisma().find_first(
        where={"id": task_id, "userId": user_id}
    )
    return row.originSessionId if row else None


async def cancel_task(user_id: str, task_id: str) -> DelegatedTaskDetail:
    """Cancel a task and every open task beneath it, then stop the executions
    those tasks were driving.

    The status flip runs in one transaction so a partially-cancelled tree is
    never observable. Stopping executions happens afterwards: it talks to the
    executor over RPC and must not hold a DB transaction open, and a stop that
    fails leaves a CANCELLED task with a run that finishes on its own — which
    the outcome hook then ignores because the task is no longer open.
    """
    open_ids = await _open_subtree_ids(user_id, task_id)
    if open_ids is None:
        raise DelegatedTaskNotFoundError(task_id)

    if open_ids:
        async with transaction() as tx:
            await tx.delegatedtask.update_many(
                where={"id": {"in": open_ids}, "userId": user_id},
                data={"status": prisma.enums.DelegatedTaskStatus.CANCELLED},
            )
        await _stop_running_executions(user_id, open_ids)

    detail = await get_task(user_id, task_id)
    if detail is None:
        raise DelegatedTaskNotFoundError(task_id)
    return detail


async def _open_subtree_ids(user_id: str, task_id: str) -> list[str] | None:
    """Ids of *task_id* and its open descendants, or None if the task isn't
    the caller's.

    Walks level by level rather than following ``rootTaskId``: a cancel must
    only reach the requested task's own branch, not its siblings under a
    shared root. ``seen`` guards the walk — a parent chain corrupted into a
    cycle would otherwise spin forever.
    """
    root = await prisma.models.DelegatedTask.prisma().find_first(
        where={"id": task_id, "userId": user_id}
    )
    if root is None:
        return None

    open_statuses = [prisma.enums.DelegatedTaskStatus(s) for s in OPEN_TASK_STATUSES]
    seen = {task_id}
    open_ids = [task_id] if root.status in open_statuses else []
    frontier = [task_id]
    while frontier:
        children = await prisma.models.DelegatedTask.prisma().find_many(
            where={"parentTaskId": {"in": frontier}, "userId": user_id}
        )
        frontier = [child.id for child in children if child.id not in seen]
        seen.update(frontier)
        open_ids.extend(
            child.id
            for child in children
            if child.id in seen and child.status in open_statuses
        )
    return open_ids


async def _stop_running_executions(user_id: str, task_ids: list[str]) -> None:
    executions = await prisma.models.AgentGraphExecution.prisma().find_many(
        where={
            "userId": user_id,
            "delegatedTaskId": {"in": task_ids},
            "executionStatus": {"in": _RUNNING_EXECUTION_STATUSES},
        }
    )
    for execution in executions:
        try:
            await execution_utils.stop_graph_execution(
                graph_exec_id=execution.id, user_id=user_id
            )
        except Exception:
            # A run that refuses to stop must not fail the cancel: the task is
            # already CANCELLED, so its outcome will be discarded either way.
            logger.warning(
                "Failed to stop execution #%s for cancelled task",
                execution.id,
                exc_info=True,
            )


async def _library_agents_by_graph(
    user_id: str, tasks: list[prisma.models.DelegatedTask]
) -> dict[str, str]:
    """Graph id → the caller's library agent id, for the run deep links.

    One batched query for every graph across every listed task; without it a
    Tasks tab with N runs would issue N lookups just to build hrefs.
    """
    graph_ids = {
        execution.agentGraphId for task in tasks for execution in task.Executions or []
    }
    if not graph_ids:
        return {}
    rows = await prisma.models.LibraryAgent.prisma().find_many(
        where={
            "userId": user_id,
            "agentGraphId": {"in": list(graph_ids)},
            "isDeleted": False,
        }
    )
    return {row.agentGraphId: row.id for row in rows}


def _to_model(
    row: prisma.models.DelegatedTask, library_agents: dict[str, str]
) -> DelegatedTask:
    return DelegatedTask(
        id=row.id,
        title=row.title,
        spec=row.spec,
        # Prisma-client-py declares enum columns as its own generated enums
        # but hands back plain strings at runtime, which the ``Literal``
        # aliases in models.py already match value-for-value.
        status=cast(TaskStatus, row.status),
        acceptance=cast(TaskAcceptance, row.acceptance),
        created_by_type=cast(TaskCreatedBy, row.createdByType),
        created_by_id=row.createdById,
        owner=_to_expert_ref(row.Owner),
        parent_task_id=row.parentTaskId,
        root_task_id=row.rootTaskId,
        origin_session_id=row.originSessionId,
        ancestor_expert_ids=row.ancestorExpertIds,
        handoff_count=row.handoffCount,
        revision_count=row.revisionCount,
        spend_total=row.spendTotal,
        outcome_summary=row.outcomeSummary,
        amendments=_to_amendments(row.amendments),
        created_at=row.createdAt,
        updated_at=row.updatedAt,
        runs=[
            _to_run_ref(execution, library_agents.get(execution.agentGraphId))
            for execution in row.Executions or []
        ],
    )


def _to_expert_ref(row: prisma.models.Expert | None) -> TaskExpertRef | None:
    if row is None:
        return None
    return TaskExpertRef(
        id=row.id, name=row.name, avatar_url=row.avatarUrl, role=row.role
    )


def _to_run_ref(
    row: prisma.models.AgentGraphExecution, library_agent_id: str | None
) -> TaskRunRef:
    graph = row.AgentGraph
    return TaskRunRef(
        execution_id=row.id,
        graph_id=row.agentGraphId,
        library_agent_id=library_agent_id,
        agent_name=(graph.name if graph and graph.name else DEFAULT_AGENT_NAME),
        status=row.executionStatus,
        started_at=row.startedAt,
        ended_at=row.endedAt,
        link=run_link(library_agent_id, row.id),
    )


def _to_amendments(value: object) -> list[TaskAmendment]:
    """Amendments are stored as free Json, so a hand-edited or legacy blob
    must degrade to an empty list rather than 500 the whole Tasks tab."""
    if not isinstance(value, list):
        return []
    amendments = []
    for entry in value:
        try:
            amendments.append(TaskAmendment.model_validate(entry))
        except Exception:
            logger.warning("Skipping malformed task amendment", exc_info=True)
    return amendments
