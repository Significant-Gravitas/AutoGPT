"""Prisma access for the task spine.

Every read and write filters on ``userId``. The owning expert is not a
sufficient guard on its own: experts are soft-deleted, so an archived
expert's id stays readable and would otherwise leak its receipts.
"""

import logging
from datetime import datetime, timezone

import prisma.enums
import prisma.models
import prisma.types

from backend.data.db import transaction
from backend.executor import utils as execution_utils
from backend.util.exceptions import TaskDelegationRefusedError

from .errors import DelegatedTaskNotFoundError
from .mapping import RUNNING_EXECUTION_STATUSES as _RUNNING_EXECUTION_STATUSES
from .mapping import TASK_INCLUDE as _TASK_INCLUDE
from .mapping import credentials_from_nodes
from .mapping import library_agents_by_graph as _library_agents_by_graph
from .mapping import to_model as _to_model
from .models import (
    MAX_TASK_DEPTH,
    MAX_TASK_EVENTS,
    MAX_TASKS_PER_PAGE,
    OPEN_TASK_STATUSES,
    TASK_OUTCOME_MAX_LENGTH,
    TASK_SPEC_MAX_LENGTH,
    TASK_TITLE_MAX_LENGTH,
    DelegatedTask,
    DelegatedTaskDetail,
    TaskCreatedBy,
    TaskCredentialRef,
    TaskEvent,
    TaskStatus,
)

logger = logging.getLogger(__name__)


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


async def list_task_events(
    user_id: str, *, since: datetime | None = None
) -> list[TaskEvent]:
    """The user's task state changes since *since*, oldest first — one event
    per changed task, carrying its current status. Feeds the office view's
    lightweight poll; the client passes the last event's ``ts`` back as the
    next ``since``."""
    where: prisma.types.DelegatedTaskWhereInput = {"userId": user_id}
    if since is not None:
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        where["updatedAt"] = {"gt": since}
    rows = await prisma.models.DelegatedTask.prisma().find_many(
        where=where,
        order={"updatedAt": "asc"},
        take=MAX_TASK_EVENTS,
    )
    return [
        TaskEvent(
            task_id=row.id,
            expert_id=row.ownerId,
            event=row.status.lower(),
            ts=row.updatedAt.isoformat(),
        )
        for row in rows
    ]


async def get_task(user_id: str, task_id: str) -> DelegatedTaskDetail | None:
    """One task plus every task beneath it (flat, parented — the drawer nests
    them client-side). Returns None when the task does not exist or belongs to
    someone else — callers turn that into a 404 so the two cases are
    indistinguishable from outside."""
    row = await prisma.models.DelegatedTask.prisma().find_first(
        where={"id": task_id, "userId": user_id},
        include=_TASK_INCLUDE,
    )
    if row is None:
        return None

    children = await _descendants(user_id, task_id)
    library_agents = await _library_agents_by_graph(user_id, [row, *children])
    task = _to_model(row, library_agents).model_copy(
        update={"credentials": await _credentials_used([row, *children])}
    )
    return DelegatedTaskDetail(
        task=task,
        children=[_to_model(child, library_agents) for child in children],
    )


async def _credentials_used(
    rows: list[prisma.models.DelegatedTask],
) -> list[TaskCredentialRef]:
    """Credentials wired into the graphs this task (and its subtasks) ran.

    Detail-only on purpose: it scans every node of every run's graph, which
    the list endpoint must never pay per row.
    """
    graph_keys = {
        (execution.agentGraphId, execution.agentGraphVersion)
        for row in rows
        for execution in row.Executions or []
    }
    if not graph_keys:
        return []
    nodes = await prisma.models.AgentNode.prisma().find_many(
        where={
            "OR": [
                {"agentGraphId": graph_id, "agentGraphVersion": version}
                for graph_id, version in graph_keys
            ]
        }
    )
    return credentials_from_nodes(nodes)


async def get_delegated_task(user_id: str, task_id: str) -> DelegatedTaskDetail | None:
    """RPC-facing alias for :func:`get_task` — the DatabaseManager route and
    the wrapped function must share a name, and ``get_task`` is too generic
    to claim there."""
    return await get_task(user_id, task_id)


async def _descendants(user_id: str, task_id: str) -> list[prisma.models.DelegatedTask]:
    """Every task under *task_id*, oldest first, capped at a page.

    Level-by-level like ``_open_subtree_ids`` — depth is bounded at
    ``MAX_TASK_DEPTH`` on write, but a read that trusts stored parent ids
    must still not be able to spin on a corrupted cycle.
    """
    seen = {task_id}
    collected: list[prisma.models.DelegatedTask] = []
    frontier = [task_id]
    while frontier and len(collected) < MAX_TASKS_PER_PAGE:
        children = await prisma.models.DelegatedTask.prisma().find_many(
            where={"parentTaskId": {"in": frontier}, "userId": user_id},
            include=_TASK_INCLUDE,
            order={"createdAt": "asc"},
            take=MAX_TASKS_PER_PAGE,
        )
        fresh = [child for child in children if child.id not in seen]
        frontier = [child.id for child in fresh]
        seen.update(frontier)
        collected.extend(fresh)
    return collected[:MAX_TASKS_PER_PAGE]


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
    parent_task_id: str | None = None,
) -> DelegatedTask:
    """Open a receipt for a delegation, as a root or under *parent_task_id*.

    A root stamps ``rootTaskId`` with its own id — a tree read is then one
    indexed lookup. A subtask inherits the parent's root and ancestor trail,
    after the delegation policy (no loops, bounded depth) has approved the
    hop — raising :class:`TaskDelegationRefusedError` with a message the
    calling agent can act on. Takes the literal ``created_by_type`` rather
    than the Prisma enum so copilot (which calls this over RPC, Prisma-less)
    never has to import ``prisma``.
    """
    ancestors = [owner_id] if owner_id else []
    root_task_id: str | None = None
    if parent_task_id is not None:
        parent = await _approve_subtask(user_id, parent_task_id, owner_id)
        root_task_id = parent.rootTaskId or parent.id
        trail = _ancestor_trail(parent)
        ancestors = trail + [e for e in ancestors if e not in trail]

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
            "ancestorExpertIds": ancestors,
            "parentTaskId": parent_task_id,
            "rootTaskId": root_task_id,
        }
    )
    if root_task_id is None:
        stamped = await prisma.models.DelegatedTask.prisma().update(
            where={"id": row.id}, data={"rootTaskId": row.id}
        )
        return _to_model(stamped or row, {})
    return _to_model(row, {})


async def _approve_subtask(
    user_id: str, parent_task_id: str, owner_id: str | None
) -> prisma.models.DelegatedTask:
    """The parent row, if delegation policy allows hanging a subtask owned by
    *owner_id* beneath it."""
    parent = await prisma.models.DelegatedTask.prisma().find_first(
        where={"id": parent_task_id, "userId": user_id}
    )
    if parent is None:
        raise TaskDelegationRefusedError(
            "The task this delegation belongs to no longer exists. "
            "Escalate to the user with escalate_task instead."
        )
    if owner_id and owner_id in _ancestor_trail(parent):
        raise TaskDelegationRefusedError(
            "Delegation refused: that expert already holds this task further "
            "up the chain, so delegating to them would loop. Escalate to the "
            "user with escalate_task instead."
        )
    if await _task_depth(user_id, parent) >= MAX_TASK_DEPTH:
        raise TaskDelegationRefusedError(
            f"Delegation refused: this task tree is already "
            f"{MAX_TASK_DEPTH} levels deep. Finish the work yourself or "
            "escalate to the user with escalate_task instead."
        )
    return parent


def _ancestor_trail(task: prisma.models.DelegatedTask) -> list[str]:
    """Every expert the chain has passed through, current owner included."""
    trail = list(task.ancestorExpertIds)
    if task.ownerId and task.ownerId not in trail:
        trail.append(task.ownerId)
    return trail


async def _task_depth(user_id: str, task: prisma.models.DelegatedTask) -> int:
    """*task*'s depth in its tree (root = 1), walking the parent chain.

    ``seen`` guards the walk: parent ids are trusted stored data, and a
    corrupted cycle must terminate rather than spin.
    """
    depth = 1
    seen = {task.id}
    parent_id = task.parentTaskId
    while parent_id and parent_id not in seen and depth <= MAX_TASK_DEPTH:
        depth += 1
        seen.add(parent_id)
        parent = await prisma.models.DelegatedTask.prisma().find_first(
            where={"id": parent_id, "userId": user_id}
        )
        parent_id = parent.parentTaskId if parent else None
    return depth


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


async def claim_task_for_session(user_id: str, task_id: str, session_id: str) -> bool:
    """Bind a freshly opened worker session to a QUEUED task and start it.

    The one write that turns a receipt nobody is working into live work:
    without an ``originSessionId`` a task has no thread to be nudged in, so
    the overseer's stall retry cannot reach it and it sits QUEUED forever.
    Only claims a QUEUED task, so two kickoffs racing the same task (a hire
    retry and the overseer sweep) produce exactly one worker.
    """
    updated = await prisma.models.DelegatedTask.prisma().update_many(
        where={
            "id": task_id,
            "userId": user_id,
            "status": prisma.enums.DelegatedTaskStatus.QUEUED,
        },
        data={
            "status": prisma.enums.DelegatedTaskStatus.WORKING,
            "originSessionId": session_id,
        },
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
