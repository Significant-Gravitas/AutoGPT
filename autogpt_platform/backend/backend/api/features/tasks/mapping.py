"""Row→model mapping shared by the task-spine query modules.

A leaf module on purpose: ``tasks_db`` imports the executor (to stop runs on
cancel) and the executor's import chain reaches back into this package via
the experts feature, so anything the sibling query modules share must live
where it imports neither.
"""

import logging
from typing import cast

import prisma.enums
import prisma.models
import prisma.types

from backend.copilot.briefing.outcome import DEFAULT_AGENT_NAME, run_link
from backend.data.model import is_credentials_field_name

from .models import (
    DelegatedTask,
    TaskAcceptance,
    TaskAmendment,
    TaskCreatedBy,
    TaskCredentialRef,
    TaskExpertRef,
    TaskRunRef,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# The owner join is always needed (the card shows who did the work) and the
# execution join is what makes a task a receipt rather than a label.
TASK_INCLUDE: prisma.types.DelegatedTaskInclude = {
    "Owner": True,
    "Executions": {"include": {"AgentGraph": True}},
}

RUNNING_EXECUTION_STATUSES = [
    prisma.enums.AgentExecutionStatus.QUEUED,
    prisma.enums.AgentExecutionStatus.RUNNING,
    prisma.enums.AgentExecutionStatus.INCOMPLETE,
]


async def library_agents_by_graph(
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


def to_model(
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
        amendments=to_amendments(row.amendments),
        stale_at=row.staleAt,
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


def credentials_from_nodes(
    nodes: list[prisma.models.AgentNode],
) -> list[TaskCredentialRef]:
    """Distinct credentials the run graphs' nodes are configured with.

    ``constantInput`` is free Json, so every shape check here is boundary
    validation — a hand-edited node must skip, not 500 the detail page.
    """
    creds: dict[str, TaskCredentialRef] = {}
    for node in nodes:
        constant_input = node.constantInput
        if not isinstance(constant_input, dict):
            continue
        for field_name, value in constant_input.items():
            if not is_credentials_field_name(field_name):
                continue
            if not isinstance(value, dict) or not value.get("id"):
                continue
            cred_id = str(value["id"])
            creds.setdefault(
                cred_id,
                TaskCredentialRef(
                    id=cred_id,
                    provider=str(value.get("provider") or ""),
                    title=value.get("title") or None,
                ),
            )
    return list(creds.values())


def to_amendments(value: object) -> list[TaskAmendment]:
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
