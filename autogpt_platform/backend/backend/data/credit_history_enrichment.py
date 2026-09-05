import asyncio
from typing import cast

from prisma.enums import SubmissionStatus
from prisma.models import (
    AgentGraph,
    AgentGraphExecution,
    ChatSession,
    LibraryAgent,
    StoreListingVersion,
)
from prisma.types import (
    AgentGraphExecutionWhereInput,
    AgentGraphWhereInput,
    ChatSessionWhereInput,
)
from pydantic import BaseModel

from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.data.model import CreditHistoryRelatedExecution, CreditTransactionItem
from backend.data.tenancy import get_user_team_ids, visibility_filter


class _AgentRef(BaseModel):
    name: str | None = None
    library_agent_id: str | None = None
    execution_available: bool = False


async def enrich_credit_history(
    items: list[CreditTransactionItem],
    user_id: str,
    organization_id: str | None = None,
) -> list[CreditTransactionItem]:
    if not any(item.usage_execution_id or item.usage_graph_id for item in items):
        return items
    team_ids = (
        await get_user_team_ids(user_id, organization_id) if organization_id else []
    )
    scope = visibility_filter(user_id, organization_id, team_ids)
    execution_data, sessions = await asyncio.gather(
        _load_executions(items, user_id, cast(AgentGraphExecutionWhereInput, scope)),
        _load_sessions(items, user_id, organization_id),
    )
    executions, child_counts = execution_data
    refs, fallback = await _load_agent_refs(
        items, executions, user_id, cast(AgentGraphWhereInput, scope)
    )
    execution_map = {execution.id: execution for execution in executions}
    return [
        _enrich_item(item, execution_map, refs, fallback, sessions, child_counts)
        for item in items
    ]


async def _load_executions(
    items: list[CreditTransactionItem],
    user_id: str,
    scope: AgentGraphExecutionWhereInput,
) -> tuple[list[AgentGraphExecution], dict[str, int]]:
    ids = sorted(
        {
            item.usage_execution_id
            for item in items
            if item.usage_execution_id
            and not item.usage_execution_id.startswith(COPILOT_SESSION_PREFIX)
        }
    )
    if not ids:
        return [], {}
    # Match execution detail's expert guard as well as its org/team predicate.
    access: AgentGraphExecutionWhereInput = {
        "isDeleted": False,
        "AND": [scope, {"OR": [{"expertId": None}, {"userId": user_id}]}],
    }
    executions = await AgentGraphExecution.prisma().find_many(
        where={**access, "id": {"in": ids}},
    )
    if not executions:
        return [], {}
    return await _load_related_executions(executions, access)


async def _load_related_executions(
    executions: list[AgentGraphExecution], access: AgentGraphExecutionWhereInput
) -> tuple[list[AgentGraphExecution], dict[str, int]]:
    parents = sorted(
        {e.parentGraphExecutionId for e in executions if e.parentGraphExecutionId}
    )
    parent_rows = (
        await AgentGraphExecution.prisma().find_many(
            where={**access, "id": {"in": parents}}
        )
        if parents
        else []
    )
    child_where: AgentGraphExecutionWhereInput = {
        **access,
        "parentGraphExecutionId": {"in": [e.id for e in executions]},
    }
    children, counts = await asyncio.gather(
        AgentGraphExecution.prisma().find_many(
            where=child_where,
            take=100,
            order=[{"createdAt": "asc"}, {"id": "asc"}],
        ),
        AgentGraphExecution.prisma().group_by(
            by=["parentGraphExecutionId"],
            where=child_where,
            count=True,
        ),
    )
    records = {e.id: e for e in [*executions, *parent_rows, *children]}
    child_counts = {
        parent_id: int((row.get("_count") or {}).get("_all") or 0)
        for row in counts
        if (parent_id := row.get("parentGraphExecutionId"))
    }
    return list(records.values()), child_counts


async def _load_sessions(
    items: list[CreditTransactionItem], user_id: str, organization_id: str | None
) -> dict[str, ChatSession]:
    ids = sorted(
        {
            item.usage_execution_id.removeprefix(COPILOT_SESSION_PREFIX)
            for item in items
            if item.usage_execution_id
            and item.usage_execution_id.startswith(COPILOT_SESSION_PREFIX)
        }
    )
    if not ids:
        return {}
    where: ChatSessionWhereInput = {"id": {"in": ids}, "userId": user_id}
    if organization_id:
        # Match the chat detail endpoint: expert sessions remain owner-accessible.
        where["AND"] = [
            {
                "OR": [
                    {"organizationId": organization_id},
                    {"organizationId": None},
                    {"expertId": {"not": None}},
                ]
            }
        ]
    sessions = await ChatSession.prisma().find_many(where=where)
    return {session.id: session for session in sessions}


async def _load_agent_refs(
    items: list[CreditTransactionItem],
    executions: list[AgentGraphExecution],
    user_id: str,
    scope: AgentGraphWhereInput,
) -> tuple[dict[tuple[str, int], _AgentRef], dict[str, _AgentRef]]:
    graph_ids = sorted(
        {e.agentGraphId for e in executions}
        | {
            item.usage_graph_id
            for item in items
            if item.usage_graph_id
            and not (item.usage_execution_id or "").startswith(COPILOT_SESSION_PREFIX)
        }
    )
    if not graph_ids:
        return {}, {}
    libraries = await LibraryAgent.prisma().find_many(
        where={
            "userId": user_id,
            "agentGraphId": {"in": graph_ids},
            "isDeleted": False,
        },
        include={"AgentGraph": True},
        order=[{"agentGraphVersion": "asc"}, {"id": "asc"}],
    )
    libraries = [library for library in libraries if library.AgentGraph]
    graphs = await _load_accessible_graphs(executions, libraries, scope)
    current = {library.agentGraphId: library for library in libraries}
    exact = {
        (library.agentGraphId, library.agentGraphVersion): library
        for library in libraries
    }
    fallback = {
        graph_id: _library_ref(library) for graph_id, library in current.items()
    }
    return {
        (execution.agentGraphId, execution.agentGraphVersion): _execution_ref(
            execution, graphs, exact, current
        )
        for execution in executions
    }, fallback


async def _load_accessible_graphs(
    executions: list[AgentGraphExecution],
    libraries: list[LibraryAgent],
    scope: AgentGraphWhereInput,
) -> dict[tuple[str, int], AgentGraph]:
    if not executions:
        return {}
    versions = sorted({(e.agentGraphId, e.agentGraphVersion) for e in executions})
    graphs, published = await asyncio.gather(
        AgentGraph.prisma().find_many(
            where={
                "AND": [
                    scope,
                    {
                        "OR": [
                            {"id": graph_id, "version": version}
                            for graph_id, version in versions
                        ]
                    },
                ]
            }
        ),
        StoreListingVersion.prisma().find_many(
            where={
                "submissionStatus": SubmissionStatus.APPROVED,
                "isDeleted": False,
                "OR": [
                    {"agentGraphId": graph_id, "agentGraphVersion": version}
                    for graph_id, version in versions
                ],
            },
            include={"AgentGraph": True},
        ),
    )
    accessible = [
        *graphs,
        *[listing.AgentGraph for listing in published if listing.AgentGraph],
        *[
            library.AgentGraph
            for library in libraries
            if not library.isArchived and library.AgentGraph
        ],
    ]
    return {(graph.id, graph.version): graph for graph in accessible}


def _library_ref(library: LibraryAgent) -> _AgentRef:
    name = library.name
    if name is None and library.AgentGraph:
        name = library.AgentGraph.name
    return _AgentRef(name=name or None, library_agent_id=library.id)


def _execution_ref(
    execution: AgentGraphExecution,
    graphs: dict[tuple[str, int], AgentGraph],
    exact: dict[tuple[str, int], LibraryAgent],
    current: dict[str, LibraryAgent],
) -> _AgentRef:
    key = (execution.agentGraphId, execution.agentGraphVersion)
    library = exact.get(key)
    graph = graphs.get(key)
    destination = library or current.get(execution.agentGraphId)
    name = (
        _library_ref(destination).name
        if destination
        else (graph.name if graph else None)
    )
    return _AgentRef(
        name=name or None,
        library_agent_id=destination.id if destination else None,
        execution_available=graph is not None,
    )


def _enrich_item(
    original: CreditTransactionItem,
    executions: dict[str, AgentGraphExecution],
    refs: dict[tuple[str, int], _AgentRef],
    fallback: dict[str, _AgentRef],
    sessions: dict[str, ChatSession],
    child_counts: dict[str, int],
) -> CreditTransactionItem:
    item = original.model_copy(deep=True)
    execution_id = item.usage_execution_id or ""
    if execution_id.startswith(COPILOT_SESSION_PREFIX):
        session = sessions.get(execution_id.removeprefix(COPILOT_SESSION_PREFIX))
        item.conversation_id = session.id if session else None
        item.conversation_title = session.title if session else None
        return item
    execution = executions.get(execution_id)
    ref = (
        refs.get((execution.agentGraphId, execution.agentGraphVersion))
        if execution
        else fallback.get(item.usage_graph_id or "")
    )
    if ref:
        item.agent_name = ref.name
        item.library_agent_id = ref.library_agent_id
    if not execution or not ref:
        return item
    item.execution_available = ref.execution_available
    item.execution_graph_version = execution.agentGraphVersion
    item.execution_status = execution.executionStatus.value
    item.execution_started_at = execution.startedAt
    _add_related(item, execution, executions, refs)
    item.related_executions_has_more = len(item.related_executions) < child_counts.get(
        execution.id, 0
    )
    return item


def _related_ref(
    execution: AgentGraphExecution, refs: dict[tuple[str, int], _AgentRef]
) -> CreditHistoryRelatedExecution:
    ref = refs.get((execution.agentGraphId, execution.agentGraphVersion), _AgentRef())
    return CreditHistoryRelatedExecution(
        execution_id=execution.id,
        agent_name=ref.name,
        library_agent_id=ref.library_agent_id,
        execution_available=ref.execution_available,
    )


def _add_related(
    item: CreditTransactionItem,
    execution: AgentGraphExecution,
    executions: dict[str, AgentGraphExecution],
    refs: dict[tuple[str, int], _AgentRef],
) -> None:
    parent = executions.get(execution.parentGraphExecutionId or "")
    if parent:
        related = _related_ref(parent, refs)
        item.parent_execution_id = related.execution_id
        item.parent_agent_name = related.agent_name
        item.parent_library_agent_id = (
            related.library_agent_id if related.execution_available else None
        )
    item.related_executions = [
        _related_ref(child, refs)
        for child in executions.values()
        if child.parentGraphExecutionId == execution.id
    ]
