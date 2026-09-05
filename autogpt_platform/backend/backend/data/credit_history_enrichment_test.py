from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import AgentExecutionStatus, ResourceVisibility
from prisma.models import (
    AgentGraph,
    AgentGraphExecution,
    ChatSession,
    LibraryAgent,
    StoreListingVersion,
)

from backend.data import credit_history_enrichment as history
from backend.data.model import CreditTransactionItem


def _graph(graph_id="graph", version=2, name="Original agent"):
    graph = MagicMock(spec=AgentGraph, id=graph_id, version=version)
    graph.name = name
    return graph


def _execution(execution_id="run", graph_id="graph", parent=None):
    return AgentGraphExecution(
        id=execution_id,
        createdAt=datetime(2026, 9, 5, 9, tzinfo=timezone.utc),
        userId="user",
        isDeleted=False,
        isShared=False,
        visibility=ResourceVisibility.PRIVATE,
        agentGraphId=graph_id,
        agentGraphVersion=2,
        parentGraphExecutionId=parent,
        executionStatus=AgentExecutionStatus.COMPLETED,
        startedAt=datetime(2026, 9, 5, 9, tzinfo=timezone.utc),
    )


def _library(graph=None, *, title=None, archived=False, library_id="library"):
    graph = graph or _graph()
    library = MagicMock(
        spec=LibraryAgent,
        id=library_id,
        userId="user",
        agentGraphId=graph.id,
        agentGraphVersion=graph.version,
        AgentGraph=graph,
        isArchived=archived,
        isDeleted=False,
    )
    library.name = title
    return library


def _item(execution_id="run", graph_id="graph"):
    return CreditTransactionItem(
        user_id="user",
        usage_graph_id=graph_id,
        usage_execution_id=execution_id,
        amount=-12,
    )


def _database(
    mocker, *, executions=(), related=(), libraries=(), graphs=(), sessions=()
):
    actions = {}
    for name, rows in {
        "AgentGraphExecution": executions,
        "LibraryAgent": libraries,
        "AgentGraph": graphs,
        "StoreListingVersion": (),
        "ChatSession": sessions,
    }.items():
        action = MagicMock()
        action.find_many = AsyncMock(return_value=list(rows))
        actions[name] = action
        mocker.patch(f"{history.__name__}.{name}.prisma", return_value=action)
    actions["AgentGraphExecution"].find_many.side_effect = [
        list(executions),
        list(related),
        list(related),
    ]
    parent_ids = {
        run.parentGraphExecutionId for run in related if run.parentGraphExecutionId
    }
    actions["AgentGraphExecution"].group_by = AsyncMock(
        return_value=[
            {
                "parentGraphExecutionId": parent_id,
                "_count": {
                    "_all": sum(
                        run.parentGraphExecutionId == parent_id for run in related
                    )
                },
            }
            for parent_id in parent_ids
        ]
    )
    mocker.patch.object(history, "get_user_team_ids", AsyncMock(return_value=["team"]))
    return actions


@pytest.mark.asyncio
async def test_exact_run_version_and_familiar_name_survive_library_update(mocker):
    old_graph = _graph()
    current = _library(_graph(version=3, name="New name"))
    _database(
        mocker, executions=[_execution()], libraries=[current], graphs=[old_graph]
    )
    original = _item()

    item = (await history.enrich_credit_history([original], "user"))[0]

    assert item.agent_name == "New name"
    assert item.library_agent_id == "library"
    assert item.execution_graph_version == 2
    assert item.execution_available is True
    assert item.execution_status == "COMPLETED"
    assert item.execution_started_at == _execution().startedAt
    assert item.amount == original.amount
    assert original.agent_name is None


@pytest.mark.asyncio
async def test_marketplace_snapshot_takes_precedence_over_graph_name(mocker):
    _database(
        mocker,
        executions=[_execution()],
        libraries=[_library(title="Installed title")],
    )
    item = (await history.enrich_credit_history([_item()], "user"))[0]
    assert item.agent_name == "Installed title"
    assert item.execution_available is True


@pytest.mark.asyncio
async def test_exact_library_snapshot_preferred_when_multiple_versions_exist(mocker):
    _database(
        mocker,
        executions=[_execution()],
        libraries=[
            _library(title="Installed title", library_id="old-library"),
            _library(_graph(version=3), title="Current title"),
        ],
    )
    item = (await history.enrich_credit_history([_item()], "user"))[0]
    assert item.agent_name == "Installed title"
    assert item.library_agent_id == "old-library"
    assert item.execution_graph_version == 2


@pytest.mark.asyncio
async def test_archived_library_is_linkable_but_does_not_grant_run_access(mocker):
    _database(mocker, executions=[_execution()], libraries=[_library(archived=True)])
    item = (await history.enrich_credit_history([_item()], "user"))[0]
    assert item.agent_name == "Original agent"
    assert item.library_agent_id == "library"
    assert item.execution_available is False


@pytest.mark.asyncio
async def test_deleted_execution_keeps_safe_current_library_name(mocker):
    _database(mocker, libraries=[_library(title="Known agent")])
    item = (await history.enrich_credit_history([_item()], "user"))[0]
    assert item.agent_name == "Known agent"
    assert item.library_agent_id == "library"
    assert item.execution_available is False
    assert item.execution_status is None
    assert item.execution_graph_version is None


@pytest.mark.asyncio
async def test_deleted_library_never_gets_replacement_graph_uuid_link(mocker):
    _database(mocker, executions=[_execution()], graphs=[_graph()])
    item = (await history.enrich_credit_history([_item()], "user"))[0]
    assert item.agent_name == "Original agent"
    assert item.library_agent_id is None
    assert item.execution_available is True


@pytest.mark.asyncio
async def test_public_nested_agent_can_be_named_without_library_entry(mocker):
    actions = _database(mocker, executions=[_execution()])
    actions["StoreListingVersion"].find_many.return_value = [
        MagicMock(spec=StoreListingVersion, AgentGraph=_graph())
    ]
    item = (await history.enrich_credit_history([_item()], "user"))[0]
    assert item.agent_name == "Original agent"
    assert item.execution_available is True
    assert item.library_agent_id is None
    where = actions["StoreListingVersion"].find_many.call_args.kwargs["where"]
    assert where["submissionStatus"] == "APPROVED"
    assert where["isDeleted"] is False
    assert where["OR"] == [{"agentGraphId": "graph", "agentGraphVersion": 2}]


@pytest.mark.asyncio
async def test_org_billing_does_not_grant_private_names_or_links(mocker):
    actions = _database(mocker)
    item = (await history.enrich_credit_history([_item()], "user", "org"))[0]
    assert item.agent_name is None
    assert item.library_agent_id is None
    assert item.execution_available is False
    where = actions["AgentGraphExecution"].find_many.call_args_list[0].kwargs["where"]
    assert where["isDeleted"] is False
    assert {"OR": [{"expertId": None}, {"userId": "user"}]} in where["AND"]
    assert (
        actions["LibraryAgent"].find_many.call_args.kwargs["where"]["userId"] == "user"
    )
    assert (
        actions["LibraryAgent"].find_many.call_args.kwargs["where"]["isDeleted"]
        is False
    )


@pytest.mark.asyncio
async def test_nested_runs_have_relationships_without_rolling_up_costs(mocker):
    parent = _execution()
    child = _execution("child", "child-graph", "run")
    _database(
        mocker,
        executions=[parent, child],
        related=[parent, child],
        graphs=[_graph(), _graph("child-graph", name="Research helper")],
        libraries=[_library()],
    )
    parent_item, child_item = await history.enrich_credit_history(
        [_item(), _item("child", "child-graph")], "user"
    )
    assert parent_item.amount == child_item.amount == -12
    assert child_item.parent_execution_id == "run"
    assert child_item.parent_agent_name == "Original agent"
    assert child_item.parent_library_agent_id == "library"
    assert len(parent_item.related_executions) == 1
    related = parent_item.related_executions[0]
    assert related.execution_id == "child"
    assert related.agent_name == "Research helper"
    assert related.library_agent_id is None
    assert related.amount is None


@pytest.mark.asyncio
async def test_copilot_links_only_existing_owned_conversations(mocker):
    session = MagicMock(spec=ChatSession, id="chat", title="Research leads")
    actions = _database(mocker, sessions=[session])
    existing, unavailable = await history.enrich_credit_history(
        [_item("copilot-session-chat"), _item("copilot-session-missing")], "user", "org"
    )
    assert existing.conversation_id == "chat"
    assert existing.conversation_title == "Research leads"
    assert existing.execution_available is False
    assert unavailable.conversation_id is None
    assert unavailable.conversation_title is None
    where = actions["ChatSession"].find_many.call_args.kwargs["where"]
    assert where["userId"] == "user"
    assert where["id"]["in"] == ["chat", "missing"]
    assert where["AND"] == [
        {
            "OR": [
                {"organizationId": "org"},
                {"organizationId": None},
                {"expertId": {"not": None}},
            ]
        }
    ]


@pytest.mark.asyncio
async def test_nonusage_entries_do_not_trigger_metadata_queries(mocker):
    actions = _database(mocker)
    items = [
        CreditTransactionItem(user_id="user", description="Credits added", amount=500)
    ]
    assert await history.enrich_credit_history(items, "user") == items
    for action in actions.values():
        action.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_metadata_queries_are_batched_for_repeated_agent_runs(mocker):
    runs = [_execution(f"run-{number}") for number in range(20)]
    actions = _database(mocker, executions=runs, libraries=[_library()])
    items = await history.enrich_credit_history([_item(run.id) for run in runs], "user")
    assert len(items) == 20
    assert all(item.agent_name == "Original agent" for item in items)
    assert actions["AgentGraphExecution"].find_many.await_count == 2
    assert actions["LibraryAgent"].find_many.await_count == 1
    assert actions["AgentGraph"].find_many.await_count == 1
    assert actions["StoreListingVersion"].find_many.await_count == 1
    actions["ChatSession"].find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_many_children_are_bounded_and_reported_as_incomplete(mocker):
    children = [_execution(f"child-{index}", "graph", "run") for index in range(100)]
    actions = _database(
        mocker, executions=[_execution()], related=children, graphs=[_graph()]
    )
    actions["AgentGraphExecution"].group_by.return_value = [
        {"parentGraphExecutionId": "run", "_count": {"_all": 101}}
    ]
    item = (await history.enrich_credit_history([_item()], "user"))[0]
    assert len(item.related_executions) == 100
    assert item.related_executions_has_more is True
    assert item.amount == -12
    assert actions["AgentGraphExecution"].find_many.call_args.kwargs["take"] == 100
