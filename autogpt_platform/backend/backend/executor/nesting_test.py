"""A sub-workflow run carries the workflows of the runs it is nested in. A run
of one of those workflows, which would start itself again without end, and a
run nested too deep are refused before anything is written."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    ExecutionTrigger,
    GraphExecutionEntry,
    GraphExecutionWithNodes,
)
from backend.data.graph import SUB_WORKFLOW_RUNS_ITSELF, GraphSettings
from backend.data.model import User
from backend.executor.nesting import (
    MAX_SUB_WORKFLOW_DEPTH,
    nested_context,
    sub_workflow_depth,
)
from backend.executor.utils import add_graph_execution
from backend.util.exceptions import GraphValidationError


@pytest.fixture(autouse=True)
def _account_state(mocker: MockerFixture) -> None:
    mocker.patch(
        "backend.executor.utils.onboarding_db.increment_onboarding_runs",
        new=AsyncMock(),
    )
    mocker.patch(
        "backend.executor.utils._spend_approval_required",
        new=AsyncMock(return_value=None),
    )


def test_a_sub_workflow_run_carries_the_workflows_it_is_nested_in():
    assert _chain("outer", "middle").ancestor_graph_ids == ["outer"]
    assert nested_context(_chain("outer", "middle"), "inner").ancestor_graph_ids == [
        "outer",
        "middle",
    ]


@pytest.mark.parametrize(
    "chain, target",
    [
        (["itself"], "itself"),
        (["outer", "inner"], "outer"),
        (["outer", "middle", "inner"], "middle"),
    ],
)
def test_a_run_of_a_workflow_already_in_its_chain_is_refused(
    chain: list[str], target: str
):
    with pytest.raises(GraphValidationError) as refused:
        nested_context(_chain(*chain), target)

    assert refused.value.message == SUB_WORKFLOW_RUNS_ITSELF


def test_sub_workflows_nest_at_most_the_maximum_depth():
    deepest = nested_context(_chain(*_levels(MAX_SUB_WORKFLOW_DEPTH)), "deepest")
    assert len(deepest.ancestor_graph_ids) == MAX_SUB_WORKFLOW_DEPTH

    with pytest.raises(GraphValidationError, match="at most"):
        nested_context(_chain(*_levels(MAX_SUB_WORKFLOW_DEPTH + 1)), "too-deep")


def test_a_queued_runs_depth_is_how_many_runs_it_is_nested_in():
    top = ExecutionContext(root_execution_id="run-1")
    nested = ExecutionContext(root_execution_id="run-0", ancestor_graph_ids=["a", "b"])
    # Queued before runs carried their chain: its depth isn't known
    older_nested = ExecutionContext(root_execution_id="run-0")

    assert sub_workflow_depth(_queued("run-1", top)) == 0
    assert sub_workflow_depth(_queued("run-1", nested)) == 2
    assert sub_workflow_depth(_queued("run-1", older_nested)) is None


@pytest.mark.asyncio
async def test_a_sub_workflow_run_of_a_workflow_in_its_chain_is_refused_before_it_is_written(
    mocker: MockerFixture,
) -> None:
    db = _database_manager(mocker, _created_run(mocker, graph_id="outer"))
    # A node of a run of "inner", itself nested in a run of "outer"
    inner_node = _node_context(
        graph_id="inner", run_id="run-inner", root="run-outer", ancestors=["outer"]
    )

    with pytest.raises(GraphValidationError):
        await add_graph_execution(
            graph_id="outer",
            graph_version=1,
            user_id="user-1",
            execution_context=inner_node,
            trigger=ExecutionTrigger.SUBGRAPH,
            trigger_ref="run-inner",
        )

    db.create_graph_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_sub_workflow_run_is_queued_with_its_chain(
    mocker: MockerFixture,
) -> None:
    run = _created_run(mocker, graph_id="inner")
    _database_manager(mocker, run)
    outer_node = _node_context(graph_id="outer", run_id="run-outer", root="run-outer")

    await add_graph_execution(
        graph_id="inner",
        graph_version=1,
        user_id="user-1",
        execution_context=outer_node,
        trigger=ExecutionTrigger.SUBGRAPH,
        trigger_ref="run-outer",
    )

    context = run.to_graph_execution_entry.call_args.kwargs["execution_context"]
    assert context.ancestor_graph_ids == ["outer"]
    assert context.root_execution_id == "run-outer"
    assert context.parent_execution_id == "run-outer"


def _stored(graph_id: str, parent: str | None = None) -> MagicMock:
    """A stored run of `graph_id`, started by the run `parent` if one is named."""
    return MagicMock(
        id=f"run-{graph_id}",
        graph_id=graph_id,
        graph_version=1,
        trigger_source=ExecutionTrigger.SUBGRAPH if parent else ExecutionTrigger.MANUAL,
        trigger_ref=parent,
    )


@pytest.mark.parametrize(
    "parent, stored, chain",
    [
        ("run-outer", {"run-outer": _stored("outer")}, ["outer"]),
        (
            "run-middle",
            {
                "run-middle": _stored("middle", parent="run-outer"),
                "run-outer": _stored("outer"),
            },
            ["outer", "middle"],
        ),
        # The run the chain started in is gone: its level is left out
        (
            "run-middle",
            {"run-middle": _stored("middle", parent="run-outer")},
            ["middle"],
        ),
    ],
    ids=["one-level", "two-levels", "top-run-gone"],
)
@pytest.mark.asyncio
async def test_a_resumed_sub_workflow_run_gets_its_chain_back(
    mocker: MockerFixture, parent: str, stored: dict, chain: list[str]
) -> None:
    # Resumed after a review, from a context that names no chain
    run = _created_run(mocker, graph_id="inner")
    run.trigger_source = ExecutionTrigger.SUBGRAPH
    run.trigger_ref = parent
    run.nodes_input_masks = {}
    db = _database_manager(mocker, run)
    db.get_graph_execution = AsyncMock(return_value=run)
    db.get_graph_execution_meta = AsyncMock(
        side_effect=lambda user_id, execution_id: stored.get(execution_id)
    )
    now = datetime.now(UTC)
    db.get_user_by_id = AsyncMock(
        return_value=User(
            id="user-1",
            email="user@example.com",
            name=None,
            created_at=now,
            updated_at=now,
            stripe_customer_id=None,
            top_up_config=None,
        )
    )
    db.get_graph_settings = AsyncMock(return_value=GraphSettings())
    db.get_or_create_workspace = AsyncMock(return_value=MagicMock(id="workspace-1"))

    await add_graph_execution(
        graph_id="inner", user_id="user-1", graph_exec_id="run-inner"
    )

    context = run.to_graph_execution_entry.call_args.kwargs["execution_context"]
    assert context.ancestor_graph_ids == chain


def _levels(count: int) -> list[str]:
    return [f"level-{i}" for i in range(count)]


def _chain(*graph_ids: str) -> ExecutionContext:
    """The context of a run of the last workflow, nested in runs of the ones
    before it, as `add_graph_execution` hands it to that run's nodes."""
    context = ExecutionContext(graph_id=graph_ids[0], root_execution_id="run-0")
    for graph_id in graph_ids[1:]:
        context = nested_context(context, graph_id).model_copy(
            update={"graph_id": graph_id}
        )
    return context


def _queued(run_id: str, context: ExecutionContext) -> GraphExecutionEntry:
    return GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id=run_id,
        graph_id="graph-1",
        graph_version=1,
        execution_context=context,
    )


def _node_context(
    *, graph_id: str, run_id: str, root: str, ancestors: list[str] | None = None
) -> ExecutionContext:
    """What a sub-workflow node of that run passes on, as AgentExecutorBlock does."""
    return ExecutionContext(
        user_id="user-1",
        graph_id=graph_id,
        graph_exec_id=run_id,
        root_execution_id=root,
        parent_execution_id=run_id,
        ancestor_graph_ids=ancestors or [],
    )


def _created_run(mocker: MockerFixture, *, graph_id: str) -> MagicMock:
    run = mocker.MagicMock(spec=GraphExecutionWithNodes)
    run.is_dry_run = False
    run.user_id = "user-1"
    run.graph_id = graph_id
    run.graph_version = 1
    run.credential_inputs = None
    run.organization_id = None
    run.team_id = None
    run.expert_id = None
    run.id = f"run-{graph_id}"
    run.node_executions = []
    run.status = ExecutionStatus.QUEUED
    return run


def _database_manager(mocker: MockerFixture, run: MagicMock) -> MagicMock:
    db = mocker.MagicMock()
    db.create_graph_execution = AsyncMock(return_value=run)
    db.update_graph_execution_stats = AsyncMock(return_value=run)
    db.update_node_execution_status_batch = AsyncMock()
    db.increment_onboarding_runs = AsyncMock()
    db.resolve_default_tenancy = AsyncMock(return_value=(None, None))
    mocker.patch(
        "backend.executor.utils.validate_and_construct_node_execution_input",
        return_value=(MagicMock(version=1), [], {}, set()),
    )
    mocker.patch("backend.executor.utils.prisma").is_connected.return_value = False
    mocker.patch(
        "backend.executor.utils.get_database_manager_async_client", return_value=db
    )
    mocker.patch(
        "backend.executor.utils.get_async_execution_queue", return_value=AsyncMock()
    )
    event_bus = MagicMock()
    event_bus.publish = AsyncMock()
    mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus", return_value=event_bus
    )
    return db
