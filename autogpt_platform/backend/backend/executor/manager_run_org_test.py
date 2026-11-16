"""Tests for the org check the executor makes when it picks a run up."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionEntry,
    GraphExecutionMeta,
)
from backend.executor.manager import (
    RUN_OUTSIDE_ITS_ORG_ERROR,
    ExecutionProcessor,
    _in_the_runners_org,
)
from backend.util.decorator import TimingInfo


def _run(context: ExecutionContext) -> GraphExecutionEntry:
    return GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
        execution_context=context,
    )


def _run_in_another_org() -> GraphExecutionEntry:
    return _run(ExecutionContext(organization_id="other-org", team_id="other-team"))


@pytest.mark.parametrize(
    "replacement",
    [("own-org", "own-team"), ("other-org", None), (None, None)],
    ids=["has-an-org-of-their-own", "not-on-the-team", "no-org-to-move-to"],
)
def test_a_run_queued_in_an_org_or_team_its_user_isnt_in_fails_the_check(
    replacement: tuple[str | None, str | None],
):
    db_client = MagicMock()
    db_client.resolve_member_tenancy.return_value = replacement

    assert not _in_the_runners_org(db_client, _run_in_another_org())
    db_client.resolve_member_tenancy.assert_called_once_with(
        "user-1", "other-org", "other-team"
    )


def test_a_run_queued_in_its_users_org_passes_the_check():
    db_client = MagicMock()
    db_client.resolve_member_tenancy.return_value = ("org-1", "team-1")

    assert _in_the_runners_org(
        db_client, _run(ExecutionContext(organization_id="org-1", team_id="team-1"))
    )


@pytest.mark.parametrize(
    "context",
    [
        ExecutionContext(organization_id="org-1", expert_id="expert-1"),
        ExecutionContext(),
    ],
    ids=["expert-run", "untagged-run"],
)
def test_expert_and_untagged_runs_arent_checked(context: ExecutionContext):
    db_client = MagicMock()

    assert _in_the_runners_org(db_client, _run(context))
    db_client.resolve_member_tenancy.assert_not_called()


def _start(
    run: GraphExecutionEntry,
    membership: tuple[str | None, str | None],
    status: ExecutionStatus = ExecutionStatus.QUEUED,
    stats: GraphExecutionMeta.Stats | None = None,
    stop_applies: bool = True,
):
    db_client = MagicMock()
    db_client.get_graph_execution_meta.return_value = SimpleNamespace(
        status=status, stats=stats, expert_id=None, trigger_source=None
    )
    db_client.get_node_executions.return_value = [
        SimpleNamespace(node_exec_id="node-exec-1"),
        SimpleNamespace(node_exec_id="node-exec-2"),
    ]
    db_client.resolve_member_tenancy.return_value = membership
    processor = ExecutionProcessor()
    processor._on_graph_execution = MagicMock(
        return_value=(
            TimingInfo(cpu_time=0.0, wall_time=0.0),
            ExecutionStatus.TERMINATED,
        )
    )
    with (
        patch("backend.executor.manager.get_db_client", return_value=db_client),
        patch("backend.executor.manager.send_execution_update"),
        patch("backend.executor.manager.update_graph_execution_state") as state,
        patch("backend.executor.manager.expert_posts.handle_expert_run_post"),
        patch(
            "backend.executor.manager.product_analytics.handle_run_finished"
        ) as analytics,
    ):
        if not stop_applies:
            state.return_value = None
        processor.on_graph_execution(run, threading.Event(), MagicMock())
    return processor, state, db_client, analytics


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.QUEUED,
        ExecutionStatus.INCOMPLETE,
        ExecutionStatus.RUNNING,
        ExecutionStatus.REVIEW,
    ],
)
def test_a_run_queued_in_an_org_its_user_isnt_in_is_stopped(
    status: ExecutionStatus,
):
    """A run queued before runs were checked, or whose user has since left
    the org or team, is stopped before it runs any further, with the steps
    still waiting in it, and its user is told why. A run picked up again
    keeps the record of what it has already done. Like any failed run, it
    shows in the user's activity and analytics."""
    processor, state, db_client, analytics = _start(
        _run_in_another_org(),
        ("own-org", "own-team"),
        status=status,
        stats=GraphExecutionMeta.Stats(cost=27, node_exec_count=3),
    )

    processor._on_graph_execution.assert_not_called()
    db_client.get_node_executions.assert_called_once_with(
        "exec-1", statuses=[ExecutionStatus.QUEUED, ExecutionStatus.REVIEW]
    )
    failed_nodes = {
        call.args[0]
        for call in db_client.update_node_execution_status.call_args_list
        if call.args[1] == ExecutionStatus.FAILED
    }
    assert failed_nodes == {"node-exec-1", "node-exec-2"}
    assert state.call_args.kwargs["status"] == ExecutionStatus.FAILED
    stats = state.call_args.kwargs["stats"]
    assert stats.error == RUN_OUTSIDE_ITS_ORG_ERROR
    assert (stats.cost, stats.node_count) == (27, 3)
    event = db_client.create_activity_event.call_args.kwargs["draft"]
    assert (event.event_type, event.graph_exec_id) == ("run.failed", "exec-1")
    assert analytics.call_args.args[1].status == ExecutionStatus.FAILED


def test_a_stop_that_doesnt_take_effect_isnt_recorded_as_a_failure():
    """A run that has meanwhile ended some other way (its user stopped it,
    say) can't be failed, so no failure is recorded for it."""
    _, _, db_client, analytics = _start(
        _run_in_another_org(), ("own-org", "own-team"), stop_applies=False
    )

    db_client.create_activity_event.assert_not_called()
    analytics.assert_not_called()


def test_a_run_queued_in_its_users_org_runs():
    run = _run(ExecutionContext(organization_id="org-1", team_id="team-1"))

    processor, _, _, _ = _start(run, ("org-1", "team-1"))

    started = processor._on_graph_execution.call_args.kwargs["graph_exec"]
    assert started.execution_context.organization_id == "org-1"
