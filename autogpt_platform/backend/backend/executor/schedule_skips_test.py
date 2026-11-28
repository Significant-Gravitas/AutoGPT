"""A scheduled run that fails the workflow's checks when it fires (#17245): the
schedule is kept, the run skipped, and its owner told once per problem.

These drive the job function and ``schedule_skips`` with the scheduler and the
database manager mocked. The notice's storage is in
``scheduler_runs_skipped_test.py``, and the whole of it, on the database, in
``schedule_skips_db_test.py``.
"""

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import AlertCause
from pytest_mock import MockerFixture

from backend.blocks.basic import StoreValueBlock
from backend.data.graph import Graph, GraphModel, Node
from backend.data.model import CredentialsMetaInput
from backend.executor import scheduler as scheduler_module
from backend.executor.schedule_skips import (
    RunsSkipped,
    note_run_started,
    skip_invalid_run,
    skip_run_of_disabled_step,
)
from backend.executor.scheduler import GraphExecutionJobArgs, _execute_graph
from backend.util.exceptions import BlockDisabledError, GraphValidationError

_PATH = "backend.executor.schedule_skips"
_SCHEDULER = "backend.executor.scheduler"

REMOVED = GraphValidationError(
    "Workflow validation failed: 1 issues on 1 nodes",
    node_errors={"node-1": {"credentials": "Unknown credentials #key-1"}},
)


def _args(**update) -> GraphExecutionJobArgs:
    args = GraphExecutionJobArgs(
        schedule_id="sched-1",
        user_id="user-1",
        graph_id="graph-1",
        graph_version=3,
        agent_name="Morning run",
        cron="* * * * *",
        input_data={},
        input_credentials={
            "unreal_speech_credentials": CredentialsMetaInput.model_validate(
                {
                    "id": "key-1",
                    "provider": "unreal_speech",
                    "type": "api_key",
                    "title": "Work",
                }
            )
        },
    )
    return args.model_copy(update=update)


def _invalid(*messages: str) -> GraphValidationError:
    return GraphValidationError(
        "Workflow validation failed",
        node_errors={"node-1": {f"field_{i}": m for i, m in enumerate(messages)}},
    )


@pytest.fixture
def db(mocker: MockerFixture) -> SimpleNamespace:
    client = SimpleNamespace(
        get_graph_metadata=AsyncMock(return_value=SimpleNamespace(name="Digest")),
        get_library_agent_id_by_graph_id=AsyncMock(return_value="lib-1"),
        raise_alert_condition=AsyncMock(),
        resolve_alert_condition=AsyncMock(return_value=True),
    )
    mocker.patch(f"{_PATH}.get_database_manager_async_client", return_value=client)
    return client


@pytest.fixture
def scheduler(mocker: MockerFixture) -> SimpleNamespace:
    client = SimpleNamespace(
        mark_runs_skipped=AsyncMock(return_value=True),
        clear_runs_skipped=AsyncMock(return_value=True),
    )
    mocker.patch(f"{_PATH}.get_scheduler_client", return_value=client)
    return client


def _raised(db: SimpleNamespace) -> dict:
    db.raise_alert_condition.assert_awaited_once()
    return db.raise_alert_condition.await_args.kwargs


def _kept(scheduler: SimpleNamespace) -> RunsSkipped:
    scheduler.mark_runs_skipped.assert_awaited_once()
    return scheduler.mark_runs_skipped.await_args.kwargs["runs_skipped"]


@pytest.mark.asyncio
async def test_a_run_on_a_removed_key_tells_the_owner_which_key_and_where(
    db, scheduler
):
    args = _args()

    await skip_invalid_run(args, REMOVED)

    raised = _raised(db)
    assert raised["user_id"] == "user-1"
    assert raised["cause"] is AlertCause.SCHEDULE_SKIPPED
    assert raised["cause_key"] == "schedule_skipped:sched-1"
    assert raised["data"] == {
        "cause": "SCHEDULE_SKIPPED",
        "cta_path": "/library/agents/lib-1?activeTab=scheduled&activeItem=sched-1",
        "agent": "Digest",
        "problem": "key_removed",
        "schedule": "Morning run",
        "provider": "Unreal Speech",
        "key_title": "Work",
        "step": None,
    }
    assert scheduler.mark_runs_skipped.await_args.kwargs["fired"] is args
    assert _kept(scheduler).problem == "key_removed"


@pytest.mark.asyncio
async def test_a_run_that_fails_as_its_owner_was_told_tells_them_nothing_more(
    db, scheduler
):
    await skip_invalid_run(_args(), REMOVED)
    told = _kept(scheduler)
    db.raise_alert_condition.reset_mock()
    scheduler.mark_runs_skipped.reset_mock()

    await skip_invalid_run(_args(runs_skipped=told), REMOVED)

    db.raise_alert_condition.assert_not_awaited()
    scheduler.mark_runs_skipped.assert_not_awaited()


@pytest.mark.asyncio
async def test_another_problem_tells_them_again_and_keeps_when_skipping_began(
    db, scheduler
):
    since = datetime(2026, 11, 1, tzinfo=timezone.utc)
    told = RunsSkipped(problem="key_removed", fingerprint="earlier", since=since)

    await skip_invalid_run(_args(runs_skipped=told), _invalid("This field is required"))

    assert _raised(db)["data"]["problem"] == "inputs_invalid"
    assert _kept(scheduler).since == since


@pytest.mark.parametrize(
    "messages, problem",
    [
        pytest.param(["Unknown credentials #key-1"], "key_removed", id="removed"),
        pytest.param(
            [
                "Credentials not available: API key 'Work' expired on "
                "2026-11-01 09:00 UTC. Add a new key."
            ],
            "key_unusable",
            id="expired",
        ),
        pytest.param(
            [
                "Credentials not available: This block keeps what it creates "
                "in the Exa account behind its key, so it can't use the "
                "platform's shared Exa key."
            ],
            "key_unusable",
            id="platform-key-refused",
        ),
        pytest.param(
            ["Invalid credentials: type/provider mismatch"],
            "key_unusable",
            id="mismatch",
        ),
        pytest.param(["These credentials are required"], "key_unusable", id="unset"),
        pytest.param(["This field is required"], "inputs_invalid", id="input"),
        pytest.param(
            ["This field is required", "Unknown credentials #key-1"],
            "key_removed",
            id="removed-first",
        ),
    ],
)
@pytest.mark.asyncio
async def test_each_failure_is_named_for_its_fix(db, scheduler, messages, problem):
    await skip_invalid_run(_args(), _invalid(*messages))

    assert _raised(db)["data"]["problem"] == problem


@pytest.mark.asyncio
async def test_a_key_store_that_cannot_be_read_is_one_problem_whatever_it_says(
    db, scheduler
):
    """Its own error text goes into the message and differs on every run, so
    the notice is keyed by the failing fields, not their messages."""
    await skip_invalid_run(_args(), _invalid("Credentials not available: timeout"))
    told = _kept(scheduler)
    db.raise_alert_condition.reset_mock()

    await skip_invalid_run(
        _args(runs_skipped=told), _invalid("Credentials not available: reset")
    )

    db.raise_alert_condition.assert_not_awaited()


@pytest.mark.parametrize(
    "skip, error",
    [
        pytest.param(skip_invalid_run, REMOVED, id="removed-key"),
        # Its link names no schedule, so nothing fails on the way to the alert.
        pytest.param(
            skip_run_of_disabled_step,
            BlockDisabledError("switched off", block_id=StoreValueBlock().id),
            id="disabled-step",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_schedule_without_an_id_is_kept_but_has_nothing_to_tell_by(
    db, scheduler, caplog, skip, error
):
    with caplog.at_level(logging.WARNING, logger=_PATH):
        await skip(_args(schedule_id=None), error)

    db.raise_alert_condition.assert_not_awaited()
    scheduler.mark_runs_skipped.assert_not_awaited()
    assert "the schedule is kept" in caplog.text


@pytest.mark.asyncio
async def test_a_schedule_edited_since_its_run_withdraws_the_alert(db, scheduler):
    scheduler.mark_runs_skipped.return_value = False

    await skip_invalid_run(_args(), REMOVED)

    db.resolve_alert_condition.assert_awaited_once_with(
        "user-1", "schedule_skipped:sched-1"
    )


@pytest.mark.asyncio
async def test_an_alert_that_cannot_be_raised_is_tried_again_next_run(db, scheduler):
    db.raise_alert_condition.side_effect = ConnectionError("database manager down")

    await skip_invalid_run(_args(), REMOVED)

    scheduler.mark_runs_skipped.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_workflow_the_library_has_no_entry_for_links_to_the_library(
    db, scheduler
):
    db.get_library_agent_id_by_graph_id.return_value = None

    await skip_invalid_run(_args(), REMOVED)

    assert _raised(db)["data"]["cta_path"] == "/library"


@pytest.mark.asyncio
async def test_a_switched_off_step_is_named_and_replaced_in_the_builder(db, scheduler):
    block = StoreValueBlock()
    error = BlockDisabledError("switched off", block_id=block.id)

    await skip_run_of_disabled_step(_args(), error)

    data = _raised(db)["data"]
    assert (data["problem"], data["step"], data["cta_path"]) == (
        "step_disabled",
        block.display_name,
        "/build?flowID=graph-1",
    )
    assert _kept(scheduler).problem == "step_disabled"


def test_a_switched_off_block_fails_the_check_with_a_type_of_its_own(
    mocker: MockerFixture,
):
    """A ``ValueError`` still, for every handler written before the type."""
    block = StoreValueBlock()
    block.disabled = True
    mocker.patch("backend.data.graph.get_block", return_value=block)
    graph = Graph(name="g", description="", nodes=[Node(block_id=block.id)])

    with pytest.raises(BlockDisabledError) as raised:
        GraphModel._validate_graph_get_errors(graph, for_run=True)

    assert raised.value.block_id == block.id
    assert isinstance(raised.value, ValueError)


@pytest.mark.asyncio
async def test_a_run_that_starts_withdraws_the_alert_then_clears_the_notice(
    db, scheduler
):
    told = RunsSkipped(
        problem="key_removed", fingerprint="f", since=datetime.now(timezone.utc)
    )
    order = MagicMock()
    db.resolve_alert_condition.side_effect = lambda *a: order("resolve")
    scheduler.clear_runs_skipped.side_effect = lambda **k: order("clear")

    await note_run_started(_args(runs_skipped=told))

    assert [c.args[0] for c in order.call_args_list] == ["resolve", "clear"]
    db.resolve_alert_condition.assert_awaited_once_with(
        "user-1", "schedule_skipped:sched-1"
    )
    scheduler.clear_runs_skipped.assert_awaited_once_with(
        schedule_id="sched-1", user_id="user-1"
    )


@pytest.mark.asyncio
async def test_a_notice_whose_alert_stays_is_kept_for_the_next_run(db, scheduler):
    told = RunsSkipped(
        problem="key_removed", fingerprint="f", since=datetime.now(timezone.utc)
    )
    db.resolve_alert_condition.side_effect = ConnectionError("down")

    await note_run_started(_args(runs_skipped=told))

    scheduler.clear_runs_skipped.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_run_with_nothing_to_clear_costs_no_calls(db, scheduler):
    await note_run_started(_args())

    db.resolve_alert_condition.assert_not_awaited()
    scheduler.clear_runs_skipped.assert_not_awaited()


@pytest.mark.parametrize(
    "error, handler",
    [
        (REMOVED, "skip_invalid_run"),
        (BlockDisabledError("switched off", block_id="b"), "skip_run_of_disabled_step"),
    ],
)
@pytest.mark.asyncio
async def test_the_job_keeps_a_schedule_whose_run_fails_the_checks(
    mocker: MockerFixture, error: Exception, handler: str
):
    """Before #17245 it deleted the schedule, and told no one."""
    mocker.patch(
        f"{_SCHEDULER}.execution_utils.add_graph_execution",
        new=AsyncMock(side_effect=error),
    )
    db = MagicMock(increment_onboarding_runs=AsyncMock())
    mocker.patch(f"{_SCHEDULER}.get_database_manager_async_client", return_value=db)
    unschedule = MagicMock(delete_schedule=AsyncMock())
    mocker.patch(f"{_SCHEDULER}.get_scheduler_client", return_value=unschedule)
    skip = mocker.patch.object(scheduler_module.schedule_skips, handler, AsyncMock())
    args = _args()

    await _execute_graph(**args.model_dump(mode="json"))

    skip.assert_awaited_once_with(args, error)
    unschedule.delete_schedule.assert_not_awaited()
    db.increment_onboarding_runs.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_job_clears_the_notice_once_a_run_starts(mocker: MockerFixture):
    mocker.patch(
        f"{_SCHEDULER}.execution_utils.add_graph_execution",
        new=AsyncMock(return_value=MagicMock(id="exec-1")),
    )
    mocker.patch(
        f"{_SCHEDULER}.get_database_manager_async_client",
        return_value=MagicMock(increment_onboarding_runs=AsyncMock()),
    )
    started = mocker.patch.object(
        scheduler_module.schedule_skips, "note_run_started", AsyncMock()
    )
    args = _args()

    await _execute_graph(**args.model_dump(mode="json"))

    started.assert_awaited_once_with(args)
