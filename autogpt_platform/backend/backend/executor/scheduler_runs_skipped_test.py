"""A graph schedule's skipped-runs notice (#17245), against a real APScheduler.

The notice lives in the job's own kwargs. A run that fails the workflow's
checks keeps it there (``mark_runs_skipped``), only while the schedule is as
that run read it; a run that starts clears it, and an edit clears it and
withdraws its alert, as a delete does. The fire path is in
``schedule_skips_test.py``, and the whole of it, on the database, in
``schedule_skips_db_test.py``.
"""

import asyncio
from collections.abc import Iterator
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.executor import scheduler as scheduler_module
from backend.executor.schedule_skips import RunsSkipped
from backend.executor.scheduler import _build_trigger
from backend.executor.scheduler_jobstore_test_helpers import (
    CRON,
    USER,
    add_graph_job,
    graph_args,
    pause,
    sqlite_scheduler,
    stored,
)

NOTICE = RunsSkipped(
    problem="key_removed",
    fingerprint="f1",
    since=datetime(2026, 11, 28, 9, tzinfo=timezone.utc),
)


@pytest.fixture
def alerts(mocker: MockerFixture) -> Iterator[AsyncMock]:
    """The alert store the scheduler withdraws from, reached the way the
    service does: through ``run_async`` onto an event loop."""
    loop = asyncio.new_event_loop()
    resolve = AsyncMock(return_value=True)
    mocker.patch(
        "backend.executor.schedule_skips.get_database_manager_async_client",
        return_value=SimpleNamespace(resolve_alert_condition=resolve),
    )
    mocker.patch.object(
        scheduler_module,
        "run_async",
        side_effect=lambda coro, **_: loop.run_until_complete(coro),
    )
    yield resolve
    loop.close()


def test_a_skipped_run_keeps_its_notice_on_the_schedule(tmp_path):
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)

        assert scheduler.mark_runs_skipped(USER, graph_args(), NOTICE) is True

        assert stored(scheduler).runs_skipped == NOTICE


def test_a_notice_is_not_kept_on_a_schedule_edited_since_its_run(tmp_path):
    """The run read the inputs before an edit landed: its notice is about
    inputs the schedule no longer has."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        read_before_the_edit = graph_args().model_copy(
            update={"input_data": {"topic": "before the edit"}}
        )

        assert scheduler.mark_runs_skipped(USER, read_before_the_edit, NOTICE) is False

        assert stored(scheduler).runs_skipped is None


def test_a_notice_is_kept_on_a_schedule_paused_since_its_run(tmp_path):
    """A pause changes when it runs, not what a run of it is made of."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        assert pause(scheduler, for_expert_archive=True)

        assert scheduler.mark_runs_skipped(USER, graph_args(), NOTICE) is True

        after = stored(scheduler)
        assert after.runs_skipped == NOTICE and after.paused_by_expert_archive


def test_a_notice_for_a_deleted_schedule_is_not_kept(tmp_path):
    with sqlite_scheduler(tmp_path) as scheduler:
        assert scheduler.mark_runs_skipped(USER, graph_args(), NOTICE) is False


def test_a_run_that_starts_clears_the_notice(tmp_path):
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        scheduler.mark_runs_skipped(USER, graph_args(), NOTICE)

        assert scheduler.clear_runs_skipped("job-1", USER) is True

        assert stored(scheduler).runs_skipped is None
        assert scheduler.clear_runs_skipped("job-1", USER) is False


@pytest.mark.parametrize(
    "edit",
    [{"name": "Renamed"}, {"cron": "0 7 * * *"}, {"input_data": {"topic": "new"}}],
)
def test_an_edit_clears_the_notice_and_withdraws_its_alert(tmp_path, alerts, edit):
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        scheduler.mark_runs_skipped(USER, graph_args(), NOTICE)
        if "input_data" in edit:
            # The inputs check reads the database; it passes here.
            scheduler._validate_schedule_inputs = MagicMock()

        scheduler.update_graph_execution_schedule("job-1", USER, **edit)

        assert stored(scheduler).runs_skipped is None
    alerts.assert_awaited_once_with(USER, "schedule_skipped:job-1")


def test_deleting_a_schedule_withdraws_its_alert(tmp_path, alerts):
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        scheduler.mark_runs_skipped(USER, graph_args(), NOTICE)

        scheduler.delete_graph_execution_schedule("job-1", USER)

    alerts.assert_awaited_once_with(USER, "schedule_skipped:job-1")


def test_a_schedule_without_a_notice_withdraws_nothing(tmp_path, mocker):
    """It can't have an alert, and an edit or delete waits on no extra call."""
    run_async = mocker.patch.object(scheduler_module, "run_async")
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)

        scheduler.update_graph_execution_schedule("job-1", USER, name="Renamed")
        scheduler.delete_graph_execution_schedule("job-1", USER)

    run_async.assert_not_called()


def test_a_retried_create_after_a_skipped_run_still_returns_its_schedule(tmp_path):
    """The notice is state the scheduler set, like the next run time: a
    create sent again still matches the schedule it made."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        scheduler.mark_runs_skipped(USER, graph_args(), NOTICE)
        trigger = _build_trigger(cron=CRON, run_at=None, user_timezone="UTC")

        replayed = scheduler._replayed_graph_schedule(graph_args(), trigger)

        assert replayed is not None and replayed.id == "job-1"
