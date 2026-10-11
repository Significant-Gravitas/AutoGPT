"""stop_user_work: a closing account's schedules are removed and its runs stopped."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.scheduled_work import (
    STOP_WAIT_TIMEOUT_SECONDS,
    stop_user_work,
)
from backend.data.execution import ExecutionStatus
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo
from backend.util.exceptions import NotFoundError

MODULE = "backend.api.features.account_deletion.scheduled_work"
USER_ID = "user-1"


def _graph_schedule(job_id: str, schedule_id: str | None) -> GraphExecutionJobInfo:
    return GraphExecutionJobInfo(
        id=job_id,
        name="Daily digest",
        next_run_time="2026-10-11T09:00:00+00:00",
        schedule_id=schedule_id,
        user_id=USER_ID,
        graph_id="graph-1",
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
    )


def _follow_up() -> CopilotTurnJobInfo:
    return CopilotTurnJobInfo(
        id="job-follow-up",
        name="copilot turn (new chat)",
        next_run_time="2026-10-11T10:00:00+00:00",
        schedule_id="job-follow-up",
        user_id=USER_ID,
        message="Check the CI run",
        run_at=None,
        cron="0 10 * * *",
    )


def _paused_expert_routine() -> CopilotTurnJobInfo:
    return CopilotTurnJobInfo(
        id="job-routine",
        name="Weekly report",
        next_run_time="",
        schedule_id="job-routine",
        user_id=USER_ID,
        message="Write the weekly report",
        cron="0 8 * * 1",
        expert_id="expert-1",
        routine_id="routine-1",
    )


def _run(run_id: str) -> MagicMock:
    run = MagicMock()
    run.id = run_id
    return run


@pytest.fixture
def scheduler(mocker: MockerFixture) -> MagicMock:
    client = MagicMock()
    client.get_execution_schedules = AsyncMock(return_value=[])
    client.delete_schedule = AsyncMock()
    client.remove_morning_briefing_schedule = AsyncMock()
    client.delete_nightly_batch_schedule = AsyncMock()
    client.delete_community_rebuild_schedule = AsyncMock()
    mocker.patch(f"{MODULE}.get_scheduler_client", return_value=client)
    return client


@pytest.fixture
def list_runs(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{MODULE}.get_graph_executions", new_callable=AsyncMock, return_value=[]
    )


@pytest.fixture
def stop_run(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(f"{MODULE}.stop_graph_execution", new_callable=AsyncMock)


def _assert_per_user_jobs_removed(scheduler: MagicMock) -> None:
    scheduler.remove_morning_briefing_schedule.assert_awaited_once_with(user_id=USER_ID)
    scheduler.delete_nightly_batch_schedule.assert_awaited_once_with(user_id=USER_ID)
    scheduler.delete_community_rebuild_schedule.assert_awaited_once_with(
        user_id=USER_ID
    )


def _errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_every_schedule_is_deleted_with_the_users_id(
    scheduler: MagicMock, list_runs: AsyncMock, stop_run: AsyncMock
):
    scheduler.get_execution_schedules.return_value = [
        _graph_schedule("job-graph", "job-graph"),
        _follow_up(),
        _paused_expert_routine(),
        _graph_schedule("job-legacy", None),
    ]

    await stop_user_work(USER_ID)

    scheduler.get_execution_schedules.assert_awaited_once_with(
        user_id=USER_ID, include_paused=True
    )
    assert sorted(
        scheduler.delete_schedule.await_args_list, key=lambda c: c.kwargs["schedule_id"]
    ) == [
        call(schedule_id="job-follow-up", user_id=USER_ID),
        call(schedule_id="job-graph", user_id=USER_ID),
        call(schedule_id="job-legacy", user_id=USER_ID),
        call(schedule_id="job-routine", user_id=USER_ID),
    ]


@pytest.mark.asyncio
async def test_the_three_per_user_jobs_are_removed(
    scheduler: MagicMock, list_runs: AsyncMock, stop_run: AsyncMock
):
    await stop_user_work(USER_ID)

    _assert_per_user_jobs_removed(scheduler)


@pytest.mark.asyncio
async def test_running_executions_are_stopped(
    scheduler: MagicMock, list_runs: AsyncMock, stop_run: AsyncMock
):
    list_runs.return_value = [_run("run-1"), _run("run-2")]

    await stop_user_work(USER_ID)

    list_runs.assert_awaited_once_with(
        user_id=USER_ID,
        statuses=[
            ExecutionStatus.INCOMPLETE,
            ExecutionStatus.QUEUED,
            ExecutionStatus.RUNNING,
        ],
    )
    assert stop_run.await_args_list == [
        call(
            user_id=USER_ID,
            graph_exec_id=run_id,
            wait_timeout=STOP_WAIT_TIMEOUT_SECONDS,
            cascade=False,
        )
        for run_id in ("run-1", "run-2")
    ]


@pytest.mark.asyncio
async def test_runs_are_stopped_in_parallel(
    scheduler: MagicMock, list_runs: AsyncMock, stop_run: AsyncMock
):
    list_runs.return_value = [_run("run-1"), _run("run-2")]
    both_started = asyncio.Event()
    started: list[str] = []
    finished: list[str] = []

    async def stop(*, graph_exec_id: str, **_: object) -> None:
        started.append(graph_exec_id)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        finished.append(graph_exec_id)

    stop_run.side_effect = stop

    await stop_user_work(USER_ID)

    assert sorted(finished) == ["run-1", "run-2"]


@pytest.mark.asyncio
async def test_schedules_are_removed_before_runs_are_listed(
    scheduler: MagicMock, list_runs: AsyncMock, stop_run: AsyncMock
):
    scheduler.get_execution_schedules.return_value = [_follow_up()]
    order = MagicMock()
    order.attach_mock(scheduler.delete_schedule, "delete_schedule")
    order.attach_mock(list_runs, "list_runs")

    await stop_user_work(USER_ID)

    assert [c[0] for c in order.mock_calls] == ["delete_schedule", "list_runs"]


@pytest.mark.asyncio
async def test_a_failing_schedule_listing_is_logged_and_the_rest_still_run(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    scheduler.get_execution_schedules.side_effect = RuntimeError("scheduler down")
    list_runs.return_value = [_run("run-1")]
    caplog.set_level(logging.INFO, logger=MODULE)

    await stop_user_work(USER_ID)

    _assert_per_user_jobs_removed(scheduler)
    stop_run.assert_awaited_once()
    [error] = _errors(caplog)
    assert USER_ID in error and "scheduler down" in error


@pytest.mark.asyncio
async def test_one_failing_delete_does_not_stop_the_others(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    scheduler.get_execution_schedules.return_value = [
        _graph_schedule("job-graph", "job-graph"),
        _follow_up(),
    ]
    scheduler.delete_schedule.side_effect = [RuntimeError("rpc failed"), None]
    caplog.set_level(logging.INFO, logger=MODULE)

    await stop_user_work(USER_ID)

    assert scheduler.delete_schedule.await_count == 2
    [error] = _errors(caplog)
    assert USER_ID in error and "job-graph" in error


@pytest.mark.asyncio
async def test_a_failing_per_user_job_does_not_stop_the_rest(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    scheduler.get_execution_schedules.return_value = [_follow_up()]
    scheduler.remove_morning_briefing_schedule.side_effect = RuntimeError("boom")
    list_runs.return_value = [_run("run-1")]
    caplog.set_level(logging.INFO, logger=MODULE)

    await stop_user_work(USER_ID)

    _assert_per_user_jobs_removed(scheduler)
    scheduler.delete_schedule.assert_awaited_once()
    stop_run.assert_awaited_once()
    [error] = _errors(caplog)
    assert "morning briefing" in error and USER_ID in error


@pytest.mark.asyncio
async def test_an_unreachable_scheduler_still_stops_runs(
    mocker: MockerFixture,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    mocker.patch(
        f"{MODULE}.get_scheduler_client", side_effect=ConnectionError("no route")
    )
    list_runs.return_value = [_run("run-1")]
    caplog.set_level(logging.INFO, logger=MODULE)

    await stop_user_work(USER_ID)

    stop_run.assert_awaited_once()
    assert _errors(caplog)


@pytest.mark.asyncio
async def test_one_failing_stop_is_logged_and_the_others_still_stop(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    list_runs.return_value = [_run("run-1"), _run("run-2"), _run("run-3")]
    stop_run.side_effect = [RuntimeError("queue down"), None, TimeoutError("slow")]
    caplog.set_level(logging.INFO, logger=MODULE)

    await stop_user_work(USER_ID)

    assert stop_run.await_count == 3
    [error] = _errors(caplog)
    assert "run-1" in error and USER_ID in error
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("run-3" in w for w in warnings)


@pytest.mark.asyncio
async def test_a_schedule_or_run_already_gone_is_not_an_error(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    scheduler.get_execution_schedules.return_value = [_follow_up()]
    scheduler.delete_schedule.side_effect = NotFoundError("Job not found.")
    list_runs.return_value = [_run("run-1")]
    stop_run.side_effect = NotFoundError("Graph execution not found.")
    caplog.set_level(logging.INFO, logger=MODULE)

    await stop_user_work(USER_ID)

    scheduler.delete_schedule.assert_awaited_once()
    stop_run.assert_awaited_once()
    assert _errors(caplog) == []
    gone = [r.getMessage() for r in caplog.records if "already gone" in r.getMessage()]
    assert len(gone) == 2


@pytest.mark.asyncio
async def test_nothing_raises_when_every_step_fails(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    scheduler.get_execution_schedules.side_effect = RuntimeError("down")
    scheduler.remove_morning_briefing_schedule.side_effect = RuntimeError("down")
    scheduler.delete_nightly_batch_schedule.side_effect = RuntimeError("down")
    scheduler.delete_community_rebuild_schedule.side_effect = RuntimeError("down")
    list_runs.side_effect = RuntimeError("db down")
    caplog.set_level(logging.INFO, logger=MODULE)

    await stop_user_work(USER_ID)

    assert len(_errors(caplog)) == 5


@pytest.mark.asyncio
async def test_a_hung_step_is_cut_off(
    mocker: MockerFixture,
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    mocker.patch(f"{MODULE}.STEP_TIMEOUT_SECONDS", 0.05)

    async def hang(**_: object) -> list[object]:
        await asyncio.sleep(10)
        return []

    scheduler.get_execution_schedules.side_effect = hang
    list_runs.return_value = [_run("run-1")]
    caplog.set_level(logging.INFO, logger=MODULE)

    await asyncio.wait_for(stop_user_work(USER_ID), timeout=2)

    stop_run.assert_awaited_once()
    [error] = _errors(caplog)
    assert "list schedules" in error and USER_ID in error


@pytest.mark.asyncio
async def test_an_empty_user_id_stops_nothing(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger=MODULE)

    await stop_user_work("")

    scheduler.get_execution_schedules.assert_not_awaited()
    scheduler.remove_morning_briefing_schedule.assert_not_awaited()
    list_runs.assert_not_awaited()
    stop_run.assert_not_awaited()
    assert _errors(caplog)
