"""stop_user_work: a closing account's schedules and triggers are removed, its
AutoPilot turns cancelled and its runs stopped."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.autopilot_work import (
    cancel_autopilot_turns,
    unlink_chat_platforms,
)
from backend.api.features.account_deletion.scheduled_work import (
    STOP_WAIT_TIMEOUT_SECONDS,
    stop_user_work,
)
from backend.data.execution import ExecutionStatus
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo
from backend.util.exceptions import NotFoundError

MODULE = "backend.api.features.account_deletion.scheduled_work"
AUTOPILOT_MODULE = "backend.api.features.account_deletion.autopilot_work"
BEST_EFFORT_MODULE = "backend.api.features.account_deletion.best_effort"
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


def _node(node_id: str) -> MagicMock:
    node = MagicMock()
    node.id = node_id
    return node


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


@pytest.fixture(autouse=True)
def deactivate_presets(mocker: MockerFixture) -> AsyncMock:
    model = mocker.patch(f"{MODULE}.AgentPreset")
    model.prisma.return_value.update_many = AsyncMock(return_value=0)
    return model.prisma.return_value.update_many


@pytest.fixture(autouse=True)
def list_triggered_nodes(mocker: MockerFixture) -> AsyncMock:
    model = mocker.patch(f"{MODULE}.AgentNode")
    model.prisma.return_value.find_many = AsyncMock(return_value=[])
    return model.prisma.return_value.find_many


@pytest.fixture(autouse=True)
def detach_node(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(f"{MODULE}.set_node_webhook", new_callable=AsyncMock)


@pytest.fixture(autouse=True)
def unlink_chats(mocker: MockerFixture) -> AsyncMock:
    """Tested in autopilot_work_test; it never raises."""
    return mocker.patch(f"{MODULE}.unlink_chat_platforms", new_callable=AsyncMock)


@pytest.fixture(autouse=True)
def cancel_turns(mocker: MockerFixture) -> AsyncMock:
    """Tested in autopilot_work_test; it never raises."""
    return mocker.patch(f"{MODULE}.cancel_autopilot_turns", new_callable=AsyncMock)


def _assert_per_user_jobs_removed(scheduler: MagicMock) -> None:
    """The briefing goes; the memory crons stay, gated, so they resume if the
    account is kept."""
    scheduler.remove_morning_briefing_schedule.assert_awaited_once_with(user_id=USER_ID)
    scheduler.delete_nightly_batch_schedule.assert_not_awaited()
    scheduler.delete_community_rebuild_schedule.assert_not_awaited()


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
    caplog.set_level(logging.INFO, logger=BEST_EFFORT_MODULE)

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
    deactivate_presets: AsyncMock,
    list_triggered_nodes: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    scheduler.get_execution_schedules.side_effect = RuntimeError("down")
    scheduler.remove_morning_briefing_schedule.side_effect = RuntimeError("down")
    deactivate_presets.side_effect = RuntimeError("db down")
    list_triggered_nodes.side_effect = RuntimeError("db down")
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
    mocker.patch(f"{BEST_EFFORT_MODULE}.STEP_TIMEOUT_SECONDS", 0.05)

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
    deactivate_presets: AsyncMock,
    list_triggered_nodes: AsyncMock,
    unlink_chats: AsyncMock,
    cancel_turns: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger=MODULE)

    await stop_user_work("")

    scheduler.get_execution_schedules.assert_not_awaited()
    scheduler.remove_morning_briefing_schedule.assert_not_awaited()
    deactivate_presets.assert_not_awaited()
    list_triggered_nodes.assert_not_awaited()
    unlink_chats.assert_not_awaited()
    cancel_turns.assert_not_awaited()
    list_runs.assert_not_awaited()
    stop_run.assert_not_awaited()
    assert _errors(caplog)


@pytest.mark.asyncio
async def test_every_active_preset_is_switched_off_with_the_users_id(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    deactivate_presets: AsyncMock,
):
    await stop_user_work(USER_ID)

    deactivate_presets.assert_awaited_once_with(
        where={"userId": USER_ID, "isActive": True},
        data={"isActive": False},
    )


@pytest.mark.asyncio
async def test_nodes_on_the_users_webhooks_are_detached(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    list_triggered_nodes: AsyncMock,
    detach_node: AsyncMock,
):
    list_triggered_nodes.return_value = [_node("node-1"), _node("node-2")]

    await stop_user_work(USER_ID)

    list_triggered_nodes.assert_awaited_once_with(
        where={"Webhook": {"is": {"userId": USER_ID}}}
    )
    assert detach_node.await_args_list == [
        call("node-1", None),
        call("node-2", None),
    ]


@pytest.mark.asyncio
async def test_chat_links_and_autopilot_turns_are_stopped_with_the_users_id(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    unlink_chats: AsyncMock,
    cancel_turns: AsyncMock,
):
    await stop_user_work(USER_ID)

    unlink_chats.assert_awaited_once_with(USER_ID)
    cancel_turns.assert_awaited_once_with(USER_ID)


@pytest.mark.asyncio
async def test_triggers_go_before_turns_and_turns_before_runs(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    deactivate_presets: AsyncMock,
    list_triggered_nodes: AsyncMock,
    detach_node: AsyncMock,
    unlink_chats: AsyncMock,
    cancel_turns: AsyncMock,
):
    scheduler.get_execution_schedules.return_value = [_follow_up()]
    list_triggered_nodes.return_value = [_node("node-1")]
    order = MagicMock()
    order.attach_mock(scheduler.delete_schedule, "delete_schedule")
    order.attach_mock(deactivate_presets, "deactivate_presets")
    order.attach_mock(detach_node, "detach_node")
    order.attach_mock(unlink_chats, "unlink_chats")
    order.attach_mock(cancel_turns, "cancel_turns")
    order.attach_mock(list_runs, "list_runs")

    await stop_user_work(USER_ID)

    names = [c[0] for c in order.mock_calls]
    assert sorted(names[:4]) == [
        "deactivate_presets",
        "delete_schedule",
        "detach_node",
        "unlink_chats",
    ]
    assert names[4:] == ["cancel_turns", "list_runs"]


@pytest.mark.asyncio
async def test_a_failing_trigger_step_is_logged_and_the_rest_still_run(
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    deactivate_presets: AsyncMock,
    list_triggered_nodes: AsyncMock,
    detach_node: AsyncMock,
    unlink_chats: AsyncMock,
    cancel_turns: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    scheduler.get_execution_schedules.return_value = [_follow_up()]
    deactivate_presets.side_effect = RuntimeError("db down")
    list_triggered_nodes.return_value = [_node("node-1"), _node("node-2")]
    detach_node.side_effect = [RuntimeError("update failed"), None]
    list_runs.return_value = [_run("run-1")]
    caplog.set_level(logging.INFO, logger=MODULE)

    await stop_user_work(USER_ID)

    assert detach_node.await_count == 2
    scheduler.delete_schedule.assert_awaited_once()
    _assert_per_user_jobs_removed(scheduler)
    unlink_chats.assert_awaited_once_with(USER_ID)
    cancel_turns.assert_awaited_once_with(USER_ID)
    stop_run.assert_awaited_once()
    errors = _errors(caplog)
    assert len(errors) == 2
    assert all(USER_ID in error for error in errors)
    assert any("preset" in error and "db down" in error for error in errors)
    assert any("node-1" in error and "update failed" in error for error in errors)


@pytest.mark.asyncio
async def test_a_hung_trigger_listing_is_cut_off(
    mocker: MockerFixture,
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    list_triggered_nodes: AsyncMock,
    cancel_turns: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    mocker.patch(f"{BEST_EFFORT_MODULE}.STEP_TIMEOUT_SECONDS", 0.05)

    async def hang(**_: object) -> list[object]:
        await asyncio.sleep(10)
        return []

    list_triggered_nodes.side_effect = hang
    list_runs.return_value = [_run("run-1")]
    caplog.set_level(logging.INFO, logger=MODULE)

    await asyncio.wait_for(stop_user_work(USER_ID), timeout=2)

    cancel_turns.assert_awaited_once_with(USER_ID)
    stop_run.assert_awaited_once()
    [error] = _errors(caplog)
    assert "webhook-triggered nodes" in error and USER_ID in error


@pytest.mark.asyncio
async def test_with_every_call_hung_it_still_returns_promptly(
    mocker: MockerFixture,
    scheduler: MagicMock,
    list_runs: AsyncMock,
    stop_run: AsyncMock,
    deactivate_presets: AsyncMock,
    list_triggered_nodes: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    mocker.patch(f"{BEST_EFFORT_MODULE}.STEP_TIMEOUT_SECONDS", 0.05)
    mocker.patch(f"{AUTOPILOT_MODULE}.TURN_STOP_WAIT_SECONDS", 0.05)
    mocker.patch(f"{MODULE}.unlink_chat_platforms", unlink_chat_platforms)
    mocker.patch(f"{MODULE}.cancel_autopilot_turns", cancel_autopilot_turns)

    async def hang(*_: object, **__: object) -> None:
        await asyncio.sleep(10)

    links = MagicMock()
    links.list_server_links = AsyncMock(side_effect=hang)
    links.list_user_links = AsyncMock(side_effect=hang)
    mocker.patch(f"{AUTOPILOT_MODULE}.platform_linking_db", return_value=links)
    mocker.patch(f"{AUTOPILOT_MODULE}.list_queued_sessions", side_effect=hang)
    mocker.patch(f"{AUTOPILOT_MODULE}.get_running_session_ids", side_effect=hang)
    for step in (
        scheduler.get_execution_schedules,
        scheduler.remove_morning_briefing_schedule,
        deactivate_presets,
        list_triggered_nodes,
        list_runs,
    ):
        step.side_effect = hang
    caplog.set_level(logging.INFO, logger=MODULE)

    await asyncio.wait_for(stop_user_work(USER_ID), timeout=2)

    assert len(_errors(caplog)) == 8
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("AutoPilot turn" in w and USER_ID in w for w in warnings)
