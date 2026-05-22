"""Unit tests for scheduler helpers and dispatch — no SpinTestServer required.

These cover the pure functions and the in-process dispatch paths added by
the copilot-turn scheduling feature so they're exercised by the regular
backend test job (and counted by codecov), not just the integration suite.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from backend.executor.scheduler import (
    _MAX_CAP_RETRIES,
    CopilotTurnJobArgs,
    CopilotTurnJobInfo,
    GraphExecutionJobArgs,
    GraphExecutionJobInfo,
    _build_trigger,
    _execute_copilot_turn,
    _job_to_info,
    _next_run_time_iso,
    _reschedule_one_shot_after_cap,
    _self_delete_copilot_turn_schedule,
)

_SCHEDULER_PATH = "backend.executor.scheduler"


# ---------------------------------------------------------------------------
# _build_trigger
# ---------------------------------------------------------------------------


def test_build_trigger_requires_exactly_one_of_cron_or_run_at():
    with pytest.raises(ValueError, match="Exactly one"):
        _build_trigger(cron=None, run_at=None, user_timezone="UTC")
    with pytest.raises(ValueError, match="Exactly one"):
        _build_trigger(
            cron="* * * * *",
            run_at=datetime.now(tz=timezone.utc),
            user_timezone="UTC",
        )


def test_build_trigger_cron_returns_crontrigger():
    trigger = _build_trigger(cron="0 9 * * 1", run_at=None, user_timezone="UTC")
    assert isinstance(trigger, CronTrigger)


def test_build_trigger_run_at_returns_datetrigger():
    when = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    trigger = _build_trigger(cron=None, run_at=when, user_timezone="UTC")
    assert isinstance(trigger, DateTrigger)


# ---------------------------------------------------------------------------
# _next_run_time_iso
# ---------------------------------------------------------------------------


def test_next_run_time_iso_returns_isoformat_when_set():
    job = MagicMock()
    job.next_run_time = datetime(2026, 5, 22, 10, 0, tzinfo=timezone.utc)
    assert _next_run_time_iso(job) == "2026-05-22T10:00:00+00:00"


def test_next_run_time_iso_returns_empty_for_fired_job():
    job = MagicMock()
    job.next_run_time = None
    assert _next_run_time_iso(job) == ""


# ---------------------------------------------------------------------------
# _job_to_info dispatch on kind
# ---------------------------------------------------------------------------


def _mock_job(kwargs: dict) -> MagicMock:
    job = MagicMock()
    job.kwargs = kwargs
    job.id = kwargs.get("schedule_id", "fake-id")
    job.name = "fake-name"
    job.next_run_time = datetime(2026, 5, 22, 10, 0, tzinfo=timezone.utc)
    job.trigger = MagicMock()
    job.trigger.timezone = "UTC"
    return job


def test_job_to_info_graph_kind():
    job = _mock_job(
        {
            "kind": "graph",
            "user_id": "u",
            "graph_id": "g",
            "graph_version": 1,
            "cron": "* * * * *",
            "input_data": {},
            "input_credentials": {},
        }
    )
    info = _job_to_info(job)
    assert isinstance(info, GraphExecutionJobInfo)
    assert info.graph_id == "g"


def test_job_to_info_copilot_turn_kind():
    job = _mock_job(
        {
            "kind": "copilot_turn",
            "user_id": "u",
            "session_id": "s",
            "message": "m",
            "cron": "* * * * *",
        }
    )
    info = _job_to_info(job)
    assert isinstance(info, CopilotTurnJobInfo)
    assert info.session_id == "s"


def test_job_to_info_legacy_rows_without_kind_default_to_graph():
    job = _mock_job(
        {
            # no 'kind' key — predates the discriminator
            "user_id": "u",
            "graph_id": "g",
            "graph_version": 1,
            "cron": "* * * * *",
            "input_data": {},
            "input_credentials": {},
        }
    )
    info = _job_to_info(job)
    assert isinstance(info, GraphExecutionJobInfo)


def test_job_to_info_unknown_kind_returns_none():
    job = _mock_job({"kind": "future_kind_we_dont_know"})
    assert _job_to_info(job) is None


def test_job_to_info_unparseable_returns_none():
    # graph kind but missing required fields
    job = _mock_job({"kind": "graph", "user_id": "u"})
    assert _job_to_info(job) is None


# ---------------------------------------------------------------------------
# _execute_copilot_turn
# ---------------------------------------------------------------------------


def _args(**overrides) -> CopilotTurnJobArgs:
    base = dict(
        schedule_id="sched-1",
        user_id="user-1",
        session_id="session-1",
        message="check CI",
        run_at=datetime.now(tz=timezone.utc) + timedelta(seconds=60),
    )
    base.update(overrides)
    return CopilotTurnJobArgs(**base)


@pytest.mark.asyncio
async def test_execute_copilot_turn_skips_and_self_deletes_when_session_gone():
    args = _args()
    mock_schedule_turn = AsyncMock()
    mock_get_session = AsyncMock(return_value=None)
    mock_self_delete = AsyncMock()

    with (
        patch("backend.executor.scheduler.schedule_turn", new=mock_schedule_turn),
        patch("backend.executor.scheduler.get_chat_session", new=mock_get_session),
        patch(
            f"{_SCHEDULER_PATH}._self_delete_copilot_turn_schedule",
            new=mock_self_delete,
        ),
    ):
        await _execute_copilot_turn(**args.model_dump(mode="json"))

    mock_get_session.assert_awaited_once_with("session-1", "user-1")
    mock_schedule_turn.assert_not_awaited()
    mock_self_delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_copilot_turn_dispatches_when_session_exists():
    args = _args()
    mock_schedule_turn = AsyncMock()
    mock_get_session = AsyncMock(return_value=MagicMock())

    with (
        patch("backend.executor.scheduler.schedule_turn", new=mock_schedule_turn),
        patch("backend.executor.scheduler.get_chat_session", new=mock_get_session),
    ):
        await _execute_copilot_turn(**args.model_dump(mode="json"))

    mock_schedule_turn.assert_awaited_once()
    call_kwargs = mock_schedule_turn.call_args.kwargs
    assert call_kwargs["session_id"] == "session-1"
    assert call_kwargs["message"] == "check CI"
    assert call_kwargs["tool_name"] == "schedule_followup"


@pytest.mark.asyncio
async def test_execute_copilot_turn_concurrency_cap_reschedules_one_shot():
    """When schedule_turn raises ConcurrentTurnLimitError on a one-shot
    schedule, the dispatcher reschedules instead of silently dropping."""
    from backend.copilot.active_turns import ConcurrentTurnLimitError

    args = _args()  # run_at set → one-shot
    mock_schedule_turn = AsyncMock(side_effect=ConcurrentTurnLimitError("cap"))
    mock_get_session = AsyncMock(return_value=MagicMock())
    mock_reschedule = AsyncMock()

    with (
        patch("backend.executor.scheduler.schedule_turn", new=mock_schedule_turn),
        patch("backend.executor.scheduler.get_chat_session", new=mock_get_session),
        patch(f"{_SCHEDULER_PATH}._reschedule_one_shot_after_cap", new=mock_reschedule),
    ):
        await _execute_copilot_turn(**args.model_dump(mode="json"))

    mock_reschedule.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_copilot_turn_concurrency_cap_does_not_reschedule_cron():
    """Cron schedules don't need an explicit reschedule — APScheduler will
    re-fire on the next cron tick."""
    from backend.copilot.active_turns import ConcurrentTurnLimitError

    args = _args(run_at=None, cron="* * * * *")  # recurring
    mock_schedule_turn = AsyncMock(side_effect=ConcurrentTurnLimitError("cap"))
    mock_get_session = AsyncMock(return_value=MagicMock())
    mock_reschedule = AsyncMock()

    with (
        patch("backend.executor.scheduler.schedule_turn", new=mock_schedule_turn),
        patch("backend.executor.scheduler.get_chat_session", new=mock_get_session),
        patch(f"{_SCHEDULER_PATH}._reschedule_one_shot_after_cap", new=mock_reschedule),
    ):
        await _execute_copilot_turn(**args.model_dump(mode="json"))

    mock_reschedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_copilot_turn_swallows_generic_exceptions():
    """Non-concurrency errors should be logged but not crash the scheduler."""
    args = _args()
    mock_schedule_turn = AsyncMock(side_effect=RuntimeError("transient queue error"))
    mock_get_session = AsyncMock(return_value=MagicMock())

    with (
        patch("backend.executor.scheduler.schedule_turn", new=mock_schedule_turn),
        patch("backend.executor.scheduler.get_chat_session", new=mock_get_session),
    ):
        # Must not raise — scheduler can't propagate exceptions out of jobs.
        await _execute_copilot_turn(**args.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# _self_delete_copilot_turn_schedule
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_delete_copilot_turn_no_op_without_schedule_id():
    args = CopilotTurnJobArgs(
        schedule_id=None,
        user_id="u",
        session_id="s",
        message="m",
        run_at=datetime.now(tz=timezone.utc),
    )
    mock_client = AsyncMock()
    with patch(f"{_SCHEDULER_PATH}.get_scheduler_client", return_value=mock_client):
        await _self_delete_copilot_turn_schedule(args)
    mock_client.delete_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_self_delete_copilot_turn_calls_delete_schedule():
    args = _args()
    mock_client = AsyncMock()
    with patch(f"{_SCHEDULER_PATH}.get_scheduler_client", return_value=mock_client):
        await _self_delete_copilot_turn_schedule(args)
    mock_client.delete_schedule.assert_awaited_once_with(
        schedule_id="sched-1", user_id="user-1"
    )


@pytest.mark.asyncio
async def test_self_delete_copilot_turn_swallows_errors():
    args = _args()
    mock_client = AsyncMock()
    mock_client.delete_schedule.side_effect = RuntimeError("scheduler down")
    with patch(f"{_SCHEDULER_PATH}.get_scheduler_client", return_value=mock_client):
        # Must not raise — best-effort cleanup.
        await _self_delete_copilot_turn_schedule(args)


# ---------------------------------------------------------------------------
# _reschedule_one_shot_after_cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reschedule_after_cap_creates_new_schedule_and_bumps_counter():
    args = _args(cap_retry_count=0)
    mock_client = AsyncMock()
    with patch(f"{_SCHEDULER_PATH}.get_scheduler_client", return_value=mock_client):
        await _reschedule_one_shot_after_cap(args)
    mock_client.add_copilot_turn_schedule.assert_awaited_once()
    kwargs = mock_client.add_copilot_turn_schedule.call_args.kwargs
    assert kwargs["session_id"] == "session-1"
    assert kwargs["message"] == "check CI"
    assert kwargs["cap_retry_count"] == 1
    assert kwargs["run_at"] is not None


@pytest.mark.asyncio
async def test_reschedule_after_cap_drops_when_max_retries_reached():
    args = _args(cap_retry_count=_MAX_CAP_RETRIES)
    mock_client = AsyncMock()
    with patch(f"{_SCHEDULER_PATH}.get_scheduler_client", return_value=mock_client):
        await _reschedule_one_shot_after_cap(args)
    mock_client.add_copilot_turn_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_reschedule_after_cap_swallows_errors():
    args = _args(cap_retry_count=0)
    mock_client = AsyncMock()
    mock_client.add_copilot_turn_schedule.side_effect = RuntimeError("scheduler down")
    with patch(f"{_SCHEDULER_PATH}.get_scheduler_client", return_value=mock_client):
        # Must not raise — best-effort retry.
        await _reschedule_one_shot_after_cap(args)


# ---------------------------------------------------------------------------
# GraphExecutionJobArgs back-compat — legacy rows have no `kind`
# ---------------------------------------------------------------------------


def test_graph_args_defaults_kind_to_graph():
    """Existing persisted job kwargs predate the kind discriminator. The
    default must let them deserialize unchanged."""
    args = GraphExecutionJobArgs(
        user_id="u",
        graph_id="g",
        graph_version=1,
        cron="* * * * *",
        input_data={},
        input_credentials={},
    )
    assert args.kind == "graph"


def test_copilot_turn_args_cap_retry_count_defaults_to_zero():
    args = CopilotTurnJobArgs(
        user_id="u",
        session_id="s",
        message="m",
        run_at=datetime.now(tz=timezone.utc),
    )
    assert args.cap_retry_count == 0
