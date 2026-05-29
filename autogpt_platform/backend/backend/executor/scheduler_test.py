import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import pytz
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from backend.api.model import CreateGraph
from backend.copilot.graphiti.communities import CommunityRebuildEnqueueResult
from backend.data import db
from backend.executor.scheduler import (
    Jobstores,
    Scheduler,
    _build_trigger,
    _normalize_cron_day_of_week,
    execute_community_rebuild,
)
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.clients import get_scheduler_client
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_agent_schedule(server: SpinTestServer):
    await db.connect()
    test_user = await create_test_user()
    test_graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()),
        user_id=test_user.id,
    )

    scheduler = get_scheduler_client()
    schedules = await scheduler.get_execution_schedules(test_graph.id, test_user.id)
    assert len(schedules) == 0

    schedule = await scheduler.add_execution_schedule(
        graph_id=test_graph.id,
        user_id=test_user.id,
        graph_version=1,
        cron="0 0 * * *",
        input_data={"input": "data"},
        input_credentials={},
    )
    assert schedule

    schedules = await scheduler.get_execution_schedules(test_graph.id, test_user.id)
    assert len(schedules) == 1
    assert schedules[0].cron == "0 0 * * *"

    await scheduler.delete_schedule(schedule.id, user_id=test_user.id)
    schedules = await scheduler.get_execution_schedules(
        test_graph.id, user_id=test_user.id
    )
    assert len(schedules) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_copilot_turn_schedule_one_shot(server: SpinTestServer):
    await db.connect()
    test_user = await create_test_user(alt_user=True)
    session_id = f"session-{uuid.uuid4()}"

    scheduler = get_scheduler_client()
    # Schedule should not yet exist for this fresh session.
    existing = await scheduler.get_execution_schedules(
        session_id=session_id, user_id=test_user.id
    )
    assert existing == []

    run_at = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    schedule = await scheduler.add_copilot_turn_schedule(
        user_id=test_user.id,
        session_id=session_id,
        message="check on the long-running task",
        run_at=run_at,
        user_timezone="UTC",
    )
    assert schedule.kind == "copilot_turn"
    assert schedule.session_id == session_id
    assert schedule.run_at is not None
    assert schedule.cron is None

    # Polymorphic listing returns the copilot-turn schedule.
    listed = await scheduler.get_execution_schedules(
        session_id=session_id, user_id=test_user.id
    )
    assert len(listed) == 1
    assert listed[0].kind == "copilot_turn"
    assert listed[0].id == schedule.id

    # Graph-only filter excludes copilot-turn schedules.
    graph_only = await scheduler.get_graph_execution_schedules(user_id=test_user.id)
    assert all(s.kind == "graph" for s in graph_only)
    assert schedule.id not in {s.id for s in graph_only}

    # Cleanup — delete_schedule is polymorphic.
    await scheduler.delete_schedule(schedule.id, user_id=test_user.id)
    remaining = await scheduler.get_execution_schedules(
        session_id=session_id, user_id=test_user.id
    )
    assert remaining == []


@pytest.mark.asyncio(loop_scope="session")
async def test_copilot_turn_schedule_requires_cron_xor_run_at(server: SpinTestServer):
    await db.connect()
    test_user = await create_test_user(alt_user=True)
    scheduler = get_scheduler_client()
    session_id = f"session-{uuid.uuid4()}"

    with pytest.raises(Exception) as exc:
        await scheduler.add_copilot_turn_schedule(
            user_id=test_user.id,
            session_id=session_id,
            message="x",
            user_timezone="UTC",
        )
    # ValueError from _build_trigger propagates as a RemoteError
    # through the AppService transport; just verify the call rejected.
    assert exc.value is not None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("*", "*"),
        ("?", "?"),
        ("mon-fri", "mon-fri"),
        ("1-5", "0-4"),
        ("1,3,5", "0,2,4"),
        ("0", "6"),
        ("7", "6"),
        ("0,6", "6,5"),
        ("6-7", "5-6"),
        ("0-4", "6-6,0-3"),
        ("5-2", "4-6,0-1"),
        ("*/2", "*/2"),
        ("1-5/1", "0-4/1"),
    ],
)
def test_normalize_cron_day_of_week_field_translates_unix_to_apscheduler(
    raw: str, expected: str
):
    """Unix-cron uses 0=Sun..6=Sat; APScheduler uses 0=Mon..6=Sun.
    The numbers must be translated, but named tokens and ``*``/``?`` pass
    through unchanged. Wrap-around ranges split into two APS ranges."""
    cron = f"0 9 * * {raw}"
    assert _normalize_cron_day_of_week(cron) == f"0 9 * * {expected}"


def test_build_trigger_with_unix_dow_numbers_fires_on_correct_weekday():
    """Regression: ``0 9 * * 1-5`` on a Saturday must fire next on Monday
    (Unix-cron semantics), not Tuesday (APScheduler's untranslated numbering).
    """
    tz = pytz.timezone("Asia/Jakarta")
    saturday = tz.localize(datetime(2026, 5, 23, 16, 0, 0))

    trigger = _build_trigger(
        cron="0 9 * * 1-5", run_at=None, user_timezone="Asia/Jakarta"
    )
    assert isinstance(trigger, CronTrigger)
    nxt = trigger.get_next_fire_time(None, saturday)
    assert nxt is not None
    assert nxt.strftime("%A %Y-%m-%d %H:%M") == "Monday 2026-05-25 09:00"


@pytest.mark.parametrize(
    "cron,from_dt,expected_dow",
    [
        # Mon-only from Saturday -> Monday
        ("0 9 * * 1", datetime(2026, 5, 23, 16, 0), "Monday"),
        # Sun-only (unix 0) from Saturday -> Sunday
        ("0 9 * * 0", datetime(2026, 5, 23, 16, 0), "Sunday"),
        # Wrap range Sat-Mon (6-1) from Sunday -> Monday
        ("0 9 * * 6-1", datetime(2026, 5, 24, 16, 0), "Monday"),
        # Weekends only (Sat,Sun) from Friday -> Saturday
        ("0 9 * * 0,6", datetime(2026, 5, 22, 16, 0), "Saturday"),
    ],
)
def test_build_trigger_unix_dow_various_cases(
    cron: str, from_dt: datetime, expected_dow: str
):
    tz = pytz.timezone("Asia/Jakarta")
    start = tz.localize(from_dt)
    trigger = _build_trigger(cron=cron, run_at=None, user_timezone="Asia/Jakarta")
    nxt = trigger.get_next_fire_time(None, start)
    assert nxt is not None
    assert nxt.strftime("%A") == expected_dow


def _make_scheduler_with_mock_aps():
    """Build a ``Scheduler`` whose APScheduler is a MagicMock.

    Lets us assert on ``add_job`` kwargs without spinning up the real
    SQLAlchemyJobStore / event loop / thread pool.
    """
    scheduler = Scheduler(register_system_tasks=False)
    scheduler.scheduler = MagicMock()
    return scheduler


def test_execute_community_rebuild_pass_skipped_when_flag_off():
    """LD flag off -> return skipped, never touch the APScheduler."""
    scheduler = _make_scheduler_with_mock_aps()

    with patch(
        "backend.executor.scheduler.run_async", return_value=False
    ) as run_async_mock:
        result = scheduler.execute_community_rebuild_pass(user_id="user-flag-off")

    assert isinstance(result, CommunityRebuildEnqueueResult)
    assert result.skipped is True
    assert result.queued is False
    assert result.skipped_reason == "graphiti_communities_disabled"
    assert result.job_id is None
    scheduler.scheduler.add_job.assert_not_called()
    # Only the flag-check ran_async — no rebuild was driven inline.
    assert run_async_mock.call_count == 1


def test_execute_community_rebuild_pass_enqueues_when_flag_on():
    """LD flag on -> enqueue a one-shot job, return immediately.

    Guards the fix for the original review thread: the @expose endpoint
    must NOT call ``rebuild_communities_for_user`` inline (it'd block
    the small APScheduler thread pool on Leiden + LLM summarization).
    """
    scheduler = _make_scheduler_with_mock_aps()
    fake_job = MagicMock()
    fake_job.id = "community_rebuild_manual_user-flag-on_deadbeef"
    scheduler.scheduler.add_job.return_value = fake_job

    with patch("backend.executor.scheduler.run_async", return_value=True):
        result = scheduler.execute_community_rebuild_pass(user_id="user-flag-on")

    assert isinstance(result, CommunityRebuildEnqueueResult)
    assert result.queued is True
    assert result.skipped is False
    assert result.job_id == fake_job.id
    assert result.user_id == "user-flag-on"

    scheduler.scheduler.add_job.assert_called_once()
    _, kwargs = scheduler.scheduler.add_job.call_args
    assert kwargs["kwargs"] == {"user_id": "user-flag-on"}
    assert kwargs["jobstore"] == Jobstores.EXECUTION.value
    assert kwargs["max_instances"] == 1
    assert isinstance(kwargs["trigger"], DateTrigger)
    # Sanity: the enqueued callable is the same sync wrapper the cron job uses.
    args, _ = scheduler.scheduler.add_job.call_args
    assert args[0] is execute_community_rebuild
