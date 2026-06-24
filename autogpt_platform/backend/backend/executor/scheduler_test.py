import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import pytz
from apscheduler.triggers.cron import CronTrigger

from backend.api.model import CreateGraph
from backend.data import db
from backend.executor.scheduler import (
    Jobstores,
    Scheduler,
    _build_trigger,
    _normalize_cron_day_of_week,
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


# ---------------------------------------------------------------------------
# Community rebuild @expose methods — lightweight unit tests
#
# Avoid SpinTestServer for these (Postgres + RabbitMQ overhead is overkill
# for verifying job-args plumbing). Instantiate Scheduler via __new__ so we
# skip AppService.__init__ side effects, then assign a MagicMock to
# ``self.scheduler`` (the APScheduler instance) and assert on its add_job /
# get_job / remove_job calls.
# ---------------------------------------------------------------------------


def _stub_scheduler() -> Scheduler:
    """Build a Scheduler with all real init skipped — for @expose unit tests."""
    s = Scheduler.__new__(Scheduler)
    s.scheduler = MagicMock()
    return s


class TestAddCommunityRebuildSchedule:
    def test_registers_with_expected_cron_and_jobstore(self) -> None:
        s = _stub_scheduler()
        fake_job = MagicMock(id="community_rebuild_abc", next_run_time=None)
        s.scheduler.add_job.return_value = fake_job

        with patch("backend.executor.scheduler.run_async", return_value=True):
            result = s.add_community_rebuild_schedule(
                user_id="abc", user_timezone="America/New_York"
            )

        s.scheduler.add_job.assert_called_once()
        kwargs = s.scheduler.add_job.call_args.kwargs
        # Job id matches the documented per-user convention.
        assert kwargs["id"] == "community_rebuild_abc"
        # Single-fire safety + drop-in replace.
        assert kwargs["max_instances"] == 1
        assert kwargs["replace_existing"] is True
        # Lands in the EXECUTION jobstore (Postgres-backed).
        assert kwargs["jobstore"] == Jobstores.EXECUTION.value
        # Cron is 04:00 Sunday per spec (P-1.7). Trigger repr is the
        # most stable surface to assert against without depending on
        # APScheduler's internal CronTrigger.field accessors.
        trigger_repr = repr(kwargs["trigger"])
        assert "hour='4'" in trigger_repr
        assert "day_of_week='sun'" in trigger_repr or "day_of_week='0'" in trigger_repr
        # User kwargs are passed through to the job body.
        assert kwargs["kwargs"] == {"user_id": "abc"}

        # Returned dict surfaces job id and tz.
        assert result["id"] == "community_rebuild_abc"
        assert result["user_id"] == "abc"
        assert result["user_timezone"] == "America/New_York"

    def test_empty_timezone_falls_back_to_utc(self) -> None:
        s = _stub_scheduler()
        s.scheduler.add_job.return_value = MagicMock(
            id="community_rebuild_abc", next_run_time=None
        )
        with patch("backend.executor.scheduler.run_async", return_value=True):
            result = s.add_community_rebuild_schedule(user_id="abc", user_timezone="")
        assert result["user_timezone"] == "UTC"

    def test_next_run_time_isoformatted_when_present(self) -> None:
        s = _stub_scheduler()
        nrt = datetime(2026, 5, 24, 4, 0, tzinfo=timezone.utc)
        s.scheduler.add_job.return_value = MagicMock(
            id="community_rebuild_abc", next_run_time=nrt
        )
        with patch("backend.executor.scheduler.run_async", return_value=True):
            result = s.add_community_rebuild_schedule(user_id="abc")
        assert result["next_run_time"] == nrt.isoformat()


class TestDeleteCommunityRebuildSchedule:
    def test_returns_true_when_job_exists(self) -> None:
        s = _stub_scheduler()
        fake_job = MagicMock()
        s.scheduler.get_job.return_value = fake_job
        assert s.delete_community_rebuild_schedule("abc") is True
        # Look up by the canonical job id
        s.scheduler.get_job.assert_called_once_with(
            "community_rebuild_abc", jobstore=Jobstores.EXECUTION.value
        )
        fake_job.remove.assert_called_once()

    def test_returns_false_when_no_job(self) -> None:
        s = _stub_scheduler()
        s.scheduler.get_job.return_value = None
        assert s.delete_community_rebuild_schedule("abc") is False


class TestExecuteCommunityRebuildPass:
    def test_default_force_false(self) -> None:
        s = _stub_scheduler()
        sentinel = {"ok": True}
        with (
            patch(
                "backend.executor.scheduler.run_async", return_value=sentinel
            ) as run_async_mock,
            patch(
                "backend.executor.scheduler.rebuild_communities_for_user"
            ) as rebuild_mock,
        ):
            result = s.execute_community_rebuild_pass(user_id="abc")
        # We forwarded to rebuild_communities_for_user with force=False default
        rebuild_mock.assert_called_once_with("abc", force=False)
        # And ran it through run_async (the sync-over-async bridge)
        run_async_mock.assert_called_once()
        assert result == sentinel

    def test_force_propagates_through(self) -> None:
        s = _stub_scheduler()
        with (
            patch("backend.executor.scheduler.run_async", return_value={}),
            patch(
                "backend.executor.scheduler.rebuild_communities_for_user"
            ) as rebuild_mock,
        ):
            s.execute_community_rebuild_pass(user_id="abc", force=True)
        rebuild_mock.assert_called_once_with("abc", force=True)


# ---------------------------------------------------------------------------
# Dream nightly batch @expose methods — registration-time flag gating
#
# The dream pass and ratification pass crons were consolidated into a
# single nightly batch cron. The dream pass + nightly fan-out admin
# entry points moved to fire-and-forget + JobStatus polling via
# ``schedule_immediate_*``; only ``execute_ratification_pass_now``
# remains as a sync @expose method (Cypher-only, finishes in seconds).
# ---------------------------------------------------------------------------


class TestAddNightlyBatchSchedule:
    def test_flag_on_registers_with_03_00_daily_cron(self) -> None:
        s = _stub_scheduler()
        s.scheduler.add_job.return_value = MagicMock(
            id="dream_nightly_batch_abc", next_run_time=None
        )
        with patch("backend.executor.scheduler.run_async", return_value=True):
            result = s.add_nightly_batch_schedule(
                user_id="abc", user_timezone="America/New_York"
            )
        kwargs = s.scheduler.add_job.call_args.kwargs
        assert kwargs["id"] == "dream_nightly_batch_abc"
        assert kwargs["max_instances"] == 1
        assert kwargs["replace_existing"] is True
        assert kwargs["jobstore"] == Jobstores.EXECUTION.value
        trigger_repr = repr(kwargs["trigger"])
        # Daily 03:00 cron — same as the former dream pass cron, but
        # carries the consolidated submitter set.
        assert "hour='3'" in trigger_repr
        assert result["id"] == "dream_nightly_batch_abc"
        assert result.get("skipped") is not True

    def test_flag_off_returns_skipped_dict_without_calling_add_job(self) -> None:
        """Layer 2 of the 3-layer flag gating — direct callers
        (admin endpoint, ad-hoc scripts) that bypass
        ``ensure_dream_system_scheduled`` must STILL be refused when
        the flag is off."""
        s = _stub_scheduler()
        with patch("backend.executor.scheduler.run_async", return_value=False):
            result = s.add_nightly_batch_schedule(user_id="abc")
        s.scheduler.add_job.assert_not_called()
        assert result == {
            "id": None,
            "user_id": "abc",
            "user_timezone": "UTC",
            "next_run_time": None,
            "skipped": True,
            "reason": "dream_pass_disabled",
        }


class TestDeleteNightlyBatchSchedule:
    def test_returns_true_when_job_exists(self) -> None:
        s = _stub_scheduler()
        fake_job = MagicMock()
        s.scheduler.get_job.return_value = fake_job
        assert s.delete_nightly_batch_schedule("abc") is True
        s.scheduler.get_job.assert_called_once_with(
            "dream_nightly_batch_abc", jobstore=Jobstores.EXECUTION.value
        )
        fake_job.remove.assert_called_once()

    def test_returns_false_when_no_job(self) -> None:
        s = _stub_scheduler()
        s.scheduler.get_job.return_value = None
        assert s.delete_nightly_batch_schedule("abc") is False


# ---------------------------------------------------------------------------
# Execution-time flag gate for the nightly batch cron (layer 3)
# ---------------------------------------------------------------------------


class TestExecuteNightlyBatchSyncRuntimeGate:
    def test_flag_off_short_circuits_before_calling_submitter_fanout(self) -> None:
        """``DREAM_PASS_ENABLED`` flipped off after registration → the
        cron still fires but never calls ``run_nightly_batch_submit``,
        so no submitter runs."""
        from backend.executor.scheduler import execute_nightly_batch_sync

        with (
            patch(
                "backend.executor.scheduler.run_async", return_value=False
            ) as run_async_mock,
            patch(
                "backend.copilot.dream.nightly_batch.run_nightly_batch_submit"
            ) as fanout_mock,
        ):
            execute_nightly_batch_sync("abc")

        # Exactly one run_async call (the flag check). The fan-out
        # function never gets invoked.
        run_async_mock.assert_called_once()
        fanout_mock.assert_not_called()


# ---------------------------------------------------------------------------
# JobStatus transitions for the admin-triggered nightly fan-out wrapper
# ---------------------------------------------------------------------------


def _nightly_result(**overrides):
    from backend.copilot.dream.nightly_batch import NightlyBatchResult

    defaults = {
        "user_id": "abc",
        "nightly_id": "nightly-1",
        "started_at": datetime.now(timezone.utc),
        "completed_at": datetime.now(timezone.utc),
        "elapsed_seconds": 1.0,
    }
    defaults.update(overrides)
    return NightlyBatchResult(**defaults)


def _dream_result(**overrides):
    from backend.copilot.dream.schemas import DreamPassResult

    defaults = {"user_id": "abc", "pass_id": "pass-1"}
    defaults.update(overrides)
    return DreamPassResult(**defaults)


def _ratification_result(**overrides):
    from backend.copilot.dream.ratification import RatificationResult

    defaults = {"user_id": "abc", "started_at": datetime.now(timezone.utc)}
    defaults.update(overrides)
    return RatificationResult(**defaults)


def _run_nightly_wrapper(result):
    """Invoke the wrapper with the work body + status writers mocked out.

    ``mark_*`` / ``update_status_phase`` are imported inside the wrapper,
    so patching them at their definition module intercepts the call-time
    import. ``run_async`` is stubbed so the (mocked, non-coroutine)
    status writes don't hit an event loop.
    """
    from backend.executor.scheduler import execute_nightly_batch_with_status

    with (
        patch("backend.executor.scheduler.run_async"),
        patch(
            "backend.executor.scheduler.execute_nightly_batch_sync",
            return_value=result,
        ),
        patch("backend.copilot.dream.job_status.mark_complete") as complete_mock,
        patch("backend.copilot.dream.job_status.mark_errored") as errored_mock,
        patch("backend.copilot.dream.job_status.update_status_phase") as phase_mock,
    ):
        execute_nightly_batch_with_status("abc", "job-1")
    return complete_mock, errored_mock, phase_mock


class TestExecuteNightlyBatchWithStatus:
    def test_clean_sync_result_marks_complete(self) -> None:
        result = _nightly_result(dream=_dream_result())
        complete_mock, errored_mock, phase_mock = _run_nightly_wrapper(result)

        complete_mock.assert_called_once_with(
            kind="nightly", job_id="job-1", result=result
        )
        errored_mock.assert_not_called()
        # Only the initial 'running' transition — never 'submitted'.
        phase_mock.assert_called_once_with(
            kind="nightly", job_id="job-1", state="running"
        )

    def test_error_result_marks_errored_not_complete(self) -> None:
        """``run_nightly_batch_submit`` never raises — a crashed dream
        submitter surfaces in ``result.error``. The admin row must read
        'errored', not 'complete'."""
        result = _nightly_result(error="dream: boom")
        complete_mock, errored_mock, _ = _run_nightly_wrapper(result)

        errored_mock.assert_called_once_with(
            kind="nightly", job_id="job-1", error="dream: boom"
        )
        complete_mock.assert_not_called()

    def test_dream_error_result_marks_errored_not_complete(self) -> None:
        """A dream submitter that ran but returned an error RESULT
        (``result.dream.error`` set, top-level error unset) must also
        surface as errored — otherwise the admin row reads 'complete'
        for a run whose dream pass entirely failed."""
        result = _nightly_result(dream=_dream_result(error="phase 1 LLM down"))
        complete_mock, errored_mock, _ = _run_nightly_wrapper(result)

        errored_mock.assert_called_once_with(
            kind="nightly", job_id="job-1", error="dream: phase 1 LLM down"
        )
        complete_mock.assert_not_called()

    def test_ratification_error_result_marks_errored_not_complete(self) -> None:
        """Same contract for the ratification sub-result — an error
        result from the sweep must not be swallowed by mark_complete."""
        result = _nightly_result(
            dream=_dream_result(),
            ratification=_ratification_result(error="graph down"),
        )
        complete_mock, errored_mock, _ = _run_nightly_wrapper(result)

        errored_mock.assert_called_once_with(
            kind="nightly", job_id="job-1", error="ratification: graph down"
        )
        complete_mock.assert_not_called()

    def test_crash_and_error_result_errors_are_joined(self) -> None:
        """A top-level crash capture and a submitter error result can
        coexist (e.g. dream error result + ratification crash) — both
        must surface on the errored row."""
        result = _nightly_result(
            dream=_dream_result(error="phase 1 LLM down"),
            error="ratification: boom",
        )
        complete_mock, errored_mock, _ = _run_nightly_wrapper(result)

        errored_mock.assert_called_once_with(
            kind="nightly",
            job_id="job-1",
            error="ratification: boom | dream: phase 1 LLM down",
        )
        complete_mock.assert_not_called()

    def test_in_flight_anthropic_batch_marks_complete_with_dream_in_flight(
        self,
    ) -> None:
        """With DREAM_PASS_BATCH_ENABLED on, the dream submitter returns
        as soon as the batch is ENQUEUED. The nightly fan-out is still
        complete — its dream step handed off to the BatchExecutor, whose
        callbacks only ever finalize ``dream_pass`` rows, never this
        nightly row. The row must close out 'complete' (with
        ``dream_in_flight`` on the persisted envelope as the async
        marker), NOT park at 'submitted' until the 6h TTL reaps it."""
        result = _nightly_result(
            dream=_dream_result(execution_path="anthropic_batch"),
            dream_in_flight=True,
        )
        complete_mock, errored_mock, phase_mock = _run_nightly_wrapper(result)

        complete_mock.assert_called_once_with(
            kind="nightly", job_id="job-1", result=result
        )
        assert result.dream_in_flight is True
        errored_mock.assert_not_called()
        # Only the initial 'running' transition — never 'submitted'.
        phase_mock.assert_called_once_with(
            kind="nightly", job_id="job-1", state="running"
        )

    def test_ratification_error_wins_over_in_flight_dream_batch(self) -> None:
        """A ratification-sweep crash alongside an in-flight dream
        batch surfaces as errored — error visibility beats the
        in-flight bookkeeping."""
        result = _nightly_result(
            dream=_dream_result(execution_path="anthropic_batch"),
            dream_in_flight=True,
            error="ratification: boom",
        )
        complete_mock, errored_mock, _ = _run_nightly_wrapper(result)

        errored_mock.assert_called_once_with(
            kind="nightly", job_id="job-1", error="ratification: boom"
        )
        complete_mock.assert_not_called()

    def test_skipped_dream_batch_result_still_marks_complete(self) -> None:
        """A batch-path dream that was SKIPPED (lock held, no input)
        has nothing in flight — the nightly row closes out as complete."""
        result = _nightly_result(
            dream=_dream_result(
                execution_path="anthropic_batch",
                skipped=True,
                skip_reason="no_input",
            )
        )
        complete_mock, errored_mock, _ = _run_nightly_wrapper(result)

        complete_mock.assert_called_once_with(
            kind="nightly", job_id="job-1", result=result
        )
        errored_mock.assert_not_called()


class TestExecuteCommunityRebuildRuntimeGate:
    def test_flag_off_short_circuits_before_rebuild_runs(self) -> None:
        from backend.executor.scheduler import execute_community_rebuild

        # First run_async returns False (flag check). If we let the gate
        # pass, a second call would invoke rebuild_communities_for_user;
        # asserting that doesn't happen is the contract.
        with (
            patch(
                "backend.executor.scheduler.run_async", return_value=False
            ) as run_async_mock,
            patch(
                "backend.executor.scheduler.rebuild_communities_for_user"
            ) as rebuild_mock,
        ):
            execute_community_rebuild("abc")

        run_async_mock.assert_called_once()
        rebuild_mock.assert_not_called()


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
