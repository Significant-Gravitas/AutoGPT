from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from backend.api.model import CreateGraph
from backend.data import db
from backend.executor.scheduler import Jobstores, Scheduler
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
# single nightly batch cron. The underlying ``execute_dream_pass_now``
# and ``execute_ratification_pass_now`` @expose methods remain as
# admin debug endpoints (not cron-registered) — tested below.
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
