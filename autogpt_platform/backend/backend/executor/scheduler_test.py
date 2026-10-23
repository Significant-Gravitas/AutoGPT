import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz
from apscheduler.triggers.cron import CronTrigger

from backend.api.model import CreateGraph
from backend.copilot.dream import deadline, scope_crons
from backend.copilot.dream.reaper import REAPER_BUDGET_SECONDS
from backend.copilot.dream.retention import RETENTION_BUDGET_SECONDS
from backend.copilot.dream.scheduling import DREAM_SYSTEM_JOBS
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.learning.dispositions import SkillLearningResult
from backend.data import db
from backend.data.execution import get_graph_executions
from backend.executor import scheduler as scheduler_module
from backend.executor.scheduler import (
    GraphExecutionJobArgs,
    Jobstores,
    Scheduler,
    _build_trigger,
    _execute_graph,
    _memory_scope_may_fire,
    _normalize_cron_day_of_week,
    _register_dream_pass_jobs,
    execute_dream_pass_reaper,
    execute_dream_pass_retention,
    execute_dream_pass_with_status,
    execute_skill_learning_sync,
)
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import NotAuthorizedError, NotFoundError
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
async def test_paused_schedule_excluded_unless_include_paused(server: SpinTestServer):
    """AUTOGPT-SERVER-9NB regression: get_execution_schedules must exclude a
    paused schedule via the SQL-level ``next_run_time IS NOT NULL`` filter
    (Scheduler._get_active_jobs_cached), not just via the old in-Python
    check — and ``include_paused=True`` must still find it through the
    unfiltered ``_get_jobs_cached`` path used by pause/resume lifecycle
    callers. Exercised against a real SQL jobstore table."""
    await db.connect()
    test_user = await create_test_user()
    test_graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()),
        user_id=test_user.id,
    )

    scheduler = get_scheduler_client()
    schedule = await scheduler.add_execution_schedule(
        graph_id=test_graph.id,
        user_id=test_user.id,
        graph_version=1,
        cron="0 0 * * *",
        input_data={"input": "data"},
        input_credentials={},
    )

    assert await scheduler.pause_schedule(schedule.id, user_id=test_user.id) is True

    active_only = await scheduler.get_execution_schedules(
        graph_id=test_graph.id, user_id=test_user.id
    )
    assert len(active_only) == 0

    with_paused = await scheduler.get_execution_schedules(
        graph_id=test_graph.id, user_id=test_user.id, include_paused=True
    )
    assert [s.id for s in with_paused] == [schedule.id]

    assert await scheduler.resume_schedule(schedule.id, user_id=test_user.id) is True
    active_after_resume = await scheduler.get_execution_schedules(
        graph_id=test_graph.id, user_id=test_user.id
    )
    assert [s.id for s in active_after_resume] == [schedule.id]


@pytest.mark.asyncio(loop_scope="session")
async def test_update_schedule_changes_timing_and_inputs_in_place(
    server: SpinTestServer,
):
    """#15281: editing a schedule rewrites its cron, timezone, inputs and name
    on the same job, and the next run moves to the new time."""
    await db.connect()
    test_user = await create_test_user()
    test_graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()),
        user_id=test_user.id,
    )
    scheduler = get_scheduler_client()
    schedule = await scheduler.add_execution_schedule(
        graph_id=test_graph.id,
        user_id=test_user.id,
        graph_version=1,
        cron="0 9 * * 1-5",
        input_data={"input": "data"},
        input_credentials={},
        name="Weekday report",
        user_timezone="UTC",
    )

    try:
        await scheduler.update_execution_schedule(
            schedule_id=schedule.id,
            user_id=test_user.id,
            name="Evening report",
            cron="30 18 * * 1-5",
            user_timezone="America/New_York",
            input_data={"input": "changed"},
        )

        [stored] = await scheduler.get_execution_schedules(test_graph.id, test_user.id)
        assert stored.id == schedule.id
        assert (stored.name, stored.cron, stored.timezone) == (
            "Evening report",
            "30 18 * * 1-5",
            "America/New_York",
        )
        assert stored.input_data == {"input": "changed"}
        next_run = datetime.fromisoformat(stored.next_run_time).astimezone(
            pytz.timezone("America/New_York")
        )
        assert (next_run.hour, next_run.minute) == (18, 30)
        assert next_run.weekday() < 5

        # Fire the stored job the way APScheduler does: the run it starts
        # must carry the edited inputs and point back at this schedule.
        await _execute_graph(
            **stored.model_dump(include=set(GraphExecutionJobArgs.model_fields))
        )
        [fired] = await get_graph_executions(
            graph_id=test_graph.id, user_id=test_user.id
        )
        assert fired.trigger_ref == schedule.id
        assert fired.inputs == {"input": "changed"}
    finally:
        await scheduler.delete_schedule(schedule.id, user_id=test_user.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_update_schedule_is_owner_only_and_all_or_nothing(
    server: SpinTestServer,
):
    """Another user can't edit the schedule, a bad cron changes nothing (not
    even the name sent with it), and editing a paused schedule keeps it
    paused instead of silently re-arming it."""
    await db.connect()
    owner = await create_test_user()
    other = await create_test_user(alt_user=True)
    test_graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()),
        user_id=owner.id,
    )
    scheduler = get_scheduler_client()
    schedule = await scheduler.add_execution_schedule(
        graph_id=test_graph.id,
        user_id=owner.id,
        graph_version=1,
        cron="0 9 * * 1-5",
        input_data={"input": "data"},
        input_credentials={},
        name="Weekday report",
        user_timezone="UTC",
    )

    try:
        with pytest.raises(NotAuthorizedError):
            await scheduler.update_execution_schedule(
                schedule_id=schedule.id, user_id=other.id, cron="0 1 * * *"
            )
        with pytest.raises(ValueError):
            await scheduler.update_execution_schedule(
                schedule_id=schedule.id,
                user_id=owner.id,
                name="Renamed",
                cron="not a cron",
            )
        [stored] = await scheduler.get_execution_schedules(test_graph.id, owner.id)
        assert (stored.name, stored.cron) == ("Weekday report", "0 9 * * 1-5")

        assert await scheduler.pause_schedule(schedule.id, user_id=owner.id)
        await scheduler.update_execution_schedule(
            schedule_id=schedule.id, user_id=owner.id, cron="0 7 * * *"
        )
        [paused] = await scheduler.get_execution_schedules(
            test_graph.id, owner.id, include_paused=True
        )
        assert (paused.cron, paused.next_run_time) == ("0 7 * * *", "")
    finally:
        await scheduler.delete_schedule(schedule.id, user_id=owner.id)


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


class TestAddCopilotTurnScheduleExpertAttribution:
    """A follow-up scheduled from an expert chat must persist that expert id in
    the job args, so the fresh session minted at fire time is scoped to her and
    its runs count as the expert's work. Plain chats persist no expert.

    In-process (no SpinTestServer): the RPC return-value round-trip is unrelated
    to attribution — what matters is the persisted CopilotTurnJobArgs that fire
    time reads back, which is exactly what ``_persist_schedule`` receives here.
    """

    @staticmethod
    def _fake_job() -> MagicMock:
        job = MagicMock(id="cop-1", next_run_time=None)
        job.name = "copilot turn"
        job.trigger = MagicMock(timezone="UTC")
        return job

    def _persisted_args(self, *, expert_id=None):
        s = _stub_scheduler()
        experts_store = MagicMock()
        experts_store.resolve_private_expert_tenancy = MagicMock(
            return_value=("personal-org", "personal-team")
        )
        with (
            patch.object(
                s, "_persist_schedule", return_value=self._fake_job()
            ) as persist,
            # Creation-time expert validation resolves tenancy via
            # run_async; the stub scheduler has no event loop, so hand
            # run_async the mock's (non-coroutine) return value directly.
            patch(
                "backend.executor.scheduler.experts_db",
                return_value=experts_store,
            ),
            patch("backend.executor.scheduler.run_async", new=lambda v: v),
        ):
            s.add_copilot_turn_schedule(
                user_id="user-1",
                session_id=None,
                message="check CI",
                run_at=datetime(2026, 5, 24, 4, 0, tzinfo=timezone.utc),
                user_timezone="UTC",
                expert_id=expert_id,
            )
        return persist.call_args.kwargs["job_args"]

    def test_expert_session_persists_expert_id(self) -> None:
        assert self._persisted_args(expert_id="expert-1").expert_id == "expert-1"

    def test_plain_session_persists_no_expert(self) -> None:
        assert self._persisted_args().expert_id is None


def test_build_trigger_reports_an_unknown_timezone_as_bad_input():
    """APScheduler raises ZoneInfoNotFoundError (a KeyError) for an unknown
    zone. Creating or editing a schedule with one must be a 400, not a 500."""
    with pytest.raises(ValueError, match="Unknown timezone"):
        _build_trigger(cron="0 9 * * *", run_at=None, user_timezone="Mars/Olympus")


class TestUpdateGraphExecutionSchedule:
    """In-process checks of the change set ``update_graph_execution_schedule``
    hands APScheduler. The end-to-end behavior is covered against the real
    scheduler service above; these pin the branches that matter for a
    single, minimal ``modify_job`` write."""

    @staticmethod
    def _job(*, paused: bool = False, kind: str = "graph") -> MagicMock:
        job = MagicMock(id="sched-1")
        job.name = "Weekday report"
        job.trigger = _build_trigger(
            cron="0 9 * * 1,2,3,4,5", run_at=None, user_timezone="UTC"
        )
        job.next_run_time = None if paused else datetime.now(timezone.utc)
        if kind == "graph":
            job.kwargs = GraphExecutionJobArgs(
                schedule_id="sched-1",
                user_id="user-1",
                graph_id="graph-1",
                graph_version=1,
                agent_name="Weekday report",
                cron="0 9 * * 1,2,3,4,5",
                input_data={"input": "data"},
            ).model_dump(mode="json")
        else:
            job.kwargs = {
                "kind": "copilot_turn",
                "schedule_id": "sched-1",
                "user_id": "user-1",
                "message": "hi",
                "cron": "0 9 * * *",
            }
        return job

    def _update(self, job: MagicMock, **fields):
        s = _stub_scheduler()
        s.scheduler.get_job.return_value = job
        s.scheduler.modify_job.return_value = job
        with (
            patch.object(s, "_invalidate_jobs_cache"),
            patch("backend.executor.scheduler.run_async") as run_async,
        ):
            s.update_graph_execution_schedule(
                schedule_id="sched-1", user_id="user-1", **fields
            )
        return s.scheduler.modify_job.call_args.kwargs, run_async

    def test_new_time_and_timezone_move_the_next_run(self) -> None:
        changes, run_async = self._update(
            self._job(), cron="30 18 * * 1,2,3,4,5", user_timezone="America/New_York"
        )
        assert str(changes["trigger"].timezone) == "America/New_York"
        assert changes["kwargs"]["cron"] == "30 18 * * 1,2,3,4,5"
        next_run = changes["next_run_time"].astimezone(
            pytz.timezone("America/New_York")
        )
        assert (next_run.hour, next_run.minute, next_run.weekday() < 5) == (
            18,
            30,
            True,
        )
        assert "name" not in changes
        run_async.assert_not_called()

    def test_a_paused_schedule_stays_paused(self) -> None:
        changes, _ = self._update(self._job(paused=True), cron="0 7 * * *")
        assert "trigger" in changes
        assert "next_run_time" not in changes

    def test_a_rename_leaves_the_timing_alone(self) -> None:
        changes, _ = self._update(self._job(), name="  Evening report  ")
        assert changes["name"] == "Evening report"
        assert changes["kwargs"]["agent_name"] == "Evening report"
        assert changes["kwargs"]["input_data"] == {"input": "data"}
        assert "trigger" not in changes

    def test_new_inputs_are_validated_against_the_graph(self) -> None:
        changes, run_async = self._update(self._job(), input_data={"input": "new"})
        assert changes["kwargs"]["input_data"] == {"input": "new"}
        run_async.assert_called_once()

    def test_a_copilot_follow_up_is_not_an_agent_schedule(self) -> None:
        s = _stub_scheduler()
        s.scheduler.get_job.return_value = self._job(kind="copilot_turn")
        with pytest.raises(NotFoundError):
            s.update_graph_execution_schedule(
                schedule_id="sched-1", user_id="user-1", cron="0 7 * * *"
            )
        s.scheduler.modify_job.assert_not_called()


class TestAddCommunityRebuildSchedule:
    def test_registers_with_expected_cron_and_jobstore(self) -> None:
        s = _stub_scheduler()
        fake_job = MagicMock(id="community_rebuild_abc", next_run_time=None)
        s.scheduler.add_job.return_value = fake_job

        with patch(
            "backend.executor.scheduler.run_async",
            side_effect=RuntimeError("flag provider unavailable"),
        ) as flag_read:
            result = s.add_community_rebuild_schedule(
                user_id="abc", user_timezone="America/New_York"
            )

        flag_read.assert_not_called()
        s.scheduler.add_job.assert_called_once()
        kwargs = s.scheduler.add_job.call_args.kwargs
        # Job id matches the documented per-user convention.
        assert kwargs["id"] == "community_rebuild_abc"
        # Single-fire safety + drop-in replace.
        assert kwargs["max_instances"] == 1
        assert kwargs["replace_existing"] is True
        # Lands in the EXECUTION jobstore (Postgres-backed).
        assert kwargs["jobstore"] == Jobstores.EXECUTION.value
        # Cron is 04:00 on Mondays (P-1.7; see the next test). Trigger repr is the
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

    def test_the_weekly_rebuild_fires_on_mondays(self) -> None:
        """``0 4 * * 0`` is Monday to APScheduler, whose weekdays count from
        0 = Monday (the docs used to say Sunday; the expression is kept)."""
        s = _stub_scheduler()
        s.scheduler.add_job.return_value = MagicMock(id="j", next_run_time=None)
        with patch("backend.executor.scheduler.run_async", return_value=True):
            s.add_community_rebuild_schedule(user_id="abc", user_timezone="UTC")
        trigger = s.scheduler.add_job.call_args.kwargs["trigger"]
        sunday = datetime(2026, 9, 27, tzinfo=timezone.utc)

        fires = trigger.get_next_fire_time(None, sunday)

        assert fires == datetime(2026, 9, 28, 4, tzinfo=timezone.utc)
        assert fires.strftime("%A") == "Monday"

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
        with (
            patch("backend.executor.scheduler.run_async"),
            patch("backend.executor.scheduler.forget_registration"),
        ):
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

    def test_delete_forgets_the_registration_so_ensure_can_re_register(
        self,
    ) -> None:
        """An in-band delete must forget the job in the registry (its id on
        the scope's row, and the Redis marker) — otherwise
        ``ensure_scope_scheduled`` reads the recorded job id and never
        re-registers a cron that no longer exists in APScheduler."""
        s = _stub_scheduler()
        s.scheduler.get_job.return_value = MagicMock()
        with (
            patch("backend.executor.scheduler.run_async") as run_async_mock,
            patch("backend.executor.scheduler.forget_registration") as forget_mock,
        ):
            assert s.delete_community_rebuild_schedule("abc") is True
        scope, job = forget_mock.call_args.args
        assert scope == MemoryScope.for_user("abc")
        assert job.job_id_prefix == "community_rebuild"
        run_async_mock.assert_called_once()


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
        # The account's cron keeps its per-user kwargs, so existing jobs,
        # re-registered ones and a rolled-back scheduler agree.
        assert kwargs["kwargs"] == {"user_id": "abc"}
        trigger_repr = repr(kwargs["trigger"])
        # Daily 03:00 cron — same as the former dream pass cron, but
        # carries the consolidated submitter set.
        assert "hour='3'" in trigger_repr
        assert result["id"] == "dream_nightly_batch_abc"
        assert result["scope_key"] == "abc"
        assert result.get("skipped") is not True

    def test_flag_off_returns_skipped_dict_without_calling_add_job(self) -> None:
        """Layer 2 of the 3-layer flag gating — direct callers
        (admin endpoint, ad-hoc scripts) that bypass the registry must
        STILL be refused when the flag is off."""
        s = _stub_scheduler()
        with patch("backend.executor.scheduler.run_async", return_value=False):
            result = s.add_nightly_batch_schedule(user_id="abc")
        s.scheduler.add_job.assert_not_called()
        assert result == {
            "id": None,
            "user_id": "abc",
            "scope_key": "abc",
            "user_timezone": "UTC",
            "next_run_time": None,
            "skipped": True,
            "reason": "dream_pass_disabled",
        }


class TestAddScopeNightlyBatchAndCommunityRebuildSchedules:
    """The scope-keyed registrations the registry calls."""

    EXPERT = MemoryScope.for_expert("abc", "expert-1")

    def _register(self, method: str, scope: MemoryScope) -> dict:
        s = _stub_scheduler()
        s.scheduler.add_job.side_effect = lambda *a, **kw: MagicMock(
            id=kw["id"], next_run_time=None
        )
        with patch("backend.executor.scheduler.run_async", return_value=True):
            getattr(s, method)(scope=scope, user_timezone="Asia/Tokyo")
        return s.scheduler.add_job.call_args.kwargs

    @pytest.mark.parametrize(
        "method, prefix",
        [
            ("add_scope_nightly_batch_schedule", "dream_nightly_batch"),
            ("add_scope_community_rebuild_schedule", "community_rebuild"),
        ],
    )
    def test_expert_scope_is_keyed_by_its_scope_key(
        self, method: str, prefix: str
    ) -> None:
        kwargs = self._register(method, self.EXPERT)

        assert kwargs["id"] == f"{prefix}_{self.EXPERT.scope_key}"
        assert kwargs["kwargs"] == {
            "user_id": "abc",
            "expert_id": "expert-1",
            "func": scheduler_module._MEMORY_CRONS[prefix][0],
        }
        assert "Asia/Tokyo" in repr(kwargs["trigger"])

    def test_scope_methods_are_reachable_over_rpc(self) -> None:
        """The RPC request schema is built from the signature: a
        ``MemoryScope`` argument has to be constructible there."""
        s = _stub_scheduler()
        s._create_fastapi_endpoint(s.add_scope_nightly_batch_schedule)
        s._create_fastapi_endpoint(s.add_scope_community_rebuild_schedule)
        s._create_fastapi_endpoint(s.remove_scope_memory_jobs)


class TestRemoveScopeMemoryJobsForNightlyBatchAndCommunityRebuild:
    def test_removes_only_that_scopes_crons(self) -> None:
        scope = MemoryScope.for_expert("abc", "expert-1")
        s = _stub_scheduler()
        nightly = MagicMock(id=f"dream_nightly_batch_{scope.scope_key}")
        s.scheduler.get_job.side_effect = lambda job_id, jobstore: (
            nightly if job_id == nightly.id else None
        )

        assert s.remove_scope_memory_jobs(scope=scope) == [nightly.id]

        nightly.remove.assert_called_once()
        looked_up = [call.args[0] for call in s.scheduler.get_job.call_args_list]
        assert looked_up == [
            f"community_rebuild_{scope.scope_key}",
            f"dream_nightly_batch_{scope.scope_key}",
        ]


class TestDeleteNightlyBatchSchedule:
    def test_returns_true_when_job_exists(self) -> None:
        s = _stub_scheduler()
        fake_job = MagicMock()
        s.scheduler.get_job.return_value = fake_job
        with (
            patch("backend.executor.scheduler.run_async"),
            patch("backend.executor.scheduler.forget_registration"),
        ):
            assert s.delete_nightly_batch_schedule("abc") is True
        s.scheduler.get_job.assert_called_once_with(
            "dream_nightly_batch_abc", jobstore=Jobstores.EXECUTION.value
        )
        fake_job.remove.assert_called_once()

    def test_returns_false_when_no_job(self) -> None:
        s = _stub_scheduler()
        s.scheduler.get_job.return_value = None
        assert s.delete_nightly_batch_schedule("abc") is False

    def test_the_registry_forgets_the_cron_before_it_is_removed(self) -> None:
        """A cron a row records is restored by the reconcile, so the record
        goes first; a forget that fails keeps the cron and fails the delete."""
        s = _stub_scheduler()
        job = MagicMock()
        s.scheduler.get_job.return_value = job
        order: list[str] = []
        job.remove.side_effect = lambda: order.append("remove")

        def forgot(*args, **kwargs):
            order.append("forget")
            return True

        with (
            patch("backend.executor.scheduler.run_async", side_effect=forgot),
            patch("backend.executor.scheduler.forget_registration"),
        ):
            assert s.delete_nightly_batch_schedule("abc") is True
        assert order == ["forget", "remove"]

        job.remove.reset_mock()
        with (
            patch("backend.executor.scheduler.run_async", return_value=False),
            patch("backend.executor.scheduler.forget_registration"),
        ):
            assert s.delete_nightly_batch_schedule("abc") is False
        job.remove.assert_not_called()

    def test_delete_forgets_the_registration_so_ensure_can_re_register(
        self,
    ) -> None:
        """Same contract as the community-rebuild delete: removing the
        cron in-band must also forget it in the registry so the next
        memory write registers it again."""
        s = _stub_scheduler()
        s.scheduler.get_job.return_value = MagicMock()
        with (
            patch("backend.executor.scheduler.run_async") as run_async_mock,
            patch("backend.executor.scheduler.forget_registration") as forget_mock,
        ):
            assert s.delete_nightly_batch_schedule("abc") is True
        scope, job = forget_mock.call_args.args
        assert scope == MemoryScope.for_user("abc")
        assert job.job_id_prefix == "dream_nightly_batch"
        run_async_mock.assert_called_once()


class TestAddSkillLearningSchedule:
    """The account's skill-learning cron (#14551), on the registry's
    scope-keyed pattern: recorded like the scope's other crons, the
    account's alone."""

    def _stub(self, existing=None):
        s = _stub_scheduler()
        s.scheduler.get_job.return_value = existing
        s.scheduler.add_job.side_effect = lambda *a, **kw: MagicMock(
            id=kw["id"], next_run_time=None
        )
        return s

    def test_registers_the_accounts_cron_caught_up_only_the_same_morning(
        self,
    ) -> None:
        s = self._stub()
        with patch("backend.executor.scheduler.run_async", return_value=True):
            result = s.add_skill_learning_schedule(
                user_id="abc", user_timezone="Asia/Tokyo"
            )

        kwargs = s.scheduler.add_job.call_args.kwargs
        assert kwargs["id"] == "skill_learning_nightly_abc"
        assert kwargs["kwargs"] == {"user_id": "abc"}
        assert kwargs["max_instances"] == 1
        assert kwargs["misfire_grace_time"] == 6 * 3600
        assert "hour='3'" in repr(kwargs["trigger"])
        assert "Asia/Tokyo" in repr(kwargs["trigger"])
        assert result["id"] == "skill_learning_nightly_abc"
        assert result.get("skipped") is not True

    def test_the_other_memory_crons_keep_the_schedulers_catch_up(self) -> None:
        s = self._stub()
        with patch("backend.executor.scheduler.run_async", return_value=True):
            s.add_nightly_batch_schedule(user_id="abc", user_timezone="UTC")

        assert s.scheduler.add_job.call_args.kwargs["misfire_grace_time"] is None

    def test_an_experts_scope_is_refused(self) -> None:
        """One pass reviews the account's skills and its experts': an expert
        has no skill-learning cron, whoever asks."""
        s = self._stub()
        with patch(
            "backend.executor.scheduler.run_async", return_value=True
        ) as run_async_mock:
            result = s.add_scope_skill_learning_schedule(
                scope=MemoryScope.for_expert("abc", "expert-1"), user_timezone="UTC"
            )

        s.scheduler.add_job.assert_not_called()
        run_async_mock.assert_not_called()
        assert (result["skipped"], result["reason"]) == (True, "account_only")

    def test_flag_off_returns_skipped_without_calling_add_job(self) -> None:
        s = self._stub()
        with patch("backend.executor.scheduler.run_async", return_value=False):
            result = s.add_skill_learning_schedule(user_id="abc")

        s.scheduler.add_job.assert_not_called()
        assert (result["skipped"], result["reason"]) == (
            True,
            "skill_learning_disabled",
        )

    def test_a_cron_in_that_timezone_is_kept_and_reported_registered(self) -> None:
        """The registry records the id it gets back, so a kept cron must not
        come back as a skip; its overdue next run is not pushed back."""
        overdue = datetime(2026, 8, 7, 8, tzinfo=timezone.utc)
        existing = MagicMock(
            id="skill_learning_nightly_abc",
            next_run_time=overdue,
            trigger=CronTrigger.from_crontab("0 3 * * *", timezone="Asia/Tokyo"),
        )
        s = self._stub(existing)
        with patch("backend.executor.scheduler.run_async", return_value=True):
            result = s.add_skill_learning_schedule(
                user_id="abc", user_timezone="Asia/Tokyo"
            )

        s.scheduler.add_job.assert_not_called()
        assert result["id"] == "skill_learning_nightly_abc"
        assert result["next_run_time"] == overdue.isoformat()
        assert result.get("skipped") is not True

    def test_reachable_over_rpc(self) -> None:
        s = _stub_scheduler()
        s._create_fastapi_endpoint(s.add_scope_skill_learning_schedule)


class TestRemoveScopeMemoryJobsWithSkillLearning:
    def test_the_accounts_learning_cron_goes_with_its_scope(self) -> None:
        scope = MemoryScope.for_user("abc")
        s = _stub_scheduler()
        learning = MagicMock(id="skill_learning_nightly_abc")
        s.scheduler.get_job.side_effect = lambda job_id, jobstore: (
            learning if job_id == learning.id else None
        )

        assert s.remove_scope_memory_jobs(scope=scope) == [learning.id]

        learning.remove.assert_called_once()
        looked_up = [call.args[0] for call in s.scheduler.get_job.call_args_list]
        assert looked_up == [
            "community_rebuild_abc",
            "dream_nightly_batch_abc",
            "skill_learning_nightly_abc",
        ]


class TestDeleteSkillLearningSchedule:
    def test_the_registry_forgets_the_cron_before_it_is_removed(self) -> None:
        s = _stub_scheduler()
        job = MagicMock()
        s.scheduler.get_job.return_value = job
        order: list[str] = []
        job.remove.side_effect = lambda: order.append("remove")

        def forgot(*args, **kwargs):
            order.append("forget")
            return True

        with (
            patch("backend.executor.scheduler.run_async", side_effect=forgot),
            patch("backend.executor.scheduler.forget_registration") as forget_mock,
        ):
            assert s.delete_skill_learning_schedule("abc") is True
        assert order == ["forget", "remove"]
        s.scheduler.get_job.assert_called_once_with(
            "skill_learning_nightly_abc", jobstore=Jobstores.EXECUTION.value
        )
        scope, forgotten = forget_mock.call_args.args
        assert scope == MemoryScope.for_user("abc")
        assert forgotten.job_id_prefix == "skill_learning_nightly"

    def test_returns_false_when_no_job(self) -> None:
        s = _stub_scheduler()
        s.scheduler.get_job.return_value = None
        assert s.delete_skill_learning_schedule("abc") is False


# ---------------------------------------------------------------------------
# Execution-time gates for the nightly batch and community crons
# ---------------------------------------------------------------------------


def _run_coroutine(coro, timeout=None, *, cancel_on_timeout=False):
    """``run_async`` stand-in: run the (mocked) coroutine to completion on a
    private loop, leaving the test session's loop alone."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestExecuteNightlyBatchSyncRuntimeGate:
    def test_flag_off_short_circuits_before_calling_submitter_fanout(self) -> None:
        """``DREAM_PASS_ENABLED`` flipped off after registration → the
        cron still fires but never calls ``run_nightly_batch_submit``,
        so no submitter runs."""
        from backend.executor.scheduler import _submit_nightly_batch

        with (
            patch(
                "backend.executor.scheduler.run_async", return_value=False
            ) as run_async_mock,
            patch(
                "backend.copilot.dream.nightly_batch.run_nightly_batch_submit"
            ) as fanout_mock,
        ):
            assert _submit_nightly_batch("abc") is None

        # Exactly one run_async call (the flag check). The fan-out
        # function never gets invoked.
        run_async_mock.assert_called_once()
        fanout_mock.assert_not_called()


class TestNightlyBatchScopeGate:
    """The cron body runs only while the registry lets the scope fire, and
    stamps the scope's last clean run."""

    def _fire(self, *, gate: bool, result, expert_id: str | None = None):
        from backend.executor.scheduler import execute_nightly_batch_sync

        with (
            patch("backend.executor.scheduler.run_async", new=_run_coroutine),
            patch(
                "backend.executor.scheduler._memory_scope_may_fire",
                return_value=gate,
            ),
            patch(
                "backend.executor.scheduler._submit_nightly_batch",
                return_value=result,
            ) as submit,
            patch(
                "backend.executor.scheduler.record_scope_run", new=AsyncMock()
            ) as stamp,
        ):
            returned = execute_nightly_batch_sync("abc", expert_id)
        return returned, submit, stamp

    def test_a_scope_the_registry_holds_back_does_not_run(self) -> None:
        returned, submit, stamp = self._fire(gate=False, result=_nightly_result())

        assert returned is None
        submit.assert_not_called()
        stamp.assert_not_awaited()

    def test_a_clean_run_stamps_the_scope(self) -> None:
        result = _nightly_result(dream=_dream_result())
        returned, submit, stamp = self._fire(gate=True, result=result)

        assert returned is result
        submit.assert_called_once_with("abc", None)
        stamp.assert_awaited_once_with(MemoryScope.for_user("abc"), "nightly")

    def test_an_errored_run_is_not_stamped(self) -> None:
        result = _nightly_result(dream=_dream_result(error="phase 1 LLM down"))
        _, _, stamp = self._fire(gate=True, result=result)

        stamp.assert_not_awaited()

    def test_an_expert_cron_runs_the_expert_scope(self) -> None:
        _, submit, stamp = self._fire(
            gate=True, result=_nightly_result(), expert_id="expert-1"
        )

        submit.assert_called_once_with("abc", "expert-1")
        stamp.assert_awaited_once_with(
            MemoryScope.for_expert("abc", "expert-1"), "nightly"
        )


class TestSkillLearningScopeGate:
    """The learning cron body runs only while the registry lets the
    account's scope fire: a cron straggling past a pause or a wipe does
    nothing."""

    def _fire(self, *, gate: bool, flag: bool = True):
        learned = SkillLearningResult(
            user_id="abc",
            run_id="learn-1",
            trigger="nightly",
            started_at=datetime.now(timezone.utc),
        )
        with (
            patch("backend.executor.scheduler.run_async", new=_run_coroutine),
            patch(
                "backend.executor.scheduler._memory_scope_may_fire",
                return_value=gate,
            ) as gate_mock,
            patch(
                "backend.util.feature_flag.is_feature_enabled",
                new=AsyncMock(return_value=flag),
            ),
            patch(
                "backend.copilot.learning.nightly.run_skill_learning_pass",
                new=AsyncMock(return_value=learned),
            ) as learn,
        ):
            returned = execute_skill_learning_sync("abc")
        return returned, learned, gate_mock, learn

    def test_an_account_the_registry_holds_back_does_not_learn(self) -> None:
        returned, _, gate_mock, learn = self._fire(gate=False)

        assert returned is None
        gate_mock.assert_called_once_with(MemoryScope.for_user("abc"))
        learn.assert_not_awaited()

    def test_an_account_the_registry_lets_fire_learns(self) -> None:
        returned, learned, _, learn = self._fire(gate=True)

        assert returned is learned
        learn.assert_awaited_once_with("abc", trigger="nightly")

    def test_the_flag_off_still_short_circuits(self) -> None:
        returned, _, _, learn = self._fire(gate=True, flag=False)

        assert returned is None
        learn.assert_not_awaited()


class TestCommunityRebuildScopeGate:
    def _fire(self, *, gate: bool, result: dict, expert_id=None):
        from backend.executor.scheduler import execute_community_rebuild

        with (
            patch("backend.executor.scheduler.run_async", new=_run_coroutine),
            patch(
                "backend.executor.scheduler._memory_scope_may_fire",
                return_value=gate,
            ),
            patch(
                "backend.executor.scheduler._rebuild_scope_communities",
                return_value=result,
            ) as rebuild,
            patch(
                "backend.executor.scheduler.record_scope_run", new=AsyncMock()
            ) as stamp,
        ):
            execute_community_rebuild("abc", expert_id)
        return rebuild, stamp

    def test_a_scope_the_registry_holds_back_does_not_run(self) -> None:
        rebuild, stamp = self._fire(gate=False, result={"error": None})

        rebuild.assert_not_called()
        stamp.assert_not_awaited()

    def test_a_clean_rebuild_of_an_expert_stamps_its_scope(self) -> None:
        rebuild, stamp = self._fire(
            gate=True, result={"error": None}, expert_id="expert-1"
        )

        scope = MemoryScope.for_expert("abc", "expert-1")
        rebuild.assert_called_once_with(scope)
        stamp.assert_awaited_once_with(scope, "community")

    def test_an_errored_rebuild_is_not_stamped(self) -> None:
        _, stamp = self._fire(gate=True, result={"error": "OpenRouterError: 502"})
        stamp.assert_not_awaited()


class _SchedulerLoop:
    """A real event loop on its own thread, installed as the scheduler's
    shared loop so ``run_async`` bridges to it as in production."""

    def __enter__(self) -> asyncio.AbstractEventLoop:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.patch = patch.object(scheduler_module, "_event_loop", self.loop)
        self.patch.start()
        return self.loop

    def __exit__(self, *exc) -> None:
        self.patch.stop()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)
        self.loop.close()


def _short_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deadline, "REGISTRY_CALL_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(deadline, "BRIDGE_MARGIN_SECONDS", 0.05)


class _HangRecorder:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def forever(self, *args, **kwargs):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class TestMemoryScopeGateBridge:
    """The scheduler side of the registry gate: the gate's answer passes
    through, and a gate the bridge gives up on skips the run and is
    cancelled instead of raising after 300 s."""

    def test_the_gate_gets_the_expert_lifecycle_lookup(self) -> None:
        gate = AsyncMock(return_value=True)
        scope = MemoryScope.for_expert("abc", "expert-1")
        with (
            patch("backend.executor.scheduler.run_async", new=_run_coroutine),
            patch("backend.executor.scheduler.memory_scope_may_fire", new=gate),
        ):
            assert _memory_scope_may_fire(scope) is True
        gate.assert_awaited_once_with(scope, scheduler_module._expert_scope_status)

    def test_a_gate_the_bridge_gives_up_on_skips_and_is_cancelled(
        self, monkeypatch
    ) -> None:
        _short_deadline(monkeypatch)
        hang = _HangRecorder()
        monkeypatch.setattr(scheduler_module, "memory_scope_may_fire", hang.forever)
        with _SchedulerLoop():
            started = time.monotonic()
            fires = _memory_scope_may_fire(MemoryScope.for_user("abc"))
            elapsed = time.monotonic() - started
            assert hang.cancelled.wait(timeout=2)

        assert fires is False
        assert elapsed < 1.0

    @pytest.mark.parametrize("bridge", ["stamp", "forget"])
    def test_the_other_memory_bridges_cancel_what_they_give_up_on(
        self, monkeypatch, bridge
    ) -> None:
        """Like the gate, the run stamp and the in-band delete's bookkeeping
        are safe to stop anywhere, so they ask ``run_async`` to cancel."""
        _short_deadline(monkeypatch)
        hang = _HangRecorder()
        monkeypatch.setattr(scheduler_module, "record_scope_run", hang.forever)
        monkeypatch.setattr(scheduler_module, "forget_registration", hang.forever)
        scope = MemoryScope.for_user("abc")
        with _SchedulerLoop():
            if bridge == "stamp":
                scheduler_module._stamp_scope_run(scope, "nightly")
            else:
                scheduler_module._forget_dream_registration(scope, DREAM_SYSTEM_JOBS[0])
            assert hang.cancelled.wait(timeout=2)

    @pytest.mark.parametrize(
        "body",
        [
            scheduler_module.execute_nightly_batch_sync,
            scheduler_module.execute_community_rebuild,
        ],
    )
    def test_a_hanging_registry_skips_the_cron_promptly(self, monkeypatch, body):
        """Codex's reproduction through the real bridge and the real gate: a
        registry read that never answers used to raise TimeoutError out of
        the cron body after 300 s and leave the read running."""
        _short_deadline(monkeypatch)
        hang = _HangRecorder()
        registry = SimpleNamespace(get_scope_schedule=hang.forever)
        monkeypatch.setattr(scope_crons, "memory_schedule_db", lambda: registry)
        with (
            _SchedulerLoop(),
            patch("backend.executor.scheduler._submit_nightly_batch") as submit,
            patch("backend.executor.scheduler._rebuild_scope_communities") as rebuild,
        ):
            started = time.monotonic()
            body("abc")
            elapsed = time.monotonic() - started
            assert hang.cancelled.wait(timeout=2)

        assert elapsed < 1.0
        submit.assert_not_called()
        rebuild.assert_not_called()


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
    status writes don't hit an event loop. The admin trigger runs the
    fan-out directly, past the cron body's registry gate.
    """
    from backend.executor.scheduler import execute_nightly_batch_with_status

    with (
        patch("backend.executor.scheduler.run_async"),
        patch(
            "backend.executor.scheduler._submit_nightly_batch",
            return_value=result,
        ) as submit_mock,
        patch("backend.copilot.dream.job_status.mark_complete") as complete_mock,
        patch("backend.copilot.dream.job_status.mark_errored") as errored_mock,
        patch("backend.copilot.dream.job_status.update_status_phase") as phase_mock,
    ):
        execute_nightly_batch_with_status("abc", "job-1")
    submit_mock.assert_called_once_with("abc", trigger="admin")
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


# ---------------------------------------------------------------------------
# The admin dream trigger's ``force``, from the scheduled job to the pass
# ---------------------------------------------------------------------------


class TestAdminDreamPassForce:
    """``force`` rides the one-shot job's kwargs to the pass, whose guard then
    expires a fresh open pass instead of skipping behind it."""

    @pytest.mark.parametrize("force", [True, False])
    def test_the_one_shot_job_carries_force(self, force: bool) -> None:
        s = _stub_scheduler()
        scheduler = MagicMock()
        s.scheduler = scheduler

        s.schedule_immediate_dream_pass(user_id="abc", job_id="job-1", force=force)

        kwargs = scheduler.add_job.call_args.kwargs
        assert kwargs["kwargs"] == {"user_id": "abc", "job_id": "job-1", "force": force}
        assert kwargs["id"] == "adhoc_dream_job-1"

    def test_a_job_persisted_before_force_existed_runs_unforced(self) -> None:
        execute = self._run_wrapper(kwargs={})

        execute.assert_called_once_with(
            "abc", status_id="job-1", trigger="admin", force=False
        )

    def test_force_reaches_the_pass(self) -> None:
        execute = self._run_wrapper(kwargs={"force": True})

        execute.assert_called_once_with(
            "abc", status_id="job-1", trigger="admin", force=True
        )

    def _run_wrapper(self, kwargs: dict) -> MagicMock:
        """Run the wrapper with the pass and the status writers mocked;
        ``run_async`` hands back a clean result for the pass only."""
        execute, _ = _run_dream_wrapper(_dream_result(), **kwargs)
        return execute


class TestExecuteDreamPassWithStatus:
    def test_a_failed_pass_keeps_its_result_on_the_job(self) -> None:
        """A pass stopped by a cancel returns its failure with the usage of
        the phases it was billed for; the errored job keeps that result."""
        failed = _dream_result(error="cancelled: testing")

        _, errored = _run_dream_wrapper(failed)

        errored.assert_called_once_with(
            kind="dream_pass", job_id="job-1", error="cancelled: testing", result=failed
        )


def _run_dream_wrapper(result, **kwargs) -> tuple[MagicMock, MagicMock]:
    """Run the dream pass wrapper with the pass returning *result* and the
    status writers mocked; the pass mock and ``mark_errored``."""
    sentinel = object()
    execute = MagicMock(return_value=sentinel)

    def fake_run_async(coro, timeout=None):
        return result if coro is sentinel else None

    with (
        patch("backend.executor.scheduler.run_async", side_effect=fake_run_async),
        patch("backend.copilot.dream.orchestrator.execute_dream_pass", new=execute),
        patch("backend.copilot.dream.job_status.mark_complete"),
        patch("backend.copilot.dream.job_status.mark_errored") as errored,
        patch("backend.copilot.dream.job_status.update_status_phase"),
    ):
        execute_dream_pass_with_status("abc", "job-1", **kwargs)
    return execute, errored


# ---------------------------------------------------------------------------
# The dream pass reaper and retention: system jobs over every user's passes
# ---------------------------------------------------------------------------


class TestDreamPassReaperAndRetentionJobs:
    def test_both_are_registered_once_for_every_user_not_per_user(self) -> None:
        scheduler = MagicMock()

        _register_dream_pass_jobs(scheduler)

        jobs = {c.kwargs["id"]: c for c in scheduler.add_job.call_args_list}
        assert set(jobs) == {"dream_pass_reaper", "dream_pass_retention"}
        reaper = jobs["dream_pass_reaper"]
        assert reaper.args[0] is execute_dream_pass_reaper
        assert (reaper.kwargs["trigger"], reaper.kwargs["minutes"]) == ("interval", 10)
        retention = jobs["dream_pass_retention"]
        assert retention.args[0] is execute_dream_pass_retention
        trigger = repr(retention.args[1])
        assert "day_of_week='sun'" in trigger and "hour='5'" in trigger
        for job in jobs.values():
            assert job.kwargs["max_instances"] == 1
            assert job.kwargs["replace_existing"] is True
            assert job.kwargs["jobstore"] == Jobstores.EXECUTION.value
            assert "kwargs" not in job.kwargs

    def test_the_reaper_runs_on_the_shared_loop_bounded_by_its_budget(self) -> None:
        sentinel = object()
        with (
            patch("backend.executor.scheduler.run_async") as run_async,
            patch(
                "backend.copilot.dream.reaper.reap_expired_passes",
                new=MagicMock(return_value=sentinel),
            ),
        ):
            execute_dream_pass_reaper()

        run_async.assert_called_once_with(sentinel, timeout=REAPER_BUDGET_SECONDS + 30)

    def test_retention_keeps_the_configured_number_of_days(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "backend.executor.scheduler.config.dream_pass_retention_days", 30
        )
        sentinel = object()
        delete = MagicMock(return_value=sentinel)
        with (
            patch("backend.executor.scheduler.run_async") as run_async,
            patch("backend.copilot.dream.retention.delete_expired_records", new=delete),
        ):
            execute_dream_pass_retention()

        delete.assert_called_once_with(30)
        run_async.assert_called_once_with(
            sentinel, timeout=RETENTION_BUDGET_SECONDS + 60
        )


# ---------------------------------------------------------------------------
# JobStatus transitions for the admin-triggered community rebuild wrapper
# ---------------------------------------------------------------------------


def _run_rebuild_wrapper(result):
    """Invoke the rebuild wrapper with the work body + status writers mocked.

    ``rebuild_communities_for_user`` is stubbed to return a sentinel so the
    ``run_async`` fake can hand back the rebuild ``result`` dict for that
    call only and ``None`` for the (mocked) status writes.
    """
    from backend.executor.scheduler import execute_community_rebuild_with_status

    rebuild_sentinel = object()

    def fake_run_async(coro, timeout=None):
        return result if coro is rebuild_sentinel else None

    with (
        patch("backend.executor.scheduler.run_async", side_effect=fake_run_async),
        # ``new=MagicMock(...)`` — a bare patch() would auto-detect the
        # async target and install an AsyncMock, whose call returns a
        # coroutine instead of the sentinel.
        patch(
            "backend.executor.scheduler.rebuild_communities_for_user",
            new=MagicMock(return_value=rebuild_sentinel),
        ),
        patch("backend.copilot.dream.job_status.mark_complete") as complete_mock,
        patch("backend.copilot.dream.job_status.mark_errored") as errored_mock,
        patch("backend.copilot.dream.job_status.update_status_phase"),
    ):
        execute_community_rebuild_with_status("abc", "job-1")
    return complete_mock, errored_mock


class TestExecuteCommunityRebuildWithStatus:
    def test_errored_rebuild_result_marks_errored_not_complete(self) -> None:
        """``rebuild_communities_for_user`` never raises — failures land
        in ``result['error']``. The admin row must read 'errored';
        'complete' would toast success on a rebuild that DETACH-DELETEd
        every :Community node and then crashed mid-summarization."""
        result = {
            "user_id": "abc",
            "error": "OpenRouterError: 502",
            "skipped": False,
        }
        complete_mock, errored_mock = _run_rebuild_wrapper(result)

        errored_mock.assert_called_once_with(
            kind="rebuild", job_id="job-1", error="OpenRouterError: 502"
        )
        complete_mock.assert_not_called()

    def test_clean_rebuild_result_marks_complete(self) -> None:
        result = {
            "user_id": "abc",
            "error": None,
            "communities_built": 4,
            "skipped": False,
        }
        complete_mock, errored_mock = _run_rebuild_wrapper(result)

        complete_mock.assert_called_once_with(
            kind="rebuild", job_id="job-1", result=result
        )
        errored_mock.assert_not_called()

    def test_skipped_rebuild_result_still_marks_complete(self) -> None:
        """An activity-gated skip is a successful no-op — the result dict
        carries ``skip_reason`` for the visualizer, the row stays
        'complete'."""
        result = {
            "user_id": "abc",
            "error": None,
            "skipped": True,
            "skip_reason": "no_activity",
        }
        complete_mock, errored_mock = _run_rebuild_wrapper(result)

        complete_mock.assert_called_once_with(
            kind="rebuild", job_id="job-1", result=result
        )
        errored_mock.assert_not_called()


class TestExecuteCommunityRebuild:
    def test_rebuild_does_not_evaluate_a_feature_flag(self) -> None:
        from backend.executor.scheduler import _rebuild_scope_communities

        result = {"error": None, "communities_built": 2}
        with (
            patch(
                "backend.executor.scheduler.run_async", return_value=result
            ) as run_async_mock,
            patch(
                "backend.util.feature_flag.is_feature_enabled",
                new=AsyncMock(side_effect=RuntimeError("provider unavailable")),
            ) as flag_mock,
            patch(
                "backend.executor.scheduler.rebuild_communities_for_user",
                new=MagicMock(return_value="rebuild"),
            ) as rebuild_mock,
        ):
            assert _rebuild_scope_communities(MemoryScope.for_user("abc")) == result

        run_async_mock.assert_called_once()
        rebuild_mock.assert_called_once_with("abc", expert_id=None)
        flag_mock.assert_not_called()

    def test_an_expert_scope_rebuilds_the_expert_graph(self) -> None:
        from backend.executor.scheduler import _rebuild_scope_communities

        result = {"error": None, "communities_built": 2}
        with (
            patch("backend.executor.scheduler.run_async", return_value=result),
            patch(
                "backend.executor.scheduler.rebuild_communities_for_user",
                new=MagicMock(return_value="rebuild"),
            ) as rebuild_mock,
        ):
            scope = MemoryScope.for_expert("abc", "expert-1")
            assert _rebuild_scope_communities(scope) == result

        rebuild_mock.assert_called_once_with("abc", expert_id="expert-1")


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


def _counter(name: str, **labels) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_get_active_jobs_cached_sets_the_scheduler_jobs_gauge():
    """autogpt_scheduler_jobs has an alert on it and was never set."""
    from unittest.mock import MagicMock

    from backend.executor.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s._execution_jobstore = MagicMock()
    s._execution_jobstore._get_jobs.return_value = [object(), object(), object()]

    assert len(s._get_active_jobs_cached()) == 3
    assert (
        _counter("autogpt_scheduler_jobs", job_type="execution", status="scheduled")
        == 3
    )


def test_stale_read_invalidated_mid_query_does_not_overwrite_the_gauge():
    """An invalidation that lands while the DB query is in flight rejects the
    cache write; the gauge must be rejected with it, or a slow stale read can
    overwrite a newer count that another reader already published."""
    from unittest.mock import MagicMock

    from backend.executor.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s._execution_jobstore = MagicMock()

    # A fresh, accepted read publishes 5.
    s._execution_jobstore._get_jobs.return_value = [object()] * 5
    s._get_active_jobs_cached()

    def gauge() -> float:
        return _counter(
            "autogpt_scheduler_jobs", job_type="execution", status="scheduled"
        )

    assert gauge() == 5

    # Now a read whose query is interrupted by an invalidation: it returns a
    # different count, but the version moved, so the cache write is skipped.
    s._invalidate_jobs_cache()

    def _slow_query_then_invalidated(*_args, **_kwargs):
        s._invalidate_jobs_cache()
        return [object()] * 2

    s._execution_jobstore._get_jobs.side_effect = _slow_query_then_invalidated
    assert len(s._get_active_jobs_cached()) == 2  # caller still gets the list
    assert gauge() == 5  # but the gauge update was rejected
    s._execution_jobstore._get_jobs.side_effect = None
    s._execution_jobstore._get_jobs.return_value = [object()] * 7
    assert len(s._get_active_jobs_cached()) == 7  # and so was the cache write
