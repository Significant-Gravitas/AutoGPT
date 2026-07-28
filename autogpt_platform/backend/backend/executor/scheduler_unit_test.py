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
    Scheduler,
    _best_effort_unschedule,
    _build_trigger,
    _cleanup_old_schedules_without_id,
    _execute_copilot_turn,
    _job_to_info,
    _next_run_time_iso,
    _reschedule_one_shot_after_cap,
    _self_delete_copilot_turn_schedule,
    reconcile_stripe_tiers,
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
    mock_create_session = AsyncMock()
    mock_self_delete = AsyncMock()

    with (
        patch("backend.executor.scheduler.schedule_turn", new=mock_schedule_turn),
        patch("backend.executor.scheduler.get_chat_session", new=mock_get_session),
        patch(
            "backend.executor.scheduler.create_chat_session", new=mock_create_session
        ),
        patch(
            f"{_SCHEDULER_PATH}._self_delete_copilot_turn_schedule",
            new=mock_self_delete,
        ),
    ):
        await _execute_copilot_turn(**args.model_dump(mode="json"))

    mock_get_session.assert_awaited_once_with("session-1", "user-1")
    mock_create_session.assert_not_awaited()  # only fires when session_id is None
    mock_schedule_turn.assert_not_awaited()
    mock_self_delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_copilot_turn_creates_fresh_session_when_session_id_is_none():
    """Sentinel: when ``session_id`` is ``None`` the executor creates a brand-
    new chat at fire-time and routes the turn into it.  This is the path that
    powers ``schedule_followup`` calls with no explicit session_id.

    The fresh session must land in the org/team captured at schedule time —
    not the user's default org — so an org chat's followups stay in-tenant."""
    args = _args(session_id=None, organization_id="org-sched", team_id="team-sched")
    mock_schedule_turn = AsyncMock()
    mock_get_session = AsyncMock()  # should NOT be called
    new_session = MagicMock(session_id="new-session-uuid")
    mock_create_session = AsyncMock(return_value=new_session)

    with (
        patch("backend.executor.scheduler.schedule_turn", new=mock_schedule_turn),
        patch("backend.executor.scheduler.get_chat_session", new=mock_get_session),
        patch(
            "backend.executor.scheduler.create_chat_session", new=mock_create_session
        ),
    ):
        await _execute_copilot_turn(**args.model_dump(mode="json"))

    mock_create_session.assert_awaited_once_with(
        "user-1",
        dry_run=False,
        organization_id="org-sched",
        team_id="team-sched",
    )
    mock_get_session.assert_not_awaited()  # we created a new one, no lookup
    mock_schedule_turn.assert_awaited_once()
    call_kwargs = mock_schedule_turn.call_args.kwargs
    assert call_kwargs["session_id"] == "new-session-uuid"
    assert call_kwargs["message"] == "check CI"
    assert call_kwargs["organization_id"] == "org-sched"
    assert call_kwargs["team_id"] == "team-sched"


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
# _best_effort_unschedule / _cleanup_old_schedules_without_id
# ---------------------------------------------------------------------------


def _schedule_info(schedule_id: str | None, job_id: str) -> MagicMock:
    info = MagicMock()
    info.schedule_id = schedule_id
    info.id = job_id
    return info


@pytest.mark.asyncio
async def test_unschedule_without_id_runs_targeted_graph_cleanup():
    mock_client = AsyncMock()
    mock_client.get_execution_schedules.return_value = [
        _schedule_info(None, "legacy-job"),
        _schedule_info("sched-9", "valid-job"),
    ]
    with patch(f"{_SCHEDULER_PATH}.get_scheduler_client", return_value=mock_client):
        await _best_effort_unschedule(None, "graph-1", "user-1", reason="test")
    # Only the schedule_id-less legacy job is deleted; valid ones survive.
    mock_client.delete_schedule.assert_awaited_once_with(
        schedule_id="legacy-job", user_id="user-1"
    )


@pytest.mark.asyncio
async def test_unschedule_without_id_or_graph_is_a_no_op():
    mock_client = AsyncMock()
    with patch(f"{_SCHEDULER_PATH}.get_scheduler_client", return_value=mock_client):
        await _best_effort_unschedule(None, None, "user-1", reason="test")
    mock_client.delete_schedule.assert_not_awaited()
    mock_client.get_execution_schedules.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_old_schedules_swallows_per_schedule_delete_errors():
    mock_client = AsyncMock()
    mock_client.get_execution_schedules.return_value = [
        _schedule_info(None, "legacy-1"),
        _schedule_info(None, "legacy-2"),
    ]
    mock_client.delete_schedule.side_effect = [RuntimeError("boom"), None]
    with patch(f"{_SCHEDULER_PATH}.get_scheduler_client", return_value=mock_client):
        # Must not raise, and must still attempt the second delete.
        await _cleanup_old_schedules_without_id("graph-1", user_id="user-1")
    assert mock_client.delete_schedule.await_count == 2


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
async def test_reschedule_after_cap_preserves_user_timezone():
    """The reschedule path must forward the original user_timezone, otherwise
    the new one-shot job's trigger defaults to UTC and the timezone surfaced
    in the job info / logs no longer matches what the user requested."""
    args = _args(cap_retry_count=0, user_timezone="America/New_York")
    mock_client = AsyncMock()
    with patch(f"{_SCHEDULER_PATH}.get_scheduler_client", return_value=mock_client):
        await _reschedule_one_shot_after_cap(args)
    kwargs = mock_client.add_copilot_turn_schedule.call_args.kwargs
    assert kwargs["user_timezone"] == "America/New_York"


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


# ---------------------------------------------------------------------------
# System-job registration — the Stripe tier reconciliation sweep
# ---------------------------------------------------------------------------


def _registered_jobs(monkeypatch, interval_hours: int) -> list:
    """Drive ``Scheduler.run_service`` with every heavy dependency stubbed and
    a mock APScheduler, returning the list of ``add_job`` mock calls."""
    monkeypatch.setattr(
        f"{_SCHEDULER_PATH}.config.stripe_tier_reconcile_interval_hours",
        interval_hours,
    )
    mock_scheduler = MagicMock()
    with (
        patch(f"{_SCHEDULER_PATH}.BackgroundScheduler", return_value=mock_scheduler),
        patch(f"{_SCHEDULER_PATH}.load_dotenv"),
        patch(f"{_SCHEDULER_PATH}.asyncio.new_event_loop", return_value=MagicMock()),
        patch(f"{_SCHEDULER_PATH}.threading.Thread", return_value=MagicMock()),
        patch(f"{_SCHEDULER_PATH}.create_engine", return_value=MagicMock()),
        patch(f"{_SCHEDULER_PATH}.SQLAlchemyJobStore", return_value=MagicMock()),
        patch(f"{_SCHEDULER_PATH}.MemoryJobStore", return_value=MagicMock()),
        patch(
            f"{_SCHEDULER_PATH}._extract_schema_from_url",
            return_value=("public", "sqlite://"),
        ),
        patch(f"{_SCHEDULER_PATH}.ensure_embeddings_coverage", return_value=None),
        # super().run_service() blocks forever keeping the service alive; no-op it.
        patch("backend.util.service.AppService.run_service", return_value=None),
    ):
        Scheduler(register_system_tasks=True).run_service()
    return mock_scheduler.add_job.call_args_list


def test_reconcile_stripe_tiers_job_registered_with_interval_and_single_instance(
    monkeypatch,
):
    """The sweep must be registered as an interval job keyed off the configured
    interval setting and capped to a single concurrent instance."""
    calls = _registered_jobs(monkeypatch, interval_hours=6)

    matches = [c for c in calls if c.args and c.args[0] is reconcile_stripe_tiers]
    assert len(matches) == 1
    call = matches[0]
    assert call.kwargs["id"] == "reconcile_stripe_tiers"
    assert call.kwargs["trigger"] == "interval"
    assert call.kwargs["max_instances"] == 1
    # 6 hours -> 6 * 3600 seconds, driven by the config setting.
    assert call.kwargs["seconds"] == 6 * 3600


def test_reconcile_stripe_tiers_interval_follows_config_setting(monkeypatch):
    """Changing the configured interval changes the registered ``seconds``."""
    calls = _registered_jobs(monkeypatch, interval_hours=12)
    match = next(c for c in calls if c.args and c.args[0] is reconcile_stripe_tiers)
    assert match.kwargs["seconds"] == 12 * 3600


class TestScheduleOrgVisibility:
    """Org/team visibility filtering in get_execution_schedules."""

    def _scheduler_with_jobs(self, infos):
        from unittest.mock import MagicMock

        from backend.executor.scheduler import Scheduler

        sched = Scheduler.__new__(Scheduler)
        jobs = []
        for info in infos:
            job = MagicMock()
            job.next_run_time = datetime(2026, 5, 22, 10, 0, tzinfo=timezone.utc)
            jobs.append((job, info))

        def fake_job_to_info(job):
            for j, i in jobs:
                if j is job:
                    return i
            return None

        return sched, [j for j, _ in jobs], fake_job_to_info

    def _graph_info(self, *, user_id, organization_id="", team_id=None, sid="s1"):
        return GraphExecutionJobInfo(
            id=sid,
            name="n",
            next_run_time="2026-05-22T10:00:00+00:00",
            schedule_id=sid,
            user_id=user_id,
            graph_id="g1",
            graph_version=1,
            cron="* * * * *",
            input_data={},
            input_credentials={},
            organization_id=organization_id,
            team_id=team_id,
        )

    def _run(self, infos, **kwargs):
        from unittest.mock import patch

        sched, jobs, fake_job_to_info = self._scheduler_with_jobs(infos)
        with (
            patch.object(Scheduler, "_get_jobs_cached", lambda self: jobs),
            patch(
                "backend.executor.scheduler._job_to_info",
                side_effect=fake_job_to_info,
            ),
        ):
            return sched.get_execution_schedules(**kwargs)

    def test_own_schedules_always_visible_in_org_mode(self):
        infos = [self._graph_info(user_id="me", organization_id="", sid="mine")]
        result = self._run(infos, user_id="me", organization_id="org-1", team_ids=[])
        assert [r.schedule_id for r in result] == ["mine"]

    def test_org_home_schedule_visible_to_member(self):
        infos = [
            self._graph_info(
                user_id="teammate", organization_id="org-1", sid="org-home"
            )
        ]
        result = self._run(infos, user_id="me", organization_id="org-1", team_ids=[])
        assert [r.schedule_id for r in result] == ["org-home"]

    def test_team_schedule_only_visible_to_team_members(self):
        infos = [
            self._graph_info(
                user_id="teammate",
                organization_id="org-1",
                team_id="team-x",
                sid="team-x-job",
            )
        ]
        visible = self._run(
            infos, user_id="me", organization_id="org-1", team_ids=["team-x"]
        )
        hidden = self._run(
            infos, user_id="me", organization_id="org-1", team_ids=["team-y"]
        )
        assert [r.schedule_id for r in visible] == ["team-x-job"]
        assert hidden == []

    def test_other_org_schedule_hidden(self):
        infos = [
            self._graph_info(
                user_id="stranger", organization_id="org-OTHER", sid="foreign"
            )
        ]
        result = self._run(infos, user_id="me", organization_id="org-1", team_ids=[])
        assert result == []

    def test_no_org_mode_is_strict_ownership(self):
        infos = [
            self._graph_info(user_id="me", sid="mine"),
            self._graph_info(user_id="other", organization_id="org-1", sid="theirs"),
        ]
        result = self._run(infos, user_id="me")
        assert [r.schedule_id for r in result] == ["mine"]


# ---------------------------------------------------------------------------
# pause/resume_user_graph_schedules
# ---------------------------------------------------------------------------


def _graph_job_kwargs(
    user_id="u1", organization_id="", paused_reason=None, schedule_id="s1"
):
    return {
        "kind": "graph",
        "schedule_id": schedule_id,
        "user_id": user_id,
        "graph_id": "g1",
        "graph_version": 1,
        "cron": "* * * * *",
        "input_data": {},
        "input_credentials": {},
        "organization_id": organization_id,
        "paused_reason": paused_reason,
    }


class TestPauseResumeUserGraphSchedules:
    def _scheduler(self, jobs):
        sched = Scheduler.__new__(Scheduler)
        aps = MagicMock()
        aps.get_jobs.return_value = jobs
        object.__setattr__(sched, "scheduler", aps)
        return sched, aps

    def test_pause_stamps_reason_and_pauses_jobs(self):
        job = _mock_job(_graph_job_kwargs())
        sched, aps = self._scheduler([job])
        with patch.object(Scheduler, "_invalidate_jobs_cache", MagicMock()):
            count = sched.pause_user_graph_schedules("u1", "payment_lapsed")
        assert count == 1
        modify_kwargs = aps.modify_job.call_args.kwargs["kwargs"]
        assert modify_kwargs["paused_reason"] == "payment_lapsed"
        aps.pause_job.assert_called_once()

    def test_pause_skips_other_users_org_jobs_and_already_paused(self):
        jobs = [
            _mock_job(_graph_job_kwargs(user_id="other", schedule_id="s2")),
            _mock_job(_graph_job_kwargs(organization_id="org-1", schedule_id="s3")),
            _mock_job(
                _graph_job_kwargs(paused_reason="payment_lapsed", schedule_id="s4")
            ),
        ]
        sched, aps = self._scheduler(jobs)
        with patch.object(Scheduler, "_invalidate_jobs_cache", MagicMock()):
            count = sched.pause_user_graph_schedules("u1", "payment_lapsed")
        assert count == 0
        aps.pause_job.assert_not_called()

    def test_pause_includes_personal_org_tagged_jobs(self):
        jobs = [
            _mock_job(
                _graph_job_kwargs(organization_id="personal-org", schedule_id="s1")
            ),
            _mock_job(_graph_job_kwargs(organization_id="team-org", schedule_id="s2")),
        ]
        sched, aps = self._scheduler(jobs)
        with patch.object(Scheduler, "_invalidate_jobs_cache", MagicMock()):
            count = sched.pause_user_graph_schedules(
                "u1", "payment_lapsed", personal_org_id="personal-org"
            )
        assert count == 1
        assert aps.pause_job.call_count == 1

    def test_pause_rolls_back_reason_stamp_when_pause_job_fails(self):
        job = _mock_job(_graph_job_kwargs())
        sched, aps = self._scheduler([job])
        aps.pause_job.side_effect = RuntimeError("jobstore down")
        with patch.object(Scheduler, "_invalidate_jobs_cache", MagicMock()):
            count = sched.pause_user_graph_schedules("u1", "payment_lapsed")
        assert count == 0
        assert aps.modify_job.call_count == 2
        rollback_kwargs = aps.modify_job.call_args.kwargs["kwargs"]
        assert rollback_kwargs["paused_reason"] is None

    def test_resume_resumes_before_clearing_reason(self):
        job = _mock_job(
            _graph_job_kwargs(paused_reason="payment_lapsed", schedule_id="s1")
        )
        sched, aps = self._scheduler([job])
        with patch.object(Scheduler, "_invalidate_jobs_cache", MagicMock()):
            sched.resume_user_graph_schedules("u1", "payment_lapsed")
        calls = [c[0] for c in aps.method_calls]
        assert calls.index("resume_job") < calls.index("modify_job")

    def test_resume_failure_keeps_reason_for_retry(self):
        job = _mock_job(
            _graph_job_kwargs(paused_reason="payment_lapsed", schedule_id="s1")
        )
        sched, aps = self._scheduler([job])
        aps.resume_job.side_effect = RuntimeError("jobstore down")
        with patch.object(Scheduler, "_invalidate_jobs_cache", MagicMock()):
            count = sched.resume_user_graph_schedules("u1", "payment_lapsed")
        assert count == 0
        aps.modify_job.assert_not_called()

    def test_resume_only_matching_reason(self):
        jobs = [
            _mock_job(
                _graph_job_kwargs(paused_reason="payment_lapsed", schedule_id="s1")
            ),
            _mock_job(_graph_job_kwargs(schedule_id="s2")),
            _mock_job(
                _graph_job_kwargs(paused_reason="other_reason", schedule_id="s3")
            ),
        ]
        sched, aps = self._scheduler(jobs)
        with patch.object(Scheduler, "_invalidate_jobs_cache", MagicMock()):
            count = sched.resume_user_graph_schedules("u1", "payment_lapsed")
        assert count == 1
        modify_kwargs = aps.modify_job.call_args.kwargs["kwargs"]
        assert modify_kwargs["paused_reason"] is None
        aps.resume_job.assert_called_once()

    def test_paused_schedule_still_listed_with_is_paused(self):
        job = _mock_job(_graph_job_kwargs(paused_reason="payment_lapsed"))
        job.next_run_time = None
        fired_one_shot = _mock_job(
            {
                "kind": "copilot_turn",
                "schedule_id": "c1",
                "user_id": "u1",
                "session_id": "sess",
                "message": "m",
                "run_at": "2026-05-22T10:00:00+00:00",
            }
        )
        fired_one_shot.next_run_time = None
        sched = Scheduler.__new__(Scheduler)
        with patch.object(
            Scheduler, "_get_jobs_cached", lambda self: [job, fired_one_shot]
        ):
            result = sched.get_execution_schedules(user_id="u1")
        assert [r.schedule_id for r in result] == ["s1"]
        assert isinstance(result[0], GraphExecutionJobInfo)
        assert result[0].is_paused is True
        assert result[0].paused_reason == "payment_lapsed"
