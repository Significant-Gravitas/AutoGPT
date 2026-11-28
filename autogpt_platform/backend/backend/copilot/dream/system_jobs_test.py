from datetime import timedelta
from unittest.mock import MagicMock, patch

from backend.copilot.dream import system_jobs
from backend.copilot.dream.reaper import REAPER_BUDGET_SECONDS
from backend.copilot.dream.retention import RETENTION_BUDGET_SECONDS
from backend.copilot.dream.system_jobs import (
    dream_system_jobs,
    execute_dream_pass_reaper,
    execute_dream_pass_retention,
)
from backend.util.settings import Config
from backend.util.system_jobs import Jobstores


def test_both_are_registered_once_for_every_user_not_per_user():
    jobs = {job.id: job for job in dream_system_jobs(Config())}

    assert set(jobs) == {"dream_pass_reaper", "dream_pass_retention"}
    reaper = jobs["dream_pass_reaper"]
    assert reaper.func is execute_dream_pass_reaper
    assert reaper.trigger.interval == timedelta(minutes=10)
    retention = jobs["dream_pass_retention"]
    assert retention.func is execute_dream_pass_retention
    trigger = repr(retention.trigger)
    assert "day_of_week='sun'" in trigger and "hour='5'" in trigger
    for job in jobs.values():
        assert job.max_instances == 1
        assert job.jobstore is Jobstores.EXECUTION


def test_the_reaper_runs_on_the_scheduler_loop_bounded_by_its_budget():
    sentinel = object()
    with (
        patch.object(system_jobs, "run_on_scheduler_loop") as run,
        patch.object(
            system_jobs, "reap_expired_passes", MagicMock(return_value=sentinel)
        ),
    ):
        execute_dream_pass_reaper()

    run.assert_called_once_with(sentinel, timeout=REAPER_BUDGET_SECONDS + 30)


def test_retention_keeps_the_configured_number_of_days(monkeypatch):
    monkeypatch.setattr(system_jobs.config, "dream_pass_retention_days", 30)
    sentinel = object()
    delete = MagicMock(return_value=sentinel)
    with (
        patch.object(system_jobs, "run_on_scheduler_loop") as run,
        patch.object(system_jobs, "delete_expired_records", delete),
    ):
        execute_dream_pass_retention()

    delete.assert_called_once_with(30)
    run.assert_called_once_with(sentinel, timeout=RETENTION_BUDGET_SECONDS + 60)
