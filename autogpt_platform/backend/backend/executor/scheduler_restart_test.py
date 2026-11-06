"""A scheduler restart keeps the stored next run of every interval system job,
unless its interval changed. Two real starts of ``Scheduler.run_service`` share
one SQLite jobstore; APScheduler starts paused, so no job runs."""

import logging
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from backend.executor import scheduler as scheduler_module
from backend.executor.scheduler import Jobstores, Scheduler

_PERSISTENT = (Jobstores.EXECUTION.value, Jobstores.BATCHED_NOTIFICATIONS.value)
# Due at every start on purpose: the start-up embedding backfill.
_DUE_AT_START = "ensure_embeddings_coverage"


def test_a_restart_keeps_every_interval_jobs_next_run(tmp_path):
    jobstore = f"sqlite:///{tmp_path}/jobs.sqlite"

    first = _interval_jobs_after_a_start(jobstore)
    second = _interval_jobs_after_a_start(jobstore)

    kept = {
        job_id: run for job_id, (run, _) in first.items() if job_id != _DUE_AT_START
    }
    assert "cleanup_auth_ip_events" in kept and "dream_pass_reaper" in kept
    assert {job_id: second[job_id][0] for job_id in kept} == kept


def test_a_changed_interval_takes_effect_on_restart(monkeypatch, tmp_path):
    jobstore = f"sqlite:///{tmp_path}/jobs.sqlite"
    config = scheduler_module.config
    monkeypatch.setattr(config, "block_error_rate_check_interval_secs", 24 * 3600)
    first_run, _ = _interval_jobs_after_a_start(jobstore)["report_block_error_rates"]

    monkeypatch.setattr(config, "block_error_rate_check_interval_secs", 12 * 3600)
    second_run, interval = _interval_jobs_after_a_start(jobstore)[
        "report_block_error_rates"
    ]

    assert interval == timedelta(hours=12)
    assert second_run < first_run


def test_an_unreadable_stored_job_is_reported_but_a_first_start_is_not(
    caplog, tmp_path
):
    jobstore = f"sqlite:///{tmp_path}/jobs.sqlite"
    caplog.set_level(logging.WARNING)
    _interval_jobs_after_a_start(jobstore)
    assert _unreadable_warnings(caplog) == []

    db = sqlite3.connect(tmp_path / "jobs.sqlite")
    db.execute(
        "UPDATE apscheduler_jobs SET job_state = ? WHERE id = ?",
        (b"not a pickle", "report_block_error_rates"),
    )
    db.commit()
    db.close()
    _interval_jobs_after_a_start(jobstore)

    warnings = _unreadable_warnings(caplog)
    assert len(warnings) == 1 and "report_block_error_rates" in warnings[0]


def _interval_jobs_after_a_start(
    jobstore: str,
) -> dict[str, tuple[datetime | None, timedelta]]:
    real_start = BackgroundScheduler.start

    def start_paused(self, *args, **kwargs):
        real_start(self, paused=True)

    service = Scheduler(register_system_tasks=True)
    with (
        patch.object(Scheduler, "_jobstore_database", return_value=(None, jobstore)),
        patch.object(Scheduler, "_start_schedule_index_backfill"),
        patch.object(BackgroundScheduler, "start", start_paused),
        patch.object(scheduler_module, "load_dotenv"),
        # Blocks forever serving RPC; the start-up work before it is under test.
        patch("backend.util.service.AppService.run_service"),
    ):
        service.run_service()
    try:
        return {
            job.id: (job.next_run_time, job.trigger.interval)
            for alias in _PERSISTENT
            for job in service.scheduler.get_jobs(jobstore=alias)
            if isinstance(job.trigger, IntervalTrigger)
        }
    finally:
        service.cleanup()


def _unreadable_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "unreadable" in r.getMessage()
    ]
