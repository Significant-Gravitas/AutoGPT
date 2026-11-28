"""The dream pass system jobs, over every user's passes: the reaper and the
retention sweep. ``backend/scheduler.py`` lists them for the scheduler."""

from apscheduler.triggers.cron import CronTrigger

from backend.util.settings import Config
from backend.util.system_jobs import SystemJob, every, run_on_scheduler_loop

from .reaper import REAPER_BUDGET_SECONDS, REAPER_INTERVAL_MINUTES, reap_expired_passes
from .retention import RETENTION_BUDGET_SECONDS, delete_expired_records

config = Config()


def dream_system_jobs(_: Config) -> list[SystemJob]:
    return [
        SystemJob(
            id="dream_pass_reaper",
            func=execute_dream_pass_reaper,
            trigger=every(minutes=REAPER_INTERVAL_MINUTES),
            max_instances=1,
        ),
        SystemJob(
            id="dream_pass_retention",
            func=execute_dream_pass_retention,
            trigger=CronTrigger(day_of_week="sun", hour=5, minute=30, timezone="UTC"),
            max_instances=1,
        ),
    ]


def execute_dream_pass_reaper() -> None:
    """Close the dream passes of every user that outlived their lease
    (``reaper.py``). The run bounds itself, and that budget is what stops it:
    the wait's timeout frees the job thread should the event loop stall, but
    does not cancel the run."""
    run_on_scheduler_loop(reap_expired_passes(), timeout=REAPER_BUDGET_SECONDS + 30)


def execute_dream_pass_retention() -> None:
    """Delete the closed dream passes past their retention (``retention.py``),
    bounded the same way."""
    run_on_scheduler_loop(
        delete_expired_records(config.dream_pass_retention_days),
        timeout=RETENTION_BUDGET_SECONDS + 60,
    )
