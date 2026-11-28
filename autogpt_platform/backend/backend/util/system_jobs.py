"""Periodic jobs a feature owns. The scheduler registers every declared job at
each start; a feature declares its own beside its code."""

from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel, ConfigDict

from backend.util.settings import Config


class Jobstores(Enum):
    EXECUTION = "execution"
    BATCHED_NOTIFICATIONS = "batched_notifications"
    WEEKLY_NOTIFICATIONS = "weekly_notifications"
    MEMORY_CRON_RECONCILE = "memory_cron_reconcile"


class SystemJob(BaseModel):
    """One periodic job. A persistent jobstore saves it under ``id`` and the
    import path of ``func``, so changing either changes a stored row."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id: str
    func: Callable[[], object]
    trigger: IntervalTrigger | CronTrigger
    jobstore: Jobstores = Jobstores.EXECUTION
    enabled: bool = True
    # None: the scheduler's default, which never skips a run.
    max_instances: int | None = None
    # None: one interval after start, or the cron's next time.
    first_run: datetime | None = None
    # Registered once the scheduler runs, leaving a stored job whose trigger is
    # unchanged alone: replacing it moves its next run, so a restart between a
    # due time and the run would skip that run.
    after_start: bool = False
    keep_if_unchanged: bool = False


# A feature's declarations, built from the scheduler's config at each start.
SystemJobSource = Callable[[Config], list[SystemJob]]

_loop_runner: Callable[..., Any] | None = None


def run_on_scheduler_loop(coro: Coroutine[Any, Any, Any], timeout: float | None) -> Any:
    """Run a system job's coroutine on the scheduler's event loop and wait for
    it, up to *timeout* seconds."""
    if _loop_runner is None:
        coro.close()
        raise RuntimeError("The scheduler has not started its event loop")
    return _loop_runner(coro, timeout=timeout)


def lend_event_loop(runner: Callable[..., Any]) -> None:
    """How the scheduler, once its loop runs, gives system jobs that loop
    without them importing the scheduler."""
    global _loop_runner
    _loop_runner = runner


def every(**interval: int) -> IntervalTrigger:
    """An interval in UTC, the timezone the scheduler gives one it builds."""
    return IntervalTrigger(timezone=timezone.utc, **interval)
