"""Limits every user-supplied cron meets before it reaches APScheduler."""

from datetime import datetime, timedelta, tzinfo

from apscheduler.triggers.cron import CronTrigger

MAX_CRON_LENGTH = 200
# The rarest cron that fires at all (Feb 29 on one weekday) can go 40 years between runs.
_MAX_CRON_FIRE_GAP = timedelta(days=50 * 366)


def validate_cron(cron: str, timezone: str | tzinfo) -> CronTrigger:
    """Parse *cron* as given, refusing it unless it is short enough and fires."""
    check_cron_length(cron)
    trigger = CronTrigger.from_crontab(cron, timezone=timezone)
    check_cron_fires(trigger, cron)
    return trigger


def check_cron_length(cron: str) -> None:
    if len(cron) > MAX_CRON_LENGTH:
        raise ValueError(
            f"Cron expression is too long (max {MAX_CRON_LENGTH} characters)"
        )


def check_cron_fires(trigger: CronTrigger, cron: str) -> None:
    # A cron with no fire time would be stored and never run.
    if not cron_fires(trigger):
        raise ValueError(f"Cron expression {cron!r} never fires")


def cron_fires(trigger: CronTrigger) -> bool:
    """Whether *trigger* has a fire time within the window any firing cron needs."""
    now = datetime.now(trigger.timezone)
    trigger.end_date = now + _MAX_CRON_FIRE_GAP
    try:
        return trigger.get_next_fire_time(None, now) is not None
    finally:
        trigger.end_date = None
