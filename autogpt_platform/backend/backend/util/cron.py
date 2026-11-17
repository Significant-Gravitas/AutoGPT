"""Limits every user-supplied cron meets before it reaches APScheduler, and the
day-of-week renumbering it needs to fire on the days Unix cron means."""

from datetime import datetime, timedelta, tzinfo

from apscheduler.triggers.cron import CronTrigger

MAX_CRON_LENGTH = 200
# The rarest cron that fires at all (Feb 29 on one weekday) can go 40 years between runs.
_MAX_CRON_FIRE_GAP = timedelta(days=50 * 366)
# Unix cron counts its weekdays from Sunday (0, and 7 again); APScheduler from Monday.
_UNIX_DAY_NAMES = {
    name: day
    for day, name in enumerate(["sun", "mon", "tue", "wed", "thu", "fri", "sat"])
}


def validate_cron(cron: str, timezone: str | tzinfo) -> CronTrigger:
    """Parse Unix *cron* into the trigger the scheduler runs, refusing it
    unless it is short enough and fires."""
    check_cron_length(cron)
    trigger = CronTrigger.from_crontab(
        normalize_cron_day_of_week(cron), timezone=timezone
    )
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


def normalize_cron_day_of_week(cron: str) -> str:
    """*cron* with its day-of-week field rewritten as the APScheduler day
    numbers (0=Mon..6=Sun) of the days Unix cron (0 or 7=Sun..6=Sat) means.

    ``CronTrigger.from_crontab`` reads the field in its own numbering, so as
    given ``1`` fires on Tuesday, ``0-7`` only on Sunday, ``*/2`` steps from
    Monday, ``sun-thu`` is refused and ``mon-fri/2`` loses its step (#15276).
    Expanding each token to its days before renumbering keeps all of them.
    ``*``, ``?``, forms this doesn't read and a cron without five fields are
    left for APScheduler to judge.
    """
    parts = cron.split()
    if len(parts) != 5 or parts[4] in ("*", "?"):
        return cron
    parts[4] = ",".join(_apscheduler_days(token) for token in parts[4].split(","))
    return " ".join(parts)


def _apscheduler_days(token: str) -> str:
    days = _unix_days(token)
    if days is None:
        return token
    return ",".join(str(day) for day in sorted({(day - 1) % 7 for day in days}))


def _unix_days(token: str) -> list[int] | None:
    """The Unix days (0=Sun..6=Sat) one token of the field covers, or ``None``
    for a form this doesn't read. Refuses what cron refuses (a slash without a
    step, a zero step, a day above 7) instead of coercing it into days nobody
    asked for."""
    body, slash, step_text = token.partition("/")
    if slash and not (step_text.isdigit() and int(step_text) > 0):
        raise ValueError(
            f"Day-of-week step in {token!r} must be a whole number above 0"
        )
    step = int(step_text) if slash else 1

    if body in ("*", "?"):
        days = list(range(7))
    elif "-" in body:
        first_text, last_text = body.split("-", 1)
        first, last = _unix_day(first_text, token), _unix_day(last_text, token)
        if first is None or last is None:
            return None
        if first <= last:
            days = list(range(first, last + 1))
        else:
            # A range past Saturday (``5-2``) carries on from Sunday.
            days = [*range(first, 7), *range(last + 1)]
    else:
        first = _unix_day(body, token)
        if first is None:
            return None
        # ``1/2`` steps from its day to Saturday, as croniter reads it.
        days = list(range(first, max(first, 6) + 1)) if slash else [first]
    return sorted({day % 7 for day in days[::step]})


def _unix_day(value: str, token: str) -> int | None:
    if not value.isdigit():
        return _UNIX_DAY_NAMES.get(value.lower())
    day = int(value)
    if day > 7:
        raise ValueError(f"Day-of-week {day} out of range 0-7 in {token!r}")
    return day
