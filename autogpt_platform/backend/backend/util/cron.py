"""Limits every user-supplied cron meets before it reaches APScheduler, and the
day-of-week renumbering it needs to fire on the days Unix cron means."""

from collections.abc import Callable
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
    Expanding the field to its days before renumbering keeps all of them.
    """
    return _spell_out_day_of_week(cron, lambda day: (day - 1) % 7)


def spell_out_cron_day_of_week(cron: str) -> str:
    """*cron* with its day-of-week field as the Unix days the scheduler fires
    it on, for another cron reader to count: croniter reads a few rare forms
    its own way (``1-1`` as every day, ``6/2`` as ``*/2``)."""
    return _spell_out_day_of_week(cron, lambda day: day)


def _spell_out_day_of_week(cron: str, number: Callable[[int], int]) -> str:
    """The field as one list of *number*-ed days. ``*``, ``?`` and a cron
    without five fields are left for APScheduler to judge."""
    parts = cron.split()
    if len(parts) != 5 or parts[4] in ("*", "?"):
        return cron
    days = {number(day) for token in parts[4].split(",") for day in _unix_days(token)}
    parts[4] = ",".join(str(day) for day in sorted(days))
    return " ".join(parts)


def _unix_days(token: str) -> list[int]:
    """The Unix days (0=Sun..6=Sat) one token of the field covers. Refuses
    what cron refuses (a slash without a step, a zero step, a day above 7)
    and anything else it can't read, rather than coerce it into days nobody
    asked for: APScheduler alone reads ``sun-`` as Sunday."""
    body, slash, step_text = token.partition("/")
    if slash and not (_is_number(step_text) and int(step_text) > 0):
        raise ValueError(
            f"Day-of-week step in {token!r} must be a whole number above 0"
        )
    step = int(step_text) if slash else 1

    if body in ("*", "?"):
        days = list(range(7))
    elif "-" in body:
        first_text, last_text = body.split("-", 1)
        first, last = _unix_day(first_text, token), _unix_day(last_text, token)
        if first <= last:
            days = list(range(first, last + 1))
        else:
            # A range past Saturday (``5-2``) carries on from Sunday.
            days = [*range(first, 7), *range(last + 1)]
    else:
        first = _unix_day(body, token)
        # ``1/2`` steps from its day to Saturday; ``7/2`` is Sunday alone.
        days = list(range(first, max(first, 6) + 1)) if slash else [first]
    return sorted({day % 7 for day in days[::step]})


def _unix_day(value: str, token: str) -> int:
    if not _is_number(value):
        if value.lower() not in _UNIX_DAY_NAMES:
            raise ValueError(f"Unrecognized day of the week {value!r} in {token!r}")
        return _UNIX_DAY_NAMES[value.lower()]
    day = int(value)
    if day > 7:
        raise ValueError(f"Day-of-week {day} out of range 0-7 in {token!r}")
    return day


def _is_number(text: str) -> bool:
    # ASCII only: str.isdigit() also takes other scripts' digits, such as "٣".
    return text.isascii() and text.isdigit()
