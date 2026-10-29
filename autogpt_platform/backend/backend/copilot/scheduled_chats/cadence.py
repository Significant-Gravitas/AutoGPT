"""How often a scheduled chat runs, and the cadences that run it less often
(SECRT-2879).

A chat skipped because its owner's usage limit ran out has a way out besides
upgrading: run less often, so its runs fit the allowance. This counts how often
the schedule fires in a week and offers the rungs of a short ladder below that,
at a time the schedule already uses, and never more often on any day than it
runs now.

Crons are stored as Unix cron, and the scheduler hands them to APScheduler with
their weekdays renumbered. Two things still differ from Unix cron, and croniter
is told both, so it counts the fires the scheduler makes: APScheduler fires when
the day of the month and the day of the week both match, not either, and it
counts a step over every weekday from Monday. The cadences offered name their
weekdays (``mon-fri``), which both read the same way.

Fires are counted on the zone's clock as it stands now, at a fixed offset, so
the week the clocks change has no hour twice or missing: an hourly schedule
still runs 168 times a week, at the minute it always does.
"""

from datetime import datetime, timedelta
from datetime import timezone as fixed_offset
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from pydantic import BaseModel

# Every rung of the ladder fires well under this many times a week, so a cron
# counted this far runs more often than all of them: stop there.
_COUNT_CAP = 1000
# The busiest rung runs this often a day: a day counted this far has room for
# every rung.
_BUSIEST_DAY = 96
_WEEK = timedelta(days=7)
_DAY = timedelta(days=1)
_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_DAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
# A schedule with a handful of times a day keeps one of them; anything busier
# gets a once-a-day time at the start of a working day.
_FEW_TIMES = 4
_MORNING_HOUR = 9
# Where a day's times start: a 06:00 run comes before a 00:30 one.
_DAY_STARTS = (6, 0)
_HOURLY = 7 * 24


class Cadence(BaseModel):
    cron: str
    label: str
    # Monday first.
    runs_per_day: list[int]

    @property
    def runs_per_week(self) -> int:
        return sum(self.runs_per_day)


def less_often(crons: list[str], timezone: str, now: datetime) -> list[Cadence]:
    """The cadences that run the schedule less often, most often first.

    Empty when there is nothing to offer: a one-off, a schedule that fires
    once a week or less, or crons that do not parse.
    """
    fires = _fires(crons, timezone, now)
    if not fires:
        return []
    per_day = _runs_per_day(crons, timezone, now)
    hour, minute = _anchor_time(fires)
    day = _anchor_day(per_day)
    at = f"{hour:02d}:{minute:02d}"
    ladder = [
        Cadence(cron="*/15 * * * *", label="Every 15 minutes", runs_per_day=_on(96)),
        Cadence(cron=f"{minute} * * * *", label="Every hour", runs_per_day=_on(24)),
        Cadence(cron=f"{minute} */4 * * *", label="Every 4 hours", runs_per_day=_on(6)),
        Cadence(
            cron=f"{minute} {hour} * * *",
            label=f"Every day at {at}",
            runs_per_day=_on(1),
        ),
        Cadence(
            cron=f"{minute} {hour} * * mon-fri",
            label=f"Weekdays at {at}",
            runs_per_day=_on(1, days=range(5)),
        ),
        Cadence(
            cron=f"{minute} {hour} * * {_DAYS[day]}",
            label=f"Every {_DAY_NAMES[day]} at {at}",
            runs_per_day=_on(1, days=[day]),
        ),
    ]
    # Less often in all, and on no day more often than now: a rung that ran
    # on a day the schedule leaves free would spend that day's allowance.
    return [
        rung
        for rung in ladder
        if rung.runs_per_week < len(fires)
        and all(runs <= limit for runs, limit in zip(rung.runs_per_day, per_day))
    ]


def runs_more_than_weekly(crons: list[str], timezone: str, now: datetime) -> bool:
    """Whether ``less_often`` has something to offer, without building the
    ladder: a schedule that runs more than once a week can run once a week on
    its busiest day."""
    return len(_fires(crons, timezone, now, limit=2)) >= 2


def describe(crons: list[str], timezone: str, now: datetime) -> str | None:
    """How often the schedule runs, in words: "Every minute", "Every day at
    08:30", "Twice a day". None for one that runs less than twice a week and
    not on a weekly beat, such as a monthly one.

    Each cron is read for at most ``_COUNT_CAP`` runs. A beat found within
    them stands for the week only if the schedule runs every day, and a count
    that reaches the cap says so."""
    if len(crons) == 1:
        # Two weeks, so every gap in a week's runs falls between two of them
        # whatever day it is read on: weekdays read on a Saturday do not pass
        # for daily. A weekly beat needs weeks to show.
        beat = _fires(crons, timezone, now, span=2 * _WEEK)
        if len(beat) < 3:
            beat = _fires(crons, timezone, now, span=4 * _WEEK, limit=3)
        gaps = {later - earlier for earlier, later in zip(beat, beat[1:])}
        if len(beat) >= 3 and len(gaps) == 1:
            every = _every(gaps.pop(), beat[0])
            # Cut short at the cap, every minute on Saturdays beats like every
            # minute: the beat holds only if it runs every day.
            whole = len(beat) < _COUNT_CAP or all(_runs_per_day(crons, timezone, now))
            if every is not None and whole:
                return every
    runs = len(_fires(crons, timezone, now))
    if runs < 2:
        return None
    if runs >= _COUNT_CAP:
        return f"At least {_COUNT_CAP} times a week"
    # A day's runs only when every day has them: seven on Mondays is not
    # once a day.
    if runs % 7 == 0 and len(set(_runs_per_day(crons, timezone, now))) == 1:
        per_day = runs // 7
        if per_day <= 2:
            return "Once a day" if per_day == 1 else "Twice a day"
        return f"{per_day} times a day"
    return f"{runs} times a week"


def _fires(
    crons: list[str],
    timezone: str,
    now: datetime,
    *,
    span: timedelta = _WEEK,
    limit: int = _COUNT_CAP,
) -> list[datetime]:
    """Each cron's fires from *now* until *span* after it, at most *limit* of
    each, soonest first, in *timezone*. One due right now counts.

    Whole seconds, croniter's own step: within the second a run fires, *now*
    would otherwise count that run and the one a week on, eight days' worth.
    The cadences offered when the dialog opened must still be offered when
    the owner saves.
    """
    start = _on_its_clock(now, timezone).replace(microsecond=0)
    return _fires_from(crons, start, span=span, limit=limit)


def _fires_from(
    crons: list[str], start: datetime, *, span: timedelta, limit: int
) -> list[datetime]:
    end = start + span
    fires: list[datetime] = []
    for cron in crons:
        try:
            # croniter starts after the time it is given; a cron second is
            # the smallest step it takes.
            schedule = croniter(
                _as_scheduled(cron), start - timedelta(seconds=1), day_or=False
            )
            for _ in range(limit):
                fire = schedule.get_next(datetime)
                if fire >= end:
                    break
                fires.append(fire)
        except (ValueError, KeyError):
            # Unreadable, or a date that never comes (31 February), which
            # croniter only finds out once it steps.
            return []
    return sorted(fires)


def _as_scheduled(cron: str) -> str:
    """*cron* with a step over every weekday spelled out as the days APScheduler
    fires it on, counting from Monday: ``*/2`` is Monday, Wednesday, Friday and
    Sunday there, where Unix cron means Sunday, Tuesday, Thursday and Saturday.
    The scheduler renumbers every other weekday form itself."""
    parts = cron.split()
    if len(parts) != 5:
        return cron
    parts[4] = ",".join(_scheduled_days(token) for token in parts[4].split(","))
    return " ".join(parts)


def _scheduled_days(token: str) -> str:
    body, _, step = token.partition("/")
    if body not in ("*", "?") or not step.isdigit():
        return token
    # APScheduler's Monday is 0; Unix cron's Sunday is.
    return ",".join(str((day + 1) % 7) for day in range(0, 7, int(step)))


def _anchor_time(fires: list[datetime]) -> tuple[int, int]:
    times = sorted({(fire.hour, fire.minute) for fire in fires})
    if len(times) <= _FEW_TIMES:
        # Its first time from the morning on.
        daytime = [time for time in times if time >= _DAY_STARTS]
        return (daytime or times)[0]
    # A routine that runs at most hourly keeps its minute, which was picked to
    # spread the load; one that runs more often has no minute of its own. The
    # least of its minutes, so the next run does not decide.
    minute = min(fire.minute for fire in fires) if len(fires) <= _HOURLY else 0
    return _MORNING_HOUR, minute


def _anchor_day(runs_per_day: list[int]) -> int:
    """The weekday it runs on most, as ``_runs_per_day`` counts; Monday first
    when that is a tie."""
    return min(range(7), key=lambda day: (-runs_per_day[day], day))


def _runs_per_day(crons: list[str], timezone: str, now: datetime) -> list[int]:
    """How often it runs on each of the coming seven days, Monday first, on
    the zone's clock as it stands now. Each day is counted as far as the
    busiest rung runs in a day: past that, no rung runs more that day."""
    midnight = _on_its_clock(now, timezone).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    runs = [0] * 7
    for offset in range(7):
        day = midnight + offset * _DAY
        fires = _fires_from(crons, day, span=_DAY, limit=_BUSIEST_DAY)
        runs[day.weekday()] = min(len(fires), _BUSIEST_DAY)
    return runs


def _on(runs: int, *, days: range | list[int] = range(7)) -> list[int]:
    """A rung's runs on each weekday, Monday first."""
    return [runs if day in days else 0 for day in range(7)]


def _on_its_clock(now: datetime, timezone: str) -> datetime:
    """*now* on the zone's clock as it stands, at that fixed offset."""
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        zone = ZoneInfo("UTC")
    offset = now.astimezone(zone).utcoffset() or timedelta(0)
    return now.astimezone(fixed_offset(offset))


def _every(gap: timedelta, first: datetime) -> str | None:
    minutes = int(gap.total_seconds() // 60)
    at = f"{first.hour:02d}:{first.minute:02d}"
    if minutes == 1:
        return "Every minute"
    if minutes < 60:
        return f"Every {minutes} minutes"
    if minutes == 60:
        return "Every hour"
    if minutes < 24 * 60 and minutes % 60 == 0:
        return f"Every {minutes // 60} hours"
    if minutes == 24 * 60:
        return f"Every day at {at}"
    if minutes == 7 * 24 * 60:
        return f"Every {_DAY_NAMES[first.weekday()]} at {at}"
    return None
