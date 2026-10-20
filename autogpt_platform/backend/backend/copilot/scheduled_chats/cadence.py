"""How often a scheduled chat runs, and the cadences that run it less often
(SECRT-2879).

A chat skipped because its owner's usage limit ran out has a way out besides
upgrading: run less often, so its runs fit the allowance. This counts how often
the schedule fires in a week and offers the rungs of a short ladder below that,
at a time the schedule already uses.

Crons are read as Unix cron, the way they are stored: the scheduler translates
the day-of-week field when it builds an APScheduler trigger, so croniter sees
the fire times the scheduler does. The cadences offered name their weekdays
(``mon-fri``), which both read the same way.

Fires are counted on the zone's clock as it stands now, at a fixed offset, so
the week the clocks change has no hour twice or missing: an hourly schedule
still runs 168 times a week, at the minute it always does.
"""

from collections import Counter
from datetime import datetime, timedelta
from datetime import timezone as fixed_offset
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from pydantic import BaseModel

# Every rung of the ladder fires well under this many times a week, so a cron
# counted this far runs more often than all of them: stop there.
_COUNT_CAP = 1000
_WEEK = timedelta(days=7)
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
    runs_per_week: int


def less_often(crons: list[str], timezone: str, now: datetime) -> list[Cadence]:
    """The cadences that run the schedule less often, most often first.

    Empty when there is nothing to offer: a one-off, a schedule that fires
    once a week or less, or crons that do not parse.
    """
    fires = _fires(crons, timezone, now)
    if not fires:
        return []
    hour, minute = _anchor_time(fires)
    day = _anchor_day(fires)
    at = f"{hour:02d}:{minute:02d}"
    ladder = [
        Cadence(cron="*/15 * * * *", label="Every 15 minutes", runs_per_week=672),
        Cadence(cron=f"{minute} * * * *", label="Every hour", runs_per_week=168),
        Cadence(cron=f"{minute} */4 * * *", label="Every 4 hours", runs_per_week=42),
        Cadence(
            cron=f"{minute} {hour} * * *", label=f"Every day at {at}", runs_per_week=7
        ),
        Cadence(
            cron=f"{minute} {hour} * * mon-fri",
            label=f"Weekdays at {at}",
            runs_per_week=5,
        ),
        Cadence(
            cron=f"{minute} {hour} * * {_DAYS[day]}",
            label=f"Every {_DAY_NAMES[day]} at {at}",
            runs_per_week=1,
        ),
    ]
    return [rung for rung in ladder if rung.runs_per_week < len(fires)]


def describe(crons: list[str], timezone: str, now: datetime) -> str | None:
    """How often the schedule runs, in words: "Every minute", "Every day at
    08:30", "Twice a day". None for one that runs less than twice a week and
    not on a weekly beat, such as a monthly one.

    Each cron is read for at most ``_COUNT_CAP`` runs: a beat found within
    them stands for the week, and a count that reaches the cap says so."""
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
            if every is not None:
                return every
    runs = len(_fires(crons, timezone, now))
    if runs < 2:
        return None
    if runs >= _COUNT_CAP:
        return f"At least {_COUNT_CAP} times a week"
    if runs % 7 == 0:
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
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        zone = ZoneInfo("UTC")
    offset = now.astimezone(zone).utcoffset() or timedelta(0)
    start = now.astimezone(fixed_offset(offset)).replace(microsecond=0)
    end = start + span
    fires: list[datetime] = []
    for cron in crons:
        try:
            # croniter starts after the time it is given; a cron second is
            # the smallest step it takes.
            schedule = croniter(cron, start - timedelta(seconds=1))
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


def _anchor_day(fires: list[datetime]) -> int:
    """The weekday it runs on most; Monday first when that is a tie, or when
    it runs more than hourly and the count stopped short of a week."""
    if len(fires) > _HOURLY:
        return 0
    counts = Counter(fire.weekday() for fire in fires)
    return min(counts, key=lambda day: (-counts[day], day))


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
