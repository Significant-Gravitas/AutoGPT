"""Find meeting times in free/busy data. Pure functions: no Google calls."""

from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Iterable

from pydantic import BaseModel, Field

Interval = tuple[datetime, datetime]

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class MeetingSlot(BaseModel):
    """A suggested meeting time."""

    start: datetime = Field(description="When the meeting would start")
    end: datetime = Field(description="When the meeting would end")
    free_until: datetime = Field(
        description="When the free time containing this slot ends, so the meeting could start later or run longer"
    )


def find_meeting_slots(
    busy: Iterable[Interval],
    window_start: datetime,
    window_end: datetime,
    duration: timedelta,
    zone: tzinfo,
    *,
    day_start: time | None = None,
    day_end: time | None = None,
    include_weekends: bool = True,
    max_slots: int = 5,
    step: timedelta = timedelta(minutes=15),
) -> list[MeetingSlot]:
    """Suggest one meeting slot at the start of each free period.

    A free period is time inside the window, on an allowed day and between
    ``day_start`` and ``day_end`` local time in ``zone``, that no busy interval
    covers. Each slot starts on the period's first ``step`` boundary and must
    fit inside the period. All datetimes must be timezone-aware; the slots come
    back in ``zone``.
    """
    allowed = _allowed_periods(
        window_start, window_end, zone, day_start, day_end, include_weekends
    )
    slots: list[MeetingSlot] = []
    for period_start, period_end in _subtract(allowed, _merge(busy)):
        start = _round_up(period_start, step)
        if start + duration > period_end:
            continue
        slots.append(
            MeetingSlot(
                start=start.astimezone(zone),
                end=(start + duration).astimezone(zone),
                free_until=period_end.astimezone(zone),
            )
        )
        if len(slots) >= max_slots:
            break
    return slots


def _allowed_periods(
    window_start: datetime,
    window_end: datetime,
    zone: tzinfo,
    day_start: time | None,
    day_end: time | None,
    include_weekends: bool,
) -> list[Interval]:
    """The parts of the window on allowed days and within the allowed hours."""
    if day_start is None and day_end is None and include_weekends:
        return _merge([(window_start, window_end)])
    periods: list[Interval] = []
    day = window_start.astimezone(zone).date()
    last_day = window_end.astimezone(zone).date()
    while day <= last_day:
        if include_weekends or day.weekday() < 5:
            opens = _local(day, day_start or time.min, zone)
            closes = (
                _local(day, day_end, zone)
                if day_end is not None
                else _local(day + timedelta(days=1), time.min, zone)
            )
            periods.append((max(opens, window_start), min(closes, window_end)))
        day += timedelta(days=1)
    return _merge(periods)


def _local(day: date, moment: time, zone: tzinfo) -> datetime:
    return datetime.combine(day, moment, tzinfo=zone).astimezone(timezone.utc)


def _merge(intervals: Iterable[Interval]) -> list[Interval]:
    """Sort intervals in UTC, drop empty ones and join those that overlap or touch."""
    merged: list[Interval] = []
    for start, end in sorted(
        (s.astimezone(timezone.utc), e.astimezone(timezone.utc)) for s, e in intervals
    ):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract(periods: list[Interval], busy: list[Interval]) -> list[Interval]:
    """The parts of ``periods`` that no ``busy`` interval covers (both merged)."""
    free: list[Interval] = []
    for period_start, period_end in periods:
        cursor = period_start
        for busy_start, busy_end in busy:
            if busy_end <= cursor or busy_start >= period_end:
                continue
            if busy_start > cursor:
                free.append((cursor, busy_start))
            cursor = busy_end
            if cursor >= period_end:
                break
        if cursor < period_end:
            free.append((cursor, period_end))
    return free


def _round_up(moment: datetime, step: timedelta) -> datetime:
    """Round up to a multiple of ``step`` since the epoch.

    Every current UTC offset is a whole number of quarter hours, so 15-minute
    steps land on local quarter hours in any time zone.
    """
    remainder = (moment - _EPOCH) % step
    return moment + (step - remainder) if remainder else moment
