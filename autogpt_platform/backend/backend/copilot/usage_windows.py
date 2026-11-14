"""Where a moment falls in a customer's billing clock: pure integer maths.

A billing period (Stripe's, to the second) is cut into billing months, each
month into four weekly windows, each window into 24-hour days from its start.
Everything is Unix seconds in UTC and every interval is half-open: at
``now == end`` the next one has begun. Nothing here reads a clock or a time
zone, so daylight saving and local time cannot move a reset.

- A period of 28 to 31 days is one billing month.
- A longer one (a yearly plan) is split into calendar months on the anchor's
  day and time, clamped to each month's last day as Stripe clamps monthly
  billing (31 Jan, 28 Feb, 31 Mar).
- Anything else (a trial, a period staff cut short) gets plain 7-day windows
  from its start, the last one shorter and its weekly limit scaled to match.
"""

import calendar
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

DAY = 86_400
WEEK = 7 * DAY
MIN_MONTH = 28 * DAY
MAX_MONTH = 31 * DAY
AVERAGE_MONTH = 2_629_746
MAX_SUB_PERIODS = 60

# How long after a period's end the first re-check of a held window is
# reported, doubling from there up to the cap.
HOLD_FIRST_STEP = 300
HOLD_MAX_STEP = 6 * 3600


class UsageWindows(BaseModel):
    """The weekly window and the day inside it that a moment falls in."""

    model_config = ConfigDict(frozen=True)

    week_start: int
    # When the weekly allowance is reported to come back.
    week_end: int
    # The window's own end, which sizes its pool: ``week_end`` unless held.
    pool_end: int
    day_start: int
    day_end: int
    # 7-day windows of a period that isn't whole months: a short last window
    # gets a proportionally smaller weekly limit.
    irregular: bool = False
    # The stored period has ended and Stripe hasn't confirmed the next one:
    # the last window stays open, with no new weekly allowance.
    held: bool = False


def windows_for_period(
    period_start: int, period_end: int, anchor: int | None, now: int
) -> UsageWindows:
    """The windows of a billing period at ``now``; held when it has ended."""
    if now >= period_end:
        last = windows_for_period(period_start, period_end, anchor, period_end - 1)
        return _held(last, period_end, now)
    moment = max(now, period_start)
    month = billing_month(period_start, period_end, anchor, moment)
    if month is None:
        return _seven_day_windows(period_start, period_end, moment)
    return _quarter_windows(month[0], month[1], moment)


def windows_from_anchor(anchor: int, now: int) -> UsageWindows:
    """Monthly billing months on the anchor's day and time, for a plan with no
    Stripe period."""
    then, moment = _utc(anchor), _utc(now)
    index = (moment.year - then.year) * 12 + moment.month - then.month
    if month_after(anchor, index, then.day) > now:
        index -= 1
    start = month_after(anchor, index, then.day)
    end = month_after(anchor, index + 1, then.day)
    return _quarter_windows(start, end, now)


def billing_month(
    period_start: int, period_end: int, anchor: int | None, now: int
) -> tuple[int, int] | None:
    """The billing month of the period that holds ``now``, or None when the
    period isn't a whole number of months."""
    length = period_end - period_start
    if MIN_MONTH <= length <= MAX_MONTH:
        return period_start, period_end
    count = round(length / AVERAGE_MONTH)
    if not 2 <= count <= MAX_SUB_PERIODS:
        return None
    bounds = _calendar_split(period_start, period_end, anchor, count)
    if not _whole_months(bounds):
        bounds = [period_start + (i * length) // count for i in range(count + 1)]
    if not _whole_months(bounds):
        return None
    index = max(i for i in range(count) if bounds[i] <= now)
    return bounds[index], bounds[index + 1]


def weekly_bounds(month_start: int, month_end: int) -> list[int]:
    """The five edges of a billing month's four weekly windows."""
    length = month_end - month_start
    return [month_start + (k * length) // 4 for k in range(5)]


def month_after(moment: int, months: int, day: int) -> int:
    """Same time of day ``months`` calendar months from ``moment``, on ``day``
    or that month's last day when it is shorter."""
    then = _utc(moment)
    year, month0 = divmod(then.year * 12 + then.month - 1 + months, 12)
    last_day = calendar.monthrange(year, month0 + 1)[1]
    return calendar.timegm(
        (year, month0 + 1, min(day, last_day), then.hour, then.minute, then.second)
    )


def day_lengths(windows: UsageWindows) -> list[int]:
    """The length of each day of the window; only the last can be short."""
    length = windows.pool_end - windows.week_start
    full, rest = divmod(length, DAY)
    return [DAY] * full + ([rest] if rest else [])


def reserve_for(reserve: int, day_length: int) -> int:
    """A day's reserve: the daily figure, scaled down for a short day."""
    return reserve * min(day_length, DAY) // DAY


def reserve_today(reserve: int, windows: UsageWindows) -> int:
    return reserve_for(reserve, windows.day_end - windows.day_start)


def weekly_limit_for(weekly: int, windows: UsageWindows) -> int:
    """The window's weekly limit: scaled down for a short 7-day window."""
    length = windows.pool_end - windows.week_start
    if not windows.irregular or length >= WEEK or weekly < 0:
        return weekly
    return weekly * length // WEEK


def main_pool(weekly: int, reserve: int, windows: UsageWindows) -> int:
    """What the window allows above its days' reserves. The reserves and the
    pool add up to the weekly limit exactly; negative means no weekly cap."""
    limit = weekly_limit_for(weekly, windows)
    if limit < 0:
        return limit
    reserves = sum(reserve_for(reserve, length) for length in day_lengths(windows))
    return max(0, limit - reserves)


def hold_reset(period_end: int, now: int) -> int:
    """When a held window is next reported to reset: 5, 10, 20 ... minutes
    after the period's end, then every 6 hours, whichever is first ahead."""
    step = HOLD_FIRST_STEP
    while step < HOLD_MAX_STEP:
        if period_end + step > now:
            return period_end + step
        step *= 2
    past = now - period_end
    return period_end + (past // HOLD_MAX_STEP + 1) * HOLD_MAX_STEP


def _quarter_windows(month_start: int, month_end: int, now: int) -> UsageWindows:
    bounds = weekly_bounds(month_start, month_end)
    index = max((k for k in range(4) if bounds[k] <= now), default=0)
    return _with_day(bounds[index], bounds[index + 1], now, irregular=False)


def _seven_day_windows(period_start: int, period_end: int, now: int) -> UsageWindows:
    start = period_start + ((now - period_start) // WEEK) * WEEK
    return _with_day(start, min(start + WEEK, period_end), now, irregular=True)


def _with_day(
    week_start: int, week_end: int, now: int, irregular: bool
) -> UsageWindows:
    day_start = week_start + (max(now - week_start, 0) // DAY) * DAY
    return UsageWindows(
        week_start=week_start,
        week_end=week_end,
        pool_end=week_end,
        day_start=day_start,
        day_end=min(day_start + DAY, week_end),
        irregular=irregular,
    )


def _held(last: UsageWindows, period_end: int, now: int) -> UsageWindows:
    day_start = last.week_start + ((now - last.week_start) // DAY) * DAY
    return last.model_copy(
        update={
            "week_end": hold_reset(period_end, now),
            "day_start": day_start,
            "day_end": day_start + DAY,
            "held": True,
        }
    )


def _calendar_split(
    period_start: int, period_end: int, anchor: int | None, count: int
) -> list[int]:
    start_day = _utc(period_start).day
    day = _utc(anchor).day if anchor is not None else start_day
    if month_after(period_start, 0, day) != period_start:
        # The anchor doesn't explain where this period starts: use its own day.
        day = start_day
    inner = [month_after(period_start, i, day) for i in range(1, count)]
    return [period_start, *inner, period_end]


def _whole_months(bounds: list[int]) -> bool:
    return all(
        MIN_MONTH <= later - earlier <= MAX_MONTH
        for earlier, later in zip(bounds, bounds[1:])
    )


def _utc(moment: int) -> datetime:
    return datetime.fromtimestamp(moment, UTC)
