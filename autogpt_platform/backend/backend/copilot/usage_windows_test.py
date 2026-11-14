"""The billing-clock window maths: every month length, month-end anchors, leap
years, yearly plans, odd periods and held windows. No test reads a clock."""

import calendar
import random
import subprocess
import sys
from datetime import UTC, datetime

import pytest

from backend.copilot.usage_windows import (
    DAY,
    WEEK,
    UsageWindows,
    billing_month,
    day_lengths,
    hold_reset,
    main_pool,
    month_after,
    reserve_for,
    reserve_today,
    weekly_bounds,
    weekly_limit_for,
    windows_for_period,
    windows_from_anchor,
)

HOUR = 3600
MAX = {"weekly": 68_000_000, "daily": 22_666_667, "reserve": 2_000_000}
PRO = {"weekly": 8_000_000, "daily": 3_200_000, "reserve": 250_000}


def ts(*parts: int) -> int:
    return calendar.timegm(datetime(*parts, tzinfo=UTC).timetuple())


def monthly_periods() -> list[tuple[int, int]]:
    """36 consecutive Stripe-style monthly periods for anchors on the 1st,
    15th and 28th to 31st, through the leap February of 2028."""
    periods = []
    for day in (1, 15, 28, 29, 30, 31):
        anchor = ts(2026, 1, day, 20, 30, 56)
        periods += [
            (month_after(anchor, i, day), month_after(anchor, i + 1, day))
            for i in range(36)
        ]
    return periods


def flat_out(limits: dict[str, int], windows: UsageWindows) -> list[int]:
    """What a customer who spends everything the moment it's allowed gets on
    each day of a window."""
    pool = main_pool(limits["weekly"], limits["reserve"], windows)
    used, spent = 0, []
    for length in day_lengths(windows):
        reserve = reserve_for(limits["reserve"], length)
        today = min(limits["daily"], reserve + max(0, pool - used))
        used += max(0, today - reserve)
        spent.append(today)
    return spent


def test_the_tickets_worked_example_resets_to_the_second():
    start, end = ts(2026, 11, 4, 20, 30, 56), ts(2026, 12, 4, 20, 30, 56)

    assert weekly_bounds(start, end) == [
        start,
        ts(2026, 11, 12, 8, 30, 56),
        ts(2026, 11, 19, 20, 30, 56),
        ts(2026, 11, 27, 8, 30, 56),
        end,
    ]
    # Days restart at each weekly reset: 20:30:56 in windows 1 and 3, 08:30:56
    # in windows 2 and 4, and each window's last day is 12 hours.
    second = windows_for_period(start, end, start, ts(2026, 11, 13, 9, 0, 0))
    assert second.day_start == ts(2026, 11, 13, 8, 30, 56)
    last_day = windows_for_period(start, end, start, ts(2026, 11, 12, 8, 30, 55))
    assert last_day.day_end - last_day.day_start == 12 * HOUR
    assert last_day.week_end == ts(2026, 11, 12, 8, 30, 56)


@pytest.mark.parametrize("start,end", monthly_periods())
def test_a_monthly_period_is_four_windows_that_tile_it(start: int, end: int):
    bounds = weekly_bounds(start, end)

    assert 28 <= (end - start) // DAY <= 31
    assert bounds[0] == start and bounds[4] == end
    assert all(later > earlier for earlier, later in zip(bounds, bounds[1:]))
    assert {later - earlier for earlier, later in zip(bounds, bounds[1:])} == {
        (end - start) // 4
    }
    assert billing_month(start, end, None, start) == (start, end)


@pytest.mark.parametrize(
    "days,window",
    [
        (28, 7 * DAY),
        (29, 7 * DAY + 6 * HOUR),
        (30, 7 * DAY + 12 * HOUR),
        (31, 7 * DAY + 18 * HOUR),
    ],
)
def test_window_length_for_each_month_length(days: int, window: int):
    start = ts(2026, 3, 1, 10, 0, 0)
    bounds = weekly_bounds(start, start + days * DAY)

    assert [later - earlier for earlier, later in zip(bounds, bounds[1:])] == [
        window
    ] * 4


@pytest.mark.parametrize("remainder", [1, 2, 3])
def test_a_length_not_divisible_by_four_still_tiles_exactly(remainder: int):
    start = ts(2026, 3, 1)
    end = start + 30 * DAY + remainder
    lengths = [
        b - a for a, b in zip(weekly_bounds(start, end), weekly_bounds(start, end)[1:])
    ]

    assert sum(lengths) == end - start
    assert max(lengths) - min(lengths) <= 1


@pytest.mark.parametrize("start,end", monthly_periods()[::7])
def test_every_moment_is_in_one_window_and_one_day(start: int, end: int):
    bounds = weekly_bounds(start, end)
    for edge in bounds[:4]:
        for now in (edge, edge + 1, edge + DAY - 1, edge + DAY):
            windows = windows_for_period(start, end, None, now)
            assert windows.week_start <= now < windows.week_end
            assert windows.day_start <= now < windows.day_end
            assert windows.week_start <= windows.day_start
            assert windows.day_end <= windows.week_end
            assert (windows.day_start - windows.week_start) % DAY == 0
    before_reset = windows_for_period(start, end, None, bounds[1] - 1)
    assert before_reset.week_end == bounds[1]
    at_reset = windows_for_period(start, end, None, bounds[1])
    assert at_reset.week_start == at_reset.day_start == bounds[1]


def test_a_moment_before_the_period_is_its_first_window():
    start, end = ts(2026, 11, 4), ts(2026, 12, 4)

    early = windows_for_period(start, end, None, start - 500)

    assert (early.week_start, early.day_start) == (start, start)


@pytest.mark.parametrize(
    "anchor,expected_days",
    [
        (ts(2026, 1, 31, 9, 0, 0), [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
        (ts(2028, 2, 29, 9, 0, 0), [29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29]),
        (ts(2026, 8, 30, 9, 0, 0), [30, 30, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30]),
        (
            ts(2026, 12, 31, 23, 59, 59),
            [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30],
        ),
    ],
)
def test_a_yearly_plan_splits_on_the_anchor_day_like_stripe(
    anchor: int, expected_days: list[int]
):
    day = datetime.fromtimestamp(anchor, UTC).day
    end = month_after(anchor, 12, day)

    months = [
        billing_month(anchor, end, anchor, month_after(anchor, i, day))
        for i in range(12)
    ]

    assert months[0][0] == anchor and months[-1][1] == end
    assert all(a[1] == b[0] for a, b in zip(months, months[1:]))
    assert [
        datetime.fromtimestamp(start, UTC).day for start, _ in months
    ] == expected_days
    assert all(28 * DAY <= stop - start <= 31 * DAY for start, stop in months)


def test_the_year_after_a_leap_day_start_keeps_the_29th():
    # Stripe renews a 29 Feb 2028 yearly plan on 28 Feb 2029; its months still
    # fall on the 29th, as the anchor says.
    anchor = ts(2028, 2, 29, 9, 0, 0)
    start, end = ts(2029, 2, 28, 9, 0, 0), ts(2030, 2, 28, 9, 0, 0)

    second = billing_month(start, end, anchor, ts(2029, 4, 1))

    assert second == (ts(2029, 3, 29, 9, 0, 0), ts(2029, 4, 29, 9, 0, 0))


def test_a_yearly_plan_has_48_windows():
    start, end = ts(2026, 11, 4, 20, 30, 56), ts(2027, 11, 4, 20, 30, 56)
    starts = set()
    now = start
    while now < end:
        starts.add(windows_for_period(start, end, start, now).week_start)
        now += 6 * HOUR

    assert len(starts) == 48


def test_an_anchor_that_does_not_explain_the_period_is_ignored():
    start, end = ts(2026, 3, 20, 9, 0, 0), ts(2027, 3, 20, 9, 0, 0)

    first = billing_month(start, end, ts(2025, 1, 5, 9, 0, 0), start)

    assert first == (start, ts(2026, 4, 20, 9, 0, 0))


@pytest.mark.parametrize("days", [7, 10, 14, 45])
def test_an_odd_period_gets_seven_day_windows(days: int):
    start = ts(2026, 5, 1, 12, 0, 0)
    end = start + days * DAY

    assert billing_month(start, end, None, start) is None
    first = windows_for_period(start, end, None, start + DAY)
    assert (first.week_start, first.irregular) == (start, True)
    last = windows_for_period(start, end, None, end - 1)
    assert last.week_end == end
    stub = (end - start) % WEEK or WEEK
    assert weekly_limit_for(7_000_000, last) == 7_000_000 * stub // WEEK
    assert weekly_limit_for(-1, last) == -1


def test_an_account_clock_resets_monthly_on_the_signup_day_and_time():
    created = ts(2025, 1, 31, 7, 15, 30)

    february = windows_from_anchor(created, ts(2026, 2, 10))
    march = windows_from_anchor(created, ts(2026, 3, 31, 7, 15, 30))

    assert february.week_start == ts(2026, 2, 7, 7, 15, 30)  # second quarter
    assert february.week_end - february.week_start == 7 * DAY
    assert march.week_start == ts(2026, 3, 31, 7, 15, 30)
    assert windows_from_anchor(created, created - 10).week_start < created


def test_a_period_that_ended_holds_its_last_window_open():
    start, end = ts(2026, 11, 4, 20, 30, 56), ts(2026, 12, 4, 20, 30, 56)
    last = windows_for_period(start, end, None, end - 1)

    held = windows_for_period(start, end, None, end + 3 * DAY + 100)

    assert held.held and not last.held
    assert held.week_start == last.week_start  # same counter, no new allowance
    assert held.pool_end == end  # the pool is still the last window's
    assert held.day_end - held.day_start == DAY  # days keep turning
    assert held.day_start <= end + 3 * DAY + 100 < held.day_end
    assert held.week_end > end + 3 * DAY + 100
    assert reserve_today(2_000_000, held) == 2_000_000


def test_a_held_windows_reported_reset_backs_off():
    end = ts(2026, 12, 4, 20, 30, 56)

    resets = [hold_reset(end, end + past) for past in (0, 299, 300, 900, 5000)]

    assert resets == [end + 300, end + 300, end + 600, end + 1200, end + 9600]
    assert hold_reset(end, end + 7 * 3600) == end + 12 * 3600
    assert hold_reset(end, end + 30 * 3600) == end + 36 * 3600


@pytest.mark.parametrize(
    "days,limits,expected,total",
    [
        (30, MAX, [22_666_667, 22_666_667, 13_666_666, 2e6, 2e6, 2e6, 2e6, 1e6], 68e6),
        (
            30,
            PRO,
            [3.2e6, 3.2e6, 475_000, 250_000, 250_000, 250_000, 250_000, 125_000],
            8e6,
        ),
        (28, MAX, [22_666_667, 22_666_667, 14_666_666, 2e6, 2e6, 2e6, 2e6], 68e6),
        (
            31,
            PRO,
            [3.2e6, 3.2e6, 412_500, 250_000, 250_000, 250_000, 250_000, 187_500],
            8e6,
        ),
    ],
)
def test_spending_flat_out_gives_the_tickets_days_and_exactly_the_weekly_limit(
    days: int, limits: dict[str, int], expected: list[float], total: float
):
    start = ts(2026, 3, 1, 10, 0, 0)
    first = windows_for_period(start, start + days * DAY, None, start)

    spent = flat_out(limits, first)

    assert spent == expected
    assert sum(spent) == total


def test_the_reserves_and_the_pool_add_up_to_the_weekly_limit():
    rng = random.Random(3441)
    for _ in range(300):
        start = rng.randrange(1_700_000_000, 1_900_000_000)
        end = start + rng.randrange(28 * DAY, 31 * DAY + 1)
        windows = windows_for_period(start, end, None, rng.randrange(start, end))
        reserves = sum(reserve_for(2_000_000, n) for n in day_lengths(windows))

        assert windows.week_start <= windows.day_start < windows.day_end
        assert windows.day_end <= windows.week_end <= end
        assert main_pool(68_000_000, 2_000_000, windows) + reserves == 68_000_000


def test_no_weekly_cap_stays_uncapped_and_a_huge_reserve_cannot_go_negative():
    start = ts(2026, 3, 1)
    windows = windows_for_period(start, start + 30 * DAY, None, start)

    assert main_pool(-1, 2_000_000, windows) == -1
    assert main_pool(1_000_000, 2_000_000, windows) == 0


def test_local_time_and_daylight_saving_cannot_move_a_reset():
    # The same maths under three time zones, across both 2026 clock changes.
    script = (
        "from backend.copilot.usage_windows import windows_for_period as w;"
        "print([w(1772366400 + m * 2629746, 1772366400 + m * 2629746 + 2592000,"
        " None, 1772366400 + m * 2629746 + 700000).model_dump() for m in range(12)])"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            env={"TZ": zone, "PYTHONPATH": ":".join(sys.path)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for zone in ("UTC", "America/New_York", "Pacific/Kiritimati")
    }

    assert len(outputs) == 1


def test_a_period_far_longer_than_any_plan_gets_seven_day_windows():
    start = ts(2026, 1, 1)

    assert billing_month(start, ts(2032, 1, 1), None, start) is None


def test_whole_months_that_do_not_sit_on_the_calendar_are_split_evenly():
    """Three 31-day months from 1 January: on the calendar the last would run
    from 1 March for 34 days, so the period is cut in three equal parts."""
    start = ts(2027, 1, 1, 9, 0, 0)
    end = start + 93 * DAY

    months = [billing_month(start, end, None, start + i * 31 * DAY) for i in range(3)]

    assert months == [
        (start, start + 31 * DAY),
        (start + 31 * DAY, start + 62 * DAY),
        (start + 62 * DAY, end),
    ]
