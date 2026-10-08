"""Regression tests for #15276: Unix day-of-week fields must keep their set of
days when translated to APScheduler numbering (Mon=0)."""

from datetime import datetime, timedelta, timezone

import pytest

from backend.executor.scheduler import _build_trigger

# 2026-10-04 is a Sunday.
START = datetime(2026, 10, 4, 0, 0, tzinfo=timezone.utc) - timedelta(minutes=1)
NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _fire_weekdays(cron: str, n: int = 7) -> list[str]:
    trigger = _build_trigger(cron=cron, run_at=None, user_timezone="UTC")
    fired, prev, now = [], None, START
    for _ in range(n):
        nxt = trigger.get_next_fire_time(prev, now)
        assert nxt is not None
        fired.append(NAMES[nxt.weekday()])
        prev, now = nxt, nxt + timedelta(seconds=1)
    return fired


@pytest.mark.parametrize(
    "dow, expected",
    [
        ("0-7", ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]),
        ("*/2", ["Sun", "Tue", "Thu", "Sat", "Sun", "Tue", "Thu"]),
        ("5-2/2", ["Sun", "Tue", "Fri", "Sun", "Tue", "Fri", "Sun"]),
        ("1-5/2", ["Mon", "Wed", "Fri", "Mon", "Wed", "Fri", "Mon"]),
        ("6-2", ["Sun", "Mon", "Tue", "Sat", "Sun", "Mon", "Tue"]),
        ("1-5", ["Mon", "Tue", "Wed", "Thu", "Fri", "Mon", "Tue"]),
        ("7", ["Sun"] * 7),
        ("0", ["Sun"] * 7),
        ("mon-fri", ["Mon", "Tue", "Wed", "Thu", "Fri", "Mon", "Tue"]),
    ],
)
def test_dow_field_keeps_its_days(dow, expected):
    assert _fire_weekdays(f"0 0 * * {dow}") == expected
