"""Regression tests for #15276: Unix day-of-week fields must keep their set of
days when translated to APScheduler numbering (Mon=0)."""

from datetime import datetime, timedelta, timezone

import pytest

from backend.api.model import CreateGraph
from backend.data import db
from backend.executor.scheduler import _build_trigger
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.clients import get_scheduler_client
from backend.util.test import SpinTestServer

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
        ("*/3", ["Sun", "Wed", "Sat", "Sun", "Wed", "Sat", "Sun"]),
        ("0-7/2", ["Sun", "Tue", "Thu", "Sat", "Sun", "Tue", "Thu"]),
        ("1/2", ["Mon", "Wed", "Fri", "Mon", "Wed", "Fri", "Mon"]),
        ("5/2", ["Fri"] * 7),
        ("6/2", ["Sat"] * 7),
        ("7/2", ["Sun"] * 7),
        ("5-2/2", ["Sun", "Tue", "Fri", "Sun", "Tue", "Fri", "Sun"]),
        ("1-5/2", ["Mon", "Wed", "Fri", "Mon", "Wed", "Fri", "Mon"]),
        ("6-2", ["Sun", "Mon", "Tue", "Sat", "Sun", "Mon", "Tue"]),
        ("1-5", ["Mon", "Tue", "Wed", "Thu", "Fri", "Mon", "Tue"]),
        ("5-1/2", ["Sun", "Fri", "Sun", "Fri", "Sun", "Fri", "Sun"]),
        ("7", ["Sun"] * 7),
        ("0", ["Sun"] * 7),
        ("1,3,5", ["Mon", "Wed", "Fri", "Mon", "Wed", "Fri", "Mon"]),
        ("mon-fri", ["Mon", "Tue", "Wed", "Thu", "Fri", "Mon", "Tue"]),
        ("SUN-THU", ["Sun", "Mon", "Tue", "Wed", "Thu", "Sun", "Mon"]),
        ("mon-fri/2", ["Mon", "Wed", "Fri", "Mon", "Wed", "Fri", "Mon"]),
    ],
)
def test_dow_field_keeps_its_days(dow, expected):
    assert _fire_weekdays(f"0 0 * * {dow}") == expected


@pytest.mark.parametrize(
    "dow", ["1/", "0-7/", "*/", "1,3/", "*/0", "8", "mon-fri/", "*/x", "sun-"]
)
def test_malformed_dow_field_is_rejected(dow):
    """A slash with no step, a zero step or a day above 7 must raise instead of
    being coerced into a schedule that fires on days nobody asked for."""
    with pytest.raises(ValueError):
        _build_trigger(cron=f"0 0 * * {dow}", run_at=None, user_timezone="UTC")


# Python weekday numbers (Mon=0..Sun=6) each Unix day-of-week field covers.
_DOW_FIELD_WEEKDAYS = {
    "1-5": {0, 1, 2, 3, 4},
    "0-7": {0, 1, 2, 3, 4, 5, 6},
    "*/2": {6, 1, 3, 5},
    "5-1/2": {4, 6},
    "sun-thu": {6, 0, 1, 2, 3},
}


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("dow", list(_DOW_FIELD_WEEKDAYS))
async def test_schedule_next_run_lands_on_the_cron_weekdays(
    server: SpinTestServer, dow: str
):
    """#15276: a schedule created through the real scheduler service must
    report a next run on one of the days its Unix day-of-week field names.
    The hour sits 12 h from now so the reference can't straddle a tick."""
    await db.connect()
    test_user = await create_test_user()
    test_graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()),
        user_id=test_user.id,
    )
    now = datetime.now(timezone.utc)
    hour = (now.hour + 12) % 24
    expected = next(
        candidate
        for day in range(8)
        if (
            candidate := (now + timedelta(days=day)).replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
        )
        > now
        and candidate.weekday() in _DOW_FIELD_WEEKDAYS[dow]
    )

    scheduler = get_scheduler_client()
    schedule = await scheduler.add_execution_schedule(
        graph_id=test_graph.id,
        user_id=test_user.id,
        graph_version=1,
        cron=f"0 {hour} * * {dow}",
        input_data={"input": "data"},
        input_credentials={},
        user_timezone="UTC",
    )
    try:
        next_run = datetime.fromisoformat(schedule.next_run_time)
        assert next_run == expected, (
            f"{dow!r} next run {next_run:%A %Y-%m-%d %H:%M}, "
            f"expected {expected:%A %Y-%m-%d %H:%M}"
        )
    finally:
        await scheduler.delete_schedule(schedule.id, user_id=test_user.id)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("dow", ["1/", "0-7/"])
async def test_schedule_with_an_empty_dow_step_is_refused_not_saved(
    server: SpinTestServer, dow: str
):
    """A slash with nothing after it is malformed cron. The scheduler service
    must refuse it (the API turns the ValueError into a 400) instead of saving
    a Monday or every-day schedule the user never asked for."""
    await db.connect()
    test_user = await create_test_user()
    test_graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()),
        user_id=test_user.id,
    )
    scheduler = get_scheduler_client()

    with pytest.raises(ValueError):
        await scheduler.add_execution_schedule(
            graph_id=test_graph.id,
            user_id=test_user.id,
            graph_version=1,
            cron=f"0 9 * * {dow}",
            input_data={"input": "data"},
            input_credentials={},
            user_timezone="UTC",
        )

    saved = await scheduler.get_execution_schedules(
        graph_id=test_graph.id, user_id=test_user.id
    )
    assert saved == []
