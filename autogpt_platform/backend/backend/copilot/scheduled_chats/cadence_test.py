from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from backend.copilot.scheduled_chats.cadence import (
    _fires,
    _runs_per_day,
    describe,
    less_often,
    runs_more_than_weekly,
)
from backend.executor.scheduler import _build_trigger

_CHICAGO = "America/Chicago"
# A Sunday afternoon in Chicago.
_NOW = datetime(2026, 10, 11, 13, 0, tzinfo=ZoneInfo(_CHICAGO))


def _labels(crons: list[str]) -> list[str]:
    return [rung.label for rung in less_often(crons, _CHICAGO, _NOW)]


def test_the_every_minute_monitor_is_offered_the_whole_ladder():
    assert _labels(["* * * * *"]) == [
        "Every 15 minutes",
        "Every hour",
        "Every 4 hours",
        "Every day at 09:00",
        "Weekdays at 09:00",
        "Every Monday at 09:00",
    ]


def test_an_hourly_routine_keeps_the_minute_it_was_spread_to():
    rungs = less_often(["17 * * * *"], _CHICAGO, _NOW)

    assert [(rung.cron, rung.label) for rung in rungs] == [
        ("17 */4 * * *", "Every 4 hours"),
        ("17 9 * * *", "Every day at 09:17"),
        ("17 9 * * mon-fri", "Weekdays at 09:17"),
        ("17 9 * * mon", "Every Monday at 09:17"),
    ]


def test_a_routine_with_two_times_a_day_keeps_its_morning_one():
    assert _labels(["0 13 * * *", "30 8 * * *"]) == [
        "Every day at 08:30",
        "Weekdays at 08:30",
        "Every Monday at 08:30",
    ]


def test_a_night_run_gives_way_to_the_first_one_after_six():
    assert _labels(["30 0 * * *", "0 18 * * *"])[0] == "Every day at 18:00"


def test_weekdays_can_only_drop_to_once_a_week_on_the_day_it_already_runs():
    # Unix numbering, as stored: 1-5 is Monday to Friday.
    assert _labels(["0 9 * * 1-5"]) == ["Every Monday at 09:00"]
    assert _labels(["0 9,17 * * 4"]) == ["Every Thursday at 09:00"]


@pytest.mark.parametrize(
    "crons",
    [["0 9 * * mon"], ["0 9 1 * *"], [], ["not a cron"]],
    ids=["weekly", "monthly", "one-off", "unreadable"],
)
def test_nothing_runs_less_often_than_once_a_week(crons):
    assert less_often(crons, _CHICAGO, _NOW) == []


def test_a_date_that_never_comes_reads_as_unreadable():
    """31 February parses, and croniter only refuses it once it steps."""
    assert less_often(["0 9 * * *", "0 0 31 2 *"], _CHICAGO, _NOW) == []
    assert describe(["0 0 31 2 *"], _CHICAGO, _NOW) is None


def test_times_stay_on_the_schedules_own_clock():
    utc_now = _NOW.astimezone(timezone.utc)

    assert less_often(["30 8 * * *"], _CHICAGO, utc_now)[0].cron == "30 8 * * mon-fri"


def test_each_rung_runs_as_often_as_it_says():
    for rung in less_often(["* * * * *"], _CHICAGO, _NOW):
        assert len(_fires([rung.cron], _CHICAGO, _NOW)) == rung.runs_per_week
        assert _runs_per_day([rung.cron], _CHICAGO, _NOW) == rung.runs_per_day


@pytest.mark.parametrize(
    "now",
    [
        _NOW.replace(minute=16, second=59, microsecond=999_999),
        _NOW.replace(minute=17),
        _NOW.replace(minute=17, microsecond=50_000),
        _NOW.replace(minute=17, second=1),
    ],
    ids=["just-before-a-run", "on-a-run", "inside-its-second", "just-after"],
)
def test_the_offer_holds_through_the_second_a_run_fires(now):
    """CI opened the dialog at 20:16:59 and saved at 20:17:00.05: counting from
    a second before then caught the 20:17 run and the one a week on, so the
    hour it offered was gone by the save."""
    assert [rung.cron for rung in less_often(["17 * * * *"], _CHICAGO, now)] == [
        "17 */4 * * *",
        "17 9 * * *",
        "17 9 * * mon-fri",
        "17 9 * * mon",
    ]


def test_the_offer_does_not_depend_on_which_run_is_next():
    # Its next run is 14:17 at 14:10, and 14:47 at 14:30.
    early = less_often(["17,47 */2 * * *"], _CHICAGO, _NOW.replace(hour=14, minute=10))
    late = less_often(["17,47 */2 * * *"], _CHICAGO, _NOW.replace(hour=14, minute=30))

    assert [rung.cron for rung in early] == [rung.cron for rung in late]
    assert early[0].cron == "17 */4 * * *"


@pytest.mark.parametrize(
    ("crons", "labels"),
    [
        (["*/5 * * * sat"], ["Every Saturday at 09:00"]),
        # Past the weekly cap: it still runs on Saturdays only.
        (["* * * * sat"], ["Every Saturday at 09:00"]),
        (["0 * * * sat"], ["Every Saturday at 09:00"]),
        (["*/5 * * * mon-fri"], ["Weekdays at 09:00", "Every Monday at 09:00"]),
    ],
    ids=[
        "every-5-min-saturdays",
        "every-minute-saturdays",
        "hourly-saturdays",
        "weekdays",
    ],
)
def test_running_less_often_adds_no_runs_on_a_day_it_leaves_free(crons, labels):
    """Every hour has fewer runs in a week than every five minutes on
    Saturdays, but runs on six days that had none, spending their allowance.
    Its weekly rung is on the day it runs."""
    assert _labels(crons) == labels


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 10, 30, 13, 0, tzinfo=ZoneInfo(_CHICAGO)),
        datetime(2027, 3, 12, 13, 0, tzinfo=ZoneInfo(_CHICAGO)),
    ],
    ids=["clocks-fall-back", "clocks-spring-forward"],
)
def test_the_week_the_clocks_change_reads_like_any_other(now):
    """Counted on the zone's own clock, Chicago's fall-back Sunday has 01:17
    twice and its spring-forward one none, so an hourly schedule was offered
    "Every hour" and lost its minute."""
    rungs = less_often(["17 * * * *"], _CHICAGO, now)

    assert [rung.label for rung in rungs] == [
        "Every 4 hours",
        "Every day at 09:17",
        "Weekdays at 09:17",
        "Every Monday at 09:17",
    ]
    assert describe(["17 * * * *"], _CHICAGO, now) == "Every hour"
    assert describe(["30 1 * * *"], _CHICAGO, now) == "Every day at 01:30"


@pytest.mark.parametrize(
    ("crons", "words"),
    [
        (["* * * * *"], "Every minute"),
        (["*/15 * * * *"], "Every 15 minutes"),
        (["17 * * * *"], "Every hour"),
        (["0 */4 * * *"], "Every 4 hours"),
        (["30 8 * * *"], "Every day at 08:30"),
        (["0 9 * * mon"], "Every Monday at 09:00"),
        (["30 8 * * *", "0 13 * * *"], "Twice a day"),
        (["0 9 * * 1-5"], "5 times a week"),
        # Seven runs, all on Mondays.
        (["0 9-15 * * mon"], "7 times a week"),
        # Every minute for 21 hours a day: more runs than are counted.
        (["* 0-20 * * *"], "At least 1000 times a week"),
        (["0 9 1 * *"], None),
    ],
)
def test_describe(crons, words):
    assert describe(crons, _CHICAGO, _NOW) == words


def test_a_busy_beat_cut_short_at_the_cap_is_not_read_as_continuous():
    """A thousand runs of every minute on Saturdays are all a minute apart, all
    on one Saturday."""
    assert describe(["* * * * sat"], _CHICAGO, _NOW) == "At least 1000 times a week"
    assert (
        describe(["*/2 * * * mon-fri"], _CHICAGO, _NOW) == "At least 1000 times a week"
    )
    assert describe(["*/2 * * * *"], _CHICAGO, _NOW) == "Every 2 minutes"


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 10, 16, 10, 0, tzinfo=ZoneInfo(_CHICAGO)),
        datetime(2026, 10, 17, 10, 0, tzinfo=ZoneInfo(_CHICAGO)),
    ],
    ids=["friday-after-its-run", "saturday"],
)
def test_weekdays_read_late_in_the_week_do_not_pass_for_daily(now):
    """The weekend falls after their last run of the week, and must still
    fall between two runs of the ones read."""
    assert describe(["0 9 * * 1-5"], _CHICAGO, now) == "5 times a week"
    assert describe(["*/30 * * * mon-fri"], _CHICAGO, now) == "240 times a week"


@pytest.mark.parametrize(
    "crons",
    [
        ["* * * * *"],
        ["17 * * * *"],
        ["0 9 * * 1-5"],
        ["0 9,17 * * 4"],
        ["*/5 * * * sat"],
        ["* * * * sat"],
        ["0 9 * * mon"],
        ["0 9 * * mon", "0 9 * * thu"],
        ["0 9 1 * *"],
        [],
        ["not a cron"],
    ],
)
def test_the_skip_note_offers_run_less_often_exactly_when_the_ladder_has_a_rung(
    crons,
):
    """The note asks the cheap question on every skipped run; the dialog
    builds the ladder. They must agree, or the button opens on nothing."""
    assert runs_more_than_weekly(crons, _CHICAGO, _NOW) == bool(
        less_often(crons, _CHICAGO, _NOW)
    )


def _scheduled(cron: str, start: datetime, end: datetime) -> list[datetime]:
    """When the scheduler's own trigger fires *cron* between *start* and *end*."""
    trigger = _build_trigger(cron=cron, run_at=None, user_timezone="UTC")
    fires: list[datetime] = []
    previous = None
    while True:
        fire = trigger.get_next_fire_time(previous, previous or start)
        if fire is None or fire >= end:
            return fires
        fires.append(fire)
        previous = fire


@pytest.mark.parametrize(
    "cron",
    [
        # The first Monday of the month: both days must match, not either.
        "0 9 1-7 * mon",
        # A step over every weekday counts from Monday.
        "0 9 * * */2",
        "0 9 * * */3",
        "0 9 * * 5-2",
        "0 9 * * 1-5",
        "0 9 * * 1-5/2",
        "0 9 * * 7",
        "0 9 1,15 * *",
        "*/30 * * * mon-fri",
    ],
)
def test_runs_are_counted_as_the_scheduler_fires_them(cron):
    start = datetime(2026, 10, 11, tzinfo=timezone.utc)
    four_weeks = timedelta(weeks=4)

    counted = _fires([cron], "UTC", start, span=four_weeks)

    assert counted == _scheduled(cron, start, start + four_weeks)
