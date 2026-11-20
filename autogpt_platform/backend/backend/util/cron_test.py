from unittest.mock import patch

import pytest
from apscheduler.triggers.cron import CronTrigger

from backend.util.cron import (
    MAX_CRON_LENGTH,
    normalize_cron_day_of_week,
    spell_out_cron_day_of_week,
    validate_cron,
)


def test_validate_cron_refuses_an_overlong_cron_before_parsing_it():
    with (
        patch.object(CronTrigger, "from_crontab") as parse,
        pytest.raises(ValueError, match="too long"),
    ):
        validate_cron("0 9 * * *" + " " * MAX_CRON_LENGTH, "UTC")
    parse.assert_not_called()


def test_validate_cron_returns_an_open_ended_trigger_for_a_cron_that_fires():
    trigger = validate_cron("0 0 29 2 1", "UTC")
    assert trigger.end_date is None


@pytest.mark.parametrize("dow", ["7", "0-7", "5-2/2", "sun-thu"])
def test_validate_cron_accepts_unix_days_apscheduler_alone_refuses(dow: str):
    """Otto's follow-ups and expert routines validate with this before the
    scheduler builds its trigger, so it must read the weekdays the same way:
    APScheduler's own parse refuses 7 (Sunday) and ranges through Sunday."""
    validate_cron(f"0 9 * * {dow}", "UTC")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("*", "*"),
        ("?", "?"),
        ("1-5", "0,1,2,3,4"),
        ("1,3,5", "0,2,4"),
        ("0", "6"),
        ("7", "6"),
        ("0,6", "5,6"),
        ("6-7", "5,6"),
        ("0-4", "0,1,2,3,6"),
        ("5-2", "0,1,4,5,6"),
        # Each token expands to its days before renumbering (#15276): */2 is
        # Sun/Tue/Thu/Sat, 0-7 every day, and a wrap with a step keeps its phase.
        ("*/2", "1,3,5,6"),
        ("1-5/1", "0,1,2,3,4"),
        ("0-7", "0,1,2,3,4,5,6"),
        ("5-2/2", "1,4,6"),
        ("1/2", "0,2,4"),
        ("5/2", "4"),
        # Names read as Unix days too, so a range through Sunday and a step
        # survive APScheduler, which refuses the one and drops the other.
        ("mon-fri", "0,1,2,3,4"),
        ("SUN-THU", "0,1,2,3,6"),
        ("mon-fri/2", "0,2,4"),
        ("sat,sun", "5,6"),
    ],
)
def test_normalize_cron_day_of_week_translates_unix_to_apscheduler(
    raw: str, expected: str
):
    """Unix cron numbers weekdays 0=Sun..6=Sat (7 is Sunday again);
    APScheduler 0=Mon..6=Sun. The field becomes one explicit list of
    APScheduler days; ``*`` and ``?`` pass through."""
    assert normalize_cron_day_of_week(f"0 9 * * {raw}") == f"0 9 * * {expected}"


@pytest.mark.parametrize(
    "raw",
    [
        "*/0",
        "8",
        "1-9",
        "9-2",
        "0-5/0",
        "1/",
        "0-7/",
        "1,3/",
        "*/x",
        "mon-fri/",
        "sun-",
        "1x",
        "L",
        "\u0663",
    ],
)
def test_normalize_cron_day_of_week_rejects_what_cron_rejects(raw: str):
    """A missing or zero step, a day above 7 or a token that isn't a day is
    invalid cron. Coercing it (``*/0`` to every day, ``8`` to Monday, ``1/`` to
    Monday, ``sun-`` to Sunday) would save a schedule that fires on days the
    user never asked for."""
    with pytest.raises(ValueError):
        normalize_cron_day_of_week(f"0 9 * * {raw}")


@pytest.mark.parametrize(
    "raw,expected",
    [("*", "*"), ("1-1", "1"), ("6/2", "6"), ("7/2", "0"), ("sun-thu", "0,1,2,3,4")],
)
def test_spell_out_cron_day_of_week_lists_the_unix_days_the_scheduler_fires(
    raw: str, expected: str
):
    """What the scheduled-chat cadence counter hands croniter, which reads
    ``1-1`` as every day and ``6/2`` as ``*/2``."""
    assert spell_out_cron_day_of_week(f"0 9 * * {raw}") == f"0 9 * * {expected}"
