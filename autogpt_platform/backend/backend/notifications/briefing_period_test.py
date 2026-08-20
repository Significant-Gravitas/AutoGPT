"""A briefing lands at ~07:30 in the reader's own morning, not ours."""

from datetime import datetime, timezone

from prisma.enums import BriefingFrequency

from backend.notifications.briefing_period import (
    BRIEFING_HOUR,
    is_briefing_due,
    period_window,
    resolve_zone,
)

# 07:30 in New York is 11:30 UTC in August.
NY_MORNING = datetime(2026, 8, 3, 11, 30, tzinfo=timezone.utc)  # a Monday


def test_due_in_the_users_own_morning_not_ours():
    assert is_briefing_due(
        BriefingFrequency.DAILY, "America/New_York", NY_MORNING, None
    )
    # Same instant is mid-afternoon in Berlin, so nothing is due there.
    assert not is_briefing_due(
        BriefingFrequency.DAILY, "Europe/Berlin", NY_MORNING, None
    )


def test_weekly_only_fires_on_a_monday():
    tuesday = datetime(2026, 8, 4, 11, 30, tzinfo=timezone.utc)
    assert is_briefing_due(
        BriefingFrequency.WEEKLY, "America/New_York", NY_MORNING, None
    )
    assert not is_briefing_due(
        BriefingFrequency.WEEKLY, "America/New_York", tuesday, None
    )


def test_monthly_only_fires_on_the_first():
    first = datetime(2026, 9, 1, 11, 30, tzinfo=timezone.utc)
    assert is_briefing_due(BriefingFrequency.MONTHLY, "America/New_York", first, None)
    assert not is_briefing_due(
        BriefingFrequency.MONTHLY, "America/New_York", NY_MORNING, None
    )


def test_off_is_never_due():
    assert not is_briefing_due(
        BriefingFrequency.OFF, "America/New_York", NY_MORNING, None
    )


def test_a_period_is_never_briefed_twice():
    window = period_window(BriefingFrequency.WEEKLY, "America/New_York", NY_MORNING)
    already_sent = window.end
    assert not is_briefing_due(
        BriefingFrequency.WEEKLY, "America/New_York", NY_MORNING, already_sent
    )


def test_an_unset_timezone_is_treated_as_utc_rather_than_skipped():
    utc_morning = datetime(2026, 8, 3, BRIEFING_HOUR, 30, tzinfo=timezone.utc)
    assert is_briefing_due(BriefingFrequency.DAILY, "not-set", utc_morning, None)
    assert resolve_zone("Nowhere/Fictional").key == "UTC"


def test_daily_window_covers_yesterday_in_local_time():
    window = period_window(BriefingFrequency.DAILY, "America/New_York", NY_MORNING)
    assert window.period.noun == "yesterday"
    assert window.period.frequency == "daily"
    assert (window.end - window.start).days == 1


def test_monthly_window_covers_the_previous_calendar_month():
    first = datetime(2026, 9, 1, 11, 30, tzinfo=timezone.utc)
    window = period_window(BriefingFrequency.MONTHLY, "America/New_York", first)
    assert window.period.label == "August 2026"
    assert window.period.noun == "in August"
