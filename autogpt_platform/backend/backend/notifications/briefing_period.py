"""Which period a Briefing covers, and when it is due.

Briefings land at ~07:30 in the user's own local time — a digest that arrives
at 3am reads as a machine talking, not a chief of staff. The scheduler runs
hourly and this module decides, per user, whether this is their hour.
"""

from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from prisma.enums import BriefingFrequency
from pydantic import BaseModel

from backend.data.notifications import BriefingPeriod

# The hour, in the user's local time, that briefings go out.
BRIEFING_HOUR = 7
BRIEFING_MINUTE = 30

BriefingFrequencyLabel = Literal["daily", "weekly", "monthly"]

_FREQUENCY_LABELS: dict[BriefingFrequency, tuple[BriefingFrequencyLabel, str]] = {
    BriefingFrequency.DAILY: ("daily", "day"),
    BriefingFrequency.WEEKLY: ("weekly", "week"),
    BriefingFrequency.MONTHLY: ("monthly", "month"),
}


class PeriodWindow(BaseModel):
    """The UTC window a briefing covers, plus the words that describe it."""

    start: datetime
    end: datetime
    period: BriefingPeriod


def resolve_zone(timezone_name: str) -> ZoneInfo:
    """Users who never set a timezone get UTC rather than no briefing."""
    if not timezone_name or timezone_name == "not-set":
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def is_briefing_due(
    frequency: BriefingFrequency,
    timezone_name: str,
    now: datetime,
    last_sent: datetime | None,
) -> bool:
    """Whether `now` is this user's briefing moment for their frequency.

    Daily fires every local morning, weekly on Mondays, monthly on the 1st.
    `last_sent` guards against a re-run of the hourly job sending the same
    period twice.
    """
    if frequency is BriefingFrequency.OFF:
        return False

    local_now = now.astimezone(resolve_zone(timezone_name))
    if local_now.hour != BRIEFING_HOUR:
        return False
    if frequency is BriefingFrequency.WEEKLY and local_now.weekday() != 0:
        return False
    if frequency is BriefingFrequency.MONTHLY and local_now.day != 1:
        return False

    if last_sent is None:
        return True
    # One send per period: anything inside the window we are about to report on
    # means this user already has it.
    window = period_window(frequency, timezone_name, now)
    return last_sent < window.end


def period_window(
    frequency: BriefingFrequency, timezone_name: str, now: datetime
) -> PeriodWindow:
    """The window to report on, in UTC, with its plain-English labels."""
    zone = resolve_zone(timezone_name)
    local_now = now.astimezone(zone)
    today = local_now.date()

    if frequency is BriefingFrequency.DAILY:
        start_date, end_date = today - timedelta(days=1), today
        label = _format_day(start_date)
        noun = "yesterday"
    elif frequency is BriefingFrequency.MONTHLY:
        end_date = today.replace(day=1)
        start_date = (end_date - timedelta(days=1)).replace(day=1)
        label = start_date.strftime("%B %Y")
        noun = f"in {start_date.strftime('%B')}"
    else:
        start_date, end_date = today - timedelta(days=7), today
        label = (
            f"{_format_day(start_date)} – {_format_day(end_date - timedelta(days=1))}"
        )
        noun = "this week"

    frequency_word, adjective = _FREQUENCY_LABELS[frequency]
    return PeriodWindow(
        start=_utc_midnight(start_date, zone),
        end=_utc_midnight(end_date, zone),
        period=BriefingPeriod(
            label=label,
            noun=noun,
            adjective=adjective,
            frequency=frequency_word,
        ),
    )


def _utc_midnight(day: date, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, time.min, tzinfo=zone).astimezone(timezone.utc)


def _format_day(day: date) -> str:
    return f"{day.strftime('%a')} {day.day} {day.strftime('%b')}"
