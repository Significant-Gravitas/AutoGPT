"""A task's start time: read it from what was asked, and say it back.

Shared by ``start_task`` (a start Otto sets), the task routes (a start the
user sets) and the due-task sweep. Kept apart from what starts a task, so a
later repeating task can reuse the same reading.
"""

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from backend.data.db_accessors import user_db
from backend.util.timezone_utils import get_user_timezone_or_utc

logger = logging.getLogger(__name__)

# Sooner than this is "now" (the design's shortest delay for a one-off).
MIN_START_DELAY = timedelta(seconds=60)
MAX_START_DELAY = timedelta(days=366)


def resolve_start_time(
    start_in_seconds: int | None,
    start_at: str | None,
    timezone: str,
    now: datetime,
) -> datetime | None:
    """The start as a UTC time, or None to start now (no start given, under a
    minute away, or already past). A ``start_at`` with no offset is read in
    ``timezone``. Raises ValueError, written for the model."""
    if start_in_seconds is not None and start_at:
        raise ValueError("Give start_in_seconds or start_at, not both.")
    if start_in_seconds is not None:
        at = now + timedelta(seconds=max(0, start_in_seconds))
    elif start_at:
        at = _parse_time(start_at, timezone)
    else:
        return None
    if at - now < MIN_START_DELAY:
        return None
    if at - now > MAX_START_DELAY:
        raise ValueError("A start time can be at most a year ahead.")
    return at.astimezone(UTC)


def describe_start(at: datetime, timezone: str) -> str:
    """The start as a person reads it, in their time zone."""
    local = at.astimezone(ZoneInfo(timezone))
    return f"{local:%a} {local.day} {local:%b %Y, %H:%M} {timezone}"


async def user_timezone(user_id: str) -> tuple[str, bool]:
    """The user's time zone, and whether they saved one (else UTC)."""
    try:
        saved = (await user_db().get_user_by_id(user_id)).timezone
    except ValueError:
        logger.warning("User %s… not found, reading times in UTC", user_id[:8])
        saved = None
    timezone = get_user_timezone_or_utc(saved)
    return timezone, timezone == saved


def _parse_time(text: str, timezone: str) -> datetime:
    try:
        at = datetime.fromisoformat(text.strip())
    except ValueError:
        raise ValueError(
            "start_at must be an ISO date-time, such as 2026-11-16T09:00."
        ) from None
    if at.tzinfo is None:
        at = at.replace(tzinfo=ZoneInfo(timezone))
    return at
