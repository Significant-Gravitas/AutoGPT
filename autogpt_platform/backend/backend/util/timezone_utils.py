"""
Timezone conversion utilities for API endpoints.
Handles conversion between user timezones and UTC for scheduler operations.
"""

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from croniter import croniter

from backend.data.model import USER_TIMEZONE_NOT_SET

logger = logging.getLogger(__name__)


def convert_cron_to_utc(cron_expr: str, user_timezone: str) -> str:
    """
    Convert a cron expression from user timezone to UTC.

    NOTE: This is a simplified conversion that only adjusts minute and hour fields.
    Complex cron expressions with specific day/month/weekday patterns may not
    convert accurately due to timezone offset variations throughout the year.

    Args:
        cron_expr: Cron expression in user timezone
        user_timezone: User's IANA timezone identifier

    Returns:
        Cron expression adjusted for UTC execution

    Raises:
        ValueError: If timezone or cron expression is invalid
    """
    try:
        user_tz = ZoneInfo(user_timezone)
        utc_tz = ZoneInfo("UTC")

        # Split the cron expression into its five fields
        cron_fields = cron_expr.strip().split()
        if len(cron_fields) != 5:
            raise ValueError(
                "Cron expression must have 5 fields (minute hour day month weekday)"
            )

        # Get the current time in the user's timezone
        now_user = datetime.now(user_tz)

        # Get the next scheduled time in user timezone
        cron = croniter(cron_expr, now_user)
        next_user_time = cron.get_next(datetime)

        # Convert to UTC
        next_utc_time = next_user_time.astimezone(utc_tz)

        # Adjust minute and hour fields for UTC, keep day/month/weekday as in original
        utc_cron_parts = [
            str(next_utc_time.minute),
            str(next_utc_time.hour),
            cron_fields[2],  # day of month
            cron_fields[3],  # month
            cron_fields[4],  # day of week
        ]

        utc_cron = " ".join(utc_cron_parts)

        logger.debug(
            f"Converted cron '{cron_expr}' from {user_timezone} to UTC: '{utc_cron}'"
        )
        return utc_cron

    except Exception as e:
        logger.error(
            f"Failed to convert cron expression '{cron_expr}' from {user_timezone} to UTC: {e}"
        )
        raise ValueError(f"Invalid cron expression or timezone: {e}")


def convert_utc_time_to_user_timezone(utc_time_str: str, user_timezone: str) -> str:
    """
    Convert a UTC datetime string to user timezone.

    Args:
        utc_time_str: ISO format datetime string in UTC
        user_timezone: User's IANA timezone identifier

    Returns:
        ISO format datetime string in user timezone
    """
    try:
        # Parse the time string
        parsed_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))

        user_tz = ZoneInfo(user_timezone)

        # If the time already has timezone info, convert it to user timezone
        if parsed_time.tzinfo is not None:
            # Convert to user timezone regardless of source timezone
            user_time = parsed_time.astimezone(user_tz)
            return user_time.isoformat()

        # If no timezone info, treat as UTC and convert to user timezone
        parsed_time = parsed_time.replace(tzinfo=ZoneInfo("UTC"))
        user_time = parsed_time.astimezone(user_tz)
        return user_time.isoformat()

    except Exception as e:
        logger.error(
            f"Failed to convert UTC time '{utc_time_str}' to {user_timezone}: {e}"
        )
        # Return original time if conversion fails
        return utc_time_str


def validate_timezone(timezone: str) -> bool:
    """
    Validate if a timezone string is a valid IANA timezone identifier.

    Args:
        timezone: Timezone string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        ZoneInfo(timezone)
        return True
    except Exception:
        return False


def get_user_timezone_or_utc(user_timezone: Optional[str]) -> str:
    """
    Get user timezone or default to UTC if invalid/missing.

    Args:
        user_timezone: User's timezone preference

    Returns:
        Valid timezone string (user's preference or UTC fallback)
    """
    if not user_timezone or user_timezone == USER_TIMEZONE_NOT_SET:
        return "UTC"

    if validate_timezone(user_timezone):
        return user_timezone

    logger.warning(f"Invalid user timezone '{user_timezone}', falling back to UTC")
    return "UTC"
