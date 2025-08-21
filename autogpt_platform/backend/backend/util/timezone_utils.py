"""
Timezone conversion utilities for API endpoints.
Handles conversion between user timezones and UTC for scheduler operations.
"""

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from croniter import croniter

logger = logging.getLogger(__name__)


def convert_cron_to_utc(cron_expr: str, user_timezone: str) -> str:
    """
    Convert a cron expression from user timezone to UTC.

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

        # Get the current time in the user's timezone
        now_user = datetime.now(user_tz)

        # Create croniter with user timezone
        cron = croniter(cron_expr, now_user)

        # Get the next few scheduled times in user timezone
        user_times = [cron.get_next(datetime) for _ in range(5)]

        # Convert to UTC
        utc_times = [dt.astimezone(utc_tz) for dt in user_times]

        # Find the pattern in UTC
        # For simplicity, we'll use the first UTC time to create a new cron expression
        first_utc = utc_times[0]

        # Create a new cron expression based on the UTC time
        # This is a simplified approach - for complex expressions, more sophisticated conversion is needed
        utc_cron_parts = [
            str(first_utc.minute),
            str(first_utc.hour),
            "*",  # day of month - keep flexible
            "*",  # month - keep flexible
            "*",  # day of week - keep flexible
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
        # Parse UTC time
        utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=ZoneInfo("UTC"))

        # Convert to user timezone
        user_tz = ZoneInfo(user_timezone)
        user_time = utc_time.astimezone(user_tz)

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
    if not user_timezone or user_timezone == "not-set":
        return "UTC"

    if validate_timezone(user_timezone):
        return user_timezone

    logger.warning(f"Invalid user timezone '{user_timezone}', falling back to UTC")
    return "UTC"
