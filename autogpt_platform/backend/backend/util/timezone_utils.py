"""
Timezone conversion utilities for API endpoints.
Handles conversion between user timezones and UTC for scheduler operations.
"""

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from backend.data.model import USER_TIMEZONE_NOT_SET

logger = logging.getLogger(__name__)


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
