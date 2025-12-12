"""
Time utilities for the backend.

Common datetime operations used across the codebase.
"""

from datetime import datetime, timedelta, timezone


def expiration_datetime(seconds: int) -> datetime:
    """
    Calculate an expiration datetime from now.

    Args:
        seconds: Number of seconds until expiration

    Returns:
        Datetime when the item will expire (UTC)
    """
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def is_expired(dt: datetime) -> bool:
    """
    Check if a datetime has passed.

    Args:
        dt: The datetime to check (should be timezone-aware)

    Returns:
        True if the datetime is in the past
    """
    return dt < datetime.now(timezone.utc)


def utc_now() -> datetime:
    """
    Get the current UTC time.

    Returns:
        Current datetime in UTC
    """
    return datetime.now(timezone.utc)
