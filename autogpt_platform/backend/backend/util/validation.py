"""Validation utilities."""

import re

_UUID_V4_PATTERN = re.compile(
    r"[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}",
    re.IGNORECASE,
)


def is_uuid_v4(text: str) -> bool:
    """Check if text is a valid UUID v4.

    Args:
        text: String to validate

    Returns:
        True if the text is a valid UUID v4, False otherwise
    """
    return bool(_UUID_V4_PATTERN.fullmatch(text.strip()))


def extract_uuids(text: str) -> list[str]:
    """Extract all UUID v4 strings from text.

    Args:
        text: String to search for UUIDs

    Returns:
        List of unique UUIDs found (lowercase)
    """
    return list({m.lower() for m in _UUID_V4_PATTERN.findall(text)})
