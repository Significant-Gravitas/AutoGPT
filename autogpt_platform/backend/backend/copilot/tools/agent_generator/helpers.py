"""Shared helpers for agent generation."""

import re
import uuid

from .blocks import get_blocks_as_dicts

__all__ = [
    "AGENT_EXECUTOR_BLOCK_ID",
    "UUID_REGEX",
    "generate_uuid",
    "get_blocks_as_dicts",
    "is_uuid",
]

UUID_REGEX = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[a-f0-9]{4}-[a-f0-9]{12}$"
)

AGENT_EXECUTOR_BLOCK_ID = "e189baac-8c20-45a1-94a7-55177ea42565"


def is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    return isinstance(value, str) and UUID_REGEX.match(value) is not None


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())
