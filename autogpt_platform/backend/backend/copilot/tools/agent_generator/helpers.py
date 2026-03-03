"""Shared helpers for agent generation."""

import re
import uuid

from .blocks import get_blocks_as_dicts

__all__ = [
    "AGENT_EXECUTOR_BLOCK_ID",
    "AGENT_INPUT_BLOCK_ID",
    "AGENT_OUTPUT_BLOCK_ID",
    "UUID_REGEX",
    "generate_uuid",
    "get_blocks_as_dicts",
    "is_uuid",
]

UUID_REGEX = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[a-f0-9]{4}-[a-f0-9]{12}$"
)

AGENT_EXECUTOR_BLOCK_ID = "e189baac-8c20-45a1-94a7-55177ea42565"
AGENT_INPUT_BLOCK_ID = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b"
AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4"


def is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    return isinstance(value, str) and UUID_REGEX.match(value) is not None


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())
