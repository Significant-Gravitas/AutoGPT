"""Shared helpers for agent generation."""

import re
import uuid
from typing import Any

from backend.data.dynamic_fields import DICT_SPLIT

from .blocks import get_blocks_as_dicts

__all__ = [
    "AGENT_EXECUTOR_BLOCK_ID",
    "AGENT_INPUT_BLOCK_ID",
    "AGENT_OUTPUT_BLOCK_ID",
    "AgentDict",
    "MCP_TOOL_BLOCK_ID",
    "TOOL_ORCHESTRATOR_BLOCK_ID",
    "UUID_REGEX",
    "are_types_compatible",
    "generate_uuid",
    "get_blocks_as_dicts",
    "get_defined_property_type",
    "is_uuid",
]


# Type alias for the agent JSON structure passed through
# the validation and fixing pipeline.
AgentDict = dict[str, Any]

# Shared base pattern (unanchored, lowercase hex); used for both full-string
# validation (UUID_REGEX) and text extraction (core._UUID_PATTERN).
UUID_RE_STR = r"[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[a-f0-9]{4}-[a-f0-9]{12}"

UUID_REGEX = re.compile(r"^" + UUID_RE_STR + r"$")

AGENT_EXECUTOR_BLOCK_ID = "e189baac-8c20-45a1-94a7-55177ea42565"
MCP_TOOL_BLOCK_ID = "a0a4b1c2-d3e4-4f56-a7b8-c9d0e1f2a3b4"
TOOL_ORCHESTRATOR_BLOCK_ID = "3b191d9f-356f-482d-8238-ba04b6d18381"
AGENT_INPUT_BLOCK_ID = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b"
AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4"


def is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    return isinstance(value, str) and UUID_REGEX.match(value) is not None


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def get_defined_property_type(schema: dict[str, Any], name: str) -> str | None:
    """Get property type from a schema, handling nested `_#_` notation."""
    if DICT_SPLIT in name:
        parent, child = name.split(DICT_SPLIT, 1)
        parent_schema = schema.get(parent, {})
        if "properties" in parent_schema and isinstance(
            parent_schema["properties"], dict
        ):
            return parent_schema["properties"].get(child, {}).get("type")
        return None
    return schema.get(name, {}).get("type")


def are_types_compatible(src: str, sink: str) -> bool:
    """Check if two schema types are compatible."""
    if {src, sink} <= {"integer", "number"}:
        return True
    return src == sink
