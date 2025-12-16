"""Utilities for agent generation."""

import json
import re
from typing import Any

from backend.data.block import get_blocks

# UUID validation regex
UUID_REGEX = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$"
)

# Block IDs for various fixes
STORE_VALUE_BLOCK_ID = "1ff065e9-88e8-4358-9d82-8dc91f622ba9"
CONDITION_BLOCK_ID = "715696a0-e1da-45c8-b209-c2fa9c3b0be6"
ADDTOLIST_BLOCK_ID = "aeb08fc1-2fc1-4141-bc8e-f758f183a822"
ADDTODICTIONARY_BLOCK_ID = "31d1064e-7446-4693-a7d4-65e5ca1180d1"
CREATELIST_BLOCK_ID = "a912d5c7-6e00-4542-b2a9-8034136930e4"
CREATEDICT_BLOCK_ID = "b924ddf4-de4f-4b56-9a85-358930dcbc91"
CODE_EXECUTION_BLOCK_ID = "0b02b072-abe7-11ef-8372-fb5d162dd712"
DATA_SAMPLING_BLOCK_ID = "4a448883-71fa-49cf-91cf-70d793bd7d87"
UNIVERSAL_TYPE_CONVERTER_BLOCK_ID = "95d1b990-ce13-4d88-9737-ba5c2070c97b"
GET_CURRENT_DATE_BLOCK_ID = "b29c1b50-5d0e-4d9f-8f9d-1b0e6fcbf0b1"

DOUBLE_CURLY_BRACES_BLOCK_IDS = [
    "44f6c8ad-d75c-4ae1-8209-aad1c0326928",  # FillTextTemplateBlock
    "6ab085e2-20b3-4055-bc3e-08036e01eca6",
    "90f8c45e-e983-4644-aa0b-b4ebe2f531bc",
    "363ae599-353e-4804-937e-b2ee3cef3da4",  # AgentOutputBlock
    "3b191d9f-356f-482d-8238-ba04b6d18381",
    "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
    "3a7c4b8d-6e2f-4a5d-b9c1-f8d23c5a9b0e",
    "ed1ae7a0-b770-4089-b520-1f0005fad19a",
    "a892b8d9-3e4e-4e9c-9c1e-75f8efcf1bfa",
    "b29c1b50-5d0e-4d9f-8f9d-1b0e6fcbf0b1",
    "716a67b3-6760-42e7-86dc-18645c6e00fc",
    "530cf046-2ce0-4854-ae2c-659db17c7a46",
    "ed55ac19-356e-4243-a6cb-bc599e9b716f",
    "1f292d4a-41a4-4977-9684-7c8d560b9f91",  # LLM blocks
    "32a87eab-381e-4dd4-bdb8-4c47151be35a",
]


def is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID v4."""
    return isinstance(value, str) and UUID_REGEX.match(value) is not None


def get_block_summaries(include_schemas: bool = True) -> str:
    """Generate block summaries for prompts.

    Args:
        include_schemas: Whether to include full input/output schemas

    Returns:
        Formatted string of block summaries
    """
    blocks = get_blocks()
    summaries = []

    for block_id, block_cls in blocks.items():
        block = block_cls()
        name = block.name
        desc = getattr(block, "description", "") or ""

        if not include_schemas:
            # Simple format
            if len(desc) > 200:
                desc = desc[:197] + "..."
            summaries.append(f"- {name} (id: {block_id}): {desc}")
        else:
            # Full format with schemas
            input_schema = {}
            output_schema = {}

            if hasattr(block, "input_schema"):
                try:
                    full_schema = block.input_schema.jsonschema()
                    input_schema = {
                        "properties": full_schema.get("properties", {}),
                        "required": full_schema.get("required", []),
                    }
                except Exception:
                    pass

            if hasattr(block, "output_schema"):
                try:
                    full_schema = block.output_schema.jsonschema()
                    output_schema = {
                        "properties": full_schema.get("properties", {}),
                    }
                except Exception:
                    pass

            block_info = {
                "name": name,
                "id": block_id,
                "description": desc[:500] if len(desc) > 500 else desc,
                "inputSchema": input_schema,
                "outputSchema": output_schema,
            }

            # Check for static output
            if getattr(block, "static_output", False):
                block_info["staticOutput"] = True

            summaries.append(json.dumps(block_info, indent=2))

    if include_schemas:
        return "\n\n".join(summaries)
    return "\n".join(summaries)


def get_blocks_info() -> list[dict[str, Any]]:
    """Get block information with schemas for validation and fixing."""
    blocks = get_blocks()
    blocks_info = []
    for block_id, block_cls in blocks.items():
        block = block_cls()
        blocks_info.append(
            {
                "id": block_id,
                "name": block.name,
                "description": getattr(block, "description", ""),
                "categories": getattr(block, "categories", []),
                "staticOutput": getattr(block, "static_output", False),
                "inputSchema": (
                    block.input_schema.jsonschema()
                    if hasattr(block, "input_schema")
                    else {}
                ),
                "outputSchema": (
                    block.output_schema.jsonschema()
                    if hasattr(block, "output_schema")
                    else {}
                ),
            }
        )
    return blocks_info


def parse_json_from_llm(text: str) -> dict[str, Any] | None:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    if not text:
        return None

    # Try fenced code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try raw text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try finding {...} span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Try finding [...] span
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None
