"""Utilities for the json_fixes package."""

import io
import logging
import re
from typing import Any

import demjson3

logger = logging.getLogger(__name__)


def json_loads(json_str: str) -> Any:
    """Parse a JSON string, this function is tolerant
    to minor issues in the JSON string:
    - Missing commas between elements
    - Trailing commas or extra commas in objects
    - Extraneous newlines and spaces outside of string literals
    - Inconsistent spacing after colons and commas
    - Missing closing brackets or braces
    - Comments

    Args:
        json_str: The JSON string to parse.

    Returns:
        The parsed JSON object, same as built-in json.loads.
    """
    error_buffer = io.StringIO()
    json_result = demjson3.decode(
        json_str, return_errors=True, write_errors=error_buffer
    )

    if error_buffer.getvalue():
        logger.debug(f"JSON parse errors:\n{error_buffer.getvalue()}")

    if json_result is None:
        raise ValueError(f"Failed to parse JSON string: {json_str}")

    return json_result.object


def extract_dict_from_response(response_content: str) -> dict[str, Any]:
    # Sometimes the response includes the JSON in a code block with ```
    pattern = r"```(?:json|JSON)*([\s\S]*?)```"
    match = re.search(pattern, response_content)

    if match:
        response_content = match.group(1).strip()
    else:
        # The string may contain JSON.
        json_pattern = r"{[\s\S]*}"
        match = re.search(json_pattern, response_content)

        if match:
            response_content = match.group()

    result = json_loads(response_content)
    if not isinstance(result, dict):
        raise ValueError(
            f"Response '''{response_content}''' evaluated to "
            f"non-dict value {repr(result)}"
        )
    return result


def extract_list_from_response(response_content: str) -> list[Any]:
    # Sometimes the response includes the JSON in a code block with ```
    pattern = r"```(?:json|JSON)*([\s\S]*?)```"
    match = re.search(pattern, response_content)

    if match:
        response_content = match.group(1).strip()
    else:
        # The string may contain JSON.
        json_pattern = r"\[[\s\S]*\]"
        match = re.search(json_pattern, response_content)

        if match:
            response_content = match.group()

    result = json_loads(response_content)
    if not isinstance(result, list):
        raise ValueError(
            f"Response '''{response_content}''' evaluated to "
            f"non-list value {repr(result)}"
        )
    return result
