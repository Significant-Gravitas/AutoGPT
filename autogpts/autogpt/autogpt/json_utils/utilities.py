"""Utilities for the json_fixes package."""
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


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

    result = json.loads(response_content)
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

    result = json.loads(response_content)
    if not isinstance(result, list):
        raise ValueError(
            f"Response '''{response_content}''' evaluated to "
            f"non-list value {repr(result)}"
        )
    return result
