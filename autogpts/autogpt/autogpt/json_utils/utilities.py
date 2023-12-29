"""Utilities for the json_fixes package."""
import ast
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_dict_from_response(response_content: str) -> dict[str, Any]:
    # Sometimes the response includes the JSON in a code block with ```
    pattern = r"```([\s\S]*?)```"
    match = re.search(pattern, response_content)

    if match:
        response_content = match.group(1).strip()
        # Remove language names in code blocks
        response_content = response_content.lstrip("json")
    else:
        # The string may contain JSON.
        json_pattern = r"{.*}"
        match = re.search(json_pattern, response_content)

        if match:
            response_content = match.group()

    # Response content comes from OpenAI as a Python `str(content_dict)`.
    # `literal_eval` does the reverse of `str(dict)`.
    result = ast.literal_eval(response_content)
    if not isinstance(result, dict):
        raise ValueError(
            f"Response '''{response_content}''' evaluated to "
            f"non-dict value {repr(result)}"
        )
    return result
