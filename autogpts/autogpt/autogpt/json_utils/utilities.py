"""Utilities for the json_fixes package."""
import re
import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_dict_from_response(response_content: str) -> dict[str, Any]:
    # Sometimes the response includes the JSON in a code block with ```
    pattern = r'```([\s\S]*?)```'
    match = re.search(pattern, response_content)

    if match:
        response_content = match.group(1).strip()
        # Remove language names in code blocks
        response_content = response_content.lstrip("json")
    else:
        # The string may contain JSON.
        json_pattern = r'{.*}'
        match = re.search(json_pattern, response_content)

        if match:
            response_content = match.group()

    # response content comes from OpenAI as a Python `str(content_dict)`, literal_eval reverses this
    try:
        return ast.literal_eval(response_content)
    except BaseException as e:
        logger.info(f"Error parsing JSON response with literal_eval {e}")
        logger.debug(f"Invalid JSON received in response: {response_content}")
        # TODO: How to raise an error here without causing the program to exit?
        return {}
