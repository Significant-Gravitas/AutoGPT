"""Utilities for the json_fixes package."""
import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_dict_from_response(response_content: str) -> dict[str, Any]:
    # Sometimes the response includes the JSON in a code block with ```
    if response_content.startswith("```") and response_content.endswith("```"):
        # Discard the first and last ```, then re-join in case the response naturally included ```
        response_content = "```".join(response_content.split("```")[1:-1])

    # response content comes from OpenAI as a Python `str(content_dict)`, literal_eval reverses this
    try:
        return ast.literal_eval(response_content)
    except BaseException as e:
        logger.info(f"Error parsing JSON response with literal_eval {e}")
        logger.debug(f"Invalid JSON received in response: {response_content}")
        # TODO: How to raise an error here without causing the program to exit?
        return {}
