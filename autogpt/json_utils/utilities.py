"""Utilities for the json_fixes package."""
import ast
import json
import os.path
from typing import Any

from jsonschema import Draft7Validator

from autogpt.config import Config
from autogpt.logs import logger

LLM_DEFAULT_RESPONSE_FORMAT = "llm_response_format_1"


def extract_json_from_response(response_content: str) -> dict:
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


def llm_response_schema(
    config: Config, schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT
) -> dict[str, Any]:
    filename = os.path.join(os.path.dirname(__file__), f"{schema_name}.json")
    with open(filename, "r") as f:
        json_schema = json.load(f)
    if config.openai_functions:
        del json_schema["properties"]["command"]
        json_schema["required"].remove("command")
    return json_schema


def validate_json(
    json_object: object, config: Config, schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT
) -> bool:
    """
    :type schema_name: object
    :param schema_name: str
    :type json_object: object

    Returns:
        bool: Whether the json_object is valid or not
    """
    schema = llm_response_schema(config, schema_name)
    validator = Draft7Validator(schema)

    if errors := sorted(validator.iter_errors(json_object), key=lambda e: e.path):
        for error in errors:
            logger.debug(f"JSON Validation Error: {error}")

        if config.debug_mode:
            logger.error(
                json.dumps(json_object, indent=4)
            )  # Replace 'json_object' with the variable containing the JSON data
            logger.error("The following issues were found:")

            for error in errors:
                logger.error(f"Error: {error.message}")
        return False

    logger.debug("The JSON object is valid.")

    return True
