"""Utilities for the json_fixes package."""
import json
import os.path
import re

from jsonschema import Draft7Validator

from autogpt.config import Config
from autogpt.logs import logger

CFG = Config()
LLM_DEFAULT_RESPONSE_FORMAT = "llm_response_format_1"


def extract_char_position(error_message: str) -> int:
    """Extract the character position from the JSONDecodeError message.

    Args:
        error_message (str): The error message from the JSONDecodeError
          exception.

    Returns:
        int: The character position.
    """

    char_pattern = re.compile(r"\(char (\d+)\)")
    if match := char_pattern.search(error_message):
        return int(match[1])
    else:
        raise ValueError("Character position not found in the error message.")


def validate_json(json_object: object, schema_name: str) -> dict | None:
    """
    :type schema_name: object
    :param schema_name: str
    :type json_object: object
    """
    scheme_file = os.path.join(os.path.dirname(__file__), f"{schema_name}.json")
    with open(scheme_file, "r") as f:
        schema = json.load(f)
    validator = Draft7Validator(schema)

    if errors := sorted(validator.iter_errors(json_object), key=lambda e: e.path):
        logger.error("The JSON object is invalid.")
        if CFG.debug_mode:
            logger.error(
                json.dumps(json_object, indent=4)
            )  # Replace 'json_object' with the variable containing the JSON data
            logger.error("The following issues were found:")

            for error in errors:
                logger.error(f"Error: {error.message}")
    else:
        logger.debug("The JSON object is valid.")

    return json_object


def validate_json_string(json_string: str, schema_name: str) -> dict | None:
    """
    :type schema_name: object
    :param schema_name: str
    :type json_object: object
    """

    try:
        json_loaded = json.loads(json_string)
        return validate_json(json_loaded, schema_name)
    except:
        return None


def is_string_valid_json(json_string: str, schema_name: str) -> bool:
    """
    :type schema_name: object
    :param schema_name: str
    :type json_object: object
    """

    return validate_json_string(json_string, schema_name) is not None
