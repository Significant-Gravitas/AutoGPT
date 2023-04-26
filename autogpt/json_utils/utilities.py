"""Utilities for the json_fixes package."""
import json
import os.path
import re

from jsonschema import Draft7Validator

from autogpt.config import Config
from autogpt.logs import logger

CFG = Config()


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


def validate_json(json_object: object, schema_name: object) -> object:
    """
    :type schema_name: object
    :param schema_name:
    :type json_object: object
    """
    script_dir = os.path.dirname(__file__)

    with open(os.path.join(script_dir, f"{schema_name}.json"), "r") as f:
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
    elif CFG.debug_mode:
        print("The JSON object is valid.")

    return json_object
