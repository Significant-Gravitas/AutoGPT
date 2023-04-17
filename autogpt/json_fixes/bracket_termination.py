import json
import contextlib
import regex
from typing import Optional
from jsonschema import validate, ValidationError
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "args": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["name", "args"],
        },
        "thoughts": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "reasoning": {"type": "string"},
                "plan": {"type": "string"},
                "criticism": {"type": "string"},
                "speak": {"type": "string"},
            },
            "required": ["text", "reasoning", "plan", "criticism", "speak"],
        },
    },
    "required": ["command", "thoughts"],
}


def attempt_to_fix_json_by_finding_outermost_brackets(json_string: str):
    logger.debug("Attempting to fix JSON by finding outermost brackets")
    json_pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
    json_match = json_pattern.search(json_string)

    if json_match:
        json_string = json_match.group(0)
        logger.debug("JSON was fixed.")
    else:
        logger.error("No valid JSON object found")
        json_string = "{}"

    return json_string


def balance_braces(json_string: str) -> Optional[str]:
    open_braces_count = json_string.count("{")
    close_braces_count = json_string.count("}")

    while open_braces_count > close_braces_count:
        json_string += "}"
        close_braces_count += 1

    while close_braces_count > open_braces_count:
        json_string = json_string.rstrip("}")
        close_braces_count -= 1

    with contextlib.suppress(json.JSONDecodeError):
        json.loads(json_string)
        return json_string


def fix_and_parse_json(json_to_load: str) -> Optional[dict]:
    with contextlib.suppress(json.JSONDecodeError):
        json_to_load = json_to_load.replace("\t", "")
        json_obj = json.loads(json_to_load)
        validate(instance=json_obj, schema=JSON_SCHEMA)
        return json_obj

    with contextlib.suppress(json.JSONDecodeError):
        json_to_load = balance_braces(json_to_load)
        json_obj = json.loads(json_to_load)
        validate(instance=json_obj, schema=JSON_SCHEMA)
        return json_obj

    json_to_load = attempt_to_fix_json_by_finding_outermost_brackets(json_to_load)
    try:
        json_obj = json.loads(json_to_load)
        validate(instance=json_obj, schema=JSON_SCHEMA)
        return json_obj
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error("Failed to fix and validate JSON: %s", e)
        return None


if __name__ == "__main__":
    sample_json = """
    {
        "command": {
            "name": "command name",
            "args": {
                "arg name": "value"
            }
        },
        "thoughts":
        {
            "text": "thought",
            "reasoning": "reasoning",
            "plan": "- short bulleted\n- list that conveys\n- long-term plan",
            "criticism": "constructive self-criticism",
            "speak": "thoughts summary to say to user"
        }
    }
    """

    corrupted_json = sample_json.replace(":", "").replace(",", "")
    fixed_json = fix_and_parse_json(corrupted_json)

    if fixed_json:
        print("Fixed JSON:", fixed_json)
    else:
        print("Failed to fix and parse JSON.")

