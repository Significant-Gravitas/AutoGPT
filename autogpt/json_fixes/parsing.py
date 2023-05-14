"""Fix and parse JSON strings."""
from __future__ import annotations

import contextlib
import json
import re
from typing import Any, Tuple, Union

from autogpt.config import Config
from autogpt.json_utils.json_fix_general import add_quotes_to_property_names, balance_braces, fix_invalid_escape
from autogpt.json_utils.json_fix_llm import fix_json_using_multiple_techniques
from autogpt.logs import logger

CFG = Config()

class IgnoreTrailingCommasDecoder(json.JSONDecoder):
    def decode(self, s, _w=None):
        s = s.replace(",}", "}")
        s = s.replace(",]", "]")
        return super(IgnoreTrailingCommasDecoder, self).decode(s, _w)

def loads_ignore_trailing_commas(s):
    return json.loads(s, cls=IgnoreTrailingCommasDecoder)


def correct_json(json_to_load: str) -> str:
    """
    Correct common JSON errors.

    Args:
        json_to_load (str): The JSON string.
    """

    try:
        if CFG.debug_mode:
            print("json", json_to_load)
        json.loads(json_to_load)
        return json_to_load
    except json.JSONDecodeError as e:
        if CFG.debug_mode:
            print("json loads error", e)
        error_message = str(e)
        if error_message.startswith("Invalid \\escape"):
            json_to_load = fix_invalid_escape(json_to_load, error_message)
        if error_message.startswith(
            "Expecting property name enclosed in double quotes"
        ):
            json_to_load = add_quotes_to_property_names(json_to_load)
            try:
                json.loads(json_to_load)
                return json_to_load
            except json.JSONDecodeError as e:
                if CFG.debug_mode:
                    print("json loads error - add quotes", e)
                error_message = str(e)
        if balanced_str := balance_braces(json_to_load):
            return balanced_str
    return json_to_load


def fix_and_parse_json(
    json_to_load: str, try_to_fix_with_gpt: bool = True
) -> str | dict[Any, Any]:
    """Fix and parse JSON string

    Args:
        json_to_load (str): The JSON string.
        try_to_fix_with_gpt (bool, optional): Try to fix the JSON with GPT.
            Defaults to True.

    Returns:
        str or dict[Any, Any]: The parsed JSON.
    """

    # print(json_to_load)
    json_to_load = extract_code_block(json_to_load)

    if isinstance(json_to_load, dict):
        json_to_load = json.dumps(json_to_load)

    with contextlib.suppress(json.JSONDecodeError):
        json_to_load = json_to_load.replace("\t", "")
        return loads_ignore_trailing_commas(json_to_load)

    with contextlib.suppress(json.JSONDecodeError):
        json_to_load = correct_json(json_to_load)
        return loads_ignore_trailing_commas(json_to_load)
    # Let's do something manually:
    # sometimes GPT responds with something BEFORE the braces:
    # "I'm sorry, I don't understand. Please try again."
    # {"text": "I'm sorry, I don't understand. Please try again.",
    #  "confidence": 0.0}
    # So let's try to find the first brace and then parse the rest
    #  of the string
    try:
        brace_index = json_to_load.index("{")
        maybe_fixed_json = json_to_load[brace_index:]
        last_brace_index = maybe_fixed_json.rindex("}")
        maybe_fixed_json = maybe_fixed_json[: last_brace_index + 1]
        return json.loads(maybe_fixed_json)
    except (json.JSONDecodeError, ValueError) as e:
        return try_ai_fix(try_to_fix_with_gpt, e, json_to_load)


def extract_code_block(response: str) -> str:
    code_block_pattern = r"```(.*?)```"
    code_block_match = re.search(code_block_pattern, response, re.DOTALL)

    if code_block_match:
        return code_block_match.group(1).strip()
    else:
        return response


def try_ai_fix(
    try_to_fix_with_gpt: bool, exception: Exception, json_to_load: str
) -> str | dict[Any, Any]:
    """Try to fix the JSON with the AI

    Args:
        try_to_fix_with_gpt (bool): Whether to try to fix the JSON with the AI.
        exception (Exception): The exception that was raised.
        json_to_load (str): The JSON string to load.

    Raises:
        exception: If try_to_fix_with_gpt is False.

    Returns:
        str or dict[Any, Any]: The JSON string or dictionary.
    """
    if not try_to_fix_with_gpt:
        raise exception

    logger.warn(
        "Warning: Failed to parse AI output, attempting to fix."
        "\n If you see this warning frequently, it's likely that"
        " your prompt is confusing the AI. Try changing it up"
        " slightly."
    )
    # Now try to fix this up using the ai_functions
    ai_fixed_json = fix_json_using_multiple_techniques(json_to_load, JSON_SCHEMA)

    if ai_fixed_json != "failed":
        return json.loads(ai_fixed_json)
    # This allows the AI to react to the error message,
    #   which usually results in it correcting its ways.
    logger.error("Failed to fix AI output, telling the AI.")
    return json_to_load


def extract_json_and_nl(response: str) -> Tuple[str, Union[dict, None]]:
        separator = "|||"
        if separator not in response:
            return response.strip(), None

        nl_part, json_part = response.split(separator, 1)
        try:
            json_data = json.loads(json_part)
        except json.JSONDecodeError:
            json_data = None

        return nl_part.strip(), json_data


