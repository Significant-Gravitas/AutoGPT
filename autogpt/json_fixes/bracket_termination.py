"""Fix JSON brackets."""
import contextlib
import json
from typing import Optional
import regex
from colorama import Fore

from autogpt.logs import logger
from autogpt.config import Config
from autogpt.speech import say_text

CFG = Config()

def contains_json(json_string: str):
    json_pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
    json_match = json_pattern.search(json_string)
    if json_match:
         return (True, json_match.group(0))
    else:
        return (False, None)

def is_valid_json(string):
    try:
        json.loads(string)
        return True
    except ValueError:
        return False

def attempt_to_fix_json_by_finding_outermost_brackets(json_string: str):
    if CFG.speak_mode and CFG.debug_mode:
        say_text(
            "I have received an invalid JSON response from the OpenAI API. "
            "Trying to fix it now."
        )
    logger.typewriter_log("Attempting to fix JSON by finding outermost brackets\n")

    try:
        is_contains_json, the_json = contains_json(json_string)
        if is_contains_json and is_valid_json(the_json):
            # Extract the valid JSON object from the string
            json_string = the_json
            logger.typewriter_log(
                title="Apparently json was fixed.", title_color=Fore.GREEN
            )
            if CFG.speak_mode and CFG.debug_mode:
                say_text("Apparently json was fixed.")

        elif contains_json:
            pattern = regex.compile(",\\n *}") # 80% of the returned json has commas after the last element of an array that disqualifies it as valid json.
            json_string = pattern.sub(" }", the_json)
            
        else:
            if CFG.speak_mode and CFG.debug_mode:
                say_text("No valid JSON object found")
            raise ValueError("No valid JSON object found")
        
    except (json.JSONDecodeError, ValueError):
        if CFG.debug_mode:
            logger.error(f"Error: Invalid JSON: {json_string}\n")
        if CFG.speak_mode:
            say_text("Didn't work. I will have to ignore this response then.")
        logger.error("Error: Invalid JSON, setting it to empty JSON now.\n")
        json_string = {}

    return json_string


def balance_braces(json_string: str) -> Optional[str]:
    """
    Balance the braces in a JSON string.

    Args:
        json_string (str): The JSON string.

    Returns:
        str: The JSON string with braces balanced.
    """

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
