"""This module contains the function to fix JSON strings using GPT-3."""
import json

from autogpt.llm_utils import call_ai_function
from autogpt.logs import logger
from autogpt.config import Config

CFG = Config()


def fix_json(json_string: str, schema: str) -> str:
    """Fix the given JSON string to make it parseable and fully compliant with
        the provided schema.

    Args:
        json_string (str): The JSON string to fix.
        schema (str): The schema to use to fix the JSON.
    Returns:
        str: The fixed JSON string.
    """
    # Try to fix the JSON using GPT:
    function_string = "def fix_json(json_string: str, schema:str=None) -> str:"
    args = [f"'''{json_string}'''", f"'''{schema}'''"]
    description_string = (
        "This function takes a JSON string and ensures that it"
        " is parseable and fully compliant with the provided schema. If an object"
        " or field specified in the schema isn't contained within the correct JSON,"
        " it is omitted. The function also escapes any double quotes within JSON"
        " string values to ensure that they are valid. If the JSON string contains"
        " any None or NaN values, they are replaced with null before being parsed."
    )

    # If it doesn't already start with a "`", add one:
    if not json_string.startswith("`"):
        json_string = "```json\n" + json_string + "\n```"
    result_string = call_ai_function(
        function_string, args, description_string, model=CFG.fast_llm_model
    )
    logger.debug("------------ JSON FIX ATTEMPT ---------------")
    logger.debug(f"Original JSON: {json_string}")
    logger.debug("-----------")
    logger.debug(f"Fixed JSON: {result_string}")
    logger.debug("----------- END OF FIX ATTEMPT ----------------")

    try:
        json.loads(result_string)  # just check the validity
        return result_string
    except json.JSONDecodeError:  # noqa: E722
        # Get the call stack:
        # import traceback
        # call_stack = traceback.format_exc()
        # print(f"Failed to fix JSON: '{json_string}' "+call_stack)
        return "failed"
