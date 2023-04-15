"""This module contains the function to fix JSON strings using GPT-3."""
import json
from autogpt.llm_utils import call_ai_function
from autogpt.logs import logger


def fix_json(json_str: str, schema: str) -> str:
    """Fix the given JSON string to make it parseable and fully compliant with the provided schema."""
    function_string = "def fix_json(json_str: str, schema:str=None) -> str:"
    args = [f"'''{json_str}'''", f"'''{schema}'''"]
    description_string = (
        "Fixes the provided JSON string to make it parseable"
        " and fully compliant with the provided schema.\n If an object or"
        " field specified in the schema isn't contained within the correct"
        " JSON, it is omitted.\n This function is brilliant at guessing"
        " when the format is incorrect."
    )

    json_str = ensure_code_block(json_str)

    result_string = call_ai_function(
        function_string, args, description_string, model=cfg.fast_llm_model
    )

    log_attempt(json_str, result_string)

    try:
        json.loads(result_string)  # just check the validity
        return result_string
    except json.JSONDecodeError:
        return "failed"


def ensure_code_block(json_str: str) -> str:
    """Ensure the JSON string is wrapped in a code block."""
    if not json_str.startswith("```"):
        return f"```json\n{json_str}\n```"
    return json_str


def log_attempt(original_json: str, fixed_json: str) -> None:
    """Log JSON fix attempt."""
    logger.debug("------------ JSON FIX ATTEMPT ---------------")
    logger.debug(f"Original JSON: {original_json}")
    logger.debug("-----------")
    logger.debug(f"Fixed JSON: {fixed_json}")
    logger.debug("----------- END OF FIX ATTEMPT ----------------")
