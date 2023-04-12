import json
from typing import Any, Dict, Union
from autogpt.call_ai_function import call_ai_function
from autogpt.config import Config
from autogpt.json_utils import correct_json
from autogpt.logger import logger


class JsonParser:
    JSON_SCHEMA = """
    {
        "command": {
            "name": "command name",
            "args":{
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

    def __init__(self, cfg: Config):
        self._cfg = cfg

    def fix_and_parse_json(self,
                           json_str: str,
                           try_to_fix_with_gpt: bool = True
                           ) -> Union[str, Dict[Any, Any]]:
        """Fix and parse JSON string"""

        json_transformations = [
            lambda json_string: json_string,
            lambda json_string: json_string.replace('\t', ''),
            correct_json,
            JsonParser._cut_to_braces
        ]

        if try_to_fix_with_gpt:
            json_transformations.append(self._fix_json_with_ai)

        last_error = None
        for transformation in json_transformations:
            try:
                json_str = transformation(json_str)
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.debug(f"Failed to parse JSON after transformation:\n{json_str}")
                pass

        # This allows the AI to react to the error message,
        #   which usually results in it correcting its ways.
        logger.error("Failed to fix AI output, telling the AI.")
        raise last_error

    @staticmethod
    def _cut_to_braces(json_str):
        # Let's do something manually:
        # sometimes GPT responds with something BEFORE the braces:
        # "I'm sorry, I don't understand. Please try again."
        # {"text": "I'm sorry, I don't understand. Please try again.",
        #  "confidence": 0.0}
        # So let's try to find the first brace and then parse the rest
        #  of the string
        brace_index = json_str.index("{")
        json_str = json_str[brace_index:]
        last_brace_index = json_str.rindex("}")
        json_str = json_str[:last_brace_index + 1]
        return json_str

    def _fix_json_with_ai(self, json_str: str) -> str:
        """Fix the given JSON string to make it parseable and fully compliant with the provided schema."""
        logger.warn("Warning: Failed to parse AI output, attempting to fix."
                    "\n If you see this warning frequently, it's likely that"
                    " your prompt is confusing the AI. Try changing it up"
                    " slightly.")

        # Try to fix the JSON using GPT:
        function_string = "def fix_json(json_str: str, schema:str=None) -> str:"
        args = [f"'''{json_str}'''", f"'''{JsonParser.JSON_SCHEMA}'''"]
        description_string = "Fixes the provided JSON string to make it parseable"\
            " and fully compliant with the provided schema.\n If an object or"\
            " field specified in the schema isn't contained within the correct"\
            " JSON, it is omitted.\n This function is brilliant at guessing"\
            " when the format is incorrect."

        # If it doesn't already start with a "`", add one:
        if not json_str.startswith("`"):
            json_str = "```json\n" + json_str + "\n```"
        result_string = call_ai_function(
            function_string, args, description_string, model=self._cfg.fast_llm_model
        )
        logger.debug("------------ JSON FIX ATTEMPT ---------------")
        logger.debug(f"Original JSON: {json_str}")
        logger.debug("-----------")
        logger.debug(f"Fixed JSON: {result_string}")
        logger.debug("----------- END OF FIX ATTEMPT ----------------")
