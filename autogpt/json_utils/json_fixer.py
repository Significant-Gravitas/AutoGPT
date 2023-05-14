import contextlib
import json
from typing import Any, Dict, Union

from regex import regex

class JsonFixer:
    JSON_SCHEMA = """
    {
        "braindump": "Dump your verbose thoughts here",
        "command": {
            "name": "command name",
            "args": {
                "arg name": "value"
            }
        },
        "key_updates": {
            "essence": "A phrase boiling down the essence of the current task",
            "reasoning": "reasoning",
             "plan": "- short bulleted\n- list that conveys\n- long-term plan",
            "criticism": "constructive self-criticism",
            "big_picture": "big picture alignment check"
        },
    }
    """

    @staticmethod
    def fix_json(json_string: str, schema: str = None) -> str:
        # Add logic to fix JSON string based on different techniques
        fixed_json = JsonFixer.attempt_to_fix_json_by_finding_outermost_brackets(json_string)
        if fixed_json:
            return json.dumps(fixed_json)
        return json_string

    @staticmethod
    def fix_and_parse_json(json_to_load: Union[str, Dict]) -> Dict[Any, Any]:
        if isinstance(json_to_load, dict):
            return json_to_load

        if not json_to_load.strip():
            return {}

        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(json_to_load)

        fixed_json = JsonFixer.fix_json(json_to_load, JsonFixer.JSON_SCHEMA)
        return json.loads(fixed_json)

    @staticmethod
    def is_valid_json(json_string: str) -> bool:
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def attempt_to_fix_json_by_finding_outermost_brackets(json_string: str) -> Dict[Any, Any]:
        json_pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
        json_match = json_pattern.search(json_string)

        if json_match:
            fixed_json = json_match.group(0)
            if JsonFixer.is_valid_json(fixed_json):
                return json.loads(fixed_json)

        return {}
