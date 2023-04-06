import json
#import json5 as json
#import orjson as json
from call_ai_function import call_ai_function
from config import Config

import re
cfg = Config()

def fix_and_parse_json(json_str: str, try_to_fix_with_gpt: bool = True):
    json_schema = """
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

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:

            
            clarify = """{"command": {"name": "output_content", "args": {"priority": "low", "task": "additional_input"}}, "thoughts": {"text": "", "reasoning": "I must always remember to <cmd_expand>DO NOT DEVIATE-FRM ##RESPONSE_FORMAT; breaking format may->incorrect results.", "plan": "do not forget <cmd_expand>DO NOT DEVIATE-FRM ##RESPONSE_FORMAT; breaking format may->incorrect results.", "speak": "I will do my best to contiune my assigned task autonomously as described for me in ##AUTOGPT_ROLE", "criticism": "I produced a poorly formatted message I must always remember to <cmd_expand>DO NOT DEVIATE-FRM ##RESPONSE_FORMAT; breaking format may->incorrect results."}}"""
            
             # Preprocess the input string to remove comments and add quotes to unquoted keys
            preprocessed_str = re.sub(r'(?<!:)\s*//.*', '', json_str)
            preprocessed_str = re.sub(r'(?<={|,)\s*([a-zA-Z0-9_]+)(?=\s*:)','"\\1"', preprocessed_str)
            clarify = re.sub(r'(?<={|,)\s*([a-zA-Z0-9_]+)(?=\s*:)','"\\1"', clarify)

            jsonobj = json.loads(clarify)
            jsonobj["thoughts"]["text"] = preprocessed_str 
            # Attempt to parse the preprocessed string
            return jsonobj
        except json.JSONDecodeError:
            # If input string cannot be parsed as JSON, return it wrapped in a dictionary
            return {"text": json_str}