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

            
            clarify = """{"command": {"name": "operator_clarification", "args": {"priority": "high", "task": "additional_input"}}, "thoughts": {"text": "", "reasoning": "Completing the task with the present data may lead to inaccurate or unreliable results. Obtaining further input or clarification is crucial for producing the desired outcome.", "plan": "Initiate an operator_clarification process to acquire the required input or clarifying details. This will ensure that the data processing and analysis can continue in a correct and effective manner.", "speak": "Please provide the necessary additional input or clarifying details to facilitate accurate and comprehensive processing of your data.", "criticism": "In the current configuration, there might be a risk of misunderstanding the operator's needs or missing crucial details. It is essential to establish clear communication channels and be specific about the input or clarification required."}}"""
            
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