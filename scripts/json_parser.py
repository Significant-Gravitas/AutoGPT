import json
from typing import Any, Dict, Union
from call_ai_function import call_ai_function
from config import Config
from json_utils import correct_json
from logger import logger

cfg = Config()

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


def fix_and_parse_json(
    json_str: str,
    try_to_fix_with_gpt: bool = True
) -> Union[str, Dict[Any, Any]]:
    """Fix and parse JSON string"""
    try:
        json_str = json_str.replace('\t', '')
        return json.loads(json_str)
    except json.JSONDecodeError as _:  # noqa: F841
        try:
            json_str = correct_json(json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as _:  # noqa: F841
            pass
    # Let's do something manually:
    # sometimes GPT responds with something BEFORE the braces:
    # "I'm sorry, I don't understand. Please try again."
    # {"text": "I'm sorry, I don't understand. Please try again.",
    #  "confidence": 0.0}
    # So let's try to find the first brace and then parse the rest
    #  of the string
    try:
        brace_index = json_str.index("{")
        json_str = json_str[brace_index:]
        last_brace_index = json_str.rindex("}")
        json_str = json_str[:last_brace_index+1]
        return json.loads(json_str)
    # Can throw a ValueError if there is no "{" or "}" in the json_str
    except (json.JSONDecodeError, ValueError) as e:  # noqa: F841
        if try_to_fix_with_gpt:
            logger.warn("警告：解析人工智能输出失败，正在尝试修复。"
                    "\n 如果您经常看到此警告，则可能是您的提示混淆了人工智能。"
                    "尝试稍微更改提示。")
            # Now try to fix this up using the ai_functions
            ai_fixed_json = fix_json(json_str, JSON_SCHEMA)

            if ai_fixed_json != "failed":
                return json.loads(ai_fixed_json)
            else:
                # This allows the AI to react to the error message,
                #   which usually results in it correcting its ways.
                logger.error("无法修复 AI 输出，告诉 AI 让它修复。")
                return json_str
        else:
            raise e


def fix_json(json_str: str, schema: str) -> str:
    """修复给定的 JSON 字符串，使其可解析并完全符合提供的模式。"""
    # Try to fix the JSON using GPT:
    function_string = "def fix_json(json_str: str, schema:str=None) -> str:"
    args = [f"'''{json_str}'''", f"'''{schema}'''"]
    description_string = "修复提供的 JSON 字符串以使其可解析"\
         " 并完全符合提供的架构。\n 如果一个对象或"\
         "架构中指定的字段未包含在正确的"\
         " JSON, 省略了。\n 这个函数很擅长猜测"\
         "当格式不正确时。"

    # If it doesn't already start with a "`", add one:
    if not json_str.startswith("`"):
        json_str = "```json\n" + json_str + "\n```"
    result_string = call_ai_function(
        function_string, args, description_string, model=cfg.fast_llm_model
    )
    logger.debug("------------ 尝试修复JSON ---------------")
    logger.debug(f"原始 JSON: {json_str}")
    logger.debug("-----------")
    logger.debug(f"修复后 JSON: {result_string}")
    logger.debug("----------- 尝试修复结束 ----------------")

    try:
        json.loads(result_string)  # just check the validity
        return result_string
    except:  # noqa: E722
        # Get the call stack:
        # import traceback
        # call_stack = traceback.format_exc()
        # print(f"Failed to fix JSON: '{json_str}' "+call_stack)
        return "failed"
