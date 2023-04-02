from typing import List, Optional
import json
import openai
import dirtyjson
from config import Config

cfg = Config()

# This is a magic function that can do anything with no-code. See
# https://github.com/Torantulino/AI-Functions for more info.
def call_ai_function(function, args, description, model=cfg.smart_llm_model):
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma seperated string
    args = ", ".join(args)
    messages = [
        {
            "role": "system",
            "content": f"You are now the following python function: ```# {description}\n{function}```\n\nOnly respond with your `return` value.",
        },
        {"role": "user", "content": args},
    ]

    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0
    )

    return response.choices[0].message["content"]


# Evaluating code


def evaluate_code(code: str) -> List[str]:
    function_string = "def analyze_code(code: str) -> List[str]:"
    args = [code]
    description_string = """Analyzes the given code and returns a list of suggestions for improvements."""

    result_string = call_ai_function(function_string, args, description_string)
    return json.loads(result_string)


# Improving code


def improve_code(suggestions: List[str], code: str) -> str:
    function_string = (
        "def generate_improved_code(suggestions: List[str], code: str) -> str:"
    )
    args = [json.dumps(suggestions), code]
    description_string = """Improves the provided code based on the suggestions provided, making no other changes."""

    result_string = call_ai_function(function_string, args, description_string)
    return result_string


# Writing tests


def write_tests(code: str, focus: List[str]) -> str:
    function_string = (
        "def create_test_cases(code: str, focus: Optional[str] = None) -> str:"
    )
    args = [code, json.dumps(focus)]
    description_string = """Generates test cases for the existing code, focusing on specific areas if required."""

    result_string = call_ai_function(function_string, args, description_string)
    return result_string


# TODO: Make debug a global config var
def fix_json(json_str: str, schema:str = None, debug=True) -> str:
    # Try to fix the JSON using gpt:
    function_string = "def fix_json(json_str: str, schema:str=None) -> str:"
    args = [json_str, schema]
    description_string = """Fixes the provided JSON string to make it parseable. If the schema is provided, the JSON will be made to look like the schema, otherwise it will be made to look like a valid JSON object."""

    result_string = call_ai_function(
        function_string, args, description_string, model=cfg.fast_llm_model
    )
    if debug:
        print("------------ JSON FIX ATTEMPT ---------------")
        print(f"Original JSON: {json_str}")
        print(f"Fixed JSON: {result_string}")
        print("----------- END OF FIX ATTEMPT ----------------")
    try:
        return dirtyjson.loads(result_string)
    except:
        # Log the exception:
        print("Failed to fix JSON")
        # Get the call stack:
        import traceback
        call_stack = traceback.format_exc()
        print(call_stack)
        return {}