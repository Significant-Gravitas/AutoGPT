from typing import List, Optional
import json
import openai
# Insert AI endpoint library
from ai_function_lib import call_ai_function

# This is a magic function that can do anything with no-code. See
# https://github.com/Torantulino/AI-Functions for more info.



# The magic function was moved to ai_function_lib.py. It is the call_ai_function library that is now loaded in import.

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
