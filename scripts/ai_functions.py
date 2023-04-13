from typing import List, Optional
import json
from config import Config
from call_ai_function import call_ai_function
from json_parser import fix_and_parse_json
cfg = Config()


def evaluate_code(code: str) -> List[str]:
    """
    A function that takes in a string and returns a response from create chat completion api call.

    Parameters:
        code (str): Code to be evaluated.
    Returns:
        A result string from create chat completion. A list of suggestions to improve the code.
    """

    function_string = "def analyze_code(code: str) -> List[str]:"
    args = [code]
    description_string = """Analyzes the given code and returns a list of suggestions for improvements."""

    result_string = call_ai_function(function_string, args, description_string)

    return result_string


def improve_code(suggestions: List[str], code: str) -> str:
    """
    A function that takes in code and suggestions and returns a response from create chat completion api call.

    Parameters:
        suggestions (List): A list of suggestions around what needs to be improved.
        code (str): Code to be improved.
    Returns:
        A result string from create chat completion. Improved code in response.
    """

    function_string = (
        "def generate_improved_code(suggestions: List[str], code: str) -> str:"
    )
    args = [json.dumps(suggestions), code]
    description_string = """Improves the provided code based on the suggestions provided, making no other changes."""

    result_string = call_ai_function(function_string, args, description_string)
    return result_string

def write_tests(code: str, focus: List[str]) -> str:
    """
    A function that takes in code and focus topics and returns a response from create chat completion api call.

    Parameters:
        focus (List): A list of suggestions around what needs to be improved.
        code (str): Code for test cases to be generated against.
    Returns:
        A result string from create chat completion. Test cases for the submitted code in response.
    """

    function_string = (
        "def create_test_cases(code: str, focus: Optional[str] = None) -> str:"
    )
    args = [code, json.dumps(focus)]
    description_string = """Generates test cases for the existing code, focusing on specific areas if required."""

    result_string = call_ai_function(function_string, args, description_string)
    return result_string
