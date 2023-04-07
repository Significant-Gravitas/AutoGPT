from typing import List
import json
from config import Config
from call_ai_function import call_ai_function
from auto_gpt.commands import command

cfg = Config()

# Evaluating code
@command("evaluate_code", "Evaluate Code", '"code": "<full _code_string>"')
def evaluate_code(code: str) -> List[str]:
    function_string = "def analyze_code(code: str) -> List[str]:"
    args = [code]
    description_string = """Analyzes the given code and returns a list of suggestions for improvements."""

    result_string = call_ai_function(function_string, args, description_string)
    
    return result_string


# Improving code
@command("improve_code", "Get Improved Code", '"suggestions": "<list_of_suggestions>", "code": "<full_code_string>"')
def improve_code(suggestions: List[str], code: str) -> str:
    function_string = (
        "def generate_improved_code(suggestions: List[str], code: str) -> str:"
    )
    args = [json.dumps(suggestions), code]
    description_string = """Improves the provided code based on the suggestions provided, making no other changes."""

    result_string = call_ai_function(function_string, args, description_string)
    return result_string


# Writing tests

@command("write_tests", "Write Tests", '"code": "<full_code_string>", "focus": "<list_of_focus_areas>"')
def write_tests(code: str, focus: List[str]) -> str:
    function_string = (
        "def create_test_cases(code: str, focus: Optional[str] = None) -> str:"
    )
    args = [code, json.dumps(focus)]
    description_string = """Generates test cases for the existing code, focusing on specific areas if required."""

    result_string = call_ai_function(function_string, args, description_string)
    return result_string


