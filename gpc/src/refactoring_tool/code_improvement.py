from typing import List
import json
from src.refactoring_tool.ai_functions import call_ai_function

def refactor_variables(suggestions: List[str], code: str) -> str:
    function_string = "def refactor_variables(suggestions: List[str], code: str) -> str:"
    args = [json.dumps(suggestions), code]
    description_string = "Refactors variable names in the provided code based on the suggestions provided."

    result_string = call_ai_function(function_string, args, description_string)
    return result_string

def optimize_loops(suggestions: List[str], code: str) -> str:
    function_string = "def optimize_loops(suggestions: List[str], code: str) -> str:"
    args = [json.dumps(suggestions), code]
    description_string = "Optimizes loops in the provided code based on the suggestions provided."

    result_string = call_ai_function(function_string, args, description_string)
    return result_string

def simplify_conditionals(suggestions: List[str], code: str) -> str:
    function_string = "def simplify_conditionals(suggestions: List[str], code: str) -> str:"
    args = [json.dumps(suggestions), code]
    description_string = "Simplifies conditional statements in the provided code based on the suggestions provided."

    result_string = call_ai_function(function_string, args, description_string)
    return result_string

def refactor_functions(suggestions: List[str], code: str) -> str:
    function_string = "def refactor_functions(suggestions: List[str], code: str) -> str:"
    args = [json.dumps(suggestions), code]
    description_string = "Refactors functions in the provided code based on the suggestions provided."

    result_string = call_ai_function(function_string, args, description_string)
    return result_string
