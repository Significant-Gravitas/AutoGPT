from typing import List, Optional
import json
from my_project.my_project.refactoring_tool.ai_functions import call_ai
from typing import List, Optional
import json
from my_project.my_project.refactoring_tool.ai_functions import call_ai_function

def write_tests(code: str, focus: Optional[List[str]] = None) -> str:
    function_string = "def write_tests(code: str, focus: Optional[List[str]] = None) -> str:"
    args = [code, json.dumps(focus)]
    description_string = "Generates unit test cases for the provided code, focusing on specific areas if required."

    result_string = call_ai_function(function_string, args, description_string)
    return result_string

def write_integration_tests(code: str, focus: Optional[List[str]] = None) -> str:
    function_string = "def write_integration_tests(code: str, focus: Optional[List[str]] = None) -> str:"
    args = [code, json.dumps(focus)]
    description_string = "Generates integration test cases for the provided code, focusing on specific areas if required."

    result_string = call_ai_function(function_string, args, description_string)
    return result_string

def write_system_tests(code: str, focus: Optional[List[str]] = None) -> str:
    function_string = "def write_system_tests(code: str, focus: Optional[List[str]] = None) -> str:"
    args = [code, json.dumps(focus)]
    description_string = "Generates system test cases for the provided code, focusing on specific areas if required."

    result_string = call_ai_function(function_string, args, description_string)
    return result_string
