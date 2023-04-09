from typing import List
import json
from ai_functions import call_ai_function

def analyze_code_readability(code: str) -> List[str]:
    function_string = "def analyze_code_readability(code: str) -> List[str]:"
    args = [code]
    description_string = "Analyzes the given code and returns a list of suggestions for improving readability."

    result_string = call_ai_function(function_string, args, description_string)
    return json.loads(result_string)

def analyze_code_performance(code: str) -> List[str]:
    function_string = "def analyze_code_performance(code: str) -> List[str]:"
    args = [code]
    description_string = "Analyzes the given code and returns a list of suggestions for optimizing performance."

    result_string = call_ai_function(function_string, args, description_string)
    return json.loads(result_string)

def analyze_code_security(code: str) -> List[str]:
    function_string = "def analyze_code_security(code: str) -> List[str]:"
    args = [code]
    description_string = "Analyzes the given code and returns a list of suggestions for improving security."

    result_string = call_ai_function(function_string, args, description_string)
    return json.loads(result_string)

def analyze_code_modularity(code: str) -> List[str]:
    function_string = "def analyze_code_modularity(code: str) -> List[str]:"
    args = [code]
    description_string = "Analyzes the given code and returns a list of suggestions for improving modularity."

    result_string = call_ai_function(function_string, args, description_string)
    return json.loads(result_string)
