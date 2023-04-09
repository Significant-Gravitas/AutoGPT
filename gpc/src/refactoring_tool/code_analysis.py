from typing import List
import json
from src.refactoring_tool.ai_functions import call_ai_function

class CodeAnalysis:
    
    @staticmethod
    def analyze_code_readability(code: str) -> str:
        function_string = "def analyze_code_readability(code: str) -> str:"
        args = [code]
        description_string = "Analyze the readability of the given code and return a plain text result."

        result_string = call_ai_function(
            function_string, tuple(args), description_string)
        return result_string.strip()  # Strip any leading/trailing whitespace

    @staticmethod
    def analyze_code_performance(code: str) -> List[str]:
        function_string = "def analyze_code_performance(code: str) -> List[str]:"
        args = [code]
        description_string = "Analyzes the given code and returns a list of suggestions for optimizing performance."
        print("Result string:", description_string)

        result_string = call_ai_function(
            function_string, tuple(args), description_string)
        return json.loads(result_string)

    @staticmethod
    def analyze_code_security(code: str) -> List[str]:
        function_string = "def analyze_code_security(code: str) -> List[str]:"
        args = [code]
        description_string = "Analyzes the given code and returns a list of suggestions for improving security."

        result_string = call_ai_function(
            function_string, tuple(args), description_string)
        return json.loads(result_string)

    @staticmethod
    def analyze_code_modularity(code: str) -> List[str]:
        function_string = "def analyze_code_modularity(code: str) -> List[str]:"
        args = [code]
        description_string = "Analyzes the given code and returns a list of suggestions for improving modularity."

        result_string = call_ai_function(
            function_string, tuple(args), description_string)
        return json.loads(result_string)
