from typing import List
import json
from config import Config
from call_ai_function import call_ai_function
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
    description_string = """分析给定的代码，并返回改进建议的列表。"""

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
    description_string = """基于提供的建议改进提供的代码，不做其他更改。"""

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
    description_string = """为现有代码生成测试用例，如果需要，重点关注特定区域。"""

    result_string = call_ai_function(function_string, args, description_string)
    return result_string
