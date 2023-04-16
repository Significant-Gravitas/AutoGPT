"""A module that contains a function to generate test cases for the submitted code."""
import json
from typing import List
from autogpt.llm_utils import call_ai_function


def write_tests(code: str, focus: List[str]) -> str:
    """
    A function that takes in code and focus topics and returns a response from create
      chat completion api call.

    Parameters:
        focus (List): A list of suggestions around what needs to be improved.
        code (str): Code for test cases to be generated against.
    Returns:
        A result string from create chat completion. Test cases for the submitted code
          in response.
    """

    function_string = (
        "def create_test_cases(code: str, focus: Optional[str] = None) -> str:"
    )
    args = [code, json.dumps(focus)]
    description_string = (
        "Generates test cases for the existing code, focusing on"
        " specific areas if required."
    )

    return call_ai_function(function_string, args, description_string)
