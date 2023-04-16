from __future__ import annotations

import json

from autogpt.llm_utils import call_ai_function


def improve_code(suggestions: list[str], code: str) -> str:
    """
    A function that takes in code and suggestions and returns a response from create
      chat completion api call.

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
    description_string = (
        "Improves the provided code based on the suggestions"
        " provided, making no other changes."
    )

    return call_ai_function(function_string, args, description_string)
