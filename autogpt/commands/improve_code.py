from __future__ import annotations

import json

from autogpt.commands.command import command
from autogpt.llm import call_ai_function


@command(
    "improve_code",
    "Get Improved Code",
    '"suggestions": "<list_of_suggestions>", "code": "<full_code_string>"',
)
def improve_code(suggestions: list[str], code: str) -> str:
    """
    A function that takes in code and suggestions and returns a response from create
      chat completion api call.

    Parameters:
        suggestions (list): A list of suggestions around what needs to be improved.
        code (str): Code to be improved.
    Returns:
        A result string from create chat completion. Improved code in response.
    """

    function_string = (
        "def generate_improved_code(suggestions: list[str], code: str) -> str:"
    )
    args = [json.dumps(suggestions), code]
    description_string = (
        "Improves the provided code based on the suggestions"
        " provided, making no other changes."
    )

    return call_ai_function(function_string, args, description_string)
