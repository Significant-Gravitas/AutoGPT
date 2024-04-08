import re
from typing import Callable, Optional, ParamSpec, TypeVar

from autogpt.agents.base import CommandArgs
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.command import CommandOutput, CommandParameter, ValidityResult

# Unique identifier for AutoGPT commands
AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"

P = ParamSpec("P")
CO = TypeVar("CO", bound=CommandOutput)


def command(
    names: list[str] = [],
    description: Optional[str] = None,
    parameters: dict[str, JSONSchema] = {},
    is_valid: Callable[[CommandArgs], ValidityResult] = lambda _: ValidityResult(True),
) -> Callable[[Callable[P, CO]], Callable[P, CO]]:
    """
    The command decorator is used to add command metadata to ordinary functions.

    Args:
        names (list[str]): The names of the command.
            If not provided, the function name will be used.
        description (str): A brief description of what the command does.
            If not provided, the docstring until double line break will be used
            (or entire docstring if no double line break is found)
        parameters (dict[str, JSONSchema]): The parameters of the function
            that the command executes.
        is_valid (Callable[[CommandArgs], ValidityResult]):
            A function that checks if the command with provided arguments is valid.
    """

    def decorator(func: Callable[P, CO]) -> Callable[P, CO]:
        doc = func.__doc__ or ""
        # If names is not provided, use the function name
        command_names = names or [func.__name__]
        # If description is not provided, use the first part of the docstring
        command_description = description
        if not command_description:
            if not func.__doc__:
                raise ValueError("Description is required if function has no docstring")
            # Return the part of the docstring before double line break or everything
            command_description = doc.split("\n\n")[0].strip()

        # Parameters
        typed_parameters = [
            CommandParameter(
                name=param_name,
                spec=spec,
            )
            for param_name, spec in parameters.items()
        ]

        setattr(func, "names", command_names)
        setattr(func, "description", command_description)
        setattr(func, "parameters", typed_parameters)
        setattr(func, "is_valid", is_valid)

        return func

    return decorator
