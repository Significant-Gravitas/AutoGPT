import re
from typing import Callable, Optional, ParamSpec, TypeVar

from forge.json.schema import JSONSchema

from .command import Command, CommandOutput, CommandParameter

# Unique identifier for AutoGPT commands
AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"

P = ParamSpec("P")
CO = TypeVar("CO", bound=CommandOutput)


def command(
    names: list[str] = [],
    description: Optional[str] = None,
    parameters: dict[str, JSONSchema] = {},
) -> Callable[[Callable[P, CommandOutput]], Command]:
    """
    The command decorator is used to make a Command from a function.

    Args:
        names (list[str]): The names of the command.
            If not provided, the function name will be used.
        description (str): A brief description of what the command does.
            If not provided, the docstring until double line break will be used
            (or entire docstring if no double line break is found)
        parameters (dict[str, JSONSchema]): The parameters of the function
            that the command executes.
    """

    def decorator(func: Callable[P, CO]) -> Command:
        doc = func.__doc__ or ""
        # If names is not provided, use the function name
        command_names = names or [func.__name__]
        # If description is not provided, use the first part of the docstring
        if not (command_description := description):
            if not func.__doc__:
                raise ValueError("Description is required if function has no docstring")
            # Return the part of the docstring before double line break or everything
            command_description = re.sub(r"\s+", " ", doc.split("\n\n")[0].strip())

        # Parameters
        typed_parameters = [
            CommandParameter(
                name=param_name,
                spec=spec,
            )
            for param_name, spec in parameters.items()
        ]

        # Wrap func with Command
        command = Command(
            names=command_names,
            description=command_description,
            method=func,
            parameters=typed_parameters,
        )

        return command

    return decorator
