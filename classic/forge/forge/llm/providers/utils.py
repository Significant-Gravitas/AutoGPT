from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from forge.command.command import Command

from .schema import AssistantToolCall, CompletionModelFunction


class InvalidFunctionCallError(Exception):
    def __init__(self, name: str, arguments: dict[str, Any], message: str):
        self.message = message
        self.name = name
        self.arguments = arguments
        super().__init__(message)

    def __str__(self) -> str:
        return f"Invalid function call for {self.name}: {self.message}"


def validate_tool_calls(
    tool_calls: list[AssistantToolCall], functions: list[CompletionModelFunction]
) -> list[InvalidFunctionCallError]:
    """
    Validates a list of tool calls against a list of functions.

    1. Tries to find a function matching each tool call
    2. If a matching function is found, validates the tool call's arguments,
    reporting any resulting errors
    2. If no matching function is found, an error "Unknown function X" is reported
    3. A list of all errors encountered during validation is returned

    Params:
        tool_calls: A list of tool calls to validate.
        functions: A list of functions to validate against.

    Returns:
        list[InvalidFunctionCallError]: All errors encountered during validation.
    """
    errors: list[InvalidFunctionCallError] = []
    for tool_call in tool_calls:
        function_call = tool_call.function

        if function := next(
            (f for f in functions if f.name == function_call.name),
            None,
        ):
            is_valid, validation_errors = function.validate_call(function_call)
            if not is_valid:
                fmt_errors = [
                    f"{'.'.join(str(p) for p in f.path)}: {f.message}"
                    if f.path
                    else f.message
                    for f in validation_errors
                ]
                errors.append(
                    InvalidFunctionCallError(
                        name=function_call.name,
                        arguments=function_call.arguments,
                        message=(
                            "The set of arguments supplied is invalid:\n"
                            + "\n".join(fmt_errors)
                        ),
                    )
                )
        else:
            errors.append(
                InvalidFunctionCallError(
                    name=function_call.name,
                    arguments=function_call.arguments,
                    message=f"Unknown function {function_call.name}",
                )
            )

    return errors


def function_specs_from_commands(
    commands: Iterable["Command"],
) -> list[CompletionModelFunction]:
    """Get LLM-consumable function specs for the agent's available commands."""
    return [
        CompletionModelFunction(
            name=command.names[0],
            description=command.description,
            parameters={param.name: param.spec for param in command.parameters},
        )
        for command in commands
    ]
