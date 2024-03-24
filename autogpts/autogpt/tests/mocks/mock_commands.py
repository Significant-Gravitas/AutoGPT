from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema

COMMAND_CATEGORY = "mock"


@command(
    "function_based_cmd",
    "Function-based test command",
    {
        "arg1": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="arg 1",
            required=True,
        ),
        "arg2": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="arg 2",
            required=True,
        ),
    },
)
def function_based_cmd(arg1: int, arg2: str) -> str:
    """A function-based test command.

    Returns:
        str: the two arguments separated by a dash.
    """
    return f"{arg1} - {arg2}"
