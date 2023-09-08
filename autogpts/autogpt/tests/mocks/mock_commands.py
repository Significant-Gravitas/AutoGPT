from autogpt.command_decorator import command

COMMAND_CATEGORY = "mock"


@command(
    "function_based",
    "Function-based test command",
    {
        "arg1": {"type": "int", "description": "arg 1", "required": True},
        "arg2": {"type": "str", "description": "arg 2", "required": True},
    },
)
def function_based(arg1: int, arg2: str) -> str:
    """A function-based test command that returns a string with the two arguments separated by a dash."""
    return f"{arg1} - {arg2}"
