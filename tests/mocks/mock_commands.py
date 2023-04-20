from autogpt.commands.command import command


@command("function_based", "Function-based test command")
def function_based(arg1: int, arg2: str) -> str:
    """Test function-based command."""
    # Return a string with the two arguments separated by a dash
    return f"{arg1} - {arg2}"
