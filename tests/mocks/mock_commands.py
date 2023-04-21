from autogpt.commands.command import command


@command("function_based", "Function-based test command")
def function_based(arg1: int, arg2: str) -> str:
    """
    A function-based test command that returns a string with the two arguments separated by a dash.
    Args:
        arg1 (int): The first argument.
        arg2 (str): The second argument.
    """
    return f"{arg1} - {arg2}"
