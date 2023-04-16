from autogpt.commands.command import command


@command("function_based", "Function-based test command")
def function_based(arg1: int, arg2: str) -> str:
    return f"{arg1} - {arg2}"
