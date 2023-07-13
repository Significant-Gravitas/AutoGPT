import functools
from pathlib import Path
from typing import Callable

from autogpt.agents.agent import Agent
from autogpt.logs import logger


def sanitize_path_arg(arg_name: str):
    def decorator(func: Callable):
        # Get position of path parameter, in case it is passed as a positional argument
        try:
            arg_index = list(func.__annotations__.keys()).index(arg_name)
        except ValueError:
            raise TypeError(
                f"Sanitized parameter '{arg_name}' absent or not annotated on function '{func.__name__}'"
            )

        # Get position of agent parameter, in case it is passed as a positional argument
        try:
            agent_arg_index = list(func.__annotations__.keys()).index("agent")
        except ValueError:
            raise TypeError(
                f"Parameter 'agent' absent or not annotated on function '{func.__name__}'"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Sanitizing arg '{arg_name}' on function '{func.__name__}'")
            logger.debug(f"Function annotations: {func.__annotations__}")

            # Get Agent from the called function's arguments
            agent = kwargs.get(
                "agent", len(args) > agent_arg_index and args[agent_arg_index]
            )
            logger.debug(f"Args: {args}")
            logger.debug(f"KWArgs: {kwargs}")
            logger.debug(f"Agent argument lifted from function call: {agent}")
            if not isinstance(agent, Agent):
                raise RuntimeError("Could not get Agent from decorated command's args")

            # Sanitize the specified path argument, if one is given
            given_path: str | Path | None = kwargs.get(
                arg_name, len(args) > arg_index and args[arg_index] or None
            )
            if given_path:
                if given_path in {"", "/"}:
                    sanitized_path = str(agent.workspace.root)
                else:
                    sanitized_path = str(agent.workspace.get_path(given_path))

                if arg_name in kwargs:
                    kwargs[arg_name] = sanitized_path
                else:
                    # args is an immutable tuple; must be converted to a list to update
                    arg_list = list(args)
                    arg_list[arg_index] = sanitized_path
                    args = tuple(arg_list)

            return func(*args, **kwargs)

        return wrapper

    return decorator
