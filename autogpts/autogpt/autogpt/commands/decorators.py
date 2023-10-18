import functools
import logging
import re
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

from autogpt.agents.agent import Agent

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


def sanitize_path_arg(
    arg_name: str, make_relative: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Sanitizes the specified path (str | Path) argument, resolving it to a Path"""

    def decorator(func: Callable) -> Callable:
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

            # Get Agent from the called function's arguments
            agent = kwargs.get(
                "agent", len(args) > agent_arg_index and args[agent_arg_index]
            )
            if not isinstance(agent, Agent):
                raise RuntimeError("Could not get Agent from decorated command's args")

            # Sanitize the specified path argument, if one is given
            given_path: str | Path | None = kwargs.get(
                arg_name, len(args) > arg_index and args[arg_index] or None
            )
            if given_path:
                if type(given_path) is str:
                    # Fix workspace path from output in docker environment
                    given_path = re.sub(r"^\/workspace", ".", given_path)

                if given_path in {"", "/", "."}:
                    sanitized_path = agent.workspace.root
                else:
                    sanitized_path = agent.workspace.get_path(given_path)

                # Make path relative if possible
                if make_relative and sanitized_path.is_relative_to(
                    agent.workspace.root
                ):
                    sanitized_path = sanitized_path.relative_to(agent.workspace.root)

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
