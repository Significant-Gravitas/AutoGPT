from typing import Callable

from autogpt.agents.agent import Agent


def sanitize_path_arg(arg_name: str):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            agent: Agent = kwargs["agent"]

            if arg_name in kwargs:
                if kwargs[arg_name] in {"", "/"}:
                    kwargs[arg_name] = str(agent.workspace.root)
                else:
                    kwargs[arg_name] = str(
                        agent.workspace.get_path(kwargs[arg_name])
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator
