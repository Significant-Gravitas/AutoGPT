import functools
from typing import Any, Callable, Dict, Optional

from autogpt.config import Config
from autogpt.models.command import Command
from autogpt.models.command_argument import CommandArgument

# Unique identifier for auto-gpt commands
AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"


def command(
    name: str,
    description: str,
    arguments: Dict[str, Dict[str, Any]],
    enabled: bool | Callable[[Config], bool] = True,
    disabled_reason: Optional[str] = None,
) -> Callable[..., Any]:
    """The command decorator is used to create Command objects from ordinary functions."""

    def decorator(func: Callable[..., Any]) -> Command:
        typed_arguments = [
            CommandArgument(
                name=arg_name,
                description=argument.get("description"),
                type=argument.get("type", "string"),
                required=argument.get("required", False),
            )
            for arg_name, argument in arguments.items()
        ]
        cmd = Command(
            name=name,
            description=description,
            method=func,
            arguments=typed_arguments,
            enabled=enabled,
            disabled_reason=disabled_reason,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)

        wrapper.command = cmd

        setattr(wrapper, AUTO_GPT_COMMAND_IDENTIFIER, True)

        return wrapper

    return decorator
