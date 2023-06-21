import functools
from typing import Any, Callable, Dict, Optional

from autogpt.config import Config
from autogpt.models.command import Command

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
        cmd = Command(
            name=name,
            description=description,
            method=func,
            signature=arguments,
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
