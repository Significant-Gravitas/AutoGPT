import functools
from typing import Any, Callable, Optional

from autogpt.config import Config
from autogpt.logs import logger
from autogpt.models.command import Command

# Unique identifier for auto-gpt commands
AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"


def command(
    name: str,
    description: str,
    signature: str,
    enabled: bool | Callable[[Config], bool] = True,
    disabled_reason: Optional[str] = None,
) -> Callable[..., Any]:
    """The command decorator is used to create Command objects from ordinary functions."""

    # TODO: Remove this in favor of better command management
    CFG = Config()

    if callable(enabled):
        enabled = enabled(CFG)
    if not enabled:
        if disabled_reason is not None:
            logger.debug(f"Command '{name}' is disabled: {disabled_reason}")
        return lambda func: func

    def decorator(func: Callable[..., Any]) -> Command:
        cmd = Command(
            name=name,
            description=description,
            method=func,
            signature=signature,
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
