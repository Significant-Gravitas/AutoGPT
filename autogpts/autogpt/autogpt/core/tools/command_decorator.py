from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from autogpt.core.agents.base import BaseAgent
    from autogpt.config import Config

from autogpt.core.utils.json_schema import JSONSchema
from autogpt.core.tools.tools import Tool, ToolOutput
from autogpt.core.tools.tool_parameters import ToolParameter

# Unique identifier for AutoGPT commands
AUTO_GPT_TOOL_IDENTIFIER = "auto_gpt_command"


def tool(
    name: str,
    description: str,
    parameters: dict[str, JSONSchema],
    enabled: Literal[True] | Callable[[Config], bool] = True,
    disabled_reason: Optional[str] = None,
    aliases: list[str] = [],
    available: Literal[True] | Callable[[BaseAgent], bool] = True,
) -> Callable[..., ToolOutput]:
    """The command decorator is used to create Tool objects from ordinary functions."""

    def decorator(func: Callable[..., ToolOutput]):
        typed_parameters = [
            ToolParameter(
                name=param_name,
                spec=spec,
            )
            for param_name, spec in parameters.items()
        ]
        cmd = Tool(
            name=name,
            description=description,
            method=func,
            parameters=typed_parameters,
            enabled=enabled,
            disabled_reason=disabled_reason,
            aliases=aliases,
            available=available,
        )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                return await func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                return func(*args, **kwargs)

        setattr(wrapper, "tool", cmd)
        setattr(wrapper, AUTO_GPT_TOOL_IDENTIFIER, True)

        return wrapper

    return decorator
