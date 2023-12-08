from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from AFAAS.app.core.agents.base import BaseAgent
    from AFAAS.app.core.configuration.config import Config

from AFAAS.app.core.tools.tool_parameters import ToolParameter
from AFAAS.app.core.tools.tools import Tool, ToolOutput
from AFAAS.app.core.utils.json_schema import JSONSchema

# Unique identifier for AutoGPT commands
AUTO_GPT_TOOL_IDENTIFIER = "auto_gpt_command"


def tool(
    name: str,
    description: str,
    parameters: dict[str, JSONSchema] = {},
    enabled: Literal[True] | Callable[[Config], bool] = True,
    disabled_reason: Optional[str] = None,
    aliases: list[str] = [],
    available: Literal[True] | Callable[[BaseAgent], bool] = True,
    hide=False,
    success_check_callback: Optional[
        Callable[..., Any]
    ] = Tool.default_success_check_callback,  # Add this line
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
            exec_function=func,
            parameters=typed_parameters,
            enabled=enabled,
            disabled_reason=disabled_reason,
            aliases=aliases,
            available=available,
            hide=hide,
            success_check_callback=success_check_callback,
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
