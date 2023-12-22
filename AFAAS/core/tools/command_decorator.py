from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, ParamSpec, TypeVar

if TYPE_CHECKING:
    from AFAAS.interfaces.agent import BaseAgent
    from AFAAS.core.configuration.config import Config

from AFAAS.core.tools.tool_parameters import ToolParameter
from AFAAS.core.tools.tools import Tool, ToolOutput
from AFAAS.lib.utils.json_schema import JSONSchema

# Unique identifier for AutoGPT commands
AUTO_GPT_TOOL_IDENTIFIER = "auto_gpt_command"

P = ParamSpec("P")
CO = TypeVar("CO", bound=ToolOutput)


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
    tech_description: Optional[str] = None,
) -> Callable[[Callable[P, CO]], Callable[P, CO]]:
    """The command decorator is used to create Tool objects from ordinary functions."""

    def decorator(func: Callable[P, CO]) -> Callable[P, CO]:
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
            tech_description=tech_description,
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
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return await func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return func(*args, **kwargs)

        setattr(wrapper, "tool", cmd)
        setattr(wrapper, AUTO_GPT_TOOL_IDENTIFIER, True)

        return wrapper

    return decorator
