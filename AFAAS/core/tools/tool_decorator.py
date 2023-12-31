from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, ParamSpec, TypeVar

from langchain_core.tools import BaseTool

from AFAAS.interfaces.tools.tool_output import ToolOutput
from AFAAS.interfaces.tools.tool_parameters import ToolParameter

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent
    from AFAAS.configs.config import Config

from AFAAS.core.tools.tools import Tool
from AFAAS.lib.utils.json_schema import JSONSchema

# Unique identifier for AutoGPT commands
AFAAS_TOOL_IDENTIFIER = "afaas_tool"

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
        setattr(wrapper, AFAAS_TOOL_IDENTIFIER, True)

        return wrapper

    return decorator


def tool_from_langchain(
    arg_converter: Optional[Callable] = None,
    enabled: bool = True,
    disabled_reason: Optional[str] = None,
    aliases: list[str] = [],
    available: bool = True,
    hide: bool = False,
    success_check_callback: Callable = Tool.default_success_check_callback,
):
    def decorator(base_tool: BaseTool):
        def wrapper(*args, **kwargs):
            # Extract 'agent' from kwargs if it exists, as it's not used in this context
            agent = kwargs.pop("agent", None)

            # Perform argument conversion if an arg_converter is provided
            if arg_converter:
                tool_input = arg_converter(kwargs, agent)
            else:
                tool_input = kwargs

            # Run the BaseTool's run method and return its result
            return base_tool.arun(tool_input=tool_input)

        # Apply the @tool decorator to the wrapper function
        # We use the properties of base_tool (name, description, etc.) to define the tool
        return tool(
            name=base_tool.name,
            description=base_tool.description,
            tech_description=base_tool.description,  # Assuming this is intentional
            exec_function=wrapper,
            parameters=[
                ToolParameter(name=name, spec=schema)
                for name, schema in base_tool.args.items()
            ],
            enabled=enabled,
            disabled_reason=disabled_reason,
            aliases=aliases,
            available=available,
            hide=hide,
            success_check_callback=success_check_callback,
        )(wrapper)

    return decorator


# from langchain_community.tools.github.tool import GitHubAction

# @tool_from_langchain()
# class AdaptedGitHubTool(GitHubAction):
#     pass
