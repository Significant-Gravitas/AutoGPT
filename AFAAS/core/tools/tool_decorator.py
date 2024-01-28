from __future__ import annotations

import asyncio
import functools
import inspect
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
)

from langchain.tools import BaseTool

from AFAAS.interfaces.tools.tool_output import ToolOutput
from AFAAS.interfaces.tools.tool_parameters import ToolParameter

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent
    from AFAAS.configs.config import Config

from dotenv import load_dotenv

from AFAAS.core.tools.tool import Tool
from AFAAS.lib.utils.json_schema import JSONSchema

# Unique identifier for AutoGPT commands
TOOL_WRAPPER_MARKER = "afaas_tool"

P = ParamSpec("P")
CO = TypeVar("CO", bound=ToolOutput)


# Load the .env file
load_dotenv()
# TODO: Allow tool to define if they are safe or not (safe mode is not implemented yet)
SAFE_MODE = os.environ.get("SAFE_MODE", "true").lower() == "true"


def tool(
    name: str,
    description: str,
    parameters: dict[str, JSONSchema] = {},
    categories: list[str] = ["uncategorized"],
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
            categories=categories,
            enabled=enabled,
            disabled_reason=disabled_reason,
            aliases=aliases,
            available=available,
            hide=hide,
            success_check_callback=success_check_callback,
        )

        from AFAAS.core.tools.builtins.not_implemented_tool import not_implemented_tool

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                try:
                    return await func(*args, **kwargs)
                except NotImplementedError as e:
                    return await not_implemented_tool(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except NotImplementedError as e:
                    return asyncio.run(not_implemented_tool(*args, **kwargs))

        setattr(wrapper, "tool", cmd)
        setattr(wrapper, TOOL_WRAPPER_MARKER, True)

        return wrapper

    return decorator


## TO REMOVE
def tool_from_langchain(
    categories: list[str] = ["uncategorized"],
    arg_converter: Optional[Callable] = None,
    enabled: bool = True,
    disabled_reason: Optional[str] = None,
    aliases: list[str] = [],
    available: bool = True,
    hide: bool = False,
    success_check_callback: Callable = Tool.default_success_check_callback,
):
    def decorator(base_tool: Type[BaseTool]):
        base_tool_instance = base_tool()

        def wrapper(*args, **kwargs):
            # Extract 'agent' from kwargs if it exists, as it's not used in this context
            agent = kwargs.pop("agent", None)

            # Perform argument conversion if an arg_converter is provided
            if arg_converter:
                tool_input = arg_converter(kwargs, agent)
            else:
                tool_input = kwargs

            # Run the BaseTool's run method and return its result
            return base_tool_instance.arun(tool_input=tool_input)

        # Apply the @tool decorator to the wrapper function
        # We use the properties of base_tool_instance (name, description, etc.) to define the tool
        return tool(
            name=base_tool_instance.name,
            description=base_tool_instance.description,
            tech_description=base_tool_instance.description,  # Assuming this is intentional
            categories=categories,
            # exec_function=wrapper,
            # parameters=[
            #     ToolParameter(name=name, spec=schema)
            #     for name, schema in base_tool_instance.args.items()
            # ],
            parameters=base_tool_instance.args,
            enabled=enabled,
            disabled_reason=disabled_reason,
            aliases=aliases,
            available=available,
            hide=hide,
            success_check_callback=success_check_callback,
        )(wrapper)

    return decorator


# TODO/FIXME:
# Huging face Tools :
# https://github.com/huggingface/transformers/blob/main/src/transformers/tools/base.py#L471
