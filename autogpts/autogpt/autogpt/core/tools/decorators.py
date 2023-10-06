# from __future__ import annotations

# import functools
# from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, TypedDict

# # Unique identifier for auto-gpt commands
# AUTO_GPT_TOOL_IDENTIFIER = "auto_gpt_command"

# if TYPE_CHECKING:
#     from autogpt.core.agents.simple.lib.models.command import ToolOutput, ToolParameter
#     from autogpt.core.tools.base import Tool, ToolResult, ToolConfiguration
#     from autogpt.core.agents.base import BaseAgent


# class ToolParameterSpec(TypedDict):
#     type: str
#     description: str
#     required: bool


# def ability(
#     name: str,
#     description: str,
#     parameters: dict[str, ToolParameterSpec],
#     enabled: Literal[True] | Callable[[ToolConfiguration], bool] = True,
#     disabled_reason: Optional[str] = None,
#     aliases: list[str] = [],
#     available: Literal[True] | Callable[[BaseAgent], bool] = True,
# ) -> Callable[
#     ..., ToolOutput
# ]:  # Assuming there's ToolOutput analogous to ToolOutput
#     """The ability decorator is used to create Tool objects from ordinary functions."""

#     def decorator(func: Callable[..., ToolOutput]):
#         typed_parameters = [
#             ToolParameter(
#                 name=param_name,
#                 description=parameter.get("description"),
#                 type=parameter.get("type", "string"),
#                 required=parameter.get("required", False),
#             )
#             for param_name, parameter in parameters.items()
#         ]

#         ablt = Tool(
#             name=name,
#             description=description,
#             method=func,
#             parameters=typed_parameters,
#             enabled=enabled,
#             disabled_reason=disabled_reason,
#             aliases=aliases,
#             available=available,
#         )

#         @functools.wraps(func)
#         def wrapper(*args, **kwargs) -> Any:
#             return func(*args, **kwargs)

#         setattr(wrapper, "ability", ablt)
#         setattr(
#             wrapper, AUTO_GPT_TOOL_IDENTIFIER, True
#         )  # Assuming you have an analogous identifier

#         return wrapper

#     return decorator
