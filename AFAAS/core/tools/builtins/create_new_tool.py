# from __future__ import annotations

# # from typing import TYPE_CHECKING
# # if TYPE_CHECKING:
# from AFAAS.core.tools.base import BaseTool, ToolConfiguration
# from AFAAS.core.tools.schema import ToolResult


# class CreateNewTool(BaseTool):
#     default_configuration = ToolConfiguration(
#         location=PluginLocation(
#             storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
#             storage_route="AFAAS.core.tools.builtins.CreateNewTool",
#         ),
#     )
# #
#     def __init__(
#         self,

#         configuration: ToolConfiguration,
#     ):
#         self._configuration = configuration

#     @classmethod
#     def description(cls) -> str:
#         return "Create a new ability by writing python code."

#     @classmethod
#     def arguments(cls) -> dict:
#         return {
#             "ability_name": {
#                 "type": "string",
#                 "description": "A meaningful and concise name for the new ability.",
#             },
#             "description": {
#                 "type": "string",
#                 "description": "A detailed description of the ability and its uses, including any limitations.",
#             },
#             "arguments": {
#                 "type": "array",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "name": {
#                             "type": "string",
#                             "description": "The name of the argument.",
#                         },
#                         "type": {
#                             "type": "string",
#                             "description": "The type of the argument. Must be a standard json schema type.",
#                         },
#                         "description": {
#                             "type": "string",
#                             "description": "A detailed description of the argument and its uses.",
#                         },
#                     },
#                 },
#                 "description": "A list of arguments that the ability will accept.",
#             },
#             "required_arguments": {
#                 "type": "array",
#                 "items": {
#                     "type": "string",
#                     "description": "The names of the arguments that are required.",
#                 },
#                 "description": "A list of the names of the arguments that are required.",
#             },
#             "package_requirements": {
#                 "type": "array",
#                 "items": {
#                     "type": "string",
#                     "description": "The of the Python package that is required to execute the ability.",
#                 },
#                 "description": "A list of the names of the Python packages that are required to execute the ability.",
#             },
#             "code": {
#                 "type": "string",
#                 "description": "The Python code that will be executed when the ability is called.",
#             },
#         }

#     @classmethod
#     def required_arguments(cls) -> list[str]:
#         return [
#             "ability_name",
#             "description",
#             "arguments",
#             "required_arguments",
#             "package_requirements",
#             "code",
#         ]

#     async def __call__(
#         self,
#         ability_name: str,
#         description: str,
#         arguments: list[dict],
#         required_arguments: list[str],
#         package_requirements: list[str],
#         code: str,
#     ) -> ToolResult:
#         raise NotImplementedError
