import abc
from pprint import pformat
from typing import Any, ClassVar

import inflection
from pydantic import Field

from autogpt.core.tools.schema import ToolResult
from autogpt.core.configuration import SystemConfiguration
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.resource.model_providers import CompletionModelFunction
from autogpt.core.utils.json_schema import JSONSchema
from .schema import ToolResult


class ToolConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    from autogpt.core.plugin.base import PluginLocation

    location: PluginLocation
    packages_required: list[str] = Field(default_factory=list)
    language_model_required: LanguageModelConfiguration = None
    memory_provider_required: bool = False
    workspace_required: bool = False


class Tool(abc.ABC):
    """A class representing an agent ability."""

    default_configuration: ClassVar[ToolConfiguration]

    @classmethod
    def name(cls) -> str:
        """The name of the ability."""
        return inflection.underscore(cls.__name__)

    @property
    @classmethod
    @abc.abstractmethod
    def description(cls) -> str:
        """A detailed description of what the ability does."""
        ...

    @property
    @classmethod
    @abc.abstractmethod
    def parameters(cls) -> dict[str, JSONSchema]:
        ...

    @abc.abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> ToolResult:
        ...

    def __str__(self) -> str:
        return pformat(self.spec)
    

    # def dump(self) -> dict:
    #     return {
    #         "name": self.name(),
    #         "description": self.description(),
    #         "parameters": {
    #             "type": "object",
    #             "properties": self.arguments(),
    #             "required": self.required_arguments(),
    #         },
    #     }


    #@property
    @classmethod
    def spec(cls) -> CompletionModelFunction:
        return CompletionModelFunction(
            name=cls.name(),
            description=cls.description,
            parameters=cls.parameters,
        )

class ToolsRegistry(abc.ABC):
    @abc.abstractmethod
    def register_tool(
        self, tool_name: str, tool_configuration: ToolConfiguration
    ) -> None:
        ...

    @abc.abstractmethod
    def list_tools_descriptions(self) -> list[str]:
        ...

    @abc.abstractmethod
    def dump_tools(self)  -> list[CompletionModelFunction]:
        ...

    @abc.abstractmethod
    def get_tool(self, tool_name: str) -> Tool:
        ...

    @abc.abstractmethod
    async def perform(self, tool_name: str, **kwargs: Any) -> ToolResult:
        ...
