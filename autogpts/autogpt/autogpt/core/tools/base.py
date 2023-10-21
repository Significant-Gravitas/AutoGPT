from __future__ import annotations

import abc
from pprint import pformat
from typing import Any, ClassVar, TYPE_CHECKING

import inflection
from pydantic import Field
from .schema import ToolResult


from ..tools.schema import ToolResult
from ..configuration import SystemConfiguration
from ..agents.base.features.agentmixin import AgentMixin
from ..agents.base.prompt_manager import LanguageModelConfiguration
from ..resource.model_providers import CompletionModelFunction
from ..utils.json_schema import JSONSchema
from ..plugin.base import PluginLocation


class ToolConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    location: PluginLocation
    packages_required: list[str] = Field(default_factory=list)
    language_model_required: LanguageModelConfiguration = None
    memory_provider_required: bool = False
    workspace_required: bool = False


ToolConfiguration.update_forward_refs()


class Tool(AgentMixin, abc.ABC):
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

    @property
    @classmethod
    def spec(cls) -> CompletionModelFunction:
        return CompletionModelFunction(
            name=cls.name(),
            description=cls.description,
            parameters=cls.parameters,
        )


class BaseToolsRegistry(AgentMixin, abc.ABC):
    def __init__(self, settings, logger):
        pass  # NOTE : Avoid passing too many arguments to AgentMixin

    @abc.abstractmethod
    def register_tool(
        self, tool_name: str, tool_configuration: ToolConfiguration
    ) -> None:
        ...

    @abc.abstractmethod
    def list_tools_descriptions(self) -> list[str]:
        ...

    @abc.abstractmethod
    def dump_tools(self) -> list[CompletionModelFunction]:
        ...

    @abc.abstractmethod
    def get_tool(self, tool_name: str) -> Tool:
        ...

    @abc.abstractmethod
    async def perform(self, tool_name: str, **kwargs: Any) -> ToolResult:
        ...
