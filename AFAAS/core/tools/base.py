from __future__ import annotations

import abc
from pprint import pformat
from typing import Any, ClassVar

import inflection
from pydantic import Field

from AFAAS.interfaces.agent.assistants import LanguageModelConfiguration
from AFAAS.interfaces.agent.features.agentmixin import AgentMixin
from AFAAS.interfaces.configuration import SystemConfiguration
from ..plugin.base import PluginLocation
from ..resource.model_providers import CompletionModelFunction
from ..tools.schema import ToolResult
from ..utils.json_schema import JSONSchema
from .schema import ToolResult


class ToolConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    location: PluginLocation
    packages_required: list[str] = Field(default_factory=list)
    language_model_required: LanguageModelConfiguration = None
    memory_provider_required: bool = False
    workspace_required: bool = False


ToolConfiguration.update_forward_refs()


class BaseTool(AgentMixin, abc.ABC):
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
    def __init__(self, settings):
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
    def get_tool(self, tool_name: str) -> BaseTool:
        ...

    @abc.abstractmethod
    async def perform(self, tool_name: str, **kwargs: Any) -> ToolResult:
        ...
