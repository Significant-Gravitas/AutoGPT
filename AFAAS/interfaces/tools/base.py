from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pprint import pformat
from types import ModuleType
from typing import Any, ClassVar

import inflection
from pydantic import Field

from AFAAS.configs.schema import SystemConfiguration
from AFAAS.interfaces.adapters import CompletionModelFunction
from AFAAS.interfaces.adapters.language_model import AbstractPromptConfiguration
from AFAAS.interfaces.agent.features.agentmixin import AgentMixin
from AFAAS.interfaces.tools.schema import ToolResult
from AFAAS.lib.utils.json_schema import JSONSchema


class ToolConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    packages_required: list[str] = Field(default_factory=list)
    language_model_required: AbstractPromptConfiguration = None
    db_provider_required: bool = False
    workspace_required: bool = False


ToolConfiguration.update_forward_refs()


class AbstractTool(AgentMixin, abc.ABC):
    """A class representing an agent ability."""

    default_configuration: ClassVar[ToolConfiguration]

    FRAMEWORK_CATEGORY = "framework"

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
    def parameters(cls) -> dict[str, JSONSchema]: ...

    @abc.abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> ToolResult: ...

    def __str__(self) -> str:
        return pformat(self.spec)

    @property
    @classmethod
    def spec(cls) -> CompletionModelFunction:
        return CompletionModelFunction(
            name=cls.name(),
            description=cls.description,
            parameters=cls.parameters,
        )


class AbstractToolRegistry(AgentMixin, abc.ABC):
    def __init__(self, settings):
        pass  # NOTE : Avoid passing too many arguments to AgentMixin

    @abc.abstractmethod
    def register_tool(
        self, tool_name: str, tool_configuration: ToolConfiguration
    ) -> None: ...

    @abc.abstractmethod
    def list_tools_descriptions(self) -> list[str]: ...

    @abc.abstractmethod
    def dump_tools(self) -> list[CompletionModelFunction]: ...

    @abc.abstractmethod
    def get_tool(self, tool_name: str) -> AbstractTool: ...

    @abc.abstractmethod
    async def perform(self, tool_name: str, **kwargs: Any) -> ToolResult: ...


@dataclass
class ToolCategory:
    """
    Represents a category of tools.

    Attributes:
        name: Name of the category.
        title: Display title for the category.
        description: Description of the category.
        tools: List of Tool objects associated with the category.
        modules: List of ModuleType objects related to the category.
    """

    name: str
    title: str
    description: str
    tools: list[AbstractTool] = field(default_factory=list[AbstractTool])
    modules: list[ModuleType] = field(default_factory=list[ModuleType])

    class Config:
        arbitrary_types_allowed = True
