import abc
from pprint import pformat
from typing import Any, ClassVar

import inflection

from autogpt.core.configuration import SystemConfiguration
from autogpt.core.configuration.schema import UserConfigurable
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.base import PluginLocation
from autogpt.core.resource.model_providers import CompletionModelFunction
from autogpt.core.utils.json_schema import JSONSchema

from .schema import AbilityResult


class AbilityConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    location: PluginLocation
    packages_required: list[str] = UserConfigurable(default_factory=list)
    language_model_required: LanguageModelConfiguration = None
    memory_provider_required: bool = UserConfigurable(default=False)
    workspace_required: bool = UserConfigurable(default=False)


class Ability(abc.ABC):
    """A class representing an agent ability."""

    default_configuration: ClassVar[AbilityConfiguration]

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
    async def __call__(self, *args: Any, **kwargs: Any) -> AbilityResult:
        ...

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


class AbilityRegistry(abc.ABC):
    @abc.abstractmethod
    def register_ability(
        self, ability_name: str, ability_configuration: AbilityConfiguration
    ) -> None:
        ...

    @abc.abstractmethod
    def list_abilities(self) -> list[str]:
        ...

    @abc.abstractmethod
    def dump_abilities(self) -> list[CompletionModelFunction]:
        ...

    @abc.abstractmethod
    def get_ability(self, ability_name: str) -> Ability:
        ...

    @abc.abstractmethod
    async def perform(self, ability_name: str, **kwargs: Any) -> AbilityResult:
        ...
