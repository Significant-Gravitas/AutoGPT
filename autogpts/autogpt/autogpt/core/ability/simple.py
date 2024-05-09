import logging

from forge.llm.providers import (
    ChatModelProvider,
    CompletionModelFunction,
    ModelProviderName,
)
from forge.models.config import Configurable, SystemConfiguration, SystemSettings

from autogpt.core.ability.base import Ability, AbilityConfiguration, AbilityRegistry
from autogpt.core.ability.builtins import BUILTIN_ABILITIES
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.memory.base import Memory
from autogpt.core.plugin.simple import SimplePluginService
from autogpt.core.workspace.base import Workspace


class AbilityRegistryConfiguration(SystemConfiguration):
    """Configuration for the AbilityRegistry subsystem."""

    abilities: dict[str, AbilityConfiguration]


class AbilityRegistrySettings(SystemSettings):
    configuration: AbilityRegistryConfiguration


class SimpleAbilityRegistry(AbilityRegistry, Configurable):
    default_settings = AbilityRegistrySettings(
        name="simple_ability_registry",
        description="A simple ability registry.",
        configuration=AbilityRegistryConfiguration(
            abilities={
                ability_name: ability.default_configuration
                for ability_name, ability in BUILTIN_ABILITIES.items()
            },
        ),
    )

    def __init__(
        self,
        settings: AbilityRegistrySettings,
        logger: logging.Logger,
        memory: Memory,
        workspace: Workspace,
        model_providers: dict[ModelProviderName, ChatModelProvider],
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._memory = memory
        self._workspace = workspace
        self._model_providers = model_providers
        self._abilities: list[Ability] = []
        for (
            ability_name,
            ability_configuration,
        ) in self._configuration.abilities.items():
            self.register_ability(ability_name, ability_configuration)

    def register_ability(
        self, ability_name: str, ability_configuration: AbilityConfiguration
    ) -> None:
        ability_class = SimplePluginService.get_plugin(ability_configuration.location)
        ability_args = {
            "logger": self._logger.getChild(ability_name),
            "configuration": ability_configuration,
        }
        if ability_configuration.packages_required:
            # TODO: Check packages are installed and maybe install them.
            pass
        if ability_configuration.memory_provider_required:
            ability_args["memory"] = self._memory
        if ability_configuration.workspace_required:
            ability_args["workspace"] = self._workspace
        if ability_configuration.language_model_required:
            ability_args["language_model_provider"] = self._model_providers[
                ability_configuration.language_model_required.provider_name
            ]
        ability = ability_class(**ability_args)
        self._abilities.append(ability)

    def list_abilities(self) -> list[str]:
        return [
            f"{ability.name()}: {ability.description}" for ability in self._abilities
        ]

    def dump_abilities(self) -> list[CompletionModelFunction]:
        return [ability.spec for ability in self._abilities]

    def get_ability(self, ability_name: str) -> Ability:
        for ability in self._abilities:
            if ability.name() == ability_name:
                return ability
        raise ValueError(f"Ability '{ability_name}' not found.")

    async def perform(self, ability_name: str, **kwargs) -> AbilityResult:
        ability = self.get_ability(ability_name)
        return await ability(**kwargs)
