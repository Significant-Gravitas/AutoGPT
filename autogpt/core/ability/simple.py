import logging

from autogpt.core.ability.schema import AbilityResult
from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)

from autogpt.core.ability.base import Ability, AbilityRegistry


class AbilityRegistryConfiguration(SystemConfiguration):
    """Configuration for the AbilityRegistry subsystem."""
    abilities: list[str] = UserConfigurable()


class AbilityRegistrySettings(SystemSettings):
    configuration: AbilityRegistryConfiguration


class SimpleAbilityRegistry(AbilityRegistry, Configurable):

    defaults = AbilityRegistrySettings(
        name="simple_ability_registry",
        description="A simple ability registry.",
        configuration=AbilityRegistryConfiguration(
            abilities=[],
        ),
    )

    def __init__(
        self,
        settings: AbilityRegistrySettings,
        logger: logging.Logger,
    ):
        self._configuration = settings.configuration
        self._logger = logger

    def register_ability(self, ability: Ability) -> None:
        pass

    def list_abilities(self) -> list[str]:
        pass

    def get_ability(self, ability_name: str) -> Ability:
        pass

    def perform(self, ability_name: str, **kwargs) -> AbilityResult:
        pass
