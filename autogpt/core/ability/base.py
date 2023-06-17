import abc

import inflection

from autogpt.core.ability.schema import AbilityRequirements, AbilityResult


class Ability(abc.ABC):
    """A class representing an agent ability."""

    @property
    def name(self) -> str:
        """The name of the ability."""
        return inflection.underscore(self.__class__.__name__)

    @property
    def signature(self) -> str:
        return " ".join(self.arguments) if self.arguments else ""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """A detailed description of what the ability does."""
        ...

    @property
    @abc.abstractmethod
    def limitations(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def arguments(self) -> list[str]:
        ...

    @property
    @abc.abstractmethod
    def requirements(self) -> AbilityRequirements:
        ...

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> AbilityResult:
        ...

    def __str__(self) -> str:
        return f"{self.name}: {self.description}, args: {self.signature}"


class AbilityRegistry(abc.ABC):

    @abc.abstractmethod
    def register_ability(self, ability: Ability) -> None:
        ...

    @abc.abstractmethod
    def list_abilities(self) -> list[str]:
        ...

    @abc.abstractmethod
    def get_ability(self, ability_name: str) -> Ability:
        ...

    @abc.abstractmethod
    def perform(self, ability_name: str, **kwargs) -> AbilityResult:
        ...
