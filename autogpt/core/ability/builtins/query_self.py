import logging

from autogpt.core.ability.base import Ability
from autogpt.core.ability.schema import (
    AbilityArguments,
    AbilityRequirements,
    AbilityResult,
)
from autogpt.core.workspace import Workspace


class QuerySelf(Ability):


    @property
    def name(self) -> str:
        return "query_self"

    @property
    def description(self) -> str:
        return ""

    @property
    def arguments(self) -> list[str]:
        pass

    @property
    def limitations(self) -> str:
        pass

    @property
    def requirements(self) -> AbilityRequirements:
        pass

    def __call__(self, *args, **kwargs) -> AbilityResult:
        pass