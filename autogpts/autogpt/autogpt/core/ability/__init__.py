"""The command system provides a way to extend the functionality of the AI agent."""
from autogpt.core.ability.base import Ability, AbilityConfiguration, AbilityRegistry
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.ability.simple import (
    AbilityRegistryConfiguration,
    AbilityRegistrySettings,
    SimpleAbilityRegistry,
)

__all__ = [
    "Ability",
    "AbilityConfiguration",
    "AbilityRegistry",
    "AbilityResult",
    "AbilityRegistryConfiguration",
    "AbilityRegistrySettings",
    "SimpleAbilityRegistry",
]
