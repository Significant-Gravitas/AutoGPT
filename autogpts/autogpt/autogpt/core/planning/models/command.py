from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from autogpt.core.agents.base import agent

from .context_items import ContextItem

AbilityReturnValue = Any
AbilityOutput = AbilityReturnValue | tuple[AbilityReturnValue, ContextItem]


class AbilityParameter:
    name: str
    type: str
    description: str
    required: bool

    def __repr__(self):
        return f"AbilityParameter('{self.name}', '{self.type}', '{self.description}', {self.required})"