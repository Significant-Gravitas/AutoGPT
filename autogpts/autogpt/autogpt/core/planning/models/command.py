from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from autogpt.core.agents.base import agent

from .context_items import ContextItem

ToolReturnValue = Any
ToolOutput = ToolReturnValue | tuple[ToolReturnValue, ContextItem]


class ToolParameter:
    name: str
    type: str
    description: str
    required: bool

    def __repr__(self):
        return f"ToolParameter('{self.name}', '{self.type}', '{self.description}', {self.required})"
