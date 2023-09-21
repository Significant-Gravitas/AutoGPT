from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autogpt.core.prompting import ChatPrompt
    from autogpt.models.context_item import ContextItem

    from ..base import BaseAgent

from autogpt.core.resource.model_providers import ChatMessage


class AgentContext:
    items: list[ContextItem]

    def __init__(self, items: list[ContextItem] = []):
        self.items = items

    def __bool__(self) -> bool:
        return len(self.items) > 0

    def __contains__(self, item: ContextItem) -> bool:
        return any([i.source == item.source for i in self.items])

    def add(self, item: ContextItem) -> None:
        self.items.append(item)

    def close(self, index: int) -> None:
        self.items.pop(index - 1)

    def clear(self) -> None:
        self.items.clear()

    def format_numbered(self) -> str:
        return "\n\n".join([f"{i}. {c.fmt()}" for i, c in enumerate(self.items, 1)])


class ContextMixin:
    """Mixin that adds context support to a BaseAgent subclass"""

    context: AgentContext

    def __init__(self, **kwargs: Any):
        self.context = AgentContext()

        super(ContextMixin, self).__init__(**kwargs)

    def construct_base_prompt(self, *args: Any, **kwargs: Any) -> ChatPrompt:
        if kwargs.get("append_messages") is None:
            kwargs["append_messages"] = []

        # Add context section to prompt
        if self.context:
            kwargs["append_messages"].insert(
                0,
                ChatMessage.system(
                    "## Context\n"
                    + self.context.format_numbered()
                    + "\n\nWhen a context item is no longer needed and you are not done yet,"
                    " you can hide the item by specifying its number in the list above"
                    " to `hide_context_item`.",
                ),
            )

        return super(ContextMixin, self).construct_base_prompt(*args, **kwargs)  # type: ignore


def get_agent_context(agent: BaseAgent) -> AgentContext | None:
    if isinstance(agent, ContextMixin):
        return agent.context

    return None
