from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional


if TYPE_CHECKING:
    from autogpt.core.prompting import ChatPrompt
    from autogpt.models.context_item import ContextItem

    from ..base import BaseAgent

from autogpt.commands.system import close_context_item
from autogpt.commands.file_context import open_file, open_folder
from autogpt.models.command import Command
from autogpt.core.resource.model_providers import ChatMessage
from autogpt.models.command import Command
from autogpt.agents.components import (
    BuildPrompt,
    CommandProvider,
    Component,
)


class AgentContext:
    items: list[ContextItem]

    def __init__(self, items: Optional[list[ContextItem]] = None):
        self.items = items or []

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


class ContextComponent(Component, BuildPrompt, CommandProvider):
    def __init__(self):
        self.context = AgentContext()

    def build_prompt(self, result: BuildPrompt.Result) -> None:
        if self.context:
            result.extra_messages.insert(
                0,
                ChatMessage.system(
                    "## Context\n"
                    f"{self.context.format_numbered()}\n\n"
                    "When a context item is no longer needed and you are not done yet, "
                    "you can hide the item by specifying its number in the list above "
                    "to `hide_context_item`.",
                ),
            )

    def get_commands(self) -> Iterator[Command]:
        yield open_file.command
        yield open_folder.command
        yield close_context_item.command
        #TODO kcze
        # No need to check if context is available - it's inside ContextComponent so yes
        # commands.append(
        #     Command(
        #         "open_file",
        #         "Opens a file for editing or continued viewing;"
        #         " creates it if it does not exist yet. "
        #         "Note: If you only need to read or write a file once, use `write_to_file` instead.",
        #         open_file,
        #         [
        #             CommandParameter(
        #                 "file_path",
        #                 JSONSchema(
        #                     type=JSONSchema.Type.STRING,
        #                     description="The path of the file to open",
        #                     required=True,
        #                 ),
        #             )
        #         ],
        #     )
        # )


def get_agent_context(agent: BaseAgent) -> AgentContext | None:
    if hasattr(agent, "context"):
        return getattr(agent, "context")

    return None
