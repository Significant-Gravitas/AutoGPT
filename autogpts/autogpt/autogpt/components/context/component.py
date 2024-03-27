from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from autogpt.agents import Agent
    from autogpt.models.context_item import ContextItem

from autogpt.agents.utils.exceptions import (
    CommandExecutionError,
    DuplicateOperationError,
    InvalidArgumentError,
)
from autogpt.agents.protocols import CommandProvider, MessageProvider
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.context_item import FileContextItem, FolderContextItem
from autogpt.file_storage.base import FileStorage
from autogpt.models.command_parameter import CommandParameter
from autogpt.agents.components import (
    Component,
)
from autogpt.core.resource.model_providers import ChatMessage
from autogpt.models.command import Command


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


class ContextComponent(Component, MessageProvider, CommandProvider):
    def __init__(self, workspace: FileStorage):
        self.context = AgentContext()
        self.workspace = workspace

    def get_messages(self) -> Iterator[ChatMessage]:
        if self.context:
            yield ChatMessage.system(
                "## Context\n"
                f"{self.context.format_numbered()}\n\n"
                "When a context item is no longer needed and you are not done yet, "
                "you can hide the item by specifying its number in the list above "
                "to `hide_context_item`.",
            )

    def get_commands(self) -> Iterator[Command]:
        yield Command(
                "open_file",
                "Opens a file for editing or continued viewing;"
                " creates it if it does not exist yet. "
                "Note: If you only need to read or write a file once, use `write_to_file` instead.",
                self.open_file,
                [
                    CommandParameter(
                        "file_path",
                        JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="The path of the file to open",
                            required=True,
                        ),
                    )
                ],
            )
        yield Command(
                "open_folder",
                "Open a folder to keep track of its content",
                self.open_folder,
                [
                    CommandParameter(
                        "path",
                        JSONSchema(
                            type=JSONSchema.Type.STRING,
                            description="The path of the folder to open",
                            required=True,
                        ),
                    )
                ],
            )
        yield Command(
                "hide_context_item",
                "Hide an open file, folder or other context item, to save memory.",
                self.close_context_item,
                [
                    CommandParameter(
                        "number",
                        JSONSchema(
                            type=JSONSchema.Type.INTEGER,
                            description="The 1-based index of the context item to hide",
                            required=True,
                        ),
                )
            ],
        )
        
    #TODO kcze return just str
    def open_file(self, file_path: Path, agent: Agent) -> tuple[str, FileContextItem]:
        """Open a file to the context

        Args:
            file_path (Path): The path of the file to open

        Returns:
            str: A status message indicating what happened
            FileContextItem: A ContextItem representing the opened file
        """
        # Try to make the file path relative
        relative_file_path = None
        with contextlib.suppress(ValueError):
            relative_file_path = file_path.relative_to(self.workspace.root)

        created = False
        if not file_path.exists():
            file_path.touch()
            created = True
        elif not file_path.is_file():
            raise CommandExecutionError(f"{file_path} exists but is not a file")

        file_path = relative_file_path or file_path

        file = FileContextItem(
            file_path_in_workspace=file_path,
            workspace_path=self.workspace.root,
        )
        if file in self.context:
            raise DuplicateOperationError(f"The file {file_path} is already open")
        
        self.context.add(file)

        return (
            f"File {file_path}{' created,' if created else ''} has been opened"
            " and added to the context ✅",
            file,
        )

    def open_folder(self, path: Path, agent: Agent) -> tuple[str, FolderContextItem]:
        """Open a folder and return a context item

        Args:
            path (Path): The path of the folder to open

        Returns:
            str: A status message indicating what happened
            FolderContextItem: A ContextItem representing the opened folder
        """
        # Try to make the path relative
        relative_path = None
        with contextlib.suppress(ValueError):
            relative_path = path.relative_to(self.workspace.root)

        if not path.exists():
            raise FileNotFoundError(f"open_folder {path} failed: no such file or directory")
        elif not path.is_dir():
            raise CommandExecutionError(f"{path} exists but is not a folder")

        path = relative_path or path

        folder = FolderContextItem(
            path_in_workspace=path,
            workspace_path=self.workspace.root,
        )
        if folder in self.context:
            raise DuplicateOperationError(f"The folder {path} is already open")
        
        self.context.add(folder)

        return f"Folder {path} has been opened and added to the context ✅", folder

    def close_context_item(self, number: int, agent: Agent) -> str:
        assert (context := agent.context.context) is not None

        if number > len(context.items) or number == 0:
            raise InvalidArgumentError(f"Index {number} out of range")

        context.close(number)
        return f"Context item {number} hidden ✅"
