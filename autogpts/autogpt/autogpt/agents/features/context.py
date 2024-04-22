import contextlib
from pathlib import Path
from typing import Iterator, Optional

from autogpt.agents.protocols import CommandProvider, MessageProvider
from autogpt.command_decorator import command
from autogpt.core.resource.model_providers import ChatMessage
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.file_storage.base import FileStorage
from autogpt.models.command import Command
from autogpt.models.context_item import ContextItem, FileContextItem, FolderContextItem
from autogpt.utils.exceptions import InvalidArgumentError


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

    def format_numbered(self, workspace: FileStorage) -> str:
        return "\n\n".join(
            [f"{i}. {c.fmt(workspace)}" for i, c in enumerate(self.items, 1)]
        )


class ContextComponent(MessageProvider, CommandProvider):
    """Adds ability to keep files and folders open in the context (prompt)."""

    def __init__(self, workspace: FileStorage):
        self.context = AgentContext()
        self.workspace = workspace

    def get_messages(self) -> Iterator[ChatMessage]:
        if self.context:
            yield ChatMessage.system(
                "## Context\n"
                f"{self.context.format_numbered(self.workspace)}\n\n"
                "When a context item is no longer needed and you are not done yet, "
                "you can hide the item by specifying its number in the list above "
                "to `hide_context_item`.",
            )

    def get_commands(self) -> Iterator[Command]:
        yield self.open_file
        yield self.open_folder
        if self.context:
            yield self.close_context_item

    @command(
        parameters={
            "file_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the file to open",
                required=True,
            )
        }
    )
    async def open_file(self, file_path: Path) -> str:
        """Opens a file for editing or continued viewing;
        creates it if it does not exist yet.
        Note: If you only need to read or write a file once,
        use `write_to_file` instead.

        Args:
            file_path (Path): The path of the file to open

        Returns:
            str: A status message indicating what happened
        """
        # Try to make the file path relative
        relative_file_path = None
        with contextlib.suppress(ValueError):
            relative_file_path = file_path.relative_to(self.workspace.root)

        created = False
        if not self.workspace.exists(file_path):
            await self.workspace.write_file(file_path, "")
            created = True

        file_path = relative_file_path or file_path

        file = FileContextItem(path=file_path)

        self.context.add(file)

        return (
            f"File {file_path}{' created,' if created else ''} has been opened"
            " and added to the context ✅"
        )

    @command(
        parameters={
            "path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the folder to open",
                required=True,
            )
        }
    )
    def open_folder(self, path: Path) -> str:
        """Open a folder to keep track of its content

        Args:
            path (Path): The path of the folder to open

        Returns:
            str: A status message indicating what happened
        """
        # Try to make the path relative
        relative_path = None
        with contextlib.suppress(ValueError):
            relative_path = path.relative_to(self.workspace.root)

        if not self.workspace.exists(path):
            raise FileNotFoundError(
                f"open_folder {path} failed: no such file or directory"
            )

        path = relative_path or path

        folder = FolderContextItem(path=path)

        self.context.add(folder)

        return f"Folder {path} has been opened and added to the context ✅"

    @command(
        parameters={
            "number": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="The 1-based index of the context item to hide",
                required=True,
            )
        }
    )
    def close_context_item(self, number: int) -> str:
        """Hide an open file, folder or other context item, to save tokens.

        Args:
            number (int): The 1-based index of the context item to hide

        Returns:
            str: A status message indicating what happened
        """
        if number > len(self.context.items) or number == 0:
            raise InvalidArgumentError(f"Index {number} out of range")

        self.context.close(number)
        return f"Context item {number} hidden ✅"
