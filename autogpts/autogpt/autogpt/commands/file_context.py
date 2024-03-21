"""Commands to perform operations on files"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogpt.agents import Agent, BaseAgent

from autogpt.agents.features.context import ContextMixin
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.command import ValidityResult
from autogpt.models.context_item import FileContextItem, FolderContextItem

from .decorators import sanitize_path_arg

COMMAND_CATEGORY = "file_operations"
COMMAND_CATEGORY_TITLE = "File Operations"


def agent_implements_context(agent: BaseAgent) -> bool:
    return isinstance(agent, ContextMixin)


def is_in_context(agent: Agent, source: str) -> ValidityResult:
    if not isinstance(agent, ContextMixin):
        return ValidityResult(False, f"{agent} does not implement the ContextMixin")

    if agent.context.uses_source(source):
        return ValidityResult(
            False, f"{source} is already loaded into the agent context"
        )
    return ValidityResult(True)


@command(
    "open_file",
    "Opens a file for editing or continued viewing;"
    " creates it if it does not exist yet. "
    "Note: If you only need to read or write a file once, use `write_to_file` instead.",
    {
        "file_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the file to open",
            required=True,
        )
    },
    available=agent_implements_context,
    is_valid=lambda agent, args: is_in_context(agent, args["file_path"]),
)
@sanitize_path_arg("file_path")
async def open_file(file_path: Path, agent: Agent) -> tuple[str, FileContextItem]:
    """Open a file and return a context item

    Args:
        file_path (Path): The path of the file to open

    Returns:
        str: A status message indicating what happened
        FileContextItem: A ContextItem representing the opened file
    """
    # Try to make the file path relative
    relative_file_path = None
    with contextlib.suppress(ValueError):
        relative_file_path = file_path.relative_to(agent.workspace.root)

    created = False
    if not agent.workspace.exists(file_path):
        await agent.workspace.write_file(file_path, "")
        created = True

    file_path = relative_file_path or file_path

    file = FileContextItem(path=file_path)

    return (
        f"File {file_path}{' created,' if created else ''} has been opened"
        " and added to the context ✅",
        file,
    )


@command(
    "open_folder",
    "Open a folder to keep track of its content",
    {
        "path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the folder to open",
            required=True,
        )
    },
    available=agent_implements_context,
    is_valid=lambda agent, args: is_in_context(agent, args["path"]),
)
@sanitize_path_arg("path")
def open_folder(path: Path, agent: Agent) -> tuple[str, FolderContextItem]:
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
        relative_path = path.relative_to(agent.workspace.root)

    if not agent.workspace.exists(path):
        raise FileNotFoundError(f"open_folder {path} failed: no such file or directory")

    path = relative_path or path

    folder = FolderContextItem(path=path)

    return f"Folder {path} has been opened and added to the context ✅", folder
