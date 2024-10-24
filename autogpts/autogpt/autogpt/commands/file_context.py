"""Commands to perform operations on files"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from autogpt.agents.features.context import ContextMixin, get_agent_context
from autogpt.agents.utils.exceptions import (
    CommandExecutionError,
    DuplicateOperationError,
)
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.context_item import FileContextItem, FolderContextItem

from .decorators import sanitize_path_arg

COMMAND_CATEGORY = "file_operations"
COMMAND_CATEGORY_TITLE = "File Operations"


if TYPE_CHECKING:
    from autogpt.agents import Agent, BaseAgent


def agent_implements_context(agent: BaseAgent) -> bool:
    return isinstance(agent, ContextMixin)


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
)
@sanitize_path_arg("file_path")
def open_file(file_path: Path, agent: Agent) -> tuple[str, FileContextItem]:
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

    assert (agent_context := get_agent_context(agent)) is not None

    created = False
    if not file_path.exists():
        file_path.touch()
        created = True
    elif not file_path.is_file():
        raise CommandExecutionError(f"{file_path} exists but is not a file")

    file_path = relative_file_path or file_path

    file = FileContextItem(
        file_path_in_workspace=file_path,
        workspace_path=agent.workspace.root,
    )
    if file in agent_context:
        raise DuplicateOperationError(f"The file {file_path} is already open")

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

    assert (agent_context := get_agent_context(agent)) is not None

    if not path.exists():
        raise FileNotFoundError(f"open_folder {path} failed: no such file or directory")
    elif not path.is_dir():
        raise CommandExecutionError(f"{path} exists but is not a folder")

    path = relative_path or path

    folder = FolderContextItem(
        path_in_workspace=path,
        workspace_path=agent.workspace.root,
    )
    if folder in agent_context:
        raise DuplicateOperationError(f"The folder {path} is already open")

    return f"Folder {path} has been opened and added to the context ✅", folder
