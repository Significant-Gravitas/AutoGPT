"""Commands to perform operations on files"""

from __future__ import annotations

COMMAND_CATEGORY = "file_operations"
COMMAND_CATEGORY_TITLE = "File Operations"

import contextlib
from pathlib import Path

from autogpt.agents.agent import Agent
from autogpt.agents.utils.context import get_agent_context
from autogpt.agents.utils.exceptions import (
    CommandExecutionError,
    DuplicateOperationError,
)
from autogpt.command_decorator import command
from autogpt.models.context_item import FileContextItem, FolderContextItem

from .decorators import sanitize_path_arg


@command(
    "open_file",
    "Open a file for editing, creating it if it does not exist yet",
    {
        "file_path": {
            "type": "string",
            "description": "The path of the file to open",
            "required": True,
        }
    },
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
    with contextlib.suppress(ValueError):
        file_path = file_path.relative_to(agent.workspace.root)

    if (agent_context := get_agent_context(agent)) is None:
        raise NotImplementedError(
            f"{agent.__class__.__name__} does not implement context"
        )

    created = False
    if not file_path.exists():
        file_path.touch()
        created = True
    elif not file_path.is_file():
        raise CommandExecutionError(f"{file_path} exists but is not a file")

    file = FileContextItem(file_path)
    if file in agent_context:
        raise DuplicateOperationError(f"The file {file_path} is already open")

    return (
        f"File {file}{' created,' if created else ''} opened and added to context ✅",
        file,
    )


@command(
    "open_folder",
    "Open a folder to keep track of its content",
    {
        "path": {
            "type": "string",
            "description": "The path of the folder to open",
            "required": True,
        }
    },
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
    with contextlib.suppress(ValueError):
        path = path.relative_to(agent.workspace.root)

    if (agent_context := get_agent_context(agent)) is None:
        raise NotImplementedError(
            f"{agent.__class__.__name__} does not implement context"
        )

    if not path.exists():
        raise FileNotFoundError(f"open_folder {path} failed: no such file or directory")
    elif not path.is_dir():
        raise CommandExecutionError(f"{path} exists but is not a folder")

    folder = FolderContextItem(path)
    if folder in agent_context:
        raise DuplicateOperationError(f"The folder {path} is already open")

    return f"Folder {folder} opened and added to context ✅", folder
