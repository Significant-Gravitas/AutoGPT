import logging
import os
from pathlib import Path
from typing import Iterator, Optional

from forge.agent import BaseAgentSettings
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.file_storage.base import FileStorage
from forge.json.schema import JSONSchema
from forge.utils.file_operations import decode_textual_file

logger = logging.getLogger(__name__)


class FileManagerComponent(DirectiveProvider, CommandProvider):
    """
    Adds general file manager (e.g. Agent state),
    workspace manager (e.g. Agent output files) support and
    commands to perform operations on files and folders.
    """

    files: FileStorage
    """Agent-related files, e.g. state, logs.
    Use `workspace` to access the agent's workspace files."""

    workspace: FileStorage
    """Workspace that the agent has access to, e.g. for reading/writing files.
    Use `files` to access agent-related files, e.g. state, logs."""

    STATE_FILE = "state.json"
    """The name of the file where the agent's state is stored."""

    def __init__(self, state: BaseAgentSettings, file_storage: FileStorage):
        self.state = state

        if not state.agent_id:
            raise ValueError("Agent must have an ID.")

        self.files = file_storage.clone_with_subroot(f"agents/{state.agent_id}/")
        self.workspace = file_storage.clone_with_subroot(
            f"agents/{state.agent_id}/workspace"
        )
        self._file_storage = file_storage

    async def save_state(self, save_as: Optional[str] = None) -> None:
        """Save the agent's state to the state file."""
        state: BaseAgentSettings = getattr(self, "state")
        if save_as:
            temp_id = state.agent_id
            state.agent_id = save_as
            self._file_storage.make_dir(f"agents/{save_as}")
            # Save state
            await self._file_storage.write_file(
                f"agents/{save_as}/{self.STATE_FILE}", state.json()
            )
            # Copy workspace
            self._file_storage.copy(
                f"agents/{temp_id}/workspace",
                f"agents/{save_as}/workspace",
            )
            state.agent_id = temp_id
        else:
            await self.files.write_file(self.files.root / self.STATE_FILE, state.json())

    def change_agent_id(self, new_id: str):
        """Change the agent's ID and update the file storage accordingly."""
        state: BaseAgentSettings = getattr(self, "state")
        # Rename the agent's files and workspace
        self._file_storage.rename(f"agents/{state.agent_id}", f"agents/{new_id}")
        # Update the file storage objects
        self.files = self._file_storage.clone_with_subroot(f"agents/{new_id}/")
        self.workspace = self._file_storage.clone_with_subroot(
            f"agents/{new_id}/workspace"
        )
        state.agent_id = new_id

    def get_resources(self) -> Iterator[str]:
        yield "The ability to read and write files."

    def get_commands(self) -> Iterator[Command]:
        yield self.read_file
        yield self.write_to_file
        yield self.list_folder

    @command(
        parameters={
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the file to read",
                required=True,
            )
        },
    )
    def read_file(self, filename: str | Path) -> str:
        """Read a file and return the contents

        Args:
            filename (str): The name of the file to read

        Returns:
            str: The contents of the file
        """
        file = self.workspace.open_file(filename, binary=True)
        content = decode_textual_file(file, os.path.splitext(filename)[1], logger)

        return content

    @command(
        ["write_file", "create_file"],
        "Write a file, creating it if necessary. "
        "If the file exists, it is overwritten.",
        {
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the file to write to",
                required=True,
            ),
            "contents": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The contents to write to the file",
                required=True,
            ),
        },
    )
    async def write_to_file(self, filename: str | Path, contents: str) -> str:
        """Write contents to a file

        Args:
            filename (str): The name of the file to write to
            contents (str): The contents to write to the file

        Returns:
            str: A message indicating success or failure
        """
        if directory := os.path.dirname(filename):
            self.workspace.make_dir(directory)
        await self.workspace.write_file(filename, contents)
        return f"File {filename} has been written successfully."

    @command(
        parameters={
            "folder": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The folder to list files in",
                required=True,
            )
        },
    )
    def list_folder(self, folder: str | Path) -> list[str]:
        """Lists files in a folder recursively

        Args:
            folder (str): The folder to search in

        Returns:
            list[str]: A list of files found in the folder
        """
        return [str(p) for p in self.workspace.list_files(folder)]
