import logging
import os
import os.path
from pathlib import Path
from typing import Iterator

from autogpt.agents.base import CommandArgs
from autogpt.agents.protocols import CommandProvider, DirectiveProvider
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.file_storage.base import FileStorage
from autogpt.models.command import Command, ValidityResult

from ..agents.base import BaseAgentSettings
from ..file_operations_utils import decode_textual_file, text_checksum

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

    async def save_state(self) -> None:
        """Save the agent's state to the state file."""
        state: BaseAgentSettings = getattr(self, "state")
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
        yield Command.from_decorated_function(self.read_file)
        yield Command.from_decorated_function(self.write_to_file)
        yield Command.from_decorated_function(self.list_folder)

    def _is_write_valid(self, arguments: CommandArgs) -> ValidityResult:
        if not self.workspace.exists(arguments["filename"]):
            return ValidityResult(True)

        if self.workspace.read_file(arguments["filename"]):
            if text_checksum(arguments["contents"]) == text_checksum(
                self.workspace.read_file(arguments["filename"])
            ):
                return ValidityResult(
                    False, "Trying to write the same content to the same file."
                )

        return ValidityResult(True)

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
        logger.info(f"self: {self}")
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
