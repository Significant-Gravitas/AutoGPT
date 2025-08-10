import logging
import os
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict

from forge.agent import BaseAgentSettings
from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.file_storage.base import FileStorage
from forge.models.json_schema import JSONSchema
from forge.utils.file_operations import decode_textual_file

logger = logging.getLogger(__name__)


class FileManagerConfiguration(BaseModel):
    storage_path: str
    """Path to agent files, e.g. state"""
    workspace_path: str
    """Path to files that agent has access to"""

    model_config = ConfigDict(
        # Prevent mutation of the configuration
        # as this wouldn't be reflected in the file storage
        frozen=False
    )


class FileManagerComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[FileManagerConfiguration]
):
    """
    Adds general file manager (e.g. Agent state),
    workspace manager (e.g. Agent output files) support and
    commands to perform operations on files and folders.
    """

    config_class = FileManagerConfiguration

    STATE_FILE = "state.json"
    """The name of the file where the agent's state is stored."""

    def __init__(
        self,
        file_storage: FileStorage,
        agent_state: BaseAgentSettings,
        config: Optional[FileManagerConfiguration] = None,
    ):
        """Initialise the FileManagerComponent.
        Either `agent_id` or `config` must be provided.

        Args:
            file_storage (FileStorage): The file storage instance to use.
            state (BaseAgentSettings): The agent's state.
            config (FileManagerConfiguration, optional): The configuration for
            the file manager. Defaults to None.
        """
        if not agent_state.agent_id:
            raise ValueError("Agent must have an ID.")

        self.agent_state = agent_state

        if not config:
            storage_path = f"agents/{self.agent_state.agent_id}/"
            workspace_path = f"agents/{self.agent_state.agent_id}/workspace"
            ConfigurableComponent.__init__(
                self,
                FileManagerConfiguration(
                    storage_path=storage_path, workspace_path=workspace_path
                ),
            )
        else:
            ConfigurableComponent.__init__(self, config)

        self.storage = file_storage.clone_with_subroot(self.config.storage_path)
        """Agent-related files, e.g. state, logs.
        Use `workspace` to access the agent's workspace files."""
        self.workspace = file_storage.clone_with_subroot(self.config.workspace_path)
        """Workspace that the agent has access to, e.g. for reading/writing files.
        Use `storage` to access agent-related files, e.g. state, logs."""
        self._file_storage = file_storage

    async def save_state(self, save_as_id: Optional[str] = None) -> None:
        """Save the agent's data and state."""
        if save_as_id:
            self._file_storage.make_dir(f"agents/{save_as_id}")
            # Save state
            await self._file_storage.write_file(
                f"agents/{save_as_id}/{self.STATE_FILE}",
                self.agent_state.model_dump_json(),
            )
            # Copy workspace
            self._file_storage.copy(
                self.config.workspace_path,
                f"agents/{save_as_id}/workspace",
            )
        else:
            await self.storage.write_file(
                self.storage.root / self.STATE_FILE, self.agent_state.model_dump_json()
            )

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
