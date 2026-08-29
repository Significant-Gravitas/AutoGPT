import fnmatch
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict

from forge.agent import BaseAgentSettings
from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.file_storage.base import FileStorage
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError
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
        workspace_root: Optional[Path] = None,
    ):
        """Initialise the FileManagerComponent.
        Either `agent_id` or `config` must be provided.

        Args:
            file_storage (FileStorage): The file storage instance to use.
            state (BaseAgentSettings): The agent's state.
            config (FileManagerConfiguration, optional): The configuration for
            the file manager. Defaults to None.
            workspace_root (Path, optional): When provided (CLI mode), indicates that
            file_storage is rooted at the workspace directory. Agent files go to
            .autogpt/agents/{id}/ and the workspace is the current directory.
            When None (server mode), uses sandboxed paths under agents/{id}/.
        """
        if not agent_state.agent_id:
            raise ValueError("Agent must have an ID.")

        self.agent_state = agent_state
        self._workspace_root = workspace_root

        if not config:
            if workspace_root:
                # CLI mode: file_storage root = workspace, agent works in cwd
                storage_path = f".autogpt/agents/{self.agent_state.agent_id}/"
                workspace_path = "."
            else:
                # Server mode: file_storage root = .autogpt, sandboxed workspace
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
            # Determine path prefix based on mode
            if self._workspace_root:
                # CLI mode: file storage is rooted at workspace, state goes to .autogpt/
                agents_prefix = ".autogpt/agents"
            else:
                # Server mode: file storage is rooted at .autogpt/
                agents_prefix = "agents"

            self._file_storage.make_dir(f"{agents_prefix}/{save_as_id}")
            # Save state
            await self._file_storage.write_file(
                f"{agents_prefix}/{save_as_id}/{self.STATE_FILE}",
                self.agent_state.model_dump_json(),
            )
            # Copy workspace (only in server mode, each agent has its own sandbox)
            if not self._workspace_root:
                self._file_storage.copy(
                    self.config.workspace_path,
                    f"{agents_prefix}/{save_as_id}/workspace",
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
        yield self.append_to_file
        yield self.copy_file
        yield self.move_file
        yield self.delete_file
        yield self.search_in_files
        yield self.get_file_info

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

    @command(
        ["append_to_file", "append_file"],
        "Append content to an existing file. Creates the file if it doesn't exist.",
        {
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the file to append to",
                required=True,
            ),
            "contents": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The content to append to the file",
                required=True,
            ),
        },
    )
    async def append_to_file(self, filename: str | Path, contents: str) -> str:
        """Append content to a file, creating it if necessary.

        Args:
            filename (str): The name of the file to append to
            contents (str): The content to append

        Returns:
            str: A message indicating success
        """
        if directory := os.path.dirname(filename):
            self.workspace.make_dir(directory)

        existing_content = ""
        if self.workspace.exists(filename):
            file = self.workspace.open_file(filename, binary=True)
            existing_content = decode_textual_file(
                file, os.path.splitext(filename)[1], logger
            )

        await self.workspace.write_file(filename, existing_content + contents)
        return f"Content appended to {filename} successfully."

    @command(
        parameters={
            "source": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the file or directory to copy",
                required=True,
            ),
            "destination": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The destination path",
                required=True,
            ),
        },
    )
    def copy_file(self, source: str | Path, destination: str | Path) -> str:
        """Copy a file or directory to a new location.

        Args:
            source (str): The source path
            destination (str): The destination path

        Returns:
            str: A message indicating success
        """
        if not self.workspace.exists(source):
            raise CommandExecutionError(f"Source path '{source}' does not exist.")

        if directory := os.path.dirname(destination):
            self.workspace.make_dir(directory)

        self.workspace.copy(source, destination)
        return f"Copied '{source}' to '{destination}' successfully."

    @command(
        ["move_file", "rename_file"],
        "Move or rename a file or directory.",
        {
            "source": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The current path of the file or directory",
                required=True,
            ),
            "destination": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The new path",
                required=True,
            ),
        },
    )
    def move_file(self, source: str | Path, destination: str | Path) -> str:
        """Move or rename a file or directory.

        Args:
            source (str): The source path
            destination (str): The destination path

        Returns:
            str: A message indicating success
        """
        if not self.workspace.exists(source):
            raise CommandExecutionError(f"Source path '{source}' does not exist.")

        if directory := os.path.dirname(destination):
            self.workspace.make_dir(directory)

        self.workspace.rename(source, destination)
        return f"Moved '{source}' to '{destination}' successfully."

    @command(
        ["delete_file", "remove_file"],
        "Delete a file. Use with caution - this operation cannot be undone.",
        {
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the file to delete",
                required=True,
            ),
        },
    )
    def delete_file(self, filename: str | Path) -> str:
        """Delete a file.

        Args:
            filename (str): The name of the file to delete

        Returns:
            str: A message indicating success
        """
        if not self.workspace.exists(filename):
            raise CommandExecutionError(f"File '{filename}' does not exist.")

        self.workspace.delete_file(filename)
        return f"File '{filename}' deleted successfully."

    @command(
        ["search_in_files", "grep_files"],
        "Search for a pattern in files. Returns matches with filenames and lines.",
        {
            "pattern": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The regex pattern to search for",
                required=True,
            ),
            "directory": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The directory to search in (default: current directory)",
                required=False,
            ),
            "file_pattern": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Glob pattern to filter files (e.g., '*.py', '*.txt')",
                required=False,
            ),
            "max_results": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Maximum number of results to return (default: 100)",
                required=False,
            ),
        },
    )
    def search_in_files(
        self,
        pattern: str,
        directory: str | Path = ".",
        file_pattern: str = "*",
        max_results: int = 100,
    ) -> str:
        """Search for a pattern in files.

        Args:
            pattern (str): The regex pattern to search for
            directory (str): The directory to search in
            file_pattern (str): Glob pattern to filter files
            max_results (int): Maximum number of results

        Returns:
            str: Matching lines with file names and line numbers
        """
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise CommandExecutionError(f"Invalid regex pattern: {e}")

        results = []
        files = self.workspace.list_files(directory)

        for file_path in files:
            if not fnmatch.fnmatch(str(file_path), file_pattern):
                continue

            try:
                file = self.workspace.open_file(file_path, binary=True)
                content = decode_textual_file(
                    file, os.path.splitext(file_path)[1], logger
                )

                for line_num, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{file_path}:{line_num}: {line.strip()}")
                        if len(results) >= max_results:
                            break

                if len(results) >= max_results:
                    break
            except Exception:
                # Skip files that can't be read as text
                continue

        if not results:
            return f"No matches found for pattern '{pattern}'"

        header = f"Found {len(results)} match(es)"
        if len(results) >= max_results:
            header += f" (limited to {max_results})"
        header += ":"

        return header + "\n" + "\n".join(results)

    @command(
        ["get_file_info", "file_info", "file_stats"],
        "Get information about a file including size, modification time, and type.",
        {
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the file to get info for",
                required=True,
            ),
        },
    )
    def get_file_info(self, filename: str | Path) -> str:
        """Get information about a file.

        Args:
            filename (str): The name of the file

        Returns:
            str: File information in a readable format
        """
        if not self.workspace.exists(filename):
            raise CommandExecutionError(f"File '{filename}' does not exist.")

        file_path = self.workspace.get_path(filename)
        stat_info = file_path.stat()

        size_bytes = stat_info.st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.2f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.2f} MB"

        modified_time = datetime.fromtimestamp(stat_info.st_mtime)
        created_time = datetime.fromtimestamp(stat_info.st_ctime)

        file_type = "directory" if file_path.is_dir() else "file"
        extension = file_path.suffix if file_path.suffix else "none"

        info = [
            f"File: {filename}",
            f"Type: {file_type}",
            f"Extension: {extension}",
            f"Size: {size_str}",
            f"Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Created: {created_time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        return "\n".join(info)
