"""
=========
Workspace
=========

The workspace is a directory containing configuration and working files for an AutoGPT
agent.

"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from autogpt.config import Config
from autogpt.logs import logger


class Workspace:
    """A class that represents a workspace for an AutoGPT agent."""

    NULL_BYTES = ["\0", "\000", "\x00", r"\z", "\u0000", "%00"]

    def __init__(self, workspace_root: str | Path, restrict_to_workspace: bool):
        self._root = self._sanitize_path(workspace_root)
        self._restrict_to_workspace = restrict_to_workspace

    @property
    def root(self) -> Path:
        """The root directory of the workspace."""
        return self._root

    @property
    def restrict_to_workspace(self):
        """Whether to restrict generated paths to the workspace."""
        return self._restrict_to_workspace

    @classmethod
    def make_workspace(cls, workspace_directory: str | Path, *args, **kwargs) -> Path:
        """Create a workspace directory and return the path to it.

        Parameters
        ----------
        workspace_directory
            The path to the workspace directory.

        Returns
        -------
        Path
            The path to the workspace directory.

        """
        # TODO: have this make the env file and ai settings file in the directory.
        workspace_directory = cls._sanitize_path(workspace_directory)
        workspace_directory.mkdir(exist_ok=True, parents=True)
        return workspace_directory

    def get_path(self, relative_path: str | Path) -> Path:
        """Get the full path for an item in the workspace.

        Parameters
        ----------
        relative_path
            The relative path to resolve in the workspace.

        Returns
        -------
        Path
            The resolved path relative to the workspace.

        """
        return self._sanitize_path(
            relative_path,
            root=self.root,
            restrict_to_root=self.restrict_to_workspace,
        )

    @staticmethod
    def _sanitize_path(
        relative_path: str | Path,
        root: Optional[str | Path] = None,
        restrict_to_root: bool = True,
    ) -> Path:
        """Resolve the relative path within the given root if possible.

        Parameters
        ----------
        relative_path
            The relative path to resolve.
        root
            The root path to resolve the relative path within.
        restrict_to_root
            Whether to restrict the path to the root.

        Returns
        -------
        Path
            The resolved path.

        Raises
        ------
        ValueError
            If the path is absolute and a root is provided.
        ValueError
            If the path is outside the root and the root is restricted.

        """

        # Posix systems disallow null bytes in paths. Windows is agnostic about it.
        # Do an explicit check here for all sorts of null byte representations.

        for null_byte in Workspace.NULL_BYTES:
            if null_byte in str(relative_path) or null_byte in str(root):
                raise ValueError("embedded null byte")

        if root is None:
            return Path(relative_path).resolve()

        logger.debug(f"Resolving path '{relative_path}' in workspace '{root}'")

        root, relative_path = Path(root).resolve(), Path(relative_path)

        logger.debug(f"Resolved root as '{root}'")

        # Allow exception for absolute paths if they are contained in your workspace directory.
        if (
            relative_path.is_absolute()
            and restrict_to_root
            and not relative_path.is_relative_to(root)
        ):
            raise ValueError(
                f"Attempted to access absolute path '{relative_path}' in workspace '{root}'."
            )

        full_path = root.joinpath(relative_path).resolve()

        logger.debug(f"Joined paths as '{full_path}'")

        if restrict_to_root and not full_path.is_relative_to(root):
            raise ValueError(
                f"Attempted to access path '{full_path}' outside of workspace '{root}'."
            )

        return full_path

    @staticmethod
    def build_file_logger_path(config: Config, workspace_directory: Path):
        file_logger_path = workspace_directory / "file_logger.txt"
        if not file_logger_path.exists():
            with file_logger_path.open(mode="w", encoding="utf-8") as f:
                f.write("File Operation Logger ")
        config.file_logger_path = str(file_logger_path)

    @staticmethod
    def get_workspace_directory(
        config: Config, workspace_directory: Optional[str | Path] = None
    ):
        if workspace_directory is None:
            workspace_directory = Path(__file__).parent / "auto_gpt_workspace"
        elif type(workspace_directory) == str:
            workspace_directory = Path(workspace_directory)
        # TODO: pass in the ai_settings file and the env file and have them cloned into
        #   the workspace directory so we can bind them to the agent.
        workspace_directory = Workspace.make_workspace(workspace_directory)
        config.workspace_path = str(workspace_directory)
        return workspace_directory
