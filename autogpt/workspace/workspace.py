"""
=========
Workspace
=========

The workspace is a directory containing configuration and working files for an AutoGPT
agent.

"""
from __future__ import annotations

from pathlib import Path


class Workspace:
    """A class that represents a workspace for an AutoGPT agent."""

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
        root: str | Path = None,
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

        if root is None:
            return Path(relative_path).resolve()

        root, relative_path = Path(root), Path(relative_path)

        if relative_path.is_absolute():
            raise ValueError(
                f"Attempted to access absolute path '{relative_path}' in workspace '{root}'."
            )

        full_path = root.joinpath(relative_path).resolve()

        if restrict_to_root and not full_path.is_relative_to(root):
            raise ValueError(
                f"Attempted to access path '{full_path}' outside of workspace '{root}'."
            )

        return full_path
