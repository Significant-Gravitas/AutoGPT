"""
The FileWorkspace class provides an interface for interacting with a file workspace.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class FileWorkspace:
    """A class that represents a file workspace."""

    NULL_BYTES = ["\0", "\000", "\x00", "\u0000"]

    on_write_file: Callable[[Path], Any] | None = None
    """
    Event hook, executed after writing a file.

    Params:
        Path: The path of the file that was written, relative to the workspace root.
    """

    def __init__(self, root: str | Path, restrict_to_root: bool):
        self._root = self._sanitize_path(root)
        self._restrict_to_root = restrict_to_root

    @property
    def root(self) -> Path:
        """The root directory of the file workspace."""
        return self._root

    @property
    def restrict_to_root(self):
        """Whether to restrict generated paths to the root."""
        return self._restrict_to_root

    def initialize(self) -> None:
        self.root.mkdir(exist_ok=True, parents=True)

    def get_path(self, relative_path: str | Path) -> Path:
        """Get the full path for an item in the workspace.

        Parameters:
            relative_path: The relative path to resolve in the workspace.

        Returns:
            Path: The resolved path relative to the workspace.
        """
        return self._sanitize_path(
            relative_path,
            root=self.root,
            restrict_to_root=self.restrict_to_root,
        )

    def open_file(self, path: str | Path, mode: str = "r"):
        """Open a file in the workspace."""
        full_path = self.get_path(path)
        return open(full_path, mode)

    def read_file(self, path: str | Path, binary: bool = False):
        """Read a file in the workspace."""
        with self.open_file(path, "rb" if binary else "r") as file:
            return file.read()

    async def write_file(self, path: str | Path, content: str | bytes):
        """Write to a file in the workspace."""
        with self.open_file(path, "wb" if type(content) is bytes else "w") as file:
            file.write(content)

        if self.on_write_file:
            path = Path(path)
            if path.is_absolute():
                path = path.relative_to(self.root)
            res = self.on_write_file(path)
            if inspect.isawaitable(res):
                await res

    def list_files(self, path: str | Path = "."):
        """List all files in a directory in the workspace."""
        full_path = self.get_path(path)
        return [str(file) for file in full_path.glob("*") if file.is_file()]

    def delete_file(self, path: str | Path):
        """Delete a file in the workspace."""
        full_path = self.get_path(path)
        full_path.unlink()

    @staticmethod
    def _sanitize_path(
        relative_path: str | Path,
        root: Optional[str | Path] = None,
        restrict_to_root: bool = True,
    ) -> Path:
        """Resolve the relative path within the given root if possible.

        Parameters:
            relative_path: The relative path to resolve.
            root: The root path to resolve the relative path within.
            restrict_to_root: Whether to restrict the path to the root.

        Returns:
            Path: The resolved path.

        Raises:
            ValueError: If the path is absolute and a root is provided.
            ValueError: If the path is outside the root and the root is restricted.
        """

        # Posix systems disallow null bytes in paths. Windows is agnostic about it.
        # Do an explicit check here for all sorts of null byte representations.

        for null_byte in FileWorkspace.NULL_BYTES:
            if null_byte in str(relative_path) or null_byte in str(root):
                raise ValueError("embedded null byte")

        if root is None:
            return Path(relative_path).resolve()

        logger.debug(f"Resolving path '{relative_path}' in workspace '{root}'")

        root, relative_path = Path(root).resolve(), Path(relative_path)

        logger.debug(f"Resolved root as '{root}'")

        # Allow absolute paths if they are contained in the workspace.
        if (
            relative_path.is_absolute()
            and restrict_to_root
            and not relative_path.is_relative_to(root)
        ):
            raise ValueError(
                f"Attempted to access absolute path '{relative_path}' "
                f"in workspace '{root}'."
            )

        full_path = root.joinpath(relative_path).resolve()

        logger.debug(f"Joined paths as '{full_path}'")

        if restrict_to_root and not full_path.is_relative_to(root):
            raise ValueError(
                f"Attempted to access path '{full_path}' outside of workspace '{root}'."
            )

        return full_path
