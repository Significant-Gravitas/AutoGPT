"""
The FileStorage class provides an interface for interacting with a file storage.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from io import IOBase, TextIOBase
from pathlib import Path
from typing import IO, Any, BinaryIO, Callable, Literal, TextIO, overload

from autogpt.core.configuration.schema import SystemConfiguration

logger = logging.getLogger(__name__)


class FileStorageConfiguration(SystemConfiguration):
    restrict_to_root: bool = True
    root: Path = Path("/")


class FileStorage(ABC):
    """A class that represents a file storage."""

    on_write_file: Callable[[Path], Any] | None = None
    """
    Event hook, executed after writing a file.

    Params:
        Path: The path of the file that was written, relative to the storage root.
    """

    @property
    @abstractmethod
    def root(self) -> Path:
        """The root path of the file storage."""

    @property
    @abstractmethod
    def restrict_to_root(self) -> bool:
        """Whether to restrict file access to within the storage's root path."""

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Whether the storage is local (i.e. on the same machine, not cloud-based)."""

    @abstractmethod
    def initialize(self) -> None:
        """
        Calling `initialize()` should bring the storage to a ready-to-use state.
        For example, it can create the resource in which files will be stored, if it
        doesn't exist yet. E.g. a folder on disk, or an S3 Bucket.
        """

    @overload
    @abstractmethod
    def open_file(
        self,
        path: str | Path,
        mode: Literal["w", "r"] = "r",
        binary: Literal[False] = False,
    ) -> TextIO | TextIOBase:
        """Returns a readable text file-like object representing the file."""

    @overload
    @abstractmethod
    def open_file(
        self,
        path: str | Path,
        mode: Literal["w", "r"] = "r",
        binary: Literal[True] = True,
    ) -> BinaryIO | IOBase:
        """Returns a readable binary file-like object representing the file."""

    @abstractmethod
    def open_file(
        self, path: str | Path, mode: Literal["w", "r"] = "r", binary: bool = False
    ) -> IO | IOBase:
        """Returns a readable file-like object representing the file."""

    @overload
    @abstractmethod
    def read_file(self, path: str | Path, binary: Literal[False] = False) -> str:
        """Read a file in the storage as text."""
        ...

    @overload
    @abstractmethod
    def read_file(self, path: str | Path, binary: Literal[True] = True) -> bytes:
        """Read a file in the storage as binary."""
        ...

    @abstractmethod
    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the storage."""

    @abstractmethod
    async def write_file(self, path: str | Path, content: str | bytes) -> None:
        """Write to a file in the storage."""

    @abstractmethod
    def list_files(self, path: str | Path = ".") -> list[Path]:
        """List all files (recursively) in a directory in the storage."""

    @abstractmethod
    def list_folders(
        self, path: str | Path = ".", recursive: bool = False
    ) -> list[Path]:
        """List all folders in a directory in the storage."""

    @abstractmethod
    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the storage."""

    @abstractmethod
    def delete_dir(self, path: str | Path) -> None:
        """Delete an empty folder in the storage."""

    @abstractmethod
    def exists(self, path: str | Path) -> bool:
        """Check if a file or folder exists in the storage."""

    @abstractmethod
    def rename(self, old_path: str | Path, new_path: str | Path) -> None:
        """Rename a file or folder in the storage."""

    @abstractmethod
    def make_dir(self, path: str | Path) -> None:
        """Create a directory in the storage if doesn't exist."""

    @abstractmethod
    def clone_with_subroot(self, subroot: str | Path) -> FileStorage:
        """Create a new FileStorage with a subroot of the current storage."""

    def get_path(self, relative_path: str | Path) -> Path:
        """Get the full path for an item in the storage.

        Parameters:
            relative_path: The relative path to resolve in the storage.

        Returns:
            Path: The resolved path relative to the storage.
        """
        return self._sanitize_path(relative_path)

    def _sanitize_path(
        self,
        path: str | Path,
    ) -> Path:
        """Resolve the relative path within the given root if possible.

        Parameters:
            relative_path: The relative path to resolve.

        Returns:
            Path: The resolved path.

        Raises:
            ValueError: If the path is absolute and a root is provided.
            ValueError: If the path is outside the root and the root is restricted.
        """

        # Posix systems disallow null bytes in paths. Windows is agnostic about it.
        # Do an explicit check here for all sorts of null byte representations.
        if "\0" in str(path):
            raise ValueError("Embedded null byte")

        logger.debug(f"Resolving path '{path}' in storage '{self.root}'")

        relative_path = Path(path)

        # Allow absolute paths if they are contained in the storage.
        if (
            relative_path.is_absolute()
            and self.restrict_to_root
            and not relative_path.is_relative_to(self.root)
        ):
            raise ValueError(
                f"Attempted to access absolute path '{relative_path}' "
                f"in storage '{self.root}'"
            )

        full_path = self.root / relative_path
        if self.is_local:
            full_path = full_path.resolve()
        else:
            full_path = Path(os.path.normpath(full_path))

        logger.debug(f"Joined paths as '{full_path}'")

        if self.restrict_to_root and not full_path.is_relative_to(self.root):
            raise ValueError(
                f"Attempted to access path '{full_path}' "
                f"outside of storage '{self.root}'."
            )

        return full_path
