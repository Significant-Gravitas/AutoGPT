"""
The LocalFileStorage class implements a FileStorage that works with local files.
"""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Generator, Literal

from .base import FileStorage, FileStorageConfiguration

logger = logging.getLogger(__name__)


class LocalFileStorage(FileStorage):
    """A class that represents a file storage."""

    def __init__(self, config: FileStorageConfiguration):
        self._root = config.root.resolve()
        self._restrict_to_root = config.restrict_to_root
        self.make_dir(self.root)
        super().__init__()

    @property
    def root(self) -> Path:
        """The root directory of the file storage."""
        return self._root

    @property
    def restrict_to_root(self) -> bool:
        """Whether to restrict generated paths to the root."""
        return self._restrict_to_root

    @property
    def is_local(self) -> bool:
        """Whether the storage is local (i.e. on the same machine, not cloud-based)."""
        return True

    def initialize(self) -> None:
        self.root.mkdir(exist_ok=True, parents=True)

    def open_file(
        self, path: str | Path, mode: Literal["w", "r"] = "r", binary: bool = False
    ) -> IO:
        """Open a file in the storage."""
        return self._open_file(path, f"{mode}b" if binary else mode)

    def _open_file(self, path: str | Path, mode: str) -> IO:
        full_path = self.get_path(path)
        return open(full_path, mode)  # type: ignore

    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the storage."""
        with self._open_file(path, "rb" if binary else "r") as file:
            return file.read()

    async def write_file(self, path: str | Path, content: str | bytes) -> None:
        """Write to a file in the storage."""
        with self._open_file(path, "wb" if type(content) is bytes else "w") as file:
            file.write(content)

        if self.on_write_file:
            path = Path(path)
            if path.is_absolute():
                path = path.relative_to(self.root)
            res = self.on_write_file(path)
            if inspect.isawaitable(res):
                await res

    def list_files(self, path: str | Path = ".") -> list[Path]:
        """List all files (recursively) in a directory in the storage."""
        path = self.get_path(path)
        return [file.relative_to(path) for file in path.rglob("*") if file.is_file()]

    def list_folders(
        self, path: str | Path = ".", recursive: bool = False
    ) -> list[Path]:
        """List directories directly in a given path or recursively."""
        path = self.get_path(path)
        if recursive:
            return [
                folder.relative_to(path)
                for folder in path.rglob("*")
                if folder.is_dir()
            ]
        else:
            return [
                folder.relative_to(path) for folder in path.iterdir() if folder.is_dir()
            ]

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the storage."""
        full_path = self.get_path(path)
        full_path.unlink()

    def delete_dir(self, path: str | Path) -> None:
        """Delete an empty folder in the storage."""
        full_path = self.get_path(path)
        full_path.rmdir()

    def exists(self, path: str | Path) -> bool:
        """Check if a file or folder exists in the storage."""
        return self.get_path(path).exists()

    def make_dir(self, path: str | Path) -> None:
        """Create a directory in the storage if doesn't exist."""
        full_path = self.get_path(path)
        full_path.mkdir(exist_ok=True, parents=True)

    def rename(self, old_path: str | Path, new_path: str | Path) -> None:
        """Rename a file or folder in the storage."""
        old_path = self.get_path(old_path)
        new_path = self.get_path(new_path)
        old_path.rename(new_path)

    def copy(self, source: str | Path, destination: str | Path) -> None:
        """Copy a file or folder with all contents in the storage."""
        source = self.get_path(source)
        destination = self.get_path(destination)
        if source.is_file():
            destination.write_bytes(source.read_bytes())
        else:
            destination.mkdir(exist_ok=True, parents=True)
            for file in source.rglob("*"):
                if file.is_file():
                    target = destination / file.relative_to(source)
                    target.parent.mkdir(exist_ok=True, parents=True)
                    target.write_bytes(file.read_bytes())

    def clone_with_subroot(self, subroot: str | Path) -> FileStorage:
        """Create a new LocalFileStorage with a subroot of the current storage."""
        return LocalFileStorage(
            FileStorageConfiguration(
                root=self.get_path(subroot),
                restrict_to_root=self.restrict_to_root,
            )
        )

    @contextmanager
    def mount(self, path: str | Path = ".") -> Generator[Path, Any, None]:
        """Mount the file storage and provide a local path."""
        # No need to do anything for local storage
        yield Path(self.get_path(".")).absolute()
