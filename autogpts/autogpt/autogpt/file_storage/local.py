"""
The LocalFileStorage class implements a FileStorage that works with local files.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import IO

from .base import FileStorage, FileStorageConfiguration

logger = logging.getLogger(__name__)


class LocalFileStorage(FileStorage):
    """A class that represents a file storage."""

    def __init__(self, config: FileStorageConfiguration):
        self._root = self._sanitize_path(config.root)
        self._restrict_to_root = config.restrict_to_root
        super().__init__()

    @property
    def root(self) -> Path:
        """The root directory of the file storage."""
        return self._root

    @property
    def restrict_to_root(self) -> bool:
        """Whether to restrict generated paths to the root."""
        return self._restrict_to_root

    def initialize(self) -> None:
        self.root.mkdir(exist_ok=True, parents=True)

    def open_file(self, path: str | Path, binary: bool = False) -> IO:
        """Open a file in the storage."""
        return self._open_file(path, "rb" if binary else "r")

    def _open_file(self, path: str | Path, mode: str = "r") -> IO:
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

    def list(self, path: str | Path = ".") -> list[Path]:
        """List all files (recursively) in a directory in the storage."""
        path = self.get_path(path)
        return [file.relative_to(path) for file in path.rglob("*") if file.is_file()]
    
    def list_folders(self, path: str | Path = ".", recursive: bool = False) -> list[Path]:
        """List directories directly in a given path or recursively."""
        path = self.get_path(path)
        if recursive:
            return [folder.relative_to(path) for folder in path.rglob("*") if folder.is_dir()]
        else:
            return [folder.relative_to(path) for folder in path.iterdir() if folder.is_dir()]

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the storage."""
        full_path = self.get_path(path)
        full_path.unlink()

    def exists(self, path: str | Path) -> bool:
        """Check if a file or folder exists in the storage."""
        return self.get_path(path).exists()

    def make_dir(self, path: str | Path) -> None:
        """Create a directory in the storage if doesn't exist."""
        full_path = self.get_path(path)
        full_path.mkdir(exist_ok=True, parents=True)