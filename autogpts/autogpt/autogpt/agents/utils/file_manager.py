from pathlib import Path

from autogpt.file_storage.base import FileStorage
from io import IOBase, TextIOBase
from typing import BinaryIO, TextIO

class FileManager:
    """Uses existing FileStorage object to manage file operations.
    This way one FileStorage object can be used in multiple places with custom root folders."""
    
    def __init__(self, file_storage: FileStorage, path: Path = "/"):
        """
        Args:
            file_storage (FileStorage): The file storage object to use for file operations.
            path (Path): The suffix for FileStorage root."""
        self.file_storage = file_storage
        self._root = path
        self.file_storage.make_dir(self.root)

    @property
    def root(self) -> Path:
        """The full root directory of the FileManager."""
        return self.file_storage.root / self._root

    def open_file(self, path: Path, binary: bool = False) -> TextIO | TextIOBase | BinaryIO | IOBase:
        """Returns a readable file-like object representing the file."""
        return self.file_storage.open_file(self.root / path, binary)
    
    def read_file(self, path: Path, binary: bool = False) -> str:
        """Read a file in the storage."""
        return self.file_storage.read_file(self.root / path, binary)

    async def write_file(self, path: Path, content: str | bytes) -> None:
        """Write to a file in the storage."""
        return await self.file_storage.write_file(self.root / path, content)
    
    def list(self, path: Path = ".") -> list[Path]:
        """List all files (recursively) in a directory in the storage."""
        return self.file_storage.list(self.root / path)
    
    def list_folders(self, path: Path = ".", recursive: bool = False) -> list[Path]:
        """List all folders in a directory in the storage."""
        return self.file_storage.list_folders(self.root / path, recursive)
    
    def delete_file(self, path: Path) -> None:
        """Delete a file in the storage."""
        return self.file_storage.delete_file(self.root / path)
    
    def exists(self, path: Path) -> bool:
        """Check if a file exists in the storage."""
        return self.file_storage.exists(self.root / path)
    
    def make_dir(self, path: Path) -> None:
        """Create a directory in the storage if doesn't exist."""
        return self.file_storage.make_dir(self.root / path)