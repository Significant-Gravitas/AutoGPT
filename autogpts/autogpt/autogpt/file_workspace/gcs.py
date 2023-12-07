"""
The GCSWorkspace class provides an interface for interacting with a file workspace, and
stores the files in a Google Cloud Storage bucket.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

from google.cloud import storage

from autogpt.core.configuration.schema import UserConfigurable

from .base import FileWorkspace, FileWorkspaceConfiguration

logger = logging.getLogger(__name__)


class GCSFileWorkspaceConfiguration(FileWorkspaceConfiguration):
    bucket: str = UserConfigurable("autogpt", from_env="WORKSPACE_STORAGE_BUCKET")


class GCSFileWorkspace(FileWorkspace):
    """A class that represents a Google Cloud Storage workspace."""

    _bucket: storage.Bucket

    def __init__(self, config: GCSFileWorkspaceConfiguration):
        self._bucket_name = config.bucket
        self._root = config.root

        self._gcs = storage.Client()
        super().__init__()

    @property
    def root(self) -> Path:
        """The root directory of the file workspace."""
        return self._root

    @property
    def restrict_to_root(self):
        """Whether to restrict generated paths to the root."""
        return True

    def initialize(self) -> None:
        self._bucket = self._gcs.get_bucket(self._bucket_name)

    def get_path(self, relative_path: str | Path) -> Path:
        return super().get_path(relative_path).relative_to(Path("/"))

    def open_file(self, path: str | Path, mode: str = "r"):
        """Open a file in the workspace."""
        path = self.get_path(path)
        blob = self._bucket.blob(str(path))
        return blob

    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the workspace."""
        blob = self.open_file(path, "r")
        file_content = (
            blob.download_as_text() if not binary else blob.download_as_bytes()
        )
        return file_content

    async def write_file(self, path: str | Path, content: str | bytes):
        """Write to a file in the workspace."""
        blob = self.open_file(path, "w")
        blob.upload_from_string(content) if isinstance(
            content, str
        ) else blob.upload_from_file(content)

        if self.on_write_file:
            path = Path(path)
            if path.is_absolute():
                path = path.relative_to(self.root)
            res = self.on_write_file(path)
            if inspect.isawaitable(res):
                await res

    def list_files(self, path: str | Path = ".") -> list[Path]:
        """List all files in a directory in the workspace."""
        path = self.get_path(path)
        blobs = self._bucket.list_blobs(prefix=str(path))
        return [Path(blob.name) for blob in blobs if not blob.name.endswith("/")]

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the workspace."""
        path = self.get_path(path)
        blob = self._bucket.blob(str(path))
        blob.delete()
