"""
The GCSWorkspace class provides an interface for interacting with a file workspace, and
stores the files in a Google Cloud Storage bucket.
"""
from __future__ import annotations

import inspect
import logging
from io import IOBase
from pathlib import Path

from google.cloud import storage
from google.cloud.exceptions import NotFound

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
        assert self._root.is_absolute()

        self._gcs = storage.Client()
        super().__init__()

    @property
    def root(self) -> Path:
        """The root directory of the file workspace."""
        return self._root

    @property
    def restrict_to_root(self) -> bool:
        """Whether to restrict generated paths to the root."""
        return True

    def initialize(self) -> None:
        logger.debug(f"Initializing {repr(self)}...")
        try:
            self._bucket = self._gcs.get_bucket(self._bucket_name)
        except NotFound:
            logger.info(f"Bucket '{self._bucket_name}' does not exist; creating it...")
            self._bucket = self._gcs.create_bucket(self._bucket_name)

    def get_path(self, relative_path: str | Path) -> Path:
        return super().get_path(relative_path).relative_to("/")

    def _get_blob(self, path: str | Path) -> storage.Blob:
        path = self.get_path(path)
        return self._bucket.blob(str(path))

    def open_file(self, path: str | Path, binary: bool = False) -> IOBase:
        """Open a file in the workspace."""
        blob = self._get_blob(path)
        blob.reload()  # pin revision number to prevent version mixing while reading
        return blob.open("rb" if binary else "r")

    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the workspace."""
        return self.open_file(path, binary).read()

    async def write_file(self, path: str | Path, content: str | bytes) -> None:
        """Write to a file in the workspace."""
        blob = self._get_blob(path)

        if isinstance(content, str):
            blob.upload_from_string(content)
        else:
            blob.upload_from_file(content)

        if self.on_write_file:
            path = Path(path)
            if path.is_absolute():
                path = path.relative_to(self.root)
            res = self.on_write_file(path)
            if inspect.isawaitable(res):
                await res

    def list(self, path: str | Path = ".") -> list[Path]:
        """List all files (recursively) in a directory in the workspace."""
        path = self.get_path(path)
        return [
            Path(blob.name).relative_to(path)
            for blob in self._bucket.list_blobs(
                prefix=f"{path}/" if path != Path(".") else None
            )
        ]

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the workspace."""
        path = self.get_path(path)
        blob = self._bucket.blob(str(path))
        blob.delete()

    def __repr__(self) -> str:
        return f"{__class__.__name__}(bucket='{self._bucket_name}', root={self._root})"
