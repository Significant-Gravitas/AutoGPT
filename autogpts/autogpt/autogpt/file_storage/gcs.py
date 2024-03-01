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

from .base import FileStorage, FileStorageConfiguration

logger = logging.getLogger(__name__)


class GCSFileStorageConfiguration(FileStorageConfiguration):
    bucket: str = UserConfigurable("autogpt", from_env="STORAGE_BUCKET")


class GCSFileStorage(FileStorage):
    """A class that represents a Google Cloud Storage."""

    _bucket: storage.Bucket

    def __init__(self, config: GCSFileStorageConfiguration):
        self._bucket_name = config.bucket
        self._root = config.root
        assert self._root.is_absolute()

        self._gcs = storage.Client()
        super().__init__()

    @property
    def root(self) -> Path:
        """The root directory of the file storage."""
        return self._root

    @property
    def restrict_to_root(self) -> bool:
        """Whether to restrict generated paths to the root."""
        return True

    @property
    def is_local(self) -> bool:
        """Whether the storage is local (i.e. on the same machine, not cloud-based)."""
        return False

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
        """Open a file in the storage."""
        blob = self._get_blob(path)
        blob.reload()  # pin revision number to prevent version mixing while reading
        return blob.open("rb" if binary else "r")

    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the storage."""
        return self.open_file(path, binary).read()

    async def write_file(self, path: str | Path, content: str | bytes) -> None:
        """Write to a file in the storage."""
        blob = self._get_blob(path)

        blob.upload_from_string(
            data=content,
            content_type=(
                "text/plain"
                if type(content) is str
                # TODO: get MIME type from file extension or binary content
                else "application/octet-stream"
            ),
        )

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
        return [
            Path(blob.name).relative_to(path)
            for blob in self._bucket.list_blobs(
                prefix=f"{path}/" if path != Path(".") else None
            )
        ]

    def list_folders(
        self, path: str | Path = ".", recursive: bool = False
    ) -> list[Path]:
        """List 'directories' directly in a given path or recursively in the storage."""
        path = str(path)
        prefix = f"{path}/" if path != "." else ""
        delimiter = "/" if not recursive else None
        iterator = self._bucket.list_blobs(prefix=prefix, delimiter=delimiter)
        prefixes = set()
        for page in iterator.pages:
            prefixes.update(page.prefixes)
        return [Path(p).relative_to(Path(prefix)) for p in prefixes]

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the storage."""
        path = self.get_path(path)
        blob = self._bucket.blob(str(path))
        blob.delete()

    def delete_dir(self, path: str | Path) -> None:
        """Delete an empty folder in the storage."""
        # Since GCS does not have directories, we don't need to do anything
        pass

    def exists(self, path: str | Path) -> bool:
        """Check if a file or folder exists in GCS storage."""
        path = self.get_path(path)
        # Check for exact blob match (file)
        blob = self._bucket.blob(str(path))
        if blob.exists():
            return True
        # Check for any blobs with prefix (folder)
        prefix = f"{str(path).rstrip('/')}/"
        blobs = self._bucket.list_blobs(prefix=prefix, max_results=1)
        for _ in blobs:
            return True  # If there is at least one object, the folder exists
        return False

    def make_dir(self, path: str | Path) -> None:
        """Create a directory in the storage if doesn't exist."""
        # GCS does not have directories, so we don't need to do anything
        pass

    def clone_with_subroot(self, subroot: str | Path) -> GCSFileStorage:
        """Create a new GCSFileStorage with a subroot of the current storage."""
        file_storage = GCSFileStorage(
            GCSFileStorageConfiguration(
                root=self.get_path(subroot),
                bucket=self._bucket_name,
            )
        )
        file_storage._gcs = self._gcs
        file_storage._bucket = self._bucket
        return file_storage

    def __repr__(self) -> str:
        return f"{__class__.__name__}(bucket='{self._bucket_name}', root={self._root})"
