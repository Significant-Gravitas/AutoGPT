"""
The S3Workspace class provides an interface for interacting with a file workspace, and
stores the files in an S3 bucket.
"""
from __future__ import annotations

import contextlib
import inspect
import logging
import os
from io import IOBase, TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import boto3
import botocore.exceptions
from pydantic import SecretStr

from autogpt.core.configuration.schema import UserConfigurable

from .base import FileStorage, FileStorageConfiguration

if TYPE_CHECKING:
    import mypy_boto3_s3

logger = logging.getLogger(__name__)


class S3FileStorageConfiguration(FileStorageConfiguration):
    bucket: str = UserConfigurable("autogpt", from_env="STORAGE_BUCKET")
    s3_endpoint_url: Optional[SecretStr] = UserConfigurable(
        from_env=lambda: SecretStr(v) if (v := os.getenv("S3_ENDPOINT_URL")) else None
    )


class S3FileStorage(FileStorage):
    """A class that represents an S3 storage."""

    _bucket: mypy_boto3_s3.service_resource.Bucket

    def __init__(self, config: S3FileStorageConfiguration):
        self._bucket_name = config.bucket
        self._root = config.root
        assert self._root.is_absolute()

        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
        self._s3 = boto3.resource(
            "s3",
            endpoint_url=config.s3_endpoint_url.get_secret_value()
            if config.s3_endpoint_url
            else None,
        )

        super().__init__()

    @property
    def root(self) -> Path:
        """The root directory of the file storage."""
        return self._root

    @property
    def restrict_to_root(self):
        """Whether to restrict generated paths to the root."""
        return True

    def initialize(self) -> None:
        logger.debug(f"Initializing {repr(self)}...")
        try:
            self._s3.meta.client.head_bucket(Bucket=self._bucket_name)
            self._bucket = self._s3.Bucket(self._bucket_name)
        except botocore.exceptions.ClientError as e:
            if "(404)" not in str(e):
                raise
            logger.info(f"Bucket '{self._bucket_name}' does not exist; creating it...")
            self._bucket = self._s3.create_bucket(Bucket=self._bucket_name)

    def get_path(self, relative_path: str | Path) -> Path:
        return super().get_path(relative_path).relative_to("/")

    def _get_obj(self, path: str | Path) -> mypy_boto3_s3.service_resource.Object:
        """Get an S3 object."""
        path = self.get_path(path)
        obj = self._bucket.Object(str(path))
        with contextlib.suppress(botocore.exceptions.ClientError):
            obj.load()
        return obj

    def open_file(self, path: str | Path, binary: bool = False) -> IOBase:
        """Open a file in the storage."""
        obj = self._get_obj(path)
        return obj.get()["Body"] if binary else TextIOWrapper(obj.get()["Body"])

    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the storage."""
        return self.open_file(path, binary).read()

    async def write_file(self, path: str | Path, content: str | bytes) -> None:
        """Write to a file in the storage."""
        obj = self._get_obj(path)
        obj.put(Body=content)

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
        if path == Path("."):  # root level of bucket
            return [Path(obj.key) for obj in self._bucket.objects.all()]
        else:
            return [
                Path(obj.key).relative_to(path)
                for obj in self._bucket.objects.filter(Prefix=f"{path}/")
            ]

    def list_folders(
        self, path: str | Path = ".", recursive: bool = False
    ) -> list[Path]:
        """List 'directories' directly in a given path or recursively in the storage."""
        path_str = str(path)
        prefix = f"{path_str.strip('/')}/" if path_str != "." else ""
        delimiter = "/" if not recursive else None

        # Initialize an empty set to hold unique folder names
        folder_names = set()

        # List objects with the specified prefix and delimiter
        for obj_summary in self.bucket.objects.filter(
            Prefix=prefix, Delimiter=delimiter
        ):
            if delimiter:
                # If a delimiter is used, we're not listing recursively, so include common prefixes
                response = obj_summary.bucket.meta.client.list_objects_v2(
                    Bucket=self.bucket.name, Prefix=prefix, Delimiter=delimiter
                )
                for prefix_info in response.get("CommonPrefixes", []):
                    folder_names.add(
                        Path(prefix_info["Prefix"]).relative_to(Path(prefix)).parent
                    )
            else:
                # For a recursive list, add all unique 'folder' paths by splitting object keys
                folder_path = Path(obj_summary.key).parent
                if folder_path != Path(prefix).parent:
                    folder_names.add(folder_path.relative_to(Path(prefix).parent))

        return list(folder_names)

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the storage."""
        path = self.get_path(path)
        obj = self._s3.Object(self._bucket_name, str(path))
        obj.delete()

    def exists(self, path: str | Path) -> bool:
        """Check if a file or folder exists in S3 storage."""
        path_str = str(path)
        # Check for exact object match (file)
        obj = self._bucket.Object(path_str)
        try:
            obj.load()  # Will succeed if the object exists
            return True
        except botocore.exceptions.ClientError as e:
            if int(e.response["ResponseMetadata"]["HTTPStatusCode"]) == 404:
                # If the object does not exist, check for objects with the prefix (folder)
                prefix = f"{path_str.rstrip('/')}/"
                objs = list(self._bucket.objects.filter(Prefix=prefix, MaxKeys=1))
                return len(objs) > 0  # True if any objects exist with the prefix
            else:
                raise  # Re-raise for any other client errors

    def make_dir(self, path: str | Path) -> None:
        """Create a directory in the storage if doesn't exist."""
        # S3 does not have directories, so we don't need to do anything
        pass

    def __repr__(self) -> str:
        return f"{__class__.__name__}(bucket='{self._bucket_name}', root={self._root})"
