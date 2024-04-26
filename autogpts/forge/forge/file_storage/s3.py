"""
The S3Workspace class provides an interface for interacting with a file workspace, and
stores the files in an S3 bucket.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from io import IOBase, TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import boto3
import botocore.exceptions
from pydantic import SecretStr

from forge.config.schema import UserConfigurable

from .base import FileStorage, FileStorageConfiguration

if TYPE_CHECKING:
    import mypy_boto3_s3

logger = logging.getLogger(__name__)


class S3FileStorageConfiguration(FileStorageConfiguration):
    bucket: str = UserConfigurable("autogpt", from_env="STORAGE_BUCKET")
    s3_endpoint_url: Optional[SecretStr] = UserConfigurable(from_env="S3_ENDPOINT_URL")


class S3FileStorage(FileStorage):
    """A class that represents an S3 storage."""

    _bucket: mypy_boto3_s3.service_resource.Bucket

    def __init__(self, config: S3FileStorageConfiguration):
        self._bucket_name = config.bucket
        self._root = config.root
        # Add / at the beginning of the root path
        if not self._root.is_absolute():
            self._root = Path("/").joinpath(self._root)

        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
        self._s3 = boto3.resource(
            "s3",
            endpoint_url=(
                config.s3_endpoint_url.get_secret_value()
                if config.s3_endpoint_url
                else None
            ),
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

    @property
    def is_local(self) -> bool:
        """Whether the storage is local (i.e. on the same machine, not cloud-based)."""
        return False

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
        # We set S3 root with "/" at the beginning
        # but relative_to("/") will remove it
        # because we don't actually want it in the storage filenames
        return super().get_path(relative_path).relative_to("/")

    def _get_obj(self, path: str | Path) -> mypy_boto3_s3.service_resource.Object:
        """Get an S3 object."""
        path = self.get_path(path)
        obj = self._bucket.Object(str(path))
        with contextlib.suppress(botocore.exceptions.ClientError):
            obj.load()
        return obj

    def open_file(
        self, path: str | Path, mode: Literal["w", "r"] = "r", binary: bool = False
    ) -> IOBase:
        """Open a file in the storage."""
        obj = self._get_obj(path)
        return obj.get()["Body"] if binary else TextIOWrapper(obj.get()["Body"])

    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the storage."""
        return self.open_file(path, binary=binary).read()

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
        path = self.get_path(path)
        folder_names = set()

        # List objects with the specified prefix and delimiter
        for obj_summary in self._bucket.objects.filter(Prefix=str(path)):
            # Remove path prefix and the object name (last part)
            folder = Path(obj_summary.key).relative_to(path).parent
            if not folder or folder == Path("."):
                continue
            # For non-recursive, only add the first level of folders
            if not recursive:
                folder_names.add(folder.parts[0])
            else:
                # For recursive, need to add all nested folders
                for i in range(len(folder.parts)):
                    folder_names.add("/".join(folder.parts[: i + 1]))

        return [Path(f) for f in folder_names]

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the storage."""
        path = self.get_path(path)
        obj = self._s3.Object(self._bucket_name, str(path))
        obj.delete()

    def delete_dir(self, path: str | Path) -> None:
        """Delete an empty folder in the storage."""
        # S3 does not have directories, so we don't need to do anything
        pass

    def exists(self, path: str | Path) -> bool:
        """Check if a file or folder exists in S3 storage."""
        path = self.get_path(path)
        try:
            # Check for exact object match (file)
            self._s3.meta.client.head_object(Bucket=self._bucket_name, Key=str(path))
            return True
        except botocore.exceptions.ClientError as e:
            if int(e.response["ResponseMetadata"]["HTTPStatusCode"]) == 404:
                # If the object does not exist,
                # check for objects with the prefix (folder)
                prefix = f"{str(path).rstrip('/')}/"
                objs = list(self._bucket.objects.filter(Prefix=prefix, MaxKeys=1))
                return len(objs) > 0  # True if any objects exist with the prefix
            else:
                raise  # Re-raise for any other client errors

    def make_dir(self, path: str | Path) -> None:
        """Create a directory in the storage if doesn't exist."""
        # S3 does not have directories, so we don't need to do anything
        pass

    def rename(self, old_path: str | Path, new_path: str | Path) -> None:
        """Rename a file or folder in the storage."""
        old_path = str(self.get_path(old_path))
        new_path = str(self.get_path(new_path))

        try:
            # If file exists, rename it
            self._s3.meta.client.head_object(Bucket=self._bucket_name, Key=old_path)
            self._s3.meta.client.copy_object(
                CopySource={"Bucket": self._bucket_name, "Key": old_path},
                Bucket=self._bucket_name,
                Key=new_path,
            )
            self._s3.meta.client.delete_object(Bucket=self._bucket_name, Key=old_path)
        except botocore.exceptions.ClientError as e:
            if int(e.response["ResponseMetadata"]["HTTPStatusCode"]) == 404:
                # If the object does not exist,
                # it may be a folder
                prefix = f"{old_path.rstrip('/')}/"
                objs = list(self._bucket.objects.filter(Prefix=prefix))
                for obj in objs:
                    new_key = new_path + obj.key[len(old_path) :]
                    self._s3.meta.client.copy_object(
                        CopySource={"Bucket": self._bucket_name, "Key": obj.key},
                        Bucket=self._bucket_name,
                        Key=new_key,
                    )
                    self._s3.meta.client.delete_object(
                        Bucket=self._bucket_name, Key=obj.key
                    )
            else:
                raise  # Re-raise for any other client errors

    def copy(self, source: str | Path, destination: str | Path) -> None:
        """Copy a file or folder with all contents in the storage."""
        source = str(self.get_path(source))
        destination = str(self.get_path(destination))

        try:
            # If source is a file, copy it
            self._s3.meta.client.head_object(Bucket=self._bucket_name, Key=source)
            self._s3.meta.client.copy_object(
                CopySource={"Bucket": self._bucket_name, "Key": source},
                Bucket=self._bucket_name,
                Key=destination,
            )
        except botocore.exceptions.ClientError as e:
            if int(e.response["ResponseMetadata"]["HTTPStatusCode"]) == 404:
                # If the object does not exist,
                # it may be a folder
                prefix = f"{source.rstrip('/')}/"
                objs = list(self._bucket.objects.filter(Prefix=prefix))
                for obj in objs:
                    new_key = destination + obj.key[len(source) :]
                    self._s3.meta.client.copy_object(
                        CopySource={"Bucket": self._bucket_name, "Key": obj.key},
                        Bucket=self._bucket_name,
                        Key=new_key,
                    )
            else:
                raise

    def clone_with_subroot(self, subroot: str | Path) -> S3FileStorage:
        """Create a new S3FileStorage with a subroot of the current storage."""
        file_storage = S3FileStorage(
            S3FileStorageConfiguration(
                bucket=self._bucket_name,
                root=Path("/").joinpath(self.get_path(subroot)),
                s3_endpoint_url=self._s3.meta.client.meta.endpoint_url,
            )
        )
        file_storage._s3 = self._s3
        file_storage._bucket = self._bucket
        return file_storage

    def __repr__(self) -> str:
        return f"{__class__.__name__}(bucket='{self._bucket_name}', root={self._root})"
