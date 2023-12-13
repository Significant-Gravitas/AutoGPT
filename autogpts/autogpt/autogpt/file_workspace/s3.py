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

from .base import FileWorkspace, FileWorkspaceConfiguration

if TYPE_CHECKING:
    import mypy_boto3_s3

logger = logging.getLogger(__name__)


class S3FileWorkspaceConfiguration(FileWorkspaceConfiguration):
    bucket: str = UserConfigurable("autogpt", from_env="WORKSPACE_STORAGE_BUCKET")
    s3_endpoint_url: Optional[SecretStr] = UserConfigurable(
        from_env=lambda: SecretStr(v) if (v := os.getenv("S3_ENDPOINT_URL")) else None
    )


class S3FileWorkspace(FileWorkspace):
    """A class that represents an S3 workspace."""

    _bucket: mypy_boto3_s3.service_resource.Bucket

    def __init__(self, config: S3FileWorkspaceConfiguration):
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
        """The root directory of the file workspace."""
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
        """Open a file in the workspace."""
        obj = self._get_obj(path)
        return obj.get()["Body"] if binary else TextIOWrapper(obj.get()["Body"])

    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the workspace."""
        return self.open_file(path, binary).read()

    async def write_file(self, path: str | Path, content: str | bytes) -> None:
        """Write to a file in the workspace."""
        obj = self._get_obj(path)
        obj.put(Body=content)

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
        if path == Path("."):  # root level of bucket
            return [Path(obj.key) for obj in self._bucket.objects.all()]
        else:
            return [
                Path(obj.key).relative_to(path)
                for obj in self._bucket.objects.filter(Prefix=f"{path}/")
            ]

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the workspace."""
        path = self.get_path(path)
        obj = self._s3.Object(self._bucket_name, str(path))
        obj.delete()

    def __repr__(self) -> str:
        return f"{__class__.__name__}(bucket='{self._bucket_name}', root={self._root})"
