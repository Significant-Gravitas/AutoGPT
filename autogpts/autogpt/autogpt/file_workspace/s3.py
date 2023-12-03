"""
The S3Workspace class provides an interface for interacting with a file workspace, and
stores the files in an S3 bucket.
"""
from __future__ import annotations

import contextlib
import inspect
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import boto3
import botocore.exceptions

from .base import FileWorkspace

if TYPE_CHECKING:
    import mypy_boto3_s3

logger = logging.getLogger(__name__)


class S3FileWorkspace(FileWorkspace):
    """A class that represents an S3 workspace."""

    on_write_file: Callable[[Path], Any] | None = None
    """
    Event hook, executed after writing a file.

    Params:
        Path: The path of the file that was written, relative to the workspace root.
    """

    _bucket: mypy_boto3_s3.service_resource.Bucket

    def __init__(self, bucket_name: str, root: Path = Path("/")):
        self._bucket_name = bucket_name
        self._root = root
        super().__init__()

        s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", None)
        s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID", None)
        s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY", None)
        self._s3 = boto3.resource(
            "s3",
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=s3_access_key_id,
            aws_secret_access_key=s3_secret_access_key,
        )

    @property
    def root(self) -> Path:
        """The root directory of the file workspace."""
        return self._root

    @property
    def restrict_to_root(self):
        """Whether to restrict generated paths to the root."""
        return True

    def initialize(self) -> None:
        try:
            self._s3.meta.client.head_bucket(Bucket=self._bucket_name)
            self._bucket = self._s3.Bucket(self._bucket_name)
        except botocore.exceptions.ClientError as e:
            if "(404)" not in str(e):
                raise
            self._bucket = self._s3.create_bucket(Bucket=self._bucket_name)

    def get_path(self, relative_path: str | Path) -> Path:
        return super().get_path(relative_path).relative_to(Path("/"))

    def open_file(self, path: str | Path, mode: str = "r"):
        """Open a file in the workspace."""
        path = self.get_path(path)
        obj = self._bucket.Object(str(path))
        with contextlib.suppress(botocore.exceptions.ClientError):
            obj.load()
        return obj

    def read_file(self, path: str | Path, binary: bool = False) -> str | bytes:
        """Read a file in the workspace."""
        file_content = self.open_file(path, "r").get()["Body"].read()
        return file_content if binary else file_content.decode()

    async def write_file(self, path: str | Path, content: str | bytes):
        """Write to a file in the workspace."""
        obj = self.open_file(path, "w")
        obj.put(Body=content)

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
        if path == Path("."):
            return [
                Path(obj.key)
                for obj in self._bucket.objects.all()
                if not obj.key.endswith("/")
            ]
        else:
            return [
                Path(obj.key)
                for obj in self._bucket.objects.filter(Prefix=str(path))
                if not obj.key.endswith("/")
            ]

    def delete_file(self, path: str | Path) -> None:
        """Delete a file in the workspace."""
        path = self.get_path(path)
        obj = self._s3.Object(self._bucket_name, str(path))
        obj.delete()
