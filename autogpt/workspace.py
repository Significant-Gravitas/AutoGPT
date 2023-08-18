import abc
import os
import typing
from pathlib import Path

from google.cloud import storage


class Workspace(abc.ABC):
    @abc.abstractclassmethod
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    @abc.abstractclassmethod
    def read(self, path: str) -> bytes:
        pass

    @abc.abstractclassmethod
    def write(self, path: str, data: bytes) -> None:
        pass

    @abc.abstractclassmethod
    def delete(
        self, path: str, directory: bool = False, recursive: bool = False
    ) -> None:
        pass

    @abc.abstractclassmethod
    def exists(self, path: str) -> bool:
        pass

    @abc.abstractclassmethod
    def list(self, path: str) -> typing.List[str]:
        pass


class LocalWorkspace(Workspace):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()

    def _resolve_path(self, path: str) -> Path:
        abs_path = (self.base_path / path).resolve()
        if not str(abs_path).startswith(str(self.base_path)):
            raise ValueError("Directory traversal is not allowed!")
        return abs_path

    def read(self, path: str) -> bytes:
        path = self.base_path / path
        with open(self._resolve_path(path), "rb") as f:
            return f.read()

    def write(self, path: str, data: bytes) -> None:
        path = self.base_path / path
        with open(self._resolve_path(path), "wb") as f:
            f.write(data)

    def delete(
        self, path: str, directory: bool = False, recursive: bool = False
    ) -> None:
        path = self.base_path / path
        resolved_path = self._resolve_path(path)
        if directory:
            if recursive:
                os.rmdir(resolved_path)
            else:
                os.removedirs(resolved_path)
        else:
            os.remove(resolved_path)

    def exists(self, path: str) -> bool:
        path = self.base_path / path
        return self._resolve_path(path).exists()

    def list(self, path: str) -> typing.List[str]:
        path = self.base_path / path
        base = self._resolve_path(path)
        return [str(p.relative_to(self.base_path)) for p in base.iterdir()]


class GCSWorkspace(Workspace):
    def __init__(self, base_path: str, bucket_name: str):
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.base_path = base_path.strip("/")  # Ensure no trailing or leading slash

    def _resolve_path(self, path: str) -> str:
        resolved = os.path.join(self.base_path, path).strip("/")
        if not resolved.startswith(self.base_path):
            raise ValueError("Directory traversal is not allowed!")
        return resolved

    def read(self, path: str) -> bytes:
        path = self.base_path / path
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.get_blob(self._resolve_path(path))
        return blob.download_as_bytes()

    def write(self, path: str, data: bytes) -> None:
        path = self.base_path / path
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(self._resolve_path(path))
        blob.upload_from_string(data)

    def delete(
        self, path: str, directory: bool = False, recursive: bool = False
    ) -> None:
        path = self.base_path / path
        bucket = self.client.get_bucket(self.bucket_name)
        if directory and recursive:
            # Note: GCS doesn't really have directories, so this will just delete all blobs with the given prefix
            blobs = bucket.list_blobs(prefix=self._resolve_path(path))
            bucket.delete_blobs(blobs)
        else:
            blob = bucket.blob(self._resolve_path(path))
            blob.delete()

    def exists(self, path: str) -> bool:
        path = self.base_path / path
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(self._resolve_path(path))
        return blob.exists()

    def list(self, path: str) -> typing.List[str]:
        path = self.base_path / path
        bucket = self.client.get_bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self._resolve_path(path))
        return [blob.name for blob in blobs]
