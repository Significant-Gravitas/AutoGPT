import enum
from pathlib import Path
from typing import Optional

from .base import FileStorage


class FileStorageBackendName(str, enum.Enum):
    LOCAL = "local"
    GCS = "gcs"
    S3 = "s3"


def get_storage(
    backend: FileStorageBackendName, *, id: str = "", root_path: Optional[Path] = None
) -> FileStorage:
    assert bool(root_path) != bool(id), "Specify root_path or id to get storage"
    if root_path is None:
        root_path = Path(f"/workspaces/{id}")

    match backend:
        case FileStorageBackendName.LOCAL:
            from .local import FileStorageConfiguration, LocalFileStorage

            config = FileStorageConfiguration.from_env()
            config.root = root_path
            return LocalFileStorage(config)
        case FileStorageBackendName.S3:
            from .s3 import S3FileStorage, S3FileStorageConfiguration

            config = S3FileStorageConfiguration.from_env()
            config.root = root_path
            return S3FileStorage(config)
        case FileStorageBackendName.GCS:
            from .gcs import GCSFileStorage, GCSFileStorageConfiguration

            config = GCSFileStorageConfiguration.from_env()
            config.root = root_path
            return GCSFileStorage(config)


__all__ = [
    "FileStorage",
    "FileStorageBackendName",
    "get_storage",
]
