import enum
from pathlib import Path

from .base import FileWorkspace


class FileWorkspaceBackendName(str, enum.Enum):
    LOCAL = "local"
    GCS = "gcs"
    S3 = "s3"


def get_workspace(backend: FileWorkspaceBackendName, root_path: Path) -> FileWorkspace:
    match backend:
        case FileWorkspaceBackendName.LOCAL:
            from .local import FileWorkspaceConfiguration, LocalFileWorkspace

            config = FileWorkspaceConfiguration.from_env()
            config.root = root_path
            return LocalFileWorkspace(config)
        case FileWorkspaceBackendName.S3:
            from .s3 import S3FileWorkspace, S3FileWorkspaceConfiguration

            config = S3FileWorkspaceConfiguration.from_env()
            config.root = root_path
            return S3FileWorkspace(config)
        case FileWorkspaceBackendName.GCS:
            raise NotImplementedError("Google Cloud Storage is not implemented yet")


__all__ = [
    "FileWorkspace",
    "FileWorkspaceBackendName",
    "get_workspace",
]
