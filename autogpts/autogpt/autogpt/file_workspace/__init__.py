import enum
from pathlib import Path
from typing import Optional

from .base import FileWorkspace


class FileWorkspaceBackendName(str, enum.Enum):
    LOCAL = "local"
    GCS = "gcs"
    S3 = "s3"


def get_workspace(
    backend: FileWorkspaceBackendName, *, id: str = "", root_path: Optional[Path] = None
) -> FileWorkspace:
    assert bool(root_path) != bool(id), "Specify root_path or id to get workspace"
    if root_path is None:
        root_path = Path(f"/workspaces/{id}")

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
            from .gcs import GCSFileWorkspace, GCSFileWorkspaceConfiguration

            config = GCSFileWorkspaceConfiguration.from_env()
            config.root = root_path
            return GCSFileWorkspace(config)


__all__ = [
    "FileWorkspace",
    "FileWorkspaceBackendName",
    "get_workspace",
]
