import enum
from pathlib import Path
from typing import Optional

from .base import AbstractFileWorkspace, AbstractFileWorkspaceConfiguration


class AbstractFileWorkspaceBackendName(str, enum.Enum):
    LOCAL = "local"
    GCS = "gcs"
    S3 = "s3"


def get_workspace(
    backend: AbstractFileWorkspaceBackendName, *, id: str = "", root_path: Optional[Path] = None
) -> AbstractFileWorkspace:
    assert bool(root_path) != bool(id), "Specify root_path or id to get workspace"
    if root_path is None:
        root_path = Path(f"workspaces/{id}")

    match backend:
        case AbstractFileWorkspaceBackendName.LOCAL:
            from .local import AGPTLocalFileWorkspace
            workspace_class = AGPTLocalFileWorkspace
            # config = AbstractFileWorkspace.SystemSettings.configuration.from_env()
            # config.root = root_path
            # return AGPTLocalFileWorkspace(config)
        case AbstractFileWorkspaceBackendName.S3:
            from .s3 import S3FileWorkspace
            workspace_class = S3FileWorkspace
            # config = S3FileWorkspace.SystemSettings.configuration.from_env()
            # config.root = root_path
            # return S3FileWorkspace(config)
        case AbstractFileWorkspaceBackendName.GCS:
            from .gcs import GCSFileWorkspace
            workspace_class = GCSFileWorkspace
            # config = GCSFileWorkspace.SystemSettings.configuration.from_env()
            # config.root = root_path
            # return GCSFileWorkspace(config)
        case _:
            raise ValueError(f"Unknown workspace backend {backend}")
        
    config : AbstractFileWorkspaceConfiguration =  workspace_class.SystemSettings.configuration.from_env()
    config.root = root_path
    return workspace_class(config)

__all__ = [
    "FileWorkspace",
    "FileWorkspaceBackendName",
    "get_workspace",
]
