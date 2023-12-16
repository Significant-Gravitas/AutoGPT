import enum
from pathlib import Path
from typing import Optional

from AFAAS.interfaces.workspace import AbstractFileWorkspace, AbstractFileWorkspaceConfiguration
from .simple import LocalFileWorkspace


class FileWorkspaceBackendName(str, enum.Enum):
    LOCAL = "local"
    GCS = "gcs"
    S3 = "s3"


def get_workspace(
    backend: FileWorkspaceBackendName
    , *
    , user_id: str
    , agent_id : str
    #, root_path: Optional[Path] = None
    )  -> AbstractFileWorkspace :
    root_path = Path(f"/workspaces/{user_id}/{agent_id}")

    match backend:
        case FileWorkspaceBackendName.LOCAL:
            from .local import AGPTLocalFileWorkspace
            workspace_class = AGPTLocalFileWorkspace
        case FileWorkspaceBackendName.S3:
            from .s3 import S3FileWorkspace_AlphaRelease
            workspace_class = S3FileWorkspace_AlphaRelease
        case FileWorkspaceBackendName.GCS:
            from .gcs import GCSFileWorkspace_AlphaRealease
            workspace_class = GCSFileWorkspace_AlphaRealease
        case _:
            raise ValueError(f"Unknown workspace backend {backend}")
        
    config : AbstractFileWorkspaceConfiguration =  workspace_class.SystemSettings.configuration.from_env()
    config.agent_workspace = root_path
    return workspace_class(config)

__all__ = [
    "FileWorkspace",
    "FileWorkspaceBackendName",
    "get_workspace",
]
