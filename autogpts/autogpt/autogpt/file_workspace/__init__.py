import enum

from .base import FileWorkspace


class FileWorkspaceBackendName(str, enum.Enum):
    S3 = "s3"
    LOCAL = "local"


__all__ = [
    "FileWorkspace",
    "FileWorkspaceBackendName",
]
