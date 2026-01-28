"""
Pydantic models for the Workspace API.
"""

from datetime import datetime
from typing import Any, Optional

from prisma.enums import WorkspaceFileSource
from pydantic import BaseModel, Field


class WorkspaceInfo(BaseModel):
    """Response model for workspace information."""

    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    file_count: int = 0


class WorkspaceFileInfo(BaseModel):
    """Response model for workspace file information."""

    id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int
    checksum: Optional[str] = None
    source: WorkspaceFileSource
    source_exec_id: Optional[str] = None
    source_session_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceFileListResponse(BaseModel):
    """Response model for listing workspace files."""

    files: list[WorkspaceFileInfo]
    total_count: int
    path_filter: Optional[str] = None


class UploadFileRequest(BaseModel):
    """Request model for file upload metadata."""

    filename: str
    path: Optional[str] = None
    mime_type: Optional[str] = None
    overwrite: bool = False


class WriteFileRequest(BaseModel):
    """Request model for writing file content directly (for CoPilot tools)."""

    filename: str
    content_base64: str = Field(description="Base64-encoded file content")
    path: Optional[str] = None
    mime_type: Optional[str] = None
    overwrite: bool = False


class UploadFileResponse(BaseModel):
    """Response model for file upload."""

    file: WorkspaceFileInfo
    message: str


class DeleteFileResponse(BaseModel):
    """Response model for file deletion."""

    success: bool
    file_id: str
    message: str


class DownloadUrlResponse(BaseModel):
    """Response model for download URL."""

    url: str
    expires_in_seconds: int
