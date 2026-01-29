"""CoPilot tools for workspace file operations."""

import base64
import logging
from typing import Any, Optional

from pydantic import BaseModel

from backend.api.features.chat.model import ChatSession
from backend.data.workspace import get_or_create_workspace
from backend.util.settings import Config
from backend.util.virus_scanner import scan_content_safe
from backend.util.workspace import WorkspaceManager

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)


class WorkspaceFileInfoData(BaseModel):
    """Data model for workspace file information (not a response itself)."""

    file_id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int


class WorkspaceFileListResponse(ToolResponseBase):
    """Response containing list of workspace files."""

    type: ResponseType = ResponseType.WORKSPACE_FILE_LIST
    files: list[WorkspaceFileInfoData]
    total_count: int


class WorkspaceFileContentResponse(ToolResponseBase):
    """Response containing workspace file content (legacy, for small text files)."""

    type: ResponseType = ResponseType.WORKSPACE_FILE_CONTENT
    file_id: str
    name: str
    path: str
    mime_type: str
    content_base64: str


class WorkspaceFileMetadataResponse(ToolResponseBase):
    """Response containing workspace file metadata and download URL (prevents context bloat)."""

    type: ResponseType = ResponseType.WORKSPACE_FILE_METADATA
    file_id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int
    download_url: str
    preview: str | None = None  # First 500 chars for text files


class WorkspaceWriteResponse(ToolResponseBase):
    """Response after writing a file to workspace."""

    type: ResponseType = ResponseType.WORKSPACE_FILE_WRITTEN
    file_id: str
    name: str
    path: str
    size_bytes: int


class WorkspaceDeleteResponse(ToolResponseBase):
    """Response after deleting a file from workspace."""

    type: ResponseType = ResponseType.WORKSPACE_FILE_DELETED
    file_id: str
    success: bool


class ListWorkspaceFilesTool(BaseTool):
    """Tool for listing files in user's workspace."""

    @property
    def name(self) -> str:
        return "list_workspace_files"

    @property
    def description(self) -> str:
        return (
            "List files in the user's workspace. "
            "Returns file names, paths, sizes, and metadata. "
            "Optionally filter by path prefix."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path_prefix": {
                    "type": "string",
                    "description": (
                        "Optional path prefix to filter files "
                        "(e.g., '/documents/' to list only files in documents folder). "
                        "By default, only files from the current session are listed."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of files to return (default 50, max 100)",
                    "minimum": 1,
                    "maximum": 100,
                },
                "include_all_sessions": {
                    "type": "boolean",
                    "description": (
                        "If true, list files from all sessions. "
                        "Default is false (only current session's files)."
                    ),
                },
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id

        if not user_id:
            return ErrorResponse(
                message="Authentication required",
                session_id=session_id,
            )

        path_prefix: Optional[str] = kwargs.get("path_prefix")
        limit = min(kwargs.get("limit", 50), 100)
        include_all_sessions: bool = kwargs.get("include_all_sessions", False)

        try:
            workspace = await get_or_create_workspace(user_id)
            # Pass session_id for session-scoped file access
            manager = WorkspaceManager(user_id, workspace.id, session_id)

            files = await manager.list_files(
                path=path_prefix,
                limit=limit,
                include_all_sessions=include_all_sessions,
            )
            total = await manager.get_file_count(
                path=path_prefix,
                include_all_sessions=include_all_sessions,
            )

            file_infos = [
                WorkspaceFileInfoData(
                    file_id=f.id,
                    name=f.name,
                    path=f.path,
                    mime_type=f.mimeType,
                    size_bytes=f.sizeBytes,
                )
                for f in files
            ]

            scope_msg = "all sessions" if include_all_sessions else "current session"
            return WorkspaceFileListResponse(
                files=file_infos,
                total_count=total,
                message=f"Found {len(files)} files in workspace ({scope_msg})",
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Error listing workspace files: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to list workspace files: {str(e)}",
                error=str(e),
                session_id=session_id,
            )


class ReadWorkspaceFileTool(BaseTool):
    """Tool for reading file content from workspace."""

    # Size threshold for returning full content vs metadata+URL
    # Files larger than this return metadata with download URL to prevent context bloat
    MAX_INLINE_SIZE_BYTES = 32 * 1024  # 32KB
    # Preview size for text files
    PREVIEW_SIZE = 500

    @property
    def name(self) -> str:
        return "read_workspace_file"

    @property
    def description(self) -> str:
        return (
            "Read a file from the user's workspace. "
            "Specify either file_id or path to identify the file. "
            "For small text files, returns content directly. "
            "For large or binary files, returns metadata and a download URL. "
            "Paths are scoped to the current session by default. "
            "Use /sessions/<session_id>/... for cross-session access."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The file's unique ID (from list_workspace_files)",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The virtual file path (e.g., '/documents/report.pdf'). "
                        "Scoped to current session by default."
                    ),
                },
                "force_download_url": {
                    "type": "boolean",
                    "description": (
                        "If true, always return metadata+URL instead of inline content. "
                        "Default is false (auto-selects based on file size/type)."
                    ),
                },
            },
            "required": [],  # At least one must be provided
        }

    @property
    def requires_auth(self) -> bool:
        return True

    def _is_text_mime_type(self, mime_type: str) -> bool:
        """Check if the MIME type is a text-based type."""
        text_types = [
            "text/",
            "application/json",
            "application/xml",
            "application/javascript",
            "application/x-python",
            "application/x-sh",
        ]
        return any(mime_type.startswith(t) for t in text_types)

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id

        if not user_id:
            return ErrorResponse(
                message="Authentication required",
                session_id=session_id,
            )

        file_id: Optional[str] = kwargs.get("file_id")
        path: Optional[str] = kwargs.get("path")
        force_download_url: bool = kwargs.get("force_download_url", False)

        if not file_id and not path:
            return ErrorResponse(
                message="Please provide either file_id or path",
                session_id=session_id,
            )

        try:
            workspace = await get_or_create_workspace(user_id)
            # Pass session_id for session-scoped file access
            manager = WorkspaceManager(user_id, workspace.id, session_id)

            # Get file info
            if file_id:
                file_info = await manager.get_file_info(file_id)
                if file_info is None:
                    return ErrorResponse(
                        message=f"File not found: {file_id}",
                        session_id=session_id,
                    )
                target_file_id = file_id
            else:
                # path is guaranteed to be non-None here due to the check above
                assert path is not None
                file_info = await manager.get_file_info_by_path(path)
                if file_info is None:
                    return ErrorResponse(
                        message=f"File not found at path: {path}",
                        session_id=session_id,
                    )
                target_file_id = file_info.id

            # Decide whether to return inline content or metadata+URL
            is_small_file = file_info.sizeBytes <= self.MAX_INLINE_SIZE_BYTES
            is_text_file = self._is_text_mime_type(file_info.mimeType)

            # Return inline content for small text files (unless force_download_url)
            if is_small_file and is_text_file and not force_download_url:
                content = await manager.read_file_by_id(target_file_id)
                content_b64 = base64.b64encode(content).decode("utf-8")

                return WorkspaceFileContentResponse(
                    file_id=file_info.id,
                    name=file_info.name,
                    path=file_info.path,
                    mime_type=file_info.mimeType,
                    content_base64=content_b64,
                    message=f"Successfully read file: {file_info.name}",
                    session_id=session_id,
                )

            # Return metadata + workspace:// reference for large or binary files
            # This prevents context bloat (100KB file = ~133KB as base64)
            # Use workspace:// format so frontend urlTransform can add proxy prefix
            download_url = f"workspace://{target_file_id}"

            # Generate preview for text files
            preview: str | None = None
            if is_text_file:
                try:
                    content = await manager.read_file_by_id(target_file_id)
                    preview_text = content[: self.PREVIEW_SIZE].decode(
                        "utf-8", errors="replace"
                    )
                    if len(content) > self.PREVIEW_SIZE:
                        preview_text += "..."
                    preview = preview_text
                except Exception:
                    pass  # Preview is optional

            return WorkspaceFileMetadataResponse(
                file_id=file_info.id,
                name=file_info.name,
                path=file_info.path,
                mime_type=file_info.mimeType,
                size_bytes=file_info.sizeBytes,
                download_url=download_url,
                preview=preview,
                message=f"File: {file_info.name} ({file_info.sizeBytes} bytes). Use download_url to retrieve content.",
                session_id=session_id,
            )

        except FileNotFoundError as e:
            return ErrorResponse(
                message=str(e),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error reading workspace file: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to read workspace file: {str(e)}",
                error=str(e),
                session_id=session_id,
            )


class WriteWorkspaceFileTool(BaseTool):
    """Tool for writing files to workspace."""

    @property
    def name(self) -> str:
        return "write_workspace_file"

    @property
    def description(self) -> str:
        return (
            "Write or create a file in the user's workspace. "
            "Provide the content as a base64-encoded string. "
            f"Maximum file size is {Config().max_file_size_mb}MB. "
            "Files are saved to the current session's folder by default. "
            "Use /sessions/<session_id>/... for cross-session access."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name for the file (e.g., 'report.pdf')",
                },
                "content_base64": {
                    "type": "string",
                    "description": "Base64-encoded file content",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Optional virtual path where to save the file "
                        "(e.g., '/documents/report.pdf'). "
                        "Defaults to '/{filename}'. Scoped to current session."
                    ),
                },
                "mime_type": {
                    "type": "string",
                    "description": (
                        "Optional MIME type of the file. "
                        "Auto-detected from filename if not provided."
                    ),
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Whether to overwrite if file exists at path (default: false)",
                },
            },
            "required": ["filename", "content_base64"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id

        if not user_id:
            return ErrorResponse(
                message="Authentication required",
                session_id=session_id,
            )

        filename: str = kwargs.get("filename", "")
        content_b64: str = kwargs.get("content_base64", "")
        path: Optional[str] = kwargs.get("path")
        mime_type: Optional[str] = kwargs.get("mime_type")
        overwrite: bool = kwargs.get("overwrite", False)

        if not filename:
            return ErrorResponse(
                message="Please provide a filename",
                session_id=session_id,
            )

        if not content_b64:
            return ErrorResponse(
                message="Please provide content_base64",
                session_id=session_id,
            )

        # Decode content
        try:
            content = base64.b64decode(content_b64)
        except Exception:
            return ErrorResponse(
                message="Invalid base64-encoded content",
                session_id=session_id,
            )

        # Check size
        max_file_size = Config().max_file_size_mb * 1024 * 1024
        if len(content) > max_file_size:
            return ErrorResponse(
                message=f"File too large. Maximum size is {Config().max_file_size_mb}MB",
                session_id=session_id,
            )

        try:
            # Virus scan
            await scan_content_safe(content, filename=filename)

            workspace = await get_or_create_workspace(user_id)
            # Pass session_id for session-scoped file access
            manager = WorkspaceManager(user_id, workspace.id, session_id)

            file_record = await manager.write_file(
                content=content,
                filename=filename,
                path=path,
                mime_type=mime_type,
                overwrite=overwrite,
            )

            return WorkspaceWriteResponse(
                file_id=file_record.id,
                name=file_record.name,
                path=file_record.path,
                size_bytes=file_record.sizeBytes,
                message=f"Successfully wrote file: {file_record.name}",
                session_id=session_id,
            )

        except ValueError as e:
            return ErrorResponse(
                message=str(e),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error writing workspace file: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to write workspace file: {str(e)}",
                error=str(e),
                session_id=session_id,
            )


class DeleteWorkspaceFileTool(BaseTool):
    """Tool for deleting files from workspace."""

    @property
    def name(self) -> str:
        return "delete_workspace_file"

    @property
    def description(self) -> str:
        return (
            "Delete a file from the user's workspace. "
            "Specify either file_id or path to identify the file. "
            "Paths are scoped to the current session by default. "
            "Use /sessions/<session_id>/... for cross-session access."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The file's unique ID (from list_workspace_files)",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The virtual file path (e.g., '/documents/report.pdf'). "
                        "Scoped to current session by default."
                    ),
                },
            },
            "required": [],  # At least one must be provided
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id

        if not user_id:
            return ErrorResponse(
                message="Authentication required",
                session_id=session_id,
            )

        file_id: Optional[str] = kwargs.get("file_id")
        path: Optional[str] = kwargs.get("path")

        if not file_id and not path:
            return ErrorResponse(
                message="Please provide either file_id or path",
                session_id=session_id,
            )

        try:
            workspace = await get_or_create_workspace(user_id)
            # Pass session_id for session-scoped file access
            manager = WorkspaceManager(user_id, workspace.id, session_id)

            # Determine the file_id to delete
            target_file_id: str
            if file_id:
                target_file_id = file_id
            else:
                # path is guaranteed to be non-None here due to the check above
                assert path is not None
                file_info = await manager.get_file_info_by_path(path)
                if file_info is None:
                    return ErrorResponse(
                        message=f"File not found at path: {path}",
                        session_id=session_id,
                    )
                target_file_id = file_info.id

            success = await manager.delete_file(target_file_id)

            if not success:
                return ErrorResponse(
                    message=f"File not found: {target_file_id}",
                    session_id=session_id,
                )

            return WorkspaceDeleteResponse(
                file_id=target_file_id,
                success=True,
                message="File deleted successfully",
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Error deleting workspace file: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to delete workspace file: {str(e)}",
                error=str(e),
                session_id=session_id,
            )
