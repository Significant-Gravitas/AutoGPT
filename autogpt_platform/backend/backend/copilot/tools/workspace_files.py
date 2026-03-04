"""CoPilot tools for workspace file operations."""

import base64
import logging
import os
from typing import Any, Optional

from pydantic import BaseModel

from backend.copilot.model import ChatSession
from backend.copilot.tools.sandbox import make_session_path
from backend.data.db_accessors import workspace_db
from backend.util.settings import Config
from backend.util.virus_scanner import scan_content_safe
from backend.util.workspace import WorkspaceManager

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)


def _resolve_write_content(
    content_text: str | None,
    content_b64: str | None,
    source_path: str | None,
    session_id: str,
) -> bytes | ErrorResponse:
    """Resolve file content from exactly one of three input sources.

    Returns the raw bytes on success, or an ``ErrorResponse`` on validation
    failure (wrong number of sources, invalid path, file not found, etc.).
    """
    # Normalise empty strings to None so counting and dispatch stay in sync.
    if content_text is not None and content_text == "":
        content_text = None
    if content_b64 is not None and content_b64 == "":
        content_b64 = None
    if source_path is not None and source_path == "":
        source_path = None

    sources_provided = sum(
        x is not None for x in [content_text, content_b64, source_path]
    )
    if sources_provided == 0:
        return ErrorResponse(
            message="Please provide one of: content, content_base64, or source_path",
            session_id=session_id,
        )
    if sources_provided > 1:
        return ErrorResponse(
            message="Provide only one of: content, content_base64, or source_path",
            session_id=session_id,
        )

    if source_path is not None:
        validated = _validate_ephemeral_path(
            source_path, param_name="source_path", session_id=session_id
        )
        if isinstance(validated, ErrorResponse):
            return validated
        try:
            with open(validated, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return ErrorResponse(
                message=f"Source file not found: {source_path}",
                session_id=session_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to read source file: {e}",
                session_id=session_id,
            )

    if content_b64 is not None:
        try:
            return base64.b64decode(content_b64)
        except Exception:
            return ErrorResponse(
                message=(
                    "Invalid base64 encoding in content_base64. "
                    "Please encode the file content with standard base64, "
                    "or use the 'content' parameter for plain text, "
                    "or 'source_path' to copy from the working directory."
                ),
                session_id=session_id,
            )

    assert content_text is not None
    return content_text.encode("utf-8")


def _validate_ephemeral_path(
    path: str, *, param_name: str, session_id: str
) -> ErrorResponse | str:
    """Validate that *path* is inside the session's ephemeral directory.

    Uses the session-specific directory (``make_session_path(session_id)``)
    rather than the bare prefix, so ``/tmp/copilot-evil/...`` is rejected.

    Returns the resolved real path on success, or an ``ErrorResponse`` when the
    path escapes the session directory.
    """
    session_dir = os.path.realpath(make_session_path(session_id)) + os.sep
    real = os.path.realpath(path)
    if not real.startswith(session_dir):
        return ErrorResponse(
            message=(
                f"{param_name} must be within the ephemeral working "
                f"directory ({make_session_path(session_id)})"
            ),
            session_id=session_id,
        )
    return real


_TEXT_MIME_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-python",
    "application/x-sh",
)

_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def _is_text_mime(mime_type: str) -> bool:
    return any(mime_type.startswith(t) for t in _TEXT_MIME_PREFIXES)


async def _get_manager(user_id: str, session_id: str) -> WorkspaceManager:
    """Create a session-scoped WorkspaceManager."""
    workspace = await workspace_db().get_or_create_workspace(user_id)
    return WorkspaceManager(user_id, workspace.id, session_id)


async def _resolve_file(
    manager: WorkspaceManager,
    file_id: str | None,
    path: str | None,
    session_id: str,
) -> tuple[str, Any] | ErrorResponse:
    """Resolve a file by file_id or path.

    Returns ``(target_file_id, file_info)`` on success, or an
    ``ErrorResponse`` if the file was not found.
    """
    if file_id:
        file_info = await manager.get_file_info(file_id)
        if file_info is None:
            return ErrorResponse(
                message=f"File not found: {file_id}", session_id=session_id
            )
        return file_id, file_info

    assert path is not None
    file_info = await manager.get_file_info_by_path(path)
    if file_info is None:
        return ErrorResponse(
            message=f"File not found at path: {path}", session_id=session_id
        )
    return file_info.id, file_info


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
    mime_type: str
    size_bytes: int
    # workspace:// URL the agent can embed directly in chat to give the user a link.
    # Format: workspace://<file_id>#<mime_type>  (frontend resolves to download URL)
    download_url: str
    source: str | None = None  # "content", "base64", or "copied from <path>"
    content_preview: str | None = None  # First 200 chars for text files


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
            "List files in the user's persistent workspace (cloud storage). "
            "These files survive across sessions. "
            "For ephemeral session files, use the SDK Read/Glob tools instead. "
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
                message="Authentication required", session_id=session_id
            )

        path_prefix: Optional[str] = kwargs.get("path_prefix")
        limit = min(kwargs.get("limit", 50), 100)
        include_all_sessions: bool = kwargs.get("include_all_sessions", False)

        try:
            manager = await _get_manager(user_id, session_id)
            files = await manager.list_files(
                path=path_prefix, limit=limit, include_all_sessions=include_all_sessions
            )
            total = await manager.get_file_count(
                path=path_prefix, include_all_sessions=include_all_sessions
            )
            file_infos = [
                WorkspaceFileInfoData(
                    file_id=f.id,
                    name=f.name,
                    path=f.path,
                    mime_type=f.mime_type,
                    size_bytes=f.size_bytes,
                )
                for f in files
            ]
            scope = "all sessions" if include_all_sessions else "current session"
            total_size = sum(f.size_bytes for f in file_infos)

            # Build a human-readable summary so the agent can relay details.
            lines = [f"Found {len(files)} file(s) in workspace ({scope}):"]
            for f in file_infos:
                lines.append(f"  - {f.path} ({f.size_bytes:,} bytes, {f.mime_type})")
            if total > len(files):
                lines.append(f"  ... and {total - len(files)} more")
            lines.append(f"Total size: {total_size:,} bytes")

            return WorkspaceFileListResponse(
                files=file_infos,
                total_count=total,
                message="\n".join(lines),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error listing workspace files: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to list workspace files: {e}",
                error=str(e),
                session_id=session_id,
            )


class ReadWorkspaceFileTool(BaseTool):
    """Tool for reading file content from workspace."""

    MAX_INLINE_SIZE_BYTES = 32 * 1024  # 32KB
    PREVIEW_SIZE = 500

    @property
    def name(self) -> str:
        return "read_workspace_file"

    @property
    def description(self) -> str:
        return (
            "Read a file from the user's persistent workspace (cloud storage). "
            "These files survive across sessions. "
            "For ephemeral session files, use the SDK Read tool instead. "
            "Specify either file_id or path to identify the file. "
            "For small text files, returns content directly. "
            "For large or binary files, returns metadata and a download URL. "
            "Optionally use 'save_to_path' to copy the file to the ephemeral "
            "working directory for processing with bash_exec or SDK tools. "
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
                "save_to_path": {
                    "type": "string",
                    "description": (
                        "If provided, save the file to this path in the ephemeral "
                        "working directory (e.g., '/tmp/copilot-.../data.csv') "
                        "so it can be processed with bash_exec or SDK tools. "
                        "The file content is still returned in the response."
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

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )

        file_id: Optional[str] = kwargs.get("file_id")
        path: Optional[str] = kwargs.get("path")
        save_to_path: Optional[str] = kwargs.get("save_to_path")
        force_download_url: bool = kwargs.get("force_download_url", False)

        if not file_id and not path:
            return ErrorResponse(
                message="Please provide either file_id or path", session_id=session_id
            )

        # Validate and resolve save_to_path (use sanitized real path).
        if save_to_path:
            validated_save = _validate_ephemeral_path(
                save_to_path, param_name="save_to_path", session_id=session_id
            )
            if isinstance(validated_save, ErrorResponse):
                return validated_save
            save_to_path = validated_save

        try:
            manager = await _get_manager(user_id, session_id)
            resolved = await _resolve_file(manager, file_id, path, session_id)
            if isinstance(resolved, ErrorResponse):
                return resolved
            target_file_id, file_info = resolved

            # If save_to_path, read + save; cache bytes for possible inline reuse.
            cached_content: bytes | None = None
            if save_to_path:
                cached_content = await manager.read_file_by_id(target_file_id)
                dir_path = os.path.dirname(save_to_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                with open(save_to_path, "wb") as f:
                    f.write(cached_content)

            is_small = file_info.size_bytes <= self.MAX_INLINE_SIZE_BYTES
            is_text = _is_text_mime(file_info.mime_type)
            is_image = file_info.mime_type in _IMAGE_MIME_TYPES

            # Inline content for small text/image files
            if is_small and (is_text or is_image) and not force_download_url:
                content = cached_content or await manager.read_file_by_id(
                    target_file_id
                )
                msg = (
                    f"Read {file_info.name} from workspace:{file_info.path} "
                    f"({file_info.size_bytes:,} bytes, {file_info.mime_type})"
                )
                if save_to_path:
                    msg += f" — also saved to {save_to_path}"
                return WorkspaceFileContentResponse(
                    file_id=file_info.id,
                    name=file_info.name,
                    path=file_info.path,
                    mime_type=file_info.mime_type,
                    content_base64=base64.b64encode(content).decode("utf-8"),
                    message=msg,
                    session_id=session_id,
                )

            # Metadata + download URL for large/binary files
            preview: str | None = None
            if is_text:
                try:
                    raw = cached_content or await manager.read_file_by_id(
                        target_file_id
                    )
                    preview = raw[: self.PREVIEW_SIZE].decode("utf-8", errors="replace")
                    if len(raw) > self.PREVIEW_SIZE:
                        preview += "..."
                except Exception:
                    pass

            msg = (
                f"File: {file_info.name} at workspace:{file_info.path} "
                f"({file_info.size_bytes:,} bytes, {file_info.mime_type})"
            )
            if save_to_path:
                msg += f" — saved to {save_to_path}"
            else:
                msg += (
                    " — use read_workspace_file with this file_id to retrieve content"
                )
            return WorkspaceFileMetadataResponse(
                file_id=file_info.id,
                name=file_info.name,
                path=file_info.path,
                mime_type=file_info.mime_type,
                size_bytes=file_info.size_bytes,
                download_url=f"workspace://{target_file_id}",
                preview=preview,
                message=msg,
                session_id=session_id,
            )
        except FileNotFoundError as e:
            return ErrorResponse(message=str(e), session_id=session_id)
        except Exception as e:
            logger.error(f"Error reading workspace file: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to read workspace file: {e}",
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
            "Write or create a file in the user's persistent workspace (cloud storage). "
            "These files survive across sessions. "
            "For ephemeral session files, use the SDK Write tool instead. "
            "Provide content as plain text via 'content', OR base64-encoded via "
            "'content_base64', OR copy a file from the ephemeral working directory "
            "via 'source_path'. Exactly one of these three is required. "
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
                "content": {
                    "type": "string",
                    "description": (
                        "Plain text content to write. Use this for text files "
                        "(code, configs, documents, etc.). "
                        "Mutually exclusive with content_base64 and source_path."
                    ),
                },
                "content_base64": {
                    "type": "string",
                    "description": (
                        "Base64-encoded file content. Use this for binary files "
                        "(images, PDFs, etc.). "
                        "Mutually exclusive with content and source_path."
                    ),
                },
                "source_path": {
                    "type": "string",
                    "description": (
                        "Path to a file in the ephemeral working directory to "
                        "copy to workspace (e.g., '/tmp/copilot-.../output.csv'). "
                        "Use this to persist files created by bash_exec or SDK Write. "
                        "Mutually exclusive with content and content_base64."
                    ),
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
            "required": ["filename"],
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
                message="Authentication required", session_id=session_id
            )

        filename: str = kwargs.get("filename", "")
        if not filename:
            return ErrorResponse(
                message="Please provide a filename", session_id=session_id
            )

        source_path_arg: str | None = kwargs.get("source_path")
        content_text: str | None = kwargs.get("content")
        content_b64: str | None = kwargs.get("content_base64")

        resolved = _resolve_write_content(
            content_text,
            content_b64,
            source_path_arg,
            session_id,
        )
        if isinstance(resolved, ErrorResponse):
            return resolved
        content: bytes = resolved

        max_size = Config().max_file_size_mb * 1024 * 1024
        if len(content) > max_size:
            return ErrorResponse(
                message=f"File too large. Maximum size is {Config().max_file_size_mb}MB",
                session_id=session_id,
            )

        try:
            await scan_content_safe(content, filename=filename)
            manager = await _get_manager(user_id, session_id)
            rec = await manager.write_file(
                content=content,
                filename=filename,
                path=kwargs.get("path"),
                mime_type=kwargs.get("mime_type"),
                overwrite=kwargs.get("overwrite", False),
            )

            # Build informative source label and message.
            if source_path_arg:
                source = f"copied from {source_path_arg}"
                msg = (
                    f"Copied {source_path_arg} → workspace:{rec.path} "
                    f"({rec.size_bytes:,} bytes)"
                )
            elif content_b64:
                source = "base64"
                msg = (
                    f"Wrote {rec.name} to workspace ({rec.size_bytes:,} bytes, "
                    f"decoded from base64)"
                )
            else:
                source = "content"
                msg = f"Wrote {rec.name} to workspace ({rec.size_bytes:,} bytes)"

            # Include a short preview for text content.
            preview: str | None = None
            if _is_text_mime(rec.mime_type):
                try:
                    preview = content[:200].decode("utf-8", errors="replace")
                    if len(content) > 200:
                        preview += "..."
                except Exception:
                    pass

            # Strip MIME parameters (e.g. "text/html; charset=utf-8" → "text/html")
            # and normalise to lowercase so the fragment is URL-safe.
            normalized_mime = (rec.mime_type or "").split(";", 1)[0].strip().lower()
            download_url = (
                f"workspace://{rec.id}#{normalized_mime}"
                if normalized_mime
                else f"workspace://{rec.id}"
            )
            return WorkspaceWriteResponse(
                file_id=rec.id,
                name=rec.name,
                path=rec.path,
                mime_type=normalized_mime,
                size_bytes=rec.size_bytes,
                download_url=download_url,
                source=source,
                content_preview=preview,
                message=msg,
                session_id=session_id,
            )
        except ValueError as e:
            return ErrorResponse(message=str(e), session_id=session_id)
        except Exception as e:
            logger.error(f"Error writing workspace file: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to write workspace file: {e}",
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
            "Delete a file from the user's persistent workspace (cloud storage). "
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
                message="Authentication required", session_id=session_id
            )

        file_id: Optional[str] = kwargs.get("file_id")
        path: Optional[str] = kwargs.get("path")
        if not file_id and not path:
            return ErrorResponse(
                message="Please provide either file_id or path", session_id=session_id
            )

        try:
            manager = await _get_manager(user_id, session_id)
            resolved = await _resolve_file(manager, file_id, path, session_id)
            if isinstance(resolved, ErrorResponse):
                return resolved
            target_file_id, file_info = resolved

            if not await manager.delete_file(target_file_id):
                return ErrorResponse(
                    message=f"File not found: {target_file_id}", session_id=session_id
                )
            return WorkspaceDeleteResponse(
                file_id=target_file_id,
                success=True,
                message=(
                    f"Deleted {file_info.name} from workspace:{file_info.path} "
                    f"({file_info.size_bytes:,} bytes)"
                ),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error deleting workspace file: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to delete workspace file: {e}",
                error=str(e),
                session_id=session_id,
            )
