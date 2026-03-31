"""CoPilot tools for workspace file operations."""

import base64
import logging
import mimetypes
import os
from typing import Any, Optional

from pydantic import BaseModel

from backend.copilot.context import (
    E2B_WORKDIR,
    get_current_sandbox,
    get_sdk_cwd,
    get_workspace_manager,
    is_allowed_local_path,
    resolve_sandbox_path,
)
from backend.copilot.model import ChatSession
from backend.copilot.tools.sandbox import make_session_path
from backend.util.settings import Config
from backend.util.virus_scanner import scan_content_safe
from backend.util.workspace import WorkspaceManager

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)

_MAX_FILE_SIZE_MB = Config().max_file_size_mb

# Sentinel file_id used when a tool-result file is read directly from the local
# host filesystem (rather than from workspace storage).
_LOCAL_TOOL_RESULT_FILE_ID = "local"


async def _resolve_write_content(
    content_text: str | None,
    content_b64: str | None,
    source_path: str | None,
    session_id: str,
) -> bytes | ErrorResponse:
    """Resolve file content from exactly one of three input sources.

    Returns the raw bytes on success, or an ``ErrorResponse`` on validation
    failure (wrong number of sources, invalid path, file not found, etc.).

    When an E2B sandbox is active, ``source_path`` reads from the sandbox
    filesystem instead of the local ephemeral directory.
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
        return await _read_source_path(source_path, session_id)

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


def _resolve_sandbox_path(
    path: str, session_id: str | None, param_name: str
) -> str | ErrorResponse:
    """Normalize *path* to an absolute sandbox path under :data:`E2B_WORKDIR`.

    Delegates to :func:`~backend.copilot.sdk.e2b_file_tools.resolve_sandbox_path`
    and wraps any ``ValueError`` into an :class:`ErrorResponse`.
    """
    try:
        return resolve_sandbox_path(path)
    except ValueError:
        return ErrorResponse(
            message=f"{param_name} must be within {E2B_WORKDIR}",
            session_id=session_id,
        )


async def _read_source_path(source_path: str, session_id: str) -> bytes | ErrorResponse:
    """Read *source_path* from E2B sandbox or local ephemeral directory."""

    sandbox = get_current_sandbox()
    if sandbox is not None:
        remote = _resolve_sandbox_path(source_path, session_id, "source_path")
        if isinstance(remote, ErrorResponse):
            return remote
        try:
            data = await sandbox.files.read(remote, format="bytes")
            return bytes(data)
        except Exception as exc:
            return ErrorResponse(
                message=f"Source file not found on sandbox: {source_path} ({exc})",
                session_id=session_id,
            )

    # Local fallback: validate path stays within ephemeral directory.
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


async def _save_to_path(
    path: str, content: bytes, session_id: str
) -> str | ErrorResponse:
    """Write *content* to *path* on E2B sandbox or local ephemeral directory.

    Returns the resolved path on success, or an ``ErrorResponse`` on failure.
    """

    sandbox = get_current_sandbox()
    if sandbox is not None:
        remote = _resolve_sandbox_path(path, session_id, "save_to_path")
        if isinstance(remote, ErrorResponse):
            return remote
        try:
            await sandbox.files.write(remote, content)
        except Exception as exc:
            return ErrorResponse(
                message=f"Failed to write to sandbox: {path} ({exc})",
                session_id=session_id,
            )
        return remote

    validated = _validate_ephemeral_path(
        path, param_name="save_to_path", session_id=session_id
    )
    if isinstance(validated, ErrorResponse):
        return validated
    try:
        dir_path = os.path.dirname(validated)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(validated, "wb") as f:
            f.write(content)
    except Exception as exc:
        return ErrorResponse(
            message=f"Failed to write to local path: {path} ({exc})",
            session_id=session_id,
        )
    return validated


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


_MAX_LOCAL_TOOL_RESULT_BYTES = 10 * 1024 * 1024  # 10 MB


def _read_local_tool_result(
    path: str,
    char_offset: int,
    char_length: Optional[int],
    session_id: str,
    sdk_cwd: str | None = None,
) -> ToolResponseBase:
    """Read an SDK tool-result file from local disk.

    This is a fallback for when the model mistakenly calls
    ``read_workspace_file`` with an SDK tool-result path that only exists on
    the host filesystem, not in cloud workspace storage.

    Defence-in-depth: validates *path* via :func:`is_allowed_local_path`
    regardless of what the caller has already checked.
    """
    # TOCTOU: path validated then opened separately. Acceptable because
    # the tool-results directory is server-controlled, not user-writable.
    expanded = os.path.realpath(os.path.expanduser(path))
    # Defence-in-depth: re-check with resolved path (caller checked raw path).
    if not is_allowed_local_path(expanded, sdk_cwd or get_sdk_cwd()):
        return ErrorResponse(
            message=f"Path not allowed: {os.path.basename(path)}", session_id=session_id
        )
    try:
        # The 10 MB cap (_MAX_LOCAL_TOOL_RESULT_BYTES) bounds memory usage.
        # Pre-read size check prevents loading files far above the cap;
        # the remaining TOCTOU gap is acceptable for server-controlled paths.
        file_size = os.path.getsize(expanded)
        if file_size > _MAX_LOCAL_TOOL_RESULT_BYTES:
            return ErrorResponse(
                message=(f"File too large: {os.path.basename(path)}"),
                session_id=session_id,
            )

        # Detect binary files: try strict UTF-8 first, fall back to
        # base64-encoding the raw bytes for binary content.
        with open(expanded, "rb") as fh:
            raw = fh.read()
        try:
            text_content = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Binary file — return raw base64, ignore char_offset/char_length
            return WorkspaceFileContentResponse(
                file_id=_LOCAL_TOOL_RESULT_FILE_ID,
                name=os.path.basename(path),
                path=path,
                mime_type=mimetypes.guess_type(path)[0] or "application/octet-stream",
                content_base64=base64.b64encode(raw).decode("ascii"),
                message=(
                    f"Read {file_size:,} bytes (binary) from local tool-result "
                    f"{os.path.basename(path)}"
                ),
                session_id=session_id,
            )

        end = (
            char_offset + char_length if char_length is not None else len(text_content)
        )
        slice_text = text_content[char_offset:end]
    except FileNotFoundError:
        return ErrorResponse(
            message=f"File not found: {os.path.basename(path)}", session_id=session_id
        )
    except Exception as exc:
        return ErrorResponse(
            message=f"Error reading file: {type(exc).__name__}", session_id=session_id
        )

    return WorkspaceFileContentResponse(
        file_id=_LOCAL_TOOL_RESULT_FILE_ID,
        name=os.path.basename(path),
        path=path,
        mime_type=mimetypes.guess_type(path)[0] or "text/plain",
        content_base64=base64.b64encode(slice_text.encode("utf-8")).decode("ascii"),
        message=(
            f"Read chars {char_offset}\u2013{char_offset + len(slice_text)} "
            f"of {len(text_content):,} chars from local tool-result "
            f"{os.path.basename(path)}"
        ),
        session_id=session_id,
    )


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
        return "List persistent workspace files. For ephemeral session files, use SDK Glob/Read instead. Optionally filter by path prefix."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path_prefix": {
                    "type": "string",
                    "description": "Filter by path prefix (e.g. '/documents/').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max files to return (default 50, max 100).",
                    "minimum": 1,
                    "maximum": 100,
                },
                "include_all_sessions": {
                    "type": "boolean",
                    "description": "Include files from all sessions (default: false).",
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
            manager = await get_workspace_manager(user_id, session_id)
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

    MAX_INLINE_SIZE_BYTES = 32 * 1024  # 32KB for text/image files
    PREVIEW_SIZE = 500

    @property
    def name(self) -> str:
        return "read_workspace_file"

    @property
    def description(self) -> str:
        return (
            "Read a file from persistent workspace. Specify file_id or path. "
            "Small text/image files return inline; large/binary return metadata+URL. "
            "Use save_to_path to copy to working dir for processing. "
            "Use offset/length for paginated reads. "
            "Paths scoped to current session; use /sessions/<id>/... for cross-session access."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "File ID from list_workspace_files.",
                },
                "path": {
                    "type": "string",
                    "description": "Virtual file path (e.g. '/documents/report.pdf').",
                },
                "save_to_path": {
                    "type": "string",
                    "description": "Copy file to this working directory path for processing.",
                },
                "force_download_url": {
                    "type": "boolean",
                    "description": "Always return metadata+URL instead of inline content.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset for paginated reads (0-based).",
                },
                "length": {
                    "type": "integer",
                    "description": "Max characters to return for paginated reads.",
                },
            },
            "required": [],  # At least one of file_id or path must be provided
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
        char_offset: int = max(0, kwargs.get("offset", 0))
        char_length: Optional[int] = kwargs.get("length")

        if not file_id and not path:
            return ErrorResponse(
                message="Please provide either file_id or path", session_id=session_id
            )

        try:
            manager = await get_workspace_manager(user_id, session_id)
            resolved = await _resolve_file(manager, file_id, path, session_id)
            if isinstance(resolved, ErrorResponse):
                # Fallback: if the path is an SDK tool-result on local disk,
                # read it directly instead of failing.  The model sometimes
                # calls read_workspace_file for these paths by mistake.
                sdk_cwd = get_sdk_cwd()
                if path and is_allowed_local_path(path, sdk_cwd):
                    return _read_local_tool_result(
                        path, char_offset, char_length, session_id, sdk_cwd=sdk_cwd
                    )
                return resolved
            target_file_id, file_info = resolved

            # If save_to_path, read + save; cache bytes for possible inline reuse.
            cached_content: bytes | None = None
            if save_to_path:
                cached_content = await manager.read_file_by_id(target_file_id)
                result = await _save_to_path(save_to_path, cached_content, session_id)
                if isinstance(result, ErrorResponse):
                    return result
                save_to_path = result

            # Ranged read: return a character slice directly.
            if char_offset > 0 or char_length is not None:
                raw = cached_content or await manager.read_file_by_id(target_file_id)
                text = raw.decode("utf-8", errors="replace")
                total_chars = len(text)
                end = (
                    char_offset + char_length
                    if char_length is not None
                    else total_chars
                )
                slice_text = text[char_offset:end]
                return WorkspaceFileContentResponse(
                    file_id=file_info.id,
                    name=file_info.name,
                    path=file_info.path,
                    mime_type="text/plain",
                    content_base64=base64.b64encode(slice_text.encode("utf-8")).decode(
                        "utf-8"
                    ),
                    message=(
                        f"Read chars {char_offset}–"
                        f"{char_offset + len(slice_text)} "
                        f"of {total_chars:,} total "
                        f"from {file_info.name}"
                    ),
                    session_id=session_id,
                )

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
            "Write a file to persistent workspace (survives across sessions). "
            "Provide exactly one of: content (text), content_base64 (binary), "
            f"or source_path (copy from working dir). Max {_MAX_FILE_SIZE_MB}MB. "
            "Paths scoped to current session; use /sessions/<id>/... for cross-session access."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename (e.g. 'report.pdf').",
                },
                "content": {
                    "type": "string",
                    "description": "Plain text content. Mutually exclusive with content_base64/source_path.",
                },
                "content_base64": {
                    "type": "string",
                    "description": "Base64-encoded binary content. Mutually exclusive with content/source_path.",
                },
                "source_path": {
                    "type": "string",
                    "description": "Working directory path to copy to workspace. Mutually exclusive with content/content_base64.",
                },
                "path": {
                    "type": "string",
                    "description": "Virtual path (e.g. '/documents/report.pdf'). Defaults to '/{filename}'.",
                },
                "mime_type": {
                    "type": "string",
                    "description": "MIME type. Auto-detected from filename if omitted.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Overwrite if file exists (default: false).",
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

        resolved = await _resolve_write_content(
            content_text,
            content_b64,
            source_path_arg,
            session_id,
        )
        if isinstance(resolved, ErrorResponse):
            return resolved
        content: bytes = resolved

        max_size = _MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content) > max_size:
            return ErrorResponse(
                message=f"File too large. Maximum size is {_MAX_FILE_SIZE_MB}MB",
                session_id=session_id,
            )

        try:
            await scan_content_safe(content, filename=filename)
            manager = await get_workspace_manager(user_id, session_id)
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
        return "Delete a file from persistent workspace. Specify file_id or path. Paths scoped to current session; use /sessions/<id>/... for cross-session access."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "File ID from list_workspace_files.",
                },
                "path": {
                    "type": "string",
                    "description": "Virtual file path.",
                },
            },
            "required": [],  # At least one of file_id or path must be provided
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
            manager = await get_workspace_manager(user_id, session_id)
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
