"""
Workspace API routes for managing user file storage.
"""

import asyncio
import logging
import os
import re
from typing import Annotated, Any, Literal
from urllib.parse import quote

import fastapi
from autogpt_libs.auth.dependencies import get_user_id, requires_user
from fastapi import Query, UploadFile
from fastapi.responses import Response
from prisma.errors import UniqueViolationError
from pydantic import BaseModel, Field, field_validator

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.api.features.workspace.preview import build_preview_response
from backend.copilot.db import get_chat_session_expert_ids
from backend.copilot.rate_limit import get_workspace_storage_limit_bytes
from backend.data.workspace import (
    WorkspaceFile,
    count_workspace_files,
    get_or_create_workspace,
    get_workspace,
    get_workspace_file,
    get_workspace_total_size,
    rename_workspace_file,
)
from backend.data.workspace_scope import (
    resolve_expert_workspace_scope,
    session_path_prefix,
)
from backend.util.settings import Config
from backend.util.workspace import WorkspaceManager, format_bytes
from backend.util.workspace_storage import get_workspace_storage


def _sanitize_filename_for_header(
    filename: str, disposition: str = "attachment"
) -> str:
    """
    Sanitize filename for Content-Disposition header to prevent header injection.

    Removes/replaces characters that could break the header or inject new headers.
    Uses RFC5987 encoding for non-ASCII characters.
    """
    # Remove CR, LF, and null bytes (header injection prevention)
    sanitized = re.sub(r"[\r\n\x00]", "", filename)
    # Escape quotes
    sanitized = sanitized.replace('"', '\\"')
    # For non-ASCII, use RFC5987 filename* parameter
    # Check if filename has non-ASCII characters
    try:
        sanitized.encode("ascii")
        return f'{disposition}; filename="{sanitized}"'
    except UnicodeEncodeError:
        # Use RFC5987 encoding for UTF-8 filenames
        encoded = quote(sanitized, safe="")
        return f"{disposition}; filename*=UTF-8''{encoded}"


logger = logging.getLogger(__name__)

router = fastapi.APIRouter(
    dependencies=[fastapi.Security(requires_user)],
)


def _create_streaming_response(
    content: bytes, file: WorkspaceFile, *, inline: bool = False
) -> Response:
    """Create a streaming response for file content."""
    disposition = _sanitize_filename_for_header(
        file.name, disposition="inline" if inline else "attachment"
    )
    return Response(
        content=content,
        media_type=file.mime_type,
        headers={
            "Content-Disposition": disposition,
            "Content-Security-Policy": "sandbox",
            "X-Content-Type-Options": "nosniff",
            "Content-Length": str(len(content)),
        },
    )


async def create_file_download_response(
    file: WorkspaceFile, *, inline: bool = False
) -> Response:
    """
    Create a download response for a workspace file.

    Handles both local storage (direct streaming) and GCS (signed URL redirect
    with fallback to streaming).
    """
    storage = await get_workspace_storage()

    # For local storage, stream the file directly
    if file.storage_path.startswith("local://"):
        content = await storage.retrieve(file.storage_path)
        return _create_streaming_response(content, file, inline=inline)

    # For GCS, try to redirect to signed URL, fall back to streaming
    try:
        url = await storage.get_download_url(file.storage_path, expires_in=300)
        # If we got back an API path (fallback), stream directly instead
        if url.startswith("/api/"):
            content = await storage.retrieve(file.storage_path)
            return _create_streaming_response(content, file, inline=inline)
        return fastapi.responses.RedirectResponse(url=url, status_code=302)
    except Exception as e:
        # Log the signed URL failure with context
        logger.error(
            f"Failed to get signed URL for file {file.id} "
            f"(storagePath={file.storage_path}): {e}",
            exc_info=True,
        )
        # Fall back to streaming directly from GCS
        try:
            content = await storage.retrieve(file.storage_path)
            return _create_streaming_response(content, file, inline=inline)
        except Exception as fallback_error:
            logger.error(
                f"Fallback streaming also failed for file {file.id} "
                f"(storagePath={file.storage_path}): {fallback_error}",
                exc_info=True,
            )
            raise


class UploadFileResponse(BaseModel):
    file_id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int


class DeleteFileResponse(BaseModel):
    deleted: bool


class StorageUsageResponse(BaseModel):
    used_bytes: int
    limit_bytes: int
    used_percent: float
    file_count: int


class WorkspaceFileItem(BaseModel):
    id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int
    folder_id: str | None = None
    metadata: dict = Field(default_factory=dict)
    origin: Literal["uploaded", "generated"]
    created_at: str
    # Hired expert whose conversation the file lives in; None for personal
    # AutoPilot chats, Builder output and uploads outside a chat.
    expert_id: str | None = None


class ListFilesResponse(BaseModel):
    files: list[WorkspaceFileItem]
    offset: int = 0
    has_more: bool = False


class RenameFileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)

    @field_validator("name")
    @classmethod
    def _plain_file_name(cls, value: str) -> str:
        name = value.strip()
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("File name must be a plain name without slashes")
        return name


# Exact metadata stamped on user uploads by ``upload_file``. Used to split
# "Uploaded" vs "Generated" on the Artifacts page.
_UPLOADED_METADATA = {"origin": "user-upload"}


def _derive_origin(metadata: dict | None) -> Literal["uploaded", "generated"]:
    """Classify a file as user-uploaded vs agent/block-generated."""
    if (metadata or {}).get("origin") == "user-upload":
        return "uploaded"
    return "generated"


@router.get(
    "/files/{file_id}/download",
    summary="Download file by ID",
    operation_id="getWorkspaceDownloadFileById",
)
async def download_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file_id: str,
) -> Response:
    """
    Download a file by its ID.

    Returns the file content directly or redirects to a signed URL for GCS.
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    file = await get_workspace_file(file_id, workspace.id)
    if file is None:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    return await create_file_download_response(file)


@router.get(
    "/files/{file_id}/preview",
    summary="Preview file by ID",
    operation_id="getWorkspaceFilePreview",
)
async def preview_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file_id: str,
    w: int = Query(default=400, ge=16, le=1024),
    bytes_: int = Query(default=4096, ge=256, le=131072, alias="bytes"),
) -> Response:
    """
    Return a cheap preview of a file.

    Images/PDFs/Office docs are returned as a small WebP thumbnail; text-like
    files return only their first ``bytes`` bytes. Used by the Artifacts page so
    a grid of files no longer downloads every file in full.
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    file = await get_workspace_file(file_id, workspace.id)
    if file is None:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    return await build_preview_response(file, width=w, max_bytes=bytes_)


@router.delete(
    "/files/{file_id}",
    summary="Delete a workspace file",
    operation_id="deleteWorkspaceFile",
)
async def delete_workspace_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file_id: str,
) -> DeleteFileResponse:
    """
    Soft-delete a workspace file and attempt to remove it from storage.

    Used when a user clears a file input in the builder.
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    manager = WorkspaceManager(user_id, workspace.id)
    deleted = await manager.delete_file(file_id)
    if not deleted:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    return DeleteFileResponse(deleted=True)


@router.patch(
    "/files/{file_id}",
    summary="Rename a workspace file",
    operation_id="renameWorkspaceFile",
    responses={
        404: {"description": "File not found"},
        409: {"description": "A file with this name already exists here"},
    },
)
async def rename_workspace_file_route(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file_id: str,
    payload: RenameFileRequest,
) -> WorkspaceFileItem:
    """Rename a file; it stays in its folder and conversation."""
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")
    try:
        renamed = await rename_workspace_file(file_id, workspace.id, payload.name)
    except UniqueViolationError:
        raise fastapi.HTTPException(
            status_code=409, detail="A file with this name already exists here"
        )
    if renamed is None:
        raise fastapi.HTTPException(status_code=404, detail="File not found")
    expert_by_session = await _expert_ids_by_session(user_id, [renamed])
    return _to_file_item(renamed, expert_by_session)


@router.post(
    "/files/upload",
    summary="Upload file to workspace",
    operation_id="uploadWorkspaceFile",
)
async def upload_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file: UploadFile,
    session_id: str | None = Query(default=None),
    overwrite: bool = Query(default=False),
) -> UploadFileResponse:
    """
    Upload a file to the user's workspace.

    Files are stored in session-scoped paths when session_id is provided,
    so the agent's session-scoped tools can discover them automatically.
    """
    # Empty-string session_id drops session scoping; normalize to None.
    session_id = session_id or None

    config = Config()

    # Sanitize filename — strip any directory components
    filename = os.path.basename(file.filename or "upload") or "upload"

    # Read file content with early abort on size limit
    max_file_bytes = config.max_file_size_mb * 1024 * 1024
    chunks: list[bytes] = []
    total_size = 0
    while chunk := await file.read(64 * 1024):  # 64KB chunks
        total_size += len(chunk)
        if total_size > max_file_bytes:
            raise fastapi.HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {config.max_file_size_mb} MB",
            )
        chunks.append(chunk)
    content = b"".join(chunks)

    # Get or create workspace
    workspace = await get_or_create_workspace(user_id)

    # Write file via WorkspaceManager (handles virus scan, per-file size,
    # and per-user tier-based storage quota internally).
    manager = WorkspaceManager(user_id, workspace.id, session_id)
    try:
        workspace_file = await manager.write_file(
            content, filename, overwrite=overwrite, metadata={"origin": "user-upload"}
        )
    except VirusDetectedError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e)) from e
    except VirusScanError as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e)) from e
    except ValueError as e:
        # write_file raises ValueError for path-conflict, size-limit, and
        # storage-quota cases; map each to its correct HTTP status.
        message = str(e)
        if message.startswith(("File too large", "Storage limit exceeded")):
            raise fastapi.HTTPException(status_code=413, detail=message) from e
        raise fastapi.HTTPException(status_code=409, detail=message) from e

    # Post-write storage check — eliminates TOCTOU race on the quota.
    # If a concurrent upload pushed us over the limit, undo this write.
    storage_limit_bytes = await get_workspace_storage_limit_bytes(user_id)
    new_total = await get_workspace_total_size(workspace.id)
    if storage_limit_bytes and new_total > storage_limit_bytes:
        try:
            # Route through WorkspaceManager so the storage backend blob is
            # removed too — soft_delete_workspace_file alone leaks the blob.
            await manager.delete_file(workspace_file.id)
        except Exception as e:
            logger.warning(
                f"Failed to delete over-quota file {workspace_file.id} "
                f"in workspace {workspace.id}: {e}"
            )
        raise fastapi.HTTPException(
            status_code=413,
            detail=(
                f"Storage limit exceeded. "
                f"You've used {format_bytes(new_total)} of your "
                f"{format_bytes(storage_limit_bytes)} quota. "
                f"Delete some files or upgrade your plan for more storage."
            ),
        )

    return UploadFileResponse(
        file_id=workspace_file.id,
        name=workspace_file.name,
        path=workspace_file.path,
        mime_type=workspace_file.mime_type,
        size_bytes=workspace_file.size_bytes,
    )


@router.get(
    "/storage/usage",
    summary="Get workspace storage usage",
    operation_id="getWorkspaceStorageUsage",
)
async def get_storage_usage(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
) -> StorageUsageResponse:
    """
    Get storage usage information for the user's workspace.
    """
    workspace = await get_or_create_workspace(user_id)

    used_bytes, file_count, limit_bytes = await asyncio.gather(
        get_workspace_total_size(workspace.id),
        count_workspace_files(workspace.id),
        get_workspace_storage_limit_bytes(user_id),
    )

    return StorageUsageResponse(
        used_bytes=used_bytes,
        limit_bytes=limit_bytes,
        used_percent=round((used_bytes / limit_bytes) * 100, 1) if limit_bytes else 0,
        file_count=file_count,
    )


_SESSION_PATH_RE = re.compile(r"^/sessions/([^/]+)/")


def _session_id_of(path: str) -> str | None:
    match = _SESSION_PATH_RE.match(path)
    return match.group(1) if match else None


async def _expert_ids_by_session(
    user_id: str, files: list[WorkspaceFile]
) -> dict[str, str | None]:
    """Attribute listed files to the expert whose conversation they live in."""
    session_ids = sorted({sid for f in files if (sid := _session_id_of(f.path))})
    if not session_ids:
        return {}
    return await get_chat_session_expert_ids(user_id, session_ids)


def _to_file_item(
    f: WorkspaceFile, expert_by_session: dict[str, str | None]
) -> WorkspaceFileItem:
    session_id_of_file = _session_id_of(f.path)
    return WorkspaceFileItem(
        id=f.id,
        name=f.name,
        path=f.path,
        mime_type=f.mime_type,
        size_bytes=f.size_bytes,
        folder_id=f.folder_id,
        metadata=f.metadata or {},
        origin=_derive_origin(f.metadata),
        created_at=f.created_at.isoformat(),
        expert_id=(
            expert_by_session.get(session_id_of_file) if session_id_of_file else None
        ),
    )


@router.get(
    "/files",
    summary="List workspace files",
    operation_id="listWorkspaceFiles",
    responses={400: {"description": "Conflicting filters"}},
)
async def list_workspace_files(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    session_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str | None = Query(
        default=None,
        description=(
            "Case-insensitive substring search on file name. Applied "
            "in the database for fresh results without waiting on "
            "embedding generation."
        ),
    ),
    origin: Literal["uploaded", "generated"] | None = Query(
        default=None,
        description=(
            "Filter by file origin. ``uploaded`` matches files the user "
            "uploaded (``metadata.origin == 'user-upload'``, set by the "
            "upload endpoint for both Builder and CoPilot uploads); "
            "``generated`` matches everything else (agent/block output). "
            "Ignored when ``session_id`` is set."
        ),
    ),
    folder_id: str | None = Query(
        default=None,
        min_length=1,
        description="Only return files in this folder.",
    ),
    root_only: bool = Query(
        default=False,
        description="Only return root-level files (not in any folder).",
    ),
    expert_id: str | None = Query(
        default=None,
        min_length=1,
        description=(
            "Only return files from this hired expert's conversations. "
            "Cannot be combined with session_id, folder_id or root_only."
        ),
    ),
) -> ListFilesResponse:
    """
    List files in the user's workspace.

    When session_id is provided, only files for that session are returned.
    Otherwise, all files across sessions are listed. Results are paginated
    via `limit`/`offset`; `has_more` indicates whether additional pages exist.

    The Artifacts page uses ``q`` for name search and ``origin`` to filter
    between Uploaded (user-uploaded) and Generated (agent/block output) files.

    ``session_id`` (a per-session view) and the folder filters (``folder_id`` /
    ``root_only``) are distinct, mutually exclusive axes, and ``folder_id`` and
    ``root_only`` likewise conflict; passing conflicting filters returns a 400
    rather than silently yielding an empty list.

    ``expert_id`` narrows the listing to files from that hired expert's own
    conversations. It excludes the other axes for the same reason. An expert
    the caller does not own (or no longer has) yields an empty list.
    """
    # Treat empty-string session_id the same as omitted — an empty value
    # would otherwise silently list files across every session instead of
    # scoping to one.
    session_id = session_id or None

    # Reject conflicting filters instead of silently combining them into an
    # (almost always) empty result. Validate before touching the DB so an
    # invalid GET can't create a workspace row for a first-time user.
    # session_id scopes to one chat session; folder_id/root_only organize
    # files across the whole workspace.
    if session_id is not None and (folder_id is not None or root_only):
        raise fastapi.HTTPException(
            status_code=400,
            detail="session_id cannot be combined with folder_id or root_only",
        )
    if folder_id is not None and root_only:
        raise fastapi.HTTPException(
            status_code=400,
            detail="folder_id and root_only are mutually exclusive",
        )
    if expert_id is not None and (
        session_id is not None or folder_id is not None or root_only
    ):
        raise fastapi.HTTPException(
            status_code=400,
            detail="expert_id cannot be combined with session_id, folder_id or root_only",
        )

    workspace = await get_or_create_workspace(user_id)
    manager = WorkspaceManager(user_id, workspace.id, session_id)
    include_all = session_id is None

    # Origin → metadata filter. Uploads carry an exact ``{"origin":
    # "user-upload"}`` metadata; everything else (agent/block output, which
    # stores ``{}`` or other metadata) is "generated". ``metadata`` is never
    # SQL NULL (column default ``{}``), so whole-object (in)equality is
    # null-safe. Only applied when not session-scoped.
    metadata_equals: dict | None = None
    metadata_not_equals: dict | None = None
    if session_id is None and origin is not None:
        if origin == "uploaded":
            metadata_equals = _UPLOADED_METADATA
        else:  # "generated"
            metadata_not_equals = _UPLOADED_METADATA

    name_contains = (q or "").strip() or None

    # Fetch one extra to compute has_more without a separate count query.
    list_kwargs: dict[str, Any] = dict(
        limit=limit + 1,
        offset=offset,
        include_all_sessions=include_all,
        name_contains=name_contains,
        metadata_equals=metadata_equals,
        metadata_not_equals=metadata_not_equals,
        folder_id=folder_id,
        root_only=root_only,
    )
    if expert_id is not None:
        # Fails closed: an unowned or archived expert resolves to no sessions,
        # and an empty prefix list matches nothing.
        scope = await resolve_expert_workspace_scope(user_id, expert_id)
        list_kwargs["allowed_path_prefixes"] = [
            session_path_prefix(sid) for sid in scope.session_ids
        ]
    files = await manager.list_files(**list_kwargs)
    has_more = len(files) > limit
    page = files[:limit]
    expert_by_session = await _expert_ids_by_session(user_id, page)

    return ListFilesResponse(
        files=[_to_file_item(f, expert_by_session) for f in page],
        offset=offset,
        has_more=has_more,
    )
