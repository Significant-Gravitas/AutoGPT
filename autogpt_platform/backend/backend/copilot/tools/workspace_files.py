"""CoPilot tools for workspace file operations."""

import base64
import logging
import mimetypes
import os
from collections import deque
from typing import Any, Optional

from prisma.enums import APIKeyPermission

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.context import (
    get_current_sandbox,
    get_sdk_cwd,
    get_workspace_manager,
    is_allowed_local_path,
    looks_like_sdk_tool_result_path,
    sdk_tool_result_redirect_hint,
)
from backend.copilot.model import ChatSession
from backend.copilot.tools.workdir import (
    resolve_sandbox_path_or_error,
    save_to_workdir,
    validate_ephemeral_path,
)
from backend.data.activity_event import ActivityEventDraft
from backend.data.workspace_folder import WorkspaceFolder
from backend.data.workspace_scope import WorkspaceAccessDeniedError
from backend.util.settings import Config
from backend.util.workspace import WorkspaceManager

from .base import BaseTool
from .models import (
    ErrorResponse,
    ResponseType,
    ToolResponseBase,
    WorkspaceFileInfoData,
    WorkspaceFolderInfoData,
)

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


async def _read_source_path(source_path: str, session_id: str) -> bytes | ErrorResponse:
    """Read *source_path* from E2B sandbox or local ephemeral directory."""

    sandbox = get_current_sandbox()
    if sandbox is not None:
        remote = resolve_sandbox_path_or_error(source_path, session_id, "source_path")
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
    validated = validate_ephemeral_path(
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


class WorkspaceFileListResponse(ToolResponseBase):
    """Response containing list of workspace files."""

    type: ResponseType = ResponseType.WORKSPACE_FILE_LIST
    files: list[WorkspaceFileInfoData]
    total_count: int
    # Folders at the listed level, so the model can walk the tree without a
    # second tool. Folders are user-level: an expert sees the whole tree and
    # only the files inside it are filtered by its scope.
    folders: list[WorkspaceFolderInfoData] = []


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
    def allow_external_use(self):
        return True, [APIKeyPermission.READ_FILES]

    @property
    def description(self) -> str:
        return (
            "List persistent workspace files, with the folders at that level. "
            "Files uploaded on the Files page sit at the workspace root or in "
            "a folder; pass folder_id to list one, then read a file with "
            "read_workspace_file. For ephemeral session files, use SDK "
            "Glob/Read instead."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path_prefix": {
                    "type": "string",
                    "description": "Filter by path prefix (e.g. '/documents/').",
                },
                "folder_id": {
                    "type": "string",
                    "description": (
                        "Files in this folder. Omit for this chat's files and "
                        "the folders at the workspace root."
                    ),
                },
                "recursive": {
                    "type": "boolean",
                    "description": (
                        "With folder_id, also list files in its subfolders "
                        "(default: false)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max files to return (default 50, max 100).",
                    "minimum": 1,
                    "maximum": 100,
                },
                "include_all_sessions": {
                    "type": "boolean",
                    "description": (
                        "Include files from every chat, not just this one "
                        "(default: false). An expert chat sees its own "
                        "conversations, ones it delegated, and the user's files."
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
        path_prefix: Optional[str] = None,
        folder_id: Optional[str] = None,
        recursive: bool = False,
        limit: int = 50,
        include_all_sessions: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )

        limit = min(limit, 100)

        # "" is not "no folder": it survives the manager's `is not None` test,
        # which drops the current-session filter, and then reads as false in the
        # query, which drops the folder filter — listing the whole workspace.
        if folder_id is not None and not folder_id.strip():
            return ErrorResponse(
                message="folder_id must name a folder; omit it to list the root",
                session_id=session_id,
            )

        try:
            manager = await get_workspace_manager(user_id, session_id)
            folders = await manager.list_folders()
            descend = folder_id if recursive else None
            subtree, unsearched = (
                _folder_subtree(folders, descend) if descend else ([], 0)
            )
            list_kwargs: dict[str, Any] = (
                {"folder_ids": subtree} if descend else {"folder_id": folder_id}
            )
            files = await manager.list_files(
                path=path_prefix,
                limit=limit,
                include_all_sessions=include_all_sessions,
                **list_kwargs,
            )
            total = await manager.get_file_count(
                path=path_prefix,
                include_all_sessions=include_all_sessions,
                **list_kwargs,
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
            folder_infos = [
                WorkspaceFolderInfoData(
                    folder_id=f.id,
                    name=f.name,
                    parent_id=f.parent_id,
                    file_count=f.file_count,
                )
                for f in folders
                if f.parent_id == folder_id
            ]
            scope = "all sessions" if include_all_sessions else "current session"
            names = {f.id: f.name for f in folders}
            where = (
                f"folder {names.get(folder_id, folder_id)}"
                if folder_id
                else f"workspace ({scope})"
            )
            total_size = sum(f.size_bytes for f in file_infos)

            # Build a human-readable summary so the agent can relay details.
            lines = [f"Found {len(files)} file(s) in {where}:"]
            for f in file_infos:
                lines.append(f"  - {f.path} ({f.size_bytes:,} bytes, {f.mime_type})")
            if total > len(files):
                lines.append(f"  ... and {total - len(files)} more")
            for d in folder_infos:
                lines.append(
                    f"  [folder] {d.name} ({d.file_count} file(s)), "
                    f"folder_id={d.folder_id}"
                )
            if unsearched:
                lines.append(
                    f"  ... and {unsearched} subfolder(s) not searched; "
                    "list them with folder_id."
                )
            lines.append(f"Total size: {total_size:,} bytes")

            return WorkspaceFileListResponse(
                files=file_infos,
                total_count=total,
                folders=folder_infos,
                message="\n".join(lines),
                session_id=session_id,
            )
        except WorkspaceAccessDeniedError as e:
            return ErrorResponse(
                message=str(e), error="access_denied", session_id=session_id
            )
        except Exception as e:
            logger.error(f"Error listing workspace files: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to list workspace files: {e}",
                error=str(e),
                session_id=session_id,
            )


# A recursive listing walks at most this many folders. Deep trees are the
# user's own making, and an unbounded ``IN`` grows with every folder they add.
_MAX_RECURSIVE_FOLDERS = 200


def _folder_subtree(
    folders: list[WorkspaceFolder], folder_id: str
) -> tuple[list[str], int]:
    """*folder_id* plus its descendants, capped, and how many were left out.

    Nearest first, so a cap truncates the deepest folders rather than an
    arbitrary set, and the caller can name the remainder for the model to
    list directly. Deliberately not shared with ``workspace_folder._subtree_ids``:
    that one walks DB rows and must not cap, since it drives a delete.
    """
    children: dict[str | None, list[str]] = {}
    for folder in folders:
        children.setdefault(folder.parent_id, []).append(folder.id)

    subtree = [folder_id]
    seen = {folder_id}
    queue = deque([folder_id])
    overflow = 0
    while queue:
        for child in children.get(queue.popleft(), []):
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
            # Past the cap the walk keeps going but stops collecting, so the
            # count the caller reports is every folder left out, not just the
            # first one over the line.
            if len(subtree) >= _MAX_RECURSIVE_FOLDERS:
                overflow += 1
            else:
                subtree.append(child)
    return subtree, overflow


class ReadWorkspaceFileTool(BaseTool):
    """Tool for reading file content from workspace."""

    MAX_INLINE_SIZE_BYTES = 32 * 1024  # 32KB for text/image files
    PREVIEW_SIZE = 500

    @property
    def name(self) -> str:
        return "read_workspace_file"

    @property
    def allow_external_use(self):
        return True, [APIKeyPermission.READ_FILES]

    @property
    def description(self) -> str:
        return (
            "Read a file from persistent workspace. Specify file_id or path. "
            "Small text/image files return inline; large/binary return metadata+URL. "
            "Use save_to_path to copy to working dir for processing. "
            "Use offset/length for paginated reads. "
            "Paths resolve in the current session; use /sessions/<id>/... or "
            "file_id to reach elsewhere. An expert chat reads its own "
            "conversations, ones it delegated, and the user's files."
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
        file_id: Optional[str] = None,
        path: Optional[str] = None,
        save_to_path: Optional[str] = None,
        force_download_url: bool = False,
        offset: int = 0,
        length: Optional[int] = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )

        char_offset: int = max(0, offset)
        char_length: Optional[int] = length

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
                # Relative SDK-tool-result shorthand must short-circuit
                # *before* the ``is_allowed_local_path`` fallback: when
                # ``sdk_cwd`` is set, the shorthand resolves under it and
                # passes the allow check, but the file doesn't actually
                # exist at that resolved path → ``_read_local_tool_result``
                # returns a generic "Path not allowed" and the redirect
                # branch below never runs. Catch relative shorthands here
                # so the model sees the helpful redirect on the first try.
                # Absolute SDK paths still take the local-read path below
                # (legit fallback for ``read_workspace_file`` with an
                # absolute SDK tool-results path).
                if (
                    path
                    and not os.path.isabs(path)
                    and not path.startswith("~")
                    and looks_like_sdk_tool_result_path(path)
                ):
                    return ErrorResponse(
                        message=sdk_tool_result_redirect_hint(path),
                        session_id=session_id,
                    )
                if path and is_allowed_local_path(path, sdk_cwd):
                    return _read_local_tool_result(
                        path, char_offset, char_length, session_id, sdk_cwd=sdk_cwd
                    )
                # Path looks like SDK tool-results but isn't reachable
                # via the local-read path either (e.g. absolute SDK path
                # not under sdk_cwd) — redirect to read_tool_result /
                # @@agptfile rather than returning a generic
                # "Path not allowed".
                if path and looks_like_sdk_tool_result_path(path):
                    return ErrorResponse(
                        message=sdk_tool_result_redirect_hint(path),
                        session_id=session_id,
                    )
                return resolved
            target_file_id, file_info = resolved

            # If save_to_path, read + save; cache bytes for possible inline reuse.
            cached_content: bytes | None = None
            if save_to_path:
                cached_content = await manager.read_file_by_id(target_file_id)
                result = await save_to_workdir(save_to_path, cached_content, session_id)
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
        except WorkspaceAccessDeniedError as e:
            return ErrorResponse(
                message=str(e), error="access_denied", session_id=session_id
            )
        except Exception as e:
            logger.error(f"Error reading workspace file: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to read workspace file: {e}",
                error=str(e),
                session_id=session_id,
            )


# Paths under ``/skills/`` and ``/experts/<id>/skills/`` are managed by the
# skills registry — the ``store_skill`` / ``delete_skill`` tools enforce
# frontmatter validation, the per-expert cap, name regex, and content
# sanitisation. Allowing plain write_workspace_file / delete_workspace_file
# there would bypass all of that and let the model accidentally (or
# maliciously) corrupt the registry. Reads stay open so the model can still
# inspect sibling references inside a skill bundle.
_SKILLS_REGISTRY_PREFIX = "skills/"
_EXPERTS_PREFIX = "experts/"
_SKILLS_REGISTRY_ERROR = (
    "Path is managed by the skills registry; use tool:store_skill / "
    "tool:delete_skill instead. (read_workspace_file can still read "
    "sibling files inside a skill bundle.)"
)


def _path_under_skills_registry(path: str | None) -> bool:
    """Return ``True`` when *path* normalises to a location under either
    skills-registry folder — Otto's ``/skills/...`` or an expert's
    ``/experts/<id>/skills/...`` — case-insensitively."""
    if not path:
        return False
    # Strip leading slashes + whitespace, lower-case so case variants
    # (``Skills/foo``) cannot bypass the check.
    normalised = path.strip().lstrip("/").lower()
    if normalised.startswith(_SKILLS_REGISTRY_PREFIX) or normalised == "skills":
        return True
    if not normalised.startswith(_EXPERTS_PREFIX):
        return False
    # ``experts/<id>/skills`` and anything below it; the id is any single
    # segment, so a deeper path under another expert folder is not caught
    # here — nothing else lives under ``/experts/`` in the workspace.
    rest = normalised[len(_EXPERTS_PREFIX) :].split("/", 1)
    return len(rest) == 2 and (
        rest[1].startswith(_SKILLS_REGISTRY_PREFIX) or rest[1] == "skills"
    )


class WriteWorkspaceFileTool(BaseTool):
    """Tool for writing files to workspace."""

    @property
    def name(self) -> str:
        return "write_workspace_file"

    @property
    def allow_external_use(self):
        return True, [APIKeyPermission.WRITE_FILES]

    @property
    def description(self) -> str:
        return (
            "Write a file to persistent workspace (survives across sessions). "
            "Provide exactly one of: content (text), content_base64 (binary), "
            f"or source_path (copy from working dir). Max {_MAX_FILE_SIZE_MB}MB. "
            "Paths scoped to current session; use /sessions/<id>/... for "
            "cross-session access (expert chats are limited to their own "
            "conversations)."
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

    def activity_event(
        self,
        session: ChatSession,
        result: ToolResponseBase,
        **kwargs,
    ) -> ActivityEventDraft | None:
        if not isinstance(result, WorkspaceWriteResponse):
            return None
        return ActivityEventDraft(
            category="FILE",
            event_type="file.updated" if kwargs.get("overwrite") else "file.created",
            title=result.name,
            object_id=result.file_id,
            data={
                "path": result.path,
                "mime_type": result.mime_type,
                "size_bytes": result.size_bytes,
            },
        )

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        filename: str = "",
        source_path: str | None = None,
        content: str | None = None,
        content_base64: str | None = None,
        path: str | None = None,
        mime_type: str | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )

        if not filename:
            # When ALL parameters are missing, the most likely cause is
            # output token truncation: the LLM tried to inline a very large
            # file as `content`, the SDK silently truncated the tool call
            # arguments to `{}`, and we receive nothing.  Return an
            # actionable error instead of a generic "filename required".
            has_any_content = any(
                kwargs.get(k) for k in ("content", "content_base64", "source_path")
            )
            if not has_any_content:
                return ErrorResponse(
                    message=(
                        "Tool call appears truncated (no arguments received). "
                        "This happens when the content is too large for a "
                        "single tool call. Instead of passing content inline, "
                        "first write the file to the working directory using "
                        "bash_exec (e.g. cat > /home/user/file.md << 'EOF'... "
                        "EOF), then use source_path to copy it to workspace: "
                        "write_workspace_file(filename='file.md', "
                        "source_path='/home/user/file.md')"
                    ),
                    session_id=session_id,
                )
            return ErrorResponse(
                message="Please provide a filename", session_id=session_id
            )

        # Block writes to the skills registry folder — they would bypass
        # store_skill's validation (cap, body limits, name regex,
        # sanitisation of server-injected tags).  Either an explicit
        # ``path`` or a ``filename`` defaulting to ``skills/...`` count.
        candidate_path = path if path is not None else f"/{filename}"
        if _path_under_skills_registry(candidate_path):
            return ErrorResponse(message=_SKILLS_REGISTRY_ERROR, session_id=session_id)

        source_path_arg: str | None = source_path
        content_text: str | None = content
        content_b64: str | None = content_base64

        resolved = await _resolve_write_content(
            content_text,
            content_b64,
            source_path_arg,
            session_id,
        )
        if isinstance(resolved, ErrorResponse):
            return resolved
        content_bytes: bytes = resolved

        max_size = _MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content_bytes) > max_size:
            return ErrorResponse(
                message=f"File too large. Maximum size is {_MAX_FILE_SIZE_MB}MB",
                session_id=session_id,
            )

        try:
            manager = await get_workspace_manager(user_id, session_id)
            rec = await manager.write_file(
                content=content_bytes,
                filename=filename,
                path=path,
                mime_type=mime_type,
                overwrite=overwrite,
                metadata={"origin": "agent-created"},
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
                    preview = content_bytes[:200].decode("utf-8", errors="replace")
                    if len(content_bytes) > 200:
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
        except VirusDetectedError as e:
            logger.warning(f"Virus detected in uploaded file: {e.threat_name}")
            return ErrorResponse(message=str(e), session_id=session_id)
        except VirusScanError as e:
            logger.error(f"Virus scan infrastructure error: {e}", exc_info=True)
            return ErrorResponse(message=str(e), session_id=session_id)
        except WorkspaceAccessDeniedError as e:
            return ErrorResponse(
                message=str(e), error="access_denied", session_id=session_id
            )
        except ValueError as e:
            msg = str(e)
            if msg.startswith("Storage limit exceeded"):
                msg += (
                    " Use tool:list_workspace_files to find candidates, then "
                    "tool:delete_workspace_file to free space and retry — or ask "
                    "the user to upgrade their plan."
                )
            return ErrorResponse(message=msg, session_id=session_id)
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
    def allow_external_use(self):
        return True, [APIKeyPermission.WRITE_FILES]

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

    def activity_event(
        self,
        session: ChatSession,
        result: ToolResponseBase,
        **kwargs,
    ) -> ActivityEventDraft | None:
        if not isinstance(result, WorkspaceDeleteResponse) or not result.success:
            return None
        return ActivityEventDraft(
            category="FILE",
            event_type="file.deleted",
            title=kwargs.get("path") or "Workspace file",
            object_id=result.file_id,
        )

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        file_id: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        if not file_id and not path:
            return ErrorResponse(
                message="Please provide either file_id or path", session_id=session_id
            )

        # Reject deletes targeting the skills registry by path up-front.
        # file_id targets are checked AFTER resolution below, since the
        # path is only known once the file is looked up.
        if _path_under_skills_registry(path):
            return ErrorResponse(message=_SKILLS_REGISTRY_ERROR, session_id=session_id)

        try:
            manager = await get_workspace_manager(user_id, session_id)
            resolved = await _resolve_file(manager, file_id, path, session_id)
            if isinstance(resolved, ErrorResponse):
                return resolved
            target_file_id, file_info = resolved

            # Fail closed: if the resolved file has no ``path`` attribute the
            # workspace shape has drifted and we cannot verify the ACL, so
            # refuse to delete rather than fall through with ``None``.
            resolved_path = getattr(file_info, "path", None)
            if not isinstance(resolved_path, str) or _path_under_skills_registry(
                resolved_path
            ):
                return ErrorResponse(
                    message=_SKILLS_REGISTRY_ERROR, session_id=session_id
                )

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
        except WorkspaceAccessDeniedError as e:
            return ErrorResponse(
                message=str(e), error="access_denied", session_id=session_id
            )
        except Exception as e:
            logger.error(f"Error deleting workspace file: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to delete workspace file: {e}",
                error=str(e),
                session_id=session_id,
            )
