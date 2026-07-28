"""CoPilot tools for moving and copying workspace files.

Both tools are server-side: the file's bytes never enter the model's context or
this process's memory. That replaces the old three-call workaround (read into
the sandbox → write back to the new path → delete the original), which loaded
the entire file — base64-encoded, for binary — into the context window purely
to relocate it.
"""

import logging
from typing import Any, Optional

from backend.copilot.context import get_workspace_manager
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase
from .workspace_files import (
    _SKILLS_REGISTRY_ERROR,
    _path_under_skills_registry,
    _resolve_file,
)

logger = logging.getLogger(__name__)


class WorkspaceFileMovedResponse(ToolResponseBase):
    """Response after moving a file to a new workspace path."""

    type: ResponseType = ResponseType.WORKSPACE_FILE_MOVED
    file_id: str
    name: str
    path: str
    previous_path: str
    mime_type: str
    size_bytes: int


class WorkspaceFileCopiedResponse(ToolResponseBase):
    """Response after copying a file to a new workspace path."""

    type: ResponseType = ResponseType.WORKSPACE_FILE_COPIED
    file_id: str
    source_file_id: str
    name: str
    path: str
    source_path: str
    mime_type: str
    size_bytes: int
    download_url: str


_TARGET_PARAMS: dict[str, Any] = {
    "file_id": {
        "type": "string",
        "description": "File ID from list_workspace_files.",
    },
    "path": {
        "type": "string",
        "description": "Path of the file to act on (alternative to file_id).",
    },
    "new_path": {
        "type": "string",
        "description": "Destination path incl. filename (e.g. '/reports/q1.pdf').",
    },
    "folder_id": {
        "type": "string",
        "description": (
            "Workspace folder for the result. Defaults to the source's folder."
        ),
    },
    "overwrite": {
        "type": "boolean",
        "description": "Replace an existing file at new_path (default: false).",
    },
}


class _WorkspaceTransferTool(BaseTool):
    """Shared argument handling for the move and copy tools."""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": dict(_TARGET_PARAMS),
            # file_id and path are alternatives, so only new_path is required.
            "required": ["new_path"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _resolve_target(
        self,
        user_id: str | None,
        session_id: str,
        file_id: Optional[str],
        path: Optional[str],
        new_path: Optional[str],
    ):
        """Validate arguments and resolve the source file.

        Returns ``(manager, source_file_id, file_info)`` or an ``ErrorResponse``.
        """
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        if not file_id and not path:
            return ErrorResponse(
                message="Please provide either file_id or path", session_id=session_id
            )
        if not new_path:
            return ErrorResponse(
                message="Please provide new_path (the destination virtual path)",
                session_id=session_id,
            )

        # The skills registry owns its own tooling (store_skill / delete_skill),
        # which enforces frontmatter validation, the per-user cap and content
        # sanitisation. Moving or copying files in or out of it by raw path
        # would bypass every one of those checks.
        if _path_under_skills_registry(path) or _path_under_skills_registry(new_path):
            return ErrorResponse(message=_SKILLS_REGISTRY_ERROR, session_id=session_id)

        manager = await get_workspace_manager(user_id, session_id)
        resolved = await _resolve_file(manager, file_id, path, session_id)
        if isinstance(resolved, ErrorResponse):
            return resolved
        source_file_id, file_info = resolved

        # A file_id target only reveals its path after lookup, so re-check.
        if _path_under_skills_registry(file_info.path):
            return ErrorResponse(message=_SKILLS_REGISTRY_ERROR, session_id=session_id)

        return manager, source_file_id, file_info


class MoveWorkspaceFileTool(_WorkspaceTransferTool):
    """Tool for moving/renaming a file within the workspace."""

    @property
    def name(self) -> str:
        return "move_workspace_file"

    @property
    def description(self) -> str:
        return (
            "Move or rename a workspace file in one call. Specify file_id or "
            "path, plus new_path. Server-side: content is never read, so cost "
            "is independent of file size. Use this instead of "
            "read_workspace_file + write_workspace_file + delete_workspace_file. "
            "Paths are session-scoped; use /sessions/<id>/... to cross sessions."
        )

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        file_id: Optional[str] = None,
        path: Optional[str] = None,
        new_path: Optional[str] = None,
        folder_id: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        target = await self._resolve_target(
            user_id, session_id, file_id, path, new_path
        )
        if isinstance(target, ErrorResponse):
            return target
        manager, source_file_id, file_info = target
        assert new_path is not None
        previous_path = file_info.path

        try:
            moved = await manager.move_file(
                file_id=source_file_id,
                new_path=new_path,
                folder_id=folder_id or None,
                overwrite=overwrite,
            )
        except FileNotFoundError as e:
            return ErrorResponse(message=str(e), session_id=session_id)
        except ValueError as e:
            return ErrorResponse(message=str(e), session_id=session_id)
        except Exception as e:
            logger.error(f"Error moving workspace file: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to move workspace file: {e}",
                error=str(e),
                session_id=session_id,
            )

        if moved.path == previous_path:
            msg = f"{moved.name} is already at workspace:{moved.path} — nothing to do"
        else:
            msg = (
                f"Moved {moved.name} from workspace:{previous_path} to "
                f"workspace:{moved.path} ({moved.size_bytes:,} bytes)"
            )
        return WorkspaceFileMovedResponse(
            file_id=moved.id,
            name=moved.name,
            path=moved.path,
            previous_path=previous_path,
            mime_type=moved.mime_type,
            size_bytes=moved.size_bytes,
            message=msg,
            session_id=session_id,
        )


class CopyWorkspaceFileTool(_WorkspaceTransferTool):
    """Tool for duplicating a file within the workspace."""

    @property
    def name(self) -> str:
        return "copy_workspace_file"

    @property
    def description(self) -> str:
        return (
            "Copy a workspace file to a new path in one call. Specify file_id "
            "or path, plus new_path. Server-side byte copy — content is never "
            "read into context. Counts against your storage quota. Paths are "
            "session-scoped; use /sessions/<id>/... to cross sessions."
        )

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        file_id: Optional[str] = None,
        path: Optional[str] = None,
        new_path: Optional[str] = None,
        folder_id: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        target = await self._resolve_target(
            user_id, session_id, file_id, path, new_path
        )
        if isinstance(target, ErrorResponse):
            return target
        manager, source_file_id, file_info = target
        assert new_path is not None

        try:
            copied = await manager.copy_file(
                file_id=source_file_id,
                new_path=new_path,
                folder_id=folder_id or None,
                overwrite=overwrite,
            )
        except FileNotFoundError as e:
            return ErrorResponse(message=str(e), session_id=session_id)
        except ValueError as e:
            msg = str(e)
            if msg.startswith("Storage limit exceeded"):
                msg += (
                    " Use list_workspace_files to find candidates, then "
                    "delete_workspace_file to free space and retry — or ask "
                    "the user to upgrade their plan."
                )
            return ErrorResponse(message=msg, session_id=session_id)
        except Exception as e:
            logger.error(f"Error copying workspace file: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to copy workspace file: {e}",
                error=str(e),
                session_id=session_id,
            )

        normalized_mime = (copied.mime_type or "").split(";", 1)[0].strip().lower()
        download_url = (
            f"workspace://{copied.id}#{normalized_mime}"
            if normalized_mime
            else f"workspace://{copied.id}"
        )
        return WorkspaceFileCopiedResponse(
            file_id=copied.id,
            source_file_id=source_file_id,
            name=copied.name,
            path=copied.path,
            source_path=file_info.path,
            mime_type=normalized_mime,
            size_bytes=copied.size_bytes,
            download_url=download_url,
            message=(
                f"Copied {file_info.name} from workspace:{file_info.path} to "
                f"workspace:{copied.path} ({copied.size_bytes:,} bytes)"
            ),
            session_id=session_id,
        )
