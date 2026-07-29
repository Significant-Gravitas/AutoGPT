"""CoPilot tools for organising workspace files into folders.

Workspace folders are a DB-level grouping over workspace files — a file's
membership is its ``folderId`` and its storage path is unaffected. They are the
same folders the Artifacts page shows, so folders the agent creates here are
immediately visible to the user.

Distinct from the ``create_folder`` / ``delete_folder`` tools, which manage
*library* folders holding agents.
"""

import logging
from typing import Any, Optional

from backend.api.features.library.exceptions import FolderAlreadyExistsError
from backend.copilot.model import ChatSession
from backend.data.db_accessors import workspace_db, workspace_folder_db
from backend.data.workspace_folder import WorkspaceFolder
from backend.util.exceptions import NotFoundError

from .base import BaseTool
from .models import (
    ErrorResponse,
    ResponseType,
    ToolResponseBase,
    WorkspaceFolderInfoData,
)

logger = logging.getLogger(__name__)

_MAX_FOLDER_NAME_LENGTH = 100


class WorkspaceFolderCreatedResponse(ToolResponseBase):
    """Response after creating a workspace folder."""

    type: ResponseType = ResponseType.WORKSPACE_FOLDER_CREATED
    folder: WorkspaceFolderInfoData


class WorkspaceFolderDeletedResponse(ToolResponseBase):
    """Response after deleting a workspace folder."""

    type: ResponseType = ResponseType.WORKSPACE_FOLDER_DELETED
    folder_id: str
    name: str
    files_moved_to_root: int


class WorkspaceFolderListResponse(ToolResponseBase):
    """Response listing the workspace's folders."""

    type: ResponseType = ResponseType.WORKSPACE_FOLDER_LIST
    folders: list[WorkspaceFolderInfoData]


class WorkspaceFilesMovedToFolderResponse(ToolResponseBase):
    """Response after moving files into a workspace folder."""

    type: ResponseType = ResponseType.WORKSPACE_FILES_MOVED_TO_FOLDER
    folder_id: str | None
    folder_name: str
    file_ids: list[str]


def _to_info(folder: WorkspaceFolder) -> WorkspaceFolderInfoData:
    return WorkspaceFolderInfoData(
        folder_id=folder.id,
        name=folder.name,
        icon=folder.icon,
        file_count=folder.file_count,
    )


async def _workspace_id(user_id: str) -> str:
    workspace = await workspace_db().get_or_create_workspace(user_id)
    return workspace.id


async def _resolve_folder(
    workspace_id: str,
    folder_id: Optional[str],
    name: Optional[str],
) -> WorkspaceFolder | None:
    """Look a folder up by ID, else by exact (case-insensitive) name."""
    folders = await workspace_folder_db().list_workspace_folders(workspace_id)
    if folder_id:
        return next((f for f in folders if f.id == folder_id), None)
    if name:
        wanted = name.strip().lower()
        return next((f for f in folders if f.name.lower() == wanted), None)
    return None


class CreateWorkspaceFolderTool(BaseTool):
    """Tool for creating a folder in the user's file workspace."""

    @property
    def name(self) -> str:
        return "create_workspace_folder"

    @property
    def description(self) -> str:
        return (
            "Create a folder for workspace FILES (shown on the Artifacts page). "
            "Names are unique. For agent folders, use create_folder."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Folder name (e.g. 'Invoices'). Max 100 chars.",
                },
                "icon": {
                    "type": "string",
                    "description": "Optional icon identifier.",
                },
            },
            "required": ["name"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        name: str = "",
        icon: Optional[str] = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )

        name = (name or "").strip()
        if not name:
            return ErrorResponse(
                message="Please provide a folder name", session_id=session_id
            )
        if len(name) > _MAX_FOLDER_NAME_LENGTH:
            return ErrorResponse(
                message=(
                    f"Folder name too long "
                    f"(max {_MAX_FOLDER_NAME_LENGTH} characters)"
                ),
                session_id=session_id,
            )

        try:
            workspace_id = await _workspace_id(user_id)
            folder = await workspace_folder_db().create_workspace_folder(
                workspace_id=workspace_id, name=name, icon=icon
            )
        except FolderAlreadyExistsError:
            return ErrorResponse(
                message=(
                    f"A workspace folder named '{name}' already exists. "
                    f"Use list_workspace_folders to see existing folders."
                ),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error creating workspace folder: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to create workspace folder: {e}",
                error=str(e),
                session_id=session_id,
            )

        return WorkspaceFolderCreatedResponse(
            folder=_to_info(folder),
            message=f"Created workspace folder '{folder.name}' (id: {folder.id})",
            session_id=session_id,
        )


class DeleteWorkspaceFolderTool(BaseTool):
    """Tool for deleting a folder from the user's file workspace."""

    @property
    def name(self) -> str:
        return "delete_workspace_folder"

    @property
    def description(self) -> str:
        return (
            "Delete a workspace FILE folder. Files inside are kept and moved to "
            "the workspace root. Specify folder_id or name. For agent folders, "
            "use delete_folder."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "folder_id": {
                    "type": "string",
                    "description": "Folder ID from list_workspace_folders.",
                },
                "name": {
                    "type": "string",
                    "description": "Folder name (alternative to folder_id).",
                },
            },
            "required": [],  # At least one of folder_id or name must be provided
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        folder_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        if not folder_id and not name:
            return ErrorResponse(
                message="Please provide either folder_id or name",
                session_id=session_id,
            )

        try:
            workspace_id = await _workspace_id(user_id)
            folder = await _resolve_folder(workspace_id, folder_id, name)
            if folder is None:
                return ErrorResponse(
                    message=f"Workspace folder not found: {folder_id or name}",
                    session_id=session_id,
                )
            await workspace_folder_db().delete_workspace_folder(
                folder_id=folder.id, workspace_id=workspace_id
            )
        except NotFoundError:
            return ErrorResponse(
                message=f"Workspace folder not found: {folder_id or name}",
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error deleting workspace folder: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to delete workspace folder: {e}",
                error=str(e),
                session_id=session_id,
            )

        moved = folder.file_count
        detail = (
            f"; {moved} file(s) moved to the workspace root"
            if moved
            else " (it was empty)"
        )
        return WorkspaceFolderDeletedResponse(
            folder_id=folder.id,
            name=folder.name,
            files_moved_to_root=moved,
            message=f"Deleted workspace folder '{folder.name}'{detail}",
            session_id=session_id,
        )


class ListWorkspaceFoldersTool(BaseTool):
    """Tool for listing folders in the user's file workspace."""

    @property
    def name(self) -> str:
        return "list_workspace_folders"

    @property
    def description(self) -> str:
        return (
            "List workspace FILE folders with their file counts. "
            "For agent folders, use list_folders."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

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

        try:
            workspace_id = await _workspace_id(user_id)
            folders = await workspace_folder_db().list_workspace_folders(workspace_id)
        except Exception as e:
            logger.error(f"Error listing workspace folders: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to list workspace folders: {e}",
                error=str(e),
                session_id=session_id,
            )

        infos = [_to_info(f) for f in folders]
        lines = [f"Found {len(infos)} workspace folder(s):"] + [
            f"  - {f.name} ({f.file_count} file(s), id: {f.folder_id})" for f in infos
        ]
        return WorkspaceFolderListResponse(
            folders=infos,
            message="\n".join(lines) if infos else "No workspace folders yet.",
            session_id=session_id,
        )


class MoveWorkspaceFilesToFolderTool(BaseTool):
    """Tool for moving workspace files into a folder (or back to root)."""

    @property
    def name(self) -> str:
        return "move_workspace_files_to_folder"

    @property
    def description(self) -> str:
        return (
            "Move workspace files into a folder, or back to the root. Specify "
            "folder_id or folder_name, or set to_root. Changes folder "
            "membership only — paths and content are untouched. To change a "
            "file's path, use move_workspace_file."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File IDs from list_workspace_files.",
                },
                "folder_id": {
                    "type": "string",
                    "description": "Destination folder ID from list_workspace_folders.",
                },
                "folder_name": {
                    "type": "string",
                    "description": "Destination folder name (alternative to folder_id).",
                },
                "to_root": {
                    "type": "boolean",
                    "description": "Move the files to the workspace root instead.",
                },
            },
            "required": ["file_ids"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        file_ids: Optional[list[str]] = None,
        folder_id: Optional[str] = None,
        folder_name: Optional[str] = None,
        to_root: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        if not file_ids:
            return ErrorResponse(
                message="Please provide file_ids to move", session_id=session_id
            )
        if to_root and (folder_id or folder_name):
            return ErrorResponse(
                message=(
                    "Provide either a destination folder (folder_id/folder_name) "
                    "or to_root, not both"
                ),
                session_id=session_id,
            )
        if not to_root and not folder_id and not folder_name:
            return ErrorResponse(
                message=(
                    "Provide a destination: a folder (folder_id/folder_name) "
                    "or to_root=true to move the files to the workspace root"
                ),
                session_id=session_id,
            )

        try:
            workspace_id = await _workspace_id(user_id)
            target: WorkspaceFolder | None = None
            if not to_root and (folder_id or folder_name):
                target = await _resolve_folder(workspace_id, folder_id, folder_name)
                if target is None:
                    return ErrorResponse(
                        message=(
                            f"Workspace folder not found: "
                            f"{folder_id or folder_name}. Use "
                            f"create_workspace_folder to create it first."
                        ),
                        session_id=session_id,
                    )

            moved = await workspace_folder_db().bulk_move_files_to_folder(
                workspace_id=workspace_id,
                file_ids=file_ids,
                folder_id=target.id if target else None,
            )
        except NotFoundError as e:
            return ErrorResponse(message=str(e), session_id=session_id)
        except Exception as e:
            logger.error(f"Error moving workspace files to folder: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to move workspace files: {e}",
                error=str(e),
                session_id=session_id,
            )

        destination = f"'{target.name}'" if target else "the workspace root"
        moved_ids = [f.id for f in moved]
        # bulk_move_files_to_folder silently drops IDs outside this workspace.
        skipped = len(file_ids) - len(moved_ids)
        if not moved_ids:
            return ErrorResponse(
                message=(
                    f"No files were moved: none of the {len(file_ids)} file ID(s) "
                    f"were found in your workspace"
                ),
                session_id=session_id,
            )
        msg = f"Moved {len(moved_ids)} file(s) to {destination}"
        if skipped > 0:
            msg += f"; {skipped} file ID(s) were not found in your workspace"
        return WorkspaceFilesMovedToFolderResponse(
            folder_id=target.id if target else None,
            folder_name=target.name if target else "",
            file_ids=moved_ids,
            message=msg,
            session_id=session_id,
        )
