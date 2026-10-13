"""CoPilot tools for organising workspace files into folders.

Workspace folders are a DB-level grouping over workspace files: a file's
membership is its ``folderId`` and its path is unaffected. They are the folders
the Files page shows, so what the agent creates here is visible to the user at
once. Folders nest; ``list_workspace_files`` lists them beside the files.

Folders are the account owner's, so these tools refuse an expert session.
Distinct from ``create_folder`` / ``delete_folder``, which manage *library*
folders holding agents.
"""

import logging
from typing import Any, Optional

from backend.api.features.library.exceptions import FolderAlreadyExistsError
from backend.copilot.context import get_workspace_manager
from backend.copilot.model import ChatSession
from backend.data.db_accessors import workspace_folder_db
from backend.data.workspace_folder import WorkspaceFolder
from backend.util.exceptions import NotFoundError
from backend.util.workspace_transfer import WORKSPACE_FOLDERS_OWNER_ONLY

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
    """Response after deleting a workspace folder and its subfolders."""

    type: ResponseType = ResponseType.WORKSPACE_FOLDER_DELETED
    folder_id: str
    name: str
    subfolders_deleted: int
    files_moved_to_root: int


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
        parent_id=folder.parent_id,
        file_count=folder.file_count,
    )


async def _owner_workspace_id(user_id: str, session_id: str) -> str | ErrorResponse:
    """The caller's workspace, or a refusal when an expert is acting."""
    manager = await get_workspace_manager(user_id, session_id)
    if manager.scope is not None:
        return ErrorResponse(
            message=WORKSPACE_FOLDERS_OWNER_ONLY,
            error="access_denied",
            session_id=session_id,
        )
    return manager.workspace_id


def _find_folder(
    folders: list[WorkspaceFolder],
    folder_id: Optional[str],
    name: Optional[str],
) -> WorkspaceFolder | str:
    """A folder by ID, else by name (case-insensitive, exact case first); a
    message when it can't.

    Folders nest, so two can share a name under different parents: a name that
    matches more than one is refused rather than guessed.
    """
    if folder_id:
        found = next((f for f in folders if f.id == folder_id), None)
        return found or f"Workspace folder not found: {folder_id}"
    wanted = (name or "").strip()
    matches = [f for f in folders if f.name.lower() == wanted.lower()]
    # Names are unique per parent by exact case, so an exact match settles a
    # tie between "Invoices" and "invoices".
    exact = [f for f in matches if f.name == wanted]
    matches = exact or matches
    if not matches:
        return f"Workspace folder not found: {name}"
    if len(matches) > 1:
        ids = ", ".join(f.id for f in matches)
        return (
            f"More than one workspace folder is named '{name}' ({ids}); pass folder_id."
        )
    return matches[0]


def _subtree(folders: list[WorkspaceFolder], root_id: str) -> list[WorkspaceFolder]:
    """*root_id*'s folder and every folder beneath it."""
    children: dict[Optional[str], list[WorkspaceFolder]] = {}
    for f in folders:
        children.setdefault(f.parent_id, []).append(f)
    by_id = {f.id: f for f in folders}
    found, queue, seen = [], [root_id], {root_id}
    while queue:
        current = queue.pop()
        found.append(by_id[current])
        for child in children.get(current, []):
            if child.id not in seen:
                seen.add(child.id)
                queue.append(child.id)
    return found


class CreateWorkspaceFolderTool(BaseTool):
    """Tool for creating a folder in the user's file workspace."""

    @property
    def name(self) -> str:
        return "create_workspace_folder"

    @property
    def description(self) -> str:
        return (
            "Create a folder for workspace FILES, at the root or in parent_id. "
            "For agent folders, use tool:create_folder."
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
                    "description": "Folder name, max 100 chars.",
                },
                "parent_id": {
                    "type": "string",
                    "description": "Parent folder ID; omit for the root.",
                    "entity": None,
                },
            },
            "required": ["name"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        name: str = "",
        parent_id: Optional[str] = None,
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
                message=f"Folder name too long (max {_MAX_FOLDER_NAME_LENGTH} characters)",
                session_id=session_id,
            )
        parent_id = parent_id or None
        taken = (
            f"A workspace folder named '{name}' already exists there. "
            "Use tool:list_workspace_files to see the existing folders."
        )

        try:
            workspace_id = await _owner_workspace_id(user_id, session_id)
            if isinstance(workspace_id, ErrorResponse):
                return workspace_id
            folders = await workspace_folder_db().list_workspace_folders(workspace_id)
            if parent_id and all(f.id != parent_id for f in folders):
                return ErrorResponse(
                    message=f"Parent workspace folder not found: {parent_id}",
                    session_id=session_id,
                )
            # Checked here as well as by the database because the duplicate
            # error does not survive the DatabaseManager RPC hop as itself.
            if any(f.name == name and f.parent_id == parent_id for f in folders):
                return ErrorResponse(message=taken, session_id=session_id)
            folder = await workspace_folder_db().create_workspace_folder(
                workspace_id=workspace_id, name=name, parent_id=parent_id
            )
        except FolderAlreadyExistsError:
            return ErrorResponse(message=taken, session_id=session_id)
        except NotFoundError as e:
            return ErrorResponse(message=str(e), session_id=session_id)
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
            "Delete a workspace FILE folder and its subfolders; their files "
            "move to the root. For agent folders, use tool:delete_folder."
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
                    "description": "Folder ID.",
                    "entity": None,
                },
                "name": {
                    "type": "string",
                    "description": "Folder name (alternative to folder_id).",
                },
            },
            # One of folder_id or name; the schema cannot say "either".
            "required": [],
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
            workspace_id = await _owner_workspace_id(user_id, session_id)
            if isinstance(workspace_id, ErrorResponse):
                return workspace_id
            folders = await workspace_folder_db().list_workspace_folders(workspace_id)
            folder = _find_folder(folders, folder_id, name)
            if isinstance(folder, str):
                return ErrorResponse(message=folder, session_id=session_id)
            doomed = _subtree(folders, folder.id)
            files = await workspace_folder_db().delete_workspace_folder(
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

        subfolders = len(doomed) - 1
        detail = f"; {files} file(s) moved to the workspace root" if files else ""
        if subfolders:
            detail += f"; {subfolders} subfolder(s) deleted with it"
        return WorkspaceFolderDeletedResponse(
            folder_id=folder.id,
            name=folder.name,
            subfolders_deleted=subfolders,
            files_moved_to_root=files,
            message=f"Deleted workspace folder '{folder.name}'{detail or ' (it was empty)'}",
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
            "Put workspace files in a folder (folder_id or folder_name) or "
            "back at the root (to_root). Paths are untouched; to change one, "
            "use tool:move_workspace_file."
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
                    "description": "File IDs.",
                },
                "folder_id": {
                    "type": "string",
                    "description": "Destination folder ID.",
                    "entity": None,
                },
                "folder_name": {
                    "type": "string",
                    "description": "Destination folder name.",
                },
                "to_root": {
                    "type": "boolean",
                    "description": "Move to the workspace root.",
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
            workspace_id = await _owner_workspace_id(user_id, session_id)
            if isinstance(workspace_id, ErrorResponse):
                return workspace_id
            target: WorkspaceFolder | None = None
            if folder_id:
                target = await workspace_folder_db().get_workspace_folder(
                    folder_id, workspace_id
                )
            elif folder_name:
                found = _find_folder(
                    await workspace_folder_db().list_workspace_folders(workspace_id),
                    None,
                    folder_name,
                )
                if isinstance(found, str):
                    return ErrorResponse(
                        message=f"{found}. Use tool:create_workspace_folder to create it first.",
                        session_id=session_id,
                    )
                target = found

            moved = await workspace_folder_db().bulk_move_files_to_folder(
                workspace_id=workspace_id,
                file_ids=file_ids,
                folder_id=target.id if target else None,
            )
        except NotFoundError:
            return ErrorResponse(
                message=(
                    f"Workspace folder not found: {folder_id or folder_name}. "
                    "Use tool:create_workspace_folder to create it first."
                ),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error moving workspace files to folder: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to move workspace files: {e}",
                error=str(e),
                session_id=session_id,
            )

        moved_ids = [f.id for f in moved]
        if not moved_ids:
            return ErrorResponse(
                message=(
                    f"No files were moved: none of the {len(file_ids)} file ID(s) "
                    "were found in your workspace"
                ),
                session_id=session_id,
            )
        # bulk_move_files_to_folder silently drops IDs outside this workspace.
        skipped = len(set(file_ids)) - len(moved_ids)
        destination = f"'{target.name}'" if target else "the workspace root"
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
