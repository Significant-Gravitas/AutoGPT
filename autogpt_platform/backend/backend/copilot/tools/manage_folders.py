"""Folder management tools for the copilot."""

import logging
from typing import Any

from backend.api.features.library import db as library_db
from backend.api.features.library import model as library_model
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import (
    AgentsMovedToFolderResponse,
    ErrorResponse,
    FolderCreatedResponse,
    FolderDeletedResponse,
    FolderInfo,
    FolderListResponse,
    FolderMovedResponse,
    FolderTreeInfo,
    FolderUpdatedResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


def _folder_to_info(folder: library_model.LibraryFolder) -> FolderInfo:
    return FolderInfo(
        id=folder.id,
        name=folder.name,
        parent_id=folder.parent_id,
        icon=folder.icon,
        color=folder.color,
        agent_count=folder.agent_count,
        subfolder_count=folder.subfolder_count,
    )


def _tree_to_info(tree: library_model.LibraryFolderTree) -> FolderTreeInfo:
    return FolderTreeInfo(
        id=tree.id,
        name=tree.name,
        parent_id=tree.parent_id,
        icon=tree.icon,
        color=tree.color,
        agent_count=tree.agent_count,
        subfolder_count=tree.subfolder_count,
        children=[_tree_to_info(child) for child in tree.children],
    )


def _count_tree(nodes: list[library_model.LibraryFolderTree]) -> int:
    return sum(1 + _count_tree(n.children) for n in nodes)


class CreateFolderTool(BaseTool):
    """Tool for creating a library folder."""

    @property
    def name(self) -> str:
        return "create_folder"

    @property
    def description(self) -> str:
        return (
            "Create a new folder in the user's library to organize agents. "
            "Optionally nest it inside an existing folder using parent_id."
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
                    "description": "Name for the new folder (max 100 chars).",
                },
                "parent_id": {
                    "type": "string",
                    "description": (
                        "ID of the parent folder to nest inside. "
                        "Omit to create at root level."
                    ),
                },
                "icon": {
                    "type": "string",
                    "description": "Optional icon identifier for the folder.",
                },
                "color": {
                    "type": "string",
                    "description": "Optional hex color code (#RRGGBB).",
                },
            },
            "required": ["name"],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        name = kwargs.get("name", "").strip()
        parent_id = kwargs.get("parent_id")
        icon = kwargs.get("icon")
        color = kwargs.get("color")
        session_id = session.session_id if session else None

        if not name:
            return ErrorResponse(
                message="Please provide a folder name.",
                error="missing_name",
                session_id=session_id,
            )

        try:
            folder = await library_db.create_folder(
                user_id=user_id,
                name=name,
                parent_id=parent_id,
                icon=icon,
                color=color,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to create folder: {e}",
                error="create_folder_failed",
                session_id=session_id,
            )

        return FolderCreatedResponse(
            message=f"Folder '{folder.name}' created successfully!",
            folder=_folder_to_info(folder),
            session_id=session_id,
        )


class ListFoldersTool(BaseTool):
    """Tool for listing library folders."""

    @property
    def name(self) -> str:
        return "list_folders"

    @property
    def description(self) -> str:
        return (
            "List the user's library folders. "
            "Omit parent_id to get the full folder tree. "
            "Provide parent_id to list only direct children of that folder."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "parent_id": {
                    "type": "string",
                    "description": (
                        "List children of this folder. "
                        "Omit to get the full folder tree."
                    ),
                },
            },
            "required": [],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        parent_id = kwargs.get("parent_id")
        session_id = session.session_id if session else None

        try:
            if parent_id:
                folders = await library_db.list_folders(
                    user_id=user_id, parent_id=parent_id
                )
                return FolderListResponse(
                    message=f"Found {len(folders)} folder(s).",
                    folders=[_folder_to_info(f) for f in folders],
                    count=len(folders),
                    session_id=session_id,
                )
            else:
                tree = await library_db.get_folder_tree(user_id=user_id)
                flat_count = _count_tree(tree)
                return FolderListResponse(
                    message=f"Found {flat_count} folder(s) in your library.",
                    tree=[_tree_to_info(t) for t in tree],
                    count=flat_count,
                    session_id=session_id,
                )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to list folders: {e}",
                error="list_folders_failed",
                session_id=session_id,
            )


class UpdateFolderTool(BaseTool):
    """Tool for updating a folder's properties."""

    @property
    def name(self) -> str:
        return "update_folder"

    @property
    def description(self) -> str:
        return "Update a folder's name, icon, or color."

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
                    "description": "ID of the folder to update.",
                },
                "name": {
                    "type": "string",
                    "description": "New name for the folder.",
                },
                "icon": {
                    "type": "string",
                    "description": "New icon identifier.",
                },
                "color": {
                    "type": "string",
                    "description": "New hex color code (#RRGGBB).",
                },
            },
            "required": ["folder_id"],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        folder_id = kwargs.get("folder_id", "").strip()
        name = kwargs.get("name")
        icon = kwargs.get("icon")
        color = kwargs.get("color")
        session_id = session.session_id if session else None

        if not folder_id:
            return ErrorResponse(
                message="Please provide a folder_id.",
                error="missing_folder_id",
                session_id=session_id,
            )

        try:
            folder = await library_db.update_folder(
                folder_id=folder_id,
                user_id=user_id,
                name=name,
                icon=icon,
                color=color,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to update folder: {e}",
                error="update_folder_failed",
                session_id=session_id,
            )

        return FolderUpdatedResponse(
            message=f"Folder updated to '{folder.name}'.",
            folder=_folder_to_info(folder),
            session_id=session_id,
        )


class MoveFolderTool(BaseTool):
    """Tool for moving a folder to a new parent."""

    @property
    def name(self) -> str:
        return "move_folder"

    @property
    def description(self) -> str:
        return (
            "Move a folder to a different parent folder. "
            "Set target_parent_id to null to move to root level."
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
                    "description": "ID of the folder to move.",
                },
                "target_parent_id": {
                    "type": ["string", "null"],
                    "description": (
                        "ID of the new parent folder. "
                        "Use null to move to root level."
                    ),
                },
            },
            "required": ["folder_id"],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        folder_id = kwargs.get("folder_id", "").strip()
        target_parent_id = kwargs.get("target_parent_id")
        session_id = session.session_id if session else None

        if not folder_id:
            return ErrorResponse(
                message="Please provide a folder_id.",
                error="missing_folder_id",
                session_id=session_id,
            )

        try:
            folder = await library_db.move_folder(
                folder_id=folder_id,
                user_id=user_id,
                target_parent_id=target_parent_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to move folder: {e}",
                error="move_folder_failed",
                session_id=session_id,
            )

        dest = f"folder '{folder.name}'" if target_parent_id else "root level"
        return FolderMovedResponse(
            message=f"Folder moved to {dest}.",
            folder=_folder_to_info(folder),
            target_parent_id=target_parent_id,
            session_id=session_id,
        )


class DeleteFolderTool(BaseTool):
    """Tool for deleting a folder."""

    @property
    def name(self) -> str:
        return "delete_folder"

    @property
    def description(self) -> str:
        return (
            "Delete a folder from the user's library. "
            "Agents inside the folder are moved to root level (not deleted)."
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
                    "description": "ID of the folder to delete.",
                },
            },
            "required": ["folder_id"],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        folder_id = kwargs.get("folder_id", "").strip()
        session_id = session.session_id if session else None

        if not folder_id:
            return ErrorResponse(
                message="Please provide a folder_id.",
                error="missing_folder_id",
                session_id=session_id,
            )

        try:
            await library_db.delete_folder(
                folder_id=folder_id,
                user_id=user_id,
                soft_delete=True,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to delete folder: {e}",
                error="delete_folder_failed",
                session_id=session_id,
            )

        return FolderDeletedResponse(
            message="Folder deleted. Any agents inside were moved to root level.",
            folder_id=folder_id,
            session_id=session_id,
        )


class MoveAgentsToFolderTool(BaseTool):
    """Tool for moving agents into a folder."""

    @property
    def name(self) -> str:
        return "move_agents_to_folder"

    @property
    def description(self) -> str:
        return (
            "Move one or more agents to a folder. "
            "Set folder_id to null to move agents to root level."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of library agent IDs to move.",
                },
                "folder_id": {
                    "type": ["string", "null"],
                    "description": (
                        "Target folder ID. Use null to move to root level."
                    ),
                },
            },
            "required": ["agent_ids"],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        agent_ids = kwargs.get("agent_ids", [])
        folder_id = kwargs.get("folder_id")
        session_id = session.session_id if session else None

        if not agent_ids:
            return ErrorResponse(
                message="Please provide at least one agent ID.",
                error="missing_agent_ids",
                session_id=session_id,
            )

        try:
            await library_db.bulk_move_agents_to_folder(
                agent_ids=agent_ids,
                folder_id=folder_id,
                user_id=user_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to move agents: {e}",
                error="move_agents_failed",
                session_id=session_id,
            )

        return AgentsMovedToFolderResponse(
            message=f"Moved {len(agent_ids)} agent(s) to {'the folder' if folder_id else 'root level'}.",
            agent_ids=agent_ids,
            folder_id=folder_id,
            count=len(agent_ids),
            session_id=session_id,
        )
