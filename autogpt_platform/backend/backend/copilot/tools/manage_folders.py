"""Folder management tools for the copilot."""

from typing import Any

from backend.api.features.library import model as library_model
from backend.api.features.library.db import collect_tree_ids
from backend.copilot.model import ChatSession
from backend.data.db_accessors import library_db

from .base import BaseTool
from .models import (
    AgentsMovedToFolderResponse,
    ErrorResponse,
    FolderAgentSummary,
    FolderCreatedResponse,
    FolderDeletedResponse,
    FolderInfo,
    FolderListResponse,
    FolderMovedResponse,
    FolderTreeInfo,
    FolderUpdatedResponse,
    ToolResponseBase,
)


def _folder_to_info(
    folder: library_model.LibraryFolder,
    agents: list[FolderAgentSummary] | None = None,
) -> FolderInfo:
    """Convert a LibraryFolder DB model to a FolderInfo response model."""
    return FolderInfo(
        id=folder.id,
        name=folder.name,
        parent_id=folder.parent_id,
        icon=folder.icon,
        color=folder.color,
        agent_count=folder.agent_count,
        subfolder_count=folder.subfolder_count,
        agents=agents,
    )


def _tree_to_info(
    tree: library_model.LibraryFolderTree,
    agents_map: dict[str, list[FolderAgentSummary]] | None = None,
) -> FolderTreeInfo:
    """Recursively convert a LibraryFolderTree to a FolderTreeInfo response."""
    return FolderTreeInfo(
        id=tree.id,
        name=tree.name,
        parent_id=tree.parent_id,
        icon=tree.icon,
        color=tree.color,
        agent_count=tree.agent_count,
        subfolder_count=tree.subfolder_count,
        children=[_tree_to_info(child, agents_map) for child in tree.children],
        agents=agents_map.get(tree.id) if agents_map else None,
    )


def _to_agent_summaries(
    raw: list[dict[str, str | None]],
) -> list[FolderAgentSummary]:
    """Convert raw agent dicts to typed FolderAgentSummary models."""
    return [
        FolderAgentSummary(
            id=a["id"] or "",
            name=a["name"] or "",
            description=a["description"] or "",
        )
        for a in raw
    ]


def _to_agent_summaries_map(
    raw: dict[str, list[dict[str, str | None]]],
) -> dict[str, list[FolderAgentSummary]]:
    """Convert a folder-id-keyed dict of raw agents to typed summaries."""
    return {fid: _to_agent_summaries(agents) for fid, agents in raw.items()}


class CreateFolderTool(BaseTool):
    """Tool for creating a library folder."""

    @property
    def name(self) -> str:
        return "create_folder"

    @property
    def description(self) -> str:
        return "Create a library folder. Use parent_id to nest inside another folder."

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
                    "description": "Folder name (max 100 chars).",
                },
                "parent_id": {
                    "type": "string",
                    "description": "Parent folder ID (omit for root).",
                },
                "icon": {
                    "type": "string",
                    "description": "Icon identifier.",
                },
                "color": {
                    "type": "string",
                    "description": "Hex color (#RRGGBB).",
                },
            },
            "required": ["name"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        name: str = "",
        parent_id: str | None = None,
        icon: str | None = None,
        color: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        """Create a folder with the given name and optional parent/icon/color."""
        assert user_id is not None  # guaranteed by requires_auth
        name = (name or "").strip()
        session_id = session.session_id if session else None

        if not name:
            return ErrorResponse(
                message="Please provide a folder name.",
                error="missing_name",
                session_id=session_id,
            )

        try:
            folder = await library_db().create_folder(
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
            "List library folders. Omit parent_id for full tree. "
            "Set include_agents=true when user asks about agents, wants to see "
            "what's in their folders, or mentions agents alongside folders."
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
                    "description": "List children of this folder (omit for full tree).",
                },
                "include_agents": {
                    "type": "boolean",
                    "description": "Include agents in each folder (default: false).",
                },
            },
            "required": [],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        parent_id: str | None = None,
        include_agents: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        """List folders as a flat list (by parent) or full tree."""
        assert user_id is not None  # guaranteed by requires_auth
        session_id = session.session_id if session else None

        try:
            if parent_id:
                folders = await library_db().list_folders(
                    user_id=user_id, parent_id=parent_id
                )
                raw_map = (
                    await library_db().get_folder_agents_map(
                        user_id, [f.id for f in folders]
                    )
                    if include_agents
                    else None
                )
                agents_map = _to_agent_summaries_map(raw_map) if raw_map else None
                return FolderListResponse(
                    message=f"Found {len(folders)} folder(s).",
                    folders=[
                        _folder_to_info(f, agents_map.get(f.id) if agents_map else None)
                        for f in folders
                    ],
                    count=len(folders),
                    session_id=session_id,
                )
            else:
                tree = await library_db().get_folder_tree(user_id=user_id)
                all_ids = collect_tree_ids(tree)
                agents_map = None
                root_agents = None
                if include_agents:
                    raw_map = await library_db().get_folder_agents_map(user_id, all_ids)
                    agents_map = _to_agent_summaries_map(raw_map)
                    root_agents = _to_agent_summaries(
                        await library_db().get_root_agent_summaries(user_id)
                    )
                return FolderListResponse(
                    message=f"Found {len(all_ids)} folder(s) in your library.",
                    tree=[_tree_to_info(t, agents_map) for t in tree],
                    root_agents=root_agents,
                    count=len(all_ids),
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
        self,
        user_id: str | None,
        session: ChatSession,
        folder_id: str = "",
        name: str | None = None,
        icon: str | None = None,
        color: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        """Update a folder's name, icon, or color."""
        assert user_id is not None  # guaranteed by requires_auth
        folder_id = (folder_id or "").strip()
        session_id = session.session_id if session else None

        if not folder_id:
            return ErrorResponse(
                message="Please provide a folder_id.",
                error="missing_folder_id",
                session_id=session_id,
            )

        try:
            folder = await library_db().update_folder(
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
        return "Move a folder. Set target_parent_id to null for root."

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
                },
                "target_parent_id": {
                    "type": ["string", "null"],
                    "description": "New parent folder ID (null for root).",
                },
            },
            "required": ["folder_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        folder_id: str = "",
        target_parent_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        """Move a folder to a new parent or to root level."""
        assert user_id is not None  # guaranteed by requires_auth
        folder_id = (folder_id or "").strip()
        session_id = session.session_id if session else None

        if not folder_id:
            return ErrorResponse(
                message="Please provide a folder_id.",
                error="missing_folder_id",
                session_id=session_id,
            )

        try:
            folder = await library_db().move_folder(
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

        dest = "a subfolder" if target_parent_id else "root level"
        return FolderMovedResponse(
            message=f"Folder '{folder.name}' moved to {dest}.",
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
        return "Delete a folder. Agents inside move to root (not deleted)."

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
        self,
        user_id: str | None,
        session: ChatSession,
        folder_id: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        """Soft-delete a folder; agents inside are moved to root level."""
        assert user_id is not None  # guaranteed by requires_auth
        folder_id = (folder_id or "").strip()
        session_id = session.session_id if session else None

        if not folder_id:
            return ErrorResponse(
                message="Please provide a folder_id.",
                error="missing_folder_id",
                session_id=session_id,
            )

        try:
            await library_db().delete_folder(
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
        return "Move agents to a folder. Set folder_id to null for root."

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
                    "description": "Library agent IDs to move.",
                },
                "folder_id": {
                    "type": ["string", "null"],
                    "description": "Target folder ID (null for root).",
                },
            },
            "required": ["agent_ids"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        agent_ids: list[str] | None = None,
        folder_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        """Move one or more agents to a folder or to root level."""
        assert user_id is not None  # guaranteed by requires_auth
        if agent_ids is None:
            agent_ids = []
        session_id = session.session_id if session else None

        if not agent_ids:
            return ErrorResponse(
                message="Please provide at least one agent ID.",
                error="missing_agent_ids",
                session_id=session_id,
            )

        try:
            moved = await library_db().bulk_move_agents_to_folder(
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

        moved_ids = [a.id for a in moved]
        agent_names = [a.name for a in moved]
        dest = "the folder" if folder_id else "root level"
        names_str = (
            ", ".join(agent_names) if agent_names else f"{len(agent_ids)} agent(s)"
        )
        return AgentsMovedToFolderResponse(
            message=f"Moved {names_str} to {dest}.",
            agent_ids=moved_ids,
            agent_names=agent_names,
            folder_id=folder_id,
            count=len(moved),
            session_id=session_id,
        )
