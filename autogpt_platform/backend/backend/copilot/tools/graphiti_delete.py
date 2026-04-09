"""Tool for deleting all Graphiti memories for a user."""

import logging
from typing import Any

from backend.copilot.graphiti.client import (
    derive_group_id,
    evict_client,
    get_graphiti_client,
)
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ErrorResponse, GraphitiDeleteResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class GraphitiDeleteTool(BaseTool):
    """Delete all stored memories for the current user."""

    @property
    def name(self) -> str:
        return "graphiti_delete_user_data"

    @property
    def description(self) -> str:
        return (
            "Delete ALL stored memories for the current user. "
            "This is irreversible and removes all entities, relationships, "
            "facts, and episodes from the knowledge graph."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to proceed with deletion",
                },
            },
            "required": ["confirm"],
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
        if not user_id:
            return ErrorResponse(
                message="Authentication required to delete memories.",
                session_id=session.session_id,
            )

        from backend.copilot.graphiti.config import is_enabled_for_user

        if not await is_enabled_for_user(user_id):
            return ErrorResponse(
                message="Memory features are not enabled for your account.",
                session_id=session.session_id,
            )

        confirm = kwargs.get("confirm", False)
        if not confirm:
            return ErrorResponse(
                message="You must set confirm=true to delete all memories. This action is irreversible.",
                session_id=session.session_id,
            )

        from graphiti_core.utils.maintenance.graph_data_operations import clear_data

        group_id = derive_group_id(user_id)
        client = await get_graphiti_client(group_id)

        await clear_data(client.driver, group_ids=[group_id])

        evict_client(group_id)

        return GraphitiDeleteResponse(
            message="All memories have been permanently deleted.",
            session_id=session.session_id,
        )
