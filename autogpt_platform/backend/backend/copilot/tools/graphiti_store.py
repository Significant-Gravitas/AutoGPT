"""Tool for storing memories in the Graphiti temporal knowledge graph."""

import logging
from datetime import datetime, timezone
from typing import Any

from backend.copilot.graphiti.client import derive_group_id, get_graphiti_client
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ErrorResponse, GraphitiStoreResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class GraphitiStoreTool(BaseTool):
    """Store a memory/fact in the user's temporal knowledge graph."""

    @property
    def name(self) -> str:
        return "graphiti_store"

    @property
    def description(self) -> str:
        return (
            "Store a memory or fact about the user for future recall. "
            "Use when the user shares preferences, business context, decisions, "
            "relationships, or other important information worth remembering "
            "across sessions."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Brief descriptive name for this memory (e.g. 'user_prefers_python')",
                },
                "content": {
                    "type": "string",
                    "description": "The information to remember. Be concise but complete.",
                },
                "source_description": {
                    "type": "string",
                    "description": "Context about where this info came from",
                    "default": "Conversation memory",
                },
            },
            "required": ["name", "content"],
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
                message="Authentication required to store memories.",
                session_id=session.session_id,
            )

        from backend.copilot.graphiti.config import is_enabled_for_user

        if not await is_enabled_for_user(user_id):
            return ErrorResponse(
                message="Memory features are not enabled for your account.",
                session_id=session.session_id,
            )

        name = kwargs.get("name", "")
        content = kwargs.get("content", "")
        source_description = kwargs.get("source_description", "Conversation memory")

        if not name or not content:
            return ErrorResponse(
                message="Both 'name' and 'content' are required.",
                session_id=session.session_id,
            )

        from graphiti_core.nodes import EpisodeType

        from backend.copilot.graphiti.ingest import CUSTOM_EXTRACTION_INSTRUCTIONS

        group_id = derive_group_id(user_id)
        client = await get_graphiti_client(group_id)

        await client.add_episode(
            name=name,
            episode_body=content,
            source=EpisodeType.text,
            source_description=source_description,
            reference_time=datetime.now(timezone.utc),
            group_id=group_id,
            custom_extraction_instructions=CUSTOM_EXTRACTION_INSTRUCTIONS,
        )

        return GraphitiStoreResponse(
            message=f"Memory '{name}' stored successfully.",
            session_id=session.session_id,
            memory_name=name,
        )
