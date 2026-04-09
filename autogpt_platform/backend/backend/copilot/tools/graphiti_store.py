"""Tool for storing memories in the Graphiti temporal knowledge graph."""

import logging
from typing import Any

from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.graphiti.ingest import enqueue_episode
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ErrorResponse, MemoryStoreResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class MemoryStoreTool(BaseTool):
    """Store a memory/fact in the user's temporal knowledge graph."""

    @property
    def name(self) -> str:
        return "memory_store"

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
        *,
        name: str = "",
        content: str = "",
        source_description: str = "Conversation memory",
        **kwargs,
    ) -> ToolResponseBase:
        if not user_id:
            return ErrorResponse(
                message="Authentication required to store memories.",
                session_id=session.session_id,
            )

        if not await is_enabled_for_user(user_id):
            return ErrorResponse(
                message="Memory features are not enabled for your account.",
                session_id=session.session_id,
            )

        if not name or not content:
            return ErrorResponse(
                message="Both 'name' and 'content' are required.",
                session_id=session.session_id,
            )

        queued = await enqueue_episode(
            user_id,
            session.session_id,
            name=name,
            episode_body=content,
            source_description=source_description,
        )

        if not queued:
            return ErrorResponse(
                message="Memory queue is full — please try again shortly.",
                session_id=session.session_id,
            )

        return MemoryStoreResponse(
            message=f"Memory '{name}' queued for storage.",
            session_id=session.session_id,
            memory_name=name,
        )
