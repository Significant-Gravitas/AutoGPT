"""Two-step tool for targeted memory deletion.

Step 1 (memory_forget_search): search for matching facts, return candidates.
Step 2 (memory_forget_confirm): delete specific edges by UUID after user confirms.
"""

import logging
from typing import Any

from backend.copilot.graphiti._format import extract_fact, extract_temporal_validity
from backend.copilot.graphiti.client import derive_group_id, get_graphiti_client
from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import (
    ErrorResponse,
    MemoryForgetCandidatesResponse,
    MemoryForgetConfirmResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


class MemoryForgetSearchTool(BaseTool):
    """Search for memories to forget — returns candidates for user confirmation."""

    @property
    def name(self) -> str:
        return "memory_forget_search"

    @property
    def description(self) -> str:
        return (
            "Search for stored memories matching a description so the user can "
            "choose which to delete. Returns candidate facts with UUIDs. "
            "Use memory_forget_confirm with the UUIDs to actually delete them."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what to forget (e.g. 'the Q2 marketing budget')",
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        query: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                session_id=session.session_id,
            )

        if not await is_enabled_for_user(user_id):
            return ErrorResponse(
                message="Memory features are not enabled for your account.",
                session_id=session.session_id,
            )

        if not query:
            return ErrorResponse(
                message="A search query is required to find memories to forget.",
                session_id=session.session_id,
            )

        try:
            group_id = derive_group_id(user_id)
        except ValueError:
            return ErrorResponse(
                message="Invalid user ID for memory operations.",
                session_id=session.session_id,
            )

        try:
            client = await get_graphiti_client(group_id)
            edges = await client.search(
                query=query,
                group_ids=[group_id],
                num_results=10,
            )
        except Exception:
            logger.warning("Memory forget search failed for user %s", user_id[:12], exc_info=True)
            return ErrorResponse(
                message="Memory search is temporarily unavailable.",
                session_id=session.session_id,
            )

        if not edges:
            return MemoryForgetCandidatesResponse(
                message="No matching memories found.",
                session_id=session.session_id,
                candidates=[],
            )

        candidates = []
        for e in edges:
            edge_uuid = getattr(e, "uuid", None) or getattr(e, "id", None)
            if not edge_uuid:
                continue
            fact = extract_fact(e)
            valid_from, valid_to = extract_temporal_validity(e)
            candidates.append(
                {
                    "uuid": str(edge_uuid),
                    "fact": fact,
                    "valid_from": str(valid_from),
                    "valid_to": str(valid_to),
                }
            )

        return MemoryForgetCandidatesResponse(
            message=f"Found {len(candidates)} candidate(s). Show these to the user and ask which to delete, then call memory_forget_confirm with the UUIDs.",
            session_id=session.session_id,
            candidates=candidates,
        )


class MemoryForgetConfirmTool(BaseTool):
    """Delete specific memory edges by UUID after user confirmation."""

    @property
    def name(self) -> str:
        return "memory_forget_confirm"

    @property
    def description(self) -> str:
        return (
            "Delete specific memories by UUID. Use after memory_forget_search "
            "returns candidates and the user confirms which to delete. "
            "This action is irreversible."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "uuids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of edge UUIDs to delete (from memory_forget_search results)",
                },
            },
            "required": ["uuids"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        uuids: list[str] | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                session_id=session.session_id,
            )

        if not await is_enabled_for_user(user_id):
            return ErrorResponse(
                message="Memory features are not enabled for your account.",
                session_id=session.session_id,
            )

        if not uuids:
            return ErrorResponse(
                message="At least one UUID is required. Use memory_forget_search first.",
                session_id=session.session_id,
            )

        try:
            group_id = derive_group_id(user_id)
        except ValueError:
            return ErrorResponse(
                message="Invalid user ID for memory operations.",
                session_id=session.session_id,
            )

        try:
            client = await get_graphiti_client(group_id)
        except Exception:
            logger.warning("Failed to get Graphiti client for user %s", user_id[:12], exc_info=True)
            return ErrorResponse(
                message="Memory service is temporarily unavailable.",
                session_id=session.session_id,
            )

        deleted = []
        failed = []
        driver = getattr(client, "graph_driver", None) or getattr(client, "driver", None)
        if not driver:
            return ErrorResponse(
                message="Could not access graph driver for deletion.",
                session_id=session.session_id,
            )

        for uuid in uuids:
            try:
                await driver.execute_query(
                    "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() DELETE e",
                    uuid=uuid,
                )
                deleted.append(uuid)
            except Exception:
                logger.warning("Failed to delete edge %s for user %s", uuid, user_id[:12], exc_info=True)
                failed.append(uuid)

        return MemoryForgetConfirmResponse(
            message=f"Deleted {len(deleted)} memory edge(s)." + (f" {len(failed)} failed." if failed else ""),
            session_id=session.session_id,
            deleted_uuids=deleted,
            failed_uuids=failed,
        )
