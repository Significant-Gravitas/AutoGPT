"""Tool for searching the Graphiti temporal knowledge graph."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from backend.copilot.graphiti._format import (
    extract_episode_body,
    extract_episode_timestamp,
    extract_fact,
    extract_temporal_validity,
)
from backend.copilot.graphiti.client import derive_group_id, get_graphiti_client
from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ErrorResponse, MemorySearchResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_MAX_LIMIT = 50


class MemorySearchTool(BaseTool):
    """Search the user's temporal knowledge graph for stored memories."""

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search the user's memory graph for facts, preferences, and context "
            "from prior sessions. Use before answering context-dependent questions "
            "or when the user references something from a past conversation."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 15,
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
        limit: int = 15,
        **kwargs,
    ) -> ToolResponseBase:
        if not user_id:
            return ErrorResponse(
                message="Authentication required to search memories.",
                session_id=session.session_id,
            )

        if not await is_enabled_for_user(user_id):
            return ErrorResponse(
                message="Memory features are not enabled for your account.",
                session_id=session.session_id,
            )

        if not query:
            return ErrorResponse(
                message="A search query is required.",
                session_id=session.session_id,
            )

        limit = min(limit, _MAX_LIMIT)

        try:
            group_id = derive_group_id(user_id)
        except ValueError:
            return ErrorResponse(
                message="Invalid user ID for memory operations.",
                session_id=session.session_id,
            )

        try:
            client = await get_graphiti_client(group_id)

            edges, episodes = await asyncio.gather(
                client.search(
                    query=query,
                    group_ids=[group_id],
                    num_results=limit,
                ),
                client.retrieve_episodes(
                    reference_time=datetime.now(timezone.utc),
                    group_ids=[group_id],
                    last_n=5,
                ),
            )
        except Exception:
            logger.warning(
                "Memory search failed for user %s", user_id[:12], exc_info=True
            )
            return ErrorResponse(
                message="Memory search is temporarily unavailable.",
                session_id=session.session_id,
            )

        facts = _format_edges(edges)
        recent = _format_episodes(episodes)

        if not facts and not recent:
            return MemorySearchResponse(
                message="No memories found matching your query.",
                session_id=session.session_id,
                facts=[],
                recent_episodes=[],
            )

        return MemorySearchResponse(
            message=(
                f"Found {len(facts)} relationship facts and {len(recent)} stored memories. "
                "Use BOTH sections to answer — stored memories often contain operational "
                "rules and instructions that relationship facts summarize."
            ),
            session_id=session.session_id,
            facts=facts,
            recent_episodes=recent,
        )


def _format_edges(edges) -> list[str]:
    results = []
    for e in edges:
        fact = extract_fact(e)
        valid_from, valid_to = extract_temporal_validity(e)
        results.append(f"{fact} (valid: {valid_from} — {valid_to})")
    return results


def _format_episodes(episodes) -> list[str]:
    results = []
    for ep in episodes:
        ts = extract_episode_timestamp(ep)
        body = extract_episode_body(ep)
        results.append(f"[{ts}] {body}")
    return results
