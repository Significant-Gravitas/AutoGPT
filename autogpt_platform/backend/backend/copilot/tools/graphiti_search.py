"""Tool for searching the Graphiti temporal knowledge graph."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from backend.copilot.graphiti._format import (
    extract_episode_body,
    extract_episode_body_raw,
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
                "scope": {
                    "type": "string",
                    "description": (
                        "Optional scope filter. When set, only memories matching "
                        "this scope are returned (hard filter). "
                        "Examples: 'real:global', 'project:crm', 'book:my-novel'. "
                        "Omit to search all scopes."
                    ),
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
        scope: str = "",
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

        # Scope hard-filter: if a scope was requested, filter episodes
        # whose MemoryEnvelope JSON contains a different scope.
        # Skip redundant _format_episodes() when scope is set.
        if scope:
            recent = _filter_episodes_by_scope(episodes, scope)
        else:
            recent = _format_episodes(episodes)

        if not facts and not recent:
            return MemorySearchResponse(
                message="No memories found matching your query.",
                session_id=session.session_id,
                facts=[],
                recent_episodes=[],
            )

        scope_note = f" (scope filter: {scope})" if scope else ""
        return MemorySearchResponse(
            message=(
                f"Found {len(facts)} relationship facts and {len(recent)} stored memories{scope_note}. "
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


def _filter_episodes_by_scope(episodes, scope: str) -> list[str]:
    """Filter episodes by scope — hard filter on MemoryEnvelope JSON content.

    Episodes that are plain conversation text (not JSON envelopes) are
    included by default since they have no scope metadata and belong
    to the implicit ``real:global`` scope.

    Uses ``extract_episode_body_raw`` (no truncation) for JSON parsing
    so that long MemoryEnvelope payloads are parsed correctly.
    """
    import json

    results = []
    for ep in episodes:
        raw_body = extract_episode_body_raw(ep)
        try:
            data = json.loads(raw_body)
            if not isinstance(data, dict):
                raise TypeError("non-dict JSON")
            ep_scope = data.get("scope", "real:global")
            if ep_scope != scope:
                continue
        except (json.JSONDecodeError, TypeError):
            # Not JSON or non-dict JSON — plain conversation episode, treat as real:global
            if scope != "real:global":
                continue
        display_body = extract_episode_body(ep)
        ts = extract_episode_timestamp(ep)
        results.append(f"[{ts}] {display_body}")
    return results
