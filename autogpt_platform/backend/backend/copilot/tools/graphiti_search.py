"""Tool for searching the Graphiti temporal knowledge graph."""

import asyncio
import logging
from typing import Any

from graphiti_core.edges import EntityEdge

from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.graphiti.recall import recent_episodes, search_facts
from backend.copilot.graphiti.recall_recheck import recheck
from backend.copilot.graphiti.recall_render import episode_scope, render, render_episode
from backend.copilot.graphiti.recall_stamp import record_recall
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.model import ChatSession
from backend.util.background import spawn_background_task

from .base import BaseTool
from .models import ErrorResponse, MemorySearchResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_MAX_LIMIT = 50
# Recent raw episodes returned next to the facts.
_RECENT_EPISODES = 5


class MemorySearchTool(BaseTool):
    """Search the current assistant's temporal knowledge graph."""

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search the current assistant's memory graph for facts, preferences, "
            "and context from prior sessions. Use before answering context-dependent "
            "questions or when the user references a past conversation."
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
            memory_scope = MemoryScope.build(user_id, session.expert_id)
        except ValueError:
            return ErrorResponse(
                message="Invalid user ID for memory operations.",
                session_id=session.session_id,
            )

        try:
            edges, episodes = await asyncio.gather(
                search_facts(memory_scope, query, limit=limit),
                recent_episodes(memory_scope, _RECENT_EPISODES),
            )
            # The last read before rendering: a forget that answered while
            # the search ran is not shown (``recall_recheck.py``).
            edges, episodes = await recheck(memory_scope, edges, episodes)
        except Exception:
            logger.warning(
                "Memory search failed for user %s", user_id[:12], exc_info=True
            )
            return ErrorResponse(
                message="Memory search is temporarily unavailable.",
                session_id=session.session_id,
            )

        _count_hits(memory_scope, edges)
        facts = [render(edge) for edge in edges]
        # Scope hard-filter: when a scope is requested, drop episodes whose
        # MemoryEnvelope names a different one (plain conversation counts as
        # ``real:global``).
        recent = [
            render_episode(ep)
            for ep in episodes
            if not scope or episode_scope(ep) == scope
        ]

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


def _count_hits(memory_scope: MemoryScope, edges: list[EntityEdge]) -> None:
    """Count the returned facts as used, so a tentative one can be ratified,
    and stamp the recall on each (``recall_stamp.record_recall``).

    Detached, like warm context's hit hook: the answer never waits on Redis
    or on the stamp.
    """
    if not edges:
        return
    spawn_background_task(
        record_recall(memory_scope, [edge.uuid for edge in edges]),
        name=f"memory-search-hits-{memory_scope.owner_user_id[:12]}",
    )
