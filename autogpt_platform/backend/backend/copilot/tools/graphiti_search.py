"""Tool for searching the Graphiti temporal knowledge graph."""

import asyncio
import logging
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode

from backend.copilot.graphiti.client import get_graphiti_client
from backend.copilot.graphiti.lifecycle import (
    active_shared_search_filter,
    filter_active_shared_edges,
    filter_edges_by_scope,
)
from backend.copilot.graphiti.recall import recent_episodes, search_facts
from backend.copilot.graphiti.recall_recheck import recheck
from backend.copilot.graphiti.recall_render import episode_scope, render, render_episode
from backend.copilot.graphiti.recall_stamp import record_recall
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.graphiti.scope_resolution import resolve_memory_scope
from backend.copilot.graphiti.tiers import (
    MemoryTier,
    TierError,
    TierTarget,
    merge_tiered,
    resolve_search_targets,
)
from backend.copilot.model import ChatSession
from backend.data.tenancy import ResourceAccess
from backend.util.background import spawn_background_task

from .base import BaseTool
from .models import ErrorResponse, MemorySearchResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_MAX_LIMIT = 50
# Recent raw episodes returned next to the facts.
_RECENT_EPISODES = 5


def _search_config(limit: int):
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

    return EDGE_HYBRID_SEARCH_RRF.model_copy(update={"limit": limit}, deep=True)


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
                "tier": {
                    "type": "string",
                    "enum": ["all", "personal", "team", "org"],
                    "description": (
                        "Tiers to search: 'all' (default: personal + org + the "
                        "current workspace), or 'personal'/'team'/'org'. Shared results are "
                        "labelled with their source (e.g. 'org memory')."
                    ),
                    "default": "all",
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def resource_access(self) -> ResourceAccess:
        return "view"

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        query: str = "",
        limit: int = 15,
        scope: str = "",
        tier: str = "all",
        **kwargs,
    ) -> ToolResponseBase:
        if not user_id:
            return ErrorResponse(
                message="Authentication required to search memories.",
                session_id=session.session_id,
            )

        if not query:
            return ErrorResponse(
                message="A search query is required.",
                session_id=session.session_id,
            )

        limit = min(limit, _MAX_LIMIT)

        # Resolve which tier groups this search may read. Only ACTIVE team
        # memberships are ever included, and org/team tiers require an org
        # on the session — non-members can never read shared memory.
        try:
            targets = await resolve_search_targets(
                user_id,
                session.organization_id,
                tier,
                session_team_id=session.team_id,
                expert_id=session.expert_id,
            )
            personal_scope = await _personal_scope(user_id, session, targets)
        except TierError as exc:
            return ErrorResponse(message=exc.message, session_id=session.session_id)
        except ValueError:
            return ErrorResponse(
                message="Invalid user ID for memory operations.",
                session_id=session.session_id,
            )

        if not targets:
            # e.g. tier='team' when the user has no active team memberships.
            return MemorySearchResponse(
                message="No memories found matching your query.",
                session_id=session.session_id,
                facts=[],
                recent_episodes=[],
            )

        # Per-tier failures are non-fatal — a flaky shared graph must not
        # sink a personal search. Only surface "unavailable" when every
        # tier failed.
        results = await asyncio.gather(
            *(
                (
                    _search_personal(personal_scope, query, limit, scope)
                    if personal_scope is not None and target.tier == MemoryTier.personal
                    else _search_shared(target, query, scope)
                )
                for target in targets
            ),
            return_exceptions=True,
        )

        personal_edges: list[EntityEdge] = []
        personal_eps: list[str] = []
        shared_edges: list[tuple[str | None, list[EntityEdge]]] = []
        shared_eps: list[tuple[str | None, list[str]]] = []
        any_ok = False

        for target, res in zip(targets, results):
            if isinstance(res, BaseException):
                logger.warning(
                    "Memory search tier %s failed for user %s",
                    target.group_id[:20],
                    user_id[:12],
                    exc_info=res,
                )
                continue
            any_ok = True
            edges, episodes = res
            ep_lines = _episode_lines(episodes, scope)
            if target.tier == MemoryTier.personal:
                personal_edges = edges
                personal_eps = ep_lines
            else:
                shared_edges.append((target.label, edges))
                shared_eps.append((target.label, ep_lines))

        if not any_ok:
            return ErrorResponse(
                message="Memory search is temporarily unavailable.",
                session_id=session.session_id,
            )

        merged_facts = merge_tiered(personal_edges, shared_edges, limit)
        if personal_scope is not None:
            # Only the personal facts shown count as recalled: they live in
            # the personal graph, and shared-tier ratification is an admin's.
            _count_hits(
                personal_scope, [edge for edge, label in merged_facts if label is None]
            )
        facts = [
            f"{_label_prefix(label)}{render(edge)}" for edge, label in merged_facts
        ]

        merged_eps = merge_tiered(personal_eps, shared_eps, limit)
        recent = [f"{_label_prefix(label)}{line}" for line, label in merged_eps]

        if not facts and not recent:
            return MemorySearchResponse(
                message="No memories found matching your query.",
                session_id=session.session_id,
                facts=[],
                recent_episodes=[],
            )

        scope_note = f" (scope filter: {scope})" if scope else ""
        tier_note = "" if tier in ("", "all") else f" (tier: {tier})"
        shared_note = (
            " Facts labelled 'org memory' / 'team memory (<name>)' are shared "
            "context — weigh them against your personal (unlabelled) memory."
            if any(label for _, label in [*merged_facts, *merged_eps])
            else ""
        )
        return MemorySearchResponse(
            message=(
                f"Found {len(facts)} relationship facts and {len(recent)} stored memories{scope_note}{tier_note}. "
                "Use BOTH sections to answer — stored memories often contain operational "
                "rules and instructions that relationship facts summarize."
                f"{shared_note}"
            ),
            session_id=session.session_id,
            facts=facts,
            recent_episodes=recent,
        )


async def _personal_scope(
    user_id: str, session: ChatSession, targets: list[TierTarget]
) -> MemoryScope | None:
    """The scope the personal tier is searched, rechecked and stamped in;
    None when the search reads no personal tier."""
    if not any(target.tier == MemoryTier.personal for target in targets):
        return None
    return await resolve_memory_scope(
        user_id, session.expert_id, session.organization_id
    )


async def _search_personal(
    memory_scope: MemoryScope, query: str, limit: int, scope: str
) -> tuple[list[EntityEdge], list[EpisodicNode]]:
    """Personal facts and recent episodes, read through the recall policy."""
    edges, episodes = await asyncio.gather(
        search_facts(memory_scope, query, limit=limit),
        recent_episodes(memory_scope, _RECENT_EPISODES),
    )
    if scope and edges:
        client = await get_graphiti_client(memory_scope.group_id)
        edges = await filter_edges_by_scope(
            memory_scope.group_id, edges, scope, driver=client.driver
        )
    # The last read before rendering: a forget that answered while the
    # search ran is not shown (``recall_recheck.py``).
    return await recheck(memory_scope, edges, episodes)


async def _search_shared(
    target: TierTarget, query: str, scope: str
) -> tuple[list[EntityEdge], list[EpisodicNode]]:
    """A shared tier's ratified, unretracted facts; shared tiers keep no
    episodes for search."""
    client = await get_graphiti_client(target.group_id)
    edge_results = await client.search_(
        query=query,
        config=_search_config(_MAX_LIMIT),
        group_ids=[target.group_id],
        search_filter=active_shared_search_filter(),
    )
    edges = edge_results.edges if edge_results is not None else []
    active_edges = await filter_active_shared_edges(
        target.group_id, edges, driver=client.driver
    )
    if scope:
        active_edges = await filter_edges_by_scope(
            target.group_id, active_edges, scope, driver=client.driver
        )
    return active_edges, []


def _label_prefix(label: str | None) -> str:
    """Render a provenance label as a bracketed prefix (empty for personal)."""
    return f"[{label}] " if label else ""


def _episode_lines(episodes: list[EpisodicNode], scope: str) -> list[str]:
    """Render recent episodes. Scope hard-filter: when a scope is requested,
    drop episodes whose MemoryEnvelope names a different one (plain
    conversation counts as ``real:global``)."""
    return [
        render_episode(ep) for ep in episodes if not scope or episode_scope(ep) == scope
    ]


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
