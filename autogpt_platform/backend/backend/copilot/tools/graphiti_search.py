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
from backend.copilot.graphiti.client import get_graphiti_client
from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.graphiti.tiers import (
    MemoryTier,
    TierError,
    merge_tiered,
    resolve_search_targets,
)
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
                "tier": {
                    "type": "string",
                    "enum": ["all", "personal", "team", "org"],
                    "description": (
                        "Tiers to search: 'all' (default: personal + org + your "
                        "teams), or 'personal'/'team'/'org'. Shared results are "
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

        # Resolve which tier groups this search may read. Only ACTIVE team
        # memberships are ever included, and org/team tiers require an org
        # on the session — non-members can never read shared memory.
        try:
            targets = await resolve_search_targets(
                user_id, session.organization_id, tier
            )
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

        now = datetime.now(timezone.utc)

        async def _search_one(target):
            client = await get_graphiti_client(target.group_id)
            return await asyncio.gather(
                client.search(
                    query=query,
                    group_ids=[target.group_id],
                    num_results=limit,
                ),
                client.retrieve_episodes(
                    reference_time=now,
                    group_ids=[target.group_id],
                    last_n=5,
                ),
            )

        # Per-tier failures are non-fatal — a flaky shared graph must not
        # sink a personal search. Only surface "unavailable" when every
        # tier failed.
        results = await asyncio.gather(
            *(_search_one(t) for t in targets), return_exceptions=True
        )

        personal_edges: list = []
        personal_eps: list[str] = []
        shared_edges: list[tuple[str | None, list]] = []
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
        facts = [
            f"{_label_prefix(label)}{_edge_to_fact_line(e)}"
            for e, label in merged_facts
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
        return MemorySearchResponse(
            message=(
                f"Found {len(facts)} relationship facts and {len(recent)} stored memories{scope_note}{tier_note}. "
                "Use BOTH sections to answer — stored memories often contain operational "
                "rules and instructions that relationship facts summarize. Facts labelled "
                "'org memory' / 'team memory (<name>)' are shared context — weigh them "
                "against your personal (unlabelled) memory."
            ),
            session_id=session.session_id,
            facts=facts,
            recent_episodes=recent,
        )


def _label_prefix(label: str | None) -> str:
    """Render a provenance label as a bracketed prefix (empty for personal)."""
    return f"[{label}] " if label else ""


def _edge_to_fact_line(e) -> str:
    fact = extract_fact(e)
    valid_from, valid_to = extract_temporal_validity(e)
    return f"{fact} (valid: {valid_from} — {valid_to})"


def _episode_lines(episodes, scope: str) -> list[str]:
    """Format (and, when a scope is requested, hard-filter) recent episodes."""
    if scope:
        return _filter_episodes_by_scope(episodes, scope)
    return _format_episodes(episodes)


def _format_edges(edges) -> list[str]:
    return [_edge_to_fact_line(e) for e in edges]


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
