"""Warm context retrieval — pre-loads relevant facts at session start."""

import asyncio
import logging
from datetime import datetime, timezone

from ._format import (
    extract_episode_body,
    extract_episode_body_raw,
    extract_episode_timestamp,
    extract_fact,
    extract_temporal_validity,
)
from .client import get_graphiti_client
from .config import graphiti_config
from .tiers import MemoryTier, TierTarget, merge_tiered, resolve_warm_targets

logger = logging.getLogger(__name__)

# Recent-episode budget for warm context, split personal-first across tiers
# (same >= half rule as facts). Kept small so episodes stay a recency hint,
# not the bulk of the injected context.
_WARM_EPISODE_BUDGET = 8


async def fetch_warm_context(
    user_id: str,
    message: str,
    *,
    organization_id: str | None = None,
    team_id: str | None = None,
) -> str | None:
    """Fetch relevant temporal context for the current user and message.

    Called at the start of a session (first turn) to pre-load facts from
    prior conversations. Fans out across the personal tier, the org tier
    (when ``organization_id`` is set), and the SESSION'S team tier (when
    ``team_id`` is set AND the user is an active member) — merging the
    results with provenance labels. Returns a formatted
    ``<temporal_context>`` block, or ``None`` on failure.

    Graceful degradation: any error (timeout, connection, graphiti-core bug)
    returns ``None`` so the copilot continues without temporal context.
    """
    if not user_id:
        return None

    try:
        return await asyncio.wait_for(
            _fetch(user_id, message, organization_id, team_id),
            timeout=graphiti_config.context_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Graphiti warm context timed out after %.1fs",
            graphiti_config.context_timeout,
        )
        return None
    except Exception:
        logger.warning("Graphiti warm context fetch failed", exc_info=True)
        return None


async def _fetch(
    user_id: str,
    message: str,
    organization_id: str | None = None,
    team_id: str | None = None,
) -> str | None:
    # Imported lazily so the module can be imported without graphiti-core
    # installed (matches the pattern in client.py).
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    )

    targets = await resolve_warm_targets(user_id, organization_id, team_id)

    # P-1.4: warm context is the single most-impactful retrieval per
    # session — the one place where the cross-encoder rerank earns its
    # ~10–15% precision lift (per the audit) at the cost of one extra
    # batch of boolean-classifier prompts. The EDGE_HYBRID_SEARCH_CROSS_ENCODER
    # recipe combines BM25 + cosine + BFS edge search with cross-encoder
    # reranking. The recipe defaults ``limit=10``; we override to our
    # configured ``context_max_facts`` so existing operator tuning still
    # applies.
    search_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(
        update={"limit": graphiti_config.context_max_facts}
    )
    now = datetime.now(timezone.utc)

    async def _fetch_one(target: TierTarget):
        client = await get_graphiti_client(target.group_id)
        edge_results, episodes = await asyncio.gather(
            client.search_(
                query=message,
                config=search_config,
                group_ids=[target.group_id],
            ),
            client.retrieve_episodes(
                reference_time=now,
                group_ids=[target.group_id],
                last_n=5,
            ),
        )
        edges = edge_results.edges if edge_results is not None else []
        return edges, episodes

    # Per-tier failures are non-fatal: a flaky org/team graph must not
    # nuke the personal warm context. Collect exceptions and skip that tier.
    results = await asyncio.gather(
        *(_fetch_one(t) for t in targets), return_exceptions=True
    )

    personal_edges: list = []
    personal_eps: list = []
    shared_edges: list[tuple[str | None, list]] = []
    shared_eps: list[tuple[str | None, list]] = []

    for target, res in zip(targets, results):
        if isinstance(res, BaseException):
            logger.warning(
                "Warm context tier %s failed — skipping",
                target.group_id[:20],
                exc_info=res,
            )
            continue
        edges, episodes = res
        if target.tier == MemoryTier.personal:
            personal_edges = edges
            personal_eps = episodes
        else:
            shared_edges.append((target.label, edges))
            shared_eps.append((target.label, episodes))

    # Ratification sync hit-hook (P0.4 layer-2): promotes tentative edges to
    # active inline + bumps warm-context hit counters. It keys off
    # ``derive_group_id(user_id)`` (the PERSONAL graph), so it may only ever
    # see personal edges — shared-tier ratification is an admin concern
    # (out of scope). Fire-and-forget so the chat turn never blocks.
    if personal_edges:
        _spawn_ratification_hits(user_id, personal_edges)

    merged_facts = merge_tiered(
        personal_edges, shared_edges, graphiti_config.context_max_facts
    )
    merged_episodes = merge_tiered(personal_eps, shared_eps, _WARM_EPISODE_BUDGET)

    if not merged_facts and not merged_episodes:
        return None

    return _format_context(merged_facts, merged_episodes)


def _spawn_ratification_hits(user_id: str, edges) -> None:
    """Fire-and-forget the ratification hit-hook for retrieved edges.

    Imports lazily so the dream/ratification module isn't pulled into
    every retrieval boot path; keeps the cold-start cost zero for
    users on the rare GRAPHITI_MEMORY=on / DREAM_PASS_ENABLED=off
    combination.
    """
    edge_uuids = [uuid for uuid in (getattr(e, "uuid", None) for e in edges) if uuid]
    if not edge_uuids:
        return

    from backend.copilot.dream.ratification import try_ratify_on_hit

    asyncio.create_task(
        try_ratify_on_hit(user_id, edge_uuids),
        name=f"ratify-hits-{user_id[:12]}",
    )


def _label_prefix(label: str | None) -> str:
    """Render a provenance label as a bracketed prefix (empty for personal)."""
    return f"[{label}] " if label else ""


def _format_context(labeled_facts, labeled_episodes) -> str | None:
    """Render merged, provenance-labelled facts + episodes.

    ``labeled_facts`` / ``labeled_episodes`` are ``[(item, label)]`` pairs
    where ``label`` is ``None`` for personal (rendered plain) or a shared-
    tier label like ``"org memory"`` / ``"team memory (Platform)"`` that is
    prefixed so the model can weigh provenance.
    """
    sections: list[str] = []

    if labeled_facts:
        fact_lines = []
        for e, label in labeled_facts:
            valid_from, valid_to = extract_temporal_validity(e)
            fact = extract_fact(e)
            fact_lines.append(
                f"  - {_label_prefix(label)}{fact} ({valid_from} — {valid_to})"
            )
        sections.append("<FACTS>\n" + "\n".join(fact_lines) + "\n</FACTS>")

    if labeled_episodes:
        ep_lines = []
        for ep, label in labeled_episodes:
            # Use raw body (no truncation) for scope parsing — truncated
            # JSON from extract_episode_body() would fail json.loads().
            raw_body = extract_episode_body_raw(ep)
            if _is_non_global_scope(raw_body):
                continue
            display_body = extract_episode_body(ep)
            ts = extract_episode_timestamp(ep)
            ep_lines.append(f"  - {_label_prefix(label)}[{ts}] {display_body}")
        if ep_lines:
            sections.append(
                "<RECENT_EPISODES>\n" + "\n".join(ep_lines) + "\n</RECENT_EPISODES>"
            )

    if not sections:
        return None

    body = "\n\n".join(sections)
    return f"<temporal_context>\n{body}\n</temporal_context>"


def _is_non_global_scope(body: str) -> bool:
    """Check if an episode body is a MemoryEnvelope with a non-global scope."""
    import json

    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            return False
        scope = data.get("scope", "real:global")
        return scope != "real:global"
    except (json.JSONDecodeError, TypeError):
        return False
