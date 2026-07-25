"""Warm context retrieval — deterministic memory recall for a chat turn.

The first turn of a session pre-loads relevant facts via
``fetch_warm_context`` (cross-encoder recipe, highest precision).  Later
turns — a new task started mid-session, or the turn right after a context
compaction — refresh via ``refresh_warm_context`` (cheap RRF recipe, gated
on message substance) so recall never depends solely on the model choosing
to call the memory tool.  See SECRT-2378.
"""

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
from .client import derive_group_id, get_graphiti_client
from .config import graphiti_config

logger = logging.getLogger(__name__)

# Minimum word count for a follow-up user message to trigger a warm-context
# refresh.  Short acknowledgements ("ok", "thanks", "yes go ahead") carry no
# new retrieval signal, so refreshing on them would waste a graph search +
# embedding call every turn.  A post-compaction turn bypasses this gate via
# ``refresh_warm_context(..., force=True)``.
WARM_CONTEXT_REFRESH_MIN_WORDS = 4


def should_refresh_warm_context(message: str | None) -> bool:
    """Whether a follow-up user message carries enough signal to re-fetch.

    Pure, deterministic cost gate: only messages with at least
    ``WARM_CONTEXT_REFRESH_MIN_WORDS`` words re-run retrieval, keeping the
    added per-turn graph-search cost off trivial acknowledgement turns.
    """
    if not message:
        return False
    return len(message.split()) >= WARM_CONTEXT_REFRESH_MIN_WORDS


async def refresh_warm_context(
    user_id: str | None,
    message: str | None,
    *,
    force: bool = False,
) -> str | None:
    """Re-fetch warm context on a FOLLOW-UP turn, keyed on the current message.

    The first turn pre-loads memory via ``fetch_warm_context`` (cross-encoder,
    high precision).  Later turns — a new task mid-session, or the turn right
    after a context compaction — otherwise get no deterministic recall and
    depend on the model choosing to call the memory tool, which it often skips
    (SECRT-2378).  This refresh closes that gap.

    Cost is bounded two ways: ``should_refresh_warm_context`` skips trivial
    turns (unless ``force`` — e.g. just after a compaction, where the current
    message may be short), and the fetch runs the RRF recipe
    (``use_cross_encoder=False``) — graph search + embeddings only, no
    per-candidate cross-encoder LLM prompts.

    Returns the ``<temporal_context>`` block, or ``None`` when skipped/empty.
    """
    if not user_id:
        return None
    if not force and not should_refresh_warm_context(message):
        return None
    return await fetch_warm_context(user_id, message or "", use_cross_encoder=False)


async def fetch_warm_context(
    user_id: str, message: str, *, use_cross_encoder: bool = True
) -> str | None:
    """Fetch relevant temporal context for the current user and message.

    Returns a formatted ``<temporal_context>`` block suitable for appending
    to the current turn's user message, or ``None`` on failure/empty.

    ``use_cross_encoder`` selects the search recipe: ``True`` (first turn)
    uses the cross-encoder recipe for maximum precision at the cost of one
    batch of classifier prompts; ``False`` (follow-up refresh) uses the
    cheaper RRF recipe — BM25 + cosine + BFS with no LLM rerank.

    Graceful degradation: any error (timeout, connection, graphiti-core bug)
    returns ``None`` so the copilot continues without temporal context.
    """
    if not user_id:
        return None

    try:
        return await asyncio.wait_for(
            _fetch(user_id, message, use_cross_encoder=use_cross_encoder),
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
    user_id: str, message: str, *, use_cross_encoder: bool = True
) -> str | None:
    # Imported lazily so the module can be imported without graphiti-core
    # installed (matches the pattern in client.py).
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_CROSS_ENCODER,
        EDGE_HYBRID_SEARCH_RRF,
    )

    group_id = derive_group_id(user_id)
    client = await get_graphiti_client(group_id)

    # P-1.4: the first-turn warm context is the single most-impactful
    # retrieval per session — the one place where the cross-encoder rerank
    # earns its ~10–15% precision lift (per the audit) at the cost of one
    # extra batch of boolean-classifier prompts. Follow-up refreshes
    # (SECRT-2378) drop to the RRF recipe (no LLM rerank) so re-running
    # recall every substantive turn stays cheap. Both recipes combine BM25 +
    # cosine + BFS edge search; each defaults ``limit=10``, overridden to our
    # configured ``context_max_facts`` so existing operator tuning still
    # applies.
    recipe = (
        EDGE_HYBRID_SEARCH_CROSS_ENCODER
        if use_cross_encoder
        else EDGE_HYBRID_SEARCH_RRF
    )
    search_config = recipe.model_copy(
        update={"limit": graphiti_config.context_max_facts}
    )
    edge_results, episodes = await asyncio.gather(
        client.search_(
            query=message,
            config=search_config,
            group_ids=[group_id],
        ),
        client.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            group_ids=[group_id],
            last_n=5,
        ),
    )
    edges = edge_results.edges if edge_results is not None else []

    # Ratification sync hit-hook (P0.4 layer-2): every retrieved edge
    # that's currently ``status='tentative'`` gets promoted to
    # ``active`` inline, and every retrieved edge bumps its
    # warm-context hit counter. Fire-and-forget so the chat turn
    # never blocks on Redis or FalkorDB writes.
    if edges:
        _spawn_ratification_hits(user_id, edges)

    if not edges and not episodes:
        return None

    return _format_context(edges, episodes)


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


def _format_context(edges, episodes) -> str | None:
    sections: list[str] = []

    if edges:
        fact_lines = []
        for e in edges:
            valid_from, valid_to = extract_temporal_validity(e)
            fact = extract_fact(e)
            fact_lines.append(f"  - {fact} ({valid_from} — {valid_to})")
        sections.append("<FACTS>\n" + "\n".join(fact_lines) + "\n</FACTS>")

    if episodes:
        ep_lines = []
        for ep in episodes:
            # Use raw body (no truncation) for scope parsing — truncated
            # JSON from extract_episode_body() would fail json.loads().
            raw_body = extract_episode_body_raw(ep)
            if _is_non_global_scope(raw_body):
                continue
            display_body = extract_episode_body(ep)
            ts = extract_episode_timestamp(ep)
            ep_lines.append(f"  - [{ts}] {display_body}")
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
