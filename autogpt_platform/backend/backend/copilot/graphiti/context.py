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
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ._format import (
    extract_episode_body,
    extract_episode_body_raw,
    extract_episode_timestamp,
    extract_fact,
    extract_temporal_validity,
)
from .client import derive_group_id, get_graphiti_client
from .config import graphiti_config

if TYPE_CHECKING:
    from graphiti_core.search.search_config import SearchConfig

logger = logging.getLogger(__name__)

# Minimum "signal unit" count for a follow-up user message to trigger a
# warm-context refresh.  Short acknowledgements ("ok", "thanks", "yes go
# ahead") carry no new retrieval signal, so refreshing on them would waste a
# graph search + embedding call every turn.  A post-compaction turn bypasses
# this gate via ``refresh_warm_context(..., force=True)``.
WARM_CONTEXT_REFRESH_MIN_WORDS = 4


def should_refresh_warm_context(message: str | None) -> bool:
    """Whether a follow-up user message carries enough signal to re-fetch.

    Pure, deterministic cost gate: only messages with at least
    ``WARM_CONTEXT_REFRESH_MIN_WORDS`` signal units (whitespace words plus
    individually-counted CJK characters) re-run retrieval, keeping the added
    per-turn graph-search cost off trivial acknowledgement turns while still
    firing for whitespace-less languages.
    """
    if not message:
        return False
    return _signal_units(message) >= WARM_CONTEXT_REFRESH_MIN_WORDS


def _signal_units(message: str) -> int:
    """Count retrieval "signal units" in *message* for the substance gate.

    A plain ``str.split()`` word count under-counts languages that don't
    separate words with whitespace (Japanese, Chinese, Thai) — a long CJK
    message would score 1 "word" and never pass the gate, silently disabling
    the refresh for those users.  So each CJK/ideographic character counts as
    its own unit and is added to the whitespace-word count of the rest.
    """
    cjk = sum(1 for ch in message if _is_ideographic(ch))
    # Words made of non-CJK runs; CJK chars are counted individually above, so
    # exclude them from the whitespace-split count to avoid double counting.
    non_cjk = "".join(" " if _is_ideographic(ch) else ch for ch in message)
    return cjk + len(non_cjk.split())


def _is_ideographic(ch: str) -> bool:
    # CJK Unified Ideographs, Hiragana, Katakana, Hangul, Thai — scripts whose
    # tokens are not whitespace-delimited.  Range checks only, no deps.
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana + Katakana
        or 0x3400 <= code <= 0x4DBF  # CJK Ext A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified
        or 0xAC00 <= code <= 0xD7A3  # Hangul syllables
        or 0x0E00 <= code <= 0x0E7F  # Thai
    )


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

    Cost is bounded three ways: ``should_refresh_warm_context`` skips trivial
    turns (unless ``force`` — e.g. just after a compaction, where the current
    message may be short); the fetch runs the RRF recipe
    (``use_cross_encoder=False``) — graph search + embeddings only, no
    per-candidate cross-encoder LLM prompts; and it uses the shorter
    ``context_refresh_timeout`` budget. Unlike the first-turn fetch (which runs
    concurrently inside a gather), this refresh is a serial ``await`` on the
    pre-stream hot path, so the tighter budget caps its worst-case
    time-to-first-token hit on a cold graph.

    Returns the ``<temporal_context>`` block, or ``None`` when skipped/empty.
    """
    if not user_id:
        return None
    if not force and not should_refresh_warm_context(message):
        return None
    return await fetch_warm_context(
        user_id,
        message or "",
        use_cross_encoder=False,
        timeout=graphiti_config.context_refresh_timeout,
    )


async def fetch_warm_context(
    user_id: str,
    message: str,
    *,
    use_cross_encoder: bool = True,
    timeout: float | None = None,
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

    effective_timeout = (
        timeout if timeout is not None else graphiti_config.context_timeout
    )
    try:
        return await asyncio.wait_for(
            _fetch(user_id, message, use_cross_encoder=use_cross_encoder),
            timeout=effective_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Graphiti warm context timed out after %.1fs",
            effective_timeout,
        )
        return None
    except Exception:
        logger.warning("Graphiti warm context fetch failed", exc_info=True)
        return None


def _build_search_config(use_cross_encoder: bool) -> "SearchConfig":
    """Edge-search recipe for a warm-context fetch.

    Both variants use the SAME search methods as graphiti's
    ``EDGE_HYBRID_SEARCH_CROSS_ENCODER`` recipe — BM25 + cosine + BFS graph
    traversal — so recall *breadth* is identical on the first turn and on
    follow-up refreshes. They differ only in the reranker:

    - ``use_cross_encoder=True`` (first turn): the cross-encoder recipe as-is —
      a per-candidate classifier LLM rerank for the ~10–15% precision lift the
      audit measured on the single most-impactful retrieval per session.
    - ``use_cross_encoder=False`` (SECRT-2378 follow-up refresh): the same
      recipe with the reranker swapped to reciprocal-rank-fusion. No LLM calls,
      so re-running recall every substantive turn stays cheap — but it keeps
      the BFS method, so a fact reachable only by graph expansion from an
      entity named in the message is still surfaced on follow-up turns.

    The recipe defaults ``limit=10``; both are overridden to the configured
    ``context_max_facts`` so existing operator tuning still applies.
    """
    # Imported lazily so the module can be imported without graphiti-core
    # installed (matches the pattern in client.py).
    from graphiti_core.search.search_config import EdgeReranker
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    )

    limit = graphiti_config.context_max_facts
    if use_cross_encoder:
        return EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(update={"limit": limit})
    base_edge_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.edge_config
    if base_edge_config is None:
        # The recipe always carries an edge_config; this satisfies the type
        # checker and fails loudly if graphiti ever ships a broken recipe.
        raise RuntimeError("EDGE_HYBRID_SEARCH_CROSS_ENCODER has no edge_config")
    edge_config = base_edge_config.model_copy(update={"reranker": EdgeReranker.rrf})
    return EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(
        update={"limit": limit, "edge_config": edge_config}
    )


async def _fetch(
    user_id: str, message: str, *, use_cross_encoder: bool = True
) -> str | None:
    search_config = _build_search_config(use_cross_encoder)

    group_id = derive_group_id(user_id)
    client = await get_graphiti_client(group_id)

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

    # Ratification sync hit-hook (P0.4 layer-2): every retrieved edge that's
    # currently ``status='tentative'`` gets promoted to ``active`` inline, and
    # every retrieved edge bumps its warm-context hit counter. Fire-and-forget
    # so the chat turn never blocks on Redis or FalkorDB writes.
    #
    # Gated to the cross-encoder (first-turn) path ONLY: those results passed a
    # per-candidate classifier, so promoting a tentative edge on a hit is
    # earned. RRF follow-up refreshes have no classifier and default
    # ``reranker_min_score=0``, so a weak BM25/cosine match could auto-promote
    # an unratified memory — and would do so once per substantive turn rather
    # than once per session. Refreshes still retrieve; they just don't ratify.
    if edges and use_cross_encoder:
        _spawn_ratification_hits(user_id, edges)

    if not edges and not episodes:
        return None

    return _format_context(edges, episodes)


# Strong refs to in-flight hit tasks — the event loop holds only weak
# references, so an unretained fire-and-forget task can be GC'd
# mid-execution and silently drop the hit recording. Same pattern as
# ``backend/data/user.py``'s ``_background_tasks``.
_pending_hit_tasks: set[asyncio.Task] = set()


def _on_hit_task_done(task: asyncio.Task) -> None:
    _pending_hit_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("Ratification hit task %s failed", task.get_name(), exc_info=exc)


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

    task = asyncio.create_task(
        try_ratify_on_hit(user_id, edge_uuids),
        name=f"ratify-hits-{user_id[:12]}",
    )
    _pending_hit_tasks.add(task)
    task.add_done_callback(_on_hit_task_done)


# Retrieved memory is user/tool/web-authored. A fact containing the literal
# ``</temporal_context>`` would close the block early: everything after it
# reads as the user's own words (a self-scoped prompt-injection breakout),
# and the SDK transcript scrub — which matches up to the first closing tag —
# would leave the remainder of the block in the persisted transcript to
# replay on --resume. Neutralising the sequence at build time fixes every
# consumer at once rather than each reader separately.
#
# Matched by pattern, not exact string: an LLM parses XML fuzzily, so
# ``</temporal_context >``, ``</Temporal_Context>`` and ``</ temporal_context>``
# all read as a closing tag to the model even though none of them equals the
# literal. An exact-string replace would neutralise the tidy spelling and let
# every variant through — which is the only spelling an attacker would use.
_CONTEXT_CLOSE_TAG_NEUTRALISED = "<!/temporal_context>"
_CONTEXT_CLOSE_TAG_RE = re.compile(r"<\s*/\s*temporal_context\s*>", re.IGNORECASE)


def _neutralise_context_tags(text: str) -> str:
    return _CONTEXT_CLOSE_TAG_RE.sub(_CONTEXT_CLOSE_TAG_NEUTRALISED, text)


def _format_context(edges, episodes) -> str | None:
    sections: list[str] = []

    if edges:
        fact_lines = []
        for e in edges:
            valid_from, valid_to = extract_temporal_validity(e)
            fact = _neutralise_context_tags(extract_fact(e))
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
            display_body = _neutralise_context_tags(extract_episode_body(ep))
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
