"""Warm context: deterministic memory recall written into a chat turn.

Recall must not depend on the model choosing to call ``memory_search``
(SECRT-2378), so the chat engines put a ``<temporal_context>`` block, keyed
on the user's message, into the turn themselves:

- The first turn of a session calls ``fetch_warm_context``. Its ranking is
  unchanged by the follow-up refresh: graphiti's cross-encoder recipe (BM25,
  cosine and BFS edge search, then a per-candidate LLM rerank),
  ``context_max_facts`` facts, the five newest recallable episodes,
  ``context_timeout``, and a ratification hit for every fact shown.
- Every later user turn calls ``refresh_warm_context`` with that turn's
  message. It fetches when the message carries at least
  ``WARM_CONTEXT_REFRESH_MIN_WORDS`` signal units
  (``should_refresh_warm_context``), or when the SDK engine forces it because
  it compacted the history for the query it is sending (the initial query or
  a context-overflow retry; the baseline engine cannot tell and never
  forces); otherwise it returns ``None`` without touching the graph. A fetch
  is one retrieval: the same search methods reranked with reciprocal rank
  fusion (one query embedding, no LLM call), the recent-episode read and the
  last check below, bounded by ``context_refresh_timeout``, recording no
  ratification hits. An error or a timeout returns ``None`` and the turn
  goes ahead without a refresh. The engines decide which turns are
  follow-ups and where the block goes: ``sdk/service.py``
  (``_start_follow_up_warm_context``, ``_append_follow_up_warm_context``) and
  ``baseline/service.py`` (``_refresh_follow_up_warm_context``).

Both read through the recall policy (``recall.py``) the same way: live facts
only, recallable episodes only, one last check of both by uuid right before
rendering (``recall_recheck.recheck``), written out by ``recall_render.py``.
A fact forgotten between two turns, and every episode it came from, is
therefore not in the next turn's refresh. Every tag start in the rendered
memory is neutralised, after truncation, so stored text can neither open,
close nor complete the block's delimiters (``recall_render.neutralise_tags``).
"""

import asyncio
import logging

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode
from graphiti_core.search.search_config import EdgeReranker, SearchConfig
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER

from .config import graphiti_config
from .recall import recent_episodes, search_facts
from .recall_recheck import recheck
from .recall_render import (
    GLOBAL_SCOPE,
    episode_scope,
    neutralise_tags,
    render,
    render_episode,
)
from .scope import MemoryScope

logger = logging.getLogger(__name__)

# Recent raw episodes shown next to the facts.
_RECENT_EPISODES = 5

# Minimum "signal unit" count for a follow-up user message to trigger a
# warm-context refresh.  Short acknowledgements ("ok", "thanks", "yes") carry
# no new retrieval signal, so refreshing on them would waste a graph search +
# embedding call every turn.  A post-compaction turn bypasses this gate via
# ``refresh_warm_context(..., force=True)``.
#
# Three, not four: the failures this exists to fix are task STARTS mid-session
# ("restart the executor", "deploy prod now", "resume the migration" — all
# exactly three units), so a four-unit floor would exclude the very case it
# targets.  Three lets a stray acknowledgement through ("yes go ahead"), which
# costs one RRF graph query — no LLM calls, and on the SDK engine the refresh
# runs concurrently with the query build, off the time-to-first-token path.
# A missed recall costs the user the bug in SECRT-2378; the asymmetry decides.
WARM_CONTEXT_REFRESH_MIN_WORDS = 3


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
    return _has_signal_units(message, WARM_CONTEXT_REFRESH_MIN_WORDS)


def _has_signal_units(message: str, threshold: int) -> bool:
    """Whether *message* carries at least *threshold* retrieval signal units.

    Short-circuits: the gate only needs to know whether the threshold is
    reached, and this runs on the pre-query path where a user can paste a
    large log. Counting the whole message would make that a full character
    walk for an answer settled in the first few words.

    A plain ``str.split()`` word count under-counts languages that don't
    separate words with whitespace (Japanese, Chinese, Thai) — a long CJK
    message would score 1 "word" and never pass the gate, silently disabling
    the refresh for those users.  So each CJK/ideographic character counts as
    its own unit and is added to the whitespace-word count of the rest.
    """
    units = 0
    in_word = False
    for ch in message:
        # CJK/ideographic characters count individually; everything else is
        # counted by whitespace-delimited run, so neither is double counted.
        if _is_unspaced_script(ch):
            units += 1
            in_word = False
        elif ch.isspace():
            in_word = False
        elif not in_word:
            in_word = True
            units += 1
        if units >= threshold:
            return True
    return False


def _is_unspaced_script(ch: str) -> bool:
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
    expert_id: str | None = None,
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
    ``context_refresh_timeout`` budget. The SDK engine starts it before the
    query build, so it overlaps compaction, attachments and builder context;
    a refresh forced by a compaction, a context-overflow retry and the
    baseline engine await it in front of the model call, which is what the
    tighter budget caps.

    Returns the ``<temporal_context>`` block, or ``None`` when skipped, empty,
    failed or timed out.
    """
    if not user_id:
        return None
    if not force and not should_refresh_warm_context(message):
        return None
    return await fetch_warm_context(
        user_id,
        message or "",
        expert_id,
        use_cross_encoder=False,
        timeout=graphiti_config.context_refresh_timeout,
    )


async def fetch_warm_context(
    user_id: str,
    message: str,
    expert_id: str | None = None,
    *,
    use_cross_encoder: bool = True,
    timeout: float | None = None,
) -> str | None:
    """Fetch relevant temporal context for the current memory owner and message.

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
        scope = MemoryScope.build(user_id, expert_id)
        return await asyncio.wait_for(
            _fetch(scope, message, use_cross_encoder=use_cross_encoder),
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


def _build_search_config(use_cross_encoder: bool) -> SearchConfig:
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

    Neither sets the limit: ``recall.search_facts`` replaces the recipe's
    default ``limit=10`` with ``context_max_facts`` on its own copy, so
    existing operator tuning still applies to both.
    """
    if use_cross_encoder:
        return EDGE_HYBRID_SEARCH_CROSS_ENCODER
    base_edge_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.edge_config
    if base_edge_config is None:
        # The recipe always carries an edge_config; this satisfies the type
        # checker and fails loudly if graphiti ever ships a broken recipe.
        raise RuntimeError("EDGE_HYBRID_SEARCH_CROSS_ENCODER has no edge_config")
    edge_config = base_edge_config.model_copy(update={"reranker": EdgeReranker.rrf})
    return EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(
        update={"edge_config": edge_config}
    )


async def _fetch(
    scope: MemoryScope, message: str, *, use_cross_encoder: bool = True
) -> str | None:
    # P-1.4: warm context is the single most-impactful retrieval per
    # session — the one place where the cross-encoder rerank earns its
    # ~10–15% precision lift (per the audit) at the cost of one extra
    # batch of boolean-classifier prompts. The EDGE_HYBRID_SEARCH_CROSS_ENCODER
    # recipe combines BM25 + cosine + BFS edge search with cross-encoder
    # reranking; a follow-up refresh keeps its search methods and swaps the
    # reranker to RRF (``_build_search_config``). ``context_max_facts``
    # replaces the recipe's default ``limit=10`` so existing operator tuning
    # still applies. Both reads go through the recall policy, so forgotten
    # facts and their episodes stay out, and both lists are read again right
    # before rendering (``recall_recheck.py``): the episodes wait here for the
    # slower search, a forget can answer meanwhile, and what it hid is then
    # not shown.
    edges, episodes = await asyncio.gather(
        search_facts(
            scope,
            message,
            limit=graphiti_config.context_max_facts,
            recipe=_build_search_config(use_cross_encoder),
        ),
        recent_episodes(scope, _RECENT_EPISODES),
    )
    edges, episodes = await recheck(scope, edges, episodes)

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
        _spawn_ratification_hits(scope, edges)

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


def _spawn_ratification_hits(scope: MemoryScope, edges: list[EntityEdge]) -> None:
    """Fire-and-forget the ratification hit-hook for retrieved edges.

    Imports lazily so the dream/ratification module isn't pulled into
    every retrieval boot path; keeps the cold-start cost zero for
    users on the rare GRAPHITI_MEMORY=on / DREAM_PASS_ENABLED=off
    combination.
    """
    edge_uuids = [edge.uuid for edge in edges]
    if not edge_uuids:
        return

    from backend.copilot.dream.ratification import try_ratify_on_hit

    task = asyncio.create_task(
        try_ratify_on_hit(scope, edge_uuids),
        name=f"ratify-hits-{scope.owner_user_id[:12]}",
    )
    _pending_hit_tasks.add(task)
    task.add_done_callback(_on_hit_task_done)


# The block's delimiter, exported so the SDK engine's transcript scrub keys
# off the same constant instead of re-spelling the tag (a rename must not be
# able to leave one module matching and another not).
CONTEXT_TAG_NAME = "temporal_context"


def _format_context(
    edges: list[EntityEdge], episodes: list[EpisodicNode]
) -> str | None:
    sections: list[str] = []

    # Every line is neutralised whole (``recall_render.neutralise_tags``)
    # after it was rendered and, for an episode, cut to display length: the
    # fact's text and validity stamps, and the episode's timestamp and body,
    # all come off the same untrusted memory, so no line may open, close or
    # complete a tag, the block's own delimiters and sections included.
    if edges:
        fact_lines = [f"  - {neutralise_tags(render(edge))}" for edge in edges]
        sections.append("<FACTS>\n" + "\n".join(fact_lines) + "\n</FACTS>")

    # Warm context is scope-agnostic, so a project- or book-scoped memory
    # stays out of it.
    ep_lines = [
        f"  - {neutralise_tags(render_episode(ep))}"
        for ep in episodes
        if episode_scope(ep) == GLOBAL_SCOPE
    ]
    if ep_lines:
        sections.append(
            "<RECENT_EPISODES>\n" + "\n".join(ep_lines) + "\n</RECENT_EPISODES>"
        )

    if not sections:
        return None

    body = "\n\n".join(sections)
    return f"<{CONTEXT_TAG_NAME}>\n{body}\n</{CONTEXT_TAG_NAME}>"
