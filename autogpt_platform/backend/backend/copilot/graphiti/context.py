"""Warm context: deterministic memory recall written into a chat turn.

Recall must not depend on the model choosing to call ``memory_search``
(SECRT-2378), so the chat engines put a ``<temporal_context>`` block, keyed
on the user's message, into the turn themselves. The first turn of a
session calls ``fetch_warm_context``: graphiti's cross-encoder recipe (BM25,
cosine and BFS edge search, then a per-candidate LLM rerank),
``context_max_facts`` facts, the five newest recallable episodes,
``context_timeout``, and a ratification hit for every fact shown. Every
later user turn refreshes it through the same fetch with a cheaper recipe
(``context_refresh.py``).

Both read through the recall policy (``recall.py``) the same way: live facts
only, recallable episodes only, one last check of both by uuid right before
rendering (``recall_recheck.recheck``), written out by ``recall_render.py``.
Every tag start in the rendered memory is neutralised, after truncation, so
stored text can neither open, close nor complete the block's delimiters
(``recall_render.neutralise_tags``).
"""

import asyncio
import logging

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode
from graphiti_core.search.search_config import SearchConfig
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
from .recall_stamp import stamp_recalls_in_scope
from .scope import MemoryScope

logger = logging.getLogger(__name__)

# Recent raw episodes shown next to the facts.
_RECENT_EPISODES = 5


async def fetch_warm_context(
    user_id: str,
    message: str,
    expert_id: str | None = None,
    *,
    recipe: SearchConfig = EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    ratify: bool = True,
    timeout: float | None = None,
) -> str | None:
    """Fetch relevant temporal context for the current memory owner and message.

    Returns a formatted ``<temporal_context>`` block suitable for appending
    to the current turn's user message, or ``None`` on failure/empty. The
    defaults are the first turn's: the cross-encoder ``recipe``, a
    ratification hit for every fact shown, ``context_timeout``. The
    follow-up refresh passes its own (``context_refresh.py``).

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
            _fetch(scope, message, recipe=recipe, ratify=ratify),
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


async def _fetch(
    scope: MemoryScope,
    message: str,
    *,
    recipe: SearchConfig = EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    ratify: bool = True,
) -> str | None:
    # P-1.4: warm context is the single most-impactful retrieval per
    # session — the one place where the cross-encoder rerank earns its
    # ~10–15% precision lift (per the audit) at the cost of one extra
    # batch of boolean-classifier prompts. The EDGE_HYBRID_SEARCH_CROSS_ENCODER
    # recipe combines BM25 + cosine + BFS edge search with cross-encoder
    # reranking; ``context_max_facts`` replaces its default ``limit=10`` so
    # existing operator tuning still applies. Both reads go through the
    # recall policy, so forgotten facts and their episodes stay out, and both
    # lists are read again right before rendering (``recall_recheck.py``):
    # the episodes wait here for the slower search, a forget can answer
    # meanwhile, and what it hid is then not shown.
    edges, episodes = await asyncio.gather(
        search_facts(
            scope,
            message,
            limit=graphiti_config.context_max_facts,
            recipe=recipe,
        ),
        recent_episodes(scope, _RECENT_EPISODES),
    )
    edges, episodes = await recheck(scope, edges, episodes)

    # Ratification sync hit-hook (P0.4 layer-2): every retrieved edge
    # that's currently ``status='tentative'`` gets promoted to
    # ``active`` inline, and every retrieved edge bumps its
    # warm-context hit counter and gets its recall stamped
    # (``recall_stamp.py``). Fire-and-forget so the chat turn
    # never blocks on Redis or FalkorDB writes. A refresh does not
    # ratify (``context_refresh.refresh_warm_context`` says why), but the
    # facts it surfaces were used, so their recalls are still stamped.
    if edges and ratify:
        _spawn_ratification_hits(scope, edges)
    elif edges:
        _spawn_recall_stamps(scope, edges)

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


def _spawn_recall_stamps(scope: MemoryScope, edges: list[EntityEdge]) -> None:
    """Fire-and-forget the recall stamp for edges a follow-up refresh
    surfaced: they were used, so a dream should not demote them for
    staleness, but a refresh records no ratification hits."""
    task = asyncio.create_task(
        stamp_recalls_in_scope(scope, [edge.uuid for edge in edges]),
        name=f"stamp-recalls-{scope.owner_user_id[:12]}",
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
