"""Warm context retrieval — pre-loads relevant facts at session start."""

import asyncio
import logging

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER

from .config import graphiti_config
from .recall import recent_episodes, search_facts
from .recall_render import GLOBAL_SCOPE, episode_scope, render, render_episode
from .scope import MemoryScope

logger = logging.getLogger(__name__)

# Recent raw episodes shown next to the facts.
_RECENT_EPISODES = 5


async def fetch_warm_context(
    user_id: str, message: str, expert_id: str | None = None
) -> str | None:
    """Fetch relevant temporal context for the current memory owner and message.

    Called at the start of a session (first turn) to pre-load facts from
    prior conversations.  Returns a formatted ``<temporal_context>`` block
    suitable for appending to the system prompt, or ``None`` on failure.

    Graceful degradation: any error (timeout, connection, graphiti-core bug)
    returns ``None`` so the copilot continues without temporal context.
    """
    if not user_id:
        return None

    try:
        scope = MemoryScope.build(user_id, expert_id)
        return await asyncio.wait_for(
            _fetch(scope, message),
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


async def _fetch(scope: MemoryScope, message: str) -> str | None:
    # P-1.4: warm context is the single most-impactful retrieval per
    # session — the one place where the cross-encoder rerank earns its
    # ~10–15% precision lift (per the audit) at the cost of one extra
    # batch of boolean-classifier prompts. The EDGE_HYBRID_SEARCH_CROSS_ENCODER
    # recipe combines BM25 + cosine + BFS edge search with cross-encoder
    # reranking; ``context_max_facts`` replaces its default ``limit=10`` so
    # existing operator tuning still applies. Both reads go through the
    # recall policy, so forgotten facts and their episodes stay out.
    edges, episodes = await asyncio.gather(
        search_facts(
            scope,
            message,
            limit=graphiti_config.context_max_facts,
            recipe=EDGE_HYBRID_SEARCH_CROSS_ENCODER,
        ),
        recent_episodes(scope, _RECENT_EPISODES),
    )

    # Ratification sync hit-hook (P0.4 layer-2): every retrieved edge
    # that's currently ``status='tentative'`` gets promoted to
    # ``active`` inline, and every retrieved edge bumps its
    # warm-context hit counter. Fire-and-forget so the chat turn
    # never blocks on Redis or FalkorDB writes.
    if edges:
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


def _format_context(
    edges: list[EntityEdge], episodes: list[EpisodicNode]
) -> str | None:
    sections: list[str] = []

    if edges:
        fact_lines = [f"  - {render(edge)}" for edge in edges]
        sections.append("<FACTS>\n" + "\n".join(fact_lines) + "\n</FACTS>")

    # Warm context is scope-agnostic, so a project- or book-scoped memory
    # stays out of it.
    ep_lines = [
        f"  - {render_episode(ep)}"
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
    return f"<temporal_context>\n{body}\n</temporal_context>"
