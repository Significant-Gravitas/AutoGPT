"""Warm context: deterministic memory recall written into a chat turn.

Recall must not depend on the model choosing to call ``memory_search``
(SECRT-2378), so the chat engines put a ``<temporal_context>`` block, keyed
on the user's message, into the turn themselves: into that turn's model
input only, marked, and never into the stored message or a transcript
(``context_marker.py``). The first turn of a session calls
``fetch_warm_context``: graphiti's cross-encoder recipe (BM25, cosine and
BFS edge search, then a per-candidate LLM rerank), ``context_max_facts``
facts, the five newest recallable episodes, ``context_timeout``, and a
ratification hit for every fact shown. Every later user turn refreshes it
through the same fetch with a cheaper recipe (``context_refresh.py``).

Both read through the recall policy (``recall.py``) the same way: live facts
only, recallable episodes only, one last check of both by uuid right before
rendering (``recall_recheck.recheck``), written out by ``recall_render.py``.
Every tag start in the rendered memory is neutralised, after truncation, so
stored text can neither open, close nor complete the block's delimiters
(``recall_render.neutralise_tags``).

A session attached to an organization also reads that organization's shared
tiers (``tiers.resolve_warm_targets``): the org's memory and the session's
team, each only for an active member, under the organization's live workspace
lease. Shared facts come through their lifecycle filter (``lifecycle.py``)
and are labelled with their provenance; only the personal tier goes through
the recall policy and records ratification hits or recall stamps.
"""

import asyncio
import logging

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER

from backend.data.db_accessors import live_resource_lease

from .client import get_graphiti_client
from .config import graphiti_config
from .lifecycle import active_shared_search_filter, filter_active_shared_edges
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
from .scope_resolution import resolve_memory_scope
from .tiers import MemoryTier, TierTarget, merge_tiered, resolve_warm_targets

logger = logging.getLogger(__name__)

# Recent raw episodes of the personal tier shown next to the facts.
_RECENT_EPISODES = 5
# Recent-episode budget for warm context, split personal-first across tiers
# (same >= half rule as facts). Kept small so episodes stay a recency hint,
# not the bulk of the injected context.
_WARM_EPISODE_BUDGET = 8
# Candidates a shared tier's search returns before its lifecycle filter drops
# the facts that are not active.
_SHARED_CANDIDATE_CAP = 50


async def fetch_warm_context(
    user_id: str,
    message: str,
    expert_id: str | None = None,
    *,
    organization_id: str | None = None,
    team_id: str | None = None,
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

    The memory owner's scope is resolved in the session's saved organization
    (``organization_id``), and the read fans out across the personal tier,
    the org tier (when ``organization_id`` is set) and the SESSION'S team
    tier (when ``team_id`` is set AND the user is an active member), merging
    the results with provenance labels.

    Graceful degradation: any error (timeout, connection, graphiti-core bug)
    returns ``None`` so the copilot continues without temporal context.
    """
    if not user_id:
        return None

    effective_timeout = (
        timeout if timeout is not None else graphiti_config.context_timeout
    )
    try:
        scope = await resolve_memory_scope(user_id, expert_id, organization_id)
        return await asyncio.wait_for(
            _fetch(
                scope,
                message,
                organization_id=organization_id,
                team_id=team_id,
                recipe=recipe,
                ratify=ratify,
            ),
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
    organization_id: str | None = None,
    team_id: str | None = None,
    *,
    recipe: SearchConfig = EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    ratify: bool = True,
) -> str | None:
    """The read, under the live workspace lease of the session's organization
    and team. A team the user may no longer view falls back to the
    organization alone; an organization they may no longer view reads
    nothing, its personal tier included."""
    user_id = scope.owner_user_id
    if organization_id is None:
        return await _fetch_authorized(
            scope, message, None, None, recipe=recipe, ratify=ratify
        )

    async with live_resource_lease(
        user_id, organization_id, team_id, "view"
    ) as allowed:
        if allowed:
            action = _fetch_authorized(
                scope, message, organization_id, team_id, recipe=recipe, ratify=ratify
            )
            return (
                await allowed.run(action) if hasattr(allowed, "run") else await action
            )

    if team_id is not None:
        async with live_resource_lease(
            user_id, organization_id, None, "view"
        ) as org_allowed:
            if org_allowed:
                action = _fetch_authorized(
                    scope, message, organization_id, None, recipe=recipe, ratify=ratify
                )
                return (
                    await org_allowed.run(action)
                    if hasattr(org_allowed, "run")
                    else await action
                )

    return None


async def _fetch_authorized(
    scope: MemoryScope,
    message: str,
    organization_id: str | None = None,
    team_id: str | None = None,
    *,
    recipe: SearchConfig = EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    ratify: bool = True,
) -> str | None:
    targets = await resolve_warm_targets(
        scope.owner_user_id, organization_id, team_id, expert_id=scope.expert_id
    )

    # Per-tier failures are non-fatal: a flaky org/team graph must not
    # nuke the personal warm context. Collect exceptions and skip that tier.
    results = await asyncio.gather(
        *(_fetch_tier(scope, target, message, recipe) for target in targets),
        return_exceptions=True,
    )

    personal_edges: list[EntityEdge] = []
    personal_eps: list[EpisodicNode] = []
    shared_edges: list[tuple[str | None, list[EntityEdge]]] = []

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
            personal_edges, personal_eps = edges, episodes
        else:
            shared_edges.append((target.label, edges))

    # Ratification sync hit-hook (P0.4 layer-2): every retrieved personal edge
    # that's currently ``status='tentative'`` gets promoted to ``active``
    # inline, and every retrieved personal edge bumps its warm-context hit
    # counter and gets its recall stamped (``recall_stamp.py``). Only personal
    # edges may reach the assistant-specific ratification; shared-tier
    # ratification is an admin concern. Fire-and-forget so the chat turn
    # never blocks on Redis or FalkorDB writes. A refresh does not ratify
    # (``context_refresh.refresh_warm_context`` says why), but the facts it
    # surfaces were used, so their recalls are still stamped.
    if personal_edges and ratify:
        _spawn_ratification_hits(scope, personal_edges)
    elif personal_edges:
        _spawn_recall_stamps(scope, personal_edges)

    merged_facts = merge_tiered(
        personal_edges, shared_edges, graphiti_config.context_max_facts
    )
    merged_episodes = merge_tiered(personal_eps, [], _WARM_EPISODE_BUDGET)

    if not merged_facts and not merged_episodes:
        return None

    return _format_labeled_context(merged_facts, merged_episodes)


async def _fetch_tier(
    scope: MemoryScope, target: TierTarget, message: str, recipe: SearchConfig
) -> tuple[list[EntityEdge], list[EpisodicNode]]:
    """One tier's facts and recent episodes.

    P-1.4: warm context is the single most-impactful retrieval per session —
    the one place where the cross-encoder rerank earns its ~10–15% precision
    lift (per the audit) at the cost of one extra batch of boolean-classifier
    prompts. The EDGE_HYBRID_SEARCH_CROSS_ENCODER recipe combines BM25 +
    cosine + BFS edge search with cross-encoder reranking;
    ``context_max_facts`` replaces its default ``limit=10`` so existing
    operator tuning still applies.

    The personal tier reads through the recall policy, so forgotten facts and
    their episodes stay out, and both lists are read again right before
    rendering (``recall_recheck.py``): the episodes wait here for the slower
    search, a forget can answer meanwhile, and what it hid is then not shown.
    A shared tier returns its active facts only (``lifecycle.py``), no
    episodes.
    """
    if target.tier == MemoryTier.personal:
        if target.group_id != scope.group_id:
            raise ValueError("Warm context personal tier is not the memory scope")
        edges, episodes = await asyncio.gather(
            search_facts(
                scope,
                message,
                limit=graphiti_config.context_max_facts,
                recipe=recipe,
            ),
            recent_episodes(scope, _RECENT_EPISODES),
        )
        return await recheck(scope, edges, episodes)

    client = await get_graphiti_client(target.group_id)
    shared_config = recipe.model_copy(
        update={
            "limit": min(graphiti_config.context_max_facts * 2, _SHARED_CANDIDATE_CAP)
        }
    )
    edge_results = await client.search_(
        query=message,
        config=shared_config,
        group_ids=[target.group_id],
        search_filter=active_shared_search_filter(),
    )
    edges = edge_results.edges if edge_results is not None else []
    active_edges = await filter_active_shared_edges(
        target.group_id,
        edges,
        driver=getattr(client, "graph_driver", None) or getattr(client, "driver", None),
    )
    return active_edges, []


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
    users whose independent DREAM_PASS_ENABLED flag is off.
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
    """Render unlabelled (personal) facts and episodes."""
    return _format_labeled_context(
        [(edge, None) for edge in edges], [(ep, None) for ep in episodes]
    )


def _format_labeled_context(
    labeled_facts: list[tuple[EntityEdge, str | None]],
    labeled_episodes: list[tuple[EpisodicNode, str | None]],
) -> str | None:
    """Render merged, provenance-labelled facts + episodes.

    ``labeled_facts`` / ``labeled_episodes`` are ``[(item, label)]`` pairs
    where ``label`` is ``None`` for personal (rendered plain) or a shared-
    tier label like ``"org memory"`` / ``"team memory (Platform)"`` that is
    prefixed so the model can weigh provenance.
    """
    sections: list[str] = []

    # Every line is neutralised whole (``recall_render.neutralise_tags``)
    # after it was rendered and, for an episode, cut to display length: the
    # fact's text and validity stamps, and the episode's timestamp and body,
    # all come off the same untrusted memory, so no line may open, close or
    # complete a tag, the block's own delimiters and sections included. The
    # label is inside what is neutralised: a team's name is its admins' text.
    if labeled_facts:
        fact_lines = [
            f"  - {neutralise_tags(_label_prefix(label) + render(edge))}"
            for edge, label in labeled_facts
        ]
        sections.append("<FACTS>\n" + "\n".join(fact_lines) + "\n</FACTS>")

    # Warm context is scope-agnostic, so a project- or book-scoped memory
    # stays out of it.
    ep_lines = [
        f"  - {neutralise_tags(_label_prefix(label) + render_episode(ep))}"
        for ep, label in labeled_episodes
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


def _label_prefix(label: str | None) -> str:
    """Render a provenance label as a bracketed prefix (empty for personal)."""
    return f"[{label}] " if label else ""
