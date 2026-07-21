"""Graphiti community detection + scheduled rebuilds.

Per the audit (``dream/dreaming-graphiti.md`` §8 item 1), community
detection is "not free": Leiden / label-propagation over a non-trivial
graph is ``O(|V| · log|V|)`` plus an LLM summarization call per
community. The audit's default recommendation was to leave communities
off in P0 and revisit only if retrieval relevance plateaued.

The user direction for P-1 overrides that default: enable communities
now, behind a LaunchDarkly flag, run rebuilds during off-peak windows
to bound cost. The scheduler hook in ``backend/executor/scheduler.py``
calls into this module.

Operational notes:

- ``Graphiti.build_communities()`` upstream calls ``remove_communities()``
  first — destroy-and-rebuild — but the multi-episode research found
  that older Graphiti versions did not. We add a defensive explicit
  ``MATCH (c:Community {group_id: $group_id}) DETACH DELETE c`` before
  the rebuild so the behavior is identical across versions.

- The rebuild is per-user. We do **not** support cross-user community
  detection (cross-user memory is deferred to org-scoped memory; see
  ``dream/TODO.md`` deferred section).

- Failures are non-fatal. A community rebuild that crashes leaves
  the graph in a usable state (we either have the previous communities
  or none); the next scheduled run will retry. Caller is expected to
  log failures but not halt other scheduled work.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.data.redis_client import get_redis_async

from .client import (
    close_graphiti_client,
    derive_group_id,
    get_graphiti_client,
    make_flex_graphiti_client,
)
from .config import graphiti_config

logger = logging.getLogger(__name__)


# Upstream graphiti-core's ``label_propagation`` has an unbounded
# ``while True:`` with no convergence guarantee. Synchronous label
# propagation is known to oscillate on bipartite-ish subgraphs: two
# nodes can flip labels in lock-step every iteration, forever.
#
# Reproduced in dev on a 107-entity / 104-edge graph: ``min_max`` in a
# tight loop, 100% CPU for 12+ minutes with no progress, sample-trace
# evidence. See ``dream/dreaming-graphiti.md`` §8 (anti-patterns) and
# the inline comment in upstream
# ``graphiti_core/utils/maintenance/community_operations.py`` for the
# original algorithm.
#
# We cap the loop at this many iterations. Real-world label-propagation
# converges in << 20 iterations; 50 is a generous ceiling. If we hit
# the cap, we log a warning and return the partial labeling — still a
# valid clustering (every node has *some* community assignment), just
# not optimal. Upstream issue to file: add max_iterations parameter.
MAX_LABEL_PROP_ITERATIONS = 50


def _bounded_label_propagation(projection):
    """Drop-in replacement for graphiti-core's ``label_propagation`` that
    caps iterations to avoid the synchronous-LP oscillation infinite loop.

    Same algorithm and same tie-breaking semantics as upstream — only
    the loop is bounded. On convergence the result is identical to the
    unbounded version.
    """
    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}
    converged = False

    for _ in range(MAX_LABEL_PROP_ITERATIONS):
        no_change = True
        new_community_map: dict[str, int] = {}

        for uuid, neighbors in projection.items():
            curr_community = community_map[uuid]

            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                community_candidates[
                    community_map[neighbor.node_uuid]
                ] += neighbor.edge_count
            community_lst = [
                (count, community) for community, count in community_candidates.items()
            ]

            community_lst.sort(reverse=True)
            candidate_rank, community_candidate = (
                community_lst[0] if community_lst else (0, -1)
            )
            if community_candidate != -1 and candidate_rank > 1:
                new_community = community_candidate
            else:
                new_community = max(community_candidate, curr_community)

            new_community_map[uuid] = new_community

            if new_community != curr_community:
                no_change = False

        if no_change:
            converged = True
            break

        community_map = new_community_map

    if not converged:
        logger.warning(
            "label_propagation hit %d-iteration cap without converging on "
            "projection with %d nodes — returning partial clustering. "
            "This usually means the graph has a bipartite-ish subgraph "
            "that causes synchronous LP to oscillate.",
            MAX_LABEL_PROP_ITERATIONS,
            len(projection),
        )

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)
    return [cluster for cluster in community_cluster_map.values()]


# Monkey-patch upstream at import time. ``get_community_clusters`` in
# the same module references ``label_propagation`` via module-level
# name lookup, so swapping the attribute works.
def _patch_upstream_label_propagation() -> None:
    try:
        from graphiti_core.utils.maintenance import community_operations as _ops
    except ImportError:
        logger.debug("graphiti_core not importable; skipping label_propagation patch")
        return
    if getattr(_ops.label_propagation, "_autogpt_bounded", False):
        return  # idempotent
    _bounded_label_propagation._autogpt_bounded = True  # type: ignore[attr-defined]
    _ops.label_propagation = _bounded_label_propagation
    logger.info(
        "Patched graphiti_core.label_propagation with bounded variant "
        "(max %d iterations).",
        MAX_LABEL_PROP_ITERATIONS,
    )


_patch_upstream_label_propagation()


async def _activity_since_last_rebuild(
    driver,
    group_id: str,
    min_new_episodes: int,
) -> tuple[bool, str, dict[str, Any]]:
    """Decide whether the user's graph has changed enough to justify a rebuild.

    Returns ``(should_rebuild, reason, stats)``. The graph itself is the
    source of truth — we compare the newest ``:Episodic.created_at`` to
    the newest ``:Community.created_at`` and additionally require at least
    ``min_new_episodes`` net-new episodes since the last build.

    No external state to maintain (no Redis key, no Postgres column).
    Two indexed Cypher queries; sub-millisecond on FalkorDB.
    """
    ep_q = """
    MATCH (n:Episodic {group_id: $g})
    RETURN max(n.created_at) AS latest, count(n) AS total
    """
    co_q = """
    MATCH (n:Community {group_id: $g})
    RETURN max(n.created_at) AS latest, count(n) AS total
    """
    ep_rows, _, _ = await driver.execute_query(ep_q, g=group_id)
    co_rows, _, _ = await driver.execute_query(co_q, g=group_id)

    latest_ep = ep_rows[0]["latest"] if ep_rows else None
    total_ep = ep_rows[0]["total"] if ep_rows else 0
    latest_co = co_rows[0]["latest"] if co_rows else None
    total_co = co_rows[0]["total"] if co_rows else 0

    stats: dict[str, Any] = {
        "episodes_total": total_ep,
        "communities_total": total_co,
        "latest_episode_at": str(latest_ep) if latest_ep is not None else None,
        "latest_community_at": str(latest_co) if latest_co is not None else None,
        "min_new_episodes_threshold": min_new_episodes,
    }

    if total_ep == 0:
        return False, "no_episodes", stats

    if total_co == 0:
        # First-ever rebuild — go.
        stats["new_episodes_since_last_rebuild"] = total_ep
        return True, "first_rebuild", stats

    if latest_ep is None or latest_co is None or latest_ep <= latest_co:
        stats["new_episodes_since_last_rebuild"] = 0
        return False, "no_new_episodes_since_last_rebuild", stats

    # Count exactly how many episodes are newer than the latest community
    # to enforce the activity threshold.
    count_q = """
    MATCH (n:Episodic {group_id: $g})
    WHERE n.created_at > $since
    RETURN count(n) AS c
    """
    rows, _, _ = await driver.execute_query(count_q, g=group_id, since=latest_co)
    new_count = rows[0]["c"] if rows else 0
    stats["new_episodes_since_last_rebuild"] = new_count

    if new_count < min_new_episodes:
        return False, "below_activity_threshold", stats

    return True, "activity_above_threshold", stats


# Per-user community-rebuild lock. The rebuild does a destructive
# ``DETACH DELETE`` + Leiden clustering + per-community LLM summarization;
# two racing on the same user's graph corrupt each other. Every rebuild
# caller (weekly cron, admin-immediate, sync debug) funnels through
# ``rebuild_communities_for_user``, so the guard lives here rather than in
# any single caller.
#
# Cluster-safe by construction: production Redis is an ``AsyncRedisCluster``,
# so we use only single-key primitives — ``SET key token NX EX`` to acquire
# and a single-key Lua compare-and-delete to release, both of which route to
# the slot owner. The random token makes release safe: we delete the key
# only if we still own it, so a rebuild that overran its TTL can never drop a
# lock another worker has since taken.
#
# The TTL exceeds the scheduler's hard rebuild bound
# (``SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS`` = 1800s; every caller wraps
# the rebuild in ``run_async(..., timeout=...)``), so the lock cannot expire
# mid-rebuild and no lease-renewal watchdog is needed — the TTL is purely a
# crash backstop.
_REBUILD_LOCK_KEY_PREFIX = "graphiti:community_rebuild_lock:"
_REBUILD_LOCK_TTL_SECONDS = 1800 + 120

_REBUILD_UNLOCK_SCRIPT = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) else return 0 end"
)


def _rebuild_lock_key(group_id: str) -> str:
    return f"{_REBUILD_LOCK_KEY_PREFIX}{group_id}"


async def _release_rebuild_lock(redis, key: str, token: str) -> None:
    """Compare-and-delete: drop the lock only if we still hold this token.

    Single-key Lua so it routes on Redis Cluster. On any redis error the TTL
    clears the key, so a failed release never wedges the user.
    """
    try:
        await redis.eval(_REBUILD_UNLOCK_SCRIPT, 1, key, token)
    except Exception:
        logger.warning(
            "Failed to release community-rebuild lock %s — TTL will clear it",
            key,
        )


async def rebuild_communities_for_user(
    user_id: str, *, force: bool = False
) -> dict[str, Any]:
    """Destroy and rebuild ``:Community`` nodes for one user's graph.

    Gated by an activity check (``_activity_since_last_rebuild``) so we
    only pay LLM-summarization cost when the graph has actually changed
    by at least ``GraphitiConfig.community_rebuild_min_new_episodes``
    since the last successful rebuild. Skipping also avoids non-
    deterministic clustering drift (LP tie-breaks, summary text)
    on essentially-unchanged graphs.

    Set ``force=True`` to bypass the gate (admin override / debugging).

    Returns a result dict with ``user_id``, ``communities_built``,
    ``elapsed_seconds``, ``error`` (if any), ``skipped`` (bool), and
    ``activity`` (the gate's stats). Always returns a dict even on
    failure so the scheduler can record the outcome.
    """
    started_at = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "user_id": user_id,
        "started_at": started_at.isoformat(),
        "communities_built": None,
        "elapsed_seconds": None,
        "error": None,
        "skipped": False,
        "skip_reason": None,
        "activity": None,
        "forced": force,
        "execution_path": None,
    }

    try:
        group_id = derive_group_id(user_id)
    except ValueError as exc:
        result["error"] = f"invalid_user_id: {exc}"
        logger.warning("Skipping community rebuild — invalid user_id %s", user_id[:12])
        return result

    # When the flex flag is on we build a one-shot Graphiti client whose
    # LLM calls run on OpenAI's flex tier (~50% discount, best-effort
    # latency). The cached interactive client stays on sync tier so
    # live ingest dedup remains responsive. The flex client owns its
    # own FalkorDB driver — close it in finally to avoid leaks.
    #
    # Transport veto: ``make_flex_graphiti_client`` silently falls
    # back to the regular ``OpenAIClient`` when the active transport
    # can't honour ``service_tier="flex"`` (local Ollama et al.). To
    # keep the cost-log audit trail honest we record ``execution_path``
    # against what *actually* dispatched, not what the operator asked
    # for. Read the same gate the helper reads so the two stay in sync.
    from backend.copilot.sdk.env import config as chat_cfg

    flex_requested = graphiti_config.community_rebuild_use_flex_tier
    use_flex = flex_requested and chat_cfg.transport.supports_flex_tier
    flex_client = None

    # Per-user mutual exclusion — acquired BEFORE the activity gate so a
    # concurrent rebuild short-circuits instead of racing the DETACH DELETE
    # below. A contended acquire rides the existing ``skipped`` contract.
    redis = await get_redis_async()
    lock_key = _rebuild_lock_key(group_id)
    lock_token = uuid4().hex
    if not await redis.set(lock_key, lock_token, nx=True, ex=_REBUILD_LOCK_TTL_SECONDS):
        result["skipped"] = True
        result["skip_reason"] = "rebuild already in progress"
        logger.info(
            "Community rebuild skipped for user %s — already in progress",
            user_id[:12],
        )
        return result

    try:
        if use_flex:
            flex_client = await make_flex_graphiti_client(group_id)
            client = flex_client
            result["execution_path"] = "flex"
        else:
            client = await get_graphiti_client(group_id)
            result["execution_path"] = "sync"

        driver = getattr(client, "graph_driver", None) or getattr(
            client, "driver", None
        )
        if driver is None:
            raise RuntimeError("Graphiti client has no graph_driver")

        # Activity gate — skip when nothing has changed since the last
        # rebuild. Cheap two-query check; the LLM cost we save is the
        # whole point of community detection being a scheduled job
        # rather than an always-run pass.
        if not force:
            should_rebuild, reason, stats = await _activity_since_last_rebuild(
                driver,
                group_id,
                min_new_episodes=graphiti_config.community_rebuild_min_new_episodes,
            )
            result["activity"] = stats
            if not should_rebuild:
                result["skipped"] = True
                result["skip_reason"] = reason
                logger.info(
                    "Community rebuild skipped for user %s — %s (stats=%s)",
                    user_id[:12],
                    reason,
                    stats,
                )
                return result

        # Defensive: clean up any orphan :Community nodes regardless of
        # upstream version. Per multi-episode research, modern Graphiti
        # already does this inside build_communities(), but older
        # versions did not. Idempotent either way.
        await driver.execute_query(
            """
            MATCH (c:Community {group_id: $group_id})
            DETACH DELETE c
            """,
            group_id=group_id,
        )

        # Rebuild via Graphiti. The result shape changes between graphiti
        # versions; we record whatever it returns rather than asserting
        # a specific shape.
        summary = await client.build_communities(group_ids=[group_id])
        result["communities_built"] = _summarize_communities(summary)

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Community rebuild failed for user %s", user_id[:12], exc_info=True
        )

    finally:
        if flex_client is not None:
            await close_graphiti_client(flex_client)
        ended_at = datetime.now(timezone.utc)
        result["elapsed_seconds"] = (ended_at - started_at).total_seconds()
        await _release_rebuild_lock(redis, lock_key, lock_token)

    return result


def _summarize_communities(summary: Any) -> Any:
    """Reduce graphiti-core's build_communities return value to something
    JSON-loggable.

    Upstream signature is ``tuple[list[CommunityNode], list[CommunityEdge]]``
    in the version we pin (0.28.2). Older versions returned a bare list of
    CommunityNode; some return dicts. Cover all three so telemetry stays
    readable across version bumps.
    """
    if summary is None:
        return None
    if isinstance(summary, tuple) and len(summary) == 2:
        nodes, edges = summary
        return {
            "nodes": len(nodes) if hasattr(nodes, "__len__") else None,
            "edges": len(edges) if hasattr(edges, "__len__") else None,
        }
    if isinstance(summary, list):
        return {"count": len(summary)}
    if isinstance(summary, dict):
        return summary
    # Fallback for unknown shapes — best-effort string coercion.
    return {"raw": str(summary)[:200]}
