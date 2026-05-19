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

from .client import derive_group_id, get_graphiti_client

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
                community_candidates[community_map[neighbor.node_uuid]] += (
                    neighbor.edge_count
                )
            community_lst = [
                (count, community)
                for community, count in community_candidates.items()
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


async def rebuild_communities_for_user(user_id: str) -> dict[str, Any]:
    """Destroy and rebuild ``:Community`` nodes for one user's graph.

    Returns a result dict with ``user_id``, ``communities_built``,
    ``elapsed_seconds``, and ``error`` (if any). Always returns a dict
    even on failure so the scheduler can record the outcome.
    """
    started_at = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "user_id": user_id,
        "started_at": started_at.isoformat(),
        "communities_built": None,
        "elapsed_seconds": None,
        "error": None,
    }

    try:
        group_id = derive_group_id(user_id)
    except ValueError as exc:
        result["error"] = f"invalid_user_id: {exc}"
        logger.warning(
            "Skipping community rebuild — invalid user_id %s", user_id[:12]
        )
        return result

    try:
        client = await get_graphiti_client(group_id)

        # Defensive: clean up any orphan :Community nodes regardless of
        # upstream version. Per multi-episode research, modern Graphiti
        # already does this inside build_communities(), but older
        # versions did not. Idempotent either way.
        driver = getattr(client, "graph_driver", None) or getattr(
            client, "driver", None
        )
        if driver is None:
            raise RuntimeError("Graphiti client has no graph_driver")

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
        ended_at = datetime.now(timezone.utc)
        result["elapsed_seconds"] = (ended_at - started_at).total_seconds()

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
