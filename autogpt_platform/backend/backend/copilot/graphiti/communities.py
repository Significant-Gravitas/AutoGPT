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
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from .client import derive_group_id, get_graphiti_client

logger = logging.getLogger(__name__)


class CommunityRebuildResult(BaseModel):
    """Outcome of one per-user community rebuild pass.

    Returned by ``rebuild_communities_for_user`` on both success and
    failure paths so the scheduler can record telemetry uniformly.
    ``communities_built`` is a free-form summary because graphiti-core's
    ``build_communities`` return shape changes between versions.

    ``skipped`` is set by callers that gate the rebuild behind a flag
    (e.g. the LD-gated manual trigger) so the response shape stays
    uniform regardless of whether the rebuild actually ran.
    """

    user_id: str
    started_at: str
    elapsed_seconds: float | None = None
    communities_built: dict[str, Any] | None = None
    error: str | None = None
    skipped: bool = False
    skipped_reason: str | None = None


class CommunityRebuildEnqueueResult(BaseModel):
    """Outcome of enqueuing a manual community rebuild.

    The manual-trigger endpoint enqueues the rebuild as a one-shot
    APScheduler job and returns immediately so the caller doesn't block
    the small APScheduler thread pool on Leiden + LLM-summarization. The
    actual rebuild outcome is logged from ``execute_community_rebuild``;
    callers that need the result can poll their own job-status surface.
    """

    user_id: str
    job_id: str | None = None
    queued: bool = False
    skipped: bool = False
    skipped_reason: str | None = None


async def rebuild_communities_for_user(user_id: str) -> CommunityRebuildResult:
    """Destroy and rebuild ``:Community`` nodes for one user's graph.

    Always returns a ``CommunityRebuildResult`` even on failure so the
    scheduler can record the outcome.
    """
    started_at = datetime.now(timezone.utc)
    result = CommunityRebuildResult(
        user_id=user_id,
        started_at=started_at.isoformat(),
    )

    try:
        try:
            group_id = derive_group_id(user_id)
        except ValueError as exc:
            result.error = f"invalid_user_id: {exc}"
            logger.warning(
                f"Skipping community rebuild — invalid user_id {user_id[:12]}"
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
            result.communities_built = _summarize_communities(summary)

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"Community rebuild failed for user {user_id[:12]}", exc_info=True
            )
    finally:
        ended_at = datetime.now(timezone.utc)
        result.elapsed_seconds = (ended_at - started_at).total_seconds()

    return result


def _summarize_communities(summary: Any) -> dict[str, Any] | None:
    """Reduce graphiti-core's build_communities return value to something
    JSON-loggable.

    Older versions return a list of CommunityNode objects, newer return
    a dict. We just need a count for telemetry; preserve the raw form
    on the side for debugging when possible.
    """
    if summary is None:
        return None
    if isinstance(summary, list):
        return {"count": len(summary)}
    if isinstance(summary, dict):
        return summary
    # Fallback for unknown shapes — best-effort string coercion.
    return {"raw": str(summary)[:200]}
