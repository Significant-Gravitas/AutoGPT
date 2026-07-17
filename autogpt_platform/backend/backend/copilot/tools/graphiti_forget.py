"""Two-step tool for targeted memory deletion.

Step 1 (memory_forget_search): search for matching facts, return candidates.
Step 2 (memory_forget_confirm): delete specific edges by UUID after user confirms.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from backend.copilot.graphiti._format import extract_fact, extract_temporal_validity
from backend.copilot.graphiti.client import derive_group_id, get_graphiti_client
from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import (
    ErrorResponse,
    MemoryForgetCandidatesResponse,
    MemoryForgetConfirmResponse,
    ToolResponseBase,
)


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string for Cypher parameter binding.

    FalkorDB does not implement Cypher's no-arg ``datetime()`` function
    (the error is ``Unknown function 'datetime'``), so timestamp values
    have to be generated in Python and passed as a parameter.  ISO
    strings work for the comparison + ordering we use (lexical sort on
    ISO-8601 matches chronological sort) and round-trip cleanly through
    ``toString(...)`` reads we already do.
    """
    return datetime.now(timezone.utc).isoformat()


logger = logging.getLogger(__name__)


class MemoryForgetSearchTool(BaseTool):
    """Search for memories to forget — returns candidates for user confirmation."""

    @property
    def name(self) -> str:
        return "memory_forget_search"

    @property
    def description(self) -> str:
        return (
            "Search for stored memories matching a description so the user can "
            "choose which to delete. Returns candidate facts with UUIDs. "
            "Use memory_forget_confirm with the UUIDs to actually delete them."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what to forget (e.g. 'the Q2 marketing budget')",
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        query: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                session_id=session.session_id,
            )

        if not await is_enabled_for_user(user_id):
            return ErrorResponse(
                message="Memory features are not enabled for your account.",
                session_id=session.session_id,
            )

        if not query:
            return ErrorResponse(
                message="A search query is required to find memories to forget.",
                session_id=session.session_id,
            )

        try:
            group_id = derive_group_id(user_id)
        except ValueError:
            return ErrorResponse(
                message="Invalid user ID for memory operations.",
                session_id=session.session_id,
            )

        try:
            client = await get_graphiti_client(group_id)
            edges = await client.search(
                query=query,
                group_ids=[group_id],
                num_results=10,
            )
        except Exception:
            logger.warning(
                "Memory forget search failed for user %s", user_id[:12], exc_info=True
            )
            return ErrorResponse(
                message="Memory search is temporarily unavailable.",
                session_id=session.session_id,
            )

        if not edges:
            return MemoryForgetCandidatesResponse(
                message="No matching memories found.",
                session_id=session.session_id,
                candidates=[],
            )

        candidates = []
        for e in edges:
            edge_uuid = getattr(e, "uuid", None) or getattr(e, "id", None)
            if not edge_uuid:
                continue
            fact = extract_fact(e)
            valid_from, valid_to = extract_temporal_validity(e)
            candidates.append(
                {
                    "uuid": str(edge_uuid),
                    "fact": fact,
                    "valid_from": str(valid_from),
                    "valid_to": str(valid_to),
                }
            )

        return MemoryForgetCandidatesResponse(
            message=f"Found {len(candidates)} candidate(s). Show these to the user and ask which to delete, then call memory_forget_confirm with the UUIDs.",
            session_id=session.session_id,
            candidates=candidates,
        )


class MemoryForgetConfirmTool(BaseTool):
    """Delete specific memory edges by UUID after user confirmation.

    Supports both soft delete (temporal invalidation — reversible) and
    hard delete (remove from graph — irreversible, for GDPR).
    """

    @property
    def name(self) -> str:
        return "memory_forget_confirm"

    @property
    def description(self) -> str:
        return (
            "Delete specific memories by UUID. Use after memory_forget_search "
            "returns candidates and the user confirms which to delete. "
            "Default is soft delete (marks as expired but keeps history). "
            "Set hard_delete=true for permanent removal (GDPR)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "uuids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of edge UUIDs to delete (from memory_forget_search results)",
                },
                "hard_delete": {
                    "type": "boolean",
                    "description": "If true, permanently removes edges from the graph (GDPR). Default false (soft delete — marks as expired).",
                    "default": False,
                },
            },
            "required": ["uuids"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        uuids: list[str] | None = None,
        hard_delete: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                session_id=session.session_id,
            )

        if not await is_enabled_for_user(user_id):
            return ErrorResponse(
                message="Memory features are not enabled for your account.",
                session_id=session.session_id,
            )

        if not uuids:
            return ErrorResponse(
                message="At least one UUID is required. Use memory_forget_search first.",
                session_id=session.session_id,
            )

        try:
            group_id = derive_group_id(user_id)
        except ValueError:
            return ErrorResponse(
                message="Invalid user ID for memory operations.",
                session_id=session.session_id,
            )

        try:
            client = await get_graphiti_client(group_id)
        except Exception:
            logger.warning(
                "Failed to get Graphiti client for user %s", user_id[:12], exc_info=True
            )
            return ErrorResponse(
                message="Memory service is temporarily unavailable.",
                session_id=session.session_id,
            )

        driver = getattr(client, "graph_driver", None) or getattr(
            client, "driver", None
        )
        if not driver:
            return ErrorResponse(
                message="Could not access graph driver for deletion.",
                session_id=session.session_id,
            )

        if hard_delete:
            deleted, failed = await _hard_delete_edges(driver, uuids, user_id)
            mode = "permanently deleted"
        else:
            # User-initiated forget is a *system* retraction, not a world
            # change. Per Snodgrass bi-temporal semantics, only `expired_at`
            # is set. `_soft_delete_edges` (which also sets `invalid_at`)
            # is reserved for the contradiction detector.
            deleted, failed = await _retract_edges(driver, uuids, user_id)
            mode = "retracted from memory"

        return MemoryForgetConfirmResponse(
            message=(
                f"{len(deleted)} memory edge(s) {mode}."
                + (f" {len(failed)} failed." if failed else "")
            ),
            session_id=session.session_id,
            deleted_uuids=deleted,
            failed_uuids=failed,
        )


async def _retract_edges(
    driver, uuids: list[str], user_id: str
) -> tuple[list[str], list[str]]:
    """System retraction — set ONLY ``expired_at`` on the edge.

    Per Snodgrass bi-temporal semantics (see ``dream/dreaming-graphiti.md``
    §6.13), ``expired_at`` is *transaction time* ("we retracted the
    record") and ``invalid_at`` is *valid time* ("the world changed").
    User-initiated forget, dream demotion, and entity invalidation are
    all system retractions and must NOT set ``invalid_at``.

    For contradiction detection (the world really did change) use
    ``_soft_delete_edges`` below, which sets both.

    Matches the same edge types as ``_hard_delete_edges`` so that edges of
    any type (RELATES_TO, MENTIONS, HAS_MEMBER) can be retracted.
    """
    deleted = []
    failed = []
    for uuid in uuids:
        try:
            records, _, _ = await driver.execute_query(
                """
                MATCH ()-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->()
                SET e.expired_at = $now
                RETURN e.uuid AS uuid
                """,
                uuid=uuid,
                now=_now_iso(),
            )
            if records:
                deleted.append(uuid)
            else:
                failed.append(uuid)
        except Exception:
            logger.warning(
                "Failed to retract edge %s for user %s",
                uuid,
                user_id[:12],
                exc_info=True,
            )
            failed.append(uuid)
    return deleted, failed


async def _soft_delete_edges(
    driver, uuids: list[str], user_id: str
) -> tuple[list[str], list[str]]:
    """Bi-temporal invalidation — mark edges as both expired AND invalid.

    Reserved for the *contradiction detector*: when new evidence proves
    a fact ceased being true in the world, set ``invalid_at`` (valid time)
    in addition to ``expired_at`` (transaction time). User-initiated
    forget should use ``_retract_edges`` instead; conflating the two
    breaks the bi-temporal model (audit §6.13).

    Matches RELATES_TO, MENTIONS, HAS_MEMBER edges.
    """
    deleted = []
    failed = []
    for uuid in uuids:
        try:
            records, _, _ = await driver.execute_query(
                """
                MATCH ()-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->()
                SET e.invalid_at = $now,
                    e.expired_at = $now
                RETURN e.uuid AS uuid
                """,
                uuid=uuid,
                now=_now_iso(),
            )
            if records:
                deleted.append(uuid)
            else:
                failed.append(uuid)
        except Exception:
            logger.warning(
                "Failed to soft-delete edge %s for user %s",
                uuid,
                user_id[:12],
                exc_info=True,
            )
            failed.append(uuid)
    return deleted, failed


async def mark_edges_superseded(
    driver,
    uuids: list[str],
    reason: str,
    new_status: Literal["superseded", "contradicted"] = "superseded",
    user_id: str | None = None,
    group_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """Retract edges AND set the custom audit-trail ``status`` property.

    Intended for the dream pass (P0.3 stale-fact deprecation): retract
    the edge per ``_retract_edges`` semantics and stamp
    ``status='superseded'`` (or ``'contradicted'``) plus
    ``expiration_reason=<reason>`` so the demotion is queryable from
    search (``WHERE e.status = 'superseded'``).

    ``group_id`` adds defense-in-depth: the driver is normally opened
    against the per-user FalkorDB database, but when provided the
    Cypher predicate also requires the edge's ``group_id`` to match so
    a future caller holding the wrong driver can't touch another
    user's edges. ``None`` keeps the unscoped match (current
    ratification behavior).

    Returns ``(succeeded_uuids, failed_uuids)``.
    """
    deleted = []
    failed = []
    user_log = (user_id or "?")[:12]
    edge_match = (
        "MATCH ()-[e:RELATES_TO {uuid: $uuid, group_id: $group_id}]->()"
        if group_id is not None
        else "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()"
    )
    query = f"""
                {edge_match}
                SET e.expired_at = $now,
                    e.status = $new_status,
                    e.expiration_reason = $reason
                RETURN e.uuid AS uuid
                """
    for uuid in uuids:
        try:
            params: dict[str, str] = {
                "uuid": uuid,
                "new_status": new_status,
                "reason": reason,
                "now": _now_iso(),
            }
            if group_id is not None:
                params["group_id"] = group_id
            records, _, _ = await driver.execute_query(query, **params)
            if records:
                deleted.append(uuid)
            else:
                failed.append(uuid)
        except Exception:
            logger.warning(
                "Failed to mark edge %s superseded for user %s",
                uuid,
                user_log,
                exc_info=True,
            )
            failed.append(uuid)
    return deleted, failed


async def invalidate_entity_direct_neighbors(
    driver,
    group_id: str,
    entity_uuid: str,
    reason: str,
) -> list[str]:
    """Demote every ``:RELATES_TO`` edge directly attached to an entity.

    **Single-hop only** — does NOT propagate to neighbors-of-neighbors.
    The instinct to write ``[r:RELATES_TO*1..N]`` is exactly the
    runaway-demotion bug we are protecting against (P0.3b in the dream
    spec). Keep the single-hop discipline; ratification (P0.4) re-promotes
    good facts that get caught in the cascade.

    Returns the list of edge UUIDs that were demoted. ``DISTINCT``
    matters: the undirected ``-[r]-`` pattern can yield the same edge
    from both traversal directions, and duplicate uuids inflate the
    demotion counts reported in ``DreamPassResult`` / the admin UI
    (the ``SET`` itself is idempotent).
    """
    query = """
    MATCH (e:Entity {uuid: $entity_uuid, group_id: $group_id})
    MATCH (e)-[r:RELATES_TO]-(other)
    SET r.expired_at = $now,
        r.status = 'superseded',
        r.expiration_reason = $reason
    RETURN DISTINCT r.uuid AS edge_uuid
    """
    try:
        records, _, _ = await driver.execute_query(
            query,
            entity_uuid=entity_uuid,
            group_id=group_id,
            reason=reason,
            now=_now_iso(),
        )
        return [r["edge_uuid"] for r in records]
    except Exception:
        logger.warning(
            "Failed to invalidate direct neighbors of entity %s in group %s",
            entity_uuid,
            group_id,
            exc_info=True,
        )
        return []


async def _hard_delete_edges(
    driver, uuids: list[str], user_id: str
) -> tuple[list[str], list[str]]:
    """Permanent removal — delete edges and clean up back-references.

    Uses graphiti's ``Edge.delete()`` pattern (handles MENTIONS,
    RELATES_TO, HAS_MEMBER in one query).  Does NOT delete orphaned
    entity nodes — they may have summaries, embeddings, or future
    connections.  Cleans up episode ``entity_edges`` back-references.
    """
    deleted = []
    failed = []
    for uuid in uuids:
        try:
            # Use WITH to capture the uuid before DELETE so we don't
            # access properties of deleted relationships (FalkorDB #1393).
            # Single atomic query avoids TOCTOU between check and delete.
            records, _, _ = await driver.execute_query(
                """
                MATCH ()-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->()
                WITH e.uuid AS uuid, e
                DELETE e
                RETURN uuid
                """,
                uuid=uuid,
            )
            if not records:
                failed.append(uuid)
                continue
            # Edge was deleted — report success regardless of cleanup outcome.
            deleted.append(uuid)
            # Clean up episode back-references (best-effort).
            try:
                await driver.execute_query(
                    """
                    MATCH (ep:Episodic)
                    WHERE $uuid IN ep.entity_edges
                    SET ep.entity_edges = [x IN ep.entity_edges WHERE x <> $uuid]
                    """,
                    uuid=uuid,
                )
            except Exception:
                logger.warning(
                    "Edge %s deleted but back-ref cleanup failed for user %s",
                    uuid,
                    user_id[:12],
                    exc_info=True,
                )
        except Exception:
            logger.warning(
                "Failed to hard-delete edge %s for user %s",
                uuid,
                user_id[:12],
                exc_info=True,
            )
            failed.append(uuid)
    return deleted, failed
