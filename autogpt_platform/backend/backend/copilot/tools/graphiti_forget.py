"""Two-step tool for targeted memory deletion.

Step 1 (memory_forget_search): search for matching facts, return candidates.
Step 2 (memory_forget_confirm): delete specific edges by UUID after user confirms.

Both steps go through the recall policy (``graphiti/recall.py`` and
``graphiti/recall_forget.py``): the candidates are the facts recall would
return, and a confirmed forget is ``recall_forget.retract``, with the
guarantees and limits ``graphiti/AGENTS.md`` lists.

``mark_edges_superseded`` below writes only over live facts, so a demotion
can never overwrite a user's forget. The dream's own writers, which also
leave alone a fact the user recalled within the protection window, are in
``graphiti/guarded_writes.py``.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal

from graphiti_core.edges import EntityEdge

from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.graphiti.memory_model import (
    ForgetResult,
    MemoryForgetFailure,
    MemoryForgetFailureCode,
)
from backend.copilot.graphiti.recall import live_fact_predicate, search_facts
from backend.copilot.graphiti.recall_forget import retract
from backend.copilot.graphiti.recall_render import fact_text, fact_validity
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import (
    ErrorResponse,
    MemoryForgetCandidatesResponse,
    MemoryForgetConfirmResponse,
    ToolResponseBase,
)

# A forget that found memory busy (another writer, usually an ingestion,
# holding the graph's write lock) is tried once more after this long.
_BUSY_RETRY_SECONDS = 5

# Cap on how many per-UUID failure reasons are inlined into the confirm
# message. Keeps a wholesale-failure batch from blowing past the tool-output
# size threshold (base.py) and losing all detail to truncation.
_MAX_FAILURE_DETAIL = 5


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
    """Search the current assistant's memories for deletion candidates."""

    @property
    def name(self) -> str:
        return "memory_forget_search"

    @property
    def description(self) -> str:
        return (
            "Search the current assistant's stored memories for a description so "
            "the user can choose which to delete. Returns candidate facts with UUIDs. "
            "Use tool:memory_forget_confirm with the UUIDs to actually delete them."
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
            memory_scope = MemoryScope.build(user_id, session.expert_id)
        except ValueError:
            return ErrorResponse(
                message="Invalid user ID for memory operations.",
                session_id=session.session_id,
            )

        try:
            # Only facts recall would still return: a forgotten one is not
            # offered for forgetting again.
            edges = await search_facts(memory_scope, query, limit=10)
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

        candidates = [_candidate(edge) for edge in edges]

        return MemoryForgetCandidatesResponse(
            message=f"Found {len(candidates)} candidate(s). Show these to the user and ask which to delete, then call tool:memory_forget_confirm with the UUIDs.",
            session_id=session.session_id,
            candidates=candidates,
        )


class MemoryForgetConfirmTool(BaseTool):
    """Delete edges from the current assistant's memory after confirmation.

    Supports both soft delete (temporal invalidation — reversible) and
    hard delete (remove from graph — irreversible, for GDPR).
    """

    @property
    def name(self) -> str:
        return "memory_forget_confirm"

    @property
    def description(self) -> str:
        return (
            "Delete specific memories from the current assistant by UUID. Use after "
            "memory_forget_search returns candidates and the user confirms which to "
            "delete. "
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
                message="At least one UUID is required. Use tool:memory_forget_search first.",
                session_id=session.session_id,
            )

        try:
            memory_scope = MemoryScope.build(user_id, session.expert_id)
        except ValueError:
            return ErrorResponse(
                message="Invalid user ID for memory operations.",
                session_id=session.session_id,
            )

        # A soft forget is a *system* retraction, not a world change: the
        # edge keeps its ``invalid_at``. See ``retract``.
        try:
            result = await _retract_once_more_if_busy(memory_scope, uuids, hard_delete)
        except Exception:
            logger.warning(
                "Memory forget failed for user %s", user_id[:12], exc_info=True
            )
            return ErrorResponse(
                message="Memory service is temporarily unavailable.",
                session_id=session.session_id,
            )

        mode = "permanently deleted" if hard_delete else "retracted from memory"
        return MemoryForgetConfirmResponse(
            message=_build_confirm_message(len(result.deleted), mode, result.failures),
            session_id=session.session_id,
            deleted_uuids=result.deleted,
            failed_uuids=[f.uuid for f in result.failures],
            failures=result.failures,
        )


async def _retract_once_more_if_busy(
    scope: MemoryScope, uuids: list[str], hard: bool
) -> ForgetResult:
    """``retract``, tried again once after ``_BUSY_RETRY_SECONDS`` when
    memory was busy; a forget that finds it busy writes nothing, so the
    retry is safe."""
    result = await retract(scope, uuids, hard=hard)
    if not any(f.code == MemoryForgetFailureCode.BUSY for f in result.failures):
        return result
    await asyncio.sleep(_BUSY_RETRY_SECONDS)
    return await retract(scope, uuids, hard=hard)


def _build_confirm_message(
    deleted_count: int, mode: str, failures: list[MemoryForgetFailure]
) -> str:
    """Human/model-readable summary that spells out *why* edges failed.

    A bare "N failed" gives the model nothing to act on (SECRT-2371); listing
    each UUID with its reason lets it retry, hard-delete, or tell the user.

    Only the first ``_MAX_FAILURE_DETAIL`` reasons are inlined: a large batch
    failing wholesale (e.g. a driver outage) would otherwise push the tool
    output past the persist-and-summarize threshold and lose *all* detail. The
    full per-UUID list stays available in the structured ``failures`` field.
    """
    summary = f"{deleted_count} memory edge(s) {mode}."
    if not failures:
        return summary
    shown = failures[:_MAX_FAILURE_DETAIL]
    detail = "; ".join(f"{f.uuid}: {f.reason}" for f in shown)
    remaining = len(failures) - len(shown)
    if remaining > 0:
        detail += f"; …and {remaining} more"
    return f"{summary} {len(failures)} failed — {detail}"


def _candidate(edge: EntityEdge) -> dict[str, str]:
    """One forget candidate, in the shape ``memory_forget_search`` returns."""
    valid_from, valid_to = fact_validity(edge)
    return {
        "uuid": edge.uuid,
        "fact": fact_text(edge),
        "valid_from": valid_from,
        "valid_to": valid_to,
    }


async def mark_edges_superseded(
    driver,
    uuids: list[str],
    reason: str,
    new_status: Literal["superseded", "contradicted"] = "superseded",
    user_id: str | None = None,
    group_id: str | None = None,
    expected_status: str | None = None,
) -> tuple[list[str], list[str]]:
    """Retract edges AND set the custom audit-trail ``status`` property.

    Retire the edge (``expired_at``; ``invalid_at`` is left alone) and
    stamp ``status='superseded'`` (or ``'contradicted'``) plus
    ``expiration_reason=<reason>`` so the demotion is queryable from
    search (``WHERE e.status = 'superseded'``). The dream pass's demotions
    and the ratification sweep's supersessions go through
    ``graphiti/guarded_writes.py`` instead, which adds the recall guard.

    ``group_id`` adds defense-in-depth: the driver is normally opened
    against the per-user FalkorDB database, but when provided the
    Cypher predicate also requires the edge's ``group_id`` to match so
    a future caller holding the wrong driver can't touch another
    user's edges. ``None`` keeps the unscoped match.

    The write lands only on a live fact (``recall.live_fact_predicate``), or,
    with ``expected_status``, on an unexpired, unforgotten edge still in that
    status, so a forget or other change made since the caller read the edge
    is never overwritten; an edge that no longer qualifies is reported failed.

    Returns ``(succeeded_uuids, failed_uuids)``.
    """
    deleted = []
    failed = []
    user_log = (user_id or "?")[:12]
    query = _supersede_query(
        scoped=group_id is not None, guarded=expected_status is not None
    )
    params: dict[str, str] = {"new_status": new_status, "reason": reason}
    if group_id is not None:
        params["group_id"] = group_id
    if expected_status is not None:
        params["expected_status"] = expected_status
    for uuid in uuids:
        try:
            records, _, _ = await driver.execute_query(
                query, uuid=uuid, now=_now_iso(), **params
            )
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


def _supersede_query(*, scoped: bool, guarded: bool) -> str:
    """``mark_edges_superseded``'s Cypher, with its ``group_id`` match and its
    ``expected_status`` guard when asked for."""
    edge_match = (
        "MATCH ()-[e:RELATES_TO {uuid: $uuid, group_id: $group_id}]->()"
        if scoped
        else "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()"
    )
    guard = (
        "WHERE e.status = $expected_status AND e.expired_at IS NULL"
        " AND e.forgotten_at IS NULL"
        if guarded
        else f"WHERE {live_fact_predicate('e')}"
    )
    return f"""
                {edge_match}
                {guard}
                SET e.expired_at = $now,
                    e.status = $new_status,
                    e.expiration_reason = $reason
                RETURN e.uuid AS uuid
                """
