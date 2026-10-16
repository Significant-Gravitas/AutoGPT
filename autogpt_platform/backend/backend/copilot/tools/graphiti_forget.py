"""Two-step tool for targeted memory deletion.

Step 1 (memory_forget_search): search for matching facts, return candidates.
Step 2 (memory_forget_confirm): delete specific edges by UUID after user confirms.

Both steps go through the recall policy (``graphiti/recall.py`` and
``graphiti/recall_forget.py``): the candidates are the facts recall would
return, and, when asked (``include_forgotten``), the forgotten ones
(``recall.search_forgotten_facts``), so a soft forget can be made permanent;
a confirmed forget is ``recall_forget.retract``, with the guarantees and
limits ``graphiti/AGENTS.md`` lists.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core.edges import EntityEdge

from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.graphiti.memory_model import (
    ForgetResult,
    MemoryForgetFailure,
    MemoryForgetFailureCode,
)
from backend.copilot.graphiti.recall import (
    FORGOTTEN_FACT,
    search_facts,
    search_forgotten_facts,
)
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
                "include_forgotten": {
                    "type": "boolean",
                    "description": "Also find already-forgotten memories, to delete them permanently.",
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
        include_forgotten: bool = False,
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
            # offered for forgetting again, unless asked for, so a soft
            # forget can be made permanent.
            edges = await search_facts(memory_scope, query, limit=10)
            forgotten = (
                await search_forgotten_facts(memory_scope, query, limit=10)
                if include_forgotten
                else []
            )
        except Exception:
            logger.warning(
                "Memory forget search failed for user %s", user_id[:12], exc_info=True
            )
            return ErrorResponse(
                message="Memory search is temporarily unavailable.",
                session_id=session.session_id,
            )

        if not edges and not forgotten:
            return MemoryForgetCandidatesResponse(
                message="No matching memories found.",
                session_id=session.session_id,
                candidates=[],
            )

        candidates = [_candidate(edge) for edge in edges]
        candidates += [_forgotten_candidate(edge) for edge in forgotten]

        return MemoryForgetCandidatesResponse(
            message=_candidates_message(len(edges), len(forgotten)),
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
                    "title": "Memories",
                    "entity": None,
                },
                "hard_delete": {
                    "type": "boolean",
                    "description": "If true, permanently removes edges from the graph (GDPR). Default false (soft delete — marks as expired).",
                    "default": False,
                    "title": "Deletes permanently",
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


def _forgotten_candidate(edge: EntityEdge) -> dict[str, str]:
    """A forgotten fact as a candidate for a permanent delete: that it
    exists and when, never what it said. Its sentence, the audit copy its
    forget kept (``fact_redacted``), its scope and the dates it held true are
    all left out, the forget keeping them from the model; the delete needs
    only the uuid. The two dates are read as timestamps and written out
    anew (``_timestamp``), so no text an edge property holds goes with them."""
    forgotten_at = edge.attributes.get("forgotten_at") or edge.expired_at
    return {
        "uuid": edge.uuid,
        "fact": FORGOTTEN_FACT,
        "stored_at": _timestamp(edge.created_at),
        "forgotten_at": _timestamp(forgotten_at),
        "forgotten": "true",
    }


def _timestamp(value: Any) -> str:
    """``value`` as an ISO 8601 UTC timestamp to the second, or ``"unknown"``
    when it does not read as one (a missing value included)."""
    try:
        moment = datetime.fromisoformat(str(value))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OverflowError):
        return "unknown"


def _candidates_message(live: int, forgotten: int) -> str:
    """What ``memory_forget_search`` tells the model to do with its
    candidates: the live ones shown for the user to choose from, the
    forgotten ones offered for a permanent delete by count and date alone."""
    parts = []
    if live:
        parts.append(
            f"Found {live} candidate(s). Show these to the user and ask which to "
            "delete, then call tool:memory_forget_confirm with the UUIDs."
        )
    if forgotten:
        parts.append(
            f"{forgotten} memory(ies) matched that are already forgotten; what "
            "they said stays hidden, so do not guess at it. Tell the user how "
            "many and when they were stored and forgotten, and offer to delete "
            "them permanently: tool:memory_forget_confirm with their UUIDs and "
            "hard_delete=true."
        )
    return " ".join(parts)
