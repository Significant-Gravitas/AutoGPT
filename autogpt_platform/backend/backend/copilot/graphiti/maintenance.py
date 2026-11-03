"""A dream-system cron's work, over every memory scope it is responsible for.

Personal scopes (the account, and each expert of the owner's personal
organization) have crons of their own (``copilot/dream/registry.py``): each
cron runs its own scope. A saved organization's scope (the owner's memory, or
an expert's, written while working in that organization) has none: the
account's cron runs it too, so the owner's memory in each organization they
work in is maintained, each scope only while the owner may still write to
that organization (``scope_resolution.memory_scope_access``). The scopes are
the persisted ones (``MemoryScopeRegistry``), never derived from navigation.
"""

import logging
from typing import Any

from backend.copilot.dream.nightly_batch import (
    NightlyBatchResult,
    run_nightly_batch_submit,
)
from backend.copilot.dream.pass_record import DreamTrigger
from backend.data.db_accessors import experts_db, memory_db

from .communities import rebuild_communities_for_user
from .scope import MemoryScope
from .scope_resolution import memory_scope_access

logger = logging.getLogger(__name__)

SCOPES_FAILED = "One or more memory maintenance scopes failed"


async def run_all_memory_nightly(
    user_id: str, *, expert_id: str | None = None, trigger: DreamTrigger = "cron"
) -> NightlyBatchResult:
    """The nightly batch of the cron's own scope, then, for the account's
    cron, of each saved organization's scope of the owner's. Each
    organization's result is kept under its scope key in ``scope_results``;
    a failed one marks the account's result errored. One the budget check
    stopped stops them all: the owner pays for every scope."""
    result = await run_nightly_batch_submit(
        user_id, expert_id=expert_id, trigger=trigger
    )
    if expert_id is not None or _stopped_by_budget(result):
        return result
    for scope in await _organization_scopes(user_id):
        try:
            async with memory_scope_access(scope) as allowed:
                if not allowed:
                    continue
                outcome = await allowed.run(
                    run_nightly_batch_submit(
                        user_id,
                        expert_id=scope.expert_id,
                        organization_id=scope.organization_id,
                        include_learning=False,
                        trigger=trigger,
                    )
                )
            result.scope_results[scope.scope_key] = outcome
            result.dream_in_flight |= outcome.dream_in_flight
            if _errored(outcome):
                result.error = _with_scope_failure(result.error)
        except Exception:
            logger.exception(f"Memory maintenance failed for scope {scope.scope_key}")
            result.error = _with_scope_failure(result.error)
    return result


async def rebuild_all_memory_communities(
    user_id: str, *, expert_id: str | None = None, force: bool = False
) -> dict[str, Any]:
    """The community rebuild of the cron's own scope, then, for the
    account's cron, of each saved organization's scope of the owner's, kept
    under its scope key in ``scopes``; a failed one sets the account's
    result's ``error``."""
    result = await rebuild_communities_for_user(
        user_id, expert_id=expert_id, force=force
    )
    if expert_id is not None:
        return result
    scopes: dict[str, dict[str, Any]] = {}
    for scope in await _organization_scopes(user_id):
        try:
            async with memory_scope_access(scope) as allowed:
                if not allowed:
                    continue
                outcome = await allowed.run(
                    rebuild_communities_for_user(
                        user_id,
                        expert_id=scope.expert_id,
                        organization_id=scope.organization_id,
                        force=force,
                    )
                )
            scopes[scope.scope_key] = outcome
            if outcome.get("error"):
                result["error"] = _with_scope_failure(result.get("error"))
        except Exception:
            logger.exception(f"Memory rebuild failed for scope {scope.scope_key}")
            result["error"] = _with_scope_failure(result.get("error"))
    result["scopes"] = scopes
    return result


async def _organization_scopes(user_id: str) -> list[MemoryScope]:
    """The owner's persisted memory scopes in a saved organization, the ones
    no cron of their own runs, that run tonight: an expert's only while the
    expert is active (``_expert_is_active``)."""
    return [
        scope
        for scope in await memory_db().list_memory_scopes(user_id)
        if scope.organization_id is not None
        and (scope.expert_id is None or await _expert_is_active(scope))
    ]


async def _expert_is_active(scope: MemoryScope) -> bool:
    """Whether the scope's expert may have its memory maintained: not
    archived and its schedules not paused (a manual pause or a budget
    breach), as a personal expert's own crons are held
    (``scheduler._expert_scope_status``). A lookup that fails skips it."""
    if scope.expert_id is None:
        return True
    try:
        expert = await experts_db().get_expert(
            scope.owner_user_id, scope.expert_id, include_workflows=False
        )
    except Exception:
        logger.warning(
            f"Memory maintenance skipped scope {scope.scope_key}: its expert "
            "could not be read",
            exc_info=True,
        )
        return False
    return (
        expert is not None
        and not expert.is_archived
        and expert.schedules_paused_at is None
    )


def _stopped_by_budget(result: NightlyBatchResult) -> bool:
    """Whether the pre-flight budget check stopped the run before any
    submitter ran: it skipped the run, or the budget could not be read
    (``nightly_batch._budget_stopped``). Either holds for every scope of the
    owner, who pays for all of them."""
    return result.skipped or (result.error or "").startswith("billing:")


def _errored(result: NightlyBatchResult) -> bool:
    """Whether a scope's run, or one of its submitters, reported an error."""
    return bool(
        result.error
        or (result.dream is not None and result.dream.error)
        or (result.ratification is not None and result.ratification.error)
    )


def _with_scope_failure(error: str | None) -> str:
    if error and SCOPES_FAILED in error:
        return error
    return " | ".join(part for part in (error, SCOPES_FAILED) if part)
