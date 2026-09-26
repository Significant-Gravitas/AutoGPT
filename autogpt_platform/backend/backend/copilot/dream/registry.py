"""Durable per-scope registry of the dream-system crons.

One ``MemoryScopeSchedule`` row per memory scope (the account, or one hired
expert) is the source of truth for "is this scope scheduled": its state
(ACTIVE, PAUSED, WIPED), the timezone its crons were registered in and their
APScheduler job ids. The crons themselves are the rows of
``scheduling.DREAM_SYSTEM_JOBS``, keyed by scope: the account's scope key is
its user id, so its job ids are the historical per-user ones. The work
against the scheduler and the row lives in ``scope_jobs.py``.

Who calls what:

  * :func:`ensure_scope_scheduled` — the first memory write per group in a
    process (``graphiti/ingest.py``), the backfill (``registry_backfill.py``)
    and the functions below.
  * :func:`sync_expert_scope` — hire, archive, pause and resume of an expert.
  * :func:`reregister_user` — the owner's timezone changed.
  * :func:`mark_wiped` — for the scope wipe (workstream 3); nothing calls it
    yet, and it does not erase the graph itself.

State only changes through pause, resume and wipe: ensure never overrides a
PAUSED or WIPED scope, so a stray memory write cannot bring back the crons of
an archived expert. Everything here fails soft — logged, never raised —
because none of its callers may break over scheduling.
"""

import logging
from typing import Any

from prisma.enums import MemoryScopeScheduleState

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import memory_schedule_db

from .scheduling import DREAM_SYSTEM_JOBS, resolve_user_timezone
from .scope_jobs import (
    flag_gate,
    register_scope_jobs,
    remove_scope_jobs,
    scope_label,
    skip_all,
    state_reason,
)

logger = logging.getLogger(__name__)


async def ensure_scope_scheduled(
    scope: MemoryScope, *, force_refresh: bool = False
) -> dict[str, Any]:
    """Idempotently register every flag-enabled dream-system cron of ``scope``.

    Decides from the scope's registry row, never from Redis: a PAUSED or
    WIPED scope is left alone, and in an ACTIVE one a cron is (re-)registered
    only when it has no recorded job or the owner's timezone moved.
    ``force_refresh`` re-registers every enabled cron regardless.

    Returns one entry per cron, keyed by ``job_id_prefix``: ``None`` when it
    was already registered (no RPC made), ``{"skipped": True, "reason": ...}``
    when it was not registered (flag off, ``scope_paused``, ``scope_wiped``,
    ``timezone_lookup_failed``, ``registry_unavailable``,
    ``registration_failed``), otherwise the scheduler's own result.
    """
    results, enabled = await flag_gate(scope)
    if not enabled:
        return results
    try:
        row = await memory_schedule_db().get_scope_schedule(
            scope.owner_user_id, scope.scope_key
        )
    except Exception:
        logger.warning(f"Registry read failed for {scope_label(scope)}", exc_info=True)
        return results | skip_all(enabled, "registry_unavailable")
    if row is not None and row.state != MemoryScopeScheduleState.ACTIVE:
        return results | skip_all(enabled, state_reason(row))
    user_timezone = await resolve_user_timezone(scope.owner_user_id)
    if user_timezone is None:
        # "Unknown" is not "UTC": keep the existing crons until a later call
        # can read the real timezone.
        return results | skip_all(enabled, "timezone_lookup_failed")
    return results | await register_scope_jobs(
        scope, enabled, row, user_timezone, force_refresh
    )


async def ensure_dream_system_scheduled(
    user_id: str, *, force_refresh: bool = False
) -> dict[str, Any]:
    """The account's crons: :func:`ensure_scope_scheduled` for
    ``MemoryScope.for_user(user_id)``. ``{}`` for an empty or invalid id."""
    try:
        scope = MemoryScope.for_user(user_id)
    except ValueError:
        return {}
    return await ensure_scope_scheduled(scope, force_refresh=force_refresh)


async def sync_expert_scope(user_id: str, expert_id: str, *, active: bool) -> None:
    """Keep an expert's memory crons in step with its lifecycle.

    ``active=True`` (hired, re-hired, schedules resumed) resumes the scope,
    which registers it if it had no row; ``active=False`` (archived,
    schedules paused) pauses it.
    """
    try:
        scope = MemoryScope.for_expert(user_id, expert_id)
        if active:
            await resume_scope(scope)
        else:
            await pause_scope(scope)
    except Exception:
        # Both already fail soft; this keeps a bug in them from ever
        # failing the hire, archive, pause or resume that called in.
        logger.exception(f"Memory schedule sync failed for expert #{expert_id}")


async def pause_scope(scope: MemoryScope) -> bool:
    """Stop the scope's crons until :func:`resume_scope`.

    The row goes PAUSED (created that way if the scope had none, so a later
    memory write cannot register it) and the jobs are removed. Returns
    whether the row now says PAUSED.
    """
    return await _leave_active(scope, MemoryScopeScheduleState.PAUSED)


async def mark_wiped(scope: MemoryScope) -> bool:
    """Stop the crons of a scope whose memory was erased, until
    :func:`resume_scope`. Erasing the graph is the caller's job."""
    return await _leave_active(scope, MemoryScopeScheduleState.WIPED)


async def resume_scope(
    scope: MemoryScope, *, force_refresh: bool = False
) -> dict[str, Any]:
    """Undo :func:`pause_scope` or :func:`mark_wiped` and register the crons
    again; a scope with no row is simply registered."""
    try:
        await memory_schedule_db().set_scope_state(
            scope.owner_user_id, scope.scope_key, MemoryScopeScheduleState.ACTIVE
        )
    except Exception:
        logger.warning(f"Could not resume {scope_label(scope)}", exc_info=True)
        return skip_all(DREAM_SYSTEM_JOBS, "registry_unavailable")
    return await ensure_scope_scheduled(scope, force_refresh=force_refresh)


async def reregister_user(user_id: str) -> dict[str, dict[str, Any]]:
    """Re-register every scope of an owner whose timezone changed.

    The account always goes through, row or not: its crons may predate the
    registry and still run in the old timezone. Each expert with a row
    follows; paused and wiped ones are skipped and take the new timezone
    when they resume. An owner with every dream flag off costs no database
    call, as with :func:`ensure_scope_scheduled`. Keyed by scope key.
    """
    try:
        account = MemoryScope.for_user(user_id)
    except ValueError:
        return {}
    skipped, enabled = await flag_gate(account)
    if not enabled:
        return {account.scope_key: skipped}
    scopes = [account, *await _expert_scopes(user_id)]
    return {
        scope.scope_key: await ensure_scope_scheduled(scope, force_refresh=True)
        for scope in scopes
    }


async def _leave_active(scope: MemoryScope, state: MemoryScopeScheduleState) -> bool:
    """Set the row to ``state`` (creating it if missing), then remove the
    crons. On a failed write the crons stay: their bodies check the expert's
    lifecycle themselves, and removing them under a row that still says
    ACTIVE would hide them from the next ensure."""
    try:
        db = memory_schedule_db()
        if not await db.set_scope_state(scope.owner_user_id, scope.scope_key, state):
            user_timezone = await resolve_user_timezone(scope.owner_user_id)
            row = await db.claim_scope_schedule(
                scope.owner_user_id,
                scope.scope_key,
                scope.expert_id,
                user_timezone or "UTC",
                state,
            )
            if row.state != state:
                await db.set_scope_state(scope.owner_user_id, scope.scope_key, state)
    except Exception:
        logger.warning(
            f"Could not mark {scope_label(scope)} {state.value}", exc_info=True
        )
        return False
    await remove_scope_jobs(scope)
    logger.info(f"Dream-system: {scope_label(scope)} is now {state.value}")
    return True


async def _expert_scopes(user_id: str) -> list[MemoryScope]:
    try:
        rows = await memory_schedule_db().list_user_scope_schedules(user_id)
    except Exception:
        logger.warning(f"Could not list scopes of user {user_id[:12]}", exc_info=True)
        return []
    return [
        MemoryScope.for_expert(user_id, row.expert_id)
        for row in rows
        if row.expert_id is not None
    ]
