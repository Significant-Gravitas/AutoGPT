"""Durable per-scope registry of the dream-system crons.

One ``MemoryScopeSchedule`` row per memory scope (the account, or one hired
expert) is the source of truth for "is this scope scheduled": its state
(ACTIVE, PAUSED, WIPED), the timezone its crons were registered in and their
APScheduler job ids. The crons themselves are the rows of
``scheduling.DREAM_SYSTEM_JOBS``, keyed by scope: the account's scope key is
its user id, so its job ids are the historical per-user ones. The work
against the scheduler and the row lives in ``scope_jobs.py``.

Who calls what:

  * :func:`ensure_scope_scheduled` — ingest, when a memory group's queue is
    created (``graphiti/ingest.py``), the backfill (``registry_backfill.py``)
    and the functions below.
  * :func:`ensure_expert_scheduled` — a new hire or raise, in the background.
  * :func:`sync_expert_scope` — archive, pause, resume and revive.
  * :func:`reregister_user` — the owner's timezone changed.
  * :func:`mark_wiped` — a seam for the scope wipe (workstream 3); nothing
    calls it yet, and it does not erase the graph itself.

State only changes through pause, resume and wipe. Ensure never overrides a
PAUSED or WIPED scope, and resume only reactivates a PAUSED one, so neither
a stray memory write nor a late hire can bring back the crons of an archived
expert. Every dependency call runs under the registry deadline
(``deadline.py``), and everything here fails soft (logged, never raised),
because none of its callers may break or hang over scheduling.
"""

import logging
from typing import Any

from prisma.enums import MemoryScopeScheduleState

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import memory_schedule_db
from backend.data.memory_schedule import MemoryScopeSchedule

from .deadline import within_deadline
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
    ``scope_changed``, ``timezone_lookup_failed``, ``registry_unavailable``,
    ``registration_failed``, ``record_failed``), otherwise the scheduler's
    own result.
    """
    results, enabled = await flag_gate(scope)
    if not enabled:
        return results
    try:
        row = await within_deadline(
            memory_schedule_db().get_scope_schedule(
                scope.owner_user_id, scope.scope_key
            )
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


async def ensure_expert_scheduled(user_id: str, expert_id: str) -> None:
    """Register a newly hired or raised expert's crons.

    Runs in the background after the hire returns, so it may land after the
    expert was already archived or paused: like any ensure, it leaves a
    PAUSED or WIPED scope alone rather than resuming it.
    """
    try:
        await ensure_scope_scheduled(MemoryScope.for_expert(user_id, expert_id))
    except Exception:
        logger.exception(f"Memory schedule registration failed for #{expert_id}")


async def sync_expert_scope(user_id: str, expert_id: str, *, active: bool) -> None:
    """Keep an expert's memory crons in step with an explicit lifecycle change.

    ``active=True`` (schedules resumed, expert revived) resumes a PAUSED
    scope or registers one with no row; ``active=False`` (archived, schedules
    paused) pauses it. Called from API requests and the run-budget gate, so
    the whole change runs under the registry deadline; a change it cuts off
    is settled by the next registration and, meanwhile, by the cron gate.
    """
    try:
        scope = MemoryScope.for_expert(user_id, expert_id)
        change = resume_scope(scope) if active else pause_scope(scope)
        await within_deadline(change)
    except TimeoutError:
        logger.warning(f"Memory schedule sync timed out for expert #{expert_id}")
    except Exception:
        # Both already fail soft; this keeps a bug in them from ever
        # failing the archive, pause or resume that called in.
        logger.exception(f"Memory schedule sync failed for expert #{expert_id}")


async def pause_scope(scope: MemoryScope) -> bool:
    """Stop the scope's crons until :func:`resume_scope`.

    The row goes PAUSED (created that way if the scope had none, so a later
    registration cannot activate it) and the jobs are removed. True only if
    both happened.
    """
    return await _leave_active(scope, MemoryScopeScheduleState.PAUSED)


async def mark_wiped(scope: MemoryScope) -> bool:
    """Stop the crons of a scope whose memory was erased. :func:`resume_scope`
    does not bring a wiped scope back; the wipe work decides what does.
    Erasing the graph is the caller's job."""
    return await _leave_active(scope, MemoryScopeScheduleState.WIPED)


async def resume_scope(
    scope: MemoryScope, *, force_refresh: bool = False
) -> dict[str, Any]:
    """Undo :func:`pause_scope` and register the crons again; a scope with no
    row is simply registered. A WIPED scope stays WIPED."""
    try:
        await within_deadline(
            memory_schedule_db().set_scope_state(
                scope.owner_user_id,
                scope.scope_key,
                MemoryScopeScheduleState.ACTIVE,
                only_from=MemoryScopeScheduleState.PAUSED,
            )
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
    """Make the row say ``state``, then remove the crons; True only if both
    happened. When the state write fails the crons stay: removing them under
    a row that still says ACTIVE would hide them from the next ensure, and
    their bodies check the expert's lifecycle themselves."""
    if not await _establish_state(scope, state):
        return False
    if not await remove_scope_jobs(scope):
        return False
    logger.info(f"Dream-system: {scope_label(scope)} is now {state.value}")
    return True


async def _establish_state(scope: MemoryScope, state: MemoryScopeScheduleState) -> bool:
    """Set the row to ``state``, creating it in that state if it has none.

    A registration can claim the row ACTIVE between the first write and the
    claim, so whatever the claim returns short of ``state``, the state is
    then set explicitly.
    """
    db = memory_schedule_db()
    try:
        if await within_deadline(
            db.set_scope_state(scope.owner_user_id, scope.scope_key, state)
        ):
            return True
        claimed = await _claim_in_state(scope, state)
        if claimed is not None and claimed.state == state:
            return True
        return await within_deadline(
            db.set_scope_state(scope.owner_user_id, scope.scope_key, state)
        )
    except Exception:
        logger.warning(
            f"Could not mark {scope_label(scope)} {state.value}", exc_info=True
        )
        return False


async def _claim_in_state(
    scope: MemoryScope, state: MemoryScopeScheduleState
) -> MemoryScopeSchedule | None:
    """Create the scope's row in ``state``; the row as it stands when one
    already existed, and None when the claim failed."""
    try:
        user_timezone = await resolve_user_timezone(scope.owner_user_id)
        return await within_deadline(
            memory_schedule_db().claim_scope_schedule(
                scope.owner_user_id,
                scope.scope_key,
                scope.expert_id,
                user_timezone or "UTC",
                state,
            )
        )
    except Exception:
        logger.warning(f"Could not claim {scope_label(scope)}", exc_info=True)
        return None


async def _expert_scopes(user_id: str) -> list[MemoryScope]:
    try:
        rows = await within_deadline(
            memory_schedule_db().list_user_scope_schedules(user_id)
        )
    except Exception:
        logger.warning(f"Could not list scopes of user {user_id[:12]}", exc_info=True)
        return []
    return [
        MemoryScope.for_expert(user_id, row.expert_id)
        for row in rows
        if row.expert_id is not None
    ]
