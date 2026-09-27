"""Writes the durable record of every dream pass as the pass moves.

Both routes write here, through ``db_accessors.dream_db()``: the Prisma module
where it is connected, the DatabaseManager RPC in the scheduler and the batch
executor. The sync orchestrator records the start with its lease, the
gathered window, each phase's output, each renewal of the lease, the
operations it is about to apply and how the pass ended. The batch path
records the submit, each phase that lands and the lease it renewed, each next
batch, the apply and the end, with what the landed phases used; every end
clears the lease and the input bundle. What each step writes is
``pass_record.py``; how a write lands on the row (forward only, JSON merged in
the database) is ``backend/data/dream_pass_update.py``.

A record write is best-effort and bounded. Each one gets
``RECORD_WRITE_TIMEOUT_SECONDS``, whatever the client does inside it
(connecting, retrying): a write that fails or runs out of time is logged at
warning and dropped, and the pass goes on. So a stalled DatabaseManager costs
a pass at most that deadline per write, and the nightly fan-out and the batch
walker keep moving.

The row is additive to the Redis state: the lock, the batch state and the
admin job status keep their keys, TTLs, scripts and formats. What changed is
timing: the pass now waits on these bounded writes before it locks, inside the
lock, before it chains a batch or applies, and before it cleans up.

``read_dream_pass`` is the read side, for the admin API and the eval driver.
The guard, the stop checks and the cancel read and write here too, under the
same deadline, but a failure reaches them: each decides for itself whether to
go on (``guard.py``, ``cancel.py``). So do the reaper and the retention job,
which read and delete across users (``reaper.py``, ``retention.py``).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TypeVar

from pydantic import BaseModel

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import dream_db
from backend.data.dream_pass_models import DreamPassRecord, DreamPassUpdate

from .fetch import DreamInput
from .pass_record import (
    DreamTrigger,
    applying,
    expired,
    failed,
    gathered,
    handed_to_batch,
    lease,
    new_pass,
    next_batch,
    outcome,
    phase_output,
    submitted,
)
from .pass_run import DreamPassRun
from .schemas import DreamOperations, DreamPassResult, DreamPassUsage, DreamPhase

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# How long one record write may take, retries included, before the pass
# moves on without it.
RECORD_WRITE_TIMEOUT_SECONDS = 10.0


async def start_pass(
    run: DreamPassRun, scope: MemoryScope, *, trigger: DreamTrigger
) -> None:
    """Insert the run's row: running, gathering its input, holding the lease
    of the lock it is about to take."""
    try:
        draft = new_pass(
            run.pass_id,
            scope,
            route=run.execution_path,
            trigger=trigger,
            started_at=run.started_at,
            lease_token=run.lease_token,
        )
        await _bounded(dream_db().create_dream_pass(draft))
    except Exception:
        logger.warning(
            f"Dream pass {run.pass_id}: could not insert its record; the pass goes on",
            exc_info=True,
        )


async def record_lease(pass_id: str, lease_token: str, ttl_seconds: int) -> bool | None:
    """The pass renewed its lock for *ttl_seconds* (see ``_write``)."""
    return await _write(pass_id, "the lease", lambda: lease(lease_token, ttl_seconds))


async def record_gathered(pass_id: str, input_bundle: DreamInput) -> None:
    await _write(pass_id, "the gathered window", lambda: gathered(input_bundle))


async def record_phase_output(
    pass_id: str, phase: DreamPhase, output: BaseModel
) -> None:
    await _write(pass_id, f"the {phase} output", lambda: phase_output(phase, output))


async def record_applying(pass_id: str, ops: DreamOperations) -> None:
    await _write(pass_id, "the start of apply", lambda: applying(ops))


async def record_submitted(
    pass_id: str,
    *,
    input_bundle: DreamInput,
    provider_batch_id: str,
    lease_token: str,
    lease_ttl_seconds: int,
) -> bool | None:
    """The batch submit, and whether it landed (see ``_write``): the handoff
    never hands the lock on once the row has refused it."""
    return await _write(
        pass_id,
        "the batch submit",
        lambda: submitted(
            input_bundle=input_bundle,
            provider_batch_id=provider_batch_id,
            lease_token=lease_token,
            lease_ttl_seconds=lease_ttl_seconds,
        ),
    )


async def record_next_batch(
    pass_id: str, phase: DreamPhase, provider_batch_id: str
) -> None:
    await _write(
        pass_id, "the next batch", lambda: next_batch(phase, provider_batch_id)
    )


async def record_sync_outcome(result: DreamPassResult) -> None:
    """How a pass the orchestrator ran ended. A pass it handed to the batch
    route stays open: the batch callbacks write its end."""
    if handed_to_batch(result):
        return
    await _write(result.pass_id, "the outcome", lambda: outcome(result, result.usage))


async def record_batch_complete(
    result: DreamPassResult, usage: DreamPassUsage | None
) -> None:
    """A batch pass applied: *result* holds what apply reported, *usage* what
    its phases used."""
    await _write(result.pass_id, "the outcome", lambda: outcome(result, usage))


async def record_batch_failed(
    pass_id: str, error: str, usage: DreamPassUsage | None
) -> None:
    """A batch pass failed; *usage* is what its landed phases used."""
    await _write(
        pass_id,
        "the failure",
        lambda: failed(error, usage, datetime.now(timezone.utc)),
    )


async def record_expired(pass_id: str, error: str) -> None:
    """A delivery of the pass closes its row EXPIRED with *error*: one that
    found its apply claimed by an earlier delivery that never finished."""
    await _write(pass_id, "the expiry", lambda: expired(error, not_updated_since=None))


async def read_dream_pass(pass_id: str, *, user_id: str) -> DreamPassRecord | None:
    """The pass's row when *user_id* owns it, else ``None``. Unlike the
    writes, a read that fails or runs out of time raises: its caller is
    answering a request."""
    return await _bounded(dream_db().get_dream_pass_for_user(pass_id, user_id))


async def read_pass(
    pass_id: str, *, timeout: float | None = None
) -> DreamPassRecord | None:
    """The pass's row, or ``None`` when it was never inserted. Raises when
    the read fails or runs out of *timeout* (the write deadline by
    default)."""
    return await _bounded(dream_db().get_dream_pass(pass_id), timeout=timeout)


async def read_open_passes(
    scope: MemoryScope, *, limit: int | None = None
) -> list[DreamPassRecord]:
    """The scope's passes that are still open, newest first, at most *limit*
    when given. Raises when the read fails or runs out of time."""
    return await _bounded(
        dream_db().list_open_dream_passes(scope.scope_key, limit=limit)
    )


async def read_expired_passes(
    expired_before: datetime, *, limit: int
) -> list[DreamPassRecord]:
    """Open passes, of every user, whose lease lapsed before
    *expired_before*, the oldest lapse first. Raises when the read fails or
    runs out of time."""
    return await _bounded(
        dream_db().list_expired_dream_passes(expired_before, limit=limit)
    )


async def read_user_passes(
    user_id: str, *, open_only: bool, limit: int
) -> list[DreamPassRecord]:
    """*user_id*'s passes, every scope, newest first; only the open ones when
    *open_only*. Raises when the read fails or runs out of time."""
    return await _bounded(
        dream_db().list_dream_passes(user_id, limit=limit, open_only=open_only)
    )


async def delete_old_passes(
    created_before: datetime, *, limit: int, timeout: float
) -> int:
    """Delete at most *limit* closed passes created before *created_before*
    and say how many went; under *timeout*, a delete being slower than a
    record write. Raises when it fails or runs out of time."""
    return await _bounded(
        dream_db().delete_old_dream_passes(created_before, limit=limit),
        timeout=timeout,
    )


async def write_stop(pass_id: str, update: DreamPassUpdate) -> bool:
    """Write a stop from outside the pass (a cancel, an expiry) and say
    whether it landed, which its caller acts on, unlike a pass's own writes.
    Raises when the write fails or runs out of time."""
    return await _bounded(dream_db().update_dream_pass(pass_id, update))


async def _write(
    pass_id: str, step: str, build: Callable[[], DreamPassUpdate]
) -> bool | None:
    """Write one transition, bounded, building it inside the guard so no part
    of it can fail the pass: ``True`` when it landed, ``False`` when the row
    refused it, ``None`` when it failed or ran out of time."""
    try:
        written = await _bounded(dream_db().update_dream_pass(pass_id, build()))
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not record {step}; the pass goes on",
            exc_info=True,
        )
        return None
    if not written:
        # Expected, not a fault: the row is closed (a late or repeated
        # delivery, which the transition rules refuse by design) or was
        # never inserted (that failure was logged when it happened).
        logger.debug("Dream pass %s: no open record to write %s to", pass_id, step)
    return written


async def _bounded(call: Awaitable[_T], timeout: float | None = None) -> _T:
    """*call*, or ``TimeoutError`` once it has taken *timeout* (the write
    deadline by default); the abandoned call is cancelled, not left running."""
    return await asyncio.wait_for(call, timeout=timeout or RECORD_WRITE_TIMEOUT_SECONDS)
