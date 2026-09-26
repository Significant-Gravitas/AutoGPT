"""Writes the durable record of every dream pass as the pass moves.

Both routes write here, through ``db_accessors.dream_db()``: the Prisma module
where it is connected, the DatabaseManager RPC in the scheduler and the batch
executor. The sync orchestrator records the start, the gathered window, each
phase's output, the operations it is about to apply and how the pass ended.
The batch path records the submit, each phase that lands, each next batch, the
apply and the end, with what the landed phases used. What each step writes is
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
    failed,
    gathered,
    handed_to_batch,
    new_pass,
    next_batch,
    outcome,
    phase_output,
    submitted,
)
from .routing import ExecutionPath
from .schemas import DreamOperations, DreamPassResult, DreamPassUsage, DreamPhase

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# How long one record write may take, retries included, before the pass
# moves on without it.
RECORD_WRITE_TIMEOUT_SECONDS = 10.0


async def start_pass(
    pass_id: str,
    scope: MemoryScope,
    *,
    route: ExecutionPath,
    trigger: DreamTrigger,
    started_at: datetime,
) -> None:
    """Insert the pass's row: running, gathering its input."""
    try:
        draft = new_pass(
            pass_id, scope, route=route, trigger=trigger, started_at=started_at
        )
        await _bounded(dream_db().create_dream_pass(draft))
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not insert its record; the pass goes on",
            exc_info=True,
        )


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
) -> None:
    await _write(
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


async def read_dream_pass(pass_id: str, *, user_id: str) -> DreamPassRecord | None:
    """The pass's row when *user_id* owns it, else ``None``. Unlike the
    writes, a failed read raises: its caller is answering a request."""
    return await dream_db().get_dream_pass_for_user(pass_id, user_id)


async def _write(pass_id: str, step: str, build: Callable[[], DreamPassUpdate]) -> None:
    """Write one transition, bounded, building it inside the guard so no part
    of it can fail the pass."""
    try:
        written = await _bounded(dream_db().update_dream_pass(pass_id, build()))
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not record {step}; the pass goes on",
            exc_info=True,
        )
        return
    if not written:
        # Expected, not a fault: the row is closed (a late or repeated
        # delivery, which the transition rules refuse by design) or was
        # never inserted (that failure was logged when it happened).
        logger.debug("Dream pass %s: no open record to write %s to", pass_id, step)


async def _bounded(call: Awaitable[_T]) -> _T:
    """*call*, or ``TimeoutError`` once it has taken the write deadline; the
    abandoned call is cancelled, not left running."""
    return await asyncio.wait_for(call, timeout=RECORD_WRITE_TIMEOUT_SECONDS)
