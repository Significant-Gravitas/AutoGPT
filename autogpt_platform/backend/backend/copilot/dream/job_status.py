"""Redis-backed status registry for fire-and-forget admin jobs.

Admin trigger endpoints used to block synchronously waiting for the
dream pass / nightly batch / community rebuild to finish — easy
5-minute httpx ``ReadTimeout`` when an LLM phase legitimately took
longer than the RPC timeout, plus no way to recover the result if the
client tab closed mid-run.

The new pattern: admin POST writes an initial :class:`JobStatus` row
keyed by ``job_id`` (UUID), kicks off the work via APScheduler's
``add_job(run_date=now)`` and returns 202 immediately. The work body
calls :func:`update_status_phase` at each phase transition and
:func:`mark_complete` / :func:`mark_errored` at the end. Admin UI
polls a GET endpoint that reads the row back. 6-hour TTL bounds
abandoned rows.

This module is dream-shaped (kinds are ``nightly``, ``dream_pass``,
``rebuild`` — the three long-running admin triggers we have today),
but the storage primitive is generic enough to extend to any future
fire-and-forget admin job.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ``JobStatus`` is parametrized by the concrete ``result`` payload type.
# Each job kind specializes T at the admin-route layer (e.g.
# ``DreamJobStatus = JobStatus[DreamPassResult]``) so FastAPI emits the
# full nested OpenAPI schema for the per-kind result without any
# parallel ``*Response`` shim classes.
ResultT = TypeVar("ResultT", bound=BaseModel)


# Key prefix lives at the dream namespace today because all three job
# kinds are dream-system-adjacent. If a non-dream caller ever needs
# this registry, hoist the prefix to ``admin:job:`` then.
STATUS_KEY_PREFIX = "dream:status"

# 6 hours. Longer than the dream lock TTL (1800s) so results stay
# visible after the work completes. An admin who triggers a run, walks
# away, and comes back hours later still sees the result.
STATUS_TTL_SECONDS = 6 * 60 * 60


JobKind = Literal["nightly", "dream_pass", "rebuild"]

JobState = Literal[
    "queued",  # initial state — admin endpoint wrote the row, scheduler hasn't picked it up yet
    "running",  # work body has started; phase transitions update current_phase
    "submitted",  # batch path — work body submitted to provider's batch API and is awaiting result
    "complete",  # terminal: result populated
    "errored",  # terminal: error populated
]


class JobStatus(BaseModel, Generic[ResultT]):
    """Status row for one fire-and-forget admin job.

    Stored as a JSON-serialized Redis string at
    ``dream:status:{kind}:{job_id}``. Each write is one ``SET`` of the
    whole row, so a reader never sees half of one. The helpers in this
    module read the row, compose their update and then SET it, and that
    read-modify-write is not atomic: two writers racing on one job_id can
    interleave, and the later SET wins over what the earlier wrote. What
    keeps a dream pass's job right is who writes it, not atomicity: the
    terminal guards (``mark_complete`` and ``mark_errored`` never rewrite a
    completed row), the rule that an errored row keeps the usage its
    result already carried (``_errored_result``), and the batch executor
    claiming a delivery before its handler or drop hook runs, so one walker
    ends a pass.

    Generic over ``ResultT`` (the work body's return shape). The Redis
    layer always reads and writes the row as JSON — the type parameter
    only kicks in at the read-back seam where ``read_status`` is
    parameterized by the caller (the admin route knows it expects a
    ``DreamPassResult``; the polling internals don't need to).
    """

    job_id: str
    user_id: str
    kind: JobKind
    state: JobState
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    current_phase: str | None = Field(
        default=None,
        description=(
            "Free-form phase label set by the work body. Dream pass uses "
            "'consolidate' | 'recombine' | 'sanitize' | 'apply'; nightly "
            "batch reuses those plus 'ratification'; community rebuild "
            "uses 'leiden' | 'summarize' | 'persist'."
        ),
    )
    # When set, work body submitted to a provider's batch API; the
    # eventual apply step lives in the batch result handler.
    batch_id: str | None = None
    # Populated at terminal state. ``result`` is the work body's return
    # value typed as ``ResultT``; ``error`` is a short human-readable
    # error string.
    result: ResultT | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key(kind: JobKind, job_id: str) -> str:
    return f"{STATUS_KEY_PREFIX}:{kind}:{job_id}"


async def write_initial_status(
    *, kind: JobKind, job_id: str, user_id: str
) -> JobStatus[Any]:
    """Write the first ``state='queued'`` row at admin-trigger time.

    Called from the admin endpoint before it returns 202. The work
    body is responsible for flipping state to ``running`` when it
    actually starts. ``result`` is ``None`` at this point — the type
    parameter is only meaningful once the work body produces a result.
    """
    from backend.data.redis_client import get_redis_async

    now = datetime.now(timezone.utc)
    status: JobStatus[Any] = JobStatus(
        job_id=job_id,
        user_id=user_id,
        kind=kind,
        state="queued",
        started_at=now,
        updated_at=now,
    )
    redis = await get_redis_async()
    await redis.set(
        _key(kind, job_id),
        status.model_dump_json(),
        ex=STATUS_TTL_SECONDS,
    )
    return status


async def update_status_phase(
    *,
    kind: JobKind,
    job_id: str,
    state: JobState | None = None,
    current_phase: str | None = None,
    batch_id: str | None = None,
) -> JobStatus[Any] | None:
    """Update one or more fields on an existing status row.

    Returns the updated status, or ``None`` if the row is missing (TTL
    expired or wrong job_id). Callers should treat ``None`` as a soft
    failure — log it and continue rather than raising, since failing
    to write a status update should never crash the work body itself.
    """
    existing = await read_status(kind=kind, job_id=job_id)
    if existing is None:
        logger.warning(
            "Tried to update missing status row kind=%s job_id=%s",
            kind,
            job_id[:12],
        )
        return None

    update: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
    if state is not None:
        update["state"] = state
    if current_phase is not None:
        update["current_phase"] = current_phase
    if batch_id is not None:
        update["batch_id"] = batch_id

    updated = existing.model_copy(update=update)
    await _persist(updated)
    return updated


async def mark_complete(
    *, kind: JobKind, job_id: str, result: BaseModel | dict[str, Any]
) -> JobStatus[Any] | None:
    """Transition to ``state='complete'`` with the work body's result.

    Accepts either a Pydantic model (preferred — preserves typed
    fields through JSON serialization) or a plain dict for legacy
    callers that build the payload by hand.

    Refuses terminal→terminal transitions (symmetric with
    ``mark_errored``): once a row is complete or errored, its result
    is authoritative and later writers must not rewrite it.
    """
    existing = await read_status(kind=kind, job_id=job_id)
    if existing is None:
        logger.warning(
            "mark_complete: missing status row kind=%s job_id=%s",
            kind,
            job_id[:12],
        )
        return None
    if existing.state in ("complete", "errored"):
        logger.warning(
            "mark_complete: refusing to overwrite terminal job kind=%s "
            "job_id=%s state=%s",
            kind,
            job_id[:12],
            existing.state,
        )
        return existing
    serialized = result.model_dump() if isinstance(result, BaseModel) else result
    now = datetime.now(timezone.utc)
    updated = existing.model_copy(
        update={
            "state": "complete",
            "updated_at": now,
            "completed_at": now,
            "result": serialized,
        }
    )
    await _persist(updated)
    return updated


async def mark_errored(
    *, kind: JobKind, job_id: str, error: str, result: BaseModel | None = None
) -> JobStatus[Any] | None:
    """Transition to ``state='errored'`` with a short error string, and the
    work body's *result* when it returned one (a failed dream pass's usage).

    Refuses to overwrite a job already in ``state='complete'`` — the
    batch tail runs cleanup after ``mark_complete``, and a transient
    error routed through the crash guard must not rewrite a completed
    job (whose writes and billing landed) as errored. A job errored again
    keeps the usage its result already carries when the new result has
    none: a failure replayed after cleanup knows less than the first.
    """
    existing = await read_status(kind=kind, job_id=job_id)
    if existing is None:
        logger.warning(
            "mark_errored: missing status row kind=%s job_id=%s",
            kind,
            job_id[:12],
        )
        return None
    if existing.state == "complete":
        logger.warning(
            "mark_errored: refusing to overwrite completed job kind=%s "
            "job_id=%s (error was: %s)",
            kind,
            job_id[:12],
            error[:200],
        )
        return existing
    now = datetime.now(timezone.utc)
    updated = existing.model_copy(
        update={
            "state": "errored",
            "updated_at": now,
            "completed_at": now,
            "error": error[:2000],  # cap; error strings can be huge stack traces
            "result": _errored_result(existing.result, result),
        }
    )
    await _persist(updated)
    return updated


def _errored_result(
    existing: dict[str, Any] | None, result: BaseModel | None
) -> dict[str, Any] | None:
    """The result an errored job keeps: *result*, but never without the
    usage the existing one carries."""
    if result is None:
        return existing
    replacement = result.model_dump()
    kept_usage = existing.get("usage") if existing is not None else None
    if replacement.get("usage") is None and kept_usage is not None:
        replacement["usage"] = kept_usage
    return replacement


async def read_status(*, kind: JobKind, job_id: str) -> JobStatus[Any] | None:
    """Fetch a status row by ``(kind, job_id)``. Returns ``None`` if missing.

    Returns the row with ``result`` as the raw JSON-decoded value
    (typically a ``dict`` from the work body's ``model_dump()``). Route
    handlers re-validate the row through their kind-specific
    ``JobStatus[T]`` subclass (e.g. ``DreamJobStatus``) at response
    construction time — that walks the dict and types ``result`` per
    the concrete subclass's parametrization, no extra read helper
    needed.
    """
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    raw = await redis.get(_key(kind, job_id))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return JobStatus[Any].model_validate(json.loads(raw))
    except Exception:
        logger.exception(
            "Corrupted job status row at key=%s — treating as missing",
            _key(kind, job_id),
        )
        return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _persist(status: JobStatus[Any]) -> None:
    """Serialize and write the whole row with its TTL in one SET. The write
    is atomic; the read-modify-write around it is not (``JobStatus``)."""
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    await redis.set(
        _key(status.kind, status.job_id),
        status.model_dump_json(),
        ex=STATUS_TTL_SECONDS,
    )
