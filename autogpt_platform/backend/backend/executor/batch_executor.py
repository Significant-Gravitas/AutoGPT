"""Background service that polls submitted LLM batches and dispatches results.

The dream-pass orchestrator (and any future server-side batch caller)
submits requests to Anthropic / OpenAI batch APIs via
``backend/util/llm/providers.call_provider(execution_mode="batch")``.
Submissions return immediately with a ``BatchSubmissionRef``; results
arrive asynchronously up to ~24h later. This service owns:

  * **Polling**: walks a Redis-backed pending queue with exponential
    backoff (30s → 60s → 120s → 300s cap). Anthropic's typical
    completion is <30min so the early-aggressive ramp pays off.
  * **Dispatch**: when a batch reaches ``ended``, downloads results
    via ``providers.download_batch_results`` and routes each row to
    a caller-registered handler keyed on the entry's
    ``callback_namespace`` (e.g. ``"dream_pass"``). Handlers run the
    apply step, write cost-log rows, and update whatever JobStatus
    row the caller cares about.

Lives at the same architectural level as ``Scheduler`` — own process,
own Redis connection, own service-port for health checks. Subscribing
to a queue rather than driving submissions itself keeps the contract
small: callers don't care which subprocess owns the poll loop, just
that results land via their registered handler.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field, field_validator

from backend.util.llm.providers import (
    BatchResultRow,
    ProviderLiteral,
    download_batch_results,
    poll_batch,
)
from backend.util.service import AppService, UnhealthyServiceError
from backend.util.settings import Config

logger = logging.getLogger(__name__)

config = Config()


# Redis key for the pending-batches hash. Each field is one
# ``provider_batch_id``; each value is a JSON-serialized ``PendingEntry``.
# Both keys share the ``{llm:batch}`` hash tag so they land on the same
# Redis Cluster slot — required by ``claim_batch_dispatch_atomic``'s
# multi-key Lua script. Don't add or rename either without also updating
# the tag on the other.
PENDING_KEY = "{llm:batch}:pending"
# Tombstones for batches we've already started dispatching: one STRING
# key per batch — ``{llm:batch}:dispatched:<batch_id>`` — each with its
# own 7-day TTL so tombstones self-expire instead of accumulating in a
# shared set forever. 7 days is the dedup window: longer than any
# realistic Anthropic in-flight lifetime (24h SLA, our
# MAX_BATCH_LIFETIME_SECONDS cap is also 24h). The ``{llm:batch}``
# hash tag keeps every tombstone on the same cluster slot as the
# pending hash — required by ``claim_batch_dispatch_atomic``'s
# multi-key Lua script.
DISPATCHED_KEY_PREFIX = "{llm:batch}:dispatched"
DISPATCHED_TTL_SECONDS = 7 * 24 * 60 * 60

# 24h hard ceiling — Anthropic's promised SLA. Beyond this we mark the
# entry failed even if the provider still says ``processing``.
MAX_BATCH_LIFETIME_SECONDS = 24 * 60 * 60

# Poll cadence (exponential backoff). The BatchExecutor walks the queue
# every ``WALK_INTERVAL_SECONDS`` but each individual entry is only
# polled when its ``next_poll_at`` has elapsed.
WALK_INTERVAL_SECONDS = 10
INITIAL_POLL_DELAY_SECONDS = 30
MAX_POLL_DELAY_SECONDS = 300


# ---------------------------------------------------------------------------
# Pending-entry persistence
# ---------------------------------------------------------------------------


class PendingEntry(BaseModel):
    """One submitted batch we're waiting on.

    ``callback_namespace`` is the dispatcher's primary routing key:
    ``"dream_pass"`` → ``dream.batch_callbacks.handle``,
    ``"copilot_chat"`` → (future, copilot chat batch handler), etc.
    ``payload`` is opaque to the BatchExecutor — the namespace handler
    knows what shape its callers stored there (typically user_id,
    pass_id, job_id, phase, ...).
    """

    provider: ProviderLiteral
    provider_batch_id: str
    callback_namespace: str
    submitted_at: datetime
    next_poll_at: datetime
    poll_delay_seconds: int = INITIAL_POLL_DELAY_SECONDS
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("submitted_at", "next_poll_at")
    @classmethod
    def _coerce_naive_to_utc(cls, value: datetime) -> datetime:
        """Naive datetimes poison the walk: comparing them against an
        aware ``now`` raises TypeError on every walk, forever, because
        the bad entry is never removed. Coerce at the model boundary so
        a careless producer (the queue is an open contract) can't write
        the poison — and ``_deserialize`` can't re-hydrate it."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


async def enqueue_pending(entry: PendingEntry) -> None:
    """Persist a pending batch to the Redis hash.

    Callers invoke this right after their ``call_provider(execution_mode=
    "batch")`` returns a ``BatchSubmissionRef``. The BatchExecutor's
    walk picks it up within ``WALK_INTERVAL_SECONDS``.
    """
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    # redis-py's typed stubs mark hash ops as returning bare values on
    # AsyncRedisCluster — same workaround used in
    # ``copilot/stream_registry.py``.
    await redis.hset(  # type: ignore[misc]
        PENDING_KEY,
        entry.provider_batch_id,
        _serialize(entry),
    )


async def list_pending() -> list[PendingEntry]:
    """Return every pending batch in the queue."""
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    raw = await redis.hgetall(PENDING_KEY)  # type: ignore[misc]
    entries: list[PendingEntry] = []
    for batch_id, body in (raw or {}).items():
        try:
            entries.append(_deserialize(batch_id, body))
        except Exception:
            logger.exception(
                "Corrupted pending-batch row at %s — leaving in place",
                batch_id,
            )
    return entries


async def update_pending(entry: PendingEntry) -> None:
    """Rewrite an existing entry — used to advance ``next_poll_at`` /
    ``poll_delay_seconds`` after a poll that returned ``processing``."""
    await enqueue_pending(entry)


async def remove_pending(provider_batch_id: str) -> None:
    """Drop an entry from the queue without dispatching.

    Never used to *bypass* the dispatch claim: ended, failed, and
    lifetime-timeout all go through ``_claim_dispatch``, which
    atomically tombstones + HDELs in one Lua script so concurrent
    walkers (or a crash-replay) can never double-dispatch. The walk
    loop's only callers are the refused-claim zombie cleanups on the
    ended, failed, and lifetime-timeout paths — clearing a pending row
    whose batch already carries a dispatch tombstone.
    """
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    await redis.hdel(PENDING_KEY, provider_batch_id)  # type: ignore[misc]


async def _claim_dispatch(provider_batch_id: str) -> bool:
    """Atomically claim the right to dispatch results for this batch.

    Returns True when the caller won the claim (must dispatch); False
    when another walker already claimed it (must skip silently — the
    other walker is responsible for the callback). The Lua script
    inside :func:`claim_batch_dispatch_atomic` SETs a per-batch
    tombstone key (``NX EX`` so each one self-expires) + HDELs the
    pending entry in one indivisible step, closing the race we'd
    otherwise have between ``_dispatch`` and a separate
    ``remove_pending``.
    """
    from backend.data.redis_client import get_redis_async
    from backend.data.redis_helpers import claim_batch_dispatch_atomic

    redis = await get_redis_async()
    return await claim_batch_dispatch_atomic(
        redis,
        pending_key=PENDING_KEY,
        dispatched_key_prefix=DISPATCHED_KEY_PREFIX,
        batch_id=provider_batch_id,
        ttl_seconds=DISPATCHED_TTL_SECONDS,
    )


# ---------------------------------------------------------------------------
# Callback registry
# ---------------------------------------------------------------------------


# A handler takes (entry, result_rows) and returns when apply is done.
# It is the handler's responsibility to log costs, update any caller-
# supplied job status, and decide what to do with errored rows.
BatchResultHandler = Callable[[PendingEntry, list[BatchResultRow]], Awaitable[None]]

_HANDLERS: dict[str, BatchResultHandler] = {}


def register_handler(namespace: str, handler: BatchResultHandler) -> None:
    """Register a result handler for one namespace.

    Called at module-import time from the caller's ``__init__.py``
    (or from the BatchExecutor's bootstrap). Re-registering a
    namespace overwrites — last registration wins.
    """
    _HANDLERS[namespace] = handler


def clear_handlers_for_test() -> None:
    """Used by tests to reset the registry between cases."""
    _HANDLERS.clear()


# ---------------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------------


async def walk_once(api_key_for: Callable[[ProviderLiteral], str | None]) -> None:
    """One walk of the pending queue.

    For each entry whose ``next_poll_at`` has elapsed:
      * Poll the provider for status.
      * If ``ended``: download results, dispatch to the namespace
        handler, remove from queue.
      * If ``failed``: dispatch an error result, remove from queue.
      * If ``processing`` / ``pending``: bump ``poll_delay_seconds``
        (exponential backoff, capped at ``MAX_POLL_DELAY_SECONDS``)
        and write back.
      * If past ``MAX_BATCH_LIFETIME_SECONDS``: dispatch a timeout
        error, remove from queue (claimed atomically, exactly like the
        ended / failed paths).

    The API key is looked up by provider via the caller-supplied
    factory so this module doesn't need to know how settings are
    organized.
    """
    now = datetime.now(timezone.utc)
    entries = await list_pending()
    for entry in entries:
        try:
            await _walk_entry(entry, now=now, api_key_for=api_key_for)
        except Exception:
            # One poisoned entry must not starve the rest of the queue:
            # without this guard an uncaught per-entry error aborts the
            # walk, and every entry after it in hash order is never
            # polled again — the bad entry is still pending next walk.
            logger.exception(
                "Failed processing pending batch %s — skipping this walk",
                entry.provider_batch_id,
            )


async def _walk_entry(
    entry: PendingEntry,
    *,
    now: datetime,
    api_key_for: Callable[[ProviderLiteral], str | None],
) -> None:
    """Process a single pending entry — gates, poll, then dispatch."""
    if (now - entry.submitted_at).total_seconds() > MAX_BATCH_LIFETIME_SECONDS:
        await _handle_timeout(entry)
        return

    if entry.next_poll_at > now:
        return

    if entry.callback_namespace not in _HANDLERS:
        # Never claim (= tombstone + irrecoverably drop) results we
        # cannot deliver. An unregistered namespace usually means a
        # deploy whose callback module failed to import (_bootstrap_
        # callbacks swallows import errors) — leave the entry pending
        # with backoff so a fixed deploy can still dispatch it within
        # the provider's retention window. This can't loop forever:
        # an entry whose handler never shows up eventually crosses
        # MAX_BATCH_LIFETIME_SECONDS and the timeout path above claims
        # + removes it.
        logger.error(
            "No handler registered for namespace=%s — leaving batch %s pending",
            entry.callback_namespace,
            entry.provider_batch_id,
        )
        await _push_back(entry)
        return

    api_key = api_key_for(entry.provider)
    if not api_key:
        logger.warning(
            "No API key for provider=%s — leaving batch %s in queue",
            entry.provider,
            entry.provider_batch_id,
        )
        entry.next_poll_at = _next_poll_at(entry.poll_delay_seconds)
        await update_pending(entry)
        return

    try:
        status = await poll_batch(
            provider=entry.provider,
            provider_batch_id=entry.provider_batch_id,
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning(
            "Poll failed for batch %s: %s — will retry",
            entry.provider_batch_id,
            exc,
            exc_info=True,
        )
        await _push_back(entry)
        return

    if status == "ended":
        try:
            rows = await download_batch_results(
                provider=entry.provider,
                provider_batch_id=entry.provider_batch_id,
                api_key=api_key,
            )
        except Exception:
            logger.exception(
                "Download failed for batch %s — will retry once",
                entry.provider_batch_id,
            )
            await _push_back(entry)
            return
        # Atomic claim: only one walker dispatches each batch.
        # Without this, a slow remove_pending or a crash between
        # _dispatch and remove_pending re-fires the whole callback
        # chain (re-submits phases, re-runs apply, etc). The Lua
        # script tombstones the batch_id and HDELs the pending
        # entry in one indivisible step.
        if not await _claim_dispatch(entry.provider_batch_id):
            logger.info(
                "Batch %s already dispatched — skipping",
                entry.provider_batch_id,
            )
            # Refused claim = the per-batch tombstone exists, so this
            # pending row is a zombie (same reasoning as the timeout
            # path in _handle_timeout). Clear it — left alone it would
            # be re-polled AND its results re-downloaded on every walk
            # until the lifetime timeout finally reaps it.
            await remove_pending(entry.provider_batch_id)
            return
        await _dispatch(entry, rows)
    elif status == "failed":
        if not await _claim_dispatch(entry.provider_batch_id):
            # Refused claim = zombie row — clear without dispatching,
            # same as the ended / timeout paths.
            await remove_pending(entry.provider_batch_id)
            return
        await _dispatch_error(entry, error="provider reported failed")
    else:
        await _push_back(entry)


async def _handle_timeout(entry: PendingEntry) -> None:
    """Terminal path for entries past ``MAX_BATCH_LIFETIME_SECONDS``."""
    logger.warning(
        "Batch %s for namespace=%s exceeded %ds — dispatching timeout",
        entry.provider_batch_id,
        entry.callback_namespace,
        MAX_BATCH_LIFETIME_SECONDS,
    )
    # Claim before dispatching — the same atomic tombstone + HDEL as
    # the ended / failed paths — so a racing walker or a crash-replay
    # can never fire the timeout callback twice (the dream handler
    # releases the user's dream lock on error; a double fire could
    # delete a NEWER pass's lock acquired after the first release).
    # This path is also the backstop for entries whose namespace never
    # got a handler registered: the claim still removes the entry, and
    # _dispatch_error no-ops when the handler is absent.
    if not await _claim_dispatch(entry.provider_batch_id):
        # Refused claim = this batch already dispatched once, so the
        # pending row is a zombie (e.g. re-enqueued by a buggy
        # producer). Clear it WITHOUT dispatching — left alone it
        # would be re-walked until the tombstone expires, after which
        # the claim would win and fire a week-late duplicate dispatch.
        await remove_pending(entry.provider_batch_id)
        return
    await _dispatch_error(
        entry,
        error="exceeded MAX_BATCH_LIFETIME_SECONDS without completion",
    )


async def _push_back(entry: PendingEntry) -> None:
    """Re-queue *entry* with exponential backoff for a later walk."""
    entry.poll_delay_seconds = _bump_delay(entry.poll_delay_seconds)
    entry.next_poll_at = _next_poll_at(entry.poll_delay_seconds)
    await update_pending(entry)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class BatchExecutor(AppService):
    """Subprocess service that runs the BatchExecutor's poll loop.

    Lives alongside ``Scheduler``, ``ExecutionManager``, etc. in
    ``backend/app.py``. The service has no RPC surface today — callers
    interact via the Redis-backed pending queue, not through methods.
    Health check just confirms the loop thread is alive.
    """

    _loop_thread: threading.Thread | None = None
    _shutdown_event: threading.Event

    @classmethod
    def get_port(cls) -> int:
        return config.batch_executor_port

    async def health_check(self) -> str:
        if self._loop_thread is None or not self._loop_thread.is_alive():
            raise UnhealthyServiceError("BatchExecutor loop thread is not running")
        return await super().health_check()

    def run_service(self):
        from dotenv import load_dotenv

        load_dotenv()
        self._shutdown_event = threading.Event()
        # Eagerly import callback modules so their handlers are
        # registered before the first walk.
        _bootstrap_callbacks()

        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="BatchExecutorLoop",
        )
        self._loop_thread.start()
        logger.info("BatchExecutor poll loop started")
        # Hand off to the base class which pumps the shared event loop
        # so uvicorn's serve() coroutine (submitted from the FastAPI
        # init thread) can actually bind the RPC port + handle health
        # checks. Without this the AppService process appears alive
        # but its port never binds — see Scheduler.run_service tail for
        # the canonical pattern.
        super().run_service()

    def cleanup(self):
        if hasattr(self, "_shutdown_event"):
            self._shutdown_event.set()
        super().cleanup()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_loop_forever(self._shutdown_event))
        finally:
            loop.close()


async def _loop_forever(shutdown: threading.Event) -> None:
    """Walk the queue every ``WALK_INTERVAL_SECONDS`` until shutdown."""
    while not shutdown.is_set():
        try:
            await walk_once(api_key_for=_default_api_key_for)
        except Exception:
            logger.exception("BatchExecutor walk failed — continuing")
        # ``shutdown.wait`` returns True if shutdown was set during the
        # sleep, letting us bail cleanly.
        if shutdown.wait(WALK_INTERVAL_SECONDS):
            break


# ---------------------------------------------------------------------------
# Bootstrap + default API key lookup
# ---------------------------------------------------------------------------


def _bootstrap_callbacks() -> None:
    """Import every module that registers a batch result handler.

    Lazy-import to avoid pulling dream-pass dependencies into modules
    that just need the registry types. Each callback module's import
    side-effect calls ``register_handler`` at module load.
    """
    try:
        from backend.copilot.dream import batch_callbacks  # noqa: F401
    except Exception:
        logger.exception(
            "Failed to import dream batch_callbacks — "
            "dream-namespace results will fail to dispatch"
        )


def _default_api_key_for(provider: ProviderLiteral) -> str | None:
    """Look up the provider API key from settings / copilot config.

    Anthropic batch is the only path today (Step 4a). When other
    providers' batch APIs land they get added here.
    """
    if provider == "anthropic":
        # The dream pass uses the copilot config's direct Anthropic key.
        # Fall back to the shared settings key so other future callers
        # (block-layer if it ever opts in) still work.
        try:
            from backend.copilot.config import ChatConfig

            chat = ChatConfig()
            if chat.direct_anthropic_api_key:
                return chat.direct_anthropic_api_key
        except Exception:
            logger.debug("ChatConfig unavailable for batch API key lookup")
        from backend.util.settings import Settings

        return Settings().secrets.anthropic_api_key or None
    return None


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


async def _dispatch(entry: PendingEntry, rows: list[BatchResultRow]) -> None:
    """Route downloaded result rows to the namespace handler."""
    handler = _HANDLERS.get(entry.callback_namespace)
    if handler is None:
        logger.warning(
            "No handler registered for namespace=%s — batch %s results dropped",
            entry.callback_namespace,
            entry.provider_batch_id,
        )
        return
    try:
        await handler(entry, rows)
    except Exception:
        logger.exception(
            "Handler for namespace=%s failed on batch %s — results NOT retried",
            entry.callback_namespace,
            entry.provider_batch_id,
        )


async def _dispatch_error(entry: PendingEntry, *, error: str) -> None:
    """Dispatch a synthetic error row to the namespace handler.

    Used when the provider itself reports the batch failed (or when we
    timed out). Each ``custom_id`` the caller registered ends up with
    an errored ``BatchResultRow`` so the handler can mark every phase
    of the work failed and the JobStatus row gets a clean terminal state.
    """
    handler = _HANDLERS.get(entry.callback_namespace)
    if handler is None:
        return
    try:
        # Row construction stays inside the guard: ``payload`` is an
        # open contract, so malformed ``custom_ids`` (wrong types, junk
        # values) must be logged here rather than escape into the walk.
        custom_ids: list[str] = entry.payload.get("custom_ids") or []
        rows = [
            BatchResultRow(
                custom_id=cid,
                content="",
                input_tokens=0,
                output_tokens=0,
                error=error,
            )
            for cid in custom_ids
        ]
        await handler(entry, rows)
    except Exception:
        logger.exception(
            "Error-path handler for namespace=%s failed on batch %s",
            entry.callback_namespace,
            entry.provider_batch_id,
        )


# ---------------------------------------------------------------------------
# Backoff math
# ---------------------------------------------------------------------------


def _bump_delay(current_seconds: int) -> int:
    """Exponential backoff capped at ``MAX_POLL_DELAY_SECONDS``."""
    return min(current_seconds * 2, MAX_POLL_DELAY_SECONDS)


def _next_poll_at(delay_seconds: int) -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0) + _seconds(delay_seconds)


def _seconds(s: int):
    from datetime import timedelta

    return timedelta(seconds=s)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize(entry: PendingEntry) -> str:
    return json.dumps(
        {
            "provider": entry.provider,
            "provider_batch_id": entry.provider_batch_id,
            "callback_namespace": entry.callback_namespace,
            "submitted_at": entry.submitted_at.isoformat(),
            "next_poll_at": entry.next_poll_at.isoformat(),
            "poll_delay_seconds": entry.poll_delay_seconds,
            "payload": entry.payload,
        }
    )


def _deserialize(provider_batch_id: str | bytes, body: str | bytes) -> PendingEntry:
    if isinstance(provider_batch_id, bytes):
        provider_batch_id = provider_batch_id.decode("utf-8")
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    data = json.loads(body)
    return PendingEntry(
        provider=data["provider"],
        provider_batch_id=data["provider_batch_id"],
        callback_namespace=data["callback_namespace"],
        submitted_at=datetime.fromisoformat(data["submitted_at"]),
        next_poll_at=datetime.fromisoformat(data["next_poll_at"]),
        poll_delay_seconds=int(
            data.get("poll_delay_seconds") or INITIAL_POLL_DELAY_SECONDS
        ),
        payload=dict(data.get("payload") or {}),
    )
