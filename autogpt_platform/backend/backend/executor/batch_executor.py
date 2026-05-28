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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

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
PENDING_KEY = "llm:batch:pending"

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


@dataclass(slots=True)
class PendingEntry:
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
    payload: dict[str, Any] = field(default_factory=dict)


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
    """Drop a finished or expired entry from the queue."""
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    await redis.hdel(PENDING_KEY, provider_batch_id)  # type: ignore[misc]


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
        error, remove from queue.

    The API key is looked up by provider via the caller-supplied
    factory so this module doesn't need to know how settings are
    organized.
    """
    now = datetime.now(timezone.utc)
    entries = await list_pending()
    for entry in entries:
        if (now - entry.submitted_at).total_seconds() > MAX_BATCH_LIFETIME_SECONDS:
            logger.warning(
                "Batch %s for namespace=%s exceeded %ds — dispatching timeout",
                entry.provider_batch_id,
                entry.callback_namespace,
                MAX_BATCH_LIFETIME_SECONDS,
            )
            await _dispatch_error(
                entry,
                error="exceeded MAX_BATCH_LIFETIME_SECONDS without completion",
            )
            await remove_pending(entry.provider_batch_id)
            continue

        if entry.next_poll_at > now:
            continue

        api_key = api_key_for(entry.provider)
        if not api_key:
            logger.warning(
                "No API key for provider=%s — leaving batch %s in queue",
                entry.provider,
                entry.provider_batch_id,
            )
            entry.next_poll_at = _next_poll_at(entry.poll_delay_seconds)
            await update_pending(entry)
            continue

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
            entry.poll_delay_seconds = _bump_delay(entry.poll_delay_seconds)
            entry.next_poll_at = _next_poll_at(entry.poll_delay_seconds)
            await update_pending(entry)
            continue

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
                entry.poll_delay_seconds = _bump_delay(entry.poll_delay_seconds)
                entry.next_poll_at = _next_poll_at(entry.poll_delay_seconds)
                await update_pending(entry)
                continue
            await _dispatch(entry, rows)
            await remove_pending(entry.provider_batch_id)
        elif status == "failed":
            await _dispatch_error(entry, error="provider reported failed")
            await remove_pending(entry.provider_batch_id)
        else:
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
        # Reuse a fixed offset from the scheduler. The actual value
        # doesn't matter much — the service has no inbound RPCs today,
        # the port is just for health checks.
        return config.execution_scheduler_port + 100

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
        # Block forever — AppService stops the process when the manager
        # signals shutdown; the thread is daemon so it dies with us.
        self._shutdown_event.wait()

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
    try:
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
