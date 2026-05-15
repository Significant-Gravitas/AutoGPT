"""CoPilot execution processor - per-worker execution logic.

This module contains the processor class that handles CoPilot session execution
in a thread-local context, following the graph executor pattern.
"""

import asyncio
import concurrent.futures
import logging
import os
import subprocess
import threading
import time

from backend.copilot import stream_registry
from backend.copilot.baseline import stream_chat_completion_baseline
from backend.copilot.config import ChatConfig, CopilotMode
from backend.copilot.response_model import StreamError
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.dummy import stream_chat_completion_dummy
from backend.copilot.stream_heartbeat import wrap_stream_with_heartbeat
from backend.executor.cluster_lock import ClusterLock
from backend.util.decorator import error_logged
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.logging import TruncatedLogger, configure_logging
from backend.util.process import set_service_name
from backend.util.retry import func_retry
from backend.util.workspace_storage import shutdown_workspace_storage

from .utils import CoPilotExecutionEntry, CoPilotLogMetadata

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[CoPilotExecutor]")


SHUTDOWN_ERROR_MESSAGE = (
    "Copilot executor shut down before this turn finished. Please retry."
)

# Max time execute() blocks after requesting async turn cancellation. The worker
# waits for normal cleanup so late stream writes do not race the manager, but it
# must still escape to the sync fail-close safety net if cleanup wedges.
_CANCEL_GRACE_SECONDS = 5.0

# How long to wait before logging again that a cancelled turn is still draining.
_CANCEL_DRAIN_LOG_INTERVAL_SECONDS = 1.0

# Max time the sync safety net itself spends on a single Redis CAS. Without
# this bound the whole point of ``sync_fail_close_session`` is defeated —
# ``mark_session_completed`` would hang on the same broken Redis that caused
# the original failure. On timeout we give up silently; worst case the
# session stays ``running`` until the stale-session watchdog reaps it, but
# at least the pool worker thread isn't blocked forever.
_FAIL_CLOSE_REDIS_TIMEOUT = 10.0


# Module-level symbol preserved for backward-compat with callers that import
# ``sync_fail_close_session``; the real implementation now lives on
# ``CoPilotProcessor`` so it can reuse ``self.execution_loop`` (same
# pattern as ``backend.executor.manager``'s ``node_execution_loop`` bridge
# at :meth:`ExecutionProcessor.on_graph_execution`).


def sync_fail_close_session(
    session_id: str,
    log: "CoPilotLogMetadata | TruncatedLogger",
    execution_loop: asyncio.AbstractEventLoop,
) -> None:
    """Synchronously mark *session_id* as failed from the pool worker thread.

    Submits the CAS coroutine to the long-lived *execution_loop* via
    ``run_coroutine_threadsafe`` — the same shape agent-executor uses at
    :meth:`backend.executor.manager.ExecutionProcessor.on_graph_execution`
    to reach its ``node_execution_loop`` from the pool worker. Reusing the
    persistent loop means:

    * no fresh TCP connection per turn (the ``@thread_cached``
      ``AsyncRedis`` on the execution thread stays bound to the same loop
      and is reused across every turn);
    * no loop-teardown overhead;
    * no ``clear_cache()`` gymnastics to dodge the "loop is closed" pitfall.

    ``mark_session_completed`` is an atomic CAS on ``status == "running"``,
    so when the async path already wrote a terminal state the sync call is
    a cheap no-op. The inner ``asyncio.wait_for`` bounds the Redis call so
    a wedged Redis can't hang the safety net for the full redis-py default
    TCP timeout; the outer ``.result(timeout=...)`` is a belt-and-braces
    upper bound for the cross-thread wait.
    """

    async def _bounded() -> None:
        await asyncio.wait_for(
            stream_registry.mark_session_completed(
                session_id, error_message=SHUTDOWN_ERROR_MESSAGE
            ),
            timeout=_FAIL_CLOSE_REDIS_TIMEOUT,
        )

    coro = _bounded()
    try:
        future = asyncio.run_coroutine_threadsafe(coro, execution_loop)
    except RuntimeError as e:
        coro.close()
        # execution_loop is closed — happens if cleanup() already ran the
        # per-worker teardown. Nothing we can do; let the stale-session
        # watchdog reap it.
        log.warning(f"sync fail-close skipped (execution_loop closed): {e}")
        return
    try:
        future.result(timeout=_FAIL_CLOSE_REDIS_TIMEOUT + 2)
    except concurrent.futures.TimeoutError:
        log.warning(
            f"sync fail-close timed out after {_FAIL_CLOSE_REDIS_TIMEOUT}s "
            f"(session={session_id})"
        )
        future.cancel()
    except Exception as e:
        log.warning(f"sync fail-close mark_session_completed failed: {e}")


# ============ Mode Routing ============ #


async def resolve_effective_mode(
    mode: CopilotMode | None,
    user_id: str | None,
) -> CopilotMode | None:
    """Strip ``mode`` when the user is not entitled to the toggle.

    The UI gates the mode toggle behind ``CHAT_MODE_OPTION``; the
    processor enforces the same gate server-side so an authenticated
    user cannot bypass the flag by crafting a request directly.
    """
    if mode is None:
        return None
    allowed = await is_feature_enabled(
        Flag.CHAT_MODE_OPTION,
        user_id or "anonymous",
        default=False,
    )
    if not allowed:
        logger.info(f"Ignoring mode={mode} — CHAT_MODE_OPTION is disabled for user")
        return None
    return mode


async def resolve_use_sdk_for_mode(
    mode: CopilotMode | None,
    user_id: str | None,
    *,
    use_claude_code_subscription: bool,
    config_default: bool,
) -> bool:
    """Pick the SDK vs baseline path for a single turn.

    Per-request ``mode`` wins whenever it is set (after the
    ``CHAT_MODE_OPTION`` gate has been applied upstream).  Otherwise
    falls back to the Claude Code subscription override, then the
    ``COPILOT_SDK`` LaunchDarkly flag, then the config default.
    """
    if mode == "fast":
        return False
    if mode == "extended_thinking":
        return True
    return use_claude_code_subscription or await is_feature_enabled(
        Flag.COPILOT_SDK,
        user_id or "anonymous",
        default=config_default,
    )


# ============ Module Entry Points ============ #

# Thread-local storage for processor instances
_tls = threading.local()


def execute_copilot_turn(
    entry: CoPilotExecutionEntry,
    cancel: threading.Event,
    cluster_lock: ClusterLock,
):
    """Execute a single CoPilot turn (user message → AI response).

    This function is the entry point called by the thread pool executor.

    Args:
        entry: The turn payload
        cancel: Threading event to signal cancellation
        cluster_lock: Distributed lock for this execution
    """
    processor: CoPilotProcessor = _tls.processor
    return processor.execute(entry, cancel, cluster_lock)


def init_worker():
    """Initialize the processor for the current worker thread.

    This function is called by the thread pool executor when a new worker
    thread is created. It ensures each worker has its own processor instance.
    """
    _tls.processor = CoPilotProcessor()
    _tls.processor.on_executor_start()


def cleanup_worker():
    """Clean up the processor for the current worker thread.

    Should be called before the worker thread's event loop is destroyed so
    that event-loop-bound resources (e.g. ``aiohttp.ClientSession``) are
    closed on the correct loop.
    """
    processor: CoPilotProcessor | None = getattr(_tls, "processor", None)
    if processor is not None:
        processor.cleanup()


# ============ Processor Class ============ #


class CoPilotProcessor:
    """Per-worker execution logic for CoPilot sessions.

    This class is instantiated once per worker thread and handles the execution
    of CoPilot chat generation sessions. It maintains an async event loop for
    running the async service code.

    The execution flow:
        1. Session entry is picked from RabbitMQ queue
        2. Manager submits to thread pool
        3. Processor executes in its event loop
        4. Results are published to Redis Streams
    """

    @func_retry
    def on_executor_start(self):
        """Initialize the processor when the worker thread starts.

        This method is called once per worker thread to set up the async event
        loop and initialize any required resources.

        DB operations route through DatabaseManagerAsyncClient (RPC) via the
        db_accessors pattern — no direct Prisma connection is needed here.
        """
        configure_logging()
        set_service_name("CoPilotExecutor")
        self.tid = threading.get_ident()
        self.execution_loop = asyncio.new_event_loop()
        self.execution_thread = threading.Thread(
            target=self.execution_loop.run_forever, daemon=True
        )
        self.execution_thread.start()

        # Skip the SDK's per-request CLI version check — the bundled CLI is
        # already version-matched to the SDK package.
        os.environ.setdefault("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")

        # Pre-warm the bundled CLI binary so the OS page-caches the ~185 MB
        # executable.  First spawn pays ~1.2 s; subsequent spawns ~0.65 s.
        # Read cli_path directly from env here so _prewarm_cli does not have
        # to construct a ChatConfig() (which can raise and abort the worker).
        # Priority: CHAT_CLAUDE_AGENT_CLI_PATH (prefixed) first, then
        # CLAUDE_AGENT_CLI_PATH (unprefixed) — matches config.py's validator
        # order so both paths resolve to the same binary.
        cli_path = os.getenv("CHAT_CLAUDE_AGENT_CLI_PATH") or os.getenv(
            "CLAUDE_AGENT_CLI_PATH"
        )
        self._prewarm_cli(cli_path=cli_path or None)

        logger.info(f"[CoPilotExecutor] Worker {self.tid} started")

    def _prewarm_cli(self, cli_path: str | None = None) -> None:
        """Run the Claude Code CLI binary once to warm OS page caches.

        Accepts an explicit ``cli_path`` so the caller can pass the value
        already resolved at startup rather than constructing a full
        ``ChatConfig()`` here (which reads env vars, runs validators, and
        can raise — aborting the worker prewarm silently).  Falls back to
        the ``CLAUDE_AGENT_CLI_PATH`` / ``CHAT_CLAUDE_AGENT_CLI_PATH`` env
        vars (same precedence as ``ChatConfig``), and then to the SDK's
        bundled binary when neither is set.
        """
        try:
            if not cli_path:
                from claude_agent_sdk._internal.transport.subprocess_cli import (
                    SubprocessCLITransport,
                )

                cli_path = SubprocessCLITransport._find_bundled_cli(None)  # type: ignore[arg-type]
            if cli_path:
                result = subprocess.run(
                    [cli_path, "-v"],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    logger.info(f"[CoPilotExecutor] CLI pre-warm done: {cli_path}")
                else:
                    logger.warning(
                        "[CoPilotExecutor] CLI pre-warm failed (rc=%d): %s",
                        result.returncode,  # type: ignore[reportCallIssue]
                        cli_path,
                    )
        except Exception as e:
            logger.debug(f"[CoPilotExecutor] CLI pre-warm skipped: {e}")

    def cleanup(self):
        """Clean up event-loop-bound resources before the loop is destroyed.

        Shuts down the workspace storage instance that belongs to this
        worker's event loop, ensuring ``aiohttp.ClientSession.close()``
        runs on the same loop that created the session.

        Sub-AutoPilots are enqueued on the copilot_execution queue, so
        rolling deploys survive via RabbitMQ redelivery — no bespoke
        shutdown notifier needed.
        """
        coro = shutdown_workspace_storage()
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self.execution_loop)
            future.result(timeout=5)
        except Exception as e:
            coro.close()  # Prevent "coroutine was never awaited" warning
            error_msg = str(e) or type(e).__name__
            logger.warning(
                f"[CoPilotExecutor] Worker {self.tid} cleanup error: {error_msg}"
            )

        # Stop the event loop
        self.execution_loop.call_soon_threadsafe(self.execution_loop.stop)
        self.execution_thread.join(timeout=5)
        logger.info(f"[CoPilotExecutor] Worker {self.tid} cleaned up")

    @error_logged(swallow=False)
    def execute(
        self,
        entry: CoPilotExecutionEntry,
        cancel: threading.Event,
        cluster_lock: ClusterLock,
    ):
        """Execute a CoPilot turn.

        Thin wrapper around :meth:`_execute`. The ``try/finally`` here
        guarantees :func:`sync_fail_close_session` runs on every exit
        path — normal completion or exception.
        ``mark_session_completed`` is an atomic CAS on
        ``status == "running"``, so when the async path already wrote a
        terminal state the sync call is a cheap no-op.
        """
        log = CoPilotLogMetadata(
            logging.getLogger(__name__),
            session_id=entry.session_id,
            user_id=entry.user_id,
        )
        log.info("Starting execution")
        start_time = time.monotonic()
        try:
            self._execute(entry, cancel, cluster_lock, log)
        finally:
            sync_fail_close_session(entry.session_id, log, self.execution_loop)
            elapsed = time.monotonic() - start_time
            log.info(f"Execution completed in {elapsed:.2f}s")

    def _execute(
        self,
        entry: CoPilotExecutionEntry,
        cancel: threading.Event,
        cluster_lock: ClusterLock,
        log: CoPilotLogMetadata,
    ):
        """Submit the async turn to ``self.execution_loop`` and drive it.

        Handles the sync/async boundary (cancel-event checks, cluster-lock
        refresh, bounded waits) without any Redis-state cleanup logic —
        that lives in :func:`sync_fail_close_session` which the outer
        :meth:`execute` always invokes on exit.
        """
        task_ready: concurrent.futures.Future[asyncio.Task] = (
            concurrent.futures.Future()
        )

        async def run_async_turn():
            task = asyncio.current_task()
            if task is not None and not task_ready.done():
                task_ready.set_result(task)
            return await self._execute_async(entry, cancel, cluster_lock, log)

        future = asyncio.run_coroutine_threadsafe(
            run_async_turn(),
            self.execution_loop,
        )

        cancel_requested = False
        cancel_started_at: float | None = None
        last_cancel_log_at: float | None = None

        def request_cancel() -> None:
            nonlocal cancel_requested, cancel_started_at, last_cancel_log_at
            log.info("Cancellation requested")
            try:
                task = task_ready.result(timeout=0)
            except concurrent.futures.TimeoutError:
                # Sub-millisecond race: ``run_coroutine_threadsafe`` returned
                # before ``run_async_turn`` actually started, so
                # ``task_ready.set_result`` has not run yet.  ``future.cancel``
                # on a ``concurrent.futures.Future`` whose underlying task may
                # already be picked up by the loop is best-effort — frequently
                # a no-op.  The slow path is intentional: ``cancel.is_set()``
                # is polled inside ``_execute_async`` and the bounded
                # ``_CANCEL_GRACE_SECONDS`` drain below force-cancels and falls
                # through to ``sync_fail_close_session``, so the worst-case
                # observable behaviour is "cancel takes ~5s in this rare race"
                # rather than a stuck session.
                future.cancel()
            else:
                self.execution_loop.call_soon_threadsafe(task.cancel)
            cancel_requested = True
            cancel_started_at = time.monotonic()
            last_cancel_log_at = cancel_started_at

        def log_cancel_wait() -> None:
            nonlocal last_cancel_log_at
            if cancel_started_at is None or last_cancel_log_at is None:
                return
            now = time.monotonic()
            if now - last_cancel_log_at < _CANCEL_DRAIN_LOG_INTERVAL_SECONDS:
                return
            elapsed = now - cancel_started_at
            log.warning(f"Waiting for cancelled turn to drain ({elapsed:.1f}s elapsed)")
            last_cancel_log_at = now

        def cancel_drain_timed_out() -> bool:
            if cancel_started_at is None:
                return False
            elapsed = time.monotonic() - cancel_started_at
            if elapsed < _CANCEL_GRACE_SECONDS:
                return False
            log.warning(
                f"Cancelled turn did not drain within {_CANCEL_GRACE_SECONDS:.1f}s; "
                "falling through to sync fail-close"
            )
            future.cancel()
            return True

        # Wait for completion, checking cancel periodically. A cancellation
        # request waits for normal async cleanup, but remains bounded so the
        # worker does not refresh the per-session lock forever on a wedged turn.
        while True:
            try:
                future.result(timeout=1.0)
                return
            except concurrent.futures.CancelledError:
                if cancel_requested or cancel.is_set():
                    return
                raise
            except concurrent.futures.TimeoutError:
                if cancel.is_set() and not cancel_requested:
                    request_cancel()
                elif cancel_requested and cancel_started_at is not None:
                    if cancel_drain_timed_out():
                        return
                    log_cancel_wait()
                cluster_lock.refresh()

    async def _execute_async(
        self,
        entry: CoPilotExecutionEntry,
        cancel: threading.Event,
        cluster_lock: ClusterLock,
        log: CoPilotLogMetadata,
    ):
        """Async execution logic for a CoPilot turn.

        Calls the chat completion service (SDK or baseline) and publishes
        results to the stream registry.

        Args:
            entry: The turn payload
            cancel: Threading event to signal cancellation
            cluster_lock: Distributed lock for refresh
            log: Structured logger
        """
        last_refresh = time.monotonic()
        refresh_interval = 30.0  # Refresh lock every 30 seconds
        error_msg = None

        try:
            # Choose service based on LaunchDarkly flag.
            # Claude Code subscription forces SDK mode (CLI subprocess auth).
            config = ChatConfig()

            if config.test_mode:
                stream_fn = stream_chat_completion_dummy
                log.warning("Using DUMMY service (CHAT_TEST_MODE=true)")
                effective_mode = None
            else:
                # Enforce server-side feature-flag gate so unauthorised
                # users cannot force a mode by crafting the request.
                effective_mode = await resolve_effective_mode(entry.mode, entry.user_id)
                use_sdk = await resolve_use_sdk_for_mode(
                    effective_mode,
                    entry.user_id,
                    use_claude_code_subscription=config.use_claude_code_subscription,
                    config_default=config.use_claude_agent_sdk,
                )
                stream_fn = (
                    sdk_service.stream_chat_completion_sdk
                    if use_sdk
                    else stream_chat_completion_baseline
                )
                log.info(
                    f"Using {'SDK' if use_sdk else 'baseline'} service "
                    f"(mode={effective_mode or 'default'})"
                )

            # Stream chat completion and publish chunks to Redis.
            # stream_and_publish wraps the raw stream with registry
            # publishing so subscribers on the session Redis stream
            # (e.g. wait_for_session_result, SSE clients) receive the
            # same events as they are produced.
            raw_stream = stream_fn(
                session_id=entry.session_id,
                message=entry.message if entry.message else None,
                is_user_message=entry.is_user_message,
                user_id=entry.user_id,
                context=entry.context,
                file_ids=entry.file_ids,
                mode=effective_mode,
                model=entry.model,
                permissions=entry.permissions,
                request_arrival_at=entry.request_arrival_at,
            )
            # Surround the driver stream with a silence watchdog so the
            # FE shows escalating ``StreamStatus`` messages during long
            # silent gaps (deep thinking, slow tool execution, inter-tool
            # gaps) instead of looping the generic "Thinking…" phrases.
            heartbeat_stream = wrap_stream_with_heartbeat(raw_stream)
            published_stream = stream_registry.stream_and_publish(
                session_id=entry.session_id,
                turn_id=entry.turn_id,
                stream=heartbeat_stream,
            )
            # Explicit aclose() on early exit: ``async for … break`` does
            # not close the generator, so GeneratorExit would never reach
            # stream_chat_completion_sdk, leaving its stream lock held
            # until GC eventually runs.
            try:
                async for chunk in published_stream:
                    if cancel.is_set():
                        log.info("Cancel requested, breaking stream")
                        break

                    # Capture StreamError so mark_session_completed receives
                    # the error message (stream_and_publish yields but does
                    # not publish StreamError — that's done by mark_session_completed).
                    if isinstance(chunk, StreamError):
                        error_msg = chunk.errorText
                        break

                    current_time = time.monotonic()
                    if current_time - last_refresh >= refresh_interval:
                        cluster_lock.refresh()
                        last_refresh = current_time
            finally:
                await published_stream.aclose()

            # Stream loop completed
            if cancel.is_set():
                log.info("Stream cancelled by user")

        except BaseException as e:
            # Handle all exceptions (including CancelledError) with appropriate logging
            if isinstance(e, asyncio.CancelledError):
                log.info("Turn cancelled")
                error_msg = "Operation cancelled"
            else:
                error_msg = str(e) or type(e).__name__
                log.error(f"Turn failed: {error_msg}")
            raise
        finally:
            # If no exception but user cancelled, still mark as cancelled
            if not error_msg and cancel.is_set():
                error_msg = "Operation cancelled"
            try:
                await stream_registry.mark_session_completed(
                    entry.session_id, error_message=error_msg
                )
            except Exception as mark_err:
                log.error(f"Failed to mark session completed: {mark_err}")
