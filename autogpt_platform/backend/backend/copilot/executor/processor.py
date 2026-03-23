"""CoPilot execution processor - per-worker execution logic.

This module contains the processor class that handles CoPilot session execution
in a thread-local context, following the graph executor pattern.
"""

import asyncio
import logging
import os
import subprocess
import threading
import time

from backend.copilot import stream_registry
from backend.copilot.baseline import stream_chat_completion_baseline
from backend.copilot.config import ChatConfig
from backend.copilot.response_model import StreamError
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.dummy import stream_chat_completion_dummy
from backend.executor.cluster_lock import ClusterLock
from backend.util.decorator import error_logged
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.logging import TruncatedLogger, configure_logging
from backend.util.process import set_service_name
from backend.util.retry import func_retry
from backend.util.workspace_storage import shutdown_workspace_storage

from .utils import CoPilotExecutionEntry, CoPilotLogMetadata

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[CoPilotExecutor]")


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

        Database is accessed only through DatabaseManager, so we don't need to connect
        to Prisma directly.
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
        self._prewarm_cli()

        logger.info(f"[CoPilotExecutor] Worker {self.tid} started")

    def _prewarm_cli(self) -> None:
        """Run the bundled CLI binary once to warm OS page caches."""
        try:
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

        Runs the async logic in the worker's event loop and handles errors.

        Args:
            entry: The turn payload containing session and message info
            cancel: Threading event to signal cancellation
            cluster_lock: Distributed lock to prevent duplicate execution
        """
        log = CoPilotLogMetadata(
            logging.getLogger(__name__),
            session_id=entry.session_id,
            user_id=entry.user_id,
        )
        log.info("Starting execution")

        start_time = time.monotonic()

        # Run the async execution in our event loop
        future = asyncio.run_coroutine_threadsafe(
            self._execute_async(entry, cancel, cluster_lock, log),
            self.execution_loop,
        )

        # Wait for completion, checking cancel periodically
        while not future.done():
            try:
                future.result(timeout=1.0)
            except asyncio.TimeoutError:
                if cancel.is_set():
                    log.info("Cancellation requested")
                    future.cancel()
                    break
                # Refresh cluster lock to maintain ownership
                cluster_lock.refresh()

        if not future.cancelled():
            # Get result to propagate any exceptions
            future.result()

        elapsed = time.monotonic() - start_time
        log.info(f"Execution completed in {elapsed:.2f}s")

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
            else:
                use_sdk = (
                    config.use_claude_code_subscription
                    or await is_feature_enabled(
                        Flag.COPILOT_SDK,
                        entry.user_id or "anonymous",
                        default=config.use_claude_agent_sdk,
                    )
                )
                stream_fn = (
                    sdk_service.stream_chat_completion_sdk
                    if use_sdk
                    else stream_chat_completion_baseline
                )
                log.info(f"Using {'SDK' if use_sdk else 'baseline'} service")

            # Stream chat completion and publish chunks to Redis.
            # stream_and_publish wraps the raw stream with registry
            # publishing (shared with collect_copilot_response).
            raw_stream = stream_fn(
                session_id=entry.session_id,
                message=entry.message if entry.message else None,
                is_user_message=entry.is_user_message,
                user_id=entry.user_id,
                context=entry.context,
                file_ids=entry.file_ids,
            )
            async for chunk in stream_registry.stream_and_publish(
                session_id=entry.session_id,
                turn_id=entry.turn_id,
                stream=raw_stream,
            ):
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
