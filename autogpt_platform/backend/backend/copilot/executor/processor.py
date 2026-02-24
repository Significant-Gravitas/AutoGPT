"""CoPilot execution processor - per-worker execution logic.

This module contains the processor class that handles CoPilot task execution
in a thread-local context, following the graph executor pattern.
"""

import asyncio
import logging
import threading
import time

from backend.copilot import service as copilot_service
from backend.copilot import stream_registry
from backend.copilot.config import ChatConfig
from backend.copilot.response_model import StreamError, StreamFinish, StreamFinishStep
from backend.copilot.sdk import service as sdk_service
from backend.executor.cluster_lock import ClusterLock
from backend.util.decorator import error_logged
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.logging import TruncatedLogger, configure_logging
from backend.util.process import set_service_name
from backend.util.retry import func_retry

from .utils import CoPilotExecutionEntry, CoPilotLogMetadata

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[CoPilotExecutor]")


# ============ Module Entry Points ============ #

# Thread-local storage for processor instances
_tls = threading.local()


def execute_copilot_task(
    entry: CoPilotExecutionEntry,
    cancel: threading.Event,
    cluster_lock: ClusterLock,
):
    """Execute a CoPilot task using the thread-local processor.

    This function is the entry point called by the thread pool executor.

    Args:
        entry: The task payload
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
    """Per-worker execution logic for CoPilot tasks.

    This class is instantiated once per worker thread and handles the execution
    of CoPilot chat generation tasks. It maintains an async event loop for
    running the async service code.

    The execution flow:
        1. CoPilot task is picked from RabbitMQ queue
        2. Manager submits task to thread pool
        3. Processor executes the task in its event loop
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

        logger.info(f"[CoPilotExecutor] Worker {self.tid} started")

    def cleanup(self):
        """Clean up event-loop-bound resources before the loop is destroyed.

        Shuts down the workspace storage instance that belongs to this
        worker's event loop, ensuring ``aiohttp.ClientSession.close()``
        runs on the same loop that created the session.
        """
        from backend.util.workspace_storage import shutdown_workspace_storage

        try:
            future = asyncio.run_coroutine_threadsafe(
                shutdown_workspace_storage(), self.execution_loop
            )
            future.result(timeout=5)
        except Exception as e:
            logger.warning(f"[CoPilotExecutor] Worker {self.tid} cleanup error: {e}")

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
        """Execute a CoPilot task.

        This is the main entry point for task execution. It runs the async
        execution logic in the worker's event loop and handles errors.

        Args:
            entry: The task payload containing session and message info
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

        try:
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

        except Exception as e:
            elapsed = time.monotonic() - start_time
            log.error(f"Execution failed after {elapsed:.2f}s: {e}")
            # Note: _execute_async already marks the task as failed before re-raising,
            # so we don't call _mark_task_failed here to avoid duplicate error events.
            raise

    async def _execute_async(
        self,
        entry: CoPilotExecutionEntry,
        cancel: threading.Event,
        cluster_lock: ClusterLock,
        log: CoPilotLogMetadata,
    ):
        """Async execution logic for CoPilot task.

        This method calls the existing stream_chat_completion service function
        and publishes results to the stream registry.

        Args:
            entry: The task payload
            cancel: Threading event to signal cancellation
            cluster_lock: Distributed lock for refresh
            log: Structured logger for this task
        """
        last_refresh = time.monotonic()
        refresh_interval = 30.0  # Refresh lock every 30 seconds

        try:
            # Choose service based on LaunchDarkly flag
            config = ChatConfig()
            use_sdk = await is_feature_enabled(
                Flag.COPILOT_SDK,
                entry.user_id or "anonymous",
                default=config.use_claude_agent_sdk,
            )
            stream_fn = (
                sdk_service.stream_chat_completion_sdk
                if use_sdk
                else copilot_service.stream_chat_completion
            )
            log.info(f"Using {'SDK' if use_sdk else 'standard'} service")

            # Stream chat completion and publish chunks to Redis
            async for chunk in stream_fn(
                session_id=entry.session_id,
                message=entry.message if entry.message else None,
                is_user_message=entry.is_user_message,
                user_id=entry.user_id,
                context=entry.context,
            ):
                # Check for cancellation
                if cancel.is_set():
                    log.info("Cancelled during streaming")
                    await stream_registry.publish_chunk(
                        entry.turn_id, StreamError(errorText="Operation cancelled")
                    )
                    await stream_registry.publish_chunk(
                        entry.turn_id, StreamFinishStep()
                    )
                    await stream_registry.publish_chunk(entry.turn_id, StreamFinish())
                    await stream_registry.mark_task_completed(
                        entry.session_id, status="failed"
                    )
                    return

                # Refresh cluster lock periodically
                current_time = time.monotonic()
                if current_time - last_refresh >= refresh_interval:
                    cluster_lock.refresh()
                    last_refresh = current_time

                # Publish chunk to stream registry
                try:
                    await stream_registry.publish_chunk(entry.turn_id, chunk)
                except Exception as e:
                    log.error(
                        f"Error publishing chunk {type(chunk).__name__}: {e}",
                        exc_info=True,
                    )

            # Mark task as completed
            await stream_registry.mark_task_completed(
                entry.session_id, status="completed"
            )
            log.info("Task completed successfully")

            if entry.turn_id:
                await stream_registry.cleanup_turn_stream(entry.turn_id)

        except asyncio.CancelledError:
            log.info("Task cancelled")
            await stream_registry.mark_task_completed(
                entry.session_id,
                status="failed",
                error_message="Task was cancelled",
            )
            raise

        except Exception as e:
            log.error(f"Task failed: {e}")
            await self._mark_task_failed(entry.session_id, str(e), entry.turn_id)
            raise

    async def _mark_task_failed(
        self, session_id: str, error_message: str, turn_id: str = ""
    ):
        """Mark a task as failed and publish error to stream registry."""
        try:
            await stream_registry.publish_chunk(
                turn_id, StreamError(errorText=error_message)
            )
            await stream_registry.publish_chunk(turn_id, StreamFinishStep())
            await stream_registry.publish_chunk(turn_id, StreamFinish())
            await stream_registry.mark_task_completed(session_id, status="failed")
        except Exception as e:
            logger.error(f"Failed to mark task {session_id} as failed: {e}")
