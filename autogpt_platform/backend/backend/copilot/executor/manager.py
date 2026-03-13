"""CoPilot Executor Manager - main service for CoPilot task execution.

This module contains the CoPilotExecutor class that consumes chat tasks from
RabbitMQ and processes them using a thread pool, following the graph executor pattern.
"""

import asyncio
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPChannelError, AMQPConnectionError
from pika.spec import Basic, BasicProperties
from prometheus_client import Gauge, start_http_server

from backend.data import redis_client as redis
from backend.data.rabbitmq import SyncRabbitMQ
from backend.executor.cluster_lock import ClusterLock
from backend.util.decorator import error_logged
from backend.util.logging import TruncatedLogger
from backend.util.process import AppProcess
from backend.util.retry import continuous_retry
from backend.util.settings import Settings

from .processor import execute_copilot_turn, init_worker
from .utils import (
    COPILOT_CANCEL_QUEUE_NAME,
    COPILOT_EXECUTION_QUEUE_NAME,
    GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,
    CancelCoPilotEvent,
    CoPilotExecutionEntry,
    create_copilot_queue_config,
)

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[CoPilotExecutor]")
settings = Settings()

# Prometheus metrics
active_tasks_gauge = Gauge(
    "copilot_executor_active_tasks",
    "Number of active CoPilot tasks",
)
pool_size_gauge = Gauge(
    "copilot_executor_pool_size",
    "Maximum number of CoPilot executor workers",
)
utilization_gauge = Gauge(
    "copilot_executor_utilization_ratio",
    "Ratio of active tasks to pool size",
)


class CoPilotExecutor(AppProcess):
    """CoPilot Executor service for processing chat generation tasks.

    This service consumes tasks from RabbitMQ, processes them using a thread pool,
    and publishes results to Redis Streams. It follows the graph executor pattern
    for reliable message handling and graceful shutdown.

    Key features:
    - RabbitMQ-based task distribution with manual acknowledgment
    - Thread pool executor for concurrent task processing
    - Cluster lock for duplicate prevention across pods
    - Graceful shutdown with timeout for in-flight tasks
    - FANOUT exchange for cancellation broadcast
    """

    def __init__(self):
        super().__init__()
        self.pool_size = settings.config.num_copilot_workers
        self.active_tasks: dict[str, tuple[Future, threading.Event]] = {}
        self.executor_id = str(uuid.uuid4())

        self._executor = None
        self._stop_consuming = None

        self._cancel_thread = None
        self._cancel_client = None
        self._run_thread = None
        self._run_client = None

        self._task_locks: dict[str, ClusterLock] = {}
        self._active_tasks_lock = threading.Lock()

    # ============ Main Entry Points (AppProcess interface) ============ #

    def run(self):
        """Main service loop - consume from RabbitMQ."""
        logger.info(f"Pod assigned executor_id: {self.executor_id}")
        logger.info(f"Spawn max-{self.pool_size} workers...")

        pool_size_gauge.set(self.pool_size)
        self._update_metrics()
        start_http_server(settings.config.copilot_executor_port)

        self.cancel_thread.start()
        self.run_thread.start()

        while True:
            time.sleep(1e5)

    def cleanup(self):
        """Graceful shutdown with active execution waiting."""
        pid = os.getpid()
        logger.info(f"[cleanup {pid}] Starting graceful shutdown...")

        # Signal the consumer thread to stop
        try:
            self.stop_consuming.set()
            run_channel = self.run_client.get_channel()
            run_channel.connection.add_callback_threadsafe(
                lambda: run_channel.stop_consuming()
            )
            logger.info(f"[cleanup {pid}] Consumer has been signaled to stop")
        except Exception as e:
            logger.error(f"[cleanup {pid}] Error stopping consumer: {e}")

        # Wait for active executions to complete
        if self.active_tasks:
            logger.info(
                f"[cleanup {pid}] Waiting for {len(self.active_tasks)} active tasks to complete (timeout: {GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS}s)..."
            )

            start_time = time.monotonic()
            last_refresh = start_time
            lock_refresh_interval = settings.config.cluster_lock_timeout / 10

            while (
                self.active_tasks
                and (time.monotonic() - start_time) < GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS
            ):
                self._cleanup_completed_tasks()
                if not self.active_tasks:
                    break

                # Refresh cluster locks periodically
                current_time = time.monotonic()
                if current_time - last_refresh >= lock_refresh_interval:
                    for lock in list(self._task_locks.values()):
                        try:
                            lock.refresh()
                        except Exception as e:
                            logger.warning(
                                f"[cleanup {pid}] Failed to refresh lock: {e}"
                            )
                    last_refresh = current_time

                logger.info(
                    f"[cleanup {pid}] {len(self.active_tasks)} tasks still active, waiting..."
                )
                time.sleep(10.0)

        # Stop message consumers
        if self._run_thread:
            self._stop_message_consumers(
                self._run_thread, self.run_client, "[cleanup][run]"
            )
        if self._cancel_thread:
            self._stop_message_consumers(
                self._cancel_thread, self.cancel_client, "[cleanup][cancel]"
            )

        # Clean up worker threads (closes per-loop workspace storage sessions)
        if self._executor:
            from .processor import cleanup_worker

            logger.info(f"[cleanup {pid}] Cleaning up workers...")
            futures = []
            for _ in range(self._executor._max_workers):
                futures.append(self._executor.submit(cleanup_worker))
            for f in futures:
                try:
                    f.result(timeout=10)
                except Exception as e:
                    logger.warning(f"[cleanup {pid}] Worker cleanup error: {e}")

            logger.info(f"[cleanup {pid}] Shutting down executor...")
            self._executor.shutdown(wait=False)

        # Release any remaining locks
        for session_id, lock in list(self._task_locks.items()):
            try:
                lock.release()
                logger.info(f"[cleanup {pid}] Released lock for {session_id}")
            except Exception as e:
                logger.error(
                    f"[cleanup {pid}] Failed to release lock for {session_id}: {e}"
                )

        logger.info(f"[cleanup {pid}] Graceful shutdown completed")

    # ============ RabbitMQ Consumer Methods ============ #

    @continuous_retry()
    def _consume_cancel(self):
        """Consume cancellation messages from FANOUT exchange."""
        if self.stop_consuming.is_set() and not self.active_tasks:
            logger.info("Stop reconnecting cancel consumer - service cleaned up")
            return

        if not self.cancel_client.is_ready:
            self.cancel_client.disconnect()
        self.cancel_client.connect()

        # Check again after connect - shutdown may have been requested
        if self.stop_consuming.is_set() and not self.active_tasks:
            logger.info("Stop consuming requested during reconnect - disconnecting")
            self.cancel_client.disconnect()
            return

        cancel_channel = self.cancel_client.get_channel()
        cancel_channel.basic_consume(
            queue=COPILOT_CANCEL_QUEUE_NAME,
            on_message_callback=self._handle_cancel_message,
            auto_ack=True,
        )
        logger.info("Starting to consume cancel messages...")
        cancel_channel.start_consuming()
        if not self.stop_consuming.is_set() or self.active_tasks:
            raise RuntimeError("Cancel message consumer stopped unexpectedly")
        logger.info("Cancel message consumer stopped gracefully")

    @continuous_retry()
    def _consume_run(self):
        """Consume run messages from DIRECT exchange."""
        if self.stop_consuming.is_set():
            logger.info("Stop reconnecting run consumer - service cleaned up")
            return

        if not self.run_client.is_ready:
            self.run_client.disconnect()
        self.run_client.connect()

        # Check again after connect - shutdown may have been requested
        if self.stop_consuming.is_set():
            logger.info("Stop consuming requested during reconnect - disconnecting")
            self.run_client.disconnect()
            return

        run_channel = self.run_client.get_channel()
        run_channel.basic_qos(prefetch_count=self.pool_size)

        run_channel.basic_consume(
            queue=COPILOT_EXECUTION_QUEUE_NAME,
            on_message_callback=self._handle_run_message,
            auto_ack=False,
            consumer_tag="copilot_execution_consumer",
        )
        logger.info("Starting to consume run messages...")
        run_channel.start_consuming()
        if not self.stop_consuming.is_set():
            raise RuntimeError("Run message consumer stopped unexpectedly")
        logger.info("Run message consumer stopped gracefully")

    # ============ Message Handlers ============ #

    @error_logged(swallow=True)
    def _handle_cancel_message(
        self,
        _channel: BlockingChannel,
        _method: Basic.Deliver,
        _properties: BasicProperties,
        body: bytes,
    ):
        """Handle cancel message from FANOUT exchange."""
        request = CancelCoPilotEvent.model_validate_json(body)
        session_id = request.session_id
        if not session_id:
            logger.warning("Cancel message missing 'session_id'")
            return
        if session_id not in self.active_tasks:
            logger.debug(f"Cancel received for {session_id} but not active")
            return

        _, cancel_event = self.active_tasks[session_id]
        logger.info(f"Received cancel for {session_id}")
        if not cancel_event.is_set():
            cancel_event.set()
        else:
            logger.debug(f"Cancel already set for {session_id}")

    def _handle_run_message(
        self,
        _channel: BlockingChannel,
        method: Basic.Deliver,
        _properties: BasicProperties,
        body: bytes,
    ):
        """Handle run message from DIRECT exchange."""
        delivery_tag = method.delivery_tag
        # Capture the channel used at message delivery time to ensure we ack
        # on the correct channel. Delivery tags are channel-scoped and become
        # invalid if the channel is recreated after reconnection.
        delivery_channel = _channel

        def ack_message(reject: bool, requeue: bool):
            """Acknowledge or reject the message.

            Uses the channel from the original message delivery. If the channel
            is no longer open (e.g., after reconnection), logs a warning and
            skips the ack - RabbitMQ will redeliver the message automatically.
            """
            try:
                if not delivery_channel.is_open:
                    logger.warning(
                        f"Channel closed, cannot ack delivery_tag={delivery_tag}. "
                        "Message will be redelivered by RabbitMQ."
                    )
                    return

                if reject:
                    delivery_channel.connection.add_callback_threadsafe(
                        lambda: delivery_channel.basic_nack(
                            delivery_tag, requeue=requeue
                        )
                    )
                else:
                    delivery_channel.connection.add_callback_threadsafe(
                        lambda: delivery_channel.basic_ack(delivery_tag)
                    )
            except (AMQPChannelError, AMQPConnectionError) as e:
                # Channel/connection errors indicate stale delivery tag - don't retry
                logger.warning(
                    f"Cannot ack delivery_tag={delivery_tag} due to channel/connection "
                    f"error: {e}. Message will be redelivered by RabbitMQ."
                )
            except Exception as e:
                # Other errors might be transient, but log and skip to avoid blocking
                logger.error(
                    f"Unexpected error acking delivery_tag={delivery_tag}: {e}"
                )

        # Check if we're shutting down
        if self.stop_consuming.is_set():
            logger.info("Rejecting new task during shutdown")
            ack_message(reject=True, requeue=True)
            return

        # Check if we can accept more tasks
        self._cleanup_completed_tasks()
        if len(self.active_tasks) >= self.pool_size:
            ack_message(reject=True, requeue=True)
            return

        try:
            entry = CoPilotExecutionEntry.model_validate_json(body)
        except Exception as e:
            logger.error(f"Could not parse run message: {e}, body={body}")
            ack_message(reject=True, requeue=False)
            return

        session_id = entry.session_id

        # Check for local duplicate - session is already running on this executor
        if session_id in self.active_tasks:
            logger.warning(
                f"Session {session_id} already running locally, rejecting duplicate"
            )
            ack_message(reject=True, requeue=False)
            return

        # Try to acquire cluster-wide lock
        cluster_lock = ClusterLock(
            redis=redis.get_redis(),
            key=f"copilot:session:{session_id}:lock",
            owner_id=self.executor_id,
            timeout=settings.config.cluster_lock_timeout,
        )
        current_owner = cluster_lock.try_acquire()
        if current_owner != self.executor_id:
            if current_owner is not None:
                logger.warning(
                    f"Session {session_id} already running on pod {current_owner}"
                )
                ack_message(reject=True, requeue=False)
            else:
                logger.warning(
                    f"Could not acquire lock for {session_id} - Redis unavailable"
                )
                ack_message(reject=True, requeue=True)
            return

        # Execute the task
        try:
            self._task_locks[session_id] = cluster_lock

            logger.info(
                f"Acquired cluster lock for {session_id}, "
                f"executor_id={self.executor_id}"
            )

            cancel_event = threading.Event()
            future = self.executor.submit(
                execute_copilot_turn, entry, cancel_event, cluster_lock
            )
            self.active_tasks[session_id] = (future, cancel_event)
        except Exception as e:
            logger.warning(f"Failed to setup execution for {session_id}: {e}")
            cluster_lock.release()
            if session_id in self._task_locks:
                del self._task_locks[session_id]
            ack_message(reject=True, requeue=True)
            return

        self._update_metrics()

        def on_run_done(f: Future):
            logger.info(f"Run completed for {session_id}")
            error_msg = None
            try:
                if exec_error := f.exception():
                    error_msg = str(exec_error) or type(exec_error).__name__
                    logger.error(f"Execution for {session_id} failed: {error_msg}")
                    ack_message(reject=True, requeue=False)
                else:
                    ack_message(reject=False, requeue=False)
            except asyncio.CancelledError:
                logger.info(f"Run completion callback cancelled for {session_id}")
            except BaseException as e:
                error_msg = str(e) or type(e).__name__
                logger.exception(f"Error in run completion callback: {error_msg}")
            finally:
                # Release the cluster lock
                if session_id in self._task_locks:
                    logger.info(f"Releasing cluster lock for {session_id}")
                    self._task_locks[session_id].release()
                    del self._task_locks[session_id]
                self._cleanup_completed_tasks()

        future.add_done_callback(on_run_done)

    # ============ Helper Methods ============ #

    def _cleanup_completed_tasks(self) -> list[str]:
        """Remove completed futures from active_tasks and update metrics."""
        completed_tasks = []
        with self._active_tasks_lock:
            for session_id, (future, _) in list(self.active_tasks.items()):
                if future.done():
                    completed_tasks.append(session_id)
                    self.active_tasks.pop(session_id, None)
                    logger.info(f"Cleaned up completed session {session_id}")

        self._update_metrics()
        return completed_tasks

    def _update_metrics(self):
        """Update Prometheus metrics."""
        active_count = len(self.active_tasks)
        active_tasks_gauge.set(active_count)
        if self.stop_consuming.is_set():
            utilization_gauge.set(1.0)
        else:
            utilization_gauge.set(
                active_count / self.pool_size if self.pool_size > 0 else 0
            )

    def _stop_message_consumers(
        self, thread: threading.Thread, client: SyncRabbitMQ, prefix: str
    ):
        """Stop a message consumer thread."""
        try:
            channel = client.get_channel()
            channel.connection.add_callback_threadsafe(lambda: channel.stop_consuming())

            thread.join(timeout=300)
            if thread.is_alive():
                logger.error(
                    f"{prefix} Thread did not finish in time, forcing disconnect"
                )

            client.disconnect()
            logger.info(f"{prefix} Client disconnected")
        except Exception as e:
            logger.error(f"{prefix} Error disconnecting client: {e}")

    # ============ Lazy-initialized Properties ============ #

    @property
    def cancel_thread(self) -> threading.Thread:
        if self._cancel_thread is None:
            self._cancel_thread = threading.Thread(
                target=lambda: self._consume_cancel(),
                daemon=True,
            )
        return self._cancel_thread

    @property
    def run_thread(self) -> threading.Thread:
        if self._run_thread is None:
            self._run_thread = threading.Thread(
                target=lambda: self._consume_run(),
                daemon=True,
            )
        return self._run_thread

    @property
    def stop_consuming(self) -> threading.Event:
        if self._stop_consuming is None:
            self._stop_consuming = threading.Event()
        return self._stop_consuming

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.pool_size,
                initializer=init_worker,
            )
        return self._executor

    @property
    def cancel_client(self) -> SyncRabbitMQ:
        if self._cancel_client is None:
            self._cancel_client = SyncRabbitMQ(create_copilot_queue_config())
        return self._cancel_client

    @property
    def run_client(self) -> SyncRabbitMQ:
        if self._run_client is None:
            self._run_client = SyncRabbitMQ(create_copilot_queue_config())
        return self._run_client
