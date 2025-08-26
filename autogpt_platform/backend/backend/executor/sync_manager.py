import logging
import threading
from typing import Optional

from backend.executor.execution_data_client import create_execution_data_client
from backend.executor.simple_cache import get_cache

logger = logging.getLogger(__name__)


class SyncManager:
    def __init__(self, interval: int = 5):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._sync_loop, name="sync-manager")
            self._thread.daemon = True
            self._thread.start()
            logger.info("Sync manager started")

    def stop(self, timeout: int = 10):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("Sync manager did not stop gracefully")
            else:
                logger.info("Sync manager stopped")

    def _sync_loop(self):
        while not self._stop_event.is_set():
            try:
                self._sync_pending_updates()
            except Exception as e:
                logger.error(f"Sync error: {e}")

            # Wait for interval or stop event
            self._stop_event.wait(self.interval)

    def _sync_pending_updates(self):
        cache = get_cache()
        outputs, statuses = cache.get_pending_updates()

        if not outputs and not statuses:
            return

        db_client = create_execution_data_client()

        # Sync output updates
        for output in outputs:
            try:
                db_client.upsert_execution_output(
                    node_exec_id=output["node_exec_id"], output=output["output"]
                )
            except Exception as e:
                logger.error(f"Failed to sync output for {output['node_exec_id']}: {e}")

        # Sync status updates
        for status in statuses:
            try:
                db_client.update_node_execution_status(
                    node_exec_id=status["node_exec_id"], status=status["status"]
                )
            except Exception as e:
                logger.error(f"Failed to sync status for {status['node_exec_id']}: {e}")

        if outputs or statuses:
            logger.debug(f"Synced {len(outputs)} outputs and {len(statuses)} statuses")


_sync_manager: Optional[SyncManager] = None


def get_sync_manager() -> SyncManager:
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = SyncManager()
    return _sync_manager
