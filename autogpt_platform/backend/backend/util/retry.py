import logging
import os
from uuid import uuid4

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.util.process import get_service_name

logger = logging.getLogger(__name__)


def _log_prefix(resource_name: str, conn_id: str):
    """
    Returns a prefix string for logging purposes.
    This needs to be called on the fly to get the current process ID & service name,
    not the parent process ID & service name.
    """
    return f"[PID-{os.getpid()}|{get_service_name()}|{resource_name}-{conn_id}]"


def conn_retry(resource_name: str, action_name: str, max_retry: int = 5):
    conn_id = str(uuid4())

    def before_call(retry_state):
        prefix = _log_prefix(resource_name, conn_id)
        logger.info(f"{prefix} {action_name} started...")

    def after_call(retry_state):
        prefix = _log_prefix(resource_name, conn_id)
        if retry_state.outcome.failed:
            # Optionally, you can log something here if needed
            pass
        else:
            logger.info(f"{prefix} {action_name} completed!")

    def on_retry(retry_state):
        prefix = _log_prefix(resource_name, conn_id)
        exception = retry_state.outcome.exception()
        logger.info(f"{prefix} {action_name} failed: {exception}. Retrying now...")

    return retry(
        stop=stop_after_attempt(max_retry + 1),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before=before_call,
        after=after_call,
        before_sleep=on_retry,
        reraise=True,
    )
