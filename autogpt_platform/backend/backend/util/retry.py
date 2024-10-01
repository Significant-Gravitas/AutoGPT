import logging
import os
from uuid import uuid4

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
pid = os.getpid()


def conn_retry(resource_name: str, action_name: str, max_retry: int = 5):
    conn_id = uuid4()
    prefix = f"[PID-{pid}|{resource_name}-{conn_id}]"

    def before_call(retry_state):
        logger.info(f"{prefix} {action_name} started...")

    def after_call(retry_state):
        if retry_state.outcome.failed:
            # Optionally, you can log something here if needed
            pass
        else:
            logger.info(f"{prefix} {action_name} completed!")

    def on_retry(retry_state):
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
