import asyncio
import logging
import os
import threading
from functools import wraps
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
    return f"[PID-{os.getpid()}|THREAD-{threading.get_native_id()}|{get_service_name()}|{resource_name}-{conn_id}]"


def conn_retry(
    resource_name: str,
    action_name: str,
    max_retry: int = 5,
    multiplier: int = 1,
    min_wait: float = 1,
    max_wait: float = 30,
):
    conn_id = str(uuid4())

    def on_retry(retry_state):
        prefix = _log_prefix(resource_name, conn_id)
        exception = retry_state.outcome.exception()
        logger.error(f"{prefix} {action_name} failed: {exception}. Retrying now...")

    def decorator(func):
        is_coroutine = asyncio.iscoroutinefunction(func)
        retry_decorator = retry(
            stop=stop_after_attempt(max_retry + 1),
            wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
            before_sleep=on_retry,
            reraise=True,
        )
        wrapped_func = retry_decorator(func)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            prefix = _log_prefix(resource_name, conn_id)
            logger.info(f"{prefix} {action_name} started...")
            try:
                result = wrapped_func(*args, **kwargs)
                logger.info(f"{prefix} {action_name} completed successfully.")
                return result
            except Exception as e:
                logger.error(f"{prefix} {action_name} failed after retries: {e}")
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            prefix = _log_prefix(resource_name, conn_id)
            logger.info(f"{prefix} {action_name} started...")
            try:
                result = await wrapped_func(*args, **kwargs)
                logger.info(f"{prefix} {action_name} completed successfully.")
                return result
            except Exception as e:
                logger.error(f"{prefix} {action_name} failed after retries: {e}")
                raise

        return async_wrapper if is_coroutine else sync_wrapper

    return decorator
