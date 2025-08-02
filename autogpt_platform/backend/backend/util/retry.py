import asyncio
import logging
import os
import threading
import time
from functools import wraps
from uuid import uuid4

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_exponential_jitter,
)

from backend.util.process import get_service_name

logger = logging.getLogger(__name__)


def create_retry_config(
    max_attempts: int = 5,
    use_jitter: bool = False,
    multiplier: int = 1,
    min_wait: float = 1,
    max_wait: float = 30,
    exclude_exceptions: tuple[type[BaseException], ...] = (),
    before_sleep_callback=None,
    retry_error_callback=None,
):
    """
    Create a shared retry configuration.

    Args:
        max_attempts: Maximum number of attempts (default: 5)
        use_jitter: Whether to use exponential jitter (default: False)
        multiplier: Multiplier for exponential backoff (default: 1)
        min_wait: Minimum wait time in seconds (default: 1)
        max_wait: Maximum wait time in seconds (default: 30)
        exclude_exceptions: Tuple of exception types to not retry on
        before_sleep_callback: Callback function called before retry sleep
        retry_error_callback: Callback function called on final retry error

    Returns:
        Dictionary of retry configuration parameters for tenacity
    """
    config = {
        "stop": stop_after_attempt(max_attempts),
        "reraise": True,
    }

    # Configure wait strategy
    if use_jitter:
        config["wait"] = wait_exponential_jitter(max=max_wait)
    else:
        config["wait"] = wait_exponential(
            multiplier=multiplier, min=min_wait, max=max_wait
        )

    # Configure retry conditions
    if exclude_exceptions:
        config["retry"] = retry_if_not_exception_type(exclude_exceptions)

    # Configure callbacks
    if before_sleep_callback:
        config["before_sleep"] = before_sleep_callback

    if retry_error_callback:
        config["retry_error_callback"] = retry_error_callback

    return config


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
        logger.warning(f"{prefix} {action_name} failed: {exception}. Retrying now...")

    def decorator(func):
        is_coroutine = asyncio.iscoroutinefunction(func)
        # Use shared configuration
        retry_config = create_retry_config(
            max_attempts=max_retry + 1,  # +1 for the initial attempt
            use_jitter=False,
            multiplier=multiplier,
            min_wait=min_wait,
            max_wait=max_wait,
            before_sleep_callback=on_retry,
        )
        retry_decorator = retry(**retry_config)
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


def _on_func_retry_callback(retry_state):
    """Log warning on retry or error when giving up."""
    attempt_number = retry_state.attempt_number
    exception = retry_state.outcome.exception()
    func_name = getattr(retry_state.fn, "__name__", "unknown")

    if retry_state.outcome.failed and retry_state.next_action is None:
        # This is the final failure - log error
        logger.error(
            f"Giving up after {attempt_number} attempts for function '{func_name}': {exception}"
        )
    else:
        # This is a retry - log warning
        logger.warning(
            f"Retry attempt {attempt_number}/5 for function '{func_name}': {exception}"
        )


# Use shared configuration for func_retry
func_retry_config = create_retry_config(
    max_attempts=5,
    use_jitter=False,
    multiplier=1,
    min_wait=1,
    max_wait=30,
    before_sleep_callback=_on_func_retry_callback,
    retry_error_callback=_on_func_retry_callback,
)
func_retry_config["reraise"] = False  # Override reraise for func_retry
func_retry = retry(**func_retry_config)


def continuous_retry(*, retry_delay: float = 1.0):
    def decorator(func):
        is_coroutine = asyncio.iscoroutinefunction(func)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    logger.exception(
                        "%s failed with %s — retrying in %.2f s",
                        func.__name__,
                        exc,
                        retry_delay,
                    )
                    time.sleep(retry_delay)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    logger.exception(
                        "%s failed with %s — retrying in %.2f s",
                        func.__name__,
                        exc,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)

        return async_wrapper if is_coroutine else sync_wrapper

    return decorator
