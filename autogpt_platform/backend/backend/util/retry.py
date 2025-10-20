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
    wait_exponential_jitter,
)

from backend.util.settings import get_service_name

logger = logging.getLogger(__name__)

# Alert threshold for excessive retries
EXCESSIVE_RETRY_THRESHOLD = 50

# Rate limiting for alerts - track last alert time per function+error combination
_alert_rate_limiter = {}
_rate_limiter_lock = threading.Lock()
ALERT_RATE_LIMIT_SECONDS = 300  # 5 minutes between same alerts


def should_send_alert(func_name: str, exception: Exception, context: str = "") -> bool:
    """Check if we should send an alert based on rate limiting."""
    # Create a unique key for this function+error+context combination
    error_signature = (
        f"{context}:{func_name}:{type(exception).__name__}:{str(exception)[:100]}"
    )
    current_time = time.time()

    with _rate_limiter_lock:
        last_alert_time = _alert_rate_limiter.get(error_signature, 0)
        if current_time - last_alert_time >= ALERT_RATE_LIMIT_SECONDS:
            _alert_rate_limiter[error_signature] = current_time
            return True
        return False


def send_rate_limited_discord_alert(
    func_name: str, exception: Exception, context: str, alert_msg: str, channel=None
) -> bool:
    """
    Send a Discord alert with rate limiting.

    Returns True if alert was sent, False if rate limited.
    """
    if not should_send_alert(func_name, exception, context):
        return False

    try:
        from backend.util.clients import get_notification_manager_client
        from backend.util.metrics import DiscordChannel

        notification_client = get_notification_manager_client()
        notification_client.discord_system_alert(
            alert_msg, channel or DiscordChannel.PLATFORM
        )
        return True

    except Exception as alert_error:
        logger.error(f"Failed to send Discord alert: {alert_error}")
        return False


def _send_critical_retry_alert(
    func_name: str, attempt_number: int, exception: Exception, context: str = ""
):
    """Send alert when a function is approaching the retry failure threshold."""

    prefix = f"{context}: " if context else ""
    if send_rate_limited_discord_alert(
        func_name,
        exception,
        context,
        f"🚨 CRITICAL: Operation Approaching Failure Threshold: {prefix}'{func_name}'\n\n"
        f"Current attempt: {attempt_number}/{EXCESSIVE_RETRY_THRESHOLD}\n"
        f"Error: {type(exception).__name__}: {exception}\n\n"
        f"This operation is about to fail permanently. Investigate immediately.",
    ):
        logger.critical(
            f"CRITICAL ALERT SENT: Operation {func_name} at attempt {attempt_number}"
        )


def _create_retry_callback(context: str = ""):
    """Create a retry callback with optional context."""

    def callback(retry_state):
        attempt_number = retry_state.attempt_number
        exception = retry_state.outcome.exception()
        func_name = getattr(retry_state.fn, "__name__", "unknown")

        prefix = f"{context}: " if context else ""

        if retry_state.outcome.failed and retry_state.next_action is None:
            # Final failure - just log the error (alert was already sent at excessive threshold)
            logger.error(
                f"{prefix}Giving up after {attempt_number} attempts for '{func_name}': "
                f"{type(exception).__name__}: {exception}"
            )
        else:
            # Retry attempt - send critical alert only once at threshold (rate limited)
            if attempt_number == EXCESSIVE_RETRY_THRESHOLD:
                _send_critical_retry_alert(
                    func_name, attempt_number, exception, context
                )
            else:
                logger.warning(
                    f"{prefix}Retry attempt {attempt_number} for '{func_name}': "
                    f"{type(exception).__name__}: {exception}"
                )

    return callback


def create_retry_decorator(
    max_attempts: int = 5,
    exclude_exceptions: tuple[type[BaseException], ...] = (),
    max_wait: float = 30.0,
    context: str = "",
    reraise: bool = True,
):
    """
    Create a preconfigured retry decorator with sensible defaults.

    Uses exponential backoff with jitter by default.

    Args:
        max_attempts: Maximum number of attempts (default: 5)
        exclude_exceptions: Tuple of exception types to not retry on
        max_wait: Maximum wait time in seconds (default: 30)
        context: Optional context string for log messages
        reraise: Whether to reraise the final exception (default: True)

    Returns:
        Configured retry decorator
    """
    if exclude_exceptions:
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(max=max_wait),
            before_sleep=_create_retry_callback(context),
            reraise=reraise,
            retry=retry_if_not_exception_type(exclude_exceptions),
        )
    else:
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(max=max_wait),
            before_sleep=_create_retry_callback(context),
            reraise=reraise,
        )


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
    max_retry: int = 100,
    max_wait: float = 30,
):
    conn_id = str(uuid4())

    def on_retry(retry_state):
        prefix = _log_prefix(resource_name, conn_id)
        exception = retry_state.outcome.exception()
        attempt_number = retry_state.attempt_number
        func_name = getattr(retry_state.fn, "__name__", "unknown")

        if retry_state.outcome.failed and retry_state.next_action is None:
            logger.error(f"{prefix} {action_name} failed after retries: {exception}")
        else:
            if attempt_number == EXCESSIVE_RETRY_THRESHOLD:
                if send_rate_limited_discord_alert(
                    func_name,
                    exception,
                    f"{resource_name}_infrastructure",
                    f"🚨 **Critical Infrastructure Connection Issue**\n"
                    f"Resource: {resource_name}\n"
                    f"Action: {action_name}\n"
                    f"Function: {func_name}\n"
                    f"Current attempt: {attempt_number}/{max_retry + 1}\n"
                    f"Error: {type(exception).__name__}: {str(exception)[:200]}{'...' if len(str(exception)) > 200 else ''}\n\n"
                    f"Infrastructure component is approaching failure threshold. Investigate immediately.",
                ):
                    logger.critical(
                        f"INFRASTRUCTURE ALERT SENT: {resource_name} at {attempt_number} attempts"
                    )

            logger.warning(
                f"{prefix} {action_name} failed: {exception}. Retrying now..."
            )

    def decorator(func):
        is_coroutine = asyncio.iscoroutinefunction(func)
        # Use static retry configuration
        retry_decorator = retry(
            stop=stop_after_attempt(max_retry + 1),  # +1 for the initial attempt
            wait=wait_exponential_jitter(max=max_wait),
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


# Preconfigured retry decorator for general functions
func_retry = create_retry_decorator(max_attempts=5)


def continuous_retry(*, retry_delay: float = 1.0):
    def decorator(func):
        is_coroutine = asyncio.iscoroutinefunction(func)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            counter = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    counter += 1
                    if counter % 10 == 0:
                        log = logger.exception
                    else:
                        log = logger.warning
                    log(
                        "%s failed for the %s times, error: [%s] — retrying in %.2fs",
                        func.__name__,
                        counter,
                        str(exc) or type(exc).__name__,
                        retry_delay,
                    )
                    time.sleep(retry_delay)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            counter = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    counter += 1
                    if counter % 10 == 0:
                        log = logger.exception
                    else:
                        log = logger.warning
                    log(
                        "%s failed for the %s times, error: [%s] — retrying in %.2fs",
                        func.__name__,
                        counter,
                        str(exc) or type(exc).__name__,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)

        return async_wrapper if is_coroutine else sync_wrapper

    return decorator
