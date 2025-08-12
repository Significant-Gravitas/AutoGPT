import contextlib
import logging
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

import ldclient
from fastapi import HTTPException
from ldclient import Context, LDClient
from ldclient.config import Config
from typing_extensions import ParamSpec

from autogpt_libs.utils.cache import async_ttl_cache

from .config import SETTINGS

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

_is_initialized = False


def get_client() -> LDClient:
    """Get the LaunchDarkly client singleton."""
    if not _is_initialized:
        initialize_launchdarkly()
    return ldclient.get()


def initialize_launchdarkly() -> None:
    sdk_key = SETTINGS.launch_darkly_sdk_key
    logger.debug(
        f"Initializing LaunchDarkly with SDK key: {'present' if sdk_key else 'missing'}"
    )

    if not sdk_key:
        logger.warning("LaunchDarkly SDK key not configured")
        return

    config = Config(sdk_key)
    ldclient.set_config(config)

    if ldclient.get().is_initialized():
        global _is_initialized
        _is_initialized = True
        logger.info("LaunchDarkly client initialized successfully")
    else:
        logger.error("LaunchDarkly client failed to initialize")


def shutdown_launchdarkly() -> None:
    """Shutdown the LaunchDarkly client."""
    if ldclient.get().is_initialized():
        ldclient.get().close()
        logger.info("LaunchDarkly client closed successfully")


@async_ttl_cache(maxsize=1000, ttl_seconds=86400)  # 1000 entries, 24 hours TTL
async def _fetch_user_context_data(user_id: str) -> Context:
    """
    Fetch user context for LaunchDarkly from Supabase.

    Args:
        user_id: The user ID to fetch data for

    Returns:
        LaunchDarkly Context object
    """
    builder = Context.builder(user_id).kind("user").anonymous(True)

    try:
        from backend.util.clients import get_supabase

        # If we have user data, update context
        response = get_supabase().auth.admin.get_user_by_id(user_id)
        if response and response.user:
            user = response.user
            builder.anonymous(False)
            if user.role:
                builder.set("role", user.role)
            if user.email:
                builder.set("email", user.email)
                builder.set("email_domain", user.email.split("@")[-1])

    except Exception as e:
        logger.warning(f"Failed to fetch user context for {user_id}: {e}")

    return builder.build()


async def is_feature_enabled(
    flag_key: str,
    user_id: str,
    default: bool = False,
) -> bool:
    """
    Check if a feature flag is enabled for a user.

    Args:
        flag_key: The LaunchDarkly feature flag key
        user_id: The user ID to evaluate the flag for
        default: Default value if LaunchDarkly is unavailable or flag evaluation fails

    Returns:
        True if feature is enabled, False otherwise
    """
    try:
        client = get_client()

        # Get user context from Supabase
        context = await _fetch_user_context_data(user_id)

        # Evaluate flag
        result = client.variation(flag_key, context, default)

        logger.debug(f"Feature flag {flag_key} for user {user_id}: {result}")
        return result

    except Exception as e:
        logger.warning(
            f"LaunchDarkly flag evaluation failed for {flag_key}: {e}, using default={default}"
        )
        return default


def feature_flag(
    flag_key: str,
    default: bool = False,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for async feature flag protected endpoints.

    Args:
        flag_key: The LaunchDarkly feature flag key
        default: Default value if flag evaluation fails

    Returns:
        Decorator that only works with async functions
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                user_id = kwargs.get("user_id")
                if not user_id:
                    raise ValueError("user_id is required")

                if not get_client().is_initialized():
                    logger.warning(
                        f"LaunchDarkly not initialized, using default={default}"
                    )
                    is_enabled = default
                else:
                    # Use the simplified function
                    is_enabled = await is_feature_enabled(
                        flag_key, str(user_id), default
                    )

                if not is_enabled:
                    raise HTTPException(status_code=404, detail="Feature not available")

                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error evaluating feature flag {flag_key}: {e}")
                raise

        return async_wrapper

    return decorator


@contextlib.contextmanager
def mock_flag_variation(flag_key: str, return_value: Any):
    """Context manager for testing feature flags."""
    original_variation = get_client().variation
    get_client().variation = lambda key, context, default: (
        return_value if key == flag_key else original_variation(key, context, default)
    )
    try:
        yield
    finally:
        get_client().variation = original_variation
