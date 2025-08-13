import contextlib
import logging
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, TypedDict, TypeVar

import ldclient
from autogpt_libs.utils.cache import async_ttl_cache
from fastapi import HTTPException
from ldclient import Context, LDClient
from ldclient.config import Config
from typing_extensions import ParamSpec

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

# Load settings at module level
settings = Settings()

P = ParamSpec("P")
T = TypeVar("T")

_is_initialized = False


class Flag(str, Enum):
    """
    Centralized enum for all LaunchDarkly feature flags.

    Add new flags here to ensure consistency across the codebase.
    """

    AUTOMOD = "AutoMod"
    AI_ACTIVITY_STATUS = "ai-agent-execution-summary"
    BETA_BLOCKS = "beta-blocks"
    AGENT_ACTIVITY = "agent-activity"


class FlagValues(TypedDict, total=False):
    """
    Type definitions for feature flag return values.

    This ensures type safety when accessing flag values.
    """

    # Boolean flags
    AUTOMOD: bool
    AI_ACTIVITY_STATUS: bool
    AGENT_ACTIVITY: bool

    # String array flags
    BETA_BLOCKS: list[str]


# Default values for flags when LaunchDarkly is unavailable
DEFAULT_FLAG_VALUES: dict[Flag, Any] = {
    Flag.AUTOMOD: False,
    Flag.AI_ACTIVITY_STATUS: False,
    Flag.BETA_BLOCKS: [],
    Flag.AGENT_ACTIVITY: True,
}


def get_client() -> LDClient:
    """Get the LaunchDarkly client singleton."""
    if not _is_initialized:
        initialize_launchdarkly()
    return ldclient.get()


def initialize_launchdarkly() -> None:
    sdk_key = settings.secrets.launch_darkly_sdk_key
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
                # It's weird, I know, but it is what it is.
                builder.set("custom", {"role": user.role})
            if user.email:
                builder.set("email", user.email)
                builder.set("email_domain", user.email.split("@")[-1])

    except Exception as e:
        logger.warning(f"Failed to fetch user context for {user_id}: {e}")

    return builder.build()


async def get_feature_flag_value(
    flag_key: str,
    user_id: str,
    default: Any = None,
) -> Any:
    """
    Get the raw value of a feature flag for a user.

    This is the generic function that returns the actual flag value,
    which could be a boolean, string, number, or JSON object.

    Args:
        flag_key: The LaunchDarkly feature flag key
        user_id: The user ID to evaluate the flag for
        default: Default value if LaunchDarkly is unavailable or flag evaluation fails

    Returns:
        The flag value from LaunchDarkly
    """
    try:
        client = get_client()

        # Check if client is initialized
        if not client.is_initialized():
            logger.debug(
                f"LaunchDarkly not initialized, using default={default} for {flag_key}"
            )
            return default

        # Get user context from Supabase
        context = await _fetch_user_context_data(user_id)

        # Evaluate flag
        result = client.variation(flag_key, context, default)

        logger.debug(
            f"Feature flag {flag_key} for user {user_id}: {result} (type: {type(result).__name__})"
        )
        return result

    except Exception as e:
        logger.warning(
            f"LaunchDarkly flag evaluation failed for {flag_key}: {e}, using default={default}"
        )
        return default


async def is_feature_enabled(
    flag_key: str,
    user_id: str,
    default: bool = False,
) -> bool:
    """
    Check if a boolean feature flag is enabled for a user.

    This function is specifically for boolean flags. It will:
    1. Get the flag value from LaunchDarkly
    2. Ensure it's a boolean (log warning if not)
    3. Return the boolean value

    Args:
        flag_key: The LaunchDarkly feature flag key (should be configured as boolean in LD)
        user_id: The user ID to evaluate the flag for
        default: Default value if LaunchDarkly is unavailable or flag evaluation fails

    Returns:
        True if feature is enabled, False otherwise
    """
    result = await get_feature_flag_value(flag_key, user_id, default)

    # If the result is already a boolean, return it
    if isinstance(result, bool):
        return result

    # Log a warning if the flag is not returning a boolean
    logger.warning(
        f"Feature flag {flag_key} returned non-boolean value: {result} (type: {type(result).__name__}). "
        f"This flag should be configured as a boolean in LaunchDarkly. Using default={default}"
    )

    # Return the default if we get a non-boolean value
    # This prevents objects from being incorrectly treated as True
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


async def get_flag_value(flag: Flag, user_id: str) -> Any:
    """
    Get the value of a feature flag for a user.

    Args:
        flag: The feature flag to evaluate
        user_id: The user ID to evaluate the flag for

    Returns:
        The flag value from LaunchDarkly, or default if unavailable
    """
    default = DEFAULT_FLAG_VALUES.get(flag)
    return await get_feature_flag_value(flag.value, user_id, default)


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
