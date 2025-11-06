import contextlib
import logging
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

import ldclient
from autogpt_libs.auth.dependencies import get_optional_user_id
from fastapi import HTTPException, Security
from ldclient import Context, LDClient
from ldclient.config import Config
from typing_extensions import ParamSpec

from backend.util.cache import cached
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
    ENABLE_PLATFORM_PAYMENT = "enable-platform-payment"
    CHAT = "chat"


def is_configured() -> bool:
    """Check if LaunchDarkly is configured with an SDK key."""
    return bool(settings.secrets.launch_darkly_sdk_key)


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

    global _is_initialized
    _is_initialized = True
    if ldclient.get().is_initialized():
        logger.info("LaunchDarkly client initialized successfully")
    else:
        logger.error("LaunchDarkly client failed to initialize")


def shutdown_launchdarkly() -> None:
    """Shutdown the LaunchDarkly client."""
    if ldclient.get().is_initialized():
        ldclient.get().close()
        logger.info("LaunchDarkly client closed successfully")


@cached(maxsize=1000, ttl_seconds=86400)  # 1000 entries, 24 hours TTL
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
    flag_key: Flag,
    user_id: str,
    default: bool = False,
) -> bool:
    """
    Check if a feature flag is enabled for a user.

    Args:
        flag_key: The Flag enum value
        user_id: The user ID to evaluate the flag for
        default: Default value if LaunchDarkly is unavailable or flag evaluation fails

    Returns:
        True if feature is enabled, False otherwise
    """
    result = await get_feature_flag_value(flag_key.value, user_id, default)

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
                        "LaunchDarkly not initialized, "
                        f"using default {flag_key}={repr(default)}"
                    )
                    is_enabled = default
                else:
                    # Use the internal function directly since we have a raw string flag_key
                    flag_value = await get_feature_flag_value(
                        flag_key, str(user_id), default
                    )
                    # Ensure we treat flag value as boolean
                    if isinstance(flag_value, bool):
                        is_enabled = flag_value
                    else:
                        # Log warning and use default for non-boolean values
                        logger.warning(
                            f"Feature flag {flag_key} returned non-boolean value: "
                            f"{repr(flag_value)} (type: {type(flag_value).__name__}). "
                            f"Using default value {repr(default)}"
                        )
                        is_enabled = default

                if not is_enabled:
                    raise HTTPException(status_code=404, detail="Feature not available")

                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error evaluating feature flag {flag_key}: {e}")
                raise

        return async_wrapper

    return decorator


def create_feature_flag_dependency(
    flag_key: Flag,
    default: bool = False,
) -> Callable[[str | None], Awaitable[None]]:
    """
    Create a FastAPI dependency that checks a feature flag.

    This dependency automatically extracts the user_id from the JWT token
    (if present) for proper LaunchDarkly user targeting, while still
    supporting anonymous access.

    Args:
        flag_key: The Flag enum value to check
        default: Default value if flag evaluation fails

    Returns:
        An async dependency function that raises HTTPException if flag is disabled

    Example:
        router = APIRouter(
            dependencies=[Depends(create_feature_flag_dependency(Flag.CHAT))]
        )
    """

    async def check_feature_flag(
        user_id: str | None = Security(get_optional_user_id),
    ) -> None:
        """Check if feature flag is enabled for the user.

        The user_id is automatically injected from JWT authentication if present,
        or None for anonymous access.
        """
        # For routes that don't require authentication, use anonymous context
        check_user_id = user_id or "anonymous"

        if not is_configured():
            logger.debug(
                f"LaunchDarkly not configured, using default {flag_key.value}={default}"
            )
            if not default:
                raise HTTPException(status_code=404, detail="Feature not available")
            return

        try:
            client = get_client()
            if not client.is_initialized():
                logger.debug(
                    f"LaunchDarkly not initialized, using default {flag_key.value}={default}"
                )
                if not default:
                    raise HTTPException(status_code=404, detail="Feature not available")
                return

            is_enabled = await is_feature_enabled(flag_key, check_user_id, default)

            if not is_enabled:
                raise HTTPException(status_code=404, detail="Feature not available")
        except Exception as e:
            logger.warning(
                f"LaunchDarkly error for flag {flag_key.value}: {e}, using default={default}"
            )
            raise HTTPException(status_code=500, detail="Failed to check feature flag")

    return check_feature_flag


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
