import contextlib
import logging
from functools import wraps
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from backend.data.model import User

import ldclient
from backend.util.json import loads as json_loads
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


def create_context(
    user_id: str, additional_attributes: Optional[dict[str, Any]] = None
) -> Context:
    """Create LaunchDarkly context with optional additional attributes."""
    # Use the key from attributes if provided, otherwise use user_id
    context_key = user_id
    if additional_attributes and "key" in additional_attributes:
        context_key = additional_attributes["key"]

    builder = Context.builder(str(context_key)).kind("user")

    if additional_attributes:
        for key, value in additional_attributes.items():
            # Skip kind and key as they're already set
            if key in ["kind", "key"]:
                continue
            elif key == "custom" and isinstance(value, dict):
                # Handle custom attributes object - these go as individual attributes
                for custom_key, custom_value in value.items():
                    builder.set(custom_key, custom_value)
            else:
                builder.set(key, value)
    return builder.build()


@async_ttl_cache(maxsize=1000, ttl_seconds=86400)  # 1000 entries, 24 hours TTL
async def _fetch_user_context_data(user_id: str) -> dict[str, Any]:
    """
    Fetch user data and build complete LaunchDarkly context.

    First attempts to fetch role from Supabase auth system, then falls back
    to database metadata if needed.

    Args:
        user_id: The user ID to fetch data for

    Returns:
        Dictionary with user context data including role

    Raises:
        Exception: If database query fails (not cached)
    """
    # Try to get auth data from Supabase first
    supabase_auth_data = None
    try:
        from backend.data.supabase_auth import get_user_auth_data_from_supabase

        supabase_auth_data = await get_user_auth_data_from_supabase(user_id)
        logger.debug(f"Fetched Supabase auth data for {user_id}: {supabase_auth_data}")
    except ImportError:
        logger.debug("Supabase auth module not available")
    except Exception as e:
        logger.warning(f"Failed to fetch Supabase auth data for {user_id}: {e}")

    # Import here to avoid circular dependencies
    from backend.data.db import prisma

    # Check if we're in a context with direct database access
    if prisma.is_connected():
        # Direct database query using existing function
        from backend.data.user import get_user_by_id

        user = await get_user_by_id(user_id)
    else:
        # Use database manager client for RPC calls
        from backend.util.clients import get_database_manager_async_client

        db_client = get_database_manager_async_client()
        user = await db_client.get_user_by_id(user_id)

    # Build LaunchDarkly context with Supabase auth data if available
    context_data = _build_launchdarkly_context(user)

    # Override with Supabase auth data if available
    if supabase_auth_data:
        # Update role from Supabase if present
        if "role" in supabase_auth_data:
            if "custom" not in context_data:
                context_data["custom"] = {}
            context_data["custom"]["role"] = supabase_auth_data["role"]
            logger.debug(
                f"Using Supabase role for {user_id}: {supabase_auth_data['role']}"
            )

        # Update email if different
        if "email" in supabase_auth_data and supabase_auth_data["email"]:
            context_data["email"] = supabase_auth_data["email"]

    return context_data


def _build_launchdarkly_context(user: "User") -> dict[str, Any]:
    """
    Build LaunchDarkly context data matching frontend format.

    Returns a context like:
    {
        "kind": "user",
        "key": "user-id",
        "email": "user@example.com",
        "custom": {
            "role": "admin",
            "age": 365
        }
    }

    Args:
        user: User object from database

    Returns:
        Dictionary with user context data
    """
    from datetime import datetime

    from autogpt_libs.auth.models import DEFAULT_USER_ID

    # Build basic context data with kind, key, and email at root level
    context_data = {
        "kind": "user",
        "key": user.id,
        "email": user.email,
    }

    # Initialize custom attributes dictionary
    custom = {}

    # Determine user role from metadata
    role = "authenticated"  # Default role for authenticated users

    # Check if user is default/system user
    if user.id == DEFAULT_USER_ID:
        role = "admin"  # Default user has admin privileges when auth is disabled

    # Check for role in metadata to override default
    if user.metadata:
        try:
            # Handle both string (direct DB) and dict (RPC) formats
            if isinstance(user.metadata, str):
                metadata = json_loads(user.metadata)
            elif isinstance(user.metadata, dict):
                metadata = user.metadata
            else:
                metadata = {}  # Fallback for unexpected types

            # Extract role from metadata if present
            if "role" in metadata and metadata["role"]:
                role = metadata["role"]

        except (JSONDecodeError, TypeError) as e:
            logger.debug(f"Failed to parse user metadata for context: {e}")

    # Set the role in custom attributes
    custom["role"] = role

    # Add account age in days if available
    if user.created_at:
        account_age_days = (datetime.now(user.created_at.tzinfo) - user.created_at).days
        custom["age"] = account_age_days

    # Add the custom object to context
    context_data["custom"] = custom  # type: ignore

    return context_data


async def is_feature_enabled(
    flag_key: str,
    user_id: str,
    default: bool = False,
    use_user_id_only: bool = False,
    additional_attributes: Optional[dict[str, Any]] = None,
    user_role: Optional[str] = None,
) -> bool:
    """
    Check if a feature flag is enabled for a user with full LaunchDarkly context support.

    Args:
        flag_key: The LaunchDarkly feature flag key
        user_id: The user ID to evaluate the flag for
        default: Default value if LaunchDarkly is unavailable or flag evaluation fails
        use_user_id_only: If True, only use user_id without fetching database context
        additional_attributes: Additional attributes to include in the context
        user_role: Optional user role (e.g., "admin", "user") to add to segments

    Returns:
        True if feature is enabled, False otherwise
    """
    try:
        client = get_client()

        if use_user_id_only:
            # Simple context with just user ID (for backward compatibility)
            attrs = additional_attributes or {}
            if user_role:
                # Add role to custom attributes for consistency
                if "custom" not in attrs:
                    attrs["custom"] = {}
                if isinstance(attrs["custom"], dict):
                    attrs["custom"]["role"] = user_role
            context = create_context(str(user_id), attrs)
        else:
            # Full context with user segments and metadata from database
            try:
                user_data = await _fetch_user_context_data(user_id)
            except ImportError as e:
                # Database modules not available - fallback to simple context
                logger.debug(f"Database modules not available: {e}")
                user_data = {}
            except Exception as e:
                # Database error - log and fallback to simple context
                logger.warning(f"Failed to fetch user context for {user_id}: {e}")
                user_data = {}

            # Merge additional attributes and role
            attrs = additional_attributes or {}

            # If user_role is provided, add it to custom attributes
            if user_role:
                if "custom" not in user_data:
                    user_data["custom"] = {}
                user_data["custom"]["role"] = user_role

            # Merge additional attributes with user data
            # Handle custom attributes specially
            if "custom" in attrs and isinstance(attrs["custom"], dict):
                if "custom" not in user_data:
                    user_data["custom"] = {}
                user_data["custom"].update(attrs["custom"])
                # Remove custom from attrs to avoid duplication
                attrs = {k: v for k, v in attrs.items() if k != "custom"}

            # Merge remaining attributes
            final_attrs = {**user_data, **attrs}

            context = create_context(str(user_id), final_attrs)

        # Evaluate the flag
        result = client.variation(flag_key, context, default)

        logger.debug(
            f"Feature flag {flag_key} for user {user_id}: {result} "
            f"(use_user_id_only: {use_user_id_only})"
        )

        return result

    except Exception as e:
        logger.debug(
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
                    # Use the unified function with full context support
                    is_enabled = await is_feature_enabled(
                        flag_key, str(user_id), default, use_user_id_only=False
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
