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
    builder = Context.builder(str(user_id)).kind("user")
    if additional_attributes:
        for key, value in additional_attributes.items():
            builder.set(key, value)
    return builder.build()


@async_ttl_cache(maxsize=1000, ttl_seconds=86400)  # 1000 entries, 24 hours TTL
async def _fetch_user_context_data(user_id: str) -> dict[str, Any]:
    """
    Fetch user data and build complete LaunchDarkly context with segments.

    Uses the existing get_user_by_id function and applies segmentation logic
    within the LaunchDarkly client for clean separation of concerns.

    Args:
        user_id: The user ID to fetch data for

    Returns:
        Dictionary with user context data including segments

    Raises:
        Exception: If database query fails (not cached)
    """
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

    # Build and return LaunchDarkly context with segments
    return _build_launchdarkly_context(user)


def _add_role_to_attributes(
    attrs: dict[str, Any], user_role: str, existing_segments: Optional[list[str]] = None
) -> dict[str, Any]:
    """Add user role to attributes and segments, avoiding duplication."""
    attrs["role"] = user_role

    # Get existing segments from attrs or use provided existing_segments
    segments = attrs.get("segments", existing_segments or [])
    if not isinstance(segments, list):
        segments = [user_role] if segments else [user_role]
    else:
        segments.append(user_role)

    attrs["segments"] = list(set(segments))  # Remove duplicates
    return attrs


def _build_launchdarkly_context(user: "User") -> dict[str, Any]:
    """
    Build LaunchDarkly context data with segments from user object.

    Args:
        user: User object from database

    Returns:
        Dictionary with user context data including segments
    """
    from datetime import datetime

    from autogpt_libs.auth.models import DEFAULT_USER_ID

    # Build basic context data
    context_data = {
        "email": user.email,
        "name": user.name,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }

    # Determine user segments for LaunchDarkly targeting
    segments = []

    # Add role-based segment
    if user.id == DEFAULT_USER_ID:
        segments.extend(
            ["system", "admin"]
        )  # Default user has admin privileges when auth is disabled
    else:
        segments.append("user")  # Regular users

    # Add email domain-based segments for targeting
    if user.email:
        domain = user.email.split("@")[-1] if "@" in user.email else ""
        if domain:
            context_data["email_domain"] = domain
            if domain in ["agpt.co"]:
                segments.append("employee")

    # Parse metadata for additional segments and custom attributes
    if user.metadata:
        try:
            # Handle both string (direct DB) and dict (RPC) formats
            if isinstance(user.metadata, str):
                metadata = json_loads(user.metadata)
            elif isinstance(user.metadata, dict):
                metadata = user.metadata
            else:
                metadata = {}  # Fallback for unexpected types

            # Extract explicit segments from metadata if they exist
            if "segments" in metadata:
                if isinstance(metadata["segments"], list):
                    segments.extend(metadata["segments"])
                elif isinstance(metadata["segments"], str):
                    segments.append(metadata["segments"])

            # Extract role from metadata if present
            if "role" in metadata:
                role = metadata["role"]
                if role in ["admin", "moderator", "employee"]:
                    segments.append(role)
                context_data["role"] = role

            # Add custom attributes with prefix to avoid conflicts
            for key, value in metadata.items():
                if key not in ["segments", "role"]:  # Skip processed fields
                    context_data[f"custom_{key}"] = value

        except (JSONDecodeError, TypeError) as e:
            logger.debug(f"Failed to parse user metadata for context: {e}")

    # Add account age segment for targeting new vs old users
    if user.created_at:
        account_age_days = (datetime.now(user.created_at.tzinfo) - user.created_at).days
        if account_age_days < 7:
            segments.append("new_user")
        elif account_age_days < 30:
            segments.append("recent_user")
        else:
            segments.append("established_user")

        context_data["account_age_days"] = account_age_days

    # Remove duplicates and sort for consistency
    context_data["segments"] = sorted(list(set(segments)))

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
                attrs = _add_role_to_attributes(attrs, user_role)
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
            if user_role:
                attrs = _add_role_to_attributes(
                    attrs, user_role, user_data.get("segments")
                )

            # Merge attributes with user data
            final_attrs = {**user_data, **attrs}

            # Handle segment merging if both have segments
            if "segments" in user_data and "segments" in attrs:
                combined = list(set(user_data["segments"] + attrs["segments"]))
                final_attrs["segments"] = combined

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
