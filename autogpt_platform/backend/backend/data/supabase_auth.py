"""
Supabase auth integration for fetching user authentication data.

This module provides functions to fetch user data from Supabase auth system,
including roles and metadata that are stored in the auth.users table.
"""

import logging
from typing import Any, Dict, Optional

from autogpt_libs.utils.cache import async_ttl_cache

logger = logging.getLogger(__name__)


@async_ttl_cache(maxsize=1000, ttl_seconds=3600)  # Cache for 1 hour
async def get_user_auth_data_from_supabase(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user authentication data from Supabase auth.users table.

    This function uses the service role key to query Supabase's auth system
    and retrieve user metadata including role information.

    Args:
        user_id: The user's UUID from Supabase auth

    Returns:
        Dictionary containing user auth data including role and email,
        or None if user not found or error occurs

    Note:
        Results are cached for 1 hour to minimize API calls to Supabase.
    """
    try:
        from backend.server.integrations.utils import get_supabase

        supabase = get_supabase()

        # Use admin API to get user data including app_metadata
        response = supabase.auth.admin.get_user_by_id(user_id)

        if not response or not response.user:
            logger.warning(f"User {user_id} not found in Supabase auth")
            return None

        user = response.user

        # Extract role from various possible locations
        # Priority: app_metadata.role > user role > default
        role = "authenticated"  # Default role

        # Check app_metadata for role (admin-set metadata)
        if hasattr(user, "app_metadata") and user.app_metadata:
            if "role" in user.app_metadata:
                role = user.app_metadata["role"]

        # Check if user has a direct role attribute
        if hasattr(user, "role") and user.role:
            # This might override app_metadata if present
            role = user.role

        # Build auth data dictionary
        auth_data = {
            "role": role,
            "email": user.email if hasattr(user, "email") else None,
        }

        # Include any other relevant app_metadata
        if hasattr(user, "app_metadata") and user.app_metadata:
            # Add other app_metadata fields that might be useful
            for key, value in user.app_metadata.items():
                if key not in ["role"]:  # Don't duplicate role
                    auth_data[f"app_{key}"] = value

        logger.debug(f"Fetched auth data for user {user_id}: role={role}")
        return auth_data

    except ImportError as e:
        logger.error(f"Failed to import Supabase integration: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching user auth data from Supabase for {user_id}: {e}")
        return None


def clear_user_auth_cache(user_id: str) -> None:
    """
    Clear the cached auth data for a specific user.

    This should be called when a user's role or auth data is updated
    to ensure the next fetch gets fresh data.

    Args:
        user_id: The user's UUID to clear from cache
    """
    # Clear the specific cache entry
    cache_key = (user_id,)
    if hasattr(get_user_auth_data_from_supabase, "cache"):
        get_user_auth_data_from_supabase.cache.pop(cache_key, None)  # type: ignore
        logger.debug(f"Cleared auth cache for user {user_id}")
