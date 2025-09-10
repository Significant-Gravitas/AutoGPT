"""
Cache configuration and TTL settings for different endpoint types.

This module centralizes cache TTL configurations for different types of data
across the application.
"""

# TTL configurations in seconds
CACHE_TTL = {
    # Static/public data - long cache
    "blocks": 3600,  # 1 hour - block definitions rarely change
    "graph_templates": 1800,  # 30 minutes
    "store_agents": 1800,  # 30 minutes - public store listings
    "store_creators": 1800,  # 30 minutes
    "store_downloads": 7200,  # 2 hours - agent downloads
    # User-specific data - short cache
    "user_graphs": 60,  # 1 minute - user's own graphs
    "user_credits": 60,  # 1 minute - credits change frequently
    "user_library": 60,  # 1 minute - user's library
    "user_api_keys": 60,  # 1 minute - API keys list
    "graph_executions": 60,  # 1 minute - execution results
    # Semi-static user data - medium cache
    "user_profile": 300,  # 5 minutes
    "user_preferences": 300,  # 5 minutes
    "user_timezone": 300,  # 5 minutes
    "user_onboarding": 300,  # 5 minutes
    "credit_history": 300,  # 5 minutes - historical data
    # Real-time data - no cache or very short
    "graph_execution_status": 10,  # 10 seconds - near real-time
    "notifications": 0,  # No cache - always fresh
    "webhooks": 0,  # No cache
}


def get_ttl(data_type: str, default: int = 60) -> int:
    """
    Get TTL for a specific data type.

    Args:
        data_type: The type of data being cached
        default: Default TTL if type not found

    Returns:
        TTL in seconds
    """
    return CACHE_TTL.get(data_type, default)


# Cache key prefixes for different data types
CACHE_KEY_PREFIXES = {
    "user_graphs": "graphs:user:",
    "user_library": "library:user:",
    "user_credits": "credits:user:",
    "user_profile": "profile:user:",
    "user_api_keys": "apikeys:user:",
    "graph": "graph:",
    "store_agent": "store:agent:",
    "store_creator": "store:creator:",
}


def get_cache_key_prefix(data_type: str, user_id: str = None, **kwargs) -> str:
    """
    Generate a cache key prefix for a specific data type.

    Args:
        data_type: The type of data being cached
        user_id: Optional user ID to include in the key
        **kwargs: Additional parameters to include in the key

    Returns:
        Cache key prefix string
    """
    prefix = CACHE_KEY_PREFIXES.get(data_type, data_type)

    if user_id and "user:" in prefix:
        prefix = f"{prefix}{user_id}:"

    # Add any additional parameters
    for key, value in kwargs.items():
        if value is not None:
            prefix = f"{prefix}{key}:{value}:"

    return prefix
