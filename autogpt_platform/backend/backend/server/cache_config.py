"""
Shared cache configuration constants.

This module defines all page_size defaults used across the application.
By centralizing these values, we ensure that cache invalidation always
uses the same page_size as the routes that populate the cache.

CRITICAL: If you change any of these values, the tests in
test_cache_invalidation_consistency.py will fail to remind you to
update all dependent code.
"""

# V1 API (legacy) page sizes
V1_GRAPHS_PAGE_SIZE = 250
"""Default page size for listing user graphs in v1 API."""

V1_LIBRARY_AGENTS_PAGE_SIZE = 10
"""Default page size for library agents in v1 API."""

V1_GRAPH_EXECUTIONS_PAGE_SIZE = 25
"""Default page size for graph executions in v1 API."""

# V2 Store API page sizes
V2_STORE_AGENTS_PAGE_SIZE = 20
"""Default page size for store agents listing."""

V2_STORE_CREATORS_PAGE_SIZE = 20
"""Default page size for store creators listing."""

V2_STORE_SUBMISSIONS_PAGE_SIZE = 20
"""Default page size for user submissions listing."""

V2_MY_AGENTS_PAGE_SIZE = 20
"""Default page size for user's own agents listing."""

# V2 Library API page sizes
V2_LIBRARY_AGENTS_PAGE_SIZE = 10
"""Default page size for library agents listing in v2 API."""

V2_LIBRARY_PRESETS_PAGE_SIZE = 20
"""Default page size for library presets listing."""

# Alternative page sizes (for backward compatibility or special cases)
V2_LIBRARY_PRESETS_ALT_PAGE_SIZE = 10
"""
Alternative page size for library presets.
Some clients may use this smaller page size, so cache clearing must handle both.
"""

V2_GRAPH_EXECUTIONS_ALT_PAGE_SIZE = 10
"""
Alternative page size for graph executions.
Some clients may use this smaller page size, so cache clearing must handle both.
"""

# Cache clearing configuration
MAX_PAGES_TO_CLEAR = 20
"""
Maximum number of pages to clear when invalidating paginated caches.
This prevents infinite loops while ensuring we clear most cached pages.
For users with more than 20 pages, those pages will expire naturally via TTL.
"""


def get_page_sizes_for_clearing(
    primary_page_size: int, alt_page_size: int | None = None
) -> list[int]:
    """
    Get all page_size values that should be cleared for a given cache.

    Args:
        primary_page_size: The main page_size used by the route
        alt_page_size: Optional alternative page_size if multiple clients use different sizes

    Returns:
        List of page_size values to clear

    Example:
        >>> get_page_sizes_for_clearing(20)
        [20]
        >>> get_page_sizes_for_clearing(20, 10)
        [20, 10]
    """
    if alt_page_size is None:
        return [primary_page_size]
    return [primary_page_size, alt_page_size]
