"""
Exa Websets API utilities.

This module provides common building blocks for interacting with the Exa API:
- URL construction (ExaApiUrls)
- Header building
- Item counting
- Pagination helpers
- Generic polling
"""

import asyncio
import time
from typing import Any, Callable, Iterator, Tuple

from backend.sdk import Requests


class ExaApiUrls:
    """Centralized URL builder for Exa Websets API endpoints."""

    BASE = "https://api.exa.ai/websets/v0"

    # ==================== Webset Endpoints ====================

    @classmethod
    def websets(cls) -> str:
        """List all websets endpoint."""
        return f"{cls.BASE}/websets"

    @classmethod
    def webset(cls, webset_id: str) -> str:
        """Get/update/delete webset endpoint."""
        return f"{cls.BASE}/websets/{webset_id}"

    @classmethod
    def webset_cancel(cls, webset_id: str) -> str:
        """Cancel webset endpoint."""
        return f"{cls.BASE}/websets/{webset_id}/cancel"

    @classmethod
    def webset_preview(cls) -> str:
        """Preview webset query endpoint."""
        return f"{cls.BASE}/websets/preview"

    # ==================== Item Endpoints ====================

    @classmethod
    def webset_items(cls, webset_id: str) -> str:
        """List webset items endpoint."""
        return f"{cls.BASE}/websets/{webset_id}/items"

    @classmethod
    def webset_item(cls, webset_id: str, item_id: str) -> str:
        """Get/delete specific item endpoint."""
        return f"{cls.BASE}/websets/{webset_id}/items/{item_id}"

    # ==================== Search Endpoints ====================

    @classmethod
    def webset_searches(cls, webset_id: str) -> str:
        """List/create searches endpoint."""
        return f"{cls.BASE}/websets/{webset_id}/searches"

    @classmethod
    def webset_search(cls, webset_id: str, search_id: str) -> str:
        """Get specific search endpoint."""
        return f"{cls.BASE}/websets/{webset_id}/searches/{search_id}"

    @classmethod
    def webset_search_cancel(cls, webset_id: str, search_id: str) -> str:
        """Cancel search endpoint."""
        return f"{cls.BASE}/websets/{webset_id}/searches/{search_id}/cancel"

    # ==================== Enrichment Endpoints ====================

    @classmethod
    def webset_enrichments(cls, webset_id: str) -> str:
        """List/create enrichments endpoint."""
        return f"{cls.BASE}/websets/{webset_id}/enrichments"

    @classmethod
    def webset_enrichment(cls, webset_id: str, enrichment_id: str) -> str:
        """Get/update/delete enrichment endpoint."""
        return f"{cls.BASE}/websets/{webset_id}/enrichments/{enrichment_id}"

    @classmethod
    def webset_enrichment_cancel(cls, webset_id: str, enrichment_id: str) -> str:
        """Cancel enrichment endpoint."""
        return f"{cls.BASE}/websets/{webset_id}/enrichments/{enrichment_id}/cancel"

    # ==================== Monitor Endpoints ====================

    @classmethod
    def monitors(cls) -> str:
        """List/create monitors endpoint."""
        return f"{cls.BASE}/monitors"

    @classmethod
    def monitor(cls, monitor_id: str) -> str:
        """Get/update/delete monitor endpoint."""
        return f"{cls.BASE}/monitors/{monitor_id}"

    # ==================== Import Endpoints ====================

    @classmethod
    def imports(cls) -> str:
        """List/create imports endpoint."""
        return f"{cls.BASE}/imports"

    @classmethod
    def import_(cls, import_id: str) -> str:
        """Get/delete import endpoint."""
        return f"{cls.BASE}/imports/{import_id}"


def build_headers(api_key: str, include_content_type: bool = False) -> dict:
    """
    Build standard Exa API headers.

    Args:
        api_key: The API key for authentication
        include_content_type: Whether to include Content-Type: application/json header

    Returns:
        Dictionary of headers ready for API requests

    Example:
        >>> headers = build_headers("sk-123456")
        >>> headers = build_headers("sk-123456", include_content_type=True)
    """
    headers = {"x-api-key": api_key}
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers


async def get_item_count(webset_id: str, headers: dict) -> int:
    """
    Get the total item count for a webset efficiently.

    This makes a request with limit=1 and reads from pagination data
    to avoid fetching all items.

    Args:
        webset_id: The webset ID
        headers: Request headers with API key

    Returns:
        Total number of items in the webset

    Example:
        >>> count = await get_item_count("ws-123", headers)
    """
    url = ExaApiUrls.webset_items(webset_id)
    response = await Requests().get(url, headers=headers, params={"limit": 1})
    data = response.json()

    # Prefer pagination total if available
    if "pagination" in data:
        return data["pagination"].get("total", 0)

    # Fallback to data length
    return len(data.get("data", []))


def yield_paginated_results(
    data: dict, list_key: str = "items", item_key: str = "item"
) -> Iterator[Tuple[str, Any]]:
    """
    Yield paginated API results in standard format.

    This helper yields both the full list and individual items for flexible
    graph connections, plus pagination metadata.

    Args:
        data: API response data containing 'data', 'hasMore', 'nextCursor' fields
        list_key: Output key name for the full list (default: "items")
        item_key: Output key name for individual items (default: "item")

    Yields:
        Tuples of (key, value) for block output:
        - (list_key, list): Full list of items
        - (item_key, item): Each individual item (yielded separately)
        - ("has_more", bool): Whether more results exist
        - ("next_cursor", str|None): Cursor for next page

    Example:
        >>> for key, value in yield_paginated_results(response_data, "websets", "webset"):
        >>>     yield key, value
    """
    items = data.get("data", [])

    # Yield full list for batch processing
    yield list_key, items

    # Yield individual items for single-item processing chains
    for item in items:
        yield item_key, item

    # Yield pagination metadata
    yield "has_more", data.get("hasMore", False)
    yield "next_cursor", data.get("nextCursor")


async def poll_until_complete(
    url: str,
    headers: dict,
    is_complete: Callable[[dict], bool],
    extract_result: Callable[[dict], Any],
    timeout: int = 300,
    initial_interval: float = 5.0,
    max_interval: float = 30.0,
    backoff_factor: float = 1.5,
) -> Any:
    """
    Generic polling function with exponential backoff for async operations.

    This function polls an API endpoint until a completion condition is met,
    using exponential backoff to reduce API load.

    Args:
        url: API endpoint to poll
        headers: Request headers with API key
        is_complete: Function that takes response data and returns True when complete
        extract_result: Function that extracts the result from response data
        timeout: Maximum time to wait in seconds (default: 300)
        initial_interval: Starting interval between polls in seconds (default: 5.0)
        max_interval: Maximum interval between polls in seconds (default: 30.0)
        backoff_factor: Factor to multiply interval by each iteration (default: 1.5)

    Returns:
        The result extracted by extract_result function when operation completes

    Raises:
        TimeoutError: If operation doesn't complete within timeout

    Example:
        >>> result = await poll_until_complete(
        >>>     url=ExaApiUrls.webset(webset_id),
        >>>     headers=build_headers(api_key),
        >>>     is_complete=lambda data: data.get("status") == "idle",
        >>>     extract_result=lambda data: data.get("itemsCount", 0),
        >>>     timeout=300
        >>> )
    """
    start_time = time.time()
    interval = initial_interval

    while time.time() - start_time < timeout:
        response = await Requests().get(url, headers=headers)
        data = response.json()

        if is_complete(data):
            return extract_result(data)

        await asyncio.sleep(interval)
        interval = min(interval * backoff_factor, max_interval)

    # Timeout reached - raise error
    raise TimeoutError(f"Operation did not complete within {timeout} seconds")


async def poll_webset_until_idle(
    webset_id: str, headers: dict, timeout: int = 300
) -> int:
    """
    Poll a webset until it reaches 'idle' status.

    Convenience wrapper around poll_until_complete specifically for websets.

    Args:
        webset_id: The webset ID to poll
        headers: Request headers with API key
        timeout: Maximum time to wait in seconds

    Returns:
        The item count when webset becomes idle

    Raises:
        TimeoutError: If webset doesn't become idle within timeout
    """
    return await poll_until_complete(
        url=ExaApiUrls.webset(webset_id),
        headers=headers,
        is_complete=lambda data: data.get("status", {}).get("type") == "idle",
        extract_result=lambda data: data.get("itemsCount", 0),
        timeout=timeout,
    )


async def poll_search_until_complete(
    webset_id: str, search_id: str, headers: dict, timeout: int = 300
) -> int:
    """
    Poll a search until it completes (completed/failed/cancelled).

    Convenience wrapper around poll_until_complete specifically for searches.

    Args:
        webset_id: The webset ID
        search_id: The search ID to poll
        headers: Request headers with API key
        timeout: Maximum time to wait in seconds

    Returns:
        The number of items found when search completes

    Raises:
        TimeoutError: If search doesn't complete within timeout
    """
    return await poll_until_complete(
        url=ExaApiUrls.webset_search(webset_id, search_id),
        headers=headers,
        is_complete=lambda data: data.get("status")
        in ["completed", "failed", "cancelled"],
        extract_result=lambda data: data.get("progress", {}).get("found", 0),
        timeout=timeout,
    )


async def poll_enrichment_until_complete(
    webset_id: str, enrichment_id: str, headers: dict, timeout: int = 300
) -> int:
    """
    Poll an enrichment until it completes (completed/failed/cancelled).

    Convenience wrapper around poll_until_complete specifically for enrichments.

    Args:
        webset_id: The webset ID
        enrichment_id: The enrichment ID to poll
        headers: Request headers with API key
        timeout: Maximum time to wait in seconds

    Returns:
        The number of items enriched when operation completes

    Raises:
        TimeoutError: If enrichment doesn't complete within timeout
    """
    return await poll_until_complete(
        url=ExaApiUrls.webset_enrichment(webset_id, enrichment_id),
        headers=headers,
        is_complete=lambda data: data.get("status")
        in ["completed", "failed", "cancelled"],
        extract_result=lambda data: data.get("progress", {}).get(
            "processedItems", 0
        ),
        timeout=timeout,
    )
