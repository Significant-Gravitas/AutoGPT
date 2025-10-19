"""
Exa Websets Item Management Blocks

This module provides blocks for managing items within Exa websets, including
retrieving, listing, deleting, and bulk operations on webset items.
"""

from typing import Any, Dict, List, Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import exa


class WebsetItem(Dict[str, Any]):
    """
    Represents a webset item with its properties.
    Using Dict to allow flexible schema based on entity type.
    """
    pass


class ExaGetWebsetItemBlock(Block):
    """Get a specific item from a webset by its ID."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        item_id: str = SchemaField(
            description="The ID of the specific item to retrieve",
            placeholder="item-id",
        )

    class Output(BlockSchema):
        item_id: str = SchemaField(
            description="The unique identifier for the item"
        )
        url: str = SchemaField(
            description="The URL of the original source"
        )
        title: str = SchemaField(
            description="The title of the item"
        )
        content: str = SchemaField(
            description="The main content of the item"
        )
        entity_data: dict = SchemaField(
            description="Entity-specific structured data",
            default_factory=dict,
        )
        enrichments: dict = SchemaField(
            description="Enrichment data added to the item",
            default_factory=dict,
        )
        verification_status: str = SchemaField(
            description="Verification status against criteria"
        )
        created_at: str = SchemaField(
            description="When the item was added to the webset"
        )
        updated_at: str = SchemaField(
            description="When the item was last updated"
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="c4a7d9e2-8f3b-4a6c-9d8e-a5b6c7d8e9f0",
            description="Get a specific item from a webset by its ID",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetItemBlock.Input,
            output_schema=ExaGetWebsetItemBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/items/{input_data.item_id}"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            response = await Requests().get(url, headers=headers)
            data = response.json()

            # Extract common fields
            yield "item_id", data.get("id", "")
            yield "url", data.get("url", "")
            yield "title", data.get("title", "")
            yield "content", data.get("content", "")

            # Entity-specific data will vary based on entity type
            entity_data = {}
            for key in ["company", "person", "article", "researchPaper", "custom"]:
                if key in data:
                    entity_data = data[key]
                    break

            yield "entity_data", entity_data
            yield "enrichments", data.get("enrichments", {})
            yield "verification_status", data.get("verificationStatus", "")
            yield "created_at", data.get("createdAt", "")
            yield "updated_at", data.get("updatedAt", "")

        except Exception as e:
            yield "error", str(e)
            yield "item_id", ""
            yield "url", ""
            yield "title", ""
            yield "content", ""
            yield "entity_data", {}
            yield "enrichments", {}
            yield "verification_status", ""
            yield "created_at", ""
            yield "updated_at", ""


class ExaListWebsetItemsBlock(Block):
    """List items in a webset with pagination and optional filtering."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        limit: int = SchemaField(
            default=25,
            description="Number of items to return (1-100)",
            ge=1,
            le=100,
        )
        cursor: Optional[str] = SchemaField(
            default=None,
            description="Cursor for pagination through results",
            advanced=True,
        )
        wait_for_items: bool = SchemaField(
            default=False,
            description="Wait for items to be available if webset is still processing",
            advanced=True,
        )
        wait_timeout: int = SchemaField(
            default=60,
            description="Maximum time to wait for items in seconds",
            advanced=True,
            ge=1,
            le=300,
        )

    class Output(BlockSchema):
        items: list[WebsetItem] = SchemaField(
            description="List of webset items",
            default_factory=list,
        )
        item: WebsetItem = SchemaField(
            description="Individual item (yielded for each item in the list)",
            default_factory=dict,
        )
        total_count: Optional[int] = SchemaField(
            description="Total number of items in the webset",
            default=None,
        )
        has_more: bool = SchemaField(
            description="Whether there are more items to paginate through",
            default=False,
        )
        next_cursor: Optional[str] = SchemaField(
            description="Cursor for the next page of results",
            default=None,
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="b5e8c9f0-1a2b-3c4d-5e6f-7a8b9c0d1e2f",
            description="List items in a webset with pagination support",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListWebsetItemsBlock.Input,
            output_schema=ExaListWebsetItemsBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/items"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        params: dict[str, Any] = {
            "limit": input_data.limit,
        }
        if input_data.cursor:
            params["cursor"] = input_data.cursor

        try:
            # If wait_for_items is True, poll until items are available
            if input_data.wait_for_items:
                items_data = await self._wait_for_items(
                    url, headers, params, input_data.wait_timeout
                )
            else:
                response = await Requests().get(url, headers=headers, params=params)
                items_data = response.json()

            items = items_data.get("data", [])
            pagination = items_data.get("pagination", {})

            # Yield the full list
            yield "items", items

            # Also yield individual items for easier chaining
            for item in items:
                yield "item", item

            # Pagination metadata
            yield "total_count", pagination.get("total")
            yield "has_more", items_data.get("hasMore", False)
            yield "next_cursor", items_data.get("nextCursor")

        except Exception as e:
            yield "error", str(e)
            yield "items", []
            yield "has_more", False

    async def _wait_for_items(
        self, url: str, headers: dict, params: dict, timeout: int
    ) -> dict:
        """Poll until items are available or timeout."""
        import asyncio
        import time

        start_time = time.time()
        interval = 2

        while time.time() - start_time < timeout:
            try:
                response = await Requests().get(url, headers=headers, params=params)
                data = response.json()

                # Check if we have any items
                if data.get("data"):
                    return data

                await asyncio.sleep(interval)
                interval = min(interval * 1.2, 10)  # Cap at 10 seconds

            except Exception:
                await asyncio.sleep(interval)

        # Return whatever we have on timeout
        response = await Requests().get(url, headers=headers, params=params)
        return response.json()


class ExaDeleteWebsetItemBlock(Block):
    """Delete a specific item from a webset."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        item_id: str = SchemaField(
            description="The ID of the item to delete",
            placeholder="item-id",
        )

    class Output(BlockSchema):
        item_id: str = SchemaField(
            description="The ID of the deleted item"
        )
        success: str = SchemaField(
            description="Whether the deletion was successful",
            default="true",
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="d7f0a1b2-3c4d-5e6f-8g9h-0i1j2k3l4m5n",
            description="Delete a specific item from a webset",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteWebsetItemBlock.Input,
            output_schema=ExaDeleteWebsetItemBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/items/{input_data.item_id}"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            response = await Requests().delete(url, headers=headers)

            # API typically returns 204 No Content on successful deletion
            if response.status_code in [200, 204]:
                yield "item_id", input_data.item_id
                yield "success", "true"
            else:
                data = response.json()
                yield "item_id", input_data.item_id
                yield "success", "false"
                yield "error", data.get("message", "Deletion failed")

        except Exception as e:
            yield "error", str(e)
            yield "item_id", input_data.item_id
            yield "success", "false"


class ExaBulkWebsetItemsBlock(Block):
    """Get all items from a webset in a single operation (with size limits)."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        max_items: int = SchemaField(
            default=100,
            description="Maximum number of items to retrieve (1-1000). Note: Large values may take longer.",
            ge=1,
            le=1000,
        )
        include_enrichments: bool = SchemaField(
            default=True,
            description="Include enrichment data for each item",
        )
        include_content: bool = SchemaField(
            default=True,
            description="Include full content for each item",
        )

    class Output(BlockSchema):
        items: list[WebsetItem] = SchemaField(
            description="All items from the webset",
            default_factory=list,
        )
        item: WebsetItem = SchemaField(
            description="Individual item (yielded for each item)",
            default_factory=dict,
        )
        total_retrieved: int = SchemaField(
            description="Total number of items retrieved",
            default=0,
        )
        total_in_webset: Optional[int] = SchemaField(
            description="Total number of items in the webset",
            default=None,
        )
        truncated: bool = SchemaField(
            description="Whether results were truncated due to max_items limit",
            default=False,
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="e8a9b0c1-2d3e-4f5g-6h7i-8j9k0l1m2n3o",
            description="Get all items from a webset in bulk (with configurable limits)",
            categories={BlockCategory.SEARCH},
            input_schema=ExaBulkWebsetItemsBlock.Input,
            output_schema=ExaBulkWebsetItemsBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            all_items = []
            cursor = None
            has_more = True
            batch_size = min(100, input_data.max_items)  # API limit per request

            while has_more and len(all_items) < input_data.max_items:
                # Build URL and params for this batch
                url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/items"
                params: dict[str, Any] = {
                    "limit": min(batch_size, input_data.max_items - len(all_items)),
                }
                if cursor:
                    params["cursor"] = cursor

                # Add field filters if requested
                if not input_data.include_enrichments:
                    params["exclude"] = "enrichments"
                if not input_data.include_content:
                    if "exclude" in params:
                        params["exclude"] += ",content"
                    else:
                        params["exclude"] = "content"

                # Fetch this batch
                response = await Requests().get(url, headers=headers, params=params)
                data = response.json()

                items = data.get("data", [])
                all_items.extend(items)

                # Check if there are more items
                has_more = data.get("hasMore", False)
                cursor = data.get("nextCursor")

                # Stop if we've reached the max_items limit
                if len(all_items) >= input_data.max_items:
                    break

            # Truncate if we got more than requested
            truncated = len(all_items) > input_data.max_items
            if truncated:
                all_items = all_items[:input_data.max_items]

            # Get total count if available
            total_in_webset = None
            if "pagination" in data:
                total_in_webset = data["pagination"].get("total")

            # Yield results
            yield "items", all_items

            # Yield individual items for chaining
            for item in all_items:
                yield "item", item

            yield "total_retrieved", len(all_items)
            yield "total_in_webset", total_in_webset
            yield "truncated", truncated or has_more

        except Exception as e:
            yield "error", str(e)
            yield "items", []
            yield "total_retrieved", 0
            yield "truncated", False


class ExaWebsetItemsSummaryBlock(Block):
    """Get a summary of items in a webset without retrieving all data."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        sample_size: int = SchemaField(
            default=5,
            description="Number of sample items to include",
            ge=0,
            le=10,
        )

    class Output(BlockSchema):
        total_items: int = SchemaField(
            description="Total number of items in the webset",
            default=0,
        )
        entity_type: str = SchemaField(
            description="Type of entities in the webset"
        )
        sample_items: list[dict] = SchemaField(
            description="Sample of items from the webset",
            default_factory=list,
        )
        enrichment_columns: list[str] = SchemaField(
            description="List of enrichment columns available",
            default_factory=list,
        )
        verification_stats: dict = SchemaField(
            description="Statistics about item verification status",
            default_factory=dict,
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="f9b0c1d2-3e4f-5g6h-7i8j-9k0l1m2n3o4p",
            description="Get a summary of webset items without retrieving all data",
            categories={BlockCategory.SEARCH},
            input_schema=ExaWebsetItemsSummaryBlock.Input,
            output_schema=ExaWebsetItemsSummaryBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            # First get webset details
            webset_url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}"
            webset_response = await Requests().get(webset_url, headers=headers)
            webset_data = webset_response.json()

            # Get entity type from searches
            entity_type = "unknown"
            if webset_data.get("searches"):
                first_search = webset_data["searches"][0] if webset_data["searches"] else {}
                entity_type = first_search.get("entity", {}).get("type", "unknown")

            # Get enrichment columns
            enrichment_columns = []
            if webset_data.get("enrichments"):
                enrichment_columns = [e.get("title", e.get("description", ""))
                                     for e in webset_data["enrichments"]]

            # Get sample items if requested
            sample_items = []
            if input_data.sample_size > 0:
                items_url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/items"
                params = {"limit": input_data.sample_size}
                items_response = await Requests().get(items_url, headers=headers, params=params)
                items_data = items_response.json()
                sample_items = items_data.get("data", [])

            # Calculate verification stats from sample
            verification_stats = {}
            if sample_items:
                for item in sample_items:
                    status = item.get("verificationStatus", "unknown")
                    verification_stats[status] = verification_stats.get(status, 0) + 1

            # Get total count
            total_items = 0
            if sample_items and "pagination" in items_data:
                total_items = items_data["pagination"].get("total", len(sample_items))
            else:
                # Estimate from webset status
                for search in webset_data.get("searches", []):
                    progress = search.get("progress", {})
                    total_items += progress.get("found", 0)

            yield "total_items", total_items
            yield "entity_type", entity_type
            yield "sample_items", sample_items
            yield "enrichment_columns", enrichment_columns
            yield "verification_stats", verification_stats

        except Exception as e:
            yield "error", str(e)
            yield "total_items", 0
            yield "entity_type", ""
            yield "sample_items", []
            yield "enrichment_columns", []
            yield "verification_stats", {}