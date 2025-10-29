"""
Exa Websets Item Management Blocks

This module provides blocks for managing items within Exa websets, including
retrieving, listing, deleting, and bulk operations on webset items.
"""

from typing import Any, Dict, List, Optional

from exa_py import AsyncExa
from exa_py.websets.types import WebsetItem as SdkWebsetItem
from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa


# Mirrored model for stability - don't use SDK types directly in block outputs
class WebsetItemModel(BaseModel):
    """Stable output model mirroring SDK WebsetItem."""

    id: str
    url: str
    title: str
    content: str
    entity_data: Dict[str, Any]
    enrichments: Dict[str, Any]
    verification_status: str
    created_at: str
    updated_at: str

    @classmethod
    def from_sdk(cls, item: SdkWebsetItem) -> "WebsetItemModel":
        """Convert SDK WebsetItem to our stable model."""
        # Extract properties from the union type
        properties_dict = {}
        if hasattr(item, "properties") and item.properties:
            properties_dict = item.properties.model_dump(
                by_alias=True, exclude_none=True
            )

        # Convert enrichments from list to dict keyed by enrichment_id
        enrichments_dict = {}
        if hasattr(item, "enrichments") and item.enrichments:
            for enrich in item.enrichments:
                enrichment_data = enrich.model_dump(by_alias=True, exclude_none=True)
                enrichments_dict[enrich.enrichment_id] = enrichment_data

        return cls(
            id=item.id,
            url=properties_dict.get("url", ""),
            title=properties_dict.get("title", ""),
            content=properties_dict.get("content", ""),
            entity_data=properties_dict,
            enrichments=enrichments_dict,
            verification_status="",  # Not yet exposed in SDK
            created_at=item.created_at.isoformat() if item.created_at else "",
            updated_at=item.updated_at.isoformat() if item.updated_at else "",
        )


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
        item_id: str = SchemaField(description="The unique identifier for the item")
        url: str = SchemaField(description="The URL of the original source")
        title: str = SchemaField(description="The title of the item")
        content: str = SchemaField(description="The main content of the item")
        entity_data: dict = SchemaField(description="Entity-specific structured data")
        enrichments: dict = SchemaField(description="Enrichment data added to the item")
        verification_status: str = SchemaField(
            description="Verification status against criteria"
        )
        created_at: str = SchemaField(
            description="When the item was added to the webset"
        )
        updated_at: str = SchemaField(description="When the item was last updated")
        error: str = SchemaField(description="Error message if the request failed")

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
        # Use AsyncExa SDK (methods are sync but class is AsyncExa)
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        # Get item using SDK - no await needed
        sdk_item = aexa.websets.items.get(
            webset_id=input_data.webset_id, id=input_data.item_id
        )

        # Convert to our stable model
        item = WebsetItemModel.from_sdk(sdk_item)

        # Yield all fields
        yield "item_id", item.id
        yield "url", item.url
        yield "title", item.title
        yield "content", item.content
        yield "entity_data", item.entity_data
        yield "enrichments", item.enrichments
        yield "verification_status", item.verification_status
        yield "created_at", item.created_at
        yield "updated_at", item.updated_at


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
        items: list[WebsetItemModel] = SchemaField(
            description="List of webset items",
        )
        item: WebsetItemModel = SchemaField(
            description="Individual item (yielded for each item in the list)",
        )
        total_count: Optional[int] = SchemaField(
            description="Total number of items in the webset",
        )
        has_more: bool = SchemaField(
            description="Whether there are more items to paginate through",
        )
        next_cursor: Optional[str] = SchemaField(
            description="Cursor for the next page of results",
        )
        error: str = SchemaField(description="Error message if the request failed")

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
        # Use AsyncExa SDK (methods are sync despite class name)
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        # If wait_for_items is True, poll until items are available
        if input_data.wait_for_items:
            import asyncio
            import time

            start_time = time.time()
            interval = 2
            response = None

            while time.time() - start_time < input_data.wait_timeout:
                response = aexa.websets.items.list(
                    webset_id=input_data.webset_id,
                    cursor=input_data.cursor,
                    limit=input_data.limit,
                )

                # Check if we have any items
                if response.data:
                    break

                await asyncio.sleep(interval)
                interval = min(interval * 1.2, 10)

            if not response:
                response = aexa.websets.items.list(
                    webset_id=input_data.webset_id,
                    cursor=input_data.cursor,
                    limit=input_data.limit,
                )
        else:
            response = aexa.websets.items.list(
                webset_id=input_data.webset_id,
                cursor=input_data.cursor,
                limit=input_data.limit,
            )

        # Convert SDK items to our stable models
        items = [WebsetItemModel.from_sdk(item) for item in response.data]

        # Yield the full list
        yield "items", items

        # Yield individual items for graph chaining
        for item in items:
            yield "item", item

        # Yield pagination metadata
        yield "total_count", None  # SDK doesn't provide total in pagination
        yield "has_more", response.has_more
        yield "next_cursor", response.next_cursor


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
        item_id: str = SchemaField(description="The ID of the deleted item")
        success: str = SchemaField(description="Whether the deletion was successful")
        error: str = SchemaField(description="Error message if the request failed")

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
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        # Delete item using SDK
        deleted_item = aexa.websets.items.delete(
            webset_id=input_data.webset_id, id=input_data.item_id
        )

        yield "item_id", deleted_item.id
        yield "success", "true"


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
        items: list[WebsetItemModel] = SchemaField(
            description="All items from the webset"
        )
        item: WebsetItemModel = SchemaField(
            description="Individual item (yielded for each item)"
        )
        total_retrieved: int = SchemaField(
            description="Total number of items retrieved"
        )
        total_in_webset: Optional[int] = SchemaField(
            description="Total number of items in the webset"
        )
        truncated: bool = SchemaField(
            description="Whether results were truncated due to max_items limit"
        )
        error: str = SchemaField(description="Error message if the request failed")

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
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        # Use list_all iterator to get items up to max_items
        all_items: List[WebsetItemModel] = []
        item_iterator = aexa.websets.items.list_all(
            webset_id=input_data.webset_id, limit=input_data.max_items
        )

        for sdk_item in item_iterator:
            if len(all_items) >= input_data.max_items:
                break

            # Convert to our stable model
            item = WebsetItemModel.from_sdk(sdk_item)

            # Apply field filters by setting to empty values
            if not input_data.include_enrichments:
                item.enrichments = {}
            if not input_data.include_content:
                item.content = ""

            all_items.append(item)

        # Yield results
        yield "items", all_items

        # Yield individual items for chaining
        for item in all_items:
            yield "item", item

        yield "total_retrieved", len(all_items)
        yield "total_in_webset", None  # SDK doesn't provide total count
        yield "truncated", len(all_items) >= input_data.max_items


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
            description="Total number of items in the webset"
        )
        entity_type: str = SchemaField(description="Type of entities in the webset")
        sample_items: list[WebsetItemModel] = SchemaField(
            description="Sample of items from the webset"
        )
        enrichment_columns: list[str] = SchemaField(
            description="List of enrichment columns available"
        )
        verification_stats: dict = SchemaField(
            description="Statistics about item verification status"
        )
        error: str = SchemaField(description="Error message if the request failed")

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
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        # Get webset details
        webset = aexa.websets.get(id=input_data.webset_id)

        # Get entity type from searches
        entity_type = "unknown"
        if webset.searches:
            first_search = webset.searches[0]
            if first_search.entity:
                # The entity is a union type, extract type field
                entity_dict = first_search.entity.model_dump(by_alias=True)
                entity_type = entity_dict.get("type", "unknown")

        # Get enrichment columns
        enrichment_columns = []
        if webset.enrichments:
            enrichment_columns = [
                e.title if e.title else e.description for e in webset.enrichments
            ]

        # Get sample items if requested
        sample_items: List[WebsetItemModel] = []
        if input_data.sample_size > 0:
            items_response = aexa.websets.items.list(
                webset_id=input_data.webset_id, limit=input_data.sample_size
            )
            # Convert to our stable models
            sample_items = [
                WebsetItemModel.from_sdk(item) for item in items_response.data
            ]

        # Calculate verification stats from sample (not yet in SDK)
        verification_stats: dict[str, int] = {}

        # Get total count - estimate from search progress
        total_items = 0
        if webset.searches:
            for search in webset.searches:
                if search.progress:
                    total_items += search.progress.found

        yield "total_items", total_items
        yield "entity_type", entity_type
        yield "sample_items", sample_items
        yield "enrichment_columns", enrichment_columns
        yield "verification_stats", verification_stats


class ExaGetNewItemsBlock(Block):
    """Get items added to a webset since a specific cursor (incremental processing helper)."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        since_cursor: Optional[str] = SchemaField(
            default=None,
            description="Cursor from previous run - only items after this will be returned. Leave empty on first run.",
            placeholder="cursor-from-previous-run",
        )
        max_items: int = SchemaField(
            default=100,
            description="Maximum number of new items to retrieve",
            ge=1,
            le=1000,
        )

    class Output(BlockSchema):
        new_items: list[WebsetItemModel] = SchemaField(
            description="Items added since the cursor"
        )
        item: WebsetItemModel = SchemaField(
            description="Individual item (yielded for each new item)"
        )
        count: int = SchemaField(description="Number of new items found")
        next_cursor: Optional[str] = SchemaField(
            description="Save this cursor for the next run to get only newer items"
        )
        has_more: bool = SchemaField(
            description="Whether there are more new items beyond max_items"
        )
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-7890-1234-5678-901234567890",
            description="Get items added since a cursor - enables incremental processing without reprocessing",
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=ExaGetNewItemsBlock.Input,
            output_schema=ExaGetNewItemsBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        # Get items starting from cursor
        response = aexa.websets.items.list(
            webset_id=input_data.webset_id,
            cursor=input_data.since_cursor,
            limit=input_data.max_items,
        )

        # Convert SDK items to our stable models
        new_items = [WebsetItemModel.from_sdk(item) for item in response.data]

        # Yield the full list
        yield "new_items", new_items

        # Yield individual items for processing
        for item in new_items:
            yield "item", item

        # Yield metadata for next run
        yield "count", len(new_items)
        yield "next_cursor", response.next_cursor
        yield "has_more", response.has_more
