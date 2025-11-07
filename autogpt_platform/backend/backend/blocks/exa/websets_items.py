"""
Exa Websets Item Management Blocks

This module provides blocks for managing items within Exa websets, including
retrieving, listing, deleting, and bulk operations on webset items.
"""

from typing import Any, Dict, List, Optional

from exa_py import AsyncExa
from exa_py.websets.types import WebsetItem as SdkWebsetItem
from exa_py.websets.types import (
    WebsetItemArticleProperties,
    WebsetItemCompanyProperties,
    WebsetItemCustomProperties,
    WebsetItemPersonProperties,
    WebsetItemResearchPaperProperties,
)
from pydantic import AnyUrl, BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa


# Mirrored model for enrichment results
class EnrichmentResultModel(BaseModel):
    """Stable output model mirroring SDK EnrichmentResult."""

    enrichment_id: str
    format: str
    result: Optional[List[str]]
    reasoning: Optional[str]
    references: List[Dict[str, Any]]

    @classmethod
    def from_sdk(cls, sdk_enrich) -> "EnrichmentResultModel":
        """Convert SDK EnrichmentResult to our model."""
        format_str = (
            sdk_enrich.format.value
            if hasattr(sdk_enrich.format, "value")
            else str(sdk_enrich.format)
        )

        # Convert references to dicts
        references_list = []
        if sdk_enrich.references:
            for ref in sdk_enrich.references:
                references_list.append(ref.model_dump(by_alias=True, exclude_none=True))

        return cls(
            enrichment_id=sdk_enrich.enrichment_id,
            format=format_str,
            result=sdk_enrich.result,
            reasoning=sdk_enrich.reasoning,
            references=references_list,
        )


# Mirrored model for stability - don't use SDK types directly in block outputs
class WebsetItemModel(BaseModel):
    """Stable output model mirroring SDK WebsetItem."""

    id: str
    url: Optional[AnyUrl]
    title: str
    content: str
    entity_data: Dict[str, Any]
    enrichments: Dict[str, EnrichmentResultModel]
    created_at: str
    updated_at: str

    @classmethod
    def from_sdk(cls, item: SdkWebsetItem) -> "WebsetItemModel":
        """Convert SDK WebsetItem to our stable model."""
        # Extract properties from the union type
        properties_dict = {}
        url_value = None
        title = ""
        content = ""

        if hasattr(item, "properties") and item.properties:
            properties_dict = item.properties.model_dump(
                by_alias=True, exclude_none=True
            )

            # URL is always available on all property types
            url_value = item.properties.url

            # Extract title using isinstance checks on the union type
            if isinstance(item.properties, WebsetItemPersonProperties):
                title = item.properties.person.name
                content = ""  # Person type has no content
            elif isinstance(item.properties, WebsetItemCompanyProperties):
                title = item.properties.company.name
                content = item.properties.content or ""
            elif isinstance(item.properties, WebsetItemArticleProperties):
                title = item.properties.description
                content = item.properties.content or ""
            elif isinstance(item.properties, WebsetItemResearchPaperProperties):
                title = item.properties.description
                content = item.properties.content or ""
            elif isinstance(item.properties, WebsetItemCustomProperties):
                title = item.properties.description
                content = item.properties.content or ""
            else:
                # Fallback
                title = item.properties.description
                content = getattr(item.properties, "content", "")

        # Convert enrichments from list to dict keyed by enrichment_id using Pydantic models
        enrichments_dict: Dict[str, EnrichmentResultModel] = {}
        if hasattr(item, "enrichments") and item.enrichments:
            for sdk_enrich in item.enrichments:
                enrich_model = EnrichmentResultModel.from_sdk(sdk_enrich)
                enrichments_dict[enrich_model.enrichment_id] = enrich_model

        return cls(
            id=item.id,
            url=url_value,
            title=title,
            content=content or "",
            entity_data=properties_dict,
            enrichments=enrichments_dict,
            created_at=item.created_at.isoformat() if item.created_at else "",
            updated_at=item.updated_at.isoformat() if item.updated_at else "",
        )


class ExaGetWebsetItemBlock(Block):
    """Get a specific item from a webset by its ID."""

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        item_id: str = SchemaField(description="The unique identifier for the item")
        url: str = SchemaField(description="The URL of the original source")
        title: str = SchemaField(description="The title of the item")
        content: str = SchemaField(description="The main content of the item")
        entity_data: dict = SchemaField(description="Entity-specific structured data")
        enrichments: dict = SchemaField(description="Enrichment data added to the item")
        created_at: str = SchemaField(
            description="When the item was added to the webset"
        )
        updated_at: str = SchemaField(description="When the item was last updated")

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
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        sdk_item = aexa.websets.items.get(
            webset_id=input_data.webset_id, id=input_data.item_id
        )

        item = WebsetItemModel.from_sdk(sdk_item)

        yield "item_id", item.id
        yield "url", item.url
        yield "title", item.title
        yield "content", item.content
        yield "entity_data", item.entity_data
        yield "enrichments", item.enrichments
        yield "created_at", item.created_at
        yield "updated_at", item.updated_at


class ExaListWebsetItemsBlock(Block):
    """List items in a webset with pagination and optional filtering."""

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        items: list[WebsetItemModel] = SchemaField(
            description="List of webset items",
        )
        webset_id: str = SchemaField(
            description="The ID of the webset",
        )
        item: WebsetItemModel = SchemaField(
            description="Individual item (yielded for each item in the list)",
        )
        has_more: bool = SchemaField(
            description="Whether there are more items to paginate through",
        )
        next_cursor: Optional[str] = SchemaField(
            description="Cursor for the next page of results",
        )

    def __init__(self):
        super().__init__(
            id="7b5e8c9f-01a2-43c4-95e6-f7a8b9c0d1e2",
            description="List items in a webset with pagination support",
            categories={BlockCategory.SEARCH},
            input_schema=ExaListWebsetItemsBlock.Input,
            output_schema=ExaListWebsetItemsBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

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

        items = [WebsetItemModel.from_sdk(item) for item in response.data]

        yield "items", items

        for item in items:
            yield "item", item

        yield "has_more", response.has_more
        yield "next_cursor", response.next_cursor
        yield "webset_id", input_data.webset_id


class ExaDeleteWebsetItemBlock(Block):
    """Delete a specific item from a webset."""

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        item_id: str = SchemaField(description="The ID of the deleted item")
        success: str = SchemaField(description="Whether the deletion was successful")

    def __init__(self):
        super().__init__(
            id="12c57fbe-c270-4877-a2b6-d2d05529ba79",
            description="Delete a specific item from a webset",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteWebsetItemBlock.Input,
            output_schema=ExaDeleteWebsetItemBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        deleted_item = aexa.websets.items.delete(
            webset_id=input_data.webset_id, id=input_data.item_id
        )

        yield "item_id", deleted_item.id
        yield "success", "true"


class ExaBulkWebsetItemsBlock(Block):
    """Get all items from a webset in a single operation (with size limits)."""

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        items: list[WebsetItemModel] = SchemaField(
            description="All items from the webset"
        )
        item: WebsetItemModel = SchemaField(
            description="Individual item (yielded for each item)"
        )
        total_retrieved: int = SchemaField(
            description="Total number of items retrieved"
        )
        truncated: bool = SchemaField(
            description="Whether results were truncated due to max_items limit"
        )

    def __init__(self):
        super().__init__(
            id="dbd619f5-476e-4395-af9a-a7a7c0fb8c4e",
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

        all_items: List[WebsetItemModel] = []
        item_iterator = aexa.websets.items.list_all(
            webset_id=input_data.webset_id, limit=input_data.max_items
        )

        for sdk_item in item_iterator:
            if len(all_items) >= input_data.max_items:
                break

            item = WebsetItemModel.from_sdk(sdk_item)

            if not input_data.include_enrichments:
                item.enrichments = {}
            if not input_data.include_content:
                item.content = ""

            all_items.append(item)

        yield "items", all_items

        for item in all_items:
            yield "item", item

        yield "total_retrieved", len(all_items)
        yield "truncated", len(all_items) >= input_data.max_items


class ExaWebsetItemsSummaryBlock(Block):
    """Get a summary of items in a webset without retrieving all data."""

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
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

    def __init__(self):
        super().__init__(
            id="db7813ad-10bd-4652-8623-5667d6fecdd5",
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

        webset = aexa.websets.get(id=input_data.webset_id)

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

        total_items = 0
        if webset.searches:
            for search in webset.searches:
                if search.progress:
                    total_items += search.progress.found

        yield "total_items", total_items
        yield "entity_type", entity_type
        yield "sample_items", sample_items
        yield "enrichment_columns", enrichment_columns


class ExaGetNewItemsBlock(Block):
    """Get items added to a webset since a specific cursor (incremental processing helper)."""

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
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

    def __init__(self):
        super().__init__(
            id="3ff9bdf5-9613-4d21-8a60-90eb8b69c414",
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
