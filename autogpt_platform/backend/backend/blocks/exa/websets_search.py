"""
Exa Websets Search Management Blocks

This module provides blocks for creating and managing searches within websets,
including adding new searches, checking status, and canceling operations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from exa_py import AsyncExa
from exa_py.websets.types import WebsetSearch as SdkWebsetSearch
from pydantic import BaseModel

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


# Mirrored model for stability
class WebsetSearchModel(BaseModel):
    """Stable output model mirroring SDK WebsetSearch."""

    id: str
    webset_id: str
    status: str
    query: str
    entity_type: str
    criteria: List[Dict[str, Any]]
    count: int
    behavior: str
    progress: Dict[str, Any]
    recall: Optional[Dict[str, Any]]
    created_at: str
    updated_at: str
    canceled_at: Optional[str]
    canceled_reason: Optional[str]
    metadata: Dict[str, Any]

    @classmethod
    def from_sdk(cls, search: SdkWebsetSearch) -> "WebsetSearchModel":
        """Convert SDK WebsetSearch to our stable model."""
        # Extract entity type
        entity_type = "auto"
        if search.entity:
            entity_dict = search.entity.model_dump(by_alias=True)
            entity_type = entity_dict.get("type", "auto")

        # Convert criteria
        criteria = [c.model_dump(by_alias=True) for c in search.criteria]

        # Convert progress
        progress_dict = {}
        if search.progress:
            progress_dict = search.progress.model_dump(by_alias=True)

        # Convert recall
        recall_dict = None
        if search.recall:
            recall_dict = search.recall.model_dump(by_alias=True)

        return cls(
            id=search.id,
            webset_id=search.webset_id,
            status=(
                search.status.value
                if hasattr(search.status, "value")
                else str(search.status)
            ),
            query=search.query,
            entity_type=entity_type,
            criteria=criteria,
            count=search.count,
            behavior=search.behavior.value if search.behavior else "override",
            progress=progress_dict,
            recall=recall_dict,
            created_at=search.created_at.isoformat() if search.created_at else "",
            updated_at=search.updated_at.isoformat() if search.updated_at else "",
            canceled_at=search.canceled_at.isoformat() if search.canceled_at else None,
            canceled_reason=(
                search.canceled_reason.value if search.canceled_reason else None
            ),
            metadata=search.metadata if search.metadata else {},
        )


class SearchBehavior(str, Enum):
    """Behavior for how new search results interact with existing items."""

    OVERRIDE = "override"  # Replace existing items
    APPEND = "append"  # Add to existing items
    MERGE = "merge"  # Merge with existing items


class SearchEntityType(str, Enum):
    COMPANY = "company"
    PERSON = "person"
    ARTICLE = "article"
    RESEARCH_PAPER = "research_paper"
    CUSTOM = "custom"
    AUTO = "auto"


class ExaCreateWebsetSearchBlock(Block):
    """Add a new search to an existing webset."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        query: str = SchemaField(
            description="Search query describing what to find",
            placeholder="Engineering managers at Fortune 500 companies",
        )
        count: int = SchemaField(
            default=10,
            description="Number of items to find",
            ge=1,
            le=1000,
        )

        # Entity configuration
        entity_type: SearchEntityType = SchemaField(
            default=SearchEntityType.AUTO,
            description="Type of entity to search for",
        )
        entity_description: Optional[str] = SchemaField(
            default=None,
            description="Description for custom entity type",
            advanced=True,
        )

        # Criteria for verification
        criteria: list[str] = SchemaField(
            default_factory=list,
            description="List of criteria that items must meet. If not provided, auto-detected from query.",
            advanced=True,
        )

        # Advanced search options
        behavior: SearchBehavior = SchemaField(
            default=SearchBehavior.APPEND,
            description="How new results interact with existing items",
            advanced=True,
        )
        recall: bool = SchemaField(
            default=True,
            description="Enable recall estimation for expected results",
            advanced=True,
        )

        # Exclude sources
        exclude_source_ids: list[str] = SchemaField(
            default_factory=list,
            description="IDs of imports/websets to exclude from results",
            advanced=True,
        )
        exclude_source_types: list[str] = SchemaField(
            default_factory=list,
            description="Types of sources to exclude ('import' or 'webset')",
            advanced=True,
        )

        # Scope sources
        scope_source_ids: list[str] = SchemaField(
            default_factory=list,
            description="IDs of imports/websets to limit search scope to",
            advanced=True,
        )
        scope_source_types: list[str] = SchemaField(
            default_factory=list,
            description="Types of scope sources ('import' or 'webset')",
            advanced=True,
        )
        scope_relationships: list[str] = SchemaField(
            default_factory=list,
            description="Relationship definitions for hop searches",
            advanced=True,
        )
        scope_relationship_limits: list[int] = SchemaField(
            default_factory=list,
            description="Limits on related entities to find",
            advanced=True,
        )

        metadata: Optional[dict] = SchemaField(
            default=None,
            description="Metadata to attach to the search",
            advanced=True,
        )

        # Polling options
        wait_for_completion: bool = SchemaField(
            default=False,
            description="Wait for the search to complete before returning",
        )
        polling_timeout: int = SchemaField(
            default=300,
            description="Maximum time to wait for completion in seconds",
            advanced=True,
            ge=1,
            le=600,
        )

    class Output(BlockSchemaOutput):
        search_id: str = SchemaField(
            description="The unique identifier for the created search"
        )
        webset_id: str = SchemaField(description="The webset this search belongs to")
        status: str = SchemaField(description="Current status of the search")
        query: str = SchemaField(description="The search query")
        expected_results: dict = SchemaField(
            description="Recall estimation of expected results"
        )
        items_found: Optional[int] = SchemaField(
            description="Number of items found (if wait_for_completion was True)"
        )
        completion_time: Optional[float] = SchemaField(
            description="Time taken to complete in seconds (if wait_for_completion was True)"
        )

    def __init__(self):
        super().__init__(
            id="342ff776-2e2c-4cdb-b392-4eeb34b21d5f",
            description="Add a new search to an existing webset to find more items",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateWebsetSearchBlock.Input,
            output_schema=ExaCreateWebsetSearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        import time

        # Build the payload
        payload = {
            "query": input_data.query,
            "count": input_data.count,
            "behavior": input_data.behavior.value,
            "recall": input_data.recall,
        }

        # Add entity configuration
        if input_data.entity_type != SearchEntityType.AUTO:
            entity = {"type": input_data.entity_type.value}
            if (
                input_data.entity_type == SearchEntityType.CUSTOM
                and input_data.entity_description
            ):
                entity["description"] = input_data.entity_description
            payload["entity"] = entity

        # Add criteria if provided
        if input_data.criteria:
            payload["criteria"] = [{"description": c} for c in input_data.criteria]

        # Add exclude sources
        if input_data.exclude_source_ids:
            exclude_list = []
            for idx, src_id in enumerate(input_data.exclude_source_ids):
                src_type = "import"
                if input_data.exclude_source_types and idx < len(
                    input_data.exclude_source_types
                ):
                    src_type = input_data.exclude_source_types[idx]
                exclude_list.append({"source": src_type, "id": src_id})
            payload["exclude"] = exclude_list

        # Add scope sources
        if input_data.scope_source_ids:
            scope_list: list[dict[str, Any]] = []
            for idx, src_id in enumerate(input_data.scope_source_ids):
                scope_item: dict[str, Any] = {"source": "import", "id": src_id}

                if input_data.scope_source_types and idx < len(
                    input_data.scope_source_types
                ):
                    scope_item["source"] = input_data.scope_source_types[idx]

                # Add relationship if provided
                if input_data.scope_relationships and idx < len(
                    input_data.scope_relationships
                ):
                    relationship: dict[str, Any] = {
                        "definition": input_data.scope_relationships[idx]
                    }
                    if input_data.scope_relationship_limits and idx < len(
                        input_data.scope_relationship_limits
                    ):
                        relationship["limit"] = input_data.scope_relationship_limits[
                            idx
                        ]
                    scope_item["relationship"] = relationship

                scope_list.append(scope_item)
            payload["scope"] = scope_list

        # Add metadata if provided
        if input_data.metadata:
            payload["metadata"] = input_data.metadata

        start_time = time.time()

        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        sdk_search = aexa.websets.searches.create(
            webset_id=input_data.webset_id, params=payload
        )

        search_id = sdk_search.id
        status = (
            sdk_search.status.value
            if hasattr(sdk_search.status, "value")
            else str(sdk_search.status)
        )

        # Extract expected results from recall
        expected_results = {}
        if sdk_search.recall:
            recall_dict = sdk_search.recall.model_dump(by_alias=True)
            expected = recall_dict.get("expected", {})
            expected_results = {
                "total": expected.get("total", 0),
                "confidence": expected.get("confidence", ""),
                "min": expected.get("bounds", {}).get("min", 0),
                "max": expected.get("bounds", {}).get("max", 0),
                "reasoning": recall_dict.get("reasoning", ""),
            }

        # If wait_for_completion is True, poll for completion
        if input_data.wait_for_completion:
            import asyncio

            poll_interval = 5
            max_interval = 30
            poll_start = time.time()

            while time.time() - poll_start < input_data.polling_timeout:
                current_search = aexa.websets.searches.get(
                    webset_id=input_data.webset_id, id=search_id
                )
                current_status = (
                    current_search.status.value
                    if hasattr(current_search.status, "value")
                    else str(current_search.status)
                )

                if current_status in ["completed", "failed", "cancelled"]:
                    items_found = 0
                    if current_search.progress:
                        items_found = current_search.progress.found
                    completion_time = time.time() - start_time

                    yield "search_id", search_id
                    yield "webset_id", input_data.webset_id
                    yield "status", current_status
                    yield "query", input_data.query
                    yield "expected_results", expected_results
                    yield "items_found", items_found
                    yield "completion_time", completion_time
                    return

                await asyncio.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, max_interval)

            # Timeout - yield what we have
            yield "search_id", search_id
            yield "webset_id", input_data.webset_id
            yield "status", status
            yield "query", input_data.query
            yield "expected_results", expected_results
            yield "items_found", 0
            yield "completion_time", time.time() - start_time
        else:
            yield "search_id", search_id
            yield "webset_id", input_data.webset_id
            yield "status", status
            yield "query", input_data.query
            yield "expected_results", expected_results


class ExaGetWebsetSearchBlock(Block):
    """Get the status and details of a webset search."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        search_id: str = SchemaField(
            description="The ID of the search to retrieve",
            placeholder="search-id",
        )

    class Output(BlockSchemaOutput):
        search_id: str = SchemaField(description="The unique identifier for the search")
        status: str = SchemaField(description="Current status of the search")
        query: str = SchemaField(description="The search query")
        entity_type: str = SchemaField(description="Type of entity being searched")
        criteria: list[dict] = SchemaField(description="Criteria used for verification")
        progress: dict = SchemaField(description="Search progress information")
        recall: dict = SchemaField(description="Recall estimation information")
        created_at: str = SchemaField(description="When the search was created")
        updated_at: str = SchemaField(description="When the search was last updated")
        canceled_at: Optional[str] = SchemaField(
            description="When the search was canceled (if applicable)"
        )
        canceled_reason: Optional[str] = SchemaField(
            description="Reason for cancellation (if applicable)"
        )
        metadata: dict = SchemaField(description="Metadata attached to the search")

    def __init__(self):
        super().__init__(
            id="4fa3e627-a0ff-485f-8732-52148051646c",
            description="Get the status and details of a webset search",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetSearchBlock.Input,
            output_schema=ExaGetWebsetSearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        sdk_search = aexa.websets.searches.get(
            webset_id=input_data.webset_id, id=input_data.search_id
        )

        search = WebsetSearchModel.from_sdk(sdk_search)

        # Extract progress information
        progress_info = {
            "found": search.progress.get("found", 0),
            "analyzed": search.progress.get("analyzed", 0),
            "completion": search.progress.get("completion", 0),
            "time_left": search.progress.get("timeLeft", 0),
        }

        # Extract recall information
        recall_data = {}
        if search.recall:
            expected = search.recall.get("expected", {})
            recall_data = {
                "expected_total": expected.get("total", 0),
                "confidence": expected.get("confidence", ""),
                "min_expected": expected.get("bounds", {}).get("min", 0),
                "max_expected": expected.get("bounds", {}).get("max", 0),
                "reasoning": search.recall.get("reasoning", ""),
            }

        yield "search_id", search.id
        yield "status", search.status
        yield "query", search.query
        yield "entity_type", search.entity_type
        yield "criteria", search.criteria
        yield "progress", progress_info
        yield "recall", recall_data
        yield "created_at", search.created_at
        yield "updated_at", search.updated_at
        yield "canceled_at", search.canceled_at
        yield "canceled_reason", search.canceled_reason
        yield "metadata", search.metadata


class ExaCancelWebsetSearchBlock(Block):
    """Cancel a running webset search."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        search_id: str = SchemaField(
            description="The ID of the search to cancel",
            placeholder="search-id",
        )

    class Output(BlockSchemaOutput):
        search_id: str = SchemaField(description="The ID of the canceled search")
        status: str = SchemaField(description="Status after cancellation")
        items_found_before_cancel: int = SchemaField(
            description="Number of items found before cancellation"
        )
        success: str = SchemaField(
            description="Whether the cancellation was successful"
        )

    def __init__(self):
        super().__init__(
            id="74ef9f1e-ae89-4c7f-9d7d-d217214815b4",
            description="Cancel a running webset search",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCancelWebsetSearchBlock.Input,
            output_schema=ExaCancelWebsetSearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        canceled_search = aexa.websets.searches.cancel(
            webset_id=input_data.webset_id, id=input_data.search_id
        )

        # Extract items found before cancellation
        items_found = 0
        if canceled_search.progress:
            items_found = canceled_search.progress.found

        status = (
            canceled_search.status.value
            if hasattr(canceled_search.status, "value")
            else str(canceled_search.status)
        )

        yield "search_id", canceled_search.id
        yield "status", status
        yield "items_found_before_cancel", items_found
        yield "success", "true"


class ExaFindOrCreateSearchBlock(Block):
    """Find existing search by query or create new one (prevents duplicate searches)."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        query: str = SchemaField(
            description="Search query to find or create",
            placeholder="AI companies in San Francisco",
        )
        count: int = SchemaField(
            default=10,
            description="Number of items to find (only used if creating new search)",
            ge=1,
            le=1000,
        )
        entity_type: SearchEntityType = SchemaField(
            default=SearchEntityType.AUTO,
            description="Entity type (only used if creating)",
            advanced=True,
        )
        behavior: SearchBehavior = SchemaField(
            default=SearchBehavior.OVERRIDE,
            description="Search behavior (only used if creating)",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        search_id: str = SchemaField(description="The search ID (existing or new)")
        webset_id: str = SchemaField(description="The webset ID")
        status: str = SchemaField(description="Current search status")
        query: str = SchemaField(description="The search query")
        was_created: bool = SchemaField(
            description="True if search was newly created, False if already existed"
        )
        items_found: int = SchemaField(
            description="Number of items found (0 if still running)"
        )

    def __init__(self):
        super().__init__(
            id="cbdb05ac-cb73-4b03-a493-6d34e9a011da",
            description="Find existing search by query or create new - prevents duplicate searches in workflows",
            categories={BlockCategory.SEARCH},
            input_schema=ExaFindOrCreateSearchBlock.Input,
            output_schema=ExaFindOrCreateSearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        # Get webset to check existing searches
        webset = aexa.websets.get(id=input_data.webset_id)

        # Look for existing search with same query
        existing_search = None
        if webset.searches:
            for search in webset.searches:
                if search.query.strip().lower() == input_data.query.strip().lower():
                    existing_search = search
                    break

        if existing_search:
            # Found existing search
            search = WebsetSearchModel.from_sdk(existing_search)

            yield "search_id", search.id
            yield "webset_id", input_data.webset_id
            yield "status", search.status
            yield "query", search.query
            yield "was_created", False
            yield "items_found", search.progress.get("found", 0)
        else:
            # Create new search
            payload: Dict[str, Any] = {
                "query": input_data.query,
                "count": input_data.count,
                "behavior": input_data.behavior.value,
            }

            # Add entity if not auto
            if input_data.entity_type != SearchEntityType.AUTO:
                payload["entity"] = {"type": input_data.entity_type.value}

            sdk_search = aexa.websets.searches.create(
                webset_id=input_data.webset_id, params=payload
            )

            search = WebsetSearchModel.from_sdk(sdk_search)

            yield "search_id", search.id
            yield "webset_id", input_data.webset_id
            yield "status", search.status
            yield "query", search.query
            yield "was_created", True
            yield "items_found", 0  # Newly created, no items yet
