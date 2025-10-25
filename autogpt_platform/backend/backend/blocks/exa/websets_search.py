"""
Exa Websets Search Management Blocks

This module provides blocks for creating and managing searches within websets,
including adding new searches, checking status, and canceling operations.
"""

from enum import Enum
from typing import Optional

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

    class Input(BlockSchema):
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

    class Output(BlockSchema):
        search_id: str = SchemaField(
            description="The unique identifier for the created search"
        )
        webset_id: str = SchemaField(description="The webset this search belongs to")
        status: str = SchemaField(description="Current status of the search")
        query: str = SchemaField(description="The search query")
        expected_results: dict = SchemaField(
            description="Recall estimation of expected results",
            default_factory=dict,
        )
        items_found: Optional[int] = SchemaField(
            description="Number of items found (if wait_for_completion was True)",
            default=None,
        )
        completion_time: Optional[float] = SchemaField(
            description="Time taken to complete in seconds (if wait_for_completion was True)",
            default=None,
        )
        error: str = SchemaField(
            description="Error message if the operation failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-b8c9-0123-def4-567890123456",
            description="Add a new search to an existing webset to find more items",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateWebsetSearchBlock.Input,
            output_schema=ExaCreateWebsetSearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        import time

        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/searches"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

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
            scope_list = []
            for idx, src_id in enumerate(input_data.scope_source_ids):
                scope_item = {"source": "import", "id": src_id}

                if input_data.scope_source_types and idx < len(
                    input_data.scope_source_types
                ):
                    scope_item["source"] = input_data.scope_source_types[idx]

                # Add relationship if provided
                if input_data.scope_relationships and idx < len(
                    input_data.scope_relationships
                ):
                    relationship = {"definition": input_data.scope_relationships[idx]}
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

        try:
            start_time = time.time()

            # Create the search
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()

            search_id = data.get("id", "")
            status = data.get("status", "")

            # Extract expected results from recall
            expected_results = {}
            if "recall" in data:
                recall = data["recall"]
                expected = recall.get("expected", {})
                expected_results = {
                    "total": expected.get("total", 0),
                    "confidence": expected.get("confidence", ""),
                    "min": expected.get("bounds", {}).get("min", 0),
                    "max": expected.get("bounds", {}).get("max", 0),
                    "reasoning": recall.get("reasoning", ""),
                }

            # If wait_for_completion is True, poll for completion
            if input_data.wait_for_completion:
                items_found = await self._poll_for_completion(
                    input_data.webset_id,
                    search_id,
                    credentials.api_key.get_secret_value(),
                    input_data.polling_timeout,
                )
                completion_time = time.time() - start_time

                yield "search_id", search_id
                yield "webset_id", input_data.webset_id
                yield "status", "completed"
                yield "query", input_data.query
                yield "expected_results", expected_results
                yield "items_found", items_found
                yield "completion_time", completion_time
            else:
                yield "search_id", search_id
                yield "webset_id", input_data.webset_id
                yield "status", status
                yield "query", input_data.query
                yield "expected_results", expected_results

        except Exception as e:
            yield "error", str(e)
            yield "search_id", ""
            yield "webset_id", input_data.webset_id
            yield "status", ""
            yield "query", input_data.query
            yield "expected_results", {}

    async def _poll_for_completion(
        self, webset_id: str, search_id: str, api_key: str, timeout: int
    ) -> int:
        """Poll search status until it completes or times out."""
        import asyncio
        import time

        start_time = time.time()
        interval = 5
        max_interval = 30

        url = f"https://api.exa.ai/websets/v0/websets/{webset_id}/searches/{search_id}"
        headers = {"x-api-key": api_key}

        while time.time() - start_time < timeout:
            try:
                response = await Requests().get(url, headers=headers)
                data = response.json()

                status = data.get("status", "")
                progress = data.get("progress", {})

                if status in ["completed", "failed", "canceled"]:
                    return progress.get("found", 0)

                await asyncio.sleep(interval)
                interval = min(interval * 1.5, max_interval)

            except Exception:
                await asyncio.sleep(interval)

        return 0


class ExaGetWebsetSearchBlock(Block):
    """Get the status and details of a webset search."""

    class Input(BlockSchema):
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

    class Output(BlockSchema):
        search_id: str = SchemaField(description="The unique identifier for the search")
        status: str = SchemaField(description="Current status of the search")
        query: str = SchemaField(description="The search query")
        entity_type: str = SchemaField(description="Type of entity being searched")
        criteria: list[dict] = SchemaField(
            description="Criteria used for verification",
            default_factory=list,
        )
        progress: dict = SchemaField(
            description="Search progress information",
            default_factory=dict,
        )
        recall: dict = SchemaField(
            description="Recall estimation information",
            default_factory=dict,
        )
        created_at: str = SchemaField(description="When the search was created")
        updated_at: str = SchemaField(description="When the search was last updated")
        canceled_at: Optional[str] = SchemaField(
            description="When the search was canceled (if applicable)",
            default=None,
        )
        canceled_reason: Optional[str] = SchemaField(
            description="Reason for cancellation (if applicable)",
            default=None,
        )
        metadata: dict = SchemaField(
            description="Metadata attached to the search",
            default_factory=dict,
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="e5f6a7b8-c9d0-1234-ef56-789012345678",
            description="Get the status and details of a webset search",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetSearchBlock.Input,
            output_schema=ExaGetWebsetSearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/searches/{input_data.search_id}"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            response = await Requests().get(url, headers=headers)
            data = response.json()

            # Extract entity information
            entity = data.get("entity", {})
            entity_type = entity.get("type", "unknown")

            # Extract progress information
            progress = data.get("progress", {})
            progress_info = {
                "found": progress.get("found", 0),
                "analyzed": progress.get("analyzed", 0),
                "completion": progress.get("completion", 0),
                "time_left": progress.get("timeLeft", 0),
            }

            # Extract recall information
            recall_data = {}
            if "recall" in data:
                recall = data["recall"]
                expected = recall.get("expected", {})
                recall_data = {
                    "expected_total": expected.get("total", 0),
                    "confidence": expected.get("confidence", ""),
                    "min_expected": expected.get("bounds", {}).get("min", 0),
                    "max_expected": expected.get("bounds", {}).get("max", 0),
                    "reasoning": recall.get("reasoning", ""),
                }

            yield "search_id", data.get("id", "")
            yield "status", data.get("status", "")
            yield "query", data.get("query", "")
            yield "entity_type", entity_type
            yield "criteria", data.get("criteria", [])
            yield "progress", progress_info
            yield "recall", recall_data
            yield "created_at", data.get("createdAt", "")
            yield "updated_at", data.get("updatedAt", "")
            yield "canceled_at", data.get("canceledAt")
            yield "canceled_reason", data.get("canceledReason")
            yield "metadata", data.get("metadata", {})

        except Exception as e:
            yield "error", str(e)
            yield "search_id", ""
            yield "status", ""
            yield "query", ""
            yield "entity_type", ""
            yield "criteria", []
            yield "progress", {}
            yield "recall", {}
            yield "created_at", ""
            yield "updated_at", ""
            yield "metadata", {}


class ExaCancelWebsetSearchBlock(Block):
    """Cancel a running webset search."""

    class Input(BlockSchema):
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
        reason: Optional[str] = SchemaField(
            default=None,
            description="Optional reason for cancellation",
            advanced=True,
        )

    class Output(BlockSchema):
        search_id: str = SchemaField(description="The ID of the canceled search")
        status: str = SchemaField(description="Status after cancellation")
        items_found_before_cancel: int = SchemaField(
            description="Number of items found before cancellation",
            default=0,
        )
        success: str = SchemaField(
            description="Whether the cancellation was successful",
            default="true",
        )
        error: str = SchemaField(
            description="Error message if the cancellation failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="f6a7b8c9-d0e1-2345-f678-901234567890",
            description="Cancel a running webset search",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCancelWebsetSearchBlock.Input,
            output_schema=ExaCancelWebsetSearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/searches/{input_data.search_id}/cancel"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        # Add reason to payload if provided
        payload = {}
        if input_data.reason:
            headers["Content-Type"] = "application/json"
            payload = {"reason": input_data.reason}

        try:
            if payload:
                response = await Requests().post(url, headers=headers, json=payload)
            else:
                response = await Requests().post(url, headers=headers)

            data = response.json()

            # Get the search details to see how many items were found
            search_url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/searches/{input_data.search_id}"
            search_response = await Requests().get(
                search_url, headers={"x-api-key": headers["x-api-key"]}
            )
            search_data = search_response.json()

            progress = search_data.get("progress", {})
            items_found = progress.get("found", 0)

            yield "search_id", input_data.search_id
            yield "status", search_data.get("status", "canceled")
            yield "items_found_before_cancel", items_found
            yield "success", "true"

        except Exception as e:
            yield "error", str(e)
            yield "search_id", input_data.search_id
            yield "status", ""
            yield "items_found_before_cancel", 0
            yield "success", "false"
