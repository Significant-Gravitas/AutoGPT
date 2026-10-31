"""
Exa Websets Search Management Blocks

This module provides blocks for creating and managing searches within websets,
including adding new searches, checking status, and canceling operations.
"""

import time
from asyncio import sleep
from enum import Enum
from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .websets_api import ExaWebsetsBlockBase, segment
from .websets_fixtures import SEARCH_ID, WEBSET_ID, sample_search
from .websets_payloads import (
    criteria_payload,
    entity_payload,
    scope_payload,
    sources_payload,
)
from .websets_types import ApiWebsetSearch, recall_summary

SEARCH_DONE_STATUSES = ("completed", "canceled", "failed")


class SearchBehavior(str, Enum):
    """Behavior for how new search results interact with existing items."""

    OVERRIDE = "override"  # Replace existing items
    APPEND = "append"  # Add to existing items
    MERGE = "merge"  # Legacy: Exa no longer has 'merge'; sent as 'append'


class SearchEntityType(str, Enum):
    COMPANY = "company"
    PERSON = "person"
    ARTICLE = "article"
    RESEARCH_PAPER = "research_paper"
    CUSTOM = "custom"
    AUTO = "auto"


BEHAVIOR_DESCRIPTION = (
    "What happens to the webset's existing items: 'override' re-evaluates them "
    "against this search's criteria and drops the ones that fail; 'append' keeps "
    "them and adds the new results. 'merge' is legacy and is sent as 'append'."
)


class ExaCreateWebsetSearchBlock(ExaWebsetsBlockBase):
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
            description="What to find, in plain language (up to 5,000 characters)",
            placeholder="Engineering managers at Fortune 500 companies",
        )
        count: int = SchemaField(
            default=10,
            description="Number of items to find. Exa may find fewer.",
            ge=1,
            le=1000,
        )

        # Entity configuration
        entity_type: SearchEntityType = SchemaField(
            default=SearchEntityType.AUTO,
            description="Kind of result to find; 'auto' lets Exa detect it from the query, 'custom' needs a description",
        )
        entity_description: str | None = SchemaField(
            default=None,
            description="What each result is, when the entity type is 'custom' (2-200 characters)",
            advanced=True,
        )

        # Criteria for verification
        criteria: list[str] = SchemaField(
            default_factory=list,
            description="Up to 5 criteria every item must meet. If not provided, auto-detected from query.",
            advanced=True,
        )

        # Advanced search options
        behavior: SearchBehavior = SchemaField(
            default=SearchBehavior.APPEND,
            description=BEHAVIOR_DESCRIPTION,
            advanced=True,
        )
        recall: bool = SchemaField(
            default=True,
            description="Also estimate how many matching results exist (returned in 'expected_results')",
            advanced=True,
        )
        max_people_per_company: int | None = SchemaField(
            default=None,
            description="For people searches: try to include at most this many people from the same current employer (soft cap)",
            ge=1,
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
            description="Type of each exclude source, in the same order ('import' or 'webset'; defaults to 'import')",
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
            description="Type of each scope source, in the same order ('import' or 'webset'; defaults to 'import')",
            advanced=True,
        )
        scope_relationships: list[str] = SchemaField(
            default_factory=list,
            description="Optional, one per scope source: how results relate to that source's items, for hop searches (e.g. 'CEOs of these companies')",
            advanced=True,
        )
        scope_relationship_limits: list[int] = SchemaField(
            default_factory=list,
            description="Required for each relationship: related entities to find per source item (1-10)",
            advanced=True,
        )

        metadata: dict | None = SchemaField(
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
        items_found: int | None = SchemaField(
            description="Number of items found (if wait_for_completion was True)"
        )
        completion_time: float | None = SchemaField(
            description="Time taken to complete in seconds (if wait_for_completion was True)"
        )

    def __init__(self):
        super().__init__(
            id="342ff776-2e2c-4cdb-b392-4eeb34b21d5f",
            description="Add a new search to an existing webset to find more items",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCreateWebsetSearchBlock.Input,
            output_schema=ExaCreateWebsetSearchBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "webset_id": WEBSET_ID,
                "query": "AI startups in Europe",
            },
            test_output=[
                ("search_id", SEARCH_ID),
                ("webset_id", WEBSET_ID),
                ("status", "pending"),
                ("query", "AI startups in Europe"),
                ("expected_results", {}),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_api": lambda *a, **k: sample_search(status="pending")},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        start_time = time.time()
        searches_path = f"/websets/{segment(input_data.webset_id)}/searches"
        created = ApiWebsetSearch.model_validate(
            await self._api(
                credentials,
                "POST",
                searches_path,
                json=create_search_payload(input_data),
            )
        )
        expected_results = _expected_results(created.recall)

        yield "search_id", created.id
        yield "webset_id", input_data.webset_id
        if not input_data.wait_for_completion:
            yield "status", created.status
            yield "query", input_data.query
            yield "expected_results", expected_results
            return

        final = await self._poll_search(
            credentials,
            f"{searches_path}/{segment(created.id)}",
            input_data.polling_timeout,
        )
        done = final is not None and final.status in SEARCH_DONE_STATUSES
        yield "status", final.status if final and done else created.status
        yield "query", input_data.query
        yield "expected_results", expected_results
        yield "items_found", final.found if final and done else 0
        yield "completion_time", time.time() - start_time

    async def _poll_search(
        self, credentials: APIKeyCredentials, path: str, timeout: int
    ) -> ApiWebsetSearch | None:
        """Poll until the search finishes; return the last state seen."""
        interval, poll_start = 5.0, time.time()
        latest: ApiWebsetSearch | None = None
        while time.time() - poll_start < timeout:
            latest = ApiWebsetSearch.model_validate(
                await self._api(credentials, "GET", path)
            )
            if latest.status in SEARCH_DONE_STATUSES:
                return latest
            await sleep(interval)
            interval = min(interval * 1.5, 30)
        return latest


def create_search_payload(data: ExaCreateWebsetSearchBlock.Input) -> dict[str, Any]:
    """Build the ``POST /v0/websets/{id}/searches`` body."""
    payload: dict[str, Any] = {
        "query": data.query,
        "count": data.count,
        "behavior": search_behavior_value(data.behavior),
        "recall": data.recall,
    }
    optional = {
        "entity": entity_payload(data.entity_type.value, data.entity_description),
        "criteria": criteria_payload(data.criteria),
        "exclude": sources_payload(data.exclude_source_ids, data.exclude_source_types),
        "scope": scope_payload(
            data.scope_source_ids,
            data.scope_source_types,
            data.scope_relationships,
            data.scope_relationship_limits,
        ),
        "maxPeoplePerCompany": data.max_people_per_company,
        "metadata": data.metadata,
    }
    payload.update({k: v for k, v in optional.items() if v})
    return payload


def search_behavior_value(behavior: SearchBehavior) -> str:
    """The spec value for a behavior; legacy 'merge' maps to 'append'."""
    if behavior == SearchBehavior.MERGE:
        return SearchBehavior.APPEND.value
    return behavior.value


def _expected_results(recall: dict[str, Any] | None) -> dict[str, Any]:
    summary = recall_summary(recall)
    if not summary:
        return {}
    return {
        "total": summary["expected_total"],
        "confidence": summary["confidence"],
        "min": summary["min_expected"],
        "max": summary["max_expected"],
        "reasoning": summary["reasoning"],
    }


class ExaGetWebsetSearchBlock(ExaWebsetsBlockBase):
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
        status: str = SchemaField(
            description="Current status: created, pending, running, completed or canceled"
        )
        query: str = SchemaField(description="The search query")
        entity_type: str = SchemaField(description="Type of entity being searched")
        criteria: list[dict] = SchemaField(description="Criteria used for verification")
        count: int = SchemaField(description="Number of items the search aims to find")
        behavior: str | None = SchemaField(
            description="How the search treats existing items: override or append"
        )
        max_people_per_company: int | None = SchemaField(
            description="Soft cap on people per employer, if one was requested"
        )
        exclude: list[dict] = SchemaField(
            description="Imports or websets whose items this search leaves out"
        )
        scope: list[dict] = SchemaField(
            description="Imports or websets the search is limited to (empty for the whole web)"
        )
        progress: dict = SchemaField(description="Search progress information")
        recall: dict = SchemaField(description="Recall estimation information")
        created_at: str = SchemaField(description="When the search was created")
        updated_at: str = SchemaField(description="When the search was last updated")
        canceled_at: str | None = SchemaField(
            description="When the search was canceled (if applicable)"
        )
        canceled_reason: str | None = SchemaField(
            description="Why it was canceled: webset_deleted, webset_canceled or out_of_credits"
        )
        metadata: dict = SchemaField(description="Metadata attached to the search")

    def __init__(self):
        super().__init__(
            id="4fa3e627-a0ff-485f-8732-52148051646c",
            description="Get the status and details of a webset search",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetWebsetSearchBlock.Input,
            output_schema=ExaGetWebsetSearchBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        path = (
            f"/websets/{segment(input_data.webset_id)}"
            f"/searches/{segment(input_data.search_id)}"
        )
        search = ApiWebsetSearch.model_validate(
            await self._api(credentials, "GET", path)
        )
        progress = search.progress

        yield "search_id", search.id
        yield "status", search.status
        yield "query", search.query
        yield "entity_type", search.entity_type
        yield "criteria", search.criteria
        yield "count", int(search.count)
        yield "behavior", search.behavior
        yield "max_people_per_company", search.max_people_per_company
        yield "exclude", search.exclude
        yield "scope", search.scope
        yield (
            "progress",
            {
                "found": progress.found if progress else 0,
                "analyzed": progress.analyzed if progress else 0,
                "completion": progress.completion if progress else 0,
                "time_left": (progress.time_left if progress else None) or 0,
            },
        )
        yield "recall", recall_summary(search.recall)
        yield "created_at", search.created_at or ""
        yield "updated_at", search.updated_at or ""
        yield "canceled_at", search.canceled_at
        yield "canceled_reason", search.canceled_reason
        yield "metadata", search.metadata or {}


class ExaCancelWebsetSearchBlock(ExaWebsetsBlockBase):
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
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        path = (
            f"/websets/{segment(input_data.webset_id)}"
            f"/searches/{segment(input_data.search_id)}/cancel"
        )
        search = ApiWebsetSearch.model_validate(
            await self._api(credentials, "POST", path)
        )

        yield "search_id", search.id
        yield "status", search.status
        yield "items_found_before_cancel", search.found
        yield "success", "true"


class ExaFindOrCreateSearchBlock(ExaWebsetsBlockBase):
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
            description=f"Only used if creating. {BEHAVIOR_DESCRIPTION}",
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
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        webset = await self._get_webset(credentials, input_data.webset_id)
        wanted = input_data.query.strip().lower()
        existing = next(
            (s for s in webset.searches if s.query.strip().lower() == wanted), None
        )

        if existing:
            yield "search_id", existing.id
            yield "webset_id", input_data.webset_id
            yield "status", existing.status
            yield "query", existing.query
            yield "was_created", False
            yield "items_found", existing.found
            return

        payload: dict[str, Any] = {
            "query": input_data.query,
            "count": input_data.count,
            "behavior": search_behavior_value(input_data.behavior),
        }
        entity = entity_payload(input_data.entity_type.value, None)
        if entity:
            payload["entity"] = entity
        created = ApiWebsetSearch.model_validate(
            await self._api(
                credentials,
                "POST",
                f"/websets/{segment(input_data.webset_id)}/searches",
                json=payload,
            )
        )

        yield "search_id", created.id
        yield "webset_id", input_data.webset_id
        yield "status", created.status
        yield "query", created.query
        yield "was_created", True
        yield "items_found", 0  # Newly created, no items yet
