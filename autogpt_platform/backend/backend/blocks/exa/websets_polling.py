"""
Exa Websets Polling Blocks

This module provides dedicated polling blocks for waiting on webset operations
to complete, with progress tracking and timeout management.
"""

import time
from asyncio import sleep
from enum import Enum
from typing import Any

from exa_py import AsyncExa
from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from .helpers import merge_exa_cost
from .websets_api import ExaWebsetsBlockBase, segment

# Import WebsetItemModel for use in enrichment samples
# This is safe as websets_items doesn't import from websets_polling
from .websets_items import WebsetItemModel
from .websets_types import ApiSearchProgress, ApiWebset, ApiWebsetSearch, recall_summary


# Model for sample enrichment data
class SampleEnrichmentModel(BaseModel):
    """Sample enrichment result for display."""

    item_id: str
    item_title: str
    enrichment_data: dict[str, Any]


class WebsetTargetStatus(str, Enum):
    IDLE = "idle"
    COMPLETED = "completed"  # Legacy: webset status has no 'completed'; waits for idle
    RUNNING = "running"
    PAUSED = "paused"
    ANY_COMPLETE = "any_complete"  # Either idle or completed


# Targets that mean "all work finished", which for a webset is status 'idle'.
IDLE_TARGETS = (
    WebsetTargetStatus.IDLE,
    WebsetTargetStatus.COMPLETED,
    WebsetTargetStatus.ANY_COMPLETE,
)


class ExaWaitForWebsetBlock(ExaWebsetsBlockBase):
    """Wait for a webset to reach a specific status with progress tracking."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset to monitor",
            placeholder="webset-id-or-external-id",
        )
        target_status: WebsetTargetStatus = SchemaField(
            default=WebsetTargetStatus.IDLE,
            description="Status to wait for: idle (all operations complete), running (actively processing) or paused. 'completed' and 'any_complete' are legacy and wait for idle.",
        )
        timeout: int = SchemaField(
            default=300,
            description="Maximum time to wait in seconds",
            ge=1,
            le=1800,  # 30 minutes max
        )
        check_interval: int = SchemaField(
            default=5,
            description="Initial interval between status checks in seconds",
            advanced=True,
            ge=1,
            le=60,
        )
        max_interval: int = SchemaField(
            default=30,
            description="Maximum interval between checks (for exponential backoff)",
            advanced=True,
            ge=5,
            le=120,
        )
        include_progress: bool = SchemaField(
            default=True,
            description="Include detailed progress information in output",
        )

    class Output(BlockSchemaOutput):
        webset_id: str = SchemaField(description="The webset ID that was monitored")
        final_status: str = SchemaField(description="The final status of the webset")
        elapsed_time: float = SchemaField(description="Total time elapsed in seconds")
        item_count: int = SchemaField(description="Number of items found")
        search_progress: dict = SchemaField(
            description="Detailed search progress information"
        )
        enrichment_progress: dict = SchemaField(
            description="Detailed enrichment progress information"
        )
        timed_out: bool = SchemaField(description="Whether the operation timed out")

    def __init__(self):
        super().__init__(
            id="619d71e8-b72a-434d-8bd4-23376dd0342c",
            description="Wait for a webset to reach a specific status with progress tracking",
            categories={BlockCategory.SEARCH},
            input_schema=ExaWaitForWebsetBlock.Input,
            output_schema=ExaWaitForWebsetBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        start_time = time.time()
        try:
            if input_data.target_status in IDLE_TARGETS:
                webset = await self._wait_until_idle(
                    credentials,
                    input_data.webset_id,
                    timeout=input_data.timeout,
                    poll_interval=input_data.check_interval,
                )
                timed_out = False
            else:
                webset, timed_out = await self._poll_for_status(credentials, input_data)
        except TimeoutError:
            raise ValueError(
                f"Polling timed out after {input_data.timeout} seconds"
            ) from None

        yield "webset_id", input_data.webset_id
        yield "final_status", webset.status
        yield "elapsed_time", time.time() - start_time
        yield "item_count", webset.items_found
        if input_data.include_progress:
            webset_dict = webset.model_dump(by_alias=True)
            yield "search_progress", self._extract_search_progress(webset_dict)
            yield "enrichment_progress", self._extract_enrichment_progress(webset_dict)
        yield "timed_out", timed_out

    async def _poll_for_status(
        self, credentials: APIKeyCredentials, input_data: Input
    ) -> tuple[ApiWebset, bool]:
        """Poll with backoff until the target status; return (webset, timed_out)."""
        start_time = time.time()
        interval: float = input_data.check_interval
        while time.time() - start_time < input_data.timeout:
            webset = await self._get_webset(credentials, input_data.webset_id)
            if webset.status == input_data.target_status.value:
                return webset, False
            await sleep(interval)
            interval = min(interval * 1.5, input_data.max_interval)
        return await self._get_webset(credentials, input_data.webset_id), True

    def _extract_search_progress(self, webset_data: dict) -> dict:
        """Extract search progress information from webset data."""
        progress = {}
        searches = webset_data.get("searches", [])

        for idx, search in enumerate(searches):
            search_id = search.get("id", f"search_{idx}")
            search_progress = search.get("progress", {})

            progress[search_id] = {
                "status": search.get("status", "unknown"),
                "found": search_progress.get("found", 0),
                "analyzed": search_progress.get("analyzed", 0),
                "completion": search_progress.get("completion", 0),
                "time_left": search_progress.get("timeLeft", 0),
            }

        return progress

    def _extract_enrichment_progress(self, webset_data: dict) -> dict:
        """Extract enrichment progress information from webset data."""
        progress = {}
        enrichments = webset_data.get("enrichments", [])

        for idx, enrichment in enumerate(enrichments):
            enrich_id = enrichment.get("id", f"enrichment_{idx}")

            progress[enrich_id] = {
                "status": enrichment.get("status", "unknown"),
                "title": enrichment.get("title", ""),
                "description": enrichment.get("description", ""),
            }

        return progress


class ExaWaitForSearchBlock(ExaWebsetsBlockBase):
    """Wait for a specific webset search to complete with progress tracking."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        search_id: str = SchemaField(
            description="The ID of the search to monitor",
            placeholder="search-id",
        )
        timeout: int = SchemaField(
            default=300,
            description="Maximum time to wait in seconds",
            ge=1,
            le=1800,
        )
        check_interval: int = SchemaField(
            default=5,
            description="Initial interval between status checks in seconds",
            advanced=True,
            ge=1,
            le=60,
        )

    class Output(BlockSchemaOutput):
        search_id: str = SchemaField(description="The search ID that was monitored")
        final_status: str = SchemaField(description="The final status of the search")
        items_found: int = SchemaField(
            description="Number of items found by the search"
        )
        items_analyzed: int = SchemaField(description="Number of items analyzed")
        completion_percentage: int = SchemaField(
            description="Completion percentage (0-100)"
        )
        elapsed_time: float = SchemaField(description="Total time elapsed in seconds")
        recall_info: dict = SchemaField(
            description="Information about expected results and confidence"
        )
        timed_out: bool = SchemaField(description="Whether the operation timed out")

    def __init__(self):
        super().__init__(
            id="14da21ae-40a1-41bc-a111-c8e5c9ef012b",
            description="Wait for a specific webset search to complete with progress tracking",
            categories={BlockCategory.SEARCH},
            input_schema=ExaWaitForSearchBlock.Input,
            output_schema=ExaWaitForSearchBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        start_time = time.time()
        interval: float = input_data.check_interval
        max_interval = 30
        path = (
            f"/websets/{segment(input_data.webset_id)}"
            f"/searches/{segment(input_data.search_id)}"
        )

        while time.time() - start_time < input_data.timeout:
            search = ApiWebsetSearch.model_validate(
                await self._api(credentials, "GET", path)
            )
            if search.status in ["completed", "failed", "canceled"]:
                progress = search.progress or ApiSearchProgress()
                yield "search_id", input_data.search_id
                yield "final_status", search.status
                yield "items_found", int(progress.found)
                yield "items_analyzed", int(progress.analyzed)
                yield "completion_percentage", int(progress.completion)
                yield "elapsed_time", time.time() - start_time
                yield "recall_info", recall_summary(search.recall)
                yield "timed_out", False
                return

            # Wait before next check with exponential backoff
            await sleep(interval)
            interval = min(interval * 1.5, max_interval)

        # Timeout reached: report the last known state
        search = ApiWebsetSearch.model_validate(
            await self._api(credentials, "GET", path)
        )
        progress = search.progress or ApiSearchProgress()
        yield "search_id", input_data.search_id
        yield "final_status", search.status
        yield "items_found", int(progress.found)
        yield "items_analyzed", int(progress.analyzed)
        yield "completion_percentage", int(progress.completion)
        yield "elapsed_time", time.time() - start_time
        yield "timed_out", True


class ExaWaitForEnrichmentBlock(Block):
    """Wait for a webset enrichment to complete with progress tracking."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        enrichment_id: str = SchemaField(
            description="The ID of the enrichment to monitor",
            placeholder="enrichment-id",
        )
        timeout: int = SchemaField(
            default=300,
            description="Maximum time to wait in seconds",
            ge=1,
            le=1800,
        )
        check_interval: int = SchemaField(
            default=5,
            description="Initial interval between status checks in seconds",
            advanced=True,
            ge=1,
            le=60,
        )
        sample_results: bool = SchemaField(
            default=True,
            description="Include sample enrichment results in output",
        )

    class Output(BlockSchemaOutput):
        enrichment_id: str = SchemaField(
            description="The enrichment ID that was monitored"
        )
        final_status: str = SchemaField(
            description="The final status of the enrichment"
        )
        items_enriched: int = SchemaField(
            description="Number of items successfully enriched"
        )
        enrichment_title: str = SchemaField(
            description="Title/description of the enrichment"
        )
        elapsed_time: float = SchemaField(description="Total time elapsed in seconds")
        sample_data: list[SampleEnrichmentModel] = SchemaField(
            description="Sample of enriched data (if requested)"
        )
        timed_out: bool = SchemaField(description="Whether the operation timed out")

    def __init__(self):
        super().__init__(
            id="a11865c3-ac80-4721-8a40-ac4e3b71a558",
            description="Wait for a webset enrichment to complete with progress tracking",
            categories={BlockCategory.SEARCH},
            input_schema=ExaWaitForEnrichmentBlock.Input,
            output_schema=ExaWaitForEnrichmentBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        start_time = time.time()
        interval = input_data.check_interval
        max_interval = 30
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        try:
            while time.time() - start_time < input_data.timeout:
                # Get current enrichment status using SDK
                enrichment = await aexa.websets.enrichments.get(
                    webset_id=input_data.webset_id, id=input_data.enrichment_id
                )
                merge_exa_cost(self, enrichment)

                # Extract status
                status = (
                    enrichment.status.value
                    if hasattr(enrichment.status, "value")
                    else str(enrichment.status)
                )

                # Check if enrichment is complete
                if status in ["completed", "failed", "canceled"]:
                    elapsed = time.time() - start_time

                    # Get sample enriched items if requested
                    sample_data = []
                    items_enriched = 0

                    if input_data.sample_results and status == "completed":
                        (
                            sample_data,
                            items_enriched,
                        ) = await self._get_sample_enrichments(
                            input_data.webset_id, input_data.enrichment_id, aexa
                        )

                    yield "enrichment_id", input_data.enrichment_id
                    yield "final_status", status
                    yield "items_enriched", items_enriched
                    yield (
                        "enrichment_title",
                        enrichment.title or enrichment.description or "",
                    )
                    yield "elapsed_time", elapsed
                    if input_data.sample_results:
                        yield "sample_data", sample_data
                    yield "timed_out", False

                    return

                # Wait before next check with exponential backoff
                await sleep(interval)
                interval = min(interval * 1.5, max_interval)

            # Timeout reached
            elapsed = time.time() - start_time

            # Get last known status
            enrichment = await aexa.websets.enrichments.get(
                webset_id=input_data.webset_id, id=input_data.enrichment_id
            )
            merge_exa_cost(self, enrichment)
            final_status = (
                enrichment.status.value
                if hasattr(enrichment.status, "value")
                else str(enrichment.status)
            )
            title = enrichment.title or enrichment.description or ""

            yield "enrichment_id", input_data.enrichment_id
            yield "final_status", final_status
            yield "items_enriched", 0
            yield "enrichment_title", title
            yield "elapsed_time", elapsed
            yield "timed_out", True

        except TimeoutError:
            raise ValueError(
                f"Enrichment polling timed out after {input_data.timeout} seconds"
            ) from None

    async def _get_sample_enrichments(
        self, webset_id: str, enrichment_id: str, aexa: AsyncExa
    ) -> tuple[list[SampleEnrichmentModel], int]:
        """Get sample enriched data and count."""
        # Get a few items to see enrichment results using SDK
        response = await aexa.websets.items.list(webset_id=webset_id, limit=5)
        merge_exa_cost(self, response)

        sample_data: list[SampleEnrichmentModel] = []
        enriched_count = 0

        for sdk_item in response.data:
            # Convert to our WebsetItemModel first
            item = WebsetItemModel.from_sdk(sdk_item)

            # Check if this item has the enrichment we're looking for
            if enrichment_id in item.enrichments:
                enriched_count += 1
                enrich_model = item.enrichments[enrichment_id]

                # Create sample using our typed model
                sample = SampleEnrichmentModel(
                    item_id=item.id,
                    item_title=item.title,
                    enrichment_data=enrich_model.model_dump(exclude_none=True),
                )
                sample_data.append(sample)

        return sample_data, enriched_count
