"""
Exa Websets Polling Blocks

This module provides dedicated polling blocks for waiting on webset operations
to complete, with progress tracking and timeout management.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Dict

from exa_py import AsyncExa
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

# Import WebsetItemModel for use in enrichment samples
# This is safe as websets_items doesn't import from websets_polling
from .websets_items import WebsetItemModel


# Model for sample enrichment data
class SampleEnrichmentModel(BaseModel):
    """Sample enrichment result for display."""

    item_id: str
    item_title: str
    enrichment_data: Dict[str, Any]


class WebsetTargetStatus(str, Enum):
    IDLE = "idle"
    COMPLETED = "completed"
    RUNNING = "running"
    PAUSED = "paused"
    ANY_COMPLETE = "any_complete"  # Either idle or completed


class ExaWaitForWebsetBlock(Block):
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
            description="Status to wait for (idle=all operations complete, completed=search done, running=actively processing)",
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
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        start_time = time.time()
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        try:
            if input_data.target_status in [
                WebsetTargetStatus.IDLE,
                WebsetTargetStatus.ANY_COMPLETE,
            ]:
                final_webset = aexa.websets.wait_until_idle(
                    id=input_data.webset_id,
                    timeout=input_data.timeout,
                    poll_interval=input_data.check_interval,
                )

                elapsed = time.time() - start_time

                status_str = (
                    final_webset.status.value
                    if hasattr(final_webset.status, "value")
                    else str(final_webset.status)
                )

                item_count = 0
                if final_webset.searches:
                    for search in final_webset.searches:
                        if search.progress:
                            item_count += search.progress.found

                # Extract progress if requested
                search_progress = {}
                enrichment_progress = {}
                if input_data.include_progress:
                    webset_dict = final_webset.model_dump(
                        by_alias=True, exclude_none=True
                    )
                    search_progress = self._extract_search_progress(webset_dict)
                    enrichment_progress = self._extract_enrichment_progress(webset_dict)

                yield "webset_id", input_data.webset_id
                yield "final_status", status_str
                yield "elapsed_time", elapsed
                yield "item_count", item_count
                if input_data.include_progress:
                    yield "search_progress", search_progress
                    yield "enrichment_progress", enrichment_progress
                yield "timed_out", False
            else:
                # For other status targets, manually poll
                interval = input_data.check_interval
                while time.time() - start_time < input_data.timeout:
                    # Get current webset status
                    webset = aexa.websets.get(id=input_data.webset_id)
                    current_status = (
                        webset.status.value
                        if hasattr(webset.status, "value")
                        else str(webset.status)
                    )

                    # Check if target status reached
                    if current_status == input_data.target_status.value:
                        elapsed = time.time() - start_time

                        # Estimate item count from search progress
                        item_count = 0
                        if webset.searches:
                            for search in webset.searches:
                                if search.progress:
                                    item_count += search.progress.found

                        search_progress = {}
                        enrichment_progress = {}
                        if input_data.include_progress:
                            webset_dict = webset.model_dump(
                                by_alias=True, exclude_none=True
                            )
                            search_progress = self._extract_search_progress(webset_dict)
                            enrichment_progress = self._extract_enrichment_progress(
                                webset_dict
                            )

                        yield "webset_id", input_data.webset_id
                        yield "final_status", current_status
                        yield "elapsed_time", elapsed
                        yield "item_count", item_count
                        if input_data.include_progress:
                            yield "search_progress", search_progress
                            yield "enrichment_progress", enrichment_progress
                        yield "timed_out", False
                        return

                    # Wait before next check with exponential backoff
                    await asyncio.sleep(interval)
                    interval = min(interval * 1.5, input_data.max_interval)

                # Timeout reached
                elapsed = time.time() - start_time
                webset = aexa.websets.get(id=input_data.webset_id)
                final_status = (
                    webset.status.value
                    if hasattr(webset.status, "value")
                    else str(webset.status)
                )

                item_count = 0
                if webset.searches:
                    for search in webset.searches:
                        if search.progress:
                            item_count += search.progress.found

                search_progress = {}
                enrichment_progress = {}
                if input_data.include_progress:
                    webset_dict = webset.model_dump(by_alias=True, exclude_none=True)
                    search_progress = self._extract_search_progress(webset_dict)
                    enrichment_progress = self._extract_enrichment_progress(webset_dict)

                yield "webset_id", input_data.webset_id
                yield "final_status", final_status
                yield "elapsed_time", elapsed
                yield "item_count", item_count
                if input_data.include_progress:
                    yield "search_progress", search_progress
                    yield "enrichment_progress", enrichment_progress
                yield "timed_out", True

        except asyncio.TimeoutError:
            raise ValueError(
                f"Polling timed out after {input_data.timeout} seconds"
            ) from None

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


class ExaWaitForSearchBlock(Block):
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
                # Get current search status using SDK
                search = aexa.websets.searches.get(
                    webset_id=input_data.webset_id, id=input_data.search_id
                )

                # Extract status
                status = (
                    search.status.value
                    if hasattr(search.status, "value")
                    else str(search.status)
                )

                # Check if search is complete
                if status in ["completed", "failed", "canceled"]:
                    elapsed = time.time() - start_time

                    # Extract progress information
                    progress_dict = {}
                    if search.progress:
                        progress_dict = search.progress.model_dump(
                            by_alias=True, exclude_none=True
                        )

                    # Extract recall information
                    recall_info = {}
                    if search.recall:
                        recall_dict = search.recall.model_dump(
                            by_alias=True, exclude_none=True
                        )
                        expected = recall_dict.get("expected", {})
                        recall_info = {
                            "expected_total": expected.get("total", 0),
                            "confidence": expected.get("confidence", ""),
                            "min_expected": expected.get("bounds", {}).get("min", 0),
                            "max_expected": expected.get("bounds", {}).get("max", 0),
                            "reasoning": recall_dict.get("reasoning", ""),
                        }

                    yield "search_id", input_data.search_id
                    yield "final_status", status
                    yield "items_found", progress_dict.get("found", 0)
                    yield "items_analyzed", progress_dict.get("analyzed", 0)
                    yield "completion_percentage", progress_dict.get("completion", 0)
                    yield "elapsed_time", elapsed
                    yield "recall_info", recall_info
                    yield "timed_out", False

                    return

                # Wait before next check with exponential backoff
                await asyncio.sleep(interval)
                interval = min(interval * 1.5, max_interval)

            # Timeout reached
            elapsed = time.time() - start_time

            # Get last known status
            search = aexa.websets.searches.get(
                webset_id=input_data.webset_id, id=input_data.search_id
            )
            final_status = (
                search.status.value
                if hasattr(search.status, "value")
                else str(search.status)
            )

            progress_dict = {}
            if search.progress:
                progress_dict = search.progress.model_dump(
                    by_alias=True, exclude_none=True
                )

            yield "search_id", input_data.search_id
            yield "final_status", final_status
            yield "items_found", progress_dict.get("found", 0)
            yield "items_analyzed", progress_dict.get("analyzed", 0)
            yield "completion_percentage", progress_dict.get("completion", 0)
            yield "elapsed_time", elapsed
            yield "timed_out", True

        except asyncio.TimeoutError:
            raise ValueError(
                f"Search polling timed out after {input_data.timeout} seconds"
            ) from None


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
                enrichment = aexa.websets.enrichments.get(
                    webset_id=input_data.webset_id, id=input_data.enrichment_id
                )

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
                        sample_data, items_enriched = (
                            await self._get_sample_enrichments(
                                input_data.webset_id, input_data.enrichment_id, aexa
                            )
                        )

                    yield "enrichment_id", input_data.enrichment_id
                    yield "final_status", status
                    yield "items_enriched", items_enriched
                    yield "enrichment_title", enrichment.title or enrichment.description or ""
                    yield "elapsed_time", elapsed
                    if input_data.sample_results:
                        yield "sample_data", sample_data
                    yield "timed_out", False

                    return

                # Wait before next check with exponential backoff
                await asyncio.sleep(interval)
                interval = min(interval * 1.5, max_interval)

            # Timeout reached
            elapsed = time.time() - start_time

            # Get last known status
            enrichment = aexa.websets.enrichments.get(
                webset_id=input_data.webset_id, id=input_data.enrichment_id
            )
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

        except asyncio.TimeoutError:
            raise ValueError(
                f"Enrichment polling timed out after {input_data.timeout} seconds"
            ) from None

    async def _get_sample_enrichments(
        self, webset_id: str, enrichment_id: str, aexa: AsyncExa
    ) -> tuple[list[SampleEnrichmentModel], int]:
        """Get sample enriched data and count."""
        # Get a few items to see enrichment results using SDK
        response = aexa.websets.items.list(webset_id=webset_id, limit=5)

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
