"""
Exa Websets Polling Blocks

This module provides dedicated polling blocks for waiting on webset operations
to complete, with progress tracking and timeout management.
"""

import asyncio
import time
from enum import Enum

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


class WebsetTargetStatus(str, Enum):
    IDLE = "idle"
    COMPLETED = "completed"
    RUNNING = "running"
    PAUSED = "paused"
    ANY_COMPLETE = "any_complete"  # Either idle or completed


class ExaWaitForWebsetBlock(Block):
    """Wait for a webset to reach a specific status with progress tracking."""

    class Input(BlockSchema):
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

    class Output(BlockSchema):
        webset_id: str = SchemaField(description="The webset ID that was monitored")
        final_status: str = SchemaField(description="The final status of the webset")
        elapsed_time: float = SchemaField(description="Total time elapsed in seconds")
        item_count: int = SchemaField(
            description="Number of items found",
            default=0,
        )
        search_progress: dict = SchemaField(
            description="Detailed search progress information",
            default_factory=dict,
        )
        enrichment_progress: dict = SchemaField(
            description="Detailed enrichment progress information",
            default_factory=dict,
        )
        timed_out: bool = SchemaField(
            description="Whether the operation timed out",
            default=False,
        )
        error: str = SchemaField(
            description="Error message if the operation failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description="Wait for a webset to reach a specific status with progress tracking",
            categories={BlockCategory.SEARCH},
            input_schema=ExaWaitForWebsetBlock.Input,
            output_schema=ExaWaitForWebsetBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        start_time = time.time()
        interval = input_data.check_interval
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            while time.time() - start_time < input_data.timeout:
                # Get current webset status
                url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}"
                response = await Requests().get(url, headers=headers)
                data = response.json()

                current_status = data.get("status", "")

                # Check if target status reached
                status_reached = False
                if input_data.target_status == WebsetTargetStatus.ANY_COMPLETE:
                    status_reached = current_status in ["idle", "completed"]
                else:
                    status_reached = current_status == input_data.target_status.value

                if status_reached:
                    # Get final counts and progress
                    search_progress = self._extract_search_progress(data)
                    enrichment_progress = self._extract_enrichment_progress(data)
                    item_count = await self._get_item_count(
                        input_data.webset_id, headers
                    )

                    elapsed = time.time() - start_time

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

            # Get last known status
            try:
                response = await Requests().get(
                    f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}",
                    headers=headers,
                )
                data = response.json()
                final_status = data.get("status", "unknown")
                search_progress = self._extract_search_progress(data)
                enrichment_progress = self._extract_enrichment_progress(data)
                item_count = await self._get_item_count(input_data.webset_id, headers)
            except:
                final_status = "unknown"
                search_progress = {}
                enrichment_progress = {}
                item_count = 0

            yield "webset_id", input_data.webset_id
            yield "final_status", final_status
            yield "elapsed_time", elapsed
            yield "item_count", item_count
            if input_data.include_progress:
                yield "search_progress", search_progress
                yield "enrichment_progress", enrichment_progress
            yield "timed_out", True

        except Exception as e:
            yield "error", str(e)
            yield "webset_id", input_data.webset_id
            yield "final_status", ""
            yield "elapsed_time", time.time() - start_time
            yield "item_count", 0
            yield "search_progress", {}
            yield "enrichment_progress", {}
            yield "timed_out", False

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

    async def _get_item_count(self, webset_id: str, headers: dict) -> int:
        """Get the current item count for the webset."""
        try:
            url = f"https://api.exa.ai/websets/v0/websets/{webset_id}/items"
            response = await Requests().get(url, headers=headers, params={"limit": 1})
            data = response.json()

            # Try to get total from pagination
            if "pagination" in data:
                return data["pagination"].get("total", 0)

            # Fall back to counting items
            return len(data.get("data", []))
        except:
            return 0


class ExaWaitForSearchBlock(Block):
    """Wait for a specific webset search to complete with progress tracking."""

    class Input(BlockSchema):
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

    class Output(BlockSchema):
        search_id: str = SchemaField(description="The search ID that was monitored")
        final_status: str = SchemaField(description="The final status of the search")
        items_found: int = SchemaField(
            description="Number of items found by the search",
            default=0,
        )
        items_analyzed: int = SchemaField(
            description="Number of items analyzed",
            default=0,
        )
        completion_percentage: int = SchemaField(
            description="Completion percentage (0-100)",
            default=0,
        )
        elapsed_time: float = SchemaField(description="Total time elapsed in seconds")
        recall_info: dict = SchemaField(
            description="Information about expected results and confidence",
            default_factory=dict,
        )
        timed_out: bool = SchemaField(
            description="Whether the operation timed out",
            default=False,
        )
        error: str = SchemaField(
            description="Error message if the operation failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f23456789012",
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
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            while time.time() - start_time < input_data.timeout:
                # Get current search status
                url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/searches/{input_data.search_id}"
                response = await Requests().get(url, headers=headers)
                data = response.json()

                status = data.get("status", "")
                progress = data.get("progress", {})

                # Check if search is complete
                if status in ["completed", "failed", "canceled"]:
                    elapsed = time.time() - start_time

                    # Extract recall information
                    recall_info = {}
                    if "recall" in data:
                        recall = data["recall"]
                        expected = recall.get("expected", {})
                        recall_info = {
                            "expected_total": expected.get("total", 0),
                            "confidence": expected.get("confidence", ""),
                            "min_expected": expected.get("bounds", {}).get("min", 0),
                            "max_expected": expected.get("bounds", {}).get("max", 0),
                            "reasoning": recall.get("reasoning", ""),
                        }

                    yield "search_id", input_data.search_id
                    yield "final_status", status
                    yield "items_found", progress.get("found", 0)
                    yield "items_analyzed", progress.get("analyzed", 0)
                    yield "completion_percentage", progress.get("completion", 0)
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
            try:
                response = await Requests().get(url, headers=headers)
                data = response.json()
                final_status = data.get("status", "unknown")
                progress = data.get("progress", {})
            except:
                final_status = "unknown"
                progress = {}

            yield "search_id", input_data.search_id
            yield "final_status", final_status
            yield "items_found", progress.get("found", 0)
            yield "items_analyzed", progress.get("analyzed", 0)
            yield "completion_percentage", progress.get("completion", 0)
            yield "elapsed_time", elapsed
            yield "recall_info", {}
            yield "timed_out", True

        except Exception as e:
            yield "error", str(e)
            yield "search_id", input_data.search_id
            yield "final_status", ""
            yield "items_found", 0
            yield "items_analyzed", 0
            yield "completion_percentage", 0
            yield "elapsed_time", time.time() - start_time
            yield "recall_info", {}
            yield "timed_out", False


class ExaWaitForEnrichmentBlock(Block):
    """Wait for a webset enrichment to complete with progress tracking."""

    class Input(BlockSchema):
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

    class Output(BlockSchema):
        enrichment_id: str = SchemaField(
            description="The enrichment ID that was monitored"
        )
        final_status: str = SchemaField(
            description="The final status of the enrichment"
        )
        items_enriched: int = SchemaField(
            description="Number of items successfully enriched",
            default=0,
        )
        enrichment_title: str = SchemaField(
            description="Title/description of the enrichment"
        )
        elapsed_time: float = SchemaField(description="Total time elapsed in seconds")
        sample_data: list[dict] = SchemaField(
            description="Sample of enriched data (if requested)",
            default_factory=list,
        )
        timed_out: bool = SchemaField(
            description="Whether the operation timed out",
            default=False,
        )
        error: str = SchemaField(
            description="Error message if the operation failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="c3d4e5f6-a7b8-9012-cdef-345678901234",
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
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            while time.time() - start_time < input_data.timeout:
                # Get current enrichment status
                url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/enrichments/{input_data.enrichment_id}"
                response = await Requests().get(url, headers=headers)
                data = response.json()

                status = data.get("status", "")

                # Check if enrichment is complete
                if status in ["completed", "failed", "canceled"]:
                    elapsed = time.time() - start_time

                    # Get sample enriched items if requested
                    sample_data = []
                    items_enriched = 0

                    if input_data.sample_results and status == "completed":
                        sample_data, items_enriched = (
                            await self._get_sample_enrichments(
                                input_data.webset_id, input_data.enrichment_id, headers
                            )
                        )

                    yield "enrichment_id", input_data.enrichment_id
                    yield "final_status", status
                    yield "items_enriched", items_enriched
                    yield "enrichment_title", data.get(
                        "title", data.get("description", "")
                    )
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
            try:
                response = await Requests().get(url, headers=headers)
                data = response.json()
                final_status = data.get("status", "unknown")
                title = data.get("title", data.get("description", ""))
            except:
                final_status = "unknown"
                title = ""

            yield "enrichment_id", input_data.enrichment_id
            yield "final_status", final_status
            yield "items_enriched", 0
            yield "enrichment_title", title
            yield "elapsed_time", elapsed
            yield "sample_data", []
            yield "timed_out", True

        except Exception as e:
            yield "error", str(e)
            yield "enrichment_id", input_data.enrichment_id
            yield "final_status", ""
            yield "items_enriched", 0
            yield "enrichment_title", ""
            yield "elapsed_time", time.time() - start_time
            yield "sample_data", []
            yield "timed_out", False

    async def _get_sample_enrichments(
        self, webset_id: str, enrichment_id: str, headers: dict
    ) -> tuple[list[dict], int]:
        """Get sample enriched data and count."""
        try:
            # Get a few items to see enrichment results
            url = f"https://api.exa.ai/websets/v0/websets/{webset_id}/items"
            response = await Requests().get(url, headers=headers, params={"limit": 5})
            data = response.json()

            items = data.get("data", [])
            sample_data = []
            enriched_count = 0

            for item in items:
                enrichments = item.get("enrichments", {})
                if enrichment_id in enrichments:
                    enriched_count += 1
                    sample_data.append(
                        {
                            "item_id": item.get("id"),
                            "item_title": item.get("title", ""),
                            "enrichment_data": enrichments[enrichment_id],
                        }
                    )

            # Get total count if available
            if "pagination" in data:
                # This is an estimate - would need to check all items for accurate count
                total_items = data["pagination"].get("total", 0)
                if enriched_count > 0 and len(items) > 0:
                    # Estimate based on sample
                    enriched_count = int(total_items * (enriched_count / len(items)))

            return sample_data, enriched_count

        except Exception:
            return [], 0
