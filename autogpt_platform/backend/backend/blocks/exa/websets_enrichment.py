"""
Exa Websets Enrichment Management Blocks

This module provides blocks for creating and managing enrichments on webset items,
allowing extraction of additional structured data from existing items.
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


class EnrichmentFormat(str, Enum):
    """Format types for enrichment responses."""

    TEXT = "text"  # Free text response
    DATE = "date"  # Date/datetime format
    NUMBER = "number"  # Numeric value
    OPTIONS = "options"  # Multiple choice from provided options
    EMAIL = "email"  # Email address format
    PHONE = "phone"  # Phone number format


class ExaCreateEnrichmentBlock(Block):
    """Create a new enrichment to extract additional data from webset items."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        description: str = SchemaField(
            description="What data to extract from each item",
            placeholder="Extract the company's main product or service offering",
        )
        title: Optional[str] = SchemaField(
            default=None,
            description="Short title for this enrichment (auto-generated if not provided)",
            placeholder="Main Product",
        )
        format: EnrichmentFormat = SchemaField(
            default=EnrichmentFormat.TEXT,
            description="Expected format of the extracted data",
        )
        options: list[str] = SchemaField(
            default_factory=list,
            description="Available options when format is 'options'",
            placeholder=["B2B", "B2C", "Both", "Unknown"],
            advanced=True,
        )
        apply_to_existing: bool = SchemaField(
            default=True,
            description="Apply this enrichment to existing items in the webset",
        )
        metadata: Optional[dict] = SchemaField(
            default=None,
            description="Metadata to attach to the enrichment",
            advanced=True,
        )
        wait_for_completion: bool = SchemaField(
            default=False,
            description="Wait for the enrichment to complete on existing items",
        )
        polling_timeout: int = SchemaField(
            default=300,
            description="Maximum time to wait for completion in seconds",
            advanced=True,
            ge=1,
            le=600,
        )

    class Output(BlockSchema):
        enrichment_id: str = SchemaField(
            description="The unique identifier for the created enrichment"
        )
        webset_id: str = SchemaField(
            description="The webset this enrichment belongs to"
        )
        status: str = SchemaField(description="Current status of the enrichment")
        title: str = SchemaField(description="Title of the enrichment")
        description: str = SchemaField(
            description="Description of what data is extracted"
        )
        format: str = SchemaField(description="Format of the extracted data")
        instructions: str = SchemaField(
            description="Generated instructions for the enrichment"
        )
        items_enriched: Optional[int] = SchemaField(
            description="Number of items enriched (if wait_for_completion was True)",
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
            id="a7b8c9d0-e1f2-3456-789a-bcdef0123456",
            description="Create enrichments to extract additional structured data from webset items",
            categories={BlockCategory.AI, BlockCategory.SEARCH},
            input_schema=ExaCreateEnrichmentBlock.Input,
            output_schema=ExaCreateEnrichmentBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        import time

        url = (
            f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/enrichments"
        )
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        # Build the payload
        payload = {
            "description": input_data.description,
            "format": input_data.format.value,
        }

        # Add title if provided
        if input_data.title:
            payload["title"] = input_data.title

        # Add options for 'options' format
        if input_data.format == EnrichmentFormat.OPTIONS and input_data.options:
            payload["options"] = [{"label": opt} for opt in input_data.options]

        # Add metadata if provided
        if input_data.metadata:
            payload["metadata"] = input_data.metadata

        try:
            start_time = time.time()

            # Create the enrichment
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()

            enrichment_id = data.get("id", "")
            status = data.get("status", "")
            title = data.get("title", "")
            instructions = data.get("instructions", "")

            # If wait_for_completion is True and apply_to_existing is True, poll for completion
            if input_data.wait_for_completion and input_data.apply_to_existing:
                items_enriched = await self._poll_for_completion(
                    input_data.webset_id,
                    enrichment_id,
                    credentials.api_key.get_secret_value(),
                    input_data.polling_timeout,
                )
                completion_time = time.time() - start_time

                yield "enrichment_id", enrichment_id
                yield "webset_id", input_data.webset_id
                yield "status", "completed"
                yield "title", title
                yield "description", input_data.description
                yield "format", input_data.format.value
                yield "instructions", instructions
                yield "items_enriched", items_enriched
                yield "completion_time", completion_time
            else:
                yield "enrichment_id", enrichment_id
                yield "webset_id", input_data.webset_id
                yield "status", status
                yield "title", title
                yield "description", input_data.description
                yield "format", input_data.format.value
                yield "instructions", instructions

        except Exception as e:
            yield "error", str(e)
            yield "enrichment_id", ""
            yield "webset_id", input_data.webset_id
            yield "status", ""
            yield "title", ""
            yield "description", input_data.description
            yield "format", input_data.format.value
            yield "instructions", ""

    async def _poll_for_completion(
        self, webset_id: str, enrichment_id: str, api_key: str, timeout: int
    ) -> int:
        """Poll enrichment status until it completes or times out."""
        import asyncio
        import time

        start_time = time.time()
        interval = 5
        max_interval = 30

        url = f"https://api.exa.ai/websets/v0/websets/{webset_id}/enrichments/{enrichment_id}"
        headers = {"x-api-key": api_key}

        while time.time() - start_time < timeout:
            try:
                response = await Requests().get(url, headers=headers)
                data = response.json()

                status = data.get("status", "")

                if status in ["completed", "failed", "canceled"]:
                    # Try to count enriched items
                    items_url = (
                        f"https://api.exa.ai/websets/v0/websets/{webset_id}/items"
                    )
                    items_response = await Requests().get(
                        items_url, headers=headers, params={"limit": 100}
                    )
                    items_data = items_response.json()

                    enriched_count = 0
                    for item in items_data.get("data", []):
                        enrichments = item.get("enrichments", {})
                        if enrichment_id in enrichments:
                            enriched_count += 1

                    return enriched_count

                await asyncio.sleep(interval)
                interval = min(interval * 1.5, max_interval)

            except Exception:
                await asyncio.sleep(interval)

        return 0


class ExaGetEnrichmentBlock(Block):
    """Get the status and details of a webset enrichment."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        enrichment_id: str = SchemaField(
            description="The ID of the enrichment to retrieve",
            placeholder="enrichment-id",
        )

    class Output(BlockSchema):
        enrichment_id: str = SchemaField(
            description="The unique identifier for the enrichment"
        )
        status: str = SchemaField(description="Current status of the enrichment")
        title: str = SchemaField(description="Title of the enrichment")
        description: str = SchemaField(
            description="Description of what data is extracted"
        )
        format: str = SchemaField(description="Format of the extracted data")
        options: list[str] = SchemaField(
            description="Available options (for 'options' format)",
            default_factory=list,
        )
        instructions: str = SchemaField(
            description="Generated instructions for the enrichment"
        )
        created_at: str = SchemaField(description="When the enrichment was created")
        updated_at: str = SchemaField(
            description="When the enrichment was last updated"
        )
        metadata: dict = SchemaField(
            description="Metadata attached to the enrichment",
            default_factory=dict,
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="b8c9d0e1-f2a3-4567-89ab-cdef01234567",
            description="Get the status and details of a webset enrichment",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetEnrichmentBlock.Input,
            output_schema=ExaGetEnrichmentBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/enrichments/{input_data.enrichment_id}"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            response = await Requests().get(url, headers=headers)
            data = response.json()

            # Extract options if present
            options = []
            if data.get("options"):
                options = [opt.get("label", "") for opt in data["options"]]

            yield "enrichment_id", data.get("id", "")
            yield "status", data.get("status", "")
            yield "title", data.get("title", "")
            yield "description", data.get("description", "")
            yield "format", data.get("format", "")
            yield "options", options
            yield "instructions", data.get("instructions", "")
            yield "created_at", data.get("createdAt", "")
            yield "updated_at", data.get("updatedAt", "")
            yield "metadata", data.get("metadata", {})

        except Exception as e:
            yield "error", str(e)
            yield "enrichment_id", ""
            yield "status", ""
            yield "title", ""
            yield "description", ""
            yield "format", ""
            yield "options", []
            yield "instructions", ""
            yield "created_at", ""
            yield "updated_at", ""
            yield "metadata", {}


class ExaUpdateEnrichmentBlock(Block):
    """Update an existing enrichment configuration."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        enrichment_id: str = SchemaField(
            description="The ID of the enrichment to update",
            placeholder="enrichment-id",
        )
        description: Optional[str] = SchemaField(
            default=None,
            description="New description for what data to extract",
        )
        format: Optional[EnrichmentFormat] = SchemaField(
            default=None,
            description="New format for the extracted data",
        )
        options: Optional[list[str]] = SchemaField(
            default=None,
            description="New options when format is 'options'",
        )
        metadata: Optional[dict] = SchemaField(
            default=None,
            description="New metadata to attach to the enrichment",
        )

    class Output(BlockSchema):
        enrichment_id: str = SchemaField(
            description="The unique identifier for the enrichment"
        )
        status: str = SchemaField(description="Current status of the enrichment")
        title: str = SchemaField(description="Title of the enrichment")
        description: str = SchemaField(description="Updated description")
        format: str = SchemaField(description="Updated format")
        success: str = SchemaField(
            description="Whether the update was successful",
            default="true",
        )
        error: str = SchemaField(
            description="Error message if the update failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="c9d0e1f2-a3b4-5678-9abc-def012345678",
            description="Update an existing enrichment configuration",
            categories={BlockCategory.SEARCH},
            input_schema=ExaUpdateEnrichmentBlock.Input,
            output_schema=ExaUpdateEnrichmentBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/enrichments/{input_data.enrichment_id}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        # Build the update payload
        payload = {}

        if input_data.description is not None:
            payload["description"] = input_data.description

        if input_data.format is not None:
            payload["format"] = input_data.format.value

        if input_data.options is not None:
            payload["options"] = [{"label": opt} for opt in input_data.options]

        if input_data.metadata is not None:
            payload["metadata"] = input_data.metadata

        try:
            response = await Requests().patch(url, headers=headers, json=payload)
            data = response.json()

            yield "enrichment_id", data.get("id", "")
            yield "status", data.get("status", "")
            yield "title", data.get("title", "")
            yield "description", data.get("description", "")
            yield "format", data.get("format", "")
            yield "success", "true"

        except Exception as e:
            yield "error", str(e)
            yield "enrichment_id", input_data.enrichment_id
            yield "status", ""
            yield "title", ""
            yield "description", ""
            yield "format", ""
            yield "success", "false"


class ExaDeleteEnrichmentBlock(Block):
    """Delete an enrichment from a webset."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        enrichment_id: str = SchemaField(
            description="The ID of the enrichment to delete",
            placeholder="enrichment-id",
        )

    class Output(BlockSchema):
        enrichment_id: str = SchemaField(description="The ID of the deleted enrichment")
        success: str = SchemaField(
            description="Whether the deletion was successful",
            default="true",
        )
        error: str = SchemaField(
            description="Error message if the deletion failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="d0e1f2a3-b4c5-6789-0abc-def123456789",
            description="Delete an enrichment from a webset",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteEnrichmentBlock.Input,
            output_schema=ExaDeleteEnrichmentBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/enrichments/{input_data.enrichment_id}"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            response = await Requests().delete(url, headers=headers)

            # API typically returns 204 No Content on successful deletion
            if response.status_code in [200, 204]:
                yield "enrichment_id", input_data.enrichment_id
                yield "success", "true"
            else:
                data = response.json()
                yield "enrichment_id", input_data.enrichment_id
                yield "success", "false"
                yield "error", data.get("message", "Deletion failed")

        except Exception as e:
            yield "error", str(e)
            yield "enrichment_id", input_data.enrichment_id
            yield "success", "false"


class ExaCancelEnrichmentBlock(Block):
    """Cancel a running enrichment operation."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        enrichment_id: str = SchemaField(
            description="The ID of the enrichment to cancel",
            placeholder="enrichment-id",
        )

    class Output(BlockSchema):
        enrichment_id: str = SchemaField(
            description="The ID of the canceled enrichment"
        )
        status: str = SchemaField(description="Status after cancellation")
        items_enriched_before_cancel: int = SchemaField(
            description="Approximate number of items enriched before cancellation",
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
            id="e1f2a3b4-c5d6-7890-1abc-def234567890",
            description="Cancel a running enrichment operation",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCancelEnrichmentBlock.Input,
            output_schema=ExaCancelEnrichmentBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/enrichments/{input_data.enrichment_id}/cancel"
        headers = {
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        try:
            response = await Requests().post(url, headers=headers)
            data = response.json()

            # Try to estimate how many items were enriched before cancellation
            items_enriched = 0
            try:
                items_url = f"https://api.exa.ai/websets/v0/websets/{input_data.webset_id}/items"
                items_response = await Requests().get(
                    items_url, headers=headers, params={"limit": 100}
                )
                items_data = items_response.json()

                for item in items_data.get("data", []):
                    enrichments = item.get("enrichments", {})
                    if input_data.enrichment_id in enrichments:
                        items_enriched += 1

            except:
                pass

            yield "enrichment_id", input_data.enrichment_id
            yield "status", data.get("status", "canceled")
            yield "items_enriched_before_cancel", items_enriched
            yield "success", "true"

        except Exception as e:
            yield "error", str(e)
            yield "enrichment_id", input_data.enrichment_id
            yield "status", ""
            yield "items_enriched_before_cancel", 0
            yield "success", "false"
