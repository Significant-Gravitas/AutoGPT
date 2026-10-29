"""
Exa Websets Enrichment Management Blocks

This module provides blocks for creating and managing enrichments on webset items,
allowing extraction of additional structured data from existing items.
"""

import time
from asyncio import sleep
from enum import Enum
from typing import Any

from exa_py import AsyncExa
from exa_py.websets.types import WebsetEnrichment as SdkWebsetEnrichment
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
from .websets_payloads import options_payload

ENRICHMENT_DONE_STATUSES = ("completed", "canceled", "failed")


# Mirrored model for stability
class WebsetEnrichmentModel(BaseModel):
    """Stable output model mirroring SDK WebsetEnrichment."""

    id: str
    webset_id: str
    status: str
    title: str | None
    description: str
    format: str
    options: list[str]
    instructions: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    @classmethod
    def from_sdk(cls, enrichment: SdkWebsetEnrichment) -> "WebsetEnrichmentModel":
        """Convert SDK WebsetEnrichment to our stable model."""
        # Extract options
        options_list = []
        if enrichment.options:
            for option in enrichment.options:
                option_dict = option.model_dump(by_alias=True)
                options_list.append(option_dict.get("label", ""))

        return cls(
            id=enrichment.id,
            webset_id=enrichment.webset_id,
            status=(
                enrichment.status.value
                if hasattr(enrichment.status, "value")
                else str(enrichment.status)
            ),
            title=enrichment.title,
            description=enrichment.description,
            format=(
                enrichment.format.value
                if enrichment.format and hasattr(enrichment.format, "value")
                else "text"
            ),
            options=options_list,
            instructions=enrichment.instructions,
            metadata=enrichment.metadata if enrichment.metadata else {},
            created_at=(
                enrichment.created_at.isoformat() if enrichment.created_at else ""
            ),
            updated_at=(
                enrichment.updated_at.isoformat() if enrichment.updated_at else ""
            ),
        )


class EnrichmentFormat(str, Enum):
    """Format types for enrichment responses."""

    TEXT = "text"  # Free text response
    DATE = "date"  # Date/datetime format
    NUMBER = "number"  # Numeric value
    OPTIONS = "options"  # Multiple choice from provided options
    EMAIL = "email"  # Email address format
    PHONE = "phone"  # Phone number format
    URL = "url"  # A web address


class ExaCreateEnrichmentBlock(ExaWebsetsBlockBase):
    """Create a new enrichment to extract additional data from webset items."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        webset_id: str = SchemaField(
            description="The ID or external ID of the Webset",
            placeholder="webset-id-or-external-id",
        )
        description: str = SchemaField(
            description="What data to extract from each item (up to 5,000 characters)",
            placeholder="Extract the company's main product or service offering",
        )
        title: str | None = SchemaField(
            default=None,
            description="Not part of Exa's current API: Exa generates the title from the description and format. Sent as-is if set.",
            placeholder="Main Product",
        )
        format: EnrichmentFormat = SchemaField(
            default=EnrichmentFormat.TEXT,
            description="Expected format of the extracted data: text, date, number, options (pick from a list), email, phone or url",
        )
        options: list[str] = SchemaField(
            default_factory=list,
            description="Labels to choose from when format is 'options' (1-150)",
            placeholder='["B2B", "B2C", "Both", "Unknown"]',
            advanced=True,
        )
        apply_to_existing: bool = SchemaField(
            default=True,
            description="Exa always enriches existing and future items; this only controls whether 'wait for completion' waits for them",
        )
        metadata: dict | None = SchemaField(
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

    class Output(BlockSchemaOutput):
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
        items_enriched: int | None = SchemaField(
            description="Number of items enriched (if wait_for_completion was True)"
        )
        completion_time: float | None = SchemaField(
            description="Time taken to complete in seconds (if wait_for_completion was True)"
        )

    def __init__(self):
        super().__init__(
            id="71146ae8-0cb1-4a15-8cde-eae30de71cb6",
            description="Create enrichments to extract additional structured data from webset items",
            categories={BlockCategory.AI, BlockCategory.SEARCH},
            input_schema=ExaCreateEnrichmentBlock.Input,
            output_schema=ExaCreateEnrichmentBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Build the payload
        payload: dict[str, Any] = {
            "description": input_data.description,
            "format": input_data.format.value,
        }

        # Add title if provided
        if input_data.title:
            payload["title"] = input_data.title

        # Add options for 'options' format
        if input_data.format == EnrichmentFormat.OPTIONS and input_data.options:
            payload["options"] = options_payload(input_data.options)

        # Add metadata if provided
        if input_data.metadata:
            payload["metadata"] = input_data.metadata

        start_time = time.time()

        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        sdk_enrichment = await aexa.websets.enrichments.create(
            webset_id=input_data.webset_id, params=payload
        )
        merge_exa_cost(self, sdk_enrichment)

        enrichment_id = sdk_enrichment.id
        status = (
            sdk_enrichment.status.value
            if hasattr(sdk_enrichment.status, "value")
            else str(sdk_enrichment.status)
        )

        # If wait_for_completion is True and apply_to_existing is True, poll for completion
        if input_data.wait_for_completion and input_data.apply_to_existing:
            poll_interval = 5
            max_interval = 30
            poll_start = time.time()
            items_enriched = 0

            while time.time() - poll_start < input_data.polling_timeout:
                current_enrich = await aexa.websets.enrichments.get(
                    webset_id=input_data.webset_id, id=enrichment_id
                )
                merge_exa_cost(self, current_enrich)
                current_status = (
                    current_enrich.status.value
                    if hasattr(current_enrich.status, "value")
                    else str(current_enrich.status)
                )

                if current_status in ENRICHMENT_DONE_STATUSES:
                    # Estimate items from webset searches
                    webset = await self._get_webset(credentials, input_data.webset_id)
                    items_enriched = webset.items_found
                    completion_time = time.time() - start_time

                    yield "enrichment_id", enrichment_id
                    yield "webset_id", input_data.webset_id
                    yield "status", current_status
                    yield "title", sdk_enrichment.title
                    yield "description", input_data.description
                    yield "format", input_data.format.value
                    yield "instructions", sdk_enrichment.instructions
                    yield "items_enriched", items_enriched
                    yield "completion_time", completion_time
                    return

                await sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, max_interval)

            # Timeout
            completion_time = time.time() - start_time
            yield "enrichment_id", enrichment_id
            yield "webset_id", input_data.webset_id
            yield "status", status
            yield "title", sdk_enrichment.title
            yield "description", input_data.description
            yield "format", input_data.format.value
            yield "instructions", sdk_enrichment.instructions
            yield "items_enriched", 0
            yield "completion_time", completion_time
        else:
            yield "enrichment_id", enrichment_id
            yield "webset_id", input_data.webset_id
            yield "status", status
            yield "title", sdk_enrichment.title
            yield "description", input_data.description
            yield "format", input_data.format.value
            yield "instructions", sdk_enrichment.instructions


class ExaGetEnrichmentBlock(Block):
    """Get the status and details of a webset enrichment."""

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
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
            description="Available options (for 'options' format)"
        )
        instructions: str = SchemaField(
            description="Generated instructions for the enrichment"
        )
        created_at: str = SchemaField(description="When the enrichment was created")
        updated_at: str = SchemaField(
            description="When the enrichment was last updated"
        )
        metadata: dict = SchemaField(description="Metadata attached to the enrichment")

    def __init__(self):
        super().__init__(
            id="b8c9d0e1-f2a3-4567-89ab-cdef01234567",
            description="Get the status and details of a webset enrichment",
            categories={BlockCategory.SEARCH},
            input_schema=ExaGetEnrichmentBlock.Input,
            output_schema=ExaGetEnrichmentBlock.Output,
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        sdk_enrichment = await aexa.websets.enrichments.get(
            webset_id=input_data.webset_id, id=input_data.enrichment_id
        )
        merge_exa_cost(self, sdk_enrichment)

        enrichment = WebsetEnrichmentModel.from_sdk(sdk_enrichment)

        yield "enrichment_id", enrichment.id
        yield "status", enrichment.status
        yield "title", enrichment.title
        yield "description", enrichment.description
        yield "format", enrichment.format
        yield "options", enrichment.options
        yield "instructions", enrichment.instructions
        yield "created_at", enrichment.created_at
        yield "updated_at", enrichment.updated_at
        yield "metadata", enrichment.metadata


class ExaUpdateEnrichmentBlock(ExaWebsetsBlockBase):
    """Update an existing enrichment configuration."""

    class Input(BlockSchemaInput):
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
        description: str | None = SchemaField(
            default=None,
            description="New description for what data to extract (up to 5,000 characters)",
        )
        format: EnrichmentFormat | None = SchemaField(
            default=None,
            description="New format: text, date, number, options, email, phone or url",
        )
        options: list[str] | None = SchemaField(
            default=None,
            description="New labels to choose from when format is 'options' (1-150)",
        )
        metadata: dict | None = SchemaField(
            default=None,
            description="New metadata to attach to the enrichment",
        )

    class Output(BlockSchemaOutput):
        enrichment_id: str = SchemaField(
            description="The unique identifier for the enrichment"
        )
        status: str = SchemaField(description="Current status of the enrichment")
        title: str = SchemaField(description="Title of the enrichment")
        description: str = SchemaField(description="Updated description")
        format: str = SchemaField(description="Updated format")
        success: str = SchemaField(description="Whether the update was successful")

    def __init__(self):
        super().__init__(
            id="c8d5c5fb-9684-4a29-bd2a-5b38d71776c9",
            description="Update an existing enrichment configuration",
            categories={BlockCategory.SEARCH},
            input_schema=ExaUpdateEnrichmentBlock.Input,
            output_schema=ExaUpdateEnrichmentBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        path = (
            f"/websets/{segment(input_data.webset_id)}"
            f"/enrichments/{segment(input_data.enrichment_id)}"
        )

        # Build the update payload
        payload = {}

        if input_data.description is not None:
            payload["description"] = input_data.description

        if input_data.format is not None:
            payload["format"] = input_data.format.value

        if input_data.options is not None:
            payload["options"] = options_payload(input_data.options)

        if input_data.metadata is not None:
            payload["metadata"] = input_data.metadata

        data = await self._api(credentials, "PATCH", path, json=payload)

        yield "enrichment_id", data.get("id", "")
        yield "status", data.get("status", "")
        yield "title", data.get("title") or ""
        yield "description", data.get("description", "")
        yield "format", data.get("format") or ""
        yield "success", "true"


class ExaDeleteEnrichmentBlock(Block):
    """Delete an enrichment from a webset."""

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        enrichment_id: str = SchemaField(description="The ID of the deleted enrichment")
        success: str = SchemaField(description="Whether the deletion was successful")

    def __init__(self):
        super().__init__(
            id="b250de56-2ca6-4237-a7b8-b5684892189f",
            description="Delete an enrichment from a webset",
            categories={BlockCategory.SEARCH},
            input_schema=ExaDeleteEnrichmentBlock.Input,
            output_schema=ExaDeleteEnrichmentBlock.Output,
            is_irreversible_action=True,
            effect=BlockEffect.EXTERNAL,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        deleted_enrichment = await aexa.websets.enrichments.delete(
            webset_id=input_data.webset_id, id=input_data.enrichment_id
        )
        merge_exa_cost(self, deleted_enrichment)

        yield "enrichment_id", deleted_enrichment.id
        yield "success", "true"


class ExaCancelEnrichmentBlock(Block):
    """Cancel a running enrichment operation."""

    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        enrichment_id: str = SchemaField(
            description="The ID of the canceled enrichment"
        )
        status: str = SchemaField(description="Status after cancellation")
        items_enriched_before_cancel: int = SchemaField(
            description="Approximate number of items enriched before cancellation"
        )
        success: str = SchemaField(
            description="Whether the cancellation was successful"
        )

    def __init__(self):
        super().__init__(
            id="7e1f8f0f-b6ab-43b3-bd1d-0c534a649295",
            description="Cancel a running enrichment operation",
            categories={BlockCategory.SEARCH},
            input_schema=ExaCancelEnrichmentBlock.Input,
            output_schema=ExaCancelEnrichmentBlock.Output,
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Use AsyncExa SDK
        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        canceled_enrichment = await aexa.websets.enrichments.cancel(
            webset_id=input_data.webset_id, id=input_data.enrichment_id
        )
        merge_exa_cost(self, canceled_enrichment)

        # Try to estimate how many items were enriched before cancellation
        items_enriched = 0
        items_response = await aexa.websets.items.list(
            webset_id=input_data.webset_id, limit=100
        )
        merge_exa_cost(self, items_response)

        for sdk_item in items_response.data:
            # Check if this enrichment is present
            for enrich_result in sdk_item.enrichments:
                if enrich_result.enrichment_id == input_data.enrichment_id:
                    items_enriched += 1
                    break

        status = (
            canceled_enrichment.status.value
            if hasattr(canceled_enrichment.status, "value")
            else str(canceled_enrichment.status)
        )

        yield "enrichment_id", canceled_enrichment.id
        yield "status", status
        yield "items_enriched_before_cancel", items_enriched
        yield "success", "true"
