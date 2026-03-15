"""
Exa Webhook Blocks

These blocks handle webhook events from Exa's API for websets and other events.
"""

from backend.sdk import (
    BaseModel,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockType,
    BlockWebhookConfig,
    CredentialsMetaInput,
    Field,
    ProviderName,
    SchemaField,
)

from ._config import exa
from ._webhook import ExaEventType


class WebsetEventFilter(BaseModel):
    """Filter configuration for Exa webset events."""

    webset_created: bool = Field(
        default=True, description="Receive notifications when websets are created"
    )
    webset_deleted: bool = Field(
        default=False, description="Receive notifications when websets are deleted"
    )
    webset_paused: bool = Field(
        default=False, description="Receive notifications when websets are paused"
    )
    webset_idle: bool = Field(
        default=False, description="Receive notifications when websets become idle"
    )
    search_created: bool = Field(
        default=True,
        description="Receive notifications when webset searches are created",
    )
    search_completed: bool = Field(
        default=True, description="Receive notifications when webset searches complete"
    )
    search_canceled: bool = Field(
        default=False,
        description="Receive notifications when webset searches are canceled",
    )
    search_updated: bool = Field(
        default=False,
        description="Receive notifications when webset searches are updated",
    )
    item_created: bool = Field(
        default=True, description="Receive notifications when webset items are created"
    )
    item_enriched: bool = Field(
        default=True, description="Receive notifications when webset items are enriched"
    )
    export_created: bool = Field(
        default=False,
        description="Receive notifications when webset exports are created",
    )
    export_completed: bool = Field(
        default=True, description="Receive notifications when webset exports complete"
    )
    import_created: bool = Field(
        default=False, description="Receive notifications when imports are created"
    )
    import_completed: bool = Field(
        default=True, description="Receive notifications when imports complete"
    )
    import_processing: bool = Field(
        default=False, description="Receive notifications when imports are processing"
    )


class ExaWebsetWebhookBlock(Block):
    """
    Receives webhook notifications for Exa webset events.

    This block allows you to monitor various events related to Exa websets,
    including creation, updates, searches, and exports.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="Exa API credentials for webhook management"
        )
        webhook_url: str = SchemaField(
            description="URL to receive webhooks (auto-generated)",
            default="",
            hidden=True,
        )
        webset_id: str = SchemaField(
            description="The webset ID to monitor (optional, monitors all if empty)",
            default="",
        )
        event_filter: WebsetEventFilter = SchemaField(
            description="Configure which events to receive", default=WebsetEventFilter()
        )
        payload: dict = SchemaField(
            description="Webhook payload data", default={}, hidden=True
        )

    class Output(BlockSchemaOutput):
        event_type: str = SchemaField(description="Type of event that occurred")
        event_id: str = SchemaField(description="Unique identifier for this event")
        webset_id: str = SchemaField(description="ID of the affected webset")
        data: dict = SchemaField(description="Event-specific data")
        timestamp: str = SchemaField(description="When the event occurred")
        metadata: dict = SchemaField(description="Additional event metadata")

    def __init__(self):
        super().__init__(
            disabled=True,
            id="d0204ed8-8b81-408d-8b8d-ed087a546228",
            description="Receive webhook notifications for Exa webset events",
            categories={BlockCategory.INPUT},
            input_schema=ExaWebsetWebhookBlock.Input,
            output_schema=ExaWebsetWebhookBlock.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("exa"),
                webhook_type="webset",
                event_filter_input="event_filter",
                resource_format="{webset_id}",
            ),
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """Process incoming Exa webhook payload."""
        payload = input_data.payload

        # Extract event details
        event_type = payload.get("eventType", "unknown")
        event_id = payload.get("eventId", "")

        # Get webset ID from payload or input
        webset_id = payload.get("websetId", input_data.webset_id)

        # Check if we should process this event based on filter
        should_process = self._should_process_event(event_type, input_data.event_filter)

        if not should_process:
            # Skip events that don't match our filter
            return

        # Extract event data
        event_data = payload.get("data", {})
        timestamp = payload.get("occurredAt", payload.get("createdAt", ""))
        metadata = payload.get("metadata", {})

        yield "event_type", event_type
        yield "event_id", event_id
        yield "webset_id", webset_id
        yield "data", event_data
        yield "timestamp", timestamp
        yield "metadata", metadata

    def _should_process_event(
        self, event_type: str, event_filter: WebsetEventFilter
    ) -> bool:
        """Check if an event should be processed based on the filter."""
        filter_mapping = {
            ExaEventType.WEBSET_CREATED: event_filter.webset_created,
            ExaEventType.WEBSET_DELETED: event_filter.webset_deleted,
            ExaEventType.WEBSET_PAUSED: event_filter.webset_paused,
            ExaEventType.WEBSET_IDLE: event_filter.webset_idle,
            ExaEventType.WEBSET_SEARCH_CREATED: event_filter.search_created,
            ExaEventType.WEBSET_SEARCH_COMPLETED: event_filter.search_completed,
            ExaEventType.WEBSET_SEARCH_CANCELED: event_filter.search_canceled,
            ExaEventType.WEBSET_SEARCH_UPDATED: event_filter.search_updated,
            ExaEventType.WEBSET_ITEM_CREATED: event_filter.item_created,
            ExaEventType.WEBSET_ITEM_ENRICHED: event_filter.item_enriched,
            ExaEventType.WEBSET_EXPORT_CREATED: event_filter.export_created,
            ExaEventType.WEBSET_EXPORT_COMPLETED: event_filter.export_completed,
            ExaEventType.IMPORT_CREATED: event_filter.import_created,
            ExaEventType.IMPORT_COMPLETED: event_filter.import_completed,
            ExaEventType.IMPORT_PROCESSING: event_filter.import_processing,
        }

        # Try to convert string to ExaEventType enum
        try:
            event_type_enum = ExaEventType(event_type)
            return filter_mapping.get(event_type_enum, True)
        except ValueError:
            # If event_type is not a valid enum value, process it by default
            return True
