"""
Exa Webhook Blocks

These blocks handle webhook events from Exa's API for websets and other events.
"""

from typing import Any

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
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._webhook import ExaWebhookManager, filter_field_for


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
        default=False,
        description="Receive notifications when websets become idle "
        "(no searches or enrichments still running)",
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
        description="Receive notifications when webset search progress is updated",
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
        default=False,
        description="Legacy: Exa no longer sends an import-processing event, so "
        "this never fires. Use 'import completed' instead.",
    )
    monitor_created: bool = Field(
        default=False,
        description="Receive notifications when a Websets monitor is created",
    )
    monitor_updated: bool = Field(
        default=False,
        description="Receive notifications when a Websets monitor's settings change",
    )
    monitor_deleted: bool = Field(
        default=False,
        description="Receive notifications when a Websets monitor is deleted",
    )
    monitor_run_created: bool = Field(
        default=False,
        description="Receive notifications when a Websets monitor run starts",
    )
    monitor_run_completed: bool = Field(
        default=False,
        description="Receive notifications when a Websets monitor run finishes",
    )


EXAMPLE_ITEM_CREATED_EVENT = {
    "id": "event_01jz2h3k4m5n6p7q8r9s0t1u2v",
    "object": "event",
    "type": "webset.item.created",
    "data": {
        "id": "witem_01jz2h3k4m5n6p7q8r9s0t1u2w",
        "object": "webset_item",
        "websetId": "webset_01jz2h3k4m5n6p7q8r9s0t1u2x",
        "properties": {"url": "https://example.com"},
        "metadata": {"team": "research"},
    },
    "createdAt": "2026-10-28T12:00:00.000Z",
}


class ExaWebsetWebhookBlock(Block):
    """
    Receives webhook notifications for Exa webset events.

    The platform registers a webhook with Exa's Websets API for the selected
    events and verifies each delivery's `Exa-Signature` header. Exa webhooks are
    team-wide, so when a webset ID is set, events about other websets are
    skipped here; events that name no webset (such as imports and monitor runs)
    always pass through.
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
            description="Only fire for events about this webset ID. Events that "
            "name no webset (imports, monitor runs) still fire. Leave empty to "
            "receive events for every webset on your Exa team.",
            default="",
            placeholder="webset_01jz2h3k4m5n6p7q8r9s0t1u2x",
            advanced=False,
        )
        event_filter: WebsetEventFilter = SchemaField(
            description="Configure which events to receive",
            default=WebsetEventFilter(),
            advanced=False,
        )
        payload: dict = SchemaField(
            description="Webhook payload data", default={}, hidden=True
        )

    class Output(BlockSchemaOutput):
        event_type: str = SchemaField(
            description="Type of event that occurred, e.g. webset.item.created"
        )
        event_id: str = SchemaField(description="Unique identifier for this event")
        webset_id: str = SchemaField(
            description="ID of the affected webset, or the Webset ID filter when "
            "the event names none"
        )
        data: dict = SchemaField(
            description="The full resource the event is about (webset, item, "
            "search, import or monitor)"
        )
        timestamp: str = SchemaField(description="When the event occurred (ISO 8601)")
        metadata: dict = SchemaField(
            description="Metadata attached to the resource the event is about"
        )

    def __init__(self):
        super().__init__(
            id="d0204ed8-8b81-408d-8b8d-ed087a546228",
            description="Triggers when Exa reports a Websets event, such as a "
            "new item, a finished search, a completed export or a monitor run",
            categories={BlockCategory.INPUT, BlockCategory.SEARCH},
            input_schema=ExaWebsetWebhookBlock.Input,
            output_schema=ExaWebsetWebhookBlock.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("exa"),
                webhook_type=ExaWebhookManager.WebhookType.WEBSET.value,
                event_filter_input="event_filter",
                resource_format="{webset_id}",
            ),
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "webset_id": "",
                "event_filter": WebsetEventFilter().model_dump(),
                "payload": EXAMPLE_ITEM_CREATED_EVENT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("event_type", "webset.item.created"),
                ("event_id", "event_01jz2h3k4m5n6p7q8r9s0t1u2v"),
                ("webset_id", "webset_01jz2h3k4m5n6p7q8r9s0t1u2x"),
                ("data", EXAMPLE_ITEM_CREATED_EVENT["data"]),
                ("timestamp", "2026-10-28T12:00:00.000Z"),
                ("metadata", {"team": "research"}),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """Process incoming Exa webhook payload."""
        payload = input_data.payload
        event_type = str(payload.get("type") or payload.get("eventType") or "unknown")

        if not self._should_process_event(event_type, input_data.event_filter):
            return

        event_data: dict[str, Any] = payload.get("data") or {}
        webset_id = webset_id_of(event_data)
        if input_data.webset_id and webset_id and webset_id != input_data.webset_id:
            return

        yield "event_type", event_type
        yield "event_id", str(payload.get("id") or payload.get("eventId") or "")
        yield "webset_id", webset_id or input_data.webset_id
        yield "data", event_data
        yield "timestamp", str(
            payload.get("createdAt") or payload.get("occurredAt") or ""
        )
        yield "metadata", payload.get("metadata") or event_data.get("metadata") or {}

    def _should_process_event(
        self, event_type: str, event_filter: WebsetEventFilter
    ) -> bool:
        """Check if an event should be processed based on the filter.

        Event types the filter has no field for are processed, so new Exa
        events are not dropped silently.
        """
        selections = event_filter.model_dump()
        field = filter_field_for(event_type)
        return selections.get(field, True)


def webset_id_of(event_data: dict[str, Any]) -> str:
    """Return the webset an event's resource belongs to, or "" if none."""
    if event_data.get("object") == "webset":
        return str(event_data.get("id") or "")
    return str(event_data.get("websetId") or "")
