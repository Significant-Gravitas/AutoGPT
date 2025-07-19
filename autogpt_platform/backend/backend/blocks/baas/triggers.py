"""
Meeting BaaS webhook trigger blocks.
"""

from pydantic import BaseModel

from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockType,
    BlockWebhookConfig,
    CredentialsMetaInput,
    ProviderName,
    SchemaField,
)

from ._config import baas


class BaasOnMeetingEventBlock(Block):
    """
    Trigger when Meeting BaaS sends meeting-related events:
    bot.status_change, complete, failed, transcription_complete
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        webhook_url: str = SchemaField(
            description="URL to receive webhooks (auto-generated)",
            default="",
            hidden=True,
        )

        class EventsFilter(BaseModel):
            """Meeting event types to subscribe to"""

            bot_status_change: bool = SchemaField(
                description="Bot status changes", default=True
            )
            complete: bool = SchemaField(description="Meeting completed", default=True)
            failed: bool = SchemaField(description="Meeting failed", default=True)
            transcription_complete: bool = SchemaField(
                description="Transcription completed", default=True
            )

        events: EventsFilter = SchemaField(
            title="Events", description="The events to subscribe to"
        )

        payload: dict = SchemaField(
            description="Webhook payload data",
            default={},
            hidden=True,
        )

    class Output(BlockSchema):
        event_type: str = SchemaField(description="Type of event received")
        data: dict = SchemaField(description="Event data payload")

    def __init__(self):
        super().__init__(
            id="3d4e5f6a-7b8c-9d0e-1f2a-3b4c5d6e7f8a",
            description="Receive meeting events from Meeting BaaS webhooks",
            categories={BlockCategory.INPUT},
            input_schema=self.Input,
            output_schema=self.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("baas"),
                webhook_type="meeting_event",
                event_filter_input="events",
                resource_format="meeting",
            ),
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload

        # Extract event type and data
        event_type = payload.get("event", "unknown")
        data = payload.get("data", {})

        # Map event types to filter fields
        event_filter_map = {
            "bot.status_change": input_data.events.bot_status_change,
            "complete": input_data.events.complete,
            "failed": input_data.events.failed,
            "transcription_complete": input_data.events.transcription_complete,
        }

        # Filter events if needed
        if not event_filter_map.get(event_type, False):
            return  # Skip unwanted events

        yield "event_type", event_type
        yield "data", data


class BaasOnCalendarEventBlock(Block):
    """
    Trigger when Meeting BaaS sends calendar-related events:
    event.added, event.updated, event.deleted, calendar.synced
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = baas.credentials_field(
            description="Meeting BaaS API credentials"
        )
        webhook_url: str = SchemaField(
            description="URL to receive webhooks (auto-generated)",
            default="",
            hidden=True,
        )

        class EventsFilter(BaseModel):
            """Calendar event types to subscribe to"""

            event_added: bool = SchemaField(
                description="Calendar event added", default=True
            )
            event_updated: bool = SchemaField(
                description="Calendar event updated", default=True
            )
            event_deleted: bool = SchemaField(
                description="Calendar event deleted", default=True
            )
            calendar_synced: bool = SchemaField(
                description="Calendar synced", default=True
            )

        events: EventsFilter = SchemaField(
            title="Events", description="The events to subscribe to"
        )

        payload: dict = SchemaField(
            description="Webhook payload data",
            default={},
            hidden=True,
        )

    class Output(BlockSchema):
        event_type: str = SchemaField(description="Type of event received")
        data: dict = SchemaField(description="Event data payload")

    def __init__(self):
        super().__init__(
            id="4e5f6a7b-8c9d-0e1f-2a3b-4c5d6e7f8a9b",
            description="Receive calendar events from Meeting BaaS webhooks",
            categories={BlockCategory.INPUT},
            input_schema=self.Input,
            output_schema=self.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("baas"),
                webhook_type="calendar_event",
                event_filter_input="events",
                resource_format="calendar",
            ),
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload

        # Extract event type and data
        event_type = payload.get("event", "unknown")
        data = payload.get("data", {})

        # Map event types to filter fields
        event_filter_map = {
            "event.added": input_data.events.event_added,
            "event.updated": input_data.events.event_updated,
            "event.deleted": input_data.events.event_deleted,
            "calendar.synced": input_data.events.calendar_synced,
        }

        # Filter events if needed
        if not event_filter_map.get(event_type, False):
            return  # Skip unwanted events

        yield "event_type", event_type
        yield "data", data
