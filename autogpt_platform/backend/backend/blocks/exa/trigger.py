import json
import logging
from pathlib import Path
from pydantic import BaseModel

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockWebhookConfig,
)
from backend.data.model import SchemaField, APIKeyCredentials
from backend.integrations.providers import ProviderName

from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ExaCredentialsField,
    ExaCredentialsInput,
)

logger = logging.getLogger(__name__)


class ExaTriggerBase:
    """Base class for Exa webhook triggers."""
    
    class Input(BlockSchema):
        """Base input schema for Exa triggers."""
        credentials: ExaCredentialsInput = ExaCredentialsField()
        # --8<-- [start:example-payload-field]
        payload: dict = SchemaField(hidden=True, default_factory=dict)
        # --8<-- [end:example-payload-field]

    class Output(BlockSchema):
        """Base output schema for Exa triggers."""
        payload: dict = SchemaField(
            description="The complete webhook payload that was received from Exa. "
            "Includes information about the event type, data, and creation timestamp."
        )
        error: str = SchemaField(
            description="Error message if the payload could not be processed"
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """Process the webhook payload from Exa.
        
        Args:
            input_data: The input data containing the webhook payload
            
        Yields:
            The complete webhook payload
        """
        yield "payload", input_data.payload


class ExaWebsetTriggerBlock(ExaTriggerBase, Block):
    """Block for handling Exa Webset webhook events.
    
    This block triggers on various Exa Webset events such as webset creation,
    deletion, search completion, etc. and outputs the event details.
    """
    
    EXAMPLE_PAYLOAD_FILE = (
        Path(__file__).parent / "example_payloads" / "webset.created.json"
    )
    
    class Input(ExaTriggerBase.Input):
        """Input schema for Exa Webset trigger with event filtering options."""
        
        class EventsFilter(BaseModel):
            """
            Event filter options for Exa Webset webhooks.
            
            See: https://docs.exa.ai/api-reference/webhooks
            """
            # Webset events
            webset_created: bool = False
            webset_deleted: bool = False
            webset_paused: bool = False
            webset_idle: bool = False
            
            # Search events
            webset_search_created: bool = False
            webset_search_updated: bool = False
            webset_search_completed: bool = False
            webset_search_canceled: bool = False
            
            # Item events
            webset_item_created: bool = False
            webset_item_enriched: bool = False
            
            # Export events
            webset_export_created: bool = False
            webset_export_completed: bool = False

        events: EventsFilter = SchemaField(
            title="Events", description="The events to subscribe to"
        )

    class Output(ExaTriggerBase.Output):
        """Output schema for Exa Webset trigger with event-specific fields."""
        
        event_type: str = SchemaField(
            description="The type of event that triggered the webhook"
        )
        webset_id: str = SchemaField(
            description="The ID of the affected webset"
        )
        created_at: str = SchemaField(
            description="Timestamp when the event was created"
        )
        data: dict = SchemaField(
            description="Object containing the full resource that triggered the event"
        )

    def __init__(self):
        """Initialize the Exa Webset trigger block with its configuration."""
        
        # Define a webhook type constant for Exa
        class ExaWebhookType:
            """Constants for Exa webhook types."""
            WEBSET = "webset"

        # Create example payload
        example_payload = {
            "id": "663de972-bfe7-47ef-b4d7-179cfed7aa44",
            "object": "event",
            "type": "webset.created",
            "data": {
                "id": "wbs_123456789",
                "name": "Example Webset",
                "description": "An example webset for testing"
            },
            "createdAt": "2023-06-01T12:00:00Z"
        }

        # Map UI event names to API event names
        self.event_mapping = {
            "webset_created": "webset.created",
            "webset_deleted": "webset.deleted",
            "webset_paused": "webset.paused",
            "webset_idle": "webset.idle",
            "webset_search_created": "webset.search.created",
            "webset_search_updated": "webset.search.updated",
            "webset_search_completed": "webset.search.completed",
            "webset_search_canceled": "webset.search.canceled",
            "webset_item_created": "webset.item.created",
            "webset_item_enriched": "webset.item.enriched",
            "webset_export_created": "webset.export.created",
            "webset_export_completed": "webset.export.completed"
        }

        super().__init__(
            id="804ac1ed-d692-4ccb-a390-739a846a2667",
            description="This block triggers on Exa Webset events and outputs the event type and payload.",
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.INPUT},
            input_schema=ExaWebsetTriggerBlock.Input,
            output_schema=ExaWebsetTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.EXA,
                webhook_type=ExaWebhookType.WEBSET,
                resource_format="",  # Exa doesn't require a specific resource format
                event_filter_input="events",
                event_format="webset.{event}",
            ),
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "events": {"webset_created": True, "webset_search_completed": True},
                "payload": example_payload,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", example_payload),
                ("event_type", example_payload["type"]),
                ("webset_id", "wbs_123456789"),
                ("created_at", "2023-06-01T12:00:00Z"),
                ("data", example_payload["data"]),
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:  # type: ignore
        """Process Exa Webset webhook events.
        
        Args:
            input_data: The input data containing the webhook payload and event filter
            
        Yields:
            Event details including event type, webset ID, creation timestamp, and data,
            or an error message if the event type doesn't match the filter or if required
            fields are missing from the payload.
        """
        yield from super().run(input_data, **kwargs)
        try:
            # Get the event type from the payload
            event_type = input_data.payload["type"]
            
            # Check if this event type is in the user's selected events
            # Convert API event name to UI event name (reverse mapping)
            ui_event_name = next((k for k, v in self.event_mapping.items() if v == event_type), None)
            
            # Only process events that match the filter
            if ui_event_name and getattr(input_data.events, ui_event_name, False):
                yield "event_type", event_type
                yield "webset_id", input_data.payload["data"]["id"]
                yield "created_at", input_data.payload["createdAt"]
                yield "data", input_data.payload["data"]
            else:
                yield "error", f"Event type {event_type} not in selected events filter"
        except KeyError as e:
            yield "error", f"Missing expected field in payload: {str(e)}"
