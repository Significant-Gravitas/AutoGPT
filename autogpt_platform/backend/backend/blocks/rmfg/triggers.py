"""Trigger block fired by RMFG lifecycle events.

Dropping this block on a graph registers a webhook endpoint with RMFG through
its API; removing it deletes the endpoint again. Deliveries are verified
against the signing secret RMFG returned at registration.
"""

from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockWebhookConfig,
    CredentialsMetaInput,
    ProviderName,
    SchemaField,
)

from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, rmfg
from ._inputs import credentials_field
from ._webhook import RMFGWebhookType

_JSON_OBJECT = TypeAdapter(dict[str, Any])

EXAMPLE_PAYLOAD = {
    "id": "evt_01J9RMFG0000000000000001",
    "type": "order.status_changed",
    "created_at": "2026-09-08T16:00:00Z",
    "data": {
        "id": "ord_001",
        "object": "order",
        "status": "shipped",
        "status_url": "https://api.rmfg.com/v1/orders/ord_001",
    },
}


class RMFGEventTriggerBlock(Block):
    """Start a graph when a design, quote, cart or order changes at RMFG."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()

        class EventsFilter(BaseModel):
            """Lifecycle events RMFG can deliver. Names mirror the API's event types."""

            design_ready: bool = False
            design_failed: bool = False
            production_files_ready: bool = False
            production_files_failed: bool = False
            quote_ready: bool = False
            quote_failed: bool = False
            cart_checked_out: bool = False
            order_status_changed: bool = True

        events: EventsFilter = SchemaField(
            title="Events",
            description="Which RMFG events start this graph",
            default_factory=EventsFilter,
        )
        payload: dict = SchemaField(hidden=True, default_factory=dict)

    class Output(BlockSchemaOutput):
        payload: dict = SchemaField(description="The raw event RMFG delivered")
        event: str = SchemaField(
            description="RMFG event type, e.g. order.status_changed"
        )
        event_id: str = SchemaField(description="Unique ID of this event")
        resource_id: str = SchemaField(
            description="ID of the design, DFM report, quote, cart or order concerned"
        )
        resource_type: str = SchemaField(
            description="design, dfm_report, quote, cart or order"
        )
        status: str = SchemaField(
            description="The resource's new status, when the event carries one"
        )
        status_url: str = SchemaField(
            description="API URL of the resource, for fetching its full state"
        )
        created_at: str = SchemaField(description="When RMFG emitted the event")

    def __init__(self):
        super().__init__(
            id="62db5c4c-186e-43d3-8078-886b2c29848a",
            description="Triggers when an RMFG design, quote, cart or order changes",
            categories={BlockCategory.INPUT, BlockCategory.HARDWARE},
            input_schema=RMFGEventTriggerBlock.Input,
            output_schema=RMFGEventTriggerBlock.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName(rmfg.name),
                webhook_type=RMFGWebhookType.ACCOUNT,
                resource_format="",
                event_filter_input="events",
                event_format="{event}",
            ),
            test_input={
                "events": {"order_status_changed": True},
                "credentials": TEST_CREDENTIALS_INPUT,
                "payload": EXAMPLE_PAYLOAD,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("payload", EXAMPLE_PAYLOAD),
                ("event", "order.status_changed"),
                ("event_id", "evt_01J9RMFG0000000000000001"),
                ("resource_id", "ord_001"),
                ("resource_type", "order"),
                ("status", "shipped"),
                ("status_url", "https://api.rmfg.com/v1/orders/ord_001"),
                ("created_at", "2026-09-08T16:00:00Z"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload
        yield "payload", payload
        yield "event", str(payload.get("type") or "")
        yield "event_id", str(payload.get("id") or "")

        data = _event_data(payload)
        yield "resource_id", str(data.get("id") or "")
        yield "resource_type", str(data.get("object") or "")
        yield "status", str(data.get("status") or "")
        yield "status_url", str(data.get("status_url") or "")
        yield "created_at", str(payload.get("created_at") or "")


def _event_data(payload: dict[str, Any]) -> dict[str, Any]:
    """The event's ``data`` object, or empty when the body has another shape."""
    try:
        return _JSON_OBJECT.validate_python(payload.get("data") or {})
    except ValidationError:
        return {}
