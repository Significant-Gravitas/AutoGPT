from pydantic import BaseModel

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockWebhookConfig,
)
from backend.data.model import SchemaField
from backend.util import settings
from backend.util.settings import AppEnvironment, BehaveAs

from ._api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    Slant3DCredentialsField,
    Slant3DCredentialsInput,
)


class Slant3DTriggerBase:
    """Base class for Slant3D webhook triggers"""

    class Input(BlockSchema):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        # Webhook URL is handled by the webhook system
        payload: dict = SchemaField(hidden=True, default={})

    class Output(BlockSchema):
        payload: dict = SchemaField(
            description="The complete webhook payload received from Slant3D"
        )
        order_id: str = SchemaField(description="The ID of the affected order")
        error: str = SchemaField(
            description="Error message if payload processing failed"
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "payload", input_data.payload
        yield "order_id", input_data.payload["orderId"]


class Slant3DOrderWebhookBlock(Slant3DTriggerBase, Block):
    """Block for handling Slant3D order webhooks"""

    class Input(Slant3DTriggerBase.Input):
        class EventsFilter(BaseModel):
            """
            Currently Slant3D only supports 'SHIPPED' status updates
            Could be expanded in the future with more status types
            """

            shipped: bool = True

        events: EventsFilter = SchemaField(
            title="Events",
            description="Order status events to subscribe to",
            default=EventsFilter(shipped=True),
        )

    class Output(Slant3DTriggerBase.Output):
        status: str = SchemaField(description="The new status of the order")
        tracking_number: str = SchemaField(
            description="The tracking number for the shipment"
        )
        carrier_code: str = SchemaField(description="The carrier code (e.g., 'usps')")

    def __init__(self):
        super().__init__(
            id="8a74c2ad-0104-4640-962f-26c6b69e58cd",
            description=(
                "This block triggers on Slant3D order status updates and outputs "
                "the event details, including tracking information when orders are shipped."
            ),
            # All webhooks are currently subscribed to for all orders. This works for self hosted, but not for cloud hosted prod
            disabled=(
                settings.Settings().config.behave_as == BehaveAs.CLOUD
                and settings.Settings().config.app_env != AppEnvironment.LOCAL
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            webhook_config=BlockWebhookConfig(
                provider="slant3d",
                webhook_type="orders",  # Only one type for now
                resource_format="",  # No resource format needed
                event_filter_input="events",
                event_format="order.{event}",
            ),
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "events": {"shipped": True},
                "payload": {
                    "orderId": "1234567890",
                    "status": "SHIPPED",
                    "trackingNumber": "ABCDEF123456",
                    "carrierCode": "usps",
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "payload",
                    {
                        "orderId": "1234567890",
                        "status": "SHIPPED",
                        "trackingNumber": "ABCDEF123456",
                        "carrierCode": "usps",
                    },
                ),
                ("order_id", "1234567890"),
                ("status", "SHIPPED"),
                ("tracking_number", "ABCDEF123456"),
                ("carrier_code", "usps"),
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:  # type: ignore
        yield from super().run(input_data, **kwargs)

        # Extract and normalize values from the payload
        yield "status", input_data.payload["status"]
        yield "tracking_number", input_data.payload["trackingNumber"]
        yield "carrier_code", input_data.payload["carrierCode"]
