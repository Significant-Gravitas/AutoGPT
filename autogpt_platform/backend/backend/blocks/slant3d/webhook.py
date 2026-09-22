from pydantic import BaseModel

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockWebhookConfig,
)
from backend.data.model import SchemaField
from backend.integrations.providers import ProviderName
from backend.util.settings import AppEnvironment, BehaveAs, Settings

from ._api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    Slant3DCredentialsField,
    Slant3DCredentialsInput,
)

settings = Settings()


class Slant3DTriggerBase:
    """Base class for Slant3D webhook triggers"""

    class Input(BlockSchemaInput):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        # Webhook URL is handled by the webhook system
        payload: dict = SchemaField(hidden=True, default_factory=dict)

    class Output(BlockSchemaOutput):
        payload: dict = SchemaField(
            description="The complete webhook payload received from Slant3D"
        )
        order_id: str = SchemaField(description="The ID of the affected order")
        error: str = SchemaField(
            description="Error message if payload processing failed"
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "payload", input_data.payload
        yield "order_id", input_data.payload["orderId"]


class Slant3DOrderWebhookBlock(Slant3DTriggerBase, Block):
    """Block for handling Slant3D order webhooks"""

    class Input(Slant3DTriggerBase.Input):
        platform_id: str = SchemaField(
            default="",
            description="Required for new v2 subscriptions; retained v1 subscriptions may omit this platform ID. Use a platform without another webhook.",
        )

        class EventsFilter(BaseModel):
            shipped: bool = True
            created: bool = False
            updated: bool = False
            cancelled: bool = False
            delivered: bool = False
            error: bool = False

        events: EventsFilter = SchemaField(
            title="Events",
            description="Order status events to subscribe to",
            default=EventsFilter(shipped=True),
        )

    class Output(Slant3DTriggerBase.Output):
        status: str = SchemaField(description="The new status of the order")
        tracking_number: str = SchemaField(
            description="Shipment tracking number, empty before shipment"
        )
        carrier_code: str = SchemaField(
            description="Carrier code when supplied by Slant3D, otherwise empty"
        )

    def __init__(self):
        super().__init__(
            id="8a74c2ad-0104-4640-962f-26c6b69e58cd",
            description=(
                "This block triggers on Slant3D order status updates and outputs "
                "the event details, including tracking information when orders are shipped."
            ),
            # All webhooks are currently subscribed to for all orders. This works for self hosted, but not for cloud hosted prod
            disabled=(
                settings.config.behave_as == BehaveAs.CLOUD
                and settings.config.app_env != AppEnvironment.LOCAL
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName.SLANT3D,
                webhook_type="orders",
                resource_format="{platform_id}",
                event_filter_input="events",
                event_format="order.{event}",
            ),
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "platform_id": "55555555-5555-4555-8555-555555555555",
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

    async def run(self, input_data: Slant3DTriggerBase.Input, **kwargs) -> BlockOutput:
        async for name, value in super().run(input_data, **kwargs):
            yield name, value

        # Extract and normalize values from the payload
        yield "status", input_data.payload["status"]
        yield "tracking_number", input_data.payload["trackingNumber"]
        yield "carrier_code", input_data.payload["carrierCode"]
