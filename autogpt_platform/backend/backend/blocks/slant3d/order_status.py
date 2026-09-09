from urllib.parse import quote

from backend.blocks._base import BlockOutput, BlockSchemaInput, BlockSchemaOutput
from backend.data.model import APIKeyCredentials, SchemaField
from backend.util.settings import BehaveAs, Settings

from ._api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    Slant3DCredentialsField,
    Slant3DCredentialsInput,
)
from .base import Slant3DBlockBase


class Slant3DGetOrdersBlock(Slant3DBlockBase):
    class Input(BlockSchemaInput):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()

    class Output(BlockSchemaOutput):
        orders: list[str] = SchemaField(
            description="Public IDs of all orders for the account"
        )

    def __init__(self):
        super().__init__(
            id="42283bf5-8a32-4fb4-92a2-60a9ea48e105",
            description="Get all orders for the account",
            input_schema=self.Input,
            output_schema=self.Output,
            disabled=Settings().config.behave_as == BehaveAs.CLOUD,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("orders", ["SLANT_1234567890"])],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "data": [{"publicId": "SLANT_1234567890"}],
                    "pagination": {"totalPages": 1},
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        page = 1
        orders = []
        while True:
            result = await self._make_request(
                "GET",
                "orders",
                credentials.api_key.get_secret_value(),
                params={"page": page, "limit": 100},
            )
            orders.extend(order["publicId"] for order in result["data"])
            total_pages = result.get("pagination", {}).get("totalPages")
            if not result["data"] or (total_pages is not None and page >= total_pages):
                break
            if total_pages is None and len(result["data"]) < 100:
                break
            page += 1
        yield "orders", orders


class Slant3DTrackingBlock(Slant3DBlockBase):
    class Input(BlockSchemaInput):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        order_id: str = SchemaField(description="Slant3D public order ID to track")

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Order status")
        tracking_numbers: list[str] = SchemaField(
            description="Shipment tracking numbers"
        )

    def __init__(self):
        super().__init__(
            id="dd7c0293-c5af-4551-ba3e-fc162fb1fb89",
            description="Track order status and shipping",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "order_id": "SLANT_1234567890",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "PAID"), ("tracking_numbers", [])],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "data": {"order": {"status": "PAID", "fulfillment": None}}
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        result = await self._make_request(
            "GET",
            f"orders/{quote(input_data.order_id, safe='')}",
            credentials.api_key.get_secret_value(),
        )
        order = result["data"]["order"]
        yield "status", order["status"]
        yield "tracking_numbers", (order.get("fulfillment") or {}).get(
            "trackingNumbers"
        ) or []


class Slant3DCancelOrderBlock(Slant3DBlockBase):
    class Input(BlockSchemaInput):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        order_id: str = SchemaField(description="Slant3D public order ID to cancel")

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Cancellation status message")

    def __init__(self):
        super().__init__(
            id="54de35e1-407f-450b-b5fa-3b5e2eba8185",
            description="Cancel an order before production starts",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "order_id": "SLANT_1234567890",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Order cancelled")],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "success": True,
                    "message": "Order cancelled",
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        result = await self._make_request(
            "DELETE",
            f"orders/{quote(input_data.order_id, safe='')}",
            credentials.api_key.get_secret_value(),
        )
        yield "status", result["message"]


class Slant3DProcessOrderBlock(Slant3DBlockBase):
    class Input(BlockSchemaInput):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        order_id: str = SchemaField(
            description="Uncharged draft order ID from an estimate block"
        )

    class Output(BlockSchemaOutput):
        order_id: str = SchemaField(description="Processed Slant3D public order ID")

    def __init__(self):
        super().__init__(
            id="f46fe306-86d8-4da4-8043-ad93614ed2b7",
            description=(
                "Submit an approved Slant3D draft to order physical 3D-printed parts for manufacturing and delivery. "
                "Uses the order_id from Estimate Order or Estimate Shipping, charges the connected payment method, "
                "and starts production. Run only after the customer approves the quoted order."
            ),
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "order_id": "SLANT_1234567890",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("order_id", "SLANT_1234567890")],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "data": {"publicId": "SLANT_1234567890"}
                }
            },
            is_sensitive_action=True,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        result = await self._process_order(
            input_data.order_id, credentials.api_key.get_secret_value()
        )
        yield "order_id", result["data"]["publicId"]
