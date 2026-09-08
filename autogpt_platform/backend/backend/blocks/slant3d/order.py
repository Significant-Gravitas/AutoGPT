from backend.blocks._base import BlockOutput, BlockSchemaOutput
from backend.data.model import APIKeyCredentials, SchemaField

from ._api import TEST_CREDENTIALS
from ._order import TEST_DRAFT, TEST_ORDER_INPUT, OrderInput
from .base import Slant3DBlockBase


class Slant3DCreateOrderBlock(Slant3DBlockBase):
    Input = OrderInput

    class Output(BlockSchemaOutput):
        order_id: str = SchemaField(description="Slant3D public order ID")

    def __init__(self):
        super().__init__(
            id="f73007d6-f48f-4aaf-9e6b-6883998a09b4",
            description="Create and process a print order, charging the Slant3D account payment method",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input=TEST_ORDER_INPUT,
            test_credentials=TEST_CREDENTIALS,
            test_output=[("order_id", "SLANT_1234567890")],
            test_mock={
                "_make_request": lambda *args, **kwargs: {"data": TEST_DRAFT},
                "_process_order": lambda *args, **kwargs: {
                    "data": {"publicId": "SLANT_1234567890"}
                },
            },
            is_sensitive_action=True,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        order_data = await self._format_order_data(
            input_data.customer,
            input_data.order_number,
            input_data.items,
            api_key,
            input_data.platform_id,
        )
        result = await self._make_request("POST", "orders", api_key, json=order_data)
        order_id = result["data"]["order"]["publicId"]
        try:
            await self._process_order(order_id, api_key)
        except Exception as exc:
            raise RuntimeError(
                f"Could not confirm processing of draft {order_id}. "
                "Check its status before retrying to avoid a duplicate order."
            ) from exc
        yield "order_id", order_id


class Slant3DEstimateOrderBlock(Slant3DBlockBase):
    Input = OrderInput

    class Output(BlockSchemaOutput):
        total_price: float = SchemaField(description="Total price in USD")
        shipping_cost: float = SchemaField(description="Shipping cost in USD")
        printing_cost: float = SchemaField(description="Printing cost in USD")
        order_id: str = SchemaField(
            description="Uncharged draft ID; pass to Process Order to place it"
        )

    def __init__(self):
        super().__init__(
            id="bf8823d6-b42a-48c7-b558-d7c117f2ae85",
            description="Create an uncharged draft order to estimate printing and shipping costs",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input=TEST_ORDER_INPUT,
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("total_price", 9.31),
                ("shipping_cost", 5.56),
                ("printing_cost", 3.75),
                ("order_id", "SLANT_1234567890"),
            ],
            test_mock={"_make_request": lambda *args, **kwargs: {"data": TEST_DRAFT}},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        order_data = await self._format_order_data(
            input_data.customer,
            input_data.order_number,
            input_data.items,
            api_key,
            input_data.platform_id,
        )
        result = await self._make_request("POST", "orders", api_key, json=order_data)
        draft = result["data"]
        yield "total_price", float(draft["totals"]["totalCost"])
        yield "shipping_cost", float(draft["totals"]["deliveryCost"])
        yield "printing_cost", float(draft["totals"]["printingCost"])
        yield "order_id", draft["order"]["publicId"]


class Slant3DEstimateShippingBlock(Slant3DBlockBase):
    Input = OrderInput

    class Output(BlockSchemaOutput):
        shipping_cost: float = SchemaField(description="Estimated shipping cost in USD")
        currency_code: str = SchemaField(description="Currency code")
        order_id: str = SchemaField(
            description="Uncharged draft ID; pass to Process Order to place it"
        )

    def __init__(self):
        super().__init__(
            id="00aae2a1-caf6-4a74-8175-39a0615d44e1",
            description="Create an uncharged draft order to estimate shipping costs",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input=TEST_ORDER_INPUT,
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("shipping_cost", 5.56),
                ("currency_code", "usd"),
                ("order_id", "SLANT_1234567890"),
            ],
            test_mock={"_make_request": lambda *args, **kwargs: {"data": TEST_DRAFT}},
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()
        order_data = await self._format_order_data(
            input_data.customer,
            input_data.order_number,
            input_data.items,
            api_key,
            input_data.platform_id,
        )
        result = await self._make_request("POST", "orders", api_key, json=order_data)
        yield "shipping_cost", float(result["data"]["totals"]["deliveryCost"])
        yield "currency_code", "usd"
        yield "order_id", result["data"]["order"]["publicId"]
