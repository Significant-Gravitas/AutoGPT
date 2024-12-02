import uuid
from typing import List

import requests as baserequests

from backend.data.block import BlockOutput, BlockSchema
from backend.data.model import APIKeyCredentials, SchemaField
from backend.util import settings
from backend.util.settings import BehaveAs

from ._api import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    CustomerDetails,
    OrderItem,
    Slant3DCredentialsField,
    Slant3DCredentialsInput,
)
from .base import Slant3DBlockBase


class Slant3DCreateOrderBlock(Slant3DBlockBase):
    """Block for creating new orders"""

    class Input(BlockSchema):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        order_number: str = SchemaField(
            description="Your custom order number (or leave blank for a random one)",
            default_factory=lambda: str(uuid.uuid4()),
        )
        customer: CustomerDetails = SchemaField(
            description="Customer details for where to ship the item",
            advanced=False,
        )
        items: List[OrderItem] = SchemaField(
            description="List of items to print",
            advanced=False,
        )

    class Output(BlockSchema):
        order_id: str = SchemaField(description="Slant3D order ID")
        error: str = SchemaField(description="Error message if order failed")

    def __init__(self):
        super().__init__(
            id="f73007d6-f48f-4aaf-9e6b-6883998a09b4",
            description="Create a new print order",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "order_number": "TEST-001",
                "customer": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "phone": "123-456-7890",
                    "address": "123 Test St",
                    "city": "Test City",
                    "state": "TS",
                    "zip": "12345",
                },
                "items": [
                    {
                        "file_url": "https://example.com/model.stl",
                        "quantity": "1",
                        "color": "black",
                        "profile": "PLA",
                    }
                ],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("order_id", "314144241")],
            test_mock={
                "_make_request": lambda *args, **kwargs: {"orderId": "314144241"},
                "_convert_to_color": lambda *args, **kwargs: "black",
            },
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            order_data = self._format_order_data(
                input_data.customer,
                input_data.order_number,
                input_data.items,
                credentials.api_key.get_secret_value(),
            )
            result = self._make_request(
                "POST", "order", credentials.api_key.get_secret_value(), json=order_data
            )
            yield "order_id", result["orderId"]
        except Exception as e:
            yield "error", str(e)
            raise


class Slant3DEstimateOrderBlock(Slant3DBlockBase):
    """Block for getting order cost estimates"""

    class Input(BlockSchema):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        order_number: str = SchemaField(
            description="Your custom order number (or leave blank for a random one)",
            default_factory=lambda: str(uuid.uuid4()),
        )
        customer: CustomerDetails = SchemaField(
            description="Customer details for where to ship the item",
            advanced=False,
        )
        items: List[OrderItem] = SchemaField(
            description="List of items to print",
            advanced=False,
        )

    class Output(BlockSchema):
        total_price: float = SchemaField(description="Total price in USD")
        shipping_cost: float = SchemaField(description="Shipping cost")
        printing_cost: float = SchemaField(description="Printing cost")
        error: str = SchemaField(description="Error message if estimation failed")

    def __init__(self):
        super().__init__(
            id="bf8823d6-b42a-48c7-b558-d7c117f2ae85",
            description="Get order cost estimate",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "order_number": "TEST-001",
                "customer": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "phone": "123-456-7890",
                    "address": "123 Test St",
                    "city": "Test City",
                    "state": "TS",
                    "zip": "12345",
                },
                "items": [
                    {
                        "file_url": "https://example.com/model.stl",
                        "quantity": "1",
                        "color": "black",
                        "profile": "PLA",
                    }
                ],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("total_price", 9.31),
                ("shipping_cost", 5.56),
                ("printing_cost", 3.75),
            ],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "totalPrice": 9.31,
                    "shippingCost": 5.56,
                    "printingCost": 3.75,
                },
                "_convert_to_color": lambda *args, **kwargs: "black",
            },
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        order_data = self._format_order_data(
            input_data.customer,
            input_data.order_number,
            input_data.items,
            credentials.api_key.get_secret_value(),
        )
        try:
            result = self._make_request(
                "POST",
                "order/estimate",
                credentials.api_key.get_secret_value(),
                json=order_data,
            )
            yield "total_price", result["totalPrice"]
            yield "shipping_cost", result["shippingCost"]
            yield "printing_cost", result["printingCost"]
        except baserequests.HTTPError as e:
            yield "error", str(f"Error estimating order: {e} {e.response.text}")
            raise


class Slant3DEstimateShippingBlock(Slant3DBlockBase):
    """Block for getting shipping cost estimates"""

    class Input(BlockSchema):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        order_number: str = SchemaField(
            description="Your custom order number (or leave blank for a random one)",
            default_factory=lambda: str(uuid.uuid4()),
        )
        customer: CustomerDetails = SchemaField(
            description="Customer details for where to ship the item"
        )
        items: List[OrderItem] = SchemaField(
            description="List of items to print",
            advanced=False,
        )

    class Output(BlockSchema):
        shipping_cost: float = SchemaField(description="Estimated shipping cost")
        currency_code: str = SchemaField(description="Currency code (e.g., 'usd')")
        error: str = SchemaField(description="Error message if estimation failed")

    def __init__(self):
        super().__init__(
            id="00aae2a1-caf6-4a74-8175-39a0615d44e1",
            description="Get shipping cost estimate",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "order_number": "TEST-001",
                "customer": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "phone": "123-456-7890",
                    "address": "123 Test St",
                    "city": "Test City",
                    "state": "TS",
                    "zip": "12345",
                },
                "items": [
                    {
                        "file_url": "https://example.com/model.stl",
                        "quantity": "1",
                        "color": "black",
                        "profile": "PLA",
                    }
                ],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("shipping_cost", 4.81), ("currency_code", "usd")],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "shippingCost": 4.81,
                    "currencyCode": "usd",
                },
                "_convert_to_color": lambda *args, **kwargs: "black",
            },
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            order_data = self._format_order_data(
                input_data.customer,
                input_data.order_number,
                input_data.items,
                credentials.api_key.get_secret_value(),
            )
            result = self._make_request(
                "POST",
                "order/estimateShipping",
                credentials.api_key.get_secret_value(),
                json=order_data,
            )
            yield "shipping_cost", result["shippingCost"]
            yield "currency_code", result["currencyCode"]
        except Exception as e:
            yield "error", str(e)
            raise


class Slant3DGetOrdersBlock(Slant3DBlockBase):
    """Block for retrieving all orders"""

    class Input(BlockSchema):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()

    class Output(BlockSchema):
        orders: List[str] = SchemaField(description="List of orders with their details")
        error: str = SchemaField(description="Error message if request failed")

    def __init__(self):
        super().__init__(
            id="42283bf5-8a32-4fb4-92a2-60a9ea48e105",
            description="Get all orders for the account",
            input_schema=self.Input,
            output_schema=self.Output,
            # This block is disabled for cloud hosted because it allows access to all orders for the account
            disabled=settings.Settings().config.behave_as == BehaveAs.CLOUD,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "orders",
                    [
                        "1234567890",
                    ],
                )
            ],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "ordersData": [
                        {
                            "orderId": 1234567890,
                            "orderTimestamp": {
                                "_seconds": 1719510986,
                                "_nanoseconds": 710000000,
                            },
                        }
                    ]
                }
            },
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = self._make_request(
                "GET", "order", credentials.api_key.get_secret_value()
            )
            yield "orders", [str(order["orderId"]) for order in result["ordersData"]]
        except Exception as e:
            yield "error", str(e)
            raise


class Slant3DTrackingBlock(Slant3DBlockBase):
    """Block for tracking order status and shipping"""

    class Input(BlockSchema):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        order_id: str = SchemaField(description="Slant3D order ID to track")

    class Output(BlockSchema):
        status: str = SchemaField(description="Order status")
        tracking_numbers: List[str] = SchemaField(
            description="List of tracking numbers"
        )
        error: str = SchemaField(description="Error message if tracking failed")

    def __init__(self):
        super().__init__(
            id="dd7c0293-c5af-4551-ba3e-fc162fb1fb89",
            description="Track order status and shipping",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "order_id": "314144241",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "awaiting_shipment"), ("tracking_numbers", [])],
            test_mock={
                "_make_request": lambda *args, **kwargs: {
                    "status": "awaiting_shipment",
                    "trackingNumbers": [],
                }
            },
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = self._make_request(
                "GET",
                f"order/{input_data.order_id}/get-tracking",
                credentials.api_key.get_secret_value(),
            )
            yield "status", result["status"]
            yield "tracking_numbers", result["trackingNumbers"]
        except Exception as e:
            yield "error", str(e)
            raise


class Slant3DCancelOrderBlock(Slant3DBlockBase):
    """Block for canceling orders"""

    class Input(BlockSchema):
        credentials: Slant3DCredentialsInput = Slant3DCredentialsField()
        order_id: str = SchemaField(description="Slant3D order ID to cancel")

    class Output(BlockSchema):
        status: str = SchemaField(description="Cancellation status message")
        error: str = SchemaField(description="Error message if cancellation failed")

    def __init__(self):
        super().__init__(
            id="54de35e1-407f-450b-b5fa-3b5e2eba8185",
            description="Cancel an existing order",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "order_id": "314144241",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Order cancelled")],
            test_mock={
                "_make_request": lambda *args, **kwargs: {"status": "Order cancelled"}
            },
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            result = self._make_request(
                "DELETE",
                f"order/{input_data.order_id}",
                credentials.api_key.get_secret_value(),
            )
            yield "status", result["status"]
        except Exception as e:
            yield "error", str(e)
            raise
