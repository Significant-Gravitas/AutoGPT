"""Blocks that track paid RMFG orders."""

from typing import Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import RMFGClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._inputs import credentials_field
from ._models_commerce import Order, OrderTracking
from ._testdata import TEST_ORDER
from ._types import OrderStatus

CATEGORIES = {BlockCategory.HARDWARE, BlockCategory.DATA}


class RMFGGetOrderBlock(Block):
    """Read an order's status and tracking."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        order_id: str = SchemaField(
            description="Order ID from Pay Cart or an order.status_changed event"
        )

    class Output(BlockSchemaOutput):
        order: Order = SchemaField(description="The full order")
        order_id: str = SchemaField(description="Order ID")
        status: OrderStatus = SchemaField(
            description="received, in_production, ready_for_pickup, shipped, delivered, cancelled or refunded"
        )
        tracking: Optional[OrderTracking] = SchemaField(
            description="Carrier, number and link once shipped"
        )
        tracking_url: str = SchemaField(
            description="Carrier tracking link, once shipped"
        )
        tracking_number: str = SchemaField(description="Carrier tracking number")
        estimated_ship_date: str = SchemaField(description="Planned ship date")
        amount_total_cents: int = SchemaField(description="Amount charged, USD cents")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="281a1730-e414-4773-8a20-7c3472cc267d",
            description="Fetches an RMFG order's status and shipment tracking",
            categories=CATEGORIES,
            input_schema=RMFGGetOrderBlock.Input,
            output_schema=RMFGGetOrderBlock.Output,
            test_input={
                "order_id": TEST_ORDER.id,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("order", TEST_ORDER),
                ("order_id", TEST_ORDER.id),
                ("status", OrderStatus.SHIPPED),
                ("tracking", TEST_ORDER.tracking),
                (
                    "tracking_url",
                    "https://www.ups.com/track?tracknum=1Z999AA10123456784",
                ),
                ("tracking_number", "1Z999AA10123456784"),
                ("estimated_ship_date", "2026-09-08"),
                ("amount_total_cents", 28849),
            ],
            test_mock={"get_order": lambda *args, **kwargs: TEST_ORDER},
        )

    @staticmethod
    async def get_order(credentials: APIKeyCredentials, order_id: str) -> Order:
        return await RMFGClient(credentials).get_order(order_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        order = await self.get_order(credentials, input_data.order_id)
        yield "order", order
        yield "order_id", order.id
        yield "status", order.status
        yield "tracking", order.tracking
        if order.tracking and order.tracking.url:
            yield "tracking_url", order.tracking.url
        if order.tracking and order.tracking.number:
            yield "tracking_number", order.tracking.number
        if order.estimated_ship_date:
            yield "estimated_ship_date", order.estimated_ship_date
        yield "amount_total_cents", order.amount_total_cents or 0


class RMFGListOrdersBlock(Block):
    """Page through the account's orders, newest first."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        limit: int = SchemaField(
            description="Orders per page.", default=20, ge=1, le=100
        )
        cursor: str = SchemaField(
            description="next_cursor from a previous page; empty for the first page.",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        orders: list[Order] = SchemaField(description="This page of orders")
        order: Order = SchemaField(description="One order at a time")
        order_ids: list[str] = SchemaField(description="IDs in the same order")
        next_cursor: str = SchemaField(
            description="Pass back as cursor to fetch the next page; empty on the last page"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="4dd0f5c1-aaa6-4c20-b942-613aea6528c9",
            description="Lists the RMFG account's manufacturing orders",
            categories=CATEGORIES,
            input_schema=RMFGListOrdersBlock.Input,
            output_schema=RMFGListOrdersBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("orders", [TEST_ORDER]),
                ("order", TEST_ORDER),
                ("order_ids", [TEST_ORDER.id]),
                ("next_cursor", ""),
            ],
            test_mock={"list_orders": lambda *args, **kwargs: ([TEST_ORDER], None)},
        )

    @staticmethod
    async def list_orders(
        credentials: APIKeyCredentials, limit: int, cursor: str
    ) -> tuple[list[Order], Optional[str]]:
        return await RMFGClient(credentials).list_orders(limit, cursor)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        orders, next_cursor = await self.list_orders(
            credentials, input_data.limit, input_data.cursor
        )
        yield "orders", orders
        for order in orders:
            yield "order", order
        yield "order_ids", [order.id for order in orders]
        yield "next_cursor", next_cursor or ""
