"""Blocks that price a configured design."""

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
from ._inputs import RMFGBasketInput, build_items, credentials_field
from ._models import DFMIssue, Requirement
from ._models_commerce import QuantityOption, Quote, QuotedDesign, ShippingOption
from ._testdata import TEST_DFM_ISSUE, TEST_MATERIAL, TEST_QUOTE, TEST_SHIPPING_OPTION
from ._types import QuoteItemRequest, QuoteStatus, ShipTo

CATEGORIES = {BlockCategory.HARDWARE, BlockCategory.DATA}


class RMFGQuoteOutput(BlockSchemaOutput):
    quote: Quote = SchemaField(description="The full quote")
    quote_id: str = SchemaField(description="Quote ID")
    status: QuoteStatus = SchemaField(
        description="processing, requires_input, ready, blocked, expired or failed"
    )
    is_ready: bool = SchemaField(description="True when the quote can be ordered")
    amount_total_cents: int = SchemaField(
        description="Total in USD cents (shipping and tax excluded until a cart)"
    )
    amount_subtotal_cents: int = SchemaField(description="Sum of items in USD cents")
    unit_amount_cents: int = SchemaField(
        description="Price per completed unit of the first design, in USD cents"
    )
    items: list[QuotedDesign] = SchemaField(
        description="Per-design pricing, line items and DFM findings"
    )
    quantity_options: list[QuantityOption] = SchemaField(
        description="Prices at the other quantities requested, for the first design"
    )
    shipping_options: list[ShippingOption] = SchemaField(
        description="Delivery choices when ship_to was given; pick an id for the cart"
    )
    requirements: list[Requirement] = SchemaField(
        description="Selections or decisions still needed before ordering"
    )
    dfm_issues: list[DFMIssue] = SchemaField(
        description="Manufacturability findings across every design"
    )
    error: str = SchemaField(description="Error message if the request failed")


async def emit_quote(quote: Quote) -> BlockOutput:
    yield "quote", quote
    yield "quote_id", quote.id
    yield "status", quote.status
    yield "is_ready", quote.status == QuoteStatus.READY
    yield "amount_total_cents", quote.amount_total_cents or 0
    yield "amount_subtotal_cents", quote.amount_subtotal_cents or 0
    first = quote.items[0] if quote.items else None
    yield "unit_amount_cents", (first.unit_amount_cents or 0) if first else 0
    yield "items", quote.items
    yield "quantity_options", first.quantity_options if first else []
    yield "shipping_options", quote.shipping_options
    yield "requirements", quote.all_requirements
    yield "dfm_issues", [issue for item in quote.items for issue in item.dfm.issues]


class RMFGCreateQuoteBlock(Block):
    """Price one or more configured designs. Quoting also runs DFM."""

    class Input(RMFGBasketInput):
        ship_to: Optional[ShipTo] = SchemaField(
            description="Destination, to include delivery options in the quote.",
            default=None,
            advanced=True,
        )
        wait_for_ready: bool = SchemaField(
            description="Poll until pricing finishes instead of returning at once.",
            default=True,
            advanced=True,
        )
        timeout_seconds: int = SchemaField(
            description="How long to wait for pricing when wait_for_ready is on.",
            default=300,
            ge=5,
            le=1500,
            advanced=True,
        )
        idempotency_key: str = SchemaField(
            description="Stable key for identical retries; defaults to the node execution ID.",
            default="",
            advanced=True,
        )

    class Output(RMFGQuoteOutput):
        pass

    def __init__(self):
        super().__init__(
            id="916c21ad-f85e-4a08-be83-3a9bffb83404",
            description="Gets an RMFG price and manufacturability findings for a configured design",
            categories=CATEGORIES,
            input_schema=RMFGCreateQuoteBlock.Input,
            output_schema=RMFGCreateQuoteBlock.Output,
            test_input={
                "design_id": TEST_QUOTE.design_id,
                "quantity": 10,
                "material_id": TEST_MATERIAL.id,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("quote", TEST_QUOTE),
                ("quote_id", TEST_QUOTE.id),
                ("status", QuoteStatus.READY),
                ("is_ready", True),
                ("amount_total_cents", 24800),
                ("amount_subtotal_cents", 24800),
                ("unit_amount_cents", 2480),
                ("items", TEST_QUOTE.items),
                ("quantity_options", []),
                ("shipping_options", [TEST_SHIPPING_OPTION]),
                ("requirements", []),
                ("dfm_issues", [TEST_DFM_ISSUE]),
            ],
            test_mock={"create_quote": lambda *args, **kwargs: TEST_QUOTE},
        )

    @staticmethod
    async def create_quote(
        credentials: APIKeyCredentials,
        items: list[QuoteItemRequest],
        ship_to: Optional[ShipTo],
        idempotency_key: str,
        wait_for_ready: bool,
        timeout_seconds: int,
    ) -> Quote:
        client = RMFGClient(credentials)
        quote = await client.create_quote(items, ship_to, idempotency_key)
        if not wait_for_ready:
            return quote
        return await client.wait_for_quote(quote, timeout_seconds)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        node_exec_id: str = "",
        **kwargs,
    ) -> BlockOutput:
        quote = await self.create_quote(
            credentials,
            build_items(input_data),
            input_data.ship_to,
            input_data.idempotency_key or node_exec_id,
            input_data.wait_for_ready,
            input_data.timeout_seconds,
        )
        async for output in emit_quote(quote):
            yield output


class RMFGGetQuoteBlock(Block):
    """Re-read a quote, e.g. one that was still processing."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        quote_id: str = SchemaField(description="Quote ID from Create Quote")
        wait_for_ready: bool = SchemaField(
            description="Poll until pricing finishes instead of returning at once.",
            default=False,
            advanced=True,
        )
        timeout_seconds: int = SchemaField(
            description="How long to wait for pricing when wait_for_ready is on.",
            default=300,
            ge=5,
            le=1500,
            advanced=True,
        )

    class Output(RMFGQuoteOutput):
        pass

    def __init__(self):
        super().__init__(
            id="3e9bb152-0505-4a4e-91a8-a30f0cc3dbfb",
            description="Fetches an RMFG quote by ID",
            categories=CATEGORIES,
            input_schema=RMFGGetQuoteBlock.Input,
            output_schema=RMFGGetQuoteBlock.Output,
            test_input={
                "quote_id": TEST_QUOTE.id,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("quote", TEST_QUOTE),
                ("quote_id", TEST_QUOTE.id),
                ("status", QuoteStatus.READY),
                ("is_ready", True),
                ("amount_total_cents", 24800),
                ("amount_subtotal_cents", 24800),
                ("unit_amount_cents", 2480),
                ("items", TEST_QUOTE.items),
                ("quantity_options", []),
                ("shipping_options", [TEST_SHIPPING_OPTION]),
                ("requirements", []),
                ("dfm_issues", [TEST_DFM_ISSUE]),
            ],
            test_mock={"get_quote": lambda *args, **kwargs: TEST_QUOTE},
        )

    @staticmethod
    async def get_quote(
        credentials: APIKeyCredentials,
        quote_id: str,
        wait_for_ready: bool,
        timeout_seconds: int,
    ) -> Quote:
        client = RMFGClient(credentials)
        quote = await client.get_quote(quote_id)
        if not wait_for_ready:
            return quote
        return await client.wait_for_quote(quote, timeout_seconds)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        quote = await self.get_quote(
            credentials,
            input_data.quote_id,
            input_data.wait_for_ready,
            input_data.timeout_seconds,
        )
        async for output in emit_quote(quote):
            yield output
