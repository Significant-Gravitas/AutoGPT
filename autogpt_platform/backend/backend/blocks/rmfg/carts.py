"""Blocks that build and update an RMFG cart.

A cart re-quotes on every change and carries a website link a person can pay
from. Paying through the API lives in ``pay_cart``.
"""

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
from ._models import ManufacturingReviewWarning, Requirement
from ._models_commerce import Cart, CartTotals, ShippingOption
from ._testdata import TEST_CART, TEST_MATERIAL, TEST_SHIP_TO, TEST_SHIPPING_OPTION
from ._types import CartStatus, QuoteItemRequest, QuoteStatus, ShipTo

CATEGORIES = {BlockCategory.HARDWARE, BlockCategory.DATA}


class RMFGCartOutput(BlockSchemaOutput):
    cart: Cart = SchemaField(description="The full cart, including its latest quote")
    cart_id: str = SchemaField(description="Cart ID")
    cart_url: str = SchemaField(
        description="Website checkout link; anyone holding it can pay, keep it private"
    )
    status: CartStatus = SchemaField(description="open, checked_out or expired")
    quote_status: QuoteStatus = SchemaField(
        description="Status of the cart's latest quote; only ready carts can be paid"
    )
    is_payable: bool = SchemaField(
        description="True when the cart is open, quoted ready, and has an address and shipping option"
    )
    totals: CartTotals = SchemaField(description="Subtotal, shipping, tax and total")
    amount_total_cents: int = SchemaField(
        description="What checkout will charge, in USD cents"
    )
    shipping_options: list[ShippingOption] = SchemaField(
        description="Delivery choices once ship_to is set; pass an id to Update Cart"
    )
    requirements: list[Requirement] = SchemaField(
        description="Selections or decisions still needed before ordering"
    )
    manufacturing_warnings: list[ManufacturingReviewWarning] = SchemaField(
        description="Advisories from automatic file preparation; they do not block ordering"
    )
    order_id: str = SchemaField(description="Order ID, once the cart has been paid")
    error: str = SchemaField(description="Error message if the request failed")


def is_payable(cart: Cart) -> bool:
    return (
        cart.status == CartStatus.OPEN
        and cart.quote.status == QuoteStatus.READY
        and cart.ship_to is not None
        and bool(cart.shipping_option_id)
    )


async def emit_cart(cart: Cart) -> BlockOutput:
    yield "cart", cart
    yield "cart_id", cart.id
    yield "cart_url", cart.cart_url
    yield "status", cart.status
    yield "quote_status", cart.quote.status
    yield "is_payable", is_payable(cart)
    yield "totals", cart.totals
    yield "amount_total_cents", cart.totals.amount_total_cents
    yield "shipping_options", cart.quote.shipping_options
    yield "requirements", cart.quote.all_requirements
    yield "manufacturing_warnings", cart.manufacturing_warnings
    if cart.order_id:
        yield "order_id", cart.order_id


CART_TEST_OUTPUT = [
    ("cart", TEST_CART),
    ("cart_id", TEST_CART.id),
    ("cart_url", TEST_CART.cart_url),
    ("status", CartStatus.OPEN),
    ("quote_status", QuoteStatus.READY),
    ("is_payable", True),
    ("totals", TEST_CART.totals),
    ("amount_total_cents", 28849),
    ("shipping_options", [TEST_SHIPPING_OPTION]),
    ("requirements", []),
    ("manufacturing_warnings", []),
]


class RMFGCreateCartBlock(Block):
    """Create a priced cart and a checkout link for a configured design."""

    class Input(RMFGBasketInput):
        ship_to: Optional[ShipTo] = SchemaField(
            description="Delivery address. Needed for shipping options, tax and API payment.",
            default=None,
            advanced=False,
        )
        shipping_option_id: str = SchemaField(
            description=(
                "A shipping_options[].id from a quote or cart with the same "
                "address. Can be chosen later with Update Cart."
            ),
            default="",
            advanced=True,
        )
        idempotency_key: str = SchemaField(
            description="Stable key for identical retries; defaults to the node execution ID.",
            default="",
            advanced=True,
        )

    class Output(RMFGCartOutput):
        pass

    def __init__(self):
        super().__init__(
            id="4302f685-7fab-4396-abbb-f7421f2ad511",
            description="Creates an RMFG cart with a website checkout link for a configured design",
            categories=CATEGORIES,
            input_schema=RMFGCreateCartBlock.Input,
            output_schema=RMFGCreateCartBlock.Output,
            test_input={
                "design_id": TEST_CART.quote.design_id,
                "quantity": 10,
                "material_id": TEST_MATERIAL.id,
                "ship_to": TEST_SHIP_TO.model_dump(),
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=CART_TEST_OUTPUT,
            test_mock={"create_cart": lambda *args, **kwargs: TEST_CART},
        )

    @staticmethod
    async def create_cart(
        credentials: APIKeyCredentials,
        items: list[QuoteItemRequest],
        ship_to: Optional[ShipTo],
        shipping_option_id: str,
        client_reference_id: str,
        idempotency_key: str,
    ) -> Cart:
        return await RMFGClient(credentials).create_cart(
            items,
            ship_to,
            shipping_option_id,
            client_reference_id,
            idempotency_key,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        node_exec_id: str = "",
        **kwargs,
    ) -> BlockOutput:
        cart = await self.create_cart(
            credentials,
            build_items(input_data),
            input_data.ship_to,
            input_data.shipping_option_id,
            input_data.client_reference_id,
            input_data.idempotency_key or node_exec_id,
        )
        async for output in emit_cart(cart):
            yield output


class RMFGGetCartBlock(Block):
    """Re-read a cart, e.g. after a person edited it on the website."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        cart_id: str = SchemaField(description="Cart ID from Create Cart")

    class Output(RMFGCartOutput):
        pass

    def __init__(self):
        super().__init__(
            id="c1104739-f8dc-4d7f-a99a-4bb418102287",
            description="Fetches an RMFG cart and its latest quote by ID",
            categories=CATEGORIES,
            input_schema=RMFGGetCartBlock.Input,
            output_schema=RMFGGetCartBlock.Output,
            test_input={"cart_id": TEST_CART.id, "credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=CART_TEST_OUTPUT,
            test_mock={"get_cart": lambda *args, **kwargs: TEST_CART},
        )

    @staticmethod
    async def get_cart(credentials: APIKeyCredentials, cart_id: str) -> Cart:
        return await RMFGClient(credentials).get_cart(cart_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        cart = await self.get_cart(credentials, input_data.cart_id)
        async for output in emit_cart(cart):
            yield output


class RMFGUpdateCartBlock(Block):
    """Set the address or shipping option on an open cart; it re-quotes."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        cart_id: str = SchemaField(description="Cart ID from Create Cart")
        ship_to: Optional[ShipTo] = SchemaField(
            description="New delivery address; leave empty to keep the current one.",
            default=None,
            advanced=False,
        )
        shipping_option_id: str = SchemaField(
            description="A shipping_options[].id to select; empty keeps the current one.",
            default="",
            advanced=False,
        )
        items: list[QuoteItemRequest] = SchemaField(
            description="Replacement basket; empty keeps the current items.",
            default_factory=list,
            advanced=True,
        )
        idempotency_key: str = SchemaField(
            description="Stable key for identical retries; defaults to the node execution ID.",
            default="",
            advanced=True,
        )

    class Output(RMFGCartOutput):
        pass

    def __init__(self):
        super().__init__(
            id="7ecccdcf-9879-433e-8325-7ccc6b3b4870",
            description="Updates an open RMFG cart's address, shipping option or items",
            categories=CATEGORIES,
            input_schema=RMFGUpdateCartBlock.Input,
            output_schema=RMFGUpdateCartBlock.Output,
            test_input={
                "cart_id": TEST_CART.id,
                "shipping_option_id": TEST_SHIPPING_OPTION.id,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=CART_TEST_OUTPUT,
            test_mock={"update_cart": lambda *args, **kwargs: TEST_CART},
        )

    @staticmethod
    async def update_cart(
        credentials: APIKeyCredentials, input_data: Input, idempotency_key: str
    ) -> Cart:
        return await RMFGClient(credentials).update_cart(
            input_data.cart_id,
            items=input_data.items,
            ship_to=input_data.ship_to,
            shipping_option_id=input_data.shipping_option_id,
            idempotency_key=idempotency_key,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        node_exec_id: str = "",
        **kwargs,
    ) -> BlockOutput:
        if not (
            input_data.ship_to or input_data.shipping_option_id or input_data.items
        ):
            raise ValueError(
                "Nothing to update: set ship_to, shipping_option_id or items."
            )
        cart = await self.update_cart(
            credentials, input_data, input_data.idempotency_key or node_exec_id
        )
        async for output in emit_cart(cart):
            yield output
