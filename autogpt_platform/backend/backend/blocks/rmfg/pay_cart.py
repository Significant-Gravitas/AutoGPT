"""Block that pays an RMFG cart through the API.

Paying charges the account's saved card and places a real production order,
so this is a sensitive action that the platform asks a person to approve.
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockOutput,
    BlockSchemaInput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import RMFGClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._inputs import credentials_field
from ._models_commerce import Cart
from ._testdata import TEST_CART, TEST_PAID_CART, TEST_SHIPPING_OPTION
from ._types import CartStatus, PaymentStatus, PaymentType, QuoteStatus
from .carts import CATEGORIES, RMFGCartOutput, emit_cart


class RMFGPayCartBlock(Block):
    """Charge the account's saved card for a ready cart and place the order."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = credentials_field()
        cart_id: str = SchemaField(
            description=(
                "Open cart whose quote is ready and which has a ship_to and "
                "shipping option. Its totals.amount_total_cents will be charged."
            )
        )
        payment_type: PaymentType = SchemaField(
            description=(
                "card_on_file charges the card saved on the RMFG account page; "
                "payment_method charges a Stripe PaymentMethod you manage."
            ),
            default=PaymentType.CARD_ON_FILE,
            advanced=True,
        )
        payment_method_id: str = SchemaField(
            description="Stripe PaymentMethod id (pm_...) when payment_type is payment_method.",
            default="",
            advanced=True,
        )
        customer_email: str = SchemaField(
            description="Receipt and order emails; defaults to the account email.",
            default="",
            advanced=True,
        )
        customer_phone: str = SchemaField(
            description="Contact number for the order.", default="", advanced=True
        )
        idempotency_key: str = SchemaField(
            description=(
                "Stable key so a retry never charges twice; defaults to the "
                "node execution ID. Use a new key only for a different purchase."
            ),
            default="",
            advanced=True,
        )

    class Output(RMFGCartOutput):
        payment_status: PaymentStatus = SchemaField(
            description="paid, processing (check the cart again later), failed or refunded"
        )
        checked_out: bool = SchemaField(description="True once the order exists")

    def __init__(self):
        super().__init__(
            id="4a72f115-17a1-4843-899f-fdbeb480903e",
            description="Pays an RMFG cart with the saved card and places a real production order",
            categories=CATEGORIES,
            input_schema=RMFGPayCartBlock.Input,
            output_schema=RMFGPayCartBlock.Output,
            is_sensitive_action=True,
            test_input={"cart_id": TEST_CART.id, "credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("cart", TEST_PAID_CART),
                ("cart_id", TEST_PAID_CART.id),
                ("cart_url", TEST_PAID_CART.cart_url),
                ("status", CartStatus.CHECKED_OUT),
                ("quote_status", QuoteStatus.READY),
                ("is_payable", False),
                ("totals", TEST_PAID_CART.totals),
                ("amount_total_cents", 28849),
                ("shipping_options", [TEST_SHIPPING_OPTION]),
                ("requirements", []),
                ("manufacturing_warnings", []),
                ("order_id", "ord_001"),
                ("payment_status", PaymentStatus.PAID),
                ("checked_out", True),
            ],
            test_mock={"pay_cart": lambda *args, **kwargs: TEST_PAID_CART},
        )

    @staticmethod
    async def pay_cart(
        credentials: APIKeyCredentials, input_data: Input, idempotency_key: str
    ) -> Cart:
        return await RMFGClient(credentials).pay_cart(
            input_data.cart_id,
            payment_type=input_data.payment_type,
            payment_method_id=input_data.payment_method_id,
            customer_email=input_data.customer_email,
            customer_phone=input_data.customer_phone,
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
        if (
            input_data.payment_type == PaymentType.PAYMENT_METHOD
            and not input_data.payment_method_id
        ):
            raise ValueError("payment_method_id is required for payment_method")
        cart = await self.pay_cart(
            credentials, input_data, input_data.idempotency_key or node_exec_id
        )
        async for output in emit_cart(cart):
            yield output
        payment_status = (
            cart.payment.status if cart.payment else PaymentStatus.PROCESSING
        )
        yield "payment_status", payment_status
        yield "checked_out", cart.status == CartStatus.CHECKED_OUT
