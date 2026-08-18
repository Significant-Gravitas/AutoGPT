"""
Stripe Link — Spend Request blocks.

These blocks interact with the Link API (api.link.com) to create, retrieve,
and approve spend requests. A spend request provisions a one-time-use virtual
card or shared payment token from the user's Link wallet.
"""

import logging
from typing import Any

import httpx

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.stripe_link._auth import (
    LINK_API_BASE_URL,
    LINK_HTTP_TIMEOUT,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    StripeLinkCredentials,
    StripeLinkCredentialsField,
    StripeLinkCredentialsInput,
)
from backend.data.model import SchemaField
from backend.util.settings import BehaveAs, Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Raw virtual-card numbers cannot be handed out on the hosted platform: block
# outputs are persisted with the execution and surface into AutoPilot
# transcripts, so a PAN there is cardholder data at rest and a stored CVC is
# prohibited outright. Rather than returning a crippled card block, the card
# flow is absent from Cloud entirely — you cannot create a spend request you
# would have no way to redeem. Self-hosted installs get the full flow: the
# operator is the cardholder, holding their own card in their own database.
#
# The Shared Payment Token flow is unaffected and runs on both. An SPT is a
# token rather than a PAN, and the MPP blocks consume it in-process without
# ever emitting it.
CARD_FLOW_DISABLED = settings.config.behave_as == BehaveAs.CLOUD


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
CREDENTIAL_TYPE_CARD = "card"
CREDENTIAL_TYPE_SPT = "shared_payment_token"


def _nested_dict(source: Any, *keys: str) -> dict[str, Any]:
    """Walk nested dict keys, yielding {} at the first non-dict.

    `.get(key, {})` only defaults when the key is *missing*; Link sends
    explicit nulls, and a `None` in the middle of a chain raises
    AttributeError. Retrieve had already yielded `status` by that point, so
    the block emitted a partial result and then errored.
    """
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


async def link_api_request(
    credentials: StripeLinkCredentials,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Make an authenticated request to the Link API.

    Uses the access_token from OAuth2Credentials as a Bearer token.

    Refresh is deliberately not handled here: `IntegrationCredentialsManager`
    already refreshes on acquire (`_refresh_locked`), under a per-credential
    lock, and persists the rotated tokens. Refreshing inside the block would
    bypass both and let concurrent nodes stampede the token endpoint.
    """
    headers = {
        "Authorization": f"Bearer {credentials.access_token.get_secret_value()}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=LINK_HTTP_TIMEOUT) as client:
        response = await client.request(
            method=method,
            url=f"{LINK_API_BASE_URL}{path}",
            headers=headers,
            json=body,
        )
        if response.is_error:
            # Link explains itself in the body, so surface that rather than a
            # bare "400 Bad Request" with the explanation discarded.
            try:
                detail = response.json().get("error", {}).get("message")
            # ValueError: not JSON. AttributeError/TypeError: JSON, but not the
            # object shape we index into. Anything else is our bug, and masking
            # it as "API text" would hide it.
            except (ValueError, AttributeError, TypeError):
                detail = None
            raise RuntimeError(
                f"Link API {response.status_code} on {method} {path}: "
                f"{detail or response.text[:200]}"
            )
        return response.json()


# ---------------------------------------------------------------------------
# Block: List Payment Methods
# ---------------------------------------------------------------------------
class StripeLinkListPaymentMethodsBlock(Block):
    """List payment methods (cards and bank accounts) from the user's Link wallet."""

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()

    class Output(BlockSchemaOutput):
        payment_methods: list[dict[str, Any]] = SchemaField(
            description="List of payment methods in the Link wallet"
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="6eacc954-2218-4dc7-a485-5bf21549ecbe",
            description="List payment methods from a Stripe Link wallet",
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "payment_methods",
                    [
                        {
                            "id": "csmrpd_test",
                            "type": "CARD",
                            "name": "Test Debit Card",
                            "is_default": False,
                            "card_details": {
                                "brand": "visa",
                                "last4": "4242",
                                "exp_month": 12,
                                "exp_year": 2030,
                            },
                        }
                    ],
                )
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "payment_details": [
                        {
                            "id": "csmrpd_test",
                            "type": "CARD",
                            "name": "Test Debit Card",
                            "is_default": False,
                            "card_details": {
                                "brand": "visa",
                                "last4": "4242",
                                "exp_month": 12,
                                "exp_year": 2030,
                            },
                        }
                    ]
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            # `/payment-details`, not `/payment_methods` — the latter 404s. The
            # list is nested under `payment_details`, matching @stripe/link-cli's
            # SDK (packages/sdk/src/resources/payment-methods.ts).
            response = await self._link_api_request(
                credentials, "GET", "/payment-details"
            )
            yield "payment_methods", response.get("payment_details", [])
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Presentation fields shared by both create blocks
# ---------------------------------------------------------------------------
def _presentation_body(input_data: Any) -> dict[str, Any]:
    """Approval-sheet fields common to both spend-request types.

    Each is omitted when empty rather than sent as [] / {}: these drive how the
    request is presented to the user, and an explicit empty is not the same as
    "unspecified".
    """
    body: dict[str, Any] = {}
    if input_data.line_items:
        body["line_items"] = input_data.line_items
    if input_data.totals:
        body["totals"] = input_data.totals
    if input_data.metadata:
        body["metadata"] = input_data.metadata
    return body


# ---------------------------------------------------------------------------
# Block: Create Card Spend Request
# ---------------------------------------------------------------------------
class StripeLinkCreateCardSpendRequestBlock(Block):
    """
    Create a spend request for a one-time-use virtual card.

    The user approves the request in the Link app, after which
    `StripeLinkRetrieveCardBlock` returns the card details. Poll for that
    approval with `StripeLinkGetSpendRequestStatusBlock`.

    Self-hosted only — see CARD_FLOW_DISABLED.
    """

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()
        payment_method_id: str = SchemaField(
            description="ID of the payment method to use (from list payment methods)"
        )
        merchant_name: str = SchemaField(
            description="Name of the merchant, shown on the approval sheet."
        )
        merchant_url: str = SchemaField(description="URL of the merchant website")
        context: str = SchemaField(
            description=(
                "Description of the purchase context (min 100 characters). "
                "Shown to the user when they approve the request."
            ),
            # Enforced, not just documented: this is the text the user reads
            # when deciding whether to approve a charge.
            min_length=100,
        )
        amount: int = SchemaField(
            description="Amount in cents (max 50000)", ge=1, le=50000
        )
        currency: str = SchemaField(
            description="3-letter ISO currency code", default="usd"
        )
        request_approval: bool = SchemaField(
            description=(
                "If true, immediately sends a push notification to the user "
                "for approval. Otherwise, call request-approval separately."
            ),
            default=True,
        )
        test_mode: bool = SchemaField(
            description="Use test mode (fake card 4242424242424242)",
            default=False,
        )
        line_items: list[dict[str, Any]] = SchemaField(
            description=(
                "Itemised breakdown shown to the user on the approval sheet. "
                "Each item takes `name` (required) plus optional `quantity`, "
                "`unit_amount`, `description`, `sku`, `url`, `image_url` and "
                "`product_url`."
            ),
            default_factory=list,
            advanced=True,
        )
        totals: list[dict[str, Any]] = SchemaField(
            description=(
                "Total lines shown on the approval sheet. Each takes `type`, "
                "`display_text` and `amount`. `type` is one of: subtotal, tax, "
                "total, items_base_amount, items_discount, discount, "
                "fulfillment, shipping, fee, gift_wrap, tip, store_credit."
            ),
            default_factory=list,
            advanced=True,
        )
        metadata: dict[str, str] = SchemaField(
            description=(
                "Arbitrary key/value data stored on the spend request. Max 50 "
                "keys; keys <= 40 chars, values <= 500 chars."
            ),
            default_factory=dict,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        spend_request_id: str = SchemaField(
            description="ID of the created spend request"
        )
        status: str = SchemaField(
            description="Status: created, pending_approval, approved, denied, etc."
        )
        approval_url: str = SchemaField(
            description="URL the user can visit to approve (if not using push)",
            default="",
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="932c3c12-1e80-4392-8fb3-37824eb8a427",
            description=(
                "Create a Stripe Link spend request for a one-time virtual "
                "card. Self-hosted only; on AutoGPT Cloud use Create Token "
                "Spend Request with the MPP blocks instead."
            ),
            categories={BlockCategory.DATA},
            disabled=CARD_FLOW_DISABLED,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "payment_method_id": "csmrpd_test",
                "merchant_name": "Test Store",
                "merchant_url": "https://example.com",
                "context": "x" * 100,
                "amount": 1000,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("spend_request_id", "lsrq_test123"),
                ("status", "pending_approval"),
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "id": "lsrq_test123",
                    "status": "pending_approval",
                    "approval_url": "",
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            result = await self._link_api_request(
                credentials,
                "POST",
                "/spend_requests",
                body={
                    "payment_details": input_data.payment_method_id,
                    "merchant_name": input_data.merchant_name,
                    "merchant_url": input_data.merchant_url,
                    "context": input_data.context,
                    "amount": input_data.amount,
                    "currency": input_data.currency,
                    "request_approval": input_data.request_approval,
                    "test": input_data.test_mode,
                    **_presentation_body(input_data),
                },
            )
            # Note: do NOT also call POST /spend_requests/{id}/request_approval
            # here. `request_approval` in the body already moves the request to
            # pending_approval, and the dedicated endpoint then 409s. That
            # endpoint is for requesting approval on a request created without
            # it — verified against the live API.
            yield "spend_request_id", result["id"]
            yield "status", result["status"]
            if result.get("approval_url"):
                yield "approval_url", result["approval_url"]
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Block: Create Token Spend Request
# ---------------------------------------------------------------------------
class StripeLinkCreateTokenSpendRequestBlock(Block):
    """
    Create a spend request for a Shared Payment Token.

    The SPT authorises a charge at a merchant speaking the Machine Payments
    Protocol (HTTP 402). The merchant is identified by `network_id` from its
    402 challenge rather than by name and URL — Link rejects merchant_name and
    merchant_url outright for this credential type.

    Available on every deployment: an SPT is a token rather than a card
    number, and `StripeLinkMPPPayBlock` consumes it in-process without ever
    emitting it.
    """

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()
        payment_method_id: str = SchemaField(
            description="ID of the payment method to use (from list payment methods)"
        )
        network_id: str = SchemaField(
            description=(
                "Merchant network ID, read from the merchant's HTTP 402 "
                "`WWW-Authenticate: Payment` challenge — see the Get Payment "
                "Challenge block. This identifies the merchant in place of "
                "merchant_name/merchant_url."
            ),
            min_length=1,
        )
        context: str = SchemaField(
            description=(
                "Description of the purchase context (min 100 characters). "
                "Shown to the user when they approve the request."
            ),
            # Enforced, not just documented: this is the text the user reads
            # when deciding whether to approve a charge.
            min_length=100,
        )
        amount: int = SchemaField(
            description="Amount in cents (max 50000)", ge=1, le=50000
        )
        currency: str = SchemaField(
            description="3-letter ISO currency code", default="usd"
        )
        request_approval: bool = SchemaField(
            description=(
                "If true, immediately sends a push notification to the user "
                "for approval. Otherwise, call request-approval separately."
            ),
            default=True,
        )
        test_mode: bool = SchemaField(
            description="Use test mode (fake card 4242424242424242)",
            default=False,
        )
        line_items: list[dict[str, Any]] = SchemaField(
            description=(
                "Itemised breakdown shown to the user on the approval sheet. "
                "Each item takes `name` (required) plus optional `quantity`, "
                "`unit_amount`, `description`, `sku`, `url`, `image_url` and "
                "`product_url`."
            ),
            default_factory=list,
            advanced=True,
        )
        totals: list[dict[str, Any]] = SchemaField(
            description=(
                "Total lines shown on the approval sheet. Each takes `type`, "
                "`display_text` and `amount`. `type` is one of: subtotal, tax, "
                "total, items_base_amount, items_discount, discount, "
                "fulfillment, shipping, fee, gift_wrap, tip, store_credit."
            ),
            default_factory=list,
            advanced=True,
        )
        metadata: dict[str, str] = SchemaField(
            description=(
                "Arbitrary key/value data stored on the spend request. Max 50 "
                "keys; keys <= 40 chars, values <= 500 chars."
            ),
            default_factory=dict,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        spend_request_id: str = SchemaField(
            description="ID of the created spend request"
        )
        status: str = SchemaField(
            description="Status: created, pending_approval, approved, denied, etc."
        )
        approval_url: str = SchemaField(
            description="URL the user can visit to approve (if not using push)",
            default="",
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="b12877be-06cd-4a96-bf79-438bb4cb5517",
            description=(
                "MPP step 2 of 3: create a Stripe Link spend request for a "
                "Shared Payment Token, using the network ID from the "
                "merchant's 402 challenge. Step 3 is MPP Pay."
            ),
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "payment_method_id": "csmrpd_test",
                "network_id": "profile_test",
                "context": "x" * 100,
                "amount": 1000,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("spend_request_id", "lsrq_test123"),
                ("status", "pending_approval"),
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "id": "lsrq_test123",
                    "status": "pending_approval",
                    "approval_url": "",
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            result = await self._link_api_request(
                credentials,
                "POST",
                "/spend_requests",
                body={
                    "payment_details": input_data.payment_method_id,
                    "credential_type": CREDENTIAL_TYPE_SPT,
                    "network_id": input_data.network_id,
                    "context": input_data.context,
                    "amount": input_data.amount,
                    "currency": input_data.currency,
                    "request_approval": input_data.request_approval,
                    "test": input_data.test_mode,
                    **_presentation_body(input_data),
                },
            )
            yield "spend_request_id", result["id"]
            yield "status", result["status"]
            if result.get("approval_url"):
                yield "approval_url", result["approval_url"]
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Block: Get Spend Request Status
# ---------------------------------------------------------------------------
class StripeLinkGetSpendRequestStatusBlock(Block):
    """
    Poll a spend request until the user approves it.

    Every flow needs this step — a freshly created request is
    `pending_approval` until the user acts in the Link app, and both the card
    and MPP paths must wait for `approved` before they can spend. It carries
    no payment credential, so it runs on every deployment.

    It also reports `requires_action`, which is neither success nor failure:
    the card or account needs attention (typically 3D Secure) before approval
    can proceed.
    """

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()
        spend_request_id: str = SchemaField(
            description="ID of the spend request to check (e.g., lsrq_...)"
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(
            description=(
                "Current status: pending_approval, requires_action, approved, "
                "denied, expired. Wait for `approved` before spending."
            )
        )
        next_action_type: str = SchemaField(
            description=(
                "Set when status is `requires_action`: what the user must "
                "resolve before approval can proceed, e.g. a 3D Secure "
                "challenge. Empty otherwise."
            ),
            default="",
        )
        next_action_message: str = SchemaField(
            description="Human-readable explanation of the required action",
            default="",
        )
        next_action_url: str = SchemaField(
            description="Where the user resolves the required action",
            default="",
        )
        auto_resumes: bool = SchemaField(
            description=(
                "True when the request clears itself once the action is done "
                "(3D Secure), so keep polling. False means it needs a fresh "
                "spend request."
            ),
            default=False,
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="b5ef8aa2-f0bc-424f-b613-c1961ec0028d",
            description=(
                "Check whether a Stripe Link spend request has been approved "
                "yet. Poll this after creating a request and before spending."
            ),
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "spend_request_id": "lsrq_test123",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "approved")],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {"status": "approved"}
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            result = await self._link_api_request(
                credentials,
                "GET",
                f"/spend_requests/{input_data.spend_request_id}",
            )

            status = result["status"]
            yield "status", status

            # For 3D Secure the resolution is `auto_resume` and the request
            # clears itself, so the caller should keep polling rather than give
            # up; anything else needs a fresh request.
            if status == "requires_action":
                action = _nested_dict(
                    result, "status_details", "requires_action", "next_action"
                )
                yield "next_action_type", action.get("type", "")
                yield "next_action_message", action.get("display_message", "")
                yield "next_action_url", action.get("action_url", "")
                yield "auto_resumes", action.get("resolution") == "auto_resume"
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Block: Retrieve Card
# ---------------------------------------------------------------------------
class StripeLinkRetrieveCardBlock(Block):
    """
    Return the virtual card for an approved spend request.

    Single-use, capped at the approved amount, and expiring at `valid_until`.
    There is no opt-in flag: emitting the card is the entire purpose of this
    block, so its availability is the control. Self-hosted only — see
    CARD_FLOW_DISABLED.
    """

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()
        spend_request_id: str = SchemaField(
            description="ID of an approved spend request (e.g., lsrq_...)"
        )

    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Current status of the spend request")
        card_number: str = SchemaField(
            description=(
                "Virtual card number. Single-use, capped at the approved "
                "amount, and expires at `valid_until`. Block outputs are "
                "persisted with the execution, so avoid wiring this anywhere "
                "that logs, exports, or reaches a model prompt."
            ),
            default="",
            secret=True,
        )
        card_cvc: str = SchemaField(
            description=(
                "Virtual card CVC. See the note on `card_number`: this is "
                "persisted with the execution record."
            ),
            default="",
            secret=True,
        )
        card_exp_month: int = SchemaField(
            description="Card expiry month",
            default=0,
        )
        card_exp_year: int = SchemaField(
            description="Card expiry year",
            default=0,
        )
        card_brand: str = SchemaField(
            description="Card brand (visa, mastercard, etc.)",
            default="",
        )
        valid_until: str = SchemaField(
            description="ISO timestamp when the virtual card expires",
            default="",
        )
        error: str = SchemaField(
            description="Error message if the request failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="1aff59ef-e8a2-413e-9410-4ce7e4849337",
            description=(
                "Get the one-time virtual card for an approved Stripe Link "
                "spend request. Self-hosted only; on AutoGPT Cloud use the "
                "Shared Payment Token flow with the MPP blocks instead."
            ),
            categories={BlockCategory.DATA},
            disabled=CARD_FLOW_DISABLED,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "spend_request_id": "lsrq_test123",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("status", "approved"),
                ("card_number", "4242424242424242"),
                ("card_cvc", "123"),
                ("card_exp_month", 12),
                ("card_exp_year", 2030),
                ("card_brand", "visa"),
                ("valid_until", "2025-12-31T23:59:59Z"),
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "status": "approved",
                    "card": {
                        "number": "4242424242424242",
                        "cvc": "123",
                        "exp_month": 12,
                        "exp_year": 2030,
                        "brand": "visa",
                        "valid_until": "2025-12-31T23:59:59Z",
                    },
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: StripeLinkCredentials,
        **kwargs: Any,
    ) -> BlockOutput:
        try:
            result = await self._link_api_request(
                credentials,
                "GET",
                f"/spend_requests/{input_data.spend_request_id}?include=card",
            )

            yield "status", result["status"]

            card = _nested_dict(result, CREDENTIAL_TYPE_CARD)
            if card:
                yield "card_number", card.get("number", "")
                yield "card_cvc", card.get("cvc", "")
                yield "card_exp_month", card.get("exp_month", 0)
                yield "card_exp_year", card.get("exp_year", 0)
                yield "card_brand", card.get("brand", "")
                yield "valid_until", card.get("valid_until", "")
        except Exception as e:
            yield "error", str(e)
