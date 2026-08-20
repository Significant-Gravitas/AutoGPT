"""
Stripe Link — Spend Request blocks.

These blocks interact with the Link API (api.link.com) to create, retrieve,
and approve spend requests. A spend request provisions a one-time-use virtual
card or shared payment token from the user's Link wallet.
"""

import logging
import re
from typing import Any
from urllib.parse import quote

from pydantic import field_validator, model_validator

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.stripe_link._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    StripeLinkCredentials,
    StripeLinkCredentialsField,
    StripeLinkCredentialsInput,
    link_api_request,
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
def card_flow_disabled() -> bool:
    """Whether this deployment withholds the virtual-card blocks.

    A function so the predicate itself is testable: the blocks read the
    constant below at class-definition time, so a test that patches the
    constant proves only that the wiring works, not that it is derived from
    the right thing.
    """
    return settings.config.behave_as == BehaveAs.CLOUD


CARD_FLOW_DISABLED = card_flow_disabled()

# Link's own id shape. Validated rather than only escaped: the status block is
# the one spend-request read reachable on Cloud, and an id like
# `lsrq_x?include=card` would make it ask Link for card material — the
# guarantee that it cannot pull card data otherwise holds only for well-formed
# input.
_SPEND_REQUEST_ID = re.compile(r"^lsrq_[A-Za-z0-9_-]+$")


def _validate_spend_request_id(value: str) -> str:
    if not _SPEND_REQUEST_ID.match(value):
        raise ValueError(
            "spend_request_id must look like 'lsrq_...' — anything else could "
            "steer the authenticated call at a different Link endpoint"
        )
    return value


# What a payment method may contribute to a block output. Listing payment
# methods is not behind CARD_FLOW_DISABLED, so it runs on Cloud — projecting
# explicitly makes "no cardholder data at rest on Cloud" a property of this
# repo rather than of whatever Link happens to add to `payment_details` next.
_PAYMENT_METHOD_FIELDS = ("id", "type", "name", "is_default")
_CARD_DETAIL_FIELDS = ("brand", "last4", "exp_month", "exp_year")


def _project_payment_method(pm: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields a graph needs to choose a payment method."""
    projected: dict[str, Any] = {k: pm[k] for k in _PAYMENT_METHOD_FIELDS if k in pm}
    details = pm.get("card_details")
    if isinstance(details, dict):
        projected["card_details"] = {
            k: details[k] for k in _CARD_DETAIL_FIELDS if k in details
        }
    return projected


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
# Only the token path sends a `credential_type`; the card path omits it and
# lets Link default. There is deliberately no CREDENTIAL_TYPE_CARD constant —
# it read as though the card request sends one, which invites someone to start
# sending a field Link does not expect.
CREDENTIAL_TYPE_SPT = "shared_payment_token"

# The key Link returns card data under — a response envelope, not a request
# value, and free to diverge from anything sent.
CARD_RESPONSE_KEY = "card"


def _nested_dict(source: Any, *keys: str) -> dict[str, Any]:
    """Walk nested dict keys, yielding {} at the first non-dict.

    `.get(key, {})` only defaults when the key is *missing*, and Link sends
    explicit nulls — so a `None` mid-chain raises AttributeError.
    """
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


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
            description=(
                "List the cards and bank accounts in the user's Link wallet. "
                "Use this first to pick a payment method ID for Create Spend "
                "Request."
            ),
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
            yield "payment_methods", [
                _project_payment_method(pm)
                for pm in response.get("payment_details", [])
                if isinstance(pm, dict)
            ]
        except Exception as e:
            yield "error", str(e)


# ---------------------------------------------------------------------------
# Presentation fields shared by both create blocks
# ---------------------------------------------------------------------------
def _presentation_body(input_data: "_BaseSpendRequestInput") -> dict[str, Any]:
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


async def _create_spend_request(
    api_request: Any,
    credentials: StripeLinkCredentials,
    input_data: "_BaseSpendRequestInput",
    merchant_identity: dict[str, Any],
) -> dict[str, Any]:
    """POST a spend request, differing only in how the merchant is named.

    The base-schema rationale applies here too: leaving the two bodies
    copy-pasted means a field added later lands on one create block and not
    the other.
    """
    return await api_request(
        credentials,
        "POST",
        "/spend_requests",
        body={
            "payment_details": input_data.payment_method_id,
            **merchant_identity,
            "context": input_data.context,
            "amount": input_data.amount,
            "currency": input_data.currency,
            "request_approval": input_data.request_approval,
            "test": input_data.test_mode,
            **_presentation_body(input_data),
        },
    )


# ---------------------------------------------------------------------------
# Block: Create Card Spend Request
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Shared spend-request schema
# ---------------------------------------------------------------------------
class _BaseSpendRequestInput(BlockSchemaInput):
    """Fields common to both spend-request types.

    The two create blocks differ only in how the merchant is identified — name
    and URL for a card, network ID for a Shared Payment Token. Everything else
    is the same request, so it lives here rather than being copied into each,
    where a later field would inevitably be added to only one of them.
    """

    credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()
    payment_method_id: str = SchemaField(
        description="ID of the payment method to use (from list payment methods)"
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
        description=(
            "Amount in the currency's smallest unit — cents for USD, but "
            "whole units for zero-decimal currencies like JPY (max 50000)"
        ),
        ge=1,
        le=50000,
    )
    currency: str = SchemaField(description="3-letter ISO currency code", default="usd")
    request_approval: bool = SchemaField(
        description=(
            "If true, immediately sends a push notification to the user "
            "for approval. Otherwise, call request-approval separately."
        ),
        default=True,
    )
    test_mode: bool = SchemaField(
        description=(
            "Use Stripe test mode — no real money moves. A card request yields "
            "the 4242… test card; a token request yields a test token."
        ),
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

    @model_validator(mode="after")
    def _totals_must_match_the_authorized_amount(self):
        """The approval sheet is the only aggregate control in this flow.

        `line_items` and `totals` are agent-supplied and rendered to the user
        verbatim, so an untrusted upstream node could show a "total" of $4.99
        while the credential issued is for `amount`. Nothing else caps
        spending, so the headline the user reads has to be the number they are
        actually authorizing.
        """
        for entry in self.totals:
            if not isinstance(entry, dict) or entry.get("type") != "total":
                continue
            stated = entry.get("amount")
            if stated is not None and stated != self.amount:
                raise ValueError(
                    f"totals entry of type 'total' is {stated} but the spend "
                    f"request authorizes {self.amount}; the approval sheet must "
                    "show the amount actually being authorized"
                )
        return self


class _SpendRequestCreatedOutput(BlockSchemaOutput):
    """What creating a spend request returns, for either credential type."""

    spend_request_id: str = SchemaField(description="ID of the created spend request")
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

    class Input(_BaseSpendRequestInput):
        merchant_name: str = SchemaField(
            description="Name of the merchant, shown on the approval sheet."
        )
        merchant_url: str = SchemaField(description="URL of the merchant website")

    class Output(_SpendRequestCreatedOutput):
        pass

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
            result = await _create_spend_request(
                self._link_api_request,
                credentials,
                input_data,
                {
                    "merchant_name": input_data.merchant_name,
                    "merchant_url": input_data.merchant_url,
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

    class Input(_BaseSpendRequestInput):
        network_id: str = SchemaField(
            description=(
                "Merchant network ID, read from the merchant's HTTP 402 "
                "`WWW-Authenticate: Payment` challenge — see the Get Payment "
                "Challenge block. This identifies the merchant in place of "
                "merchant_name/merchant_url."
            ),
            min_length=1,
        )

        @field_validator("network_id")
        @classmethod
        def _network_id_must_not_be_blank(cls, value: str) -> str:
            """`min_length=1` accepts "   ".

            Without a real network ID Link receives a request with no merchant
            identity at all and fails obscurely, so catch it at the boundary.
            A trim guard existed before the block split removed the cross-field
            validator it lived on.
            """
            if not value.strip():
                raise ValueError(
                    "network_id is required — read it from the merchant's HTTP "
                    "402 challenge (see the Get Payment Challenge block)"
                )
            return value

    class Output(_SpendRequestCreatedOutput):
        pass

    def __init__(self):
        super().__init__(
            id="b12877be-06cd-4a96-bf79-438bb4cb5517",
            description=(
                "MPP step 2 of 3: ask the user to authorize a payment to a "
                "merchant that answers HTTP 402, and provision a Shared "
                "Payment Token for it. Takes the network ID from the Get "
                "Payment Challenge block; step 3 is MPP Pay. For an ordinary "
                "checkout form, use Create Card Spend Request instead."
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
            result = await _create_spend_request(
                self._link_api_request,
                credentials,
                input_data,
                {
                    "credential_type": CREDENTIAL_TYPE_SPT,
                    "network_id": input_data.network_id.strip(),
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

        _check_id = field_validator("spend_request_id")(_validate_spend_request_id)

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
                "yet. Poll this after creating a request and before spending, "
                "for both the card and the Shared Payment Token flows. If the "
                "status is 'requires_action' the payment method needs "
                "attention first — keep polling when `auto_resumes` is true, "
                "otherwise resolve the action and create a new request."
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
                f"/spend_requests/{quote(input_data.spend_request_id, safe='')}",
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
                # `or ""`, not `get(..., "")`: Link sends explicit nulls, and
                # a None yielded into a str-typed output fails the output
                # jsonschema check *outside* this try, hard-failing the node
                # after `status` was already emitted.
                yield "next_action_type", action.get("type") or ""
                yield "next_action_message", action.get("display_message") or ""
                yield "next_action_url", action.get("action_url") or ""
                # Unknown or missing resolution keeps the caller polling. The
                # unsafe direction is "needs a fresh request": the agent makes
                # a second spend request, the first auto-resumes, and the user
                # can end up approving two credentials for one purchase.
                yield "auto_resumes", action.get("resolution") != "new_spend_request"
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

    The card number and CVC are written to the execution record in clear text
    and are readable through the execution-results API. Nothing redacts them.
    Do not enable this block in an environment where PCI compliance matters.
    """

    # Exposed as a class attribute so `test_mock` can patch it; the harness
    # only replaces names it can find on the block instance.
    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()
        spend_request_id: str = SchemaField(
            description="ID of an approved spend request (e.g., lsrq_...)"
        )

        _check_id = field_validator("spend_request_id")(_validate_spend_request_id)

    # if you use this block note the card number will be logged. Do not use
    # this in environments where PCI compliance matters.
    #
    # Deliberately no `secret=True` on the card fields: that flag reaches only
    # `json_schema_extra`, whose sole consumer strips `input_default` on graph
    # export, so it does nothing for outputs. Marking them would advertise a
    # redaction that does not exist.
    class Output(BlockSchemaOutput):
        status: str = SchemaField(description="Current status of the spend request")
        card_number: str = SchemaField(
            description=(
                "Virtual card number. Single-use, capped at the approved "
                "amount, and expires at `valid_until`. Stored in clear text "
                "with the execution record and readable through the "
                "execution-results API — do not enable this block where PCI "
                "compliance matters."
            ),
            default="",
        )
        card_cvc: str = SchemaField(
            description=(
                "Virtual card CVC. Stored in clear text with the execution "
                "record, same as `card_number`. Retaining a CVC after "
                "authorization is prohibited under PCI DSS 3.2."
            ),
            default="",
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
                "Get the one-time virtual card number and CVC for an approved "
                "spend request, to type into a normal checkout form. Both are "
                "stored in clear text with the execution record — do not use "
                "this where PCI compliance matters. Self-hosted only; on "
                "AutoGPT Cloud use the Shared Payment Token flow with the MPP "
                "blocks instead."
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
                f"/spend_requests/{quote(input_data.spend_request_id, safe='')}?include=card",
            )

            status = result["status"]
            yield "status", status

            # Only for a spend the user actually approved. If a denied or
            # expired request ever came back with card material attached,
            # emitting it would put a PAN and CVC in the execution record for
            # a charge that was refused.
            if status != "approved":
                # Terminal and unusable. Without this the graph stalls: nothing
                # fires downstream and the agent stops mid-checkout with no
                # message about why.
                yield "error", (
                    f"Spend request is {status}, so no card was issued. "
                    "Create a new spend request if the user still wants to pay."
                )
                return

            card = _nested_dict(result, CARD_RESPONSE_KEY)
            if card:
                yield "card_number", card.get("number", "")
                yield "card_cvc", card.get("cvc", "")
                yield "card_exp_month", card.get("exp_month", 0)
                yield "card_exp_year", card.get("exp_year", 0)
                yield "card_brand", card.get("brand", "")
                yield "valid_until", card.get("valid_until", "")
        except Exception as e:
            yield "error", str(e)
