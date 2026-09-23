"""
Stripe Link — Machine Payments Protocol (MPP) blocks.

MPP merchants answer an unauthenticated request with HTTP 402 and a
`WWW-Authenticate: Payment` challenge. The agent turns a Shared Payment Token
into a credential for that challenge and retries. No checkout form, no card
number, nothing for a human to type.

Pair these with the Create Token Spend Request block.
For ordinary merchants that need a card typed into a checkout form, use the
virtual-card flow instead — see `spend_request.py`.
"""

import base64
import json
import logging
import re
from typing import Any
from urllib.parse import quote

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
)
from backend.blocks.stripe_link.spend_request import link_api_request
from backend.data.model import SchemaField
from backend.util.request import Requests, Response

logger = logging.getLogger(__name__)


class BlockedURLError(Exception):
    """`Requests` refused the URL (SSRF / private-range guard).

    Redacted at the point of refusal rather than by inspecting exception types
    later: `Requests` signals a refusal with a bare `ValueError`, and so does
    every other parse failure in this module, so a catch-all would report a
    merchant's malformed challenge as a blocked URL.
    """


async def guarded_request(client: Requests, method: str, url: str, **kwargs: Any):
    """Make a request, converting a refusal into a message safe to surface.

    The rejection names the host or resolved IP, which belongs in the logs but
    not in an agent- and user-visible `error` output, where it turns a blocked
    request into a readout of what resolves on the internal network.
    """
    try:
        return await client.request(method, url, **kwargs)
    except ValueError as e:
        logger.warning("Blocked MPP request to %s: %s", url, e)
        raise BlockedURLError("Request blocked: the URL is not allowed.") from e


PAYMENT_SCHEME = "Payment"
# Merchant-controlled data: bound what we decode and what we echo back.
# Bounds the base64 challenge blob we decode from a 402 header.
MAX_REQUEST_BLOB_BYTES = 16 * 1024
# Bounds a merchant response body before we deserialize it. Separate from the
# challenge bound so the two can move independently.
MAX_RESPONSE_BYTES = 256 * 1024
# How much of an oversized or unparseable body we keep for diagnostics.
MAX_TRUNCATED_BODY_CHARS = 1000
# A JSON number literal — no leading zeros, so identifiers survive as strings.
JSON_NUMBER_PATTERN = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")
# Fields `mppx` keeps on the wire challenge; anything else the server sends is
# dropped rather than echoed back.
CHALLENGE_FIELDS = frozenset(
    {
        "description",
        "digest",
        "expires",
        "id",
        "intent",
        "method",
        "opaque",
        "realm",
        "request",
    }
)


def parse_payment_challenges(header: str) -> list[dict[str, str]]:
    """Split a `WWW-Authenticate` value into its `Payment` challenges.

    A merchant may offer several (Stripe and an onchain method, say), so this
    returns all of them and the caller picks.
    """
    challenges: list[dict[str, str]] = []
    for chunk in re.split(r"(?i)(?=\bPayment\s+\w+=)", header):
        chunk = chunk.strip().rstrip(",")
        if not chunk.lower().startswith(PAYMENT_SCHEME.lower()):
            continue
        fields = dict(re.findall(r'(\w+)="([^"]*)"', chunk))
        if fields:
            challenges.append(fields)
    return challenges


def decode_payment_request(encoded: str) -> dict[str, Any]:
    """Decode a challenge's base64url `request` blob.

    The blob comes from the merchant, so it is size-bounded and the result is
    type-checked — `json.loads` can return a list or a scalar just as easily.
    """
    if len(encoded) > MAX_REQUEST_BLOB_BYTES:
        raise ValueError("Payment challenge request blob is implausibly large")
    padded = encoded + "=" * (-len(encoded) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(padded))
    if not isinstance(decoded, dict):
        raise ValueError("Payment challenge request must decode to an object")
    return decoded


def build_credential(challenge: dict[str, str], spt: str) -> str:
    """Build the `Authorization: Payment ...` value for a challenge.

    Mirrors `Credential.serialize` in `mppx`: base64url (unpadded) of
    `{challenge, payload}`, with the challenge's `request` passed through
    exactly as the server sent it.
    """
    wire = {
        "challenge": {k: v for k, v in challenge.items() if k in CHALLENGE_FIELDS},
        "payload": {"spt": spt},
    }
    encoded = (
        base64.urlsafe_b64encode(json.dumps(wire, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    return f"{PAYMENT_SCHEME} {encoded}"


def select_stripe_challenge(header: str) -> dict[str, str] | None:
    """Pick the Stripe challenge from a `WWW-Authenticate` value, if offered.

    Merchants may advertise several methods (an onchain one alongside Stripe);
    only the Stripe one is payable with a Link token.
    """
    return next(
        (c for c in parse_payment_challenges(header) if c.get("method") == "stripe"),
        None,
    )


class StripeLinkGetPaymentChallengeBlock(Block):
    """Ask an MPP merchant what it wants paid, before creating a spend request."""

    class Input(BlockSchemaInput):
        url: str = SchemaField(description="The merchant endpoint to purchase from")
        method: str = SchemaField(
            description="HTTP method the purchase uses", default="POST"
        )
        body: dict[str, Any] = SchemaField(
            description="JSON body for the purchase request",
            default_factory=dict,
        )

    class Output(BlockSchemaOutput):
        supports_mpp: bool = SchemaField(
            description=(
                "True when the merchant answered 402 with a Stripe payment "
                "challenge, so a Shared Payment Token can pay it. False means "
                "it cannot be paid this way — check `payment_required` to see "
                "whether that is because nothing is owed or because the "
                "merchant only accepts a method this block cannot provide. A "
                "probe that got no answer at all raises instead, so it is "
                "never reported as False."
            )
        )
        payment_required: bool = SchemaField(
            description=(
                "True when the merchant demanded payment (HTTP 402) but not "
                "via Stripe — an onchain-only merchant, say. With "
                "`supports_mpp` false this distinguishes 'pays another way, "
                "unreachable from here' from 'served without charging', where "
                "the virtual-card flow is the sensible fallback."
            ),
            default=False,
        )
        network_id: str = SchemaField(
            description="Merchant network ID — pass this to Create Spend "
            "Request as `network_id`",
            default="",
        )
        amount: int = SchemaField(
            description="Amount the merchant wants, in the smallest currency unit",
            default=0,
        )
        currency: str = SchemaField(
            description="Three-letter currency code", default=""
        )
        description: str = SchemaField(
            description="What the merchant says the charge is for", default=""
        )
        error: str = SchemaField(description="Error message on failure", default="")

    def __init__(self):
        super().__init__(
            id="0518625e-5c08-40d4-b1fe-7953250cbe80",
            description=(
                "MPP step 1 of 3: read a merchant's HTTP 402 payment challenge "
                "to learn its network ID and amount. Step 2 is Create Spend "
                "Request with credential type 'shared_payment_token' and that "
                "network ID; step 3 is MPP Pay. Returns supports_mpp=false for "
                "ordinary merchants — use the virtual-card flow for those."
            ),
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"url": "https://merchant.example/api/buy"},
            test_output=[
                ("supports_mpp", True),
                ("network_id", "profile_test"),
                ("amount", 100),
                ("currency", "usd"),
                ("description", "Test charge"),
            ],
            test_mock={
                "_probe": lambda *args, **kwargs: (
                    402,
                    'Payment id="ch_1", realm="merchant.example", method="stripe", '
                    'intent="charge", request="'
                    + base64.urlsafe_b64encode(
                        json.dumps(
                            {
                                "amount": "100",
                                "currency": "usd",
                                "methodDetails": {"networkId": "profile_test"},
                            }
                        ).encode()
                    )
                    .decode()
                    .rstrip("=")
                    + '", description="Test charge"',
                )
            },
        )

    # Instance method to match `_pay_with_token`, so both API seams patch
    # the same way in tests.
    async def _probe(
        self, url: str, method: str, body: dict[str, Any]
    ) -> tuple[int, str]:
        """Make the unauthenticated request and return (status, challenge header).

        Goes through `Requests`, which validates the URL before connecting. The
        URL comes from the agent, so a raw client here would be an SSRF hole.
        """
        response = await guarded_request(
            Requests(raise_for_status=False, retry_max_attempts=1),
            method,
            url,
            json=body or None,
            headers={"Content-Type": "application/json"},
            allow_redirects=False,
        )
        return response.status, response.headers.get("www-authenticate", "")

    async def run(self, input_data: Input, **kwargs: Any) -> BlockOutput:
        try:
            status, header = await self._probe(
                input_data.url, input_data.method, input_data.body
            )
            # "Not payable from here" and "the probe got no answer" are
            # different outcomes: only the first is a statement about the
            # merchant. Collapsing them would route an agent into the
            # virtual-card flow on a transient 503 as though the merchant had
            # been checked.
            if status == 402 and not header:
                # Wants payment, but offers no challenge we can read.
                yield "supports_mpp", False
                yield "payment_required", True
                return
            if status != 402:
                if 200 <= status < 300:
                    # Served without demanding payment — a definitive answer.
                    yield "supports_mpp", False
                    return
                raise RuntimeError(
                    f"Payment challenge probe failed with HTTP {status}; "
                    "cannot tell whether this merchant supports MPP"
                )

            stripe_challenge = select_stripe_challenge(header)
            if not stripe_challenge:
                # 402 with challenges, but none we can pay — onchain-only
                # merchants are the common case. The virtual-card flow will
                # not help here either, so say so rather than implying it.
                yield "supports_mpp", False
                yield "payment_required", True
                return

            request = decode_payment_request(stripe_challenge.get("request", ""))
            yield "supports_mpp", True
            yield "network_id", request.get("methodDetails", {}).get("networkId", "")
            yield "amount", int(request.get("amount", 0))
            yield "currency", request.get("currency", "")
            yield "description", stripe_challenge.get("description", "")
        except Exception as e:
            yield "error", str(e)


class StripeLinkMPPPayBlock(Block):
    """Complete a purchase at an MPP merchant using an approved spend request."""

    _link_api_request = staticmethod(link_api_request)

    class Input(BlockSchemaInput):
        credentials: StripeLinkCredentialsInput = StripeLinkCredentialsField()
        spend_request_id: str = SchemaField(
            description="An approved spend request created with credential "
            "type 'shared_payment_token'"
        )
        url: str = SchemaField(description="The merchant endpoint to purchase from")
        method: str = SchemaField(description="HTTP method", default="POST")
        body: dict[str, Any] = SchemaField(
            description="JSON body for the purchase request", default_factory=dict
        )
        headers: dict[str, str] = SchemaField(
            description="Extra headers to send to the merchant",
            default_factory=dict,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        status_code: int = SchemaField(description="HTTP status the merchant returned")
        paid: bool = SchemaField(
            description=(
                "True when the merchant accepted the credential-bearing payment "
                "request (2xx)"
            )
        )
        response: dict[str, Any] = SchemaField(
            description="Merchant's JSON response, e.g. an order or receipt",
            default_factory=dict,
        )
        error: str = SchemaField(description="Error message on failure", default="")

    def __init__(self):
        super().__init__(
            id="b219415c-8b36-4f49-937b-f11b1bbddfb2",
            description=(
                "MPP step 3 of 3: spend an approved Shared Payment Token at "
                "the merchant's endpoint. Follows Get Payment Challenge (step "
                "1) and Create Token Spend Request (step 2). No card number and no "
                "checkout form. The token is single-use, so a failed payment "
                "needs a fresh spend request."
            ),
            categories={BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "spend_request_id": "lsrq_test",
                "url": "https://merchant.example/api/buy",
                "body": {"amount": 100},
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("status_code", 200),
                ("paid", True),
                ("response", {"ok": True}),
            ],
            test_mock={
                "_link_api_request": lambda *args, **kwargs: {
                    "status": "approved",
                    "shared_payment_token": {"id": "spt_test"},
                },
                "_pay_with_token": lambda *args, **kwargs: (200, {"ok": True}, True),
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
            spend_request = await self._link_api_request(
                credentials,
                "GET",
                # No `?include=` here. Link returns `shared_payment_token` on
                # the plain representation, and naming it as an include
                # *suppresses* the field instead of expanding it — the request
                # succeeds and the field is simply absent. `card` is a valid
                # include; this is not.
                f"/spend_requests/{quote(input_data.spend_request_id, safe='')}",
            )
            if spend_request.get("status") != "approved":
                yield "error", (
                    f"Spend request {input_data.spend_request_id} is "
                    f"{spend_request.get('status')}, not approved"
                )
                return

            spt = (spend_request.get("shared_payment_token") or {}).get("id")
            if not spt:
                yield "error", (
                    "Spend request has no shared payment token — it was likely "
                    "created for a virtual card. Use credential type "
                    "'shared_payment_token' to pay an MPP merchant."
                )
                return

            status_code, payload, payment_attempted = await self._pay_with_token(
                spt,
                input_data.url,
                input_data.method,
                _coerce_numeric_strings(input_data.body),
                input_data.headers,
            )
            yield "status_code", status_code
            yield "paid", payment_attempted and 200 <= status_code < 300
            yield "response", payload
        except Exception as e:
            yield "error", str(e)

    async def _pay_with_token(
        self,
        spt: str,
        url: str,
        method: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[int, dict[str, Any], bool]:
        """Probe for a challenge, then retry the request carrying the token.

        Every hop goes through `Requests`. The URL is agent-supplied and the
        retry carries the SPT — a bearer credential that can authorize a charge
        — so an unvalidated client could be steered into handing a payment
        token to an arbitrary or internal host.
        """
        client = Requests(raise_for_status=False, retry_max_attempts=1)
        # Caller headers first, so they cannot displace Content-Type or, more
        # importantly, the Authorization we are about to attach.
        base_headers = {**headers, "Content-Type": "application/json"}
        # Strip any caller Authorization case-insensitively. Re-adding ours
        # under the canonical spelling would otherwise leave a lowercase
        # `authorization` in place and send the merchant two of them.
        base_headers = {
            k: v for k, v in base_headers.items() if k.lower() != "authorization"
        }
        # The probe must be unauthenticated — that is what elicits the 402
        # challenge.
        first = await guarded_request(
            client,
            method,
            url,
            json=body or None,
            headers=base_headers,
            allow_redirects=False,
        )
        if first.status != 402:
            # Nothing to pay — it either succeeded outright or failed for an
            # unrelated reason. Report what happened either way.
            return first.status, _json_or_text(first), False

        challenge = select_stripe_challenge(first.headers.get("www-authenticate", ""))
        if challenge is None:
            raise RuntimeError(
                "Merchant returned 402 without a Stripe payment challenge"
            )

        retry = await guarded_request(
            client,
            method,
            url,
            json=body or None,
            headers={**base_headers, "Authorization": build_credential(challenge, spt)},
            allow_redirects=False,
        )
        return retry.status, _json_or_text(retry), True


def _coerce_numeric_strings(body: dict[str, Any]) -> dict[str, Any]:
    """Restore numbers the builder's key/value editor turned into strings.

    Only valid JSON number literals convert, so `"0012345"` stays a string.
    """
    return {
        key: _as_json_number(value) if isinstance(value, str) else value
        for key, value in body.items()
    }


def _as_json_number(value: str) -> int | float | str:
    if not JSON_NUMBER_PATTERN.match(value):
        return value
    return float(value) if {".", "e", "E"} & set(value) else int(value)


def _json_or_text(response: Response) -> dict[str, Any]:
    """Normalize a merchant response into a dict output, bounded in size.

    The size check happens *before* parsing. Deserializing JSON amplifies:
    a few hundred KB of text becomes far more as Python objects, so checking
    the parsed result would already have paid the cost this guards against.
    (The body has still been read into memory by the client — fully bounding
    that needs streaming support in `Requests`, which this cannot reach.)
    """
    try:
        raw = response.content
    except Exception:
        raw = b""

    if len(raw) > MAX_RESPONSE_BYTES:
        logger.warning(
            "Merchant response of %d bytes exceeds the %d-byte cap; truncating",
            len(raw),
            MAX_RESPONSE_BYTES,
        )
        return {
            "truncated": True,
            "body": raw[:MAX_TRUNCATED_BODY_CHARS].decode("utf-8", errors="replace"),
        }

    try:
        parsed = response.json()
    except Exception:
        return {"body": response.text()[:MAX_TRUNCATED_BODY_CHARS]}

    return parsed if isinstance(parsed, dict) else {"data": parsed}
