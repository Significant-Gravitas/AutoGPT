"""Tests for the MPP challenge parsing and credential construction.

The credential format is not documented; it was derived from `Credential
.serialize` in `mppx` and confirmed against a live MPP merchant, which
accepted it and settled a real payment. These tests pin the shape so a
refactor cannot silently break it.
"""

import base64
import json

import pytest

from backend.blocks.stripe_link._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.stripe_link.mpp import (
    StripeLinkGetPaymentChallengeBlock,
    StripeLinkMPPPayBlock,
    build_credential,
    decode_payment_request,
    parse_payment_challenges,
    select_stripe_challenge,
)

# Shape of a real climate.stripe.dev response: two challenges in one header.
TWO_CHALLENGE_HEADER = (
    'Payment id="abc", realm="merchant.example", method="tempo", intent="charge", '
    'request="eyJhIjoxfQ", description="Climate contribution", '
    'expires="2026-08-18T04:05:32.228Z", '
    'Payment id="def", realm="merchant.example", method="stripe", intent="charge", '
    'request="eyJiIjoyfQ", description="Climate contribution", '
    'expires="2026-08-18T04:05:32.231Z"'
)


def test_both_challenges_are_parsed_from_one_header():
    """A merchant can offer several methods; picking the wrong one fails to pay."""
    challenges = parse_payment_challenges(TWO_CHALLENGE_HEADER)

    assert [c["method"] for c in challenges] == ["tempo", "stripe"]
    assert challenges[1]["id"] == "def"
    assert challenges[1]["realm"] == "merchant.example"


def test_a_header_without_payment_challenges_yields_nothing():
    assert parse_payment_challenges('Bearer realm="example"') == []
    assert parse_payment_challenges("") == []


def test_payment_request_decodes_without_padding():
    """The `request` blob is base64url with padding stripped."""
    payload = {
        "amount": "100",
        "currency": "usd",
        "methodDetails": {"networkId": "p_1"},
    }
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )

    assert decode_payment_request(encoded) == payload


def test_credential_round_trips_to_the_expected_wire_shape():
    challenge = {
        "id": "def",
        "realm": "merchant.example",
        "method": "stripe",
        "intent": "charge",
        "request": "eyJiIjoyfQ",
        "description": "Climate contribution",
        "expires": "2026-08-18T04:05:32.231Z",
    }
    credential = build_credential(challenge, "spt_123")

    scheme, _, encoded = credential.partition(" ")
    assert scheme == "Payment"
    assert "=" not in encoded, "mppx emits unpadded base64url"

    decoded = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    assert decoded["payload"] == {"spt": "spt_123"}
    # `request` must survive byte-for-byte: the server HMAC-binds the challenge.
    assert decoded["challenge"]["request"] == "eyJiIjoyfQ"
    assert decoded["challenge"]["id"] == "def"


def test_unknown_challenge_fields_are_not_echoed_back():
    """Only the fields mppx keeps travel back, or verification fails."""
    challenge = {
        "id": "def",
        "realm": "m",
        "method": "stripe",
        "intent": "charge",
        "request": "eyJiIjoyfQ",
        "vendor_extension": "should-be-dropped",
    }
    encoded = build_credential(challenge, "spt_1").split(" ", 1)[1]
    decoded = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))

    assert "vendor_extension" not in decoded["challenge"]


async def run_pay(spend_request: dict) -> dict:
    block = StripeLinkMPPPayBlock()

    async def _fake_link(credentials, method, path, body=None):
        return spend_request

    block._link_api_request = _fake_link  # type: ignore[method-assign]
    inp = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "spend_request_id": "lsrq_test",
            "url": "https://merchant.example/api/buy",
        }
    )
    return {n: v async for n, v in block.run(inp, credentials=TEST_CREDENTIALS)}


@pytest.mark.asyncio
async def test_paying_with_an_unapproved_request_is_refused():
    """The token does not exist yet; hitting the merchant would just fail."""
    outputs = await run_pay({"status": "pending_approval"})

    assert "not approved" in outputs["error"]
    assert "status_code" not in outputs


@pytest.mark.asyncio
async def test_paying_with_a_card_request_explains_the_mismatch():
    """A card spend request has no SPT — say so instead of failing obscurely."""
    outputs = await run_pay({"status": "approved", "card": {"number": "4242"}})

    assert "shared_payment_token" in outputs["error"]


@pytest.mark.asyncio
async def test_challenge_block_reports_non_mpp_merchants():
    block = StripeLinkGetPaymentChallengeBlock()

    async def _probe(*_a, **_kw):
        return 200, ""

    block._probe = _probe  # type: ignore[method-assign]

    inp = block.Input.model_validate({"url": "https://shop.example/checkout"})
    outputs = {n: v async for n, v in block.run(inp)}

    assert outputs["supports_mpp"] is False
    assert "network_id" not in outputs


@pytest.mark.asyncio
async def test_challenge_block_ignores_a_402_without_a_stripe_method():
    """Onchain-only merchants answer 402 but we cannot pay them."""
    block = StripeLinkGetPaymentChallengeBlock()
    onchain_only = 'Payment id="abc", realm="m", method="tempo", intent="charge", request="eyJhIjoxfQ"'

    async def _probe(*_a, **_kw):
        return 402, onchain_only

    block._probe = _probe  # type: ignore[method-assign]

    inp = block.Input.model_validate({"url": "https://shop.example/checkout"})
    outputs = {n: v async for n, v in block.run(inp)}

    assert outputs["supports_mpp"] is False


class _FakeResponse:
    """Stands in for `backend.util.request.Response`."""

    def __init__(self, status: int, payload=None, headers=None, text=""):
        self.status = status
        self._payload = payload
        self.headers = headers or {}
        self._text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def text(self):
        return self._text


class _RecordingClient:
    """Returns queued responses and records every request made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def request(self, method, url, *, json=None, headers=None):
        self.calls.append(
            {"method": method, "url": url, "json": json, "headers": headers or {}}
        )
        return self._responses.pop(0)


def _patch_requests(monkeypatch, client):
    from backend.blocks.stripe_link import mpp

    monkeypatch.setattr(mpp, "Requests", lambda *a, **kw: client)


CHALLENGE_HEADER = (
    'Payment id="ch_1", realm="merchant.example", method="stripe", '
    'intent="charge", request="eyJhIjoxfQ"'
)


@pytest.mark.asyncio
async def test_pay_attaches_the_credential_on_the_retry(monkeypatch):
    """The heart of MPP: probe unauthenticated, then retry carrying the token."""
    client = _RecordingClient(
        [
            _FakeResponse(402, headers={"www-authenticate": CHALLENGE_HEADER}),
            _FakeResponse(200, payload={"contribution_id": "pi_1"}),
        ]
    )
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    status, payload = await block._pay_with_token(
        "spt_abc", "https://merchant.example/buy", "POST", {"amount": 100}, {}
    )

    assert status == 200
    assert payload == {"contribution_id": "pi_1"}
    assert len(client.calls) == 2
    # First hop must be unauthenticated — that is what elicits the challenge.
    assert "Authorization" not in client.calls[0]["headers"]
    credential = client.calls[1]["headers"]["Authorization"]
    assert credential.startswith("Payment ")
    assert (
        "spt_abc"
        in base64.urlsafe_b64decode(credential.split(" ", 1)[1] + "==").decode()
    )


@pytest.mark.asyncio
async def test_pay_does_not_retry_when_the_merchant_did_not_ask_for_payment(
    monkeypatch,
):
    """A 200 on the first hop means nothing is owed; sending the token anyway
    would leak a bearer credential for no reason."""
    client = _RecordingClient([_FakeResponse(200, payload={"ok": True})])
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    status, payload = await block._pay_with_token(
        "spt_abc", "https://merchant.example/buy", "POST", {}, {}
    )

    assert status == 200 and payload == {"ok": True}
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_pay_raises_when_402_offers_no_stripe_method(monkeypatch):
    onchain_only = (
        'Payment id="c", realm="m", method="tempo", intent="charge", request="e30"'
    )
    client = _RecordingClient(
        [_FakeResponse(402, headers={"www-authenticate": onchain_only})]
    )
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    with pytest.raises(RuntimeError, match="without a Stripe payment challenge"):
        await block._pay_with_token(
            "spt_abc", "https://merchant.example/buy", "POST", {}, {}
        )


@pytest.mark.asyncio
async def test_caller_headers_cannot_displace_the_credential(monkeypatch):
    """A caller-supplied Authorization must not survive into the paid retry."""
    client = _RecordingClient(
        [
            _FakeResponse(402, headers={"www-authenticate": CHALLENGE_HEADER}),
            _FakeResponse(200, payload={"ok": True}),
        ]
    )
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    await block._pay_with_token(
        "spt_abc",
        "https://merchant.example/buy",
        "POST",
        {},
        {"Authorization": "Bearer attacker", "Content-Type": "text/plain"},
    )

    retry_headers = client.calls[1]["headers"]
    assert retry_headers["Authorization"].startswith("Payment ")
    assert retry_headers["Content-Type"] == "application/json"


def test_oversized_request_blob_is_refused():
    """The blob is merchant-controlled; decoding it unbounded is a DoS vector."""
    with pytest.raises(ValueError, match="implausibly large"):
        decode_payment_request("A" * (17 * 1024))


def test_non_object_request_blob_is_refused():
    """`json.loads` can return a list; downstream `.get()` would explode."""
    encoded = base64.urlsafe_b64encode(b"[1,2,3]").decode().rstrip("=")
    with pytest.raises(ValueError, match="must decode to an object"):
        decode_payment_request(encoded)


def test_challenges_parse_regardless_of_field_order():
    """RFC 7235 does not fix auth-param order; Link merely happens to send
    `id` first. Splitting on `id=` would have merged these into one."""
    header = (
        'Payment realm="m", method="stripe", intent="charge", id="x", '
        'request="eyJhIjoxfQ", '
        'Payment method="tempo", id="y", realm="m", intent="charge", request="e30"'
    )
    challenges = parse_payment_challenges(header)

    assert [c["method"] for c in challenges] == ["stripe", "tempo"]
    assert challenges[0]["id"] == "x"
    assert select_stripe_challenge(header)["id"] == "x"  # type: ignore[index]


@pytest.mark.asyncio
async def test_probe_drops_a_caller_supplied_authorization(monkeypatch):
    """The first hop must be unauthenticated — an Authorization header the
    caller passed through would suppress the 402 and break the flow."""
    client = _RecordingClient(
        [
            _FakeResponse(402, headers={"www-authenticate": CHALLENGE_HEADER}),
            _FakeResponse(200, payload={"ok": True}),
        ]
    )
    _patch_requests(monkeypatch, client)

    block = StripeLinkMPPPayBlock()
    await block._pay_with_token(
        "spt_abc",
        "https://merchant.example/buy",
        "POST",
        {},
        {"Authorization": "Bearer caller-token"},
    )

    assert "Authorization" not in client.calls[0]["headers"]
    assert client.calls[1]["headers"]["Authorization"].startswith("Payment ")
