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
