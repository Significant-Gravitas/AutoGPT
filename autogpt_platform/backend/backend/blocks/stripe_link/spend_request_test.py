"""Tests for the spend-request block's approval-sheet fields and 3DS handling.

The bundled test_input/test_output cover only the happy path. These cover the
presentation fields that reach the user's approval sheet, and the
`requires_action` status, which is neither success nor failure.
"""

from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from backend.blocks.stripe_link import spend_request as sr
from backend.blocks.stripe_link._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.stripe_link.spend_request import (
    StripeLinkCreateSpendRequestBlock,
    StripeLinkRetrieveSpendRequestBlock,
)

CONTEXT = "x" * 100


def capture_body(captured: dict) -> Any:
    """A `_link_api_request` stand-in that records the request body."""

    async def _fake(credentials, method, path, body=None):
        captured["method"] = method
        captured["path"] = path
        captured["body"] = body
        return {"id": "lsrq_test", "status": "pending_approval"}

    return _fake


async def run_create(**overrides) -> dict:
    captured: dict = {}
    block = StripeLinkCreateSpendRequestBlock()
    object.__setattr__(block, "_link_api_request", capture_body(captured))
    payload = {
        "credentials": TEST_CREDENTIALS_INPUT,
        "payment_method_id": "csmrpd_test",
        "merchant_name": "Test Merchant",
        "merchant_url": "https://example.com",
        "context": CONTEXT,
        "amount": 1000,
        **overrides,
    }
    async for _ in block.run(
        block.Input.model_validate(payload), credentials=TEST_CREDENTIALS
    ):
        pass
    return captured["body"]


@pytest.mark.asyncio
async def test_presentation_fields_are_omitted_when_unset():
    """An explicit empty list is not the same as 'unspecified' to Link."""
    body = await run_create()
    assert "line_items" not in body
    assert "totals" not in body
    assert "metadata" not in body


@pytest.mark.asyncio
async def test_line_items_and_totals_reach_the_request():
    items = [{"name": "Running Shoes", "unit_amount": 12000, "quantity": 1}]
    totals = [{"type": "total", "display_text": "Total", "amount": 12000}]
    body = await run_create(line_items=items, totals=totals, metadata={"order": "1"})

    assert body["line_items"] == items
    assert body["totals"] == totals
    assert body["metadata"] == {"order": "1"}


@pytest.mark.asyncio
async def test_context_below_the_minimum_is_rejected():
    """This string is what the user reads before approving a charge."""
    with pytest.raises(ValidationError):
        await run_create(context="too short")


def retrieve_returning(payload: dict) -> StripeLinkRetrieveSpendRequestBlock:
    block = StripeLinkRetrieveSpendRequestBlock()

    async def _fake(credentials, method, path, body=None):
        return payload

    object.__setattr__(block, "_link_api_request", _fake)
    return block


async def run_retrieve(payload: dict) -> dict:
    block = retrieve_returning(payload)
    inp = block.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "spend_request_id": "lsrq_test"}
    )
    return {n: v async for n, v in block.run(inp, credentials=TEST_CREDENTIALS)}


@pytest.mark.asyncio
async def test_three_d_secure_is_reported_as_resumable():
    """3DS clears itself, so the caller must poll rather than start over."""
    outputs = await run_retrieve(
        {
            "status": "requires_action",
            "status_details": {
                "requires_action": {
                    "next_action": {
                        "type": "three_d_secure",
                        "display_message": "Confirm this payment with your bank",
                        "action_url": "https://app.link.com/3ds/abc",
                        "resolution": "auto_resume",
                    }
                }
            },
        }
    )

    assert outputs["status"] == "requires_action"
    assert outputs["next_action_type"] == "three_d_secure"
    assert outputs["next_action_url"] == "https://app.link.com/3ds/abc"
    assert outputs["auto_resumes"] is True
    assert "card_number" not in outputs


@pytest.mark.asyncio
async def test_non_resumable_action_is_flagged_as_such():
    outputs = await run_retrieve(
        {
            "status": "requires_action",
            "status_details": {
                "requires_action": {
                    "next_action": {
                        "type": "update_payment_method",
                        "display_message": "This card has expired",
                        "resolution": "user_action",
                    }
                }
            },
        }
    )

    assert outputs["auto_resumes"] is False
    assert outputs["next_action_url"] == ""


@pytest.mark.asyncio
async def test_approved_request_emits_no_action_fields():
    outputs, _ = await run_retrieve_with(
        {
            "status": "approved",
            "card": {"number": "4242424242424242", "cvc": "123", "brand": "visa"},
        },
        include_card=True,
    )

    assert outputs["status"] == "approved"
    assert outputs["card_number"] == "4242424242424242"
    assert "next_action_type" not in outputs


@pytest.mark.asyncio
async def test_card_is_withheld_unless_explicitly_included():
    """The opt-in has to gate emission, not just the request.

    Card number and CVC are persisted with the execution, so a response
    carrying them when we did not ask must not be written out — otherwise
    "defaults to off" only ever described the request we sent.
    """
    outputs = await run_retrieve(
        {
            "status": "approved",
            "card": {"number": "4242424242424242", "cvc": "123"},
        }
    )

    assert outputs["status"] == "approved"
    assert "card_number" not in outputs
    assert "card_cvc" not in outputs


@pytest.mark.asyncio
async def test_spt_request_drops_merchant_fields_and_sends_network_id():
    """Link rejects merchant_name/merchant_url outright for SPT; the merchant
    is identified by network_id from the 402 challenge instead."""
    body = await run_create(
        credential_type="shared_payment_token",
        network_id="profile_abc",
    )

    assert "merchant_name" not in body
    assert "merchant_url" not in body
    assert body["credential_type"] == "shared_payment_token"
    assert body["network_id"] == "profile_abc"


@pytest.mark.asyncio
async def test_card_request_keeps_merchant_fields_and_omits_credential_type():
    """`card` is the default; sending it explicitly is noise on the wire."""
    body = await run_create()

    assert body["merchant_name"] == "Test Merchant"
    assert body["merchant_url"] == "https://example.com"
    assert "credential_type" not in body
    assert "network_id" not in body


@pytest.mark.asyncio
async def test_spt_without_a_network_id_is_rejected_before_the_request():
    """Otherwise Link gets a request with no merchant identity at all."""
    with pytest.raises(ValidationError, match="network_id is required"):
        await run_create(credential_type="shared_payment_token")


@pytest.mark.asyncio
async def test_link_error_message_is_surfaced_not_swallowed():
    """`raise_for_status()` alone reports "400 Bad Request" and discards the
    explanation, which is how the SPT merchant-field constraint stayed hidden."""

    class _Resp:
        is_error = True
        status_code = 400

        def json(self):
            return {"error": {"message": "amount below minimum"}}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, *a, **kw):
            return _Resp()

    original = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **kw: _Client()  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="amount below minimum"):
            await sr.link_api_request(TEST_CREDENTIALS, "POST", "/spend_requests")
    finally:
        httpx.AsyncClient = original  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_link_error_falls_back_when_the_body_is_not_json():
    """A proxy in front of Link can answer with HTML; don't mask it."""

    class _Resp:
        is_error = True
        status_code = 502
        text = "<html>bad gateway</html>"

        def json(self):
            raise ValueError("not json")

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, *a, **kw):
            return _Resp()

    original = httpx.AsyncClient
    httpx.AsyncClient = lambda *a, **kw: _Client()  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="502"):
            await sr.link_api_request(TEST_CREDENTIALS, "GET", "/spend_requests")
    finally:
        httpx.AsyncClient = original  # type: ignore[assignment]


async def run_retrieve_with(payload: dict, **input_overrides) -> tuple[dict, dict]:
    """Retrieve with extra inputs, also returning what reached the API."""
    seen: dict = {}
    block = StripeLinkRetrieveSpendRequestBlock()

    async def _fake(credentials, method, path, body=None):
        seen["method"] = method
        seen["path"] = path
        return payload

    object.__setattr__(block, "_link_api_request", _fake)
    inp = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "spend_request_id": "lsrq_test",
            **input_overrides,
        }
    )
    outputs = {n: v async for n, v in block.run(inp, credentials=TEST_CREDENTIALS)}
    return outputs, seen


@pytest.mark.asyncio
async def test_shared_payment_token_is_requested_and_emitted():
    """The SPT is the value that authorizes a charge — the highest-risk output.

    It is only fetched when explicitly opted into, and the emitted value must
    be the token itself, not some other field of the wrapper object.
    """
    outputs, seen = await run_retrieve_with(
        {
            "status": "approved",
            "shared_payment_token": {"id": "spt_live_abc123"},
        },
        include_shared_payment_token=True,
    )

    assert "shared_payment_token" in seen["path"]
    assert outputs["shared_payment_token"] == "spt_live_abc123"


@pytest.mark.asyncio
async def test_shared_payment_token_is_not_requested_by_default():
    outputs, seen = await run_retrieve_with(
        {"status": "approved", "shared_payment_token": {"id": "spt_live_abc123"}}
    )

    assert "shared_payment_token" not in seen["path"]
    assert "shared_payment_token" not in outputs


@pytest.mark.asyncio
async def test_a_non_object_shared_payment_token_does_not_crash_retrieve():
    """Link is documented to send an object; a bare string or null must not
    take down a retrieve that has already yielded `status`."""
    for payload_value in ("spt_live_abc123", None, []):
        outputs, _ = await run_retrieve_with(
            {"status": "approved", "shared_payment_token": payload_value},
            include_shared_payment_token=True,
        )
        assert outputs["status"] == "approved"
        assert "error" not in outputs
        assert not outputs.get("shared_payment_token")


@pytest.mark.asyncio
async def test_a_null_status_details_does_not_break_requires_action():
    """`.get(key, {})` only defaults when the key is *missing*.

    An explicit null used to raise mid-chain, after `status` had already been
    yielded — so the block emitted a partial result and then errored.
    """
    outputs = await run_retrieve({"status": "requires_action", "status_details": None})

    assert outputs["status"] == "requires_action"
    assert outputs["next_action_type"] == ""
    assert outputs["auto_resumes"] is False
    assert "error" not in outputs


@pytest.mark.asyncio
async def test_card_request_without_merchant_fields_is_rejected():
    """Mirror of the SPT network_id check: a card request with no merchant
    identity must fail before it reaches Link."""
    with pytest.raises(ValidationError, match="merchant"):
        await run_create(credential_type="card", merchant_name="", merchant_url="")
