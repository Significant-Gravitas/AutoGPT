"""Spend-request block behaviour, and the deployment gate on the card flow.

The virtual-card flow is self-hosted only: block outputs are persisted with
the execution and surface into AutoPilot transcripts, so a PAN there is
cardholder data at rest and a stored CVC is prohibited outright. The blocks
are split along that line — card create/retrieve are gated, while everything
the Shared Payment Token flow needs stays available on every deployment.
"""

from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from backend.blocks.stripe_link import spend_request as sr
from backend.blocks.stripe_link._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT

CONTEXT = "x" * 100

CARD_BLOCKS = (
    sr.StripeLinkCreateCardSpendRequestBlock,
    sr.StripeLinkRetrieveCardBlock,
)
ALWAYS_AVAILABLE_BLOCKS = (
    sr.StripeLinkListPaymentMethodsBlock,
    sr.StripeLinkCreateTokenSpendRequestBlock,
    sr.StripeLinkGetSpendRequestStatusBlock,
)


# ---------------------------------------------------------------------------
# Deployment gating
# ---------------------------------------------------------------------------
def test_card_blocks_are_unavailable_on_cloud(monkeypatch):
    monkeypatch.setattr(sr, "CARD_FLOW_DISABLED", True)

    for block_cls in CARD_BLOCKS:
        assert block_cls().disabled is True, block_cls.__name__


def test_card_blocks_are_available_when_self_hosted(monkeypatch):
    monkeypatch.setattr(sr, "CARD_FLOW_DISABLED", False)

    for block_cls in CARD_BLOCKS:
        assert block_cls().disabled is False, block_cls.__name__


def test_the_token_flow_survives_on_cloud(monkeypatch):
    """Gating the card flow must not take the token flow with it.

    Creating a spend request is pointless unless something can wait for the
    user to approve it, so both the token create and the status poll have to
    remain available where no card may be issued.
    """
    monkeypatch.setattr(sr, "CARD_FLOW_DISABLED", True)

    for block_cls in ALWAYS_AVAILABLE_BLOCKS:
        assert block_cls().disabled is False, block_cls.__name__


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def _capturing(block, payload: dict):
    """Patch the block's API seam, recording what was sent."""
    seen: dict = {}

    async def _fake(credentials, method, path, body=None):
        seen["method"] = method
        seen["path"] = path
        seen["body"] = body
        return payload

    object.__setattr__(block, "_link_api_request", _fake)
    return seen


async def _run(block, inputs: dict, payload: dict) -> tuple[dict, dict]:
    seen = _capturing(block, payload)
    inp = block.Input.model_validate({"credentials": TEST_CREDENTIALS_INPUT, **inputs})
    outputs = {n: v async for n, v in block.run(inp, credentials=TEST_CREDENTIALS)}
    return outputs, seen


CREATED = {"id": "lsrq_test", "status": "pending_approval"}


async def run_create_card(**overrides) -> dict:
    _, seen = await _run(
        sr.StripeLinkCreateCardSpendRequestBlock(),
        {
            "payment_method_id": "csmrpd_test",
            "merchant_name": "Test Merchant",
            "merchant_url": "https://example.com",
            "context": CONTEXT,
            "amount": 1000,
            **overrides,
        },
        CREATED,
    )
    return seen["body"]


async def run_create_token(**overrides) -> dict:
    _, seen = await _run(
        sr.StripeLinkCreateTokenSpendRequestBlock(),
        {
            "payment_method_id": "csmrpd_test",
            "network_id": "profile_test",
            "context": CONTEXT,
            "amount": 1000,
            **overrides,
        },
        CREATED,
    )
    return seen["body"]


async def run_status(payload: dict) -> tuple[dict, dict]:
    return await _run(
        sr.StripeLinkGetSpendRequestStatusBlock(),
        {"spend_request_id": "lsrq_test"},
        payload,
    )


async def run_card(payload: dict) -> tuple[dict, dict]:
    return await _run(
        sr.StripeLinkRetrieveCardBlock(),
        {"spend_request_id": "lsrq_test"},
        payload,
    )


# ---------------------------------------------------------------------------
# Card vs status separation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_the_status_block_cannot_pull_card_data():
    """It is the one spend-request read reachable on Cloud, so it must have no
    way to ask Link for a card in the first place."""
    outputs, seen = await run_status(
        {"status": "approved", "card": {"number": "4242424242424242"}}
    )

    assert "include" not in seen["path"]
    assert outputs["status"] == "approved"
    # Even when Link volunteers a card, this block has no output for it.
    assert "card_number" not in outputs


@pytest.mark.asyncio
async def test_the_card_block_asks_for_the_card_without_an_opt_in():
    """Availability is the control now, so there is no flag to forget."""
    outputs, seen = await run_card(
        {
            "status": "approved",
            "card": {
                "number": "4242424242424242",
                "cvc": "123",
                "exp_month": 12,
                "exp_year": 2030,
            },
        }
    )

    assert "include=card" in seen["path"]
    assert outputs["card_number"] == "4242424242424242"
    assert outputs["card_cvc"] == "123"


@pytest.mark.asyncio
async def test_a_non_object_card_does_not_crash_the_card_block():
    for payload_value in ("4242424242424242", None, []):
        outputs, _ = await run_card({"status": "approved", "card": payload_value})
        assert outputs["status"] == "approved"
        assert "error" not in outputs
        assert not outputs.get("card_number")


# ---------------------------------------------------------------------------
# Create: card vs token
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_the_card_request_identifies_the_merchant_by_name_and_url():
    body = await run_create_card()

    assert body["merchant_name"] == "Test Merchant"
    assert body["merchant_url"] == "https://example.com"
    # Card is the default credential type; sending it explicitly is noise.
    assert "credential_type" not in body
    assert "network_id" not in body


@pytest.mark.asyncio
async def test_the_token_request_identifies_the_merchant_by_network_id():
    """Link rejects merchant_name/merchant_url outright for an SPT — the
    merchant comes from the 402 challenge's network ID instead."""
    body = await run_create_token()

    assert body["credential_type"] == sr.CREDENTIAL_TYPE_SPT
    assert body["network_id"] == "profile_test"
    assert "merchant_name" not in body
    assert "merchant_url" not in body


@pytest.mark.asyncio
async def test_a_token_request_without_a_network_id_is_rejected():
    """Otherwise Link gets a request with no merchant identity at all."""
    with pytest.raises(ValidationError):
        await run_create_token(network_id="")


@pytest.mark.asyncio
async def test_a_card_request_without_a_merchant_is_rejected():
    with pytest.raises(ValidationError):
        await run_create_card(merchant_name=None)


@pytest.mark.asyncio
async def test_context_below_the_minimum_is_rejected():
    """This is the text the user reads when approving a charge."""
    with pytest.raises(ValidationError):
        await run_create_card(context="too short")


@pytest.mark.asyncio
async def test_presentation_fields_are_omitted_when_unset():
    """An explicit empty is not the same as unspecified to the approval sheet."""
    for body in (await run_create_card(), await run_create_token()):
        assert "line_items" not in body
        assert "totals" not in body
        assert "metadata" not in body


@pytest.mark.asyncio
async def test_line_items_and_totals_reach_both_request_types():
    line_items = [{"name": "Widget", "quantity": 2, "unit_amount": 500}]
    totals = [{"type": "total", "display_text": "Total", "amount": 1000}]

    for body in (
        await run_create_card(line_items=line_items, totals=totals),
        await run_create_token(line_items=line_items, totals=totals),
    ):
        assert body["line_items"] == line_items
        assert body["totals"] == totals


# ---------------------------------------------------------------------------
# 3D Secure, reported by the status block
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_three_d_secure_is_reported_as_resumable():
    """3DS clears itself, so the caller must keep polling rather than start over."""
    outputs, _ = await run_status(
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

    assert outputs["next_action_type"] == "three_d_secure"
    assert outputs["next_action_url"] == "https://app.link.com/3ds/abc"
    assert outputs["auto_resumes"] is True


@pytest.mark.asyncio
async def test_non_resumable_action_is_flagged_as_such():
    outputs, _ = await run_status(
        {
            "status": "requires_action",
            "status_details": {
                "requires_action": {
                    "next_action": {
                        "type": "update_payment_method",
                        "resolution": "new_spend_request",
                    }
                }
            },
        }
    )

    assert outputs["auto_resumes"] is False


@pytest.mark.asyncio
async def test_a_null_status_details_does_not_break_requires_action():
    """`.get(key, {})` only defaults when the key is *missing*.

    An explicit null used to raise mid-chain, after `status` had already been
    yielded — so the block emitted a partial result and then errored.
    """
    outputs, _ = await run_status({"status": "requires_action", "status_details": None})

    assert outputs["status"] == "requires_action"
    assert outputs["next_action_type"] == ""
    assert "error" not in outputs


@pytest.mark.asyncio
async def test_an_approved_request_emits_no_action_fields():
    outputs, _ = await run_status({"status": "approved"})

    assert outputs["status"] == "approved"
    assert "next_action_type" not in outputs


# ---------------------------------------------------------------------------
# Error surfacing
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, payload: Any, text: str = ""):
        self.is_error = True
        self.status_code = 400
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.mark.asyncio
async def test_link_error_message_is_surfaced_not_swallowed(monkeypatch):
    """`raise_for_status()` alone reports "400 Bad Request" and discards the
    explanation, which is how the SPT merchant-field constraint stayed hidden."""

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, **kwargs):
            return _Resp({"error": {"message": "merchant_name is not allowed"}})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _Client())

    with pytest.raises(Exception, match="merchant_name is not allowed"):
        await sr.link_api_request(TEST_CREDENTIALS, "POST", "/spend_requests", {})


@pytest.mark.asyncio
async def test_link_error_falls_back_when_the_body_is_not_json(monkeypatch):
    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, **kwargs):
            return _Resp(None, text="<html>gateway timeout</html>")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _Client())

    with pytest.raises(Exception, match="400"):
        await sr.link_api_request(TEST_CREDENTIALS, "GET", "/spend_requests/x")
