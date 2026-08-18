"""Tests for the spend-request block's approval-sheet fields and 3DS handling.

The bundled test_input/test_output cover only the happy path. These cover the
presentation fields that reach the user's approval sheet, and the
`requires_action` status, which is neither success nor failure.
"""

from typing import Any

import pytest

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
    block._link_api_request = capture_body(captured)  # type: ignore[method-assign]
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
    with pytest.raises(Exception):
        await run_create(context="too short")


def retrieve_returning(payload: dict) -> StripeLinkRetrieveSpendRequestBlock:
    block = StripeLinkRetrieveSpendRequestBlock()

    async def _fake(credentials, method, path, body=None):
        return payload

    block._link_api_request = _fake  # type: ignore[method-assign]
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
    outputs = await run_retrieve(
        {
            "status": "approved",
            "card": {"number": "4242424242424242", "cvc": "123", "brand": "visa"},
        }
    )

    assert outputs["status"] == "approved"
    assert outputs["card_number"] == "4242424242424242"
    assert "next_action_type" not in outputs
