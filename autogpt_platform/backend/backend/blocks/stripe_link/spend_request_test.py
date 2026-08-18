"""Deployment gating for the Stripe Link spend-request blocks.

The virtual-card flow is self-hosted only: block outputs are persisted with
the execution and surface into AutoPilot transcripts, so a PAN there is
cardholder data at rest and a stored CVC is prohibited outright. The blocks
are split along that line — card create/retrieve are gated, while the pieces
the Shared Payment Token flow needs stay available everywhere.
"""

import pytest

from backend.blocks.stripe_link import spend_request as sr
from backend.blocks.stripe_link._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT

CARD_BLOCKS = (
    sr.StripeLinkCreateCardSpendRequestBlock,
    sr.StripeLinkRetrieveCardBlock,
)
ALWAYS_AVAILABLE_BLOCKS = (
    sr.StripeLinkListPaymentMethodsBlock,
    sr.StripeLinkGetSpendRequestStatusBlock,
)


def test_card_blocks_are_unavailable_on_cloud(monkeypatch):
    monkeypatch.setattr(sr, "CARD_FLOW_DISABLED", True)

    for block_cls in CARD_BLOCKS:
        assert block_cls().disabled is True, block_cls.__name__


def test_card_blocks_are_available_when_self_hosted(monkeypatch):
    monkeypatch.setattr(sr, "CARD_FLOW_DISABLED", False)

    for block_cls in CARD_BLOCKS:
        assert block_cls().disabled is False, block_cls.__name__


def test_the_spt_flow_survives_on_cloud(monkeypatch):
    """Gating the card flow must not take the token flow with it.

    Creating a spend request is pointless unless something can wait for the
    user to approve it, so the status poll has to remain available even where
    no card may be issued.
    """
    monkeypatch.setattr(sr, "CARD_FLOW_DISABLED", True)

    for block_cls in ALWAYS_AVAILABLE_BLOCKS:
        assert block_cls().disabled is False, block_cls.__name__


@pytest.mark.asyncio
async def test_the_status_block_cannot_pull_card_data():
    """It is the one spend-request block reachable on Cloud, so it must have
    no way to ask Link for a card in the first place."""
    seen: dict = {}
    block = sr.StripeLinkGetSpendRequestStatusBlock()

    async def _fake(credentials, method, path, body=None):
        seen["path"] = path
        return {"status": "approved", "card": {"number": "4242424242424242"}}

    object.__setattr__(block, "_link_api_request", _fake)
    inp = block.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "spend_request_id": "lsrq_test"}
    )

    outputs = {n: v async for n, v in block.run(inp, credentials=TEST_CREDENTIALS)}

    assert "include" not in seen["path"]
    assert outputs["status"] == "approved"
    # Even when Link volunteers a card, this block has no output for it.
    assert "card_number" not in outputs


@pytest.mark.asyncio
async def test_the_card_block_asks_for_the_card_without_an_opt_in():
    """Availability is the control now, so there is no include_card flag to
    forget to set."""
    seen: dict = {}
    block = sr.StripeLinkRetrieveCardBlock()

    async def _fake(credentials, method, path, body=None):
        seen["path"] = path
        return {
            "status": "approved",
            "card": {"number": "4242424242424242", "cvc": "123"},
        }

    object.__setattr__(block, "_link_api_request", _fake)
    inp = block.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "spend_request_id": "lsrq_test"}
    )

    outputs = {n: v async for n, v in block.run(inp, credentials=TEST_CREDENTIALS)}

    assert "include=card" in seen["path"]
    assert outputs["card_number"] == "4242424242424242"
    assert outputs["card_cvc"] == "123"
